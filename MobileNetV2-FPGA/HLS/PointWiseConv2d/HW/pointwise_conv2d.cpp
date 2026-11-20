#include "pointwise_conv2d.hpp"

// ------------------------------------------------------------
// Kernel HLS optimizado para la operación PointwiseConv2D.
//
// Una convolución 1x1 equivale a un producto de matrices:
//   salida[co][p] = bias[co] + sum_ci peso[co][ci] * entrada[ci][p]
// donde p recorre las posiciones espaciales (H*W) en formato NCHW.
//
// Cambios respecto a la versión naive:
//  1) Se reordenan los bucles para que la suma acumulada de una
//     salida no dependa de la iteración inmediatamente anterior.
//     El bucle interno avanza por posiciones espaciales, y la misma
//     salida se vuelve a actualizar hasta RUN_LEN iteraciones después.
//     Esto elimina la dependencia que forzaba II = 9 y permite II = 1.
//  2) Los datos se copian a buffers locales (BRAM/URAM) mediante
//     ráfagas (bursts), en lugar de leer la DDR en cada operación.
//  3) Se calculan PARALLEL_OUT canales de salida en paralelo
//     (PARALLEL_OUT multiplicaciones-suma por ciclo).
// ------------------------------------------------------------

// ------------------------------------------------------------
// Parámetros de la arquitectura
// ------------------------------------------------------------
// Máximo de canales de entrada soportado (MobileNetV2 llega a 960).
static constexpr int MAX_IN_CHANNELS = 960;

// Posiciones espaciales procesadas por bloque (tile).
// Potencia de 2 para que el índice del buffer sea un desplazamiento.
static constexpr int TILE_PIXELS = 64;

// Canales de salida calculados en paralelo.
// Con 1 solo se aplica la reestructuración (II = 1);
// con 16 se hacen 16 multiplicaciones-suma por ciclo.
static constexpr int PARALLEL_OUT = 16;

// Longitud mínima del recorrido espacial en el bucle de cómputo.
// Garantiza que entre dos actualizaciones del mismo acumulador
// pasen suficientes ciclos para que termine la suma flotante.
// Debe ser menor o igual que TILE_PIXELS.
static constexpr int MIN_RUN = 32;

void pointwise_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
)
{

// ------------------------------------------------------------
// Interfaces AXI4 Master (m_axi)
// El parámetro depth solo lo usa la co-simulación; se ajusta al
// caso más grande del testbench para que la cosim sea rápida.
// No limita el tamaño de los tensores en hardware.
// ------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=28224
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=3456
#pragma HLS INTERFACE m_axi port=bias   offset=slave bundle=gmem2 depth=24
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3 depth=4800

// ------------------------------------------------------------
// Interfaces AXI4-Lite (s_axilite)
// ------------------------------------------------------------
#pragma HLS INTERFACE s_axilite port=input
#pragma HLS INTERFACE s_axilite port=weight
#pragma HLS INTERFACE s_axilite port=bias
#pragma HLS INTERFACE s_axilite port=output

#pragma HLS INTERFACE s_axilite port=batch_size
#pragma HLS INTERFACE s_axilite port=input_channels
#pragma HLS INTERFACE s_axilite port=input_height
#pragma HLS INTERFACE s_axilite port=input_width
#pragma HLS INTERFACE s_axilite port=output_channels
#pragma HLS INTERFACE s_axilite port=use_bias
#pragma HLS INTERFACE s_axilite port=return

    // --------------------------------------------------------
    // Buffers locales
    // --------------------------------------------------------
    // Bloque de entrada [canal][posición]. Es grande (960 x 64),
    // por eso se ubica en URAM y no consume BRAM.
    float input_buffer[MAX_IN_CHANNELS][TILE_PIXELS];
#pragma HLS BIND_STORAGE variable=input_buffer type=ram_1p impl=uram

    // Pesos del grupo de canales de salida. Se particiona por
    // canal de salida para leer PARALLEL_OUT pesos en el mismo ciclo.
    float weight_buffer[PARALLEL_OUT][MAX_IN_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=weight_buffer type=complete dim=1

    // Acumuladores [canal de salida][posición], particionados
    // igual que los pesos para actualizarlos en paralelo.
    float accumulator[PARALLEL_OUT][TILE_PIXELS];
#pragma HLS ARRAY_PARTITION variable=accumulator type=complete dim=1

    float bias_buffer[PARALLEL_OUT];
#pragma HLS ARRAY_PARTITION variable=bias_buffer type=complete

    // Fuera del rango soportado por los buffers: no se procesa.
    if (input_channels > MAX_IN_CHANNELS) {
        return;
    }

    const int pixels = input_height * input_width;

    batch_loop:
    for (int batch_index = 0; batch_index < batch_size; batch_index++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1

        const int input_base  = batch_index * input_channels  * pixels;
        const int output_base = batch_index * output_channels * pixels;

        // ----------------------------------------------------
        // Recorrido por bloques de posiciones espaciales
        // ----------------------------------------------------
        pixel_tile_loop:
        for (int tile_start = 0; tile_start < pixels; tile_start += TILE_PIXELS) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=196 avg=49

            const int tile_length =
                (pixels - tile_start < TILE_PIXELS)
                ? (pixels - tile_start) : TILE_PIXELS;

            const int run_length =
                (tile_length < MIN_RUN) ? MIN_RUN : tile_length;

            // ------------------------------------------------
            // 1) Carga del bloque de entrada: una ráfaga por canal
            // ------------------------------------------------
            load_input_channel:
            for (int in_ch = 0; in_ch < input_channels; in_ch++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=960 avg=144
                load_input_pixel:
                for (int px = 0; px < tile_length; px++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=64 avg=64
#pragma HLS PIPELINE II=1
                    input_buffer[in_ch][px] =
                        input[input_base + in_ch * pixels + tile_start + px];
                }
            }

            // ------------------------------------------------
            // Recorrido por grupos de PARALLEL_OUT canales de salida
            // ------------------------------------------------
            out_group_loop:
            for (int out_start = 0; out_start < output_channels; out_start += PARALLEL_OUT) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=80 avg=4

                const int group_size =
                    (output_channels - out_start < PARALLEL_OUT)
                    ? (output_channels - out_start) : PARALLEL_OUT;

                // --------------------------------------------
                // 2) Carga de pesos del grupo. Las filas del grupo
                //    son contiguas en memoria: una sola ráfaga.
                // --------------------------------------------
                int load_out = 0;
                int load_in  = 0;
                load_weight:
                for (int i = 0; i < group_size * input_channels; i++) {
#pragma HLS LOOP_TRIPCOUNT min=24 max=15360 avg=2304
#pragma HLS PIPELINE II=1
                    weight_buffer[load_out][load_in] =
                        weight[out_start * input_channels + i];

                    if (load_in == input_channels - 1) {
                        load_in = 0;
                        load_out++;
                    } else {
                        load_in++;
                    }
                }

                load_bias:
                for (int out = 0; out < PARALLEL_OUT; out++) {
#pragma HLS PIPELINE II=1
                    bias_buffer[out] =
                        (use_bias && out < group_size)
                        ? bias[out_start + out] : 0.0f;
                }

                // --------------------------------------------
                // 3) Cómputo: un solo bucle (canal de entrada x
                //    posición) con PARALLEL_OUT MACs por ciclo.
                //    El mismo acumulador se actualiza cada
                //    run_length iteraciones (>= MIN_RUN), por eso
                //    se declara que no hay dependencia entre
                //    iteraciones consecutivas.
                // --------------------------------------------
                int px = 0;
                int in_ch = 0;
                compute_loop:
                for (int i = 0; i < input_channels * run_length; i++) {
#pragma HLS LOOP_TRIPCOUNT min=96 max=61440 avg=9216
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accumulator type=inter false

                    const float input_value =
                        (px < tile_length) ? input_buffer[in_ch][px] : 0.0f;

                    parallel_out:
                    for (int out = 0; out < PARALLEL_OUT; out++) {
#pragma HLS UNROLL
                        const float previous =
                            (in_ch == 0) ? bias_buffer[out] : accumulator[out][px];

                        accumulator[out][px] =
                            previous + weight_buffer[out][in_ch] * input_value;
                    }

                    if (px == run_length - 1) {
                        px = 0;
                        in_ch++;
                    } else {
                        px++;
                    }
                }

                // --------------------------------------------
                // 4) Escritura de resultados: una ráfaga por
                //    canal de salida del grupo.
                // --------------------------------------------
                store_channel:
                for (int out = 0; out < group_size; out++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=16 avg=16
                    store_pixel:
                    for (int q = 0; q < tile_length; q++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=64 avg=64
#pragma HLS PIPELINE II=1
                        output[output_base + (out_start + out) * pixels + tile_start + q] =
                            accumulator[out][q];
                    }
                }
            }
        }
    }
}
