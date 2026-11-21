#include "pointwise_conv2d.hpp"

// PointwiseConv2D FP32, formato NCHW.
// La interfaz conserva dimensiones dinamicas: los tamanos siguientes
// describen la microarquitectura, NO limites de input_channels/output_channels.
static constexpr int CHANNEL_TILE = 32;  // canales de entrada por bloque
static constexpr int TILE_PIXELS  = 64;  // posiciones espaciales por bloque
static constexpr int PARALLEL_OUT = 16;  // canales de salida simultaneos
static constexpr int MIN_RUN      = 32;  // distancia entre actualizaciones del mismo acumulador

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
) {
    // Se mantienen los puertos y las convenciones del kernel naive.
    // depth es metadato para co-simulacion HLS, no un limite del hardware.
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=650
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=195
#pragma HLS INTERFACE m_axi port=bias   offset=slave bundle=gmem2 depth=3
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3 depth=30
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

    // Unicamente buffers de tamano CONSTANTE: no se reserva espacio
    // proporcional a input_channels ni a output_channels.
    float input_buffer[CHANNEL_TILE][TILE_PIXELS];
#pragma HLS BIND_STORAGE variable=input_buffer type=ram_1p impl=bram

    float weight_buffer[PARALLEL_OUT][CHANNEL_TILE];
#pragma HLS ARRAY_PARTITION variable=weight_buffer type=complete dim=1

    float accumulator[PARALLEL_OUT][TILE_PIXELS];
#pragma HLS ARRAY_PARTITION variable=accumulator type=complete dim=1

    float bias_buffer[PARALLEL_OUT];
#pragma HLS ARRAY_PARTITION variable=bias_buffer type=complete

    if (batch_size <= 0 || input_channels <= 0 || input_height <= 0 ||
        input_width <= 0 || output_channels <= 0) {
        return;  // el host debe validar dimensiones no positivas
    }

    const int pixels = input_height * input_width;

    batch_loop:
    for (int batch = 0; batch < batch_size; ++batch) {
        const long long input_base  = (long long)batch * input_channels * pixels;
        const long long output_base = (long long)batch * output_channels * pixels;

        pixel_tile_loop:
        for (int tile_start = 0; tile_start < pixels; tile_start += TILE_PIXELS) {
            const int remaining_pixels = pixels - tile_start;
            const int tile_length = remaining_pixels < TILE_PIXELS
                                  ? remaining_pixels : TILE_PIXELS;
            const int run_length = tile_length < MIN_RUN ? MIN_RUN : tile_length;

            out_group_loop:
            for (int out_start = 0; out_start < output_channels;
                 out_start += PARALLEL_OUT) {
                const int remaining_out = output_channels - out_start;
                const int group_size = remaining_out < PARALLEL_OUT
                                     ? remaining_out : PARALLEL_OUT;

                // Bias del grupo actual; las vias no usadas se mantienen a cero.
                load_bias:
                for (int out = 0; out < PARALLEL_OUT; ++out) {
#pragma HLS PIPELINE II=1
                    bias_buffer[out] = (use_bias && out < group_size)
                        ? bias[out_start + out] : 0.0f;
                }

                // Un acumulador por canal de salida y posicion.
                // Se inicializa UNA sola vez por grupo espacial/salida;
                // los bloques sucesivos de canales de entrada lo actualizan.
                init_accumulator:
                for (int px = 0; px < tile_length; ++px) {
#pragma HLS PIPELINE II=1
                    init_parallel_out:
                    for (int out = 0; out < PARALLEL_OUT; ++out) {
#pragma HLS UNROLL
                        if (out < group_size) {
                            accumulator[out][px] = bias_buffer[out];
                        }
                    }
                }

                // Channel tiling: el numero total de canales es dinamico.
                input_channel_tile_loop:
                for (int channel_start = 0; channel_start < input_channels;
                     channel_start += CHANNEL_TILE) {
                    const int remaining_in = input_channels - channel_start;
                    const int channel_length = remaining_in < CHANNEL_TILE
                                             ? remaining_in : CHANNEL_TILE;

                    // Lecturas consecutivas en la dimension espacial NCHW.
                    load_input_channel:
                    for (int ci = 0; ci < channel_length; ++ci) {
                        load_input_pixel:
                        for (int px = 0; px < tile_length; ++px) {
#pragma HLS PIPELINE II=1
                            input_buffer[ci][px] =
                                input[input_base
                                      + (long long)(channel_start + ci) * pixels
                                      + tile_start + px];
                        }
                    }

                    // Cada fila de pesos es contigua en memoria.
                    load_weight_channel:
                    for (int out = 0; out < group_size; ++out) {
                        load_weight_input:
                        for (int ci = 0; ci < channel_length; ++ci) {
#pragma HLS PIPELINE II=1
                            weight_buffer[out][ci] =
                                weight[(long long)(out_start + out) * input_channels
                                       + channel_start + ci];
                        }
                    }

                    // El mismo accumulator[out][px] se visita solo despues
                    // de run_length iteraciones (>= MIN_RUN).
                    // El pragma DEPENDENCE presupone que esta distancia
                    // supera la latencia real de lectura/suma/escritura:
                    // debe verificarse en csynth y co-simulacion RTL.
                    int px = 0;
                    int ci = 0;
                    compute_loop:
                    for (int i = 0; i < channel_length * run_length; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accumulator type=inter false
                        if (px < tile_length) {
                            const float x = input_buffer[ci][px];
                            compute_parallel_out:
                            for (int out = 0; out < PARALLEL_OUT; ++out) {
#pragma HLS UNROLL
                                if (out < group_size) {
                                    accumulator[out][px] +=
                                        weight_buffer[out][ci] * x;
                                }
                            }
                        }

                        if (px == run_length - 1) {
                            px = 0;
                            ++ci;
                        } else {
                            ++px;
                        }
                    }
                }

                // Escrituras consecutivas para cada canal de salida.
                store_channel:
                for (int out = 0; out < group_size; ++out) {
                    store_pixel:
                    for (int px = 0; px < tile_length; ++px) {
#pragma HLS PIPELINE II=1
                        output[output_base
                               + (long long)(out_start + out) * pixels
                               + tile_start + px] = accumulator[out][px];
                    }
                }
            }
        }
    }
}
