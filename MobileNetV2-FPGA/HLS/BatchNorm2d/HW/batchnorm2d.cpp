#include "batchnorm2d.hpp"
#include <cmath> // Funciones matemáticas para calcular la raíz cuadrada.

// ------------------------------------------------------------
// Kernel HLS para la operación Batch Normalization 2D
// Normaliza los valores de cada canal utilizando la media
// y varianza almacenadas durante el entrenamiento
// Opcionalmente, aplica una transformación afín mediante
// los parámetros gamma (weight) y beta (bias)
// ------------------------------------------------------------
void batchnorm2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    float eps,
    bool affine,
    bool use_bias
)
{

// ------------------------------------------------------------
// Interfaces AXI4 Master (m_axi)
// Permiten acceder a los datos de entrada, parámetros de
// normalización (weight, bias, running_mean y running_var)
// y escribir los resultados en memoria externa
// ------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=input        offset=slave bundle=gmem0 depth=48
#pragma HLS INTERFACE m_axi port=weight       offset=slave bundle=gmem1 depth=3
#pragma HLS INTERFACE m_axi port=bias         offset=slave bundle=gmem2 depth=3
#pragma HLS INTERFACE m_axi port=running_mean offset=slave bundle=gmem3 depth=3
#pragma HLS INTERFACE m_axi port=running_var  offset=slave bundle=gmem4 depth=3
#pragma HLS INTERFACE m_axi port=output       offset=slave bundle=gmem5 depth=48

// ------------------------------------------------------------
// Interfaces AXI4-Lite (s_axilite)
// Permiten configurar las direcciones de memoria, dimensiones
// del tensor, epsilon y parámetros de transformación afín
// desde el host
// ------------------------------------------------------------
#pragma HLS INTERFACE s_axilite port=input
#pragma HLS INTERFACE s_axilite port=weight
#pragma HLS INTERFACE s_axilite port=bias
#pragma HLS INTERFACE s_axilite port=running_mean
#pragma HLS INTERFACE s_axilite port=running_var
#pragma HLS INTERFACE s_axilite port=output

#pragma HLS INTERFACE s_axilite port=batch_size
#pragma HLS INTERFACE s_axilite port=channels
#pragma HLS INTERFACE s_axilite port=input_height
#pragma HLS INTERFACE s_axilite port=input_width
#pragma HLS INTERFACE s_axilite port=eps
#pragma HLS INTERFACE s_axilite port=affine
#pragma HLS INTERFACE s_axilite port=use_bias
#pragma HLS INTERFACE s_axilite port=return


    // ------------------------------------------------------------
    // Recorre los batches y canales del tensor de entrada.
    // Cada canal se normaliza utilizando sus propios parámetros
    // estadísticos y de transformación afín.
    // ------------------------------------------------------------
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int channel_index = 0;
             channel_index < channels;
             channel_index++) {

	    // Obtiene la media y varianza almacenadas para el canal actual
            const float mean =
                running_mean[channel_index];

            const float variance =
                running_var[channel_index];

	    // Calcula la raíz cuadrada de la varianza más epsilon
	    // Epsilon evita la división entre cero y mejora la
	    // estabilidad numérica de la normalización
            const float denominator =
                std::sqrt(variance + eps);


	    // ------------------------------------------------------------
    	    // Inicializa gamma en 1 y beta en 0.
	    // Si affine está habilitado, utiliza los parámetros aprendidos
    	    // para escalar y desplazar los valores normalizados.
            // ------------------------------------------------------------
            float gamma = 1.0f;
            float beta = 0.0f;

            if (affine) {

                gamma =
                    weight[channel_index];

                if (use_bias) {
                    beta =
                        bias[channel_index];
                }
            }

	    // ------------------------------------------------------------
	    // Recorre las filas y columnas del canal actual para aplicar
	    // la normalización a cada elemento del tensor.
	   // ------------------------------------------------------------
            for (int row_index = 0;
                 row_index < input_height;
                 row_index++) {

                for (int column_index = 0;
                     column_index < input_width;
                     column_index++) {
                     
// Solicita el solapamiento de las iteraciones del bucle
// de columnas para mejorar el rendimiento del kernel
#pragma HLS PIPELINE

		    // Calcula el índice lineal del elemento actual
		    // en el tensor almacenado en formato NCHW
                    const int position =
                        (((batch_index * channels
                        + channel_index)
                        * input_height
                        + row_index)
                        * input_width)
                        + column_index;

		    // Obtiene el valor del elemento actual del tensor de entrada.
                    const float input_value =
                        input[position];

		    // Normaliza el valor restando la media del canal y
		    // dividiendo entre la raíz cuadrada de la varianza más epsilon
                    const float normalized_value =
                        (input_value - mean)
                        / denominator;

		    // Aplica la transformación afín mediante gamma y beta,
		    // y almacena el resultado en el tensor de salida
                    output[position] =
                        gamma
                        * normalized_value
                        + beta;
                }
            }
        }
    }
}
