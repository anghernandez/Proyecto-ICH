#include "pointwise_conv2d.hpp"

// ------------------------------------------------------------
// Kernel HLS para la operación PointwiseConv2D.
// Implementa una convolución 1x1 que combina los valores
// de los canales de entrada para generar nuevos canales.
// Mantiene las dimensiones espaciales del tensor y permite
// modificar la cantidad de canales de salida.
// ------------------------------------------------------------
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
// Permiten leer los datos de entrada, pesos y bias,
// y escribir los resultados en memoria externa
// Se utilizan cuatro interfaces independientes
// ------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=12845056
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=409600
#pragma HLS INTERFACE m_axi port=bias   offset=slave bundle=gmem2 depth=1280
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3 depth=77070336

// ------------------------------------------------------------
// Interfaces AXI4-Lite (s_axilite)
// Permiten configurar las direcciones de memoria,
// dimensiones del tensor, cantidad de canales de entrada
// y salida, y uso del bias desde el host
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

   // ------------------------------------------------------------
   // Recorre los batches, canales de salida y posiciones
   // espaciales del tensor.
   // Cada elemento de salida combina los valores de todos
   // los canales de entrada en una misma posición espacial.
   // ------------------------------------------------------------
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int output_channel = 0;
             output_channel < output_channels;
             output_channel++) {

            for (int output_row = 0;
                 output_row < input_height;
                 output_row++) {

                for (int output_column = 0;
                     output_column < input_width;
                     output_column++) {
		    
		    // Inicializa el acumulador con el bias del canal de salida,
	            // si está habilitado; de lo contrario, comienza en cero
                    float accumulated_value = 0.0f;

                    if (use_bias) {
                        accumulated_value = bias[output_channel];
                    }

		    // ------------------------------------------------------------
		    // Recorre todos los canales de entrada para calcular
		    // la suma ponderada correspondiente al canal de salida.
		    // ------------------------------------------------------------
                    for (int input_channel = 0;
                         input_channel < input_channels;
                         input_channel++) {
                         
// Solicita el solapamiento de las iteraciones del bucle
// de canales de entrada para mejorar el rendimiento
#pragma HLS PIPELINE

			// Calcula el índice lineal del elemento de entrada
			// en formato NCHW, utilizando la posición espacial actual
                        const int input_position =
                            (((batch_index * input_channels
                            + input_channel)
                            * input_height
                            + output_row)
                            * input_width)
                            + output_column;

			// Calcula el índice del peso asociado al canal de entrada
			// y al canal de salida actuales
                        const int weight_position =
                            (output_channel * input_channels)
                            + input_channel;


		    	// Multiplica el valor de entrada por su peso correspondiente
			// y suma el producto al acumulador
                        accumulated_value +=
                            input[input_position]
                            * weight[weight_position];
                    }

		    // Calcula el índice lineal del elemento de salida
		    // en formato NCHW
                    const int output_position =
                        (((batch_index * output_channels
                        + output_channel)
                        * input_height
                        + output_row)
                        * input_width)
                        + output_column;

		    // Almacena el resultado de la convolución 1x1
		    // en la posición correspondiente del tensor de salida
                    output[output_position] =
                        accumulated_value;
                }
            }
        }
    }
}
