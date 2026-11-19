#include "depthwise_conv2d.hpp"

// ------------------------------------------------------------
// Kernel HLS para la operación DepthwiseConv2D
// Aplica un filtro independiente a cada canal de entrada
// sin combinar información entre canales
// Recibe los datos de los tensores y los parámetros
// necesarios para configurar la convolución
// ------------------------------------------------------------
void depthwise_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{

// ------------------------------------------------------------
// Interfaces AXI4 Master (m_axi)
// Permiten acceder a los datos de entrada, pesos y bias
// así como escribir los resultados en memoria externa
// Se utilizan cuatro interfaces de memoria independientes
// ------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=77070336
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=8640
#pragma HLS INTERFACE m_axi port=bias   offset=slave bundle=gmem2 depth=960
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3 depth=28901376

// ------------------------------------------------------------
// Interfaces AXI4-Lite (s_axilite).
// Permiten configurar las direcciones de memoria,
// dimensiones de los tensores, stride, padding
// y demás parámetros de ejecución desde el host.
// ------------------------------------------------------------
#pragma HLS INTERFACE s_axilite port=input
#pragma HLS INTERFACE s_axilite port=weight
#pragma HLS INTERFACE s_axilite port=bias
#pragma HLS INTERFACE s_axilite port=output

#pragma HLS INTERFACE s_axilite port=batch_size
#pragma HLS INTERFACE s_axilite port=channels
#pragma HLS INTERFACE s_axilite port=input_height
#pragma HLS INTERFACE s_axilite port=input_width
#pragma HLS INTERFACE s_axilite port=output_height
#pragma HLS INTERFACE s_axilite port=output_width
#pragma HLS INTERFACE s_axilite port=kernel_height
#pragma HLS INTERFACE s_axilite port=kernel_width
#pragma HLS INTERFACE s_axilite port=stride_height
#pragma HLS INTERFACE s_axilite port=stride_width
#pragma HLS INTERFACE s_axilite port=padding_height
#pragma HLS INTERFACE s_axilite port=padding_width
#pragma HLS INTERFACE s_axilite port=use_bias
#pragma HLS INTERFACE s_axilite port=return

// ------------------------------------------------------------
// Recorre los batches, canales y posiciones espaciales
// del tensor de salida.
// Cada canal se procesa de manera independiente.
// ------------------------------------------------------------
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int channel = 0;
             channel < channels;
             channel++) {

            for (int output_row = 0;
                 output_row < output_height;
                 output_row++) {

                for (int output_column = 0;
                     output_column < output_width;
                     output_column++) {

                    // Inicializa el acumulador con el bias del canal,
		    // si está habilitado; de lo contrario, comienza en cero
                    float accumulated_value = 0.0f;

                    if (use_bias) {
                        accumulated_value = bias[channel];
                    }

		    // ------------------------------------------------------------
		    // Recorre las filas y columnas del filtro correspondiente
		    // al canal actual para calcular la convolución
		    // ------------------------------------------------------------
                    for (int kernel_row = 0;
                         kernel_row < kernel_height;
                         kernel_row++) {

			// Calcula la fila de entrada considerando stride y padding
                        const int input_row =
                            output_row * stride_height
                            + kernel_row
                            - padding_height;

			// Calcula la columna de entrada considerando stride y padding
                        for (int kernel_column = 0;
                             kernel_column < kernel_width;
                             kernel_column++) {

// Solicita el solapamiento de las iteraciones del bucle
// interno para mejorar el rendimiento del procesamiento
#pragma HLS PIPELINE

                            const int input_column =
                                output_column * stride_width
                                + kernel_column
                                - padding_width;

                            // Verifica que las coordenadas estén dentro del tensor
			    // de entrada, implementando padding con ceros
                            if (
                                input_row >= 0
                                && input_row < input_height
                                && input_column >= 0
                                && input_column < input_width
                            ) {
                            
                            	// Calcula el índice lineal del tensor de entrada
				// almacenado en formato NCHW
                                const int input_position =
                                    (((batch_index * channels
                                    + channel)
                                    * input_height
                                    + input_row)
                                    * input_width)
                                    + input_column;

				// Calcula el índice lineal del peso correspondiente
				// al canal y a la posición actual del filtro
                                const int weight_position =
                                    ((channel * kernel_height
                                    + kernel_row)
                                    * kernel_width)
                                    + kernel_column;

				// Multiplica el valor de entrada por el peso correspondiente
				// y suma el producto al acumulador
                                accumulated_value +=
                                    input[input_position]
                                    * weight[weight_position];
                            }
                        }
                    }
		    
		    // Calcula el índice lineal del tensor de salida
		    // en formato NCHW
                    const int output_position =
                        (((batch_index * channels
                        + channel)
                        * output_height
                        + output_row)
                        * output_width)
                        + output_column;

		    // Almacena el resultado de la convolución del canal actual
                    output[output_position] =
                        accumulated_value;
                }
            }
        }
    }
}
