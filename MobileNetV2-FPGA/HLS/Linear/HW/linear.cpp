#include "linear.hpp"

// ------------------------------------------------------------
// Kernel HLS para la operación Linear (capa totalmente conectada)
// Calcula una transformación lineal mediante la multiplicación
// de las entradas por sus pesos y la suma de los productos
// Opcionalmente, incorpora el bias correspondiente a cada salida
// ------------------------------------------------------------
void linear_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    bool use_bias
)
{

// ------------------------------------------------------------
// Interfaces AXI4 Master (m_axi)
// Permiten leer los datos de entrada, pesos y bias, y escribir
// los resultados en memoria externa
// Se utilizan cuatro interfaces independientes
// ------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=81920
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=1280000
#pragma HLS INTERFACE m_axi port=bias   offset=slave bundle=gmem2 depth=1000
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3 depth=64000

// ------------------------------------------------------------
// Interfaces AXI4-Lite (s_axilite).
// Permiten configurar las direcciones de memoria, el tamaño
// del batch, las dimensiones de entrada y salida, y el uso
// del bias desde el host mediante el bundle de control.
// ------------------------------------------------------------
#pragma HLS INTERFACE s_axilite port=input        bundle=control
#pragma HLS INTERFACE s_axilite port=weight       bundle=control
#pragma HLS INTERFACE s_axilite port=bias         bundle=control
#pragma HLS INTERFACE s_axilite port=output       bundle=control
#pragma HLS INTERFACE s_axilite port=batch_size   bundle=control
#pragma HLS INTERFACE s_axilite port=in_features  bundle=control
#pragma HLS INTERFACE s_axilite port=out_features bundle=control
#pragma HLS INTERFACE s_axilite port=use_bias     bundle=control
#pragma HLS INTERFACE s_axilite port=return       bundle=control



    // ------------------------------------------------------------
    // Recorre las muestras del batch y las características de
    // salida (out_features).
    // Cada característica de salida se calcula mediante una
    // suma ponderada de las características de entrada.
    // ------------------------------------------------------------
    for (
        int sample_index = 0;
        sample_index < batch_size;
        sample_index++
    ) {
        for (
            int output_index = 0;
            output_index < out_features;
            output_index++
        ) {
        
            // Inicializa el acumulador en cero para calcular la suma
	    // de los productos correspondientes a una salida
            float accumulated_value = 0.0f;


	    // ------------------------------------------------------------
	    // Recorre las características de entrada (in_features)
	    // para calcular el producto entre cada entrada y su peso
	    // ------------------------------------------------------------
            for (
                int input_index = 0;
                input_index < in_features;
                input_index++
            ) {
 
// Solicita el solapamiento de las iteraciones del bucle
// de características de entrada para mejorar el rendimiento
#pragma HLS PIPELINE

		// Calcula el índice lineal de la característica de entrada
		// correspondiente a la muestra actual
                const int input_position =
                    sample_index * in_features
                    + input_index;

		// Calcula el índice lineal del peso correspondiente
		// a la característica de entrada y salida actuales
                const int weight_position =
                    output_index * in_features
                    + input_index;

		// Lee el valor de entrada y su peso correspondiente
		// desde los buffers de memoria
                const float input_value =
                    input[input_position];

                const float weight_value =
                    weight[weight_position];

		// Multiplica la característica de entrada por su peso
		// y suma el producto al acumulador
                accumulated_value += (
                    input_value * weight_value
                );
            }

	    // Agrega el bias correspondiente a la característica
	    // de salida cuando está habilitado
            if (use_bias) {
                accumulated_value +=
                    bias[output_index];
            }

	    // Calcula el índice lineal de la característica de salida
	    // correspondiente a la muestra actual
            const int output_position =
                sample_index * out_features
                + output_index;

	    // Almacena el resultado de la transformación lineal
	    // en el tensor de salida
            output[output_position] =
                accumulated_value;
        }
    }
}
