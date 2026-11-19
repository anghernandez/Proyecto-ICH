#include "relu6.hpp"

// ------------------------------------------------------------
// Kernel HLS para la función de activación ReLU6.
// Aplica la activación elemento a elemento, limitando
// los valores de entrada al intervalo [0, 6].
// ------------------------------------------------------------
void relu6_forward(
    const float* input,
    float* output,
    int size
)
{

// ------------------------------------------------------------
// Interfaces AXI4 Master (m_axi).
// Permiten leer los datos de entrada y escribir los resultados
// en memoria externa mediante dos interfaces independientes
// El parámetro depth utiliza size como profundidad declarada
// ------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=size
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=size

// ------------------------------------------------------------
// Interfaces AXI4-Lite (s_axilite).
// Permiten configurar las direcciones de los buffers,
// la cantidad de elementos (size) y el control de ejecución
// del kernel desde el host
// ------------------------------------------------------------
#pragma HLS INTERFACE s_axilite port=input
#pragma HLS INTERFACE s_axilite port=output
#pragma HLS INTERFACE s_axilite port=size
#pragma HLS INTERFACE s_axilite port=return


    // ------------------------------------------------------------
    // Recorre secuencialmente los índices de todos los elementos
    // del tensor para aplicar la función de activación
    // ------------------------------------------------------------
    for (int index = 0;
         index < size;
         index++) {

// Solicita el solapamiento de las iteraciones del bucle
// para mejorar el rendimiento del procesamiento
#pragma HLS PIPELINE

	// Solicita el solapamiento de las iteraciones del bucle
	// para mejorar el rendimiento del procesamiento
        const float input_value =
            input[index];

	// ------------------------------------------------------------
	// Aplica la función ReLU6:
	// - Si el valor es menor que 0, devuelve 0.
	// - Si el valor es mayor que 6, devuelve 6.
	// - En caso contrario, conserva el valor original.
	// ------------------------------------------------------------
        float output_value;

        if (input_value < 0.0f) {
            output_value = 0.0f;
        }
        else if (input_value > 6.0f) {
            output_value = 6.0f;
        }
        else {
            output_value = input_value;
        }

	// Almacena el valor resultante de la activación ReLU6
	// en la posición correspondiente del tensor de salida
        output[index] = output_value;
    }
}
