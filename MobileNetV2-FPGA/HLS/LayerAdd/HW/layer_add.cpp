#include "layer_add.hpp"

// ------------------------------------------------------------
// Kernel HLS para la operación LayerAdd
// Realiza la suma elemento a elemento de dos tensores
// multiplicando los valores del segundo tensor por alpha
// Ambos tensores deben tener la misma cantidad de elementos
// ------------------------------------------------------------
void layer_add_forward(
    const float* input1,
    const float* input2,
    float* output,
    int size,
    float alpha
)
{
// ------------------------------------------------------------
// Interfaces AXI4 Master (m_axi)
// Permiten acceder a los dos tensores de entrada y escribir
// los resultados en memoria externa
// Se utilizan tres interfaces independientes.
// ------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=input1 offset=slave bundle=gmem0 depth=64
#pragma HLS INTERFACE m_axi port=input2 offset=slave bundle=gmem1 depth=64
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2 depth=64

// ------------------------------------------------------------
// Interfaces AXI4-Lite (s_axilite)
// Permiten configurar las direcciones de memoria, la cantidad
// de elementos (size) y el factor de escala (alpha)
// desde el host
// ------------------------------------------------------------
#pragma HLS INTERFACE s_axilite port=input1
#pragma HLS INTERFACE s_axilite port=input2
#pragma HLS INTERFACE s_axilite port=output
#pragma HLS INTERFACE s_axilite port=size
#pragma HLS INTERFACE s_axilite port=alpha
#pragma HLS INTERFACE s_axilite port=return

    // ------------------------------------------------------------
    // Recorre los elementos de ambos tensores utilizando
    // un índice lineal para realizar la suma elemento a elemento.
    // ------------------------------------------------------------
    for (int index = 0;
         index < size;
         index++) {
         
// Solicita el solapamiento de las iteraciones del bucle
// para mejorar el rendimiento del procesamiento
#pragma HLS PIPELINE

	// Lee los valores correspondientes a la misma posición
	// de ambos tensores de entrada
        const float input1_value =
            input1[index];

        const float input2_value =
            input2[index];

	// Suma el primer valor con el segundo multiplicado por alpha
	// y almacena el resultado en el tensor de salida
        output[index] =
            input1_value
            + alpha * input2_value;
    }
}
