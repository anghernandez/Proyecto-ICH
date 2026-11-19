#include "tanh.hpp"
#include <hls_math.h>// Funciones matemáticas optimizadas para síntesis HLS.

// ------------------------------------------------------------
// Función auxiliar para calcular la tangente hiperbólica.
// Utiliza una formulación basada en funciones exponenciales
// para obtener valores de salida en el intervalo [-1, 1].
// ------------------------------------------------------------
static float tanh_scalar(float value)
{

// Solicita integrar la lógica de esta función directamente
// en el hardware de la función que la invoca
#pragma HLS INLINE

    // ------------------------------------------------------------
    // Para valores mayores o iguales a cero, calcula Tanh
    // utilizando una exponencial negativa.
    // Esta formulación evita calcular exponenciales positivas
    // grandes que podrían provocar desbordamiento numérico
    // ------------------------------------------------------------
    if (value >= 0.0f) {
    	
    	// Calcula e^(-2x) mediante la función exponencial de HLS
        const float exponential = hls::expf(
            -2.0f * value
        );

	// Calcula Tanh mediante la expresión (1 - e^(-2x))
	// dividida entre (1 + e^(-2x))
        return (
            (1.0f - exponential)
            / (1.0f + exponential)
        );
    }

    // Calcula e^(2x) para valores negativos de entrada,
    // evitando exponenciales positivas de gran magnitud
    const float exponential = hls::expf(
        2.0f * value
    );

    // Calcula e^(2x) para valores negativos de entrada,
    // evitando exponenciales positivas de gran magnitud
    return (
        (exponential - 1.0f)
        / (exponential + 1.0f)
    );
}

// ------------------------------------------------------------
// Kernel HLS para la función de activación Tanh.
// Aplica la tangente hiperbólica a cada elemento del tensor
// de entrada mediante la función auxiliar tanh_scalar
// ------------------------------------------------------------
void tanh_forward(
    const float* input,
    float* output,
    int size
)
{

// ------------------------------------------------------------
// Interfaces AXI4 Master (m_axi)
// Permiten leer los datos de entrada y escribir los resultados
// en memoria externa mediante dos interfaces independientes
// ------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=size
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=size

// ------------------------------------------------------------
// Interfaces AXI4-Lite (s_axilite)
// Permiten configurar las direcciones de memoria, la cantidad
// de elementos (size) y el control de ejecución desde el host
// Los argumentos se agrupan en el bundle control.
// ------------------------------------------------------------
#pragma HLS INTERFACE s_axilite port=input  bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=size   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // ------------------------------------------------------------
    // Recorre los elementos del tensor de entrada para aplicar
    // la función de activación Tanh de manera independiente.
    // ------------------------------------------------------------
    for (int index = 0; index < size; index++) {
  
// Solicita el solapamiento de las iteraciones del bucle
// para mejorar el rendimiento del procesamiento
#pragma HLS PIPELINE

	// Calcula la tangente hiperbólica del elemento actual
	// y almacena el resultado en el tensor de salida
        output[index] = tanh_scalar(
            input[index]
        );
    }
}
