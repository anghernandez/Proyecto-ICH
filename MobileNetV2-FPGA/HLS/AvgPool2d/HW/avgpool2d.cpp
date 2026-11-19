#include "avgpool2d.hpp"


// ------------------------------------------------------------
// Kernel HLS para la operación Average Pooling 2D
// Reduce las dimensiones espaciales de los tensores mediante
// el cálculo del promedio de los elementos de cada ventana
// Procesa cada canal de entrada de manera independiente
// ------------------------------------------------------------
void avg_pool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int input_channels,
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
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
)
{


// ------------------------------------------------------------
// Interfaces AXI4 Master (m_axi)
// Permiten leer los datos de entrada y escribir los resultados
// en memoria externa mediante interfaces independientes
// El parámetro depth especifica la profundidad para HLS
// ------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=4014080
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=81920

// ------------------------------------------------------------
// Interfaces AXI4-Lite (s_axilite)
// Permiten configurar las direcciones de memoria, dimensiones,
// tamaño de ventana, stride, padding y parámetros del promedio
// desde el host
// ------------------------------------------------------------
#pragma HLS INTERFACE s_axilite port=input			bundle=control
#pragma HLS INTERFACE s_axilite port=output

#pragma HLS INTERFACE s_axilite port=batch_size			bundle=control
#pragma HLS INTERFACE s_axilite port=input_channels		bundle=control
#pragma HLS INTERFACE s_axilite port=input_height		bundle=control
#pragma HLS INTERFACE s_axilite port=input_width		bundle=control

#pragma HLS INTERFACE s_axilite port=output_height		bundle=control
#pragma HLS INTERFACE s_axilite port=output_width		bundle=control

#pragma HLS INTERFACE s_axilite port=kernel_height		bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_width		bundle=control

#pragma HLS INTERFACE s_axilite port=stride_height		bundle=control
#pragma HLS INTERFACE s_axilite port=stride_width		bundle=control

#pragma HLS INTERFACE s_axilite port=padding_height		bundle=control
#pragma HLS INTERFACE s_axilite port=padding_width		bundle=control

#pragma HLS INTERFACE s_axilite port=count_include_pad		bundle=control
#pragma HLS INTERFACE s_axilite port=use_divisor_override	bundle=control
#pragma HLS INTERFACE s_axilite port=divisor_override		bundle=control

#pragma HLS INTERFACE s_axilite port=return  

/* se usan los s_axilite para las interfaces de tipo AXI4 Lite para  los parametros de control
    o sea que se tiene:
    AXI4 Master
        ↓
    datos grandes
    input / output

    AXI4-Lite
        ↓
    control y parámetros
    dimensiones / configuración
*/

    // ------------------------------------------------------------
    // Recorre los batches, canales y posiciones espaciales
    // del tensor de salida.
    // Cada posición corresponde al promedio de una ventana
    // del tensor de entrada.
    // ------------------------------------------------------------
    for (int batch_index = 0;  
         batch_index < batch_size;
         batch_index++)
    {
        for (int channel_index = 0;
             channel_index < input_channels;
             channel_index++)
        {
            for (int output_row = 0;
                 output_row < output_height;
                 output_row++)
            {
                for (int output_column = 0;
                     output_column < output_width;
                     output_column++)
                {
#pragma HLS PIPELINE
/*
    HLS: aplica pipelining al ciclo que recorre las columnas de salida.
    Permite solapar la ejecución de diferentes iteraciones del ciclo,
    buscando mejorar el rendimiento del kernel en hardware

    Se coloca en el loop de cada iteración de output_column porque cada elemento implica:
    determinar ventana
        ↓
    leer elementos de input
        ↓
    sumarlos
        ↓
    calcular divisor
        ↓
    dividir
        ↓
    escribir output

    Además los otros loop recorren imagenes, canales, filas o datos de una ventana
*/
		    // Calcula las coordenadas iniciales de la ventana de pooling
          	    // considerando el desplazamiento (stride) y el padding
                    const int window_start_row =
                        output_row * stride_height
                        - padding_height;

                    const int window_start_column =
                        output_column * stride_width
                        - padding_width;

		    // ------------------------------------------------------------
     		    // Calcula los límites finales de la ventana considerando
		    // el padding y restringiéndolos al tamaño extendido
                    // de la entrada
		    // ------------------------------------------------------------
                    int padded_end_row =
                        window_start_row + kernel_height;

                    if (padded_end_row >
                        input_height + padding_height)
                    {
                        padded_end_row =
                            input_height + padding_height;
                    }

                    //calcula el final de la ventana
                    int padded_end_column =
                        window_start_column + kernel_width;

                    if (padded_end_column >
                        input_width + padding_width)
                    {
                        padded_end_column =
                            input_width + padding_width;
                    }


		    // ------------------------------------------------------------
		    // Determina las coordenadas válidas de la ventana dentro
		    // del tensor de entrada, excluyendo las posiciones
		    // correspondientes al padding
		    // ------------------------------------------------------------
                    int valid_start_row =
                        window_start_row;

                    if (valid_start_row < 0)
                    {
                        valid_start_row = 0;
                    }

                    int valid_start_column =
                        window_start_column;

                    if (valid_start_column < 0)
                    {
                        valid_start_column = 0;
                    }

                    int valid_end_row =
                        padded_end_row;

                    if (valid_end_row > input_height)
                    {
                        valid_end_row = input_height;
                    }

                    int valid_end_column =
                        padded_end_column;

                    if (valid_end_column > input_width)
                    {
                        valid_end_column = input_width;
                    }

		    // Calcula el número de posiciones de la ventana considerando
		    // los límites establecidos y el padding
                    const int pool_height =  
                        padded_end_row
                        - window_start_row;

                    const int pool_width =
                        padded_end_column
                        - window_start_column;

                    const int pool_size =
                        pool_height * pool_width;



                    float accumulated_value = 0.0f; 

                    for (int input_row = valid_start_row;
                         input_row < valid_end_row;
                         input_row++)
                    {
                        for (int input_column = valid_start_column;
                             input_column < valid_end_column;
                             input_column++)
                        {
                            
                            // Calcula el índice lineal del elemento de entrada
			    // almacenado en formato NCHW
                            const int input_position =
                                (
                                    (
                                        batch_index
                                        * input_channels
                                        + channel_index
                                    )
                                    * input_height
                                    + input_row
                                )
                                * input_width
                                + input_column;

	 		    // Suma el valor actual de la ventana al acumulador
                            accumulated_value +=
                                input[input_position];
                        }
                    }

		    // Calcula la cantidad de elementos válidos de la ventana,
		    // excluyendo las posiciones correspondientes al padding
                    const int valid_height =
                        valid_end_row
                        - valid_start_row;

                    const int valid_width =
                        valid_end_column
                        - valid_start_column;

                    const int valid_value_count =
                        valid_height
                        * valid_width;

		    
		   // ------------------------------------------------------------
		   // Determina el divisor utilizado para calcular el promedio
		   // Prioriza divisor_override si está habilitado
		   // En caso contrario, incluye o excluye las posiciones
		   // de padding según count_include_pad.
		   // ------------------------------------------------------------
                    int divisor;

                    if (use_divisor_override)
                    {
                        divisor =
                            divisor_override;
                    }
                    else if (count_include_pad)
                    {
                        divisor =
                            pool_size;
                    }
                    else
                    {
                        divisor =
                            valid_value_count;
                    }

		    // Calcula el promedio dividiendo la suma acumulada
		    // entre el divisor seleccionado
                    const float average_value =
                        accumulated_value
                        / (float) divisor;

		    // Calcula el índice lineal del tensor de salida
                    // almacenado en formato NCHW
                    const int output_position =
                        (
                            (
                                batch_index
                                * input_channels
                                + channel_index
                            )
                            * output_height
                            + output_row
                        )
                        * output_width
                        + output_column;
                    
		    // Almacena el promedio calculado en la posición de salida
                    output[output_position] =
                        average_value;
                }
            }
        }
    }
}
