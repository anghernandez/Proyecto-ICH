#include "Depthwise_Conv2d.hpp"

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
    // Tamaño espacial de cada canal.
    const int input_channel_size =
        input_height * input_width;

    const int output_channel_size =
        output_height * output_width;

    const int kernel_size =
        kernel_height * kernel_width;

    for (
        int batch_index = 0;
        batch_index < batch_size;
        batch_index++
    ) {
        for (
            int channel = 0;
            channel < channels;
            channel++
        ) {
            // Bases que no cambian mientras se procesa este canal.
            const int input_channel_base =
                (
                    batch_index * channels
                    + channel
                )
                * input_channel_size;

            const int output_channel_base =
                (
                    batch_index * channels
                    + channel
                )
                * output_channel_size;

            const int weight_channel_base =
                channel * kernel_size;

            const float bias_value =
                use_bias
                    ? bias[channel]
                    : 0.0f;

            for (
                int output_row = 0;
                output_row < output_height;
                output_row++
            ) {
                // Primera fila de entrada correspondiente
                // a esta fila de salida.
                const int input_row_start =
                    output_row * stride_height
                    - padding_height;

                const int output_row_base =
                    output_channel_base
                    + output_row * output_width;

                for (
                    int output_column = 0;
                    output_column < output_width;
                    output_column++
                ) {
                    float accumulated_value =
                        bias_value;

                    // Primera columna de entrada correspondiente
                    // a esta posición de salida.
                    const int input_column_start =
                        output_column * stride_width
                        - padding_width;

                    for (
                        int kernel_row = 0;
                        kernel_row < kernel_height;
                        kernel_row++
                    ) {
                        const int input_row =
                            input_row_start
                            + kernel_row;

                        // Si toda esta fila del kernel está
                        // fuera de la imagen, no procesarla.
                        if (
                            input_row < 0
                            || input_row >= input_height
                        ) {
                            continue;
                        }

                        const int input_row_base =
                            input_channel_base
                            + input_row * input_width;

                        const int weight_row_base =
                            weight_channel_base
                            + kernel_row * kernel_width;

                        for (
                            int kernel_column = 0;
                            kernel_column < kernel_width;
                            kernel_column++
                        ) {
                            const int input_column =
                                input_column_start
                                + kernel_column;

                            if (
                                input_column < 0
                                || input_column >= input_width
                            ) {
                                continue;
                            }

                            const float input_value =
                                input[
                                    input_row_base
                                    + input_column
                                ];

                            const float weight_value =
                                weight[
                                    weight_row_base
                                    + kernel_column
                                ];

                            accumulated_value +=
                                input_value
                                * weight_value;
                        }
                    }

                    output[
                        output_row_base
                        + output_column
                    ] = accumulated_value;
                }
            }
        }
    }
}