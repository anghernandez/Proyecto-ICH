#include "Conv2d.hpp"

void conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
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
    for (
        int batch_index = 0;
        batch_index < batch_size;
        batch_index++
    ) {
        for (
            int output_channel = 0;
            output_channel < output_channels;
            output_channel++
        ) {
            for (
                int output_row = 0;
                output_row < output_height;
                output_row++
            ) {
                for (
                    int output_column = 0;
                    output_column < output_width;
                    output_column++
                ) {
                    float accumulated_value = 0.0f;

                    if (use_bias) {
                        accumulated_value = bias[
                            output_channel
                        ];
                    }

                    // Aquí agregaremos la convolución.
                    for (
                        int input_channel = 0;
                        input_channel < input_channels;
                        input_channel++
                    ) {
                        for (
                            int kernel_row = 0;
                            kernel_row < kernel_height;
                            kernel_row++
                        ) {
                            const int input_row =
                                output_row * stride_height
                                + kernel_row
                                - padding_height;

                            for (
                                int kernel_column = 0;
                                kernel_column < kernel_width;
                                kernel_column++
                            ) {
                                const int input_column =
                                    output_column * stride_width
                                    + kernel_column
                                    - padding_width;

                                if (
                                    input_row >= 0
                                    && input_row < input_height
                                    && input_column >= 0
                                    && input_column < input_width
                                ) {
                                    const int input_position =
                                        (
                                            (
                                                batch_index
                                                * input_channels
                                                + input_channel
                                            )
                                            * input_height
                                            + input_row
                                        )
                                        * input_width
                                        + input_column;

                                    const int weight_position =
                                        (
                                            (
                                                output_channel
                                                * input_channels
                                                + input_channel
                                            )
                                            * kernel_height
                                            + kernel_row
                                        )
                                        * kernel_width
                                        + kernel_column;

                                    const float input_value =
                                        input[input_position];

                                    const float weight_value =
                                        weight[weight_position];

                                    accumulated_value += (
                                        input_value * weight_value
                                    );
                                }
                            }
                        }
                    }

                    const int output_position =
                        (
                            (
                                batch_index
                                * output_channels
                                + output_channel
                            )
                            * output_height
                            + output_row
                        )
                        * output_width
                        + output_column;

                    output[output_position] =
                        accumulated_value;

                }
            }
        }
    }
}