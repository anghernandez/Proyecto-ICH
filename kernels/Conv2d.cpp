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
        const int input_batch_base =
            batch_index
            * input_channels
            * input_channel_size;

        const int output_batch_base =
            batch_index
            * output_channels
            * output_channel_size;

        for (
            int output_channel = 0;
            output_channel < output_channels;
            output_channel++
        ) {
            const int output_channel_base =
                output_batch_base
                + output_channel
                * output_channel_size;

            const int weight_output_base =
                output_channel
                * input_channels
                * kernel_size;

            const float bias_value =
                use_bias
                    ? bias[output_channel]
                    : 0.0f;

            for (
                int output_row = 0;
                output_row < output_height;
                output_row++
            ) {
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

                    const int input_column_start =
                        output_column * stride_width
                        - padding_width;

                    for (
                        int input_channel = 0;
                        input_channel < input_channels;
                        input_channel++
                    ) {
                        const int input_channel_base =
                            input_batch_base
                            + input_channel
                            * input_channel_size;

                        const int weight_channel_base =
                            weight_output_base
                            + input_channel
                            * kernel_size;

                        for (
                            int kernel_row = 0;
                            kernel_row < kernel_height;
                            kernel_row++
                        ) {
                            const int input_row =
                                input_row_start
                                + kernel_row;

                            if (
                                input_row < 0
                                || input_row >= input_height
                            ) {
                                continue;
                            }

                            const int input_row_base =
                                input_channel_base
                                + input_row
                                * input_width;

                            const int weight_row_base =
                                weight_channel_base
                                + kernel_row
                                * kernel_width;

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

                                accumulated_value +=
                                    input[
                                        input_row_base
                                        + input_column
                                    ]
                                    *
                                    weight[
                                        weight_row_base
                                        + kernel_column
                                    ];
                            }
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