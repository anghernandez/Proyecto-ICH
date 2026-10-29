#include "Pointwise_Conv2d.hpp"

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
                output_row < input_height;
                output_row++
            ) {
                for (
                    int output_column = 0;
                    output_column < input_width;
                    output_column++
                ) {
                    float accumulated_value = 0.0f;

                    if (use_bias) {
                        accumulated_value = bias[
                            output_channel
                        ];
                    }

                    for (
                        int input_channel = 0;
                        input_channel < input_channels;
                        input_channel++
                    ) {
                        const int input_position =
                            (
                                (
                                    batch_index
                                    * input_channels
                                    + input_channel
                                )
                                * input_height
                                + output_row
                            )
                            * input_width
                            + output_column;

                        const int weight_position =
                            (
                                output_channel
                                * input_channels
                                + input_channel
                            );

                        const float input_value =
                            input[input_position];

                        const float weight_value =
                            weight[weight_position];

                        accumulated_value += (
                            input_value * weight_value
                        );
                    }

                    const int output_position =
                        (
                            (
                                batch_index
                                * output_channels
                                + output_channel
                            )
                            * input_height
                            + output_row
                        )
                        * input_width
                        + output_column;

                    output[output_position] =
                        accumulated_value;
                }
            }
        }
    }
}