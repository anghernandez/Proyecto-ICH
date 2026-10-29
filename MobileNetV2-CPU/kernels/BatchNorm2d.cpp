#include "BatchNorm2d.hpp"

#include <cmath>

void batchnorm2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    float eps,
    bool affine,
    bool use_bias
)
{
    for (
        int batch_index = 0;
        batch_index < batch_size;
        batch_index++
    ) {
        for (
            int channel_index = 0;
            channel_index < channels;
            channel_index++
        ) {
            const float mean =
                running_mean[channel_index];

            const float variance =
                running_var[channel_index];

            const float denominator =
                std::sqrt(variance + eps);

            float gamma = 1.0f;
            float beta = 0.0f;

            if (affine) {
                gamma =
                    weight[channel_index];

                if (use_bias) {
                    beta =
                        bias[channel_index];
                }
            }

            for (
                int row_index = 0;
                row_index < input_height;
                row_index++
            ) {
                for (
                    int column_index = 0;
                    column_index < input_width;
                    column_index++
                ) {
                    const int position =
                        (
                            (
                                (
                                    batch_index
                                    * channels
                                    + channel_index
                                )
                                * input_height
                                + row_index
                            )
                            * input_width
                            + column_index
                        );

                    const float input_value =
                        input[position];

                    const float normalized_value =
                        (
                            input_value
                            - mean
                        )
                        / denominator;

                    output[position] =
                        gamma
                        * normalized_value
                        + beta;
                }
            }
        }
    }
}