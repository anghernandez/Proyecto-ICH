#include "Linear.hpp"

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
            float accumulated_value = 0.0f;

            for (
                int input_index = 0;
                input_index < in_features;
                input_index++
            ) {
                const int input_position =
                    sample_index * in_features
                    + input_index;

                const int weight_position =
                    output_index * in_features
                    + input_index;

                const float input_value =
                    input[input_position];

                const float weight_value =
                    weight[weight_position];

                accumulated_value += (
                    input_value * weight_value
                );
            }

            if (use_bias) {
                accumulated_value += bias[
                    output_index
                ];
            }

            const int output_position =
                sample_index * out_features
                + output_index;

            output[output_position] =
                accumulated_value;
        }
    }
}