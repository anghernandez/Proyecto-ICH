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
    const int spatial_size =
        input_height * input_width;

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
            const int output_base =
                (
                    batch_index * output_channels
                    + output_channel
                )
                * spatial_size;

            const float bias_value =
                use_bias ? bias[output_channel] : 0.0f;

            // Inicializar el mapa de salida.
            for (
                int spatial_index = 0;
                spatial_index < spatial_size;
                spatial_index++
            ) {
                output[
                    output_base + spatial_index
                ] = bias_value;
            }

            // Acumular cada canal de entrada.
            for (
                int input_channel = 0;
                input_channel < input_channels;
                input_channel++
            ) {
                const int input_base =
                    (
                        batch_index * input_channels
                        + input_channel
                    )
                    * spatial_size;

                const float weight_value =
                    weight[
                        output_channel * input_channels
                        + input_channel
                    ];

                for (
                    int spatial_index = 0;
                    spatial_index < spatial_size;
                    spatial_index++
                ) {
                    output[
                        output_base + spatial_index
                    ] +=
                        input[
                            input_base + spatial_index
                        ]
                        * weight_value;
                }
            }
        }
    }
}