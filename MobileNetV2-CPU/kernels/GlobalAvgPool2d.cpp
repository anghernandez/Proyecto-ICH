#include "GlobalAvgPool2d.hpp"

void global_avgpool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width
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
            int channel_index = 0;
            channel_index < channels;
            channel_index++
        ) {
            float accumulated_value = 0.0f;

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
                    const int input_position =
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

                    accumulated_value +=
                        input[input_position];
                }
            }

            const float average_value =
                accumulated_value
                / static_cast<float>(spatial_size);

            const int output_position =
                batch_index * channels
                + channel_index;

            output[output_position] =
                average_value;
        }
    }
}