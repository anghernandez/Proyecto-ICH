#include "ReLU6.hpp"

void relu6_forward(
    const float* input,
    float* output,
    int size
)
{
    for (
        int index = 0;
        index < size;
        index++
    ) {
        const float input_value =
            input[index];

        float output_value;

        if (input_value < 0.0f) {
            output_value = 0.0f;
        }
        else if (input_value > 6.0f) {
            output_value = 6.0f;
        }
        else {
            output_value = input_value;
        }

        output[index] = output_value;
    }
}