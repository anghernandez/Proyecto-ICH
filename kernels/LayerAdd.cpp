#include "LayerAdd.hpp"

void layer_add_forward(
    const float* input1,
    const float* input2,
    float* output,
    int size,
    float alpha
)
{
    for (
        int index = 0;
        index < size;
        index++
    ) {
        const float input1_value =
            input1[index];

        const float input2_value =
            input2[index];

        output[index] =
            input1_value
            + alpha * input2_value;
    }
}