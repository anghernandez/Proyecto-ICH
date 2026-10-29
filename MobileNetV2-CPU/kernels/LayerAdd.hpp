#ifndef LAYER_ADD_HPP
#define LAYER_ADD_HPP

void layer_add_forward(
    const float* input1,
    const float* input2,
    float* output,
    int size,
    float alpha
);

#endif