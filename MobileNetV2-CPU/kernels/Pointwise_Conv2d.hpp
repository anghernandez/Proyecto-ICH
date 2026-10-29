#ifndef POINTWISE_CONV2D_HPP
#define POINTWISE_CONV2D_HPP

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
);

#endif