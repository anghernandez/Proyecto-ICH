#ifndef GLOBAL_AVG_POOL2D_HPP
#define GLOBAL_AVG_POOL2D_HPP

void global_avgpool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width
);

#endif