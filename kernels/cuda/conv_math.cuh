#pragma once

// ============================================================
// Lo que calcula cada hilo de las convoluciones CUDA.
//
// Se separa del kernel (__global__) para poder ejecutarlo
// también en CPU y compararlo contra los kernels originales
// (ver tests/test_conv_math.cpp). Las fórmulas e índices son
// los mismos de kernels/*.cpp: formato NCHW, pesos OIHW.
//
// Cada función recibe las coordenadas de UNA salida y escribe
// ese único valor; los kernels .cu solo traducen
// blockIdx/threadIdx a esas coordenadas.
// ============================================================

#include <cstddef>

#if !defined(__CUDACC__)
    #ifndef __host__
        #define __host__
    #endif
    #ifndef __device__
        #define __device__
    #endif
#endif


// ------------------------------------------------------------
// Pointwise (1x1): un hilo por (lote, canal de salida, píxel).
// Hilos consecutivos -> píxeles consecutivos, así la lectura de
// la entrada queda alineada (coalescente) y el peso es el mismo
// para todo el warp.
// ------------------------------------------------------------

__host__ __device__ inline void pointwise_conv2d_thread(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_index,
    int output_channel,
    int pixel,
    int input_channels,
    int output_channels,
    int pixels,          // input_height * input_width
    bool use_bias
)
{
    if (pixel >= pixels) {
        return;
    }

    float accumulated_value = use_bias ? bias[output_channel] : 0.0f;

    const float* input_pixel =
        input
        + static_cast<std::size_t>(batch_index) * input_channels * pixels
        + pixel;

    const float* weight_row =
        weight + static_cast<std::size_t>(output_channel) * input_channels;

    for (int input_channel = 0;
         input_channel < input_channels;
         input_channel++) {
        accumulated_value +=
            input_pixel[static_cast<std::size_t>(input_channel) * pixels]
            * weight_row[input_channel];
    }

    output[
        (static_cast<std::size_t>(batch_index) * output_channels
         + output_channel) * pixels
        + pixel
    ] = accumulated_value;
}


// ------------------------------------------------------------
// Depthwise: un hilo por elemento de salida (índice lineal
// sobre lote, canal, fila, columna).
// ------------------------------------------------------------

__host__ __device__ inline void depthwise_conv2d_thread(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    std::size_t index,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    const std::size_t total =
        static_cast<std::size_t>(batch_size) * channels
        * output_height * output_width;

    if (index >= total) {
        return;
    }

    const int output_column = static_cast<int>(index % output_width);
    const int output_row =
        static_cast<int>((index / output_width) % output_height);
    const int channel = static_cast<int>(
        (index / (static_cast<std::size_t>(output_width) * output_height))
        % channels);
    const int batch_index = static_cast<int>(
        index / (static_cast<std::size_t>(output_width) * output_height
                 * channels));

    float accumulated_value = use_bias ? bias[channel] : 0.0f;

    const float* input_channel =
        input
        + (static_cast<std::size_t>(batch_index) * channels + channel)
          * input_height * input_width;

    const float* weight_channel =
        weight
        + static_cast<std::size_t>(channel) * kernel_height * kernel_width;

    for (int kernel_row = 0; kernel_row < kernel_height; kernel_row++) {
        const int input_row =
            output_row * stride_height + kernel_row - padding_height;

        if (input_row < 0 || input_row >= input_height) {
            continue;
        }

        for (int kernel_column = 0;
             kernel_column < kernel_width;
             kernel_column++) {
            const int input_column =
                output_column * stride_width + kernel_column - padding_width;

            if (input_column < 0 || input_column >= input_width) {
                continue;
            }

            accumulated_value +=
                input_channel[input_row * input_width + input_column]
                * weight_channel[kernel_row * kernel_width + kernel_column];
        }
    }

    output[index] = accumulated_value;
}


// ------------------------------------------------------------
// Conv2d general (groups = 1): un hilo por elemento de salida.
// En MobileNetV2 solo es la primera capa (3 -> 32, 3x3, s=2).
// ------------------------------------------------------------

__host__ __device__ inline void conv2d_thread(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    std::size_t index,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    const std::size_t total =
        static_cast<std::size_t>(batch_size) * output_channels
        * output_height * output_width;

    if (index >= total) {
        return;
    }

    const int output_column = static_cast<int>(index % output_width);
    const int output_row =
        static_cast<int>((index / output_width) % output_height);
    const int output_channel = static_cast<int>(
        (index / (static_cast<std::size_t>(output_width) * output_height))
        % output_channels);
    const int batch_index = static_cast<int>(
        index / (static_cast<std::size_t>(output_width) * output_height
                 * output_channels));

    float accumulated_value = use_bias ? bias[output_channel] : 0.0f;

    for (int input_channel = 0;
         input_channel < input_channels;
         input_channel++) {
        const float* input_plane =
            input
            + (static_cast<std::size_t>(batch_index) * input_channels
               + input_channel) * input_height * input_width;

        const float* weight_plane =
            weight
            + (static_cast<std::size_t>(output_channel) * input_channels
               + input_channel) * kernel_height * kernel_width;

        for (int kernel_row = 0; kernel_row < kernel_height; kernel_row++) {
            const int input_row =
                output_row * stride_height + kernel_row - padding_height;

            if (input_row < 0 || input_row >= input_height) {
                continue;
            }

            for (int kernel_column = 0;
                 kernel_column < kernel_width;
                 kernel_column++) {
                const int input_column =
                    output_column * stride_width + kernel_column
                    - padding_width;

                if (input_column < 0 || input_column >= input_width) {
                    continue;
                }

                accumulated_value +=
                    input_plane[input_row * input_width + input_column]
                    * weight_plane[kernel_row * kernel_width
                                   + kernel_column];
            }
        }
    }

    output[index] = accumulated_value;
}
