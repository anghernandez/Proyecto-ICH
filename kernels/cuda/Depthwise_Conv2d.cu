// ============================================================
// Depthwise Conv2d en GPU.
//
// Misma función que kernels/Depthwise_Conv2d.cpp. Un hilo por
// elemento de salida; fases registradas con cudaEvent.
// ============================================================

#include "Depthwise_Conv2d.hpp"

#include "conv_common.cuh"
#include "conv_math.cuh"


__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    depthwise_conv2d_thread(
        input, weight, bias, output, index,
        batch_size, channels, input_height, input_width,
        output_height, output_width, kernel_height, kernel_width,
        stride_height, stride_width, padding_height, padding_width,
        use_bias
    );
}


void depthwise_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
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
    const std::size_t input_count =
        static_cast<std::size_t>(batch_size) * channels
        * input_height * input_width;
    const std::size_t weight_count =
        static_cast<std::size_t>(channels) * kernel_height * kernel_width;
    const std::size_t output_count =
        static_cast<std::size_t>(batch_size) * channels
        * output_height * output_width;

    float* device_input = gpu_buffer(GPU_BUF_ENTRADA, input_count);
    float* device_weight = gpu_buffer(GPU_BUF_PESOS, weight_count);
    float* device_bias =
        use_bias ? gpu_buffer(GPU_BUF_SESGO, channels) : nullptr;
    float* device_output = gpu_buffer(GPU_BUF_SALIDA, output_count);

    GPU_PROFILE(perfil, "depthwise_conv",
                forma_nchw(batch_size, channels,
                           input_height, input_width));

    copiar_a_gpu(device_weight, weight, weight_count);
    if (use_bias) {
        copiar_a_gpu(device_bias, bias, channels);
    }
    GPU_MARK(perfil, "h2d_pesos");

    copiar_a_gpu(device_input, input, input_count);
    GPU_MARK(perfil, "h2d_entrada");

    depthwise_conv2d_kernel<<<bloques_para(output_count),
                              HILOS_POR_BLOQUE>>>(
        device_input, device_weight, device_bias, device_output,
        batch_size, channels, input_height, input_width,
        output_height, output_width, kernel_height, kernel_width,
        stride_height, stride_width, padding_height, padding_width,
        use_bias
    );
    CUDA_CHECK(cudaGetLastError());
    GPU_MARK(perfil, "kernel");

    copiar_a_cpu(output, device_output, output_count);
    GPU_MARK(perfil, "d2h");
}
