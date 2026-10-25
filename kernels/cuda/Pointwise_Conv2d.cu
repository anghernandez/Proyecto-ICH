// ============================================================
// Pointwise Conv2d (1x1) en GPU.
//
// Misma función que kernels/Pointwise_Conv2d.cpp: recibe y
// devuelve arreglos de CPU. Internamente copia a la GPU, ejecuta
// el kernel y copia el resultado de vuelta. Cada fase queda
// registrada con cudaEvent (GPU_MARK) cuando se compila con
// perfilado.
// ============================================================

#include "Pointwise_Conv2d.hpp"

#include "conv_common.cuh"
#include "conv_math.cuh"


__global__ void pointwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int input_channels,
    int output_channels,
    int pixels,
    bool use_bias
)
{
    // x: píxel, y: canal de salida, z: elemento del lote.
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;

    pointwise_conv2d_thread(
        input, weight, bias, output,
        blockIdx.z, blockIdx.y, pixel,
        input_channels, output_channels, pixels, use_bias
    );
}


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
    const int pixels = input_height * input_width;

    const std::size_t input_count =
        static_cast<std::size_t>(batch_size) * input_channels * pixels;
    const std::size_t weight_count =
        static_cast<std::size_t>(output_channels) * input_channels;
    const std::size_t output_count =
        static_cast<std::size_t>(batch_size) * output_channels * pixels;

    float* device_input = gpu_buffer(GPU_BUF_ENTRADA, input_count);
    float* device_weight = gpu_buffer(GPU_BUF_PESOS, weight_count);
    float* device_bias =
        use_bias ? gpu_buffer(GPU_BUF_SESGO, output_channels) : nullptr;
    float* device_output = gpu_buffer(GPU_BUF_SALIDA, output_count);

    GPU_PROFILE(perfil, "pointwise_conv",
                forma_nchw(batch_size, input_channels,
                           input_height, input_width));

    copiar_a_gpu(device_weight, weight, weight_count);
    if (use_bias) {
        copiar_a_gpu(device_bias, bias, output_channels);
    }
    GPU_MARK(perfil, "h2d_pesos");

    copiar_a_gpu(device_input, input, input_count);
    GPU_MARK(perfil, "h2d_entrada");

    const dim3 block(HILOS_POR_BLOQUE);
    const dim3 grid(bloques_para(pixels), output_channels, batch_size);

    pointwise_conv2d_kernel<<<grid, block>>>(
        device_input, device_weight, device_bias, device_output,
        input_channels, output_channels, pixels, use_bias
    );
    CUDA_CHECK(cudaGetLastError());
    GPU_MARK(perfil, "kernel");

    copiar_a_cpu(output, device_output, output_count);
    GPU_MARK(perfil, "d2h");
}
