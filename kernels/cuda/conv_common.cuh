#pragma once

// Utilidades compartidas por las convoluciones CUDA.

#include <cstddef>
#include <string>

#include "GpuProfiler.hpp"
#include "GpuBuffers.hpp"

// Hilos por bloque. 256 es un valor estándar para Maxwell
// (Jetson Nano: 1 SM, hasta 2048 hilos residentes).
constexpr unsigned int HILOS_POR_BLOQUE = 256;

inline unsigned int bloques_para(std::size_t elementos)
{
    return static_cast<unsigned int>(
        (elementos + HILOS_POR_BLOQUE - 1) / HILOS_POR_BLOQUE
    );
}

// Misma etiqueta que arma el binding para el ScopedTimer
// ("1x32x112x112"), para poder cruzar ambos registros.
inline std::string forma_nchw(int batch, int channels,
                              int height, int width)
{
    return std::to_string(batch) + "x" + std::to_string(channels) + "x"
         + std::to_string(height) + "x" + std::to_string(width);
}

inline void copiar_a_gpu(float* destination, const float* source,
                         std::size_t count)
{
    CUDA_CHECK(cudaMemcpy(destination, source, count * sizeof(float),
                          cudaMemcpyHostToDevice));
}

inline void copiar_a_cpu(float* destination, const float* source,
                         std::size_t count)
{
    CUDA_CHECK(cudaMemcpy(destination, source, count * sizeof(float),
                          cudaMemcpyDeviceToHost));
}
