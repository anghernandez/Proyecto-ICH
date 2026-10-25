#pragma once

// ============================================================
// Buffers persistentes en GPU para las convoluciones CUDA.
//
// Cada llamada necesita entrada, pesos, sesgo y salida en la
// GPU. En lugar de reservar y liberar en cada capa (cudaMalloc
// y cudaFree son lentos y sincronizan), se reutilizan 4 buffers
// que solo crecen cuando una capa necesita más espacio. Después
// del calentamiento ya no hay reservas: el perfilado lo confirma
// con gpu_memory()["reservas"] == 0 durante la medición.
//
// Los datos se siguen copiando en cada llamada (pesos incluidos),
// porque la interfaz recibe arreglos de CPU. Mantener los pesos
// y las activaciones en la GPU es trabajo futuro.
// ============================================================

#include <cstddef>

#include "GpuProfiler.hpp"

enum GpuBufferSlot
{
    GPU_BUF_ENTRADA = 0,
    GPU_BUF_PESOS,
    GPU_BUF_SESGO,
    GPU_BUF_SALIDA,
    GPU_BUF_TOTAL
};

struct GpuBuffer
{
    float* pointer = nullptr;
    std::size_t capacity = 0;   // en número de floats
};

inline GpuBuffer* gpu_buffers()
{
    static GpuBuffer buffers[GPU_BUF_TOTAL];
    return buffers;
}

// Devuelve un buffer con espacio para `count` floats.
inline float* gpu_buffer(int slot, std::size_t count)
{
    GpuBuffer& buffer = gpu_buffers()[slot];

    if (count > buffer.capacity) {
        gpu_free(buffer.pointer);
        buffer.pointer = nullptr;
        buffer.capacity = 0;

        buffer.pointer = static_cast<float*>(
            gpu_malloc(count * sizeof(float))
        );
        buffer.capacity = count;
    }

    return buffer.pointer;
}
