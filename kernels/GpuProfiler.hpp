#pragma once

// ============================================================
// Perfilado del lado GPU para los kernels CUDA de MobileNetV2.
//
// Complementa a Profiler.hpp (ScopedTimer, std::chrono):
//
//   ScopedTimer   -> tiempo total de la llamada visto desde la
//                    CPU (H2D + kernel + D2H + lanzamiento +
//                    sincronización). Es el que se compara 1:1
//                    contra la versión CPU.
//
//   GpuCallProfile -> desglose medido en la propia GPU con
//                    cudaEvent: cuánto de esa llamada es copia
//                    host->device, cómputo y copia device->host.
//                    Sirve para el criterio de descarga de
//                    trabajo (¿compensa transferir?).
//
// Además lleva la cuenta de la memoria que reservan nuestros
// kernels (actual y pico) y expone cudaMemGetInfo.
//
// Compatible con CUDA 10.2 (JetPack 4.x) y C++11/14, para que
// pueda incluirse tanto desde g++ (bindings) como desde nvcc.
// Todo se activa con -DPROFILE_KERNELS (make profile-build),
// igual que el ScopedTimer.
// ============================================================

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>


// ------------------------------------------------------------
// Verificación de errores CUDA
// ------------------------------------------------------------

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t cuda_status_ = (call);                        \
        if (cuda_status_ != cudaSuccess) {                        \
            throw std::runtime_error(                             \
                std::string(#call) + " -> " +                     \
                cudaGetErrorString(cuda_status_));                \
        }                                                         \
    } while (0)


// ------------------------------------------------------------
// Registros del desglose por fase
// ------------------------------------------------------------

struct GpuProfileRecord
{
    std::string etapa;         // pointwise_conv, depthwise_conv, conv2d
    std::string forma_tensor;  // "1x32x112x112" (misma convención que CPU)
    std::string fase;          // h2d, kernel, d2h (o la que se marque)
    double tiempo_ms;
};

inline std::vector<GpuProfileRecord>& gpu_profiler_records()
{
    static std::vector<GpuProfileRecord> records;
    return records;
}

inline bool& gpu_profiler_enabled()
{
    static bool enabled = true;
    return enabled;
}


// ------------------------------------------------------------
// Pool de eventos: se reutilizan entre llamadas para no pagar
// cudaEventCreate/Destroy en cada capa (el host es de un hilo).
// ------------------------------------------------------------

inline std::vector<cudaEvent_t>& gpu_event_pool()
{
    static std::vector<cudaEvent_t> pool;
    return pool;
}

inline cudaEvent_t gpu_event_at(std::size_t index)
{
    std::vector<cudaEvent_t>& pool = gpu_event_pool();

    while (pool.size() <= index) {
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreate(&event));
        pool.push_back(event);
    }

    return pool[index];
}


// ------------------------------------------------------------
// GpuCallProfile: una instancia por llamada a kernel.
//
//   GPU_PROFILE(perfil, "pointwise_conv", forma);
//   cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice);
//   GPU_MARK(perfil, "h2d");
//   pointwise_kernel<<<grid, block>>>(...);
//   GPU_MARK(perfil, "kernel");
//   cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
//   GPU_MARK(perfil, "d2h");
//   // al salir del bloque: una sola sincronización y se guardan
//   // los tiempos entre marcas consecutivas.
//
// Cada marca registra un cudaEvent en el stream; el tiempo de
// una fase es el intervalo entre su marca y la anterior.
// ------------------------------------------------------------

class GpuCallProfile
{
public:
    GpuCallProfile(
        const std::string& etapa,
        const std::string& forma_tensor,
        cudaStream_t stream = 0
    )
        : etapa_(etapa),
          forma_tensor_(forma_tensor),
          stream_(stream),
          active_(gpu_profiler_enabled())
    {
        if (active_) {
            CUDA_CHECK(cudaEventRecord(gpu_event_at(0), stream_));
        }
    }

    void mark(const char* fase)
    {
        if (!active_) {
            return;
        }

        const std::size_t index = fases_.size() + 1;
        CUDA_CHECK(cudaEventRecord(gpu_event_at(index), stream_));
        fases_.push_back(fase);
    }

    ~GpuCallProfile()
    {
        if (!active_ || fases_.empty()) {
            return;
        }

        // Un destructor no debe lanzar: si algo falla aquí,
        // se descarta el registro en lugar de abortar.
        cudaEvent_t last = gpu_event_at(fases_.size());

        if (cudaEventSynchronize(last) != cudaSuccess) {
            return;
        }

        for (std::size_t i = 0; i < fases_.size(); i++) {
            float elapsed_ms = 0.0f;

            if (cudaEventElapsedTime(
                    &elapsed_ms,
                    gpu_event_at(i),
                    gpu_event_at(i + 1)) != cudaSuccess) {
                return;
            }

            GpuProfileRecord record;
            record.etapa = etapa_;
            record.forma_tensor = forma_tensor_;
            record.fase = fases_[i];
            record.tiempo_ms = static_cast<double>(elapsed_ms);

            gpu_profiler_records().push_back(record);
        }
    }

private:
    std::string etapa_;
    std::string forma_tensor_;
    cudaStream_t stream_;
    bool active_;
    std::vector<std::string> fases_;
};


#ifdef PROFILE_KERNELS
    #define GPU_PROFILE(var, etapa, forma) \
        GpuCallProfile var(etapa, forma)
    #define GPU_MARK(var, fase) \
        var.mark(fase)
#else
    #define GPU_PROFILE(var, etapa, forma) \
        do {} while (0)
    #define GPU_MARK(var, fase) \
        do {} while (0)
#endif


// ------------------------------------------------------------
// Memoria GPU reservada por nuestros kernels.
//
// En la Jetson la GPU y la CPU comparten la misma DRAM, así que
// cudaMemGetInfo describe la memoria de TODO el sistema. Por
// eso se lleva además una cuenta exacta de lo que reservamos
// con gpu_malloc/gpu_free: es la huella real de la aceleración.
// ------------------------------------------------------------

struct GpuMemoryStats
{
    std::size_t actual_bytes = 0;
    std::size_t pico_bytes = 0;
    std::size_t reservas = 0;   // número de cudaMalloc hechos
};

inline GpuMemoryStats& gpu_memory_stats()
{
    static GpuMemoryStats stats;
    return stats;
}

inline std::unordered_map<void*, std::size_t>& gpu_allocations()
{
    static std::unordered_map<void*, std::size_t> allocations;
    return allocations;
}

inline void* gpu_malloc(std::size_t bytes)
{
    void* pointer = nullptr;
    CUDA_CHECK(cudaMalloc(&pointer, bytes));

    GpuMemoryStats& stats = gpu_memory_stats();
    gpu_allocations()[pointer] = bytes;

    stats.actual_bytes += bytes;
    stats.reservas += 1;

    if (stats.actual_bytes > stats.pico_bytes) {
        stats.pico_bytes = stats.actual_bytes;
    }

    return pointer;
}

inline void gpu_free(void* pointer)
{
    if (pointer == nullptr) {
        return;
    }

    std::unordered_map<void*, std::size_t>& allocations =
        gpu_allocations();

    std::unordered_map<void*, std::size_t>::iterator found =
        allocations.find(pointer);

    if (found != allocations.end()) {
        gpu_memory_stats().actual_bytes -= found->second;
        allocations.erase(found);
    }

    CUDA_CHECK(cudaFree(pointer));
}

inline void gpu_reset_memory_peak()
{
    GpuMemoryStats& stats = gpu_memory_stats();
    stats.pico_bytes = stats.actual_bytes;
    stats.reservas = 0;
}

// Memoria libre/total según el driver (en Jetson: del sistema).
inline void gpu_device_memory(std::size_t& free_bytes,
                              std::size_t& total_bytes)
{
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
}
