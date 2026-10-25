#pragma once

// ============================================================
// Expone a Python el perfilado GPU de GpuProfiler.hpp.
//
// Uso en bindings/pybind_module.cpp, dentro de
// PYBIND11_MODULE(cpp_kernels, module):
//
//     #include "GpuProfilerBindings.hpp"
//     ...
//     register_gpu_profiler(module);
//
// Funciones nuevas en cpp_kernels:
//   cuda_init()                     crea el contexto CUDA (medible aparte)
//   set_gpu_profile_enabled(bool)
//   get_and_clear_gpu_records()     -> [{etapa, forma_tensor, fase, tiempo_ms}]
//   clear_gpu_records()
//   gpu_memory()                    -> dict con memoria propia y del driver
//   reset_gpu_memory_peak()
//   cuda_profiler_start() / cuda_profiler_stop()   (para nvprof)
// ============================================================

#include <pybind11/pybind11.h>

#include "GpuProfiler.hpp"

inline void register_gpu_profiler(pybind11::module& module)
{
    namespace py = pybind11;

    module.def(
        "cuda_init",
        []() {
            // cudaFree(0) fuerza la creación del contexto.
            CUDA_CHECK(cudaFree(0));
        },
        "Crea el contexto CUDA para medir su costo por separado"
    );

    module.def(
        "set_gpu_profile_enabled",
        [](bool enabled) {
            gpu_profiler_enabled() = enabled;
        },
        py::arg("enabled")
    );

    module.def(
        "get_and_clear_gpu_records",
        []() {
            py::list result;

            for (const auto& record : gpu_profiler_records()) {
                py::dict entry;
                entry["etapa"] = record.etapa;
                entry["forma_tensor"] = record.forma_tensor;
                entry["fase"] = record.fase;
                entry["tiempo_ms"] = record.tiempo_ms;
                result.append(entry);
            }

            gpu_profiler_records().clear();
            return result;
        },
        "Desglose h2d/kernel/d2h medido con cudaEvent"
    );

    module.def(
        "clear_gpu_records",
        []() {
            gpu_profiler_records().clear();
        }
    );

    module.def(
        "gpu_memory",
        []() {
            const double mib = 1024.0 * 1024.0;

            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            gpu_device_memory(free_bytes, total_bytes);

            const GpuMemoryStats& stats = gpu_memory_stats();

            py::dict result;
            result["propia_actual_mib"] = stats.actual_bytes / mib;
            result["propia_pico_mib"] = stats.pico_bytes / mib;
            result["reservas"] = stats.reservas;
            result["driver_libre_mib"] = free_bytes / mib;
            result["driver_total_mib"] = total_bytes / mib;
            result["driver_usada_mib"] =
                (total_bytes - free_bytes) / mib;
            return result;
        },
        "Memoria reservada por los kernels y cudaMemGetInfo"
    );

    module.def(
        "reset_gpu_memory_peak",
        []() {
            gpu_reset_memory_peak();
        }
    );

    module.def(
        "cuda_profiler_start",
        []() {
            CUDA_CHECK(cudaProfilerStart());
        },
        "Inicio de la región que captura nvprof --profile-from-start off"
    );

    module.def(
        "cuda_profiler_stop",
        []() {
            CUDA_CHECK(cudaProfilerStop());
        }
    );
}
