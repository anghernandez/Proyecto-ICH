#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <utility>
// ============================================================
// Profiler minimalista basado en std::chrono.
//
// Cada operación registra: etapa (tipo de operación),
// forma del tensor de entrada, y tiempo en milisegundos.
//
// El buffer se acumula en un vector global y se expone a
// Python (ver pybind_module.cpp) para poder vaciarlo y
// escribirlo a CSV desde ahí, capa por capa.
// ============================================================

struct ProfileRecord
{
    std::string etapa;
    std::string forma_tensor;
    double tiempo_ms;
};

inline std::vector<ProfileRecord>& profiler_records()
{
    static std::vector<ProfileRecord> records;
    return records;
}


inline bool& profiler_enabled()
{
    static bool enabled = true;
    return enabled;
}

// RAII: mide desde que se construye hasta que se destruye
// (o sea, hasta que termina el bloque { } donde se declaró).
class ScopedTimer
{
public:
    ScopedTimer(
        std::string etapa,
        std::string forma_tensor
    )
        : etapa_(std::move(etapa)),
          forma_tensor_(std::move(forma_tensor)),
          active_(profiler_enabled()),
          start_(
              active_
                  ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{}
          )
    {
    }

    ~ScopedTimer()
    {
        if (!active_) {
            return;
        }

        const auto end = std::chrono::steady_clock::now();

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                end - start_
            ).count();

        profiler_records().push_back(
            {
                etapa_,
                forma_tensor_,
                elapsed_ms
            }
        );
    }

private:
    std::string etapa_;
    std::string forma_tensor_;
    bool active_;
    std::chrono::steady_clock::time_point start_;
};

    

// ============================================================
// Macro para (des)activar la instrumentación en tiempo de
// compilación. Con -DPROFILE_KERNELS (make build PROFILE=1)
// se crea el ScopedTimer real. Sin el flag, la macro no
// genera ningún código: el binario queda igual que si nunca
// se hubiera instrumentado. Esto es lo que permite medir el
// overhead real de la instrumentación (paso 4 de PERFILADO.md).
// ============================================================

#ifdef PROFILE_KERNELS
    #define SCOPED_TIMER(etapa, forma) \
        ScopedTimer _scoped_timer_(etapa, forma)
#else
    #define SCOPED_TIMER(etapa, forma) \
        do {} while (0)
#endif
