# MobileNetV2 C++/CUDA — Prototipo en Acelerador I (Jetson Nano)

Esta rama contiene la **descarga a la GPU** de las operaciones convolucionales
de la implementación manual de MobileNetV2 y el **perfilado** de la aplicación
en la Jetson Nano, con y sin el acelerador.

Las operaciones aceleradas con CUDA son:

- Pointwise Conv2D
- Depthwise Conv2D
- Conv2D

El resto de operaciones (BatchNorm2D, ReLU6, LayerAdd, Global Average Pooling
y Linear) se mantiene en CPU. La arquitectura de MobileNetV2, los pesos de la
red, la imagen de entrada y el resultado esperado de la inferencia no cambian
respecto al Prototipo en C.

---

# 1. Plataforma y condiciones

## 1.1 Plataforma

| Elemento | Valor |
|---|---|
| Placa | Jetson Nano 4 GB (Tegra X1, `t210`) |
| CPU | 4 × Cortex-A57 a 1479 MHz |
| GPU | Maxwell, 128 núcleos CUDA, `sm_53`, a 921.6 MHz |
| Memoria | 4 GB LPDDR4 compartida entre CPU y GPU |
| JetPack / L4T | 4.6.1 / R32.7.1 |
| CUDA | 10.2 |
| Python | 3.6.9 |
| torch / torchvision | 1.10.0 / 0.11.1 (ruedas de NVIDIA, instaladas en el sistema) |
| Compilador | g++ 7.5, nvcc 10.2 |

## 1.2 Condiciones del experimento

Se mantienen las mismas condiciones del perfilado del Prototipo en C:

| Parámetro | Valor |
|---|---|
| Imagen de entrada | `test_images/imagen.jpeg` |
| Iteraciones medidas | 120 |
| Warm-up | 8 iteraciones |
| Número de hilos (CPU) | 1 |
| Preprocesamiento | `official` (resize 256, recorte central 224) |
| Perfilado interno de kernels | habilitado |
| Optimización de compilación | `-O2` |
| Modo de potencia | `nvpmodel -m 0` (MAXN) |
| Relojes | fijos con `jetson_clocks` |

Se realizan dos corridas en la misma Jetson:

- **`jetson_cpu`**: línea base, todas las operaciones en CPU.
- **`jetson_gpu`**: convoluciones en GPU, resto en CPU.

La línea base se repite en la Jetson porque su CPU (Cortex-A57) no es la
misma que la de la Kria (Cortex-A53). Así, el speedup del acelerador se mide
en la misma placa, y los resultados de la Kria quedan como referencia entre
plataformas.

La rama parte de `feature/perfilado-kria-cpu` y no de
`feature/optimizacion-kernels-v1`, porque el perfilado que identificó los
cuellos de botella es el baseline, y porque Optimization V1 resultó más lenta
en la Kria (3076.21 ms contra 2779.90 ms).

---

# 2. Herramientas de perfilado y justificación

## 2.1 Resumen

| Medida | Herramienta | Archivo |
|---|---|---|
| Tiempo por etapa del flujo | `app/profile_jetson.py` | `resumen_etapas.csv` |
| Tiempo por capa, CPU y GPU | ScopedTimer (`std::chrono`) | `resumen_kernels.csv` |
| Desglose GPU por capa (copias y cómputo) | cudaEvent (`kernels/GpuProfiler.hpp`) | `resumen_kernels_gpu.csv` |
| Verificación independiente | nvprof | `nvprof_gpu_trace.csv` |
| Uso y frecuencia de GPU | sysfs, cada 20 ms (`app/jetson_monitor.py`) | `monitoreo.csv` |
| Potencia por riel y energía | sensor INA3221, cada 20 ms | `monitoreo.csv`, `resumen_energia.csv` |
| Memoria del proceso | RSS (`/proc/self/statm`) | `resumen_etapas.csv` |
| Memoria GPU propia | contador de reservas (`gpu_malloc`) | `resumen_etapas.csv` |
| Costo del contexto CUDA | `cuda_init()` | `inicializacion.json` |

## 2.2 Tiempo por etapa: `profile_jetson.py`

Se usa como base `app/profile_complete.py`, el mismo script del Prototipo en
C, para que las etapas (lectura, preprocesamiento, inferencia,
posprocesamiento y flujo total), las columnas y las estadísticas (media,
mediana, desviación, p95, mínimo y máximo) sean directamente comparables.

`profile_complete.py` no corre en la Jetson: usa sintaxis de Python 3.9/3.10 y
la API de pesos de torchvision 0.13 o superior, mientras que JetPack 4.6 trae
Python 3.6 y torchvision 0.11. Por eso se creó `profile_jetson.py`, que
conserva el mismo procedimiento y agrega:

- marcas de tiempo de cada etapa (`t_inicio_ns`, `t_fin_ns`) para cruzarlas
  con el monitor de potencia;
- el desglose GPU, la memoria GPU y el costo del contexto CUDA, cuando el
  módulo se compila con CUDA;
- la región de captura para nvprof.

Con el módulo solo CPU funciona igual, por lo que la misma herramienta mide
las dos corridas.

## 2.3 Tiempo por capa: ScopedTimer

El ScopedTimer ya estaba integrado en los bindings desde el Prototipo en C y
mide **por llamada** a cada operación. En la versión GPU envuelve la llamada
completa: copia a la GPU, kernel, copia de vuelta y sincronización.

Es la medida que se usa para comparar CPU contra GPU, porque representa el
tiempo real que ve la aplicación por capa. Se usa la misma instrumentación en
las dos versiones.

## 2.4 Desglose dentro de la capa: cudaEvent

cudaEvent registra marcas de tiempo en la propia GPU, con resolución de
aproximadamente medio microsegundo. Se usa como **complemento** del
ScopedTimer, no como reemplazo, para separar cada llamada en fases:

| Fase | Contenido |
|---|---|
| `h2d_pesos` | Copia de pesos (y sesgo) de CPU a GPU |
| `h2d_entrada` | Copia del tensor de entrada de CPU a GPU |
| `kernel` | Ejecución del kernel |
| `d2h` | Copia de la salida de GPU a CPU |

Este desglose es el que permite evaluar el criterio de descarga: cuánto del
tiempo de una capa en GPU es cómputo y cuánto es movimiento de datos.

Limitaciones:

- No ve costos del lado de la CPU fuera de la secuencia de la GPU (lanzamiento
  y paso por pybind). Se obtienen como ScopedTimer − suma de fases.
- Con memoria paginable, el intervalo de una copia incluye la copia intermedia
  que hace el driver en CPU (ver sección 5.5).

Los eventos se toman de un conjunto reutilizable y se sincroniza una sola vez
por llamada.

## 2.5 Verificación independiente: nvprof

En la Jetson Nano (Maxwell, `sm_53`, CUDA 10.2) nvprof es la herramienta de
perfilado soportada: Nsight Compute no soporta GPUs Maxwell. Se usa para
verificar, con una herramienta de NVIDIA ajena al código, los tiempos de
kernel y de copia.

- Se ejecuta en una corrida aparte, porque agrega sobrecarga; los tiempos de
  esa corrida no se usan como medida.
- Se ejecuta con ScopedTimer y cudaEvent desactivados (`--kernel-profile off`)
  y sin monitor, para que solo mida nvprof.
- Captura solo las 120 iteraciones medidas (`--profile-from-start off` con
  `cudaProfilerStart/Stop`).
- En Jetson requiere root (`ERR_NVGPUCTRPERM`) y `--unified-memory-profiling off`.

## 2.6 Uso de la GPU: sysfs

`nvidia-smi` no existe en Jetson. `tegrastats` y `jtop` leen la carga de la GPU
de `/sys/devices/gpu.0/load`; aquí se lee directamente esa misma fuente cada
20 ms, junto con la frecuencia de la GPU, la RAM del sistema y el uso de CPU.
Esto da más resolución que `tegrastats`/`jtop` y permite asociar cada muestra
a una etapa. nvprof aporta una segunda medida: la fracción del tiempo en que la
GPU está activa.

## 2.7 Potencia y energía: sensor INA3221

La Jetson Nano trae soldado en el módulo un sensor INA3221 de tres canales,
expuesto por el sistema en
`/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/`:

| Riel | Mide |
|---|---|
| `POM_5V_IN` | Potencia total de entrada al módulo |
| `POM_5V_GPU` | Riel de la GPU |
| `POM_5V_CPU` | Riel de la CPU |

Se eligió porque separa la potencia de GPU y CPU, se puede leer desde código
con marca de tiempo y no requiere hardware adicional. La energía de cada etapa
se obtiene integrando la potencia en su ventana de tiempo, y la energía
dinámica resta la potencia en reposo medida 10 s antes de la corrida.

Limitaciones:

- Mide el módulo, no la placa portadora ni los periféricos.
- Precisión de pocos porcentajes y actualización no instantánea; las etapas
  más cortas que el período de muestreo (20 ms) tienen energía estimada.
- Los archivos solo son legibles por root; se habilitan con `chmod` en cada
  arranque (sección 4.2).
- Se leen solo los archivos `in_powerN_input` (medición continua); los
  `in_powerN_trigger_input` disparan una medición aparte y se descartan.

Un medidor externo daría el consumo desde el enchufe, pero sin separar CPU y
GPU ni sincronizarse con las etapas.

## 2.8 Memoria

| Medida | Qué representa |
|---|---|
| RSS del proceso | Memoria residente, igual que en el Prototipo en C |
| `gpu_propia_pico_mib` | Memoria reservada por los kernels (contador propio de `gpu_malloc`/`gpu_free`) |
| MemAvailable antes y después de `cuda_init()` | Costo de crear el contexto CUDA |
| `cudaMemGetInfo` | Se registra, pero en Jetson describe la memoria de todo el sistema, incluida la caché del sistema operativo; no se interpreta como memoria de la GPU |

CPU y GPU comparten la misma DRAM, por lo que no existe una "memoria de la
GPU" separada; la huella propia de la aceleración es el contador de reservas.

## 2.9 Monitor en un proceso separado

Las funciones de pybind11 retienen el GIL de Python mientras ejecutan C++. Un
hilo de muestreo dentro del mismo proceso quedaría congelado durante cada capa,
así que el monitor corre como proceso aparte, en otro núcleo (la inferencia usa
un hilo). Ambos usan el reloj `CLOCK_MONOTONIC`, por lo que las muestras se
alinean con las marcas de tiempo de las etapas. El intervalo obtenido fue de
20.0 ms en promedio, con un máximo de 21.2 ms.

## 2.10 Relojes fijos

Con `nvpmodel -m 0` y `jetson_clocks`, la CPU queda en 1479 MHz y la GPU en
921.6 MHz (mínimo igual al máximo). Esto evita que el escalado dinámico de
frecuencia cambie los tiempos entre corridas. `metadata.json` registra el modo
y las frecuencias de cada corrida.

---

# 3. Implementación en GPU

## 3.1 Decisiones que motivaron la aceleración

Los perfilados del Prototipo en C identificaron a las convoluciones como el
principal cuello de botella:

| Plataforma | Pointwise | Depthwise | Conv2D | Total convoluciones |
|---|---:|---:|---:|---:|
| Kria KV260 (Cortex-A53) | 80.1 % | 7.9 % | 4.2 % | **92.2 %** |
| Jetson Nano (Cortex-A57) | 76.7 % | 11.0 % | 5.7 % | **93.4 %** |

*Porcentaje del tiempo acumulado de los kernels perfilados.*

Las tres operaciones superan con amplitud el 50 % del tiempo de ejecución
requerido para este avance; Pointwise Conv2D por sí sola representa más del
75 %. Las demás operaciones suman 6.6 % en la Jetson y se mantienen en CPU para
limitar el alcance de esta etapa.

Según la ley de Amdahl, si las convoluciones tardaran cero, el speedup máximo
de la inferencia en la Jetson sería:

```text
Speedup máximo = 1 / (1 − 0.934) ≈ 15.2×
```

## 3.2 Diseño

- **Mismas funciones que la versión CPU.** `kernels/cuda/*.cu` implementan
  `pointwise_conv2d_forward`, `depthwise_conv2d_forward` y `conv2d_forward`
  con la misma firma que `kernels/*.cpp`. El modelo en Python y los wrappers
  no cambian; al compilar se elige qué implementación enlazar.
- **Asignación de hilos.**
  - Pointwise: un hilo por (lote, canal de salida, píxel). Hilos consecutivos
    leen píxeles consecutivos (acceso coalescente) y comparten el mismo peso.
  - Depthwise y Conv2D: un hilo por elemento de salida.
  - 256 hilos por bloque.
- **Buffers persistentes.** Entrada, pesos, sesgo y salida usan cuatro
  buffers en GPU que solo crecen cuando una capa necesita más espacio
  (`kernels/GpuBuffers.hpp`). Después del calentamiento no hay más reservas.
- **Copias en cada llamada.** La interfaz recibe arreglos de CPU, por lo que
  pesos y entrada se copian a la GPU y la salida se copia de vuelta en cada
  capa. Este costo queda medido por separado.
- **Sin optimizar.** Cada hilo calcula una salida directamente, sin memoria
  compartida ni fusión de operaciones, como se permite en este avance.

## 3.3 Archivos

| Archivo | Cambio |
|---|---|
| `kernels/cuda/Pointwise_Conv2d.cu` | Nuevo: Pointwise Conv2D en GPU |
| `kernels/cuda/Depthwise_Conv2d.cu` | Nuevo: Depthwise Conv2D en GPU |
| `kernels/cuda/Conv2d.cu` | Nuevo: Conv2D en GPU |
| `kernels/cuda/conv_math.cuh` | Nuevo: cálculo de cada hilo (compartido con la prueba en CPU) |
| `kernels/cuda/conv_common.cuh` | Nuevo: copias, tamaño de bloque y etiqueta de forma |
| `kernels/GpuBuffers.hpp` | Nuevo: buffers persistentes en GPU |
| `kernels/GpuProfiler.hpp` | Nuevo: cudaEvent por fase y contador de memoria GPU |
| `bindings/GpuProfilerBindings.hpp` | Nuevo: funciones de perfilado GPU para Python |
| `bindings/pybind_module.cpp` | Registra el perfilado GPU con `USE_CUDA` |
| `Makefile` | Objetivos `gpu-build` y `profile-build-gpu` |
| `app/profile_jetson.py` | Nuevo: perfilado en Jetson (Python 3.6) |
| `app/jetson_monitor.py` | Nuevo: monitor de GPU, potencia y RAM |
| `app/comparar_resultados.py` | Nuevo: comparación por etapa, capa y tipo |
| `app/imagenet_classes.txt` | Nuevo: nombres de clases (torchvision 0.11 no los incluye) |
| `tests/test_conv_math.cpp` | Nuevo: verificación de los kernels sin GPU |
| `wrappers/Conv2d.py` | Anotaciones de tipo compatibles con Python 3.6 |

## 3.4 Verificación de exactitud

1. **Sin GPU.** `tests/test_conv_math.cpp` ejecuta en CPU el cálculo de cada
   hilo, recorriendo la misma grilla de bloques e hilos que lanzan los `.cu`,
   y lo compara con los kernels CPU originales en las 52 convoluciones de
   MobileNetV2. La diferencia máxima es **0**, porque las sumas se hacen en el
   mismo orden. Con un error introducido a propósito, la prueba lo detecta.
2. **En la Jetson.** `profile_jetson.py` valida la salida C++/CUDA contra
   PyTorch antes de medir: `PASS`, RMSE = 3.07 × 10⁻⁶ (igual que la versión
   CPU).
3. **Durante la medición.** Las 120 predicciones dan la clase 239 (*Bernese
   mountain dog*), igual que en la Kria.

---

# 4. Reproducibilidad

## 4.1 Entrar al proyecto

```bash
cd Proyecto-ICH
```

torch viene instalado en el sistema (ruedas de NVIDIA), por lo que no se usa
el `venv` del Makefile; se compila con el Python del sistema. La primera vez
se instala pybind11:

```bash
pip3 install --user pybind11
```

---

## 4.2 Preparar la Jetson

Estos pasos se pierden al reiniciar la Jetson y deben repetirse en cada
arranque.

Habilitar la lectura de los sensores de potencia:

```bash
sudo chmod a+r /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/rail_name_* /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power*_input
```

Fijar el modo de máxima potencia y los relojes:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Comprobar que el monitor encuentra los tres rieles:

```bash
python3 app/jetson_monitor.py --fuentes
```

---

## 4.3 Línea base en CPU

Compilar:

```bash
make clean
make profile-build PYTHON=python3
```

Ejecutar el perfilado:

```bash
PYTHONPATH="$PWD" python3 app/profile_jetson.py \
    --images test_images/imagen.jpeg \
    --iterations 120 \
    --warmup 8 \
    --threads 1 \
    --preprocess official \
    --kernel-profile on \
    --output-dir resultados/jetson_cpu
```

---

## 4.4 Con aceleración en GPU

Compilar:

```bash
make clean
make profile-build-gpu PYTHON=python3
```

Ejecutar el perfilado con los mismos parámetros:

```bash
PYTHONPATH="$PWD" python3 app/profile_jetson.py \
    --images test_images/imagen.jpeg \
    --iterations 120 \
    --warmup 8 \
    --threads 1 \
    --preprocess official \
    --kernel-profile on \
    --output-dir resultados/jetson_gpu
```

En ambas corridas la validación debe terminar en `PASS`; si falla, el
perfilado se detiene.

---

## 4.5 Verificación con nvprof

Requiere root. `python3 -B` evita archivos de caché de root dentro del repo, y
la ruta de pesos es explícita porque root tiene otra carpeta personal:

```bash
sudo env PYTHONPATH="$HOME/.local/lib/python3.6/site-packages" \
    /usr/local/cuda/bin/nvprof \
    --profile-from-start off \
    --unified-memory-profiling off \
    --print-gpu-trace --csv \
    --log-file resultados/nvprof_gpu_trace.csv \
    python3 -B app/profile_jetson.py \
    --images test_images/imagen.jpeg \
    --iterations 120 --warmup 8 --threads 1 \
    --preprocess official \
    --kernel-profile off --monitor off --nvprof-region on \
    --weights-file ~/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth \
    --output-dir resultados/jetson_gpu_nvprof
```

Devolver los archivos al usuario:

```bash
sudo chown -R $USER:$USER resultados/jetson_gpu_nvprof resultados/nvprof_gpu_trace.csv
```

---

## 4.6 Comparar resultados

```bash
python3 app/comparar_resultados.py \
    --base resultados/jetson_cpu \
    --nuevo resultados/jetson_gpu \
    --out resultados/comparacion_jetson
```

Genera la comparación por etapa, por capa y por tipo de operación, junto con
la cota de Amdahl.

---

## 4.7 Verificar los kernels sin GPU

```bash
g++ -O2 -std=c++14 -Ikernels -Ikernels/cuda tests/test_conv_math.cpp \
    kernels/Conv2d.cpp kernels/Pointwise_Conv2d.cpp kernels/Depthwise_Conv2d.cpp \
    -o test_conv_math && ./test_conv_math
```

Resultado esperado: `Fallas: 0`.

---

## 4.8 Resultados generados

Además de los archivos del Prototipo en C (`etapas.csv`, `kernels.csv`,
`resumen_etapas.csv`, `resumen_kernels.csv`, `validacion.json`,
`metadata.json`, etc.), cada corrida genera:

| Archivo | Contenido |
|---|---|
| `monitoreo.csv` | Muestras cada 20 ms: carga y frecuencia de GPU, RAM, CPU y potencia por riel |
| `energia.csv` / `resumen_energia.csv` | Energía, potencia media y uso de GPU por etapa e iteración |
| `kernels_gpu.csv` / `resumen_kernels_gpu.csv` | Desglose h2d / kernel / d2h por llamada (solo GPU) |
| `inicializacion.json` | Importaciones, contexto CUDA y carga de pesos |

---

# 5. Resultados

## 5.1 Comparabilidad

- La huella de los pesos (`a5f431d4…`) y de la imagen (`3f1e1349…`) en
  `metadata.json` coincide con la de `resultados/kria_cpu`.
- La predicción (clase 239, probabilidad 0.4315) coincide con la de la Kria.
- Las mediciones son estables: la desviación de la inferencia es menor al
  1.2 % de la media en ambas corridas.

## 5.2 Tiempo por etapa

| Etapa | Kria CPU | Jetson CPU | Jetson GPU | Speedup GPU |
|---|---:|---:|---:|---:|
| Lectura y decodificación | 7.74 ms | 6.20 ms | 6.23 ms | — |
| Preprocesamiento | 20.91 ms | 13.48 ms | 13.48 ms | — |
| Inferencia media | 2779.90 ms | 1276.11 ms | 248.04 ms | **5.14×** |
| Inferencia mediana | 2779.11 ms | 1277.75 ms | 246.95 ms | 5.17× |
| Inferencia p95 | 2786.51 ms | 1288.37 ms | 255.26 ms | 5.05× |
| Posprocesamiento | 0.36 ms | 0.22 ms | 0.23 ms | — |
| Flujo total medio | 2811.95 ms | 1298.49 ms | 270.63 ms | **4.80×** |

```text
Speedup = Tiempo Jetson CPU / Tiempo Jetson GPU
```

Con el mismo código, la CPU de la Jetson es 2.18× más rápida que la de la Kria.
La GPU reduce la inferencia de **1276.11 ms a 248.04 ms**.

## 5.3 Tiempo por operación

Suma de las llamadas de cada tipo en una inferencia (ScopedTimer).

| Operación | Llamadas | Jetson CPU | Jetson GPU | Speedup | % en GPU |
|---|---:|---:|---:|---:|---:|
| Pointwise Conv2D | 34 | 950.14 ms | 87.65 ms | **10.84×** | 41.9 % |
| Depthwise Conv2D | 17 | 135.70 ms | 35.17 ms | **3.86×** | 16.8 % |
| Conv2D | 1 | 70.74 ms | 7.50 ms | **9.43×** | 3.6 % |
| BatchNorm2D (CPU) | 52 | 42.22 ms | 38.95 ms | — | 18.6 % |
| ReLU6 (CPU) | 35 | 35.20 ms | 35.52 ms | — | 17.0 % |
| Linear (CPU) | 1 | 3.54 ms | 3.69 ms | — | 1.8 % |
| LayerAdd y pooling (CPU) | 11 | 0.74 ms | 0.74 ms | — | 0.4 % |
| **Total kernels** | 151 | 1238.28 ms | 209.23 ms | 5.92× | 100 % |

Las tres convoluciones pasan de **1156.58 ms a 130.32 ms** (8.9×).

## 5.4 Desglose de las convoluciones en GPU

Suma por inferencia de cada fase medida con cudaEvent.

| Operación | h2d pesos | h2d entrada | kernel | d2h | Total |
|---|---:|---:|---:|---:|---:|
| Pointwise Conv2D | 6.90 ms | 7.79 ms | 59.06 ms | 10.78 ms | 84.53 ms |
| Depthwise Conv2D | 1.73 ms | 8.33 ms | 17.04 ms | 5.91 ms | 33.01 ms |
| Conv2D | 0.12 ms | 0.39 ms | 5.85 ms | 0.86 ms | 7.23 ms |
| **Total** | 8.75 ms | 16.52 ms | 81.94 ms | 17.55 ms | 124.76 ms |

Del tiempo de las convoluciones en GPU (130.32 ms según ScopedTimer):

- **63 %** es cómputo en GPU (81.94 ms);
- **33 %** es copia de datos (42.82 ms);
- **4 %** es lanzamiento y paso por pybind (5.56 ms).

## 5.5 Verificación con nvprof

| Fase (por inferencia) | cudaEvent | nvprof |
|---|---:|---:|
| Kernels | 81.94 ms | 79.61 ms |
| Copias CPU → GPU | 25.27 ms | 6.98 ms |
| Copias GPU → CPU | 17.55 ms | 3.98 ms |

- **Kernels.** Ambas herramientas coinciden con 2.9 % de diferencia en el
  total (5.4 % en promedio por capa). cudaEvent incluye el breve intervalo
  previo al inicio de cada kernel, más visible en las capas pequeñas de 7×7.
  Esto valida el desglose por capa obtenido con cudaEvent.
- **Copias.** nvprof mide solo la transferencia (11 ms en total, a unos
  8 GB/s), mientras que a la aplicación cada copia le cuesta casi cuatro veces
  más. Los arreglos de NumPy están en memoria paginable: el driver primero los
  copia con la CPU a un buffer intermedio fijo y después los transfiere. Esa
  copia intermedia explica unos 32 ms por inferencia.
- **Datos movidos.** 34.2 MB hacia la GPU y 25.5 MB de vuelta por inferencia.
- **Actividad de la GPU.** La GPU está activa unos 91 ms de los 248 ms de cada
  inferencia (37 %), en línea con la carga medida por el monitor.
- Los kernels usan entre 32 y 40 registros por hilo.

## 5.6 Uso de GPU, potencia y energía

Valores medios en la ventana de inferencia (120 iteraciones).

| Métrica | Reposo | Jetson CPU | Jetson GPU |
|---|---:|---:|---:|
| Potencia total (`POM_5V_IN`) | 1.94 W | 2.97 W | 4.00 W |
| Potencia CPU (`POM_5V_CPU`) | 0.45 W | 1.40 W | 1.02 W |
| Potencia GPU (`POM_5V_GPU`) | 0.17 W | 0.16 W | 1.21 W |
| Carga de GPU | 0 % | 0 % | 32 % |
| Energía por inferencia | — | **3.79 J** | **0.99 J** |
| Energía dinámica por inferencia | — | 1.30 J | 0.51 J |

```text
Energía dinámica = Energía − Potencia en reposo × Duración
```

La versión GPU consume más potencia (4.00 W contra 2.97 W), pero termina
5.1 veces antes, por lo que la energía por inferencia baja **3.8×**. La carga
media de la GPU es de solo 32 % (máximo 60 %): pasa la mayor parte de cada
inferencia esperando a la CPU y a las copias.

## 5.7 Memoria

| Métrica | Jetson CPU | Jetson GPU |
|---|---:|---:|
| RSS del proceso durante la inferencia | 257.6 MiB | 2168.5 MiB |
| RAM del sistema durante la inferencia | 714 MiB | 2522 MiB |
| Memoria GPU propia (buffers) | — | 10.75 MiB |
| Creación del contexto CUDA | — | 6.71 s |
| MemAvailable antes / después del contexto | — | 3290 / 1499 MiB |

La memoria propia de la aceleración es de **10.75 MiB** y se mantiene
constante. El aumento de RSS y el tiempo de creación del contexto no se deben
a los kernels sino a torch, que también está compilado con CUDA: al crear el
contexto, CUDA 10.2 carga en él todos los kernels de las bibliotecas presentes
en el proceso. Una prueba aislada lo confirma:

| Prueba | Memoria máxima del proceso |
|---|---:|
| `cuda_init()` sin importar torch | 22 MiB |
| `cuda_init()` después de importar torch | 2081 MiB |

## 5.8 Distribución del tiempo de inferencia en GPU

| Componente | Tiempo | % de la inferencia |
|---|---:|---:|
| Convoluciones en GPU (llamada completa) | 130.32 ms | 52.5 % |
| Operaciones en CPU (BatchNorm2D, ReLU6, Linear, etc.) | 78.91 ms | 31.8 % |
| Python y pybind entre capas | 38.81 ms | 15.6 % |
| **Inferencia** | 248.04 ms | 100 % |

El tiempo de Python entre capas es prácticamente el mismo que en la versión
CPU (37.8 ms), pero ahora pesa 15.6 % de la inferencia.

---

# 6. Análisis de la descarga de trabajo

El beneficio de llevar una operación a la GPU depende de cuánto cálculo hace
por cada byte que hay que copiar:

| Operación | MAC por inferencia | Datos copiados | MAC / byte | Speedup |
|---|---:|---:|---:|---:|
| Pointwise Conv2D | 267.9 M | 35.9 MB | 7.5 | 10.84× |
| Conv2D | 10.8 M | 2.2 MB | 4.9 | 9.43× |
| Depthwise Conv2D | 20.7 M | 24.4 MB | 0.8 | 3.86× |

*MAC: multiplicación-acumulación. Datos copiados: entrada, pesos y salida en
float32 con la interfaz actual.*

- **Pointwise Conv2D** es la operación más favorable: concentra la mayor parte
  del cálculo con relativamente pocos datos por operación, y obtiene el mayor
  speedup.
- **Depthwise Conv2D** hace menos de una operación por byte copiado. En GPU,
  sus copias (15.97 ms) cuestan casi lo mismo que su cómputo (17.04 ms), lo que
  limita su speedup a 3.86×.
- Como criterio de descarga, conviene enviar a la GPU las operaciones con alta
  intensidad aritmética, y para las de baja intensidad evitar las copias
  manteniendo los datos en la GPU entre capas.

El speedup obtenido (5.14×) queda lejos de la cota de Amdahl (15.2×) por tres
factores medidos: las copias en cada capa (42.82 ms), las operaciones que
siguen en CPU (78.91 ms) y el tiempo de Python entre capas (38.81 ms).


