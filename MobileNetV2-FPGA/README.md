# MobileNetV2 en FPGA — AMD Kria KV260

## 1. Objetivo

Evaluar el rendimiento de una implementación de **MobileNetV2** ejecutada en la **AMD Kria KV260** mediante una interfaz PyTorch–FPGA, y comparar dos versiones de los kernels HLS:

- **Sin optimizar:** versión de referencia de los kernels FPGA.
- **Optimizada:** versión con optimización del kernel **Pointwise Conv2D**.

Se analiza la latencia de inferencia observada desde el host, su variabilidad y el tiempo acumulado por tipo de operación. El protocolo de una sola imagen sigue las condiciones empleadas para las mediciones CPU/GPU del proyecto ICH.

## 2. Configuración experimental

| Parámetro | Configuración |
|---|---|
| Plataforma | AMD Kria KV260 |
| Aceleración | Kernels HLS ejecutados mediante XRT |
| Modelo | MobileNetV2 |
| Pesos | `torchvision`: `MobileNet_V2_Weights.IMAGENET1K_V1` (salvo uso explícito de `--weights`) |
| Precisión | FP32 |
| Entrada | Imagen RGB, preprocesada a 224 × 224 píxeles |
| Imagen utilizada | `test_images/test_1.jpeg` |
| Imágenes por inferencia | 1 |
| Tamaño de batch | 1 |
| Batches simultáneos (`inflight`) | 1 |
| Ejecución | Secuencial (`sequential`) |
| Calentamientos (`warm-up`) | 8 |
| Repeticiones medidas | 120 por versión |
| Pausa entre repeticiones | 0 s (`--rest 0`) |
| Script de medición | `performance_benchmarks/performance/benchmark_ich.py` |

La misma imagen se utiliza en todas las repeticiones. Los calentamientos **no** se incluyen en las estadísticas. La construcción del grafo, la carga de parámetros persistentes, la lectura/preparación de la imagen y la inferencia PyTorch de referencia se realizan fuera del intervalo cronometrado de cada repetición.

### Ejecución del benchmark

Desde la raíz del repositorio, con el ambiente de Python configurado y la versión FPGA correspondiente cargada:

```bash
PYTHONPATH="$PWD:$PWD/host:$PWD/host/bindings" \
python performance_benchmarks/performance/benchmark_ich.py \
   --image test_images/test_1.jpeg \
   --batch-size 1 \
   --inflight-batches 1 \
   --warmup 8 \
   --repeats 120 \
   --mode sequential
```

El benchmark crea una carpeta independiente por ejecución en:

```text
performance_benchmarks/results/mobilenetv2_fpga_paper_b1/<marca_temporal>/
```

Entre sus archivos se encuentran `summary.csv`, `timings.csv`, `python_profile.csv`, `operation_summary.csv` y los perfiles secuenciales de cada repetición (`sequential_profile_rep*.csv`).

## 3. Resultados generales

Los valores corresponden a las **120 repeticiones medidas** de cada versión y a una sola imagen por inferencia.

| Métrica | Sin optimizar | Optimizada |
|---|---:|---:|
| Media (ms) | 18 742.499 | **3 113.977** |
| Media (s) | 18.742 | **3.114** |
| Mediana (ms) | 18 742.039 | 3 113.833 |
| Desviación estándar (ms) | 3.313 | 2.436 |
| Percentil 95, P95 (ms) | 18 746.723 | 3 117.386 |
| Mínimo (ms) | 18 737.841 | 3 108.296 |
| Máximo (ms) | 18 762.371 | 3 125.353 |
| Rendimiento (imágenes/s) | 0.0534 | **0.3211** |

**P95** indica el tiempo por debajo del cual se encuentra aproximadamente el 95 % de las mediciones.

### Aceleración global

Se calcula a partir de las medias de ambas versiones:

- **Aceleración:** `18 742.499 / 3 113.977 = 6.02×`.
- **Reducción de latencia:** `(1 - 3 113.977 / 18 742.499) × 100 = 83.39 %`.

La optimización ahorra, en promedio, **15 628.522 ms (15.629 s) por inferencia**.

## 4. Utilización de recursos FPGA

Se presentan los recursos estimados para el diseño integrado de ocho kernels, a partir de los reportes de implementación de Vivado 2024.2. Se comparan la implementación original y la versión con optimización de Pointwise Conv2D.

### Diseño original

| Recurso | Utilizado | Disponible | Utilización (%) |
|---|---:|---:|---:|
| LUT | 63 401 | 117 120 | 54,13 % |
| FF | 77 953 | 234 240 | 33,28 % |
| BRAM | 82,5 | 144 | 57,29 % |
| DSP | 145 | 1 248 | 11,62 % |

### Diseño con Pointwise Conv2D optimizado

| Recurso | Utilizado | Disponible | Utilización (%) |
|---|---:|---:|---:|
| LUT | 70 864 | 117 120 | 60,51 % |
| FF | 89 395 | 234 240 | 38,16 % |
| BRAM | 84,5 | 144 | 58,68 % |
| DSP | 228 | 1 248 | 18,27 % |

### Efecto de la optimización de Pointwise Conv2D sobre los recursos

La optimización de Pointwise Conv2D incorpora procesamiento paralelo de canales de salida, almacenamiento local y segmentación de los datos (*tiling*).

La siguiente tabla presenta la variación de recursos del diseño integrado al sustituir el kernel Pointwise original por su versión optimizada.

| Recurso | Antes de optimizar | Después de optimizar | Incremento |
|---|---:|---:|---:|
| LUT | 63 401 | 70 864 | +7 463 |
| FF | 77 953 | 89 395 | +11 442 |
| BRAM | 82,5 | 84,5 | +2 |
| DSP | 145 | 228 | +83 |

La versión optimizada presenta un incremento en la utilización de recursos, principalmente asociado al mayor paralelismo introducido en Pointwise Conv2D.


## 5. Tiempo acumulado por operación

Los resultados de `operation_summary.csv` representan la **suma de `wrapper_ms` de todos los nodos de cada tipo en una inferencia**, promediada sobre 120 repeticiones.

| Operación | Sin optimizar (ms) | Optimizada (ms) |
|---|---:|---:|
| AvgPool2D | 7.084 | 7.075 |
| BatchNorm2D | 56.423 | 53.630 |
| Conv2D | 705.410 | 705.679 |
| Depthwise | 2 022.276 | 2 021.461 |
| LayerAdd | 2.631 | 2.481 |
| Linear | 58.255 | 58.226 |
| **Pointwise** | **15 844.249** | **220.363** |
| ReLU6 | 42.922 | 42.187 |
| **Suma de wrappers** | **18 739.250** | **3 111.101** |

### Efecto específico de Pointwise

| Indicador | Valor |
|---|---:|
| Tiempo de Pointwise sin optimizar | 15 844.249 ms |
| Tiempo de Pointwise optimizado | 220.363 ms |
| Aceleración de Pointwise | **71.90×** |
| Reducción del tiempo de Pointwise | **98.61 %** |
| Tiempo reducido en Pointwise | **15 623.887 ms** |

La disminución del tiempo acumulado de Pointwise representa aproximadamente el **99.97 % de la reducción observada en la latencia global**. Se trata de una comparación de tiempos medidos en los wrappers, no de una atribución causal basada en contadores internos del hardware.

### Distribución de los cuellos de botella

Las proporciones siguientes se calculan respecto a la **latencia global promedio** de cada versión.

| Operación | Sin optimizar | Optimizada |
|---|---:|---:|
| Pointwise | **84.54 %** | 7.08 % |
| Depthwise | 10.79 % | **64.92 %** |
| Conv2D | 3.76 % | 22.66 % |

En la versión sin optimizar, **Pointwise** domina la latencia. Después de optimizarlo, la mayor contribución corresponde a **Depthwise**, seguida de **Conv2D**.

## 6. Desglose del tiempo observado desde Python

Promedios de las 120 filas de fase `measured` en los archivos `python_profile.csv`.

| Etapa | Sin optimizar (ms) | Optimizada (ms) |
|---|---:|---:|
| Preparación (`preparation_ms`) | 0.698 | 0.726 |
| Carga de entrada (`upload_ms`) | 0.370 | 0.372 |
| Ejecución (`execution_ms`) | **18 741.088** | **3 112.544** |
| Recuperación de salida (`download_ms`) | 0.113 | 0.112 |
| Conversión (`conversion_ms`) | 0.090 | 0.091 |
| Limpieza (`cleanup_ms`) | 0.000 | 0.000 |
| **Total perfilado (`total_profile_ms`)** | **18 742.359** | **3 113.845** |
| Tiempo no desglosado (`unaccounted_ms`) | 0.140 | 0.132 |

El tiempo de ejecución (`execution_ms`) incluye las operaciones lanzadas mediante el runtime FPGA, las esperas y la sobrecarga asociada a XRT, además de cualquier gestión o transferencia interna que se realice durante esa etapa.

Asimismo, `upload_ms` y `download_ms` son etapas registradas por la interfaz a nivel de host.

La latencia de `summary.csv` se obtiene con `time.perf_counter()` alrededor de `graph.run_batches()`. Por ello, puede diferir ligeramente de la suma de etapas perfiladas. La validación numérica posterior a la ejecución queda fuera del intervalo cronometrado.

## 7. Validación numérica

Para cada repetición, el benchmark compara la salida FPGA con la referencia **PyTorch** mediante:

- **RMSE:** raíz del error cuadrático medio entre los *logits* (valores de salida antes de aplicar probabilidades).
- **Error máximo absoluto:** mayor diferencia entre los valores comparados.
- **Predicciones diferentes:** cantidad de imágenes cuya clase predicha por FPGA no coincide con PyTorch.
- **Tolerancia:** `1e-3` para el error máximo por imagen, salvo que se cambie `--tolerance`.

En una repetición comunicada de la versión **sin optimizar** (repetición 60) se registraron:

| Métrica | Resultado |
|---|---:|
| RMSE | `3.047e-06` |
| Error máximo absoluto | `1.335e-05` |
| Predicciones diferentes frente a PyTorch | 0 |

## 8. Alcance y consideraciones para la comparación

- Los resultados se obtuvieron con **una sola imagen y ejecución secuencial**. No se evalúa aquí la ganancia de un *pipeline* con múltiples imágenes o batches simultáneos.
- Las métricas globales describen **latencia de inferencia desde el host**, no únicamente tiempo de cómputo del circuito FPGA.
- La agrupación por operación incluye sobrecarga de los wrappers y XRT.
- Se excluyen del intervalo de cada inferencia la carga inicial del modelo, la construcción del grafo, la preparación previa de la imagen y la carga inicial de los pesos persistentes.
- Las comparaciones con CPU/GPU de otras plataformas deben especificar si incluyen preprocesamiento y transferencias, pues sus límites de medición pueden diferir.

## 9. Conclusión

En el protocolo de una imagen (`batch=1`, 8 calentamientos, 120 repeticiones), la versión optimizada de MobileNetV2 sobre la Kria KV260 redujo la latencia promedio de **18.742 s a 3.114 s**, equivalente a una **aceleración de 6.02×** y una **reducción de 83.39 %**.

La mayor parte de la mejora registrada corresponde a **Pointwise**. Tras optimizar esta operación, **Depthwise** pasa a concentrar la mayor proporción del tiempo observado, seguida de **Conv2D**. Estos resultados caracterizan el rendimiento de la interfaz y los kernels en las condiciones medidas, sin atribuir los tiempos reportados exclusivamente a la ejecución interna del hardware.

## 10. Reproducción del proyecto

El repositorio **Proyecto-ICH**, en `MobileNetV2-FPGA/`, contiene el código HLS, los Makefiles, la configuración de enlace, una copia del benchmark y los resultados experimentales. La ejecución de la red neuronal desde Python utiliza una **interfaz externa**, disponible en [pytorch-fpga-configurable-interface](https://github.com/anghernandez/pytorch-fpga-configurable-interface). La interfaz se instala y compila por separado en la Kria KV260.

### 10.1. Herramientas y requisitos

- **Compilación del hardware:** AMD Vitis/Vivado 2024.2 y la plataforma `MobileNetV2-FPGA/platform/kv260_custom.xpfm` incluida en el proyecto.
- **Ejecución (Kria KV260):** XRT y `xmutil` configurados; para las mediciones se utilizó Ubuntu 22.04 y XRT 2.13.0.
- **Interfaz:** repositorio PyTorch–FPGA con el runtime C++, bindings Pybind11, dependencias de Python y el modelo MobileNetV2. Consultar sus instrucciones de instalación para preparar el entorno de Python.
- **Imagen de prueba:** `test_images/test_1.jpeg` en el repositorio de la interfaz; pesos `MobileNet_V2_Weights.IMAGENET1K_V1` de `torchvision`.

### 10.2. Compilación y empaquetado del acelerador

En el equipo de desarrollo, desde la raíz de `Proyecto-ICH`:

```bash
source ~/Xilinx/Vitis/2024.2/settings64.sh
cd MobileNetV2-FPGA/HLS
make build
```

El Makefile general invoca los Makefiles HLS individuales, integra los nueve kernels y genera los archivos para el despliegue en `package.hw/`. El kernel Pointwise seleccionado es `PointWiseConv2d_Optimizado/`. Se incluyen nueve kernels en el diseño, aunque MobileNetV2 no utiliza Tanh.

Los cuatro archivos destinados a la Kria son:

```text
package.hw/fpga_kernels.bit.bin
package.hw/fpga_kernels.dtbo
package.hw/fpga_kernels.xclbin
package.hw/shell.json
```

Los archivos `.xo`, los directorios temporales de Vitis/Vivado y los archivos de `package.hw/` se generan localmente y no se incluyen en Git. La compilación desde cero debe realizarse con las herramientas y la plataforma indicadas; los comandos de construcción no constituyen por sí solos una validación de una compilación limpia en otro equipo.

### 10.3. Transferencia desde el equipo de desarrollo

Desde la raíz de `Proyecto-ICH`, en el equipo de desarrollo**:

```bash
ssh kria 'mkdir -p ~/fpga_accelerators_optimizado'
scp MobileNetV2-FPGA/HLS/package.hw/{fpga_kernels.bit.bin,fpga_kernels.dtbo,fpga_kernels.xclbin,shell.json} \
  kria:~/fpga_accelerators_optimizado/
```

`kria` es el alias SSH usado en el entorno de pruebas. En otro equipo, sustituirlo por el usuario y la dirección IP de la placa.

### 10.4. Instalación y carga del acelerador en la Kria

En la **Kria KV260**:

```bash
sudo mkdir -p /lib/firmware/xilinx/fpga_accelerators_optimizado
sudo cp ~/fpga_accelerators_optimizado/{fpga_kernels.bit.bin,fpga_kernels.dtbo,fpga_kernels.xclbin,shell.json} \
  /lib/firmware/xilinx/fpga_accelerators_optimizado/

# Descargar la aplicación anterior, si hay una cargada:
sudo xmutil unloadapp
sudo xmutil loadapp fpga_accelerators_optimizado
sudo xmutil listapps
```

Si no hay una aplicación activa, `unloadapp` puede indicar que no hay nada que descargar. `listapps` sirve para comprobar el registro y la carga del diseño. El archivo `shell.json` empleado declara `shell_type: XRT_FLAT` y `num_slots: 1`.

### 10.5. Configuración y recompilación de la interfaz PyTorch–FPGA

La interfaz está en un [repositorio independiente](https://github.com/anghernandez/pytorch-fpga-configurable-interface). Se debe utilizar una versión que contenga el runtime DAG y el benchmark indicados en este documento.

En la **Kria**, dentro de la raíz de la implementación de la interfaz, configurar en `host/runtime/fpga_runtime.cpp`:

```cpp
const std::string xclbin_path =
    "/lib/firmware/xilinx/fpga_accelerators_optimizado/fpga_kernels.xclbin";
```

La ruta debe corresponder a la aplicación cargada con `xmutil`. Con el entorno de Python activado, recompilar los bindings C++:

```bash
make rebuild
```

Este comando genera nuevamente el módulo Python `fpga_dag*.so`; **no** recompila los kernels HLS. Para ejecutar la versión original, deben cargarse `fpga_accelerators` y su correspondiente ruta `/lib/firmware/xilinx/fpga_accelerators/fpga_kernels.xclbin`. El código HLS de Pointwise original no forma parte de esta entrega: aquí se conserva la versión optimizada.

### 10.6. Ejecución del benchmark

Desde la raíz del repositorio **de la interfaz**, en la Kria, con el entorno de Python activado y el acelerador correspondiente cargado:

```bash
PYTHONPATH="$PWD:$PWD/host:$PWD/host/bindings" \
python performance_benchmarks/performance/benchmark_ich.py \
  --image test_images/test_1.jpeg \
  --batch-size 1 \
  --inflight-batches 1 \
  --warmup 8 \
  --repeats 120 \
  --rest 0 \
  --mode sequential
```

Las mediciones corresponden a **una imagen por inferencia**, en modo secuencial, con ocho calentamientos y 120 repeticiones medidas. El script genera una carpeta con marca temporal en `performance_benchmarks/results/mobilenetv2_fpga_paper_b1/`, que incluye `summary.csv`, `timings.csv`, `python_profile.csv`, `operation_summary.csv` y `sequential_profile_rep*.csv`.

La copia `MobileNetV2-FPGA/benchmarks/benchmark_ich.py` coincide exactamente (SHA-256) con la utilizada en la Kria durante la evaluación; se conserva como respaldo en el proyecto ICH. Los CSV de ambas mediciones están disponibles en `MobileNetV2-FPGA/results/baseline/` y `MobileNetV2-FPGA/results/optimized/`.

**Alcance:** el repositorio ICH permite reconstruir el diseño **optimizado** y reproducir su medición usando la interfaz externa. Los resultados del diseño **sin optimizar** se incluyen como referencia experimental, pero su código HLS de Pointwise no se distribuye aquí. Para comparar versiones, debe cargarse en cada ejecución el acelerador y el `.xclbin` correspondiente; no deben mezclarse.