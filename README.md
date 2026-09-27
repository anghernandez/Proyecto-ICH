# MobileNetV2 — PyTorch vs C++


## Requisitos

El proyecto requiere:

- Linux
- Python 3
- `g++`
- `make`
- Python `venv`

Las dependencias de Python utilizadas son:

- PyTorch
- Torchvision
- NumPy
- pybind11
- Pillow

Estas dependencias pueden instalarse automáticamente utilizando el `Makefile`.

## Instalación

Después de clonar el repositorio, entrar al directorio del proyecto:

```bash
cd MobileNetV2
```

Crear el entorno virtual e instalar las dependencias:

```bash
make setup
```

Este comando crea el directorio:

```text
venv/
```

e instala las dependencias necesarias.

## Compilación

Para compilar los kernels C++ y generar el módulo utilizado desde Python:

```bash
make build
```

La compilación genera un módulo similar a:

```text
cpp_kernels.cpython-310-x86_64-linux-gnu.so
```

El nombre exacto puede variar dependiendo de la versión de Python y de la arquitectura del sistema.

Para compilar con la instrumentación de perfilado (`std::chrono`) activa:

```bash
make build PROFILE=1
```

Por defecto (`PROFILE=0`) el binario queda igual que si nunca se hubiera instrumentado — esto es lo que permite medir el overhead real de la instrumentación (ver sección de Resultados).

## Ejecución

Para compilar, si es necesario, y ejecutar la comparación completa:

```bash
make run
```

Por defecto se utiliza:

```text
test_images/imagen.jpeg
```

También se puede utilizar otra imagen:

```bash
make run IMAGE=ruta/a/imagen.jpg
```

Por ejemplo:

```bash
make run IMAGE=test_images/perro.jpg
```

`app/run_cpp_mobilenetv2.py` también acepta una resolución de entrada opcional como segundo argumento (por defecto 224):

```bash
PYTHONPATH="$(pwd)" venv/bin/python app/run_cpp_mobilenetv2.py test_images/imagen.jpeg 96
```

## Ejemplo de resultado

Una ejecución correcta produce una salida similar a:

```text
============================================================
MOBILENETV2 - PYTORCH vs C++
============================================================

Forma salida PyTorch: (1, 1000)
Forma salida C++:     (1, 1000)

RMSE logits:          3.0507894735e-06
Error máximo logits: 1.6212463379e-05

Predicción PyTorch:
  [239] Bernese mountain dog

Predicción C++:
  [239] Bernese mountain dog

RESULTADO: PASS
```

Este resultado indica que ambas implementaciones producen salidas numéricamente cercanas y la misma clasificación para la imagen evaluada.

## Comandos disponibles

```bash
make setup
```

Crea el entorno virtual e instala las dependencias.

```bash
make build
```

Compila los kernels C++ y el módulo de pybind11.

```bash
make run
```

Ejecuta la comparación MobileNetV2 PyTorch vs C++.

```bash
make clean
```

Elimina los archivos generados durante la compilación.

```bash
make help
```

Muestra los comandos disponibles.

---

## Perfilado (Prototipo en C)

Se instrumentó cada capa C++ (`conv2d`, `depthwise_conv`, `pointwise_conv`, `batchnorm`, `activacion`, `pooling`, `layer_add`, `linear`) con `std::chrono`, dentro de `bindings/pybind_module.cpp` (ver `kernels/Profiler.hpp`). El preprocesamiento en Python no se mide — el perfilado se enfoca solo en la inferencia en C++.

Reproducir el perfilado:

```bash
make clean && make build PROFILE=1
PYTHONPATH="$(pwd)" venv/bin/python app/profile_run.py --iterations 120 --warmup 8 --output resultados/perfilado.csv
```

Comparar overhead (con/sin instrumentación):

```bash
make clean && make build
PYTHONPATH="$(pwd)" venv/bin/python app/profile_run.py --overhead-only --iterations 200 --warmup 10
make clean && make build PROFILE=1
PYTHONPATH="$(pwd)" venv/bin/python app/profile_run.py --overhead-only --iterations 200 --warmup 10
```

Validar con `perf`:

```bash
PYTHONPATH="$(pwd)" perf stat -- venv/bin/python app/profile_run.py --overhead-only --iterations 60 --warmup 8
PYTHONPATH="$(pwd)" perf record -o perf.data -- venv/bin/python app/profile_run.py --overhead-only --iterations 60 --warmup 8
perf report -i perf.data --stdio --sort=overhead,symbol | head -40
```

### Resultados — Milagro

**Entorno de prueba:**

| Característica | Valor |
|---|---|
| Equipo | Dell G16-7620 |
| Procesador | Intel Core i7-12700H (12ª gen) |
| Arquitectura | x86-64 |
| Set de instrucciones vectoriales | AVX2 |
| Núcleos / hilos | 14 núcleos / 20 hilos |
| Frecuencia base / turbo | 2.3 GHz / 4.7 GHz |
| RAM | 16 GiB (2×8 GiB), SODIMM 4800 MHz |

**Resultados obtenidos:**

- **Cuello de botella:** `pointwise_conv` domina el tiempo total de inferencia — ~71% según `std::chrono` (muestreo de 120 corridas) y **59.45%** según `perf report` (ciclos de CPU). Ambas técnicas coinciden en el orden: `pointwise_conv` > `depthwise_conv` (8.74%) > `batchnorm` (5.71%) > `conv2d` inicial (4.51%) > `relu6` (4.25%).
- **Muestreo:** 120 corridas medidas + 8 de warm-up descartadas, sobre el mismo tensor de entrada (224×224). Total: 18,120 filas en el CSV (151 operaciones × 120 corridas, sin pérdidas).
- **Varianza:** al agrupar por (etapa, forma del tensor), la mayoría de las combinaciones tienen desviación estándar baja — la "varianza" observada al agrupar solo por etapa era mayormente por mezclar capas de tamaños distintos, no ruido real. Tres combinaciones sí muestran ruido genuino incluso con forma idéntica (`pointwise_conv 1x960x7x7`, `depthwise_conv 1x144x56x56`, `batchnorm 1x32x112x112`) — pendiente de investigar más a fondo.
- **Overhead de instrumentación:** no medible / menor al ruido de fondo del sistema (~1-2%) — comparando 200 corridas con y sin `PROFILE_KERNELS`, la versión instrumentada no fue medible más lenta.
- **Evidencia para optimización:** IPC (instrucciones por ciclo) de solo ~1.9-2.4 y ~33-37% de ciclos en `tma_backend_bound` (según `perf stat`) — el código no está vectorizado y el acceso a memoria es un factor limitante. Punto de partida claro para la etapa de paralelismo/optimización.

### Resultados — Angie

Falta correr el mismo procedimiento en la segunda computadora y comparar contra lo anterior. La comparación contra el sistema empotrado se hace en una etapa posterior.

