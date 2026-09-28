# MobileNetV2 — PyTorch vs C++

Implementación de **MobileNetV2** utilizando capas desarrolladas en **C++** e integradas con Python mediante **pybind11**.

El proyecto utiliza la implementación de MobileNetV2 de PyTorch como referencia y permite comparar la inferencia realizada por PyTorch con la implementación basada en kernels C++.

## Objetivo

Implementar las principales operaciones utilizadas por MobileNetV2 en C++ y comprobar su funcionamiento comparando la salida del modelo con la implementación de referencia en PyTorch.

Actualmente se implementan las siguientes operaciones:

- Conv2D
- Depthwise Conv2D
- Pointwise Conv2D
- BatchNorm2D
- ReLU6
- Layer Add
- Global Average Pooling
- Linear

Los parámetros del modelo de referencia se cargan en la implementación C++ para realizar la comparación bajo las mismas condiciones.

## Estructura del proyecto

```text
MobileNetV2/
├── app/
│   └── run_cpp_mobilenetv2.py
│
├── bindings/
│   └── pybind_module.cpp
│
├── kernels/
│   ├── BatchNorm2d.cpp
│   ├── BatchNorm2d.hpp
│   ├── Conv2d.cpp
│   ├── Conv2d.hpp
│   ├── Depthwise_Conv2d.cpp
│   ├── Depthwise_Conv2d.hpp
│   ├── GlobalAvgPool2d.cpp
│   ├── GlobalAvgPool2d.hpp
│   ├── LayerAdd.cpp
│   ├── LayerAdd.hpp
│   ├── Linear.cpp
│   ├── Linear.hpp
│   ├── Pointwise_Conv2d.cpp
│   ├── Pointwise_Conv2d.hpp
│   ├── ReLU6.cpp
│   └── ReLU6.hpp
│
├── models/
│   ├── CppMobileNetV2.py
│   ├── ManualMobileNetV2.py
│   └── MobileNetV2.py
│
├── wrappers/
│   ├── BatchNorm2d.py
│   ├── Conv2d.py
│   ├── DepthwiseConv2d.py
│   ├── GlobalAvgPool2d.py
│   ├── LayerAdd.py
│   ├── Linear.py
│   └── PointwiseConv2d.py
│
├── test_images/
│   └── imagen.jpeg
│
├── Makefile
└── README.md
```

## Flujo de ejecución

La implementación sigue el siguiente flujo:

<p align="center">
  <img src="diagrama.png" alt="Flujo del proyecto" width="400">
</p>

<p align="center">
  <em>Figura 1. Flujo del proyecto. Imagen generada con inteligencia artificial.</em>
</p>

Se comparan los **1000 logits** producidos por ambos modelos y se calculan métricas de error como RMSE y error máximo.

También se compara la clase predicha por ambas implementaciones.

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

Para capturar desde la cámara se necesitan las bibliotecas de desarrollo
de OpenCV para C++ y `pkg-config`:

```bash
sudo apt install libopencv-dev pkg-config
```

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

## Ejecución

### Captura desde cámara

Con el entorno virtual del proyecto existente, instala OpenCV para C++
si todavía no lo tienes:

```bash
sudo apt install libopencv-dev pkg-config
```

Luego abre la vista previa y toma la foto con **espacio**:

```bash
make camera
```

La imagen se guarda en `test_images/captura.jpg`. **Q** o **Esc** cancelan la captura.
Con `make camera CAMERA=1` puedes elegir otra cámara; con
`make camera SAVE=test_images/gato.jpg` puedes cambiar el archivo de salida.
Para tomar una foto automáticamente, sin vista previa, ejecuta `make camera-auto`.

La cámara se abre y se captura en `bindings/camera_capture.cpp` mediante
OpenCV C++. Ese módulo entrega una imagen RGB a Python. Python ejecuta las
transformaciones de los pesos de ImageNet-1K y arma el modelo. La inferencia
de la implementación comparada utiliza los kernels C++ del proyecto.
El programa muestra la clase más probable, el top 5 y la comparación con
PyTorch.

### Comparación con imagen almacenada

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
