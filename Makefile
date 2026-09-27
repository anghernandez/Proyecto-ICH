# *******************************
# MobileNetV2 - PyTorch vs C++
#********************************

PYTHON := venv/bin/python
PIP := venv/bin/pip
CXX := g++

VENV := venv

TARGET := cpp_kernels$(shell python3-config --extension-suffix)
CAMERA_TARGET := camera_capture$(shell python3-config --extension-suffix)

KERNELS := \
	kernels/Linear.cpp \
	kernels/Conv2d.cpp \
	kernels/Pointwise_Conv2d.cpp \
	kernels/Depthwise_Conv2d.cpp \
	kernels/BatchNorm2d.cpp \
	kernels/GlobalAvgPool2d.cpp \
	kernels/LayerAdd.cpp \
	kernels/ReLU6.cpp

BINDINGS := bindings/pybind_module.cpp

CXXFLAGS := -O2 -Wall -shared -std=c++17 -fPIC
INCLUDES := $(shell $(PYTHON) -m pybind11 --includes) -Ikernels

IMAGE ?= test_images/imagen.jpeg
CAMERA ?= 0
SAVE ?= test_images/captura.jpg
APP := app/run_cpp_mobilenetv2.py


# ------------------------------------------------------------
# Comandos principales
# ------------------------------------------------------------

.PHONY: all setup build camera-module run camera camera-auto clean help

all: build


# Crear entorno virtual e instalar dependencias
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install --no-cache-dir pybind11 numpy pillow
	$(PIP) install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu


# Compilar módulo C++/pybind11
build: $(TARGET)

camera-module: $(CAMERA_TARGET)

$(CAMERA_TARGET): bindings/camera_capture.cpp
	$(CXX) $(CXXFLAGS) \
		$(shell $(PYTHON) -m pybind11 --includes) \
		$(shell pkg-config --cflags opencv4) \
		$< -o $@ $(shell pkg-config --libs opencv4)

$(TARGET): $(BINDINGS) $(KERNELS)
	$(CXX) $(CXXFLAGS) \
		$(INCLUDES) \
		$(BINDINGS) \
		$(KERNELS) \
		-o $(TARGET)


# Ejecutar comparación con una imagen guardada
run: build
	PYTHONPATH="$(CURDIR)" $(PYTHON) $(APP) --image "$(IMAGE)"

# Capturar desde la cámara y comparar ambas implementaciones
camera: build camera-module
	PYTHONPATH="$(CURDIR)" $(PYTHON) $(APP) --camera $(CAMERA) --save "$(SAVE)"

# Captura automática sin ventana de vista previa
camera-auto: build camera-module
	PYTHONPATH="$(CURDIR)" $(PYTHON) $(APP) --camera $(CAMERA) --no-preview --save "$(SAVE)"


# Eliminar archivos generados
clean:
	rm -f cpp_kernels*.so
	rm -f camera_capture*.so
	rm -f kernels/*.o
	rm -rf __pycache__
	rm -rf app/__pycache__
	rm -rf models/__pycache__
	rm -rf wrappers/__pycache__


# Mostrar ayuda
help:
	@echo "MobileNetV2 - PyTorch vs C++"
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make setup   Crear venv e instalar dependencias"
	@echo "  make build   Compilar los kernels C++"
	@echo "  make run     Compilar y ejecutar la prueba"
	@echo "  make camera-module      Compilar captura C++ (requiere libopencv-dev)"
	@echo "  make camera CAMERA=0  Abrir la cámara y comparar las predicciones"
	@echo "  make camera-auto       Capturar sin vista previa"
	@echo "  make clean   Eliminar archivos generados"
	@echo "  make help    Mostrar esta ayuda"
	@echo ""
	@echo "Para utilizar otra imagen:"
	@echo "  make run IMAGE=ruta/a/imagen.jpg"
	@echo "  make camera SAVE=test_images/gato.jpg"
