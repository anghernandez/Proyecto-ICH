# ********************************
# MobileNetV2 - PyTorch vs C++
# ********************************

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


# ------------------------------------------------------------
# Entorno virtual
# ------------------------------------------------------------

# Crear entorno virtual e instalar dependencias
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install pybind11 numpy torch torchvision pillow


# ------------------------------------------------------------
# MobileNetV2 C++ / pybind11
# ------------------------------------------------------------

# Compilar módulo C++/pybind11
build: $(TARGET)

$(TARGET): $(BINDINGS) $(KERNELS)
	$(CXX) $(CXXFLAGS) \
		$(INCLUDES) \
		$(BINDINGS) \
		$(KERNELS) \
		-o $(TARGET)


# ------------------------------------------------------------
# Cámara C++ / OpenCV / pybind11
# ------------------------------------------------------------

# Compilar módulo de captura de cámara
camera-module: $(CAMERA_TARGET)

$(CAMERA_TARGET): bindings/camera_capture.cpp
	$(CXX) $(CXXFLAGS) \
		$(shell $(PYTHON) -m pybind11 --includes) \
		$(shell pkg-config --cflags opencv4) \
		$< \
		-o $@ \
		$(shell pkg-config --libs opencv4)


# ------------------------------------------------------------
# Ejecución
# ------------------------------------------------------------

# Ejecutar comparación PyTorch vs C++ con una imagen almacenada
run: build
	PYTHONPATH="$(CURDIR)" $(PYTHON) $(APP) --image "$(IMAGE)"


# Capturar imagen desde cámara y ejecutar MobileNetV2
camera: build camera-module
	PYTHONPATH="$(CURDIR)" $(PYTHON) $(APP) \
		--camera $(CAMERA) \
		--save "$(SAVE)"


# Capturar automáticamente sin ventana de previsualización
camera-auto: build camera-module
	PYTHONPATH="$(CURDIR)" $(PYTHON) $(APP) \
		--camera $(CAMERA) \
		--no-preview \
		--save "$(SAVE)"


# ------------------------------------------------------------
# Limpieza
# ------------------------------------------------------------

clean:
	rm -f cpp_kernels*.so
	rm -f camera_capture*.so
	rm -f kernels/*.o
	rm -rf __pycache__
	rm -rf app/__pycache__
	rm -rf models/__pycache__
	rm -rf wrappers/__pycache__


# ------------------------------------------------------------
# Ayuda
# ------------------------------------------------------------

help:
	@echo "MobileNetV2 - PyTorch vs C++"
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make setup          Crear venv e instalar dependencias"
	@echo "  make build          Compilar los kernels C++"
	@echo "  make run            Ejecutar con una imagen almacenada"
	@echo "  make camera-module  Compilar el módulo de cámara"
	@echo "  make camera         Capturar imagen y ejecutar MobileNetV2"
	@echo "  make camera-auto    Capturar sin vista previa y ejecutar"
	@echo "  make clean          Eliminar archivos generados"
	@echo "  make help           Mostrar esta ayuda"
	@echo ""
	@echo "Opciones:"
	@echo "  make run IMAGE=ruta/a/imagen.jpg"
	@echo "  make camera CAMERA=0"
	@echo "  make camera CAMERA=0 SAVE=test_images/captura.jpg"
	@echo "  make camera-auto CAMERA=0 SAVE=test_images/captura.jpg"