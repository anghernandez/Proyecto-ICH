# *******************************
# MobileNetV2 - PyTorch vs C++
#********************************

PYTHON := venv/bin/python
PIP := venv/bin/pip
CXX := g++

VENV := venv

EXT_SUFFIX = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET = cpp_kernels$(EXT_SUFFIX)

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
INCLUDES = $(shell $(PYTHON) -m pybind11 --includes) -Ikernels

IMAGE ?= test_images/imagen.jpeg

# PROFILE=1 activa la instrumentacion std::chrono (ScopedTimer).
# Por defecto (PROFILE=0) el binario queda igual que si nunca
# se hubiera instrumentado -- esto es lo que permite comparar
# el overhead de medir (ver PERFILADO.md, paso 4).
PROFILE ?= 0

ifeq ($(PROFILE),1)
CXXFLAGS += -DPROFILE_KERNELS
endif


# ------------------------------------------------------------
# Comandos principales
# ------------------------------------------------------------

.PHONY: all setup build run clean help

all: build


# Crear entorno virtual e instalar dependencias
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install pybind11 numpy torch torchvision pillow


# Compilar módulo C++/pybind11
build: $(TARGET)

$(TARGET): $(BINDINGS) $(KERNELS) $(wildcard kernels/*.hpp)
	$(CXX) $(CXXFLAGS) \
		$(INCLUDES) \
		$(BINDINGS) \
		$(KERNELS) \
		-o $(TARGET)


# Ejecutar comparación PyTorch vs C++
run: build
	PYTHONPATH="$(CURDIR)" $(PYTHON) app/run_cpp_mobilenetv2.py $(IMAGE)


# Eliminar archivos generados
clean:
	rm -f cpp_kernels*.so
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
	@echo "  make clean   Eliminar archivos generados"
	@echo "  make help    Mostrar esta ayuda"
	@echo ""
	@echo "Para utilizar otra imagen:"
	@echo "  make run IMAGE=ruta/a/imagen.jpg"
	@echo ""
	@echo "Para compilar con instrumentacion std::chrono activa:"
	@echo "  make build PROFILE=1"

# Compilar con perfilado habilitado
.PHONY: profile-build

profile-build:
	$(MAKE) -B build PROFILE=1