# ****************************************************
# MobileNetV2 - NVIDIA Jetson Nano
# PyTorch + CUDA + cuDNN + Fusion Conv/BN + TensorRT
# ****************************************************

PYTHON := python3
TRTEXEC := /usr/src/tensorrt/bin/trtexec

SCRIPTS := scripts
MODELOS := modelos
IMAGES := test_images
RESULTADOS := resultados

ENGINE := $(MODELOS)/mobilenetv2_fp32.engine
ONNX := $(MODELOS)/mobilenetv2_fusion.onnx

export OPENBLAS_CORETYPE := ARMV8

.PHONY: help check prepare baseline profile-baseline \
        cudnn fusion profile-fusion export engine tensorrt \
        profile-tensorrt analyze-tensorrt collect all

help:
	@echo "-----------------------------------------------"
	@echo " MobileNetV2 - Jetson Nano"
	@echo "-----------------------------------------------"
	@echo "make check             Verificar dependencias"
	@echo "make prepare           Preparar rutas"
	@echo "make baseline          Benchmark PyTorch CUDA"
	@echo "make profile-baseline  Perfilado PyTorch CUDA"
	@echo "make cudnn             Benchmark cuDNN"
	@echo "make fusion            Benchmark fusion Conv/BN"
	@echo "make profile-fusion    Perfilado optimizado"
	@echo "make export            Exportar modelo ONNX"
	@echo "make engine            Construir motor TensorRT"
	@echo "make tensorrt          Benchmark TensorRT"
	@echo "make profile-tensorrt  Perfilado por capas TensorRT"
	@echo "make analyze-tensorrt  Analizar perfil TensorRT"
	@echo "make collect           Organizar resultados"
	@echo "make all               Ejecutar benchmarks"

check:
	@$(PYTHON) -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
	@$(PYTHON) -c "import torchvision; print('Torchvision:', torchvision.__version__)"
	@$(PYTHON) -c "import tensorrt; print('TensorRT:', tensorrt.__version__)"
	@$(PYTHON) -c "import onnx; print('ONNX:', onnx.__version__)"

prepare:
	@mkdir -p $(MODELOS)
	@mkdir -p $(RESULTADOS)/baseline
	@mkdir -p $(RESULTADOS)/cudnn
	@mkdir -p $(RESULTADOS)/fusion
	@mkdir -p $(RESULTADOS)/tensorrt/perfilado
	@ln -sfn $(IMAGES) test_images_link
	@ln -sfn $(ONNX) mobilenetv2_fusion.onnx
	@ln -sfn $(ENGINE) mobilenetv2_fp32.engine
	@echo "Directorios preparados."

baseline: prepare
	$(PYTHON) $(SCRIPTS)/profile_mobilenetv2_gpu.py $(IMAGES)/imagen.jpeg

profile-baseline: prepare
	$(PYTHON) $(SCRIPTS)/profile_torch_gpu.py

cudnn: prepare
	$(PYTHON) $(SCRIPTS)/benchmark_cudnn.py

fusion: prepare
	$(PYTHON) $(SCRIPTS)/benchmark_fusion.py

profile-fusion: prepare
	$(PYTHON) $(SCRIPTS)/profile_mobilenetv2_optimized.py

export: prepare
	$(PYTHON) $(SCRIPTS)/export_mobilenetv2_onnx.py
	@mv -f mobilenetv2_fusion.onnx $(ONNX)

engine: export
	$(TRTEXEC) \
		--onnx=$(ONNX) \
		--saveEngine=$(ENGINE) \
		--workspace=512 \
		--buildOnly

tensorrt: prepare
	$(PYTHON) $(SCRIPTS)/benchmark_tensorrt.py

profile-tensorrt: prepare
	$(TRTEXEC) \
		--loadEngine=$(ENGINE) \
		--dumpProfile \
		--exportProfile=$(RESULTADOS)/tensorrt/perfilado/perfil_capas.json \
		--separateProfileRun \
		--warmUp=0 \
		--duration=0 \
		--iterations=120 \
		2>&1 | tee $(RESULTADOS)/tensorrt/perfilado/perfilado.log

analyze-tensorrt: prepare
	$(PYTHON) $(SCRIPTS)/analizar_perfil_tensorrt.py

collect:
	@cp -n inferencias_gpu.csv $(RESULTADOS)/baseline/ 2>/dev/null || true
	@cp -n resumen_operaciones_gpu.csv $(RESULTADOS)/baseline/ 2>/dev/null || true
	@cp -n detalle_operaciones_gpu.csv $(RESULTADOS)/baseline/ 2>/dev/null || true
	@cp -n cudnn_benchmark_*.csv $(RESULTADOS)/cudnn/ 2>/dev/null || true
	@cp -n fusion_*.csv $(RESULTADOS)/fusion/ 2>/dev/null || true
	@cp -n resumen_operaciones_gpu_optimized.csv $(RESULTADOS)/fusion/ 2>/dev/null || true
	@echo "Resultados organizados."

all: baseline cudnn fusion tensorrt
