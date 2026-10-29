import torch
import torch.nn as nn
import onnx

from torchvision.models import mobilenet_v2
from torch.nn.utils.fusion import fuse_conv_bn_eval



# Configuracion
OUTPUT_FILE = "mobilenetv2_fusion.onnx"
OPSET = 11

device = torch.device("cpu")

print("PyTorch:", torch.__version__)
print("ONNX:", onnx.__version__)



# Cargar MobileNetV2
model = mobilenet_v2(pretrained=True)
model.eval()



# Fusion Conv2D + BatchNorm2D
def fusionar_modelo(model):

    fusiones = 0

    for module in model.modules():

        if not isinstance(module, nn.Sequential):
            continue

        hijos = list(module.named_children())

        for i in range(len(hijos) - 1):

            nombre_conv, conv = hijos[i]
            nombre_bn, bn = hijos[i + 1]

            if (
                isinstance(conv, nn.Conv2d)
                and isinstance(bn, nn.BatchNorm2d)
            ):

                module._modules[nombre_conv] = (
                    fuse_conv_bn_eval(conv, bn)
                )

                module._modules[nombre_bn] = nn.Identity()

                fusiones += 1

    return fusiones


fusiones = fusionar_modelo(model)

batchnorm_restantes = sum(
    isinstance(m, nn.BatchNorm2d)
    for m in model.modules()
)

print("Fusiones realizadas:", fusiones)
print("BatchNorm restantes:", batchnorm_restantes)

if fusiones != 52 or batchnorm_restantes != 0:
    raise RuntimeError("La fusion no se completo correctamente")


# Entrada de referencia FP32
dummy_input = torch.randn(
    1, 3, 224, 224,
    dtype=torch.float32
)


# Exportar ONNX
print("\nExportando MobileNetV2...")

with torch.no_grad():

    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_FILE,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None
    )



# Validar ONNX
onnx_model = onnx.load(OUTPUT_FILE)

onnx.checker.check_model(onnx_model)

print("\n========== EXPORTACION ==========")
print("Archivo:", OUTPUT_FILE)
print("Opset:", OPSET)
print("Fusiones:", fusiones)
print("Modelo ONNX valido")


# Revisar operaciones exportadas
from collections import Counter

operaciones = Counter(
    node.op_type
    for node in onnx_model.graph.node
)

print("\nOperaciones ONNX:")

for nombre, cantidad in sorted(operaciones.items()):
    print("{}: {}".format(nombre, cantidad))
