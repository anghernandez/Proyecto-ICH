#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Linear.hpp"
#include "Conv2d.hpp"
#include "Pointwise_Conv2d.hpp"
#include "Depthwise_Conv2d.hpp"
#include "BatchNorm2d.hpp"
#include "GlobalAvgPool2d.hpp"
#include "LayerAdd.hpp"
#include "ReLU6.hpp"
#include "Profiler.hpp"

namespace py = pybind11;


//Alias para arreglos continuos
using FloatArray = py::array_t<
    float,
    py::array::c_style
>;


// ------------------------------------------------------------
// Helper para armar la forma del tensor como texto,
// ej. "1x32x112x112"
// ------------------------------------------------------------

std::string shape_to_string(
    const std::vector<py::ssize_t>& shape
)
{
    std::string result;

    for (std::size_t index = 0; index < shape.size(); index++)
    {
        if (index > 0) {
            result += "x";
        }

        result += std::to_string(shape[index]);
    }

    return result;
}



//Linear

py::array_t<float> linear_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const py::object& bias
)
{
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info weight_info =
        weight.request();

    if (input_info.ndim != 2) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, in_features]"
        );
    }

    if (weight_info.ndim != 2) {
        throw py::value_error(
            "weight debe tener forma "
            "[out_features, in_features]"
        );
    }

    const py::ssize_t batch_size =
        input_info.shape[0];

    const py::ssize_t in_features =
        input_info.shape[1];

    const py::ssize_t out_features =
        weight_info.shape[0];

    if (weight_info.shape[1] != in_features) {
        throw py::value_error(
            "input y weight tienen "
            "in_features incompatibles"
        );
    }

    py::array_t<float> output(
        {
            batch_size,
            out_features
        }
    );

    py::buffer_info output_info =
        output.request();

    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    const auto* weight_pointer =
        static_cast<const float*>(
            weight_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    const std::string shape_str = shape_to_string(
        {batch_size, in_features}
    );

    if (bias.is_none()) {
        {
            SCOPED_TIMER("linear", shape_str);

            linear_forward(
                input_pointer,
                weight_pointer,
                nullptr,
                output_pointer,
                static_cast<int>(batch_size),
                static_cast<int>(in_features),
                static_cast<int>(out_features),
                false
            );
        }

        return output;
    }

    FloatArray bias_array =
        FloatArray::ensure(bias);

    if (!bias_array) {
        throw py::type_error(
            "bias debe ser un arreglo NumPy "
            "float32 y contiguo, o None"
        );
    }

    const py::buffer_info bias_info =
        bias_array.request();

    if (
        bias_info.ndim != 1
        || bias_info.shape[0] != out_features
    ) {
        throw py::value_error(
            "bias debe tener forma [out_features]"
        );
    }

    const auto* bias_pointer =
        static_cast<const float*>(
            bias_info.ptr
        );

    {
        SCOPED_TIMER("linear", shape_str);

        linear_forward(
            input_pointer,
            weight_pointer,
            bias_pointer,
            output_pointer,
            static_cast<int>(batch_size),
            static_cast<int>(in_features),
            static_cast<int>(out_features),
            true
        );
    }

    return output;
}

//Conv2d
py::array_t<float> conv2d_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const py::object& bias,
    int output_height,
    int output_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width
)
{
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info weight_info =
        weight.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch, in_channels, height, width]"
        );
    }

    if (weight_info.ndim != 4) {
        throw py::value_error(
            "weight debe tener forma "
            "[out_channels, in_channels, "
            "kernel_height, kernel_width]"
        );
    }

    const py::ssize_t batch_size =
        input_info.shape[0];

    const py::ssize_t input_channels =
        input_info.shape[1];

    const py::ssize_t input_height =
        input_info.shape[2];

    const py::ssize_t input_width =
        input_info.shape[3];

    const py::ssize_t output_channels =
        weight_info.shape[0];

    const py::ssize_t weight_input_channels =
        weight_info.shape[1];

    const py::ssize_t kernel_height =
        weight_info.shape[2];

    const py::ssize_t kernel_width =
        weight_info.shape[3];

    if (
        input_channels
        != weight_input_channels
    ) {
        throw py::value_error(
            "Los canales de input y weight "
            "no son compatibles"
        );
    }

    if (
        output_height <= 0
        || output_width <= 0
    ) {
        throw py::value_error(
            "Las dimensiones de salida deben "
            "ser mayores que cero"
        );
    }

    py::array_t<float> output(
        {
            batch_size,
            output_channels,
            static_cast<py::ssize_t>(
                output_height
            ),
            static_cast<py::ssize_t>(
                output_width
            )
        }
    );

    py::buffer_info output_info =
        output.request();

    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    const auto* weight_pointer =
        static_cast<const float*>(
            weight_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    const std::string shape_str = shape_to_string(
        {batch_size, input_channels, input_height, input_width}
    );

    if (bias.is_none()) {
        {
            SCOPED_TIMER("conv2d", shape_str);

            conv2d_forward(
                input_pointer,
                weight_pointer,
                nullptr,
                output_pointer,
                static_cast<int>(batch_size),
                static_cast<int>(input_channels),
                static_cast<int>(input_height),
                static_cast<int>(input_width),
                static_cast<int>(output_channels),
                output_height,
                output_width,
                static_cast<int>(kernel_height),
                static_cast<int>(kernel_width),
                stride_height,
                stride_width,
                padding_height,
                padding_width,
                false
            );
        }

        return output;
    }

    FloatArray bias_array =
        FloatArray::ensure(bias);

    if (!bias_array) {
        throw py::type_error(
            "bias debe ser un arreglo NumPy "
            "float32 y contiguo, o None"
        );
    }

    const py::buffer_info bias_info =
        bias_array.request();

    if (
        bias_info.ndim != 1
        || bias_info.shape[0]
            != output_channels
    ) {
        throw py::value_error(
            "bias debe tener forma [out_channels]"
        );
    }

    const auto* bias_pointer =
        static_cast<const float*>(
            bias_info.ptr
        );

    {
        SCOPED_TIMER("conv2d", shape_str);

        conv2d_forward(
            input_pointer,
            weight_pointer,
            bias_pointer,
            output_pointer,
            static_cast<int>(batch_size),
            static_cast<int>(input_channels),
            static_cast<int>(input_height),
            static_cast<int>(input_width),
            static_cast<int>(output_channels),
            output_height,
            output_width,
            static_cast<int>(kernel_height),
            static_cast<int>(kernel_width),
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            true
        );
    }

    return output;
}

// Pointwise Conv2d
py::array_t<float> pointwise_conv2d_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const py::object& bias
)
{
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info weight_info =
        weight.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch, in_channels, height, width]"
        );
    }

    if (weight_info.ndim != 4) {
        throw py::value_error(
            "weight debe tener forma "
            "[out_channels, in_channels, 1, 1]"
        );
    }

    const py::ssize_t batch_size =
        input_info.shape[0];

    const py::ssize_t input_channels =
        input_info.shape[1];

    const py::ssize_t input_height =
        input_info.shape[2];

    const py::ssize_t input_width =
        input_info.shape[3];

    const py::ssize_t output_channels =
        weight_info.shape[0];

    if (
        weight_info.shape[1] != input_channels
        || weight_info.shape[2] != 1
        || weight_info.shape[3] != 1
    ) {
        throw py::value_error(
            "weight debe tener forma "
            "[out_channels, in_channels, 1, 1]"
        );
    }

    py::array_t<float> output(
        {
            batch_size,
            output_channels,
            input_height,
            input_width
        }
    );

    py::buffer_info output_info =
        output.request();

    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    const auto* weight_pointer =
        static_cast<const float*>(
            weight_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    const std::string shape_str = shape_to_string(
        {batch_size, input_channels, input_height, input_width}
    );

    if (bias.is_none()) {
        {
            SCOPED_TIMER("pointwise_conv", shape_str);

            pointwise_conv2d_forward(
                input_pointer,
                weight_pointer,
                nullptr,
                output_pointer,
                static_cast<int>(batch_size),
                static_cast<int>(input_channels),
                static_cast<int>(input_height),
                static_cast<int>(input_width),
                static_cast<int>(output_channels),
                false
            );
        }

        return output;
    }

    FloatArray bias_array =
        FloatArray::ensure(bias);

    if (!bias_array) {
        throw py::type_error(
            "bias debe ser un arreglo NumPy "
            "float32 y contiguo, o None"
        );
    }

    const py::buffer_info bias_info =
        bias_array.request();

    if (
        bias_info.ndim != 1
        || bias_info.shape[0] != output_channels
    ) {
        throw py::value_error(
            "bias debe tener forma [out_channels]"
        );
    }

    const auto* bias_pointer =
        static_cast<const float*>(
            bias_info.ptr
        );

    {
        SCOPED_TIMER("pointwise_conv", shape_str);

        pointwise_conv2d_forward(
            input_pointer,
            weight_pointer,
            bias_pointer,
            output_pointer,
            static_cast<int>(batch_size),
            static_cast<int>(input_channels),
            static_cast<int>(input_height),
            static_cast<int>(input_width),
            static_cast<int>(output_channels),
            true
        );
    }

    return output;
}

// Depthwise Conv2d

py::array_t<float> depthwise_conv2d_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const py::object& bias,
    int output_height,
    int output_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width
)
{
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info weight_info =
        weight.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch, channels, height, width]"
        );
    }

    if (weight_info.ndim != 4) {
        throw py::value_error(
            "weight debe tener forma "
            "[channels, 1, kernel_height, kernel_width]"
        );
    }

    const py::ssize_t batch_size =
        input_info.shape[0];

    const py::ssize_t channels =
        input_info.shape[1];

    const py::ssize_t input_height =
        input_info.shape[2];

    const py::ssize_t input_width =
        input_info.shape[3];

    const py::ssize_t kernel_height =
        weight_info.shape[2];

    const py::ssize_t kernel_width =
        weight_info.shape[3];

    if (
        weight_info.shape[0] != channels
        || weight_info.shape[1] != 1
    ) {
        throw py::value_error(
            "Para Depthwise, weight debe tener forma "
            "[channels, 1, kernel_height, kernel_width]"
        );
    }

    if (
        output_height <= 0
        || output_width <= 0
    ) {
        throw py::value_error(
            "Las dimensiones de salida deben "
            "ser mayores que cero"
        );
    }

    py::array_t<float> output(
        {
            batch_size,
            channels,
            static_cast<py::ssize_t>(output_height),
            static_cast<py::ssize_t>(output_width)
        }
    );

    py::buffer_info output_info =
        output.request();

    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    const auto* weight_pointer =
        static_cast<const float*>(
            weight_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    const std::string shape_str = shape_to_string(
        {batch_size, channels, input_height, input_width}
    );

    if (bias.is_none()) {
        {
            SCOPED_TIMER("depthwise_conv", shape_str);

            depthwise_conv2d_forward(
                input_pointer,
                weight_pointer,
                nullptr,
                output_pointer,
                static_cast<int>(batch_size),
                static_cast<int>(channels),
                static_cast<int>(input_height),
                static_cast<int>(input_width),
                output_height,
                output_width,
                static_cast<int>(kernel_height),
                static_cast<int>(kernel_width),
                stride_height,
                stride_width,
                padding_height,
                padding_width,
                false
            );
        }

        return output;
    }

    FloatArray bias_array =
        FloatArray::ensure(bias);

    if (!bias_array) {
        throw py::type_error(
            "bias debe ser un arreglo NumPy "
            "float32 y contiguo, o None"
        );
    }

    const py::buffer_info bias_info =
        bias_array.request();

    if (
        bias_info.ndim != 1
        || bias_info.shape[0] != channels
    ) {
        throw py::value_error(
            "bias debe tener forma [channels]"
        );
    }

    const auto* bias_pointer =
        static_cast<const float*>(
            bias_info.ptr
        );

    {
        SCOPED_TIMER("depthwise_conv", shape_str);

        depthwise_conv2d_forward(
            input_pointer,
            weight_pointer,
            bias_pointer,
            output_pointer,
            static_cast<int>(batch_size),
            static_cast<int>(channels),
            static_cast<int>(input_height),
            static_cast<int>(input_width),
            output_height,
            output_width,
            static_cast<int>(kernel_height),
            static_cast<int>(kernel_width),
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            true
        );
    }

    return output;
}

// BatchNorm2d
py::array_t<float> batchnorm2d_forward_binding(
    const FloatArray& input,
    const py::object& weight,
    const py::object& bias,
    const FloatArray& running_mean,
    const FloatArray& running_var,
    float eps,
    bool affine
)
{
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info mean_info =
        running_mean.request();

    const py::buffer_info var_info =
        running_var.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch, channels, height, width]"
        );
    }

    const py::ssize_t batch_size =
        input_info.shape[0];

    const py::ssize_t channels =
        input_info.shape[1];

    const py::ssize_t input_height =
        input_info.shape[2];

    const py::ssize_t input_width =
        input_info.shape[3];

    if (
        mean_info.ndim != 1
        || mean_info.shape[0] != channels
    ) {
        throw py::value_error(
            "running_mean debe tener forma [channels]"
        );
    }

    if (
        var_info.ndim != 1
        || var_info.shape[0] != channels
    ) {
        throw py::value_error(
            "running_var debe tener forma [channels]"
        );
    }

    py::array_t<float> output(
        input_info.shape
    );

    py::buffer_info output_info =
        output.request();

    const auto* input_pointer =
        static_cast<const float*>(input_info.ptr);

    const auto* mean_pointer =
        static_cast<const float*>(mean_info.ptr);

    const auto* var_pointer =
        static_cast<const float*>(var_info.ptr);

    auto* output_pointer =
        static_cast<float*>(output_info.ptr);

    const float* weight_pointer = nullptr;
    const float* bias_pointer = nullptr;

    FloatArray weight_array;
    FloatArray bias_array;

    if (affine) {
        weight_array = FloatArray::ensure(weight);

        if (!weight_array) {
            throw py::type_error(
                "weight debe ser un arreglo NumPy "
                "float32 y contiguo cuando affine=true"
            );
        }

        const py::buffer_info weight_info =
            weight_array.request();

        if (
            weight_info.ndim != 1
            || weight_info.shape[0] != channels
        ) {
            throw py::value_error(
                "weight debe tener forma [channels]"
            );
        }

        weight_pointer =
            static_cast<const float*>(
                weight_info.ptr
            );

        if (!bias.is_none()) {
            bias_array = FloatArray::ensure(bias);

            if (!bias_array) {
                throw py::type_error(
                    "bias debe ser un arreglo NumPy "
                    "float32 y contiguo, o None"
                );
            }

            const py::buffer_info bias_info =
                bias_array.request();

            if (
                bias_info.ndim != 1
                || bias_info.shape[0] != channels
            ) {
                throw py::value_error(
                    "bias debe tener forma [channels]"
                );
            }

            bias_pointer =
                static_cast<const float*>(
                    bias_info.ptr
                );
        }
    }

    const bool use_bias =
        affine && !bias.is_none();

    {
        const std::string shape_str = shape_to_string(
            {batch_size, channels, input_height, input_width}
        );

        SCOPED_TIMER("batchnorm", shape_str);

        batchnorm2d_forward(
            input_pointer,
            weight_pointer,
            bias_pointer,
            mean_pointer,
            var_pointer,
            output_pointer,
            static_cast<int>(batch_size),
            static_cast<int>(channels),
            static_cast<int>(input_height),
            static_cast<int>(input_width),
            eps,
            affine,
            use_bias
        );
    }

    return output;
}

//GlobalAvgPool2d
py::array_t<float> global_avgpool2d_forward_binding(
    const FloatArray& input
)
{
    const py::buffer_info input_info =
        input.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch, channels, height, width]"
        );
    }

    const py::ssize_t batch_size =
        input_info.shape[0];

    const py::ssize_t channels =
        input_info.shape[1];

    const py::ssize_t input_height =
        input_info.shape[2];

    const py::ssize_t input_width =
        input_info.shape[3];

    py::array_t<float> output(
        {
            batch_size,
            channels,
            static_cast<py::ssize_t>(1),
            static_cast<py::ssize_t>(1)
        }
    );

    py::buffer_info output_info =
        output.request();

    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    {
        const std::string shape_str = shape_to_string(
            {batch_size, channels, input_height, input_width}
        );

        SCOPED_TIMER("pooling", shape_str);

        global_avgpool2d_forward(
            input_pointer,
            output_pointer,
            static_cast<int>(batch_size),
            static_cast<int>(channels),
            static_cast<int>(input_height),
            static_cast<int>(input_width)
        );
    }

    return output;
}

//LayerAdd
py::array_t<float> layer_add_forward_binding(
    const FloatArray& input1,
    const FloatArray& input2,
    float alpha
)
{
    const py::buffer_info input1_info =
        input1.request();

    const py::buffer_info input2_info =
        input2.request();

    if (
        input1_info.ndim
        != input2_info.ndim
    ) {
        throw py::value_error(
            "input1 e input2 deben tener "
            "la misma forma"
        );
    }

    if (
        input1_info.shape
        != input2_info.shape
    ) {
        throw py::value_error(
            "input1 e input2 deben tener "
            "la misma forma"
        );
    }

    if (input1_info.size <= 0) {
        throw py::value_error(
            "Las entradas no pueden estar vacias"
        );
    }

    py::array_t<float> output(
        input1_info.shape
    );

    py::buffer_info output_info =
        output.request();

    const auto* input1_pointer =
        static_cast<const float*>(
            input1_info.ptr
        );

    const auto* input2_pointer =
        static_cast<const float*>(
            input2_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    {
        const std::string shape_str = shape_to_string(
            std::vector<py::ssize_t>(
                input1_info.shape.begin(),
                input1_info.shape.end()
            )
        );

        SCOPED_TIMER("layer_add", shape_str);

        layer_add_forward(
            input1_pointer,
            input2_pointer,
            output_pointer,
            static_cast<int>(input1_info.size),
            alpha
        );
    }

    return output;
}

//ReLU6
py::array_t<float> relu6_forward_binding(
    const FloatArray& input
)
{
    const py::buffer_info input_info =
        input.request();

    if (input_info.size <= 0) {
        throw py::value_error(
            "La entrada no puede estar vacia"
        );
    }

    py::array_t<float> output(
        input_info.shape
    );

    py::buffer_info output_info =
        output.request();

    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    {
        const std::string shape_str = shape_to_string(
            std::vector<py::ssize_t>(
                input_info.shape.begin(),
                input_info.shape.end()
            )
        );

        SCOPED_TIMER("activacion", shape_str);

        relu6_forward(
            input_pointer,
            output_pointer,
            static_cast<int>(input_info.size)
        );
    }

    return output;
}

PYBIND11_MODULE(cpp_kernels, module)
{
    module.def(
        "set_profile_enabled",
        [](bool enabled) {
            profiler_enabled() = enabled;
        },
        py::arg("enabled"),
        "Activa o desactiva los registros del perfilado"
    );

#ifdef PROFILE_KERNELS
    module.attr("profiling_compiled") = true;
#else
    module.attr("profiling_compiled") = false;
#endif

    module.doc() =
        "Kernels C++ para capas de PyTorch";

    module.def(
        "linear_forward",
        &linear_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        "Ejecuta una capa Linear usando arreglos NumPy float32"
    );

    module.def(
        "conv2d_forward",
        &conv2d_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("output_height"),
        py::arg("output_width"),
        py::arg("stride_height"),
        py::arg("stride_width"),
        py::arg("padding_height"),
        py::arg("padding_width"),
        "Ejecuta Conv2d para groups=1 y dilation=1"
    );

        module.def(
        "pointwise_conv2d_forward",
        &pointwise_conv2d_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        "Ejecuta una convolucion Pointwise 1x1"
    );

    module.def(
        "depthwise_conv2d_forward",
        &depthwise_conv2d_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("output_height"),
        py::arg("output_width"),
        py::arg("stride_height"),
        py::arg("stride_width"),
        py::arg("padding_height"),
        py::arg("padding_width"),
        "Ejecuta una convolucion Depthwise"
    );

        module.def(
        "batchnorm2d_forward",
        &batchnorm2d_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("running_mean"),
        py::arg("running_var"),
        py::arg("eps") = 1e-5f,
        py::arg("affine") = true,
        "Ejecuta BatchNorm2d en modo inferencia"
    );

    module.def(
        "global_avgpool2d_forward",
        &global_avgpool2d_forward_binding,
        py::arg("input"),
        "Ejecuta Global Average Pooling 2D"
    );

    module.def(
        "layer_add_forward",
        &layer_add_forward_binding,
        py::arg("input1"),
        py::arg("input2"),
        py::arg("alpha") = 1.0f,
        "Suma dos tensores elemento a elemento"
    );

    module.def(
        "relu6_forward",
        &relu6_forward_binding,
        py::arg("input"),
        "Ejecuta ReLU6 sobre un arreglo NumPy float32"
    );

    // ------------------------------------------------------
    // Profiler (std::chrono)
    // ------------------------------------------------------

    module.def(
        "get_and_clear_profile_records",
        []() {
            py::list result;

            for (const auto& record : profiler_records()) {
                py::dict entry;

                entry["etapa"] = record.etapa;
                entry["forma_tensor"] = record.forma_tensor;
                entry["tiempo_ms"] = record.tiempo_ms;

                result.append(entry);
            }

            profiler_records().clear();

            return result;
        },
        "Devuelve todos los registros de tiempo acumulados "
        "desde la última llamada y limpia el buffer"
    );

    module.def(
        "clear_profile_records",
        []() {
            profiler_records().clear();
        },
        "Limpia el buffer de registros de tiempo sin leerlos"
    );

}