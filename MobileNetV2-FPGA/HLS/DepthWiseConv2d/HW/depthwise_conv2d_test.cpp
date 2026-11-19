#include "depthwise_conv2d.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


static void reference_depthwise_conv2d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int channel = 0;
             channel < channels;
             channel++) {

            for (int output_row = 0;
                 output_row < output_height;
                 output_row++) {

                for (int output_column = 0;
                     output_column < output_width;
                     output_column++) {

                    float accumulated_value = 0.0f;

                    if (use_bias) {
                        accumulated_value = bias[channel];
                    }

                    for (int kernel_row = 0;
                         kernel_row < kernel_height;
                         kernel_row++) {

                        const int input_row =
                            output_row * stride_height
                            + kernel_row
                            - padding_height;

                        for (int kernel_column = 0;
                             kernel_column < kernel_width;
                             kernel_column++) {

                            const int input_column =
                                output_column * stride_width
                                + kernel_column
                                - padding_width;

                            if (
                                input_row >= 0
                                && input_row < input_height
                                && input_column >= 0
                                && input_column < input_width
                            ) {
                                const int input_position =
                                    (((batch_index * channels
                                    + channel)
                                    * input_height
                                    + input_row)
                                    * input_width)
                                    + input_column;

                                const int weight_position =
                                    ((channel * kernel_height
                                    + kernel_row)
                                    * kernel_width)
                                    + kernel_column;

                                accumulated_value +=
                                    input[input_position]
                                    * weight[weight_position];
                            }
                        }
                    }

                    const int output_position =
                        (((batch_index * channels
                        + channel)
                        * output_height
                        + output_row)
                        * output_width)
                        + output_column;

                    output[output_position] =
                        accumulated_value;
                }
            }
        }
    }
}


static bool run_test(
    const char* test_name,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    const int output_height =
        ((input_height + 2 * padding_height - kernel_height)
        / stride_height) + 1;

    const int output_width =
        ((input_width + 2 * padding_width - kernel_width)
        / stride_width) + 1;

    const int input_size =
        batch_size
        * channels
        * input_height
        * input_width;

    const int weight_size =
        channels
        * kernel_height
        * kernel_width;

    const int bias_size =
        channels;

    const int output_size =
        batch_size
        * channels
        * output_height
        * output_width;

    // Tamaños máximos declarados mediante depth
    // en depthwise_conv2d.cpp.
    constexpr int MAX_INPUT_SIZE  = 77070336;
    constexpr int MAX_WEIGHT_SIZE = 8640;
    constexpr int MAX_BIAS_SIZE   = 960;
    constexpr int MAX_OUTPUT_SIZE = 28901376;

    // Verificación de límites.
    if (
        input_size > MAX_INPUT_SIZE
        || weight_size > MAX_WEIGHT_SIZE
        || bias_size > MAX_BIAS_SIZE
        || output_size > MAX_OUTPUT_SIZE
    ) {
        std::cout << "\n" << test_name << "\n";
        std::cout << "ERROR: test exceeds kernel limits.\n";

        return false;
    }

    // Buffers enviados al kernel.
    std::vector<float> input(
        MAX_INPUT_SIZE,
        0.0f
    );

    std::vector<float> weight(
        MAX_WEIGHT_SIZE,
        0.0f
    );

    std::vector<float> bias(
        MAX_BIAS_SIZE,
        0.0f
    );

    std::vector<float> output(
        MAX_OUTPUT_SIZE,
        0.0f
    );

    // La referencia solo necesita el tamaño lógico.
    std::vector<float> expected(
        output_size,
        0.0f
    );

    for (int index = 0;
         index < input_size;
         index++) {

        input[index] =
            static_cast<float>(
                (index % 17) - 8
            ) * 0.05f;
    }

    for (int index = 0;
         index < weight_size;
         index++) {

        weight[index] =
            static_cast<float>(
                (index % 11) - 5
            ) * 0.02f;
    }

    for (int index = 0;
         index < bias_size;
         index++) {

        bias[index] =
            static_cast<float>(
                index - 3
            ) * 0.01f;
    }

    depthwise_conv2d_forward(
        input.data(),
        weight.data(),
        bias.data(),
        output.data(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        use_bias
    );

    reference_depthwise_conv2d(
        input.data(),
        weight.data(),
        bias.data(),
        expected.data(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        use_bias
    );

    double squared_error_sum = 0.0;
    float maximum_error = 0.0f;

    for (int index = 0;
         index < output_size;
         index++) {

        const float error =
            std::fabs(
                output[index]
                - expected[index]
            );

        squared_error_sum +=
            static_cast<double>(error)
            * static_cast<double>(error);

        maximum_error =
            std::max(
                maximum_error,
                error
            );
    }

    const double rmse =
        std::sqrt(
            squared_error_sum
            / static_cast<double>(
                output_size
            )
        );

    const float tolerance = 1.0e-5f;

    const bool passed =
        maximum_error <= tolerance;

    std::cout << "\n"
              << test_name
              << "\n";

    std::cout
        << "Input shape  = "
        << batch_size << "x"
        << channels << "x"
        << input_height << "x"
        << input_width << "\n";

    std::cout
        << "Output shape = "
        << batch_size << "x"
        << channels << "x"
        << output_height << "x"
        << output_width << "\n";

    std::cout
        << "Kernel       = "
        << kernel_height << "x"
        << kernel_width << "\n";

    std::cout
        << "Stride       = "
        << stride_height << "x"
        << stride_width << "\n";

    std::cout
        << "Padding      = "
        << padding_height << "x"
        << padding_width << "\n";

    std::cout
        << "Input size   = "
        << input_size << "\n";

    std::cout
        << "Weight size  = "
        << weight_size << "\n";

    std::cout
        << "Output size  = "
        << output_size << "\n";

    std::cout
        << std::scientific
        << std::setprecision(8);

    std::cout
        << "RMSE         = "
        << rmse << "\n";

    std::cout
        << "Max error    = "
        << maximum_error << "\n";

    std::cout
        << "Result       = "
        << (passed ? "PASS" : "FAIL")
        << "\n";

    return passed;
}


int main()
{
    bool all_tests_passed = true;

    // ---------------------------------------------------------
    // Test 1
    // Caso pequeño con bias.
    // Comprueba operación básica y padding.
    // ---------------------------------------------------------

    all_tests_passed &= run_test(
        "Test 1 - Small depthwise with bias",
        1,      // batch_size
        3,      // channels
        8,      // input_height
        8,      // input_width
        3,      // kernel_height
        3,      // kernel_width
        1,      // stride_height
        1,      // stride_width
        1,      // padding_height
        1,      // padding_width
        true    // use_bias
    );

    // ---------------------------------------------------------
    // Test 2
    // MobileNetV2: Depthwise con stride 2.
    //
    // [1, 96, 112, 112]
    // ->
    // [1, 96, 56, 56]
    // ---------------------------------------------------------

    all_tests_passed &= run_test(
        "Test 2 - MobileNetV2 depthwise stride 2",
        1,      // batch_size
        4,     // channels
        8,    // input_height
        8,    // input_width
        3,      // kernel_height
        3,      // kernel_width
        2,      // stride_height
        2,      // stride_width
        1,      // padding_height
        1,      // padding_width
        false   // use_bias
    );
    /*
    // ---------------------------------------------------------
    // Test 3
    // MobileNetV2: Depthwise con stride 1.
    //
    // [1, 144, 56, 56]
    // ->
    // [1, 144, 56, 56]
    // ---------------------------------------------------------

    all_tests_passed &= run_test(
        "Test 3 - MobileNetV2 depthwise stride 1",
        1,      // batch_size
        144,    // channels
        56,     // input_height
        56,     // input_width
        3,      // kernel_height
        3,      // kernel_width
        1,      // stride_height
        1,      // stride_width
        1,      // padding_height
        1,      // padding_width
        false   // use_bias
    );

    // ---------------------------------------------------------
    // Test 4
    // Mayor cantidad de canales de una Depthwise MobileNetV2.
    //
    // [1, 960, 7, 7]
    // ->
    // [1, 960, 7, 7]
    // ---------------------------------------------------------

    all_tests_passed &= run_test(
        "Test 4 - MobileNetV2 depthwise 960 channels",
        1,      // batch_size
        960,    // channels
        7,      // input_height
        7,      // input_width
        3,      // kernel_height
        3,      // kernel_width
        1,      // stride_height
        1,      // stride_width
        1,      // padding_height
        1,      // padding_width
        false   // use_bias
    ); */

    std::cout
        << "\nOverall result: "
        << (all_tests_passed ? "PASS" : "FAIL")
        << "\n";

    return all_tests_passed ? 0 : 1;
}