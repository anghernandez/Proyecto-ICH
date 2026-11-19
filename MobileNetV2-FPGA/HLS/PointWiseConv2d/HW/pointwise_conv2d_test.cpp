#include "pointwise_conv2d.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


static void reference_pointwise_conv2d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
)
{
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int output_channel = 0;
             output_channel < output_channels;
             output_channel++) {

            for (int output_row = 0;
                 output_row < input_height;
                 output_row++) {

                for (int output_column = 0;
                     output_column < input_width;
                     output_column++) {

                    float accumulated_value = 0.0f;

                    if (use_bias) {
                        accumulated_value = bias[output_channel];
                    }

                    for (int input_channel = 0;
                         input_channel < input_channels;
                         input_channel++) {

                        const int input_position =
                            (((batch_index * input_channels
                            + input_channel)
                            * input_height
                            + output_row)
                            * input_width)
                            + output_column;

                        const int weight_position =
                            (output_channel * input_channels)
                            + input_channel;

                        accumulated_value +=
                            input[input_position]
                            * weight[weight_position];
                    }

                    const int output_position =
                        (((batch_index * output_channels
                        + output_channel)
                        * input_height
                        + output_row)
                        * input_width)
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
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
)
{
    const int input_size =
        batch_size
        * input_channels
        * input_height
        * input_width;

    const int weight_size =
        output_channels
        * input_channels;

    const int bias_size =
        output_channels;

    const int output_size =
        batch_size
        * output_channels
        * input_height
        * input_width;

    // Tamaños máximos declarados mediante depth
    // en pointwise_conv2d.cpp.
    constexpr int MAX_INPUT_SIZE  = 12845056;
    constexpr int MAX_WEIGHT_SIZE = 409600;
    constexpr int MAX_BIAS_SIZE   = 1280;
    constexpr int MAX_OUTPUT_SIZE = 77070336;

    // Verificación de que la prueba se encuentra dentro
    // de los límites soportados por el kernel.
    if (input_size > MAX_INPUT_SIZE
        || weight_size > MAX_WEIGHT_SIZE
        || bias_size > MAX_BIAS_SIZE
        || output_size > MAX_OUTPUT_SIZE) {

        std::cout << "\n" << test_name << "\n";
        std::cout << "ERROR: test exceeds kernel limits.\n";

        return false;
    }

    // Los buffers enviados al kernel utilizan las
    // profundidades máximas declaradas en la interfaz.
    std::vector<float> input(MAX_INPUT_SIZE, 0.0f);
    std::vector<float> weight(MAX_WEIGHT_SIZE, 0.0f);
    std::vector<float> bias(MAX_BIAS_SIZE, 0.0f);
    std::vector<float> output(MAX_OUTPUT_SIZE, 0.0f);

    // La referencia solamente necesita el tamaño lógico
    // de la salida de esta prueba.
    std::vector<float> expected(output_size, 0.0f);

    for (int index = 0; index < input_size; index++) {
        input[index] =
            static_cast<float>((index % 17) - 8) * 0.05f;
    }

    for (int index = 0; index < weight_size; index++) {
        weight[index] =
            static_cast<float>((index % 11) - 5) * 0.02f;
    }

    for (int index = 0; index < bias_size; index++) {
        bias[index] =
            static_cast<float>(index - 3) * 0.01f;
    }

    pointwise_conv2d_forward(
        input.data(),
        weight.data(),
        bias.data(),
        output.data(),
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        use_bias
    );

    reference_pointwise_conv2d(
        input.data(),
        weight.data(),
        bias.data(),
        expected.data(),
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        use_bias
    );

    double squared_error_sum = 0.0;
    float maximum_error = 0.0f;

    for (int index = 0; index < output_size; index++) {

        const float error =
            std::fabs(output[index] - expected[index]);

        squared_error_sum +=
            static_cast<double>(error)
            * static_cast<double>(error);

        maximum_error =
            std::max(maximum_error, error);
    }

    const double rmse =
        std::sqrt(
            squared_error_sum
            / static_cast<double>(output_size)
        );

    const float tolerance = 1.0e-5f;

    const bool passed =
        maximum_error <= tolerance;

    std::cout << "\n" << test_name << "\n";

    std::cout
        << "Input shape  = "
        << batch_size << "x"
        << input_channels << "x"
        << input_height << "x"
        << input_width << "\n";

    std::cout
        << "Output shape = "
        << batch_size << "x"
        << output_channels << "x"
        << input_height << "x"
        << input_width << "\n";

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
    // Permite comprobar fácilmente la operación básica.
    // ---------------------------------------------------------

    all_tests_passed &= run_test(
        "Test 1 - Small pointwise with bias",
        1,      // batch_size
        3,      // input_channels
        4,      // input_height
        4,      // input_width
        8,      // output_channels
        true    // use_bias
    );

    // ---------------------------------------------------------
    // Test 2 (LOS ELIMINE)
    std::cout
        << "\nOverall result: "
        << (all_tests_passed ? "PASS" : "FAIL")
        << "\n";

    return all_tests_passed ? 0 : 1;
}