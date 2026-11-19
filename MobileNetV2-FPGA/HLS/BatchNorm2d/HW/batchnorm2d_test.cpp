#include "batchnorm2d.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


static void reference_batchnorm2d(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    float eps,
    bool affine,
    bool use_bias
)
{
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int channel_index = 0;
             channel_index < channels;
             channel_index++) {

            const float mean =
                running_mean[channel_index];

            const float variance =
                running_var[channel_index];

            const float denominator =
                std::sqrt(variance + eps);

            float gamma = 1.0f;
            float beta = 0.0f;

            if (affine) {

                gamma =
                    weight[channel_index];

                if (use_bias) {

                    beta =
                        bias[channel_index];
                }
            }

            for (int row_index = 0;
                 row_index < input_height;
                 row_index++) {

                for (int column_index = 0;
                     column_index < input_width;
                     column_index++) {

                    const int position =
                        (((batch_index * channels
                        + channel_index)
                        * input_height
                        + row_index)
                        * input_width)
                        + column_index;

                    const float normalized_value =
                        (input[position] - mean)
                        / denominator;

                    output[position] =
                        gamma
                        * normalized_value
                        + beta;
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
    float eps,
    bool affine,
    bool use_bias
)
{
    const int tensor_size =
        batch_size
        * channels
        * input_height
        * input_width;

    std::vector<float> input(
        tensor_size
    );

    std::vector<float> weight(
        channels
    );

    std::vector<float> bias(
        channels
    );

    std::vector<float> running_mean(
        channels
    );

    std::vector<float> running_var(
        channels
    );

    std::vector<float> output(
        tensor_size,
        0.0f
    );

    std::vector<float> expected(
        tensor_size,
        0.0f
    );


    // ---------------------------------------------------------
    // Datos de entrada deterministas.
    // ---------------------------------------------------------

    for (int index = 0;
         index < tensor_size;
         index++) {

        input[index] =
            static_cast<float>(
                (index % 31) - 15
            ) * 0.1f;
    }


    // ---------------------------------------------------------
    // Parámetros distintos para cada canal.
    // ---------------------------------------------------------

    for (int channel = 0;
         channel < channels;
         channel++) {

        weight[channel] =
            0.5f
            + static_cast<float>(
                channel % 7
            ) * 0.1f;

        bias[channel] =
            static_cast<float>(
                (channel % 5) - 2
            ) * 0.05f;

        running_mean[channel] =
            static_cast<float>(
                (channel % 9) - 4
            ) * 0.1f;

        // Varianza siempre positiva.
        running_var[channel] =
            0.5f
            + static_cast<float>(
                channel % 11
            ) * 0.1f;
    }


    // ---------------------------------------------------------
    // Kernel HLS.
    // ---------------------------------------------------------

    batchnorm2d_forward(
        input.data(),
        weight.data(),
        bias.data(),
        running_mean.data(),
        running_var.data(),
        output.data(),
        batch_size,
        channels,
        input_height,
        input_width,
        eps,
        affine,
        use_bias
    );


    // ---------------------------------------------------------
    // Implementación de referencia.
    // ---------------------------------------------------------

    reference_batchnorm2d(
        input.data(),
        weight.data(),
        bias.data(),
        running_mean.data(),
        running_var.data(),
        expected.data(),
        batch_size,
        channels,
        input_height,
        input_width,
        eps,
        affine,
        use_bias
    );


    // ---------------------------------------------------------
    // Comparación numérica.
    // ---------------------------------------------------------

    double squared_error_sum = 0.0;
    float maximum_error = 0.0f;

    for (int index = 0;
         index < tensor_size;
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
                tensor_size
            )
        );

    const float tolerance = 1.0e-5f;

    const bool passed =
        maximum_error <= tolerance;


    // ---------------------------------------------------------
    // Resultados.
    // ---------------------------------------------------------

    std::cout
        << "\n"
        << test_name
        << "\n";

    std::cout
        << "Shape        = "
        << batch_size << "x"
        << channels << "x"
        << input_height << "x"
        << input_width << "\n";

    std::cout
        << "Elements     = "
        << tensor_size << "\n";

    std::cout
        << "eps          = "
        << std::scientific
        << eps << "\n";

    std::cout
        << "affine       = "
        << (affine ? "true" : "false")
        << "\n";

    std::cout
        << "use_bias     = "
        << (use_bias ? "true" : "false")
        << "\n";

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
    // C/RTL Co-Simulation
    //
    // Shape:
    // [batch, channels, height, width]
    // [1, 3, 4, 4]
    //
    // Tensor:
    // 1 * 3 * 4 * 4 = 48 elementos
    //
    // Parámetros:
    // 3 canales
    // ---------------------------------------------------------

    all_tests_passed &= run_test(
        "BatchNorm2d C/RTL validation",
        1,          // batch
        3,          // channels
        4,          // height
        4,          // width
        1.0e-5f,    // eps
        true,       // affine
        true        // use_bias
    );


    std::cout
        << "\nOverall result: "
        << (all_tests_passed ? "PASS" : "FAIL")
        << "\n";


    return all_tests_passed ? 0 : 1;
}