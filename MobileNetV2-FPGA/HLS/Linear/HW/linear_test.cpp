#include "linear.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


static void linear_reference(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    bool use_bias
)
{
    for (int sample_index = 0;
         sample_index < batch_size;
         sample_index++)
    {
        for (int output_index = 0;
             output_index < out_features;
             output_index++)
        {
            float accumulated_value = 0.0f;

            for (int input_index = 0;
                 input_index < in_features;
                 input_index++)
            {
                const int input_position =
                    sample_index * in_features
                    + input_index;

                const int weight_position =
                    output_index * in_features
                    + input_index;

                accumulated_value +=
                    input[input_position]
                    * weight[weight_position];
            }

            if (use_bias)
            {
                accumulated_value +=
                    bias[output_index];
            }

            const int output_position =
                sample_index * out_features
                + output_index;

            output[output_position] =
                accumulated_value;
        }
    }
}


static bool run_test(
    const char* test_name,
    int batch_size,
    int in_features,
    int out_features,
    bool use_bias
)
{
    constexpr int MAX_INPUT_SIZE  = 81920;
    constexpr int MAX_WEIGHT_SIZE = 1280000;
    constexpr int MAX_BIAS_SIZE   = 1000;
    constexpr int MAX_OUTPUT_SIZE = 64000;

    const int input_size =
        batch_size * in_features;

    const int weight_size =
        out_features * in_features;

    const int bias_size =
        out_features;

    const int output_size =
        batch_size * out_features;


    if (input_size > MAX_INPUT_SIZE ||
        weight_size > MAX_WEIGHT_SIZE ||
        bias_size > MAX_BIAS_SIZE ||
        output_size > MAX_OUTPUT_SIZE)
    {
        std::cout
            << "\n"
            << test_name
            << "\nERROR: test exceeds kernel limits.\n";

        return false;
    }


    std::vector<float> input(input_size);
    std::vector<float> weight(weight_size);
    std::vector<float> bias(bias_size);

    std::vector<float> output_hls(
        output_size,
        0.0f
    );

    std::vector<float> output_ref(
        output_size,
        0.0f
    );


    // Datos deterministas de prueba
    for (int index = 0;
         index < input_size;
         index++)
    {
        input[index] =
            static_cast<float>(
                (index % 17) - 8
            ) * 0.05f;
    }


    for (int index = 0;
         index < weight_size;
         index++)
    {
        weight[index] =
            static_cast<float>(
                (index % 13) - 6
            ) * 0.02f;
    }


    for (int index = 0;
         index < bias_size;
         index++)
    {
        bias[index] =
            static_cast<float>(
                (index % 7) - 3
            ) * 0.01f;
    }


    // HLS
    linear_forward(
        input.data(),
        weight.data(),
        bias.data(),
        output_hls.data(),
        batch_size,
        in_features,
        out_features,
        use_bias
    );


    // Referencia software
    linear_reference(
        input.data(),
        weight.data(),
        bias.data(),
        output_ref.data(),
        batch_size,
        in_features,
        out_features,
        use_bias
    );


    double squared_error_sum = 0.0;
    float maximum_error = 0.0f;


    for (int index = 0;
         index < output_size;
         index++)
    {
        const float error =
            std::fabs(
                output_hls[index]
                - output_ref[index]
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
            / static_cast<double>(output_size)
        );


    constexpr float TOLERANCE = 1.0e-3f;

    const bool passed =
        rmse <= TOLERANCE &&
        maximum_error <= TOLERANCE;


    std::cout
        << "\n"
        << test_name
        << "\n";

    std::cout
        << "Batch size   = "
        << batch_size << "\n";

    std::cout
        << "In features  = "
        << in_features << "\n";

    std::cout
        << "Out features = "
        << out_features << "\n";

    std::cout
        << "Use bias     = "
        << (use_bias ? "true" : "false")
        << "\n";

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

/*
    // ---------------------------------------------------------
    // Test 1
    // Linear utilizado en LeNet-5
    //
    // 400 -> 120
    // ---------------------------------------------------------

    all_tests_passed &=
        run_test(
            "Test 1 - LeNet Linear 400 -> 120",
            1,
            400,
            120,
            true
        );


    // ---------------------------------------------------------
    // Test 2
    // Mismo Linear de LeNet con batch máximo
    // ---------------------------------------------------------

    all_tests_passed &=
        run_test(
            "Test 2 - LeNet Linear batch 64",
            64,
            400,
            120,
            true
        );

*/
    // ---------------------------------------------------------
    // Test 3
    // MobileNetV2 classifier
    //
    // Después de Global Average Pooling:
    //
    // [N, 1280, 1, 1]
    //        ↓
    // [N, 1280]
    //        ↓
    // Linear
    //        ↓
    // [N, 1000]
    // ---------------------------------------------------------

    all_tests_passed &=
        run_test(
            "Test 3 - MobileNetV2 Linear 1280 -> 1000",
            1,
            1280,
            1000,
            true
        );

/*
    // ---------------------------------------------------------
    // Test 4
    // MobileNetV2 con batch máximo
    // ---------------------------------------------------------

    all_tests_passed &=
        run_test(
            "Test 4 - MobileNetV2 Linear batch 64",
            64,
            1280,
            1000,
            true
        );


    // ---------------------------------------------------------
    // Test 5
    // Comprobar configuración sin bias
    // ---------------------------------------------------------

    all_tests_passed &=
        run_test(
            "Test 5 - Linear without bias",
            4,
            1280,
            1000,
            false
        );
*/

    std::cout
        << "\nOverall result: "
        << (
            all_tests_passed
            ? "PASS"
            : "FAIL"
        )
        << "\n";


    return all_tests_passed ? 0 : 1;
}