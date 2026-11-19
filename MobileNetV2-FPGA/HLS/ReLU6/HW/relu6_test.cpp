#include "relu6.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


static void reference_relu6(
    const float* input,
    float* output,
    int size
)
{
    for (int index = 0; index < size; index++) {

        const float value = input[index];

        if (value < 0.0f) {
            output[index] = 0.0f;
        }
        else if (value > 6.0f) {
            output[index] = 6.0f;
        }
        else {
            output[index] = value;
        }
    }
}


static bool run_test(
    const char* test_name,
    int size
)
{
    constexpr int MAX_SIZE = 77070336;

    if (size > MAX_SIZE) {
        std::cout << "\n" << test_name << "\n";
        std::cout << "ERROR: test exceeds kernel limits.\n";
        return false;
    }

    std::vector<float> input(size);
    std::vector<float> output(size, 0.0f);
    std::vector<float> expected(size, 0.0f);

    // Genera valores negativos, entre 0 y 6,
    // y mayores que 6.
    for (int index = 0; index < size; index++) {
        input[index] =
            static_cast<float>((index % 101) - 20)
            * 0.1f;
    }

    relu6_forward(
        input.data(),
        output.data(),
        size
    );

    reference_relu6(
        input.data(),
        expected.data(),
        size
    );

    double squared_error_sum = 0.0;
    float maximum_error = 0.0f;

    for (int index = 0; index < size; index++) {

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
            / static_cast<double>(size)
        );

    const float tolerance = 1.0e-5f;

    const bool passed =
        maximum_error <= tolerance;

    std::cout << "\n" << test_name << "\n";

    std::cout
        << "Elements     = "
        << size << "\n";

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

    // Caso pequeño:
    // verifica negativos, rango [0,6] y saturación > 6.
    all_tests_passed &= run_test(
        "Test 1 - Small ReLU6",
        128
    );

    // Tensor representativo de MobileNetV2:
    // [1, 96, 112, 112]
    all_tests_passed &= run_test(
        "Test 2 - MobileNetV2 ReLU6",
        96 * 112 * 112
    );

    // Caso con mayor cantidad de canales:
    // [1, 1280, 7, 7]
    all_tests_passed &= run_test(
        "Test 3 - MobileNetV2 1280 channels",
        1280 * 7 * 7
    );

    std::cout
        << "\nOverall result: "
        << (all_tests_passed ? "PASS" : "FAIL")
        << "\n";

    return all_tests_passed ? 0 : 1;
}