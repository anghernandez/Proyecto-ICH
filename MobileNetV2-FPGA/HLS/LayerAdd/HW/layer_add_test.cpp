#include "layer_add.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


static void reference_layer_add(
    const float* input1,
    const float* input2,
    float* output,
    int size,
    float alpha
)
{
    for (int index = 0;
         index < size;
         index++) {

        output[index] =
            input1[index]
            + alpha * input2[index];
    }
}


static bool run_test(
    const char* test_name,
    int size,
    float alpha
)
{
    constexpr int MAX_SIZE = 4816896;

    if (size > MAX_SIZE) {

        std::cout
            << "\n"
            << test_name
            << "\nERROR: test exceeds kernel limits.\n";

        return false;
    }

    std::vector<float> input1(size);
    std::vector<float> input2(size);

    std::vector<float> output(
        size,
        0.0f
    );

    std::vector<float> expected(
        size,
        0.0f
    );

    // Datos deterministas con valores
    // positivos y negativos.
    for (int index = 0;
         index < size;
         index++) {

        input1[index] =
            static_cast<float>(
                (index % 23) - 11
            ) * 0.1f;

        input2[index] =
            static_cast<float>(
                (index % 17) - 8
            ) * 0.05f;
    }

    layer_add_forward(
        input1.data(),
        input2.data(),
        output.data(),
        size,
        alpha
    );

    reference_layer_add(
        input1.data(),
        input2.data(),
        expected.data(),
        size,
        alpha
    );

    double squared_error_sum = 0.0;
    float maximum_error = 0.0f;

    for (int index = 0;
         index < size;
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
            / static_cast<double>(size)
        );

    const float tolerance = 1.0e-5f;

    const bool passed =
        maximum_error <= tolerance;

    std::cout
        << "\n"
        << test_name
        << "\n";

    std::cout
        << "Elements     = "
        << size << "\n";

    std::cout
        << "Alpha        = "
        << alpha << "\n";

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

    all_tests_passed &= run_test(
        "LayerAdd C/RTL validation",
        64,
        1.0f
    );

    std::cout
        << "\nOverall result: "
        << (all_tests_passed ? "PASS" : "FAIL")
        << "\n";

    return all_tests_passed ? 0 : 1;
}
