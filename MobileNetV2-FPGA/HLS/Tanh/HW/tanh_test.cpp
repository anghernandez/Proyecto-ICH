#include <iostream>
#include <cmath>
#include "tanh.hpp"

int main()
{
    constexpr int MAX_SIZE = 32;
    constexpr double TOLERANCE = 1e-5;

    const int sizes[] = {1, 7, 32};

    const float values[] = {
         0.0f,
        -0.0001f,  0.0001f,
        -0.1f,     0.1f,
        -1.0f,     1.0f,
        -2.0f,     2.0f,
        -5.0f,     5.0f,
        -10.0f,   10.0f
    };

    constexpr int VALUE_COUNT =
        sizeof(values) / sizeof(values[0]);

    float input[MAX_SIZE];
    float output[MAX_SIZE];

    bool all_pass = true;

    for (int size : sizes) {
        for (int i = 0; i < size; ++i) {
            input[i] = values[i % VALUE_COUNT];

            // Permite detectar una salida que no fue escrita.
            output[i] = 123.0f;
        }

        tanh_forward(input, output, size);

        double squared_error_sum = 0.0;
        double max_error = 0.0;
        bool pass = true;
        bool finite = true;

        for (int i = 0; i < size; ++i) {
            const double reference =
                std::tanh(static_cast<double>(input[i]));

            if (!std::isfinite(output[i])) {
                finite = false;
                pass = false;
                continue;
            }

            const double error =
                static_cast<double>(output[i]) - reference;

            const double absolute_error = std::fabs(error);

            squared_error_sum += error * error;

            if (absolute_error > max_error)
                max_error = absolute_error;

            if (absolute_error > TOLERANCE)
                pass = false;
        }

        std::cout << "size=" << size;

        if (finite) {
            const double rmse =
                std::sqrt(squared_error_sum / size);

            std::cout << " | RMSE=" << rmse
                      << " | max_error=" << max_error;
        } else {
            std::cout << " | salida no finita";
        }

        std::cout << " | " << (pass ? "PASS" : "FAIL")
                  << '\n';

        all_pass = all_pass && pass;
    }

    return all_pass ? 0 : 1;
}