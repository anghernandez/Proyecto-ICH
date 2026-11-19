#include "avgpool2d.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define MAX_INPUT_SIZE  4014080
#define MAX_OUTPUT_SIZE 81920
#define EPSILON 1e-3f
#define EPSILON 1e-3f

static void avg_pool2d_reference(
    const float* input,
    float* output,
    int batch_size,
    int input_channels,
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
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
)
{
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++)
    {
        for (int channel_index = 0;
             channel_index < input_channels;
             channel_index++)
        {
            for (int output_row = 0;
                 output_row < output_height;
                 output_row++)
            {
                for (int output_column = 0;
                     output_column < output_width;
                     output_column++)
                {
                    const int window_start_row =
                        output_row * stride_height
                        - padding_height;

                    const int window_start_column =
                        output_column * stride_width
                        - padding_width;

                    int padded_end_row =
                        window_start_row + kernel_height;

                    if (padded_end_row >
                        input_height + padding_height)
                    {
                        padded_end_row =
                            input_height + padding_height;
                    }

                    int padded_end_column =
                        window_start_column + kernel_width;

                    if (padded_end_column >
                        input_width + padding_width)
                    {
                        padded_end_column =
                            input_width + padding_width;
                    }

                    int valid_start_row = window_start_row;

                    if (valid_start_row < 0)
                    {
                        valid_start_row = 0;
                    }

                    int valid_start_column = window_start_column;

                    if (valid_start_column < 0)
                    {
                        valid_start_column = 0;
                    }

                    int valid_end_row = padded_end_row;

                    if (valid_end_row > input_height)
                    {
                        valid_end_row = input_height;
                    }

                    int valid_end_column = padded_end_column;

                    if (valid_end_column > input_width)
                    {
                        valid_end_column = input_width;
                    }

                    const int pool_height =
                        padded_end_row - window_start_row;

                    const int pool_width =
                        padded_end_column - window_start_column;

                    const int pool_size =
                        pool_height * pool_width;

                    float accumulated_value = 0.0f;

                    for (int input_row = valid_start_row;
                         input_row < valid_end_row;
                         input_row++)
                    {
                        for (int input_column = valid_start_column;
                             input_column < valid_end_column;
                             input_column++)
                        {
                            const int input_position =
                                (
                                    (
                                        batch_index
                                        * input_channels
                                        + channel_index
                                    )
                                    * input_height
                                    + input_row
                                )
                                * input_width
                                + input_column;

                            accumulated_value +=
                                input[input_position];
                        }
                    }

                    const int valid_height =
                        valid_end_row - valid_start_row;

                    const int valid_width =
                        valid_end_column - valid_start_column;

                    const int valid_value_count =
                        valid_height * valid_width;

                    int divisor;

                    if (use_divisor_override)
                    {
                        divisor = divisor_override;
                    }
                    else if (count_include_pad)
                    {
                        divisor = pool_size;
                    }
                    else
                    {
                        divisor = valid_value_count;
                    }

                    const int output_position =
                        (
                            (
                                batch_index
                                * input_channels
                                + channel_index
                            )
                            * output_height
                            + output_row
                        )
                        * output_width
                        + output_column;

                    output[output_position] =
                        accumulated_value
                        / (float)divisor;
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
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
)
{
    static float input[MAX_INPUT_SIZE];
    static float output_hls[MAX_OUTPUT_SIZE];
    static float output_ref[MAX_OUTPUT_SIZE];

    const int input_size =
        batch_size
        * input_channels
        * input_height
        * input_width;

    const int output_size =
        batch_size
        * input_channels
        * output_height
        * output_width;

    if (input_size > MAX_INPUT_SIZE ||
        output_size > MAX_OUTPUT_SIZE)
    {
        std::printf(
            "%s: ERROR - buffer size exceeded\n",
            test_name
        );

        return false;
    }

    for (int i = 0; i < input_size; i++)
    {
        input[i] =
            ((float)((i % 31) - 15)) / 10.0f;
    }

    for (int i = 0; i < output_size; i++)
    {
        output_hls[i] = 0.0f;
        output_ref[i] = 0.0f;
    }

    avg_pool2d_reference(
        input,
        output_ref,
        batch_size,
        input_channels,
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
        count_include_pad,
        use_divisor_override,
        divisor_override
    );

    avg_pool2d_forward(
        input,
        output_hls,
        batch_size,
        input_channels,
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
        count_include_pad,
        use_divisor_override,
        divisor_override
    );

    double squared_error_sum = 0.0;
    float max_absolute_error = 0.0f;

    for (int i = 0; i < output_size; i++)
    {
        const float difference =
            output_hls[i] - output_ref[i];

        const float absolute_error =
            std::fabs(difference);

        squared_error_sum +=
            (double)difference
            * (double)difference;

        if (absolute_error > max_absolute_error)
        {
            max_absolute_error = absolute_error;
        }
    }

    const double rmse =
        std::sqrt(
            squared_error_sum
            / (double)output_size
        );

    std::printf("\n%s\n", test_name);
    std::printf("Input size  = %d\n", input_size);
    std::printf("Output size = %d\n", output_size);
    std::printf("RMSE        = %.8e\n", rmse);
    std::printf(
        "Max error   = %.8e\n",
        max_absolute_error
    );

    const bool passed =
        (rmse <= EPSILON) &&
        (max_absolute_error <= EPSILON);

    std::printf(
        "Result      = %s\n",
        passed ? "PASS" : "FAIL"
    );

    return passed;
}

int main()
{
    bool all_passed = true;

    // LeNet AvgPool2D #1
    all_passed &=
        run_test(
            "Test 1 - LeNet AvgPool2D 6x28x28 -> 6x14x14",
            64,
            6,
            28,
            28,
            14,
            14,
            2,
            2,
            2,
            2,
            0,
            0,
            false,
            false,
            0
        );

    // LeNet AvgPool2D #2
    all_passed &=
        run_test(
            "Test 2 - LeNet AvgPool2D 16x10x10 -> 16x5x5",
            1,
            16,
            10,
            10,
            5,
            5,
            2,
            2,
            2,
            2,
            0,
            0,
            false,
            false,
            0
        );

    // Caso general con padding
    all_passed &=
        run_test(
            "Test 3 - Padding, count_include_pad=false",
            1,
            1,
            3,
            3,
            4,
            4,
            2,
            2,
            1,
            1,
            1,
            1,
            false,
            false,
            0
        );

    // Mismo caso contando padding
    all_passed &=
        run_test(
            "Test 4 - Padding, count_include_pad=true",
            1,
            1,
            3,
            3,
            4,
            4,
            2,
            2,
            1,
            1,
            1,
            1,
            true,
            false,
            0
        );

        // MobileNetV2 Global Average Pooling
        //
        // Input:
        // [64, 1280, 7, 7]
        //
        // Output:
        // [64, 1280, 1, 1]
        //
        // Global Average Pooling se implementa mediante
        // AvgPool2D usando una ventana que cubre toda
        // la dimensión espacial de entrada.
        //
    all_passed &=
        run_test(
            "Test 5 - MobileNetV2 GlobalAvgPool 1280x7x7 -> 1280x1x1",
            64,     // batch_size
            1280,   // input_channels
            7,      // input_height
            7,      // input_width
            1,      // output_height
            1,      // output_width
            7,      // kernel_height
            7,      // kernel_width
            1,      // stride_height
            1,      // stride_width
            0,      // padding_height
            0,      // padding_width
            false,  // count_include_pad
            false,  // use_divisor_override
            0       // divisor_override
        );

    if (all_passed)
    {
        std::printf(
            "\nALL AVGPOOL2D TESTS PASSED\n"
        );

        return 0;
    }

    std::printf(
        "\nAVGPOOL2D TEST FAILED\n"
    );

    return 1;
}