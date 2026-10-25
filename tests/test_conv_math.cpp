// ============================================================
// Verificación de los kernels CUDA sin GPU.
//
// Ejecuta en CPU lo que calcula cada hilo (kernels/cuda/
// conv_math.cuh), recorriendo la misma grilla de bloques e hilos
// que lanzan los .cu, y compara contra los kernels CPU
// originales (kernels/*.cpp) en las 52 convoluciones de
// MobileNetV2, con entradas y pesos aleatorios.
//
// Compilar y ejecutar desde la raíz del repo:
//   g++ -O2 -std=c++14 -Ikernels -Ikernels/cuda
//       tests/test_conv_math.cpp kernels/Conv2d.cpp
//       kernels/Pointwise_Conv2d.cpp kernels/Depthwise_Conv2d.cpp
//       -o test_conv_math && ./test_conv_math
//   (todo en una sola línea)
// ============================================================

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "Conv2d.hpp"
#include "Depthwise_Conv2d.hpp"
#include "Pointwise_Conv2d.hpp"
#include "conv_math.cuh"

static const unsigned int THREADS = 256;   // = HILOS_POR_BLOQUE

static std::mt19937 generator(12345);

static std::vector<float> random_vector(std::size_t count)
{
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> values(count);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
}

static double max_difference(const std::vector<float>& a,
                             const std::vector<float>& b)
{
    double maximum = 0.0;
    for (std::size_t i = 0; i < a.size(); i++) {
        maximum = std::fmax(maximum, std::fabs(double(a[i]) - b[i]));
    }
    return maximum;
}

static unsigned int blocks_for(std::size_t count)
{
    return static_cast<unsigned int>((count + THREADS - 1) / THREADS);
}

static int failures = 0;
static double worst = 0.0;

static void report(const char* name, int batch, int channels, int height,
                   int width, double difference)
{
    const bool ok = difference <= 1e-4;
    worst = std::fmax(worst, difference);
    if (!ok) {
        failures++;
    }
    std::printf("%-15s %dx%dx%dx%d  max|dif| = %.3e  %s\n", name, batch,
                channels, height, width, difference, ok ? "OK" : "FALLA");
}

static void test_pointwise(int batch, int in_c, int out_c, int h, int w,
                           bool bias)
{
    const int pixels = h * w;
    std::vector<float> input = random_vector(std::size_t(batch) * in_c * pixels);
    std::vector<float> weight = random_vector(std::size_t(out_c) * in_c);
    std::vector<float> bias_v = random_vector(out_c);
    std::vector<float> expected(std::size_t(batch) * out_c * pixels);
    std::vector<float> actual(expected.size(), NAN);

    pointwise_conv2d_forward(input.data(), weight.data(), bias_v.data(),
                             expected.data(), batch, in_c, h, w, out_c, bias);

    // Grilla del .cu: (bloques sobre píxeles, canales de salida, lote).
    for (int z = 0; z < batch; z++)
        for (int y = 0; y < out_c; y++)
            for (unsigned int x = 0; x < blocks_for(pixels); x++)
                for (unsigned int t = 0; t < THREADS; t++)
                    pointwise_conv2d_thread(
                        input.data(), weight.data(), bias_v.data(),
                        actual.data(), z, y, int(x * THREADS + t),
                        in_c, out_c, pixels, bias);

    report("pointwise_conv", batch, in_c, h, w,
           max_difference(expected, actual));
}

static void test_depthwise(int batch, int c, int h, int w, int stride,
                           bool bias)
{
    const int k = 3, pad = 1;
    const int oh = (h + 2 * pad - k) / stride + 1;
    const int ow = (w + 2 * pad - k) / stride + 1;
    std::vector<float> input = random_vector(std::size_t(batch) * c * h * w);
    std::vector<float> weight = random_vector(std::size_t(c) * k * k);
    std::vector<float> bias_v = random_vector(c);
    std::vector<float> expected(std::size_t(batch) * c * oh * ow);
    std::vector<float> actual(expected.size(), NAN);

    depthwise_conv2d_forward(input.data(), weight.data(), bias_v.data(),
                             expected.data(), batch, c, h, w, oh, ow, k, k,
                             stride, stride, pad, pad, bias);

    const std::size_t launched =
        std::size_t(blocks_for(expected.size())) * THREADS;
    for (std::size_t index = 0; index < launched; index++)
        depthwise_conv2d_thread(input.data(), weight.data(), bias_v.data(),
                                actual.data(), index, batch, c, h, w, oh, ow,
                                k, k, stride, stride, pad, pad, bias);

    report("depthwise_conv", batch, c, h, w,
           max_difference(expected, actual));
}

static void test_conv2d(int batch, bool bias)
{
    const int in_c = 3, out_c = 32, h = 224, w = 224, k = 3, s = 2, p = 1;
    const int oh = (h + 2 * p - k) / s + 1, ow = (w + 2 * p - k) / s + 1;
    std::vector<float> input = random_vector(std::size_t(batch) * in_c * h * w);
    std::vector<float> weight = random_vector(std::size_t(out_c) * in_c * k * k);
    std::vector<float> bias_v = random_vector(out_c);
    std::vector<float> expected(std::size_t(batch) * out_c * oh * ow);
    std::vector<float> actual(expected.size(), NAN);

    conv2d_forward(input.data(), weight.data(), bias_v.data(),
                   expected.data(), batch, in_c, h, w, out_c, oh, ow, k, k,
                   s, s, p, p, bias);

    const std::size_t launched =
        std::size_t(blocks_for(expected.size())) * THREADS;
    for (std::size_t index = 0; index < launched; index++)
        conv2d_thread(input.data(), weight.data(), bias_v.data(),
                      actual.data(), index, batch, in_c, h, w, out_c, oh, ow,
                      k, k, s, s, p, p, bias);

    report("conv2d", batch, in_c, h, w, max_difference(expected, actual));
}

int main()
{
    // Conv2d inicial (sin sesgo, como en MobileNetV2) y variantes.
    test_conv2d(1, false);
    test_conv2d(2, true);

    // Bloques invertidos de MobileNetV2: (t, c, n, s).
    const int config[7][4] = {
        {1, 16, 1, 1}, {6, 24, 2, 2}, {6, 32, 3, 2}, {6, 64, 4, 2},
        {6, 96, 3, 1}, {6, 160, 3, 2}, {6, 320, 1, 1},
    };

    int in_c = 32, size = 112;
    int pointwise = 0, depthwise = 0;

    for (const auto& block : config) {
        for (int i = 0; i < block[2]; i++) {
            const int stride = (i == 0) ? block[3] : 1;
            const int hidden = in_c * block[0];

            if (block[0] != 1) {
                test_pointwise(1, in_c, hidden, size, size, false);
                pointwise++;
            }
            test_depthwise(1, hidden, size, size, stride, false);
            depthwise++;

            const int out_size = (size + 2 - 3) / stride + 1;
            test_pointwise(1, hidden, block[1], out_size, out_size, false);
            pointwise++;

            in_c = block[1];
            size = out_size;
        }
    }
    test_pointwise(1, 320, 1280, 7, 7, false);
    pointwise++;

    // Lote > 1 y sesgo, para cubrir esos índices.
    test_pointwise(2, 24, 144, 56, 56, true);
    test_depthwise(2, 144, 56, 56, 2, true);

    std::printf("\nCapas de MobileNetV2: %d pointwise, %d depthwise, 1 conv2d\n",
                pointwise, depthwise);
    std::printf("Peor diferencia: %.3e | Fallas: %d\n", worst, failures);
    return failures == 0 ? 0 : 1;
}
