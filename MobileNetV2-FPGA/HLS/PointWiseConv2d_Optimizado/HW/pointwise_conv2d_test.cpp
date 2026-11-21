#include "pointwise_conv2d.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

static void reference(const float* x, const float* w, const float* b,
                      float* y, int N, int C, int H, int W, int O, bool bias) {
    const int P = H * W;
    for (int n = 0; n < N; ++n)
        for (int o = 0; o < O; ++o)
            for (int p = 0; p < P; ++p) {
                float sum = bias ? b[o] : 0.0f;
                for (int c = 0; c < C; ++c)
                    sum += x[(n * C + c) * P + p] * w[o * C + c];
                y[(n * O + o) * P + p] = sum;
            }
}

static bool check(int N, int C, int H, int W, int O, bool bias) {
    const int in_size = N * C * H * W;
    const int out_size = N * O * H * W;
    std::vector<float> x(in_size), w(O * C), b(O), y(out_size), ref(out_size);
    for (int i = 0; i < in_size; ++i) x[i] = ((i * 7) % 23 - 11) * 0.025f;
    for (int i = 0; i < O * C; ++i) w[i] = ((i * 3) % 29 - 14) * 0.0125f;
    for (int i = 0; i < O; ++i) b[i] = (i % 7 - 3) * 0.03f;

    pointwise_conv2d_forward(x.data(), w.data(), b.data(), y.data(),
                             N, C, H, W, O, bias);
    reference(x.data(), w.data(), b.data(), ref.data(), N, C, H, W, O, bias);
    float max_abs = 0.0f, max_rel = 0.0f;
    bool passed = true;
    for (int i = 0; i < out_size; ++i) {
        float delta = std::abs(y[i] - ref[i]);
        float scale = std::max(1.0f, std::abs(ref[i]));
        max_abs = std::max(max_abs, delta);
        max_rel = std::max(max_rel, delta / scale);
        if (delta > 1e-5f * scale) passed = false;
    }
    std::cout << (passed ? "PASS" : "FAIL")
              << " N=" << N << " Cin=" << C << " H=" << H << " W=" << W
              << " Cout=" << O << " bias=" << bias
              << " max_abs=" << max_abs << " max_scaled=" << max_rel << '\n';
    return passed;
}

int main() {
    bool ok = true;
    ok &= check(1, 65, 2, 5, 3, true);       // pequeno: inferior al MIN_RUN

    return ok ? 0 : 1;
}
