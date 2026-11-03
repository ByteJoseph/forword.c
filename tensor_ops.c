#include "tensor_ops.h"
#include "simd.h"

#include <math.h>
#include <stdint.h>
#include <stddef.h>

int tensor_matvec_f32(
    const float *weights,
    const float *x,
    float *y,
    size_t rows,
    size_t cols)
{
    if (!weights || !x || !y || !rows || !cols)
        return -1;

    for (size_t r = 0; r < rows; ++r) {
        const float *row = weights + r * cols;
        y[r] = simd_dot_f32(row, x, cols);
    }

    return 0;
}

int tensor_add(
    const float *a,
    const float *b,
    float *out,
    size_t n)
{
    if (!a || !b || !out || !n)
        return -1;

    simd_add_f32(a, b, out, n);
    return 0;
}

int tensor_rms_norm(
    const float *x,
    const float *weight,
    float *out,
    size_t n,
    float epsilon)
{
    if (!x || !weight || !out || !n ||
        !isfinite(epsilon) || epsilon <= 0.0f)
        return -1;

    float ss = simd_sum_squares_f32(x, n);
    float scale = 1.0f / sqrtf((ss / (float)n) + epsilon);

    simd_mul_f32(x, weight, out, n);
    simd_scale_f32(out, scale, out, n);

    return 0;
}

int tensor_silu(
    const float *x,
    float *out,
    size_t n)
{
    if (!x || !out || !n)
        return -1;

    for (size_t i = 0; i < n; ++i) {
        float sigmoid;

        if (x[i] >= 0.0f) {
            sigmoid = 1.0f / (1.0f + expf(-x[i]));
        } else {
            float e = expf(x[i]);
            sigmoid = e / (1.0f + e);
        }

        out[i] = x[i] * sigmoid;
    }

    return 0;
}
