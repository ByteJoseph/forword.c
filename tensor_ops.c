#include "tensor_ops.h"

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
        double sum = 0.0;

        for (size_t c = 0; c < cols; ++c)
            sum += (double)weights[r * cols + c] * x[c];

        y[r] = (float)sum;
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

    for (size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];

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

    double sum = 0.0;

    for (size_t i = 0; i < n; ++i)
        sum += (double)x[i] * x[i];

    float scale = 1.0f / sqrtf((float)(sum / n) + epsilon);

    for (size_t i = 0; i < n; ++i)
        out[i] = x[i] * scale * weight[i];

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