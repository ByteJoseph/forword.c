#define _POSIX_C_SOURCE 200809L
#include "norm.h"
#include "simd.h"
#include <math.h>
#include <stddef.h>

void rms_norm(const float *x, const float *weight, float *out, size_t n, float eps) {
    float ss = simd_sum_squares_f32(x, n);
    float scale = 1.0f / sqrtf((ss / (float)n) + eps);
    if (weight) {
        simd_mul_f32(x, weight, out, n);
        simd_scale_f32(out, scale, out, n);
    } else {
        simd_scale_f32(x, scale, out, n);
    }
}

void layer_norm(const float *x, const float *weight, const float *bias,
                float *out, size_t n, float eps)
{
    /* Compute mean */
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += x[i];
    float mean = (float)(sum / (double)n);

    /* Compute variance */
    double var = 0.0;
    for (size_t i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += (double)d * d;
    }
    float scale = 1.0f / sqrtf((float)(var / (double)n) + eps);

    for (size_t i = 0; i < n; i++) {
        float val = (x[i] - mean) * scale;
        if (weight) val *= weight[i];
        if (bias)   val += bias[i];
        out[i] = val;
    }
}
