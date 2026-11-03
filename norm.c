#define _POSIX_C_SOURCE 200809L
#include "norm.h"
#include <math.h>
#include <stddef.h>

void rms_norm(const float *x, const float *weight, float *out, size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    if (weight) {
        for (size_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
    } else {
        for (size_t i = 0; i < n; i++) out[i] = x[i] * scale;
    }
}

void layer_norm(const float *x, const float *weight, const float *bias,
                float *out, size_t n, float eps)
{
    /* Compute mean */
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) mean += x[i];
    mean /= n;
    /* Compute variance */
    double var = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    float scale = 1.0f / sqrtf((float)var + eps);
    for (size_t i = 0; i < n; i++) {
        out[i] = (float)((x[i] - mean) * scale);
        if (weight) out[i] *= weight[i];
        if (bias)   out[i] += bias[i];
    }
}
