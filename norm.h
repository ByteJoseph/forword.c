#ifndef NORM_H
#define NORM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
 * weight may be NULL (no scale, plain normalisation).
 */
void rms_norm(const float *x, const float *weight, float *out, size_t n, float eps);

/*
 * LayerNorm: out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
 * bias may be NULL.
 */
void layer_norm(const float *x, const float *weight, const float *bias,
                float *out, size_t n, float eps);

#ifdef __cplusplus
}
#endif
#endif /* NORM_H */
