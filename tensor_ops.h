#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stddef.h>

int tensor_matvec_f32(
    const float *weights,
    const float *x,
    float *y,
    size_t rows,
    size_t cols
);

int tensor_add(
    const float *a,
    const float *b,
    float *out,
    size_t n
);

int tensor_rms_norm(
    const float *x,
    const float *weight,
    float *out,
    size_t n,
    float epsilon
);

int tensor_silu(
    const float *x,
    float *out,
    size_t n
);

#endif