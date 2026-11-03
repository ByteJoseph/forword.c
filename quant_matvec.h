#ifndef QUANT_MATVEC_H
#define QUANT_MATVEC_H

#include <stddef.h>
#include <stdint.h>

int quant_matvec(
    const void *weights,
    size_t weights_size,
    uint32_t ggml_type,
    const float *input,
    float *output,
    size_t input_size,
    size_t output_size
);

#endif