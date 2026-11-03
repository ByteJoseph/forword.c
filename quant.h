
#ifndef QUANT_H
#define QUANT_H

#include <stddef.h>
#include <stdint.h>

/*
 * GGML Q4_K block:
 * 256 values, packed into 4-bit quantized values.
 * The actual structure must match the GGML file layout.
 */
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

/*
 * GGML Q6_K block:
 * 256 values, packed into 6-bit quantized values.
 */
typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} block_q6_K;

#define QK_K 256

int q4_k_decode(
    const void *input,
    size_t input_size,
    float *output,
    size_t output_count
);

int q6_k_decode(
    const void *input,
    size_t input_size,
    float *output,
    size_t output_count
);
int q5_1_decode(
    const void *input,
    size_t input_size,
    float *output,
    size_t output_count
);

int q8_0_decode(
    const void *input,
    size_t input_size,
    float *output,
    size_t output_count
);
#endif
