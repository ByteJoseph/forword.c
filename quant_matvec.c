
#include "quant_matvec.h"
#include "quant.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TYPE_Q5_0 6
#define TYPE_Q5_1 7
#define TYPE_Q8_0 8
#define TYPE_Q4_K 12
#define TYPE_Q6_K 14

#define K_BLOCK 256

static uint16_t read_u16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 31u;
    uint32_t mant = h & 1023u;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 1024u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 1023u;
            bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static int block_layout(uint32_t type, size_t *values, size_t *bytes)
{
    switch (type) {
    case TYPE_Q5_0: *values = 32;  *bytes = 22;  return 0;
    case TYPE_Q5_1: *values = 32;  *bytes = 24;  return 0;
    case TYPE_Q8_0: *values = 32;  *bytes = 34;  return 0;
    case TYPE_Q4_K: *values = 256; *bytes = 144; return 0;
    case TYPE_Q6_K: *values = 256; *bytes = 210; return 0;
    default: return -1;
    }
}

static int decode_small_block(
    const uint8_t *p,
    uint32_t type,
    float *out)
{
    float d = half_to_float(read_u16(p));

    if (type == TYPE_Q5_0 || type == TYPE_Q5_1) {
        float m = type == TYPE_Q5_1 ? half_to_float(read_u16(p + 2)) : 0.0f;
        const uint8_t *qh = p + (type == TYPE_Q5_1 ? 4 : 2);
        const uint8_t *qs = p + (type == TYPE_Q5_1 ? 8 : 6);

        for (int j = 0; j < 16; ++j) {
            int h0 = (qh[j / 8] >> (j % 8)) & 1;
            int h1 = (qh[(j + 16) / 8] >> ((j + 16) % 8)) & 1;
            int q0 = (qs[j] & 15) | (h0 << 4);
            int q1 = (qs[j] >> 4) | (h1 << 4);

            if (type == TYPE_Q5_0) {
                out[j] = d * (float)(q0 - 16);
                out[j + 16] = d * (float)(q1 - 16);
            } else {
                out[j] = d * (float)q0 + m;
                out[j + 16] = d * (float)q1 + m;
            }
        }
        return 0;
    }

    if (type == TYPE_Q8_0) {
        const int8_t *qs = (const int8_t *)(p + 2);
        for (int j = 0; j < 32; ++j)
            out[j] = d * (float)qs[j];
        return 0;
    }

    return -1;
}

static int dot_quant_block(
    const uint8_t *block,
    uint32_t type,
    const float *x,
    float *result)
{
    size_t values, bytes;
    if (block_layout(type, &values, &bytes) != 0)
        return -1;

    float decoded[K_BLOCK];
    if (values == 32) {
        if (decode_small_block(block, type, decoded) != 0)
            return -1;
    } else if (type == TYPE_Q4_K) {
        if (q4_k_decode(block, bytes, decoded, values) != 0)
            return -1;
    } else if (type == TYPE_Q6_K) {
        if (q6_k_decode(block, bytes, decoded, values) != 0)
            return -1;
    } else {
        return -1;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < values; ++i)
        sum += decoded[i] * x[i];

    *result = sum;
    return 0;
}

int quant_matvec(
    const void *weights,
    size_t weights_size,
    uint32_t ggml_type,
    const float *input,
    float *output,
    size_t input_size,
    size_t output_size)
{
    size_t block_values, block_bytes;

    if (!weights || !input || !output ||
        input_size == 0 || output_size == 0 ||
        block_layout(ggml_type, &block_values, &block_bytes) != 0 ||
        input_size % block_values != 0)
        return -1;

    size_t blocks_per_row = input_size / block_values;

    if (blocks_per_row > SIZE_MAX / block_bytes)
        return -1;

    size_t bytes_per_row = blocks_per_row * block_bytes;

    if (output_size > SIZE_MAX / bytes_per_row ||
        weights_size < output_size * bytes_per_row)
        return -1;

    const uint8_t *raw = (const uint8_t *)weights;

    for (size_t row = 0; row < output_size; ++row) {
        float sum = 0.0f;
        const uint8_t *row_weights = raw + row * bytes_per_row;

        for (size_t b = 0; b < blocks_per_row; ++b) {
            float partial;
            if (block_values == 32) {
                float decoded[32];
                if (decode_small_block(
                        row_weights + b * block_bytes,
                        ggml_type, decoded) != 0)
                    return -1;

                partial = 0.0f;
                const float *x = input + b * block_values;
                for (size_t i = 0; i < block_values; ++i)
                    partial += decoded[i] * x[i];
            } else {
                if (dot_quant_block(
                        row_weights + b * block_bytes,
                        ggml_type,
                        input + b * block_values,
                        &partial) != 0)
                    return -1;
            }
            sum += partial;
        }

        output[row] = sum;
    }

    return 0;
}
