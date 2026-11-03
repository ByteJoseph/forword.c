#define _POSIX_C_SOURCE 200809L
#include "linear.h"
#include "quant.h"
#include "quant_matvec.h"
#include "simd.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* ── Float conversion helpers ───────────────────────────────────────────── */

float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (!mant) { bits = sign; }
        else {
            /* Denormal */
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13); /* Inf or NaN */
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

float bf16_to_f32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f; memcpy(&f, &bits, 4); return f;
}

static int q5_0_decode(const void *block, float *out)
{
    const uint8_t *p = (const uint8_t *)block;
    uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    float d = f16_to_f32(h);
    const uint8_t *qh = p + 2;
    const uint8_t *qs = p + 6;
    for (int j = 0; j < 16; ++j) {
        int h0 = (qh[j / 8] >> (j % 8)) & 1;
        int h1 = (qh[(j + 16) / 8] >> ((j + 16) % 8)) & 1;
        int q0 = (qs[j] & 15) | (h0 << 4);
        int q1 = (qs[j] >> 4) | (h1 << 4);
        out[j] = d * (float)(q0 - 16);
        out[j + 16] = d * (float)(q1 - 16);
    }
    return 0;
}

/* ── Embedding row dequantisation ────────────────────────────────────────── */

int linear_get_row(const gguf_file_t   *g,
                   const gguf_tensor_t *t,
                   size_t               row_idx,
                   float               *out)
{
    size_t   raw_size = 0;
    const void *raw   = tensor_data(g, t, &raw_size);
    if (!raw) return -1;

    uint32_t type  = t->ggml_type;
    size_t   ncols = (size_t)t->dims[0]; /* elements per row */

    /* ── F32 ──────────────────────────────────────────────────────────── */
    if (type == 0) {
        size_t offset = row_idx * ncols * sizeof(float);
        if (offset + ncols * sizeof(float) > raw_size) return -1;
        memcpy(out, (const char *)raw + offset, ncols * sizeof(float));
        return 0;
    }
    /* ── BF16 (type 30) ──────────────────────────────────────────────── */
    if (type == 30) {
        size_t offset = row_idx * ncols * 2;
        if (offset + ncols * 2 > raw_size) return -1;
        const uint16_t *src = (const uint16_t *)((const char *)raw + offset);
        for (size_t i = 0; i < ncols; i++) out[i] = bf16_to_f32(src[i]);
        return 0;
    }
    /* ── F16 (type 1) ────────────────────────────────────────────────── */
    if (type == 1) {
        size_t offset = row_idx * ncols * 2;
        if (offset + ncols * 2 > raw_size) return -1;
        const uint16_t *src = (const uint16_t *)((const char *)raw + offset);
        for (size_t i = 0; i < ncols; i++) out[i] = f16_to_f32(src[i]);
        return 0;
    }

    /* ── Block quantisation: dequantise row block-by-block ───────────── */
    size_t block_size  = 256; /* Default for K-quants */
    size_t block_bytes = 0;

    switch (type) {
    case 2:  block_size = 32;  block_bytes = 18;  break; /* Q4_0 */
    case 3:  block_size = 32;  block_bytes = 20;  break; /* Q4_1 */
    case 6:  block_size = 32;  block_bytes = 22;  break; /* Q5_0 */
    case 7:  block_size = 32;  block_bytes = 24;  break; /* Q5_1 */
    case 8:  block_size = 32;  block_bytes = 34;  break; /* Q8_0 */
    case 10: block_size = 256; block_bytes = 84;  break; /* Q2_K */
    case 11: block_size = 256; block_bytes = 110; break; /* Q3_K */
    case 12: block_size = 256; block_bytes = 144; break; /* Q4_K */
    case 13: block_size = 256; block_bytes = 176; break; /* Q5_K */
    case 14: block_size = 256; block_bytes = 210; break; /* Q6_K */
    default:
        fprintf(stderr, "linear_get_row: unsupported ggml_type %u\n", type);
        return -1;
    }

    if (ncols % block_size != 0) return -1;
    size_t blocks_per_row = ncols / block_size;
    size_t bytes_per_row  = blocks_per_row * block_bytes;
    size_t row_offset     = row_idx * bytes_per_row;
    if (row_offset + bytes_per_row > raw_size) return -1;

    const uint8_t *row_raw = (const uint8_t *)raw + row_offset;
    for (size_t b = 0; b < blocks_per_row; b++) {
        const void *block = row_raw + b * block_bytes;
        float *dst = out + b * block_size;
        int rc = 0;
switch (type) {
        case 6:
            rc = q5_0_decode(block, dst);
            break;
        case 7:
            rc = q5_1_decode(block, block_bytes, dst, block_size);
            break;
        case 8:
            rc = q8_0_decode(block, block_bytes, dst, block_size);
            break;
        case 12:
            rc = q4_k_decode(block, block_bytes, dst, block_size);
            break;
        case 14:
            rc = q6_k_decode(block, block_bytes, dst, block_size);
            break;
        default:
            fprintf(stderr,
                    "linear_get_row: no decoder for type %u\n", type);
            return -1;
        }
        if (rc) return -1;
    }
    return 0;
}

/* ── Unified matrix-vector forward ──────────────────────────────────────── */

int linear_forward(const gguf_file_t    *g,
                   const gguf_tensor_t  *weight,
                   const gguf_tensor_t  *bias,
                   const float          *x,
                   float                *y,
                   float                *tmp)
{
    (void)tmp; /* may be used in future for dequant scratch */

    size_t   raw_size = 0;
    const void *raw   = tensor_data(g, weight, &raw_size);
    if (!raw) return -1;

    uint32_t type     = weight->ggml_type;
    size_t   in_size  = (size_t)weight->dims[0];
    size_t   out_size = (size_t)weight->dims[1];

    /* ── F32 weights ──────────────────────────────────────────────────── */
    if (type == 0) {
        const float *W = (const float *)raw;
        for (size_t r = 0; r < out_size; r++) {
            const float *row = W + r * in_size;
            y[r] = simd_dot_f32(row, x, in_size);
        }
        goto apply_bias;
    }

    /* ── BF16 weights (type 30) ──────────────────────────────────────── */
    if (type == 30) {
        const uint16_t *W = (const uint16_t *)raw;
        for (size_t r = 0; r < out_size; r++) {
            double sum = 0.0;
            const uint16_t *row = W + r * in_size;
            for (size_t c = 0; c < in_size; c++)
                sum += (double)bf16_to_f32(row[c]) * x[c];
            y[r] = (float)sum;
        }
        goto apply_bias;
    }

    /* ── F16 weights (type 1) ────────────────────────────────────────── */
    if (type == 1) {
        const uint16_t *W = (const uint16_t *)raw;
        for (size_t r = 0; r < out_size; r++) {
            double sum = 0.0;
            const uint16_t *row = W + r * in_size;
            for (size_t c = 0; c < in_size; c++)
                sum += (double)f16_to_f32(row[c]) * x[c];
            y[r] = (float)sum;
        }
        goto apply_bias;
    }

    /* ── Block-quantised weights ─────────────────────────────────────── */
    if (quant_matvec(raw, raw_size, type, x, y, in_size, out_size) != 0) {
        fprintf(stderr, "linear_forward: quant_matvec failed (type %u)\n", type);
        return -1;
    }

apply_bias:
    if (bias) {
        size_t   bs = 0;
        const float *bdata = (const float *)tensor_data(g, bias, &bs);
        if (bdata && bias->ggml_type == 0) {  /* F32 bias only */
            for (size_t i = 0; i < out_size; i++) y[i] += bdata[i];
        }
    }
    return 0;
}
