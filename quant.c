
#include "quant.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>

/*
 * GGML Q4_K block:
 *   fp16 d, fp16 dmin, scales[12], qs[128]
 *   256 values / 144 bytes
 *
 * GGML Q6_K block:
 *   ql[128], qh[64], scales[16], fp16 d
 *   256 values / 210 bytes
 *
 * These routines assume the little-endian host used by your
 * current x86 Linux/WSL environment.
 */

static float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t) h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t) h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* Normalize a half-precision subnormal. */
            int e = -14;

            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }

            mant &= 0x03ffu;
            bits = sign |
                   ((uint32_t)(e + 127) << 23) |
                   (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &bits, sizeof result);
    return result;
}

static uint16_t read_fp16(const uint8_t *p)
{
    /* GGUF data is little-endian. */
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static void get_scale_min_k4(
    int j,
    const uint8_t scales[12],
    uint8_t *scale,
    uint8_t *minimum)
{
    if (j < 4) {
        *scale   = scales[j] & 0x3f;
        *minimum = scales[j + 4] & 0x3f;
    } else {
        *scale = (uint8_t)(
            (scales[j + 4] & 0x0f) |
            ((scales[j - 4] >> 6) << 4)
        );

        *minimum = (uint8_t)(
            (scales[j + 4] >> 4) |
            ((scales[j] >> 6) << 4)
        );
    }
}

int q4_k_decode(
    const void *input,
    size_t input_bytes,
    float *output,
    size_t output_count)
{
    const size_t block_bytes = 144;
    const size_t block_values = 256;

    if (!input || !output ||
        input_bytes == 0 ||
        input_bytes % block_bytes != 0 ||
        output_count % block_values != 0 ||
        input_bytes / block_bytes != output_count / block_values) {
        return -1;
    }

    const uint8_t *src = (const uint8_t *)input;
    const size_t blocks = input_bytes / block_bytes;

    for (size_t b = 0; b < blocks; ++b, src += block_bytes) {
        const float d = fp16_to_fp32(read_fp16(src));
        const float dmin = fp16_to_fp32(read_fp16(src + 2));
        const uint8_t *scales = src + 4;
        const uint8_t *qs = src + 16;
        float *y = output + b * block_values;

        for (int group = 0; group < 8; ++group) {
            uint8_t scale, minimum;
            get_scale_min_k4(group, scales, &scale, &minimum);

            const float ds = d * scale;
            const float dm = dmin * minimum;
            const uint8_t *q = qs + (group / 2) * 32;

            for (int i = 0; i < 32; ++i) {
                const uint8_t packed = q[i];
                const uint8_t value =
                    (group & 1) ? (packed >> 4) : (packed & 0x0f);

                y[group * 32 + i] = ds * value - dm;
            }
        }
    }

    return 0;
}

int q6_k_decode(
    const void *input,
    size_t input_bytes,
    float *output,
    size_t output_count)
{
    const size_t block_bytes = 210;
    const size_t block_values = 256;

    if (!input || !output ||
        input_bytes == 0 ||
        input_bytes % block_bytes != 0 ||
        output_count % block_values != 0 ||
        input_bytes / block_bytes != output_count / block_values) {
        return -1;
    }

    const uint8_t *src = (const uint8_t *)input;
    const size_t blocks = input_bytes / block_bytes;

    for (size_t b = 0; b < blocks; ++b, src += block_bytes) {
        const uint8_t *ql = src;
        const uint8_t *qh = src + 128;
        const int8_t *sc = (const int8_t *)(src + 192);
        const float d = fp16_to_fp32(read_fp16(src + 208));
        float *y = output + b * block_values;

        for (int half = 0; half < 2; ++half) {
            const uint8_t *ql_half = ql + half * 64;
            const uint8_t *qh_half = qh + half * 32;
            const int8_t *sc_half = sc + half * 8;
            float *out = y + half * 128;

            for (int l = 0; l < 32; ++l) {
                const uint8_t low_a = ql_half[l];
                const uint8_t low_b = ql_half[l + 32];
                const uint8_t high = qh_half[l];

                const int q1 =
                    (low_a & 0x0f) | ((high & 0x03) << 4);
                const int q2 =
                    (low_b & 0x0f) | (((high >> 2) & 0x03) << 4);
                const int q3 =
                    (low_a >> 4) | (((high >> 4) & 0x03) << 4);
                const int q4 =
                    (low_b >> 4) | (((high >> 6) & 0x03) << 4);

                out[l]      = d * sc_half[l / 16]     * (q1 - 32);
                out[l + 32] = d * sc_half[l / 16 + 2] * (q2 - 32);
                out[l + 64] = d * sc_half[l / 16 + 4] * (q3 - 32);
                out[l + 96] = d * sc_half[l / 16 + 6] * (q4 - 32);
            }
        }
    }

    return 0;
}
int q5_1_decode(
    const void *input,
    size_t input_bytes,
    float *output,
    size_t output_count)
{
    const size_t block_bytes = 24;
    const size_t block_values = 32;

    if (!input || !output ||
        input_bytes == 0 ||
        input_bytes % block_bytes != 0 ||
        output_count % block_values != 0 ||
        input_bytes / block_bytes != output_count / block_values) {
        return -1;
    }

    const uint8_t *src = (const uint8_t *)input;
    const size_t blocks = input_bytes / block_bytes;

    for (size_t b = 0; b < blocks; ++b, src += block_bytes) {
        float d = fp16_to_fp32(read_fp16(src));
        float m = fp16_to_fp32(read_fp16(src + 2));
        const uint8_t *qh = src + 4;
        const uint8_t *qs = src + 8;
        float *y = output + b * block_values;

        for (int j = 0; j < 16; ++j) {
            int h0 = (qh[j / 8] >> (j % 8)) & 1;
            int h1 = (qh[(j + 16) / 8] >> ((j + 16) % 8)) & 1;

            int q0 = (qs[j] & 0x0f) | (h0 << 4);
            int q1 = (qs[j] >> 4) | (h1 << 4);

            y[j]      = d * (float)q0 + m;
            y[j + 16] = d * (float)q1 + m;
        }
    }

    return 0;
}

int q8_0_decode(
    const void *input,
    size_t input_bytes,
    float *output,
    size_t output_count)
{
    const size_t block_bytes = 34;
    const size_t block_values = 32;

    if (!input || !output ||
        input_bytes == 0 ||
        input_bytes % block_bytes != 0 ||
        output_count % block_values != 0 ||
        input_bytes / block_bytes != output_count / block_values) {
        return -1;
    }

    const uint8_t *src = (const uint8_t *)input;
    const size_t blocks = input_bytes / block_bytes;

    for (size_t b = 0; b < blocks; ++b, src += block_bytes) {
        float d = fp16_to_fp32(read_fp16(src));
        const int8_t *qs = (const int8_t *)(src + 2);
        float *y = output + b * block_values;

        for (int j = 0; j < 32; ++j) {
            y[j] = d * (float)qs[j];
        }
    }

    return 0;
}