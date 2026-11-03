#include "quant_matvec.h"
#include "quant.h"
#include "simd.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TYPE_Q5_0 6
#define TYPE_Q5_1 7
#define TYPE_Q8_0 8
#define TYPE_Q4_K 12
#define TYPE_Q6_K 14

#define K_BLOCK 256

static inline uint16_t read_u16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline float half_to_float(uint16_t h)
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

/* Fast SIMD dot product for Q8_0 block (32 int8 weights * float inputs) */
static inline float dot_q8_0_block(const uint8_t *p, const float *x)
{
    float d = half_to_float(read_u16(p));
    const int8_t *qs = (const int8_t *)(p + 2);

#if defined(SIMD_AVX2)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    /* Load 32 int8s into two 128-bit halves */
    __m128i q16_0 = _mm_loadu_si128((const __m128i *)qs);
    __m128i q16_1 = _mm_loadu_si128((const __m128i *)(qs + 16));

    /* Sign-extend to 32-bit integers, convert to float */
    __m256i q32_0 = _mm256_cvtepi8_epi32(q16_0);
    __m256i q32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q16_0, 8));
    __m256i q32_2 = _mm256_cvtepi8_epi32(q16_1);
    __m256i q32_3 = _mm256_cvtepi8_epi32(_mm_srli_si128(q16_1, 8));

    __m256 f0 = _mm256_cvtepi32_ps(q32_0);
    __m256 f1 = _mm256_cvtepi32_ps(q32_1);
    __m256 f2 = _mm256_cvtepi32_ps(q32_2);
    __m256 f3 = _mm256_cvtepi32_ps(q32_3);

    __m256 x0 = _mm256_loadu_ps(x);
    __m256 x1 = _mm256_loadu_ps(x + 8);
    __m256 x2 = _mm256_loadu_ps(x + 16);
    __m256 x3 = _mm256_loadu_ps(x + 24);

#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(f0, x0, acc0);
    acc1 = _mm256_fmadd_ps(f1, x1, acc1);
    acc2 = _mm256_fmadd_ps(f2, x2, acc2);
    acc3 = _mm256_fmadd_ps(f3, x3, acc3);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(f0, x0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(f1, x1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(f2, x2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(f3, x3));
#endif

    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    return d * _mm_cvtss_f32(sum128);

#elif defined(SIMD_NEON)
    int8x16_t q0 = vld1q_s8(qs);
    int8x16_t q1 = vld1q_s8(qs + 16);

    int16x8_t q0_lo = vmovl_s8(vget_low_s8(q0));
    int16x8_t q0_hi = vmovl_s8(vget_high_s8(q0));
    int16x8_t q1_lo = vmovl_s8(vget_low_s8(q1));
    int16x8_t q1_hi = vmovl_s8(vget_high_s8(q1));

    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_lo)));
    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0_lo)));
    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_hi)));
    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0_hi)));

    float32x4_t f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_lo)));
    float32x4_t f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_lo)));
    float32x4_t f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_hi)));
    float32x4_t f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_hi)));

    float32x4_t acc = vdupq_n_f32(0.0f);
#if defined(__aarch64__)
    acc = vfmaq_f32(acc, f0, vld1q_f32(x));
    acc = vfmaq_f32(acc, f1, vld1q_f32(x + 4));
    acc = vfmaq_f32(acc, f2, vld1q_f32(x + 8));
    acc = vfmaq_f32(acc, f3, vld1q_f32(x + 12));
    acc = vfmaq_f32(acc, f4, vld1q_f32(x + 16));
    acc = vfmaq_f32(acc, f5, vld1q_f32(x + 20));
    acc = vfmaq_f32(acc, f6, vld1q_f32(x + 24));
    acc = vfmaq_f32(acc, f7, vld1q_f32(x + 28));
    return d * vaddvq_f32(acc);
#else
    acc = vaddq_f32(acc, vmulq_f32(f0, vld1q_f32(x)));
    acc = vaddq_f32(acc, vmulq_f32(f1, vld1q_f32(x + 4)));
    acc = vaddq_f32(acc, vmulq_f32(f2, vld1q_f32(x + 8)));
    acc = vaddq_f32(acc, vmulq_f32(f3, vld1q_f32(x + 12)));
    acc = vaddq_f32(acc, vmulq_f32(f4, vld1q_f32(x + 16)));
    acc = vaddq_f32(acc, vmulq_f32(f5, vld1q_f32(x + 20)));
    acc = vaddq_f32(acc, vmulq_f32(f6, vld1q_f32(x + 24)));
    acc = vaddq_f32(acc, vmulq_f32(f7, vld1q_f32(x + 28)));
    float32x2_t r = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    return d * vget_lane_f32(vpadd_f32(r, r), 0);
#endif

#elif defined(SIMD_WASM)
    v128_t acc = wasm_f32x4_splat(0.0f);
    for (int i = 0; i < 32; i += 4) {
        v128_t fi = wasm_f32x4_make((float)qs[i], (float)qs[i+1], (float)qs[i+2], (float)qs[i+3]);
        v128_t xi = wasm_v128_load(x + i);
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(fi, xi));
    }
    float s = wasm_f32x4_extract_lane(acc, 0) +
              wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) +
              wasm_f32x4_extract_lane(acc, 3);
    return d * s;

#elif defined(SIMD_SSE2)
    __m128 acc = _mm_setzero_ps();
    for (int i = 0; i < 32; i += 4) {
        __m128 fi = _mm_set_ps((float)qs[i+3], (float)qs[i+2], (float)qs[i+1], (float)qs[i]);
        __m128 xi = _mm_loadu_ps(x + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(fi, xi));
    }
    acc = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
    acc = _mm_add_ss(acc, _mm_shuffle_ps(acc, acc, 1));
    return d * _mm_cvtss_f32(acc);

#else
    float sum = 0.0f;
    for (int j = 0; j < 32; ++j) {
        sum += (float)qs[j] * x[j];
    }
    return d * sum;
#endif
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

static inline int dot_quant_block(
    const uint8_t *block,
    uint32_t type,
    const float *x,
    float *result)
{
    if (type == TYPE_Q8_0) {
        *result = dot_q8_0_block(block, x);
        return 0;
    }

    float decoded[K_BLOCK];
    if (type == TYPE_Q5_0 || type == TYPE_Q5_1) {
        if (decode_small_block(block, type, decoded) != 0)
            return -1;
        *result = simd_dot_f32(decoded, x, 32);
        return 0;
    } else if (type == TYPE_Q4_K) {
        if (q4_k_decode(block, 144, decoded, K_BLOCK) != 0)
            return -1;
        *result = simd_dot_f32(decoded, x, K_BLOCK);
        return 0;
    } else if (type == TYPE_Q6_K) {
        if (q6_k_decode(block, 210, decoded, K_BLOCK) != 0)
            return -1;
        *result = simd_dot_f32(decoded, x, K_BLOCK);
        return 0;
    }

    return -1;
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
            if (dot_quant_block(
                    row_weights + b * block_bytes,
                    ggml_type,
                    input + b * block_values,
                    &partial) != 0)
                return -1;
            sum += partial;
        }

        output[row] = sum;
    }

    return 0;
}
