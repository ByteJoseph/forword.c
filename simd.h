#ifndef SIMD_H
#define SIMD_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

/* Architecture detection */
#if !defined(NO_SIMD)
  #if defined(__AVX2__)
    #define SIMD_AVX2 1
    #include <immintrin.h>
  #elif defined(__SSE2__)
    #define SIMD_SSE2 1
    #include <immintrin.h>
  #elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define SIMD_NEON 1
    #include <arm_neon.h>
  #elif defined(__wasm_simd128__)
    #define SIMD_WASM 1
    #include <wasm_simd128.h>
  #endif
#endif

/* ── Portable F32 Vector Dot Product ───────────────────────────────────────── */

static inline float simd_dot_f32(const float *a, const float *b, size_t n) {
    size_t i = 0;
    float sum = 0.0f;

#if defined(SIMD_AVX2)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
#endif
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
#if defined(__FMA__)
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
#else
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
#endif
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va, vb));
#endif
    }
    /* Horizontal reduction of acc0 */
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    sum += _mm_cvtss_f32(sum128);

#elif defined(SIMD_SSE2)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m128 va0 = _mm_loadu_ps(a + i);
        __m128 vb0 = _mm_loadu_ps(b + i);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(va0, vb0));

        __m128 va1 = _mm_loadu_ps(a + i + 4);
        __m128 vb1 = _mm_loadu_ps(b + i + 4);
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(va1, vb1));
    }
    acc0 = _mm_add_ps(acc0, acc1);
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(va, vb));
    }
    acc0 = _mm_add_ps(acc0, _mm_movehl_ps(acc0, acc0));
    acc0 = _mm_add_ss(acc0, _mm_shuffle_ps(acc0, acc0, 1));
    sum += _mm_cvtss_f32(acc0);

#elif defined(SIMD_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
#if defined(__aarch64__)
        acc0 = vfmaq_f32(acc0, va0, vb0);
#else
        acc0 = vaddq_f32(acc0, vmulq_f32(va0, vb0));
#endif
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
#if defined(__aarch64__)
        acc1 = vfmaq_f32(acc1, va1, vb1);
#else
        acc1 = vaddq_f32(acc1, vmulq_f32(va1, vb1));
#endif
    }
    acc0 = vaddq_f32(acc0, acc1);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
#if defined(__aarch64__)
        acc0 = vfmaq_f32(acc0, va, vb);
#else
        acc0 = vaddq_f32(acc0, vmulq_f32(va, vb));
#endif
    }
#if defined(__aarch64__)
    sum += vaddvq_f32(acc0);
#else
    float32x2_t r = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
    sum += vget_lane_f32(vpadd_f32(r, r), 0);
#endif

#elif defined(SIMD_WASM)
    v128_t acc0 = wasm_f32x4_splat(0.0f);
    v128_t acc1 = wasm_f32x4_splat(0.0f);
    for (; i + 8 <= n; i += 8) {
        v128_t va0 = wasm_v128_load(a + i);
        v128_t vb0 = wasm_v128_load(b + i);
        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(va0, vb0));

        v128_t va1 = wasm_v128_load(a + i + 4);
        v128_t vb1 = wasm_v128_load(b + i + 4);
        acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(va1, vb1));
    }
    acc0 = wasm_f32x4_add(acc0, acc1);
    for (; i + 4 <= n; i += 4) {
        v128_t va = wasm_v128_load(a + i);
        v128_t vb = wasm_v128_load(b + i);
        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(va, vb));
    }
    sum += wasm_f32x4_extract_lane(acc0, 0) +
           wasm_f32x4_extract_lane(acc0, 1) +
           wasm_f32x4_extract_lane(acc0, 2) +
           wasm_f32x4_extract_lane(acc0, 3);
#endif

    /* Scalar remainder */
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* ── Portable F32 Vector Sum of Squares (for RMS norm) ─────────────────────── */

static inline float simd_sum_squares_f32(const float *x, size_t n) {
    return simd_dot_f32(x, x, n);
}

/* ── Portable Vector Add / Scale / FMA Operations ──────────────────────────── */

static inline void simd_add_f32(const float *a, const float *b, float *out, size_t n) {
    size_t i = 0;
#if defined(SIMD_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
#elif defined(SIMD_SSE2)
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(out + i, _mm_add_ps(va, vb));
    }
#elif defined(SIMD_NEON)
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
#elif defined(SIMD_WASM)
    for (; i + 4 <= n; i += 4) {
        v128_t va = wasm_v128_load(a + i);
        v128_t vb = wasm_v128_load(b + i);
        wasm_v128_store(out + i, wasm_f32x4_add(va, vb));
    }
#endif
    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static inline void simd_scale_f32(const float *x, float scale, float *out, size_t n) {
    size_t i = 0;
#if defined(SIMD_AVX2)
    __m256 vs = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(vx, vs));
    }
#elif defined(SIMD_SSE2)
    __m128 vs = _mm_set1_ps(scale);
    for (; i + 4 <= n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(vx, vs));
    }
#elif defined(SIMD_NEON)
    float32x4_t vs = vdupq_n_f32(scale);
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vst1q_f32(out + i, vmulq_f32(vx, vs));
    }
#elif defined(SIMD_WASM)
    v128_t vs = wasm_f32x4_splat(scale);
    for (; i + 4 <= n; i += 4) {
        v128_t vx = wasm_v128_load(x + i);
        wasm_v128_store(out + i, wasm_f32x4_mul(vx, vs));
    }
#endif
    for (; i < n; i++) {
        out[i] = x[i] * scale;
    }
}

static inline void simd_mul_f32(const float *a, const float *b, float *out, size_t n) {
    size_t i = 0;
#if defined(SIMD_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
#elif defined(SIMD_SSE2)
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(va, vb));
    }
#elif defined(SIMD_NEON)
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
#elif defined(SIMD_WASM)
    for (; i + 4 <= n; i += 4) {
        v128_t va = wasm_v128_load(a + i);
        v128_t vb = wasm_v128_load(b + i);
        wasm_v128_store(out + i, wasm_f32x4_mul(va, vb));
    }
#endif
    for (; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

/* out[i] += a[i] * scale */
static inline void simd_axpy_f32(float *out, const float *a, float scale, size_t n) {
    size_t i = 0;
#if defined(SIMD_AVX2)
    __m256 vs = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        __m256 vo = _mm256_loadu_ps(out + i);
        __m256 va = _mm256_loadu_ps(a + i);
#if defined(__FMA__)
        vo = _mm256_fmadd_ps(va, vs, vo);
#else
        vo = _mm256_add_ps(vo, _mm256_mul_ps(va, vs));
#endif
        _mm256_storeu_ps(out + i, vo);
    }
#elif defined(SIMD_SSE2)
    __m128 vs = _mm_set1_ps(scale);
    for (; i + 4 <= n; i += 4) {
        __m128 vo = _mm_loadu_ps(out + i);
        __m128 va = _mm_loadu_ps(a + i);
        vo = _mm_add_ps(vo, _mm_mul_ps(va, vs));
        _mm_storeu_ps(out + i, vo);
    }
#elif defined(SIMD_NEON)
    float32x4_t vs = vdupq_n_f32(scale);
    for (; i + 4 <= n; i += 4) {
        float32x4_t vo = vld1q_f32(out + i);
        float32x4_t va = vld1q_f32(a + i);
#if defined(__aarch64__)
        vo = vfmaq_f32(vo, va, vs);
#else
        vo = vaddq_f32(vo, vmulq_f32(va, vs));
#endif
        vst1q_f32(out + i, vo);
    }
#elif defined(SIMD_WASM)
    v128_t vs = wasm_f32x4_splat(scale);
    for (; i + 4 <= n; i += 4) {
        v128_t vo = wasm_v128_load(out + i);
        v128_t va = wasm_v128_load(a + i);
        wasm_v128_store(out + i, wasm_f32x4_add(vo, wasm_f32x4_mul(va, vs)));
    }
#endif
    for (; i < n; i++) {
        out[i] += a[i] * scale;
    }
}

#endif /* SIMD_H */
