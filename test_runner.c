#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "norm.h"
#include "rope.h"
#include "sampler.h"
#include "quant.h"
#include "quant_matvec.h"
#include "tensor_ops.h"
#include "chat_template.h"

static int g_tests_passed = 0;
static int g_tests_total = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_tests_total++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        return -1; \
    } \
    g_tests_passed++; \
} while (0)

/* ── Test Normalization ──────────────────────────────────────────────── */
static int test_rms_norm(void) {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4] = {0};
    /* sum of squares = 1 + 4 + 9 + 16 = 30. mean = 7.5. sqrt(7.5 + 1e-5) ~= 2.738612788 */
    rms_norm(x, w, out, 4, 1e-5f);
    float ss = 0.0f;
    for (int i = 0; i < 4; i++) ss += out[i] * out[i];
    float mean_ss = ss / 4.0f;
    TEST_ASSERT(fabsf(mean_ss - 1.0f) < 1e-4f, "rms_norm unit variance");
    return 0;
}

static int test_layer_norm(void) {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4] = {0};
    layer_norm(x, w, b, out, 4, 1e-5f);
    /* mean of out should be 0, variance should be 1 */
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += out[i];
    TEST_ASSERT(fabsf(sum / 4.0f) < 1e-5f, "layer_norm mean zero");
    float var = 0.0f;
    for (int i = 0; i < 4; i++) var += out[i] * out[i];
    TEST_ASSERT(fabsf(var / 4.0f - 1.0f) < 1e-4f, "layer_norm unit variance");
    return 0;
}

/* ── Test RoPE ───────────────────────────────────────────────────────── */
static int test_rope(void) {
    size_t head_dim = 64;
    size_t n_ctx = 16;
    size_t half = head_dim / 2;
    float *cos_tab = malloc(n_ctx * half * sizeof(float));
    float *sin_tab = malloc(n_ctx * half * sizeof(float));
    TEST_ASSERT(cos_tab && sin_tab, "malloc rope tables");

    rope_precompute(cos_tab, sin_tab, n_ctx, head_dim, 10000.0f);

    /* At pos 0, theta should be 0, so cos=1, sin=0 */
    for (size_t i = 0; i < half; i++) {
        TEST_ASSERT(fabsf(cos_tab[i] - 1.0f) < 1e-6f, "pos 0 cos is 1");
        TEST_ASSERT(fabsf(sin_tab[i] - 0.0f) < 1e-6f, "pos 0 sin is 0");
    }

    /* Apply rope at pos 0 should not change qk */
    float qk[64];
    for (size_t i = 0; i < 64; i++) qk[i] = (float)(i + 1);
    float qk_orig[64];
    memcpy(qk_orig, qk, sizeof(qk));

    rope_apply(qk, 0, 1, head_dim, cos_tab, sin_tab, 0);
    for (size_t i = 0; i < 64; i++) {
        TEST_ASSERT(fabsf(qk[i] - qk_orig[i]) < 1e-5f, "pos 0 rope preserves values");
    }

    free(cos_tab);
    free(sin_tab);
    return 0;
}

/* ── Test Sampler ────────────────────────────────────────────────────── */
static int test_sampler(void) {
    float logits[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    uint32_t best = sampler_argmax(logits, 5);
    TEST_ASSERT(best == 4, "argmax selects index 4");

    sampler_softmax(logits, 5);
    float sum = 0.0f;
    for (int i = 0; i < 5; i++) {
        TEST_ASSERT(logits[i] > 0.0f, "softmax probability positive");
        sum += logits[i];
    }
    TEST_ASSERT(fabsf(sum - 1.0f) < 1e-5f, "softmax sum to 1");

    return 0;
}

/* ── Test Quantization Decoders ──────────────────────────────────────── */
static int test_q8_0(void) {
    /* Q8_0 block: 2 bytes fp16 d, 32 bytes int8 qs = 34 bytes */
    uint8_t block[34] = {0};
    /* fp16 1.0f: sign=0, exp=15 (0b01111), mant=0 -> 0x3C00 */
    block[0] = 0x00;
    block[1] = 0x3C;
    for (int i = 0; i < 32; i++) {
        block[2 + i] = (int8_t)(i - 16);
    }
    float out[32] = {0};
    int rc = q8_0_decode(block, 34, out, 32);
    TEST_ASSERT(rc == 0, "q8_0_decode succeeds");
    for (int i = 0; i < 32; i++) {
        TEST_ASSERT(fabsf(out[i] - (float)(i - 16)) < 1e-5f, "q8_0 decoded value matches");
    }
    return 0;
}

static int test_q4_k(void) {
    /* Q4_K block: 144 bytes for 256 values */
    uint8_t block[144] = {0};
    /* d = 1.0 (0x3C00), dmin = 0.0 (0x0000) */
    block[0] = 0x00;
    block[1] = 0x3C;
    /* scales: group 0 scale = 1, min = 0 */
    block[4] = 1;

    float out[256] = {0};
    int rc = q4_k_decode(block, 144, out, 256);
    TEST_ASSERT(rc == 0, "q4_k_decode succeeds");
    for (int i = 0; i < 256; i++) {
        TEST_ASSERT(isfinite(out[i]), "q4_k out is finite");
    }
    return 0;
}

static int test_q6_k(void) {
    /* Q6_K block: 210 bytes for 256 values */
    uint8_t block[210] = {0};
    /* fp16 d = 1.0f at offset 208 */
    block[208] = 0x00;
    block[209] = 0x3C;
    float out[256] = {0};
    int rc = q6_k_decode(block, 210, out, 256);
    TEST_ASSERT(rc == 0, "q6_k_decode succeeds");
    for (int i = 0; i < 256; i++) {
        TEST_ASSERT(isfinite(out[i]), "q6_k out is finite");
    }
    return 0;
}

static int test_quant_matvec(void) {
    /* Test Q8_0 matvec */
    /* 1 row of 32 inputs -> 1 output */
    uint8_t block[34] = {0};
    block[0] = 0x00;
    block[1] = 0x3C; /* d = 1.0f */
    for (int i = 0; i < 32; i++) {
        block[2 + i] = 2; /* value = 2.0f */
    }
    float x[32];
    for (int i = 0; i < 32; i++) x[i] = 1.0f;
    float y[1] = {0};
    int rc = quant_matvec(block, sizeof(block), 8 /* TYPE_Q8_0 */, x, y, 32, 1);
    TEST_ASSERT(rc == 0, "quant_matvec Q8_0 succeeds");
    /* 32 * (2.0 * 1.0) = 64.0 */
    TEST_ASSERT(fabsf(y[0] - 64.0f) < 1e-4f, "quant_matvec Q8_0 dot product matches");
    return 0;
}

/* ── Test Chat Template ──────────────────────────────────────────────── */
static int test_chat_template(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.arch = ARCH_QWEN2;

    char buf[512];
    int n = chat_template_format(&cfg, "System prompt", "Hello", buf, sizeof(buf));
    TEST_ASSERT(n > 0, "chat_template formats successfully");
    TEST_ASSERT(strstr(buf, "<|im_start|>system\nSystem prompt<|im_end|>\n") != NULL, "qwen2 system template matches");
    TEST_ASSERT(strstr(buf, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n") != NULL, "qwen2 user template matches");
    return 0;
}

int main(void) {
    printf("Running unit tests...\n");

    if (test_rms_norm() != 0) return 1;
    if (test_layer_norm() != 0) return 1;
    if (test_rope() != 0) return 1;
    if (test_sampler() != 0) return 1;
    if (test_q8_0() != 0) return 1;
    if (test_q4_k() != 0) return 1;
    if (test_q6_k() != 0) return 1;
    if (test_quant_matvec() != 0) return 1;
    if (test_chat_template() != 0) return 1;

    printf("ALL TESTS PASSED! (%d/%d checks passed)\n", g_tests_passed, g_tests_total);
    return 0;
}
