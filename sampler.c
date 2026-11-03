#define _POSIX_C_SOURCE 200809L
#include "sampler.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Simple xorshift64 PRNG */
static uint64_t xorshift64(uint64_t *s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s;
}

void sampler_temperature(float *logits, size_t n, float temp) {
    if (temp <= 0.0f || temp == 1.0f) return;
    float inv = 1.0f / temp;
    for (size_t i = 0; i < n; i++) logits[i] *= inv;
}

void sampler_softmax(float *logits, size_t n) {
    if (!n) return;
    float mx = logits[0];
    for (size_t i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) { logits[i] = expf(logits[i] - mx); sum += logits[i]; }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) logits[i] *= inv;
}

uint32_t sampler_argmax(const float *logits, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return (uint32_t)best;
}

/* Scratch buffer for sorting; allocated once, grown as needed. */
static float   *g_probs = NULL;
static uint32_t *g_idx  = NULL;
static size_t   g_cap   = 0;

static int cmp_prob_desc(const void *a, const void *b) {
    size_t ia = *(const uint32_t *)a, ib = *(const uint32_t *)b;
    if (g_probs[ib] > g_probs[ia]) return  1;
    if (g_probs[ib] < g_probs[ia]) return -1;
    return 0;
}

uint32_t sampler_top_p(float *logits, size_t n,
                        float temperature, float top_p, uint64_t *rng)
{
    if (temperature <= 0.0f) return sampler_argmax(logits, n);

    /* Ensure scratch capacity */
    if (n > g_cap) {
        free(g_probs); free(g_idx);
        g_probs = (float    *)malloc(n * sizeof(float));
        g_idx   = (uint32_t *)malloc(n * sizeof(uint32_t));
        if (!g_probs || !g_idx) return sampler_argmax(logits, n);
        g_cap = n;
    }

    /* Copy, apply temperature, softmax */
    memcpy(g_probs, logits, n * sizeof(float));
    sampler_temperature(g_probs, n, temperature);
    sampler_softmax(g_probs, n);

    /* Sort indices by descending probability */
    for (size_t i = 0; i < n; i++) g_idx[i] = (uint32_t)i;
    qsort(g_idx, n, sizeof(uint32_t), cmp_prob_desc);

    /* Truncate to top-p nucleus */
    float cum = 0.0f; size_t nucleus = 0;
    for (size_t i = 0; i < n; i++) {
        cum += g_probs[g_idx[i]];
        nucleus = i + 1;
        if (cum >= top_p) break;
    }

    /* Renormalise and sample */
    float norm = 0.0f;
    for (size_t i = 0; i < nucleus; i++) norm += g_probs[g_idx[i]];
    float r = (float)(xorshift64(rng) >> 11) / (float)(UINT64_MAX >> 11) * norm;
    cum = 0.0f;
    for (size_t i = 0; i < nucleus; i++) {
        cum += g_probs[g_idx[i]];
        if (r <= cum) return g_idx[i];
    }
    return g_idx[nucleus - 1];
}

void sampler_rep_penalty(float *logits, size_t n_vocab,
                          const uint32_t *ids, size_t n_ids, float penalty)
{
    if (penalty == 1.0f) return;
    for (size_t i = 0; i < n_ids; i++) {
        uint32_t id = ids[i];
        if (id < (uint32_t)n_vocab) {
            logits[id] = logits[id] > 0.0f
                         ? logits[id] / penalty
                         : logits[id] * penalty;
        }
    }
}
