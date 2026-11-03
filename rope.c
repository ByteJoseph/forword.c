#define _POSIX_C_SOURCE 200809L
#include "rope.h"
#include <math.h>
#include <stddef.h>

void rope_precompute(float *cos_tab, float *sin_tab,
                     size_t n_ctx, size_t head_dim, float freq_base)
{
    size_t half = head_dim / 2;
    for (size_t pos = 0; pos < n_ctx; pos++) {
        for (size_t i = 0; i < half; i++) {
            float theta = (float)pos / powf(freq_base, (float)(2 * i) / (float)head_dim);
            cos_tab[pos * half + i] = cosf(theta);
            sin_tab[pos * half + i] = sinf(theta);
        }
    }
}

/*
 * Standard (LLaMA/Qwen2) RoPE:
 *   For each pair (x[2i], x[2i+1]):
 *     x'[2i]   =  x[2i]   * cos(theta_i) - x[2i+1] * sin(theta_i)
 *     x'[2i+1] =  x[2i]   * sin(theta_i) + x[2i+1] * cos(theta_i)
 *
 * GPT-NeoX style:
 *   The pairs are (x[i], x[half+i]) instead of (x[2i], x[2i+1]).
 */
void rope_apply(float *qk, size_t pos, size_t n_heads, size_t head_dim,
                const float *cos_tab, const float *sin_tab, int neox_style)
{
    size_t half = head_dim / 2;
    const float *c = cos_tab + pos * half;
    const float *s = sin_tab + pos * half;

    for (size_t h = 0; h < n_heads; h++) {
        float *v = qk + h * head_dim;
        if (!neox_style) {
            /* Pairs (v[0],v[1]), (v[2],v[3]), … */
            for (size_t i = 0; i < half; i++) {
                float v0 = v[2*i], v1 = v[2*i+1];
                v[2*i]   = v0 * c[i] - v1 * s[i];
                v[2*i+1] = v0 * s[i] + v1 * c[i];
            }
        } else {
            /* Pairs (v[i], v[half+i]) */
            for (size_t i = 0; i < half; i++) {
                float v0 = v[i], v1 = v[half + i];
                v[i]        = v0 * c[i] - v1 * s[i];
                v[half + i] = v0 * s[i] + v1 * c[i];
            }
        }
    }
}
