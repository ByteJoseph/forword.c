#ifndef ROPE_H
#define ROPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Precompute cos/sin tables for RoPE.
 *
 *   cos_tab[pos * half_dim + i] = cos(pos / freq_base^(2i/head_dim))
 *   sin_tab[pos * half_dim + i] = sin(...)
 *
 * Arrays must be allocated by the caller with size n_ctx * (head_dim/2) floats.
 * head_dim must be even.
 */
void rope_precompute(float *cos_tab, float *sin_tab,
                     size_t n_ctx, size_t head_dim, float freq_base);

/*
 * Apply RoPE in-place to a set of heads at position pos.
 *
 * qk        : contiguous buffer [n_heads * head_dim]
 * pos        : token position (0-based)
 * n_heads    : number of heads in the buffer (Q heads or KV heads)
 * head_dim   : per-head dimension (must match what was used in rope_precompute)
 * cos_tab / sin_tab : precomputed tables (size n_ctx * head_dim/2)
 * neox_style : 0 = pair (0,1),(2,3)…  (LLaMA/Qwen2 style)
 *              1 = pair (0,half),(1,half+1)… (GPT-NeoX style)
 */
void rope_apply(float *qk, size_t pos, size_t n_heads, size_t head_dim,
                const float *cos_tab, const float *sin_tab, int neox_style);

#ifdef __cplusplus
}
#endif
#endif /* ROPE_H */
