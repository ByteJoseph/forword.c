#ifndef SAMPLER_H
#define SAMPLER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Apply temperature scaling to logits (in-place). */
void sampler_temperature(float *logits, size_t n, float temp);

/* Apply softmax in-place. */
void sampler_softmax(float *logits, size_t n);

/* Greedy argmax. */
uint32_t sampler_argmax(const float *logits, size_t n);

/*
 * Top-p (nucleus) sampling.
 * Apply temperature first, then softmax, then sample.
 * rng is a simple xorshift64 state; seed with any non-zero value.
 */
uint32_t sampler_top_p(float *logits, size_t n,
                        float temperature, float top_p, uint64_t *rng);

/*
 * Apply repetition penalty to logits.
 * ids   : previously generated token IDs
 * n_ids : number of IDs
 * penalty: >1.0 suppresses repetition (typical 1.1-1.3)
 */
void sampler_rep_penalty(float *logits, size_t n_vocab,
                          const uint32_t *ids, size_t n_ids, float penalty);

#ifdef __cplusplus
}
#endif
#endif /* SAMPLER_H */
