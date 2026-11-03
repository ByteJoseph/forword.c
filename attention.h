#ifndef ATTENTION_H
#define ATTENTION_H

#include "model_config.h"
#include "kv_cache.h"
#include "gguf.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Run one attention layer forward pass for a single token at position pos.
 *
 * x_in     : input residual [n_embd]  (read-only)
 * x_out    : output [n_embd]
 * scratch  : working buffer, must be >= max(n_embd * 4, n_ctx + n_embd) floats
 * layer    : which transformer layer (0-based)
 * pos      : current sequence position (0-based)
 */
int attention_forward(
    const model_config_t *cfg,
    const gguf_file_t    *g,
    kv_cache_t           *kv,
    const float          *cos_tab,
    const float          *sin_tab,
    const float          *x_in,
    float                *x_out,
    float                *scratch,
    size_t                layer,
    size_t                pos
);

#ifdef __cplusplus
}
#endif
#endif /* ATTENTION_H */
