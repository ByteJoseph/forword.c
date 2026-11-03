#ifndef FFN_H
#define FFN_H

#include "model_config.h"
#include "gguf.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Run one FFN layer forward pass for a single token.
 *
 * x_in  : input [n_embd]   (read-only)
 * x_out : output [n_embd]
 * scratch : working buffer, must be >= n_ff * 3 floats
 * layer : which transformer layer (0-based)
 */
int ffn_forward(
    const model_config_t *cfg,
    const gguf_file_t    *g,
    const float          *x_in,
    float                *x_out,
    float                *scratch,
    size_t                layer
);

#ifdef __cplusplus
}
#endif
#endif /* FFN_H */
