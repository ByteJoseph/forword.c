#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "gguf.h"
#include "model_config.h"
#include "tokenizer.h"
#include "kv_cache.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    gguf_file_t    g;         /* memory-mapped model */
    model_config_t cfg;
    tokenizer_t   *tok;
    kv_cache_t     kv;

    /* RoPE tables [n_ctx * head_dim/2] */
    float *cos_tab;
    float *sin_tab;

    /* Working buffers */
    float *x;       /* [n_embd]  current hidden state */
    float *xn;      /* [n_embd]  post-norm buffer */
    float *attn_out;/* [n_embd]  attention output */
    float *ffn_out; /* [n_embd]  FFN output */
    float *logits;  /* [n_vocab] */
    float *scratch; /* large scratch (>=max(n_ff*2+n_embd, n_ctx+n_embd)) */
} transformer_t;

/*
 * Load a GGUF model file and initialise all buffers.
 * max_ctx limits KV cache size (defaults to min(cfg.n_ctx, 4096)).
 * Returns 0 on success, -1 on error.
 */
int transformer_init(transformer_t *t, const char *path, size_t max_ctx);

/* Free all resources. */
void transformer_free(transformer_t *t);

/*
 * Run one forward step: feed token_id at sequence position pos.
 * Returns pointer to logits[n_vocab] (valid until next call), or NULL on error.
 */
float *transformer_forward(transformer_t *t, uint32_t token_id, size_t pos);

#ifdef __cplusplus
}
#endif
#endif /* TRANSFORMER_H */
