#define _POSIX_C_SOURCE 200809L
#include "transformer.h"
#include "attention.h"
#include "ffn.h"
#include "norm.h"
#include "rope.h"
#include "linear.h"
#include "tensor_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int transformer_init(transformer_t *t, const char *path, size_t max_ctx) {
    memset(t, 0, sizeof(*t));

    char errbuf[256];
    if (gguf_open(&t->g, path, errbuf, sizeof(errbuf)) != 0) {
        fprintf(stderr, "gguf_open: %s\n", errbuf);
        return -1;
    }

    if (model_config_load(&t->cfg, &t->g) != 0) {
        fprintf(stderr, "transformer_init: failed to read model config\n");
        gguf_close(&t->g);
        return -1;
    }

    model_config_t *c = &t->cfg;

    fprintf(stderr,
        "Model: %s arch=%s layers=%u embd=%u heads=%u/%u ff=%u vocab=%u\n",
        gguf_get_metadata(&t->g, "general.name")
            ? gguf_get_metadata(&t->g, "general.name") : "(unnamed)",
        c->arch_str, c->n_layer, c->n_embd, c->n_head,
        c->n_head_kv, c->n_ff, c->n_vocab);

    if (!c->n_embd || !c->n_head || !c->n_layer) {
        fprintf(stderr, "transformer_init: invalid model dimensions\n");
        goto fail;
    }

    if (!max_ctx)
        max_ctx = c->n_ctx < 4096 ? c->n_ctx : 4096;

    if (max_ctx > c->n_ctx)
        max_ctx = c->n_ctx;

    if (!max_ctx) {
        fprintf(stderr, "transformer_init: invalid context length\n");
        goto fail;
    }

    if (tokenizer_init(&t->tok, &t->g, c) != 0) {
        fprintf(stderr, "transformer_init: tokenizer_init failed\n");
        goto fail;
    }

    if (!c->n_vocab)
        c->n_vocab = (uint32_t)tokenizer_vocab_size(t->tok);

    if (!c->n_vocab) {
        fprintf(stderr, "transformer_init: invalid vocabulary size\n");
        goto fail;
    }

    if (kv_cache_init(&t->kv, c->n_layer, max_ctx,
                      c->n_head_kv, c->head_dim) != 0) {
        fprintf(stderr, "transformer_init: KV cache allocation failed\n");
        goto fail;
    }

    size_t half = c->head_dim / 2;
    size_t rope_count = max_ctx * half;

    t->cos_tab = (float *)malloc((rope_count ? rope_count : 1) *
                                 sizeof(float));
    t->sin_tab = (float *)malloc((rope_count ? rope_count : 1) *
                                 sizeof(float));

    if (!t->cos_tab || !t->sin_tab) {
        fprintf(stderr, "transformer_init: RoPE allocation failed\n");
        goto fail;
    }

    if (c->rope_type != ROPE_NONE && half != 0) {
        rope_precompute(t->cos_tab, t->sin_tab, max_ctx,
                        c->head_dim, c->rope_freq_base);
    }

    t->x        = (float *)malloc(c->n_embd * sizeof(float));
    t->xn       = (float *)malloc(c->n_embd * sizeof(float));
    t->attn_out = (float *)malloc(c->n_embd * sizeof(float));
    t->ffn_out  = (float *)malloc(c->n_embd * sizeof(float));
    t->logits   = (float *)malloc((size_t)c->n_vocab * sizeof(float));

    /*
     * Attention scratch:
     * Q + K + V + attention scores + attention output + combined QKV.
     * Combined QKV is only needed by GPT-2, but reserving it here
     * keeps the attention scratch layout simple.
     */
    size_t attn_scratch =
          (size_t)c->n_head * c->head_dim
        + (size_t)c->n_head_kv * c->head_dim
        + (size_t)c->n_head_kv * c->head_dim
        + max_ctx
        + c->n_embd
        + (size_t)3 * c->n_embd;

    size_t ffn_scratch = (size_t)2 * c->n_ff + c->n_embd;
    size_t scratch_sz = attn_scratch > ffn_scratch
                      ? attn_scratch : ffn_scratch;

    t->scratch = (float *)malloc(scratch_sz * sizeof(float));

    if (!t->x || !t->xn || !t->attn_out || !t->ffn_out ||
        !t->logits || !t->scratch) {
        fprintf(stderr, "transformer_init: working buffer allocation failed\n");
        goto fail;
    }

    return 0;

fail:
    transformer_free(t);
    return -1;
}

void transformer_free(transformer_t *t) {
    if (!t) return;

    free(t->cos_tab);
    free(t->sin_tab);
    free(t->x);
    free(t->xn);
    free(t->attn_out);
    free(t->ffn_out);
    free(t->logits);
    free(t->scratch);

    kv_cache_free(&t->kv);
    tokenizer_free(t->tok);
    gguf_close(&t->g);

    memset(t, 0, sizeof(*t));
}

float *transformer_forward(transformer_t *t,
                           uint32_t token_id,
                           size_t pos) {
    const model_config_t *c = &t->cfg;
    const gguf_file_t *g = &t->g;
    const size_t n_embd = c->n_embd;

    if (pos >= t->kv.n_ctx) {
        fprintf(stderr, "transformer_forward: position exceeds KV cache\n");
        return NULL;
    }

    if (token_id >= c->n_vocab) {
        fprintf(stderr, "transformer_forward: token ID out of range\n");
        return NULL;
    }

    /* 1. Token embedding */
    const gguf_tensor_t *embd_w = tensor_find(g, "token_embd.weight");

    if (!embd_w) {
        fprintf(stderr, "transformer_forward: token_embd.weight not found\n");
        return NULL;
    }

    if (linear_get_row(g, embd_w, token_id, t->x) != 0) {
        fprintf(stderr, "transformer_forward: token embedding lookup failed\n");
        return NULL;
    }

    if (c->embd_scale != 1.0f) {
        for (size_t i = 0; i < n_embd; i++)
            t->x[i] *= c->embd_scale;
    }

    /*
     * GPT-2/TinyStories uses learned absolute positional embeddings:
     * hidden = token_embedding + position_embedding.
     */
    if (c->arch == ARCH_GPT2) {
        const gguf_tensor_t *pos_w =
            tensor_find(g, "position_embd.weight");

        if (!pos_w) {
            fprintf(stderr,
                    "transformer_forward: position_embd.weight not found\n");
            return NULL;
        }

        if (linear_get_row(g, pos_w, pos, t->xn) != 0) {
            fprintf(stderr,
                    "transformer_forward: position embedding lookup failed\n");
            return NULL;
        }

        for (size_t i = 0; i < n_embd; i++)
            t->x[i] += t->xn[i];
    }

    /* 2. Transformer blocks */
    for (size_t layer = 0; layer < c->n_layer; layer++) {
        char nm[128];

        snprintf(nm, sizeof(nm), "blk.%zu.attn_norm.weight", layer);
        const gguf_tensor_t *attn_nw = tensor_find(g, nm);

        snprintf(nm, sizeof(nm), "blk.%zu.attn_norm.bias", layer);
        const gguf_tensor_t *attn_nb = tensor_find(g, nm);

        snprintf(nm, sizeof(nm), "blk.%zu.ffn_norm.weight", layer);
        const gguf_tensor_t *ffn_nw = tensor_find(g, nm);

        snprintf(nm, sizeof(nm), "blk.%zu.ffn_norm.bias", layer);
        const gguf_tensor_t *ffn_nb = tensor_find(g, nm);

        const float *an_w = attn_nw
            ? (const float *)tensor_data(g, attn_nw, NULL) : NULL;
        const float *an_b = attn_nb
            ? (const float *)tensor_data(g, attn_nb, NULL) : NULL;
        const float *fn_w = ffn_nw
            ? (const float *)tensor_data(g, ffn_nw, NULL) : NULL;
        const float *fn_b = ffn_nb
            ? (const float *)tensor_data(g, ffn_nb, NULL) : NULL;

        if (c->parallel_res) {
            /* Parallel residual architecture */
            if (c->norm_type == NORM_LAYER)
                layer_norm(t->x, an_w, an_b, t->xn,
                           n_embd, c->norm_eps);
            else
                rms_norm(t->x, an_w, t->xn,
                         n_embd, c->norm_eps);

            if (attention_forward(c, g, &t->kv,
                                  t->cos_tab, t->sin_tab,
                                  t->xn, t->attn_out, t->scratch,
                                  layer, pos) != 0)
                return NULL;

            if (ffn_forward(c, g, t->xn, t->ffn_out,
                            t->scratch, layer) != 0)
                return NULL;

            for (size_t i = 0; i < n_embd; i++)
                t->x[i] += t->attn_out[i] + t->ffn_out[i];

        } else {
            /* Attention pre-norm */
            if (c->norm_type == NORM_LAYER)
                layer_norm(t->x, an_w, an_b, t->xn,
                           n_embd, c->norm_eps);
            else
                rms_norm(t->x, an_w, t->xn,
                         n_embd, c->norm_eps);

            if (attention_forward(c, g, &t->kv,
                                  t->cos_tab, t->sin_tab,
                                  t->xn, t->attn_out, t->scratch,
                                  layer, pos) != 0)
                return NULL;

            /* Attention residual */
            for (size_t i = 0; i < n_embd; i++)
                t->x[i] += t->attn_out[i];

            /* FFN pre-norm */
            if (c->norm_type == NORM_LAYER)
                layer_norm(t->x, fn_w, fn_b, t->xn,
                           n_embd, c->norm_eps);
            else
                rms_norm(t->x, fn_w, t->xn,
                         n_embd, c->norm_eps);

            if (ffn_forward(c, g, t->xn, t->ffn_out,
                            t->scratch, layer) != 0)
                return NULL;

            /* FFN residual */
            for (size_t i = 0; i < n_embd; i++)
                t->x[i] += t->ffn_out[i];
        }
    }

    /* 3. Final normalization */
    const gguf_tensor_t *on_w = tensor_find(g, "output_norm.weight");
    const gguf_tensor_t *on_b = tensor_find(g, "output_norm.bias");

    if (on_w) {
        const float *w = (const float *)tensor_data(g, on_w, NULL);
        const float *b = on_b
            ? (const float *)tensor_data(g, on_b, NULL) : NULL;

        if (c->norm_type == NORM_LAYER)
            layer_norm(t->x, w, b, t->xn, n_embd, c->norm_eps);
        else
            rms_norm(t->x, w, t->xn, n_embd, c->norm_eps);
    } else {
        memcpy(t->xn, t->x, n_embd * sizeof(float));
    }

    /* 4. Language-model head */
    const gguf_tensor_t *out_w = tensor_find(g, "output.weight");
    if (!out_w)
        out_w = tensor_find(g, "lm_head.weight");

    if (!out_w)
        out_w = tensor_find(g, "token_embd.weight");

    if (!out_w) {
        fprintf(stderr, "transformer_forward: no output weight tensor\n");
        return NULL;
    }

    if (linear_forward(g, out_w, NULL, t->xn, t->logits, NULL) != 0)
        return NULL;

    return t->logits;
}