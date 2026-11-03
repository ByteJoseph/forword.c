#define _POSIX_C_SOURCE 200809L
#include "attention.h"
#include "linear.h"
#include "rope.h"
#include "norm.h"
#include "tensor_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void layer_tensor_name(char *buf, size_t sz,
                              size_t layer, const char *suffix) {
    snprintf(buf, sz, "blk.%zu.%s", layer, suffix);
}

static void softmax(float *x, size_t n) {
    if (n == 0) return;
    float mx = x[0];
    for (size_t i = 1; i < n; i++)
        if (x[i] > mx) mx = x[i];

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return;

    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

int attention_forward(
    const model_config_t *cfg,
    const gguf_file_t *g,
    kv_cache_t *kv,
    const float *cos_tab,
    const float *sin_tab,
    const float *x_in,
    float *x_out,
    float *scratch,
    size_t layer,
    size_t pos)
{
    const size_t n_embd = cfg->n_embd;
    const size_t n_head = cfg->n_head;
    const size_t n_kv = cfg->n_head_kv;
    const size_t head_dim = cfg->head_dim;

    if (n_kv == 0 || n_head == 0 || head_dim == 0 ||
        n_head % n_kv != 0) {
        fprintf(stderr, "attention: invalid head configuration\n");
        return -1;
    }

    float *q_buf = scratch;
    float *k_buf = q_buf + n_head * head_dim;
    float *v_buf = k_buf + n_kv * head_dim;
    float *attn_sc = v_buf + n_kv * head_dim;
    float *out_buf = attn_sc + (pos + 1);

    char nm[128];
    const gguf_tensor_t *wo = NULL;
    const gguf_tensor_t *bo = NULL;

    if (cfg->arch == ARCH_GPT2) {
        /*
         * GPT-2 combined projection output is [Q | K | V].
         * The three sections are contiguous in q_buf, k_buf, v_buf
         * only when the projection result is written to a temporary
         * buffer, so use the start of scratch for the full projection.
         */
        float *qkv = out_buf + n_embd;
        layer_tensor_name(nm, sizeof(nm), layer, "attn_qkv.weight");
        const gguf_tensor_t *wqkv = tensor_find(g, nm);
        layer_tensor_name(nm, sizeof(nm), layer, "attn_qkv.bias");
        const gguf_tensor_t *bqkv = tensor_find(g, nm);

        layer_tensor_name(nm, sizeof(nm), layer, "attn_output.weight");
        wo = tensor_find(g, nm);
        layer_tensor_name(nm, sizeof(nm), layer, "attn_output.bias");
        bo = tensor_find(g, nm);

        if (!wqkv || !wo) {
            fprintf(stderr, "attention: missing GPT-2 tensor at layer %zu\n", layer);
            return -1;
        }

        if (linear_forward(g, wqkv, bqkv, x_in, qkv, NULL) != 0)
            return -1;

        memcpy(q_buf, qkv, n_embd * sizeof(float));
        memcpy(k_buf, qkv + n_embd, n_embd * sizeof(float));
        memcpy(v_buf, qkv + 2 * n_embd, n_embd * sizeof(float));
    } else {
        layer_tensor_name(nm, sizeof(nm), layer, "attn_q.weight");
        const gguf_tensor_t *wq = tensor_find(g, nm);
        layer_tensor_name(nm, sizeof(nm), layer, "attn_k.weight");
        const gguf_tensor_t *wk = tensor_find(g, nm);
        layer_tensor_name(nm, sizeof(nm), layer, "attn_v.weight");
        const gguf_tensor_t *wv = tensor_find(g, nm);
        layer_tensor_name(nm, sizeof(nm), layer, "attn_output.weight");
        wo = tensor_find(g, nm);
        layer_tensor_name(nm, sizeof(nm), layer, "attn_output.bias");
        bo = tensor_find(g, nm);

        if (!wq || !wk || !wv || !wo) {
            fprintf(stderr, "attention: missing weight tensor at layer %zu\n", layer);
            return -1;
        }

        const gguf_tensor_t *bq = NULL, *bk = NULL, *bv = NULL;
        if (cfg->use_qkv_bias) {
            layer_tensor_name(nm, sizeof(nm), layer, "attn_q.bias");
            bq = tensor_find(g, nm);
            layer_tensor_name(nm, sizeof(nm), layer, "attn_k.bias");
            bk = tensor_find(g, nm);
            layer_tensor_name(nm, sizeof(nm), layer, "attn_v.bias");
            bv = tensor_find(g, nm);
        }

        if (linear_forward(g, wq, bq, x_in, q_buf, NULL) != 0) return -1;
        if (linear_forward(g, wk, bk, x_in, k_buf, NULL) != 0) return -1;
        if (linear_forward(g, wv, bv, x_in, v_buf, NULL) != 0) return -1;
    }

    if (cfg->rope_type != ROPE_NONE) {
        int neox = (cfg->rope_type == ROPE_NEOX);
        rope_apply(q_buf, pos, n_head, head_dim, cos_tab, sin_tab, neox);
        rope_apply(k_buf, pos, n_kv, head_dim, cos_tab, sin_tab, neox);
    }

    for (size_t h = 0; h < n_kv; h++) {
        float *kc = kv_k(kv, layer, pos, h);
        float *vc = kv_v(kv, layer, pos, h);
        memcpy(kc, k_buf + h * head_dim, head_dim * sizeof(float));
        memcpy(vc, v_buf + h * head_dim, head_dim * sizeof(float));
    }

    const size_t gqa_group = n_head / n_kv;
    const float scale = 1.0f / sqrtf((float)head_dim);
    memset(out_buf, 0, n_embd * sizeof(float));

    for (size_t h = 0; h < n_head; h++) {
        size_t kv_h = h / gqa_group;
        const float *qh = q_buf + h * head_dim;
        float *oh = out_buf + h * head_dim;

        for (size_t t = 0; t <= pos; t++) {
            const float *kt = kv_k(kv, layer, t, kv_h);
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; d++)
                dot += qh[d] * kt[d];
            attn_sc[t] = dot * scale;
        }

        softmax(attn_sc, pos + 1);

        for (size_t t = 0; t <= pos; t++) {
            const float *vt = kv_v(kv, layer, t, kv_h);
            float w = attn_sc[t];
            for (size_t d = 0; d < head_dim; d++)
                oh[d] += w * vt[d];
        }
    }

    if (linear_forward(g, wo, bo, out_buf, x_out, NULL) != 0)
        return -1;

    return 0;
}