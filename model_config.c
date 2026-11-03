#define _POSIX_C_SOURCE 200809L
#include "model_config.h"
#include "gguf_raw.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ── Architecture table ─────────────────────────────────────────────────── */

static const struct { const char *name; model_arch_t id; } arch_map[] = {
    { "llama",     ARCH_LLAMA     },
    { "qwen2",     ARCH_QWEN2     },
    { "mistral",   ARCH_MISTRAL   },
    { "gemma",     ARCH_GEMMA     },
    { "gemma2",    ARCH_GEMMA     },
    { "phi2",      ARCH_PHI2      },
    { "phi3",      ARCH_PHI3      },
    { "starcoder", ARCH_STARCODER },
    { "falcon",    ARCH_FALCON    },
        { "gpt2", ARCH_GPT2 },
    { NULL, ARCH_UNKNOWN }
};

const char *model_arch_name(model_arch_t arch) {
    for (int i = 0; arch_map[i].name; i++)
        if (arch_map[i].id == arch) return arch_map[i].name;
    return "unknown";
}

/* ── Helpers to read scalar metadata values ─────────────────────────────── */

/* Read a uint32 metadata value; return def on missing/error. */
static uint32_t meta_u32(const gguf_file_t *g, const char *key, uint32_t def) {
    gguf_raw_t r; uint32_t vtype;
    if (gguf_raw_find(&r, g, key, &vtype)) return def;
    if (vtype == GGUF_VTYPE_UINT32 || vtype == GGUF_VTYPE_INT32) {
        uint32_t v; if (!gguf_raw_u32(&r, &v)) return v;
    }
    return def;
}

/* Read a float32 metadata value; return def on missing/error. */
static float meta_f32(const gguf_file_t *g, const char *key, float def) {
    gguf_raw_t r; uint32_t vtype;
    if (gguf_raw_find(&r, g, key, &vtype)) return def;
    if (vtype == GGUF_VTYPE_FLOAT32) {
        float v; if (!gguf_raw_f32(&r, &v)) return v;
    }
    return def;
}

/* Build a prefixed key: e.g. "qwen2" + ".embedding_length" → "qwen2.embedding_length" */
static void make_key(char *buf, size_t bufsz,
                     const char *arch, const char *suffix) {
    snprintf(buf, bufsz, "%s.%s", arch, suffix);
}

/* ── Main loader ─────────────────────────────────────────────────────────── */

int model_config_load(model_config_t *cfg, const gguf_file_t *g) {
    memset(cfg, 0, sizeof(*cfg));

    /* Architecture */
    const char *arch_s = gguf_get_metadata(g, "general.architecture");
    if (!arch_s) { fprintf(stderr, "model_config: missing general.architecture\n"); return -1; }
    snprintf(cfg->arch_str, sizeof(cfg->arch_str), "%s", arch_s);

    cfg->arch = ARCH_UNKNOWN;
    for (int i = 0; arch_map[i].name; i++) {
        if (!strcmp(arch_map[i].name, arch_s)) { cfg->arch = arch_map[i].id; break; }
    }

    const char *a = arch_s; /* alias for key building */
    char key[128];

    /* Core dimensions */
    make_key(key, sizeof(key), a, "embedding_length");
    cfg->n_embd = meta_u32(g, key, 0);
    if (!cfg->n_embd) { fprintf(stderr, "model_config: missing n_embd\n"); return -1; }

    make_key(key, sizeof(key), a, "block_count");
    cfg->n_layer = meta_u32(g, key, 0);
    if (!cfg->n_layer) { fprintf(stderr, "model_config: missing n_layer\n"); return -1; }

    make_key(key, sizeof(key), a, "attention.head_count");
    cfg->n_head = meta_u32(g, key, 0);
    if (!cfg->n_head) { fprintf(stderr, "model_config: missing n_head\n"); return -1; }

    make_key(key, sizeof(key), a, "attention.head_count_kv");
    cfg->n_head_kv = meta_u32(g, key, cfg->n_head); /* default: no GQA */

    make_key(key, sizeof(key), a, "feed_forward_length");
    cfg->n_ff = meta_u32(g, key, 0);

    make_key(key, sizeof(key), a, "context_length");
    cfg->n_ctx = meta_u32(g, key, 2048);

    /* Derive head_dim */
    cfg->head_dim = cfg->n_embd / cfg->n_head;

    /* RoPE dimensions */
    make_key(key, sizeof(key), a, "rope.dimension_count");
    cfg->n_rot = meta_u32(g, key, cfg->head_dim);

    /* Norm epsilon */
    make_key(key, sizeof(key), a, "attention.layer_norm_rms_epsilon");
    cfg->norm_eps = meta_f32(g, key, 1e-5f);
    if (cfg->norm_eps == 1e-5f) {
        /* Some arches use a different key */
        make_key(key, sizeof(key), a, "attention.layer_norm_epsilon");
        cfg->norm_eps = meta_f32(g, key, 1e-5f);
    }

    /* RoPE freq base */
    make_key(key, sizeof(key), a, "rope.freq_base");
    cfg->rope_freq_base = meta_f32(g, key, 10000.0f);

    make_key(key, sizeof(key), a, "rope.scale_linear");
    cfg->rope_freq_scale = meta_f32(g, key, 1.0f);

    /* Tokenizer */
    const char *tok_model = gguf_get_metadata(g, "tokenizer.ggml.model");
    if (tok_model) snprintf(cfg->token_model, sizeof(cfg->token_model), "%s", tok_model);
    else           snprintf(cfg->token_model, sizeof(cfg->token_model), "gpt2");

    cfg->bos_token_id = meta_u32(g, "tokenizer.ggml.bos_token_id", 1);
    cfg->eos_token_id = meta_u32(g, "tokenizer.ggml.eos_token_id", 2);
    cfg->pad_token_id = meta_u32(g, "tokenizer.ggml.padding_token_id", cfg->eos_token_id);

    /* Vocabulary size from the token array count via metadata string (approx) */
    /* We'll refine this from the actual tokenizer load; set a rough value for now */
    cfg->n_vocab = meta_u32(g, "tokenizer.ggml.token_type_count", 0);
    if (!cfg->n_vocab) {
        /* Try to find output.weight dims */
        const gguf_tensor_t *out_w = NULL;
        for (size_t i = 0; i < g->tensors_len; i++) {
            if (!strcmp(g->tensors[i].name, "output.weight") ||
                !strcmp(g->tensors[i].name, "lm_head.weight")) {
                out_w = &g->tensors[i];
                break;
            }
        }
        if (out_w && out_w->n_dims >= 2) cfg->n_vocab = (uint32_t)out_w->dims[1];
        else if (out_w && out_w->n_dims == 1) cfg->n_vocab = (uint32_t)out_w->dims[0];
    }

    /* ── Per-architecture defaults ─────────────────────────────────────── */
    switch (cfg->arch) {
    case ARCH_LLAMA:
    case ARCH_MISTRAL:
    case ARCH_PHI3:
    case ARCH_STARCODER:
        cfg->norm_type     = NORM_RMS;
        cfg->act_type      = ACT_SILU;
        cfg->use_ffn_gate  = 1;
        cfg->rope_type     = ROPE_NORMAL;
        cfg->embd_scale    = 1.0f;
        break;

    case ARCH_QWEN2:
        cfg->norm_type     = NORM_RMS;
        cfg->act_type      = ACT_SILU;
        cfg->use_ffn_gate  = 1;
        cfg->rope_type     = ROPE_NORMAL;
        cfg->use_qkv_bias  = 1; /* Qwen2 adds Q, K, V bias vectors */
        cfg->embd_scale    = 1.0f;
        if (cfg->rope_freq_base < 10001.0f)
            cfg->rope_freq_base = 1000000.0f; /* Qwen2 default: 1M */
        break;

    case ARCH_GEMMA:
        cfg->norm_type     = NORM_RMS;
        cfg->act_type      = ACT_GELU_FAST;
        cfg->use_ffn_gate  = 1;
        cfg->rope_type     = ROPE_NORMAL;
        cfg->embd_scale    = sqrtf((float)cfg->n_embd);
        break;

    case ARCH_PHI2:
        cfg->norm_type     = NORM_LAYER;
        cfg->act_type      = ACT_GELU_FAST;
        cfg->use_ffn_gate  = 0;
        cfg->rope_type     = ROPE_NEOX;
        cfg->parallel_res  = 1;
        cfg->embd_scale    = 1.0f;
        break;

    case ARCH_FALCON:
        cfg->norm_type     = NORM_LAYER;
        cfg->act_type      = ACT_GELU;
        cfg->use_ffn_gate  = 0;
        cfg->rope_type     = ROPE_NORMAL;
        cfg->use_alibi     = 0;
        cfg->embd_scale    = 1.0f;
        break;
    case ARCH_GPT2:
        cfg->norm_type     = NORM_LAYER;
        cfg->act_type      = ACT_GELU;
        cfg->use_ffn_gate  = 0;
        cfg->rope_type     = ROPE_NONE;
        cfg->use_qkv_bias  = 1;
        cfg->use_alibi     = 0;
        cfg->parallel_res  = 0;
        cfg->embd_scale    = 1.0f;
        cfg->n_head_kv     = cfg->n_head;
        cfg->n_rot         = 0;
        break;
    default:
        /* Best-effort defaults for unknown architectures */
        cfg->norm_type     = NORM_RMS;
        cfg->act_type      = ACT_SILU;
        cfg->use_ffn_gate  = 1;
        cfg->rope_type     = ROPE_NORMAL;
        cfg->embd_scale    = 1.0f;
        break;
    }

    return 0;
}
