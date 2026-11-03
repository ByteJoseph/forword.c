#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include "gguf.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Architecture ────────────────────────────────────────────────────────── */
typedef enum {
    ARCH_UNKNOWN = 0,
    ARCH_GPT2,
    ARCH_LLAMA,       /* llama 1/2/3, code-llama  */
    ARCH_QWEN2,       /* Qwen / Qwen2 / Qwen2.5   */
    ARCH_MISTRAL,     /* Mistral / Mixtral         */
    ARCH_GEMMA,       /* Gemma / Gemma2            */
    ARCH_PHI2,        /* Phi-2                     */
    ARCH_PHI3,        /* Phi-3 (llama-like)        */
    ARCH_STARCODER,   /* StarCoder 2               */
    ARCH_FALCON,      /* Falcon                    */
} model_arch_t;

/* ── Normalisation type ──────────────────────────────────────────────────── */
typedef enum {
    NORM_RMS = 0,   /* RMSNorm (llama, qwen2, mistral, gemma …) */
    NORM_LAYER,     /* LayerNorm (phi-2, falcon, gpt-2 …)        */
} norm_type_t;

/* ── FFN activation ──────────────────────────────────────────────────────── */
typedef enum {
    ACT_SILU = 0,   /* SiLU / Swish (SwiGLU)                    */
    ACT_GELU,       /* GELU (exact)                              */
    ACT_GELU_FAST,  /* approx GELU  (phi-2 uses this)           */
    ACT_RELU,
} act_type_t;

/* ── RoPE variant ───────────────────────────────────────────────────────── */
typedef enum {
    ROPE_NORMAL = 0,  /* standard RoPE (llama, qwen2, mistral …) */
    ROPE_NEOX,        /* GPT-NeoX style interleaving              */
    ROPE_NONE,        /* no positional encoding                   */
} rope_type_t;

/* ── Main configuration struct ──────────────────────────────────────────── */
typedef struct {
    /* Identity */
    model_arch_t arch;
    char arch_str[64];   /* raw string from GGUF, e.g. "qwen2" */

    /* Dimensions */
    uint32_t n_vocab;
    uint32_t n_ctx;      /* max context length */
    uint32_t n_embd;     /* hidden / embedding dimension */
    uint32_t n_layer;    /* number of transformer blocks */
    uint32_t n_head;     /* number of query heads */
    uint32_t n_head_kv;  /* number of key/value heads (GQA; == n_head if no GQA) */
    uint32_t n_ff;       /* intermediate FFN dimension */
    uint32_t head_dim;   /* n_embd / n_head */
    uint32_t n_rot;      /* RoPE rotation dimensions (default = head_dim) */

    /* Normalisation */
    norm_type_t norm_type;
    float       norm_eps;

    /* FFN */
    act_type_t act_type;
    int        use_ffn_gate;   /* 1 = SwiGLU (gate × up), 0 = plain up */

    /* Attention */
    int use_qkv_bias;    /* 1 = add bias after QKV projections (Qwen2) */
    int use_alibi;       /* 1 = ALiBi position bias (no RoPE)           */
    int parallel_res;    /* 1 = parallel residual stream (Phi-2)         */

    /* RoPE */
    rope_type_t rope_type;
    float       rope_freq_base;
    float       rope_freq_scale;

    /* Embedding */
    float embd_scale;   /* Multiply embeddings by this (Gemma: sqrt(n_embd)) */

    /* Tokenizer */
    char     token_model[32];   /* "gpt2", "llama", "bert" */
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
} model_config_t;

/*
 * Populate *cfg from the GGUF metadata.
 * Returns 0 on success, -1 if the GGUF lacks required fields.
 */
int model_config_load(model_config_t *cfg, const gguf_file_t *g);

/* Human-readable arch name. */
const char *model_arch_name(model_arch_t arch);

#ifdef __cplusplus
}
#endif
#endif /* MODEL_CONFIG_H */
