#include <stdarg.h>
#define _POSIX_C_SOURCE 200809L
#include "chat_template.h"
#include <stdio.h>
#include <string.h>

/*
 * Chat templates for common architectures.
 *
 * These are the standard instruction-tuned prompt formats.
 * We detect them from the architecture name.
 */

/* ── Helper: safe snprintf into growing buffer ───────────────────────── */
static int safe_append(char *buf, size_t bufsz, size_t *used,
                        const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + *used, bufsz - *used, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= bufsz - *used) return -1;
    *used += (size_t)n;
    return 0;
}
#include <stdarg.h>

int chat_template_format(const model_config_t *cfg,
                         const char *system_msg,
                         const char *user_msg,
                         char *buf, size_t bufsz)
{
    if (!buf || !bufsz || !user_msg) return -1;
    size_t used = 0;
    const char *sys = (system_msg && *system_msg) ? system_msg : NULL;

#define APP(...) do { if (safe_append(buf,bufsz,&used,__VA_ARGS__)<0) return -1; } while(0)

    switch (cfg->arch) {
    /* ── Qwen2 / Qwen2.5 ─────────────────────────────────────────────── */
    case ARCH_QWEN2:
        if (sys) APP("<|im_start|>system\n%s<|im_end|>\n", sys);
        APP("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_msg);
        break;

    /* ── LLaMA-3 (uses begin_of_text + header tokens) ───────────────── */
    case ARCH_LLAMA: {
        /* Detect LLaMA-3 by checking for <|begin_of_text|> token */
        APP("<|begin_of_text|>");
        if (sys) APP("<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>", sys);
        APP("<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n", user_msg);
        break;
    }

    /* ── Mistral / Mixtral ───────────────────────────────────────────── */
    case ARCH_MISTRAL:
        if (sys) APP("[INST] %s\n\n%s [/INST]", sys, user_msg);
        else     APP("[INST] %s [/INST]", user_msg);
        break;

    /* ── Gemma ────────────────────────────────────────────────────────── */
    case ARCH_GEMMA:
        APP("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", user_msg);
        break;

    /* ── Phi-2 ────────────────────────────────────────────────────────── */
    case ARCH_PHI2:
        if (sys) APP("System: %s\n", sys);
        APP("Instruct: %s\nOutput:", user_msg);
        break;

    /* ── Phi-3 ────────────────────────────────────────────────────────── */
    case ARCH_PHI3:
        if (sys) APP("<|system|>\n%s<|end|>\n", sys);
        APP("<|user|>\n%s<|end|>\n<|assistant|>\n", user_msg);
        break;

    /* ── StarCoder2 ────────────────────────────────────────────────────── */
    case ARCH_STARCODER:
        /* No chat template for StarCoder; just pass the prompt raw */
        APP("%s", user_msg);
        break;

    /* ── Falcon ───────────────────────────────────────────────────────── */
    case ARCH_FALCON:
        if (sys) APP("System: %s\n", sys);
        APP("User: %s\nFalcon:", user_msg);
        break;

    /* ── Default / unknown: just pass raw ─────────────────────────────── */
    default:
        APP("%s", user_msg);
        break;
    }
#undef APP

    buf[used] = '\0';
    return (int)used;
}

const char *chat_template_stop_str(const model_config_t *cfg) {
    switch (cfg->arch) {
    case ARCH_QWEN2:     return "<|im_end|>";
    case ARCH_LLAMA:     return "<|eot_id|>";
    case ARCH_MISTRAL:   return "</s>";
    case ARCH_GEMMA:     return "<end_of_turn>";
    case ARCH_PHI2:
    case ARCH_PHI3:      return "<|end|>";
    case ARCH_STARCODER: return "<|endoftext|>";
    default:             return NULL;
    }
}
