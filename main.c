#include <unistd.h>
#define _POSIX_C_SOURCE 200809L
#include "transformer.h"
#include "sampler.h"
#include "chat_template.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ── Command-line defaults ───────────────────────────────────────────── */
#define DEFAULT_MAX_TOKENS 512
#define DEFAULT_TEMP       0.7f
#define DEFAULT_TOP_P      0.9f
#define DEFAULT_MAX_CTX    2048
#define DEFAULT_REP_PEN    1.1f

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <model.gguf> [OPTIONS] [prompt]\n"
        "\n"
        "Options:\n"
        "  --system <msg>   System prompt (default: 'You are a helpful assistant.')\n"
        "  --max-tokens N   Max tokens to generate (default: %d)\n"
        "  --ctx N          KV-cache context size (default: %d)\n"
        "  --temp T         Temperature (default: %.1f; 0 = greedy)\n"
        "  --top-p P        Top-p nucleus sampling (default: %.1f)\n"
        "  --rep-pen R      Repetition penalty (default: %.1f; 1.0 = off)\n"
        "  --no-template    Pass prompt directly without chat template\n"
        "  --seed N         RNG seed (default: time-based)\n"
        "  -h, --help       Show this help\n"
        "\n"
        "If prompt is omitted, reads from stdin.\n",
        prog, DEFAULT_MAX_TOKENS, DEFAULT_MAX_CTX, DEFAULT_TEMP, DEFAULT_TOP_P, DEFAULT_REP_PEN);
}

/* ── Read a line from stdin ──────────────────────────────────────────── */
static char *read_stdin(void) {
    char *buf = NULL; size_t cap = 0, len = 0;
    int c;
    while ((c = getchar()) != EOF) {
        if (len + 2 > cap) {
            cap = cap ? cap * 2 : 256;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        buf[len++] = (char)c;
    }
    if (buf) buf[len] = '\0';
    else { buf = (char *)calloc(1, 1); }
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        usage(argv[0]); return argc < 2 ? 2 : 0;
    }

    const char *model_path  = argv[1];
    const char *system_msg  = "You are a helpful assistant.";
    const char *user_prompt = NULL;
    int   max_tokens        = DEFAULT_MAX_TOKENS;
    int   max_ctx           = DEFAULT_MAX_CTX;
    float temperature       = DEFAULT_TEMP;
    float top_p             = DEFAULT_TOP_P;
    float rep_pen           = DEFAULT_REP_PEN;
    int   no_template       = 0;
    uint64_t rng_seed       = (uint64_t)time(NULL);

    /* ── Parse arguments ─────────────────────────────────────────────── */
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--system") && i+1 < argc) {
            system_msg = argv[++i];
        } else if (!strcmp(argv[i], "--max-tokens") && i+1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ctx") && i+1 < argc) {
            max_ctx = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--temp") && i+1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--top-p") && i+1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--rep-pen") && i+1 < argc) {
            rep_pen = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i+1 < argc) {
            rng_seed = (uint64_t)atoll(argv[++i]);
        } else if (!strcmp(argv[i], "--no-template")) {
            no_template = 1;
        } else if (argv[i][0] != '-') {
            user_prompt = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]); return 2;
        }
    }

    /* Ensure RNG seed is non-zero */
    if (!rng_seed) rng_seed = 12345;

    /* ── Load model ───────────────────────────────────────────────────── */
    transformer_t model;
    if (transformer_init(&model, model_path, (size_t)max_ctx) != 0) return 1;

    /* ── Read prompt ───────────────────────────────────────────────────── */
    char *prompt_owned = NULL;
    if (!user_prompt) {
        if (!isatty(0)) {
            user_prompt = prompt_owned = read_stdin();
        } else {
            fprintf(stderr, "Enter prompt: ");
            fflush(stderr);
            user_prompt = prompt_owned = read_stdin();
        }
    }
    if (!user_prompt || !*user_prompt) {
        fprintf(stderr, "No prompt provided.\n");
        transformer_free(&model); free(prompt_owned); return 1;
    }

    /* ── Apply chat template ───────────────────────────────────────────── */
    static char formatted[65536];
    const char *prompt_to_encode = user_prompt;
    if (!no_template) {
        int n = chat_template_format(&model.cfg, system_msg, user_prompt,
                                     formatted, sizeof(formatted));
        if (n < 0) {
            fprintf(stderr, "Chat template output too large\n");
            transformer_free(&model); free(prompt_owned); return 1;
        }
        prompt_to_encode = formatted;
    }

    /* ── Tokenize prompt ──────────────────────────────────────────────── */
    uint32_t *prompt_ids = NULL; size_t prompt_n = 0;
    if (tokenizer_encode(model.tok, prompt_to_encode, &prompt_ids, &prompt_n) < 0) {
        fprintf(stderr, "Tokenizer encode failed\n");
        transformer_free(&model); free(prompt_owned); return 1;
    }

    fprintf(stderr, "Prompt tokens: %zu\n", prompt_n);

    /* ── Prefill (process prompt) ──────────────────────────────────────── */
    float *logits = NULL;
    for (size_t i = 0; i < prompt_n; i++) {
        logits = transformer_forward(&model, prompt_ids[i], i);
        if (!logits) {
            fprintf(stderr, "Forward pass failed at prompt position %zu\n", i);
            free(prompt_ids); transformer_free(&model); free(prompt_owned); return 1;
        }
    }

    /* ── Generation loop ───────────────────────────────────────────────── */
    const char *stop_str = chat_template_stop_str(&model.cfg);
    const size_t n_vocab  = model.cfg.n_vocab;

    /* Token history for repetition penalty */
    uint32_t *history = (uint32_t *)malloc((prompt_n + (size_t)max_tokens) * sizeof(uint32_t));
    if (!history) { free(prompt_ids); transformer_free(&model); free(prompt_owned); return 1; }
    memcpy(history, prompt_ids, prompt_n * sizeof(uint32_t));
    size_t history_n = prompt_n;

    /* Accumulate decoded output to detect stop string */
    static char output_acc[8192];
    size_t output_len = 0;
    output_acc[0] = '\0';

    float *logits_copy = (float *)malloc(n_vocab * sizeof(float));
    if (!logits_copy) { free(history); free(prompt_ids); transformer_free(&model); free(prompt_owned); return 1; }

    size_t gen = 0;

    for (; gen < (size_t)max_tokens; gen++) {
        if (!logits) break;

        /* Copy logits (sampler may modify in place) */
        memcpy(logits_copy, logits, n_vocab * sizeof(float));

        /* Repetition penalty */
        sampler_rep_penalty(logits_copy, n_vocab, history, history_n, rep_pen);

        /* Sample */
        uint32_t next_tok;
        if (temperature <= 0.0f) {
            next_tok = sampler_argmax(logits_copy, n_vocab);
        } else {
            next_tok = sampler_top_p(logits_copy, n_vocab, temperature, top_p, &rng_seed);
        }

        /* Check EOS */
        if (next_tok == tokenizer_eos(model.tok)) break;

        /* Decode and print */
        const char *tok_str = tokenizer_token_str(model.tok, next_tok);
        if (tok_str && *tok_str) {
            fputs(tok_str, stdout);
            fflush(stdout);

            /* Accumulate for stop-string detection */
            size_t ts_len = strlen(tok_str);
            if (output_len + ts_len < sizeof(output_acc) - 1) {
                memcpy(output_acc + output_len, tok_str, ts_len);
                output_len += ts_len;
                output_acc[output_len] = '\0';
            }
            /* Check stop string */
            if (stop_str && strstr(output_acc, stop_str)) break;
        }

        /* Update history */
        history[history_n++] = next_tok;

        /* Forward pass to compute logits for the next token */
        if (gen + 1 < (size_t)max_tokens) {
            logits = transformer_forward(&model, next_tok, prompt_n + gen);
            if (!logits) break;
        }
    }

    /* Ensure output ends with a newline */
    if (output_len && output_acc[output_len-1] != '\n') putchar('\n');

    fprintf(stderr, "\n[Generated %zu tokens]\n", gen);

    /* ── Cleanup ──────────────────────────────────────────────────────── */
    free(logits_copy);
    free(history);
    free(prompt_ids);
    free(prompt_owned);
    transformer_free(&model);
    return 0;
}
