#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf.h"
#include "model_config.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tokenizer handle. */
typedef struct tokenizer tokenizer_t;

/*
 * Build the tokenizer from a loaded GGUF file.
 * Reads tokenizer.ggml.tokens, tokenizer.ggml.merges and related metadata.
 * Returns 0 on success; -1 on error (writes message to stderr).
 */
int  tokenizer_init(tokenizer_t **out, const gguf_file_t *g,
                    const model_config_t *cfg);

void tokenizer_free(tokenizer_t *tok);

/*
 * Encode a UTF-8 string to token IDs.
 * Does NOT prepend BOS automatically; caller decides.
 * *out_ids is malloc'd; caller must free().
 * Returns number of tokens on success, -1 on error.
 */
int tokenizer_encode(const tokenizer_t *tok, const char *text,
                     uint32_t **out_ids, size_t *out_n);

/*
 * Decode token IDs to a UTF-8 string.
 * Returned string is malloc'd; caller must free().
 * Returns NULL on error.
 */
char *tokenizer_decode(const tokenizer_t *tok,
                       const uint32_t *ids, size_t n);

/* Decode a single token to UTF-8 (static buffer; not thread-safe). */
const char *tokenizer_token_str(const tokenizer_t *tok, uint32_t id);

/* Special token IDs (as configured). */
uint32_t tokenizer_bos(const tokenizer_t *tok);
uint32_t tokenizer_eos(const tokenizer_t *tok);

/* Return total vocab size. */
size_t tokenizer_vocab_size(const tokenizer_t *tok);

#ifdef __cplusplus
}
#endif
#endif /* TOKENIZER_H */
