#ifndef GGUF_RAW_H
#define GGUF_RAW_H

#include "gguf.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF scalar value types */
#define GGUF_VTYPE_UINT8    0
#define GGUF_VTYPE_INT8     1
#define GGUF_VTYPE_UINT16   2
#define GGUF_VTYPE_INT16    3
#define GGUF_VTYPE_UINT32   4
#define GGUF_VTYPE_INT32    5
#define GGUF_VTYPE_FLOAT32  6
#define GGUF_VTYPE_BOOL     7
#define GGUF_VTYPE_STRING   8
#define GGUF_VTYPE_ARRAY    9
#define GGUF_VTYPE_UINT64   10
#define GGUF_VTYPE_INT64    11
#define GGUF_VTYPE_FLOAT64  12

/*
 * Low-level binary reader that walks the GGUF metadata section directly.
 * Needed for reading large array metadata (e.g. tokenizer vocab, merges)
 * that the summary-string API condenses.
 */
typedef struct {
    const uint8_t *base;  /* mmap start */
    const uint8_t *p;     /* current read cursor */
    const uint8_t *end;   /* exclusive end (= data_offset) */
    int error;            /* set non-zero on any read error */
} gguf_raw_t;

/* Position the reader at the start of the metadata KV section. */
void gguf_raw_init(gguf_raw_t *r, const gguf_file_t *g);

/*
 * Scan metadata for the given key.
 * On success r->p points at the value bytes and *vtype_out is set.
 * Returns 0 on success, -1 if not found or parse error.
 */
int gguf_raw_find(gguf_raw_t *r, const gguf_file_t *g,
                  const char *key, uint32_t *vtype_out);

/* Read a single uint32 from current position. */
int gguf_raw_u32(gguf_raw_t *r, uint32_t *out);
/* Read a single int32 from current position. */
int gguf_raw_i32(gguf_raw_t *r, int32_t *out);
/* Read a single float32 from current position. */
int gguf_raw_f32(gguf_raw_t *r, float *out);
/* Read a single uint64 from current position. */
int gguf_raw_u64(gguf_raw_t *r, uint64_t *out);

/*
 * Reader is positioned at an ARRAY value.
 * Reads the element type and count, calls cb for each string element.
 * Returns element count on success, -1 on error / type mismatch.
 */
int64_t gguf_raw_str_array(gguf_raw_t *r,
    void (*cb)(size_t idx, const char *s, size_t slen, void *ud),
    void *ud);

/*
 * Reader is positioned at an ARRAY value.
 * Reads the element type and count, calls cb for each uint32/int32 element.
 * Returns element count on success, -1 on error / type mismatch.
 */
int64_t gguf_raw_u32_array(gguf_raw_t *r,
    void (*cb)(size_t idx, uint32_t val, void *ud),
    void *ud);

/*
 * Reader is positioned at an ARRAY value.
 * Reads the element type and count, calls cb for each float32 element.
 * Returns element count on success, -1 on error / type mismatch.
 */
int64_t gguf_raw_f32_array(gguf_raw_t *r,
    void (*cb)(size_t idx, float val, void *ud),
    void *ud);

#ifdef __cplusplus
}
#endif
#endif /* GGUF_RAW_H */
