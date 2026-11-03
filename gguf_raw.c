#define _POSIX_C_SOURCE 200809L
#include "gguf_raw.h"
#include <string.h>

/* ── Low-level helpers ───────────────────────────────────────────────────── */

static int rr_u8(gguf_raw_t *r, uint8_t *v) {
    if (r->p + 1 > r->end) { r->error = 1; return -1; }
    *v = *r->p++; return 0;
}
static int rr_u16(gguf_raw_t *r, uint16_t *v) {
    if (r->p + 2 > r->end) { r->error = 1; return -1; }
    memcpy(v, r->p, 2); r->p += 2; return 0;
}
static int rr_u32(gguf_raw_t *r, uint32_t *v) {
    if (r->p + 4 > r->end) { r->error = 1; return -1; }
    memcpy(v, r->p, 4); r->p += 4; return 0;
}
static int rr_u64(gguf_raw_t *r, uint64_t *v) {
    if (r->p + 8 > r->end) { r->error = 1; return -1; }
    memcpy(v, r->p, 8); r->p += 8; return 0;
}
static int rr_f32(gguf_raw_t *r, float *v) {
    return rr_u32(r, (uint32_t *)v);
}
static int rr_f64(gguf_raw_t *r, double *v) {
    return rr_u64(r, (uint64_t *)v);
}
static int rr_str(gguf_raw_t *r, const char **sp, uint64_t *lp) {
    uint64_t len;
    if (rr_u64(r, &len)) return -1;
    if (r->p + len > r->end) { r->error = 1; return -1; }
    if (sp) *sp = (const char *)r->p;
    if (lp) *lp = len;
    r->p += len;
    return 0;
}

/* Skip one GGUF value (recursive for arrays). */
static int rr_skip(gguf_raw_t *r, uint32_t type) {
    uint8_t  u8;  uint16_t u16; uint32_t u32; uint64_t u64;
    float f; double d;
    switch (type) {
    case GGUF_VTYPE_UINT8:   return rr_u8(r, &u8);
    case GGUF_VTYPE_INT8:    return rr_u8(r, &u8);
    case GGUF_VTYPE_UINT16:  return rr_u16(r, &u16);
    case GGUF_VTYPE_INT16:   return rr_u16(r, &u16);
    case GGUF_VTYPE_UINT32:  return rr_u32(r, &u32);
    case GGUF_VTYPE_INT32:   return rr_u32(r, &u32);
    case GGUF_VTYPE_FLOAT32: return rr_f32(r, &f);
    case GGUF_VTYPE_BOOL:    return rr_u8(r, &u8);
    case GGUF_VTYPE_STRING:  return rr_str(r, NULL, NULL);
    case GGUF_VTYPE_UINT64:  return rr_u64(r, &u64);
    case GGUF_VTYPE_INT64:   return rr_u64(r, &u64);
    case GGUF_VTYPE_FLOAT64: return rr_f64(r, &d);
    case GGUF_VTYPE_ARRAY: {
        uint32_t etype; uint64_t cnt;
        if (rr_u32(r, &etype) || rr_u64(r, &cnt)) return -1;
        for (uint64_t i = 0; i < cnt; i++)
            if (rr_skip(r, etype)) return -1;
        return 0;
    }
    default: r->error = 1; return -1;
    }
}

/* ── Public API ──────────────────────────────────────────────────────────── */

void gguf_raw_init(gguf_raw_t *r, const gguf_file_t *g) {
    r->base  = (const uint8_t *)g->mapping;
    r->end   = r->base + g->data_offset;
    /* Skip: magic(4) + version(4) + tensor_count(8) + metadata_count(8) = 24 bytes */
    r->p     = r->base + 24;
    r->error = 0;
}

int gguf_raw_find(gguf_raw_t *r, const gguf_file_t *g,
                  const char *key, uint32_t *vtype_out)
{
    size_t klen = strlen(key);
    gguf_raw_init(r, g);

    for (uint64_t i = 0; i < g->metadata_count; i++) {
        const char *k; uint64_t kl;
        if (rr_str(r, &k, &kl)) return -1;

        uint32_t vtype;
        if (rr_u32(r, &vtype)) return -1;

        if (kl == klen && memcmp(k, key, klen) == 0) {
            if (vtype_out) *vtype_out = vtype;
            return 0;   /* r->p is now at the value */
        }

        if (rr_skip(r, vtype)) return -1;
    }
    return -1; /* not found */
}

int gguf_raw_u32(gguf_raw_t *r, uint32_t *out) { return rr_u32(r, out); }
int gguf_raw_i32(gguf_raw_t *r, int32_t  *out) { return rr_u32(r, (uint32_t *)out); }
int gguf_raw_f32(gguf_raw_t *r, float    *out) { return rr_f32(r, out); }
int gguf_raw_u64(gguf_raw_t *r, uint64_t *out) { return rr_u64(r, out); }

int64_t gguf_raw_str_array(gguf_raw_t *r,
    void (*cb)(size_t idx, const char *s, size_t slen, void *ud), void *ud)
{
    uint32_t etype; uint64_t cnt;
    if (rr_u32(r, &etype) || rr_u64(r, &cnt)) return -1;
    if (etype != GGUF_VTYPE_STRING) return -1;
    for (uint64_t i = 0; i < cnt; i++) {
        const char *s; uint64_t sl;
        if (rr_str(r, &s, &sl)) return -1;
        if (cb) cb((size_t)i, s, (size_t)sl, ud);
    }
    return (int64_t)cnt;
}

int64_t gguf_raw_u32_array(gguf_raw_t *r,
    void (*cb)(size_t idx, uint32_t val, void *ud), void *ud)
{
    uint32_t etype; uint64_t cnt;
    if (rr_u32(r, &etype) || rr_u64(r, &cnt)) return -1;
    if (etype != GGUF_VTYPE_UINT32 && etype != GGUF_VTYPE_INT32) return -1;
    for (uint64_t i = 0; i < cnt; i++) {
        uint32_t v;
        if (rr_u32(r, &v)) return -1;
        if (cb) cb((size_t)i, v, ud);
    }
    return (int64_t)cnt;
}

int64_t gguf_raw_f32_array(gguf_raw_t *r,
    void (*cb)(size_t idx, float val, void *ud), void *ud)
{
    uint32_t etype; uint64_t cnt;
    if (rr_u32(r, &etype) || rr_u64(r, &cnt)) return -1;
    if (etype != GGUF_VTYPE_FLOAT32) return -1;
    for (uint64_t i = 0; i < cnt; i++) {
        float v;
        if (rr_f32(r, &v)) return -1;
        if (cb) cb((size_t)i, v, ud);
    }
    return (int64_t)cnt;
}
