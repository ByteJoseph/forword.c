#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Key-Value cache for a decoder-only transformer.
 * Layout: [layer][pos][head][dim_element]  (row-major)
 */
typedef struct {
    float  *k;          /* [n_layer * n_ctx * n_head_kv * head_dim] */
    float  *v;          /* same shape */
    size_t  n_layer;
    size_t  n_ctx;
    size_t  n_head_kv;
    size_t  head_dim;
} kv_cache_t;

/* Initialise and zero the cache. Returns 0 on success, -1 on OOM. */
static inline int kv_cache_init(kv_cache_t *kv,
    size_t n_layer, size_t n_ctx, size_t n_head_kv, size_t head_dim)
{
    kv->n_layer   = n_layer;
    kv->n_ctx     = n_ctx;
    kv->n_head_kv = n_head_kv;
    kv->head_dim  = head_dim;
    size_t total  = n_layer * n_ctx * n_head_kv * head_dim;
    kv->k = (float *)calloc(total, sizeof(float));
    kv->v = (float *)calloc(total, sizeof(float));
    if (!kv->k || !kv->v) { free(kv->k); free(kv->v); kv->k = kv->v = NULL; return -1; }
    return 0;
}

static inline void kv_cache_free(kv_cache_t *kv) {
    free(kv->k); free(kv->v);
    kv->k = kv->v = NULL;
}

/* Pointer to K vector for layer l, position p, KV head h. */
static inline float *kv_k(kv_cache_t *kv, size_t l, size_t p, size_t h) {
    return kv->k + ((l * kv->n_ctx + p) * kv->n_head_kv + h) * kv->head_dim;
}
/* Pointer to V vector for layer l, position p, KV head h. */
static inline float *kv_v(kv_cache_t *kv, size_t l, size_t p, size_t h) {
    return kv->v + ((l * kv->n_ctx + p) * kv->n_head_kv + h) * kv->head_dim;
}

#ifdef __cplusplus
}
#endif
#endif /* KV_CACHE_H */
