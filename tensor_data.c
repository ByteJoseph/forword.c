
#include "tensor_data.h"

#include <stdint.h>
#include <string.h>
#include <stdint.h>

const gguf_tensor_t *tensor_find(
    const gguf_file_t *g,
    const char *name)
{
    if (!g || !name || !g->tensors)
        return NULL;

    for (size_t i = 0; i < g->tensors_len; ++i) {
        if (g->tensors[i].name &&
            strcmp(g->tensors[i].name, name) == 0) {
            return &g->tensors[i];
        }
    }

    return NULL;
}

const void *tensor_data(
    const gguf_file_t *g,
    const gguf_tensor_t *tensor,
    size_t *size_out)
{
    if (size_out)
        *size_out = 0;

    if (!g || !tensor || !g->mapping)
        return NULL;

    /*
     * Reject offsets that overflow or point outside the mapped file.
     */
    if (g->data_offset > (uint64_t)g->file_size)
        return NULL;

    if (tensor->offset > UINT64_MAX - g->data_offset)
        return NULL;

    uint64_t absolute = g->data_offset + tensor->offset;

    if (absolute > (uint64_t)g->file_size)
        return NULL;

    uint64_t available = (uint64_t)g->file_size - absolute;

    if (tensor->byte_size > available)
        return NULL;

    if (tensor->byte_size > (uint64_t)SIZE_MAX)
        return NULL;

    if (size_out)
        *size_out = (size_t)tensor->byte_size;

    return (const unsigned char *)g->mapping + (size_t)absolute;
}
