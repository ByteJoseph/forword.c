
#ifndef TENSOR_DATA_H
#define TENSOR_DATA_H

#include "gguf.h"
#include <stddef.h>

/* Find a tensor by its exact GGUF name. */
const gguf_tensor_t *tensor_find(
    const gguf_file_t *g,
    const char *name
);

/* Return a pointer to tensor bytes, or NULL if bounds are invalid. */
const void *tensor_data(
    const gguf_file_t *g,
    const gguf_tensor_t *tensor,
    size_t *size_out
);

#endif
