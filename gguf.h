#ifndef GGUF_H
#define GGUF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *name;
    char *value; /* Human-readable representation; arrays are summarized. */
} gguf_metadata_t;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t ggml_type;
    uint64_t offset;       /* Relative to tensor-data section. */
    uint64_t n_elements;
    uint64_t byte_size;    /* Known only for supported tensor types. */
} gguf_tensor_t;

typedef struct {
    int fd;
    void *mapping;
    size_t file_size;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint32_t alignment;
    uint64_t data_offset;
    gguf_metadata_t *metadata;
    size_t metadata_len;
    gguf_tensor_t *tensors;
    size_t tensors_len;
} gguf_file_t;

/* Returns 0 on success; -1 on error and writes a message to errbuf. */
int gguf_open(gguf_file_t *g, const char *path, char *errbuf, size_t errbuf_size);
void gguf_close(gguf_file_t *g);
const char *gguf_get_metadata(const gguf_file_t *g, const char *key);
void gguf_print_summary(const gguf_file_t *g);

/* GGML type IDs used by current GGUF files. */
const char *gguf_type_name(uint32_t type);

#ifdef __cplusplus
}
#endif
#endif
