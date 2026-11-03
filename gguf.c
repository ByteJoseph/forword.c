#define _POSIX_C_SOURCE 200809L
#include "gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u /* "GGUF" little-endian */
#define MAX_STRING_BYTES (64u * 1024u * 1024u)
#define MAX_ITEMS 10000000u

enum {
    GGUF_METADATA_UINT8 = 0, GGUF_METADATA_INT8 = 1,
    GGUF_METADATA_UINT16 = 2, GGUF_METADATA_INT16 = 3,
    GGUF_METADATA_UINT32 = 4, GGUF_METADATA_INT32 = 5,
    GGUF_METADATA_FLOAT32 = 6, GGUF_METADATA_BOOL = 7,
    GGUF_METADATA_STRING = 8, GGUF_METADATA_ARRAY = 9,
    GGUF_METADATA_UINT64 = 10, GGUF_METADATA_INT64 = 11,
    GGUF_METADATA_FLOAT64 = 12
};

typedef struct {
    const unsigned char *p;
    const unsigned char *end;
} reader_t;

static void seterr(char *dst, size_t cap, const char *fmt, ...) {
    if (!dst || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dst, cap, fmt, ap);
    va_end(ap);
}

static int take(reader_t *r, void *out, size_t n) {
    if ((size_t)(r->end - r->p) < n) return -1;
    if (out) memcpy(out, r->p, n);
    r->p += n;
    return 0;
}

static int read_u32(reader_t *r, uint32_t *v) { return take(r, v, sizeof(*v)); }
static int read_u64(reader_t *r, uint64_t *v) { return take(r, v, sizeof(*v)); }

static int read_string(reader_t *r, char **out) {
    uint64_t n;
    if (read_u64(r, &n) || n > MAX_STRING_BYTES || n > (uint64_t)(r->end-r->p))
        return -1;
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) return -1;
    memcpy(s, r->p, (size_t)n);
    s[n] = '\0';
    r->p += (size_t)n;
    *out = s;
    return 0;
}

static int skip_bytes(reader_t *r, uint64_t n) {
    if (n > (uint64_t)(r->end-r->p)) return -1;
    r->p += (size_t)n;
    return 0;
}

static int read_scalar(reader_t *r, uint32_t type, char *out, size_t cap) {
    uint8_t u8; int8_t i8; uint16_t u16; int16_t i16;
    uint32_t u32; int32_t i32; uint64_t u64; int64_t i64;
    float f32; double f64;
    switch (type) {
    case GGUF_METADATA_UINT8:
        if (take(r,&u8,1)) return -1; snprintf(out,cap,"%" PRIu8,u8); return 0;
    case GGUF_METADATA_INT8:
        if (take(r,&i8,1)) return -1; snprintf(out,cap,"%" PRId8,i8); return 0;
    case GGUF_METADATA_UINT16:
        if (take(r,&u16,2)) return -1; snprintf(out,cap,"%" PRIu16,u16); return 0;
    case GGUF_METADATA_INT16:
        if (take(r,&i16,2)) return -1; snprintf(out,cap,"%" PRId16,i16); return 0;
    case GGUF_METADATA_UINT32:
        if (take(r,&u32,4)) return -1; snprintf(out,cap,"%" PRIu32,u32); return 0;
    case GGUF_METADATA_INT32:
        if (take(r,&i32,4)) return -1; snprintf(out,cap,"%" PRId32,i32); return 0;
    case GGUF_METADATA_FLOAT32:
        if (take(r,&f32,4)) return -1; snprintf(out,cap,"%.9g",(double)f32); return 0;
    case GGUF_METADATA_BOOL:
        if (take(r,&u8,1) || u8 > 1) return -1;
        snprintf(out,cap,"%s",u8 ? "true":"false"); return 0;
    case GGUF_METADATA_UINT64:
        if (take(r,&u64,8)) return -1; snprintf(out,cap,"%" PRIu64,u64); return 0;
    case GGUF_METADATA_INT64:
        if (take(r,&i64,8)) return -1; snprintf(out,cap,"%" PRId64,i64); return 0;
    case GGUF_METADATA_FLOAT64:
        if (take(r,&f64,8)) return -1; snprintf(out,cap,"%.17g",f64); return 0;
    case GGUF_METADATA_STRING:
        return -1; /* handled separately */
    default: return -1;
    }
}

static int skip_value(reader_t *r, uint32_t type, unsigned depth) {
    if (depth > 8) return -1;
    char tmp[64], *s = NULL;
    uint32_t elem_type;
    uint64_t count;
    switch (type) {
    case GGUF_METADATA_STRING:
        if (read_string(r, &s)) return -1;
        free(s); return 0;
    case GGUF_METADATA_ARRAY:
        if (read_u32(r, &elem_type) || read_u64(r, &count) || count > MAX_ITEMS) return -1;
        for (uint64_t i=0; i<count; ++i)
            if (skip_value(r, elem_type, depth+1)) return -1;
        return 0;
    default:
        return read_scalar(r, type, tmp, sizeof(tmp));
    }
}

static int value_to_text(reader_t *r, uint32_t type, char **out) {
    char tmp[128];
    if (type == GGUF_METADATA_STRING) return read_string(r, out);
    if (type == GGUF_METADATA_ARRAY) {
        uint32_t elem_type; uint64_t count;
        if (read_u32(r,&elem_type) || read_u64(r,&count) || count > MAX_ITEMS) return -1;
        for (uint64_t i=0; i<count; ++i)
            if (skip_value(r, elem_type, 0)) return -1;
        snprintf(tmp,sizeof(tmp),"[array: %" PRIu64 " items, type %u]",count,elem_type);
    } else if (read_scalar(r,type,tmp,sizeof(tmp))) {
        return -1;
    }
    *out = strdup(tmp);
    return *out ? 0 : -1;
}

const char *gguf_type_name(uint32_t t) {
    switch (t) {
    case 0: return "F32"; case 1: return "F16";
    case 2: return "Q4_0"; case 3: return "Q4_1";
    case 6: return "Q5_0"; case 7: return "Q5_1";
    case 8: return "Q8_0"; case 9: return "Q8_1";
    case 10: return "Q2_K"; case 11: return "Q3_K";
    case 12: return "Q4_K"; case 13: return "Q5_K";
    case 14: return "Q6_K"; case 15: return "Q8_K";
    case 16: return "IQ2_XXS"; case 17: return "IQ2_XS";
    case 18: return "IQ3_XXS"; case 19: return "IQ1_S";
    case 20: return "IQ4_NL"; case 21: return "IQ3_S";
    case 22: return "IQ2_S"; case 23: return "IQ4_XS";
    case 24: return "I8"; case 25: return "I16";
    case 26: return "I32"; case 27: return "I64";
    case 28: return "F64"; case 29: return "IQ1_M";
    case 30: return "BF16"; case 34: return "TQ1_0";
    case 35: return "TQ2_0"; default: return "UNKNOWN";
    }
}


/*
 * Calculate GGML tensor storage sizes.
 *
 * Return values:
 *   0  = size calculated successfully
 *   1  = unsupported tensor type; size remains unknown
 *  -1  = invalid dimensions or arithmetic overflow
 *
 * Quantized formats store blocks, not individual bytes per element.
 */
static int tensor_size(
    uint32_t type,
    uint32_t n_dims,
    const uint64_t *dims,
    uint64_t elems,
    uint64_t *bytes
) {
    uint64_t block_size = 1;
    uint64_t block_bytes = 0;

    *bytes = 0;

    /* Ordinary scalar tensor types. */
    switch (type) {
        case 0:  block_bytes = 4;  break; /* F32 */
        case 1:  block_bytes = 2;  break; /* F16 */
        case 24: block_bytes = 1;  break; /* I8 */
        case 25: block_bytes = 2;  break; /* I16 */
        case 26: block_bytes = 4;  break; /* I32 */
        case 27: block_bytes = 8;  break; /* I64 */
        case 28: block_bytes = 8;  break; /* F64 */
        case 30: block_bytes = 2;  break; /* BF16 */

        /* Legacy and common block quantization formats. */
        case 2:  block_size = 32;  block_bytes = 18;  break; /* Q4_0 */
        case 3:  block_size = 32;  block_bytes = 20;  break; /* Q4_1 */
        case 6:  block_size = 32;  block_bytes = 22;  break; /* Q5_0 */
        case 7:  block_size = 32;  block_bytes = 24;  break; /* Q5_1 */
        case 8:  block_size = 32;  block_bytes = 34;  break; /* Q8_0 */
        case 9:  block_size = 32;  block_bytes = 40;  break; /* Q8_1 */

        /* K-quant formats used by many GGUF models. */
        case 10: block_size = 256; block_bytes = 84;  break; /* Q2_K */
        case 11: block_size = 256; block_bytes = 110; break; /* Q3_K */
        case 12: block_size = 256; block_bytes = 144; break; /* Q4_K */
        case 13: block_size = 256; block_bytes = 176; break; /* Q5_K */
        case 14: block_size = 256; block_bytes = 210; break; /* Q6_K */
        case 15: block_size = 256; block_bytes = 292; break; /* Q8_K */

        default:
            /* Do not guess sizes for unknown or unimplemented formats. */
            return 1;
    }

    if (n_dims == 0 || dims == NULL) {
        return -1;
    }

    if (elems == 0) {
        *bytes = 0;
        return 0;
    }

    /* Every quantized row must contain complete blocks. */
    if (dims[0] % block_size != 0 || elems % block_size != 0) {
        return -1;
    }

    uint64_t blocks = elems / block_size;

    if (blocks > UINT64_MAX / block_bytes) {
        return -1;
    }

    *bytes = blocks * block_bytes;
    return 0;
}


int gguf_open(gguf_file_t *g, const char *path, char *errbuf, size_t errcap) {
    memset(g,0,sizeof(*g)); g->fd=-1;
    g->alignment=32;
    g->fd=open(path,O_RDONLY);
    if (g->fd<0) { seterr(errbuf,errcap,"open %s: %s",path,strerror(errno)); return -1; }
    struct stat st;
    if (fstat(g->fd,&st) || st.st_size < 24 || (uintmax_t)st.st_size > SIZE_MAX) {
        seterr(errbuf,errcap,"invalid or too-large file"); gguf_close(g); return -1;
    }
    g->file_size=(size_t)st.st_size;
    g->mapping=mmap(NULL,g->file_size,PROT_READ,MAP_PRIVATE,g->fd,0);
    if (g->mapping==MAP_FAILED) {
        g->mapping=NULL; seterr(errbuf,errcap,"mmap: %s",strerror(errno)); gguf_close(g); return -1;
    }
    reader_t r={(const unsigned char *)g->mapping,(const unsigned char *)g->mapping+g->file_size};
    uint32_t magic;
    if (read_u32(&r,&magic) || magic!=GGUF_MAGIC || read_u32(&r,&g->version) ||
        read_u64(&r,&g->tensor_count) || read_u64(&r,&g->metadata_count)) {
        seterr(errbuf,errcap,"not a valid GGUF header"); gguf_close(g); return -1;
    }
    if (g->version < 2 || g->version > 3 || g->tensor_count > MAX_ITEMS || g->metadata_count > MAX_ITEMS) {
        seterr(errbuf,errcap,"unsupported GGUF version/counts"); gguf_close(g); return -1;
    }
    g->metadata=calloc((size_t)g->metadata_count,sizeof(*g->metadata));
    if (g->metadata_count && !g->metadata) { seterr(errbuf,errcap,"out of memory"); gguf_close(g); return -1; }
    for (uint64_t i=0;i<g->metadata_count;i++) {
        char *key=NULL,*value=NULL; uint32_t type;
        if (read_string(&r,&key) || read_u32(&r,&type) || value_to_text(&r,type,&value)) {
            free(key); free(value); seterr(errbuf,errcap,"malformed metadata entry %" PRIu64,i); gguf_close(g); return -1;
        }
        g->metadata[g->metadata_len].name=key;
        g->metadata[g->metadata_len].value=value;
        g->metadata_len++;
        if (!strcmp(key,"general.alignment")) {
            char *end=NULL; unsigned long n=strtoul(value,&end,10);
            if (end && *end=='\\0' && n>0 && n<=4096 && (n&(n-1))==0) g->alignment=(uint32_t)n;
        }
    }
    g->tensors=calloc((size_t)g->tensor_count,sizeof(*g->tensors));
    if (g->tensor_count && !g->tensors) { seterr(errbuf,errcap,"out of memory"); gguf_close(g); return -1; }
    for (uint64_t i=0;i<g->tensor_count;i++) {
        gguf_tensor_t *t=&g->tensors[g->tensors_len];
        if (read_string(&r,&t->name) || read_u32(&r,&t->n_dims) || t->n_dims>4 || t->n_dims==0) {
            seterr(errbuf,errcap,"malformed tensor descriptor %" PRIu64,i); gguf_close(g); return -1;
        }
        t->n_elements=1;
        for (uint32_t d=0;d<t->n_dims;d++) {
            if (read_u64(&r,&t->dims[d]) || (t->dims[d] && t->n_elements>UINT64_MAX/t->dims[d])) {
                seterr(errbuf,errcap,"invalid tensor dimensions"); gguf_close(g); return -1;
            }
            t->n_elements*=t->dims[d];
        }
        if (read_u32(&r,&t->ggml_type) || read_u64(&r,&t->offset)) {
            seterr(errbuf,errcap,"truncated tensor descriptor"); gguf_close(g); return -1;
        }
        
int size_status = tensor_size(
    t->ggml_type,
    t->n_dims,
    t->dims,
    t->n_elements,
    &t->byte_size
);

if (size_status < 0) {
    seterr(
        errbuf,
        errcap,
        "invalid dimensions for tensor '%s' (type %s)",
        t->name,
        gguf_type_name(t->ggml_type)
    );

    free(t->name);
    t->name = NULL;
    gguf_close(g);
    return -1;
}

/* size_status == 1 means the format is not implemented yet. */

        g->tensors_len++;
    }
    uintptr_t after_desc=(uintptr_t)(r.p-(const unsigned char *)g->mapping);
    uint64_t aligned=((uint64_t)after_desc + g->alignment-1) & ~((uint64_t)g->alignment-1);
    if (aligned > g->file_size) { seterr(errbuf,errcap,"tensor data offset outside file"); gguf_close(g); return -1; }
    g->data_offset=aligned;
    for (size_t i=0;i<g->tensors_len;i++) {
        gguf_tensor_t *t=&g->tensors[i];
        if (t->offset > g->file_size-g->data_offset) {
            seterr(errbuf,errcap,"tensor '%s' offset outside file",t->name); gguf_close(g); return -1;
        }
        if (t->byte_size && t->byte_size > g->file_size-g->data_offset-t->offset) {
            seterr(errbuf,errcap,"tensor '%s' extends beyond file",t->name); gguf_close(g); return -1;
        }
    }
    return 0;
}

void gguf_close(gguf_file_t *g) {
    if (!g) return;
    for (size_t i=0;i<g->metadata_len;i++) { free(g->metadata[i].name); free(g->metadata[i].value); }
    for (size_t i=0;i<g->tensors_len;i++) free(g->tensors[i].name);
    free(g->metadata); free(g->tensors);
    if (g->mapping) munmap(g->mapping,g->file_size);
    if (g->fd>=0) close(g->fd);
    memset(g,0,sizeof(*g)); g->fd=-1;
}

const char *gguf_get_metadata(const gguf_file_t *g,const char *key) {
    for (size_t i=0;i<g->metadata_len;i++)
        if (!strcmp(g->metadata[i].name,key)) return g->metadata[i].value;
    return NULL;
}

void gguf_print_summary(const gguf_file_t *g) {
    printf("GGUF version:      %u\n",g->version);
    printf("File size:         %zu bytes\n",g->file_size);
    printf("Metadata entries:  %zu\n",g->metadata_len);
    printf("Tensor count:      %zu\n",g->tensors_len);
    printf("Alignment:         %u\n",g->alignment);
    printf("Tensor data start: %" PRIu64 "\n",g->data_offset);
    const char *arch=gguf_get_metadata(g,"general.architecture");
    const char *name=gguf_get_metadata(g,"general.name");
    const char *ctx=arch ? gguf_get_metadata(g,"llama.context_length") : NULL;
    printf("Architecture:      %s\n",arch?arch:"(not specified)");
    printf("Model name:        %s\n",name?name:"(not specified)");
    if (ctx) printf("Llama context:     %s\n",ctx);
    puts("\nTensor descriptors:");
    for (size_t i=0;i<g->tensors_len;i++) {
        const gguf_tensor_t *t=&g->tensors[i];
        printf("  %-40s type=%-8s dims=",t->name,gguf_type_name(t->ggml_type));
        for (uint32_t d=0;d<t->n_dims;d++) printf("%s%" PRIu64,d?"x":"",t->dims[d]);
        printf(" offset=%" PRIu64,t->offset);
        if (t->byte_size) printf(" bytes=%" PRIu64,t->byte_size);
        else printf(" bytes=(type layout required)");
        putchar('\n');
    }
}
