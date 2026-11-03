#ifndef LINEAR_H
#define LINEAR_H

/*
 * Unified linear layer forward pass.
 * Dispatches over F32, Q4_K, Q6_K (and BF16/F16 with conversion).
 * Also provides row dequantisation for embedding tables.
 */

#include "gguf.h"
#include "tensor_data.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * y = W x + b
 *
 * weight : GGUF tensor (dims[0]=in_size, dims[1]=out_size)
 * bias   : optional GGUF tensor (may be NULL)
 * x      : float input  [in_size]
 * y      : float output [out_size]
 * tmp    : scratch buffer of at least out_size floats (used for dequant)
 *
 * Returns 0 on success, -1 on error.
 */
int linear_forward(const gguf_file_t    *g,
                   const gguf_tensor_t  *weight,
                   const gguf_tensor_t  *bias,
                   const float          *x,
                   float                *y,
                   float                *tmp);

/*
 * Dequantise row `row_idx` of a quantised weight matrix into `out` [row_width].
 * row_width == dims[0] of the tensor.
 *
 * Used for embedding table lookup (row = token_id).
 * Returns 0 on success, -1 on error.
 */
int linear_get_row(const gguf_file_t   *g,
                   const gguf_tensor_t *weight,
                   size_t               row_idx,
                   float               *out);

/*
 * Convert a 16-bit brain-float (BF16) or half-float (F16) value to float32.
 * ggml_type 1 = F16, 30 = BF16.
 */
float f16_to_f32(uint16_t v);
float bf16_to_f32(uint16_t v);

#ifdef __cplusplus
}
#endif
#endif /* LINEAR_H */
