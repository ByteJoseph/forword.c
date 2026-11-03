
#include "quant_matvec.h"
#include "gguf.h"
#include "tensor_data.h"
#include "quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int test_type(const gguf_file_t *g, uint32_t type)
{
    const size_t block_bytes = (type == 12) ? 144 : 210;
    const char *type_name = (type == 12) ? "Q4_K" : "Q6_K";

    for (size_t i = 0; i < g->tensors_len; ++i) {
        const gguf_tensor_t *t = &g->tensors[i];

        if (t->ggml_type != type)
            continue;

        if (t->byte_size < block_bytes)
            continue;

        size_t actual_size = 0;
        const void *raw = tensor_data(g, t, &actual_size);

        if (!raw || actual_size < block_bytes) {
            fprintf(stderr, "%s: invalid tensor data: %s\n",
                    type_name, t->name);
            return -1;
        }

        float values[QK_K];

        int rc;
        if (type == 12) {
            rc = q4_k_decode(raw, block_bytes, values, QK_K);
        } else {
            rc = q6_k_decode(raw, block_bytes, values, QK_K);
        }

        if (rc != 0) {
            fprintf(stderr, "%s: decoder failed for %s\n",
                    type_name, t->name);
            return -1;
        }

        float minimum = values[0];
        float maximum = values[0];

        for (size_t j = 0; j < QK_K; ++j) {
            if (!isfinite(values[j])) {
                fprintf(stderr,
                        "%s: non-finite value at index %zu\n",
                        type_name, j);
                return -1;
            }

            if (values[j] < minimum)
                minimum = values[j];

            if (values[j] > maximum)
                maximum = values[j];
        }

        printf("\n%s test: PASS\n", type_name);
        printf("Tensor: %s\n", t->name);
        printf("Decoded values: %d\n", QK_K);
        printf("Minimum: %.7g\n", minimum);
        printf("Maximum: %.7g\n", maximum);
        printf("First 12 values:\n");

        for (size_t j = 0; j < 12; ++j) {
            printf("  [%zu] %.7g\n", j, values[j]);
        }

        return 0;
    }

    fprintf(stderr, "No suitable %s tensor found\n", type_name);
    return -1;
}


int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 2;
    }

    gguf_file_t g;
    char err[256] = {0};

    if (gguf_open(&g, argv[1], err, sizeof err) != 0) {
        fprintf(stderr, "GGUF open failed: %s\n", err);
        return 1;
    }

    int q4_result = test_type(&g, 12);
    int q6_result = test_type(&g, 14);

    /*
     * Test a real Q4_K matrix from the model.
     * GGUF dimensions are stored with dims[0] as columns
     * and dims[1] as rows for this matrix-vector operation.
     */
    const gguf_tensor_t *t =
        tensor_find(&g, "blk.0.attn_q.weight");

    int matvec_result = -1;

    if (!t) {
        fprintf(stderr, "Attention Q tensor not found\n");
    } else if (t->ggml_type != 12 ||
               t->n_dims < 2 ||
               t->dims[0] > SIZE_MAX ||
               t->dims[1] > SIZE_MAX) {
        fprintf(stderr, "Unexpected attention Q tensor layout\n");
    } else {
        size_t input_size = (size_t)t->dims[0];
        size_t output_size = (size_t)t->dims[1];
        size_t raw_size = 0;

        const void *raw = tensor_data(&g, t, &raw_size);

        float *input = calloc(input_size, sizeof(float));
        float *output = calloc(output_size, sizeof(float));

        if (!raw || !input || !output) {
            fprintf(stderr, "Could not allocate matrix test buffers\n");
        } else {
            /* Deterministic, nonzero test input. */
            for (size_t i = 0; i < input_size; ++i)
                input[i] = 0.01f * ((float)(i % 17) - 8.0f);

            matvec_result = quant_matvec(
                raw, raw_size, t->ggml_type,
                input, output, input_size, output_size);

            if (matvec_result == 0) {
                int finite = 1;

                for (size_t i = 0; i < output_size; ++i) {
                    if (!isfinite(output[i])) {
                        finite = 0;
                        break;
                    }
                }

                if (finite) {
                    puts("\nQ4_K matrix-vector test: PASS");
                    printf("Tensor: %s\n", t->name);
                    printf("Input size: %zu\n", input_size);
                    printf("Output size: %zu\n", output_size);
                    printf("First 8 output values:\n");

                    size_t count = output_size < 8
                        ? output_size : 8;

                    for (size_t i = 0; i < count; ++i)
                        printf("  [%zu] %.7g\n", i, output[i]);
                } else {
                    fprintf(stderr, "Non-finite matrix output\n");
                    matvec_result = -1;
                }
            } else {
                fprintf(stderr, "quant_matvec failed\n");
            }
        }

        free(input);
        free(output);
    }

    gguf_close(&g);

    if (q4_result != 0 || q6_result != 0 ||
        matvec_result != 0)
        return 1;

    puts("\nAll current tests passed.");
    puts("Matrix output sanity is not yet reference validation.");
    return 0;
}

