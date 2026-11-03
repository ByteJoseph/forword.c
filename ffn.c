
#define _POSIX_C_SOURCE 200809L
#include "ffn.h"
#include "linear.h"
#include "tensor_data.h"
#include "norm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void layer_name(char *buf, size_t sz, size_t layer,
                       const char *sfx) {
    snprintf(buf, sz, "blk.%zu.%s", layer, sfx);
}

static void apply_silu(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        float sig = v >= 0.0f
            ? 1.0f / (1.0f + expf(-v))
            : expf(v) / (1.0f + expf(v));
        x[i] = v * sig;
    }
}

static void apply_gelu(float *x, size_t n) {
    const float inv_sqrt2 = 0.7071067811865476f;
    for (size_t i = 0; i < n; i++)
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * inv_sqrt2));
}

static void apply_gelu_fast(float *x, size_t n) {
    const float c1 = 0.7978845608028654f;
    const float c2 = 0.044715f;

    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        float inner = c1 * (v + c2 * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

int ffn_forward(
    const model_config_t *cfg,
    const gguf_file_t *g,
    const float *x_in,
    float *x_out,
    float *scratch,
    size_t layer)
{
    const size_t n_ff = cfg->n_ff;

    float *gate_buf = scratch;
    float *up_buf = scratch + n_ff;

    char nm[128];

    layer_name(nm, sizeof(nm), layer, "ffn_down.weight");
    const gguf_tensor_t *wd = tensor_find(g, nm);

    if (!wd) {
        fprintf(stderr,
                "ffn: missing ffn_down.weight at layer %zu\n", layer);
        return -1;
    }

    if (cfg->use_ffn_gate) {
        /* Gated FFN: down(activation(gate(x)) * up(x)) */
        layer_name(nm, sizeof(nm), layer, "ffn_gate.weight");
        const gguf_tensor_t *wg = tensor_find(g, nm);

        layer_name(nm, sizeof(nm), layer, "ffn_up.weight");
        const gguf_tensor_t *wu = tensor_find(g, nm);

        if (!wg || !wu) {
            fprintf(stderr,
                    "ffn: missing gate/up weights at layer %zu\n", layer);
            return -1;
        }

        if (linear_forward(g, wg, NULL, x_in, gate_buf, NULL) != 0)
            return -1;

        if (linear_forward(g, wu, NULL, x_in, up_buf, NULL) != 0)
            return -1;

        switch (cfg->act_type) {
        case ACT_SILU:
            apply_silu(gate_buf, n_ff);
            break;
        case ACT_GELU:
            apply_gelu(gate_buf, n_ff);
            break;
        case ACT_GELU_FAST:
            apply_gelu_fast(gate_buf, n_ff);
            break;
        default:
            apply_silu(gate_buf, n_ff);
            break;
        }

        for (size_t i = 0; i < n_ff; i++)
            gate_buf[i] *= up_buf[i];

        if (linear_forward(g, wd, NULL, gate_buf, x_out, NULL) != 0)
            return -1;

    } else {
        /* Plain FFN, including GPT-2's learned projection biases. */
        layer_name(nm, sizeof(nm), layer, "ffn_up.weight");
        const gguf_tensor_t *wu = tensor_find(g, nm);

        if (!wu) {
            layer_name(nm, sizeof(nm), layer, "fc1.weight");
            wu = tensor_find(g, nm);
        }

        if (!wu) {
            fprintf(stderr,
                    "ffn: missing ffn_up.weight at layer %zu\n", layer);
            return -1;
        }

        const gguf_tensor_t *bu = NULL;
        const gguf_tensor_t *bd = NULL;

        if (cfg->arch == ARCH_GPT2) {
            layer_name(nm, sizeof(nm), layer, "ffn_up.bias");
            bu = tensor_find(g, nm);

            layer_name(nm, sizeof(nm), layer, "ffn_down.bias");
            bd = tensor_find(g, nm);

            if (!bu || !bd) {
                fprintf(stderr,
                        "ffn: missing GPT-2 FFN bias at layer %zu\n",
                        layer);
                return -1;
            }
        }

        if (linear_forward(g, wu, bu, x_in, up_buf, NULL) != 0)
            return -1;

        switch (cfg->act_type) {
        case ACT_GELU:
            apply_gelu(up_buf, n_ff);
            break;
        case ACT_GELU_FAST:
            apply_gelu_fast(up_buf, n_ff);
            break;
        case ACT_SILU:
            apply_silu(up_buf, n_ff);
            break;
        case ACT_RELU:
            for (size_t i = 0; i < n_ff; i++)
                if (up_buf[i] < 0.0f) up_buf[i] = 0.0f;
            break;
        default:
            apply_gelu(up_buf, n_ff);
            break;
        }

        layer_name(nm, sizeof(nm), layer, "fc2.weight");
        const gguf_tensor_t *wd2 = tensor_find(g, nm);
        if (wd2) {
            wd = wd2;
            bd = NULL;
        }

        if (linear_forward(g, wd, bd, up_buf, x_out, NULL) != 0)
            return -1;
    }

    return 0;
}
