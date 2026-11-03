#include <stdio.h>
#include "gguf.h"

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 2;
    }

    gguf_file_t g;
    char err[256] = {0};

    if (gguf_open(&g, argv[1], err, sizeof(err)) != 0) {
        fprintf(stderr, "GGUF error: %s\n", err);
        return 1;
    }

    gguf_print_summary(&g);
    gguf_close(&g);
    return 0;
}
