
CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wpedantic -march=native
LDFLAGS ?=
LDLIBS  ?= -lm

# ── Source lists ────────────────────────────────────────────────────────────
INFERENCE_SRCS = \
    main.c         \
    transformer.c  \
    attention.c    \
    ffn.c          \
    tokenizer.c    \
    chat_template.c \
    sampler.c      \
    rope.c         \
    norm.c         \
    linear.c       \
    gguf.c         \
    gguf_raw.c     \
    model_config.c \
    quant.c        \
    quant_matvec.c \
    tensor_data.c  \
    tensor_ops.c

INSPECT_SRCS = \
    gguf-inspect-main.c \
    gguf.c

QUANT_TEST_SRCS = \
    quant_test.c \
    gguf.c       \
    gguf_raw.c   \
    quant.c      \
    quant_matvec.c \
    tensor_data.c

# ── Object lists ─────────────────────────────────────────────────────────────
INFERENCE_OBJS  = $(INFERENCE_SRCS:.c=.o)
QUANT_TEST_OBJS = $(QUANT_TEST_SRCS:.c=.o)

# ── Targets ──────────────────────────────────────────────────────────────────
.PHONY: all clean run inspect

all: forword quant-test gguf-inspect

forword: $(INFERENCE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

quant-test: $(QUANT_TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# ── Pattern rule ──────────────────────────────────────────────────────────────
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── Dependency declarations ────────────────────────────────────────────────────
main.o:         main.c transformer.h sampler.h chat_template.h
transformer.o:  transformer.c transformer.h attention.h ffn.h norm.h rope.h linear.h tensor_data.h
attention.o:    attention.c attention.h linear.h rope.h norm.h tensor_data.h
ffn.o:          ffn.c ffn.h linear.h tensor_data.h norm.h
tokenizer.o:    tokenizer.c tokenizer.h gguf_raw.h model_config.h
chat_template.o: chat_template.c chat_template.h model_config.h
sampler.o:      sampler.c sampler.h
rope.o:         rope.c rope.h
norm.o:         norm.c norm.h
linear.o:       linear.c linear.h quant.h quant_matvec.h tensor_data.h
gguf.o:         gguf.c gguf.h
gguf_raw.o:     gguf_raw.c gguf_raw.h gguf.h
model_config.o: model_config.c model_config.h gguf_raw.h gguf.h
quant.o:        quant.c quant.h
quant_matvec.o: quant_matvec.c quant_matvec.h quant.h
tensor_data.o:  tensor_data.c tensor_data.h gguf.h
tensor_ops.o:   tensor_ops.c tensor_ops.h
quant_test.o:   quant_test.c gguf.h quant.h tensor_data.h quant_matvec.h

# ── Convenience targets ────────────────────────────────────────────────────────
MODEL ?= /mnt/c/Users/josep/Documents/software/humlang/src/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
PROMPT ?= Write a hello world in Python.

run: forword
	./forword "$(MODEL)" "$(PROMPT)"

test: quant-test
	./quant-test "$(MODEL)"

clean:
	rm -f *.o forword quant-test gguf-inspect

