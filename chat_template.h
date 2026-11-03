#ifndef CHAT_TEMPLATE_H
#define CHAT_TEMPLATE_H

#include "model_config.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Format a chat prompt using the architecture-appropriate template.
 *
 * system_msg : optional system message (may be NULL or empty)
 * user_msg   : the user's prompt
 * buf        : output buffer (caller allocates)
 * bufsz      : size of buf
 *
 * Returns number of bytes written (excluding null terminator),
 * or -1 if buf is too small.
 */
int chat_template_format(const model_config_t *cfg,
                         const char *system_msg,
                         const char *user_msg,
                         char *buf, size_t bufsz);

/*
 * Return the stop-string for the model (e.g. "<|im_end|>").
 * The returned string is static and must not be freed.
 */
const char *chat_template_stop_str(const model_config_t *cfg);

#ifdef __cplusplus
}
#endif
#endif /* CHAT_TEMPLATE_H */
