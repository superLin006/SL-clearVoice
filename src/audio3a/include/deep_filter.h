// SPDX-License-Identifier: MIT OR Apache-2.0
#ifndef DEEP_FILTER_H
#define DEEP_FILTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DFState DFState;

DFState *df_create(const char *path, float atten_lim, const char *log_level);
size_t df_get_frame_length(DFState *st);
char *df_next_log_msg(DFState *st);
void df_free_log_msg(char *ptr);
void df_set_atten_lim(DFState *st, float lim_db);
void df_set_post_filter_beta(DFState *st, float beta);
float df_process_frame(DFState *st, float *input, float *output);
float df_process_frame_raw(DFState *st, float *input, float **out_gains_p, float **out_coefs_p);
void df_free(DFState *model);

#ifdef __cplusplus
}
#endif

#endif /* DEEP_FILTER_H */
