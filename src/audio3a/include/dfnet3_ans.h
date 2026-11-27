#ifndef DFNET3_ANS_H
#define DFNET3_ANS_H

#include <stddef.h>
#include <stdint.h>
#include "deep_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DFNET3_ANS_IN_RATE       16000
#define DFNET3_ANS_MODEL_RATE    48000
#define DFNET3_ANS_RESAMP_RATIO  3
#define DFNET3_ANS_FIR_TAPS      96
#define DFNET3_ANS_POLY_TAPS     (DFNET3_ANS_FIR_TAPS / DFNET3_ANS_RESAMP_RATIO) /* 32 */
#define DFNET3_ANS_MAX_HOP       960   /* Supports up to 20ms @ 48kHz (default 480 = 10ms) */
#define DFNET3_ANS_FIFO_CAP      4096

typedef struct {
    DFState *df_state;
    int      df_hop_samples; /* 480 for standard 48kHz / 10ms */

    /* 16kHz -> 48kHz polyphase upsampler state */
    float    up_hist[DFNET3_ANS_POLY_TAPS];

    /* 48kHz -> 16kHz decimation FIR state */
    float    dn_hist[DFNET3_ANS_FIR_TAPS];
    int      dn_pos;

    /* 48kHz circular FIFOs bridging arbitrary 16kHz block size <-> 480-sample DF hop */
    float    in_fifo[DFNET3_ANS_FIFO_CAP];
    int      in_r;
    int      in_w;
    int      in_count;

    float    out_fifo[DFNET3_ANS_FIFO_CAP];
    int      out_r;
    int      out_w;
    int      out_count;

    float    last_lsnr_db;
    int      initialized;
} DFNet3ANS;

/*
 * Initialize streaming DeepFilterNet 3 denoiser with 16kHz <-> 48kHz bridge.
 *
 * @param eng               Pre-allocated DFNet3ANS state struct
 * @param model_tar_gz      Path to DeepFilterNet3_onnx.tar.gz (or _ll_onnx.tar.gz)
 * @param atten_lim_db      Max noise attenuation limit in dB (e.g. 60.0f or 100.0f for unlimited)
 * @param post_filter_beta  Post-filter strength (0.0f disables, 0.02f matches DFNet3 default)
 * @return 0 on success, -1 on failure
 */
int DFNet3ANS_Init(DFNet3ANS *eng,
                   const char *model_tar_gz,
                   float atten_lim_db,
                   float post_filter_beta);

/*
 * Process one 16kHz int16 frame (e.g. 256 samples = 16ms) in-place or out-of-place.
 * Pre-fills 1 hop (10ms) at Init so output count always matches num_samples without underrun.
 */
void DFNet3ANS_ProcessInt16(DFNet3ANS *eng,
                            const int16_t *in_16k,
                            int16_t *out_16k,
                            int num_samples);

void DFNet3ANS_ProcessFloat(DFNet3ANS *eng,
                            const float *in_16k,
                            float *out_16k,
                            int num_samples);

void DFNet3ANS_SetAttenLim(DFNet3ANS *eng, float atten_lim_db);
void DFNet3ANS_Destroy(DFNet3ANS *eng);

#ifdef __cplusplus
}
#endif

#endif /* DFNET3_ANS_H */
