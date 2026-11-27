#ifndef DPDFNET_ANS_H
#define DPDFNET_ANS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DPDFNET_SAMPLE_RATE 16000
#define DPDFNET_WIN_LEN     320   /* 20 ms STFT window */
#define DPDFNET_HOP_LEN     160   /* 10 ms STFT hop */
#define DPDFNET_FREQ_BINS   161   /* WIN_LEN / 2 + 1 */
#define DPDFNET_FIFO_CAP    1024  /* Power-of-2 ring FIFO (> 160 + 256) */

typedef struct DPDFNetOrtCtx DPDFNetOrtCtx;

typedef struct {
    DPDFNetOrtCtx *ctx;
    int initialized;

    float vorbis_win[DPDFNET_WIN_LEN];
    float ana_hist[DPDFNET_WIN_LEN];
    float ola_buf[DPDFNET_HOP_LEN];

    float in_fifo[DPDFNET_FIFO_CAP];
    int   in_r;
    int   in_w;
    int   in_count;

    float out_fifo[DPDFNET_FIFO_CAP];
    int   out_r;
    int   out_w;
    int   out_count;

    int num_threads;
} DPDFNetANS;

/*
 * Initialize DPDFNet ONNX streaming dereverb/denoiser @ 16kHz.
 * Reads state_size, erb_norm_init, and spec_norm_init from ONNX custom metadata,
 * warms up the session, and pre-fills 1 hop (10ms) in out_fifo for 256-sample P2P cadence.
 * Returns 0 on success, negative on error.
 */
int DPDFNetANS_Init(DPDFNetANS *eng, const char *model_onnx, int num_threads);

/* Reset STFT history, ONNX state buffer, and 10ms FIFO prefill without reloading model. */
void DPDFNetANS_Reset(DPDFNetANS *eng);

/* Process one 16kHz int16 frame (e.g. 256 samples = 16ms) in-place or out-of-place. */
void DPDFNetANS_ProcessInt16(DPDFNetANS *eng,
                             const int16_t *in_16k,
                             int16_t *out_16k,
                             int num_samples);

/* Process one 16kHz float frame ([-1.0, 1.0]) in-place or out-of-place. */
void DPDFNetANS_ProcessFloat(DPDFNetANS *eng,
                             const float *in_16k,
                             float *out_16k,
                             int num_samples);

void DPDFNetANS_Destroy(DPDFNetANS *eng);

#ifdef __cplusplus
}
#endif

#endif /* DPDFNET_ANS_H */
