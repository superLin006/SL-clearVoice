#ifndef ONLINE_WPE_H
#define ONLINE_WPE_H

#include <stdint.h>
#include <stdbool.h>
#include "fast_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ONLINE_WPE_MAX_MICS      8
#define ONLINE_WPE_MAX_TAPS      8
#define ONLINE_WPE_MAX_DELAY     6
#define ONLINE_WPE_HIST_DEPTH    16 /* Power-of-2 > MAX_DELAY + MAX_TAPS */
#define ONLINE_WPE_FRAME_SAMPLES 256
#define ONLINE_WPE_FFT_SIZE      512
#define ONLINE_WPE_NUM_BINS      (ONLINE_WPE_FFT_SIZE / 2 + 1) /* 257 bins */

typedef struct {
    int   num_mics;    /* 1 to 8 mics (default: 8) */
    int   taps;        /* Prediction filter taps T per mic (1..8, default: 4) */
    int   delay;       /* Prediction delay D in frames (1..6, default: 2 -> 32 ms) */
    float alpha;       /* RLS forgetting factor (default: 0.99f) */
    float power_floor; /* Quiet-bin update gate & PSD floor (default: 1e-4f) */
    float diag_init;   /* Initial R^-1 diagonal (default: 1.0f) */
    float diag_max;    /* Max allowed R^-1 diagonal to prevent pause windup (default: 4.0f) */
    bool  enabled;     /* Enable/disable flag */
} OnlineWpeConfig;

typedef struct {
    double r;
    double i;
} OnlineWpeCpx64;

typedef struct {
    OnlineWpeConfig config;
    FastFFT         fft;

    float window[ONLINE_WPE_FFT_SIZE]; /* sqrt-Hann analysis/synthesis window */

    float mic_history[ONLINE_WPE_MAX_MICS][ONLINE_WPE_FRAME_SAMPLES];
    float ola_buffer[ONLINE_WPE_MAX_MICS][ONLINE_WPE_FRAME_SAMPLES];

    /* Circular STFT spectrum history: [HIST_DEPTH][NUM_BINS][M] */
    Complex32 *spec_hist;
    /* Double-precision SPD inverse covariance: [NUM_BINS][P][P], P = num_mics * taps */
    OnlineWpeCpx64 *rinv;
    /* Prediction filters: [NUM_BINS][P][num_mics] */
    OnlineWpeCpx64 *filt;

    int      P;          /* num_mics * taps */
    int      hist_write; /* Current frame slot in spec_hist */
    uint32_t frame_count;
    int      initialized;
} OnlineWpe;

/* Populate config struct with real-time 8-mic defaults (T=4, D=2, alpha=0.99). */
void OnlineWpe_DefaultConfig(OnlineWpeConfig *cfg, int num_mics);

/* Initialize engine and allocate internal RLS states; returns 0 on success. */
int  OnlineWpe_Init(OnlineWpe *wpe, const OnlineWpeConfig *cfg);

/* Reset STFT histories, RLS covariance, and filter weights without reallocation. */
void OnlineWpe_Reset(OnlineWpe *wpe);

/* Free internal allocations cleanly. */
void OnlineWpe_Destroy(OnlineWpe *wpe);

/*
 * Process one 16ms multi-channel frame (256 samples @ 16kHz).
 * in_mics and out_mics may alias for in-place processing.
 */
void OnlineWpe_ProcessInt16(OnlineWpe *wpe,
                            const int16_t * const in_mics[],
                            int16_t * const out_mics[]);

#ifdef __cplusplus
}
#endif

#endif /* ONLINE_WPE_H */
