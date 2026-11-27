#ifndef LINEAR_AEC_H
#define LINEAR_AEC_H

#include "fast_fft.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AEC_FRAME_SIZE       256   /* 16ms @ 16kHz */
#define AEC_FFT_SIZE         512   /* 2 * AEC_FRAME_SIZE */
#define AEC_NUM_BINS         (AEC_FFT_SIZE / 2 + 1) /* 257 bins */
#define AEC_MAX_PARTITIONS   16    /* Up to 16 * 16ms = 256ms tail */
#define AEC_NLMS_TAPS        640   /* 40ms sample-by-sample Two-Path VSS-NLMS */
#define AEC_MAX_PRE_DELAY    512   /* Up to 32ms pre-delay */

typedef struct {
    int   sample_rate;       /* Default 16000 */
    int   num_partitions;    /* Default 8 (128ms echo tail) */
    float step_size;         /* Adaptation step size mu (default 0.25f) */
    float leakage;           /* Weight leakage factor (default 0.99995f) */
    float min_erle_freeze;   /* DTD sensitivity threshold */
    int   pre_delay_samples; /* Pre-delay in samples (default 130 for RK3588 loopback) */
} LinearAECConfig;

typedef struct {
    LinearAECConfig config;
    FastFFT         fft;

    /* Pre-delay ring buffer for reference signal */
    float           ref_delay_ring[AEC_MAX_PRE_DELAY];
    int             ref_delay_pos;

    /* Sample-by-sample Two-Path VSS-NLMS state */
    float           w_bg[AEC_NLMS_TAPS];
    float           w_fg[AEC_NLMS_TAPS];
    float           x_ring[AEC_NLMS_TAPS * 2];
    int             ring_pos;
    double          norm_x;
    float           err_pow_smooth;
    bool            fg_valid;

    /* Reference history in frequency domain [partition][bin] */
    Complex32       X_history[AEC_MAX_PARTITIONS][AEC_NUM_BINS];
    /* Adaptive filter weights in frequency domain [partition][bin] */
    Complex32       W[AEC_MAX_PARTITIONS][AEC_NUM_BINS];

    /* Reference signal overlap buffer (time domain, size 256) */
    float           x_old[AEC_FRAME_SIZE];
    /* Reference power spectrum per bin */
    float           power_X[AEC_NUM_BINS];

    /* Double-talk detector state */
    bool            is_double_talk;
    float           erle_db;
    float           smoothed_erle_db;
    float           near_energy;
    float           echo_energy;
    float           error_energy;
    int             peak_tap;
    float           ref_reverb_energy;

    /* Scratch buffers for zero allocation */
    float           scratch_time[AEC_FFT_SIZE];
    Complex32       scratch_freq[AEC_NUM_BINS];
} LinearAEC;

/*
 * Initialize LinearAEC instance with configuration
 */
void LinearAEC_GetDefaultConfig(LinearAECConfig *config);
int  LinearAEC_Init(LinearAEC *aec, const LinearAECConfig *config);
void LinearAEC_Reset(LinearAEC *aec);

/*
 * Process one frame of audio:
 * @param aec: Pointer to initialized LinearAEC
 * @param mic_in: Near-end mic signal (from ChannelReducer fused mono), size 256 float [-1.0f, 1.0f]
 * @param ref_in: Far-end speaker reference signal (from Channel 6), size 256 float [-1.0f, 1.0f]
 * @param out_error: Linear echo cancelled output e[n] = y[n] - d_hat[n], size 256
 * @param out_echo_est: Estimated linear echo d_hat[n] (for downstream Neural RES), size 256
 */
void LinearAEC_ProcessFloat(
    LinearAEC *aec,
    const float *mic_in,
    const float *ref_in,
    float *out_error,
    float *out_echo_est
);

/*
 * Process int16_t PCM frame directly
 */
void LinearAEC_ProcessInt16(
    LinearAEC *aec,
    const int16_t *mic_in,
    const int16_t *ref_in,
    int16_t *out_error,
    int16_t *out_echo_est
);

/* Get current instantaneous and smoothed ERLE in dB */
float LinearAEC_GetERLE(const LinearAEC *aec);
bool  LinearAEC_IsDoubleTalk(const LinearAEC *aec);

#ifdef __cplusplus
}
#endif

#endif /* LINEAR_AEC_H */
