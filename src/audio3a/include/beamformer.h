#ifndef BEAMFORMER_H
#define BEAMFORMER_H

#include <stdint.h>
#include <stdbool.h>
#include "fast_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BEAM_NUM_MICS        6
#define BEAM_TOTAL_IN_CH     8
#define BEAM_FRAME_SIZE      256   /* 16 ms at 16kHz */
#define BEAM_FFT_SIZE        512   /* 32 ms window */
#define BEAM_NUM_BINS        (BEAM_FFT_SIZE / 2 + 1) /* 257 bins */
#define BEAM_NUM_DOA_ANGLES  72    /* 360 / 5 degrees */

typedef enum {
    BEAM_MODE_AUTO_TRACK = 0, /* Adaptive 360-degree DOA tracking beam */
    BEAM_MODE_FIXED_ANGLE = 1  /* Fixed steer angle (e.g. 0 deg, 135 deg) */
} BeamformerMode;

typedef struct {
    float           sample_rate;          /* Default: 16000.0f */
    float           array_radius_m;       /* Default: 0.028f (56mm diameter) */
    BeamformerMode  mode;                 /* Default: BEAM_MODE_AUTO_TRACK */
    float           fixed_angle_deg;      /* Used when mode == BEAM_MODE_FIXED_ANGLE */
    float           doa_smooth_alpha;     /* DOA tracking smoothing factor (0.1 ~ 0.2) */
    float           postfilter_min_gain;  /* Min gain for spatial dereverberation (e.g. 0.2f) */
    float           ref_scale;            /* Gain scale for AEC loopback channel */
    bool            enable_postfilter;    /* Enable spatial coherence dereverberation */
} BeamformerConfig;

typedef struct {
    BeamformerConfig config;

    /* FFT engine */
    FastFFT         fft;

    /* Window & Trigonometric tables */
    float           analysis_window[BEAM_FFT_SIZE];
    float           synthesis_window[BEAM_FFT_SIZE];
    float           mic_angles_rad[BEAM_NUM_MICS];
    float           mic_x[BEAM_NUM_MICS];
    float           mic_y[BEAM_NUM_MICS];

    /* Time domain history buffers for 6 mics + 1 ref (overlap-save / overlap-add) */
    float           in_history[BEAM_NUM_MICS][BEAM_FFT_SIZE];
    float           ref_history[BEAM_FFT_SIZE];
    float           ola_buffer[BEAM_FFT_SIZE];

    /* Pre-calculated steering phase delays for 72 DOA candidate angles */
    float           candidate_delays[BEAM_NUM_DOA_ANGLES][BEAM_NUM_MICS];

    /* Current tracking state */
    float           current_doa_deg;       /* Estimated continuous speaker angle (0~360) */
    float           smoothed_doa_rad;      /* Radian smoothed angle */
    float           doa_confidence;        /* Confidence metric of current speaker (0~1) */
    int             active_speech_count;   /* Speech activity frame counter */

    /* Temporal smoothing for spatial coherence dereverberation */
    float           prev_cross_power[BEAM_NUM_BINS];
    float           prev_total_power[BEAM_NUM_BINS];

    /* Sub-band HPF for DC & hum removal (4th-order 85Hz on output) */
    float           hpf_state[4];
} Beamformer;

/*
 * Initialize the Beamformer module
 * @param bf: Beamformer instance pointer
 * @param cfg: Configuration settings (NULL for default)
 * @return 0 on success, non-zero on error
 */
int Beamformer_Init(Beamformer *bf, const BeamformerConfig *cfg);

/*
 * Reset internal buffers and states
 */
void Beamformer_Reset(Beamformer *bf);

/*
 * Process a single 16ms frame (256 samples, 8 channels interleaved)
 * @param bf: Beamformer instance
 * @param in_interleaved_8ch: 8-channel interleaved 16-bit PCM (256 * 8 int16_t)
 * @param out_fused_mono: Pointer to 256 int16_t samples of beamformed clean speech
 * @param out_ref_mono: Pointer to 256 int16_t samples of aligned AEC reference
 * @return 0 on success
 */
int Beamformer_Process(
    Beamformer *bf,
    const int16_t *in_interleaved_8ch,
    int16_t *out_fused_mono,
    int16_t *out_ref_mono
);

/*
 * Get the current real-time estimated speaker direction of arrival (DOA)
 * @param bf: Beamformer instance
 * @return Azimuth angle in degrees (0.0f ~ 360.0f)
 */
float Beamformer_GetDOAAngle(const Beamformer *bf);

#ifdef __cplusplus
}
#endif

#endif /* BEAMFORMER_H */
