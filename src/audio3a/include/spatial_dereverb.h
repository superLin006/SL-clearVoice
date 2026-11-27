#ifndef SPATIAL_DEREVERB_H
#define SPATIAL_DEREVERB_H

#include <stdint.h>
#include <stdbool.h>
#include "fast_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEREVERB_MAX_MICS    8
#define DEREVERB_FRAME_SAMPLES 256
#define DEREVERB_FFT_SIZE    512
#define DEREVERB_NUM_BINS    (DEREVERB_FFT_SIZE / 2 + 1) /* 257 bins */

typedef struct {
    int   sample_rate;         /* 16000 Hz */
    int   num_mics;            /* 2 to 8 mics (default: 8) */
    float min_gain;            /* Minimum dereverb suppression gain (default: 0.25f = -12 dB) */
    float smooth_alpha_rise;   /* Fast tracking for speech onsets (default: 0.50f) */
    float smooth_alpha_decay;  /* Smoothing for diffuse tail suppression (default: 0.82f) */
    bool  enabled;             /* Enable/disable bypass flag */
} SpatialDereverbConfig;

typedef struct {
    SpatialDereverbConfig config;
    FastFFT               fft;

    /* Analysis & synthesis window buffers */
    float analysis_window[DEREVERB_FFT_SIZE];
    float synthesis_window[DEREVERB_FFT_SIZE];

    /* Overlap history for each microphone (512 samples) */
    float mic_history[DEREVERB_MAX_MICS][DEREVERB_FFT_SIZE];

    /* Overlap-Add reconstruction buffer */
    float ola_buffer[DEREVERB_FFT_SIZE];

    /* Smoothed power spectra across frames */
    float prev_cross_power[DEREVERB_NUM_BINS];
    float prev_auto_power[DEREVERB_NUM_BINS];

    /* Stats & metrics */
    float last_avg_coherence;
    uint32_t frame_count;
} SpatialDereverb;

/*
 * Initialize the Spatial Dereverberation engine.
 * Passing NULL for cfg uses default optimal settings (8 mics, 16kHz, min_gain=0.25).
 */
int SpatialDereverb_Init(SpatialDereverb *dr, const SpatialDereverbConfig *cfg);

/*
 * Reset internal history and smoothing state.
 */
void SpatialDereverb_Reset(SpatialDereverb *dr);

/*
 * Process one 16ms frame (256 samples @ 16kHz).
 * mic_inputs: Array of pointers to num_mics channels, each 256 samples.
 * out_clean: Output buffer for dereverberated mono signal (256 samples).
 * Returns: Average spatial coherence of speech in [0.0, 1.0].
 */
float SpatialDereverb_ProcessInt16(SpatialDereverb *dr,
                                  const int16_t * const mic_inputs[],
                                  int16_t out_clean[DEREVERB_FRAME_SAMPLES]);

#ifdef __cplusplus
}
#endif

#endif /* SPATIAL_DEREVERB_H */
