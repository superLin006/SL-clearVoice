#ifndef ADAPTIVE_AGC_H
#define ADAPTIVE_AGC_H

#include "audio_buffer.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float s1, s2;
} BiquadFilter;

typedef struct {
    float target_level_rms;     /* Target RMS level (e.g. 0.20f = -14dBFS, default 0.20f) */
    float max_gain_db;          /* Maximum boost in dB (default +15.0dB) */
    float min_gain_db;          /* Minimum gain in dB (default -12.0dB) */
    float noise_gate_threshold; /* RMS threshold below which gain is frozen (default 0.003f ~ -50dBFS) */
    float attack_time_ms;       /* Gain reduction speed (default 10ms) */
    float release_time_ms;      /* Gain recovery speed (default 300ms) */
    float silence_standby_gain; /* Target linear gain during silence (default 1.0f; pre-ANS can use 2.5f) */
    bool  enable_eq;            /* Enable 80Hz rumble cut & vocal clarity EQ (default true) */
    bool  enable_limiter;       /* Enable smooth soft-knee peak limiter (default true) */

    /* Downward expansion for frames below the tracked speech envelope.
     * 0 = disabled. Cuts reverb tails / background noise back down so the AGC's
     * speech lift does not also lift the inter-word floor. */
    float expand_range_db;      /* How far below speech level expansion applies (e.g. 18 dB) */
    float expand_ratio;         /* dB of cut per dB below speech (e.g. 1.0 = 1:1) */
    float expand_max_cut_db;    /* Hard cap on the extra cut (e.g. 18 dB) */
} AGCConfig;

typedef struct {
    AGCConfig     config;
    float         current_gain;
    float         smoothed_rms;
    float         expand_env;      /* slow speech envelope used by downward expansion */
    float         expand_gain;     /* smoothed attenuation applied by the expander (1.0 = none) */
    float         noise_floor_rms;
    float         alpha_attack;
    float         alpha_release;
    float         max_gain_linear;
    float         min_gain_linear;

    /* Equalizer filters */
    BiquadFilter  hpf_80hz_stage1;
    BiquadFilter  hpf_80hz_stage2;     /* 4th-order 24dB/oct steep cut below 85Hz */
    BiquadFilter  presence_eq;         /* Subtle presence enhancement */
    int           vad_speech_active;   /* 1 = speech, 0 = silence/pause */
} AdaptiveAGC;

/*
 * Initialize AGC and DRC engine
 */
void AdaptiveAGC_GetDefaultConfig(AGCConfig *config);
int  AdaptiveAGC_Init(AdaptiveAGC *agc, const AGCConfig *config, int sample_rate);
void AdaptiveAGC_Reset(AdaptiveAGC *agc);
void AdaptiveAGC_SetVAD(AdaptiveAGC *agc, int is_speech);

/*
 * Process a single frame (Float interface [-1.0f, 1.0f])
 * @param in_audio: Input audio frame (size num_samples)
 * @param out_audio: Output leveled and limited audio frame (size num_samples)
 */
void AdaptiveAGC_ProcessFloat(
    AdaptiveAGC *agc,
    const float *in_audio,
    float *out_audio,
    int num_samples
);

/*
 * Process int16_t PCM frame
 */
void AdaptiveAGC_ProcessInt16(
    AdaptiveAGC *agc,
    const int16_t *in_audio,
    int16_t *out_audio,
    int num_samples
);

/* Get current linear gain factor applied */
float AdaptiveAGC_GetCurrentGain(const AdaptiveAGC *agc);

#ifdef __cplusplus
}
#endif

#endif /* ADAPTIVE_AGC_H */
