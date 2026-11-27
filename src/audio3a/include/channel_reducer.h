#ifndef CHANNEL_REDUCER_H
#define CHANNEL_REDUCER_H

#include "audio_buffer.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_MIC_CHANNELS      6
#define NUM_TOTAL_IN_CHANNELS 8
#define REF_CHANNEL_INDEX     6

/*
 * Biquad Direct Form II Transposed Structure
 */
typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float s1, s2;
} BiquadDF2T;

/*
 * Linkwitz-Riley 4th Order Filter (Cascade of two 2nd order Butterworth)
 */
typedef struct {
    BiquadDF2T stage1;
    BiquadDF2T stage2;
} LR4Filter;

/*
 * ChannelReducer Configuration
 */
typedef struct {
    int sample_rate;             /* Default 16000 */
    float crossover_freq_hz;     /* Default 1200.0f */
    int primary_mic_index;       /* Default 0 (0-5) */
    float ref_scale;             /* Linear scaling for loopback ref (default 1.0f) */
    float hpf_cutoff_hz;         /* High-pass filter cutoff for DC/50Hz/100Hz rejection (default 160.0f) */
    bool apply_hardware_swap;    /* Apply Ch0<->Ch1, Ch2<->Ch3 swap (default true) */
} ChannelReducerConfig;

/*
 * ChannelReducer Engine State
 */
typedef struct {
    ChannelReducerConfig config;
    /* High-pass filters (160Hz 4th-order) for 6 mic channels to kill electrical hum */
    LR4Filter hpf_mics[NUM_MIC_CHANNELS];
    /* High-pass filter for reference channel (phase matched) */
    LR4Filter hpf_ref;
    /* Low-pass filters for 6 microphone channels (<1200Hz crossover) */
    LR4Filter lpf_mics[NUM_MIC_CHANNELS];
    /* High-pass filter for the primary microphone (>1200Hz passthrough) */
    LR4Filter hpf_primary;
    /* Temporary float buffers for real-time processing (Zero-malloc) */
    float mapped_mics[NUM_MIC_CHANNELS][DEFAULT_FRAME_SIZE];
    float low_pass_accum[DEFAULT_FRAME_SIZE];
} ChannelReducer;

/*
 * Public API
 */
void ChannelReducer_GetDefaultConfig(ChannelReducerConfig *config);
int  ChannelReducer_Init(ChannelReducer *reducer, const ChannelReducerConfig *config);
void ChannelReducer_Reset(ChannelReducer *reducer);

/*
 * Process a single frame:
 * @param reducer: Pointer to initialized ChannelReducer
 * @param in_interleaved: 8-channel interleaved int16_t PCM from hardware/WAV
 * @param num_samples: Number of samples per channel (e.g. 256)
 * @param out_fused_mono: Pointer to output buffer for fused mic audio (size num_samples)
 * @param out_ref: Pointer to output buffer for reference signal (size num_samples)
 */
void ChannelReducer_Process(
    ChannelReducer *reducer,
    const int16_t *in_interleaved,
    int num_samples,
    int16_t *out_fused_mono,
    int16_t *out_ref
);

/*
 * Float-level processing for modular DSP chains
 */
void ChannelReducer_ProcessFloat(
    ChannelReducer *reducer,
    const float raw_8ch[NUM_TOTAL_IN_CHANNELS][DEFAULT_FRAME_SIZE],
    int num_samples,
    float out_fused[DEFAULT_FRAME_SIZE],
    float out_ref[DEFAULT_FRAME_SIZE]
);

#ifdef __cplusplus
}
#endif

#endif /* CHANNEL_REDUCER_H */
