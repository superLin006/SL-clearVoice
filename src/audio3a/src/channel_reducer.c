#include "channel_reducer.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float biquad_df2t_process(BiquadDF2T *b, float x) {
    float y = b->b0 * x + b->s1;
    b->s1 = b->b1 * x - b->a1 * y + b->s2;
    b->s2 = b->b2 * x - b->a2 * y;
    return y;
}

static inline float lr4_process(LR4Filter *filter, float x) {
    float mid = biquad_df2t_process(&filter->stage1, x);
    return biquad_df2t_process(&filter->stage2, mid);
}

static void biquad_df2t_reset(BiquadDF2T *b) {
    b->s1 = 0.0f;
    b->s2 = 0.0f;
}

static void lr4_reset(LR4Filter *filter) {
    biquad_df2t_reset(&filter->stage1);
    biquad_df2t_reset(&filter->stage2);
}

static void calculate_butterworth_2nd(float fc, float fs, bool is_highpass, BiquadDF2T *stage) {
    float omega = (float)(2.0 * M_PI * fc / fs);
    float k = tanf(omega * 0.5f);
    float k2 = k * k;
    float sqrt2 = 1.41421356237309504880f;
    float denom = 1.0f + sqrt2 * k + k2;

    if (!is_highpass) {
        /* 2nd-order Butterworth Lowpass */
        stage->b0 = k2 / denom;
        stage->b1 = 2.0f * k2 / denom;
        stage->b2 = k2 / denom;
    } else {
        /* 2nd-order Butterworth Highpass */
        stage->b0 = 1.0f / denom;
        stage->b1 = -2.0f / denom;
        stage->b2 = 1.0f / denom;
    }

    stage->a1 = 2.0f * (k2 - 1.0f) / denom;
    stage->a2 = (1.0f - sqrt2 * k + k2) / denom;
    stage->s1 = 0.0f;
    stage->s2 = 0.0f;
}

static void calculate_lr4(float fc, float fs, bool is_highpass, LR4Filter *filter) {
    /* LR4 is two identical 2nd-order Butterworth filters in series */
    calculate_butterworth_2nd(fc, fs, is_highpass, &filter->stage1);
    calculate_butterworth_2nd(fc, fs, is_highpass, &filter->stage2);
}

void ChannelReducer_GetDefaultConfig(ChannelReducerConfig *config) {
    if (!config) return;
    config->sample_rate = DEFAULT_SAMPLE_RATE;
    config->crossover_freq_hz = 700.0f;     /* 700Hz: strictly in-phase coherent zone for 70mm circular array */
    config->hpf_cutoff_hz = 85.0f;          /* 85Hz 4th-order HPF: cuts sub-80Hz rumble while keeping 100-150Hz voice formants */
    config->primary_mic_index = 0;
    config->ref_scale = 1.0f;
    config->apply_hardware_swap = true;
}

int ChannelReducer_Init(ChannelReducer *reducer, const ChannelReducerConfig *config) {
    if (!reducer) return -1;
    if (config) {
        reducer->config = *config;
    } else {
        ChannelReducer_GetDefaultConfig(&reducer->config);
    }

    float fs = (float)reducer->config.sample_rate;
    float fc = reducer->config.crossover_freq_hz;
    float fc_hpf = reducer->config.hpf_cutoff_hz;

    /* Guardrails */
    if (fc < 200.0f) fc = 200.0f;
    if (fc > (fs * 0.45f)) fc = fs * 0.45f;
    if (fc_hpf < 40.0f) fc_hpf = 40.0f;
    if (fc_hpf > 300.0f) fc_hpf = 300.0f;

    /* Initialize 160Hz HPF for all 6 mics and reference */
    for (int m = 0; m < NUM_MIC_CHANNELS; ++m) {
        calculate_lr4(fc_hpf, fs, true, &reducer->hpf_mics[m]);
    }
    calculate_lr4(fc_hpf, fs, true, &reducer->hpf_ref);

    /* Initialize 6 LPFs for the 6 mics (crossover lowpass) */
    for (int m = 0; m < NUM_MIC_CHANNELS; ++m) {
        calculate_lr4(fc, fs, false, &reducer->lpf_mics[m]);
    }

    /* Initialize 1 HPF for the primary mic (crossover highpass) */
    calculate_lr4(fc, fs, true, &reducer->hpf_primary);

    memset(reducer->mapped_mics, 0, sizeof(reducer->mapped_mics));
    memset(reducer->low_pass_accum, 0, sizeof(reducer->low_pass_accum));
    return 0;
}

void ChannelReducer_Reset(ChannelReducer *reducer) {
    if (!reducer) return;
    for (int m = 0; m < NUM_MIC_CHANNELS; ++m) {
        lr4_reset(&reducer->hpf_mics[m]);
        lr4_reset(&reducer->lpf_mics[m]);
    }
    lr4_reset(&reducer->hpf_ref);
    lr4_reset(&reducer->hpf_primary);
    memset(reducer->mapped_mics, 0, sizeof(reducer->mapped_mics));
    memset(reducer->low_pass_accum, 0, sizeof(reducer->low_pass_accum));
}

static inline int16_t sat_int16(float val) {
    float scaled = val * 32767.0f;
    if (scaled > 32767.0f) return 32767;
    if (scaled < -32768.0f) return -32768;
    return (int16_t)roundf(scaled);
}

void ChannelReducer_ProcessFloat(
    ChannelReducer *reducer,
    const float raw_8ch[NUM_TOTAL_IN_CHANNELS][DEFAULT_FRAME_SIZE],
    int num_samples,
    float out_fused[DEFAULT_FRAME_SIZE],
    float out_ref[DEFAULT_FRAME_SIZE]
) {
    if (!reducer || !out_fused || !out_ref) return;
    int samples = (num_samples > DEFAULT_FRAME_SIZE) ? DEFAULT_FRAME_SIZE : num_samples;
    const float inv_mics = 1.0f / (float)NUM_MIC_CHANNELS;

    /* 1. Hardware Pin Remap */
    if (reducer->config.apply_hardware_swap) {
        /* Ch0 <-> Ch1, Ch2 <-> Ch3, Ch4, Ch5 */
        for (int i = 0; i < samples; ++i) {
            reducer->mapped_mics[0][i] = raw_8ch[1][i];
            reducer->mapped_mics[1][i] = raw_8ch[0][i];
            reducer->mapped_mics[2][i] = raw_8ch[3][i];
            reducer->mapped_mics[3][i] = raw_8ch[2][i];
            reducer->mapped_mics[4][i] = raw_8ch[4][i];
            reducer->mapped_mics[5][i] = raw_8ch[5][i];
        }
    } else {
        for (int m = 0; m < NUM_MIC_CHANNELS; ++m) {
            for (int i = 0; i < samples; ++i) {
                reducer->mapped_mics[m][i] = raw_8ch[m][i];
            }
        }
    }

    /* 1.5. 160Hz 4th-Order High-Pass Filter: eradicates DC offset & 50Hz/100Hz electrical hum */
    for (int m = 0; m < NUM_MIC_CHANNELS; ++m) {
        LR4Filter *hpf = &reducer->hpf_mics[m];
        float *in_mic = reducer->mapped_mics[m];
        for (int i = 0; i < samples; ++i) {
            in_mic[i] = lr4_process(hpf, in_mic[i]);
        }
    }

    /* 2. Speaker Reference Extraction (Phase-matched 160Hz HPF) */
    float ref_gain = reducer->config.ref_scale;
    LR4Filter *hpf_ref = &reducer->hpf_ref;
    for (int i = 0; i < samples; ++i) {
        float ref_val = raw_8ch[REF_CHANNEL_INDEX][i] * ref_gain;
        out_ref[i] = lr4_process(hpf_ref, ref_val);
    }

    /* 3. Low-Frequency Fusion across 6 mics (<1.2 kHz) */
    memset(reducer->low_pass_accum, 0, samples * sizeof(float));
    for (int m = 0; m < NUM_MIC_CHANNELS; ++m) {
        LR4Filter *lpf = &reducer->lpf_mics[m];
        const float *in_mic = reducer->mapped_mics[m];
        for (int i = 0; i < samples; ++i) {
            float low_val = lr4_process(lpf, in_mic[i]);
            reducer->low_pass_accum[i] += low_val;
        }
    }

    /* 4. High-Frequency Passthrough on Primary Mic (>1.2 kHz) */
    int prim_idx = reducer->config.primary_mic_index;
    if (prim_idx < 0 || prim_idx >= NUM_MIC_CHANNELS) prim_idx = 0;
    const float *primary_mic = reducer->mapped_mics[prim_idx];
    LR4Filter *hpf = &reducer->hpf_primary;

    /* 5. Crossover Recombination: y = (Sum(LPF_m) / 6) + HPF_primary */
    for (int i = 0; i < samples; ++i) {
        float y_low = reducer->low_pass_accum[i] * inv_mics;
        float y_high = lr4_process(hpf, primary_mic[i]);
        out_fused[i] = y_low + y_high;
    }
}

void ChannelReducer_Process(
    ChannelReducer *reducer,
    const int16_t *in_interleaved,
    int num_samples,
    int16_t *out_fused_mono,
    int16_t *out_ref
) {
    if (!reducer || !in_interleaved || !out_fused_mono || !out_ref) return;
    int samples = (num_samples > DEFAULT_FRAME_SIZE) ? DEFAULT_FRAME_SIZE : num_samples;
    const float scale = 1.0f / 32768.0f;

    /* De-interleave and normalize into static buffers */
    static float deint_buf[NUM_TOTAL_IN_CHANNELS][DEFAULT_FRAME_SIZE];
    for (int i = 0; i < samples; ++i) {
        int base = i * NUM_TOTAL_IN_CHANNELS;
        for (int ch = 0; ch < NUM_TOTAL_IN_CHANNELS; ++ch) {
            deint_buf[ch][i] = ((float)in_interleaved[base + ch]) * scale;
        }
    }

    static float float_fused[DEFAULT_FRAME_SIZE];
    static float float_ref[DEFAULT_FRAME_SIZE];

    ChannelReducer_ProcessFloat(reducer, deint_buf, samples, float_fused, float_ref);

    /* Convert back to int16_t with saturation */
    for (int i = 0; i < samples; ++i) {
        out_fused_mono[i] = sat_int16(float_fused[i]);
        out_ref[i] = sat_int16(float_ref[i]);
    }
}
