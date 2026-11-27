#include "adaptive_agc.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float biquad_process(BiquadFilter *b, float x) {
    float y = b->b0 * x + b->s1;
    b->s1 = b->b1 * x - b->a1 * y + b->s2;
    b->s2 = b->b2 * x - b->a2 * y;
    return y;
}

static void biquad_reset(BiquadFilter *b) {
    b->s1 = 0.0f;
    b->s2 = 0.0f;
}

static void design_hpf(BiquadFilter *b, float fc, float fs) {
    float omega = 2.0f * (float)M_PI * fc / fs;
    float k = tanf(omega * 0.5f);
    float k2 = k * k;
    float sqrt2 = 1.41421356f;
    float denom = 1.0f + sqrt2 * k + k2;

    b->b0 = 1.0f / denom;
    b->b1 = -2.0f / denom;
    b->b2 = 1.0f / denom;
    b->a1 = 2.0f * (k2 - 1.0f) / denom;
    b->a2 = (1.0f - sqrt2 * k + k2) / denom;
    b->s1 = 0.0f;
    b->s2 = 0.0f;
}

static void design_peaking_eq(BiquadFilter *b, float f0, float q, float gain_db, float fs) {
    float A = powf(10.0f, gain_db / 40.0f);
    float omega = 2.0f * (float)M_PI * f0 / fs;
    float alpha = sinf(omega) / (2.0f * q);
    float cos_w = cosf(omega);

    float a0 = 1.0f + alpha / A;
    b->b0 = (1.0f + alpha * A) / a0;
    b->b1 = (-2.0f * cos_w) / a0;
    b->b2 = (1.0f - alpha * A) / a0;
    b->a1 = (-2.0f * cos_w) / a0;
    b->a2 = (1.0f - alpha / A) / a0;
    b->s1 = 0.0f;
    b->s2 = 0.0f;
}

static void design_notch(BiquadFilter *b, float f0, float q, float fs) {
    float omega = 2.0f * (float)M_PI * f0 / fs;
    float alpha = sinf(omega) / (2.0f * q);
    float cos_w = cosf(omega);

    float a0 = 1.0f + alpha;
    b->b0 = 1.0f / a0;
    b->b1 = (-2.0f * cos_w) / a0;
    b->b2 = 1.0f / a0;
    b->a1 = (-2.0f * cos_w) / a0;
    b->a2 = (1.0f - alpha) / a0;
    b->s1 = 0.0f;
    b->s2 = 0.0f;
}

void AdaptiveAGC_GetDefaultConfig(AGCConfig *config) {
    if (!config) return;
    config->target_level_rms = 0.125f;      /* -18 dBFS standard broadcast speech target */
    config->max_gain_db = 24.0f;            /* Up to +24dB boost for comfortable audibility */
    config->min_gain_db = -12.0f;           /* Down to -12dB attenuation for shouting */
    config->noise_gate_threshold = 0.001f;  /* -60 dBFS silence freeze threshold */
    config->attack_time_ms = 10.0f;         /* Fast attack against loudness spikes */
    config->release_time_ms = 300.0f;       /* Slow release to prevent breathing effects */
    config->silence_standby_gain = 1.0f;    /* Relax to unity in silence unless pre-ANS readiness overrides */
    config->enable_eq = true;
    config->enable_limiter = true;
    /* Downward expansion is off by default so existing callers/tests keep their
     * exact behaviour; the board front-end opts in explicitly. */
    config->expand_range_db = 0.0f;
    config->expand_ratio = 1.0f;
    config->expand_max_cut_db = 0.0f;
}

int AdaptiveAGC_Init(AdaptiveAGC *agc, const AGCConfig *config, int sample_rate) {
    if (!agc) return -1;
    if (config) {
        agc->config = *config;
    } else {
        AdaptiveAGC_GetDefaultConfig(&agc->config);
    }

    float fs = (float)sample_rate;
    float frame_ms = 16.0f;

    /* Time constants for gain adaptation */
    agc->alpha_attack = 1.0f - expf(-frame_ms / agc->config.attack_time_ms);
    agc->alpha_release = 1.0f - expf(-frame_ms / agc->config.release_time_ms);

    agc->max_gain_linear = powf(10.0f, agc->config.max_gain_db / 20.0f);
    agc->min_gain_linear = powf(10.0f, agc->config.min_gain_db / 20.0f);

    /* Setup 4th-order (24dB/oct) steep HPF at 85Hz: cuts sub-80Hz rumble cleanly */
    design_hpf(&agc->hpf_80hz_stage1, 85.0f, fs);
    design_hpf(&agc->hpf_80hz_stage2, 85.0f, fs);
    design_peaking_eq(&agc->presence_eq, 2800.0f, 1.2f, 1.5f, fs);

    AdaptiveAGC_Reset(agc);
    return 0;
}

void AdaptiveAGC_Reset(AdaptiveAGC *agc) {
    if (!agc) return;
    agc->current_gain = 1.0f;
    agc->smoothed_rms = 0.001f;
    agc->expand_env = 0.001f;
    agc->expand_gain = 1.0f;
    agc->noise_floor_rms = 0.0005f;  /* Initial baseline ~ -66 dBFS */
    agc->vad_speech_active = 1;
    biquad_reset(&agc->hpf_80hz_stage1);
    biquad_reset(&agc->hpf_80hz_stage2);
    biquad_reset(&agc->presence_eq);
}

void AdaptiveAGC_SetVAD(AdaptiveAGC *agc, int is_speech) {
    if (agc) agc->vad_speech_active = is_speech ? 1 : 0;
}

static inline float soft_limiter(float x) {
    const float knee = 0.72f;
    const float ceiling = 0.90f; /* -0.9 dBFS anti-clipping headroom */
    const float headroom = ceiling - knee;
    if (x > knee) {
        float delta = x - knee;
        return knee + headroom * tanhf(delta / headroom);
    } else if (x < -knee) {
        float delta = -x - knee;
        return -(knee + headroom * tanhf(delta / headroom));
    }
    return x;
}

void AdaptiveAGC_ProcessFloat(
    AdaptiveAGC *agc,
    const float *in_audio,
    float *out_audio,
    int num_samples
) {
    if (!agc || !in_audio || !out_audio || num_samples <= 0) return;

    /* 1. Calculate frame RMS and peak (zero-latency frame lookahead) */
    float sum_sq = 0.0f;
    float frame_peak = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
        float x = in_audio[i];
        sum_sq += x * x;
        float ax = fabsf(x);
        if (ax > frame_peak) frame_peak = ax;
    }
    float rms = sqrtf(sum_sq / (float)num_samples);

    /* Smooth active-speech RMS tracking (fast attack prevents onset overshoot) */
    float min_gate = (agc->config.noise_gate_threshold > 0.0f) ? agc->config.noise_gate_threshold : 0.0002f;

    /* Track ambient noise floor (only drift upward during non-speech frames) */
    if (rms < agc->noise_floor_rms) {
        agc->noise_floor_rms = 0.85f * agc->noise_floor_rms + 0.15f * rms;
    } else if (!agc->vad_speech_active) {
        agc->noise_floor_rms = 0.992f * agc->noise_floor_rms + 0.008f * rms;
    }

    /* 2. Compute Target Gain with Active Speech Tracking & Pause Hold */
    float start_gain = agc->current_gain;
    float target_gain = agc->current_gain;

    /* Gate: must have active VAD AND frame energy above noise floor + min_gate */
    float speech_threshold = agc->noise_floor_rms * 1.35f;
    if (speech_threshold < min_gate) speech_threshold = min_gate;

    if (agc->vad_speech_active && rms > speech_threshold) {
        /* Update active speech RMS envelope:
         * Fast upward tracking on loud onsets, responsive downward tracking for quiet far-field words. */
        float rms_alpha = (rms > agc->smoothed_rms) ? 0.40f : 0.15f;
        agc->smoothed_rms = (1.0f - rms_alpha) * agc->smoothed_rms + rms_alpha * rms;

        /* Speech present: calculate desired gain to hit target level */
        target_gain = agc->config.target_level_rms / (agc->smoothed_rms + 1e-6f);
        if (target_gain > agc->max_gain_linear) target_gain = agc->max_gain_linear;
        if (target_gain < agc->min_gain_linear) target_gain = agc->min_gain_linear;

        /* Asymmetric smoothing: fast attack down, slow release up */
        if (target_gain < agc->current_gain) {
            agc->current_gain += agc->alpha_attack * (target_gain - agc->current_gain);
        } else {
            agc->current_gain += agc->alpha_release * (target_gain - agc->current_gain);
        }
    } else if (agc->vad_speech_active) {
        /* Brief inter-word pause inside hangover: HOLD gain steady */
        if (agc->current_gain > agc->max_gain_linear) {
            agc->current_gain = agc->max_gain_linear;
        }
    } else {
        /* True long silence or single-talk echo:
         * Relax toward configured standby gain (default 1.0x for deep-black silence post-ANS,
         * or +8 dB for pre-ANS far-field readiness), while the 2ms guard prevents 1m bursts. */
        float standby_gain = (agc->config.silence_standby_gain > 0.0f) ? agc->config.silence_standby_gain : 1.0f;
        if (standby_gain > agc->max_gain_linear) standby_gain = agc->max_gain_linear;
        if (standby_gain < agc->min_gain_linear) standby_gain = agc->min_gain_linear;
        agc->current_gain += agc->alpha_release * (standby_gain - agc->current_gain);
        float ready_rms = fminf(agc->config.target_level_rms * 0.25f, 0.025f);
        agc->smoothed_rms = 0.90f * agc->smoothed_rms + 0.10f * ready_rms;
    }

    /* Downward expansion below the speech envelope.
     * The gain above is derived from speech RMS, so reverb tails and background
     * noise inherit the same +9..+16 dB lift and become audible between words
     * (measured: non-speech frames were being amplified 0.8-4.2 dB MORE than
     * speech). Pull the gain down in proportion to how far the current frame
     * sits below the tracked speech level, so inter-word gaps stay quiet. */
    if (agc->config.expand_range_db > 0.0f) {
        /* Slow speech-envelope reference: only frames well below the recent
         * speech level are candidates for attenuation. */
        if (rms > agc->expand_env) {
            agc->expand_env += 0.45f * (rms - agc->expand_env);
        } else {
            agc->expand_env += 0.12f * (rms - agc->expand_env);
        }
        float ref = agc->expand_env + 1e-9f;
        float rel_db = 20.0f * log10f((rms + 1e-9f) / ref);

        float cut_db = 0.0f;
        if (rel_db < -1.0f) {
            float below = fminf(-rel_db, agc->config.expand_range_db);
            cut_db = below * agc->config.expand_ratio;
            if (cut_db > agc->config.expand_max_cut_db) cut_db = agc->config.expand_max_cut_db;
        }

        /* Smooth the cut itself. Recomputing it per frame and multiplying the
         * gain directly made 16% of frames jump >2 dB, which is audible as
         * choppiness. Attenuate promptly, recover slowly, and never step more
         * than a fraction of the remaining distance in one frame. */
        float target_cut = powf(10.0f, -cut_db / 20.0f);
        float a = (target_cut < agc->expand_gain) ? 0.15f : 0.04f;
        agc->expand_gain += a * (target_cut - agc->expand_gain);
        /* NOTE: not applied to current_gain here. current_gain is a persistent
         * integrator, so multiplying into it every frame compounds the cut and
         * collapses the level. The smoothed factor is applied per-sample below. */
    }

    /* Zero-latency frame-level overshoot guard:
     * Separate sustained RMS clamping from brief plosive peak clamping so first-word bursts
     * and clipping vanish while keeping sustained speech loud, warm, and full. */
    float end_gain = agc->current_gain;
    bool fast_clamp = false;
    if (rms > speech_threshold || frame_peak > 0.02f) {
        float rms_safe_gain = (agc->config.target_level_rms * 1.45f) / (rms + 1e-6f);
        float peak_safe_gain = 0.65f / (frame_peak + 1e-6f);
        if (rms_safe_gain < agc->min_gain_linear) rms_safe_gain = agc->min_gain_linear;
        if (peak_safe_gain < agc->min_gain_linear) peak_safe_gain = agc->min_gain_linear;

        if (agc->current_gain > rms_safe_gain) {
            agc->current_gain = rms_safe_gain;
            end_gain = agc->current_gain;
            if (rms > agc->smoothed_rms) {
                agc->smoothed_rms = fmaxf(agc->smoothed_rms, rms * 0.85f);
            }
            fast_clamp = true;
        }
        if (end_gain > peak_safe_gain) {
            end_gain = peak_safe_gain;
            /* Gently nudge sustained gain on brief plosive peaks without dragging down loudness */
            agc->current_gain = 0.75f * agc->current_gain + 0.25f * end_gain;
            fast_clamp = true;
        }
    }

    /* On sharp onset clamping, ramp down within 2ms (32 samples @ 16kHz) so the first syllable never pops. */
    int ramp_samples = (fast_clamp && end_gain < start_gain && num_samples > 32) ? 32 : num_samples;
    float gain_step = (end_gain - start_gain) / (float)ramp_samples;

    /* 3. Sample-by-Sample Gain Interpolation, EQ, and Soft Limiter */
    float g = start_gain;
    for (int i = 0; i < num_samples; ++i) {
        if (i < ramp_samples) {
            g += gain_step;
        } else {
            g = end_gain;
        }
        /* expand_gain is an independent, smoothed attenuation factor; applying it
         * here keeps it from compounding into the persistent AGC integrator. */
        float y = in_audio[i] * g * agc->expand_gain;

        /* Equalizer stages: 4th-order 85Hz HPF + Presence EQ (NO 5.5kHz notch!) */
        if (agc->config.enable_eq) {
            y = biquad_process(&agc->hpf_80hz_stage1, y);
            y = biquad_process(&agc->hpf_80hz_stage2, y);
            y = biquad_process(&agc->presence_eq, y);
        }

        /* Anti-clipping soft peak limiter */
        if (agc->config.enable_limiter) {
            y = soft_limiter(y);
        }

        out_audio[i] = y;
    }
}

static inline int16_t sat_int16(float val) {
    float scaled = val * 32767.0f;
    if (scaled > 32767.0f) return 32767;
    if (scaled < -32768.0f) return -32768;
    return (int16_t)roundf(scaled);
}

void AdaptiveAGC_ProcessInt16(
    AdaptiveAGC *agc,
    const int16_t *in_audio,
    int16_t *out_audio,
    int num_samples
) {
    if (!agc || !in_audio || !out_audio || num_samples <= 0) return;

    float f_in[DEFAULT_FRAME_SIZE];
    float f_out[DEFAULT_FRAME_SIZE];
    int count = (num_samples > DEFAULT_FRAME_SIZE) ? DEFAULT_FRAME_SIZE : num_samples;

    const float scale = 1.0f / 32768.0f;
    for (int i = 0; i < count; ++i) {
        f_in[i] = (float)in_audio[i] * scale;
    }

    AdaptiveAGC_ProcessFloat(agc, f_in, f_out, count);

    for (int i = 0; i < count; ++i) {
        out_audio[i] = sat_int16(f_out[i]);
    }
}

float AdaptiveAGC_GetCurrentGain(const AdaptiveAGC *agc) {
    return agc ? agc->current_gain : 1.0f;
}
