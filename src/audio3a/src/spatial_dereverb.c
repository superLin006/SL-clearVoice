#include "spatial_dereverb.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float clipf(float v, float min_v, float max_v) {
    if (v < min_v) return min_v;
    if (v > max_v) return max_v;
    return v;
}

static inline int16_t float_to_int16(float v) {
    if (v > 32767.0f) return 32767;
    if (v < -32768.0f) return -32768;
    return (int16_t)roundf(v);
}

int SpatialDereverb_Init(SpatialDereverb *dr, const SpatialDereverbConfig *cfg) {
    if (!dr) return -1;
    memset(dr, 0, sizeof(SpatialDereverb));

    if (cfg) {
        dr->config = *cfg;
    } else {
        dr->config.sample_rate = 16000;
        dr->config.num_mics = 8;
        dr->config.min_gain = 0.25f;           /* -12 dB suppression floor */
        dr->config.smooth_alpha_rise = 0.50f;  /* Fast attack on speech direct-path */
        dr->config.smooth_alpha_decay = 0.82f; /* Smooth tail decay to eliminate reverb */
        dr->config.enabled = true;
    }

    if (dr->config.num_mics < 1) dr->config.num_mics = 1;
    if (dr->config.num_mics > DEREVERB_MAX_MICS) dr->config.num_mics = DEREVERB_MAX_MICS;

    if (FastFFT_Init(&dr->fft, DEREVERB_FFT_SIZE) != 0) {
        return -2;
    }

    /* Periodic Hann window for Perfect Reconstruction COLA (hop = 256, N = 512) */
    for (int n = 0; n < DEREVERB_FFT_SIZE; ++n) {
        /* Standard Sqrt-Hann window */
        float w = sinf((float)M_PI * (n + 0.5f) / (float)DEREVERB_FFT_SIZE);
        dr->analysis_window[n] = w;
        dr->synthesis_window[n] = w;
    }

    SpatialDereverb_Reset(dr);
    return 0;
}

void SpatialDereverb_Reset(SpatialDereverb *dr) {
    if (!dr) return;
    memset(dr->mic_history, 0, sizeof(dr->mic_history));
    memset(dr->ola_buffer, 0, sizeof(dr->ola_buffer));
    memset(dr->prev_cross_power, 0, sizeof(dr->prev_cross_power));
    memset(dr->prev_auto_power, 0, sizeof(dr->prev_auto_power));
    dr->last_avg_coherence = 1.0f;
    dr->frame_count = 0;
}

float SpatialDereverb_ProcessInt16(SpatialDereverb *dr,
                                  const int16_t * const mic_inputs[],
                                  int16_t out_clean[DEREVERB_FRAME_SAMPLES]) {
    if (!dr || !mic_inputs || !out_clean) return 0.0f;

    int M = dr->config.num_mics;

    /* Handle single mic or disabled mode directly */
    if (!dr->config.enabled || M <= 1) {
        for (int i = 0; i < DEREVERB_FRAME_SAMPLES; ++i) {
            out_clean[i] = mic_inputs[0][i];
        }
        dr->last_avg_coherence = 1.0f;
        return 1.0f;
    }

    float inv_m = 1.0f / (float)M;
    float cross_norm_factor = 2.0f / (float)(M * (M - 1));

    /* Frequency domain spectra */
    Complex32 X[DEREVERB_MAX_MICS][DEREVERB_NUM_BINS];
    Complex32 S[DEREVERB_NUM_BINS]; /* Summed beam spectrum */

    /* 1. Analysis STFT for all M channels */
    for (int m = 0; m < M; ++m) {
        float in_buf[DEREVERB_FFT_SIZE];
        /* First half from history */
        for (int i = 0; i < DEREVERB_FRAME_SAMPLES; ++i) {
            in_buf[i] = dr->mic_history[m][i] * dr->analysis_window[i];
        }
        /* Second half from current frame input */
        for (int i = 0; i < DEREVERB_FRAME_SAMPLES; ++i) {
            float val = (float)mic_inputs[m][i];
            in_buf[DEREVERB_FRAME_SAMPLES + i] = val * dr->analysis_window[DEREVERB_FRAME_SAMPLES + i];
            dr->mic_history[m][i] = val; /* Shift for next frame */
        }
        FastFFT_RealForward(&dr->fft, in_buf, X[m]);
    }

    /* 2. Compute Beam spectrum S[k] and Spatial Coherence */
    float coherence_sum = 0.0f;
    int coherence_count = 0;

    for (int k = 0; k < DEREVERB_NUM_BINS; ++k) {
        float sum_r = 0.0f;
        float sum_i = 0.0f;
        float total_power = 0.0f;

        for (int m = 0; m < M; ++m) {
            sum_r += X[m][k].r;
            sum_i += X[m][k].i;
            total_power += (X[m][k].r * X[m][k].r + X[m][k].i * X[m][k].i);
        }

        S[k].r = sum_r * inv_m;
        S[k].i = sum_i * inv_m;

        /* Skip DC and low frequencies (< 125 Hz) to preserve fundamental bass without distortion */
        if (k < 4) {
            /* Pass-through low frequencies intact */
            continue;
        }

        /* Power of array sum S[k] */
        float beam_power = S[k].r * S[k].r + S[k].i * S[k].i;

        /*
         * Rigorous Cross-Power extraction:
         * |sum_m X_m|^2 = sum_m |X_m|^2 + 2 sum_{i < j} Re{X_i * X_j^*}
         * => sum_{i < j} Re{X_i * X_j^*} = 0.5 * (M^2 * |S|^2 - total_power)
         */
        float cross_sum = 0.5f * (((float)M * (float)M * beam_power) - total_power);
        float inst_cross = cross_sum * cross_norm_factor;
        if (inst_cross < 0.0f) inst_cross = 0.0f;

        float inst_auto = (total_power * inv_m) + 1e-9f;

        /* Asymmetric temporal smoothing: fast onset for speech, smooth decay for diffuse reverb */
        float alpha_cross = (inst_cross > dr->prev_cross_power[k]) ? dr->config.smooth_alpha_rise : dr->config.smooth_alpha_decay;
        float alpha_auto = (inst_auto > dr->prev_auto_power[k]) ? dr->config.smooth_alpha_rise : dr->config.smooth_alpha_decay;

        dr->prev_cross_power[k] = alpha_cross * dr->prev_cross_power[k] + (1.0f - alpha_cross) * inst_cross;
        dr->prev_auto_power[k] = alpha_auto * dr->prev_auto_power[k] + (1.0f - alpha_auto) * inst_auto;

        /* Spatial Coherence Gain: coherent speech ratio */
        float coherence_gain = dr->prev_cross_power[k] / (dr->prev_auto_power[k] + 1e-9f);

        /* Statistics for monitoring */
        coherence_sum += coherence_gain;
        coherence_count++;

        /* Clamp to floor suppression */
        coherence_gain = clipf(coherence_gain, dr->config.min_gain, 1.0f);

        /* Apply dereverberation post-filter gain */
        S[k].r *= coherence_gain;
        S[k].i *= coherence_gain;
    }

    if (coherence_count > 0) {
        dr->last_avg_coherence = coherence_sum / (float)coherence_count;
    }

    /* 3. Synthesis IFFT and Overlap-Add */
    float time_synth[DEREVERB_FFT_SIZE];
    FastFFT_RealInverse(&dr->fft, S, time_synth);

    for (int n = 0; n < DEREVERB_FFT_SIZE; ++n) {
        dr->ola_buffer[n] += time_synth[n] * dr->synthesis_window[n];
    }

    /* 4. Output the reconstructed 256 samples */
    for (int i = 0; i < DEREVERB_FRAME_SAMPLES; ++i) {
        out_clean[i] = float_to_int16(dr->ola_buffer[i]);
    }

    /* 5. Shift OLA buffer by 256 samples */
    memmove(dr->ola_buffer, dr->ola_buffer + DEREVERB_FRAME_SAMPLES,
            (DEREVERB_FFT_SIZE - DEREVERB_FRAME_SAMPLES) * sizeof(float));
    memset(dr->ola_buffer + (DEREVERB_FFT_SIZE - DEREVERB_FRAME_SAMPLES), 0,
           DEREVERB_FRAME_SAMPLES * sizeof(float));

    dr->frame_count++;
    return dr->last_avg_coherence;
}
