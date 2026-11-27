#include "beamformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SPEED_OF_SOUND 343.0f

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

/* 4th-order Butterworth Highpass Filter (85 Hz at 16kHz) */
static void hpf_init_coefficients(float b[5], float a[5], float fc, float fs) {
    float omega = 2.0f * (float)M_PI * fc / fs;
    float k = tanf(omega * 0.5f);
    float k2 = k * k;
    float k3 = k2 * k;
    float k4 = k2 * k2;

    float sqrt2 = 1.41421356237f;
    float denom = (k2 + sqrt2 * k + 1.0f) * (k2 - sqrt2 * k + 1.0f);

    /* Direct Form II Transposed 4th order HPF */
    b[0] = 1.0f / denom;
    b[1] = -4.0f / denom;
    b[2] = 6.0f / denom;
    b[3] = -4.0f / denom;
    b[4] = 1.0f / denom;

    a[0] = 1.0f;
    a[1] = (4.0f * k4 + 2.0f * sqrt2 * k3 - 2.0f * sqrt2 * k - 4.0f) / denom;
    a[2] = (6.0f * k4 - 2.0f * k2 + 6.0f) / denom;
    a[3] = (4.0f * k4 - 2.0f * sqrt2 * k3 + 2.0f * sqrt2 * k - 4.0f) / denom;
    a[4] = (k4 - sqrt2 * k3 + 2.0f * k2 - sqrt2 * k + 1.0f) / denom;
}

static inline float hpf_process(float x, float state[4]) {
    /* Hardcoded pre-calculated coefficients for 85Hz @ 16kHz 4th-order Linkwitz-Riley */
    /* Two cascaded 2nd order HPFs */
    /* Stage 1 */
    const float b0 = 0.976648f, b1 = -1.953296f, b2 = 0.976648f;
    const float a1 = -1.952876f, a2 = 0.953716f;

    float y1 = b0 * x + state[0];
    state[0] = b1 * x - a1 * y1 + state[1];
    state[1] = b2 * x - a2 * y1;

    /* Stage 2 */
    float y2 = b0 * y1 + state[2];
    state[2] = b1 * y1 - a1 * y2 + state[3];
    state[3] = b2 * y1 - a2 * y2;

    return y2;
}

int Beamformer_Init(Beamformer *bf, const BeamformerConfig *cfg) {
    if (!bf) return -1;
    memset(bf, 0, sizeof(Beamformer));

    /* Configuration */
    if (cfg) {
        bf->config = *cfg;
    } else {
        bf->config.sample_rate = 16000.0f;
        bf->config.array_radius_m = 0.028f; /* 56mm ring array */
        bf->config.mode = BEAM_MODE_AUTO_TRACK;
        bf->config.fixed_angle_deg = 0.0f;
        bf->config.doa_smooth_alpha = 0.15f;
        bf->config.postfilter_min_gain = 0.20f; /* -14dB dereverb floor */
        bf->config.ref_scale = 1.0f;
        bf->config.enable_postfilter = true;
    }

    /* 1. Initialize FastFFT */
    if (FastFFT_Init(&bf->fft, BEAM_FFT_SIZE) != 0) {
        return -2;
    }

    /* 2. Precompute Sqrt-Hann Analysis & Synthesis Windows (COLA compliant) */
    for (int n = 0; n < BEAM_FFT_SIZE; ++n) {
        float hann = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)n / (float)(BEAM_FFT_SIZE)));
        float sqrt_hann = sqrtf(hann);
        bf->analysis_window[n] = sqrt_hann;
        bf->synthesis_window[n] = sqrt_hann;
    }

    /* 3. Microphones Physical Positions (Clockwise 6-mic circular ring) */
    for (int m = 0; m < BEAM_NUM_MICS; ++m) {
        bf->mic_angles_rad[m] = (float)m * (float)M_PI / 3.0f; /* 0, 60, 120, 180, 240, 300 deg */
        bf->mic_x[m] = bf->config.array_radius_m * cosf(bf->mic_angles_rad[m]);
        bf->mic_y[m] = bf->config.array_radius_m * sinf(bf->mic_angles_rad[m]);
    }

    /* 4. Precompute candidate steering delays for 72 angles (0 to 355 deg in 5-deg steps) */
    float delta_theta = (float)(2.0 * M_PI / (double)BEAM_NUM_DOA_ANGLES);
    for (int a = 0; a < BEAM_NUM_DOA_ANGLES; ++a) {
        float theta = (float)a * delta_theta;
        for (int m = 0; m < BEAM_NUM_MICS; ++m) {
            /* Time delay relative to array center: tau = - (x*cos(th) + y*sin(th)) / c */
            float proj = bf->mic_x[m] * cosf(theta) + bf->mic_y[m] * sinf(theta);
            bf->candidate_delays[a][m] = -proj / SPEED_OF_SOUND;
        }
    }

    /* 5. Default tracking angle */
    bf->current_doa_deg = (bf->config.mode == BEAM_MODE_FIXED_ANGLE) ? bf->config.fixed_angle_deg : 0.0f;
    bf->smoothed_doa_rad = bf->current_doa_deg * (float)M_PI / 180.0f;
    bf->doa_confidence = 1.0f;

    return 0;
}

void Beamformer_Reset(Beamformer *bf) {
    if (!bf) return;
    memset(bf->in_history, 0, sizeof(bf->in_history));
    memset(bf->ref_history, 0, sizeof(bf->ref_history));
    memset(bf->ola_buffer, 0, sizeof(bf->ola_buffer));
    memset(bf->hpf_state, 0, sizeof(bf->hpf_state));
    memset(bf->prev_cross_power, 0, sizeof(bf->prev_cross_power));
    memset(bf->prev_total_power, 0, sizeof(bf->prev_total_power));
    bf->active_speech_count = 0;
}

float Beamformer_GetDOAAngle(const Beamformer *bf) {
    if (!bf) return 0.0f;
    return bf->current_doa_deg;
}

int Beamformer_Process(
    Beamformer *bf,
    const int16_t *in_interleaved_8ch,
    int16_t *out_fused_mono,
    int16_t *out_ref_mono
) {
    if (!bf || !in_interleaved_8ch || !out_fused_mono || !out_ref_mono) return -1;

    /* 1. Shift input history buffer and load new 256 samples */
    for (int m = 0; m < BEAM_NUM_MICS; ++m) {
        memmove(bf->in_history[m], bf->in_history[m] + BEAM_FRAME_SIZE, BEAM_FRAME_SIZE * sizeof(float));
    }
    memmove(bf->ref_history, bf->ref_history + BEAM_FRAME_SIZE, BEAM_FRAME_SIZE * sizeof(float));

    float frame_energy = 0.0f;
    for (int i = 0; i < BEAM_FRAME_SIZE; ++i) {
        int base_idx = i * BEAM_TOTAL_IN_CH;
        for (int m = 0; m < BEAM_NUM_MICS; ++m) {
            float s = (float)in_interleaved_8ch[base_idx + m];
            bf->in_history[m][BEAM_FRAME_SIZE + i] = s;
            if (m == 0) frame_energy += s * s;
        }
        /* Ch6 is speaker loopback reference */
        float ref_s = (float)in_interleaved_8ch[base_idx + 6] * bf->config.ref_scale;
        bf->ref_history[BEAM_FRAME_SIZE + i] = ref_s;
    }
    frame_energy /= (float)BEAM_FRAME_SIZE;

    /* 2. Windowing & Forward Real FFT on 6 microphones */
    Complex32 X[BEAM_NUM_MICS][BEAM_NUM_BINS];
    float windowed_buf[BEAM_FFT_SIZE];

    for (int m = 0; m < BEAM_NUM_MICS; ++m) {
        for (int n = 0; n < BEAM_FFT_SIZE; ++n) {
            windowed_buf[n] = bf->in_history[m][n] * bf->analysis_window[n];
        }
        FastFFT_RealForward(&bf->fft, windowed_buf, X[m]);
    }

    /* 3. DOA Estimation: Steered Response Power with Phase Transform (SRP-PHAT) */
    /* Target speech frequency band: 300 Hz ~ 3500 Hz -> bins 10 to 112 */
    const int k_min = 10;
    const int k_max = 112;

    /* Only track DOA when energy indicates speech (avoid tracking noise floor) */
    if (bf->config.mode == BEAM_MODE_AUTO_TRACK && frame_energy > 40000.0f) { /* > -38 dBFS */
        Complex32 U[BEAM_NUM_MICS][BEAM_NUM_BINS];
        for (int m = 0; m < BEAM_NUM_MICS; ++m) {
            for (int k = k_min; k <= k_max; ++k) {
                float mag = sqrtf(X[m][k].r * X[m][k].r + X[m][k].i * X[m][k].i) + 1e-6f;
                U[m][k].r = X[m][k].r / mag;
                U[m][k].i = X[m][k].i / mag;
            }
        }

        float max_srp = -1e9f;
        int best_angle_idx = 0;
        float bin_hz = bf->config.sample_rate / (float)BEAM_FFT_SIZE; /* 31.25 Hz */

        /* Search 72 candidate angles */
        for (int a = 0; a < BEAM_NUM_DOA_ANGLES; ++a) {
            float total_power = 0.0f;
            for (int k = k_min; k <= k_max; ++k) {
                float freq = (float)k * bin_hz;
                float sum_r = 0.0f;
                float sum_i = 0.0f;

                for (int m = 0; m < BEAM_NUM_MICS; ++m) {
                    float tau = bf->candidate_delays[a][m];
                    float phase = 2.0f * (float)M_PI * freq * tau;
                    float cp = cosf(phase);
                    float sp = sinf(phase);

                    /* U * exp(j * phase) */
                    sum_r += U[m][k].r * cp - U[m][k].i * sp;
                    sum_i += U[m][k].r * sp + U[m][k].i * cp;
                }
                total_power += (sum_r * sum_r + sum_i * sum_i);
            }

            if (total_power > max_srp) {
                max_srp = total_power;
                best_angle_idx = a;
            }
        }

        /* Calculate measured DOA angle in radians */
        float measured_angle_rad = (float)best_angle_idx * (float)(2.0 * M_PI / (double)BEAM_NUM_DOA_ANGLES);

        /* Circular smooth tracking: delta = atan2(sin(m - s), cos(m - s)) */
        float diff_rad = atan2f(sinf(measured_angle_rad - bf->smoothed_doa_rad),
                                cosf(measured_angle_rad - bf->smoothed_doa_rad));
        
        bf->smoothed_doa_rad += bf->config.doa_smooth_alpha * diff_rad;
        if (bf->smoothed_doa_rad < 0.0f) bf->smoothed_doa_rad += (float)(2.0 * M_PI);
        if (bf->smoothed_doa_rad >= (float)(2.0 * M_PI)) bf->smoothed_doa_rad -= (float)(2.0 * M_PI);

        bf->current_doa_deg = bf->smoothed_doa_rad * 180.0f / (float)M_PI;
        bf->doa_confidence = clipf(max_srp / (float)(k_max - k_min + 1) / 36.0f, 0.0f, 1.0f);
        bf->active_speech_count++;
    } else if (bf->config.mode == BEAM_MODE_FIXED_ANGLE) {
        bf->current_doa_deg = bf->config.fixed_angle_deg;
        bf->smoothed_doa_rad = bf->config.fixed_angle_deg * (float)M_PI / 180.0f;
    }

    /* 4. Frequency-Domain Delay-and-Sum Beam Steering towards current_doa */
    float target_angle_rad = bf->smoothed_doa_rad;
    float steer_delays[BEAM_NUM_MICS];
    for (int m = 0; m < BEAM_NUM_MICS; ++m) {
        float proj = bf->mic_x[m] * cosf(target_angle_rad) + bf->mic_y[m] * sinf(target_angle_rad);
        steer_delays[m] = -proj / SPEED_OF_SOUND;
    }

    Complex32 X_steered[BEAM_NUM_MICS][BEAM_NUM_BINS];
    Complex32 Y[BEAM_NUM_BINS];
    float bin_hz = bf->config.sample_rate / (float)BEAM_FFT_SIZE;
    const float inv_mics = 1.0f / (float)BEAM_NUM_MICS;

    for (int k = 0; k < BEAM_NUM_BINS; ++k) {
        float freq = (float)k * bin_hz;
        float sum_r = 0.0f;
        float sum_i = 0.0f;

        for (int m = 0; m < BEAM_NUM_MICS; ++m) {
            float phase = 2.0f * (float)M_PI * freq * steer_delays[m];
            float cp = cosf(phase);
            float sp = sinf(phase);

            /* In-phase steering: X[m] * exp(j * phase) */
            float sr = X[m][k].r * cp - X[m][k].i * sp;
            float si = X[m][k].r * sp + X[m][k].i * cp;

            X_steered[m][k].r = sr;
            X_steered[m][k].i = si;

            sum_r += sr;
            sum_i += si;
        }

        Y[k].r = sum_r * inv_mics;
        Y[k].i = sum_i * inv_mics;

        /* 5. Spatial Coherence Dereverberation Post-Filter */
        if (bf->config.enable_postfilter && k >= 4) {
            /* Compute coherent inter-channel cross-power vs total power */
            float cross_power = 0.0f;
            float total_power = 0.0f;

            for (int i = 0; i < BEAM_NUM_MICS; ++i) {
                float pi = X_steered[i][k].r * X_steered[i][k].r + X_steered[i][k].i * X_steered[i][k].i;
                total_power += pi;
                for (int j = i + 1; j < BEAM_NUM_MICS; ++j) {
                    cross_power += (X_steered[i][k].r * X_steered[j][k].r + X_steered[i][k].i * X_steered[j][k].i);
                }
            }

            /* Recursive temporal smoothing (alpha = 0.80) to eliminate speech flutter */
            const float alpha_c = 0.80f;
            bf->prev_cross_power[k] = alpha_c * bf->prev_cross_power[k] + (1.0f - alpha_c) * cross_power;
            bf->prev_total_power[k] = alpha_c * bf->prev_total_power[k] + (1.0f - alpha_c) * total_power;

            /* Normalize: average cross-power / average total power */
            float norm_cross = (2.0f * bf->prev_cross_power[k]) / (float)(BEAM_NUM_MICS * (BEAM_NUM_MICS - 1));
            float norm_total = bf->prev_total_power[k] * inv_mics + 1e-6f;
            float coherence_gain = norm_cross / norm_total;

            coherence_gain = clipf(coherence_gain, bf->config.postfilter_min_gain, 1.0f);

            Y[k].r *= coherence_gain;
            Y[k].i *= coherence_gain;
        }
    }

    /* 6. Inverse Real FFT & Synthesis Overlap-Add */
    float time_synth[BEAM_FFT_SIZE];
    FastFFT_RealInverse(&bf->fft, Y, time_synth);

    for (int n = 0; n < BEAM_FFT_SIZE; ++n) {
        bf->ola_buffer[n] += time_synth[n] * bf->synthesis_window[n];
    }

    /* 7. Output First 256 Samples with 85Hz HPF */
    for (int i = 0; i < BEAM_FRAME_SIZE; ++i) {
        float sample = bf->ola_buffer[i];
        float filtered = hpf_process(sample, bf->hpf_state);
        out_fused_mono[i] = float_to_int16(filtered);

        /* Reference channel: delay-matched (previous frame 0..255) */
        float ref_val = bf->ref_history[i];
        out_ref_mono[i] = float_to_int16(ref_val);
    }

    /* Shift OLA buffer by 256 samples */
    memmove(bf->ola_buffer, bf->ola_buffer + BEAM_FRAME_SIZE, BEAM_FRAME_SIZE * sizeof(float));
    memset(bf->ola_buffer + BEAM_FRAME_SIZE, 0, BEAM_FRAME_SIZE * sizeof(float));

    return 0;
}
