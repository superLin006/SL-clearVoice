#include "linear_aec.h"
#include <math.h>
#include <string.h>

#define REGULARIZATION_DELTA 1e-4f

void LinearAEC_GetDefaultConfig(LinearAECConfig *config) {
    if (!config) return;
    config->sample_rate = 16000;
    config->num_partitions = 8;
    config->step_size = 0.25f;
    config->leakage = 0.99995f;
    config->min_erle_freeze = 1.5f;
    config->pre_delay_samples = 130; /* 130 samples aligns RK3588 soft loopback to direct path at tap 16 */
}

int LinearAEC_Init(LinearAEC *aec, const LinearAECConfig *config) {
    if (!aec) return -1;
    if (config) {
        aec->config = *config;
    } else {
        LinearAEC_GetDefaultConfig(&aec->config);
    }

    if (aec->config.num_partitions > AEC_MAX_PARTITIONS) {
        aec->config.num_partitions = AEC_MAX_PARTITIONS;
    }
    if (aec->config.num_partitions < 2) {
        aec->config.num_partitions = 2;
    }

    if (FastFFT_Init(&aec->fft, AEC_FFT_SIZE) != 0) {
        return -1;
    }

    LinearAEC_Reset(aec);
    return 0;
}

void LinearAEC_Reset(LinearAEC *aec) {
    if (!aec) return;
    memset(aec->ref_delay_ring, 0, sizeof(aec->ref_delay_ring));
    aec->ref_delay_pos = 0;

    memset(aec->w_bg, 0, sizeof(aec->w_bg));
    memset(aec->w_fg, 0, sizeof(aec->w_fg));
    memset(aec->x_ring, 0, sizeof(aec->x_ring));
    aec->ring_pos = AEC_NLMS_TAPS;
    aec->norm_x = 0.0;
    aec->err_pow_smooth = 1e-4f;
    aec->fg_valid = false;

    memset(aec->X_history, 0, sizeof(aec->X_history));
    memset(aec->W, 0, sizeof(aec->W));
    memset(aec->x_old, 0, sizeof(aec->x_old));
    for (int k = 0; k < AEC_NUM_BINS; ++k) {
        aec->power_X[k] = REGULARIZATION_DELTA;
    }
    aec->is_double_talk = false;
    aec->erle_db = 0.0f;
    aec->smoothed_erle_db = 0.0f;
    aec->near_energy = 0.0f;
    aec->echo_energy = 0.0f;
    aec->error_energy = 0.0f;
    aec->peak_tap = 16;
    aec->ref_reverb_energy = 0.0f;
}

void LinearAEC_ProcessFloat(
    LinearAEC *aec,
    const float *mic_in,
    const float *ref_in,
    float *out_error,
    float *out_echo_est
) {
    if (!aec || !mic_in || !ref_in || !out_error || !out_echo_est) return;

    const int L = AEC_NLMS_TAPS;
    const float mu_base = (aec->config.step_size > 0.01f && aec->config.step_size <= 0.30f) ? aec->config.step_size : 0.20f;
    float *w_bg = aec->w_bg;
    float *w_fg = aec->w_fg;

    float f_ebg[AEC_FRAME_SIZE];
    float f_efg[AEC_FRAME_SIZE];
    float f_dbg[AEC_FRAME_SIZE];
    float f_dfg[AEC_FRAME_SIZE];
    float f_xc[AEC_FRAME_SIZE];

    double y_sum = 0.0;
    double e_bg_sum = 0.0;
    double e_fg_sum = 0.0;
    double d_fg_sum = 0.0;
    double d_bg_sum = 0.0;
    double xc_sum = 0.0;
    double e_ref = 0.0;
    double cross_ex_sum = 0.0;

    /* Dynamically track acoustic direct-path tap */
    int best_tap = 32;
    float max_mag = 0.0f;
    const float *w_inspect = aec->fg_valid ? w_fg : w_bg;
    for (int k = 0; k < L; ++k) {
        float mag = fabsf(w_inspect[k]);
        if (mag > max_mag) {
            max_mag = mag;
            best_tap = k;
        }
    }
    if (max_mag > 0.005f) {
        aec->peak_tap = best_tap;
    }

    /* 1. Sample-accurate pre-delay alignment on reference signal */
    float delayed_ref[AEC_FRAME_SIZE];
    int delay = aec->config.pre_delay_samples;
    if (delay > 0 && delay < AEC_MAX_PRE_DELAY) {
        for (int i = 0; i < AEC_FRAME_SIZE; ++i) {
            aec->ref_delay_ring[aec->ref_delay_pos] = ref_in[i];
            int read_pos = (aec->ref_delay_pos - delay + AEC_MAX_PRE_DELAY) % AEC_MAX_PRE_DELAY;
            delayed_ref[i] = aec->ref_delay_ring[read_pos];
            aec->ref_delay_pos = (aec->ref_delay_pos + 1) % AEC_MAX_PRE_DELAY;
        }
    } else {
        memcpy(delayed_ref, ref_in, AEC_FRAME_SIZE * sizeof(float));
    }

    const float leak = aec->config.leakage;

    for (int i = 0; i < AEC_FRAME_SIZE; ++i) {
        int pos = aec->ring_pos;
        float x_old_val = aec->x_ring[pos + L - 1];
        float x_new_val = delayed_ref[i];

        aec->norm_x += (double)x_new_val * (double)x_new_val - (double)x_old_val * (double)x_old_val;
        if (aec->norm_x < 0.0) aec->norm_x = 0.0;

        pos--;
        if (pos < 0) pos = L - 1;
        aec->x_ring[pos] = x_new_val;
        aec->x_ring[pos + L] = x_new_val;
        aec->ring_pos = pos;

        const float *x_vec = &aec->x_ring[pos];
        float y = mic_in[i];

        float d_bg = 0.0f;
        float d_fg = 0.0f;
        for (int k = 0; k < L; ++k) {
            float xk = x_vec[k];
            d_bg += w_bg[k] * xk;
            d_fg += w_fg[k] * xk;
        }

        float e_bg = y - d_bg;
        float e_fg = y - d_fg;
        float xc = x_vec[aec->peak_tap]; /* Dynamic direct-path acoustic tap */

        f_ebg[i] = e_bg;
        f_efg[i] = e_fg;
        f_dbg[i] = d_bg;
        f_dfg[i] = d_fg;
        f_xc[i]  = xc;

        y_sum += (double)y * (double)y;
        e_bg_sum += (double)e_bg * (double)e_bg;
        e_fg_sum += (double)e_fg * (double)e_fg;
        d_bg_sum += (double)d_bg * (double)d_bg;
        d_fg_sum += (double)d_fg * (double)d_fg;
        xc_sum += (double)xc * (double)xc;
        e_ref += (double)x_new_val * (double)x_new_val;

        float e_best = (aec->fg_valid && e_fg_sum < e_bg_sum) ? e_fg : e_bg;
        cross_ex_sum += (double)y * (double)d_fg;
        float e2 = e_best * e_best;
        if (e2 > aec->err_pow_smooth) {
            aec->err_pow_smooth = 0.85f * aec->err_pow_smooth + 0.15f * e2;
        } else {
            aec->err_pow_smooth = 0.992f * aec->err_pow_smooth + 0.008f * e2;
        }

        if (aec->norm_x > 0.01f) {
            float step_factor = aec->is_double_talk ? 0.01f : 1.0f;
            float denom = (float)aec->norm_x + 0.02f;
            float step = (step_factor * mu_base * e_bg) / denom;
            for (int k = 0; k < L; ++k) {
                w_bg[k] = leak * w_bg[k] + step * x_vec[k];
            }
        }
    }

    float cur_ref = (float)e_ref;
    aec->ref_reverb_energy = 0.85f * aec->ref_reverb_energy;
    if (cur_ref > aec->ref_reverb_energy) {
        aec->ref_reverb_energy = cur_ref;
    }

    /* Bounded echo path impulse response constraint (Physical acoustic coupling with 15x mic gain) */
    float w_norm_sq = 0.0f;
    for (int k = 0; k < L; ++k) {
        w_norm_sq += w_bg[k] * w_bg[k];
    }
    if (w_norm_sq > 3.5f) {
        float norm_scale = sqrtf(3.5f / w_norm_sq);
        for (int k = 0; k < L; ++k) {
            w_bg[k] *= norm_scale;
        }
    }

    /* Two-Path Foreground / Background transfer logic */
    if (!aec->is_double_talk && e_ref > 1e-4) {
        if (!aec->fg_valid && e_bg_sum < 0.92 * y_sum) {
            memcpy(w_fg, w_bg, L * sizeof(float));
            aec->fg_valid = true;
        } else if (aec->fg_valid && e_bg_sum < 0.90 * e_fg_sum && e_bg_sum < y_sum) {
            memcpy(w_fg, w_bg, L * sizeof(float));
        } else if (aec->fg_valid && e_bg_sum > 2.0 * e_fg_sum && y_sum > 1e-4) {
            memcpy(w_bg, w_fg, L * sizeof(float));
        }
    }

    /* Select the optimal filter path */
    bool use_fg = false;
    if (aec->fg_valid && e_fg_sum <= e_bg_sum) {
        use_fg = true;
    } else if (e_bg_sum < y_sum) {
        use_fg = false;
    } else if (aec->fg_valid) {
        use_fg = true;
    }

    float best_err_sum = (float)(use_fg ? e_fg_sum : e_bg_sum);
    bool filter_diverged = (best_err_sum > 1.25f * (float)y_sum && y_sum > 1e-4);

    float e_err_final = best_err_sum;
    float e_est_final = (float)(use_fg ? d_fg_sum : d_bg_sum);

    if (filter_diverged) {
        for (int i = 0; i < AEC_FRAME_SIZE; ++i) {
            out_error[i] = mic_in[i];
            out_echo_est[i] = 0.0f;
        }
        e_err_final = (float)y_sum;
        e_est_final = 0.0f;
    } else {
        for (int i = 0; i < AEC_FRAME_SIZE; ++i) {
            out_error[i] = use_fg ? f_efg[i] : f_ebg[i];
            out_echo_est[i] = use_fg ? f_dfg[i] : f_dbg[i];
        }
    }

    aec->near_energy = (float)y_sum;
    aec->echo_energy = e_est_final;
    aec->error_energy = e_err_final;

    float inst_erle = 10.0f * log10f(((float)y_sum + 1e-6f) / (e_err_final + 1e-6f));
    aec->erle_db = inst_erle;
    aec->smoothed_erle_db = 0.90f * aec->smoothed_erle_db + 0.10f * inst_erle;

    /* Robust Normalized Cross-Correlation (NCC) Double-Talk Detection */
    double rho_sq = (cross_ex_sum * cross_ex_sum) / (y_sum * d_fg_sum + 1e-9);
    bool is_pure_echo = (aec->fg_valid && rho_sq > 0.65 && d_fg_sum > 1e-4);
    bool is_reverb_tail = (e_err_final < 0.10f * aec->ref_reverb_energy);
    bool is_dt_raw = (!filter_diverged && !is_reverb_tail && e_ref > 1e-4 && aec->fg_valid && !is_pure_echo &&
                      rho_sq < 0.60 && e_err_final > 1.0e-5f);

    aec->is_double_talk = is_dt_raw;
}

static inline int16_t sat_int16(float val) {
    float scaled = val * 32767.0f;
    if (scaled > 32767.0f) return 32767;
    if (scaled < -32768.0f) return -32768;
    return (int16_t)roundf(scaled);
}

void LinearAEC_ProcessInt16(
    LinearAEC *aec,
    const int16_t *mic_in,
    const int16_t *ref_in,
    int16_t *out_error,
    int16_t *out_echo_est
) {
    if (!aec || !mic_in || !ref_in || !out_error || !out_echo_est) return;

    static float f_mic[AEC_FRAME_SIZE];
    static float f_ref[AEC_FRAME_SIZE];
    static float f_err[AEC_FRAME_SIZE];
    static float f_est[AEC_FRAME_SIZE];

    const float scale = 1.0f / 32768.0f;
    for (int i = 0; i < AEC_FRAME_SIZE; ++i) {
        f_mic[i] = (float)mic_in[i] * scale;
        f_ref[i] = (float)ref_in[i] * scale;
    }

    LinearAEC_ProcessFloat(aec, f_mic, f_ref, f_err, f_est);

    for (int i = 0; i < AEC_FRAME_SIZE; ++i) {
        out_error[i] = sat_int16(f_err[i]);
        out_echo_est[i] = sat_int16(f_est[i]);
    }
}

float LinearAEC_GetERLE(const LinearAEC *aec) {
    return aec ? aec->smoothed_erle_db : 0.0f;
}

bool LinearAEC_IsDoubleTalk(const LinearAEC *aec) {
    return aec ? aec->is_double_talk : false;
}
