#include "online_wpe.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ONLINE_WPE_HIST_MASK   (ONLINE_WPE_HIST_DEPTH - 1)
#define ONLINE_WPE_SCALE_FWD   (1.0f / 16.0f) /* 1 / sqrt(FRAME_SAMPLES) for unitary STFT */
#define ONLINE_WPE_SCALE_INV   16.0f

static inline int16_t clamp_float_to_int16(double v) {
    if (v > 32767.0) return 32767;
    if (v < -32768.0) return -32768;
    return (int16_t)lrint(v);
}

static void init_rinv_diag(OnlineWpe *wpe) {
    int P = wpe->P;
    double d0 = (double)wpe->config.diag_init;
    memset(wpe->rinv, 0, (size_t)ONLINE_WPE_NUM_BINS * P * P * sizeof(OnlineWpeCpx64));
    for (int k = 0; k < ONLINE_WPE_NUM_BINS; ++k) {
        OnlineWpeCpx64 *rk = wpe->rinv + (size_t)k * P * P;
        for (int p = 0; p < P; ++p) {
            rk[p * P + p].r = d0;
            rk[p * P + p].i = 0.0;
        }
    }
}

void OnlineWpe_DefaultConfig(OnlineWpeConfig *cfg, int num_mics) {
    if (!cfg) return;
    if (num_mics < 1) num_mics = 1;
    if (num_mics > ONLINE_WPE_MAX_MICS) num_mics = ONLINE_WPE_MAX_MICS;
    cfg->num_mics = num_mics;
    cfg->taps = 4;
    cfg->delay = 2;
    cfg->alpha = 0.99f;
    cfg->power_floor = 1e-4f;
    cfg->diag_init = 0.05f;
    cfg->diag_max = 1.0f;
    cfg->enabled = true;
}

int OnlineWpe_Init(OnlineWpe *wpe, const OnlineWpeConfig *cfg) {
    if (!wpe) return -1;
    memset(wpe, 0, sizeof(*wpe));

    if (cfg) {
        wpe->config = *cfg;
    } else {
        OnlineWpe_DefaultConfig(&wpe->config, ONLINE_WPE_MAX_MICS);
    }

    if (wpe->config.num_mics < 1) wpe->config.num_mics = 1;
    if (wpe->config.num_mics > ONLINE_WPE_MAX_MICS) wpe->config.num_mics = ONLINE_WPE_MAX_MICS;
    if (wpe->config.taps < 1) wpe->config.taps = 1;
    if (wpe->config.taps > ONLINE_WPE_MAX_TAPS) wpe->config.taps = ONLINE_WPE_MAX_TAPS;
    if (wpe->config.delay < 1) wpe->config.delay = 1;
    if (wpe->config.delay > ONLINE_WPE_MAX_DELAY) wpe->config.delay = ONLINE_WPE_MAX_DELAY;
    if (wpe->config.alpha < 0.80f) wpe->config.alpha = 0.80f;
    if (wpe->config.alpha > 0.9999f) wpe->config.alpha = 0.9999f;
    if (wpe->config.power_floor < 1e-9f) wpe->config.power_floor = 1e-9f;
    if (wpe->config.diag_init <= 0.0f) wpe->config.diag_init = 0.05f;
    if (wpe->config.diag_max < wpe->config.diag_init) wpe->config.diag_max = wpe->config.diag_init;

    if (FastFFT_Init(&wpe->fft, ONLINE_WPE_FFT_SIZE) != 0) {
        return -2;
    }

    for (int n = 0; n < ONLINE_WPE_FFT_SIZE; ++n) {
        wpe->window[n] = sinf((float)M_PI * (n + 0.5f) / (float)ONLINE_WPE_FFT_SIZE);
    }

    int M = wpe->config.num_mics;
    int P = M * wpe->config.taps;
    wpe->P = P;

    wpe->spec_hist = (Complex32 *)calloc(
        (size_t)ONLINE_WPE_HIST_DEPTH * ONLINE_WPE_NUM_BINS * M, sizeof(Complex32));
    wpe->rinv = (OnlineWpeCpx64 *)calloc(
        (size_t)ONLINE_WPE_NUM_BINS * P * P, sizeof(OnlineWpeCpx64));
    wpe->filt = (OnlineWpeCpx64 *)calloc(
        (size_t)ONLINE_WPE_NUM_BINS * P * M, sizeof(OnlineWpeCpx64));
    if (!wpe->spec_hist || !wpe->rinv || !wpe->filt) {
        OnlineWpe_Destroy(wpe);
        return -3;
    }

    OnlineWpe_Reset(wpe);
    wpe->initialized = 1;
    return 0;
}

void OnlineWpe_Reset(OnlineWpe *wpe) {
    if (!wpe) return;
    memset(wpe->mic_history, 0, sizeof(wpe->mic_history));
    memset(wpe->ola_buffer, 0, sizeof(wpe->ola_buffer));
    if (wpe->spec_hist) {
        memset(wpe->spec_hist, 0,
               (size_t)ONLINE_WPE_HIST_DEPTH * ONLINE_WPE_NUM_BINS * wpe->config.num_mics * sizeof(Complex32));
    }
    if (wpe->filt) {
        memset(wpe->filt, 0,
               (size_t)ONLINE_WPE_NUM_BINS * wpe->P * wpe->config.num_mics * sizeof(OnlineWpeCpx64));
    }
    if (wpe->rinv) {
        init_rinv_diag(wpe);
    }
    wpe->hist_write = 0;
    wpe->frame_count = 0;
}

void OnlineWpe_Destroy(OnlineWpe *wpe) {
    if (!wpe) return;
    free(wpe->spec_hist);
    free(wpe->rinv);
    free(wpe->filt);
    wpe->spec_hist = NULL;
    wpe->rinv = NULL;
    wpe->filt = NULL;
    wpe->initialized = 0;
}

void OnlineWpe_ProcessInt16(OnlineWpe *wpe,
                            const int16_t * const in_mics[],
                            int16_t * const out_mics[]) {
    if (!wpe || !in_mics || !out_mics) return;

    int M = wpe->config.num_mics;
    if (!wpe->initialized || !wpe->config.enabled) {
        for (int m = 0; m < M; ++m) {
            if (in_mics[m] && out_mics[m] && in_mics[m] != out_mics[m]) {
                memcpy(out_mics[m], in_mics[m], ONLINE_WPE_FRAME_SAMPLES * sizeof(int16_t));
            }
        }
        return;
    }

    const int T = wpe->config.taps;
    const int D = wpe->config.delay;
    const int P = wpe->P;
    const double alpha = (double)wpe->config.alpha;
    const double inv_alpha = 1.0 / alpha;
    const double inv_m = 1.0 / (double)M;
    const double inv_p = 1.0 / (double)P;
    const double power_floor = (double)wpe->config.power_floor;
    const double diag_max = (double)wpe->config.diag_max;
    const double diag_load = 1e-6;

    Complex32 X[ONLINE_WPE_MAX_MICS][ONLINE_WPE_NUM_BINS];
    OnlineWpeCpx64 Y[ONLINE_WPE_MAX_MICS][ONLINE_WPE_NUM_BINS];

    /* 1. Forward STFT for all M channels and store in ring history */
    int wslot = wpe->hist_write;
    Complex32 *hist_cur = wpe->spec_hist + (size_t)wslot * ONLINE_WPE_NUM_BINS * M;

    for (int m = 0; m < M; ++m) {
        float in_buf[ONLINE_WPE_FFT_SIZE];
        for (int i = 0; i < ONLINE_WPE_FRAME_SAMPLES; ++i) {
            in_buf[i] = wpe->mic_history[m][i] * wpe->window[i];
        }
        for (int i = 0; i < ONLINE_WPE_FRAME_SAMPLES; ++i) {
            float cur = (float)in_mics[m][i] * (1.0f / 32768.0f);
            wpe->mic_history[m][i] = cur;
            in_buf[ONLINE_WPE_FRAME_SAMPLES + i] =
                cur * wpe->window[ONLINE_WPE_FRAME_SAMPLES + i];
        }
        FastFFT_RealForward(&wpe->fft, in_buf, X[m]);
        for (int k = 0; k < ONLINE_WPE_NUM_BINS; ++k) {
            X[m][k].r *= ONLINE_WPE_SCALE_FWD;
            X[m][k].i *= ONLINE_WPE_SCALE_FWD;
            hist_cur[k * M + m] = X[m][k];
        }
        /* Keep DC untouched to avoid sub-bin drift */
        Y[m][0].r = (double)X[m][0].r;
        Y[m][0].i = (double)X[m][0].i;
    }

    int ready_for_wpe = ((int)wpe->frame_count >= (D + T - 1));

    /* 2. Per-bin multi-channel causal SPD RLS-WPE filtering and update */
    for (int k = 1; k < ONLINE_WPE_NUM_BINS; ++k) {
        if (!ready_for_wpe) {
            for (int m = 0; m < M; ++m) {
                Y[m][k].r = (double)X[m][k].r;
                Y[m][k].i = (double)X[m][k].i;
            }
            continue;
        }

        OnlineWpeCpx64 x_tilde[ONLINE_WPE_MAX_MICS * ONLINE_WPE_MAX_TAPS];
        double reg_sum = 0.0;
        for (int tap = 0; tap < T; ++tap) {
            int slot = (wslot - (D + tap)) & ONLINE_WPE_HIST_MASK;
            const Complex32 *hk = wpe->spec_hist +
                                  ((size_t)slot * ONLINE_WPE_NUM_BINS + (size_t)k) * M;
            OnlineWpeCpx64 *dst = x_tilde + tap * M;
            for (int m = 0; m < M; ++m) {
                double vr = (double)hk[m].r;
                double vi = (double)hk[m].i;
                dst[m].r = vr;
                dst[m].i = vi;
                reg_sum += vr * vr + vi * vi;
            }
        }

        OnlineWpeCpx64 *rk = wpe->rinv + (size_t)k * P * P;
        OnlineWpeCpx64 *gk = wpe->filt + (size_t)k * P * M;

        /* Linear prediction error: y[m] = X[m][k] - sum_p conj(G[p][m]) * x_tilde[p] */
        OnlineWpeCpx64 err[ONLINE_WPE_MAX_MICS];
        double pred_r[ONLINE_WPE_MAX_MICS] = {0};
        double pred_i[ONLINE_WPE_MAX_MICS] = {0};
        for (int p = 0; p < P; ++p) {
            OnlineWpeCpx64 xp = x_tilde[p];
            const OnlineWpeCpx64 *g_row = gk + p * M;
            for (int m = 0; m < M; ++m) {
                OnlineWpeCpx64 gp = g_row[m];
                pred_r[m] += gp.r * xp.r + gp.i * xp.i;
                pred_i[m] += gp.r * xp.i - gp.i * xp.r;
            }
        }
        double cur_sum = 0.0;
        for (int m = 0; m < M; ++m) {
            double xr = (double)X[m][k].r;
            double xi = (double)X[m][k].i;
            cur_sum += xr * xr + xi * xi;
            err[m].r = xr - pred_r[m];
            err[m].i = xi - pred_i[m];
            Y[m][k] = err[m];
        }

        double p_cur = cur_sum * inv_m;
        double p_reg = reg_sum * inv_p;

        /* Gate updates below quiet-room floor so R^-1 never winds up during pauses */
        if (p_cur < power_floor || p_reg < power_floor * 0.25) {
            continue;
        }

        /* Rinv_x = Rinv * x_tilde */
        OnlineWpeCpx64 rinv_x[ONLINE_WPE_MAX_MICS * ONLINE_WPE_MAX_TAPS];
        double x_rinv_x = 0.0;
        for (int i = 0; i < P; ++i) {
            const OnlineWpeCpx64 *row = rk + i * P;
            double sr = 0.0;
            double si = 0.0;
            for (int j = 0; j < P; ++j) {
                OnlineWpeCpx64 a = row[j];
                OnlineWpeCpx64 b = x_tilde[j];
                sr += a.r * b.r - a.i * b.i;
                si += a.r * b.i + a.i * b.r;
            }
            rinv_x[i].r = sr;
            rinv_x[i].i = si;
            x_rinv_x += x_tilde[i].r * sr + x_tilde[i].i * si;
        }

        if (!(x_rinv_x >= 0.0) || !isfinite(x_rinv_x)) {
            memset(gk, 0, (size_t)P * M * sizeof(OnlineWpeCpx64));
            memset(rk, 0, (size_t)P * P * sizeof(OnlineWpeCpx64));
            for (int p = 0; p < P; ++p) {
                rk[p * P + p].r = (double)wpe->config.diag_init;
            }
            continue;
        }

        double denom = alpha * p_cur + x_rinv_x;
        if (!(denom > 1e-12)) {
            continue;
        }
        double inv_denom = 1.0 / denom;

        /* k_vec = rinv_x / denom */
        OnlineWpeCpx64 k_vec[ONLINE_WPE_MAX_MICS * ONLINE_WPE_MAX_TAPS];
        for (int i = 0; i < P; ++i) {
            k_vec[i].r = rinv_x[i].r * inv_denom;
            k_vec[i].i = rinv_x[i].i * inv_denom;
        }

        /* Filter update: G[p][m] += k_vec[p] * conj(err[m]) */
        for (int p = 0; p < P; ++p) {
            OnlineWpeCpx64 kp = k_vec[p];
            OnlineWpeCpx64 *g_row = gk + p * M;
            for (int m = 0; m < M; ++m) {
                OnlineWpeCpx64 em = err[m];
                g_row[m].r += kp.r * em.r + kp.i * em.i;
                g_row[m].i += kp.i * em.r - kp.r * em.i;
            }
        }

        /* Hermitian SPD-preserving covariance update:
         * Rinv = (1/alpha) * (Rinv - k_vec * rinv_x^H) + diag_load * I */
        double max_diag = 0.0;
        int need_reset = 0;
        for (int i = 0; i < P; ++i) {
            OnlineWpeCpx64 ki = k_vec[i];
            OnlineWpeCpx64 *row_i = rk + i * P;
            for (int j = 0; j < i; ++j) {
                OnlineWpeCpx64 rj = rinv_x[j];
                double sub_r = ki.r * rj.r + ki.i * rj.i;
                double sub_i = ki.i * rj.r - ki.r * rj.i;
                double nr = (row_i[j].r - sub_r) * inv_alpha;
                double ni = (row_i[j].i - sub_i) * inv_alpha;
                row_i[j].r = nr;
                row_i[j].i = ni;
                rk[j * P + i].r = nr;
                rk[j * P + i].i = -ni;
            }
            double diag_sub = ki.r * rinv_x[i].r + ki.i * rinv_x[i].i;
            double d = (row_i[i].r - diag_sub) * inv_alpha + diag_load;
            if (!isfinite(d) || d <= 0.0) {
                need_reset = 1;
                break;
            }
            row_i[i].r = d;
            row_i[i].i = 0.0;
            if (d > max_diag) max_diag = d;
        }

        if (need_reset) {
            memset(gk, 0, (size_t)P * M * sizeof(OnlineWpeCpx64));
            memset(rk, 0, (size_t)P * P * sizeof(OnlineWpeCpx64));
            for (int p = 0; p < P; ++p) {
                rk[p * P + p].r = (double)wpe->config.diag_init;
            }
        } else if (max_diag > diag_max) {
            /* Uniform scalar downscaling strictly preserves Hermitian SPD */
            double sc = diag_max / max_diag;
            for (int i = 0; i < P * P; ++i) {
                rk[i].r *= sc;
                rk[i].i *= sc;
            }
        }
    }

    wpe->hist_write = (wslot + 1) & ONLINE_WPE_HIST_MASK;
    wpe->frame_count++;

    /* 3. Inverse STFT and Overlap-Add per channel */
    for (int m = 0; m < M; ++m) {
        Complex32 spec_out[ONLINE_WPE_NUM_BINS];
        float time_buf[ONLINE_WPE_FFT_SIZE];

        for (int k = 0; k < ONLINE_WPE_NUM_BINS; ++k) {
            spec_out[k].r = (float)(Y[m][k].r * (double)ONLINE_WPE_SCALE_INV);
            spec_out[k].i = (float)(Y[m][k].i * (double)ONLINE_WPE_SCALE_INV);
        }
        FastFFT_RealInverse(&wpe->fft, spec_out, time_buf);

        for (int i = 0; i < ONLINE_WPE_FRAME_SAMPLES; ++i) {
            double sample = (double)wpe->ola_buffer[m][i] + (double)time_buf[i] * (double)wpe->window[i];
            out_mics[m][i] = clamp_float_to_int16(sample * 32768.0);
            wpe->ola_buffer[m][i] = time_buf[ONLINE_WPE_FRAME_SAMPLES + i] *
                                    wpe->window[ONLINE_WPE_FRAME_SAMPLES + i];
        }
    }
}
