#include "bark_features.h"
#include <math.h>
#include <string.h>

static const int g_bark_edges[NUM_BARK_BANDS + 1] = {
    0, 4, 7, 10, 13, 17, 21, 25, 30, 35, 41, 48, 56, 65, 75, 87, 101, 119, 141, 170, 204, 247, 257
};

void BarkFeatures_Init(BarkFeatureExtractor *fe) {
    if (!fe) return;
    memset(fe, 0, sizeof(BarkFeatureExtractor));

    for (int b = 0; b < NUM_BARK_BANDS; ++b) {
        fe->bands[b].start_bin = g_bark_edges[b];
        fe->bands[b].end_bin = g_bark_edges[b + 1];
        int count = fe->bands[b].end_bin - fe->bands[b].start_bin;
        fe->bands[b].inv_bandwidth = (count > 0) ? (1.0f / (float)count) : 1.0f;
    }
}

void BarkFeatures_Extract(
    BarkFeatureExtractor *fe,
    const Complex32 *E_freq,
    const Complex32 *D_hat_freq
) {
    if (!fe || !E_freq || !D_hat_freq) return;

    float flux = 0.0f;

    for (int b = 0; b < NUM_BARK_BANDS; ++b) {
        int start = fe->bands[b].start_bin;
        int end = fe->bands[b].end_bin;

        float sum_err = 0.0f;
        float sum_echo = 0.0f;

        for (int k = start; k < end; ++k) {
            float e_r = E_freq[k].r;
            float e_i = E_freq[k].i;
            float p_e = e_r * e_r + e_i * e_i;
            sum_err += p_e;

            float d_r = D_hat_freq[k].r;
            float d_i = D_hat_freq[k].i;
            float p_d = d_r * d_r + d_i * d_i;
            sum_echo += p_d;
        }

        float avg_err = sum_err * fe->bands[b].inv_bandwidth;
        float avg_echo = sum_echo * fe->bands[b].inv_bandwidth;

        fe->bark_energies_err[b] = avg_err;
        fe->bark_energies_echo[b] = avg_echo;

        /* Logarithmic scaling with floor to prevent log(0) */
        float log_err = log10f(avg_err + 1e-6f);
        float log_echo = log10f(avg_echo + 1e-6f);

        fe->log_bark_err[b] = log_err;
        fe->log_bark_echo[b] = log_echo;

        /* Spectral flux calculation (temporal energy change) */
        float diff = avg_err - fe->prev_bark_err[b];
        if (diff > 0.0f) flux += diff;
        fe->prev_bark_err[b] = avg_err;
    }

    fe->spectral_flux = flux;
}

void BarkFeatures_InterpolateGains(
    const BarkFeatureExtractor *fe,
    const float bark_gains[NUM_BARK_BANDS],
    float fft_gains[NUM_FFT_BINS]
) {
    if (!fe || !bark_gains || !fft_gains) return;

    /* Center bin for each band */
    float band_centers[NUM_BARK_BANDS];
    for (int b = 0; b < NUM_BARK_BANDS; ++b) {
        band_centers[b] = 0.5f * (float)(fe->bands[b].start_bin + fe->bands[b].end_bin - 1);
    }

    /* Linear interpolation between band centers */
    int current_band = 0;
    for (int k = 0; k < NUM_FFT_BINS; ++k) {
        while (current_band < NUM_BARK_BANDS - 1 && (float)k > band_centers[current_band + 1]) {
            current_band++;
        }

        if (current_band >= NUM_BARK_BANDS - 1) {
            fft_gains[k] = bark_gains[NUM_BARK_BANDS - 1];
        } else if ((float)k <= band_centers[0]) {
            fft_gains[k] = bark_gains[0];
        } else {
            float c0 = band_centers[current_band];
            float c1 = band_centers[current_band + 1];
            float alpha = ((float)k - c0) / (c1 - c0);
            fft_gains[k] = (1.0f - alpha) * bark_gains[current_band] + alpha * bark_gains[current_band + 1];
        }

        /* Clamp to [0.0f, 1.0f] */
        if (fft_gains[k] < 0.0f) fft_gains[k] = 0.0f;
        if (fft_gains[k] > 1.0f) fft_gains[k] = 1.0f;
    }
}
