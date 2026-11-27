#ifndef BARK_FEATURES_H
#define BARK_FEATURES_H

#include <stdint.h>
#include "fast_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_BARK_BANDS   22
#define NUM_FFT_BINS     257   /* For 512-point FFT @ 16kHz */

typedef struct {
    int start_bin;
    int end_bin;
    float inv_bandwidth;
} BarkBandDef;

typedef struct {
    BarkBandDef bands[NUM_BARK_BANDS];
    float bark_energies_err[NUM_BARK_BANDS];
    float bark_energies_echo[NUM_BARK_BANDS];
    float log_bark_err[NUM_BARK_BANDS];
    float log_bark_echo[NUM_BARK_BANDS];
    float spectral_flux;
    float prev_bark_err[NUM_BARK_BANDS];
} BarkFeatureExtractor;

void BarkFeatures_Init(BarkFeatureExtractor *fe);

/*
 * Extract Bark subband log-energies from error signal FFT and estimated echo FFT
 */
void BarkFeatures_Extract(
    BarkFeatureExtractor *fe,
    const Complex32 *E_freq,
    const Complex32 *D_hat_freq
);

/*
 * Interpolate 22 Bark gains to 257 linear FFT bins with smooth piecewise linear interpolation
 */
void BarkFeatures_InterpolateGains(
    const BarkFeatureExtractor *fe,
    const float bark_gains[NUM_BARK_BANDS],
    float fft_gains[NUM_FFT_BINS]
);

#ifdef __cplusplus
}
#endif

#endif /* BARK_FEATURES_H */
