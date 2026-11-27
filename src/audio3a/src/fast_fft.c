#include "fast_fft.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint16_t reverse_bits(uint16_t x, int bits) {
    uint16_t result = 0;
    for (int i = 0; i < bits; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

int FastFFT_Init(FastFFT *fft, int nfft) {
    if (!fft || nfft > MAX_FFT_SIZE || (nfft & (nfft - 1)) != 0 || nfft < 4) {
        return -1;
    }

    fft->nfft = nfft;
    fft->half_nfft = nfft / 2;

    int log2n = 0;
    int temp = nfft;
    while (temp > 1) {
        temp >>= 1;
        log2n++;
    }

    for (int i = 0; i < nfft; ++i) {
        fft->bitrev[i] = reverse_bits((uint16_t)i, log2n);
    }

    /* Precompute twiddle factors: W_N^k = e^(-j * 2pi * k / N) */
    for (int k = 0; k < fft->half_nfft; ++k) {
        double theta = -2.0 * M_PI * (double)k / (double)nfft;
        fft->twiddles[k].r = (float)cos(theta);
        fft->twiddles[k].i = (float)sin(theta);
    }

    return 0;
}

static void complex_fft(const FastFFT *fft, Complex32 *data, int is_inverse) {
    int n = fft->nfft;

    /* Bit-reversal permutation */
    for (int i = 0; i < n; ++i) {
        int rev = fft->bitrev[i];
        if (i < rev) {
            Complex32 tmp = data[i];
            data[i] = data[rev];
            data[rev] = tmp;
        }
    }

    /* Cooley-Tukey butterfly */
    for (int len = 2; len <= n; len <<= 1) {
        int half_len = len >> 1;
        int step = n / len;

        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half_len; ++j) {
                int twiddle_idx = j * step;
                Complex32 w = fft->twiddles[twiddle_idx];
                if (is_inverse) {
                    w.i = -w.i; /* Conjugate for inverse FFT */
                }

                Complex32 u = data[i + j];
                Complex32 v;
                /* v = data[i + j + half_len] * w */
                Complex32 d = data[i + j + half_len];
                v.r = d.r * w.r - d.i * w.i;
                v.i = d.r * w.i + d.i * w.r;

                data[i + j].r = u.r + v.r;
                data[i + j].i = u.i + v.i;
                data[i + j + half_len].r = u.r - v.r;
                data[i + j + half_len].i = u.i - v.i;
            }
        }
    }

    if (is_inverse) {
        float scale = 1.0f / (float)n;
        for (int i = 0; i < n; ++i) {
            data[i].r *= scale;
            data[i].i *= scale;
        }
    }
}

void FastFFT_RealForward(const FastFFT *fft, const float *real_in, Complex32 *cplx_out) {
    if (!fft || !real_in || !cplx_out) return;
    int n = fft->nfft;
    Complex32 buf[MAX_FFT_SIZE];

    for (int i = 0; i < n; ++i) {
        buf[i].r = real_in[i];
        buf[i].i = 0.0f;
    }

    complex_fft(fft, buf, 0);

    /* Real input has conjugate symmetric spectrum, so copy first n/2 + 1 bins */
    int num_bins = fft->half_nfft + 1;
    for (int k = 0; k < num_bins; ++k) {
        cplx_out[k] = buf[k];
    }
}

void FastFFT_RealInverse(const FastFFT *fft, const Complex32 *cplx_in, float *real_out) {
    if (!fft || !cplx_in || !real_out) return;
    int n = fft->nfft;
    int half = fft->half_nfft;
    Complex32 buf[MAX_FFT_SIZE];

    /* Copy positive frequencies */
    for (int k = 0; k <= half; ++k) {
        buf[k] = cplx_in[k];
    }

    /* Reconstruct negative frequencies by conjugate symmetry: X[N-k] = X*[k] */
    for (int k = 1; k < half; ++k) {
        buf[n - k].r = cplx_in[k].r;
        buf[n - k].i = -cplx_in[k].i;
    }

    complex_fft(fft, buf, 1);

    for (int i = 0; i < n; ++i) {
        real_out[i] = buf[i].r;
    }
}
