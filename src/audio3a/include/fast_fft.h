#ifndef FAST_FFT_H
#define FAST_FFT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_FFT_SIZE 512

typedef struct {
    float r;
    float i;
} Complex32;

typedef struct {
    int nfft;            /* Must be power of 2, e.g. 512 */
    int half_nfft;       /* nfft / 2 */
    uint16_t bitrev[MAX_FFT_SIZE];
    Complex32 twiddles[MAX_FFT_SIZE / 2];
} FastFFT;

/* Initialize FFT tables (pre-computed sin/cos and bit-reversal) */
int  FastFFT_Init(FastFFT *fft, int nfft);

/* Forward real FFT: real_in (size nfft) -> cplx_out (size nfft/2 + 1) */
void FastFFT_RealForward(const FastFFT *fft, const float *real_in, Complex32 *cplx_out);

/* Inverse real FFT: cplx_in (size nfft/2 + 1) -> real_out (size nfft) */
void FastFFT_RealInverse(const FastFFT *fft, const Complex32 *cplx_in, float *real_out);

#ifdef __cplusplus
}
#endif

#endif /* FAST_FFT_H */
