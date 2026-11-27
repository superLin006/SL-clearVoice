#ifndef NEURAL_RES_ANS_H
#define NEURAL_RES_ANS_H

#include "bark_features.h"
#include "fast_fft.h"
#include "audio_buffer.h"
#include "rnnoise_v1.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEURAL_FEAT_DIM     (NUM_BARK_BANDS * 2 + 1) /* 22 err + 22 echo + 1 flux = 45 */
#define NEURAL_DENSE_DIM    32
#define NEURAL_GRU_DIM      32
#define NEURAL_OUT_DIM      NUM_BARK_BANDS            /* 22 band gains */

typedef struct {
    float min_gain_linear;       /* Minimum gain floor (e.g. 0.01f = -40dB, default 0.03f = -30dB) */
    float res_aggressiveness;    /* Residual echo suppression weight (default 1.2f) */
    float ans_aggressiveness;    /* Noise suppression weight (default 1.0f) */
    bool  enable_neural_ans;     /* Enable neural noise suppression */
    bool  enable_neural_res;     /* Enable neural residual echo suppression */
} NeuralConfig;

typedef struct {
    NeuralConfig          config;
    BarkFeatureExtractor  feat_extractor;
    FastFFT               fft;

    /* Neural network recurrent state */
    float                 gru_state[NEURAL_GRU_DIM];

    /* RNNoise Deep Learning Neural Network Engine */
    DenoiseState         *rnnoise_state;
    float                 rnnoise_fifo_in[512];
    int                   rnnoise_fifo_in_count;
    float                 rnnoise_fifo_out[512];
    int                   rnnoise_fifo_out_count;

    /* Overlap-Save state buffers */
    float                 prev_error_time[DEFAULT_FRAME_SIZE];
    float                 prev_echo_time[DEFAULT_FRAME_SIZE];
    float                 prev_ref_time[DEFAULT_FRAME_SIZE];
    float                 synthesis_buf[DEFAULT_FRAME_SIZE * 2];

    /* Last computed gains & 257-bin WOLA state */
    float                 wola_win[DEFAULT_FRAME_SIZE * 2];
    float                 echo_reverb_psd[NUM_FFT_BINS];
    float                 noise_psd[NUM_FFT_BINS];
    float                 echo_coupling[NUM_FFT_BINS];
    float                 beta_res[NUM_FFT_BINS];
    float                 last_bark_gains[NUM_BARK_BANDS];
    float                 last_fft_gains[NUM_FFT_BINS];
    float                 bark_noise_floor[NUM_BARK_BANDS];
    float                 speech_probability;
    int                   vad_speech_active; /* 1 = active voice, 0 = silence / pause */
    int                   hangover_cnt;
    int                   onset_cnt;
} NeuralRESANS;

/*
 * Initialize Neural Engine
 */
void NeuralRESANS_GetDefaultConfig(NeuralConfig *config);
int  NeuralRESANS_Init(NeuralRESANS *engine, const NeuralConfig *config);
void NeuralRESANS_Reset(NeuralRESANS *engine);
void NeuralRESANS_Destroy(NeuralRESANS *engine);
void NeuralRESANS_SetVAD(NeuralRESANS *engine, int is_speech);

/*
 * Process one frame of audio (Float interface [-1.0f, 1.0f])
 * @param engine: Initialized NeuralRESANS instance
 * @param error_in: Linear AEC error signal e[n] (size 256)
 * @param echo_est_in: Linear AEC estimated echo d_hat[n] (size 256)
 * @param ref_in: Far-end speaker reference signal x[n] (size 256)
 * @param clean_out: Enhanced speech output (size 256)
 */
void NeuralRESANS_ProcessFloat(
    NeuralRESANS *engine,
    const float *error_in,
    const float *echo_est_in,
    const float *ref_in,
    float *clean_out
);

/*
 * Process int16_t PCM frame
 */
void NeuralRESANS_ProcessInt16(
    NeuralRESANS *engine,
    const int16_t *error_in,
    const int16_t *echo_est_in,
    const int16_t *ref_in,
    int16_t *clean_out
);

/* Get speech activity / presence probability [0.0f, 1.0f] */
float NeuralRESANS_GetSpeechProb(const NeuralRESANS *engine);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_RES_ANS_H */
