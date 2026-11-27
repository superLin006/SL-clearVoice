#include "neural_res_ans.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void NeuralRESANS_GetDefaultConfig(NeuralConfig *config) {
    if (!config) return;
    config->min_gain_linear = 0.00001f;
    config->res_aggressiveness = 1.35f;
    config->ans_aggressiveness = 1.0f;
    config->enable_neural_ans = true;
    config->enable_neural_res = true;
}

int NeuralRESANS_Init(NeuralRESANS *engine, const NeuralConfig *config) {
    if (!engine) return -1;
    if (config) {
        engine->config = *config;
    } else {
        NeuralRESANS_GetDefaultConfig(&engine->config);
    }

    BarkFeatures_Init(&engine->feat_extractor);
    if (FastFFT_Init(&engine->fft, DEFAULT_FRAME_SIZE * 2) != 0) {
        return -1;
    }

    const int N2 = DEFAULT_FRAME_SIZE * 2;
    for (int n = 0; n < N2; ++n) {
        engine->wola_win[n] = sinf((float)M_PI * ((float)n + 0.5f) / (float)N2);
    }

    engine->rnnoise_state = rnnoise_create_v1();
    if (!engine->rnnoise_state) {
        fprintf(stderr, "[NeuralRESANS] Warning: Failed to create RNNoise AI instance\n");
    }

    NeuralRESANS_Reset(engine);
    return 0;
}

void NeuralRESANS_Reset(NeuralRESANS *engine) {
    if (!engine) return;
    memset(engine->gru_state, 0, sizeof(engine->gru_state));
    memset(engine->prev_error_time, 0, sizeof(engine->prev_error_time));
    memset(engine->prev_echo_time, 0, sizeof(engine->prev_echo_time));
    memset(engine->prev_ref_time, 0, sizeof(engine->prev_ref_time));
    memset(engine->synthesis_buf, 0, sizeof(engine->synthesis_buf));
    for (int b = 0; b < NUM_BARK_BANDS; ++b) {
        engine->last_bark_gains[b] = 1.0f;
        engine->bark_noise_floor[b] = 1e-4f;
    }
    for (int k = 0; k < NUM_FFT_BINS; ++k) {
        engine->echo_reverb_psd[k] = 0.0f;
        engine->noise_psd[k] = 1e-5f;
        engine->echo_coupling[k] = 0.10f;
        engine->beta_res[k] = 0.35f;
        engine->last_fft_gains[k] = 0.001f;
    }
    engine->speech_probability = 0.0f;
    engine->vad_speech_active = 0;
    engine->hangover_cnt = 0;
    engine->onset_cnt = 0;

    engine->rnnoise_fifo_in_count = 0;
    engine->rnnoise_fifo_out_count = 160;
    memset(engine->rnnoise_fifo_in, 0, sizeof(engine->rnnoise_fifo_in));
    memset(engine->rnnoise_fifo_out, 0, sizeof(engine->rnnoise_fifo_out));
}

void NeuralRESANS_Destroy(NeuralRESANS *engine) {
    if (!engine) return;
    if (engine->rnnoise_state) {
        rnnoise_destroy_v1(engine->rnnoise_state);
        engine->rnnoise_state = NULL;
    }
}

void NeuralRESANS_SetVAD(NeuralRESANS *engine, int is_speech) {
    if (engine) engine->vad_speech_active = is_speech ? 1 : 0;
}

void NeuralRESANS_ProcessFloat(
    NeuralRESANS *engine,
    const float *error_in,
    const float *echo_est_in,
    const float *ref_in,
    float *clean_out
) {
    if (!engine || !error_in || !clean_out) return;

    const int N = DEFAULT_FRAME_SIZE;

    if (engine->config.enable_neural_res && (echo_est_in || ref_in)) {
        float time_err[DEFAULT_FRAME_SIZE * 2];
        float time_echo[DEFAULT_FRAME_SIZE * 2];
        float time_ref[DEFAULT_FRAME_SIZE * 2];

        for (int i = 0; i < N; ++i) {
            time_err[i]      = engine->prev_error_time[i] * engine->wola_win[i];
            time_err[N + i]  = error_in[i]                * engine->wola_win[N + i];
            time_echo[i]     = echo_est_in ? (engine->prev_echo_time[i] * engine->wola_win[i]) : 0.0f;
            time_echo[N + i] = echo_est_in ? (echo_est_in[i]            * engine->wola_win[N + i]) : 0.0f;
            time_ref[i]      = ref_in ? (engine->prev_ref_time[i] * engine->wola_win[i]) : 0.0f;
            time_ref[N + i]  = ref_in ? (ref_in[i]                * engine->wola_win[N + i]) : 0.0f;
        }
        memcpy(engine->prev_error_time, error_in, N * sizeof(float));
        if (echo_est_in) memcpy(engine->prev_echo_time, echo_est_in, N * sizeof(float));
        if (ref_in) memcpy(engine->prev_ref_time, ref_in, N * sizeof(float));

        Complex32 E_freq[NUM_FFT_BINS];
        Complex32 D_freq[NUM_FFT_BINS];
        Complex32 R_freq[NUM_FFT_BINS];
        FastFFT_RealForward(&engine->fft, time_err, E_freq);
        FastFFT_RealForward(&engine->fft, time_echo, D_freq);
        FastFFT_RealForward(&engine->fft, time_ref, R_freq);

        float P_e[NUM_FFT_BINS];
        float P_d[NUM_FFT_BINS];
        float P_r[NUM_FFT_BINS];
        float e_sum = 0.0f;
        float r_sum = 0.0f;

        for (int k = 0; k < NUM_FFT_BINS; ++k) {
            float er = E_freq[k].r, ei = E_freq[k].i;
            float dr = D_freq[k].r, di = D_freq[k].i;
            float rr = R_freq[k].r, ri = R_freq[k].i;
            float pe = er * er + ei * ei;
            float pd = dr * dr + di * di;
            float pr = rr * rr + ri * ri;
            P_e[k] = pe;
            P_d[k] = pd;
            P_r[k] = pr;

            e_sum += pe;
            r_sum += pr;

            /* Noise floor update */
            if (pe < engine->noise_psd[k]) {
                engine->noise_psd[k] = 0.85f * engine->noise_psd[k] + 0.15f * pe;
            } else {
                engine->noise_psd[k] = 0.995f * engine->noise_psd[k] + 0.005f * pe;
            }
        }
        float e_energy = e_sum / (float)NUM_FFT_BINS;
        float r_energy = r_sum / (float)NUM_FFT_BINS;

        /* Reverb decay on echo estimate */
        for (int k = 0; k < NUM_FFT_BINS; ++k) {
            float rev = 0.75f * engine->echo_reverb_psd[k];
            engine->echo_reverb_psd[k] = (P_d[k] > rev) ? P_d[k] : rev;
        }

        Complex32 Y_freq[NUM_FFT_BINS];

        for (int k = 0; k < NUM_FFT_BINS; ++k) {
            /* True joint residual echo + ambient noise PSD modeling:
             * Linear AEC residual ratio is ~0.15 of P_d (0.28f provides margin).
             * Spectral ANS factor applies over-subtraction on tracked noise_psd.
             */
            float p_res = 0.28f * engine->echo_reverb_psd[k] + 1e-6f;
            float p_interf = p_res;
            if (engine->config.enable_neural_ans) {
                p_interf += engine->config.ans_aggressiveness * engine->noise_psd[k];
            }

            float target_g;
            if (!engine->vad_speech_active) {
                target_g = 0.0005f; /* -66 dB deep suppression during single-talk */
            } else {
                float s_near = P_e[k] - 1.0f * p_interf;
                if (s_near < 0.0f) s_near = 0.0f;
                float snr = s_near / p_interf;
                target_g = snr / (1.0f + snr + 1e-6f);
                float floor_g = (engine->config.min_gain_linear > 0.0001f) ? engine->config.min_gain_linear : 0.01f;
                if (target_g < floor_g) target_g = floor_g;
                if (target_g > 1.0f)  target_g = 1.0f;
            }

            float alpha_g = engine->vad_speech_active ? 0.45f : 0.80f;
            engine->last_fft_gains[k] = alpha_g * engine->last_fft_gains[k] + (1.0f - alpha_g) * target_g;
            Y_freq[k].r = E_freq[k].r * engine->last_fft_gains[k];
            Y_freq[k].i = E_freq[k].i * engine->last_fft_gains[k];
        }

        float time_out[DEFAULT_FRAME_SIZE * 2];
        FastFFT_RealInverse(&engine->fft, Y_freq, time_out);

        for (int i = 0; i < N; ++i) {
            clean_out[i] = engine->synthesis_buf[i] + time_out[i] * engine->wola_win[i];
            engine->synthesis_buf[i] = time_out[N + i] * engine->wola_win[N + i];
        }
    } else {
        memcpy(clean_out, error_in, N * sizeof(float));
    }

    /* Stage 3.5: Cascaded Deep AI ANS (RNNoise Recurrent Neural Denoising) */
    if (engine->config.enable_neural_ans && engine->rnnoise_state) {
        const int RNN_FRAME = 160;
        /* Push 256 samples into input FIFO */
        if (engine->rnnoise_fifo_in_count + N <= 512) {
            memcpy(engine->rnnoise_fifo_in + engine->rnnoise_fifo_in_count, clean_out, N * sizeof(float));
            engine->rnnoise_fifo_in_count += N;
        }

        /* Process all full 160-sample RNNoise frames */
        while (engine->rnnoise_fifo_in_count >= RNN_FRAME) {
            float rnn_in[160];
            float rnn_out[160];
            for (int i = 0; i < RNN_FRAME; ++i) {
                rnn_in[i] = engine->rnnoise_fifo_in[i] * 32767.0f;
            }
            memmove(engine->rnnoise_fifo_in, engine->rnnoise_fifo_in + RNN_FRAME,
                    (engine->rnnoise_fifo_in_count - RNN_FRAME) * sizeof(float));
            engine->rnnoise_fifo_in_count -= RNN_FRAME;

            rnnoise_process_frame_v1(engine->rnnoise_state, rnn_out, rnn_in);

            if (engine->rnnoise_fifo_out_count + RNN_FRAME <= 512) {
                for (int i = 0; i < RNN_FRAME; ++i) {
                    engine->rnnoise_fifo_out[engine->rnnoise_fifo_out_count + i] = rnn_out[i] / 32767.0f;
                }
                engine->rnnoise_fifo_out_count += RNN_FRAME;
            }
        }

        /* Pull 256 samples from output FIFO */
        if (engine->rnnoise_fifo_out_count >= N) {
            for (int i = 0; i < N; ++i) {
                clean_out[i] = engine->rnnoise_fifo_out[i];
            }
            memmove(engine->rnnoise_fifo_out, engine->rnnoise_fifo_out + N,
                    (engine->rnnoise_fifo_out_count - N) * sizeof(float));
            engine->rnnoise_fifo_out_count -= N;
        }
    }
}

static inline int16_t sat_int16(float val) {
    float scaled = val * 32767.0f;
    if (scaled > 32767.0f) return 32767;
    if (scaled < -32768.0f) return -32768;
    return (int16_t)roundf(scaled);
}

void NeuralRESANS_ProcessInt16(
    NeuralRESANS *engine,
    const int16_t *error_in,
    const int16_t *echo_est_in,
    const int16_t *ref_in,
    int16_t *clean_out
) {
    if (!engine || !error_in || !clean_out) return;

    static float f_err[DEFAULT_FRAME_SIZE];
    static float f_echo[DEFAULT_FRAME_SIZE];
    static float f_ref[DEFAULT_FRAME_SIZE];
    static float f_out[DEFAULT_FRAME_SIZE];

    const float scale = 1.0f / 32768.0f;
    for (int i = 0; i < DEFAULT_FRAME_SIZE; ++i) {
        f_err[i] = (float)error_in[i] * scale;
        f_echo[i] = echo_est_in ? ((float)echo_est_in[i] * scale) : 0.0f;
        f_ref[i] = ref_in ? ((float)ref_in[i] * scale) : 0.0f;
    }

    NeuralRESANS_ProcessFloat(engine, f_err, echo_est_in ? f_echo : NULL, ref_in ? f_ref : NULL, f_out);

    for (int i = 0; i < DEFAULT_FRAME_SIZE; ++i) {
        clean_out[i] = sat_int16(f_out[i]);
    }
}

float NeuralRESANS_GetSpeechProb(const NeuralRESANS *engine) {
    return engine ? engine->speech_probability : 0.0f;
}
