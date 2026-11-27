#include "transient_crossover.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void tc_biquad_init_lp(TCBiquad *bq, float sample_rate, float cutoff_hz, float q) {
    memset(bq, 0, sizeof(*bq));
    double w0 = 2.0 * M_PI * (double)cutoff_hz / (double)sample_rate;
    double cos_w0 = cos(w0);
    double sin_w0 = sin(w0);
    double alpha = sin_w0 / (2.0 * (double)q);

    double b0 = (1.0 - cos_w0) * 0.5;
    double b1 = 1.0 - cos_w0;
    double b2 = (1.0 - cos_w0) * 0.5;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cos_w0;
    double a2 = 1.0 - alpha;

    bq->b0 = (float)(b0 / a0);
    bq->b1 = (float)(b1 / a0);
    bq->b2 = (float)(b2 / a0);
    bq->a1 = (float)(a1 / a0);
    bq->a2 = (float)(a2 / a0);
}

static inline float tc_biquad_step(TCBiquad *bq, float x) {
    /* Transposed Direct Form II for numerical stability */
    float y = bq->b0 * x + bq->z1;
    bq->z1 = bq->b1 * x - bq->a1 * y + bq->z2;
    bq->z2 = bq->b2 * x - bq->a2 * y;
    return y;
}

void TransientCrossover_Init(TransientCrossover *tc, float sample_rate,
                             float crossover_hz, float crest_thresh,
                             float peak_thresh) {
    memset(tc, 0, sizeof(*tc));
    /* Cascaded 2nd-order Butterworth sections produce a 4th-order (-24 dB/oct) split */
    tc_biquad_init_lp(&tc->lp_stage1, sample_rate, crossover_hz, 0.70710678f);
    tc_biquad_init_lp(&tc->lp_stage2, sample_rate, crossover_hz, 0.70710678f);
    tc->hp_gain = 1.0f;
    tc->min_hp_gain = 0.05f;
    tc->tanh_limit = 0.15f;
    tc->attack_alpha = 0.06f;
    tc->release_alpha = 0.01f;
    tc->crest_thresh = crest_thresh;
    tc->peak_thresh = peak_thresh;
    tc->hold_frames = 0;
    tc->hold_duration = 5;
}

void TransientCrossover_Reset(TransientCrossover *tc) {
    tc->lp_stage1.z1 = tc->lp_stage1.z2 = 0.0f;
    tc->lp_stage2.z1 = tc->lp_stage2.z2 = 0.0f;
    tc->hp_gain = 1.0f;
    tc->hold_frames = 0;
}

bool TransientCrossover_ProcessFloat(TransientCrossover *tc, float *pcm,
                                     int num_samples) {
    if (!tc || !pcm || num_samples <= 0) return false;

    float peak = 0.0f;
    double sum_sq = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        float a = fabsf(pcm[i]);
        if (a > peak) peak = a;
        sum_sq += (double)pcm[i] * (double)pcm[i];
    }
    float rms = sqrtf((float)(sum_sq / num_samples)) + 1e-6f;
    float crest = peak / rms;

    bool triggered = (crest > tc->crest_thresh && peak > tc->peak_thresh);
    if (triggered) {
        tc->hold_frames = tc->hold_duration;
    } else if (tc->hold_frames > 0) {
        tc->hold_frames--;
    }

    bool active = (triggered || tc->hold_frames > 0);
    float target_gain = active ? tc->min_hp_gain : 1.0f;
    float alpha = active ? tc->attack_alpha : tc->release_alpha;
    float limit = tc->tanh_limit;

    for (int i = 0; i < num_samples; ++i) {
        float x = pcm[i];
        float lp = tc_biquad_step(&tc->lp_stage2, tc_biquad_step(&tc->lp_stage1, x));
        float hp = x - lp;

        tc->hp_gain += alpha * (target_gain - tc->hp_gain);
        float y = lp + tc->hp_gain * hp;

        /* Soft limiter only while attenuating/recovering so clean speech stays linear */
        if (active || tc->hp_gain < 0.99f) {
            y = limit * tanhf(y / limit);
        }
        pcm[i] = y;
    }

    return triggered;
}

bool TransientCrossover_ProcessInt16(TransientCrossover *tc, int16_t *pcm,
                                     int num_samples) {
    if (!tc || !pcm || num_samples <= 0) return false;

    float buf[512];
    int offset = 0;
    bool any_triggered = false;

    while (offset < num_samples) {
        int chunk = num_samples - offset;
        if (chunk > 512) chunk = 512;
        for (int i = 0; i < chunk; ++i) {
            buf[i] = (float)pcm[offset + i] / 32768.0f;
        }
        if (TransientCrossover_ProcessFloat(tc, buf, chunk)) {
            any_triggered = true;
        }
        for (int i = 0; i < chunk; ++i) {
            float v = buf[i] * 32767.0f;
            if (v > 32767.0f) v = 32767.0f;
            else if (v < -32768.0f) v = -32768.0f;
            pcm[offset + i] = (int16_t)v;
        }
        offset += chunk;
    }
    return any_triggered;
}

