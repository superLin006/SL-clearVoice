#ifndef TRANSIENT_CROSSOVER_H
#define TRANSIENT_CROSSOVER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float b0, b1, b2, a1, a2;
    float z1, z2;
} TCBiquad;

typedef struct {
    TCBiquad lp_stage1;
    TCBiquad lp_stage2;
    float    hp_gain;        /* Smoothed gain applied to >crossover high-band */
    float    min_hp_gain;    /* Target high-band gain during transient hold (e.g. 0.05f = -26 dB) */
    float    tanh_limit;     /* Soft limiter ceiling during transient hold (e.g. 0.15f) */
    float    attack_alpha;   /* Per-sample smoothing rate toward min_hp_gain */
    float    release_alpha;  /* Per-sample smoothing rate back toward 1.0f */
    float    crest_thresh;   /* Frame crest factor threshold for transient hit */
    float    peak_thresh;    /* Normalized peak [0, 1] threshold for transient hit */
    int      hold_frames;    /* Countdown of active hold frames */
    int      hold_duration;  /* Hold duration in frames after trigger (e.g. 5 = 80ms @ 16ms) */
} TransientCrossover;

/**
 * Initialize a 4th-order Butterworth complementary crossover transient suppressor.
 * Splits audio at crossover_hz: preserves <crossover_hz harmonics while attenuating
 * >crossover_hz transient energy when impulsive frames are detected.
 */
void TransientCrossover_Init(TransientCrossover *tc, float sample_rate,
                             float crossover_hz, float crest_thresh,
                             float peak_thresh);

/**
 * Reset filter and envelope state.
 */
void TransientCrossover_Reset(TransientCrossover *tc);

/**
 * Evaluate a float frame and apply harmonic-preserving transient suppression in-place.
 * Returns true if the current frame triggered a new transient hit.
 */
bool TransientCrossover_ProcessFloat(TransientCrossover *tc, float *pcm,
                                     int num_samples);

/**
 * Evaluate an int16 PCM frame and apply harmonic-preserving transient suppression in-place.
 * Returns true if the current frame triggered a new transient hit.
 */
bool TransientCrossover_ProcessInt16(TransientCrossover *tc, int16_t *pcm,
                                     int num_samples);

#ifdef __cplusplus
}
#endif

#endif /* TRANSIENT_CROSSOVER_H */

