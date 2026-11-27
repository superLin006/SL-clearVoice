#ifndef SYNTHETIC_GENERATOR_H
#define SYNTHETIC_GENERATOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Generates an 8-channel 16-bit 16kHz test WAV simulating:
 * - 6-mic circular array (R = 56mm)
 * - Near-end speaker at specified angle (e.g. 60 deg)
 * - Diffuse ambient noise
 * - Intermittent keyboard tapping (non-stationary noise)
 * - Far-end audio on Channel 6 with acoustic echo into the mics
 * - Channel 7 muted (0)
 * - Physical pin swapping (Ch0<->Ch1, Ch2<->Ch3) matching board hardware
 */
int GenerateSynthetic8ChWav(const char *output_wav_path, float duration_seconds);

#ifdef __cplusplus
}
#endif

#endif /* SYNTHETIC_GENERATOR_H */
