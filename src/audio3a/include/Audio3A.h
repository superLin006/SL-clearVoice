#ifndef AUDIO3A_H
#define AUDIO3A_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int   use_ans;          /* 1: Enable AI ANS noise reduction (RNNoise), 0: Disable */
    int   use_agc;          /* 1: Enable Adaptive AGC & Limiter, 0: Disable */
    int   use_aec;          /* 1: Enable Linear AEC + AI RES, 0: Disable */
    int   use_beamforming;  /* 1: Enable 6-mic beamforming/crossover, 0: Single-channel */
    int   use_usb;          /* 1: Enable USB output, 0: Line-out / ALSA */
    int   delay_frames;     /* Acoustic loopback delay in 16ms frames (default: 7 = 112ms) */
    float target_rms_dbfs;  /* Target AGC loudness level (e.g. -14.0f dBFS, default: -14.0f) */
} Audio3AConfig;

typedef struct Audio3A Audio3A;

/*
 * Create and initialize the unified 3A Engine
 * @param audio3a_st: Output handle pointer
 * @param config: Configuration options (pass NULL for default)
 * @return 0 on success, negative error code on failure
 */
int Audio3A_Init(Audio3A** audio3a_st, const Audio3AConfig* config);

/*
 * Standard streaming pipeline process (records from ALSA capture, processes, plays to ALSA)
 * @param audio3a_st: Engine handle
 * @param mute: 1 to mute microphone, 0 for normal audio
 * @return 0 on success
 */
int Audio3A_Process(Audio3A* audio3a_st, int mute);

/*
 * Buffer-level direct processing interface (Zero ALSA dependency, for VoIP/P2P/Offline integration)
 * @param audio3a_st: Engine handle
 * @param mic_pcm: Mic input PCM (int16_t, 256 samples mono or primary mic)
 * @param ref_pcm: Far-end speaker reference PCM (int16_t, 256 samples mono)
 * @param out_clean: Output cleaned speech PCM (int16_t, 256 samples mono)
 * @param num_samples: Frame sample count (typically 256)
 * @param mute: 1 to mute, 0 for normal
 * @return 0 on success
 */
int Audio3A_ProcessDirect(
    Audio3A* audio3a_st,
    const int16_t *mic_pcm,
    const int16_t *ref_pcm,
    int16_t *out_clean,
    int num_samples,
    int mute
);

/*
 * 8-channel interleaved frame processor (6 mics + 2 ref channels)
 * @param audio3a_st: Engine handle
 * @param in_8ch: 8-channel interleaved int16_t PCM (256 * 8 samples)
 * @param out_stereo: 2-channel interleaved int16_t PCM output (256 * 2 samples)
 * @param mute: 1 to mute, 0 for normal
 * @return 0 on success
 */
int Audio3A_ProcessInterleaved(
    Audio3A* audio3a_st,
    const int16_t *in_8ch,
    int16_t *out_stereo,
    int mute
);

/*
 * Get real-time engine metrics
 */
float Audio3A_GetERLE(const Audio3A* audio3a_st);
float Audio3A_GetAGCGain(const Audio3A* audio3a_st);
int   Audio3A_GetVAD(const Audio3A* audio3a_st);

/*
 * Free engine resources
 */
void Audio3A_Free(Audio3A* audio3a_st);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO3A_H */
