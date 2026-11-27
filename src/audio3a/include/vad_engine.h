#ifndef VAD_ENGINE_H
#define VAD_ENGINE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEFAULT_VAD_MODE             3    /* 3 = Very Aggressive for conference room noise rejection */
#define DEFAULT_VAD_HANGOVER_FRAMES  25   /* 25 * 16ms = 400ms smooth tail hangover */

typedef struct {
    int  mode;              /* 0: Normal, 1: Low bitrate, 2: Aggressive, 3: Very Aggressive */
    int  sample_rate;       /* Default: 16000 Hz */
    int  hangover_frames;   /* Default: 12 frames (192ms) */
} VADEngineConfig;

typedef struct {
    void            *vad_inst;           /* WebRtc VadInst handle */
    void            *silero_inst;        /* SileroVadEngine handle (sherpa-ncnn) */
    VADEngineConfig config;
    int             hangover_counter;    /* Remaining hangover frames */
    int             raw_decision;        /* 1 = speech, 0 = silence (current frame) */
    int             smoothed_decision;   /* 1 = speech, 0 = silence (with hangover) */
    float           speech_confidence;   /* 0.0 ~ 1.0 smooth confidence */
} VADEngine;

void VADEngine_GetDefaultConfig(VADEngineConfig *config);
int  VADEngine_Init(VADEngine *engine, const VADEngineConfig *config);
void VADEngine_Reset(VADEngine *engine);
void VADEngine_Destroy(VADEngine *engine);

/*
 * Process a 16kHz audio frame (typically 256 samples / 16ms).
 * Returns: 1 if active speech (including hangover protection), 0 if silence/noise.
 */
int  VADEngine_Process(VADEngine *engine, const int16_t *audio_frame, int num_samples);

int   VADEngine_GetRawDecision(const VADEngine *engine);
int   VADEngine_GetSmoothedDecision(const VADEngine *engine);
float VADEngine_GetConfidence(const VADEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* VAD_ENGINE_H */
