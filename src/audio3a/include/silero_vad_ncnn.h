#ifndef SILERO_VAD_NCNN_H
#define SILERO_VAD_NCNN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SileroVadEngine SileroVadEngine;

typedef struct {
    const char *param_path;        /* path to silero.ncnn.param */
    const char *bin_path;          /* path to silero.ncnn.bin */
    float threshold;               /* Speech threshold: default 0.50f */
    float min_silence_duration;    /* in seconds: default 0.25f */
    float min_speech_duration;     /* in seconds: default 0.10f */
    int sample_rate;               /* 16000 */
    int hangover_frames;           /* hangover frames, e.g. 12 = 192ms */
} SileroVadConfig;

void  SileroVad_GetDefaultConfig(SileroVadConfig *config);
int   SileroVad_Init(SileroVadEngine **engine, const SileroVadConfig *config);
int   SileroVad_Process(SileroVadEngine *engine, const int16_t *audio_frame, int num_samples);
float SileroVad_GetConfidence(const SileroVadEngine *engine);
void  SileroVad_Reset(SileroVadEngine *engine);
void  SileroVad_Destroy(SileroVadEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* SILERO_VAD_NCNN_H */
