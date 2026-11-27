#include "vad_engine.h"
#include "webrtc_vad/webrtc_vad.h"
#ifdef USE_SILERO_NCNN
#include "silero_vad_ncnn.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void VADEngine_GetDefaultConfig(VADEngineConfig *config) {
    if (!config) return;
    config->mode = DEFAULT_VAD_MODE;                     /* Mode 3: Very aggressive */
    config->sample_rate = 16000;
    config->hangover_frames = DEFAULT_VAD_HANGOVER_FRAMES; /* 12 frames = 192ms */
}

int VADEngine_Init(VADEngine *engine, const VADEngineConfig *config) {
    if (!engine) return -1;
    if (config) {
        engine->config = *config;
    } else {
        VADEngine_GetDefaultConfig(&engine->config);
    }

    engine->silero_inst = NULL;
    engine->vad_inst = NULL;

#ifdef USE_SILERO_NCNN
    /* 1. Attempt to initialize sherpa-ncnn Silero Deep Learning AI VAD */
    SileroVadConfig silero_cfg;
    SileroVad_GetDefaultConfig(&silero_cfg);
    silero_cfg.sample_rate = engine->config.sample_rate;
    silero_cfg.hangover_frames = engine->config.hangover_frames;

    SileroVadEngine *s_engine = NULL;
    if (SileroVad_Init(&s_engine, &silero_cfg) == 0 && s_engine) {
        engine->silero_inst = (void *)s_engine;
        printf("[VADEngine] Active Mode: sherpa-ncnn Silero Deep Neural VAD\n");
    } else {
        printf("[VADEngine] Active Mode: WebRTC VAD Mode %d (Fallback)\n", engine->config.mode);
    }
#endif

    /* 2. Initialize WebRTC VAD */
    VadInst *inst = WebRtcVad_Create();
    if (inst) {
        if (WebRtcVad_Init(inst) == 0 && WebRtcVad_set_mode(inst, engine->config.mode) == 0) {
            engine->vad_inst = (void *)inst;
        } else {
            WebRtcVad_Free(inst);
        }
    }

    VADEngine_Reset(engine);
    return 0;
}

void VADEngine_Reset(VADEngine *engine) {
    if (!engine) return;
#ifdef USE_SILERO_NCNN
    if (engine->silero_inst) {
        SileroVad_Reset((SileroVadEngine *)engine->silero_inst);
    }
#endif
    engine->hangover_counter = 0;
    engine->raw_decision = 0;
    engine->smoothed_decision = 0;
    engine->speech_confidence = 0.0f;
}

void VADEngine_Destroy(VADEngine *engine) {
    if (!engine) return;
#ifdef USE_SILERO_NCNN
    if (engine->silero_inst) {
        SileroVad_Destroy((SileroVadEngine *)engine->silero_inst);
        engine->silero_inst = NULL;
    }
#endif
    if (engine->vad_inst) {
        WebRtcVad_Free((VadInst *)engine->vad_inst);
        engine->vad_inst = NULL;
    }
}

int VADEngine_Process(VADEngine *engine, const int16_t *audio_frame, int num_samples) {
    if (!engine || !audio_frame || num_samples <= 0) return 0;

#ifdef USE_SILERO_NCNN
    /* Primary: sherpa-ncnn Silero Deep Learning VAD */
    if (engine->silero_inst) {
        int decision = SileroVad_Process((SileroVadEngine *)engine->silero_inst, audio_frame, num_samples);
        engine->raw_decision = decision;
        engine->smoothed_decision = decision;
        engine->speech_confidence = SileroVad_GetConfidence((SileroVadEngine *)engine->silero_inst);
        return decision;
    }
#endif

    /* Secondary: WebRTC VAD Fallback */
    if (!engine->vad_inst) return 0;
    VadInst *inst = (VadInst *)engine->vad_inst;
    int fs = engine->config.sample_rate;
    int raw = 0;

    /*
     * WebRTC VAD requires 10ms (160 samples), 20ms (320 samples), or 30ms (480 samples) at 16kHz.
     * For our 16ms (256 samples) frame, evaluate two overlapping 10ms (160 samples) slices:
     *   Slice 1: [0 .. 159]   (0ms to 10ms)
     *   Slice 2: [96 .. 255]  (6ms to 16ms)
     */
    if (num_samples >= 256 && fs == 16000) {
        int r1 = WebRtcVad_Process(inst, 16000, &audio_frame[0], 160);
        int r2 = WebRtcVad_Process(inst, 16000, &audio_frame[96], 160);
        raw = (r1 > 0 || r2 > 0) ? 1 : 0;
    } else if (num_samples >= 160 && fs == 16000) {
        int r = WebRtcVad_Process(inst, 16000, audio_frame, 160);
        raw = (r > 0) ? 1 : 0;
    } else {
        int r = WebRtcVad_Process(inst, fs, audio_frame, num_samples);
        raw = (r > 0) ? 1 : 0;
    }

    engine->raw_decision = raw;

    /* Hangover logic to protect speech word endings and plosives */
    if (raw == 1) {
        engine->hangover_counter = engine->config.hangover_frames;
        engine->smoothed_decision = 1;
    } else {
        if (engine->hangover_counter > 0) {
            engine->hangover_counter--;
            engine->smoothed_decision = 1;
        } else {
            engine->smoothed_decision = 0;
        }
    }

    /* Exponentially smoothed confidence */
    float target_conf = engine->smoothed_decision ? 1.0f : 0.0f;
    engine->speech_confidence = 0.80f * engine->speech_confidence + 0.20f * target_conf;

    return engine->smoothed_decision;
}

int VADEngine_GetRawDecision(const VADEngine *engine) {
    return engine ? engine->raw_decision : 0;
}

int VADEngine_GetSmoothedDecision(const VADEngine *engine) {
    return engine ? engine->smoothed_decision : 0;
}

float VADEngine_GetConfidence(const VADEngine *engine) {
    return engine ? engine->speech_confidence : 0.0f;
}
