#ifndef HYBRID_3A_H
#define HYBRID_3A_H

#include "Audio3A.h"
#include "channel_reducer.h"
#include "linear_aec.h"
#include "neural_res_ans.h"
#include "adaptive_agc.h"
#include "vad_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HYBRID_3A_RING_SAMPLES 16384 /* 1024ms @ 16kHz */

typedef struct {
    int16_t ref_ring[HYBRID_3A_RING_SAMPLES];
    int     write_idx;
    int     total_pushed;
    int     estimated_delay_samples;
    bool    delay_locked;
    int     check_counter;
    int16_t mic_ds_ring[512]; /* 512 downsampled samples @ 4kHz = 128ms mic history */
    int     mic_ds_idx;
} Hybrid3ADelayEstimator;

/*
 * Unified Hybrid 3A Full Engine State
 */
struct Audio3A {
    Audio3AConfig          config;
    ChannelReducer         reducer;
    LinearAEC              aec;
    NeuralRESANS           neural_res_ans;
    AdaptiveAGC            agc;
    VADEngine              vad;
    Hybrid3ADelayEstimator delay_estimator;

    /* Frame buffers (Zero-allocation) */
    int16_t              fused_mic[DEFAULT_FRAME_SIZE];
    int16_t              ref_raw[DEFAULT_FRAME_SIZE];
    int16_t              ref_delayed[DEFAULT_FRAME_SIZE];
    int16_t              aec_error[DEFAULT_FRAME_SIZE];
    int16_t              echo_est[DEFAULT_FRAME_SIZE];
    int16_t              res_ans_clean[DEFAULT_FRAME_SIZE];
    int16_t              agc_out[DEFAULT_FRAME_SIZE];

    /* Statistics */
    uint32_t             total_frames_processed;
    int                  dt_onset_counter;
    double               ref_reverb_pwr;
    float                last_erle_db;
    float                last_gain_factor;
    int                  last_vad_decision;
    float                last_vad_confidence;
};

/*
 * Process a single 8-channel input block offline or directly from buffer
 * @param engine: Initialized Audio3A handle
 * @param in_8ch: 8-channel interleaved int16_t PCM (256 * 8 samples)
 * @param out_stereo: 2-channel interleaved int16_t PCM output (256 * 2 samples)
 * @param mute: 1 to mute microphone, 0 for normal audio
 */
int Hybrid3A_ProcessFrame(
    Audio3A *engine,
    const int16_t *in_8ch,
    int16_t *out_stereo,
    int mute
);

#ifdef __cplusplus
}
#endif

#endif /* HYBRID_3A_H */
