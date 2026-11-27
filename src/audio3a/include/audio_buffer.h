#ifndef AUDIO_BUFFER_H
#define AUDIO_BUFFER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_AUDIO_CHANNELS    8
#define DEFAULT_FRAME_SIZE    256   /* 16ms @ 16kHz */
#define DEFAULT_SAMPLE_RATE   16000

/*
 * Static Audio Frame Buffer (Zero-allocation)
 * Supports both interleaved int16_t (hardware I/O) and de-interleaved float (DSP processing).
 */
typedef struct {
    int channels;
    int frame_size;
    /* De-interleaved float buffers for numerical stability in DSP algorithms */
    float channel_data[MAX_AUDIO_CHANNELS][DEFAULT_FRAME_SIZE];
} AudioFrame;

/*
 * Static Audio Ring Buffer
 * Lock-free for single-producer single-consumer or bounded synchronous buffer
 */
#define RING_BUFFER_CAPACITY (DEFAULT_FRAME_SIZE * 8)

typedef struct {
    int16_t buffer[MAX_AUDIO_CHANNELS][RING_BUFFER_CAPACITY];
    int read_idx;
    int write_idx;
    int count;
    int channels;
} AudioRingBuffer;

/*
 * AudioFrame Functions
 */
void AudioFrame_Init(AudioFrame *frame, int channels, int frame_size);

/* Convert interleaved int16_t (e.g. from ALSA/TDM) to normalized float de-interleaved [-1.0f, 1.0f] */
void AudioFrame_FromInterleavedInt16(AudioFrame *frame, const int16_t *interleaved, int channels, int frame_size);

/* Convert de-interleaved float [-1.0f, 1.0f] back to interleaved int16_t with saturation clipping */
void AudioFrame_ToInterleavedInt16(const AudioFrame *frame, int16_t *interleaved, int channels, int frame_size);

/* Convert a single float channel to int16_t buffer */
void AudioFrame_ChannelToInt16(const float *src, int16_t *dst, int frame_size);

/*
 * RingBuffer Functions (Zero dynamic allocation)
 */
void AudioRingBuffer_Init(AudioRingBuffer *rb, int channels);
int  AudioRingBuffer_AvailableRead(const AudioRingBuffer *rb);
int  AudioRingBuffer_AvailableWrite(const AudioRingBuffer *rb);
int  AudioRingBuffer_Write(AudioRingBuffer *rb, const float *const *channels_in, int num_samples);
int  AudioRingBuffer_Read(AudioRingBuffer *rb, float *const *channels_out, int num_samples);
void AudioRingBuffer_Reset(AudioRingBuffer *rb);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_BUFFER_H */
