#include "audio_buffer.h"
#include <string.h>
#include <math.h>

void AudioFrame_Init(AudioFrame *frame, int channels, int frame_size) {
    if (!frame) return;
    frame->channels = (channels > MAX_AUDIO_CHANNELS) ? MAX_AUDIO_CHANNELS : channels;
    frame->frame_size = (frame_size > DEFAULT_FRAME_SIZE) ? DEFAULT_FRAME_SIZE : frame_size;
    memset(frame->channel_data, 0, sizeof(frame->channel_data));
}

void AudioFrame_FromInterleavedInt16(AudioFrame *frame, const int16_t *interleaved, int channels, int frame_size) {
    if (!frame || !interleaved) return;
    int ch_count = (channels < frame->channels) ? channels : frame->channels;
    int samples = (frame_size < frame->frame_size) ? frame_size : frame->frame_size;
    const float scale = 1.0f / 32768.0f;

    for (int i = 0; i < samples; ++i) {
        int base = i * channels;
        for (int ch = 0; ch < ch_count; ++ch) {
            frame->channel_data[ch][i] = ((float)interleaved[base + ch]) * scale;
        }
    }
}

static inline int16_t float_to_int16(float val) {
    float scaled = val * 32767.0f;
    if (scaled > 32767.0f) return 32767;
    if (scaled < -32768.0f) return -32768;
    return (int16_t)roundf(scaled);
}

void AudioFrame_ToInterleavedInt16(const AudioFrame *frame, int16_t *interleaved, int channels, int frame_size) {
    if (!frame || !interleaved) return;
    int ch_count = (channels < frame->channels) ? channels : frame->channels;
    int samples = (frame_size < frame->frame_size) ? frame_size : frame->frame_size;

    for (int i = 0; i < samples; ++i) {
        int base = i * channels;
        for (int ch = 0; ch < ch_count; ++ch) {
            interleaved[base + ch] = float_to_int16(frame->channel_data[ch][i]);
        }
    }
}

void AudioFrame_ChannelToInt16(const float *src, int16_t *dst, int frame_size) {
    if (!src || !dst) return;
    for (int i = 0; i < frame_size; ++i) {
        dst[i] = float_to_int16(src[i]);
    }
}

void AudioRingBuffer_Init(AudioRingBuffer *rb, int channels) {
    if (!rb) return;
    rb->channels = (channels > MAX_AUDIO_CHANNELS) ? MAX_AUDIO_CHANNELS : channels;
    rb->read_idx = 0;
    rb->write_idx = 0;
    rb->count = 0;
    memset(rb->buffer, 0, sizeof(rb->buffer));
}

int AudioRingBuffer_AvailableRead(const AudioRingBuffer *rb) {
    return rb ? rb->count : 0;
}

int AudioRingBuffer_AvailableWrite(const AudioRingBuffer *rb) {
    return rb ? (RING_BUFFER_CAPACITY - rb->count) : 0;
}

void AudioRingBuffer_Reset(AudioRingBuffer *rb) {
    if (!rb) return;
    rb->read_idx = 0;
    rb->write_idx = 0;
    rb->count = 0;
}
