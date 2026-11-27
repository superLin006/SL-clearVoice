#ifndef WAV_FILE_H
#define WAV_FILE_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    FILE *fp;
    int channels;
    int sample_rate;
    int bits_per_sample;
    uint32_t total_samples_per_channel;
    uint32_t data_bytes;
} WavFile;

/* Open a WAV file for reading */
WavFile* WavFile_OpenRead(const char *filename);

/* Read interleaved 16-bit PCM samples (reads up to num_samples * channels values) */
int WavFile_ReadInt16(WavFile *wav, int16_t *buffer, int num_samples_per_channel);

/* Open a WAV file for writing (16-bit PCM) */
WavFile* WavFile_OpenWrite(const char *filename, int channels, int sample_rate);

/* Write interleaved 16-bit PCM samples */
int WavFile_WriteInt16(WavFile *wav, const int16_t *buffer, int num_samples_per_channel);

/* Close WAV file and update RIFF header sizes */
void WavFile_Close(WavFile *wav);

#ifdef __cplusplus
}
#endif

#endif /* WAV_FILE_H */
