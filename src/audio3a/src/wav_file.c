#include "wav_file.h"
#include <stdlib.h>
#include <string.h>

#pragma pack(push, 1)
typedef struct {
    char     riff[4];        /* "RIFF" */
    uint32_t overall_size;   /* File size - 8 */
    char     wave[4];        /* "WAVE" */
    char     fmt_chunk[4];   /* "fmt " */
    uint32_t length_format;  /* 16 for PCM */
    uint16_t format_type;    /* 1 for PCM */
    uint16_t channels;       /* 1, 2, 6, 8 */
    uint32_t sample_rate;    /* 16000 */
    uint32_t byte_rate;      /* sample_rate * channels * 2 */
    uint16_t block_align;    /* channels * 2 */
    uint16_t bits_per_sample;/* 16 */
    char     data_chunk[4];  /* "data" */
    uint32_t data_size;      /* size of PCM data */
} RiffHeader;
#pragma pack(pop)

WavFile* WavFile_OpenRead(const char *filename) {
    if (!filename) return NULL;
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;

    RiffHeader header;
    if (fread(&header, 1, sizeof(header), fp) != sizeof(header)) {
        fclose(fp);
        return NULL;
    }

    if (memcmp(header.riff, "RIFF", 4) != 0 || memcmp(header.wave, "WAVE", 4) != 0) {
        fclose(fp);
        return NULL;
    }

    /* Handle possible non-standard chunk ordering if needed, or standard PCM */
    WavFile *wav = (WavFile*)calloc(1, sizeof(WavFile));
    if (!wav) {
        fclose(fp);
        return NULL;
    }

    wav->fp = fp;
    wav->channels = header.channels;
    wav->sample_rate = header.sample_rate;
    wav->bits_per_sample = header.bits_per_sample;
    wav->data_bytes = header.data_size;
    int bytes_per_sample = (wav->bits_per_sample / 8) * wav->channels;
    wav->total_samples_per_channel = (bytes_per_sample > 0) ? (header.data_size / bytes_per_sample) : 0;
    return wav;
}

int WavFile_ReadInt16(WavFile *wav, int16_t *buffer, int num_samples_per_channel) {
    if (!wav || !wav->fp || !buffer) return 0;
    size_t total_items = (size_t)num_samples_per_channel * wav->channels;
    size_t read_items = fread(buffer, sizeof(int16_t), total_items, wav->fp);
    return (int)(read_items / wav->channels);
}

WavFile* WavFile_OpenWrite(const char *filename, int channels, int sample_rate) {
    if (!filename) return NULL;
    FILE *fp = fopen(filename, "wb");
    if (!fp) return NULL;

    WavFile *wav = (WavFile*)calloc(1, sizeof(WavFile));
    if (!wav) {
        fclose(fp);
        return NULL;
    }

    wav->fp = fp;
    wav->channels = channels;
    wav->sample_rate = sample_rate;
    wav->bits_per_sample = 16;
    wav->data_bytes = 0;
    wav->total_samples_per_channel = 0;

    /* Write placeholder header */
    RiffHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.riff, "RIFF", 4);
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt_chunk, "fmt ", 4);
    header.length_format = 16;
    header.format_type = 1;
    header.channels = (uint16_t)channels;
    header.sample_rate = (uint32_t)sample_rate;
    header.bits_per_sample = 16;
    header.byte_rate = (uint32_t)(sample_rate * channels * 2);
    header.block_align = (uint16_t)(channels * 2);
    memcpy(header.data_chunk, "data", 4);
    header.data_size = 0;
    header.overall_size = sizeof(RiffHeader) - 8;

    fwrite(&header, 1, sizeof(header), fp);
    return wav;
}

int WavFile_WriteInt16(WavFile *wav, const int16_t *buffer, int num_samples_per_channel) {
    if (!wav || !wav->fp || !buffer) return 0;
    size_t total_items = (size_t)num_samples_per_channel * wav->channels;
    size_t written = fwrite(buffer, sizeof(int16_t), total_items, wav->fp);
    wav->data_bytes += (uint32_t)(written * sizeof(int16_t));
    wav->total_samples_per_channel += (uint32_t)(written / wav->channels);
    return (int)(written / wav->channels);
}

void WavFile_Close(WavFile *wav) {
    if (!wav) return;
    if (wav->fp) {
        /* Rewind and fix sizes */
        fseek(wav->fp, 0, SEEK_SET);
        RiffHeader header;
        memset(&header, 0, sizeof(header));
        memcpy(header.riff, "RIFF", 4);
        header.overall_size = (uint32_t)(wav->data_bytes + sizeof(RiffHeader) - 8);
        memcpy(header.wave, "WAVE", 4);
        memcpy(header.fmt_chunk, "fmt ", 4);
        header.length_format = 16;
        header.format_type = 1;
        header.channels = (uint16_t)wav->channels;
        header.sample_rate = (uint32_t)wav->sample_rate;
        header.bits_per_sample = 16;
        header.byte_rate = (uint32_t)(wav->sample_rate * wav->channels * 2);
        header.block_align = (uint16_t)(wav->channels * 2);
        memcpy(header.data_chunk, "data", 4);
        header.data_size = wav->data_bytes;

        fwrite(&header, 1, sizeof(header), wav->fp);
        fclose(wav->fp);
        wav->fp = NULL;
    }
    free(wav);
}
