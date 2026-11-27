#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <fcntl.h>
#include "p2p_protocol.h"

static volatile bool g_running = true;

static void sig_handler(int sig) {
    (void)sig;
    g_running = false;
}

static uint32_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

/* Minimal WAV Header Writer */
static void write_wav_header(FILE *fp, uint32_t total_samples, uint32_t sample_rate, uint16_t channels) {
    uint32_t data_bytes = total_samples * channels * sizeof(int16_t);
    uint32_t riff_bytes = 36 + data_bytes;
    uint32_t byte_rate = sample_rate * channels * sizeof(int16_t);
    uint16_t block_align = channels * sizeof(int16_t);
    uint16_t bits_per_sample = 16;
    uint16_t audio_format = 1;

    fseek(fp, 0, SEEK_SET);
    fwrite("RIFF", 1, 4, fp);
    fwrite(&riff_bytes, 4, 1, fp);
    fwrite("WAVEfmt ", 1, 8, fp);
    uint32_t subchunk1_size = 16;
    fwrite(&subchunk1_size, 4, 1, fp);
    fwrite(&audio_format, 2, 1, fp);
    fwrite(&channels, 2, 1, fp);
    fwrite(&sample_rate, 4, 1, fp);
    fwrite(&byte_rate, 4, 1, fp);
    fwrite(&block_align, 2, 1, fp);
    fwrite(&bits_per_sample, 2, 1, fp);
    fwrite("data", 1, 4, fp);
    fwrite(&data_bytes, 4, 1, fp);
}

typedef struct {
    char target_ip[64];
    int  target_port;
    int  listen_port;
    char in_wav[256];
    char out_wav[256];
    int  duration_sec;      /* 0 = infinite loop / continuous */
    int  use_mic;           /* 1 = live mic capture via arecord, 0 = WAV file */
    char mic_device[64];    /* ALSA mic capture device (e.g. plughw:3,0 or plughw:4,0) */
    int  play_audio;        /* 1 = live playback via aplay */
    char play_device[64];   /* ALSA device for live audio output */
} CallerArgs;

/* Receiver Thread State */
static uint32_t g_recv_count = 0;
static double   g_recv_energy = 0.0;
static uint32_t g_recv_samples = 0;
static float    g_last_rtt_ms = 0.0f;
static float    g_last_recv_rms = 0.0f;

static void* recv_thread_func(void *arg) {
    CallerArgs *args = (CallerArgs*)arg;
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("[Caller] Recv socket failed");
        return NULL;
    }

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in bind_addr;
    memset(&bind_addr, 0, sizeof(bind_addr));
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = htons(args->listen_port);
    bind_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
        perror("[Caller] Bind recv port failed");
        close(sock);
        return NULL;
    }

    struct timeval tv = { .tv_sec = 0, .tv_usec = 100000 };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    FILE *out_fp = fopen(args->out_wav, "wb");
    if (!out_fp) {
        perror("[Caller] Open out WAV failed");
        close(sock);
        return NULL;
    }
    /* Write initial WAV header (placeholder) */
    write_wav_header(out_fp, 60 * P2P_SAMPLE_RATE, P2P_SAMPLE_RATE, 1);
    fflush(out_fp);

    /* Optional: Real-time Audio Playback via ALSA */
    FILE *play_pipe = NULL;
    if (args->play_audio) {
        char play_cmd[512];
        const char *dev = args->play_device[0] ? args->play_device : "plughw:3,0";
        snprintf(play_cmd, sizeof(play_cmd),
                 "aplay -D %s --period-size=256 --buffer-size=2048 -r 16000 -c 1 -f S16_LE -t raw 2>/dev/null",
                 dev);
        play_pipe = popen(play_cmd, "w");
        if (play_pipe) {
            printf("[Caller] Live audio output active on %s\n", dev);
        } else {
            fprintf(stderr, "[Caller] Warning: Failed to open live playback device: %s\n", dev);
        }
    }

    P2PPacket pkt;
    while (g_running) {
        ssize_t n = recvfrom(sock, &pkt, sizeof(pkt), 0, NULL, NULL);
        if (n <= 0) continue;

        if (pkt.magic != P2P_MAGIC) continue;

        uint32_t now = get_time_ms();
        if (now >= pkt.timestamp_ms) {
            g_last_rtt_ms = (float)(now - pkt.timestamp_ms);
        }

        double frame_e = 0.0;
        for (int i = 0; i < pkt.samples; ++i) {
            float s = pkt.pcm[i] / 32768.0f;
            frame_e += s * s;
        }
        g_last_recv_rms = sqrtf(frame_e / pkt.samples);
        g_recv_energy += frame_e;
        g_recv_samples += pkt.samples;
        g_recv_count++;

        /* 1. Save to clean audio recording file */
        fwrite(pkt.pcm, sizeof(int16_t), pkt.samples, out_fp);

        /* 2. Stream to ALSA speaker/headphone if enabled */
        if (play_pipe) {
            fwrite(pkt.pcm, sizeof(int16_t), pkt.samples, play_pipe);
            fflush(play_pipe);
        }
    }

    if (play_pipe) {
        pclose(play_pipe);
        play_pipe = NULL;
    }

    write_wav_header(out_fp, g_recv_samples, P2P_SAMPLE_RATE, 1);
    fclose(out_fp);
    close(sock);
    return NULL;
}

int main(int argc, char **argv) {
    CallerArgs args;
    memset(&args, 0, sizeof(args));
    strncpy(args.target_ip, P2P_DEFAULT_DUT_IP, sizeof(args.target_ip) - 1);
    args.target_port = P2P_PORT_TO_DUT;
    args.listen_port = P2P_PORT_FROM_DUT;
    strncpy(args.in_wav, "/root/duiji_16k.wav", sizeof(args.in_wav) - 1);
    strncpy(args.out_wav, "/tmp/p2p_returned_clean.wav", sizeof(args.out_wav) - 1);
    args.duration_sec = 0; /* Default: Continuous Infinite Loop */
    args.use_mic = 0;      /* Default: Use WAV file */
    strncpy(args.mic_device, "plughw:3,0", sizeof(args.mic_device) - 1);
    args.play_audio = 0;   /* Default: Off, can enable with --play */
    strncpy(args.play_device, "plughw:3,0", sizeof(args.play_device) - 1);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--target-ip") == 0 && i + 1 < argc) {
            strncpy(args.target_ip, argv[++i], sizeof(args.target_ip) - 1);
        } else if (strcmp(argv[i], "--wav-in") == 0 && i + 1 < argc) {
            strncpy(args.in_wav, argv[++i], sizeof(args.in_wav) - 1);
            args.use_mic = 0;
        } else if (strcmp(argv[i], "--wav-out") == 0 && i + 1 < argc) {
            strncpy(args.out_wav, argv[++i], sizeof(args.out_wav) - 1);
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            args.duration_sec = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--loop") == 0 || strcmp(argv[i], "--continuous") == 0) {
            args.duration_sec = 0;
        } else if (strcmp(argv[i], "--mic") == 0) {
            args.use_mic = 1;
        } else if (strcmp(argv[i], "--mic-device") == 0 && i + 1 < argc) {
            args.use_mic = 1;
            strncpy(args.mic_device, argv[++i], sizeof(args.mic_device) - 1);
        } else if (strcmp(argv[i], "--play") == 0) {
            args.play_audio = 1;
        } else if (strcmp(argv[i], "--play-device") == 0 && i + 1 < argc) {
            args.play_audio = 1;
            strncpy(args.play_device, argv[++i], sizeof(args.play_device) - 1);
        }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    printf("==============================================================\n");
    printf("  P2P Intercom Caller / Continuous Loop Tester\n");
    printf("  Target (DUT) IP:   %s:%d\n", args.target_ip, args.target_port);
    printf("  Listen Return:     0.0.0.0:%d\n", args.listen_port);
    if (args.use_mic) {
        printf("  Audio Input:       [LIVE MIC] (%s)\n", args.mic_device);
    } else {
        printf("  Audio Input:       [WAV FILE] (%s)\n", args.in_wav);
    }
    printf("  Saved Return File: %s\n", args.out_wav);
    if (args.duration_sec <= 0) {
        printf("  Test Duration:     Continuous / Infinite Loop (Press Ctrl+C to terminate)\n");
    } else {
        printf("  Test Duration:     %d seconds\n", args.duration_sec);
    }
    printf("  Live Audio Output: %s (%s)\n",
           args.play_audio ? "[ENABLED]" : "[DISABLED] (Pass --play to hear live)",
           args.play_device);
    printf("==============================================================\n\n");

    FILE *src_fp = NULL;
    bool is_mic_pipe = false;

    if (args.use_mic) {
        char mic_cmd[512];
        const char *mdev = args.mic_device[0] ? args.mic_device : "plughw:3,0";
        snprintf(mic_cmd, sizeof(mic_cmd),
                 "arecord -D %s --period-size=256 --buffer-size=2048 -r 16000 -c 1 -f S16_LE -t raw 2>/dev/null",
                 mdev);
        src_fp = popen(mic_cmd, "r");
        if (!src_fp) {
            fprintf(stderr, "[Error] Failed to open microphone capture: %s\n", mdev);
            return 1;
        }
        is_mic_pipe = true;
        printf("[Caller] Real-time microphone capture started on %s\n", mdev);
    } else {
        /* Open source audio file */
        src_fp = fopen(args.in_wav, "rb");
        if (!src_fp) {
            src_fp = fopen("/tmp/duiji_16k.wav", "rb");
        }
        if (!src_fp) {
            src_fp = fopen("/root/my_doubletalk_test.wav", "rb");
        }

        if (src_fp) {
            fseek(src_fp, 44, SEEK_SET); /* Skip WAV header */
        } else {
            /* Fallback: try sox */
            char pipe_cmd[512];
            snprintf(pipe_cmd, sizeof(pipe_cmd), "sox \"%s\" -r 16000 -c 1 -t raw - 2>/dev/null", args.in_wav);
            src_fp = popen(pipe_cmd, "r");
            if (src_fp) is_mic_pipe = true;
        }

        if (!src_fp) {
            fprintf(stderr, "[Error] Failed to open source audio: %s\n", args.in_wav);
            return 1;
        }
    }

    pthread_t recv_tid;
    pthread_create(&recv_tid, NULL, recv_thread_func, &args);

    int send_sock = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in target_addr;
    memset(&target_addr, 0, sizeof(target_addr));
    target_addr.sin_family = AF_INET;
    target_addr.sin_port = htons(args.target_port);
    inet_pton(AF_INET, args.target_ip, &target_addr.sin_addr);

    uint32_t start_time = get_time_ms();
    uint32_t seq = 0;
    P2PPacket pkt;
    pkt.magic = P2P_MAGIC;
    pkt.samples = P2P_FRAME_SAMPLES;
    pkt.flags = 0;

    struct timespec frame_interval = { .tv_sec = 0, .tv_nsec = 16000000 }; /* 16ms */
    struct timespec next_time;
    clock_gettime(CLOCK_MONOTONIC, &next_time);

    double total_sent_energy = 0.0;
    uint32_t total_sent_samples = 0;

    while (g_running) {
        if (args.duration_sec > 0) {
            uint32_t elapsed = (get_time_ms() - start_time) / 1000;
            if (elapsed >= (uint32_t)args.duration_sec) break;
        }

        if (is_mic_pipe) {
            /* Read live microphone stream */
            size_t nread = fread(pkt.pcm, sizeof(int16_t), P2P_FRAME_SAMPLES, src_fp);
            if (nread < P2P_FRAME_SAMPLES) {
                if (feof(src_fp) || ferror(src_fp)) break;
                memset(&pkt.pcm[nread], 0, (P2P_FRAME_SAMPLES - nread) * sizeof(int16_t));
            }
        } else {
            /* Seamless WAV file looping */
            size_t nread = fread(pkt.pcm, sizeof(int16_t), P2P_FRAME_SAMPLES, src_fp);
            if (nread < P2P_FRAME_SAMPLES) {
                fseek(src_fp, 44, SEEK_SET);
                size_t n2 = fread(&pkt.pcm[nread], sizeof(int16_t), P2P_FRAME_SAMPLES - nread, src_fp);
                if (nread + n2 < P2P_FRAME_SAMPLES) {
                    memset(&pkt.pcm[nread + n2], 0, (P2P_FRAME_SAMPLES - nread - n2) * sizeof(int16_t));
                }
            }
        }

        pkt.seq = ++seq;
        pkt.timestamp_ms = get_time_ms();

        double frame_e = 0.0;
        for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
            float s = pkt.pcm[i] / 32768.0f;
            frame_e += s * s;
        }
        total_sent_energy += frame_e;
        total_sent_samples += P2P_FRAME_SAMPLES;
        float sent_rms = sqrtf(frame_e / P2P_FRAME_SAMPLES);

        sendto(send_sock, &pkt, sizeof(pkt), 0, (struct sockaddr*)&target_addr, sizeof(target_addr));

        if (seq % 50 == 0) {
            float sent_dbfs = 20.0f * log10f(sent_rms + 1e-6f);
            float recv_dbfs = 20.0f * log10f(g_last_recv_rms + 1e-6f);
            float instant_erle = sent_dbfs - recv_dbfs;

            printf("[P2P Stats] Seq:%-5u | Sent: %5.1f dBFS | Recv: %5.1f dBFS | Return Suppression: %5.1f dB | RTT: %4.1f ms\n",
                   seq, sent_dbfs, recv_dbfs, instant_erle, g_last_rtt_ms);
            fflush(stdout);
        }

        if (!is_mic_pipe) {
            next_time.tv_nsec += frame_interval.tv_nsec;
            if (next_time.tv_nsec >= 1000000000) {
                next_time.tv_sec += 1;
                next_time.tv_nsec -= 1000000000;
            }
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_time, NULL);
        }
    }

    g_running = false;
    printf("\n[Caller] Stopping transmission, waiting for receiver...\n");
    pthread_join(recv_tid, NULL);

    close(send_sock);
    if (src_fp) {
        if (is_mic_pipe) pclose(src_fp);
        else fclose(src_fp);
    }

    double overall_sent_rms = sqrt(total_sent_energy / (total_sent_samples + 1e-9));
    double overall_recv_rms = sqrt(g_recv_energy / (g_recv_samples + 1e-9));
    double overall_erle = 20.0 * log10((overall_sent_rms + 1e-6) / (overall_recv_rms + 1e-6));

    printf("\n==================== P2P Test Summary ====================\n");
    printf("  Packets Sent:      %u\n", seq);
    printf("  Packets Received:  %u (Loss: %.1f%%)\n",
           g_recv_count, 100.0f * (seq > g_recv_count ? (seq - g_recv_count) : 0) / (seq + 1e-6f));
    printf("  Sent Audio RMS:    %.4f (%.1f dBFS)\n", overall_sent_rms, 20.0*log10(overall_sent_rms+1e-6));
    printf("  Recv Audio RMS:    %.4f (%.1f dBFS)\n", overall_recv_rms, 20.0*log10(overall_recv_rms+1e-6));
    printf("  Overall Suppression: %.2f dB\n", overall_erle);
    printf("  Recorded Audio:    %s\n", args.out_wav);
    printf("==========================================================\n\n");

    return 0;
}
