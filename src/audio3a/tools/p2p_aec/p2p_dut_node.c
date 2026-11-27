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
#include <sys/stat.h>
#include <sys/wait.h>
#include <errno.h>
#include "p2p_protocol.h"
#include "linear_aec.h"
#include "spatial_dereverb.h"
#include "online_wpe.h"
#include "neural_res_ans.h"
#include "dfnet3_ans.h"
#include "dpdfnet_ans.h"
#include "adaptive_agc.h"
#include "transient_crossover.h"
#include "wav_file.h"
#include "webrtc_vad/webrtc_vad.h"
#include "sherpa-onnx/c-api/c-api.h"

static volatile bool g_running = true;
static volatile uint32_t g_last_far_timestamp_ms = 0;

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

#define FIFO_MAX_FRAMES 128
#define MAX_MIC_CHANNELS 16
#define HANGOVER_FRAMES_MAX 75  /* 75 * 16ms = 1200ms continuous speech reading protection */

typedef struct {
    int16_t frames[FIFO_MAX_FRAMES][P2P_FRAME_SAMPLES];
    int write_idx;
    int read_idx;
    int count;
    bool is_primed;
    pthread_mutex_t lock;
} NetJitterFIFO;

static NetJitterFIFO g_net_fifo;

static void net_fifo_init(NetJitterFIFO *f) {
    memset(f, 0, sizeof(*f));
    pthread_mutex_init(&f->lock, NULL);
}

static void net_fifo_push(NetJitterFIFO *f, const int16_t *pcm) {
    pthread_mutex_lock(&f->lock);
    if (f->count < FIFO_MAX_FRAMES) {
        memcpy(f->frames[f->write_idx], pcm, P2P_FRAME_SAMPLES * sizeof(int16_t));
        f->write_idx = (f->write_idx + 1) % FIFO_MAX_FRAMES;
        f->count++;
    }
    pthread_mutex_unlock(&f->lock);
}

static bool net_fifo_pull(NetJitterFIFO *f, int16_t *out_pcm, int min_prebuffer) {
    pthread_mutex_lock(&f->lock);
    if (!f->is_primed) {
        if (f->count >= min_prebuffer) {
            f->is_primed = true;
        } else {
            pthread_mutex_unlock(&f->lock);
            memset(out_pcm, 0, P2P_FRAME_SAMPLES * sizeof(int16_t));
            return false;
        }
    }
    if (f->count <= 0) {
        f->is_primed = false;
        pthread_mutex_unlock(&f->lock);
        memset(out_pcm, 0, P2P_FRAME_SAMPLES * sizeof(int16_t));
        return false;
    }
    memcpy(out_pcm, f->frames[f->read_idx], P2P_FRAME_SAMPLES * sizeof(int16_t));
    f->read_idx = (f->read_idx + 1) % FIFO_MAX_FRAMES;
    f->count--;
    pthread_mutex_unlock(&f->lock);
    return true;
}

typedef struct {
    int16_t frames[FIFO_MAX_FRAMES][P2P_FRAME_SAMPLES];
    int write_idx;
    int total_pushed;
} SyncDelayLine;

static SyncDelayLine g_ref_delay_line;

static void delay_line_init(SyncDelayLine *dl) {
    memset(dl, 0, sizeof(*dl));
}

static void delay_line_step(SyncDelayLine *dl, const int16_t *in_pcm, int16_t *out_delayed, int delay_frames) {
    if (delay_frames <= 0) {
        memcpy(out_delayed, in_pcm, P2P_FRAME_SAMPLES * sizeof(int16_t));
    } else if (dl->total_pushed < delay_frames) {
        memset(out_delayed, 0, P2P_FRAME_SAMPLES * sizeof(int16_t));
    } else {
        int read_idx = (dl->write_idx - delay_frames + FIFO_MAX_FRAMES) % FIFO_MAX_FRAMES;
        memcpy(out_delayed, dl->frames[read_idx], P2P_FRAME_SAMPLES * sizeof(int16_t));
    }
    memcpy(dl->frames[dl->write_idx], in_pcm, P2P_FRAME_SAMPLES * sizeof(int16_t));
    dl->write_idx = (dl->write_idx + 1) % FIFO_MAX_FRAMES;
    dl->total_pushed++;
}

typedef enum {
    ANS_BACKEND_AUTO = 0,
    ANS_BACKEND_DFNET3,
    ANS_BACKEND_DPDFNET,
    ANS_BACKEND_GTCRN,
    ANS_BACKEND_RNNOISE
} AnsBackend;

typedef struct {
    int   listen_port;
    int   send_port;
    char  peer_ip[64];
    bool  peer_known;
    struct sockaddr_in peer_addr;
    int   delay_frames;
    bool  enable_aec;
    char  mic_device[64];
    int   mic_channels;
    int   ref_channel;       /* -1: use net_ref; >=0: use mic_raw channel as hardware loopback */
    char  spk_device[64];
    char  inject_wav_path[256];
    int   inject_delay_sec;
    float mic_pregain;
    bool  use_calib_table;
    bool  enable_gtcrn;
    AnsBackend ans_backend;
    char  gtcrn_model_path[256];
    char  dfnet_model_path[256];
    float dfnet_atten_lim;
    float dfnet_pf_beta;
    char  dpdfnet_model_path[256];
    int   dpdfnet_threads;
    char  offline_in_wav[256];
    char  offline_out_wav[256];
    char  dump_stages_dir[256];
    bool  debug_transient;
    bool  enable_dereverb;
    float dereverb_floor;
    bool  enable_wpe;
    int   wpe_taps;
    int   wpe_delay;
    float wpe_alpha;
    float wpe_floor;
    float agc_target_db;
    float agc_max_boost_db;
    bool  enable_input_hpf;
    float hpf_cutoff_hz;
    bool  enable_nvidia;
    char  nvidia_bin_path[256];
    char  nvidia_model_path[256];
    int   nvidia_core_mask;
    float nvidia_dry_gain;
} DUTConfig;

static DUTConfig g_cfg;
static FILE *g_spk_pipe = NULL;
static FILE *g_mic_pipe = NULL;

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float x1, x2;
    float y1, y2;
} MicHPF;

static void mic_hpf_init(MicHPF *hpf, float fc, float fs) {
    if (fc <= 10.0f) {
        hpf->b0 = 1.0f; hpf->b1 = 0.0f; hpf->b2 = 0.0f;
        hpf->a1 = 0.0f; hpf->a2 = 0.0f;
        hpf->x1 = hpf->x2 = hpf->y1 = hpf->y2 = 0.0f;
        return;
    }
    float omega = 2.0f * (float)M_PI * fc / fs;
    float cos_w = cosf(omega);
    float sin_w = sinf(omega);
    float alpha = sin_w / (2.0f * 0.70710678f);
    float a0 = 1.0f + alpha;
    hpf->b0 = (1.0f + cos_w) / (2.0f * a0);
    hpf->b1 = -(1.0f + cos_w) / a0;
    hpf->b2 = (1.0f + cos_w) / (2.0f * a0);
    hpf->a1 = -2.0f * cos_w / a0;
    hpf->a2 = (1.0f - alpha) / a0;
    hpf->x1 = hpf->x2 = 0.0f;
    hpf->y1 = hpf->y2 = 0.0f;
}

static inline float mic_hpf_process(MicHPF *hpf, float x) {
    float y = hpf->b0 * x + hpf->b1 * hpf->x1 + hpf->b2 * hpf->x2 - hpf->a1 * hpf->y1 - hpf->a2 * hpf->y2;
    hpf->x2 = hpf->x1;
    hpf->x1 = x;
    hpf->y2 = hpf->y1;
    hpf->y1 = y;
    return y;
}

typedef struct {
    int num_mics;
    int selected_mic_idx;
    /* Auto channel selection: pick the mic with the widest energy dynamic range.
     * Measured on 6 real board recordings, that mic is also the one with the
     * strongest post-speech decay in 5/6 cases (the 6th differed by 0.2 dB).
     * Static per-mic choice is not viable: the best index shifted across seats. */
    bool auto_select_mic;
    int  asel_frame;
    double asel_hi[MAX_MIC_CHANNELS];   /* running loud-frame energy */
    double asel_lo[MAX_MIC_CHANNELS];   /* running quiet-frame energy */
    bool enable_aec;
    bool enable_dereverb;
    bool enable_wpe;
    bool enable_input_hpf;
    MicHPF mic_hpf[MAX_MIC_CHANNELS];
    MicHPF ref_hpf;
    bool debug_transient;
    bool use_dfnet3;
    bool use_dpdfnet;
    float agc_target_db;
    float agc_max_boost_db;
    VadInst *vad;
    VadInst *vad_near;
    float vad_hpf_x;
    float vad_hpf_y;
    double near_noise_energy;
    LinearAEC aec[MAX_MIC_CHANNELS];
    OnlineWpe wpe;
    SpatialDereverb dereverb;
    float last_coherence;
    NeuralRESANS neural_engine;
    AdaptiveAGC agc;
    AdaptiveAGC post_agc;
    TransientCrossover post_tc;
    DFNet3ANS dfnet3;
    DPDFNetANS dpdfnet;
    const SherpaOnnxOnlineSpeechDenoiser *gtcrn;
    int hangover_counter;
    int dt_onset_streak;
    double ref_energy_env;
    float smooth_gate;
    double e_ref_sum;
    uint32_t frame_idx;
} DUTDspState;

#define AUTO_SELECT_MIC   (-2)   /* sentinel for --select-mic auto */
#define AUTO_SELECT_FRAMES 250   /* ~4 s of audio before locking the channel */

/* Track per-mic dynamic range and lock in the widest once enough audio has been
 * seen. Called before WPE so the decision is based on the raw array. */
static void auto_select_update(DUTDspState *st,
                               int16_t mic_ch[MAX_MIC_CHANNELS][P2P_FRAME_SAMPLES]) {
    if (!st->auto_select_mic || st->selected_mic_idx >= 0) return;

    double frame_e[MAX_MIC_CHANNELS];
    double max_e = 0.0;
    for (int m = 0; m < st->num_mics; ++m) {
        double s = 0.0;
        for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
            double v = (double)mic_ch[m][i] / 32768.0;
            s += v * v;
        }
        frame_e[m] = s / P2P_FRAME_SAMPLES;
        if (frame_e[m] > max_e) max_e = frame_e[m];
    }

    /* Only sample frames that are clearly speech or clearly quiet; the middle
     * band is dominated by reverb tails and would flatten the two extremes. */
    if (max_e > 1e-8) {
        for (int m = 0; m < st->num_mics; ++m) {
            if (frame_e[m] > max_e * 0.25) {
                st->asel_hi[m] = fmax(st->asel_hi[m], frame_e[m]);
            } else if (frame_e[m] < max_e * 0.01) {
                st->asel_lo[m] = fmax(st->asel_lo[m], frame_e[m]);
            }
        }
    }

    if (++st->asel_frame < AUTO_SELECT_FRAMES) return;

    int best = 0;
    double best_dr = -1.0;
    for (int m = 0; m < st->num_mics; ++m) {
        double dr = 10.0 * log10(st->asel_hi[m] / st->asel_lo[m]);
        if (dr > best_dr) { best_dr = dr; best = m; }
    }
    st->selected_mic_idx = best;
    printf("[DUT-Main] Auto mic select: Ch%d (dynamic range %.1f dB) after %.1fs\n",
           best, best_dr, st->asel_frame * 0.016f);
}

static bool resolve_dfnet_model_path(const char *explicit_path, char *out_path, size_t out_size) {
    if (explicit_path && explicit_path[0] != '\0') {
        if (access(explicit_path, R_OK) == 0) {
            snprintf(out_path, out_size, "%s", explicit_path);
            return true;
        }
        return false;
    }

    char exe_dir[512] = "";
    ssize_t len = readlink("/proc/self/exe", exe_dir, sizeof(exe_dir) - 1);
    if (len > 0) {
        exe_dir[len] = '\0';
        char *slash = strrchr(exe_dir, '/');
        if (slash) *slash = '\0';
        const char *rel_candidates[] = {
            "../models/DeepFilterNet3_onnx.tar.gz",
            "../../deepfilternet/models/DeepFilterNet3_onnx.tar.gz",
            "models/DeepFilterNet3_onnx.tar.gz"
        };
        for (size_t i = 0; i < sizeof(rel_candidates) / sizeof(rel_candidates[0]); ++i) {
            char cand[768];
            int nw = snprintf(cand, sizeof(cand), "%s/%s", exe_dir, rel_candidates[i]);
            if (nw > 0 && (size_t)nw < out_size && access(cand, R_OK) == 0) {
                memcpy(out_path, cand, (size_t)nw + 1);
                return true;
            }
        }
    }

    const char *fallback_candidates[] = {
        "./models/DeepFilterNet3_onnx.tar.gz",
        "./deepfilternet/models/DeepFilterNet3_onnx.tar.gz",
        "/root/audio3a_release/models/DeepFilterNet3_onnx.tar.gz"
    };
    for (size_t i = 0; i < sizeof(fallback_candidates) / sizeof(fallback_candidates[0]); ++i) {
        if (access(fallback_candidates[i], R_OK) == 0) {
            snprintf(out_path, out_size, "%s", fallback_candidates[i]);
            return true;
        }
    }
    return false;
}

static bool resolve_gtcrn_model_path(const char *explicit_path, char *out_path, size_t out_size) {
    if (explicit_path && explicit_path[0] != '\0') {
        if (access(explicit_path, R_OK) == 0) {
            snprintf(out_path, out_size, "%s", explicit_path);
            return true;
        }
        return false;
    }

    char exe_dir[512] = "";
    ssize_t len = readlink("/proc/self/exe", exe_dir, sizeof(exe_dir) - 1);
    if (len > 0) {
        exe_dir[len] = '\0';
        char *slash = strrchr(exe_dir, '/');
        if (slash) *slash = '\0';
        const char *rel_candidates[] = {
            "../models/gtcrn_simple.onnx",
            "../models/onnx/gtcrn_simple.onnx",
            "../../models/onnx/gtcrn_simple.onnx",
            "models/gtcrn_simple.onnx"
        };
        for (size_t i = 0; i < sizeof(rel_candidates) / sizeof(rel_candidates[0]); ++i) {
            char cand[768];
            int nw = snprintf(cand, sizeof(cand), "%s/%s", exe_dir, rel_candidates[i]);
            if (nw > 0 && (size_t)nw < out_size && access(cand, R_OK) == 0) {
                memcpy(out_path, cand, (size_t)nw + 1);
                return true;
            }
        }
    }

    const char *fallback_candidates[] = {
        "./models/gtcrn_simple.onnx",
        "./models/onnx/gtcrn_simple.onnx",
        "/root/audio3a_release/models/gtcrn_simple.onnx",
        "/tmp/gtcrn_bench/gtcrn_simple.onnx"
    };
    for (size_t i = 0; i < sizeof(fallback_candidates) / sizeof(fallback_candidates[0]); ++i) {
        if (access(fallback_candidates[i], R_OK) == 0) {
            snprintf(out_path, out_size, "%s", fallback_candidates[i]);
            return true;
        }
    }
    return false;
}

static bool resolve_dpdfnet_model_path(const char *explicit_path, char *out_path, size_t out_size) {
    if (explicit_path && explicit_path[0] != '\0') {
        if (access(explicit_path, R_OK) == 0) {
            snprintf(out_path, out_size, "%s", explicit_path);
            return true;
        }
        return false;
    }

    char exe_dir[512] = "";
    ssize_t len = readlink("/proc/self/exe", exe_dir, sizeof(exe_dir) - 1);
    if (len > 0) {
        exe_dir[len] = '\0';
        char *slash = strrchr(exe_dir, '/');
        if (slash) *slash = '\0';
        const char *rel_candidates[] = {
            "../models/dpdfnet2.onnx",
            "../models/dpdfnet_baseline.onnx",
            "../../models/onnx/dpdfnet/dpdfnet2.onnx",
            "../../models/onnx/dpdfnet/dpdfnet_baseline.onnx",
            "models/dpdfnet2.onnx"
        };
        for (size_t i = 0; i < sizeof(rel_candidates) / sizeof(rel_candidates[0]); ++i) {
            char cand[768];
            int nw = snprintf(cand, sizeof(cand), "%s/%s", exe_dir, rel_candidates[i]);
            if (nw > 0 && (size_t)nw < out_size && access(cand, R_OK) == 0) {
                memcpy(out_path, cand, (size_t)nw + 1);
                return true;
            }
        }
    }

    const char *fallback_candidates[] = {
        "./models/dpdfnet2.onnx",
        "./models/dpdfnet_baseline.onnx",
        "./models/onnx/dpdfnet/dpdfnet2.onnx",
        "./models/onnx/dpdfnet/dpdfnet_baseline.onnx",
        "/root/audio3a_release/models/dpdfnet2.onnx",
        "/root/audio3a_release/models/dpdfnet_baseline.onnx"
    };
    for (size_t i = 0; i < sizeof(fallback_candidates) / sizeof(fallback_candidates[0]); ++i) {
        if (access(fallback_candidates[i], R_OK) == 0) {
            snprintf(out_path, out_size, "%s", fallback_candidates[i]);
            return true;
        }
    }
    return false;
}

static bool resolve_nvidia_model_path(const char *explicit_path, char *out_path, size_t out_size) {
    if (explicit_path && explicit_path[0] != '\0') {
        if (access(explicit_path, R_OK) == 0) {
            snprintf(out_path, out_size, "%s", explicit_path);
            return true;
        }
        return false;
    }

    char exe_dir[512] = "";
    ssize_t len = readlink("/proc/self/exe", exe_dir, sizeof(exe_dir) - 1);
    if (len > 0) {
        exe_dir[len] = '\0';
        char *slash = strrchr(exe_dir, '/');
        if (slash) *slash = '\0';
        const char *rel_candidates[] = {
            "../models/nvidia-t32-spatial-norm.rknn",
            "models/nvidia-t32-spatial-norm.rknn",
            "../../models/rknn/live/nvidia-t32-spatial-norm.rknn",
            "../../SL-clearVoice/models/rknn/live/nvidia-t32-spatial-norm.rknn",
            "../models/nvidia-t16-spatial-norm.rknn",
            "models/nvidia-t16-spatial-norm.rknn",
            "../../models/rknn/live/nvidia-t16-spatial-norm.rknn",
            "../../SL-clearVoice/models/rknn/live/nvidia-t16-spatial-norm.rknn"
        };
        for (size_t i = 0; i < sizeof(rel_candidates) / sizeof(rel_candidates[0]); ++i) {
            char cand[768];
            int nw = snprintf(cand, sizeof(cand), "%s/%s", exe_dir, rel_candidates[i]);
            if (nw > 0 && (size_t)nw < out_size && access(cand, R_OK) == 0) {
                memcpy(out_path, cand, (size_t)nw + 1);
                return true;
            }
        }
    }

    const char *fallback_candidates[] = {
        "./models/nvidia-t32-spatial-norm.rknn",
        "/root/audio3a_release/models/nvidia-t32-spatial-norm.rknn",
        "/mnt/data/zjw/source/3308/Audio3A/SL-clearVoice/models/rknn/live/nvidia-t32-spatial-norm.rknn",
        "./models/nvidia-t16-spatial-norm.rknn",
        "/root/audio3a_release/models/nvidia-t16-spatial-norm.rknn",
        "/tmp/test_g4e_nv/nvidia-t16-spatial-norm.rknn",
        "/mnt/data/zjw/source/3308/Audio3A/SL-clearVoice/models/rknn/live/nvidia-t16-spatial-norm.rknn",
        "/mnt/data/zjw/source/3308/Audio3A/models/rknn/live/nvidia-t16-spatial-norm.rknn"
    };
    for (size_t i = 0; i < sizeof(fallback_candidates) / sizeof(fallback_candidates[0]); ++i) {
        if (access(fallback_candidates[i], R_OK) == 0) {
            snprintf(out_path, out_size, "%s", fallback_candidates[i]);
            return true;
        }
    }
    return false;
}

static bool resolve_nvidia_bin_path(const char *explicit_path, char *out_path, size_t out_size) {
    if (explicit_path && explicit_path[0] != '\0') {
        if (access(explicit_path, X_OK) == 0) {
            snprintf(out_path, out_size, "%s", explicit_path);
            return true;
        }
        return false;
    }

    char exe_dir[512] = "";
    ssize_t len = readlink("/proc/self/exe", exe_dir, sizeof(exe_dir) - 1);
    if (len > 0) {
        exe_dir[len] = '\0';
        char *slash = strrchr(exe_dir, '/');
        if (slash) *slash = '\0';
        const char *rel_candidates[] = {
            "nvidia_sb_ode_pcm",
            "../bin/nvidia_sb_ode_pcm",
            "bin/nvidia_sb_ode_pcm",
            "nvidia_sb_pcm",
            "../bin/nvidia_sb_pcm"
        };
        for (size_t i = 0; i < sizeof(rel_candidates) / sizeof(rel_candidates[0]); ++i) {
            char cand[768];
            int nw = snprintf(cand, sizeof(cand), "%s/%s", exe_dir, rel_candidates[i]);
            if (nw > 0 && (size_t)nw < out_size && access(cand, X_OK) == 0) {
                memcpy(out_path, cand, (size_t)nw + 1);
                return true;
            }
        }
    }

    const char *fallback_candidates[] = {
        "./nvidia_sb_ode_pcm",
        "./bin/nvidia_sb_ode_pcm",
        "/root/audio3a_release/bin/nvidia_sb_ode_pcm",
        "/tmp/test_g4e_nv/nvidia_sb_ode_pcm",
        "/mnt/data/zjw/source/3308/Audio3A/releases/nvidia_sb_ode_pcm",
        "/mnt/data/zjw/source/3308/Audio3A/SL-clearVoice/out/rk3588-live-freeze/bundle/bin/nvidia_sb_pcm"
    };
    for (size_t i = 0; i < sizeof(fallback_candidates) / sizeof(fallback_candidates[0]); ++i) {
        if (access(fallback_candidates[i], X_OK) == 0) {
            snprintf(out_path, out_size, "%s", fallback_candidates[i]);
            return true;
        }
    }
    return false;
}

typedef struct {
    pid_t pid;
    int to_nv_fd;
    int from_nv_fd;
    pthread_t thread;
    volatile bool running;
    int send_sock;
    FILE *fp_record_clean;
    uint32_t out_seq;
    uint32_t recorded_samples;
    WavFile *out_wav;
    AdaptiveAGC post_nv_agc;
    int post_nv_hangover;
} NvidiaBridge;

static NvidiaBridge g_nv_bridge;

static const float kNvidiaPostInitGainLinear = 1.68f; /* +4.5 dB initial makeup baseline after NVIDIA ODE */

typedef enum {
    STAGE_00_RAW_INPUT_AVG = 0,
    STAGE_01_PREGAIN_AVG,
    STAGE_02_LINEAR_AEC,
    STAGE_03_WPE,
    STAGE_04_SPATIAL_FUSION,
    STAGE_05_NEURAL_RES,
    STAGE_06_PRE_AGC,
    STAGE_07_TRANSIENT_XOVER,
    STAGE_08_POST_ANS,
    STAGE_09_POST_AGC,
    STAGE_10_SMOOTH_GATE,
    STAGE_11_NVIDIA_ODE_RAW,
    STAGE_12_FINAL_OUTPUT,
    STAGE_COUNT
} StageIndex;

typedef struct {
    bool enabled;
    WavFile *files[STAGE_COUNT];
} StageDump;

static StageDump g_stage_dump = {0};

static const char *kStageFileNames[STAGE_COUNT] = {
    "00_raw_input_avg.wav",
    "01_pregain_avg.wav",
    "02_linear_aec.wav",
    "03_wpe.wav",
    "04_spatial_fusion.wav",
    "05_neural_res.wav",
    "06_pre_agc.wav",
    "07_transient_xover.wav",
    "08_post_ans.wav",
    "09_post_agc.wav",
    "10_smooth_gate.wav",
    "11_nvidia_ode_raw.wav",
    "12_final_output.wav"
};

static bool stage_dump_open(const char *dir) {
    if (!dir || dir[0] == '\0') return false;
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "[Warning] Failed to create stage dump dir %s: %s\n", dir, strerror(errno));
        return false;
    }
    for (int i = 0; i < STAGE_COUNT; ++i) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", dir, kStageFileNames[i]);
        g_stage_dump.files[i] = WavFile_OpenWrite(path, 1, P2P_SAMPLE_RATE);
        if (!g_stage_dump.files[i]) {
            fprintf(stderr, "[Warning] Failed to open stage dump file %s\n", path);
        }
    }
    g_stage_dump.enabled = true;
    printf("[DUT-Dump] Saving per-stage WAVs to: %s\n", dir);
    return true;
}

static void stage_dump_write(StageIndex idx, const int16_t *pcm, int samples) {
    if (!g_stage_dump.enabled || idx < 0 || idx >= STAGE_COUNT) return;
    if (g_stage_dump.files[idx]) {
        WavFile_WriteInt16(g_stage_dump.files[idx], pcm, samples);
    }
}

static void stage_dump_write_ch_avg(StageIndex idx, int16_t ch[MAX_MIC_CHANNELS][P2P_FRAME_SAMPLES], int num_mics) {
    if (!g_stage_dump.enabled || num_mics <= 0) return;
    int16_t avg[P2P_FRAME_SAMPLES];
    for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
        int32_t sum = 0;
        for (int m = 0; m < num_mics; ++m) sum += ch[m][i];
        avg[i] = (int16_t)(sum / num_mics);
    }
    stage_dump_write(idx, avg, P2P_FRAME_SAMPLES);
}

static void stage_dump_close(void) {
    if (!g_stage_dump.enabled) return;
    for (int i = 0; i < STAGE_COUNT; ++i) {
        if (g_stage_dump.files[i]) {
            WavFile_Close(g_stage_dump.files[i]);
            g_stage_dump.files[i] = NULL;
        }
    }
    g_stage_dump.enabled = false;
}

/* Smooth post-NVIDIA dynamic loudness leveling:
 * Compensates uneven dereverb RMS attenuation across 1m~5m without amplifying silence or clipping peaks. */
static void apply_nvidia_post_leveling(NvidiaBridge *b, int16_t *pcm, int count) {
    if (!b || !pcm || count <= 0) return;
    double sum_sq = 0.0;
    for (int i = 0; i < count; ++i) {
        double x = (double)pcm[i] / 32768.0;
        sum_sq += x * x;
    }
    float rms = (float)sqrt(sum_sq / (double)count);
    /* NVIDIA ODE silence floor is <-90 dBFS; 0.0015f (-56.5 dBFS) catches quiet far-field words */
    if (rms > 0.0015f) {
        b->post_nv_hangover = 40; /* 640ms hangover */
    } else if (b->post_nv_hangover > 0) {
        b->post_nv_hangover--;
    }
    AdaptiveAGC_SetVAD(&b->post_nv_agc, (b->post_nv_hangover > 0) ? 1 : 0);
    AdaptiveAGC_ProcessInt16(&b->post_nv_agc, pcm, pcm, count);
}

static void init_mic_gains(float gain_tbl[MAX_MIC_CHANNELS], int num_mics,
                           bool has_custom_gains, const float custom_gains[MAX_MIC_CHANNELS]) {
    /* Calibrated 8-Mic EQ table for RK3588 hw:3,0 PDM capture (-72~-82 dBFS raw -> -30 dBFS balanced, <0.3 dB spread) */
    static const float kCalib8[8] = {290.0f, 84.0f, 81.0f, 80.0f, 89.0f, 250.0f, 92.0f, 88.0f};
    for (int m = 0; m < MAX_MIC_CHANNELS; ++m) {
        gain_tbl[m] = g_cfg.mic_pregain;
    }
    if (has_custom_gains) {
        for (int m = 0; m < num_mics; ++m) gain_tbl[m] = custom_gains[m];
    } else if (g_cfg.use_calib_table && num_mics == 8) {
        for (int m = 0; m < 8; ++m) gain_tbl[m] = kCalib8[m];
    }
}

static void* nvidia_online_reader_thread(void *arg) {
    NvidiaBridge *b = (NvidiaBridge*)arg;
    int16_t frame[P2P_FRAME_SAMPLES];
    P2PPacket return_pkt;
    memset(&return_pkt, 0, sizeof(return_pkt));
    return_pkt.magic = P2P_MAGIC;
    return_pkt.samples = P2P_FRAME_SAMPLES;

    while (b->running) {
        size_t needed = sizeof(frame);
        uint8_t *ptr = (uint8_t*)frame;
        while (needed > 0 && b->running) {
            ssize_t r = read(b->from_nv_fd, ptr, needed);
            if (r <= 0) {
                b->running = false;
                break;
            }
            ptr += r;
            needed -= (size_t)r;
        }
        if (!b->running) break;

        apply_nvidia_post_leveling(b, frame, P2P_FRAME_SAMPLES);

        if (b->fp_record_clean) {
            fwrite(frame, sizeof(int16_t), P2P_FRAME_SAMPLES, b->fp_record_clean);
            b->recorded_samples += P2P_FRAME_SAMPLES;
        }

        if (g_cfg.peer_known && b->send_sock >= 0) {
            return_pkt.seq = ++b->out_seq;
            return_pkt.timestamp_ms = g_last_far_timestamp_ms;
            return_pkt.flags = g_cfg.enable_aec ? 1 : 0;
            memcpy(return_pkt.pcm, frame, sizeof(frame));

            sendto(b->send_sock, &return_pkt, sizeof(return_pkt), 0,
                   (struct sockaddr*)&g_cfg.peer_addr, sizeof(g_cfg.peer_addr));

            if (b->out_seq % 50 == 0) {
                printf("[DUT-Nvidia Stats] P2P Frame:%-5u | Sent to %s:%d\n",
                       b->out_seq, g_cfg.peer_ip, g_cfg.send_port);
                fflush(stdout);
            }
        }
    }
    return NULL;
}

static void* nvidia_offline_reader_thread(void *arg) {
    NvidiaBridge *b = (NvidiaBridge*)arg;
    int16_t frame[P2P_FRAME_SAMPLES];

    while (1) {
        size_t needed = sizeof(frame);
        uint8_t *ptr = (uint8_t*)frame;
        while (needed > 0) {
            ssize_t r = read(b->from_nv_fd, ptr, needed);
            if (r <= 0) break;
            ptr += r;
            needed -= (size_t)r;
        }
        if (needed > 0) {
            size_t valid_bytes = sizeof(frame) - needed;
            int valid_samples = (int)(valid_bytes / sizeof(int16_t));
            if (valid_samples > 0) {
                stage_dump_write(STAGE_11_NVIDIA_ODE_RAW, frame, valid_samples);
                apply_nvidia_post_leveling(b, frame, valid_samples);
                stage_dump_write(STAGE_12_FINAL_OUTPUT, frame, valid_samples);
                if (b->out_wav) {
                    WavFile_WriteInt16(b->out_wav, frame, valid_samples);
                    b->recorded_samples += (uint32_t)valid_samples;
                }
            }
            break;
        }
        stage_dump_write(STAGE_11_NVIDIA_ODE_RAW, frame, P2P_FRAME_SAMPLES);
        apply_nvidia_post_leveling(b, frame, P2P_FRAME_SAMPLES);
        stage_dump_write(STAGE_12_FINAL_OUTPUT, frame, P2P_FRAME_SAMPLES);
        if (b->out_wav) {
            WavFile_WriteInt16(b->out_wav, frame, P2P_FRAME_SAMPLES);
            b->recorded_samples += P2P_FRAME_SAMPLES;
        }
    }
    return NULL;
}

static bool nvidia_bridge_start(NvidiaBridge *b, const char *model_path, const char *bin_path,
                                int core_mask, float dry_gain, bool is_offline,
                                int send_sock, FILE *fp_record_clean, WavFile *out_wav) {
    memset(b, 0, sizeof(*b));
    b->send_sock = send_sock;
    b->fp_record_clean = fp_record_clean;
    b->out_wav = out_wav;

    AGCConfig nv_agc_cfg;
    AdaptiveAGC_GetDefaultConfig(&nv_agc_cfg);
    nv_agc_cfg.target_level_rms = powf(10.0f, (g_cfg.agc_target_db + 1.0f) / 20.0f);
    nv_agc_cfg.max_gain_db = 14.0f;         /* Smoothly recovers 1m~5m dereverb energy drop */
    nv_agc_cfg.min_gain_db = -6.0f;
    nv_agc_cfg.noise_gate_threshold = 0.002f; /* -54 dBFS: catches quiet 5m syllables while >35dB above silence floor */
    nv_agc_cfg.attack_time_ms = 10.0f;
    nv_agc_cfg.release_time_ms = 260.0f;
    nv_agc_cfg.silence_standby_gain = 1.0f;
    nv_agc_cfg.enable_eq = false;
    nv_agc_cfg.enable_limiter = true;
    /* Measured on the board: this stage lifts the ODE's silence floor by
     * +8.8 dB while lifting speech only +6.9 dB, so the gap shrinks from 36.9
     * to 35.1 dB and the residue becomes audible. Downward expansion targets
     * exactly that: below-envelope frames get pulled back down while speech
     * keeps its level.
     * Note: an earlier offline trial judged this worse on DNSMOS (OVRL -0.010,
     * P808 -0.052) with only BAK +0.032. DNSMOS penalises the gain movement
     * itself, so it was re-evaluated on physical speech/silence separation,
     * which is what the subjective complaint tracks. */
    nv_agc_cfg.expand_range_db = 18.0f;
    nv_agc_cfg.expand_ratio = 1.0f;
    nv_agc_cfg.expand_max_cut_db = 18.0f;
    AdaptiveAGC_Init(&b->post_nv_agc, &nv_agc_cfg, P2P_SAMPLE_RATE);
    b->post_nv_agc.current_gain = kNvidiaPostInitGainLinear;
    b->post_nv_agc.smoothed_rms = nv_agc_cfg.target_level_rms / kNvidiaPostInitGainLinear;

    int pipe_to[2], pipe_from[2];
    if (pipe(pipe_to) != 0 || pipe(pipe_from) != 0) {
        fprintf(stderr, "[Error] Failed to create pipes for Nvidia bridge: %s\n", strerror(errno));
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "[Error] Failed to fork Nvidia worker: %s\n", strerror(errno));
        close(pipe_to[0]); close(pipe_to[1]);
        close(pipe_from[0]); close(pipe_from[1]);
        return false;
    }

    if (pid == 0) {
        close(pipe_to[1]);
        close(pipe_from[0]);
        dup2(pipe_to[0], STDIN_FILENO);
        dup2(pipe_from[1], STDOUT_FILENO);
        close(pipe_to[0]);
        close(pipe_from[1]);

        char env_mask[32], env_dry[32];
        snprintf(env_mask, sizeof(env_mask), "%d", core_mask);
        snprintf(env_dry, sizeof(env_dry), "%.4f", dry_gain);
        setenv("SB_RKNN_CORE_MASK", env_mask, 1);
        setenv("SB_DRY_GAIN", env_dry, 1);

        /* Tuned for guaranteed over_budget=0 under full 8-mic WPE + DPDFNet2 + NPU concurrency:
         * - T32 (default): overlap_ms=48, history_ms=48 (248ms window, 152ms stride, 3-core NPU)
         * - T16 (low-latency): overlap_ms=8, history_ms=16 (120ms window, 96ms stride) */
        bool is_t32 = (strstr(model_path, "t32") != NULL || strstr(model_path, "T32") != NULL);
        const char *overlap_arg = is_t32 ? "48" : "8";
        const char *history_arg = is_t32 ? "48" : "16";
        execl(bin_path, bin_path, model_path, overlap_arg, history_arg, NULL);
        fprintf(stderr, "[Child Error] execl failed for %s: %s\n", bin_path, strerror(errno));
        _exit(127);
    }

    close(pipe_to[0]);
    close(pipe_from[1]);
    b->pid = pid;
    b->to_nv_fd = pipe_to[1];
    b->from_nv_fd = pipe_from[0];
    b->running = true;

    if (is_offline) {
        pthread_create(&b->thread, NULL, nvidia_offline_reader_thread, b);
    } else {
        pthread_create(&b->thread, NULL, nvidia_online_reader_thread, b);
    }
    bool is_t32_log = (strstr(model_path, "t32") != NULL || strstr(model_path, "T32") != NULL);
    printf("[DUT-Main] Nvidia ODE Neural Dereverb Active (%s, overlap=%s, history=%s): bin=%s model=%s (core_mask=%d, dry_gain=%.3f)\n",
           is_t32_log ? "T32" : "T16", is_t32_log ? "48ms" : "8ms", is_t32_log ? "48ms" : "16ms",
           bin_path, model_path, core_mask, dry_gain);
    return true;
}

static void nvidia_bridge_stop(NvidiaBridge *b) {
    if (!b->running && b->pid <= 0) return;
    if (b->to_nv_fd >= 0) {
        close(b->to_nv_fd);
        b->to_nv_fd = -1;
    }
    if (b->thread) {
        pthread_join(b->thread, NULL);
        b->thread = 0;
    }
    if (b->from_nv_fd >= 0) {
        close(b->from_nv_fd);
        b->from_nv_fd = -1;
    }
    if (b->pid > 0) {
        int status;
        waitpid(b->pid, &status, 0);
        b->pid = 0;
    }
    b->running = false;
}

static int dut_dsp_init(DUTDspState *st, int num_mics, int selected_mic_idx,
                        bool enable_aec, AnsBackend ans_backend,
                        bool enable_gtcrn, const char *gtcrn_model_override,
                        const char *dfnet_model_override,
                        float dfnet_atten_lim, float dfnet_pf_beta,
                        const char *dpdfnet_model_override, int dpdfnet_threads,
                        bool debug_transient,
                        bool enable_dereverb, float dereverb_floor,
                        bool enable_wpe, int wpe_taps, int wpe_delay,
                        float wpe_alpha, float wpe_floor,
                        float agc_target_db, float agc_max_boost_db,
                        bool enable_input_hpf, float hpf_cutoff_hz) {
    memset(st, 0, sizeof(*st));
    st->num_mics = num_mics;
    st->selected_mic_idx = selected_mic_idx;
    st->auto_select_mic = (selected_mic_idx == AUTO_SELECT_MIC);
    if (st->auto_select_mic) {
        /* Keep the whole array until the decision is made, then lock the winner. */
        st->selected_mic_idx = -1;
        for (int m = 0; m < MAX_MIC_CHANNELS; ++m) {
            st->asel_hi[m] = 1e-12;
            st->asel_lo[m] = 1e-12;
        }
    }
    st->enable_aec = enable_aec;
    st->debug_transient = debug_transient;
    st->agc_target_db = agc_target_db;
    st->agc_max_boost_db = agc_max_boost_db;
    st->smooth_gate = 0.05f;
    st->near_noise_energy = 1.0e-6;

    /* 0. Array Input High-Pass Filter (cuts 50Hz hum & structural vibration on mics & reference) */
    st->enable_input_hpf = enable_input_hpf && (hpf_cutoff_hz > 10.0f);
    if (st->enable_input_hpf) {
        for (int m = 0; m < MAX_MIC_CHANNELS; ++m) {
            mic_hpf_init(&st->mic_hpf[m], hpf_cutoff_hz, (float)P2P_SAMPLE_RATE);
        }
        mic_hpf_init(&st->ref_hpf, hpf_cutoff_hz, (float)P2P_SAMPLE_RATE);
        printf("[DUT-Main] Array Input HPF Active: %.1f Hz (2nd-order Butterworth, suppresses 50Hz hum)\n",
               hpf_cutoff_hz);
    } else {
        printf("[DUT-Main] Array Input HPF: [DISABLED]\n");
    }

    /* 1. WebRTC VAD: Mode 3 for active-echo DT guard, Mode 2 for quiet pure-near-end speech */
    st->vad = WebRtcVad_Create();
    if (!st->vad || WebRtcVad_Init(st->vad) != 0 || WebRtcVad_set_mode(st->vad, 3) != 0) {
        fprintf(stderr, "[Warning] WebRTC VAD init failed\n");
    }
    st->vad_near = WebRtcVad_Create();
    if (!st->vad_near || WebRtcVad_Init(st->vad_near) != 0 || WebRtcVad_set_mode(st->vad_near, 2) != 0) {
        fprintf(stderr, "[Warning] WebRTC near-end VAD init failed\n");
    }

    /* 2. Linear AEC for each microphone */
    LinearAECConfig aec_cfg;
    LinearAEC_GetDefaultConfig(&aec_cfg);
    aec_cfg.sample_rate = P2P_SAMPLE_RATE;
    aec_cfg.num_partitions = 8;
    aec_cfg.step_size = 0.25f;
    for (int m = 0; m < num_mics; ++m) {
        LinearAEC_Init(&st->aec[m], &aec_cfg);
    }

    /* 3. Causal Streaming Neural Speech Denoiser (DeepFilterNet 3 or GTCRN) */
    bool try_dfnet3 = (ans_backend == ANS_BACKEND_DFNET3 || ans_backend == ANS_BACKEND_AUTO);
    if (try_dfnet3) {
        char df_path[512];
        if (resolve_dfnet_model_path(dfnet_model_override, df_path, sizeof(df_path))) {
            if (DFNet3ANS_Init(&st->dfnet3, df_path, dfnet_atten_lim, dfnet_pf_beta) == 0) {
                st->use_dfnet3 = true;
                printf("[DUT-Main] Streaming DeepFilterNet 3 ANS Active (16k<->48k Bridge): %s (atten=%.0fdB, pf=%.3f)\n",
                       df_path, dfnet_atten_lim, dfnet_pf_beta);
            } else {
                fprintf(stderr, "[Warning] Failed to initialize DeepFilterNet 3 from %s; falling back.\n", df_path);
            }
        } else if (ans_backend == ANS_BACKEND_DFNET3) {
            fprintf(stderr, "[Warning] DeepFilterNet3_onnx.tar.gz not found; falling back.\n");
        }
    }

    bool try_dpdfnet = (!st->use_dfnet3 && ans_backend == ANS_BACKEND_DPDFNET);
    if (try_dpdfnet) {
        char dp_path[512];
        if (resolve_dpdfnet_model_path(dpdfnet_model_override, dp_path, sizeof(dp_path))) {
            if (DPDFNetANS_Init(&st->dpdfnet, dp_path, dpdfnet_threads) == 0) {
                st->use_dpdfnet = true;
                printf("[DUT-Main] Streaming DPDFNet ANS Active (%dT, 10ms FIFO Bridge): %s\n",
                       dpdfnet_threads > 0 ? dpdfnet_threads : 1, dp_path);
            } else {
                fprintf(stderr, "[Warning] Failed to initialize DPDFNet from %s; falling back.\n", dp_path);
            }
        } else {
            fprintf(stderr, "[Warning] DPDFNet ONNX model not found; falling back.\n");
        }
    }

    bool try_gtcrn = (!st->use_dfnet3 && !st->use_dpdfnet && ans_backend != ANS_BACKEND_RNNOISE && enable_gtcrn);
    if (try_gtcrn) {
        char model_path[512];
        if (resolve_gtcrn_model_path(gtcrn_model_override, model_path, sizeof(model_path))) {
            SherpaOnnxOnlineSpeechDenoiserConfig sd_cfg;
            memset(&sd_cfg, 0, sizeof(sd_cfg));
            sd_cfg.model.gtcrn.model = model_path;
            sd_cfg.model.num_threads = 1;
            sd_cfg.model.provider = "cpu";
            sd_cfg.model.debug = 0;
            st->gtcrn = SherpaOnnxCreateOnlineSpeechDenoiser(&sd_cfg);
            if (st->gtcrn) {
                /* Pre-warm ONNX Runtime arena and thread state BEFORE opening ALSA streams */
                float zero_pcm[P2P_FRAME_SAMPLES] = {0};
                for (int w = 0; w < 10; ++w) {
                    const SherpaOnnxDenoisedAudio *wu =
                        SherpaOnnxOnlineSpeechDenoiserRun(st->gtcrn, zero_pcm, P2P_FRAME_SAMPLES, P2P_SAMPLE_RATE);
                    if (wu) SherpaOnnxDestroyDenoisedAudio(wu);
                }
                SherpaOnnxOnlineSpeechDenoiserReset(st->gtcrn);
                printf("[DUT-Main] Streaming Online GTCRN ANS Active (Warmed, 1T): %s (shift=%d @ %dHz)\n",
                       model_path,
                       SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(st->gtcrn),
                       SherpaOnnxOnlineSpeechDenoiserGetSampleRate(st->gtcrn));
            } else {
                fprintf(stderr, "[Warning] Failed to create GTCRN denoiser from %s; falling back to RNNoise+Crossover.\n",
                        model_path);
            }
        } else {
            fprintf(stderr, "[Warning] gtcrn_simple.onnx not found; falling back to RNNoise+Crossover.\n");
        }
    } else if (ans_backend == ANS_BACKEND_RNNOISE || (!st->use_dfnet3 && !st->use_dpdfnet && !enable_gtcrn)) {
        printf("[DUT-Main] Neural post-ANS set to RNNoise fallback.\n");
    }

    /* 4. Neural RES (+ RNNoise fallback ANS only when DFNet3/DPDFNet/GTCRN are inactive) */
    NeuralConfig neural_cfg;
    NeuralRESANS_GetDefaultConfig(&neural_cfg);
    neural_cfg.min_gain_linear = 0.005f; /* -46dB deep silence floor */
    neural_cfg.res_aggressiveness = 1.25f;
    neural_cfg.ans_aggressiveness = 1.5f;
    neural_cfg.enable_neural_ans = (!st->use_dfnet3 && !st->use_dpdfnet && st->gtcrn == NULL);
    neural_cfg.enable_neural_res = true;
    if (NeuralRESANS_Init(&st->neural_engine, &neural_cfg) != 0) {
        fprintf(stderr, "[Error] Failed to initialize Neural RES engine!\n");
        return -1;
    }
    printf("[DUT-Main] Neural AI RES Engine initialized (%s)\n",
           st->use_dfnet3 ? "RES only; DeepFilterNet 3 handles ANS"
                          : (st->use_dpdfnet ? "RES only; DPDFNet handles ANS"
                                             : (st->gtcrn ? "RES only; GTCRN handles ANS" : "Deep GRU + RNNoise ANS fallback")));

    /* 5. Two-Stage Adaptive AGC:
     *    - Pre-ANS AGC lifts quiet far-field PDM speech into DFNet3/DPDFNet/GTCRN working band (-21 dBFS)
     *    - Post-ANS AGC runs on denoised speech (zero noise bias) to lock 1m~5m output to target dBFS
     */
    bool has_post_ans = (st->use_dfnet3 || st->use_dpdfnet || st->gtcrn != NULL);
    AGCConfig agc_cfg;
    AdaptiveAGC_GetDefaultConfig(&agc_cfg);
    agc_cfg.target_level_rms = has_post_ans ? 0.0891f : powf(10.0f, st->agc_target_db / 20.0f);
    agc_cfg.max_gain_db = has_post_ans ? fminf(18.0f, st->agc_max_boost_db) : st->agc_max_boost_db;
    agc_cfg.min_gain_db = -12.0f;
    agc_cfg.noise_gate_threshold = 0.0005f; /* -66 dBFS: catches 5m PDM speech pre-ANS */
    agc_cfg.attack_time_ms = 10.0f;
    agc_cfg.release_time_ms = 220.0f;
    agc_cfg.silence_standby_gain = has_post_ans ? 2.5f : 1.0f; /* +8 dB far-field readiness pre-ANS only */
    /* Keep reverb tails and background below the speech lift. Without this the
     * AGC applies its full speech gain to inter-word frames, which is what makes
     * the "ghost noise" audible (measured 0.8-4.2 dB MORE gain on non-speech). */
    agc_cfg.expand_range_db = 18.0f;
    agc_cfg.expand_ratio = 1.0f;
    agc_cfg.expand_max_cut_db = 18.0f;
    AdaptiveAGC_Init(&st->agc, &agc_cfg, P2P_SAMPLE_RATE);

    AGCConfig post_agc_cfg;
    AdaptiveAGC_GetDefaultConfig(&post_agc_cfg);
    post_agc_cfg.target_level_rms = powf(10.0f, st->agc_target_db / 20.0f);
    post_agc_cfg.max_gain_db = has_post_ans ? fmaxf(0.0f, st->agc_max_boost_db - 18.0f) : 0.0f;
    post_agc_cfg.min_gain_db = -12.0f;
    post_agc_cfg.noise_gate_threshold = 0.003f; /* -50.5 dBFS post-ANS: retains quiet far-field syllables */
    post_agc_cfg.attack_time_ms = 10.0f;
    post_agc_cfg.release_time_ms = 280.0f;
    post_agc_cfg.silence_standby_gain = 1.0f; /* Keep post-ANS silence floor deep black (<-85 dBFS) */
    post_agc_cfg.enable_eq = false; /* Pre-AGC already applied 85Hz HPF + presence EQ */
    post_agc_cfg.enable_limiter = true;
    AdaptiveAGC_Init(&st->post_agc, &post_agc_cfg, P2P_SAMPLE_RATE);

    printf("[DUT-Main] Calibrated AGC Initialized: Target=%.1f dBFS, TotalMaxBoost=+%.1f dB (%s)\n",
           st->agc_target_db, st->agc_max_boost_db,
           has_post_ans ? "Two-Stage Pre+Post ANS" : "Single-Stage Pre");

    /* 6. Harmonic-Preserving 2.2 kHz Transient Crossover Filter (post-AGC, pre-GTCRN) */
    TransientCrossover_Init(&st->post_tc, (float)P2P_SAMPLE_RATE, 2200.0f, 5.45f, 0.18f);
    st->post_tc.hold_duration = 4;
    st->post_tc.min_hp_gain = 0.08f;
    printf("[DUT-Main] Harmonic-Preserving 2.2kHz Crossover Suppressor Active\n");

    /* 7. Multi-Channel Online RLS-WPE + Spatial Dereverberation */
    st->enable_wpe = enable_wpe;
    if (st->enable_wpe && st->num_mics >= 2) {
        OnlineWpeConfig wpe_cfg;
        OnlineWpe_DefaultConfig(&wpe_cfg, st->num_mics);
        if (wpe_taps > 0) wpe_cfg.taps = wpe_taps;
        if (wpe_delay > 0) wpe_cfg.delay = wpe_delay;
        if (wpe_alpha > 0.0f) wpe_cfg.alpha = wpe_alpha;
        if (wpe_floor > 0.0f) wpe_cfg.power_floor = wpe_floor;
        if (OnlineWpe_Init(&st->wpe, &wpe_cfg) == 0) {
            printf("[DUT-Main] %d-Mic Online RLS-WPE Active (T=%d, D=%d, alpha=%.3f, floor=%.1e)\n",
                   st->wpe.config.num_mics, st->wpe.config.taps, st->wpe.config.delay,
                   st->wpe.config.alpha, st->wpe.config.power_floor);
        } else {
            fprintf(stderr, "[Warning] Failed to initialize OnlineWpe; disabling WPE.\n");
            st->enable_wpe = false;
        }
    } else {
        st->enable_wpe = false;
    }

    st->enable_dereverb = enable_dereverb;
    st->last_coherence = 1.0f;
    if (st->enable_dereverb && st->num_mics >= 2 && st->selected_mic_idx < 0) {
        SpatialDereverbConfig dr_cfg;
        dr_cfg.sample_rate = P2P_SAMPLE_RATE;
        dr_cfg.num_mics = st->num_mics;
        dr_cfg.min_gain = dereverb_floor > 0.0f ? dereverb_floor : 0.25f;
        dr_cfg.smooth_alpha_rise = 0.50f;
        dr_cfg.smooth_alpha_decay = 0.82f;
        dr_cfg.enabled = true;
        if (SpatialDereverb_Init(&st->dereverb, &dr_cfg) == 0) {
            printf("[DUT-Main] %d-Mic Blind Spatial Dereverberation Active (Floor=%.2f, DOA-free)\n",
                   st->num_mics, dr_cfg.min_gain);
        } else {
            st->enable_dereverb = false;
        }
    } else {
        st->enable_dereverb = false;
    }

    return 0;
}

static void dut_dsp_destroy(DUTDspState *st) {
    if (st->enable_wpe) {
        OnlineWpe_Destroy(&st->wpe);
        st->enable_wpe = false;
    }
    if (st->enable_dereverb) {
        SpatialDereverb_Reset(&st->dereverb);
    }
    if (st->use_dfnet3) {
        DFNet3ANS_Destroy(&st->dfnet3);
        st->use_dfnet3 = false;
    }
    if (st->use_dpdfnet) {
        DPDFNetANS_Destroy(&st->dpdfnet);
        st->use_dpdfnet = false;
    }
    if (st->gtcrn) {
        SherpaOnnxDestroyOnlineSpeechDenoiser(st->gtcrn);
        st->gtcrn = NULL;
    }
    NeuralRESANS_Destroy(&st->neural_engine);
    if (st->vad) {
        WebRtcVad_Free(st->vad);
        st->vad = NULL;
    }
    if (st->vad_near) {
        WebRtcVad_Free(st->vad_near);
        st->vad_near = NULL;
    }
}

static void dut_dsp_process_frame(DUTDspState *st,
                                  int16_t mic_ch[MAX_MIC_CHANNELS][P2P_FRAME_SAMPLES],
                                  const int16_t ref_signal[P2P_FRAME_SAMPLES],
                                  int16_t aec_error_fused[P2P_FRAME_SAMPLES],
                                  int16_t clean_out[P2P_FRAME_SAMPLES]) {
    int16_t ref_delayed[P2P_FRAME_SAMPLES];
    int16_t aec_error_ch[MAX_MIC_CHANNELS][P2P_FRAME_SAMPLES];
    int16_t echo_est_ch[MAX_MIC_CHANNELS][P2P_FRAME_SAMPLES];
    int16_t echo_est_fused[P2P_FRAME_SAMPLES];

    delay_line_step(&g_ref_delay_line, ref_signal, ref_delayed, g_cfg.delay_frames);
    bool has_ref = (g_ref_delay_line.total_pushed > g_cfg.delay_frames);
    st->e_ref_sum = 0.0;

    /* 0. Array Input High-Pass Filtering (removes 50Hz electrical hum & mechanical vibration) */
    if (st->enable_input_hpf) {
        for (int m = 0; m < st->num_mics; ++m) {
            for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                float filtered = mic_hpf_process(&st->mic_hpf[m], (float)mic_ch[m][i]);
                if (filtered > 32767.0f) filtered = 32767.0f;
                else if (filtered < -32768.0f) filtered = -32768.0f;
                mic_ch[m][i] = (int16_t)filtered;
            }
        }
        for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
            float filtered = mic_hpf_process(&st->ref_hpf, (float)ref_delayed[i]);
            if (filtered > 32767.0f) filtered = 32767.0f;
            else if (filtered < -32768.0f) filtered = -32768.0f;
            ref_delayed[i] = (int16_t)filtered;
        }
    }

    if (st->enable_aec && has_ref) {
        /* 1. Multi-AEC: independently cancel echo on each mic to preserve phase */
        for (int m = 0; m < st->num_mics; ++m) {
            LinearAEC_ProcessInt16(&st->aec[m], mic_ch[m], ref_delayed, aec_error_ch[m], echo_est_ch[m]);
        }
        stage_dump_write_ch_avg(STAGE_02_LINEAR_AEC, aec_error_ch, st->num_mics);

        /* 1b. Multi-channel Online RLS-WPE dereverberation before spatial fusion.
         *     Runs on all mics even in single-mic mode: the other channels still
         *     provide the prediction context, only the output channel is selected. */
        auto_select_update(st, mic_ch);
        if (st->enable_wpe && st->num_mics >= 2) {
            const int16_t *wpe_in[MAX_MIC_CHANNELS];
            int16_t *wpe_out[MAX_MIC_CHANNELS];
            for (int m = 0; m < st->num_mics; ++m) {
                wpe_in[m] = aec_error_ch[m];
                wpe_out[m] = aec_error_ch[m];
            }
            OnlineWpe_ProcessInt16(&st->wpe, wpe_in, wpe_out);
        }
        stage_dump_write_ch_avg(STAGE_03_WPE, aec_error_ch, st->num_mics);

        /* 2. Broadside Beamforming & Spatial Dereverberation across linear residual channels */
        double mic_fused_energy = 0.0;
        if (st->selected_mic_idx >= 0 && st->selected_mic_idx < st->num_mics) {
            for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                aec_error_fused[i] = aec_error_ch[st->selected_mic_idx][i];
                echo_est_fused[i] = echo_est_ch[st->selected_mic_idx][i];
                float mf = mic_ch[st->selected_mic_idx][i] / 32768.0f;
                mic_fused_energy += mf * mf;
            }
        } else if (st->enable_dereverb && st->num_mics >= 2) {
            const int16_t *err_ptrs[MAX_MIC_CHANNELS];
            for (int m = 0; m < st->num_mics; ++m) {
                err_ptrs[m] = aec_error_ch[m];
            }
            st->last_coherence = SpatialDereverb_ProcessInt16(&st->dereverb, err_ptrs, aec_error_fused);

            for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                int32_t est_sum = 0;
                int32_t mic_sum = 0;
                for (int m = 0; m < st->num_mics; ++m) {
                    est_sum += echo_est_ch[m][i];
                    mic_sum += mic_ch[m][i];
                }
                echo_est_fused[i] = (int16_t)(est_sum / st->num_mics);
                float mf = (mic_sum / st->num_mics) / 32768.0f;
                mic_fused_energy += mf * mf;
            }
        } else {
            for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                int32_t err_sum = 0;
                int32_t est_sum = 0;
                int32_t mic_sum = 0;
                for (int m = 0; m < st->num_mics; ++m) {
                    err_sum += aec_error_ch[m][i];
                    est_sum += echo_est_ch[m][i];
                    mic_sum += mic_ch[m][i];
                }
                aec_error_fused[i] = (int16_t)(err_sum / st->num_mics);
                echo_est_fused[i] = (int16_t)(est_sum / st->num_mics);
                float mf = (mic_sum / st->num_mics) / 32768.0f;
                mic_fused_energy += mf * mf;
            }
        }

        /* 3. Double-talk Detection across all mics */
        for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
            float r = ref_delayed[i] / 32768.0f;
            st->e_ref_sum += r * r;
        }

        double err_fused_energy = 0.0;
        double err_hpf_energy = 0.0;
        int16_t vad_hpf[P2P_FRAME_SAMPLES];
        for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
            float e = aec_error_fused[i] / 32768.0f;
            err_fused_energy += e * e;

            /* Strip DC & <100 Hz fan/mic hum before VAD & near-end SNR check */
            float x = (float)aec_error_fused[i];
            float y = 0.96f * (st->vad_hpf_y + x - st->vad_hpf_x);
            st->vad_hpf_x = x;
            st->vad_hpf_y = y;
            float clamped = y;
            if (clamped > 32767.0f) clamped = 32767.0f;
            else if (clamped < -32768.0f) clamped = -32768.0f;
            vad_hpf[i] = (int16_t)clamped;
            float eh = clamped / 32768.0f;
            err_hpf_energy += eh * eh;
        }
        err_fused_energy /= (double)P2P_FRAME_SAMPLES;
        err_hpf_energy /= (double)P2P_FRAME_SAMPLES;

        float fused_erle = 10.0f * log10f(((float)mic_fused_energy + 1e-9f) /
                                          ((float)(err_fused_energy * P2P_FRAME_SAMPLES) + 1e-9f));
        float ratio = (float)(mic_fused_energy / (st->e_ref_sum + 1e-9));

        int is_vad_active = (st->vad && WebRtcVad_Process(st->vad, 16000, vad_hpf, 160) == 1) ? 1 : 0;
        if (!is_vad_active && st->vad) {
            if (WebRtcVad_Process(st->vad, 16000, &vad_hpf[96], 160) == 1) {
                is_vad_active = 1;
            }
        }
        int is_vad_near = (st->vad_near && WebRtcVad_Process(st->vad_near, 16000, vad_hpf, 160) == 1) ? 1 : 0;
        if (!is_vad_near && st->vad_near) {
            if (WebRtcVad_Process(st->vad_near, 16000, &vad_hpf[96], 160) == 1) {
                is_vad_near = 1;
            }
        }

        int dt_mic_count = 0;
        for (int m = 0; m < st->num_mics; ++m) {
            if (LinearAEC_IsDoubleTalk(&st->aec[m])) dt_mic_count++;
        }

        st->ref_energy_env = fmax(st->e_ref_sum, st->ref_energy_env * 0.85);
        float env_ratio = (float)(mic_fused_energy / (st->ref_energy_env + 1e-9));

        if (st->ref_energy_env < 0.02) {
            if (st->frame_idx < 25) {
                if (!is_vad_near) {
                    st->near_noise_energy = 0.80 * st->near_noise_energy + 0.20 * err_hpf_energy;
                }
            } else if (err_hpf_energy < st->near_noise_energy) {
                st->near_noise_energy = 0.85 * st->near_noise_energy + 0.15 * err_hpf_energy;
            } else if (!is_vad_near) {
                st->near_noise_energy = 0.98 * st->near_noise_energy + 0.02 * err_hpf_energy;
            } else {
                st->near_noise_energy = 0.999 * st->near_noise_energy + 0.001 * err_hpf_energy;
            }
        }

        int dt_req = (st->num_mics >= 8) ? 2 : 1;
        bool is_dt_onset = false;
        if (st->ref_energy_env < 0.02) {
            /* Pure near-end (no active far-end nor recent reverb tail): HPF-cleaned SNR + Mode 2 VAD */
            double onset_energy = fmax(8.0e-7, st->near_noise_energy * 2.0);
            if (st->frame_idx >= 15 && is_vad_near && err_hpf_energy > onset_energy) {
                st->dt_onset_streak++;
            } else {
                st->dt_onset_streak = 0;
            }
            if (st->dt_onset_streak >= 2) {
                is_dt_onset = true;
            }
        } else {
            bool raw_dt = false;
            if (st->e_ref_sum < 0.02) {
                /* Far-end paused within decaying echo envelope window */
                raw_dt = (is_vad_active && err_fused_energy > 1.2e-3 && env_ratio > 0.18f);
            } else {
                /* Active far-end double-talk */
                raw_dt = (is_vad_active && fused_erle < 4.0f && err_fused_energy > 1.2e-3 &&
                          env_ratio > 0.15f &&
                          ((dt_mic_count >= dt_req && ratio > 0.18f) || (ratio > 0.55f && dt_mic_count >= 1)));
            }
            if (raw_dt) {
                st->dt_onset_streak++;
            } else {
                st->dt_onset_streak = 0;
            }
            if (st->dt_onset_streak >= 2) {
                is_dt_onset = true;
            }
        }

        /* VAD/DTD onset guard for high-crest impulses (leave spectrum intact for post-AGC crossover) */
        float pk_fused = 0.0f;
        for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
            float v = fabsf((float)aec_error_fused[i] / 32768.0f);
            if (v > pk_fused) pk_fused = v;
        }
        float cr_fused = pk_fused / (sqrtf((float)err_fused_energy) + 1e-6f);
        bool is_transient_hit = (cr_fused > 5.0f && pk_fused > 0.06f);
        if (st->debug_transient && is_transient_hit) {
            printf("[Transient-Pre] t=%.2fs cr=%.2f pk=%.3f\n", st->frame_idx * 0.016f, cr_fused, pk_fused);
        }

        if (is_dt_onset && !is_transient_hit) {
            st->hangover_counter = HANGOVER_FRAMES_MAX;
        } else if (st->hangover_counter > 0) {
            bool continue_speech = false;
            if (st->ref_energy_env < 0.02) {
                double cont_energy = fmax(5.0e-7, st->near_noise_energy * 1.5);
                if (is_vad_near && !is_transient_hit && err_hpf_energy > cont_energy) continue_speech = true;
            } else {
                bool aec_or_pause = (st->e_ref_sum < 0.02) ||
                                    (fused_erle < 4.0f && (dt_mic_count >= dt_req || ratio > 0.28f));
                if (is_vad_active && !is_transient_hit && env_ratio > 0.15f && err_fused_energy > 4.0e-4 && aec_or_pause) {
                    continue_speech = true;
                }
            }
            if (continue_speech) {
                st->hangover_counter = HANGOVER_FRAMES_MAX;
            } else {
                st->hangover_counter--;
            }
        }

        /* 4. Neural RES (+ RNNoise fallback only if GTCRN inactive) */
        stage_dump_write(STAGE_04_SPATIAL_FUSION, aec_error_fused, P2P_FRAME_SAMPLES);
        NeuralRESANS_SetVAD(&st->neural_engine, (st->hangover_counter > 0) ? 1 : 0);
        NeuralRESANS_ProcessInt16(&st->neural_engine, aec_error_fused, echo_est_fused, ref_delayed, clean_out);
        stage_dump_write(STAGE_05_NEURAL_RES, clean_out, P2P_FRAME_SAMPLES);
    } else {
        /* Bypass AEC */
        const int16_t *arr_ptrs[MAX_MIC_CHANNELS];
        int16_t wpe_ch[MAX_MIC_CHANNELS][P2P_FRAME_SAMPLES];
        stage_dump_write_ch_avg(STAGE_02_LINEAR_AEC, mic_ch, st->num_mics);
        auto_select_update(st, mic_ch);
        if (st->enable_wpe && st->num_mics >= 2) {
            const int16_t *wpe_in[MAX_MIC_CHANNELS];
            int16_t *wpe_out[MAX_MIC_CHANNELS];
            for (int m = 0; m < st->num_mics; ++m) {
                wpe_in[m] = mic_ch[m];
                wpe_out[m] = wpe_ch[m];
                arr_ptrs[m] = wpe_ch[m];
            }
            OnlineWpe_ProcessInt16(&st->wpe, wpe_in, wpe_out);
            stage_dump_write_ch_avg(STAGE_03_WPE, wpe_ch, st->num_mics);
        } else {
            for (int m = 0; m < st->num_mics; ++m) {
                arr_ptrs[m] = mic_ch[m];
            }
            stage_dump_write_ch_avg(STAGE_03_WPE, mic_ch, st->num_mics);
        }

        if (st->selected_mic_idx >= 0 && st->selected_mic_idx < st->num_mics) {
            /* arr_ptrs already points at the WPE output when WPE is active */
            for (int i = 0; i < P2P_FRAME_SAMPLES; ++i)
                aec_error_fused[i] = arr_ptrs[st->selected_mic_idx][i];
        } else if (st->enable_dereverb && st->num_mics >= 2) {
            st->last_coherence = SpatialDereverb_ProcessInt16(&st->dereverb, arr_ptrs, aec_error_fused);
        } else {
            for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                int32_t sum = 0;
                for (int m = 0; m < st->num_mics; ++m) sum += arr_ptrs[m][i];
                aec_error_fused[i] = (int16_t)(sum / st->num_mics);
            }
        }
        stage_dump_write(STAGE_04_SPATIAL_FUSION, aec_error_fused, P2P_FRAME_SAMPLES);
        if (st->vad && WebRtcVad_Process(st->vad, 16000, aec_error_fused, 160) == 1) {
            st->hangover_counter = HANGOVER_FRAMES_MAX;
        } else if (st->hangover_counter > 0) {
            st->hangover_counter--;
        }
        NeuralRESANS_SetVAD(&st->neural_engine, (st->hangover_counter > 0) ? 1 : 0);
        NeuralRESANS_ProcessInt16(&st->neural_engine, aec_error_fused, NULL, NULL, clean_out);
        stage_dump_write(STAGE_05_NEURAL_RES, clean_out, P2P_FRAME_SAMPLES);
    }

    /* 5. Stage-1 Adaptive AGC (Conditions speech into DFNet3/DPDFNet/GTCRN; caps Double-Talk pumping) */
    bool has_post_ans = (st->use_dfnet3 || st->use_dpdfnet || st->gtcrn != NULL);
    float pre_max_boost = (st->ref_energy_env >= 0.02)
                              ? (has_post_ans ? 4.0f : 8.0f)
                              : (has_post_ans ? fminf(18.0f, st->agc_max_boost_db) : st->agc_max_boost_db);
    st->agc.max_gain_linear = powf(10.0f, pre_max_boost / 20.0f);
    AdaptiveAGC_SetVAD(&st->agc, (st->hangover_counter > 0) ? 1 : 0);
    AdaptiveAGC_ProcessInt16(&st->agc, clean_out, clean_out, P2P_FRAME_SAMPLES);
    stage_dump_write(STAGE_06_PRE_AGC, clean_out, P2P_FRAME_SAMPLES);

    /* 6. Post-AGC Harmonic-Preserving Transient Crossover + Causal Streaming ANS */
    bool post_hit = TransientCrossover_ProcessInt16(&st->post_tc, clean_out, P2P_FRAME_SAMPLES);
    if (st->debug_transient && post_hit) {
        printf("[Transient-Post] t=%.2fs frame=%u\n", st->frame_idx * 0.016f, st->frame_idx);
    }
    stage_dump_write(STAGE_07_TRANSIENT_XOVER, clean_out, P2P_FRAME_SAMPLES);

    if (st->use_dfnet3) {
        DFNet3ANS_ProcessInt16(&st->dfnet3, clean_out, clean_out, P2P_FRAME_SAMPLES);
    } else if (st->use_dpdfnet) {
        DPDFNetANS_ProcessInt16(&st->dpdfnet, clean_out, clean_out, P2P_FRAME_SAMPLES);
    } else if (st->gtcrn) {
        float gtcrn_in[P2P_FRAME_SAMPLES];
        for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
            gtcrn_in[i] = (float)clean_out[i] / 32768.0f;
        }
        const SherpaOnnxDenoisedAudio *denoised =
            SherpaOnnxOnlineSpeechDenoiserRun(st->gtcrn, gtcrn_in, P2P_FRAME_SAMPLES, P2P_SAMPLE_RATE);
        if (denoised) {
            if (denoised->n >= P2P_FRAME_SAMPLES && denoised->samples) {
                for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                    float v = denoised->samples[i] * 32767.0f;
                    if (v > 32767.0f) v = 32767.0f;
                    else if (v < -32768.0f) v = -32768.0f;
                    clean_out[i] = (int16_t)v;
                }
            } else {
                /* First 16ms STFT OLA warmup frame emits 0 samples */
                memset(clean_out, 0, sizeof(int16_t) * P2P_FRAME_SAMPLES);
            }
            SherpaOnnxDestroyDenoisedAudio(denoised);
        }
    }
    stage_dump_write(STAGE_08_POST_ANS, clean_out, P2P_FRAME_SAMPLES);

    /* 6b. Stage-2 Post-ANS Broadcast AGC (runs on denoised speech -> levels 1m~5m accurately) */
    if (has_post_ans) {
        float post_max_boost = (st->ref_energy_env >= 0.02)
                                   ? 5.0f
                                   : fmaxf(0.0f, st->agc_max_boost_db - 18.0f);
        st->post_agc.max_gain_linear = powf(10.0f, post_max_boost / 20.0f);
        AdaptiveAGC_SetVAD(&st->post_agc, (st->hangover_counter > 0) ? 1 : 0);
        AdaptiveAGC_ProcessInt16(&st->post_agc, clean_out, clean_out, P2P_FRAME_SAMPLES);
    }
    stage_dump_write(STAGE_09_POST_AGC, clean_out, P2P_FRAME_SAMPLES);

    /* 7. Soft Exponential Gain Smoother
     *    Keep pure near-end idle floor gentle (-12 dB) when neural post-ANS is active
     *    so weak syllable onsets/tails aren't truncated, while keeping active echo at -40 dB. */
    float near_idle_gate = has_post_ans ? 0.25f : 0.05f;
    float idle_gate = (st->ref_energy_env < 0.02) ? near_idle_gate : 0.01f;
    float target_gate = (st->hangover_counter > 0) ? 1.0f : idle_gate;
    float alpha = (target_gate > st->smooth_gate) ? 0.35f : 0.03f;
    st->smooth_gate = (1.0f - alpha) * st->smooth_gate + alpha * target_gate;
    for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
        clean_out[i] = (int16_t)(clean_out[i] * st->smooth_gate);
    }
    stage_dump_write(STAGE_10_SMOOTH_GATE, clean_out, P2P_FRAME_SAMPLES);
    st->frame_idx++;
}

static void* spk_thread_func(void *arg) {
    (void)arg;
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("[DUT-Spk] Socket failed");
        return NULL;
    }

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in bind_addr;
    memset(&bind_addr, 0, sizeof(bind_addr));
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = htons(g_cfg.listen_port);
    bind_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
        perror("[DUT-Spk] Bind failed");
        close(sock);
        return NULL;
    }

    struct timeval tv = { .tv_sec = 0, .tv_usec = 100000 };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    printf("[DUT-Spk] Listening on UDP 0.0.0.0:%d for Far-end audio...\n", g_cfg.listen_port);

    P2PPacket pkt;
    while (g_running) {
        struct sockaddr_in from_addr;
        socklen_t from_len = sizeof(from_addr);
        ssize_t n = recvfrom(sock, &pkt, sizeof(pkt), 0, (struct sockaddr*)&from_addr, &from_len);
        if (n <= 0) continue;

        if (pkt.magic != P2P_MAGIC) continue;

        g_last_far_timestamp_ms = pkt.timestamp_ms;

        if (!g_cfg.peer_known || from_addr.sin_addr.s_addr != g_cfg.peer_addr.sin_addr.s_addr) {
            memcpy(&g_cfg.peer_addr, &from_addr, sizeof(from_addr));
            g_cfg.peer_addr.sin_port = htons(g_cfg.send_port);
            inet_ntop(AF_INET, &from_addr.sin_addr, g_cfg.peer_ip, sizeof(g_cfg.peer_ip));
            g_cfg.peer_known = true;
            printf("[DUT-Spk] Connected to Far-End Peer: %s:%d\n", g_cfg.peer_ip, g_cfg.send_port);
        }

        net_fifo_push(&g_net_fifo, pkt.pcm);
    }

    close(sock);
    return NULL;
}

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    memset(&g_cfg, 0, sizeof(g_cfg));
    g_cfg.listen_port = P2P_PORT_TO_DUT;
    g_cfg.send_port = P2P_PORT_FROM_DUT;
    g_cfg.delay_frames = 0;   /* Default 0 for hardware soft loopback */
    g_cfg.enable_aec = true;
    strncpy(g_cfg.mic_device, "hw:3,0", sizeof(g_cfg.mic_device) - 1);
    g_cfg.mic_channels = 10;  /* Default: 10ch (8 Mics + 2 Loopback) */
    g_cfg.ref_channel = 8;    /* Default: Ch8 is Primary Loopback Ref */
    strncpy(g_cfg.spk_device, "hw:3,0", sizeof(g_cfg.spk_device) - 1);
    g_cfg.inject_delay_sec = 0;
    g_cfg.mic_pregain = 8.0f; /* 8.0x (+18dB) uniform fallback when --no-calib-table */
    g_cfg.use_calib_table = true; /* Default true: equalizes 8-mic physical sensitivity */
    g_cfg.enable_input_hpf = true;
    g_cfg.hpf_cutoff_hz = 80.0f; /* 80Hz 2nd-order HPF: suppresses 50Hz hum while preserving 85-150Hz voice */
    g_cfg.enable_gtcrn = true;
    g_cfg.ans_backend = ANS_BACKEND_AUTO;
    g_cfg.dfnet_atten_lim = 100.0f;
    g_cfg.dfnet_pf_beta = 0.02f;
    g_cfg.dpdfnet_threads = 1;
    g_cfg.enable_dereverb = true;
    g_cfg.dereverb_floor = 0.25f; /* -12 dB suppression floor */
    g_cfg.enable_wpe = false;
    g_cfg.wpe_taps = 4;
    g_cfg.wpe_delay = 2;
    g_cfg.wpe_alpha = 0.99f;
    g_cfg.wpe_floor = 1e-4f;
    g_cfg.agc_target_db = -17.0f;
    g_cfg.agc_max_boost_db = 40.0f;
    g_cfg.enable_nvidia = false;
    g_cfg.nvidia_core_mask = 7;
    g_cfg.nvidia_dry_gain = 0.0f;

    float custom_gains[MAX_MIC_CHANNELS];
    bool has_custom_gains = false;
    bool explicit_gain_override = false;
    int selected_mic_idx = -1; /* -1: all mics array fusion; >=0: test specific single mic */

    const char *raw_multi_path = NULL;
    const char *raw_clean_path = NULL;
    const char *raw_err_path = NULL;
    int record_sec = 0;
    bool record_exit = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--peer-ip") == 0 && i + 1 < argc) {
            strncpy(g_cfg.peer_ip, argv[++i], sizeof(g_cfg.peer_ip) - 1);
            inet_pton(AF_INET, g_cfg.peer_ip, &g_cfg.peer_addr.sin_addr);
            g_cfg.peer_addr.sin_family = AF_INET;
            g_cfg.peer_addr.sin_port = htons(g_cfg.send_port);
            g_cfg.peer_known = true;
        } else if (strcmp(argv[i], "--delay-frames") == 0 && i + 1 < argc) {
            g_cfg.delay_frames = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bypass") == 0) {
            g_cfg.enable_aec = false;
        } else if (strcmp(argv[i], "--aec") == 0) {
            g_cfg.enable_aec = true;
        } else if (strcmp(argv[i], "--inject-speech") == 0 && i + 1 < argc) {
            strncpy(g_cfg.inject_wav_path, argv[++i], sizeof(g_cfg.inject_wav_path) - 1);
        } else if (strcmp(argv[i], "--inject-delay-sec") == 0 && i + 1 < argc) {
            g_cfg.inject_delay_sec = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--record-raw") == 0 ||
                    strcmp(argv[i], "--record-10ch") == 0 ||
                    strcmp(argv[i], "--record-8ch") == 0 ||
                    strcmp(argv[i], "--record-6ch") == 0) && i + 1 < argc) {
            raw_multi_path = argv[++i];
        } else if (strcmp(argv[i], "--record-err") == 0 && i + 1 < argc) {
            raw_err_path = argv[++i];
        } else if (strcmp(argv[i], "--record-clean") == 0 && i + 1 < argc) {
            raw_clean_path = argv[++i];
        } else if (strcmp(argv[i], "--mic-device") == 0 && i + 1 < argc) {
            strncpy(g_cfg.mic_device, argv[++i], sizeof(g_cfg.mic_device) - 1);
        } else if (strcmp(argv[i], "--mic-channels") == 0 && i + 1 < argc) {
            g_cfg.mic_channels = atoi(argv[++i]);
            if (g_cfg.mic_channels == 10) g_cfg.ref_channel = 8;
            else if (g_cfg.mic_channels == 6) g_cfg.ref_channel = 4;
            else if (g_cfg.mic_channels == 8) g_cfg.ref_channel = -1;
        } else if (strcmp(argv[i], "--ref-channel") == 0 && i + 1 < argc) {
            g_cfg.ref_channel = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mic-pregain") == 0 && i + 1 < argc) {
            g_cfg.mic_pregain = (float)atof(argv[++i]);
            g_cfg.use_calib_table = false;
            explicit_gain_override = true;
        } else if (strcmp(argv[i], "--calib-table") == 0) {
            g_cfg.use_calib_table = true;
            explicit_gain_override = true;
        } else if (strcmp(argv[i], "--no-calib-table") == 0) {
            g_cfg.use_calib_table = false;
            explicit_gain_override = true;
        } else if (strcmp(argv[i], "--hpf-cutoff") == 0 && i + 1 < argc) {
            g_cfg.hpf_cutoff_hz = (float)atof(argv[++i]);
            g_cfg.enable_input_hpf = (g_cfg.hpf_cutoff_hz > 10.0f);
        } else if (strcmp(argv[i], "--no-hpf") == 0) {
            g_cfg.enable_input_hpf = false;
            g_cfg.hpf_cutoff_hz = 0.0f;
        } else if (strcmp(argv[i], "--mic-gains") == 0 && i + 1 < argc) {
            char *gstr = strdup(argv[++i]);
            char *token = strtok(gstr, ",");
            int c = 0;
            while (token && c < MAX_MIC_CHANNELS) {
                custom_gains[c++] = (float)atof(token);
                token = strtok(NULL, ",");
            }
            free(gstr);
            has_custom_gains = (c > 0);
            explicit_gain_override = true;
        } else if (strcmp(argv[i], "--select-mic") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            selected_mic_idx = (strcmp(v, "auto") == 0) ? AUTO_SELECT_MIC : atoi(v);
        } else if (strcmp(argv[i], "--spk-device") == 0 && i + 1 < argc) {
            strncpy(g_cfg.spk_device, argv[++i], sizeof(g_cfg.spk_device) - 1);
        } else if (strcmp(argv[i], "--record-sec") == 0 && i + 1 < argc) {
            record_sec = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--record-exit") == 0) {
            record_exit = true;
        } else if (strcmp(argv[i], "--ans") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "dfnet3") == 0) g_cfg.ans_backend = ANS_BACKEND_DFNET3;
            else if (strcmp(mode, "dpdfnet") == 0) g_cfg.ans_backend = ANS_BACKEND_DPDFNET;
            else if (strcmp(mode, "gtcrn") == 0) g_cfg.ans_backend = ANS_BACKEND_GTCRN;
            else if (strcmp(mode, "rnnoise") == 0) {
                g_cfg.ans_backend = ANS_BACKEND_RNNOISE;
                g_cfg.enable_gtcrn = false;
            } else {
                g_cfg.ans_backend = ANS_BACKEND_AUTO;
            }
        } else if (strcmp(argv[i], "--dfnet-model") == 0 && i + 1 < argc) {
            strncpy(g_cfg.dfnet_model_path, argv[++i], sizeof(g_cfg.dfnet_model_path) - 1);
            if (g_cfg.ans_backend == ANS_BACKEND_AUTO) g_cfg.ans_backend = ANS_BACKEND_DFNET3;
        } else if (strcmp(argv[i], "--dfnet-atten-lim") == 0 && i + 1 < argc) {
            g_cfg.dfnet_atten_lim = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--dfnet-pf-beta") == 0 && i + 1 < argc) {
            g_cfg.dfnet_pf_beta = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--dpdfnet-model") == 0 && i + 1 < argc) {
            strncpy(g_cfg.dpdfnet_model_path, argv[++i], sizeof(g_cfg.dpdfnet_model_path) - 1);
            g_cfg.ans_backend = ANS_BACKEND_DPDFNET;
        } else if (strcmp(argv[i], "--dpdfnet-threads") == 0 && i + 1 < argc) {
            g_cfg.dpdfnet_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gtcrn-model") == 0 && i + 1 < argc) {
            strncpy(g_cfg.gtcrn_model_path, argv[++i], sizeof(g_cfg.gtcrn_model_path) - 1);
            if (g_cfg.ans_backend == ANS_BACKEND_AUTO) g_cfg.ans_backend = ANS_BACKEND_GTCRN;
        } else if (strcmp(argv[i], "--no-gtcrn") == 0) {
            g_cfg.enable_gtcrn = false;
            g_cfg.ans_backend = ANS_BACKEND_RNNOISE;
        } else if (strcmp(argv[i], "--no-dereverb") == 0) {
            g_cfg.enable_dereverb = false;
        } else if (strcmp(argv[i], "--dereverb") == 0) {
            g_cfg.enable_dereverb = true;
        } else if (strcmp(argv[i], "--dereverb-floor") == 0 && i + 1 < argc) {
            g_cfg.dereverb_floor = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--wpe") == 0) {
            g_cfg.enable_wpe = true;
        } else if (strcmp(argv[i], "--no-wpe") == 0) {
            g_cfg.enable_wpe = false;
        } else if (strcmp(argv[i], "--wpe-taps") == 0 && i + 1 < argc) {
            g_cfg.wpe_taps = atoi(argv[++i]);
            g_cfg.enable_wpe = true;
        } else if (strcmp(argv[i], "--wpe-delay") == 0 && i + 1 < argc) {
            g_cfg.wpe_delay = atoi(argv[++i]);
            g_cfg.enable_wpe = true;
        } else if (strcmp(argv[i], "--wpe-alpha") == 0 && i + 1 < argc) {
            g_cfg.wpe_alpha = (float)atof(argv[++i]);
            g_cfg.enable_wpe = true;
        } else if (strcmp(argv[i], "--wpe-floor") == 0 && i + 1 < argc) {
            g_cfg.wpe_floor = (float)atof(argv[++i]);
            g_cfg.enable_wpe = true;
        } else if (strcmp(argv[i], "--agc-target-db") == 0 && i + 1 < argc) {
            g_cfg.agc_target_db = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--agc-max-boost") == 0 && i + 1 < argc) {
            g_cfg.agc_max_boost_db = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--debug-transient") == 0) {
            g_cfg.debug_transient = true;
        } else if (strcmp(argv[i], "--nvidia") == 0 || strcmp(argv[i], "--enable-nvidia") == 0) {
            g_cfg.enable_nvidia = true;
        } else if (strcmp(argv[i], "--no-nvidia") == 0) {
            g_cfg.enable_nvidia = false;
        } else if (strcmp(argv[i], "--nvidia-rknn") == 0 && i + 1 < argc) {
            strncpy(g_cfg.nvidia_model_path, argv[++i], sizeof(g_cfg.nvidia_model_path) - 1);
            g_cfg.enable_nvidia = true;
        } else if (strcmp(argv[i], "--nvidia-bin") == 0 && i + 1 < argc) {
            strncpy(g_cfg.nvidia_bin_path, argv[++i], sizeof(g_cfg.nvidia_bin_path) - 1);
            g_cfg.enable_nvidia = true;
        } else if (strcmp(argv[i], "--nvidia-core-mask") == 0 && i + 1 < argc) {
            g_cfg.nvidia_core_mask = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nvidia-dry-gain") == 0 && i + 1 < argc) {
            g_cfg.nvidia_dry_gain = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--offline-wav") == 0 && i + 2 < argc) {
            strncpy(g_cfg.offline_in_wav, argv[++i], sizeof(g_cfg.offline_in_wav) - 1);
            strncpy(g_cfg.offline_out_wav, argv[++i], sizeof(g_cfg.offline_out_wav) - 1);
        } else if (strcmp(argv[i], "--dump-stages") == 0 && i + 1 < argc) {
            strncpy(g_cfg.dump_stages_dir, argv[++i], sizeof(g_cfg.dump_stages_dir) - 1);
        }
    }

    if (g_cfg.offline_in_wav[0] != '\0' && !explicit_gain_override) {
        /* Raw recorded WAVs (/tmp/dut_raw10ch.wav) already store post-calibrated PCM */
        g_cfg.mic_pregain = 1.0f;
        g_cfg.use_calib_table = false;
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    int num_mics = (g_cfg.mic_channels == 10) ? 8 :
                   ((g_cfg.mic_channels == 6) ? 4 :
                   (g_cfg.ref_channel >= 0 ? g_cfg.ref_channel : g_cfg.mic_channels));
    if (num_mics < 1) num_mics = 1;
    if (num_mics > MAX_MIC_CHANNELS) num_mics = MAX_MIC_CHANNELS;

    printf("==============================================================\n");
    printf("  P2P AI Echo Cancellation DUT Node (RK3588 %d-Mic Array)\n", num_mics);
    printf("  Microphone Array:  %d Mics\n", num_mics);
    printf("  Ref Channel:       %s (Ch%d)\n", (g_cfg.ref_channel >= 0) ? "Hardware Loopback" : "Network Stream", g_cfg.ref_channel);
    printf("  AEC Mode:          %s\n", g_cfg.enable_aec ? "[ENABLED] (Multi-AEC + Neural RES)" : "[BYPASS]");
    printf("  Mic Device:        %s (%d channels total)\n", g_cfg.mic_device, g_cfg.mic_channels);
    printf("  Speaker Device:    %s\n", g_cfg.spk_device);
    if (selected_mic_idx >= 0) {
        printf("  Mic Mode:          [SINGLE MIC TEST MODE] (Only Ch%d active)\n", selected_mic_idx);
    } else {
        printf("  Mic Mode:          [8-MIC BEAMFORMING ARRAY]\n");
    }
    if (has_custom_gains) {
        printf("  Pre-Gain:          Custom Per-Mic Gains Table\n");
    } else if (g_cfg.use_calib_table && num_mics == 8) {
        printf("  Pre-Gain:          Auto-Calibrated 8-Mic EQ Table (290/84/81/80/89/250/92/88x)\n");
    } else {
        printf("  Pre-Gain:          %.1fx (Uniform)\n", g_cfg.mic_pregain);
    }
    printf("  Input HPF Mode:    %s (Cutoff=%.1f Hz, 2nd-order Butterworth)\n",
           g_cfg.enable_input_hpf ? "[ENABLED]" : "[DISABLED]", g_cfg.hpf_cutoff_hz);
    printf("  Pre-Delay Offset:  %d frames (%d ms)\n", g_cfg.delay_frames, g_cfg.delay_frames * 16);
    printf("  Online WPE Mode:   %s (T=%d, D=%d, alpha=%.3f, floor=%.1e)\n",
           g_cfg.enable_wpe ? "[ENABLED] (Multi-Ch Causal RLS-WPE)" : "[DISABLED]",
           g_cfg.wpe_taps, g_cfg.wpe_delay, g_cfg.wpe_alpha, g_cfg.wpe_floor);
    printf("  Dereverb Mode:     %s (Floor=%.2f, DOA-free)\n",
           g_cfg.enable_dereverb ? "[ENABLED] (Blind Spatial Coherence Filter)" : "[DISABLED]",
           g_cfg.dereverb_floor);
    printf("  Nvidia Dereverb:   %s (CoreMask=%d, DryGain=%.3f)\n",
           g_cfg.enable_nvidia ? "[ENABLED] (NPU Single-Step ODE Dereverb)" : "[DISABLED]",
           g_cfg.nvidia_core_mask, g_cfg.nvidia_dry_gain);
    printf("==============================================================\n\n");

    net_fifo_init(&g_net_fifo);
    delay_line_init(&g_ref_delay_line);

    DUTDspState dsp;
    if (dut_dsp_init(&dsp, num_mics, selected_mic_idx, g_cfg.enable_aec,
                     g_cfg.ans_backend, g_cfg.enable_gtcrn, g_cfg.gtcrn_model_path,
                     g_cfg.dfnet_model_path, g_cfg.dfnet_atten_lim, g_cfg.dfnet_pf_beta,
                     g_cfg.dpdfnet_model_path, g_cfg.dpdfnet_threads,
                     g_cfg.debug_transient,
                     g_cfg.enable_dereverb, g_cfg.dereverb_floor,
                     g_cfg.enable_wpe, g_cfg.wpe_taps, g_cfg.wpe_delay,
                     g_cfg.wpe_alpha, g_cfg.wpe_floor,
                     g_cfg.agc_target_db, g_cfg.agc_max_boost_db,
                     g_cfg.enable_input_hpf, g_cfg.hpf_cutoff_hz) != 0) {
        return 1;
    }

    /* Optional Offline Multi-Channel WAV Mode (shares identical DSP pipeline with live P2P) */
    if (g_cfg.offline_in_wav[0] != '\0') {
        WavFile *in_wav = WavFile_OpenRead(g_cfg.offline_in_wav);
        if (!in_wav) {
            fprintf(stderr, "[Error] Cannot open offline input WAV: %s\n", g_cfg.offline_in_wav);
            dut_dsp_destroy(&dsp);
            return 1;
        }
        WavFile *out_wav = WavFile_OpenWrite(g_cfg.offline_out_wav, 1, P2P_SAMPLE_RATE);
        if (!out_wav) {
            fprintf(stderr, "[Error] Cannot open offline output WAV: %s\n", g_cfg.offline_out_wav);
            WavFile_Close(in_wav);
            dut_dsp_destroy(&dsp);
            return 1;
        }
        int in_ch = in_wav->channels;
        int16_t raw_buf[P2P_FRAME_SAMPLES * MAX_MIC_CHANNELS];
        int16_t mic_ch[MAX_MIC_CHANNELS][P2P_FRAME_SAMPLES];
        int16_t ref_sig[P2P_FRAME_SAMPLES];
        int16_t aec_err[P2P_FRAME_SAMPLES];
        int16_t clean_out[P2P_FRAME_SAMPLES];
        float gain_tbl[MAX_MIC_CHANNELS];
        init_mic_gains(gain_tbl, num_mics, has_custom_gains, custom_gains);
        if (g_cfg.dump_stages_dir[0] != '\0') {
            stage_dump_open(g_cfg.dump_stages_dir);
        }

        if (g_cfg.enable_nvidia) {
            char nv_model[512] = "", nv_bin[512] = "";
            if (!resolve_nvidia_model_path(g_cfg.nvidia_model_path, nv_model, sizeof(nv_model)) ||
                !resolve_nvidia_bin_path(g_cfg.nvidia_bin_path, nv_bin, sizeof(nv_bin))) {
                fprintf(stderr, "[Warning] Nvidia model or bin not found, bypassing Nvidia stage\n");
                g_cfg.enable_nvidia = false;
            } else {
                nvidia_bridge_start(&g_nv_bridge, nv_model, nv_bin,
                                    g_cfg.nvidia_core_mask, g_cfg.nvidia_dry_gain,
                                    true, -1, NULL, out_wav);
            }
        }

        size_t cap_frames = 32768;
        double *frame_ms = (double *)malloc(cap_frames * sizeof(double));
        double sum_ms = 0.0;
        double max_ms = 0.0;
        uint32_t over16 = 0;
        uint32_t frames = 0;
        while (WavFile_ReadInt16(in_wav, raw_buf, P2P_FRAME_SAMPLES) == P2P_FRAME_SAMPLES) {
            if (g_stage_dump.enabled) {
                int16_t raw_avg[P2P_FRAME_SAMPLES];
                int active_ch = (num_mics < in_ch) ? num_mics : in_ch;
                for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                    int32_t sum = 0;
                    for (int m = 0; m < active_ch; ++m) sum += raw_buf[i * in_ch + m];
                    raw_avg[i] = (int16_t)(active_ch > 0 ? (sum / active_ch) : 0);
                }
                stage_dump_write(STAGE_00_RAW_INPUT_AVG, raw_avg, P2P_FRAME_SAMPLES);
            }
            for (int m = 0; m < num_mics && m < in_ch; ++m) {
                float g = gain_tbl[m];
                for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                    int32_t v = (int32_t)((float)raw_buf[i * in_ch + m] * g);
                    if (v > 32767) v = 32767;
                    else if (v < -32768) v = -32768;
                    mic_ch[m][i] = (int16_t)v;
                }
            }
            stage_dump_write_ch_avg(STAGE_01_PREGAIN_AVG, mic_ch, num_mics);
            if (g_cfg.ref_channel >= 0 && g_cfg.ref_channel < in_ch) {
                for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                    ref_sig[i] = raw_buf[i * in_ch + g_cfg.ref_channel];
                }
            } else {
                memset(ref_sig, 0, sizeof(ref_sig));
            }
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            dut_dsp_process_frame(&dsp, mic_ch, ref_sig, aec_err, clean_out);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double dt_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
            sum_ms += dt_ms;
            if (dt_ms > max_ms) max_ms = dt_ms;
            if (dt_ms > 16.0) over16++;
            if (frame_ms) {
                if (frames >= cap_frames) {
                    cap_frames *= 2;
                    double *tmp = (double *)realloc(frame_ms, cap_frames * sizeof(double));
                    if (tmp) frame_ms = tmp;
                    else { free(frame_ms); frame_ms = NULL; }
                }
                if (frame_ms) frame_ms[frames] = dt_ms;
            }
            if (g_cfg.enable_nvidia) {
                write(g_nv_bridge.to_nv_fd, clean_out, sizeof(clean_out));
            } else {
                stage_dump_write(STAGE_12_FINAL_OUTPUT, clean_out, P2P_FRAME_SAMPLES);
                WavFile_WriteInt16(out_wav, clean_out, P2P_FRAME_SAMPLES);
            }
            frames++;
        }
        WavFile_Close(in_wav);
        if (g_cfg.enable_nvidia) {
            nvidia_bridge_stop(&g_nv_bridge);
        }
        WavFile_Close(out_wav);
        stage_dump_close();
        double avg_ms = frames ? (sum_ms / frames) : 0.0;
        double p95_ms = max_ms;
        if (frame_ms && frames > 0) {
            qsort(frame_ms, frames, sizeof(double), cmp_double_asc);
            size_t p95_idx = (size_t)(0.95 * (double)(frames - 1));
            p95_ms = frame_ms[p95_idx];
        }
        free(frame_ms);
        printf("[DUT-Offline] Processed %u frames (%.2f s) | avg=%.3f ms/16ms (RTF=%.4f), p95=%.3f ms, max=%.3f ms, >16ms=%u -> %s\n",
               frames, frames * (float)P2P_FRAME_SAMPLES / P2P_SAMPLE_RATE,
               avg_ms, avg_ms / 16.0, p95_ms, max_ms, over16, g_cfg.offline_out_wav);
        dut_dsp_destroy(&dsp);
        return 0;
    }

    /* 5. Open Audio Pipes:
     * CRITICAL for rockchip_multi Software Digital Loopback Mode 14:
     * arecord (capture DMA + loopback tap) MUST open and start streaming BEFORE aplay,
     * otherwise aplay advances the loopback TX FIFO while capture isn't reading, shifting
     * Ch8/9 behind Ch0~7 by 600~900ms and breaking LinearAEC causal alignment!
     */
    char mic_cmd[256];
    snprintf(mic_cmd, sizeof(mic_cmd),
             "arecord -D %s --period-size=256 --buffer-size=8192 -r 16000 -c %d -f S16_LE -t raw 2>/dev/null",
             g_cfg.mic_device, g_cfg.mic_channels);
    g_mic_pipe = popen(mic_cmd, "r");

    if (g_mic_pipe) {
        int16_t warmup_mic[P2P_FRAME_SAMPLES * MAX_MIC_CHANNELS];
        for (int w = 0; w < 6; ++w) {
            size_t got = fread(warmup_mic, sizeof(int16_t), P2P_FRAME_SAMPLES * g_cfg.mic_channels, g_mic_pipe);
            if (got == 0) break;
        }
    }

    /* Use an 8192-sample playback ring with 2048-sample (128ms) start_threshold (-R 128000):
     * gives ~380ms headroom above the steady-state 2048-sample fill so rockchip-i2s-tdm
     * neither wraps/repeats 2048-sample DMA blocks in Ch8 nor hits high latency. */
    char spk_cmd[256];
    snprintf(spk_cmd, sizeof(spk_cmd),
             "aplay -D %s --period-size=256 --buffer-size=8192 -R 128000 -r 16000 -c 2 -f S16_LE -t raw 2>/dev/null",
             g_cfg.spk_device);
    g_spk_pipe = popen(spk_cmd, "w");

    /* Pre-fill aplay buffer with 8 frames (2048 stereo samples = 128ms) of silence to match -R 128000 */
    if (g_spk_pipe) {
        int16_t zero_spk[P2P_FRAME_SAMPLES * 2] = {0};
        for (int p = 0; p < 8; ++p) {
            fwrite(zero_spk, sizeof(int16_t), P2P_FRAME_SAMPLES * 2, g_spk_pipe);
        }
        fflush(g_spk_pipe);
    }

    net_fifo_init(&g_net_fifo);
    pthread_t spk_tid;
    pthread_create(&spk_tid, NULL, spk_thread_func, NULL);

    int send_sock = socket(AF_INET, SOCK_DGRAM, 0);

    FILE *fp_inject = NULL;
    if (g_cfg.inject_wav_path[0] != '\0') {
        fp_inject = fopen(g_cfg.inject_wav_path, "rb");
        if (fp_inject) {
            fseek(fp_inject, 44, SEEK_SET); /* Skip 44-byte WAV header */
            printf("[DUT-Main] Benchmark Speech Injection Active: %s\n", g_cfg.inject_wav_path);
        }
    }

    FILE *fp_record_raw = NULL;
    FILE *fp_record_err = NULL;
    FILE *fp_record_clean = NULL;
    uint32_t recorded_samples = 0;
    uint32_t target_samples = (uint32_t)record_sec * P2P_SAMPLE_RATE;

    if (raw_multi_path) {
        fp_record_raw = fopen(raw_multi_path, "wb");
        if (fp_record_raw) {
            write_wav_header(fp_record_raw, target_samples ? target_samples : 60 * P2P_SAMPLE_RATE, P2P_SAMPLE_RATE, (uint16_t)g_cfg.mic_channels);
            fflush(fp_record_raw);
            printf("[DUT-Main] %d-Ch Raw Recording Active: %s\n", g_cfg.mic_channels, raw_multi_path);
        }
    }
    if (raw_err_path) {
        fp_record_err = fopen(raw_err_path, "wb");
        if (fp_record_err) {
            write_wav_header(fp_record_err, target_samples ? target_samples : 60 * P2P_SAMPLE_RATE, P2P_SAMPLE_RATE, 1);
            fflush(fp_record_err);
            printf("[DUT-Main] Linear AEC Error Recording Active: %s\n", raw_err_path);
        }
    }
    if (raw_clean_path) {
        fp_record_clean = fopen(raw_clean_path, "wb");
        if (fp_record_clean) {
            write_wav_header(fp_record_clean, target_samples ? target_samples : 60 * P2P_SAMPLE_RATE, P2P_SAMPLE_RATE, 1);
            fflush(fp_record_clean);
            printf("[DUT-Main] 3A Clean Output Recording Active: %s\n", raw_clean_path);
        }
    }

    int16_t mic_raw[P2P_FRAME_SAMPLES * MAX_MIC_CHANNELS];
    int16_t mic_ch[MAX_MIC_CHANNELS][P2P_FRAME_SAMPLES];
    int16_t ref_signal[P2P_FRAME_SAMPLES];
    int16_t spk_now[P2P_FRAME_SAMPLES];
    int16_t aec_error_fused[P2P_FRAME_SAMPLES];
    int16_t clean_out[P2P_FRAME_SAMPLES];

    memset(spk_now, 0, sizeof(spk_now));

    P2PPacket return_pkt;
    return_pkt.magic = P2P_MAGIC;
    return_pkt.samples = P2P_FRAME_SAMPLES;
    uint32_t out_seq = 0;
    bool s_recording_started = (target_samples == 0);
    int last_printed_sec = -1;

    /* Gain vector across mic channels */
    float mic_gain[MAX_MIC_CHANNELS];
    init_mic_gains(mic_gain, num_mics, has_custom_gains, custom_gains);

    if (g_cfg.enable_nvidia) {
        char nv_model[512] = "", nv_bin[512] = "";
        if (!resolve_nvidia_model_path(g_cfg.nvidia_model_path, nv_model, sizeof(nv_model)) ||
            !resolve_nvidia_bin_path(g_cfg.nvidia_bin_path, nv_bin, sizeof(nv_bin))) {
            fprintf(stderr, "[Warning] Nvidia model or bin not found, bypassing Nvidia stage\n");
            g_cfg.enable_nvidia = false;
        } else {
            nvidia_bridge_start(&g_nv_bridge, nv_model, nv_bin,
                                g_cfg.nvidia_core_mask, g_cfg.nvidia_dry_gain,
                                false, send_sock, fp_record_clean, NULL);
        }
    }

    printf("[DUT-Main] Audio loop running! (Press Ctrl+C to terminate)\n");

    while (g_running) {
        size_t n = fread(mic_raw, sizeof(int16_t), P2P_FRAME_SAMPLES * g_cfg.mic_channels, g_mic_pipe);
        if (n < (size_t)(P2P_FRAME_SAMPLES * g_cfg.mic_channels)) {
            if (feof(g_mic_pipe) || ferror(g_mic_pipe)) break;
            continue;
        }

        /* 1. Untouched hardware frames read; post-calibrated PCM recorded below */


        /* 2. Pull net far-end audio and write to speaker (apply -3 dB headroom to avoid PA clipping) */
        bool has_net_ref = net_fifo_pull(&g_net_fifo, spk_now, 3);
        if (has_net_ref) {
            if (!s_recording_started) {
                s_recording_started = true;
                printf("\n======================================================================\n");
                printf(" [DUT-Record] Far-End Audio Stream Detected! 60s Test Started!\n");
                printf(" >>>>> SECONDS  0 - 30: SINGLE-TALK (Far-end playing, keep quiet) <<<<<\n");
                printf(" >>>>> SECONDS 30 - 60: DOUBLE-TALK (Speak into the microphone array) <<<<<\n");
                printf("======================================================================\n\n");
                fflush(stdout);
            }
            for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                spk_now[i] = (int16_t)(spk_now[i] * 0.707f);
            }
        }
        if (g_spk_pipe) {
            int16_t spk_stereo[P2P_FRAME_SAMPLES * 2];
            for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                int16_t val = spk_now[i];
                spk_stereo[2 * i] = val;
                spk_stereo[2 * i + 1] = val;
            }
            size_t w = fwrite(spk_stereo, sizeof(int16_t), P2P_FRAME_SAMPLES * 2, g_spk_pipe);
            if (w < (size_t)(P2P_FRAME_SAMPLES * 2)) {
                pclose(g_spk_pipe);
                g_spk_pipe = popen(spk_cmd, "w");
            } else {
                fflush(g_spk_pipe);
            }
        }

        /* 2. De-interleave mic channels and apply PDM pre-gain (Ch8/9 loopback remains 1.0x) */
        static const int kPhysicalMicMap4[4] = {3, 2, 0, 1};
        for (int m = 0; m < num_mics; ++m) {
            int raw_ch = (num_mics == 4 && g_cfg.mic_channels == 6) ? kPhysicalMicMap4[m] : m;
            float gain = mic_gain[m];
            for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                int32_t val = (int32_t)((float)mic_raw[i * g_cfg.mic_channels + raw_ch] * gain);
                if (val > 32767) val = 32767;
                if (val < -32768) val = -32768;
                mic_ch[m][i] = (int16_t)val;
                mic_raw[i * g_cfg.mic_channels + raw_ch] = (int16_t)val;
            }
        }

        /* Record post-calibrated raw multi-channel frames (audible in Audacity & matching offline replay) */
        if (s_recording_started && fp_record_raw) {
            fwrite(mic_raw, sizeof(int16_t), P2P_FRAME_SAMPLES * g_cfg.mic_channels, fp_record_raw);
        }



        if (g_cfg.ref_channel >= 0 && g_cfg.ref_channel < g_cfg.mic_channels) {
            /* True Hardware Loopback Reference */
            for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                ref_signal[i] = mic_raw[i * g_cfg.mic_channels + g_cfg.ref_channel];
            }
        } else {
            /* Network Reference */
            memcpy(ref_signal, spk_now, sizeof(spk_now));
        }

        /* Optional speech injection for local simulation */
        if (fp_inject && has_net_ref) {
            bool allow_inject = true;
            if (g_cfg.inject_delay_sec > 0 && recorded_samples < (uint32_t)g_cfg.inject_delay_sec * P2P_SAMPLE_RATE) {
                allow_inject = false;
            }
            if (allow_inject) {
                int16_t inj[P2P_FRAME_SAMPLES];
                size_t r = fread(inj, sizeof(int16_t), P2P_FRAME_SAMPLES, fp_inject);
                if (r < (size_t)P2P_FRAME_SAMPLES) {
                    fseek(fp_inject, 44, SEEK_SET);
                    size_t r2 = fread(&inj[r], sizeof(int16_t), P2P_FRAME_SAMPLES - r, fp_inject);
                    (void)r2;
                }
                for (int m = 0; m < num_mics; ++m) {
                    int raw_ch = (num_mics == 4 && g_cfg.mic_channels == 6) ? kPhysicalMicMap4[m] : m;
                    for (int i = 0; i < P2P_FRAME_SAMPLES; ++i) {
                        int32_t mixed = (int32_t)mic_ch[m][i] + (int32_t)inj[i];
                        if (mixed > 32767) mixed = 32767;
                        if (mixed < -32768) mixed = -32768;
                        mic_ch[m][i] = (int16_t)mixed;
                        mic_raw[i * g_cfg.mic_channels + raw_ch] = (int16_t)mixed;
                    }
                }
            }
        }



        dut_dsp_process_frame(&dsp, mic_ch, ref_signal, aec_error_fused, clean_out);

        if (s_recording_started && fp_record_err) {
            fwrite(aec_error_fused, sizeof(int16_t), P2P_FRAME_SAMPLES, fp_record_err);
        }

        if (g_cfg.enable_nvidia) {
            write(g_nv_bridge.to_nv_fd, clean_out, sizeof(clean_out));
        } else {
            if (s_recording_started && fp_record_clean) {
                fwrite(clean_out, sizeof(int16_t), P2P_FRAME_SAMPLES, fp_record_clean);
            }

            if (g_cfg.peer_known) {
                return_pkt.seq = ++out_seq;
                return_pkt.timestamp_ms = g_last_far_timestamp_ms;
                return_pkt.flags = g_cfg.enable_aec ? 1 : 0;
                memcpy(return_pkt.pcm, clean_out, sizeof(clean_out));

                sendto(send_sock, &return_pkt, sizeof(return_pkt), 0,
                       (struct sockaddr*)&g_cfg.peer_addr, sizeof(g_cfg.peer_addr));

                if (out_seq % 50 == 0) {
                    float agc_gain = AdaptiveAGC_GetCurrentGain(&dsp.agc) *
                                     ((dsp.use_dfnet3 || dsp.use_dpdfnet || dsp.gtcrn != NULL) ? AdaptiveAGC_GetCurrentGain(&dsp.post_agc) : 1.0f);
                    if (num_mics >= 8) {
                        printf("[DUT Stats] Frame:%-5u | ERLE M0:%3.0f M1:%3.0f M2:%3.0f M3:%3.0f M4:%3.0f M5:%3.0f M6:%3.0f M7:%3.0f dB | State:%s | AGC:%+5.1fdB\n",
                               out_seq,
                               LinearAEC_GetERLE(&dsp.aec[0]), LinearAEC_GetERLE(&dsp.aec[1]),
                               LinearAEC_GetERLE(&dsp.aec[2]), LinearAEC_GetERLE(&dsp.aec[3]),
                               LinearAEC_GetERLE(&dsp.aec[4]), LinearAEC_GetERLE(&dsp.aec[5]),
                               LinearAEC_GetERLE(&dsp.aec[6]), LinearAEC_GetERLE(&dsp.aec[7]),
                               (dsp.hangover_counter > 0) ? "TALK   " : "SILENCE",
                               20.0f * log10f(agc_gain + 1e-6f));
                    } else {
                        printf("[DUT Stats] Frame:%-5u | ERLE M0:%4.1f M1:%4.1f M2:%4.1f M3:%4.1f dB | State:%s | AGC:%+5.1fdB\n",
                               out_seq,
                               LinearAEC_GetERLE(&dsp.aec[0]),
                               (num_mics > 1) ? LinearAEC_GetERLE(&dsp.aec[1]) : 0,
                               (num_mics > 2) ? LinearAEC_GetERLE(&dsp.aec[2]) : 0,
                               (num_mics > 3) ? LinearAEC_GetERLE(&dsp.aec[3]) : 0,
                               (dsp.hangover_counter > 0) ? "TALK   " : "SILENCE",
                               20.0f * log10f(agc_gain + 1e-6f));
                    }
                    fflush(stdout);
                }
            }
        }

        if (s_recording_started && (raw_multi_path || raw_clean_path || raw_err_path)) {
            recorded_samples += P2P_FRAME_SAMPLES;
            int cur_sec = (int)(recorded_samples / P2P_SAMPLE_RATE);
            if (cur_sec != last_printed_sec && cur_sec % 5 == 0 && cur_sec <= record_sec) {
                last_printed_sec = cur_sec;
                if (cur_sec == 30) {
                    printf("\n**********************************************************************\n");
                    printf(" [Test Time: %2d / %2ds] >>>>> PLEASE SPEAK NOW! (Double-talk started) <<<<<\n", cur_sec, record_sec);
                    printf("**********************************************************************\n\n");
                } else if (cur_sec < 30) {
                    printf(" [Test Time: %2d / %2ds] Far-end playing... (Please remain silent)\n", cur_sec, record_sec);
                } else {
                    printf(" [Test Time: %2d / %2ds] Double-talk active... (Keep talking naturally)\n", cur_sec, record_sec);
                }
                fflush(stdout);
            }
            if (target_samples > 0 && recorded_samples >= target_samples) {
                printf("\n[DUT-Record] Target %u seconds completed!\n", record_sec);
                fflush(stdout);
                if (record_exit) g_running = false;
            }
        }
    }

    printf("\n[DUT] Shutting down...\n");
    g_running = false;
    pthread_join(spk_tid, NULL);
    if (g_cfg.enable_nvidia) {
        nvidia_bridge_stop(&g_nv_bridge);
    }

    dut_dsp_destroy(&dsp);
    if (fp_inject) fclose(fp_inject);
    if (fp_record_raw) {
        write_wav_header(fp_record_raw, recorded_samples, P2P_SAMPLE_RATE, (uint16_t)g_cfg.mic_channels);
        fclose(fp_record_raw);
        printf("[DUT-Main] Finalized Raw WAV: %s (%.1f s)\n", raw_multi_path, (float)recorded_samples / P2P_SAMPLE_RATE);
    }
    if (fp_record_err) {
        write_wav_header(fp_record_err, recorded_samples, P2P_SAMPLE_RATE, 1);
        fclose(fp_record_err);
        printf("[DUT-Main] Finalized Linear Error WAV: %s (%.1f s)\n", raw_err_path, (float)recorded_samples / P2P_SAMPLE_RATE);
    }
    if (fp_record_clean) {
        uint32_t final_clean_samples = g_cfg.enable_nvidia ? g_nv_bridge.recorded_samples : recorded_samples;
        write_wav_header(fp_record_clean, final_clean_samples, P2P_SAMPLE_RATE, 1);
        fclose(fp_record_clean);
        printf("[DUT-Main] Finalized 3A Clean WAV: %s (%.1f s)\n", raw_clean_path, (float)final_clean_samples / P2P_SAMPLE_RATE);
    }
    close(send_sock);
    if (g_mic_pipe) pclose(g_mic_pipe);
    if (g_spk_pipe) pclose(g_spk_pipe);

    return 0;
}
