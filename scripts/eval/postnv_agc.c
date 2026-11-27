/* Apply the board's post-NVIDIA AGC offline, so a replacement NVIDIA stage can
 * be compared against the shipped one with everything downstream identical.
 *
 * Mirrors nvidia_bridge_start() + apply_nvidia_post_leveling() exactly:
 *   init gains seeded the same way, same hangover VAD (40 frames above
 *   0.0015 RMS), same per-frame AdaptiveAGC_ProcessInt16.
 *
 * usage: postnv_agc <in.wav|in.f32> <out.f32> [--expand]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "adaptive_agc.h"

#define SR 16000
#define FRAME 256
#define INIT_GAIN 1.68f          /* kNvidiaPostInitGainLinear */
#define HANGOVER 40

static float *load_pcm(const char *path, long *n) {
    size_t len = strlen(path);
    if (len > 4 && strcmp(path + len - 4, ".f32") == 0) {
        FILE *fp = fopen(path, "rb");
        if (!fp) return NULL;
        fseek(fp, 0, SEEK_END); long bytes = ftell(fp); fseek(fp, 0, SEEK_SET);
        float *x = malloc((size_t)bytes);
        if (!x) { fclose(fp); return NULL; }
        if (fread(x, 1, (size_t)bytes, fp) != (size_t)bytes) { fclose(fp); free(x); return NULL; }
        fclose(fp);
        *n = bytes / (long)sizeof(float);
        return x;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); long bytes = ftell(fp); fseek(fp, 0, SEEK_SET);
    int16_t *raw = malloc((size_t)bytes);
    if (!raw) { fclose(fp); return NULL; }
    if (fread(raw, 1, (size_t)bytes, fp) != (size_t)bytes) { fclose(fp); free(raw); return NULL; }
    fclose(fp);
    long cnt = (bytes - 44) / 2;
    float *x = malloc(sizeof(float) * (size_t)cnt);
    if (!x) { free(raw); return NULL; }
    for (long i = 0; i < cnt; ++i) x[i] = raw[22 + i] / 32768.0f;
    free(raw);
    *n = cnt;
    return x;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <in.wav|in.f32> <out.f32> [--expand]\n", argv[0]);
        return 2;
    }
    int expand = (argc > 3 && strcmp(argv[3], "--expand") == 0);

    long n = 0;
    float *in = load_pcm(argv[1], &n);
    if (!in) { perror(argv[1]); return 1; }

    AGCConfig cfg;
    AdaptiveAGC_GetDefaultConfig(&cfg);
    /* agc_target_db defaults to -17.0 in p2p_dut_node */
    cfg.target_level_rms = powf(10.0f, (-17.0f + 1.0f) / 20.0f);
    cfg.max_gain_db = 14.0f;
    cfg.min_gain_db = -6.0f;
    cfg.noise_gate_threshold = 0.002f;
    cfg.attack_time_ms = 10.0f;
    cfg.release_time_ms = 260.0f;
    cfg.silence_standby_gain = 1.0f;
    cfg.enable_eq = false;
    cfg.enable_limiter = true;
    if (expand) {
        cfg.expand_range_db = 18.0f;
        cfg.expand_ratio = 1.0f;
        cfg.expand_max_cut_db = 18.0f;
    }

    AdaptiveAGC agc;
    if (AdaptiveAGC_Init(&agc, &cfg, SR) != 0) { fprintf(stderr, "init failed\n"); return 1; }
    agc.current_gain = INIT_GAIN;
    agc.smoothed_rms = cfg.target_level_rms / INIT_GAIN;

    float *out = calloc((size_t)n, sizeof(float));
    int hangover = 0;
    for (long pos = 0; pos + FRAME <= n; pos += FRAME) {
        double ss = 0.0;
        for (int i = 0; i < FRAME; ++i) ss += (double)in[pos + i] * in[pos + i];
        float rms = (float)sqrt(ss / FRAME);
        if (rms > 0.0015f) hangover = HANGOVER;
        else if (hangover > 0) hangover--;
        AdaptiveAGC_SetVAD(&agc, hangover > 0 ? 1 : 0);
        AdaptiveAGC_ProcessFloat(&agc, in + pos, out + pos, FRAME);
    }

    FILE *fp = fopen(argv[2], "wb");
    if (!fp) { perror(argv[2]); return 1; }
    fwrite(out, sizeof(float), (size_t)n, fp);
    fclose(fp);
    fprintf(stderr, "%s: %ld samples, expand=%s\n", argv[2], n, expand ? "on" : "off");

    free(out); free(in);
    return 0;
}
