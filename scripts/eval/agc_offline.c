/* Offline AGC experiment on the board's own pre-AGC signal.
 *
 * Reads 05_neural_res.wav (the exact input the on-board pre-AGC sees), runs the
 * real AdaptiveAGC twice -- once with expansion off (current board behaviour)
 * and once with it on -- and writes both so the difference can be listened to
 * and measured. This isolates the AGC change from every other stage.
 *
 * usage: agc_offline <in.wav> <out_off.f32> <out_on.f32>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "adaptive_agc.h"

#define SR 16000
#define FRAME 256

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.wav> <out_off.f32> <out_on.f32>\n", argv[0]);
        return 2;
    }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror(argv[1]); return 1; }
    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    /* skip 44-byte canonical wav header */
    int16_t *pcm = malloc((size_t)bytes);
    if (!pcm) return 1;
    if (fread(pcm, 1, (size_t)bytes, fp) != (size_t)bytes) { fclose(fp); return 1; }
    fclose(fp);
    long n = (bytes - 44) / 2;
    int16_t *s = pcm + 22;   /* 44 bytes = 22 int16 */

    float *in = malloc(sizeof(float) * n);
    for (long i = 0; i < n; ++i) in[i] = s[i] / 32768.0f;

    AGCConfig base;
    AdaptiveAGC_GetDefaultConfig(&base);
    base.target_level_rms = 0.125f;
    base.max_gain_db = 40.0f;
    base.min_gain_db = -12.0f;
    base.noise_gate_threshold = 0.0005f;
    base.attack_time_ms = 10.0f;
    base.release_time_ms = 220.0f;
    base.silence_standby_gain = 2.5f;
    base.enable_eq = true;
    base.enable_limiter = true;

    for (int variant = 0; variant < 2; ++variant) {
        AGCConfig cfg = base;
        if (variant == 1) {
            cfg.expand_range_db = 18.0f;
            cfg.expand_ratio = 1.0f;
            cfg.expand_max_cut_db = 18.0f;
        }
        AdaptiveAGC agc;
        if (AdaptiveAGC_Init(&agc, &cfg, SR) != 0) { fprintf(stderr, "init failed\n"); return 1; }

        float *out = calloc((size_t)n, sizeof(float));
        for (long pos = 0; pos + FRAME <= n; pos += FRAME)
            AdaptiveAGC_ProcessFloat(&agc, in + pos, out + pos, FRAME);

        const char *path = (variant == 0) ? argv[2] : argv[3];
        FILE *of = fopen(path, "wb");
        if (of) { fwrite(out, sizeof(float), (size_t)n, of); fclose(of); }
        fprintf(stderr, "%s: expand=%s\n", path, variant ? "ON" : "OFF");
        free(out);
    }

    free(in);
    free(pcm);
    return 0;
}
