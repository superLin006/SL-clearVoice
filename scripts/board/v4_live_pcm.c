/* Low-latency v4 equalization and dynamics for mono 16 kHz S16_LE PCM. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { RATE = 16000, FRAME = 256, EQ_FILTERS = 7 };

typedef struct {
    float b0, b1, b2, a1, a2;
    float z1, z2;
} Biquad;

typedef struct {
    Biquad eq[EQ_FILTERS];
    float speech_power, program_power, gain, gate, limiter;
    float speech_rise, speech_fall, program_rise, program_fall;
    float gain_down, gain_up, gate_open, gate_close, limiter_release;
    size_t limited_samples;
} Postprocessor;

typedef struct {
    double *origins;
    size_t blocks, capacity;
    double first_origin, sum_ms, max_ms;
} Timing;

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static double now_ms(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time)) fail("PCM clock read failed");
    return 1000.0 * time.tv_sec + 1e-6 * time.tv_nsec;
}

static float coefficient(float milliseconds) {
    return -expm1f(-1000.0f / (RATE * milliseconds));
}

static Biquad band_edge(double hz, int highpass) {
    const double w = 6.283185307179586 * hz / RATE;
    const double c = cos(w);
    const double alpha = sin(w) / 1.414213562373095;
    const double a0 = 1 + alpha;
    const double b0 = (highpass ? 1 + c : 1 - c) / 2;
    return (Biquad) {.b0 = b0 / a0, .b1 = (highpass ? -2 : 2) * b0 / a0,
                     .b2 = b0 / a0, .a1 = -2 * c / a0, .a2 = (1 - alpha) / a0};
}

static Biquad peak_eq(double hz, double q, double db) {
    const double w = 6.283185307179586 * hz / RATE;
    const double alpha = sin(w) / (2 * q);
    const double a = pow(10.0, db / 40);
    const double a0 = 1 + alpha / a;
    const double b1 = -2 * cos(w) / a0;
    return (Biquad) {.b0 = (1 + alpha * a) / a0, .b1 = b1,
                     .b2 = (1 - alpha * a) / a0, .a1 = b1, .a2 = (1 - alpha / a) / a0};
}

static void init(Postprocessor *post) {
    /* The same filter order, frequencies, Q and gains as v4_post.sh. */
    post->eq[0] = band_edge(150, 1);
    post->eq[1] = peak_eq(300, 1.2, 0.5);
    post->eq[2] = peak_eq(1200, 1.0, 6.0);
    post->eq[3] = peak_eq(2200, 1.0, 2.0);
    post->eq[4] = peak_eq(3200, 1.0, 4.0);
    post->eq[5] = peak_eq(5000, 1.0, -1.0);
    post->eq[6] = band_edge(6000, 0);
    post->speech_rise = coefficient(28);
    post->speech_fall = coefficient(55);
    post->program_rise = coefficient(140);
    post->program_fall = coefficient(650);
    post->gain_down = coefficient(45);
    post->gain_up = coefficient(330);
    post->gate_open = coefficient(4);
    post->gate_close = coefficient(150);
    post->limiter_release = coefficient(80);
    post->program_power = 0.000126f; /* Initial program level of -39 dBFS. */
    post->gain = 9.0f;
    post->gate = 0.10f;
    post->limiter = 1.0f;
}

static float filter(Biquad *eq, float x) {
    float y = eq->b0 * x + eq->z1;
    eq->z1 = eq->b1 * x - eq->a1 * y + eq->z2;
    eq->z2 = eq->b2 * x - eq->a2 * y;
    /* Keep filter state normal after a long digital silence. */
    if (fabsf(eq->z1) < 1e-24f) eq->z1 = 0;
    if (fabsf(eq->z2) < 1e-24f) eq->z2 = 0;
    return y;
}

static float dynamics(Postprocessor *post, float audio) {
    const float level_target = 0.22f;
    const float max_gain = 31.62f; /* Limit makeup gain to 30 dB. */
    const float peak_limit = 0.741310f; /* Leave sample peak headroom at -2.6 dBFS. */
    const float power = audio * audio;
    const float detector = power > post->speech_power ? post->speech_rise : post->speech_fall;
    post->speech_power += detector * (power - post->speech_power);
    /* Freeze the program level during quiet gaps so they cannot drive gain upward. */
    const float active_floor = fmaxf(1.0e-7f, 0.024f * post->program_power);
    if (post->speech_power > active_floor) {
        const float level_rate = post->speech_power > post->program_power
            ? post->program_rise : post->program_fall;
        post->program_power += level_rate * (post->speech_power - post->program_power);
        const float desired = fminf(max_gain, level_target / sqrtf(post->program_power));
        const float gain_rate = desired < post->gain ? post->gain_down : post->gain_up;
        post->gain += gain_rate * (desired - post->gain);
    }
    const float level = sqrtf(post->speech_power) * post->gain;
    const float gate = fmaxf(0.10f, fminf(1.0f, level / 0.010f));
    const float gate_rate = gate > post->gate ? post->gate_open : post->gate_close;
    post->gate += gate_rate * (gate - post->gate);

    audio *= post->gain * post->gate;
    post->limiter += post->limiter_release * (1.0f - post->limiter);
    if (fabsf(audio) * post->limiter > peak_limit) {
        post->limiter = peak_limit / fabsf(audio);
        ++post->limited_samples;
    }
    return audio * post->limiter;
}

static void process(Postprocessor *post, int16_t *pcm, size_t count, int eq_only) {
    for (size_t i = 0; i < count; ++i) {
        float audio = pcm[i] / 32768.0f;
        for (size_t band = 0; band < EQ_FILTERS; ++band)
            audio = filter(&post->eq[band], audio);
        if (!eq_only) audio = dynamics(post, audio);
        pcm[i] = (int16_t)lrintf(fmaxf(-32768.0f, fminf(32767.0f, 32768.0f * audio)));
    }
}

static void record(Timing *timing, size_t emitted, double start, double finish, double ready) {
    const double origin = ready - 1000.0 * emitted / RATE;
    if (!timing->blocks) timing->first_origin = origin;
    if (timing->blocks == timing->capacity) {
        size_t capacity = timing->capacity ? 2 * timing->capacity : 256;
        double *updated = realloc(timing->origins, capacity * sizeof(*updated));
        if (!updated) fail("Could not allocate PCM timing");
        timing->origins = updated;
        timing->capacity = capacity;
    }
    timing->origins[timing->blocks++] = origin;
    timing->sum_ms += finish - start;
    timing->max_ms = fmax(timing->max_ms, finish - start);
}

static int compare(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    int measure = 0, eq_only = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--timing")) measure = 1;
        else if (!strcmp(argv[i], "--eq-only")) eq_only = 1;
        else fail("Usage: v4_live_pcm [--timing] [--eq-only] < input.pcm > output.pcm");
    }
    Postprocessor post = {0};
    Timing timing = {0};
    init(&post);
    int16_t pcm[FRAME];
    size_t emitted = 0;
    size_t bytes;
    while ((bytes = fread(pcm, 1, sizeof(pcm), stdin)) != 0) {
        if (ferror(stdin) || bytes % sizeof(*pcm)) fail("Incomplete S16_LE input");
        const size_t count = bytes / sizeof(*pcm);
        const double start = measure ? now_ms() : 0;
        process(&post, pcm, count, eq_only);
        const double finish = measure ? now_ms() : 0;
        if (fwrite(pcm, sizeof(*pcm), count, stdout) != count || fflush(stdout))
            fail("PCM output write failed");
        if (measure) record(&timing, emitted, start, finish, now_ms());
        emitted += count;
    }
    if (ferror(stdin)) fail("PCM input read failed");
    if (measure && timing.blocks) {
        qsort(timing.origins, timing.blocks, sizeof(*timing.origins), compare);
        fprintf(stderr, "{\"samples\":%zu,\"blocks\":%zu,\"mean_frame_ms\":%.5f,"
                "\"max_frame_ms\":%.5f,\"first_origin_ms\":%.3f,\"p95_origin_ms\":%.3f,"
                "\"max_origin_ms\":%.3f,\"limited_samples\":%zu}\n",
                emitted, timing.blocks, timing.sum_ms / timing.blocks, timing.max_ms,
                timing.first_origin, timing.origins[(size_t)(.95 * (timing.blocks - 1))],
                timing.origins[timing.blocks - 1], post.limited_samples);
    }
    free(timing.origins);
    return 0;
}
