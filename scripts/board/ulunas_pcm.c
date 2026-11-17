/* Stateful UL-UNAS over 16 kHz mono S16_LE, with the existing 512/256 WOLA. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fftw3.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnxruntime_c_api.h"

enum { FFT = 512, HOP = 256, BINS = 257, CONV = 5358, TFA = 402, INTER = 1056 };

typedef struct {
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *mem;
    OrtValue *input[4], *output[4];
    const OrtValue *feed[4];
    float conv[2][CONV], tfa[2][TFA], inter[2][INTER];
    float frame[FFT], window[FFT], square[FFT];
    fftwf_complex mix[BINS], estimate[BINS];
    fftwf_plan encode, decode;
    double inference_ms;
    size_t frames;
} Denoiser;

static const OrtApi *ort;
static const char *const inputs[] = {"mix", "conv_cache", "tfa_cache", "inter_cache"};
static const char *const outputs[] = {"enh", "conv_cache_out", "tfa_cache_out", "inter_cache_out"};

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static void check(OrtStatus *status) {
    if (status == NULL) return;
    fprintf(stderr, "%s\n", ort->GetErrorMessage(status));
    ort->ReleaseStatus(status);
    exit(1);
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1000.0*t.tv_sec + t.tv_nsec/1e6;
}

static void init(Denoiser *dn, const char *model) {
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (ort == NULL) fail("ONNX Runtime C API is unavailable");
    OrtSessionOptions *options = NULL;
    check(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ulunas_pcm", &dn->env));
    check(ort->CreateSessionOptions(&options));
    check(ort->SetIntraOpNumThreads(options, 1));
    check(ort->SetInterOpNumThreads(options, 1));
    check(ort->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL));
    check(ort->CreateSession(dn->env, model, options, &dn->session));
    ort->ReleaseSessionOptions(options);
    check(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &dn->mem));

    int64_t dims[4][4] = {{1,BINS,1,2}, {1,CONV}, {1,TFA}, {1,INTER}};
    void *buffers[2][4] = {{dn->mix, dn->conv[0], dn->tfa[0], dn->inter[0]},
                           {dn->estimate, dn->conv[1], dn->tfa[1], dn->inter[1]}};
    const size_t sizes[] = {sizeof(dn->mix), sizeof(dn->conv[0]),
                            sizeof(dn->tfa[0]), sizeof(dn->inter[0])};
    for (int i=0; i<4; ++i) {
        size_t rank = i==0 ? 4 : 2;
        check(ort->CreateTensorWithDataAsOrtValue(dn->mem, buffers[0][i], sizes[i],
            dims[i], rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &dn->input[i]));
        check(ort->CreateTensorWithDataAsOrtValue(dn->mem, buffers[1][i], sizes[i],
            dims[i], rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &dn->output[i]));
        dn->feed[i] = dn->input[i];
    }
    dn->encode = fftwf_plan_dft_r2c_1d(FFT, dn->frame, dn->mix, FFTW_ESTIMATE);
    dn->decode = fftwf_plan_dft_c2r_1d(FFT, dn->estimate, dn->frame, FFTW_ESTIMATE);
    if (!dn->encode || !dn->decode) fail("FFTW plan creation failed");
    for (int i=0; i<FFT; ++i) {
        dn->window[i] = 0.5f - 0.5f*cosf(6.283185307179586f*i/FFT);
        dn->square[i] = dn->window[i]*dn->window[i];
    }
}

static void destroy(Denoiser *dn) {
    fftwf_destroy_plan(dn->encode);
    fftwf_destroy_plan(dn->decode);
    for (int i=0; i<4; ++i) {
        ort->ReleaseValue(dn->input[i]);
        ort->ReleaseValue(dn->output[i]);
    }
    ort->ReleaseMemoryInfo(dn->mem);
    ort->ReleaseSession(dn->session);
    ort->ReleaseEnv(dn->env);
}

static void transform(Denoiser *dn) {
    double start=now_ms();
    fftwf_execute(dn->encode);
    check(ort->Run(dn->session, NULL, inputs, dn->feed, 4, outputs, 4, dn->output));
    memcpy(dn->conv[0], dn->conv[1], sizeof(dn->conv[0]));
    memcpy(dn->tfa[0], dn->tfa[1], sizeof(dn->tfa[0]));
    memcpy(dn->inter[0], dn->inter[1], sizeof(dn->inter[0]));
    fftwf_execute(dn->decode);
    for (int i=0; i<FFT; ++i) dn->frame[i] *= dn->window[i]/FFT;
    dn->inference_ms += now_ms()-start;
    ++dn->frames;
}

typedef struct {
    double input_clock_ms;
    size_t samples;
    int enabled;
} InputClock;

static int read_pcm(int16_t *data, InputClock *clock) {
    size_t bytes=fread(data, 1, HOP*sizeof(*data), stdin);
    if (ferror(stdin) || bytes%sizeof(*data)) fail("Incomplete S16_LE input");
    int count = (int)(bytes/sizeof(*data));
    clock->samples += count;
    if (clock->enabled && count) {
        double ms = clock->input_clock_ms + clock->samples/16.0;
        struct timespec deadline = {.tv_sec=(time_t)(ms/1000)};
        deadline.tv_nsec=(long)((ms-1000.0*deadline.tv_sec)*1e6);
        int error;
        do { error=clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&deadline,NULL); }
        while (error==EINTR);
        if (error) fail("PCM replay clock failed");
    }
    return count;
}

static int reflect(int i, int count) {
    if (count<=1) return 0;
    int mod=2*(count-1);
    i=abs(i)%mod;
    return i<count ? i : mod-i;
}

static int16_t clip(float value) {
    if (!isfinite(value)) fail("Non-finite enhanced audio");
    return (int16_t)lrintf(fmaxf(-32768.0f, fminf(32767.0f, value*32768.0f)));
}

typedef struct {
    size_t samples, blocks, capacity;
    double first_origin_ms;
    double *origins;
} Timing;

static void send(Timing *timing, const int16_t *audio, int count) {
    if (fwrite(audio, sizeof(*audio), count, stdout)!=(size_t)count || fflush(stdout))
        fail("PCM output write failed");
    double origin = now_ms() - timing->samples/16.0;
    if (!timing->blocks) timing->first_origin_ms=origin;
    if (timing->blocks==timing->capacity) {
        timing->capacity=timing->capacity ? 2*timing->capacity : 256;
        double *updated=realloc(timing->origins,timing->capacity*sizeof(*timing->origins));
        if (!updated) fail("Could not allocate timing data");
        timing->origins=updated;
    }
    timing->origins[timing->blocks++]=origin;
    timing->samples+=count;
}

static int cmp(const void *a,const void *b) {
    double x=*(const double*)a, y=*(const double*)b;
    return (x>y)-(x<y);
}

int main(int argc,char **argv) {
    int pace = argc==3 && strcmp(argv[2],"--pace")==0;
    if (argc!=2 && !pace) fail("Usage: ulunas_pcm MODEL.onnx [--pace] <input.pcm >output.pcm");
    Denoiser *dn=calloc(1,sizeof(*dn));
    if (!dn) fail("Could not allocate denoiser");
    init(dn,argv[1]);
    int16_t pcm[FFT], output[HOP], left=0;
    float tail[HOP];
    InputClock clock={.input_clock_ms=now_ms(),.enabled=pace};
    int cur=read_pcm(pcm,&clock);
    if (!cur) fail("Empty PCM input");
    int next=cur==HOP ? read_pcm(pcm+HOP,&clock) : 0;
    size_t read_samples=cur+next;
    for (int i=0; i<FFT; ++i)
        dn->frame[i]=pcm[reflect(i-HOP,cur+next)]*dn->window[i]/32768.0f;
    transform(dn);
    memcpy(tail,dn->frame+HOP,sizeof(tail));
    Timing timing={0};
    while (cur==HOP) {
        for (int i=0; i<FFT; ++i) {
            int index=i;
            if (index>=HOP+next) index=2*(HOP+next-1)-index;
            int16_t sample=index>=0 ? pcm[index] : read_samples==(size_t)HOP ? pcm[1] : left;
            dn->frame[i]=sample*dn->window[i]/32768.0f;
        }
        transform(dn);
        for (int i=0; i<HOP; ++i)
            output[i]=clip((tail[i]+dn->frame[i])/(dn->square[i]+dn->square[HOP+i]));
        send(&timing,output,HOP);
        memcpy(tail,dn->frame+HOP,sizeof(tail));
        left=pcm[HOP-1];
        memcpy(pcm,pcm+HOP,(size_t)next*sizeof(*pcm));
        cur=next;
        next=cur==HOP ? read_pcm(pcm+HOP,&clock) : 0;
        read_samples+=next;
    }
    for (int i=0; i<cur; ++i) output[i]=clip(tail[i]/dn->square[HOP+i]);
    if (cur) send(&timing,output,cur);
    if (read_samples!=timing.samples) fail("PCM input and output sample counts differ");
    qsort(timing.origins,timing.blocks,sizeof(*timing.origins),cmp);
    fprintf(stderr,"{\"samples\":%zu,\"frames\":%zu,\"mean_frame_ms\":%.4f,"
        "\"input_clock_ms\":%.3f,\"first_origin_ms\":%.3f,\"p95_origin_ms\":%.3f,\"max_origin_ms\":%.3f}\n",
        timing.samples,dn->frames,dn->inference_ms/dn->frames,clock.input_clock_ms,timing.first_origin_ms,
        timing.origins[(size_t)(.95*(timing.blocks-1))],timing.origins[timing.blocks-1]);
    free(timing.origins);
    destroy(dn);
    free(dn);
    return 0;
}
