/* Native 16 kHz S16_LE streaming runner for the fixed first SB sampling step. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fftw3.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "rknn_api.h"

enum { RATE=16000, FFT=510, HOP=128, BINS=256, MAX_FRAMES=64, MAX_WINDOW=(MAX_FRAMES-1)*HOP };
static const float spectral_scale=0.33f, pi=3.14159265358979323846f;

static void fail(const char *msg) {
    fprintf(stderr,"%s\n",msg);
    exit(1);
}

static void check(int rc, const char *what) {
    if (rc) {
        fprintf(stderr,"%s failed: %d\n",what,rc);
        exit(1);
    }
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0+t.tv_nsec/1e6;
}

static void wait_until(double ms) {
    struct timespec t;
    t.tv_sec=(time_t)(ms/1000);
    t.tv_nsec=(long)((ms-t.tv_sec*1000.0)*1e6);
    int error;
    do { error=clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&t,NULL); } while (error==EINTR);
    check(error,"clock_nanosleep");
}

static int frame_count(const rknn_tensor_attr *attr) {
    if (attr->n_dims != 4 || attr->dims[0] != 1) return 0;
    if (attr->fmt == RKNN_TENSOR_NHWC && attr->dims[1] == BINS && attr->dims[3] == 2)
        return (int)attr->dims[2];
    if (attr->fmt == RKNN_TENSOR_NCHW && attr->dims[1] == 2 && attr->dims[2] == BINS)
        return (int)attr->dims[3];
    return 0;
}

static int index_of(int frames, int nhwc, int bin, int frame, int part) {
    return nhwc ? (bin*frames+frame)*2+part : (part*BINS+bin)*frames+frame;
}

typedef struct {
    rknn_context ctx;
    int in_nhwc, out_nhwc, frames, window;
    rknn_tensor_format input_format;
    float hann[FFT], synthesis_gain[MAX_WINDOW];
    float frame[FFT];
    fftwf_complex bins[BINS];
    fftwf_plan forward, inverse;
    float input[2*BINS*MAX_FRAMES];
} Pipeline;

static void init(Pipeline *p, const char *model, rknn_core_mask cores) {
    check(rknn_init(&p->ctx,(void *)model,0,0,NULL),"rknn_init");
    check(rknn_set_core_mask(p->ctx,cores),"rknn_set_core_mask");
    rknn_input_output_num counts={0};
    check(rknn_query(p->ctx,RKNN_QUERY_IN_OUT_NUM,&counts,sizeof(counts)),"query count");
    if (counts.n_input!=1 || counts.n_output!=1) fail("Expected one-step SB with one input and one output");
    rknn_tensor_attr input={0},output={0};
    check(rknn_query(p->ctx,RKNN_QUERY_INPUT_ATTR,&input,sizeof(input)),"query input");
    check(rknn_query(p->ctx,RKNN_QUERY_OUTPUT_ATTR,&output,sizeof(output)),"query output");
    p->frames = frame_count(&input);
    if (p->frames < 8 || p->frames > MAX_FRAMES || frame_count(&output) != p->frames)
        fail("Expected fixed SB I/O with 256 bins and matched time frames");
    p->window = (p->frames - 1) * HOP;
    p->input_format=input.fmt;
    p->in_nhwc=input.fmt==RKNN_TENSOR_NHWC;
    p->out_nhwc=output.fmt==RKNN_TENSOR_NHWC;
    rknn_sdk_version version={0};
    check(rknn_query(p->ctx,RKNN_QUERY_SDK_VERSION,&version,sizeof(version)),"query version");
    fprintf(stderr,"RKNN runtime %s, NPU driver %s\n",version.api_version,version.drv_version);
    p->forward=fftwf_plan_dft_r2c_1d(FFT,p->frame,p->bins,FFTW_ESTIMATE);
    p->inverse=fftwf_plan_dft_c2r_1d(FFT,p->bins,p->frame,FFTW_ESTIMATE);
    if (!p->forward || !p->inverse) fail("Could not initialize FFT");
    for (int j=0;j<FFT;++j) p->hann[j]=0.5f-0.5f*cosf(2*pi*j/FFT);
    for (int k=0;k<p->frames;++k)
        for (int j=0;j<FFT;++j) {
            int i=k*HOP+j-FFT/2;
            if (i>=0 && i<p->window) p->synthesis_gain[i]+=p->hann[j]*p->hann[j];
        }
    for (int i=0;i<p->window;++i) p->synthesis_gain[i]=1.0f/(FFT*p->synthesis_gain[i]);
}

static double enhance(Pipeline *p,const int16_t *pcm,int count,float *audio) {
    float peak=0.0f;
    for (int i=0;i<count;++i) peak=fmaxf(peak,fabsf((float)pcm[i]));
    float input_gain=1/(peak+32768.0f*1e-8f);
    for (int k=0;k<p->frames;++k) {
        for (int j=0;j<FFT;++j) {
            int i=k*HOP+j-FFT/2;
            p->frame[j]=i>=0 && i<count ? pcm[i]*input_gain*p->hann[j] : 0.0f;
        }
        fftwf_execute(p->forward);
        for (int f=0;f<BINS;++f) {
            float re=p->bins[f][0],im=p->bins[f][1];
            float abs_sq=re*re+im*im;
            float scale=abs_sq>0 ? spectral_scale/sqrtf(sqrtf(abs_sq)) : 0.0f;
            p->input[index_of(p->frames,p->in_nhwc,f,k,0)]=re*scale;
            p->input[index_of(p->frames,p->in_nhwc,f,k,1)]=im*scale;
        }
    }
    unsigned bytes = 2 * BINS * p->frames * sizeof(float);
    rknn_input in={.index=0,.buf=p->input,.size=bytes,.type=RKNN_TENSOR_FLOAT32,.fmt=p->input_format};
    rknn_output out={.index=0,.want_float=1};
    double start=now_ms();
    check(rknn_inputs_set(p->ctx,1,&in),"rknn_inputs_set");
    check(rknn_run(p->ctx,NULL),"rknn_run");
    check(rknn_outputs_get(p->ctx,1,&out,NULL),"rknn_outputs_get");
    double npu_ms=now_ms()-start;
    if (out.size!=bytes) fail("Unexpected float estimate size");
    for (unsigned i=0;i<bytes/sizeof(float);++i)
        if (!isfinite(((const float *)out.buf)[i])) fail("Non-finite RKNN estimate");
    const float *estimate=out.buf;
    memset(audio,0,p->window*sizeof(*audio));
    for (int k=0;k<p->frames;++k) {
        for (int f=0;f<BINS;++f) {
            int out_re=index_of(p->frames,p->out_nhwc,f,k,0),out_im=index_of(p->frames,p->out_nhwc,f,k,1);
            int in_re=index_of(p->frames,p->in_nhwc,f,k,0),in_im=index_of(p->frames,p->in_nhwc,f,k,1);
            /* Keep the original one-step mean; Gaussian sampling is disabled. */
            float re=(0.0000331757874f*p->input[in_re]+0.9999668f*estimate[out_re])/spectral_scale;
            float im=(0.0000331757874f*p->input[in_im]+0.9999668f*estimate[out_im])/spectral_scale;
            float power=sqrtf(re*re+im*im);
            p->bins[f][0]=re*power;
            p->bins[f][1]=im*power;
        }
        fftwf_execute(p->inverse);
        for (int j=0;j<FFT;++j) {
            int i=k*HOP+j-FFT/2;
            if (i>=0 && i<p->window) audio[i]+=p->frame[j]*p->hann[j];
        }
    }
    check(rknn_outputs_release(p->ctx,1,&out),"rknn_outputs_release");
    for (int i=0;i<p->window;++i) audio[i]*=p->synthesis_gain[i]*peak/32768.0f;
    return npu_ms;
}

static int read_pcm(int16_t *buf,int count) {
    size_t got=fread(buf,1,(size_t)count*sizeof(*buf),stdin);
    if (ferror(stdin) || got%sizeof(*buf)) fail("Invalid or incomplete PCM16 input");
    return (int)(got/sizeof(*buf));
}

static void write_pcm(FILE *dest, const float *audio, const int16_t *dry,
                      int count, float dry_gain) {
    int16_t pcm[MAX_WINDOW];
    for (int i=0;i<count;++i) {
        if (!isfinite(audio[i])) fail("Non-finite audio estimate");
        float value=audio[i]*32768.0f + dry_gain*dry[i];
        value=fmaxf(-32768.0f,fminf(32767.0f,value));
        pcm[i]=(int16_t)lrintf(value);
    }
    if (fwrite(pcm,sizeof(*pcm),count,dest)!=(size_t)count || fflush(dest)) fail("PCM output failed");
}

static int cmp(const void *a,const void *b) {
    double x=*(const double *)a,y=*(const double *)b;
    return (x>y)-(x<y);
}

static int parse_int(const char *text, int max) {
    char *end;
    long value=strtol(text,&end,10);
    if (end==text || *end || value<0 || value>max) fail("Invalid integer option");
    return (int)value;
}

int main(int argc,char **argv) {
    if (argc<2 || argc>5) fail("Usage: nvidia_sb_pcm MODEL [overlap_ms=16] [history_ms=0] [--pace]");
    int overlap=argc>=3 ? 16*parse_int(argv[2],MAX_WINDOW/16) : 256;
    int history=argc>=4 ? 16*parse_int(argv[3],MAX_WINDOW/16) : 0;
    int pace=argc==5 && strcmp(argv[4],"--pace")==0;
    if (argc==5 && !pace) fail("Invalid pacing option");
    const char *core_env = getenv("SB_RKNN_CORE_MASK");
    rknn_core_mask cores = RKNN_NPU_CORE_0_1_2;
    if (core_env != NULL) {
        int requested = parse_int(core_env,7);
        if (requested < 1 || requested > 7) fail("SB_RKNN_CORE_MASK must be in 1..7");
        cores = (rknn_core_mask)requested;
    }
    float dry_gain = 0.0f;
    const char *dry_env = getenv("SB_DRY_GAIN");
    if (dry_env != NULL) {
        char *end;
        dry_gain = strtof(dry_env, &end);
        if (end == dry_env || *end || !(dry_gain >= 0.0f && dry_gain <= 1.0f))
            fail("SB_DRY_GAIN must be between 0 and 1");
    }
    /* Keep the native runtime's diagnostic stdout separate from the audio bytes. */
    int audio_fd=dup(fileno(stdout));
    if (audio_fd<0 || dup2(fileno(stderr),fileno(stdout))<0) fail("Could not separate audio and log output");
    FILE *dest=fdopen(audio_fd,"wb");
    if (!dest) fail("Could not open PCM output");
    Pipeline *p=calloc(1,sizeof(*p));
    if (!p) fail("Could not allocate FFT state");
    init(p,argv[1],cores);
    const char *expected_frames = getenv("SB_EXPECT_FRAMES");
    if (expected_frames && p->frames != parse_int(expected_frames,MAX_FRAMES))
        fail("RKNN frame count does not match the configured profile");
    int fresh = p->window - history;
    int stride = fresh - overlap;
    if (stride <= 0 || overlap > stride) fail("Invalid window options");
    int16_t pcm[MAX_WINDOW]={0}, pending_dry[MAX_WINDOW];
    float audio[MAX_WINDOW],pending[MAX_WINDOW];
    enhance(p,pcm,p->window,audio);

    double sum=0,npu_sum=0,max=0,queue=0,peak_queue=0,first=0;
    uint64_t read_samples=0,emitted=0;
    size_t blocks=0,late=0,capacity=512;
    double *lags=malloc(capacity*sizeof(*lags));
    double *origins=malloc(capacity*sizeof(*origins));
    if (!lags || !origins) fail("Could not allocate timing samples");
    double start=now_ms();
    int got=read_pcm(pcm+history,fresh);
    read_samples+=got;
    int tail=0;
    while (got>0) {
        if (pace) wait_until(start+read_samples/16.0);
        double tick=now_ms();
        npu_sum+=enhance(p,pcm,history+got,audio);
        float *out=audio+history;
        for (int i=0;i<tail;++i) {
            float mix=overlap>1 ? (float)i/(overlap-1) : 0;
            out[i]=pending[i]*(1-mix)+out[i]*mix;
        }
        int emit=got<stride ? got : stride;
        write_pcm(dest,out,pcm+history,emit,dry_gain);
        double ready=now_ms();
        double ms=ready-tick;
        sum+=ms;
        max=fmax(max,ms);
        if (!blocks) first=pace ? ready-start : got/16.0+ms;
        queue=fmax(0.0,queue+ms-stride/16.0);
        peak_queue=fmax(peak_queue,queue);
        if (ms>stride/16.0) ++late;
        if (blocks==capacity) {
            capacity*=2;
            double *grown=realloc(lags,capacity*sizeof(*lags));
            if (!grown) fail("Could not extend timing storage");
            lags=grown;
            grown=realloc(origins,capacity*sizeof(*origins));
            if (!grown) fail("Could not extend timing storage");
            origins=grown;
        }
        origins[blocks]=ready-emitted/16.0;
        lags[blocks++]=pace ? ready-start-emitted/16.0 : ms;
        emitted+=emit;
        tail=got-emit;
        memcpy(pending,out+emit,tail*sizeof(*pending));
        memcpy(pending_dry,pcm+history+emit,tail*sizeof(*pending_dry));
        if (got<fresh) break;
        memmove(pcm,pcm+stride,(history+overlap)*sizeof(*pcm));
        int next=read_pcm(pcm+history+overlap,stride);
        read_samples+=next;
        if (!next) break;
        got=overlap+next;
    }
    if (tail) write_pcm(dest,pending,pending_dry,tail,dry_gain);
    emitted+=tail;
    if (!blocks || emitted!=read_samples) fail("Unexpected PCM sample count");
    qsort(lags,blocks,sizeof(*lags),cmp);
    qsort(origins,blocks,sizeof(*origins),cmp);
    size_t index=(size_t)(0.95*(blocks-1));
    double p95=lags[index];
    fprintf(stderr,"{\"frames\":%d,\"core_mask\":%d,\"dry_gain\":%.3f,\"gaussian_sampling\":false,\"blocks\":%zu,\"samples\":%llu,\"mean_ms\":%.3f,\"npu_mean_ms\":%.3f,"
        "\"max_ms\":%.3f,\"budget_ms\":%.3f,\"rtf\":%.4f,\"over_budget\":%zu,"
        "\"max_queue_ms\":%.3f,\"input_clock_ms\":%.3f,\"first_ready_ms\":%.3f,\"p95_%s_ms\":%.3f,\"p95_origin_ms\":%.3f}\n",
        p->frames,cores,dry_gain,blocks,(unsigned long long)emitted,sum/blocks,npu_sum/blocks,max,stride/16.0,
        sum/blocks/(stride/16.0),late,peak_queue,start,first,pace?"output_lag":"service",p95,origins[index]);
    free(lags);
    free(origins);
    fflush(dest);
    fclose(dest);
    fftwf_destroy_plan(p->forward);
    fftwf_destroy_plan(p->inverse);
    rknn_destroy(p->ctx);
    free(p);
    return 0;
}
