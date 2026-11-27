#include "dpdfnet_ans.h"
#include "onnxruntime_c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    float r;
    float i;
} DpCpx;

struct DPDFNetOrtCtx {
    const OrtApi    *ort;
    OrtEnv          *env;
    OrtSession      *sess;
    OrtMemoryInfo   *mem_info;
    OrtAllocator    *alloc;

    char *in_names[2];
    char *out_names[2];

    size_t state_size;
    float *state_init;
    float *state_buf[2];
    int    pong;

    float  spec_in[DPDFNET_FREQ_BINS * 2];
    float  spec_e[DPDFNET_FREQ_BINS * 2];

    OrtValue *spec_in_val;
    OrtValue *spec_e_val;
    OrtValue *state_in_val[2];
    OrtValue *state_out_val[2];

    /* Precomputed tables for exact 320-point (5 x 64) Cooley-Tukey FFT */
    uint8_t bitrev64[64];
    DpCpx   tw64[32];
    DpCpx   tw320[5][64];
    DpCpx   w5[5][5];
};

static inline void dp_fifo_push(float *fifo, int *w, int *count, const float *src, int n) {
    for (int i = 0; i < n; ++i) {
        if (*count < DPDFNET_FIFO_CAP) {
            fifo[*w] = src[i];
            *w = (*w + 1) & (DPDFNET_FIFO_CAP - 1);
            (*count)++;
        }
    }
}

static inline void dp_fifo_pop(float *fifo, int *r, int *count, float *dst, int n) {
    for (int i = 0; i < n; ++i) {
        if (*count > 0) {
            dst[i] = fifo[*r];
            *r = (*r + 1) & (DPDFNET_FIFO_CAP - 1);
            (*count)--;
        } else {
            dst[i] = 0.0f;
        }
    }
}

static int ort_ok(const OrtApi *ort, OrtStatus *st, const char *ctx) {
    if (!st) return 1;
    fprintf(stderr, "[DPDFNetANS] ORT error at %s: %s\n",
            ctx, ort ? ort->GetErrorMessage(st) : "(null ort)");
    if (ort) ort->ReleaseStatus(st);
    return 0;
}

static inline void ort_free_ptr(const OrtApi *ort, OrtAllocator *alloc, void *p) {
    if (ort && alloc && p) {
        OrtStatus *st = ort->AllocatorFree(alloc, p);
        if (st) ort->ReleaseStatus(st);
    }
}

static void init_fft320_tables(DPDFNetOrtCtx *ctx) {
    for (int i = 0; i < 64; ++i) {
        uint8_t v = (uint8_t)i;
        uint8_t rev = 0;
        for (int b = 0; b < 6; ++b) {
            rev = (uint8_t)((rev << 1) | (v & 1));
            v >>= 1;
        }
        ctx->bitrev64[i] = rev;
    }
    for (int k = 0; k < 32; ++k) {
        double a = -2.0 * M_PI * (double)k / 64.0;
        ctx->tw64[k].r = (float)cos(a);
        ctx->tw64[k].i = (float)sin(a);
    }
    for (int n1 = 0; n1 < 5; ++n1) {
        for (int k2 = 0; k2 < 64; ++k2) {
            double a = -2.0 * M_PI * (double)(n1 * k2) / 320.0;
            ctx->tw320[n1][k2].r = (float)cos(a);
            ctx->tw320[n1][k2].i = (float)sin(a);
        }
    }
    for (int k1 = 0; k1 < 5; ++k1) {
        for (int n1 = 0; n1 < 5; ++n1) {
            double a = -2.0 * M_PI * (double)(k1 * n1) / 5.0;
            ctx->w5[k1][n1].r = (float)cos(a);
            ctx->w5[k1][n1].i = (float)sin(a);
        }
    }
}

static void fft64_inplace(const DPDFNetOrtCtx *ctx, DpCpx d[64]) {
    for (int i = 0; i < 64; ++i) {
        int r = ctx->bitrev64[i];
        if (i < r) {
            DpCpx t = d[i];
            d[i] = d[r];
            d[r] = t;
        }
    }
    for (int len = 2; len <= 64; len <<= 1) {
        int half = len >> 1;
        int step = 64 / len;
        for (int i = 0; i < 64; i += len) {
            for (int j = 0; j < half; ++j) {
                DpCpx w = ctx->tw64[j * step];
                DpCpx u = d[i + j];
                DpCpx t = d[i + j + half];
                DpCpx v;
                v.r = t.r * w.r - t.i * w.i;
                v.i = t.r * w.i + t.i * w.r;
                d[i + j].r = u.r + v.r;
                d[i + j].i = u.i + v.i;
                d[i + j + half].r = u.r - v.r;
                d[i + j + half].i = u.i - v.i;
            }
        }
    }
}

/* Forward 320-pt unnormalized real DFT -> bins [0..160] interleaved (re, im). */
static void rfft320_forward(const DPDFNetOrtCtx *ctx,
                            const float in_320[DPDFNET_WIN_LEN],
                            float out_161x2[DPDFNET_FREQ_BINS * 2]) {
    DpCpx H[5][64];
    for (int n1 = 0; n1 < 5; ++n1) {
        DpCpx sub[64];
        for (int n2 = 0; n2 < 64; ++n2) {
            sub[n2].r = in_320[5 * n2 + n1];
            sub[n2].i = 0.0f;
        }
        fft64_inplace(ctx, sub);
        for (int k2 = 0; k2 < 64; ++k2) {
            DpCpx w = ctx->tw320[n1][k2];
            DpCpx g = sub[k2];
            H[n1][k2].r = g.r * w.r - g.i * w.i;
            H[n1][k2].i = g.r * w.i + g.i * w.r;
        }
    }

    /* k = 64 * k1 + k2, evaluate only bins 0..160 (k1=0,1 full, k1=2 for k2<=32) */
    for (int k1 = 0; k1 <= 2; ++k1) {
        int k2_limit = (k1 < 2) ? 64 : 33;
        for (int k2 = 0; k2 < k2_limit; ++k2) {
            float sr = 0.0f;
            float si = 0.0f;
            for (int n1 = 0; n1 < 5; ++n1) {
                DpCpx w = ctx->w5[k1][n1];
                DpCpx h = H[n1][k2];
                sr += h.r * w.r - h.i * w.i;
                si += h.r * w.i + h.i * w.r;
            }
            int k = 64 * k1 + k2;
            out_161x2[2 * k + 0] = sr;
            out_161x2[2 * k + 1] = si;
        }
    }
}

/* Inverse 320-pt real DFT from bins [0..160] with 1/320 normalization. */
static void irfft320_inverse(const DPDFNetOrtCtx *ctx,
                             const float in_161x2[DPDFNET_FREQ_BINS * 2],
                             float out_320[DPDFNET_WIN_LEN]) {
    DpCpx Z[DPDFNET_WIN_LEN];
    Z[0].r = in_161x2[0];
    Z[0].i = 0.0f;
    Z[160].r = in_161x2[320];
    Z[160].i = 0.0f;
    for (int k = 1; k < 160; ++k) {
        float re = in_161x2[2 * k + 0];
        float im = in_161x2[2 * k + 1];
        Z[k].r = re;
        Z[k].i = -im;
        Z[DPDFNET_WIN_LEN - k].r = re;
        Z[DPDFNET_WIN_LEN - k].i = im;
    }

    DpCpx H[5][64];
    for (int n1 = 0; n1 < 5; ++n1) {
        DpCpx sub[64];
        for (int n2 = 0; n2 < 64; ++n2) {
            sub[n2] = Z[5 * n2 + n1];
        }
        fft64_inplace(ctx, sub);
        for (int k2 = 0; k2 < 64; ++k2) {
            DpCpx w = ctx->tw320[n1][k2];
            DpCpx g = sub[k2];
            H[n1][k2].r = g.r * w.r - g.i * w.i;
            H[n1][k2].i = g.r * w.i + g.i * w.r;
        }
    }

    const float inv_n = 1.0f / (float)DPDFNET_WIN_LEN;
    for (int k1 = 0; k1 < 5; ++k1) {
        for (int k2 = 0; k2 < 64; ++k2) {
            float sr = 0.0f;
            for (int n1 = 0; n1 < 5; ++n1) {
                DpCpx w = ctx->w5[k1][n1];
                DpCpx h = H[n1][k2];
                sr += h.r * w.r - h.i * w.i;
            }
            out_320[64 * k1 + k2] = sr * inv_n;
        }
    }
}

static size_t parse_csv_floats(const char *csv, float *dst, size_t max_n) {
    if (!csv || !dst || max_n == 0) return 0;
    size_t idx = 0;
    const char *p = csv;
    while (*p != '\0' && idx < max_n) {
        char *end = NULL;
        float val = strtof(p, &end);
        if (end == p) break;
        dst[idx++] = val;
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
    }
    return idx;
}

static int load_state_metadata(DPDFNetOrtCtx *ctx) {
    const OrtApi *ort = ctx->ort;
    OrtTypeInfo *tinfo = NULL;
    const OrtTensorTypeAndShapeInfo *sinfo = NULL;
    size_t ndims = 0;
    int64_t dims[4] = {0, 0, 0, 0};

    if (!ort_ok(ort, ort->SessionGetInputTypeInfo(ctx->sess, 1, &tinfo), "GetInputTypeInfo(1)")) {
        return -1;
    }
    if (!ort_ok(ort, ort->CastTypeInfoToTensorInfo(tinfo, &sinfo), "CastTensorInfo")) {
        ort->ReleaseTypeInfo(tinfo);
        return -1;
    }
    if (!ort_ok(ort, ort->GetDimensionsCount(sinfo, &ndims), "GetDimensionsCount")) {
        ort->ReleaseTypeInfo(tinfo);
        return -1;
    }
    if (ndims >= 1 && ndims <= 4) {
        if (!ort_ok(ort, ort->GetDimensions(sinfo, dims, ndims), "GetDimensions")) {
            ort->ReleaseTypeInfo(tinfo);
            return -1;
        }
    }
    ort->ReleaseTypeInfo(tinfo);

    int64_t state_elems = (ndims >= 1) ? dims[ndims - 1] : 0;

    OrtModelMetadata *meta = NULL;
    if (!ort_ok(ort, ort->SessionGetModelMetadata(ctx->sess, &meta), "SessionGetModelMetadata")) {
        meta = NULL;
    }

    char *val_str = NULL;
    size_t erb_sz = 32;
    size_t spec_sz = 96;

    if (meta) {
        if (ort->ModelMetadataLookupCustomMetadataMap(meta, ctx->alloc, "state_size", &val_str) == NULL && val_str) {
            int64_t m_sz = strtoll(val_str, NULL, 10);
            if (state_elems <= 0 && m_sz > 0) state_elems = m_sz;
            ort_free_ptr(ort, ctx->alloc, val_str);
            val_str = NULL;
        }
        if (ort->ModelMetadataLookupCustomMetadataMap(meta, ctx->alloc, "erb_norm_state_size", &val_str) == NULL && val_str) {
            long es = strtol(val_str, NULL, 10);
            if (es > 0) erb_sz = (size_t)es;
            ort_free_ptr(ort, ctx->alloc, val_str);
            val_str = NULL;
        }
        if (ort->ModelMetadataLookupCustomMetadataMap(meta, ctx->alloc, "spec_norm_state_size", &val_str) == NULL && val_str) {
            long ss = strtol(val_str, NULL, 10);
            if (ss > 0) spec_sz = (size_t)ss;
            ort_free_ptr(ort, ctx->alloc, val_str);
            val_str = NULL;
        }
    }

    if (state_elems <= 0 || state_elems > 1048576) {
        fprintf(stderr, "[DPDFNetANS] Invalid ONNX state_in size: %lld\n", (long long)state_elems);
        if (meta) ort->ReleaseModelMetadata(meta);
        return -1;
    }

    ctx->state_size = (size_t)state_elems;
    ctx->state_init = (float *)calloc(ctx->state_size, sizeof(float));
    ctx->state_buf[0] = (float *)calloc(ctx->state_size, sizeof(float));
    ctx->state_buf[1] = (float *)calloc(ctx->state_size, sizeof(float));
    if (!ctx->state_init || !ctx->state_buf[0] || !ctx->state_buf[1]) {
        if (meta) ort->ReleaseModelMetadata(meta);
        return -1;
    }

    if (meta) {
        if (ort->ModelMetadataLookupCustomMetadataMap(meta, ctx->alloc, "erb_norm_init", &val_str) == NULL && val_str) {
            size_t avail = (erb_sz <= ctx->state_size) ? erb_sz : ctx->state_size;
            parse_csv_floats(val_str, ctx->state_init, avail);
            ort_free_ptr(ort, ctx->alloc, val_str);
            val_str = NULL;
        }
        if (ort->ModelMetadataLookupCustomMetadataMap(meta, ctx->alloc, "spec_norm_init", &val_str) == NULL && val_str) {
            if (erb_sz < ctx->state_size) {
                size_t rem = ctx->state_size - erb_sz;
                size_t avail = (spec_sz <= rem) ? spec_sz : rem;
                parse_csv_floats(val_str, ctx->state_init + erb_sz, avail);
            }
            ort_free_ptr(ort, ctx->alloc, val_str);
            val_str = NULL;
        }
        ort->ReleaseModelMetadata(meta);
    }

    return 0;
}

static int run_single_hop(DPDFNetANS *eng, const float hop_in[DPDFNET_HOP_LEN], float hop_out[DPDFNET_HOP_LEN]) {
    DPDFNetOrtCtx *ctx = eng->ctx;
    const OrtApi *ort = ctx->ort;

    memmove(eng->ana_hist, eng->ana_hist + DPDFNET_HOP_LEN, DPDFNET_HOP_LEN * sizeof(float));
    memcpy(eng->ana_hist + DPDFNET_HOP_LEN, hop_in, DPDFNET_HOP_LEN * sizeof(float));

    float win_time[DPDFNET_WIN_LEN];
    for (int i = 0; i < DPDFNET_WIN_LEN; ++i) {
        win_time[i] = eng->ana_hist[i] * eng->vorbis_win[i];
    }
    rfft320_forward(ctx, win_time, ctx->spec_in);

    int in_idx = ctx->pong;
    int out_idx = 1 - in_idx;
    const OrtValue *inputs[2] = {ctx->spec_in_val, ctx->state_in_val[in_idx]};
    OrtValue *outputs[2] = {ctx->spec_e_val, ctx->state_out_val[out_idx]};
    const char *in_names[2] = {ctx->in_names[0], ctx->in_names[1]};
    const char *out_names[2] = {ctx->out_names[0], ctx->out_names[1]};

    OrtStatus *st = ort->Run(ctx->sess, NULL, in_names, inputs, 2, out_names, 2, outputs);
    if (st != NULL) {
        fprintf(stderr, "[DPDFNetANS] ORT Run error: %s\n", ort->GetErrorMessage(st));
        ort->ReleaseStatus(st);
        memcpy(hop_out, hop_in, DPDFNET_HOP_LEN * sizeof(float));
        return -1;
    }
    ctx->pong = out_idx;

    float syn_time[DPDFNET_WIN_LEN];
    irfft320_inverse(ctx, ctx->spec_e, syn_time);
    for (int i = 0; i < DPDFNET_WIN_LEN; ++i) {
        syn_time[i] *= eng->vorbis_win[i];
    }
    for (int i = 0; i < DPDFNET_HOP_LEN; ++i) {
        hop_out[i] = eng->ola_buf[i] + syn_time[i];
        eng->ola_buf[i] = syn_time[DPDFNET_HOP_LEN + i];
    }
    return 0;
}

int DPDFNetANS_Init(DPDFNetANS *eng, const char *model_onnx, int num_threads) {
    if (!eng || !model_onnx || model_onnx[0] == '\0') return -1;
    memset(eng, 0, sizeof(*eng));

    if (num_threads <= 0) num_threads = 1;
    eng->num_threads = num_threads;

    for (int n = 0; n < DPDFNET_WIN_LEN; ++n) {
        float s = sinf((float)M_PI * (n + 0.5f) / (float)DPDFNET_WIN_LEN);
        eng->vorbis_win[n] = sinf(0.5f * (float)M_PI * s * s);
    }

    DPDFNetOrtCtx *ctx = (DPDFNetOrtCtx *)calloc(1, sizeof(DPDFNetOrtCtx));
    if (!ctx) return -1;
    eng->ctx = ctx;

    init_fft320_tables(ctx);

    const OrtApiBase *base = OrtGetApiBase();
    ctx->ort = base ? base->GetApi(ORT_API_VERSION) : NULL;
    if (!ctx->ort) {
        fprintf(stderr, "[DPDFNetANS] Failed to get ORT API version %d\n", ORT_API_VERSION);
        DPDFNetANS_Destroy(eng);
        return -1;
    }
    const OrtApi *ort = ctx->ort;

    if (!ort_ok(ort, ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "dpdfnet", &ctx->env), "CreateEnv")) {
        DPDFNetANS_Destroy(eng);
        return -1;
    }

    OrtSessionOptions *opts = NULL;
    if (!ort_ok(ort, ort->CreateSessionOptions(&opts), "CreateSessionOptions")) {
        DPDFNetANS_Destroy(eng);
        return -1;
    }
    ort_ok(ort, ort->SetIntraOpNumThreads(opts, num_threads), "SetIntraOpNumThreads");
    ort_ok(ort, ort->SetInterOpNumThreads(opts, 1), "SetInterOpNumThreads");
    ort_ok(ort, ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL), "SetGraphOpt");

    OrtStatus *sess_st = ort->CreateSession(ctx->env, model_onnx, opts, &ctx->sess);
    ort->ReleaseSessionOptions(opts);
    if (!ort_ok(ort, sess_st, "CreateSession")) {
        DPDFNetANS_Destroy(eng);
        return -1;
    }

    if (!ort_ok(ort, ort->GetAllocatorWithDefaultOptions(&ctx->alloc), "GetAllocator")) {
        DPDFNetANS_Destroy(eng);
        return -1;
    }
    if (!ort_ok(ort, ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->mem_info),
                "CreateCpuMemoryInfo")) {
        DPDFNetANS_Destroy(eng);
        return -1;
    }

    for (size_t i = 0; i < 2; ++i) {
        if (!ort_ok(ort, ort->SessionGetInputName(ctx->sess, i, ctx->alloc, &ctx->in_names[i]), "GetInputName") ||
            !ort_ok(ort, ort->SessionGetOutputName(ctx->sess, i, ctx->alloc, &ctx->out_names[i]), "GetOutputName")) {
            DPDFNetANS_Destroy(eng);
            return -1;
        }
    }

    if (load_state_metadata(ctx) != 0) {
        DPDFNetANS_Destroy(eng);
        return -1;
    }

    const int64_t spec_shape[4] = {1, 1, DPDFNET_FREQ_BINS, 2};
    const int64_t state_shape[1] = {(int64_t)ctx->state_size};
    const size_t spec_bytes = sizeof(ctx->spec_in);
    const size_t state_bytes = ctx->state_size * sizeof(float);

    if (!ort_ok(ort,
                ort->CreateTensorWithDataAsOrtValue(ctx->mem_info, ctx->spec_in, spec_bytes, spec_shape, 4,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ctx->spec_in_val),
                "CreateSpecIn") ||
        !ort_ok(ort,
                ort->CreateTensorWithDataAsOrtValue(ctx->mem_info, ctx->spec_e, spec_bytes, spec_shape, 4,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ctx->spec_e_val),
                "CreateSpecOut") ||
        !ort_ok(ort,
                ort->CreateTensorWithDataAsOrtValue(ctx->mem_info, ctx->state_buf[0], state_bytes, state_shape, 1,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ctx->state_in_val[0]),
                "CreateStateIn0") ||
        !ort_ok(ort,
                ort->CreateTensorWithDataAsOrtValue(ctx->mem_info, ctx->state_buf[1], state_bytes, state_shape, 1,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ctx->state_in_val[1]),
                "CreateStateIn1") ||
        !ort_ok(ort,
                ort->CreateTensorWithDataAsOrtValue(ctx->mem_info, ctx->state_buf[0], state_bytes, state_shape, 1,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ctx->state_out_val[0]),
                "CreateStateOut0") ||
        !ort_ok(ort,
                ort->CreateTensorWithDataAsOrtValue(ctx->mem_info, ctx->state_buf[1], state_bytes, state_shape, 1,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ctx->state_out_val[1]),
                "CreateStateOut1")) {
        DPDFNetANS_Destroy(eng);
        return -1;
    }

    /* Warm up ONNX runtime buffers on zero hops, then reset cleanly to init state */
    DPDFNetANS_Reset(eng);
    float zeros[DPDFNET_HOP_LEN] = {0};
    float discard[DPDFNET_HOP_LEN];
    for (int i = 0; i < 10; ++i) {
        run_single_hop(eng, zeros, discard);
    }
    DPDFNetANS_Reset(eng);

    eng->initialized = 1;
    return 0;
}

void DPDFNetANS_Reset(DPDFNetANS *eng) {
    if (!eng || !eng->ctx) return;
    DPDFNetOrtCtx *ctx = eng->ctx;
    if (ctx->state_init && ctx->state_buf[0] && ctx->state_buf[1]) {
        memcpy(ctx->state_buf[0], ctx->state_init, ctx->state_size * sizeof(float));
        memcpy(ctx->state_buf[1], ctx->state_init, ctx->state_size * sizeof(float));
    }
    ctx->pong = 0;

    memset(eng->ana_hist, 0, sizeof(eng->ana_hist));
    memset(eng->ola_buf, 0, sizeof(eng->ola_buf));
    eng->in_r = eng->in_w = eng->in_count = 0;
    eng->out_r = eng->out_w = eng->out_count = 0;

    /* Pre-fill 1 hop (10ms) of silence so 256-sample P2P cadence never underruns out_fifo */
    float zeros[DPDFNET_HOP_LEN] = {0};
    dp_fifo_push(eng->out_fifo, &eng->out_w, &eng->out_count, zeros, DPDFNET_HOP_LEN);
}

static void process_chunk(DPDFNetANS *eng, const float *in_16k, float *out_16k, int n) {
    dp_fifo_push(eng->in_fifo, &eng->in_w, &eng->in_count, in_16k, n);

    float hop_in[DPDFNET_HOP_LEN];
    float hop_out[DPDFNET_HOP_LEN];
    while (eng->in_count >= DPDFNET_HOP_LEN) {
        dp_fifo_pop(eng->in_fifo, &eng->in_r, &eng->in_count, hop_in, DPDFNET_HOP_LEN);
        run_single_hop(eng, hop_in, hop_out);
        dp_fifo_push(eng->out_fifo, &eng->out_w, &eng->out_count, hop_out, DPDFNET_HOP_LEN);
    }

    dp_fifo_pop(eng->out_fifo, &eng->out_r, &eng->out_count, out_16k, n);
}

void DPDFNetANS_ProcessFloat(DPDFNetANS *eng, const float *in_16k, float *out_16k, int num_samples) {
    if (!in_16k || !out_16k || num_samples <= 0) return;
    if (!eng || !eng->initialized || !eng->ctx) {
        if (in_16k != out_16k) memcpy(out_16k, in_16k, (size_t)num_samples * sizeof(float));
        return;
    }
    int offset = 0;
    while (offset < num_samples) {
        int chunk = num_samples - offset;
        if (chunk > 256) chunk = 256;
        process_chunk(eng, in_16k + offset, out_16k + offset, chunk);
        offset += chunk;
    }
}

void DPDFNetANS_ProcessInt16(DPDFNetANS *eng, const int16_t *in_16k, int16_t *out_16k, int num_samples) {
    if (!in_16k || !out_16k || num_samples <= 0) return;
    if (!eng || !eng->initialized || !eng->ctx) {
        if (in_16k != out_16k) memcpy(out_16k, in_16k, (size_t)num_samples * sizeof(int16_t));
        return;
    }
    float fin[256];
    float fout[256];
    int offset = 0;
    while (offset < num_samples) {
        int chunk = num_samples - offset;
        if (chunk > 256) chunk = 256;
        for (int i = 0; i < chunk; ++i) {
            fin[i] = (float)in_16k[offset + i] * (1.0f / 32768.0f);
        }
        process_chunk(eng, fin, fout, chunk);
        for (int i = 0; i < chunk; ++i) {
            float v = fout[i] * 32768.0f;
            if (v > 32767.0f) v = 32767.0f;
            else if (v < -32768.0f) v = -32768.0f;
            out_16k[offset + i] = (int16_t)lrintf(v);
        }
        offset += chunk;
    }
}

void DPDFNetANS_Destroy(DPDFNetANS *eng) {
    if (!eng) return;
    DPDFNetOrtCtx *ctx = eng->ctx;
    if (ctx) {
        const OrtApi *ort = ctx->ort;
        if (ort) {
            if (ctx->spec_in_val) ort->ReleaseValue(ctx->spec_in_val);
            if (ctx->spec_e_val) ort->ReleaseValue(ctx->spec_e_val);
            for (int i = 0; i < 2; ++i) {
                if (ctx->state_in_val[i]) ort->ReleaseValue(ctx->state_in_val[i]);
                if (ctx->state_out_val[i]) ort->ReleaseValue(ctx->state_out_val[i]);
                ort_free_ptr(ort, ctx->alloc, ctx->in_names[i]);
                ort_free_ptr(ort, ctx->alloc, ctx->out_names[i]);
            }
            if (ctx->mem_info) ort->ReleaseMemoryInfo(ctx->mem_info);
            if (ctx->sess) ort->ReleaseSession(ctx->sess);
            if (ctx->env) ort->ReleaseEnv(ctx->env);
        }
        free(ctx->state_init);
        free(ctx->state_buf[0]);
        free(ctx->state_buf[1]);
        free(ctx);
        eng->ctx = NULL;
    }
    eng->initialized = 0;
}
