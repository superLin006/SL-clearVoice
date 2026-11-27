/* Stateful 16 kHz mono DPDFNet2 PCM runner. Model weights remain unchanged. */
#define _POSIX_C_SOURCE 200809L
#include "fftw3.h"
#include "onnxruntime_c_api.h"
#include "rknn_api.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
  RATE = 16000,
  HOP = 160,
  FFT = 320,
  BINS = 161,
  STATE = 45424,
  MODEL_DELAY_HOPS = 4,
  STARTUP_HOPS = MODEL_DELAY_HOPS + 1
};
typedef struct {
  rknn_context npu;
  rknn_tensor_mem *input_mem[4], *output_mem[2];
  rknn_tensor_attr native_input[4], native_output[2];
  const OrtApi *ort;
  OrtEnv *env;
  OrtSession *session;
  OrtMemoryInfo *memory;
  OrtValue *spec_input, *spec_output, *state_values[2];
  int pong, use_npu;
  __fp16 half_state[STATE];
  float state[2][STATE], norm[128], erb[32], features[192],
      filterbank[BINS * 32];
  float window[FFT], history[FFT], ola[HOP], frame[FFT];
  fftwf_complex spectrum[BINS], enhanced[BINS];
  fftwf_plan forward, inverse;
} Pipeline;

static void fail(const char *text) {
  fprintf(stderr, "%s\n", text);
  exit(1);
}
static void check(int code, const char *text) {
  if (code) {
    fprintf(stderr, "%s failed %d\n", text, code);
    exit(1);
  }
}
static void ort_check(Pipeline *p, OrtStatus *status) {
  if (status) {
    fprintf(stderr, "ORT: %s\n", p->ort->GetErrorMessage(status));
    exit(1);
  }
}
static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return 1000.0 * t.tv_sec + t.tv_nsec / 1e6;
}
static void read_asset(const char *dir, const char *name, float *buf,
                       size_t count) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "rb");
  if (!f || fread(buf, sizeof(float), count, f) != count || fgetc(f) != EOF)
    fail("Invalid model asset");
  fclose(f);
}
static void prepare(Pipeline *p, const char *model, const char *assets,
                    int cores, int threads) {
  read_asset(assets, "initial-state.bin", p->state[0], STATE);
  memcpy(p->state[1], p->state[0], sizeof(p->state[0]));
  memcpy(p->norm, p->state[0], sizeof(p->norm));
  for (int i = 0; i < STATE; ++i)
    p->half_state[i] = (__fp16)p->state[0][i];
  read_asset(assets, "erb-filterbank.bin", p->filterbank, BINS * 32);
  for (int i = 0; i < FFT; ++i) {
    float s = sinf(3.14159265358979323846f * (i + .5f) / FFT);
    p->window[i] = sinf(.5f * 3.14159265358979323846f * s * s);
  }
  p->forward = fftwf_plan_dft_r2c_1d(FFT, p->frame, p->spectrum, FFTW_ESTIMATE);
  p->inverse = fftwf_plan_dft_c2r_1d(FFT, p->enhanced, p->frame, FFTW_ESTIMATE);
  if (!p->forward || !p->inverse)
    fail("FFTW initialization failed");
  if (p->use_npu) {
    check(rknn_init(&p->npu, (void *)model, 0, 0, NULL), "rknn_init");
    check(rknn_set_core_mask(p->npu, (rknn_core_mask)cores), "core mask");

    const unsigned input_elements[] = {322, STATE, 32, 192};
    const unsigned output_elements[] = {322, STATE};
    for (int i = 0; i < 4; i++) {
      rknn_tensor_attr *a = &p->native_input[i];
      a->index = i;
      check(
          rknn_query(p->npu, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, a, sizeof(*a)),
          "native input");
      if (a->type != RKNN_TENSOR_FLOAT16 || a->n_elems != input_elements[i] ||
          a->size_with_stride < input_elements[i] * sizeof(__fp16))
        fail("Unexpected native input tensor");
      if (i == 0 && (a->fmt != RKNN_TENSOR_NHWC || a->n_dims != 4 ||
                     a->dims[3] != 1 || a->w_stride < 2))
        fail("Unexpected spectrum stride");
      if (i == 3 && (a->fmt != RKNN_TENSOR_NCHW || a->w_stride != 2))
        fail("Unexpected feature stride");
      p->input_mem[i] = rknn_create_mem(p->npu, a->size_with_stride);
      if (!p->input_mem[i])
        fail("input mem");
      memset(p->input_mem[i]->virt_addr, 0, a->size_with_stride);
      /* Native FP16 data already uses the queried layout and row padding. */
      a->pass_through = 1;
      check(rknn_set_io_mem(p->npu, p->input_mem[i], a), "bind input");
    }
    for (int i = 0; i < 2; i++) {
      rknn_tensor_attr *a = &p->native_output[i];
      a->index = i;
      check(
          rknn_query(p->npu, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, a, sizeof(*a)),
          "native output");
      if (a->type != RKNN_TENSOR_FLOAT16 || a->n_elems != output_elements[i] ||
          a->size_with_stride < output_elements[i] * sizeof(__fp16))
        fail("Unexpected native output tensor");
      if (i == 0 && (a->fmt != RKNN_TENSOR_NHWC || a->w_stride != 0))
        fail("Unexpected enhanced spectrum layout");
      p->output_mem[i] = rknn_create_mem(p->npu, a->size_with_stride);
      if (!p->output_mem[i])
        fail("output mem");
      check(rknn_set_io_mem(p->npu, p->output_mem[i], a), "bind output");
    }

  } else {
    p->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!p->ort)
      fail("ORT API unavailable");
    ort_check(p,
              p->ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "dpdfnet", &p->env));
    OrtSessionOptions *options = NULL;
    ort_check(p, p->ort->CreateSessionOptions(&options));
    ort_check(p, p->ort->SetIntraOpNumThreads(options, threads));
    ort_check(p, p->ort->SetInterOpNumThreads(options, 1));
    ort_check(
        p, p->ort->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL));
    ort_check(p, p->ort->CreateSession(p->env, model, options, &p->session));
    p->ort->ReleaseSessionOptions(options);
    ort_check(p, p->ort->CreateCpuMemoryInfo(OrtArenaAllocator,
                                             OrtMemTypeDefault, &p->memory));
    const int64_t spec_shape[] = {1, 1, BINS, 2}, state_shape[] = {STATE};
    ort_check(p, p->ort->CreateTensorWithDataAsOrtValue(
                     p->memory, p->spectrum, sizeof(p->spectrum), spec_shape, 4,
                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &p->spec_input));
    ort_check(p, p->ort->CreateTensorWithDataAsOrtValue(
                     p->memory, p->enhanced, sizeof(p->enhanced), spec_shape, 4,
                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &p->spec_output));
    for (int i = 0; i < 2; ++i)
      ort_check(p,
                p->ort->CreateTensorWithDataAsOrtValue(
                    p->memory, p->state[i], sizeof(p->state[i]), state_shape, 1,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &p->state_values[i]));
  }
}
static void normalize(Pipeline *p) {
  float magnitude[BINS];
  for (int f = 0; f < BINS; ++f) {
    float r = p->spectrum[f][0] / FFT, im = p->spectrum[f][1] / FFT;
    magnitude[f] = r * r + im * im;
  }
  for (int b = 0; b < 32; ++b) {
    float power = 0;
    for (int f = 0; f < BINS; ++f)
      power += magnitude[f] * p->filterbank[f * 32 + b];
    float value = 10.0f * log10f(power + 1e-10f);
    p->norm[b] = .98f * p->norm[b] + .02f * value;
    p->erb[b] = (value - p->norm[b]) * .025f;
  }
  for (int f = 0; f < 96; ++f) {
    p->norm[32 + f] = .98f * p->norm[32 + f] + .02f * sqrtf(magnitude[f]);
    float scale = 1.0f / (FFT * sqrtf(p->norm[32 + f] + 1e-12f));
    p->features[2 * f] = p->spectrum[f][0] * scale;
    p->features[2 * f + 1] = p->spectrum[f][1] * scale;
  }
  for (int i = 0; i < 128; ++i)
    p->half_state[i] = (__fp16)p->norm[i];
}
static void inference(Pipeline *p) {
  if (p->use_npu) {
    normalize(p);

    __fp16 *spec = p->input_mem[0]->virt_addr;
    unsigned stride = p->native_input[0].w_stride;
    for (int i = 0; i < BINS; i++) {
      spec[i * stride] = p->spectrum[i][0];
      spec[i * stride + 1] = p->spectrum[i][1];
    }
    memcpy(p->input_mem[1]->virt_addr, p->half_state, sizeof(p->half_state));
    __fp16 *erb = p->input_mem[2]->virt_addr,
           *feature = p->input_mem[3]->virt_addr;
    for (int i = 0; i < 32; i++)
      erb[i] = p->erb[i];
    for (int i = 0; i < 192; i++)
      feature[i] = p->features[i];
    /* Synchronous run with the runtime's default cache synchronization enabled.
     */
    check(rknn_run(p->npu, NULL), "run bound I/O");
    __fp16 *enhanced = p->output_mem[0]->virt_addr;
    for (int i = 0; i < BINS * 2; i++)
      ((float *)p->enhanced)[i] = enhanced[i];
    memcpy(p->half_state, p->output_mem[1]->virt_addr, sizeof(p->half_state));

  } else {
    const char *in_names[] = {"spec", "state_in"},
               *out_names[] = {"spec_e", "state_out"};
    const OrtValue *inputs[] = {p->spec_input, p->state_values[p->pong]};
    OrtValue *outputs[] = {p->spec_output, p->state_values[1 - p->pong]};
    ort_check(p, p->ort->Run(p->session, NULL, in_names, inputs, 2, out_names,
                             2, outputs));
    p->pong = 1 - p->pong;
  }
  float *out = (float *)p->enhanced;
  for (int i = 0; i < BINS * 2; ++i)
    if (!isfinite(out[i]))
      fail("Nonfinite enhanced spectrum");
  if (p->use_npu) {
    for (int i = 0; i < STATE; ++i) {
      uint16_t bits;
      memcpy(&bits, &p->half_state[i], sizeof(bits));
      if ((bits & 0x7c00) == 0x7c00)
        fail("Nonfinite recurrent state");
    }
  } else {
    float *state = p->state[p->pong];
    for (int i = 0; i < STATE; ++i)
      if (!isfinite(state[i]))
        fail("Nonfinite recurrent state");
  }
}
static void process(Pipeline *p, const int16_t *pcm, float *output) {
  memmove(p->history, p->history + HOP, HOP * sizeof(float));
  for (int i = 0; i < HOP; ++i)
    p->history[HOP + i] = pcm[i] / 32768.0f;
  for (int i = 0; i < FFT; ++i)
    p->frame[i] = p->history[i] * p->window[i];
  fftwf_execute(p->forward);
  inference(p);
  fftwf_execute(p->inverse);
  for (int i = 0; i < HOP; ++i) {
    output[i] = p->ola[i] + p->frame[i] * p->window[i] / FFT;
    p->ola[i] = p->frame[HOP + i] * p->window[HOP + i] / FFT;
  }
}
static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static void pace_until(double target) {
  struct timespec t = {(time_t)(target / 1000.0),
                       (long)(fmod(target, 1000.0) * 1e6)};
  int err;
  do {
    err = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);
  } while (err == EINTR);
  check(err, "pace");
}
static void release(Pipeline *p) {
  fftwf_destroy_plan(p->forward);
  fftwf_destroy_plan(p->inverse);
  if (p->use_npu) {
    for (int i = 0; i < 4; i++)
      check(rknn_destroy_mem(p->npu, p->input_mem[i]), "destroy input mem");
    for (int i = 0; i < 2; i++)
      check(rknn_destroy_mem(p->npu, p->output_mem[i]), "destroy output mem");
    check(rknn_destroy(p->npu), "destroy");
  } else {
    p->ort->ReleaseValue(p->spec_input);
    p->ort->ReleaseValue(p->spec_output);
    p->ort->ReleaseValue(p->state_values[0]);
    p->ort->ReleaseValue(p->state_values[1]);
    p->ort->ReleaseSession(p->session);
    p->ort->ReleaseMemoryInfo(p->memory);
    p->ort->ReleaseEnv(p->env);
  }
}
int main(int argc, char **argv) {
  if (argc < 4 || argc > 7)
    fail("Usage: dpdfnet_pcm ort|rknn MODEL ASSETS [cores=1] "
         "[--pace|--unpaced] [threads=1]");
  int use_npu = !strcmp(argv[1], "rknn");
  if (!use_npu && strcmp(argv[1], "ort"))
    fail("Expected ort or rknn");
  char *end = NULL;
  long requested = argc > 4 ? strtol(argv[4], &end, 10) : 1;
  if (argc > 4 && (end == argv[4] || *end))
    fail("Invalid core mask");
  if (requested < 1 || requested > 7)
    fail("Invalid core mask");
  int cores = (int)requested;
  int paced = argc > 5 && !strcmp(argv[5], "--pace");
  if (argc > 5 && !paced && strcmp(argv[5], "--unpaced"))
    fail("Invalid pacing option");
  long requested_threads = argc > 6 ? strtol(argv[6], &end, 10) : 1;
  if (argc > 6 && (end == argv[6] || *end))
    fail("Invalid thread count");
  int threads = (int)requested_threads;
  if (requested_threads < 1 || requested_threads > 8)
    fail("Invalid core or thread count");
  int audio_fd = dup(fileno(stdout));
  if (audio_fd < 0 || dup2(fileno(stderr), fileno(stdout)) < 0)
    fail("Cannot separate audio and logs");
  FILE *dest = fdopen(audio_fd, "wb");
  if (!dest)
    fail("Cannot open PCM output");
  Pipeline *p = calloc(1, sizeof(*p));
  if (!p)
    fail("Allocation failed");
  p->use_npu = use_npu;
  prepare(p, argv[2], argv[3], cores, threads);
  /* Warm up kernels, then restore all recurrent and spectral history. */
  int16_t zero[HOP] = {0};
  float output[HOP];
  process(p, zero, output);
  read_asset(argv[3], "initial-state.bin", p->state[0], STATE);
  memcpy(p->state[1], p->state[0], sizeof(p->state[0]));
  memcpy(p->norm, p->state[0], sizeof(p->norm));
  for (int i = 0; i < STATE; ++i)
    p->half_state[i] = (__fp16)p->state[0][i];
  p->pong = 0;
  memset(p->history, 0, sizeof(p->history));
  memset(p->ola, 0, sizeof(p->ola));
  /* Bound statistics storage for continuous microphone streams. */
  size_t capacity = 65536, blocks = 0, input_count = 0, emitted = 0, late = 0;
  double *service = malloc(capacity * sizeof(double)),
         *lag = malloc(capacity * sizeof(double));
  if (!service || !lag)
    fail("Timing allocation failed");
  double start = now_ms(), max_queue = 0, queue = 0, first = 0;
  double total = 0, maximum_service = 0, maximum_pcm = 0;
  size_t read_count, bytes;
  while ((bytes = fread(zero, 1, sizeof(zero), stdin)) > 0) {
    if (bytes % sizeof(int16_t))
      fail("Incomplete S16_LE sample");
    read_count = bytes / sizeof(int16_t);
    input_count += read_count;
    memset(zero + read_count, 0, (HOP - read_count) * sizeof(int16_t));
    if (paced)
      pace_until(start + input_count / 16.0);
    double tick = now_ms();
    process(p, zero, output);
    double cost = now_ms() - tick;
    size_t slot = blocks % capacity;
    total += cost;
    maximum_service = fmax(maximum_service, cost);
    service[slot] = cost;
    if (cost > 10)
      ++late;
    queue = fmax(0, queue + cost - 10);
    max_queue = fmax(max_queue, queue);
    lag[slot] = 0;
    /* One STFT startup hop and four delayed spectrum hops precede aligned
     * audio. */
    if (blocks >= STARTUP_HOPS) {
      size_t n = input_count - emitted;
      if (n > HOP)
        n = HOP;
      int16_t pcm[HOP];
      for (size_t i = 0; i < n; ++i)
        pcm[i] =
            (int16_t)lrintf(fminf(32767, fmaxf(-32768, output[i] * 32768)));
      if (fwrite(pcm, sizeof(int16_t), n, dest) != n || fflush(dest))
        fail("PCM write failed");
      double origin = now_ms() - start - emitted / 16.0;
      if (!emitted)
        first = origin;
      lag[slot] = paced ? origin : 0;
      maximum_pcm = fmax(maximum_pcm, origin);
      emitted += n;
    }
    ++blocks;
  }
  if (ferror(stdin))
    fail("PCM read failed");
  size_t processed = blocks;
  memset(zero, 0, sizeof(zero));
  while (emitted < input_count) {
    process(p, zero, output);
    if (processed >= STARTUP_HOPS) {
      size_t n = input_count - emitted;
      if (n > HOP)
        n = HOP;
      int16_t pcm[HOP];
      for (size_t i = 0; i < n; ++i)
        pcm[i] =
            (int16_t)lrintf(fminf(32767, fmaxf(-32768, output[i] * 32768)));
      if (fwrite(pcm, sizeof(int16_t), n, dest) != n || fflush(dest))
        fail("Flush failed");
      emitted += n;
    }
    ++processed;
  }
  if (emitted != input_count)
    fail("PCM length mismatch");
  size_t stored = blocks < capacity ? blocks : capacity;
  qsort(service, stored, sizeof(double), compare_double);
  qsort(lag, stored, sizeof(double), compare_double);
  size_t p95 = stored ? (size_t)(.95 * (stored - 1)) : 0;
  fprintf(stderr,
          "{\"backend\":\"%s\",\"samples\":%zu,\"blocks\":%zu,\"timing_window_"
          "blocks\":%zu,"
          "\"mean_ms\":%.6f,\"p95_service_ms\":%.6f,\"max_service_ms\":%.6f,"
          "\"over_budget\":%zu,\"rtf\":%.6f,\"max_queue_ms\":%.6f,\"paced\":%s,"
          "\"first_pcm_ms\":%.6f,\"p95_pcm_ms\":%.6f,\"max_pcm_ms\":%.6f,"
          "\"compensated_delay_"
          "samples\":640}\n",
          argv[1], emitted, blocks, stored, blocks ? total / blocks : 0,
          stored ? service[p95] : 0, maximum_service, late,
          input_count ? total / (input_count / 16.0) : 0, max_queue,
          paced ? "true" : "false", paced ? first : 0,
          paced && stored ? lag[p95] : 0, paced ? maximum_pcm : 0);
  free(service);
  free(lag);
  release(p);
  free(p);
  fclose(dest);
  return 0;
}
