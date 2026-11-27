#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "onnxruntime_c_api.h"

enum { MAX_IO = 32, MAX_DIMS = 8, FRAMES = 8, BINS = 256, BLOCK_NS = 128000000 };
static const OrtApi *ort;

static void checked(OrtStatus *status) {
    if (status) {
        fprintf(stderr, "%s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        exit(1);
    }
}

static int64_t timestamp_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}

static void wait_until(int64_t ns) {
    struct timespec deadline = {.tv_sec = ns / 1000000000, .tv_nsec = ns % 1000000000};
    int rc;
    do {
        rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, NULL);
    } while (rc == EINTR);
    if (rc) exit(3);
}

static void open_input(OrtSession *session, size_t index, OrtMemoryInfo *mem, void *buffer,
                       size_t buffer_size, OrtValue **value, float **owned_buffer) {
    OrtTypeInfo *type;
    const OrtTensorTypeAndShapeInfo *shape;
    int64_t dims[MAX_DIMS];
    size_t rank, size;
    checked(ort->SessionGetInputTypeInfo(session, index, &type));
    checked(ort->CastTypeInfoToTensorInfo(type, &shape));
    checked(ort->GetDimensionsCount(shape, &rank));
    if (rank > MAX_DIMS) exit(2);
    checked(ort->GetDimensions(shape, dims, rank));
    checked(ort->GetTensorShapeElementCount(shape, &size));
    ort->ReleaseTypeInfo(type);
    if (!size || size > 2000000) exit(2);
    if (!buffer) {
        *owned_buffer = calloc(size, sizeof(float));
        buffer = *owned_buffer;
        buffer_size = size * sizeof(float);
        if (!buffer) exit(2);
    }
    if (buffer_size != size * sizeof(float)) exit(2);
    checked(ort->CreateTensorWithDataAsOrtValue(mem, buffer, buffer_size, dims, rank,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, value));
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "Usage: %s MODEL SPECTRUM_F32 THREADS BLOCKS PACE_0_OR_1 OUTPUT_F32\n",
                argv[0]);
        return 2;
    }
    int threads = atoi(argv[3]), blocks = atoi(argv[4]), pace = atoi(argv[5]);
    if (threads < 1 || threads > 8 || blocks <= 0 || (pace != 0 && pace != 1)) return 2;
    FILE *source = fopen(argv[2], "rb"), *destination = fopen(argv[6], "wb");
    if (!source || !destination) return 2;

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv *env;
    OrtSessionOptions *options;
    OrtSession *session;
    OrtMemoryInfo *mem;
    OrtAllocator *allocator;
    checked(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "driftse-paced", &env));
    checked(ort->CreateSessionOptions(&options));
    checked(ort->SetIntraOpNumThreads(options, threads));
    checked(ort->SetInterOpNumThreads(options, 1));
    checked(ort->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL));
    checked(ort->CreateSession(env, argv[1], options, &session));
    checked(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem));
    checked(ort->GetAllocatorWithDefaultOptions(&allocator));
    size_t n_inputs, n_outputs;
    checked(ort->SessionGetInputCount(session, &n_inputs));
    checked(ort->SessionGetOutputCount(session, &n_outputs));
    if (n_inputs != n_outputs || n_inputs < 2 || n_inputs > MAX_IO) return 2;
    OrtTypeInfo *estimate_type;
    const OrtTensorTypeAndShapeInfo *estimate_shape;
    ONNXTensorElementDataType estimate_dtype;
    size_t estimate_size;
    checked(ort->SessionGetOutputTypeInfo(session, 0, &estimate_type));
    checked(ort->CastTypeInfoToTensorInfo(estimate_type, &estimate_shape));
    checked(ort->GetTensorElementType(estimate_shape, &estimate_dtype));
    checked(ort->GetTensorShapeElementCount(estimate_shape, &estimate_size));
    ort->ReleaseTypeInfo(estimate_type);
    if (estimate_dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        estimate_size != 2 * FRAMES * BINS) return 2;

    char *input_names[MAX_IO], *output_names[MAX_IO];
    OrtValue *input[MAX_IO] = {0};
    float *owned_input[MAX_IO] = {0};
    float features[4 * FRAMES * BINS];
    for (size_t i = 0; i < n_inputs; ++i) {
        checked(ort->SessionGetInputName(session, i, allocator, &input_names[i]));
        checked(ort->SessionGetOutputName(session, i, allocator, &output_names[i]));
        open_input(session, i, mem, i ? NULL : features, sizeof(features),
                   &input[i], &owned_input[i]);
    }
    printf("onnxruntime=%s threads=%d\n", OrtGetApiBase()->GetVersionString(), threads);
    int64_t origin = timestamp_ns();
    for (int frame = 0; frame < blocks; ++frame) {
        if (fread(features, 1, sizeof(features), source) != sizeof(features)) return 2;
        int64_t ready = origin + (int64_t)(frame + 1) * BLOCK_NS;
        if (pace) wait_until(ready);
        int64_t start = timestamp_ns();
        OrtValue *result[MAX_IO] = {0};
        checked(ort->Run(session, NULL, (const char *const *)input_names,
                         (const OrtValue *const *)input, n_inputs,
                         (const char *const *)output_names, n_outputs, result));
        int64_t finished = timestamp_ns();
        void *estimate;
        checked(ort->GetTensorMutableData(result[0], &estimate));
        for (size_t i = 0; i < 2 * FRAMES * BINS; ++i)
            if (!isfinite(((float *)estimate)[i])) return 4;
        if (fwrite(estimate, sizeof(float), 2 * FRAMES * BINS, destination) != 2 * FRAMES * BINS)
            return 2;
        ort->ReleaseValue(result[0]);
        for (size_t i = 1; i < n_inputs; ++i) {
            ort->ReleaseValue(input[i]);
            input[i] = result[i];
        }
        printf("block=%d compute_ms=%.3f queue_ms=%.3f ready_to_result_ms=%.3f\n",
               frame, (finished - start) / 1e6, pace ? (start - ready) / 1e6 : 0.0,
               pace ? (finished - ready) / 1e6 : 0.0);
        if ((frame + 1) % 50 == 0) fflush(stdout);
    }
    int input_closed = fclose(source);
    int output_closed = fclose(destination);
    if (input_closed || output_closed) return 2;
    for (size_t i = 0; i < n_inputs; ++i) {
        ort->ReleaseValue(input[i]);
        allocator->Free(allocator, input_names[i]);
        allocator->Free(allocator, output_names[i]);
        free(owned_input[i]);
    }
    ort->ReleaseMemoryInfo(mem);
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(options);
    ort->ReleaseEnv(env);
    return 0;
}
