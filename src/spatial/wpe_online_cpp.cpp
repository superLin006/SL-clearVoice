#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

using cpx = std::complex<float>;
static constexpr int NFFT = 512, HOP = 256, NF = NFFT / 2 + 1;
static constexpr int TAPS = 10, DELAY = 3, D = 6;
struct Geometry {
    int positions = D;
    // Input channel associated with each populated position in the ring.
    std::array<int, D> channels = {0, 1, 2, 3, 4, 5};
    float degrees = 86.0f;
    float radius = 0.042f;
};
static constexpr int BUF = TAPS + DELAY + 1;
static constexpr float PI = 3.14159265358979323846f;
static constexpr float ALPHA = 0.9999f;

static void fft(std::vector<cpx>& a, bool inv) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * PI / len * (inv ? 1 : -1);
        cpx wl(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            cpx w(1, 0);
            for (int j = 0; j < len / 2; ++j) {
                cpx u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wl;
            }
        }
    }
    if (inv) for (auto& x : a) x /= (float)n;
}

struct OnlineWpe {
    std::vector<cpx> inv_cov;  // F * (TAPS*D)^2
    std::vector<cpx> taps;     // F * (TAPS*D) * D
    std::vector<float> power;  // F
    std::vector<cpx> buffer;   // BUF * F * D
    std::vector<cpx> window_, nominator_, gain_, v_;
    int K = TAPS * D;
    OnlineWpe() {
        inv_cov.assign((size_t)NF * K * K, cpx(0, 0));
        for (int f = 0; f < NF; ++f)
            for (int i = 0; i < K; ++i) inv_cov[((size_t)f * K + i) * K + i] = cpx(1, 0);
        taps.assign((size_t)NF * K * D, cpx(0, 0));
        power.assign(NF, 1.0f);
        buffer.assign((size_t)BUF * NF * D, cpx(0, 0));
        window_.assign((size_t)NF * K, cpx(0, 0));
        nominator_.assign((size_t)NF * K, cpx(0, 0));
        gain_.assign((size_t)NF * K, cpx(0, 0));
        v_.assign((size_t)NF * K, cpx(0, 0));
    }
    void step(const std::vector<cpx>& frame, std::vector<cpx>& pred) {
        pred.resize(NF * D);
        // window = buffer[0..TAPS-1]
        #pragma omp parallel for
        for (int f = 0; f < NF; ++f) {
            for (int i = 0; i < TAPS; ++i)
                for (int d = 0; d < D; ++d)
                    window_[f * K + d * TAPS + i] = buffer[((size_t)i * NF + f) * D + d];
            for (int d = 0; d < D; ++d) {
                cpx acc(0, 0);
                for (int i = 0; i < K; ++i)
                    acc += std::conj(taps[((size_t)f * K + i) * D + d]) * window_[f * K + i];
                pred[f * D + d] = frame[f * D + d] - acc;
            }
        }
        // update buffer
        #pragma omp parallel for
        for (int f = 0; f < NF; ++f)
            for (int i = 1; i < BUF; ++i)
                for (int d = 0; d < D; ++d)
                    buffer[((size_t)(i - 1) * NF + f) * D + d] = buffer[((size_t)i * NF + f) * D + d];
        for (int f = 0; f < NF; ++f)
            for (int d = 0; d < D; ++d)
                buffer[((size_t)(BUF - 1) * NF + f) * D + d] = frame[f * D + d];
        // power mean over buffer frames and channels
        #pragma omp parallel for
        for (int f = 0; f < NF; ++f) {
            float p = 0;
            for (int i = 0; i < BUF; ++i)
                for (int d = 0; d < D; ++d) {
                    cpx v = buffer[((size_t)i * NF + f) * D + d];
                    p += std::norm(v);
                }
            power[f] = p / (BUF * D);
        }
        #pragma omp parallel for
        for (int f = 0; f < NF; ++f) {
            cpx denom = cpx(ALPHA * power[f], 0);
            for (int i = 0; i < K; ++i) {
                cpx acc(0, 0);
                for (int j = 0; j < K; ++j)
                    acc += inv_cov[((size_t)f * K + i) * K + j] * window_[f * K + j];
                nominator_[f * K + i] = acc;
                denom += std::conj(window_[f * K + i]) * acc;
            }
            float dd = std::max(std::abs(denom), 1e-10f);
            for (int i = 0; i < K; ++i) gain_[f * K + i] = nominator_[f * K + i] / dd;
            // inv_cov -= gain_i * sum_j conj(window_j) * inv_cov[j][m]
            for (int m = 0; m < K; ++m) {
                cpx acc(0, 0);
                for (int j = 0; j < K; ++j)
                    acc += std::conj(window_[f * K + j]) * inv_cov[((size_t)f * K + j) * K + m];
                v_[f * K + m] = acc;
            }
            for (int i = 0; i < K; ++i)
                for (int m = 0; m < K; ++m)
                    inv_cov[((size_t)f * K + i) * K + m] =
                        (inv_cov[((size_t)f * K + i) * K + m] - gain_[f * K + i] * v_[f * K + m]) / ALPHA;
            for (int i = 0; i < K; ++i)
                for (int d = 0; d < D; ++d)
                    taps[((size_t)f * K + i) * D + d] += gain_[f * K + i] * std::conj(pred[f * D + d]);
        }
    }
};

static bool parse_geometry(int argc, char** argv, int channels, Geometry& geometry) {
    if (argc > 4) geometry.positions = std::atoi(argv[4]);
    if (argc > 5) {
        const char* token = argv[5];
        for (int d = 0; d < D; ++d) {
            char* end = nullptr;
            long index = std::strtol(token, &end, 10);
            if (end == token || index < 0 || index >= channels ||
                *end != (d + 1 < D ? ',' : '\0')) return false;
            if (std::find(geometry.channels.begin(), geometry.channels.begin() + d, index) !=
                geometry.channels.begin() + d) return false;
            geometry.channels[d] = (int)index;
            token = end + 1;
        }
    }
    if (argc > 6) {
        char* end = nullptr;
        geometry.degrees = std::strtof(argv[6], &end);
        if (end == argv[6] || *end != '\0') return false;
    }
    if (argc > 7) {
        char* end = nullptr;
        geometry.radius = std::strtof(argv[7], &end);
        if (end == argv[7] || *end != '\0') return false;
    }
    return geometry.positions >= D && std::isfinite(geometry.degrees) &&
           std::isfinite(geometry.radius) && geometry.radius > 0;
}

static std::vector<cpx> steering(const Geometry& geometry) {
    std::vector<cpx> weights(NF * D);
    for (int d = 0; d < D; ++d) {
        float tau = -(geometry.radius / 343.0f) *
                    std::cos(geometry.degrees * PI / 180.0f - 2 * PI * d / geometry.positions);
        for (int f = 0; f < NF; ++f) {
            float phase = 2 * PI * f * 16000.0f / NFFT * tau;
            weights[f * D + d] = cpx(std::cos(phase), std::sin(phase)) / (float)D;
        }
    }
    return weights;
}

static bool read_raw(const char* path, int channels, std::vector<float>& audio) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    std::streamoff bytes = input.tellg();
    if (bytes <= 0 || bytes % (sizeof(float) * channels)) return false;
    audio.resize(bytes / sizeof(float));
    input.seekg(0);
    return bool(input.read(reinterpret_cast<char*>(audio.data()), bytes));
}

static bool usable_mics(const std::vector<float>& audio, int channels, const Geometry& geometry) {
    if (channels < D) return false;
    std::array<double, D> rms{};
    const size_t length = audio.size() / channels;
    for (int d = 0; d < D; ++d) {
        for (size_t i = 0; i < length; ++i) {
            double sample = audio[i * channels + geometry.channels[d]];
            rms[d] += sample * sample;
        }
        rms[d] = std::sqrt(rms[d] / length);
    }
    std::sort(rms.begin(), rms.end());
    return rms.front() >= 1e-6 && rms.front() >= 0.05 * rms[D / 2];
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 8) {
        std::cerr << "usage: " << argv[0]
                  << " in.f32 out.f32 [channels [positions [channel_order [degrees [radius_m]]]]]\n";
        return 1;
    }
    int channels = argc > 3 ? std::atoi(argv[3]) : D;
    Geometry geometry;
    std::vector<float> audio;
    if (channels < 1 || !parse_geometry(argc, argv, channels, geometry) ||
        !read_raw(argv[1], channels, audio)) {
        std::cerr << "invalid microphone geometry or interleaved float input\n";
        return 1;
    }
    const int length = (int)(audio.size() / channels);
    std::vector<float> mono(length);
    if (!usable_mics(audio, channels, geometry)) {
        for (int i = 0; i < length; ++i) mono[i] = audio[i * channels];
        std::cerr << "unusable microphone array, passing through channel 0\n";
    } else {
        const auto weights = steering(geometry);
        std::array<float, NFFT> window;
        for (int i = 0; i < NFFT; ++i) window[i] = 0.5f - 0.5f * std::cos(2 * PI * i / NFFT);
        std::vector<cpx> buffer(NFFT), spectrum(NF * D), dereverbed(NF * D);
        std::array<float, HOP> tail{};
        const bool bypass = std::getenv("BYPASS") != nullptr;
        OnlineWpe wpe;
        using Clock = std::chrono::steady_clock;
        std::vector<double> timings;
        const int hops = 1 + (length - 1) / HOP;
        timings.reserve(hops + 1);
        const auto started = Clock::now();
        // Include a zero-padded flush frame so the last partial hop is
        // reconstructed with the same two-window overlap as earlier hops.
        for (int hop = 0; hop <= hops; ++hop) {
            const auto start = Clock::now();
            const int offset = (hop - 1) * HOP;
            for (int d = 0; d < D; ++d) {
                for (int k = 0; k < NFFT; ++k) {
                    int i = offset + k;
                    float sample = i >= 0 && i < length ? audio[i * channels + geometry.channels[d]] : 0;
                    buffer[k] = sample * window[k];
                }
                fft(buffer, false);
                for (int f = 0; f < NF; ++f) spectrum[f * D + d] = buffer[f];
            }
            if (!bypass) wpe.step(spectrum, dereverbed);
            const auto& input = bypass ? spectrum : dereverbed;
            for (int f = 0; f < NF; ++f) {
                cpx sum(0, 0);
                for (int d = 0; d < D; ++d) sum += input[f * D + d] * weights[f * D + d];
                buffer[f] = sum;
            }
            for (int f = NF; f < NFFT; ++f) buffer[f] = std::conj(buffer[NFFT - f]);
            fft(buffer, true);
            for (int k = 0; k < HOP; ++k) {
                if (hop > 0 && offset + k < length) {
                    float norm = window[k] * window[k] + window[k + HOP] * window[k + HOP];
                    mono[offset + k] = (tail[k] + buffer[k].real() * window[k]) / norm;
                }
                tail[k] = buffer[k + HOP].real() * window[k + HOP];
            }
            timings.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        }
        double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
        std::sort(timings.begin(), timings.end());
        size_t over = std::count_if(timings.begin(), timings.end(), [](double ms) { return ms > 16.0; });
        std::cerr << (bypass ? "ds" : "wpe-ds") << " frames " << timings.size()
                  << " p95_ms " << timings[(95 * timings.size() + 99) / 100 - 1]
                  << " max_ms " << timings.back() << " over_16ms " << over
                  << " rtf " << elapsed * 16000.0 / length << '\n';
    }
    if (!std::all_of(mono.begin(), mono.end(), [](float x) { return std::isfinite(x); })) {
        std::cerr << "non-finite spatial output\n";
        return 1;
    }

    std::ofstream output(argv[2], std::ios::binary);
    if (!output.write(reinterpret_cast<const char*>(mono.data()), mono.size() * sizeof(float))) return 1;
    return 0;
}
