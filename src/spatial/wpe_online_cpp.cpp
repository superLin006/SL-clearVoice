#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using cpx = std::complex<float>;
static constexpr int NFFT = 512, HOP = 256, NF = NFFT / 2 + 1;
static constexpr int TAPS = 10, DELAY = 3, D = 6;
static int N_POS = 8;
static int ORDER[D] = {0, 1, 5, 4, 2, 3};
static float THETA = 294.0f, RADIUS = 0.0476f;
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

struct Stft {
    std::vector<float> win;
    Stft() {
        win.resize(NFFT);
        for (int i = 0; i < NFFT; ++i) win[i] = 0.5f - 0.5f * std::cos(2 * PI * i / NFFT);
    }
    int frames(int len) const { return 1 + (len + NFFT - NFFT) / HOP; }
    std::vector<std::vector<cpx>> forward(const std::vector<float>& x) const {
        int len = (int)x.size();
        int pad = NFFT / 2;
        int n = 1 + (len + 2 * pad - NFFT) / HOP;
        if (n < 1) n = 1;
        std::vector<std::vector<cpx>> out(n, std::vector<cpx>(NF));
        std::vector<cpx> buf(NFFT);
        for (int t = 0; t < n; ++t) {
            int s = t * HOP - pad;
            for (int i = 0; i < NFFT; ++i) {
                int p = s + i;
                float v = (p >= 0 && p < len) ? x[p] : 0.0f;
                buf[i] = cpx(v * win[i], 0);
            }
            fft(buf, false);
            for (int f = 0; f < NF; ++f) out[t][f] = buf[f];
        }
        return out;
    }
    std::vector<float> inverse(const std::vector<std::vector<cpx>>& spec, int len) const {
        int n = (int)spec.size();
        int pad = NFFT / 2;
        std::vector<float> out(len + 2 * pad, 0.0f), norm(len + 2 * pad, 0.0f);
        std::vector<cpx> buf(NFFT);
        for (int t = 0; t < n; ++t) {
            for (int f = 0; f < NF; ++f) buf[f] = spec[t][f];
            for (int f = NF; f < NFFT; ++f) buf[f] = std::conj(spec[t][NFFT - f]);
            fft(buf, true);
            int s = t * HOP;
            for (int i = 0; i < NFFT; ++i) {
                out[s + i] += buf[i].real() * win[i];
                norm[s + i] += win[i] * win[i];
            }
        }
        std::vector<float> y(len);
        for (int i = 0; i < len; ++i) y[i] = out[i + pad] / std::max(norm[i + pad], 1e-8f);
        return y;
    }
};

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

static void beamform(const std::vector<std::vector<cpx>>& spec, std::vector<std::vector<cpx>>& out) {
    const float R = 0.042f, C = 343.0f, theta = 86.0f * PI / 180.0f;
    float tau[D];
    for (int d = 0; d < D; ++d) tau[d] = -(R / C) * std::cos(theta - 2 * PI * d / D);
    int n = (int)spec.size();
    out.assign(n, std::vector<cpx>(NF));
    #pragma omp parallel for
    for (int t = 0; t < n; ++t)
        for (int f = 0; f < NF; ++f) {
            float hz = (float)f * 16000.0f / NFFT;
            cpx acc(0, 0);
            for (int d = 0; d < D; ++d) {
                float ph = -2 * PI * hz * tau[d];
                acc += spec[t][f * D + d] * cpx(std::cos(ph), -std::sin(ph));
            }
            out[t][f] = acc / (float)D;
        }
}

static std::vector<float> read_raw(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t n = f.tellg();
    f.seekg(0);
    std::vector<float> x(n / sizeof(float));
    f.read(reinterpret_cast<char*>(x.data()), n);
    return x;
}

int main(int argc, char** argv) {
    auto t_start = std::chrono::steady_clock::now();
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " in.f32 out.f32 [channels]\\n";
        return 1;
    }
    int ch = argc > 3 ? std::atoi(argv[3]) : D;
    if (argc > 4) N_POS = std::atoi(argv[4]);
    if (argc > 5) {
        std::string s = argv[5];
        for (int d = 0; d < D; ++d) {
            auto p = s.find(',');
            ORDER[d] = std::atoi(s.substr(0, p).c_str());
            if (p == std::string::npos) break;
            s = s.substr(p + 1);
        }
    }
    if (argc > 6) THETA = std::atof(argv[6]);
    if (argc > 7) RADIUS = std::atof(argv[7]);
    auto inter = read_raw(argv[1]);
    int frames = (int)inter.size() / ch;
    std::vector<float> mono(frames);
    bool spatial = ch >= D;
    if (spatial) {
        std::vector<float> rms(D, 0.0f);
        for (int d = 0; d < D; ++d) {
            double e = 0;
            for (int i = 0; i < frames; ++i) e += (double)inter[i * ch + d] * inter[i * ch + d];
            rms[d] = std::sqrt(e / std::max(1, frames));
        }
        std::vector<float> sorted = rms;
        std::sort(sorted.begin(), sorted.end());
        float med = sorted[D / 2], mn = sorted.front();
        if (mn < 1e-6f || mn < 0.05f * med) spatial = false;
        if (!spatial) std::cerr << "bad microphone rms " << mn << " median " << med << std::endl;
    }
    if (!spatial) {
        for (int i = 0; i < frames; ++i) mono[i] = inter[i * ch];
        std::ofstream o(argv[2], std::ios::binary);
        o.write(reinterpret_cast<char*>(mono.data()), mono.size() * sizeof(float));
        std::cerr << "channel " << ch << " bypass spatial chain" << std::endl;
        return 0;
    }
    Stft stft;
    std::vector<std::vector<cpx>> spec(frames > 0 ? stft.frames(frames) : 0);
    std::vector<std::vector<cpx>> input(spec.size(), std::vector<cpx>(NF * D));
    #pragma omp parallel for
    for (int t = 0; t < (int)spec.size(); ++t) {
        int s = t * HOP - NFFT / 2;
        for (int f = 0; f < NF; ++f) {
            // build per-channel frames via small DFT on the fly is too slow; use Stft per channel below
        }
    }
    // per-channel STFT
    std::vector<std::vector<std::vector<cpx>>> chspec(D);
    for (int d = 0; d < D; ++d) {
        std::vector<float> x(frames);
        for (int i = 0; i < frames; ++i) x[i] = inter[i * ch + d];
        chspec[d] = stft.forward(x);
    }
    int T = (int)chspec[0].size();
    OnlineWpe wpe;
    std::vector<std::vector<cpx>> wpe_out(T, std::vector<cpx>(NF * D));
    bool bypass = std::getenv("BYPASS") != nullptr;
    for (int t = 0; t < T; ++t) {
        if (bypass) {
            for (int f = 0; f < NF; ++f)
                for (int d = 0; d < D; ++d) wpe_out[t][f * D + d] = chspec[d][t][f];
            continue;
        }
        std::vector<cpx> frame(NF * D), pred;
        for (int f = 0; f < NF; ++f)
            for (int d = 0; d < D; ++d) frame[f * D + d] = chspec[d][t][f];
        wpe.step(frame, pred);
        wpe_out[t] = pred;
    }
    std::vector<std::vector<cpx>> bf;
    beamform(wpe_out, bf);
    auto y = stft.inverse(bf, frames);
    std::ofstream o(argv[2], std::ios::binary);
    o.write(reinterpret_cast<char*>(y.data()), y.size() * sizeof(float));
    std::cerr << "wrote " << y.size() << " samples\\n";
    return 0;
}
