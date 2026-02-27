#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "util.h"

using Clock = std::chrono::high_resolution_clock;

template <typename Fn>
static double time_op_us(Fn &&fn, int iters)
{
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = Clock::now();
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
}

int main()
{
    constexpr size_t N = 2048;
    constexpr uint64_t log_q = 54;
    constexpr int64_t shift = 13;
    constexpr int iters = 200000;

    const uint64_t mask = (1ULL << log_q) - 1ULL;

    std::mt19937_64 rng(20260227ULL);
    std::uniform_int_distribution<uint64_t> dist(0, mask);

    std::vector<uint64_t> src(N, 0), dst(N, 0);
    for (size_t i = 0; i < N; ++i) src[i] = dist(rng);

    // Benchmark only rotate_poly_coeff_mod2k kernel:
    // no Ciphertext copy, no src<-ci / ci<-dst overhead.
    const double rot_poly_us =
        time_op_us([&]() { rotate_poly_coeff_mod2k(src.data(), dst.data(), N, shift, log_q); }, iters);

    const uint64_t checksum = dst[0] ^ dst[17] ^ dst[511] ^ dst[1023];

    std::cout << "[Speed: rotate_poly_coeff_mod2k kernel only]\n";
    std::cout << "N=" << N << ", log_q=" << log_q << ", shift=" << shift << ", iters=" << iters << "\n";
    std::cout << "  total_us=" << rot_poly_us << ", avg_us=" << (rot_poly_us / iters) << "\n";
    std::cout << "  checksum=" << checksum << "\n";

    return 0;
}

