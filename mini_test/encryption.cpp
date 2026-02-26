#include "encryption.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "util.h"
#include "seal/util/polycore.h"
#include "seal/util/rlwe.h"

using namespace seal;

static inline uint64_t convert_qseal_to_mod2k_signed(uint64_t x_qseal, uint64_t q_seal, uint64_t log_q)
{
    const uint64_t mask = (1ULL << log_q) - 1ULL;
    // Interpret coefficient as centered integer in (-q_seal/2, q_seal/2], then remap to mod 2^k.
    int64_t centered = (x_qseal <= (q_seal >> 1))
                           ? static_cast<int64_t>(x_qseal)
                           : static_cast<int64_t>(x_qseal) - static_cast<int64_t>(q_seal);
    return static_cast<uint64_t>(centered) & mask;
}

void sample_noise_mod2k_seal_cbd(
    const EncryptionParameters &parms, uint64_t q_seal, uint64_t log_q, std::vector<uint64_t> &noise_out)
{
    if (log_q == 0 || log_q > 62) throw std::invalid_argument("log_q must be in [1,62]");
    const size_t n = parms.poly_modulus_degree();

    noise_out.assign(n, 0);
    MemoryPoolHandle pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);
    auto noise = seal::util::allocate_poly(n, /*coeff_modulus_size=*/1, pool);
    auto noise_prng = parms.random_generator()->create();
    seal::util::sample_poly_cbd(noise_prng, parms, noise.get());
    for (size_t i = 0; i < n; ++i) {
        noise_out[i] = convert_qseal_to_mod2k_signed(noise[i], q_seal, log_q);
    }
}

void sample_noise_mod2k_direct_ternary(size_t coeff_count, uint64_t log_q, std::vector<uint64_t> &noise_out)
{
    if (log_q == 0 || log_q > 62) throw std::invalid_argument("log_q must be in [1,62]");
    const uint64_t mask = (1ULL << log_q) - 1ULL;

    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int> ternary(-1, 1);

    noise_out.assign(coeff_count, 0);
    for (size_t i = 0; i < coeff_count; ++i) {
        const int e = ternary(rng);
        if (e < 0) noise_out[i] = mask;
        else noise_out[i] = static_cast<uint64_t>(e);
    }
}

static inline size_t idx_nhwc(int h, int w, int c, int W, int C) {
    return static_cast<size_t>((h * W + w) * C + c);
}

// stride = 1로 가정
// 출력 크기가 유지되게 패딩
// 우선 이미지 크기가 N보다 작을 때만 생각하고, packing도 일단 고려하지 않음. 남은 부분은 0으로 채움
std::vector<uint64_t> pad_same_to_poly_n(
    const std::vector<uint64_t> &input_flat,
    int H, int W, int C,
    int kernel_k,
    int poly_N)
{
    if (H <= 0 || W <= 0 || C <= 0) {
        throw std::invalid_argument("Invalid H/W/C");
    }
    if (kernel_k <= 0 || (kernel_k % 2 == 0)) {
        throw std::invalid_argument("kernel_k must be a positive odd number");
    }
    const size_t expected = static_cast<size_t>(H) * W * C;
    if (input_flat.size() != expected) {
        throw std::invalid_argument("input_flat size mismatch (expected H*W*C)");
    }
    if (poly_N <= 0) {
        throw std::invalid_argument("poly_N must be positive");
    }

    const int pad_h = (kernel_k - 1) / 2;
    const int pad_w = (kernel_k - 1) / 2;

    const int Hpad = H + 2 * pad_h;
    const int Wpad = W + 2 * pad_w;

    const size_t padded_elems = static_cast<size_t>(Hpad) * Wpad * C;
    if (padded_elems > static_cast<size_t>(poly_N)) {
        throw std::invalid_argument("poly_N is too small for padded tensor (Hpad*Wpad*C)");
    }

    std::vector<uint64_t> out(static_cast<size_t>(poly_N), 0);

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const int ph = h + pad_h;
            const int pw = w + pad_w;
            for (int c = 0; c < C; ++c) {
                out[idx_nhwc(ph, pw, c, Wpad, C)] =
                    input_flat[idx_nhwc(h, w, c, W, C)];
            }
        }
    }

    return out;
}


seal::Plaintext vector_to_plaintext_coeff(
    const std::vector<uint64_t> &coeffs_mod2k,
    size_t poly_N,
    int k // for defensive masking; can be omitted if already masked
) {
    if (poly_N == 0) throw std::invalid_argument("poly_N must be > 0");
    if (coeffs_mod2k.size() != poly_N) {
        throw std::invalid_argument("coeffs_mod2k size must equal poly_N");
    }
    if (k <= 0 || k > 62) throw std::invalid_argument("k must be in [1, 62]");
    const uint64_t mask = (1ULL << k) - 1ULL;

    seal::Plaintext pt;
    pt.resize(poly_N);
    for (size_t i = 0; i < poly_N; ++i) {
        pt[i] = coeffs_mod2k[i] & mask;
    }
    return pt;
}

void scale_by_pow2_inplace(
    std::vector<uint64_t> &coeffs,
    int delta_shift,   // log2(delta), e.g., 20
    int k              // modulus bitwidth, e.g., 60
) {
    if (k <= 0 || k > 62) throw std::invalid_argument("k must be in [1, 62]");
    if (delta_shift < 0 || delta_shift >= k) {
        throw std::invalid_argument("delta_shift must satisfy 0 <= delta_shift < k");
    }
    const uint64_t mask = (1ULL << k) - 1ULL;

    for (auto &x : coeffs) {
        uint64_t u = static_cast<uint64_t>(x) & mask;      // mod 2^k
        u = (u << delta_shift) & mask;                     // multiply by 2^delta_shift mod 2^k
        x = static_cast<int64_t>(u);                       // keep stored as int64_t (still mod space)
    }
}

void encrypt_zero_nttfree(
    const SEALContext &context, Ciphertext &temp_out, const Plaintext &sk, uint64_t log_q)
{
    auto &context_data = *context.first_context_data();
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();

    // Assumptions for this path:
    // - Single coeff modulus in SEAL parameters (even if it is prime)
    // - We "quantize" all arithmetic to mod 2^log_q by masking (NTT-free path)
    if (coeff_modulus.size() != 1) {
        throw std::invalid_argument("encrypt_zero_nttfree expects single coeff modulus");
    }
    if (log_q == 0 || log_q > 62) {
        throw std::invalid_argument("log_q must be in [1,62]");
    }

    const size_t N = parms.poly_modulus_degree();
    const uint64_t q_seal = coeff_modulus[0].value();
    const uint64_t mask = (1ULL << log_q) - 1ULL;

    if (sk.coeff_count() != N) {
        throw std::invalid_argument("Secret key plaintext size mismatch");
    }

    // Allocate ciphertext (c0, c1)
    temp_out.resize(context, 2);
    temp_out.is_ntt_form() = false;

    uint64_t *c0 = temp_out.data(0);
    uint64_t *c1 = temp_out.data(1);

    // 1) Sample a <- uniform  (store into c1)
    prng_seed_type seed{};
    auto prng = UniformRandomGeneratorFactory::DefaultFactory()->create(seed);
    seal::util::sample_poly_uniform(prng, parms, c1);

    // Quantize c1 to mod 2^log_q
    reduce_poly_mod2k(c1, c1, N, log_q);

    // 2) Sample error e mod 2^k.
    // Default: SEAL CBD sampler -> centered-lift from mod q_seal -> mod 2^k.
    // Alternative (direct ternary in mod 2^k): sample_noise_mod2k_direct_ternary(N, log_q, noise).
    std::vector<uint64_t> noise;
    // sample_noise_mod2k_seal_cbd(parms, q_seal, log_q, noise);
    sample_noise_mod2k_direct_ternary(N, log_q, noise);

    // 3) Compute t = a*s mod (x^N+1) without NTT, then quantize
    Plaintext t;
    t.resize(N);
    multiply_poly_secret_mod2k(c1, sk.data(), q_seal, t.data(), N, log_q);

    // 4) t = t + e mod 2^log_q
    for (size_t i = 0; i < N; ++i) {
        t.data()[i] = (t.data()[i] + noise[i]) & mask;
    }

    // 5) c0 = -t mod 2^log_q
    for (size_t i = 0; i < N; ++i) {
        c0[i] = (0ULL - (t.data()[i] & mask)) & mask;
    }
}
