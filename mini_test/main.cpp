// mini_test/main.cpp
#include <seal/seal.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "decryption.h"
#include "encryption.h"
#include "util.h"

using namespace seal;

static inline int64_t centered_from_mod2k_main(uint64_t u, uint64_t log_q)
{
    const uint64_t q = (1ULL << log_q);
    const uint64_t half = q >> 1;
    u &= (q - 1ULL);
    if (u < half) return static_cast<int64_t>(u);
    return static_cast<int64_t>(u) - static_cast<int64_t>(q);
}

static std::vector<uint64_t> negacyclic_convolution_mod2k_main(
    const std::vector<uint64_t> &a, const std::vector<uint64_t> &b, uint64_t log_q)
{
    if (a.size() != b.size()) throw std::invalid_argument("size mismatch");
    const size_t n = a.size();
    const uint64_t mask = (1ULL << log_q) - 1ULL;
    std::vector<uint64_t> out(n, 0);

    for (size_t i = 0; i < n; ++i) {
        const uint64_t ai = a[i] & mask;
        if (!ai) continue;
        for (size_t j = 0; j < n; ++j) {
            const uint64_t bj = b[j] & mask;
            if (!bj) continue;
            const uint64_t prod = static_cast<uint64_t>(static_cast<unsigned __int128>(ai) * bj) & mask;
            size_t pos = i + j;
            bool wrapped = false;
            if (pos >= n) {
                pos -= n;
                wrapped = true;
            }
            if (!wrapped)
                out[pos] = (out[pos] + prod) & mask;
            else
                out[pos] = (out[pos] - prod) & mask;
        }
    }
    return out;
}

int main()
{
    // NTT-free custom path uses single coeff modulus + mod 2^k masking.
    constexpr size_t N = 2048;
    constexpr uint64_t log_q = 50;     // mask modulus: 2^50
    constexpr int delta_shift = 20;    // Delta = 2^20

    // Image size is intentionally much smaller than N.
    constexpr int H = 8;
    constexpr int W = 8;
    constexpr int C = 3;
    constexpr int kernel_k = 3;        // SAME padding for conv-like layout

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(N);
    // For N=2048, 55-bit single coeff modulus is typically invalid in SEAL.
    parms.set_coeff_modulus(CoeffModulus::Create(N, { 50 })); // single prime for this path
    parms.set_plain_modulus(PlainModulus::Batching(N, 20));

    SEALContext context(parms);
    if (!context.parameters_set()) {
        std::cerr << "SEAL parameter error: " << context.parameter_error_name()
                  << " - " << context.parameter_error_message() << "\n";
        return 2;
    }
    KeyGenerator keygen(context);
    SecretKey sk_obj = keygen.secret_key();
    const Plaintext &sk_pt = sk_obj.data();

    // Build two toy input images (NHWC flattened), then pad into polynomial layout.
    std::vector<uint64_t> input_flat(static_cast<size_t>(H) * W * C, 0);
    std::vector<uint64_t> input_flat2(static_cast<size_t>(H) * W * C, 0);
    // Option 1 (active): pseudo-random with fixed seed (reproducible).
    std::mt19937_64 rng(20260226ULL);
    std::uniform_int_distribution<uint64_t> dist1(0, 15);
    std::uniform_int_distribution<uint64_t> dist2(0, 15);

    // Option 2 (disabled): non-deterministic random each run.
    // std::random_device rd;
    // std::mt19937_64 rng(rd());
    // std::uniform_int_distribution<uint64_t> dist1(0, 15);
    // std::uniform_int_distribution<uint64_t> dist2(0, 15);

    for (size_t i = 0; i < input_flat.size(); ++i) {
        input_flat[i] = dist1(rng);
        input_flat2[i] = dist2(rng);
    }

    std::vector<uint64_t> padded = pad_same_to_poly_n(input_flat, H, W, C, kernel_k, static_cast<int>(N));
    std::vector<uint64_t> padded2 = pad_same_to_poly_n(input_flat2, H, W, C, kernel_k, static_cast<int>(N));
    std::vector<uint64_t> expected_plain = padded;   // reference #1 before scaling
    std::vector<uint64_t> expected_plain2 = padded2; // reference #2 before scaling

    // Encode: scale by Delta and serialize to coeff-form plaintext.
    scale_by_pow2_inplace(padded, delta_shift, static_cast<int>(log_q));
    scale_by_pow2_inplace(padded2, delta_shift, static_cast<int>(log_q));
    Plaintext m_pt = vector_to_plaintext_coeff(padded, N, static_cast<int>(log_q));
    Plaintext m_pt2 = vector_to_plaintext_coeff(padded2, N, static_cast<int>(log_q));

    // Encrypt: first encrypt zero, then inject message into c0 (RLWE form).
    Ciphertext ct;
    encrypt_zero_nttfree(context, ct, sk_pt, log_q);
    add_plain_to_ct_inplace_mod2k(ct, m_pt, log_q);

    Ciphertext ct2;
    encrypt_zero_nttfree(context, ct2, sk_pt, log_q);
    add_plain_to_ct_inplace_mod2k(ct2, m_pt2, log_q);

    // Decrypt + decode with same (k, Delta) parameters.
    const int Hpad = H + (kernel_k - 1);
    const int Wpad = W + (kernel_k - 1);
    const size_t used_coeff_count = static_cast<size_t>(Hpad) * Wpad * C;

    std::vector<int64_t> decoded =
        decrypt_and_decode_nttfree(context, ct, sk_pt, log_q, delta_shift, used_coeff_count);

    size_t mismatch_base = 0;
    for (size_t i = 0; i < used_coeff_count; ++i) {
        if (decoded[i] != static_cast<int64_t>(expected_plain[i])) {
            ++mismatch_base;
        }
    }

    // Test: ciphertext * constant (mod 2^k), then decrypt/decode again.
    constexpr uint64_t scalar = 3;
    Ciphertext ct_mul = ct;
    mul_const_ct_inplace_mod2k(ct_mul, scalar, log_q);

    std::vector<int64_t> decoded_mul =
        decrypt_and_decode_nttfree(context, ct_mul, sk_pt, log_q, delta_shift, used_coeff_count);

    size_t mismatch_mul = 0;
    for (size_t i = 0; i < used_coeff_count; ++i) {
        const uint64_t prod_mod2k =
            (static_cast<unsigned __int128>(expected_plain[i]) * scalar) & ((1ULL << log_q) - 1ULL);
        const int64_t expected_mul = centered_from_mod2k_main(prod_mod2k, log_q);
        if (decoded_mul[i] != expected_mul) {
            ++mismatch_mul;
        }
    }

    // Test: ciphertext + ciphertext (mod 2^k), then decrypt/decode again.
    Ciphertext ct_add = ct;
    add_ct_inplace_mod2k(ct_add, ct, log_q);

    std::vector<int64_t> decoded_add =
        decrypt_and_decode_nttfree(context, ct_add, sk_pt, log_q, delta_shift, used_coeff_count);

    size_t mismatch_add = 0;
    for (size_t i = 0; i < used_coeff_count; ++i) {
        const uint64_t sum_mod2k =
            (static_cast<unsigned __int128>(expected_plain[i]) + expected_plain[i]) & ((1ULL << log_q) - 1ULL);
        const int64_t expected_add = centered_from_mod2k_main(sum_mod2k, log_q);
        if (decoded_add[i] != expected_add) {
            ++mismatch_add;
        }
    }

    // Test: ciphertext #1 - ciphertext #2 (mod 2^k), then decrypt/decode again.
    Ciphertext ct_sub = ct;
    sub_ct_inplace_mod2k(ct_sub, ct2, log_q);

    std::vector<int64_t> decoded_sub =
        decrypt_and_decode_nttfree(context, ct_sub, sk_pt, log_q, delta_shift, used_coeff_count);

    size_t mismatch_sub = 0;
    for (size_t i = 0; i < used_coeff_count; ++i) {
        const uint64_t diff_mod2k =
            (static_cast<unsigned __int128>(expected_plain[i]) + ((1ULL << log_q) - expected_plain2[i])) &
            ((1ULL << log_q) - 1ULL);
        const int64_t expected_sub = centered_from_mod2k_main(diff_mod2k, log_q);
        if (decoded_sub[i] != expected_sub) {
            ++mismatch_sub;
        }
    }

    // Test: ciphertext + plaintext (mod 2^k), then decrypt/decode again.
    Ciphertext ct_addpt = ct;
    add_plain_to_ct_inplace_mod2k(ct_addpt, m_pt2, log_q);

    std::vector<int64_t> decoded_addpt =
        decrypt_and_decode_nttfree(context, ct_addpt, sk_pt, log_q, delta_shift, used_coeff_count);

    size_t mismatch_addpt = 0;
    for (size_t i = 0; i < used_coeff_count; ++i) {
        const uint64_t addpt_mod2k =
            (static_cast<unsigned __int128>(expected_plain[i]) + expected_plain2[i]) & ((1ULL << log_q) - 1ULL);
        const int64_t expected_addpt = centered_from_mod2k_main(addpt_mod2k, log_q);
        if (decoded_addpt[i] != expected_addpt) {
            ++mismatch_addpt;
        }
    }

    // Test: ciphertext - plaintext (mod 2^k), then decrypt/decode again.
    Ciphertext ct_subpt = ct;
    sub_plain_to_ct_inplace_mod2k(ct_subpt, m_pt2, log_q);

    std::vector<int64_t> decoded_subpt =
        decrypt_and_decode_nttfree(context, ct_subpt, sk_pt, log_q, delta_shift, used_coeff_count);

    size_t mismatch_subpt = 0;
    for (size_t i = 0; i < used_coeff_count; ++i) {
        const uint64_t subpt_mod2k =
            (static_cast<unsigned __int128>(expected_plain[i]) + ((1ULL << log_q) - expected_plain2[i])) &
            ((1ULL << log_q) - 1ULL);
        const int64_t expected_subpt = centered_from_mod2k_main(subpt_mod2k, log_q);
        if (decoded_subpt[i] != expected_subpt) {
            ++mismatch_subpt;
        }
    }

    // Test: plaintext * ciphertext (mod 2^k), then decrypt/decode again.
    // Use unscaled pt2 so decode stays in the same plaintext domain.
    Plaintext pt_mul = vector_to_plaintext_coeff(expected_plain2, N, static_cast<int>(log_q));
    Ciphertext ct_ptmul = ct;
    mul_plain_to_ct_inplace_mod2k(ct_ptmul, pt_mul, log_q);

    std::vector<int64_t> decoded_ptmul =
        decrypt_and_decode_nttfree(context, ct_ptmul, sk_pt, log_q, delta_shift, used_coeff_count);

    const std::vector<uint64_t> expected_ptmul_mod2k =
        negacyclic_convolution_mod2k_main(expected_plain, expected_plain2, log_q);
    size_t mismatch_ptmul = 0;
    for (size_t i = 0; i < used_coeff_count; ++i) {
        const int64_t expected_ptmul = centered_from_mod2k_main(expected_ptmul_mod2k[i], log_q);
        if (decoded_ptmul[i] != expected_ptmul) {
            ++mismatch_ptmul;
        }
    }

    size_t sample_begin = 33;
    size_t sample_end_excl = std::min<size_t>(43, used_coeff_count); // prints [33..42]

    std::cout << "[Test: Base Encrypt->Decrypt]\n";
    std::cout << "N=" << N
              << ", used_coeff_count=" << used_coeff_count
              << ", base_mismatches=" << mismatch_base << "\n";
    std::cout << "sample(base, idx " << sample_begin << "~" << (sample_end_excl - 1) << ")\n";
    for (size_t i = sample_begin; i < sample_end_excl; ++i) {
        std::cout << "  [" << i << "] expected=" << expected_plain[i]
                  << ", decoded=" << decoded[i] << "\n";
    }

    std::cout << "[Test: ct * const]\n";
    std::cout << "mul_const=" << scalar << ", mul_mismatches=" << mismatch_mul << "\n";
    std::cout << "sample(mul, idx " << sample_begin << "~" << (sample_end_excl - 1) << ")\n";
    for (size_t i = sample_begin; i < sample_end_excl; ++i) {
        const uint64_t prod_mod2k =
            (static_cast<unsigned __int128>(expected_plain[i]) * scalar) & ((1ULL << log_q) - 1ULL);
        const int64_t expected_mul = centered_from_mod2k_main(prod_mod2k, log_q);
        std::cout << "  [" << i << "] expected_mul=" << expected_mul
                  << ", decoded_mul=" << decoded_mul[i] << "\n";
    }

    std::cout << "[Test: ct + ct]\n";
    std::cout << "add_ct_mismatches=" << mismatch_add << "\n";
    std::cout << "sample(add, idx " << sample_begin << "~" << (sample_end_excl - 1) << ")\n";
    for (size_t i = sample_begin; i < sample_end_excl; ++i) {
        const uint64_t sum_mod2k =
            (static_cast<unsigned __int128>(expected_plain[i]) + expected_plain[i]) & ((1ULL << log_q) - 1ULL);
        const int64_t expected_add = centered_from_mod2k_main(sum_mod2k, log_q);
        std::cout << "  [" << i << "] expected_add=" << expected_add
                  << ", decoded_add=" << decoded_add[i] << "\n";
    }

    std::cout << "[Test: ct1 - ct2]\n";
    std::cout << "sub_ct_mismatches=" << mismatch_sub << "\n";
    std::cout << "sample(sub, idx " << sample_begin << "~" << (sample_end_excl - 1) << ")\n";
    for (size_t i = sample_begin; i < sample_end_excl; ++i) {
        const uint64_t diff_mod2k =
            (static_cast<unsigned __int128>(expected_plain[i]) + ((1ULL << log_q) - expected_plain2[i])) &
            ((1ULL << log_q) - 1ULL);
        const int64_t expected_sub = centered_from_mod2k_main(diff_mod2k, log_q);
        std::cout << "  [" << i << "] expected_sub=" << expected_sub
                  << ", decoded_sub=" << decoded_sub[i] << "\n";
    }

    std::cout << "[Test: ct + pt]\n";
    std::cout << "add_pt_mismatches=" << mismatch_addpt << "\n";
    std::cout << "sample(addpt, idx " << sample_begin << "~" << (sample_end_excl - 1) << ")\n";
    for (size_t i = sample_begin; i < sample_end_excl; ++i) {
        const uint64_t addpt_mod2k =
            (static_cast<unsigned __int128>(expected_plain[i]) + expected_plain2[i]) & ((1ULL << log_q) - 1ULL);
        const int64_t expected_addpt = centered_from_mod2k_main(addpt_mod2k, log_q);
        std::cout << "  [" << i << "] expected_addpt=" << expected_addpt
                  << ", decoded_addpt=" << decoded_addpt[i] << "\n";
    }

    std::cout << "[Test: ct - pt]\n";
    std::cout << "sub_pt_mismatches=" << mismatch_subpt << "\n";
    std::cout << "sample(subpt, idx " << sample_begin << "~" << (sample_end_excl - 1) << ")\n";
    for (size_t i = sample_begin; i < sample_end_excl; ++i) {
        const uint64_t subpt_mod2k =
            (static_cast<unsigned __int128>(expected_plain[i]) + ((1ULL << log_q) - expected_plain2[i])) &
            ((1ULL << log_q) - 1ULL);
        const int64_t expected_subpt = centered_from_mod2k_main(subpt_mod2k, log_q);
        std::cout << "  [" << i << "] expected_subpt=" << expected_subpt
                  << ", decoded_subpt=" << decoded_subpt[i] << "\n";
    }

    sample_begin = sample_begin*2;
    sample_end_excl = std::min<size_t>(sample_begin+10, used_coeff_count);

    std::cout << "[Test: pt * ct]\n";
    std::cout << "pt_mul_ct_mismatches=" << mismatch_ptmul << "\n";
    std::cout << "sample(pt*ct, idx " << sample_begin << "~" << (sample_end_excl - 1) << ")\n";
    for (size_t i = sample_begin; i < sample_end_excl; ++i) {
        const int64_t expected_ptmul = centered_from_mod2k_main(expected_ptmul_mod2k[i], log_q);
        std::cout << "  [" << i << "] expected_ptmul=" << expected_ptmul
                  << ", decoded_ptmul=" << decoded_ptmul[i] << "\n";
    }

    return (mismatch_base == 0 && mismatch_mul == 0 && mismatch_add == 0 && mismatch_sub == 0 &&
            mismatch_addpt == 0 && mismatch_subpt == 0 && mismatch_ptmul == 0)
               ? 0
               : 1;
}
