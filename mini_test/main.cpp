// mini_test/main.cpp
#include <seal/seal.h>

#include <cstdint>
#include <iostream>
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

    // Build toy input image (NHWC flattened), then pad into polynomial layout.
    std::vector<uint64_t> input_flat(static_cast<size_t>(H) * W * C, 0);
    for (size_t i = 0; i < input_flat.size(); ++i) {
        input_flat[i] = static_cast<uint64_t>((i % 13) + 1); // small positive payload
    }

    std::vector<uint64_t> padded = pad_same_to_poly_n(input_flat, H, W, C, kernel_k, static_cast<int>(N));
    std::vector<uint64_t> expected_plain = padded; // reference before scaling

    // Encode: scale by Delta and serialize to coeff-form plaintext.
    scale_by_pow2_inplace(padded, delta_shift, static_cast<int>(log_q));
    Plaintext m_pt = vector_to_plaintext_coeff(padded, N, static_cast<int>(log_q));

    // Encrypt: first encrypt zero, then inject message into c0 (RLWE form).
    Ciphertext ct;
    encrypt_zero_nttfree(context, ct, sk_pt, log_q);
    add_plain_to_ct_inplace_mod2k(ct, m_pt, log_q);

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

    // Test: ciphertext - ciphertext (mod 2^k), then decrypt/decode again.
    Ciphertext ct_sub = ct;
    sub_ct_inplace_mod2k(ct_sub, ct, log_q);

    std::vector<int64_t> decoded_sub =
        decrypt_and_decode_nttfree(context, ct_sub, sk_pt, log_q, delta_shift, used_coeff_count);

    size_t mismatch_sub = 0;
    for (size_t i = 0; i < used_coeff_count; ++i) {
        const int64_t expected_sub = 0;
        if (decoded_sub[i] != expected_sub) {
            ++mismatch_sub;
        }
    }

    const size_t sample_begin = 33;
    const size_t sample_end_excl = std::min<size_t>(43, used_coeff_count); // prints [33..42]

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

    std::cout << "[Test: ct - ct]\n";
    std::cout << "sub_ct_mismatches=" << mismatch_sub << "\n";
    std::cout << "sample(sub, idx " << sample_begin << "~" << (sample_end_excl - 1) << ")\n";
    for (size_t i = sample_begin; i < sample_end_excl; ++i) {
        std::cout << "  [" << i << "] expected_sub=0"
                  << ", decoded_sub=" << decoded_sub[i] << "\n";
    }

    return (mismatch_base == 0 && mismatch_mul == 0 && mismatch_add == 0 && mismatch_sub == 0) ? 0 : 1;
}
