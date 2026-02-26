// mini_test/main.cpp
#include <seal/seal.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "encryption.h"
#include "util.h"

using namespace seal;
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
    constexpr uint64_t log_q = 50;
    constexpr int delta_shift = 20;
    constexpr uint64_t scalar = 3;
    constexpr int iters = 5000;

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, { 50 })); // single coeff modulus
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
    PublicKey pk;
    keygen.create_public_key(pk);

    // Full-N data (no padding), reproducible random.
    std::mt19937_64 rng(20260226ULL);
    std::uniform_int_distribution<uint64_t> dist(0, 15);
    std::vector<uint64_t> m1(N, 0), m2(N, 0);
    for (size_t i = 0; i < N; ++i) {
        m1[i] = dist(rng);
        m2[i] = dist(rng);
    }

    // ----- custom mod2k setup (not timed) -----
    std::vector<uint64_t> m1_scaled = m1, m2_scaled = m2;
    scale_by_pow2_inplace(m1_scaled, delta_shift, static_cast<int>(log_q));
    scale_by_pow2_inplace(m2_scaled, delta_shift, static_cast<int>(log_q));
    Plaintext pt1_mod2k = vector_to_plaintext_coeff(m1_scaled, N, static_cast<int>(log_q));
    Plaintext pt2_mod2k = vector_to_plaintext_coeff(m2_scaled, N, static_cast<int>(log_q));

    Ciphertext ct1_mod2k, ct2_mod2k;
    encrypt_zero_nttfree(context, ct1_mod2k, sk_pt, log_q);
    add_plain_to_ct_inplace_mod2k(ct1_mod2k, pt1_mod2k, log_q);
    encrypt_zero_nttfree(context, ct2_mod2k, sk_pt, log_q);
    add_plain_to_ct_inplace_mod2k(ct2_mod2k, pt2_mod2k, log_q);

    // ----- SEAL setup (not timed) -----
    BatchEncoder batch_encoder(context);
    Encryptor encryptor(context, pk);
    Evaluator evaluator(context);

    std::vector<uint64_t> slots1(batch_encoder.slot_count(), 0), slots2(batch_encoder.slot_count(), 0);
    for (size_t i = 0; i < N; ++i) {
        slots1[i] = m1[i];
        slots2[i] = m2[i];
    }
    Plaintext pt1_seal, pt2_seal;
    batch_encoder.encode(slots1, pt1_seal);
    batch_encoder.encode(slots2, pt2_seal);

    Ciphertext ct1_seal, ct2_seal;
    encryptor.encrypt(pt1_seal, ct1_seal);
    encryptor.encrypt(pt2_seal, ct2_seal);

    // constant plaintext for multiply_plain_inplace
    Plaintext scalar_pt;
    scalar_pt.resize(N);
    scalar_pt[0] = scalar;
    for (size_t i = 1; i < N; ++i) scalar_pt[i] = 0;

    // ----- timing: ct+ct -----
    Ciphertext w_add_mod2k_orig = ct1_mod2k;
    const double add_mod2k_orig_us =
        time_op_us([&]() { add_ct_inplace_mod2k(w_add_mod2k_orig, ct2_mod2k, log_q); }, iters);

    Ciphertext w_add_mod2k_fast = ct1_mod2k;
    const double add_mod2k_fast_us =
        time_op_us([&]() { add_ct_inplace_mod2k_fast(w_add_mod2k_fast, ct2_mod2k, log_q); }, iters);

    Ciphertext w_add_seal = ct1_seal;
    const double add_seal_us = time_op_us([&]() { evaluator.add_inplace(w_add_seal, ct2_seal); }, iters);

    // ----- timing: ct*const -----
    Ciphertext w_mulc_mod2k = ct1_mod2k;
    const double mulc_mod2k_us =
        time_op_us([&]() { mul_const_ct_inplace_mod2k(w_mulc_mod2k, scalar, log_q); }, iters);

    Ciphertext w_mulc_seal = ct1_seal;
    const double mulc_seal_us =
        time_op_us([&]() { evaluator.multiply_plain_inplace(w_mulc_seal, scalar_pt); }, iters);

    const uint64_t checksum = w_add_mod2k_orig.data(0)[0] ^ w_add_mod2k_fast.data(0)[1] ^ w_mulc_mod2k.data(0)[2] ^
                              w_add_seal.data(0)[3] ^ w_mulc_seal.data(0)[4];

    std::cout << "[Speed Comparison: custom _mod2k vs SEAL(NTT)]\n";
    std::cout << "N=" << N << ", log_q=" << log_q << ", iters=" << iters << "\n";
    std::cout << "op=ct+ct\n";
    std::cout << "  custom_mod2k_orig total_us=" << add_mod2k_orig_us << ", avg_us=" << (add_mod2k_orig_us / iters)
              << "\n";
    std::cout << "  custom_mod2k_fast total_us=" << add_mod2k_fast_us << ", avg_us=" << (add_mod2k_fast_us / iters)
              << "\n";
    std::cout << "  seal_ntt     total_us=" << add_seal_us << ", avg_us=" << (add_seal_us / iters) << "\n";
    std::cout << "op=ct*const (const=" << scalar << ")\n";
    std::cout << "  custom_mod2k total_us=" << mulc_mod2k_us << ", avg_us=" << (mulc_mod2k_us / iters) << "\n";
    std::cout << "  seal_ntt     total_us=" << mulc_seal_us << ", avg_us=" << (mulc_seal_us / iters) << "\n";
    std::cout << "checksum=" << checksum << "\n";

    return 0;
}
