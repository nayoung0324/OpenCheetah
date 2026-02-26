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
    constexpr int64_t rot_shift = 7;

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, { 50 }));
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

    // Fill all N coefficients (no padding).
    std::mt19937_64 rng(20260226ULL);
    std::uniform_int_distribution<uint64_t> dist(0, 15);
    std::vector<uint64_t> m1(N, 0), m2(N, 0);
    for (size_t i = 0; i < N; ++i) {
        m1[i] = dist(rng);
        m2[i] = dist(rng);
    }

    std::vector<uint64_t> m1_scaled = m1;
    std::vector<uint64_t> m2_scaled = m2;
    scale_by_pow2_inplace(m1_scaled, delta_shift, static_cast<int>(log_q));
    scale_by_pow2_inplace(m2_scaled, delta_shift, static_cast<int>(log_q));

    Plaintext pt1 = vector_to_plaintext_coeff(m1_scaled, N, static_cast<int>(log_q));
    Plaintext pt2 = vector_to_plaintext_coeff(m2_scaled, N, static_cast<int>(log_q));

    // Setup (not timed): encryption/decryption timing is intentionally excluded.
    Ciphertext ct1, ct2;
    encrypt_zero_nttfree(context, ct1, sk_pt, log_q);
    add_plain_to_ct_inplace_mod2k(ct1, pt1, log_q);
    encrypt_zero_nttfree(context, ct2, sk_pt, log_q);
    add_plain_to_ct_inplace_mod2k(ct2, pt2, log_q);

    constexpr int iters_light = 1000;
    constexpr int iters_heavy = 10; // pt*ct is O(N^2)

    Ciphertext w_add_ct = ct1;
    const double add_ct_us = time_op_us([&]() { add_ct_inplace_mod2k(w_add_ct, ct2, log_q); }, iters_light);

    Ciphertext w_sub_ct = ct1;
    const double sub_ct_us = time_op_us([&]() { sub_ct_inplace_mod2k(w_sub_ct, ct2, log_q); }, iters_light);

    Ciphertext w_add_pt = ct1;
    const double add_pt_us = time_op_us([&]() { add_plain_to_ct_inplace_mod2k(w_add_pt, pt2, log_q); }, iters_light);

    Ciphertext w_sub_pt = ct1;
    const double sub_pt_us = time_op_us([&]() { sub_plain_to_ct_inplace_mod2k(w_sub_pt, pt2, log_q); }, iters_light);

    Ciphertext w_mul_const = ct1;
    const double mul_const_us =
        time_op_us([&]() { mul_const_ct_inplace_mod2k(w_mul_const, scalar, log_q); }, iters_light);

    Ciphertext w_mul_pt = ct1;
    const double mul_pt_us =
        time_op_us([&]() { mul_plain_to_ct_inplace_mod2k(w_mul_pt, pt2, log_q); }, iters_heavy);

    Ciphertext w_rot = ct1;
    const double rot_us = time_op_us([&]() { rotate_ct_coeff_inplace_mod2k(w_rot, rot_shift, log_q); }, iters_light);

    // Tiny checksum to keep compiler honest.
    const uint64_t checksum = w_add_ct.data(0)[0] ^ w_sub_ct.data(0)[1] ^ w_mul_pt.data(0)[2] ^ w_rot.data(0)[3];

    std::cout << "[Operation Timing Only]\n";
    std::cout << "N=" << N << ", log_q=" << log_q << ", delta_shift=" << delta_shift << "\n";
    std::cout << "iters_light=" << iters_light << ", iters_heavy=" << iters_heavy << "\n";
    std::cout << "add_ct   total_us=" << add_ct_us << ", avg_us=" << (add_ct_us / iters_light) << "\n";
    std::cout << "sub_ct   total_us=" << sub_ct_us << ", avg_us=" << (sub_ct_us / iters_light) << "\n";
    std::cout << "add_pt   total_us=" << add_pt_us << ", avg_us=" << (add_pt_us / iters_light) << "\n";
    std::cout << "sub_pt   total_us=" << sub_pt_us << ", avg_us=" << (sub_pt_us / iters_light) << "\n";
    std::cout << "mul_const total_us=" << mul_const_us << ", avg_us=" << (mul_const_us / iters_light) << "\n";
    std::cout << "mul_pt   total_us=" << mul_pt_us << ", avg_us=" << (mul_pt_us / iters_heavy) << "\n";
    std::cout << "rotate   total_us=" << rot_us << ", avg_us=" << (rot_us / iters_light) << "\n";
    std::cout << "checksum=" << checksum << "\n";

    return 0;
}
