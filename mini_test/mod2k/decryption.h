#pragma once

#include <seal/seal.h>

#include <cstdint>
#include <vector>

void decrypt_nttfree(
    const seal::SEALContext &context,
    const seal::Ciphertext &ct,
    const seal::Plaintext &sk,
    seal::Plaintext &out,
    uint64_t log_q);

std::vector<int64_t> decode_divide_pow2(
    const seal::Plaintext &scaled_pt, int delta_shift, int k, size_t used_coeff_count = 0);

std::vector<int64_t> decrypt_and_decode_nttfree(
    const seal::SEALContext &context,
    const seal::Ciphertext &ct,
    const seal::Plaintext &sk,
    uint64_t log_q,
    int delta_shift,
    size_t used_coeff_count = 0);

// Experimental path:
// Compute c1*s in an auxiliary NTT prime q', then reduce to mod 2^k.
// Safe only when |(c1*s)_i| bound is strictly below q'/2.
void decrypt_auxprime_ntt_to_mod2k(
    const seal::SEALContext &context,
    const seal::Ciphertext &ct,
    const seal::Plaintext &sk,
    uint64_t log_q,
    int aux_mod_bit_count,
    seal::Plaintext &out);

std::vector<int64_t> decrypt_and_decode_auxprime_ntt(
    const seal::SEALContext &context,
    const seal::Ciphertext &ct,
    const seal::Plaintext &sk,
    uint64_t log_q,
    int delta_shift,
    int aux_mod_bit_count,
    size_t used_coeff_count = 0);
