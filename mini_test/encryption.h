#pragma once

#include <seal/seal.h>

#include <cstdint>
#include <vector>

std::vector<uint64_t> pad_same_to_poly_n(
    const std::vector<uint64_t> &input_flat, int h, int w, int c, int kernel_k, int poly_n);

seal::Plaintext vector_to_plaintext_coeff(const std::vector<uint64_t> &coeffs_mod2k, size_t poly_n, int k);

void scale_by_pow2_inplace(std::vector<uint64_t> &coeffs, int delta_shift, int k);

void encrypt_zero_nttfree(
    const seal::SEALContext &context, seal::Ciphertext &temp_out, const seal::Plaintext &sk, uint64_t log_q);
