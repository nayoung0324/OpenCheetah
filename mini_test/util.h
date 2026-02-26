#pragma once

#include <cstddef>
#include <cstdint>

namespace seal
{
class Ciphertext;
class Plaintext;
}

void reduce_poly_mod2k(uint64_t *dst, const uint64_t *src, size_t n, uint64_t log_q);

void multiply_poly_secret_mod2k(
    const uint64_t *a,
    const uint64_t *s,
    uint64_t q_seal,
    uint64_t *out,
    size_t n,
    uint64_t log_q);

void rotate_poly_coeff_mod2k(
    const uint64_t *src, uint64_t *dst, size_t n, int64_t shift, uint64_t log_q);

void add_plain_to_ct_inplace_mod2k(seal::Ciphertext &ct, const seal::Plaintext &pt, uint64_t log_q);
void sub_plain_to_ct_inplace_mod2k(seal::Ciphertext &ct, const seal::Plaintext &pt, uint64_t log_q);

void add_ct_inplace_mod2k(seal::Ciphertext &ct_dst, const seal::Ciphertext &ct_src, uint64_t log_q);
void add_ct_inplace_mod2k_fast(seal::Ciphertext &ct_dst, const seal::Ciphertext &ct_src, uint64_t log_q);
void add_ct_inplace_mod2k_seal_iter(seal::Ciphertext &ct_dst, const seal::Ciphertext &ct_src, uint64_t log_q);
void sub_ct_inplace_mod2k(seal::Ciphertext &ct_dst, const seal::Ciphertext &ct_src, uint64_t log_q);

void mul_const_ct_inplace_mod2k(seal::Ciphertext &ct, uint64_t scalar, uint64_t log_q);

void mul_plain_to_ct_inplace_mod2k(seal::Ciphertext &ct, const seal::Plaintext &pt, uint64_t log_q);

void rotate_ct_coeff_inplace_mod2k(seal::Ciphertext &ct, int64_t shift, uint64_t log_q);
