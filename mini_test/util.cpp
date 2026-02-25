#include "util.h"

#include <seal/seal.h>

#include <stdexcept>

void reduce_poly_mod2k(uint64_t *dst, const uint64_t *src, size_t n, uint64_t log_q)
{
    if (log_q == 0 || log_q > 62) throw std::invalid_argument("log_q must be in [1,62]");
    const uint64_t mask = (1ULL << log_q) - 1ULL;
    for (size_t i = 0; i < n; ++i) dst[i] = src[i] & mask;
}

void multiply_poly_secret_mod2k(
    const uint64_t *a, const uint64_t *s, uint64_t q_seal, uint64_t *out, size_t n, uint64_t log_q)
{
    (void)q_seal; // secret coeff is assumed ternary-like (0, 1, q-1)
    const uint64_t mask = (1ULL << log_q) - 1ULL;

    for (size_t i = 0; i < n; ++i) out[i] = 0;

    for (size_t i = 0; i < n; ++i) {
        const uint64_t ai = a[i] & mask;
        if (!ai) continue;

        for (size_t j = 0; j < n; ++j) {
            const uint64_t sj = s[j];
            if (sj == 0) continue;

            size_t pos = i + j;
            bool wrapped = false;
            if (pos >= n) {
                pos -= n;
                wrapped = true;
            }

            bool neg = (sj != 1);
            if (wrapped) neg = !neg;

            if (!neg)
                out[pos] = (out[pos] + ai) & mask;
            else
                out[pos] = (out[pos] - ai) & mask;
        }
    }
}

void add_plain_to_ct_inplace_mod2k(seal::Ciphertext &ct, const seal::Plaintext &pt, uint64_t log_q)
{
    if (ct.size() < 1) throw std::invalid_argument("ciphertext must have at least one component");
    if (pt.coeff_count() != ct.poly_modulus_degree()) {
        throw std::invalid_argument("plaintext/ciphertext coeff count mismatch");
    }

    const uint64_t mask = (1ULL << log_q) - 1ULL;
    uint64_t *c0 = ct.data(0);
    for (size_t i = 0; i < pt.coeff_count(); ++i) {
        c0[i] = ((c0[i] & mask) + (pt[i] & mask)) & mask;
    }
}

void add_ct_inplace_mod2k(seal::Ciphertext &ct_dst, const seal::Ciphertext &ct_src, uint64_t log_q)
{
    if (ct_dst.size() != ct_src.size()) throw std::invalid_argument("ciphertext size mismatch");
    if (ct_dst.poly_modulus_degree() != ct_src.poly_modulus_degree()) {
        throw std::invalid_argument("ciphertext poly_modulus_degree mismatch");
    }
    if (log_q == 0 || log_q > 62) throw std::invalid_argument("log_q must be in [1,62]");

    const uint64_t mask = (1ULL << log_q) - 1ULL;
    const size_t n = ct_dst.poly_modulus_degree();

    for (size_t comp = 0; comp < ct_dst.size(); ++comp) {
        uint64_t *dst = ct_dst.data(comp);
        const uint64_t *src = ct_src.data(comp);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = ((dst[i] & mask) + (src[i] & mask)) & mask;
        }
    }
}

void mul_const_ct_inplace_mod2k(seal::Ciphertext &ct, uint64_t scalar, uint64_t log_q)
{
    if (ct.size() < 2) throw std::invalid_argument("ciphertext must have at least two components");
    if (log_q == 0 || log_q > 62) throw std::invalid_argument("log_q must be in [1,62]");

    const uint64_t mask = (1ULL << log_q) - 1ULL;
    const uint64_t a = scalar & mask;
    const size_t n = ct.poly_modulus_degree();

    for (size_t comp = 0; comp < ct.size(); ++comp) {
        uint64_t *ci = ct.data(comp);
        for (size_t i = 0; i < n; ++i) {
            const unsigned __int128 prod = static_cast<unsigned __int128>(ci[i] & mask) * a;
            ci[i] = static_cast<uint64_t>(prod) & mask;
        }
    }
}
