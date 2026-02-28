#include "util.h"

#include <seal/seal.h>

#include <cstring>
#include <stdexcept>
#include <vector>

static inline void negacyclic_convolution_mod2k(
    const uint64_t *a, const uint64_t *b, uint64_t *out, size_t n, uint64_t log_q)
{
    const uint64_t mask = (1ULL << log_q) - 1ULL;
    for (size_t i = 0; i < n; ++i) out[i] = 0;

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
                out[pos] = (out[pos] - prod) & mask; // x^N = -1
        }
    }
}

// Reduce polynomial coefficients into [0, 2^k) by bit masking.
void reduce_poly_mod2k(uint64_t *dst, const uint64_t *src, size_t n, uint64_t log_q)
{
    if (log_q == 0 || log_q > 62) throw std::invalid_argument("log_q must be in [1,62]");
    const uint64_t mask = (1ULL << log_q) - 1ULL;
    for (size_t i = 0; i < n; ++i) dst[i] = src[i] & mask;
}

// Negacyclic multiply by secret-key-like ternary polynomial under mod 2^k.
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

// Negacyclic coefficient rotation by `shift` under x^N = -1 in mod 2^k.
void rotate_poly_coeff_mod2k(
    const uint64_t *src, uint64_t *dst, size_t n, int64_t shift, uint64_t log_q)
{
    if (!src || !dst) throw std::invalid_argument("src/dst must not be null");
    if (n == 0) throw std::invalid_argument("n must be > 0");
    if (log_q == 0 || log_q > 62) throw std::invalid_argument("log_q must be in [1,62]");

    const uint64_t mask = (1ULL << log_q) - 1ULL;

    // Handle aliasing (in-place call) safely.
    std::vector<uint64_t> tmp;
    const uint64_t *in = src;
    if (src == dst) {
        tmp.assign(src, src + n);
        in = tmp.data();
    }

    // shift = q*n + r, r in [0, n-1]. q parity controls global sign.
    int64_t q = shift / static_cast<int64_t>(n);
    int64_t r = shift % static_cast<int64_t>(n);
    if (r < 0) {
        r += static_cast<int64_t>(n);
        --q;
    }

    const size_t rs = static_cast<size_t>(r);
    const bool base_neg = static_cast<bool>(q & 1LL);

    // Fast-path for zero rotation: only optional global sign + masking.
    if (rs == 0) {
        if (!base_neg) {
            if (in != dst) {
                std::memcpy(dst, in, n * sizeof(uint64_t));
            }
            for (size_t i = 0; i < n; ++i) dst[i] &= mask;
        } else {
            for (size_t i = 0; i < n; ++i) {
                dst[i] = (0ULL - (in[i] & mask)) & mask;
            }
        }
        return;
    }

    // Destination split:
    // dst[0..rs-1]       <- in[n-rs .. n-1] (wrapped part; sign flips)
    // dst[rs..n-1]       <- in[0 .. n-rs-1] (non-wrapped part; base sign)
    const size_t head_len = rs;
    const size_t tail_len = n - rs;

    if (tail_len > 0) {
        std::memcpy(dst + rs, in, tail_len * sizeof(uint64_t));
    }
    if (head_len > 0) {
        std::memcpy(dst, in + tail_len, head_len * sizeof(uint64_t));
    }

    // Apply mask/sign only once per contiguous segment.
    if (!base_neg) {
        for (size_t i = rs; i < n; ++i) dst[i] &= mask; // non-wrapped
        for (size_t i = 0; i < rs; ++i) dst[i] = (0ULL - (dst[i] & mask)) & mask; // wrapped
    } else {
        for (size_t i = rs; i < n; ++i) dst[i] = (0ULL - (dst[i] & mask)) & mask; // non-wrapped
        for (size_t i = 0; i < rs; ++i) dst[i] &= mask; // wrapped
    }
}

// Add plaintext to ciphertext c0 component in-place under mod 2^k.
void add_inplace_mod2k(seal::Ciphertext &ct, const seal::Plaintext &pt, uint64_t log_q)
{
    if (ct.size() < 1) throw std::invalid_argument("ciphertext must have at least one component");
    if (pt.coeff_count() != ct.poly_modulus_degree()) {
        throw std::invalid_argument("plaintext/ciphertext coeff count mismatch");
    }

    const uint64_t mask = (1ULL << log_q) - 1ULL;
    uint64_t *c0 = ct.data(0);
    for (size_t i = 0; i < pt.coeff_count(); ++i) {
        c0[i] = (c0[i] + pt[i]) & mask;
    }
}

// Backward-compatible wrapper: ct += pt (c0 only).
void add_plain_to_ct_inplace_mod2k(seal::Ciphertext &ct, const seal::Plaintext &pt, uint64_t log_q)
{
    add_inplace_mod2k(ct, pt, log_q);
}

// Subtract plaintext from ciphertext c0 component in-place under mod 2^k.
void sub_inplace_mod2k(seal::Ciphertext &ct, const seal::Plaintext &pt, uint64_t log_q)
{
    if (ct.size() < 1) throw std::invalid_argument("ciphertext must have at least one component");
    if (pt.coeff_count() != ct.poly_modulus_degree()) {
        throw std::invalid_argument("plaintext/ciphertext coeff count mismatch");
    }
    const uint64_t mask = (1ULL << log_q) - 1ULL;
    uint64_t *c0 = ct.data(0);
    for (size_t i = 0; i < pt.coeff_count(); ++i) {
        c0[i] = (c0[i] - pt[i]) & mask;
    }
}

// Backward-compatible wrapper: ct -= pt (c0 only).
void sub_plain_to_ct_inplace_mod2k(seal::Ciphertext &ct, const seal::Plaintext &pt, uint64_t log_q)
{
    sub_inplace_mod2k(ct, pt, log_q);
}

// Add ciphertext to ciphertext component-wise in-place under mod 2^k.
void add_inplace_mod2k(seal::Ciphertext &ct_dst, const seal::Ciphertext &ct_src, uint64_t log_q)
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
            dst[i] = (dst[i] + src[i]) & mask;
        }
    }
}

// Backward-compatible wrapper: ct += ct.
void add_ct_inplace_mod2k(seal::Ciphertext &ct_dst, const seal::Ciphertext &ct_src, uint64_t log_q)
{
    add_inplace_mod2k(ct_dst, ct_src, log_q);
}

// Slightly unrolled variant of ciphertext addition for micro-benchmarking.
void add_ct_inplace_mod2k_fast(seal::Ciphertext &ct_dst, const seal::Ciphertext &ct_src, uint64_t log_q)
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

        // Fallback when aliased.
        if (dst == src) {
            for (size_t i = 0; i < n; ++i) dst[i] = (dst[i] + src[i]) & mask;
            continue;
        }

        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            dst[i + 0] = (dst[i + 0] + src[i + 0]) & mask;
            dst[i + 1] = (dst[i + 1] + src[i + 1]) & mask;
            dst[i + 2] = (dst[i + 2] + src[i + 2]) & mask;
            dst[i + 3] = (dst[i + 3] + src[i + 3]) & mask;
        }
        for (; i < n; ++i) {
            dst[i] = (dst[i] + src[i]) & mask;
        }
    }
}

// Subtract ciphertext from ciphertext component-wise in-place under mod 2^k.
void sub_inplace_mod2k(seal::Ciphertext &ct_dst, const seal::Ciphertext &ct_src, uint64_t log_q)
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
            dst[i] = (dst[i] - src[i]) & mask;
        }
    }
}

// Backward-compatible wrapper: ct -= ct.
void sub_ct_inplace_mod2k(seal::Ciphertext &ct_dst, const seal::Ciphertext &ct_src, uint64_t log_q)
{
    sub_inplace_mod2k(ct_dst, ct_src, log_q);
}

// Multiply all ciphertext components by a scalar in-place under mod 2^k.
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
            const unsigned __int128 prod = static_cast<unsigned __int128>(ci[i]) * a;
            ci[i] = static_cast<uint64_t>(prod) & mask;
        }
    }
}

// Multiply ciphertext by plaintext (negacyclic convolution per component) under mod 2^k.
void mul_plain_to_ct_inplace_mod2k(seal::Ciphertext &ct, const seal::Plaintext &pt, uint64_t log_q)
{
    if (ct.size() < 2) throw std::invalid_argument("ciphertext must have at least two components");
    if (pt.coeff_count() != ct.poly_modulus_degree()) {
        throw std::invalid_argument("plaintext/ciphertext coeff count mismatch");
    }
    if (log_q == 0 || log_q > 62) throw std::invalid_argument("log_q must be in [1,62]");

    const size_t n = ct.poly_modulus_degree();
    std::vector<uint64_t> src(n, 0), dst(n, 0);

    for (size_t comp = 0; comp < ct.size(); ++comp) {
        uint64_t *ci = ct.data(comp);
        for (size_t i = 0; i < n; ++i) src[i] = ci[i];
        negacyclic_convolution_mod2k(src.data(), pt.data(), dst.data(), n, log_q);
        for (size_t i = 0; i < n; ++i) ci[i] = dst[i];
    }
}

// Rotate each ciphertext component by `shift` under mod 2^k.
void rotate_ct_coeff_inplace_mod2k(seal::Ciphertext &ct, int64_t shift, uint64_t log_q)
{
    if (ct.size() < 1) throw std::invalid_argument("ciphertext must have at least one component");
    if (log_q == 0 || log_q > 62) throw std::invalid_argument("log_q must be in [1,62]");

    const size_t n = ct.poly_modulus_degree();
    std::vector<uint64_t> src(n, 0), dst(n, 0);

    for (size_t comp = 0; comp < ct.size(); ++comp) {
        uint64_t *ci = ct.data(comp);
        for (size_t i = 0; i < n; ++i) src[i] = ci[i];
        rotate_poly_coeff_mod2k(src.data(), dst.data(), n, shift, log_q);
        for (size_t i = 0; i < n; ++i) ci[i] = dst[i];
    }
}
