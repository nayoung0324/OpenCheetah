#include "decryption.h"

#include <seal/util/ntt.h>
#include <seal/util/polyarithsmallmod.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "util.h"

using namespace seal;

// For ct = (c0, c1): compute m_tilde = c0 + c1*s mod 2^k.
void decrypt_nttfree(
    const SEALContext &context, const Ciphertext &ct, const Plaintext &sk, Plaintext &out, uint64_t log_q)
{
    auto &context_data = *context.first_context_data();
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();

    if (coeff_modulus.size() != 1) {
        throw std::invalid_argument("decrypt_nttfree expects single coeff modulus");
    }
    if (ct.size() != 2) {
        throw std::invalid_argument("decrypt_nttfree currently expects ciphertext size == 2");
    }
    if (log_q == 0 || log_q > 62) {
        throw std::invalid_argument("log_q must be in [1,62]");
    }

    const size_t N = parms.poly_modulus_degree();
    if (sk.coeff_count() != N) {
        throw std::invalid_argument("Secret key plaintext size mismatch");
    }

    const uint64_t q_seal = coeff_modulus[0].value();
    const uint64_t mask = (1ULL << log_q) - 1ULL;
    const uint64_t *c0 = ct.data(0);
    const uint64_t *c1 = ct.data(1);

    std::vector<uint64_t> c1s(N, 0);
    multiply_poly_secret_mod2k(c1, sk.data(), q_seal, c1s.data(), N, log_q);

    out.resize(N);
    for (size_t i = 0; i < N; ++i) {
        out[i] = ((c0[i] & mask) + c1s[i]) & mask;
    }
}

// Convert [0, 2^k) residue to centered signed representative.
static inline int64_t centered_from_mod2k_dec(uint64_t u, int k)
{
    const uint64_t q = (1ULL << k);
    const uint64_t half = q >> 1;
    u &= (q - 1ULL);
    if (u < half) return static_cast<int64_t>(u);
    return static_cast<int64_t>(u) - static_cast<int64_t>(q);
}

// Decode m_tilde ~= Delta*m + noise by dividing by Delta=2^delta_shift with symmetric rounding.
std::vector<int64_t> decode_divide_pow2(
    const Plaintext &scaled_pt, int delta_shift, int k, size_t used_coeff_count)
{
    if (k <= 0 || k > 62) throw std::invalid_argument("k must be in [1,62]");
    if (delta_shift < 0 || delta_shift >= k) {
        throw std::invalid_argument("delta_shift must satisfy 0 <= delta_shift < k");
    }

    const size_t N = scaled_pt.coeff_count();
    const size_t out_n = (used_coeff_count == 0 ? N : used_coeff_count);
    if (out_n > N) throw std::invalid_argument("used_coeff_count exceeds coeff_count");

    std::vector<int64_t> out(out_n, 0);
    const int64_t round = (delta_shift == 0 ? 0 : (1LL << (delta_shift - 1)));

    for (size_t i = 0; i < out_n; ++i) {
        int64_t x = centered_from_mod2k_dec(scaled_pt[i], k);

        if (delta_shift == 0) {
            out[i] = x;
            continue;
        }

        // Symmetric round-to-nearest: sign-aware handling avoids C++ negative shift quirks.
        if (x >= 0) {
            out[i] = (x + round) >> delta_shift;
        } else {
            const int64_t ax = -x;
            out[i] = -((ax + round) >> delta_shift);
        }
    }
    return out;
}

// Convenience wrapper: decrypt with mod-2^k path and decode by dividing Delta.
std::vector<int64_t> decrypt_and_decode_nttfree(
    const SEALContext &context,
    const Ciphertext &ct,
    const Plaintext &sk,
    uint64_t log_q,
    int delta_shift,
    size_t used_coeff_count)
{
    Plaintext scaled_pt;
    decrypt_nttfree(context, ct, sk, scaled_pt, log_q);
    return decode_divide_pow2(scaled_pt, delta_shift, static_cast<int>(log_q), used_coeff_count);
}

static inline int coeff_count_power_from_n(size_t n)
{
    if (n < 2 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("poly_modulus_degree must be power of two and >= 2");
    }
    int p = 0;
    while ((size_t(1) << p) < n) ++p;
    return p;
}

static inline int64_t center_lift_prime_signed(uint64_t x, uint64_t p)
{
    const uint64_t half = p >> 1;
    if (x <= half) return static_cast<int64_t>(x);
    return static_cast<int64_t>(x) - static_cast<int64_t>(p);
}

// Decrypt by computing c1*s in an auxiliary NTT prime q', then map the result back to mod 2^k.
void decrypt_auxprime_ntt_to_mod2k(
    const SEALContext &context,
    const Ciphertext &ct,
    const Plaintext &sk,
    uint64_t log_q,
    int aux_mod_bit_count,
    Plaintext &out)
{
    auto &context_data = *context.first_context_data();
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();

    if (coeff_modulus.size() != 1) {
        throw std::invalid_argument("decrypt_auxprime_ntt_to_mod2k expects single coeff modulus");
    }
    if (ct.size() != 2) {
        throw std::invalid_argument("decrypt_auxprime_ntt_to_mod2k currently expects ciphertext size == 2");
    }
    if (log_q == 0 || log_q > 62) {
        throw std::invalid_argument("log_q must be in [1,62]");
    }
    if (aux_mod_bit_count < 2 || aux_mod_bit_count > 60) {
        throw std::invalid_argument("aux_mod_bit_count must be in [2,60]");
    }

    const size_t n = parms.poly_modulus_degree();
    if (sk.coeff_count() != n) {
        throw std::invalid_argument("Secret key plaintext size mismatch");
    }

    // Build an auxiliary NTT prime q' compatible with this N.
    auto aux_vec = CoeffModulus::Create(n, { aux_mod_bit_count });
    if (aux_vec.empty()) {
        throw std::invalid_argument("failed to create auxiliary modulus");
    }
    const Modulus aux_mod = aux_vec[0];
    const uint64_t aux_p = aux_mod.value();

    // Safety check (single-prime exact lift): N*(2^k-1) must fit in (-q'/2, q'/2).
    const unsigned __int128 bnd = (static_cast<unsigned __int128>(n) * ((static_cast<unsigned __int128>(1) << log_q) - 1));
    if (bnd >= (static_cast<unsigned __int128>(aux_p) >> 1)) {
        throw std::invalid_argument(
            "auxiliary modulus too small for exact center-lift; use larger aux_mod_bit_count or multi-prime CRT");
    }

    seal::util::Pointer<seal::util::NTTTables> ntt_tables;
    seal::util::CreateNTTTables(
        coeff_count_power_from_n(n), std::vector<Modulus>{ aux_mod }, ntt_tables, MemoryManager::GetPool());
    const auto &tbl = ntt_tables[0];

    const uint64_t q_seal = coeff_modulus[0].value();
    const uint64_t q_mask = (1ULL << log_q) - 1ULL;
    const uint64_t *c0 = ct.data(0);
    const uint64_t *c1 = ct.data(1);

    std::vector<uint64_t> c1_aux(n, 0), s_aux(n, 0), prod_aux(n, 0);
    for (size_t i = 0; i < n; ++i) {
        c1_aux[i] = (c1[i] & q_mask) % aux_p;

        const uint64_t sj = sk[i];
        if (sj == 0) {
            s_aux[i] = 0;
        } else if (sj == 1) {
            s_aux[i] = 1;
        } else if (sj == (q_seal - 1)) {
            s_aux[i] = aux_p - 1; // -1 mod q'
        } else {
            // Keep same assumption as NTT-free path: non-zero/non-one is treated as -1.
            s_aux[i] = aux_p - 1;
        }
    }

    seal::util::ntt_negacyclic_harvey(c1_aux.data(), tbl);
    seal::util::ntt_negacyclic_harvey(s_aux.data(), tbl);
    seal::util::dyadic_product_coeffmod(c1_aux.data(), s_aux.data(), n, aux_mod, prod_aux.data());
    seal::util::inverse_ntt_negacyclic_harvey(prod_aux.data(), tbl);

    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        // center-lift product from mod q' then map back to mod 2^k.
        const int64_t centered = center_lift_prime_signed(prod_aux[i], aux_p);
        const uint64_t prod_mod2k = static_cast<uint64_t>(centered) & q_mask;
        out[i] = ((c0[i] & q_mask) + prod_mod2k) & q_mask;
    }
}

// Convenience wrapper: auxiliary-prime decryption followed by Delta decode.
std::vector<int64_t> decrypt_and_decode_auxprime_ntt(
    const SEALContext &context,
    const Ciphertext &ct,
    const Plaintext &sk,
    uint64_t log_q,
    int delta_shift,
    int aux_mod_bit_count,
    size_t used_coeff_count)
{
    Plaintext scaled_pt;
    decrypt_auxprime_ntt_to_mod2k(context, ct, sk, log_q, aux_mod_bit_count, scaled_pt);
    return decode_divide_pow2(scaled_pt, delta_shift, static_cast<int>(log_q), used_coeff_count);
}
