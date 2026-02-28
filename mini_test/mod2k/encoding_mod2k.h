#pragma once

#include <seal/seal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/util.h"
#include "mod2k/encryption.h"

namespace mini_test::mod2k
{
inline void encode_message_inplace(std::vector<uint64_t> &coeffs, int delta_shift, int log_q)
{
    scale_by_pow2_inplace(coeffs, delta_shift, log_q);
}

inline seal::Plaintext to_plaintext(const std::vector<uint64_t> &coeffs_mod2k, size_t poly_n, int log_q)
{
    return vector_to_plaintext_coeff(coeffs_mod2k, poly_n, log_q);
}
} // namespace mini_test::mod2k
