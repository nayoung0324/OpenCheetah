#pragma once

#include <seal/seal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/conv.h"

namespace mini_test::mod2k
{
inline void conv2d_multi_out_rot_cmult(
    const std::vector<seal::Ciphertext> &input_cts,
    const std::vector<int64_t> &kernels_flat_co_cin,
    size_t Co,
    size_t Cin,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t log_q,
    std::vector<seal::Ciphertext> &out_cts)
{
    conv2d_rot_cmult_multi_out_mod2k(input_cts, kernels_flat_co_cin, Co, Cin, H, W, KH, KW, log_q, out_cts);
}

inline void remove_unused_inplace(seal::Ciphertext &ct, const std::vector<size_t> &used_indices)
{
    remove_unused_coeffs_mod2k_inplace(ct, used_indices);
}
} // namespace mini_test::mod2k

