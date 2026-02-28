#pragma once

#include <seal/seal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/conv.h"

namespace mini_test::cheetah
{
inline void conv2d_multi_out_pmult(
    const std::vector<seal::Ciphertext> &input_cts_packed,
    const std::vector<int64_t> &kernels_flat_co_cin,
    size_t Co,
    size_t Cin,
    size_t channels_per_ct,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t plain_modulus,
    const seal::Evaluator &evaluator,
    std::vector<seal::Ciphertext> &out_cts)
{
    conv2d_pmult_multi_out_packed(
        input_cts_packed,
        kernels_flat_co_cin,
        Co,
        Cin,
        channels_per_ct,
        H,
        W,
        KH,
        KW,
        plain_modulus,
        evaluator,
        out_cts);
}

inline void remove_unused_inplace(
    seal::Ciphertext &ct, const seal::Evaluator &evaluator, const std::vector<size_t> &used_indices)
{
    remove_unused_coeffs_cheetah_inplace(ct, evaluator, used_indices);
}
} // namespace mini_test::cheetah

