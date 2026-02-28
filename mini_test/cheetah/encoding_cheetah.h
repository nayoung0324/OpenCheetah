#pragma once

#include <seal/seal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cheetah/conv_cheetah.h"

namespace mini_test::cheetah
{
inline seal::Plaintext encode_image_packed(
    const std::vector<std::vector<int64_t>> &images,
    size_t ch_begin,
    size_t channels_per_ct,
    size_t H,
    size_t W,
    size_t N,
    uint64_t plain_modulus)
{
    return encode_image_coeff_plain_packed(images, ch_begin, channels_per_ct, H, W, N, plain_modulus);
}

inline seal::Plaintext encode_kernel_packed(
    const std::vector<int64_t> &kernel_flat_cin,
    size_t Cin,
    size_t ch_begin,
    size_t channels_per_ct,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    size_t N,
    uint64_t plain_modulus)
{
    return build_conv_kernel_plain_packed(
        kernel_flat_cin, Cin, ch_begin, channels_per_ct, H, W, KH, KW, N, plain_modulus);
}
} // namespace mini_test::cheetah
