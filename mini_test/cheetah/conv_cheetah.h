#pragma once

#include <seal/seal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/conv_common.h"

// Build convolution kernel polynomial for PMult.
// kernel_flat is row-major KH*KW. Shift for (r,c) is r*W + c.
seal::Plaintext build_conv_kernel_plain(
    const std::vector<int64_t> &kernel_flat, size_t KH, size_t KW, size_t W, size_t N, uint64_t plain_modulus);

// Cheetah-style base index for filter placement (single-channel): begin = W*(KH-1) + (KW-1).
size_t cheetah_filter_base_index(size_t KH, size_t KW, size_t W);

// Compute integer reference matching Cheetah filter indexing + PMult.
std::vector<int64_t> conv2d_reference_poly_cheetah_valid(
    const std::vector<int64_t> &image_flat,
    size_t H,
    size_t W,
    const std::vector<int64_t> &kernel_flat,
    size_t KH,
    size_t KW,
    size_t N);

// Collect valid output coefficient indices for Cheetah-style conv layout.
std::vector<size_t> cheetah_valid_output_indices(size_t H, size_t W, size_t KH, size_t KW, size_t N);

// Build 0/1 plaintext mask at valid indices (for extract via multiply_plain).
seal::Plaintext build_extract_mask_plain(
    const std::vector<size_t> &valid_indices, size_t N, uint64_t plain_modulus);

// Cheetah-style extract/remove-unused on ciphertext c0.
void extract_valid_coeffs_inplace(
    seal::Ciphertext &ct, const seal::Evaluator &evaluator, const std::vector<size_t> &valid_indices);
void remove_unused_coeffs_cheetah_inplace(
    seal::Ciphertext &ct, const seal::Evaluator &evaluator, const std::vector<size_t> &used_indices);

// Multi-input-channel conv (channel-accumulate) with Cheetah-style PMult.
void conv2d_pmult_accum_multi_in(
    const std::vector<seal::Ciphertext> &input_cts,
    const std::vector<int64_t> &kernel_flat_cin,
    size_t Cin,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t plain_modulus,
    const seal::Evaluator &evaluator,
    seal::Ciphertext &out_ct);

// Pack up to channels_per_ct input channels into one plaintext (coefficient encoding).
seal::Plaintext encode_image_coeff_plain_packed(
    const std::vector<std::vector<int64_t>> &images,
    size_t ch_begin,
    size_t channels_per_ct,
    size_t H,
    size_t W,
    size_t N,
    uint64_t plain_modulus);

// Build packed multi-channel kernel plaintext for PMult.
seal::Plaintext build_conv_kernel_plain_packed(
    const std::vector<int64_t> &kernel_flat_cin,
    size_t Cin,
    size_t ch_begin,
    size_t channels_per_ct,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    size_t N,
    uint64_t plain_modulus);

// Cheetah-style PMult conv with packed input channels.
void conv2d_pmult_accum_multi_in_packed(
    const std::vector<seal::Ciphertext> &input_cts_packed,
    const std::vector<int64_t> &kernel_flat_cin,
    size_t Cin,
    size_t channels_per_ct,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t plain_modulus,
    const seal::Evaluator &evaluator,
    seal::Ciphertext &out_ct);

// Multi-output-channel PMult wrapper.
void conv2d_pmult_multi_out_packed(
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
    std::vector<seal::Ciphertext> &out_cts);

