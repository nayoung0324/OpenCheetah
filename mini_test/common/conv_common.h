#pragma once

#include <seal/seal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

struct Conv2DTile
{
    size_t out_row;
    size_t out_col;
    size_t out_h;
    size_t out_w;
    size_t in_h;
    size_t in_w;
};

// Coefficient-encoding helper: signed integer -> [0, plain_modulus).
uint64_t encode_signed_to_plain_coeff(int64_t v, uint64_t plain_modulus);

// Coefficient-encoding helper: [0, plain_modulus) -> centered signed integer.
int64_t decode_plain_coeff_to_signed(uint64_t v, uint64_t plain_modulus);

// Build a plaintext polynomial of size N from flattened image coefficients.
seal::Plaintext encode_image_coeff_plain(
    const std::vector<int64_t> &image_flat, size_t H, size_t W, size_t N, uint64_t plain_modulus);

// Compact valid coefficients to the prefix [0..valid_count-1] in the given order.
// Arbitrary ciphertext compaction is unsupported; compact on decoded vectors.
void compact_valid_coeffs_inplace(seal::Ciphertext &ct, const std::vector<size_t> &valid_indices);
void compact_valid_coeffs_inplace(
    seal::Ciphertext &ct, const seal::Evaluator &evaluator, const std::vector<size_t> &valid_indices);

// Build tiling plan for valid conv (stride=1, no padding) when a single input patch
// must satisfy in_h * in_w <= max_input_coeffs.
std::vector<Conv2DTile> make_conv2d_tiles_valid(
    size_t H, size_t W, size_t KH, size_t KW, size_t max_input_coeffs);

// Extract one input patch by tile (row-major).
std::vector<int64_t> extract_input_patch_by_tile(
    const std::vector<int64_t> &image_flat, size_t H, size_t W, const Conv2DTile &tile);

// Scatter one output patch to a full output tensor (row-major).
void scatter_output_patch_by_tile(
    const std::vector<int64_t> &patch_out,
    const Conv2DTile &tile,
    size_t out_W,
    std::vector<int64_t> &full_out);

// Compact selected coefficients to prefix order:
// out[0] = coeffs[valid_indices[0]], ..., out[m-1] = coeffs[valid_indices[m-1]], rest zero.
std::vector<int64_t> compact_coeffs_to_prefix(
    const std::vector<int64_t> &coeffs,
    const std::vector<size_t> &valid_indices,
    size_t out_size);

