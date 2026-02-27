#pragma once

#include <seal/seal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

// Coefficient-encoding helper: signed integer -> [0, plain_modulus).
uint64_t encode_signed_to_plain_coeff(int64_t v, uint64_t plain_modulus);

// Coefficient-encoding helper: [0, plain_modulus) -> centered signed integer.
int64_t decode_plain_coeff_to_signed(uint64_t v, uint64_t plain_modulus);

// Build a plaintext polynomial of size N from flattened image coefficients.
// Input layout is row-major with size H*W.
seal::Plaintext encode_image_coeff_plain(
    const std::vector<int64_t> &image_flat, size_t H, size_t W, size_t N, uint64_t plain_modulus);

// Build convolution kernel polynomial for PMult.
// kernel_flat is row-major KH*KW. Shift for (r,c) is r*W + c.
seal::Plaintext build_conv_kernel_plain(
    const std::vector<int64_t> &kernel_flat, size_t KH, size_t KW, size_t W, size_t N, uint64_t plain_modulus);

// Cheetah-style base index for filter placement (single-channel):
// begin = W*(KH-1) + (KW-1)
size_t cheetah_filter_base_index(size_t KH, size_t KW, size_t W);

// Compute integer reference matching Cheetah filter indexing + PMult:
// output at coeff begin + r*W + c for valid (r,c).
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

// Cheetah-style extract:
// Keep only valid coefficients in ciphertext; set all others to zero.
// If ciphertext is in NTT form, transform back before zeroing coefficients.
void extract_valid_coeffs_inplace(
    seal::Ciphertext &ct, const seal::Evaluator &evaluator, const std::vector<size_t> &valid_indices);

// _mod2k convolution (single-channel) without PMult:
// For each kernel coefficient w[r,c], rotate input by -(r*W+c), multiply by scalar w[r,c], and accumulate.
// Output valid region is placed at indices r*W + c for 0<=r<=H-KH, 0<=c<=W-KW.
void conv2d_rot_cmult_accum_mod2k(
    const seal::Ciphertext &input_ct,
    const std::vector<int64_t> &kernel_flat,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t log_q,
    seal::Ciphertext &out_ct);

// Valid output coefficient indices for conv2d_rot_cmult_accum_mod2k layout.
std::vector<size_t> valid_output_indices_rot_cmult_mod2k(
    size_t H, size_t W, size_t KH, size_t KW, size_t N);

// Fused primitive for _mod2k path:
// acc_ct += Rot(input_ct, shift) * scalar   (all operations mod 2^k).
void rotate_multiply_scalar_add_ct_mod2k(
    const seal::Ciphertext &input_ct,
    int64_t shift,
    int64_t scalar,
    uint64_t log_q,
    seal::Ciphertext &acc_ct);
