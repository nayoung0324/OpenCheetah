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

// Multi-input-channel conv (channel-accumulate) with Cheetah-style PMult.
// input_cts.size() must be Cin, kernel_flat_cin size must be Cin*KH*KW.
// Each channel is convolved independently and accumulated into out_ct.
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

// Multi-input-channel conv (channel-accumulate) with _mod2k rotate+CMult.
// input_cts.size() must be Cin, kernel_flat_cin size must be Cin*KH*KW.
void conv2d_rot_cmult_accum_multi_in_mod2k(
    const std::vector<seal::Ciphertext> &input_cts,
    const std::vector<int64_t> &kernel_flat_cin,
    size_t Cin,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t log_q,
    seal::Ciphertext &out_ct);

// Pack up to channels_per_ct input channels into one plaintext (coefficient encoding).
// Global channel g maps to local channel (g - ch_begin) at coefficient offset local*H*W.
seal::Plaintext encode_image_coeff_plain_packed(
    const std::vector<std::vector<int64_t>> &images,
    size_t ch_begin,
    size_t channels_per_ct,
    size_t H,
    size_t W,
    size_t N,
    uint64_t plain_modulus);

// Build packed multi-channel kernel plaintext for PMult.
// Uses Cheetah-style index:
// begin = H*W*(channels_per_ct-1) + W*(KH-1) + (KW-1)
// coeff = begin - local_ch*H*W - kr*W - kc.
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

// _mod2k rotate+CMult conv with packed input channels.
void conv2d_rot_cmult_accum_multi_in_packed_mod2k(
    const std::vector<seal::Ciphertext> &input_cts_packed,
    const std::vector<int64_t> &kernel_flat_cin,
    size_t Cin,
    size_t channels_per_ct,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t log_q,
    seal::Ciphertext &out_ct);
