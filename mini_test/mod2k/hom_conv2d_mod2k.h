#pragma once

#include <seal/seal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mini_test::mod2k
{
struct Conv2DMeta
{
    size_t poly_degree{ 0 };
    size_t kernel_h{ 0 };
    size_t kernel_w{ 0 };
    size_t pad_h{ 0 };
    size_t pad_w{ 0 };
    size_t stride{ 1 };
    uint64_t log_q{ 0 };
    int delta_shift{ 0 };
};

// _mod2k core helpers/functions (used by wrapper and benchmarks).
void remove_unused_coeffs_mod2k_inplace(seal::Ciphertext &ct, const std::vector<size_t> &used_indices);

void conv2d_rot_cmult_accum_mod2k(
    const seal::Ciphertext &input_ct,
    const std::vector<int64_t> &kernel_flat,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t log_q,
    seal::Ciphertext &out_ct);

std::vector<size_t> valid_output_indices_rot_cmult_mod2k(
    size_t H, size_t W, size_t KH, size_t KW, size_t N);

void rotate_multiply_scalar_add_ct_mod2k(
    const seal::Ciphertext &input_ct,
    int64_t shift,
    int64_t scalar,
    uint64_t log_q,
    seal::Ciphertext &acc_ct);

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

void conv2d_rot_cmult_multi_out_mod2k(
    const std::vector<seal::Ciphertext> &input_cts,
    const std::vector<int64_t> &kernels_flat_co_cin,
    size_t Co,
    size_t Cin,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t log_q,
    std::vector<seal::Ciphertext> &out_cts);

// End-to-end mod2k conv wrapper:
// padding -> tiling -> channel-wise encrypt -> conv core -> remove-unused ->
// decrypt/decode -> compact(valid) -> scatter(full output).
//
// input_images layout: [Cin][H*W]
// kernels_flat_co_cin layout: [Co][Cin][KH][KW], flattened.
// output_maps layout: [Co][out_h*out_w], where out_h/out_w follow SAME/stride.
void run_conv_layer_mod2k(
    const seal::SEALContext &context,
    const seal::Plaintext &sk_pt,
    const Conv2DMeta &meta,
    const std::vector<std::vector<int64_t>> &input_images,
    const std::vector<int64_t> &kernels_flat_co_cin,
    size_t Co,
    std::vector<std::vector<int64_t>> &output_maps);
} // namespace mini_test::mod2k
