#include "common/conv_common.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Encode signed integer to non-negative residue modulo plaintext modulus.
uint64_t encode_signed_to_plain_coeff(int64_t v, uint64_t plain_modulus)
{
    if (!plain_modulus) throw std::invalid_argument("plain_modulus must be non-zero");
    const int64_t t = static_cast<int64_t>(plain_modulus);
    int64_t x = v % t;
    if (x < 0) x += t;
    return static_cast<uint64_t>(x);
}

// Decode plaintext residue to centered signed integer representative.
int64_t decode_plain_coeff_to_signed(uint64_t v, uint64_t plain_modulus)
{
    if (!plain_modulus) throw std::invalid_argument("plain_modulus must be non-zero");
    v %= plain_modulus;
    const uint64_t half = plain_modulus >> 1;
    if (v <= half) return static_cast<int64_t>(v);
    return static_cast<int64_t>(v) - static_cast<int64_t>(plain_modulus);
}

// Encode single-channel image coefficients directly into plaintext polynomial.
seal::Plaintext encode_image_coeff_plain(
    const std::vector<int64_t> &image_flat, size_t H, size_t W, size_t N, uint64_t plain_modulus)
{
    if (H == 0 || W == 0 || N == 0) throw std::invalid_argument("H/W/N must be non-zero");
    if (image_flat.size() != H * W) throw std::invalid_argument("image_flat size must be H*W");
    if (H * W > N) throw std::invalid_argument("H*W must be <= N");

    seal::Plaintext pt;
    pt.resize(N);
    for (size_t i = 0; i < H * W; ++i) {
        pt[i] = encode_signed_to_plain_coeff(image_flat[i], plain_modulus);
    }
    for (size_t i = H * W; i < N; ++i) {
        pt[i] = 0;
    }
    return pt;
}

// NOTE:
// Arbitrary index compaction on ciphertext coefficients is not a valid HE operation
// unless implemented via proper automorphisms/keyswitching pipeline.
// Use compact_coeffs_to_prefix on decoded coefficients instead.
void compact_valid_coeffs_inplace(seal::Ciphertext &, const std::vector<size_t> &)
{
    throw std::logic_error(
        "compact_valid_coeffs_inplace is unsupported on ciphertext; compact after decryption");
}

// NOTE:
// Same limitation as above. Evaluator is accepted for API symmetry.
void compact_valid_coeffs_inplace(
    seal::Ciphertext &, const seal::Evaluator &, const std::vector<size_t> &)
{
    throw std::logic_error(
        "compact_valid_coeffs_inplace is unsupported on ciphertext; compact after decryption");
}

// Build valid-conv tile plan under an input coefficient budget.
std::vector<Conv2DTile> make_conv2d_tiles_valid(
    size_t H, size_t W, size_t KH, size_t KW, size_t max_input_coeffs)
{
    if (H == 0 || W == 0 || KH == 0 || KW == 0 || max_input_coeffs == 0) {
        throw std::invalid_argument("invalid zero parameter");
    }
    if (KH > H || KW > W) throw std::invalid_argument("KH/KW must be <= H/W");

    const size_t out_h = H - KH + 1;
    const size_t out_w = W - KW + 1;

    // Need at least one receptive field.
    if (KH * KW > max_input_coeffs) {
        throw std::invalid_argument("max_input_coeffs is too small for one kernel receptive field");
    }

    // Choose tile output shape (tile_out_h, tile_out_w) so that
    // (tile_out_h+KH-1) * (tile_out_w+KW-1) <= max_input_coeffs.
    size_t tile_out_h = out_h;
    size_t tile_out_w = out_w;

    // Start from a near-square patch and clamp to output dims.
    size_t patch_side = static_cast<size_t>(std::sqrt(static_cast<double>(max_input_coeffs)));
    if (patch_side < KH) patch_side = KH;
    if (patch_side < KW) patch_side = KW;
    tile_out_h = (patch_side >= KH) ? (patch_side - KH + 1) : 1;
    if (tile_out_h == 0) tile_out_h = 1;
    if (tile_out_h > out_h) tile_out_h = out_h;

    size_t in_h = tile_out_h + KH - 1;
    size_t max_in_w = max_input_coeffs / in_h;
    if (max_in_w < KW) {
        tile_out_h = 1;
        in_h = KH;
        max_in_w = max_input_coeffs / in_h;
    }
    tile_out_w = (max_in_w >= KW) ? (max_in_w - KW + 1) : 1;
    if (tile_out_w == 0) tile_out_w = 1;
    if (tile_out_w > out_w) tile_out_w = out_w;

    std::vector<Conv2DTile> tiles;
    for (size_t r = 0; r < out_h; r += tile_out_h) {
        const size_t oh = std::min(tile_out_h, out_h - r);
        for (size_t c = 0; c < out_w; c += tile_out_w) {
            const size_t ow = std::min(tile_out_w, out_w - c);
            Conv2DTile t;
            t.out_row = r;
            t.out_col = c;
            t.out_h = oh;
            t.out_w = ow;
            t.in_h = oh + KH - 1;
            t.in_w = ow + KW - 1;
            if (t.in_h * t.in_w > max_input_coeffs) {
                throw std::logic_error("internal error: tile input exceeds max_input_coeffs");
            }
            tiles.push_back(t);
        }
    }
    return tiles;
}

// Extract one input patch corresponding to a tile (including halo overlap).
std::vector<int64_t> extract_input_patch_by_tile(
    const std::vector<int64_t> &image_flat, size_t H, size_t W, const Conv2DTile &tile)
{
    if (image_flat.size() != H * W) throw std::invalid_argument("image_flat size must be H*W");
    if (tile.out_row + tile.in_h > H || tile.out_col + tile.in_w > W) {
        throw std::invalid_argument("tile input window out of range");
    }

    std::vector<int64_t> patch(tile.in_h * tile.in_w, 0);
    for (size_t r = 0; r < tile.in_h; ++r) {
        for (size_t c = 0; c < tile.in_w; ++c) {
            patch[r * tile.in_w + c] = image_flat[(tile.out_row + r) * W + (tile.out_col + c)];
        }
    }
    return patch;
}

// Scatter one tile output patch back to the full output tensor.
void scatter_output_patch_by_tile(
    const std::vector<int64_t> &patch_out,
    const Conv2DTile &tile,
    size_t out_W,
    std::vector<int64_t> &full_out)
{
    if (patch_out.size() != tile.out_h * tile.out_w) {
        throw std::invalid_argument("patch_out size must be tile.out_h*tile.out_w");
    }
    if (full_out.empty() || out_W == 0) throw std::invalid_argument("invalid full_out/out_W");

    const size_t out_H = full_out.size() / out_W;
    if (out_H * out_W != full_out.size()) {
        throw std::invalid_argument("full_out size must be divisible by out_W");
    }
    if (tile.out_row + tile.out_h > out_H || tile.out_col + tile.out_w > out_W) {
        throw std::invalid_argument("tile output window out of range");
    }

    for (size_t r = 0; r < tile.out_h; ++r) {
        for (size_t c = 0; c < tile.out_w; ++c) {
            full_out[(tile.out_row + r) * out_W + (tile.out_col + c)] = patch_out[r * tile.out_w + c];
        }
    }
}

// Compact selected coefficients into prefix order for next-stage consumption.
std::vector<int64_t> compact_coeffs_to_prefix(
    const std::vector<int64_t> &coeffs,
    const std::vector<size_t> &valid_indices,
    size_t out_size)
{
    if (out_size == 0) {
        throw std::invalid_argument("out_size must be non-zero");
    }
    if (valid_indices.size() > out_size) {
        throw std::invalid_argument("valid_indices size must be <= out_size");
    }

    std::vector<int64_t> out(out_size, 0);
    for (size_t i = 0; i < valid_indices.size(); ++i) {
        const size_t src = valid_indices[i];
        if (src >= coeffs.size()) {
            throw std::out_of_range("valid index out of range");
        }
        out[i] = coeffs[src];
    }
    return out;
}
