#include "conv.h"

#include <stdexcept>

namespace
{
size_t idx2d(size_t r, size_t c, size_t W)
{
    return r * W + c;
}
} // namespace

uint64_t encode_signed_to_plain_coeff(int64_t v, uint64_t plain_modulus)
{
    if (!plain_modulus) throw std::invalid_argument("plain_modulus must be non-zero");
    const int64_t t = static_cast<int64_t>(plain_modulus);
    int64_t x = v % t;
    if (x < 0) x += t;
    return static_cast<uint64_t>(x);
}

int64_t decode_plain_coeff_to_signed(uint64_t v, uint64_t plain_modulus)
{
    if (!plain_modulus) throw std::invalid_argument("plain_modulus must be non-zero");
    v %= plain_modulus;
    const uint64_t half = plain_modulus >> 1;
    if (v <= half) return static_cast<int64_t>(v);
    return static_cast<int64_t>(v) - static_cast<int64_t>(plain_modulus);
}

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

seal::Plaintext build_conv_kernel_plain(
    const std::vector<int64_t> &kernel_flat, size_t KH, size_t KW, size_t W, size_t N, uint64_t plain_modulus)
{
    if (KH == 0 || KW == 0 || W == 0 || N == 0) throw std::invalid_argument("KH/KW/W/N must be non-zero");
    if (kernel_flat.size() != KH * KW) throw std::invalid_argument("kernel_flat size must be KH*KW");
    const size_t begin = cheetah_filter_base_index(KH, KW, W);
    if (begin >= N) throw std::invalid_argument("kernel support exceeds polynomial size");

    seal::Plaintext pt;
    pt.resize(N);
    for (size_t i = 0; i < N; ++i) pt[i] = 0;

    // Cheetah FilterIndexer (single channel):
    // index = begin - r*W - c
    for (size_t r = 0; r < KH; ++r) {
        for (size_t c = 0; c < KW; ++c) {
            const size_t coeff_idx = begin - r * W - c;
            const int64_t w = kernel_flat[idx2d(r, c, KW)];
            pt[coeff_idx] = encode_signed_to_plain_coeff(w, plain_modulus);
        }
    }
    return pt;
}

size_t cheetah_filter_base_index(size_t KH, size_t KW, size_t W)
{
    if (KH == 0 || KW == 0 || W == 0) throw std::invalid_argument("KH/KW/W must be non-zero");
    return W * (KH - 1) + (KW - 1);
}

std::vector<int64_t> conv2d_reference_poly_cheetah_valid(
    const std::vector<int64_t> &image_flat,
    size_t H,
    size_t W,
    const std::vector<int64_t> &kernel_flat,
    size_t KH,
    size_t KW,
    size_t N)
{
    if (image_flat.size() != H * W) throw std::invalid_argument("image_flat size must be H*W");
    if (kernel_flat.size() != KH * KW) throw std::invalid_argument("kernel_flat size must be KH*KW");
    if (H * W > N) throw std::invalid_argument("H*W must be <= N");

    std::vector<int64_t> out(N, 0);
    if (H < KH || W < KW) return out;
    const size_t begin = cheetah_filter_base_index(KH, KW, W);
    if (begin >= N) throw std::invalid_argument("begin index out of polynomial range");

    // PMult + Cheetah filter index gives valid output at coeff begin + r*W + c:
    // out(r,c) = sum_{kr,kc} image(r+kr, c+kc) * kernel(kr,kc)
    for (size_t r = 0; r + KH <= H; ++r) {
        for (size_t c = 0; c + KW <= W; ++c) {
            int64_t acc = 0;
            for (size_t kr = 0; kr < KH; ++kr) {
                for (size_t kc = 0; kc < KW; ++kc) {
                    const int64_t x = image_flat[idx2d(r + kr, c + kc, W)];
                    const int64_t w = kernel_flat[idx2d(kr, kc, KW)];
                    acc += x * w;
                }
            }
            out[begin + idx2d(r, c, W)] = acc;
        }
    }
    return out;
}

std::vector<size_t> cheetah_valid_output_indices(size_t H, size_t W, size_t KH, size_t KW, size_t N)
{
    if (H == 0 || W == 0 || KH == 0 || KW == 0 || N == 0) {
        throw std::invalid_argument("H/W/KH/KW/N must be non-zero");
    }
    if (H < KH || W < KW) return {};

    const size_t begin = cheetah_filter_base_index(KH, KW, W);
    std::vector<size_t> out;
    out.reserve((H - KH + 1) * (W - KW + 1));

    for (size_t r = 0; r + KH <= H; ++r) {
        for (size_t c = 0; c + KW <= W; ++c) {
            const size_t idx = begin + r * W + c;
            if (idx >= N) throw std::invalid_argument("valid index exceeds polynomial degree");
            out.push_back(idx);
        }
    }
    return out;
}

seal::Plaintext build_extract_mask_plain(
    const std::vector<size_t> &valid_indices, size_t N, uint64_t plain_modulus)
{
    if (!plain_modulus) throw std::invalid_argument("plain_modulus must be non-zero");
    seal::Plaintext pt;
    pt.resize(N);
    for (size_t i = 0; i < N; ++i) pt[i] = 0;
    for (size_t idx : valid_indices) {
        if (idx >= N) throw std::invalid_argument("valid index out of range");
        pt[idx] = 1 % plain_modulus;
    }
    return pt;
}
