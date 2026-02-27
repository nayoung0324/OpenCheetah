#include "conv.h"

#include <algorithm>
#include <stdexcept>

#include "util.h"

namespace
{
size_t idx2d(size_t r, size_t c, size_t W)
{
    return r * W + c;
}

void rotate_multiply_scalar_add_ct_mod2k_impl(
    const seal::Ciphertext &input_ct,
    int64_t shift,
    uint64_t scalar_mod2k,
    uint64_t log_q,
    seal::Ciphertext &acc_ct,
    std::vector<uint64_t> &rotated)
{
    const size_t n = input_ct.poly_modulus_degree();
    const uint64_t mask = (1ULL << log_q) - 1ULL;
    for (size_t comp = 0; comp < input_ct.size(); ++comp) {
        const uint64_t *in = input_ct.data(comp);
        uint64_t *out = acc_ct.data(comp);
        rotate_poly_coeff_mod2k(in, rotated.data(), n, shift, log_q);
        for (size_t i = 0; i < n; ++i) {
            const auto prod = static_cast<unsigned __int128>(rotated[i]) * scalar_mod2k;
            out[i] = (out[i] + (static_cast<uint64_t>(prod) & mask)) & mask;
        }
    }
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

void extract_valid_coeffs_inplace(
    seal::Ciphertext &ct, const seal::Evaluator &evaluator, const std::vector<size_t> &valid_indices)
{
    if (ct.size() == 0) return;
    const size_t N = ct.poly_modulus_degree();
    const size_t L = ct.coeff_modulus_size();

    if (valid_indices.empty() || valid_indices.size() > N) {
        throw std::invalid_argument("invalid valid_indices");
    }
    if (std::any_of(valid_indices.begin(), valid_indices.end(), [N](size_t c) { return c >= N; })) {
        throw std::invalid_argument("valid index out of range");
    }

    if (ct.is_ntt_form()) {
        evaluator.transform_from_ntt_inplace(ct);
    }

    std::vector<size_t> keep = valid_indices;
    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());

    // Cheetah-style extract: zero-out only c0 coefficients.
    uint64_t *c0_rns = ct.data(0);
    for (size_t idx = 0; idx < N; ++idx) {
        if (std::binary_search(keep.begin(), keep.end(), idx)) continue;
        uint64_t *ptr = c0_rns + idx;
        for (size_t l = 0; l < L; ++l) {
            *ptr = 0;
            ptr += N;
        }
    }
}

void conv2d_rot_cmult_accum_mod2k(
    const seal::Ciphertext &input_ct,
    const std::vector<int64_t> &kernel_flat,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t log_q,
    seal::Ciphertext &out_ct)
{
    if (H == 0 || W == 0 || KH == 0 || KW == 0) {
        throw std::invalid_argument("H/W/KH/KW must be non-zero");
    }
    if (kernel_flat.size() != KH * KW) {
        throw std::invalid_argument("kernel_flat size must be KH*KW");
    }
    if (input_ct.poly_modulus_degree() < H * W) {
        throw std::invalid_argument("input ciphertext poly degree is smaller than H*W");
    }
    if (KH > H || KW > W) {
        throw std::invalid_argument("KH/KW must be <= H/W");
    }
    if (input_ct.coeff_modulus_size() != 1) {
        throw std::invalid_argument("conv2d_rot_cmult_accum_mod2k expects single coeff modulus");
    }

    const size_t n = input_ct.poly_modulus_degree();

    // Allocate destination with the same metadata and zero coefficients.
    out_ct = input_ct;
    for (size_t comp = 0; comp < out_ct.size(); ++comp) {
        uint64_t *out = out_ct.data(comp);
        for (size_t i = 0; i < n; ++i) out[i] = 0;
    }

    std::vector<uint64_t> rotated(n, 0);
    // y = sum_{kr,kc} w[kr,kc] * Rot(input, -(kr*W+kc))
    for (size_t kr = 0; kr < KH; ++kr) {
        for (size_t kc = 0; kc < KW; ++kc) {
            const int64_t w = kernel_flat[idx2d(kr, kc, KW)];
            if (w == 0) continue;
            const int64_t shift = -static_cast<int64_t>(kr * W + kc);
            const uint64_t w_mod2k = static_cast<uint64_t>(w) & ((1ULL << log_q) - 1ULL);
            rotate_multiply_scalar_add_ct_mod2k_impl(input_ct, shift, w_mod2k, log_q, out_ct, rotated);
        }
    }
}

std::vector<size_t> valid_output_indices_rot_cmult_mod2k(
    size_t H, size_t W, size_t KH, size_t KW, size_t N)
{
    if (H == 0 || W == 0 || KH == 0 || KW == 0 || N == 0) {
        throw std::invalid_argument("H/W/KH/KW/N must be non-zero");
    }
    if (KH > H || KW > W) {
        throw std::invalid_argument("KH/KW must be <= H/W");
    }
    if (H * W > N) {
        throw std::invalid_argument("H*W must be <= N");
    }

    std::vector<size_t> out;
    out.reserve((H - KH + 1) * (W - KW + 1));
    for (size_t r = 0; r + KH <= H; ++r) {
        for (size_t c = 0; c + KW <= W; ++c) {
            out.push_back(r * W + c);
        }
    }
    return out;
}

void rotate_multiply_scalar_add_ct_mod2k(
    const seal::Ciphertext &input_ct,
    int64_t shift,
    int64_t scalar,
    uint64_t log_q,
    seal::Ciphertext &acc_ct)
{
    if (input_ct.size() == 0 || acc_ct.size() == 0) {
        throw std::invalid_argument("ciphertext must have at least one component");
    }
    if (input_ct.size() != acc_ct.size()) {
        throw std::invalid_argument("ciphertext size mismatch");
    }
    if (input_ct.poly_modulus_degree() != acc_ct.poly_modulus_degree()) {
        throw std::invalid_argument("poly_modulus_degree mismatch");
    }
    if (input_ct.coeff_modulus_size() != 1 || acc_ct.coeff_modulus_size() != 1) {
        throw std::invalid_argument("rotate_multiply_scalar_add_ct_mod2k expects single coeff modulus");
    }
    if (log_q == 0 || log_q > 62) {
        throw std::invalid_argument("log_q must be in [1,62]");
    }

    const uint64_t mask = (1ULL << log_q) - 1ULL;
    const uint64_t a = static_cast<uint64_t>(scalar) & mask;
    std::vector<uint64_t> rotated(input_ct.poly_modulus_degree(), 0);
    rotate_multiply_scalar_add_ct_mod2k_impl(input_ct, shift, a, log_q, acc_ct, rotated);
}
