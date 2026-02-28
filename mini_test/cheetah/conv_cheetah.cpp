#include "cheetah/conv_cheetah.h"

#include <algorithm>
#include <stdexcept>

namespace
{
size_t idx2d(size_t r, size_t c, size_t W)
{
    return r * W + c;
}

void validate_used_indices(const std::vector<size_t> &used_indices, size_t N)
{
    if (used_indices.empty() || used_indices.size() > N) {
        throw std::invalid_argument("invalid used_indices");
    }
    if (std::any_of(used_indices.begin(), used_indices.end(), [N](size_t c) { return c >= N; })) {
        throw std::invalid_argument("used index out of range");
    }
}

void zero_unused_in_c0_rns(seal::Ciphertext &ct, const std::vector<size_t> &used_indices)
{
    const size_t N = ct.poly_modulus_degree();
    const size_t L = ct.coeff_modulus_size();
    std::vector<size_t> keep = used_indices;
    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());

    // Zero only c0 as in Cheetah's remove_unused_coeffs.
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
} // namespace

// Build Cheetah-style filter plaintext for PMult (single-channel).
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

// Return Cheetah filter base index for coefficient layout.
size_t cheetah_filter_base_index(size_t KH, size_t KW, size_t W)
{
    if (KH == 0 || KW == 0 || W == 0) throw std::invalid_argument("KH/KW/W must be non-zero");
    return W * (KH - 1) + (KW - 1);
}

// Plaintext reference conv matching Cheetah coefficient placement (valid region only).
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

// List valid output coefficient indices for Cheetah conv layout.
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

// Build a plaintext 0/1 mask that keeps only valid output coefficients.
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

// Cheetah-style extract: zero out unused coefficients in c0.
void extract_valid_coeffs_inplace(
    seal::Ciphertext &ct, const seal::Evaluator &evaluator, const std::vector<size_t> &valid_indices)
{
    remove_unused_coeffs_cheetah_inplace(ct, evaluator, valid_indices);
}

// Cheetah-style remove-unused: zero out unused coefficients in c0 only.
void remove_unused_coeffs_cheetah_inplace(
    seal::Ciphertext &ct, const seal::Evaluator &evaluator, const std::vector<size_t> &used_indices)
{
    if (ct.size() == 0) return;
    const size_t N = ct.poly_modulus_degree();
    validate_used_indices(used_indices, N);

    if (ct.is_ntt_form()) {
        evaluator.transform_from_ntt_inplace(ct);
    }
    zero_unused_in_c0_rns(ct, used_indices);
}

// Multi-input-channel PMult conv with one ciphertext per input channel.
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
    seal::Ciphertext &out_ct)
{
    if (Cin == 0) throw std::invalid_argument("Cin must be non-zero");
    if (input_cts.size() != Cin) throw std::invalid_argument("input_cts size must be Cin");
    if (kernel_flat_cin.size() != Cin * KH * KW) {
        throw std::invalid_argument("kernel_flat_cin size must be Cin*KH*KW");
    }
    if (H == 0 || W == 0 || KH == 0 || KW == 0) {
        throw std::invalid_argument("H/W/KH/KW must be non-zero");
    }
    if (KH > H || KW > W) throw std::invalid_argument("KH/KW must be <= H/W");

    const size_t N = input_cts[0].poly_modulus_degree();
    for (size_t ch = 1; ch < Cin; ++ch) {
        if (input_cts[ch].size() != input_cts[0].size()) {
            throw std::invalid_argument("ciphertext size mismatch across channels");
        }
        if (input_cts[ch].poly_modulus_degree() != N) {
            throw std::invalid_argument("poly_modulus_degree mismatch across channels");
        }
    }

    bool init = false;
    for (size_t ch = 0; ch < Cin; ++ch) {
        std::vector<int64_t> kernel_ch(KH * KW, 0);
        const size_t base = ch * KH * KW;
        for (size_t i = 0; i < KH * KW; ++i) {
            kernel_ch[i] = kernel_flat_cin[base + i];
        }

        seal::Plaintext kernel_pt = build_conv_kernel_plain(kernel_ch, KH, KW, W, N, plain_modulus);
        seal::Ciphertext term = input_cts[ch];
        evaluator.multiply_plain_inplace(term, kernel_pt);
        if (!init) {
            out_ct = term;
            init = true;
        } else {
            evaluator.add_inplace(out_ct, term);
        }
    }
}

// Pack multiple input channels into one plaintext polynomial.
seal::Plaintext encode_image_coeff_plain_packed(
    const std::vector<std::vector<int64_t>> &images,
    size_t ch_begin,
    size_t channels_per_ct,
    size_t H,
    size_t W,
    size_t N,
    uint64_t plain_modulus)
{
    if (channels_per_ct == 0 || H == 0 || W == 0 || N == 0) {
        throw std::invalid_argument("channels_per_ct/H/W/N must be non-zero");
    }
    const size_t one_ch = H * W;
    if (channels_per_ct * one_ch > N) {
        throw std::invalid_argument("packed channels exceed polynomial size");
    }

    seal::Plaintext pt;
    pt.resize(N);
    for (size_t i = 0; i < N; ++i) pt[i] = 0;

    for (size_t lc = 0; lc < channels_per_ct; ++lc) {
        const size_t gc = ch_begin + lc;
        if (gc >= images.size()) break;
        if (images[gc].size() != one_ch) {
            throw std::invalid_argument("each channel image size must be H*W");
        }
        const size_t base = lc * one_ch;
        for (size_t i = 0; i < one_ch; ++i) {
            pt[base + i] = encode_signed_to_plain_coeff(images[gc][i], plain_modulus);
        }
    }
    return pt;
}

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
    uint64_t plain_modulus)
{
    if (channels_per_ct == 0 || H == 0 || W == 0 || KH == 0 || KW == 0 || N == 0) {
        throw std::invalid_argument("invalid zero parameter");
    }
    if (kernel_flat_cin.size() != Cin * KH * KW) {
        throw std::invalid_argument("kernel_flat_cin size must be Cin*KH*KW");
    }
    const size_t one_ch = H * W;
    if (channels_per_ct * one_ch > N) {
        throw std::invalid_argument("packed channels exceed polynomial size");
    }

    const size_t begin = one_ch * (channels_per_ct - 1) + W * (KH - 1) + (KW - 1);
    if (begin >= N) throw std::invalid_argument("kernel begin index out of range");

    seal::Plaintext pt;
    pt.resize(N);
    for (size_t i = 0; i < N; ++i) pt[i] = 0;

    for (size_t lc = 0; lc < channels_per_ct; ++lc) {
        const size_t gc = ch_begin + lc;
        if (gc >= Cin) break; // zero-padded channels
        const size_t kbase = gc * KH * KW;
        for (size_t kr = 0; kr < KH; ++kr) {
            for (size_t kc = 0; kc < KW; ++kc) {
                const size_t idx = begin - lc * one_ch - kr * W - kc;
                pt[idx] = encode_signed_to_plain_coeff(kernel_flat_cin[kbase + kr * KW + kc], plain_modulus);
            }
        }
    }
    return pt;
}

// PMult conv with packed input channels, then channel-group accumulation.
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
    seal::Ciphertext &out_ct)
{
    if (Cin == 0 || channels_per_ct == 0) throw std::invalid_argument("Cin/channels_per_ct must be non-zero");
    const size_t n_ct = (Cin + channels_per_ct - 1) / channels_per_ct;
    if (input_cts_packed.size() != n_ct) throw std::invalid_argument("input_cts_packed size mismatch");

    bool init = false;
    for (size_t g = 0; g < n_ct; ++g) {
        const size_t ch_begin = g * channels_per_ct;
        const size_t N = input_cts_packed[g].poly_modulus_degree();

        seal::Plaintext kernel_pt = build_conv_kernel_plain_packed(
            kernel_flat_cin, Cin, ch_begin, channels_per_ct, H, W, KH, KW, N, plain_modulus);
        if (kernel_pt.is_zero()) {
            continue;
        }

        seal::Ciphertext term = input_cts_packed[g];
        evaluator.multiply_plain_inplace(term, kernel_pt);
        if (!init) {
            out_ct = term;
            init = true;
        } else {
            evaluator.add_inplace(out_ct, term);
        }
    }

    if (!init) {
        throw std::invalid_argument("all packed kernels are zero; PMult output would be transparent");
    }
}

// Multi-output PMult wrapper: run one packed PMult conv per output channel.
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
    std::vector<seal::Ciphertext> &out_cts)
{
    if (Co == 0 || Cin == 0) throw std::invalid_argument("Co/Cin must be non-zero");
    if (kernels_flat_co_cin.size() != Co * Cin * KH * KW) {
        throw std::invalid_argument("kernels_flat_co_cin size must be Co*Cin*KH*KW");
    }

    out_cts.resize(Co);
    const size_t one_filter = Cin * KH * KW;
    for (size_t co = 0; co < Co; ++co) {
        std::vector<int64_t> kernel_flat_cin(one_filter, 0);
        const size_t base = co * one_filter;
        for (size_t i = 0; i < one_filter; ++i) kernel_flat_cin[i] = kernels_flat_co_cin[base + i];
        conv2d_pmult_accum_multi_in_packed(
            input_cts_packed,
            kernel_flat_cin,
            Cin,
            channels_per_ct,
            H,
            W,
            KH,
            KW,
            plain_modulus,
            evaluator,
            out_cts[co]);
    }
}
