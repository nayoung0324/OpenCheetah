#include "mod2k/hom_conv2d_mod2k.h"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include "common/conv_common.h"
#include "common/util.h"
#include "mod2k/decryption.h"
#include "mod2k/encoding_mod2k.h"
#include "mod2k/encryption.h"

namespace mini_test::mod2k
{
#ifndef MOD2K_WRAPPER_BENCH
#define MOD2K_WRAPPER_BENCH 0
#endif

namespace
{
#if MOD2K_WRAPPER_BENCH
using BenchClock = std::chrono::high_resolution_clock;
static long long g_zero_unused_us = 0;
static uint64_t g_zero_unused_calls = 0;
#endif

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
#if MOD2K_WRAPPER_BENCH
    const auto t0 = BenchClock::now();
#endif
    const size_t N = ct.poly_modulus_degree();
    const size_t L = ct.coeff_modulus_size();
    std::vector<size_t> keep = used_indices;
    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());

    uint64_t *c0_rns = ct.data(0);
    for (size_t idx = 0; idx < N; ++idx) {
        if (std::binary_search(keep.begin(), keep.end(), idx)) continue;
        uint64_t *ptr = c0_rns + idx;
        for (size_t l = 0; l < L; ++l) {
            *ptr = 0;
            ptr += N;
        }
    }
#if MOD2K_WRAPPER_BENCH
    const auto t1 = BenchClock::now();
    g_zero_unused_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    ++g_zero_unused_calls;
#endif
}

// Internal fused primitive: rotate one ciphertext, multiply by scalar, and accumulate.
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

std::vector<std::vector<int64_t>> zero_pad_same(
    const std::vector<std::vector<int64_t>> &images, size_t H, size_t W, size_t pad_h, size_t pad_w)
{
    const size_t Cin = images.size();
    const size_t Hp = H + 2 * pad_h;
    const size_t Wp = W + 2 * pad_w;
    std::vector<std::vector<int64_t>> out(Cin, std::vector<int64_t>(Hp * Wp, 0));
    for (size_t ci = 0; ci < Cin; ++ci) {
        if (images[ci].size() != H * W) {
            throw std::invalid_argument("input channel size mismatch");
        }
        for (size_t r = 0; r < H; ++r) {
            for (size_t c = 0; c < W; ++c) {
                out[ci][(r + pad_h) * Wp + (c + pad_w)] = images[ci][r * W + c];
            }
        }
    }
    return out;
}
} // namespace

void remove_unused_coeffs_mod2k_inplace(seal::Ciphertext &ct, const std::vector<size_t> &used_indices)
{
    if (ct.size() == 0) return;
    const size_t N = ct.poly_modulus_degree();
    validate_used_indices(used_indices, N);

    if (ct.is_ntt_form()) {
        throw std::invalid_argument("remove_unused_coeffs_mod2k_inplace expects coefficient-form ciphertext");
    }
    zero_unused_in_c0_rns(ct, used_indices);
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
    out_ct = input_ct;
    for (size_t comp = 0; comp < out_ct.size(); ++comp) {
        uint64_t *out = out_ct.data(comp);
        for (size_t i = 0; i < n; ++i) out[i] = 0;
    }

    std::vector<uint64_t> rotated(n, 0);
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

// Cin개 ciphertext 각각에 대해 conv2d_rot_cmult_accum_mod2k을 수행한 뒤 결과를 모두 더해서 하나의 ciphertext로 반환
// 즉 Cout = 1
void conv2d_rot_cmult_accum_multi_in_mod2k(
    const std::vector<seal::Ciphertext> &input_cts,
    const std::vector<int64_t> &kernel_flat_cin,
    size_t Cin,
    size_t H,
    size_t W,
    size_t KH,
    size_t KW,
    uint64_t log_q,
    seal::Ciphertext &out_ct)
{
    if (Cin == 0) throw std::invalid_argument("Cin must be non-zero");
    if (input_cts.size() != Cin) throw std::invalid_argument("input_cts size must be Cin");
    if (kernel_flat_cin.size() != Cin * KH * KW) {
        throw std::invalid_argument("kernel_flat_cin size must be Cin*KH*KW");
    }

    bool init = false;
    for (size_t ch = 0; ch < Cin; ++ch) {
        std::vector<int64_t> kernel_ch(KH * KW, 0);
        const size_t base = ch * KH * KW;
        for (size_t i = 0; i < KH * KW; ++i) {
            kernel_ch[i] = kernel_flat_cin[base + i];
        }

        seal::Ciphertext term;
        conv2d_rot_cmult_accum_mod2k(input_cts[ch], kernel_ch, H, W, KH, KW, log_q, term);
        if (!init) {
            out_ct = term;
            init = true;
        } else {
            add_ct_inplace_mod2k(out_ct, term, log_q);
        }
    }
}

// 이건 안 쓰는거
/*
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
    seal::Ciphertext &out_ct)
{
    if (Cin == 0 || channels_per_ct == 0) throw std::invalid_argument("Cin/channels_per_ct must be non-zero");
    if (kernel_flat_cin.size() != Cin * KH * KW) {
        throw std::invalid_argument("kernel_flat_cin size must be Cin*KH*KW");
    }
    const size_t n_ct = (Cin + channels_per_ct - 1) / channels_per_ct;
    if (input_cts_packed.size() != n_ct) throw std::invalid_argument("input_cts_packed size mismatch");

    const size_t one_ch = H * W;
    bool init = false;
    for (size_t g = 0; g < n_ct; ++g) {
        const auto &ct = input_cts_packed[g];
        seal::Ciphertext term = ct;
        for (size_t comp = 0; comp < term.size(); ++comp) {
            uint64_t *ptr = term.data(comp);
            for (size_t i = 0; i < term.poly_modulus_degree(); ++i) ptr[i] = 0;
        }

        for (size_t lc = 0; lc < channels_per_ct; ++lc) {
            const size_t gc = g * channels_per_ct + lc;
            if (gc >= Cin) break;
            const size_t kbase = gc * KH * KW;
            for (size_t kr = 0; kr < KH; ++kr) {
                for (size_t kc = 0; kc < KW; ++kc) {
                    const int64_t w = kernel_flat_cin[kbase + kr * KW + kc];
                    if (w == 0) continue;
                    const int64_t shift = -static_cast<int64_t>(lc * one_ch + kr * W + kc);
                    rotate_multiply_scalar_add_ct_mod2k(ct, shift, w, log_q, term);
                }
            }
        }

        if (!init) {
            out_ct = term;
            init = true;
        } else {
            add_ct_inplace_mod2k(out_ct, term, log_q);
        }
    }
}
*/

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
        conv2d_rot_cmult_accum_multi_in_mod2k(
            input_cts, kernel_flat_cin, Cin, H, W, KH, KW, log_q, out_cts[co]);
    }
}

void run_conv_layer_mod2k(
    const seal::SEALContext &context,
    const seal::Plaintext &sk_pt,
    const Conv2DMeta &meta,
    const std::vector<std::vector<int64_t>> &input_images,
    const std::vector<int64_t> &kernels_flat_co_cin,
    size_t Co,
    std::vector<std::vector<int64_t>> &output_maps)
{
#if MOD2K_WRAPPER_BENCH
    long long us_pad = 0;
    long long us_prepare_tile_patch = 0;
    long long us_encode_encrypt = 0;
    long long us_core_conv = 0;
    long long us_remove_unused = 0;
    long long us_decrypt_decode = 0;
    long long us_compact_scatter = 0;
    const long long zero_unused_us_before = g_zero_unused_us;
    const uint64_t zero_unused_calls_before = g_zero_unused_calls;
#endif

    if (meta.stride != 1) {
        throw std::invalid_argument("run_conv_layer_mod2k currently supports stride=1 only");
    }
    if (meta.kernel_h == 0 || meta.kernel_w == 0 || meta.poly_degree == 0) {
        throw std::invalid_argument("invalid meta: kernel/poly_degree");
    }
    if (meta.log_q == 0 || meta.log_q > 62) {
        throw std::invalid_argument("invalid meta.log_q");
    }
    if (input_images.empty()) {
        throw std::invalid_argument("input_images must be non-empty");
    }

    const size_t Cin = input_images.size();
    const size_t flat = input_images[0].size();
    for (size_t ci = 1; ci < Cin; ++ci) {
        if (input_images[ci].size() != flat) {
            throw std::invalid_argument("all input channels must share the same H*W");
        }
    }

    const size_t H = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(flat))));
    const size_t W = H;
    if (H * W != flat) {
        throw std::invalid_argument("input_images channels must be square (H*W)");
    }
    if (kernels_flat_co_cin.size() != Co * Cin * meta.kernel_h * meta.kernel_w) {
        throw std::invalid_argument("kernels_flat_co_cin size mismatch");
    }

    const size_t Hp = H + 2 * meta.pad_h;
    const size_t Wp = W + 2 * meta.pad_w;
    if (Hp < meta.kernel_h || Wp < meta.kernel_w) {
        throw std::invalid_argument("padded input is smaller than kernel");
    }
    const size_t out_h = Hp - meta.kernel_h + 1;
    const size_t out_w = Wp - meta.kernel_w + 1;
    const uint64_t mask = (1ULL << meta.log_q) - 1ULL;

#if MOD2K_WRAPPER_BENCH
    const auto t_pad0 = BenchClock::now();
#endif
    auto images_padded = zero_pad_same(input_images, H, W, meta.pad_h, meta.pad_w);
#if MOD2K_WRAPPER_BENCH
    const auto t_pad1 = BenchClock::now();
    us_pad += std::chrono::duration_cast<std::chrono::microseconds>(t_pad1 - t_pad0).count();
#endif
    output_maps.assign(Co, std::vector<int64_t>(out_h * out_w, 0));

    struct TilePrepared
    {
        Conv2DTile tile;
        size_t tile_in_h{ 0 };
        size_t tile_in_w{ 0 };
        std::vector<size_t> valid_indices;
        std::vector<seal::Ciphertext> image_cts;
    };
    std::vector<TilePrepared> prepared_tiles;

    const size_t one_ch = Hp * Wp;
    if (one_ch <= meta.poly_degree) {
#if MOD2K_WRAPPER_BENCH
        const auto t_tp0 = BenchClock::now();
#endif
        TilePrepared tp;
        tp.tile = { 0, 0, out_h, out_w, Hp, Wp };
        tp.tile_in_h = Hp;
        tp.tile_in_w = Wp;
        tp.valid_indices = valid_output_indices_rot_cmult_mod2k(Hp, Wp, meta.kernel_h, meta.kernel_w, meta.poly_degree);
#if MOD2K_WRAPPER_BENCH
        const auto t_tp1 = BenchClock::now();
        us_prepare_tile_patch += std::chrono::duration_cast<std::chrono::microseconds>(t_tp1 - t_tp0).count();
#endif
        tp.image_cts.resize(Cin);
        for (size_t ci = 0; ci < Cin; ++ci) {
#if MOD2K_WRAPPER_BENCH
            const auto t_enc0 = BenchClock::now();
#endif
            std::vector<uint64_t> coeffs(meta.poly_degree, 0);
            for (size_t i = 0; i < one_ch; ++i) {
                coeffs[i] = static_cast<uint64_t>(images_padded[ci][i]) & mask;
            }
            encode_message_inplace(coeffs, meta.delta_shift, static_cast<int>(meta.log_q));
            seal::Plaintext pt = to_plaintext(coeffs, meta.poly_degree, static_cast<int>(meta.log_q));
            encrypt_zero_nttfree(context, tp.image_cts[ci], sk_pt, meta.log_q);
            add_plain_to_ct_inplace_mod2k(tp.image_cts[ci], pt, meta.log_q);
#if MOD2K_WRAPPER_BENCH
            const auto t_enc1 = BenchClock::now();
            us_encode_encrypt += std::chrono::duration_cast<std::chrono::microseconds>(t_enc1 - t_enc0).count();
#endif
        }
        prepared_tiles.push_back(std::move(tp));
    } else {
#if MOD2K_WRAPPER_BENCH
        const auto t_tiles0 = BenchClock::now();
#endif
        const auto tiles = make_conv2d_tiles_valid(Hp, Wp, meta.kernel_h, meta.kernel_w, meta.poly_degree);
#if MOD2K_WRAPPER_BENCH
        const auto t_tiles1 = BenchClock::now();
        us_prepare_tile_patch += std::chrono::duration_cast<std::chrono::microseconds>(t_tiles1 - t_tiles0).count();
#endif
        for (const auto &tile : tiles) {
#if MOD2K_WRAPPER_BENCH
            const auto t_tp0 = BenchClock::now();
#endif
            TilePrepared tp;
            tp.tile = tile;
            tp.tile_in_h = tile.in_h;
            tp.tile_in_w = tile.in_w;
            tp.valid_indices = valid_output_indices_rot_cmult_mod2k(
                tile.in_h, tile.in_w, meta.kernel_h, meta.kernel_w, meta.poly_degree);
            tp.image_cts.resize(Cin);
#if MOD2K_WRAPPER_BENCH
            const auto t_tp1 = BenchClock::now();
            us_prepare_tile_patch += std::chrono::duration_cast<std::chrono::microseconds>(t_tp1 - t_tp0).count();
#endif

            for (size_t ci = 0; ci < Cin; ++ci) {
#if MOD2K_WRAPPER_BENCH
                const auto t_patch0 = BenchClock::now();
#endif
                const auto patch = extract_input_patch_by_tile(images_padded[ci], Hp, Wp, tile);
#if MOD2K_WRAPPER_BENCH
                const auto t_patch1 = BenchClock::now();
                us_prepare_tile_patch += std::chrono::duration_cast<std::chrono::microseconds>(t_patch1 - t_patch0).count();

                const auto t_enc0 = BenchClock::now();
#endif
                std::vector<uint64_t> coeffs(meta.poly_degree, 0);
                for (size_t i = 0; i < patch.size(); ++i) {
                    coeffs[i] = static_cast<uint64_t>(patch[i]) & mask;
                }
                encode_message_inplace(coeffs, meta.delta_shift, static_cast<int>(meta.log_q));
                seal::Plaintext pt = to_plaintext(coeffs, meta.poly_degree, static_cast<int>(meta.log_q));
                encrypt_zero_nttfree(context, tp.image_cts[ci], sk_pt, meta.log_q);
                add_plain_to_ct_inplace_mod2k(tp.image_cts[ci], pt, meta.log_q);
#if MOD2K_WRAPPER_BENCH
                const auto t_enc1 = BenchClock::now();
                us_encode_encrypt += std::chrono::duration_cast<std::chrono::microseconds>(t_enc1 - t_enc0).count();
#endif
            }
            prepared_tiles.push_back(std::move(tp));
        }
    }
    for (const auto &tp : prepared_tiles) {
        std::vector<seal::Ciphertext> out_cts;
#if MOD2K_WRAPPER_BENCH
        const auto t_core0 = BenchClock::now();
#endif
        conv2d_rot_cmult_multi_out_mod2k(
            tp.image_cts,
            kernels_flat_co_cin,
            Co,
            Cin,
            tp.tile_in_h,
            tp.tile_in_w,
            meta.kernel_h,
            meta.kernel_w,
            meta.log_q,
            out_cts);
#if MOD2K_WRAPPER_BENCH
        const auto t_core1 = BenchClock::now();
        us_core_conv += std::chrono::duration_cast<std::chrono::microseconds>(t_core1 - t_core0).count();
#endif

        for (size_t co = 0; co < Co; ++co) {
#if MOD2K_WRAPPER_BENCH
            const auto t_rm0 = BenchClock::now();
#endif
            remove_unused_coeffs_mod2k_inplace(out_cts[co], tp.valid_indices);
#if MOD2K_WRAPPER_BENCH
            const auto t_rm1 = BenchClock::now();
            us_remove_unused += std::chrono::duration_cast<std::chrono::microseconds>(t_rm1 - t_rm0).count();

            const auto t_dec0 = BenchClock::now();
#endif
            seal::Plaintext scaled;
            decrypt_nttfree(context, out_cts[co], sk_pt, scaled, meta.log_q);
            std::vector<int64_t> decoded =
                decode_divide_pow2(scaled, meta.delta_shift, static_cast<int>(meta.log_q), meta.poly_degree);
#if MOD2K_WRAPPER_BENCH
            const auto t_dec1 = BenchClock::now();
            us_decrypt_decode += std::chrono::duration_cast<std::chrono::microseconds>(t_dec1 - t_dec0).count();

            const auto t_post0 = BenchClock::now();
#endif
            const auto compact = compact_coeffs_to_prefix(decoded, tp.valid_indices, meta.poly_degree);

            std::vector<int64_t> patch(tp.tile.out_h * tp.tile.out_w, 0);
            for (size_t i = 0; i < patch.size(); ++i) {
                patch[i] = compact[i];
            }
            scatter_output_patch_by_tile(patch, tp.tile, out_w, output_maps[co]);
#if MOD2K_WRAPPER_BENCH
            const auto t_post1 = BenchClock::now();
            us_compact_scatter += std::chrono::duration_cast<std::chrono::microseconds>(t_post1 - t_post0).count();
#endif
        }
    }

#if MOD2K_WRAPPER_BENCH
    const long long zero_unused_us_delta = g_zero_unused_us - zero_unused_us_before;
    const uint64_t zero_unused_calls_delta = g_zero_unused_calls - zero_unused_calls_before;
    std::cout << "[mod2k wrapper bench] "
              << "pad_us=" << us_pad
              << ", prep_tile_patch_us=" << us_prepare_tile_patch
              << ", encode_encrypt_us=" << us_encode_encrypt
              << ", core_conv_us=" << us_core_conv
              << ", remove_unused_us=" << us_remove_unused
              << ", decrypt_decode_us=" << us_decrypt_decode
              << ", compact_scatter_us=" << us_compact_scatter
              << "\n";
    std::cout << "[mod2k wrapper bench] zero_unused(c0-only): calls=" << zero_unused_calls_delta
              << ", total_us=" << zero_unused_us_delta
              << ", avg_us=" << (zero_unused_calls_delta ? (static_cast<double>(zero_unused_us_delta) / zero_unused_calls_delta) : 0.0)
              << "\n";
#endif
}
} // namespace mini_test::mod2k
