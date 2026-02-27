#include <seal/seal.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "conv.h"
#include "decryption.h"
#include "encryption.h"
#include "util.h"

using namespace seal;
using Clock = std::chrono::high_resolution_clock;

int main()
{
    constexpr size_t N = 2048;
    constexpr size_t H = 32;
    constexpr size_t W = 32;
    constexpr size_t KH = 3;
    constexpr size_t KW = 3;
    constexpr size_t Cin = 2;
    constexpr uint64_t log_q = 50;
    constexpr int delta_shift = 20;
    constexpr int iters = 200;

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, { 50 }));
    parms.set_plain_modulus(PlainModulus::Batching(N, 20));

    SEALContext context(parms);
    if (!context.parameters_set()) {
        std::cerr << "SEAL parameter error: " << context.parameter_error_name()
                  << " - " << context.parameter_error_message() << "\n";
        return 2;
    }

    const uint64_t plain_mod = parms.plain_modulus().value();
    const uint64_t mask = (1ULL << log_q) - 1ULL;

    KeyGenerator keygen(context);
    SecretKey sk = keygen.secret_key();
    const Plaintext &sk_pt = sk.data();
    PublicKey pk;
    keygen.create_public_key(pk);
    Encryptor encryptor(context, pk);
    Evaluator evaluator(context);
    Decryptor decryptor(context, sk);

    std::mt19937_64 rng(20260227ULL);
    std::uniform_int_distribution<int> dist(-2, 2);

    std::vector<std::vector<int64_t>> images(Cin, std::vector<int64_t>(H * W, 0));
    for (size_t ch = 0; ch < Cin; ++ch) {
        for (size_t i = 0; i < H * W; ++i) {
            images[ch][i] = dist(rng);
        }
    }

    // kernel layout: [ch][kh][kw] flattened as Cin*KH*KW.
    std::vector<int64_t> kernel_flat_cin(Cin * KH * KW, 0);
    const std::vector<int64_t> k0 = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };
    const std::vector<int64_t> k1 = {
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1
    };
    for (size_t i = 0; i < KH * KW; ++i) {
        kernel_flat_cin[i] = k0[i];
        kernel_flat_cin[KH * KW + i] = k1[i];
    }

    // Reference:
    // - Cheetah PMult layout at out_base + r*W + c.
    // - mod2k rotate+cmult layout at r*W + c.
    const size_t out_base = cheetah_filter_base_index(KH, KW, W);
    std::vector<int64_t> ref_poly(N, 0);
    std::vector<int64_t> ref_rot(N, 0);
    for (size_t r = 0; r + KH <= H; ++r) {
        for (size_t c = 0; c + KW <= W; ++c) {
            int64_t acc = 0;
            for (size_t ch = 0; ch < Cin; ++ch) {
                const size_t kbase = ch * KH * KW;
                for (size_t kr = 0; kr < KH; ++kr) {
                    for (size_t kc = 0; kc < KW; ++kc) {
                        acc += images[ch][(r + kr) * W + (c + kc)] * kernel_flat_cin[kbase + kr * KW + kc];
                    }
                }
            }
            ref_poly[out_base + r * W + c] = acc;
            ref_rot[r * W + c] = acc;
        }
    }

    std::vector<Ciphertext> image_cts_seal(Cin);
    std::vector<Ciphertext> image_cts_mod2k(Cin);
    for (size_t ch = 0; ch < Cin; ++ch) {
        Plaintext image_pt = encode_image_coeff_plain(images[ch], H, W, N, plain_mod);
        encryptor.encrypt(image_pt, image_cts_seal[ch]);

        std::vector<uint64_t> image_mod2k(N, 0);
        for (size_t i = 0; i < H * W; ++i) {
            image_mod2k[i] = static_cast<uint64_t>(images[ch][i]) & mask;
        }
        scale_by_pow2_inplace(image_mod2k, delta_shift, static_cast<int>(log_q));
        Plaintext image_pt_mod2k = vector_to_plaintext_coeff(image_mod2k, N, static_cast<int>(log_q));
        encrypt_zero_nttfree(context, image_cts_mod2k[ch], sk_pt, log_q);
        add_plain_to_ct_inplace_mod2k(image_cts_mod2k[ch], image_pt_mod2k, log_q);
    }

    // ---------------------------
    // Cheetah-style PMult (multi-in-channel accumulate)
    // ---------------------------
    Ciphertext conv_pmult_ct;
    conv2d_pmult_accum_multi_in(
        image_cts_seal, kernel_flat_cin, Cin, H, W, KH, KW, plain_mod, evaluator, conv_pmult_ct);
    const std::vector<size_t> valid_indices = cheetah_valid_output_indices(H, W, KH, KW, N);
    extract_valid_coeffs_inplace(conv_pmult_ct, evaluator, valid_indices);

    Plaintext conv_pmult_pt;
    decryptor.decrypt(conv_pmult_ct, conv_pmult_pt);
    auto coeff_at = [&](size_t idx) -> uint64_t { return (idx < conv_pmult_pt.coeff_count()) ? conv_pmult_pt[idx] : 0ULL; };

    size_t pmult_mismatches = 0;
    for (size_t idx : valid_indices) {
        const int64_t got = decode_plain_coeff_to_signed(coeff_at(idx), plain_mod);
        if (got != ref_poly[idx]) ++pmult_mismatches;
    }

    std::cout << "[SEAL PMult conv: multi input channels]\n";
    std::cout << "Cin=" << Cin << ", N=" << N << ", HxW=" << H << "x" << W << ", K=" << KH << "x" << KW
              << ", valid_count=" << valid_indices.size() << ", valid_mismatches=" << pmult_mismatches << "\n";
    std::cout << "sample(valid outputs, first 12)\n";
    size_t printed = 0;
    for (size_t r = 0; r + KH <= H && printed < 12; ++r) {
        for (size_t c = 0; c + KW <= W && printed < 12; ++c) {
            const size_t idx = out_base + r * W + c;
            const int64_t got = decode_plain_coeff_to_signed(coeff_at(idx), plain_mod);
            std::cout << "  [" << idx << "] expected=" << ref_poly[idx] << ", got=" << got << "\n";
            ++printed;
        }
    }

    // ---------------------------
    // mod2k rotate+CMult (multi-in-channel accumulate)
    // ---------------------------
    Ciphertext conv_rot_ct_mod2k;
    conv2d_rot_cmult_accum_multi_in_mod2k(
        image_cts_mod2k, kernel_flat_cin, Cin, H, W, KH, KW, log_q, conv_rot_ct_mod2k);

    Plaintext conv_rot_scaled_pt_mod2k;
    decrypt_nttfree(context, conv_rot_ct_mod2k, sk_pt, conv_rot_scaled_pt_mod2k, log_q);
    std::vector<int64_t> conv_rot_decoded =
        decode_divide_pow2(conv_rot_scaled_pt_mod2k, delta_shift, static_cast<int>(log_q), N);

    const auto valid_rot_indices = valid_output_indices_rot_cmult_mod2k(H, W, KH, KW, N);
    size_t rot_mismatches = 0;
    for (size_t idx : valid_rot_indices) {
        if (conv_rot_decoded[idx] != ref_rot[idx]) ++rot_mismatches;
    }

    std::cout << "[_mod2k rotate+CMult conv: multi input channels]\n";
    std::cout << "Cin=" << Cin << ", N=" << N << ", log_q=" << log_q
              << ", valid_count=" << valid_rot_indices.size() << ", valid_mismatches=" << rot_mismatches << "\n";
    std::cout << "sample(valid outputs, first 12)\n";
    printed = 0;
    for (size_t r = 0; r + KH <= H && printed < 12; ++r) {
        for (size_t c = 0; c + KW <= W && printed < 12; ++c) {
            const size_t idx = r * W + c;
            std::cout << "  [" << idx << "] expected=" << ref_rot[idx] << ", got=" << conv_rot_decoded[idx] << "\n";
            ++printed;
        }
    }

    // ---------------------------
    // Speed benchmark (conv core only)
    // ---------------------------
    uint64_t bench_checksum = 0;

    const auto t0_pmult = Clock::now();
    for (int i = 0; i < iters; ++i) {
        Ciphertext ct_work;
        conv2d_pmult_accum_multi_in(
            image_cts_seal, kernel_flat_cin, Cin, H, W, KH, KW, plain_mod, evaluator, ct_work);
        extract_valid_coeffs_inplace(ct_work, evaluator, valid_indices);
        bench_checksum ^= ct_work.data(0)[out_base];
    }
    const auto t1_pmult = Clock::now();
    const double pmult_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_pmult - t0_pmult).count());

    const auto t0_rot = Clock::now();
    for (int i = 0; i < iters; ++i) {
        Ciphertext ct_work;
        conv2d_rot_cmult_accum_multi_in_mod2k(
            image_cts_mod2k, kernel_flat_cin, Cin, H, W, KH, KW, log_q, ct_work);
        bench_checksum ^= ct_work.data(0)[0];
    }
    const auto t1_rot = Clock::now();
    const double rot_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_rot - t0_rot).count());

    std::cout << "[Speed comparison: conv core only, multi input channels]\n";
    std::cout << "iters=" << iters << "\n";
    std::cout << "  pmult+extract total_us=" << pmult_us << ", avg_us=" << (pmult_us / iters) << "\n";
    std::cout << "  rot+cmult+accum_mod2k total_us=" << rot_us << ", avg_us=" << (rot_us / iters) << "\n";
    std::cout << "  bench_checksum=" << bench_checksum << "\n";

    return 0;
}
