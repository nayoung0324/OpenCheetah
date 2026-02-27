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

    KeyGenerator keygen(context);
    SecretKey sk = keygen.secret_key();
    const Plaintext &sk_pt = sk.data();
    PublicKey pk;
    keygen.create_public_key(pk);
    Encryptor encryptor(context, pk);
    Evaluator evaluator(context);
    Decryptor decryptor(context, sk);

    // Reproducible small integer input/kernel to avoid wrap in plain modulus.
    std::mt19937_64 rng(20260226ULL);
    std::uniform_int_distribution<int> img_dist(0, 7);

    std::vector<int64_t> image(H * W, 0);
    for (size_t i = 0; i < image.size(); ++i) {
        image[i] = img_dist(rng);
    }

    // Example 3x3 kernel (coefficient encoding / PMult style).
    std::vector<int64_t> kernel = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };

    // Direct valid conv reference for rot+cmult layout (stored at idx=r*W+c).
    std::vector<int64_t> ref_rot(N, 0);
    for (size_t r = 0; r + KH <= H; ++r) {
        for (size_t c = 0; c + KW <= W; ++c) {
            int64_t acc = 0;
            for (size_t kr = 0; kr < KH; ++kr) {
                for (size_t kc = 0; kc < KW; ++kc) {
                    acc += image[(r + kr) * W + (c + kc)] * kernel[kr * KW + kc];
                }
            }
            ref_rot[r * W + c] = acc;
        }
    }

    Plaintext image_pt = encode_image_coeff_plain(image, H, W, N, plain_mod);
    Plaintext kernel_pt = build_conv_kernel_plain(kernel, KH, KW, W, N, plain_mod);

    Ciphertext image_ct;
    encryptor.encrypt(image_pt, image_ct);

    // PMult-based conv: one multiply_plain computes polynomial-domain convolution.
    Ciphertext conv_ct = image_ct;
    evaluator.multiply_plain_inplace(conv_ct, kernel_pt);
    const std::vector<size_t> valid_indices = cheetah_valid_output_indices(H, W, KH, KW, N);
    extract_valid_coeffs_inplace(conv_ct, evaluator, valid_indices);

    Plaintext conv_pt;
    decryptor.decrypt(conv_ct, conv_pt);
    const auto coeff_at = [&](size_t idx) -> uint64_t {
        return (idx < conv_pt.coeff_count()) ? conv_pt[idx] : 0ULL;
    };

    const size_t out_base = cheetah_filter_base_index(KH, KW, W);
    std::vector<int64_t> ref_poly = conv2d_reference_poly_cheetah_valid(image, H, W, kernel, KH, KW, N);

    size_t valid_count = 0;
    size_t valid_mismatches = 0;
    for (size_t i : valid_indices) {
        ++valid_count;
        const int64_t got = decode_plain_coeff_to_signed(coeff_at(i), plain_mod);
        if (got != ref_poly[i]) ++valid_mismatches;
    }

    // Optional sanity signal for non-extracted coefficients (not used for correctness).
    size_t nonvalid_nonzero = 0;
    for (size_t i = 0; i < N; ++i) {
        bool is_valid = false;
        if (i >= out_base) {
            const size_t rel = i - out_base;
            const size_t rr = rel / W;
            const size_t cc = rel % W;
            is_valid = (rr + KH <= H) && (cc + KW <= W);
        }
        if (is_valid) continue;
        const int64_t got = decode_plain_coeff_to_signed(coeff_at(i), plain_mod);
        if (got != 0) ++nonvalid_nonzero;
    }

    std::cout << "[SEAL coefficient-encoding conv via PMult]\n";
    std::cout << "N=" << N << ", HxW=" << H << "x" << W << ", K=" << KH << "x" << KW
              << ", out_base=" << out_base << ", valid_count=" << valid_count
              << ", valid_mismatches=" << valid_mismatches
              << ", nonvalid_nonzero=" << nonvalid_nonzero << "\n";

    std::cout << "sample(valid outputs, first 12)\n";
    size_t printed = 0;
    for (size_t r = 0; r + KH <= H && printed < 12; ++r) {
        for (size_t c = 0; c + KW <= W && printed < 12; ++c) {
            const size_t idx = out_base + r * W + c; // same as valid_indices order
            const int64_t got = decode_plain_coeff_to_signed(coeff_at(idx), plain_mod);
            const int64_t exp = ref_poly[idx];
            std::cout << "  [" << idx << "] expected=" << exp << ", got=" << got << "\n";
            ++printed;
        }
    }

    // ---------------------------
    // _mod2k conv via rotate + CMult + accumulate
    // ---------------------------
    const uint64_t mask = (1ULL << log_q) - 1ULL;
    std::vector<uint64_t> image_mod2k(N, 0);
    for (size_t i = 0; i < H * W; ++i) {
        image_mod2k[i] = static_cast<uint64_t>(image[i]) & mask;
    }
    scale_by_pow2_inplace(image_mod2k, delta_shift, static_cast<int>(log_q));
    Plaintext image_pt_mod2k = vector_to_plaintext_coeff(image_mod2k, N, static_cast<int>(log_q));

    Ciphertext image_ct_mod2k;
    encrypt_zero_nttfree(context, image_ct_mod2k, sk_pt, log_q);
    add_plain_to_ct_inplace_mod2k(image_ct_mod2k, image_pt_mod2k, log_q);

    // ---------------------------
    // Speed benchmark (conv core only)
    // ---------------------------
    uint64_t bench_checksum = 0;

    const auto t0_pmult = Clock::now();
    for (int i = 0; i < iters; ++i) {
        Ciphertext ct_work = image_ct;
        evaluator.multiply_plain_inplace(ct_work, kernel_pt);
        extract_valid_coeffs_inplace(ct_work, evaluator, valid_indices);
        bench_checksum ^= ct_work.data(0)[out_base];
    }
    const auto t1_pmult = Clock::now();
    const double pmult_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_pmult - t0_pmult).count());

    const auto valid_rot_indices = valid_output_indices_rot_cmult_mod2k(H, W, KH, KW, N);
    const auto t0_rot = Clock::now();
    for (int i = 0; i < iters; ++i) {
        Ciphertext ct_work;
        conv2d_rot_cmult_accum_mod2k(image_ct_mod2k, kernel, H, W, KH, KW, log_q, ct_work);
        bench_checksum ^= ct_work.data(0)[0];
    }
    const auto t1_rot = Clock::now();
    const double rot_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_rot - t0_rot).count());

    Ciphertext conv_rot_ct_mod2k;
    conv2d_rot_cmult_accum_mod2k(image_ct_mod2k, kernel, H, W, KH, KW, log_q, conv_rot_ct_mod2k);

    Plaintext conv_rot_scaled_pt_mod2k;
    decrypt_nttfree(context, conv_rot_ct_mod2k, sk_pt, conv_rot_scaled_pt_mod2k, log_q);
    std::vector<int64_t> conv_rot_decoded =
        decode_divide_pow2(conv_rot_scaled_pt_mod2k, delta_shift, static_cast<int>(log_q), N);

    const auto centered_mod2k = [&](size_t idx) -> int64_t { return conv_rot_decoded[idx]; };

    size_t valid_rot_mismatches = 0;
    for (size_t idx : valid_rot_indices) {
        if (centered_mod2k(idx) != ref_rot[idx]) ++valid_rot_mismatches;
    }

    size_t nonvalid_rot_nonzero = 0;
    std::vector<char> is_valid_rot(N, 0);
    for (size_t idx : valid_rot_indices) is_valid_rot[idx] = 1;
    for (size_t i = 0; i < N; ++i) {
        if (is_valid_rot[i]) continue;
        if (centered_mod2k(i) != 0) ++nonvalid_rot_nonzero;
    }

    std::cout << "[_mod2k conv via rotate+CMult+accumulate]\n";
    std::cout << "N=" << N << ", log_q=" << log_q << ", delta_shift=" << delta_shift
              << ", valid_count=" << valid_rot_indices.size()
              << ", valid_mismatches=" << valid_rot_mismatches
              << ", nonvalid_nonzero=" << nonvalid_rot_nonzero << "\n";

    std::cout << "sample(valid outputs, first 12)\n";
    printed = 0;
    for (size_t r = 0; r + KH <= H && printed < 12; ++r) {
        for (size_t c = 0; c + KW <= W && printed < 12; ++c) {
            const size_t idx = r * W + c;
            const int64_t got = centered_mod2k(idx);
            const int64_t exp = ref_rot[idx];
            std::cout << "  [" << idx << "] expected=" << exp << ", got=" << got << "\n";
            ++printed;
        }
    }

    std::cout << "[Speed comparison: conv core only]\n";
    std::cout << "iters=" << iters << "\n";
    std::cout << "  pmult+extract total_us=" << pmult_us << ", avg_us=" << (pmult_us / iters) << "\n";
    std::cout << "  rot+cmult+accum_mod2k total_us=" << rot_us << ", avg_us=" << (rot_us / iters) << "\n";
    std::cout << "  bench_checksum=" << bench_checksum << "\n";

    return 0;
}
