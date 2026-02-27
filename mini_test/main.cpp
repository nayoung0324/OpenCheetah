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
    constexpr size_t Cin = 3;
    constexpr size_t Co = 4;
    constexpr uint64_t log_q = 50;
    constexpr int delta_shift = 20;
    constexpr int iters = 100;

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
    const size_t one_ch = H * W;
    const size_t channels_per_ct = N / one_ch;
    if (channels_per_ct == 0) {
        std::cerr << "N is too small for one input channel\n";
        return 3;
    }
    const size_t n_packed_ct = (Cin + channels_per_ct - 1) / channels_per_ct;
    const size_t out_base_packed = one_ch * (channels_per_ct - 1) + W * (KH - 1) + (KW - 1);

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
        for (size_t i = 0; i < H * W; ++i) images[ch][i] = dist(rng);
    }

    const std::vector<int64_t> k0 = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
    const std::vector<int64_t> k1 = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
    std::vector<int64_t> kernels_flat_co_cin(Co * Cin * KH * KW, 0);
    for (size_t co = 0; co < Co; ++co) {
        for (size_t ci = 0; ci < Cin; ++ci) {
            const std::vector<int64_t> &base = ((co + ci) % 2 == 0) ? k0 : k1;
            const size_t off = (co * Cin + ci) * KH * KW;
            for (size_t i = 0; i < KH * KW; ++i) kernels_flat_co_cin[off + i] = base[i];
        }
    }

    // References per output channel.
    std::vector<std::vector<int64_t>> ref_poly(Co, std::vector<int64_t>(N, 0));
    std::vector<std::vector<int64_t>> ref_rot(Co, std::vector<int64_t>(N, 0));
    for (size_t co = 0; co < Co; ++co) {
        for (size_t r = 0; r + KH <= H; ++r) {
            for (size_t c = 0; c + KW <= W; ++c) {
                int64_t acc = 0;
                for (size_t ci = 0; ci < Cin; ++ci) {
                    const size_t kbase = (co * Cin + ci) * KH * KW;
                    for (size_t kr = 0; kr < KH; ++kr) {
                        for (size_t kc = 0; kc < KW; ++kc) {
                            acc += images[ci][(r + kr) * W + (c + kc)] * kernels_flat_co_cin[kbase + kr * KW + kc];
                        }
                    }
                }
                ref_poly[co][out_base_packed + r * W + c] = acc;
                ref_rot[co][r * W + c] = acc;
            }
        }
    }

    // Inputs:
    // - PMult/Cheetah path: packed input channels.
    std::vector<Ciphertext> image_cts_seal_packed(n_packed_ct);
    for (size_t g = 0; g < n_packed_ct; ++g) {
        const size_t ch_begin = g * channels_per_ct;
        Plaintext image_pt_packed =
            encode_image_coeff_plain_packed(images, ch_begin, channels_per_ct, H, W, N, plain_mod);
        encryptor.encrypt(image_pt_packed, image_cts_seal_packed[g]);
    }

    // - _mod2k path: one channel per ciphertext.
    std::vector<Ciphertext> image_cts_mod2k(Cin);
    for (size_t ci = 0; ci < Cin; ++ci) {
        std::vector<uint64_t> image_mod2k(N, 0);
        for (size_t i = 0; i < one_ch; ++i) image_mod2k[i] = static_cast<uint64_t>(images[ci][i]) & mask;
        scale_by_pow2_inplace(image_mod2k, delta_shift, static_cast<int>(log_q));
        Plaintext image_pt_mod2k = vector_to_plaintext_coeff(image_mod2k, N, static_cast<int>(log_q));
        encrypt_zero_nttfree(context, image_cts_mod2k[ci], sk_pt, log_q);
        add_plain_to_ct_inplace_mod2k(image_cts_mod2k[ci], image_pt_mod2k, log_q);
    }

    std::vector<size_t> valid_indices_pmult;
    std::vector<size_t> valid_indices_rot;
    valid_indices_pmult.reserve((H - KH + 1) * (W - KW + 1));
    for (size_t r = 0; r + KH <= H; ++r) {
        for (size_t c = 0; c + KW <= W; ++c) {
            valid_indices_pmult.push_back(out_base_packed + r * W + c);
            valid_indices_rot.push_back(r * W + c);
        }
    }

    // ---------------------------
    // PMult: packed input, multi output channels
    // ---------------------------
    std::vector<Ciphertext> out_pmult_cts;
    conv2d_pmult_multi_out_packed(
        image_cts_seal_packed,
        kernels_flat_co_cin,
        Co,
        Cin,
        channels_per_ct,
        H,
        W,
        KH,
        KW,
        plain_mod,
        evaluator,
        out_pmult_cts);

    size_t pmult_total_mismatches = 0;
    for (size_t co = 0; co < Co; ++co) {
        extract_valid_coeffs_inplace(out_pmult_cts[co], evaluator, valid_indices_pmult);
        Plaintext pt;
        decryptor.decrypt(out_pmult_cts[co], pt);
        for (size_t idx : valid_indices_pmult) {
            const uint64_t coeff = (idx < pt.coeff_count()) ? pt[idx] : 0ULL;
            const int64_t got = decode_plain_coeff_to_signed(coeff, plain_mod);
            if (got != ref_poly[co][idx]) ++pmult_total_mismatches;
        }
    }

    std::cout << "[SEAL PMult conv: packed input, multi output]\n";
    std::cout << "Cin=" << Cin << ", Co=" << Co
              << ", channels_per_ct=" << channels_per_ct
              << ", packed_ct=" << n_packed_ct
              << ", valid_per_output=" << valid_indices_pmult.size()
              << ", total_mismatches=" << pmult_total_mismatches << "\n";

    // ---------------------------
    // _mod2k: per-channel input, multi output channels
    // ---------------------------
    std::vector<Ciphertext> out_mod2k_cts;
    conv2d_rot_cmult_multi_out_mod2k(
        image_cts_mod2k, kernels_flat_co_cin, Co, Cin, H, W, KH, KW, log_q, out_mod2k_cts);

    size_t mod2k_total_mismatches = 0;
    for (size_t co = 0; co < Co; ++co) {
        Plaintext scaled_pt;
        decrypt_nttfree(context, out_mod2k_cts[co], sk_pt, scaled_pt, log_q);
        std::vector<int64_t> decoded = decode_divide_pow2(scaled_pt, delta_shift, static_cast<int>(log_q), N);
        for (size_t idx : valid_indices_rot) {
            if (decoded[idx] != ref_rot[co][idx]) ++mod2k_total_mismatches;
        }
    }

    std::cout << "[_mod2k rotate+CMult conv: per-channel input, multi output]\n";
    std::cout << "Cin=" << Cin << ", Co=" << Co
              << ", input_ct=" << Cin
              << ", valid_per_output=" << valid_indices_rot.size()
              << ", total_mismatches=" << mod2k_total_mismatches << "\n";

    // ---------------------------
    // Speed benchmark (conv core only)
    // ---------------------------
    uint64_t bench_checksum = 0;

    const auto t0_pmult = Clock::now();
    for (int i = 0; i < iters; ++i) {
        std::vector<Ciphertext> cts;
        conv2d_pmult_multi_out_packed(
            image_cts_seal_packed,
            kernels_flat_co_cin,
            Co,
            Cin,
            channels_per_ct,
            H,
            W,
            KH,
            KW,
            plain_mod,
            evaluator,
            cts);
        for (size_t co = 0; co < Co; ++co) {
            extract_valid_coeffs_inplace(cts[co], evaluator, valid_indices_pmult);
            bench_checksum ^= cts[co].data(0)[out_base_packed];
        }
    }
    const auto t1_pmult = Clock::now();
    const double pmult_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_pmult - t0_pmult).count());

    const auto t0_mod2k = Clock::now();
    for (int i = 0; i < iters; ++i) {
        std::vector<Ciphertext> cts;
        conv2d_rot_cmult_multi_out_mod2k(
            image_cts_mod2k, kernels_flat_co_cin, Co, Cin, H, W, KH, KW, log_q, cts);
        for (size_t co = 0; co < Co; ++co) {
            bench_checksum ^= cts[co].data(0)[0];
        }
    }
    const auto t1_mod2k = Clock::now();
    const double mod2k_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_mod2k - t0_mod2k).count());

    std::cout << "[Speed comparison: conv core only, multi output]\n";
    std::cout << "iters=" << iters << "\n";
    std::cout << "  pmult(packed-in)+extract total_us=" << pmult_us << ", avg_us=" << (pmult_us / iters) << "\n";
    std::cout << "  mod2k(per-ch-in) total_us=" << mod2k_us << ", avg_us=" << (mod2k_us / iters) << "\n";
    std::cout << "  bench_checksum=" << bench_checksum << "\n";

    return 0;
}
