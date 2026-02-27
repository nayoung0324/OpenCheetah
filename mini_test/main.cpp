#include <seal/seal.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "conv.h"
#include "encryption.h"
#include "util.h"

using namespace seal;
using Clock = std::chrono::high_resolution_clock;

struct BenchCase
{
    size_t hw;
    size_t cin;
    size_t cout;
    int iters;
};

int main()
{
    constexpr size_t N = 2048;
    constexpr size_t KH = 3;
    constexpr size_t KW = 3;
    constexpr uint64_t log_q = 50;
    constexpr int delta_shift = 20;

    const std::vector<BenchCase> cases = {
        { 32, 16, 16, 40 },
        { 16, 64, 64, 16 },
        { 4, 128, 128, 8 },
    };

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

    uint64_t global_checksum = 0;
    std::mt19937_64 rng(20260227ULL);
    std::uniform_int_distribution<int> xdist(-2, 2);

    std::cout << "[Latency compare: Cheetah PMult vs _mod2k rotate+CMult]\n";
    std::cout << "N=" << N << ", K=" << KH << "x" << KW << ", log_q=" << log_q << "\n\n";

    for (const auto &bc : cases) {
        const size_t H = bc.hw;
        const size_t W = bc.hw;
        const size_t Cin = bc.cin;
        const size_t Co = bc.cout;
        const int iters = bc.iters;
        const size_t one_ch = H * W;

        if (one_ch > N) {
            std::cout << "Case(H=W=" << H << ", Cin=" << Cin << ", Cout=" << Co
                      << "): skipped (H*W > N)\n";
            continue;
        }

        const size_t channels_per_ct = N / one_ch;
        if (channels_per_ct == 0) {
            std::cout << "Case(H=W=" << H << ", Cin=" << Cin << ", Cout=" << Co
                      << "): skipped (channels_per_ct=0)\n";
            continue;
        }
        const size_t n_packed_ct = (Cin + channels_per_ct - 1) / channels_per_ct;
        const size_t out_base_packed = one_ch * (channels_per_ct - 1) + W * (KH - 1) + (KW - 1);

        // Input images.
        std::vector<std::vector<int64_t>> images(Cin, std::vector<int64_t>(one_ch, 0));
        for (size_t ci = 0; ci < Cin; ++ci) {
            for (size_t i = 0; i < one_ch; ++i) images[ci][i] = xdist(rng);
        }

        // Kernels [co][ci][kh][kw].
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

        // PMult path input: packed.
        std::vector<Ciphertext> image_cts_seal_packed(n_packed_ct);
        for (size_t g = 0; g < n_packed_ct; ++g) {
            const size_t ch_begin = g * channels_per_ct;
            Plaintext image_pt_packed =
                encode_image_coeff_plain_packed(images, ch_begin, channels_per_ct, H, W, N, plain_mod);
            encryptor.encrypt(image_pt_packed, image_cts_seal_packed[g]);
        }

        // _mod2k path input: one ct per channel.
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
        valid_indices_pmult.reserve((H - KH + 1) * (W - KW + 1));
        for (size_t r = 0; r + KH <= H; ++r) {
            for (size_t c = 0; c + KW <= W; ++c) {
                valid_indices_pmult.push_back(out_base_packed + r * W + c);
            }
        }

        // PMult benchmark (include extract like cheetah cleanup).
        const auto t0_pmult = Clock::now();
        for (int i = 0; i < iters; ++i) {
            std::vector<Ciphertext> outs;
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
                outs);
            for (size_t co = 0; co < Co; ++co) {
                extract_valid_coeffs_inplace(outs[co], evaluator, valid_indices_pmult);
                global_checksum ^= outs[co].data(0)[out_base_packed];
            }
        }
        const auto t1_pmult = Clock::now();
        const double pmult_us =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_pmult - t0_pmult).count());

        // _mod2k benchmark (conv core only).
        const auto t0_mod2k = Clock::now();
        for (int i = 0; i < iters; ++i) {
            std::vector<Ciphertext> outs;
            conv2d_rot_cmult_multi_out_mod2k(
                image_cts_mod2k, kernels_flat_co_cin, Co, Cin, H, W, KH, KW, log_q, outs);
            for (size_t co = 0; co < Co; ++co) {
                global_checksum ^= outs[co].data(0)[0];
            }
        }
        const auto t1_mod2k = Clock::now();
        const double mod2k_us =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_mod2k - t0_mod2k).count());

        std::cout << "Case(H=W=" << H << ", Cin=" << Cin << ", Cout=" << Co << ")\n";
        std::cout << "  channels_per_ct=" << channels_per_ct << ", packed_ct=" << n_packed_ct
                  << ", iters=" << iters << "\n";
        std::cout << "  cheetah_pmult total_us=" << pmult_us << ", avg_us=" << (pmult_us / iters) << "\n";
        std::cout << "  mod2k_rot_cmult total_us=" << mod2k_us << ", avg_us=" << (mod2k_us / iters) << "\n";
        std::cout << "  speedup(pmult/mod2k)=" << (pmult_us / mod2k_us) << "\n\n";
    }

    std::cout << "checksum=" << global_checksum << "\n";
    return 0;
}
