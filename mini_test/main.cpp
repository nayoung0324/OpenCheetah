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
    constexpr size_t PH = (KH - 1) / 2;
    constexpr size_t PW = (KW - 1) / 2;
    constexpr uint64_t log_q_cheetah = 54;
    constexpr uint64_t log_q_mod2k = 60;
    constexpr int delta_shift = 20;

    const std::vector<BenchCase> cases = {
        { 224, 64, 64, 2 },
        { 64, 64, 64, 3 },
        { 56, 256, 256, 1 },
        { 32, 16, 16, 20 },
        { 32, 128, 128, 2 },
        { 28, 512, 512, 1 },
        { 16, 64, 64, 10 },
        { 16, 256, 256, 2 },
        // Skip (14,1024,256) because fw=1 in the provided table.
        { 8, 128, 128, 6 },
        { 4, 512, 512, 3 },
    };

    EncryptionParameters parms_cheetah(scheme_type::bfv);
    parms_cheetah.set_poly_modulus_degree(N);
    parms_cheetah.set_coeff_modulus(CoeffModulus::Create(N, { static_cast<int>(log_q_cheetah) }));
    parms_cheetah.set_plain_modulus(PlainModulus::Batching(N, 19));
    SEALContext context_cheetah(parms_cheetah);
    if (!context_cheetah.parameters_set()) {
        std::cerr << "SEAL cheetah-context parameter error: " << context_cheetah.parameter_error_name()
                  << " - " << context_cheetah.parameter_error_message() << "\n";
        return 2;
    }

    EncryptionParameters parms_mod2k(scheme_type::bfv);
    parms_mod2k.set_poly_modulus_degree(N);
    parms_mod2k.set_coeff_modulus(CoeffModulus::Create(N, { static_cast<int>(log_q_mod2k) }));
    parms_mod2k.set_plain_modulus(PlainModulus::Batching(N, 19));
    SEALContext context_mod2k(parms_mod2k);
    if (!context_mod2k.parameters_set()) {
        std::cerr << "SEAL mod2k-context parameter error: " << context_mod2k.parameter_error_name()
                  << " - " << context_mod2k.parameter_error_message() << "\n";
        return 2;
    }

    const uint64_t plain_mod_cheetah = parms_cheetah.plain_modulus().value();
    const uint64_t mask_mod2k = (1ULL << log_q_mod2k) - 1ULL;

    KeyGenerator keygen_cheetah(context_cheetah);
    SecretKey sk_cheetah = keygen_cheetah.secret_key();
    PublicKey pk_cheetah;
    keygen_cheetah.create_public_key(pk_cheetah);
    Encryptor encryptor_cheetah(context_cheetah, pk_cheetah);
    Evaluator evaluator_cheetah(context_cheetah);
    Decryptor decryptor_cheetah(context_cheetah, sk_cheetah);

    KeyGenerator keygen_mod2k(context_mod2k);
    SecretKey sk_mod2k = keygen_mod2k.secret_key();
    const Plaintext &sk_pt_mod2k = sk_mod2k.data();

    uint64_t global_checksum = 0;
    std::mt19937_64 rng(20260227ULL);
    std::uniform_int_distribution<int> xdist(-2, 2);

    std::cout << "[Latency compare: Cheetah PMult vs _mod2k rotate+CMult]\n";
    std::cout << "N=" << N << ", K=" << KH << "x" << KW
              << ", q_cheetah=" << log_q_cheetah << ", q_mod2k=" << log_q_mod2k << "\n\n";

    for (const auto &bc : cases) {
        const size_t H = bc.hw;
        const size_t W = bc.hw;
        const size_t Hp = H + 2 * PH;
        const size_t Wp = W + 2 * PW;
        const size_t Cin = bc.cin;
        const size_t Co = bc.cout;
        const int iters = bc.iters;
        const size_t one_ch = Hp * Wp;
        const size_t out_h = H;
        const size_t out_w = W;

        // Input images (unpadded).
        std::vector<std::vector<int64_t>> images(Cin, std::vector<int64_t>(H * W, 0));
        for (size_t ci = 0; ci < Cin; ++ci) {
            for (size_t i = 0; i < H * W; ++i) images[ci][i] = xdist(rng);
        }

        // Zero-padded inputs for SAME output.
        std::vector<std::vector<int64_t>> images_padded(Cin, std::vector<int64_t>(Hp * Wp, 0));
        for (size_t ci = 0; ci < Cin; ++ci) {
            for (size_t r = 0; r < H; ++r) {
                for (size_t c = 0; c < W; ++c) {
                    images_padded[ci][(r + PH) * Wp + (c + PW)] = images[ci][r * W + c];
                }
            }
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

        // Reference full outputs per output channel.
        std::vector<std::vector<int64_t>> ref_out(Co, std::vector<int64_t>(out_h * out_w, 0));
        for (size_t co = 0; co < Co; ++co) {
            for (size_t r = 0; r < out_h; ++r) {
                for (size_t c = 0; c < out_w; ++c) {
                    int64_t acc = 0;
                    for (size_t ci = 0; ci < Cin; ++ci) {
                        const size_t kbase = (co * Cin + ci) * KH * KW;
                        for (size_t kr = 0; kr < KH; ++kr) {
                            for (size_t kc = 0; kc < KW; ++kc) {
                                acc += images_padded[ci][(r + kr) * Wp + (c + kc)] * kernels_flat_co_cin[kbase + kr * KW + kc];
                            }
                        }
                    }
                    ref_out[co][r * out_w + c] = acc;
                }
            }
        }

        struct TilePrepared
        {
            Conv2DTile tile;
            size_t tile_in_h;
            size_t tile_in_w;
            size_t channels_per_ct;
            size_t out_base_packed;
            std::vector<size_t> valid_indices_pmult;
            std::vector<Ciphertext> image_cts_seal_packed;
            std::vector<Ciphertext> image_cts_mod2k;
        };

        std::vector<TilePrepared> prepared_tiles;
        if (one_ch <= N) {
            TilePrepared tp;
            tp.tile = { 0, 0, out_h, out_w, Hp, Wp };
            tp.tile_in_h = Hp;
            tp.tile_in_w = Wp;
            tp.channels_per_ct = N / one_ch;
            tp.out_base_packed = one_ch * (tp.channels_per_ct - 1) + Wp * (KH - 1) + (KW - 1);

            const size_t n_packed_ct = (Cin + tp.channels_per_ct - 1) / tp.channels_per_ct;
            tp.image_cts_seal_packed.resize(n_packed_ct);
            for (size_t g = 0; g < n_packed_ct; ++g) {
                const size_t ch_begin = g * tp.channels_per_ct;
                Plaintext image_pt_packed = encode_image_coeff_plain_packed(
                    images_padded, ch_begin, tp.channels_per_ct, Hp, Wp, N, plain_mod_cheetah);
                encryptor_cheetah.encrypt(image_pt_packed, tp.image_cts_seal_packed[g]);
            }

            tp.image_cts_mod2k.resize(Cin);
            for (size_t ci = 0; ci < Cin; ++ci) {
                std::vector<uint64_t> image_mod2k(N, 0);
                for (size_t i = 0; i < one_ch; ++i) image_mod2k[i] = static_cast<uint64_t>(images_padded[ci][i]) & mask_mod2k;
                scale_by_pow2_inplace(image_mod2k, delta_shift, static_cast<int>(log_q_mod2k));
                Plaintext image_pt_mod2k = vector_to_plaintext_coeff(image_mod2k, N, static_cast<int>(log_q_mod2k));
                encrypt_zero_nttfree(context_mod2k, tp.image_cts_mod2k[ci], sk_pt_mod2k, log_q_mod2k);
                add_plain_to_ct_inplace_mod2k(tp.image_cts_mod2k[ci], image_pt_mod2k, log_q_mod2k);
            }

            tp.valid_indices_pmult.reserve(H * W);
            for (size_t r = 0; r < H; ++r) {
                for (size_t c = 0; c < W; ++c) {
                    tp.valid_indices_pmult.push_back(tp.out_base_packed + r * Wp + c);
                }
            }
            prepared_tiles.push_back(std::move(tp));
        } else {
            const auto tiles = make_conv2d_tiles_valid(Hp, Wp, KH, KW, N);
            for (const auto &tile : tiles) {
                TilePrepared tp;
                tp.tile = tile;
                tp.tile_in_h = tile.in_h;
                tp.tile_in_w = tile.in_w;
                const size_t tile_one_ch = tile.in_h * tile.in_w;
                tp.channels_per_ct = N / tile_one_ch;
                tp.out_base_packed = tile_one_ch * (tp.channels_per_ct - 1) + tile.in_w * (KH - 1) + (KW - 1);

                std::vector<std::vector<int64_t>> tile_images(Cin, std::vector<int64_t>(tile_one_ch, 0));
                for (size_t ci = 0; ci < Cin; ++ci) {
                    tile_images[ci] = extract_input_patch_by_tile(images_padded[ci], Hp, Wp, tile);
                }

                const size_t n_packed_ct = (Cin + tp.channels_per_ct - 1) / tp.channels_per_ct;
                tp.image_cts_seal_packed.resize(n_packed_ct);
                for (size_t g = 0; g < n_packed_ct; ++g) {
                    const size_t ch_begin = g * tp.channels_per_ct;
                    Plaintext image_pt_packed = encode_image_coeff_plain_packed(
                        tile_images, ch_begin, tp.channels_per_ct, tile.in_h, tile.in_w, N, plain_mod_cheetah);
                    encryptor_cheetah.encrypt(image_pt_packed, tp.image_cts_seal_packed[g]);
                }

                tp.image_cts_mod2k.resize(Cin);
                for (size_t ci = 0; ci < Cin; ++ci) {
                    std::vector<uint64_t> image_mod2k(N, 0);
                    for (size_t i = 0; i < tile_one_ch; ++i) {
                        image_mod2k[i] = static_cast<uint64_t>(tile_images[ci][i]) & mask_mod2k;
                    }
                    scale_by_pow2_inplace(image_mod2k, delta_shift, static_cast<int>(log_q_mod2k));
                    Plaintext image_pt_mod2k = vector_to_plaintext_coeff(image_mod2k, N, static_cast<int>(log_q_mod2k));
                    encrypt_zero_nttfree(context_mod2k, tp.image_cts_mod2k[ci], sk_pt_mod2k, log_q_mod2k);
                    add_plain_to_ct_inplace_mod2k(tp.image_cts_mod2k[ci], image_pt_mod2k, log_q_mod2k);
                }

                tp.valid_indices_pmult.reserve(tile.out_h * tile.out_w);
                for (size_t r = 0; r < tile.out_h; ++r) {
                    for (size_t c = 0; c < tile.out_w; ++c) {
                        tp.valid_indices_pmult.push_back(tp.out_base_packed + r * tile.in_w + c);
                    }
                }
                prepared_tiles.push_back(std::move(tp));
            }
        }

        // Correctness check (single run): assemble tile outputs and compare with reference.
        std::vector<std::vector<int64_t>> got_pmult(Co, std::vector<int64_t>(out_h * out_w, 0));
        std::vector<std::vector<int64_t>> got_mod2k(Co, std::vector<int64_t>(out_h * out_w, 0));

        for (const auto &tp : prepared_tiles) {
            std::vector<Ciphertext> outs_pmult;
            conv2d_pmult_multi_out_packed(
                tp.image_cts_seal_packed,
                kernels_flat_co_cin,
                Co,
                Cin,
                tp.channels_per_ct,
                tp.tile_in_h,
                tp.tile_in_w,
                KH,
                KW,
                plain_mod_cheetah,
                evaluator_cheetah,
                outs_pmult);
            for (size_t co = 0; co < Co; ++co) {
                extract_valid_coeffs_inplace(outs_pmult[co], evaluator_cheetah, tp.valid_indices_pmult);
                Plaintext pt;
                decryptor_cheetah.decrypt(outs_pmult[co], pt);
                std::vector<int64_t> patch(tp.tile.out_h * tp.tile.out_w, 0);
                for (size_t r = 0; r < tp.tile.out_h; ++r) {
                    for (size_t c = 0; c < tp.tile.out_w; ++c) {
                        const size_t idx = tp.out_base_packed + r * tp.tile_in_w + c;
                        const uint64_t coeff = (idx < pt.coeff_count()) ? pt[idx] : 0ULL;
                        patch[r * tp.tile.out_w + c] = decode_plain_coeff_to_signed(coeff, plain_mod_cheetah);
                    }
                }
                scatter_output_patch_by_tile(patch, tp.tile, out_w, got_pmult[co]);
            }

            std::vector<Ciphertext> outs_mod2k;
            conv2d_rot_cmult_multi_out_mod2k(
                tp.image_cts_mod2k, kernels_flat_co_cin, Co, Cin, tp.tile_in_h, tp.tile_in_w, KH, KW, log_q_mod2k, outs_mod2k);
            for (size_t co = 0; co < Co; ++co) {
                Plaintext scaled;
                decrypt_nttfree(context_mod2k, outs_mod2k[co], sk_pt_mod2k, scaled, log_q_mod2k);
                std::vector<int64_t> decoded =
                    decode_divide_pow2(scaled, delta_shift, static_cast<int>(log_q_mod2k), N);
                std::vector<int64_t> patch(tp.tile.out_h * tp.tile.out_w, 0);
                for (size_t r = 0; r < tp.tile.out_h; ++r) {
                    for (size_t c = 0; c < tp.tile.out_w; ++c) {
                        const size_t idx = r * tp.tile_in_w + c;
                        patch[r * tp.tile.out_w + c] = decoded[idx];
                    }
                }
                scatter_output_patch_by_tile(patch, tp.tile, out_w, got_mod2k[co]);
            }
        }

        size_t pmult_mismatches = 0;
        size_t mod2k_mismatches = 0;
        for (size_t co = 0; co < Co; ++co) {
            for (size_t i = 0; i < out_h * out_w; ++i) {
                if (got_pmult[co][i] != ref_out[co][i]) ++pmult_mismatches;
                if (got_mod2k[co][i] != ref_out[co][i]) ++mod2k_mismatches;
            }
        }

        const auto t0_pmult = Clock::now();
        for (int i = 0; i < iters; ++i) {
            for (const auto &tp : prepared_tiles) {
                std::vector<Ciphertext> outs;
                conv2d_pmult_multi_out_packed(
                    tp.image_cts_seal_packed,
                    kernels_flat_co_cin,
                    Co,
                    Cin,
                    tp.channels_per_ct,
                    tp.tile_in_h,
                    tp.tile_in_w,
                    KH,
                    KW,
                    plain_mod_cheetah,
                    evaluator_cheetah,
                    outs);
                for (size_t co = 0; co < Co; ++co) {
                    global_checksum ^= outs[co].data(0)[tp.out_base_packed];
                }
            }
        }
        const auto t1_pmult = Clock::now();
        const double pmult_ms =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_pmult - t0_pmult).count()) /
            1000.0;

        const auto t0_mod2k = Clock::now();
        for (int i = 0; i < iters; ++i) {
            for (const auto &tp : prepared_tiles) {
                std::vector<Ciphertext> outs;
                conv2d_rot_cmult_multi_out_mod2k(
                    tp.image_cts_mod2k, kernels_flat_co_cin, Co, Cin, tp.tile_in_h, tp.tile_in_w, KH, KW, log_q_mod2k, outs);
                for (size_t co = 0; co < Co; ++co) {
                    global_checksum ^= outs[co].data(0)[0];
                }
            }
        }
        const auto t1_mod2k = Clock::now();
        const double mod2k_ms =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_mod2k - t0_mod2k).count()) /
            1000.0;

        std::cout << "Case(H=W=" << H << ", Cin=" << Cin << ", Cout=" << Co << ")\n";
        if (one_ch <= N) {
            const size_t channels_per_ct = N / one_ch;
            const size_t n_packed_ct = (Cin + channels_per_ct - 1) / channels_per_ct;
            std::cout << "  mode=single-ct-image, channels_per_ct=" << channels_per_ct
                      << ", packed_ct=" << n_packed_ct << ", iters=" << iters << "\n";
        } else {
            std::cout << "  mode=tiled(H*W>N), tiles=" << prepared_tiles.size() << ", iters=" << iters << "\n";
        }
        std::cout << "  padding=zero(SAME), input_padded=" << Hp << "x" << Wp
                  << ", output=" << out_h << "x" << out_w << "\n";
        std::cout << "  cheetah_pmult total_ms=" << pmult_ms << ", avg_ms=" << (pmult_ms / iters) << "\n";
        std::cout << "  mod2k_rot_cmult total_ms=" << mod2k_ms << ", avg_ms=" << (mod2k_ms / iters) << "\n";
        std::cout << "  correctness: pmult_mismatches=" << pmult_mismatches
                  << ", mod2k_mismatches=" << mod2k_mismatches << "\n";
        std::cout << "  speedup(pmult/mod2k)=" << (pmult_ms / mod2k_ms) << "\n\n";
    }

    std::cout << "checksum=" << global_checksum << "\n";
    return 0;
}
