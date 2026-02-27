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
    constexpr uint64_t log_q = 50;
    constexpr int delta_shift = 20;

    const std::vector<BenchCase> cases = {
        { 32, 16, 16, 40 },
        { 16, 64, 64, 16 },
        { 4, 128, 128, 8 },
        { 64, 8, 8, 10 },
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
    Decryptor decryptor(context, sk);

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
        const size_t out_h = H - KH + 1;
        const size_t out_w = W - KW + 1;

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
                                acc += images[ci][(r + kr) * W + (c + kc)] * kernels_flat_co_cin[kbase + kr * KW + kc];
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
            tp.tile = { 0, 0, out_h, out_w, H, W };
            tp.tile_in_h = H;
            tp.tile_in_w = W;
            tp.channels_per_ct = N / one_ch;
            tp.out_base_packed = one_ch * (tp.channels_per_ct - 1) + W * (KH - 1) + (KW - 1);

            const size_t n_packed_ct = (Cin + tp.channels_per_ct - 1) / tp.channels_per_ct;
            tp.image_cts_seal_packed.resize(n_packed_ct);
            for (size_t g = 0; g < n_packed_ct; ++g) {
                const size_t ch_begin = g * tp.channels_per_ct;
                Plaintext image_pt_packed =
                    encode_image_coeff_plain_packed(images, ch_begin, tp.channels_per_ct, H, W, N, plain_mod);
                encryptor.encrypt(image_pt_packed, tp.image_cts_seal_packed[g]);
            }

            tp.image_cts_mod2k.resize(Cin);
            for (size_t ci = 0; ci < Cin; ++ci) {
                std::vector<uint64_t> image_mod2k(N, 0);
                for (size_t i = 0; i < one_ch; ++i) image_mod2k[i] = static_cast<uint64_t>(images[ci][i]) & mask;
                scale_by_pow2_inplace(image_mod2k, delta_shift, static_cast<int>(log_q));
                Plaintext image_pt_mod2k = vector_to_plaintext_coeff(image_mod2k, N, static_cast<int>(log_q));
                encrypt_zero_nttfree(context, tp.image_cts_mod2k[ci], sk_pt, log_q);
                add_plain_to_ct_inplace_mod2k(tp.image_cts_mod2k[ci], image_pt_mod2k, log_q);
            }

            tp.valid_indices_pmult.reserve((H - KH + 1) * (W - KW + 1));
            for (size_t r = 0; r + KH <= H; ++r) {
                for (size_t c = 0; c + KW <= W; ++c) {
                    tp.valid_indices_pmult.push_back(tp.out_base_packed + r * W + c);
                }
            }
            prepared_tiles.push_back(std::move(tp));
        } else {
            const auto tiles = make_conv2d_tiles_valid(H, W, KH, KW, N);
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
                    tile_images[ci] = extract_input_patch_by_tile(images[ci], H, W, tile);
                }

                const size_t n_packed_ct = (Cin + tp.channels_per_ct - 1) / tp.channels_per_ct;
                tp.image_cts_seal_packed.resize(n_packed_ct);
                for (size_t g = 0; g < n_packed_ct; ++g) {
                    const size_t ch_begin = g * tp.channels_per_ct;
                    Plaintext image_pt_packed = encode_image_coeff_plain_packed(
                        tile_images, ch_begin, tp.channels_per_ct, tile.in_h, tile.in_w, N, plain_mod);
                    encryptor.encrypt(image_pt_packed, tp.image_cts_seal_packed[g]);
                }

                tp.image_cts_mod2k.resize(Cin);
                for (size_t ci = 0; ci < Cin; ++ci) {
                    std::vector<uint64_t> image_mod2k(N, 0);
                    for (size_t i = 0; i < tile_one_ch; ++i) {
                        image_mod2k[i] = static_cast<uint64_t>(tile_images[ci][i]) & mask;
                    }
                    scale_by_pow2_inplace(image_mod2k, delta_shift, static_cast<int>(log_q));
                    Plaintext image_pt_mod2k = vector_to_plaintext_coeff(image_mod2k, N, static_cast<int>(log_q));
                    encrypt_zero_nttfree(context, tp.image_cts_mod2k[ci], sk_pt, log_q);
                    add_plain_to_ct_inplace_mod2k(tp.image_cts_mod2k[ci], image_pt_mod2k, log_q);
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
                plain_mod,
                evaluator,
                outs_pmult);
            for (size_t co = 0; co < Co; ++co) {
                extract_valid_coeffs_inplace(outs_pmult[co], evaluator, tp.valid_indices_pmult);
                Plaintext pt;
                decryptor.decrypt(outs_pmult[co], pt);
                std::vector<int64_t> patch(tp.tile.out_h * tp.tile.out_w, 0);
                for (size_t r = 0; r < tp.tile.out_h; ++r) {
                    for (size_t c = 0; c < tp.tile.out_w; ++c) {
                        const size_t idx = tp.out_base_packed + r * tp.tile_in_w + c;
                        const uint64_t coeff = (idx < pt.coeff_count()) ? pt[idx] : 0ULL;
                        patch[r * tp.tile.out_w + c] = decode_plain_coeff_to_signed(coeff, plain_mod);
                    }
                }
                scatter_output_patch_by_tile(patch, tp.tile, out_w, got_pmult[co]);
            }

            std::vector<Ciphertext> outs_mod2k;
            conv2d_rot_cmult_multi_out_mod2k(
                tp.image_cts_mod2k, kernels_flat_co_cin, Co, Cin, tp.tile_in_h, tp.tile_in_w, KH, KW, log_q, outs_mod2k);
            for (size_t co = 0; co < Co; ++co) {
                Plaintext scaled;
                decrypt_nttfree(context, outs_mod2k[co], sk_pt, scaled, log_q);
                std::vector<int64_t> decoded = decode_divide_pow2(scaled, delta_shift, static_cast<int>(log_q), N);
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

        // PMult benchmark (include extract like cheetah cleanup).
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
                    plain_mod,
                    evaluator,
                    outs);
                for (size_t co = 0; co < Co; ++co) {
                    global_checksum ^= outs[co].data(0)[tp.out_base_packed];
                }
            }
        }
        const auto t1_pmult = Clock::now();
        const double pmult_us =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_pmult - t0_pmult).count());

        // _mod2k benchmark (conv core only).
        const auto t0_mod2k = Clock::now();
        for (int i = 0; i < iters; ++i) {
            for (const auto &tp : prepared_tiles) {
                std::vector<Ciphertext> outs;
                conv2d_rot_cmult_multi_out_mod2k(
                    tp.image_cts_mod2k,
                    kernels_flat_co_cin,
                    Co,
                    Cin,
                    tp.tile_in_h,
                    tp.tile_in_w,
                    KH,
                    KW,
                    log_q,
                    outs);
                for (size_t co = 0; co < Co; ++co) {
                    global_checksum ^= outs[co].data(0)[0];
                }
            }
        }
        const auto t1_mod2k = Clock::now();
        const double mod2k_us =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1_mod2k - t0_mod2k).count());

        std::cout << "Case(H=W=" << H << ", Cin=" << Cin << ", Cout=" << Co << ")\n";
        if (one_ch <= N) {
            const size_t channels_per_ct = N / one_ch;
            const size_t n_packed_ct = (Cin + channels_per_ct - 1) / channels_per_ct;
            std::cout << "  mode=single-ct-image, channels_per_ct=" << channels_per_ct
                      << ", packed_ct=" << n_packed_ct << ", iters=" << iters << "\n";
        } else {
            std::cout << "  mode=tiled(H*W>N), tiles=" << prepared_tiles.size() << ", iters=" << iters << "\n";
        }
        std::cout << "  cheetah_pmult total_us=" << pmult_us << ", avg_us=" << (pmult_us / iters) << "\n";
        std::cout << "  mod2k_rot_cmult total_us=" << mod2k_us << ", avg_us=" << (mod2k_us / iters) << "\n";
        std::cout << "  correctness: pmult_mismatches=" << pmult_mismatches
                  << ", mod2k_mismatches=" << mod2k_mismatches << "\n";
        std::cout << "  speedup(pmult/mod2k)=" << (pmult_us / mod2k_us) << "\n\n";
    }

    std::cout << "checksum=" << global_checksum << "\n";
    return 0;
}
