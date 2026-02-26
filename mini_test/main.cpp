#include <seal/seal.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "conv.h"

using namespace seal;

int main()
{
    constexpr size_t N = 2048;
    constexpr size_t H = 32;
    constexpr size_t W = 32;
    constexpr size_t KH = 3;
    constexpr size_t KW = 3;

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

    return 0;
}
