#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "globals.h"
#include "library_fixed.h"

using namespace std;

int party = 0;
int port = 32000;
string address = "127.0.0.1";
int num_threads = 4;
int32_t bitlength = 37;
int32_t kScale = 12;
int32_t kDoExtractTruncate = 1;

namespace {

using Secret = uint64_t;

struct Tensor4D {
  int n;
  int h;
  int w;
  int c;
  Secret *data;
};

int64_t get_sign_value(uint64_t x) {
  static int64_t upper = 1LL << bitlength;
  static uint64_t mask = static_cast<uint64_t>(upper) - 1;
  static uint64_t half = static_cast<uint64_t>(upper) >> 1;

  x &= mask;
  return static_cast<int64_t>(x >= half ? x - upper : x);
}

int64_t encode_fixed(double value) {
  return static_cast<int64_t>(std::llround(value * (1LL << kScale)));
}

int64_t patterned_value(size_t idx, int owner) {
  if (owner == CLIENT) {
    static const int64_t values[] = {
        0, 1, -1, 2};
    return values[idx % 4];
  }

  static const int64_t values[] = {-1, 0, 1};
  return values[idx % 3];
}

void fill_private_values(Secret *dst, size_t size, int owner, bool is_scale = false,
                         bool is_bias = false) {
  for (size_t i = 0; i < size; ++i) {
    if (party != owner) {
      dst[i] = 0;
      continue;
    }

    int64_t plain = 0;
    if (is_scale) {
      plain = encode_fixed(1.0);
    } else if (is_bias) {
      plain = 0;
    } else {
      plain = patterned_value(i, owner);
    }
    dst[i] = funcSSCons(plain);
  }
}

Tensor4D make_tensor4d(int n, int h, int w, int c) {
  return Tensor4D{n, h, w, c, make_array<Secret>(n, h, w, c)};
}

void free_tensor(Tensor4D &tensor) {
  if (tensor.data != nullptr) {
    ClearMemSecret4(tensor.n, tensor.h, tensor.w, tensor.c, tensor.data);
    tensor.data = nullptr;
  }
}

Tensor4D make_private_input() {
  Tensor4D input = make_tensor4d(1, 224, 224, 3);
  fill_private_values(input.data, static_cast<size_t>(input.n) * input.h * input.w * input.c,
                      CLIENT);
  return input;
}

Secret *make_model_1d(int size, bool is_scale = false, bool is_bias = false) {
  Secret *arr = make_array<Secret>(size);
  fill_private_values(arr, size, SERVER, is_scale, is_bias);
  return arr;
}

Secret *make_model_2d(int d1, int d2) {
  Secret *arr = make_array<Secret>(d1, d2);
  fill_private_values(arr, static_cast<size_t>(d1) * d2, SERVER);
  return arr;
}

Secret *make_model_4d(int d1, int d2, int d3, int d4) {
  Secret *arr = make_array<Secret>(d1, d2, d3, d4);
  fill_private_values(arr, static_cast<size_t>(d1) * d2 * d3 * d4, SERVER);
  return arr;
}

void add_inplace(Tensor4D &dst, const Tensor4D &src) {
  const size_t size = static_cast<size_t>(dst.n) * dst.h * dst.w * dst.c;
  for (size_t i = 0; i < size; ++i) {
    dst.data[i] = SecretAdd(dst.data[i], src.data[i]);
  }
}

Tensor4D relu_tensor(const Tensor4D &input) {
  Tensor4D output = make_tensor4d(input.n, input.h, input.w, input.c);
  Relu(input.n * input.h * input.w * input.c, input.data, output.data, kScale,
       kDoExtractTruncate);
  return output;
}

Tensor4D conv2d(const Tensor4D &input, int out_channels, int kernel, int stride,
                int padding) {
  Tensor4D output = make_tensor4d(
      input.n, ((input.h + 2 * padding - kernel) / stride) + 1,
      ((input.w + 2 * padding - kernel) / stride) + 1, out_channels);
  Secret *weights = make_model_4d(kernel, kernel, input.c, out_channels);

  Conv2DWrapper(input.n, input.h, input.w, input.c, kernel, kernel, out_channels,
                padding, padding, padding, padding, stride, stride, input.data,
                weights, output.data);

  ClearMemSecret4(kernel, kernel, input.c, out_channels, weights);
  return output;
}

Tensor4D batch_norm(const Tensor4D &input) {
  Tensor4D output = make_tensor4d(input.n, input.h, input.w, input.c);
  Secret *scale = make_model_1d(input.c, true, false);
  Secret *bias = make_model_1d(input.c, false, true);

  BatchNorm(input.n, input.h, input.w, input.c, input.data, scale, bias,
            output.data);

  ClearMemSecret1(input.c, scale);
  ClearMemSecret1(input.c, bias);
  return output;
}

Tensor4D max_pool_3x3_stride2(const Tensor4D &input) {
  Tensor4D output = make_tensor4d(input.n, input.h / 2, input.w / 2, input.c);
  MaxPool(input.n, output.h, output.w, input.c, 3, 3, 0, 1, 0, 1, 2, 2,
          input.n, input.h, input.w, input.c, input.data, output.data);
  return output;
}

Tensor4D avg_pool_global(const Tensor4D &input) {
  Tensor4D output = make_tensor4d(input.n, 1, 1, input.c);
  AvgPool(input.n, 1, 1, input.c, input.h, input.w, 0, 0, 0, 0, 1, 1, input.n,
          input.h, input.w, input.c, input.data, output.data);
  return output;
}

Tensor4D basic_block(const Tensor4D &input, int out_channels, int stride) {
  Tensor4D shortcut = make_tensor4d(input.n, input.h, input.w, input.c);
  std::copy(input.data, input.data + static_cast<size_t>(input.n) * input.h * input.w * input.c,
            shortcut.data);

  Tensor4D x = conv2d(input, out_channels, 3, stride, 1);
  Tensor4D x_bn = batch_norm(x);
  free_tensor(x);
  x = relu_tensor(x_bn);
  free_tensor(x_bn);

  Tensor4D y = conv2d(x, out_channels, 3, 1, 1);
  free_tensor(x);
  Tensor4D y_bn = batch_norm(y);
  free_tensor(y);

  if (stride != 1 || input.c != out_channels) {
    free_tensor(shortcut);
    Tensor4D down = conv2d(input, out_channels, 1, stride, 0);
    Tensor4D down_bn = batch_norm(down);
    free_tensor(down);
    shortcut = down_bn;
  }

  add_inplace(y_bn, shortcut);
  free_tensor(shortcut);

  Tensor4D out = relu_tensor(y_bn);
  free_tensor(y_bn);
  return out;
}

Tensor4D run_resnet18_body(const Tensor4D &input) {
#if USE_CHEETAH
  kIsSharedInput = false;
#endif
  Tensor4D x = conv2d(input, 64, 7, 2, 3);
#if USE_CHEETAH
  kIsSharedInput = true;
#endif
  Tensor4D x_bn = batch_norm(x);
  free_tensor(x);
  x = relu_tensor(x_bn);
  free_tensor(x_bn);

  Tensor4D pooled = max_pool_3x3_stride2(x);
  free_tensor(x);
  x = pooled;

  Tensor4D next = basic_block(x, 64, 1);
  free_tensor(x);
  x = next;
  next = basic_block(x, 64, 1);
  free_tensor(x);
  x = next;

  next = basic_block(x, 128, 2);
  free_tensor(x);
  x = next;
  next = basic_block(x, 128, 1);
  free_tensor(x);
  x = next;

  next = basic_block(x, 256, 2);
  free_tensor(x);
  x = next;
  next = basic_block(x, 256, 1);
  free_tensor(x);
  x = next;

  next = basic_block(x, 512, 2);
  free_tensor(x);
  x = next;
  next = basic_block(x, 512, 1);
  free_tensor(x);
  x = next;

  return x;
}

Secret *flatten_nhwc_to_2d(const Tensor4D &input) {
  Secret *flat = make_array<Secret>(input.n, input.c);
  for (int n = 0; n < input.n; ++n) {
    for (int c = 0; c < input.c; ++c) {
      Arr2DIdxRowM(flat, input.n, input.c, n, c) =
          Arr4DIdxRowM(input.data, input.n, input.h, input.w, input.c, n, 0, 0, c);
    }
  }
  return flat;
}

Secret *fc_logits(const Secret *input, int in_dim, int out_dim) {
  Secret *weights = make_model_2d(in_dim, out_dim);
  Secret *bias = make_model_1d(out_dim, false, true);
  Secret *logits = make_array<Secret>(1, out_dim);

  MatMul2D(1, in_dim, out_dim, input, weights, logits, false);
  ScaleUp(out_dim, bias, kScale);
  for (int j = 0; j < out_dim; ++j) {
    Arr2DIdxRowM(logits, 1, out_dim, 0, j) =
        SecretAdd(Arr2DIdxRowM(logits, 1, out_dim, 0, j), bias[j]);
  }

  ClearMemSecret2(in_dim, out_dim, weights);
  ClearMemSecret1(out_dim, bias);
  return logits;
}

}  // namespace

int main(int argc, char **argv) {
  ArgMapping amap;

  amap.arg("r", party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("nt", num_threads, "Number of Threads");
  amap.arg("ell", bitlength, "Uniform Bitwidth");
  amap.arg("k", kScale, "Scaling factor");
  amap.arg("dt", kDoExtractTruncate, "Whether ReLU performs truncation");
  amap.parse(argc, argv);

  assert(party == SERVER || party == CLIENT);

  cerr << "Generating dummy input/model for ResNet18 benchmark..." << endl;
  Tensor4D input = make_private_input();

  StartComputation();
  Tensor4D stem_out = run_resnet18_body(input);
  free_tensor(input);

  Tensor4D pooled = avg_pool_global(stem_out);
  free_tensor(stem_out);

  Secret *flat = flatten_nhwc_to_2d(pooled);
  free_tensor(pooled);

  constexpr int kNumClasses = 1001;
  Secret *logits = fc_logits(flat, 512, kNumClasses);
  ClearMemSecret2(1, 512, flat);

  Secret *pred = make_array<Secret>(1);
  ArgMax(1, kNumClasses, logits, pred);
  EndComputation();

  if (party == CLIENT) {
    auto label = funcReconstruct2PCCons(pred[0], CLIENT);
    cout << "dummy ResNet18 predicted label=" << label << endl;

    vector<pair<double, int>> topk;
    topk.reserve(kNumClasses);
    for (int i = 0; i < kNumClasses; ++i) {
      double value = static_cast<double>(
                         get_sign_value(Arr2DIdxRowM(logits, 1, kNumClasses, 0, i))) /
                     std::pow(2.0, kScale);
      topk.push_back({value, i});
    }
    partial_sort(topk.begin(), topk.begin() + 5, topk.end(),
                 [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
    cout << "top-5 dummy logits: ";
    for (int i = 0; i < 5; ++i) {
      cout << "(" << topk[i].second << ", " << topk[i].first << ")";
      if (i + 1 != 5) {
        cout << " ";
      }
    }
    cout << endl;
  }

  ClearMemSecret2(1, kNumClasses, logits);
  ClearMemSecret1(1, pred);
  finalize();
  return 0;
}
