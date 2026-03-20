#!/usr/bin/env python3

from pathlib import Path


SCALE = 1 << 12


def write_values(path: Path, values):
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")


def cycle_values(pattern, count):
    plen = len(pattern)
    for i in range(count):
        yield pattern[i % plen]


def tensor_size(*dims):
    size = 1
    for d in dims:
        size *= d
    return size


def emit_conv(values, kh, kw, cin, cout):
    values.extend(cycle_values([-SCALE, 0, SCALE], tensor_size(kh, kw, cin, cout)))


def emit_bn(values, channels):
    values.extend(cycle_values([SCALE], channels))  # scale = 1.0
    values.extend(cycle_values([0], channels))      # bias = 0


def emit_basic_block(values, in_channels, out_channels, stride):
    emit_conv(values, 3, 3, in_channels, out_channels)
    emit_bn(values, out_channels)
    emit_conv(values, 3, 3, out_channels, out_channels)
    emit_bn(values, out_channels)
    if stride != 1 or in_channels != out_channels:
        emit_conv(values, 1, 1, in_channels, out_channels)
        emit_bn(values, out_channels)


def build_input_values():
    return list(cycle_values([0, SCALE, -SCALE, 2 * SCALE], tensor_size(1, 224, 224, 3)))


def build_model_values():
    values = []

    emit_conv(values, 7, 7, 3, 64)
    emit_bn(values, 64)

    emit_basic_block(values, 64, 64, 1)
    emit_basic_block(values, 64, 64, 1)

    emit_basic_block(values, 64, 128, 2)
    emit_basic_block(values, 128, 128, 1)

    emit_basic_block(values, 128, 256, 2)
    emit_basic_block(values, 256, 256, 1)

    emit_basic_block(values, 256, 512, 2)
    emit_basic_block(values, 512, 512, 1)

    values.extend(cycle_values([-SCALE, 0, SCALE], tensor_size(512, 1001)))
    values.extend(cycle_values([0], 1001))
    return values


def main():
    repo_root = Path(__file__).resolve().parent.parent
    pretrained = repo_root / "pretrained"
    pretrained.mkdir(exist_ok=True)

    input_path = pretrained / "resnet18_input_scale12_pred0.inp"
    model_path = pretrained / "resnet18_model_scale12.inp"

    write_values(input_path, build_input_values())
    write_values(model_path, build_model_values())

    print(f"Wrote {input_path}")
    print(f"Wrote {model_path}")


if __name__ == "__main__":
    main()
