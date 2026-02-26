#!/usr/bin/env bash
set -euo pipefail

. scripts/common.sh

check_tools

target=SEAL
seal_src_dir="$DEPS_DIR/$target"
seal_build_dir="$BUILD_DIR/deps/$target"

clean=0
use_hexl=1
build_type=Release

for arg in "$@"; do
  case "$arg" in
    --clean)
      clean=1
      ;;
    --no-hexl)
      use_hexl=0
      ;;
    --debug)
      build_type=Debug
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: bash scripts/build-seal.sh [--clean] [--no-hexl] [--debug]"
      exit 1
      ;;
  esac
done

if [ ! -d "$seal_src_dir" ]; then
  echo -e "${RED}$seal_src_dir${NC} not found. Run scripts/build-deps.sh first."
  exit 1
fi

echo "[info] SEAL source: $seal_src_dir"
echo "[info] SEAL build:  $seal_build_dir"
echo "[info] build_type=$build_type, use_hexl=$use_hexl, clean=$clean"

cd "$seal_src_dir"
git checkout 7923472 # v3.7.2
patch --quiet --no-backup-if-mismatch -N -p1 -i "$WORK_DIR/patch/SEAL.patch" -d "$seal_src_dir" || true

if [ "$clean" -eq 1 ]; then
  echo "[info] Cleaning previous SEAL build/install artifacts..."
  rm -rf "$seal_build_dir"
  rm -rf "$BUILD_DIR/include/SEAL-3.7"
  rm -f "$BUILD_DIR/lib/libseal-3.7.a" "$BUILD_DIR/lib/libseal-3.7.so" "$BUILD_DIR/lib/libseal-3.7.dylib"
  rm -rf "$BUILD_DIR/lib/cmake/SEAL-3.7" "$BUILD_DIR/lib64/cmake/SEAL-3.7"
fi

mkdir -p "$seal_build_dir"
cd "$seal_build_dir"

hexl_flag=OFF
if [ "$use_hexl" -eq 1 ]; then
  hexl_flag=ON
fi

echo "[info] Configuring SEAL..."
cmake "$seal_src_dir" \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
  -DCMAKE_PREFIX_PATH="$BUILD_DIR" \
  -DSEAL_USE_MSGSL=OFF \
  -DSEAL_USE_ZLIB=OFF \
  -DSEAL_USE_ZSTD=ON \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DSEAL_USE_INTEL_HEXL="$hexl_flag" \
  -DSEAL_BUILD_DEPS=OFF

echo "[info] Building & installing SEAL..."
make install -j4

if [ ! -d "$BUILD_DIR/include/SEAL-3.7" ]; then
  echo -e "${RED}SEAL include directory missing after install: $BUILD_DIR/include/SEAL-3.7${NC}"
  exit 1
fi

echo -e "${GREEN}[done] SEAL rebuild complete.${NC}"
