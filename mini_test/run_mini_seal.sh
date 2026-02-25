#!/usr/bin/env bash
set -euo pipefail

# repo root 기준 경로
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINI_DIR="${ROOT_DIR}/mini_test"
BUILD_DIR="${MINI_DIR}/build"
PREFIX_PATH="${ROOT_DIR}/build"

echo "[info] ROOT_DIR=${ROOT_DIR}"
echo "[info] PREFIX_PATH=${PREFIX_PATH}"

# 옵션: --clean 이면 빌드 폴더 삭제
if [[ "${1:-}" == "--clean" ]]; then
  echo "[info] Cleaning ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

echo "[info] Configuring..."
cmake -S "${MINI_DIR}" -B "${BUILD_DIR}" -DCMAKE_PREFIX_PATH="${PREFIX_PATH}"

echo "[info] Building..."
cmake --build "${BUILD_DIR}" -j

echo "[info] Running..."
"${BUILD_DIR}/mini_seal"