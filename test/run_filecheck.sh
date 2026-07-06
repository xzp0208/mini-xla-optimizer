#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"

MINI_OPT="${MINI_OPT:-$BUILD_DIR/tools/mini-opt/mini-opt}"

if [[ -n "${FILECHECK:-}" ]]; then
  FILECHECK_BIN="$FILECHECK"
elif [[ -n "${LLVM_BUILD_DIR:-}" && -x "$LLVM_BUILD_DIR/bin/FileCheck" ]]; then
  FILECHECK_BIN="$LLVM_BUILD_DIR/bin/FileCheck"
else
  FILECHECK_BIN="$(command -v FileCheck || true)"
fi

if [[ ! -x "$MINI_OPT" ]]; then
  echo "error: mini-opt not found: $MINI_OPT"
  echo "hint: build the project first, or set MINI_OPT=/path/to/mini-opt"
  exit 1
fi

if [[ -z "$FILECHECK_BIN" || ! -x "$FILECHECK_BIN" ]]; then
  echo "error: FileCheck not found"
  echo "hint: set LLVM_BUILD_DIR=/path/to/llvm/build or FILECHECK=/path/to/FileCheck"
  exit 1
fi

echo "[1/3] canonicalize"
"$MINI_OPT" "$ROOT_DIR/test/Transforms/canonicalize.mlir" \
  --mini-canonicalize \
  | "$FILECHECK_BIN" "$ROOT_DIR/test/Transforms/canonicalize.mlir" \
      --check-prefix=CANON

echo "[2/3] math_eliminate"
"$MINI_OPT" "$ROOT_DIR/test/Transforms/math_eliminate.mlir" \
  --mini-math-eliminate \
  | "$FILECHECK_BIN" "$ROOT_DIR/test/Transforms/math_eliminate.mlir" \
      --check-prefix=MATH

echo "[3/3] pipeline"
"$MINI_OPT" "$ROOT_DIR/test/Transforms/pipeline.mlir" \
  --mini-math-eliminate \
  --mini-canonicalize \
  | "$FILECHECK_BIN" "$ROOT_DIR/test/Transforms/pipeline.mlir" \
      --check-prefix=PIPE

echo "All FileCheck tests passed."
