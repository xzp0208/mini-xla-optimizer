# Mini-XLA-Optimizer

Mini-XLA-Optimizer is a lightweight MLIR-based AI graph optimizer project.

The goal of this project is to build a small but complete compiler optimization pipeline for tensor-style elementwise computation graphs.

## Project Goal

This project focuses on several core ideas commonly used in AI compilers:

- high-level graph IR representation
- operation canonicalization
- math function elimination
- elementwise fusion
- conservative fusion policy
- lowering from high-level tensor operations to lower-level loop/codegen dialects

## Compilation Pipeline

```text
Input Mini IR
    |
    v
Mini Dialect
    |
    v
Canonicalization Pass
    |
    v
Math Function Elimination Pass
    |
    v
Elementwise Fusion Pass
    |
    v
Lowering to Affine / Linalg / LLVM
    |
    v
Correctness Test + Benchmark


## Build

```bash
conda activate mlir-dev

export LLVM_BUILD_DIR=$HOME/compiler/llvm-project/build-mlir
export PATH=$LLVM_BUILD_DIR/bin:$PATH

mkdir -p build
cd build

cmake -G Ninja .. \
  -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release

ninja -j8
