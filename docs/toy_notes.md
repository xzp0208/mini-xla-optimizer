# Toy Tutorial Notes

## 1. Toy Tutorial Pipeline

```text
Toy Source
    |
    v
AST
    |
    v
Toy Dialect MLIR
    |
    v
High-level Optimization
    |
    v
Lowering to Affine
    |
    v
Lowering to LLVM
    |
    v
Codegen

## Day 2 Notes: Mini Dialect Skeleton

Today I implemented the first version of the Mini dialect skeleton.

Implemented files:

include/Mini/MiniOps.td
include/Mini/MiniDialect.h
include/Mini/MiniOps.h
lib/Dialect/Mini/MiniDialect.cpp
lib/Dialect/Mini/MiniOps.cpp
tools/mini-opt/mini-opt.cpp

Current supported ops:

mini.add
mini.mul


The project now supports parsing and printing the following IR:

%0 = "mini.add"(%a, %b)
  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

%1 = "mini.mul"(%0, %b)
  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>


Important concepts learned:

Dialect registration
ODS / TableGen op definition
Generated .inc files
Op traits
mini-opt driver
func dialect registration
Generic MLIR syntax

## Day 3: Mini Canonicalization Pass

The project now supports a simple canonicalization pass:

```bash
./tools/mini-opt/mini-opt ../examples/add_zero.mlir --mini-canonicalize
./tools/mini-opt/mini-opt ../examples/mul_one.mlir --mini-canonicalize
./tools/mini-opt/mini-opt ../examples/mul_zero.mlir --mini-canonicalize

Implemented rewrite rules:

x + 0 -> x
0 + x -> x
x * 1 -> x
1 * x -> x
x * 0 -> 0
0 * x -> 0


Example:

Before:

%0 = "mini.add"(%x, %c0)
  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>


After:

return %x : tensor<4xf32>

## Current Status

- [x] Run MLIR Toy Tutorial
- [x] Understand Toy Dialect
- [x] Define Mini Dialect design
- [x] Implement `mini.add`
- [x] Implement `mini.mul`
- [x] Build `mini-opt`
- [x] Parse and print Mini IR
- [x] Implement first canonicalization pass
- [x] Support `x + 0 -> x`
- [x] Support `x * 1 -> x`
- [x] Support `x * 0 -> 0`
- [ ] Implement `mini.div`
- [ ] Implement `mini.pow`
- [ ] Implement math function elimination
- [ ] Implement simple elementwise fusion
- [ ] Add correctness tests
- [ ] Add benchmark results