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

## Day 4: Math Function Elimination

The project now supports a math function elimination pass:

```bash
./tools/mini-opt/mini-opt ../examples/pow_neg1.mlir --mini-math-eliminate
./tools/mini-opt/mini-opt ../examples/pow_neg2.mlir --mini-math-eliminate

Implemented rewrite rules:

pow(x, -1) -> 1 / x
pow(x, -2) -> (1 / x) * (1 / x)


Example:

Before:

%c_neg2 = arith.constant dense<-2.0> : tensor<4xf32>

%0 = "mini.pow"(%x, %c_neg2)
  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>


After:

%cst = arith.constant dense<1.000000e+00> : tensor<4xf32>

%0 = "mini.div"(%cst, %x)
  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

%1 = "mini.mul"(%0, %0)
  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>


This pass simulates a common AI compiler optimization where expensive math function calls are rewritten into simpler arithmetic operations.


然后更新状态：

```markdown
## Current Status

- [x] Run MLIR Toy Tutorial
- [x] Understand Toy Dialect
- [x] Define Mini Dialect design
- [x] Implement `mini.add`
- [x] Implement `mini.mul`
- [x] Implement `mini.div`
- [x] Implement `mini.pow`
- [x] Build `mini-opt`
- [x] Parse and print Mini IR
- [x] Implement first canonicalization pass
- [x] Support `x + 0 -> x`
- [x] Support `x * 1 -> x`
- [x] Support `x * 0 -> 0`
- [x] Implement math function elimination pass
- [x] Support `pow(x, -1) -> 1 / x`
- [x] Support `pow(x, -2) -> (1 / x) * (1 / x)`
- [ ] Implement `mini.exp`
- [ ] Implement `mini.log`
- [ ] Implement `mini.tanh`
- [ ] Implement `mini.sqrt`
- [ ] Implement simple elementwise fusion
- [ ] Add correctness tests
- [ ] Add benchmark results