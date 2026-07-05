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
