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

```text
include/Mini/MiniOps.td
include/Mini/MiniDialect.h
include/Mini/MiniOps.h
lib/Dialect/Mini/MiniDialect.cpp
lib/Dialect/Mini/MiniOps.cpp
tools/mini-opt/mini-opt.cpp
