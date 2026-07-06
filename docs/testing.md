# Testing Guide

This project uses FileCheck-style regression tests to verify Mini dialect transformations.

## Test Layout

```text
test/
├── run_filecheck.sh
└── Transforms/
    ├── canonicalize.mlir
    ├── math_eliminate.mlir
    └── pipeline.mlir

