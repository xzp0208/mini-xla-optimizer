# Fusion Policy

## 1. Goal

This document describes the first version of the elementwise fusion analysis policy in Mini-XLA-Optimizer.

The goal is not to aggressively fuse every elementwise operation. Instead, the pass uses a conservative policy to identify safe and explainable fusion candidates.

## 2. Fusion Candidate

A root operation can become a fusion candidate if its producer chain satisfies:

```text
1. all operations are mini elementwise ops
2. producer and consumer are in the same block
3. producer has a single result
4. producer result has only one use
5. fusion op count <= max_ops_in_fusion
6. expensive math op count <= max_expensive_math_ops
