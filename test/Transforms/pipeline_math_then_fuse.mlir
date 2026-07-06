// RUN: mini-opt %s --mini-math-eliminate --mini-fuse-elementwise | FileCheck %s

module {
  func.func @math_then_fuse(
      %x: tensor<4xf32>,
      %bias: tensor<4xf32>) -> tensor<4xf32> {
    %c_neg2 = arith.constant dense<-2.0> : tensor<4xf32>

    %0 = "mini.pow"(%x, %c_neg2)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %1 = "mini.add"(%0, %bias)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %1 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @math_then_fuse
  // CHECK-NOT: "mini.pow"
  // CHECK: "mini.fused_elementwise"
}
