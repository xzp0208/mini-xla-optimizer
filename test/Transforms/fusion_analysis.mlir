// RUN: mini-opt %s --mini-fusion-analysis | FileCheck %s

module {
  func.func @fusion_candidate(%x: tensor<4xf32>, %scale: tensor<4xf32>, %bias: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "mini.mul"(%x, %scale)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %1 = "mini.add"(%0, %bias)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %1 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @fusion_candidate
  // CHECK: "mini.add"
  // CHECK-SAME: mini_fusion_expensive_math_ops = 0 : i32
  // CHECK-SAME: mini_fusion_kind = "candidate"
  // CHECK-SAME: mini_fusion_ops = 2 : i32


  func.func @fusion_skip_expensive_math(%x: tensor<4xf32>, %bias: tensor<4xf32>) -> tensor<4xf32> {
    %c_neg2 = arith.constant dense<-2.0> : tensor<4xf32>

    %0 = "mini.pow"(%x, %c_neg2)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %1 = "mini.pow"(%0, %c_neg2)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %2 = "mini.add"(%1, %bias)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %2 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @fusion_skip_expensive_math
  // CHECK: "mini.add"
  // CHECK-SAME: mini_fusion_expensive_math_ops = 2 : i32
  // CHECK-SAME: mini_fusion_kind = "skip"
  // CHECK-SAME: mini_fusion_ops = 3 : i32
  // CHECK-SAME: mini_fusion_skip_reason = "too_many_expensive_math_ops"

}
