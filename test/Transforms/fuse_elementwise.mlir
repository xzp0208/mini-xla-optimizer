// RUN: mini-opt %s --mini-fuse-elementwise | FileCheck %s

module {
  func.func @fuse_mul_add(
      %x: tensor<4xf32>,
      %scale: tensor<4xf32>,
      %bias: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "mini.mul"(%x, %scale)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %1 = "mini.add"(%0, %bias)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %1 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @fuse_mul_add
  // CHECK: "mini.fused_elementwise"
  // CHECK-SAME: expensive_math_ops = 0 : i32
  // CHECK-SAME: fusion_expr = "mini.mul -> mini.add"
  // CHECK-SAME: fusion_ops = 2 : i32


  func.func @skip_expensive_math(
      %x: tensor<4xf32>,
      %bias: tensor<4xf32>) -> tensor<4xf32> {
    %c_neg2 = arith.constant dense<-2.0> : tensor<4xf32>

    %0 = "mini.pow"(%x, %c_neg2)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %1 = "mini.pow"(%0, %c_neg2)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %2 = "mini.add"(%1, %bias)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %2 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @skip_expensive_math
  // CHECK: "mini.pow"
  // CHECK: "mini.pow"
  // CHECK: "mini.add"
  // CHECK-NOT: "mini.fused_elementwise"

  func.func @do_not_fuse_multi_use(
      %x: tensor<4xf32>,
      %scale: tensor<4xf32>,
      %bias: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "mini.mul"(%x, %scale)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %1 = "mini.add"(%0, %bias)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %2 = "mini.add"(%0, %1)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %2 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @do_not_fuse_multi_use
  // CHECK: "mini.fused_elementwise"
  // CHECK-SAME: expensive_math_ops = 0 : i32
  // CHECK-SAME: fusion_expr = "mini.add -> mini.add"
  // CHECK-SAME: fusion_ops = 2 : i32

}
