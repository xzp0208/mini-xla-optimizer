// RUN: %mini-opt %s --mini-canonicalize | FileCheck %s --check-prefix=CANON

module {
  // CANON-LABEL: func.func @add_zero_rhs
  // CANON-NOT: "mini.add"
  // CANON: return %{{.*}} : tensor<4xf32>
  func.func @add_zero_rhs(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant dense<0.0> : tensor<4xf32>
    %0 = "mini.add"(%x, %c0)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CANON-LABEL: func.func @add_zero_lhs
  // CANON-NOT: "mini.add"
  // CANON: return %{{.*}} : tensor<4xf32>
  func.func @add_zero_lhs(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant dense<0.0> : tensor<4xf32>
    %0 = "mini.add"(%c0, %x)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CANON-LABEL: func.func @mul_one_rhs
  // CANON-NOT: "mini.mul"
  // CANON: return %{{.*}} : tensor<4xf32>
  func.func @mul_one_rhs(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c1 = arith.constant dense<1.0> : tensor<4xf32>
    %0 = "mini.mul"(%x, %c1)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CANON-LABEL: func.func @mul_one_lhs
  // CANON-NOT: "mini.mul"
  // CANON: return %{{.*}} : tensor<4xf32>
  func.func @mul_one_lhs(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c1 = arith.constant dense<1.0> : tensor<4xf32>
    %0 = "mini.mul"(%c1, %x)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CANON-LABEL: func.func @mul_zero_rhs
  // CANON: arith.constant dense<0
  // CANON-NOT: "mini.mul"
  // CANON: return %{{.*}} : tensor<4xf32>
  func.func @mul_zero_rhs(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant dense<0.0> : tensor<4xf32>
    %0 = "mini.mul"(%x, %c0)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CANON-LABEL: func.func @mul_zero_lhs
  // CANON: arith.constant dense<0
  // CANON-NOT: "mini.mul"
  // CANON: return %{{.*}} : tensor<4xf32>
  func.func @mul_zero_lhs(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant dense<0.0> : tensor<4xf32>
    %0 = "mini.mul"(%c0, %x)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
