// RUN: %mini-opt %s --mini-math-eliminate --mini-canonicalize | FileCheck %s --check-prefix=PIPE

module {
  // PIPE-LABEL: func.func @pow_neg2_then_mul_one
  // PIPE-NOT: "mini.pow"
  // PIPE: "mini.div"
  // PIPE: "mini.mul"
  // PIPE-NOT: dense<1.000000e+00> : tensor<4xf32>
  // PIPE: return %{{.*}} : tensor<4xf32>
  func.func @pow_neg2_then_mul_one(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c_neg2 = arith.constant dense<-2.0> : tensor<4xf32>
    %c1 = arith.constant dense<1.0> : tensor<4xf32>

    %0 = "mini.pow"(%x, %c_neg2)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %1 = "mini.mul"(%0, %c1)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %1 : tensor<4xf32>
  }
}
