// RUN: %mini-opt %s --mini-math-eliminate | FileCheck %s --check-prefix=MATH

module {
  // MATH-LABEL: func.func @pow_neg1
  // MATH-NOT: "mini.pow"
  // MATH: arith.constant dense<1
  // MATH: "mini.div"
  // MATH: return %{{.*}} : tensor<4xf32>
  func.func @pow_neg1(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c_neg1 = arith.constant dense<-1.0> : tensor<4xf32>
    %0 = "mini.pow"(%x, %c_neg1)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // MATH-LABEL: func.func @pow_neg2
  // MATH-NOT: "mini.pow"
  // MATH: arith.constant dense<1
  // MATH: "mini.div"
  // MATH: "mini.mul"
  // MATH: return %{{.*}} : tensor<4xf32>
  func.func @pow_neg2(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c_neg2 = arith.constant dense<-2.0> : tensor<4xf32>
    %0 = "mini.pow"(%x, %c_neg2)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // MATH-LABEL: func.func @pow_neg3_no_eliminate
  // MATH: "mini.pow"
  // MATH: return %{{.*}} : tensor<4xf32>
  func.func @pow_neg3_no_eliminate(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c_neg3 = arith.constant dense<-3.0> : tensor<4xf32>
    %0 = "mini.pow"(%x, %c_neg3)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
