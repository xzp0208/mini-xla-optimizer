module {
  func.func @pow_neg2(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c_neg2 = arith.constant dense<-2.0> : tensor<4xf32>

    %0 = "mini.pow"(%x, %c_neg2)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %0 : tensor<4xf32>
  }
}

