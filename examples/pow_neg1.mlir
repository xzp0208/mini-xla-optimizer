module {
  func.func @pow_neg1(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c_neg1 = arith.constant dense<-1.0> : tensor<4xf32>

    %0 = "mini.pow"(%x, %c_neg1)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %0 : tensor<4xf32>
  }
}

