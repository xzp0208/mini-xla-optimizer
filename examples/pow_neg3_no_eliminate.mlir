module {
  func.func @pow_neg3(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c_neg3 = arith.constant dense<-3.0> : tensor<4xf32>

    %0 = "mini.pow"(%x, %c_neg3)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %0 : tensor<4xf32>
  }
}

