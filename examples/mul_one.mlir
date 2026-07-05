module {
  func.func @mul_one(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c1 = arith.constant dense<1.0> : tensor<4xf32>

    %0 = "mini.mul"(%x, %c1)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %0 : tensor<4xf32>
  }
}
