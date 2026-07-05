module {
  func.func @add_zero(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant dense<0.0> : tensor<4xf32>

    %0 = "mini.add"(%x, %c0)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %0 : tensor<4xf32>
  }
}
