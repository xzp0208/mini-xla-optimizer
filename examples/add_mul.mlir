module {
  func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "mini.add"(%a, %b)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %1 = "mini.mul"(%0, %b)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    return %1 : tensor<4xf32>
  }
}
