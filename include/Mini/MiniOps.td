#ifndef MINI_OPS
#define MINI_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Mini Dialect
//===----------------------------------------------------------------------===//

def Mini_Dialect : Dialect {
  let name = "mini";
  let cppNamespace = "::mini";
  let summary = "A lightweight dialect for AI graph optimization experiments";
  let description = [{
    The Mini dialect is designed for small AI compiler experiments, including
    elementwise operations, math function elimination, fusion, and lowering.
  }];
}

//===----------------------------------------------------------------------===//
// Base Mini Op
//===----------------------------------------------------------------------===//

class Mini_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mini_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// mini.add
//===----------------------------------------------------------------------===//

def AddOp : Mini_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "elementwise add operation";

  let description = [{
    The mini.add operation represents an elementwise addition.

    Example:

      %0 = "mini.add"(%a, %b)
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// mini.mul
//===----------------------------------------------------------------------===//

def MulOp : Mini_Op<"mul", [Pure, SameOperandsAndResultType]> {
  let summary = "elementwise multiply operation";

  let description = [{
    The mini.mul operation represents an elementwise multiplication.

    Example:

      %0 = "mini.mul"(%a, %b)
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// mini.div
//===----------------------------------------------------------------------===//

def DivOp : Mini_Op<"div", [Pure, SameOperandsAndResultType]> {
  let summary = "elementwise divide operation";

  let description = [{
    The mini.div operation represents an elementwise division.

    Example:

      %0 = "mini.div"(%a, %b)
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// mini.pow
//===----------------------------------------------------------------------===//

def PowOp : Mini_Op<"pow", [Pure, SameOperandsAndResultType]> {
  let summary = "elementwise power operation";

  let description = [{
    The mini.pow operation represents an elementwise power operation.

    In V1, the exponent is represented as a tensor constant with the same type
    as the base tensor. For example, pow(x, -2) is represented as:

      %c_neg2 = arith.constant dense<-2.0> : tensor<4xf32>
      %0 = "mini.pow"(%x, %c_neg2)
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    This design keeps the first implementation simple.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

#endif // MINI_OPS
