#include "Mini/MiniDialect.h"
#include "Mini/MiniOps.h"
#include "Mini/MiniPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<mini::MiniDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  mini::registerMiniPasses();
//   mlir::registerAllPasses();

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "Mini-XLA optimizer driver\n", registry));
}
