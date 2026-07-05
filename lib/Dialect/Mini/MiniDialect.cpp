#include "Mini/MiniDialect.h"
#include "Mini/MiniOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mini;

#include "Mini/MiniDialect.cpp.inc"

void MiniDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mini/MiniOps.cpp.inc"
      >();
}
