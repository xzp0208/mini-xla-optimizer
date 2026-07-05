#ifndef MINI_MINIOPS_H
#define MINI_MINIOPS_H

#include "Mini/MiniDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// ===== 追加下面这两个关键的 MLIR C++ 头文件 =====
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
// =============================================

#define GET_OP_CLASSES
#include "Mini/MiniOps.h.inc"

#endif // MINI_MINIOPS_H