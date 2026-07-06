#ifndef MINI_MINIPASSES_H
#define MINI_MINIPASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mini {

std::unique_ptr<mlir::Pass> createCanonicalizeMiniPass();
std::unique_ptr<mlir::Pass> createMathEliminateMiniPass();
std::unique_ptr<mlir::Pass> createFusionAnalysisMiniPass();
std::unique_ptr<mlir::Pass> createFuseElementwiseMiniPass();

void registerCanonicalizeMiniPass();
void registerMathEliminateMiniPass();
void registerFusionAnalysisMiniPass();
void registerFuseElementwiseMiniPass();

inline void registerMiniPasses() {
  registerCanonicalizeMiniPass();
  registerMathEliminateMiniPass();
  registerFusionAnalysisMiniPass();
  registerFuseElementwiseMiniPass();
}

} // namespace mini

#endif // MINI_MINIPASSES_H
