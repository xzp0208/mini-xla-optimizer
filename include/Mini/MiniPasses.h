#ifndef MINI_MINIPASSES_H
#define MINI_MINIPASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mini {

std::unique_ptr<mlir::Pass> createCanonicalizeMiniPass();
std::unique_ptr<mlir::Pass> createMathEliminateMiniPass();
std::unique_ptr<mlir::Pass> createFusionAnalysisMiniPass();


void registerCanonicalizeMiniPass();
void registerMathEliminateMiniPass();
void registerFusionAnalysisMiniPass();

inline void registerMiniPasses() {
  registerCanonicalizeMiniPass();
  registerMathEliminateMiniPass();
  registerFusionAnalysisMiniPass();
}

} // namespace mini

#endif // MINI_MINIPASSES_H
