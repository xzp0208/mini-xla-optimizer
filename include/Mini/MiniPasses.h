#ifndef MINI_MINIPASSES_H
#define MINI_MINIPASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mini {

std::unique_ptr<mlir::Pass> createCanonicalizeMiniPass();
std::unique_ptr<mlir::Pass> createMathEliminateMiniPass();

void registerCanonicalizeMiniPass();
void registerMathEliminateMiniPass();

inline void registerMiniPasses() {
  registerCanonicalizeMiniPass();
  registerMathEliminateMiniPass();
}

} // namespace mini

#endif // MINI_MINIPASSES_H
