#include "Mini/MiniPasses.h"

#include "Mini/MiniOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <memory>

using namespace mlir;

namespace {

static constexpr unsigned kMaxOpsInFusion = 4;        // 一个融合组里最多只能有 4 个算子
static constexpr unsigned kMaxExpensiveMathOps = 1;  // 一个融合组里最多只能有 1 个高开销数学算子（如 pow）

// 目前 V1 只把 add/mul/div/pow 当作 elementwise op。
// 后面加 mini.exp / mini.log / mini.tanh / mini.sqrt 后，再扩展这里。
static bool isElementwiseOp(Operation *op) {
  return isa<mini::AddOp, mini::MulOp, mini::DivOp, mini::PowOp>(op);
}

// V1 先只把 pow 当作 expensive math op。
// 后续再加入 exp/log/tanh/sqrt。
static bool isExpensiveMathOp(Operation *op) {
  return isa<mini::PowOp>(op);
}

static bool alreadyCollected(ArrayRef<Operation *> ops, Operation *op) {
  for (Operation *item : ops) {
    if (item == op)
      return true;
  }
  return false;
}

// 从 root op 反向收集 producer chain。
// 只收集：
// 1. mini elementwise producer
// 2. 同一个 block 内的 producer
// 3. 单 result 且只有一个 use 的 producer
//
// 为什么要求 hasOneUse：
// 如果 producer 被多个地方使用，强行融合可能会复制计算，导致性能变差。
static void collectFusionOps(Operation *op,
                             SmallVectorImpl<Operation *> &fusionOps) {
  if (!op || !isElementwiseOp(op))
    return;

  if (alreadyCollected(fusionOps, op))
    return;

  for (Value operand : op->getOperands()) {
    Operation *producer = operand.getDefiningOp(); // 顺着输入连线找到上游的“生产者”操作
    if (!producer) continue;
  
    if (!isElementwiseOp(producer)) continue; // 门槛 1：生产者也必须是 elementwise 算子
  
    if (producer->getBlock() != op->getBlock()) continue; // 门槛 2：必须在同一个控制流块（Block）内
  
    if (producer->getNumResults() != 1) continue; // 门槛 3：生产者只能有一个输出结果
  
    if (!producer->getResult(0).hasOneUse()) continue; // 门槛 4：核心！生产者的结果只能被当前这个消费者使用！
    
    collectFusionOps(producer, fusionOps); // 递归向上追溯
  }
  fusionOps.push_back(op); // 递归归来时放入数组，确保了拓扑排序（生产者永远在消费者前面）
    
}

struct FusionStats {
  unsigned opCount = 0;
  unsigned expensiveMathOpCount = 0;
};

static FusionStats analyzeFusionGroup(ArrayRef<Operation *> fusionOps) {
  FusionStats stats;
  stats.opCount = fusionOps.size();

  for (Operation *op : fusionOps) {
    if (isExpensiveMathOp(op))
      ++stats.expensiveMathOpCount;
  }

  return stats;
}

struct FusionAnalysisMiniPass
    : public PassWrapper<FusionAnalysisMiniPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionAnalysisMiniPass)

  StringRef getArgument() const override { return "mini-fusion-analysis"; }

  StringRef getDescription() const override {
    return "Analyze simple elementwise fusion candidates with conservative policy";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());

    module.walk([&](Operation *op) {
      if (!isElementwiseOp(op))
        return;

      SmallVector<Operation *, 8> fusionOps;
      collectFusionOps(op, fusionOps);

      // 单个 op 不算 fusion group，不打标。
      if (fusionOps.size() <= 1)
        return;

      FusionStats stats = analyzeFusionGroup(fusionOps);

      bool tooManyOps = stats.opCount > kMaxOpsInFusion;
      bool tooManyExpensiveMath =
          stats.expensiveMathOpCount > kMaxExpensiveMathOps;

      bool shouldFuse = !tooManyOps && !tooManyExpensiveMath;

      op->setAttr("mini_fusion_ops",
                  builder.getI32IntegerAttr(stats.opCount));

      op->setAttr("mini_fusion_expensive_math_ops",
                  builder.getI32IntegerAttr(stats.expensiveMathOpCount));

      if (shouldFuse) {
        op->setAttr("mini_fusion_kind",
                    builder.getStringAttr("candidate"));
      } else {
        op->setAttr("mini_fusion_kind",
                    builder.getStringAttr("skip"));

        if (tooManyOps) {
          op->setAttr("mini_fusion_skip_reason",
                      builder.getStringAttr("too_many_ops"));
        } else if (tooManyExpensiveMath) {
          op->setAttr("mini_fusion_skip_reason",
                      builder.getStringAttr("too_many_expensive_math_ops"));
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mini::createFusionAnalysisMiniPass() {
  return std::make_unique<FusionAnalysisMiniPass>();
}

void mini::registerFusionAnalysisMiniPass() {
  static PassRegistration<FusionAnalysisMiniPass> pass;
}
