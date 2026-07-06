#include "Mini/MiniPasses.h"
#include "Mini/MiniOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
// ====== 新增下面这行 ======
#include "llvm/ADT/SmallString.h" 
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// 融合策略硬门槛常数定义（Conservative Fusion Policy）
//===----------------------------------------------------------------------===//
static constexpr unsigned kMaxOpsInFusion = 4;        // 一个融合组内最多容纳的算子数量
static constexpr unsigned kMaxExpensiveMathOps = 1;  // 一个融合组内最多容纳的高开销算子(如 pow)数量

/// 过滤器：判断当前操作是否属于合法的逐元素算子
static bool isElementwiseOp(Operation *op) {
  return isa<mini::AddOp, mini::MulOp, mini::DivOp, mini::PowOp>(op);
}

/// 过滤器：判断当前操作是否属于高开销的数学算子
static bool isExpensiveMathOp(Operation *op) {
  return isa<mini::PowOp>(op);
}

/// 辅助函数：检查一个算子指针是否已经存在于给定的容器中
static bool containsOp(ArrayRef<Operation *> ops, Operation *target) {
  for (Operation *op : ops) {
    if (op == target) return true;
  }
  return false;
}

/// 辅助函数：检查一个 MLIR Value（SSA值）是否已经存在于给定的容器中
static bool containsValue(ArrayRef<Value> values, Value target) {
  for (Value value : values) {
    if (value == target) return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// 核心图遍历算法：反向收集融合算子链
//===----------------------------------------------------------------------===//
/// 从根节点（最终的消费者）出发，深度优先遍历（DFS）逆流向上收集合法的生产者算子
/// 严格坚守 4 大安全熔断门槛：
/// 1. 生产者必须是 Mini 认可的逐元素算子
/// 2. 生产者与消费者必须在同一个基本块（Block）内
/// 3. 生产者有且仅有一个输出结果（Single Result）
/// 4. 生产者的输出结果有且仅有一个消费者（Single Use），防止强行融合引入重复计算
static void collectFusionOps(Operation *op,
                             SmallVectorImpl<Operation *> &fusionOps) {
  if (!op || !isElementwiseOp(op)) return;

  // 防止图结构中存在环或重复访问
  if (containsOp(fusionOps, op)) return;

  // 遍历当前算子的所有输入参数（Operands）
  for (Value operand : op->getOperands()) {
    // 寻找定义该输入参数的上游操作（生产者）
    Operation *producer = operand.getDefiningOp();

    if (!producer) continue; // 如果输入是函数参数而不是算子生成的，跳过
    if (!isElementwiseOp(producer)) continue; // 门槛 1
    if (producer->getBlock() != op->getBlock()) continue; // 门槛 2
    if (producer->getNumResults() != 1) continue; // 门槛 3
    if (!producer->getResult(0).hasOneUse()) continue; // 门槛 4

    // 递归向上追溯生产者的生产者
    collectFusionOps(producer, fusionOps);
  }

  // 递归归来弹栈时才推入队列，这确保了拓扑排序：生产者永远在消费者前面
  fusionOps.push_back(op);
}

//===----------------------------------------------------------------------===//
// 融合收益与合法性评估机制
//===----------------------------------------------------------------------===//
struct FusionStats {
  unsigned opCount = 0;              // 组内算子总数
  unsigned expensiveMathOpCount = 0; // 组内昂贵算子总数
};

/// 扫描融合组，统计算子各项指标
static FusionStats analyzeFusionGroup(ArrayRef<Operation *> fusionOps) {
  FusionStats stats;
  stats.opCount = fusionOps.size();

  for (Operation *op : fusionOps) {
    if (isExpensiveMathOp(op))
      ++stats.expensiveMathOpCount;
  }
  return stats;
}

/// 判定当前融合组是否具有真正的优化收益并符合安全限制
static bool isProfitableFusion(const FusionStats &stats) {
  if (stats.opCount <= 1) return false; // 单个算子不需要融
  if (stats.opCount > kMaxOpsInFusion) return false; // 算子太多，拒绝融合以防撑爆寄存器
  if (stats.expensiveMathOpCount > kMaxExpensiveMathOps) return false; // 昂贵算子超标

  return true;
}

//===----------------------------------------------------------------------===//
// 融合边界提取与元数据构建
//===----------------------------------------------------------------------===//
/// 收集整个融合圈子的“外部输入”
/// 如果一个操作数的生产者处于融合组内部，说明它是圈子里的自产自销，无需作为外部输入。
/// 反之，如果是从圈子外面传进来的，它就必须作为新融合算子的输入参数。
static void collectExternalInputs(ArrayRef<Operation *> fusionOps,
                                  SmallVectorImpl<Value> &externalInputs) {
  for (Operation *op : fusionOps) {
    for (Value operand : op->getOperands()) {
      Operation *producer = operand.getDefiningOp();

      // 如果生产者也在这个融合组里，说明是内部中间结果，跳过
      if (producer && containsOp(fusionOps, producer))
        continue;

      // 如果不是内部产物，且还没被记录过，则归纳为真正的外部输入
      if (!containsValue(externalInputs, operand))
        externalInputs.push_back(operand);
    }
  }
}

/// 打印融合表达式字符串，用于调试和生成可视化的 IR (例如生成 "mini.mul -> mini.add")
static std::string buildFusionExpr(ArrayRef<Operation *> fusionOps) {
  llvm::SmallString<128> buffer;
  llvm::raw_svector_ostream os(buffer);

  for (size_t i = 0; i < fusionOps.size(); ++i) {
    if (i != 0) os << " -> ";
    os << fusionOps[i]->getName().getStringRef();
  }
  return std::string(os.str());
}

/// 检查某个算子的所有输出是否都处于“无人使用”的状态（死代码检测）
static bool allResultsHaveNoUses(Operation *op) {
  for (Value result : op->getResults()) {
    if (!result.use_empty()) return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// 核心实现：融合改写模式 (FuseElementwisePattern)
//===----------------------------------------------------------------------===//
struct FuseElementwisePattern : public RewritePattern {
  // 显式构造函数：使用 MatchAnyOpTypeTag() 声明该模式拦截所有类型的操作
  // benefit=1 表示此项优化的匹配优先级
  explicit FuseElementwisePattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *root,
                                PatternRewriter &rewriter) const override {
    // 1. 如果当前的 root 算子连逐元素算子都不是，直接拒绝改写
    if (!isElementwiseOp(root))
      return failure();

    // 2. 召集以 root 算子为终点的整条融合候选链
    SmallVector<Operation *, 8> fusionOps;
    collectFusionOps(root, fusionOps);

    // 3. 统计并评估该链条融合是否合规且有利可图
    FusionStats stats = analyzeFusionGroup(fusionOps);
    if (!isProfitableFusion(stats))
      return failure();

    // 4. 提取出暴露在圈子外面的外部输入参数
    SmallVector<Value, 8> externalInputs;
    collectExternalInputs(fusionOps, externalInputs);

    // 5. 组装元数据：构建融合轨迹描述文本
    std::string fusionExpr = buildFusionExpr(fusionOps);

    // 6. 核心：动态创建全新的高聚合算子 mini.fused_elementwise 的状态
    OperationState state(root->getLoc(), "mini.fused_elementwise");
    state.addOperands(externalInputs);              // 喂入提取出的外部输入
    state.addTypes(root->getResult(0).getType());   // 继承原本 Root 算子的输出数据类型

    // 7. 为全新的算子贴上详细的元数据属性“标签”
    state.addAttribute("fusion_expr", rewriter.getStringAttr(fusionExpr));
    state.addAttribute("fusion_ops", rewriter.getI32IntegerAttr(stats.opCount));
    state.addAttribute("expensive_math_ops", rewriter.getI32IntegerAttr(stats.expensiveMathOpCount));

    // 8. 正式在 IR 树中创建这个全聚合的新算子
    Operation *fusedOp = rewriter.create(state);

    // 9. 用新聚合算子的结果，整体替换掉原来旧的 Root 算子
    // 这一步之后，原本依赖于 root 结果的下游算子会自动接到新算子的输出上
    rewriter.replaceOp(root, fusedOp->getResults());

    // 10. 垃圾清理（死代码消除）：
    // 此时 root 已经被物理删除，但上游那些旧的中间生产者还残留在 IR 中。
    // 我们必须顺着拓扑逆序（使用反向迭代器 rbegin -> rend，即从消费者往生产者方向）安全擦除它们。
    for (auto it = fusionOps.rbegin(); it != fusionOps.rend(); ++it) {
      Operation *op = *it;

      if (op == root) continue; // root 已经被 replaceOp 处理过，跳过

      // 只要当前生产者的所有下游在刚才的擦除中全部归零了，就说明它彻底沦为死代码，可以安全地灰飞烟灭
      if (allResultsHaveNoUses(op))
        rewriter.eraseOp(op);
    }

    return success(); // 宣告融合转换圆满成功
  }
};

//===----------------------------------------------------------------------===//
// Pass 驱动类实现
//===----------------------------------------------------------------------===//
struct FuseElementwiseMiniPass
    : public PassWrapper<FuseElementwiseMiniPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseElementwiseMiniPass)

  // 命令行参数：-mini-fuse-elementwise
  StringRef getArgument() const override { return "mini-fuse-elementwise"; }

  StringRef getDescription() const override {
    return "Fuse simple mini elementwise producer-consumer chains";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // 注册我们写好的高聚合泛型融合重写模式
    RewritePatternSet patterns(context);
    patterns.add<FuseElementwisePattern>(context);

    // 召唤贪婪重写驱动器，遍历 Module 节点，把所有零散的加减乘除链条融合成强大的高聚合内核
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// 工厂函数：创建该 Pass 实例
std::unique_ptr<Pass> mini::createFuseElementwiseMiniPass() {
  return std::make_unique<FuseElementwiseMiniPass>();
}

// 全局注册函数：挂载到 MLIR 基础设施上，使工具链可以通过命令行直接调用它
void mini::registerFuseElementwiseMiniPass() {
  static PassRegistration<FuseElementwiseMiniPass> pass;
}
