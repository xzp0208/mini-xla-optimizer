#include "Mini/MiniPasses.h"
#include "Mini/MiniOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

static std::optional<double> getSplatFloatConstant(Value value) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp) return std::nullopt;

  Attribute attr = constOp.getValue();

  // 情况 A：处理密集的、所有元素都相同的张量/向量常量 (如 dense<0.0> : tensor<4xf32>)
  if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(attr)) {
    if (!denseAttr.isSplat()) return std::nullopt; // 如果内部数字不全相等，则无法优化
    APFloat splatValue = denseAttr.getSplatValue<APFloat>();
    return splatValue.convertToDouble();
  }

  // 情况 B：处理普通的标量浮点数常量 (如 -1.0 : f32)
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValueAsDouble();
  }

  return std::nullopt;
}

static bool isSplatFloat(Value value, double expected) {
  std::optional<double> maybeValue = getSplatFloatConstant(value);
  if (!maybeValue.has_value())
    return false;

  constexpr double epsilon = 1e-6;
  return std::abs(maybeValue.value() - expected) < epsilon;
}

static Value createSplatFloatConstant(PatternRewriter &rewriter, Location loc, Type type, double value) {
  // 如果结果需要的是一个高维张量类型 (Tensor Type)
  if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
    Type elementType = rankedTensorType.getElementType();
    auto floatType = dyn_cast<FloatType>(elementType);
    if (!floatType) return Value();

    auto floatAttr = rewriter.getFloatAttr(floatType, value);
    // 动态构建 dense<1.0> : tensor<4xf32>
    auto denseAttr = DenseElementsAttr::get(rankedTensorType, floatAttr);
    return rewriter.create<arith::ConstantOp>(loc, denseAttr);
  }

  // 如果结果需要的只是普通的标量浮点类型 (Float Type)
  if (auto floatType = dyn_cast<FloatType>(type)) {
    auto floatAttr = rewriter.getFloatAttr(floatType, value);
    return rewriter.create<arith::ConstantOp>(loc, floatAttr);
  }
  return Value();
}


static void eraseUnusedArithConstants(ModuleOp module) {
  SmallVector<arith::ConstantOp, 16> constantsToErase;

  module.walk([&](arith::ConstantOp op) {
    if (op.getResult().use_empty())
      constantsToErase.push_back(op);
  });

  for (arith::ConstantOp op : constantsToErase)
    op.erase();
}

//===----------------------------------------------------------------------===//
// pow(x, -1) -> 1 / x
// pow(x, -2) -> (1 / x) * (1 / x)
//===----------------------------------------------------------------------===//

struct PowEliminatePattern : public OpRewritePattern<mini::PowOp> {
  using OpRewritePattern<mini::PowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mini::PowOp op, PatternRewriter &rewriter) const override {
    Value base = op->getOperand(0);     // 底数 x
    Value exponent = op->getOperand(1); // 指数 y

    Location loc = op.getLoc();
    Type resultType = op.getResult().getType(); // 获取操作原本产出的数据类型

    // 规则 1：pow(x, -1) -> 1 / x
    if (isSplatFloat(exponent, -1.0)) {
      // 1. 创建一个与结果类型一致的 1.0 常量
      Value one = createSplatFloatConstant(rewriter, loc, resultType, 1.0);
      if (!one) return failure();

      // 2. 动态创建 mini.div 操作计算 1 / x
      auto div = rewriter.create<mini::DivOp>(loc, resultType, one, base);
      // 3. 用 div 的结果替换掉原本的 pow
      rewriter.replaceOp(op, div.getResult());
      return success();
    }

    // 规则 2：pow(x, -2) -> (1 / x) * (1 / x)
    if (isSplatFloat(exponent, -2.0)) {
      Value one = createSplatFloatConstant(rewriter, loc, resultType, 1.0);
      if (!one) return failure();

      // 1. 先造出倒数：reciprocal = 1 / x
      auto reciprocal = rewriter.create<mini::DivOp>(loc, resultType, one, base);
      // 2. 再对自己做乘法：mul = reciprocal * reciprocal
      auto mul = rewriter.create<mini::MulOp>(
          loc, resultType, reciprocal.getResult(), reciprocal.getResult());
      
      // 3. 替换原操作
      rewriter.replaceOp(op, mul.getResult());
      return success();
    }

    return failure();
  }
};

struct MathEliminateMiniPass
    : public PassWrapper<MathEliminateMiniPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MathEliminateMiniPass)

  StringRef getArgument() const final { return "mini-math-eliminate"; }

  StringRef getDescription() const final {
    return "Eliminate expensive math functions in the Mini dialect";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<PowEliminatePattern>(&getContext());

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    eraseUnusedArithConstants(module);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mini::createMathEliminateMiniPass() {
  return std::make_unique<MathEliminateMiniPass>();
}

void mini::registerMathEliminateMiniPass() {
  static PassRegistration<MathEliminateMiniPass> pass;
}

