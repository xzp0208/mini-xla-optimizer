// 引入自定义方言 Mini 的 Pass（通道）声明与操作（Ops）定义的头文件
#include "Mini/MiniPasses.h"
#include "Mini/MiniOps.h"

// 引入 MLIR 官方算术方言（Arith Dialect）的头文件，用于后续处理 arith.constant
#include "mlir/Dialect/Arith/IR/Arith.h"
// 引入 MLIR 内置核心操作（如 ModuleOp）的头文件
#include "mlir/IR/BuiltinOps.h"
// 引入 MLIR 模式匹配器（Matchers），用于检查 SSA 值是否为特定常量
#include "mlir/IR/Matchers.h"
// 引入 MLIR 图案匹配重写机制的核心类（如 PatternRewriter、OpRewritePattern）
#include "mlir/IR/PatternMatch.h"
// 引入 MLIR Pass 基础设施，用于自定义和包装优化通道
#include "mlir/Pass/Pass.h"
// 引入贪婪模式重写驱动器，用于循环迭代地应用我们的重写规则
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// 命名空间引用，避免写冗长的 mlir:: 前缀
using namespace mlir;

// 使用匿名命名空间，确保这些类和辅助函数只在当前编译单元（当前源文件）中可见
namespace {
// 新增一个通用的常量值检查函数，完美兼容标量和张量
static bool isConstantValue(Value value, double target) {
    Attribute attr;
    // 基础检查：必须是一个编译期常量
    if (!matchPattern(value, m_Constant(&attr)))
      return false;
  
    // 情况 1：处理标量浮点数 (例如：0.0 : f32)
    if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attr))
      return floatAttr.getValueAsDouble() == target;
  
    // 情况 2：处理标量整数 (例如：0 : i32)
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt() == target;
  
    // 情况 3：处理张量/向量的密集全等常量 (例如：dense<0.0> : tensor<4xf32>)
    if (auto elementsAttr = llvm::dyn_cast<DenseElementsAttr>(attr)) {
      // isSplat 检查张量内所有元素是否完全相同
      if (elementsAttr.isSplat()) {
        Type elemType = elementsAttr.getType().getElementType();
        // 检查张量内部元素的类型
        if (llvm::isa<FloatType>(elemType))
          return elementsAttr.getSplatValue<FloatAttr>().getValueAsDouble() == target;
        if (llvm::isa<IntegerType>(elemType))
          return elementsAttr.getSplatValue<IntegerAttr>().getInt() == target;
      }
    }
  
    return false;
  }

/// 辅助函数：检查给定的 MLIR Value 是否为常量 0
/// matchPattern 会自动向上追溯到定义该 value 的操作，并判断其是否为符合 m_Zero() 规则的常量
static bool isZero(Value value) {
    return isConstantValue(value, 0.0);
}

/// 辅助函数：检查给定的 MLIR Value 是否为常量 1
/// 同样利用了 MLIR 内置的匹配器 m_One()
static bool isOne(Value value) {
    return isConstantValue(value, 1.0);
}

/// 辅助函数：将当前操作替换为另一个已有的 SSA 值
/// @param rewriter MLIR 的重写器，负责安全地修改 IR 结构
/// @param op 准备被替换掉的旧操作
/// @param value 用于替代旧操作结果的新值
static void replaceOpWithValue(PatternRewriter &rewriter, Operation *op,
                               Value value) {
  // 创建一个包含单个值的小型向量（SmallVector 是 LLVM 优化的紧凑型数组）
  SmallVector<Value, 1> replacement{value};
  // 调用重写器替换操作。这会把所有使用旧 op 结果的地方自动指向新 value，并在稍后安全删除旧 op
  rewriter.replaceOp(op, replacement);
}

//===----------------------------------------------------------------------===//
// 加法优化模式：
// x + 0 -> x
// 0 + x -> x
//===----------------------------------------------------------------------===//

// 继承自基础类 OpRewritePattern，并明确指定只对 mini::AddOp 类型的操作进行匹配
struct AddZeroPattern : public OpRewritePattern<mini::AddOp> {
  // 继承父类的构造函数
  using OpRewritePattern<mini::AddOp>::OpRewritePattern;

  // 核心的匹配与改写逻辑
  LogicalResult matchAndRewrite(mini::AddOp op,
                                PatternRewriter &rewriter) const override {
    // 获取加法操作的两个输入操作数（0位置为左操作数，1位置为右操作数）
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);

    // 规则 1：如果右操作数是 0 (x + 0)，则整个加法操作可以直接用左操作数 x 替代
    if (isZero(rhs)) {
      replaceOpWithValue(rewriter, op.getOperation(), lhs);
      return success(); // 返回成功，通知框架 IR 已被修改
    }

    // 规则 2：如果左操作数是 0 (0 + x)，则整个加法操作可以直接用右操作数 x 替代
    if (isZero(lhs)) {
      replaceOpWithValue(rewriter, op.getOperation(), rhs);
      return success(); // 返回成功
    }

    // 如果两个操作数都不是 0，说明无法应用此优化规则，返回 failure()
    // 贪婪驱动器收到失败信号后不会对这个 op 做任何改动
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// 乘法优化模式：
// x * 1 -> x
// 1 * x -> x
// x * 0 -> 0
// 0 * x -> 0
//===----------------------------------------------------------------------===//

// 继承自基础类 OpRewritePattern，明确指定只匹配 mini::MulOp 类型的操作
struct MulCanonicalizePattern : public OpRewritePattern<mini::MulOp> {
  using OpRewritePattern<mini::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mini::MulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);

    // 规则 1：x * 0 -> 0
    // 直接复用代表 0 的右操作数(rhs)来作为替换值，既高效又避免了重新创建新常量操作
    if (isZero(rhs)) {
      replaceOpWithValue(rewriter, op.getOperation(), rhs);
      return success();
    }

    // 规则 2：0 * x -> 0
    // 直接复用代表 0 的左操作数(lhs)作为替换值
    if (isZero(lhs)) {
      replaceOpWithValue(rewriter, op.getOperation(), lhs);
      return success();
    }

    // 规则 3：x * 1 -> x
    // 乘 1 不改变大小，直接用左操作数 x (lhs) 替换整个乘法
    if (isOne(rhs)) {
      replaceOpWithValue(rewriter, op.getOperation(), lhs);
      return success();
    }

    // 规则 4：1 * x -> x
    // 同上，直接用右操作数 x (rhs) 替换整个乘法
    if (isOne(lhs)) {
      replaceOpWithValue(rewriter, op.getOperation(), rhs);
      return success();
    }

    // 四条规则均未匹配成功，保持原样
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass 驱动类实现
//===----------------------------------------------------------------------===//

// 继承自 PassWrapper，声明这是一个专门作用于 ModuleOp（整个 IR 模块级别）的 OperationPass
struct CanonicalizeMiniPass
    : public PassWrapper<CanonicalizeMiniPass, OperationPass<ModuleOp>> {
  
  // MLIR 内部的类型安全标识符宏，用于 Pass 的注册和识别机制
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeMiniPass)

  // 定义该 Pass 在命令行工具（如 mlir-opt）中调用时的参数名：-mini-canonicalize
  StringRef getArgument() const final { return "mini-canonicalize"; }

  // 提供对该 Pass 功能的简短文本描述
  StringRef getDescription() const final {
    return "Apply simple canonicalization patterns for the Mini dialect";
  }

  // Pass 的核心执行入口函数
  void runOnOperation() override {
    // 创建一个重写模式集合，传入当前的上下文（MLIRContext）
    RewritePatternSet patterns(&getContext());

    // 将上面写好的两个自定义优化模式注册到集合中
    patterns.add<AddZeroPattern>(&getContext());
    patterns.add<MulCanonicalizePattern>(&getContext());

    // 调用 MLIR 官方的贪婪模式改写驱动器（Greedy Pattern Rewrite Driver）
    // 它会遍历整个 ModuleOp（由 getOperation() 获取），在工作队列中反复应用注册的模式，直到 IR 达到不动点（不再发生改变）
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      // 如果重写过程中发生致命错误，向框架发出 Pass 失败信号并退出
      signalPassFailure();
      return;
    }

    //===------------------------------------------------------------------===//
    // 死代码消除 (DCE - Dead Code Elimination)
    // 目的：清理因为上面的代数化简而变成“孤儿”的、无任何使用者的小测试常量
    //===------------------------------------------------------------------===//
    
    // 存储等待被抹除的常量操作的数组容器
    SmallVector<arith::ConstantOp, 8> constantsToErase;
    
    // getOperation().walk 是一个深度优先遍历函数，它会扫描整个模块中所有的 arith::ConstantOp
    getOperation().walk([&](arith::ConstantOp op) {
      // 检查当前常量操作产出的结果值（Result）的使用计数。如果为 0，说明没有任何操作在引用它
      if (op.getResult().use_empty())
        constantsToErase.push_back(op); // 扔进待删除队列
    });

    // 安全地遍历并从 IR 树中彻底抹去这些无用的常量操作
    for (arith::ConstantOp op : constantsToErase)
      op.erase();
  }
};

} // namespace

// 提供给外部 C++ 代码（如 PassManager 管道配置）用于创建该 Pass 实例的工厂函数
std::unique_ptr<mlir::Pass> mini::createCanonicalizeMiniPass() {
  return std::make_unique<CanonicalizeMiniPass>();
}

void mini::registerCanonicalizeMiniPass() {
  static PassRegistration<CanonicalizeMiniPass> pass;
}

