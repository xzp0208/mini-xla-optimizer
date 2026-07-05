set(LLVM_TARGET_DEFINITIONS MiniOps.td)

mlir_tablegen(MiniOps.h.inc -gen-op-decls)
mlir_tablegen(MiniOps.cpp.inc -gen-op-defs)

mlir_tablegen(MiniDialect.h.inc -gen-dialect-decls -dialect=mini)
mlir_tablegen(MiniDialect.cpp.inc -gen-dialect-defs -dialect=mini)

add_public_tablegen_target(MiniOpsIncGen)
