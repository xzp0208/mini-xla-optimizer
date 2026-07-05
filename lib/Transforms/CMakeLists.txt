add_mlir_library(MLIRMiniTransforms
  CanonicalizeMini.cpp

  LINK_LIBS PUBLIC
  MLIRMini
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransforms
  MLIRArithDialect
)
