add_mlir_library(MLIRMiniTransforms
  CanonicalizeMini.cpp
  MathEliminateMini.cpp
  FusionAnalysisMini.cpp
  FuseElementwiseMini.cpp
  
  LINK_LIBS PUBLIC
  MLIRMini
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransforms
  MLIRArithDialect
)
