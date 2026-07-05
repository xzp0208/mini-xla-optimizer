add_mlir_dialect_library(MLIRMini
  MiniDialect.cpp
  MiniOps.cpp

  DEPENDS
  MiniOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
