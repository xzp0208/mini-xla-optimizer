add_llvm_executable(mini-opt
  mini-opt.cpp
)

llvm_update_compile_flags(mini-opt)

target_link_libraries(mini-opt
  PRIVATE
  MLIRMini
  MLIRMiniTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIROptLib
)

mlir_check_all_link_libraries(mini-opt)
