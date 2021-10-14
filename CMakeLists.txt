# CMakeList.txt : CMake project for spam5, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("LinearRegressionPytorch")

find_package(Torch REQUIRED)

# Add source to this project's executable.
add_executable (LinearRegressionPytorch "LinearRegressionPytorch.cpp" "LinearRegressionPytorch.h")
target_link_libraries(LinearRegressionPytorch "${TORCH_LIBRARIES}")
# TODO: Add tests and install targets if needed.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET LinearRegressionPytorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:LinearRegressionPytorch>)
endif (MSVC)