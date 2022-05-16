cmake_minimum_required(VERSION "3.14.0")

project(L2Projection DESCRIPTION "L2 Projection" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

### Configuration
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

################################################################################
# L2 Projection Python Library
################################################################################

include(pybind11)
pybind11_add_module(L2 "bindings.cpp" "L2_projection.cpp")

# Move Python library to build root
set_target_properties(L2 PROPERTIES LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Public include directory for IPC Toolkit
target_include_directories(L2 PUBLIC "${PROJECT_SOURCE_DIR}")

################################################################################
# Optional Definitions
################################################################################

# For MSVC, do not use the min and max macros.
target_compile_definitions(L2 PUBLIC NOMINMAX)

################################################################################
# Dependencies
################################################################################

# Extra warnings
include(warnings)
target_link_libraries(L2 PRIVATE L2::warnings)

# libigl
include(eigen)
include(libigl)
target_link_libraries(L2 PUBLIC
  Eigen3::Eigen
  igl::core
  igl::embree
)

################################################################################
# Compiler options
################################################################################

# Use C++17
target_compile_features(L2 PUBLIC cxx_std_17)


