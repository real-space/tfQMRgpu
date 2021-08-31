cmake_minimum_required(VERSION 3.11.4)

# set the project name and version
project(tfqmrgpu VERSION 0.1 LANGUAGES C CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

### include debug symbols e.g. for GDB
# set(CMAKE_BUILD_TYPE Debug)

### folders:
###     example --> generator for finite-difference examples
###     fortran --> small Fortran test
###     tfQMRgpu --> main library
###     tfQMRgpu/include --> header-only library
###     tfQMRgpu/source  --> C-interfaced static/shared library for block-sparse

# external dependencies provided in source tree
add_subdirectory(third_party)

add_subdirectory(tfQMRgpu)
add_subdirectory(example)

# TODO: add submodule to git repo?
### submodules: https://github.com/dwd/rapidxml

