cmake_minimum_required(VERSION 3.11.4)

# set the project name and version
project(tfqmrgpu VERSION 0.1 LANGUAGES C CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)

### folders:
###     include --> header-only library
###     example --> generator for finite-difference examples
###     fortran --> small Fortran test
###     source  --> C-interfaced static/shared library for block-sparse

### submodules: https://github.com/dwd/rapidxml
