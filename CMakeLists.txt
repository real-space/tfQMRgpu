# This file is part of tfQMRgpu under MIT-License

cmake_minimum_required(VERSION 3.11.4)

# set the project name and version
project(tfqmrgpu VERSION 1.0 LANGUAGES C CXX)
# CUDA will be enabled if present

option(HAS_COMPILED_LIB "Providing a compiled shared/static library for a selected block sizes" OFF)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

if(HAS_COMPILED_LIB)

    option(WITH_CUDA "Try to find a CUDA compiler" ON)
    if(WITH_CUDA)
        find_package(CUDA)
        if(CUDA_FOUND)
            option(HAS_CUDA "Using the CUDA programming model for GPUs" ON )
        else(CUDA_FOUND)
            option(HAS_CUDA "Using the CUDA programming model for GPUs" OFF)
        endif(CUDA_FOUND)
    else(WITH_CUDA)
            option(HAS_CUDA "Using the CUDA programming model for GPUs" OFF)
    endif(WITH_CUDA)

    ### include debug symbols e.g. for GDB
    # set(CMAKE_BUILD_TYPE Debug)

    ### folders:
    ###     example --> generator for finite-difference examples
    ###     fortran --> small Fortran test
    ###     tfQMRgpu --> main library
    ###     tfQMRgpu/include --> header-only library
    ###     tfQMRgpu/source  --> C-interfaced static/shared library for block-sparse [optional]

    # external dependencies provided in source tree
    add_subdirectory(third_party)

    # add main functionality, header-only library and shared objects and tools
    add_subdirectory(tfQMRgpu)

    # compile and link an example
    add_subdirectory(example)

else(HAS_COMPILED_LIB)

    # use tfQMRgpu as header-only library

    # define which header files are public
    set(tfQMRgpu_public_headers
        tfQMRgpu/include/tfqmrgpu.hxx
        tfQMRgpu/include/tfqmrgpu_memWindow.h
        tfQMRgpu/include/tfqmrgpu_bsr.hxx
        tfQMRgpu/include/tfqmrgpu_linalg.hxx
        tfQMRgpu/include/tfqmrgpu_util.hxx
        tfQMRgpu/include/tfqmrgpu_core.hxx
        tfQMRgpu/include/tfqmrgpu_cudaStubs.hxx
		)

    # default installation paths
    include(GNUInstallDirs)

    # install public headers to installation directory
    install(FILES ${tfQMRgpu_public_headers} 
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

endif(HAS_COMPILED_LIB)
