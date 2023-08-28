# This file is part of tfQMRgpu under MIT-License

# CMake 3.18 required for LAPACK::LAPACK and BLAS::BLAS
# Earlier versions possible if using LAPACK_LIBRARIES or LAPACK95_LIBRARIES, see FindLAPACK documentation (same for BLAS)
cmake_minimum_required(VERSION 3.18)

enable_language(Fortran)


# set the project name and version
add_executable(tfqmrgpu_Fortran_example ../tfQMRgpu/include/tfqmrgpu_Fortran_module.F90 tfqmrgpu_Fortran_example.F90)

# __MAIN__ must be defined otherwise an executable can't be generated
target_compile_definitions(tfqmrgpu_Fortran_example PRIVATE -D__MAIN__)

# link with libtfQMRgpu.so build for block-sparse operators
target_link_libraries(tfqmrgpu_Fortran_example tfQMRgpu_Fortran)



option(USE_LAPACK_REFERENCE "Calculate reference solution using LAPACK" OFF)

if(USE_LAPACK_REFERENCE)
	find_package(BLAS REQUIRED)
	find_package(LAPACK REQUIRED)
	# print a warning if LAPACK requested but not found
	if(NOT LAPACK_FOUND OR NOT BLAS_FOUND)
		message(WARNING "Couldn't find LAPACK, disabling reference solution calculation")
	endif()
endif()


add_executable(generate_FD_example tfqmrgpu_generate_FD_example.cxx)

add_executable(test_xml_reader tfqmrgpu_example_xml_reader.cxx)

add_executable(test_legacy_reader tfqmrgpu_example_reader.cxx)

add_executable(tfqmrgpu_C_example tfqmrgpu_C_example.c ../tfQMRgpu/include/tfqmrgpu.h)
target_compile_definitions(tfqmrgpu_C_example PRIVATE -DHAS_TFQMRGPU)

# link our tfQMRgpu library, also adds the include dir
foreach(example generate_FD_example test_xml_reader test_legacy_reader tfqmrgpu_C_example)
    target_link_libraries(${example} tfQMRgpu)
endforeach()


# generate_FD_example requires -D__MAIN__
target_compile_definitions(generate_FD_example PRIVATE -D__MAIN__)

if(LAPACK_FOUND AND BLAS_FOUND)
	target_compile_definitions(generate_FD_example PRIVATE -DHAS_LAPACK)
	target_link_libraries(generate_FD_example LAPACK::LAPACK BLAS::BLAS)
endif()


# default installation paths
include(GNUInstallDirs)

set(tfqmrgpu_example_targets 
                generate_FD_example 
                test_xml_reader
                test_legacy_reader
                tfqmrgpu_C_example
                tfqmrgpu_Fortran_example)

# install examples
install(TARGETS ${tfqmrgpu_example_targets}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

