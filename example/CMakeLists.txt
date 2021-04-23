# CMake 3.18 required for LAPACK::LAPACK and BLAS::BLAS
# Earlier versions possible if using LAPACK_LIBRARIES or LAPACK95_LIBRARIES, see FindLAPACK documentation (same for BLAS)
cmake_minimum_required(VERSION 3.18)

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

add_executable(test_legacy_reader  tfqmrgpu_example_reader.cxx)

#Link our tfQMRgpu library, also adds the include dir
foreach(example generate_FD_example test_xml_reader test_legacy_reader)
	target_link_libraries(${example} tfQMRgpu)
endforeach()


# generate_FD_example requires -D__MAIN__
target_compile_definitions(generate_FD_example PRIVATE -D__MAIN__)

if(LAPACK_FOUND AND BLAS_FOUND)
	target_compile_definitions(generate_FD_example PRIVATE -DHAS_LAPACK)
	target_link_libraries(generate_FD_example LAPACK::LAPACK BLAS::BLAS)
endif()
