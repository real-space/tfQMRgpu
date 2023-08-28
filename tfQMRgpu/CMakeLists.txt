# This file is part of tfQMRgpu under MIT-License

set(tfQMRgpu_version 0.0.1)

# For cuRAND
find_package(CUDAToolkit REQUIRED)

set(tfQMRgpu_sources source/tfqmrgpu.cu source/tfqmrgpu_error_tool.cxx)

# Explicitly depend on the headers to trigger rebuilds on changes
set(tfQMRgpu_public_headers 
		     include/bsr.hxx
		     include/tfqmrgpu_blockmult.hxx
		     include/tfqmrgpu_blocksparse.hxx
		     include/tfqmrgpu_core.hxx
		     include/tfqmrgpu_cudaStubs.hxx
		     include/tfqmrgpu_example_reader.hxx
		     include/tfqmrgpu_example_xml_reader.hxx
		     include/tfqmrgpu_Fortran.h
		     include/tfqmrgpu.h
		     include/tfqmrgpu_handle.hxx
		     include/tfqmrgpu.hxx
		     include/tfqmrgpu_linalg.hxx
		     include/tfqmrgpu_memWindow.h
		     include/tfqmrgpu_plan.hxx
		     include/tfqmrgpu_util.hxx
		     include/bsr.hxx
		     include/allowed_block_sizes.h
		     )

# TODO: split into public and private headers
# TODO: split out C++ header-only interface library

set(tfQMRgpu_internal_headers )

set(tfQMRgpu_headers ${tfQMRgpu_internal_headers} ${tfQMRgpu_public_headers})

# build shared library
add_library(tfQMRgpu SHARED ${tfQMRgpu_sources} ${tfQMRgpu_headers})

# CUDA architectures 70=Volta, 75=Turing, 80=Ampere
set_property(TARGET tfQMRgpu PROPERTY CUDA_ARCHITECTURES 70)


# Make the source find the headers
# tfQMRgpu is a library, so we use PUBLIC, this way any target that depends on tfQMRgpu will know to add this include
# And we differentiate between build and install - on build the include in the source directory is used while 
# on build we use the include subdirectory in the library installation path
target_include_directories(tfQMRgpu PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/include/private
	)

# Public libraries = libraries the dependents are also linked against
target_link_libraries(tfQMRgpu PUBLIC RapidXML CUDA::curand)

option(HAS_OPENMP "Usage of OpenMP" ON)
if(HAS_OPENMP)
    # add OpenMP support
    find_package(OpenMP REQUIRED)
    # switch OpenMP on for CUDA
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
    # switch OpenMP on for C and C++
    target_link_libraries(tfQMRgpu PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
endif(HAS_OPENMP)

set(tfqmrgpu_targets tfQMRgpu)


add_library(tfQMRgpu_Fortran STATIC source/tfqmrgpu_Fortran_wrappers.c)
target_link_libraries(tfQMRgpu_Fortran PUBLIC tfQMRgpu)
set(tfqmrgpu_targets ${tfqmrgpu_targets} tfQMRgpu_Fortran)


option(HAS_BENCHMARK "generates bench_tfqmrgpu for multiplication and full_solver performance tests" ON)

if(HAS_BENCHMARK)
    add_executable(bench_tfqmrgpu source/bench_tfqmrgpu.cu)
    set_property(TARGET bench_tfqmrgpu PROPERTY CUDA_ARCHITECTURES 70 80)
    target_link_libraries(bench_tfqmrgpu tfQMRgpu)
    set(tfqmrgpu_targets ${tfqmrgpu_targets} bench_tfqmrgpu)
endif(HAS_BENCHMARK)

# new tool to decypher error codes
add_executable(tfqmrgpu_error_tool source/tfqmrgpu_error_tool.cxx ${tfQMRgpu_headers})
target_link_libraries(tfqmrgpu_error_tool tfQMRgpu)
set(tfqmrgpu_targets ${tfqmrgpu_targets} tfqmrgpu_error_tool)


# default installation paths
include(GNUInstallDirs)

# install library
install(TARGETS ${tfqmrgpu_targets}
	EXPORT tfQMRgpuTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

#install public headers
install(FILES ${tfQMRgpu_public_headers} 
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


# install CMake files so the library can be used by other cmake projects
install(EXPORT tfQMRgpuTargets
	FILE tfQMRgpuTargets.cmake
	NAMESPACE tfQMRgpu::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tfQMRgpu
	)

# Create a CMake package so the library can be found with find_package
include(CMakePackageConfigHelpers)

# A Config.cmake.in file is required as input
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/tfQMRgpuConfig.cmake"
	INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tfQMRgpu
	)

# package version
set_property(TARGET tfQMRgpu PROPERTY VERSION ${tfQMRgpu_version})
set_property(TARGET tfQMRgpu PROPERTY SOVERSION 1)
set_property(TARGET tfQMRgpu PROPERTY INTERFACE_tfQMRgpu_MAJOR_VERSION 1)
set_property(TARGET tfQMRgpu APPEND PROPERTY
	COMPATIBLE_INTERFACE_STRING tfQMRgpu_MAJOR_VERSION
	)

# generate the version file for the config file, we are specifying that any newer version will be compatible
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/tfQMRgpuConfigVersion.cmake"
	VERSION "${tfQMRgpu_version}"
	COMPATIBILITY AnyNewerVersion
	)

# install cmake files
install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/tfQMRgpuConfig.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/tfQMRgpuConfigVersion.cmake"
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tfQMRgpu
	)
