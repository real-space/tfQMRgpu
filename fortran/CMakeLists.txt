cmake_minimum_required(VERSION 3.17)
enable_language(Fortran)

find_package(CUDAToolkit REQUIRED)

# set the project name and version
add_executable(tfqmrgpu_Fortran_example tfqmrgpu_Fortran.F90 tfqmrgpu_Fortran_wrappers.c)

# __MAIN__ must be defined otherwise an executable can't be generated
target_compile_definitions(tfqmrgpu_Fortran_example PRIVATE -D__MAIN__)

target_link_libraries(tfqmrgpu_Fortran_example tfQMRgpu CUDA::curand)


