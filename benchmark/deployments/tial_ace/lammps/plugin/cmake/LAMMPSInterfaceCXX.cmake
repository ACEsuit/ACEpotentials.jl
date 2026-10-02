# CMake module to define the LAMMPS C++ interface
# Required for building LAMMPS plugins

################################################################################
# LAMMPS C++ interface - header-only for shared linkage
add_library(lammps INTERFACE)
target_include_directories(lammps INTERFACE ${LAMMPS_HEADER_DIR})

################################################################################
# MPI configuration
if(NOT CMAKE_CROSSCOMPILING)
  set(MPI_CXX_SKIP_MPICXX TRUE)
  find_package(MPI QUIET)
  option(BUILD_MPI "Build MPI version" ${MPI_FOUND})
else()
  option(BUILD_MPI "Build MPI version" OFF)
endif()

if(BUILD_MPI)
  set(MPI_CXX_SKIP_MPICXX TRUE)
  find_package(MPI REQUIRED)
  target_link_libraries(lammps INTERFACE MPI::MPI_CXX)
else()
  # Check if STUBS directory exists for serial builds
  if(EXISTS "${LAMMPS_HEADER_DIR}/STUBS")
    target_include_directories(lammps INTERFACE "${LAMMPS_HEADER_DIR}/STUBS")
  endif()
endif()

################################################################################
# Integer size selection (match LAMMPS build)
set(LAMMPS_SIZES "smallbig" CACHE STRING "LAMMPS integer sizes (smallsmall: all 32-bit, smallbig: 64-bit atoms/timesteps, bigbig: also 64-bit imageint)")
set_property(CACHE LAMMPS_SIZES PROPERTY STRINGS smallbig bigbig smallsmall)
string(TOUPPER ${LAMMPS_SIZES} LAMMPS_SIZES)
target_compile_definitions(lammps INTERFACE -DLAMMPS_${LAMMPS_SIZES})

################################################################################
# OpenMP support (optional)
find_package(OpenMP QUIET)
if(OpenMP_FOUND)
  check_include_file_cxx(omp.h HAVE_OMP_H_INCLUDE)
  if(HAVE_OMP_H_INCLUDE)
    set(BUILD_OMP_DEFAULT ON)
  else()
    set(BUILD_OMP_DEFAULT OFF)
  endif()
else()
  set(BUILD_OMP_DEFAULT OFF)
endif()

option(BUILD_OMP "Build with OpenMP support" ${BUILD_OMP_DEFAULT})

if(BUILD_OMP)
  find_package(OpenMP REQUIRED)
  check_include_file_cxx(omp.h HAVE_OMP_H_INCLUDE)
  if(NOT HAVE_OMP_H_INCLUDE)
    message(FATAL_ERROR "Cannot find 'omp.h' header required for OpenMP support")
  endif()
  target_link_libraries(lammps INTERFACE OpenMP::OpenMP_CXX)
endif()
