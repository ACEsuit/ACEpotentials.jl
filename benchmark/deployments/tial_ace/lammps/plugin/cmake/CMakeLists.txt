# ----------------------------------------------------------------------
# CMake configuration for LAMMPS ACE Plugin
#
# This builds the aceplugin.so which provides pair_style ace for
# loading ACE potential models compiled from ACEpotentials.jl.
#
# Usage:
#   mkdir build && cd build
#   cmake ../cmake -DLAMMPS_HEADER_DIR=/path/to/lammps/src
#   make
#
# This produces aceplugin.so which can be loaded via:
#   plugin load /path/to/aceplugin.so
# ----------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# enforce out-of-source build
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed. Please create a build directory.")
endif()

project(aceplugin VERSION 1.0 LANGUAGES CXX)

# when this file is included as subdirectory in the LAMMPS build, many settings are directly imported
if(LAMMPS_DIR)
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR})
else()
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR} CACHE PATH "Location of LAMMPS headers")
  if(NOT LAMMPS_HEADER_DIR)
    message(FATAL_ERROR "Must set LAMMPS_HEADER_DIR to the location of LAMMPS source headers")
  endif()
  # by default, install into $HOME/.local
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Default install path" FORCE)
  endif()
endif()

# C++17 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Need -restrict with Intel compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -restrict")
endif()

# OpenMP support (enabled by default)
option(BUILD_OMP "Build with OpenMP support" ON)
if(BUILD_OMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
  else()
    message(WARNING "OpenMP not found, building without OpenMP support")
    set(BUILD_OMP OFF)
  endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(CheckIncludeFileCXX)
if(NOT LAMMPS_DIR)
  include(LAMMPSInterfaceCXX)
endif()

##########################
# building the plugin
set(SRC_DIR ../src)

add_library(aceplugin MODULE
  ${SRC_DIR}/aceplugin.cpp
  ${SRC_DIR}/pair_ace.cpp
  ${SRC_DIR}/pair_ace.h)

target_link_libraries(aceplugin PRIVATE lammps)

# Link against dl for dlopen/dlsym
target_link_libraries(aceplugin PRIVATE ${CMAKE_DL_LIBS})

# Link OpenMP if available
if(BUILD_OMP AND OpenMP_CXX_FOUND)
  target_link_libraries(aceplugin PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(aceplugin PROPERTIES PREFIX "" SUFFIX ".so")

# Platform-specific link flags
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  set_target_properties(aceplugin PROPERTIES
    LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set_target_properties(aceplugin PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  if(CMAKE_CROSSCOMPILING)
    set_target_properties(aceplugin PROPERTIES
      LINK_FLAGS "-Wl,--export-all-symbols")
  endif()
else()
  set_target_properties(aceplugin PROPERTIES
    LINK_FLAGS "-rdynamic")
endif()

# Installation
install(TARGETS aceplugin LIBRARY DESTINATION lib)

# Print configuration summary
message(STATUS "")
message(STATUS "ACE Plugin Configuration:")
message(STATUS "  LAMMPS headers: ${LAMMPS_HEADER_DIR}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
if(BUILD_OMP AND OpenMP_CXX_FOUND)
  message(STATUS "  OpenMP: enabled (${OpenMP_CXX_VERSION})")
else()
  message(STATUS "  OpenMP: disabled")
endif()
message(STATUS "")
