cmake_minimum_required(VERSION 4.0)

project(TECS
  VERSION 0.1
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(Kokkos_ENABLE_DEPRECATED_CODE OFF CACHE BOOL "" FORCE)
set(Kokkos_BACKEND "" CACHE STRING
  "Choose Kokkos backend: SERIAL, OPENMP, THREADS, HPX, CUDA, HIP, SYCL, OPENMPTARGET, OPENACC"
)

# TODO: error if backend is unknown
if (Kokkos_BACKEND STREQUAL "OPENMP")
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "THREADS")
  set(Kokkos_ENABLE_THREADS ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "HPX")
  set(Kokkos_ENABLE_HPX ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "CUDA")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "HIP")
  set(Kokkos_ENABLE_HIP ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "SYCL")
  set(Kokkos_ENABLE_SYCL ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "OPENMPTARGET")
  set(Kokkos_ENABLE_OPENMPTARGET ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "OPENACC")
  set(Kokkos_ENABLE_OPENACC ON CACHE BOOL "" FORCE)
else()
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
endif()

# Debug mode — enable Kokkos debug features automatically
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_DEBUG_ALGORITHM ON CACHE BOOL "" FORCE)
  message(STATUS "Debug build: Kokkos debug & bounds checking enabled")
endif()

# Kokkos
find_package(Kokkos 5.1 CONFIG)
if (Kokkos_FOUND)
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version ${Kokkos_VERSION})")
else()
  message(STATUS "Kokkos not found externally. Fetching via FetchContent.")

  include(FetchContent)

  FetchContent_Declare(
    Kokkos
    URL https://github.com/kokkos/kokkos/releases/download/5.1.0/kokkos-5.1.0.tar.gz
    URL_HASH SHA256=7bdbdfc88033ed7d940c7940ed8919e1f2b78a9656c69276beb76ad45c41ec4e
  )
  # TODO: use this instead
  # FetchContent_Declare(
    # kokkos
    # GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    # GIT_TAG        5.2.0
  # )
  FetchContent_MakeAvailable(Kokkos)
endif()

# HighFive
find_package(HighFive CONFIG QUIET)
if (HighFive_FOUND)
    message(STATUS "Found HighFive: ${HighFive_DIR}")
else()
    message(STATUS "HighFive not found externally. Fetching...")

    include(FetchContent)

    FetchContent_Declare(
        HighFive
        GIT_REPOSITORY https://github.com/highfive-devs/highfive
        GIT_TAG v3.3.0
        # TODO: URL_HASH
    )
    FetchContent_MakeAvailable(HighFive)
endif()
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# argparse
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

# ccache support (faster compilation)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
else()
  message(STATUS "ccache not found — compile caching disabled")
endif()

add_library(TECS_core INTERFACE)

target_include_directories(TECS_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(TECS_core INTERFACE
  Kokkos::kokkos
  HighFive::HighFive
  HDF5::HDF5
  argparse
)

target_sources(TECS_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/runtime_guard.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/utils.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/types/vector.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/types/polarity.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/types/plane.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/types/link.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/types/view.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/types/device_var.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/forces/predefined.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/initializers/relax_force.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/initializers/line.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/initializers/spheres.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/initializers/hexagon.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/initializers/cuboid.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/initializers/rectangle.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/initializers/disk.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/initializers/heart.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/pair_finders/naive_all_pairs.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pair_finders/naive_all_pairs_symmetric.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pair_finders/binned_all_pairs.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pair_finders/binned_all_pairs_symmetric.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pair_finders/naive_gabriel.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pair_finders/naive_gabriel_symmetric.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pair_finders/binned_gabriel.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pair_finders/binned_gabriel_symmetric.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pair_finders/naive_delaunay.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/integrators/base.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/integrators/euler.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/integrators/heun.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/com_fixers/com_fixers.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/simulation_config.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/simulation.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/io/dummy.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/io/hdf5_writer.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/io/hdf5_reader.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/functions.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/grid.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/argparse/arguments.hpp
)



function(add_TECS_simulation target_name source_file)
  add_executable(${target_name} ${source_file})

  target_link_libraries(${target_name} PRIVATE
    TECS_core
  )

  # ccache: wrap CXX compiler for faster incremental rebuilds
  # for CUDA builds with Kokkos, the RULE_LAUNCH_COMPILE chain is managed
  # globally by Kokkos (kokkos_launch_compiler → nvcc_wrapper → nvcc)
  if(CCACHE_PROGRAM AND NOT CMAKE_CUDA_COMPILER)
    set_target_properties(${target_name} PROPERTIES
      RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}"
      RULE_LAUNCH_LINK "${CCACHE_PROGRAM}"
    )
  endif()

  # Debug: no opt + full debug symbols
  # Release: -O3
  target_compile_options(${target_name} PRIVATE
    $<$<CONFIG:Debug>:-O0;-g3;-ggdb>
    $<$<CONFIG:Release>:-O3>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native;-mtune=native>
  )

  if(CMAKE_CUDA_COMPILER)
    target_compile_options(${target_name} PRIVATE
      $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:SHELL:--use_fast_math>
    )
  endif()
endfunction()

set(TECS_USER_TARGET "tecs"
  CACHE STRING "Name of the generated user simulation target"
)

set(TECS_USER_MAIN ""
  CACHE FILEPATH "Path to a user-defined simulation source file"
)

if (TECS_USER_MAIN)
  message(STATUS
    "Configuring user simulation from ${TECS_USER_MAIN} "
    "as target ${TECS_USER_TARGET}"
  )
  add_TECS_simulation(${TECS_USER_TARGET} "${TECS_USER_MAIN}")
endif()

option(TECS_BUILD_TESTS "Build tests" OFF)
if (TECS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
