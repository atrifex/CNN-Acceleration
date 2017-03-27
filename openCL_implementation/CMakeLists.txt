
cmake_minimum_required (VERSION 3.1)

if (POLICY CMP0007)
    # NEW = list command no longer ignores empty elements.
    cmake_policy(SET CMP0007 OLD)
endif(POLICY CMP0007)
cmake_policy(SET CMP0054 OLD)
if (POLICY CMP0025)
    cmake_policy(SET CMP0025 OLD) # report Apple's Clang as just Clang
endif (POLICY CMP0025)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CONFIG_USE_HUNTER "ON" CACHE BOOL "Turn on to enable using the hunteger package manager")

if (CONFIG_USE_HUNTER)
    include("cmake/modules/HunterGate.cmake")
    HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.16.24.tar.gz"
        SHA1 "00ab627d3d4fe5151e15b6ab2b1b1da17650984a"
    )
endif (CONFIG_USE_HUNTER)

project(OpenCL_CNN)

include(CMakeToolsHelpers OPTIONAL)

if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif (UNIX AND NOT APPLE)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})
set(TOP_DIR "${PROJECT_SOURCE_DIR}")
set(SRC_DIR "${TOP_DIR}/src")


if (CONFIG_USE_HUNTER)
    # now hunter can be used, like
    if (NOT CONFIG_USE_SYSTEM_HDF5)
      # hunter_add_package(ZLIB)
      hunter_add_package(hdf5)
      # hunter_add_package(szip)
    endif (NOT CONFIG_USE_SYSTEM_HDF5)
endif (CONFIG_USE_HUNTER)

### Find dependencies ###
Find_Package(EnableCxx11 REQUIRED)
Find_Package(ZLIB REQUIRED)
if (CONFIG_USE_HUNTER)
    Find_Package(szip REQUIRED)
    Find_Package(HDF5 CONFIG REQUIRED)
else (CONFIG_USE_HUNTER)
    Find_Package(HDF5 REQUIRED)
endif (CONFIG_USE_HUNTER)

find_package( OpenCL REQUIRED )

### Add source files ###
include("${SRC_DIR}/config.cmake")

### Setup target ###
# Set warning levels for host compilation
if (UNIX)
    add_compile_options(-Wextra)
    add_compile_options(-Wno-unused-function)
    add_compile_options(-Wno-unused-parameter)
    add_compile_options(-Wno-unused-parameter)
    add_compile_options(-Wno-vla)
    add_compile_options(-Wno-unused-variable)
endif (UNIX)


include_directories(
    "${ZLIB_INCLUDE_DIRS}"
    "${SRC_DIR}"
)

if (NOT "${HDF5_INCLUDE_DIR}" STREQUAL "")
    include_directories("${HDF5_INCLUDE_DIR}")
endif (NOT "${HDF5_INCLUDE_DIR}" STREQUAL "")

set(PROJECT_TARGET openCL_CNN)

add_executable(${PROJECT_TARGET} ${SOURCES})

target_link_libraries(${PROJECT_TARGET}
    ${HDF5_LIBRARIES}
    ${ZLIB_LIBRARIES}
)

# Add a special target to clean nvcc generated files.
