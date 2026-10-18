cmake_minimum_required(VERSION 3.18)
project(hypergamic VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ── dependencies ──────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)

# pybind11 installed via pip lives outside CMake's default search path, so ask it
# where its config is rather than requiring -Dpybind11_DIR on every configure.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_pybind11_cmakedir)
        list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR
            "pybind11 not found. Install it with: ${Python3_EXECUTABLE} -m pip install pybind11")
    endif()
endif()

if(Python3_VERSION VERSION_LESS 3.9)
    message(FATAL_ERROR "Python 3.9+ required, found ${Python3_VERSION}")
endif()

find_package(OpenMP QUIET)

# ── compiler flags (MSVC vs GCC/Clang) ────────────────────────────────────────
if(MSVC)
    add_compile_options(/O2 /W4 /permissive- /utf-8 /EHsc)
else()
    add_compile_options(-O3 -Wall -Wextra -Wpedantic)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

# ── core library ──────────────────────────────────────────────────────────────
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS src/cpp/*.cpp)
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*bindings\\.cpp$")

add_library(hypergamic_core STATIC ${CORE_SOURCES})
target_include_directories(hypergamic_core PUBLIC include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hypergamic_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── pybind11 extension ────────────────────────────────────────────────────────
pybind11_add_module(_hypergamic MODULE src/cpp/bindings.cpp)
target_link_libraries(_hypergamic PRIVATE hypergamic_core)
target_include_directories(_hypergamic PRIVATE include)

# ── tests ─────────────────────────────────────────────────────────────────────
enable_testing()
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_subdirectory(tests/cpp)
