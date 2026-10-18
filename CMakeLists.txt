cmake_minimum_required(VERSION 3.18)
project(hypergamic VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── dependencies ──────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

# ── compile options ───────────────────────────────────────────────────────────
add_compile_options(-O3 -march=native -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

# ── core library ──────────────────────────────────────────────────────────────
file(GLOB CORE_SOURCES src/cpp/*.cpp)
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*bindings\\.cpp$")

add_library(hypergamic_core STATIC ${CORE_SOURCES})
target_include_directories(hypergamic_core PUBLIC include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hypergamic_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── pybind11 extension ────────────────────────────────────────────────────────
pybind11_add_module(_hypergamic src/cpp/bindings.cpp)
target_link_libraries(_hypergamic PRIVATE hypergamic_core)
target_include_directories(_hypergamic PRIVATE include)

# ── tests ─────────────────────────────────────────────────────────────────────
enable_testing()
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

add_subdirectory(tests/cpp)
