cmake_minimum_required(VERSION 3.16)
project(tiny_autodiff
  LANGUAGES CXX
  VERSION 0.1.0
  DESCRIPTION "Tiny reverse-mode autodiff engine in C++"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Header-only interface library ---
add_library(tiny_autodiff INTERFACE)
target_include_directories(tiny_autodiff INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(tiny_autodiff INTERFACE cxx_std_17)

# --- Examples ---
add_executable(basic examples/basic.cpp)
target_link_libraries(basic PRIVATE tiny_autodiff)

# --- Warnings ---
if(MSVC)
  target_compile_options(basic PRIVATE /W4)
else()
  target_compile_options(basic PRIVATE -Wall -Wextra -Wpedantic)
endif()
