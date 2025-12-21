cmake_minimum_required(VERSION 3.16)
project(convex_mpc_go2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(convex_mpc_go2
  main.cpp
  centroidal_mpc.cpp
)

# Local headers (for "centroidal_mpc.hpp")
target_include_directories(convex_mpc_go2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# CasADi (found via conda prefix)
find_package(casadi CONFIG REQUIRED)
target_link_libraries(convex_mpc_go2 PRIVATE casadi)