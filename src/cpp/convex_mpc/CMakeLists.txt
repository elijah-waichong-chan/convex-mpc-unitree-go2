cmake_minimum_required(VERSION 3.16)
project(convex_mpc_go2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(convex_mpc_go2
  main.cpp
  centroidal_mpc_acados.cpp
  go2_pinocchio_model.cpp
)

add_subdirectory(c_generated_code/centroidal_mpc_hpipm)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pinocchio REQUIRED)
target_link_libraries(convex_mpc_go2 
PRIVATE 
  acados_solver_centroidal_mpc_hpipm
  Eigen3::Eigen
  pinocchio::pinocchio
  )

target_include_directories(convex_mpc_go2 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code/centroidal_mpc_hpipm
  ${CMAKE_CURRENT_SOURCE_DIR}  # local headers
)