cmake_minimum_required(VERSION 3.16)
project(centroidal_mpc_hpipm_codegen C)

# Resolve ACADOS_SOURCE_DIR: env var first, then repo-local fallback.
if(DEFINED ENV{ACADOS_SOURCE_DIR})
  set(ACADOS_SOURCE_DIR $ENV{ACADOS_SOURCE_DIR})
else()
  get_filename_component(CPP_DIR   "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE) # convex_mpc/cpp
  get_filename_component(REPO_ROOT "${CPP_DIR}/../.." ABSOLUTE)                   # repo root
  set(ACADOS_SOURCE_DIR "${REPO_ROOT}/third_party/acados")
endif()

message(STATUS "Using ACADOS_SOURCE_DIR=${ACADOS_SOURCE_DIR}")

# Optional sanity check
if(NOT EXISTS "${ACADOS_SOURCE_DIR}/include/acados")
  message(FATAL_ERROR "ACADOS_SOURCE_DIR looks wrong: ${ACADOS_SOURCE_DIR}")
endif()

# generated sources (exclude the example main_*.c)
file(GLOB GEN_TOP_C   "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
list(FILTER GEN_TOP_C EXCLUDE REGEX "main_.*\\.c$")

file(GLOB GEN_MODEL_C "${CMAKE_CURRENT_SOURCE_DIR}/centroidal_mpc_hpipm_model/*.c")

add_library(acados_solver_centroidal_mpc_hpipm STATIC
  ${GEN_TOP_C}
  ${GEN_MODEL_C}
)

target_include_directories(acados_solver_centroidal_mpc_hpipm PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/centroidal_mpc_hpipm_model

  # acados + deps headers
  ${ACADOS_SOURCE_DIR}/include
  ${ACADOS_SOURCE_DIR}/include/hpipm/include
  ${ACADOS_SOURCE_DIR}/include/blasfeo/include
)

target_link_directories(acados_solver_centroidal_mpc_hpipm PUBLIC
  ${ACADOS_SOURCE_DIR}/lib
)

target_link_libraries(acados_solver_centroidal_mpc_hpipm PUBLIC
  acados
  hpipm
  blasfeo
  m
  dl
)
