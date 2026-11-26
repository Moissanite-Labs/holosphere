cmake_minimum_required(VERSION 3.16)
project(holosphere_razor_abi VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(razor_abi_host_verify
    holosphere/kernels/csrc/razor_abi_host_verify.c
)
target_include_directories(razor_abi_host_verify PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
