# mini_seal/CMakeLists.txt
cmake_minimum_required(VERSION 3.13)
project(mini_seal CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCheetah deps prefix (? repo ?? build/)
# cmake -DCMAKE_PREFIX_PATH=... ? ???? ?
find_package(SEAL 3.7 REQUIRED)

add_executable(mini_seal
  main.cpp
  util.cpp
  encryption.cpp
  decryption.cpp
)
target_link_libraries(mini_seal PRIVATE SEAL::seal)

add_executable(mini_seal_functionality
  operation_funcationality_test_main.cpp
  util.cpp
  encryption.cpp
  decryption.cpp
)
target_link_libraries(mini_seal_functionality PRIVATE SEAL::seal)
