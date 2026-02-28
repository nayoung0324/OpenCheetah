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
  common/conv.cpp
  cheetah/conv_cheetah.cpp
  mod2k/hom_conv2d_mod2k.cpp
  common/util.cpp
  mod2k/encryption.cpp
  mod2k/decryption.cpp
)
target_link_libraries(mini_seal PRIVATE SEAL::seal)
target_include_directories(mini_seal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(mini_seal_compare
  compare_with_seal_main.cpp
  common/util.cpp
  mod2k/encryption.cpp
  mod2k/decryption.cpp
)
target_link_libraries(mini_seal_compare PRIVATE SEAL::seal)
target_include_directories(mini_seal_compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(mini_seal_functionality
  operation_funcationality_test_main.cpp
  common/util.cpp
  mod2k/encryption.cpp
  mod2k/decryption.cpp
)
target_link_libraries(mini_seal_functionality PRIVATE SEAL::seal)
target_include_directories(mini_seal_functionality PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
