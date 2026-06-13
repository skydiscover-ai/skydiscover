cmake_minimum_required(VERSION 3.12)
project(kvstore_unit_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# Unit benchmark harness (1000 keys, 2s run)
add_executable(unit_bench
    unit_bench_harness.cc
    trivial_kvstore.cc
)
target_include_directories(unit_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../../interface)
target_link_libraries(unit_bench pthread)

# Unit consistency harness (500 keys, 2s run)
add_executable(unit_consistency
    unit_consistency_harness.cc
    trivial_kvstore.cc
)
target_include_directories(unit_consistency PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../../interface)
target_link_libraries(unit_consistency pthread)
