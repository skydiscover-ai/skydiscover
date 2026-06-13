cmake_minimum_required(VERSION 3.12)
project(kvstore_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto=auto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto=auto")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")

# ── Language selection: cpp (default) or rust ─────────────────────────────────
set(KVSTORE_LANG "cpp" CACHE STRING "Implementation language: cpp or rust")
message(STATUS "KVSTORE_LANG=${KVSTORE_LANG}")

if(KVSTORE_LANG STREQUAL "rust")
    # ── Rust mode: C++ shim + Rust static library ────────────────────────────
    set(RUST_CRATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated/kvstore_rust")
    set(RUST_LIB "${RUST_CRATE_DIR}/target/release/libkvstore_rust.a")

    add_custom_command(
        OUTPUT  ${RUST_LIB}
        COMMAND ${CMAKE_COMMAND} -E env "PATH=$ENV{HOME}/.cargo/bin:$ENV{PATH}"
                cargo build --release
        WORKING_DIRECTORY ${RUST_CRATE_DIR}
        COMMENT "Building Rust KV store (cargo build --release)"
        VERBATIM
    )
    add_custom_target(kvstore_rust_lib DEPENDS ${RUST_LIB})

    set(IMPL_SOURCE rust_shim.cc)
    set(EXTRA_LIBS ${RUST_LIB} dl)
    set(EXTRA_DEPS kvstore_rust_lib)
else()
    # ── C++ mode (default) ────────────────────────────────────────────────────
    set(IMPL_SOURCE generated/kvstore_impl.cc)
    set(EXTRA_LIBS "")
    set(EXTRA_DEPS "")
endif()

# ── Optional: FASTER seed integration ─────────────────────────────────────────
# If the vendored FASTER source tree is present, compile it alongside whatever
# kvstore_impl.cc the workspace currently has. Priority:
#   (1) workspace-local copy at  ./generated/faster_src/   (agent-editable; set
#       up by orchestrator when --seed faster is active so agent can freely
#       modify FASTER internals),
#   (2) repo-level fallback at   setups/faster-single-machine/seeds_faster/
#       via a 5-level walk-up (for builds that don't use --seed).
# No effect when neither exists; no effect on impls that don't #include
# "core/faster.h".
set(FASTER_SEED_DIR "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/faster_src/core/faster.h")
    set(FASTER_SEED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")
else()
    set(_scan "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach(_i RANGE 0 5)
        if(EXISTS "${_scan}/setups/faster-single-machine/seeds_faster/faster_src/core/faster.h")
            set(FASTER_SEED_DIR "${_scan}/setups/faster-single-machine/seeds_faster")
            break()
        endif()
        get_filename_component(_scan "${_scan}" DIRECTORY)
    endforeach()
endif()
if(FASTER_SEED_DIR)
    message(STATUS "FASTER seed found at ${FASTER_SEED_DIR} — integrating")
    include_directories("${FASTER_SEED_DIR}/faster_src")
    # Max payload bytes per record in the FASTER seed's Value class.
    # Record size on disk = KVSTORE_MAX_PAYLOAD + 8 (size + pad header).
    # Default 100 → 108 B records (matches SIGMOD Fig 10's "100 B payloads,
    # 27 GB dataset" at 250M keys). Override to 200 for bimodal workloads
    # (200 B large value class) or to 4088 for 4 KB large-value workloads.
    # The native FASTER benchmark has the same knob (FASTER_VALUE_PADDING);
    # keep them synchronized (native_padding + 8 = our_record_size).
    set(KVSTORE_MAX_PAYLOAD "100" CACHE STRING
        "Max payload bytes per record in the FASTER seed (default: 100, paper Fig 10)")
    message(STATUS "KVSTORE_MAX_PAYLOAD=${KVSTORE_MAX_PAYLOAD} (record size = ${KVSTORE_MAX_PAYLOAD}+8 B)")
    add_compile_definitions(KVSTORE_MAX_PAYLOAD=${KVSTORE_MAX_PAYLOAD})
    set(FASTER_SOURCES
        "${FASTER_SEED_DIR}/faster_src/core/lss_allocator.cc"
        "${FASTER_SEED_DIR}/faster_src/core/address.cc"
        "${FASTER_SEED_DIR}/faster_src/core/thread.cc"
        "${FASTER_SEED_DIR}/faster_src/environment/file_linux.cc"
    )
    list(APPEND EXTRA_LIBS stdc++fs uuid tbb aio)
else()
    set(FASTER_SOURCES "")
endif()

# ── Benchmark harness ─────────────────────────────────────────────────────────
add_executable(kvstore_bench
    benchmark_harness.cc
    ${IMPL_SOURCE}
    ${FASTER_SOURCES}
)
target_include_directories(kvstore_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kvstore_bench pthread ${EXTRA_LIBS})
if(EXTRA_DEPS)
    add_dependencies(kvstore_bench ${EXTRA_DEPS})
endif()

# ── Consistency test harness ──────────────────────────────────────────────────
add_executable(consistency_test
    consistency_harness.cc
    ${IMPL_SOURCE}
    ${FASTER_SOURCES}
)
target_include_directories(consistency_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(consistency_test pthread ${EXTRA_LIBS})
if(EXTRA_DEPS)
    add_dependencies(consistency_test ${EXTRA_DEPS})
endif()

# ── Optional: AddressSanitizer in Debug ───────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(kvstore_bench PRIVATE -fsanitize=address,undefined)
    target_link_options(kvstore_bench PRIVATE -fsanitize=address,undefined)
endif()
