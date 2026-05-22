cmake_minimum_required(VERSION 3.22)

project(bigboyagent_brain VERSION 12.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # enforce -std=c++20, not -std=gnu++20

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    main.cpp
    config.cpp
    simd_math.cpp
    feature_engine.cpp
    regime_engine.cpp
    signal_engine.cpp
    risk_engine.cpp
    ranker.cpp
    alpha_engine.cpp
    tracking.cpp
    http_server.cpp
)

add_executable(brain ${SOURCES})

target_include_directories(brain PRIVATE ${CMAKE_SOURCE_DIR})

# ── Compiler flags ────────────────────────────────────────────────────────────
target_compile_options(brain PRIVATE
    # Release optimisation
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>

   
    # -march=x86-64-v3 = AVX2 + FMA + BMI2 — portable cross-Xeon binary.
    # Safer than -march=native in Docker multi-stage builds: the builder
    # container CPU may differ from the runtime container CPU.
    #
    # BUG-CM1 FIX: added -DTQC_AVX2 definition so #ifdef TQC_AVX2 guards in
    # simd_math.hpp actually activate the AVX2 code paths.
    # Without this define, all SIMD kernels silently fell through to the scalar
    # fallback (#else branch) even though -mavx2 was passed to the compiler.
    # The binary was valid and correct, but 8× slower than intended on every
    # indicator computation — the entire point of the SIMD architecture was lost.
    $<$<CONFIG:Release>:-DTQC_AVX2>
    $<$<CONFIG:Debug>:-DTQC_AVX2>

    -march=x86-64-v3
    -mavx2
    -mfma

    # Auto-vectorisation helpers
    -funroll-loops
    -fomit-frame-pointer
    -fprefetch-loop-arrays

    # Safe float optimisations (NOT -ffast-math — breaks AVX NaN checks and
    # changes semantics of std::isnan/std::isinf used in updateHistory() and
    # JsonBuilder::num()).
    -fno-math-errno
    -fno-trapping-math

    # Hardening — _FORTIFY_SOURCE requires at least -O1 to activate.
    # Only enabled in Release (which uses -O3) to avoid the undefined-behaviour
    # of combining -O0 with _FORTIFY_SOURCE=2 (GCC warns about this combination).
    $<$<CONFIG:Release>:-D_FORTIFY_SOURCE=2>
    -fstack-protector-strong

    # Debug build — AddressSanitizer + UBSan.
    # These catch every class of bug fixed in this audit:
    #   ASAN: stack overflows (BUG-AE1), use-after-free (BUG-OP1), buffer overruns
    #   UBSan: signed overflow, bad casts, misaligned access, invalid bool
    # -O1 is the minimum that keeps ASAN functional and stack traces readable.
    $<$<CONFIG:Debug>:-O1>
    $<$<CONFIG:Debug>:-g3>
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>

    # BUG-CM2 FIX: added -Wconversion and -Wunused-variable to the warning set.
    # -Wconversion catches BUG-RK5 class bugs (int passed to float-only function)
    #   and any future implicit narrowing conversions that could silently truncate.
    # -Wunused-variable catches BUG-M1 class bugs (dead variables like open_count)
    #   that signal logic errors at declaration rather than at silent runtime misuse.
    # -Werror promotes all warnings to errors — forces CI to catch them immediately.
    -Wall
    -Wextra
    -Wshadow
    -Wconversion        # BUG-CM2 FIX: catch int→float implicit narrowing
    -Wunused-variable   # BUG-CM2 FIX: catch dead variables like open_count
    -Wno-unused-parameter
    -Werror             # treat all warnings as errors in CI
)

target_link_options(brain PRIVATE
    $<$<CONFIG:Release>:-flto=auto>
    $<$<CONFIG:Release>:-s>           # strip debug symbols from release binary
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

# std::jthread (C++20) and std::atomic are part of libstdc++; pthread is the
# underlying thread library on Linux.  Required for PositionManager::saver_
# (std::jthread) and all atomic_flag spinlocks.
target_link_libraries(brain PRIVATE pthread)

# ── VPS / local build convenience target ──────────────────────────────────────
# Usage: cmake --build build --target brain_debug
# Builds with ASAN + UBSan for local development and VPS debugging.
# Do NOT deploy the debug binary to production — ASAN adds ~2× memory overhead
# and ~10× CPU overhead per sanitised allocation.
add_custom_target(brain_debug
    COMMAND ${CMAKE_COMMAND} -E echo "Build with: cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build_debug"
    VERBATIM
)

message(STATUS "──────────────────────────────────────────────────")
message(STATUS "Project:   TQC-Brain v12 | Taha Iqbal")
message(STATUS "Build:     ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:  ${CMAKE_CXX_COMPILER}")
message(STATUS "Standard:  C++20 (std::jthread, std::bit_cast, std::span)")
message(STATUS "Arch:      x86-64-v3 (AVX2 + FMA + BMI2)")
message(STATUS "SIMD:      DTQC_AVX2 defined — AVX2 code paths active")
message(STATUS "──────────────────────────────────────────────────")
