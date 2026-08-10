#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 TARGET_DIR LLAMA_CPP_ROOT" >&2
    exit 2
fi

target_dir="$1"
llama_root="$2"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${BASE_DIR:-$root/build}/bonsai-runtime"
host="${HOST_DIR:-$root/sdk/buildroot/output/host}"
model="${BONSAI_MODEL_SOURCE:-$root/../../../Bonsai-1.7B.gguf}"
manifest="$build/bonsai_model_manifest.h"
accelerator_object="$build/bonsai_accelerator.o"
matmul_object="$build/bonsai_matmul.o"
host_python="${BONSAI_HOST_PYTHON:-/usr/bin/python3}"
target_gcc="$host/bin/riscv64-buildroot-linux-gnu-gcc"
target_gxx="$host/bin/riscv64-buildroot-linux-gnu-g++"
compiler_gcc="$root/tools/riscv64f-gcc"
compiler_gxx="$root/tools/riscv64f-g++"
target_march="${BONSAI_MARCH:-rv64imafc_zicsr_zifencei_zba_zbb_zbc_zbs_zicntr_zcb_zihpm}"
target_mabi="${BONSAI_MABI:-lp64}"
target_mtune="${BONSAI_MTUNE:-sifive-7-series}"

if [[ ! -f "$llama_root/CMakeLists.txt" ]]; then
    echo "llama.cpp source is missing at $llama_root" >&2
    exit 1
fi
if [[ ! -f "$model" ]]; then
    echo "Bonsai model is missing at $model" >&2
    exit 1
fi
if [[ ! -x "$host_python" ]]; then
    echo "Host Python is missing at $host_python" >&2
    exit 1
fi

mkdir -p "$build"
PYTHONPATH="$llama_root/gguf-py" "$host_python" "$root/tools/build_bonsai_manifest.py" \
    "$model" "$manifest"
"$target_gcc" \
    -O3 -Wall -Wextra -Werror -std=gnu11 -fPIC -pthread \
    "-march=$target_march" "-mabi=$target_mabi" "-mtune=$target_mtune" \
    -I"$build" -c "$root/software/bonsai_accelerator.c" \
    -o "$accelerator_object"
"$target_gcc" \
    -O3 -Wall -Wextra -Werror -Wno-unused-function -std=gnu11 -fPIC \
    "-march=$target_march" "-mabi=$target_mabi" "-mtune=$target_mtune" \
    -I"$llama_root/ggml/include" -I"$llama_root/ggml/src" \
    -I"$llama_root/ggml/src/ggml-cpu" \
    -c "$root/software/bonsai_matmul.c" -o "$matmul_object"

export CVA6_RISCV_GCC="$target_gcc"
export CVA6_RISCV_GXX="$target_gxx"
export CVA6_RISCV_MARCH="$target_march"
export CVA6_RISCV_MABI="$target_mabi"

cmake -S "$llama_root" -B "$build" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER="$compiler_gcc" \
    -DCMAKE_CXX_COMPILER="$compiler_gxx" \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--wrap=ggml_vec_dot_q1_0_q8_0 $accelerator_object $matmul_object" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_CCACHE=OFF \
    -DGGML_CPU_REPACK=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_RVV=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_LLGUIDANCE=OFF \
    -DLLAMA_OPENSSL=OFF
# The accelerator objects are supplied through linker flags rather than a
# CMake target, so Ninja/Make cannot infer that changing them invalidates the
# existing executable. Remove only the generated final link output to force a
# cheap relink while retaining all compiled llama.cpp objects.
rm -f "$build/bin/llama-cli" "$build/bin/llama-completion"
cmake --build "$build" --target llama-completion -j"$(nproc)"
install -m 0755 "$build/bin/llama-completion" "$target_dir/usr/bin/bonsai-runtime"
"$host/bin/riscv64-buildroot-linux-gnu-strip" "$target_dir/usr/bin/bonsai-runtime"

# Slim custom decoder: all Q1_0 matvecs through the whole-matrix accelerator
# interface, everything else in tight scalar C with a small worker pool.
"$target_gcc" \
    -O3 -Wall -Wextra -std=gnu11 -pthread \
    "-march=$target_march" "-mabi=$target_mabi" "-mtune=$target_mtune" \
    -I"$build" "$root/software/bonsai_main.c" "$accelerator_object" \
    -lm -o "$build/bonsai-lite"
install -m 0755 "$build/bonsai-lite" "$target_dir/usr/bin/bonsai-lite"
"$host/bin/riscv64-buildroot-linux-gnu-strip" "$target_dir/usr/bin/bonsai-lite"
