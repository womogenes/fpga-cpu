#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 TARGET_DIR STAGING_DIR HOST_DIR" >&2
    exit 2
fi

target="$1"
staging="$2"
host="$3"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit=2be0218be4461f9e453cbdcb98f81f41a5ae8bed
archive="$root/build/downloads/tinycc-${commit:0:7}.tar.gz"
source="$root/build/tinycc-source"
build="$root/build/tinycc-riscv64"
url="https://codeload.github.com/Tiny-C-Compiler/tinycc-mirror-repository/tar.gz/$commit"
expected_sha256=95dc2ffe7d4b561bec56ced00c7155f04949679c34c7d1367bef532ac8088f57
cross="$host/bin/riscv64-buildroot-linux-gnu-"

mkdir -p "$(dirname "$archive")" "$source" "$build"
if [[ ! -f "$archive" ]]; then
    curl -L --fail --retry 3 --connect-timeout 10 -o "$archive" "$url"
fi
echo "$expected_sha256  $archive" | sha256sum -c -
if [[ ! -x "$source/configure" ]]; then
    tar -xf "$archive" -C "$source" --strip-components=1
fi

cd "$build"
"$source/configure" \
    --prefix=/usr \
    --cpu=riscv64 \
    --cc=gcc \
    --ar=ar \
    --cross-prefix="$cross" \
    --sysincludepaths=/usr/include \
    --libpaths=/usr/lib:/lib \
    --crtprefix=/usr/lib \
    --elfinterp=/lib/ld-linux-riscv64-lp64.so.1 \
    --config-pie

cc -DC2STR "$source/conftest.c" -o c2str.exe
./c2str.exe "$source/include/tccdefs.h" tccdefs_.h
make -j"$(nproc)" tcc libtcc.a

mkdir -p lib
cd lib
runtime_c=(lib-arm64.c stdatomic.c builtin.c dsohandle.c)
runtime_asm=(atomic.S alloca.S alloca-bt.S)
runtime_flags=(-O2 -fPIC -fno-omit-frame-pointer -Wno-unused-function -Wno-unused-variable -I..)
for unit in "${runtime_c[@]}"; do
    "${cross}gcc" -c "$source/lib/$unit" -o "${unit%.c}.o" "${runtime_flags[@]}"
done
for unit in "${runtime_asm[@]}"; do
    "${cross}gcc" -c "$source/lib/$unit" -o "${unit%.S}.o" "${runtime_flags[@]}"
done
"${cross}gcc" -c "$root/software/tinycc_clear_cache.c" -o armflush.o "${runtime_flags[@]}"
"${cross}ar" rcs ../libtcc1.a lib-arm64.o stdatomic.o atomic.o builtin.o alloca.o alloca-bt.o armflush.o dsohandle.o
cd ..

make DESTDIR="$target" install
"${cross}strip" "$target/usr/bin/tcc"
ln -snf tcc "$target/usr/bin/cc"

mkdir -p "$target/usr/include" "$target/usr/lib"
# This TinyCC revision installs its compiler-private headers and runtime in
# /usr/lib/tcc, but its configured RISC-V search path only probes /usr/include
# and /usr/lib.  Put the small compiler-owned pieces on those search paths so
# an ordinary `cc hello.c -o hello` works without project-specific flags.
cp -a "$target/usr/lib/tcc/include/." "$target/usr/include/"
ln -snf tcc/libtcc1.a "$target/usr/lib/libtcc1.a"
cp -a "$staging/usr/include/." "$target/usr/include/"
find "$staging/usr/lib" -maxdepth 1 -type f \( -name '*.a' -o -name '*.so' -o -name 'crt*.o' -o -name 'Scrt1.o' \) -exec cp -a -t "$target/usr/lib" {} +

echo "Installed TinyCC and target development headers."
