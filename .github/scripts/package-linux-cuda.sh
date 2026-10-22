#!/usr/bin/env bash
# Package the CUDA build of shears-cli (release, --static-swift-stdlib) into a relocatable tarball:
#   shears-cli-linux-x86_64-cuda/{bin/shears-cli, lib/*.so*, BUILD_INFO.txt, NVIDIA_LIBRARIES.txt, LICENSE, ...}
# Like package-linux.sh, the CPU libraries (OpenBLAS, LAPACK, gfortran, libstdc++, libgcc) are copied into lib/
# and the RUNPATHs rewritten. Unlike it, NO NVIDIA LIBRARY IS BUNDLED: the CUDA runtime, cuBLAS, cuBLASLt, NVRTC and
# cuDNN stay external (Colab's GPU image has them; the driver's libcuda.so.1 always comes from the machine). They are
# listed in NVIDIA_LIBRARIES.txt and in BUILD_INFO.txt. Writes dist/shears-cli-linux-x86_64-cuda.tar.gz and .sha256.
#
# This build has never been run on a GPU. The package step only proves that the binary links and starts.
set -euo pipefail

BIN="${1:-CLI/.build/release/shears-cli}"
NAME=shears-cli-linux-x86_64-cuda
DIST="$PWD/dist"
PKG="$DIST/$NAME"
rm -rf "$DIST"; mkdir -p "$PKG/bin" "$PKG/lib"; : > "$DIST/bundled.txt"

# Libraries that come with glibc: never bundled.
SYSTEM='^(linux-vdso|ld-linux-x86-64|libc|libm|libdl|libpthread|librt|libutil|libresolv|libanl|libnss_[a-z]+)\.so'
# NVIDIA libraries: never bundled, supplied by the machine (driver) or by the CUDA toolkit / pip wheels.
NVIDIA='^(libcuda|libcudart|libcublas|libcublasLt|libnvrtc|libnvrtc-builtins|libnvJitLink|libcudnn[a-z_]*|libnvidia-[a-z-]+|libcufft|libcusparse|libcusolver|libcurand|libnccl)\.so'

cp "$BIN" "$PKG/bin/shears-cli"
strip --strip-unneeded "$PKG/bin/shears-cli"

# What the binary asks for directly, in the order the linker recorded it.
echo "NEEDED entries of the binary:"
readelf -d "$PKG/bin/shears-cli" | awk '/NEEDED/{gsub(/[\[\]]/,"",$NF); print $NF}' | tee "$DIST/needed.txt"

# The NVIDIA ones are the contract with the machine it runs on.
grep -E "$NVIDIA" "$DIST/needed.txt" > "$PKG/NVIDIA_LIBRARIES.txt" || true
if ! grep -q 'libcudart' "$PKG/NVIDIA_LIBRARIES.txt"; then
  echo "::error::the binary does not link libcudart: this is not a CUDA build (SPM_CUDA was probably 0)"; exit 1
fi

echo "ldd of the build, before bundling (libcuda.so.1 may show as not found on a runner without a driver):"
ldd "$PKG/bin/shears-cli" || true

# ldd lists the transitive closure: "soname => /path (addr)".
ldd "$PKG/bin/shears-cli" | awk '/=> \//{print $1, $3}' | while read -r soname path; do
  if [[ "$soname" =~ $SYSTEM ]]; then continue; fi
  if [[ "$soname" =~ $NVIDIA ]]; then continue; fi
  cp -L "$path" "$PKG/lib/$soname"
  echo "$soname $path" >> "$DIST/bundled.txt"
  echo "bundled $soname <- $path"
done
chmod u+w "$PKG"/lib/* || true

if ls "$PKG/lib/" | grep -qE '^lib(ssl|crypto|curl)\.'; then
  echo "::warning::libcurl/OpenSSL ended up in the bundle; check whether FoundationNetworking is linked"
fi
if ls "$PKG/lib/" | grep -qiE 'cuda|cublas|cudnn|nvrtc|nvJitLink'; then
  echo "::error::an NVIDIA library ended up in the bundle"; ls "$PKG/lib"; exit 1
fi

patchelf --set-rpath '$ORIGIN/../lib' "$PKG/bin/shears-cli"
for f in "$PKG"/lib/*; do [ -e "$f" ] && patchelf --set-rpath '$ORIGIN' "$f"; done

# Resources next to the binary (SwiftPM *.resources / *.bundle), if any.
for r in "$(dirname "$BIN")"/*.resources "$(dirname "$BIN")"/*.bundle; do
  [ -e "$r" ] && { cp -r "$r" "$PKG/bin/"; echo "copied resource bundle $r"; }
done

# Verify with a stand-in libcuda.so.1 (the CUDA toolkit's link stub; a runner has no driver): nothing may be unresolved
# apart from the NVIDIA libraries, and every bundled soname must resolve into the bundle.
STUBDIR=$(mktemp -d)
STUB=$(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda*/targets/x86_64-linux/lib/stubs/libcuda.so 2>/dev/null | head -1 || true)
[ -n "$STUB" ] && ln -s "$STUB" "$STUBDIR/libcuda.so.1"
echo "ldd of the bundle (no LD_LIBRARY_PATH; libcuda.so.1 via the toolkit stub $STUB):"
LD_LIBRARY_PATH="$STUBDIR" ldd "$PKG/bin/shears-cli" | tee "$DIST/ldd.txt"
if grep "not found" "$DIST/ldd.txt" | grep -vE "$NVIDIA" | grep -q .; then echo "unresolved libraries"; exit 1; fi
if LD_LIBRARY_PATH="$STUBDIR" ldd "$PKG/bin/shears-cli" | awk '/=> \//{print $1, $3}' \
    | while read -r so path; do [[ "$so" =~ $SYSTEM ]] || [[ "$so" =~ $NVIDIA ]] || [[ "$path" == "$PKG/bin/../lib/"* ]] || echo "$so resolves outside the bundle: $path"; done | grep .; then
  exit 1
fi
# Starts at all? (--help touches no GPU.) With the stub only, to learn whether MLX initialises CUDA at start-up.
LD_LIBRARY_PATH="$STUBDIR" "$PKG/bin/shears-cli" --help >/dev/null

# Oldest glibc this binary can run on.
GLIBC_NEEDED=$(objdump -T "$PKG/bin/shears-cli" "$PKG"/lib/* 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
{
  echo "shears-cli for Linux x86_64 (CUDA, MLX), Swift runtime linked statically. EXPERIMENTAL: never run on a GPU."
  echo "commit:        ${GITHUB_SHA:-$(git rev-parse HEAD)}"
  echo "built:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "swift:         $(swift --version 2>&1 | head -1)"
  echo "build os:      $(. /etc/os-release; echo "$PRETTY_NAME"), glibc $(ldd --version | head -1 | awk '{print $NF}')"
  echo "cuda toolkit:  $(/usr/local/cuda/bin/nvcc --version | tail -2 | head -1)"
  echo "gpu archs:     ${SHEARS_CUDA_ARCHS:-unknown} (device code for these compute capabilities; sm_ = compute capability without the dot)"
  echo "needs glibc:   >= ${GLIBC_NEEDED#GLIBC_}"
  echo "run time:      g++ must be installed (MLX compiles fused CPU kernels with it); an NVIDIA driver (libcuda.so.1)"
  echo "NVIDIA libs (NOT bundled, must be on the library path): $(tr '\n' ' ' < "$PKG/NVIDIA_LIBRARIES.txt")"
  echo "bundled libs:  $(ls "$PKG/lib" | tr '\n' ' ')"
} | tee "$PKG/BUILD_INFO.txt"

cp LICENSE THIRD_PARTY_NOTICES.md "$PKG/"
# License texts for what was bundled (Debian ships them per package).
mkdir -p "$PKG/licenses"
while read -r soname path; do
  pkg=$(dpkg -S "$(readlink -f "$path")" 2>/dev/null | head -1 | cut -d: -f1 || true)
  if [ -n "$pkg" ] && [ -f "/usr/share/doc/$pkg/copyright" ]; then cp "/usr/share/doc/$pkg/copyright" "$PKG/licenses/$pkg.copyright"; fi
done < "$DIST/bundled.txt"

# Reproducible tarball + checksum.
cd "$DIST"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - "$NAME" | gzip -n -9 > "$NAME.tar.gz"
sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256"
ls -l "$NAME.tar.gz"; cat "$NAME.tar.gz.sha256"
