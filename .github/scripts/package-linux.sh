#!/usr/bin/env bash
# Package the release shears-cli (built with --static-swift-stdlib) into a relocatable tarball:
#   shears-cli-linux-x86_64/{bin/shears-cli, lib/*.so*, BUILD_INFO.txt, LICENSE, THIRD_PARTY_NOTICES.md}
# Every shared library that is not part of the C library is copied into lib/ and the RUNPATHs are
# rewritten so the bundle needs nothing but glibc and a g++ at run time (MLX compiles fused kernels
# with g++). Writes dist/shears-cli-linux-x86_64.tar.gz and dist/shears-cli-linux-x86_64.tar.gz.sha256.
set -euo pipefail

BIN="${1:-CLI/.build/release/shears-cli}"
NAME=shears-cli-linux-x86_64
DIST="$PWD/dist"
PKG="$DIST/$NAME"
rm -rf "$DIST"; mkdir -p "$PKG/bin" "$PKG/lib"; : > "$DIST/bundled.txt"

# Libraries that come with glibc: never bundled.
SYSTEM='^(linux-vdso|ld-linux-x86-64|libc|libm|libdl|libpthread|librt|libutil|libresolv|libanl|libnss_[a-z]+)\.so'

cp "$BIN" "$PKG/bin/shears-cli"
strip --strip-unneeded "$PKG/bin/shears-cli"

echo "ldd of the build, before bundling:"
ldd "$PKG/bin/shears-cli"

# ldd lists the transitive closure: "soname => /path (addr)".
ldd "$PKG/bin/shears-cli" | awk '/=> \//{print $1, $3}' | while read -r soname path; do
  if [[ "$soname" =~ $SYSTEM ]]; then continue; fi
  cp -L "$path" "$PKG/lib/$soname"
  echo "$soname $path" >> "$DIST/bundled.txt"
  echo "bundled $soname <- $path"
done
chmod u+w "$PKG"/lib/* || true

if ls "$PKG/lib/" | grep -qE '^lib(ssl|crypto|curl)\.'; then
  echo "::warning::libcurl/OpenSSL ended up in the bundle; check whether FoundationNetworking is linked"
fi

patchelf --set-rpath '$ORIGIN/../lib' "$PKG/bin/shears-cli"
for f in "$PKG"/lib/*; do patchelf --set-rpath '$ORIGIN' "$f"; done

# Resources next to the binary (SwiftPM *.resources / *.bundle), if any.
for r in "$(dirname "$BIN")"/*.resources "$(dirname "$BIN")"/*.bundle; do
  [ -e "$r" ] && { cp -r "$r" "$PKG/bin/"; echo "copied resource bundle $r"; }
done

# Verify: nothing unresolved, and every bundled soname resolves into the bundle (not the system).
echo "ldd of the bundle (no LD_LIBRARY_PATH):"
env -u LD_LIBRARY_PATH ldd "$PKG/bin/shears-cli" | tee "$DIST/ldd.txt"
if grep -q "not found" "$DIST/ldd.txt"; then echo "unresolved libraries"; exit 1; fi
if env -u LD_LIBRARY_PATH ldd "$PKG/bin/shears-cli" | awk '/=> \//{print $1, $3}' \
    | while read -r so path; do [[ "$so" =~ $SYSTEM ]] || [[ "$path" == "$PKG/lib/"* ]] || echo "$so resolves outside the bundle: $path"; done | grep .; then
  exit 1
fi
env -u LD_LIBRARY_PATH "$PKG/bin/shears-cli" --help >/dev/null

# Oldest glibc this binary can run on.
GLIBC_NEEDED=$(objdump -T "$PKG/bin/shears-cli" "$PKG"/lib/* 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
{
  echo "shears-cli for Linux x86_64 (CPU, MLX), Swift runtime linked statically"
  echo "commit:        ${GITHUB_SHA:-$(git rev-parse HEAD)}"
  echo "built:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "swift:         $(swift --version 2>&1 | head -1)"
  echo "build os:      $(. /etc/os-release; echo "$PRETTY_NAME"), glibc $(ldd --version | head -1 | awk '{print $NF}')"
  echo "needs glibc:   >= ${GLIBC_NEEDED#GLIBC_}"
  echo "run time:      g++ must be installed (MLX compiles fused CPU kernels with it)"
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
