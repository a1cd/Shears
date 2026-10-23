#!/usr/bin/env bash
# Puts ccache in front of the Swift toolchain's clang and clang++, which is what SwiftPM runs for every C, C++ and
# assembly file (it looks next to swift-build, not on PATH, so replacing the files is the only way in) and what the CUDA
# plugin's encuda tool runs as the host compiler. Replaces the SwiftPM build directory as the thing worth caching:
# SwiftPM's up-to-date check compares file identity (inode, mtime), which a restored cache does not preserve, so a
# restored build directory recompiles everything; ccache keys on content and does not care.
# Idempotent. Usage: ccache-toolchain.sh <toolchain usr/bin directory>
set -euo pipefail
bin="${1:?usage: ccache-toolchain.sh <toolchain usr/bin>}"
command -v ccache >/dev/null || { echo "ccache is not installed"; exit 1; }

if [ -f "$bin/clang" ] && grep -q SHEARS_CCACHE_WRAPPER "$bin/clang" 2>/dev/null; then
  echo "ccache is already in front of $bin/clang"
  exit 0
fi
ls -l "$bin"/clang* || true
if [ -L "$bin/clang" ]; then
  real=$(readlink -f "$bin/clang")
else
  mv "$bin/clang" "$bin/clang.real"; real="$bin/clang.real"
fi
echo "real clang: $real"
"$real" --version | head -1

# remove the links first: writing through a symlink would overwrite the real compiler
rm -f "$bin/clang" "$bin/clang++"
cat > "$bin/clang" <<WRAP
#!/bin/sh
# SHEARS_CCACHE_WRAPPER
exec ccache "$real" "\$@"
WRAP
cat > "$bin/clang++" <<WRAP
#!/bin/sh
# SHEARS_CCACHE_WRAPPER
exec ccache "$real" --driver-mode=g++ "\$@"
WRAP
chmod +x "$bin/clang" "$bin/clang++"
"$bin/clang++" --version | head -1
