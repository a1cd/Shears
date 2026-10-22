#!/usr/bin/env bash
# nvcc stand-in, put first on PATH for the SwiftPM build of mlx-swift's CUDA backend (the CudaBuild plugin runs `nvcc` from PATH).
#
# 1. Architectures. The plugin's encuda tool passes a single "-arch sm_NN" to nvcc, for compiling and for the device-link
#    step. This replaces it with one -gencode per architecture in SHEARS_CUDA_ARCHS (PTX for the last), so both steps get
#    the same list, and allows the Swift toolchain's clang as the host compiler even if this nvcc has not heard of its version.
# 2. Cache (SHEARS_NVCC_CACHE=<dir>, SHEARS_NVCC_SALT=<string>). The two expensive calls, `nvcc -cuda` (one .cu to one
#    .cpp holding the device code for every architecture, about 25 minutes for MLX's 96 files on a 4-core runner) and
#    `nvcc --device-link`, are content-addressed: the key is the salt, the arguments (without the output path) and the
#    SHA-256 of every .cu and .o input. A hit copies the stored output and does not run nvcc. SwiftPM's own up-to-date
#    check cannot do this (it compares file identity, and a restored build directory has new inodes); this does not
#    care where the files came from. The salt must change whenever something nvcc reads that is not an argument changes
#    (MLX sources, CUTLASS/CCCL headers, CUDA toolkit, host compiler, this script); the workflow computes it.
# 3. Measurement (SHEARS_NVCC_LOG=<file>): one tab-separated line per call, "mode  input  seconds  hit|miss  peakKB".
# 4. SHEARS_NVCC_THREADS=<n>: passes --threads n to the compile calls (parallel per-architecture device compilation).
set -u
REAL=${SHEARS_REAL_NVCC:-/usr/local/cuda/bin/nvcc}

args=(); skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  if [ "$a" = "-arch" ]; then skip=1; continue; fi
  args+=("$a")
done
gen=(); last=""
for a in ${SHEARS_CUDA_ARCHS:-75}; do last=$a; done
for a in ${SHEARS_CUDA_ARCHS:-75}; do
  if [ "$a" = "$last" ]; then gen+=(-gencode "arch=compute_$a,code=[sm_$a,compute_$a]"); else gen+=(-gencode "arch=compute_$a,code=sm_$a"); fi
done

mode=other; out=""; input=""; prev=""
for a in "${args[@]}"; do
  case "$a" in
    -cuda) mode=compile ;;
    --device-link) mode=link ;;
    *.cu) input="${input:-$a}" ;;
  esac
  [ "$prev" = "-o" ] && out="$a"
  prev="$a"
done
[ "$mode" = link ] && input="__cuda_link"

extra=(-allow-unsupported-compiler "${gen[@]}")
thr=()   # speed only, so it is not part of the cache key
if [ "$mode" = compile ] && [ -n "${SHEARS_NVCC_THREADS:-}" ]; then thr=(--threads "$SHEARS_NVCC_THREADS"); fi

if [ "$mode" = other ] || [ -z "$out" ]; then exec "$REAL" "${extra[@]}" "${thr[@]}" "${args[@]}"; fi

now() { date +%s.%N; }
t0=$(now)
elapsed() { awk -v a="$(now)" -v b="$t0" 'BEGIN{printf "%.1f", a-b}'; }
key=""; cached=""
if [ -n "${SHEARS_NVCC_CACHE:-}" ]; then
  key=$({
    printf '%s\0' "${SHEARS_NVCC_SALT:-}" "$mode" "${extra[@]}"
    skipnext=0
    for a in "${args[@]}"; do
      if [ "$skipnext" = 1 ]; then skipnext=0; continue; fi
      if [ "$a" = "-o" ]; then skipnext=1; continue; fi
      printf '%s\0' "$a"
    done
    for a in "${args[@]}"; do
      if [[ "$a" == *.cu || "$a" == *.o ]]; then sha256sum < "$a"; fi
    done
  } | sha256sum | cut -c1-40)
  mkdir -p "$SHEARS_NVCC_CACHE"
  cached="$SHEARS_NVCC_CACHE/$mode-$key.cpp"
  if [ -s "$cached" ]; then
    cp "$cached" "$out"
    [ -n "${SHEARS_NVCC_LOG:-}" ] && printf '%s\t%s\t%s\thit\t0\n' "$mode" "$(basename "$input")" "$(elapsed)" >> "$SHEARS_NVCC_LOG"
    exit 0
  fi
fi

timefile=$(mktemp)
if /usr/bin/time -f '%M' -o /dev/null true 2>/dev/null; then
  /usr/bin/time -f '%M' -o "$timefile" "$REAL" "${extra[@]}" "${thr[@]}" "${args[@]}"   # GNU time: peak resident KB
else
  "$REAL" "${extra[@]}" "${thr[@]}" "${args[@]}"
fi
status=$?
if [ "$status" = 0 ] && [ -n "$cached" ]; then
  tmp=$(mktemp "$SHEARS_NVCC_CACHE/.tmp.XXXXXX") && cp "$out" "$tmp" && mv "$tmp" "$cached"
fi
if [ -n "${SHEARS_NVCC_LOG:-}" ]; then
  printf '%s\t%s\t%s\tmiss\t%s\n' "$mode" "$(basename "$input")" "$(elapsed)" "$(tail -1 "$timefile")" >> "$SHEARS_NVCC_LOG"
fi
rm -f "$timefile"
exit "$status"
