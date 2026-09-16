#!/usr/bin/env bash
# compile_bench_libs.sh [tags...]   -- juliac-compile the exported benchmark models.
#
# Reads  bench_parity/<tag>_model.jl   (written and GATED by verify_bench_models.jl)
# Writes bench_parity/libace_<tag>.so  (consumed by bench_parity.sh)
#
# Default tags: cantor_poly cantor_h50 tial_poly tial_h50
# CPU_TARGET (default "native") is passed through to verify_cantor/compile_lib.jl.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT=$REPO/bench_parity
CPU_TARGET=${CPU_TARGET:-native}
TAGS=${@:-"cantor_poly cantor_h50 tial_poly tial_h50"}
rc=0
for tag in $TAGS; do
  src=$OUT/${tag}_model.jl
  lib=$OUT/libace_${tag}.so
  if [ ! -f "$src" ]; then
    echo "compile_bench_libs.sh: $src missing -- run verify_bench_models.jl $tag first" >&2
    rc=1; continue
  fi
  echo "### $tag: $(du -h "$src" | cut -f1) -> $(basename "$lib")"
  if julia --project="$REPO/export" "$REPO/verify_cantor/compile_lib.jl" \
           "$src" "$lib" "$CPU_TARGET" > "$OUT/juliac_${tag}.log" 2>&1; then
    tail -2 "$OUT/juliac_${tag}.log"
  else
    echo "  FAILED -- see $OUT/juliac_${tag}.log"; tail -5 "$OUT/juliac_${tag}.log"; rc=1
  fi
done
exit $rc
