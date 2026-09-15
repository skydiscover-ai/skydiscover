#!/usr/bin/env bash
# Run the tests in this directory against the implementation at $SKYDISCOVER_IMPL.
#
#   bash test.sh              every test
#   bash test.sh <file>...    only the named tests
#
# Each test is one C++ program, compiled together with the implementation (a .cc file, or every .cc
# under a directory) and the interface header from $SKYDISCOVER_INTERFACE (default: the evaluator
# directory this folder sits in). A test passes when it exits 0. Exit 0 if every test passed, 1 if any
# failed or did not build.
set -uo pipefail
cd "$(dirname "$0")"
impl="${SKYDISCOVER_IMPL:?set SKYDISCOVER_IMPL to the implementation (.cc file or directory)}"
iface="${SKYDISCOVER_INTERFACE:-$PWD/..}"
if [ -d "$impl" ]; then mapfile -t srcs < <(find "$impl" -name '*.cc'); else srcs=("$impl"); fi
tests=("$@")
[ ${#tests[@]} -eq 0 ] && tests=(*.cc)
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
rc=0
for t in "${tests[@]}"; do
  bin="$out/${t%.*}"
  if ! "${CXX:-g++}" -std=c++17 -O2 -I"$iface" "$t" "${srcs[@]}" -o "$bin" -lpthread 2>"$out/build.log"; then
    echo "FAIL $t: did not build"
    cat "$out/build.log"
    rc=1
    continue
  fi
  if "$bin" >"$out/run.log" 2>&1; then
    echo "PASS $t"
  else
    echo "FAIL $t (exit $?)"
    tail -n 20 "$out/run.log"
    rc=1
  fi
done
exit $rc
