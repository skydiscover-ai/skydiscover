#!/usr/bin/env bash
# Run the tests in this directory against the implementation at $SKYDISCOVER_IMPL.
#
#   bash test.sh              every test
#   bash test.sh <file>...    only the named tests
#
# Each test is one shell script (here there is one, proof.sh: the proof is the test). It sees the
# same $SKYDISCOVER_IMPL and $SKYDISCOVER_INTERFACE and passes when it exits 0. Exit 0 if every
# test passed, 1 if any failed.
set -uo pipefail
cd "$(dirname "$0")"
: "${SKYDISCOVER_IMPL:?set SKYDISCOVER_IMPL to the directory holding the .v files under test}"
tests=("$@")
[ ${#tests[@]} -eq 0 ] && tests=(*.sh)
rc=0
for t in "${tests[@]}"; do
  [ "$t" = "test.sh" ] && continue
  if bash "$t"; then
    echo "PASS $t"
  else
    echo "FAIL $t"
    rc=1
  fi
done
exit $rc
