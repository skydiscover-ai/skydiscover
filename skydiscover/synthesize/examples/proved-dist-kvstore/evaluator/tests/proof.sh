#!/usr/bin/env bash
# The proof is the test. It passes only when the agent's files (RYW_Impl.v, RYW_Refinement.v,
# RYW_Examples.v in $SKYDISCOVER_IMPL), built from an empty directory together with the read-only
# spec ($SKYDISCOVER_INTERFACE, default ..), make RYW_Correct.v type-check, so that
# ryw_impl_refines_spec holds:
#
#   1. the spec is byte-for-byte what the task shipped (else the theorem could have been weakened);
#   2. the agent's files contain no word that makes Coq accept a claim without a proof;
#   3. everything type-checks from clean;
#   4. Print Assumptions shows the theorem rests on nothing beyond the framework's own constants;
#   5. the non-vacuity example built and exercises RYW_Impl.
#
# Needs coqc, coqtop, coq_makefile. Exit 0 the proof holds, 1 it does not.
set -uo pipefail
cd "$(dirname "$0")"
impl="${SKYDISCOVER_IMPL:?set SKYDISCOVER_IMPL to the directory holding RYW_Impl.v, RYW_Refinement.v, RYW_Examples.v}"
spec="${SKYDISCOVER_INTERFACE:-$PWD/..}"
files=(RYW_Impl.v RYW_Refinement.v RYW_Examples.v)
theorem=ryw_impl_refines_spec

fail() { echo "FAIL: $*" >&2; exit 1; }

# 1. The spec is what the task shipped. Regenerate with `sha256sum ../*.v` when the task changes.
sha256sum --check --quiet - <<EOF || fail "the spec differs from the task's evaluator/; it is read-only"
75308c30e26a6b9efebc7f2fc62f936e8e6e45287c1b325ac1b673663f571b27  $spec/I_RYW_star.v
a16c634e6b2eb9cd2f87ae95cbf30bc7b77e28f4c15727fbcbdbb04ed35512b2  $spec/KVStore.v
08bbf64d786798f7f006911dd9d52c0f78b593fd3af4b2e994154490ddbdf561  $spec/RYW_Correct.v
EOF

# 2. The agent's three files, with no escape hatch anywhere in them (comments included).
[ -d "$impl" ] || fail "$impl is not a directory"
for f in "${files[@]}"; do
  [ -f "$impl/$f" ] || fail "$impl/$f is missing"
done
if hit="$(grep -nwE 'Admitted|admit|Axiom|Conjecture|sorry|bypass_check' "${files[@]/#/$impl/}")" ||
   hit="$(grep -nE 'Unset +(Guard|Positivity|Universe) +Checking' "${files[@]/#/$impl/}")"; then
  fail "escape hatch (the proof asserts rather than proves):"$'\n'"$hit"
fi

# 3. Build from an empty directory: the type checker decides.
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
mkdir "$build/spec" "$build/work"
cp "$spec"/*.v "$build/spec/"
for f in "${files[@]}"; do cp "$impl/$f" "$build/work/"; done
cd "$build"
printf '%s\n' '-Q spec KVS' '-Q work KVS' spec/KVStore.v spec/I_RYW_star.v \
  work/RYW_Impl.v work/RYW_Refinement.v work/RYW_Examples.v spec/RYW_Correct.v >_CoqProject
if ! { coq_makefile -f _CoqProject -o CoqMakefile && make -f CoqMakefile; } >build.log 2>&1; then
  tail -n 40 build.log >&2
  fail "does not type-check from clean"
fi

# 4. No smuggled axiom: the theorem depends on nothing beyond the framework's own constants
#    (KVStore.v's Parameters and Axioms, which the spec declares).
report="$(echo "Require Import KVS.RYW_Correct. Print Assumptions $theorem." |
  coqtop -Q spec KVS -Q work KVS -q 2>&1)"
if ! grep -q 'Closed under the global context' <<<"$report"; then
  grep -q '^Axioms:' <<<"$report" || fail "no assumption report for $theorem:"$'\n'"$report"
  # After the header every non-indented line must be an allowed constant; anything else (another
  # axiom, or a remark such as "x is assumed to be guarded.") fails.
  bad="$(sed -n '/^Axioms:/,$p' <<<"$report" | grep -vE '^(Axioms:| |$|Coq < |Rocq < )' |
    grep -vE '^(KVS\.)?KVStore\.[^ :]+( *:.*)?$' || true)"
  [ -z "$bad" ] || fail "$theorem rests on more than the framework declares:"$'\n'"$bad"
fi

# 5. The example is real: it built, and it uses the implementation.
[ -f work/RYW_Examples.vo ] || fail "RYW_Examples.vo was not built"
grep -qw RYW_Impl work/RYW_Examples.v || fail "RYW_Examples.v never mentions RYW_Impl"

echo "PROOF PASSED: $theorem type-checks from clean, no escape hatches, closed under the framework's assumptions, example built"
