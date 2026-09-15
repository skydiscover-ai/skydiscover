(* RYW_Correct.v: the proof obligation (IMMUTABLE; do not edit). Applies the framework's
   TraceInclusion to your work/ modules against the abstract I_RYW_star. You make it compile by
   supplying work/RYW_Impl.v and work/RYW_Refinement.v; the `<: Refinement` ascription forces the
   spec's exact statements, so a weakened proof fails to type-check. The verifier is the only oracle. *)

Require Import KVS.KVStore.
Require Import KVS.I_RYW_star.
Require Import KVS.RYW_Impl.
Require Import KVS.RYW_Refinement.

Module TI := TraceInclusion RYW_Impl I_RYW_star RYW_Refinement.

Theorem ryw_impl_refines_spec :
  forall a v0 h wi,
    TI.WI.step_star (TI.WI.initWorld a v0) h wi ->
    exists hs ws,
      TI.WS.step_star (TI.WS.initWorld a v0) hs ws /\
      TI.WI.obs_proj h = TI.WS.obs_proj hs.
Proof. exact TI.trace_inclusion_doc. Qed.
