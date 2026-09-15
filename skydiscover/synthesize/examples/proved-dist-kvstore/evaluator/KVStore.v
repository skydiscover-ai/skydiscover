(* KVStore.v: Distributed Key-Value Store Framework
   Based on Mohsen Lesani's Doc.tex
   (KeyValueStoreSpecs/Doc.tex at commit 048ae79, pulled 2026-04-26).

   Encodes:
   - Figure 1: Client Programs (syntax). Note: our `Stmt` does not bind a read
     variable `x`; sufficient for RYW/MR/MW/CC specs which are agnostic to
     client-program internals.
   - Figure 2: Implementation Interface (AlgDef module type), with the
     id-aware initialisation: `cInit : C -> CState`, `rInit : R -> V -> RState`.
   - Figure 3: World State (clients, replicas, network), with id-aware
     initial world: [c |-> <cInit(c), a(c)>] / [r |-> rInit(r, v0)].
   - Figure 4: Operational Semantics (step relation), with the
     `c <> c0` premise on Get-Req and Put-Req (c0 is a designated client
     used to stamp the default cell; it does not take operational steps).
   - Trace Inclusion (Doc.tex L481-489 in 048ae79):
     I2 trace-includes I1 iff every reachable trace of I2 has a corresponding
     trace of I1 with the same external history `ext(h)`.
   - Convergence (Doc.tex L491-502 in 048ae79):
     from a quiescent state (network empty), get at any replica yields the
     same value.
   - Refinement: simulation-based (implies trace inclusion).

   Today's pushes (2026-04-26 PT) added new model figures only (I_Rel, I_RYW*,
   I_MR*, I_MR, I_MW*, I_MW, refinement-hierarchy diagram); framework
   definitions in Doc.tex L1-510 are unchanged from 6a90029.
*)

From Coq Require Import List Bool Arith PeanoNat Lia.
From Coq Require Import FunctionalExtensionality.
Export ListNotations.

(* ===== Base types ===== *)

Definition Key := nat.
Definition Val := nat.
Definition ClientId := nat.
Definition ReplicaId := nat.
Definition Timestamp := nat.
Definition UniqueId := nat.

(* ===== Figure 1: Client Programs ===== *)

Inductive Stmt : Type :=
  | SPut : UniqueId -> Key -> Val -> Stmt -> Stmt
  | SGet : UniqueId -> Key -> Stmt -> Stmt
  | SSkip : Stmt
  | SBlockedGet : UniqueId -> Key -> Stmt -> Stmt.

Definition App := ClientId -> Stmt.

(* ===== Figure 2: Implementation Interface (AlgDef) ===== *)

Module Type AlgDef.

  Parameter CState : Type.
  Parameter RState : Type.
  (* Doc.tex Fig 2 line 126: cInit : C -> CState (client-id-aware). *)
  Parameter cInit : ClientId -> CState.
  (* Doc.tex Fig 2 line 136: rInit : R -> V -> RState (replica-id-aware). *)
  Parameter rInit : ReplicaId -> Val -> RState.

  Parameter GetReqPayload : Type.
  Parameter GetResPayload : Type.
  Parameter PutReqPayload : Type.

  (* Decidable equality on payloads, needed for message removal *)
  Parameter GetReqPayload_eq_dec : forall (x y : GetReqPayload), {x = y} + {x <> y}.
  Parameter GetResPayload_eq_dec : forall (x y : GetResPayload), {x = y} + {x <> y}.
  Parameter PutReqPayload_eq_dec : forall (x y : PutReqPayload), {x = y} + {x <> y}.

  (* Get operations *)
  Parameter getReq : Key -> ClientId -> CState -> (GetReqPayload * CState).
  Parameter getGuard : Key -> GetReqPayload -> ClientId -> ReplicaId -> RState -> bool.
  Parameter get : Key -> GetReqPayload -> ClientId -> ReplicaId -> RState -> (Val * GetResPayload * RState).
  Parameter getRes : Key -> Val -> GetResPayload -> ClientId -> CState -> CState.

  (* Put operations *)
  Parameter putReq : Key -> Val -> ClientId -> CState -> (PutReqPayload * CState).
  Parameter putGuard : Key -> Val -> PutReqPayload -> ClientId -> ReplicaId -> RState -> bool.
  Parameter put : Key -> Val -> PutReqPayload -> ClientId -> ReplicaId -> RState -> RState.

End AlgDef.

(* ===== Shared topology: clients and replicas ===== *)
(* These are OUTSIDE World so all instantiations share the same sets *)

Parameter allReplicas : list ReplicaId.
Axiom allReplicas_nonempty : allReplicas <> [].
Axiom allReplicas_NoDup : NoDup allReplicas.

Parameter allClients : list ClientId.
Axiom allClients_NoDup : NoDup allClients.

(* Doc.tex Fig 3 / Fig 4: c0 is a designated client used only to stamp the
   default cell of every replica (timestamp 0). The operational rules forbid
   c0 from issuing Get-Req or Put-Req (see StepGetReq, StepPutReq below), so
   c0 never takes a step in any reachable trace; it exists only as the
   "as-if-author" of default cells. *)
Parameter c0 : ClientId.
Axiom c0_in_clients : In c0 allClients.

(* Derived: allClients is nonempty (it contains c0). Symmetric to
   allReplicas_nonempty. Stated as a Lemma rather than an Axiom so that the
   axiomatic surface stays minimal. *)
Lemma allClients_nonempty : allClients <> [].
Proof.
  intros Hempty. pose proof c0_in_clients as Hin.
  rewrite Hempty in Hin. exact Hin.
Qed.

(* ===== Observable events (shared across all World instances) ===== *)

Inductive ObsEvent : Type :=
  | ObsPut : ClientId -> UniqueId -> Key -> Val -> ObsEvent
  | ObsGet : ClientId -> UniqueId -> Key -> Val -> ObsEvent.

(* ===== Figure 3: World State ===== *)

Module World (Alg : AlgDef).

  Import Alg.

  Record ClientState := mkClientState {
    cs_alg : CState;
    cs_prog : Stmt
  }.

  Inductive Message : Type :=
    | MGetReq : ClientId -> ReplicaId -> UniqueId -> Key -> GetReqPayload -> Message
    | MGetRes : ReplicaId -> ClientId -> UniqueId -> Key -> Val -> GetResPayload -> Message
    | MPutReq : ClientId -> ReplicaId -> Key -> Val -> PutReqPayload -> Message.

  (* Decidable equality on Messages *)
  Definition message_eq_dec : forall (m1 m2 : Message), {m1 = m2} + {m1 <> m2}.
  Proof.
    decide equality; try apply Nat.eq_dec;
      try apply GetReqPayload_eq_dec;
      try apply GetResPayload_eq_dec;
      try apply PutReqPayload_eq_dec.
  Defined.

  Record WorldState := mkWorld {
    w_clients : list (ClientId * ClientState);
    w_replicas : list (ReplicaId * RState);
    w_network : list Message
  }.

  (* ===== Finite maps via association lists (no closure chains) ===== *)

  Fixpoint lookup_client (cs : list (ClientId * ClientState)) (c : ClientId)
      (default : ClientState) : ClientState :=
    match cs with
    | [] => default
    | (c', s) :: rest => if Nat.eq_dec c c' then s else lookup_client rest c default
    end.

  Fixpoint update_client (cs : list (ClientId * ClientState)) (c : ClientId) (s : ClientState)
      : list (ClientId * ClientState) :=
    match cs with
    | [] => [(c, s)]
    | (c', s') :: rest =>
        if Nat.eq_dec c c' then (c, s) :: rest else (c', s') :: update_client rest c s
    end.

  Fixpoint lookup_replica (rs : list (ReplicaId * RState)) (r : ReplicaId)
      (default : RState) : RState :=
    match rs with
    | [] => default
    | (r', s) :: rest => if Nat.eq_dec r r' then s else lookup_replica rest r default
    end.

  Fixpoint update_replica (rs : list (ReplicaId * RState)) (r : ReplicaId) (s : RState)
      : list (ReplicaId * RState) :=
    match rs with
    | [] => [(r, s)]
    | (r', s') :: rest =>
        if Nat.eq_dec r r' then (r, s) :: rest else (r', s') :: update_replica rest r s
    end.

  (* Remove first occurrence of a message from the network *)
  Fixpoint remove_message (net : list Message) (m : Message) : list Message :=
    match net with
    | [] => []
    | m' :: rest =>
        if message_eq_dec m m' then rest
        else m' :: remove_message rest m
    end.

  Definition add_message (net : list Message) (m : Message) : list Message :=
    m :: net.

  Definition add_messages (net : list Message) (ms : list Message) : list Message :=
    ms ++ net.

  (* ===== Finite replica/client sets ===== *)

  (* allReplicas and allClients are defined at top level, shared across all World instances *)

  (* Default states for lookups *)
  (* Coq-internal placeholder for partial-map lookup when the queried client
     is not in `allClients`. In valid traces every active `c` is in
     `allClients`, so this default is never the actual returned state.
     Doc.tex does not specify a lookup default. *)
  Definition defaultClientState : ClientState := mkClientState (cInit c0) SSkip.

  (* Initial world. Doc.tex Fig 3 line 329:
     W0(a) = < [c |-> <cInit(c), a(c)>]_{c in C},
              [r |-> rInit(r, v0)]_{r in R},
              empty network >. *)
  Definition initWorld (a : App) (v0 : Val) : WorldState :=
    mkWorld
      (map (fun c => (c, mkClientState (cInit c) (a c))) allClients)
      (map (fun r => (r, rInit r v0)) allReplicas)
      [].

  (* ===== Labels (observable events) ===== *)

  Inductive Label : Type :=
    | LGetReq : ClientId -> UniqueId -> Key -> Label
    | LGetAtReplica : ReplicaId -> UniqueId -> Key -> Val -> Label
    | LGetRes : ClientId -> UniqueId -> Key -> Val -> Label
    | LPutReq : ClientId -> UniqueId -> Key -> Val -> Label   (* uid included *)
    | LPutAtReplica : ReplicaId -> Key -> Val -> Label.

  (* ===== Figure 4: Operational Semantics ===== *)

  Inductive step : WorldState -> Label -> WorldState -> Prop :=

    (* Get-Req: client c issues get request, sends to all replicas.
       Doc.tex Fig 4 line 359: premise `c <> c0` (c0 does not take steps). *)
    | StepGetReq : forall w c i k p sigma' s,
        c <> c0 ->
        cs_prog (lookup_client (w_clients w) c defaultClientState) = SGet i k s ->
        getReq k c (cs_alg (lookup_client (w_clients w) c defaultClientState)) = (p, sigma') ->
        step w
          (LGetReq c i k)
          (mkWorld
            (update_client (w_clients w) c (mkClientState sigma' (SBlockedGet i k s)))
            (w_replicas w)
            (add_messages (w_network w)
              (map (fun r => MGetReq c r i k p) allReplicas)))

    (* Get: replica r serves get request, CONSUMES GetReq message *)
    | StepGet : forall w c r i k p v p' rstate',
        In (MGetReq c r i k p) (w_network w) ->
        getGuard k p c r (lookup_replica (w_replicas w) r (rInit r 0)) = true ->
        get k p c r (lookup_replica (w_replicas w) r (rInit r 0)) = (v, p', rstate') ->
        step w
          (LGetAtReplica r i k v)
          (mkWorld
            (w_clients w)
            (update_replica (w_replicas w) r rstate')
            (add_message (remove_message (w_network w) (MGetReq c r i k p))
                         (MGetRes r c i k v p')))

    (* Get-Res: client c receives get response, CONSUMES GetRes message *)
    | StepGetRes : forall w c r i k v p sigma' s,
        cs_prog (lookup_client (w_clients w) c defaultClientState) = SBlockedGet i k s ->
        In (MGetRes r c i k v p) (w_network w) ->
        getRes k v p c (cs_alg (lookup_client (w_clients w) c defaultClientState)) = sigma' ->
        step w
          (LGetRes c i k v)
          (mkWorld
            (update_client (w_clients w) c (mkClientState sigma' s))
            (w_replicas w)
            (remove_message (w_network w) (MGetRes r c i k v p)))

    (* Put-Req: client c issues put, sends to all replicas.
       Doc.tex Fig 4 line 423: premise `c <> c0` (c0 does not take steps). *)
    | StepPutReq : forall w c i k v p sigma' s,
        c <> c0 ->
        cs_prog (lookup_client (w_clients w) c defaultClientState) = SPut i k v s ->
        putReq k v c (cs_alg (lookup_client (w_clients w) c defaultClientState)) = (p, sigma') ->
        step w
          (LPutReq c i k v)
          (mkWorld
            (update_client (w_clients w) c (mkClientState sigma' s))
            (w_replicas w)
            (add_messages (w_network w)
              (map (fun r => MPutReq c r k v p) allReplicas)))

    (* Put: replica r applies put, CONSUMES PutReq message *)
    | StepPut : forall w c r k v p rstate',
        In (MPutReq c r k v p) (w_network w) ->
        putGuard k v p c r (lookup_replica (w_replicas w) r (rInit r 0)) = true ->
        put k v p c r (lookup_replica (w_replicas w) r (rInit r 0)) = rstate' ->
        step w
          (LPutAtReplica r k v)
          (mkWorld
            (w_clients w)
            (update_replica (w_replicas w) r rstate')
            (remove_message (w_network w) (MPutReq c r k v p))).

  (* ===== step_star ===== *)

  Definition History := list Label.

  Inductive step_star : WorldState -> History -> WorldState -> Prop :=
    | StepStarRefl : forall w, step_star w [] w
    | StepStarStep : forall w1 h w2 l w3,
        step_star w1 h w2 ->
        step w2 l w3 ->
        step_star w1 (h ++ [l]) w3.

  (* ===== Precedence ===== *)

  Definition prec (h : History) (l1 l2 : Label) : Prop :=
    exists h1 h2 h3,
      h = h1 ++ l1 :: h2 ++ l2 :: h3.

  (* ===== Observable projection using shared ObsEvent =====
     This is our encoding of Doc.tex's external history `ext(h)`
     (Doc.tex Fig 3 line 307). `ext` keeps client-side put-request and
     get-response events; everything else (replica-side labels and the
     intermediate get-request label) is dropped.
     Note: our `ObsPut c i k v` carries the unique id `i` from the source
     statement `SPut i k v s`. This is stricter than Doc.tex's `c |> put k v`
     (which omits `i`); but `i` is determined by the program, identical in
     impl and spec runs of the same program, so the equality `obs_proj h1 =
     obs_proj h2` is sound w.r.t. trace inclusion. *)

  Definition proj_label (l : Label) : option ObsEvent :=
    match l with
    | LPutReq c i k v => Some (ObsPut c i k v)
    | LGetRes c i k v => Some (ObsGet c i k v)
    | _ => None
    end.

  Fixpoint obs_proj (h : History) : list ObsEvent :=
    match h with
    | [] => []
    | l :: rest =>
        match proj_label l with
        | Some e => e :: obs_proj rest
        | None => obs_proj rest
        end
    end.

  Lemma obs_proj_app : forall h1 h2,
    obs_proj (h1 ++ h2) = obs_proj h1 ++ obs_proj h2.
  Proof.
    intros h1 h2. induction h1 as [|l rest IH]; simpl.
    - reflexivity.
    - destruct (proj_label l); rewrite IH; reflexivity.
  Qed.

  (* ===== Convergence (Doc.tex line 487) =====
     An implementation is convergent if, from any reachable quiescent state
     (network empty), a successful get at any replica yields the same value
     as a successful get at any other replica:
       forall reachable W with empty network,
       forall k p c r1 r2 v,
       if getGuard succeeds at r1 with output value v,
       then getGuard also succeeds at r2 with the same value v.
     The post-state and payload of `get` are existentially quantified
     (irrelevant to convergence; only the value matters). *)
  Definition Convergent : Prop :=
    forall a v0 W h,
      step_star (initWorld a v0) h W ->
      w_network W = [] ->
      forall k p c r1 r2 v p1 rs1',
        getGuard k p c r1 (lookup_replica (w_replicas W) r1 (rInit r1 0)) = true ->
        get k p c r1 (lookup_replica (w_replicas W) r1 (rInit r1 0)) = (v, p1, rs1') ->
        getGuard k p c r2 (lookup_replica (w_replicas W) r2 (rInit r2 0)) = true /\
        exists p2 rs2',
          get k p c r2 (lookup_replica (w_replicas W) r2 (rInit r2 0)) = (v, p2, rs2').

End World.

(* ===== Refinement ===== *)

Module Type Refinement (I S : AlgDef).

  Module WI := World I.
  Module WS := World S.

  Parameter R : WI.WorldState -> WS.WorldState -> Prop.

  Axiom init_sim : forall a v0,
    R (WI.initWorld a v0) (WS.initWorld a v0).

  (* For every impl step, spec can match with same observable *)
  Axiom step_sim : forall wi ws l wi',
    R wi ws ->
    WI.step wi l wi' ->
    exists ws' hs,
      WS.step_star ws hs ws' /\
      R wi' ws' /\
      WI.obs_proj [l] = WS.obs_proj hs.

End Refinement.

(* ===== Trace Inclusion ===== *)

Module TraceInclusion (I S : AlgDef) (Ref : Refinement I S).

  Module WI := Ref.WI.
  Module WS := Ref.WS.

  (* Helper: compose step_star *)
  Lemma step_star_trans : forall w1 h1 w2 h2 w3,
    WS.step_star w1 h1 w2 ->
    WS.step_star w2 h2 w3 ->
    WS.step_star w1 (h1 ++ h2) w3.
  Proof.
    intros w1 h1 w2 h2 w3 H1 H2.
    induction H2 as [|w2' h2' w2'' l w2''' Hss IH Hstep].
    - rewrite app_nil_r. exact H1.
    - rewrite app_assoc. econstructor. apply IH. exact H1. exact Hstep.
  Qed.

  (* Full trace inclusion: related states AND matching observable histories.
     This is stronger than Doc.tex's Trace Inclusion (line 477), Doc.tex only
     requires the existence of a spec history with matching `ext`. We prove
     additionally that the impl-world is R-related to the resulting spec-world,
     because the proof carries R through anyway via `step_sim`. The literal
     Doc.tex version follows as the corollary `trace_inclusion_doc` below. *)
  Theorem trace_inclusion :
    forall a v0 h wi,
      WI.step_star (WI.initWorld a v0) h wi ->
      exists hs ws,
        WS.step_star (WS.initWorld a v0) hs ws /\
        Ref.R wi ws /\
        WI.obs_proj h = WS.obs_proj hs.
  Proof.
    intros a v0 h wi Hsteps.
    remember (WI.initWorld a v0) as w0 eqn:Hw0.
    induction Hsteps as [w | w1 h' w2 l w3 Hss IH Hstep].
    - subst w. exists [], (WS.initWorld a v0).
      split. constructor. split. apply Ref.init_sim. reflexivity.
    - destruct (IH Hw0) as [hs [ws [Hss_s [HR Hobs]]]].
      destruct (Ref.step_sim _ _ l _ HR Hstep) as [ws' [hs' [Hss' [HR' Hobs']]]].
      exists (hs ++ hs'), ws'.
      split.
      + apply step_star_trans with ws; assumption.
      + split.
        * exact HR'.
        * rewrite WI.obs_proj_app, WS.obs_proj_app.
          rewrite Hobs, Hobs'.
          reflexivity.
  Qed.

  (* Doc.tex Trace Inclusion (line 477), literal form:
     I_2 ⊑ I_1 iff every reachable impl trace has a corresponding spec trace
     with equal external history. We obtain this by dropping the `Ref.R wi ws`
     conjunct from the stronger `trace_inclusion` theorem. *)
  Corollary trace_inclusion_doc :
    forall a v0 h wi,
      WI.step_star (WI.initWorld a v0) h wi ->
      exists hs ws,
        WS.step_star (WS.initWorld a v0) hs ws /\
        WI.obs_proj h = WS.obs_proj hs.
  Proof.
    intros a v0 h wi Hsteps.
    destruct (trace_inclusion a v0 h wi Hsteps) as [hs [ws [Hss [_ Hobs]]]].
    exists hs, ws. split; assumption.
  Qed.

End TraceInclusion.
