(* I_RYW_star.v: Read Your Writes Specification (starred = abstract).
   Translates Doc.tex figure at L653-799 in commit 048ae79
   (md5 cdb839e72bd66f33c2e35d790998ab37).

   This is the abstract RYW spec. Concrete RYW implementations refine it.
   It tracks every put a client has done as a set, and the read-guard requires
   that the read replica has not missed any of the requesting client's own
   past puts.

   ===== Translation table (Doc.tex symbol -> Coq) =====
   Doc.tex                                              Coq
   --------------------------------------------------------------------
   T := Nat                                             Timestamp (= nat, in framework)
   CState := Set[K x T]                                 list (Timestamp * Key)
                                                        (see typo note below)
   cInit(c) := empty-set                                []
   RState := K |-> (V x C x T x Set[T])                 Key -> (Val * ClientId * Timestamp * list Timestamp)
   rInit(r, v0) := [k |-> <v0, c0, 0, empty>]           fun _ => (v0, c0, 0, [])
   GetReqPayload := Set[T]                              list Timestamp
   GetResPayload := Unit                                unit
   PutReqPayload := T x Set[T]                          (Timestamp * list Timestamp)
   p|_k := { t | <t,k> in p }                           p_at_k via filter + map

   getReq (k)(_, p) := ret <p|_k, p>
   getGuard (k)(p)(c, _, sigma) :=
       let <_, c', t', p'> <- sigma(k) in
       c = c' ==> p subset (p' union {t'})
   get (k)(_)(_,_, sigma) :=
       let <v, _, _, _> <- sigma(k) in ret <v, unit, sigma>
   getRes (_,_)(_)(_, p) := ret p
   putReq (k,_)(_, p) :=
       let t := |p| + 1, p' := p union {<t, k>} in
       ret <<t, p|_k>, p'>
   putGuard always true
   put (k, v)(<t, p>)(c, _, sigma) := sigma[k |-> <v, c, t, p>]

   ===== Doc.tex ambiguity noted in this translation =====
   The CState type at L673 is declared `Set[K x T]` (key first, timestamp second),
   but every subsequent operation that constructs or destructures elements uses
   <t, k> order: L774 inserts <t, k>, and the helper at L793 says
   `p|_k := { t | <t, k> in p }`. The operational form is authoritative; this
   translation orders pairs as (Timestamp, Key) to match L774 and L793. Either
   the L673 label or the L774/L793 operational order is a typo; the
   refinement-relation semantics are unaffected by which we pick, but elements
   being (T, K) makes the helper read more naturally as
   `p_at_k p k = filter-by-second-component p k`. *)

From Coq Require Import List Bool Arith PeanoNat.
Require Import KVS.KVStore.
Export ListNotations.

Module I_RYW_star <: AlgDef.

  (* ===== Data types ===== *)

  (* CState : list (Timestamp * Key). See header note on ordering. *)
  Definition CState : Type := list (Timestamp * Key).

  (* RState : Key -> (Val * ClientId * Timestamp * list Timestamp).
     For each key k, the entry stores (value, origin client, origin timestamp,
     timestamps of past puts on key k). *)
  Definition RState : Type := Key -> (Val * ClientId * Timestamp * list Timestamp).

  (* cInit(c) := empty-set *)
  Definition cInit (c : ClientId) : CState := [].

  (* rInit(r, v0) := [k |-> <v0, c0, 0, empty>] *)
  Definition rInit (r : ReplicaId) (v0 : Val) : RState :=
    fun _k : Key => (v0, c0, 0, @nil Timestamp).

  Definition GetReqPayload : Type := list Timestamp.
  Definition GetResPayload : Type := unit.
  Definition PutReqPayload : Type := (Timestamp * list Timestamp)%type.

  (* ===== Decidable equality on payloads (required by AlgDef) ===== *)

  Definition GetReqPayload_eq_dec : forall (x y : GetReqPayload), {x = y} + {x <> y}.
  Proof. apply list_eq_dec. apply Nat.eq_dec. Defined.

  Definition GetResPayload_eq_dec : forall (x y : GetResPayload), {x = y} + {x <> y}.
  Proof. decide equality. Defined.

  Definition PutReqPayload_eq_dec : forall (x y : PutReqPayload), {x = y} + {x <> y}.
  Proof.
    intros [t1 p1] [t2 p2].
    destruct (Nat.eq_dec t1 t2);
      [destruct (list_eq_dec Nat.eq_dec p1 p2) |];
      try (left; congruence); right; congruence.
  Defined.

  (* ===== Helpers from the figure ===== *)

  (* p|_k := { t | <t, k> in p }.
     Doc.tex L793. Returns the timestamps in p whose paired key is k. *)
  Definition p_at_k (p : CState) (k : Key) : list Timestamp :=
    map fst (filter (fun e : Timestamp * Key => Nat.eqb (snd e) k) p).

  (* p subset q (over list Timestamp), used in the getGuard. *)
  Definition list_subset (p q : list Timestamp) : bool :=
    forallb (fun x => existsb (Nat.eqb x) q) p.

  (* ===== Operations ===== *)

  (* Doc.tex L711-715:
       getReq (k) (_, p) := ret <p|_k, p> *)
  Definition getReq (k : Key) (c : ClientId) (sigma : CState)
      : (GetReqPayload * CState) :=
    (p_at_k sigma k, sigma).

  (* Doc.tex L717-728:
       getGuard (k)(p)(c, _, sigma) :=
         let <_, c', t', p'> <- sigma(k) in
         c = c' ==> p subset (p' union {t'})
     Encoded as a boolean: when c =? c', return list_subset p (t' :: p');
     otherwise (vacuously) true. *)
  Definition getGuard (k : Key) (p : GetReqPayload) (c : ClientId) (r : ReplicaId)
      (s : RState) : bool :=
    let '(_, c', t', p') := s k in
    if Nat.eq_dec c c'
    then list_subset p (t' :: p')
    else true.

  (* Doc.tex L747-754:
       get (k)(_)(_,_, sigma) := let <v, _, _, _> <- sigma(k) in ret <v, unit, sigma> *)
  Definition get (k : Key) (p : GetReqPayload) (c : ClientId) (r : ReplicaId)
      (s : RState) : (Val * GetResPayload * RState) :=
    let '(v, _, _, _) := s k in (v, tt, s).

  (* Doc.tex L756-760:
       getRes (_, _)(_)(_, p) := ret p
     Returns session state unchanged. *)
  Definition getRes (k : Key) (v : Val) (p : GetResPayload) (c : ClientId)
      (sigma : CState) : CState :=
    sigma.

  (* Doc.tex L767-779:
       putReq (k, _)(_, p) :=
         let t := |p| + 1
         let p' := p union {<t, k>}
         ret <<t, p|_k>, p'> *)
  Definition putReq (k : Key) (v : Val) (c : ClientId) (sigma : CState)
      : (PutReqPayload * CState) :=
    let t := length sigma + 1 in
    let sigma' := (t, k) :: sigma in
    ((t, p_at_k sigma k), sigma').

  (* Doc.tex L781-785: putGuard always true. *)
  Definition putGuard (k : Key) (v : Val) (p : PutReqPayload) (c : ClientId)
      (r : ReplicaId) (s : RState) : bool :=
    true.

  (* Doc.tex L787-790:
       put (k, v)(<t, p>)(c, _, sigma) := sigma[k |-> <v, c, t, p>] *)
  Definition put (k : Key) (v : Val) (p : PutReqPayload) (c : ClientId)
      (r : ReplicaId) (s : RState) : RState :=
    let '(t, past) := p in
    fun k' : Key => if Nat.eq_dec k k' then (v, c, t, past) else s k'.

End I_RYW_star.
