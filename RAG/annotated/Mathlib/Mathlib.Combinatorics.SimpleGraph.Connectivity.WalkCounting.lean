theorem set_walk_self_length_zero_eq (u : V) : {p : G.Walk u u | p.length = 0} = {Walk.nil} := by
  /-
    V : Type u
    G : SimpleGraph V
    u : V
    ⊢ Eq (setOf fun p => Eq p.length 0) (Singleton.singleton SimpleGraph.Walk.nil)
  -/
  ext p
  /-
    case h
    V : Type u
    G : SimpleGraph V
    u : V
    p : G.Walk u u
    ⊢ Iff (Membership.mem (setOf fun p => Eq p.length 0) p) (Membership.mem (Singl …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem set_walk_length_zero_eq_of_ne {u v : V} (h : u ≠ v) :
    {p : G.Walk u v | p.length = 0} = ∅ := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    h : Ne u v
    ⊢ Eq (setOf fun p => Eq p.length 0) EmptyCollection.emptyCollection
  -/
  ext p
  /-
    case h
    V : Type u
    G : SimpleGraph V
    u v : V
    h : Ne u v
    p : G.Walk u v
    ⊢ Iff (Membership.mem (setOf fun p => Eq p.length 0) p) (Membership.mem EmptyC …
  -/
  simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
  /-
    case h
    V : Type u
    G : SimpleGraph V
    u v : V
    h : Ne u v
    p : G.Walk u v
    ⊢ Not (Eq p.length 0)
  -/
  exact fun h' => absurd (Walk.eq_of_length_eq_zero h') h
  /-
    🎉 no goals
  -/


theorem set_walk_length_succ_eq (u v : V) (n : ℕ) :
    {p : G.Walk u v | p.length = n.succ} =
      ⋃ (w : V) (h : G.Adj u w), Walk.cons h '' {p' : G.Walk w v | p'.length = n} := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    n : Nat
    ⊢ Eq (setOf fun p => Eq p.length n.succ) (Set.iUnion fun w => Set.iUnion fun h …
  -/
  ext p
  /-
    case h
    V : Type u
    G : SimpleGraph V
    u v : V
    n : Nat
    p : G.Walk u v
    ⊢ Iff (Membership.mem (setOf fun p => Eq p.length n.succ) p) (Membership.mem ( …
  -/
  cases' p with _ _ w _ huw pwv
    /-
      case h.nil
      V : Type u
      G : SimpleGraph V
      u : V
      n : Nat
      ⊢ Iff (Membership.mem (setOf fun p => Eq p.length n.succ) SimpleGraph.Walk.nil …
    -/
  · simp [eq_comm]
    /-
      🎉 no goals
    -/
  · simp only [Nat.succ_eq_add_one, Set.mem_setOf_eq, Walk.length_cons, add_left_inj,
      Set.mem_iUnion, Set.mem_image, exists_prop]
    /-
      case h.cons
      V : Type u
      G : SimpleGraph V
      u v : V
      n : Nat
      w : V
      huw : G.Adj u w
      pwv : G.Walk w v
      ⊢ Iff (Eq pwv.length n) (Exists fun i => Exists fun h => Exists fun x => And ( …
    -/
    constructor
      /-
        case h.cons.mp
        V : Type u
        G : SimpleGraph V
        u v : V
        n : Nat
        w : V
        huw : G.Adj u w
        pwv : G.Walk w v
        ⊢ Eq pwv.length n → Exists fun i => Exists fun h => Exists fun x => And (Eq x. …
      -/
    · rintro rfl
      /-
        case h.cons.mp
        V : Type u
        G : SimpleGraph V
        u v w : V
        huw : G.Adj u w
        pwv : G.Walk w v
        ⊢ Exists fun i => Exists fun h => Exists fun x => And (Eq x.length pwv.length) …
      -/
      exact ⟨w, huw, pwv, rfl, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case h.cons.mpr
        V : Type u
        G : SimpleGraph V
        u v : V
        n : Nat
        w : V
        huw : G.Adj u w
        pwv : G.Walk w v
        ⊢ (Exists fun i => Exists fun h => Exists fun x => And (Eq x.length n) (Eq (Si …
      -/
    · rintro ⟨w, huw, pwv, rfl, rfl, rfl⟩
      /-
        case h.cons.mpr.intro.intro.intro.intro.refl
        V : Type u
        G : SimpleGraph V
        u v w : V
        huw✝ : G.Adj u w
        pwv : G.Walk w v
        huw : G.Adj u w
        ⊢ Eq pwv.length pwv.length
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- Walks of length two from `u` to `v` correspond bijectively to common neighbours of `u` and `v`.
Note that `u` and `v` may be the same. -/
@[simps]
def walkLengthTwoEquivCommonNeighbors (u v : V) :
    {p : G.Walk u v // p.length = 2} ≃ G.commonNeighbors u v where
  toFun p := ⟨p.val.getVert 1, match p with
    | ⟨.cons _ (.cons _ .nil), _⟩ => ⟨‹G.Adj u _›, ‹G.Adj _ v›.symm⟩⟩
  invFun w := ⟨w.prop.1.toWalk.concat w.prop.2.symm, rfl⟩
                                                /-
                                                  V : Type u
                                                  G : SimpleGraph V
                                                  u v v✝ : V
                                                  h✝¹ : G.Adj u v✝
                                                  h✝ : G.Adj v✝ v
                                                  hp : Eq (SimpleGraph.Walk.cons h✝¹ (SimpleGraph.Walk.cons h✝ SimpleGraph.Walk. …
                                                  ⊢ Eq ((fun w => ⟨(SimpleGraph.Adj.toWalk ⋯).concat ⋯, ⋯⟩) ((fun p => ⟨(↑p).get …
                                                -/
  left_inv | ⟨.cons _ (.cons _ .nil), hp⟩ => by rfl
                                                /-
                                                  🎉 no goals
                                                -/
  right_inv _ := rfl


/-- The `Finset` of length-`n` walks from `u` to `v`.
This is used to give `{p : G.walk u v | p.length = n}` a `Fintype` instance, and it
can also be useful as a recursive description of this set when `V` is finite.

See `SimpleGraph.coe_finsetWalkLength_eq` for the relationship between this `Finset` and
the set of length-`n` walks. -/
def finsetWalkLength (n : ℕ) (u v : V) : Finset (G.Walk u v) :=
  match n with
  | 0 =>
    if h : u = v then by
      /-
        V : Type u
        G : SimpleGraph V
        inst✝¹ : DecidableEq V
        inst✝ : G.LocallyFinite
        n : Nat
        u v : V
        h : Eq u v
        ⊢ Finset (G.Walk u v)
      -/
      subst u
      /-
        V : Type u
        G : SimpleGraph V
        inst✝¹ : DecidableEq V
        inst✝ : G.LocallyFinite
        n : Nat
        v : V
        ⊢ Finset (G.Walk v v)
      -/
      exact {Walk.nil}
      /-
        🎉 no goals
      -/
    else ∅
  | n + 1 =>
    Finset.univ.biUnion fun (w : G.neighborSet u) =>
                                                                                   /-
                                                                                     V : Type u
                                                                                     G : SimpleGraph V
                                                                                     inst✝¹ : DecidableEq V
                                                                                     inst✝ : G.LocallyFinite
                                                                                     n✝ : Nat
                                                                                     u v : V
                                                                                     n : Nat
                                                                                     w : ↑(G.neighborSet u)
                                                                                     x✝¹ x✝ : G.Walk (↑w) v
                                                                                     ⊢ Eq ((fun p => SimpleGraph.Walk.cons ⋯ p) x✝¹) ((fun p => SimpleGraph.Walk.co …
                                                                                   -/
      (finsetWalkLength n w v).map ⟨fun p => Walk.cons w.property p, fun _ _ => by simp⟩
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem coe_finsetWalkLength_eq (n : ℕ) (u v : V) :
    (G.finsetWalkLength n u v : Set (G.Walk u v)) = {p : G.Walk u v | p.length = n} := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : DecidableEq V
    inst✝ : G.LocallyFinite
    n : Nat
    u v : V
    ⊢ Eq (↑(G.finsetWalkLength n u v)) (setOf fun p => Eq p.length n)
  -/
  induction' n with n ih generalizing u v
    /-
      case zero
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      u v : V
      ⊢ Eq (↑(G.finsetWalkLength 0 u v)) (setOf fun p => Eq p.length 0)
    -/
                                         /-
                                           🎉 no goals
                                         -/
  · obtain rfl | huv := eq_or_ne u v <;> simp [finsetWalkLength, set_walk_length_zero_eq_of_ne, *]
                                         /-
                                           🎉 no goals
                                         -/
  · simp only [finsetWalkLength, set_walk_length_succ_eq, Finset.coe_biUnion, Finset.mem_coe,
      Finset.mem_univ, Set.iUnion_true]
    /-
      case succ
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      n : Nat
      ih : ∀ (u v : V), Eq (↑(G.finsetWalkLength n u v)) (setOf fun p => Eq p.length …
      u v : V
      ⊢ Eq (Set.iUnion fun x => ↑(Finset.map { toFun := fun p => SimpleGraph.Walk.co …
    -/
    ext p
    simp only [mem_neighborSet, Finset.coe_map, Embedding.coeFn_mk, Set.iUnion_coe_set,
      Set.mem_iUnion, Set.mem_image, Finset.mem_coe, Set.mem_setOf_eq]
    /-
      case succ.h
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      n : Nat
      ih : ∀ (u v : V), Eq (↑(G.finsetWalkLength n u v)) (setOf fun p => Eq p.length …
      u v : V
      p : G.Walk u v
      ⊢ Iff (Exists fun i => Exists fun h => Exists fun x => And (Membership.mem (G. …
    -/
    congr!
    /-
      case succ.h.a.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_1.a
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      n : Nat
      ih : ∀ (u v : V), Eq (↑(G.finsetWalkLength n u v)) (setOf fun p => Eq p.length …
      u v : V
      p : G.Walk u v
      x✝² : V
      x✝¹ : G.Adj u x✝²
      x✝ : G.Walk x✝² v
      ⊢ Iff (Membership.mem (G.finsetWalkLength n x✝² v) x✝) (Eq x✝.length n)
    -/
    rename_i w _ q
    /-
      case succ.h.a.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_1.a
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      n : Nat
      ih : ∀ (u v : V), Eq (↑(G.finsetWalkLength n u v)) (setOf fun p => Eq p.length …
      u v : V
      p : G.Walk u v
      w : V
      x✝ : G.Adj u w
      q : G.Walk w v
      ⊢ Iff (Membership.mem (G.finsetWalkLength n w v) q) (Eq q.length n)
    -/
    have := Set.ext_iff.mp (ih w v) q
    /-
      case succ.h.a.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_1.a
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      n : Nat
      ih : ∀ (u v : V), Eq (↑(G.finsetWalkLength n u v)) (setOf fun p => Eq p.length …
      u v : V
      p : G.Walk u v
      w : V
      x✝ : G.Adj u w
      q : G.Walk w v
      this : Iff (Membership.mem (↑(G.finsetWalkLength n w v)) q) (Membership.mem (s …
      ⊢ Iff (Membership.mem (G.finsetWalkLength n w v) q) (Eq q.length n)
    -/
    simp only [Finset.mem_coe, Set.mem_setOf_eq] at this
    /-
      case succ.h.a.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_1.a
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      n : Nat
      ih : ∀ (u v : V), Eq (↑(G.finsetWalkLength n u v)) (setOf fun p => Eq p.length …
      u v : V
      p : G.Walk u v
      w : V
      x✝ : G.Adj u w
      q : G.Walk w v
      this : Iff (Membership.mem (G.finsetWalkLength n w v) q) (Eq q.length n)
      ⊢ Iff (Membership.mem (G.finsetWalkLength n w v) q) (Eq q.length n)
    -/
    rw [← this]
    /-
      🎉 no goals
    -/


theorem mem_finsetWalkLength_iff {n : ℕ} {u v : V} {p : G.Walk u v} :
    p ∈ G.finsetWalkLength n u v ↔ p.length = n :=
  Set.ext_iff.mp (G.coe_finsetWalkLength_eq n u v) p


/-- The `Finset` of walks from `u` to `v` with length less than `n`. See `finsetWalkLength` for
context. In particular, we use this definition for `SimpleGraph.Path.instFintype`. --/
def finsetWalkLengthLT (n : ℕ) (u v : V) : Finset (G.Walk u v) :=
  (Finset.range n).disjiUnion
    (fun l ↦ G.finsetWalkLength l u v)
    (fun l _ l' _ hne _ hsl hsl' p hp ↦
      have hl : p.length = l := mem_finsetWalkLength_iff.mp (hsl hp)
      have hl' : p.length = l' := mem_finsetWalkLength_iff.mp (hsl' hp)
      False.elim <| hne <| hl.symm.trans hl')


open Finset in
theorem coe_finsetWalkLengthLT_eq (n : ℕ) (u v : V) :
    (G.finsetWalkLengthLT n u v : Set (G.Walk u v)) = {p : G.Walk u v | p.length < n} := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : DecidableEq V
    inst✝ : G.LocallyFinite
    n : Nat
    u v : V
    ⊢ Eq (↑(G.finsetWalkLengthLT n u v)) (setOf fun p => LT.lt p.length n)
  -/
  ext p
  /-
    case h
    V : Type u
    G : SimpleGraph V
    inst✝¹ : DecidableEq V
    inst✝ : G.LocallyFinite
    n : Nat
    u v : V
    p : G.Walk u v
    ⊢ Iff (Membership.mem (↑(G.finsetWalkLengthLT n u v)) p) (Membership.mem (setO …
  -/
  simp [finsetWalkLengthLT, mem_coe, mem_disjiUnion, mem_finsetWalkLength_iff]
  /-
    🎉 no goals
  -/


theorem mem_finsetWalkLengthLT_iff {n : ℕ} {u v : V} {p : G.Walk u v} :
    p ∈ G.finsetWalkLengthLT n u v ↔ p.length < n :=
  Set.ext_iff.mp (G.coe_finsetWalkLengthLT_eq n u v) p


instance fintypeSetWalkLength (u v : V) (n : ℕ) : Fintype {p : G.Walk u v | p.length = n} :=
  Fintype.ofFinset (G.finsetWalkLength n u v) fun p => by
    /-
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      u v : V
      n : Nat
      p : G.Walk u v
      ⊢ Iff (Membership.mem (G.finsetWalkLength n u v) p) (Membership.mem (setOf fun …
    -/
    rw [← Finset.mem_coe, coe_finsetWalkLength_eq]
    /-
      🎉 no goals
    -/


instance fintypeSubtypeWalkLength (u v : V) (n : ℕ) : Fintype {p : G.Walk u v // p.length = n} :=
  fintypeSetWalkLength G u v n


theorem set_walk_length_toFinset_eq (n : ℕ) (u v : V) :
    {p : G.Walk u v | p.length = n}.toFinset = G.finsetWalkLength n u v := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : DecidableEq V
    inst✝ : G.LocallyFinite
    n : Nat
    u v : V
    ⊢ Eq (setOf fun p => Eq p.length n).toFinset (G.finsetWalkLength n u v)
  -/
  ext p
  /-
    case h
    V : Type u
    G : SimpleGraph V
    inst✝¹ : DecidableEq V
    inst✝ : G.LocallyFinite
    n : Nat
    u v : V
    p : G.Walk u v
    ⊢ Iff (Membership.mem (setOf fun p => Eq p.length n).toFinset p) (Membership.m …
  -/
  simp [← coe_finsetWalkLength_eq]
  /-
    🎉 no goals
  -/

/- See `SimpleGraph.adjMatrix_pow_apply_eq_card_walk` for the cardinality in terms of the `n`th
power of the adjacency matrix. -/

theorem card_set_walk_length_eq (u v : V) (n : ℕ) :
    Fintype.card {p : G.Walk u v | p.length = n} = #(G.finsetWalkLength n u v) :=
  Fintype.card_ofFinset (G.finsetWalkLength n u v) fun p => by
    /-
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      u v : V
      n : Nat
      p : G.Walk u v
      ⊢ Iff (Membership.mem (G.finsetWalkLength n u v) p) (Membership.mem (setOf fun …
    -/
    rw [← Finset.mem_coe, coe_finsetWalkLength_eq]
    /-
      🎉 no goals
    -/


instance fintypeSetWalkLengthLT (u v : V) (n : ℕ) : Fintype {p : G.Walk u v | p.length < n} :=
  Fintype.ofFinset (G.finsetWalkLengthLT n u v) fun p ↦ by
    /-
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      u v : V
      n : Nat
      p : G.Walk u v
      ⊢ Iff (Membership.mem (G.finsetWalkLengthLT n u v) p) (Membership.mem (setOf f …
    -/
    rw [← Finset.mem_coe, coe_finsetWalkLengthLT_eq]
    /-
      🎉 no goals
    -/


instance fintypeSubtypeWalkLengthLT (u v : V) (n : ℕ) : Fintype {p : G.Walk u v // p.length < n} :=
  fintypeSetWalkLengthLT G u v n


instance fintypeSetPathLength (u v : V) (n : ℕ) :
    Fintype {p : G.Walk u v | p.IsPath ∧ p.length = n} :=
  Fintype.ofFinset {w ∈ G.finsetWalkLength n u v | w.IsPath} <| by
    /-
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      u v : V
      n : Nat
      ⊢ ∀ (x : G.Walk u v), Iff (Membership.mem (Finset.filter (fun w => w.IsPath) ( …
    -/
    simp [mem_finsetWalkLength_iff, and_comm]
    /-
      🎉 no goals
    -/


instance fintypeSubtypePathLength (u v : V) (n : ℕ) :
    Fintype {p : G.Walk u v // p.IsPath ∧ p.length = n} :=
  fintypeSetPathLength G u v n


instance fintypeSetPathLengthLT (u v : V) (n : ℕ) :
    Fintype {p : G.Walk u v | p.IsPath ∧ p.length < n} :=
  Fintype.ofFinset {w ∈ G.finsetWalkLengthLT n u v | w.IsPath} <| by
    /-
      V : Type u
      G : SimpleGraph V
      inst✝¹ : DecidableEq V
      inst✝ : G.LocallyFinite
      u v : V
      n : Nat
      ⊢ ∀ (x : G.Walk u v), Iff (Membership.mem (Finset.filter (fun w => w.IsPath) ( …
    -/
    simp [mem_finsetWalkLengthLT_iff, and_comm]
    /-
      🎉 no goals
    -/


instance fintypeSubtypePathLengthLT (u v : V) (n : ℕ) :
    Fintype {p : G.Walk u v // p.IsPath ∧ p.length < n} :=
  fintypeSetPathLengthLT G u v n


theorem reachable_iff_exists_finsetWalkLength_nonempty (u v : V) :
    G.Reachable u v ↔ ∃ n : Fin (Fintype.card V), (G.finsetWalkLength n u v).Nonempty := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    u v : V
    ⊢ Iff (G.Reachable u v) (Exists fun n => (G.finsetWalkLength (↑n) u v).Nonempty)
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      u v : V
      ⊢ G.Reachable u v → Exists fun n => (G.finsetWalkLength (↑n) u v).Nonempty
    -/
  · intro r
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      u v : V
      r : G.Reachable u v
      ⊢ Exists fun n => (G.finsetWalkLength (↑n) u v).Nonempty
    -/
    refine r.elim_path fun p => ?_
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      u v : V
      r : G.Reachable u v
      p : G.Path u v
      ⊢ Exists fun n => (G.finsetWalkLength (↑n) u v).Nonempty
    -/
    refine ⟨⟨_, p.isPath.length_lt⟩, p, ?_⟩
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      u v : V
      r : G.Reachable u v
      p : G.Path u v
      ⊢ Membership.mem (G.finsetWalkLength (↑⟨(↑p).length, ⋯⟩) u v) ↑p
    -/
    simp [mem_finsetWalkLength_iff]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      u v : V
      ⊢ (Exists fun n => (G.finsetWalkLength (↑n) u v).Nonempty) → G.Reachable u v
    -/
  · rintro ⟨_, p, _⟩
    /-
      case mpr.intro.intro
      V : Type u
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      u v : V
      w✝ : Fin (Fintype.card V)
      p : G.Walk u v
      h✝ : Membership.mem (G.finsetWalkLength (↑w✝) u v) p
      ⊢ G.Reachable u v
    -/
    exact ⟨p⟩
    /-
      🎉 no goals
    -/


instance : DecidableRel G.Reachable := fun u v =>
  decidable_of_iff' _ (reachable_iff_exists_finsetWalkLength_nonempty G u v)


instance : Fintype G.ConnectedComponent :=
  @Quotient.fintype _ _ G.reachableSetoid (inferInstance : DecidableRel G.Reachable)


instance : Decidable G.Preconnected :=
  inferInstanceAs <| Decidable (∀ u v, G.Reachable u v)


instance : Decidable G.Connected :=
  decidable_of_iff (G.Preconnected ∧ (Finset.univ : Finset V).Nonempty) <| by
    /-
      V : Type u
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      ⊢ Iff (And G.Preconnected Finset.univ.Nonempty) G.Connected
    -/
    rw [connected_iff, ← Finset.univ_nonempty_iff]
    /-
      🎉 no goals
    -/


instance Path.instFintype {u v : V} : Fintype (G.Path u v) where
  elems := (univ (α := { p : G.Walk u v | p.IsPath ∧ p.length < Fintype.card V })).map
    ⟨fun p ↦ { val := p.val, property := p.prop.left },
     fun _ _ h ↦ SetCoe.ext <| Subtype.mk.injEq .. ▸ h⟩
  complete p := mem_map.mpr ⟨
    ⟨p.val, ⟨p.prop, p.prop.length_lt⟩⟩,
    ⟨mem_univ _, rfl⟩⟩


instance instDecidableMemSupp (c : G.ConnectedComponent) (v : V) : Decidable (v ∈ c.supp) :=
                                                            /-
                                                              V : Type u
                                                              G : SimpleGraph V
                                                              inst✝² : DecidableEq V
                                                              inst✝¹ : Fintype V
                                                              inst✝ : DecidableRel G.Adj
                                                              c : G.ConnectedComponent
                                                              v w : V
                                                              ⊢ Iff (G.Reachable v w) (Membership.mem (G.connectedComponentMk w).supp v)
                                                            -/
  c.recOn (fun w ↦ decidable_of_iff (G.Reachable v w) <| by simp)
                                                            /-
                                                              🎉 no goals
                                                            -/
    (fun _ _ _ _ ↦ Subsingleton.elim _ _)


variable {G} in
lemma disjiUnion_supp_toFinset_eq_supp_toFinset {G' : SimpleGraph V} (h : G ≤ G')
    (c' : ConnectedComponent G') [Fintype c'.supp]
    [DecidablePred fun c : G.ConnectedComponent ↦ c.supp ⊆ c'.supp] :
    .disjiUnion {c : ConnectedComponent G | c.supp ⊆ c'.supp} (fun c ↦ c.supp.toFinset)
                            /-
                              V : Type u
                              G : SimpleGraph V
                              inst✝⁴ : DecidableEq V
                              inst✝³ : Fintype V
                              inst✝² : DecidableRel G.Adj
                              G' : SimpleGraph V
                              h : LE.le G G'
                              c' : G'.ConnectedComponent
                              inst✝¹ : Fintype ↑c'.supp
                              inst✝ : DecidablePred fun c => HasSubset.Subset c.supp c'.supp
                              x : G.ConnectedComponent
                              x✝¹ : Membership.mem (↑(Finset.filter (fun c => HasSubset.Subset c.supp c'.sup …
                              y : G.ConnectedComponent
                              x✝ : Membership.mem (↑(Finset.filter (fun c => HasSubset.Subset c.supp c'.supp …
                              hxy : Ne x y
                              ⊢ Function.onFun Disjoint (fun c => c.supp.toFinset) x y
                            -/
      (fun x _ y _ hxy ↦ by simpa using pairwise_disjoint_supp_connectedComponent _ hxy) =
                            /-
                              🎉 no goals
                            -/
      c'.supp.toFinset :=
                             /-
                               V : Type u
                               G : SimpleGraph V
                               inst✝⁴ : DecidableEq V
                               inst✝³ : Fintype V
                               inst✝² : DecidableRel G.Adj
                               G' : SimpleGraph V
                               h : LE.le G G'
                               c' : G'.ConnectedComponent
                               inst✝¹ : Fintype ↑c'.supp
                               inst✝ : DecidablePred fun c => HasSubset.Subset c.supp c'.supp
                               ⊢ Eq ↑((Finset.filter (fun c => HasSubset.Subset c.supp c'.supp) Finset.univ). …
                             -/
  Finset.coe_injective <| by simpa using ConnectedComponent.biUnion_supp_eq_supp h _
                             /-
                               🎉 no goals
                             -/


lemma ConnectedComponent.odd_card_supp_iff_odd_subcomponents [Finite V] {G'}
    (h : G ≤ G') (c' : ConnectedComponent G') :
    Odd (Nat.card c'.supp) ↔ Odd (Nat.card
    ({c : ConnectedComponent G | c.supp ⊆ c'.supp ∧ Odd (Nat.card c.supp) })) := by
  classical
  cases nonempty_fintype V
  rw [Nat.card_eq_card_toFinset, ← disjiUnion_supp_toFinset_eq_supp_toFinset h]
  simp only [Finset.card_disjiUnion, Set.toFinset_card]
  rw [Finset.odd_sum_iff_odd_card_odd, Nat.card_eq_fintype_card, Fintype.card_ofFinset]
  simp only [Nat.card_eq_fintype_card, Finset.filter_filter]
  rfl


lemma odd_card_iff_odd_components [Finite V] : Odd (Nat.card V) ↔
    Odd (Nat.card ({(c : ConnectedComponent G) | Odd (Nat.card c.supp)})) := by
  classical
  cases nonempty_fintype V
  rw [Nat.card_eq_fintype_card]
  simp only [← (set_fintype_card_eq_univ_iff _).mpr G.iUnion_connectedComponentSupp,
    ConnectedComponent.mem_supp_iff, Fintype.card_subtype_compl,
    ← Set.toFinset_card, Set.toFinset_iUnion ConnectedComponent.supp]
  rw [Finset.card_biUnion
    (fun x _ y _ hxy ↦ Set.disjoint_toFinset.mpr (pairwise_disjoint_supp_connectedComponent _ hxy))]
  simp_rw [Set.toFinset_card, ← Nat.card_eq_fintype_card]
  rw [Nat.card_eq_fintype_card, Fintype.card_ofFinset]
  exact (Finset.odd_sum_iff_odd_card_odd (fun x : G.ConnectedComponent ↦ Nat.card x.supp))


