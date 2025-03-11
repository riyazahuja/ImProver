/-- The edges of a trail as a finset, since each edge in a trail appears exactly once. -/
abbrev IsTrail.edgesFinset {u v : V} {p : G.Walk u v} (h : p.IsTrail) : Finset (Sym2 V) :=
  ⟨p.edges, h.edges_nodup⟩


theorem IsTrail.even_countP_edges_iff {u v : V} {p : G.Walk u v} (ht : p.IsTrail) (x : V) :
    Even (p.edges.countP fun e => x ∈ e) ↔ u ≠ v → x ≠ u ∧ x ≠ v := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    p : G.Walk u v
    ht : p.IsTrail
    x : V
    ⊢ Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p.ed …
  -/
  induction' p with u u v w huv p ih
    /-
      case nil
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u✝ v x u : V
      ht : SimpleGraph.Walk.nil.IsTrail
      ⊢ Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) Simp …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u✝ v✝ x u v w : V
      huv : G.Adj u v
      p : G.Walk v w
      ih : p.IsTrail → Iff (Even (List.countP (fun e => Decidable.decide (Membership …
      ht : (SimpleGraph.Walk.cons huv p).IsTrail
      ⊢ Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) (Sim …
    -/
  · rw [cons_isTrail_iff] at ht
    /-
      case cons
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u✝ v✝ x u v w : V
      huv : G.Adj u v
      p : G.Walk v w
      ih : p.IsTrail → Iff (Even (List.countP (fun e => Decidable.decide (Membership …
      ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
      ⊢ Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) (Sim …
    -/
    specialize ih ht.1
    /-
      case cons
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u✝ v✝ x u v w : V
      huv : G.Adj u v
      p : G.Walk v w
      ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
      ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
      ⊢ Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) (Sim …
    -/
    simp only [List.countP_cons, Ne, edges_cons, Sym2.mem_iff]
    /-
      case cons
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u✝ v✝ x u v w : V
      huv : G.Adj u v
      p : G.Walk v w
      ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
      ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
      ⊢ Iff (Even (HAdd.hAdd (List.countP (fun e => Decidable.decide (Membership.mem …
    -/
    split_ifs with h
      /-
        case pos
        V : Type u_1
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u✝ v✝ x u v w : V
        huv : G.Adj u v
        p : G.Walk v w
        ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
        ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
        h : Eq (Decidable.decide (Or (Eq x u) (Eq x v))) Bool.true
        ⊢ Iff (Even (HAdd.hAdd (List.countP (fun e => Decidable.decide (Membership.mem …
      -/
    · rw [decide_eq_true_eq] at h
      /-
        case pos
        V : Type u_1
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u✝ v✝ x u v w : V
        huv : G.Adj u v
        p : G.Walk v w
        ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
        ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
        h : Or (Eq x u) (Eq x v)
        ⊢ Iff (Even (HAdd.hAdd (List.countP (fun e => Decidable.decide (Membership.mem …
      -/
      obtain (rfl | rfl) := h
        /-
          case pos.inl
          V : Type u_1
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u v✝ x v w : V
          p : G.Walk v w
          ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
          huv : G.Adj x v
          ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := x, snd := v  …
          ⊢ Iff (Even (HAdd.hAdd (List.countP (fun e => Decidable.decide (Membership.mem …
        -/
      · rw [Nat.even_add_one, ih]
        simp only [huv.ne, imp_false, Ne, not_false_iff, true_and, not_forall,
          Classical.not_not, exists_prop, eq_self_iff_true, not_true, false_and,
          and_iff_right_iff_imp]
        /-
          case pos.inl
          V : Type u_1
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u v✝ x v w : V
          p : G.Walk v w
          ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
          huv : G.Adj x v
          ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := x, snd := v  …
          ⊢ Eq x w → Not (Eq v w)
        -/
        rintro rfl rfl
        /-
          case pos.inl
          V : Type u_1
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u v✝ v : V
          huv : G.Adj v v
          p : G.Walk v v
          ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e v)) p …
          ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := v, snd := v  …
          ⊢ False
        -/
        exact G.loopless _ huv
        /-
          🎉 no goals
        -/
        /-
          case pos.inr
          V : Type u_1
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u✝ v x u w : V
          huv : G.Adj u x
          p : G.Walk x w
          ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := x  …
          ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
          ⊢ Iff (Even (HAdd.hAdd (List.countP (fun e => Decidable.decide (Membership.mem …
        -/
      · rw [Nat.even_add_one, ih, ← not_iff_not]
        simp only [huv.ne.symm, Ne, eq_self_iff_true, not_true, false_and, not_forall,
          not_false_iff, exists_prop, and_true, Classical.not_not, true_and, iff_and_self]
        /-
          case pos.inr
          V : Type u_1
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u✝ v x u w : V
          huv : G.Adj u x
          p : G.Walk x w
          ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := x  …
          ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
          ⊢ Eq x w → Not (Eq u w)
        -/
        rintro rfl
        /-
          case pos.inr
          V : Type u_1
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u✝ v x u : V
          huv : G.Adj u x
          p : G.Walk x x
          ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := x  …
          ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
          ⊢ Not (Eq u x)
        -/
        exact huv.ne
        /-
          🎉 no goals
        -/
      /-
        case neg
        V : Type u_1
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u✝ v✝ x u v w : V
        huv : G.Adj u v
        p : G.Walk v w
        ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
        ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
        h : Not (Eq (Decidable.decide (Or (Eq x u) (Eq x v))) Bool.true)
        ⊢ Iff (Even (HAdd.hAdd (List.countP (fun e => Decidable.decide (Membership.mem …
      -/
    · rw [decide_eq_true_eq, not_or] at h
      /-
        case neg
        V : Type u_1
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u✝ v✝ x u v w : V
        huv : G.Adj u v
        p : G.Walk v w
        ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
        ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
        h : And (Not (Eq x u)) (Not (Eq x v))
        ⊢ Iff (Even (HAdd.hAdd (List.countP (fun e => Decidable.decide (Membership.mem …
      -/
      simp only [h.1, h.2, not_false_iff, true_and, add_zero, Ne] at ih ⊢
      /-
        case neg
        V : Type u_1
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u✝ v✝ x u v w : V
        huv : G.Adj u v
        p : G.Walk v w
        ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
        h : And (Not (Eq x u)) (Not (Eq x v))
        ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
        ⊢ Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p.ed …
      -/
      rw [ih]
      /-
        case neg
        V : Type u_1
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u✝ v✝ x u v w : V
        huv : G.Adj u v
        p : G.Walk v w
        ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
        h : And (Not (Eq x u)) (Not (Eq x v))
        ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
        ⊢ Iff (Not (Eq v w) → Not (Eq x w)) (Not (Eq u w) → Not (Eq x w))
      -/
      constructor <;>
          /-
            case neg.mp
            V : Type u_1
            G : SimpleGraph V
            inst✝ : DecidableEq V
            u✝ v✝ x u v w : V
            huv : G.Adj u v
            p : G.Walk v w
            ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
            h : And (Not (Eq x u)) (Not (Eq x v))
            ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
            ⊢ (Not (Eq v w) → Not (Eq x w)) → Not (Eq u w) → Not (Eq x w)
          -/
          /-
            case neg.mp
            V : Type u_1
            G : SimpleGraph V
            inst✝ : DecidableEq V
            u✝ v✝ x u v : V
            huv : G.Adj u v
            h : And (Not (Eq x u)) (Not (Eq x v))
            p : G.Walk v x
            ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
            ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
            h' : Not (Eq v x) → Not (Eq x x)
            h'' : Not (Eq u x)
            ⊢ False
          -/
          /-
            case neg.mp
            V : Type u_1
            G : SimpleGraph V
            inst✝ : DecidableEq V
            u✝ v✝ x u v : V
            huv : G.Adj u v
            h : And (Not (Eq x u)) (Not (Eq x v))
            p : G.Walk v x
            ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
            ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
            h'' : Not (Eq u x)
            h' : Eq v x
            ⊢ False
          -/
          /-
            case neg.mp.refl
            V : Type u_1
            G : SimpleGraph V
            inst✝ : DecidableEq V
            u✝ v x u : V
            h'' : Not (Eq u x)
            huv : G.Adj u x
            h : And (Not (Eq x u)) (Not (Eq x x))
            p : G.Walk x x
            ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := x  …
            ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
            ⊢ False
          -/
          /-
            🎉 no goals
          -/
          /-
            case neg.mpr
            V : Type u_1
            G : SimpleGraph V
            inst✝ : DecidableEq V
            u✝ v✝ x u v : V
            huv : G.Adj u v
            h : And (Not (Eq x u)) (Not (Eq x v))
            p : G.Walk v x
            ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v  …
            ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
            h'' : Not (Eq v x)
            h' : Eq u x
            ⊢ False
          -/
          cases h'
          /-
            case neg.mpr.refl
            V : Type u_1
            G : SimpleGraph V
            inst✝ : DecidableEq V
            u v✝ x v : V
            p : G.Walk v x
            ih : Iff (Even (List.countP (fun e => Decidable.decide (Membership.mem e x)) p …
            h'' : Not (Eq v x)
            huv : G.Adj x v
            h : And (Not (Eq x x)) (Not (Eq x v))
            ht : And p.IsTrail (Not (Membership.mem p.edges (Sym2.mk { fst := x, snd := v  …
            ⊢ False
          -/
          simp only [not_true, and_false, false_and] at h
          /-
            🎉 no goals
          -/


/-- An *Eulerian trail* (also known as an "Eulerian path") is a walk
`p` that visits every edge exactly once.  The lemma `SimpleGraph.Walk.IsEulerian.IsTrail` shows
that these are trails.

Combine with `p.IsCircuit` to get an Eulerian circuit (also known as an "Eulerian cycle"). -/
def IsEulerian {u v : V} (p : G.Walk u v) : Prop :=
  ∀ e, e ∈ G.edgeSet → p.edges.count e = 1


theorem IsEulerian.isTrail {u v : V} {p : G.Walk u v} (h : p.IsEulerian) : p.IsTrail := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    p : G.Walk u v
    h : p.IsEulerian
    ⊢ p.IsTrail
  -/
  rw [isTrail_def, List.nodup_iff_count_le_one]
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    p : G.Walk u v
    h : p.IsEulerian
    ⊢ ∀ (a : Sym2 V), LE.le (List.count a p.edges) 1
  -/
  intro e
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    p : G.Walk u v
    h : p.IsEulerian
    e : Sym2 V
    ⊢ LE.le (List.count e p.edges) 1
  -/
  by_cases he : e ∈ p.edges
    /-
      case pos
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v : V
      p : G.Walk u v
      h : p.IsEulerian
      e : Sym2 V
      he : Membership.mem p.edges e
      ⊢ LE.le (List.count e p.edges) 1
    -/
  · exact (h e (edges_subset_edgeSet _ he)).le
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v : V
      p : G.Walk u v
      h : p.IsEulerian
      e : Sym2 V
      he : Not (Membership.mem p.edges e)
      ⊢ LE.le (List.count e p.edges) 1
    -/
  · simp [List.count_eq_zero_of_not_mem he]
    /-
      🎉 no goals
    -/


theorem IsEulerian.mem_edges_iff {u v : V} {p : G.Walk u v} (h : p.IsEulerian) {e : Sym2 V} :
    e ∈ p.edges ↔ e ∈ G.edgeSet :=
  ⟨ fun h => p.edges_subset_edgeSet h
                 /-
                   V : Type u_1
                   G : SimpleGraph V
                   inst✝ : DecidableEq V
                   u v : V
                   p : G.Walk u v
                   h : p.IsEulerian
                   e : Sym2 V
                   he : Membership.mem G.edgeSet e
                   ⊢ Membership.mem p.edges e
                 -/
  , fun he => by simpa [Nat.succ_le] using (h e he).ge ⟩
                 /-
                   🎉 no goals
                 -/


/-- The edge set of an Eulerian graph is finite. -/
def IsEulerian.fintypeEdgeSet {u v : V} {p : G.Walk u v} (h : p.IsEulerian) :
    Fintype G.edgeSet :=
  Fintype.ofFinset h.isTrail.edgesFinset fun e => by
    /-
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v : V
      p : G.Walk u v
      h : p.IsEulerian
      e : Sym2 V
      ⊢ Iff (Membership.mem ⋯.edgesFinset e) (Membership.mem G.edgeSet e)
    -/
    simp only [Finset.mem_mk, Multiset.mem_coe, h.mem_edges_iff]
    /-
      🎉 no goals
    -/


theorem IsTrail.isEulerian_of_forall_mem {u v : V} {p : G.Walk u v} (h : p.IsTrail)
    (hc : ∀ e, e ∈ G.edgeSet → e ∈ p.edges) : p.IsEulerian := fun e he =>
  List.count_eq_one_of_mem h.edges_nodup (hc e he)


theorem isEulerian_iff {u v : V} (p : G.Walk u v) :
    p.IsEulerian ↔ p.IsTrail ∧ ∀ e, e ∈ G.edgeSet → e ∈ p.edges := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    p : G.Walk u v
    ⊢ Iff p.IsEulerian (And p.IsTrail (∀ (e : Sym2 V), Membership.mem G.edgeSet e  …
  -/
  constructor
    /-
      case mp
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v : V
      p : G.Walk u v
      ⊢ p.IsEulerian → And p.IsTrail (∀ (e : Sym2 V), Membership.mem G.edgeSet e → M …
    -/
  · intro h
    /-
      case mp
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v : V
      p : G.Walk u v
      h : p.IsEulerian
      ⊢ And p.IsTrail (∀ (e : Sym2 V), Membership.mem G.edgeSet e → Membership.mem p …
    -/
    exact ⟨h.isTrail, fun _ => h.mem_edges_iff.mpr⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v : V
      p : G.Walk u v
      ⊢ And p.IsTrail (∀ (e : Sym2 V), Membership.mem G.edgeSet e → Membership.mem p …
    -/
  · rintro ⟨h, hl⟩
    /-
      case mpr.intro
      V : Type u_1
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v : V
      p : G.Walk u v
      h : p.IsTrail
      hl : ∀ (e : Sym2 V), Membership.mem G.edgeSet e → Membership.mem p.edges e
      ⊢ p.IsEulerian
    -/
    exact h.isEulerian_of_forall_mem hl
    /-
      🎉 no goals
    -/


theorem IsEulerian.edgesFinset_eq [Fintype G.edgeSet] {u v : V} {p : G.Walk u v}
    (h : p.IsEulerian) : h.isTrail.edgesFinset = G.edgeFinset := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : DecidableEq V
    inst✝ : Fintype ↑G.edgeSet
    u v : V
    p : G.Walk u v
    h : p.IsEulerian
    ⊢ Eq ⋯.edgesFinset G.edgeFinset
  -/
  ext e
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    inst✝¹ : DecidableEq V
    inst✝ : Fintype ↑G.edgeSet
    u v : V
    p : G.Walk u v
    h : p.IsEulerian
    e : Sym2 V
    ⊢ Iff (Membership.mem ⋯.edgesFinset e) (Membership.mem G.edgeFinset e)
  -/
  simp [h.mem_edges_iff]
  /-
    🎉 no goals
  -/


theorem IsEulerian.even_degree_iff {x u v : V} {p : G.Walk u v} (ht : p.IsEulerian) [Fintype V]
    [DecidableRel G.Adj] : Even (G.degree x) ↔ u ≠ v → x ≠ u ∧ x ≠ v := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    x u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ Iff (Even (G.degree x)) (Ne u v → And (Ne x u) (Ne x v))
  -/
  convert ht.isTrail.even_countP_edges_iff x
  /-
    case h.e'_1.h.e'_3
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    x u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ Eq (G.degree x) (List.countP (fun e => Decidable.decide (Membership.mem e x) …
  -/
  rw [← Multiset.coe_countP, Multiset.countP_eq_card_filter, ← card_incidenceFinset_eq_degree]
  /-
    case h.e'_1.h.e'_3
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    x u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ Eq (G.incidenceFinset x).card (Multiset.filter (fun e => Membership.mem e x) …
  -/
  change Multiset.card _ = _
  /-
    case h.e'_1.h.e'_3
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    x u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ Eq (G.incidenceFinset x).val.card (Multiset.filter (fun e => Membership.mem  …
  -/
  congr 1
  /-
    case h.e'_1.h.e'_3.e_s
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    x u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ Eq (G.incidenceFinset x).val (Multiset.filter (fun e => Membership.mem e x)  …
  -/
  convert_to _ = (ht.isTrail.edgesFinset.filter (x ∈ ·)).val
  /-
    case h.e'_1.h.e'_3.e_s.convert_2
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    x u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ Eq (G.incidenceFinset x).val (Finset.filter (fun x_1 => Membership.mem x_1 x …
  -/
  have : Fintype G.edgeSet := fintypeEdgeSet ht
  /-
    case h.e'_1.h.e'_3.e_s.convert_2
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    x u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    this : Fintype ↑G.edgeSet
    ⊢ Eq (G.incidenceFinset x).val (Finset.filter (fun x_1 => Membership.mem x_1 x …
  -/
  rw [ht.edgesFinset_eq, G.incidenceFinset_eq_filter x]
  /-
    🎉 no goals
  -/


theorem IsEulerian.card_filter_odd_degree [Fintype V] [DecidableRel G.Adj] {u v : V}
    {p : G.Walk u v} (ht : p.IsEulerian) {s}
    (h : s = (Finset.univ : Finset V).filter fun v => Odd (G.degree v)) :
    s.card = 0 ∨ s.card = 2 := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    s : Finset V
    h : Eq s (Finset.filter (fun v => Odd (G.degree v)) Finset.univ)
    ⊢ Or (Eq s.card 0) (Eq s.card 2)
  -/
  subst s
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    ⊢ Or (Eq (Finset.filter (fun v => Odd (G.degree v)) Finset.univ).card 0) (Eq ( …
  -/
  simp only [← Nat.not_even_iff_odd, Finset.card_eq_zero]
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    ⊢ Or (Eq (Finset.filter (fun v => Not (Even (G.degree v))) Finset.univ) EmptyC …
  -/
  simp only [ht.even_degree_iff, Ne, not_forall, not_and, Classical.not_not, exists_prop]
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    ⊢ Or (Eq (Finset.filter (fun v_1 => And (Not (Eq u v)) (Not (Eq v_1 u) → Eq v_ …
  -/
  obtain rfl | hn := eq_or_ne u v
    /-
      case inl
      V : Type u_1
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      u : V
      p : G.Walk u u
      ht : p.IsEulerian
      ⊢ Or (Eq (Finset.filter (fun v => And (Not (Eq u u)) (Not (Eq v u) → Eq v u))  …
    -/
  · left
    /-
      case inl.h
      V : Type u_1
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      u : V
      p : G.Walk u u
      ht : p.IsEulerian
      ⊢ Eq (Finset.filter (fun v => And (Not (Eq u u)) (Not (Eq v u) → Eq v u)) Fins …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      u v : V
      p : G.Walk u v
      ht : p.IsEulerian
      hn : Ne u v
      ⊢ Or (Eq (Finset.filter (fun v_1 => And (Not (Eq u v)) (Not (Eq v_1 u) → Eq v_ …
    -/
  · right
    /-
      case inr.h
      V : Type u_1
      G : SimpleGraph V
      inst✝² : DecidableEq V
      inst✝¹ : Fintype V
      inst✝ : DecidableRel G.Adj
      u v : V
      p : G.Walk u v
      ht : p.IsEulerian
      hn : Ne u v
      ⊢ Eq (Finset.filter (fun v_1 => And (Not (Eq u v)) (Not (Eq v_1 u) → Eq v_1 v) …
    -/
    convert_to _ = ({u, v} : Finset V).card
      /-
        case h.e'_3
        V : Type u_1
        G : SimpleGraph V
        inst✝² : DecidableEq V
        inst✝¹ : Fintype V
        inst✝ : DecidableRel G.Adj
        u v : V
        p : G.Walk u v
        ht : p.IsEulerian
        hn : Ne u v
        ⊢ Eq 2 (Insert.insert u (Singleton.singleton v)).card
      -/
    · simp [hn]
      /-
        🎉 no goals
      -/
      /-
        case inr.h.convert_2
        V : Type u_1
        G : SimpleGraph V
        inst✝² : DecidableEq V
        inst✝¹ : Fintype V
        inst✝ : DecidableRel G.Adj
        u v : V
        p : G.Walk u v
        ht : p.IsEulerian
        hn : Ne u v
        ⊢ Eq (Finset.filter (fun v_1 => And (Not (Eq u v)) (Not (Eq v_1 u) → Eq v_1 v) …
      -/
    · congr
      /-
        case inr.h.convert_2.e_s
        V : Type u_1
        G : SimpleGraph V
        inst✝² : DecidableEq V
        inst✝¹ : Fintype V
        inst✝ : DecidableRel G.Adj
        u v : V
        p : G.Walk u v
        ht : p.IsEulerian
        hn : Ne u v
        ⊢ Eq (Finset.filter (fun v_1 => And (Not (Eq u v)) (Not (Eq v_1 u) → Eq v_1 v) …
      -/
      ext x
      /-
        case inr.h.convert_2.e_s.h
        V : Type u_1
        G : SimpleGraph V
        inst✝² : DecidableEq V
        inst✝¹ : Fintype V
        inst✝ : DecidableRel G.Adj
        u v : V
        p : G.Walk u v
        ht : p.IsEulerian
        hn : Ne u v
        x : V
        ⊢ Iff (Membership.mem (Finset.filter (fun v_1 => And (Not (Eq u v)) (Not (Eq v …
      -/
      simp [hn, imp_iff_not_or]
      /-
        🎉 no goals
      -/


theorem IsEulerian.card_odd_degree [Fintype V] [DecidableRel G.Adj] {u v : V} {p : G.Walk u v}
    (ht : p.IsEulerian) : Fintype.card { v : V | Odd (G.degree v) } = 0 ∨
      Fintype.card { v : V | Odd (G.degree v) } = 2 := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    ⊢ Or (Eq (Fintype.card ↑(setOf fun v => Odd (G.degree v))) 0) (Eq (Fintype.car …
  -/
  rw [← Set.toFinset_card]
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    ⊢ Or (Eq (setOf fun v => Odd (G.degree v)).toFinset.card 0) (Eq (setOf fun v = …
  -/
  apply IsEulerian.card_filter_odd_degree ht
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    u v : V
    p : G.Walk u v
    ht : p.IsEulerian
    ⊢ Eq (setOf fun v => Odd (G.degree v)).toFinset (Finset.filter (fun v => Odd ( …
  -/
  ext v
  /-
    case h
    V : Type u_1
    G : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    u v✝ : V
    p : G.Walk u v✝
    ht : p.IsEulerian
    v : V
    ⊢ Iff (Membership.mem (setOf fun v => Odd (G.degree v)).toFinset v) (Membershi …
  -/
  simp
  /-
    🎉 no goals
  -/


