variable (G) in
/-- An `r + 1`-cliquefree graph is `r`-Turán-maximal if any other `r + 1`-cliquefree graph on
the same vertex set has the same or fewer number of edges. -/
def IsTuranMaximal (r : ℕ) : Prop :=
  G.CliqueFree (r + 1) ∧ ∀ (H : SimpleGraph V) [DecidableRel H.Adj],
    H.CliqueFree (r + 1) → #H.edgeFinset ≤ #G.edgeFinset


lemma IsTuranMaximal.le_iff_eq (hG : G.IsTuranMaximal r) (hH : H.CliqueFree (r + 1)) :
    G ≤ H ↔ G = H := by
  classical exact ⟨fun hGH ↦ edgeFinset_inj.1 <| eq_of_subset_of_card_le
    (edgeFinset_subset_edgeFinset.2 hGH) (hG.2 _ hH), le_of_eq⟩


/-- The canonical `r + 1`-cliquefree Turán graph on `n` vertices. -/
def turanGraph (n r : ℕ) : SimpleGraph (Fin n) where Adj v w := v % r ≠ w % r


instance turanGraph.instDecidableRelAdj : DecidableRel (turanGraph n r).Adj := by
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    n r : Nat
    H : SimpleGraph V
    ⊢ DecidableRel (SimpleGraph.turanGraph n r).Adj
  -/
  dsimp only [turanGraph]; infer_instance
                           /-
                             🎉 no goals
                           -/


@[simp]
lemma turanGraph_zero : turanGraph n 0 = ⊤ := by
  /-
    n : Nat
    ⊢ Eq (SimpleGraph.turanGraph n 0) Top.top
  -/
  ext a b; simp_rw [turanGraph, top_adj, Nat.mod_zero, not_iff_not, Fin.val_inj]
           /-
             🎉 no goals
           -/


@[simp]
theorem turanGraph_eq_top : turanGraph n r = ⊤ ↔ r = 0 ∨ n ≤ r := by
  /-
    n r : Nat
    ⊢ Iff (Eq (SimpleGraph.turanGraph n r) Top.top) (Or (Eq r 0) (LE.le n r))
  -/
  simp_rw [SimpleGraph.ext_iff, funext_iff, turanGraph, top_adj, eq_iff_iff, not_iff_not]
  /-
    n r : Nat
    ⊢ Iff (∀ (x x_1 : Fin n), Iff (Eq (HMod.hMod (↑x) r) (HMod.hMod (↑x_1) r)) (Eq …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      n r : Nat
      h : ∀ (x x_1 : Fin n), Iff (Eq (HMod.hMod (↑x) r) (HMod.hMod (↑x_1) r)) (Eq x  …
      ⊢ Or (Eq r 0) (LE.le n r)
    -/
  · contrapose! h
    /-
      case refine_1
      n r : Nat
      h : And (Ne r 0) (LT.lt r n)
      ⊢ Exists fun x => Exists fun x_1 => Or (And (Eq (HMod.hMod (↑x) r) (HMod.hMod  …
    -/
    use ⟨0, (Nat.pos_of_ne_zero h.1).trans h.2⟩, ⟨r, h.2⟩
    /-
      case h
      n r : Nat
      h : And (Ne r 0) (LT.lt r n)
      ⊢ Or (And (Eq (HMod.hMod (↑⟨0, ⋯⟩) r) (HMod.hMod (↑⟨r, ⋯⟩) r)) (Ne ⟨0, ⋯⟩ ⟨r,  …
    -/
    simp [h.1.symm]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n r : Nat
      ⊢ Or (Eq r 0) (LE.le n r) → ∀ (x x_1 : Fin n), Iff (Eq (HMod.hMod (↑x) r) (HMo …
    -/
  · rintro (rfl | h) a b
      /-
        case refine_2.inl
        n : Nat
        a b : Fin n
        ⊢ Iff (Eq (HMod.hMod (↑a) 0) (HMod.hMod (↑b) 0)) (Eq a b)
      -/
    · simp [Fin.val_inj]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        n r : Nat
        h : LE.le n r
        a b : Fin n
        ⊢ Iff (Eq (HMod.hMod (↑a) r) (HMod.hMod (↑b) r)) (Eq a b)
      -/
    · rw [Nat.mod_eq_of_lt (a.2.trans_le h), Nat.mod_eq_of_lt (b.2.trans_le h), Fin.val_inj]
      /-
        🎉 no goals
      -/


theorem turanGraph_cliqueFree (hr : 0 < r) : (turanGraph n r).CliqueFree (r + 1) := by
  /-
    n r : Nat
    hr : LT.lt 0 r
    ⊢ (SimpleGraph.turanGraph n r).CliqueFree (HAdd.hAdd r 1)
  -/
  rw [cliqueFree_iff]
  /-
    n r : Nat
    hr : LT.lt 0 r
    ⊢ IsEmpty (Top.top.Embedding (SimpleGraph.turanGraph n r))
  -/
  by_contra h
  /-
    n r : Nat
    hr : LT.lt 0 r
    h : Not (IsEmpty (Top.top.Embedding (SimpleGraph.turanGraph n r)))
    ⊢ False
  -/
  rw [not_isEmpty_iff] at h
  /-
    n r : Nat
    hr : LT.lt 0 r
    h : Nonempty (Top.top.Embedding (SimpleGraph.turanGraph n r))
    ⊢ False
  -/
  obtain ⟨f, ha⟩ := h
  /-
    case intro.mk
    n r : Nat
    hr : LT.lt 0 r
    f : Function.Embedding (Fin (HAdd.hAdd r 1)) (Fin n)
    ha : ∀ {a b : Fin (HAdd.hAdd r 1)}, Iff ((SimpleGraph.turanGraph n r).Adj (f a …
    ⊢ False
  -/
  simp only [turanGraph, top_adj] at ha
  obtain ⟨x, y, d, c⟩ := Fintype.exists_ne_map_eq_of_card_lt (fun x ↦
    (⟨(f x).1 % r, Nat.mod_lt _ hr⟩ : Fin r)) (by simp)
  /-
    case intro.mk.intro.intro.intro
    n r : Nat
    hr : LT.lt 0 r
    f : Function.Embedding (Fin (HAdd.hAdd r 1)) (Fin n)
    ha : ∀ {a b : Fin (HAdd.hAdd r 1)}, Iff (Ne (HMod.hMod (↑(f a)) r) (HMod.hMod  …
    x y : Fin (HAdd.hAdd r 1)
    d : Ne x y
    c : Eq ⟨HMod.hMod (↑(f x)) r, ⋯⟩ ⟨HMod.hMod (↑(f y)) r, ⋯⟩
    ⊢ False
  -/
  simp only [Fin.mk.injEq] at c
  /-
    case intro.mk.intro.intro.intro
    n r : Nat
    hr : LT.lt 0 r
    f : Function.Embedding (Fin (HAdd.hAdd r 1)) (Fin n)
    ha : ∀ {a b : Fin (HAdd.hAdd r 1)}, Iff (Ne (HMod.hMod (↑(f a)) r) (HMod.hMod  …
    x y : Fin (HAdd.hAdd r 1)
    d : Ne x y
    c : Eq (HMod.hMod (↑(f x)) r) (HMod.hMod (↑(f y)) r)
    ⊢ False
  -/
  exact absurd c ((@ha x y).mpr d)
  /-
    🎉 no goals
  -/


/-- An `r + 1`-cliquefree Turán-maximal graph is _not_ `r`-cliquefree
if it can accommodate such a clique. -/
theorem not_cliqueFree_of_isTuranMaximal (hn : r ≤ Fintype.card V) (hG : G.IsTuranMaximal r) :
    ¬G.CliqueFree r := by
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    r : Nat
    hn : LE.le r (Fintype.card V)
    hG : G.IsTuranMaximal r
    ⊢ Not (G.CliqueFree r)
  -/
  rintro h
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    r : Nat
    hn : LE.le r (Fintype.card V)
    hG : G.IsTuranMaximal r
    h : G.CliqueFree r
    ⊢ False
  -/
  obtain ⟨K, _, rfl⟩ := exists_subset_card_eq hn
  obtain ⟨a, -, b, -, hab, hGab⟩ : ∃ a ∈ K, ∃ b ∈ K, a ≠ b ∧ ¬ G.Adj a b := by
    simpa only [isNClique_iff, IsClique, Set.Pairwise, mem_coe, ne_eq, and_true, not_forall,
      exists_prop, exists_and_right] using h K
  exact hGab <| le_sup_right.trans_eq ((hG.le_iff_eq <| h.sup_edge _ _).1 le_sup_left).symm <|
    (edge_adj ..).2 ⟨Or.inl ⟨rfl, rfl⟩, hab⟩


lemma exists_isTuranMaximal (hr : 0 < r):
    ∃ H : SimpleGraph V, ∃ _ : DecidableRel H.Adj, H.IsTuranMaximal r := by
  classical
  let c := {H : SimpleGraph V | H.CliqueFree (r + 1)}
  have cn : c.toFinset.Nonempty := ⟨⊥, by
    simp only [Set.toFinset_setOf, mem_filter, mem_univ, true_and, c]
    exact cliqueFree_bot (by omega)⟩
  obtain ⟨S, Sm, Sl⟩ := exists_max_image c.toFinset (#·.edgeFinset) cn
  use S, inferInstance
  rw [Set.mem_toFinset] at Sm
  refine ⟨Sm, fun I _ cf ↦ ?_⟩
  by_cases Im : I ∈ c.toFinset
  · convert Sl I Im
  · rw [Set.mem_toFinset] at Im
    contradiction


/-- In a Turán-maximal graph, non-adjacent vertices have the same degree. -/
lemma degree_eq_of_not_adj (h : G.IsTuranMaximal r) (hn : ¬G.Adj s t) :
    G.degree s = G.degree t := by
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    r : Nat
    s t : V
    h : G.IsTuranMaximal r
    hn : Not (G.Adj s t)
    ⊢ Eq (G.degree s) (G.degree t)
  -/
  rw [IsTuranMaximal] at h; contrapose! h; intro cf
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    r : Nat
    s t : V
    hn : Not (G.Adj s t)
    h : Ne (G.degree s) (G.degree t)
    cf : G.CliqueFree (HAdd.hAdd r 1)
    ⊢ Exists fun H => Exists fun [DecidableRel H.Adj] => And (H.CliqueFree (HAdd.h …
  -/
  wlog hd : G.degree t < G.degree s generalizing G t s
    /-
      case inr
      V : Type u_1
      inst✝¹ : Fintype V
      G : SimpleGraph V
      inst✝ : DecidableRel G.Adj
      r : Nat
      s t : V
      hn : Not (G.Adj s t)
      h : Ne (G.degree s) (G.degree t)
      cf : G.CliqueFree (HAdd.hAdd r 1)
      this : ∀ {G : SimpleGraph V} [inst : DecidableRel G.Adj] {s t : V}, Not (G.Adj …
      hd : Not (LT.lt (G.degree t) (G.degree s))
      ⊢ Exists fun H => Exists fun [DecidableRel H.Adj] => And (H.CliqueFree (HAdd.h …
    -/
  · replace hd : G.degree s < G.degree t := lt_of_le_of_ne (le_of_not_lt hd) h
    /-
      case inr
      V : Type u_1
      inst✝¹ : Fintype V
      G : SimpleGraph V
      inst✝ : DecidableRel G.Adj
      r : Nat
      s t : V
      hn : Not (G.Adj s t)
      h : Ne (G.degree s) (G.degree t)
      cf : G.CliqueFree (HAdd.hAdd r 1)
      this : ∀ {G : SimpleGraph V} [inst : DecidableRel G.Adj] {s t : V}, Not (G.Adj …
      hd : LT.lt (G.degree s) (G.degree t)
      ⊢ Exists fun H => Exists fun [DecidableRel H.Adj] => And (H.CliqueFree (HAdd.h …
    -/
    exact this (by rwa [adj_comm] at hn) hd.ne' cf hd
    /-
      🎉 no goals
    -/
  classical
  use G.replaceVertex s t, inferInstance, cf.replaceVertex s t
  have := G.card_edgeFinset_replaceVertex_of_not_adj hn
  omega


/-- In a Turán-maximal graph, non-adjacency is transitive. -/
lemma not_adj_trans (h : G.IsTuranMaximal r) (hts : ¬G.Adj t s) (hsu : ¬G.Adj s u) :
    ¬G.Adj t u := by
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    r : Nat
    s t u : V
    h : G.IsTuranMaximal r
    hts : Not (G.Adj t s)
    hsu : Not (G.Adj s u)
    ⊢ Not (G.Adj t u)
  -/
  have hst : ¬G.Adj s t := fun a ↦ hts a.symm
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    r : Nat
    s t u : V
    h : G.IsTuranMaximal r
    hts : Not (G.Adj t s)
    hsu : Not (G.Adj s u)
    hst : Not (G.Adj s t)
    ⊢ Not (G.Adj t u)
  -/
  have dst := h.degree_eq_of_not_adj hst
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    r : Nat
    s t u : V
    h : G.IsTuranMaximal r
    hts : Not (G.Adj t s)
    hsu : Not (G.Adj s u)
    hst : Not (G.Adj s t)
    dst : Eq (G.degree s) (G.degree t)
    ⊢ Not (G.Adj t u)
  -/
  have dsu := h.degree_eq_of_not_adj hsu
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    r : Nat
    s t u : V
    h : G.IsTuranMaximal r
    hts : Not (G.Adj t s)
    hsu : Not (G.Adj s u)
    hst : Not (G.Adj s t)
    dst : Eq (G.degree s) (G.degree t)
    dsu : Eq (G.degree s) (G.degree u)
    ⊢ Not (G.Adj t u)
  -/
  rw [IsTuranMaximal] at h; contrapose! h; intro cf
  classical
  use (G.replaceVertex s t).replaceVertex s u, inferInstance,
    (cf.replaceVertex s t).replaceVertex s u
  have nst : s ≠ t := fun a ↦ hsu (a ▸ h)
  have ntu : t ≠ u := G.ne_of_adj h
  have := (G.adj_replaceVertex_iff_of_ne s nst ntu.symm).not.mpr hsu
  rw [card_edgeFinset_replaceVertex_of_not_adj _ this,
    card_edgeFinset_replaceVertex_of_not_adj _ hst, dst, Nat.add_sub_cancel]
  have l1 : (G.replaceVertex s t).degree s = G.degree s := by
    unfold degree; congr 1; ext v
    simp only [mem_neighborFinset, SimpleGraph.irrefl, ite_self]
    by_cases eq : v = t
    · simpa only [eq, not_adj_replaceVertex_same, false_iff]
    · rw [G.adj_replaceVertex_iff_of_ne s nst eq]
  have l2 : (G.replaceVertex s t).degree u = G.degree u - 1 := by
    rw [degree, degree, ← card_singleton t, ← card_sdiff (by simp [h.symm])]
    congr 1; ext v
    simp only [mem_neighborFinset, mem_sdiff, mem_singleton, replaceVertex]
    split_ifs <;> simp_all [adj_comm]
  have l3 : 0 < G.degree u := by rw [G.degree_pos_iff_exists_adj u]; use t, h.symm
  omega


/-- In a Turán-maximal graph, non-adjacency is an equivalence relation. -/
theorem equivalence_not_adj : Equivalence (¬G.Adj · ·) where
             /-
               V : Type u_1
               inst✝¹ : Fintype V
               G : SimpleGraph V
               inst✝ : DecidableRel G.Adj
               r : Nat
               h : G.IsTuranMaximal r
               ⊢ ∀ (x : V), Not (G.Adj x x)
             -/
  refl := by simp
             /-
               🎉 no goals
             -/
             /-
               V : Type u_1
               inst✝¹ : Fintype V
               G : SimpleGraph V
               inst✝ : DecidableRel G.Adj
               r : Nat
               h : G.IsTuranMaximal r
               ⊢ ∀ {x y : V}, Not (G.Adj x y) → Not (G.Adj y x)
             -/
  symm := by simp [adj_comm]
             /-
               🎉 no goals
             -/
  trans := h.not_adj_trans


/-- The non-adjacency setoid over the vertices of a Turán-maximal graph
induced by `equivalence_not_adj`. -/
def setoid : Setoid V := ⟨_, h.equivalence_not_adj⟩


instance : DecidableRel h.setoid.r :=
  inferInstanceAs <| DecidableRel (¬G.Adj · ·)


/-- The finpartition derived from `h.setoid`. -/
def finpartition [DecidableEq V] : Finpartition (univ : Finset V) := Finpartition.ofSetoid h.setoid


lemma not_adj_iff_part_eq [DecidableEq V] :
    ¬G.Adj s t ↔ h.finpartition.part s = h.finpartition.part t := by
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    s t : V
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    ⊢ Iff (Not (G.Adj s t)) (Eq (h.finpartition.part s) (h.finpartition.part t))
  -/
  change h.setoid.r s t ↔ _
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    s t : V
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    ⊢ Iff (h.setoid s t) (Eq (h.finpartition.part s) (h.finpartition.part t))
  -/
  rw [← Finpartition.mem_part_ofSetoid_iff_rel]
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    s t : V
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    ⊢ Iff (Membership.mem ((Finpartition.ofSetoid h.setoid).part s) t) (Eq (h.finp …
  -/
  let fp := h.finpartition
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    s t : V
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    ⊢ Iff (Membership.mem ((Finpartition.ofSetoid h.setoid).part s) t) (Eq (h.finp …
  -/
  change t ∈ fp.part s ↔ fp.part s = fp.part t
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    s t : V
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    ⊢ Iff (Membership.mem (fp.part s) t) (Eq (fp.part s) (fp.part t))
  -/
  rw [fp.mem_part_iff_part_eq_part (mem_univ t) (mem_univ s), eq_comm]
  /-
    🎉 no goals
  -/


lemma degree_eq_card_sub_part_card [DecidableEq V] :
    G.degree s = Fintype.card V - #(h.finpartition.part s) :=
  calc
    _ = #{t | G.Adj s t} := by
      /-
        V : Type u_1
        inst✝² : Fintype V
        G : SimpleGraph V
        inst✝¹ : DecidableRel G.Adj
        r : Nat
        s : V
        h : G.IsTuranMaximal r
        inst✝ : DecidableEq V
        ⊢ Eq (G.degree s) (Finset.filter (fun t => G.Adj s t) Finset.univ).card
      -/
      simp [← card_neighborFinset_eq_degree, neighborFinset]
      /-
        🎉 no goals
      -/
    _ = Fintype.card V - #{t | ¬G.Adj s t} :=
      eq_tsub_of_add_eq (filter_card_add_filter_neg_card_eq_card _)
    _ = _ := by
      /-
        V : Type u_1
        inst✝² : Fintype V
        G : SimpleGraph V
        inst✝¹ : DecidableRel G.Adj
        r : Nat
        s : V
        h : G.IsTuranMaximal r
        inst✝ : DecidableEq V
        ⊢ Eq (HSub.hSub (Fintype.card V) (Finset.filter (fun t => Not (G.Adj s t)) Fin …
      -/
      congr; ext; rw [mem_filter]
      /-
        case e_a.e_s.h
        V : Type u_1
        inst✝² : Fintype V
        G : SimpleGraph V
        inst✝¹ : DecidableRel G.Adj
        r : Nat
        s : V
        h : G.IsTuranMaximal r
        inst✝ : DecidableEq V
        a✝ : V
        ⊢ Iff (And (Membership.mem Finset.univ a✝) (Not (G.Adj s a✝))) (Membership.mem …
      -/
      convert Finpartition.mem_part_ofSetoid_iff_rel.symm
      /-
        case h.e'_1.a
        V : Type u_1
        inst✝² : Fintype V
        G : SimpleGraph V
        inst✝¹ : DecidableRel G.Adj
        r : Nat
        s : V
        h : G.IsTuranMaximal r
        inst✝ : DecidableEq V
        a✝ : V
        ⊢ Iff (And (Membership.mem Finset.univ a✝) (Not (G.Adj s a✝))) (h.setoid s a✝)
      -/
      simp [setoid]
      /-
        🎉 no goals
      -/


/-- The parts of a Turán-maximal graph form an equipartition. -/
theorem isEquipartition [DecidableEq V] : h.finpartition.IsEquipartition := by
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    ⊢ h.finpartition.IsEquipartition
  -/
  set fp := h.finpartition
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    ⊢ fp.IsEquipartition
  -/
  by_contra hn
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    hn : Not fp.IsEquipartition
    ⊢ False
  -/
  rw [Finpartition.not_isEquipartition] at hn
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    hn : Exists fun a => And (Membership.mem fp.parts a) (Exists fun b => And (Mem …
    ⊢ False
  -/
  obtain ⟨large, hl, small, hs, ineq⟩ := hn
  /-
    case intro.intro.intro.intro
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    large : Finset V
    hl : Membership.mem fp.parts large
    small : Finset V
    hs : Membership.mem fp.parts small
    ineq : LT.lt (HAdd.hAdd small.card 1) large.card
    ⊢ False
  -/
  obtain ⟨w, hw⟩ := fp.nonempty_of_mem_parts hl
  /-
    case intro.intro.intro.intro.intro
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    large : Finset V
    hl : Membership.mem fp.parts large
    small : Finset V
    hs : Membership.mem fp.parts small
    ineq : LT.lt (HAdd.hAdd small.card 1) large.card
    w : V
    hw : Membership.mem large w
    ⊢ False
  -/
  obtain ⟨v, hv⟩ := fp.nonempty_of_mem_parts hs
  /-
    case intro.intro.intro.intro.intro.intro
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    large : Finset V
    hl : Membership.mem fp.parts large
    small : Finset V
    hs : Membership.mem fp.parts small
    ineq : LT.lt (HAdd.hAdd small.card 1) large.card
    w : V
    hw : Membership.mem large w
    v : V
    hv : Membership.mem small v
    ⊢ False
  -/
  apply absurd h
  /-
    case intro.intro.intro.intro.intro.intro
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    large : Finset V
    hl : Membership.mem fp.parts large
    small : Finset V
    hs : Membership.mem fp.parts small
    ineq : LT.lt (HAdd.hAdd small.card 1) large.card
    w : V
    hw : Membership.mem large w
    v : V
    hv : Membership.mem small v
    ⊢ Not (G.IsTuranMaximal r)
  -/
  rw [IsTuranMaximal]; push_neg; intro cf
  /-
    case intro.intro.intro.intro.intro.intro
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    large : Finset V
    hl : Membership.mem fp.parts large
    small : Finset V
    hs : Membership.mem fp.parts small
    ineq : LT.lt (HAdd.hAdd small.card 1) large.card
    w : V
    hw : Membership.mem large w
    v : V
    hv : Membership.mem small v
    cf : G.CliqueFree (HAdd.hAdd r 1)
    ⊢ Exists fun H => Exists fun [DecidableRel H.Adj] => And (H.CliqueFree (HAdd.h …
  -/
  use G.replaceVertex v w, inferInstance, cf.replaceVertex v w
  /-
    case right
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    large : Finset V
    hl : Membership.mem fp.parts large
    small : Finset V
    hs : Membership.mem fp.parts small
    ineq : LT.lt (HAdd.hAdd small.card 1) large.card
    w : V
    hw : Membership.mem large w
    v : V
    hv : Membership.mem small v
    cf : G.CliqueFree (HAdd.hAdd r 1)
    ⊢ LT.lt G.edgeFinset.card (G.replaceVertex v w).edgeFinset.card
  -/
  have large_eq := fp.part_eq_of_mem hl hw
  /-
    case right
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    large : Finset V
    hl : Membership.mem fp.parts large
    small : Finset V
    hs : Membership.mem fp.parts small
    ineq : LT.lt (HAdd.hAdd small.card 1) large.card
    w : V
    hw : Membership.mem large w
    v : V
    hv : Membership.mem small v
    cf : G.CliqueFree (HAdd.hAdd r 1)
    large_eq : Eq (fp.part w) large
    ⊢ LT.lt G.edgeFinset.card (G.replaceVertex v w).edgeFinset.card
  -/
  have small_eq := fp.part_eq_of_mem hs hv
  have ha : G.Adj v w := by
    by_contra hn; rw [h.not_adj_iff_part_eq, small_eq, large_eq] at hn
    rw [hn] at ineq; omega
  rw [G.card_edgeFinset_replaceVertex_of_adj ha,
    degree_eq_card_sub_part_card h, small_eq, degree_eq_card_sub_part_card h, large_eq]
  /-
    case right
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    large : Finset V
    hl : Membership.mem fp.parts large
    small : Finset V
    hs : Membership.mem fp.parts small
    ineq : LT.lt (HAdd.hAdd small.card 1) large.card
    w : V
    hw : Membership.mem large w
    v : V
    hv : Membership.mem small v
    cf : G.CliqueFree (HAdd.hAdd r 1)
    large_eq : Eq (fp.part w) large
    small_eq : Eq (fp.part v) small
    ha : G.Adj v w
    ⊢ LT.lt G.edgeFinset.card (HSub.hSub (HSub.hSub (HAdd.hAdd G.edgeFinset.card ( …
  -/
  have : #large ≤ Fintype.card V := by simpa using card_le_card large.subset_univ
  /-
    case right
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    large : Finset V
    hl : Membership.mem fp.parts large
    small : Finset V
    hs : Membership.mem fp.parts small
    ineq : LT.lt (HAdd.hAdd small.card 1) large.card
    w : V
    hw : Membership.mem large w
    v : V
    hv : Membership.mem small v
    cf : G.CliqueFree (HAdd.hAdd r 1)
    large_eq : Eq (fp.part w) large
    small_eq : Eq (fp.part v) small
    ha : G.Adj v w
    this : LE.le large.card (Fintype.card V)
    ⊢ LT.lt G.edgeFinset.card (HSub.hSub (HSub.hSub (HAdd.hAdd G.edgeFinset.card ( …
  -/
  omega
  /-
    🎉 no goals
  -/


lemma card_parts_le [DecidableEq V] : #h.finpartition.parts ≤ r := by
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    ⊢ LE.le h.finpartition.parts.card r
  -/
  by_contra! l
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    l : LT.lt r h.finpartition.parts.card
    ⊢ False
  -/
  obtain ⟨z, -, hz⟩ := h.finpartition.exists_subset_part_bijOn
  have ncf : ¬G.CliqueFree #z := by
    refine IsNClique.not_cliqueFree ⟨fun v hv w hw hn ↦ ?_, rfl⟩
    contrapose! hn
    exact hz.injOn hv hw (by rwa [← h.not_adj_iff_part_eq])
  /-
    case intro.intro
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    l : LT.lt r h.finpartition.parts.card
    z : Finset V
    hz : Set.BijOn h.finpartition.part ↑z ↑h.finpartition.parts
    ncf : Not (G.CliqueFree z.card)
    ⊢ False
  -/
  rw [Finset.card_eq_of_equiv hz.equiv] at ncf
  /-
    case intro.intro
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    l : LT.lt r h.finpartition.parts.card
    z : Finset V
    hz : Set.BijOn h.finpartition.part ↑z ↑h.finpartition.parts
    ncf : Not (G.CliqueFree h.finpartition.parts.card)
    ⊢ False
  -/
  exact absurd (h.1.mono (Nat.succ_le_of_lt l)) ncf
  /-
    🎉 no goals
  -/


/-- There are `min n r` parts in a graph on `n` vertices satisfying `G.IsTuranMaximal r`.
`min` handles the `n < r` case, when `G` is complete but still `r + 1`-cliquefree
for having insufficiently many vertices. -/
theorem card_parts [DecidableEq V] : #h.finpartition.parts = min (Fintype.card V) r := by
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    ⊢ Eq h.finpartition.parts.card (Min.min (Fintype.card V) r)
  -/
  set fp := h.finpartition
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    ⊢ Eq fp.parts.card (Min.min (Fintype.card V) r)
  -/
  apply le_antisymm (le_min fp.card_parts_le_card h.card_parts_le)
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    ⊢ LE.le (Min.min Finset.univ.card r) fp.parts.card
  -/
  by_contra! l
  /-
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    l : LT.lt fp.parts.card (Min.min Finset.univ.card r)
    ⊢ False
  -/
  rw [lt_min_iff] at l
  obtain ⟨x, -, y, -, hn, he⟩ :=
    exists_ne_map_eq_of_card_lt_of_maps_to l.1 fun a _ ↦ fp.part_mem (mem_univ a)
  /-
    case intro.intro.intro.intro.intro
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    l : And (LT.lt fp.parts.card Finset.univ.card) (LT.lt fp.parts.card r)
    x y : V
    hn : Ne x y
    he : Eq (fp.part x) (fp.part y)
    ⊢ False
  -/
  apply absurd h
  /-
    case intro.intro.intro.intro.intro
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    l : And (LT.lt fp.parts.card Finset.univ.card) (LT.lt fp.parts.card r)
    x y : V
    hn : Ne x y
    he : Eq (fp.part x) (fp.part y)
    ⊢ Not (G.IsTuranMaximal r)
  -/
  rw [IsTuranMaximal]; push_neg; rintro -
  have cf : G.CliqueFree r := by
    simp_rw [← cliqueFinset_eq_empty_iff, cliqueFinset, filter_eq_empty_iff, mem_univ,
      forall_true_left, isNClique_iff, and_comm, not_and, isClique_iff, Set.Pairwise]
    intro z zc; push_neg; simp_rw [h.not_adj_iff_part_eq]
    exact exists_ne_map_eq_of_card_lt_of_maps_to (zc.symm ▸ l.2) fun a _ ↦ fp.part_mem (mem_univ a)
  /-
    case intro.intro.intro.intro.intro
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    l : And (LT.lt fp.parts.card Finset.univ.card) (LT.lt fp.parts.card r)
    x y : V
    hn : Ne x y
    he : Eq (fp.part x) (fp.part y)
    cf : G.CliqueFree r
    ⊢ Exists fun H => Exists fun [DecidableRel H.Adj] => And (H.CliqueFree (HAdd.h …
  -/
  use G ⊔ edge x y, inferInstance, cf.sup_edge x y
  /-
    case right
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    l : And (LT.lt fp.parts.card Finset.univ.card) (LT.lt fp.parts.card r)
    x y : V
    hn : Ne x y
    he : Eq (fp.part x) (fp.part y)
    cf : G.CliqueFree r
    ⊢ LT.lt G.edgeFinset.card (Max.max G (SimpleGraph.edge x y)).edgeFinset.card
  -/
  convert Nat.lt.base #G.edgeFinset
  /-
    case h.e'_4
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    l : And (LT.lt fp.parts.card Finset.univ.card) (LT.lt fp.parts.card r)
    x y : V
    hn : Ne x y
    he : Eq (fp.part x) (fp.part y)
    cf : G.CliqueFree r
    ⊢ Eq (Max.max G (SimpleGraph.edge x y)).edgeFinset.card G.edgeFinset.card.succ
  -/
  convert G.card_edgeFinset_sup_edge _ hn
  /-
    case h.e'_4
    V : Type u_1
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    r : Nat
    h : G.IsTuranMaximal r
    inst✝ : DecidableEq V
    fp : Finpartition Finset.univ := h.finpartition
    l : And (LT.lt fp.parts.card Finset.univ.card) (LT.lt fp.parts.card r)
    x y : V
    hn : Ne x y
    he : Eq (fp.part x) (fp.part y)
    cf : G.CliqueFree r
    ⊢ Not (G.Adj x y)
  -/
  rwa [h.not_adj_iff_part_eq]
  /-
    🎉 no goals
  -/


/-- **Turán's theorem**, forward direction.

Any `r + 1`-cliquefree Turán-maximal graph on `n` vertices is isomorphic to `turanGraph n r`. -/
theorem nonempty_iso_turanGraph :
    Nonempty (G ≃g turanGraph (Fintype.card V) r) := by
  classical
  obtain ⟨zm, zp⟩ := h.isEquipartition.exists_partPreservingEquiv
  use (Equiv.subtypeUnivEquiv mem_univ).symm.trans zm
  intro a b
  simp_rw [turanGraph, Equiv.trans_apply, Equiv.subtypeUnivEquiv_symm_apply]
  have := zp ⟨a, mem_univ a⟩ ⟨b, mem_univ b⟩
  rw [← h.not_adj_iff_part_eq] at this
  rw [← not_iff_not, not_ne_iff, this, card_parts]
  rcases le_or_lt r (Fintype.card V) with c | c
  · rw [min_eq_right c]; rfl
  · have lc : ∀ x, zm ⟨x, _⟩ < Fintype.card V := fun x ↦ (zm ⟨x, mem_univ x⟩).2
    rw [min_eq_left c.le, Nat.mod_eq_of_lt (lc a), Nat.mod_eq_of_lt (lc b),
      ← Nat.mod_eq_of_lt ((lc a).trans c), ← Nat.mod_eq_of_lt ((lc b).trans c)]; rfl


/-- **Turán's theorem**, reverse direction.

Any graph isomorphic to `turanGraph n r` is itself Turán-maximal if `0 < r`. -/
theorem isTuranMaximal_of_iso (f : G ≃g turanGraph n r) (hr : 0 < r) : G.IsTuranMaximal r := by
  /-
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    n r : Nat
    f : G.Iso (SimpleGraph.turanGraph n r)
    hr : LT.lt 0 r
    ⊢ G.IsTuranMaximal r
  -/
  obtain ⟨J, _, j⟩ := exists_isTuranMaximal (V := V) hr
  /-
    case intro.intro
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    n r : Nat
    f : G.Iso (SimpleGraph.turanGraph n r)
    hr : LT.lt 0 r
    J : SimpleGraph V
    w✝ : DecidableRel J.Adj
    j : J.IsTuranMaximal r
    ⊢ G.IsTuranMaximal r
  -/
  obtain ⟨g⟩ := j.nonempty_iso_turanGraph
  /-
    case intro.intro.intro
    V : Type u_1
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    n r : Nat
    f : G.Iso (SimpleGraph.turanGraph n r)
    hr : LT.lt 0 r
    J : SimpleGraph V
    w✝ : DecidableRel J.Adj
    j : J.IsTuranMaximal r
    g : J.Iso (SimpleGraph.turanGraph (Fintype.card V) r)
    ⊢ G.IsTuranMaximal r
  -/
  rw [f.card_eq, Fintype.card_fin] at g
  use (turanGraph_cliqueFree (n := n) hr).comap f,
    fun H _ cf ↦ (f.symm.comp g).card_edgeFinset_eq ▸ j.2 H cf


/-- Turán-maximality with `0 < r` transfers across graph isomorphisms. -/
theorem IsTuranMaximal.iso {W : Type*} [Fintype W] {H : SimpleGraph W}
    [DecidableRel H.Adj] (h : G.IsTuranMaximal r) (f : G ≃g H) (hr : 0 < r) : H.IsTuranMaximal r :=
  isTuranMaximal_of_iso (h.nonempty_iso_turanGraph.some.comp f.symm) hr


/-- For `0 < r`, `turanGraph n r` is Turán-maximal. -/
theorem isTuranMaximal_turanGraph (hr : 0 < r) : (turanGraph n r).IsTuranMaximal r :=
  isTuranMaximal_of_iso Iso.refl hr


/-- **Turán's theorem**. `turanGraph n r` is, up to isomorphism, the unique
`r + 1`-cliquefree Turán-maximal graph on `n` vertices. -/
theorem isTuranMaximal_iff_nonempty_iso_turanGraph (hr : 0 < r) :
    G.IsTuranMaximal r ↔ Nonempty (G ≃g turanGraph (Fintype.card V) r) :=
  ⟨fun h ↦ h.nonempty_iso_turanGraph, fun h ↦ isTuranMaximal_of_iso h.some hr⟩


