/--
The set of vertices that are connected to all other vertices.
-/
def universalVerts (G : SimpleGraph V) : Set V := {v : V | ∀ ⦃w⦄, v ≠ w → G.Adj w v}


lemma isClique_universalVerts (G : SimpleGraph V) : G.IsClique G.universalVerts :=
  fun _ _ _ hy hxy ↦ hy hxy.symm


/--
The subgraph of `G` with the universal vertices removed.
-/
@[simps!]
def deleteUniversalVerts (G : SimpleGraph V) : Subgraph G :=
  (⊤ : Subgraph G).deleteVerts G.universalVerts


lemma Subgraph.IsMatching.exists_of_universalVerts [Fintype V] {s : Set V}
    (h : Disjoint G.universalVerts s) (hc : s.ncard ≤ G.universalVerts.ncard) :
    ∃ t ⊆ G.universalVerts, ∃ (M : Subgraph G), M.verts = s ∪ t ∧ M.IsMatching := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Fintype V
    s : Set V
    h : Disjoint G.universalVerts s
    hc : LE.le s.ncard G.universalVerts.ncard
    ⊢ Exists fun t => And (HasSubset.Subset t G.universalVerts) (Exists fun M => A …
  -/
  obtain ⟨t, ht⟩ := Set.exists_subset_card_eq hc
  /-
    case intro
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Fintype V
    s : Set V
    h : Disjoint G.universalVerts s
    hc : LE.le s.ncard G.universalVerts.ncard
    t : Set V
    ht : And (HasSubset.Subset t G.universalVerts) (Eq t.ncard s.ncard)
    ⊢ Exists fun t => And (HasSubset.Subset t G.universalVerts) (Exists fun M => A …
  -/
  refine ⟨t, ht.1, ?_⟩
  obtain ⟨f⟩ : Nonempty (s ≃ t) := by
    rw [← Cardinal.eq, ← t.cast_ncard t.toFinite, ← s.cast_ncard s.toFinite, ht.2]
  /-
    case intro.intro
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Fintype V
    s : Set V
    h : Disjoint G.universalVerts s
    hc : LE.le s.ncard G.universalVerts.ncard
    t : Set V
    ht : And (HasSubset.Subset t G.universalVerts) (Eq t.ncard s.ncard)
    f : Equiv ↑s ↑t
    ⊢ Exists fun M => And (Eq M.verts (Union.union s t)) M.IsMatching
  -/
  letI hd := Set.disjoint_of_subset_left ht.1 h
  /-
    case intro.intro
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Fintype V
    s : Set V
    h : Disjoint G.universalVerts s
    hc : LE.le s.ncard G.universalVerts.ncard
    t : Set V
    ht : And (HasSubset.Subset t G.universalVerts) (Eq t.ncard s.ncard)
    f : Equiv ↑s ↑t
    hd : Disjoint t s := Set.disjoint_of_subset_left ht.left h
    ⊢ Exists fun M => And (Eq M.verts (Union.union s t)) M.IsMatching
  -/
  have hadj (v : s) : G.Adj v (f v) := ht.1 (f v).2 (hd.ne_of_mem (f v).2 v.2)
  /-
    case intro.intro
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Fintype V
    s : Set V
    h : Disjoint G.universalVerts s
    hc : LE.le s.ncard G.universalVerts.ncard
    t : Set V
    ht : And (HasSubset.Subset t G.universalVerts) (Eq t.ncard s.ncard)
    f : Equiv ↑s ↑t
    hd : Disjoint t s := Set.disjoint_of_subset_left ht.left h
    hadj : ∀ (v : ↑s), G.Adj ↑v ↑(f v)
    ⊢ Exists fun M => And (Eq M.verts (Union.union s t)) M.IsMatching
  -/
  exact Subgraph.IsMatching.exists_of_disjoint_sets_of_equiv hd.symm f hadj
  /-
    🎉 no goals
  -/


