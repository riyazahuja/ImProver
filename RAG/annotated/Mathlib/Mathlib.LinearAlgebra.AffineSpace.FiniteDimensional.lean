/-- The `vectorSpan` of a finite set is finite-dimensional. -/
theorem finiteDimensional_vectorSpan_of_finite {s : Set P} (h : Set.Finite s) :
    FiniteDimensional k (vectorSpan k s) :=
  .span_of_finite k <| h.vsub h


/-- The `vectorSpan` of a family indexed by a `Fintype` is
finite-dimensional. -/
instance finiteDimensional_vectorSpan_range [Finite ι] (p : ι → P) :
    FiniteDimensional k (vectorSpan k (Set.range p)) :=
  finiteDimensional_vectorSpan_of_finite k (Set.finite_range _)


/-- The `vectorSpan` of a subset of a family indexed by a `Fintype`
is finite-dimensional. -/
instance finiteDimensional_vectorSpan_image_of_finite [Finite ι] (p : ι → P) (s : Set ι) :
    FiniteDimensional k (vectorSpan k (p '' s)) :=
  finiteDimensional_vectorSpan_of_finite k (Set.toFinite _)


/-- The direction of the affine span of a finite set is
finite-dimensional. -/
theorem finiteDimensional_direction_affineSpan_of_finite {s : Set P} (h : Set.Finite s) :
    FiniteDimensional k (affineSpan k s).direction :=
  (direction_affineSpan k s).symm ▸ finiteDimensional_vectorSpan_of_finite k h


/-- The direction of the affine span of a family indexed by a
`Fintype` is finite-dimensional. -/
instance finiteDimensional_direction_affineSpan_range [Finite ι] (p : ι → P) :
    FiniteDimensional k (affineSpan k (Set.range p)).direction :=
  finiteDimensional_direction_affineSpan_of_finite k (Set.finite_range _)


/-- The direction of the affine span of a subset of a family indexed
by a `Fintype` is finite-dimensional. -/
instance finiteDimensional_direction_affineSpan_image_of_finite [Finite ι] (p : ι → P) (s : Set ι) :
    FiniteDimensional k (affineSpan k (p '' s)).direction :=
  finiteDimensional_direction_affineSpan_of_finite k (Set.toFinite _)


/-- An affine-independent family of points in a finite-dimensional affine space is finite. -/
theorem finite_of_fin_dim_affineIndependent [FiniteDimensional k V] {p : ι → P}
    (hi : AffineIndependent k p) : Finite ι := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : FiniteDimensional k V
    p : ι → P
    hi : AffineIndependent k p
    ⊢ Finite ι
  -/
  nontriviality ι; inhabit ι
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : FiniteDimensional k V
    p : ι → P
    hi : AffineIndependent k p
    a✝ : Nontrivial ι
    inhabited_h : Inhabited ι
    ⊢ Finite ι
  -/
  rw [affineIndependent_iff_linearIndependent_vsub k p default] at hi
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : FiniteDimensional k V
    p : ι → P
    a✝ : Nontrivial ι
    inhabited_h : Inhabited ι
    hi : LinearIndependent k fun i => VSub.vsub (p ↑i) (p Inhabited.default)
    ⊢ Finite ι
  -/
  letI : IsNoetherian k V := IsNoetherian.iff_fg.2 inferInstance
  exact
    (Set.finite_singleton default).finite_of_compl (Set.finite_coe_iff.1 hi.finite_of_isNoetherian)


/-- An affine-independent subset of a finite-dimensional affine space is finite. -/
theorem finite_set_of_fin_dim_affineIndependent [FiniteDimensional k V] {s : Set ι} {f : s → P}
    (hi : AffineIndependent k f) : s.Finite :=
  @Set.toFinite _ s (finite_of_fin_dim_affineIndependent k hi)


/-- The `vectorSpan` of a finite subset of an affinely independent
family has dimension one less than its cardinality. -/
theorem AffineIndependent.finrank_vectorSpan_image_finset [DecidableEq P]
    {p : ι → P} (hi : AffineIndependent k p) {s : Finset ι} {n : ℕ} (hc : #s = n + 1) :
    finrank k (vectorSpan k (s.image p : Set P)) = n := by
  classical
  have hi' := hi.range.mono (Set.image_subset_range p ↑s)
  have hc' : #(s.image p) = n + 1 := by rwa [s.card_image_of_injective hi.injective]
  have hn : (s.image p).Nonempty := by simp [hc', ← Finset.card_pos]
  rcases hn with ⟨p₁, hp₁⟩
  have hp₁' : p₁ ∈ p '' s := by simpa using hp₁
  rw [affineIndependent_set_iff_linearIndependent_vsub k hp₁', ← Finset.coe_singleton,
    ← Finset.coe_image, ← Finset.coe_sdiff, Finset.sdiff_singleton_eq_erase, ← Finset.coe_image]
    at hi'
  have hc : #(((s.image p).erase p₁).image (· -ᵥ p₁)) = n := by
    rw [Finset.card_image_of_injective _ (vsub_left_injective _), Finset.card_erase_of_mem hp₁]
    exact Nat.pred_eq_of_eq_succ hc'
  rwa [vectorSpan_eq_span_vsub_finset_right_ne k hp₁, finrank_span_finset_eq_card, hc]


/-- The `vectorSpan` of a finite affinely independent family has
dimension one less than its cardinality. -/
theorem AffineIndependent.finrank_vectorSpan [Fintype ι] {p : ι → P} (hi : AffineIndependent k p)
    {n : ℕ} (hc : Fintype.card ι = n + 1) : finrank k (vectorSpan k (Set.range p)) = n := by
  classical
  rw [← Finset.card_univ] at hc
  rw [← Set.image_univ, ← Finset.coe_univ, ← Finset.coe_image]
  exact hi.finrank_vectorSpan_image_finset hc


/-- The `vectorSpan` of a finite affinely independent family has dimension one less than its
cardinality. -/
lemma AffineIndependent.finrank_vectorSpan_add_one [Fintype ι] [Nonempty ι] {p : ι → P}
    (hi : AffineIndependent k p) : finrank k (vectorSpan k (Set.range p)) + 1 = Fintype.card ι := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁵ : DivisionRing k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    inst✝² : AddTorsor V P
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : ι → P
    hi : AffineIndependent k p
    ⊢ Eq (HAdd.hAdd (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan …
  -/
  rw [hi.finrank_vectorSpan (tsub_add_cancel_of_le _).symm, tsub_add_cancel_of_le] <;>
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁵ : DivisionRing k
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module k V
      inst✝² : AddTorsor V P
      inst✝¹ : Fintype ι
      inst✝ : Nonempty ι
      p : ι → P
      hi : AffineIndependent k p
      ⊢ LE.le 1 (Fintype.card ι)
    -/
    /-
      🎉 no goals
    -/
    exact Fintype.card_pos
    /-
      🎉 no goals
    -/


/-- The `vectorSpan` of a finite affinely independent family whose
cardinality is one more than that of the finite-dimensional space is
`⊤`. -/
theorem AffineIndependent.vectorSpan_eq_top_of_card_eq_finrank_add_one [FiniteDimensional k V]
    [Fintype ι] {p : ι → P} (hi : AffineIndependent k p) (hc : Fintype.card ι = finrank k V + 1) :
    vectorSpan k (Set.range p) = ⊤ :=
  Submodule.eq_top_of_finrank_eq <| hi.finrank_vectorSpan hc


/-- The `vectorSpan` of `n + 1` points in an indexed family has
dimension at most `n`. -/
theorem finrank_vectorSpan_image_finset_le [DecidableEq P] (p : ι → P) (s : Finset ι) {n : ℕ}
    (hc : #s = n + 1) : finrank k (vectorSpan k (s.image p : Set P)) ≤ n := by
  classical
  have hn : (s.image p).Nonempty := by
    rw [Finset.image_nonempty, ← Finset.card_pos, hc]
    apply Nat.succ_pos
  rcases hn with ⟨p₁, hp₁⟩
  rw [vectorSpan_eq_span_vsub_finset_right_ne k hp₁]
  refine le_trans (finrank_span_finset_le_card (((s.image p).erase p₁).image fun p => p -ᵥ p₁)) ?_
  rw [Finset.card_image_of_injective _ (vsub_left_injective p₁), Finset.card_erase_of_mem hp₁,
    tsub_le_iff_right, ← hc]
  apply Finset.card_image_le


/-- The `vectorSpan` of an indexed family of `n + 1` points has
dimension at most `n`. -/
theorem finrank_vectorSpan_range_le [Fintype ι] (p : ι → P) {n : ℕ} (hc : Fintype.card ι = n + 1) :
    finrank k (vectorSpan k (Set.range p)) ≤ n := by
  classical
  rw [← Set.image_univ, ← Finset.coe_univ, ← Finset.coe_image]
  rw [← Finset.card_univ] at hc
  exact finrank_vectorSpan_image_finset_le _ _ _ hc


/-- The `vectorSpan` of an indexed family of `n + 1` points has dimension at most `n`. -/
lemma finrank_vectorSpan_range_add_one_le [Fintype ι] [Nonempty ι] (p : ι → P) :
    finrank k (vectorSpan k (Set.range p)) + 1 ≤ Fintype.card ι :=
  (le_tsub_iff_right <| Nat.succ_le_iff.2 Fintype.card_pos).1 <| finrank_vectorSpan_range_le _ _
    (tsub_add_cancel_of_le <| Nat.succ_le_iff.2 Fintype.card_pos).symm


/-- `n + 1` points are affinely independent if and only if their
`vectorSpan` has dimension `n`. -/
theorem affineIndependent_iff_finrank_vectorSpan_eq [Fintype ι] (p : ι → P) {n : ℕ}
    (hc : Fintype.card ι = n + 1) :
    AffineIndependent k p ↔ finrank k (vectorSpan k (Set.range p)) = n := by
  classical
  have hn : Nonempty ι := by simp [← Fintype.card_pos_iff, hc]
  cases' hn with i₁
  rw [affineIndependent_iff_linearIndependent_vsub _ _ i₁,
    linearIndependent_iff_card_eq_finrank_span, eq_comm,
    vectorSpan_range_eq_span_range_vsub_right_ne k p i₁, Set.finrank]
  rw [← Finset.card_univ] at hc
  rw [Fintype.subtype_card]
  simp [Finset.filter_ne', Finset.card_erase_of_mem, hc]


/-- `n + 1` points are affinely independent if and only if their
`vectorSpan` has dimension at least `n`. -/
theorem affineIndependent_iff_le_finrank_vectorSpan [Fintype ι] (p : ι → P) {n : ℕ}
    (hc : Fintype.card ι = n + 1) :
    AffineIndependent k p ↔ n ≤ finrank k (vectorSpan k (Set.range p)) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : Fintype ι
    p : ι → P
    n : Nat
    hc : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ⊢ Iff (AffineIndependent k p) (LE.le n (Module.finrank k (Subtype fun x => Mem …
  -/
  rw [affineIndependent_iff_finrank_vectorSpan_eq k p hc]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : Fintype ι
    p : ι → P
    n : Nat
    hc : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ⊢ Iff (Eq (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Se …
  -/
  constructor
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁴ : DivisionRing k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Fintype ι
      p : ι → P
      n : Nat
      hc : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      ⊢ Eq (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Set.ran …
    -/
  · rintro rfl
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁴ : DivisionRing k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Fintype ι
      p : ι → P
      hc : Eq (Fintype.card ι) (HAdd.hAdd (Module.finrank k (Subtype fun x => Member …
      ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Set. …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁴ : DivisionRing k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Fintype ι
      p : ι → P
      n : Nat
      hc : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      ⊢ LE.le n (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Se …
    -/
  · exact fun hle => le_antisymm (finrank_vectorSpan_range_le k p hc) hle
    /-
      🎉 no goals
    -/


/-- `n + 2` points are affinely independent if and only if their
`vectorSpan` does not have dimension at most `n`. -/
theorem affineIndependent_iff_not_finrank_vectorSpan_le [Fintype ι] (p : ι → P) {n : ℕ}
    (hc : Fintype.card ι = n + 2) :
    AffineIndependent k p ↔ ¬finrank k (vectorSpan k (Set.range p)) ≤ n := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : Fintype ι
    p : ι → P
    n : Nat
    hc : Eq (Fintype.card ι) (HAdd.hAdd n 2)
    ⊢ Iff (AffineIndependent k p) (Not (LE.le (Module.finrank k (Subtype fun x =>  …
  -/
  rw [affineIndependent_iff_le_finrank_vectorSpan k p hc, ← Nat.lt_iff_add_one_le, lt_iff_not_ge]
  /-
    🎉 no goals
  -/


/-- `n + 2` points have a `vectorSpan` with dimension at most `n` if
and only if they are not affinely independent. -/
theorem finrank_vectorSpan_le_iff_not_affineIndependent [Fintype ι] (p : ι → P) {n : ℕ}
    (hc : Fintype.card ι = n + 2) :
    finrank k (vectorSpan k (Set.range p)) ≤ n ↔ ¬AffineIndependent k p :=
  (not_iff_comm.1 (affineIndependent_iff_not_finrank_vectorSpan_le k p hc).symm).symm


lemma AffineIndependent.card_le_finrank_succ [Fintype ι] {p : ι → P} (hp : AffineIndependent k p) :
    Fintype.card ι ≤ Module.finrank k (vectorSpan k (Set.range p)) + 1 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : Fintype ι
    p : ι → P
    hp : AffineIndependent k p
    ⊢ LE.le (Fintype.card ι) (HAdd.hAdd (Module.finrank k (Subtype fun x => Member …
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁴ : DivisionRing k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Fintype ι
      p : ι → P
      hp : AffineIndependent k p
      h✝ : IsEmpty ι
      ⊢ LE.le (Fintype.card ι) (HAdd.hAdd (Module.finrank k (Subtype fun x => Member …
    -/
  · simp [Fintype.card_eq_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : Fintype ι
    p : ι → P
    hp : AffineIndependent k p
    h✝ : Nonempty ι
    ⊢ LE.le (Fintype.card ι) (HAdd.hAdd (Module.finrank k (Subtype fun x => Member …
  -/
  rw [← tsub_le_iff_right]
  exact (affineIndependent_iff_le_finrank_vectorSpan _ _
    (tsub_add_cancel_of_le <| Nat.one_le_iff_ne_zero.2 Fintype.card_ne_zero).symm).1 hp


open Finset in
/-- If an affine independent finset is contained in the affine span of another finset, then its
cardinality is at most the cardinality of that finset. -/
lemma AffineIndependent.card_le_card_of_subset_affineSpan {s t : Finset V}
    (hs : AffineIndependent k ((↑) : s → V)) (hst : (s : Set V) ⊆ affineSpan k (t : Set V)) :
    #s ≤ #t := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : HasSubset.Subset ↑s ↑(affineSpan k ↑t)
    ⊢ LE.le s.card t.card
  -/
  obtain rfl | hs' := s.eq_empty_or_nonempty
    /-
      case inl
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      t : Finset V
      hs : AffineIndependent k Subtype.val
      hst : HasSubset.Subset ↑EmptyCollection.emptyCollection ↑(affineSpan k ↑t)
      ⊢ LE.le EmptyCollection.emptyCollection.card t.card
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : HasSubset.Subset ↑s ↑(affineSpan k ↑t)
    hs' : s.Nonempty
    ⊢ LE.le s.card t.card
  -/
  obtain rfl | ht' := t.eq_empty_or_nonempty
    /-
      case inr.inl
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      s : Finset V
      hs : AffineIndependent k Subtype.val
      hs' : s.Nonempty
      hst : HasSubset.Subset ↑s ↑(affineSpan k ↑EmptyCollection.emptyCollection)
      ⊢ LE.le s.card EmptyCollection.emptyCollection.card
    -/
  · simpa [Set.subset_empty_iff] using hst
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : HasSubset.Subset ↑s ↑(affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    ⊢ LE.le s.card t.card
  -/
  have := hs'.to_subtype
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : HasSubset.Subset ↑s ↑(affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    this : Nonempty (Subtype fun x => Membership.mem s x)
    ⊢ LE.le s.card t.card
  -/
  have := ht'.to_set.to_subtype
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : HasSubset.Subset ↑s ↑(affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    this✝ : Nonempty (Subtype fun x => Membership.mem s x)
    this : Nonempty ↑↑t
    ⊢ LE.le s.card t.card
  -/
  have direction_le := AffineSubspace.direction_le (affineSpan_mono k hst)
  rw [AffineSubspace.affineSpan_coe, direction_affineSpan, direction_affineSpan,
    ← @Subtype.range_coe _ (s : Set V), ← @Subtype.range_coe _ (t : Set V)] at direction_le
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : HasSubset.Subset ↑s ↑(affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    this✝ : Nonempty (Subtype fun x => Membership.mem s x)
    this : Nonempty ↑↑t
    direction_le : LE.le (vectorSpan k (Set.range Subtype.val)) (vectorSpan k (Set …
    ⊢ LE.le s.card t.card
  -/
  have finrank_le := add_le_add_right (Submodule.finrank_mono direction_le) 1
  -- We use `erw` to elide the difference between `↥s` and `↥(s : Set V)}`
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : HasSubset.Subset ↑s ↑(affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    this✝ : Nonempty (Subtype fun x => Membership.mem s x)
    this : Nonempty ↑↑t
    direction_le : LE.le (vectorSpan k (Set.range Subtype.val)) (vectorSpan k (Set …
    finrank_le : LE.le (HAdd.hAdd (Module.finrank k (Subtype fun x => Membership.m …
    ⊢ LE.le s.card t.card
  -/
  erw [hs.finrank_vectorSpan_add_one] at finrank_le
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : HasSubset.Subset ↑s ↑(affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    this✝ : Nonempty (Subtype fun x => Membership.mem s x)
    this : Nonempty ↑↑t
    direction_le : LE.le (vectorSpan k (Set.range Subtype.val)) (vectorSpan k (Set …
    finrank_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) (HAdd. …
    ⊢ LE.le s.card t.card
  -/
  simpa using finrank_le.trans <| finrank_vectorSpan_range_add_one_le _ _
  /-
    🎉 no goals
  -/


open Finset in
/-- If the affine span of an affine independent finset is strictly contained in the affine span of
another finset, then its cardinality is strictly less than the cardinality of that finset. -/
lemma AffineIndependent.card_lt_card_of_affineSpan_lt_affineSpan {s t : Finset V}
    (hs : AffineIndependent k ((↑) : s → V))
    (hst : affineSpan k (s : Set V) < affineSpan k (t : Set V)) : #s < #t := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : LT.lt (affineSpan k ↑s) (affineSpan k ↑t)
    ⊢ LT.lt s.card t.card
  -/
  obtain rfl | hs' := s.eq_empty_or_nonempty
    /-
      case inl
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      t : Finset V
      hs : AffineIndependent k Subtype.val
      hst : LT.lt (affineSpan k ↑EmptyCollection.emptyCollection) (affineSpan k ↑t)
      ⊢ LT.lt EmptyCollection.emptyCollection.card t.card
    -/
  · simpa [card_pos] using hst
    /-
      🎉 no goals
    -/
  /-
    case inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : LT.lt (affineSpan k ↑s) (affineSpan k ↑t)
    hs' : s.Nonempty
    ⊢ LT.lt s.card t.card
  -/
  obtain rfl | ht' := t.eq_empty_or_nonempty
    /-
      case inr.inl
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      s : Finset V
      hs : AffineIndependent k Subtype.val
      hs' : s.Nonempty
      hst : LT.lt (affineSpan k ↑s) (affineSpan k ↑EmptyCollection.emptyCollection)
      ⊢ LT.lt s.card EmptyCollection.emptyCollection.card
    -/
  · simp [Set.subset_empty_iff] at hst
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : LT.lt (affineSpan k ↑s) (affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    ⊢ LT.lt s.card t.card
  -/
  have := hs'.to_subtype
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : LT.lt (affineSpan k ↑s) (affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    this : Nonempty (Subtype fun x => Membership.mem s x)
    ⊢ LT.lt s.card t.card
  -/
  have := ht'.to_set.to_subtype
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : LT.lt (affineSpan k ↑s) (affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    this✝ : Nonempty (Subtype fun x => Membership.mem s x)
    this : Nonempty ↑↑t
    ⊢ LT.lt s.card t.card
  -/
  have dir_lt := AffineSubspace.direction_lt_of_nonempty (k := k) hst <| hs'.to_set.affineSpan k
  rw [direction_affineSpan, direction_affineSpan,
    ← @Subtype.range_coe _ (s : Set V), ← @Subtype.range_coe _ (t : Set V)] at dir_lt
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : LT.lt (affineSpan k ↑s) (affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    this✝ : Nonempty (Subtype fun x => Membership.mem s x)
    this : Nonempty ↑↑t
    dir_lt : LT.lt (vectorSpan k (Set.range Subtype.val)) (vectorSpan k (Set.range …
    ⊢ LT.lt s.card t.card
  -/
  have finrank_lt := add_lt_add_right (Submodule.finrank_lt_finrank_of_lt dir_lt) 1
  -- We use `erw` to elide the difference between `↥s` and `↥(s : Set V)}`
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : LT.lt (affineSpan k ↑s) (affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    this✝ : Nonempty (Subtype fun x => Membership.mem s x)
    this : Nonempty ↑↑t
    dir_lt : LT.lt (vectorSpan k (Set.range Subtype.val)) (vectorSpan k (Set.range …
    finrank_lt : LT.lt (HAdd.hAdd (Module.finrank k (Subtype fun x => Membership.m …
    ⊢ LT.lt s.card t.card
  -/
  erw [hs.finrank_vectorSpan_add_one] at finrank_lt
  /-
    case inr.inr
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s t : Finset V
    hs : AffineIndependent k Subtype.val
    hst : LT.lt (affineSpan k ↑s) (affineSpan k ↑t)
    hs' : s.Nonempty
    ht' : t.Nonempty
    this✝ : Nonempty (Subtype fun x => Membership.mem s x)
    this : Nonempty ↑↑t
    dir_lt : LT.lt (vectorSpan k (Set.range Subtype.val)) (vectorSpan k (Set.range …
    finrank_lt : LT.lt (Fintype.card (Subtype fun x => Membership.mem s x)) (HAdd. …
    ⊢ LT.lt s.card t.card
  -/
  simpa using finrank_lt.trans_le <| finrank_vectorSpan_range_add_one_le _ _
  /-
    🎉 no goals
  -/


/-- If the `vectorSpan` of a finite subset of an affinely independent
family lies in a submodule with dimension one less than its
cardinality, it equals that submodule. -/
theorem AffineIndependent.vectorSpan_image_finset_eq_of_le_of_card_eq_finrank_add_one
    [DecidableEq P] {p : ι → P}
    (hi : AffineIndependent k p) {s : Finset ι} {sm : Submodule k V} [FiniteDimensional k sm]
    (hle : vectorSpan k (s.image p : Set P) ≤ sm) (hc : #s = finrank k sm + 1) :
    vectorSpan k (s.image p : Set P) = sm :=
  Submodule.eq_of_le_of_finrank_eq hle <| hi.finrank_vectorSpan_image_finset hc


/-- If the `vectorSpan` of a finite affinely independent
family lies in a submodule with dimension one less than its
cardinality, it equals that submodule. -/
theorem AffineIndependent.vectorSpan_eq_of_le_of_card_eq_finrank_add_one [Fintype ι] {p : ι → P}
    (hi : AffineIndependent k p) {sm : Submodule k V} [FiniteDimensional k sm]
    (hle : vectorSpan k (Set.range p) ≤ sm) (hc : Fintype.card ι = finrank k sm + 1) :
    vectorSpan k (Set.range p) = sm :=
  Submodule.eq_of_le_of_finrank_eq hle <| hi.finrank_vectorSpan hc


/-- If the `affineSpan` of a finite subset of an affinely independent
family lies in an affine subspace whose direction has dimension one
less than its cardinality, it equals that subspace. -/
theorem AffineIndependent.affineSpan_image_finset_eq_of_le_of_card_eq_finrank_add_one
    [DecidableEq P] {p : ι → P}
    (hi : AffineIndependent k p) {s : Finset ι} {sp : AffineSubspace k P}
    [FiniteDimensional k sp.direction] (hle : affineSpan k (s.image p : Set P) ≤ sp)
    (hc : #s = finrank k sp.direction + 1) : affineSpan k (s.image p : Set P) = sp := by
  have hn : s.Nonempty := by
    rw [← Finset.card_pos, hc]
    apply Nat.succ_pos
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁵ : DivisionRing k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    inst✝² : AddTorsor V P
    inst✝¹ : DecidableEq P
    p : ι → P
    hi : AffineIndependent k p
    s : Finset ι
    sp : AffineSubspace k P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem sp.direction x)
    hle : LE.le (affineSpan k ↑(Finset.image p s)) sp
    hc : Eq s.card (HAdd.hAdd (Module.finrank k (Subtype fun x => Membership.mem s …
    hn : s.Nonempty
    ⊢ Eq (affineSpan k ↑(Finset.image p s)) sp
  -/
  refine eq_of_direction_eq_of_nonempty_of_le ?_ ((hn.image p).to_set.affineSpan k) hle
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁵ : DivisionRing k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    inst✝² : AddTorsor V P
    inst✝¹ : DecidableEq P
    p : ι → P
    hi : AffineIndependent k p
    s : Finset ι
    sp : AffineSubspace k P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem sp.direction x)
    hle : LE.le (affineSpan k ↑(Finset.image p s)) sp
    hc : Eq s.card (HAdd.hAdd (Module.finrank k (Subtype fun x => Membership.mem s …
    hn : s.Nonempty
    ⊢ Eq (affineSpan k ↑(Finset.image p s)).direction sp.direction
  -/
  have hd := direction_le hle
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁵ : DivisionRing k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    inst✝² : AddTorsor V P
    inst✝¹ : DecidableEq P
    p : ι → P
    hi : AffineIndependent k p
    s : Finset ι
    sp : AffineSubspace k P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem sp.direction x)
    hle : LE.le (affineSpan k ↑(Finset.image p s)) sp
    hc : Eq s.card (HAdd.hAdd (Module.finrank k (Subtype fun x => Membership.mem s …
    hn : s.Nonempty
    hd : LE.le (affineSpan k ↑(Finset.image p s)).direction sp.direction
    ⊢ Eq (affineSpan k ↑(Finset.image p s)).direction sp.direction
  -/
  rw [direction_affineSpan] at hd ⊢
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁵ : DivisionRing k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    inst✝² : AddTorsor V P
    inst✝¹ : DecidableEq P
    p : ι → P
    hi : AffineIndependent k p
    s : Finset ι
    sp : AffineSubspace k P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem sp.direction x)
    hle : LE.le (affineSpan k ↑(Finset.image p s)) sp
    hc : Eq s.card (HAdd.hAdd (Module.finrank k (Subtype fun x => Membership.mem s …
    hn : s.Nonempty
    hd : LE.le (vectorSpan k ↑(Finset.image p s)) sp.direction
    ⊢ Eq (vectorSpan k ↑(Finset.image p s)) sp.direction
  -/
  exact hi.vectorSpan_image_finset_eq_of_le_of_card_eq_finrank_add_one hd hc
  /-
    🎉 no goals
  -/


/-- If the `affineSpan` of a finite affinely independent family lies
in an affine subspace whose direction has dimension one less than its
cardinality, it equals that subspace. -/
theorem AffineIndependent.affineSpan_eq_of_le_of_card_eq_finrank_add_one [Fintype ι] {p : ι → P}
    (hi : AffineIndependent k p) {sp : AffineSubspace k P} [FiniteDimensional k sp.direction]
    (hle : affineSpan k (Set.range p) ≤ sp) (hc : Fintype.card ι = finrank k sp.direction + 1) :
    affineSpan k (Set.range p) = sp := by
  classical
  rw [← Finset.card_univ] at hc
  rw [← Set.image_univ, ← Finset.coe_univ, ← Finset.coe_image] at hle ⊢
  exact hi.affineSpan_image_finset_eq_of_le_of_card_eq_finrank_add_one hle hc


/-- The `affineSpan` of a finite affinely independent family is `⊤` iff the
family's cardinality is one more than that of the finite-dimensional space. -/
theorem AffineIndependent.affineSpan_eq_top_iff_card_eq_finrank_add_one [FiniteDimensional k V]
    [Fintype ι] {p : ι → P} (hi : AffineIndependent k p) :
    affineSpan k (Set.range p) = ⊤ ↔ Fintype.card ι = finrank k V + 1 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁵ : DivisionRing k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    inst✝² : AddTorsor V P
    inst✝¹ : FiniteDimensional k V
    inst✝ : Fintype ι
    p : ι → P
    hi : AffineIndependent k p
    ⊢ Iff (Eq (affineSpan k (Set.range p)) Top.top) (Eq (Fintype.card ι) (HAdd.hAd …
  -/
  constructor
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁵ : DivisionRing k
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module k V
      inst✝² : AddTorsor V P
      inst✝¹ : FiniteDimensional k V
      inst✝ : Fintype ι
      p : ι → P
      hi : AffineIndependent k p
      ⊢ Eq (affineSpan k (Set.range p)) Top.top → Eq (Fintype.card ι) (HAdd.hAdd (Mo …
    -/
  · intro h_tot
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁵ : DivisionRing k
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module k V
      inst✝² : AddTorsor V P
      inst✝¹ : FiniteDimensional k V
      inst✝ : Fintype ι
      p : ι → P
      hi : AffineIndependent k p
      h_tot : Eq (affineSpan k (Set.range p)) Top.top
      ⊢ Eq (Fintype.card ι) (HAdd.hAdd (Module.finrank k V) 1)
    -/
    let n := Fintype.card ι - 1
    have hn : Fintype.card ι = n + 1 :=
      (Nat.succ_pred_eq_of_pos (card_pos_of_affineSpan_eq_top k V P h_tot)).symm
    rw [hn, ← finrank_top, ← (vectorSpan_eq_top_of_affineSpan_eq_top k V P) h_tot,
      ← hi.finrank_vectorSpan hn]
    /-
      case mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁵ : DivisionRing k
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module k V
      inst✝² : AddTorsor V P
      inst✝¹ : FiniteDimensional k V
      inst✝ : Fintype ι
      p : ι → P
      hi : AffineIndependent k p
      ⊢ Eq (Fintype.card ι) (HAdd.hAdd (Module.finrank k V) 1) → Eq (affineSpan k (S …
    -/
  · intro hc
    /-
      case mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁵ : DivisionRing k
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module k V
      inst✝² : AddTorsor V P
      inst✝¹ : FiniteDimensional k V
      inst✝ : Fintype ι
      p : ι → P
      hi : AffineIndependent k p
      hc : Eq (Fintype.card ι) (HAdd.hAdd (Module.finrank k V) 1)
      ⊢ Eq (affineSpan k (Set.range p)) Top.top
    -/
    rw [← finrank_top, ← direction_top k V P] at hc
    /-
      case mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁵ : DivisionRing k
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module k V
      inst✝² : AddTorsor V P
      inst✝¹ : FiniteDimensional k V
      inst✝ : Fintype ι
      p : ι → P
      hi : AffineIndependent k p
      hc : Eq (Fintype.card ι) (HAdd.hAdd (Module.finrank k (Subtype fun x => Member …
      ⊢ Eq (affineSpan k (Set.range p)) Top.top
    -/
    exact hi.affineSpan_eq_of_le_of_card_eq_finrank_add_one le_top hc
    /-
      🎉 no goals
    -/


theorem Affine.Simplex.span_eq_top [FiniteDimensional k V] {n : ℕ} (T : Affine.Simplex k V n)
    (hrank : finrank k V = n) : affineSpan k (Set.range T.points) = ⊤ := by
  rw [AffineIndependent.affineSpan_eq_top_iff_card_eq_finrank_add_one T.independent,
    Fintype.card_fin, hrank]


/-- The `vectorSpan` of adding a point to a finite-dimensional subspace is finite-dimensional. -/
instance finiteDimensional_vectorSpan_insert (s : AffineSubspace k P)
    [FiniteDimensional k s.direction] (p : P) :
    FiniteDimensional k (vectorSpan k (insert p (s : Set P))) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    s : AffineSubspace k P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
    p : P
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k (Insert.i …
  -/
  rw [← direction_affineSpan, ← affineSpan_insert_affineSpan]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    s : AffineSubspace k P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
    p : P
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem (affineSpan k (Insert.i …
  -/
  rcases (s : Set P).eq_empty_or_nonempty with (hs | ⟨p₀, hp₀⟩)
    /-
      case inl
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁴ : DivisionRing k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p : P
      hs : Eq (↑s) EmptyCollection.emptyCollection
      ⊢ FiniteDimensional k (Subtype fun x => Membership.mem (affineSpan k (Insert.i …
    -/
  · rw [coe_eq_bot_iff] at hs
    /-
      case inl
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁴ : DivisionRing k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p : P
      hs : Eq s Bot.bot
      ⊢ FiniteDimensional k (Subtype fun x => Membership.mem (affineSpan k (Insert.i …
    -/
    rw [hs, bot_coe, span_empty, bot_coe, direction_affineSpan]
    /-
      case inl
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁴ : DivisionRing k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p : P
      hs : Eq s Bot.bot
      ⊢ FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k (Insert.i …
    -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
    convert finiteDimensional_bot k V <;> simp
                                          /-
                                            🎉 no goals
                                          -/
    /-
      case inr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁴ : DivisionRing k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      ⊢ FiniteDimensional k (Subtype fun x => Membership.mem (affineSpan k (Insert.i …
    -/
  · rw [affineSpan_coe, direction_affineSpan_insert hp₀]
    /-
      case inr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      ι : Type u_4
      inst✝⁴ : DivisionRing k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      ⊢ FiniteDimensional k (Subtype fun x => Membership.mem (Max.max (Submodule.spa …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- The direction of the affine span of adding a point to a finite-dimensional subspace is
finite-dimensional. -/
instance finiteDimensional_direction_affineSpan_insert (s : AffineSubspace k P)
    [FiniteDimensional k s.direction] (p : P) :
    FiniteDimensional k (affineSpan k (insert p (s : Set P))).direction :=
  (direction_affineSpan k (insert p (s : Set P))).symm ▸ finiteDimensional_vectorSpan_insert s p


/-- The `vectorSpan` of adding a point to a set with a finite-dimensional `vectorSpan` is
finite-dimensional. -/
instance finiteDimensional_vectorSpan_insert_set (s : Set P) [FiniteDimensional k (vectorSpan k s)]
    (p : P) : FiniteDimensional k (vectorSpan k (insert p s)) := by
  haveI : FiniteDimensional k (affineSpan k s).direction :=
    (direction_affineSpan k s).symm ▸ inferInstance
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    s : Set P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
    p : P
    this : FiniteDimensional k (Subtype fun x => Membership.mem (affineSpan k s).d …
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k (Insert.i …
  -/
  rw [← direction_affineSpan, ← affineSpan_insert_affineSpan, direction_affineSpan]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    ι : Type u_4
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    s : Set P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
    p : P
    this : FiniteDimensional k (Subtype fun x => Membership.mem (affineSpan k s).d …
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k (Insert.i …
  -/
  exact finiteDimensional_vectorSpan_insert (affineSpan k s) p
  /-
    🎉 no goals
  -/


/-- A set of points is collinear if their `vectorSpan` has dimension
at most `1`. -/
def Collinear (s : Set P) : Prop :=
  Module.rank k (vectorSpan k s) ≤ 1


/-- The definition of `Collinear`. -/
theorem collinear_iff_rank_le_one (s : Set P) :
    Collinear k s ↔ Module.rank k (vectorSpan k s) ≤ 1 := Iff.rfl


/-- A set of points, whose `vectorSpan` is finite-dimensional, is
collinear if and only if their `vectorSpan` has dimension at most
`1`. -/
theorem collinear_iff_finrank_le_one {s : Set P} [FiniteDimensional k (vectorSpan k s)] :
    Collinear k s ↔ finrank k (vectorSpan k s) ≤ 1 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    s : Set P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
    ⊢ Iff (Collinear k s) (LE.le (Module.finrank k (Subtype fun x => Membership.me …
  -/
  have h := collinear_iff_rank_le_one k s
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    s : Set P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
    h : Iff (Collinear k s) (LE.le (Module.rank k (Subtype fun x => Membership.mem …
    ⊢ Iff (Collinear k s) (LE.le (Module.finrank k (Subtype fun x => Membership.me …
  -/
  rw [← finrank_eq_rank] at h
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    s : Set P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
    h : Iff (Collinear k s) (LE.le (↑(Module.finrank k (Subtype fun x => Membershi …
    ⊢ Iff (Collinear k s) (LE.le (Module.finrank k (Subtype fun x => Membership.me …
  -/
  exact mod_cast h
  /-
    🎉 no goals
  -/


alias ⟨Collinear.finrank_le_one, _⟩ := collinear_iff_finrank_le_one


/-- A subset of a collinear set is collinear. -/
theorem Collinear.subset {s₁ s₂ : Set P} (hs : s₁ ⊆ s₂) (h : Collinear k s₂) : Collinear k s₁ :=
  (Submodule.rank_mono (vectorSpan_mono k hs)).trans h


/-- The `vectorSpan` of collinear points is finite-dimensional. -/
theorem Collinear.finiteDimensional_vectorSpan {s : Set P} (h : Collinear k s) :
    FiniteDimensional k (vectorSpan k s) :=
  IsNoetherian.iff_fg.1
    (IsNoetherian.iff_rank_lt_aleph0.2 (lt_of_le_of_lt h Cardinal.one_lt_aleph0))


/-- The direction of the affine span of collinear points is finite-dimensional. -/
theorem Collinear.finiteDimensional_direction_affineSpan {s : Set P} (h : Collinear k s) :
    FiniteDimensional k (affineSpan k s).direction :=
  (direction_affineSpan k s).symm ▸ h.finiteDimensional_vectorSpan


/-- The empty set is collinear. -/
theorem collinear_empty : Collinear k (∅ : Set P) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ⊢ Collinear k EmptyCollection.emptyCollection
  -/
  rw [collinear_iff_rank_le_one, vectorSpan_empty]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ⊢ LE.le (Module.rank k (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A single point is collinear. -/
theorem collinear_singleton (p : P) : Collinear k ({p} : Set P) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p : P
    ⊢ Collinear k (Singleton.singleton p)
  -/
  rw [collinear_iff_rank_le_one, vectorSpan_singleton]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p : P
    ⊢ LE.le (Module.rank k (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a point `p₀` in a set of points, that set is collinear if and
only if the points can all be expressed as multiples of the same
vector, added to `p₀`. -/
theorem collinear_iff_of_mem {s : Set P} {p₀ : P} (h : p₀ ∈ s) :
    Collinear k s ↔ ∃ v : V, ∀ p ∈ s, ∃ r : k, p = r • v +ᵥ p₀ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p₀ : P
    h : Membership.mem s p₀
    ⊢ Iff (Collinear k s) (Exists fun v => ∀ (p : P), Membership.mem s p → Exists  …
  -/
  simp_rw [collinear_iff_rank_le_one, rank_submodule_le_one_iff', Submodule.le_span_singleton_iff]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p₀ : P
    h : Membership.mem s p₀
    ⊢ Iff (Exists fun v₀ => ∀ (v : V), Membership.mem (vectorSpan k s) v → Exists  …
  -/
  constructor
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      ⊢ (Exists fun v₀ => ∀ (v : V), Membership.mem (vectorSpan k s) v → Exists fun  …
    -/
  · rintro ⟨v₀, hv⟩
    /-
      case mp.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      v₀ : V
      hv : ∀ (v : V), Membership.mem (vectorSpan k s) v → Exists fun r => Eq (HSMul. …
      ⊢ Exists fun v => ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd. …
    -/
    use v₀
    /-
      case h
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      v₀ : V
      hv : ∀ (v : V), Membership.mem (vectorSpan k s) v → Exists fun r => Eq (HSMul. …
      ⊢ ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSM …
    -/
    intro p hp
    /-
      case h
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      v₀ : V
      hv : ∀ (v : V), Membership.mem (vectorSpan k s) v → Exists fun r => Eq (HSMul. …
      p : P
      hp : Membership.mem s p
      ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r v₀) p₀)
    -/
    obtain ⟨r, hr⟩ := hv (p -ᵥ p₀) (vsub_mem_vectorSpan k hp h)
    /-
      case h.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      v₀ : V
      hv : ∀ (v : V), Membership.mem (vectorSpan k s) v → Exists fun r => Eq (HSMul. …
      p : P
      hp : Membership.mem s p
      r : k
      hr : Eq (HSMul.hSMul r v₀) (VSub.vsub p p₀)
      ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r v₀) p₀)
    -/
    use r
    /-
      case h
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      v₀ : V
      hv : ∀ (v : V), Membership.mem (vectorSpan k s) v → Exists fun r => Eq (HSMul. …
      p : P
      hp : Membership.mem s p
      r : k
      hr : Eq (HSMul.hSMul r v₀) (VSub.vsub p p₀)
      ⊢ Eq p (HVAdd.hVAdd (HSMul.hSMul r v₀) p₀)
    -/
    rw [eq_vadd_iff_vsub_eq]
    /-
      case h
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      v₀ : V
      hv : ∀ (v : V), Membership.mem (vectorSpan k s) v → Exists fun r => Eq (HSMul. …
      p : P
      hp : Membership.mem s p
      r : k
      hr : Eq (HSMul.hSMul r v₀) (VSub.vsub p p₀)
      ⊢ Eq (VSub.vsub p p₀) (HSMul.hSMul r v₀)
    -/
    exact hr.symm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      ⊢ (Exists fun v => ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd …
    -/
  · rintro ⟨v, hp₀v⟩
    /-
      case mpr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      v : V
      hp₀v : ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMu …
      ⊢ Exists fun v₀ => ∀ (v : V), Membership.mem (vectorSpan k s) v → Exists fun r …
    -/
    use v
    /-
      case h
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      v : V
      hp₀v : ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMu …
      ⊢ ∀ (v_1 : V), Membership.mem (vectorSpan k s) v_1 → Exists fun r => Eq (HSMul …
    -/
    intro w hw
    have hs : vectorSpan k s ≤ k ∙ v := by
      rw [vectorSpan_eq_span_vsub_set_right k h, Submodule.span_le, Set.subset_def]
      intro x hx
      rw [SetLike.mem_coe, Submodule.mem_span_singleton]
      rw [Set.mem_image] at hx
      rcases hx with ⟨p, hp, rfl⟩
      rcases hp₀v p hp with ⟨r, rfl⟩
      use r
      simp
    /-
      case h
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      v : V
      hp₀v : ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMu …
      w : V
      hw : Membership.mem (vectorSpan k s) w
      hs : LE.le (vectorSpan k s) (Submodule.span k (Singleton.singleton v))
      ⊢ Exists fun r => Eq (HSMul.hSMul r v) w
    -/
    have hw' := SetLike.le_def.1 hs hw
    /-
      case h
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₀ : P
      h : Membership.mem s p₀
      v : V
      hp₀v : ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMu …
      w : V
      hw : Membership.mem (vectorSpan k s) w
      hs : LE.le (vectorSpan k s) (Submodule.span k (Singleton.singleton v))
      hw' : Membership.mem (Submodule.span k (Singleton.singleton v)) w
      ⊢ Exists fun r => Eq (HSMul.hSMul r v) w
    -/
    rwa [Submodule.mem_span_singleton] at hw'
    /-
      🎉 no goals
    -/


/-- A set of points is collinear if and only if they can all be
expressed as multiples of the same vector, added to the same base
point. -/
theorem collinear_iff_exists_forall_eq_smul_vadd (s : Set P) :
    Collinear k s ↔ ∃ (p₀ : P) (v : V), ∀ p ∈ s, ∃ r : k, p = r • v +ᵥ p₀ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    ⊢ Iff (Collinear k s) (Exists fun p₀ => Exists fun v => ∀ (p : P), Membership. …
  -/
  rcases Set.eq_empty_or_nonempty s with (rfl | ⟨⟨p₁, hp₁⟩⟩)
    /-
      case inl
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      ⊢ Iff (Collinear k EmptyCollection.emptyCollection) (Exists fun p₀ => Exists f …
    -/
  · simp [collinear_empty]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₁ : P
      hp₁ : Membership.mem s p₁
      ⊢ Iff (Collinear k s) (Exists fun p₀ => Exists fun v => ∀ (p : P), Membership. …
    -/
  · rw [collinear_iff_of_mem hp₁]
    /-
      case inr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p₁ : P
      hp₁ : Membership.mem s p₁
      ⊢ Iff (Exists fun v => ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (H …
    -/
    constructor
      /-
        case inr.intro.mp
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : DivisionRing k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s : Set P
        p₁ : P
        hp₁ : Membership.mem s p₁
        ⊢ (Exists fun v => ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd …
      -/
    · exact fun h => ⟨p₁, h⟩
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.mpr
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : DivisionRing k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s : Set P
        p₁ : P
        hp₁ : Membership.mem s p₁
        ⊢ (Exists fun p₀ => Exists fun v => ∀ (p : P), Membership.mem s p → Exists fun …
      -/
    · rintro ⟨p, v, hv⟩
      /-
        case inr.intro.mpr.intro.intro
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : DivisionRing k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s : Set P
        p₁ : P
        hp₁ : Membership.mem s p₁
        p : P
        v : V
        hv : ∀ (p_1 : P), Membership.mem s p_1 → Exists fun r => Eq p_1 (HVAdd.hVAdd ( …
        ⊢ Exists fun v => ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd. …
      -/
      use v
      /-
        case h
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : DivisionRing k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s : Set P
        p₁ : P
        hp₁ : Membership.mem s p₁
        p : P
        v : V
        hv : ∀ (p_1 : P), Membership.mem s p_1 → Exists fun r => Eq p_1 (HVAdd.hVAdd ( …
        ⊢ ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSM …
      -/
      intro p₂ hp₂
      /-
        case h
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : DivisionRing k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s : Set P
        p₁ : P
        hp₁ : Membership.mem s p₁
        p : P
        v : V
        hv : ∀ (p_1 : P), Membership.mem s p_1 → Exists fun r => Eq p_1 (HVAdd.hVAdd ( …
        p₂ : P
        hp₂ : Membership.mem s p₂
        ⊢ Exists fun r => Eq p₂ (HVAdd.hVAdd (HSMul.hSMul r v) p₁)
      -/
      rcases hv p₂ hp₂ with ⟨r, rfl⟩
      /-
        case h.intro
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : DivisionRing k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s : Set P
        p₁ : P
        hp₁ : Membership.mem s p₁
        p : P
        v : V
        hv : ∀ (p_1 : P), Membership.mem s p_1 → Exists fun r => Eq p_1 (HVAdd.hVAdd ( …
        r : k
        hp₂ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r v) p)
        ⊢ Exists fun r_1 => Eq (HVAdd.hVAdd (HSMul.hSMul r v) p) (HVAdd.hVAdd (HSMul.h …
      -/
      rcases hv p₁ hp₁ with ⟨r₁, rfl⟩
      /-
        case h.intro.intro
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : DivisionRing k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s : Set P
        p : P
        v : V
        hv : ∀ (p_1 : P), Membership.mem s p_1 → Exists fun r => Eq p_1 (HVAdd.hVAdd ( …
        r : k
        hp₂ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r v) p)
        r₁ : k
        hp₁ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r₁ v) p)
        ⊢ Exists fun r_1 => Eq (HVAdd.hVAdd (HSMul.hSMul r v) p) (HVAdd.hVAdd (HSMul.h …
      -/
      use r - r₁
      /-
        case h
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : DivisionRing k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s : Set P
        p : P
        v : V
        hv : ∀ (p_1 : P), Membership.mem s p_1 → Exists fun r => Eq p_1 (HVAdd.hVAdd ( …
        r : k
        hp₂ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r v) p)
        r₁ : k
        hp₁ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r₁ v) p)
        ⊢ Eq (HVAdd.hVAdd (HSMul.hSMul r v) p) (HVAdd.hVAdd (HSMul.hSMul (HSub.hSub r  …
      -/
      simp [vadd_vadd, ← add_smul]
      /-
        🎉 no goals
      -/


/-- Two points are collinear. -/
theorem collinear_pair (p₁ p₂ : P) : Collinear k ({p₁, p₂} : Set P) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    ⊢ Collinear k (Insert.insert p₁ (Singleton.singleton p₂))
  -/
  rw [collinear_iff_exists_forall_eq_smul_vadd]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    ⊢ Exists fun p₀ => Exists fun v => ∀ (p : P), Membership.mem (Insert.insert p₁ …
  -/
  use p₁, p₂ -ᵥ p₁
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    ⊢ ∀ (p : P), Membership.mem (Insert.insert p₁ (Singleton.singleton p₂)) p → Ex …
  -/
  intro p hp
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p : P
    hp : Membership.mem (Insert.insert p₁ (Singleton.singleton p₂)) p
    ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p₂ p₁)) p₁)
  -/
  rw [Set.mem_insert_iff, Set.mem_singleton_iff] at hp
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p : P
    hp : Or (Eq p p₁) (Eq p p₂)
    ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p₂ p₁)) p₁)
  -/
  cases' hp with hp hp
    /-
      case h.inl
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p₁ p₂ p : P
      hp : Eq p p₁
      ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p₂ p₁)) p₁)
    -/
  · use 0
    /-
      case h
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p₁ p₂ p : P
      hp : Eq p p₁
      ⊢ Eq p (HVAdd.hVAdd (HSMul.hSMul 0 (VSub.vsub p₂ p₁)) p₁)
    -/
    simp [hp]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p₁ p₂ p : P
      hp : Eq p p₂
      ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p₂ p₁)) p₁)
    -/
  · use 1
    /-
      case h
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p₁ p₂ p : P
      hp : Eq p p₂
      ⊢ Eq p (HVAdd.hVAdd (HSMul.hSMul 1 (VSub.vsub p₂ p₁)) p₁)
    -/
    simp [hp]
    /-
      🎉 no goals
    -/


/-- Three points are affinely independent if and only if they are not
collinear. -/
theorem affineIndependent_iff_not_collinear {p : Fin 3 → P} :
    AffineIndependent k p ↔ ¬Collinear k (Set.range p) := by
  rw [collinear_iff_finrank_le_one,
    affineIndependent_iff_not_finrank_vectorSpan_le k p (Fintype.card_fin 3)]


/-- Three points are collinear if and only if they are not affinely
independent. -/
theorem collinear_iff_not_affineIndependent {p : Fin 3 → P} :
    Collinear k (Set.range p) ↔ ¬AffineIndependent k p := by
  rw [collinear_iff_finrank_le_one,
    finrank_vectorSpan_le_iff_not_affineIndependent k p (Fintype.card_fin 3)]


/-- Three points are affinely independent if and only if they are not collinear. -/
theorem affineIndependent_iff_not_collinear_set {p₁ p₂ p₃ : P} :
    AffineIndependent k ![p₁, p₂, p₃] ↔ ¬Collinear k ({p₁, p₂, p₃} : Set P) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ Iff (AffineIndependent k (Matrix.vecCons p₁ (Matrix.vecCons p₂ (Matrix.vecCo …
  -/
  rw [affineIndependent_iff_not_collinear]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ Iff (Not (Collinear k (Set.range (Matrix.vecCons p₁ (Matrix.vecCons p₂ (Matr …
  -/
  simp_rw [Matrix.range_cons, Matrix.range_empty, Set.singleton_union, insert_emptyc_eq]
  /-
    🎉 no goals
  -/


/-- Three points are collinear if and only if they are not affinely independent. -/
theorem collinear_iff_not_affineIndependent_set {p₁ p₂ p₃ : P} :
    Collinear k ({p₁, p₂, p₃} : Set P) ↔ ¬AffineIndependent k ![p₁, p₂, p₃] :=
  affineIndependent_iff_not_collinear_set.not_left.symm


/-- Three points are affinely independent if and only if they are not collinear. -/
theorem affineIndependent_iff_not_collinear_of_ne {p : Fin 3 → P} {i₁ i₂ i₃ : Fin 3} (h₁₂ : i₁ ≠ i₂)
    (h₁₃ : i₁ ≠ i₃) (h₂₃ : i₂ ≠ i₃) :
    AffineIndependent k p ↔ ¬Collinear k ({p i₁, p i₂, p i₃} : Set P) := by
  have hu : (Finset.univ : Finset (Fin 3)) = {i₁, i₂, i₃} := by
    -- Porting note: Originally `by decide!`
    revert i₁ i₂ i₃; decide
  rw [affineIndependent_iff_not_collinear, ← Set.image_univ, ← Finset.coe_univ, hu,
    Finset.coe_insert, Finset.coe_insert, Finset.coe_singleton, Set.image_insert_eq, Set.image_pair]


/-- Three points are collinear if and only if they are not affinely independent. -/
theorem collinear_iff_not_affineIndependent_of_ne {p : Fin 3 → P} {i₁ i₂ i₃ : Fin 3} (h₁₂ : i₁ ≠ i₂)
    (h₁₃ : i₁ ≠ i₃) (h₂₃ : i₂ ≠ i₃) :
    Collinear k ({p i₁, p i₂, p i₃} : Set P) ↔ ¬AffineIndependent k p :=
  (affineIndependent_iff_not_collinear_of_ne h₁₂ h₁₃ h₂₃).not_left.symm


/-- If three points are not collinear, the first and second are different. -/
theorem ne₁₂_of_not_collinear {p₁ p₂ p₃ : P} (h : ¬Collinear k ({p₁, p₂, p₃} : Set P)) :
    p₁ ≠ p₂ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ : P
    h : Not (Collinear k (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton  …
    ⊢ Ne p₁ p₂
  -/
  rintro rfl
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₃ : P
    h : Not (Collinear k (Insert.insert p₁ (Insert.insert p₁ (Singleton.singleton  …
    ⊢ False
  -/
  simp [collinear_pair] at h
  /-
    🎉 no goals
  -/


/-- If three points are not collinear, the first and third are different. -/
theorem ne₁₃_of_not_collinear {p₁ p₂ p₃ : P} (h : ¬Collinear k ({p₁, p₂, p₃} : Set P)) :
    p₁ ≠ p₃ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ : P
    h : Not (Collinear k (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton  …
    ⊢ Ne p₁ p₃
  -/
  rintro rfl
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    h : Not (Collinear k (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton  …
    ⊢ False
  -/
  simp [collinear_pair] at h
  /-
    🎉 no goals
  -/


/-- If three points are not collinear, the second and third are different. -/
theorem ne₂₃_of_not_collinear {p₁ p₂ p₃ : P} (h : ¬Collinear k ({p₁, p₂, p₃} : Set P)) :
    p₂ ≠ p₃ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ : P
    h : Not (Collinear k (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton  …
    ⊢ Ne p₂ p₃
  -/
  rintro rfl
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    h : Not (Collinear k (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton  …
    ⊢ False
  -/
  simp [collinear_pair] at h
  /-
    🎉 no goals
  -/


/-- A point in a collinear set of points lies in the affine span of any two distinct points of
that set. -/
theorem Collinear.mem_affineSpan_of_mem_of_ne {s : Set P} (h : Collinear k s) {p₁ p₂ p₃ : P}
    (hp₁ : p₁ ∈ s) (hp₂ : p₂ ∈ s) (hp₃ : p₃ ∈ s) (hp₁p₂ : p₁ ≠ p₂) : p₃ ∈ line[k, p₁, p₂] := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    h : Collinear k s
    p₁ p₂ p₃ : P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    hp₃ : Membership.mem s p₃
    hp₁p₂ : Ne p₁ p₂
    ⊢ Membership.mem (affineSpan k (Insert.insert p₁ (Singleton.singleton p₂))) p₃
  -/
  rw [collinear_iff_of_mem hp₁] at h
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p₁ : P
    h : Exists fun v => ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAd …
    p₂ p₃ : P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    hp₃ : Membership.mem s p₃
    hp₁p₂ : Ne p₁ p₂
    ⊢ Membership.mem (affineSpan k (Insert.insert p₁ (Singleton.singleton p₂))) p₃
  -/
  rcases h with ⟨v, h⟩
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p₁ p₂ p₃ : P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    hp₃ : Membership.mem s p₃
    hp₁p₂ : Ne p₁ p₂
    v : V
    h : ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMul.h …
    ⊢ Membership.mem (affineSpan k (Insert.insert p₁ (Singleton.singleton p₂))) p₃
  -/
  rcases h p₂ hp₂ with ⟨r₂, rfl⟩
  /-
    case intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p₁ p₃ : P
    hp₁ : Membership.mem s p₁
    hp₃ : Membership.mem s p₃
    v : V
    h : ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMul.h …
    r₂ : k
    hp₂ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r₂ v) p₁)
    hp₁p₂ : Ne p₁ (HVAdd.hVAdd (HSMul.hSMul r₂ v) p₁)
    ⊢ Membership.mem (affineSpan k (Insert.insert p₁ (Singleton.singleton (HVAdd.h …
  -/
  rcases h p₃ hp₃ with ⟨r₃, rfl⟩
  /-
    case intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p₁ : P
    hp₁ : Membership.mem s p₁
    v : V
    h : ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMul.h …
    r₂ : k
    hp₂ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r₂ v) p₁)
    hp₁p₂ : Ne p₁ (HVAdd.hVAdd (HSMul.hSMul r₂ v) p₁)
    r₃ : k
    hp₃ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r₃ v) p₁)
    ⊢ Membership.mem (affineSpan k (Insert.insert p₁ (Singleton.singleton (HVAdd.h …
  -/
  rw [vadd_left_mem_affineSpan_pair]
  /-
    case intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p₁ : P
    hp₁ : Membership.mem s p₁
    v : V
    h : ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMul.h …
    r₂ : k
    hp₂ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r₂ v) p₁)
    hp₁p₂ : Ne p₁ (HVAdd.hVAdd (HSMul.hSMul r₂ v) p₁)
    r₃ : k
    hp₃ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r₃ v) p₁)
    ⊢ Exists fun r => Eq (HSMul.hSMul r (VSub.vsub (HVAdd.hVAdd (HSMul.hSMul r₂ v) …
  -/
  refine ⟨r₃ / r₂, ?_⟩
  have h₂ : r₂ ≠ 0 := by
    rintro rfl
    simp at hp₁p₂
  /-
    case intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p₁ : P
    hp₁ : Membership.mem s p₁
    v : V
    h : ∀ (p : P), Membership.mem s p → Exists fun r => Eq p (HVAdd.hVAdd (HSMul.h …
    r₂ : k
    hp₂ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r₂ v) p₁)
    hp₁p₂ : Ne p₁ (HVAdd.hVAdd (HSMul.hSMul r₂ v) p₁)
    r₃ : k
    hp₃ : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r₃ v) p₁)
    h₂ : Ne r₂ 0
    ⊢ Eq (HSMul.hSMul (HDiv.hDiv r₃ r₂) (VSub.vsub (HVAdd.hVAdd (HSMul.hSMul r₂ v) …
  -/
  simp [smul_smul, h₂]
  /-
    🎉 no goals
  -/


/-- The affine span of any two distinct points of a collinear set of points equals the affine
span of the whole set. -/
theorem Collinear.affineSpan_eq_of_ne {s : Set P} (h : Collinear k s) {p₁ p₂ : P} (hp₁ : p₁ ∈ s)
    (hp₂ : p₂ ∈ s) (hp₁p₂ : p₁ ≠ p₂) : line[k, p₁, p₂] = affineSpan k s :=
  le_antisymm (affineSpan_mono _ (Set.insert_subset_iff.2 ⟨hp₁, Set.singleton_subset_iff.2 hp₂⟩))
    (affineSpan_le.2 fun _ hp => h.mem_affineSpan_of_mem_of_ne hp₁ hp₂ hp hp₁p₂)


/-- Given a collinear set of points, and two distinct points `p₂` and `p₃` in it, a point `p₁` is
collinear with the set if and only if it is collinear with `p₂` and `p₃`. -/
theorem Collinear.collinear_insert_iff_of_ne {s : Set P} (h : Collinear k s) {p₁ p₂ p₃ : P}
    (hp₂ : p₂ ∈ s) (hp₃ : p₃ ∈ s) (hp₂p₃ : p₂ ≠ p₃) :
    Collinear k (insert p₁ s) ↔ Collinear k ({p₁, p₂, p₃} : Set P) := by
  have hv : vectorSpan k (insert p₁ s) = vectorSpan k ({p₁, p₂, p₃} : Set P) := by
    -- Porting note: Original proof used `conv_lhs` and `conv_rhs`, but these tactics timed out.
    rw [← direction_affineSpan, ← affineSpan_insert_affineSpan]
    symm
    rw [← direction_affineSpan, ← affineSpan_insert_affineSpan, h.affineSpan_eq_of_ne hp₂ hp₃ hp₂p₃]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    h : Collinear k s
    p₁ p₂ p₃ : P
    hp₂ : Membership.mem s p₂
    hp₃ : Membership.mem s p₃
    hp₂p₃ : Ne p₂ p₃
    hv : Eq (vectorSpan k (Insert.insert p₁ s)) (vectorSpan k (Insert.insert p₁ (I …
    ⊢ Iff (Collinear k (Insert.insert p₁ s)) (Collinear k (Insert.insert p₁ (Inser …
  -/
  rw [Collinear, Collinear, hv]
  /-
    🎉 no goals
  -/


/-- Adding a point in the affine span of a set does not change whether that set is collinear. -/
theorem collinear_insert_iff_of_mem_affineSpan {s : Set P} {p : P} (h : p ∈ affineSpan k s) :
    Collinear k (insert p s) ↔ Collinear k s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    h : Membership.mem (affineSpan k s) p
    ⊢ Iff (Collinear k (Insert.insert p s)) (Collinear k s)
  -/
  rw [Collinear, Collinear, vectorSpan_insert_eq_vectorSpan h]
  /-
    🎉 no goals
  -/


/-- If a point lies in the affine span of two points, those three points are collinear. -/
theorem collinear_insert_of_mem_affineSpan_pair {p₁ p₂ p₃ : P} (h : p₁ ∈ line[k, p₂, p₃]) :
    Collinear k ({p₁, p₂, p₃} : Set P) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ : P
    h : Membership.mem (affineSpan k (Insert.insert p₂ (Singleton.singleton p₃))) p₁
    ⊢ Collinear k (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃)))
  -/
  rw [collinear_insert_iff_of_mem_affineSpan h]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ : P
    h : Membership.mem (affineSpan k (Insert.insert p₂ (Singleton.singleton p₃))) p₁
    ⊢ Collinear k (Insert.insert p₂ (Singleton.singleton p₃))
  -/
  exact collinear_pair _ _ _
  /-
    🎉 no goals
  -/


/-- If two points lie in the affine span of two points, those four points are collinear. -/
theorem collinear_insert_insert_of_mem_affineSpan_pair {p₁ p₂ p₃ p₄ : P} (h₁ : p₁ ∈ line[k, p₃, p₄])
    (h₂ : p₂ ∈ line[k, p₃, p₄]) : Collinear k ({p₁, p₂, p₃, p₄} : Set P) := by
  rw [collinear_insert_iff_of_mem_affineSpan
      ((AffineSubspace.le_def' _ _).1 (affineSpan_mono k (Set.subset_insert _ _)) _ h₁),
    collinear_insert_iff_of_mem_affineSpan h₂]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ p₄ : P
    h₁ : Membership.mem (affineSpan k (Insert.insert p₃ (Singleton.singleton p₄))) …
    h₂ : Membership.mem (affineSpan k (Insert.insert p₃ (Singleton.singleton p₄))) …
    ⊢ Collinear k (Insert.insert p₃ (Singleton.singleton p₄))
  -/
  exact collinear_pair _ _ _
  /-
    🎉 no goals
  -/


/-- If three points lie in the affine span of two points, those five points are collinear. -/
theorem collinear_insert_insert_insert_of_mem_affineSpan_pair {p₁ p₂ p₃ p₄ p₅ : P}
    (h₁ : p₁ ∈ line[k, p₄, p₅]) (h₂ : p₂ ∈ line[k, p₄, p₅]) (h₃ : p₃ ∈ line[k, p₄, p₅]) :
    Collinear k ({p₁, p₂, p₃, p₄, p₅} : Set P) := by
  rw [collinear_insert_iff_of_mem_affineSpan
      ((AffineSubspace.le_def' _ _).1
        (affineSpan_mono k ((Set.subset_insert _ _).trans (Set.subset_insert _ _))) _ h₁),
    collinear_insert_iff_of_mem_affineSpan
      ((AffineSubspace.le_def' _ _).1 (affineSpan_mono k (Set.subset_insert _ _)) _ h₂),
    collinear_insert_iff_of_mem_affineSpan h₃]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ p₄ p₅ : P
    h₁ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₂ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₃ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    ⊢ Collinear k (Insert.insert p₄ (Singleton.singleton p₅))
  -/
  exact collinear_pair _ _ _
  /-
    🎉 no goals
  -/


/-- If three points lie in the affine span of two points, the first four points are collinear. -/
theorem collinear_insert_insert_insert_left_of_mem_affineSpan_pair {p₁ p₂ p₃ p₄ p₅ : P}
    (h₁ : p₁ ∈ line[k, p₄, p₅]) (h₂ : p₂ ∈ line[k, p₄, p₅]) (h₃ : p₃ ∈ line[k, p₄, p₅]) :
    Collinear k ({p₁, p₂, p₃, p₄} : Set P) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ p₄ p₅ : P
    h₁ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₂ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₃ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    ⊢ Collinear k (Insert.insert p₁ (Insert.insert p₂ (Insert.insert p₃ (Singleton …
  -/
  refine (collinear_insert_insert_insert_of_mem_affineSpan_pair h₁ h₂ h₃).subset ?_
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ p₄ p₅ : P
    h₁ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₂ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₃ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    ⊢ HasSubset.Subset (Insert.insert p₁ (Insert.insert p₂ (Insert.insert p₃ (Sing …
  -/
  repeat apply Set.insert_subset_insert
  /-
    case h.h.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ p₄ p₅ : P
    h₁ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₂ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₃ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    ⊢ HasSubset.Subset (Singleton.singleton p₄) (Insert.insert p₄ (Singleton.singl …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If three points lie in the affine span of two points, the first three points are collinear. -/
theorem collinear_triple_of_mem_affineSpan_pair {p₁ p₂ p₃ p₄ p₅ : P} (h₁ : p₁ ∈ line[k, p₄, p₅])
    (h₂ : p₂ ∈ line[k, p₄, p₅]) (h₃ : p₃ ∈ line[k, p₄, p₅]) :
    Collinear k ({p₁, p₂, p₃} : Set P) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ p₄ p₅ : P
    h₁ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₂ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₃ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    ⊢ Collinear k (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃)))
  -/
  refine (collinear_insert_insert_insert_left_of_mem_affineSpan_pair h₁ h₂ h₃).subset ?_
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ p₃ p₄ p₅ : P
    h₁ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₂ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    h₃ : Membership.mem (affineSpan k (Insert.insert p₄ (Singleton.singleton p₅))) …
    ⊢ HasSubset.Subset (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
  -/
  simp [Set.insert_subset_insert]
  /-
    🎉 no goals
  -/


/-- A set of points is coplanar if their `vectorSpan` has dimension at most `2`. -/
def Coplanar (s : Set P) : Prop :=
  Module.rank k (vectorSpan k s) ≤ 2


/-- The `vectorSpan` of coplanar points is finite-dimensional. -/
theorem Coplanar.finiteDimensional_vectorSpan {s : Set P} (h : Coplanar k s) :
    FiniteDimensional k (vectorSpan k s) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    h : Coplanar k s
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
  -/
  refine IsNoetherian.iff_fg.1 (IsNoetherian.iff_rank_lt_aleph0.2 (lt_of_le_of_lt h ?_))
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    h : Coplanar k s
    ⊢ LT.lt 2 Cardinal.aleph0
  -/
  exact Cardinal.lt_aleph0.2 ⟨2, rfl⟩
  /-
    🎉 no goals
  -/


/-- The direction of the affine span of coplanar points is finite-dimensional. -/
theorem Coplanar.finiteDimensional_direction_affineSpan {s : Set P} (h : Coplanar k s) :
    FiniteDimensional k (affineSpan k s).direction :=
  (direction_affineSpan k s).symm ▸ h.finiteDimensional_vectorSpan


/-- A set of points, whose `vectorSpan` is finite-dimensional, is coplanar if and only if their
`vectorSpan` has dimension at most `2`. -/
theorem coplanar_iff_finrank_le_two {s : Set P} [FiniteDimensional k (vectorSpan k s)] :
    Coplanar k s ↔ finrank k (vectorSpan k s) ≤ 2 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    s : Set P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
    ⊢ Iff (Coplanar k s) (LE.le (Module.finrank k (Subtype fun x => Membership.mem …
  -/
  have h : Coplanar k s ↔ Module.rank k (vectorSpan k s) ≤ 2 := Iff.rfl
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    s : Set P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
    h : Iff (Coplanar k s) (LE.le (Module.rank k (Subtype fun x => Membership.mem  …
    ⊢ Iff (Coplanar k s) (LE.le (Module.finrank k (Subtype fun x => Membership.mem …
  -/
  rw [← finrank_eq_rank] at h
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    s : Set P
    inst✝ : FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
    h : Iff (Coplanar k s) (LE.le (↑(Module.finrank k (Subtype fun x => Membership …
    ⊢ Iff (Coplanar k s) (LE.le (Module.finrank k (Subtype fun x => Membership.mem …
  -/
  exact mod_cast h
  /-
    🎉 no goals
  -/


alias ⟨Coplanar.finrank_le_two, _⟩ := coplanar_iff_finrank_le_two


/-- A subset of a coplanar set is coplanar. -/
theorem Coplanar.subset {s₁ s₂ : Set P} (hs : s₁ ⊆ s₂) (h : Coplanar k s₂) : Coplanar k s₁ :=
  (Submodule.rank_mono (vectorSpan_mono k hs)).trans h


/-- Collinear points are coplanar. -/
theorem Collinear.coplanar {s : Set P} (h : Collinear k s) : Coplanar k s :=
  le_trans h one_le_two


/-- The empty set is coplanar. -/
theorem coplanar_empty : Coplanar k (∅ : Set P) :=
  (collinear_empty k P).coplanar


/-- A single point is coplanar. -/
theorem coplanar_singleton (p : P) : Coplanar k ({p} : Set P) :=
  (collinear_singleton k p).coplanar


/-- Two points are coplanar. -/
theorem coplanar_pair (p₁ p₂ : P) : Coplanar k ({p₁, p₂} : Set P) :=
  (collinear_pair k p₁ p₂).coplanar


/-- Adding a point in the affine span of a set does not change whether that set is coplanar. -/
theorem coplanar_insert_iff_of_mem_affineSpan {s : Set P} {p : P} (h : p ∈ affineSpan k s) :
    Coplanar k (insert p s) ↔ Coplanar k s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    h : Membership.mem (affineSpan k s) p
    ⊢ Iff (Coplanar k (Insert.insert p s)) (Coplanar k s)
  -/
  rw [Coplanar, Coplanar, vectorSpan_insert_eq_vectorSpan h]
  /-
    🎉 no goals
  -/


/-- Adding a point to a finite-dimensional subspace increases the dimension by at most one. -/
theorem finrank_vectorSpan_insert_le (s : AffineSubspace k P) (p : P) :
    finrank k (vectorSpan k (insert p (s : Set P))) ≤ finrank k s.direction + 1 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Inse …
  -/
  by_cases hf : FiniteDimensional k s.direction; swap
  · have hf' : ¬FiniteDimensional k (vectorSpan k (insert p (s : Set P))) := by
      intro h
      have h' : s.direction ≤ vectorSpan k (insert p (s : Set P)) := by
        conv_lhs => rw [← affineSpan_coe s, direction_affineSpan]
        exact vectorSpan_mono k (Set.subset_insert _ _)
      exact hf (Submodule.finiteDimensional_of_le h')
    /-
      case neg
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf : Not (FiniteDimensional k (Subtype fun x => Membership.mem s.direction x))
      hf' : Not (FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k  …
      ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Inse …
    -/
    rw [finrank_of_infinite_dimensional hf, finrank_of_infinite_dimensional hf', zero_add]
    /-
      case neg
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf : Not (FiniteDimensional k (Subtype fun x => Membership.mem s.direction x))
      hf' : Not (FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k  …
      ⊢ LE.le 0 1
    -/
    exact zero_le_one
    /-
      🎉 no goals
    -/
  /-
    case pos
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hf : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
    ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Inse …
  -/
  have : FiniteDimensional k s.direction := hf
  /-
    case pos
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
    ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Inse …
  -/
  rw [← direction_affineSpan, ← affineSpan_insert_affineSpan]
  /-
    case pos
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
    ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (affineSpan k (Inse …
  -/
  rcases (s : Set P).eq_empty_or_nonempty with (hs | ⟨p₀, hp₀⟩)
    /-
      case pos.inl
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      hs : Eq (↑s) EmptyCollection.emptyCollection
      ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (affineSpan k (Inse …
    -/
  · rw [coe_eq_bot_iff] at hs
    rw [hs, bot_coe, span_empty, bot_coe, direction_affineSpan, direction_bot, finrank_bot,
      zero_add]
    /-
      case pos.inl
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      hs : Eq s Bot.bot
      ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Inse …
    -/
    convert zero_le_one' ℕ
    /-
      case h.e'_3
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      hs : Eq s Bot.bot
      ⊢ Eq (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Insert. …
    -/
    rw [← finrank_bot k V]
    /-
      case h.e'_3
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      hs : Eq s Bot.bot
      ⊢ Eq (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Insert. …
    -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
    convert rfl <;> simp
                    /-
                      🎉 no goals
                    -/
    /-
      case pos.inr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (affineSpan k (Inse …
    -/
  · rw [affineSpan_coe, direction_affineSpan_insert hp₀, add_comm]
    /-
      case pos.inr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (Max.max (Submodule …
    -/
    refine (Submodule.finrank_add_le_finrank_add_finrank _ _).trans (add_le_add_right ?_ _)
    /-
      case pos.inr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (Submodule.span k ( …
    -/
    refine finrank_le_one ⟨p -ᵥ p₀, Submodule.mem_span_singleton_self _⟩ fun v => ?_
    /-
      case pos.inr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      v : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singleton (VS …
      ⊢ Exists fun c => Eq (HSMul.hSMul c ⟨VSub.vsub p p₀, ⋯⟩) v
    -/
    have h := v.property
    /-
      case pos.inr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      v : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singleton (VS …
      h : Membership.mem (Submodule.span k (Singleton.singleton (VSub.vsub p p₀))) ↑v
      ⊢ Exists fun c => Eq (HSMul.hSMul c ⟨VSub.vsub p p₀, ⋯⟩) v
    -/
    rw [Submodule.mem_span_singleton] at h
    /-
      case pos.inr.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      v : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singleton (VS …
      h : Exists fun a => Eq (HSMul.hSMul a (VSub.vsub p p₀)) ↑v
      ⊢ Exists fun c => Eq (HSMul.hSMul c ⟨VSub.vsub p p₀, ⋯⟩) v
    -/
    rcases h with ⟨c, hc⟩
    /-
      case pos.inr.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      v : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singleton (VS …
      c : k
      hc : Eq (HSMul.hSMul c (VSub.vsub p p₀)) ↑v
      ⊢ Exists fun c => Eq (HSMul.hSMul c ⟨VSub.vsub p p₀, ⋯⟩) v
    -/
    refine ⟨c, ?_⟩
    /-
      case pos.inr.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      v : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singleton (VS …
      c : k
      hc : Eq (HSMul.hSMul c (VSub.vsub p p₀)) ↑v
      ⊢ Eq (HSMul.hSMul c ⟨VSub.vsub p p₀, ⋯⟩) v
    -/
    ext
    /-
      case pos.inr.intro.intro.a
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : DivisionRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hf this : FiniteDimensional k (Subtype fun x => Membership.mem s.direction x)
      p₀ : P
      hp₀ : Membership.mem (↑s) p₀
      v : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singleton (VS …
      c : k
      hc : Eq (HSMul.hSMul c (VSub.vsub p p₀)) ↑v
      ⊢ Eq ↑(HSMul.hSMul c ⟨VSub.vsub p p₀, ⋯⟩) ↑v
    -/
    exact hc
    /-
      🎉 no goals
    -/


/-- Adding a point to a set with a finite-dimensional span increases the dimension by at most
one. -/
theorem finrank_vectorSpan_insert_le_set (s : Set P) (p : P) :
    finrank k (vectorSpan k (insert p s)) ≤ finrank k (vectorSpan k s) + 1 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Inse …
  -/
  rw [← direction_affineSpan, ← affineSpan_insert_affineSpan, direction_affineSpan]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Inse …
  -/
  refine (finrank_vectorSpan_insert_le _ _).trans (add_le_add_right ?_ _)
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (affineSpan k s).di …
  -/
  rw [direction_affineSpan]
  /-
    🎉 no goals
  -/


/-- Adding a point to a collinear set produces a coplanar set. -/
theorem Collinear.coplanar_insert {s : Set P} (h : Collinear k s) (p : P) :
    Coplanar k (insert p s) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    h : Collinear k s
    p : P
    ⊢ Coplanar k (Insert.insert p s)
  -/
  have : FiniteDimensional k { x // x ∈ vectorSpan k s } := h.finiteDimensional_vectorSpan
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    h : Collinear k s
    p : P
    this : FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
    ⊢ Coplanar k (Insert.insert p s)
  -/
  rw [coplanar_iff_finrank_le_two]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    h : Collinear k s
    p : P
    this : FiniteDimensional k (Subtype fun x => Membership.mem (vectorSpan k s) x)
    ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k (Inse …
  -/
  exact (finrank_vectorSpan_insert_le_set k s p).trans (add_le_add_right h.finrank_le_one _)
  /-
    🎉 no goals
  -/


/-- A set of points in a two-dimensional space is coplanar. -/
theorem coplanar_of_finrank_eq_two (s : Set P) (h : finrank k V = 2) : Coplanar k s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    h : Eq (Module.finrank k V) 2
    ⊢ Coplanar k s
  -/
  have : FiniteDimensional k V := .of_finrank_eq_succ h
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    h : Eq (Module.finrank k V) 2
    this : FiniteDimensional k V
    ⊢ Coplanar k s
  -/
  rw [coplanar_iff_finrank_le_two, ← h]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    h : Eq (Module.finrank k V) 2
    this : FiniteDimensional k V
    ⊢ LE.le (Module.finrank k (Subtype fun x => Membership.mem (vectorSpan k s) x) …
  -/
  exact Submodule.finrank_le _
  /-
    🎉 no goals
  -/


/-- A set of points in a two-dimensional space is coplanar. -/
theorem coplanar_of_fact_finrank_eq_two (s : Set P) [h : Fact (finrank k V = 2)] : Coplanar k s :=
  coplanar_of_finrank_eq_two s h.out


/-- Three points are coplanar. -/
theorem coplanar_triple (p₁ p₂ p₃ : P) : Coplanar k ({p₁, p₂, p₃} : Set P) :=
  (collinear_pair k p₂ p₃).coplanar_insert p₁


protected theorem finiteDimensional [Finite ι] (b : AffineBasis ι k P) : FiniteDimensional k V :=
  let ⟨i⟩ := b.nonempty
  FiniteDimensional.of_fintype_basis (b.basisOf i)


protected theorem finite [FiniteDimensional k V] (b : AffineBasis ι k P) : Finite ι :=
  finite_of_fin_dim_affineIndependent k b.ind


protected theorem finite_set [FiniteDimensional k V] {s : Set ι} (b : AffineBasis s k P) :
    s.Finite :=
  finite_set_of_fin_dim_affineIndependent k b.ind


theorem card_eq_finrank_add_one [Fintype ι] (b : AffineBasis ι k P) :
    Fintype.card ι = Module.finrank k V + 1 :=
  have : FiniteDimensional k V := b.finiteDimensional
  b.ind.affineSpan_eq_top_iff_card_eq_finrank_add_one.mp b.tot


theorem exists_affineBasis_of_finiteDimensional [Fintype ι] [FiniteDimensional k V]
    (h : Fintype.card ι = Module.finrank k V + 1) : Nonempty (AffineBasis ι k P) := by
  /-
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : DivisionRing k
    inst✝² : Module k V
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional k V
    h : Eq (Fintype.card ι) (HAdd.hAdd (Module.finrank k V) 1)
    ⊢ Nonempty (AffineBasis ι k P)
  -/
  obtain ⟨s, b, hb⟩ := AffineBasis.exists_affineBasis k V P
  /-
    case intro.intro
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : DivisionRing k
    inst✝² : Module k V
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional k V
    h : Eq (Fintype.card ι) (HAdd.hAdd (Module.finrank k V) 1)
    s : Set P
    b : AffineBasis (↑s) k P
    hb : Eq (⇑b) Subtype.val
    ⊢ Nonempty (AffineBasis ι k P)
  -/
  lift s to Finset P using b.finite_set
  /-
    case intro.intro.intro
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : DivisionRing k
    inst✝² : Module k V
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional k V
    h : Eq (Fintype.card ι) (HAdd.hAdd (Module.finrank k V) 1)
    s : Finset P
    b : AffineBasis (↑↑s) k P
    hb : Eq (⇑b) Subtype.val
    ⊢ Nonempty (AffineBasis ι k P)
  -/
  refine ⟨b.reindex <| Fintype.equivOfCardEq ?_⟩
  /-
    case intro.intro.intro
    ι : Type u₁
    k : Type u₂
    V : Type u₃
    P : Type u₄
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : AddTorsor V P
    inst✝³ : DivisionRing k
    inst✝² : Module k V
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional k V
    h : Eq (Fintype.card ι) (HAdd.hAdd (Module.finrank k V) 1)
    s : Finset P
    b : AffineBasis (↑↑s) k P
    hb : Eq (⇑b) Subtype.val
    ⊢ Eq (Fintype.card ↑↑s) (Fintype.card ι)
  -/
  rw [h, ← b.card_eq_finrank_add_one]
  /-
    🎉 no goals
  -/


