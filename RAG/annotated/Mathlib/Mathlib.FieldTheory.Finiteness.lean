/-- A module over a division ring is noetherian if and only if
its dimension (as a cardinal) is strictly less than the first infinite cardinal `ℵ₀`.
-/
theorem iff_rank_lt_aleph0 : IsNoetherian K V ↔ Module.rank K V < ℵ₀ := by
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ⊢ Iff (IsNoetherian K V) (LT.lt (Module.rank K V) Cardinal.aleph0)
  -/
  let b := Basis.ofVectorSpace K V
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    b : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    ⊢ Iff (IsNoetherian K V) (LT.lt (Module.rank K V) Cardinal.aleph0)
  -/
  rw [← b.mk_eq_rank'', lt_aleph0_iff_set_finite]
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    b : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    ⊢ Iff (IsNoetherian K V) (Basis.ofVectorSpaceIndex K V).Finite
  -/
  constructor
    /-
      case mp
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      b : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
      ⊢ IsNoetherian K V → (Basis.ofVectorSpaceIndex K V).Finite
    -/
  · intro
    /-
      case mp
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      b : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
      a✝ : IsNoetherian K V
      ⊢ (Basis.ofVectorSpaceIndex K V).Finite
    -/
    exact (Basis.ofVectorSpaceIndex.linearIndependent K V).set_finite_of_isNoetherian
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      b : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
      ⊢ (Basis.ofVectorSpaceIndex K V).Finite → IsNoetherian K V
    -/
  · intro hbfinite
    refine
      @isNoetherian_of_linearEquiv K (⊤ : Submodule K V) V _ _ _ _ _ (LinearEquiv.ofTop _ rfl)
        (id ?_)
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      b : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
      hbfinite : (Basis.ofVectorSpaceIndex K V).Finite
      ⊢ IsNoetherian K (Subtype fun x => Membership.mem Top.top x)
    -/
    refine isNoetherian_of_fg_of_noetherian _ ⟨Set.Finite.toFinset hbfinite, ?_⟩
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      b : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
      hbfinite : (Basis.ofVectorSpaceIndex K V).Finite
      ⊢ Eq (Submodule.span K ↑hbfinite.toFinset) Top.top
    -/
    rw [Set.Finite.coe_toFinset, ← b.span_eq, Basis.coe_ofVectorSpace, Subtype.range_coe]
    /-
      🎉 no goals
    -/


/-- In a noetherian module over a division ring, all bases are indexed by a finite type. -/
noncomputable def fintypeBasisIndex {ι : Type*} [IsNoetherian K V] (b : Basis ι K V) : Fintype ι :=
  b.fintypeIndexOfRankLtAleph0 (rank_lt_aleph0 K V)


/-- In a noetherian module over a division ring,
`Basis.ofVectorSpace` is indexed by a finite type. -/
noncomputable instance [IsNoetherian K V] : Fintype (Basis.ofVectorSpaceIndex K V) :=
  fintypeBasisIndex (Basis.ofVectorSpace K V)


/-- In a noetherian module over a division ring,
if a basis is indexed by a set, that set is finite. -/
theorem finite_basis_index {ι : Type*} {s : Set ι} [IsNoetherian K V] (b : Basis s K V) :
    s.Finite :=
  b.finite_index_of_rank_lt_aleph0 (rank_lt_aleph0 K V)


/-- In a noetherian module over a division ring,
there exists a finite basis. This is the indexing `Finset`. -/
noncomputable def finsetBasisIndex [IsNoetherian K V] : Finset V :=
  (finite_basis_index (Basis.ofVectorSpace K V)).toFinset


@[simp]
theorem coe_finsetBasisIndex [IsNoetherian K V] :
    (↑(finsetBasisIndex K V) : Set V) = Basis.ofVectorSpaceIndex K V :=
  Set.Finite.coe_toFinset _


@[simp]
theorem coeSort_finsetBasisIndex [IsNoetherian K V] :
    (finsetBasisIndex K V : Type _) = Basis.ofVectorSpaceIndex K V :=
  Set.Finite.coeSort_toFinset _


/-- In a noetherian module over a division ring, there exists a finite basis.
This is indexed by the `Finset` `IsNoetherian.finsetBasisIndex`.
This is in contrast to the result `finite_basis_index (Basis.ofVectorSpace K V)`,
which provides a set and a `Set.Finite`.
-/
noncomputable def finsetBasis [IsNoetherian K V] : Basis (finsetBasisIndex K V) K V :=
                                        /-
                                          K : Type u
                                          V : Type v
                                          inst✝³ : DivisionRing K
                                          inst✝² : AddCommGroup V
                                          inst✝¹ : Module K V
                                          inst✝ : IsNoetherian K V
                                          ⊢ Equiv (↑(Basis.ofVectorSpaceIndex K V)) (Subtype fun x => Membership.mem (Is …
                                        -/
  (Basis.ofVectorSpace K V).reindex (by rw [coeSort_finsetBasisIndex])
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem range_finsetBasis [IsNoetherian K V] :
    Set.range (finsetBasis K V) = Basis.ofVectorSpaceIndex K V := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : IsNoetherian K V
    ⊢ Eq (Set.range ⇑(IsNoetherian.finsetBasis K V)) (Basis.ofVectorSpaceIndex K V)
  -/
  rw [finsetBasis, Basis.range_reindex, Basis.range_ofVectorSpace]
  /-
    🎉 no goals
  -/


/-- A module over a division ring is noetherian if and only if it is finitely generated. -/
theorem iff_fg : IsNoetherian K V ↔ Module.Finite K V := by
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ⊢ Iff (IsNoetherian K V) (Module.Finite K V)
  -/
  constructor
    /-
      case mp
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ⊢ IsNoetherian K V → Module.Finite K V
    -/
  · intro h
    exact
      ⟨⟨finsetBasisIndex K V, by
          convert (finsetBasis K V).span_eq
          simp⟩⟩
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ⊢ Module.Finite K V → IsNoetherian K V
    -/
  · rintro ⟨s, hs⟩
    /-
      case mpr.mk.intro
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      s : Finset V
      hs : Eq (Submodule.span K ↑s) Top.top
      ⊢ IsNoetherian K V
    -/
    rw [IsNoetherian.iff_rank_lt_aleph0, ← rank_top, ← hs]
    /-
      case mpr.mk.intro
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      s : Finset V
      hs : Eq (Submodule.span K ↑s) Top.top
      ⊢ LT.lt (Module.rank K (Subtype fun x => Membership.mem (Submodule.span K ↑s)  …
    -/
    exact lt_of_le_of_lt (rank_span_le _) s.finite_toSet.lt_aleph0
    /-
      🎉 no goals
    -/


