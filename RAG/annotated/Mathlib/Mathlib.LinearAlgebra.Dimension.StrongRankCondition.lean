/-- The dimension theorem: if `v` and `v'` are two bases, their index types
have the same cardinalities. -/
theorem mk_eq_mk_of_basis (v : Basis ι R M) (v' : Basis ι' R M) :
    Cardinal.lift.{w'} #ι = Cardinal.lift.{w} #ι' := by
  classical
  haveI := nontrivial_of_invariantBasisNumber R
  cases fintypeOrInfinite ι
  · -- `v` is a finite basis, so by `basis_finite_of_finite_spans` so is `v'`.
    -- haveI : Finite (range v) := Set.finite_range v
    haveI := basis_finite_of_finite_spans _ (Set.finite_range v) v.span_eq v'
    cases nonempty_fintype ι'
    -- We clean up a little:
    rw [Cardinal.mk_fintype, Cardinal.mk_fintype]
    simp only [Cardinal.lift_natCast, Nat.cast_inj]
    -- Now we can use invariant basis number to show they have the same cardinality.
    apply card_eq_of_linearEquiv R
    exact
      (Finsupp.linearEquivFunOnFinite R R ι).symm.trans v.repr.symm ≪≫ₗ v'.repr ≪≫ₗ
        Finsupp.linearEquivFunOnFinite R R ι'
  · -- `v` is an infinite basis,
    -- so by `infinite_basis_le_maximal_linearIndependent`, `v'` is at least as big,
    -- and then applying `infinite_basis_le_maximal_linearIndependent` again
    -- we see they have the same cardinality.
    have w₁ := infinite_basis_le_maximal_linearIndependent' v _ v'.linearIndependent v'.maximal
    rcases Cardinal.lift_mk_le'.mp w₁ with ⟨f⟩
    haveI : Infinite ι' := Infinite.of_injective f f.2
    have w₂ := infinite_basis_le_maximal_linearIndependent' v' _ v.linearIndependent v.maximal
    exact le_antisymm w₁ w₂


/-- Given two bases indexed by `ι` and `ι'` of an `R`-module, where `R` satisfies the invariant
basis number property, an equiv `ι ≃ ι'`. -/
def Basis.indexEquiv (v : Basis ι R M) (v' : Basis ι' R M) : ι ≃ ι' :=
  (Cardinal.lift_mk_eq'.1 <| mk_eq_mk_of_basis v v').some


theorem mk_eq_mk_of_basis' {ι' : Type w} (v : Basis ι R M) (v' : Basis ι' R M) : #ι = #ι' :=
  Cardinal.lift_inj.1 <| mk_eq_mk_of_basis v v'


/-- An auxiliary lemma for `Basis.le_span`.

If `R` satisfies the rank condition,
then for any finite basis `b : Basis ι R M`,
and any finite spanning set `w : Set M`,
the cardinality of `ι` is bounded by the cardinality of `w`.
-/
theorem Basis.le_span'' {ι : Type*} [Fintype ι] (b : Basis ι R M) {w : Set M} [Fintype w]
    (s : span R w = ⊤) : Fintype.card ι ≤ Fintype.card w := by
  -- We construct a surjective linear map `(w → R) →ₗ[R] (ι → R)`,
  -- by expressing a linear combination in `w` as a linear combination in `ι`.
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : RankCondition R
    ι : Type u_1
    inst✝¹ : Fintype ι
    b : Basis ι R M
    w : Set M
    inst✝ : Fintype ↑w
    s : Eq (Submodule.span R w) Top.top
    ⊢ LE.le (Fintype.card ι) (Fintype.card ↑w)
  -/
  fapply card_le_of_surjective' R
    /-
      case f
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : RankCondition R
      ι : Type u_1
      inst✝¹ : Fintype ι
      b : Basis ι R M
      w : Set M
      inst✝ : Fintype ↑w
      s : Eq (Submodule.span R w) Top.top
      ⊢ LinearMap (RingHom.id R) (Finsupp (↑w) R) (Finsupp ι R)
    -/
  · exact b.repr.toLinearMap.comp (Finsupp.linearCombination R (↑))
    /-
      🎉 no goals
    -/
    /-
      case i
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : RankCondition R
      ι : Type u_1
      inst✝¹ : Fintype ι
      b : Basis ι R M
      w : Set M
      inst✝ : Fintype ↑w
      s : Eq (Submodule.span R w) Top.top
      ⊢ Function.Surjective ⇑((↑b.repr).comp (Finsupp.linearCombination R Subtype.va …
    -/
  · apply Surjective.comp (g := b.repr.toLinearMap)
      /-
        case i.hg
        R : Type u
        M : Type v
        inst✝⁵ : Ring R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : RankCondition R
        ι : Type u_1
        inst✝¹ : Fintype ι
        b : Basis ι R M
        w : Set M
        inst✝ : Fintype ↑w
        s : Eq (Submodule.span R w) Top.top
        ⊢ Function.Surjective ⇑↑b.repr
      -/
    · apply LinearEquiv.surjective
      /-
        🎉 no goals
      -/
    /-
      case i.hf
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : RankCondition R
      ι : Type u_1
      inst✝¹ : Fintype ι
      b : Basis ι R M
      w : Set M
      inst✝ : Fintype ↑w
      s : Eq (Submodule.span R w) Top.top
      ⊢ Function.Surjective ⇑(Finsupp.linearCombination R Subtype.val)
    -/
    rw [← LinearMap.range_eq_top, Finsupp.range_linearCombination]
    /-
      case i.hf
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : RankCondition R
      ι : Type u_1
      inst✝¹ : Fintype ι
      b : Basis ι R M
      w : Set M
      inst✝ : Fintype ↑w
      s : Eq (Submodule.span R w) Top.top
      ⊢ Eq (Submodule.span R (Set.range Subtype.val)) Top.top
    -/
    simpa using s
    /-
      🎉 no goals
    -/


/--
Another auxiliary lemma for `Basis.le_span`, which does not require assuming the basis is finite,
but still assumes we have a finite spanning set.
-/
theorem basis_le_span' {ι : Type*} (b : Basis ι R M) {w : Set M} [Fintype w] (s : span R w = ⊤) :
    #ι ≤ Fintype.card w := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : RankCondition R
    ι : Type u_1
    b : Basis ι R M
    w : Set M
    inst✝ : Fintype ↑w
    s : Eq (Submodule.span R w) Top.top
    ⊢ LE.le (Cardinal.mk ι) ↑(Fintype.card ↑w)
  -/
  haveI := nontrivial_of_invariantBasisNumber R
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : RankCondition R
    ι : Type u_1
    b : Basis ι R M
    w : Set M
    inst✝ : Fintype ↑w
    s : Eq (Submodule.span R w) Top.top
    this : Nontrivial R
    ⊢ LE.le (Cardinal.mk ι) ↑(Fintype.card ↑w)
  -/
  haveI := basis_finite_of_finite_spans w (toFinite _) s b
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : RankCondition R
    ι : Type u_1
    b : Basis ι R M
    w : Set M
    inst✝ : Fintype ↑w
    s : Eq (Submodule.span R w) Top.top
    this✝ : Nontrivial R
    this : Finite ι
    ⊢ LE.le (Cardinal.mk ι) ↑(Fintype.card ↑w)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : RankCondition R
    ι : Type u_1
    b : Basis ι R M
    w : Set M
    inst✝ : Fintype ↑w
    s : Eq (Submodule.span R w) Top.top
    this✝ : Nontrivial R
    this : Finite ι
    val✝ : Fintype ι
    ⊢ LE.le (Cardinal.mk ι) ↑(Fintype.card ↑w)
  -/
  rw [Cardinal.mk_fintype ι]
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : RankCondition R
    ι : Type u_1
    b : Basis ι R M
    w : Set M
    inst✝ : Fintype ↑w
    s : Eq (Submodule.span R w) Top.top
    this✝ : Nontrivial R
    this : Finite ι
    val✝ : Fintype ι
    ⊢ LE.le ↑(Fintype.card ι) ↑(Fintype.card ↑w)
  -/
  simp only [Nat.cast_le]
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : RankCondition R
    ι : Type u_1
    b : Basis ι R M
    w : Set M
    inst✝ : Fintype ↑w
    s : Eq (Submodule.span R w) Top.top
    this✝ : Nontrivial R
    this : Finite ι
    val✝ : Fintype ι
    ⊢ LE.le (Fintype.card ι) (Fintype.card ↑w)
  -/
  exact Basis.le_span'' b s
  /-
    🎉 no goals
  -/

-- Note that if `R` satisfies the strong rank condition,
-- this also follows from `linearIndependent_le_span` below.

/-- If `R` satisfies the rank condition,
then the cardinality of any basis is bounded by the cardinality of any spanning set.
-/
theorem Basis.le_span {J : Set M} (v : Basis ι R M) (hJ : span R J = ⊤) : #(range v) ≤ #J := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type w
    inst✝ : RankCondition R
    J : Set M
    v : Basis ι R M
    hJ : Eq (Submodule.span R J) Top.top
    ⊢ LE.le (Cardinal.mk ↑(Set.range ⇑v)) (Cardinal.mk ↑J)
  -/
  haveI := nontrivial_of_invariantBasisNumber R
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type w
    inst✝ : RankCondition R
    J : Set M
    v : Basis ι R M
    hJ : Eq (Submodule.span R J) Top.top
    this : Nontrivial R
    ⊢ LE.le (Cardinal.mk ↑(Set.range ⇑v)) (Cardinal.mk ↑J)
  -/
  cases fintypeOrInfinite J
    /-
      case inl
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type w
      inst✝ : RankCondition R
      J : Set M
      v : Basis ι R M
      hJ : Eq (Submodule.span R J) Top.top
      this : Nontrivial R
      val✝ : Fintype ↑J
      ⊢ LE.le (Cardinal.mk ↑(Set.range ⇑v)) (Cardinal.mk ↑J)
    -/
  · rw [← Cardinal.lift_le, Cardinal.mk_range_eq_of_injective v.injective, Cardinal.mk_fintype J]
    /-
      case inl
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type w
      inst✝ : RankCondition R
      J : Set M
      v : Basis ι R M
      hJ : Eq (Submodule.span R J) Top.top
      this : Nontrivial R
      val✝ : Fintype ↑J
      ⊢ LE.le (Cardinal.lift.{v, w} (Cardinal.mk ι)) (Cardinal.lift.{w, v} ↑(Fintype …
    -/
    convert Cardinal.lift_le.{v}.2 (basis_le_span' v hJ)
    /-
      case h.e'_4
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type w
      inst✝ : RankCondition R
      J : Set M
      v : Basis ι R M
      hJ : Eq (Submodule.span R J) Top.top
      this : Nontrivial R
      val✝ : Fintype ↑J
      ⊢ Eq (Cardinal.lift.{w, v} ↑(Fintype.card ↑J)) (Cardinal.lift.{v, w} ↑(Fintype …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type w
      inst✝ : RankCondition R
      J : Set M
      v : Basis ι R M
      hJ : Eq (Submodule.span R J) Top.top
      this : Nontrivial R
      val✝ : Infinite ↑J
      ⊢ LE.le (Cardinal.mk ↑(Set.range ⇑v)) (Cardinal.mk ↑J)
    -/
  · let S : J → Set ι := fun j => ↑(v.repr j).support
    /-
      case inr
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type w
      inst✝ : RankCondition R
      J : Set M
      v : Basis ι R M
      hJ : Eq (Submodule.span R J) Top.top
      this : Nontrivial R
      val✝ : Infinite ↑J
      S : ↑J → Set ι := fun j => ↑(v.repr ↑j).support
      ⊢ LE.le (Cardinal.mk ↑(Set.range ⇑v)) (Cardinal.mk ↑J)
    -/
    let S' : J → Set M := fun j => v '' S j
    have hs : range v ⊆ ⋃ j, S' j := by
      intro b hb
      rcases mem_range.1 hb with ⟨i, hi⟩
      have : span R J ≤ comap v.repr.toLinearMap (Finsupp.supported R R (⋃ j, S j)) :=
        span_le.2 fun j hj x hx => ⟨_, ⟨⟨j, hj⟩, rfl⟩, hx⟩
      rw [hJ] at this
      replace : v.repr (v i) ∈ Finsupp.supported R R (⋃ j, S j) := this trivial
      rw [v.repr_self, Finsupp.mem_supported, Finsupp.support_single_ne_zero _ one_ne_zero] at this
      · subst b
        rcases mem_iUnion.1 (this (Finset.mem_singleton_self _)) with ⟨j, hj⟩
        exact mem_iUnion.2 ⟨j, (mem_image _ _ _).2 ⟨i, hj, rfl⟩⟩
    /-
      case inr
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type w
      inst✝ : RankCondition R
      J : Set M
      v : Basis ι R M
      hJ : Eq (Submodule.span R J) Top.top
      this : Nontrivial R
      val✝ : Infinite ↑J
      S : ↑J → Set ι := fun j => ↑(v.repr ↑j).support
      S' : ↑J → Set M := fun j => Set.image (⇑v) (S j)
      hs : HasSubset.Subset (Set.range ⇑v) (Set.iUnion fun j => S' j)
      ⊢ LE.le (Cardinal.mk ↑(Set.range ⇑v)) (Cardinal.mk ↑J)
    -/
    refine le_of_not_lt fun IJ => ?_
    /-
      case inr
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type w
      inst✝ : RankCondition R
      J : Set M
      v : Basis ι R M
      hJ : Eq (Submodule.span R J) Top.top
      this : Nontrivial R
      val✝ : Infinite ↑J
      S : ↑J → Set ι := fun j => ↑(v.repr ↑j).support
      S' : ↑J → Set M := fun j => Set.image (⇑v) (S j)
      hs : HasSubset.Subset (Set.range ⇑v) (Set.iUnion fun j => S' j)
      IJ : LT.lt (Cardinal.mk ↑J) (Cardinal.mk ↑(Set.range ⇑v))
      ⊢ False
    -/
    suffices #(⋃ j, S' j) < #(range v) by exact not_le_of_lt this ⟨Set.embeddingOfSubset _ _ hs⟩
    refine lt_of_le_of_lt (le_trans Cardinal.mk_iUnion_le_sum_mk
      (Cardinal.sum_le_sum _ (fun _ => ℵ₀) ?_)) ?_
      /-
        case inr.refine_1
        R : Type u
        M : Type v
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        ι : Type w
        inst✝ : RankCondition R
        J : Set M
        v : Basis ι R M
        hJ : Eq (Submodule.span R J) Top.top
        this : Nontrivial R
        val✝ : Infinite ↑J
        S : ↑J → Set ι := fun j => ↑(v.repr ↑j).support
        S' : ↑J → Set M := fun j => Set.image (⇑v) (S j)
        hs : HasSubset.Subset (Set.range ⇑v) (Set.iUnion fun j => S' j)
        IJ : LT.lt (Cardinal.mk ↑J) (Cardinal.mk ↑(Set.range ⇑v))
        ⊢ ∀ (i : ↑J), LE.le (Cardinal.mk ↑(S' i)) ((fun x => Cardinal.aleph0) i)
      -/
    · exact fun j => (Cardinal.lt_aleph0_of_finite _).le
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_2
        R : Type u
        M : Type v
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        ι : Type w
        inst✝ : RankCondition R
        J : Set M
        v : Basis ι R M
        hJ : Eq (Submodule.span R J) Top.top
        this : Nontrivial R
        val✝ : Infinite ↑J
        S : ↑J → Set ι := fun j => ↑(v.repr ↑j).support
        S' : ↑J → Set M := fun j => Set.image (⇑v) (S j)
        hs : HasSubset.Subset (Set.range ⇑v) (Set.iUnion fun j => S' j)
        IJ : LT.lt (Cardinal.mk ↑J) (Cardinal.mk ↑(Set.range ⇑v))
        ⊢ LT.lt (Cardinal.sum fun x => Cardinal.aleph0) (Cardinal.mk ↑(Set.range ⇑v))
      -/
    · simpa
      /-
        🎉 no goals
      -/


theorem linearIndependent_le_span_aux' {ι : Type*} [Fintype ι] (v : ι → M)
    (i : LinearIndependent R v) (w : Set M) [Fintype w] (s : range v ≤ span R w) :
    Fintype.card ι ≤ Fintype.card w := by
  -- We construct an injective linear map `(ι → R) →ₗ[R] (w → R)`,
  -- by thinking of `f : ι → R` as a linear combination of the finite family `v`,
  -- and expressing that (using the axiom of choice) as a linear combination over `w`.
  -- We can do this linearly by constructing the map on a basis.
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    ι : Type u_1
    inst✝¹ : Fintype ι
    v : ι → M
    i : LinearIndependent R v
    w : Set M
    inst✝ : Fintype ↑w
    s : LE.le (Set.range v) ↑(Submodule.span R w)
    ⊢ LE.le (Fintype.card ι) (Fintype.card ↑w)
  -/
  fapply card_le_of_injective' R
    /-
      case f
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : StrongRankCondition R
      ι : Type u_1
      inst✝¹ : Fintype ι
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      inst✝ : Fintype ↑w
      s : LE.le (Set.range v) ↑(Submodule.span R w)
      ⊢ LinearMap (RingHom.id R) (Finsupp ι R) (Finsupp (↑w) R)
    -/
  · apply Finsupp.linearCombination
    /-
      case f.v
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : StrongRankCondition R
      ι : Type u_1
      inst✝¹ : Fintype ι
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      inst✝ : Fintype ↑w
      s : LE.le (Set.range v) ↑(Submodule.span R w)
      ⊢ ι → Finsupp (↑w) R
    -/
    exact fun i => Span.repr R w ⟨v i, s (mem_range_self i)⟩
    /-
      🎉 no goals
    -/
    /-
      case i
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : StrongRankCondition R
      ι : Type u_1
      inst✝¹ : Fintype ι
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      inst✝ : Fintype ↑w
      s : LE.le (Set.range v) ↑(Submodule.span R w)
      ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => Span.repr R w ⟨v i …
    -/
  · intro f g h
    /-
      case i
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : StrongRankCondition R
      ι : Type u_1
      inst✝¹ : Fintype ι
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      inst✝ : Fintype ↑w
      s : LE.le (Set.range v) ↑(Submodule.span R w)
      f g : Finsupp ι R
      h : Eq ((Finsupp.linearCombination R fun i => Span.repr R w ⟨v i, ⋯⟩) f) ((Fin …
      ⊢ Eq f g
    -/
    apply_fun linearCombination R ((↑) : w → M) at h
    simp only [linearCombination_linearCombination, Submodule.coe_mk,
               Span.finsupp_linearCombination_repr] at h
    /-
      case i
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : StrongRankCondition R
      ι : Type u_1
      inst✝¹ : Fintype ι
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      inst✝ : Fintype ↑w
      s : LE.le (Set.range v) ↑(Submodule.span R w)
      f g : Finsupp ι R
      h : Eq ((Finsupp.linearCombination R fun b => v b) f) ((Finsupp.linearCombinat …
      ⊢ Eq f g
    -/
    rw [← sub_eq_zero, ← LinearMap.map_sub] at h
    /-
      case i
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : StrongRankCondition R
      ι : Type u_1
      inst✝¹ : Fintype ι
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      inst✝ : Fintype ↑w
      s : LE.le (Set.range v) ↑(Submodule.span R w)
      f g : Finsupp ι R
      h : Eq ((Finsupp.linearCombination R fun b => v b) (HSub.hSub f g)) 0
      ⊢ Eq f g
    -/
    exact sub_eq_zero.mp (linearIndependent_iff.mp i _ h)
    /-
      🎉 no goals
    -/


/-- If `R` satisfies the strong rank condition,
then any linearly independent family `v : ι → M`
contained in the span of some finite `w : Set M`,
is itself finite.
-/
lemma LinearIndependent.finite_of_le_span_finite {ι : Type*} (v : ι → M) (i : LinearIndependent R v)
    (w : Set M) [Finite w] (s : range v ≤ span R w) : Finite ι :=
  letI := Fintype.ofFinite w
  Fintype.finite <| fintypeOfFinsetCardLe (Fintype.card w) fun t => by
    /-
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : StrongRankCondition R
      ι : Type u_1
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      inst✝ : Finite ↑w
      s : LE.le (Set.range v) ↑(Submodule.span R w)
      this : Fintype ↑w := Fintype.ofFinite ↑w
      t : Finset ι
      ⊢ LE.le t.card (Fintype.card ↑w)
    -/
    let v' := fun x : (t : Set ι) => v x
    /-
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : StrongRankCondition R
      ι : Type u_1
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      inst✝ : Finite ↑w
      s : LE.le (Set.range v) ↑(Submodule.span R w)
      this : Fintype ↑w := Fintype.ofFinite ↑w
      t : Finset ι
      v' : ↑↑t → M := fun x => v ↑x
      ⊢ LE.le t.card (Fintype.card ↑w)
    -/
    have i' : LinearIndependent R v' := i.comp _ Subtype.val_injective
    /-
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : StrongRankCondition R
      ι : Type u_1
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      inst✝ : Finite ↑w
      s : LE.le (Set.range v) ↑(Submodule.span R w)
      this : Fintype ↑w := Fintype.ofFinite ↑w
      t : Finset ι
      v' : ↑↑t → M := fun x => v ↑x
      i' : LinearIndependent R v'
      ⊢ LE.le t.card (Fintype.card ↑w)
    -/
    have s' : range v' ≤ span R w := (range_comp_subset_range _ _).trans s
    /-
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : StrongRankCondition R
      ι : Type u_1
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      inst✝ : Finite ↑w
      s : LE.le (Set.range v) ↑(Submodule.span R w)
      this : Fintype ↑w := Fintype.ofFinite ↑w
      t : Finset ι
      v' : ↑↑t → M := fun x => v ↑x
      i' : LinearIndependent R v'
      s' : LE.le (Set.range v') ↑(Submodule.span R w)
      ⊢ LE.le t.card (Fintype.card ↑w)
    -/
    simpa using linearIndependent_le_span_aux' v' i' w s'
    /-
      🎉 no goals
    -/


/-- If `R` satisfies the strong rank condition,
then for any linearly independent family `v : ι → M`
contained in the span of some finite `w : Set M`,
the cardinality of `ι` is bounded by the cardinality of `w`.
-/
theorem linearIndependent_le_span' {ι : Type*} (v : ι → M) (i : LinearIndependent R v) (w : Set M)
    [Fintype w] (s : range v ≤ span R w) : #ι ≤ Fintype.card w := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type u_1
    v : ι → M
    i : LinearIndependent R v
    w : Set M
    inst✝ : Fintype ↑w
    s : LE.le (Set.range v) ↑(Submodule.span R w)
    ⊢ LE.le (Cardinal.mk ι) ↑(Fintype.card ↑w)
  -/
  haveI : Finite ι := i.finite_of_le_span_finite v w s
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type u_1
    v : ι → M
    i : LinearIndependent R v
    w : Set M
    inst✝ : Fintype ↑w
    s : LE.le (Set.range v) ↑(Submodule.span R w)
    this : Finite ι
    ⊢ LE.le (Cardinal.mk ι) ↑(Fintype.card ↑w)
  -/
  letI := Fintype.ofFinite ι
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type u_1
    v : ι → M
    i : LinearIndependent R v
    w : Set M
    inst✝ : Fintype ↑w
    s : LE.le (Set.range v) ↑(Submodule.span R w)
    this✝ : Finite ι
    this : Fintype ι := Fintype.ofFinite ι
    ⊢ LE.le (Cardinal.mk ι) ↑(Fintype.card ↑w)
  -/
  rw [Cardinal.mk_fintype]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type u_1
    v : ι → M
    i : LinearIndependent R v
    w : Set M
    inst✝ : Fintype ↑w
    s : LE.le (Set.range v) ↑(Submodule.span R w)
    this✝ : Finite ι
    this : Fintype ι := Fintype.ofFinite ι
    ⊢ LE.le ↑(Fintype.card ι) ↑(Fintype.card ↑w)
  -/
  simp only [Nat.cast_le]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type u_1
    v : ι → M
    i : LinearIndependent R v
    w : Set M
    inst✝ : Fintype ↑w
    s : LE.le (Set.range v) ↑(Submodule.span R w)
    this✝ : Finite ι
    this : Fintype ι := Fintype.ofFinite ι
    ⊢ LE.le (Fintype.card ι) (Fintype.card ↑w)
  -/
  exact linearIndependent_le_span_aux' v i w s
  /-
    🎉 no goals
  -/


/-- If `R` satisfies the strong rank condition,
then for any linearly independent family `v : ι → M`
and any finite spanning set `w : Set M`,
the cardinality of `ι` is bounded by the cardinality of `w`.
-/
theorem linearIndependent_le_span {ι : Type*} (v : ι → M) (i : LinearIndependent R v) (w : Set M)
    [Fintype w] (s : span R w = ⊤) : #ι ≤ Fintype.card w := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type u_1
    v : ι → M
    i : LinearIndependent R v
    w : Set M
    inst✝ : Fintype ↑w
    s : Eq (Submodule.span R w) Top.top
    ⊢ LE.le (Cardinal.mk ι) ↑(Fintype.card ↑w)
  -/
  apply linearIndependent_le_span' v i w
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type u_1
    v : ι → M
    i : LinearIndependent R v
    w : Set M
    inst✝ : Fintype ↑w
    s : Eq (Submodule.span R w) Top.top
    ⊢ LE.le (Set.range v) ↑(Submodule.span R w)
  -/
  rw [s]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type u_1
    v : ι → M
    i : LinearIndependent R v
    w : Set M
    inst✝ : Fintype ↑w
    s : Eq (Submodule.span R w) Top.top
    ⊢ LE.le (Set.range v) ↑Top.top
  -/
  exact le_top
  /-
    🎉 no goals
  -/


/-- A version of `linearIndependent_le_span` for `Finset`. -/
theorem linearIndependent_le_span_finset {ι : Type*} (v : ι → M) (i : LinearIndependent R v)
    (w : Finset M) (s : span R (w : Set M) = ⊤) : #ι ≤ w.card := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    ι : Type u_1
    v : ι → M
    i : LinearIndependent R v
    w : Finset M
    s : Eq (Submodule.span R ↑w) Top.top
    ⊢ LE.le (Cardinal.mk ι) ↑w.card
  -/
  simpa only [Finset.coe_sort_coe, Fintype.card_coe] using linearIndependent_le_span v i w s
  /-
    🎉 no goals
  -/


/-- An auxiliary lemma for `linearIndependent_le_basis`:
we handle the case where the basis `b` is infinite.
-/
theorem linearIndependent_le_infinite_basis {ι : Type w} (b : Basis ι R M) [Infinite ι] {κ : Type w}
    (v : κ → M) (i : LinearIndependent R v) : #κ ≤ #ι := by
  classical
  by_contra h
  rw [not_le, ← Cardinal.mk_finset_of_infinite ι] at h
  let Φ := fun k : κ => (b.repr (v k)).support
  obtain ⟨s, w : Infinite ↑(Φ ⁻¹' {s})⟩ := Cardinal.exists_infinite_fiber Φ h (by infer_instance)
  let v' := fun k : Φ ⁻¹' {s} => v k
  have i' : LinearIndependent R v' := i.comp _ Subtype.val_injective
  have w' : Finite (Φ ⁻¹' {s}) := by
    apply i'.finite_of_le_span_finite v' (s.image b)
    rintro m ⟨⟨p, ⟨rfl⟩⟩, rfl⟩
    simp only [SetLike.mem_coe, Subtype.coe_mk, Finset.coe_image]
    apply Basis.mem_span_repr_support
  exact w.false


/-- Over any ring `R` satisfying the strong rank condition,
if `b` is a basis for a module `M`,
and `s` is a linearly independent set,
then the cardinality of `s` is bounded by the cardinality of `b`.
-/
theorem linearIndependent_le_basis {ι : Type w} (b : Basis ι R M) {κ : Type w} (v : κ → M)
    (i : LinearIndependent R v) : #κ ≤ #ι := by
  classical
  -- We split into cases depending on whether `ι` is infinite.
  cases fintypeOrInfinite ι
  · rw [Cardinal.mk_fintype ι] -- When `ι` is finite, we have `linearIndependent_le_span`,
    haveI : Nontrivial R := nontrivial_of_invariantBasisNumber R
    rw [Fintype.card_congr (Equiv.ofInjective b b.injective)]
    exact linearIndependent_le_span v i (range b) b.span_eq
  · -- and otherwise we have `linearIndependent_le_infinite_basis`.
    exact linearIndependent_le_infinite_basis b v i


/-- Let `R` satisfy the strong rank condition. If `m` elements of a free rank `n` `R`-module are
linearly independent, then `m ≤ n`. -/
theorem Basis.card_le_card_of_linearIndependent_aux {R : Type*} [Ring R] [StrongRankCondition R]
    (n : ℕ) {m : ℕ} (v : Fin m → Fin n → R) : LinearIndependent R v → m ≤ n := fun h => by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : StrongRankCondition R
    n m : Nat
    v : Fin m → Fin n → R
    h : LinearIndependent R v
    ⊢ LE.le m n
  -/
  simpa using linearIndependent_le_basis (Pi.basisFun R (Fin n)) v h
  /-
    🎉 no goals
  -/

-- When the basis is not infinite this need not be true!

/-- Over any ring `R` satisfying the strong rank condition,
if `b` is an infinite basis for a module `M`,
then every maximal linearly independent set has the same cardinality as `b`.

This proof (along with some of the lemmas above) comes from
[Les familles libres maximales d'un module ont-elles le meme cardinal?][lazarus1973]
-/
theorem maximal_linearIndependent_eq_infinite_basis {ι : Type w} (b : Basis ι R M) [Infinite ι]
    {κ : Type w} (v : κ → M) (i : LinearIndependent R v) (m : i.Maximal) : #κ = #ι := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type w
    b : Basis ι R M
    inst✝ : Infinite ι
    κ : Type w
    v : κ → M
    i : LinearIndependent R v
    m : i.Maximal
    ⊢ Eq (Cardinal.mk κ) (Cardinal.mk ι)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : StrongRankCondition R
      ι : Type w
      b : Basis ι R M
      inst✝ : Infinite ι
      κ : Type w
      v : κ → M
      i : LinearIndependent R v
      m : i.Maximal
      ⊢ LE.le (Cardinal.mk κ) (Cardinal.mk ι)
    -/
  · exact linearIndependent_le_basis b v i
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : StrongRankCondition R
      ι : Type w
      b : Basis ι R M
      inst✝ : Infinite ι
      κ : Type w
      v : κ → M
      i : LinearIndependent R v
      m : i.Maximal
      ⊢ LE.le (Cardinal.mk ι) (Cardinal.mk κ)
    -/
  · haveI : Nontrivial R := nontrivial_of_invariantBasisNumber R
    /-
      case a
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : StrongRankCondition R
      ι : Type w
      b : Basis ι R M
      inst✝ : Infinite ι
      κ : Type w
      v : κ → M
      i : LinearIndependent R v
      m : i.Maximal
      this : Nontrivial R
      ⊢ LE.le (Cardinal.mk ι) (Cardinal.mk κ)
    -/
    exact infinite_basis_le_maximal_linearIndependent b v i m
    /-
      🎉 no goals
    -/


theorem Basis.mk_eq_rank'' {ι : Type v} (v : Basis ι R M) : #ι = Module.rank R M := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    ι : Type v
    v : Basis ι R M
    ⊢ Eq (Cardinal.mk ι) (Module.rank R M)
  -/
  haveI := nontrivial_of_invariantBasisNumber R
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    ι : Type v
    v : Basis ι R M
    this : Nontrivial R
    ⊢ Eq (Cardinal.mk ι) (Module.rank R M)
  -/
  rw [Module.rank_def]
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    ι : Type v
    v : Basis ι R M
    this : Nontrivial R
    ⊢ Eq (Cardinal.mk ι) (iSup fun ι => Cardinal.mk ↑↑ι)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : StrongRankCondition R
      ι : Type v
      v : Basis ι R M
      this : Nontrivial R
      ⊢ LE.le (Cardinal.mk ι) (iSup fun ι => Cardinal.mk ↑↑ι)
    -/
  · trans
    /-
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : StrongRankCondition R
      ι : Type v
      v : Basis ι R M
      this : Nontrivial R
      ⊢ LE.le (Cardinal.mk ι) ?m.96238
    -/
    swap
      /-
        R : Type u
        M : Type v
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : StrongRankCondition R
        ι : Type v
        v : Basis ι R M
        this : Nontrivial R
        ⊢ LE.le ?m.96238 (iSup fun ι => Cardinal.mk ↑↑ι)
      -/
    · apply le_ciSup (Cardinal.bddAbove_range _)
      exact
        ⟨Set.range v, by
          convert v.reindexRange.linearIndependent
          ext
          simp⟩
      /-
        R : Type u
        M : Type v
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : StrongRankCondition R
        ι : Type v
        v : Basis ι R M
        this : Nontrivial R
        ⊢ LE.le (Cardinal.mk ι) (Cardinal.mk ↑↑⟨Set.range ⇑v, ⋯⟩)
      -/
    · exact (Cardinal.mk_range_eq v v.injective).ge
      /-
        🎉 no goals
      -/
    /-
      case a
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : StrongRankCondition R
      ι : Type v
      v : Basis ι R M
      this : Nontrivial R
      ⊢ LE.le (iSup fun ι => Cardinal.mk ↑↑ι) (Cardinal.mk ι)
    -/
  · apply ciSup_le'
    /-
      case a.h
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : StrongRankCondition R
      ι : Type v
      v : Basis ι R M
      this : Nontrivial R
      ⊢ ∀ (i : Subtype fun s => LinearIndependent R Subtype.val), LE.le (Cardinal.mk …
    -/
    rintro ⟨s, li⟩
    /-
      case a.h.mk
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : StrongRankCondition R
      ι : Type v
      v : Basis ι R M
      this : Nontrivial R
      s : Set M
      li : LinearIndependent R Subtype.val
      ⊢ LE.le (Cardinal.mk ↑↑⟨s, li⟩) (Cardinal.mk ι)
    -/
    apply linearIndependent_le_basis v _ li
    /-
      🎉 no goals
    -/


theorem Basis.mk_range_eq_rank (v : Basis ι R M) : #(range v) = Module.rank R M :=
  v.reindexRange.mk_eq_rank''


/-- If a vector space has a finite basis, then its dimension (seen as a cardinal) is equal to the
cardinality of the basis. -/
theorem rank_eq_card_basis {ι : Type w} [Fintype ι] (h : Basis ι R M) :
    Module.rank R M = Fintype.card ι := by
  classical
  haveI := nontrivial_of_invariantBasisNumber R
  rw [← h.mk_range_eq_rank, Cardinal.mk_fintype, Set.card_range_of_injective h.injective]


theorem Basis.card_le_card_of_linearIndependent {ι : Type*} [Fintype ι] (b : Basis ι R M)
    {ι' : Type*} [Fintype ι'] {v : ι' → M} (hv : LinearIndependent R v) :
    Fintype.card ι' ≤ Fintype.card ι := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    ι : Type u_1
    inst✝¹ : Fintype ι
    b : Basis ι R M
    ι' : Type u_2
    inst✝ : Fintype ι'
    v : ι' → M
    hv : LinearIndependent R v
    ⊢ LE.le (Fintype.card ι') (Fintype.card ι)
  -/
  letI := nontrivial_of_invariantBasisNumber R
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    ι : Type u_1
    inst✝¹ : Fintype ι
    b : Basis ι R M
    ι' : Type u_2
    inst✝ : Fintype ι'
    v : ι' → M
    hv : LinearIndependent R v
    this : Nontrivial R := nontrivial_of_invariantBasisNumber R
    ⊢ LE.le (Fintype.card ι') (Fintype.card ι)
  -/
  simpa [rank_eq_card_basis b, Cardinal.mk_fintype] using hv.cardinal_lift_le_rank
  /-
    🎉 no goals
  -/


theorem Basis.card_le_card_of_submodule (N : Submodule R M) [Fintype ι] (b : Basis ι R M)
    [Fintype ι'] (b' : Basis ι' R N) : Fintype.card ι' ≤ Fintype.card ι :=
  b.card_le_card_of_linearIndependent (b'.linearIndependent.map' N.subtype N.ker_subtype)


theorem Basis.card_le_card_of_le {N O : Submodule R M} (hNO : N ≤ O) [Fintype ι] (b : Basis ι R O)
    [Fintype ι'] (b' : Basis ι' R N) : Fintype.card ι' ≤ Fintype.card ι :=
  b.card_le_card_of_linearIndependent
    (b'.linearIndependent.map' (Submodule.inclusion hNO) (N.ker_inclusion O _))


theorem Basis.mk_eq_rank (v : Basis ι R M) :
    Cardinal.lift.{v} #ι = Cardinal.lift.{w} (Module.rank R M) := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type w
    inst✝ : StrongRankCondition R
    v : Basis ι R M
    ⊢ Eq (Cardinal.lift.{v, w} (Cardinal.mk ι)) (Cardinal.lift.{w, v} (Module.rank …
  -/
  haveI := nontrivial_of_invariantBasisNumber R
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type w
    inst✝ : StrongRankCondition R
    v : Basis ι R M
    this : Nontrivial R
    ⊢ Eq (Cardinal.lift.{v, w} (Cardinal.mk ι)) (Cardinal.lift.{w, v} (Module.rank …
  -/
  rw [← v.mk_range_eq_rank, Cardinal.mk_range_eq_of_injective v.injective]
  /-
    🎉 no goals
  -/


theorem Basis.mk_eq_rank'.{m} (v : Basis ι R M) :
    Cardinal.lift.{max v m} #ι = Cardinal.lift.{max w m} (Module.rank R M) :=
  Cardinal.lift_umax_eq.{w, v, m}.mpr v.mk_eq_rank


theorem rank_span {v : ι → M} (hv : LinearIndependent R v) :
    Module.rank R ↑(span R (range v)) = #(range v) := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type w
    inst✝ : StrongRankCondition R
    v : ι → M
    hv : LinearIndependent R v
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R (Set.ra …
  -/
  haveI := nontrivial_of_invariantBasisNumber R
  rw [← Cardinal.lift_inj, ← (Basis.span hv).mk_eq_rank,
    Cardinal.mk_range_eq_of_injective (@LinearIndependent.injective ι R M v _ _ _ _ hv)]


theorem rank_span_set {s : Set M} (hs : LinearIndependent R (fun x => x : s → M)) :
    Module.rank R ↑(span R s) = #s := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    s : Set M
    hs : LinearIndependent R fun x => ↑x
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R s) x))  …
  -/
  rw [← @setOf_mem_eq _ s, ← Subtype.range_coe_subtype]
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    s : Set M
    hs : LinearIndependent R fun x => ↑x
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R (Set.ra …
  -/
  exact rank_span hs
  /-
    🎉 no goals
  -/


/-- An induction (and recursion) principle for proving results about all submodules of a fixed
finite free module `M`. A property is true for all submodules of `M` if it satisfies the following
"inductive step": the property is true for a submodule `N` if it's true for all submodules `N'`
of `N` with the property that there exists `0 ≠ x ∈ N` such that the sum `N' + Rx` is direct. -/
def Submodule.inductionOnRank [IsDomain R] [Finite ι] (b : Basis ι R M)
    (P : Submodule R M → Sort*) (ih : ∀ N : Submodule R M,
    (∀ N' ≤ N, ∀ x ∈ N, (∀ (c : R), ∀ y ∈ N', c • x + y = (0 : M) → c = 0) → P N') → P N)
    (N : Submodule R M) : P N :=
  letI := Fintype.ofFinite ι
  Submodule.inductionOnRankAux b P ih (Fintype.card ι) N fun hs hli => by
    /-
      R : Type u
      M : Type v
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type w
      ι' : Type w'
      inst✝² : StrongRankCondition R
      inst✝¹ : IsDomain R
      inst✝ : Finite ι
      b : Basis ι R M
      P : Submodule R M → Sort u_1
      ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
      N : Submodule R M
      this : Fintype ι := Fintype.ofFinite ι
      m✝ : Nat
      hs : Fin m✝ → Subtype fun x => Membership.mem N x
      hli : LinearIndependent R (Function.comp Subtype.val hs)
      ⊢ LE.le m✝ (Fintype.card ι)
    -/
    simpa using b.card_le_card_of_linearIndependent hli
    /-
      🎉 no goals
    -/


/-- If `S` a module-finite free `R`-algebra, then the `R`-rank of a nonzero `R`-free
ideal `I` of `S` is the same as the rank of `S`. -/
theorem Ideal.rank_eq {R S : Type*} [CommRing R] [StrongRankCondition R] [Ring S] [IsDomain S]
    [Algebra R S] {n m : Type*} [Fintype n] [Fintype m] (b : Basis n R S) {I : Ideal S}
    (hI : I ≠ ⊥) (c : Basis m R I) : Fintype.card m = Fintype.card n := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : StrongRankCondition R
    inst✝⁴ : Ring S
    inst✝³ : IsDomain S
    inst✝² : Algebra R S
    n : Type u_3
    m : Type u_4
    inst✝¹ : Fintype n
    inst✝ : Fintype m
    b : Basis n R S
    I : Ideal S
    hI : Ne I Bot.bot
    c : Basis m R (Subtype fun x => Membership.mem I x)
    ⊢ Eq (Fintype.card m) (Fintype.card n)
  -/
  obtain ⟨a, ha⟩ := Submodule.nonzero_mem_of_bot_lt (bot_lt_iff_ne_bot.mpr hI)
  have : LinearIndependent R fun i => b i • a := by
    have hb := b.linearIndependent
    rw [Fintype.linearIndependent_iff] at hb ⊢
    intro g hg
    apply hb g
    simp only [← smul_assoc, ← Finset.sum_smul, smul_eq_zero] at hg
    exact hg.resolve_right ha
  exact le_antisymm
    (b.card_le_card_of_linearIndependent (c.linearIndependent.map' (Submodule.subtype I)
      ((LinearMap.ker_eq_bot (f := (Submodule.subtype I : I →ₗ[R] S))).mpr Subtype.coe_injective)))
    (c.card_le_card_of_linearIndependent this)


theorem finrank_eq_nat_card_basis (h : Basis ι R M) :
    finrank R M = Nat.card ι := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type w
    inst✝ : StrongRankCondition R
    h : Basis ι R M
    ⊢ Eq (Module.finrank R M) (Nat.card ι)
  -/
  rw [Nat.card, ← toNat_lift.{v}, h.mk_eq_rank, toNat_lift, finrank]
  /-
    🎉 no goals
  -/


/-- If a vector space (or module) has a finite basis, then its dimension (or rank) is equal to the
cardinality of the basis. -/
theorem finrank_eq_card_basis {ι : Type w} [Fintype ι] (h : Basis ι R M) :
    finrank R M = Fintype.card ι :=
  finrank_eq_of_rank_eq (rank_eq_card_basis h)


/-- If a free module is of finite rank, then the cardinality of any basis is equal to its
`finrank`. -/
theorem mk_finrank_eq_card_basis [Module.Finite R M] {ι : Type w} (h : Basis ι R M) :
    (finrank R M : Cardinal.{w}) = #ι := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ι : Type w
    h : Basis ι R M
    ⊢ Eq (↑(Module.finrank R M)) (Cardinal.mk ι)
  -/
  cases @nonempty_fintype _ (Module.Finite.finite_basis h)
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ι : Type w
    h : Basis ι R M
    val✝ : Fintype ι
    ⊢ Eq (↑(Module.finrank R M)) (Cardinal.mk ι)
  -/
  rw [Cardinal.mk_fintype, finrank_eq_card_basis h]
  /-
    🎉 no goals
  -/


/-- If a vector space (or module) has a finite basis, then its dimension (or rank) is equal to the
cardinality of the basis. This lemma uses a `Finset` instead of indexed types. -/
theorem finrank_eq_card_finset_basis {ι : Type w} {b : Finset ι} (h : Basis b R M) :
                                      /-
                                        R : Type u
                                        M : Type v
                                        inst✝³ : Ring R
                                        inst✝² : AddCommGroup M
                                        inst✝¹ : Module R M
                                        inst✝ : StrongRankCondition R
                                        ι : Type w
                                        b : Finset ι
                                        h : Basis (Subtype fun x => Membership.mem b x) R M
                                        ⊢ Eq (Module.finrank R M) b.card
                                      -/
    finrank R M = Finset.card b := by rw [finrank_eq_card_basis h, Fintype.card_coe]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem rank_self : Module.rank R R = 1 := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : StrongRankCondition R
    ⊢ Eq (Module.rank R R) 1
  -/
  rw [← Cardinal.lift_inj, ← (Basis.singleton PUnit R).mk_eq_rank, Cardinal.mk_punit]
  /-
    🎉 no goals
  -/


/-- A ring satisfying `StrongRankCondition` (such as a `DivisionRing`) is one-dimensional as a
module over itself. -/
@[simp]
theorem finrank_self : finrank R R = 1 :=
                            /-
                              R : Type u
                              inst✝¹ : Ring R
                              inst✝ : StrongRankCondition R
                              ⊢ Eq (Module.rank R R) ↑1
                            -/
  finrank_eq_of_rank_eq (by simp)
                            /-
                              🎉 no goals
                            -/


/-- Given a basis of a ring over itself indexed by a type `ι`, then `ι` is `Unique`. -/
noncomputable def _root_.Basis.unique {ι : Type*} (b : Basis ι R R) : Unique ι := by
  have A : Cardinal.mk ι = ↑(Module.finrank R R) :=
    (Module.mk_finrank_eq_card_basis b).symm
  -- Porting note: replace `algebraMap.coe_one` with `Nat.cast_one`
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι✝ : Type w
    ι' : Type w'
    inst✝ : StrongRankCondition R
    ι : Type u_1
    b : Basis ι R R
    A : Eq (Cardinal.mk ι) ↑(Module.finrank R R)
    ⊢ Unique ι
  -/
  simp only [Cardinal.eq_one_iff_unique, Module.finrank_self, Nat.cast_one] at A
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι✝ : Type w
    ι' : Type w'
    inst✝ : StrongRankCondition R
    ι : Type u_1
    b : Basis ι R R
    A : And (Subsingleton ι) (Nonempty ι)
    ⊢ Unique ι
  -/
  exact Nonempty.some ((unique_iff_subsingleton_and_nonempty _).2 A)
  /-
    🎉 no goals
  -/


/-- The rank of a finite module is finite. -/
theorem rank_lt_aleph0 [Module.Finite R M] : Module.rank R M < ℵ₀ := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ⊢ LT.lt (Module.rank R M) Cardinal.aleph0
  -/
  simp only [Module.rank_def]
  -- Porting note: can't use `‹_›` as that pulls the unused `N` into the context
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ⊢ LT.lt (iSup fun ι => Cardinal.mk ↑↑ι) Cardinal.aleph0
  -/
  obtain ⟨S, hS⟩ := Module.finite_def.mp ‹Module.Finite R M›
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    S : Finset M
    hS : Eq (Submodule.span R ↑S) Top.top
    ⊢ LT.lt (iSup fun ι => Cardinal.mk ↑↑ι) Cardinal.aleph0
  -/
  refine (ciSup_le' fun i => ?_).trans_lt (nat_lt_aleph0 S.card)
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    S : Finset M
    hS : Eq (Submodule.span R ↑S) Top.top
    i : Subtype fun s => LinearIndependent R Subtype.val
    ⊢ LE.le (Cardinal.mk ↑↑i) ↑S.card
  -/
  exact linearIndependent_le_span_finset _ i.prop S hS
  /-
    🎉 no goals
  -/


noncomputable instance {R M : Type*} [DivisionRing R] [AddCommGroup M] [Module R M]
    {s t : Set M} [Module.Finite R (span R t)]
    (hs : LinearIndependent R ((↑) : s → M)) (hst : s ⊆ t) :
    Fintype (hs.extend hst) := by
  /-
    R✝ : Type u
    M✝ : Type v
    inst✝⁷ : Ring R✝
    inst✝⁶ : AddCommGroup M✝
    inst✝⁵ : Module R✝ M✝
    ι : Type w
    ι' : Type w'
    inst✝⁴ : StrongRankCondition R✝
    R : Type u_1
    M : Type u_2
    inst✝³ : DivisionRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    s t : Set M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem (Submodule.span R t) x)
    hs : LinearIndependent R Subtype.val
    hst : HasSubset.Subset s t
    ⊢ Fintype ↑(hs.extend hst)
  -/
  refine Classical.choice (Cardinal.lt_aleph0_iff_fintype.1 ?_)
  /-
    R✝ : Type u
    M✝ : Type v
    inst✝⁷ : Ring R✝
    inst✝⁶ : AddCommGroup M✝
    inst✝⁵ : Module R✝ M✝
    ι : Type w
    ι' : Type w'
    inst✝⁴ : StrongRankCondition R✝
    R : Type u_1
    M : Type u_2
    inst✝³ : DivisionRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    s t : Set M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem (Submodule.span R t) x)
    hs : LinearIndependent R Subtype.val
    hst : HasSubset.Subset s t
    ⊢ LT.lt (Cardinal.mk ↑(hs.extend hst)) Cardinal.aleph0
  -/
  rw [← rank_span_set (hs.linearIndependent_extend hst), hs.span_extend_eq_span]
  /-
    R✝ : Type u
    M✝ : Type v
    inst✝⁷ : Ring R✝
    inst✝⁶ : AddCommGroup M✝
    inst✝⁵ : Module R✝ M✝
    ι : Type w
    ι' : Type w'
    inst✝⁴ : StrongRankCondition R✝
    R : Type u_1
    M : Type u_2
    inst✝³ : DivisionRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    s t : Set M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem (Submodule.span R t) x)
    hs : LinearIndependent R Subtype.val
    hst : HasSubset.Subset s t
    ⊢ LT.lt (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R t) x …
  -/
  exact Module.rank_lt_aleph0 ..
  /-
    🎉 no goals
  -/


/-- If `M` is finite, `finrank M = rank M`. -/
@[simp]
theorem finrank_eq_rank [Module.Finite R M] : ↑(finrank R M) = Module.rank R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ⊢ Eq (↑(Module.finrank R M)) (Module.rank R M)
  -/
  rw [Module.finrank, cast_toNat_of_lt_aleph0 (rank_lt_aleph0 R M)]
  /-
    🎉 no goals
  -/


/-- If `M` is finite, then `finrank N = rank N` for all `N : Submodule M`. Note that
such an `N` need not be finitely generated. -/
protected theorem _root_.Submodule.finrank_eq_rank [Module.Finite R M] (N : Submodule R M) :
    finrank R N = Module.rank R N := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    N : Submodule R M
    ⊢ Eq (↑(Module.finrank R (Subtype fun x => Membership.mem N x))) (Module.rank  …
  -/
  rw [finrank, Cardinal.cast_toNat_of_lt_aleph0]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    N : Submodule R M
    ⊢ LT.lt (Module.rank R (Subtype fun x => Membership.mem N x)) Cardinal.aleph0
  -/
  exact lt_of_le_of_lt (Submodule.rank_le N) (rank_lt_aleph0 R M)
  /-
    🎉 no goals
  -/


theorem LinearMap.finrank_le_finrank_of_injective [Module.Finite R M'] {f : M →ₗ[R] M'}
    (hf : Function.Injective f) : finrank R M ≤ finrank R M' :=
  finrank_le_finrank_of_rank_le_rank (LinearMap.lift_rank_le_of_injective _ hf) (rank_lt_aleph0 _ _)


theorem LinearMap.finrank_range_le [Module.Finite R M] (f : M →ₗ[R] M') :
    finrank R (LinearMap.range f) ≤ finrank R M :=
  finrank_le_finrank_of_rank_le_rank (lift_rank_range_le f) (rank_lt_aleph0 _ _)


theorem LinearMap.finrank_le_of_isSMulRegular {S : Type*} [CommSemiring S] [Algebra S R]
    [Module S M] [IsScalarTower S R M] (L L' : Submodule R M) [Module.Finite R L'] {s : S}
    (hr : IsSMulRegular M s) (h : ∀ x ∈ L, s • x ∈ L') :
    Module.finrank R L ≤ Module.finrank R L' := by
  /-
    R : Type u
    M : Type v
    inst✝⁸ : Ring R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : StrongRankCondition R
    S : Type u_1
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra S R
    inst✝² : Module S M
    inst✝¹ : IsScalarTower S R M
    L L' : Submodule R M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem L' x)
    s : S
    hr : IsSMulRegular M s
    h : ∀ (x : M), Membership.mem L x → Membership.mem L' (HSMul.hSMul s x)
    ⊢ LE.le (Module.finrank R (Subtype fun x => Membership.mem L x)) (Module.finra …
  -/
  refine finrank_le_finrank_of_rank_le_rank (lift_le.mpr <| rank_le_of_isSMulRegular L L' hr h) ?_
  /-
    R : Type u
    M : Type v
    inst✝⁸ : Ring R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : StrongRankCondition R
    S : Type u_1
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra S R
    inst✝² : Module S M
    inst✝¹ : IsScalarTower S R M
    L L' : Submodule R M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem L' x)
    s : S
    hr : IsSMulRegular M s
    h : ∀ (x : M), Membership.mem L x → Membership.mem L' (HSMul.hSMul s x)
    ⊢ LT.lt (Module.rank R (Subtype fun x => Membership.mem L' x)) Cardinal.aleph0
  -/
  rw [← Module.finrank_eq_rank R L']
  /-
    R : Type u
    M : Type v
    inst✝⁸ : Ring R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : StrongRankCondition R
    S : Type u_1
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra S R
    inst✝² : Module S M
    inst✝¹ : IsScalarTower S R M
    L L' : Submodule R M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem L' x)
    s : S
    hr : IsSMulRegular M s
    h : ∀ (x : M), Membership.mem L x → Membership.mem L' (HSMul.hSMul s x)
    ⊢ LT.lt (↑(Module.finrank R (Subtype fun x => Membership.mem L' x))) Cardinal. …
  -/
  exact nat_lt_aleph0 (finrank R ↥L')
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-21")]
alias LinearMap.finrank_le_of_smul_regular := LinearMap.finrank_le_of_isSMulRegular


