/-- A one-element family `x` is algebraically independent if and only if
its element is transcendental. -/
@[simp]
theorem algebraicIndependent_unique_type_iff [Unique ι] :
    AlgebraicIndependent R x ↔ Transcendental R (x default) := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Unique ι
    ⊢ Iff (AlgebraicIndependent R x) (Transcendental R (x Inhabited.default))
  -/
  rw [transcendental_iff_injective, algebraicIndependent_iff_injective_aeval]
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Unique ι
    ⊢ Iff (Function.Injective ⇑(MvPolynomial.aeval x)) (Function.Injective ⇑(Polyn …
  -/
  let i := (renameEquiv R (Equiv.equivPUnit.{_, 1} ι)).trans (pUnitAlgEquiv R)
  have key : aeval (R := R) x = (Polynomial.aeval (R := R) (x default)).comp i := by
    ext y
    simp [i, Subsingleton.elim y default]
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Unique ι
    i : AlgEquiv R (MvPolynomial ι R) (Polynomial R) := (MvPolynomial.renameEquiv  …
    key : Eq (MvPolynomial.aeval x) ((Polynomial.aeval (x Inhabited.default)).comp …
    ⊢ Iff (Function.Injective ⇑(MvPolynomial.aeval x)) (Function.Injective ⇑(Polyn …
  -/
  simp [key]
  /-
    🎉 no goals
  -/


theorem algebraicIndependent_singleton_iff [Subsingleton ι] (i : ι) :
    AlgebraicIndependent R x ↔ Transcendental R (x i) :=
  letI := uniqueOfSubsingleton i
  algebraicIndependent_unique_type_iff


/-- The one-element family `![x]` is algebraically independent if and only if
`x` is transcendental. -/
theorem algebraicIndependent_iff_transcendental {x : A} :
    AlgebraicIndependent R ![x] ↔ Transcendental R x := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    x : A
    ⊢ Iff (AlgebraicIndependent R (Matrix.vecCons x Matrix.vecEmpty)) (Transcenden …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If a family `x` is algebraically independent, then any of its element is transcendental. -/
theorem transcendental (i : ι) : Transcendental R (x i) := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    i : ι
    ⊢ Transcendental R (x i)
  -/
  have := hx.comp ![i] (Function.injective_of_subsingleton _)
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    i : ι
    this : AlgebraicIndependent R (Function.comp x (Matrix.vecCons i Matrix.vecEmp …
    ⊢ Transcendental R (x i)
  -/
  have : AlgebraicIndependent R ![x i] := by rwa [← FinVec.map_eq] at this
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    i : ι
    this✝ : AlgebraicIndependent R (Function.comp x (Matrix.vecCons i Matrix.vecEm …
    this : AlgebraicIndependent R (Matrix.vecCons (x i) Matrix.vecEmpty)
    ⊢ Transcendental R (x i)
  -/
  rwa [← algebraicIndependent_iff_transcendental]
  /-
    🎉 no goals
  -/


/-- If `A/R` is algebraic, then all algebraically independent families are empty. -/
theorem isEmpty_of_isAlgebraic [Algebra.IsAlgebraic R A] : IsEmpty ι := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    hx : AlgebraicIndependent R x
    inst✝ : Algebra.IsAlgebraic R A
    ⊢ IsEmpty ι
  -/
  rcases isEmpty_or_nonempty ι with h | ⟨⟨i⟩⟩
    /-
      case inl
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      x : ι → A
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      hx : AlgebraicIndependent R x
      inst✝ : Algebra.IsAlgebraic R A
      h : IsEmpty ι
      ⊢ IsEmpty ι
    -/
  · exact h
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    hx : AlgebraicIndependent R x
    inst✝ : Algebra.IsAlgebraic R A
    i : ι
    ⊢ IsEmpty ι
  -/
  exact False.elim (hx.transcendental i (Algebra.IsAlgebraic.isAlgebraic _))
  /-
    🎉 no goals
  -/


theorem AlgebraicIndependent.option_iff (hx : AlgebraicIndependent R x) (a : A) :
    (AlgebraicIndependent R fun o : Option ι => o.elim a x) ↔
      Transcendental (adjoin R (Set.range x)) a := by
  rw [algebraicIndependent_iff_injective_aeval, transcendental_iff_injective,
    ← AlgHom.coe_toRingHom, ← hx.aeval_comp_mvPolynomialOptionEquivPolynomialAdjoin,
    RingHom.coe_comp]
  exact Injective.of_comp_iff' (Polynomial.aeval a)
    (mvPolynomialOptionEquivPolynomialAdjoin hx).bijective


/-- Variant of `algebraicIndependent_of_finite_type` using `Transcendental`. -/
theorem algebraicIndependent_of_finite_type'
    (hinj : Injective (algebraMap R A))
    (H : ∀ t : Set ι, t.Finite → AlgebraicIndependent R (fun i : t ↦ x i) →
      ∀ i : ι, i ∉ t → Transcendental (adjoin R (x '' t)) (x i)) :
    AlgebraicIndependent R x := by
  classical
  refine algebraicIndependent_of_finite_type fun t hfin ↦ hfin.induction_on'
    (algebraicIndependent_empty_type_iff.mpr hinj) fun {a u} ha hu ha' h ↦ ?_
  convert ((Set.image_eq_range _ _ ▸ h.option_iff <| x a).2 <| H u (hfin.subset hu) h _ ha').comp _
    (Set.subtypeInsertEquivOption ha').injective with x
  by_cases h : ↑x = a <;> simp [h, Set.subtypeInsertEquivOption]


/-- Variant of `algebraicIndependent_of_finite` using `Transcendental`. -/
theorem algebraicIndependent_of_finite' (s : Set A)
    (hinj : Injective (algebraMap R A))
    (H : ∀ t ⊆ s, t.Finite → AlgebraicIndependent R ((↑) : t → A) →
      ∀ a ∈ s, a ∉ t → Transcendental (adjoin R t) a) :
    AlgebraicIndependent R ((↑) : s → A) :=
  algebraicIndependent_of_finite_type' hinj fun t hfin h i hi ↦ H _
        /-
          R : Type u_2
          A : Type u_3
          inst✝² : CommRing R
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          s : Set A
          hinj : Function.Injective ⇑(algebraMap R A)
          H : ∀ (t : Set A), HasSubset.Subset t s → t.Finite → AlgebraicIndependent R Su …
          t : Set (Subtype fun x => Membership.mem s x)
          hfin : t.Finite
          h : AlgebraicIndependent R fun i => ↑↑i
          i : Subtype fun x => Membership.mem s x
          hi : Not (Membership.mem t i)
          ⊢ HasSubset.Subset (Set.image Subtype.val t) s
        -/
    (by rintro _ ⟨x, _, rfl⟩; exact x.2) (hfin.image _) h.image _ i.2
                              /-
                                🎉 no goals
                              -/
    (mt Subtype.val_injective.mem_set_image.mp hi)


theorem adjoin_of_disjoint {s t : Set ι} (h : Disjoint s t) :
    AlgebraicIndependent (adjoin R (x '' s)) fun i : t ↦ x i := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    s t : Set ι
    h : Disjoint s t
    ⊢ AlgebraicIndependent (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (S …
  -/
  let e := (sumAlgEquiv R t s).trans (mapAlgEquiv t (hx.comp _ Subtype.val_injective).aevalEquiv)
  have : ((aeval fun i : t ↦ x i).restrictScalars R).comp e.toAlgHom =
      (aeval x).comp (rename <| Sum.elim Subtype.val Subtype.val) := by
    ext (_|_) <;> simp [e, algebraMap_aevalEquiv]
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    s t : Set ι
    h : Disjoint s t
    e : AlgEquiv R (MvPolynomial (Sum ↑t ↑s) R) (MvPolynomial (↑t) (Subtype fun x_ …
    this : Eq ((AlgHom.restrictScalars R (MvPolynomial.aeval fun i => x ↑i)).comp  …
    ⊢ AlgebraicIndependent (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (S …
  -/
  have _ := @MvPolynomial.isScalarTower
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    s t : Set ι
    h : Disjoint s t
    e : AlgEquiv R (MvPolynomial (Sum ↑t ↑s) R) (MvPolynomial (↑t) (Subtype fun x_ …
    this : Eq ((AlgHom.restrictScalars R (MvPolynomial.aeval fun i => x ↑i)).comp  …
    x✝ : ∀ {R : Type ?u.42570} {S₁ : Type ?u.42569} {S₂ : Type ?u.42568} {σ : Type …
    ⊢ AlgebraicIndependent (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (S …
  -/
  rw [Set.image_eq_range, AlgebraicIndependent, ← AlgHom.coe_restrictScalars' R, ← e.injective_comp]
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    s t : Set ι
    h : Disjoint s t
    e : AlgEquiv R (MvPolynomial (Sum ↑t ↑s) R) (MvPolynomial (↑t) (Subtype fun x_ …
    this : Eq ((AlgHom.restrictScalars R (MvPolynomial.aeval fun i => x ↑i)).comp  …
    x✝ : ∀ {R : Type u_2} {S₁ S₂ : Type u_3} {σ : Type u_1} [inst : CommSemiring S …
    ⊢ Function.Injective (Function.comp ⇑(AlgHom.restrictScalars R (MvPolynomial.a …
  -/
  show Injective ((AlgHom.restrictScalars R <| aeval _).comp e.toAlgHom)
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    s t : Set ι
    h : Disjoint s t
    e : AlgEquiv R (MvPolynomial (Sum ↑t ↑s) R) (MvPolynomial (↑t) (Subtype fun x_ …
    this : Eq ((AlgHom.restrictScalars R (MvPolynomial.aeval fun i => x ↑i)).comp  …
    x✝ : ∀ {R : Type u_2} {S₁ S₂ : Type u_3} {σ : Type u_1} [inst : CommSemiring S …
    ⊢ Function.Injective ⇑((AlgHom.restrictScalars R (MvPolynomial.aeval fun i =>  …
  -/
  rw [this, AlgHom.coe_comp]
  exact .comp hx (rename_injective _ <| Subtype.val_injective.sum_elim
    Subtype.val_injective fun i j eq ↦ h.ne_of_mem j.2 i.2 eq.symm)


theorem adjoin_iff_disjoint [Nontrivial A] {s t : Set ι} :
    (AlgebraicIndependent (adjoin R (x '' s)) fun i : t ↦ x i) ↔ Disjoint s t := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    hx : AlgebraicIndependent R x
    inst✝ : Nontrivial A
    s t : Set ι
    ⊢ Iff (AlgebraicIndependent (Subtype fun x_1 => Membership.mem (Algebra.adjoin …
  -/
  refine ⟨fun ind ↦ of_not_not fun ndisj ↦ ?_, adjoin_of_disjoint hx⟩
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    hx : AlgebraicIndependent R x
    inst✝ : Nontrivial A
    s t : Set ι
    ind : AlgebraicIndependent (Subtype fun x_1 => Membership.mem (Algebra.adjoin  …
    ndisj : Not (Disjoint s t)
    ⊢ False
  -/
  have ⟨i, hs, ht⟩ := Set.not_disjoint_iff.mp ndisj
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    hx : AlgebraicIndependent R x
    inst✝ : Nontrivial A
    s t : Set ι
    ind : AlgebraicIndependent (Subtype fun x_1 => Membership.mem (Algebra.adjoin  …
    ndisj : Not (Disjoint s t)
    i : ι
    hs : Membership.mem s i
    ht : Membership.mem t i
    ⊢ False
  -/
  refine ind.transcendental ⟨i, ht⟩ (isAlgebraic_algebraMap (⟨_, subset_adjoin ?_⟩ : adjoin R _))
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    hx : AlgebraicIndependent R x
    inst✝ : Nontrivial A
    s t : Set ι
    ind : AlgebraicIndependent (Subtype fun x_1 => Membership.mem (Algebra.adjoin  …
    ndisj : Not (Disjoint s t)
    i : ι
    hs : Membership.mem s i
    ht : Membership.mem t i
    ⊢ Membership.mem (Set.image x s) (x ↑⟨i, ht⟩)
  -/
  exact ⟨i, hs, rfl⟩
  /-
    🎉 no goals
  -/


theorem transcendental_adjoin {s : Set ι} {i : ι} (hi : i ∉ s) :
    Transcendental (adjoin R (x '' s)) (x i) := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    s : Set ι
    i : ι
    hi : Not (Membership.mem s i)
    ⊢ Transcendental (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Set.ima …
  -/
  convert ← hx.adjoin_of_disjoint (Set.disjoint_singleton_right.mpr hi)
  /-
    case a
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    s : Set ι
    i : ι
    hi : Not (Membership.mem s i)
    ⊢ Iff (AlgebraicIndependent (Subtype fun x_1 => Membership.mem (Algebra.adjoin …
  -/
  rw [algebraicIndependent_singleton_iff ⟨i, rfl⟩]
  /-
    🎉 no goals
  -/


theorem transcendental_adjoin_iff [Nontrivial A] {s : Set ι} {i : ι} :
    Transcendental (adjoin R (x '' s)) (x i) ↔ i ∉ s := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    hx : AlgebraicIndependent R x
    inst✝ : Nontrivial A
    s : Set ι
    i : ι
    ⊢ Iff (Transcendental (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Se …
  -/
  rw [← Set.disjoint_singleton_right]
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    hx : AlgebraicIndependent R x
    inst✝ : Nontrivial A
    s : Set ι
    i : ι
    ⊢ Iff (Transcendental (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Se …
  -/
  convert ← hx.adjoin_iff_disjoint (t := {i})
  /-
    case h.e'_1.a
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    hx : AlgebraicIndependent R x
    inst✝ : Nontrivial A
    s : Set ι
    i : ι
    ⊢ Iff (AlgebraicIndependent (Subtype fun x_1 => Membership.mem (Algebra.adjoin …
  -/
  rw [algebraicIndependent_singleton_iff ⟨i, rfl⟩]
  /-
    🎉 no goals
  -/


/-- If for each `i : ι`, `f_i : R[X]` is transcendental over `R`, then `{f_i(X_i) | i : ι}`
in `MvPolynomial ι R` is algebraically independent over `R`. -/
theorem algebraicIndependent_polynomial_aeval_X
    (f : ι → Polynomial R) (hf : ∀ i, Transcendental R (f i)) :
    AlgebraicIndependent R fun i ↦ Polynomial.aeval (X i : MvPolynomial ι R) (f i) := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝ : CommRing R
    f : ι → Polynomial R
    hf : ∀ (i : ι), Transcendental R (f i)
    ⊢ AlgebraicIndependent R fun i => (Polynomial.aeval (MvPolynomial.X i)) (f i)
  -/
  set x := fun i ↦ Polynomial.aeval (X i : MvPolynomial ι R) (f i)
  /-
    ι : Type u_1
    R : Type u_2
    inst✝ : CommRing R
    f : ι → Polynomial R
    hf : ∀ (i : ι), Transcendental R (f i)
    x : ι → MvPolynomial ι R := fun i => (Polynomial.aeval (MvPolynomial.X i)) (f i)
    ⊢ AlgebraicIndependent R x
  -/
  refine algebraicIndependent_of_finite_type' (C_injective _ _) fun t _ _ i hi ↦ ?_
  have hle : adjoin R (x '' t) ≤ supported R t := by
    rw [Algebra.adjoin_le_iff, Set.image_subset_iff]
    intro _ h
    rw [Set.mem_preimage]
    refine Algebra.adjoin_mono ?_ (Polynomial.aeval_mem_adjoin_singleton R _)
    simp_rw [singleton_subset_iff, Set.mem_image_of_mem _ h]
  /-
    ι : Type u_1
    R : Type u_2
    inst✝ : CommRing R
    f : ι → Polynomial R
    hf : ∀ (i : ι), Transcendental R (f i)
    x : ι → MvPolynomial ι R := fun i => (Polynomial.aeval (MvPolynomial.X i)) (f i)
    t : Set ι
    x✝¹ : t.Finite
    x✝ : AlgebraicIndependent R fun i => x ↑i
    i : ι
    hi : Not (Membership.mem t i)
    hle : LE.le (Algebra.adjoin R (Set.image x t)) (MvPolynomial.supported R t)
    ⊢ Transcendental (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Set.ima …
  -/
  exact (transcendental_supported_polynomial_aeval_X R hi (hf i)).of_tower_top_of_subalgebra_le hle
  /-
    🎉 no goals
  -/


/-- If `{x_i : A | i : ι}` is algebraically independent over `R`, and for each `i`,
`f_i : R[X]` is transcendental over `R`, then `{f_i(x_i) | i : ι}` is also
algebraically independent over `R`. -/
theorem AlgebraicIndependent.polynomial_aeval_of_transcendental
    (hx : AlgebraicIndependent R x)
    {f : ι → Polynomial R} (hf : ∀ i, Transcendental R (f i)) :
    AlgebraicIndependent R fun i ↦ Polynomial.aeval (x i) (f i) := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    f : ι → Polynomial R
    hf : ∀ (i : ι), Transcendental R (f i)
    ⊢ AlgebraicIndependent R fun i => (Polynomial.aeval (x i)) (f i)
  -/
  convert aeval_of_algebraicIndependent hx (algebraicIndependent_polynomial_aeval_X _ hf)
  /-
    case h.e'_4.h
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    f : ι → Polynomial R
    hf : ∀ (i : ι), Transcendental R (f i)
    x✝ : ι
    ⊢ Eq ((Polynomial.aeval (x x✝)) (f x✝)) ((MvPolynomial.aeval x) ((Polynomial.a …
  -/
  rw [← AlgHom.comp_apply]
  /-
    case h.e'_4.h
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    f : ι → Polynomial R
    hf : ∀ (i : ι), Transcendental R (f i)
    x✝ : ι
    ⊢ Eq ((Polynomial.aeval (x x✝)) (f x✝)) (((MvPolynomial.aeval x).comp (Polynom …
  -/
  congr 1; ext1; simp
                 /-
                   🎉 no goals
                 -/

