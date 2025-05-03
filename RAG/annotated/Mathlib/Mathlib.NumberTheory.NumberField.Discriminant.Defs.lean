/-- The absolute discriminant of a number field. -/
noncomputable abbrev discr : ℤ := Algebra.discr ℤ (RingOfIntegers.basis K)


theorem coe_discr : (discr K : ℚ) = Algebra.discr ℚ (integralBasis K) :=
  (Algebra.discr_localizationLocalization ℤ _ K (RingOfIntegers.basis K)).symm


theorem discr_ne_zero : discr K ≠ 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Ne (NumberField.discr K) 0
  -/
  rw [← (Int.cast_injective (α := ℚ)).ne_iff, coe_discr]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Ne (Algebra.discr Rat ⇑(NumberField.integralBasis K)) ↑0
  -/
  exact Algebra.discr_not_zero_of_basis ℚ (integralBasis K)
  /-
    🎉 no goals
  -/


theorem discr_eq_discr {ι : Type*} [Fintype ι] [DecidableEq ι] (b : Basis ι ℤ (𝓞 K)) :
    Algebra.discr ℤ b = discr K := by
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : NumberField K
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι Int (NumberField.RingOfIntegers K)
    ⊢ Eq (Algebra.discr Int ⇑b) (NumberField.discr K)
  -/
  let b₀ := Basis.reindex (RingOfIntegers.basis K) (Basis.indexEquiv (RingOfIntegers.basis K) b)
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : NumberField K
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι Int (NumberField.RingOfIntegers K)
    b₀ : Basis ι Int (NumberField.RingOfIntegers K) := (NumberField.RingOfIntegers …
    ⊢ Eq (Algebra.discr Int ⇑b) (NumberField.discr K)
  -/
  rw [Algebra.discr_eq_discr (𝓞 K) b b₀, Basis.coe_reindex, Algebra.discr_reindex]
  /-
    🎉 no goals
  -/


theorem discr_eq_discr_of_algEquiv {L : Type*} [Field L] [NumberField L] (f : K ≃ₐ[ℚ] L) :
    discr K = discr L := by
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : NumberField K
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : NumberField L
    f : AlgEquiv Rat K L
    ⊢ Eq (NumberField.discr K) (NumberField.discr L)
  -/
  let f₀ : 𝓞 K ≃ₗ[ℤ] 𝓞 L := (f.restrictScalars ℤ).mapIntegralClosure.toLinearEquiv
  rw [← Rat.intCast_inj, coe_discr, Algebra.discr_eq_discr_of_algEquiv (integralBasis K) f,
    ← discr_eq_discr L ((RingOfIntegers.basis K).map f₀)]
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : NumberField K
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : NumberField L
    f : AlgEquiv Rat K L
    f₀ : LinearEquiv (RingHom.id Int) (NumberField.RingOfIntegers K) (NumberField. …
    ⊢ Eq (Algebra.discr Rat (Function.comp ⇑f ⇑(NumberField.integralBasis K))) ↑(A …
  -/
  change _ = algebraMap ℤ ℚ _
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : NumberField K
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : NumberField L
    f : AlgEquiv Rat K L
    f₀ : LinearEquiv (RingHom.id Int) (NumberField.RingOfIntegers K) (NumberField. …
    ⊢ Eq (Algebra.discr Rat (Function.comp ⇑f ⇑(NumberField.integralBasis K))) ((a …
  -/
  rw [← Algebra.discr_localizationLocalization ℤ (nonZeroDivisors ℤ) L]
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : NumberField K
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : NumberField L
    f : AlgEquiv Rat K L
    f₀ : LinearEquiv (RingHom.id Int) (NumberField.RingOfIntegers K) (NumberField. …
    ⊢ Eq (Algebra.discr Rat (Function.comp ⇑f ⇑(NumberField.integralBasis K))) (Al …
  -/
  congr
  /-
    case e_b
    K : Type u_1
    inst✝³ : Field K
    inst✝² : NumberField K
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : NumberField L
    f : AlgEquiv Rat K L
    f₀ : LinearEquiv (RingHom.id Int) (NumberField.RingOfIntegers K) (NumberField. …
    ⊢ Eq (Function.comp ⇑f ⇑(NumberField.integralBasis K)) ⇑(Basis.localizationLoc …
  -/
  ext
  simp only [Function.comp_apply, integralBasis_apply, Basis.localizationLocalization_apply,
    Basis.map_apply]
  /-
    case e_b.h
    K : Type u_1
    inst✝³ : Field K
    inst✝² : NumberField K
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : NumberField L
    f : AlgEquiv Rat K L
    f₀ : LinearEquiv (RingHom.id Int) (NumberField.RingOfIntegers K) (NumberField. …
    x✝ : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    ⊢ Eq (f ((algebraMap (NumberField.RingOfIntegers K) K) ((NumberField.RingOfInt …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The absolute discriminant of the number field `ℚ` is 1. -/
@[simp]
theorem numberField_discr : discr ℚ = 1 := by
  let b : Basis (Fin 1) ℤ (𝓞 ℚ) :=
    Basis.map (Basis.singleton (Fin 1) ℤ) ringOfIntegersEquiv.toAddEquiv.toIntLinearEquiv.symm
  calc NumberField.discr ℚ
    _ = Algebra.discr ℤ b := by convert (discr_eq_discr ℚ b).symm
    _ = Algebra.trace ℤ (𝓞 ℚ) (b default * b default) := by
      rw [Algebra.discr_def, Matrix.det_unique, Algebra.traceMatrix_apply, Algebra.traceForm_apply]
    _ = Algebra.trace ℤ (𝓞 ℚ) 1 := by
      rw [Basis.map_apply, RingEquiv.toAddEquiv_eq_coe, AddEquiv.toIntLinearEquiv_symm,
        AddEquiv.coe_toIntLinearEquiv, Basis.singleton_apply,
        show (AddEquiv.symm ↑ringOfIntegersEquiv) (1 : ℤ) = ringOfIntegersEquiv.symm 1 by rfl,
        map_one, mul_one]
    _ = 1 := by rw [Algebra.trace_eq_matrix_trace b]; norm_num


alias _root_.NumberField.discr_rat := numberField_discr


/-- If `b` and `b'` are `ℚ`-bases of a number field `K` such that
`∀ i j, IsIntegral ℤ (b.toMatrix b' i j)` and `∀ i j, IsIntegral ℤ (b'.toMatrix b i j)` then
`discr ℚ b = discr ℚ b'`. -/
theorem Algebra.discr_eq_discr_of_toMatrix_coeff_isIntegral [NumberField K]
    {b : Basis ι ℚ K} {b' : Basis ι' ℚ K} (h : ∀ i j, IsIntegral ℤ (b.toMatrix b' i j))
    (h' : ∀ i j, IsIntegral ℤ (b'.toMatrix b i j)) : discr ℚ b = discr ℚ b' := by
  replace h' : ∀ i j, IsIntegral ℤ (b'.toMatrix (b.reindex (b.indexEquiv b')) i j) := by
    intro i j
    convert h' i ((b.indexEquiv b').symm j)
    simp [Basis.toMatrix_apply]
  classical
  rw [← (b.reindex (b.indexEquiv b')).toMatrix_map_vecMul b', discr_of_matrix_vecMul,
    ← one_mul (discr ℚ b), Basis.coe_reindex, discr_reindex]
  congr
  have hint : IsIntegral ℤ ((b.reindex (b.indexEquiv b')).toMatrix b').det :=
    IsIntegral.det fun i j => h _ _
  obtain ⟨r, hr⟩ := IsIntegrallyClosed.isIntegral_iff.1 hint
  have hunit : IsUnit r := by
    have : IsIntegral ℤ (b'.toMatrix (b.reindex (b.indexEquiv b'))).det :=
      IsIntegral.det fun i j => h' _ _
    obtain ⟨r', hr'⟩ := IsIntegrallyClosed.isIntegral_iff.1 this
    refine isUnit_iff_exists_inv.2 ⟨r', ?_⟩
    suffices algebraMap ℤ ℚ (r * r') = 1 by
      rw [← RingHom.map_one (algebraMap ℤ ℚ)] at this
      exact (IsFractionRing.injective ℤ ℚ) this
    rw [RingHom.map_mul, hr, hr', ← Matrix.det_mul,
      Basis.toMatrix_mul_toMatrix_flip, Matrix.det_one]
  rw [← RingHom.map_one (algebraMap ℤ ℚ), ← hr]
  cases' Int.isUnit_iff.1 hunit with hp hm
  · simp [hp]
  · simp [hm]

