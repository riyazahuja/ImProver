/-- The isometry between a weighted sum of squares with weights `u` on the
(non-zero) real numbers and the weighted sum of squares with weights `sign ∘ u`. -/
noncomputable def isometryEquivSignWeightedSumSquares (w : ι → ℝ) :
    IsometryEquiv (weightedSumSquares ℝ w)
      (weightedSumSquares ℝ (fun i ↦ (sign (w i) : ℝ))) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    w : ι → Real
    ⊢ (QuadraticMap.weightedSumSquares Real w).IsometryEquiv (QuadraticMap.weighte …
  -/
  let u i := if h : w i = 0 then (1 : ℝˣ) else Units.mk0 (w i) h
  have hu : ∀ i : ι, 1 / √|(u i : ℝ)| ≠ 0 := fun i ↦
    have : (u i : ℝ) ≠ 0 := (u i).ne_zero
    by positivity
  have hwu : ∀ i, w i / |(u i : ℝ)| = sign (w i) := fun i ↦ by
    by_cases hi : w i = 0 <;> field_simp [hi, u]
  convert QuadraticMap.isometryEquivBasisRepr (weightedSumSquares ℝ w)
    ((Pi.basisFun ℝ ι).unitsSMul fun i => .mk0 _ (hu i))
  /-
    case h.e'_13
    ι : Type u_1
    inst✝ : Fintype ι
    w : ι → Real
    u : ι → Units Real := fun i => dite (Eq (w i) 0) (fun h => 1) fun h => Units.m …
    hu : ∀ (i : ι), Ne (HDiv.hDiv 1 (abs ↑(u i)).sqrt) 0
    hwu : ∀ (i : ι), Eq (HDiv.hDiv (w i) (abs ↑(u i))) ↑(SignType.sign (w i))
    ⊢ Eq (QuadraticMap.weightedSumSquares Real fun i => ↑(SignType.sign (w i))) (( …
  -/
  ext1 v
  classical
  suffices ∑ i, (w i / |(u i : ℝ)|) * v i ^ 2 = ∑ i, w i * (v i ^ 2 * |(u i : ℝ)|⁻¹) by
    simpa [basisRepr_apply, Basis.unitsSMul_apply, ← _root_.sq, mul_pow, ← hwu, Pi.single_apply]
  exact sum_congr rfl fun j _ ↦ by ring


/-- **Sylvester's law of inertia**: A nondegenerate real quadratic form is equivalent to a weighted
sum of squares with the weights being ±1, `SignType` version. -/
theorem equivalent_sign_ne_zero_weighted_sum_squared {M : Type*} [AddCommGroup M] [Module ℝ M]
    [FiniteDimensional ℝ M] (Q : QuadraticForm ℝ M) (hQ : (associated (R := ℝ) Q).SeparatingLeft) :
    ∃ w : Fin (Module.finrank ℝ M) → SignType,
      (∀ i, w i ≠ 0) ∧ Equivalent Q (weightedSumSquares ℝ fun i ↦ (w i : ℝ)) :=
  let ⟨w, ⟨hw₁⟩⟩ := Q.equivalent_weightedSumSquares_units_of_nondegenerate' hQ
  ⟨sign ∘ ((↑) : ℝˣ → ℝ) ∘ w, fun i => sign_ne_zero.2 (w i).ne_zero,
    ⟨hw₁.trans (isometryEquivSignWeightedSumSquares (((↑) : ℝˣ → ℝ) ∘ w))⟩⟩


/-- **Sylvester's law of inertia**: A nondegenerate real quadratic form is equivalent to a weighted
sum of squares with the weights being ±1. -/
theorem equivalent_one_neg_one_weighted_sum_squared {M : Type*} [AddCommGroup M] [Module ℝ M]
    [FiniteDimensional ℝ M] (Q : QuadraticForm ℝ M) (hQ : (associated (R := ℝ) Q).SeparatingLeft) :
    ∃ w : Fin (Module.finrank ℝ M) → ℝ,
      (∀ i, w i = -1 ∨ w i = 1) ∧ Equivalent Q (weightedSumSquares ℝ w) :=
  let ⟨w, hw₀, hw⟩ := Q.equivalent_sign_ne_zero_weighted_sum_squared hQ
                     /-
                       M : Type u_2
                       inst✝² : AddCommGroup M
                       inst✝¹ : Module Real M
                       inst✝ : FiniteDimensional Real M
                       Q : QuadraticForm Real M
                       hQ : LinearMap.SeparatingLeft (QuadraticMap.associated Q)
                       w : Fin (Module.finrank Real M) → SignType
                       hw₀ : ∀ (i : Fin (Module.finrank Real M)), Ne (w i) 0
                       hw : QuadraticMap.Equivalent Q (QuadraticMap.weightedSumSquares Real fun i =>  …
                       i : Fin (Module.finrank Real M)
                       ⊢ Or (Eq ((fun x => ↑(w x)) i) (-1)) (Eq ((fun x => ↑(w x)) i) 1)
                     -/
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
  ⟨(w ·), fun i ↦ by cases hi : w i <;> simp_all, hw⟩
                                        /-
                                          🎉 no goals
                                        -/


/-- **Sylvester's law of inertia**: A real quadratic form is equivalent to a weighted
sum of squares with the weights being ±1 or 0, `SignType` version. -/
theorem equivalent_signType_weighted_sum_squared {M : Type*} [AddCommGroup M] [Module ℝ M]
    [FiniteDimensional ℝ M] (Q : QuadraticForm ℝ M) :
    ∃ w : Fin (Module.finrank ℝ M) → SignType,
      Equivalent Q (weightedSumSquares ℝ fun i ↦ (w i : ℝ)) :=
  let ⟨w, ⟨hw₁⟩⟩ := Q.equivalent_weightedSumSquares
  ⟨sign ∘ w, ⟨hw₁.trans (isometryEquivSignWeightedSumSquares w)⟩⟩


/-- **Sylvester's law of inertia**: A real quadratic form is equivalent to a weighted
sum of squares with the weights being ±1 or 0. -/
theorem equivalent_one_zero_neg_one_weighted_sum_squared {M : Type*} [AddCommGroup M] [Module ℝ M]
    [FiniteDimensional ℝ M] (Q : QuadraticForm ℝ M) :
    ∃ w : Fin (Module.finrank ℝ M) → ℝ,
      (∀ i, w i = -1 ∨ w i = 0 ∨ w i = 1) ∧ Equivalent Q (weightedSumSquares ℝ w) :=
  let ⟨w, hw⟩ := Q.equivalent_signType_weighted_sum_squared
                     /-
                       M : Type u_2
                       inst✝² : AddCommGroup M
                       inst✝¹ : Module Real M
                       inst✝ : FiniteDimensional Real M
                       Q : QuadraticForm Real M
                       w : Fin (Module.finrank Real M) → SignType
                       hw : QuadraticMap.Equivalent Q (QuadraticMap.weightedSumSquares Real fun i =>  …
                       i : Fin (Module.finrank Real M)
                       ⊢ Or (Eq ((fun x => ↑(w x)) i) (-1)) (Or (Eq ((fun x => ↑(w x)) i) 0) (Eq ((fu …
                     -/
                                       /-
                                         🎉 no goals
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
  ⟨(w ·), fun i ↦ by cases h : w i <;> simp [h], hw⟩
                                       /-
                                         🎉 no goals
                                       -/


