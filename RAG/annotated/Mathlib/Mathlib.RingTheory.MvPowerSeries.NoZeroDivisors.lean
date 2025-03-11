/-- A multivariate power series is not a zero divisor
  when its constant coefficient is not a zero divisor -/
theorem mem_nonZeroDivisors_of_constantCoeff {φ : MvPowerSeries σ R}
    (hφ : constantCoeff σ R φ ∈ nonZeroDivisors R) :
    φ ∈ nonZeroDivisors (MvPowerSeries σ R) := by
  classical
  intro x hx
  ext d
  apply WellFoundedLT.induction d
  intro e he
  rw [map_zero, ← mul_right_mem_nonZeroDivisors_eq_zero_iff hφ, ← map_zero (f := coeff R e), ← hx]
  convert (coeff_mul e x φ).symm
  rw [Finset.sum_eq_single (e,0), coeff_zero_eq_constantCoeff]
  · rintro ⟨u, _⟩ huv _
    suffices u < e by simp only [he u this, zero_mul, map_zero]
    apply lt_of_le_of_ne
    · simp only [← mem_antidiagonal.mp huv, le_add_iff_nonneg_right, zero_le]
    · rintro rfl
      simp_all
  · simp only [mem_antidiagonal, add_zero, not_true_eq_false, coeff_zero_eq_constantCoeff,
      false_implies]


instance {σ R : Type*} [Semiring R] [NoZeroDivisors R] :
    NoZeroDivisors (MvPowerSeries σ R) where
  eq_zero_or_eq_zero_of_mul_eq_zero {φ ψ} h := by
    /-
      σ✝ : Type u_1
      R✝ : Type u_2
      σ : Type u_3
      R : Type u_4
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      φ ψ : MvPowerSeries σ R
      h : Eq (HMul.hMul φ ψ) 0
      ⊢ Or (Eq φ 0) (Eq ψ 0)
    -/
    letI : LinearOrder σ := LinearOrder.swap σ WellOrderingRel.isWellOrder.linearOrder
    letI : WellFoundedGT σ := by
      change IsWellFounded σ fun x y ↦ WellOrderingRel x y
      exact IsWellOrder.toIsWellFounded
    /-
      σ✝ : Type u_1
      R✝ : Type u_2
      σ : Type u_3
      R : Type u_4
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      φ ψ : MvPowerSeries σ R
      h : Eq (HMul.hMul φ ψ) 0
      this✝ : LinearOrder σ := LinearOrder.swap σ (IsWellOrder.linearOrder WellOrder …
      this : WellFoundedGT σ := id IsWellOrder.toIsWellFounded
      ⊢ Or (Eq φ 0) (Eq ψ 0)
    -/
    simpa only [← lexOrder_eq_top_iff_eq_zero, lexOrder_mul, WithTop.add_eq_top] using h
    /-
      🎉 no goals
    -/


