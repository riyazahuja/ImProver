/-- For `x ∈ K` the valuation of the zeroth coefficient of the minimal polynomial
of `algebra_map K L x` over `K` is equal to the valuation of `x`.-/
@[simp]
theorem coeff_zero_minpoly (x : K) : v ((minpoly K (algebraMap K L x)).coeff 0) = v x := by
  /-
    K : Type u_1
    inst✝³ : Field K
    Γ₀ : Type u_2
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : K
    ⊢ Eq (v ((minpoly K ((algebraMap K L) x)).coeff 0)) (v x)
  -/
  rw [minpoly.eq_X_sub_C, coeff_sub, coeff_X_zero, coeff_C_zero, zero_sub, Valuation.map_neg]
  /-
    🎉 no goals
  -/


theorem pow_coeff_zero_ne_zero_of_unit [FiniteDimensional K L] (x : L) (hx : IsUnit x):
    v ((minpoly K x).coeff 0) ^ (finrank K L / (minpoly K x).natDegree) ≠ (0 : Γ₀) := by
  /-
    K : Type u_1
    inst✝⁴ : Field K
    Γ₀ : Type u_2
    inst✝³ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    L : Type u_3
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    x : L
    hx : IsUnit x
    ⊢ Ne (HPow.hPow (v ((minpoly K x).coeff 0)) (HDiv.hDiv (Module.finrank K L) (m …
  -/
  have h_alg : Algebra.IsAlgebraic K L := Algebra.IsAlgebraic.of_finite K L
  /-
    K : Type u_1
    inst✝⁴ : Field K
    Γ₀ : Type u_2
    inst✝³ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    L : Type u_3
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    x : L
    hx : IsUnit x
    h_alg : Algebra.IsAlgebraic K L
    ⊢ Ne (HPow.hPow (v ((minpoly K x).coeff 0)) (HDiv.hDiv (Module.finrank K L) (m …
  -/
  have hx₀ : IsIntegral K x := (Algebra.IsAlgebraic.isAlgebraic x).isIntegral
  /-
    K : Type u_1
    inst✝⁴ : Field K
    Γ₀ : Type u_2
    inst✝³ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    L : Type u_3
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    x : L
    hx : IsUnit x
    h_alg : Algebra.IsAlgebraic K L
    hx₀ : IsIntegral K x
    ⊢ Ne (HPow.hPow (v ((minpoly K x).coeff 0)) (HDiv.hDiv (Module.finrank K L) (m …
  -/
  have hdeg := Nat.div_pos (natDegree_le x) (natDegree_pos hx₀)
  /-
    K : Type u_1
    inst✝⁴ : Field K
    Γ₀ : Type u_2
    inst✝³ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    L : Type u_3
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    x : L
    hx : IsUnit x
    h_alg : Algebra.IsAlgebraic K L
    hx₀ : IsIntegral K x
    hdeg : LT.lt 0 (HDiv.hDiv (Module.finrank K L) (minpoly K x).natDegree)
    ⊢ Ne (HPow.hPow (v ((minpoly K x).coeff 0)) (HDiv.hDiv (Module.finrank K L) (m …
  -/
  rw [ne_eq, pow_eq_zero_iff hdeg.ne.symm, Valuation.zero_iff]
  /-
    K : Type u_1
    inst✝⁴ : Field K
    Γ₀ : Type u_2
    inst✝³ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    L : Type u_3
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    x : L
    hx : IsUnit x
    h_alg : Algebra.IsAlgebraic K L
    hx₀ : IsIntegral K x
    hdeg : LT.lt 0 (HDiv.hDiv (Module.finrank K L) (minpoly K x).natDegree)
    ⊢ Not (Eq ((minpoly K x).coeff 0) 0)
  -/
  exact coeff_zero_ne_zero hx₀ hx.ne_zero
  /-
    🎉 no goals
  -/


