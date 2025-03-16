instance instNormedAddCommGroup : NormedAddCommGroup ℤ where
  norm n := ‖(n : ℝ)‖
                    /-
                      α : Type u_1
                      m n : Int
                      ⊢ Eq (Dist.dist m n) (Norm.norm (HSub.hSub m n))
                    -/
  dist_eq m n := by simp only [Int.dist_eq, norm, Int.cast_sub]
                    /-
                      🎉 no goals
                    -/


@[norm_cast]
theorem norm_cast_real (m : ℤ) : ‖(m : ℝ)‖ = ‖m‖ :=
  rfl


theorem norm_eq_abs (n : ℤ) : ‖n‖ = |(n : ℝ)| :=
  rfl


@[simp]
                                                   /-
                                                     n : Nat
                                                     ⊢ Eq (Norm.norm ↑n) ↑n
                                                   -/
theorem norm_natCast (n : ℕ) : ‖(n : ℤ)‖ = n := by simp [Int.norm_eq_abs]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-04-05")] alias norm_coe_nat := norm_natCast


theorem _root_.NNReal.natCast_natAbs (n : ℤ) : (n.natAbs : ℝ≥0) = ‖n‖₊ :=
  NNReal.eq <|
    calc
                                                    /-
                                                      n : Int
                                                      ⊢ Eq ↑↑n.natAbs ↑↑n.natAbs
                                                    -/
      ((n.natAbs : ℝ≥0) : ℝ) = (n.natAbs : ℤ) := by simp only [Int.cast_natCast, NNReal.coe_natCast]
                                                    /-
                                                      🎉 no goals
                                                    -/
                          /-
                            n : Int
                            ⊢ Eq (↑↑n.natAbs) (abs ↑n)
                          -/
      _ = |(n : ℝ)| := by simp only [Int.natCast_natAbs, Int.cast_abs]
                          /-
                            🎉 no goals
                          -/
      _ = ‖n‖ := (norm_eq_abs n).symm


theorem abs_le_floor_nnreal_iff (z : ℤ) (c : ℝ≥0) : |z| ≤ ⌊c⌋₊ ↔ ‖z‖₊ ≤ c := by
  /-
    z : Int
    c : NNReal
    ⊢ Iff (LE.le (abs z) ↑(Nat.floor c)) (LE.le (NNNorm.nnnorm z) c)
  -/
  rw [Int.abs_eq_natAbs, Int.ofNat_le, Nat.le_floor_iff (zero_le c), NNReal.natCast_natAbs z]
  /-
    🎉 no goals
  -/


@[to_additive norm_zsmul_le]
theorem norm_zpow_le_mul_norm (n : ℤ) (a : α) : ‖a ^ n‖ ≤ ‖n‖ * ‖a‖ := by
  /-
    α : Type u_1
    inst✝ : SeminormedCommGroup α
    n : Int
    a : α
    ⊢ LE.le (Norm.norm (HPow.hPow a n)) (HMul.hMul (Norm.norm n) (Norm.norm a))
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  rcases n.eq_nat_or_neg with ⟨n, rfl | rfl⟩ <;> simpa using norm_pow_le_mul_norm
                                                 /-
                                                   🎉 no goals
                                                 -/


@[to_additive nnnorm_zsmul_le]
theorem nnnorm_zpow_le_mul_norm (n : ℤ) (a : α) : ‖a ^ n‖₊ ≤ ‖n‖₊ * ‖a‖₊ := by
  /-
    α : Type u_1
    inst✝ : SeminormedCommGroup α
    n : Int
    a : α
    ⊢ LE.le (NNNorm.nnnorm (HPow.hPow a n)) (HMul.hMul (NNNorm.nnnorm n) (NNNorm.n …
  -/
  simpa only [← NNReal.coe_le_coe, NNReal.coe_mul] using norm_zpow_le_mul_norm n a
  /-
    🎉 no goals
  -/


