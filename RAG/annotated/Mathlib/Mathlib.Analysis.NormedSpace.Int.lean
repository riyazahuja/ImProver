theorem nnnorm_coe_units (e : ℤˣ) : ‖(e : ℤ)‖₊ = 1 := by
  /-
    e : Units Int
    ⊢ Eq (NNNorm.nnnorm ↑e) 1
  -/
  obtain rfl | rfl := units_eq_one_or e <;>
    /-
      case inl
      ⊢ Eq (NNNorm.nnnorm ↑1) 1
    -/
    /-
      🎉 no goals
    -/
    simp only [Units.coe_neg_one, Units.val_one, nnnorm_neg, nnnorm_one]
    /-
      🎉 no goals
    -/


theorem norm_coe_units (e : ℤˣ) : ‖(e : ℤ)‖ = 1 := by
  /-
    e : Units Int
    ⊢ Eq (Norm.norm ↑e) 1
  -/
  rw [← coe_nnnorm, nnnorm_coe_units, NNReal.coe_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem nnnorm_natCast (n : ℕ) : ‖(n : ℤ)‖₊ = n :=
  Real.nnnorm_natCast _


@[deprecated (since := "2024-04-05")] alias nnnorm_coe_nat := nnnorm_natCast


@[simp]
theorem toNat_add_toNat_neg_eq_nnnorm (n : ℤ) : ↑n.toNat + ↑(-n).toNat = ‖n‖₊ := by
  /-
    n : Int
    ⊢ Eq (HAdd.hAdd ↑n.toNat ↑(Neg.neg n).toNat) (NNNorm.nnnorm n)
  -/
  rw [← Nat.cast_add, toNat_add_toNat_neg_eq_natAbs, NNReal.natCast_natAbs]
  /-
    🎉 no goals
  -/


@[simp]
theorem toNat_add_toNat_neg_eq_norm (n : ℤ) : ↑n.toNat + ↑(-n).toNat = ‖n‖ := by
  simpa only [NNReal.coe_natCast, NNReal.coe_add] using
    congrArg NNReal.toReal (toNat_add_toNat_neg_eq_nnnorm n)


