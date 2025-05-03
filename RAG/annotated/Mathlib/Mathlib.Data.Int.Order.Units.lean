theorem isUnit_iff_abs_eq {x : ℤ} : IsUnit x ↔ abs x = 1 := by
  /-
    x : Int
    ⊢ Iff (IsUnit x) (Eq (abs x) 1)
  -/
  rw [isUnit_iff_natAbs_eq, abs_eq_natAbs, ← Int.ofNat_one, natCast_inj]
  /-
    🎉 no goals
  -/


                                                            /-
                                                              a : Int
                                                              ha : IsUnit a
                                                              ⊢ Eq (HPow.hPow a 2) 1
                                                            -/
theorem isUnit_sq {a : ℤ} (ha : IsUnit a) : a ^ 2 = 1 := by rw [sq, isUnit_mul_self ha]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem units_sq (u : ℤˣ) : u ^ 2 = 1 := by
  /-
    u : Units Int
    ⊢ Eq (HPow.hPow u 2) 1
  -/
  rw [Units.ext_iff, Units.val_pow_eq_pow_val, Units.val_one, isUnit_sq u.isUnit]
  /-
    🎉 no goals
  -/


alias units_pow_two := units_sq


@[simp]
                                                  /-
                                                    u : Units Int
                                                    ⊢ Eq (HMul.hMul u u) 1
                                                  -/
theorem units_mul_self (u : ℤˣ) : u * u = 1 := by rw [← sq, units_sq]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
                                                   /-
                                                     u : Units Int
                                                     ⊢ Eq (Inv.inv u) u
                                                   -/
theorem units_inv_eq_self (u : ℤˣ) : u⁻¹ = u := by rw [inv_eq_iff_mul_eq_one, units_mul_self]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem units_div_eq_mul (u₁ u₂ : ℤˣ) : u₁ / u₂ = u₁ * u₂ := by
  /-
    u₁ u₂ : Units Int
    ⊢ Eq (HDiv.hDiv u₁ u₂) (HMul.hMul u₁ u₂)
  -/
  rw [div_eq_mul_inv, units_inv_eq_self]
  /-
    🎉 no goals
  -/

-- `Units.val_mul` is a "wrong turn" for the simplifier, this undoes it and simplifies further

@[simp]
theorem units_coe_mul_self (u : ℤˣ) : (u * u : ℤ) = 1 := by
  /-
    u : Units Int
    ⊢ Eq (HMul.hMul ↑u ↑u) 1
  -/
  rw [← Units.val_mul, units_mul_self, Units.val_one]
  /-
    🎉 no goals
  -/


                                                             /-
                                                               n : Nat
                                                               ⊢ Ne (HPow.hPow (-1) n) 0
                                                             -/
theorem neg_one_pow_ne_zero {n : ℕ} : (-1 : ℤ) ^ n ≠ 0 := by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem sq_eq_one_of_sq_lt_four {x : ℤ} (h1 : x ^ 2 < 4) (h2 : x ≠ 0) : x ^ 2 = 1 :=
  sq_eq_one_iff.mpr
    ((abs_eq (zero_le_one' ℤ)).mp
      (le_antisymm (lt_add_one_iff.mp (abs_lt_of_sq_lt_sq h1 zero_le_two))
        (sub_one_lt_iff.mp (abs_pos.mpr h2))))


theorem sq_eq_one_of_sq_le_three {x : ℤ} (h1 : x ^ 2 ≤ 3) (h2 : x ≠ 0) : x ^ 2 = 1 :=
  sq_eq_one_of_sq_lt_four (lt_of_le_of_lt h1 (lt_add_one (3 : ℤ))) h2


theorem units_pow_eq_pow_mod_two (u : ℤˣ) (n : ℕ) : u ^ n = u ^ (n % 2) := by
  conv =>
    lhs
    rw [← Nat.mod_add_div n 2]
    rw [pow_add, pow_mul, units_sq, one_pow, mul_one]


