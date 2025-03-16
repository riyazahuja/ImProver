@[to_additive]
theorem units_inv_right : Commute a u → Commute a ↑u⁻¹ :=
  SemiconjBy.units_inv_right


@[to_additive (attr := simp)]
theorem units_inv_right_iff : Commute a ↑u⁻¹ ↔ Commute a u :=
  SemiconjBy.units_inv_right_iff


@[to_additive]
theorem units_inv_left : Commute (↑u) a → Commute (↑u⁻¹) a :=
  SemiconjBy.units_inv_symm_left


@[to_additive (attr := simp)]
theorem units_inv_left_iff : Commute (↑u⁻¹) a ↔ Commute (↑u) a :=
  SemiconjBy.units_inv_symm_left_iff


@[to_additive]
theorem units_val : Commute u₁ u₂ → Commute (u₁ : M) u₂ :=
  SemiconjBy.units_val


@[to_additive]
theorem units_of_val : Commute (u₁ : M) u₂ → Commute u₁ u₂ :=
  SemiconjBy.units_of_val


@[to_additive (attr := simp)]
theorem units_val_iff : Commute (u₁ : M) u₂ ↔ Commute u₁ u₂ :=
  SemiconjBy.units_val_iff


/-- If the product of two commuting elements is a unit, then the left multiplier is a unit. -/
@[to_additive "If the sum of two commuting elements is an additive unit, then the left summand is
an additive unit."]
def Units.leftOfMul (u : Mˣ) (a b : M) (hu : a * b = u) (hc : Commute a b) : Mˣ where
  val := a
  inv := b * ↑u⁻¹
                /-
                  M : Type u_1
                  inst✝ : Monoid M
                  n : Nat
                  a✝ b✝ : M
                  u✝ u₁ u₂ u : Units M
                  a b : M
                  hu : Eq (HMul.hMul a b) ↑u
                  hc : Commute a b
                  ⊢ Eq (HMul.hMul a (HMul.hMul b ↑(Inv.inv u))) 1
                -/
  val_inv := by rw [← mul_assoc, hu, u.mul_inv]
                /-
                  🎉 no goals
                -/
  inv_val := by
    /-
      M : Type u_1
      inst✝ : Monoid M
      n : Nat
      a✝ b✝ : M
      u✝ u₁ u₂ u : Units M
      a b : M
      hu : Eq (HMul.hMul a b) ↑u
      hc : Commute a b
      ⊢ Eq (HMul.hMul (HMul.hMul b ↑(Inv.inv u)) a) 1
    -/
    have : Commute a u := hu ▸ (Commute.refl _).mul_right hc
    /-
      M : Type u_1
      inst✝ : Monoid M
      n : Nat
      a✝ b✝ : M
      u✝ u₁ u₂ u : Units M
      a b : M
      hu : Eq (HMul.hMul a b) ↑u
      hc : Commute a b
      this : Commute a ↑u
      ⊢ Eq (HMul.hMul (HMul.hMul b ↑(Inv.inv u)) a) 1
    -/
    rw [← this.units_inv_right.right_comm, ← hc.eq, hu, u.mul_inv]
    /-
      🎉 no goals
    -/


/-- If the product of two commuting elements is a unit, then the right multiplier is a unit. -/
@[to_additive "If the sum of two commuting elements is an additive unit, then the right summand
is an additive unit."]
def Units.rightOfMul (u : Mˣ) (a b : M) (hu : a * b = u) (hc : Commute a b) : Mˣ :=
  u.leftOfMul b a (hc.eq ▸ hu) hc.symm


@[to_additive]
theorem Commute.isUnit_mul_iff (h : Commute a b) : IsUnit (a * b) ↔ IsUnit a ∧ IsUnit b :=
  ⟨fun ⟨u, hu⟩ => ⟨(u.leftOfMul a b hu.symm h).isUnit, (u.rightOfMul a b hu.symm h).isUnit⟩,
  fun H => H.1.mul H.2⟩


@[to_additive (attr := simp)]
theorem isUnit_mul_self_iff : IsUnit (a * a) ↔ IsUnit a :=
  (Commute.refl a).isUnit_mul_iff.trans and_self_iff


@[to_additive (attr := simp)]
lemma Commute.units_zpow_right (h : Commute a u) (m : ℤ) : Commute a ↑(u ^ m) :=
  SemiconjBy.units_zpow_right h m


@[to_additive (attr := simp)]
lemma Commute.units_zpow_left (h : Commute ↑u a) (m : ℤ) : Commute ↑(u ^ m) a :=
  (h.symm.units_zpow_right m).symm


/-- If a natural power of `x` is a unit, then `x` is a unit. -/
@[to_additive "If a natural multiple of `x` is an additive unit, then `x` is an additive unit."]
def Units.ofPow (u : Mˣ) (x : M) {n : ℕ} (hn : n ≠ 0) (hu : x ^ n = u) : Mˣ :=
  u.leftOfMul x (x ^ (n - 1))
        /-
          M : Type u_1
          inst✝ : Monoid M
          n✝ : Nat
          a b : M
          u✝ u₁ u₂ u : Units M
          x : M
          n : Nat
          hn : Ne n 0
          hu : Eq (HPow.hPow x n) ↑u
          ⊢ Eq (HMul.hMul x (HPow.hPow x (HSub.hSub n 1))) ↑u
        -/
    (by rwa [← _root_.pow_succ', Nat.sub_add_cancel (Nat.succ_le_of_lt <| Nat.pos_of_ne_zero hn)])
        /-
          🎉 no goals
        -/
    (Commute.self_pow _ _)


@[to_additive (attr := simp)] lemma isUnit_pow_iff (hn : n ≠ 0) : IsUnit (a ^ n) ↔ IsUnit a :=
  ⟨fun ⟨u, hu⟩ ↦ (u.ofPow a hn hu.symm).isUnit, IsUnit.pow n⟩


@[to_additive]
lemma isUnit_pow_succ_iff : IsUnit (a ^ (n + 1)) ↔ IsUnit a := isUnit_pow_iff n.succ_ne_zero


/-- If `a ^ n = 1`, `n ≠ 0`, then `a` is a unit. -/
@[to_additive (attr := simps!) "If `n • x = 0`, `n ≠ 0`, then `x` is an additive unit."]
def Units.ofPowEqOne (a : M) (n : ℕ) (ha : a ^ n = 1) (hn : n ≠ 0) : Mˣ := Units.ofPow 1 a hn ha


@[to_additive (attr := simp)]
lemma Units.pow_ofPowEqOne (ha : a ^ n = 1) (hn : n ≠ 0) :
                                                          /-
                                                            M : Type u_1
                                                            inst✝ : Monoid M
                                                            n : Nat
                                                            a : M
                                                            ha : Eq (HPow.hPow a n) 1
                                                            hn : Ne n 0
                                                            ⊢ Eq ↑(HPow.hPow (Units.ofPowEqOne a n ha hn) n) ↑1
                                                          -/
    Units.ofPowEqOne _ n ha hn ^ n = 1 := Units.ext <| by simp [ha]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[to_additive]
lemma isUnit_ofPowEqOne (ha : a ^ n = 1) (hn : n ≠ 0) : IsUnit a :=
  (Units.ofPowEqOne _ n ha hn).isUnit


@[to_additive]
lemma Commute.div_eq_div_iff_of_isUnit (hbd : Commute b d) (hb : IsUnit b) (hd : IsUnit d) :
    a / b = c / d ↔ a * d = c * b := by
  rw [← (hb.mul hd).mul_left_inj, ← mul_assoc, hb.div_mul_cancel, ← mul_assoc, hbd.right_comm,
    hd.div_mul_cancel]


