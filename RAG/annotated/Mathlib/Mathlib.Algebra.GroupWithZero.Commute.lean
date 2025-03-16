theorem mul_inverse_rev' {a b : M₀} (h : Commute a b) :
    inverse (a * b) = inverse b * inverse a := by
  /-
    M₀ : Type u_1
    inst✝ : MonoidWithZero M₀
    a b : M₀
    h : Commute a b
    ⊢ Eq (Ring.inverse (HMul.hMul a b)) (HMul.hMul (Ring.inverse b) (Ring.inverse  …
  -/
  by_cases hab : IsUnit (a * b)
    /-
      case pos
      M₀ : Type u_1
      inst✝ : MonoidWithZero M₀
      a b : M₀
      h : Commute a b
      hab : IsUnit (HMul.hMul a b)
      ⊢ Eq (Ring.inverse (HMul.hMul a b)) (HMul.hMul (Ring.inverse b) (Ring.inverse  …
    -/
  · obtain ⟨⟨a, rfl⟩, b, rfl⟩ := h.isUnit_mul_iff.mp hab
    /-
      case pos.intro.intro.intro
      M₀ : Type u_1
      inst✝ : MonoidWithZero M₀
      a b : Units M₀
      h : Commute ↑a ↑b
      hab : IsUnit (HMul.hMul ↑a ↑b)
      ⊢ Eq (Ring.inverse (HMul.hMul ↑a ↑b)) (HMul.hMul (Ring.inverse ↑b) (Ring.inver …
    -/
    rw [← Units.val_mul, inverse_unit, inverse_unit, inverse_unit, ← Units.val_mul, mul_inv_rev]
    /-
      🎉 no goals
    -/
  /-
    case neg
    M₀ : Type u_1
    inst✝ : MonoidWithZero M₀
    a b : M₀
    h : Commute a b
    hab : Not (IsUnit (HMul.hMul a b))
    ⊢ Eq (Ring.inverse (HMul.hMul a b)) (HMul.hMul (Ring.inverse b) (Ring.inverse  …
  -/
  obtain ha | hb := not_and_or.mp (mt h.isUnit_mul_iff.mpr hab)
    /-
      case neg.inl
      M₀ : Type u_1
      inst✝ : MonoidWithZero M₀
      a b : M₀
      h : Commute a b
      hab : Not (IsUnit (HMul.hMul a b))
      ha : Not (IsUnit a)
      ⊢ Eq (Ring.inverse (HMul.hMul a b)) (HMul.hMul (Ring.inverse b) (Ring.inverse  …
    -/
  · rw [inverse_non_unit _ hab, inverse_non_unit _ ha, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg.inr
      M₀ : Type u_1
      inst✝ : MonoidWithZero M₀
      a b : M₀
      h : Commute a b
      hab : Not (IsUnit (HMul.hMul a b))
      hb : Not (IsUnit b)
      ⊢ Eq (Ring.inverse (HMul.hMul a b)) (HMul.hMul (Ring.inverse b) (Ring.inverse  …
    -/
  · rw [inverse_non_unit _ hab, inverse_non_unit _ hb, zero_mul]
    /-
      🎉 no goals
    -/


theorem mul_inverse_rev {M₀} [CommMonoidWithZero M₀] (a b : M₀) :
    Ring.inverse (a * b) = inverse b * inverse a :=
  mul_inverse_rev' (Commute.all _ _)


lemma inverse_pow (r : M₀) : ∀ n : ℕ, Ring.inverse r ^ n = Ring.inverse (r ^ n)
            /-
              M₀ : Type u_1
              inst✝ : MonoidWithZero M₀
              r : M₀
              ⊢ Eq (HPow.hPow (Ring.inverse r) 0) (Ring.inverse (HPow.hPow r 0))
            -/
  | 0 => by rw [pow_zero, pow_zero, Ring.inverse_one]
            /-
              🎉 no goals
            -/
  | n + 1 => by
    rw [pow_succ', pow_succ, Ring.mul_inverse_rev' ((Commute.refl r).pow_left n),
      Ring.inverse_pow r n]


theorem Commute.ring_inverse_ring_inverse {a b : M₀} (h : Commute a b) :
    Commute (Ring.inverse a) (Ring.inverse b) :=
  (Ring.mul_inverse_rev' h.symm).symm.trans <| (congr_arg _ h.symm.eq).trans <|
    Ring.mul_inverse_rev' h


@[simp]
theorem zero_right [MulZeroClass G₀] (a : G₀) : Commute a 0 :=
  SemiconjBy.zero_right a


@[simp]
theorem zero_left [MulZeroClass G₀] (a : G₀) : Commute 0 a :=
  SemiconjBy.zero_left a a


@[simp]
theorem inv_left_iff₀ : Commute a⁻¹ b ↔ Commute a b :=
  SemiconjBy.inv_symm_left_iff₀


theorem inv_left₀ (h : Commute a b) : Commute a⁻¹ b :=
  inv_left_iff₀.2 h


@[simp]
theorem inv_right_iff₀ : Commute a b⁻¹ ↔ Commute a b :=
  SemiconjBy.inv_right_iff₀


theorem inv_right₀ (h : Commute a b) : Commute a b⁻¹ :=
  inv_right_iff₀.2 h


@[simp]
theorem div_right (hab : Commute a b) (hac : Commute a c) : Commute a (b / c) :=
  SemiconjBy.div_right hab hac


@[simp]
theorem div_left (hac : Commute a c) (hbc : Commute b c) : Commute (a / b) c := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a b c : G₀
    hac : Commute a c
    hbc : Commute b c
    ⊢ Commute (HDiv.hDiv a b) c
  -/
  rw [div_eq_mul_inv]
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a b c : G₀
    hac : Commute a c
    hbc : Commute b c
    ⊢ Commute (HMul.hMul a (Inv.inv b)) c
  -/
  exact hac.mul_left hbc.inv_left₀
  /-
    🎉 no goals
  -/


theorem pow_inv_comm₀ (a : G₀) (m n : ℕ) : a⁻¹ ^ m * a ^ n = a ^ n * a⁻¹ ^ m :=
  (Commute.refl a).inv_left₀.pow_pow m n


