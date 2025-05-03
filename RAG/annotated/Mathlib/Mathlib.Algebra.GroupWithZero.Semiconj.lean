@[simp]
theorem zero_right [MulZeroClass G₀] (a : G₀) : SemiconjBy a 0 0 := by
  /-
    G₀ : Type u_1
    inst✝ : MulZeroClass G₀
    a : G₀
    ⊢ SemiconjBy a 0 0
  -/
  simp only [SemiconjBy, mul_zero, zero_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_left [MulZeroClass G₀] (x y : G₀) : SemiconjBy 0 x y := by
  /-
    G₀ : Type u_1
    inst✝ : MulZeroClass G₀
    x y : G₀
    ⊢ SemiconjBy 0 x y
  -/
  simp only [SemiconjBy, mul_zero, zero_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_symm_left_iff₀ : SemiconjBy a⁻¹ x y ↔ SemiconjBy a y x :=
                                           /-
                                             G₀ : Type u_1
                                             inst✝ : GroupWithZero G₀
                                             a x y : G₀
                                             ha : Eq a 0
                                             ⊢ Iff (SemiconjBy (Inv.inv a) x y) (SemiconjBy a y x)
                                           -/
  Classical.by_cases (fun ha : a = 0 => by simp only [ha, inv_zero, SemiconjBy.zero_left]) fun ha =>
                                           /-
                                             🎉 no goals
                                           -/
    @units_inv_symm_left_iff _ _ (Units.mk0 a ha) _ _


theorem inv_symm_left₀ (h : SemiconjBy a x y) : SemiconjBy a⁻¹ y x :=
  SemiconjBy.inv_symm_left_iff₀.2 h


theorem inv_right₀ (h : SemiconjBy a x y) : SemiconjBy a x⁻¹ y⁻¹ := by
  /-
    G₀ : Type u_1
    inst✝ : GroupWithZero G₀
    a x y : G₀
    h : SemiconjBy a x y
    ⊢ SemiconjBy a (Inv.inv x) (Inv.inv y)
  -/
  by_cases ha : a = 0
    /-
      case pos
      G₀ : Type u_1
      inst✝ : GroupWithZero G₀
      a x y : G₀
      h : SemiconjBy a x y
      ha : Eq a 0
      ⊢ SemiconjBy a (Inv.inv x) (Inv.inv y)
    -/
  · simp only [ha, zero_left]
    /-
      🎉 no goals
    -/
  /-
    case neg
    G₀ : Type u_1
    inst✝ : GroupWithZero G₀
    a x y : G₀
    h : SemiconjBy a x y
    ha : Not (Eq a 0)
    ⊢ SemiconjBy a (Inv.inv x) (Inv.inv y)
  -/
  by_cases hx : x = 0
    /-
      case pos
      G₀ : Type u_1
      inst✝ : GroupWithZero G₀
      a x y : G₀
      h : SemiconjBy a x y
      ha : Not (Eq a 0)
      hx : Eq x 0
      ⊢ SemiconjBy a (Inv.inv x) (Inv.inv y)
    -/
  · subst x
    /-
      case pos
      G₀ : Type u_1
      inst✝ : GroupWithZero G₀
      a y : G₀
      ha : Not (Eq a 0)
      h : SemiconjBy a 0 y
      ⊢ SemiconjBy a (Inv.inv 0) (Inv.inv y)
    -/
    simp only [SemiconjBy, mul_zero, @eq_comm _ _ (y * a), mul_eq_zero] at h
    /-
      case pos
      G₀ : Type u_1
      inst✝ : GroupWithZero G₀
      a y : G₀
      ha : Not (Eq a 0)
      h : Or (Eq y 0) (Eq a 0)
      ⊢ SemiconjBy a (Inv.inv 0) (Inv.inv y)
    -/
    simp [h.resolve_right ha]
    /-
      🎉 no goals
    -/
    /-
      case neg
      G₀ : Type u_1
      inst✝ : GroupWithZero G₀
      a x y : G₀
      h : SemiconjBy a x y
      ha : Not (Eq a 0)
      hx : Not (Eq x 0)
      ⊢ SemiconjBy a (Inv.inv x) (Inv.inv y)
    -/
  · have := mul_ne_zero ha hx
    /-
      case neg
      G₀ : Type u_1
      inst✝ : GroupWithZero G₀
      a x y : G₀
      h : SemiconjBy a x y
      ha : Not (Eq a 0)
      hx : Not (Eq x 0)
      this : Ne (HMul.hMul a x) 0
      ⊢ SemiconjBy a (Inv.inv x) (Inv.inv y)
    -/
    rw [h.eq, mul_ne_zero_iff] at this
    /-
      case neg
      G₀ : Type u_1
      inst✝ : GroupWithZero G₀
      a x y : G₀
      h : SemiconjBy a x y
      ha : Not (Eq a 0)
      hx : Not (Eq x 0)
      this : And (Ne y 0) (Ne a 0)
      ⊢ SemiconjBy a (Inv.inv x) (Inv.inv y)
    -/
    exact @units_inv_right _ _ _ (Units.mk0 x hx) (Units.mk0 y this.1) h
    /-
      🎉 no goals
    -/


@[simp]
theorem inv_right_iff₀ : SemiconjBy a x⁻¹ y⁻¹ ↔ SemiconjBy a x y :=
  ⟨fun h => inv_inv x ▸ inv_inv y ▸ h.inv_right₀, inv_right₀⟩


theorem div_right (h : SemiconjBy a x y) (h' : SemiconjBy a x' y') :
    SemiconjBy a (x / x') (y / y') := by
  /-
    G₀ : Type u_1
    inst✝ : GroupWithZero G₀
    a x y x' y' : G₀
    h : SemiconjBy a x y
    h' : SemiconjBy a x' y'
    ⊢ SemiconjBy a (HDiv.hDiv x x') (HDiv.hDiv y y')
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv]
  /-
    G₀ : Type u_1
    inst✝ : GroupWithZero G₀
    a x y x' y' : G₀
    h : SemiconjBy a x y
    h' : SemiconjBy a x' y'
    ⊢ SemiconjBy a (HMul.hMul x (Inv.inv x')) (HMul.hMul y (Inv.inv y'))
  -/
  exact h.mul_right h'.inv_right₀
  /-
    🎉 no goals
  -/


lemma zpow_right₀ {a x y : G₀} (h : SemiconjBy a x y) : ∀ m : ℤ, SemiconjBy a (x ^ m) (y ^ m)
                  /-
                    G₀ : Type u_1
                    inst✝ : GroupWithZero G₀
                    a x y : G₀
                    h : SemiconjBy a x y
                    n : Nat
                    ⊢ SemiconjBy a (HPow.hPow x ↑n) (HPow.hPow y ↑n)
                  -/
  | (n : ℕ) => by simp [h.pow_right n]
                  /-
                    🎉 no goals
                  -/
                     /-
                       G₀ : Type u_1
                       inst✝ : GroupWithZero G₀
                       a x y : G₀
                       h : SemiconjBy a x y
                       n : Nat
                       ⊢ SemiconjBy a (HPow.hPow x (Int.negSucc n)) (HPow.hPow y (Int.negSucc n))
                     -/
  | .negSucc n => by simp only [zpow_negSucc, (h.pow_right (n + 1)).inv_right₀]
                     /-
                       🎉 no goals
                     -/


lemma zpow_right₀ (h : Commute a b) : ∀ m : ℤ, Commute a (b ^ m) := SemiconjBy.zpow_right₀ h


lemma zpow_left₀ (h : Commute a b) (m : ℤ) : Commute (a ^ m) b := (h.symm.zpow_right₀ m).symm


lemma zpow_zpow₀ (h : Commute a b) (m n : ℤ) : Commute (a ^ m) (b ^ n) :=
  (h.zpow_left₀ m).zpow_right₀ n


lemma zpow_self₀ (a : G₀) (n : ℤ) : Commute (a ^ n) a := (Commute.refl a).zpow_left₀ n


lemma self_zpow₀ (a : G₀) (n : ℤ) : Commute a (a ^ n) := (Commute.refl a).zpow_right₀ n


lemma zpow_zpow_self₀ (a : G₀) (m n : ℤ) : Commute (a ^ m) (a ^ n) :=
  (Commute.refl a).zpow_zpow₀ m n


