/-- In a nontrivial monoid with zero, zero and one are different. -/
instance NeZero.one : NeZero (1 : M₀) := ⟨by
  /-
    M₀ : Type u_1
    M₀' : Type u_2
    inst✝¹ : MulZeroOneClass M₀
    inst✝ : Nontrivial M₀
    ⊢ Ne 1 0
  -/
  intro h
  /-
    M₀ : Type u_1
    M₀' : Type u_2
    inst✝¹ : MulZeroOneClass M₀
    inst✝ : Nontrivial M₀
    h : Eq 1 0
    ⊢ False
  -/
  rcases exists_pair_ne M₀ with ⟨x, y, hx⟩
  /-
    case intro.intro
    M₀ : Type u_1
    M₀' : Type u_2
    inst✝¹ : MulZeroOneClass M₀
    inst✝ : Nontrivial M₀
    h : Eq 1 0
    x y : M₀
    hx : Ne x y
    ⊢ False
  -/
  apply hx
  calc
    x = 1 * x := by rw [one_mul]
    _ = 0 := by rw [h, zero_mul]
    _ = 1 * y := by rw [h, zero_mul]
    _ = y := by rw [one_mul]⟩


/-- Pullback a `Nontrivial` instance along a function sending `0` to `0` and `1` to `1`. -/
theorem domain_nontrivial [Zero M₀'] [One M₀'] (f : M₀' → M₀) (zero : f 0 = 0) (one : f 1 = 1) :
    Nontrivial M₀' :=
  ⟨⟨0, 1, mt (congr_arg f) <| by
    /-
      M₀ : Type u_1
      M₀' : Type u_2
      inst✝³ : MulZeroOneClass M₀
      inst✝² : Nontrivial M₀
      inst✝¹ : Zero M₀'
      inst✝ : One M₀'
      f : M₀' → M₀
      zero : Eq (f 0) 0
      one : Eq (f 1) 1
      ⊢ Not (Eq (f 0) (f 1))
    -/
    rw [zero, one]
    /-
      M₀ : Type u_1
      M₀' : Type u_2
      inst✝³ : MulZeroOneClass M₀
      inst✝² : Nontrivial M₀
      inst✝¹ : Zero M₀'
      inst✝ : One M₀'
      f : M₀' → M₀
      zero : Eq (f 0) 0
      one : Eq (f 1) 1
      ⊢ Not (Eq 0 1)
    -/
    exact zero_ne_one⟩⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-07")] alias pullback_nonzero := domain_nontrivial


theorem inv_ne_zero (h : a ≠ 0) : a⁻¹ ≠ 0 := fun a_eq_0 => by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a : G₀
    h : Ne a 0
    a_eq_0 : Eq (Inv.inv a) 0
    ⊢ False
  -/
  have := mul_inv_cancel₀ h
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a : G₀
    h : Ne a 0
    a_eq_0 : Eq (Inv.inv a) 0
    this : Eq (HMul.hMul a (Inv.inv a)) 1
    ⊢ False
  -/
  simp only [a_eq_0, mul_zero, zero_ne_one] at this
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_mul_cancel₀ (h : a ≠ 0) : a⁻¹ * a = 1 :=
  calc
                                          /-
                                            G₀ : Type u_3
                                            inst✝ : GroupWithZero G₀
                                            a : G₀
                                            h : Ne a 0
                                            ⊢ Eq (HMul.hMul (Inv.inv a) a) (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv a) a) …
                                          -/
    a⁻¹ * a = a⁻¹ * a * a⁻¹ * a⁻¹⁻¹ := by simp [inv_ne_zero h]
                                          /-
                                            🎉 no goals
                                          -/
                          /-
                            G₀ : Type u_3
                            inst✝ : GroupWithZero G₀
                            a : G₀
                            h : Ne a 0
                            ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv a) a) (Inv.inv a)) (Inv.inv (In …
                          -/
    _ = a⁻¹ * a⁻¹⁻¹ := by simp [h]
                          /-
                            🎉 no goals
                          -/
                /-
                  G₀ : Type u_3
                  inst✝ : GroupWithZero G₀
                  a : G₀
                  h : Ne a 0
                  ⊢ Eq (HMul.hMul (Inv.inv a) (Inv.inv (Inv.inv a))) 1
                -/
    _ = 1 := by simp [inv_ne_zero h]
                /-
                  🎉 no goals
                -/


