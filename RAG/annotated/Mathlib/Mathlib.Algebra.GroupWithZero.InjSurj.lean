/-- Pull back a `MulZeroClass` instance along an injective function.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.mulZeroClass [Mul M₀'] [Zero M₀'] (f : M₀' → M₀)
    (hf : Injective f) (zero : f 0 = 0) (mul : ∀ a b, f (a * b) = f a * f b) :
    MulZeroClass M₀' where
  mul := (· * ·)
  zero := 0
                         /-
                           M₀ : Type u_1
                           G₀ : Type u_2
                           M₀' : Type u_3
                           G₀' : Type u_4
                           inst✝² : MulZeroClass M₀
                           inst✝¹ : Mul M₀'
                           inst✝ : Zero M₀'
                           f : M₀' → M₀
                           hf : Function.Injective f
                           zero : Eq (f 0) 0
                           mul : ∀ (a b : M₀'), Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                           a : M₀'
                           ⊢ Eq (f (HMul.hMul 0 a)) (f 0)
                         -/
  zero_mul a := hf <| by simp only [mul, zero, zero_mul]
                         /-
                           🎉 no goals
                         -/
                         /-
                           M₀ : Type u_1
                           G₀ : Type u_2
                           M₀' : Type u_3
                           G₀' : Type u_4
                           inst✝² : MulZeroClass M₀
                           inst✝¹ : Mul M₀'
                           inst✝ : Zero M₀'
                           f : M₀' → M₀
                           hf : Function.Injective f
                           zero : Eq (f 0) 0
                           mul : ∀ (a b : M₀'), Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                           a : M₀'
                           ⊢ Eq (f (HMul.hMul a 0)) (f 0)
                         -/
  mul_zero a := hf <| by simp only [mul, zero, mul_zero]
                         /-
                           🎉 no goals
                         -/


/-- Push forward a `MulZeroClass` instance along a surjective function.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.mulZeroClass [Mul M₀'] [Zero M₀'] (f : M₀ → M₀')
    (hf : Surjective f) (zero : f 0 = 0) (mul : ∀ a b, f (a * b) = f a * f b) :
    MulZeroClass M₀' where
  mul := (· * ·)
  zero := 0
                                      /-
                                        M₀ : Type u_1
                                        G₀ : Type u_2
                                        M₀' : Type u_3
                                        G₀' : Type u_4
                                        inst✝² : MulZeroClass M₀
                                        inst✝¹ : Mul M₀'
                                        inst✝ : Zero M₀'
                                        f : M₀ → M₀'
                                        hf : Function.Surjective f
                                        zero : Eq (f 0) 0
                                        mul : ∀ (a b : M₀), Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                                        x : M₀
                                        ⊢ Eq (HMul.hMul (f x) 0) 0
                                      -/
                                      /-
                                        M₀ : Type u_1
                                        G₀ : Type u_2
                                        M₀' : Type u_3
                                        G₀' : Type u_4
                                        inst✝² : MulZeroClass M₀
                                        inst✝¹ : Mul M₀'
                                        inst✝ : Zero M₀'
                                        f : M₀ → M₀'
                                        hf : Function.Surjective f
                                        zero : Eq (f 0) 0
                                        mul : ∀ (a b : M₀), Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                                        x : M₀
                                        ⊢ Eq (HMul.hMul 0 (f x)) 0
                                      -/
  mul_zero := hf.forall.2 fun x => by simp only [← zero, ← mul, mul_zero]
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  zero_mul := hf.forall.2 fun x => by simp only [← zero, ← mul, zero_mul]


/-- Pull back a `NoZeroDivisors` instance along an injective function. -/
protected theorem Function.Injective.noZeroDivisors [NoZeroDivisors M₀'] : NoZeroDivisors M₀ where
  eq_zero_or_eq_zero_of_mul_eq_zero {a b} H :=
                               /-
                                 M₀ : Type u_1
                                 M₀' : Type u_3
                                 inst✝⁴ : Mul M₀
                                 inst✝³ : Zero M₀
                                 inst✝² : Mul M₀'
                                 inst✝¹ : Zero M₀'
                                 f : M₀ → M₀'
                                 hf : Function.Injective f
                                 zero : Eq (f 0) 0
                                 mul : ∀ (x y : M₀), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                 inst✝ : NoZeroDivisors M₀'
                                 a b : M₀
                                 H : Eq (HMul.hMul a b) 0
                                 ⊢ Eq (HMul.hMul (f a) (f b)) 0
                               -/
    have : f a * f b = 0 := by rw [← mul, H, zero]
                               /-
                                 🎉 no goals
                               -/
    (eq_zero_or_eq_zero_of_mul_eq_zero this).imp
                        /-
                          M₀ : Type u_1
                          M₀' : Type u_3
                          inst✝⁴ : Mul M₀
                          inst✝³ : Zero M₀
                          inst✝² : Mul M₀'
                          inst✝¹ : Zero M₀'
                          f : M₀ → M₀'
                          hf : Function.Injective f
                          zero : Eq (f 0) 0
                          mul : ∀ (x y : M₀), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                          inst✝ : NoZeroDivisors M₀'
                          a b : M₀
                          H✝ : Eq (HMul.hMul a b) 0
                          this : Eq (HMul.hMul (f a) (f b)) 0
                          H : Eq (f a) 0
                          ⊢ Eq (f a) (f 0)
                        -/
                        /-
                          🎉 no goals
                        -/
      (fun H ↦ hf <| by rwa [zero]) fun H ↦ hf <| by rwa [zero]
                                                     /-
                                                       🎉 no goals
                                                     -/


protected theorem Function.Injective.isLeftCancelMulZero
    [IsLeftCancelMulZero M₀'] : IsLeftCancelMulZero M₀ where
  mul_left_cancel_of_ne_zero Hne He := by
    /-
      M₀ : Type u_1
      M₀' : Type u_3
      inst✝⁴ : Mul M₀
      inst✝³ : Zero M₀
      inst✝² : Mul M₀'
      inst✝¹ : Zero M₀'
      f : M₀ → M₀'
      hf : Function.Injective f
      zero : Eq (f 0) 0
      mul : ∀ (x y : M₀), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      inst✝ : IsLeftCancelMulZero M₀'
      a✝ b✝ c✝ : M₀
      Hne : Ne a✝ 0
      He : Eq (HMul.hMul a✝ b✝) (HMul.hMul a✝ c✝)
      ⊢ Eq b✝ c✝
    -/
    have := congr_arg f He
    /-
      M₀ : Type u_1
      M₀' : Type u_3
      inst✝⁴ : Mul M₀
      inst✝³ : Zero M₀
      inst✝² : Mul M₀'
      inst✝¹ : Zero M₀'
      f : M₀ → M₀'
      hf : Function.Injective f
      zero : Eq (f 0) 0
      mul : ∀ (x y : M₀), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      inst✝ : IsLeftCancelMulZero M₀'
      a✝ b✝ c✝ : M₀
      Hne : Ne a✝ 0
      He : Eq (HMul.hMul a✝ b✝) (HMul.hMul a✝ c✝)
      this : Eq (f (HMul.hMul a✝ b✝)) (f (HMul.hMul a✝ c✝))
      ⊢ Eq b✝ c✝
    -/
    rw [mul, mul] at this
    /-
      M₀ : Type u_1
      M₀' : Type u_3
      inst✝⁴ : Mul M₀
      inst✝³ : Zero M₀
      inst✝² : Mul M₀'
      inst✝¹ : Zero M₀'
      f : M₀ → M₀'
      hf : Function.Injective f
      zero : Eq (f 0) 0
      mul : ∀ (x y : M₀), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      inst✝ : IsLeftCancelMulZero M₀'
      a✝ b✝ c✝ : M₀
      Hne : Ne a✝ 0
      He : Eq (HMul.hMul a✝ b✝) (HMul.hMul a✝ c✝)
      this : Eq (HMul.hMul (f a✝) (f b✝)) (HMul.hMul (f a✝) (f c✝))
      ⊢ Eq b✝ c✝
    -/
    exact hf (mul_left_cancel₀ (fun Hfa => Hne <| hf <| by rw [Hfa, zero]) this)
    /-
      🎉 no goals
    -/


protected theorem Function.Injective.isRightCancelMulZero
    [IsRightCancelMulZero M₀'] : IsRightCancelMulZero M₀ where
  mul_right_cancel_of_ne_zero Hne He := by
    /-
      M₀ : Type u_1
      M₀' : Type u_3
      inst✝⁴ : Mul M₀
      inst✝³ : Zero M₀
      inst✝² : Mul M₀'
      inst✝¹ : Zero M₀'
      f : M₀ → M₀'
      hf : Function.Injective f
      zero : Eq (f 0) 0
      mul : ∀ (x y : M₀), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      inst✝ : IsRightCancelMulZero M₀'
      a✝ b✝ c✝ : M₀
      Hne : Ne b✝ 0
      He : Eq (HMul.hMul a✝ b✝) (HMul.hMul c✝ b✝)
      ⊢ Eq a✝ c✝
    -/
    have := congr_arg f He
    /-
      M₀ : Type u_1
      M₀' : Type u_3
      inst✝⁴ : Mul M₀
      inst✝³ : Zero M₀
      inst✝² : Mul M₀'
      inst✝¹ : Zero M₀'
      f : M₀ → M₀'
      hf : Function.Injective f
      zero : Eq (f 0) 0
      mul : ∀ (x y : M₀), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      inst✝ : IsRightCancelMulZero M₀'
      a✝ b✝ c✝ : M₀
      Hne : Ne b✝ 0
      He : Eq (HMul.hMul a✝ b✝) (HMul.hMul c✝ b✝)
      this : Eq (f (HMul.hMul a✝ b✝)) (f (HMul.hMul c✝ b✝))
      ⊢ Eq a✝ c✝
    -/
    rw [mul, mul] at this
    /-
      M₀ : Type u_1
      M₀' : Type u_3
      inst✝⁴ : Mul M₀
      inst✝³ : Zero M₀
      inst✝² : Mul M₀'
      inst✝¹ : Zero M₀'
      f : M₀ → M₀'
      hf : Function.Injective f
      zero : Eq (f 0) 0
      mul : ∀ (x y : M₀), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      inst✝ : IsRightCancelMulZero M₀'
      a✝ b✝ c✝ : M₀
      Hne : Ne b✝ 0
      He : Eq (HMul.hMul a✝ b✝) (HMul.hMul c✝ b✝)
      this : Eq (HMul.hMul (f a✝) (f b✝)) (HMul.hMul (f c✝) (f b✝))
      ⊢ Eq a✝ c✝
    -/
    exact hf (mul_right_cancel₀ (fun Hfa => Hne <| hf <| by rw [Hfa, zero]) this)
    /-
      🎉 no goals
    -/


protected theorem Function.Injective.isCancelMulZero
    [IsCancelMulZero M₀'] : IsCancelMulZero M₀ where
  __ := hf.isLeftCancelMulZero f zero mul
  __ := hf.isRightCancelMulZero f zero mul


/-- Pull back a `MulZeroOneClass` instance along an injective function.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.mulZeroOneClass [Mul M₀'] [Zero M₀'] [One M₀'] (f : M₀' → M₀)
    (hf : Injective f) (zero : f 0 = 0) (one : f 1 = 1) (mul : ∀ a b, f (a * b) = f a * f b) :
    MulZeroOneClass M₀' :=
  { hf.mulZeroClass f zero mul, hf.mulOneClass f one mul with }


/-- Push forward a `MulZeroOneClass` instance along a surjective function.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.mulZeroOneClass [Mul M₀'] [Zero M₀'] [One M₀'] (f : M₀ → M₀')
    (hf : Surjective f) (zero : f 0 = 0) (one : f 1 = 1) (mul : ∀ a b, f (a * b) = f a * f b) :
    MulZeroOneClass M₀' :=
  { hf.mulZeroClass f zero mul, hf.mulOneClass f one mul with }


/-- Pull back a `SemigroupWithZero` along an injective function.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.semigroupWithZero [Zero M₀'] [Mul M₀'] [SemigroupWithZero M₀]
    (f : M₀' → M₀) (hf : Injective f) (zero : f 0 = 0) (mul : ∀ x y, f (x * y) = f x * f y) :
    SemigroupWithZero M₀' :=
  { hf.mulZeroClass f zero mul, ‹Zero M₀'›, hf.semigroup f mul with }


/-- Push forward a `SemigroupWithZero` along a surjective function.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.semigroupWithZero [SemigroupWithZero M₀] [Zero M₀'] [Mul M₀']
    (f : M₀ → M₀') (hf : Surjective f) (zero : f 0 = 0) (mul : ∀ x y, f (x * y) = f x * f y) :
    SemigroupWithZero M₀' :=
  { hf.mulZeroClass f zero mul, ‹Zero M₀'›, hf.semigroup f mul with }


/-- Pull back a `MonoidWithZero` along an injective function.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.monoidWithZero [Zero M₀'] [Mul M₀'] [One M₀'] [Pow M₀' ℕ]
    [MonoidWithZero M₀] (f : M₀' → M₀) (hf : Injective f) (zero : f 0 = 0) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) :
    MonoidWithZero M₀' :=
  { hf.monoid f one mul npow, hf.mulZeroClass f zero mul with }


/-- Push forward a `MonoidWithZero` along a surjective function.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.monoidWithZero [Zero M₀'] [Mul M₀'] [One M₀'] [Pow M₀' ℕ]
    [MonoidWithZero M₀] (f : M₀ → M₀') (hf : Surjective f) (zero : f 0 = 0) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) :
    MonoidWithZero M₀' :=
  { hf.monoid f one mul npow, hf.mulZeroClass f zero mul with }


/-- Pull back a `CommMonoidWithZero` along an injective function.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.commMonoidWithZero [Zero M₀'] [Mul M₀'] [One M₀'] [Pow M₀' ℕ]
    [CommMonoidWithZero M₀] (f : M₀' → M₀) (hf : Injective f) (zero : f 0 = 0) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) :
    CommMonoidWithZero M₀' :=
  { hf.commMonoid f one mul npow, hf.mulZeroClass f zero mul with }


/-- Push forward a `CommMonoidWithZero` along a surjective function.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.commMonoidWithZero [Zero M₀'] [Mul M₀'] [One M₀'] [Pow M₀' ℕ]
    [CommMonoidWithZero M₀] (f : M₀ → M₀') (hf : Surjective f) (zero : f 0 = 0) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) :
    CommMonoidWithZero M₀' :=
  { hf.commMonoid f one mul npow, hf.mulZeroClass f zero mul with }


/-- Pull back a `CancelMonoidWithZero` along an injective function.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.cancelMonoidWithZero [Zero M₀'] [Mul M₀'] [One M₀'] [Pow M₀' ℕ]
    (f : M₀' → M₀) (hf : Injective f) (zero : f 0 = 0) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) :
    CancelMonoidWithZero M₀' :=
  { hf.monoid f one mul npow, hf.mulZeroClass f zero mul with
    mul_left_cancel_of_ne_zero := fun hx H =>
                                                            /-
                                                              M₀ : Type u_1
                                                              G₀ : Type u_2
                                                              M₀' : Type u_3
                                                              G₀' : Type u_4
                                                              inst✝⁴ : CancelMonoidWithZero M₀
                                                              inst✝³ : Zero M₀'
                                                              inst✝² : Mul M₀'
                                                              inst✝¹ : One M₀'
                                                              inst✝ : Pow M₀' Nat
                                                              f : M₀' → M₀
                                                              hf : Function.Injective f
                                                              zero : Eq (f 0) 0
                                                              one : Eq (f 1) 1
                                                              mul : ∀ (x y : M₀'), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                              npow : ∀ (x : M₀') (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                                              a✝ b✝ c✝ : M₀'
                                                              hx : Ne a✝ 0
                                                              H : Eq (HMul.hMul a✝ b✝) (HMul.hMul a✝ c✝)
                                                              ⊢ Eq (HMul.hMul (f a✝) (f b✝)) (HMul.hMul (f a✝) (f c✝))
                                                            -/
      hf <| mul_left_cancel₀ ((hf.ne_iff' zero).2 hx) <| by rw [← mul, ← mul, H],
                                                            /-
                                                              🎉 no goals
                                                            -/
    mul_right_cancel_of_ne_zero := fun hx H =>
                                                             /-
                                                               M₀ : Type u_1
                                                               G₀ : Type u_2
                                                               M₀' : Type u_3
                                                               G₀' : Type u_4
                                                               inst✝⁴ : CancelMonoidWithZero M₀
                                                               inst✝³ : Zero M₀'
                                                               inst✝² : Mul M₀'
                                                               inst✝¹ : One M₀'
                                                               inst✝ : Pow M₀' Nat
                                                               f : M₀' → M₀
                                                               hf : Function.Injective f
                                                               zero : Eq (f 0) 0
                                                               one : Eq (f 1) 1
                                                               mul : ∀ (x y : M₀'), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                               npow : ∀ (x : M₀') (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                                               a✝ b✝ c✝ : M₀'
                                                               hx : Ne b✝ 0
                                                               H : Eq (HMul.hMul a✝ b✝) (HMul.hMul c✝ b✝)
                                                               ⊢ Eq (HMul.hMul (f a✝) (f b✝)) (HMul.hMul (f c✝) (f b✝))
                                                             -/
      hf <| mul_right_cancel₀ ((hf.ne_iff' zero).2 hx) <| by rw [← mul, ← mul, H] }
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Pull back a `CancelCommMonoidWithZero` along an injective function.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.cancelCommMonoidWithZero [Zero M₀'] [Mul M₀'] [One M₀']
    [Pow M₀' ℕ] (f : M₀' → M₀) (hf : Injective f) (zero : f 0 = 0) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) :
    CancelCommMonoidWithZero M₀' :=
  { hf.commMonoidWithZero f zero one mul npow, hf.cancelMonoidWithZero f zero one mul npow with }


/-- Pull back a `GroupWithZero` along an injective function.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.groupWithZero [Zero G₀'] [Mul G₀'] [One G₀'] [Inv G₀'] [Div G₀']
    [Pow G₀' ℕ] [Pow G₀' ℤ] (f : G₀' → G₀) (hf : Injective f) (zero : f 0 = 0) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : GroupWithZero G₀' :=
  { hf.monoidWithZero f zero one mul npow,
    hf.divInvMonoid f one mul inv div npow zpow,
    domain_nontrivial f zero one with
                         /-
                           M₀ : Type u_1
                           G₀ : Type u_2
                           M₀' : Type u_3
                           G₀' : Type u_4
                           inst✝⁷ : GroupWithZero G₀
                           inst✝⁶ : Zero G₀'
                           inst✝⁵ : Mul G₀'
                           inst✝⁴ : One G₀'
                           inst✝³ : Inv G₀'
                           inst✝² : Div G₀'
                           inst✝¹ : Pow G₀' Nat
                           inst✝ : Pow G₀' Int
                           f : G₀' → G₀
                           hf : Function.Injective f
                           zero : Eq (f 0) 0
                           one : Eq (f 1) 1
                           mul : ∀ (x y : G₀'), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                           inv : ∀ (x : G₀'), Eq (f (Inv.inv x)) (Inv.inv (f x))
                           div : ∀ (x y : G₀'), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                           npow : ∀ (x : G₀') (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                           zpow : ∀ (x : G₀') (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                           ⊢ Eq (f (Inv.inv 0)) (f 0)
                         -/
    inv_zero := hf <| by rw [inv, zero, inv_zero],
                         /-
                           🎉 no goals
                         -/
    mul_inv_cancel := fun x hx => hf <| by
      /-
        M₀ : Type u_1
        G₀ : Type u_2
        M₀' : Type u_3
        G₀' : Type u_4
        inst✝⁷ : GroupWithZero G₀
        inst✝⁶ : Zero G₀'
        inst✝⁵ : Mul G₀'
        inst✝⁴ : One G₀'
        inst✝³ : Inv G₀'
        inst✝² : Div G₀'
        inst✝¹ : Pow G₀' Nat
        inst✝ : Pow G₀' Int
        f : G₀' → G₀
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        mul : ∀ (x y : G₀'), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        inv : ∀ (x : G₀'), Eq (f (Inv.inv x)) (Inv.inv (f x))
        div : ∀ (x y : G₀'), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
        npow : ∀ (x : G₀') (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        zpow : ∀ (x : G₀') (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        x : G₀'
        hx : Ne x 0
        ⊢ Eq (f (HMul.hMul x (Inv.inv x))) (f 1)
      -/
      rw [one, mul, inv, mul_inv_cancel₀ ((hf.ne_iff' zero).2 hx)] }
      /-
        🎉 no goals
      -/


/-- Push forward a `GroupWithZero` along a surjective function.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.groupWithZero [Zero G₀'] [Mul G₀'] [One G₀'] [Inv G₀']
    [Div G₀'] [Pow G₀' ℕ] [Pow G₀' ℤ] (h01 : (0 : G₀') ≠ 1) (f : G₀ → G₀') (hf : Surjective f)
    (zero : f 0 = 0) (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y)
    (inv : ∀ x, f x⁻¹ = (f x)⁻¹) (div : ∀ x y, f (x / y) = f x / f y)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) :
    GroupWithZero G₀' :=
  { hf.monoidWithZero f zero one mul npow, hf.divInvMonoid f one mul inv div npow zpow with
                   /-
                     M₀ : Type u_1
                     G₀ : Type u_2
                     M₀' : Type u_3
                     G₀' : Type u_4
                     inst✝⁷ : GroupWithZero G₀
                     inst✝⁶ : Zero G₀'
                     inst✝⁵ : Mul G₀'
                     inst✝⁴ : One G₀'
                     inst✝³ : Inv G₀'
                     inst✝² : Div G₀'
                     inst✝¹ : Pow G₀' Nat
                     inst✝ : Pow G₀' Int
                     h01 : Ne 0 1
                     f : G₀ → G₀'
                     hf : Function.Surjective f
                     zero : Eq (f 0) 0
                     one : Eq (f 1) 1
                     mul : ∀ (x y : G₀), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                     inv : ∀ (x : G₀), Eq (f (Inv.inv x)) (Inv.inv (f x))
                     div : ∀ (x y : G₀), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                     npow : ∀ (x : G₀) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                     zpow : ∀ (x : G₀) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                     ⊢ Eq (Inv.inv 0) 0
                   -/
    inv_zero := by rw [← zero, ← inv, inv_zero],
                   /-
                     🎉 no goals
                   -/
    mul_inv_cancel := hf.forall.2 fun x hx => by
        /-
          M₀ : Type u_1
          G₀ : Type u_2
          M₀' : Type u_3
          G₀' : Type u_4
          inst✝⁷ : GroupWithZero G₀
          inst✝⁶ : Zero G₀'
          inst✝⁵ : Mul G₀'
          inst✝⁴ : One G₀'
          inst✝³ : Inv G₀'
          inst✝² : Div G₀'
          inst✝¹ : Pow G₀' Nat
          inst✝ : Pow G₀' Int
          h01 : Ne 0 1
          f : G₀ → G₀'
          hf : Function.Surjective f
          zero : Eq (f 0) 0
          one : Eq (f 1) 1
          mul : ∀ (x y : G₀), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
          inv : ∀ (x : G₀), Eq (f (Inv.inv x)) (Inv.inv (f x))
          div : ∀ (x y : G₀), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
          npow : ∀ (x : G₀) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
          zpow : ∀ (x : G₀) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
          x : G₀
          hx : Ne (f x) 0
          ⊢ Eq (HMul.hMul (f x) (Inv.inv (f x))) 1
        -/
        rw [← inv, ← mul, mul_inv_cancel₀ (mt (congr_arg f) fun h ↦ hx (h.trans zero)), one]
        /-
          🎉 no goals
        -/
    exists_pair_ne := ⟨0, 1, h01⟩ }


/-- Pull back a `CommGroupWithZero` along an injective function.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.commGroupWithZero [Zero G₀'] [Mul G₀'] [One G₀'] [Inv G₀']
    [Div G₀'] [Pow G₀' ℕ] [Pow G₀' ℤ] (f : G₀' → G₀) (hf : Injective f) (zero : f 0 = 0)
    (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : CommGroupWithZero G₀' :=
  { hf.groupWithZero f zero one mul inv div npow zpow, hf.commSemigroup f mul with }


/-- Push forward a `CommGroupWithZero` along a surjective function.
See note [reducible non-instances]. -/
protected def Function.Surjective.commGroupWithZero [Zero G₀'] [Mul G₀'] [One G₀'] [Inv G₀']
    [Div G₀'] [Pow G₀' ℕ] [Pow G₀' ℤ] (h01 : (0 : G₀') ≠ 1) (f : G₀ → G₀') (hf : Surjective f)
    (zero : f 0 = 0) (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y)
    (inv : ∀ x, f x⁻¹ = (f x)⁻¹) (div : ∀ x y, f (x / y) = f x / f y)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) :
    CommGroupWithZero G₀' :=
  { hf.groupWithZero h01 f zero one mul inv div npow zpow, hf.commSemigroup f mul with }


