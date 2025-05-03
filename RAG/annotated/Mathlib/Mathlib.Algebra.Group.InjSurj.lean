/-- A type endowed with `*` is a semigroup, if it admits an injective map that preserves `*` to
a semigroup. See note [reducible non-instances]. -/
@[to_additive "A type endowed with `+` is an additive semigroup, if it admits an
injective map that preserves `+` to an additive semigroup."]
protected abbrev semigroup [Semigroup M₂] (f : M₁ → M₂) (hf : Injective f)
    (mul : ∀ x y, f (x * y) = f x * f y) : Semigroup M₁ :=
                                                     /-
                                                       M₁ : Type u_1
                                                       M₂ : Type u_2
                                                       inst✝¹ : Mul M₁
                                                       inst✝ : Semigroup M₂
                                                       f : M₁ → M₂
                                                       hf : Function.Injective f
                                                       mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                       x y z : M₁
                                                       ⊢ Eq (f (HMul.hMul (HMul.hMul x y) z)) (f (HMul.hMul x (HMul.hMul y z)))
                                                     -/
  { ‹Mul M₁› with mul_assoc := fun x y z => hf <| by rw [mul, mul, mul, mul, mul_assoc] }
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- A type endowed with `*` is a commutative magma, if it admits a surjective map that preserves
`*` from a commutative magma. -/
@[to_additive -- See note [reducible non-instances]
"A type endowed with `+` is an additive commutative semigroup, if it admits
a surjective map that preserves `+` from an additive commutative semigroup."]
protected abbrev commMagma [CommMagma M₂] (f : M₁ → M₂) (hf : Injective f)
    (mul : ∀ x y, f (x * y) = f x * f y) : CommMagma M₁ where
                           /-
                             M₁ : Type u_1
                             M₂ : Type u_2
                             inst✝¹ : Mul M₁
                             inst✝ : CommMagma M₂
                             f : M₁ → M₂
                             hf : Function.Injective f
                             mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                             x y : M₁
                             ⊢ Eq (f (HMul.hMul x y)) (f (HMul.hMul y x))
                           -/
  mul_comm x y := hf <| by rw [mul, mul, mul_comm]
                           /-
                             🎉 no goals
                           -/


/-- A type endowed with `*` is a commutative semigroup, if it admits an injective map that
preserves `*` to a commutative semigroup.  See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `+` is an additive commutative semigroup,if it admits
an injective map that preserves `+` to an additive commutative semigroup."]
protected abbrev commSemigroup [CommSemigroup M₂] (f : M₁ → M₂) (hf : Injective f)
    (mul : ∀ x y, f (x * y) = f x * f y) : CommSemigroup M₁ where
  toSemigroup := hf.semigroup f mul
  __ := hf.commMagma f mul


/-- A type endowed with `*` is a left cancel semigroup, if it admits an injective map that
preserves `*` to a left cancel semigroup.  See note [reducible non-instances]. -/
@[to_additive "A type endowed with `+` is an additive left cancel
semigroup, if it admits an injective map that preserves `+` to an additive left cancel semigroup."]
protected abbrev leftCancelSemigroup [LeftCancelSemigroup M₂] (f : M₁ → M₂) (hf : Injective f)
    (mul : ∀ x y, f (x * y) = f x * f y) : LeftCancelSemigroup M₁ :=
  { hf.semigroup f mul with
                                                                          /-
                                                                            M₁ : Type u_1
                                                                            M₂ : Type u_2
                                                                            inst✝¹ : Mul M₁
                                                                            inst✝ : LeftCancelSemigroup M₂
                                                                            f : M₁ → M₂
                                                                            hf : Function.Injective f
                                                                            mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                                            x y z : M₁
                                                                            H : Eq (HMul.hMul x y) (HMul.hMul x z)
                                                                            ⊢ Eq (HMul.hMul (f x) (f y)) (HMul.hMul (f x) (f z))
                                                                          -/
    mul_left_cancel := fun x y z H => hf <| (mul_right_inj (f x)).1 <| by rw [← mul, ← mul, H] }
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- A type endowed with `*` is a right cancel semigroup, if it admits an injective map that
preserves `*` to a right cancel semigroup.  See note [reducible non-instances]. -/
@[to_additive "A type endowed with `+` is an additive right
cancel semigroup, if it admits an injective map that preserves `+` to an additive right cancel
semigroup."]
protected abbrev rightCancelSemigroup [RightCancelSemigroup M₂] (f : M₁ → M₂) (hf : Injective f)
    (mul : ∀ x y, f (x * y) = f x * f y) : RightCancelSemigroup M₁ :=
  { hf.semigroup f mul with
                                                                          /-
                                                                            M₁ : Type u_1
                                                                            M₂ : Type u_2
                                                                            inst✝¹ : Mul M₁
                                                                            inst✝ : RightCancelSemigroup M₂
                                                                            f : M₁ → M₂
                                                                            hf : Function.Injective f
                                                                            mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                                            x y z : M₁
                                                                            H : Eq (HMul.hMul x y) (HMul.hMul z y)
                                                                            ⊢ Eq (HMul.hMul (f x) (f y)) (HMul.hMul (f z) (f y))
                                                                          -/
    mul_right_cancel := fun x y z H => hf <| (mul_left_inj (f y)).1 <| by rw [← mul, ← mul, H] }
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- A type endowed with `1` and `*` is a `MulOneClass`, if it admits an injective map that
preserves `1` and `*` to a `MulOneClass`.  See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an `AddZeroClass`, if it admits an
injective map that preserves `0` and `+` to an `AddZeroClass`."]
protected abbrev mulOneClass [MulOneClass M₂] (f : M₁ → M₂) (hf : Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) : MulOneClass M₁ :=
  { ‹One M₁›, ‹Mul M₁› with
                                 /-
                                   M₁ : Type u_1
                                   M₂ : Type u_2
                                   inst✝² : Mul M₁
                                   inst✝¹ : One M₁
                                   inst✝ : MulOneClass M₂
                                   f : M₁ → M₂
                                   hf : Function.Injective f
                                   one : Eq (f 1) 1
                                   mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                   x : M₁
                                   ⊢ Eq (f (HMul.hMul 1 x)) (f x)
                                 -/
    one_mul := fun x => hf <| by rw [mul, one, one_mul],
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   M₁ : Type u_1
                                   M₂ : Type u_2
                                   inst✝² : Mul M₁
                                   inst✝¹ : One M₁
                                   inst✝ : MulOneClass M₂
                                   f : M₁ → M₂
                                   hf : Function.Injective f
                                   one : Eq (f 1) 1
                                   mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                   x : M₁
                                   ⊢ Eq (f (HMul.hMul x 1)) (f x)
                                 -/
    mul_one := fun x => hf <| by rw [mul, one, mul_one] }
                                 /-
                                   🎉 no goals
                                 -/


/-- A type endowed with `1` and `*` is a monoid, if it admits an injective map that preserves `1`
and `*` to a monoid.  See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive monoid, if it admits an
injective map that preserves `0` and `+` to an additive monoid. See note
[reducible non-instances]."]
protected abbrev monoid [Monoid M₂] (f : M₁ → M₂) (hf : Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) : Monoid M₁ :=
  { hf.mulOneClass f one mul, hf.semigroup f mul with
    npow := fun n x => x ^ n,
                                   /-
                                     M₁ : Type u_1
                                     M₂ : Type u_2
                                     inst✝³ : Mul M₁
                                     inst✝² : One M₁
                                     inst✝¹ : Pow M₁ Nat
                                     inst✝ : Monoid M₂
                                     f : M₁ → M₂
                                     hf : Function.Injective f
                                     one : Eq (f 1) 1
                                     mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                     npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                     x : M₁
                                     ⊢ Eq (f ((fun n x => HPow.hPow x n) 0 x)) (f 1)
                                   -/
    npow_zero := fun x => hf <| by rw [npow, one, pow_zero],
                                   /-
                                     🎉 no goals
                                   -/
                                     /-
                                       M₁ : Type u_1
                                       M₂ : Type u_2
                                       inst✝³ : Mul M₁
                                       inst✝² : One M₁
                                       inst✝¹ : Pow M₁ Nat
                                       inst✝ : Monoid M₂
                                       f : M₁ → M₂
                                       hf : Function.Injective f
                                       one : Eq (f 1) 1
                                       mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                       npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                       n : Nat
                                       x : M₁
                                       ⊢ Eq (f ((fun n x => HPow.hPow x n) (HAdd.hAdd n 1) x)) (f (HMul.hMul ((fun n  …
                                     -/
    npow_succ := fun n x => hf <| by rw [npow, pow_succ, mul, npow] }
                                     /-
                                       🎉 no goals
                                     -/


/-- A type endowed with `0`, `1` and `+` is an additive monoid with one,
if it admits an injective map that preserves `0`, `1` and `+` to an additive monoid with one.
See note [reducible non-instances]. -/
protected abbrev addMonoidWithOne {M₁} [Zero M₁] [One M₁] [Add M₁] [SMul ℕ M₁] [NatCast M₁]
    [AddMonoidWithOne M₂] (f : M₁ → M₂) (hf : Injective f) (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (natCast : ∀ n : ℕ, f n = n) : AddMonoidWithOne M₁ :=
  { hf.addMonoid f zero add (swap nsmul) with
    natCast := Nat.cast,
                           /-
                             M₁✝ : Type u_1
                             M₂ : Type u_2
                             inst✝⁸ : Mul M₁✝
                             inst✝⁷ : One M₁✝
                             inst✝⁶ : Pow M₁✝ Nat
                             M₁ : Type ?u.5903
                             inst✝⁵ : Zero M₁
                             inst✝⁴ : One M₁
                             inst✝³ : Add M₁
                             inst✝² : SMul Nat M₁
                             inst✝¹ : NatCast M₁
                             inst✝ : AddMonoidWithOne M₂
                             f : M₁ → M₂
                             hf : Function.Injective f
                             zero : Eq (f 0) 0
                             one : Eq (f 1) 1
                             add : ∀ (x y : M₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                             nsmul : ∀ (n : Nat) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                             natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                             ⊢ Eq (f (NatCast.natCast 0)) (f 0)
                           -/
    natCast_zero := hf (by rw [natCast, Nat.cast_zero, zero]),
                           /-
                             🎉 no goals
                           -/
                                    /-
                                      M₁✝ : Type u_1
                                      M₂ : Type u_2
                                      inst✝⁸ : Mul M₁✝
                                      inst✝⁷ : One M₁✝
                                      inst✝⁶ : Pow M₁✝ Nat
                                      M₁ : Type ?u.5903
                                      inst✝⁵ : Zero M₁
                                      inst✝⁴ : One M₁
                                      inst✝³ : Add M₁
                                      inst✝² : SMul Nat M₁
                                      inst✝¹ : NatCast M₁
                                      inst✝ : AddMonoidWithOne M₂
                                      f : M₁ → M₂
                                      hf : Function.Injective f
                                      zero : Eq (f 0) 0
                                      one : Eq (f 1) 1
                                      add : ∀ (x y : M₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                      nsmul : ∀ (n : Nat) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                      natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                      n : Nat
                                      ⊢ Eq (f (NatCast.natCast (HAdd.hAdd n 1))) (f (HAdd.hAdd (NatCast.natCast n) 1))
                                    -/
    natCast_succ := fun n => hf (by rw [natCast, Nat.cast_succ, add, one, natCast]), one := 1 }
                                    /-
                                      🎉 no goals
                                    -/


/-- A type endowed with `1` and `*` is a left cancel monoid, if it admits an injective map that
preserves `1` and `*` to a left cancel monoid. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive left cancel monoid, if it
admits an injective map that preserves `0` and `+` to an additive left cancel monoid."]
protected abbrev leftCancelMonoid [LeftCancelMonoid M₂] (f : M₁ → M₂) (hf : Injective f)
    (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) : LeftCancelMonoid M₁ :=
  { hf.leftCancelSemigroup f mul, hf.monoid f one mul npow with }


/-- A type endowed with `1` and `*` is a right cancel monoid, if it admits an injective map that
preserves `1` and `*` to a right cancel monoid. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive left cancel monoid,if it
admits an injective map that preserves `0` and `+` to an additive left cancel monoid."]
protected abbrev rightCancelMonoid [RightCancelMonoid M₂] (f : M₁ → M₂) (hf : Injective f)
    (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) : RightCancelMonoid M₁ :=
  { hf.rightCancelSemigroup f mul, hf.monoid f one mul npow with }


/-- A type endowed with `1` and `*` is a cancel monoid, if it admits an injective map that preserves
`1` and `*` to a cancel monoid. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive left cancel monoid,if it
admits an injective map that preserves `0` and `+` to an additive left cancel monoid."]
protected abbrev cancelMonoid [CancelMonoid M₂] (f : M₁ → M₂) (hf : Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) :
    CancelMonoid M₁ :=
  { hf.leftCancelMonoid f one mul npow, hf.rightCancelMonoid f one mul npow with }


/-- A type endowed with `1` and `*` is a commutative monoid, if it admits an injective map that
preserves `1` and `*` to a commutative monoid.  See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive commutative monoid, if it
admits an injective map that preserves `0` and `+` to an additive commutative monoid."]
protected abbrev commMonoid [CommMonoid M₂] (f : M₁ → M₂) (hf : Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) :
    CommMonoid M₁ :=
  { hf.monoid f one mul npow, hf.commSemigroup f mul with }


/-- A type endowed with `0`, `1` and `+` is an additive commutative monoid with one, if it admits an
injective map that preserves `0`, `1` and `+` to an additive commutative monoid with one.
See note [reducible non-instances]. -/
protected abbrev addCommMonoidWithOne {M₁} [Zero M₁] [One M₁] [Add M₁] [SMul ℕ M₁] [NatCast M₁]
    [AddCommMonoidWithOne M₂] (f : M₁ → M₂) (hf : Injective f) (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (natCast : ∀ n : ℕ, f n = n) : AddCommMonoidWithOne M₁ where
  __ := hf.addMonoidWithOne f zero one add nsmul natCast
  __ := hf.addCommMonoid _ zero add (swap nsmul)


/-- A type endowed with `1` and `*` is a cancel commutative monoid, if it admits an injective map
that preserves `1` and `*` to a cancel commutative monoid.  See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive cancel commutative monoid,
if it admits an injective map that preserves `0` and `+` to an additive cancel commutative monoid."]
protected abbrev cancelCommMonoid [CancelCommMonoid M₂] (f : M₁ → M₂) (hf : Injective f)
    (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) : CancelCommMonoid M₁ :=
  { hf.leftCancelSemigroup f mul, hf.commMonoid f one mul npow with }


/-- A type has an involutive inversion if it admits a surjective map that preserves `⁻¹` to a type
which has an involutive inversion. See note [reducible non-instances] -/
@[to_additive
"A type has an involutive negation if it admits a surjective map that
preserves `-` to a type which has an involutive negation."]
protected abbrev involutiveInv {M₁ : Type*} [Inv M₁] [InvolutiveInv M₂] (f : M₁ → M₂)
    (hf : Injective f) (inv : ∀ x, f x⁻¹ = (f x)⁻¹) : InvolutiveInv M₁ where
  inv := Inv.inv
                        /-
                          M₁✝ : Type u_1
                          M₂ : Type u_2
                          inst✝⁴ : Mul M₁✝
                          inst✝³ : One M₁✝
                          inst✝² : Pow M₁✝ Nat
                          M₁ : Type u_3
                          inst✝¹ : Inv M₁
                          inst✝ : InvolutiveInv M₂
                          f : M₁ → M₂
                          hf : Function.Injective f
                          inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                          x : M₁
                          ⊢ Eq (f (Inv.inv (Inv.inv x))) (f x)
                        -/
  inv_inv x := hf <| by rw [inv, inv, inv_inv]
                        /-
                          🎉 no goals
                        -/


/-- A type endowed with `1` and `⁻¹` is a `InvOneClass`, if it admits an injective map that
preserves `1` and `⁻¹` to a `InvOneClass`.  See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and unary `-` is an `NegZeroClass`, if it admits an
injective map that preserves `0` and unary `-` to an `NegZeroClass`."]
protected abbrev invOneClass [InvOneClass M₂] (f : M₁ → M₂) (hf : Injective f) (one : f 1 = 1)
    (inv : ∀ x, f (x⁻¹) = (f x)⁻¹) : InvOneClass M₁ :=
  { ‹One M₁›, ‹Inv M₁› with
                        /-
                          M₁ : Type u_1
                          M₂ : Type u_2
                          inst✝⁴ : Mul M₁
                          inst✝³ : One M₁
                          inst✝² : Pow M₁ Nat
                          inst✝¹ : Inv M₁
                          inst✝ : InvOneClass M₂
                          f : M₁ → M₂
                          hf : Function.Injective f
                          one : Eq (f 1) 1
                          inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                          ⊢ Eq (f (Inv.inv 1)) (f 1)
                        -/
    inv_one := hf <| by rw [inv, one, inv_one] }
                        /-
                          🎉 no goals
                        -/


/-- A type endowed with `1`, `*`, `⁻¹`, and `/` is a `DivInvMonoid` if it admits an injective map
that preserves `1`, `*`, `⁻¹`, and `/` to a `DivInvMonoid`. See note [reducible non-instances]. -/
@[to_additive subNegMonoid
"A type endowed with `0`, `+`, unary `-`, and binary `-` is a
`SubNegMonoid` if it admits an injective map that preserves `0`, `+`, unary `-`, and binary `-` to
a `SubNegMonoid`. This version takes custom `nsmul` and `zsmul` as `[SMul ℕ M₁]` and `[SMul ℤ M₁]`
arguments."]
protected abbrev divInvMonoid [DivInvMonoid M₂] (f : M₁ → M₂) (hf : Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : DivInvMonoid M₁ :=
  { hf.monoid f one mul npow, ‹Inv M₁›, ‹Div M₁› with
    zpow := fun n x => x ^ n,
                                    /-
                                      M₁ : Type u_1
                                      M₂ : Type u_2
                                      inst✝⁶ : Mul M₁
                                      inst✝⁵ : One M₁
                                      inst✝⁴ : Pow M₁ Nat
                                      inst✝³ : Inv M₁
                                      inst✝² : Div M₁
                                      inst✝¹ : Pow M₁ Int
                                      inst✝ : DivInvMonoid M₂
                                      f : M₁ → M₂
                                      hf : Function.Injective f
                                      one : Eq (f 1) 1
                                      mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                      inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                      div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                      npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                      zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                      x : M₁
                                      ⊢ Eq (f ((fun n x => HPow.hPow x n) 0 x)) (f 1)
                                    -/
    zpow_zero' := fun x => hf <| by rw [zpow, zpow_zero, one],
                                    /-
                                      🎉 no goals
                                    -/
                                          /-
                                            M₁ : Type u_1
                                            M₂ : Type u_2
                                            inst✝⁶ : Mul M₁
                                            inst✝⁵ : One M₁
                                            inst✝⁴ : Pow M₁ Nat
                                            inst✝³ : Inv M₁
                                            inst✝² : Div M₁
                                            inst✝¹ : Pow M₁ Int
                                            inst✝ : DivInvMonoid M₂
                                            f : M₁ → M₂
                                            hf : Function.Injective f
                                            one : Eq (f 1) 1
                                            mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                            inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                            div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                            npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                            zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                            x y : M₁
                                            ⊢ Eq (f (HDiv.hDiv x y)) (f (HMul.hMul x (Inv.inv y)))
                                          -/
                                      /-
                                        M₁ : Type u_1
                                        M₂ : Type u_2
                                        inst✝⁶ : Mul M₁
                                        inst✝⁵ : One M₁
                                        inst✝⁴ : Pow M₁ Nat
                                        inst✝³ : Inv M₁
                                        inst✝² : Div M₁
                                        inst✝¹ : Pow M₁ Int
                                        inst✝ : DivInvMonoid M₂
                                        f : M₁ → M₂
                                        hf : Function.Injective f
                                        one : Eq (f 1) 1
                                        mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                        inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                        div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                        npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                        zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                        n : Nat
                                        x : M₁
                                        ⊢ Eq (f ((fun n x => HPow.hPow x n) (↑n.succ) x)) (f (HMul.hMul ((fun n x => H …
                                      -/
                                          /-
                                            🎉 no goals
                                          -/
    zpow_succ' := fun n x => hf <| by rw [zpow, mul, zpow_natCast, pow_succ, zpow, zpow_natCast],
                                      /-
                                        🎉 no goals
                                      -/
                                     /-
                                       M₁ : Type u_1
                                       M₂ : Type u_2
                                       inst✝⁶ : Mul M₁
                                       inst✝⁵ : One M₁
                                       inst✝⁴ : Pow M₁ Nat
                                       inst✝³ : Inv M₁
                                       inst✝² : Div M₁
                                       inst✝¹ : Pow M₁ Int
                                       inst✝ : DivInvMonoid M₂
                                       f : M₁ → M₂
                                       hf : Function.Injective f
                                       one : Eq (f 1) 1
                                       mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                       inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                       div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                       npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                       zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                       n : Nat
                                       x : M₁
                                       ⊢ Eq (f ((fun n x => HPow.hPow x n) (Int.negSucc n) x)) (f (Inv.inv ((fun n x  …
                                     -/
    zpow_neg' := fun n x => hf <| by rw [zpow, zpow_negSucc, inv, zpow, zpow_natCast],
                                     /-
                                       🎉 no goals
                                     -/
    div_eq_mul_inv := fun x y => hf <| by rw [div, mul, inv, div_eq_mul_inv] }


/-- A type endowed with `1`, `*`, `⁻¹`, and `/` is a `DivInvOneMonoid` if it admits an injective
map that preserves `1`, `*`, `⁻¹`, and `/` to a `DivInvOneMonoid`. See note
[reducible non-instances]. -/
@[to_additive
"A type endowed with `0`, `+`, unary `-`, and binary `-` is a
`SubNegZeroMonoid` if it admits an injective map that preserves `0`, `+`, unary `-`, and binary
`-` to a `SubNegZeroMonoid`. This version takes custom `nsmul` and `zsmul` as `[SMul ℕ M₁]` and
`[SMul ℤ M₁]` arguments."]
protected abbrev divInvOneMonoid [DivInvOneMonoid M₂] (f : M₁ → M₂) (hf : Injective f)
    (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : DivInvOneMonoid M₁ :=
  { hf.divInvMonoid f one mul inv div npow zpow, hf.invOneClass f one inv with }


/-- A type endowed with `1`, `*`, `⁻¹`, and `/` is a `DivisionMonoid` if it admits an injective map
that preserves `1`, `*`, `⁻¹`, and `/` to a `DivisionMonoid`. See note [reducible non-instances] -/
@[to_additive
"A type endowed with `0`, `+`, unary `-`, and binary `-`
is a `SubtractionMonoid` if it admits an injective map that preserves `0`, `+`, unary `-`, and
binary `-` to a `SubtractionMonoid`. This version takes custom `nsmul` and `zsmul` as `[SMul ℕ M₁]`
and `[SMul ℤ M₁]` arguments."]
protected abbrev divisionMonoid [DivisionMonoid M₂] (f : M₁ → M₂) (hf : Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : DivisionMonoid M₁ :=
  { hf.divInvMonoid f one mul inv div npow zpow, hf.involutiveInv f inv with
                                       /-
                                         M₁ : Type u_1
                                         M₂ : Type u_2
                                         inst✝⁶ : Mul M₁
                                         inst✝⁵ : One M₁
                                         inst✝⁴ : Pow M₁ Nat
                                         inst✝³ : Inv M₁
                                         inst✝² : Div M₁
                                         inst✝¹ : Pow M₁ Int
                                         inst✝ : DivisionMonoid M₂
                                         f : M₁ → M₂
                                         hf : Function.Injective f
                                         one : Eq (f 1) 1
                                         mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                         inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                         div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                         npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                         zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                         x y : M₁
                                         ⊢ Eq (f (Inv.inv (HMul.hMul x y))) (f (HMul.hMul (Inv.inv y) (Inv.inv x)))
                                       -/
    mul_inv_rev := fun x y => hf <| by rw [inv, mul, mul_inv_rev, mul, inv, inv],
                                       /-
                                         🎉 no goals
                                       -/
    inv_eq_of_mul := fun x y h => hf <| by
      /-
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝⁶ : Mul M₁
        inst✝⁵ : One M₁
        inst✝⁴ : Pow M₁ Nat
        inst✝³ : Inv M₁
        inst✝² : Div M₁
        inst✝¹ : Pow M₁ Int
        inst✝ : DivisionMonoid M₂
        f : M₁ → M₂
        hf : Function.Injective f
        one : Eq (f 1) 1
        mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
        div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
        npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        x y : M₁
        h : Eq (HMul.hMul x y) 1
        ⊢ Eq (f (Inv.inv x)) (f y)
      -/
      rw [inv, inv_eq_of_mul_eq_one_right (by rw [← mul, h, one])] }
      /-
        🎉 no goals
      -/


/-- A type endowed with `1`, `*`, `⁻¹`, and `/` is a `DivisionCommMonoid` if it admits an
injective map that preserves `1`, `*`, `⁻¹`, and `/` to a `DivisionCommMonoid`.
See note [reducible non-instances]. -/
@[to_additive subtractionCommMonoid
"A type endowed with `0`, `+`, unary `-`, and binary
`-` is a `SubtractionCommMonoid` if it admits an injective map that preserves `0`, `+`, unary `-`,
and binary `-` to a `SubtractionCommMonoid`. This version takes custom `nsmul` and `zsmul` as
`[SMul ℕ M₁]` and `[SMul ℤ M₁]` arguments."]
protected abbrev divisionCommMonoid [DivisionCommMonoid M₂] (f : M₁ → M₂) (hf : Injective f)
    (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : DivisionCommMonoid M₁ :=
  { hf.divisionMonoid f one mul inv div npow zpow, hf.commSemigroup f mul with }


/-- A type endowed with `1`, `*` and `⁻¹` is a group, if it admits an injective map that preserves
`1`, `*` and `⁻¹` to a group. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive group, if it admits an
injective map that preserves `0` and `+` to an additive group."]
protected abbrev group [Group M₂] (f : M₁ → M₂) (hf : Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : Group M₁ :=
  { hf.divInvMonoid f one mul inv div npow zpow with
                                        /-
                                          M₁ : Type u_1
                                          M₂ : Type u_2
                                          inst✝⁶ : Mul M₁
                                          inst✝⁵ : One M₁
                                          inst✝⁴ : Pow M₁ Nat
                                          inst✝³ : Inv M₁
                                          inst✝² : Div M₁
                                          inst✝¹ : Pow M₁ Int
                                          inst✝ : Group M₂
                                          f : M₁ → M₂
                                          hf : Function.Injective f
                                          one : Eq (f 1) 1
                                          mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                          inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                          div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                          npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                          zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                          x : M₁
                                          ⊢ Eq (f (HMul.hMul (Inv.inv x) x)) (f 1)
                                        -/
    inv_mul_cancel := fun x => hf <| by rw [mul, inv, inv_mul_cancel, one] }
                                        /-
                                          🎉 no goals
                                        -/


/-- A type endowed with `0`, `1` and `+` is an additive group with one, if it admits an injective
map that preserves `0`, `1` and `+` to an additive group with one.  See note
[reducible non-instances]. -/
protected abbrev addGroupWithOne {M₁} [Zero M₁] [One M₁] [Add M₁] [SMul ℕ M₁] [Neg M₁] [Sub M₁]
    [SMul ℤ M₁] [NatCast M₁] [IntCast M₁] [AddGroupWithOne M₂] (f : M₁ → M₂) (hf : Injective f)
    (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) (natCast : ∀ n : ℕ, f n = n)
    (intCast : ∀ n : ℤ, f n = n) : AddGroupWithOne M₁ :=
  { hf.addGroup f zero add neg sub (swap nsmul) (swap zsmul),
    hf.addMonoidWithOne f zero one add nsmul natCast with
    intCast := Int.cast,
                                     /-
                                       M₁✝ : Type u_1
                                       M₂ : Type u_2
                                       inst✝¹⁵ : Mul M₁✝
                                       inst✝¹⁴ : One M₁✝
                                       inst✝¹³ : Pow M₁✝ Nat
                                       inst✝¹² : Inv M₁✝
                                       inst✝¹¹ : Div M₁✝
                                       inst✝¹⁰ : Pow M₁✝ Int
                                       M₁ : Type ?u.21328
                                       inst✝⁹ : Zero M₁
                                       inst✝⁸ : One M₁
                                       inst✝⁷ : Add M₁
                                       inst✝⁶ : SMul Nat M₁
                                       inst✝⁵ : Neg M₁
                                       inst✝⁴ : Sub M₁
                                       inst✝³ : SMul Int M₁
                                       inst✝² : NatCast M₁
                                       inst✝¹ : IntCast M₁
                                       inst✝ : AddGroupWithOne M₂
                                       f : M₁ → M₂
                                       hf : Function.Injective f
                                       zero : Eq (f 0) 0
                                       one : Eq (f 1) 1
                                       add : ∀ (x y : M₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                       neg : ∀ (x : M₁), Eq (f (Neg.neg x)) (Neg.neg (f x))
                                       sub : ∀ (x y : M₁), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
                                       nsmul : ∀ (n : Nat) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                       zsmul : ∀ (n : Int) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                       natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                       intCast : ∀ (n : Int), Eq (f ↑n) ↑n
                                       n : Nat
                                       ⊢ Eq (f (IntCast.intCast ↑n)) (f ↑n)
                                     -/
    intCast_ofNat := fun n => hf (by rw [natCast, ← Int.cast, intCast, Int.cast_natCast]),
                                     /-
                                       🎉 no goals
                                     -/
                                       /-
                                         M₁✝ : Type u_1
                                         M₂ : Type u_2
                                         inst✝¹⁵ : Mul M₁✝
                                         inst✝¹⁴ : One M₁✝
                                         inst✝¹³ : Pow M₁✝ Nat
                                         inst✝¹² : Inv M₁✝
                                         inst✝¹¹ : Div M₁✝
                                         inst✝¹⁰ : Pow M₁✝ Int
                                         M₁ : Type ?u.21328
                                         inst✝⁹ : Zero M₁
                                         inst✝⁸ : One M₁
                                         inst✝⁷ : Add M₁
                                         inst✝⁶ : SMul Nat M₁
                                         inst✝⁵ : Neg M₁
                                         inst✝⁴ : Sub M₁
                                         inst✝³ : SMul Int M₁
                                         inst✝² : NatCast M₁
                                         inst✝¹ : IntCast M₁
                                         inst✝ : AddGroupWithOne M₂
                                         f : M₁ → M₂
                                         hf : Function.Injective f
                                         zero : Eq (f 0) 0
                                         one : Eq (f 1) 1
                                         add : ∀ (x y : M₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                         neg : ∀ (x : M₁), Eq (f (Neg.neg x)) (Neg.neg (f x))
                                         sub : ∀ (x y : M₁), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
                                         nsmul : ∀ (n : Nat) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                         zsmul : ∀ (n : Int) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                         natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                         intCast : ∀ (n : Int), Eq (f ↑n) ↑n
                                         n : Nat
                                         ⊢ Eq (f (IntCast.intCast (Int.negSucc n))) (f (Neg.neg ↑(HAdd.hAdd n 1)))
                                       -/
    intCast_negSucc := fun n => hf (by rw [intCast, neg, natCast, Int.cast_negSucc] ) }
                                       /-
                                         🎉 no goals
                                       -/


/-- A type endowed with `1`, `*` and `⁻¹` is a commutative group, if it admits an injective map that
preserves `1`, `*` and `⁻¹` to a commutative group. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive commutative group, if it
admits an injective map that preserves `0` and `+` to an additive commutative group."]
protected abbrev commGroup [CommGroup M₂] (f : M₁ → M₂) (hf : Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : CommGroup M₁ :=
  { hf.commMonoid f one mul npow, hf.group f one mul inv div npow zpow with }


/-- A type endowed with `0`, `1` and `+` is an additive commutative group with one, if it admits an
injective map that preserves `0`, `1` and `+` to an additive commutative group with one.
See note [reducible non-instances]. -/
protected abbrev addCommGroupWithOne {M₁} [Zero M₁] [One M₁] [Add M₁] [SMul ℕ M₁] [Neg M₁] [Sub M₁]
    [SMul ℤ M₁] [NatCast M₁] [IntCast M₁] [AddCommGroupWithOne M₂] (f : M₁ → M₂) (hf : Injective f)
    (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) (natCast : ∀ n : ℕ, f n = n)
    (intCast : ∀ n : ℤ, f n = n) : AddCommGroupWithOne M₁ :=
  { hf.addGroupWithOne f zero one add neg sub nsmul zsmul natCast intCast,
    hf.addCommMonoid _ zero add (swap nsmul) with }


/-- A type endowed with `*` is a semigroup, if it admits a surjective map that preserves `*` from a
semigroup. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `+` is an additive semigroup, if it admits a
surjective map that preserves `+` from an additive semigroup."]
protected abbrev semigroup [Semigroup M₁] (f : M₁ → M₂) (hf : Surjective f)
    (mul : ∀ x y, f (x * y) = f x * f y) : Semigroup M₂ :=
                                                            /-
                                                              M₁ : Type u_1
                                                              M₂ : Type u_2
                                                              inst✝¹ : Mul M₂
                                                              inst✝ : Semigroup M₁
                                                              f : M₁ → M₂
                                                              hf : Function.Surjective f
                                                              mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                              x y z : M₁
                                                              ⊢ Eq (HMul.hMul (HMul.hMul (f x) (f y)) (f z)) (HMul.hMul (f x) (HMul.hMul (f  …
                                                            -/
  { ‹Mul M₂› with mul_assoc := hf.forall₃.2 fun x y z => by simp only [← mul, mul_assoc] }
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- A type endowed with `*` is a commutative semigroup, if it admits a surjective map that preserves
`*` from a commutative semigroup. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `+` is an additive commutative semigroup, if it admits
a surjective map that preserves `+` from an additive commutative semigroup."]
protected abbrev commMagma [CommMagma M₁] (f : M₁ → M₂) (hf : Surjective f)
    (mul : ∀ x y, f (x * y) = f x * f y) : CommMagma M₂ where
                                         /-
                                           M₁ : Type u_1
                                           M₂ : Type u_2
                                           inst✝¹ : Mul M₂
                                           inst✝ : CommMagma M₁
                                           f : M₁ → M₂
                                           hf : Function.Surjective f
                                           mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                           x y : M₁
                                           ⊢ Eq (HMul.hMul (f x) (f y)) (HMul.hMul (f y) (f x))
                                         -/
  mul_comm := hf.forall₂.2 fun x y => by rw [← mul, ← mul, mul_comm]
                                         /-
                                           🎉 no goals
                                         -/


/-- A type endowed with `*` is a commutative semigroup, if it admits a surjective map that preserves
`*` from a commutative semigroup. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `+` is an additive commutative semigroup, if it admits
a surjective map that preserves `+` from an additive commutative semigroup."]
protected abbrev commSemigroup [CommSemigroup M₁] (f : M₁ → M₂) (hf : Surjective f)
    (mul : ∀ x y, f (x * y) = f x * f y) : CommSemigroup M₂ where
  toSemigroup := hf.semigroup f mul
  __ := hf.commMagma f mul


/-- A type endowed with `1` and `*` is a `MulOneClass`, if it admits a surjective map that preserves
`1` and `*` from a `MulOneClass`. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an `AddZeroClass`, if it admits a
surjective map that preserves `0` and `+` to an `AddZeroClass`."]
protected abbrev mulOneClass [MulOneClass M₁] (f : M₁ → M₂) (hf : Surjective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) : MulOneClass M₂ :=
  { ‹One M₂›, ‹Mul M₂› with
                                       /-
                                         M₁ : Type u_1
                                         M₂ : Type u_2
                                         inst✝² : Mul M₂
                                         inst✝¹ : One M₂
                                         inst✝ : MulOneClass M₁
                                         f : M₁ → M₂
                                         hf : Function.Surjective f
                                         one : Eq (f 1) 1
                                         mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                         x : M₁
                                         ⊢ Eq (HMul.hMul 1 (f x)) (f x)
                                       -/
    one_mul := hf.forall.2 fun x => by rw [← one, ← mul, one_mul],
                                       /-
                                         🎉 no goals
                                       -/
                                       /-
                                         M₁ : Type u_1
                                         M₂ : Type u_2
                                         inst✝² : Mul M₂
                                         inst✝¹ : One M₂
                                         inst✝ : MulOneClass M₁
                                         f : M₁ → M₂
                                         hf : Function.Surjective f
                                         one : Eq (f 1) 1
                                         mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                         x : M₁
                                         ⊢ Eq (HMul.hMul (f x) 1) (f x)
                                       -/
    mul_one := hf.forall.2 fun x => by rw [← one, ← mul, mul_one] }
                                       /-
                                         🎉 no goals
                                       -/


/-- A type endowed with `1` and `*` is a monoid, if it admits a surjective map that preserves `1`
and `*` to a monoid. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive monoid, if it admits a
surjective map that preserves `0` and `+` to an additive monoid. This version takes a custom `nsmul`
as a `[SMul ℕ M₂]` argument."]
protected abbrev monoid [Monoid M₁] (f : M₁ → M₂) (hf : Surjective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) : Monoid M₂ :=
  { hf.semigroup f mul, hf.mulOneClass f one mul with
    npow := fun n x => x ^ n,
                                         /-
                                           M₁ : Type u_1
                                           M₂ : Type u_2
                                           inst✝³ : Mul M₂
                                           inst✝² : One M₂
                                           inst✝¹ : Pow M₂ Nat
                                           inst✝ : Monoid M₁
                                           f : M₁ → M₂
                                           hf : Function.Surjective f
                                           one : Eq (f 1) 1
                                           mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                           npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                           x : M₁
                                           ⊢ Eq ((fun n x => HPow.hPow x n) 0 (f x)) 1
                                         -/
    npow_zero := hf.forall.2 fun x => by dsimp only; rw [← npow, pow_zero, ← one],
                                                     /-
                                                       🎉 no goals
                                                     -/
    npow_succ := fun n => hf.forall.2 fun x => by
      /-
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝³ : Mul M₂
        inst✝² : One M₂
        inst✝¹ : Pow M₂ Nat
        inst✝ : Monoid M₁
        f : M₁ → M₂
        hf : Function.Surjective f
        one : Eq (f 1) 1
        mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        n : Nat
        x : M₁
        ⊢ Eq ((fun n x => HPow.hPow x n) (HAdd.hAdd n 1) (f x)) (HMul.hMul ((fun n x = …
      -/
      dsimp only
      /-
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝³ : Mul M₂
        inst✝² : One M₂
        inst✝¹ : Pow M₂ Nat
        inst✝ : Monoid M₁
        f : M₁ → M₂
        hf : Function.Surjective f
        one : Eq (f 1) 1
        mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        n : Nat
        x : M₁
        ⊢ Eq (HPow.hPow (f x) (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow (f x) n) (f x))
      -/
      rw [← npow, pow_succ, ← npow, ← mul] }
      /-
        🎉 no goals
      -/


/-- A type endowed with `0`, `1` and `+` is an additive monoid with one, if it admits a surjective
map that preserves `0`, `1` and `*` from an additive monoid with one. See note
[reducible non-instances]. -/
protected abbrev addMonoidWithOne {M₂} [Zero M₂] [One M₂] [Add M₂] [SMul ℕ M₂] [NatCast M₂]
    [AddMonoidWithOne M₁] (f : M₁ → M₂) (hf : Surjective f) (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (natCast : ∀ n : ℕ, f n = n) : AddMonoidWithOne M₂ :=
  { hf.addMonoid f zero add (swap nsmul) with
    natCast := Nat.cast,
                       /-
                         M₁ : Type u_1
                         M₂✝ : Type u_2
                         inst✝⁸ : Mul M₂✝
                         inst✝⁷ : One M₂✝
                         inst✝⁶ : Pow M₂✝ Nat
                         M₂ : Type ?u.30336
                         inst✝⁵ : Zero M₂
                         inst✝⁴ : One M₂
                         inst✝³ : Add M₂
                         inst✝² : SMul Nat M₂
                         inst✝¹ : NatCast M₂
                         inst✝ : AddMonoidWithOne M₁
                         f : M₁ → M₂
                         hf : Function.Surjective f
                         zero : Eq (f 0) 0
                         one : Eq (f 1) 1
                         add : ∀ (x y : M₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                         nsmul : ∀ (n : Nat) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                         natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                         ⊢ Eq (NatCast.natCast 0) 0
                       -/
    natCast_zero := by rw [← Nat.cast, ← natCast, Nat.cast_zero, zero]
                       /-
                         🎉 no goals
                       -/
                                /-
                                  M₁ : Type u_1
                                  M₂✝ : Type u_2
                                  inst✝⁸ : Mul M₂✝
                                  inst✝⁷ : One M₂✝
                                  inst✝⁶ : Pow M₂✝ Nat
                                  M₂ : Type ?u.30336
                                  inst✝⁵ : Zero M₂
                                  inst✝⁴ : One M₂
                                  inst✝³ : Add M₂
                                  inst✝² : SMul Nat M₂
                                  inst✝¹ : NatCast M₂
                                  inst✝ : AddMonoidWithOne M₁
                                  f : M₁ → M₂
                                  hf : Function.Surjective f
                                  zero : Eq (f 0) 0
                                  one : Eq (f 1) 1
                                  add : ∀ (x y : M₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                  nsmul : ∀ (n : Nat) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                  natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                  n : Nat
                                  ⊢ Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (NatCast.natCast n) 1)
                                -/
    natCast_succ := fun n => by rw [← Nat.cast, ← natCast, Nat.cast_succ, add, one, natCast]
                                /-
                                  🎉 no goals
                                -/
    one := 1 }


/-- A type endowed with `1` and `*` is a commutative monoid, if it admits a surjective map that
preserves `1` and `*` from a commutative monoid. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive commutative monoid, if it
admits a surjective map that preserves `0` and `+` to an additive commutative monoid."]
protected abbrev commMonoid [CommMonoid M₁] (f : M₁ → M₂) (hf : Surjective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) :
    CommMonoid M₂ :=
  { hf.commSemigroup f mul, hf.monoid f one mul npow with }


/-- A type endowed with `0`, `1` and `+` is an additive monoid with one,
if it admits a surjective map that preserves `0`, `1` and `*` from an additive monoid with one.
See note [reducible non-instances]. -/
protected abbrev addCommMonoidWithOne {M₂} [Zero M₂] [One M₂] [Add M₂] [SMul ℕ M₂] [NatCast M₂]
    [AddCommMonoidWithOne M₁] (f : M₁ → M₂) (hf : Surjective f) (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (natCast : ∀ n : ℕ, f n = n) : AddCommMonoidWithOne M₂ where
  __ := hf.addMonoidWithOne f zero one add nsmul natCast
  __ := hf.addCommMonoid _ zero add (swap nsmul)


/-- A type has an involutive inversion if it admits a surjective map that preserves `⁻¹` to a type
which has an involutive inversion. See note [reducible non-instances] -/
@[to_additive
"A type has an involutive negation if it admits a surjective map that
preserves `-` to a type which has an involutive negation."]
protected abbrev involutiveInv {M₂ : Type*} [Inv M₂] [InvolutiveInv M₁] (f : M₁ → M₂)
    (hf : Surjective f) (inv : ∀ x, f x⁻¹ = (f x)⁻¹) : InvolutiveInv M₂ where
  inv := Inv.inv
                                     /-
                                       M₁ : Type u_1
                                       M₂✝ : Type u_2
                                       inst✝⁴ : Mul M₂✝
                                       inst✝³ : One M₂✝
                                       inst✝² : Pow M₂✝ Nat
                                       M₂ : Type u_3
                                       inst✝¹ : Inv M₂
                                       inst✝ : InvolutiveInv M₁
                                       f : M₁ → M₂
                                       hf : Function.Surjective f
                                       inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                       x : M₁
                                       ⊢ Eq (Inv.inv (Inv.inv (f x))) (f x)
                                     -/
  inv_inv := hf.forall.2 fun x => by rw [← inv, ← inv, inv_inv]
                                     /-
                                       🎉 no goals
                                     -/


/-- A type endowed with `1`, `*`, `⁻¹`, and `/` is a `DivInvMonoid` if it admits a surjective map
that preserves `1`, `*`, `⁻¹`, and `/` to a `DivInvMonoid`. See note [reducible non-instances]. -/
@[to_additive subNegMonoid
"A type endowed with `0`, `+`, unary `-`, and binary `-` is a
`SubNegMonoid` if it admits a surjective map that preserves `0`, `+`, unary `-`, and binary `-` to
a `SubNegMonoid`."]
protected abbrev divInvMonoid [DivInvMonoid M₁] (f : M₁ → M₂) (hf : Surjective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : DivInvMonoid M₂ :=
  { hf.monoid f one mul npow, ‹Div M₂›, ‹Inv M₂› with
    zpow := fun n x => x ^ n,
                                          /-
                                            M₁ : Type u_1
                                            M₂ : Type u_2
                                            inst✝⁶ : Mul M₂
                                            inst✝⁵ : One M₂
                                            inst✝⁴ : Pow M₂ Nat
                                            inst✝³ : Inv M₂
                                            inst✝² : Div M₂
                                            inst✝¹ : Pow M₂ Int
                                            inst✝ : DivInvMonoid M₁
                                            f : M₁ → M₂
                                            hf : Function.Surjective f
                                            one : Eq (f 1) 1
                                            mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                            inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                            div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                            npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                            zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                            x : M₁
                                            ⊢ Eq ((fun n x => HPow.hPow x n) 0 (f x)) 1
                                          -/
    zpow_zero' := hf.forall.2 fun x => by dsimp only; rw [← zpow, zpow_zero, ← one],
                                                      /-
                                                        🎉 no goals
                                                      -/
    zpow_succ' := fun n => hf.forall.2 fun x => by
      /-
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝⁶ : Mul M₂
        inst✝⁵ : One M₂
        inst✝⁴ : Pow M₂ Nat
        inst✝³ : Inv M₂
        inst✝² : Div M₂
        inst✝¹ : Pow M₂ Int
        inst✝ : DivInvMonoid M₁
        f : M₁ → M₂
        hf : Function.Surjective f
        one : Eq (f 1) 1
        mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
        div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
        npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        n : Nat
        x : M₁
        ⊢ Eq ((fun n x => HPow.hPow x n) (↑n.succ) (f x)) (HMul.hMul ((fun n x => HPow …
      -/
      dsimp only
      /-
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝⁶ : Mul M₂
        inst✝⁵ : One M₂
        inst✝⁴ : Pow M₂ Nat
        inst✝³ : Inv M₂
        inst✝² : Div M₂
        inst✝¹ : Pow M₂ Int
        inst✝ : DivInvMonoid M₁
        f : M₁ → M₂
        hf : Function.Surjective f
        one : Eq (f 1) 1
        mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
        div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
        npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        n : Nat
        x : M₁
        ⊢ Eq (HPow.hPow (f x) ↑n.succ) (HMul.hMul (HPow.hPow (f x) ↑n) (f x))
      -/
                                                 /-
                                                   M₁ : Type u_1
                                                   M₂ : Type u_2
                                                   inst✝⁶ : Mul M₂
                                                   inst✝⁵ : One M₂
                                                   inst✝⁴ : Pow M₂ Nat
                                                   inst✝³ : Inv M₂
                                                   inst✝² : Div M₂
                                                   inst✝¹ : Pow M₂ Int
                                                   inst✝ : DivInvMonoid M₁
                                                   f : M₁ → M₂
                                                   hf : Function.Surjective f
                                                   one : Eq (f 1) 1
                                                   mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                   inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                                   div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                                   npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                                   zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                                   x y : M₁
                                                   ⊢ Eq (HDiv.hDiv (f x) (f y)) (HMul.hMul (f x) (Inv.inv (f y)))
                                                 -/
      rw [← zpow, ← zpow, zpow_natCast, zpow_natCast, pow_succ, ← mul],
                                                 /-
                                                   🎉 no goals
                                                 -/
      /-
        🎉 no goals
      -/
    zpow_neg' := fun n => hf.forall.2 fun x => by
      /-
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝⁶ : Mul M₂
        inst✝⁵ : One M₂
        inst✝⁴ : Pow M₂ Nat
        inst✝³ : Inv M₂
        inst✝² : Div M₂
        inst✝¹ : Pow M₂ Int
        inst✝ : DivInvMonoid M₁
        f : M₁ → M₂
        hf : Function.Surjective f
        one : Eq (f 1) 1
        mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
        div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
        npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        n : Nat
        x : M₁
        ⊢ Eq ((fun n x => HPow.hPow x n) (Int.negSucc n) (f x)) (Inv.inv ((fun n x =>  …
      -/
      dsimp only
      /-
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝⁶ : Mul M₂
        inst✝⁵ : One M₂
        inst✝⁴ : Pow M₂ Nat
        inst✝³ : Inv M₂
        inst✝² : Div M₂
        inst✝¹ : Pow M₂ Int
        inst✝ : DivInvMonoid M₁
        f : M₁ → M₂
        hf : Function.Surjective f
        one : Eq (f 1) 1
        mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
        div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
        npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        n : Nat
        x : M₁
        ⊢ Eq (HPow.hPow (f x) (Int.negSucc n)) (Inv.inv (HPow.hPow (f x) ↑n.succ))
      -/
      rw [← zpow, ← zpow, zpow_negSucc, zpow_natCast, inv],
      /-
        🎉 no goals
      -/
    div_eq_mul_inv := hf.forall₂.2 fun x y => by rw [← inv, ← mul, ← div, div_eq_mul_inv] }


/-- A type endowed with `1`, `*` and `⁻¹` is a group, if it admits a surjective map that preserves
`1`, `*` and `⁻¹` to a group. See note [reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive group, if it admits a
surjective map that preserves `0` and `+` to an additive group."]
protected abbrev group [Group M₁] (f : M₁ → M₂) (hf : Surjective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : Group M₂ :=
  { hf.divInvMonoid f one mul inv div npow zpow with
                                              /-
                                                M₁ : Type u_1
                                                M₂ : Type u_2
                                                inst✝⁶ : Mul M₂
                                                inst✝⁵ : One M₂
                                                inst✝⁴ : Pow M₂ Nat
                                                inst✝³ : Inv M₂
                                                inst✝² : Div M₂
                                                inst✝¹ : Pow M₂ Int
                                                inst✝ : Group M₁
                                                f : M₁ → M₂
                                                hf : Function.Surjective f
                                                one : Eq (f 1) 1
                                                mul : ∀ (x y : M₁), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                inv : ∀ (x : M₁), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                                div : ∀ (x y : M₁), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                                npow : ∀ (x : M₁) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                                zpow : ∀ (x : M₁) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                                x : M₁
                                                ⊢ Eq (HMul.hMul (Inv.inv (f x)) (f x)) 1
                                              -/
    inv_mul_cancel := hf.forall.2 fun x => by rw [← inv, ← mul, inv_mul_cancel, one] }
                                              /-
                                                🎉 no goals
                                              -/


/-- A type endowed with `0`, `1`, `+` is an additive group with one,
if it admits a surjective map that preserves `0`, `1`, and `+` to an additive group with one.
See note [reducible non-instances]. -/
protected abbrev addGroupWithOne {M₂} [Zero M₂] [One M₂] [Add M₂] [Neg M₂] [Sub M₂] [SMul ℕ M₂]
    [SMul ℤ M₂] [NatCast M₂] [IntCast M₂] [AddGroupWithOne M₁] (f : M₁ → M₂) (hf : Surjective f)
    (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) (natCast : ∀ n : ℕ, f n = n)
    (intCast : ∀ n : ℤ, f n = n) : AddGroupWithOne M₂ :=
  { hf.addMonoidWithOne f zero one add nsmul natCast,
    hf.addGroup f zero add neg sub (swap nsmul) (swap zsmul) with
    intCast := Int.cast,
                                 /-
                                   M₁ : Type u_1
                                   M₂✝ : Type u_2
                                   inst✝¹⁵ : Mul M₂✝
                                   inst✝¹⁴ : One M₂✝
                                   inst✝¹³ : Pow M₂✝ Nat
                                   inst✝¹² : Inv M₂✝
                                   inst✝¹¹ : Div M₂✝
                                   inst✝¹⁰ : Pow M₂✝ Int
                                   M₂ : Type ?u.36951
                                   inst✝⁹ : Zero M₂
                                   inst✝⁸ : One M₂
                                   inst✝⁷ : Add M₂
                                   inst✝⁶ : Neg M₂
                                   inst✝⁵ : Sub M₂
                                   inst✝⁴ : SMul Nat M₂
                                   inst✝³ : SMul Int M₂
                                   inst✝² : NatCast M₂
                                   inst✝¹ : IntCast M₂
                                   inst✝ : AddGroupWithOne M₁
                                   f : M₁ → M₂
                                   hf : Function.Surjective f
                                   zero : Eq (f 0) 0
                                   one : Eq (f 1) 1
                                   add : ∀ (x y : M₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                   neg : ∀ (x : M₁), Eq (f (Neg.neg x)) (Neg.neg (f x))
                                   sub : ∀ (x y : M₁), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
                                   nsmul : ∀ (n : Nat) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                   zsmul : ∀ (n : Int) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                   natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                   intCast : ∀ (n : Int), Eq (f ↑n) ↑n
                                   n : Nat
                                   ⊢ Eq (IntCast.intCast ↑n) ↑n
                                 -/
    intCast_ofNat := fun n => by rw [← Int.cast, ← intCast, Int.cast_natCast, natCast],
                                 /-
                                   🎉 no goals
                                 -/
    intCast_negSucc := fun n => by
      /-
        M₁ : Type u_1
        M₂✝ : Type u_2
        inst✝¹⁵ : Mul M₂✝
        inst✝¹⁴ : One M₂✝
        inst✝¹³ : Pow M₂✝ Nat
        inst✝¹² : Inv M₂✝
        inst✝¹¹ : Div M₂✝
        inst✝¹⁰ : Pow M₂✝ Int
        M₂ : Type ?u.36951
        inst✝⁹ : Zero M₂
        inst✝⁸ : One M₂
        inst✝⁷ : Add M₂
        inst✝⁶ : Neg M₂
        inst✝⁵ : Sub M₂
        inst✝⁴ : SMul Nat M₂
        inst✝³ : SMul Int M₂
        inst✝² : NatCast M₂
        inst✝¹ : IntCast M₂
        inst✝ : AddGroupWithOne M₁
        f : M₁ → M₂
        hf : Function.Surjective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : M₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        neg : ∀ (x : M₁), Eq (f (Neg.neg x)) (Neg.neg (f x))
        sub : ∀ (x y : M₁), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
        nsmul : ∀ (n : Nat) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        zsmul : ∀ (n : Int) (x : M₁), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        intCast : ∀ (n : Int), Eq (f ↑n) ↑n
        n : Nat
        ⊢ Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg ↑(HAdd.hAdd n 1))
      -/
      rw [← Int.cast, ← intCast, Int.cast_negSucc, neg, natCast] }
      /-
        🎉 no goals
      -/


/-- A type endowed with `1`, `*`, `⁻¹`, and `/` is a commutative group, if it admits a surjective
map that preserves `1`, `*`, `⁻¹`, and `/` from a commutative group. See note
[reducible non-instances]. -/
@[to_additive
"A type endowed with `0` and `+` is an additive commutative group, if it
admits a surjective map that preserves `0` and `+` to an additive commutative group."]
protected abbrev commGroup [CommGroup M₁] (f : M₁ → M₂) (hf : Surjective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n) : CommGroup M₂ :=
  { hf.commMonoid f one mul npow, hf.group f one mul inv div npow zpow with }


/-- A type endowed with `0`, `1`, `+` is an additive commutative group with one, if it admits a
surjective map that preserves `0`, `1`, and `+` to an additive commutative group with one.
See note [reducible non-instances]. -/
protected abbrev addCommGroupWithOne {M₂} [Zero M₂] [One M₂] [Add M₂] [Neg M₂] [Sub M₂] [SMul ℕ M₂]
    [SMul ℤ M₂] [NatCast M₂] [IntCast M₂] [AddCommGroupWithOne M₁] (f : M₁ → M₂) (hf : Surjective f)
    (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) (natCast : ∀ n : ℕ, f n = n)
    (intCast : ∀ n : ℤ, f n = n) : AddCommGroupWithOne M₂ :=
  { hf.addGroupWithOne f zero one add neg sub nsmul zsmul natCast intCast,
    hf.addCommMonoid _ zero add (swap nsmul) with }


