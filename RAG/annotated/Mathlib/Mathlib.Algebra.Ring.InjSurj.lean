/-- Pullback a `LeftDistribClass` instance along an injective function. -/
theorem leftDistribClass [Mul α] [Add α] [LeftDistribClass α] (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) : LeftDistribClass β where
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   f : β → α
                                   hf : Function.Injective f
                                   inst✝⁴ : Add β
                                   inst✝³ : Mul β
                                   inst✝² : Mul α
                                   inst✝¹ : Add α
                                   inst✝ : LeftDistribClass α
                                   add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                   mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                   x y z : β
                                   ⊢ Eq (f (HMul.hMul x (HAdd.hAdd y z))) (f (HAdd.hAdd (HMul.hMul x y) (HMul.hMu …
                                 -/
  left_distrib x y z := hf <| by simp only [*, left_distrib]
                                 /-
                                   🎉 no goals
                                 -/


/-- Pullback a `RightDistribClass` instance along an injective function. -/
theorem rightDistribClass [Mul α] [Add α] [RightDistribClass α] (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) : RightDistribClass β where
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    f : β → α
                                    hf : Function.Injective f
                                    inst✝⁴ : Add β
                                    inst✝³ : Mul β
                                    inst✝² : Mul α
                                    inst✝¹ : Add α
                                    inst✝ : RightDistribClass α
                                    add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                    mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                    x y z : β
                                    ⊢ Eq (f (HMul.hMul (HAdd.hAdd x y) z)) (f (HAdd.hAdd (HMul.hMul x z) (HMul.hMu …
                                  -/
  right_distrib x y z := hf <| by simp only [*, right_distrib]
                                  /-
                                    🎉 no goals
                                  -/


/-- Pullback a `Distrib` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev distrib [Distrib α] (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) : Distrib β where
  __ := hf.leftDistribClass f add mul
  __ := hf.rightDistribClass f add mul


/-- A type endowed with `-` and `*` has distributive negation, if it admits an injective map that
preserves `-` and `*` to a type which has distributive negation. -/
-- -- See note [reducible non-instances]
protected abbrev hasDistribNeg (f : β → α) (hf : Injective f) [Mul α] [HasDistribNeg α]
    (neg : ∀ a, f (-a) = -f a)
    (mul : ∀ a b, f (a * b) = f a * f b) : HasDistribNeg β :=
  { hf.involutiveNeg _ neg, ‹Mul β› with
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     f✝ : β → α
                                     hf✝ : Function.Injective f✝
                                     inst✝¹² : Add β
                                     inst✝¹¹ : Mul β
                                     inst✝¹⁰ : Zero β
                                     inst✝⁹ : One β
                                     inst✝⁸ : Neg β
                                     inst✝⁷ : Sub β
                                     inst✝⁶ : SMul Nat β
                                     inst✝⁵ : SMul Int β
                                     inst✝⁴ : Pow β Nat
                                     inst✝³ : NatCast β
                                     inst✝² : IntCast β
                                     f : β → α
                                     hf : Function.Injective f
                                     inst✝¹ : Mul α
                                     inst✝ : HasDistribNeg α
                                     neg : ∀ (a : β), Eq (f (Neg.neg a)) (Neg.neg (f a))
                                     mul : ∀ (a b : β), Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                                     x y : β
                                     ⊢ Eq (f (HMul.hMul (Neg.neg x) y)) (f (Neg.neg (HMul.hMul x y)))
                                   -/
    neg_mul := fun x y => hf <| by rw [neg, mul, neg, neg_mul, mul],
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     f✝ : β → α
                                     hf✝ : Function.Injective f✝
                                     inst✝¹² : Add β
                                     inst✝¹¹ : Mul β
                                     inst✝¹⁰ : Zero β
                                     inst✝⁹ : One β
                                     inst✝⁸ : Neg β
                                     inst✝⁷ : Sub β
                                     inst✝⁶ : SMul Nat β
                                     inst✝⁵ : SMul Int β
                                     inst✝⁴ : Pow β Nat
                                     inst✝³ : NatCast β
                                     inst✝² : IntCast β
                                     f : β → α
                                     hf : Function.Injective f
                                     inst✝¹ : Mul α
                                     inst✝ : HasDistribNeg α
                                     neg : ∀ (a : β), Eq (f (Neg.neg a)) (Neg.neg (f a))
                                     mul : ∀ (a b : β), Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                                     x y : β
                                     ⊢ Eq (f (HMul.hMul x (Neg.neg y))) (f (Neg.neg (HMul.hMul x y)))
                                   -/
    mul_neg := fun x y => hf <| by rw [neg, mul, neg, mul_neg, mul] }
                                   /-
                                     🎉 no goals
                                   -/


/-- Pullback a `NonUnitalNonAssocSemiring` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalNonAssocSemiring [NonUnitalNonAssocSemiring α] (zero : f 0 = 0)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) : NonUnitalNonAssocSemiring β where
  toAddCommMonoid := hf.addCommMonoid f zero add (swap nsmul)
  __ := hf.distrib f add mul
  __ := hf.mulZeroClass f zero mul


/-- Pullback a `NonUnitalSemiring` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalSemiring [NonUnitalSemiring α]
    (zero : f 0 = 0) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) :
    NonUnitalSemiring β where
  toNonUnitalNonAssocSemiring := hf.nonUnitalNonAssocSemiring f zero add mul nsmul
  __ := hf.semigroupWithZero f zero mul


/-- Pullback a `NonAssocSemiring` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev nonAssocSemiring [NonAssocSemiring α]
    (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (natCast : ∀ n : ℕ, f n = n) : NonAssocSemiring β where
  toNonUnitalNonAssocSemiring := hf.nonUnitalNonAssocSemiring f zero add mul nsmul
  __ := hf.mulZeroOneClass f zero one mul
  __ := hf.addMonoidWithOne f zero one add nsmul natCast


/-- Pullback a `Semiring` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev semiring [Semiring α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) : Semiring β where
  toNonUnitalSemiring := hf.nonUnitalSemiring f zero add mul nsmul
  __ := hf.nonAssocSemiring f zero one add mul nsmul natCast
  __ := hf.monoidWithZero f zero one mul npow


/-- Pullback a `NonUnitalNonAssocRing` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalNonAssocRing [NonUnitalNonAssocRing α] (f : β → α)
    (hf : Injective f) (zero : f 0 = 0) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) : NonUnitalNonAssocRing β where
  toAddCommGroup := hf.addCommGroup f zero add neg sub (swap nsmul) (swap zsmul)
  __ := hf.nonUnitalNonAssocSemiring f zero add mul nsmul


/-- Pullback a `NonUnitalRing` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalRing [NonUnitalRing α]
    (zero : f 0 = 0) (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) :
    NonUnitalRing β where
  toNonUnitalNonAssocRing := hf.nonUnitalNonAssocRing f zero add mul neg sub nsmul zsmul
  __ := hf.nonUnitalSemiring f zero add mul nsmul


/-- Pullback a `NonAssocRing` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev nonAssocRing [NonAssocRing α]
    (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) (natCast : ∀ n : ℕ, f n = n)
    (intCast : ∀ n : ℤ, f n = n) : NonAssocRing β where
  toNonUnitalNonAssocRing := hf.nonUnitalNonAssocRing f zero add mul neg sub nsmul zsmul
  __ := hf.nonAssocSemiring f zero one add mul nsmul natCast
  __ := hf.addCommGroupWithOne f zero one add neg sub nsmul zsmul natCast intCast


/-- Pullback a `Ring` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev ring [Ring α] (zero : f 0 = 0)
    (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (natCast : ∀ n : ℕ, f n = n)
    (intCast : ∀ n : ℤ, f n = n) : Ring β where
  toSemiring := hf.semiring f zero one add mul nsmul npow natCast
  __ := hf.addGroupWithOne f zero one add neg sub nsmul zsmul natCast intCast
  __ := hf.addCommGroup f zero add neg sub (swap nsmul) (swap zsmul)


/-- Pullback a `NonUnitalNonAssocCommSemiring` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalNonAssocCommSemiring [NonUnitalNonAssocCommSemiring α]
    (zero : f 0 = 0) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) :
    NonUnitalNonAssocCommSemiring β where
  toNonUnitalNonAssocSemiring := hf.nonUnitalNonAssocSemiring f zero add mul nsmul
  __ := hf.commMagma f mul


/-- Pullback a `NonUnitalCommSemiring` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalCommSemiring [NonUnitalCommSemiring α] (f : β → α)
    (hf : Injective f) (zero : f 0 = 0) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) :
    NonUnitalCommSemiring β where
  toNonUnitalSemiring := hf.nonUnitalSemiring f zero add mul nsmul
  __ := hf.commSemigroup f mul

-- `NonAssocCommSemiring` currently doesn't exist


/-- Pullback a `CommSemiring` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev commSemiring [CommSemiring α]
    (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (natCast : ∀ n : ℕ, f n = n) :
    CommSemiring β where
  toSemiring := hf.semiring f zero one add mul nsmul npow natCast
  __ := hf.commSemigroup f mul


/-- Pullback a `NonUnitalNonAssocCommRing` instance along an injective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalNonAssocCommRing [NonUnitalNonAssocCommRing α] (f : β → α)
    (hf : Injective f) (zero : f 0 = 0) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) : NonUnitalNonAssocCommRing β where
  toNonUnitalNonAssocRing := hf.nonUnitalNonAssocRing f zero add mul neg sub nsmul zsmul
  __ := hf.nonUnitalNonAssocCommSemiring f zero add mul nsmul


/-- Pullback a `NonUnitalCommRing` instance along an injective function. -/
-- -- See note [reducible non-instances]
protected abbrev nonUnitalCommRing [NonUnitalCommRing α] (f : β → α)
    (hf : Injective f) (zero : f 0 = 0) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) : NonUnitalCommRing β where
  toNonUnitalRing := hf.nonUnitalRing f zero add mul neg sub nsmul zsmul
  __ := hf.nonUnitalNonAssocCommRing f zero add mul neg sub nsmul zsmul


/-- Pullback a `CommRing` instance along an injective function. -/
-- -- See note [reducible non-instances]
protected abbrev commRing [CommRing α]
    (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (intCast : ∀ n : ℤ, f n = n) : CommRing β where
  toRing := hf.ring f zero one add mul neg sub nsmul zsmul npow natCast intCast
  __ := hf.commMonoid f one mul npow


/-- Pushforward a `LeftDistribClass` instance along a surjective function. -/
theorem leftDistribClass [Mul α] [Add α] [LeftDistribClass α] (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) : LeftDistribClass β where
                                               /-
                                                 α : Type u_1
                                                 β : Type u_2
                                                 f : α → β
                                                 hf : Function.Surjective f
                                                 inst✝⁴ : Add β
                                                 inst✝³ : Mul β
                                                 inst✝² : Mul α
                                                 inst✝¹ : Add α
                                                 inst✝ : LeftDistribClass α
                                                 add : ∀ (x y : α), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                                 mul : ∀ (x y : α), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                 x y z : α
                                                 ⊢ Eq (HMul.hMul (f x) (HAdd.hAdd (f y) (f z))) (HAdd.hAdd (HMul.hMul (f x) (f  …
                                               -/
  left_distrib := hf.forall₃.2 fun x y z => by simp only [← add, ← mul, left_distrib]
                                               /-
                                                 🎉 no goals
                                               -/


/-- Pushforward a `RightDistribClass` instance along a surjective function. -/
theorem rightDistribClass [Mul α] [Add α] [RightDistribClass α] (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) : RightDistribClass β where
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  f : α → β
                                                  hf : Function.Surjective f
                                                  inst✝⁴ : Add β
                                                  inst✝³ : Mul β
                                                  inst✝² : Mul α
                                                  inst✝¹ : Add α
                                                  inst✝ : RightDistribClass α
                                                  add : ∀ (x y : α), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                                  mul : ∀ (x y : α), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                  x y z : α
                                                  ⊢ Eq (HMul.hMul (HAdd.hAdd (f x) (f y)) (f z)) (HAdd.hAdd (HMul.hMul (f x) (f  …
                                                -/
  right_distrib := hf.forall₃.2 fun x y z => by simp only [← add, ← mul, right_distrib]
                                                /-
                                                  🎉 no goals
                                                -/


/-- Pushforward a `Distrib` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev distrib [Distrib α] (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) : Distrib β where
  __ := hf.leftDistribClass f add mul
  __ := hf.rightDistribClass f add mul


/-- A type endowed with `-` and `*` has distributive negation, if it admits a surjective map that
preserves `-` and `*` from a type which has distributive negation. -/
-- See note [reducible non-instances]
protected abbrev hasDistribNeg [Mul α] [HasDistribNeg α]
    (neg : ∀ a, f (-a) = -f a) (mul : ∀ a b, f (a * b) = f a * f b) : HasDistribNeg β :=
  { hf.involutiveNeg _ neg, ‹Mul β› with
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            f : α → β
                                            hf : Function.Surjective f
                                            inst✝¹² : Add β
                                            inst✝¹¹ : Mul β
                                            inst✝¹⁰ : Zero β
                                            inst✝⁹ : One β
                                            inst✝⁸ : Neg β
                                            inst✝⁷ : Sub β
                                            inst✝⁶ : SMul Nat β
                                            inst✝⁵ : SMul Int β
                                            inst✝⁴ : Pow β Nat
                                            inst✝³ : NatCast β
                                            inst✝² : IntCast β
                                            inst✝¹ : Mul α
                                            inst✝ : HasDistribNeg α
                                            neg : ∀ (a : α), Eq (f (Neg.neg a)) (Neg.neg (f a))
                                            mul : ∀ (a b : α), Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                                            x y : α
                                            ⊢ Eq (HMul.hMul (Neg.neg (f x)) (f y)) (Neg.neg (HMul.hMul (f x) (f y)))
                                          -/
    neg_mul := hf.forall₂.2 fun x y => by rw [← neg, ← mul, neg_mul, neg, mul]
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            f : α → β
                                            hf : Function.Surjective f
                                            inst✝¹² : Add β
                                            inst✝¹¹ : Mul β
                                            inst✝¹⁰ : Zero β
                                            inst✝⁹ : One β
                                            inst✝⁸ : Neg β
                                            inst✝⁷ : Sub β
                                            inst✝⁶ : SMul Nat β
                                            inst✝⁵ : SMul Int β
                                            inst✝⁴ : Pow β Nat
                                            inst✝³ : NatCast β
                                            inst✝² : IntCast β
                                            inst✝¹ : Mul α
                                            inst✝ : HasDistribNeg α
                                            neg : ∀ (a : α), Eq (f (Neg.neg a)) (Neg.neg (f a))
                                            mul : ∀ (a b : α), Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                                            x y : α
                                            ⊢ Eq (HMul.hMul (f x) (Neg.neg (f y))) (Neg.neg (HMul.hMul (f x) (f y)))
                                          -/
    mul_neg := hf.forall₂.2 fun x y => by rw [← neg, ← mul, mul_neg, neg, mul] }
                                          /-
                                            🎉 no goals
                                          -/


/-- Pushforward a `NonUnitalNonAssocSemiring` instance along a surjective function.
See note [reducible non-instances]. -/
protected abbrev nonUnitalNonAssocSemiring [NonUnitalNonAssocSemiring α] (zero : f 0 = 0)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) : NonUnitalNonAssocSemiring β where
  toAddCommMonoid := hf.addCommMonoid f zero add (swap nsmul)
  __ := hf.distrib f add mul
  __ := hf.mulZeroClass f zero mul


/-- Pushforward a `NonUnitalSemiring` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalSemiring [NonUnitalSemiring α] (zero : f 0 = 0)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) : NonUnitalSemiring β where
  toNonUnitalNonAssocSemiring := hf.nonUnitalNonAssocSemiring f zero add mul nsmul
  __ := hf.semigroupWithZero f zero mul


/-- Pushforward a `NonAssocSemiring` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev nonAssocSemiring [NonAssocSemiring α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (natCast : ∀ n : ℕ, f n = n) : NonAssocSemiring β where
  toNonUnitalNonAssocSemiring := hf.nonUnitalNonAssocSemiring f zero add mul nsmul
  __ := hf.mulZeroOneClass f zero one mul
  __ := hf.addMonoidWithOne f zero one add nsmul natCast


/-- Pushforward a `Semiring` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev semiring [Semiring α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (natCast : ∀ n : ℕ, f n = n) : Semiring β where
  toNonUnitalSemiring := hf.nonUnitalSemiring f zero add mul nsmul
  __ := hf.nonAssocSemiring f zero one add mul nsmul natCast
  __ := hf.monoidWithZero f zero one mul npow


/-- Pushforward a `NonUnitalNonAssocRing` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalNonAssocRing [NonUnitalNonAssocRing α] (zero : f 0 = 0)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) :
    NonUnitalNonAssocRing β where
  toAddCommGroup := hf.addCommGroup f zero add neg sub (swap nsmul) (swap zsmul)
  __ := hf.nonUnitalNonAssocSemiring f zero add mul nsmul


/-- Pushforward a `NonUnitalRing` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalRing [NonUnitalRing α] (zero : f 0 = 0)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) :
    NonUnitalRing β where
  toNonUnitalNonAssocRing := hf.nonUnitalNonAssocRing f zero add mul neg sub nsmul zsmul
  __ := hf.nonUnitalSemiring f zero add mul nsmul


/-- Pushforward a `NonAssocRing` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev nonAssocRing [NonAssocRing α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x)
    (natCast : ∀ n : ℕ, f n = n) (intCast : ∀ n : ℤ, f n = n) : NonAssocRing β where
  toNonUnitalNonAssocRing := hf.nonUnitalNonAssocRing f zero add mul neg sub nsmul zsmul
  __ := hf.nonAssocSemiring f zero one add mul nsmul natCast
  __ := hf.addCommGroupWithOne f zero one add neg sub nsmul zsmul natCast intCast


/-- Pushforward a `Ring` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev ring [Ring α] (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (natCast : ∀ n : ℕ, f n = n)
    (intCast : ∀ n : ℤ, f n = n) : Ring β where
  toSemiring := hf.semiring f zero one add mul nsmul npow natCast
  __ := hf.addGroupWithOne f zero one add neg sub nsmul zsmul natCast intCast
  __ := hf.addCommGroup f zero add neg sub (swap nsmul) (swap zsmul)


/-- Pushforward a `NonUnitalNonAssocCommSemiring` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalNonAssocCommSemiring  [NonUnitalNonAssocCommSemiring α] (zero : f 0 = 0)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) : NonUnitalNonAssocCommSemiring β where
  toNonUnitalNonAssocSemiring := hf.nonUnitalNonAssocSemiring f zero add mul nsmul
  __ := hf.commMagma f mul


/-- Pushforward a `NonUnitalCommSemiring` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalCommSemiring [NonUnitalCommSemiring α] (zero : f 0 = 0)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) : NonUnitalCommSemiring β where
  toNonUnitalSemiring := hf.nonUnitalSemiring f zero add mul nsmul
  __ := hf.commSemigroup f mul


/-- Pushforward a `CommSemiring` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev commSemiring [CommSemiring α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) : CommSemiring β where
  toSemiring := hf.semiring f zero one add mul nsmul npow natCast
  __ := hf.commSemigroup f mul


/-- Pushforward a `NonUnitalNonAssocCommRing` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalNonAssocCommRing [NonUnitalNonAssocCommRing α]
    (zero : f 0 = 0) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) : NonUnitalNonAssocCommRing β where
  toNonUnitalNonAssocRing := hf.nonUnitalNonAssocRing f zero add mul neg sub nsmul zsmul
  __ := hf.nonUnitalNonAssocCommSemiring f zero add mul nsmul


/-- Pushforward a `NonUnitalCommRing` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev nonUnitalCommRing [NonUnitalCommRing α] (zero : f 0 = 0)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) :
    NonUnitalCommRing β where
  toNonUnitalRing := hf.nonUnitalRing f zero add mul neg sub nsmul zsmul
  __ := hf.nonUnitalNonAssocCommRing f zero add mul neg sub nsmul zsmul


/-- Pushforward a `CommRing` instance along a surjective function. -/
-- See note [reducible non-instances]
protected abbrev commRing [CommRing α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (intCast : ∀ n : ℤ, f n = n) : CommRing β where
  toRing := hf.ring f zero one add mul neg sub nsmul zsmul npow natCast intCast
  __ := hf.commMonoid f one mul npow


instance AddOpposite.instHasDistribNeg : HasDistribNeg αᵃᵒᵖ :=
  unop_injective.hasDistribNeg _ unop_neg unop_mul

