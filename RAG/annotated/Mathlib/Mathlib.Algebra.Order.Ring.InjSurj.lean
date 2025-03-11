/-- Pullback an `OrderedSemiring` under an injective map. -/
-- See note [reducible non-instances]
protected abbrev orderedSemiring [OrderedSemiring α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) : OrderedSemiring β where
  toSemiring := hf.semiring f zero one add mul nsmul npow natCast
  __ := hf.orderedAddCommMonoid f zero add (swap nsmul)
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     inst✝¹³ : Zero β
                                     inst✝¹² : One β
                                     inst✝¹¹ : Add β
                                     inst✝¹⁰ : Mul β
                                     inst✝⁹ : Neg β
                                     inst✝⁸ : Sub β
                                     inst✝⁷ : SMul Nat β
                                     inst✝⁶ : SMul Int β
                                     inst✝⁵ : Pow β Nat
                                     inst✝⁴ : NatCast β
                                     inst✝³ : IntCast β
                                     inst✝² : Max β
                                     inst✝¹ : Min β
                                     f : β → α
                                     hf : Function.Injective f
                                     inst✝ : OrderedSemiring α
                                     zero : Eq (f 0) 0
                                     one : Eq (f 1) 1
                                     add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                     mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                     nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                     npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                     natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                     ⊢ LE.le (f 0) (f 1)
                                   -/
  zero_le_one := show f 0 ≤ f 1 by simp only [zero, one, zero_le_one]
                                   /-
                                     🎉 no goals
                                   -/
  mul_le_mul_of_nonneg_left a b c h hc := show f (c * a) ≤ f (c * b) by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹³ : Zero β
      inst✝¹² : One β
      inst✝¹¹ : Add β
      inst✝¹⁰ : Mul β
      inst✝⁹ : Neg β
      inst✝⁸ : Sub β
      inst✝⁷ : SMul Nat β
      inst✝⁶ : SMul Int β
      inst✝⁵ : Pow β Nat
      inst✝⁴ : NatCast β
      inst✝³ : IntCast β
      inst✝² : Max β
      inst✝¹ : Min β
      f : β → α
      hf : Function.Injective f
      inst✝ : OrderedSemiring α
      zero : Eq (f 0) 0
      one : Eq (f 1) 1
      add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
      npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
      natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
      a b c : β
      h : LE.le a b
      hc : LE.le 0 c
      ⊢ LE.le (f (HMul.hMul c a)) (f (HMul.hMul c b))
    -/
    rw [mul, mul]; refine mul_le_mul_of_nonneg_left h ?_; rwa [← zero]
                                                          /-
                                                            🎉 no goals
                                                          -/
  mul_le_mul_of_nonneg_right a b c h hc := show f (a * c) ≤ f (b * c) by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹³ : Zero β
      inst✝¹² : One β
      inst✝¹¹ : Add β
      inst✝¹⁰ : Mul β
      inst✝⁹ : Neg β
      inst✝⁸ : Sub β
      inst✝⁷ : SMul Nat β
      inst✝⁶ : SMul Int β
      inst✝⁵ : Pow β Nat
      inst✝⁴ : NatCast β
      inst✝³ : IntCast β
      inst✝² : Max β
      inst✝¹ : Min β
      f : β → α
      hf : Function.Injective f
      inst✝ : OrderedSemiring α
      zero : Eq (f 0) 0
      one : Eq (f 1) 1
      add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
      npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
      natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
      a b c : β
      h : LE.le a b
      hc : LE.le 0 c
      ⊢ LE.le (f (HMul.hMul a c)) (f (HMul.hMul b c))
    -/
    rw [mul, mul]; refine mul_le_mul_of_nonneg_right h ?_; rwa [← zero]
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- Pullback an `OrderedCommSemiring` under an injective map. -/
-- See note [reducible non-instances]
protected abbrev orderedCommSemiring [OrderedCommSemiring α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) : OrderedCommSemiring β where
  toOrderedSemiring := hf.orderedSemiring f zero one add mul nsmul npow natCast
  __ := hf.commSemiring f zero one add mul nsmul npow natCast


/-- Pullback an `OrderedRing` under an injective map. -/
-- See note [reducible non-instances]
protected abbrev orderedRing [OrderedRing α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (intCast : ∀ n : ℤ, f n = n) : OrderedRing β where
  toRing := hf.ring f zero one add mul neg sub nsmul zsmul npow natCast intCast
  __ := hf.orderedAddCommGroup f zero add neg sub (swap nsmul) (swap zsmul)
  __ := hf.orderedSemiring f zero one add mul nsmul npow natCast
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    inst✝¹³ : Zero β
                                                    inst✝¹² : One β
                                                    inst✝¹¹ : Add β
                                                    inst✝¹⁰ : Mul β
                                                    inst✝⁹ : Neg β
                                                    inst✝⁸ : Sub β
                                                    inst✝⁷ : SMul Nat β
                                                    inst✝⁶ : SMul Int β
                                                    inst✝⁵ : Pow β Nat
                                                    inst✝⁴ : NatCast β
                                                    inst✝³ : IntCast β
                                                    inst✝² : Max β
                                                    inst✝¹ : Min β
                                                    f : β → α
                                                    hf : Function.Injective f
                                                    inst✝ : OrderedRing α
                                                    zero : Eq (f 0) 0
                                                    one : Eq (f 1) 1
                                                    add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                                    mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                    neg : ∀ (x : β), Eq (f (Neg.neg x)) (Neg.neg (f x))
                                                    sub : ∀ (x y : β), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
                                                    nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                                    zsmul : ∀ (n : Int) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                                    npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                                    natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                                    intCast : ∀ (n : Int), Eq (f ↑n) ↑n
                                                    a b : β
                                                    ha : LE.le 0 a
                                                    hb : LE.le 0 b
                                                    ⊢ LE.le (f 0) (f (HMul.hMul a b))
                                                  -/
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
  mul_nonneg a b ha hb := show f 0 ≤ f (a * b) by rw [zero, mul]; apply mul_nonneg <;> rwa [← zero]
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


/-- Pullback an `OrderedCommRing` under an injective map. -/
-- See note [reducible non-instances]
protected abbrev orderedCommRing [OrderedCommRing α]
    (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (intCast : ∀ n : ℤ, f n = n) : OrderedCommRing β where
  toOrderedRing := hf.orderedRing f zero one add mul neg sub nsmul zsmul npow natCast intCast
  __ := hf.commRing f zero one add mul neg sub nsmul zsmul npow natCast intCast


/-- Pullback a `StrictOrderedSemiring` under an injective map. -/
-- See note [reducible non-instances]
protected abbrev strictOrderedSemiring [StrictOrderedSemiring α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) : StrictOrderedSemiring β where
  toSemiring := hf.semiring f zero one add mul nsmul npow natCast
  __ := hf.orderedCancelAddCommMonoid f zero add (swap nsmul)
  __ := domain_nontrivial f zero one
  __ := hf.orderedSemiring f zero one add mul nsmul npow natCast
  mul_lt_mul_of_pos_left a b c h hc := show f (c * a) < f (c * b) by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹³ : Zero β
      inst✝¹² : One β
      inst✝¹¹ : Add β
      inst✝¹⁰ : Mul β
      inst✝⁹ : Neg β
      inst✝⁸ : Sub β
      inst✝⁷ : SMul Nat β
      inst✝⁶ : SMul Int β
      inst✝⁵ : Pow β Nat
      inst✝⁴ : NatCast β
      inst✝³ : IntCast β
      inst✝² : Max β
      inst✝¹ : Min β
      f : β → α
      hf : Function.Injective f
      inst✝ : StrictOrderedSemiring α
      zero : Eq (f 0) 0
      one : Eq (f 1) 1
      add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
      npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
      natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
      a b c : β
      h : LT.lt a b
      hc : LT.lt 0 c
      ⊢ LT.lt (f (HMul.hMul c a)) (f (HMul.hMul c b))
    -/
    simpa only [mul, zero] using mul_lt_mul_of_pos_left ‹f a < f b› (by rwa [← zero])
    /-
      🎉 no goals
    -/
  mul_lt_mul_of_pos_right a b c h hc := show f (a * c) < f (b * c) by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹³ : Zero β
      inst✝¹² : One β
      inst✝¹¹ : Add β
      inst✝¹⁰ : Mul β
      inst✝⁹ : Neg β
      inst✝⁸ : Sub β
      inst✝⁷ : SMul Nat β
      inst✝⁶ : SMul Int β
      inst✝⁵ : Pow β Nat
      inst✝⁴ : NatCast β
      inst✝³ : IntCast β
      inst✝² : Max β
      inst✝¹ : Min β
      f : β → α
      hf : Function.Injective f
      inst✝ : StrictOrderedSemiring α
      zero : Eq (f 0) 0
      one : Eq (f 1) 1
      add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
      npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
      natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
      a b c : β
      h : LT.lt a b
      hc : LT.lt 0 c
      ⊢ LT.lt (f (HMul.hMul a c)) (f (HMul.hMul b c))
    -/
    simpa only [mul, zero] using mul_lt_mul_of_pos_right ‹f a < f b› (by rwa [← zero])
    /-
      🎉 no goals
    -/


/-- Pullback a `strictOrderedCommSemiring` under an injective map. -/
-- See note [reducible non-instances]
protected abbrev strictOrderedCommSemiring [StrictOrderedCommSemiring α]
    (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) : StrictOrderedCommSemiring β where
  toStrictOrderedSemiring := hf.strictOrderedSemiring f zero one add mul nsmul npow natCast
  __ := hf.commSemiring f zero one add mul nsmul npow natCast


/-- Pullback a `StrictOrderedRing` under an injective map. -/
-- See note [reducible non-instances]
protected abbrev strictOrderedRing [StrictOrderedRing α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (intCast : ∀ n : ℤ, f n = n) : StrictOrderedRing β where
  toRing := hf.ring f zero one add mul neg sub nsmul zsmul npow natCast intCast
  __ := hf.orderedAddCommGroup f zero add neg sub (swap nsmul) (swap zsmul)
  __ := hf.strictOrderedSemiring f zero one add mul nsmul npow natCast
                                               /-
                                                 α : Type u_1
                                                 β : Type u_2
                                                 inst✝¹³ : Zero β
                                                 inst✝¹² : One β
                                                 inst✝¹¹ : Add β
                                                 inst✝¹⁰ : Mul β
                                                 inst✝⁹ : Neg β
                                                 inst✝⁸ : Sub β
                                                 inst✝⁷ : SMul Nat β
                                                 inst✝⁶ : SMul Int β
                                                 inst✝⁵ : Pow β Nat
                                                 inst✝⁴ : NatCast β
                                                 inst✝³ : IntCast β
                                                 inst✝² : Max β
                                                 inst✝¹ : Min β
                                                 f : β → α
                                                 hf : Function.Injective f
                                                 inst✝ : StrictOrderedRing α
                                                 zero : Eq (f 0) 0
                                                 one : Eq (f 1) 1
                                                 add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                                 mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                 neg : ∀ (x : β), Eq (f (Neg.neg x)) (Neg.neg (f x))
                                                 sub : ∀ (x y : β), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
                                                 nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                                 zsmul : ∀ (n : Int) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                                 npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                                 natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                                 intCast : ∀ (n : Int), Eq (f ↑n) ↑n
                                                 a b : β
                                                 ha : LT.lt 0 a
                                                 hb : LT.lt 0 b
                                                 ⊢ LT.lt (f 0) (f (HMul.hMul a b))
                                               -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  mul_pos a b ha hb := show f 0 < f (a * b) by rw [zero, mul]; apply mul_pos <;> rwa [← zero]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- Pullback a `StrictOrderedCommRing` under an injective map. -/
-- See note [reducible non-instances]
protected abbrev strictOrderedCommRing [StrictOrderedCommRing α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (neg : ∀ x, f (-x) = -f x)
    (sub : ∀ x y, f (x - y) = f x - f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (intCast : ∀ n : ℤ, f n = n) : StrictOrderedCommRing β where
  toStrictOrderedRing := hf.strictOrderedRing f zero one add mul neg sub nsmul zsmul npow natCast
    intCast
  __ := hf.commRing f zero one add mul neg sub nsmul zsmul npow natCast intCast


/-- Pullback a `LinearOrderedSemiring` under an injective map. -/
-- See note [reducible non-instances]
protected abbrev linearOrderedSemiring [LinearOrderedSemiring α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (natCast : ∀ n : ℕ, f n = n)
    (sup : ∀ x y, f (x ⊔ y) = max (f x) (f y)) (inf : ∀ x y, f (x ⊓ y) = min (f x) (f y)) :
    LinearOrderedSemiring β where
  toStrictOrderedSemiring := hf.strictOrderedSemiring f zero one add mul nsmul npow natCast
  __ := hf.linearOrderedAddCommMonoid f zero add (swap nsmul) sup inf


/-- Pullback a `LinearOrderedSemiring` under an injective map. -/
-- -- See note [reducible non-instances]
protected abbrev linearOrderedCommSemiring [LinearOrderedCommSemiring α]
    (zero : f 0 = 0) (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y)
    (mul : ∀ x y, f (x * y) = f x * f y) (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (natCast : ∀ n : ℕ, f n = n)
    (hsup : ∀ x y, f (x ⊔ y) = max (f x) (f y)) (hinf : ∀ x y, f (x ⊓ y) = min (f x) (f y)) :
    LinearOrderedCommSemiring β where
  toStrictOrderedCommSemiring := hf.strictOrderedCommSemiring f zero one add mul nsmul npow natCast
  __ := hf.linearOrderedSemiring f zero one add mul nsmul npow natCast hsup hinf


/-- Pullback a `LinearOrderedRing` under an injective map. -/
-- See note [reducible non-instances]
abbrev linearOrderedRing [LinearOrderedRing α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (intCast : ∀ n : ℤ, f n = n)
    (hsup : ∀ x y, f (x ⊔ y) = max (f x) (f y)) (hinf : ∀ x y, f (x ⊓ y) = min (f x) (f y)) :
    LinearOrderedRing β where
  toStrictOrderedRing := hf.strictOrderedRing f zero one add mul neg sub nsmul zsmul npow natCast
    intCast
  __ := LinearOrder.lift f hf hsup hinf


/-- Pullback a `LinearOrderedCommRing` under an injective map. -/
-- See note [reducible non-instances]
protected abbrev linearOrderedCommRing [LinearOrderedCommRing α] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (natCast : ∀ n : ℕ, f n = n)
    (intCast : ∀ n : ℤ, f n = n) (sup : ∀ x y, f (x ⊔ y) = max (f x) (f y))
    (inf : ∀ x y, f (x ⊓ y) = min (f x) (f y)) : LinearOrderedCommRing β where
  toLinearOrderedRing := hf.linearOrderedRing f zero one add mul neg sub nsmul zsmul npow natCast
    intCast sup inf
  __ := hf.commMonoid f one mul npow


