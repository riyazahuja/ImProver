/-- Pullback an `OrderedCommMonoid` under an injective map.
See note [reducible non-instances]. -/
@[to_additive "Pullback an `OrderedAddCommMonoid` under an injective map."]
abbrev Function.Injective.orderedCommMonoid [OrderedCommMonoid α] {β : Type*} [One β] [Mul β]
    [Pow β ℕ] (f : β → α) (hf : Function.Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) :
    OrderedCommMonoid β where
  toCommMonoid := hf.commMonoid f one mul npow
  toPartialOrder := PartialOrder.lift f hf
  mul_le_mul_left a b ab c := show f (c * a) ≤ f (c * b) by
    /-
      α : Type u
      β✝ : Type u_1
      inst✝³ : OrderedCommMonoid α
      β : Type u_2
      inst✝² : One β
      inst✝¹ : Mul β
      inst✝ : Pow β Nat
      f : β → α
      hf : Function.Injective f
      one : Eq (f 1) 1
      mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
      a b : β
      ab : LE.le a b
      c : β
      ⊢ LE.le (f (HMul.hMul c a)) (f (HMul.hMul c b))
    -/
    rw [mul, mul]; apply mul_le_mul_left'; exact ab
                                           /-
                                             🎉 no goals
                                           -/


/-- Pullback an `OrderedCancelCommMonoid` under an injective map.
See note [reducible non-instances]. -/
@[to_additive Function.Injective.orderedCancelAddCommMonoid
    "Pullback an `OrderedCancelAddCommMonoid` under an injective map."]
abbrev Function.Injective.orderedCancelCommMonoid [OrderedCancelCommMonoid α] [One β] [Mul β]
    [Pow β ℕ] (f : β → α) (hf : Injective f) (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) : OrderedCancelCommMonoid β where
  toOrderedCommMonoid := hf.orderedCommMonoid f one mul npow
  le_of_mul_le_mul_left a b c (bc : f (a * b) ≤ f (a * c)) :=
                                      /-
                                        α : Type u
                                        β : Type u_1
                                        inst✝³ : OrderedCancelCommMonoid α
                                        inst✝² : One β
                                        inst✝¹ : Mul β
                                        inst✝ : Pow β Nat
                                        f : β → α
                                        hf : Function.Injective f
                                        one : Eq (f 1) 1
                                        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                        a b c : β
                                        bc : LE.le (f (HMul.hMul a b)) (f (HMul.hMul a c))
                                        ⊢ LE.le (HMul.hMul (f a) (f b)) (HMul.hMul (f a) (f c))
                                      -/
    (mul_le_mul_iff_left (f a)).1 (by rwa [← mul, ← mul])
                                      /-
                                        🎉 no goals
                                      -/


/-- Pullback a `LinearOrderedCommMonoid` under an injective map.
See note [reducible non-instances]. -/
@[to_additive "Pullback an `OrderedAddCommMonoid` under an injective map."]
abbrev Function.Injective.linearOrderedCommMonoid [LinearOrderedCommMonoid α] {β : Type*} [One β]
    [Mul β] [Pow β ℕ] [Max β] [Min β] (f : β → α) (hf : Function.Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (sup : ∀ x y, f (x ⊔ y) = max (f x) (f y)) (inf : ∀ x y, f (x ⊓ y) = min (f x) (f y)) :
    LinearOrderedCommMonoid β where
  toOrderedCommMonoid := hf.orderedCommMonoid f one mul npow
  __ := LinearOrder.lift f hf sup inf


/-- Pullback a `LinearOrderedCancelCommMonoid` under an injective map.
See note [reducible non-instances]. -/
@[to_additive Function.Injective.linearOrderedCancelAddCommMonoid
    "Pullback a `LinearOrderedCancelAddCommMonoid` under an injective map."]
abbrev Function.Injective.linearOrderedCancelCommMonoid [LinearOrderedCancelCommMonoid α] [One β]
    [Mul β] [Pow β ℕ] [Max β] [Min β] (f : β → α) (hf : Injective f) (one : f 1 = 1)
    (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (hsup : ∀ x y, f (x ⊔ y) = max (f x) (f y)) (hinf : ∀ x y, f (x ⊓ y) = min (f x) (f y)) :
    LinearOrderedCancelCommMonoid β where
  toOrderedCancelCommMonoid := hf.orderedCancelCommMonoid f one mul npow
  __ := hf.linearOrderedCommMonoid f one mul npow hsup hinf

-- TODO find a better home for the next two constructions.

/-- The order embedding sending `b` to `a * b`, for some fixed `a`.
See also `OrderIso.mulLeft` when working in an ordered group. -/
@[to_additive (attr := simps!)
      "The order embedding sending `b` to `a + b`, for some fixed `a`.
       See also `OrderIso.addLeft` when working in an additive ordered group."]
def OrderEmbedding.mulLeft {α : Type*} [Mul α] [LinearOrder α]
    [MulLeftStrictMono α] (m : α) : α ↪o α :=
  OrderEmbedding.ofStrictMono (fun n => m * n) fun _ _ w => mul_lt_mul_left' w m


/-- The order embedding sending `b` to `b * a`, for some fixed `a`.
See also `OrderIso.mulRight` when working in an ordered group. -/
@[to_additive (attr := simps!)
      "The order embedding sending `b` to `b + a`, for some fixed `a`.
       See also `OrderIso.addRight` when working in an additive ordered group."]
def OrderEmbedding.mulRight {α : Type*} [Mul α] [LinearOrder α]
    [MulRightStrictMono α] (m : α) : α ↪o α :=
  OrderEmbedding.ofStrictMono (fun n => n * m) fun _ _ w => mul_lt_mul_right' w m

