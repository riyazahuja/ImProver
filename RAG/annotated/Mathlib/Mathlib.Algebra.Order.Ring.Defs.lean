/-- An `OrderedSemiring` is a semiring with a partial order such that addition is monotone and
multiplication by a nonnegative number is monotone. -/
class OrderedSemiring (α : Type u) extends Semiring α, OrderedAddCommMonoid α where
  /-- `0 ≤ 1` in any ordered semiring. -/
  protected zero_le_one : (0 : α) ≤ 1
  /-- In an ordered semiring, we can multiply an inequality `a ≤ b` on the left
  by a non-negative element `0 ≤ c` to obtain `c * a ≤ c * b`. -/
  protected mul_le_mul_of_nonneg_left : ∀ a b c : α, a ≤ b → 0 ≤ c → c * a ≤ c * b
  /-- In an ordered semiring, we can multiply an inequality `a ≤ b` on the right
  by a non-negative element `0 ≤ c` to obtain `a * c ≤ b * c`. -/
  protected mul_le_mul_of_nonneg_right : ∀ a b c : α, a ≤ b → 0 ≤ c → a * c ≤ b * c


/-- An `OrderedCommSemiring` is a commutative semiring with a partial order such that addition is
monotone and multiplication by a nonnegative number is monotone. -/
class OrderedCommSemiring (α : Type u) extends OrderedSemiring α, CommSemiring α where
  mul_le_mul_of_nonneg_right a b c ha hc :=
    -- parentheses ensure this generates an `optParam` rather than an `autoParam`
        /-
          α✝ α : Type u
          toOrderedSemiring : OrderedSemiring α
          toSemiring : Semiring α := OrderedSemiring.toSemiring
          toNonUnitalSemiring : NonUnitalSemiring α := Semiring.toNonUnitalSemiring
          toNonUnitalNonAssocSemiring : NonUnitalNonAssocSemiring α := NonUnitalSemiring …
          toAddCommMonoid : AddCommMonoid α := NonUnitalNonAssocSemiring.toAddCommMonoid
          toAddMonoid : AddMonoid α := AddCommMonoid.toAddMonoid
          toAddSemigroup : AddSemigroup α := AddMonoid.toAddSemigroup
          toAdd : Add α := AddSemigroup.toAdd
          add : α → α → α := Add.add
          add_assoc : ∀ (a b c : α), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd …
          toZero : Zero α := AddMonoid.toZero
          zero : α := Zero.zero
          zero_add : ∀ (a : α), Eq (HAdd.hAdd 0 a) a := AddMonoid.zero_add
          add_zero : ∀ (a : α), Eq (HAdd.hAdd a 0) a := AddMonoid.add_zero
          nsmul : Nat → α → α := AddMonoid.nsmul
          nsmul_zero : ∀ (x : α), Eq (AddMonoid.nsmul 0 x) 0 := AddMonoid.nsmul_zero
          nsmul_succ : ∀ (n : Nat) (x : α), Eq (AddMonoid.nsmul (HAdd.hAdd n 1) x) (HAdd …
          add_comm : ∀ (a b : α), Eq (HAdd.hAdd a b) (HAdd.hAdd b a) := AddCommMonoid.ad …
          toMul : Mul α := NonUnitalNonAssocSemiring.toMul
          mul : α → α → α := Mul.mul
          left_distrib : ∀ (a b c : α), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMu …
          right_distrib : ∀ (a b c : α), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HM …
          zero_mul : ∀ (a : α), Eq (HMul.hMul 0 a) 0 := NonUnitalNonAssocSemiring.zero_mul
          mul_zero : ∀ (a : α), Eq (HMul.hMul a 0) 0 := NonUnitalNonAssocSemiring.mul_zero
          mul_assoc : ∀ (a b c : α), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul …
          toOne : One α := Semiring.toOne
          one : α := One.one
          one_mul : ∀ (a : α), Eq (HMul.hMul 1 a) a := Semiring.one_mul
          mul_one : ∀ (a : α), Eq (HMul.hMul a 1) a := Semiring.mul_one
          toNatCast : NatCast α := Semiring.toNatCast
          natCast : Nat → α := NatCast.natCast
          natCast_zero : Eq (NatCast.natCast 0) 0 := Semiring.natCast_zero
          natCast_succ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (N …
          npow : Nat → α → α := Semiring.npow
          npow_zero : ∀ (x : α), Eq (Semiring.npow 0 x) 1 := Semiring.npow_zero
          npow_succ : ∀ (n : Nat) (x : α), Eq (Semiring.npow (HAdd.hAdd n 1) x) (HMul.hM …
          toPartialOrder : PartialOrder α := OrderedSemiring.toPartialOrder
          toPreorder : Preorder α := PartialOrder.toPreorder
          toLE : LE α := Preorder.toLE
          le : α → α → Prop := LE.le
          toLT : LT α := Preorder.toLT
          lt : α → α → Prop := LT.lt
          le_refl : ∀ (a : α), LE.le a a := Preorder.le_refl
          le_trans : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c := Preorder.le_trans
          lt_iff_le_not_le : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b …
          le_antisymm : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b := PartialOrder.le_a …
          add_le_add_left : ∀ (a b : α), LE.le a b → ∀ (c : α), LE.le (HAdd.hAdd c a) (H …
          zero_le_one : LE.le 0 1 := OrderedSemiring.zero_le_one
          mul_le_mul_of_nonneg_left : ∀ (a b c : α), LE.le a b → LE.le 0 c → LE.le (HMul …
          mul_le_mul_of_nonneg_right : ∀ (a b c : α), LE.le a b → LE.le 0 c → LE.le (HMu …
          mul_comm : ∀ (a b : α), Eq (HMul.hMul a b) (HMul.hMul b a)
          a b c : α
          ha : LE.le a b
          hc : LE.le 0 c
          ⊢ LE.le (HMul.hMul a c) (HMul.hMul b c)
        -/
    (by simpa only [mul_comm] using mul_le_mul_of_nonneg_left a b c ha hc)
        /-
          🎉 no goals
        -/


/-- An `OrderedRing` is a ring with a partial order such that addition is monotone and
multiplication by a nonnegative number is monotone. -/
class OrderedRing (α : Type u) extends Ring α, OrderedAddCommGroup α where
  /-- `0 ≤ 1` in any ordered ring. -/
  protected zero_le_one : 0 ≤ (1 : α)
  /-- The product of non-negative elements is non-negative. -/
  protected mul_nonneg : ∀ a b : α, 0 ≤ a → 0 ≤ b → 0 ≤ a * b


/-- An `OrderedCommRing` is a commutative ring with a partial order such that addition is monotone
and multiplication by a nonnegative number is monotone. -/
class OrderedCommRing (α : Type u) extends OrderedRing α, CommRing α


/-- A `StrictOrderedSemiring` is a nontrivial semiring with a partial order such that addition is
strictly monotone and multiplication by a positive number is strictly monotone. -/
class StrictOrderedSemiring (α : Type u) extends Semiring α, OrderedCancelAddCommMonoid α,
    Nontrivial α where
  /-- In a strict ordered semiring, `0 ≤ 1`. -/
  protected zero_le_one : (0 : α) ≤ 1
  /-- Left multiplication by a positive element is strictly monotone. -/
  protected mul_lt_mul_of_pos_left : ∀ a b c : α, a < b → 0 < c → c * a < c * b
  /-- Right multiplication by a positive element is strictly monotone. -/
  protected mul_lt_mul_of_pos_right : ∀ a b c : α, a < b → 0 < c → a * c < b * c


/-- A `StrictOrderedCommSemiring` is a commutative semiring with a partial order such that
addition is strictly monotone and multiplication by a positive number is strictly monotone. -/
class StrictOrderedCommSemiring (α : Type u) extends StrictOrderedSemiring α, CommSemiring α


/-- A `StrictOrderedRing` is a ring with a partial order such that addition is strictly monotone
and multiplication by a positive number is strictly monotone. -/
class StrictOrderedRing (α : Type u) extends Ring α, OrderedAddCommGroup α, Nontrivial α where
  /-- In a strict ordered ring, `0 ≤ 1`. -/
  protected zero_le_one : 0 ≤ (1 : α)
  /-- The product of two positive elements is positive. -/
  protected mul_pos : ∀ a b : α, 0 < a → 0 < b → 0 < a * b


/-- A `StrictOrderedCommRing` is a commutative ring with a partial order such that addition is
strictly monotone and multiplication by a positive number is strictly monotone. -/
class StrictOrderedCommRing (α : Type*) extends StrictOrderedRing α, CommRing α

/- It's not entirely clear we should assume `Nontrivial` at this point; it would be reasonable to
explore changing this, but be warned that the instances involving `Domain` may cause typeclass
search loops. -/

/-- A `LinearOrderedSemiring` is a nontrivial semiring with a linear order such that
addition is monotone and multiplication by a positive number is strictly monotone. -/
class LinearOrderedSemiring (α : Type u) extends StrictOrderedSemiring α,
  LinearOrderedAddCommMonoid α


/-- A `LinearOrderedCommSemiring` is a nontrivial commutative semiring with a linear order such
that addition is monotone and multiplication by a positive number is strictly monotone. -/
class LinearOrderedCommSemiring (α : Type*) extends StrictOrderedCommSemiring α,
  LinearOrderedSemiring α


/-- A `LinearOrderedRing` is a ring with a linear order such that addition is monotone and
multiplication by a positive number is strictly monotone. -/
class LinearOrderedRing (α : Type u) extends StrictOrderedRing α, LinearOrder α


/-- A `LinearOrderedCommRing` is a commutative ring with a linear order such that addition is
monotone and multiplication by a positive number is strictly monotone. -/
class LinearOrderedCommRing (α : Type u) extends LinearOrderedRing α, CommMonoid α


instance (priority := 100) OrderedSemiring.zeroLEOneClass : ZeroLEOneClass α :=
  { ‹OrderedSemiring α› with }

-- see Note [lower instance priority]

instance (priority := 200) OrderedSemiring.toPosMulMono : PosMulMono α :=
  ⟨fun x _ _ h => OrderedSemiring.mul_le_mul_of_nonneg_left _ _ _ h x.2⟩

-- see Note [lower instance priority]

instance (priority := 200) OrderedSemiring.toMulPosMono : MulPosMono α :=
  ⟨fun x _ _ h => OrderedSemiring.mul_le_mul_of_nonneg_right _ _ _ h x.2⟩


instance (priority := 100) OrderedRing.toOrderedSemiring : OrderedSemiring α :=
  { ‹OrderedRing α›, (Ring.toSemiring : Semiring α) with
    mul_le_mul_of_nonneg_left := fun a b c h hc => by
      /-
        α : Type u
        inst✝ : OrderedRing α
        a✝ b✝ c✝ a b c : α
        h : LE.le a b
        hc : LE.le 0 c
        ⊢ LE.le (HMul.hMul c a) (HMul.hMul c b)
      -/
      simpa only [mul_sub, sub_nonneg] using OrderedRing.mul_nonneg _ _ hc (sub_nonneg.2 h),
      /-
        🎉 no goals
      -/
    mul_le_mul_of_nonneg_right := fun a b c h hc => by
      /-
        α : Type u
        inst✝ : OrderedRing α
        a✝ b✝ c✝ a b c : α
        h : LE.le a b
        hc : LE.le 0 c
        ⊢ LE.le (HMul.hMul a c) (HMul.hMul b c)
      -/
      simpa only [sub_mul, sub_nonneg] using OrderedRing.mul_nonneg _ _ (sub_nonneg.2 h) hc }
      /-
        🎉 no goals
      -/


lemma one_add_le_one_sub_mul_one_add (h : a + b + b * c ≤ c) : 1 + a ≤ (1 - b) * (1 + c) := by
  /-
    α : Type u
    inst✝ : OrderedRing α
    a b c : α
    h : LE.le (HAdd.hAdd (HAdd.hAdd a b) (HMul.hMul b c)) c
    ⊢ LE.le (HAdd.hAdd 1 a) (HMul.hMul (HSub.hSub 1 b) (HAdd.hAdd 1 c))
  -/
  rw [one_sub_mul, mul_one_add, le_sub_iff_add_le, add_assoc, ← add_assoc a]
  /-
    α : Type u
    inst✝ : OrderedRing α
    a b c : α
    h : LE.le (HAdd.hAdd (HAdd.hAdd a b) (HMul.hMul b c)) c
    ⊢ LE.le (HAdd.hAdd 1 (HAdd.hAdd (HAdd.hAdd a b) (HMul.hMul b c))) (HAdd.hAdd 1 …
  -/
  gcongr
  /-
    🎉 no goals
  -/


lemma one_add_le_one_add_mul_one_sub (h : a + c + b * c ≤ b) : 1 + a ≤ (1 + b) * (1 - c) := by
  /-
    α : Type u
    inst✝ : OrderedRing α
    a b c : α
    h : LE.le (HAdd.hAdd (HAdd.hAdd a c) (HMul.hMul b c)) b
    ⊢ LE.le (HAdd.hAdd 1 a) (HMul.hMul (HAdd.hAdd 1 b) (HSub.hSub 1 c))
  -/
  rw [mul_one_sub, one_add_mul, le_sub_iff_add_le, add_assoc, ← add_assoc a]
  /-
    α : Type u
    inst✝ : OrderedRing α
    a b c : α
    h : LE.le (HAdd.hAdd (HAdd.hAdd a c) (HMul.hMul b c)) b
    ⊢ LE.le (HAdd.hAdd 1 (HAdd.hAdd (HAdd.hAdd a c) (HMul.hMul b c))) (HAdd.hAdd 1 …
  -/
  gcongr
  /-
    🎉 no goals
  -/


lemma one_sub_le_one_sub_mul_one_add (h : b + b * c ≤ a + c) : 1 - a ≤ (1 - b) * (1 + c) := by
  /-
    α : Type u
    inst✝ : OrderedRing α
    a b c : α
    h : LE.le (HAdd.hAdd b (HMul.hMul b c)) (HAdd.hAdd a c)
    ⊢ LE.le (HSub.hSub 1 a) (HMul.hMul (HSub.hSub 1 b) (HAdd.hAdd 1 c))
  -/
  rw [one_sub_mul, mul_one_add, sub_le_sub_iff, add_assoc, add_comm c]
  /-
    α : Type u
    inst✝ : OrderedRing α
    a b c : α
    h : LE.le (HAdd.hAdd b (HMul.hMul b c)) (HAdd.hAdd a c)
    ⊢ LE.le (HAdd.hAdd 1 (HAdd.hAdd b (HMul.hMul b c))) (HAdd.hAdd 1 (HAdd.hAdd a  …
  -/
  gcongr
  /-
    🎉 no goals
  -/


lemma one_sub_le_one_add_mul_one_sub (h : c + b * c ≤ a + b) : 1 - a ≤ (1 + b) * (1 - c) := by
  /-
    α : Type u
    inst✝ : OrderedRing α
    a b c : α
    h : LE.le (HAdd.hAdd c (HMul.hMul b c)) (HAdd.hAdd a b)
    ⊢ LE.le (HSub.hSub 1 a) (HMul.hMul (HAdd.hAdd 1 b) (HSub.hSub 1 c))
  -/
  rw [mul_one_sub, one_add_mul, sub_le_sub_iff, add_assoc, add_comm b]
  /-
    α : Type u
    inst✝ : OrderedRing α
    a b c : α
    h : LE.le (HAdd.hAdd c (HMul.hMul b c)) (HAdd.hAdd a b)
    ⊢ LE.le (HAdd.hAdd 1 (HAdd.hAdd c (HMul.hMul b c))) (HAdd.hAdd 1 (HAdd.hAdd a  …
  -/
  gcongr
  /-
    🎉 no goals
  -/


instance (priority := 100) OrderedCommRing.toOrderedCommSemiring : OrderedCommSemiring α :=
  { OrderedRing.toOrderedSemiring, ‹OrderedCommRing α› with }


instance (priority := 200) StrictOrderedSemiring.toPosMulStrictMono : PosMulStrictMono α :=
  ⟨fun x _ _ h => StrictOrderedSemiring.mul_lt_mul_of_pos_left _ _ _ h x.prop⟩

-- see Note [lower instance priority]

instance (priority := 200) StrictOrderedSemiring.toMulPosStrictMono : MulPosStrictMono α :=
  ⟨fun x _ _ h => StrictOrderedSemiring.mul_lt_mul_of_pos_right _ _ _ h x.prop⟩

-- See note [reducible non-instances]

/-- A choice-free version of `StrictOrderedSemiring.toOrderedSemiring` to avoid using choice in
basic `Nat` lemmas. -/
abbrev StrictOrderedSemiring.toOrderedSemiring' [DecidableRel (α := α) (· ≤ ·)] :
    OrderedSemiring α :=
  { ‹StrictOrderedSemiring α› with
    mul_le_mul_of_nonneg_left := fun a b c hab hc => by
      /-
        α : Type u
        inst✝¹ : StrictOrderedSemiring α
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        a b c : α
        hab : LE.le a b
        hc : LE.le 0 c
        ⊢ LE.le (HMul.hMul c a) (HMul.hMul c b)
      -/
      obtain rfl | hab := Decidable.eq_or_lt_of_le hab
        /-
          case inl
          α : Type u
          inst✝¹ : StrictOrderedSemiring α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          a c : α
          hc : LE.le 0 c
          hab : LE.le a a
          ⊢ LE.le (HMul.hMul c a) (HMul.hMul c a)
        -/
      · rfl
        /-
          🎉 no goals
        -/
      /-
        case inr
        α : Type u
        inst✝¹ : StrictOrderedSemiring α
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        a b c : α
        hab✝ : LE.le a b
        hc : LE.le 0 c
        hab : LT.lt a b
        ⊢ LE.le (HMul.hMul c a) (HMul.hMul c b)
      -/
      obtain rfl | hc := Decidable.eq_or_lt_of_le hc
        /-
          case inr.inl
          α : Type u
          inst✝¹ : StrictOrderedSemiring α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          a b : α
          hab✝ : LE.le a b
          hab : LT.lt a b
          hc : LE.le 0 0
          ⊢ LE.le (HMul.hMul 0 a) (HMul.hMul 0 b)
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          α : Type u
          inst✝¹ : StrictOrderedSemiring α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          a b c : α
          hab✝ : LE.le a b
          hc✝ : LE.le 0 c
          hab : LT.lt a b
          hc : LT.lt 0 c
          ⊢ LE.le (HMul.hMul c a) (HMul.hMul c b)
        -/
      · exact (mul_lt_mul_of_pos_left hab hc).le,
        /-
          🎉 no goals
        -/
    mul_le_mul_of_nonneg_right := fun a b c hab hc => by
      /-
        α : Type u
        inst✝¹ : StrictOrderedSemiring α
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        a b c : α
        hab : LE.le a b
        hc : LE.le 0 c
        ⊢ LE.le (HMul.hMul a c) (HMul.hMul b c)
      -/
      obtain rfl | hab := Decidable.eq_or_lt_of_le hab
        /-
          case inl
          α : Type u
          inst✝¹ : StrictOrderedSemiring α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          a c : α
          hc : LE.le 0 c
          hab : LE.le a a
          ⊢ LE.le (HMul.hMul a c) (HMul.hMul a c)
        -/
      · rfl
        /-
          🎉 no goals
        -/
      /-
        case inr
        α : Type u
        inst✝¹ : StrictOrderedSemiring α
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        a b c : α
        hab✝ : LE.le a b
        hc : LE.le 0 c
        hab : LT.lt a b
        ⊢ LE.le (HMul.hMul a c) (HMul.hMul b c)
      -/
      obtain rfl | hc := Decidable.eq_or_lt_of_le hc
        /-
          case inr.inl
          α : Type u
          inst✝¹ : StrictOrderedSemiring α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          a b : α
          hab✝ : LE.le a b
          hab : LT.lt a b
          hc : LE.le 0 0
          ⊢ LE.le (HMul.hMul a 0) (HMul.hMul b 0)
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          α : Type u
          inst✝¹ : StrictOrderedSemiring α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          a b c : α
          hab✝ : LE.le a b
          hc✝ : LE.le 0 c
          hab : LT.lt a b
          hc : LT.lt 0 c
          ⊢ LE.le (HMul.hMul a c) (HMul.hMul b c)
        -/
      · exact (mul_lt_mul_of_pos_right hab hc).le }
        /-
          🎉 no goals
        -/

-- see Note [lower instance priority]

instance (priority := 100) StrictOrderedSemiring.toOrderedSemiring : OrderedSemiring α :=
  { ‹StrictOrderedSemiring α› with
    mul_le_mul_of_nonneg_left := fun _ _ _ =>
      letI := @StrictOrderedSemiring.toOrderedSemiring' α _ (Classical.decRel _)
      mul_le_mul_of_nonneg_left,
    mul_le_mul_of_nonneg_right := fun _ _ _ =>
      letI := @StrictOrderedSemiring.toOrderedSemiring' α _ (Classical.decRel _)
      mul_le_mul_of_nonneg_right }

-- see Note [lower instance priority]

instance (priority := 100) StrictOrderedSemiring.toCharZero [StrictOrderedSemiring α] :
    CharZero α where
  cast_injective :=
                                          /-
                                            α : Type u
                                            inst✝¹ inst✝ : StrictOrderedSemiring α
                                            n : Nat
                                            ⊢ LT.lt ↑n ↑(HAdd.hAdd n 1)
                                          -/
    (strictMono_nat_of_lt_succ fun n ↦ by rw [Nat.cast_succ]; apply lt_add_one).injective
                                                              /-
                                                                🎉 no goals
                                                              -/

-- see Note [lower instance priority]

instance (priority := 100) StrictOrderedSemiring.toNoMaxOrder : NoMaxOrder α :=
  ⟨fun a => ⟨a + 1, lt_add_of_pos_right _ one_pos⟩⟩


/-- A choice-free version of `StrictOrderedCommSemiring.toOrderedCommSemiring'` to avoid using
choice in basic `Nat` lemmas. -/
abbrev StrictOrderedCommSemiring.toOrderedCommSemiring' [DecidableRel (α := α) (· ≤ ·)] :
    OrderedCommSemiring α :=
  { ‹StrictOrderedCommSemiring α›, StrictOrderedSemiring.toOrderedSemiring' with }

-- see Note [lower instance priority]

instance (priority := 100) StrictOrderedCommSemiring.toOrderedCommSemiring :
    OrderedCommSemiring α :=
  { ‹StrictOrderedCommSemiring α›, StrictOrderedSemiring.toOrderedSemiring with }


instance (priority := 100) StrictOrderedRing.toStrictOrderedSemiring : StrictOrderedSemiring α :=
  { ‹StrictOrderedRing α›, (Ring.toSemiring : Semiring α) with
    le_of_add_le_add_left := @le_of_add_le_add_left α _ _ _,
    mul_lt_mul_of_pos_left := fun a b c h hc => by
      /-
        α : Type u
        inst✝ : StrictOrderedRing α
        a b c : α
        h : LT.lt a b
        hc : LT.lt 0 c
        ⊢ LT.lt (HMul.hMul c a) (HMul.hMul c b)
      -/
      simpa only [mul_sub, sub_pos] using StrictOrderedRing.mul_pos _ _ hc (sub_pos.2 h),
      /-
        🎉 no goals
      -/
    mul_lt_mul_of_pos_right := fun a b c h hc => by
      /-
        α : Type u
        inst✝ : StrictOrderedRing α
        a b c : α
        h : LT.lt a b
        hc : LT.lt 0 c
        ⊢ LT.lt (HMul.hMul a c) (HMul.hMul b c)
      -/
      simpa only [sub_mul, sub_pos] using StrictOrderedRing.mul_pos _ _ (sub_pos.2 h) hc }
      /-
        🎉 no goals
      -/

-- See note [reducible non-instances]

/-- A choice-free version of `StrictOrderedRing.toOrderedRing` to avoid using choice in basic
`Int` lemmas. -/
abbrev StrictOrderedRing.toOrderedRing' [DecidableRel (α := α) (· ≤ ·)] : OrderedRing α :=
  { ‹StrictOrderedRing α›, (Ring.toSemiring : Semiring α) with
    mul_nonneg := fun a b ha hb => by
      /-
        α : Type u
        inst✝¹ : StrictOrderedRing α
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        a b : α
        ha : LE.le 0 a
        hb : LE.le 0 b
        ⊢ LE.le 0 (HMul.hMul a b)
      -/
      obtain ha | ha := Decidable.eq_or_lt_of_le ha
        /-
          case inl
          α : Type u
          inst✝¹ : StrictOrderedRing α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          a b : α
          ha✝ : LE.le 0 a
          hb : LE.le 0 b
          ha : Eq 0 a
          ⊢ LE.le 0 (HMul.hMul a b)
        -/
      · rw [← ha, zero_mul]
        /-
          🎉 no goals
        -/
      /-
        case inr
        α : Type u
        inst✝¹ : StrictOrderedRing α
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        a b : α
        ha✝ : LE.le 0 a
        hb : LE.le 0 b
        ha : LT.lt 0 a
        ⊢ LE.le 0 (HMul.hMul a b)
      -/
      obtain hb | hb := Decidable.eq_or_lt_of_le hb
        /-
          case inr.inl
          α : Type u
          inst✝¹ : StrictOrderedRing α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          a b : α
          ha✝ : LE.le 0 a
          hb✝ : LE.le 0 b
          ha : LT.lt 0 a
          hb : Eq 0 b
          ⊢ LE.le 0 (HMul.hMul a b)
        -/
      · rw [← hb, mul_zero]
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          α : Type u
          inst✝¹ : StrictOrderedRing α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          a b : α
          ha✝ : LE.le 0 a
          hb✝ : LE.le 0 b
          ha : LT.lt 0 a
          hb : LT.lt 0 b
          ⊢ LE.le 0 (HMul.hMul a b)
        -/
      · exact (StrictOrderedRing.mul_pos _ _ ha hb).le }
        /-
          🎉 no goals
        -/

-- see Note [lower instance priority]

instance (priority := 100) StrictOrderedRing.toOrderedRing : OrderedRing α where
  __ := ‹StrictOrderedRing α›
  mul_nonneg := fun _ _ => mul_nonneg


/-- A choice-free version of `StrictOrderedCommRing.toOrderedCommRing` to avoid using
choice in basic `Int` lemmas. -/
abbrev StrictOrderedCommRing.toOrderedCommRing' [DecidableRel (α := α) (· ≤ ·)] :
    OrderedCommRing α :=
  { ‹StrictOrderedCommRing α›, StrictOrderedRing.toOrderedRing' with }

-- See note [lower instance priority]

instance (priority := 100) StrictOrderedCommRing.toStrictOrderedCommSemiring :
    StrictOrderedCommSemiring α :=
  { ‹StrictOrderedCommRing α›, StrictOrderedRing.toStrictOrderedSemiring with }

-- See note [lower instance priority]

instance (priority := 100) StrictOrderedCommRing.toOrderedCommRing : OrderedCommRing α :=
  { ‹StrictOrderedCommRing α›, StrictOrderedRing.toOrderedRing with }


instance (priority := 200) LinearOrderedSemiring.toPosMulReflectLT : PosMulReflectLT α :=
  ⟨fun a _ _ => (monotone_mul_left_of_nonneg a.2).reflect_lt⟩

-- see Note [lower instance priority]

instance (priority := 200) LinearOrderedSemiring.toMulPosReflectLT : MulPosReflectLT α :=
  ⟨fun a _ _ => (monotone_mul_right_of_nonneg a.2).reflect_lt⟩


instance (priority := 100) LinearOrderedSemiring.noZeroDivisors : NoZeroDivisors α where
  eq_zero_or_eq_zero_of_mul_eq_zero {a b} hab := by
    /-
      α : Type u
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : ExistsAddOfLE α
      a b : α
      hab : Eq (HMul.hMul a b) 0
      ⊢ Or (Eq a 0) (Eq b 0)
    -/
    contrapose! hab
    /-
      α : Type u
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : ExistsAddOfLE α
      a b : α
      hab : And (Ne a 0) (Ne b 0)
      ⊢ Ne (HMul.hMul a b) 0
    -/
    obtain ha | ha := hab.1.lt_or_lt <;> obtain hb | hb := hab.2.lt_or_lt
    exacts [(mul_pos_of_neg_of_neg ha hb).ne', (mul_neg_of_neg_of_pos ha hb).ne,
      (mul_neg_of_pos_of_neg ha hb).ne, (mul_pos ha hb).ne']

-- Note that we can't use `NoZeroDivisors.to_isDomain` since we are merely in a semiring.
-- See note [lower instance priority]

instance (priority := 100) LinearOrderedRing.isDomain : IsDomain α where
  mul_left_cancel_of_ne_zero {a b c} ha h := by
    /-
      α : Type u
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : ExistsAddOfLE α
      a b c : α
      ha : Ne a 0
      h : Eq (HMul.hMul a b) (HMul.hMul a c)
      ⊢ Eq b c
    -/
    obtain ha | ha := ha.lt_or_lt
    /-
      case inl
      α : Type u
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : ExistsAddOfLE α
      a b c : α
      ha✝ : Ne a 0
      h : Eq (HMul.hMul a b) (HMul.hMul a c)
      ha : LT.lt a 0
      ⊢ Eq b c
    -/
    exacts [(strictAnti_mul_left ha).injective h, (strictMono_mul_left_of_pos ha).injective h]
    /-
      🎉 no goals
    -/
  mul_right_cancel_of_ne_zero {b a c} ha h := by
    /-
      α : Type u
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : ExistsAddOfLE α
      b a c : α
      ha : Ne a 0
      h : Eq (HMul.hMul b a) (HMul.hMul c a)
      ⊢ Eq b c
    -/
    obtain ha | ha := ha.lt_or_lt
    /-
      case inl
      α : Type u
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : ExistsAddOfLE α
      b a c : α
      ha✝ : Ne a 0
      h : Eq (HMul.hMul b a) (HMul.hMul c a)
      ha : LT.lt a 0
      ⊢ Eq b c
    -/
    exacts [(strictAnti_mul_right ha).injective h, (strictMono_mul_right_of_pos ha).injective h]
    /-
      🎉 no goals
    -/

-- See note [lower instance priority]

instance (priority := 100) LinearOrderedSemiring.toLinearOrderedCancelAddCommMonoid :
    LinearOrderedCancelAddCommMonoid α where __ := ‹LinearOrderedSemiring α›


instance (priority := 100) LinearOrderedRing.toLinearOrderedSemiring : LinearOrderedSemiring α :=
  { ‹LinearOrderedRing α›, StrictOrderedRing.toStrictOrderedSemiring with }

-- see Note [lower instance priority]

instance (priority := 100) LinearOrderedRing.toLinearOrderedAddCommGroup :
    LinearOrderedAddCommGroup α where __ := ‹LinearOrderedRing α›


instance (priority := 100) LinearOrderedCommRing.toStrictOrderedCommRing
    [d : LinearOrderedCommRing α] : StrictOrderedCommRing α :=
  { d with }

-- see Note [lower instance priority]

instance (priority := 100) LinearOrderedCommRing.toLinearOrderedCommSemiring
    [d : LinearOrderedCommRing α] : LinearOrderedCommSemiring α :=
  { d, LinearOrderedRing.toLinearOrderedSemiring with }

