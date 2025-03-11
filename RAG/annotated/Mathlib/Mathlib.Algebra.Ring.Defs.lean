/-- A typeclass stating that multiplication is left and right distributive
over addition. -/
class Distrib (R : Type*) extends Mul R, Add R where
  /-- Multiplication is left distributive over addition -/
  protected left_distrib : ∀ a b c : R, a * (b + c) = a * b + a * c
  /-- Multiplication is right distributive over addition -/
  protected right_distrib : ∀ a b c : R, (a + b) * c = a * c + b * c


/-- A typeclass stating that multiplication is left distributive over addition. -/
class LeftDistribClass (R : Type*) [Mul R] [Add R] : Prop where
  /-- Multiplication is left distributive over addition -/
  protected left_distrib : ∀ a b c : R, a * (b + c) = a * b + a * c


/-- A typeclass stating that multiplication is right distributive over addition. -/
class RightDistribClass (R : Type*) [Mul R] [Add R] : Prop where
  /-- Multiplication is right distributive over addition -/
  protected right_distrib : ∀ a b c : R, (a + b) * c = a * c + b * c

-- see Note [lower instance priority]

instance (priority := 100) Distrib.leftDistribClass (R : Type*) [Distrib R] : LeftDistribClass R :=
  ⟨Distrib.left_distrib⟩

-- see Note [lower instance priority]

instance (priority := 100) Distrib.rightDistribClass (R : Type*) [Distrib R] :
    RightDistribClass R :=
  ⟨Distrib.right_distrib⟩


theorem left_distrib [Mul R] [Add R] [LeftDistribClass R] (a b c : R) :
    a * (b + c) = a * b + a * c :=
  LeftDistribClass.left_distrib a b c


alias mul_add := left_distrib


theorem right_distrib [Mul R] [Add R] [RightDistribClass R] (a b c : R) :
    (a + b) * c = a * c + b * c :=
  RightDistribClass.right_distrib a b c


alias add_mul := right_distrib


theorem distrib_three_right [Mul R] [Add R] [RightDistribClass R] (a b c d : R) :
                                                  /-
                                                    R : Type v
                                                    inst✝² : Mul R
                                                    inst✝¹ : Add R
                                                    inst✝ : RightDistribClass R
                                                    a b c d : R
                                                    ⊢ Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd a b) c) d) (HAdd.hAdd (HAdd.hAdd (HMul.h …
                                                  -/
    (a + b + c) * d = a * d + b * d + c * d := by simp [right_distrib]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- A not-necessarily-unital, not-necessarily-associative semiring. See `CommutatorRing` and the
  documentation thereof in case you need a `NonUnitalNonAssocSemiring` instance on a Lie ring
  or a Lie algebra. -/
class NonUnitalNonAssocSemiring (α : Type u) extends AddCommMonoid α, Distrib α, MulZeroClass α


/-- An associative but not-necessarily unital semiring. -/
class NonUnitalSemiring (α : Type u) extends NonUnitalNonAssocSemiring α, SemigroupWithZero α


/-- A unital but not-necessarily-associative semiring. -/
class NonAssocSemiring (α : Type u) extends NonUnitalNonAssocSemiring α, MulZeroOneClass α,
    AddCommMonoidWithOne α


/-- A not-necessarily-unital, not-necessarily-associative ring. -/
class NonUnitalNonAssocRing (α : Type u) extends AddCommGroup α, NonUnitalNonAssocSemiring α


/-- An associative but not-necessarily unital ring. -/
class NonUnitalRing (α : Type*) extends NonUnitalNonAssocRing α, NonUnitalSemiring α


/-- A unital but not-necessarily-associative ring. -/
class NonAssocRing (α : Type*) extends NonUnitalNonAssocRing α, NonAssocSemiring α,
    AddCommGroupWithOne α


/-- A `Semiring` is a type with addition, multiplication, a `0` and a `1` where addition is
commutative and associative, multiplication is associative and left and right distributive over
addition, and `0` and `1` are additive and multiplicative identities. -/
class Semiring (α : Type u) extends NonUnitalSemiring α, NonAssocSemiring α, MonoidWithZero α


/-- A `Ring` is a `Semiring` with negation making it an additive group. -/
class Ring (R : Type u) extends Semiring R, AddCommGroup R, AddGroupWithOne R


theorem add_one_mul [RightDistribClass α] (a b : α) : (a + 1) * b = a * b + b := by
  /-
    α : Type u
    inst✝² : Add α
    inst✝¹ : MulOneClass α
    inst✝ : RightDistribClass α
    a b : α
    ⊢ Eq (HMul.hMul (HAdd.hAdd a 1) b) (HAdd.hAdd (HMul.hMul a b) b)
  -/
  rw [add_mul, one_mul]
  /-
    🎉 no goals
  -/


theorem mul_add_one [LeftDistribClass α] (a b : α) : a * (b + 1) = a * b + a := by
  /-
    α : Type u
    inst✝² : Add α
    inst✝¹ : MulOneClass α
    inst✝ : LeftDistribClass α
    a b : α
    ⊢ Eq (HMul.hMul a (HAdd.hAdd b 1)) (HAdd.hAdd (HMul.hMul a b) a)
  -/
  rw [mul_add, mul_one]
  /-
    🎉 no goals
  -/


theorem one_add_mul [RightDistribClass α] (a b : α) : (1 + a) * b = b + a * b := by
  /-
    α : Type u
    inst✝² : Add α
    inst✝¹ : MulOneClass α
    inst✝ : RightDistribClass α
    a b : α
    ⊢ Eq (HMul.hMul (HAdd.hAdd 1 a) b) (HAdd.hAdd b (HMul.hMul a b))
  -/
  rw [add_mul, one_mul]
  /-
    🎉 no goals
  -/


theorem mul_one_add [LeftDistribClass α] (a b : α) : a * (1 + b) = a + a * b := by
  /-
    α : Type u
    inst✝² : Add α
    inst✝¹ : MulOneClass α
    inst✝ : LeftDistribClass α
    a b : α
    ⊢ Eq (HMul.hMul a (HAdd.hAdd 1 b)) (HAdd.hAdd a (HMul.hMul a b))
  -/
  rw [mul_add, mul_one]
  /-
    🎉 no goals
  -/


theorem two_mul (n : α) : 2 * n = n + n :=
                                                                                     /-
                                                                                       α : Type u
                                                                                       inst✝ : NonAssocSemiring α
                                                                                       n : α
                                                                                       ⊢ Eq (HAdd.hAdd (HMul.hMul 1 n) (HMul.hMul 1 n)) (HAdd.hAdd n n)
                                                                                     -/
  (congrArg₂ _ one_add_one_eq_two.symm rfl).trans <| (right_distrib 1 1 n).trans (by rw [one_mul])
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/

-- Porting note: was [has_add α] [mul_one_class α] [left_distrib_class α]

theorem mul_two (n : α) : n * 2 = n + n :=
                                                                                    /-
                                                                                      α : Type u
                                                                                      inst✝ : NonAssocSemiring α
                                                                                      n : α
                                                                                      ⊢ Eq (HAdd.hAdd (HMul.hMul n 1) (HMul.hMul n 1)) (HAdd.hAdd n n)
                                                                                    -/
  (congrArg₂ _ rfl one_add_one_eq_two.symm).trans <| (left_distrib n 1 1).trans (by rw [mul_one])
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


                                                           /-
                                                             α : Type u
                                                             inst✝¹ : MulZeroClass α
                                                             P : Prop
                                                             inst✝ : Decidable P
                                                             a b : α
                                                             ⊢ Eq (HMul.hMul (ite P a 0) b) (ite P (HMul.hMul a b) 0)
                                                           -/
lemma ite_zero_mul : ite P a 0 * b = ite P (a * b) 0 := by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


                                                           /-
                                                             α : Type u
                                                             inst✝¹ : MulZeroClass α
                                                             P : Prop
                                                             inst✝ : Decidable P
                                                             a b : α
                                                             ⊢ Eq (HMul.hMul a (ite P b 0)) (ite P (HMul.hMul a b) 0)
                                                           -/
lemma mul_ite_zero : a * ite P b 0 = ite P (a * b) 0 := by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma ite_zero_mul_ite_zero : ite P a 0 * ite Q b 0 = ite (P ∧ Q) (a * b) 0 := by
  /-
    α : Type u
    inst✝² : MulZeroClass α
    P Q : Prop
    inst✝¹ : Decidable P
    inst✝ : Decidable Q
    a b : α
    ⊢ Eq (HMul.hMul (ite P a 0) (ite Q b 0)) (ite (And P Q) (HMul.hMul a b) 0)
  -/
  simp only [← ite_and, ite_mul, mul_ite, mul_zero, zero_mul, and_comm]
  /-
    🎉 no goals
  -/


theorem mul_boole {α} [MulZeroOneClass α] (P : Prop) [Decidable P] (a : α) :
                                                        /-
                                                          α : Type u_1
                                                          inst✝¹ : MulZeroOneClass α
                                                          P : Prop
                                                          inst✝ : Decidable P
                                                          a : α
                                                          ⊢ Eq (HMul.hMul a (ite P 1 0)) (ite P a 0)
                                                        -/
    (a * if P then 1 else 0) = if P then a else 0 := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/

-- Porting note: no @[simp] because simp proves it

theorem boole_mul {α} [MulZeroOneClass α] (P : Prop) [Decidable P] (a : α) :
                                                        /-
                                                          α : Type u_1
                                                          inst✝¹ : MulZeroOneClass α
                                                          P : Prop
                                                          inst✝ : Decidable P
                                                          a : α
                                                          ⊢ Eq (HMul.hMul (ite P 1 0) a) (ite P a 0)
                                                        -/
    (if P then 1 else 0) * a = if P then a else 0 := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- A not-necessarily-unital, not-necessarily-associative, but commutative semiring. -/
class NonUnitalNonAssocCommSemiring (α : Type u) extends NonUnitalNonAssocSemiring α, CommMagma α


/-- A non-unital commutative semiring is a `NonUnitalSemiring` with commutative multiplication.
In other words, it is a type with the following structures: additive commutative monoid
(`AddCommMonoid`), commutative semigroup (`CommSemigroup`), distributive laws (`Distrib`), and
multiplication by zero law (`MulZeroClass`). -/
class NonUnitalCommSemiring (α : Type u) extends NonUnitalSemiring α, CommSemigroup α


/-- A commutative semiring is a semiring with commutative multiplication. -/
class CommSemiring (R : Type u) extends Semiring R, CommMonoid R

-- see Note [lower instance priority]

instance (priority := 100) CommSemiring.toNonUnitalCommSemiring [CommSemiring α] :
    NonUnitalCommSemiring α :=
  { inferInstanceAs (CommMonoid α), inferInstanceAs (CommSemiring α) with }

-- see Note [lower instance priority]

instance (priority := 100) CommSemiring.toCommMonoidWithZero [CommSemiring α] :
    CommMonoidWithZero α :=
  { inferInstanceAs (CommMonoid α), inferInstanceAs (CommSemiring α) with }


theorem add_mul_self_eq (a b : α) : (a + b) * (a + b) = a * a + 2 * a * b + b * b := by
  /-
    α : Type u
    inst✝ : CommSemiring α
    a b : α
    ⊢ Eq (HMul.hMul (HAdd.hAdd a b) (HAdd.hAdd a b)) (HAdd.hAdd (HAdd.hAdd (HMul.h …
  -/
  simp only [two_mul, add_mul, mul_add, add_assoc, mul_comm b]
  /-
    🎉 no goals
  -/


lemma add_sq (a b : α) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by
  /-
    α : Type u
    inst✝ : CommSemiring α
    a b : α
    ⊢ Eq (HPow.hPow (HAdd.hAdd a b) 2) (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 2) (HMul …
  -/
  simp only [sq, add_mul_self_eq]
  /-
    🎉 no goals
  -/


lemma add_sq' (a b : α) : (a + b) ^ 2 = a ^ 2 + b ^ 2 + 2 * a * b := by
  /-
    α : Type u
    inst✝ : CommSemiring α
    a b : α
    ⊢ Eq (HPow.hPow (HAdd.hAdd a b) 2) (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 2) (HPow …
  -/
  rw [add_sq, add_assoc, add_comm _ (b ^ 2), add_assoc]
  /-
    🎉 no goals
  -/


alias add_pow_two := add_sq


/-- Typeclass for a negation operator that distributes across multiplication.

This is useful for dealing with submonoids of a ring that contain `-1` without having to duplicate
lemmas. -/
class HasDistribNeg (α : Type*) [Mul α] extends InvolutiveNeg α where
  /-- Negation is left distributive over multiplication -/
  neg_mul : ∀ x y : α, -x * y = -(x * y)
  /-- Negation is right distributive over multiplication -/
  mul_neg : ∀ x y : α, x * -y = -(x * y)


@[simp]
theorem neg_mul (a b : α) : -a * b = -(a * b) :=
  HasDistribNeg.neg_mul _ _


@[simp]
theorem mul_neg (a b : α) : a * -b = -(a * b) :=
  HasDistribNeg.mul_neg _ _


                                                      /-
                                                        α : Type u
                                                        inst✝¹ : Mul α
                                                        inst✝ : HasDistribNeg α
                                                        a b : α
                                                        ⊢ Eq (HMul.hMul (Neg.neg a) (Neg.neg b)) (HMul.hMul a b)
                                                      -/
theorem neg_mul_neg (a b : α) : -a * -b = a * b := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem neg_mul_eq_neg_mul (a b : α) : -(a * b) = -a * b :=
  (neg_mul _ _).symm


theorem neg_mul_eq_mul_neg (a b : α) : -(a * b) = a * -b :=
  (mul_neg _ _).symm


                                                       /-
                                                         α : Type u
                                                         inst✝¹ : Mul α
                                                         inst✝ : HasDistribNeg α
                                                         a b : α
                                                         ⊢ Eq (HMul.hMul (Neg.neg a) b) (HMul.hMul a (Neg.neg b))
                                                       -/
theorem neg_mul_comm (a b : α) : -a * b = a * -b := by simp
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                       /-
                                                         α : Type u
                                                         inst✝¹ : MulOneClass α
                                                         inst✝ : HasDistribNeg α
                                                         a : α
                                                         ⊢ Eq (Neg.neg a) (HMul.hMul (-1) a)
                                                       -/
theorem neg_eq_neg_one_mul (a : α) : -a = -1 * a := by simp
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- An element of a ring multiplied by the additive inverse of one is the element's additive
  inverse. -/
                                                /-
                                                  α : Type u
                                                  inst✝¹ : MulOneClass α
                                                  inst✝ : HasDistribNeg α
                                                  a : α
                                                  ⊢ Eq (HMul.hMul a (-1)) (Neg.neg a)
                                                -/
theorem mul_neg_one (a : α) : a * -1 = -a := by simp
                                                /-
                                                  🎉 no goals
                                                -/


/-- The additive inverse of one multiplied by an element of a ring is the element's additive
  inverse. -/
                                                /-
                                                  α : Type u
                                                  inst✝¹ : MulOneClass α
                                                  inst✝ : HasDistribNeg α
                                                  a : α
                                                  ⊢ Eq (HMul.hMul (-1) a) (Neg.neg a)
                                                -/
theorem neg_one_mul (a : α) : -1 * a = -a := by simp
                                                /-
                                                  🎉 no goals
                                                -/


instance (priority := 100) MulZeroClass.negZeroClass : NegZeroClass α where
  __ := inferInstanceAs (Zero α); __ := inferInstanceAs (InvolutiveNeg α)
                 /-
                   α : Type u
                   R : Type v
                   inst✝¹ : MulZeroClass α
                   inst✝ : HasDistribNeg α
                   ⊢ Eq (-0) 0
                 -/
  neg_zero := by rw [← zero_mul (0 : α), ← neg_mul, mul_zero, mul_zero]
                 /-
                   🎉 no goals
                 -/


instance (priority := 100) NonUnitalNonAssocRing.toHasDistribNeg : HasDistribNeg α where
  neg := Neg.neg
  neg_neg := neg_neg
                                                  /-
                                                    α : Type u
                                                    R : Type v
                                                    inst✝ : NonUnitalNonAssocRing α
                                                    a b : α
                                                    ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg a) b) (HMul.hMul a b)) 0
                                                  -/
  neg_mul a b := eq_neg_of_add_eq_zero_left <| by rw [← right_distrib, neg_add_cancel, zero_mul]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    α : Type u
                                                    R : Type v
                                                    inst✝ : NonUnitalNonAssocRing α
                                                    a b : α
                                                    ⊢ Eq (HAdd.hAdd (HMul.hMul a (Neg.neg b)) (HMul.hMul a b)) 0
                                                  -/
  mul_neg a b := eq_neg_of_add_eq_zero_left <| by rw [← left_distrib, neg_add_cancel, mul_zero]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem mul_sub_left_distrib (a b c : α) : a * (b - c) = a * b - a * c := by
  /-
    α : Type u
    inst✝ : NonUnitalNonAssocRing α
    a b c : α
    ⊢ Eq (HMul.hMul a (HSub.hSub b c)) (HSub.hSub (HMul.hMul a b) (HMul.hMul a c))
  -/
  simpa only [sub_eq_add_neg, neg_mul_eq_mul_neg] using mul_add a b (-c)
  /-
    🎉 no goals
  -/


alias mul_sub := mul_sub_left_distrib


theorem mul_sub_right_distrib (a b c : α) : (a - b) * c = a * c - b * c := by
  /-
    α : Type u
    inst✝ : NonUnitalNonAssocRing α
    a b c : α
    ⊢ Eq (HMul.hMul (HSub.hSub a b) c) (HSub.hSub (HMul.hMul a c) (HMul.hMul b c))
  -/
  simpa only [sub_eq_add_neg, neg_mul_eq_neg_mul] using add_mul a (-b) c
  /-
    🎉 no goals
  -/


alias sub_mul := mul_sub_right_distrib


                                                              /-
                                                                α : Type u
                                                                inst✝ : NonAssocRing α
                                                                a b : α
                                                                ⊢ Eq (HMul.hMul (HSub.hSub a 1) b) (HSub.hSub (HMul.hMul a b) b)
                                                              -/
theorem sub_one_mul (a b : α) : (a - 1) * b = a * b - b := by rw [sub_mul, one_mul]
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                              /-
                                                                α : Type u
                                                                inst✝ : NonAssocRing α
                                                                a b : α
                                                                ⊢ Eq (HMul.hMul a (HSub.hSub b 1)) (HSub.hSub (HMul.hMul a b) a)
                                                              -/
theorem mul_sub_one (a b : α) : a * (b - 1) = a * b - a := by rw [mul_sub, mul_one]
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                              /-
                                                                α : Type u
                                                                inst✝ : NonAssocRing α
                                                                a b : α
                                                                ⊢ Eq (HMul.hMul (HSub.hSub 1 a) b) (HSub.hSub b (HMul.hMul a b))
                                                              -/
theorem one_sub_mul (a b : α) : (1 - a) * b = b - a * b := by rw [sub_mul, one_mul]
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                              /-
                                                                α : Type u
                                                                inst✝ : NonAssocRing α
                                                                a b : α
                                                                ⊢ Eq (HMul.hMul a (HSub.hSub 1 b)) (HSub.hSub a (HMul.hMul a b))
                                                              -/
theorem mul_one_sub (a b : α) : a * (1 - b) = a - a * b := by rw [mul_sub, mul_one]
                                                              /-
                                                                🎉 no goals
                                                              -/


instance (priority := 100) Ring.toNonUnitalRing : NonUnitalRing α :=
  { ‹Ring α› with }

-- A (unital, associative) ring is a not-necessarily-associative ring
-- see Note [lower instance priority]

instance (priority := 100) Ring.toNonAssocRing : NonAssocRing α :=
  { ‹Ring α› with }


/-- The instance from `Ring` to `Semiring` happens often in linear algebra, for which all the basic
definitions are given in terms of semirings, but many applications use rings or fields. We increase
a little bit its priority above 100 to try it quickly, but remaining below the default 1000 so that
more specific instances are tried first. -/
instance (priority := 200) : Semiring α :=
  { ‹Ring α› with }


/-- A non-unital non-associative commutative ring is a `NonUnitalNonAssocRing` with commutative
multiplication. -/
class NonUnitalNonAssocCommRing (α : Type u)
  extends NonUnitalNonAssocRing α, NonUnitalNonAssocCommSemiring α


/-- A non-unital commutative ring is a `NonUnitalRing` with commutative multiplication. -/
class NonUnitalCommRing (α : Type u) extends NonUnitalRing α, NonUnitalNonAssocCommRing α

-- see Note [lower instance priority]

instance (priority := 100) NonUnitalCommRing.toNonUnitalCommSemiring [s : NonUnitalCommRing α] :
    NonUnitalCommSemiring α :=
  { s with }


/-- A commutative ring is a ring with commutative multiplication. -/
class CommRing (α : Type u) extends Ring α, CommMonoid α


instance (priority := 100) CommRing.toCommSemiring [s : CommRing α] : CommSemiring α :=
  { s with }

-- see Note [lower instance priority]

instance (priority := 100) CommRing.toNonUnitalCommRing [s : CommRing α] : NonUnitalCommRing α :=
  { s with }

-- see Note [lower instance priority]

instance (priority := 100) CommRing.toAddCommGroupWithOne [s : CommRing α] :
    AddCommGroupWithOne α :=
  { s with }


/-- A domain is a nontrivial semiring such that multiplication by a non zero element
is cancellative on both sides. In other words, a nontrivial semiring `R` satisfying
`∀ {a b c : R}, a ≠ 0 → a * b = a * c → b = c` and
`∀ {a b c : R}, b ≠ 0 → a * b = c * b → a = c`.

This is implemented as a mixin for `Semiring α`.
To obtain an integral domain use `[CommRing α] [IsDomain α]`. -/
@[stacks 09FE]
class IsDomain (α : Type u) [Semiring α] extends IsCancelMulZero α, Nontrivial α : Prop

