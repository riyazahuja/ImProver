theorem lt_irrefl {α : Type u} [Preorder α] {a : α} : ¬a < a := _root_.lt_irrefl a


theorem zero_lt_one {α : Type u} [StrictOrderedSemiring α] : (0:α) < 1 := _root_.zero_lt_one


theorem eq_of_eq_of_eq {α} [OrderedSemiring α] {a b : α} (ha : a = 0) (hb : b = 0) : a + b = 0 := by
  /-
    α : Type u_1
    inst✝ : OrderedSemiring α
    a b : α
    ha : Eq a 0
    hb : Eq b 0
    ⊢ Eq (HAdd.hAdd a b) 0
  -/
  simp [*]
  /-
    🎉 no goals
  -/


theorem le_of_eq_of_le {α} [OrderedSemiring α] {a b : α} (ha : a = 0) (hb : b ≤ 0) : a + b ≤ 0 := by
  /-
    α : Type u_1
    inst✝ : OrderedSemiring α
    a b : α
    ha : Eq a 0
    hb : LE.le b 0
    ⊢ LE.le (HAdd.hAdd a b) 0
  -/
  simp [*]
  /-
    🎉 no goals
  -/


theorem lt_of_eq_of_lt {α} [OrderedSemiring α] {a b : α} (ha : a = 0) (hb : b < 0) : a + b < 0 := by
  /-
    α : Type u_1
    inst✝ : OrderedSemiring α
    a b : α
    ha : Eq a 0
    hb : LT.lt b 0
    ⊢ LT.lt (HAdd.hAdd a b) 0
  -/
  simp [*]
  /-
    🎉 no goals
  -/


theorem le_of_le_of_eq {α} [OrderedSemiring α] {a b : α} (ha : a ≤ 0) (hb : b = 0) : a + b ≤ 0 := by
  /-
    α : Type u_1
    inst✝ : OrderedSemiring α
    a b : α
    ha : LE.le a 0
    hb : Eq b 0
    ⊢ LE.le (HAdd.hAdd a b) 0
  -/
  simp [*]
  /-
    🎉 no goals
  -/


theorem lt_of_lt_of_eq {α} [OrderedSemiring α] {a b : α} (ha : a < 0) (hb : b = 0) : a + b < 0 := by
  /-
    α : Type u_1
    inst✝ : OrderedSemiring α
    a b : α
    ha : LT.lt a 0
    hb : Eq b 0
    ⊢ LT.lt (HAdd.hAdd a b) 0
  -/
  simp [*]
  /-
    🎉 no goals
  -/


theorem add_nonpos {α : Type*} [OrderedSemiring α] {a b : α} (ha : a ≤ 0) (hb : b ≤ 0) :
    a + b ≤ 0 :=
  _root_.add_nonpos ha hb


theorem add_lt_of_le_of_neg {α : Type*} [StrictOrderedSemiring α] {a b c : α} (hbc : b ≤ c)
    (ha : a < 0) : b + a < c :=
  _root_.add_lt_of_le_of_neg hbc ha


theorem add_lt_of_neg_of_le {α : Type*} [StrictOrderedSemiring α] {a b c : α} (ha : a < 0)
    (hbc : b ≤ c) : a + b < c :=
  _root_.add_lt_of_neg_of_le ha hbc


theorem add_neg {α : Type*} [StrictOrderedSemiring α] {a b : α} (ha : a < 0)
    (hb : b < 0) : a + b < 0 :=
  _root_.add_neg ha hb


theorem mul_neg {α} [StrictOrderedRing α] {a b : α} (ha : a < 0) (hb : 0 < b) : b * a < 0 :=
  have : (-b)*a > 0 := mul_pos_of_neg_of_neg (neg_neg_of_pos hb) ha
                     /-
                       α : Type u_1
                       inst✝ : StrictOrderedRing α
                       a b : α
                       ha : LT.lt a 0
                       hb : LT.lt 0 b
                       this : GT.gt (HMul.hMul (Neg.neg b) a) 0
                       ⊢ LT.lt 0 (Neg.neg (HMul.hMul b a))
                     -/
  neg_of_neg_pos (by simpa)
                     /-
                       🎉 no goals
                     -/


theorem mul_nonpos {α} [OrderedRing α] {a b : α} (ha : a ≤ 0) (hb : 0 < b) : b * a ≤ 0 :=
  have : (-b)*a ≥ 0 := mul_nonneg_of_nonpos_of_nonpos (le_of_lt (neg_neg_of_pos hb)) ha
     /-
       α : Type u_1
       inst✝ : OrderedRing α
       a b : α
       ha : LE.le a 0
       hb : LT.lt 0 b
       this : GE.ge (HMul.hMul (Neg.neg b) a) 0
       ⊢ LE.le (HMul.hMul b a) 0
     -/
  by simpa
     /-
       🎉 no goals
     -/

-- used alongside `mul_neg` and `mul_nonpos`, so has the same argument pattern for uniformity

@[nolint unusedArguments]
theorem mul_eq {α} [OrderedSemiring α] {a b : α} (ha : a = 0) (_ : 0 < b) : b * a = 0 := by
  /-
    α : Type u_1
    inst✝ : OrderedSemiring α
    a b : α
    ha : Eq a 0
    x✝ : LT.lt 0 b
    ⊢ Eq (HMul.hMul b a) 0
  -/
  simp [*]
  /-
    🎉 no goals
  -/


open Mathlib in
/-- Finds the name of a multiplicative lemma corresponding to an inequality strength. -/
def _root_.Mathlib.Ineq.toConstMulName : Ineq → Lean.Name
  | .lt => ``mul_neg
  | .le => ``mul_nonpos
  | .eq => ``mul_eq


theorem sub_nonpos_of_le {α : Type*} [OrderedRing α] {a b : α} : a ≤ b → a - b ≤ 0 :=
  _root_.sub_nonpos_of_le


theorem sub_neg_of_lt {α : Type*} [OrderedRing α] {a b : α} : a < b → a - b < 0 :=
  _root_.sub_neg_of_lt


lemma eq_of_not_lt_of_not_gt {α} [LinearOrder α] (a b : α) (h1 : ¬ a < b) (h2 : ¬ b < a) : a = b :=
  le_antisymm (le_of_not_gt h2) (le_of_not_gt h1)

-- used in the `nlinarith` normalization steps. The `_` argument is for uniformity.

@[nolint unusedArguments]
lemma mul_zero_eq {α} {R : α → α → Prop} [Semiring α] {a b : α} (_ : R a 0) (h : b = 0) :
    a * b = 0 := by
  /-
    α : Type u_1
    R : α → α → Prop
    inst✝ : Semiring α
    a b : α
    x✝ : R a 0
    h : Eq b 0
    ⊢ Eq (HMul.hMul a b) 0
  -/
  simp [h]
  /-
    🎉 no goals
  -/

-- used in the `nlinarith` normalization steps. The `_` argument is for uniformity.

@[nolint unusedArguments]
lemma zero_mul_eq {α} {R : α → α → Prop} [Semiring α] {a b : α} (h : a = 0) (_ : R b 0) :
    a * b = 0 := by
  /-
    α : Type u_1
    R : α → α → Prop
    inst✝ : Semiring α
    a b : α
    h : Eq a 0
    x✝ : R b 0
    ⊢ Eq (HMul.hMul a b) 0
  -/
  simp [h]
  /-
    🎉 no goals
  -/


lemma natCast_nonneg (α : Type u) [OrderedSemiring α] (n : ℕ) : (0 : α) ≤ n := Nat.cast_nonneg n


@[deprecated (since := "2024-04-17")]
alias nat_cast_nonneg := natCast_nonneg


theorem lt_zero_of_zero_gt {α : Type*} [Zero α] [LT α] {a : α} (h : 0 > a) : a < 0 := h


theorem le_zero_of_zero_ge {α : Type*} [Zero α] [LE α] {a : α} (h : 0 ≥ a) : a ≤ 0 := h


