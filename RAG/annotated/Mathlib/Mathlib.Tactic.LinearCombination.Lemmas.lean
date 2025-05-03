theorem add_eq_eq [Add α] (p₁ : (a₁:α) = b₁) (p₂ : a₂ = b₂) : a₁ + a₂ = b₁ + b₂ := p₁ ▸ p₂ ▸ rfl


theorem add_le_eq [OrderedAddCommMonoid α]
    (p₁ : (a₁:α) ≤ b₁) (p₂ : a₂ = b₂) : a₁ + a₂ ≤ b₁ + b₂ :=
  p₂ ▸ add_le_add_right p₁ b₂


theorem add_eq_le [OrderedAddCommMonoid α]
    (p₁ : (a₁:α) = b₁) (p₂ : a₂ ≤ b₂) : a₁ + a₂ ≤ b₁ + b₂ :=
  p₁ ▸ add_le_add_left p₂ b₁


theorem add_lt_eq [OrderedCancelAddCommMonoid α]
    (p₁ : (a₁:α) < b₁) (p₂ : a₂ = b₂) : a₁ + a₂ < b₁ + b₂ :=
  p₂ ▸ add_lt_add_right p₁ b₂


theorem add_eq_lt [OrderedCancelAddCommMonoid α] {a₁ b₁ a₂ b₂ : α}
    (p₁ : a₁ = b₁) (p₂ : a₂ < b₂) : a₁ + a₂ < b₁ + b₂ :=
  p₁ ▸ add_lt_add_left p₂ b₁


theorem mul_eq_const [Mul α] (p : a = b) (c : α) : a * c = b * c := p ▸ rfl


theorem mul_le_const [OrderedSemiring α] (p : b ≤ c) {a : α} (ha : 0 ≤ a) :
    b * a ≤ c * a :=
  mul_le_mul_of_nonneg_right p ha


theorem mul_lt_const [StrictOrderedSemiring α] (p : b < c) {a : α} (ha : 0 < a) :
    b * a < c * a :=
  mul_lt_mul_of_pos_right p ha


theorem mul_lt_const_weak [OrderedSemiring α] (p : b < c) {a : α} (ha : 0 ≤ a) :
    b * a ≤ c * a :=
  mul_le_mul_of_nonneg_right p.le ha


theorem mul_const_eq [Mul α] (p : b = c) (a : α) : a * b = a * c := p ▸ rfl


theorem mul_const_le [OrderedSemiring α] (p : b ≤ c) {a : α} (ha : 0 ≤ a) :
    a * b ≤ a * c :=
  mul_le_mul_of_nonneg_left p ha


theorem mul_const_lt [StrictOrderedSemiring α] (p : b < c) {a : α} (ha : 0 < a) :
    a * b < a * c :=
  mul_lt_mul_of_pos_left p ha


theorem mul_const_lt_weak [OrderedSemiring α] (p : b < c) {a : α} (ha : 0 ≤ a) :
    a * b ≤ a * c :=
  mul_le_mul_of_nonneg_left p.le ha


theorem smul_eq_const [SMul K α] (p : t = s) (c : α) : t • c = s • c := p ▸ rfl


theorem smul_le_const [OrderedRing K] [OrderedAddCommGroup α] [Module K α]
    [OrderedSMul K α] (p : t ≤ s) {a : α} (ha : 0 ≤ a) :
    t • a ≤ s • a :=
  smul_le_smul_of_nonneg_right p ha


theorem smul_lt_const [OrderedRing K] [OrderedAddCommGroup α] [Module K α]
    [OrderedSMul K α] (p : t < s) {a : α} (ha : 0 < a) :
    t • a < s • a :=
  smul_lt_smul_of_pos_right p ha


theorem smul_lt_const_weak [OrderedRing K] [OrderedAddCommGroup α] [Module K α]
    [OrderedSMul K α] (p : t < s) {a : α} (ha : 0 ≤ a) :
    t • a ≤ s • a :=
  smul_le_smul_of_nonneg_right p.le ha


theorem smul_const_eq [SMul K α] (p : b = c) (s : K) : s • b = s • c := p ▸ rfl


theorem smul_const_le [OrderedSemiring K] [OrderedAddCommMonoid α] [Module K α]
    [OrderedSMul K α] (p : b ≤ c) {s : K} (hs : 0 ≤ s) :
    s • b ≤ s • c :=
  smul_le_smul_of_nonneg_left p hs


theorem smul_const_lt [OrderedSemiring K] [OrderedAddCommMonoid α] [Module K α]
    [OrderedSMul K α] (p : b < c) {s : K} (hs : 0 < s) :
    s • b < s • c :=
  smul_lt_smul_of_pos_left p hs


theorem smul_const_lt_weak [OrderedSemiring K] [OrderedAddCommMonoid α] [Module K α]
    [OrderedSMul K α] (p : b < c) {s : K} (hs : 0 ≤ s) :
    s • b ≤ s • c :=
  smul_le_smul_of_nonneg_left p.le hs


theorem div_eq_const [Div α] (p : a = b) (c : α) : a / c = b / c := p ▸ rfl


theorem div_le_const [LinearOrderedSemifield α] (p : b ≤ c) {a : α} (ha : 0 ≤ a) : b / a ≤ c / a :=
  div_le_div_of_nonneg_right p ha


theorem div_lt_const [LinearOrderedSemifield α] (p : b < c) {a : α} (ha : 0 < a) : b / a < c / a :=
  div_lt_div_of_pos_right p ha


theorem div_lt_const_weak [LinearOrderedSemifield α] (p : b < c) {a : α} (ha : 0 ≤ a) :
    b / a ≤ c / a :=
  div_le_div_of_nonneg_right p.le ha


theorem eq_of_eq [Add α] [IsRightCancelAdd α] (p : (a:α) = b) (H : a' + b = b' + a) : a' = b' := by
  /-
    α : Type u_1
    a a' b b' : α
    inst✝¹ : Add α
    inst✝ : IsRightCancelAdd α
    p : Eq a b
    H : Eq (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ Eq a' b'
  -/
  rw [p] at H
  /-
    α : Type u_1
    a a' b b' : α
    inst✝¹ : Add α
    inst✝ : IsRightCancelAdd α
    p : Eq a b
    H : Eq (HAdd.hAdd a' b) (HAdd.hAdd b' b)
    ⊢ Eq a' b'
  -/
  exact add_right_cancel H
  /-
    🎉 no goals
  -/


theorem le_of_le [OrderedCancelAddCommMonoid α] (p : (a:α) ≤ b) (H : a' + b ≤ b' + a) :
    a' ≤ b' := by
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : LE.le a b
    H : LE.le (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LE.le a' b'
  -/
  rw [← add_le_add_iff_right b]
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : LE.le a b
    H : LE.le (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LE.le (HAdd.hAdd a' b) (HAdd.hAdd b' b)
  -/
  apply H.trans
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : LE.le a b
    H : LE.le (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LE.le (HAdd.hAdd b' a) (HAdd.hAdd b' b)
  -/
  apply add_le_add_left p
  /-
    🎉 no goals
  -/


theorem le_of_eq [OrderedCancelAddCommMonoid α] (p : (a:α) = b) (H : a' + b ≤ b' + a) :
    a' ≤ b' := by
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : Eq a b
    H : LE.le (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LE.le a' b'
  -/
  rwa [p, add_le_add_iff_right] at H
  /-
    🎉 no goals
  -/


theorem le_of_lt [OrderedCancelAddCommMonoid α] (p : (a:α) < b) (H : a' + b ≤ b' + a) :
    a' ≤ b' :=
  le_of_le p.le H


theorem lt_of_le [OrderedCancelAddCommMonoid α] (p : (a:α) ≤ b) (H : a' + b < b' + a) :
    a' < b' := by
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : LE.le a b
    H : LT.lt (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LT.lt a' b'
  -/
  rw [← add_lt_add_iff_right b]
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : LE.le a b
    H : LT.lt (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LT.lt (HAdd.hAdd a' b) (HAdd.hAdd b' b)
  -/
  apply H.trans_le
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : LE.le a b
    H : LT.lt (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LE.le (HAdd.hAdd b' a) (HAdd.hAdd b' b)
  -/
  apply add_le_add_left p
  /-
    🎉 no goals
  -/


theorem lt_of_eq [OrderedCancelAddCommMonoid α] (p : (a:α) = b) (H : a' + b < b' + a) :
    a' < b' := by
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : Eq a b
    H : LT.lt (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LT.lt a' b'
  -/
  rwa [p, add_lt_add_iff_right] at H
  /-
    🎉 no goals
  -/


theorem lt_of_lt [OrderedCancelAddCommMonoid α] (p : (a:α) < b) (H : a' + b ≤ b' + a) :
    a' < b' := by
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : LT.lt a b
    H : LE.le (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LT.lt a' b'
  -/
  rw [← add_lt_add_iff_right b]
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : LT.lt a b
    H : LE.le (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LT.lt (HAdd.hAdd a' b) (HAdd.hAdd b' b)
  -/
  apply H.trans_lt
  /-
    α : Type u_1
    a a' b b' : α
    inst✝ : OrderedCancelAddCommMonoid α
    p : LT.lt a b
    H : LE.le (HAdd.hAdd a' b) (HAdd.hAdd b' a)
    ⊢ LT.lt (HAdd.hAdd b' a) (HAdd.hAdd b' b)
  -/
  apply add_lt_add_left p
  /-
    🎉 no goals
  -/


alias ⟨eq_rearrange, _⟩ := sub_eq_zero


theorem le_rearrange {α : Type*} [OrderedAddCommGroup α] {a b : α} (h : a - b ≤ 0) : a ≤ b :=
  sub_nonpos.mp h


theorem lt_rearrange {α : Type*} [OrderedAddCommGroup α] {a b : α} (h : a - b < 0) : a < b :=
  sub_neg.mp h


theorem eq_of_add_pow [Ring α] [NoZeroDivisors α] (n : ℕ) (p : (a:α) = b)
    (H : (a' - b')^n - (a - b) = 0) : a' = b' := by
  /-
    α : Type u_1
    a a' b b' : α
    inst✝¹ : Ring α
    inst✝ : NoZeroDivisors α
    n : Nat
    p : Eq a b
    H : Eq (HSub.hSub (HPow.hPow (HSub.hSub a' b') n) (HSub.hSub a b)) 0
    ⊢ Eq a' b'
  -/
  rw [← sub_eq_zero] at p ⊢; apply pow_eq_zero (n := n); rwa [sub_eq_zero, p] at H
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Given two (in)equalities, look up the lemma to add them. -/
def addRelRelData : Ineq → Ineq → Name
  | eq, eq => ``add_eq_eq
  | eq, le => ``add_eq_le
  | eq, lt => ``add_eq_lt
  | le, eq => ``add_le_eq
  | le, le => ``add_le_add
  | le, lt => ``add_lt_add_of_le_of_lt
  | lt, eq => ``add_lt_eq
  | lt, le => ``add_lt_add_of_lt_of_le
  | lt, lt => ``add_lt_add


/-- Finite inductive type extending `Mathlib.Ineq`: a type of inequality (`eq`, `le` or `lt`),
together with, in the case of `lt`, a boolean, typically representing the strictness (< or ≤) of
some other inequality. -/
protected inductive WithStrictness : Type
  | eq : Ineq.WithStrictness
  | le : Ineq.WithStrictness
  | lt (strict : Bool) : Ineq.WithStrictness


/-- Given an (in)equality, look up the lemma to left-multiply it by a constant.  If relevant, also
take into account the degree of positivity which can be proved of the constant: strict or
non-strict. -/
def mulRelConstData : Ineq.WithStrictness → Name
  | .eq => ``mul_eq_const
  | .le => ``mul_le_const
  | .lt true => ``mul_lt_const
  | .lt false => ``mul_lt_const_weak


/-- Given an (in)equality, look up the lemma to right-multiply it by a constant.  If relevant, also
take into account the degree of positivity which can be proved of the constant: strict or
non-strict. -/
def mulConstRelData : Ineq.WithStrictness → Name
  | .eq => ``mul_const_eq
  | .le => ``mul_const_le
  | .lt true => ``mul_const_lt
  | .lt false => ``mul_const_lt_weak


/-- Given an (in)equality, look up the lemma to left-scalar-multiply it by a constant (scalar).
If relevant, also take into account the degree of positivity which can be proved of the constant:
strict or non-strict. -/
def smulRelConstData : Ineq.WithStrictness → Name
  | .eq => ``smul_eq_const
  | .le => ``smul_le_const
  | .lt true => ``smul_lt_const
  | .lt false => ``smul_lt_const_weak


/-- Given an (in)equality, look up the lemma to right-scalar-multiply it by a constant (vector).
If relevant, also take into account the degree of positivity which can be proved of the constant:
strict or non-strict. -/
def smulConstRelData : Ineq.WithStrictness → Name
  | .eq => ``smul_const_eq
  | .le => ``smul_const_le
  | .lt true => ``smul_const_lt
  | .lt false => ``smul_const_lt_weak


/-- Given an (in)equality, look up the lemma to divide it by a constant.  If relevant, also take
into account the degree of positivity which can be proved of the constant: strict or non-strict. -/
def divRelConstData : Ineq.WithStrictness → Name
  | .eq => ``div_eq_const
  | .le => ``div_le_const
  | .lt true => ``div_lt_const
  | .lt false => ``div_lt_const_weak


/-- Given two (in)equalities `P` and `Q`, look up the lemma to deduce `Q` from `P`, and the relation
appearing in the side condition produced by this lemma. -/
def relImpRelData : Ineq → Ineq → Option (Name × Ineq)
  | eq, eq => some (``eq_of_eq, eq)
  | eq, le => some (``Tactic.LinearCombination.le_of_eq, le)
  | eq, lt => some (``lt_of_eq, lt)
  | le, eq => none
  | le, le => some (``le_of_le, le)
  | le, lt => some (``lt_of_le, lt)
  | lt, eq => none
  | lt, le => some (``Tactic.LinearCombination.le_of_lt, le)
  | lt, lt => some (``lt_of_lt, le)


/-- Given an (in)equality, look up the lemma to move everything to the LHS. -/
def rearrangeData : Ineq → Name
  | eq => ``eq_rearrange
  | le => ``le_rearrange
  | lt => ``lt_rearrange


