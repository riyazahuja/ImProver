/-- Specialisation of `Nat.cast_nonneg'`, which seems to be easier for Lean to use. -/
@[simp]
theorem cast_nonneg {α} [OrderedSemiring α] (n : ℕ) : 0 ≤ (n : α) :=
  cast_nonneg' n


/-- Specialisation of `Nat.ofNat_nonneg'`, which seems to be easier for Lean to use. -/
@[simp]
theorem ofNat_nonneg {α} [OrderedSemiring α] (n : ℕ) [n.AtLeastTwo] :
    0 ≤ (ofNat(n) : α) :=
  ofNat_nonneg' n


@[simp, norm_cast]
theorem cast_min {α} [LinearOrderedSemiring α] (m n : ℕ) : (↑(min m n : ℕ) : α) = min (m : α) n :=
  (@mono_cast α _).map_min


@[simp, norm_cast]
theorem cast_max {α} [LinearOrderedSemiring α] (m n : ℕ) : (↑(max m n : ℕ) : α) = max (m : α) n :=
  (@mono_cast α _).map_max


/-- Specialisation of `Nat.cast_pos'`, which seems to be easier for Lean to use. -/
@[simp]
theorem cast_pos {α} [OrderedSemiring α] [Nontrivial α] {n : ℕ} : (0 : α) < n ↔ 0 < n := cast_pos'


/-- See also `Nat.ofNat_pos`, specialised for an `OrderedSemiring`. -/
@[simp low]
theorem ofNat_pos' {n : ℕ} [n.AtLeastTwo] : 0 < (ofNat(n) : α) :=
  cast_pos'.mpr (NeZero.pos n)


/-- Specialisation of `Nat.ofNat_pos'`, which seems to be easier for Lean to use. -/
@[simp]
theorem ofNat_pos {α} [OrderedSemiring α] [Nontrivial α] {n : ℕ} [n.AtLeastTwo] :
    0 < (ofNat(n) : α) :=
  ofNat_pos'


/-- A version of `Nat.cast_sub` that works for `ℝ≥0` and `ℚ≥0`. Note that this proof doesn't work
for `ℕ∞` and `ℝ≥0∞`, so we use type-specific lemmas for these types. -/
@[simp, norm_cast]
theorem cast_tsub [CanonicallyOrderedCommSemiring α] [Sub α] [OrderedSub α]
    [AddLeftReflectLE α] (m n : ℕ) : ↑(m - n) = (m - n : α) := by
  /-
    α : Type u_1
    inst✝³ : CanonicallyOrderedCommSemiring α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    inst✝ : AddLeftReflectLE α
    m n : Nat
    ⊢ Eq (↑(HSub.hSub m n)) (HSub.hSub ↑m ↑n)
  -/
  rcases le_total m n with h | h
    /-
      case inl
      α : Type u_1
      inst✝³ : CanonicallyOrderedCommSemiring α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : AddLeftReflectLE α
      m n : Nat
      h : LE.le m n
      ⊢ Eq (↑(HSub.hSub m n)) (HSub.hSub ↑m ↑n)
    -/
  · rw [Nat.sub_eq_zero_of_le h, cast_zero, tsub_eq_zero_of_le]
    /-
      case inl
      α : Type u_1
      inst✝³ : CanonicallyOrderedCommSemiring α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : AddLeftReflectLE α
      m n : Nat
      h : LE.le m n
      ⊢ LE.le ↑m ↑n
    -/
    exact mono_cast h
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝³ : CanonicallyOrderedCommSemiring α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : AddLeftReflectLE α
      m n : Nat
      h : LE.le n m
      ⊢ Eq (↑(HSub.hSub m n)) (HSub.hSub ↑m ↑n)
    -/
  · rcases le_iff_exists_add'.mp h with ⟨m, rfl⟩
    /-
      case inr.intro
      α : Type u_1
      inst✝³ : CanonicallyOrderedCommSemiring α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : AddLeftReflectLE α
      n m : Nat
      h : LE.le n (HAdd.hAdd m n)
      ⊢ Eq (↑(HSub.hSub (HAdd.hAdd m n) n)) (HSub.hSub ↑(HAdd.hAdd m n) ↑n)
    -/
    rw [add_tsub_cancel_right, cast_add, add_tsub_cancel_right]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem abs_cast [LinearOrderedRing α] (a : ℕ) : |(a : α)| = a :=
  abs_of_nonneg (cast_nonneg a)


@[simp]
theorem abs_ofNat [LinearOrderedRing α] (n : ℕ) [n.AtLeastTwo] :
    |(ofNat(n) : α)| = ofNat(n) :=
  abs_cast n


lemma mul_le_pow {a : ℕ} (ha : a ≠ 1) (b : ℕ) :
    a * b ≤ a ^ b := by
  induction b generalizing a with
  | zero => simp
  | succ b hb =>
    rw [mul_add_one, pow_succ]
    rcases a with (_|_|a)
    · simp
    · simp at ha
    · rw [mul_add_one, mul_add_one, add_comm (_ * a), add_assoc _ (_ * a)]
      rcases b with (_|b)
      · simp [add_assoc, add_comm]
      refine add_le_add (hb (by simp)) ?_
      rw [pow_succ']
      refine (le_add_left ?_ ?_).trans' ?_
      exact le_mul_of_one_le_right' (one_le_pow _ _ (by simp))


lemma two_mul_sq_add_one_le_two_pow_two_mul (k : ℕ) : 2 * k ^ 2 + 1 ≤ 2 ^ (2 * k) := by
  induction k with
  | zero => simp
  | succ k hk =>
    rw [add_pow_two, one_pow, mul_one, add_assoc, mul_add, add_right_comm]
    refine (add_le_add_right hk _).trans ?_
    rw [mul_add 2 k, pow_add, mul_one, pow_two, ← mul_assoc, mul_two, mul_two, add_assoc]
    gcongr
    rw [← two_mul, ← pow_succ']
    exact le_add_of_le_right (mul_le_pow (by simp) _)


