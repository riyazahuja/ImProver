@[simp]
theorem add_right [Distrib R] {a b c : R} : Commute a b → Commute a c → Commute a (b + c) :=
  SemiconjBy.add_right
-- for some reason mathport expected `Semiring` instead of `Distrib`?


@[simp]
theorem add_left [Distrib R] {a b c : R} : Commute a c → Commute b c → Commute (a + b) c :=
  SemiconjBy.add_left
-- for some reason mathport expected `Semiring` instead of `Distrib`?


/-- Representation of a difference of two squares of commuting elements as a product. -/
theorem mul_self_sub_mul_self_eq [NonUnitalNonAssocRing R] {a b : R} (h : Commute a b) :
    a * a - b * b = (a + b) * (a - b) := by
  /-
    R : Type u
    inst✝ : NonUnitalNonAssocRing R
    a b : R
    h : Commute a b
    ⊢ Eq (HSub.hSub (HMul.hMul a a) (HMul.hMul b b)) (HMul.hMul (HAdd.hAdd a b) (H …
  -/
  rw [add_mul, mul_sub, mul_sub, h.eq, sub_add_sub_cancel]
  /-
    🎉 no goals
  -/


theorem mul_self_sub_mul_self_eq' [NonUnitalNonAssocRing R] {a b : R} (h : Commute a b) :
    a * a - b * b = (a - b) * (a + b) := by
  /-
    R : Type u
    inst✝ : NonUnitalNonAssocRing R
    a b : R
    h : Commute a b
    ⊢ Eq (HSub.hSub (HMul.hMul a a) (HMul.hMul b b)) (HMul.hMul (HSub.hSub a b) (H …
  -/
  rw [mul_add, sub_mul, sub_mul, h.eq, sub_add_sub_cancel]
  /-
    🎉 no goals
  -/


theorem mul_self_eq_mul_self_iff [NonUnitalNonAssocRing R] [NoZeroDivisors R] {a b : R}
    (h : Commute a b) : a * a = b * b ↔ a = b ∨ a = -b := by
  rw [← sub_eq_zero, h.mul_self_sub_mul_self_eq, mul_eq_zero, or_comm, sub_eq_zero,
    add_eq_zero_iff_eq_neg]


theorem neg_right : Commute a b → Commute a (-b) :=
  SemiconjBy.neg_right


@[simp]
theorem neg_right_iff : Commute a (-b) ↔ Commute a b :=
  SemiconjBy.neg_right_iff


theorem neg_left : Commute a b → Commute (-a) b :=
  SemiconjBy.neg_left


@[simp]
theorem neg_left_iff : Commute (-a) b ↔ Commute a b :=
  SemiconjBy.neg_left_iff


theorem neg_one_right (a : R) : Commute a (-1) :=
  SemiconjBy.neg_one_right a


theorem neg_one_left (a : R) : Commute (-1) a :=
  SemiconjBy.neg_one_left a


@[simp]
theorem sub_right : Commute a b → Commute a c → Commute a (b - c) :=
  SemiconjBy.sub_right


@[simp]
theorem sub_left : Commute a c → Commute b c → Commute (a - b) c :=
  SemiconjBy.sub_left


protected lemma sq_sub_sq (h : Commute a b) : a ^ 2 - b ^ 2 = (a + b) * (a - b) := by
  /-
    R : Type u
    inst✝ : Ring R
    a b : R
    h : Commute a b
    ⊢ Eq (HSub.hSub (HPow.hPow a 2) (HPow.hPow b 2)) (HMul.hMul (HAdd.hAdd a b) (H …
  -/
  rw [sq, sq, h.mul_self_sub_mul_self_eq]
  /-
    🎉 no goals
  -/


protected lemma sq_eq_sq_iff_eq_or_eq_neg (h : Commute a b) : a ^ 2 = b ^ 2 ↔ a = b ∨ a = -b := by
  /-
    R : Type u
    inst✝¹ : Ring R
    a b : R
    inst✝ : NoZeroDivisors R
    h : Commute a b
    ⊢ Iff (Eq (HPow.hPow a 2) (HPow.hPow b 2)) (Or (Eq a b) (Eq a (Neg.neg b)))
  -/
  rw [← sub_eq_zero, h.sq_sub_sq, mul_eq_zero, add_eq_zero_iff_eq_neg, sub_eq_zero, or_comm]
  /-
    🎉 no goals
  -/


lemma neg_one_pow_eq_or : ∀ n : ℕ, (-1 : R) ^ n = 1 ∨ (-1 : R) ^ n = -1
  | 0 => Or.inl (pow_zero _)
  | n + 1 => (neg_one_pow_eq_or n).symm.imp
                /-
                  R : Type u
                  inst✝¹ : Monoid R
                  inst✝ : HasDistribNeg R
                  n : Nat
                  h : Eq (HPow.hPow (-1) n) (-1)
                  ⊢ Eq (HPow.hPow (-1) (HAdd.hAdd n 1)) 1
                -/
    (fun h ↦ by rw [pow_succ, h, neg_one_mul, neg_neg])
                /-
                  🎉 no goals
                -/
                /-
                  R : Type u
                  inst✝¹ : Monoid R
                  inst✝ : HasDistribNeg R
                  n : Nat
                  h : Eq (HPow.hPow (-1) n) 1
                  ⊢ Eq (HPow.hPow (-1) (HAdd.hAdd n 1)) (-1)
                -/
    (fun h ↦ by rw [pow_succ, h, one_mul])
                /-
                  🎉 no goals
                -/


lemma neg_pow (a : R) (n : ℕ) : (-a) ^ n = (-1) ^ n * a ^ n :=
  neg_one_mul a ▸ (Commute.neg_one_left a).mul_pow n


lemma neg_pow' (a : R) (n : ℕ) : (-a) ^ n = a ^ n * (-1) ^ n :=
  mul_neg_one a ▸ (Commute.neg_one_right a).mul_pow n


                                              /-
                                                R : Type u
                                                inst✝¹ : Monoid R
                                                inst✝ : HasDistribNeg R
                                                a : R
                                                ⊢ Eq (HPow.hPow (Neg.neg a) 2) (HPow.hPow a 2)
                                              -/
lemma neg_sq (a : R) : (-a) ^ 2 = a ^ 2 := by simp [sq]
                                              /-
                                                🎉 no goals
                                              -/

-- Porting note: removed the simp attribute to please the simpNF linter

                                          /-
                                            R : Type u
                                            inst✝¹ : Monoid R
                                            inst✝ : HasDistribNeg R
                                            ⊢ Eq (HPow.hPow (-1) 2) 1
                                          -/
lemma neg_one_sq : (-1 : R) ^ 2 = 1 := by simp [neg_sq, one_pow]
                                          /-
                                            🎉 no goals
                                          -/


alias neg_pow_two := neg_sq


alias neg_one_pow_two := neg_one_sq


@[simp] lemma neg_one_pow_mul_eq_zero_iff : (-1) ^ n * a = 0 ↔ a = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    a : R
    n : Nat
    ⊢ Iff (Eq (HMul.hMul (HPow.hPow (-1) n) a) 0) (Eq a 0)
  -/
                                              /-
                                                🎉 no goals
                                              -/
  rcases neg_one_pow_eq_or R n with h | h <;> simp [h]
                                              /-
                                                🎉 no goals
                                              -/


@[simp] lemma mul_neg_one_pow_eq_zero_iff : a * (-1) ^ n = 0 ↔ a = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    a : R
    n : Nat
    ⊢ Iff (Eq (HMul.hMul a (HPow.hPow (-1) n)) 0) (Eq a 0)
  -/
                                            /-
                                              🎉 no goals
                                            -/
  obtain h | h := neg_one_pow_eq_or R n <;> simp [h]
                                            /-
                                              🎉 no goals
                                            -/


lemma neg_one_pow_eq_pow_mod_two (n : ℕ) : (-1 : R) ^ n = (-1) ^ (n % 2) := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    ⊢ Eq (HPow.hPow (-1) n) (HPow.hPow (-1) (HMod.hMod n 2))
  -/
  rw [← Nat.mod_add_div n 2, pow_add, pow_mul]; simp [sq]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp] lemma sq_eq_one_iff : a ^ 2 = 1 ↔ a = 1 ∨ a = -1 := by
  /-
    R : Type u
    inst✝¹ : Ring R
    a : R
    inst✝ : NoZeroDivisors R
    ⊢ Iff (Eq (HPow.hPow a 2) 1) (Or (Eq a 1) (Eq a (-1)))
  -/
  rw [← (Commute.one_right a).sq_eq_sq_iff_eq_or_eq_neg, one_pow]
  /-
    🎉 no goals
  -/


lemma sq_ne_one_iff : a ^ 2 ≠ 1 ↔ a ≠ 1 ∧ a ≠ -1 := sq_eq_one_iff.not.trans not_or


/-- Representation of a difference of two squares in a commutative ring as a product. -/
theorem mul_self_sub_mul_self [CommRing R] (a b : R) : a * a - b * b = (a + b) * (a - b) :=
  (Commute.all a b).mul_self_sub_mul_self_eq


theorem mul_self_sub_one [NonAssocRing R] (a : R) : a * a - 1 = (a + 1) * (a - 1) := by
  /-
    R : Type u
    inst✝ : NonAssocRing R
    a : R
    ⊢ Eq (HSub.hSub (HMul.hMul a a) 1) (HMul.hMul (HAdd.hAdd a 1) (HSub.hSub a 1))
  -/
  rw [← (Commute.one_right a).mul_self_sub_mul_self_eq, mul_one]
  /-
    🎉 no goals
  -/


theorem mul_self_eq_mul_self_iff [CommRing R] [NoZeroDivisors R] {a b : R} :
    a * a = b * b ↔ a = b ∨ a = -b :=
  (Commute.all a b).mul_self_eq_mul_self_iff


theorem mul_self_eq_one_iff [NonAssocRing R] [NoZeroDivisors R] {a : R} :
    a * a = 1 ↔ a = 1 ∨ a = -1 := by
  /-
    R : Type u
    inst✝¹ : NonAssocRing R
    inst✝ : NoZeroDivisors R
    a : R
    ⊢ Iff (Eq (HMul.hMul a a) 1) (Or (Eq a 1) (Eq a (-1)))
  -/
  rw [← (Commute.one_right a).mul_self_eq_mul_self_iff, mul_one]
  /-
    🎉 no goals
  -/


lemma sq_sub_sq (a b : R) : a ^ 2 - b ^ 2 = (a + b) * (a - b) := (Commute.all a b).sq_sub_sq


alias pow_two_sub_pow_two := sq_sub_sq


lemma sub_sq (a b : R) : (a - b) ^ 2 = a ^ 2 - 2 * a * b + b ^ 2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    a b : R
    ⊢ Eq (HPow.hPow (HSub.hSub a b) 2) (HAdd.hAdd (HSub.hSub (HPow.hPow a 2) (HMul …
  -/
  rw [sub_eq_add_neg, add_sq, neg_sq, mul_neg, ← sub_eq_add_neg]
  /-
    🎉 no goals
  -/


alias sub_pow_two := sub_sq


lemma sub_sq' (a b : R) : (a - b) ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b := by
  /-
    R : Type u
    inst✝ : CommRing R
    a b : R
    ⊢ Eq (HPow.hPow (HSub.hSub a b) 2) (HSub.hSub (HAdd.hAdd (HPow.hPow a 2) (HPow …
  -/
  rw [sub_eq_add_neg, add_sq', neg_sq, mul_neg, ← sub_eq_add_neg]
  /-
    🎉 no goals
  -/


lemma sq_eq_sq_iff_eq_or_eq_neg : a ^ 2 = b ^ 2 ↔ a = b ∨ a = -b :=
  (Commute.all a b).sq_eq_sq_iff_eq_or_eq_neg


lemma eq_or_eq_neg_of_sq_eq_sq (a b : R) : a ^ 2 = b ^ 2 → a = b ∨ a = -b :=
  sq_eq_sq_iff_eq_or_eq_neg.1

-- Copies of the above CommRing lemmas for `Units R`.

protected lemma sq_eq_sq_iff_eq_or_eq_neg {a b : Rˣ} : a ^ 2 = b ^ 2 ↔ a = b ∨ a = -b := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    a b : Units R
    ⊢ Iff (Eq (HPow.hPow a 2) (HPow.hPow b 2)) (Or (Eq a b) (Eq a (Neg.neg b)))
  -/
  simp_rw [Units.ext_iff, val_pow_eq_pow_val, sq_eq_sq_iff_eq_or_eq_neg, Units.val_neg]
  /-
    🎉 no goals
  -/


protected lemma eq_or_eq_neg_of_sq_eq_sq (a b : Rˣ) (h : a ^ 2 = b ^ 2) : a = b ∨ a = -b :=
  Units.sq_eq_sq_iff_eq_or_eq_neg.1 h


/-- In the unit group of an integral domain, a unit is its own inverse iff the unit is one or
  one's additive inverse. -/
theorem inv_eq_self_iff [Ring R] [NoZeroDivisors R] (u : Rˣ) : u⁻¹ = u ↔ u = 1 ∨ u = -1 := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : NoZeroDivisors R
    u : Units R
    ⊢ Iff (Eq (Inv.inv u) u) (Or (Eq u 1) (Eq u (-1)))
  -/
  rw [inv_eq_iff_mul_eq_one]
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : NoZeroDivisors R
    u : Units R
    ⊢ Iff (Eq (HMul.hMul u u) 1) (Or (Eq u 1) (Eq u (-1)))
  -/
  simp only [Units.ext_iff]
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : NoZeroDivisors R
    u : Units R
    ⊢ Iff (Eq ↑(HMul.hMul u u) ↑1) (Or (Eq ↑u ↑1) (Eq ↑u ↑(-1)))
  -/
  push_cast
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : NoZeroDivisors R
    u : Units R
    ⊢ Iff (Eq (HMul.hMul ↑u ↑u) 1) (Or (Eq (↑u) 1) (Eq (↑u) (-1)))
  -/
  exact mul_self_eq_one_iff
  /-
    🎉 no goals
  -/


instance (priority := 100) instBracket : Bracket R R := ⟨fun x y => x * y - y * x⟩


theorem lie_def (x y : R) : ⁅x, y⁆ = x * y - y * x := rfl


theorem commute_iff_lie_eq {x y : R} : Commute x y ↔ ⁅x, y⁆ = 0 := sub_eq_zero.symm


theorem Commute.lie_eq {x y : R} (h : Commute x y) : ⁅x, y⁆ = 0 := sub_eq_zero_of_eq h


