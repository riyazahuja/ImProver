/-- `-⅟a` is the inverse of `-a` -/
def invertibleNeg [Mul α] [One α] [HasDistribNeg α] (a : α) [Invertible a] : Invertible (-a) :=
            /-
              α : Type u
              inst✝³ : Mul α
              inst✝² : One α
              inst✝¹ : HasDistribNeg α
              a : α
              inst✝ : Invertible a
              ⊢ Eq (HMul.hMul (Neg.neg (Invertible.invOf a)) (Neg.neg a)) 1
            -/
            /-
              🎉 no goals
            -/
  ⟨-⅟ a, by simp, by simp⟩
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem invOf_neg [Monoid α] [HasDistribNeg α] (a : α) [Invertible a] [Invertible (-a)] :
    ⅟ (-a) = -⅟ a :=
                         /-
                           α : Type u
                           inst✝³ : Monoid α
                           inst✝² : HasDistribNeg α
                           a : α
                           inst✝¹ : Invertible a
                           inst✝ : Invertible (Neg.neg a)
                           ⊢ Eq (HMul.hMul (Neg.neg a) (Neg.neg (Invertible.invOf a))) 1
                         -/
  invOf_eq_right_inv (by simp)
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem one_sub_invOf_two [Ring α] [Invertible (2 : α)] : 1 - (⅟ 2 : α) = ⅟ 2 :=
  (isUnit_of_invertible (2 : α)).mul_right_inj.1 <| by
    /-
      α : Type u
      inst✝¹ : Ring α
      inst✝ : Invertible 2
      ⊢ Eq (HMul.hMul 2 (HSub.hSub 1 (Invertible.invOf 2))) (HMul.hMul 2 (Invertible …
    -/
    rw [mul_sub, mul_invOf_self, mul_one, ← one_add_one_eq_two, add_sub_cancel_right]
    /-
      🎉 no goals
    -/


@[simp]
theorem invOf_two_add_invOf_two [NonAssocSemiring α] [Invertible (2 : α)] :
                                    /-
                                      α : Type u
                                      inst✝¹ : NonAssocSemiring α
                                      inst✝ : Invertible 2
                                      ⊢ Eq (HAdd.hAdd (Invertible.invOf 2) (Invertible.invOf 2)) 1
                                    -/
    (⅟ 2 : α) + (⅟ 2 : α) = 1 := by rw [← two_mul, mul_invOf_self]
                                    /-
                                      🎉 no goals
                                    -/


theorem pos_of_invertible_cast [Semiring α] [Nontrivial α] (n : ℕ) [Invertible (n : α)] : 0 < n :=
  Nat.zero_lt_of_ne_zero fun h => Invertible.ne_zero (n : α) (h ▸ Nat.cast_zero)


theorem invOf_add_invOf [Semiring α] (a b : α) [Invertible a] [Invertible b] :
    ⅟a + ⅟b = ⅟a * (a + b) * ⅟b := by
  /-
    α : Type u
    inst✝² : Semiring α
    a b : α
    inst✝¹ : Invertible a
    inst✝ : Invertible b
    ⊢ Eq (HAdd.hAdd (Invertible.invOf a) (Invertible.invOf b)) (HMul.hMul (HMul.hM …
  -/
  rw [mul_add, invOf_mul_self, add_mul, one_mul, mul_assoc, mul_invOf_self, mul_one, add_comm]
  /-
    🎉 no goals
  -/


/-- A version of `inv_sub_inv'` for `invOf`. -/
theorem invOf_sub_invOf [Ring α] (a b : α) [Invertible a] [Invertible b] :
    ⅟a - ⅟b = ⅟a * (b - a) * ⅟b := by
  /-
    α : Type u
    inst✝² : Ring α
    a b : α
    inst✝¹ : Invertible a
    inst✝ : Invertible b
    ⊢ Eq (HSub.hSub (Invertible.invOf a) (Invertible.invOf b)) (HMul.hMul (HMul.hM …
  -/
  rw [mul_sub, invOf_mul_self, sub_mul, one_mul, mul_assoc, mul_invOf_self, mul_one]
  /-
    🎉 no goals
  -/


/-- A version of `inv_add_inv'` for `Ring.inverse`. -/
theorem Ring.inverse_add_inverse [Semiring α] {a b : α} (h : IsUnit a ↔ IsUnit b) :
    Ring.inverse a + Ring.inverse b = Ring.inverse a * (a + b) * Ring.inverse b := by
  /-
    α : Type u
    inst✝ : Semiring α
    a b : α
    h : Iff (IsUnit a) (IsUnit b)
    ⊢ Eq (HAdd.hAdd (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
  -/
  by_cases ha : IsUnit a
    /-
      case pos
      α : Type u
      inst✝ : Semiring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : IsUnit a
      ⊢ Eq (HAdd.hAdd (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
  · have hb := h.mp ha
    /-
      case pos
      α : Type u
      inst✝ : Semiring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : IsUnit a
      hb : IsUnit b
      ⊢ Eq (HAdd.hAdd (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
    obtain ⟨ia⟩ := ha.nonempty_invertible
    /-
      case pos.intro
      α : Type u
      inst✝ : Semiring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : IsUnit a
      hb : IsUnit b
      ia : Invertible a
      ⊢ Eq (HAdd.hAdd (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
    obtain ⟨ib⟩ := hb.nonempty_invertible
    /-
      case pos.intro.intro
      α : Type u
      inst✝ : Semiring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : IsUnit a
      hb : IsUnit b
      ia : Invertible a
      ib : Invertible b
      ⊢ Eq (HAdd.hAdd (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
    simp_rw [inverse_invertible, invOf_add_invOf]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝ : Semiring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : Not (IsUnit a)
      ⊢ Eq (HAdd.hAdd (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
  · have hb := h.not.mp ha
    /-
      case neg
      α : Type u
      inst✝ : Semiring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : Not (IsUnit a)
      hb : Not (IsUnit b)
      ⊢ Eq (HAdd.hAdd (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
    simp [inverse_non_unit, ha, hb]
    /-
      🎉 no goals
    -/


/-- A version of `inv_sub_inv'` for `Ring.inverse`. -/
theorem Ring.inverse_sub_inverse [Ring α] {a b : α} (h : IsUnit a ↔ IsUnit b) :
    Ring.inverse a - Ring.inverse b = Ring.inverse a * (b - a) * Ring.inverse b := by
  /-
    α : Type u
    inst✝ : Ring α
    a b : α
    h : Iff (IsUnit a) (IsUnit b)
    ⊢ Eq (HSub.hSub (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
  -/
  by_cases ha : IsUnit a
    /-
      case pos
      α : Type u
      inst✝ : Ring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : IsUnit a
      ⊢ Eq (HSub.hSub (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
  · have hb := h.mp ha
    /-
      case pos
      α : Type u
      inst✝ : Ring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : IsUnit a
      hb : IsUnit b
      ⊢ Eq (HSub.hSub (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
    obtain ⟨ia⟩ := ha.nonempty_invertible
    /-
      case pos.intro
      α : Type u
      inst✝ : Ring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : IsUnit a
      hb : IsUnit b
      ia : Invertible a
      ⊢ Eq (HSub.hSub (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
    obtain ⟨ib⟩ := hb.nonempty_invertible
    /-
      case pos.intro.intro
      α : Type u
      inst✝ : Ring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : IsUnit a
      hb : IsUnit b
      ia : Invertible a
      ib : Invertible b
      ⊢ Eq (HSub.hSub (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
    simp_rw [inverse_invertible, invOf_sub_invOf]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝ : Ring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : Not (IsUnit a)
      ⊢ Eq (HSub.hSub (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
  · have hb := h.not.mp ha
    /-
      case neg
      α : Type u
      inst✝ : Ring α
      a b : α
      h : Iff (IsUnit a) (IsUnit b)
      ha : Not (IsUnit a)
      hb : Not (IsUnit b)
      ⊢ Eq (HSub.hSub (Ring.inverse a) (Ring.inverse b)) (HMul.hMul (HMul.hMul (Ring …
    -/
    simp [inverse_non_unit, ha, hb]
    /-
      🎉 no goals
    -/

