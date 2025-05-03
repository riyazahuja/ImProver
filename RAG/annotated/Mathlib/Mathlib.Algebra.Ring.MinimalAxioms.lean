/-- Define a `Ring` structure on a Type by proving a minimized set of axioms.
Note that this uses the default definitions for `npow`, `nsmul`, `zsmul` and `sub`
See note [reducible non-instances]. -/
abbrev Ring.ofMinimalAxioms {R : Type u}
    [Add R] [Mul R] [Neg R] [Zero R] [One R]
    (add_assoc : ∀ a b c : R, a + b + c = a + (b + c))
    (zero_add : ∀ a : R, 0 + a = a)
    (neg_add_cancel : ∀ a : R, -a + a = 0)
    (mul_assoc : ∀ a b c : R, a * b * c = a * (b * c))
    (one_mul : ∀ a : R, 1 * a = a)
    (mul_one : ∀ a : R, a * 1 = a)
    (left_distrib : ∀ a b c : R, a * (b + c) = a * b + a * c)
    (right_distrib : ∀ a b c : R, (a + b) * c = a * c + b * c) : Ring R :=
  letI := AddGroup.ofLeftAxioms add_assoc zero_add neg_add_cancel
  haveI add_comm : ∀ a b, a + b = b + a := by
    /-
      R : Type u
      inst✝⁴ : Add R
      inst✝³ : Mul R
      inst✝² : Neg R
      inst✝¹ : Zero R
      inst✝ : One R
      add_assoc : ∀ (a b c : R), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd …
      zero_add : ∀ (a : R), Eq (HAdd.hAdd 0 a) a
      neg_add_cancel : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
      mul_assoc : ∀ (a b c : R), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul …
      one_mul : ∀ (a : R), Eq (HMul.hMul 1 a) a
      mul_one : ∀ (a : R), Eq (HMul.hMul a 1) a
      left_distrib : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMu …
      right_distrib : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HM …
      this : AddGroup R := AddGroup.ofLeftAxioms add_assoc zero_add neg_add_cancel
      ⊢ ∀ (a b : R), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
    -/
    intro a b
    have h₁ : (1 + 1 : R) * (a + b) = a + (a + b) + b := by
      rw [left_distrib]
      simp only [right_distrib, one_mul, add_assoc]
    have h₂ : (1 + 1 : R) * (a + b) = a + (b + a) + b := by
      rw [right_distrib]
      simp only [left_distrib, one_mul, add_assoc]
    /-
      R : Type u
      inst✝⁴ : Add R
      inst✝³ : Mul R
      inst✝² : Neg R
      inst✝¹ : Zero R
      inst✝ : One R
      add_assoc : ∀ (a b c : R), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd …
      zero_add : ∀ (a : R), Eq (HAdd.hAdd 0 a) a
      neg_add_cancel : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
      mul_assoc : ∀ (a b c : R), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul …
      one_mul : ∀ (a : R), Eq (HMul.hMul 1 a) a
      mul_one : ∀ (a : R), Eq (HMul.hMul a 1) a
      left_distrib : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMu …
      right_distrib : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HM …
      this : AddGroup R := AddGroup.ofLeftAxioms add_assoc zero_add neg_add_cancel
      a b : R
      h₁ : Eq (HMul.hMul (HAdd.hAdd 1 1) (HAdd.hAdd a b)) (HAdd.hAdd (HAdd.hAdd a (H …
      h₂ : Eq (HMul.hMul (HAdd.hAdd 1 1) (HAdd.hAdd a b)) (HAdd.hAdd (HAdd.hAdd a (H …
      ⊢ Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
    -/
    have := h₁.symm.trans h₂
    /-
      R : Type u
      inst✝⁴ : Add R
      inst✝³ : Mul R
      inst✝² : Neg R
      inst✝¹ : Zero R
      inst✝ : One R
      add_assoc : ∀ (a b c : R), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd …
      zero_add : ∀ (a : R), Eq (HAdd.hAdd 0 a) a
      neg_add_cancel : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
      mul_assoc : ∀ (a b c : R), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul …
      one_mul : ∀ (a : R), Eq (HMul.hMul 1 a) a
      mul_one : ∀ (a : R), Eq (HMul.hMul a 1) a
      left_distrib : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMu …
      right_distrib : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HM …
      this✝ : AddGroup R := AddGroup.ofLeftAxioms add_assoc zero_add neg_add_cancel
      a b : R
      h₁ : Eq (HMul.hMul (HAdd.hAdd 1 1) (HAdd.hAdd a b)) (HAdd.hAdd (HAdd.hAdd a (H …
      h₂ : Eq (HMul.hMul (HAdd.hAdd 1 1) (HAdd.hAdd a b)) (HAdd.hAdd (HAdd.hAdd a (H …
      this : Eq (HAdd.hAdd (HAdd.hAdd a (HAdd.hAdd a b)) b) (HAdd.hAdd (HAdd.hAdd a  …
      ⊢ Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
    -/
    rwa [add_left_inj, add_right_inj] at this
    /-
      🎉 no goals
    -/
  haveI zero_mul : ∀ a, (0 : R) * a = 0 := fun a => by
    have : 0 * a = 0 * a + 0 * a :=
      calc 0 * a = (0 + 0) * a := by rw [zero_add]
      _ = 0 * a + 0 * a := by rw [right_distrib]
    /-
      R : Type u
      inst✝⁴ : Add R
      inst✝³ : Mul R
      inst✝² : Neg R
      inst✝¹ : Zero R
      inst✝ : One R
      add_assoc : ∀ (a b c : R), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd …
      zero_add : ∀ (a : R), Eq (HAdd.hAdd 0 a) a
      neg_add_cancel : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
      mul_assoc : ∀ (a b c : R), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul …
      one_mul : ∀ (a : R), Eq (HMul.hMul 1 a) a
      mul_one : ∀ (a : R), Eq (HMul.hMul a 1) a
      left_distrib : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMu …
      right_distrib : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HM …
      this✝ : AddGroup R := AddGroup.ofLeftAxioms add_assoc zero_add neg_add_cancel
      add_comm : ∀ (a b : R), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
      a : R
      this : Eq (HMul.hMul 0 a) (HAdd.hAdd (HMul.hMul 0 a) (HMul.hMul 0 a))
      ⊢ Eq (HMul.hMul 0 a) 0
    -/
    rwa [self_eq_add_right] at this
    /-
      🎉 no goals
    -/
  haveI mul_zero : ∀ a, a * (0 : R) = 0 := fun a => by
    have : a * 0 = a * 0 + a * 0 :=
      calc a * 0 = a * (0 + 0) := by rw [zero_add]
      _ = a * 0 + a * 0 := by rw [left_distrib]
    /-
      R : Type u
      inst✝⁴ : Add R
      inst✝³ : Mul R
      inst✝² : Neg R
      inst✝¹ : Zero R
      inst✝ : One R
      add_assoc : ∀ (a b c : R), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd …
      zero_add : ∀ (a : R), Eq (HAdd.hAdd 0 a) a
      neg_add_cancel : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
      mul_assoc : ∀ (a b c : R), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul …
      one_mul : ∀ (a : R), Eq (HMul.hMul 1 a) a
      mul_one : ∀ (a : R), Eq (HMul.hMul a 1) a
      left_distrib : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMu …
      right_distrib : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HM …
      this✝ : AddGroup R := AddGroup.ofLeftAxioms add_assoc zero_add neg_add_cancel
      add_comm : ∀ (a b : R), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
      zero_mul : ∀ (a : R), Eq (HMul.hMul 0 a) 0
      a : R
      this : Eq (HMul.hMul a 0) (HAdd.hAdd (HMul.hMul a 0) (HMul.hMul a 0))
      ⊢ Eq (HMul.hMul a 0) 0
    -/
    rwa [self_eq_add_right] at this
    /-
      🎉 no goals
    -/
  { add_comm := add_comm
    left_distrib := left_distrib
    right_distrib := right_distrib
    zero_mul := zero_mul
    mul_zero := mul_zero
    mul_assoc := mul_assoc
    one_mul := one_mul
    mul_one := mul_one
    neg_add_cancel := neg_add_cancel
    zsmul := (· • ·) }


/-- Define a `CommRing` structure on a Type by proving a minimized set of axioms.
Note that this uses the default definitions for `npow`, `nsmul`, `zsmul` and `sub`
See note [reducible non-instances]. -/
abbrev CommRing.ofMinimalAxioms {R : Type u}
    [Add R] [Mul R] [Neg R] [Zero R] [One R]
    (add_assoc : ∀ a b c : R, a + b + c = a + (b + c))
    (zero_add : ∀ a : R, 0 + a = a)
    (neg_add_cancel : ∀ a : R, -a + a = 0)
    (mul_assoc : ∀ a b c : R, a * b * c = a * (b * c))
    (mul_comm : ∀ a b : R, a * b = b * a)
    (one_mul : ∀ a : R, 1 * a = a)
    (left_distrib : ∀ a b c : R, a * (b + c) = a * b + a * c) : CommRing R :=
  haveI mul_one : ∀ a : R, a * 1 = a := fun a => by
    /-
      R : Type u
      inst✝⁴ : Add R
      inst✝³ : Mul R
      inst✝² : Neg R
      inst✝¹ : Zero R
      inst✝ : One R
      add_assoc : ∀ (a b c : R), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd …
      zero_add : ∀ (a : R), Eq (HAdd.hAdd 0 a) a
      neg_add_cancel : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
      mul_assoc : ∀ (a b c : R), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul …
      mul_comm : ∀ (a b : R), Eq (HMul.hMul a b) (HMul.hMul b a)
      one_mul : ∀ (a : R), Eq (HMul.hMul 1 a) a
      left_distrib : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMu …
      a : R
      ⊢ Eq (HMul.hMul a 1) a
    -/
    rw [mul_comm, one_mul]
    /-
      🎉 no goals
    -/
  haveI right_distrib : ∀ a b c : R, (a + b) * c = a * c + b * c := fun a b c => by
    /-
      R : Type u
      inst✝⁴ : Add R
      inst✝³ : Mul R
      inst✝² : Neg R
      inst✝¹ : Zero R
      inst✝ : One R
      add_assoc : ∀ (a b c : R), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd …
      zero_add : ∀ (a : R), Eq (HAdd.hAdd 0 a) a
      neg_add_cancel : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
      mul_assoc : ∀ (a b c : R), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul …
      mul_comm : ∀ (a b : R), Eq (HMul.hMul a b) (HMul.hMul b a)
      one_mul : ∀ (a : R), Eq (HMul.hMul 1 a) a
      left_distrib : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMu …
      mul_one : ∀ (a : R), Eq (HMul.hMul a 1) a
      a b c : R
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
    -/
    rw [mul_comm, left_distrib, mul_comm, mul_comm b c]
    /-
      🎉 no goals
    -/
  letI := Ring.ofMinimalAxioms add_assoc zero_add neg_add_cancel mul_assoc
    one_mul mul_one left_distrib right_distrib
  { mul_comm := mul_comm }

