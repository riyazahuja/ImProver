/-- Discriminant of a quadratic -/
def discrim [Ring R] (a b c : R) : R :=
  b ^ 2 - 4 * a * c


@[simp] lemma discrim_neg [Ring R] (a b c : R) : discrim (-a) (-b) (-c) = discrim a b c := by
  /-
    R : Type u_1
    inst✝ : Ring R
    a b c : R
    ⊢ Eq (discrim (Neg.neg a) (Neg.neg b) (Neg.neg c)) (discrim a b c)
  -/
  simp [discrim]
  /-
    🎉 no goals
  -/


lemma discrim_eq_sq_of_quadratic_eq_zero {x : R} (h : a * (x * x) + b * x + c = 0) :
    discrim a b c = (2 * a * x + b) ^ 2 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b c x : R
    h : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul.hMul b x)) c) 0
    ⊢ Eq (discrim a b c) (HPow.hPow (HAdd.hAdd (HMul.hMul (HMul.hMul 2 a) x) b) 2)
  -/
  rw [discrim]
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b c x : R
    h : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul.hMul b x)) c) 0
    ⊢ Eq (HSub.hSub (HPow.hPow b 2) (HMul.hMul (HMul.hMul 4 a) c)) (HPow.hPow (HAd …
  -/
  linear_combination -4 * a * h
  /-
    🎉 no goals
  -/


/-- A quadratic has roots if and only if its discriminant equals some square.
-/
theorem quadratic_eq_zero_iff_discrim_eq_sq [NeZero (2 : R)] [NoZeroDivisors R]
    (ha : a ≠ 0) (x : R) :
    a * (x * x) + b * x + c = 0 ↔ discrim a b c = (2 * a * x + b) ^ 2 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    a b c : R
    inst✝¹ : NeZero 2
    inst✝ : NoZeroDivisors R
    ha : Ne a 0
    x : R
    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul.hMul b x)) …
  -/
  refine ⟨discrim_eq_sq_of_quadratic_eq_zero, fun h ↦ ?_⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    a b c : R
    inst✝¹ : NeZero 2
    inst✝ : NoZeroDivisors R
    ha : Ne a 0
    x : R
    h : Eq (discrim a b c) (HPow.hPow (HAdd.hAdd (HMul.hMul (HMul.hMul 2 a) x) b) 2)
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul.hMul b x)) c) 0
  -/
  rw [discrim] at h
  /-
    R : Type u_1
    inst✝² : CommRing R
    a b c : R
    inst✝¹ : NeZero 2
    inst✝ : NoZeroDivisors R
    ha : Ne a 0
    x : R
    h : Eq (HSub.hSub (HPow.hPow b 2) (HMul.hMul (HMul.hMul 4 a) c)) (HPow.hPow (H …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul.hMul b x)) c) 0
  -/
  have ha : 2 * 2 * a ≠ 0 := mul_ne_zero (mul_ne_zero (NeZero.ne _) (NeZero.ne _)) ha
  /-
    R : Type u_1
    inst✝² : CommRing R
    a b c : R
    inst✝¹ : NeZero 2
    inst✝ : NoZeroDivisors R
    ha✝ : Ne a 0
    x : R
    h : Eq (HSub.hSub (HPow.hPow b 2) (HMul.hMul (HMul.hMul 4 a) c)) (HPow.hPow (H …
    ha : Ne (HMul.hMul (HMul.hMul 2 2) a) 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul.hMul b x)) c) 0
  -/
  apply mul_left_cancel₀ ha
  /-
    R : Type u_1
    inst✝² : CommRing R
    a b c : R
    inst✝¹ : NeZero 2
    inst✝ : NoZeroDivisors R
    ha✝ : Ne a 0
    x : R
    h : Eq (HSub.hSub (HPow.hPow b 2) (HMul.hMul (HMul.hMul 4 a) c)) (HPow.hPow (H …
    ha : Ne (HMul.hMul (HMul.hMul 2 2) a) 0
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 2) a) (HAdd.hAdd (HAdd.hAdd (HMul.hMul …
  -/
  linear_combination -h
  /-
    🎉 no goals
  -/


/-- A quadratic has no root if its discriminant has no square root. -/
theorem quadratic_ne_zero_of_discrim_ne_sq (h : ∀ s : R, discrim a b c ≠ s^2) (x : R) :
    a * (x * x) + b * x + c ≠ 0 :=
  mt discrim_eq_sq_of_quadratic_eq_zero (h _)


/-- Roots of a quadratic equation. -/
theorem quadratic_eq_zero_iff (ha : a ≠ 0) {s : K} (h : discrim a b c = s * s) (x : K) :
    a * (x * x) + b * x + c = 0 ↔ x = (-b + s) / (2 * a) ∨ x = (-b - s) / (2 * a) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NeZero 2
    a b c : K
    ha : Ne a 0
    s : K
    h : Eq (discrim a b c) (HMul.hMul s s)
    x : K
    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul.hMul b x)) …
  -/
  rw [quadratic_eq_zero_iff_discrim_eq_sq ha, h, sq, mul_self_eq_mul_self_iff]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NeZero 2
    a b c : K
    ha : Ne a 0
    s : K
    h : Eq (discrim a b c) (HMul.hMul s s)
    x : K
    ⊢ Iff (Or (Eq s (HAdd.hAdd (HMul.hMul (HMul.hMul 2 a) x) b)) (Eq s (Neg.neg (H …
  -/
  field_simp
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NeZero 2
    a b c : K
    ha : Ne a 0
    s : K
    h : Eq (discrim a b c) (HMul.hMul s s)
    x : K
    ⊢ Iff (Or (Eq s (HAdd.hAdd (HMul.hMul (HMul.hMul 2 a) x) b)) (Eq s (HAdd.hAdd  …
  -/
  apply or_congr
    /-
      case h₁
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NeZero 2
      a b c : K
      ha : Ne a 0
      s : K
      h : Eq (discrim a b c) (HMul.hMul s s)
      x : K
      ⊢ Iff (Eq s (HAdd.hAdd (HMul.hMul (HMul.hMul 2 a) x) b)) (Eq (HMul.hMul x (HMu …
    -/
                                 /-
                                   🎉 no goals
                                 -/
  · constructor <;> intro h' <;> linear_combination -h'
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case h₂
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NeZero 2
      a b c : K
      ha : Ne a 0
      s : K
      h : Eq (discrim a b c) (HMul.hMul s s)
      x : K
      ⊢ Iff (Eq s (HAdd.hAdd (Neg.neg b) (Neg.neg (HMul.hMul (HMul.hMul 2 a) x)))) ( …
    -/
                                 /-
                                   🎉 no goals
                                 -/
  · constructor <;> intro h' <;> linear_combination h'
                                 /-
                                   🎉 no goals
                                 -/


/-- A quadratic has roots if its discriminant has square roots -/
theorem exists_quadratic_eq_zero (ha : a ≠ 0) (h : ∃ s, discrim a b c = s * s) :
    ∃ x, a * (x * x) + b * x + c = 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NeZero 2
    a b c : K
    ha : Ne a 0
    h : Exists fun s => Eq (discrim a b c) (HMul.hMul s s)
    ⊢ Exists fun x => Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul …
  -/
  rcases h with ⟨s, hs⟩
  /-
    case intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NeZero 2
    a b c : K
    ha : Ne a 0
    s : K
    hs : Eq (discrim a b c) (HMul.hMul s s)
    ⊢ Exists fun x => Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul …
  -/
  use (-b + s) / (2 * a)
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NeZero 2
    a b c : K
    ha : Ne a 0
    s : K
    hs : Eq (discrim a b c) (HMul.hMul s s)
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul (HDiv.hDiv (HAdd.hAdd (Neg. …
  -/
  rw [quadratic_eq_zero_iff ha hs]
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NeZero 2
    a b c : K
    ha : Ne a 0
    s : K
    hs : Eq (discrim a b c) (HMul.hMul s s)
    ⊢ Or (Eq (HDiv.hDiv (HAdd.hAdd (Neg.neg b) s) (HMul.hMul 2 a)) (HDiv.hDiv (HAd …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Root of a quadratic when its discriminant equals zero -/
theorem quadratic_eq_zero_iff_of_discrim_eq_zero (ha : a ≠ 0) (h : discrim a b c = 0) (x : K) :
    a * (x * x) + b * x + c = 0 ↔ x = -b / (2 * a) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NeZero 2
    a b c : K
    ha : Ne a 0
    h : Eq (discrim a b c) 0
    x : K
    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul.hMul b x)) …
  -/
  have : discrim a b c = 0 * 0 := by rw [h, mul_zero]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NeZero 2
    a b c : K
    ha : Ne a 0
    h : Eq (discrim a b c) 0
    x : K
    this : Eq (discrim a b c) (HMul.hMul 0 0)
    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul.hMul b x)) …
  -/
  rw [quadratic_eq_zero_iff ha this, add_zero, sub_zero, or_self_iff]
  /-
    🎉 no goals
  -/


/-- If a polynomial of degree 2 is always nonnegative, then its discriminant is nonpositive -/
theorem discrim_le_zero (h : ∀ x : K, 0 ≤ a * (x * x) + b * x + c) : discrim a b c ≤ 0 := by
  /-
    K : Type u_1
    inst✝ : LinearOrderedField K
    a b c : K
    h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
    ⊢ LE.le (discrim a b c) 0
  -/
  rw [discrim, sq]
  /-
    K : Type u_1
    inst✝ : LinearOrderedField K
    a b c : K
    h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
    ⊢ LE.le (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 a) c)) 0
  -/
  obtain ha | rfl | ha : a < 0 ∨ a = 0 ∨ 0 < a := lt_trichotomy a 0
  -- if a < 0
  · have : Tendsto (fun x => (a * x + b) * x + c) atTop atBot :=
      tendsto_atBot_add_const_right _ c
        ((tendsto_atBot_add_const_right _ b (tendsto_id.const_mul_atTop_of_neg ha)).atBot_mul_atTop
          tendsto_id)
    /-
      case inl
      K : Type u_1
      inst✝ : LinearOrderedField K
      a b c : K
      h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
      ha : LT.lt a 0
      this : Filter.Tendsto (fun x => HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul a x …
      ⊢ LE.le (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 a) c)) 0
    -/
    rcases (this.eventually (eventually_lt_atBot 0)).exists with ⟨x, hx⟩
    /-
      case inl.intro
      K : Type u_1
      inst✝ : LinearOrderedField K
      a b c : K
      h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
      ha : LT.lt a 0
      this : Filter.Tendsto (fun x => HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul a x …
      x : K
      hx : LT.lt (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul a x) b) x) c) 0
      ⊢ LE.le (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 a) c)) 0
    -/
    exact False.elim ((h x).not_lt <| by rwa [← mul_assoc, ← add_mul])
    /-
      🎉 no goals
    -/
  -- if a = 0
    /-
      case inr.inl
      K : Type u_1
      inst✝ : LinearOrderedField K
      b c : K
      h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 0 (HMul.hMul x x)) (HM …
      ⊢ LE.le (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 0) c)) 0
    -/
  · rcases eq_or_ne b 0 with (rfl | hb)
      /-
        case inr.inl.inl
        K : Type u_1
        inst✝ : LinearOrderedField K
        c : K
        h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 0 (HMul.hMul x x)) (HM …
        ⊢ LE.le (HSub.hSub (HMul.hMul 0 0) (HMul.hMul (HMul.hMul 4 0) c)) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.inr
        K : Type u_1
        inst✝ : LinearOrderedField K
        b c : K
        h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 0 (HMul.hMul x x)) (HM …
        hb : Ne b 0
        ⊢ LE.le (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 0) c)) 0
      -/
    · have := h ((-c - 1) / b)
      /-
        case inr.inl.inr
        K : Type u_1
        inst✝ : LinearOrderedField K
        b c : K
        h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 0 (HMul.hMul x x)) (HM …
        hb : Ne b 0
        this : LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 0 (HMul.hMul (HDiv.hDiv (HSub. …
        ⊢ LE.le (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 0) c)) 0
      -/
      rw [mul_div_cancel₀ _ hb] at this
      /-
        case inr.inl.inr
        K : Type u_1
        inst✝ : LinearOrderedField K
        b c : K
        h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 0 (HMul.hMul x x)) (HM …
        hb : Ne b 0
        this : LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 0 (HMul.hMul (HDiv.hDiv (HSub. …
        ⊢ LE.le (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 0) c)) 0
      -/
      linarith
      /-
        🎉 no goals
      -/
  -- if a > 0
    /-
      case inr.inr
      K : Type u_1
      inst✝ : LinearOrderedField K
      a b c : K
      h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
      ha : LT.lt 0 a
      ⊢ LE.le (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 a) c)) 0
    -/
  · have ha' : 0 ≤ 4 * a := mul_nonneg zero_le_four ha.le
    /-
      case inr.inr
      K : Type u_1
      inst✝ : LinearOrderedField K
      a b c : K
      h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
      ha : LT.lt 0 a
      ha' : LE.le 0 (HMul.hMul 4 a)
      ⊢ LE.le (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 a) c)) 0
    -/
    convert neg_nonpos.2 (mul_nonneg ha' (h (-b / (2 * a)))) using 1
    /-
      case h.e'_3
      K : Type u_1
      inst✝ : LinearOrderedField K
      a b c : K
      h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
      ha : LT.lt 0 a
      ha' : LE.le 0 (HMul.hMul 4 a)
      ⊢ Eq (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 a) c)) (Neg.neg (HMul. …
    -/
    field_simp
    /-
      case h.e'_3
      K : Type u_1
      inst✝ : LinearOrderedField K
      a b c : K
      h : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
      ha : LT.lt 0 a
      ha' : LE.le 0 (HMul.hMul 4 a)
      ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul b b) (HMul.hMul (HMul.hMul 4 a) c)) (HMu …
    -/
    ring
    /-
      🎉 no goals
    -/


lemma discrim_le_zero_of_nonpos (h : ∀ x : K, a * (x * x) + b * x + c ≤ 0) : discrim a b c ≤ 0 :=
                                            /-
                                              K : Type u_1
                                              inst✝ : LinearOrderedField K
                                              a b c : K
                                              h : ∀ (x : K), LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul …
                                              ⊢ ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Neg.neg a) (HMul.hMul x …
                                            -/
  discrim_neg a b c ▸ discrim_le_zero <| by simpa only [neg_mul, ← neg_add, neg_nonneg]
                                            /-
                                              🎉 no goals
                                            -/


/-- If a polynomial of degree 2 is always positive, then its discriminant is negative,
at least when the coefficient of the quadratic term is nonzero.
-/
theorem discrim_lt_zero (ha : a ≠ 0) (h : ∀ x : K, 0 < a * (x * x) + b * x + c) :
    discrim a b c < 0 := by
  /-
    K : Type u_1
    inst✝ : LinearOrderedField K
    a b c : K
    ha : Ne a 0
    h : ∀ (x : K), LT.lt 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
    ⊢ LT.lt (discrim a b c) 0
  -/
  have : ∀ x : K, 0 ≤ a * (x * x) + b * x + c := fun x => le_of_lt (h x)
  /-
    K : Type u_1
    inst✝ : LinearOrderedField K
    a b c : K
    ha : Ne a 0
    h : ∀ (x : K), LT.lt 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
    this : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x))  …
    ⊢ LT.lt (discrim a b c) 0
  -/
  refine lt_of_le_of_ne (discrim_le_zero this) fun h' ↦ ?_
  /-
    K : Type u_1
    inst✝ : LinearOrderedField K
    a b c : K
    ha : Ne a 0
    h : ∀ (x : K), LT.lt 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
    this : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x))  …
    h' : Eq (discrim a b c) 0
    ⊢ False
  -/
  have := h (-b / (2 * a))
  have : a * (-b / (2 * a)) * (-b / (2 * a)) + b * (-b / (2 * a)) + c = 0 := by
    rw [mul_assoc, quadratic_eq_zero_iff_of_discrim_eq_zero ha h' (-b / (2 * a))]
  /-
    K : Type u_1
    inst✝ : LinearOrderedField K
    a b c : K
    ha : Ne a 0
    h : ∀ (x : K), LT.lt 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HM …
    this✝¹ : ∀ (x : K), LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x) …
    h' : Eq (discrim a b c) 0
    this✝ : LT.lt 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul (HDiv.hDiv (Neg. …
    this : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul a (HDiv.hDiv (Neg.neg b) …
    ⊢ False
  -/
  linarith
  /-
    🎉 no goals
  -/


lemma discrim_lt_zero_of_neg (ha : a ≠ 0) (h : ∀ x : K, a * (x * x) + b * x + c < 0) :
    discrim a b c < 0 :=
  discrim_neg a b c ▸ discrim_lt_zero (neg_ne_zero.2 ha) <| by
    /-
      K : Type u_1
      inst✝ : LinearOrderedField K
      a b c : K
      ha : Ne a 0
      h : ∀ (x : K), LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul x x)) (HMul …
      ⊢ ∀ (x : K), LT.lt 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Neg.neg a) (HMul.hMul x …
    -/
    simpa only [neg_mul, ← neg_add, neg_pos]
    /-
      🎉 no goals
    -/


