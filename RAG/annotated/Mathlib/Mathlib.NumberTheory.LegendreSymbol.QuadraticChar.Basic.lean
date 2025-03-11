/-- Define the quadratic character with values in ℤ on a monoid with zero `α`.
It takes the value zero at zero; for non-zero argument `a : α`, it is `1`
if `a` is a square, otherwise it is `-1`.

This only deserves the name "character" when it is multiplicative,
e.g., when `α` is a finite field. See `quadraticCharFun_mul`.

We will later define `quadraticChar` to be a multiplicative character
of type `MulChar F ℤ`, when the domain is a finite field `F`.
-/
def quadraticCharFun (α : Type*) [MonoidWithZero α] [DecidableEq α]
    [DecidablePred (IsSquare : α → Prop)] (a : α) : ℤ :=
  if a = 0 then 0 else if IsSquare a then 1 else -1


/-- Some basic API lemmas -/
theorem quadraticCharFun_eq_zero_iff {a : F} : quadraticCharFun F a = 0 ↔ a = 0 := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    a : F
    ⊢ Iff (Eq (quadraticCharFun F a) 0) (Eq a 0)
  -/
  simp only [quadraticCharFun]
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    a : F
    ⊢ Iff (Eq (ite (Eq a 0) 0 (ite (IsSquare a) 1 (-1))) 0) (Eq a 0)
  -/
  by_cases ha : a = 0
    /-
      case pos
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      a : F
      ha : Eq a 0
      ⊢ Iff (Eq (ite (Eq a 0) 0 (ite (IsSquare a) 1 (-1))) 0) (Eq a 0)
    -/
  · simp only [ha, if_true]
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      a : F
      ha : Not (Eq a 0)
      ⊢ Iff (Eq (ite (Eq a 0) 0 (ite (IsSquare a) 1 (-1))) 0) (Eq a 0)
    -/
  · simp only [ha, if_false]
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      a : F
      ha : Not (Eq a 0)
      ⊢ Iff (Eq (ite (IsSquare a) 1 (-1)) 0) False
    -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> simp only [neg_eq_zero, one_ne_zero, not_false_iff]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem quadraticCharFun_zero : quadraticCharFun F 0 = 0 := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    ⊢ Eq (quadraticCharFun F 0) 0
  -/
  simp only [quadraticCharFun, if_true]
  /-
    🎉 no goals
  -/


@[simp]
theorem quadraticCharFun_one : quadraticCharFun F 1 = 1 := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    ⊢ Eq (quadraticCharFun F 1) 1
  -/
  simp only [quadraticCharFun, one_ne_zero, IsSquare.one, if_true, if_false]
  /-
    🎉 no goals
  -/


/-- If `ringChar F = 2`, then `quadraticCharFun F` takes the value `1` on nonzero elements. -/
theorem quadraticCharFun_eq_one_of_char_two (hF : ringChar F = 2) {a : F} (ha : a ≠ 0) :
    quadraticCharFun F a = 1 := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Eq (ringChar F) 2
    a : F
    ha : Ne a 0
    ⊢ Eq (quadraticCharFun F a) 1
  -/
  simp only [quadraticCharFun, ha, if_false, ite_eq_left_iff]
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Eq (ringChar F) 2
    a : F
    ha : Ne a 0
    ⊢ Not (IsSquare a) → Eq (-1) 1
  -/
  exact fun h ↦ (h (FiniteField.isSquare_of_char_two hF a)).elim
  /-
    🎉 no goals
  -/


/-- If `ringChar F` is odd, then `quadraticCharFun F a` can be computed in
terms of `a ^ (Fintype.card F / 2)`. -/
theorem quadraticCharFun_eq_pow_of_char_ne_two (hF : ringChar F ≠ 2) {a : F} (ha : a ≠ 0) :
    quadraticCharFun F a = if a ^ (Fintype.card F / 2) = 1 then 1 else -1 := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    a : F
    ha : Ne a 0
    ⊢ Eq (quadraticCharFun F a) (ite (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F)  …
  -/
  simp only [quadraticCharFun, ha, if_false]
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    a : F
    ha : Ne a 0
    ⊢ Eq (ite (IsSquare a) 1 (-1)) (ite (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card  …
  -/
  simp_rw [FiniteField.isSquare_iff hF ha]
  /-
    🎉 no goals
  -/


/-- The quadratic character is multiplicative. -/
theorem quadraticCharFun_mul (a b : F) :
    quadraticCharFun F (a * b) = quadraticCharFun F a * quadraticCharFun F b := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    a b : F
    ⊢ Eq (quadraticCharFun F (HMul.hMul a b)) (HMul.hMul (quadraticCharFun F a) (q …
  -/
  by_cases ha : a = 0
    /-
      case pos
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      a b : F
      ha : Eq a 0
      ⊢ Eq (quadraticCharFun F (HMul.hMul a b)) (HMul.hMul (quadraticCharFun F a) (q …
    -/
  · rw [ha, zero_mul, quadraticCharFun_zero, zero_mul]
    /-
      🎉 no goals
    -/
  -- now `a ≠ 0`
  /-
    case neg
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    a b : F
    ha : Not (Eq a 0)
    ⊢ Eq (quadraticCharFun F (HMul.hMul a b)) (HMul.hMul (quadraticCharFun F a) (q …
  -/
  by_cases hb : b = 0
    /-
      case pos
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      a b : F
      ha : Not (Eq a 0)
      hb : Eq b 0
      ⊢ Eq (quadraticCharFun F (HMul.hMul a b)) (HMul.hMul (quadraticCharFun F a) (q …
    -/
  · rw [hb, mul_zero, quadraticCharFun_zero, mul_zero]
    /-
      🎉 no goals
    -/
  -- now `a ≠ 0` and `b ≠ 0`
  /-
    case neg
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    a b : F
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ Eq (quadraticCharFun F (HMul.hMul a b)) (HMul.hMul (quadraticCharFun F a) (q …
  -/
  have hab := mul_ne_zero ha hb
  /-
    case neg
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    a b : F
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    hab : Ne (HMul.hMul a b) 0
    ⊢ Eq (quadraticCharFun F (HMul.hMul a b)) (HMul.hMul (quadraticCharFun F a) (q …
  -/
  by_cases hF : ringChar F = 2
  ·-- case `ringChar F = 2`
    rw [quadraticCharFun_eq_one_of_char_two hF ha, quadraticCharFun_eq_one_of_char_two hF hb,
      quadraticCharFun_eq_one_of_char_two hF hab, mul_one]
  · -- case of odd characteristic
    rw [quadraticCharFun_eq_pow_of_char_ne_two hF ha, quadraticCharFun_eq_pow_of_char_ne_two hF hb,
      quadraticCharFun_eq_pow_of_char_ne_two hF hab, mul_pow]
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      a b : F
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      hab : Ne (HMul.hMul a b) 0
      hF : Not (Eq (ringChar F) 2)
      ⊢ Eq (ite (Eq (HMul.hMul (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) (HPow.hP …
    -/
    cases' FiniteField.pow_dichotomy hF hb with hb' hb'
      /-
        case neg.inl
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        a b : F
        ha : Not (Eq a 0)
        hb : Not (Eq b 0)
        hab : Ne (HMul.hMul a b) 0
        hF : Not (Eq (ringChar F) 2)
        hb' : Eq (HPow.hPow b (HDiv.hDiv (Fintype.card F) 2)) 1
        ⊢ Eq (ite (Eq (HMul.hMul (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) (HPow.hP …
      -/
    · simp only [hb', mul_one, if_true]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        a b : F
        ha : Not (Eq a 0)
        hb : Not (Eq b 0)
        hab : Ne (HMul.hMul a b) 0
        hF : Not (Eq (ringChar F) 2)
        hb' : Eq (HPow.hPow b (HDiv.hDiv (Fintype.card F) 2)) (-1)
        ⊢ Eq (ite (Eq (HMul.hMul (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) (HPow.hP …
      -/
    · have h := Ring.neg_one_ne_one_of_char_ne_two hF
      -- `-1 ≠ 1`
      /-
        case neg.inr
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        a b : F
        ha : Not (Eq a 0)
        hb : Not (Eq b 0)
        hab : Ne (HMul.hMul a b) 0
        hF : Not (Eq (ringChar F) 2)
        hb' : Eq (HPow.hPow b (HDiv.hDiv (Fintype.card F) 2)) (-1)
        h : Ne (-1) 1
        ⊢ Eq (ite (Eq (HMul.hMul (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) (HPow.hP …
      -/
      simp only [hb', mul_neg, mul_one, h, if_false]
      /-
        case neg.inr
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        a b : F
        ha : Not (Eq a 0)
        hb : Not (Eq b 0)
        hab : Ne (HMul.hMul a b) 0
        hF : Not (Eq (ringChar F) 2)
        hb' : Eq (HPow.hPow b (HDiv.hDiv (Fintype.card F) 2)) (-1)
        h : Ne (-1) 1
        ⊢ Eq (ite (Eq (Neg.neg (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2))) 1) 1 (-1) …
      -/
      cases' FiniteField.pow_dichotomy hF ha with ha' ha' <;>
        /-
          case neg.inr.inl
          F : Type u_1
          inst✝² : Field F
          inst✝¹ : Fintype F
          inst✝ : DecidableEq F
          a b : F
          ha : Not (Eq a 0)
          hb : Not (Eq b 0)
          hab : Ne (HMul.hMul a b) 0
          hF : Not (Eq (ringChar F) 2)
          hb' : Eq (HPow.hPow b (HDiv.hDiv (Fintype.card F) 2)) (-1)
          h : Ne (-1) 1
          ha' : Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1
          ⊢ Eq (ite (Eq (Neg.neg (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2))) 1) 1 (-1) …
        -/
        /-
          🎉 no goals
        -/
        simp only [ha', h, neg_neg, if_true, if_false]
        /-
          🎉 no goals
        -/


/-- The quadratic character as a multiplicative character. -/
@[simps]
def quadraticChar : MulChar F ℤ where
  toFun := quadraticCharFun F
  map_one' := quadraticCharFun_one
  map_mul' := quadraticCharFun_mul
                          /-
                            F : Type u_1
                            inst✝² : Field F
                            inst✝¹ : Fintype F
                            inst✝ : DecidableEq F
                            a : F
                            ha : Not (IsUnit a)
                            ⊢ Eq ((↑{ toFun := quadraticCharFun F, map_one' := ⋯, map_mul' := ⋯ }).toFun a …
                          -/
  map_nonunit' a ha := by rw [of_not_not (mt Ne.isUnit ha)]; exact quadraticCharFun_zero
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- The value of the quadratic character on `a` is zero iff `a = 0`. -/
theorem quadraticChar_eq_zero_iff {a : F} : quadraticChar F a = 0 ↔ a = 0 :=
  quadraticCharFun_eq_zero_iff


theorem quadraticChar_zero : quadraticChar F 0 = 0 := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    ⊢ Eq ((quadraticChar F) 0) 0
  -/
  simp only [quadraticChar_apply, quadraticCharFun_zero]
  /-
    🎉 no goals
  -/


/-- For nonzero `a : F`, `quadraticChar F a = 1 ↔ IsSquare a`. -/
theorem quadraticChar_one_iff_isSquare {a : F} (ha : a ≠ 0) :
    quadraticChar F a = 1 ↔ IsSquare a := by
  simp only [quadraticChar_apply, quadraticCharFun, ha, if_false, ite_eq_left_iff,
    (by omega : (-1 : ℤ) ≠ 1), imp_false, not_not, reduceCtorEq]


/-- The quadratic character takes the value `1` on nonzero squares. -/
theorem quadraticChar_sq_one' {a : F} (ha : a ≠ 0) : quadraticChar F (a ^ 2) = 1 := by
  simp only [quadraticChar_apply, quadraticCharFun, sq_eq_zero_iff, ha, IsSquare.sq, if_true,
    if_false]


/-- The square of the quadratic character on nonzero arguments is `1`. -/
theorem quadraticChar_sq_one {a : F} (ha : a ≠ 0) : quadraticChar F a ^ 2 = 1 := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    a : F
    ha : Ne a 0
    ⊢ Eq (HPow.hPow ((quadraticChar F) a) 2) 1
  -/
  rwa [pow_two, ← map_mul, ← pow_two, quadraticChar_sq_one']
  /-
    🎉 no goals
  -/


/-- The quadratic character is `1` or `-1` on nonzero arguments. -/
theorem quadraticChar_dichotomy {a : F} (ha : a ≠ 0) :
    quadraticChar F a = 1 ∨ quadraticChar F a = -1 :=
  sq_eq_one_iff.1 <| quadraticChar_sq_one ha


/-- The quadratic character is `1` or `-1` on nonzero arguments. -/
theorem quadraticChar_eq_neg_one_iff_not_one {a : F} (ha : a ≠ 0) :
    quadraticChar F a = -1 ↔ ¬quadraticChar F a = 1 :=
              /-
                F : Type u_1
                inst✝² : Field F
                inst✝¹ : Fintype F
                inst✝ : DecidableEq F
                a : F
                ha : Ne a 0
                h : Eq ((quadraticChar F) a) (-1)
                ⊢ Not (Eq ((quadraticChar F) a) 1)
              -/
  ⟨fun h ↦ by rw [h]; omega, fun h₂ ↦ (or_iff_right h₂).mp (quadraticChar_dichotomy ha)⟩
                      /-
                        🎉 no goals
                      -/


/-- For `a : F`, `quadraticChar F a = -1 ↔ ¬ IsSquare a`. -/
theorem quadraticChar_neg_one_iff_not_isSquare {a : F} : quadraticChar F a = -1 ↔ ¬IsSquare a := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    a : F
    ⊢ Iff (Eq ((quadraticChar F) a) (-1)) (Not (IsSquare a))
  -/
  by_cases ha : a = 0
    /-
      case pos
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      a : F
      ha : Eq a 0
      ⊢ Iff (Eq ((quadraticChar F) a) (-1)) (Not (IsSquare a))
    -/
  · simp only [ha, MulChar.map_zero, zero_eq_neg, one_ne_zero, isSquare_zero, not_true]
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      a : F
      ha : Not (Eq a 0)
      ⊢ Iff (Eq ((quadraticChar F) a) (-1)) (Not (IsSquare a))
    -/
  · rw [quadraticChar_eq_neg_one_iff_not_one ha, quadraticChar_one_iff_isSquare ha]
    /-
      🎉 no goals
    -/


/-- If `F` has odd characteristic, then `quadraticChar F` takes the value `-1`. -/
theorem quadraticChar_exists_neg_one (hF : ringChar F ≠ 2) : ∃ a, quadraticChar F a = -1 :=
  (FiniteField.exists_nonsquare hF).imp fun _ h₁ ↦ quadraticChar_neg_one_iff_not_isSquare.mpr h₁


/-- If `F` has odd characteristic, then `quadraticChar F` takes the value `-1` on some unit. -/
lemma quadraticChar_exists_neg_one' (hF : ringChar F ≠ 2) : ∃ a : Fˣ, quadraticChar F a = -1 := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    ⊢ Exists fun a => Eq ((quadraticChar F) ↑a) (-1)
  -/
  refine (fun ⟨a, ha⟩ ↦ ⟨IsUnit.unit ?_, ha⟩) (quadraticChar_exists_neg_one hF)
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    x✝ : Exists fun a => Eq ((quadraticChar F) a) (-1)
    a : F
    ha : Eq ((quadraticChar F) a) (-1)
    ⊢ IsUnit a
  -/
  contrapose ha
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    x✝ : Exists fun a => Eq ((quadraticChar F) a) (-1)
    a : F
    ha : Not (IsUnit a)
    ⊢ Not (Eq ((quadraticChar F) a) (-1))
  -/
  exact ne_of_eq_of_ne ((quadraticChar F).map_nonunit ha) (mt zero_eq_neg.mp one_ne_zero)
  /-
    🎉 no goals
  -/


/-- If `ringChar F = 2`, then `quadraticChar F` takes the value `1` on nonzero elements. -/
theorem quadraticChar_eq_one_of_char_two (hF : ringChar F = 2) {a : F} (ha : a ≠ 0) :
    quadraticChar F a = 1 :=
  quadraticCharFun_eq_one_of_char_two hF ha


/-- If `ringChar F` is odd, then `quadraticChar F a` can be computed in
terms of `a ^ (Fintype.card F / 2)`. -/
theorem quadraticChar_eq_pow_of_char_ne_two (hF : ringChar F ≠ 2) {a : F} (ha : a ≠ 0) :
    quadraticChar F a = if a ^ (Fintype.card F / 2) = 1 then 1 else -1 :=
  quadraticCharFun_eq_pow_of_char_ne_two hF ha


theorem quadraticChar_eq_pow_of_char_ne_two' (hF : ringChar F ≠ 2) (a : F) :
    (quadraticChar F a : F) = a ^ (Fintype.card F / 2) := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    a : F
    ⊢ Eq (↑((quadraticChar F) a)) (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2))
  -/
  by_cases ha : a = 0
    /-
      case pos
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      hF : Ne (ringChar F) 2
      a : F
      ha : Eq a 0
      ⊢ Eq (↑((quadraticChar F) a)) (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2))
    -/
  · have : 0 < Fintype.card F / 2 := Nat.div_pos Fintype.one_lt_card two_pos
    /-
      case pos
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      hF : Ne (ringChar F) 2
      a : F
      ha : Eq a 0
      this : LT.lt 0 (HDiv.hDiv (Fintype.card F) 2)
      ⊢ Eq (↑((quadraticChar F) a)) (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2))
    -/
    simp only [ha, quadraticChar_apply, quadraticCharFun_zero, Int.cast_zero, zero_pow this.ne']
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      hF : Ne (ringChar F) 2
      a : F
      ha : Not (Eq a 0)
      ⊢ Eq (↑((quadraticChar F) a)) (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2))
    -/
  · rw [quadraticChar_eq_pow_of_char_ne_two hF ha]
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      hF : Ne (ringChar F) 2
      a : F
      ha : Not (Eq a 0)
      ⊢ Eq (↑(ite (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1) 1 (-1))) (HPow …
    -/
    by_cases ha' : a ^ (Fintype.card F / 2) = 1
      /-
        case pos
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        ha : Not (Eq a 0)
        ha' : Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1
        ⊢ Eq (↑(ite (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1) 1 (-1))) (HPow …
      -/
    · simp only [ha', if_true, Int.cast_one]
      /-
        🎉 no goals
      -/
      /-
        case neg
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        ha : Not (Eq a 0)
        ha' : Not (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1)
        ⊢ Eq (↑(ite (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1) 1 (-1))) (HPow …
      -/
    · have ha'' := Or.resolve_left (FiniteField.pow_dichotomy hF ha) ha'
      /-
        case neg
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        ha : Not (Eq a 0)
        ha' : Not (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1)
        ha'' : Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) (-1)
        ⊢ Eq (↑(ite (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1) 1 (-1))) (HPow …
      -/
      simp only [ha'', Int.cast_ite, Int.cast_one, Int.cast_neg, ite_eq_right_iff]
      /-
        case neg
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        ha : Not (Eq a 0)
        ha' : Not (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1)
        ha'' : Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) (-1)
        ⊢ Eq (-1) 1 → Eq 1 (-1)
      -/
      exact Eq.symm
      /-
        🎉 no goals
      -/


/-- The quadratic character is quadratic as a multiplicative character. -/
theorem quadraticChar_isQuadratic : (quadraticChar F).IsQuadratic := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    ⊢ (quadraticChar F).IsQuadratic
  -/
  intro a
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    a : F
    ⊢ Or (Eq ((quadraticChar F) a) 0) (Or (Eq ((quadraticChar F) a) 1) (Eq ((quadr …
  -/
  by_cases ha : a = 0
    /-
      case pos
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      a : F
      ha : Eq a 0
      ⊢ Or (Eq ((quadraticChar F) a) 0) (Or (Eq ((quadraticChar F) a) 1) (Eq ((quadr …
    -/
  · left; rw [ha]; exact quadraticChar_zero
                   /-
                     🎉 no goals
                   -/
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      a : F
      ha : Not (Eq a 0)
      ⊢ Or (Eq ((quadraticChar F) a) 0) (Or (Eq ((quadraticChar F) a) 1) (Eq ((quadr …
    -/
  · right; exact quadraticChar_dichotomy ha
           /-
             🎉 no goals
           -/


/-- The quadratic character is nontrivial as a multiplicative character
when the domain has odd characteristic. -/
theorem quadraticChar_ne_one (hF : ringChar F ≠ 2) : quadraticChar F ≠ 1 := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    ⊢ Ne (quadraticChar F) 1
  -/
  rcases quadraticChar_exists_neg_one' hF with ⟨a, ha⟩
  /-
    case intro
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    a : Units F
    ha : Eq ((quadraticChar F) ↑a) (-1)
    ⊢ Ne (quadraticChar F) 1
  -/
  intro hχ
  /-
    case intro
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    a : Units F
    ha : Eq ((quadraticChar F) ↑a) (-1)
    hχ : Eq (quadraticChar F) 1
    ⊢ False
  -/
  simp only [hχ, one_apply a.isUnit, one_ne_zero, reduceCtorEq] at ha
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated quadraticChar_ne_one (since := "2024-06-16")]
theorem quadraticChar_isNontrivial (hF : ringChar F ≠ 2) : (quadraticChar F).IsNontrivial :=
  (isNontrivial_iff _).mpr <| quadraticChar_ne_one hF


open Finset in
/-- The number of solutions to `x^2 = a` is determined by the quadratic character. -/
theorem quadraticChar_card_sqrts (hF : ringChar F ≠ 2) (a : F) :
    #{x : F | x ^ 2 = a}.toFinset = quadraticChar F a + 1 := by
  -- we consider the cases `a = 0`, `a` is a nonzero square and `a` is a nonsquare in turn
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    a : F
    ⊢ Eq (↑(setOf fun x => Eq (HPow.hPow x 2) a).toFinset.card) (HAdd.hAdd ((quadr …
  -/
  by_cases h₀ : a = 0
  · simp only [h₀, sq_eq_zero_iff, Set.setOf_eq_eq_singleton, Set.toFinset_card,
    Set.card_singleton, Int.ofNat_succ, Int.ofNat_zero, MulChar.map_zero]
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      hF : Ne (ringChar F) 2
      a : F
      h₀ : Not (Eq a 0)
      ⊢ Eq (↑(setOf fun x => Eq (HPow.hPow x 2) a).toFinset.card) (HAdd.hAdd ((quadr …
    -/
  · set s := {x : F | x ^ 2 = a}.toFinset
    /-
      case neg
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      hF : Ne (ringChar F) 2
      a : F
      h₀ : Not (Eq a 0)
      s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
      ⊢ Eq (↑s.card) (HAdd.hAdd ((quadraticChar F) a) 1)
    -/
    by_cases h : IsSquare a
      /-
        case pos
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        h₀ : Not (Eq a 0)
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        h : IsSquare a
        ⊢ Eq (↑s.card) (HAdd.hAdd ((quadraticChar F) a) 1)
      -/
    · rw [(quadraticChar_one_iff_isSquare h₀).mpr h]
      /-
        case pos
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        h₀ : Not (Eq a 0)
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        h : IsSquare a
        ⊢ Eq (↑s.card) (HAdd.hAdd 1 1)
      -/
      rcases h with ⟨b, h⟩
      /-
        case pos.intro
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        h₀ : Not (Eq a 0)
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        b : F
        h : Eq a (HMul.hMul b b)
        ⊢ Eq (↑s.card) (HAdd.hAdd 1 1)
      -/
      rw [h, mul_self_eq_zero] at h₀
      have h₁ : s = [b, -b].toFinset := by
        ext1
        rw [← pow_two] at h
        simp only [Set.toFinset_setOf, h, mem_filter, mem_univ, true_and, List.toFinset_cons,
          List.toFinset_nil, insert_emptyc_eq, mem_insert, mem_singleton, s]
        exact sq_eq_sq_iff_eq_or_eq_neg
      /-
        case pos.intro
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        b : F
        h₀ : Not (Eq b 0)
        h : Eq a (HMul.hMul b b)
        h₁ : Eq s (List.cons b (List.cons (Neg.neg b) List.nil)).toFinset
        ⊢ Eq (↑s.card) (HAdd.hAdd 1 1)
      -/
      norm_cast
      /-
        case pos.intro
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        b : F
        h₀ : Not (Eq b 0)
        h : Eq a (HMul.hMul b b)
        h₁ : Eq s (List.cons b (List.cons (Neg.neg b) List.nil)).toFinset
        ⊢ Eq s.card (HAdd.hAdd 1 1)
      -/
      rw [h₁, List.toFinset_cons, List.toFinset_cons, List.toFinset_nil]
      /-
        case pos.intro
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        b : F
        h₀ : Not (Eq b 0)
        h : Eq a (HMul.hMul b b)
        h₁ : Eq s (List.cons b (List.cons (Neg.neg b) List.nil)).toFinset
        ⊢ Eq (Insert.insert b (Insert.insert (Neg.neg b) EmptyCollection.emptyCollecti …
      -/
      exact card_pair (Ne.symm (mt (Ring.eq_self_iff_eq_zero_of_char_ne_two hF).mp h₀))
      /-
        🎉 no goals
      -/
      /-
        case neg
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        h₀ : Not (Eq a 0)
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        h : Not (IsSquare a)
        ⊢ Eq (↑s.card) (HAdd.hAdd ((quadraticChar F) a) 1)
      -/
    · rw [quadraticChar_neg_one_iff_not_isSquare.mpr h]
      /-
        case neg
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        h₀ : Not (Eq a 0)
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        h : Not (IsSquare a)
        ⊢ Eq (↑s.card) (HAdd.hAdd (-1) 1)
      -/
      simp only [neg_add_cancel, Int.natCast_eq_zero, card_eq_zero]
      /-
        case neg
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        h₀ : Not (Eq a 0)
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        h : Not (IsSquare a)
        ⊢ Eq s EmptyCollection.emptyCollection
      -/
      ext1
      -- Porting note(https://github.com/leanprover-community/mathlib4/issues/5026):
      -- added (Set.mem_toFinset), Set.mem_setOf
      /-
        case neg.h
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        h₀ : Not (Eq a 0)
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        h : Not (IsSquare a)
        a✝ : F
        ⊢ Iff (Membership.mem s a✝) (Membership.mem EmptyCollection.emptyCollection a✝)
      -/
      simp only [s, (Set.mem_toFinset), Set.mem_setOf, not_mem_empty, iff_false]
      /-
        case neg.h
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        h₀ : Not (Eq a 0)
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        h : Not (IsSquare a)
        a✝ : F
        ⊢ Not (Eq (HPow.hPow a✝ 2) a)
      -/
      rw [isSquare_iff_exists_sq] at h
      /-
        case neg.h
        F : Type u_1
        inst✝² : Field F
        inst✝¹ : Fintype F
        inst✝ : DecidableEq F
        hF : Ne (ringChar F) 2
        a : F
        h₀ : Not (Eq a 0)
        s : Finset F := (setOf fun x => Eq (HPow.hPow x 2) a).toFinset
        h : Not (Exists fun c => Eq a (HPow.hPow c 2))
        a✝ : F
        ⊢ Not (Eq (HPow.hPow a✝ 2) a)
      -/
      exact fun h' ↦ h ⟨_, h'.symm⟩
      /-
        🎉 no goals
      -/


/-- The sum over the values of the quadratic character is zero when the characteristic is odd. -/
theorem quadraticChar_sum_zero (hF : ringChar F ≠ 2) : ∑ a : F, quadraticChar F a = 0 :=
  sum_eq_zero_of_ne_one (quadraticChar_ne_one hF)


/-- The value of the quadratic character at `-1` -/
theorem quadraticChar_neg_one [DecidableEq F] (hF : ringChar F ≠ 2) :
    quadraticChar F (-1) = χ₄ (Fintype.card F) := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    ⊢ Eq ((quadraticChar F) (-1)) (ZMod.χ₄ ↑(Fintype.card F))
  -/
  have h := quadraticChar_eq_pow_of_char_ne_two hF (neg_ne_zero.mpr one_ne_zero)
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    h : Eq ((quadraticChar F) (-1)) (ite (Eq (HPow.hPow (-1) (HDiv.hDiv (Fintype.c …
    ⊢ Eq ((quadraticChar F) (-1)) (ZMod.χ₄ ↑(Fintype.card F))
  -/
  rw [h, χ₄_eq_neg_one_pow (FiniteField.odd_card_of_char_ne_two hF)]
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    h : Eq ((quadraticChar F) (-1)) (ite (Eq (HPow.hPow (-1) (HDiv.hDiv (Fintype.c …
    ⊢ Eq (ite (Eq (HPow.hPow (-1) (HDiv.hDiv (Fintype.card F) 2)) 1) 1 (-1)) (HPow …
  -/
  generalize Fintype.card F / 2 = n
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Fintype F
    inst✝ : DecidableEq F
    hF : Ne (ringChar F) 2
    h : Eq ((quadraticChar F) (-1)) (ite (Eq (HPow.hPow (-1) (HDiv.hDiv (Fintype.c …
    n : Nat
    ⊢ Eq (ite (Eq (HPow.hPow (-1) n) 1) 1 (-1)) (HPow.hPow (-1) n)
  -/
  cases' Nat.even_or_odd n with h₂ h₂
    /-
      case inl
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      hF : Ne (ringChar F) 2
      h : Eq ((quadraticChar F) (-1)) (ite (Eq (HPow.hPow (-1) (HDiv.hDiv (Fintype.c …
      n : Nat
      h₂ : Even n
      ⊢ Eq (ite (Eq (HPow.hPow (-1) n) 1) 1 (-1)) (HPow.hPow (-1) n)
    -/
  · simp only [Even.neg_one_pow h₂, if_true]
    /-
      🎉 no goals
    -/
    /-
      case inr
      F : Type u_1
      inst✝² : Field F
      inst✝¹ : Fintype F
      inst✝ : DecidableEq F
      hF : Ne (ringChar F) 2
      h : Eq ((quadraticChar F) (-1)) (ite (Eq (HPow.hPow (-1) (HDiv.hDiv (Fintype.c …
      n : Nat
      h₂ : Odd n
      ⊢ Eq (ite (Eq (HPow.hPow (-1) n) 1) 1 (-1)) (HPow.hPow (-1) n)
    -/
  · simp only [Odd.neg_one_pow h₂, Ring.neg_one_ne_one_of_char_ne_two hF, ite_false]
    /-
      🎉 no goals
    -/


/-- `-1` is a square in `F` iff `#F` is not congruent to `3` mod `4`. -/
theorem FiniteField.isSquare_neg_one_iff : IsSquare (-1 : F) ↔ Fintype.card F % 4 ≠ 3 := by
  classical -- suggested by the linter (instead of `[DecidableEq F]`)
  by_cases hF : ringChar F = 2
  · simp only [FiniteField.isSquare_of_char_two hF, Ne, true_iff]
    exact fun hf ↦
      one_ne_zero <|
        (Nat.odd_of_mod_four_eq_three hf).symm.trans <| FiniteField.even_card_of_char_two hF
  · have h₁ := FiniteField.odd_card_of_char_ne_two hF
    rw [← quadraticChar_one_iff_isSquare (neg_ne_zero.mpr (one_ne_zero' F)),
      quadraticChar_neg_one hF, χ₄_nat_eq_if_mod_four, h₁]
    omega


