/-- The map `ℤ → ℤˣ` which sends `n` to `(-1 : ℤˣ) ^ n`. -/
def negOnePow (n : ℤ) : ℤˣ := (-1 : ℤˣ) ^ n


lemma negOnePow_def (n : ℤ) : n.negOnePow = (-1 : ℤˣ) ^ n := rfl


lemma negOnePow_add (n₁ n₂ : ℤ) :
    (n₁ + n₂).negOnePow =  n₁.negOnePow * n₂.negOnePow :=
  zpow_add _ _ _


@[simp]
lemma negOnePow_zero : negOnePow 0 = 1 := rfl


@[simp]
lemma negOnePow_one : negOnePow 1 = -1 := rfl


lemma negOnePow_succ (n : ℤ) : (n + 1).negOnePow = - n.negOnePow := by
  /-
    n : Int
    ⊢ Eq (HAdd.hAdd n 1).negOnePow (Neg.neg n.negOnePow)
  -/
  rw [negOnePow_add, negOnePow_one, mul_neg, mul_one]
  /-
    🎉 no goals
  -/


lemma negOnePow_even (n : ℤ) (hn : Even n) : n.negOnePow = 1 := by
  /-
    n : Int
    hn : Even n
    ⊢ Eq n.negOnePow 1
  -/
  obtain ⟨k, rfl⟩ := hn
  /-
    case intro
    k : Int
    ⊢ Eq (HAdd.hAdd k k).negOnePow 1
  -/
  rw [negOnePow_add, units_mul_self]
  /-
    🎉 no goals
  -/


@[simp]
lemma negOnePow_two_mul (n : ℤ) : (2 * n).negOnePow = 1 :=
  negOnePow_even _ ⟨n, two_mul n⟩


lemma negOnePow_odd (n : ℤ) (hn : Odd n) : n.negOnePow = -1 := by
  /-
    n : Int
    hn : Odd n
    ⊢ Eq n.negOnePow (-1)
  -/
  obtain ⟨k, rfl⟩ := hn
  /-
    case intro
    k : Int
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 k) 1).negOnePow (-1)
  -/
  simp only [negOnePow_add, negOnePow_two_mul, negOnePow_one, mul_neg, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
lemma negOnePow_two_mul_add_one (n : ℤ) : (2 * n + 1).negOnePow = -1 :=
  negOnePow_odd _ ⟨n, rfl⟩


lemma negOnePow_eq_one_iff (n : ℤ) : n.negOnePow = 1 ↔ Even n := by
  /-
    n : Int
    ⊢ Iff (Eq n.negOnePow 1) (Even n)
  -/
  constructor
    /-
      case mp
      n : Int
      ⊢ Eq n.negOnePow 1 → Even n
    -/
  · intro h
    /-
      case mp
      n : Int
      h : Eq n.negOnePow 1
      ⊢ Even n
    -/
    rw [← Int.not_odd_iff_even]
    /-
      case mp
      n : Int
      h : Eq n.negOnePow 1
      ⊢ Not (Odd n)
    -/
    intro h'
    /-
      case mp
      n : Int
      h : Eq n.negOnePow 1
      h' : Odd n
      ⊢ False
    -/
    simp only [negOnePow_odd _ h'] at h
    /-
      case mp
      n : Int
      h' : Odd n
      h : Eq (-1) 1
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Int
      ⊢ Even n → Eq n.negOnePow 1
    -/
  · exact negOnePow_even n
    /-
      🎉 no goals
    -/


lemma negOnePow_eq_neg_one_iff (n : ℤ) : n.negOnePow = -1 ↔ Odd n := by
  /-
    n : Int
    ⊢ Iff (Eq n.negOnePow (-1)) (Odd n)
  -/
  constructor
    /-
      case mp
      n : Int
      ⊢ Eq n.negOnePow (-1) → Odd n
    -/
  · intro h
    /-
      case mp
      n : Int
      h : Eq n.negOnePow (-1)
      ⊢ Odd n
    -/
    rw [← Int.not_even_iff_odd]
    /-
      case mp
      n : Int
      h : Eq n.negOnePow (-1)
      ⊢ Not (Even n)
    -/
    intro h'
    /-
      case mp
      n : Int
      h : Eq n.negOnePow (-1)
      h' : Even n
      ⊢ False
    -/
    rw [negOnePow_even _ h'] at h
    /-
      case mp
      n : Int
      h : Eq 1 (-1)
      h' : Even n
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Int
      ⊢ Odd n → Eq n.negOnePow (-1)
    -/
  · exact negOnePow_odd n
    /-
      🎉 no goals
    -/


@[simp]
theorem abs_negOnePow (n : ℤ) : |(n.negOnePow : ℤ)| = 1 := by
  /-
    n : Int
    ⊢ Eq (abs ↑n.negOnePow) 1
  -/
  rw [abs_eq_natAbs, Int.units_natAbs, Nat.cast_one]
  /-
    🎉 no goals
  -/


@[simp]
lemma negOnePow_neg (n : ℤ) : (-n).negOnePow = n.negOnePow := by
  /-
    n : Int
    ⊢ Eq (Neg.neg n).negOnePow n.negOnePow
  -/
  dsimp [negOnePow]
  /-
    n : Int
    ⊢ Eq (HPow.hPow (-1) (Neg.neg n)) (HPow.hPow (-1) n)
  -/
  simp only [zpow_neg, ← inv_zpow, inv_neg, inv_one]
  /-
    🎉 no goals
  -/


@[simp]
lemma negOnePow_abs (n : ℤ) : |n|.negOnePow = n.negOnePow := by
  /-
    n : Int
    ⊢ Eq (abs n).negOnePow n.negOnePow
  -/
                                 /-
                                   🎉 no goals
                                 -/
  obtain h|h := abs_choice n <;> simp only [h, negOnePow_neg]
                                 /-
                                   🎉 no goals
                                 -/


lemma negOnePow_sub (n₁ n₂ : ℤ) :
    (n₁ - n₂).negOnePow = n₁.negOnePow * n₂.negOnePow := by
  /-
    n₁ n₂ : Int
    ⊢ Eq (HSub.hSub n₁ n₂).negOnePow (HMul.hMul n₁.negOnePow n₂.negOnePow)
  -/
  simp only [sub_eq_add_neg, negOnePow_add, negOnePow_neg]
  /-
    🎉 no goals
  -/


lemma negOnePow_eq_iff (n₁ n₂ : ℤ) :
    n₁.negOnePow = n₂.negOnePow ↔ Even (n₁ - n₂) := by
  /-
    n₁ n₂ : Int
    ⊢ Iff (Eq n₁.negOnePow n₂.negOnePow) (Even (HSub.hSub n₁ n₂))
  -/
  by_cases h₂ : Even n₂
    /-
      case pos
      n₁ n₂ : Int
      h₂ : Even n₂
      ⊢ Iff (Eq n₁.negOnePow n₂.negOnePow) (Even (HSub.hSub n₁ n₂))
    -/
  · rw [negOnePow_even _ h₂, Int.even_sub, negOnePow_eq_one_iff]
    /-
      case pos
      n₁ n₂ : Int
      h₂ : Even n₂
      ⊢ Iff (Even n₁) (Iff (Even n₁) (Even n₂))
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case neg
      n₁ n₂ : Int
      h₂ : Not (Even n₂)
      ⊢ Iff (Eq n₁.negOnePow n₂.negOnePow) (Even (HSub.hSub n₁ n₂))
    -/
  · rw [Int.not_even_iff_odd] at h₂
    rw [negOnePow_odd _ h₂, Int.even_sub, negOnePow_eq_neg_one_iff,
      ← Int.not_odd_iff_even, ← Int.not_odd_iff_even]
    /-
      case neg
      n₁ n₂ : Int
      h₂ : Odd n₂
      ⊢ Iff (Odd n₁) (Iff (Not (Odd n₁)) (Not (Odd n₂)))
    -/
    tauto
    /-
      🎉 no goals
    -/


@[simp]
lemma negOnePow_mul_self (n : ℤ) : (n * n).negOnePow = n.negOnePow := by
  /-
    n : Int
    ⊢ Eq (HMul.hMul n n).negOnePow n.negOnePow
  -/
  simpa [mul_sub, negOnePow_eq_iff] using n.even_mul_pred_self
  /-
    🎉 no goals
  -/


lemma cast_negOnePow (K : Type*) (n : ℤ) [Field K] : n.negOnePow = (-1 : K) ^ n := by
  /-
    K : Type u_1
    n : Int
    inst✝ : Field K
    ⊢ Eq (↑↑n.negOnePow) (HPow.hPow (-1) n)
  -/
  rcases even_or_odd' n with ⟨k, rfl | rfl⟩
    /-
      case intro.inl
      K : Type u_1
      inst✝ : Field K
      k : Int
      ⊢ Eq (↑↑(HMul.hMul 2 k).negOnePow) (HPow.hPow (-1) (HMul.hMul 2 k))
    -/
  · simp [zpow_mul, zpow_ofNat]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      K : Type u_1
      inst✝ : Field K
      k : Int
      ⊢ Eq (↑↑(HAdd.hAdd (HMul.hMul 2 k) 1).negOnePow) (HPow.hPow (-1) (HAdd.hAdd (H …
    -/
  · rw [zpow_add_one₀ (by norm_num), zpow_mul, zpow_ofNat]
    /-
      case intro.inr
      K : Type u_1
      inst✝ : Field K
      k : Int
      ⊢ Eq (↑↑(HAdd.hAdd (HMul.hMul 2 k) 1).negOnePow) (HMul.hMul (HPow.hPow (HPow.h …
    -/
    simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-20")] alias coe_negOnePow := cast_negOnePow


lemma cast_negOnePow_natCast (R : Type*) [Ring R] (n : ℕ) : negOnePow n = (-1 : R) ^ n := by
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    ⊢ Eq (↑↑(↑n).negOnePow) (HPow.hPow (-1) n)
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  obtain ⟨k, rfl | rfl⟩ := Nat.even_or_odd' n <;> simp [pow_succ, pow_mul]
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma coe_negOnePow_natCast (n : ℕ) : negOnePow n = (-1 : ℤ) ^ n := cast_negOnePow_natCast ..


