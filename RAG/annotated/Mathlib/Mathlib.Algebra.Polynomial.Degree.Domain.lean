instance : NoZeroDivisors R[X] where
  eq_zero_or_eq_zero_of_mul_eq_zero h := by
    /-
      R : Type u
      S : Type v
      a b c d : R
      n m : Nat
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q a✝ b✝ : Polynomial R
      h : Eq (HMul.hMul a✝ b✝) 0
      ⊢ Or (Eq a✝ 0) (Eq b✝ 0)
    -/
    rw [← leadingCoeff_eq_zero, ← leadingCoeff_eq_zero]
    /-
      R : Type u
      S : Type v
      a b c d : R
      n m : Nat
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q a✝ b✝ : Polynomial R
      h : Eq (HMul.hMul a✝ b✝) 0
      ⊢ Or (Eq a✝.leadingCoeff 0) (Eq b✝.leadingCoeff 0)
    -/
    refine eq_zero_or_eq_zero_of_mul_eq_zero ?_
    /-
      R : Type u
      S : Type v
      a b c d : R
      n m : Nat
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q a✝ b✝ : Polynomial R
      h : Eq (HMul.hMul a✝ b✝) 0
      ⊢ Eq (HMul.hMul a✝.leadingCoeff b✝.leadingCoeff) 0
    -/
    rw [← leadingCoeff_zero, ← leadingCoeff_mul, h]
    /-
      🎉 no goals
    -/


lemma natDegree_mul (hp : p ≠ 0) (hq : q ≠ 0) : (p*q).natDegree = p.natDegree + q.natDegree := by
  rw [← Nat.cast_inj (R := WithBot ℕ), ← degree_eq_natDegree (mul_ne_zero hp hq),
    Nat.cast_add, ← degree_eq_natDegree hp, ← degree_eq_natDegree hq, degree_mul]


variable (p) in
lemma natDegree_smul (ha : a ≠ 0) : (a • p).natDegree = p.natDegree := by
  /-
    R : Type u
    a : R
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    ha : Ne a 0
    ⊢ Eq (HSMul.hSMul a p).natDegree p.natDegree
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      a : R
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      ha : Ne a 0
      hp : Eq p 0
      ⊢ Eq (HSMul.hSMul a p).natDegree p.natDegree
    -/
  · simp only [hp, smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      a : R
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      ha : Ne a 0
      hp : Not (Eq p 0)
      ⊢ Eq (HSMul.hSMul a p).natDegree p.natDegree
    -/
  · apply natDegree_eq_of_le_of_coeff_ne_zero
      /-
        case neg.pn
        R : Type u
        a : R
        inst✝¹ : Semiring R
        inst✝ : NoZeroDivisors R
        p : Polynomial R
        ha : Ne a 0
        hp : Not (Eq p 0)
        ⊢ LE.le (HSMul.hSMul a p).natDegree p.natDegree
      -/
    · exact (natDegree_smul_le _ _).trans (le_refl _)
      /-
        🎉 no goals
      -/
    · simpa only [coeff_smul, coeff_natDegree, smul_eq_mul, ne_eq, mul_eq_zero,
        leadingCoeff_eq_zero, not_or] using ⟨ha, hp⟩


@[simp]
lemma natDegree_pow (p : R[X]) (n : ℕ) : natDegree (p ^ n) = n * natDegree p := by
  classical
  obtain rfl | hp := eq_or_ne p 0
  · obtain rfl | hn := eq_or_ne n 0 <;> simp [*]
  exact natDegree_pow' <| by
    rw [← leadingCoeff_pow, Ne, leadingCoeff_eq_zero]; exact pow_ne_zero _ hp


lemma natDegree_le_of_dvd (h1 : p ∣ q) (h2 : q ≠ 0) : p.natDegree ≤ q.natDegree := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h1 : Dvd.dvd p q
    h2 : Ne q 0
    ⊢ LE.le p.natDegree q.natDegree
  -/
  obtain ⟨q, rfl⟩ := h1
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h2 : Ne (HMul.hMul p q) 0
    ⊢ LE.le p.natDegree (HMul.hMul p q).natDegree
  -/
  rw [mul_ne_zero_iff] at h2
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h2 : And (Ne p 0) (Ne q 0)
    ⊢ LE.le p.natDegree (HMul.hMul p q).natDegree
  -/
  rw [natDegree_mul h2.1 h2.2]; exact Nat.le_add_right _ _
                                /-
                                  🎉 no goals
                                -/


lemma degree_le_of_dvd (h1 : p ∣ q) (h2 : q ≠ 0) : degree p ≤ degree q := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h1 : Dvd.dvd p q
    h2 : Ne q 0
    ⊢ LE.le p.degree q.degree
  -/
  rcases h1 with ⟨q, rfl⟩; rw [mul_ne_zero_iff] at h2
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h2 : And (Ne p 0) (Ne q 0)
    ⊢ LE.le p.degree (HMul.hMul p q).degree
  -/
  exact degree_le_mul_left p h2.2
  /-
    🎉 no goals
  -/


lemma eq_zero_of_dvd_of_degree_lt (h₁ : p ∣ q) (h₂ : degree q < degree p) : q = 0 := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h₁ : Dvd.dvd p q
    h₂ : LT.lt q.degree p.degree
    ⊢ Eq q 0
  -/
  by_contra hc
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h₁ : Dvd.dvd p q
    h₂ : LT.lt q.degree p.degree
    hc : Not (Eq q 0)
    ⊢ False
  -/
  exact (lt_iff_not_ge _ _).mp h₂ (degree_le_of_dvd h₁ hc)
  /-
    🎉 no goals
  -/


lemma eq_zero_of_dvd_of_natDegree_lt (h₁ : p ∣ q) (h₂ : natDegree q < natDegree p) :
    q = 0 := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h₁ : Dvd.dvd p q
    h₂ : LT.lt q.natDegree p.natDegree
    ⊢ Eq q 0
  -/
  by_contra hc
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h₁ : Dvd.dvd p q
    h₂ : LT.lt q.natDegree p.natDegree
    hc : Not (Eq q 0)
    ⊢ False
  -/
  exact (lt_iff_not_ge _ _).mp h₂ (natDegree_le_of_dvd h₁ hc)
  /-
    🎉 no goals
  -/


lemma not_dvd_of_degree_lt (h0 : q ≠ 0) (hl : q.degree < p.degree) : ¬p ∣ q := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h0 : Ne q 0
    hl : LT.lt q.degree p.degree
    ⊢ Not (Dvd.dvd p q)
  -/
  by_contra hcontra
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h0 : Ne q 0
    hl : LT.lt q.degree p.degree
    hcontra : Dvd.dvd p q
    ⊢ False
  -/
  exact h0 (eq_zero_of_dvd_of_degree_lt hcontra hl)
  /-
    🎉 no goals
  -/


lemma not_dvd_of_natDegree_lt (h0 : q ≠ 0) (hl : q.natDegree < p.natDegree) :
    ¬p ∣ q := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h0 : Ne q 0
    hl : LT.lt q.natDegree p.natDegree
    ⊢ Not (Dvd.dvd p q)
  -/
  by_contra hcontra
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h0 : Ne q 0
    hl : LT.lt q.natDegree p.natDegree
    hcontra : Dvd.dvd p q
    ⊢ False
  -/
  exact h0 (eq_zero_of_dvd_of_natDegree_lt hcontra hl)
  /-
    🎉 no goals
  -/


/-- This lemma is useful for working with the `intDegree` of a rational function. -/
lemma natDegree_sub_eq_of_prod_eq {p₁ p₂ q₁ q₂ : R[X]} (hp₁ : p₁ ≠ 0) (hq₁ : q₁ ≠ 0)
    (hp₂ : p₂ ≠ 0) (hq₂ : q₂ ≠ 0) (h_eq : p₁ * q₂ = p₂ * q₁) :
    (p₁.natDegree : ℤ) - q₁.natDegree = (p₂.natDegree : ℤ) - q₂.natDegree := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p₁ p₂ q₁ q₂ : Polynomial R
    hp₁ : Ne p₁ 0
    hq₁ : Ne q₁ 0
    hp₂ : Ne p₂ 0
    hq₂ : Ne q₂ 0
    h_eq : Eq (HMul.hMul p₁ q₂) (HMul.hMul p₂ q₁)
    ⊢ Eq (HSub.hSub ↑p₁.natDegree ↑q₁.natDegree) (HSub.hSub ↑p₂.natDegree ↑q₂.natD …
  -/
  rw [sub_eq_sub_iff_add_eq_add]
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p₁ p₂ q₁ q₂ : Polynomial R
    hp₁ : Ne p₁ 0
    hq₁ : Ne q₁ 0
    hp₂ : Ne p₂ 0
    hq₂ : Ne q₂ 0
    h_eq : Eq (HMul.hMul p₁ q₂) (HMul.hMul p₂ q₁)
    ⊢ Eq (HAdd.hAdd ↑p₁.natDegree ↑q₂.natDegree) (HAdd.hAdd ↑p₂.natDegree ↑q₁.natD …
  -/
  norm_cast
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p₁ p₂ q₁ q₂ : Polynomial R
    hp₁ : Ne p₁ 0
    hq₁ : Ne q₁ 0
    hp₂ : Ne p₂ 0
    hq₂ : Ne q₂ 0
    h_eq : Eq (HMul.hMul p₁ q₂) (HMul.hMul p₂ q₁)
    ⊢ Eq (HAdd.hAdd p₁.natDegree q₂.natDegree) (HAdd.hAdd p₂.natDegree q₁.natDegree)
  -/
  rw [← natDegree_mul hp₁ hq₂, ← natDegree_mul hp₂ hq₁, h_eq]
  /-
    🎉 no goals
  -/


instance : IsDomain R[X] := NoZeroDivisors.to_isDomain _


