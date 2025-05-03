/-- `eraseLead f` for a polynomial `f` is the polynomial obtained by
subtracting from `f` the leading term of `f`. -/
def eraseLead (f : R[X]) : R[X] :=
  Polynomial.erase f.natDegree f


theorem eraseLead_support (f : R[X]) : f.eraseLead.support = f.support.erase f.natDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq f.eraseLead.support (f.support.erase f.natDegree)
  -/
  simp only [eraseLead, support_erase]
  /-
    🎉 no goals
  -/


theorem eraseLead_coeff (i : ℕ) :
    f.eraseLead.coeff i = if i = f.natDegree then 0 else f.coeff i := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    i : Nat
    ⊢ Eq (f.eraseLead.coeff i) (ite (Eq i f.natDegree) 0 (f.coeff i))
  -/
  simp only [eraseLead, coeff_erase]
  /-
    🎉 no goals
  -/


@[simp]
                                                                            /-
                                                                              R : Type u_1
                                                                              inst✝ : Semiring R
                                                                              f : Polynomial R
                                                                              ⊢ Eq (f.eraseLead.coeff f.natDegree) 0
                                                                            -/
theorem eraseLead_coeff_natDegree : f.eraseLead.coeff f.natDegree = 0 := by simp [eraseLead_coeff]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem eraseLead_coeff_of_ne (i : ℕ) (hi : i ≠ f.natDegree) : f.eraseLead.coeff i = f.coeff i := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    i : Nat
    hi : Ne i f.natDegree
    ⊢ Eq (f.eraseLead.coeff i) (f.coeff i)
  -/
  simp [eraseLead_coeff, hi]
  /-
    🎉 no goals
  -/


@[simp]
                                                        /-
                                                          R : Type u_1
                                                          inst✝ : Semiring R
                                                          ⊢ Eq (Polynomial.eraseLead 0) 0
                                                        -/
theorem eraseLead_zero : eraseLead (0 : R[X]) = 0 := by simp only [eraseLead, erase_zero]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem eraseLead_add_monomial_natDegree_leadingCoeff (f : R[X]) :
    f.eraseLead + monomial f.natDegree f.leadingCoeff = f :=
  (add_comm _ _).trans (f.monomial_add_erase _)


@[simp]
theorem eraseLead_add_C_mul_X_pow (f : R[X]) :
    f.eraseLead + C f.leadingCoeff * X ^ f.natDegree = f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq (HAdd.hAdd f.eraseLead (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPo …
  -/
  rw [C_mul_X_pow_eq_monomial, eraseLead_add_monomial_natDegree_leadingCoeff]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_sub_monomial_natDegree_leadingCoeff {R : Type*} [Ring R] (f : R[X]) :
    f - monomial f.natDegree f.leadingCoeff = f.eraseLead :=
  (eq_sub_iff_add_eq.mpr (eraseLead_add_monomial_natDegree_leadingCoeff f)).symm


@[simp]
theorem self_sub_C_mul_X_pow {R : Type*} [Ring R] (f : R[X]) :
    f - C f.leadingCoeff * X ^ f.natDegree = f.eraseLead := by
  /-
    R : Type u_2
    inst✝ : Ring R
    f : Polynomial R
    ⊢ Eq (HSub.hSub f (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow Polynomi …
  -/
  rw [C_mul_X_pow_eq_monomial, self_sub_monomial_natDegree_leadingCoeff]
  /-
    🎉 no goals
  -/


theorem eraseLead_ne_zero (f0 : 2 ≤ #f.support) : eraseLead f ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    f0 : LE.le 2 f.support.card
    ⊢ Ne f.eraseLead 0
  -/
  rw [Ne, ← card_support_eq_zero, eraseLead_support]
  exact
    (zero_lt_one.trans_le <| (tsub_le_tsub_right f0 1).trans Finset.pred_card_le_card_erase).ne.symm


theorem lt_natDegree_of_mem_eraseLead_support {a : ℕ} (h : a ∈ (eraseLead f).support) :
    a < f.natDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    a : Nat
    h : Membership.mem f.eraseLead.support a
    ⊢ LT.lt a f.natDegree
  -/
  rw [eraseLead_support, mem_erase] at h
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    a : Nat
    h : And (Ne a f.natDegree) (Membership.mem f.support a)
    ⊢ LT.lt a f.natDegree
  -/
  exact (le_natDegree_of_mem_supp a h.2).lt_of_ne h.1
  /-
    🎉 no goals
  -/


theorem ne_natDegree_of_mem_eraseLead_support {a : ℕ} (h : a ∈ (eraseLead f).support) :
    a ≠ f.natDegree :=
  (lt_natDegree_of_mem_eraseLead_support h).ne


theorem natDegree_not_mem_eraseLead_support : f.natDegree ∉ (eraseLead f).support := fun h =>
  ne_natDegree_of_mem_eraseLead_support h rfl


theorem eraseLead_support_card_lt (h : f ≠ 0) : #(eraseLead f).support < #f.support := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f 0
    ⊢ LT.lt f.eraseLead.support.card f.support.card
  -/
  rw [eraseLead_support]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f 0
    ⊢ LT.lt (f.support.erase f.natDegree).card f.support.card
  -/
  exact card_lt_card (erase_ssubset <| natDegree_mem_support_of_nonzero h)
  /-
    🎉 no goals
  -/


theorem card_support_eraseLead_add_one (h : f ≠ 0) : #f.eraseLead.support + 1 = #f.support := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f 0
    ⊢ Eq (HAdd.hAdd f.eraseLead.support.card 1) f.support.card
  -/
  set c := #f.support with hc
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f 0
    c : Nat := f.support.card
    hc : Eq c f.support.card
    ⊢ Eq (HAdd.hAdd f.eraseLead.support.card 1) c
  -/
  cases h₁ : c
  case zero =>
    by_contra
    exact h (card_support_eq_zero.mp h₁)
  case succ =>
    rw [eraseLead_support, card_erase_of_mem (natDegree_mem_support_of_nonzero h), ← hc, h₁]
    rfl


@[simp]
theorem card_support_eraseLead : #f.eraseLead.support = #f.support - 1 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq f.eraseLead.support.card (HSub.hSub f.support.card 1)
  -/
  by_cases hf : f = 0
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Eq f 0
      ⊢ Eq f.eraseLead.support.card (HSub.hSub f.support.card 1)
    -/
  · rw [hf, eraseLead_zero, support_zero, card_empty]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f 0)
      ⊢ Eq f.eraseLead.support.card (HSub.hSub f.support.card 1)
    -/
  · rw [← card_support_eraseLead_add_one hf, add_tsub_cancel_right]
    /-
      🎉 no goals
    -/


theorem card_support_eraseLead' {c : ℕ} (fc : #f.support = c + 1) :
    #f.eraseLead.support = c := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    c : Nat
    fc : Eq f.support.card (HAdd.hAdd c 1)
    ⊢ Eq f.eraseLead.support.card c
  -/
  rw [card_support_eraseLead, fc, add_tsub_cancel_right]
  /-
    🎉 no goals
  -/


theorem card_support_eq_one_of_eraseLead_eq_zero (h₀ : f ≠ 0) (h₁ : f.eraseLead = 0) :
    #f.support = 1 :=
  (card_support_eq_zero.mpr h₁ ▸ card_support_eraseLead_add_one h₀).symm


theorem card_support_le_one_of_eraseLead_eq_zero (h : f.eraseLead = 0) : #f.support ≤ 1 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Eq f.eraseLead 0
    ⊢ LE.le f.support.card 1
  -/
  by_cases hpz : f = 0
  /-
    case pos
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Eq f.eraseLead 0
    hpz : Eq f 0
    ⊢ LE.le f.support.card 1
  -/
  case pos => simp [hpz]
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Eq f.eraseLead 0
    hpz : Not (Eq f 0)
    ⊢ LE.le f.support.card 1
  -/
  case neg => exact le_of_eq (card_support_eq_one_of_eraseLead_eq_zero hpz h)
  /-
    🎉 no goals
  -/


@[simp]
theorem eraseLead_monomial (i : ℕ) (r : R) : eraseLead (monomial i r) = 0 := by
  classical
  by_cases hr : r = 0
  · subst r
    simp only [monomial_zero_right, eraseLead_zero]
  · rw [eraseLead, natDegree_monomial, if_neg hr, erase_monomial]


@[simp]
theorem eraseLead_C (r : R) : eraseLead (C r) = 0 :=
  eraseLead_monomial _ _


@[simp]
theorem eraseLead_X : eraseLead (X : R[X]) = 0 :=
  eraseLead_monomial _ _


@[simp]
theorem eraseLead_X_pow (n : ℕ) : eraseLead (X ^ n : R[X]) = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n).eraseLead 0
  -/
  rw [X_pow_eq_monomial, eraseLead_monomial]
  /-
    🎉 no goals
  -/


@[simp]
theorem eraseLead_C_mul_X_pow (r : R) (n : ℕ) : eraseLead (C r * X ^ n) = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    n : Nat
    ⊢ Eq (HMul.hMul (Polynomial.C r) (HPow.hPow Polynomial.X n)).eraseLead 0
  -/
  rw [C_mul_X_pow_eq_monomial, eraseLead_monomial]
  /-
    🎉 no goals
  -/


@[simp] lemma eraseLead_C_mul_X (r : R) : eraseLead (C r * X) = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    ⊢ Eq (HMul.hMul (Polynomial.C r) Polynomial.X).eraseLead 0
  -/
  simpa using eraseLead_C_mul_X_pow _ 1
  /-
    🎉 no goals
  -/


theorem eraseLead_add_of_degree_lt_left {p q : R[X]} (pq : q.degree < p.degree) :
    (p + q).eraseLead = p.eraseLead + q := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p q : Polynomial R
    pq : LT.lt q.degree p.degree
    ⊢ Eq (HAdd.hAdd p q).eraseLead (HAdd.hAdd p.eraseLead q)
  -/
  ext n
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    p q : Polynomial R
    pq : LT.lt q.degree p.degree
    n : Nat
    ⊢ Eq ((HAdd.hAdd p q).eraseLead.coeff n) ((HAdd.hAdd p.eraseLead q).coeff n)
  -/
  by_cases nd : n = p.natDegree
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p q : Polynomial R
      pq : LT.lt q.degree p.degree
      n : Nat
      nd : Eq n p.natDegree
      ⊢ Eq ((HAdd.hAdd p q).eraseLead.coeff n) ((HAdd.hAdd p.eraseLead q).coeff n)
    -/
  · rw [nd, eraseLead_coeff, if_pos (natDegree_add_eq_left_of_degree_lt pq).symm]
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p q : Polynomial R
      pq : LT.lt q.degree p.degree
      n : Nat
      nd : Eq n p.natDegree
      ⊢ Eq 0 ((HAdd.hAdd p.eraseLead q).coeff p.natDegree)
    -/
    simpa using (coeff_eq_zero_of_degree_lt (lt_of_lt_of_le pq degree_le_natDegree)).symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      p q : Polynomial R
      pq : LT.lt q.degree p.degree
      n : Nat
      nd : Not (Eq n p.natDegree)
      ⊢ Eq ((HAdd.hAdd p q).eraseLead.coeff n) ((HAdd.hAdd p.eraseLead q).coeff n)
    -/
  · rw [eraseLead_coeff, coeff_add, coeff_add, eraseLead_coeff, if_neg, if_neg nd]
    /-
      case neg.hnc
      R : Type u_1
      inst✝ : Semiring R
      p q : Polynomial R
      pq : LT.lt q.degree p.degree
      n : Nat
      nd : Not (Eq n p.natDegree)
      ⊢ Not (Eq n (HAdd.hAdd p q).natDegree)
    -/
    rintro rfl
    /-
      case neg.hnc
      R : Type u_1
      inst✝ : Semiring R
      p q : Polynomial R
      pq : LT.lt q.degree p.degree
      nd : Not (Eq (HAdd.hAdd p q).natDegree p.natDegree)
      ⊢ False
    -/
    exact nd (natDegree_add_eq_left_of_degree_lt pq)
    /-
      🎉 no goals
    -/


theorem eraseLead_add_of_natDegree_lt_left {p q : R[X]} (pq : q.natDegree < p.natDegree) :
    (p + q).eraseLead = p.eraseLead + q :=
  eraseLead_add_of_degree_lt_left (degree_lt_degree pq)


theorem eraseLead_add_of_degree_lt_right {p q : R[X]} (pq : p.degree < q.degree) :
    (p + q).eraseLead = p + q.eraseLead := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p q : Polynomial R
    pq : LT.lt p.degree q.degree
    ⊢ Eq (HAdd.hAdd p q).eraseLead (HAdd.hAdd p q.eraseLead)
  -/
  ext n
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    p q : Polynomial R
    pq : LT.lt p.degree q.degree
    n : Nat
    ⊢ Eq ((HAdd.hAdd p q).eraseLead.coeff n) ((HAdd.hAdd p q.eraseLead).coeff n)
  -/
  by_cases nd : n = q.natDegree
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p q : Polynomial R
      pq : LT.lt p.degree q.degree
      n : Nat
      nd : Eq n q.natDegree
      ⊢ Eq ((HAdd.hAdd p q).eraseLead.coeff n) ((HAdd.hAdd p q.eraseLead).coeff n)
    -/
  · rw [nd, eraseLead_coeff, if_pos (natDegree_add_eq_right_of_degree_lt pq).symm]
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p q : Polynomial R
      pq : LT.lt p.degree q.degree
      n : Nat
      nd : Eq n q.natDegree
      ⊢ Eq 0 ((HAdd.hAdd p q.eraseLead).coeff q.natDegree)
    -/
    simpa using (coeff_eq_zero_of_degree_lt (lt_of_lt_of_le pq degree_le_natDegree)).symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      p q : Polynomial R
      pq : LT.lt p.degree q.degree
      n : Nat
      nd : Not (Eq n q.natDegree)
      ⊢ Eq ((HAdd.hAdd p q).eraseLead.coeff n) ((HAdd.hAdd p q.eraseLead).coeff n)
    -/
  · rw [eraseLead_coeff, coeff_add, coeff_add, eraseLead_coeff, if_neg, if_neg nd]
    /-
      case neg.hnc
      R : Type u_1
      inst✝ : Semiring R
      p q : Polynomial R
      pq : LT.lt p.degree q.degree
      n : Nat
      nd : Not (Eq n q.natDegree)
      ⊢ Not (Eq n (HAdd.hAdd p q).natDegree)
    -/
    rintro rfl
    /-
      case neg.hnc
      R : Type u_1
      inst✝ : Semiring R
      p q : Polynomial R
      pq : LT.lt p.degree q.degree
      nd : Not (Eq (HAdd.hAdd p q).natDegree q.natDegree)
      ⊢ False
    -/
    exact nd (natDegree_add_eq_right_of_degree_lt pq)
    /-
      🎉 no goals
    -/


theorem eraseLead_add_of_natDegree_lt_right {p q : R[X]} (pq : p.natDegree < q.natDegree) :
    (p + q).eraseLead = p + q.eraseLead :=
  eraseLead_add_of_degree_lt_right (degree_lt_degree pq)


theorem eraseLead_degree_le : (eraseLead f).degree ≤ f.degree :=
  f.degree_erase_le _


theorem degree_eraseLead_lt (hf : f ≠ 0) : (eraseLead f).degree < f.degree :=
  f.degree_erase_lt hf


theorem eraseLead_natDegree_le_aux : (eraseLead f).natDegree ≤ f.natDegree :=
  natDegree_le_natDegree eraseLead_degree_le


theorem eraseLead_natDegree_lt (f0 : 2 ≤ #f.support) : (eraseLead f).natDegree < f.natDegree :=
  lt_of_le_of_ne eraseLead_natDegree_le_aux <|
    ne_natDegree_of_mem_eraseLead_support <|
      natDegree_mem_support_of_nonzero <| eraseLead_ne_zero f0


theorem natDegree_pos_of_eraseLead_ne_zero (h : f.eraseLead ≠ 0) : 0 < f.natDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f.eraseLead 0
    ⊢ LT.lt 0 f.natDegree
  -/
  by_contra h₂
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f.eraseLead 0
    h₂ : Not (LT.lt 0 f.natDegree)
    ⊢ False
  -/
  rw [eq_C_of_natDegree_eq_zero (Nat.eq_zero_of_not_pos h₂)] at h
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne (Polynomial.C (f.coeff 0)).eraseLead 0
    h₂ : Not (LT.lt 0 f.natDegree)
    ⊢ False
  -/
  simp at h
  /-
    🎉 no goals
  -/


theorem eraseLead_natDegree_lt_or_eraseLead_eq_zero (f : R[X]) :
    (eraseLead f).natDegree < f.natDegree ∨ f.eraseLead = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Or (LT.lt f.eraseLead.natDegree f.natDegree) (Eq f.eraseLead 0)
  -/
  by_cases h : #f.support ≤ 1
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      h : LE.le f.support.card 1
      ⊢ Or (LT.lt f.eraseLead.natDegree f.natDegree) (Eq f.eraseLead 0)
    -/
  · right
    /-
      case pos.h
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      h : LE.le f.support.card 1
      ⊢ Eq f.eraseLead 0
    -/
    rw [← C_mul_X_pow_eq_self h]
    /-
      case pos.h
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      h : LE.le f.support.card 1
      ⊢ Eq (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow Polynomial.X f.natDeg …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      h : Not (LE.le f.support.card 1)
      ⊢ Or (LT.lt f.eraseLead.natDegree f.natDegree) (Eq f.eraseLead 0)
    -/
  · left
    /-
      case neg.h
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      h : Not (LE.le f.support.card 1)
      ⊢ LT.lt f.eraseLead.natDegree f.natDegree
    -/
    apply eraseLead_natDegree_lt (lt_of_not_ge h)
    /-
      🎉 no goals
    -/


theorem eraseLead_natDegree_le (f : R[X]) : (eraseLead f).natDegree ≤ f.natDegree - 1 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ LE.le f.eraseLead.natDegree (HSub.hSub f.natDegree 1)
  -/
  rcases f.eraseLead_natDegree_lt_or_eraseLead_eq_zero with (h | h)
    /-
      case inl
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      h : LT.lt f.eraseLead.natDegree f.natDegree
      ⊢ LE.le f.eraseLead.natDegree (HSub.hSub f.natDegree 1)
    -/
  · exact Nat.le_sub_one_of_lt h
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      h : Eq f.eraseLead 0
      ⊢ LE.le f.eraseLead.natDegree (HSub.hSub f.natDegree 1)
    -/
  · simp only [h, natDegree_zero, zero_le]
    /-
      🎉 no goals
    -/


lemma natDegree_eraseLead (h : f.nextCoeff ≠ 0) : f.eraseLead.natDegree = f.natDegree - 1 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f.nextCoeff 0
    ⊢ Eq f.eraseLead.natDegree (HSub.hSub f.natDegree 1)
  -/
  have := natDegree_pos_of_nextCoeff_ne_zero h
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f.nextCoeff 0
    this : LT.lt 0 f.natDegree
    ⊢ Eq f.eraseLead.natDegree (HSub.hSub f.natDegree 1)
  -/
  refine f.eraseLead_natDegree_le.antisymm <| le_natDegree_of_ne_zero ?_
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f.nextCoeff 0
    this : LT.lt 0 f.natDegree
    ⊢ Ne (f.eraseLead.coeff (HSub.hSub f.natDegree 1)) 0
  -/
  rwa [eraseLead_coeff_of_ne _ (tsub_lt_self _ _).ne, ← nextCoeff_of_natDegree_pos]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f.nextCoeff 0
    this : LT.lt 0 f.natDegree
    ⊢ LT.lt 0 f.natDegree
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


lemma natDegree_eraseLead_add_one (h : f.nextCoeff ≠ 0) :
    f.eraseLead.natDegree + 1 = f.natDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f.nextCoeff 0
    ⊢ Eq (HAdd.hAdd f.eraseLead.natDegree 1) f.natDegree
  -/
  rw [natDegree_eraseLead h, tsub_add_cancel_of_le]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f.nextCoeff 0
    ⊢ LE.le 1 f.natDegree
  -/
  exact natDegree_pos_of_nextCoeff_ne_zero h
  /-
    🎉 no goals
  -/


theorem natDegree_eraseLead_le_of_nextCoeff_eq_zero (h : f.nextCoeff = 0) :
    f.eraseLead.natDegree ≤ f.natDegree - 2 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Eq f.nextCoeff 0
    ⊢ LE.le f.eraseLead.natDegree (HSub.hSub f.natDegree 2)
  -/
  refine natDegree_le_pred (n := f.natDegree - 1) (eraseLead_natDegree_le f) ?_
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Eq f.nextCoeff 0
    ⊢ Eq (f.eraseLead.coeff (HSub.hSub f.natDegree 1)) 0
  -/
  rw [nextCoeff_eq_zero, natDegree_eq_zero] at h
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Or (Exists fun x => Eq (Polynomial.C x) f) (And (LT.lt 0 f.natDegree) (Eq  …
    ⊢ Eq (f.eraseLead.coeff (HSub.hSub f.natDegree 1)) 0
  -/
  obtain ⟨a, rfl⟩ | ⟨hf, h⟩ := h
    /-
      case inl.intro
      R : Type u_1
      inst✝ : Semiring R
      a : R
      ⊢ Eq ((Polynomial.C a).eraseLead.coeff (HSub.hSub (Polynomial.C a).natDegree 1 …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    hf : LT.lt 0 f.natDegree
    h : Eq (f.coeff (HSub.hSub f.natDegree 1)) 0
    ⊢ Eq (f.eraseLead.coeff (HSub.hSub f.natDegree 1)) 0
  -/
  rw [eraseLead_coeff_of_ne _ (tsub_lt_self hf zero_lt_one).ne, ← nextCoeff_of_natDegree_pos hf]
  /-
    case inr.intro
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    hf : LT.lt 0 f.natDegree
    h : Eq (f.coeff (HSub.hSub f.natDegree 1)) 0
    ⊢ Eq f.nextCoeff 0
  -/
  simp [nextCoeff_eq_zero, h, eq_zero_or_pos]
  /-
    🎉 no goals
  -/


lemma two_le_natDegree_of_nextCoeff_eraseLead (hlead : f.eraseLead ≠ 0)
    (hnext : f.nextCoeff = 0) : 2 ≤ f.natDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    hlead : Ne f.eraseLead 0
    hnext : Eq f.nextCoeff 0
    ⊢ LE.le 2 f.natDegree
  -/
  contrapose! hlead
  rw [Nat.lt_succ_iff, Nat.le_one_iff_eq_zero_or_eq_one, natDegree_eq_zero, natDegree_eq_one]
    at hlead
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    hnext : Eq f.nextCoeff 0
    hlead : Or (Exists fun x => Eq (Polynomial.C x) f) (Exists fun a => And (Ne a  …
    ⊢ Eq f.eraseLead 0
  -/
  obtain ⟨a, rfl⟩ | ⟨a, ha, b, rfl⟩ := hlead
    /-
      case inl.intro
      R : Type u_1
      inst✝ : Semiring R
      a : R
      hnext : Eq (Polynomial.C a).nextCoeff 0
      ⊢ Eq (Polynomial.C a).eraseLead 0
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      a : R
      ha : Ne a 0
      b : R
      hnext : Eq (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C  …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C b)).er …
    -/
  · rw [nextCoeff_C_mul_X_add_C ha] at hnext
    /-
      case inr.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      a : R
      ha : Ne a 0
      b : R
      hnext : Eq b 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C b)).er …
    -/
    subst b
    /-
      case inr.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      a : R
      ha : Ne a 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C 0)).er …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem leadingCoeff_eraseLead_eq_nextCoeff (h : f.nextCoeff ≠ 0) :
    f.eraseLead.leadingCoeff = f.nextCoeff := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f.nextCoeff 0
    ⊢ Eq f.eraseLead.leadingCoeff f.nextCoeff
  -/
  have := natDegree_pos_of_nextCoeff_ne_zero h
  rw [leadingCoeff, nextCoeff, natDegree_eraseLead h, if_neg,
    eraseLead_coeff_of_ne _ (tsub_lt_self _ _).ne]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Ne f.nextCoeff 0
    this : LT.lt 0 f.natDegree
    ⊢ LT.lt 0 f.natDegree
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


theorem nextCoeff_eq_zero_of_eraseLead_eq_zero (h : f.eraseLead = 0) : f.nextCoeff = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Eq f.eraseLead 0
    ⊢ Eq f.nextCoeff 0
  -/
  by_contra h₂
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    h : Eq f.eraseLead 0
    h₂ : Not (Eq f.nextCoeff 0)
    ⊢ False
  -/
  exact leadingCoeff_ne_zero.mp (leadingCoeff_eraseLead_eq_nextCoeff h₂ ▸ h₂) h
  /-
    🎉 no goals
  -/


/-- An induction lemma for polynomials. It takes a natural number `N` as a parameter, that is
required to be at least as big as the `nat_degree` of the polynomial.  This is useful to prove
results where you want to change each term in a polynomial to something else depending on the
`nat_degree` of the polynomial itself and not on the specific `nat_degree` of each term. -/
theorem induction_with_natDegree_le (P : R[X] → Prop) (N : ℕ) (P_0 : P 0)
    (P_C_mul_pow : ∀ n : ℕ, ∀ r : R, r ≠ 0 → n ≤ N → P (C r * X ^ n))
    (P_C_add : ∀ f g : R[X], f.natDegree < g.natDegree → g.natDegree ≤ N → P f → P g → P (f + g)) :
    ∀ f : R[X], f.natDegree ≤ N → P f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    P : Polynomial R → Prop
    N : Nat
    P_0 : P 0
    P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
    P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
    ⊢ ∀ (f : Polynomial R), LE.le f.natDegree N → P f
  -/
  intro f df
  /-
    R : Type u_1
    inst✝ : Semiring R
    P : Polynomial R → Prop
    N : Nat
    P_0 : P 0
    P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
    P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
    f : Polynomial R
    df : LE.le f.natDegree N
    ⊢ P f
  -/
  generalize hd : #f.support = c
  /-
    R : Type u_1
    inst✝ : Semiring R
    P : Polynomial R → Prop
    N : Nat
    P_0 : P 0
    P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
    P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
    f : Polynomial R
    df : LE.le f.natDegree N
    c : Nat
    hd : Eq f.support.card c
    ⊢ P f
  -/
  revert f
  /-
    R : Type u_1
    inst✝ : Semiring R
    P : Polynomial R → Prop
    N : Nat
    P_0 : P 0
    P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
    P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
    c : Nat
    ⊢ ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card c → P f
  -/
  induction' c with c hc
    /-
      case zero
      R : Type u_1
      inst✝ : Semiring R
      P : Polynomial R → Prop
      N : Nat
      P_0 : P 0
      P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
      P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
      ⊢ ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card 0 → P f
    -/
  · intro f _ f0
    /-
      case zero
      R : Type u_1
      inst✝ : Semiring R
      P : Polynomial R → Prop
      N : Nat
      P_0 : P 0
      P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
      P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
      f : Polynomial R
      df✝ : LE.le f.natDegree N
      f0 : Eq f.support.card 0
      ⊢ P f
    -/
    convert P_0
    /-
      case h.e'_1
      R : Type u_1
      inst✝ : Semiring R
      P : Polynomial R → Prop
      N : Nat
      P_0 : P 0
      P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
      P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
      f : Polynomial R
      df✝ : LE.le f.natDegree N
      f0 : Eq f.support.card 0
      ⊢ Eq f 0
    -/
    simpa [support_eq_empty, card_eq_zero] using f0
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝ : Semiring R
      P : Polynomial R → Prop
      N : Nat
      P_0 : P 0
      P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
      P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
      c : Nat
      hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card c → P f
      ⊢ ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd c 1 …
    -/
  · intro f df f0
    /-
      case succ
      R : Type u_1
      inst✝ : Semiring R
      P : Polynomial R → Prop
      N : Nat
      P_0 : P 0
      P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
      P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
      c : Nat
      hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card c → P f
      f : Polynomial R
      df : LE.le f.natDegree N
      f0 : Eq f.support.card (HAdd.hAdd c 1)
      ⊢ P f
    -/
    rw [← eraseLead_add_C_mul_X_pow f]
    /-
      case succ
      R : Type u_1
      inst✝ : Semiring R
      P : Polynomial R → Prop
      N : Nat
      P_0 : P 0
      P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
      P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
      c : Nat
      hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card c → P f
      f : Polynomial R
      df : LE.le f.natDegree N
      f0 : Eq f.support.card (HAdd.hAdd c 1)
      ⊢ P (HAdd.hAdd f.eraseLead (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow …
    -/
    cases c
      /-
        case succ.zero
        R : Type u_1
        inst✝ : Semiring R
        P : Polynomial R → Prop
        N : Nat
        P_0 : P 0
        P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
        P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
        f : Polynomial R
        df : LE.le f.natDegree N
        hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card 0 → P f
        f0 : Eq f.support.card (HAdd.hAdd 0 1)
        ⊢ P (HAdd.hAdd f.eraseLead (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow …
      -/
    · convert P_C_mul_pow f.natDegree f.leadingCoeff ?_ df using 1
        /-
          case h.e'_1
          R : Type u_1
          inst✝ : Semiring R
          P : Polynomial R → Prop
          N : Nat
          P_0 : P 0
          P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
          P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
          f : Polynomial R
          df : LE.le f.natDegree N
          hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card 0 → P f
          f0 : Eq f.support.card (HAdd.hAdd 0 1)
          ⊢ Eq (HAdd.hAdd f.eraseLead (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPo …
        -/
      · convert zero_add (C (leadingCoeff f) * X ^ f.natDegree)
        /-
          case h.e'_2.h.e'_5
          R : Type u_1
          inst✝ : Semiring R
          P : Polynomial R → Prop
          N : Nat
          P_0 : P 0
          P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
          P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
          f : Polynomial R
          df : LE.le f.natDegree N
          hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card 0 → P f
          f0 : Eq f.support.card (HAdd.hAdd 0 1)
          ⊢ Eq f.eraseLead 0
        -/
        rw [← card_support_eq_zero, card_support_eraseLead' f0]
        /-
          🎉 no goals
        -/
        /-
          case succ.zero
          R : Type u_1
          inst✝ : Semiring R
          P : Polynomial R → Prop
          N : Nat
          P_0 : P 0
          P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
          P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
          f : Polynomial R
          df : LE.le f.natDegree N
          hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card 0 → P f
          f0 : Eq f.support.card (HAdd.hAdd 0 1)
          ⊢ Ne f.leadingCoeff 0
        -/
      · rw [leadingCoeff_ne_zero, Ne, ← card_support_eq_zero, f0]
        /-
          case succ.zero
          R : Type u_1
          inst✝ : Semiring R
          P : Polynomial R → Prop
          N : Nat
          P_0 : P 0
          P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
          P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
          f : Polynomial R
          df : LE.le f.natDegree N
          hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card 0 → P f
          f0 : Eq f.support.card (HAdd.hAdd 0 1)
          ⊢ Not (Eq (HAdd.hAdd 0 1) 0)
        -/
        exact zero_ne_one.symm
        /-
          🎉 no goals
        -/
    /-
      case succ.succ
      R : Type u_1
      inst✝ : Semiring R
      P : Polynomial R → Prop
      N : Nat
      P_0 : P 0
      P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
      P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
      f : Polynomial R
      df : LE.le f.natDegree N
      n✝ : Nat
      hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
      f0 : Eq f.support.card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
      ⊢ P (HAdd.hAdd f.eraseLead (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow …
    -/
    refine P_C_add f.eraseLead _ ?_ ?_ ?_ ?_
      /-
        case succ.succ.refine_1
        R : Type u_1
        inst✝ : Semiring R
        P : Polynomial R → Prop
        N : Nat
        P_0 : P 0
        P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
        P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
        f : Polynomial R
        df : LE.le f.natDegree N
        n✝ : Nat
        hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
        f0 : Eq f.support.card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
        ⊢ LT.lt f.eraseLead.natDegree (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.h …
      -/
    · refine (eraseLead_natDegree_lt ?_).trans_le (le_of_eq ?_)
        /-
          case succ.succ.refine_1.refine_1
          R : Type u_1
          inst✝ : Semiring R
          P : Polynomial R → Prop
          N : Nat
          P_0 : P 0
          P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
          P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
          f : Polynomial R
          df : LE.le f.natDegree N
          n✝ : Nat
          hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
          f0 : Eq f.support.card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
          ⊢ LE.le 2 f.support.card
        -/
      · exact (Nat.succ_le_succ (Nat.succ_le_succ (Nat.zero_le _))).trans f0.ge
        /-
          🎉 no goals
        -/
        /-
          case succ.succ.refine_1.refine_2
          R : Type u_1
          inst✝ : Semiring R
          P : Polynomial R → Prop
          N : Nat
          P_0 : P 0
          P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
          P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
          f : Polynomial R
          df : LE.le f.natDegree N
          n✝ : Nat
          hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
          f0 : Eq f.support.card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
          ⊢ Eq f.natDegree (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow Polynomia …
        -/
      · rw [natDegree_C_mul_X_pow _ _ (leadingCoeff_ne_zero.mpr _)]
        /-
          R : Type u_1
          inst✝ : Semiring R
          P : Polynomial R → Prop
          N : Nat
          P_0 : P 0
          P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
          P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
          f : Polynomial R
          df : LE.le f.natDegree N
          n✝ : Nat
          hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
          f0 : Eq f.support.card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
          ⊢ Ne f 0
        -/
        rintro rfl
        /-
          R : Type u_1
          inst✝ : Semiring R
          P : Polynomial R → Prop
          N : Nat
          P_0 : P 0
          P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
          P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
          n✝ : Nat
          hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
          df : LE.le (Polynomial.natDegree 0) N
          f0 : Eq (Polynomial.support 0).card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
          ⊢ False
        -/
        simp at f0
        /-
          🎉 no goals
        -/
      /-
        case succ.succ.refine_2
        R : Type u_1
        inst✝ : Semiring R
        P : Polynomial R → Prop
        N : Nat
        P_0 : P 0
        P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
        P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
        f : Polynomial R
        df : LE.le f.natDegree N
        n✝ : Nat
        hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
        f0 : Eq f.support.card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
        ⊢ LE.le (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow Polynomial.X f.nat …
      -/
    · exact (natDegree_C_mul_X_pow_le f.leadingCoeff f.natDegree).trans df
      /-
        🎉 no goals
      -/
      /-
        case succ.succ.refine_3
        R : Type u_1
        inst✝ : Semiring R
        P : Polynomial R → Prop
        N : Nat
        P_0 : P 0
        P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
        P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
        f : Polynomial R
        df : LE.le f.natDegree N
        n✝ : Nat
        hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
        f0 : Eq f.support.card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
        ⊢ P f.eraseLead
      -/
    · exact hc _ (eraseLead_natDegree_le_aux.trans df) (card_support_eraseLead' f0)
      /-
        🎉 no goals
      -/
      /-
        case succ.succ.refine_4
        R : Type u_1
        inst✝ : Semiring R
        P : Polynomial R → Prop
        N : Nat
        P_0 : P 0
        P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
        P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
        f : Polynomial R
        df : LE.le f.natDegree N
        n✝ : Nat
        hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
        f0 : Eq f.support.card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
        ⊢ P (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow Polynomial.X f.natDegr …
      -/
    · refine P_C_mul_pow _ _ ?_ df
      /-
        case succ.succ.refine_4
        R : Type u_1
        inst✝ : Semiring R
        P : Polynomial R → Prop
        N : Nat
        P_0 : P 0
        P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
        P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
        f : Polynomial R
        df : LE.le f.natDegree N
        n✝ : Nat
        hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
        f0 : Eq f.support.card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
        ⊢ Ne f.leadingCoeff 0
      -/
      rw [Ne, leadingCoeff_eq_zero, ← card_support_eq_zero, f0]
      /-
        case succ.succ.refine_4
        R : Type u_1
        inst✝ : Semiring R
        P : Polynomial R → Prop
        N : Nat
        P_0 : P 0
        P_C_mul_pow : ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → P (HMul.hMul (Polynomi …
        P_C_add : ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natD …
        f : Polynomial R
        df : LE.le f.natDegree N
        n✝ : Nat
        hc : ∀ (f : Polynomial R), LE.le f.natDegree N → Eq f.support.card (HAdd.hAdd  …
        f0 : Eq f.support.card (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
        ⊢ Not (Eq (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) 0)
      -/
      exact Nat.succ_ne_zero _
      /-
        🎉 no goals
      -/


/-- Let `φ : R[x] → S[x]` be an additive map, `k : ℕ` a bound, and `fu : ℕ → ℕ` a
"sufficiently monotone" map.  Assume also that
* `φ` maps to `0` all monomials of degree less than `k`,
* `φ` maps each monomial `m` in `R[x]` to a polynomial `φ m` of degree `fu (deg m)`.
Then, `φ` maps each polynomial `p` in `R[x]` to a polynomial of degree `fu (deg p)`. -/
theorem mono_map_natDegree_eq {S F : Type*} [Semiring S]
    [FunLike F R[X] S[X]] [AddMonoidHomClass F R[X] S[X]] {φ : F}
    {p : R[X]} (k : ℕ) (fu : ℕ → ℕ) (fu0 : ∀ {n}, n ≤ k → fu n = 0)
    (fc : ∀ {n m}, k ≤ n → n < m → fu n < fu m) (φ_k : ∀ {f : R[X]}, f.natDegree < k → φ f = 0)
    (φ_mon_nat : ∀ n c, c ≠ 0 → (φ (monomial n c)).natDegree = fu n) :
    (φ p).natDegree = fu p.natDegree := by
  refine induction_with_natDegree_le (fun p => (φ p).natDegree = fu p.natDegree)
    p.natDegree (by simp [fu0]) ?_ ?_ _ rfl.le
    /-
      case refine_1
      R : Type u_1
      inst✝³ : Semiring R
      S : Type u_2
      F : Type u_3
      inst✝² : Semiring S
      inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
      inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
      φ : F
      p : Polynomial R
      k : Nat
      fu : Nat → Nat
      fu0 : ∀ {n : Nat}, LE.le n k → Eq (fu n) 0
      fc : ∀ {n m : Nat}, LE.le k n → LT.lt n m → LT.lt (fu n) (fu m)
      φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree k → Eq (φ f) 0
      φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
      ⊢ ∀ (n : Nat) (r : R), Ne r 0 → LE.le n p.natDegree → (fun p => Eq (φ p).natDe …
    -/
  · intro n r r0 _
    /-
      case refine_1
      R : Type u_1
      inst✝³ : Semiring R
      S : Type u_2
      F : Type u_3
      inst✝² : Semiring S
      inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
      inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
      φ : F
      p : Polynomial R
      k : Nat
      fu : Nat → Nat
      fu0 : ∀ {n : Nat}, LE.le n k → Eq (fu n) 0
      fc : ∀ {n m : Nat}, LE.le k n → LT.lt n m → LT.lt (fu n) (fu m)
      φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree k → Eq (φ f) 0
      φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
      n : Nat
      r : R
      r0 : Ne r 0
      a✝ : LE.le n p.natDegree
      ⊢ Eq (φ (HMul.hMul (Polynomial.C r) (HPow.hPow Polynomial.X n))).natDegree (fu …
    -/
    rw [natDegree_C_mul_X_pow _ _ r0, C_mul_X_pow_eq_monomial, φ_mon_nat _ _ r0]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝³ : Semiring R
      S : Type u_2
      F : Type u_3
      inst✝² : Semiring S
      inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
      inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
      φ : F
      p : Polynomial R
      k : Nat
      fu : Nat → Nat
      fu0 : ∀ {n : Nat}, LE.le n k → Eq (fu n) 0
      fc : ∀ {n m : Nat}, LE.le k n → LT.lt n m → LT.lt (fu n) (fu m)
      φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree k → Eq (φ f) 0
      φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
      ⊢ ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natDegree p. …
    -/
  · intro f g fg _ fk gk
    /-
      case refine_2
      R : Type u_1
      inst✝³ : Semiring R
      S : Type u_2
      F : Type u_3
      inst✝² : Semiring S
      inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
      inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
      φ : F
      p : Polynomial R
      k : Nat
      fu : Nat → Nat
      fu0 : ∀ {n : Nat}, LE.le n k → Eq (fu n) 0
      fc : ∀ {n m : Nat}, LE.le k n → LT.lt n m → LT.lt (fu n) (fu m)
      φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree k → Eq (φ f) 0
      φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
      f g : Polynomial R
      fg : LT.lt f.natDegree g.natDegree
      a✝ : LE.le g.natDegree p.natDegree
      fk : Eq (φ f).natDegree (fu f.natDegree)
      gk : Eq (φ g).natDegree (fu g.natDegree)
      ⊢ Eq (φ (HAdd.hAdd f g)).natDegree (fu (HAdd.hAdd f g).natDegree)
    -/
    rw [natDegree_add_eq_right_of_natDegree_lt fg, _root_.map_add]
    /-
      case refine_2
      R : Type u_1
      inst✝³ : Semiring R
      S : Type u_2
      F : Type u_3
      inst✝² : Semiring S
      inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
      inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
      φ : F
      p : Polynomial R
      k : Nat
      fu : Nat → Nat
      fu0 : ∀ {n : Nat}, LE.le n k → Eq (fu n) 0
      fc : ∀ {n m : Nat}, LE.le k n → LT.lt n m → LT.lt (fu n) (fu m)
      φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree k → Eq (φ f) 0
      φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
      f g : Polynomial R
      fg : LT.lt f.natDegree g.natDegree
      a✝ : LE.le g.natDegree p.natDegree
      fk : Eq (φ f).natDegree (fu f.natDegree)
      gk : Eq (φ g).natDegree (fu g.natDegree)
      ⊢ Eq (HAdd.hAdd (φ f) (φ g)).natDegree (fu g.natDegree)
    -/
    by_cases FG : k ≤ f.natDegree
      /-
        case pos
        R : Type u_1
        inst✝³ : Semiring R
        S : Type u_2
        F : Type u_3
        inst✝² : Semiring S
        inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
        inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
        φ : F
        p : Polynomial R
        k : Nat
        fu : Nat → Nat
        fu0 : ∀ {n : Nat}, LE.le n k → Eq (fu n) 0
        fc : ∀ {n m : Nat}, LE.le k n → LT.lt n m → LT.lt (fu n) (fu m)
        φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree k → Eq (φ f) 0
        φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
        f g : Polynomial R
        fg : LT.lt f.natDegree g.natDegree
        a✝ : LE.le g.natDegree p.natDegree
        fk : Eq (φ f).natDegree (fu f.natDegree)
        gk : Eq (φ g).natDegree (fu g.natDegree)
        FG : LE.le k f.natDegree
        ⊢ Eq (HAdd.hAdd (φ f) (φ g)).natDegree (fu g.natDegree)
      -/
    · rw [natDegree_add_eq_right_of_natDegree_lt, gk]
      /-
        case pos
        R : Type u_1
        inst✝³ : Semiring R
        S : Type u_2
        F : Type u_3
        inst✝² : Semiring S
        inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
        inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
        φ : F
        p : Polynomial R
        k : Nat
        fu : Nat → Nat
        fu0 : ∀ {n : Nat}, LE.le n k → Eq (fu n) 0
        fc : ∀ {n m : Nat}, LE.le k n → LT.lt n m → LT.lt (fu n) (fu m)
        φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree k → Eq (φ f) 0
        φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
        f g : Polynomial R
        fg : LT.lt f.natDegree g.natDegree
        a✝ : LE.le g.natDegree p.natDegree
        fk : Eq (φ f).natDegree (fu f.natDegree)
        gk : Eq (φ g).natDegree (fu g.natDegree)
        FG : LE.le k f.natDegree
        ⊢ LT.lt (φ f).natDegree (φ g).natDegree
      -/
      rw [fk, gk]
      /-
        case pos
        R : Type u_1
        inst✝³ : Semiring R
        S : Type u_2
        F : Type u_3
        inst✝² : Semiring S
        inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
        inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
        φ : F
        p : Polynomial R
        k : Nat
        fu : Nat → Nat
        fu0 : ∀ {n : Nat}, LE.le n k → Eq (fu n) 0
        fc : ∀ {n m : Nat}, LE.le k n → LT.lt n m → LT.lt (fu n) (fu m)
        φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree k → Eq (φ f) 0
        φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
        f g : Polynomial R
        fg : LT.lt f.natDegree g.natDegree
        a✝ : LE.le g.natDegree p.natDegree
        fk : Eq (φ f).natDegree (fu f.natDegree)
        gk : Eq (φ g).natDegree (fu g.natDegree)
        FG : LE.le k f.natDegree
        ⊢ LT.lt (fu f.natDegree) (fu g.natDegree)
      -/
      exact fc FG fg
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝³ : Semiring R
        S : Type u_2
        F : Type u_3
        inst✝² : Semiring S
        inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
        inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
        φ : F
        p : Polynomial R
        k : Nat
        fu : Nat → Nat
        fu0 : ∀ {n : Nat}, LE.le n k → Eq (fu n) 0
        fc : ∀ {n m : Nat}, LE.le k n → LT.lt n m → LT.lt (fu n) (fu m)
        φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree k → Eq (φ f) 0
        φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
        f g : Polynomial R
        fg : LT.lt f.natDegree g.natDegree
        a✝ : LE.le g.natDegree p.natDegree
        fk : Eq (φ f).natDegree (fu f.natDegree)
        gk : Eq (φ g).natDegree (fu g.natDegree)
        FG : Not (LE.le k f.natDegree)
        ⊢ Eq (HAdd.hAdd (φ f) (φ g)).natDegree (fu g.natDegree)
      -/
    · cases k
        /-
          case neg.zero
          R : Type u_1
          inst✝³ : Semiring R
          S : Type u_2
          F : Type u_3
          inst✝² : Semiring S
          inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
          inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
          φ : F
          p : Polynomial R
          fu : Nat → Nat
          φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
          f g : Polynomial R
          fg : LT.lt f.natDegree g.natDegree
          a✝ : LE.le g.natDegree p.natDegree
          fk : Eq (φ f).natDegree (fu f.natDegree)
          gk : Eq (φ g).natDegree (fu g.natDegree)
          fu0 : ∀ {n : Nat}, LE.le n 0 → Eq (fu n) 0
          fc : ∀ {n m : Nat}, LE.le 0 n → LT.lt n m → LT.lt (fu n) (fu m)
          φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree 0 → Eq (φ f) 0
          FG : Not (LE.le 0 f.natDegree)
          ⊢ Eq (HAdd.hAdd (φ f) (φ g)).natDegree (fu g.natDegree)
        -/
      · exact (FG (Nat.zero_le _)).elim
        /-
          🎉 no goals
        -/
        /-
          case neg.succ
          R : Type u_1
          inst✝³ : Semiring R
          S : Type u_2
          F : Type u_3
          inst✝² : Semiring S
          inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
          inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
          φ : F
          p : Polynomial R
          fu : Nat → Nat
          φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
          f g : Polynomial R
          fg : LT.lt f.natDegree g.natDegree
          a✝ : LE.le g.natDegree p.natDegree
          fk : Eq (φ f).natDegree (fu f.natDegree)
          gk : Eq (φ g).natDegree (fu g.natDegree)
          n✝ : Nat
          fu0 : ∀ {n : Nat}, LE.le n (HAdd.hAdd n✝ 1) → Eq (fu n) 0
          fc : ∀ {n m : Nat}, LE.le (HAdd.hAdd n✝ 1) n → LT.lt n m → LT.lt (fu n) (fu m)
          φ_k : ∀ {f : Polynomial R}, LT.lt f.natDegree (HAdd.hAdd n✝ 1) → Eq (φ f) 0
          FG : Not (LE.le (HAdd.hAdd n✝ 1) f.natDegree)
          ⊢ Eq (HAdd.hAdd (φ f) (φ g)).natDegree (fu g.natDegree)
        -/
      · rwa [φ_k (not_le.mp FG), zero_add]
        /-
          🎉 no goals
        -/


theorem map_natDegree_eq_sub {S F : Type*} [Semiring S]
    [FunLike F R[X] S[X]] [AddMonoidHomClass F R[X] S[X]] {φ : F}
    {p : R[X]} {k : ℕ} (φ_k : ∀ f : R[X], f.natDegree < k → φ f = 0)
    (φ_mon : ∀ n c, c ≠ 0 → (φ (monomial n c)).natDegree = n - k) :
    (φ p).natDegree = p.natDegree - k :=
                                               /-
                                                 R : Type u_1
                                                 inst✝³ : Semiring R
                                                 S : Type u_2
                                                 F : Type u_3
                                                 inst✝² : Semiring S
                                                 inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
                                                 inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
                                                 φ : F
                                                 p : Polynomial R
                                                 k : Nat
                                                 φ_k : ∀ (f : Polynomial R), LT.lt f.natDegree k → Eq (φ f) 0
                                                 φ_mon : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).natDe …
                                                 ⊢ ∀ {n : Nat}, LE.le n k → Eq ((fun j => HSub.hSub j k) n) 0
                                               -/
  mono_map_natDegree_eq k (fun j => j - k) (by simp_all)
                                               /-
                                                 🎉 no goals
                                               -/
    (@fun _ _ h => (tsub_lt_tsub_iff_right h).mpr)
    (φ_k _) φ_mon


theorem map_natDegree_eq_natDegree {S F : Type*} [Semiring S]
    [FunLike F R[X] S[X]] [AddMonoidHomClass F R[X] S[X]]
    {φ : F} (p) (φ_mon_nat : ∀ n c, c ≠ 0 → (φ (monomial n c)).natDegree = n) :
    (φ p).natDegree = p.natDegree :=
                                                                    /-
                                                                      R : Type u_1
                                                                      inst✝³ : Semiring R
                                                                      S : Type u_2
                                                                      F : Type u_3
                                                                      inst✝² : Semiring S
                                                                      inst✝¹ : FunLike F (Polynomial R) (Polynomial S)
                                                                      inst✝ : AddMonoidHomClass F (Polynomial R) (Polynomial S)
                                                                      φ : F
                                                                      p : Polynomial R
                                                                      φ_mon_nat : ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).n …
                                                                      ⊢ ∀ (n : Nat) (c : R), Ne c 0 → Eq (φ ((Polynomial.monomial n) c)).natDegree ( …
                                                                    -/
  (map_natDegree_eq_sub (fun _ h => (Nat.not_lt_zero _ h).elim) (by simpa)).trans
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    p.natDegree.sub_zero


theorem card_support_eq' {n : ℕ} (k : Fin n → ℕ) (x : Fin n → R) (hk : Function.Injective k)
    (hx : ∀ i, x i ≠ 0) : #(∑ i, C (x i) * X ^ k i).support = n := by
  suffices (∑ i, C (x i) * X ^ k i).support = image k univ by
    rw [this, univ.card_image_of_injective hk, card_fin]
  simp_rw [Finset.ext_iff, mem_support_iff, finset_sum_coeff, coeff_C_mul_X_pow, mem_image,
    mem_univ, true_and]
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    k : Fin n → Nat
    x : Fin n → R
    hk : Function.Injective k
    hx : ∀ (i : Fin n), Ne (x i) 0
    ⊢ ∀ (a : Nat), Iff (Ne (Finset.univ.sum fun x_1 => ite (Eq a (k x_1)) (x x_1)  …
  -/
  refine fun i => ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      k : Fin n → Nat
      x : Fin n → R
      hk : Function.Injective k
      hx : ∀ (i : Fin n), Ne (x i) 0
      i : Nat
      h : Ne (Finset.univ.sum fun x_1 => ite (Eq i (k x_1)) (x x_1) 0) 0
      ⊢ Exists fun a => Eq (k a) i
    -/
  · obtain ⟨j, _, h⟩ := exists_ne_zero_of_sum_ne_zero h
    /-
      case refine_1.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      k : Fin n → Nat
      x : Fin n → R
      hk : Function.Injective k
      hx : ∀ (i : Fin n), Ne (x i) 0
      i : Nat
      h✝ : Ne (Finset.univ.sum fun x_1 => ite (Eq i (k x_1)) (x x_1) 0) 0
      j : Fin n
      left✝ : Membership.mem Finset.univ j
      h : Ne (ite (Eq i (k j)) (x j) 0) 0
      ⊢ Exists fun a => Eq (k a) i
    -/
    exact ⟨j, (ite_ne_right_iff.mp h).1.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      k : Fin n → Nat
      x : Fin n → R
      hk : Function.Injective k
      hx : ∀ (i : Fin n), Ne (x i) 0
      i : Nat
      ⊢ (Exists fun a => Eq (k a) i) → Ne (Finset.univ.sum fun x_1 => ite (Eq i (k x …
    -/
  · rintro ⟨j, _, rfl⟩
    /-
      case refine_2.intro.refl
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      k : Fin n → Nat
      x : Fin n → R
      hk : Function.Injective k
      hx : ∀ (i : Fin n), Ne (x i) 0
      j : Fin n
      ⊢ Ne (Finset.univ.sum fun x_1 => ite (Eq (k j) (k x_1)) (x x_1) 0) 0
    -/
    rw [sum_eq_single_of_mem j (mem_univ j), if_pos rfl]
      /-
        case refine_2.intro.refl
        R : Type u_1
        inst✝ : Semiring R
        n : Nat
        k : Fin n → Nat
        x : Fin n → R
        hk : Function.Injective k
        hx : ∀ (i : Fin n), Ne (x i) 0
        j : Fin n
        ⊢ Ne (x j) 0
      -/
    · exact hx j
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.refl
        R : Type u_1
        inst✝ : Semiring R
        n : Nat
        k : Fin n → Nat
        x : Fin n → R
        hk : Function.Injective k
        hx : ∀ (i : Fin n), Ne (x i) 0
        j : Fin n
        ⊢ ∀ (b : Fin n), Membership.mem Finset.univ b → Ne b j → Eq (ite (Eq (k j) (k  …
      -/
    · exact fun m _ hmj => if_neg fun h => hmj.symm (hk h)
      /-
        🎉 no goals
      -/


theorem card_support_eq {n : ℕ} :
    #f.support = n ↔
      ∃ (k : Fin n → ℕ) (x : Fin n → R) (_ : StrictMono k) (_ : ∀ i, x i ≠ 0),
        f = ∑ i, C (x i) * X ^ k i := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    n : Nat
    ⊢ Iff (Eq f.support.card n) (Exists fun k => Exists fun x => Exists fun x_1 => …
  -/
  refine ⟨?_, fun ⟨k, x, hk, hx, hf⟩ => hf.symm ▸ card_support_eq' k x hk.injective hx⟩
  induction n generalizing f with
  | zero => exact fun hf => ⟨0, 0, fun x => x.elim0, fun x => x.elim0, card_support_eq_zero.mp hf⟩
  | succ n hn =>
    intro h
    obtain ⟨k, x, hk, hx, hf⟩ := hn (card_support_eraseLead' h)
    have H : ¬∃ k : Fin n, Fin.castSucc k = Fin.last n := by
      rintro ⟨i, hi⟩
      exact i.castSucc_lt_last.ne hi
    refine
      ⟨Function.extend Fin.castSucc k fun _ => f.natDegree,
        Function.extend Fin.castSucc x fun _ => f.leadingCoeff, ?_, ?_, ?_⟩
    · intro i j hij
      have hi : i ∈ Set.range (Fin.castSucc : Fin n → Fin (n + 1)) := by
        rw [Fin.range_castSucc, Set.mem_def]
        exact lt_of_lt_of_le hij (Nat.lt_succ_iff.mp j.2)
      obtain ⟨i, rfl⟩ := hi
      rw [Fin.strictMono_castSucc.injective.extend_apply]
      by_cases hj : ∃ j₀, Fin.castSucc j₀ = j
      · obtain ⟨j, rfl⟩ := hj
        rwa [Fin.strictMono_castSucc.injective.extend_apply, hk.lt_iff_lt,
          ← Fin.castSucc_lt_castSucc_iff]
      · rw [Function.extend_apply' _ _ _ hj]
        apply lt_natDegree_of_mem_eraseLead_support
        rw [mem_support_iff, hf, finset_sum_coeff]
        rw [sum_eq_single, coeff_C_mul, coeff_X_pow_self, mul_one]
        · exact hx i
        · intro j _ hji
          rw [coeff_C_mul, coeff_X_pow, if_neg (hk.injective.ne hji.symm), mul_zero]
        · exact fun hi => (hi (mem_univ i)).elim
    · intro i
      by_cases hi : ∃ i₀, Fin.castSucc i₀ = i
      · obtain ⟨i, rfl⟩ := hi
        rw [Fin.strictMono_castSucc.injective.extend_apply]
        exact hx i
      · rw [Function.extend_apply' _ _ _ hi, Ne, leadingCoeff_eq_zero, ← card_support_eq_zero, h]
        exact n.succ_ne_zero
    · rw [Fin.sum_univ_castSucc]
      simp only [Fin.strictMono_castSucc.injective.extend_apply]
      rw [← hf, Function.extend_apply', Function.extend_apply', eraseLead_add_C_mul_X_pow]
      all_goals exact H


theorem card_support_eq_one : #f.support = 1 ↔
    ∃ (k : ℕ) (x : R) (_ : x ≠ 0), f = C x * X ^ k := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Iff (Eq f.support.card 1) (Exists fun k => Exists fun x => Exists fun x_1 => …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      h : Eq f.support.card 1
      ⊢ Exists fun k => Exists fun x => Exists fun x_1 => Eq f (HMul.hMul (Polynomia …
    -/
  · obtain ⟨k, x, _, hx, rfl⟩ := card_support_eq.mp h
    /-
      case refine_1.intro.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      k : Fin 1 → Nat
      x : Fin 1 → R
      w✝ : StrictMono k
      hx : ∀ (i : Fin 1), Ne (x i) 0
      h : Eq (Finset.univ.sum fun i => HMul.hMul (Polynomial.C (x i)) (HPow.hPow Pol …
      ⊢ Exists fun k_1 => Exists fun x_1 => Exists fun x_2 => Eq (Finset.univ.sum fu …
    -/
    exact ⟨k 0, x 0, hx 0, Fin.sum_univ_one _⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      ⊢ (Exists fun k => Exists fun x => Exists fun x_1 => Eq f (HMul.hMul (Polynomi …
    -/
  · rintro ⟨k, x, hx, rfl⟩
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      k : Nat
      x : R
      hx : Ne x 0
      ⊢ Eq (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k)).support.card 1
    -/
    rw [support_C_mul_X_pow k hx, card_singleton]
    /-
      🎉 no goals
    -/


theorem card_support_eq_two :
    #f.support = 2 ↔
      ∃ (k m : ℕ) (_ : k < m) (x y : R) (_ : x ≠ 0) (_ : y ≠ 0),
        f = C x * X ^ k + C y * X ^ m := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Iff (Eq f.support.card 2) (Exists fun k => Exists fun m => Exists fun x => E …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      h : Eq f.support.card 2
      ⊢ Exists fun k => Exists fun m => Exists fun x => Exists fun x => Exists fun y …
    -/
  · obtain ⟨k, x, hk, hx, rfl⟩ := card_support_eq.mp h
    /-
      case refine_1.intro.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      k : Fin 2 → Nat
      x : Fin 2 → R
      hk : StrictMono k
      hx : ∀ (i : Fin 2), Ne (x i) 0
      h : Eq (Finset.univ.sum fun i => HMul.hMul (Polynomial.C (x i)) (HPow.hPow Pol …
      ⊢ Exists fun k_1 => Exists fun m => Exists fun x_1 => Exists fun x_2 => Exists …
    -/
    refine ⟨k 0, k 1, hk Nat.zero_lt_one, x 0, x 1, hx 0, hx 1, ?_⟩
    /-
      case refine_1.intro.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      k : Fin 2 → Nat
      x : Fin 2 → R
      hk : StrictMono k
      hx : ∀ (i : Fin 2), Ne (x i) 0
      h : Eq (Finset.univ.sum fun i => HMul.hMul (Polynomial.C (x i)) (HPow.hPow Pol …
      ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (Polynomial.C (x i)) (HPow.hPow Polyn …
    -/
    rw [Fin.sum_univ_castSucc, Fin.sum_univ_one]
    /-
      case refine_1.intro.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      k : Fin 2 → Nat
      x : Fin 2 → R
      hk : StrictMono k
      hx : ∀ (i : Fin 2), Ne (x i) 0
      h : Eq (Finset.univ.sum fun i => HMul.hMul (Polynomial.C (x i)) (HPow.hPow Pol …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C (x (Fin.castSucc 0))) (HPow.hPow Poly …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      ⊢ (Exists fun k => Exists fun m => Exists fun x => Exists fun x => Exists fun  …
    -/
  · rintro ⟨k, m, hkm, x, y, hx, hy, rfl⟩
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      k m : Nat
      hkm : LT.lt k m
      x y : R
      hx : Ne x 0
      hy : Ne y 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k)) (HMul. …
    -/
    exact card_support_binomial hkm.ne hx hy
    /-
      🎉 no goals
    -/


theorem card_support_eq_three :
    #f.support = 3 ↔
      ∃ (k m n : ℕ) (_ : k < m) (_ : m < n) (x y z : R) (_ : x ≠ 0) (_ : y ≠ 0) (_ : z ≠ 0),
        f = C x * X ^ k + C y * X ^ m + C z * X ^ n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Iff (Eq f.support.card 3) (Exists fun k => Exists fun m => Exists fun n => E …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      h : Eq f.support.card 3
      ⊢ Exists fun k => Exists fun m => Exists fun n => Exists fun x => Exists fun x …
    -/
  · obtain ⟨k, x, hk, hx, rfl⟩ := card_support_eq.mp h
    refine
      ⟨k 0, k 1, k 2, hk Nat.zero_lt_one, hk (Nat.lt_succ_self 1), x 0, x 1, x 2, hx 0, hx 1, hx 2,
        ?_⟩
    /-
      case refine_1.intro.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      k : Fin 3 → Nat
      x : Fin 3 → R
      hk : StrictMono k
      hx : ∀ (i : Fin 3), Ne (x i) 0
      h : Eq (Finset.univ.sum fun i => HMul.hMul (Polynomial.C (x i)) (HPow.hPow Pol …
      ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (Polynomial.C (x i)) (HPow.hPow Polyn …
    -/
    rw [Fin.sum_univ_castSucc, Fin.sum_univ_castSucc, Fin.sum_univ_one]
    /-
      case refine_1.intro.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      k : Fin 3 → Nat
      x : Fin 3 → R
      hk : StrictMono k
      hx : ∀ (i : Fin 3), Ne (x i) 0
      h : Eq (Finset.univ.sum fun i => HMul.hMul (Polynomial.C (x i)) (HPow.hPow Pol …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C (x (Fin.castSucc 0).castSu …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      ⊢ (Exists fun k => Exists fun m => Exists fun n => Exists fun x => Exists fun  …
    -/
  · rintro ⟨k, m, n, hkm, hmn, x, y, z, hx, hy, hz, rfl⟩
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      x y z : R
      hx : Ne x 0
      hy : Ne y 0
      hz : Ne z 0
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X …
    -/
    exact card_support_trinomial hkm hmn hx hy hz
    /-
      🎉 no goals
    -/


