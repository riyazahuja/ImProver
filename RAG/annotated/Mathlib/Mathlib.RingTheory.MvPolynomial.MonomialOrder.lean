variable (m) in
/-- the degree of a multivariate polynomial with respect to a monomial ordering -/
def degree {R : Type*} [CommSemiring R] (f : MvPolynomial σ R) : σ →₀ ℕ :=
  m.toSyn.symm (f.support.sup m.toSyn)


variable (m) in
/-- the leading coefficient of a multivariate polynomial with respect to a monomial ordering -/
def lCoeff {R : Type*} [CommSemiring R] (f : MvPolynomial σ R) : R :=
  f.coeff (m.degree f)


@[simp]
theorem degree_zero : m.degree (0 : MvPolynomial σ R) = 0 := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    ⊢ Eq (m.degree 0) 0
  -/
  simp [degree]
  /-
    🎉 no goals
  -/


@[simp]
theorem lCoeff_zero : m.lCoeff (0 : MvPolynomial σ R) = 0 := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    ⊢ Eq (m.lCoeff 0) 0
  -/
  simp [degree, lCoeff]
  /-
    🎉 no goals
  -/


theorem degree_monomial_le {d : σ →₀ ℕ} (c : R) :
    m.degree (monomial d c) ≼[m] d := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    d : Finsupp σ Nat
    c : R
    ⊢ LE.le (m.toSyn (m.degree ((MvPolynomial.monomial d) c))) (m.toSyn d)
  -/
  simp only [degree, AddEquiv.apply_symm_apply]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    d : Finsupp σ Nat
    c : R
    ⊢ LE.le (((MvPolynomial.monomial d) c).support.sup ⇑m.toSyn) (m.toSyn d)
  -/
  apply le_trans (Finset.sup_mono support_monomial_subset)
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    d : Finsupp σ Nat
    c : R
    ⊢ LE.le ((Singleton.singleton d).sup ⇑m.toSyn) (m.toSyn d)
  -/
  simp only [Finset.sup_singleton, le_refl]
  /-
    🎉 no goals
  -/


theorem degree_monomial {d : σ →₀ ℕ} (c : R) [Decidable (c = 0)] :
    m.degree (monomial d c) = if c = 0 then 0 else d := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝¹ : CommSemiring R
    d : Finsupp σ Nat
    c : R
    inst✝ : Decidable (Eq c 0)
    ⊢ Eq (m.degree ((MvPolynomial.monomial d) c)) (ite (Eq c 0) 0 d)
  -/
  simp only [degree, support_monomial]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝¹ : CommSemiring R
    d : Finsupp σ Nat
    c : R
    inst✝ : Decidable (Eq c 0)
    ⊢ Eq (m.toSyn.symm ((ite (Eq c 0) EmptyCollection.emptyCollection (Singleton.s …
  -/
                        /-
                          🎉 no goals
                        -/
  split_ifs with hc <;> simp
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem lCoeff_monomial {d : σ →₀ ℕ} (c : R) :
    m.lCoeff (monomial d c) = c := by
  classical
  simp only [lCoeff, degree_monomial]
  split_ifs with hc <;> simp [hc]


theorem degree_le_iff {f : MvPolynomial σ R} {d : σ →₀ ℕ} :
    m.degree f ≼[m] d ↔ ∀ c ∈ f.support, c ≼[m] d := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    ⊢ Iff (LE.le (m.toSyn (m.degree f)) (m.toSyn d)) (∀ (c : Finsupp σ Nat), Membe …
  -/
  unfold degree
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    ⊢ Iff (LE.le (m.toSyn (m.toSyn.symm (f.support.sup ⇑m.toSyn))) (m.toSyn d)) (∀ …
  -/
  simp only [AddEquiv.apply_symm_apply, Finset.sup_le_iff, mem_support_iff, ne_eq]
  /-
    🎉 no goals
  -/


theorem degree_lt_iff {f : MvPolynomial σ R} {d : σ →₀ ℕ} (hd : 0 ≺[m] d) :
    m.degree f ≺[m] d ↔ ∀ c ∈ f.support, c ≺[m] d := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    hd : LT.lt (m.toSyn 0) (m.toSyn d)
    ⊢ Iff (LT.lt (m.toSyn (m.degree f)) (m.toSyn d)) (∀ (c : Finsupp σ Nat), Membe …
  -/
  simp only [map_zero] at hd
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    hd : LT.lt 0 (m.toSyn d)
    ⊢ Iff (LT.lt (m.toSyn (m.degree f)) (m.toSyn d)) (∀ (c : Finsupp σ Nat), Membe …
  -/
  unfold degree
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    hd : LT.lt 0 (m.toSyn d)
    ⊢ Iff (LT.lt (m.toSyn (m.toSyn.symm (f.support.sup ⇑m.toSyn))) (m.toSyn d)) (∀ …
  -/
  simp only [AddEquiv.apply_symm_apply]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    hd : LT.lt 0 (m.toSyn d)
    ⊢ Iff (LT.lt (f.support.sup ⇑m.toSyn) (m.toSyn d)) (∀ (c : Finsupp σ Nat), Mem …
  -/
  exact Finset.sup_lt_iff hd
  /-
    🎉 no goals
  -/


theorem le_degree {f : MvPolynomial σ R} {d : σ →₀ ℕ} (hd : d ∈ f.support) :
    d ≼[m] m.degree f := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    ⊢ LE.le (m.toSyn d) (m.toSyn (m.degree f))
  -/
  unfold degree
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    ⊢ LE.le (m.toSyn d) (m.toSyn (m.toSyn.symm (f.support.sup ⇑m.toSyn)))
  -/
  simp only [AddEquiv.apply_symm_apply, Finset.le_sup hd]
  /-
    🎉 no goals
  -/


theorem coeff_eq_zero_of_lt {f : MvPolynomial σ R} {d : σ →₀ ℕ} (hd : m.degree f ≺[m] d) :
    f.coeff d = 0 := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    hd : LT.lt (m.toSyn (m.degree f)) (m.toSyn d)
    ⊢ Eq (MvPolynomial.coeff d f) 0
  -/
  rw [← not_le] at hd
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    hd : Not (LE.le (m.toSyn d) (m.toSyn (m.degree f)))
    ⊢ Eq (MvPolynomial.coeff d f) 0
  -/
  by_contra hf
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    hd : Not (LE.le (m.toSyn d) (m.toSyn (m.degree f)))
    hf : Not (Eq (MvPolynomial.coeff d f) 0)
    ⊢ False
  -/
  apply hd (m.le_degree (mem_support_iff.mpr hf))
  /-
    🎉 no goals
  -/

theorem lCoeff_ne_zero_iff {f : MvPolynomial σ R} :
    m.lCoeff f ≠ 0 ↔ f ≠ 0 := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    ⊢ Iff (Ne (m.lCoeff f) 0) (Ne f 0)
  -/
  constructor
    /-
      case mp
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f : MvPolynomial σ R
      ⊢ Ne (m.lCoeff f) 0 → Ne f 0
    -/
  · rw [not_imp_not]
    /-
      case mp
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f : MvPolynomial σ R
      ⊢ Eq f 0 → Eq (m.lCoeff f) 0
    -/
    intro hf
    /-
      case mp
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f : MvPolynomial σ R
      hf : Eq f 0
      ⊢ Eq (m.lCoeff f) 0
    -/
    rw [hf, lCoeff_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f : MvPolynomial σ R
      ⊢ Ne f 0 → Ne (m.lCoeff f) 0
    -/
  · intro hf
    /-
      case mpr
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f : MvPolynomial σ R
      hf : Ne f 0
      ⊢ Ne (m.lCoeff f) 0
    -/
    rw [← support_nonempty] at hf
    /-
      case mpr
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f : MvPolynomial σ R
      hf : f.support.Nonempty
      ⊢ Ne (m.lCoeff f) 0
    -/
    rw [lCoeff, ← mem_support_iff, degree]
    suffices f.support.sup m.toSyn ∈ m.toSyn '' f.support by
      obtain ⟨d, hd, hd'⟩ := this
      rw [← hd', AddEquiv.symm_apply_apply]
      exact hd
    /-
      case mpr
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f : MvPolynomial σ R
      hf : f.support.Nonempty
      ⊢ Membership.mem (Set.image ⇑m.toSyn ↑f.support) (f.support.sup ⇑m.toSyn)
    -/
    exact Finset.sup_mem_of_nonempty hf
    /-
      🎉 no goals
    -/


@[simp]
theorem lCoeff_eq_zero_iff {f : MvPolynomial σ R} :
    lCoeff m f = 0 ↔ f = 0 := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    ⊢ Iff (Eq (m.lCoeff f) 0) (Eq f 0)
  -/
  simp only [← not_iff_not, lCoeff_ne_zero_iff]
  /-
    🎉 no goals
  -/


theorem coeff_degree_ne_zero_iff {f : MvPolynomial σ R} :
    f.coeff (m.degree f) ≠ 0 ↔ f ≠ 0 :=
  m.lCoeff_ne_zero_iff


@[simp]
theorem coeff_degree_eq_zero_iff {f : MvPolynomial σ R} :
    f.coeff (m.degree f) = 0 ↔ f = 0 :=
  m.lCoeff_eq_zero_iff


theorem degree_eq_zero_iff_totalDegree_eq_zero {f : MvPolynomial σ R} :
    m.degree f = 0 ↔ f.totalDegree = 0 := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    ⊢ Iff (Eq (m.degree f) 0) (Eq f.totalDegree 0)
  -/
  rw [← m.toSyn.injective.eq_iff]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    ⊢ Iff (Eq (m.toSyn (m.degree f)) (m.toSyn 0)) (Eq f.totalDegree 0)
  -/
  rw [map_zero, ← m.bot_eq_zero, eq_bot_iff, m.bot_eq_zero, ← m.toSyn.map_zero]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    ⊢ Iff (LE.le (m.toSyn (m.degree f)) (m.toSyn 0)) (Eq f.totalDegree 0)
  -/
  rw [degree_le_iff]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    ⊢ Iff (∀ (c : Finsupp σ Nat), Membership.mem f.support c → LE.le (m.toSyn c) ( …
  -/
  rw [totalDegree_eq_zero_iff]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    ⊢ Iff (∀ (c : Finsupp σ Nat), Membership.mem f.support c → LE.le (m.toSyn c) ( …
  -/
  apply forall_congr'
  /-
    case h
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    ⊢ ∀ (a : Finsupp σ Nat), Iff (Membership.mem f.support a → LE.le (m.toSyn a) ( …
  -/
  intro d
  /-
    case h
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    ⊢ Iff (Membership.mem f.support d → LE.le (m.toSyn d) (m.toSyn 0)) (Membership …
  -/
  apply imp_congr (rfl.to_iff)
  /-
    case h
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    ⊢ Iff (LE.le (m.toSyn d) (m.toSyn 0)) (∀ (x : σ), Eq (d x) 0)
  -/
  rw [map_zero, ← m.bot_eq_zero, ← eq_bot_iff, m.bot_eq_zero]
  /-
    case h
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    ⊢ Iff (Eq (m.toSyn d) 0) (∀ (x : σ), Eq (d x) 0)
  -/
  simp only [EmbeddingLike.map_eq_zero_iff]
  /-
    case h
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    d : Finsupp σ Nat
    ⊢ Iff (Eq d 0) (∀ (x : σ), Eq (d x) 0)
  -/
  exact Finsupp.ext_iff
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_C (r : R) :
    m.degree (C r) = 0 := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    ⊢ Eq (m.degree (MvPolynomial.C r)) 0
  -/
  rw [degree_eq_zero_iff_totalDegree_eq_zero, totalDegree_C]
  /-
    🎉 no goals
  -/


theorem degree_add_le {f g : MvPolynomial σ R} :
    m.toSyn (m.degree (f + g)) ≤ m.toSyn (m.degree f) ⊔ m.toSyn (m.degree g) := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    ⊢ LE.le (m.toSyn (m.degree (HAdd.hAdd f g))) (Max.max (m.toSyn (m.degree f)) ( …
  -/
  conv_rhs => rw [← m.toSyn.apply_symm_apply (_ ⊔ _)]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    ⊢ LE.le (m.toSyn (m.degree (HAdd.hAdd f g))) (m.toSyn (m.toSyn.symm (Max.max ( …
  -/
  rw [degree_le_iff]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    ⊢ ∀ (c : Finsupp σ Nat), Membership.mem (HAdd.hAdd f g).support c → LE.le (m.t …
  -/
  simp only [AddEquiv.apply_symm_apply, le_sup_iff]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    ⊢ ∀ (c : Finsupp σ Nat), Membership.mem (HAdd.hAdd f g).support c → Or (LE.le  …
  -/
  intro b hb
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    b : Finsupp σ Nat
    hb : Membership.mem (HAdd.hAdd f g).support b
    ⊢ Or (LE.le (m.toSyn b) (m.toSyn (m.degree f))) (LE.le (m.toSyn b) (m.toSyn (m …
  -/
  by_cases hf : b ∈ f.support
    /-
      case pos
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      b : Finsupp σ Nat
      hb : Membership.mem (HAdd.hAdd f g).support b
      hf : Membership.mem f.support b
      ⊢ Or (LE.le (m.toSyn b) (m.toSyn (m.degree f))) (LE.le (m.toSyn b) (m.toSyn (m …
    -/
  · left
    /-
      case pos.h
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      b : Finsupp σ Nat
      hb : Membership.mem (HAdd.hAdd f g).support b
      hf : Membership.mem f.support b
      ⊢ LE.le (m.toSyn b) (m.toSyn (m.degree f))
    -/
    exact m.le_degree hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      b : Finsupp σ Nat
      hb : Membership.mem (HAdd.hAdd f g).support b
      hf : Not (Membership.mem f.support b)
      ⊢ Or (LE.le (m.toSyn b) (m.toSyn (m.degree f))) (LE.le (m.toSyn b) (m.toSyn (m …
    -/
  · right
    /-
      case neg.h
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      b : Finsupp σ Nat
      hb : Membership.mem (HAdd.hAdd f g).support b
      hf : Not (Membership.mem f.support b)
      ⊢ LE.le (m.toSyn b) (m.toSyn (m.degree g))
    -/
    apply m.le_degree
    /-
      case neg.h
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      b : Finsupp σ Nat
      hb : Membership.mem (HAdd.hAdd f g).support b
      hf : Not (Membership.mem f.support b)
      ⊢ Membership.mem g.support b
    -/
    simp only [not_mem_support_iff] at hf
    /-
      case neg.h
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      b : Finsupp σ Nat
      hb : Membership.mem (HAdd.hAdd f g).support b
      hf : Eq (MvPolynomial.coeff b f) 0
      ⊢ Membership.mem g.support b
    -/
    simpa only [mem_support_iff, coeff_add, hf, zero_add] using hb
    /-
      🎉 no goals
    -/


theorem degree_add_of_lt {f g : MvPolynomial σ R} (h : m.degree g ≺[m] m.degree f) :
    m.degree (f + g) = m.degree f := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
    ⊢ Eq (m.degree (HAdd.hAdd f g)) (m.degree f)
  -/
  apply m.toSyn.injective
  /-
    case a
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
    ⊢ Eq (m.toSyn (m.degree (HAdd.hAdd f g))) (m.toSyn (m.degree f))
  -/
  apply le_antisymm
    /-
      case a.a
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
      ⊢ LE.le (m.toSyn (m.degree (HAdd.hAdd f g))) (m.toSyn (m.degree f))
    -/
  · apply le_trans degree_add_le
    /-
      case a.a
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
      ⊢ LE.le (Max.max (m.toSyn (m.degree f)) (m.toSyn (m.degree g))) (m.toSyn (m.de …
    -/
    simp only [sup_le_iff, le_refl, true_and, le_of_lt h]
    /-
      🎉 no goals
    -/
    /-
      case a.a
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
      ⊢ LE.le (m.toSyn (m.degree f)) (m.toSyn (m.degree (HAdd.hAdd f g)))
    -/
  · apply le_degree
    /-
      case a.a.hd
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
      ⊢ Membership.mem (HAdd.hAdd f g).support (m.degree f)
    -/
    rw [mem_support_iff, coeff_add, m.coeff_eq_zero_of_lt h, add_zero, ← lCoeff, lCoeff_ne_zero_iff]
    /-
      case a.a.hd
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
      ⊢ Ne f 0
    -/
    intro hf
    /-
      case a.a.hd
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
      hf : Eq f 0
      ⊢ False
    -/
    rw [← not_le, hf] at h
    /-
      case a.a.hd
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : Not (LE.le (m.toSyn (m.degree 0)) (m.toSyn (m.degree g)))
      hf : Eq f 0
      ⊢ False
    -/
    apply h
    /-
      case a.a.hd
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : Not (LE.le (m.toSyn (m.degree 0)) (m.toSyn (m.degree g)))
      hf : Eq f 0
      ⊢ LE.le (m.toSyn (m.degree 0)) (m.toSyn (m.degree g))
    -/
    simp only [degree_zero, map_zero]
    /-
      case a.a.hd
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : Not (LE.le (m.toSyn (m.degree 0)) (m.toSyn (m.degree g)))
      hf : Eq f 0
      ⊢ LE.le 0 (m.toSyn (m.degree g))
    -/
    apply bot_le
    /-
      🎉 no goals
    -/


theorem lCoeff_add_of_lt {f g : MvPolynomial σ R} (h : m.degree g ≺[m] m.degree f) :
    m.lCoeff (f + g) = m.lCoeff f := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
    ⊢ Eq (m.lCoeff (HAdd.hAdd f g)) (m.lCoeff f)
  -/
  simp only [lCoeff, m.degree_add_of_lt h, coeff_add, coeff_eq_zero_of_lt h, add_zero]
  /-
    🎉 no goals
  -/


theorem degree_add_of_ne {f g : MvPolynomial σ R}
    (h : m.degree f ≠ m.degree g) :
    m.toSyn (m.degree (f + g)) = m.toSyn (m.degree f) ⊔ m.toSyn (m.degree g) := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    h : Ne (m.degree f) (m.degree g)
    ⊢ Eq (m.toSyn (m.degree (HAdd.hAdd f g))) (Max.max (m.toSyn (m.degree f)) (m.t …
  -/
  by_cases h' : m.degree g ≺[m] m.degree f
    /-
      case pos
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : Ne (m.degree f) (m.degree g)
      h' : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
      ⊢ Eq (m.toSyn (m.degree (HAdd.hAdd f g))) (Max.max (m.toSyn (m.degree f)) (m.t …
    -/
  · simp [degree_add_of_lt h', left_eq_sup, le_of_lt h']
    /-
      🎉 no goals
    -/
    /-
      case neg
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : Ne (m.degree f) (m.degree g)
      h' : Not (LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f)))
      ⊢ Eq (m.toSyn (m.degree (HAdd.hAdd f g))) (Max.max (m.toSyn (m.degree f)) (m.t …
    -/
  · rw [not_lt, le_iff_eq_or_lt, Classical.or_iff_not_imp_left, EmbeddingLike.apply_eq_iff_eq] at h'
    /-
      case neg
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : Ne (m.degree f) (m.degree g)
      h' : Not (Eq (m.degree f) (m.degree g)) → LT.lt (m.toSyn (m.degree f)) (m.toSy …
      ⊢ Eq (m.toSyn (m.degree (HAdd.hAdd f g))) (Max.max (m.toSyn (m.degree f)) (m.t …
    -/
    rw [add_comm, degree_add_of_lt (h' h), right_eq_sup]
    /-
      case neg
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      f g : MvPolynomial σ R
      h : Ne (m.degree f) (m.degree g)
      h' : Not (Eq (m.degree f) (m.degree g)) → LT.lt (m.toSyn (m.degree f)) (m.toSy …
      ⊢ LE.le (m.toSyn (m.degree f)) (m.toSyn (m.degree g))
    -/
    simp only [le_of_lt (h' h)]
    /-
      🎉 no goals
    -/


theorem degree_mul_le {f g : MvPolynomial σ R} :
    m.degree (f * g) ≼[m] m.degree f + m.degree g := by
  classical
  rw [degree_le_iff]
  intro c
  rw [← not_lt, mem_support_iff, not_imp_not]
  intro hc
  rw [coeff_mul]
  apply Finset.sum_eq_zero
  rintro ⟨d, e⟩ hde
  simp only [Finset.mem_antidiagonal] at hde
  dsimp only
  by_cases hd : m.degree f ≺[m] d
  · rw [m.coeff_eq_zero_of_lt hd, zero_mul]
  · suffices m.degree g ≺[m] e by
      rw [m.coeff_eq_zero_of_lt this, mul_zero]
    simp only [not_lt] at hd
    apply lt_of_add_lt_add_left (a := m.toSyn d)
    simp only [← map_add, hde]
    apply lt_of_le_of_lt _ hc
    simp only [map_add]
    exact add_le_add_right hd _


/-- Multiplicativity of leading coefficients -/
theorem coeff_mul_of_degree_add {f g : MvPolynomial σ R} :
    (f * g).coeff (m.degree f + m.degree g) = m.lCoeff f * m.lCoeff g := by
  classical
  rw [coeff_mul]
  rw [Finset.sum_eq_single (m.degree f, m.degree g)]
  · rfl
  · rintro ⟨c, d⟩ hcd h
    simp only [Finset.mem_antidiagonal] at hcd
    by_cases hf : m.degree f ≺[m] c
    · rw [m.coeff_eq_zero_of_lt hf, zero_mul]
    · suffices m.degree g ≺[m] d by
        rw [coeff_eq_zero_of_lt this, mul_zero]
      apply lt_of_add_lt_add_left (a := m.toSyn c)
      simp only [← map_add, hcd]
      simp only [map_add]
      rw [← not_le]
      intro h'; apply hf
      simp only [le_iff_eq_or_lt] at h'
      cases h' with
      | inl h' =>
        simp only [← map_add, EmbeddingLike.apply_eq_iff_eq, add_left_inj] at h'
        exfalso
        apply h
        simp only [h', Prod.mk.injEq, true_and]
        simpa [h'] using hcd
      | inr h' =>
        exact lt_of_add_lt_add_right h'
  · simp


/-- Multiplicativity of leading coefficients -/
theorem degree_mul_of_isRegular_left {f g : MvPolynomial σ R}
    (hf : IsRegular (m.lCoeff f)) (hg : g ≠ 0) :
    m.degree (f * g) = m.degree f + m.degree g := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    hf : IsRegular (m.lCoeff f)
    hg : Ne g 0
    ⊢ Eq (m.degree (HMul.hMul f g)) (HAdd.hAdd (m.degree f) (m.degree g))
  -/
  apply m.toSyn.injective
  /-
    case a
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    hf : IsRegular (m.lCoeff f)
    hg : Ne g 0
    ⊢ Eq (m.toSyn (m.degree (HMul.hMul f g))) (m.toSyn (HAdd.hAdd (m.degree f) (m. …
  -/
  apply le_antisymm degree_mul_le
  /-
    case a
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    hf : IsRegular (m.lCoeff f)
    hg : Ne g 0
    ⊢ LE.le (m.toSyn (HAdd.hAdd (m.degree f) (m.degree g))) (m.toSyn (m.degree (HM …
  -/
  apply le_degree
  /-
    case a.hd
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    hf : IsRegular (m.lCoeff f)
    hg : Ne g 0
    ⊢ Membership.mem (HMul.hMul f g).support (HAdd.hAdd (m.degree f) (m.degree g))
  -/
  rw [mem_support_iff, coeff_mul_of_degree_add]
  simp only [ne_eq, hf, IsRegular.left, IsLeftRegular.mul_left_eq_zero_iff,
    lCoeff_eq_zero_iff]
  /-
    case a.hd
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    hf : IsRegular (m.lCoeff f)
    hg : Ne g 0
    ⊢ Not (Eq g 0)
  -/
  exact hg
  /-
    🎉 no goals
  -/


/-- Multiplicativity of leading coefficients -/
theorem lCoeff_mul_of_isRegular_left {f g : MvPolynomial σ R}
    (hf : IsRegular (m.lCoeff f)) (hg : g ≠ 0) :
    m.lCoeff (f * g) = m.lCoeff f * m.lCoeff g := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    hf : IsRegular (m.lCoeff f)
    hg : Ne g 0
    ⊢ Eq (m.lCoeff (HMul.hMul f g)) (HMul.hMul (m.lCoeff f) (m.lCoeff g))
  -/
  simp only [lCoeff, degree_mul_of_isRegular_left hf hg, coeff_mul_of_degree_add]
  /-
    🎉 no goals
  -/


/-- Multiplicativity of leading coefficients -/
theorem degree_mul_of_isRegular_right {f g : MvPolynomial σ R}
    (hf : f ≠ 0) (hg : IsRegular (m.lCoeff g)) :
    m.degree (f * g) = m.degree f + m.degree g := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    hf : Ne f 0
    hg : IsRegular (m.lCoeff g)
    ⊢ Eq (m.degree (HMul.hMul f g)) (HAdd.hAdd (m.degree f) (m.degree g))
  -/
  rw [mul_comm, m.degree_mul_of_isRegular_left hg hf, add_comm]
  /-
    🎉 no goals
  -/


/-- Multiplicativity of leading coefficients -/
theorem lCoeff_mul_of_isRegular_right {f g : MvPolynomial σ R}
    (hf : f ≠ 0) (hg : IsRegular (m.lCoeff g)) :
    m.lCoeff (f * g) = m.lCoeff f * m.lCoeff g := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f g : MvPolynomial σ R
    hf : Ne f 0
    hg : IsRegular (m.lCoeff g)
    ⊢ Eq (m.lCoeff (HMul.hMul f g)) (HMul.hMul (m.lCoeff f) (m.lCoeff g))
  -/
  simp only [lCoeff, degree_mul_of_isRegular_right hf hg, coeff_mul_of_degree_add]
  /-
    🎉 no goals
  -/


/-- Degree of product -/
theorem degree_mul [IsDomain R] {f g : MvPolynomial σ R} (hf : f ≠ 0) (hg : g ≠ 0) :
    m.degree (f * g) = m.degree f + m.degree g :=
  degree_mul_of_isRegular_left (isRegular_of_ne_zero (lCoeff_ne_zero_iff.mpr hf)) hg


/-- Degree of of product -/
theorem degree_mul_of_nonzero_mul [IsDomain R] {f g : MvPolynomial σ R} (hfg : f * g ≠ 0) :
    m.degree (f * g) = m.degree f + m.degree g :=
  degree_mul (left_ne_zero_of_mul hfg) (right_ne_zero_of_mul hfg)


/-- Multiplicativity of leading coefficients -/
theorem lCoeff_mul [IsDomain R] {f g : MvPolynomial σ R}
    (hf : f ≠ 0) (hg : g ≠ 0) :
    m.lCoeff (f * g) = m.lCoeff f * m.lCoeff g := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : IsDomain R
    f g : MvPolynomial σ R
    hf : Ne f 0
    hg : Ne g 0
    ⊢ Eq (m.lCoeff (HMul.hMul f g)) (HMul.hMul (m.lCoeff f) (m.lCoeff g))
  -/
  rw [lCoeff, degree_mul hf hg, ← coeff_mul_of_degree_add]
  /-
    🎉 no goals
  -/


theorem degree_smul_le {r : R} {f : MvPolynomial σ R} :
    m.degree (r • f) ≼[m] m.degree f := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    f : MvPolynomial σ R
    ⊢ LE.le (m.toSyn (m.degree (HSMul.hSMul r f))) (m.toSyn (m.degree f))
  -/
  rw [smul_eq_C_mul]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    f : MvPolynomial σ R
    ⊢ LE.le (m.toSyn (m.degree (HMul.hMul (MvPolynomial.C r) f))) (m.toSyn (m.degr …
  -/
  apply le_of_le_of_eq degree_mul_le
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    f : MvPolynomial σ R
    ⊢ Eq (m.toSyn (HAdd.hAdd (m.degree (MvPolynomial.C r)) (m.degree f))) (m.toSyn …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem degree_smul {r : R} (hr : IsRegular r) {f : MvPolynomial σ R} :
    m.degree (r • f) = m.degree f := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    hr : IsRegular r
    f : MvPolynomial σ R
    ⊢ Eq (m.degree (HSMul.hSMul r f)) (m.degree f)
  -/
  by_cases hf : f = 0
    /-
      case pos
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommSemiring R
      r : R
      hr : IsRegular r
      f : MvPolynomial σ R
      hf : Eq f 0
      ⊢ Eq (m.degree (HSMul.hSMul r f)) (m.degree f)
    -/
  · simp [hf]
    /-
      🎉 no goals
    -/
  /-
    case neg
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    hr : IsRegular r
    f : MvPolynomial σ R
    hf : Not (Eq f 0)
    ⊢ Eq (m.degree (HSMul.hSMul r f)) (m.degree f)
  -/
  apply m.toSyn.injective
  /-
    case neg.a
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    hr : IsRegular r
    f : MvPolynomial σ R
    hf : Not (Eq f 0)
    ⊢ Eq (m.toSyn (m.degree (HSMul.hSMul r f))) (m.toSyn (m.degree f))
  -/
  apply le_antisymm degree_smul_le
  /-
    case neg.a
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    hr : IsRegular r
    f : MvPolynomial σ R
    hf : Not (Eq f 0)
    ⊢ LE.le (m.toSyn (m.degree f)) (m.toSyn (m.degree (HSMul.hSMul r f)))
  -/
  apply le_degree
  /-
    case neg.a.hd
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    hr : IsRegular r
    f : MvPolynomial σ R
    hf : Not (Eq f 0)
    ⊢ Membership.mem (HSMul.hSMul r f).support (m.degree f)
  -/
  simp only [mem_support_iff, smul_eq_C_mul]
  /-
    case neg.a.hd
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    hr : IsRegular r
    f : MvPolynomial σ R
    hf : Not (Eq f 0)
    ⊢ Ne (MvPolynomial.coeff (m.degree f) (HMul.hMul (MvPolynomial.C r) f)) 0
  -/
  rw [← zero_add (degree m f), ← degree_C r, coeff_mul_of_degree_add]
  /-
    case neg.a.hd
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    hr : IsRegular r
    f : MvPolynomial σ R
    hf : Not (Eq f 0)
    ⊢ Ne (HMul.hMul (m.lCoeff (MvPolynomial.C r)) (m.lCoeff f)) 0
  -/
  simp [lCoeff, hr.left.mul_left_eq_zero_iff, hf]
  /-
    🎉 no goals
  -/


theorem eq_C_of_degree_eq_zero {f : MvPolynomial σ R} (hf : m.degree f = 0) :
    f = C (m.lCoeff f) := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    hf : Eq (m.degree f) 0
    ⊢ Eq f (MvPolynomial.C (m.lCoeff f))
  -/
  ext d
  /-
    case a
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommSemiring R
    f : MvPolynomial σ R
    hf : Eq (m.degree f) 0
    d : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff d f) (MvPolynomial.coeff d (MvPolynomial.C (m.lCoeff  …
  -/
  simp only [lCoeff, hf]
  classical
  by_cases hd : d = 0
  · simp [hd]
  · rw [coeff_C, if_neg (Ne.symm hd)]
    apply coeff_eq_zero_of_lt (m := m)
    rw [hf, map_zero, lt_iff_le_and_ne, ne_eq, eq_comm, EmbeddingLike.map_eq_zero_iff]
    exact ⟨bot_le, hd⟩


@[simp]
theorem degree_neg {f : MvPolynomial σ R} :
    m.degree (-f) = m.degree f := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f : MvPolynomial σ R
    ⊢ Eq (m.degree (Neg.neg f)) (m.degree f)
  -/
  unfold degree
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f : MvPolynomial σ R
    ⊢ Eq (m.toSyn.symm ((Neg.neg f).support.sup ⇑m.toSyn)) (m.toSyn.symm (f.suppor …
  -/
  rw [support_neg]
  /-
    🎉 no goals
  -/


theorem degree_sub_le {f g : MvPolynomial σ R} :
    m.toSyn (m.degree (f - g)) ≤ m.toSyn (m.degree f) ⊔ m.toSyn (m.degree g) := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f g : MvPolynomial σ R
    ⊢ LE.le (m.toSyn (m.degree (HSub.hSub f g))) (Max.max (m.toSyn (m.degree f)) ( …
  -/
  rw [sub_eq_add_neg]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f g : MvPolynomial σ R
    ⊢ LE.le (m.toSyn (m.degree (HAdd.hAdd f (Neg.neg g)))) (Max.max (m.toSyn (m.de …
  -/
  apply le_of_le_of_eq m.degree_add_le
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f g : MvPolynomial σ R
    ⊢ Eq (Max.max (m.toSyn (m.degree f)) (m.toSyn (m.degree (Neg.neg g)))) (Max.ma …
  -/
  rw [degree_neg]
  /-
    🎉 no goals
  -/


theorem degree_sub_of_lt {f g : MvPolynomial σ R} (h : m.degree g ≺[m] m.degree f) :
    m.degree (f - g) = m.degree f := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f g : MvPolynomial σ R
    h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
    ⊢ Eq (m.degree (HSub.hSub f g)) (m.degree f)
  -/
  rw [sub_eq_add_neg]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f g : MvPolynomial σ R
    h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
    ⊢ Eq (m.degree (HAdd.hAdd f (Neg.neg g))) (m.degree f)
  -/
  apply degree_add_of_lt
  /-
    case h
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f g : MvPolynomial σ R
    h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
    ⊢ LT.lt (m.toSyn (m.degree (Neg.neg g))) (m.toSyn (m.degree f))
  -/
  simp only [degree_neg, h]
  /-
    🎉 no goals
  -/


theorem lCoeff_sub_of_lt {f g : MvPolynomial σ R} (h : m.degree g ≺[m] m.degree f) :
    m.lCoeff (f - g) = m.lCoeff f := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f g : MvPolynomial σ R
    h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
    ⊢ Eq (m.lCoeff (HSub.hSub f g)) (m.lCoeff f)
  -/
  rw [sub_eq_add_neg]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f g : MvPolynomial σ R
    h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
    ⊢ Eq (m.lCoeff (HAdd.hAdd f (Neg.neg g))) (m.lCoeff f)
  -/
  apply lCoeff_add_of_lt
  /-
    case h
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f g : MvPolynomial σ R
    h : LT.lt (m.toSyn (m.degree g)) (m.toSyn (m.degree f))
    ⊢ LT.lt (m.toSyn (m.degree (Neg.neg g))) (m.toSyn (m.degree f))
  -/
  simp only [degree_neg, h]
  /-
    🎉 no goals
  -/


theorem lCoeff_is_unit_iff {f : MvPolynomial σ R} :
    IsUnit (m.lCoeff f) ↔ f ≠ 0 := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : Field R
    f : MvPolynomial σ R
    ⊢ Iff (IsUnit (m.lCoeff f)) (Ne f 0)
  -/
  simp only [isUnit_iff_ne_zero, ne_eq, lCoeff_eq_zero_iff]
  /-
    🎉 no goals
  -/


