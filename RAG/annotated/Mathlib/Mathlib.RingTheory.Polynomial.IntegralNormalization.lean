/-- If `p : R[X]` is a nonzero polynomial with root `z`, `integralNormalization p` is
a monic polynomial with root `leadingCoeff f * z`.

Moreover, `integralNormalization 0 = 0`.
-/
noncomputable def integralNormalization (p : R[X]) : R[X] :=
  p.sum fun i a ↦
    monomial i (if p.degree = i then 1 else a * p.leadingCoeff ^ (p.natDegree - 1 - i))


@[simp]
theorem integralNormalization_zero : integralNormalization (0 : R[X]) = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Eq (Polynomial.integralNormalization 0) 0
  -/
  simp [integralNormalization]
  /-
    🎉 no goals
  -/


@[simp]
theorem integralNormalization_C {x : R} (hx : x ≠ 0) : integralNormalization (C x) = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    x : R
    hx : Ne x 0
    ⊢ Eq (Polynomial.C x).integralNormalization 1
  -/
  simp [integralNormalization, sum_def, support_C hx, degree_C hx]
  /-
    🎉 no goals
  -/


theorem integralNormalization_coeff {i : ℕ} :
    (integralNormalization p).coeff i =
      if p.degree = i then 1 else coeff p i * p.leadingCoeff ^ (p.natDegree - 1 - i) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    i : Nat
    ⊢ Eq (p.integralNormalization.coeff i) (ite (Eq p.degree ↑i) 1 (HMul.hMul (p.c …
  -/
  have : p.coeff i = 0 → p.degree ≠ i := fun hc hd => coeff_ne_zero_of_eq_degree hd hc
  simp +contextual [sum_def, integralNormalization, coeff_monomial, this,
    mem_support_iff]


theorem support_integralNormalization_subset :
    (integralNormalization p).support ⊆ p.support := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ HasSubset.Subset p.integralNormalization.support p.support
  -/
  intro
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    a✝ : Nat
    ⊢ Membership.mem p.integralNormalization.support a✝ → Membership.mem p.support …
  -/
  simp +contextual [sum_def, integralNormalization, coeff_monomial, mem_support_iff]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias integralNormalization_support := support_integralNormalization_subset


theorem integralNormalization_coeff_degree {i : ℕ} (hi : p.degree = i) :
                                                /-
                                                  R : Type u
                                                  inst✝ : Semiring R
                                                  p : Polynomial R
                                                  i : Nat
                                                  hi : Eq p.degree ↑i
                                                  ⊢ Eq (p.integralNormalization.coeff i) 1
                                                -/
    (integralNormalization p).coeff i = 1 := by rw [integralNormalization_coeff, if_pos hi]
                                                /-
                                                  🎉 no goals
                                                -/


theorem integralNormalization_coeff_natDegree (hp : p ≠ 0) :
    (integralNormalization p).coeff (natDegree p) = 1 :=
  integralNormalization_coeff_degree (degree_eq_natDegree hp)


theorem integralNormalization_coeff_degree_ne {i : ℕ} (hi : p.degree ≠ i) :
    coeff (integralNormalization p) i = coeff p i * p.leadingCoeff ^ (p.natDegree - 1 - i) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    i : Nat
    hi : Ne p.degree ↑i
    ⊢ Eq (p.integralNormalization.coeff i) (HMul.hMul (p.coeff i) (HPow.hPow p.lea …
  -/
  rw [integralNormalization_coeff, if_neg hi]
  /-
    🎉 no goals
  -/


theorem integralNormalization_coeff_ne_natDegree {i : ℕ} (hi : i ≠ natDegree p) :
    coeff (integralNormalization p) i = coeff p i * p.leadingCoeff ^ (p.natDegree - 1 - i) :=
  integralNormalization_coeff_degree_ne (degree_ne_of_natDegree_ne hi.symm)


theorem monic_integralNormalization (hp : p ≠ 0) : Monic (integralNormalization p) :=
  monic_of_degree_le p.natDegree
    (Finset.sup_le fun i h =>
      WithBot.coe_le_coe.2 <| le_natDegree_of_mem_supp i <| support_integralNormalization_subset h)
    (integralNormalization_coeff_natDegree hp)


theorem integralNormalization_coeff_mul_leadingCoeff_pow (i : ℕ) (hp : 1 ≤ natDegree p) :
    (integralNormalization p).coeff i * p.leadingCoeff ^ i =
      p.coeff i * p.leadingCoeff ^ (p.natDegree - 1) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    i : Nat
    hp : LE.le 1 p.natDegree
    ⊢ Eq (HMul.hMul (p.integralNormalization.coeff i) (HPow.hPow p.leadingCoeff i) …
  -/
  rw [integralNormalization_coeff]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    i : Nat
    hp : LE.le 1 p.natDegree
    ⊢ Eq (HMul.hMul (ite (Eq p.degree ↑i) 1 (HMul.hMul (p.coeff i) (HPow.hPow p.le …
  -/
  split_ifs with h
  · simp [natDegree_eq_of_degree_eq_some h, leadingCoeff,
      ← pow_succ', tsub_add_cancel_of_le (natDegree_eq_of_degree_eq_some h ▸ hp)]
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      i : Nat
      hp : LE.le 1 p.natDegree
      h : Not (Eq p.degree ↑i)
      ⊢ Eq (HMul.hMul (HMul.hMul (p.coeff i) (HPow.hPow p.leadingCoeff (HSub.hSub (H …
    -/
  · simp only [mul_assoc, ← pow_add]
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      i : Nat
      hp : LE.le 1 p.natDegree
      h : Not (Eq p.degree ↑i)
      ⊢ Eq (HMul.hMul (p.coeff i) (HPow.hPow p.leadingCoeff (HAdd.hAdd (HSub.hSub (H …
    -/
    by_cases h' : i < p.degree
      /-
        case pos
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        i : Nat
        hp : LE.le 1 p.natDegree
        h : Not (Eq p.degree ↑i)
        h' : LT.lt (↑i) p.degree
        ⊢ Eq (HMul.hMul (p.coeff i) (HPow.hPow p.leadingCoeff (HAdd.hAdd (HSub.hSub (H …
      -/
    · rw [tsub_add_cancel_of_le]
      /-
        case pos
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        i : Nat
        hp : LE.le 1 p.natDegree
        h : Not (Eq p.degree ↑i)
        h' : LT.lt (↑i) p.degree
        ⊢ LE.le i (HSub.hSub p.natDegree 1)
      -/
      rw [le_tsub_iff_right hp, Nat.succ_le_iff]
      /-
        case pos
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        i : Nat
        hp : LE.le 1 p.natDegree
        h : Not (Eq p.degree ↑i)
        h' : LT.lt (↑i) p.degree
        ⊢ LT.lt i p.natDegree
      -/
      exact coe_lt_degree.mp h'
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        i : Nat
        hp : LE.le 1 p.natDegree
        h : Not (Eq p.degree ↑i)
        h' : Not (LT.lt (↑i) p.degree)
        ⊢ Eq (HMul.hMul (p.coeff i) (HPow.hPow p.leadingCoeff (HAdd.hAdd (HSub.hSub (H …
      -/
    · simp [coeff_eq_zero_of_degree_lt (lt_of_le_of_ne (le_of_not_gt h') h)]
      /-
        🎉 no goals
      -/


theorem integralNormalization_mul_C_leadingCoeff (p : R[X]) :
    integralNormalization p * C p.leadingCoeff = scaleRoots p p.leadingCoeff := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (HMul.hMul p.integralNormalization (Polynomial.C p.leadingCoeff)) (p.scal …
  -/
  ext i
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    i : Nat
    ⊢ Eq ((HMul.hMul p.integralNormalization (Polynomial.C p.leadingCoeff)).coeff  …
  -/
  rw [coeff_mul_C, integralNormalization_coeff]
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    i : Nat
    ⊢ Eq (HMul.hMul (ite (Eq p.degree ↑i) 1 (HMul.hMul (p.coeff i) (HPow.hPow p.le …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      i : Nat
      h : Eq p.degree ↑i
      ⊢ Eq (HMul.hMul 1 p.leadingCoeff) ((p.scaleRoots p.leadingCoeff).coeff i)
    -/
  · simp [natDegree_eq_of_degree_eq_some h, leadingCoeff]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      i : Nat
      h : Not (Eq p.degree ↑i)
      ⊢ Eq (HMul.hMul (HMul.hMul (p.coeff i) (HPow.hPow p.leadingCoeff (HSub.hSub (H …
    -/
  · simp only [ge_iff_le, tsub_le_iff_right, smul_eq_mul, coeff_scaleRoots]
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      i : Nat
      h : Not (Eq p.degree ↑i)
      ⊢ Eq (HMul.hMul (HMul.hMul (p.coeff i) (HPow.hPow p.leadingCoeff (HSub.hSub (H …
    -/
    by_cases h' : i < p.degree
      /-
        case pos
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        i : Nat
        h : Not (Eq p.degree ↑i)
        h' : LT.lt (↑i) p.degree
        ⊢ Eq (HMul.hMul (HMul.hMul (p.coeff i) (HPow.hPow p.leadingCoeff (HSub.hSub (H …
      -/
    · rw [mul_assoc, ← pow_succ, tsub_right_comm, tsub_add_cancel_of_le]
      /-
        case pos
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        i : Nat
        h : Not (Eq p.degree ↑i)
        h' : LT.lt (↑i) p.degree
        ⊢ LE.le 1 (HSub.hSub p.natDegree i)
      -/
      rw [le_tsub_iff_left (coe_lt_degree.mp h').le, Nat.succ_le_iff]
      /-
        case pos
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        i : Nat
        h : Not (Eq p.degree ↑i)
        h' : LT.lt (↑i) p.degree
        ⊢ LT.lt i p.natDegree
      -/
      exact coe_lt_degree.mp h'
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        i : Nat
        h : Not (Eq p.degree ↑i)
        h' : Not (LT.lt (↑i) p.degree)
        ⊢ Eq (HMul.hMul (HMul.hMul (p.coeff i) (HPow.hPow p.leadingCoeff (HSub.hSub (H …
      -/
    · simp [coeff_eq_zero_of_degree_lt (lt_of_le_of_ne (le_of_not_gt h') h)]
      /-
        🎉 no goals
      -/


theorem integralNormalization_degree : (integralNormalization p).degree = p.degree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq p.integralNormalization.degree p.degree
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ LE.le p.integralNormalization.degree p.degree
    -/
  · exact Finset.sup_mono p.support_integralNormalization_subset
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ LE.le p.degree p.integralNormalization.degree
    -/
  · rw [← degree_scaleRoots, ← integralNormalization_mul_C_leadingCoeff]
    /-
      case a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ LE.le (HMul.hMul p.integralNormalization (Polynomial.C p.leadingCoeff)).degr …
    -/
    exact (degree_mul_le _ _).trans (add_le_of_nonpos_right degree_C_le)
    /-
      🎉 no goals
    -/


theorem leadingCoeff_smul_integralNormalization (p : S[X]) :
    p.leadingCoeff • integralNormalization p = scaleRoots p p.leadingCoeff := by
  /-
    S : Type v
    inst✝ : CommSemiring S
    p : Polynomial S
    ⊢ Eq (HSMul.hSMul p.leadingCoeff p.integralNormalization) (p.scaleRoots p.lead …
  -/
  rw [Algebra.smul_def, algebraMap_eq, mul_comm, integralNormalization_mul_C_leadingCoeff]
  /-
    🎉 no goals
  -/


theorem integralNormalization_eval₂_leadingCoeff_mul_of_commute (h : 1 ≤ p.natDegree) (f : R →+* A)
    (x : A) (h₁ : Commute (f p.leadingCoeff) x) (h₂ : ∀ {r r'}, Commute (f r) (f r')) :
    (integralNormalization p).eval₂ f (f p.leadingCoeff * x) =
      f p.leadingCoeff ^ (p.natDegree - 1) * p.eval₂ f x := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    A : Type u_1
    inst✝ : Semiring A
    h : LE.le 1 p.natDegree
    f : RingHom R A
    x : A
    h₁ : Commute (f p.leadingCoeff) x
    h₂ : ∀ {r r' : R}, Commute (f r) (f r')
    ⊢ Eq (Polynomial.eval₂ f (HMul.hMul (f p.leadingCoeff) x) p.integralNormalizat …
  -/
  rw [eval₂_eq_sum_range, eval₂_eq_sum_range, Finset.mul_sum]
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    A : Type u_1
    inst✝ : Semiring A
    h : LE.le 1 p.natDegree
    f : RingHom R A
    x : A
    h₁ : Commute (f p.leadingCoeff) x
    h₂ : ∀ {r r' : R}, Commute (f r) (f r')
    ⊢ Eq ((Finset.range (HAdd.hAdd p.integralNormalization.natDegree 1)).sum fun i …
  -/
  apply Finset.sum_congr
    /-
      case h
      R : Type u
      inst✝¹ : Semiring R
      p : Polynomial R
      A : Type u_1
      inst✝ : Semiring A
      h : LE.le 1 p.natDegree
      f : RingHom R A
      x : A
      h₁ : Commute (f p.leadingCoeff) x
      h₂ : ∀ {r r' : R}, Commute (f r) (f r')
      ⊢ Eq (Finset.range (HAdd.hAdd p.integralNormalization.natDegree 1)) (Finset.ra …
    -/
  · rw [natDegree_eq_of_degree_eq p.integralNormalization_degree]
    /-
      🎉 no goals
    -/
  /-
    case a
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    A : Type u_1
    inst✝ : Semiring A
    h : LE.le 1 p.natDegree
    f : RingHom R A
    x : A
    h₁ : Commute (f p.leadingCoeff) x
    h₂ : ∀ {r r' : R}, Commute (f r) (f r')
    ⊢ ∀ (x_1 : Nat), Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) x_1 → …
  -/
  intro n _hn
  rw [h₁.mul_pow, ← mul_assoc, ← f.map_pow, ← f.map_mul,
    integralNormalization_coeff_mul_leadingCoeff_pow _ h, f.map_mul, h₂.eq, f.map_pow, mul_assoc]


theorem integralNormalization_eval₂_leadingCoeff_mul (h : 1 ≤ p.natDegree) (f : R →+* S) (x : S) :
    (integralNormalization p).eval₂ f (f p.leadingCoeff * x) =
      f p.leadingCoeff ^ (p.natDegree - 1) * p.eval₂ f x :=
  integralNormalization_eval₂_leadingCoeff_mul_of_commute h _ _ (.all _ _) (.all _ _)


theorem integralNormalization_eval₂_eq_zero_of_commute {p : R[X]} (f : R →+* A) {z : A}
    (hz : eval₂ f z p = 0) (h₁ : Commute (f p.leadingCoeff) z) (h₂ : ∀ {r r'}, Commute (f r) (f r'))
    (inj : ∀ x : R, f x = 0 → x = 0) :
    eval₂ f (f p.leadingCoeff * z) (integralNormalization p) = 0 := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    A : Type u_1
    inst✝ : Semiring A
    p : Polynomial R
    f : RingHom R A
    z : A
    hz : Eq (Polynomial.eval₂ f z p) 0
    h₁ : Commute (f p.leadingCoeff) z
    h₂ : ∀ {r r' : R}, Commute (f r) (f r')
    inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
    ⊢ Eq (Polynomial.eval₂ f (HMul.hMul (f p.leadingCoeff) z) p.integralNormalizat …
  -/
  obtain (h | h) := p.natDegree.eq_zero_or_pos
    /-
      case inl
      R : Type u
      inst✝¹ : Semiring R
      A : Type u_1
      inst✝ : Semiring A
      p : Polynomial R
      f : RingHom R A
      z : A
      hz : Eq (Polynomial.eval₂ f z p) 0
      h₁ : Commute (f p.leadingCoeff) z
      h₂ : ∀ {r r' : R}, Commute (f r) (f r')
      inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
      h : Eq p.natDegree 0
      ⊢ Eq (Polynomial.eval₂ f (HMul.hMul (f p.leadingCoeff) z) p.integralNormalizat …
    -/
  · by_cases h0 : coeff p 0 = 0
      /-
        case pos
        R : Type u
        inst✝¹ : Semiring R
        A : Type u_1
        inst✝ : Semiring A
        p : Polynomial R
        f : RingHom R A
        z : A
        hz : Eq (Polynomial.eval₂ f z p) 0
        h₁ : Commute (f p.leadingCoeff) z
        h₂ : ∀ {r r' : R}, Commute (f r) (f r')
        inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
        h : Eq p.natDegree 0
        h0 : Eq (p.coeff 0) 0
        ⊢ Eq (Polynomial.eval₂ f (HMul.hMul (f p.leadingCoeff) z) p.integralNormalizat …
      -/
    · rw [eq_C_of_natDegree_eq_zero h]
      /-
        case pos
        R : Type u
        inst✝¹ : Semiring R
        A : Type u_1
        inst✝ : Semiring A
        p : Polynomial R
        f : RingHom R A
        z : A
        hz : Eq (Polynomial.eval₂ f z p) 0
        h₁ : Commute (f p.leadingCoeff) z
        h₂ : ∀ {r r' : R}, Commute (f r) (f r')
        inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
        h : Eq p.natDegree 0
        h0 : Eq (p.coeff 0) 0
        ⊢ Eq (Polynomial.eval₂ f (HMul.hMul (f (Polynomial.C (p.coeff 0)).leadingCoeff …
      -/
      simp [h0]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝¹ : Semiring R
        A : Type u_1
        inst✝ : Semiring A
        p : Polynomial R
        f : RingHom R A
        z : A
        hz : Eq (Polynomial.eval₂ f z p) 0
        h₁ : Commute (f p.leadingCoeff) z
        h₂ : ∀ {r r' : R}, Commute (f r) (f r')
        inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
        h : Eq p.natDegree 0
        h0 : Not (Eq (p.coeff 0) 0)
        ⊢ Eq (Polynomial.eval₂ f (HMul.hMul (f p.leadingCoeff) z) p.integralNormalizat …
      -/
    · rw [eq_C_of_natDegree_eq_zero h, eval₂_C] at hz
      /-
        case neg
        R : Type u
        inst✝¹ : Semiring R
        A : Type u_1
        inst✝ : Semiring A
        p : Polynomial R
        f : RingHom R A
        z : A
        hz : Eq (f (p.coeff 0)) 0
        h₁ : Commute (f p.leadingCoeff) z
        h₂ : ∀ {r r' : R}, Commute (f r) (f r')
        inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
        h : Eq p.natDegree 0
        h0 : Not (Eq (p.coeff 0) 0)
        ⊢ Eq (Polynomial.eval₂ f (HMul.hMul (f p.leadingCoeff) z) p.integralNormalizat …
      -/
      exact absurd (inj _ hz) h0
      /-
        🎉 no goals
      -/
    /-
      case inr
      R : Type u
      inst✝¹ : Semiring R
      A : Type u_1
      inst✝ : Semiring A
      p : Polynomial R
      f : RingHom R A
      z : A
      hz : Eq (Polynomial.eval₂ f z p) 0
      h₁ : Commute (f p.leadingCoeff) z
      h₂ : ∀ {r r' : R}, Commute (f r) (f r')
      inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
      h : GT.gt p.natDegree 0
      ⊢ Eq (Polynomial.eval₂ f (HMul.hMul (f p.leadingCoeff) z) p.integralNormalizat …
    -/
  · rw [integralNormalization_eval₂_leadingCoeff_mul_of_commute h _ _ h₁ h₂, hz, mul_zero]
    /-
      🎉 no goals
    -/


theorem integralNormalization_eval₂_eq_zero {p : R[X]} (f : R →+* S) {z : S} (hz : eval₂ f z p = 0)
    (inj : ∀ x : R, f x = 0 → x = 0) :
    eval₂ f (f p.leadingCoeff * z) (integralNormalization p) = 0 :=
  integralNormalization_eval₂_eq_zero_of_commute _ hz (.all _ _) (.all _ _) inj


theorem integralNormalization_aeval_eq_zero [Algebra S A] {f : S[X]} {z : A} (hz : aeval z f = 0)
    (inj : ∀ x : S, algebraMap S A x = 0 → x = 0) :
    aeval (algebraMap S A f.leadingCoeff * z) (integralNormalization f) = 0 :=
  integralNormalization_eval₂_eq_zero_of_commute (algebraMap S A) hz
    (Algebra.commute_algebraMap_left _ _) (.map (.all _ _) _) inj


@[simp]
theorem support_integralNormalization {f : R[X]} :
    (integralNormalization f).support = f.support := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : IsCancelMulZero R
    f : Polynomial R
    ⊢ Eq f.integralNormalization.support f.support
  -/
  nontriviality R using Subsingleton.eq_zero
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : IsCancelMulZero R
    f : Polynomial R
    a✝ : Nontrivial R
    ⊢ Eq f.integralNormalization.support f.support
  -/
  have : IsDomain R := {}
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : IsCancelMulZero R
    f : Polynomial R
    a✝ : Nontrivial R
    this : IsDomain R
    ⊢ Eq f.integralNormalization.support f.support
  -/
  by_cases hf : f = 0; · simp [hf]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : IsCancelMulZero R
    f : Polynomial R
    a✝ : Nontrivial R
    this : IsDomain R
    hf : Not (Eq f 0)
    ⊢ Eq f.integralNormalization.support f.support
  -/
  ext i
  /-
    case neg.h
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : IsCancelMulZero R
    f : Polynomial R
    a✝ : Nontrivial R
    this : IsDomain R
    hf : Not (Eq f 0)
    i : Nat
    ⊢ Iff (Membership.mem f.integralNormalization.support i) (Membership.mem f.sup …
  -/
  refine ⟨fun h => support_integralNormalization_subset h, ?_⟩
  /-
    case neg.h
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : IsCancelMulZero R
    f : Polynomial R
    a✝ : Nontrivial R
    this : IsDomain R
    hf : Not (Eq f 0)
    i : Nat
    ⊢ Membership.mem f.support i → Membership.mem f.integralNormalization.support i
  -/
  simp only [integralNormalization_coeff, mem_support_iff]
  /-
    case neg.h
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : IsCancelMulZero R
    f : Polynomial R
    a✝ : Nontrivial R
    this : IsDomain R
    hf : Not (Eq f 0)
    i : Nat
    ⊢ Ne (f.coeff i) 0 → Ne (ite (Eq f.degree ↑i) 1 (HMul.hMul (f.coeff i) (HPow.h …
  -/
  intro hfi
  /-
    case neg.h
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : IsCancelMulZero R
    f : Polynomial R
    a✝ : Nontrivial R
    this : IsDomain R
    hf : Not (Eq f 0)
    i : Nat
    hfi : Ne (f.coeff i) 0
    ⊢ Ne (ite (Eq f.degree ↑i) 1 (HMul.hMul (f.coeff i) (HPow.hPow f.leadingCoeff  …
  -/
                        /-
                          🎉 no goals
                        -/
  split_ifs with hi <;> simp [hf, hfi, hi]
                        /-
                          🎉 no goals
                        -/


