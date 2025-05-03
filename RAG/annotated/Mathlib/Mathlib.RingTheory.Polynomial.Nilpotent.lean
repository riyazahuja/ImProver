lemma isNilpotent_C_mul_pow_X_of_isNilpotent (n : ℕ) (hnil : IsNilpotent r) :
    IsNilpotent ((C r) * X ^ n) := by
  /-
    R : Type u_1
    r : R
    inst✝ : Semiring R
    n : Nat
    hnil : IsNilpotent r
    ⊢ IsNilpotent (HMul.hMul (Polynomial.C r) (HPow.hPow Polynomial.X n))
  -/
  refine Commute.isNilpotent_mul_left (commute_X_pow _ _).symm ?_
  /-
    R : Type u_1
    r : R
    inst✝ : Semiring R
    n : Nat
    hnil : IsNilpotent r
    ⊢ IsNilpotent (Polynomial.C r)
  -/
  obtain ⟨m, hm⟩ := hnil
  /-
    case intro
    R : Type u_1
    r : R
    inst✝ : Semiring R
    n m : Nat
    hm : Eq (HPow.hPow r m) 0
    ⊢ IsNilpotent (Polynomial.C r)
  -/
  refine ⟨m, ?_⟩
  /-
    case intro
    R : Type u_1
    r : R
    inst✝ : Semiring R
    n m : Nat
    hm : Eq (HPow.hPow r m) 0
    ⊢ Eq (HPow.hPow (Polynomial.C r) m) 0
  -/
  rw [← C_pow, hm, C_0]
  /-
    🎉 no goals
  -/


lemma isNilpotent_pow_X_mul_C_of_isNilpotent (n : ℕ) (hnil : IsNilpotent r) :
    IsNilpotent (X ^ n * (C r)) := by
  /-
    R : Type u_1
    r : R
    inst✝ : Semiring R
    n : Nat
    hnil : IsNilpotent r
    ⊢ IsNilpotent (HMul.hMul (HPow.hPow Polynomial.X n) (Polynomial.C r))
  -/
  rw [commute_X_pow]
  /-
    R : Type u_1
    r : R
    inst✝ : Semiring R
    n : Nat
    hnil : IsNilpotent r
    ⊢ IsNilpotent (HMul.hMul (Polynomial.C r) (HPow.hPow Polynomial.X n))
  -/
  exact isNilpotent_C_mul_pow_X_of_isNilpotent n hnil
  /-
    🎉 no goals
  -/


@[simp] lemma isNilpotent_monomial_iff {n : ℕ} :
    IsNilpotent (monomial (R := R) n r) ↔ IsNilpotent r :=
                          /-
                            R : Type u_1
                            r : R
                            inst✝ : Semiring R
                            n k : Nat
                            ⊢ Iff (Eq (HPow.hPow ((Polynomial.monomial n) r) k) 0) (Eq (HPow.hPow r k) 0)
                          -/
  exists_congr fun k ↦ by simp
                          /-
                            🎉 no goals
                          -/


@[simp] lemma isNilpotent_C_iff :
    IsNilpotent (C r) ↔ IsNilpotent r :=
                          /-
                            R : Type u_1
                            r : R
                            inst✝ : Semiring R
                            k : Nat
                            ⊢ Iff (Eq (HPow.hPow (Polynomial.C r) k) 0) (Eq (HPow.hPow r k) 0)
                          -/
  exists_congr fun k ↦ by simpa only [← C_pow] using C_eq_zero
                          /-
                            🎉 no goals
                          -/


@[simp] lemma isNilpotent_X_mul_iff :
    IsNilpotent (X * P) ↔ IsNilpotent P := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    P : Polynomial R
    ⊢ Iff (IsNilpotent (HMul.hMul Polynomial.X P)) (IsNilpotent P)
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      P : Polynomial R
      h : IsNilpotent (HMul.hMul Polynomial.X P)
      ⊢ IsNilpotent P
    -/
  · rwa [Commute.isNilpotent_mul_right_iff (commute_X P) (by simp)] at h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      P : Polynomial R
      ⊢ IsNilpotent P → IsNilpotent (HMul.hMul Polynomial.X P)
    -/
  · rintro ⟨k, hk⟩
    /-
      case refine_2.intro
      R : Type u_1
      inst✝ : Semiring R
      P : Polynomial R
      k : Nat
      hk : Eq (HPow.hPow P k) 0
      ⊢ IsNilpotent (HMul.hMul Polynomial.X P)
    -/
    exact ⟨k, by simp [(commute_X P).mul_pow, hk]⟩
    /-
      🎉 no goals
    -/


@[simp] lemma isNilpotent_mul_X_iff :
    IsNilpotent (P * X) ↔ IsNilpotent P := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    P : Polynomial R
    ⊢ Iff (IsNilpotent (HMul.hMul P Polynomial.X)) (IsNilpotent P)
  -/
  rw [← commute_X P]
  /-
    R : Type u_1
    inst✝ : Semiring R
    P : Polynomial R
    ⊢ Iff (IsNilpotent (HMul.hMul Polynomial.X P)) (IsNilpotent P)
  -/
  exact isNilpotent_X_mul_iff
  /-
    🎉 no goals
  -/


protected lemma isNilpotent_iff :
    IsNilpotent P ↔ ∀ i, IsNilpotent (coeff P i) := by
  refine
    ⟨P.recOnHorner (by simp) (fun p r hp₀ _ hp hpr i ↦ ?_) (fun p _ hnp hpX i ↦ ?_), fun h ↦ ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      h : ∀ (i : Nat), IsNilpotent (P.coeff i)
      ⊢ IsNilpotent P
    -/
  · rw [← sum_monomial_eq P]
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      h : ∀ (i : Nat), IsNilpotent (P.coeff i)
      ⊢ IsNilpotent (P.sum fun n a => (Polynomial.monomial n) a)
    -/
    exact isNilpotent_sum (fun i _ ↦ by simpa only [isNilpotent_monomial_iff] using h i)
    /-
      🎉 no goals
    -/
  · have hr : IsNilpotent (C r) := by
      obtain ⟨k, hk⟩ := hpr
      replace hp : eval 0 p = 0 := by rwa [coeff_zero_eq_aeval_zero] at hp₀
      refine isNilpotent_C_iff.mpr ⟨k, ?_⟩
      simpa [coeff_zero_eq_aeval_zero, hp] using congr_arg (fun q ↦ coeff q 0) hk
    /-
      case refine_2
      R : Type u_1
      inst✝ : CommRing R
      P p : Polynomial R
      r : R
      hp₀ : Eq (p.coeff 0) 0
      x✝ : Ne r 0
      hp : IsNilpotent p → ∀ (i : Nat), IsNilpotent (p.coeff i)
      hpr : IsNilpotent (HAdd.hAdd p (Polynomial.C r))
      i : Nat
      hr : IsNilpotent (Polynomial.C r)
      ⊢ IsNilpotent ((HAdd.hAdd p (Polynomial.C r)).coeff i)
    -/
    cases' i with i
      /-
        case refine_2.zero
        R : Type u_1
        inst✝ : CommRing R
        P p : Polynomial R
        r : R
        hp₀ : Eq (p.coeff 0) 0
        x✝ : Ne r 0
        hp : IsNilpotent p → ∀ (i : Nat), IsNilpotent (p.coeff i)
        hpr : IsNilpotent (HAdd.hAdd p (Polynomial.C r))
        hr : IsNilpotent (Polynomial.C r)
        ⊢ IsNilpotent ((HAdd.hAdd p (Polynomial.C r)).coeff 0)
      -/
    · simpa [hp₀] using hr
      /-
        🎉 no goals
      -/
    /-
      case refine_2.succ
      R : Type u_1
      inst✝ : CommRing R
      P p : Polynomial R
      r : R
      hp₀ : Eq (p.coeff 0) 0
      x✝ : Ne r 0
      hp : IsNilpotent p → ∀ (i : Nat), IsNilpotent (p.coeff i)
      hpr : IsNilpotent (HAdd.hAdd p (Polynomial.C r))
      hr : IsNilpotent (Polynomial.C r)
      i : Nat
      ⊢ IsNilpotent ((HAdd.hAdd p (Polynomial.C r)).coeff (HAdd.hAdd i 1))
    -/
    simp only [coeff_add, coeff_C_succ, add_zero]
    /-
      case refine_2.succ
      R : Type u_1
      inst✝ : CommRing R
      P p : Polynomial R
      r : R
      hp₀ : Eq (p.coeff 0) 0
      x✝ : Ne r 0
      hp : IsNilpotent p → ∀ (i : Nat), IsNilpotent (p.coeff i)
      hpr : IsNilpotent (HAdd.hAdd p (Polynomial.C r))
      hr : IsNilpotent (Polynomial.C r)
      i : Nat
      ⊢ IsNilpotent (p.coeff (HAdd.hAdd i 1))
    -/
    apply hp
    /-
      case refine_2.succ.a
      R : Type u_1
      inst✝ : CommRing R
      P p : Polynomial R
      r : R
      hp₀ : Eq (p.coeff 0) 0
      x✝ : Ne r 0
      hp : IsNilpotent p → ∀ (i : Nat), IsNilpotent (p.coeff i)
      hpr : IsNilpotent (HAdd.hAdd p (Polynomial.C r))
      hr : IsNilpotent (Polynomial.C r)
      i : Nat
      ⊢ IsNilpotent p
    -/
    simpa using Commute.isNilpotent_sub (Commute.all _ _) hpr hr
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      inst✝ : CommRing R
      P p : Polynomial R
      x✝ : Ne p 0
      hnp : IsNilpotent p → ∀ (i : Nat), IsNilpotent (p.coeff i)
      hpX : IsNilpotent (HMul.hMul p Polynomial.X)
      i : Nat
      ⊢ IsNilpotent ((HMul.hMul p Polynomial.X).coeff i)
    -/
  · cases' i with i
      /-
        case refine_3.zero
        R : Type u_1
        inst✝ : CommRing R
        P p : Polynomial R
        x✝ : Ne p 0
        hnp : IsNilpotent p → ∀ (i : Nat), IsNilpotent (p.coeff i)
        hpX : IsNilpotent (HMul.hMul p Polynomial.X)
        ⊢ IsNilpotent ((HMul.hMul p Polynomial.X).coeff 0)
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case refine_3.succ
      R : Type u_1
      inst✝ : CommRing R
      P p : Polynomial R
      x✝ : Ne p 0
      hnp : IsNilpotent p → ∀ (i : Nat), IsNilpotent (p.coeff i)
      hpX : IsNilpotent (HMul.hMul p Polynomial.X)
      i : Nat
      ⊢ IsNilpotent ((HMul.hMul p Polynomial.X).coeff (HAdd.hAdd i 1))
    -/
    simpa using hnp (isNilpotent_mul_X_iff.mp hpX) i
    /-
      🎉 no goals
    -/


@[simp] lemma isNilpotent_reflect_iff {P : R[X]} {N : ℕ} (hN : P.natDegree ≤ N) :
    IsNilpotent (reflect N P) ↔ IsNilpotent P := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Polynomial R
    N : Nat
    hN : LE.le P.natDegree N
    ⊢ Iff (IsNilpotent (Polynomial.reflect N P)) (IsNilpotent P)
  -/
  simp only [Polynomial.isNilpotent_iff, coeff_reverse]
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Polynomial R
    N : Nat
    hN : LE.le P.natDegree N
    ⊢ Iff (∀ (i : Nat), IsNilpotent ((Polynomial.reflect N P).coeff i)) (∀ (i : Na …
  -/
  refine ⟨fun h i ↦ ?_, fun h i ↦ ?_⟩ <;> rcases le_or_lt i N with hi | hi
    /-
      case refine_1.inl
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      N : Nat
      hN : LE.le P.natDegree N
      h : ∀ (i : Nat), IsNilpotent ((Polynomial.reflect N P).coeff i)
      i : Nat
      hi : LE.le i N
      ⊢ IsNilpotent (P.coeff i)
    -/
  · simpa [tsub_tsub_cancel_of_le hi] using h (N - i)
    /-
      🎉 no goals
    -/
    /-
      case refine_1.inr
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      N : Nat
      hN : LE.le P.natDegree N
      h : ∀ (i : Nat), IsNilpotent ((Polynomial.reflect N P).coeff i)
      i : Nat
      hi : LT.lt N i
      ⊢ IsNilpotent (P.coeff i)
    -/
  · simp [coeff_eq_zero_of_natDegree_lt <| lt_of_le_of_lt hN hi]
    /-
      🎉 no goals
    -/
    /-
      case refine_2.inl
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      N : Nat
      hN : LE.le P.natDegree N
      h : ∀ (i : Nat), IsNilpotent (P.coeff i)
      i : Nat
      hi : LE.le i N
      ⊢ IsNilpotent ((Polynomial.reflect N P).coeff i)
    -/
  · simpa [hi, revAt_le] using h (N - i)
    /-
      🎉 no goals
    -/
    /-
      case refine_2.inr
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      N : Nat
      hN : LE.le P.natDegree N
      h : ∀ (i : Nat), IsNilpotent (P.coeff i)
      i : Nat
      hi : LT.lt N i
      ⊢ IsNilpotent ((Polynomial.reflect N P).coeff i)
    -/
  · simpa [revAt_eq_self_of_lt hi] using h i
    /-
      🎉 no goals
    -/


@[simp] lemma isNilpotent_reverse_iff :
    IsNilpotent P.reverse ↔ IsNilpotent P :=
  isNilpotent_reflect_iff (le_refl _)


/-- Let `P` be a polynomial over `R`. If its constant term is a unit and its other coefficients are
nilpotent, then `P` is a unit.

See also `Polynomial.isUnit_iff_coeff_isUnit_isNilpotent`. -/
theorem isUnit_of_coeff_isUnit_isNilpotent (hunit : IsUnit (P.coeff 0))
    (hnil : ∀ i, i ≠ 0 → IsNilpotent (P.coeff i)) : IsUnit P := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Polynomial R
    hunit : IsUnit (P.coeff 0)
    hnil : ∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i)
    ⊢ IsUnit P
  -/
  induction' h : P.natDegree using Nat.strong_induction_on with k hind generalizing P
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    k : Nat
    hind : ∀ (m : Nat), LT.lt m k → ∀ {P : Polynomial R}, IsUnit (P.coeff 0) → (∀  …
    P : Polynomial R
    hunit : IsUnit (P.coeff 0)
    hnil : ∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i)
    h : Eq P.natDegree k
    ⊢ IsUnit P
  -/
  by_cases hdeg : P.natDegree = 0
  { rw [eq_C_of_natDegree_eq_zero hdeg]
    exact hunit.map C }
  /-
    case neg
    R : Type u_1
    inst✝ : CommRing R
    k : Nat
    hind : ∀ (m : Nat), LT.lt m k → ∀ {P : Polynomial R}, IsUnit (P.coeff 0) → (∀  …
    P : Polynomial R
    hunit : IsUnit (P.coeff 0)
    hnil : ∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i)
    h : Eq P.natDegree k
    hdeg : Not (Eq P.natDegree 0)
    ⊢ IsUnit P
  -/
  set P₁ := P.eraseLead with hP₁
  suffices IsUnit P₁ by
    rw [← eraseLead_add_monomial_natDegree_leadingCoeff P, ← C_mul_X_pow_eq_monomial, ← hP₁]
    refine IsNilpotent.isUnit_add_left_of_commute ?_ this (Commute.all _ _)
    exact isNilpotent_C_mul_pow_X_of_isNilpotent _ (hnil _ hdeg)
  have hdeg₂ := lt_of_le_of_lt P.eraseLead_natDegree_le (Nat.sub_lt
    (Nat.pos_of_ne_zero hdeg) zero_lt_one)
  /-
    case neg
    R : Type u_1
    inst✝ : CommRing R
    k : Nat
    hind : ∀ (m : Nat), LT.lt m k → ∀ {P : Polynomial R}, IsUnit (P.coeff 0) → (∀  …
    P : Polynomial R
    hunit : IsUnit (P.coeff 0)
    hnil : ∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i)
    h : Eq P.natDegree k
    hdeg : Not (Eq P.natDegree 0)
    P₁ : Polynomial R := P.eraseLead
    hP₁ : Eq P₁ P.eraseLead
    hdeg₂ : LT.lt P.eraseLead.natDegree P.natDegree
    ⊢ IsUnit P₁
  -/
  refine hind P₁.natDegree ?_ ?_ (fun i hi => ?_) rfl
    /-
      case neg.refine_1
      R : Type u_1
      inst✝ : CommRing R
      k : Nat
      hind : ∀ (m : Nat), LT.lt m k → ∀ {P : Polynomial R}, IsUnit (P.coeff 0) → (∀  …
      P : Polynomial R
      hunit : IsUnit (P.coeff 0)
      hnil : ∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i)
      h : Eq P.natDegree k
      hdeg : Not (Eq P.natDegree 0)
      P₁ : Polynomial R := P.eraseLead
      hP₁ : Eq P₁ P.eraseLead
      hdeg₂ : LT.lt P.eraseLead.natDegree P.natDegree
      ⊢ LT.lt P₁.natDegree k
    -/
  · simp_rw [P₁, ← h, hdeg₂]
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      R : Type u_1
      inst✝ : CommRing R
      k : Nat
      hind : ∀ (m : Nat), LT.lt m k → ∀ {P : Polynomial R}, IsUnit (P.coeff 0) → (∀  …
      P : Polynomial R
      hunit : IsUnit (P.coeff 0)
      hnil : ∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i)
      h : Eq P.natDegree k
      hdeg : Not (Eq P.natDegree 0)
      P₁ : Polynomial R := P.eraseLead
      hP₁ : Eq P₁ P.eraseLead
      hdeg₂ : LT.lt P.eraseLead.natDegree P.natDegree
      ⊢ IsUnit (P₁.coeff 0)
    -/
  · simp_rw [P₁, eraseLead_coeff_of_ne _ (Ne.symm hdeg), hunit]
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_3
      R : Type u_1
      inst✝ : CommRing R
      k : Nat
      hind : ∀ (m : Nat), LT.lt m k → ∀ {P : Polynomial R}, IsUnit (P.coeff 0) → (∀  …
      P : Polynomial R
      hunit : IsUnit (P.coeff 0)
      hnil : ∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i)
      h : Eq P.natDegree k
      hdeg : Not (Eq P.natDegree 0)
      P₁ : Polynomial R := P.eraseLead
      hP₁ : Eq P₁ P.eraseLead
      hdeg₂ : LT.lt P.eraseLead.natDegree P.natDegree
      i : Nat
      hi : Ne i 0
      ⊢ IsNilpotent (P₁.coeff i)
    -/
  · by_cases H : i ≤ P₁.natDegree
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        k : Nat
        hind : ∀ (m : Nat), LT.lt m k → ∀ {P : Polynomial R}, IsUnit (P.coeff 0) → (∀  …
        P : Polynomial R
        hunit : IsUnit (P.coeff 0)
        hnil : ∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i)
        h : Eq P.natDegree k
        hdeg : Not (Eq P.natDegree 0)
        P₁ : Polynomial R := P.eraseLead
        hP₁ : Eq P₁ P.eraseLead
        hdeg₂ : LT.lt P.eraseLead.natDegree P.natDegree
        i : Nat
        hi : Ne i 0
        H : LE.le i P₁.natDegree
        ⊢ IsNilpotent (P₁.coeff i)
      -/
    · simp_rw [P₁, eraseLead_coeff_of_ne _ (ne_of_lt (lt_of_le_of_lt H hdeg₂)), hnil i hi]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        k : Nat
        hind : ∀ (m : Nat), LT.lt m k → ∀ {P : Polynomial R}, IsUnit (P.coeff 0) → (∀  …
        P : Polynomial R
        hunit : IsUnit (P.coeff 0)
        hnil : ∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i)
        h : Eq P.natDegree k
        hdeg : Not (Eq P.natDegree 0)
        P₁ : Polynomial R := P.eraseLead
        hP₁ : Eq P₁ P.eraseLead
        hdeg₂ : LT.lt P.eraseLead.natDegree P.natDegree
        i : Nat
        hi : Ne i 0
        H : Not (LE.le i P₁.natDegree)
        ⊢ IsNilpotent (P₁.coeff i)
      -/
    · simp_rw [coeff_eq_zero_of_natDegree_lt (lt_of_not_ge H), IsNilpotent.zero]
      /-
        🎉 no goals
      -/


/-- Let `P` be a polynomial over `R`. If `P` is a unit, then all its coefficients are nilpotent,
except its constant term which is a unit.

See also `Polynomial.isUnit_iff_coeff_isUnit_isNilpotent`. -/
theorem coeff_isUnit_isNilpotent_of_isUnit (hunit : IsUnit P) :
    IsUnit (P.coeff 0) ∧ (∀ i, i ≠ 0 → IsNilpotent (P.coeff i)) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Polynomial R
    hunit : IsUnit P
    ⊢ And (IsUnit (P.coeff 0)) (∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i))
  -/
  obtain ⟨Q, hQ⟩ := IsUnit.exists_right_inv hunit
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    P : Polynomial R
    hunit : IsUnit P
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 1
    ⊢ And (IsUnit (P.coeff 0)) (∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i))
  -/
  constructor
    /-
      case intro.left
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      hunit : IsUnit P
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 1
      ⊢ IsUnit (P.coeff 0)
    -/
  · refine isUnit_of_mul_eq_one _ (Q.coeff 0) ?_
    /-
      case intro.left
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      hunit : IsUnit P
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 1
      ⊢ Eq (HMul.hMul (P.coeff 0) (Q.coeff 0)) 1
    -/
    have h := (mul_coeff_zero P Q).symm
    /-
      case intro.left
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      hunit : IsUnit P
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 1
      h : Eq (HMul.hMul (P.coeff 0) (Q.coeff 0)) ((HMul.hMul P Q).coeff 0)
      ⊢ Eq (HMul.hMul (P.coeff 0) (Q.coeff 0)) 1
    -/
    rwa [hQ, coeff_one_zero] at h
    /-
      🎉 no goals
    -/
    /-
      case intro.right
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      hunit : IsUnit P
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 1
      ⊢ ∀ (i : Nat), Ne i 0 → IsNilpotent (P.coeff i)
    -/
  · intros n hn
    /-
      case intro.right
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      hunit : IsUnit P
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 1
      n : Nat
      hn : Ne n 0
      ⊢ IsNilpotent (P.coeff n)
    -/
    rw [nilpotent_iff_mem_prime]
    /-
      case intro.right
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      hunit : IsUnit P
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 1
      n : Nat
      hn : Ne n 0
      ⊢ ∀ (J : Ideal R), J.IsPrime → Membership.mem J (P.coeff n)
    -/
    intros I hI
    /-
      case intro.right
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      hunit : IsUnit P
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 1
      n : Nat
      hn : Ne n 0
      I : Ideal R
      hI : I.IsPrime
      ⊢ Membership.mem I (P.coeff n)
    -/
    let f := mapRingHom (Ideal.Quotient.mk I)
    have hPQ : degree (f P) = 0 ∧ degree (f Q) = 0 := by
      rw [← Nat.WithBot.add_eq_zero_iff, ← degree_mul, ← _root_.map_mul, hQ, map_one, degree_one]
    have hcoeff : (f P).coeff n = 0 := by
      refine coeff_eq_zero_of_degree_lt ?_
      rw [hPQ.1]
      exact WithBot.coe_pos.2 hn.bot_lt
    /-
      case intro.right
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      hunit : IsUnit P
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 1
      n : Nat
      hn : Ne n 0
      I : Ideal R
      hI : I.IsPrime
      f : RingHom (Polynomial R) (Polynomial (HasQuotient.Quotient R I)) := Polynomi …
      hPQ : And (Eq (f P).degree 0) (Eq (f Q).degree 0)
      hcoeff : Eq ((f P).coeff n) 0
      ⊢ Membership.mem I (P.coeff n)
    -/
    rw [coe_mapRingHom, coeff_map, ← RingHom.mem_ker, Ideal.mk_ker] at hcoeff
    /-
      case intro.right
      R : Type u_1
      inst✝ : CommRing R
      P : Polynomial R
      hunit : IsUnit P
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 1
      n : Nat
      hn : Ne n 0
      I : Ideal R
      hI : I.IsPrime
      f : RingHom (Polynomial R) (Polynomial (HasQuotient.Quotient R I)) := Polynomi …
      hPQ : And (Eq (f P).degree 0) (Eq (f Q).degree 0)
      hcoeff : Membership.mem I (P.coeff n)
      ⊢ Membership.mem I (P.coeff n)
    -/
    exact hcoeff
    /-
      🎉 no goals
    -/


/-- Let `P` be a polynomial over `R`. `P` is a unit if and only if all its coefficients are
nilpotent, except its constant term which is a unit.

See also `Polynomial.isUnit_iff'`. -/
theorem isUnit_iff_coeff_isUnit_isNilpotent :
    IsUnit P ↔ IsUnit (P.coeff 0) ∧ (∀ i, i ≠ 0 → IsNilpotent (P.coeff i)) :=
  ⟨coeff_isUnit_isNilpotent_of_isUnit, fun H => isUnit_of_coeff_isUnit_isNilpotent H.1 H.2⟩


@[simp] lemma isUnit_C_add_X_mul_iff :
    IsUnit (C r + X * P) ↔ IsUnit r ∧ IsNilpotent P := by
  /-
    R : Type u_1
    r : R
    inst✝ : CommRing R
    P : Polynomial R
    ⊢ Iff (IsUnit (HAdd.hAdd (Polynomial.C r) (HMul.hMul Polynomial.X P))) (And (I …
  -/
  have : ∀ i, coeff (C r + X * P) (i + 1) = coeff P i := by simp
  /-
    R : Type u_1
    r : R
    inst✝ : CommRing R
    P : Polynomial R
    this : ∀ (i : Nat), Eq ((HAdd.hAdd (Polynomial.C r) (HMul.hMul Polynomial.X P) …
    ⊢ Iff (IsUnit (HAdd.hAdd (Polynomial.C r) (HMul.hMul Polynomial.X P))) (And (I …
  -/
  simp_rw [isUnit_iff_coeff_isUnit_isNilpotent, Nat.forall_ne_zero_iff, this]
  simp only [coeff_add, coeff_C_zero, mul_coeff_zero, coeff_X_zero, zero_mul, add_zero,
    and_congr_right_iff, ← Polynomial.isNilpotent_iff]


lemma isUnit_iff' :
    IsUnit P ↔ IsUnit (eval 0 P) ∧ IsNilpotent (P /ₘ X)  := by
  suffices P = C (eval 0 P) + X * (P /ₘ X) by
    conv_lhs => rw [this]; simp
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Polynomial R
    ⊢ Eq P (HAdd.hAdd (Polynomial.C (Polynomial.eval 0 P)) (HMul.hMul Polynomial.X …
  -/
  conv_lhs => rw [← modByMonic_add_div P monic_X]
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Polynomial R
    ⊢ Eq (HAdd.hAdd (P.modByMonic Polynomial.X) (HMul.hMul Polynomial.X (P.divByMo …
  -/
  simp [modByMonic_X]
  /-
    🎉 no goals
  -/


theorem not_isUnit_of_natDegree_pos_of_isReduced [IsReduced R] (p : R[X])
    (hpl : 0 < p.natDegree) : ¬ IsUnit p := by
  simp only [ne_eq, isNilpotent_iff_eq_zero, not_and, not_forall, exists_prop,
    Polynomial.isUnit_iff_coeff_isUnit_isNilpotent]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsReduced R
    p : Polynomial R
    hpl : LT.lt 0 p.natDegree
    ⊢ IsUnit (p.coeff 0) → Exists fun x => And (Not (Eq x 0)) (Not (Eq (p.coeff x) …
  -/
  intro _
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsReduced R
    p : Polynomial R
    hpl : LT.lt 0 p.natDegree
    a✝ : IsUnit (p.coeff 0)
    ⊢ Exists fun x => And (Not (Eq x 0)) (Not (Eq (p.coeff x) 0))
  -/
  refine ⟨p.natDegree, hpl.ne', ?_⟩
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsReduced R
    p : Polynomial R
    hpl : LT.lt 0 p.natDegree
    a✝ : IsUnit (p.coeff 0)
    ⊢ Not (Eq (p.coeff p.natDegree) 0)
  -/
  contrapose! hpl
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsReduced R
    p : Polynomial R
    a✝ : IsUnit (p.coeff 0)
    hpl : Eq (p.coeff p.natDegree) 0
    ⊢ LE.le p.natDegree 0
  -/
  simp only [coeff_natDegree, leadingCoeff_eq_zero] at hpl
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsReduced R
    p : Polynomial R
    a✝ : IsUnit (p.coeff 0)
    hpl : Eq p 0
    ⊢ LE.le p.natDegree 0
  -/
  simp [hpl]
  /-
    🎉 no goals
  -/


theorem not_isUnit_of_degree_pos_of_isReduced [IsReduced R] (p : R[X])
    (hpl : 0 < p.degree) : ¬ IsUnit p :=
  not_isUnit_of_natDegree_pos_of_isReduced _ (natDegree_pos_iff_degree_pos.mpr hpl)


lemma isNilpotent_aeval_sub_of_isNilpotent_sub (h : IsNilpotent (a - b)) :
    IsNilpotent (aeval a P - aeval b P) := by
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    a b : S
    h : IsNilpotent (HSub.hSub a b)
    ⊢ IsNilpotent (HSub.hSub ((Polynomial.aeval a) P) ((Polynomial.aeval b) P))
  -/
  simp only [← eval_map_algebraMap]
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    a b : S
    h : IsNilpotent (HSub.hSub a b)
    ⊢ IsNilpotent (HSub.hSub (Polynomial.eval a (Polynomial.map (algebraMap R S) P …
  -/
  have ⟨c, hc⟩ := evalSubFactor (map (algebraMap R S) P) a b
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    a b : S
    h : IsNilpotent (HSub.hSub a b)
    c : S
    hc : Eq (HSub.hSub (Polynomial.eval a (Polynomial.map (algebraMap R S) P)) (Po …
    ⊢ IsNilpotent (HSub.hSub (Polynomial.eval a (Polynomial.map (algebraMap R S) P …
  -/
  exact hc ▸ (Commute.all _ _).isNilpotent_mul_right h
  /-
    🎉 no goals
  -/


lemma isUnit_aeval_of_isUnit_aeval_of_isNilpotent_sub
    (hb : IsUnit (aeval b P)) (hab : IsNilpotent (a - b)) :
    IsUnit (aeval a P) := by
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    a b : S
    hb : IsUnit ((Polynomial.aeval b) P)
    hab : IsNilpotent (HSub.hSub a b)
    ⊢ IsUnit ((Polynomial.aeval a) P)
  -/
  rw [← add_sub_cancel (aeval b P) (aeval a P)]
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    a b : S
    hb : IsUnit ((Polynomial.aeval b) P)
    hab : IsNilpotent (HSub.hSub a b)
    ⊢ IsUnit (HAdd.hAdd ((Polynomial.aeval b) P) (HSub.hSub ((Polynomial.aeval a)  …
  -/
  refine IsNilpotent.isUnit_add_left_of_commute ?_ hb (Commute.all _ _)
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    a b : S
    hb : IsUnit ((Polynomial.aeval b) P)
    hab : IsNilpotent (HSub.hSub a b)
    ⊢ IsNilpotent (HSub.hSub ((Polynomial.aeval a) P) ((Polynomial.aeval b) P))
  -/
  exact isNilpotent_aeval_sub_of_isNilpotent_sub P hab
  /-
    🎉 no goals
  -/


