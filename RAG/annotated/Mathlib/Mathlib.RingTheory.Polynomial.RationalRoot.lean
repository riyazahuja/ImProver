theorem scaleRoots_aeval_eq_zero_of_aeval_mk'_eq_zero {p : A[X]} {r : A} {s : M}
    (hr : aeval (mk' S r s) p = 0) : aeval (algebraMap A S r) (scaleRoots p s) = 0 := by
  /-
    A : Type u_1
    S : Type u_4
    inst✝³ : CommRing A
    inst✝² : CommRing S
    M : Submonoid A
    inst✝¹ : Algebra A S
    inst✝ : IsLocalization M S
    p : Polynomial A
    r : A
    s : Subtype fun x => Membership.mem M x
    hr : Eq ((Polynomial.aeval (IsLocalization.mk' S r s)) p) 0
    ⊢ Eq ((Polynomial.aeval ((algebraMap A S) r)) (p.scaleRoots ↑s)) 0
  -/
  convert scaleRoots_eval₂_eq_zero (algebraMap A S) hr
  -- Porting note: added
  /-
    case h.e'_2.h.e
    A : Type u_1
    S : Type u_4
    inst✝³ : CommRing A
    inst✝² : CommRing S
    M : Submonoid A
    inst✝¹ : Algebra A S
    inst✝ : IsLocalization M S
    p : Polynomial A
    r : A
    s : Subtype fun x => Membership.mem M x
    hr : Eq ((Polynomial.aeval (IsLocalization.mk' S r s)) p) 0
    ⊢ Eq (⇑(Polynomial.aeval ((algebraMap A S) r))) (Polynomial.eval₂ (algebraMap  …
  -/
  funext
  /-
    case h.e'_2.h.e.h
    A : Type u_1
    S : Type u_4
    inst✝³ : CommRing A
    inst✝² : CommRing S
    M : Submonoid A
    inst✝¹ : Algebra A S
    inst✝ : IsLocalization M S
    p : Polynomial A
    r : A
    s : Subtype fun x => Membership.mem M x
    hr : Eq ((Polynomial.aeval (IsLocalization.mk' S r s)) p) 0
    x✝ : Polynomial A
    ⊢ Eq ((Polynomial.aeval ((algebraMap A S) r)) x✝) (Polynomial.eval₂ (algebraMa …
  -/
  rw [aeval_def, mk'_spec' _ r s]
  /-
    🎉 no goals
  -/


theorem num_isRoot_scaleRoots_of_aeval_eq_zero [UniqueFactorizationMonoid A] {p : A[X]} {x : K}
    (hr : aeval x p = 0) : IsRoot (scaleRoots p (den A x)) (num A x) := by
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : IsDomain A
    inst✝ : UniqueFactorizationMonoid A
    p : Polynomial A
    x : K
    hr : Eq ((Polynomial.aeval x) p) 0
    ⊢ (p.scaleRoots ↑(IsFractionRing.den A x)).IsRoot (IsFractionRing.num A x)
  -/
  apply isRoot_of_eval₂_map_eq_zero (IsFractionRing.injective A K)
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : IsDomain A
    inst✝ : UniqueFactorizationMonoid A
    p : Polynomial A
    x : K
    hr : Eq ((Polynomial.aeval x) p) 0
    ⊢ Eq (Polynomial.eval₂ (algebraMap A K) ((algebraMap A K) (IsFractionRing.num  …
  -/
  refine scaleRoots_aeval_eq_zero_of_aeval_mk'_eq_zero ?_
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : IsDomain A
    inst✝ : UniqueFactorizationMonoid A
    p : Polynomial A
    x : K
    hr : Eq ((Polynomial.aeval x) p) 0
    ⊢ Eq ((Polynomial.aeval (IsLocalization.mk' K (IsFractionRing.num A x) (IsFrac …
  -/
  rw [mk'_num_den]
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : IsDomain A
    inst✝ : UniqueFactorizationMonoid A
    p : Polynomial A
    x : K
    hr : Eq ((Polynomial.aeval x) p) 0
    ⊢ Eq ((Polynomial.aeval x) p) 0
  -/
  exact hr
  /-
    🎉 no goals
  -/


/-- **Rational root theorem** part 1:
if `r : f.codomain` is a root of a polynomial over the ufd `A`,
then the numerator of `r` divides the constant coefficient -/
theorem num_dvd_of_is_root {p : A[X]} {r : K} (hr : aeval r p = 0) : num A r ∣ p.coeff 0 := by
  suffices num A r ∣ (scaleRoots p (den A r)).coeff 0 by
    simp only [coeff_scaleRoots, tsub_zero] at this
    haveI inst := Classical.propDecidable
    by_cases hr : num A r = 0
    · simp_all [nonZeroDivisors.coe_ne_zero]
    · refine dvd_of_dvd_mul_left_of_no_prime_factors hr ?_ this
      intro q dvd_num dvd_denom_pow hq
      apply hq.not_unit
      exact num_den_reduced A r dvd_num (hq.dvd_of_dvd_pow dvd_denom_pow)
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    ⊢ Dvd.dvd (IsFractionRing.num A r) ((p.scaleRoots ↑(IsFractionRing.den A r)).c …
  -/
  convert dvd_term_of_isRoot_of_dvd_terms 0 (num_isRoot_scaleRoots_of_aeval_eq_zero hr) _
    /-
      case h.e'_4
      A : Type u_1
      K : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial A
      r : K
      hr : Eq ((Polynomial.aeval r) p) 0
      ⊢ Eq ((p.scaleRoots ↑(IsFractionRing.den A r)).coeff 0) (HMul.hMul ((p.scaleRo …
    -/
  · rw [pow_zero, mul_one]
    /-
      🎉 no goals
    -/
  /-
    case convert_2
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    ⊢ ∀ (j : Nat), Ne j 0 → Dvd.dvd (IsFractionRing.num A r) (HMul.hMul ((p.scaleR …
  -/
  intro j hj
  /-
    case convert_2
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    j : Nat
    hj : Ne j 0
    ⊢ Dvd.dvd (IsFractionRing.num A r) (HMul.hMul ((p.scaleRoots ↑(IsFractionRing. …
  -/
  apply dvd_mul_of_dvd_right
  /-
    case convert_2.h
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    j : Nat
    hj : Ne j 0
    ⊢ Dvd.dvd (IsFractionRing.num A r) (HPow.hPow (IsFractionRing.num A r) j)
  -/
  convert pow_dvd_pow (num A r) (Nat.succ_le_of_lt (bot_lt_iff_ne_bot.mpr hj))
  /-
    case h.e'_3
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    j : Nat
    hj : Ne j 0
    ⊢ Eq (IsFractionRing.num A r) (HPow.hPow (IsFractionRing.num A r) Bot.bot.succ)
  -/
  exact (pow_one _).symm
  /-
    🎉 no goals
  -/


/-- Rational root theorem part 2:
if `r : f.codomain` is a root of a polynomial over the ufd `A`,
then the denominator of `r` divides the leading coefficient -/
theorem den_dvd_of_is_root {p : A[X]} {r : K} (hr : aeval r p = 0) :
    (den A r : A) ∣ p.leadingCoeff := by
  suffices (den A r : A) ∣ p.leadingCoeff * num A r ^ p.natDegree by
    refine
      dvd_of_dvd_mul_left_of_no_prime_factors (mem_nonZeroDivisors_iff_ne_zero.mp (den A r).2) ?_
        this
    intro q dvd_den dvd_num_pow hq
    apply hq.not_unit
    exact num_den_reduced A r (hq.dvd_of_dvd_pow dvd_num_pow) dvd_den
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    ⊢ Dvd.dvd (↑(IsFractionRing.den A r)) (HMul.hMul p.leadingCoeff (HPow.hPow (Is …
  -/
  rw [← coeff_scaleRoots_natDegree]
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    ⊢ Dvd.dvd (↑(IsFractionRing.den A r)) (HMul.hMul ((p.scaleRoots ?s).coeff p.na …
  -/
  apply dvd_term_of_isRoot_of_dvd_terms _ (num_isRoot_scaleRoots_of_aeval_eq_zero hr)
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    ⊢ ∀ (j : Nat), Ne j p.natDegree → Dvd.dvd (↑(IsFractionRing.den A r)) (HMul.hM …
  -/
  intro j hj
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    j : Nat
    hj : Ne j p.natDegree
    ⊢ Dvd.dvd (↑(IsFractionRing.den A r)) (HMul.hMul ((p.scaleRoots ↑(IsFractionRi …
  -/
  by_cases h : j < p.natDegree
    /-
      case pos
      A : Type u_1
      K : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial A
      r : K
      hr : Eq ((Polynomial.aeval r) p) 0
      j : Nat
      hj : Ne j p.natDegree
      h : LT.lt j p.natDegree
      ⊢ Dvd.dvd (↑(IsFractionRing.den A r)) (HMul.hMul ((p.scaleRoots ↑(IsFractionRi …
    -/
  · rw [coeff_scaleRoots]
    /-
      case pos
      A : Type u_1
      K : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial A
      r : K
      hr : Eq ((Polynomial.aeval r) p) 0
      j : Nat
      hj : Ne j p.natDegree
      h : LT.lt j p.natDegree
      ⊢ Dvd.dvd (↑(IsFractionRing.den A r)) (HMul.hMul (HMul.hMul (p.coeff j) (HPow. …
    -/
    refine (dvd_mul_of_dvd_right ?_ _).mul_right _
    /-
      case pos
      A : Type u_1
      K : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial A
      r : K
      hr : Eq ((Polynomial.aeval r) p) 0
      j : Nat
      hj : Ne j p.natDegree
      h : LT.lt j p.natDegree
      ⊢ Dvd.dvd (↑(IsFractionRing.den A r)) (HPow.hPow (↑(IsFractionRing.den A r)) ( …
    -/
    convert pow_dvd_pow (den A r : A) (Nat.succ_le_iff.mpr (lt_tsub_iff_left.mpr _))
      /-
        case h.e'_3
        A : Type u_1
        K : Type u_2
        inst✝⁵ : CommRing A
        inst✝⁴ : IsDomain A
        inst✝³ : UniqueFactorizationMonoid A
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        p : Polynomial A
        r : K
        hr : Eq ((Polynomial.aeval r) p) 0
        j : Nat
        hj : Ne j p.natDegree
        h : LT.lt j p.natDegree
        ⊢ Eq (↑(IsFractionRing.den A r)) (HPow.hPow (↑(IsFractionRing.den A r)) (Nat.s …
      -/
    · exact (pow_one _).symm
      /-
        🎉 no goals
      -/
    /-
      case pos.convert_4
      A : Type u_1
      K : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial A
      r : K
      hr : Eq ((Polynomial.aeval r) p) 0
      j : Nat
      hj : Ne j p.natDegree
      h : LT.lt j p.natDegree
      ⊢ LT.lt (HAdd.hAdd j 0) p.natDegree
    -/
    simpa using h
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    j : Nat
    hj : Ne j p.natDegree
    h : Not (LT.lt j p.natDegree)
    ⊢ Dvd.dvd (↑(IsFractionRing.den A r)) (HMul.hMul ((p.scaleRoots ↑(IsFractionRi …
  -/
  rw [← natDegree_scaleRoots p (den A r)] at *
  rw [coeff_eq_zero_of_natDegree_lt (lt_of_le_of_ne (le_of_not_gt h) hj.symm),
    zero_mul]
  /-
    case neg
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    j : Nat
    hj : Ne j (p.scaleRoots ↑(IsFractionRing.den A r)).natDegree
    h : Not (LT.lt j (p.scaleRoots ↑(IsFractionRing.den A r)).natDegree)
    ⊢ Dvd.dvd (↑(IsFractionRing.den A r)) 0
  -/
  exact dvd_zero _
  /-
    🎉 no goals
  -/


/-- **Integral root theorem**:
if `r : f.codomain` is a root of a monic polynomial over the ufd `A`,
then `r` is an integer -/
theorem isInteger_of_is_root_of_monic {p : A[X]} (hp : Monic p) {r : K} (hr : aeval r p = 0) :
    IsInteger A r :=
  isInteger_of_isUnit_den (isUnit_of_dvd_one (hp ▸ den_dvd_of_is_root hr))


theorem exists_integer_of_is_root_of_monic {p : A[X]} (hp : Monic p) {r : K} (hr : aeval r p = 0) :
    ∃ r' : A, r = algebraMap A K r' ∧ r' ∣ p.coeff 0 := by
  /- I tried deducing this from above by unwrapping IsInteger,
    but the divisibility condition is annoying -/
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    hp : p.Monic
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    ⊢ Exists fun r' => And (Eq r ((algebraMap A K) r')) (Dvd.dvd r' (p.coeff 0))
  -/
  obtain ⟨inv, h_inv⟩ := hp ▸ den_dvd_of_is_root hr
  /-
    case intro
    A : Type u_1
    K : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial A
    hp : p.Monic
    r : K
    hr : Eq ((Polynomial.aeval r) p) 0
    inv : A
    h_inv : Eq 1 (HMul.hMul (↑(IsFractionRing.den A r)) inv)
    ⊢ Exists fun r' => And (Eq r ((algebraMap A K) r')) (Dvd.dvd r' (p.coeff 0))
  -/
  use num A r * inv, ?_
    /-
      case right
      A : Type u_1
      K : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial A
      hp : p.Monic
      r : K
      hr : Eq ((Polynomial.aeval r) p) 0
      inv : A
      h_inv : Eq 1 (HMul.hMul (↑(IsFractionRing.den A r)) inv)
      ⊢ Dvd.dvd (HMul.hMul (IsFractionRing.num A r) inv) (p.coeff 0)
    -/
  · have h : inv ∣ 1 := ⟨den A r, by simpa [mul_comm] using h_inv⟩
    /-
      case right
      A : Type u_1
      K : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial A
      hp : p.Monic
      r : K
      hr : Eq ((Polynomial.aeval r) p) 0
      inv : A
      h_inv : Eq 1 (HMul.hMul (↑(IsFractionRing.den A r)) inv)
      h : Dvd.dvd inv 1
      ⊢ Dvd.dvd (HMul.hMul (IsFractionRing.num A r) inv) (p.coeff 0)
    -/
    simpa using mul_dvd_mul (num_dvd_of_is_root hr) h
    /-
      🎉 no goals
    -/
  · have d_ne_zero : algebraMap A K (den A r) ≠ 0 :=
      IsFractionRing.to_map_ne_zero_of_mem_nonZeroDivisors (den A r).prop
    /-
      case left
      A : Type u_1
      K : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial A
      hp : p.Monic
      r : K
      hr : Eq ((Polynomial.aeval r) p) 0
      inv : A
      h_inv : Eq 1 (HMul.hMul (↑(IsFractionRing.den A r)) inv)
      d_ne_zero : Ne ((algebraMap A K) ↑(IsFractionRing.den A r)) 0
      ⊢ Eq r ((algebraMap A K) (HMul.hMul (IsFractionRing.num A r) inv))
    -/
    nth_rw 1 [← mk'_num_den' A r]
    rw [div_eq_iff d_ne_zero, map_mul, mul_assoc, mul_comm ((algebraMap A K) inv),
      ← map_mul, ← h_inv, map_one, mul_one]


theorem integer_of_integral {x : K} : IsIntegral A x → IsInteger A x := fun ⟨_, hp, hx⟩ =>
  isInteger_of_is_root_of_monic hp hx

-- See library note [lower instance priority]

instance (priority := 100) instIsIntegrallyClosed : IsIntegrallyClosed A :=
  (isIntegrallyClosed_iff (FractionRing A)).mpr fun {_} => integer_of_integral


