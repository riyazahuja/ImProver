local notation "𝓟" => Submodule.span ℤ {(p : ℤ)}


theorem cyclotomic_comp_X_add_one_isEisensteinAt [hp : Fact p.Prime] :
    ((cyclotomic p ℤ).comp (X + 1)).IsEisensteinAt 𝓟 := by
  refine Monic.isEisensteinAt_of_mem_of_not_mem ?_
      (Ideal.IsPrime.ne_top <| (Ideal.span_singleton_prime (mod_cast hp.out.ne_zero)).2 <|
        Nat.prime_iff_prime_int.1 hp.out) (fun {i hi} => ?_) ?_
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ ((Polynomial.cyclotomic p Int).comp (HAdd.hAdd Polynomial.X 1)).Monic
    -/
  · rw [show (X + 1 : ℤ[X]) = X + C 1 by simp]
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ ((Polynomial.cyclotomic p Int).comp (HAdd.hAdd Polynomial.X (Polynomial.C 1) …
    -/
    refine (cyclotomic.monic p ℤ).comp (monic_X_add_C 1) fun h => ?_
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      h : Eq (HAdd.hAdd Polynomial.X (Polynomial.C 1)).natDegree 0
      ⊢ False
    -/
    rw [natDegree_X_add_C] at h
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      h : Eq 1 0
      ⊢ False
    -/
    exact zero_ne_one h.symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      i : Nat
      hi : LT.lt i ((Polynomial.cyclotomic p Int).comp (HAdd.hAdd Polynomial.X 1)).n …
      ⊢ Membership.mem (Submodule.span Int (Singleton.singleton ↑p)) (((Polynomial.c …
    -/
  · rw [cyclotomic_prime, geom_sum_X_comp_X_add_one_eq_sum, ← lcoeff_apply, map_sum]
    conv =>
      congr
      congr
      next => skip
      congr
      next => skip
      ext
      rw [lcoeff_apply, ← C_eq_natCast, C_mul_X_pow_eq_monomial, coeff_monomial]
    rw [natDegree_comp, show (X + 1 : ℤ[X]) = X + C 1 by simp, natDegree_X_add_C, mul_one,
      natDegree_cyclotomic, Nat.totient_prime hp.out] at hi
    simp only [hi.trans_le (Nat.sub_le _ _), sum_ite_eq', mem_range, if_true,
      Ideal.submodule_span_eq, Ideal.mem_span_singleton, Int.natCast_dvd_natCast]
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      i : Nat
      hi : LT.lt i (HSub.hSub p 1)
      ⊢ Dvd.dvd p (p.choose (HAdd.hAdd i 1))
    -/
    exact hp.out.dvd_choose_self i.succ_ne_zero (lt_tsub_iff_right.1 hi)
    /-
      🎉 no goals
    -/
  · rw [coeff_zero_eq_eval_zero, eval_comp, cyclotomic_prime, eval_add, eval_X, eval_one, zero_add,
      eval_geom_sum, one_geom_sum, Ideal.submodule_span_eq, Ideal.span_singleton_pow,
      Ideal.mem_span_singleton]
    /-
      case refine_3
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ Not (Dvd.dvd (HPow.hPow (↑p) 2) ↑p)
    -/
    intro h
    /-
      case refine_3
      p : Nat
      hp : Fact (Nat.Prime p)
      h : Dvd.dvd (HPow.hPow (↑p) 2) ↑p
      ⊢ False
    -/
    obtain ⟨k, hk⟩ := Int.natCast_dvd_natCast.1 h
    /-
      case refine_3.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      h : Dvd.dvd (HPow.hPow (↑p) 2) ↑p
      k : Nat
      hk : Eq p (HMul.hMul (HMul.hMul (HMul.hMul 1 p) p) k)
      ⊢ False
    -/
    rw [mul_assoc, mul_comm 1, mul_one] at hk
    /-
      case refine_3.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      h : Dvd.dvd (HPow.hPow (↑p) 2) ↑p
      k : Nat
      hk : Eq p (HMul.hMul p (HMul.hMul p k))
      ⊢ False
    -/
    nth_rw 1 [← Nat.mul_one p] at hk
    /-
      case refine_3.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      h : Dvd.dvd (HPow.hPow (↑p) 2) ↑p
      k : Nat
      hk : Eq (HMul.hMul p 1) (HMul.hMul p (HMul.hMul p k))
      ⊢ False
    -/
    rw [mul_right_inj' hp.out.ne_zero] at hk
    /-
      case refine_3.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      h : Dvd.dvd (HPow.hPow (↑p) 2) ↑p
      k : Nat
      hk : Eq 1 (HMul.hMul p k)
      ⊢ False
    -/
    exact Nat.Prime.not_dvd_one hp.out (Dvd.intro k hk.symm)
    /-
      🎉 no goals
    -/


theorem cyclotomic_prime_pow_comp_X_add_one_isEisensteinAt [hp : Fact p.Prime] (n : ℕ) :
    ((cyclotomic (p ^ (n + 1)) ℤ).comp (X + 1)).IsEisensteinAt 𝓟 := by
  refine Monic.isEisensteinAt_of_mem_of_not_mem ?_
      (Ideal.IsPrime.ne_top <| (Ideal.span_singleton_prime (mod_cast hp.out.ne_zero)).2 <|
        Nat.prime_iff_prime_int.1 hp.out) ?_ ?_
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n 1)) Int).comp (HAdd.hAdd P …
    -/
  · rw [show (X + 1 : ℤ[X]) = X + C 1 by simp]
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n 1)) Int).comp (HAdd.hAdd P …
    -/
    refine (cyclotomic.monic _ ℤ).comp (monic_X_add_C 1) fun h => ?_
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      h : Eq (HAdd.hAdd Polynomial.X (Polynomial.C 1)).natDegree 0
      ⊢ False
    -/
    rw [natDegree_X_add_C] at h
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      h : Eq 1 0
      ⊢ False
    -/
    exact zero_ne_one h.symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ ∀ {n_1 : Nat}, LT.lt n_1 ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n 1 …
    -/
  · induction' n with n hn
      /-
        case refine_2.zero
        p : Nat
        hp : Fact (Nat.Prime p)
        ⊢ ∀ {n : Nat}, LT.lt n ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd 0 1)) I …
      -/
    · intro i hi
      /-
        case refine_2.zero
        p : Nat
        hp : Fact (Nat.Prime p)
        i : Nat
        hi : LT.lt i ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd 0 1)) Int).comp ( …
        ⊢ Membership.mem (Submodule.span Int (Singleton.singleton ↑p)) (((Polynomial.c …
      -/
      rw [Nat.zero_add, pow_one] at hi ⊢
      /-
        case refine_2.zero
        p : Nat
        hp : Fact (Nat.Prime p)
        i : Nat
        hi : LT.lt i ((Polynomial.cyclotomic p Int).comp (HAdd.hAdd Polynomial.X 1)).n …
        ⊢ Membership.mem (Submodule.span Int (Singleton.singleton ↑p)) (((Polynomial.c …
      -/
      exact (cyclotomic_comp_X_add_one_isEisensteinAt p).mem hi
      /-
        🎉 no goals
      -/
      /-
        case refine_2.succ
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Nat
        hn : ∀ {n_1 : Nat}, LT.lt n_1 ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd  …
        ⊢ ∀ {n_1 : Nat}, LT.lt n_1 ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd (HA …
      -/
    · intro i hi
      rw [Ideal.submodule_span_eq, Ideal.mem_span_singleton, ← ZMod.intCast_zmod_eq_zero_iff_dvd,
        show ↑(_ : ℤ) = Int.castRingHom (ZMod p) _ by rfl, ← coeff_map, map_comp, map_cyclotomic,
        Polynomial.map_add, map_X, Polynomial.map_one, pow_add, pow_one,
        cyclotomic_mul_prime_dvd_eq_pow, pow_comp, ← ZMod.expand_card, coeff_expand hp.out.pos]
        /-
          case refine_2.succ
          p : Nat
          hp : Fact (Nat.Prime p)
          n : Nat
          hn : ∀ {n_1 : Nat}, LT.lt n_1 ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd  …
          i : Nat
          hi : LT.lt i ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n 1) 1 …
          ⊢ Eq (ite (Dvd.dvd p i) (((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n 1)) …
        -/
      · simp only [ite_eq_right_iff]
        /-
          case refine_2.succ
          p : Nat
          hp : Fact (Nat.Prime p)
          n : Nat
          hn : ∀ {n_1 : Nat}, LT.lt n_1 ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd  …
          i : Nat
          hi : LT.lt i ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n 1) 1 …
          ⊢ Dvd.dvd p i → Eq (((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n 1)) (ZMo …
        -/
        rintro ⟨k, hk⟩
        rw [natDegree_comp, show (X + 1 : ℤ[X]) = X + C 1 by simp, natDegree_X_add_C, mul_one,
          natDegree_cyclotomic, Nat.totient_prime_pow hp.out (Nat.succ_pos _), Nat.add_one_sub_one]
          at hn hi
        /-
          case refine_2.succ.intro
          p : Nat
          hp : Fact (Nat.Prime p)
          n : Nat
          hn : ∀ {n_1 : Nat}, LT.lt n_1 (HMul.hMul (HPow.hPow p n) (HSub.hSub p 1)) → Me …
          i : Nat
          hi : LT.lt i (HMul.hMul (HPow.hPow p (HAdd.hAdd n 1)) (HSub.hSub p 1))
          k : Nat
          hk : Eq i (HMul.hMul p k)
          ⊢ Eq (((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n 1)) (ZMod p)).comp (HA …
        -/
        rw [hk, pow_succ', mul_assoc] at hi
        /-
          case refine_2.succ.intro
          p : Nat
          hp : Fact (Nat.Prime p)
          n : Nat
          hn : ∀ {n_1 : Nat}, LT.lt n_1 (HMul.hMul (HPow.hPow p n) (HSub.hSub p 1)) → Me …
          i k : Nat
          hi : LT.lt (HMul.hMul p k) (HMul.hMul p (HMul.hMul (HPow.hPow p n) (HSub.hSub  …
          hk : Eq i (HMul.hMul p k)
          ⊢ Eq (((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n 1)) (ZMod p)).comp (HA …
        -/
        rw [hk, mul_comm, Nat.mul_div_cancel _ hp.out.pos]
        /-
          case refine_2.succ.intro
          p : Nat
          hp : Fact (Nat.Prime p)
          n : Nat
          hn : ∀ {n_1 : Nat}, LT.lt n_1 (HMul.hMul (HPow.hPow p n) (HSub.hSub p 1)) → Me …
          i k : Nat
          hi : LT.lt (HMul.hMul p k) (HMul.hMul p (HMul.hMul (HPow.hPow p n) (HSub.hSub  …
          hk : Eq i (HMul.hMul p k)
          ⊢ Eq (((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n 1)) (ZMod p)).comp (HA …
        -/
        replace hn := hn (lt_of_mul_lt_mul_left' hi)
        rw [Ideal.submodule_span_eq, Ideal.mem_span_singleton, ← ZMod.intCast_zmod_eq_zero_iff_dvd,
           show ↑(_ : ℤ) = Int.castRingHom (ZMod p) _ by rfl, ← coeff_map] at hn
        /-
          case refine_2.succ.intro
          p : Nat
          hp : Fact (Nat.Prime p)
          n i k : Nat
          hi : LT.lt (HMul.hMul p k) (HMul.hMul p (HMul.hMul (HPow.hPow p n) (HSub.hSub  …
          hk : Eq i (HMul.hMul p k)
          hn : Eq ((Polynomial.map (Int.castRingHom (ZMod p)) ((Polynomial.cyclotomic (H …
          ⊢ Eq (((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd n 1)) (ZMod p)).comp (HA …
        -/
        simpa [map_comp] using hn
        /-
          🎉 no goals
        -/
        /-
          case refine_2.succ.hn
          p : Nat
          hp : Fact (Nat.Prime p)
          n : Nat
          hn : ∀ {n_1 : Nat}, LT.lt n_1 ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd  …
          i : Nat
          hi : LT.lt i ((Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd (HAdd.hAdd n 1) 1 …
          ⊢ Dvd.dvd p (HPow.hPow p (HAdd.hAdd n 1))
        -/
      · exact ⟨p ^ n, by rw [pow_succ']⟩
        /-
          🎉 no goals
        -/
  · rw [coeff_zero_eq_eval_zero, eval_comp, cyclotomic_prime_pow_eq_geom_sum hp.out, eval_add,
      eval_X, eval_one, zero_add, eval_finset_sum]
    simp only [eval_pow, eval_X, one_pow, sum_const, card_range, Nat.smul_one_eq_cast,
      submodule_span_eq, Ideal.submodule_span_eq, Ideal.span_singleton_pow,
      Ideal.mem_span_singleton]
    /-
      case refine_3
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ Not (Dvd.dvd (HPow.hPow (↑p) 2) ↑p)
    -/
    intro h
    /-
      case refine_3
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      h : Dvd.dvd (HPow.hPow (↑p) 2) ↑p
      ⊢ False
    -/
    obtain ⟨k, hk⟩ := Int.natCast_dvd_natCast.1 h
    /-
      case refine_3.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      h : Dvd.dvd (HPow.hPow (↑p) 2) ↑p
      k : Nat
      hk : Eq p (HMul.hMul (HMul.hMul (HMul.hMul 1 p) p) k)
      ⊢ False
    -/
    rw [mul_assoc, mul_comm 1, mul_one] at hk
    /-
      case refine_3.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      h : Dvd.dvd (HPow.hPow (↑p) 2) ↑p
      k : Nat
      hk : Eq p (HMul.hMul p (HMul.hMul p k))
      ⊢ False
    -/
    nth_rw 1 [← Nat.mul_one p] at hk
    /-
      case refine_3.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      h : Dvd.dvd (HPow.hPow (↑p) 2) ↑p
      k : Nat
      hk : Eq (HMul.hMul p 1) (HMul.hMul p (HMul.hMul p k))
      ⊢ False
    -/
    rw [mul_right_inj' hp.out.ne_zero] at hk
    /-
      case refine_3.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      h : Dvd.dvd (HPow.hPow (↑p) 2) ↑p
      k : Nat
      hk : Eq 1 (HMul.hMul p k)
      ⊢ False
    -/
    exact Nat.Prime.not_dvd_one hp.out (Dvd.intro k hk.symm)
    /-
      🎉 no goals
    -/


local notation "𝓟" => Submodule.span R {(p : R)}


/-- Let `K` be the field of fraction of an integrally closed domain `R` and let `L` be a separable
extension of `K`, generated by an integral power basis `B` such that the minimal polynomial of
`B.gen` is Eisenstein at `p`. Given `z : L` integral over `R`, if `Q : R[X]` is such that
`aeval B.gen Q = p • z`, then `p ∣ Q.coeff 0`. -/
theorem dvd_coeff_zero_of_aeval_eq_prime_smul_of_minpoly_isEisensteinAt {B : PowerBasis K L}
    (hp : Prime p) (hBint : IsIntegral R B.gen) {z : L} {Q : R[X]} (hQ : aeval B.gen Q = p • z)
    (hzint : IsIntegral R z) (hei : (minpoly R B.gen).IsEisensteinAt 𝓟) : p ∣ Q.coeff 0 := by
  -- First define some abbreviations.
  /-
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    ⊢ Dvd.dvd p (Q.coeff 0)
  -/
  letI := B.finite
  /-
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    this : Module.Finite K L := PowerBasis.finite B
    ⊢ Dvd.dvd p (Q.coeff 0)
  -/
  let P := minpoly R B.gen
  /-
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    this : Module.Finite K L := PowerBasis.finite B
    P : Polynomial R := minpoly R B.gen
    ⊢ Dvd.dvd p (Q.coeff 0)
  -/
  obtain ⟨n, hn⟩ := Nat.exists_eq_succ_of_ne_zero B.dim_pos.ne'
  /-
    case intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    this : Module.Finite K L := PowerBasis.finite B
    P : Polynomial R := minpoly R B.gen
    n : Nat
    hn : Eq B.dim n.succ
    ⊢ Dvd.dvd p (Q.coeff 0)
  -/
  have finrank_K_L : Module.finrank K L = B.dim := B.finrank
  /-
    case intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    this : Module.Finite K L := PowerBasis.finite B
    P : Polynomial R := minpoly R B.gen
    n : Nat
    hn : Eq B.dim n.succ
    finrank_K_L : Eq (Module.finrank K L) B.dim
    ⊢ Dvd.dvd p (Q.coeff 0)
  -/
  have deg_K_P : (minpoly K B.gen).natDegree = B.dim := B.natDegree_minpoly
  have deg_R_P : P.natDegree = B.dim := by
    rw [← deg_K_P, minpoly.isIntegrallyClosed_eq_field_fractions' K hBint,
      (minpoly.monic hBint).natDegree_map (algebraMap R K)]
  choose! f hf using
    hei.isWeaklyEisensteinAt.exists_mem_adjoin_mul_eq_pow_natDegree_le (minpoly.aeval R B.gen)
      (minpoly.monic hBint)
  /-
    case intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    this : Module.Finite K L := PowerBasis.finite B
    P : Polynomial R := minpoly R B.gen
    n : Nat
    hn : Eq B.dim n.succ
    finrank_K_L : Eq (Module.finrank K L) B.dim
    deg_K_P : Eq (minpoly K B.gen).natDegree B.dim
    deg_R_P : Eq P.natDegree B.dim
    f : Nat → L
    hf : ∀ (i : Nat), LE.le (Polynomial.map (algebraMap R L) (minpoly R B.gen)).na …
    ⊢ Dvd.dvd p (Q.coeff 0)
  -/
  simp only [P, (minpoly.monic hBint).natDegree_map, deg_R_P] at hf

  -- The Eisenstein condition shows that `p` divides `Q.coeff 0`
  -- if `p^n.succ` divides the following multiple of `Q.coeff 0^n.succ`:
  suffices
      p ^ n.succ ∣ Q.coeff 0 ^ n.succ * ((-1) ^ (n.succ * n) * (minpoly R B.gen).coeff 0 ^ n) by
    have hndiv : ¬p ^ 2 ∣ (minpoly R B.gen).coeff 0 := fun h =>
      hei.not_mem ((span_singleton_pow p 2).symm ▸ Ideal.mem_span_singleton.2 h)
    refine @Prime.dvd_of_pow_dvd_pow_mul_pow_of_square_not_dvd R _ _ _ _ n hp (?_ : _ ∣ _) hndiv
    convert (IsUnit.dvd_mul_right ⟨(-1) ^ (n.succ * n), rfl⟩).mpr this using 1
    push_cast
    ring_nf
    rw [mul_comm _ 2, pow_mul, neg_one_sq, one_pow, mul_one]

  -- We claim the quotient of `Q^n * _` by `p^n` is the following `r`:
  have aux : ∀ i ∈ (range (Q.natDegree + 1)).erase 0, B.dim ≤ i + n := by
    intro i hi
    simp only [mem_range, mem_erase] at hi
    rw [hn]
    exact le_add_pred_of_pos _ hi.1
  have hintsum :
    IsIntegral R
      (z * B.gen ^ n - ∑ x ∈ (range (Q.natDegree + 1)).erase 0, Q.coeff x • f (x + n)) := by
    refine (hzint.mul (hBint.pow _)).sub (.sum _ fun i hi => .smul _ ?_)
    exact adjoin_le_integralClosure hBint (hf _ (aux i hi)).1
  /-
    case intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    this : Module.Finite K L := PowerBasis.finite B
    P : Polynomial R := minpoly R B.gen
    n : Nat
    hn : Eq B.dim n.succ
    finrank_K_L : Eq (Module.finrank K L) B.dim
    deg_K_P : Eq (minpoly K B.gen).natDegree B.dim
    deg_R_P : Eq P.natDegree B.dim
    f : Nat → L
    hf : ∀ (i : Nat), LE.le B.dim i → And (Membership.mem (Algebra.adjoin R (Singl …
    aux : ∀ (i : Nat), Membership.mem ((Finset.range (HAdd.hAdd Q.natDegree 1)).er …
    hintsum : IsIntegral R (HSub.hSub (HMul.hMul z (HPow.hPow B.gen n)) (((Finset. …
    ⊢ Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow (Q.coeff 0) n.succ) (HMul …
  -/
  obtain ⟨r, hr⟩ := isIntegral_iff.1 (isIntegral_norm K hintsum)
  /-
    case intro.intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    this : Module.Finite K L := PowerBasis.finite B
    P : Polynomial R := minpoly R B.gen
    n : Nat
    hn : Eq B.dim n.succ
    finrank_K_L : Eq (Module.finrank K L) B.dim
    deg_K_P : Eq (minpoly K B.gen).natDegree B.dim
    deg_R_P : Eq P.natDegree B.dim
    f : Nat → L
    hf : ∀ (i : Nat), LE.le B.dim i → And (Membership.mem (Algebra.adjoin R (Singl …
    aux : ∀ (i : Nat), Membership.mem ((Finset.range (HAdd.hAdd Q.natDegree 1)).er …
    hintsum : IsIntegral R (HSub.hSub (HMul.hMul z (HPow.hPow B.gen n)) (((Finset. …
    r : R
    hr : Eq ((algebraMap R K) r) ((Algebra.norm K) (HSub.hSub (HMul.hMul z (HPow.h …
    ⊢ Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow (Q.coeff 0) n.succ) (HMul …
  -/
  use r

  -- Do the computation in `K` so we can work in terms of `z` instead of `r`.
  /-
    case h
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    this : Module.Finite K L := PowerBasis.finite B
    P : Polynomial R := minpoly R B.gen
    n : Nat
    hn : Eq B.dim n.succ
    finrank_K_L : Eq (Module.finrank K L) B.dim
    deg_K_P : Eq (minpoly K B.gen).natDegree B.dim
    deg_R_P : Eq P.natDegree B.dim
    f : Nat → L
    hf : ∀ (i : Nat), LE.le B.dim i → And (Membership.mem (Algebra.adjoin R (Singl …
    aux : ∀ (i : Nat), Membership.mem ((Finset.range (HAdd.hAdd Q.natDegree 1)).er …
    hintsum : IsIntegral R (HSub.hSub (HMul.hMul z (HPow.hPow B.gen n)) (((Finset. …
    r : R
    hr : Eq ((algebraMap R K) r) ((Algebra.norm K) (HSub.hSub (HMul.hMul z (HPow.h …
    ⊢ Eq (HMul.hMul (HPow.hPow (Q.coeff 0) n.succ) (HMul.hMul (HPow.hPow (-1) (HMu …
  -/
  apply IsFractionRing.injective R K
  /-
    case h.a
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    this : Module.Finite K L := PowerBasis.finite B
    P : Polynomial R := minpoly R B.gen
    n : Nat
    hn : Eq B.dim n.succ
    finrank_K_L : Eq (Module.finrank K L) B.dim
    deg_K_P : Eq (minpoly K B.gen).natDegree B.dim
    deg_R_P : Eq P.natDegree B.dim
    f : Nat → L
    hf : ∀ (i : Nat), LE.le B.dim i → And (Membership.mem (Algebra.adjoin R (Singl …
    aux : ∀ (i : Nat), Membership.mem ((Finset.range (HAdd.hAdd Q.natDegree 1)).er …
    hintsum : IsIntegral R (HSub.hSub (HMul.hMul z (HPow.hPow B.gen n)) (((Finset. …
    r : R
    hr : Eq ((algebraMap R K) r) ((Algebra.norm K) (HSub.hSub (HMul.hMul z (HPow.h …
    ⊢ Eq ((algebraMap R K) (HMul.hMul (HPow.hPow (Q.coeff 0) n.succ) (HMul.hMul (H …
  -/
  simp only [_root_.map_mul, map_pow, map_neg, map_one]
  -- Both sides are actually norms:
  calc
    _ = norm K (Q.coeff 0 • B.gen ^ n) := ?_
    _ = norm K (p • (z * B.gen ^ n) -
          ∑ x ∈ (range (Q.natDegree + 1)).erase 0, p • Q.coeff x • f (x + n)) :=
        (congr_arg (norm K) (eq_sub_of_add_eq ?_))
    _ = _ := ?_
  · simp only [Algebra.smul_def, algebraMap_apply R K L, Algebra.norm_algebraMap, _root_.map_mul,
      map_pow, finrank_K_L, PowerBasis.norm_gen_eq_coeff_zero_minpoly,
      minpoly.isIntegrallyClosed_eq_field_fractions' K hBint, coeff_map, ← hn]
    /-
      case h.a.calc_1
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      Q : Polynomial R
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hzint : IsIntegral R z
      hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
      this : Module.Finite K L := PowerBasis.finite B
      P : Polynomial R := minpoly R B.gen
      n : Nat
      hn : Eq B.dim n.succ
      finrank_K_L : Eq (Module.finrank K L) B.dim
      deg_K_P : Eq (minpoly K B.gen).natDegree B.dim
      deg_R_P : Eq P.natDegree B.dim
      f : Nat → L
      hf : ∀ (i : Nat), LE.le B.dim i → And (Membership.mem (Algebra.adjoin R (Singl …
      aux : ∀ (i : Nat), Membership.mem ((Finset.range (HAdd.hAdd Q.natDegree 1)).er …
      hintsum : IsIntegral R (HSub.hSub (HMul.hMul z (HPow.hPow B.gen n)) (((Finset. …
      r : R
      hr : Eq ((algebraMap R K) r) ((Algebra.norm K) (HSub.hSub (HMul.hMul z (HPow.h …
      ⊢ Eq (HMul.hMul (HPow.hPow ((algebraMap R K) (Q.coeff 0)) B.dim) (HMul.hMul (H …
    -/
    ring
    /-
      🎉 no goals
    -/
  /-
    case h.a.calc_2
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    Q : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hzint : IsIntegral R z
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    this : Module.Finite K L := PowerBasis.finite B
    P : Polynomial R := minpoly R B.gen
    n : Nat
    hn : Eq B.dim n.succ
    finrank_K_L : Eq (Module.finrank K L) B.dim
    deg_K_P : Eq (minpoly K B.gen).natDegree B.dim
    deg_R_P : Eq P.natDegree B.dim
    f : Nat → L
    hf : ∀ (i : Nat), LE.le B.dim i → And (Membership.mem (Algebra.adjoin R (Singl …
    aux : ∀ (i : Nat), Membership.mem ((Finset.range (HAdd.hAdd Q.natDegree 1)).er …
    hintsum : IsIntegral R (HSub.hSub (HMul.hMul z (HPow.hPow B.gen n)) (((Finset. …
    r : R
    hr : Eq ((algebraMap R K) r) ((Algebra.norm K) (HSub.hSub (HMul.hMul z (HPow.h …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Q.coeff 0) (HPow.hPow B.gen n)) (((Finset.range  …
  -/
  swap
  · simp_rw [← smul_sum, ← smul_sub, Algebra.smul_def p, algebraMap_apply R K L, _root_.map_mul,
      Algebra.norm_algebraMap, finrank_K_L, hr, ← hn]
  calc
    _ = (Q.coeff 0 • ↑1 + ∑ x ∈ (range (Q.natDegree + 1)).erase 0, Q.coeff x • B.gen ^ x) *
          B.gen ^ n := ?_
    _ = (Q.coeff 0 • B.gen ^ 0 +
        ∑ x ∈ (range (Q.natDegree + 1)).erase 0, Q.coeff x • B.gen ^ x) * B.gen ^ n := by
      rw [pow_zero]
    _ = aeval B.gen Q * B.gen ^ n := ?_
    _ = _ := by rw [hQ, Algebra.smul_mul_assoc]
  · have : ∀ i ∈ (range (Q.natDegree + 1)).erase 0,
        Q.coeff i • (B.gen ^ i * B.gen ^ n) = p • Q.coeff i • f (i + n) := by
      intro i hi
      rw [← pow_add, ← (hf _ (aux i hi)).2, ← Algebra.smul_def, smul_smul, mul_comm _ p, smul_smul]
    /-
      case h.a.calc_2.calc_1
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      Q : Polynomial R
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hzint : IsIntegral R z
      hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
      this✝ : Module.Finite K L := PowerBasis.finite B
      P : Polynomial R := minpoly R B.gen
      n : Nat
      hn : Eq B.dim n.succ
      finrank_K_L : Eq (Module.finrank K L) B.dim
      deg_K_P : Eq (minpoly K B.gen).natDegree B.dim
      deg_R_P : Eq P.natDegree B.dim
      f : Nat → L
      hf : ∀ (i : Nat), LE.le B.dim i → And (Membership.mem (Algebra.adjoin R (Singl …
      aux : ∀ (i : Nat), Membership.mem ((Finset.range (HAdd.hAdd Q.natDegree 1)).er …
      hintsum : IsIntegral R (HSub.hSub (HMul.hMul z (HPow.hPow B.gen n)) (((Finset. …
      r : R
      hr : Eq ((algebraMap R K) r) ((Algebra.norm K) (HSub.hSub (HMul.hMul z (HPow.h …
      this : ∀ (i : Nat), Membership.mem ((Finset.range (HAdd.hAdd Q.natDegree 1)).e …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Q.coeff 0) (HPow.hPow B.gen n)) (((Finset.range  …
    -/
    simp only [add_mul, smul_mul_assoc, one_mul, sum_mul, sum_congr rfl this]
    /-
      🎉 no goals
    -/
  · rw [aeval_eq_sum_range,
      Finset.add_sum_erase (range (Q.natDegree + 1)) fun i => Q.coeff i • B.gen ^ i]
    /-
      case h.a.calc_2.calc_2
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      Q : Polynomial R
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hzint : IsIntegral R z
      hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
      this : Module.Finite K L := PowerBasis.finite B
      P : Polynomial R := minpoly R B.gen
      n : Nat
      hn : Eq B.dim n.succ
      finrank_K_L : Eq (Module.finrank K L) B.dim
      deg_K_P : Eq (minpoly K B.gen).natDegree B.dim
      deg_R_P : Eq P.natDegree B.dim
      f : Nat → L
      hf : ∀ (i : Nat), LE.le B.dim i → And (Membership.mem (Algebra.adjoin R (Singl …
      aux : ∀ (i : Nat), Membership.mem ((Finset.range (HAdd.hAdd Q.natDegree 1)).er …
      hintsum : IsIntegral R (HSub.hSub (HMul.hMul z (HPow.hPow B.gen n)) (((Finset. …
      r : R
      hr : Eq ((algebraMap R K) r) ((Algebra.norm K) (HSub.hSub (HMul.hMul z (HPow.h …
      ⊢ Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) 0
    -/
    simp
    /-
      🎉 no goals
    -/


theorem mem_adjoin_of_dvd_coeff_of_dvd_aeval {A B : Type*} [CommSemiring A] [CommRing B]
    [Algebra A B] [NoZeroSMulDivisors A B] {Q : A[X]} {p : A} {x z : B} (hp : p ≠ 0)
    (hQ : ∀ i ∈ range (Q.natDegree + 1), p ∣ Q.coeff i) (hz : aeval x Q = p • z) :
    z ∈ adjoin A ({x} : Set B) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommSemiring A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : NoZeroSMulDivisors A B
    Q : Polynomial A
    p : A
    x z : B
    hp : Ne p 0
    hQ : ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) i →  …
    hz : Eq ((Polynomial.aeval x) Q) (HSMul.hSMul p z)
    ⊢ Membership.mem (Algebra.adjoin A (Singleton.singleton x)) z
  -/
  choose! f hf using hQ
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommSemiring A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : NoZeroSMulDivisors A B
    Q : Polynomial A
    p : A
    x z : B
    hp : Ne p 0
    hz : Eq ((Polynomial.aeval x) Q) (HSMul.hSMul p z)
    f : Nat → A
    hf : ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) i →  …
    ⊢ Membership.mem (Algebra.adjoin A (Singleton.singleton x)) z
  -/
  rw [aeval_eq_sum_range, sum_range] at hz
  conv_lhs at hz =>
    congr
    next => skip
    ext i
    rw [hf i (mem_range.2 (Fin.is_lt i)), ← smul_smul]
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommSemiring A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : NoZeroSMulDivisors A B
    Q : Polynomial A
    p : A
    x z : B
    hp : Ne p 0
    f : Nat → A
    hz : Eq (Finset.univ.sum fun i => HSMul.hSMul p (HSMul.hSMul (f ↑i) (HPow.hPow …
    hf : ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) i →  …
    ⊢ Membership.mem (Algebra.adjoin A (Singleton.singleton x)) z
  -/
  rw [← smul_sum] at hz
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommSemiring A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : NoZeroSMulDivisors A B
    Q : Polynomial A
    p : A
    x z : B
    hp : Ne p 0
    f : Nat → A
    hz : Eq (HSMul.hSMul p (Finset.univ.sum fun x_1 => HSMul.hSMul (f ↑x_1) (HPow. …
    hf : ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) i →  …
    ⊢ Membership.mem (Algebra.adjoin A (Singleton.singleton x)) z
  -/
  rw [← smul_right_injective _ hp hz]
  exact
    Subalgebra.sum_mem _ fun _ _ =>
      Subalgebra.smul_mem _ (Subalgebra.pow_mem _ (subset_adjoin (Set.mem_singleton _)) _) _


/-- Let `K` be the field of fraction of an integrally closed domain `R` and let `L` be a separable
extension of `K`, generated by an integral power basis `B` such that the minimal polynomial of
`B.gen` is Eisenstein at `p`. Given `z : L` integral over `R`, if `p • z ∈ adjoin R {B.gen}`, then
`z ∈ adjoin R {B.gen}`. -/
theorem mem_adjoin_of_smul_prime_smul_of_minpoly_isEisensteinAt {B : PowerBasis K L}
    (hp : Prime p) (hBint : IsIntegral R B.gen) {z : L} (hzint : IsIntegral R z)
    (hz : p • z ∈ adjoin R ({B.gen} : Set L)) (hei : (minpoly R B.gen).IsEisensteinAt 𝓟) :
    z ∈ adjoin R ({B.gen} : Set L) := by
  -- First define some abbreviations.
  have hndiv : ¬p ^ 2 ∣ (minpoly R B.gen).coeff 0 := fun h =>
    hei.not_mem ((span_singleton_pow p 2).symm ▸ Ideal.mem_span_singleton.2 h)
  /-
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    hz : Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMu …
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) ((minpoly R B.gen).coeff 0))
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  have := B.finite
  /-
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    hz : Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMu …
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) ((minpoly R B.gen).coeff 0))
    this : Module.Finite K L
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  set P := minpoly R B.gen with hP
  /-
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    hz : Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMu …
    this : Module.Finite K L
    P : Polynomial R := minpoly R B.gen
    hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
    hP : Eq P (minpoly R B.gen)
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  obtain ⟨n, hn⟩ := Nat.exists_eq_succ_of_ne_zero B.dim_pos.ne'
  /-
    case intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    hz : Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMu …
    this : Module.Finite K L
    P : Polynomial R := minpoly R B.gen
    hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
    hP : Eq P (minpoly R B.gen)
    n : Nat
    hn : Eq B.dim n.succ
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  haveI : NoZeroSMulDivisors R L := NoZeroSMulDivisors.trans R K L
  /-
    case intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    hz : Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMu …
    this✝ : Module.Finite K L
    P : Polynomial R := minpoly R B.gen
    hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
    hP : Eq P (minpoly R B.gen)
    n : Nat
    hn : Eq B.dim n.succ
    this : NoZeroSMulDivisors R L
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  let _ := P.map (algebraMap R L)
  -- There is a polynomial `Q` such that `p • z = aeval B.gen Q`. We can assume that
  -- `Q.degree < P.degree` and `Q ≠ 0`.
  /-
    case intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    hz : Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMu …
    this✝ : Module.Finite K L
    P : Polynomial R := minpoly R B.gen
    hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
    hP : Eq P (minpoly R B.gen)
    n : Nat
    hn : Eq B.dim n.succ
    this : NoZeroSMulDivisors R L
    x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  rw [adjoin_singleton_eq_range_aeval] at hz
  /-
    case intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    hz : Membership.mem (Polynomial.aeval B.gen).range (HSMul.hSMul p z)
    this✝ : Module.Finite K L
    P : Polynomial R := minpoly R B.gen
    hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
    hP : Eq P (minpoly R B.gen)
    n : Nat
    hn : Eq B.dim n.succ
    this : NoZeroSMulDivisors R L
    x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  obtain ⟨Q₁, hQ⟩ := hz
  /-
    case intro.intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    this✝ : Module.Finite K L
    P : Polynomial R := minpoly R B.gen
    hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
    hP : Eq P (minpoly R B.gen)
    n : Nat
    hn : Eq B.dim n.succ
    this : NoZeroSMulDivisors R L
    x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
    Q₁ : Polynomial R
    hQ : Eq ((Polynomial.aeval B.gen).toRingHom Q₁) (HSMul.hSMul p z)
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  set Q := Q₁ %ₘ P with hQ₁
  replace hQ : aeval B.gen Q = p • z := by
    rw [← modByMonic_add_div Q₁ (minpoly.monic hBint)] at hQ
    simpa using hQ
  /-
    case intro.intro
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    this✝ : Module.Finite K L
    P : Polynomial R := minpoly R B.gen
    hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
    hP : Eq P (minpoly R B.gen)
    n : Nat
    hn : Eq B.dim n.succ
    this : NoZeroSMulDivisors R L
    x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
    Q₁ : Polynomial R
    Q : Polynomial R := Q₁.modByMonic P
    hQ₁ : Eq Q (Q₁.modByMonic P)
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  by_cases hQzero : Q = 0
    /-
      case pos
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hQzero : Eq Q 0
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
    -/
  · simp only [hQzero, Algebra.smul_def, zero_eq_mul, aeval_zero] at hQ
    /-
      case pos
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQzero : Eq Q 0
      hQ : Or (Eq ((algebraMap R L) p) 0) (Eq z 0)
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
    -/
    cases' hQ with H H₁
    · have : Function.Injective (algebraMap R L) := by
        rw [algebraMap_eq R K L]
        exact (algebraMap K L).injective.comp (IsFractionRing.injective R K)
      /-
        case pos.inl
        R : Type u
        K : Type v
        L : Type z
        p : R
        inst✝¹⁰ : CommRing R
        inst✝⁹ : Field K
        inst✝⁸ : Field L
        inst✝⁷ : Algebra K L
        inst✝⁶ : Algebra R L
        inst✝⁵ : Algebra R K
        inst✝⁴ : IsScalarTower R K L
        inst✝³ : Algebra.IsSeparable K L
        inst✝² : IsDomain R
        inst✝¹ : IsFractionRing R K
        inst✝ : IsIntegrallyClosed R
        B : PowerBasis K L
        hp : Prime p
        hBint : IsIntegral R B.gen
        z : L
        hzint : IsIntegral R z
        this✝¹ : Module.Finite K L
        P : Polynomial R := minpoly R B.gen
        hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
        hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
        hP : Eq P (minpoly R B.gen)
        n : Nat
        hn : Eq B.dim n.succ
        this✝ : NoZeroSMulDivisors R L
        x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
        Q₁ : Polynomial R
        Q : Polynomial R := Q₁.modByMonic P
        hQ₁ : Eq Q (Q₁.modByMonic P)
        hQzero : Eq Q 0
        H : Eq ((algebraMap R L) p) 0
        this : Function.Injective ⇑(algebraMap R L)
        ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
      -/
      exfalso
      /-
        case pos.inl
        R : Type u
        K : Type v
        L : Type z
        p : R
        inst✝¹⁰ : CommRing R
        inst✝⁹ : Field K
        inst✝⁸ : Field L
        inst✝⁷ : Algebra K L
        inst✝⁶ : Algebra R L
        inst✝⁵ : Algebra R K
        inst✝⁴ : IsScalarTower R K L
        inst✝³ : Algebra.IsSeparable K L
        inst✝² : IsDomain R
        inst✝¹ : IsFractionRing R K
        inst✝ : IsIntegrallyClosed R
        B : PowerBasis K L
        hp : Prime p
        hBint : IsIntegral R B.gen
        z : L
        hzint : IsIntegral R z
        this✝¹ : Module.Finite K L
        P : Polynomial R := minpoly R B.gen
        hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
        hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
        hP : Eq P (minpoly R B.gen)
        n : Nat
        hn : Eq B.dim n.succ
        this✝ : NoZeroSMulDivisors R L
        x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
        Q₁ : Polynomial R
        Q : Polynomial R := Q₁.modByMonic P
        hQ₁ : Eq Q (Q₁.modByMonic P)
        hQzero : Eq Q 0
        H : Eq ((algebraMap R L) p) 0
        this : Function.Injective ⇑(algebraMap R L)
        ⊢ False
      -/
      exact hp.ne_zero ((injective_iff_map_eq_zero _).1 this _ H)
      /-
        🎉 no goals
      -/
      /-
        case pos.inr
        R : Type u
        K : Type v
        L : Type z
        p : R
        inst✝¹⁰ : CommRing R
        inst✝⁹ : Field K
        inst✝⁸ : Field L
        inst✝⁷ : Algebra K L
        inst✝⁶ : Algebra R L
        inst✝⁵ : Algebra R K
        inst✝⁴ : IsScalarTower R K L
        inst✝³ : Algebra.IsSeparable K L
        inst✝² : IsDomain R
        inst✝¹ : IsFractionRing R K
        inst✝ : IsIntegrallyClosed R
        B : PowerBasis K L
        hp : Prime p
        hBint : IsIntegral R B.gen
        z : L
        hzint : IsIntegral R z
        this✝ : Module.Finite K L
        P : Polynomial R := minpoly R B.gen
        hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
        hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
        hP : Eq P (minpoly R B.gen)
        n : Nat
        hn : Eq B.dim n.succ
        this : NoZeroSMulDivisors R L
        x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
        Q₁ : Polynomial R
        Q : Polynomial R := Q₁.modByMonic P
        hQ₁ : Eq Q (Q₁.modByMonic P)
        hQzero : Eq Q 0
        H₁ : Eq z 0
        ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
      -/
    · rw [H₁]
      /-
        case pos.inr
        R : Type u
        K : Type v
        L : Type z
        p : R
        inst✝¹⁰ : CommRing R
        inst✝⁹ : Field K
        inst✝⁸ : Field L
        inst✝⁷ : Algebra K L
        inst✝⁶ : Algebra R L
        inst✝⁵ : Algebra R K
        inst✝⁴ : IsScalarTower R K L
        inst✝³ : Algebra.IsSeparable K L
        inst✝² : IsDomain R
        inst✝¹ : IsFractionRing R K
        inst✝ : IsIntegrallyClosed R
        B : PowerBasis K L
        hp : Prime p
        hBint : IsIntegral R B.gen
        z : L
        hzint : IsIntegral R z
        this✝ : Module.Finite K L
        P : Polynomial R := minpoly R B.gen
        hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
        hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
        hP : Eq P (minpoly R B.gen)
        n : Nat
        hn : Eq B.dim n.succ
        this : NoZeroSMulDivisors R L
        x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
        Q₁ : Polynomial R
        Q : Polynomial R := Q₁.modByMonic P
        hQ₁ : Eq Q (Q₁.modByMonic P)
        hQzero : Eq Q 0
        H₁ : Eq z 0
        ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) 0
      -/
      exact Subalgebra.zero_mem _
      /-
        🎉 no goals
      -/
  -- It is enough to prove that all coefficients of `Q` are divisible by `p`, by induction.
  -- The base case is `dvd_coeff_zero_of_aeval_eq_prime_smul_of_minpoly_isEisensteinAt`.
  /-
    case neg
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    this✝ : Module.Finite K L
    P : Polynomial R := minpoly R B.gen
    hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
    hP : Eq P (minpoly R B.gen)
    n : Nat
    hn : Eq B.dim n.succ
    this : NoZeroSMulDivisors R L
    x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
    Q₁ : Polynomial R
    Q : Polynomial R := Q₁.modByMonic P
    hQ₁ : Eq Q (Q₁.modByMonic P)
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hQzero : Not (Eq Q 0)
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  refine mem_adjoin_of_dvd_coeff_of_dvd_aeval hp.ne_zero (fun i => ?_) hQ
  /-
    case neg
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    z : L
    hzint : IsIntegral R z
    this✝ : Module.Finite K L
    P : Polynomial R := minpoly R B.gen
    hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
    hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
    hP : Eq P (minpoly R B.gen)
    n : Nat
    hn : Eq B.dim n.succ
    this : NoZeroSMulDivisors R L
    x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
    Q₁ : Polynomial R
    Q : Polynomial R := Q₁.modByMonic P
    hQ₁ : Eq Q (Q₁.modByMonic P)
    hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
    hQzero : Not (Eq Q 0)
    i : Nat
    ⊢ Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) i → Dvd.dvd p (Q.coe …
  -/
  induction' i using Nat.case_strong_induction_on with j hind
    /-
      case neg.hz
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hQzero : Not (Eq Q 0)
      ⊢ Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) 0 → Dvd.dvd p (Q.coe …
    -/
  · intro _
    /-
      case neg.hz
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hQzero : Not (Eq Q 0)
      a✝ : Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) 0
      ⊢ Dvd.dvd p (Q.coeff 0)
    -/
    exact dvd_coeff_zero_of_aeval_eq_prime_smul_of_minpoly_isEisensteinAt hp hBint hQ hzint hei
    /-
      🎉 no goals
    -/
    /-
      case neg.hi
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hQzero : Not (Eq Q 0)
      j : Nat
      hind : ∀ (m : Nat), LE.le m j → Membership.mem (Finset.range (HAdd.hAdd Q.natD …
      ⊢ Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) (HAdd.hAdd j 1) → Dv …
    -/
  · intro hj
    /-
      case neg.hi
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hQzero : Not (Eq Q 0)
      j : Nat
      hind : ∀ (m : Nat), LE.le m j → Membership.mem (Finset.range (HAdd.hAdd Q.natD …
      hj : Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) (HAdd.hAdd j 1)
      ⊢ Dvd.dvd p (Q.coeff (HAdd.hAdd j 1))
    -/
    convert hp.dvd_of_pow_dvd_pow_mul_pow_of_square_not_dvd (n := n) _ hndiv
    -- Two technical results we will need about `P.natDegree` and `Q.natDegree`.
    /-
      case neg.hi.convert_2
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hQzero : Not (Eq Q 0)
      j : Nat
      hind : ∀ (m : Nat), LE.le m j → Membership.mem (Finset.range (HAdd.hAdd Q.natD …
      hj : Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) (HAdd.hAdd j 1)
      ⊢ Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow (Q.coeff (HAdd.hAdd j 1)) …
    -/
    have H := degree_modByMonic_lt Q₁ (minpoly.monic hBint)
    /-
      case neg.hi.convert_2
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hQzero : Not (Eq Q 0)
      j : Nat
      hind : ∀ (m : Nat), LE.le m j → Membership.mem (Finset.range (HAdd.hAdd Q.natD …
      hj : Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) (HAdd.hAdd j 1)
      H : LT.lt (Q₁.modByMonic (minpoly R B.gen)).degree (minpoly R B.gen).degree
      ⊢ Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow (Q.coeff (HAdd.hAdd j 1)) …
    -/
    rw [← hQ₁, ← hP] at H
    replace H := Nat.lt_iff_add_one_le.1
      (lt_of_lt_of_le
        (lt_of_le_of_lt (Nat.lt_iff_add_one_le.1 (Nat.lt_of_succ_lt_succ (mem_range.1 hj)))
          (lt_succ_self _)) (Nat.lt_iff_add_one_le.1 ((natDegree_lt_natDegree_iff hQzero).2 H)))
    have Hj : Q.natDegree + 1 = j + 1 + (Q.natDegree - j) := by
      rw [← add_comm 1, ← add_comm 1, add_assoc, add_right_inj,
        ← Nat.add_sub_assoc (Nat.lt_of_succ_lt_succ (mem_range.1 hj)).le, add_comm,
        Nat.add_sub_cancel]
    -- By induction hypothesis we can find `g : ℕ → R` such that
    -- `k ∈ range (j + 1) → Q.coeff k • B.gen ^ k = (algebraMap R L) p * g k • B.gen ^ k`-
    /-
      case neg.hi.convert_2
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQ : Eq ((Polynomial.aeval B.gen) Q) (HSMul.hSMul p z)
      hQzero : Not (Eq Q 0)
      j : Nat
      hind : ∀ (m : Nat), LE.le m j → Membership.mem (Finset.range (HAdd.hAdd Q.natD …
      hj : Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) (HAdd.hAdd j 1)
      H : LE.le (HAdd.hAdd (HAdd.hAdd j 1) 1) P.natDegree
      Hj : Eq (HAdd.hAdd Q.natDegree 1) (HAdd.hAdd (HAdd.hAdd j 1) (HSub.hSub Q.natD …
      ⊢ Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow (Q.coeff (HAdd.hAdd j 1)) …
    -/
    choose! g hg using hind
    replace hg : ∀ k ∈ range (j + 1), Q.coeff k • B.gen ^ k =
        algebraMap R L p * g k • B.gen ^ k := by
      intro k hk
      rw [hg k (mem_range_succ_iff.1 hk)
        (mem_range_succ_iff.2
          (le_trans (mem_range_succ_iff.1 hk) (succ_le_iff.1 (mem_range_succ_iff.1 hj)).le)),
        Algebra.smul_def, Algebra.smul_def, RingHom.map_mul, mul_assoc]
    -- Since `minpoly R B.gen` is Eiseinstein, we can find `f : ℕ → L` such that
    -- `(map (algebraMap R L) (minpoly R B.gen)).nat_degree ≤ i` implies `f i ∈ adjoin R {B.gen}`
    -- and `(algebraMap R L) p * f i = B.gen ^ i`. We will also need `hf₁`, a reformulation of this
    -- property.
    choose! f hf using
      IsWeaklyEisensteinAt.exists_mem_adjoin_mul_eq_pow_natDegree_le (minpoly.aeval R B.gen)
        (minpoly.monic hBint) hei.isWeaklyEisensteinAt
    have hf₁ : ∀ k ∈ (range (Q.natDegree - j)).erase 0,
        Q.coeff (j + 1 + k) • B.gen ^ (j + 1 + k) * B.gen ^ (P.natDegree - (j + 2)) =
        (algebraMap R L) p * Q.coeff (j + 1 + k) • f (k + P.natDegree - 1) := by
      intro k hk
      rw [smul_mul_assoc, ← pow_add, ← Nat.add_sub_assoc H, add_comm (j + 1) 1,
        add_assoc (j + 1), add_comm _ (k + P.natDegree), Nat.add_sub_add_right,
        ← (hf (k + P.natDegree - 1) _).2, mul_smul_comm]
      rw [(minpoly.monic hBint).natDegree_map, add_comm, Nat.add_sub_assoc, le_add_iff_nonneg_right]
      · exact Nat.zero_le _
      · refine one_le_iff_ne_zero.2 fun h => ?_
        rw [h] at hk
        simp at hk

    -- The Eisenstein condition shows that `p` divides `Q.coeff j`
    -- if `p^n.succ` divides the following multiple of `Q.coeff (succ j)^n.succ`:
    suffices
        p ^ n.succ ∣ Q.coeff (succ j) ^ n.succ *
          (minpoly R B.gen).coeff 0 ^ (succ j + (P.natDegree - (j + 2))) by
      convert this
      rw [Nat.succ_eq_add_one, add_assoc, ← Nat.add_sub_assoc H, add_comm (j + 1),
        Nat.add_sub_add_left, ← Nat.add_sub_assoc, Nat.add_sub_add_left, hP, ←
        (minpoly.monic hBint).natDegree_map (algebraMap R K), ←
        minpoly.isIntegrallyClosed_eq_field_fractions' K hBint, natDegree_minpoly, hn, Nat.sub_one,
        Nat.pred_succ]
      omega

    -- Using `hQ : aeval B.gen Q = p • z`, we write `p • z` as a sum of terms of degree less than
    -- `j+1`, that are multiples of `p` by induction, and terms of degree at least `j+1`.
    rw [aeval_eq_sum_range, Hj, range_add, sum_union (disjoint_range_addLeftEmbedding _ _),
      sum_congr rfl hg, add_comm] at hQ
    -- We multiply this equality by `B.gen ^ (P.natDegree-(j+2))`, so we can use `hf₁` on the terms
    -- we didn't know were multiples of `p`, and we take the norm on both sides.
    /-
      case neg.hi.convert_2
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQzero : Not (Eq Q 0)
      j : Nat
      hj : Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) (HAdd.hAdd j 1)
      H : LE.le (HAdd.hAdd (HAdd.hAdd j 1) 1) P.natDegree
      Hj : Eq (HAdd.hAdd Q.natDegree 1) (HAdd.hAdd (HAdd.hAdd j 1) (HSub.hSub Q.natD …
      g : Nat → R
      hQ : Eq (HAdd.hAdd ((Finset.map (addLeftEmbedding (HAdd.hAdd j 1)) (Finset.ran …
      hg : ∀ (k : Nat), Membership.mem (Finset.range (HAdd.hAdd j 1)) k → Eq (HSMul. …
      f : Nat → L
      hf : ∀ (i : Nat), LE.le (Polynomial.map (algebraMap R L) (minpoly R B.gen)).na …
      hf₁ : ∀ (k : Nat), Membership.mem ((Finset.range (HSub.hSub Q.natDegree j)).er …
      ⊢ Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow (Q.coeff j.succ) n.succ)  …
    -/
    replace hQ := congr_arg (fun x => x * B.gen ^ (P.natDegree - (j + 2))) hQ
    /-
      case neg.hi.convert_2
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQzero : Not (Eq Q 0)
      j : Nat
      hj : Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) (HAdd.hAdd j 1)
      H : LE.le (HAdd.hAdd (HAdd.hAdd j 1) 1) P.natDegree
      Hj : Eq (HAdd.hAdd Q.natDegree 1) (HAdd.hAdd (HAdd.hAdd j 1) (HSub.hSub Q.natD …
      g : Nat → R
      hg : ∀ (k : Nat), Membership.mem (Finset.range (HAdd.hAdd j 1)) k → Eq (HSMul. …
      f : Nat → L
      hf : ∀ (i : Nat), LE.le (Polynomial.map (algebraMap R L) (minpoly R B.gen)).na …
      hf₁ : ∀ (k : Nat), Membership.mem ((Finset.range (HSub.hSub Q.natDegree j)).er …
      hQ : Eq ((fun x => HMul.hMul x (HPow.hPow B.gen (HSub.hSub P.natDegree (HAdd.h …
      ⊢ Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow (Q.coeff j.succ) n.succ)  …
    -/
    simp_rw [sum_map, addLeftEmbedding_apply, add_mul, sum_mul, mul_assoc] at hQ
    rw [← insert_erase
      (mem_range.2 (tsub_pos_iff_lt.2 <| Nat.lt_of_succ_lt_succ <| mem_range.1 hj)),
      sum_insert (not_mem_erase 0 _), add_zero, sum_congr rfl hf₁, ← mul_sum, ← mul_sum, add_assoc,
      ← mul_add, smul_mul_assoc, ← pow_add, Algebra.smul_def] at hQ
    /-
      case neg.hi.convert_2
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQzero : Not (Eq Q 0)
      j : Nat
      hj : Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) (HAdd.hAdd j 1)
      H : LE.le (HAdd.hAdd (HAdd.hAdd j 1) 1) P.natDegree
      Hj : Eq (HAdd.hAdd Q.natDegree 1) (HAdd.hAdd (HAdd.hAdd j 1) (HSub.hSub Q.natD …
      g : Nat → R
      hg : ∀ (k : Nat), Membership.mem (Finset.range (HAdd.hAdd j 1)) k → Eq (HSMul. …
      f : Nat → L
      hf : ∀ (i : Nat), LE.le (Polynomial.map (algebraMap R L) (minpoly R B.gen)).na …
      hf₁ : ∀ (k : Nat), Membership.mem ((Finset.range (HSub.hSub Q.natDegree j)).er …
      hQ : Eq (HAdd.hAdd (HMul.hMul ((algebraMap R L) (Q.coeff (HAdd.hAdd j 1))) (HP …
      ⊢ Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow (Q.coeff j.succ) n.succ)  …
    -/
    replace hQ := congr_arg (norm K) (eq_sub_of_add_eq hQ)

    -- We obtain an equality of elements of `K`, but everything is integral, so we can move to `R`
    -- and simplify `hQ`.
    have hintsum : IsIntegral R (z * B.gen ^ (P.natDegree - (j + 2)) -
        (∑ x ∈ (range (Q.natDegree - j)).erase 0,
          Q.coeff (j + 1 + x) • f (x + P.natDegree - 1) +
            ∑ x ∈ range (j + 1), g x • B.gen ^ x * B.gen ^ (P.natDegree - (j + 2)))) := by
      refine (hzint.mul (hBint.pow _)).sub
        (.add (.sum _ fun k hk => .smul _ ?_)
          (.sum _ fun k _ => .mul (.smul _ (.pow hBint _)) (hBint.pow _)))
      refine adjoin_le_integralClosure hBint (hf _ ?_).1
      rw [(minpoly.monic hBint).natDegree_map (algebraMap R L)]
      rw [add_comm, Nat.add_sub_assoc, le_add_iff_nonneg_right]
      · exact _root_.zero_le _
      · refine one_le_iff_ne_zero.2 fun h => ?_
        rw [h] at hk
        simp at hk
    /-
      case neg.hi.convert_2
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQzero : Not (Eq Q 0)
      j : Nat
      hj : Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) (HAdd.hAdd j 1)
      H : LE.le (HAdd.hAdd (HAdd.hAdd j 1) 1) P.natDegree
      Hj : Eq (HAdd.hAdd Q.natDegree 1) (HAdd.hAdd (HAdd.hAdd j 1) (HSub.hSub Q.natD …
      g : Nat → R
      hg : ∀ (k : Nat), Membership.mem (Finset.range (HAdd.hAdd j 1)) k → Eq (HSMul. …
      f : Nat → L
      hf : ∀ (i : Nat), LE.le (Polynomial.map (algebraMap R L) (minpoly R B.gen)).na …
      hf₁ : ∀ (k : Nat), Membership.mem ((Finset.range (HSub.hSub Q.natDegree j)).er …
      hQ : Eq ((Algebra.norm K) (HMul.hMul ((algebraMap R L) (Q.coeff (HAdd.hAdd j 1 …
      hintsum : IsIntegral R (HSub.hSub (HMul.hMul z (HPow.hPow B.gen (HSub.hSub P.n …
      ⊢ Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow (Q.coeff j.succ) n.succ)  …
    -/
    obtain ⟨r, hr⟩ := isIntegral_iff.1 (isIntegral_norm K hintsum)
    rw [Algebra.smul_def, mul_assoc, ← mul_sub, _root_.map_mul, algebraMap_apply R K L, map_pow,
      Algebra.norm_algebraMap, _root_.map_mul, algebraMap_apply R K L, Algebra.norm_algebraMap,
      finrank B, ← hr, PowerBasis.norm_gen_eq_coeff_zero_minpoly,
      minpoly.isIntegrallyClosed_eq_field_fractions' K hBint, coeff_map,
      show (-1 : K) = algebraMap R K (-1) by simp, ← map_pow, ← map_pow, ← _root_.map_mul, ←
      map_pow, ← _root_.map_mul, ← map_pow, ← _root_.map_mul] at hQ
    -- We can now finish the proof.
    /-
      case neg.hi.convert_2.intro
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      this✝ : Module.Finite K L
      P : Polynomial R := minpoly R B.gen
      hei : P.IsEisensteinAt (Submodule.span R (Singleton.singleton p))
      hndiv : Not (Dvd.dvd (HPow.hPow p 2) (P.coeff 0))
      hP : Eq P (minpoly R B.gen)
      n : Nat
      hn : Eq B.dim n.succ
      this : NoZeroSMulDivisors R L
      x✝ : Polynomial L := Polynomial.map (algebraMap R L) P
      Q₁ : Polynomial R
      Q : Polynomial R := Q₁.modByMonic P
      hQ₁ : Eq Q (Q₁.modByMonic P)
      hQzero : Not (Eq Q 0)
      j : Nat
      hj : Membership.mem (Finset.range (HAdd.hAdd Q.natDegree 1)) (HAdd.hAdd j 1)
      H : LE.le (HAdd.hAdd (HAdd.hAdd j 1) 1) P.natDegree
      Hj : Eq (HAdd.hAdd Q.natDegree 1) (HAdd.hAdd (HAdd.hAdd j 1) (HSub.hSub Q.natD …
      g : Nat → R
      hg : ∀ (k : Nat), Membership.mem (Finset.range (HAdd.hAdd j 1)) k → Eq (HSMul. …
      f : Nat → L
      hf : ∀ (i : Nat), LE.le (Polynomial.map (algebraMap R L) (minpoly R B.gen)).na …
      hf₁ : ∀ (k : Nat), Membership.mem ((Finset.range (HSub.hSub Q.natDegree j)).er …
      hintsum : IsIntegral R (HSub.hSub (HMul.hMul z (HPow.hPow B.gen (HSub.hSub P.n …
      r : R
      hQ : Eq ((algebraMap R K) (HMul.hMul (HPow.hPow (Q.coeff (HAdd.hAdd j 1)) B.di …
      hr : Eq ((algebraMap R K) r) ((Algebra.norm K) (HSub.hSub (HMul.hMul z (HPow.h …
      ⊢ Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow (Q.coeff j.succ) n.succ)  …
    -/
    have hppdiv : p ^ B.dim ∣ p ^ B.dim * r := dvd_mul_of_dvd_left dvd_rfl _
    rwa [← IsFractionRing.injective R K hQ, mul_comm, ← Units.coe_neg_one, mul_pow, ←
      Units.val_pow_eq_pow_val, ← Units.val_pow_eq_pow_val, mul_assoc,
      Units.dvd_mul_left, mul_comm, ← Nat.succ_eq_add_one, hn] at hppdiv


/-- Let `K` be the field of fraction of an integrally closed domain `R` and let `L` be a separable
extension of `K`, generated by an integral power basis `B` such that the minimal polynomial of
`B.gen` is Eisenstein at `p`. Given `z : L` integral over `R`, if `p ^ n • z ∈ adjoin R {B.gen}`,
then `z ∈ adjoin R {B.gen}`. Together with `Algebra.discr_mul_isIntegral_mem_adjoin` this result
often allows to compute the ring of integers of `L`. -/
theorem mem_adjoin_of_smul_prime_pow_smul_of_minpoly_isEisensteinAt {B : PowerBasis K L}
    (hp : Prime p) (hBint : IsIntegral R B.gen) {n : ℕ} {z : L} (hzint : IsIntegral R z)
    (hz : p ^ n • z ∈ adjoin R ({B.gen} : Set L)) (hei : (minpoly R B.gen).IsEisensteinAt 𝓟) :
    z ∈ adjoin R ({B.gen} : Set L) := by
  /-
    R : Type u
    K : Type v
    L : Type z
    p : R
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsScalarTower R K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : IsDomain R
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    B : PowerBasis K L
    hp : Prime p
    hBint : IsIntegral R B.gen
    n : Nat
    z : L
    hzint : IsIntegral R z
    hz : Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMu …
    hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
  -/
  induction' n with n hn
    /-
      case zero
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
      hz : Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMu …
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
    -/
  · simpa using hz
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      K : Type v
      L : Type z
      p : R
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Field K
      inst✝⁸ : Field L
      inst✝⁷ : Algebra K L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra R K
      inst✝⁴ : IsScalarTower R K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : IsDomain R
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      B : PowerBasis K L
      hp : Prime p
      hBint : IsIntegral R B.gen
      z : L
      hzint : IsIntegral R z
      hei : (minpoly R B.gen).IsEisensteinAt (Submodule.span R (Singleton.singleton  …
      n : Nat
      hn : Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMu …
      hz : Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMu …
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) z
    -/
  · rw [_root_.pow_succ', mul_smul] at hz
    exact
      hn (mem_adjoin_of_smul_prime_smul_of_minpoly_isEisensteinAt hp hBint (hzint.smul _) hz hei)


