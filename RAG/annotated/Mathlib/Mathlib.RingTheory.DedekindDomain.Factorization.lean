/-- Given a maximal ideal `v` and an ideal `I` of `R`, `maxPowDividing` returns the maximal
  power of `v` dividing `I`. -/
def IsDedekindDomain.HeightOneSpectrum.maxPowDividing (I : Ideal R) : Ideal R :=
  v.asIdeal ^ (Associates.mk v.asIdeal).count (Associates.mk I).factors


/-- Only finitely many maximal ideals of `R` divide a given nonzero ideal. -/
theorem Ideal.finite_factors {I : Ideal R} (hI : I ≠ 0) :
    {v : HeightOneSpectrum R | v.asIdeal ∣ I}.Finite := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ (setOf fun v => Dvd.dvd v.asIdeal I).Finite
  -/
  rw [← Set.finite_coe_iff, Set.coe_setOf]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ Finite (Subtype fun x => Dvd.dvd x.asIdeal I)
  -/
  haveI h_fin := fintypeSubtypeDvd I hI
  refine
    Finite.of_injective (fun v => (⟨(v : HeightOneSpectrum R).asIdeal, v.2⟩ : { x // x ∣ I })) ?_
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    h_fin : Fintype (Subtype fun x => Dvd.dvd x I)
    ⊢ Function.Injective fun v => ⟨(↑v).asIdeal, ⋯⟩
  -/
  intro v w hvw
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    h_fin : Fintype (Subtype fun x => Dvd.dvd x I)
    v w : Subtype fun x => Dvd.dvd x.asIdeal I
    hvw : Eq ((fun v => ⟨(↑v).asIdeal, ⋯⟩) v) ((fun v => ⟨(↑v).asIdeal, ⋯⟩) w)
    ⊢ Eq v w
  -/
  simp? at hvw says simp only [Subtype.mk.injEq] at hvw
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    h_fin : Fintype (Subtype fun x => Dvd.dvd x I)
    v w : Subtype fun x => Dvd.dvd x.asIdeal I
    hvw : Eq (↑v).asIdeal (↑w).asIdeal
    ⊢ Eq v w
  -/
  exact Subtype.coe_injective (HeightOneSpectrum.ext hvw)
  /-
    🎉 no goals
  -/


/-- For every nonzero ideal `I` of `v`, there are finitely many maximal ideals `v` such that the
  multiplicity of `v` in the factorization of `I`, denoted `val_v(I)`, is nonzero. -/
theorem Associates.finite_factors {I : Ideal R} (hI : I ≠ 0) :
    ∀ᶠ v : HeightOneSpectrum R in Filter.cofinite,
      ((Associates.mk v.asIdeal).count (Associates.mk I).factors : ℤ) = 0 := by
  have h_supp : {v : HeightOneSpectrum R | ¬((Associates.mk v.asIdeal).count
      (Associates.mk I).factors : ℤ) = 0} = {v : HeightOneSpectrum R | v.asIdeal ∣ I} := by
    ext v
    simp_rw [Int.natCast_eq_zero]
    exact Associates.count_ne_zero_iff_dvd hI v.irreducible
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    h_supp : Eq (setOf fun v => Not (Eq (↑((Associates.mk v.asIdeal).count (Associ …
    ⊢ Filter.Eventually (fun v => Eq (↑((Associates.mk v.asIdeal).count (Associate …
  -/
  rw [Filter.eventually_cofinite, h_supp]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    h_supp : Eq (setOf fun v => Not (Eq (↑((Associates.mk v.asIdeal).count (Associ …
    ⊢ (setOf fun v => Dvd.dvd v.asIdeal I).Finite
  -/
  exact Ideal.finite_factors hI
  /-
    🎉 no goals
  -/


/-- For every nonzero ideal `I` of `v`, there are finitely many maximal ideals `v` such that
  `v^(val_v(I))` is not the unit ideal. -/
theorem finite_mulSupport {I : Ideal R} (hI : I ≠ 0) :
    (mulSupport fun v : HeightOneSpectrum R => v.maxPowDividing I).Finite :=
  haveI h_subset : {v : HeightOneSpectrum R | v.maxPowDividing I ≠ 1} ⊆
      {v : HeightOneSpectrum R |
        ((Associates.mk v.asIdeal).count (Associates.mk I).factors : ℤ) ≠ 0} := by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      I : Ideal R
      hI : Ne I 0
      ⊢ HasSubset.Subset (setOf fun v => Ne (v.maxPowDividing I) 1) (setOf fun v =>  …
    -/
    intro v hv h_zero
    have hv' : v.maxPowDividing I = 1 := by
      rw [IsDedekindDomain.HeightOneSpectrum.maxPowDividing, Int.natCast_eq_zero.mp h_zero,
        pow_zero _]
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      I : Ideal R
      hI : Ne I 0
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Membership.mem (setOf fun v => Ne (v.maxPowDividing I) 1) v
      h_zero : Eq (↑((Associates.mk v.asIdeal).count (Associates.mk I).factors)) 0
      hv' : Eq (v.maxPowDividing I) 1
      ⊢ False
    -/
    exact hv hv'
    /-
      🎉 no goals
    -/
  Finite.subset (Filter.eventually_cofinite.mp (Associates.finite_factors hI)) h_subset


/-- For every nonzero ideal `I` of `v`, there are finitely many maximal ideals `v` such that
`v^(val_v(I))`, regarded as a fractional ideal, is not `(1)`. -/
theorem finite_mulSupport_coe {I : Ideal R} (hI : I ≠ 0) :
    (mulSupport fun v : HeightOneSpectrum R => (v.asIdeal : FractionalIdeal R⁰ K) ^
      ((Associates.mk v.asIdeal).count (Associates.mk I).factors : ℤ)).Finite := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ (Function.mulSupport fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdea …
  -/
  rw [mulSupport]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ (setOf fun x => Ne (HPow.hPow ↑x.asIdeal ↑((Associates.mk x.asIdeal).count ( …
  -/
  simp_rw [Ne, zpow_natCast, ← FractionalIdeal.coeIdeal_pow, FractionalIdeal.coeIdeal_eq_one]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ (setOf fun x => Not (Eq (HPow.hPow x.asIdeal ((Associates.mk x.asIdeal).coun …
  -/
  exact finite_mulSupport hI
  /-
    🎉 no goals
  -/


/-- For every nonzero ideal `I` of `v`, there are finitely many maximal ideals `v` such that
`v^-(val_v(I))` is not the unit ideal. -/
theorem finite_mulSupport_inv {I : Ideal R} (hI : I ≠ 0) :
    (mulSupport fun v : HeightOneSpectrum R => (v.asIdeal : FractionalIdeal R⁰ K) ^
      (-((Associates.mk v.asIdeal).count (Associates.mk I).factors : ℤ))).Finite := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ (Function.mulSupport fun v => HPow.hPow (↑v.asIdeal) (Neg.neg ↑((Associates. …
  -/
  rw [mulSupport]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ (setOf fun x => Ne (HPow.hPow (↑x.asIdeal) (Neg.neg ↑((Associates.mk x.asIde …
  -/
  simp_rw [zpow_neg, Ne, inv_eq_one]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ (setOf fun x => Not (Eq (HPow.hPow ↑x.asIdeal ↑((Associates.mk x.asIdeal).co …
  -/
  exact finite_mulSupport_coe hI
  /-
    🎉 no goals
  -/


/-- For every nonzero ideal `I` of `v`, `v^(val_v(I) + 1)` does not divide `∏_v v^(val_v(I))`. -/
theorem finprod_not_dvd (I : Ideal R) (hI : I ≠ 0) :
    ¬v.asIdeal ^ ((Associates.mk v.asIdeal).count (Associates.mk I).factors + 1) ∣
        ∏ᶠ v : HeightOneSpectrum R, v.maxPowDividing I := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    ⊢ Not (Dvd.dvd (HPow.hPow v.asIdeal (HAdd.hAdd ((Associates.mk v.asIdeal).coun …
  -/
  have hf := finite_mulSupport hI
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    hf : (Function.mulSupport fun v => v.maxPowDividing I).Finite
    ⊢ Not (Dvd.dvd (HPow.hPow v.asIdeal (HAdd.hAdd ((Associates.mk v.asIdeal).coun …
  -/
  have h_ne_zero : v.maxPowDividing I ≠ 0 := pow_ne_zero _ v.ne_bot
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    hf : (Function.mulSupport fun v => v.maxPowDividing I).Finite
    h_ne_zero : Ne (v.maxPowDividing I) 0
    ⊢ Not (Dvd.dvd (HPow.hPow v.asIdeal (HAdd.hAdd ((Associates.mk v.asIdeal).coun …
  -/
  rw [← mul_finprod_cond_ne v hf, pow_add, pow_one, finprod_cond_ne _ _ hf]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    hf : (Function.mulSupport fun v => v.maxPowDividing I).Finite
    h_ne_zero : Ne (v.maxPowDividing I) 0
    ⊢ Not (Dvd.dvd (HMul.hMul (HPow.hPow v.asIdeal ((Associates.mk v.asIdeal).coun …
  -/
  intro h_contr
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    hf : (Function.mulSupport fun v => v.maxPowDividing I).Finite
    h_ne_zero : Ne (v.maxPowDividing I) 0
    h_contr : Dvd.dvd (HMul.hMul (HPow.hPow v.asIdeal ((Associates.mk v.asIdeal).c …
    ⊢ False
  -/
  have hv_prime : Prime v.asIdeal := Ideal.prime_of_isPrime v.ne_bot v.isPrime
  obtain ⟨w, hw, hvw'⟩ :=
    Prime.exists_mem_finset_dvd hv_prime ((mul_dvd_mul_iff_left h_ne_zero).mp h_contr)
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    hf : (Function.mulSupport fun v => v.maxPowDividing I).Finite
    h_ne_zero : Ne (v.maxPowDividing I) 0
    h_contr : Dvd.dvd (HMul.hMul (HPow.hPow v.asIdeal ((Associates.mk v.asIdeal).c …
    hv_prime : Prime v.asIdeal
    w : IsDedekindDomain.HeightOneSpectrum R
    hw : Membership.mem (hf.toFinset.erase v) w
    hvw' : Dvd.dvd v.asIdeal (w.maxPowDividing I)
    ⊢ False
  -/
  have hw_prime : Prime w.asIdeal := Ideal.prime_of_isPrime w.ne_bot w.isPrime
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    hf : (Function.mulSupport fun v => v.maxPowDividing I).Finite
    h_ne_zero : Ne (v.maxPowDividing I) 0
    h_contr : Dvd.dvd (HMul.hMul (HPow.hPow v.asIdeal ((Associates.mk v.asIdeal).c …
    hv_prime : Prime v.asIdeal
    w : IsDedekindDomain.HeightOneSpectrum R
    hw : Membership.mem (hf.toFinset.erase v) w
    hvw' : Dvd.dvd v.asIdeal (w.maxPowDividing I)
    hw_prime : Prime w.asIdeal
    ⊢ False
  -/
  have hvw := Prime.dvd_of_dvd_pow hv_prime hvw'
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    hf : (Function.mulSupport fun v => v.maxPowDividing I).Finite
    h_ne_zero : Ne (v.maxPowDividing I) 0
    h_contr : Dvd.dvd (HMul.hMul (HPow.hPow v.asIdeal ((Associates.mk v.asIdeal).c …
    hv_prime : Prime v.asIdeal
    w : IsDedekindDomain.HeightOneSpectrum R
    hw : Membership.mem (hf.toFinset.erase v) w
    hvw' : Dvd.dvd v.asIdeal (w.maxPowDividing I)
    hw_prime : Prime w.asIdeal
    hvw : Dvd.dvd v.asIdeal w.asIdeal
    ⊢ False
  -/
  rw [Prime.dvd_prime_iff_associated hv_prime hw_prime, associated_iff_eq] at hvw
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    hf : (Function.mulSupport fun v => v.maxPowDividing I).Finite
    h_ne_zero : Ne (v.maxPowDividing I) 0
    h_contr : Dvd.dvd (HMul.hMul (HPow.hPow v.asIdeal ((Associates.mk v.asIdeal).c …
    hv_prime : Prime v.asIdeal
    w : IsDedekindDomain.HeightOneSpectrum R
    hw : Membership.mem (hf.toFinset.erase v) w
    hvw' : Dvd.dvd v.asIdeal (w.maxPowDividing I)
    hw_prime : Prime w.asIdeal
    hvw : Eq v.asIdeal w.asIdeal
    ⊢ False
  -/
  exact (Finset.mem_erase.mp hw).1 (HeightOneSpectrum.ext hvw.symm)
  /-
    🎉 no goals
  -/


theorem Associates.finprod_ne_zero (I : Ideal R) :
    Associates.mk (∏ᶠ v : HeightOneSpectrum R, v.maxPowDividing I) ≠ 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    ⊢ Ne (Associates.mk (finprod fun v => v.maxPowDividing I)) 0
  -/
  rw [Associates.mk_ne_zero, finprod_def]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    ⊢ Ne (dite (Function.mulSupport fun v => v.maxPowDividing I).Finite (fun h =>  …
  -/
  split_ifs
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      I : Ideal R
      h✝ : (Function.mulSupport fun v => v.maxPowDividing I).Finite
      ⊢ Ne (h✝.toFinset.prod fun v => v.maxPowDividing I) 0
    -/
  · rw [Finset.prod_ne_zero_iff]
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      I : Ideal R
      h✝ : (Function.mulSupport fun v => v.maxPowDividing I).Finite
      ⊢ ∀ (a : IsDedekindDomain.HeightOneSpectrum R), Membership.mem h✝.toFinset a → …
    -/
    intro v _
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      I : Ideal R
      h✝ : (Function.mulSupport fun v => v.maxPowDividing I).Finite
      v : IsDedekindDomain.HeightOneSpectrum R
      a✝ : Membership.mem h✝.toFinset v
      ⊢ Ne (v.maxPowDividing I) 0
    -/
    apply pow_ne_zero _ v.ne_bot
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      I : Ideal R
      h✝ : Not (Function.mulSupport fun v => v.maxPowDividing I).Finite
      ⊢ Ne 1 0
    -/
  · exact one_ne_zero
    /-
      🎉 no goals
    -/


/-- The multiplicity of `v` in `∏_v v^(val_v(I))` equals `val_v(I)`. -/
theorem finprod_count (I : Ideal R) (hI : I ≠ 0) : (Associates.mk v.asIdeal).count
    (Associates.mk (∏ᶠ v : HeightOneSpectrum R, v.maxPowDividing I)).factors =
    (Associates.mk v.asIdeal).count (Associates.mk I).factors := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    ⊢ Eq ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun v => v.maxPo …
  -/
  have h_ne_zero := Associates.finprod_ne_zero I
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    h_ne_zero : Ne (Associates.mk (finprod fun v => v.maxPowDividing I)) 0
    ⊢ Eq ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun v => v.maxPo …
  -/
  have hv : Irreducible (Associates.mk v.asIdeal) := v.associates_irreducible
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    h_ne_zero : Ne (Associates.mk (finprod fun v => v.maxPowDividing I)) 0
    hv : Irreducible (Associates.mk v.asIdeal)
    ⊢ Eq ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun v => v.maxPo …
  -/
  have h_dvd := finprod_mem_dvd v (Ideal.finite_mulSupport hI)
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    h_ne_zero : Ne (Associates.mk (finprod fun v => v.maxPowDividing I)) 0
    hv : Irreducible (Associates.mk v.asIdeal)
    h_dvd : Dvd.dvd (v.maxPowDividing I) (finprod fun v => v.maxPowDividing I)
    ⊢ Eq ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun v => v.maxPo …
  -/
  have h_not_dvd := Ideal.finprod_not_dvd v I hI
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    h_ne_zero : Ne (Associates.mk (finprod fun v => v.maxPowDividing I)) 0
    hv : Irreducible (Associates.mk v.asIdeal)
    h_dvd : Dvd.dvd (v.maxPowDividing I) (finprod fun v => v.maxPowDividing I)
    h_not_dvd : Not (Dvd.dvd (HPow.hPow v.asIdeal (HAdd.hAdd ((Associates.mk v.asI …
    ⊢ Eq ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun v => v.maxPo …
  -/
  simp only [IsDedekindDomain.HeightOneSpectrum.maxPowDividing] at h_dvd h_ne_zero h_not_dvd
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    h_ne_zero : Ne (Associates.mk (finprod fun v => HPow.hPow v.asIdeal ((Associat …
    hv : Irreducible (Associates.mk v.asIdeal)
    h_dvd : Dvd.dvd (HPow.hPow v.asIdeal ((Associates.mk v.asIdeal).count (Associa …
    h_not_dvd : Not (Dvd.dvd (HPow.hPow v.asIdeal (HAdd.hAdd ((Associates.mk v.asI …
    ⊢ Eq ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun v => v.maxPo …
  -/
  rw [← Associates.mk_dvd_mk] at h_dvd h_not_dvd
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    h_ne_zero : Ne (Associates.mk (finprod fun v => HPow.hPow v.asIdeal ((Associat …
    hv : Irreducible (Associates.mk v.asIdeal)
    h_dvd : Dvd.dvd (Associates.mk (HPow.hPow v.asIdeal ((Associates.mk v.asIdeal) …
    h_not_dvd : Not (Dvd.dvd (Associates.mk (HPow.hPow v.asIdeal (HAdd.hAdd ((Asso …
    ⊢ Eq ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun v => v.maxPo …
  -/
  simp only [Associates.dvd_eq_le] at h_dvd h_not_dvd
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    h_ne_zero : Ne (Associates.mk (finprod fun v => HPow.hPow v.asIdeal ((Associat …
    hv : Irreducible (Associates.mk v.asIdeal)
    h_dvd : LE.le (Associates.mk (HPow.hPow v.asIdeal ((Associates.mk v.asIdeal).c …
    h_not_dvd : Not (LE.le (Associates.mk (HPow.hPow v.asIdeal (HAdd.hAdd ((Associ …
    ⊢ Eq ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun v => v.maxPo …
  -/
  rw [Associates.mk_pow, Associates.prime_pow_dvd_iff_le h_ne_zero hv] at h_dvd h_not_dvd
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    h_ne_zero : Ne (Associates.mk (finprod fun v => HPow.hPow v.asIdeal ((Associat …
    hv : Irreducible (Associates.mk v.asIdeal)
    h_dvd : LE.le ((Associates.mk v.asIdeal).count (Associates.mk I).factors) ((As …
    h_not_dvd : Not (LE.le (HAdd.hAdd ((Associates.mk v.asIdeal).count (Associates …
    ⊢ Eq ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun v => v.maxPo …
  -/
  rw [not_le] at h_not_dvd
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : Ideal R
    hI : Ne I 0
    h_ne_zero : Ne (Associates.mk (finprod fun v => HPow.hPow v.asIdeal ((Associat …
    hv : Irreducible (Associates.mk v.asIdeal)
    h_dvd : LE.le ((Associates.mk v.asIdeal).count (Associates.mk I).factors) ((As …
    h_not_dvd : LT.lt ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun …
    ⊢ Eq ((Associates.mk v.asIdeal).count (Associates.mk (finprod fun v => v.maxPo …
  -/
  apply Nat.eq_of_le_of_lt_succ h_dvd h_not_dvd
  /-
    🎉 no goals
  -/


/-- The ideal `I` equals the finprod `∏_v v^(val_v(I))`. -/
theorem finprod_heightOneSpectrum_factorization {I : Ideal R} (hI : I ≠ 0) :
    ∏ᶠ v : HeightOneSpectrum R, v.maxPowDividing I = I := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ Eq (finprod fun v => v.maxPowDividing I) I
  -/
  rw [← associated_iff_eq, ← Associates.mk_eq_mk_iff_associated]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ Eq (Associates.mk (finprod fun v => v.maxPowDividing I)) (Associates.mk I)
  -/
  apply Associates.eq_of_eq_counts
    /-
      case ha
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      I : Ideal R
      hI : Ne I 0
      ⊢ Ne (Associates.mk (finprod fun v => v.maxPowDividing I)) 0
    -/
  · apply Associates.finprod_ne_zero I
    /-
      🎉 no goals
    -/
    /-
      case hb
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      I : Ideal R
      hI : Ne I 0
      ⊢ Ne (Associates.mk I) 0
    -/
  · apply Associates.mk_ne_zero.mpr hI
    /-
      🎉 no goals
    -/
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ ∀ (p : Associates (Ideal R)), Irreducible p → Eq (p.count (Associates.mk (fi …
  -/
  intro v hv
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    v : Associates (Ideal R)
    hv : Irreducible v
    ⊢ Eq (v.count (Associates.mk (finprod fun v => v.maxPowDividing I)).factors) ( …
  -/
  obtain ⟨J, hJv⟩ := Associates.exists_rep v
  /-
    case h.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    v : Associates (Ideal R)
    hv : Irreducible v
    J : Ideal R
    hJv : Eq (Associates.mk J) v
    ⊢ Eq (v.count (Associates.mk (finprod fun v => v.maxPowDividing I)).factors) ( …
  -/
  rw [← hJv, Associates.irreducible_mk] at hv
  /-
    case h.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    v : Associates (Ideal R)
    J : Ideal R
    hv : Irreducible J
    hJv : Eq (Associates.mk J) v
    ⊢ Eq (v.count (Associates.mk (finprod fun v => v.maxPowDividing I)).factors) ( …
  -/
  rw [← hJv]
  apply Ideal.finprod_count
    ⟨J, Ideal.isPrime_of_prime (irreducible_iff_prime.mp hv), Irreducible.ne_zero hv⟩ I hI


/-- The ideal `I` equals the finprod `∏_v v^(val_v(I))`, when both sides are regarded as fractional
ideals of `R`. -/
theorem finprod_heightOneSpectrum_factorization_coe {I : Ideal R} (hI : I ≠ 0) :
    (∏ᶠ v : HeightOneSpectrum R, (v.asIdeal : FractionalIdeal R⁰ K) ^
      ((Associates.mk v.asIdeal).count (Associates.mk I).factors : ℤ)) = I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).count  …
  -/
  conv_rhs => rw [← Ideal.finprod_heightOneSpectrum_factorization hI]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : Ne I 0
    ⊢ Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).count  …
  -/
  rw [FractionalIdeal.coeIdeal_finprod R⁰ K (le_refl _)]
  simp_rw [IsDedekindDomain.HeightOneSpectrum.maxPowDividing, FractionalIdeal.coeIdeal_pow,
    zpow_natCast]


/-- If `I` is a nonzero fractional ideal, `a ∈ R`, and `J` is an ideal of `R` such that
`I = a⁻¹J`, then `I` is equal to the product `∏_v v^(val_v(J) - val_v(a))`. -/
theorem finprod_heightOneSpectrum_factorization {I : FractionalIdeal R⁰ K} (hI : I ≠ 0) {a : R}
    {J : Ideal R} (haJ : I = spanSingleton R⁰ ((algebraMap R K) a)⁻¹ * ↑J) :
    ∏ᶠ v : HeightOneSpectrum R, (v.asIdeal : FractionalIdeal R⁰ K) ^
      ((Associates.mk v.asIdeal).count (Associates.mk J).factors -
        (Associates.mk v.asIdeal).count (Associates.mk (Ideal.span {a})).factors : ℤ) = I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  have hJ_ne_zero : J ≠ 0 := ideal_factor_ne_zero hI haJ
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    hJ_ne_zero : Ne J 0
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  have hJ := Ideal.finprod_heightOneSpectrum_factorization_coe K hJ_ne_zero
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    hJ_ne_zero : Ne J 0
    hJ : Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).cou …
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  have ha_ne_zero : Ideal.span {a} ≠ 0 := constant_factor_ne_zero hI haJ
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    hJ_ne_zero : Ne J 0
    hJ : Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).cou …
    ha_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  have ha := Ideal.finprod_heightOneSpectrum_factorization_coe K ha_ne_zero
  rw [haJ, ← div_spanSingleton, div_eq_mul_inv, ← coeIdeal_span_singleton, ← hJ, ← ha,
    ← finprod_inv_distrib]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    hJ_ne_zero : Ne J 0
    hJ : Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).cou …
    ha_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
    ha : Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).cou …
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  simp_rw [← zpow_neg]
  rw [← finprod_mul_distrib (Ideal.finite_mulSupport_coe hJ_ne_zero)
    (Ideal.finite_mulSupport_inv ha_ne_zero)]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    hJ_ne_zero : Ne J 0
    hJ : Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).cou …
    ha_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
    ha : Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).cou …
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  apply finprod_congr
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    hJ_ne_zero : Ne J 0
    hJ : Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).cou …
    ha_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
    ha : Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).cou …
    ⊢ ∀ (x : IsDedekindDomain.HeightOneSpectrum R), Eq (HPow.hPow (↑x.asIdeal) (HS …
  -/
  intro v
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    hJ_ne_zero : Ne J 0
    hJ : Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).cou …
    ha_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
    ha : Eq (finprod fun v => HPow.hPow ↑v.asIdeal ↑((Associates.mk v.asIdeal).cou …
    v : IsDedekindDomain.HeightOneSpectrum R
    ⊢ Eq (HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.asIdeal).count (Ass …
  -/
  rw [← zpow_add₀ ((@coeIdeal_ne_zero R _ K _ _ _ _).mpr v.ne_bot), sub_eq_add_neg]
  /-
    🎉 no goals
  -/


/-- For a nonzero `k = r/s ∈ K`, the fractional ideal `(k)` is equal to the product
`∏_v v^(val_v(r) - val_v(s))`. -/
theorem finprod_heightOneSpectrum_factorization_principal_fraction {n : R} (hn : n ≠ 0) (d : ↥R⁰) :
    ∏ᶠ v : HeightOneSpectrum R, (v.asIdeal : FractionalIdeal R⁰ K) ^
      ((Associates.mk v.asIdeal).count (Associates.mk (Ideal.span {n} : Ideal R)).factors -
        (Associates.mk v.asIdeal).count (Associates.mk ((Ideal.span {(↑d : R)}) :
        Ideal R)).factors : ℤ) = spanSingleton R⁰ (mk' K n d) := by
  have hd_ne_zero : (algebraMap R K) (d : R) ≠ 0 :=
    map_ne_zero_of_mem_nonZeroDivisors _ (IsFractionRing.injective R K) d.property
  have h0 : spanSingleton R⁰ (mk' K n d) ≠ 0 := by
    rw [spanSingleton_ne_zero_iff, IsFractionRing.mk'_eq_div, ne_eq, div_eq_zero_iff, not_or]
    exact ⟨(map_ne_zero_iff (algebraMap R K) (IsFractionRing.injective R K)).mpr hn, hd_ne_zero⟩
  have hI : spanSingleton R⁰ (mk' K n d) =
      spanSingleton R⁰ ((algebraMap R K) d)⁻¹ * ↑(Ideal.span {n} : Ideal R) := by
    rw [coeIdeal_span_singleton, spanSingleton_mul_spanSingleton]
    apply congr_arg
    rw [IsFractionRing.mk'_eq_div, div_eq_mul_inv, mul_comm]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    n : R
    hn : Ne n 0
    d : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    hd_ne_zero : Ne ((algebraMap R K) ↑d) 0
    h0 : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) (IsLocalization.mk' …
    hI : Eq (FractionalIdeal.spanSingleton (nonZeroDivisors R) (IsLocalization.mk' …
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  exact finprod_heightOneSpectrum_factorization h0 hI
  /-
    🎉 no goals
  -/


/-- For a nonzero `k = r/s ∈ K`, the fractional ideal `(k)` is equal to the product
`∏_v v^(val_v(r) - val_v(s))`. -/
theorem finprod_heightOneSpectrum_factorization_principal {I : FractionalIdeal R⁰ K} (hI : I ≠ 0)
    (k : K) (hk : I = spanSingleton R⁰ k) :
    ∏ᶠ v : HeightOneSpectrum R, (v.asIdeal : FractionalIdeal R⁰ K) ^
      ((Associates.mk v.asIdeal).count (Associates.mk (Ideal.span {choose
          (mk'_surjective R⁰ k)} : Ideal R)).factors -
        (Associates.mk v.asIdeal).count (Associates.mk ((Ideal.span {(↑(choose
          (choose_spec (mk'_surjective R⁰ k)) : ↥R⁰) : R)}) : Ideal R)).factors : ℤ) = I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    k : K
    hk : Eq I (FractionalIdeal.spanSingleton (nonZeroDivisors R) k)
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  set n : R := choose (mk'_surjective R⁰ k)
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    k : K
    hk : Eq I (FractionalIdeal.spanSingleton (nonZeroDivisors R) k)
    n : R := Classical.choose ⋯
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  set d : ↥R⁰ := choose (choose_spec (mk'_surjective R⁰ k))
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    k : K
    hk : Eq I (FractionalIdeal.spanSingleton (nonZeroDivisors R) k)
    n : R := Classical.choose ⋯
    d : Subtype fun x => Membership.mem (nonZeroDivisors R) x := Classical.choose ⋯
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  have hnd : mk' K n d = k := choose_spec (choose_spec (mk'_surjective R⁰ k))
  have hn0 : n ≠ 0 := by
    by_contra h
    rw [← hnd, h, IsFractionRing.mk'_eq_div, _root_.map_zero,
      zero_div, spanSingleton_zero] at hk
    exact hI hk
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    k : K
    hk : Eq I (FractionalIdeal.spanSingleton (nonZeroDivisors R) k)
    n : R := Classical.choose ⋯
    d : Subtype fun x => Membership.mem (nonZeroDivisors R) x := Classical.choose ⋯
    hnd : Eq (IsLocalization.mk' K n d) k
    hn0 : Ne n 0
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (HSub.hSub ↑((Associates.mk v.as …
  -/
  rw [finprod_heightOneSpectrum_factorization_principal_fraction hn0 d, hk, hnd]
  /-
    🎉 no goals
  -/


/-- If `I` is a nonzero fractional ideal, `a ∈ R`, and `J` is an ideal of `R` such that `I = a⁻¹J`,
then we define `val_v(I)` as `(val_v(J) - val_v(a))`. If `I = 0`, we set `val_v(I) = 0`. -/
def count (I : FractionalIdeal R⁰ K) : ℤ :=
  dite (I = 0) (fun _ : I = 0 => 0) fun _ : ¬I = 0 =>
    let a := choose (exists_eq_spanSingleton_mul I)
    let J := choose (choose_spec (exists_eq_spanSingleton_mul I))
    ((Associates.mk v.asIdeal).count (Associates.mk J).factors -
        (Associates.mk v.asIdeal).count (Associates.mk (Ideal.span {a})).factors : ℤ)


/-- val_v(0) = 0. -/
                                                                  /-
                                                                    R : Type u_1
                                                                    inst✝⁴ : CommRing R
                                                                    K : Type u_2
                                                                    inst✝³ : Field K
                                                                    inst✝² : Algebra R K
                                                                    inst✝¹ : IsFractionRing R K
                                                                    inst✝ : IsDedekindDomain R
                                                                    v : IsDedekindDomain.HeightOneSpectrum R
                                                                    ⊢ Eq (FractionalIdeal.count K v 0) 0
                                                                  -/
lemma count_zero : count K v (0 : FractionalIdeal R⁰ K) = 0 := by simp only [count, dif_pos]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma count_ne_zero {I : FractionalIdeal R⁰ K} (hI : I ≠ 0) :
    count K v I = ((Associates.mk v.asIdeal).count (Associates.mk
      (choose (choose_spec (exists_eq_spanSingleton_mul I)))).factors -
      (Associates.mk v.asIdeal).count
        (Associates.mk (Ideal.span {choose (exists_eq_spanSingleton_mul I)})).factors : ℤ) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    ⊢ Eq (FractionalIdeal.count K v I) (HSub.hSub ↑((Associates.mk v.asIdeal).coun …
  -/
  simp only [count, dif_neg hI]
  /-
    🎉 no goals
  -/


/-- `val_v(I)` does not depend on the choice of `a` and `J` used to represent `I`. -/
theorem count_well_defined {I : FractionalIdeal R⁰ K} (hI : I ≠ 0) {a : R}
    {J : Ideal R} (h_aJ : I = spanSingleton R⁰ ((algebraMap R K) a)⁻¹ * ↑J) :
    count K v I = ((Associates.mk v.asIdeal).count (Associates.mk J).factors -
      (Associates.mk v.asIdeal).count (Associates.mk (Ideal.span {a})).factors : ℤ) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    h_aJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv …
    ⊢ Eq (FractionalIdeal.count K v I) (HSub.hSub ↑((Associates.mk v.asIdeal).coun …
  -/
  set a₁ := choose (exists_eq_spanSingleton_mul I)
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    h_aJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv …
    a₁ : R := Classical.choose ⋯
    ⊢ Eq (FractionalIdeal.count K v I) (HSub.hSub ↑((Associates.mk v.asIdeal).coun …
  -/
  set J₁ := choose (choose_spec (exists_eq_spanSingleton_mul I))
  have h_a₁J₁ : I = spanSingleton R⁰ ((algebraMap R K) a₁)⁻¹ * ↑J₁ :=
    (choose_spec (choose_spec (exists_eq_spanSingleton_mul I))).2
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    h_aJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv …
    a₁ : R := Classical.choose ⋯
    J₁ : Ideal R := Classical.choose ⋯
    h_a₁J₁ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (I …
    ⊢ Eq (FractionalIdeal.count K v I) (HSub.hSub ↑((Associates.mk v.asIdeal).coun …
  -/
  have h_a₁_ne_zero : a₁ ≠ 0 := (choose_spec (choose_spec (exists_eq_spanSingleton_mul I))).1
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    h_aJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv …
    a₁ : R := Classical.choose ⋯
    J₁ : Ideal R := Classical.choose ⋯
    h_a₁J₁ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (I …
    h_a₁_ne_zero : Ne a₁ 0
    ⊢ Eq (FractionalIdeal.count K v I) (HSub.hSub ↑((Associates.mk v.asIdeal).coun …
  -/
  have h_J₁_ne_zero : J₁ ≠ 0 := ideal_factor_ne_zero hI h_a₁J₁
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    h_aJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv …
    a₁ : R := Classical.choose ⋯
    J₁ : Ideal R := Classical.choose ⋯
    h_a₁J₁ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (I …
    h_a₁_ne_zero : Ne a₁ 0
    h_J₁_ne_zero : Ne J₁ 0
    ⊢ Eq (FractionalIdeal.count K v I) (HSub.hSub ↑((Associates.mk v.asIdeal).coun …
  -/
  have h_a_ne_zero : Ideal.span {a} ≠ 0 := constant_factor_ne_zero hI h_aJ
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    h_aJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv …
    a₁ : R := Classical.choose ⋯
    J₁ : Ideal R := Classical.choose ⋯
    h_a₁J₁ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (I …
    h_a₁_ne_zero : Ne a₁ 0
    h_J₁_ne_zero : Ne J₁ 0
    h_a_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
    ⊢ Eq (FractionalIdeal.count K v I) (HSub.hSub ↑((Associates.mk v.asIdeal).coun …
  -/
  have h_J_ne_zero : J ≠ 0 := ideal_factor_ne_zero hI h_aJ
  have h_a₁' : spanSingleton R⁰ ((algebraMap R K) a₁) ≠ 0 := by
    rw [ne_eq, spanSingleton_eq_zero_iff, ← (algebraMap R K).map_zero,
      Injective.eq_iff (IsLocalization.injective K (le_refl R⁰))]
    exact h_a₁_ne_zero
  have h_a' : spanSingleton R⁰ ((algebraMap R K) a) ≠ 0 := by
    rw [ne_eq, spanSingleton_eq_zero_iff, ← (algebraMap R K).map_zero,
      Injective.eq_iff (IsLocalization.injective K (le_refl R⁰))]
    rw [ne_eq, Ideal.zero_eq_bot, Ideal.span_singleton_eq_bot] at h_a_ne_zero
    exact h_a_ne_zero
  have hv : Irreducible (Associates.mk v.asIdeal) := by
    exact Associates.irreducible_mk.mpr v.irreducible
  rw [h_a₁J₁, ← div_spanSingleton, ← div_spanSingleton, div_eq_div_iff h_a₁' h_a',
    ← coeIdeal_span_singleton, ← coeIdeal_span_singleton, ← coeIdeal_mul, ← coeIdeal_mul] at h_aJ
  rw [count, dif_neg hI, sub_eq_sub_iff_add_eq_add, ← ofNat_add, ← ofNat_add, natCast_inj,
    ← Associates.count_mul _ _ hv, ← Associates.count_mul _ _ hv, Associates.mk_mul_mk,
    Associates.mk_mul_mk, coeIdeal_injective h_aJ]
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      hI : Ne I 0
      a : R
      J : Ideal R
      a₁ : R := Classical.choose ⋯
      J₁ : Ideal R := Classical.choose ⋯
      h_aJ : Eq ↑(HMul.hMul J₁ (Ideal.span (Singleton.singleton a))) ↑(HMul.hMul J ( …
      h_a₁J₁ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (I …
      h_a₁_ne_zero : Ne a₁ 0
      h_J₁_ne_zero : Ne J₁ 0
      h_a_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
      h_J_ne_zero : Ne J 0
      h_a₁' : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) ((algebraMap R K …
      h_a' : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) ((algebraMap R K) …
      hv : Irreducible (Associates.mk v.asIdeal)
      ⊢ Ne (Associates.mk J) 0
    -/
  · rw [ne_eq, Associates.mk_eq_zero]; exact h_J_ne_zero
                                       /-
                                         🎉 no goals
                                       -/
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      hI : Ne I 0
      a : R
      J : Ideal R
      a₁ : R := Classical.choose ⋯
      J₁ : Ideal R := Classical.choose ⋯
      h_aJ : Eq ↑(HMul.hMul J₁ (Ideal.span (Singleton.singleton a))) ↑(HMul.hMul J ( …
      h_a₁J₁ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (I …
      h_a₁_ne_zero : Ne a₁ 0
      h_J₁_ne_zero : Ne J₁ 0
      h_a_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
      h_J_ne_zero : Ne J 0
      h_a₁' : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) ((algebraMap R K …
      h_a' : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) ((algebraMap R K) …
      hv : Irreducible (Associates.mk v.asIdeal)
      ⊢ Ne (Associates.mk (Ideal.span (Singleton.singleton (Classical.choose ⋯)))) 0
    -/
  · rw [ne_eq, Associates.mk_eq_zero, Ideal.zero_eq_bot, Ideal.span_singleton_eq_bot]
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      hI : Ne I 0
      a : R
      J : Ideal R
      a₁ : R := Classical.choose ⋯
      J₁ : Ideal R := Classical.choose ⋯
      h_aJ : Eq ↑(HMul.hMul J₁ (Ideal.span (Singleton.singleton a))) ↑(HMul.hMul J ( …
      h_a₁J₁ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (I …
      h_a₁_ne_zero : Ne a₁ 0
      h_J₁_ne_zero : Ne J₁ 0
      h_a_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
      h_J_ne_zero : Ne J 0
      h_a₁' : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) ((algebraMap R K …
      h_a' : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) ((algebraMap R K) …
      hv : Irreducible (Associates.mk v.asIdeal)
      ⊢ Not (Eq (Classical.choose ⋯) 0)
    -/
    exact h_a₁_ne_zero
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      hI : Ne I 0
      a : R
      J : Ideal R
      a₁ : R := Classical.choose ⋯
      J₁ : Ideal R := Classical.choose ⋯
      h_aJ : Eq ↑(HMul.hMul J₁ (Ideal.span (Singleton.singleton a))) ↑(HMul.hMul J ( …
      h_a₁J₁ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (I …
      h_a₁_ne_zero : Ne a₁ 0
      h_J₁_ne_zero : Ne J₁ 0
      h_a_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
      h_J_ne_zero : Ne J 0
      h_a₁' : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) ((algebraMap R K …
      h_a' : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) ((algebraMap R K) …
      hv : Irreducible (Associates.mk v.asIdeal)
      ⊢ Ne (Associates.mk (Classical.choose ⋯)) 0
    -/
  · rw [ne_eq, Associates.mk_eq_zero]; exact h_J₁_ne_zero
                                       /-
                                         🎉 no goals
                                       -/
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      hI : Ne I 0
      a : R
      J : Ideal R
      a₁ : R := Classical.choose ⋯
      J₁ : Ideal R := Classical.choose ⋯
      h_aJ : Eq ↑(HMul.hMul J₁ (Ideal.span (Singleton.singleton a))) ↑(HMul.hMul J ( …
      h_a₁J₁ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (I …
      h_a₁_ne_zero : Ne a₁ 0
      h_J₁_ne_zero : Ne J₁ 0
      h_a_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
      h_J_ne_zero : Ne J 0
      h_a₁' : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) ((algebraMap R K …
      h_a' : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R) ((algebraMap R K) …
      hv : Irreducible (Associates.mk v.asIdeal)
      ⊢ Ne (Associates.mk (Ideal.span (Singleton.singleton a))) 0
    -/
  · rw [ne_eq, Associates.mk_eq_zero]; exact h_a_ne_zero
                                       /-
                                         🎉 no goals
                                       -/


/-- For nonzero `I, I'`, `val_v(I*I') = val_v(I) + val_v(I')`. -/
theorem count_mul {I I' : FractionalIdeal R⁰ K} (hI : I ≠ 0) (hI' : I' ≠ 0) :
    count K v (I * I') = count K v I + count K v I' := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I I' : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    hI' : Ne I' 0
    ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) (HAdd.hAdd (FractionalIdeal. …
  -/
  have hv : Irreducible (Associates.mk v.asIdeal) := by apply v.associates_irreducible
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I I' : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    hI' : Ne I' 0
    hv : Irreducible (Associates.mk v.asIdeal)
    ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) (HAdd.hAdd (FractionalIdeal. …
  -/
  obtain ⟨a, J, ha, haJ⟩ := exists_eq_spanSingleton_mul I
  have ha_ne_zero : Associates.mk (Ideal.span {a} : Ideal R) ≠ 0 := by
    rw [ne_eq, Associates.mk_eq_zero, Ideal.zero_eq_bot, Ideal.span_singleton_eq_bot]; exact ha
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I I' : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    hI' : Ne I' 0
    hv : Irreducible (Associates.mk v.asIdeal)
    a : R
    J : Ideal R
    ha : Ne a 0
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    ha_ne_zero : Ne (Associates.mk (Ideal.span (Singleton.singleton a))) 0
    ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) (HAdd.hAdd (FractionalIdeal. …
  -/
  have hJ_ne_zero : Associates.mk J ≠ 0 := Associates.mk_ne_zero.mpr (ideal_factor_ne_zero hI haJ)
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I I' : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    hI' : Ne I' 0
    hv : Irreducible (Associates.mk v.asIdeal)
    a : R
    J : Ideal R
    ha : Ne a 0
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    ha_ne_zero : Ne (Associates.mk (Ideal.span (Singleton.singleton a))) 0
    hJ_ne_zero : Ne (Associates.mk J) 0
    ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) (HAdd.hAdd (FractionalIdeal. …
  -/
  obtain ⟨a', J', ha', haJ'⟩ := exists_eq_spanSingleton_mul I'
  have ha'_ne_zero : Associates.mk (Ideal.span {a'} : Ideal R) ≠ 0 := by
    rw [ne_eq, Associates.mk_eq_zero, Ideal.zero_eq_bot, Ideal.span_singleton_eq_bot]; exact ha'
  have hJ'_ne_zero : Associates.mk J' ≠ 0 :=
    Associates.mk_ne_zero.mpr (ideal_factor_ne_zero hI' haJ')
  have h_prod : I * I' = spanSingleton R⁰ ((algebraMap R K) (a * a'))⁻¹ * ↑(J * J') := by
    rw [haJ, haJ', mul_assoc, mul_comm (J : FractionalIdeal R⁰ K), mul_assoc, ← mul_assoc,
      spanSingleton_mul_spanSingleton, coeIdeal_mul, RingHom.map_mul, mul_inv,
      mul_comm (J : FractionalIdeal R⁰ K)]
  rw [count_well_defined K v hI haJ, count_well_defined K v hI' haJ',
    count_well_defined K v (mul_ne_zero hI hI') h_prod, ← Associates.mk_mul_mk,
    Associates.count_mul hJ_ne_zero hJ'_ne_zero hv, ← Ideal.span_singleton_mul_span_singleton,
    ← Associates.mk_mul_mk, Associates.count_mul ha_ne_zero ha'_ne_zero hv]
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I I' : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    hI' : Ne I' 0
    hv : Irreducible (Associates.mk v.asIdeal)
    a : R
    J : Ideal R
    ha : Ne a 0
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    ha_ne_zero : Ne (Associates.mk (Ideal.span (Singleton.singleton a))) 0
    hJ_ne_zero : Ne (Associates.mk J) 0
    a' : R
    J' : Ideal R
    ha' : Ne a' 0
    haJ' : Eq I' (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (In …
    ha'_ne_zero : Ne (Associates.mk (Ideal.span (Singleton.singleton a'))) 0
    hJ'_ne_zero : Ne (Associates.mk J') 0
    h_prod : Eq (HMul.hMul I I') (HMul.hMul (FractionalIdeal.spanSingleton (nonZer …
    ⊢ Eq (HSub.hSub ↑(HAdd.hAdd ((Associates.mk v.asIdeal).count (Associates.mk J) …
  -/
  push_cast
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I I' : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    hI' : Ne I' 0
    hv : Irreducible (Associates.mk v.asIdeal)
    a : R
    J : Ideal R
    ha : Ne a 0
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    ha_ne_zero : Ne (Associates.mk (Ideal.span (Singleton.singleton a))) 0
    hJ_ne_zero : Ne (Associates.mk J) 0
    a' : R
    J' : Ideal R
    ha' : Ne a' 0
    haJ' : Eq I' (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (In …
    ha'_ne_zero : Ne (Associates.mk (Ideal.span (Singleton.singleton a'))) 0
    hJ'_ne_zero : Ne (Associates.mk J') 0
    h_prod : Eq (HMul.hMul I I') (HMul.hMul (FractionalIdeal.spanSingleton (nonZer …
    ⊢ Eq (HSub.hSub (HAdd.hAdd ↑((Associates.mk v.asIdeal).count (Associates.mk J) …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- For nonzero `I, I'`, `val_v(I*I') = val_v(I) + val_v(I')`. If `I` or `I'` is zero, then
`val_v(I*I') = 0`. -/
theorem count_mul' (I I' : FractionalIdeal R⁰ K) :
    count K v (I * I') = if I ≠ 0 ∧ I' ≠ 0 then count K v I + count K v I' else 0 := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I I' : FractionalIdeal (nonZeroDivisors R) K
    ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) (ite (And (Ne I 0) (Ne I' 0) …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I I' : FractionalIdeal (nonZeroDivisors R) K
      h : And (Ne I 0) (Ne I' 0)
      ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) (HAdd.hAdd (FractionalIdeal. …
    -/
  · exact count_mul K v h.1 h.2
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I I' : FractionalIdeal (nonZeroDivisors R) K
      h : Not (And (Ne I 0) (Ne I' 0))
      ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) 0
    -/
  · push_neg at h
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I I' : FractionalIdeal (nonZeroDivisors R) K
      h : Ne I 0 → Eq I' 0
      ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) 0
    -/
    by_cases hI : I = 0
      /-
        case pos
        R : Type u_1
        inst✝⁴ : CommRing R
        K : Type u_2
        inst✝³ : Field K
        inst✝² : Algebra R K
        inst✝¹ : IsFractionRing R K
        inst✝ : IsDedekindDomain R
        v : IsDedekindDomain.HeightOneSpectrum R
        I I' : FractionalIdeal (nonZeroDivisors R) K
        h : Ne I 0 → Eq I' 0
        hI : Eq I 0
        ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) 0
      -/
    · rw [hI, MulZeroClass.zero_mul, count, dif_pos (Eq.refl _)]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝⁴ : CommRing R
        K : Type u_2
        inst✝³ : Field K
        inst✝² : Algebra R K
        inst✝¹ : IsFractionRing R K
        inst✝ : IsDedekindDomain R
        v : IsDedekindDomain.HeightOneSpectrum R
        I I' : FractionalIdeal (nonZeroDivisors R) K
        h : Ne I 0 → Eq I' 0
        hI : Not (Eq I 0)
        ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) 0
      -/
    · rw [h hI, MulZeroClass.mul_zero, count, dif_pos (Eq.refl _)]
      /-
        🎉 no goals
      -/


/-- val_v(1) = 0. -/
theorem count_one : count K v (1 : FractionalIdeal R⁰ K) = 0 := by
  have h1 : (1 : FractionalIdeal R⁰ K) =
      spanSingleton R⁰ ((algebraMap R K) 1)⁻¹ * ↑(1 : Ideal R) := by
    rw [(algebraMap R K).map_one, Ideal.one_eq_top, coeIdeal_top, mul_one, inv_one,
      spanSingleton_one]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    h1 : Eq 1 (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv.i …
    ⊢ Eq (FractionalIdeal.count K v 1) 0
  -/
  rw [count_well_defined K v one_ne_zero h1, Ideal.span_singleton_one, Ideal.one_eq_top, sub_self]
  /-
    🎉 no goals
  -/


theorem count_prod {ι} (s : Finset ι) (I : ι → FractionalIdeal R⁰ K) (hS : ∀ i ∈ s, I i ≠ 0) :
    count K v (∏ i ∈ s, I i) = ∑ i ∈ s, count K v (I i) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    ι : Type u_3
    s : Finset ι
    I : ι → FractionalIdeal (nonZeroDivisors R) K
    hS : ∀ (i : ι), Membership.mem s i → Ne (I i) 0
    ⊢ Eq (FractionalIdeal.count K v (s.prod fun i => I i)) (s.sum fun i => Fractio …
  -/
  induction' s using Finset.induction with i s hi hrec
    /-
      case empty
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      ι : Type u_3
      I : ι → FractionalIdeal (nonZeroDivisors R) K
      hS : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → Ne (I i) 0
      ⊢ Eq (FractionalIdeal.count K v (EmptyCollection.emptyCollection.prod fun i => …
    -/
  · rw [Finset.prod_empty, Finset.sum_empty, count_one]
    /-
      🎉 no goals
    -/
    /-
      case insert
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      ι : Type u_3
      I : ι → FractionalIdeal (nonZeroDivisors R) K
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      hrec : (∀ (i : ι), Membership.mem s i → Ne (I i) 0) → Eq (FractionalIdeal.coun …
      hS : ∀ (i_1 : ι), Membership.mem (Insert.insert i s) i_1 → Ne (I i_1) 0
      ⊢ Eq (FractionalIdeal.count K v ((Insert.insert i s).prod fun i => I i)) ((Ins …
    -/
  · have hS' : ∀ i ∈ s, I i ≠ 0 := fun j hj => hS j (Finset.mem_insert_of_mem hj)
    /-
      case insert
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      ι : Type u_3
      I : ι → FractionalIdeal (nonZeroDivisors R) K
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      hrec : (∀ (i : ι), Membership.mem s i → Ne (I i) 0) → Eq (FractionalIdeal.coun …
      hS : ∀ (i_1 : ι), Membership.mem (Insert.insert i s) i_1 → Ne (I i_1) 0
      hS' : ∀ (i : ι), Membership.mem s i → Ne (I i) 0
      ⊢ Eq (FractionalIdeal.count K v ((Insert.insert i s).prod fun i => I i)) ((Ins …
    -/
    have hS0 : ∏ i ∈ s, I i ≠ 0 := Finset.prod_ne_zero_iff.mpr hS'
    /-
      case insert
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      ι : Type u_3
      I : ι → FractionalIdeal (nonZeroDivisors R) K
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      hrec : (∀ (i : ι), Membership.mem s i → Ne (I i) 0) → Eq (FractionalIdeal.coun …
      hS : ∀ (i_1 : ι), Membership.mem (Insert.insert i s) i_1 → Ne (I i_1) 0
      hS' : ∀ (i : ι), Membership.mem s i → Ne (I i) 0
      hS0 : Ne (s.prod fun i => I i) 0
      ⊢ Eq (FractionalIdeal.count K v ((Insert.insert i s).prod fun i => I i)) ((Ins …
    -/
    have hi0 : I i ≠ 0 := hS i (Finset.mem_insert_self i s)
    /-
      case insert
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      ι : Type u_3
      I : ι → FractionalIdeal (nonZeroDivisors R) K
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      hrec : (∀ (i : ι), Membership.mem s i → Ne (I i) 0) → Eq (FractionalIdeal.coun …
      hS : ∀ (i_1 : ι), Membership.mem (Insert.insert i s) i_1 → Ne (I i_1) 0
      hS' : ∀ (i : ι), Membership.mem s i → Ne (I i) 0
      hS0 : Ne (s.prod fun i => I i) 0
      hi0 : Ne (I i) 0
      ⊢ Eq (FractionalIdeal.count K v ((Insert.insert i s).prod fun i => I i)) ((Ins …
    -/
    rw [Finset.prod_insert hi, Finset.sum_insert hi, count_mul K v hi0 hS0, hrec hS']
    /-
      🎉 no goals
    -/


/-- For every `n ∈ ℕ` and every ideal `I`, `val_v(I^n) = n*val_v(I)`. -/
theorem count_pow (n : ℕ) (I : FractionalIdeal R⁰ K) :
    count K v (I ^ n) = n * count K v I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    n : Nat
    I : FractionalIdeal (nonZeroDivisors R) K
    ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I n)) (HMul.hMul (↑n) (FractionalId …
  -/
  induction' n with n h
    /-
      case zero
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I 0)) (HMul.hMul (↑0) (FractionalId …
    -/
  · rw [pow_zero, ofNat_zero, MulZeroClass.zero_mul, count_one]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      n : Nat
      h : Eq (FractionalIdeal.count K v (HPow.hPow I n)) (HMul.hMul (↑n) (Fractional …
      ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I (HAdd.hAdd n 1))) (HMul.hMul (↑(H …
    -/
  · rw [pow_succ, count_mul']
    /-
      case succ
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      n : Nat
      h : Eq (FractionalIdeal.count K v (HPow.hPow I n)) (HMul.hMul (↑n) (Fractional …
      ⊢ Eq (ite (And (Ne (HPow.hPow I n) 0) (Ne I 0)) (HAdd.hAdd (FractionalIdeal.co …
    -/
    by_cases hI : I = 0
    · have h_neg : ¬(I ^ n ≠ 0 ∧ I ≠ 0) := by
        rw [not_and', not_not, ne_eq]
        intro h
        exact absurd hI h
      /-
        case pos
        R : Type u_1
        inst✝⁴ : CommRing R
        K : Type u_2
        inst✝³ : Field K
        inst✝² : Algebra R K
        inst✝¹ : IsFractionRing R K
        inst✝ : IsDedekindDomain R
        v : IsDedekindDomain.HeightOneSpectrum R
        I : FractionalIdeal (nonZeroDivisors R) K
        n : Nat
        h : Eq (FractionalIdeal.count K v (HPow.hPow I n)) (HMul.hMul (↑n) (Fractional …
        hI : Eq I 0
        h_neg : Not (And (Ne (HPow.hPow I n) 0) (Ne I 0))
        ⊢ Eq (ite (And (Ne (HPow.hPow I n) 0) (Ne I 0)) (HAdd.hAdd (FractionalIdeal.co …
      -/
      rw [if_neg h_neg, hI, count_zero, MulZeroClass.mul_zero]
      /-
        🎉 no goals
      -/
    · rw [if_pos (And.intro (pow_ne_zero n hI) hI), h, Nat.cast_add,
        Nat.cast_one]
      /-
        case neg
        R : Type u_1
        inst✝⁴ : CommRing R
        K : Type u_2
        inst✝³ : Field K
        inst✝² : Algebra R K
        inst✝¹ : IsFractionRing R K
        inst✝ : IsDedekindDomain R
        v : IsDedekindDomain.HeightOneSpectrum R
        I : FractionalIdeal (nonZeroDivisors R) K
        n : Nat
        h : Eq (FractionalIdeal.count K v (HPow.hPow I n)) (HMul.hMul (↑n) (Fractional …
        hI : Not (Eq I 0)
        ⊢ Eq (HAdd.hAdd (HMul.hMul (↑n) (FractionalIdeal.count K v I)) (FractionalIdea …
      -/
      ring
      /-
        🎉 no goals
      -/


/-- `val_v(v) = 1`, when `v` is regarded as a fractional ideal. -/
theorem count_self : count K v (v.asIdeal : FractionalIdeal R⁰ K) = 1 := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    ⊢ Eq (FractionalIdeal.count K v ↑v.asIdeal) 1
  -/
  have hv : (v.asIdeal : FractionalIdeal R⁰ K) ≠ 0 := coeIdeal_ne_zero.mpr v.ne_bot
  have h_self : (v.asIdeal : FractionalIdeal R⁰ K) =
      spanSingleton R⁰ ((algebraMap R K) 1)⁻¹ * ↑v.asIdeal := by
    rw [(algebraMap R K).map_one, inv_one, spanSingleton_one, one_mul]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    hv : Ne (↑v.asIdeal) 0
    h_self : Eq (↑v.asIdeal) (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDiv …
    ⊢ Eq (FractionalIdeal.count K v ↑v.asIdeal) 1
  -/
  have hv_irred : Irreducible (Associates.mk v.asIdeal) := by apply v.associates_irreducible
  rw [count_well_defined K v hv h_self, Associates.count_self hv_irred, Ideal.span_singleton_one,
    ← Ideal.one_eq_top, Associates.mk_one, Associates.factors_one, Associates.count_zero hv_irred,
    ofNat_zero, sub_zero, ofNat_one]


/-- `val_v(v^n) = n` for every `n ∈ ℕ`. -/
theorem count_pow_self (n : ℕ) :
    count K v ((v.asIdeal : FractionalIdeal R⁰ K) ^ n) = n := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    n : Nat
    ⊢ Eq (FractionalIdeal.count K v (HPow.hPow (↑v.asIdeal) n)) ↑n
  -/
  rw [count_pow, count_self, mul_one]
  /-
    🎉 no goals
  -/


/-- `val_v(I⁻ⁿ) = -val_v(Iⁿ)` for every `n ∈ ℤ`. -/
theorem count_neg_zpow (n : ℤ) (I : FractionalIdeal R⁰ K) :
    count K v (I ^ (-n)) = - count K v (I ^ n) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    n : Int
    I : FractionalIdeal (nonZeroDivisors R) K
    ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I (Neg.neg n))) (Neg.neg (Fractiona …
  -/
  by_cases hI : I = 0
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      n : Int
      I : FractionalIdeal (nonZeroDivisors R) K
      hI : Eq I 0
      ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I (Neg.neg n))) (Neg.neg (Fractiona …
    -/
  · by_cases hn : n = 0
      /-
        case pos
        R : Type u_1
        inst✝⁴ : CommRing R
        K : Type u_2
        inst✝³ : Field K
        inst✝² : Algebra R K
        inst✝¹ : IsFractionRing R K
        inst✝ : IsDedekindDomain R
        v : IsDedekindDomain.HeightOneSpectrum R
        n : Int
        I : FractionalIdeal (nonZeroDivisors R) K
        hI : Eq I 0
        hn : Eq n 0
        ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I (Neg.neg n))) (Neg.neg (Fractiona …
      -/
    · rw [hn, neg_zero, zpow_zero, count_one, neg_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝⁴ : CommRing R
        K : Type u_2
        inst✝³ : Field K
        inst✝² : Algebra R K
        inst✝¹ : IsFractionRing R K
        inst✝ : IsDedekindDomain R
        v : IsDedekindDomain.HeightOneSpectrum R
        n : Int
        I : FractionalIdeal (nonZeroDivisors R) K
        hI : Eq I 0
        hn : Not (Eq n 0)
        ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I (Neg.neg n))) (Neg.neg (Fractiona …
      -/
    · rw [hI, zero_zpow n hn, zero_zpow (-n) (neg_ne_zero.mpr hn), count_zero, neg_zero]
      /-
        🎉 no goals
      -/
  · rw [eq_neg_iff_add_eq_zero, ← count_mul K v (zpow_ne_zero _ hI) (zpow_ne_zero _ hI),
      ← zpow_add₀ hI, neg_add_cancel, zpow_zero]
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      n : Int
      I : FractionalIdeal (nonZeroDivisors R) K
      hI : Not (Eq I 0)
      ⊢ Eq (FractionalIdeal.count K v 1) 0
    -/
    exact count_one K v
    /-
      🎉 no goals
    -/


theorem count_inv (I : FractionalIdeal R⁰ K) :
    count K v (I⁻¹) = - count K v I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I : FractionalIdeal (nonZeroDivisors R) K
    ⊢ Eq (FractionalIdeal.count K v (Inv.inv I)) (Neg.neg (FractionalIdeal.count K …
  -/
  rw [← zpow_neg_one, count_neg_zpow K v (1 : ℤ) I, zpow_one]
  /-
    🎉 no goals
  -/


/-- `val_v(Iⁿ) = n*val_v(I)` for every `n ∈ ℤ`. -/
theorem count_zpow (n : ℤ) (I : FractionalIdeal R⁰ K) :
    count K v (I ^ n) = n * count K v I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    n : Int
    I : FractionalIdeal (nonZeroDivisors R) K
    ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I n)) (HMul.hMul n (FractionalIdeal …
  -/
  cases' n with n
    /-
      case ofNat
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      n : Nat
      ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I (Int.ofNat n))) (HMul.hMul (Int.o …
    -/
  · rw [ofNat_eq_coe, zpow_natCast]
    /-
      case ofNat
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      n : Nat
      ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I n)) (HMul.hMul (↑n) (FractionalId …
    -/
    exact count_pow K v n I
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      a✝ : Nat
      ⊢ Eq (FractionalIdeal.count K v (HPow.hPow I (Int.negSucc a✝))) (HMul.hMul (In …
    -/
  · rw [negSucc_coe, count_neg_zpow, zpow_natCast, count_pow]
    /-
      case negSucc
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I : FractionalIdeal (nonZeroDivisors R) K
      a✝ : Nat
      ⊢ Eq (Neg.neg (HMul.hMul (↑(HAdd.hAdd a✝ 1)) (FractionalIdeal.count K v I))) ( …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- `val_v(v^n) = n` for every `n ∈ ℤ`. -/
theorem count_zpow_self (n : ℤ) :
    count K v ((v.asIdeal : FractionalIdeal R⁰ K) ^ n) = n := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    n : Int
    ⊢ Eq (FractionalIdeal.count K v (HPow.hPow (↑v.asIdeal) n)) n
  -/
  rw [count_zpow, count_self, mul_one]
  /-
    🎉 no goals
  -/


/-- If `v ≠ w` are two maximal ideals of `R`, then `val_v(w) = 0`. -/
theorem count_maximal_coprime {w : HeightOneSpectrum R} (hw : w ≠ v) :
    count K v (w.asIdeal : FractionalIdeal R⁰ K) = 0 := by
  have hw_fact : (w.asIdeal : FractionalIdeal R⁰ K) =
      spanSingleton R⁰ ((algebraMap R K) 1)⁻¹ * ↑w.asIdeal := by
    rw [(algebraMap R K).map_one, inv_one, spanSingleton_one, one_mul]
  have hw_ne_zero : (w.asIdeal : FractionalIdeal R⁰ K) ≠ 0 :=
    coeIdeal_ne_zero.mpr w.ne_bot
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v w : IsDedekindDomain.HeightOneSpectrum R
    hw : Ne w v
    hw_fact : Eq (↑w.asIdeal) (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDi …
    hw_ne_zero : Ne (↑w.asIdeal) 0
    ⊢ Eq (FractionalIdeal.count K v ↑w.asIdeal) 0
  -/
  have hv : Irreducible (Associates.mk v.asIdeal) := by apply v.associates_irreducible
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v w : IsDedekindDomain.HeightOneSpectrum R
    hw : Ne w v
    hw_fact : Eq (↑w.asIdeal) (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDi …
    hw_ne_zero : Ne (↑w.asIdeal) 0
    hv : Irreducible (Associates.mk v.asIdeal)
    ⊢ Eq (FractionalIdeal.count K v ↑w.asIdeal) 0
  -/
  have hw' : Irreducible (Associates.mk w.asIdeal) := by apply w.associates_irreducible
  rw [count_well_defined K v hw_ne_zero hw_fact, Ideal.span_singleton_one, ← Ideal.one_eq_top,
    Associates.mk_one, Associates.factors_one, Associates.count_zero hv, ofNat_zero, sub_zero,
    natCast_eq_zero, ← pow_one (Associates.mk w.asIdeal), Associates.factors_prime_pow hw',
    Associates.count_some hv, Multiset.replicate_one, Multiset.count_eq_zero,
    Multiset.mem_singleton]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v w : IsDedekindDomain.HeightOneSpectrum R
    hw : Ne w v
    hw_fact : Eq (↑w.asIdeal) (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDi …
    hw_ne_zero : Ne (↑w.asIdeal) 0
    hv : Irreducible (Associates.mk v.asIdeal)
    hw' : Irreducible (Associates.mk w.asIdeal)
    ⊢ Not (Eq ⟨Associates.mk v.asIdeal, hv⟩ ⟨Associates.mk w.asIdeal, hw'⟩)
  -/
  simp only [Subtype.mk.injEq]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v w : IsDedekindDomain.HeightOneSpectrum R
    hw : Ne w v
    hw_fact : Eq (↑w.asIdeal) (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDi …
    hw_ne_zero : Ne (↑w.asIdeal) 0
    hv : Irreducible (Associates.mk v.asIdeal)
    hw' : Irreducible (Associates.mk w.asIdeal)
    ⊢ Not (Eq (Associates.mk v.asIdeal) (Associates.mk w.asIdeal))
  -/
  rw [Associates.mk_eq_mk_iff_associated, associated_iff_eq, ← HeightOneSpectrum.ext_iff]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v w : IsDedekindDomain.HeightOneSpectrum R
    hw : Ne w v
    hw_fact : Eq (↑w.asIdeal) (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDi …
    hw_ne_zero : Ne (↑w.asIdeal) 0
    hv : Irreducible (Associates.mk v.asIdeal)
    hw' : Irreducible (Associates.mk w.asIdeal)
    ⊢ Not (Eq v w)
  -/
  exact Ne.symm hw
  /-
    🎉 no goals
  -/


theorem count_maximal (w : HeightOneSpectrum R) :
    count K v (w.asIdeal : FractionalIdeal R⁰ K) = if w = v then 1 else 0 := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v w : IsDedekindDomain.HeightOneSpectrum R
    ⊢ Eq (FractionalIdeal.count K v ↑w.asIdeal) (ite (Eq w v) 1 0)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v w : IsDedekindDomain.HeightOneSpectrum R
      h : Eq w v
      ⊢ Eq (FractionalIdeal.count K v ↑w.asIdeal) 1
    -/
  · rw [h, count_self]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v w : IsDedekindDomain.HeightOneSpectrum R
      h : Not (Eq w v)
      ⊢ Eq (FractionalIdeal.count K v ↑w.asIdeal) 0
    -/
  · exact count_maximal_coprime K v h
    /-
      🎉 no goals
    -/


/-- `val_v(∏_{w ≠ v} w^{exps w}) = 0`. -/
theorem count_finprod_coprime (exps : HeightOneSpectrum R → ℤ) :
    count K v (∏ᶠ (w : HeightOneSpectrum R) (_ : w ≠ v),
      (w.asIdeal : (FractionalIdeal R⁰ K)) ^ exps w) = 0 := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    exps : IsDedekindDomain.HeightOneSpectrum R → Int
    ⊢ Eq (FractionalIdeal.count K v (finprod fun w => finprod fun x => HPow.hPow ( …
  -/
  apply finprod_mem_induction fun I => count K v I = 0
    /-
      case hp₀
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      exps : IsDedekindDomain.HeightOneSpectrum R → Int
      ⊢ Eq (FractionalIdeal.count K v 1) 0
    -/
  · exact count_one K v
    /-
      🎉 no goals
    -/
    /-
      case hp₁
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      exps : IsDedekindDomain.HeightOneSpectrum R → Int
      ⊢ ∀ (x y : FractionalIdeal (nonZeroDivisors R) K), Eq (FractionalIdeal.count K …
    -/
  · intro I I' hI hI'
    /-
      case hp₁
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      exps : IsDedekindDomain.HeightOneSpectrum R → Int
      I I' : FractionalIdeal (nonZeroDivisors R) K
      hI : Eq (FractionalIdeal.count K v I) 0
      hI' : Eq (FractionalIdeal.count K v I') 0
      ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) 0
    -/
    by_cases h : I ≠ 0 ∧ I' ≠ 0
      /-
        case pos
        R : Type u_1
        inst✝⁴ : CommRing R
        K : Type u_2
        inst✝³ : Field K
        inst✝² : Algebra R K
        inst✝¹ : IsFractionRing R K
        inst✝ : IsDedekindDomain R
        v : IsDedekindDomain.HeightOneSpectrum R
        exps : IsDedekindDomain.HeightOneSpectrum R → Int
        I I' : FractionalIdeal (nonZeroDivisors R) K
        hI : Eq (FractionalIdeal.count K v I) 0
        hI' : Eq (FractionalIdeal.count K v I') 0
        h : And (Ne I 0) (Ne I' 0)
        ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) 0
      -/
    · rw [count_mul' K v, if_pos h, hI, hI', add_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝⁴ : CommRing R
        K : Type u_2
        inst✝³ : Field K
        inst✝² : Algebra R K
        inst✝¹ : IsFractionRing R K
        inst✝ : IsDedekindDomain R
        v : IsDedekindDomain.HeightOneSpectrum R
        exps : IsDedekindDomain.HeightOneSpectrum R → Int
        I I' : FractionalIdeal (nonZeroDivisors R) K
        hI : Eq (FractionalIdeal.count K v I) 0
        hI' : Eq (FractionalIdeal.count K v I') 0
        h : Not (And (Ne I 0) (Ne I' 0))
        ⊢ Eq (FractionalIdeal.count K v (HMul.hMul I I')) 0
      -/
    · rw [count_mul' K v, if_neg h]
      /-
        🎉 no goals
      -/
    /-
      case hp₂
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      exps : IsDedekindDomain.HeightOneSpectrum R → Int
      ⊢ ∀ (x : IsDedekindDomain.HeightOneSpectrum R), Membership.mem (fun i => Eq i  …
    -/
  · intro w hw
    /-
      case hp₂
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      exps : IsDedekindDomain.HeightOneSpectrum R → Int
      w : IsDedekindDomain.HeightOneSpectrum R
      hw : Membership.mem (fun i => Eq i v → False) w
      ⊢ Eq (FractionalIdeal.count K v (HPow.hPow (↑w.asIdeal) (exps w))) 0
    -/
    rw [count_zpow, count_maximal_coprime K v hw, MulZeroClass.mul_zero]
    /-
      🎉 no goals
    -/


theorem count_finsupp_prod (exps : HeightOneSpectrum R →₀ ℤ) :
    count K v (exps.prod (HeightOneSpectrum.asIdeal · ^ ·)) = exps v := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    exps : Finsupp (IsDedekindDomain.HeightOneSpectrum R) Int
    ⊢ Eq (FractionalIdeal.count K v (exps.prod fun x1 x2 => HPow.hPow (↑x1.asIdeal …
  -/
  rw [Finsupp.prod, count_prod]
  · simp only [count_zpow, count_maximal, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq',
      exps.mem_support_iff, ne_eq, ite_not, ite_eq_right_iff, @eq_comm ℤ 0, imp_self]
    /-
      case hS
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      exps : Finsupp (IsDedekindDomain.HeightOneSpectrum R) Int
      ⊢ ∀ (i : IsDedekindDomain.HeightOneSpectrum R), Membership.mem exps.support i  …
    -/
  · exact fun v hv ↦ zpow_ne_zero _ (coeIdeal_ne_zero.mpr v.ne_bot)
    /-
      🎉 no goals
    -/


/-- If `exps` is finitely supported, then `val_v(∏_w w^{exps w}) = exps v`. -/
theorem count_finprod (exps : HeightOneSpectrum R → ℤ)
    (h_exps : ∀ᶠ v : HeightOneSpectrum R in Filter.cofinite, exps v = 0) :
    count K v (∏ᶠ v : HeightOneSpectrum R,
      (v.asIdeal : FractionalIdeal R⁰ K) ^ exps v) = exps v := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    exps : IsDedekindDomain.HeightOneSpectrum R → Int
    h_exps : Filter.Eventually (fun v => Eq (exps v) 0) Filter.cofinite
    ⊢ Eq (FractionalIdeal.count K v (finprod fun v => HPow.hPow (↑v.asIdeal) (exps …
  -/
  convert count_finsupp_prod K v (Finsupp.mk h_exps.toFinset exps (fun _ ↦ h_exps.mem_toFinset))
  /-
    case h.e'_2.h.e'_9
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    exps : IsDedekindDomain.HeightOneSpectrum R → Int
    h_exps : Filter.Eventually (fun v => Eq (exps v) 0) Filter.cofinite
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (exps v)) ({ support := Set.Fini …
  -/
  rw [finprod_eq_finset_prod_of_mulSupport_subset (s := h_exps.toFinset), Finsupp.prod]
    /-
      case h.e'_2.h.e'_9
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      exps : IsDedekindDomain.HeightOneSpectrum R → Int
      h_exps : Filter.Eventually (fun v => Eq (exps v) 0) Filter.cofinite
      ⊢ Eq ((Set.Finite.toFinset h_exps).prod fun i => HPow.hPow (↑i.asIdeal) (exps  …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_9.h
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      exps : IsDedekindDomain.HeightOneSpectrum R → Int
      h_exps : Filter.Eventually (fun v => Eq (exps v) 0) Filter.cofinite
      ⊢ HasSubset.Subset (Function.mulSupport fun v => HPow.hPow (↑v.asIdeal) (exps  …
    -/
  · rw [Finite.coe_toFinset]
    /-
      case h.e'_2.h.e'_9.h
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      exps : IsDedekindDomain.HeightOneSpectrum R → Int
      h_exps : Filter.Eventually (fun v => Eq (exps v) 0) Filter.cofinite
      ⊢ HasSubset.Subset (Function.mulSupport fun v => HPow.hPow (↑v.asIdeal) (exps  …
    -/
    intro v hv h
    /-
      case h.e'_2.h.e'_9.h
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v✝ : IsDedekindDomain.HeightOneSpectrum R
      exps : IsDedekindDomain.HeightOneSpectrum R → Int
      h_exps : Filter.Eventually (fun v => Eq (exps v) 0) Filter.cofinite
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Membership.mem (Function.mulSupport fun v => HPow.hPow (↑v.asIdeal) (exps …
      h : Membership.mem (setOf fun x => (fun v => Eq (exps v) 0) x) v
      ⊢ False
    -/
    rw [mem_mulSupport, h, zpow_zero] at hv
    /-
      case h.e'_2.h.e'_9.h
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v✝ : IsDedekindDomain.HeightOneSpectrum R
      exps : IsDedekindDomain.HeightOneSpectrum R → Int
      h_exps : Filter.Eventually (fun v => Eq (exps v) 0) Filter.cofinite
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Ne 1 1
      h : Membership.mem (setOf fun x => (fun v => Eq (exps v) 0) x) v
      ⊢ False
    -/
    exact hv (Eq.refl 1)
    /-
      🎉 no goals
    -/


theorem count_coe {J : Ideal R} (hJ : J ≠ 0) :
    count K v J = (Associates.mk v.asIdeal).count (Associates.mk J).factors := by
  rw [count_well_defined K (J := J) (a := 1), Ideal.span_singleton_one, sub_eq_self,
    Nat.cast_eq_zero, ← Ideal.one_eq_top, Associates.mk_one, Associates.factors_one,
    Associates.count_zero v.associates_irreducible]
    /-
      case hI
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      J : Ideal R
      hJ : Ne J 0
      ⊢ Ne (↑J) 0
    -/
  · simpa only [ne_eq, coeIdeal_eq_zero]
    /-
      🎉 no goals
    -/
    /-
      case h_aJ
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      J : Ideal R
      hJ : Ne J 0
      ⊢ Eq (↑J) (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv.i …
    -/
  · simp only [_root_.map_one, inv_one, spanSingleton_one, one_mul]
    /-
      🎉 no goals
    -/


theorem count_coe_nonneg (J : Ideal R) : 0 ≤ count K v J := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    J : Ideal R
    ⊢ LE.le 0 (FractionalIdeal.count K v ↑J)
  -/
  by_cases hJ : J = 0
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      J : Ideal R
      hJ : Eq J 0
      ⊢ LE.le 0 (FractionalIdeal.count K v ↑J)
    -/
  · simp only [hJ, Submodule.zero_eq_bot, coeIdeal_bot, count_zero, le_refl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      J : Ideal R
      hJ : Not (Eq J 0)
      ⊢ LE.le 0 (FractionalIdeal.count K v ↑J)
    -/
  · simp only [count_coe K v hJ, Nat.cast_nonneg]
    /-
      🎉 no goals
    -/


theorem count_mono {I J} (hI : I ≠ 0) (h : I ≤ J) : count K v J ≤ count K v I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I J : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    h : LE.le I J
    ⊢ LE.le (FractionalIdeal.count K v J) (FractionalIdeal.count K v I)
  -/
  by_cases hJ : J = 0
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I J : FractionalIdeal (nonZeroDivisors R) K
      hI : Ne I 0
      h : LE.le I J
      hJ : Eq J 0
      ⊢ LE.le (FractionalIdeal.count K v J) (FractionalIdeal.count K v I)
    -/
  · exact (hI (FractionalIdeal.le_zero_iff.mp (h.trans hJ.le))).elim
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I J : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    h : LE.le I J
    hJ : Not (Eq J 0)
    ⊢ LE.le (FractionalIdeal.count K v J) (FractionalIdeal.count K v I)
  -/
  have := FractionalIdeal.mul_le_mul_left h J⁻¹
  /-
    case neg
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I J : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    h : LE.le I J
    hJ : Not (Eq J 0)
    this : LE.le (HMul.hMul (Inv.inv J) I) (HMul.hMul (Inv.inv J) J)
    ⊢ LE.le (FractionalIdeal.count K v J) (FractionalIdeal.count K v I)
  -/
  rw [inv_mul_cancel₀ hJ, FractionalIdeal.le_one_iff_exists_coeIdeal] at this
  /-
    case neg
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I J : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    h : LE.le I J
    hJ : Not (Eq J 0)
    this : Exists fun I_1 => Eq (↑I_1) (HMul.hMul (Inv.inv J) I)
    ⊢ LE.le (FractionalIdeal.count K v J) (FractionalIdeal.count K v I)
  -/
  obtain ⟨J', hJ'⟩ := this
  /-
    case neg.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    I J : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    h : LE.le I J
    hJ : Not (Eq J 0)
    J' : Ideal R
    hJ' : Eq (↑J') (HMul.hMul (Inv.inv J) I)
    ⊢ LE.le (FractionalIdeal.count K v J) (FractionalIdeal.count K v I)
  -/
  rw [← mul_inv_cancel_left₀ hJ I, ← hJ', count_mul K v hJ, le_add_iff_nonneg_right]
    /-
      case neg.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I J : FractionalIdeal (nonZeroDivisors R) K
      hI : Ne I 0
      h : LE.le I J
      hJ : Not (Eq J 0)
      J' : Ideal R
      hJ' : Eq (↑J') (HMul.hMul (Inv.inv J) I)
      ⊢ LE.le 0 (FractionalIdeal.count K v ↑J')
    -/
  · exact count_coe_nonneg K v J'
    /-
      🎉 no goals
    -/
    /-
      case neg.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      I J : FractionalIdeal (nonZeroDivisors R) K
      hI : Ne I 0
      h : LE.le I J
      hJ : Not (Eq J 0)
      J' : Ideal R
      hJ' : Eq (↑J') (HMul.hMul (Inv.inv J) I)
      ⊢ Ne (↑J') 0
    -/
  · exact hJ' ▸ mul_ne_zero (inv_ne_zero hJ) hI
    /-
      🎉 no goals
    -/


/-- If `I` is a nonzero fractional ideal, then `I` is equal to the product `∏_v v^(count K v I)`. -/
theorem finprod_heightOneSpectrum_factorization' {I : FractionalIdeal R⁰ K} (hI : I ≠ 0) :
    ∏ᶠ v : HeightOneSpectrum R, (v.asIdeal : FractionalIdeal R⁰ K) ^ (count K v I) = I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (FractionalIdeal.count K v I)) I
  -/
  have h := (choose_spec (choose_spec (exists_eq_spanSingleton_mul I))).2
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    h : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv.in …
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (FractionalIdeal.count K v I)) I
  -/
  conv_rhs => rw [← finprod_heightOneSpectrum_factorization hI h]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    h : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv.in …
    ⊢ Eq (finprod fun v => HPow.hPow (↑v.asIdeal) (FractionalIdeal.count K v I)) ( …
  -/
  apply finprod_congr
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    h : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv.in …
    ⊢ ∀ (x : IsDedekindDomain.HeightOneSpectrum R), Eq (HPow.hPow (↑x.asIdeal) (Fr …
  -/
  intro w
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    h : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv.in …
    w : IsDedekindDomain.HeightOneSpectrum R
    ⊢ Eq (HPow.hPow (↑w.asIdeal) (FractionalIdeal.count K w I)) (HPow.hPow (↑w.asI …
  -/
  apply congr_arg
  /-
    case h.h
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    h : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv.in …
    w : IsDedekindDomain.HeightOneSpectrum R
    ⊢ Eq (FractionalIdeal.count K w I) (HSub.hSub ↑((Associates.mk w.asIdeal).coun …
  -/
  rw [count_ne_zero K w hI]
  /-
    🎉 no goals
  -/


/-- If `I ≠ 0`, then `val_v(I) = 0` for all but finitely many maximal ideals of `R`. -/
theorem finite_factors' {I : FractionalIdeal R⁰ K} (hI : I ≠ 0) {a : R}
    {J : Ideal R} (haJ : I = spanSingleton R⁰ ((algebraMap R K) a)⁻¹ * ↑J) :
    ∀ᶠ v : HeightOneSpectrum R in Filter.cofinite,
      ((Associates.mk v.asIdeal).count (Associates.mk J).factors : ℤ) -
        (Associates.mk v.asIdeal).count (Associates.mk (Ideal.span {a})).factors = 0 := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    ⊢ Filter.Eventually (fun v => Eq (HSub.hSub ↑((Associates.mk v.asIdeal).count  …
  -/
  have ha_ne_zero : Ideal.span {a} ≠ 0 := constant_factor_ne_zero hI haJ
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    ha_ne_zero : Ne (Ideal.span (Singleton.singleton a)) 0
    ⊢ Filter.Eventually (fun v => Eq (HSub.hSub ↑((Associates.mk v.asIdeal).count  …
  -/
  have hJ_ne_zero : J ≠ 0 := ideal_factor_ne_zero hI haJ
  have h_subset :
    {v : HeightOneSpectrum R | ¬((Associates.mk v.asIdeal).count (Associates.mk J).factors : ℤ) -
      ↑((Associates.mk v.asIdeal).count (Associates.mk (Ideal.span {a})).factors) = 0} ⊆
    {v : HeightOneSpectrum R | v.asIdeal ∣ J} ∪
      {v : HeightOneSpectrum R | v.asIdeal ∣ Ideal.span {a}} := by
    intro v hv
    have hv_irred : Irreducible v.asIdeal := v.irreducible
    by_contra h_nmem
    rw [mem_union, mem_setOf_eq, mem_setOf_eq] at h_nmem
    push_neg at h_nmem
    rw [← Associates.count_ne_zero_iff_dvd ha_ne_zero hv_irred, not_not,
      ← Associates.count_ne_zero_iff_dvd hJ_ne_zero hv_irred, not_not] at h_nmem
    rw [mem_setOf_eq, h_nmem.1, h_nmem.2, sub_self] at hv
    exact hv (Eq.refl 0)
  exact Finite.subset (Finite.union (Ideal.finite_factors (ideal_factor_ne_zero hI haJ))
    (Ideal.finite_factors (constant_factor_ne_zero hI haJ))) h_subset


/-- `val_v(I) = 0` for all but finitely many maximal ideals of `R`. -/
theorem finite_factors (I : FractionalIdeal R⁰ K) :
    ∀ᶠ v : HeightOneSpectrum R in Filter.cofinite, count K v I = 0 := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    I : FractionalIdeal (nonZeroDivisors R) K
    ⊢ Filter.Eventually (fun v => Eq (FractionalIdeal.count K v I) 0) Filter.cofin …
  -/
  by_cases hI : I = 0
  · simp only [hI, count_zero, Filter.eventually_cofinite, not_true_eq_false, setOf_false,
      finite_empty]
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      I : FractionalIdeal (nonZeroDivisors R) K
      hI : Not (Eq I 0)
      ⊢ Filter.Eventually (fun v => Eq (FractionalIdeal.count K v I) 0) Filter.cofin …
    -/
  · convert finite_factors' hI (choose_spec (choose_spec (exists_eq_spanSingleton_mul I))).2
    /-
      case h.e'_2.h.h.e'_2
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDedekindDomain R
      I : FractionalIdeal (nonZeroDivisors R) K
      hI : Not (Eq I 0)
      x✝ : IsDedekindDomain.HeightOneSpectrum R
      ⊢ Eq (FractionalIdeal.count K x✝ I) (HSub.hSub ↑((Associates.mk x✝.asIdeal).co …
    -/
    rw [count_ne_zero K _ hI]
    /-
      🎉 no goals
    -/


