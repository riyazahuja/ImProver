open Function Ideal IsDedekindDomain HeightOneSpectrum in
/-- For any non-zero `x` in `𝓞 K`, the prduct of `w x`, where `w` runs over `FinitePlace K`, is
equal to the inverse of the absolute value of `Algebra.norm ℤ x`. -/
theorem FinitePlace.prod_eq_inv_abs_norm_int {x : 𝓞 K} (h_x_nezero : x ≠ 0) :
    ∏ᶠ w : FinitePlace K, w x = (|norm ℤ x| : ℝ)⁻¹ := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    ⊢ Eq (finprod fun w => w ↑x) (Inv.inv (abs ↑((Algebra.norm Int) x)))
  -/
  simp only [← finprod_comp_equiv equivHeightOneSpectrum.symm, equivHeightOneSpectrum_symm_apply]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    ⊢ Eq (finprod fun i => Norm.norm ((NumberField.embedding i) ↑x)) (Inv.inv (abs …
  -/
  refine (inv_eq_of_mul_eq_one_left ?_).symm
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    ⊢ Eq (HMul.hMul (finprod fun i => Norm.norm ((NumberField.embedding i) ↑x)) (a …
  -/
  norm_cast
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    ⊢ Eq (HMul.hMul (finprod fun i => Norm.norm ((NumberField.embedding i) ↑x)) ↑( …
  -/
  have h_span_nezero : span {x} ≠ 0 := by simp [h_x_nezero]
  rw [Int.abs_eq_natAbs, ← absNorm_span_singleton,
    ← finprod_heightOneSpectrum_factorization h_span_nezero, Int.cast_natCast]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    h_span_nezero : Ne (Ideal.span (Singleton.singleton x)) 0
    ⊢ Eq (HMul.hMul (finprod fun i => Norm.norm ((NumberField.embedding i) ↑x)) ↑( …
  -/
  let t₀ := {v : HeightOneSpectrum (𝓞 K) | x ∈ v.asIdeal}
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    h_span_nezero : Ne (Ideal.span (Singleton.singleton x)) 0
    t₀ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    ⊢ Eq (HMul.hMul (finprod fun i => Norm.norm ((NumberField.embedding i) ↑x)) ↑( …
  -/
  have h_fin₀ : t₀.Finite := by simp only [← dvd_span_singleton, finite_factors h_span_nezero, t₀]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    h_span_nezero : Ne (Ideal.span (Singleton.singleton x)) 0
    t₀ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    h_fin₀ : t₀.Finite
    ⊢ Eq (HMul.hMul (finprod fun i => Norm.norm ((NumberField.embedding i) ↑x)) ↑( …
  -/
  let t₁ := (fun v : HeightOneSpectrum (𝓞 K) ↦ ‖embedding v x‖).mulSupport
  let t₂ :=
    (fun v : HeightOneSpectrum (𝓞 K) ↦ (absNorm (v.maxPowDividing (span {x})) : ℝ)).mulSupport
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    h_span_nezero : Ne (Ideal.span (Singleton.singleton x)) 0
    t₀ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    h_fin₀ : t₀.Finite
    t₁ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    t₂ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    ⊢ Eq (HMul.hMul (finprod fun i => Norm.norm ((NumberField.embedding i) ↑x)) ↑( …
  -/
  have h_fin₁ : t₁.Finite := h_fin₀.subset <| by simp [norm_eq_one_iff_not_mem, t₁, t₀]
  have h_fin₂ : t₂.Finite := by
    refine h_fin₀.subset ?_
    simp only [Set.le_eq_subset, mulSupport_subset_iff, Set.mem_setOf_eq, t₂, t₀,
      maxPowDividing, ← dvd_span_singleton]
    intro v hv
    simp only [map_pow, Nat.cast_pow, ← pow_zero (absNorm v.asIdeal : ℝ)] at hv
    classical
    refine (Associates.count_ne_zero_iff_dvd h_span_nezero (irreducible v)).1 <| fun h ↦ hv ?_
    congr
  have h_prod : (absNorm (∏ᶠ (v : HeightOneSpectrum (𝓞 K)), v.maxPowDividing (span {x})) : ℝ) =
      ∏ᶠ (v : HeightOneSpectrum (𝓞 K)), (absNorm (v.maxPowDividing (span {x})) : ℝ) :=
    ((Nat.castRingHom ℝ).toMonoidHom.comp absNorm.toMonoidHom).map_finprod_of_preimage_one
      (by simp) _
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    h_span_nezero : Ne (Ideal.span (Singleton.singleton x)) 0
    t₀ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    h_fin₀ : t₀.Finite
    t₁ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    t₂ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    h_fin₁ : t₁.Finite
    h_fin₂ : t₂.Finite
    h_prod : Eq (↑(Ideal.absNorm (finprod fun v => v.maxPowDividing (Ideal.span (S …
    ⊢ Eq (HMul.hMul (finprod fun i => Norm.norm ((NumberField.embedding i) ↑x)) ↑( …
  -/
  rw [h_prod, ← finprod_mul_distrib h_fin₁ h_fin₂]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    h_span_nezero : Ne (Ideal.span (Singleton.singleton x)) 0
    t₀ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    h_fin₀ : t₀.Finite
    t₁ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    t₂ : Set (IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)) : …
    h_fin₁ : t₁.Finite
    h_fin₂ : t₂.Finite
    h_prod : Eq (↑(Ideal.absNorm (finprod fun v => v.maxPowDividing (Ideal.span (S …
    ⊢ Eq (finprod fun i => HMul.hMul (Norm.norm ((NumberField.embedding i) ↑x)) ↑( …
  -/
  exact finprod_eq_one_of_forall_eq_one fun v ↦ v.embedding_mul_absNorm h_x_nezero
  /-
    🎉 no goals
  -/


/-- For any non-zero `x` in `K`, the prduct of `w x`, where `w` runs over `FinitePlace K`, is
equal to the inverse of the absolute value of `Algebra.norm ℚ x`. -/
theorem FinitePlace.prod_eq_inv_abs_norm {x : K} (h_x_nezero : x ≠ 0) :
    ∏ᶠ w : FinitePlace K, w x = |(Algebra.norm ℚ) x|⁻¹ := by
  --reduce to 𝓞 K
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    h_x_nezero : Ne x 0
    ⊢ Eq (finprod fun w => w x) ↑(Inv.inv (abs ((Algebra.norm Rat) x)))
  -/
  rcases IsFractionRing.div_surjective (A := 𝓞 K) x with ⟨a, b, hb, rfl⟩
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    hb : Membership.mem (nonZeroDivisors (NumberField.RingOfIntegers K)) b
    h_x_nezero : Ne (HDiv.hDiv ((algebraMap (NumberField.RingOfIntegers K) K) a) ( …
    ⊢ Eq (finprod fun w => w (HDiv.hDiv ((algebraMap (NumberField.RingOfIntegers K …
  -/
  apply nonZeroDivisors.ne_zero at hb
  have ha : a ≠ 0 := by
    rintro rfl
    simp at h_x_nezero
  simp_rw [map_div₀, Rat.cast_inv, Rat.cast_abs, finprod_div_distrib (mulSupport_finite_int ha)
    (mulSupport_finite_int hb), prod_eq_inv_abs_norm_int ha, prod_eq_inv_abs_norm_int hb]
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    h_x_nezero : Ne (HDiv.hDiv ((algebraMap (NumberField.RingOfIntegers K) K) a) ( …
    hb : Ne b 0
    ha : Ne a 0
    ⊢ Eq (HDiv.hDiv (Inv.inv (abs ↑((Algebra.norm Int) a))) (Inv.inv (abs ↑((Algeb …
  -/
  rw [← inv_eq_iff_eq_inv, inv_inv_div_inv, ← abs_div]
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    h_x_nezero : Ne (HDiv.hDiv ((algebraMap (NumberField.RingOfIntegers K) K) a) ( …
    hb : Ne b 0
    ha : Ne a 0
    ⊢ Eq (abs (HDiv.hDiv ↑((Algebra.norm Int) a) ↑((Algebra.norm Int) b))) (abs ↑( …
  -/
  congr
  /-
    case intro.intro.intro.e_a
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    h_x_nezero : Ne (HDiv.hDiv ((algebraMap (NumberField.RingOfIntegers K) K) a) ( …
    hb : Ne b 0
    ha : Ne a 0
    ⊢ Eq (HDiv.hDiv ↑((Algebra.norm Int) a) ↑((Algebra.norm Int) b)) ↑((Algebra.no …
  -/
  have hb₀ : ((Algebra.norm ℤ) b : ℝ) ≠ 0 := by simp [hb]
  /-
    case intro.intro.intro.e_a
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    h_x_nezero : Ne (HDiv.hDiv ((algebraMap (NumberField.RingOfIntegers K) K) a) ( …
    hb : Ne b 0
    ha : Ne a 0
    hb₀ : Ne (↑((Algebra.norm Int) b)) 0
    ⊢ Eq (HDiv.hDiv ↑((Algebra.norm Int) a) ↑((Algebra.norm Int) b)) ↑((Algebra.no …
  -/
  refine (eq_div_of_mul_eq hb₀ ?_).symm
  /-
    case intro.intro.intro.e_a
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    h_x_nezero : Ne (HDiv.hDiv ((algebraMap (NumberField.RingOfIntegers K) K) a) ( …
    hb : Ne b 0
    ha : Ne a 0
    hb₀ : Ne (↑((Algebra.norm Int) b)) 0
    ⊢ Eq (HMul.hMul ↑((Algebra.norm Rat) (HDiv.hDiv ((algebraMap (NumberField.Ring …
  -/
  norm_cast
  rw [coe_norm_int a, coe_norm_int b, ← MonoidHom.map_mul, div_mul_cancel₀ _
    (RingOfIntegers.coe_ne_zero_iff.mpr hb)]


open FinitePlace in
/-- The Product Formula for the Number Field `K`. -/
theorem prod_abs_eq_one {x : K} (h_x_nezero : x ≠ 0) :
    (∏ w : InfinitePlace K, w x ^ w.mult) * ∏ᶠ w : FinitePlace K, w x = 1 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    h_x_nezero : Ne x 0
    ⊢ Eq (HMul.hMul (Finset.univ.prod fun w => HPow.hPow (w x) w.mult) (finprod fu …
  -/
  simp [prod_eq_inv_abs_norm, InfinitePlace.prod_eq_abs_norm, h_x_nezero]
  /-
    🎉 no goals
  -/


