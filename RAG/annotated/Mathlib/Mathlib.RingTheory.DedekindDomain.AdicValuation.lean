open scoped Classical in
/-- The additive `v`-adic valuation of `r ∈ R` is the exponent of `v` in the factorization of the
ideal `(r)`, if `r` is nonzero, or infinity, if `r = 0`. `intValuationDef` is the corresponding
multiplicative valuation. -/
def intValuationDef (r : R) : ℤₘ₀ :=
  if r = 0 then 0
  else
    ↑(Multiplicative.ofAdd
      (-(Associates.mk v.asIdeal).count (Associates.mk (Ideal.span {r} : Ideal R)).factors : ℤ))



theorem intValuationDef_if_pos {r : R} (hr : r = 0) : v.intValuationDef r = 0 :=
  if_pos hr


@[simp]
theorem intValuationDef_zero : v.intValuationDef 0 = 0 :=
  if_pos rfl


open scoped Classical in
theorem intValuationDef_if_neg {r : R} (hr : r ≠ 0) :
    v.intValuationDef r =
      Multiplicative.ofAdd
        (-(Associates.mk v.asIdeal).count (Associates.mk (Ideal.span {r} : Ideal R)).factors : ℤ) :=
  if_neg hr


/-- Nonzero elements have nonzero adic valuation. -/
theorem intValuation_ne_zero (x : R) (hx : x ≠ 0) : v.intValuationDef x ≠ 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    x : R
    hx : Ne x 0
    ⊢ Ne (v.intValuationDef x) 0
  -/
  rw [intValuationDef, if_neg hx]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    x : R
    hx : Ne x 0
    ⊢ Ne (↑(Multiplicative.ofAdd (Neg.neg ↑((Associates.mk v.asIdeal).count (Assoc …
  -/
  exact WithZero.coe_ne_zero
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-09")]
alias int_valuation_ne_zero := intValuation_ne_zero


/-- Nonzero divisors have nonzero valuation. -/
theorem intValuation_ne_zero' (x : nonZeroDivisors R) : v.intValuationDef x ≠ 0 :=
  v.intValuation_ne_zero x (nonZeroDivisors.coe_ne_zero x)


@[deprecated (since := "2024-07-09")]
alias int_valuation_ne_zero' := intValuation_ne_zero'


/-- Nonzero divisors have valuation greater than zero. -/
theorem intValuation_zero_le (x : nonZeroDivisors R) : 0 < v.intValuationDef x := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    x : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    ⊢ LT.lt 0 (v.intValuationDef ↑x)
  -/
  rw [v.intValuationDef_if_neg (nonZeroDivisors.coe_ne_zero x)]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    x : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    ⊢ LT.lt 0 ↑(Multiplicative.ofAdd (Neg.neg ↑((Associates.mk v.asIdeal).count (A …
  -/
  exact WithZero.zero_lt_coe _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-09")]
alias int_valuation_zero_le := intValuation_zero_le


/-- The `v`-adic valuation on `R` is bounded above by 1. -/
theorem intValuation_le_one (x : R) : v.intValuationDef x ≤ 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    x : R
    ⊢ LE.le (v.intValuationDef x) 1
  -/
  rw [intValuationDef]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    v : IsDedekindDomain.HeightOneSpectrum R
    x : R
    ⊢ LE.le (ite (Eq x 0) 0 ↑(Multiplicative.ofAdd (Neg.neg ↑((Associates.mk v.asI …
  -/
  by_cases hx : x = 0
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      x : R
      hx : Eq x 0
      ⊢ LE.le (ite (Eq x 0) 0 ↑(Multiplicative.ofAdd (Neg.neg ↑((Associates.mk v.asI …
    -/
  · rw [if_pos hx]; exact WithZero.zero_le 1
                    /-
                      🎉 no goals
                    -/
  · rw [if_neg hx, ← WithZero.coe_one, ← ofAdd_zero, WithZero.coe_le_coe, ofAdd_le,
      Right.neg_nonpos_iff]
    /-
      case neg
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      v : IsDedekindDomain.HeightOneSpectrum R
      x : R
      hx : Not (Eq x 0)
      ⊢ LE.le 0 ↑((Associates.mk v.asIdeal).count (Associates.mk (Ideal.span (Single …
    -/
    exact Int.natCast_nonneg _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-09")]
alias int_valuation_le_one := intValuation_le_one


/-- The `v`-adic valuation of `r ∈ R` is less than 1 if and only if `v` divides the ideal `(r)`. -/
theorem intValuation_lt_one_iff_dvd (r : R) :
    v.intValuationDef r < 1 ↔ v.asIdeal ∣ Ideal.span {r} := by
  classical
  rw [intValuationDef]
  split_ifs with hr
  · simp [hr]
  · rw [← WithZero.coe_one, ← ofAdd_zero, WithZero.coe_lt_coe, ofAdd_lt, neg_lt_zero, ←
      Int.ofNat_zero, Int.ofNat_lt, zero_lt_iff]
    have h : (Ideal.span {r} : Ideal R) ≠ 0 := by
      rw [Ne, Ideal.zero_eq_bot, Ideal.span_singleton_eq_bot]
      exact hr
    apply Associates.count_ne_zero_iff_dvd h (by apply v.irreducible)


@[deprecated (since := "2024-07-09")]
alias int_valuation_lt_one_iff_dvd := intValuation_lt_one_iff_dvd


/-- The `v`-adic valuation of `r ∈ R` is less than `Multiplicative.ofAdd (-n)` if and only if
`vⁿ` divides the ideal `(r)`. -/
theorem intValuation_le_pow_iff_dvd (r : R) (n : ℕ) :
    v.intValuationDef r ≤ Multiplicative.ofAdd (-(n : ℤ)) ↔ v.asIdeal ^ n ∣ Ideal.span {r} := by
  classical
  rw [intValuationDef]
  split_ifs with hr
  · simp_rw [hr, Ideal.dvd_span_singleton, zero_le', Submodule.zero_mem]
  · rw [WithZero.coe_le_coe, ofAdd_le, neg_le_neg_iff, Int.ofNat_le, Ideal.dvd_span_singleton, ←
      Associates.le_singleton_iff,
      Associates.prime_pow_dvd_iff_le (Associates.mk_ne_zero'.mpr hr)
        (by apply v.associates_irreducible)]


@[deprecated (since := "2024-07-09")]
alias int_valuation_le_pow_iff_dvd := intValuation_le_pow_iff_dvd


/-- The `v`-adic valuation of `0 : R` equals 0. -/
theorem intValuation.map_zero' : v.intValuationDef 0 = 0 :=
  v.intValuationDef_if_pos (Eq.refl 0)


@[deprecated (since := "2024-07-09")]
alias IntValuation.map_zero' := intValuation.map_zero'


/-- The `v`-adic valuation of `1 : R` equals 1. -/
theorem intValuation.map_one' : v.intValuationDef 1 = 1 := by
  classical
  rw [v.intValuationDef_if_neg (zero_ne_one.symm : (1 : R) ≠ 0), Ideal.span_singleton_one, ←
    Ideal.one_eq_top, Associates.mk_one, Associates.factors_one,
    Associates.count_zero (by apply v.associates_irreducible), Int.ofNat_zero, neg_zero, ofAdd_zero,
    WithZero.coe_one]


@[deprecated (since := "2024-07-09")]
alias IntValuation.map_one' := intValuation.map_one'


/-- The `v`-adic valuation of a product equals the product of the valuations. -/
theorem intValuation.map_mul' (x y : R) :
    v.intValuationDef (x * y) = v.intValuationDef x * v.intValuationDef y := by
  classical
  simp only [intValuationDef]
  by_cases hx : x = 0
  · rw [hx, zero_mul, if_pos (Eq.refl _), zero_mul]
  · by_cases hy : y = 0
    · rw [hy, mul_zero, if_pos (Eq.refl _), mul_zero]
    · rw [if_neg hx, if_neg hy, if_neg (mul_ne_zero hx hy), ← WithZero.coe_mul, WithZero.coe_inj, ←
        ofAdd_add, ← Ideal.span_singleton_mul_span_singleton, ← Associates.mk_mul_mk, ← neg_add,
        Associates.count_mul (by apply Associates.mk_ne_zero'.mpr hx)
          (by apply Associates.mk_ne_zero'.mpr hy) (by apply v.associates_irreducible)]
      rfl


@[deprecated (since := "2024-07-09")]
alias IntValuation.map_mul' := intValuation.map_mul'


theorem intValuation.le_max_iff_min_le {a b c : ℕ} :
    Multiplicative.ofAdd (-c : ℤ) ≤
      max (Multiplicative.ofAdd (-a : ℤ)) (Multiplicative.ofAdd (-b : ℤ)) ↔
      min a b ≤ c := by
  rw [le_max_iff, ofAdd_le, ofAdd_le, neg_le_neg_iff, neg_le_neg_iff, Int.ofNat_le, Int.ofNat_le, ←
    min_le_iff]


@[deprecated (since := "2024-07-09")]
alias IntValuation.le_max_iff_min_le := intValuation.le_max_iff_min_le


/-- The `v`-adic valuation of a sum is bounded above by the maximum of the valuations. -/
theorem intValuation.map_add_le_max' (x y : R) :
    v.intValuationDef (x + y) ≤ max (v.intValuationDef x) (v.intValuationDef y) := by
  classical
  by_cases hx : x = 0
  · rw [hx, zero_add]
    conv_rhs => rw [intValuationDef, if_pos (Eq.refl _)]
    rw [max_eq_right (WithZero.zero_le (v.intValuationDef y))]
  · by_cases hy : y = 0
    · rw [hy, add_zero]
      conv_rhs => rw [max_comm, intValuationDef, if_pos (Eq.refl _)]
      rw [max_eq_right (WithZero.zero_le (v.intValuationDef x))]
    · by_cases hxy : x + y = 0
      · rw [intValuationDef, if_pos hxy]; exact zero_le'
      · rw [v.intValuationDef_if_neg hxy, v.intValuationDef_if_neg hx, v.intValuationDef_if_neg hy,
          WithZero.le_max_iff, intValuation.le_max_iff_min_le]
        set nmin :=
          min ((Associates.mk v.asIdeal).count (Associates.mk (Ideal.span { x })).factors)
            ((Associates.mk v.asIdeal).count (Associates.mk (Ideal.span { y })).factors)
        have h_dvd_x : x ∈ v.asIdeal ^ nmin := by
          rw [← Associates.le_singleton_iff x nmin _,
            Associates.prime_pow_dvd_iff_le (Associates.mk_ne_zero'.mpr hx) _]
          · exact min_le_left _ _
          apply v.associates_irreducible
        have h_dvd_y : y ∈ v.asIdeal ^ nmin := by
          rw [← Associates.le_singleton_iff y nmin _,
            Associates.prime_pow_dvd_iff_le (Associates.mk_ne_zero'.mpr hy) _]
          · exact min_le_right _ _
          apply v.associates_irreducible
        have h_dvd_xy : Associates.mk v.asIdeal ^ nmin ≤ Associates.mk (Ideal.span {x + y}) := by
          rw [Associates.le_singleton_iff]
          exact Ideal.add_mem (v.asIdeal ^ nmin) h_dvd_x h_dvd_y
        rw [Associates.prime_pow_dvd_iff_le (Associates.mk_ne_zero'.mpr hxy) _] at h_dvd_xy
        · exact h_dvd_xy
        apply v.associates_irreducible


@[deprecated (since := "2024-07-09")]
alias IntValuation.map_add_le_max' := intValuation.map_add_le_max'


/-- The `v`-adic valuation on `R`. -/
@[simps]
def intValuation : Valuation R ℤₘ₀ where
  toFun := v.intValuationDef
  map_zero' := intValuation.map_zero' v
  map_one' := intValuation.map_one' v
  map_mul' := intValuation.map_mul' v
  map_add_le_max' := intValuation.map_add_le_max' v



theorem intValuation_apply {r : R} (v : IsDedekindDomain.HeightOneSpectrum R) :
    intValuation v r = intValuationDef v r := rfl


/-- There exists `π ∈ R` with `v`-adic valuation `Multiplicative.ofAdd (-1)`. -/
theorem intValuation_exists_uniformizer :
    ∃ π : R, v.intValuationDef π = Multiplicative.ofAdd (-1 : ℤ) := by
  classical
  have hv : _root_.Irreducible (Associates.mk v.asIdeal) := v.associates_irreducible
  have hlt : v.asIdeal ^ 2 < v.asIdeal := by
    rw [← Ideal.dvdNotUnit_iff_lt]
    exact
      ⟨v.ne_bot, v.asIdeal, (not_congr Ideal.isUnit_iff).mpr (Ideal.IsPrime.ne_top v.isPrime),
        sq v.asIdeal⟩
  obtain ⟨π, mem, nmem⟩ := SetLike.exists_of_lt hlt
  have hπ : Associates.mk (Ideal.span {π}) ≠ 0 := by
    rw [Associates.mk_ne_zero']
    intro h
    rw [h] at nmem
    exact nmem (Submodule.zero_mem (v.asIdeal ^ 2))
  use π
  rw [intValuationDef, if_neg (Associates.mk_ne_zero'.mp hπ), WithZero.coe_inj]
  apply congr_arg
  rw [neg_inj, ← Int.ofNat_one, Int.natCast_inj]
  rw [← Ideal.dvd_span_singleton, ← Associates.mk_le_mk_iff_dvd] at mem nmem
  rw [← pow_one (Associates.mk v.asIdeal), Associates.prime_pow_dvd_iff_le hπ hv] at mem
  rw [Associates.mk_pow, Associates.prime_pow_dvd_iff_le hπ hv, not_le] at nmem
  exact Nat.eq_of_le_of_lt_succ mem nmem


@[deprecated (since := "2024-07-09")]
alias int_valuation_exists_uniformizer := intValuation_exists_uniformizer


/-- The `I`-adic valuation of a generator of `I` equals `(-1 : ℤₘ₀)` -/
theorem intValuation_singleton {r : R} (hr : r ≠ 0) (hv : v.asIdeal = Ideal.span {r}) :
    v.intValuation r = Multiplicative.ofAdd (-1 : ℤ) := by
  classical
  rw [intValuation_apply, v.intValuationDef_if_neg hr, ← hv, Associates.count_self, Int.ofNat_one,
    ofAdd_neg, WithZero.coe_inv]
  apply v.associates_irreducible


/-- The `v`-adic valuation of `x ∈ K` is the valuation of `r` divided by the valuation of `s`,
where `r` and `s` are chosen so that `x = r/s`. -/
def valuation (v : HeightOneSpectrum R) : Valuation K ℤₘ₀ :=
  v.intValuation.extendToLocalization
    (fun r hr => Set.mem_compl <| v.intValuation_ne_zero' ⟨r, hr⟩) K


theorem valuation_def (x : K) :
    v.valuation x =
      v.intValuation.extendToLocalization
        (fun r hr => Set.mem_compl (v.intValuation_ne_zero' ⟨r, hr⟩)) K x :=
  rfl


/-- The `v`-adic valuation of `r/s ∈ K` is the valuation of `r` divided by the valuation of `s`. -/
theorem valuation_of_mk' {r : R} {s : nonZeroDivisors R} :
    v.valuation (IsLocalization.mk' K r s) = v.intValuation r / v.intValuation s := by
  erw [valuation_def, (IsLocalization.toLocalizationMap (nonZeroDivisors R) K).lift_mk',
    div_eq_mul_inv, mul_eq_mul_left_iff]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    ⊢ Or (Eq (↑(Inv.inv ((IsUnit.liftRight ((↑v.intValuation.toMonoidWithZeroHom). …
  -/
  left
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    ⊢ Eq (↑(Inv.inv ((IsUnit.liftRight ((↑v.intValuation.toMonoidWithZeroHom).rest …
  -/
  rw [Units.val_inv_eq_inv_val, inv_inj]
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    ⊢ Eq (↑((IsUnit.liftRight ((↑v.intValuation.toMonoidWithZeroHom).restrict (non …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The `v`-adic valuation on `K` extends the `v`-adic valuation on `R`. -/
theorem valuation_of_algebraMap (r : R) : v.valuation (algebraMap R K r) = v.intValuation r := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    ⊢ Eq (v.valuation ((algebraMap R K) r)) (v.intValuation r)
  -/
  rw [valuation_def, Valuation.extendToLocalization_apply_map_apply]
  /-
    🎉 no goals
  -/


open scoped algebraMap in
lemma valuation_eq_intValuationDef (r : R) : v.valuation (r : K) = v.intValuationDef r :=
  Valuation.extendToLocalization_apply_map_apply ..


/-- The `v`-adic valuation on `R` is bounded above by 1. -/
theorem valuation_le_one (r : R) : v.valuation (algebraMap R K r) ≤ 1 := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    ⊢ LE.le (v.valuation ((algebraMap R K) r)) 1
  -/
  rw [valuation_of_algebraMap]; exact v.intValuation_le_one r
                                /-
                                  🎉 no goals
                                -/


/-- The `v`-adic valuation of `r ∈ R` is less than 1 if and only if `v` divides the ideal `(r)`. -/
theorem valuation_lt_one_iff_dvd (r : R) :
    v.valuation (algebraMap R K r) < 1 ↔ v.asIdeal ∣ Ideal.span {r} := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    ⊢ Iff (LT.lt (v.valuation ((algebraMap R K) r)) 1) (Dvd.dvd v.asIdeal (Ideal.s …
  -/
  rw [valuation_of_algebraMap]; exact v.intValuation_lt_one_iff_dvd r
                                /-
                                  🎉 no goals
                                -/


/-- There exists `π ∈ K` with `v`-adic valuation `Multiplicative.ofAdd (-1)`. -/
theorem valuation_exists_uniformizer : ∃ π : K, v.valuation π = Multiplicative.ofAdd (-1 : ℤ) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    ⊢ Exists fun π => Eq (v.valuation π) ↑(Multiplicative.ofAdd (-1))
  -/
  obtain ⟨r, hr⟩ := v.intValuation_exists_uniformizer
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    hr : Eq (v.intValuationDef r) ↑(Multiplicative.ofAdd (-1))
    ⊢ Exists fun π => Eq (v.valuation π) ↑(Multiplicative.ofAdd (-1))
  -/
  use algebraMap R K r
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    hr : Eq (v.intValuationDef r) ↑(Multiplicative.ofAdd (-1))
    ⊢ Eq (v.valuation ((algebraMap R K) r)) ↑(Multiplicative.ofAdd (-1))
  -/
  rw [valuation_def, Valuation.extendToLocalization_apply_map_apply]
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    hr : Eq (v.intValuationDef r) ↑(Multiplicative.ofAdd (-1))
    ⊢ Eq (v.intValuation r) ↑(Multiplicative.ofAdd (-1))
  -/
  exact hr
  /-
    🎉 no goals
  -/


/-- Uniformizers are nonzero. -/
theorem valuation_uniformizer_ne_zero : Classical.choose (v.valuation_exists_uniformizer K) ≠ 0 :=
  haveI hu := Classical.choose_spec (v.valuation_exists_uniformizer K)
  (Valuation.ne_zero_iff _).mp (ne_of_eq_of_ne hu WithZero.coe_ne_zero)


/-- `K` as a valued field with the `v`-adic valuation. -/
def adicValued : Valued K ℤₘ₀ :=
  Valued.mk' v.valuation


theorem adicValued_apply {x : K} : (v.adicValued.v : _) x = v.valuation x :=
  rfl


/-- The completion of `K` with respect to its `v`-adic valuation. -/
abbrev adicCompletion :=
  @UniformSpace.Completion K v.adicValued.toUniformSpace


instance : Field (v.adicCompletion K) :=
  @UniformSpace.Completion.instField K _ v.adicValued.toUniformSpace _ _
    v.adicValued.toUniformAddGroup


instance : Inhabited (v.adicCompletion K) :=
  ⟨0⟩


instance valuedAdicCompletion : Valued (v.adicCompletion K) ℤₘ₀ :=
  @Valued.valuedCompletion _ _ _ _ v.adicValued


theorem valuedAdicCompletion_def {x : v.adicCompletion K} :
    Valued.v x = @Valued.extension K _ _ _ (adicValued v) x :=
  rfl


instance adicCompletion_completeSpace : CompleteSpace (v.adicCompletion K) :=
  @UniformSpace.Completion.completeSpace K v.adicValued.toUniformSpace

-- Porting note: replaced by `Coe`
-- instance AdicCompletion.hasLiftT : HasLiftT K (v.adicCompletion K) :=
--   (inferInstance : HasLiftT K (@UniformSpace.Completion K v.adicValued.toUniformSpace))


instance adicCompletion.instCoe : Coe K (v.adicCompletion K) :=
  (inferInstance : Coe K (@UniformSpace.Completion K v.adicValued.toUniformSpace))


/-- The ring of integers of `adicCompletion`. -/
def adicCompletionIntegers : ValuationSubring (v.adicCompletion K) :=
  Valued.v.valuationSubring


instance : Inhabited (adicCompletionIntegers K v) :=
  ⟨0⟩


theorem mem_adicCompletionIntegers {x : v.adicCompletion K} :
    x ∈ v.adicCompletionIntegers K ↔ (Valued.v x : ℤₘ₀) ≤ 1 :=
  Iff.rfl


theorem not_mem_adicCompletionIntegers {x : v.adicCompletion K} :
    x ∉ v.adicCompletionIntegers K ↔ 1 < (Valued.v x : ℤₘ₀) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    x : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
    ⊢ Iff (Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionI …
  -/
  rw [not_congr <| mem_adicCompletionIntegers R K v]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    x : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
    ⊢ Iff (Not (LE.le (Valued.v x) 1)) (LT.lt 1 (Valued.v x))
  -/
  exact not_le
  /-
    🎉 no goals
  -/


instance (priority := 100) adicValued.has_uniform_continuous_const_smul' :
    @UniformContinuousConstSMul R K v.adicValued.toUniformSpace _ :=
  @uniformContinuousConstSMul_of_continuousConstSMul R K _ _ _ v.adicValued.toUniformSpace _ _


instance adicValued.uniformContinuousConstSMul :
    @UniformContinuousConstSMul K K v.adicValued.toUniformSpace _ :=
  @Ring.uniformContinuousConstSMul K _ v.adicValued.toUniformSpace _ _


instance adicCompletion.algebra' : Algebra R (v.adicCompletion K) :=
  @UniformSpace.Completion.algebra K _ v.adicValued.toUniformSpace _ _ R _ _
    (adicValued.has_uniform_continuous_const_smul' R K v)


theorem coe_smul_adicCompletion (r : R) (x : K) :
    (↑(r • x) : v.adicCompletion K) = r • (↑x : v.adicCompletion K) :=
  @UniformSpace.Completion.coe_smul R K v.adicValued.toUniformSpace _ _ r x


instance : Algebra K (v.adicCompletion K) :=
  @UniformSpace.Completion.algebra' K _ v.adicValued.toUniformSpace _ _


theorem algebraMap_adicCompletion' :
    ⇑(algebraMap R <| v.adicCompletion K) = (↑) ∘ algebraMap R K :=
  rfl


theorem algebraMap_adicCompletion :
    ⇑(algebraMap K <| v.adicCompletion K) = ((↑) : K → adicCompletion K v) :=
  rfl


instance : IsScalarTower R K (v.adicCompletion K) :=
  @UniformSpace.Completion.instIsScalarTower R K K v.adicValued.toUniformSpace _ _ _
    (adicValued.has_uniform_continuous_const_smul' R K v) _ _


instance : Algebra R (v.adicCompletionIntegers K) where
  smul r x :=
    ⟨r • (x : v.adicCompletion K), by
      have h :
        (algebraMap R (adicCompletion K v)) r = (algebraMap R K r : adicCompletion K v) := rfl
      /-
        R : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        r : R
        x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
        h : Eq ((algebraMap R (IsDedekindDomain.HeightOneSpectrum.adicCompletion K v)) …
        ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
      -/
      rw [Algebra.smul_def]
      /-
        R : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        r : R
        x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
        h : Eq ((algebraMap R (IsDedekindDomain.HeightOneSpectrum.adicCompletion K v)) …
        ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
      -/
      refine ValuationSubring.mul_mem _ _ _ ?_ x.2
      --Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance
      /-
        R : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        r : R
        x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
        h : Eq ((algebraMap R (IsDedekindDomain.HeightOneSpectrum.adicCompletion K v)) …
        ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
      -/
      letI : Valued K ℤₘ₀ := adicValued v
      /-
        R : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        r : R
        x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
        h : Eq ((algebraMap R (IsDedekindDomain.HeightOneSpectrum.adicCompletion K v)) …
        this : Valued K (WithZero (Multiplicative Int)) := v.adicValued
        ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
      -/
      rw [mem_adicCompletionIntegers, h, Valued.valuedCompletion_apply]
      /-
        R : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        r : R
        x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
        h : Eq ((algebraMap R (IsDedekindDomain.HeightOneSpectrum.adicCompletion K v)) …
        this : Valued K (WithZero (Multiplicative Int)) := v.adicValued
        ⊢ LE.le (Valued.v ((algebraMap R K) r)) 1
      -/
      exact v.valuation_le_one _⟩
      /-
        🎉 no goals
      -/
  toFun r :=
    ⟨(algebraMap R K r : adicCompletion K v), by
      simpa only [mem_adicCompletionIntegers, Valued.valuedCompletion_apply] using
        v.valuation_le_one _⟩
                 /-
                   R : Type u_1
                   inst✝⁴ : CommRing R
                   inst✝³ : IsDedekindDomain R
                   K : Type u_2
                   inst✝² : Field K
                   inst✝¹ : Algebra R K
                   inst✝ : IsFractionRing R K
                   v : IsDedekindDomain.HeightOneSpectrum R
                   ⊢ Eq ((fun r => ⟨↑K ((algebraMap R K) r), ⋯⟩) 1) 1
                 -/
  map_one' := by simp only [map_one]; rfl
                                      /-
                                        🎉 no goals
                                      -/
  map_mul' x y := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x y : R
      ⊢ Eq ({ toFun := fun r => ⟨↑K ((algebraMap R K) r), ⋯⟩, map_one' := ⋯ }.toFun  …
    -/
    ext
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x y : R
      ⊢ Eq ↑({ toFun := fun r => ⟨↑K ((algebraMap R K) r), ⋯⟩, map_one' := ⋯ }.toFun …
    -/
    simp only [map_mul, UniformSpace.Completion.coe_mul, MulMemClass.mk_mul_mk]
    /-
      🎉 no goals
    -/
                  /-
                    R : Type u_1
                    inst✝⁴ : CommRing R
                    inst✝³ : IsDedekindDomain R
                    K : Type u_2
                    inst✝² : Field K
                    inst✝¹ : Algebra R K
                    inst✝ : IsFractionRing R K
                    v : IsDedekindDomain.HeightOneSpectrum R
                    ⊢ Eq ((↑{ toFun := fun r => ⟨↑K ((algebraMap R K) r), ⋯⟩, map_one' := ⋯, map_m …
                  -/
  map_zero' := by simp only [map_zero]; rfl
                                        /-
                                          🎉 no goals
                                        -/
  map_add' x y := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x y : R
      ⊢ Eq ((↑{ toFun := fun r => ⟨↑K ((algebraMap R K) r), ⋯⟩, map_one' := ⋯, map_m …
    -/
    ext
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x y : R
      ⊢ Eq ↑((↑{ toFun := fun r => ⟨↑K ((algebraMap R K) r), ⋯⟩, map_one' := ⋯, map_ …
    -/
    simp only [map_add, UniformSpace.Completion.coe_add, AddMemClass.mk_add_mk]
    /-
      🎉 no goals
    -/
  commutes' r x := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      r : R
      x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
      ⊢ Eq (HMul.hMul ({ toFun := fun r => ⟨↑K ((algebraMap R K) r), ⋯⟩, map_one' := …
    -/
    rw [mul_comm]
    /-
      🎉 no goals
    -/
  smul_def' r x := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      r : R
      x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
      ⊢ Eq (HSMul.hSMul r x) (HMul.hMul ({ toFun := fun r => ⟨↑K ((algebraMap R K) r …
    -/
    ext
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      r : R
      x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
      ⊢ Eq ↑(HSMul.hSMul r x) ↑(HMul.hMul ({ toFun := fun r => ⟨↑K ((algebraMap R K) …
    -/
    simp only [Subring.coe_mul, Algebra.smul_def]
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      r : R
      x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
      ⊢ Eq ↑(HSMul.hSMul r x) ↑(HMul.hMul ({ toFun := fun r => ⟨↑K ((algebraMap R K) …
    -/
    rfl
    /-
      🎉 no goals
    -/


variable {R K} in
open scoped algebraMap in -- to make the coercions from `R` fire
/-- The valuation on the completion agrees with the global valuation on elements of the
integer ring. -/
theorem valuedAdicCompletion_eq_valuation (r : R) :
    Valued.v (r : v.adicCompletion K) = v.valuation (r : K) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    ⊢ Eq (Valued.v ↑r) (v.valuation ↑r)
  -/
  convert Valued.valuedCompletion_apply (r : K)
  /-
    🎉 no goals
  -/


variable {R K} in
/-- The valuation on the completion agrees with the global valuation on elements of the field. -/
theorem valuedAdicCompletion_eq_valuation' (k : K) :
    Valued.v (k : v.adicCompletion K) = v.valuation k := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    k : K
    ⊢ Eq (Valued.v (↑K k)) (v.valuation k)
  -/
  convert Valued.valuedCompletion_apply k
  /-
    🎉 no goals
  -/


variable {R K} in
open scoped algebraMap in -- to make the coercion from `R` fire
/-- A global integer is in the local integers. -/
lemma coe_mem_adicCompletionIntegers (r : R) :
    (r : adicCompletion K v) ∈ adicCompletionIntegers K v := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
  -/
  rw [mem_adicCompletionIntegers, valuedAdicCompletion_eq_valuation, valuation_eq_intValuationDef]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    r : R
    ⊢ LE.le (v.intValuationDef r) 1
  -/
  exact intValuation_le_one v r
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_smul_adicCompletionIntegers (r : R) (x : v.adicCompletionIntegers K) :
    (↑(r • x) : v.adicCompletion K) = r • (x : v.adicCompletion K) :=
  rfl


instance : NoZeroSMulDivisors R (v.adicCompletionIntegers K) where
  eq_zero_or_eq_zero_of_smul_eq_zero {c x} hcx := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      c : R
      x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
      hcx : Eq (HSMul.hSMul c x) 0
      ⊢ Or (Eq c 0) (Eq x 0)
    -/
    rw [Algebra.smul_def, mul_eq_zero] at hcx
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      c : R
      x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
      hcx : Or (Eq ((algebraMap R (Subtype fun x => Membership.mem (IsDedekindDomain …
      ⊢ Or (Eq c 0) (Eq x 0)
    -/
    refine hcx.imp_left fun hc => ?_
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      c : R
      x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
      hcx : Or (Eq ((algebraMap R (Subtype fun x => Membership.mem (IsDedekindDomain …
      hc : Eq ((algebraMap R (Subtype fun x => Membership.mem (IsDedekindDomain.Heig …
      ⊢ Eq c 0
    -/
    letI : UniformSpace K := v.adicValued.toUniformSpace
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      c : R
      x : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
      hcx : Or (Eq ((algebraMap R (Subtype fun x => Membership.mem (IsDedekindDomain …
      hc : Eq ((algebraMap R (Subtype fun x => Membership.mem (IsDedekindDomain.Heig …
      this : UniformSpace K := Valued.toUniformSpace
      ⊢ Eq c 0
    -/
    rw [← map_zero (algebraMap R (v.adicCompletionIntegers K))] at hc
    exact
      IsFractionRing.injective R K (UniformSpace.Completion.coe_injective K (Subtype.ext_iff.mp hc))


instance adicCompletion.instIsScalarTower' :
    IsScalarTower R (v.adicCompletionIntegers K) (v.adicCompletion K) where
                         /-
                           R : Type u_1
                           inst✝⁴ : CommRing R
                           inst✝³ : IsDedekindDomain R
                           K : Type u_2
                           inst✝² : Field K
                           inst✝¹ : Algebra R K
                           inst✝ : IsFractionRing R K
                           v : IsDedekindDomain.HeightOneSpectrum R
                           x : R
                           y : Subtype fun x => Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCo …
                           z : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
                         -/
  smul_assoc x y z := by simp only [Algebra.smul_def]; apply mul_assoc
                                                       /-
                                                         🎉 no goals
                                                       -/


open nonZeroDivisors algebraMap in
variable {R K} in
lemma adicCompletion.mul_nonZeroDivisor_mem_adicCompletionIntegers (v : HeightOneSpectrum R)
    (a : v.adicCompletion K) : ∃ b ∈ R⁰, a * b ∈ v.adicCompletionIntegers K := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    a : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
    ⊢ Exists fun b => And (Membership.mem (nonZeroDivisors R) b) (Membership.mem ( …
  -/
  by_cases ha : a ∈ v.adicCompletionIntegers K
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
      ha : Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers …
      ⊢ Exists fun b => And (Membership.mem (nonZeroDivisors R) b) (Membership.mem ( …
    -/
  · use 1
    /-
      case h
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
      ha : Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers …
      ⊢ And (Membership.mem (nonZeroDivisors R) 1) (Membership.mem (IsDedekindDomain …
    -/
    simp [ha, Submonoid.one_mem]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
      ha : Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionInt …
      ⊢ Exists fun b => And (Membership.mem (nonZeroDivisors R) b) (Membership.mem ( …
    -/
  · rw [not_mem_adicCompletionIntegers] at ha
    -- Let the additive valuation of a be -d with d>0
    obtain ⟨d, hd⟩ : ∃ d : ℤ, Valued.v a = ofAdd d :=
      Option.ne_none_iff_exists'.mp <| (lt_trans zero_lt_one ha).ne'
    /-
      case neg.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
      ha : LT.lt 1 (Valued.v a)
      d : Int
      hd : Eq (Valued.v a) ↑(Multiplicative.ofAdd d)
      ⊢ Exists fun b => And (Membership.mem (nonZeroDivisors R) b) (Membership.mem ( …
    -/
    rw [hd, WithZero.one_lt_coe, ← ofAdd_zero, ofAdd_lt] at ha
    -- let ϖ be a uniformiser
    /-
      case neg.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
      d : Int
      ha : LT.lt 0 d
      hd : Eq (Valued.v a) ↑(Multiplicative.ofAdd d)
      ⊢ Exists fun b => And (Membership.mem (nonZeroDivisors R) b) (Membership.mem ( …
    -/
    obtain ⟨ϖ, hϖ⟩ := intValuation_exists_uniformizer v
    /-
      case neg.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
      d : Int
      ha : LT.lt 0 d
      hd : Eq (Valued.v a) ↑(Multiplicative.ofAdd d)
      ϖ : R
      hϖ : Eq (v.intValuationDef ϖ) ↑(Multiplicative.ofAdd (-1))
      ⊢ Exists fun b => And (Membership.mem (nonZeroDivisors R) b) (Membership.mem ( …
    -/
    have hϖ0 : ϖ ≠ 0 := by rintro rfl; simp at hϖ
    -- use ϖ^d
    /-
      case neg.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
      d : Int
      ha : LT.lt 0 d
      hd : Eq (Valued.v a) ↑(Multiplicative.ofAdd d)
      ϖ : R
      hϖ : Eq (v.intValuationDef ϖ) ↑(Multiplicative.ofAdd (-1))
      hϖ0 : Ne ϖ 0
      ⊢ Exists fun b => And (Membership.mem (nonZeroDivisors R) b) (Membership.mem ( …
    -/
    refine ⟨ϖ^d.natAbs, pow_mem (mem_nonZeroDivisors_of_ne_zero hϖ0) _, ?_⟩
    -- now manually translate the goal (an inequality in ℤₘ₀) to an inequality in ℤ
    rw [mem_adicCompletionIntegers, algebraMap.coe_pow, map_mul, hd, map_pow,
      valuedAdicCompletion_eq_valuation, valuation_eq_intValuationDef, hϖ, ← WithZero.coe_pow,
      ← WithZero.coe_mul, WithZero.coe_le_one, ← toAdd_le, toAdd_mul, toAdd_ofAdd, toAdd_pow,
      toAdd_ofAdd, toAdd_one,
      show d.natAbs • (-1) = (d.natAbs : ℤ) • (-1) by simp only [nsmul_eq_mul,
        Int.natCast_natAbs, smul_eq_mul],
      ← Int.eq_natAbs_of_zero_le ha.le, smul_eq_mul]
    -- and now it's easy
    /-
      case neg.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion K v
      d : Int
      ha : LT.lt 0 d
      hd : Eq (Valued.v a) ↑(Multiplicative.ofAdd d)
      ϖ : R
      hϖ : Eq (v.intValuationDef ϖ) ↑(Multiplicative.ofAdd (-1))
      hϖ0 : Ne ϖ 0
      ⊢ LE.le (HAdd.hAdd d (HMul.hMul d (-1))) 0
    -/
    omega
    /-
      🎉 no goals
    -/


