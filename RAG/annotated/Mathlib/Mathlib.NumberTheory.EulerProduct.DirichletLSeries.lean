/-- When `s ≠ 0`, the map `n ↦ n^(-s)` is completely multiplicative and vanishes at zero. -/
noncomputable
def riemannZetaSummandHom (hs : s ≠ 0) : ℕ →*₀ ℂ where
  toFun n := (n : ℂ) ^ (-s)
                  /-
                    s : Complex
                    hs : Ne s 0
                    ⊢ Eq ((fun n => HPow.hPow (↑n) (Neg.neg s)) 0) 0
                  -/
  map_zero' := by simp [hs]
                  /-
                    🎉 no goals
                  -/
                 /-
                   s : Complex
                   hs : Ne s 0
                   ⊢ Eq ({ toFun := fun n => HPow.hPow (↑n) (Neg.neg s), map_zero' := ⋯ }.toFun 1 …
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
  map_mul' m n := by
    simpa only [Nat.cast_mul, ofReal_natCast]
      using mul_cpow_ofReal_nonneg m.cast_nonneg n.cast_nonneg _


/-- When `χ` is a Dirichlet character and `s ≠ 0`, the map `n ↦ χ n * n^(-s)` is completely
multiplicative and vanishes at zero. -/
noncomputable
def dirichletSummandHom {n : ℕ} (χ : DirichletCharacter ℂ n) (hs : s ≠ 0) : ℕ →*₀ ℂ where
  toFun n := χ n * (n : ℂ) ^ (-s)
                  /-
                    s : Complex
                    n : Nat
                    χ : DirichletCharacter Complex n
                    hs : Ne s 0
                    ⊢ Eq ((fun n_1 => HMul.hMul (χ ↑n_1) (HPow.hPow (↑n_1) (Neg.neg s))) 0) 0
                  -/
  map_zero' := by simp [hs]
                  /-
                    🎉 no goals
                  -/
                 /-
                   s : Complex
                   n : Nat
                   χ : DirichletCharacter Complex n
                   hs : Ne s 0
                   ⊢ Eq ({ toFun := fun n_1 => HMul.hMul (χ ↑n_1) (HPow.hPow (↑n_1) (Neg.neg s)), …
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
  map_mul' m n := by
    /-
      s : Complex
      n✝ : Nat
      χ : DirichletCharacter Complex n✝
      hs : Ne s 0
      m n : Nat
      ⊢ Eq ({ toFun := fun n => HMul.hMul (χ ↑n) (HPow.hPow (↑n) (Neg.neg s)), map_z …
    -/
    simp_rw [← ofReal_natCast]
    simpa only [Nat.cast_mul, IsUnit.mul_iff, not_and, map_mul, ofReal_mul,
      mul_cpow_ofReal_nonneg m.cast_nonneg n.cast_nonneg _]
      using mul_mul_mul_comm ..


/-- When `s.re > 1`, the map `n ↦ n^(-s)` is norm-summable. -/
lemma summable_riemannZetaSummand (hs : 1 < s.re) :
    Summable (fun n ↦ ‖riemannZetaSummandHom (ne_zero_of_one_lt_re hs) n‖) := by
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Summable fun n => Norm.norm ((riemannZetaSummandHom ⋯) n)
  -/
  simp only [riemannZetaSummandHom, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk]
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Summable fun n => Norm.norm (HPow.hPow (↑n) (Neg.neg s))
  -/
  convert Real.summable_nat_rpow_inv.mpr hs with n
  rw [← ofReal_natCast, Complex.norm_eq_abs,
    abs_cpow_eq_rpow_re_of_nonneg (Nat.cast_nonneg n) <| re_neg_ne_zero_of_one_lt_re hs,
    neg_re, Real.rpow_neg <| Nat.cast_nonneg n]


lemma tsum_riemannZetaSummand (hs : 1 < s.re) :
    ∑' (n : ℕ), riemannZetaSummandHom (ne_zero_of_one_lt_re hs) n = riemannZeta s := by
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (tsum fun n => (riemannZetaSummandHom ⋯) n) (riemannZeta s)
  -/
  have hsum := summable_riemannZetaSummand hs
  /-
    s : Complex
    hs : LT.lt 1 s.re
    hsum : Summable fun n => Norm.norm ((riemannZetaSummandHom ⋯) n)
    ⊢ Eq (tsum fun n => (riemannZetaSummandHom ⋯) n) (riemannZeta s)
  -/
  rw [zeta_eq_tsum_one_div_nat_add_one_cpow hs, tsum_eq_zero_add hsum.of_norm, map_zero, zero_add]
  simp only [riemannZetaSummandHom, cpow_neg, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk,
    Nat.cast_add, Nat.cast_one, one_div]


/-- When `s.re > 1`, the map `n ↦ χ(n) * n^(-s)` is norm-summable. -/
lemma summable_dirichletSummand {N : ℕ} (χ : DirichletCharacter ℂ N) (hs : 1 < s.re) :
    Summable (fun n ↦ ‖dirichletSummandHom χ (ne_zero_of_one_lt_re hs) n‖) := by
  /-
    s : Complex
    N : Nat
    χ : DirichletCharacter Complex N
    hs : LT.lt 1 s.re
    ⊢ Summable fun n => Norm.norm ((dirichletSummandHom χ ⋯) n)
  -/
  simp only [dirichletSummandHom, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk, norm_mul]
  exact (summable_riemannZetaSummand hs).of_nonneg_of_le (fun _ ↦ by positivity)
    (fun n ↦ mul_le_of_le_one_left (norm_nonneg _) <| χ.norm_le_one n)


open scoped LSeries.notation in
lemma tsum_dirichletSummand {N : ℕ} (χ : DirichletCharacter ℂ N) (hs : 1 < s.re) :
    ∑' (n : ℕ), dirichletSummandHom χ (ne_zero_of_one_lt_re hs) n = L ↗χ s := by
  simp only [dirichletSummandHom, cpow_neg, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk, LSeries,
    LSeries.term_of_ne_zero' (ne_zero_of_one_lt_re hs), div_eq_mul_inv]


/-- The Euler product for the Riemann ζ function, valid for `s.re > 1`.
This version is stated in terms of `HasProd`. -/
theorem riemannZeta_eulerProduct_hasProd (hs : 1 < s.re) :
    HasProd (fun p : Primes ↦ (1 - (p : ℂ) ^ (-s))⁻¹) (riemannZeta s) := by
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasProd (fun p => Inv.inv (HSub.hSub 1 (HPow.hPow (↑↑p) (Neg.neg s)))) (riem …
  -/
  rw [← tsum_riemannZetaSummand hs]
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasProd (fun p => Inv.inv (HSub.hSub 1 (HPow.hPow (↑↑p) (Neg.neg s)))) (tsum …
  -/
  apply eulerProduct_completely_multiplicative_hasProd <| summable_riemannZetaSummand hs
  /-
    🎉 no goals
  -/


/-- The Euler product for the Riemann ζ function, valid for `s.re > 1`.
This version is stated in terms of `tprod`. -/
theorem riemannZeta_eulerProduct_tprod (hs : 1 < s.re) :
    ∏' p : Primes, (1 - (p : ℂ) ^ (-s))⁻¹ = riemannZeta s :=
  (riemannZeta_eulerProduct_hasProd hs).tprod_eq


/-- The Euler product for the Riemann ζ function, valid for `s.re > 1`.
This version is stated in the form of convergence of finite partial products. -/
theorem riemannZeta_eulerProduct (hs : 1 < s.re) :
    Tendsto (fun n : ℕ ↦ ∏ p ∈ primesBelow n, (1 - (p : ℂ) ^ (-s))⁻¹) atTop
      (𝓝 (riemannZeta s)) := by
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Filter.Tendsto (fun n => n.primesBelow.prod fun p => Inv.inv (HSub.hSub 1 (H …
  -/
  rw [← tsum_riemannZetaSummand hs]
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Filter.Tendsto (fun n => n.primesBelow.prod fun p => Inv.inv (HSub.hSub 1 (H …
  -/
  apply eulerProduct_completely_multiplicative <| summable_riemannZetaSummand hs
  /-
    🎉 no goals
  -/


/-- The Euler product for Dirichlet L-series, valid for `s.re > 1`.
This version is stated in terms of `HasProd`. -/
theorem DirichletCharacter.LSeries_eulerProduct_hasProd {N : ℕ} (χ : DirichletCharacter ℂ N)
    (hs : 1 < s.re) :
    HasProd (fun p : Primes ↦ (1 - χ p * (p : ℂ) ^ (-s))⁻¹) (L ↗χ s) := by
  /-
    s : Complex
    N : Nat
    χ : DirichletCharacter Complex N
    hs : LT.lt 1 s.re
    ⊢ HasProd (fun p => Inv.inv (HSub.hSub 1 (HMul.hMul (χ ↑↑p) (HPow.hPow (↑↑p) ( …
  -/
  rw [← tsum_dirichletSummand χ hs]
  /-
    s : Complex
    N : Nat
    χ : DirichletCharacter Complex N
    hs : LT.lt 1 s.re
    ⊢ HasProd (fun p => Inv.inv (HSub.hSub 1 (HMul.hMul (χ ↑↑p) (HPow.hPow (↑↑p) ( …
  -/
  convert eulerProduct_completely_multiplicative_hasProd <| summable_dirichletSummand χ hs
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-14")] alias
  dirichletLSeries_eulerProduct_hasProd := DirichletCharacter.LSeries_eulerProduct_hasProd


/-- The Euler product for Dirichlet L-series, valid for `s.re > 1`.
This version is stated in terms of `tprod`. -/
theorem DirichletCharacter.LSeries_eulerProduct_tprod {N : ℕ} (χ : DirichletCharacter ℂ N)
    (hs : 1 < s.re) :
    ∏' p : Primes, (1 - χ p * (p : ℂ) ^ (-s))⁻¹ = L ↗χ s :=
  (DirichletCharacter.LSeries_eulerProduct_hasProd χ hs).tprod_eq


@[deprecated (since := "2024-11-14")] alias
  dirichlet_LSeries_eulerProduct_tprod := DirichletCharacter.LSeries_eulerProduct_tprod


/-- The Euler product for Dirichlet L-series, valid for `s.re > 1`.
This version is stated in the form of convergence of finite partial products. -/
theorem DirichletCharacter.LSeries_eulerProduct {N : ℕ} (χ : DirichletCharacter ℂ N)
    (hs : 1 < s.re) :
    Tendsto (fun n : ℕ ↦ ∏ p ∈ primesBelow n, (1 - χ p * (p : ℂ) ^ (-s))⁻¹) atTop
      (𝓝 (L ↗χ s)) := by
  /-
    s : Complex
    N : Nat
    χ : DirichletCharacter Complex N
    hs : LT.lt 1 s.re
    ⊢ Filter.Tendsto (fun n => n.primesBelow.prod fun p => Inv.inv (HSub.hSub 1 (H …
  -/
  rw [← tsum_dirichletSummand χ hs]
  /-
    s : Complex
    N : Nat
    χ : DirichletCharacter Complex N
    hs : LT.lt 1 s.re
    ⊢ Filter.Tendsto (fun n => n.primesBelow.prod fun p => Inv.inv (HSub.hSub 1 (H …
  -/
  apply eulerProduct_completely_multiplicative <| summable_dirichletSummand χ hs
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-14")] alias
  dirichletLSeries_eulerProduct := DirichletCharacter.LSeries_eulerProduct


/-- A variant of the Euler product for Dirichlet L-series. -/
theorem DirichletCharacter.LSeries_eulerProduct_exp_log {N : ℕ} (χ : DirichletCharacter ℂ N)
    {s : ℂ} (hs : 1 < s.re) :
    exp (∑' p : Nat.Primes, -log (1 - χ p * p ^ (-s))) = L ↗χ s := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (Complex.exp (tsum fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul  …
  -/
  let f := dirichletSummandHom χ <| ne_zero_of_one_lt_re hs
  have h n : term ↗χ s n = f n := by
    rcases eq_or_ne n 0 with rfl | hn
    · simp only [term_zero, map_zero]
    · simp only [ne_eq, hn, not_false_eq_true, term_of_ne_zero, div_eq_mul_inv,
        dirichletSummandHom, cpow_neg, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk, f]
  simpa only [LSeries, h]
    using exp_tsum_primes_log_eq_tsum (f := f) <| summable_dirichletSummand χ hs


/-- A variant of the Euler product for the L-series of `ζ`. -/
theorem ArithmeticFunction.LSeries_zeta_eulerProduct_exp_log {s : ℂ} (hs : 1 < s.re) :
    exp (∑' p : Nat.Primes, -Complex.log (1 - p ^ (-s))) = L 1 s := by
  convert modOne_eq_one (R := ℂ) ▸
    DirichletCharacter.LSeries_eulerProduct_exp_log (1 : DirichletCharacter ℂ 1) hs using 7
  /-
    case h.e'_2.h.e'_1.h.e'_5.h.h.e'_3.h.e'_1.h.e'_6
    s : Complex
    hs : LT.lt 1 s.re
    x✝ : Nat.Primes
    ⊢ Eq (HPow.hPow (↑↑x✝) (Neg.neg s)) (HMul.hMul (1 ↑↑x✝) (HPow.hPow (↑↑x✝) (Neg …
  -/
  rw [MulChar.one_apply <| isUnit_of_subsingleton _, one_mul]
  /-
    🎉 no goals
  -/


/-- A variant of the Euler product for the Riemann zeta function. -/
theorem riemannZeta_eulerProduct_exp_log {s : ℂ} (hs : 1 < s.re) :
    exp (∑' p : Nat.Primes, -Complex.log (1 - p ^ (-s))) = riemannZeta s :=
  LSeries_one_eq_riemannZeta hs ▸ ArithmeticFunction.LSeries_zeta_eulerProduct_exp_log hs


/-- If `χ` is a Dirichlet character and its level `M` divides `N`, then we obtain the L-series
of `χ` considered as a Dirichlet character of level `N` from the L-series of `χ` by multiplying
with `∏ p ∈ N.primeFactors, (1 - χ p * p ^ (-s))`. -/
lemma DirichletCharacter.LSeries_changeLevel {M N : ℕ} [NeZero N]
    (hMN : M ∣ N) (χ : DirichletCharacter ℂ M) {s : ℂ} (hs : 1 < s.re) :
    LSeries ↗(changeLevel hMN χ) s =
      LSeries ↗χ s * ∏ p ∈ N.primeFactors, (1 - χ p * p ^ (-s)) := by
  rw [prod_eq_tprod_mulIndicator, ← DirichletCharacter.LSeries_eulerProduct_tprod _ hs,
    ← DirichletCharacter.LSeries_eulerProduct_tprod _ hs]
  -- convert to a form suitable for `tprod_subtype`
  /-
    M N : Nat
    inst✝ : NeZero N
    hMN : Dvd.dvd M N
    χ : DirichletCharacter Complex M
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (tprod fun p => Inv.inv (HSub.hSub 1 (HMul.hMul (((DirichletCharacter.cha …
  -/
  have (f : Primes → ℂ) : ∏' (p : Primes), f p = ∏' (p : ↑{p : ℕ | p.Prime}), f p := rfl
  rw [this, tprod_subtype _ fun p : ℕ ↦ (1 - (changeLevel hMN χ) p * p ^ (-s))⁻¹,
    this, tprod_subtype _ fun p : ℕ ↦ (1 - χ p * p ^ (-s))⁻¹, ← tprod_mul]
  /-
    M N : Nat
    inst✝ : NeZero N
    hMN : Dvd.dvd M N
    χ : DirichletCharacter Complex M
    s : Complex
    hs : LT.lt 1 s.re
    this : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p => f …
    ⊢ Eq (tprod fun x => (setOf fun p => Nat.Prime p).mulIndicator (fun p => Inv.i …
  -/
  rotate_left -- deal with convergence goals first
  · exact multipliable_subtype_iff_mulIndicator.mp
      (DirichletCharacter.LSeries_eulerProduct_hasProd χ hs).multipliable
    /-
      case hg
      M N : Nat
      inst✝ : NeZero N
      hMN : Dvd.dvd M N
      χ : DirichletCharacter Complex M
      s : Complex
      hs : LT.lt 1 s.re
      this : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p => f …
      ⊢ Multipliable ((↑N.primeFactors).mulIndicator fun p => HSub.hSub 1 (HMul.hMul …
    -/
  · exact multipliable_subtype_iff_mulIndicator.mp Multipliable.of_finite
    /-
      🎉 no goals
    -/
    /-
      M N : Nat
      inst✝ : NeZero N
      hMN : Dvd.dvd M N
      χ : DirichletCharacter Complex M
      s : Complex
      hs : LT.lt 1 s.re
      this : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p => f …
      ⊢ Eq (tprod fun x => (setOf fun p => Nat.Prime p).mulIndicator (fun p => Inv.i …
    -/
  · congr 1 with p
    simp only [Set.mulIndicator_apply, Set.mem_setOf_eq, Finset.mem_coe, Nat.mem_primeFactors,
      ne_eq, mul_ite, ite_mul, one_mul, mul_one]
    /-
      case e_f.h
      M N : Nat
      inst✝ : NeZero N
      hMN : Dvd.dvd M N
      χ : DirichletCharacter Complex M
      s : Complex
      hs : LT.lt 1 s.re
      this : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p => f …
      p : Nat
      ⊢ Eq (ite (Nat.Prime p) (Inv.inv (HSub.hSub 1 (HMul.hMul (((DirichletCharacter …
    -/
    by_cases h : p.Prime; swap
      /-
        case neg
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p => f …
        p : Nat
        h : Not (Nat.Prime p)
        ⊢ Eq (ite (Nat.Prime p) (Inv.inv (HSub.hSub 1 (HMul.hMul (((DirichletCharacter …
      -/
    · simp only [h, false_and, if_false]
      /-
        🎉 no goals
      -/
    /-
      case pos
      M N : Nat
      inst✝ : NeZero N
      hMN : Dvd.dvd M N
      χ : DirichletCharacter Complex M
      s : Complex
      hs : LT.lt 1 s.re
      this : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p => f …
      p : Nat
      h : Nat.Prime p
      ⊢ Eq (ite (Nat.Prime p) (Inv.inv (HSub.hSub 1 (HMul.hMul (((DirichletCharacter …
    -/
    simp only [h, true_and, if_true]
    /-
      case pos
      M N : Nat
      inst✝ : NeZero N
      hMN : Dvd.dvd M N
      χ : DirichletCharacter Complex M
      s : Complex
      hs : LT.lt 1 s.re
      this : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p => f …
      p : Nat
      h : Nat.Prime p
      ⊢ Eq (Inv.inv (HSub.hSub 1 (HMul.hMul (((DirichletCharacter.changeLevel hMN) χ …
    -/
    by_cases hp' : p ∣ N; swap
    · simp only [hp', false_and, ↓reduceIte, inv_inj, sub_right_inj, mul_eq_mul_right_iff,
        cpow_eq_zero_iff, Nat.cast_eq_zero, h.ne_zero, ne_eq, neg_eq_zero, or_false]
      /-
        case neg
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p => f …
        p : Nat
        h : Nat.Prime p
        hp' : Not (Dvd.dvd p N)
        ⊢ Eq (((DirichletCharacter.changeLevel hMN) χ) ↑p) (χ ↑p)
      -/
      have hq : IsUnit (p : ZMod N) := (ZMod.isUnit_prime_iff_not_dvd h).mpr hp'
      simp only [hq.unit_spec ▸ DirichletCharacter.changeLevel_eq_cast_of_dvd χ hMN hq.unit,
        ZMod.cast_natCast hMN]
      /-
        case pos
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p => f …
        p : Nat
        h : Nat.Prime p
        hp' : Dvd.dvd p N
        ⊢ Eq (Inv.inv (HSub.hSub 1 (HMul.hMul (((DirichletCharacter.changeLevel hMN) χ …
      -/
    · simp only [hp', NeZero.ne N, not_false_eq_true, and_self, ↓reduceIte]
      /-
        case pos
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p => f …
        p : Nat
        h : Nat.Prime p
        hp' : Dvd.dvd p N
        ⊢ Eq (Inv.inv (HSub.hSub 1 (HMul.hMul (((DirichletCharacter.changeLevel hMN) χ …
      -/
      have : ¬IsUnit (p : ZMod N) := by rwa [ZMod.isUnit_prime_iff_not_dvd h, not_not]
      /-
        case pos
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this✝ : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p =>  …
        p : Nat
        h : Nat.Prime p
        hp' : Dvd.dvd p N
        this : Not (IsUnit ↑p)
        ⊢ Eq (Inv.inv (HSub.hSub 1 (HMul.hMul (((DirichletCharacter.changeLevel hMN) χ …
      -/
      rw [MulChar.map_nonunit _ this, zero_mul, sub_zero, inv_one]
      /-
        case pos
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this✝ : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p =>  …
        p : Nat
        h : Nat.Prime p
        hp' : Dvd.dvd p N
        this : Not (IsUnit ↑p)
        ⊢ Eq 1 (HMul.hMul (Inv.inv (HSub.hSub 1 (HMul.hMul (χ ↑p) (HPow.hPow (↑p) (Neg …
      -/
      refine (inv_mul_cancel₀ ?_).symm
      /-
        case pos
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this✝ : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p =>  …
        p : Nat
        h : Nat.Prime p
        hp' : Dvd.dvd p N
        this : Not (IsUnit ↑p)
        ⊢ Ne (HSub.hSub 1 (HMul.hMul (χ ↑p) (HPow.hPow (↑p) (Neg.neg s)))) 0
      -/
      rw [sub_ne_zero, ne_comm]
      -- Remains to show `χ p * p ^ (-s) ≠ 1`. We show its norm is strictly `< 1`.
      /-
        case pos
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this✝ : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p =>  …
        p : Nat
        h : Nat.Prime p
        hp' : Dvd.dvd p N
        this : Not (IsUnit ↑p)
        ⊢ Ne (HMul.hMul (χ ↑p) (HPow.hPow (↑p) (Neg.neg s))) 1
      -/
      apply_fun (‖·‖)
      /-
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this✝ : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p =>  …
        p : Nat
        h : Nat.Prime p
        hp' : Dvd.dvd p N
        this : Not (IsUnit ↑p)
        ⊢ Ne ((fun x => Norm.norm x) (HMul.hMul (χ ↑p) (HPow.hPow (↑p) (Neg.neg s))))  …
      -/
      simp only [norm_mul, norm_one]
      /-
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this✝ : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p =>  …
        p : Nat
        h : Nat.Prime p
        hp' : Dvd.dvd p N
        this : Not (IsUnit ↑p)
        ⊢ Ne (HMul.hMul (Norm.norm (χ ↑p)) (Norm.norm (HPow.hPow (↑p) (Neg.neg s)))) 1
      -/
      have ha : ‖χ p‖ ≤ 1 := χ.norm_le_one p
      /-
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this✝ : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p =>  …
        p : Nat
        h : Nat.Prime p
        hp' : Dvd.dvd p N
        this : Not (IsUnit ↑p)
        ha : LE.le (Norm.norm (χ ↑p)) 1
        ⊢ Ne (HMul.hMul (Norm.norm (χ ↑p)) (Norm.norm (HPow.hPow (↑p) (Neg.neg s)))) 1
      -/
      have hb : ‖(p : ℂ) ^ (-s)‖ ≤ 1 / 2 := norm_prime_cpow_le_one_half ⟨p, h⟩ hs
      /-
        M N : Nat
        inst✝ : NeZero N
        hMN : Dvd.dvd M N
        χ : DirichletCharacter Complex M
        s : Complex
        hs : LT.lt 1 s.re
        this✝ : ∀ (f : Nat.Primes → Complex), Eq (tprod fun p => f p) (tprod fun p =>  …
        p : Nat
        h : Nat.Prime p
        hp' : Dvd.dvd p N
        this : Not (IsUnit ↑p)
        ha : LE.le (Norm.norm (χ ↑p)) 1
        hb : LE.le (Norm.norm (HPow.hPow (↑p) (Neg.neg s))) (1 / 2)
        ⊢ Ne (HMul.hMul (Norm.norm (χ ↑p)) (Norm.norm (HPow.hPow (↑p) (Neg.neg s)))) 1
      -/
      exact ((mul_le_mul ha hb (norm_nonneg _) zero_le_one).trans_lt (by norm_num)).ne
      /-
        🎉 no goals
      -/

