/-- The completed Riemann zeta function with its poles removed, `Λ(s) + 1 / s - 1 / (s - 1)`. -/
def completedRiemannZeta₀ (s : ℂ) : ℂ := completedHurwitzZetaEven₀ 0 s


/-- The completed Riemann zeta function, `Λ(s)`, which satisfies
`Λ(s) = π ^ (-s / 2) Γ(s / 2) ζ(s)` (up to a minor correction at `s = 0`). -/
def completedRiemannZeta (s : ℂ) : ℂ := completedHurwitzZetaEven 0 s


lemma HurwitzZeta.completedHurwitzZetaEven_zero (s : ℂ) :
    completedHurwitzZetaEven 0 s = completedRiemannZeta s := rfl


lemma HurwitzZeta.completedHurwitzZetaEven₀_zero (s : ℂ) :
    completedHurwitzZetaEven₀ 0 s = completedRiemannZeta₀ s := rfl


lemma HurwitzZeta.completedCosZeta_zero (s : ℂ) :
    completedCosZeta 0 s = completedRiemannZeta s := by
  /-
    s : Complex
    ⊢ Eq (HurwitzZeta.completedCosZeta 0 s) (completedRiemannZeta s)
  -/
  rw [completedRiemannZeta, completedHurwitzZetaEven, completedCosZeta, hurwitzEvenFEPair_zero_symm]
  /-
    🎉 no goals
  -/


lemma HurwitzZeta.completedCosZeta₀_zero (s : ℂ) :
    completedCosZeta₀ 0 s = completedRiemannZeta₀ s := by
  rw [completedRiemannZeta₀, completedHurwitzZetaEven₀, completedCosZeta₀,
    hurwitzEvenFEPair_zero_symm]


lemma completedRiemannZeta_eq (s : ℂ) :
    completedRiemannZeta s = completedRiemannZeta₀ s - 1 / s - 1 / (1 - s) := by
  /-
    s : Complex
    ⊢ Eq (completedRiemannZeta s) (HSub.hSub (HSub.hSub (completedRiemannZeta₀ s)  …
  -/
  simp_rw [completedRiemannZeta, completedRiemannZeta₀, completedHurwitzZetaEven_eq, if_true]
  /-
    🎉 no goals
  -/


/-- The modified completed Riemann zeta function `Λ(s) + 1 / s + 1 / (1 - s)` is entire. -/
theorem differentiable_completedZeta₀ : Differentiable ℂ completedRiemannZeta₀ :=
  differentiable_completedHurwitzZetaEven₀ 0


/-- The completed Riemann zeta function `Λ(s)` is differentiable away from `s = 0` and `s = 1`. -/
theorem differentiableAt_completedZeta {s : ℂ} (hs : s ≠ 0) (hs' : s ≠ 1) :
    DifferentiableAt ℂ completedRiemannZeta s :=
  differentiableAt_completedHurwitzZetaEven 0 (Or.inl hs) hs'


/-- Riemann zeta functional equation, formulated for `Λ₀`: for any complex `s` we have
`Λ₀(1 - s) = Λ₀ s`. -/
theorem completedRiemannZeta₀_one_sub (s : ℂ) :
    completedRiemannZeta₀ (1 - s) = completedRiemannZeta₀ s := by
  /-
    s : Complex
    ⊢ Eq (completedRiemannZeta₀ (HSub.hSub 1 s)) (completedRiemannZeta₀ s)
  -/
  rw [← completedHurwitzZetaEven₀_zero, ← completedCosZeta₀_zero, completedHurwitzZetaEven₀_one_sub]
  /-
    🎉 no goals
  -/


/-- Riemann zeta functional equation, formulated for `Λ`: for any complex `s` we have
`Λ (1 - s) = Λ s`. -/
theorem completedRiemannZeta_one_sub (s : ℂ) :
    completedRiemannZeta (1 - s) = completedRiemannZeta s := by
  /-
    s : Complex
    ⊢ Eq (completedRiemannZeta (HSub.hSub 1 s)) (completedRiemannZeta s)
  -/
  rw [← completedHurwitzZetaEven_zero, ← completedCosZeta_zero, completedHurwitzZetaEven_one_sub]
  /-
    🎉 no goals
  -/


/-- The residue of `Λ(s)` at `s = 1` is equal to `1`. -/
lemma completedRiemannZeta_residue_one :
    Tendsto (fun s ↦ (s - 1) * completedRiemannZeta s) (𝓝[≠] 1) (𝓝 1) :=
  completedHurwitzZetaEven_residue_one 0


/-- The Riemann zeta function `ζ(s)`. -/
def riemannZeta := hurwitzZetaEven 0


lemma HurwitzZeta.hurwitzZetaEven_zero : hurwitzZetaEven 0 = riemannZeta := rfl


lemma HurwitzZeta.cosZeta_zero : cosZeta 0 = riemannZeta := by
  simp_rw [cosZeta, riemannZeta, hurwitzZetaEven, if_true, completedHurwitzZetaEven_zero,
    completedCosZeta_zero]


lemma HurwitzZeta.hurwitzZeta_zero : hurwitzZeta 0 = riemannZeta := by
  /-
    ⊢ Eq (HurwitzZeta.hurwitzZeta 0) riemannZeta
  -/
  ext1 s
  /-
    case h
    s : Complex
    ⊢ Eq (HurwitzZeta.hurwitzZeta 0 s) (riemannZeta s)
  -/
  simpa [hurwitzZeta, hurwitzZetaEven_zero] using hurwitzZetaOdd_neg 0 s
  /-
    🎉 no goals
  -/


lemma HurwitzZeta.expZeta_zero : expZeta 0 = riemannZeta := by
  /-
    ⊢ Eq (HurwitzZeta.expZeta 0) riemannZeta
  -/
  ext1 s
  rw [expZeta, cosZeta_zero, add_right_eq_self, mul_eq_zero, eq_false_intro I_ne_zero, false_or,
    ← eq_neg_self_iff, ← sinZeta_neg, neg_zero]


/-- The Riemann zeta function is differentiable away from `s = 1`. -/
theorem differentiableAt_riemannZeta {s : ℂ} (hs' : s ≠ 1) : DifferentiableAt ℂ riemannZeta s :=
  differentiableAt_hurwitzZetaEven _ hs'


/-- We have `ζ(0) = -1 / 2`. -/
theorem riemannZeta_zero : riemannZeta 0 = -1 / 2 := by
  /-
    ⊢ Eq (riemannZeta 0) (-1 / 2)
  -/
  simp_rw [riemannZeta, hurwitzZetaEven, Function.update_self, if_true]
  /-
    🎉 no goals
  -/


lemma riemannZeta_def_of_ne_zero {s : ℂ} (hs : s ≠ 0) :
    riemannZeta s = completedRiemannZeta s / Gammaℝ s := by
  /-
    s : Complex
    hs : Ne s 0
    ⊢ Eq (riemannZeta s) (HDiv.hDiv (completedRiemannZeta s) s.Gammaℝ)
  -/
  rw [riemannZeta, hurwitzZetaEven, Function.update_of_ne hs, completedHurwitzZetaEven_zero]
  /-
    🎉 no goals
  -/


/-- The trivial zeroes of the zeta function. -/
theorem riemannZeta_neg_two_mul_nat_add_one (n : ℕ) : riemannZeta (-2 * (n + 1)) = 0 :=
  hurwitzZetaEven_neg_two_mul_nat_add_one 0 n


/-- Riemann zeta functional equation, formulated for `ζ`: if `1 - s ∉ ℕ`, then we have
`ζ (1 - s) = 2 ^ (1 - s) * π ^ (-s) * Γ s * sin (π * (1 - s) / 2) * ζ s`. -/
theorem riemannZeta_one_sub {s : ℂ} (hs : ∀ n : ℕ, s ≠ -n) (hs' : s ≠ 1) :
    riemannZeta (1 - s) = 2 * (2 * π) ^ (-s) * Gamma s * cos (π * s / 2) * riemannZeta s := by
  /-
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Ne s 1
    ⊢ Eq (riemannZeta (HSub.hSub 1 s)) (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul …
  -/
  rw [riemannZeta, hurwitzZetaEven_one_sub 0 hs (Or.inr hs'), cosZeta_zero, hurwitzZetaEven_zero]
  /-
    🎉 no goals
  -/


/-- A formal statement of the **Riemann hypothesis** – constructing a term of this type is worth a
million dollars. -/
def RiemannHypothesis : Prop :=
  ∀ (s : ℂ) (_ : riemannZeta s = 0) (_ : ¬∃ n : ℕ, s = -2 * (n + 1)) (_ : s ≠ 1), s.re = 1 / 2


theorem completedZeta_eq_tsum_of_one_lt_re {s : ℂ} (hs : 1 < re s) :
    completedRiemannZeta s =
      (π : ℂ) ^ (-s / 2) * Gamma (s / 2) * ∑' n : ℕ, 1 / (n : ℂ) ^ s := by
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (completedRiemannZeta s) (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (HDi …
  -/
  have := (hasSum_nat_completedCosZeta 0 hs).tsum_eq.symm
  /-
    s : Complex
    hs : LT.lt 1 s.re
    this : Eq (HurwitzZeta.completedCosZeta (↑0) s) (tsum fun b => ite (Eq b 0) 0  …
    ⊢ Eq (completedRiemannZeta s) (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (HDi …
  -/
  simp only [QuotientAddGroup.mk_zero, completedCosZeta_zero] at this
  simp only [this, Gammaℝ_def, mul_zero, zero_mul, Real.cos_zero, ofReal_one, mul_one, mul_one_div,
    ← tsum_mul_left]
  /-
    s : Complex
    hs : LT.lt 1 s.re
    this : Eq (completedRiemannZeta s) (tsum fun b => ite (Eq b 0) 0 (HDiv.hDiv (H …
    ⊢ Eq (tsum fun b => ite (Eq b 0) 0 (HDiv.hDiv (HMul.hMul (HPow.hPow (↑Real.pi) …
  -/
  congr 1 with n
  /-
    case e_f.h
    s : Complex
    hs : LT.lt 1 s.re
    this : Eq (completedRiemannZeta s) (tsum fun b => ite (Eq b 0) 0 (HDiv.hDiv (H …
    n : Nat
    ⊢ Eq (ite (Eq n 0) 0 (HDiv.hDiv (HMul.hMul (HPow.hPow (↑Real.pi) (HDiv.hDiv (N …
  -/
  split_ifs with h
    /-
      case pos
      s : Complex
      hs : LT.lt 1 s.re
      this : Eq (completedRiemannZeta s) (tsum fun b => ite (Eq b 0) 0 (HDiv.hDiv (H …
      n : Nat
      h : Eq n 0
      ⊢ Eq 0 (HDiv.hDiv (HMul.hMul (HPow.hPow (↑Real.pi) (HDiv.hDiv (Neg.neg s) 2))  …
    -/
  · simp only [h, Nat.cast_zero, zero_cpow (Complex.ne_zero_of_one_lt_re hs), div_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      s : Complex
      hs : LT.lt 1 s.re
      this : Eq (completedRiemannZeta s) (tsum fun b => ite (Eq b 0) 0 (HDiv.hDiv (H …
      n : Nat
      h : Not (Eq n 0)
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (↑Real.pi) (HDiv.hDiv (Neg.neg s) 2)) (C …
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- The Riemann zeta function agrees with the naive Dirichlet-series definition when the latter
converges. (Note that this is false without the assumption: when `re s ≤ 1` the sum is divergent,
and we use a different definition to obtain the analytic continuation to all `s`.) -/
theorem zeta_eq_tsum_one_div_nat_cpow {s : ℂ} (hs : 1 < re s) :
    riemannZeta s = ∑' n : ℕ, 1 / (n : ℂ) ^ s := by
  simpa only [QuotientAddGroup.mk_zero, cosZeta_zero, mul_zero, zero_mul, Real.cos_zero,
    ofReal_one] using (hasSum_nat_cosZeta 0 hs).tsum_eq.symm


/-- Alternate formulation of `zeta_eq_tsum_one_div_nat_cpow` with a `+ 1` (to avoid relying
on mathlib's conventions for `0 ^ s`). -/
theorem zeta_eq_tsum_one_div_nat_add_one_cpow {s : ℂ} (hs : 1 < re s) :
    riemannZeta s = ∑' n : ℕ, 1 / (n + 1 : ℂ) ^ s := by
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (riemannZeta s) (tsum fun n => HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd (↑n) 1)  …
  -/
  have := zeta_eq_tsum_one_div_nat_cpow hs
  /-
    s : Complex
    hs : LT.lt 1 s.re
    this : Eq (riemannZeta s) (tsum fun n => HDiv.hDiv 1 (HPow.hPow (↑n) s))
    ⊢ Eq (riemannZeta s) (tsum fun n => HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd (↑n) 1)  …
  -/
  rw [tsum_eq_zero_add] at this
    /-
      s : Complex
      hs : LT.lt 1 s.re
      this : Eq (riemannZeta s) (HAdd.hAdd (HDiv.hDiv 1 (HPow.hPow (↑0) s)) (tsum fu …
      ⊢ Eq (riemannZeta s) (tsum fun n => HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd (↑n) 1)  …
    -/
  · simpa [zero_cpow (Complex.ne_zero_of_one_lt_re hs)]
    /-
      🎉 no goals
    -/
    /-
      s : Complex
      hs : LT.lt 1 s.re
      this : Eq (riemannZeta s) (tsum fun n => HDiv.hDiv 1 (HPow.hPow (↑n) s))
      ⊢ Summable fun n => HDiv.hDiv 1 (HPow.hPow (↑n) s)
    -/
  · rwa [Complex.summable_one_div_nat_cpow]
    /-
      🎉 no goals
    -/


/-- Special case of `zeta_eq_tsum_one_div_nat_cpow` when the argument is in `ℕ`, so the power
function can be expressed using naïve `pow` rather than `cpow`. -/
theorem zeta_nat_eq_tsum_of_gt_one {k : ℕ} (hk : 1 < k) :
    riemannZeta k = ∑' n : ℕ, 1 / (n : ℂ) ^ k := by
  simp only [zeta_eq_tsum_one_div_nat_cpow
      (by rwa [← ofReal_natCast, ofReal_re, ← Nat.cast_one, Nat.cast_lt] : 1 < re k),
    cpow_natCast]


/-- The residue of `ζ(s)` at `s = 1` is equal to 1. -/
lemma riemannZeta_residue_one : Tendsto (fun s ↦ (s - 1) * riemannZeta s) (𝓝[≠] 1) (𝓝 1) := by
  /-
    ⊢ Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s 1) (riemannZeta s)) (nhdsWit …
  -/
  exact hurwitzZetaEven_residue_one 0
  /-
    🎉 no goals
  -/


/-- The residue of `ζ(s)` at `s = 1` is equal to 1, expressed using `tsum`. -/
theorem tendsto_sub_mul_tsum_nat_cpow :
    Tendsto (fun s : ℂ ↦ (s - 1) * ∑' (n : ℕ), 1 / (n : ℂ) ^ s) (𝓝[{s | 1 < re s}] 1) (𝓝 1) := by
  /-
    ⊢ Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s 1) (tsum fun n => HDiv.hDiv  …
  -/
  refine (tendsto_nhdsWithin_mono_left ?_ riemannZeta_residue_one).congr' ?_
    /-
      case refine_1
      ⊢ HasSubset.Subset (setOf fun s => LT.lt 1 s.re) (HasCompl.compl (Singleton.si …
    -/
  · simp only [subset_compl_singleton_iff, mem_setOf_eq, one_re, not_lt, le_refl]
    /-
      🎉 no goals
    -/
  · filter_upwards [eventually_mem_nhdsWithin] with s hs using
      congr_arg _ <| zeta_eq_tsum_one_div_nat_cpow hs


/-- The residue of `ζ(s)` at `s = 1` is equal to 1 expressed using `tsum` and for a
real variable. -/
theorem tendsto_sub_mul_tsum_nat_rpow :
    Tendsto (fun s : ℝ ↦ (s - 1) * ∑' (n : ℕ), 1 / (n : ℝ) ^ s) (𝓝[>] 1) (𝓝 1) := by
  /-
    ⊢ Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s 1) (tsum fun n => HDiv.hDiv  …
  -/
  rw [← tendsto_ofReal_iff, ofReal_one]
  have : Tendsto (fun s : ℝ ↦ (s : ℂ)) (𝓝[>] 1) (𝓝[{s | 1 < re s}] 1) :=
    continuous_ofReal.continuousWithinAt.tendsto_nhdsWithin (fun _ _ ↦ by aesop)
  /-
    this : Filter.Tendsto (fun s => ↑s) (nhdsWithin 1 (Set.Ioi 1)) (nhdsWithin 1 ( …
    ⊢ Filter.Tendsto (fun x => ↑(HMul.hMul (HSub.hSub x 1) (tsum fun n => HDiv.hDi …
  -/
  apply (tendsto_sub_mul_tsum_nat_cpow.comp this).congr fun s ↦ ?_
  simp only [one_div, Function.comp_apply, ofReal_mul, ofReal_sub, ofReal_one, ofReal_tsum,
    ofReal_inv, ofReal_cpow (Nat.cast_nonneg _), ofReal_natCast]

/- naming scheme was changed from `riemannCompletedZeta` to `completedRiemannZeta`; add
aliases for the old names -/

@[deprecated (since := "2024-05-27")]
noncomputable alias riemannCompletedZeta₀ := completedRiemannZeta₀


@[deprecated (since := "2024-05-27")]
noncomputable alias riemannCompletedZeta := completedRiemannZeta


@[deprecated (since := "2024-05-27")]
alias riemannCompletedZeta₀_one_sub := completedRiemannZeta₀_one_sub


@[deprecated (since := "2024-05-27")]
alias riemannCompletedZeta_one_sub := completedRiemannZeta_one_sub


@[deprecated (since := "2024-05-27")]
alias riemannCompletedZeta_residue_one := completedRiemannZeta_residue_one


