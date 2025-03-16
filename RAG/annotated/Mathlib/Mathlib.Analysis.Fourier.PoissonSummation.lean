/-- The key lemma for Poisson summation: the `m`-th Fourier coefficient of the periodic function
`∑' n : ℤ, f (x + n)` is the value at `m` of the Fourier transform of `f`. -/
theorem Real.fourierCoeff_tsum_comp_add {f : C(ℝ, ℂ)}
    (hf : ∀ K : Compacts ℝ, Summable fun n : ℤ => ‖(f.comp (ContinuousMap.addRight n)).restrict K‖)
    (m : ℤ) : fourierCoeff (Periodic.lift <| f.periodic_tsum_comp_add_zsmul 1) m = 𝓕 f m := by
  -- NB: This proof can be shortened somewhat by telescoping together some of the steps in the calc
  -- block, but I think it's more legible this way. We start with preliminaries about the integrand.
  /-
    f : ContinuousMap Real Complex
    hf : ∀ (K : TopologicalSpace.Compacts Real), Summable fun n => Norm.norm (Cont …
    m : Int
    ⊢ Eq (fourierCoeff ⋯.lift m) (Real.fourierIntegral ⇑f ↑m)
  -/
  let e : C(ℝ, ℂ) := (fourier (-m)).comp ⟨((↑) : ℝ → UnitAddCircle), continuous_quotient_mk'⟩
  have neK : ∀ (K : Compacts ℝ) (g : C(ℝ, ℂ)), ‖(e * g).restrict K‖ = ‖g.restrict K‖ := by
    have (x : ℝ) : ‖e x‖ = 1 := (AddCircle.toCircle (-m • x)).abs_coe
    intro K g
    simp_rw [norm_eq_iSup_norm, restrict_apply, mul_apply, norm_mul, this, one_mul]
  have eadd : ∀ (n : ℤ), e.comp (ContinuousMap.addRight n) = e := by
    intro n; ext1 x
    have : Periodic e 1 := Periodic.comp (fun x => AddCircle.coe_add_period 1 x) (fourier (-m))
    simpa only [mul_one] using this.int_mul n x
  -- Now the main argument. First unwind some definitions.
  calc
    fourierCoeff (Periodic.lift <| f.periodic_tsum_comp_add_zsmul 1) m =
        ∫ x in (0 : ℝ)..1, e x * (∑' n : ℤ, f.comp (ContinuousMap.addRight n)) x := by
      simp_rw [fourierCoeff_eq_intervalIntegral _ m 0, div_one, one_smul, zero_add, e, comp_apply,
        coe_mk, Periodic.lift_coe, zsmul_one, smul_eq_mul]
    -- Transform sum in C(ℝ, ℂ) evaluated at x into pointwise sum of values.
    _ = ∫ x in (0 : ℝ)..1, ∑' n : ℤ, (e * f.comp (ContinuousMap.addRight n)) x := by
      simp_rw [coe_mul, Pi.mul_apply,
        ← ContinuousMap.tsum_apply (summable_of_locally_summable_norm hf), tsum_mul_left]
    -- Swap sum and integral.
    _ = ∑' n : ℤ, ∫ x in (0 : ℝ)..1, (e * f.comp (ContinuousMap.addRight n)) x := by
      refine (intervalIntegral.tsum_intervalIntegral_eq_of_summable_norm ?_).symm
      convert hf ⟨uIcc 0 1, isCompact_uIcc⟩ using 1
      exact funext fun n => neK _ _
    _ = ∑' n : ℤ, ∫ x in (0 : ℝ)..1, (e * f).comp (ContinuousMap.addRight n) x := by
      simp only [ContinuousMap.comp_apply, mul_comp] at eadd ⊢
      simp_rw [eadd]
    -- Rearrange sum of interval integrals into an integral over `ℝ`.
    _ = ∫ x, e x * f x := by
      suffices Integrable (e * f) from this.hasSum_intervalIntegral_comp_add_int.tsum_eq
      apply integrable_of_summable_norm_Icc
      convert hf ⟨Icc 0 1, isCompact_Icc⟩ using 1
      simp_rw [mul_comp] at eadd ⊢
      simp_rw [eadd]
      exact funext fun n => neK ⟨Icc 0 1, isCompact_Icc⟩ _
    -- Minor tidying to finish
    _ = 𝓕 f m := by
      rw [fourierIntegral_real_eq_integral_exp_smul]
      congr 1 with x : 1
      rw [smul_eq_mul, comp_apply, coe_mk, coe_mk, ContinuousMap.toFun_eq_coe, fourier_coe_apply]
      congr 2
      push_cast
      ring


/-- **Poisson's summation formula**, most general form. -/
theorem Real.tsum_eq_tsum_fourierIntegral {f : C(ℝ, ℂ)}
    (h_norm :
      ∀ K : Compacts ℝ, Summable fun n : ℤ => ‖(f.comp <| ContinuousMap.addRight n).restrict K‖)
    (h_sum : Summable fun n : ℤ => 𝓕 f n) (x : ℝ) :
    ∑' n : ℤ, f (x + n) = ∑' n : ℤ, 𝓕 f n * fourier n (x : UnitAddCircle) := by
  let F : C(UnitAddCircle, ℂ) :=
    ⟨(f.periodic_tsum_comp_add_zsmul 1).lift, continuous_coinduced_dom.mpr (map_continuous _)⟩
  have : Summable (fourierCoeff F) := by
    convert h_sum
    exact Real.fourierCoeff_tsum_comp_add h_norm _
  /-
    f : ContinuousMap Real Complex
    h_norm : ∀ (K : TopologicalSpace.Compacts Real), Summable fun n => Norm.norm ( …
    h_sum : Summable fun n => Real.fourierIntegral ⇑f ↑n
    x : Real
    F : ContinuousMap UnitAddCircle Complex := { toFun := ⋯.lift, continuous_toFun …
    this : Summable (fourierCoeff ⇑F)
    ⊢ Eq (tsum fun n => f (HAdd.hAdd x ↑n)) (tsum fun n => HMul.hMul (Real.fourier …
  -/
  convert (has_pointwise_sum_fourier_series_of_summable this x).tsum_eq.symm using 1
  · simpa only [F, coe_mk, ← QuotientAddGroup.mk_zero, Periodic.lift_coe, zsmul_one, comp_apply,
      coe_addRight, zero_add]
       using (hasSum_apply (summable_of_locally_summable_norm h_norm).hasSum x).tsum_eq
    /-
      case h.e'_3
      f : ContinuousMap Real Complex
      h_norm : ∀ (K : TopologicalSpace.Compacts Real), Summable fun n => Norm.norm ( …
      h_sum : Summable fun n => Real.fourierIntegral ⇑f ↑n
      x : Real
      F : ContinuousMap UnitAddCircle Complex := { toFun := ⋯.lift, continuous_toFun …
      this : Summable (fourierCoeff ⇑F)
      ⊢ Eq (tsum fun n => HMul.hMul (Real.fourierIntegral ⇑f ↑n) ((fourier n) ↑x)) ( …
    -/
  · simp_rw [← Real.fourierCoeff_tsum_comp_add h_norm, smul_eq_mul, F, coe_mk]
    /-
      🎉 no goals
    -/


/-- If `f` is `O(x ^ (-b))` at infinity, then so is the function
`fun x ↦ ‖f.restrict (Icc (x + R) (x + S))‖` for any fixed `R` and `S`. -/
theorem isBigO_norm_Icc_restrict_atTop {f : C(ℝ, E)} {b : ℝ} (hb : 0 < b)
    (hf : f =O[atTop] fun x : ℝ => |x| ^ (-b)) (R S : ℝ) :
    (fun x : ℝ => ‖f.restrict (Icc (x + R) (x + S))‖) =O[atTop] fun x : ℝ => |x| ^ (-b) := by
  -- First establish an explicit estimate on decay of inverse powers.
  -- This is logically independent of the rest of the proof, but of no mathematical interest in
  -- itself, so it is proved in-line rather than being formulated as a separate lemma.
  have claim : ∀ x : ℝ, max 0 (-2 * R) < x → ∀ y : ℝ, x + R ≤ y →
      y ^ (-b) ≤ (1 / 2) ^ (-b) * x ^ (-b) := fun x hx y hy ↦ by
    rw [max_lt_iff] at hx
    obtain ⟨hx1, hx2⟩ := hx
    rw [← mul_rpow] <;> try positivity
    apply rpow_le_rpow_of_nonpos <;> linarith
  -- Now the main proof.
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (ContinuousMap.restrict  …
  -/
  obtain ⟨c, hc, hc'⟩ := hf.exists_pos
  /-
    case intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    hc' : Asymptotics.IsBigOWith c Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg …
    ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (ContinuousMap.restrict  …
  -/
  simp only [IsBigO, IsBigOWith, eventually_atTop] at hc' ⊢
  /-
    case intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    hc' : Exists fun a => ∀ (b_1 : Real), GE.ge b_1 a → LE.le (Norm.norm (f b_1))  …
    ⊢ Exists fun c => Exists fun a => ∀ (b_1 : Real), GE.ge b_1 a → LE.le (Norm.no …
  -/
  obtain ⟨d, hd⟩ := hc'
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    d : Real
    hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
    ⊢ Exists fun c => Exists fun a => ∀ (b_1 : Real), GE.ge b_1 a → LE.le (Norm.no …
  -/
  refine ⟨c * (1 / 2) ^ (-b), ⟨max (1 + max 0 (-2 * R)) (d - R), fun x hx => ?_⟩⟩
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    d : Real
    hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
    x : Real
    hx : GE.ge x (Max.max (HAdd.hAdd 1 (Max.max 0 (HMul.hMul (-2) R))) (HSub.hSub  …
    ⊢ LE.le (Norm.norm (Norm.norm (ContinuousMap.restrict (Set.Icc (HAdd.hAdd x R) …
  -/
  rw [ge_iff_le, max_le_iff] at hx
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    d : Real
    hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
    x : Real
    hx : And (LE.le (HAdd.hAdd 1 (Max.max 0 (HMul.hMul (-2) R))) x) (LE.le (HSub.h …
    ⊢ LE.le (Norm.norm (Norm.norm (ContinuousMap.restrict (Set.Icc (HAdd.hAdd x R) …
  -/
  have hx' : max 0 (-2 * R) < x := by linarith
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    d : Real
    hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
    x : Real
    hx : And (LE.le (HAdd.hAdd 1 (Max.max 0 (HMul.hMul (-2) R))) x) (LE.le (HSub.h …
    hx' : LT.lt (Max.max 0 (HMul.hMul (-2) R)) x
    ⊢ LE.le (Norm.norm (Norm.norm (ContinuousMap.restrict (Set.Icc (HAdd.hAdd x R) …
  -/
  rw [max_lt_iff] at hx'
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    d : Real
    hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
    x : Real
    hx : And (LE.le (HAdd.hAdd 1 (Max.max 0 (HMul.hMul (-2) R))) x) (LE.le (HSub.h …
    hx' : And (LT.lt 0 x) (LT.lt (HMul.hMul (-2) R) x)
    ⊢ LE.le (Norm.norm (Norm.norm (ContinuousMap.restrict (Set.Icc (HAdd.hAdd x R) …
  -/
  rw [norm_norm, ContinuousMap.norm_le _ (by positivity)]
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    d : Real
    hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
    x : Real
    hx : And (LE.le (HAdd.hAdd 1 (Max.max 0 (HMul.hMul (-2) R))) x) (LE.le (HSub.h …
    hx' : And (LT.lt 0 x) (LT.lt (HMul.hMul (-2) R) x)
    ⊢ ∀ (x_1 : ↑(Set.Icc (HAdd.hAdd x R) (HAdd.hAdd x S))), LE.le (Norm.norm ((Con …
  -/
  refine fun y => (hd y.1 (by linarith [hx.1, y.2.1])).trans ?_
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    d : Real
    hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
    x : Real
    hx : And (LE.le (HAdd.hAdd 1 (Max.max 0 (HMul.hMul (-2) R))) x) (LE.le (HSub.h …
    hx' : And (LT.lt 0 x) (LT.lt (HMul.hMul (-2) R) x)
    y : ↑(Set.Icc (HAdd.hAdd x R) (HAdd.hAdd x S))
    ⊢ LE.le (HMul.hMul c (Norm.norm (HPow.hPow (abs ↑y) (Neg.neg b)))) (HMul.hMul  …
  -/
  have A : ∀ x : ℝ, 0 ≤ |x| ^ (-b) := fun x => by positivity
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    d : Real
    hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
    x : Real
    hx : And (LE.le (HAdd.hAdd 1 (Max.max 0 (HMul.hMul (-2) R))) x) (LE.le (HSub.h …
    hx' : And (LT.lt 0 x) (LT.lt (HMul.hMul (-2) R) x)
    y : ↑(Set.Icc (HAdd.hAdd x R) (HAdd.hAdd x S))
    A : ∀ (x : Real), LE.le 0 (HPow.hPow (abs x) (Neg.neg b))
    ⊢ LE.le (HMul.hMul c (Norm.norm (HPow.hPow (abs ↑y) (Neg.neg b)))) (HMul.hMul  …
  -/
  rw [mul_assoc, mul_le_mul_left hc, norm_of_nonneg (A _), norm_of_nonneg (A _)]
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
    c : Real
    hc : GT.gt c 0
    d : Real
    hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
    x : Real
    hx : And (LE.le (HAdd.hAdd 1 (Max.max 0 (HMul.hMul (-2) R))) x) (LE.le (HSub.h …
    hx' : And (LT.lt 0 x) (LT.lt (HMul.hMul (-2) R) x)
    y : ↑(Set.Icc (HAdd.hAdd x R) (HAdd.hAdd x S))
    A : ∀ (x : Real), LE.le 0 (HPow.hPow (abs x) (Neg.neg b))
    ⊢ LE.le (HPow.hPow (abs ↑y) (Neg.neg b)) (HMul.hMul (HPow.hPow (1 / 2) (Neg.ne …
  -/
  convert claim x (by linarith only [hx.1]) y.1 y.2.1
    /-
      case h.e'_3.h.e'_5
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : ContinuousMap Real E
      b : Real
      hb : LT.lt 0 b
      hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
      R S : Real
      claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
      c : Real
      hc : GT.gt c 0
      d : Real
      hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
      x : Real
      hx : And (LE.le (HAdd.hAdd 1 (Max.max 0 (HMul.hMul (-2) R))) x) (LE.le (HSub.h …
      hx' : And (LT.lt 0 x) (LT.lt (HMul.hMul (-2) R) x)
      y : ↑(Set.Icc (HAdd.hAdd x R) (HAdd.hAdd x S))
      A : ∀ (x : Real), LE.le 0 (HPow.hPow (abs x) (Neg.neg b))
      ⊢ Eq (abs ↑y) ↑y
    -/
  · apply abs_of_nonneg; linarith [y.2.1]
                         /-
                           🎉 no goals
                         -/
    /-
      case h.e'_4.h.e'_6.h.e'_5
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : ContinuousMap Real E
      b : Real
      hb : LT.lt 0 b
      hf : Asymptotics.IsBigO Filter.atTop ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
      R S : Real
      claim : ∀ (x : Real), LT.lt (Max.max 0 (HMul.hMul (-2) R)) x → ∀ (y : Real), L …
      c : Real
      hc : GT.gt c 0
      d : Real
      hd : ∀ (b_1 : Real), GE.ge b_1 d → LE.le (Norm.norm (f b_1)) (HMul.hMul c (Nor …
      x : Real
      hx : And (LE.le (HAdd.hAdd 1 (Max.max 0 (HMul.hMul (-2) R))) x) (LE.le (HSub.h …
      hx' : And (LT.lt 0 x) (LT.lt (HMul.hMul (-2) R) x)
      y : ↑(Set.Icc (HAdd.hAdd x R) (HAdd.hAdd x S))
      A : ∀ (x : Real), LE.le 0 (HPow.hPow (abs x) (Neg.neg b))
      ⊢ Eq (abs x) x
    -/
  · exact abs_of_pos hx'.1
    /-
      🎉 no goals
    -/


theorem isBigO_norm_Icc_restrict_atBot {f : C(ℝ, E)} {b : ℝ} (hb : 0 < b)
    (hf : f =O[atBot] fun x : ℝ => |x| ^ (-b)) (R S : ℝ) :
    (fun x : ℝ => ‖f.restrict (Icc (x + R) (x + S))‖) =O[atBot] fun x : ℝ => |x| ^ (-b) := by
  have h1 : (f.comp (ContinuousMap.mk _ continuous_neg)) =O[atTop] fun x : ℝ => |x| ^ (-b) := by
    convert hf.comp_tendsto tendsto_neg_atTop_atBot using 1
    ext1 x; simp only [Function.comp_apply, abs_neg]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    h1 : Asymptotics.IsBigO Filter.atTop ⇑(f.comp { toFun := fun a => Neg.neg a, c …
    ⊢ Asymptotics.IsBigO Filter.atBot (fun x => Norm.norm (ContinuousMap.restrict  …
  -/
  have h2 := (isBigO_norm_Icc_restrict_atTop hb h1 (-S) (-R)).comp_tendsto tendsto_neg_atBot_atTop
  have : (fun x : ℝ => |x| ^ (-b)) ∘ Neg.neg = fun x : ℝ => |x| ^ (-b) := by
    ext1 x; simp only [Function.comp_apply, abs_neg]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    h1 : Asymptotics.IsBigO Filter.atTop ⇑(f.comp { toFun := fun a => Neg.neg a, c …
    h2 : Asymptotics.IsBigO Filter.atBot (Function.comp (fun x => Norm.norm (Conti …
    this : Eq (Function.comp (fun x => HPow.hPow (abs x) (Neg.neg b)) Neg.neg) fun …
    ⊢ Asymptotics.IsBigO Filter.atBot (fun x => Norm.norm (ContinuousMap.restrict  …
  -/
  rw [this] at h2
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    h1 : Asymptotics.IsBigO Filter.atTop ⇑(f.comp { toFun := fun a => Neg.neg a, c …
    h2 : Asymptotics.IsBigO Filter.atBot (Function.comp (fun x => Norm.norm (Conti …
    this : Eq (Function.comp (fun x => HPow.hPow (abs x) (Neg.neg b)) Neg.neg) fun …
    ⊢ Asymptotics.IsBigO Filter.atBot (fun x => Norm.norm (ContinuousMap.restrict  …
  -/
  refine (isBigO_of_le _ fun x => ?_).trans h2
  -- equality holds, but less work to prove `≤` alone
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    h1 : Asymptotics.IsBigO Filter.atTop ⇑(f.comp { toFun := fun a => Neg.neg a, c …
    h2 : Asymptotics.IsBigO Filter.atBot (Function.comp (fun x => Norm.norm (Conti …
    this : Eq (Function.comp (fun x => HPow.hPow (abs x) (Neg.neg b)) Neg.neg) fun …
    x : Real
    ⊢ LE.le (Norm.norm (Norm.norm (ContinuousMap.restrict (Set.Icc (HAdd.hAdd x R) …
  -/
  rw [norm_norm, Function.comp_apply, norm_norm, ContinuousMap.norm_le _ (norm_nonneg _)]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    h1 : Asymptotics.IsBigO Filter.atTop ⇑(f.comp { toFun := fun a => Neg.neg a, c …
    h2 : Asymptotics.IsBigO Filter.atBot (Function.comp (fun x => Norm.norm (Conti …
    this : Eq (Function.comp (fun x => HPow.hPow (abs x) (Neg.neg b)) Neg.neg) fun …
    x : Real
    ⊢ ∀ (x_1 : ↑(Set.Icc (HAdd.hAdd x R) (HAdd.hAdd x S))), LE.le (Norm.norm ((Con …
  -/
  rintro ⟨x, hx⟩
  /-
    case mk
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    h1 : Asymptotics.IsBigO Filter.atTop ⇑(f.comp { toFun := fun a => Neg.neg a, c …
    h2 : Asymptotics.IsBigO Filter.atBot (Function.comp (fun x => Norm.norm (Conti …
    this : Eq (Function.comp (fun x => HPow.hPow (abs x) (Neg.neg b)) Neg.neg) fun …
    x✝ x : Real
    hx : Membership.mem (Set.Icc (HAdd.hAdd x✝ R) (HAdd.hAdd x✝ S)) x
    ⊢ LE.le (Norm.norm ((ContinuousMap.restrict (Set.Icc (HAdd.hAdd x✝ R) (HAdd.hA …
  -/
  rw [ContinuousMap.restrict_apply_mk]
  /-
    case mk
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
    R S : Real
    h1 : Asymptotics.IsBigO Filter.atTop ⇑(f.comp { toFun := fun a => Neg.neg a, c …
    h2 : Asymptotics.IsBigO Filter.atBot (Function.comp (fun x => Norm.norm (Conti …
    this : Eq (Function.comp (fun x => HPow.hPow (abs x) (Neg.neg b)) Neg.neg) fun …
    x✝ x : Real
    hx : Membership.mem (Set.Icc (HAdd.hAdd x✝ R) (HAdd.hAdd x✝ S)) x
    ⊢ LE.le (Norm.norm (f x)) (Norm.norm (ContinuousMap.restrict (Set.Icc (HAdd.hA …
  -/
  refine (le_of_eq ?_).trans (ContinuousMap.norm_coe_le_norm _ ⟨-x, ?_⟩)
  · rw [ContinuousMap.restrict_apply_mk, ContinuousMap.comp_apply, ContinuousMap.coe_mk,
      ContinuousMap.coe_mk, neg_neg]
    /-
      case mk.refine_2
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : ContinuousMap Real E
      b : Real
      hb : LT.lt 0 b
      hf : Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.neg b)
      R S : Real
      h1 : Asymptotics.IsBigO Filter.atTop ⇑(f.comp { toFun := fun a => Neg.neg a, c …
      h2 : Asymptotics.IsBigO Filter.atBot (Function.comp (fun x => Norm.norm (Conti …
      this : Eq (Function.comp (fun x => HPow.hPow (abs x) (Neg.neg b)) Neg.neg) fun …
      x✝ x : Real
      hx : Membership.mem (Set.Icc (HAdd.hAdd x✝ R) (HAdd.hAdd x✝ S)) x
      ⊢ Membership.mem (Set.Icc (HAdd.hAdd (Neg.neg x✝) (Neg.neg S)) (HAdd.hAdd (Neg …
    -/
  · exact ⟨by linarith [hx.2], by linarith [hx.1]⟩
    /-
      🎉 no goals
    -/


theorem isBigO_norm_restrict_cocompact (f : C(ℝ, E)) {b : ℝ} (hb : 0 < b)
    (hf : f =O[cocompact ℝ] fun x : ℝ => |x| ^ (-b)) (K : Compacts ℝ) :
    (fun x => ‖(f.comp (ContinuousMap.addRight x)).restrict K‖) =O[cocompact ℝ] (|·| ^ (-b)) := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO (Filter.cocompact Real) ⇑f fun x => HPow.hPow (abs x)  …
    K : TopologicalSpace.Compacts Real
    ⊢ Asymptotics.IsBigO (Filter.cocompact Real) (fun x => Norm.norm (ContinuousMa …
  -/
  obtain ⟨r, hr⟩ := K.isCompact.isBounded.subset_closedBall 0
  /-
    case intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO (Filter.cocompact Real) ⇑f fun x => HPow.hPow (abs x)  …
    K : TopologicalSpace.Compacts Real
    r : Real
    hr : HasSubset.Subset (↑K) (Metric.closedBall 0 r)
    ⊢ Asymptotics.IsBigO (Filter.cocompact Real) (fun x => Norm.norm (ContinuousMa …
  -/
  rw [closedBall_eq_Icc, zero_add, zero_sub] at hr
  have : ∀ x : ℝ,
      ‖(f.comp (ContinuousMap.addRight x)).restrict K‖ ≤ ‖f.restrict (Icc (x - r) (x + r))‖ := by
    intro x
    rw [ContinuousMap.norm_le _ (norm_nonneg _)]
    rintro ⟨y, hy⟩
    refine (le_of_eq ?_).trans (ContinuousMap.norm_coe_le_norm _ ⟨y + x, ?_⟩)
    · simp_rw [ContinuousMap.restrict_apply, ContinuousMap.comp_apply, ContinuousMap.coe_addRight]
    · exact ⟨by linarith [(hr hy).1], by linarith [(hr hy).2]⟩
  /-
    case intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    hf : Asymptotics.IsBigO (Filter.cocompact Real) ⇑f fun x => HPow.hPow (abs x)  …
    K : TopologicalSpace.Compacts Real
    r : Real
    hr : HasSubset.Subset (↑K) (Set.Icc (Neg.neg r) r)
    this : ∀ (x : Real), LE.le (Norm.norm (ContinuousMap.restrict (↑K) (f.comp (Co …
    ⊢ Asymptotics.IsBigO (Filter.cocompact Real) (fun x => Norm.norm (ContinuousMa …
  -/
  simp_rw [cocompact_eq_atBot_atTop, isBigO_sup] at hf ⊢
  /-
    case intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : ContinuousMap Real E
    b : Real
    hb : LT.lt 0 b
    K : TopologicalSpace.Compacts Real
    r : Real
    hr : HasSubset.Subset (↑K) (Set.Icc (Neg.neg r) r)
    this : ∀ (x : Real), LE.le (Norm.norm (ContinuousMap.restrict (↑K) (f.comp (Co …
    hf : And (Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.n …
    ⊢ And (Asymptotics.IsBigO Filter.atBot (fun x => Norm.norm (ContinuousMap.rest …
  -/
  constructor
    /-
      case intro.left
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : ContinuousMap Real E
      b : Real
      hb : LT.lt 0 b
      K : TopologicalSpace.Compacts Real
      r : Real
      hr : HasSubset.Subset (↑K) (Set.Icc (Neg.neg r) r)
      this : ∀ (x : Real), LE.le (Norm.norm (ContinuousMap.restrict (↑K) (f.comp (Co …
      hf : And (Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.n …
      ⊢ Asymptotics.IsBigO Filter.atBot (fun x => Norm.norm (ContinuousMap.restrict  …
    -/
  · refine (isBigO_of_le atBot ?_).trans (isBigO_norm_Icc_restrict_atBot hb hf.1 (-r) r)
    /-
      case intro.left
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : ContinuousMap Real E
      b : Real
      hb : LT.lt 0 b
      K : TopologicalSpace.Compacts Real
      r : Real
      hr : HasSubset.Subset (↑K) (Set.Icc (Neg.neg r) r)
      this : ∀ (x : Real), LE.le (Norm.norm (ContinuousMap.restrict (↑K) (f.comp (Co …
      hf : And (Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.n …
      ⊢ ∀ (x : Real), LE.le (Norm.norm (Norm.norm (ContinuousMap.restrict (↑K) (f.co …
    -/
    simp_rw [norm_norm]; exact this
                         /-
                           🎉 no goals
                         -/
    /-
      case intro.right
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : ContinuousMap Real E
      b : Real
      hb : LT.lt 0 b
      K : TopologicalSpace.Compacts Real
      r : Real
      hr : HasSubset.Subset (↑K) (Set.Icc (Neg.neg r) r)
      this : ∀ (x : Real), LE.le (Norm.norm (ContinuousMap.restrict (↑K) (f.comp (Co …
      hf : And (Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.n …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (ContinuousMap.restrict  …
    -/
  · refine (isBigO_of_le atTop ?_).trans (isBigO_norm_Icc_restrict_atTop hb hf.2 (-r) r)
    /-
      case intro.right
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : ContinuousMap Real E
      b : Real
      hb : LT.lt 0 b
      K : TopologicalSpace.Compacts Real
      r : Real
      hr : HasSubset.Subset (↑K) (Set.Icc (Neg.neg r) r)
      this : ∀ (x : Real), LE.le (Norm.norm (ContinuousMap.restrict (↑K) (f.comp (Co …
      hf : And (Asymptotics.IsBigO Filter.atBot ⇑f fun x => HPow.hPow (abs x) (Neg.n …
      ⊢ ∀ (x : Real), LE.le (Norm.norm (Norm.norm (ContinuousMap.restrict (↑K) (f.co …
    -/
    simp_rw [norm_norm]; exact this
                         /-
                           🎉 no goals
                         -/


/-- **Poisson's summation formula**, assuming that `f` decays as
`|x| ^ (-b)` for some `1 < b` and its Fourier transform is summable. -/
theorem Real.tsum_eq_tsum_fourierIntegral_of_rpow_decay_of_summable {f : ℝ → ℂ} (hc : Continuous f)
    {b : ℝ} (hb : 1 < b) (hf : IsBigO (cocompact ℝ) f fun x : ℝ => |x| ^ (-b))
    (hFf : Summable fun n : ℤ => 𝓕 f n) (x : ℝ) :
    ∑' n : ℤ, f (x + n) = ∑' n : ℤ, 𝓕 f n * fourier n (x : UnitAddCircle) :=
  Real.tsum_eq_tsum_fourierIntegral (fun K => summable_of_isBigO (Real.summable_abs_int_rpow hb)
    ((isBigO_norm_restrict_cocompact ⟨_, hc⟩ (zero_lt_one.trans hb) hf K).comp_tendsto
    Int.tendsto_coe_cofinite)) hFf x


/-- **Poisson's summation formula**, assuming that both `f` and its Fourier transform decay as
`|x| ^ (-b)` for some `1 < b`. (This is the one-dimensional case of Corollary VII.2.6 of Stein and
Weiss, *Introduction to Fourier analysis on Euclidean spaces*.) -/
theorem Real.tsum_eq_tsum_fourierIntegral_of_rpow_decay {f : ℝ → ℂ} (hc : Continuous f) {b : ℝ}
    (hb : 1 < b) (hf : f =O[cocompact ℝ] (|·| ^ (-b)))
    (hFf : (𝓕 f) =O[cocompact ℝ] (|·| ^ (-b))) (x : ℝ) :
    ∑' n : ℤ, f (x + n) = ∑' n : ℤ, 𝓕 f n * fourier n (x : UnitAddCircle) :=
  Real.tsum_eq_tsum_fourierIntegral_of_rpow_decay_of_summable hc hb hf (summable_of_isBigO
    (Real.summable_abs_int_rpow hb) (hFf.comp_tendsto Int.tendsto_coe_cofinite)) x


/-- **Poisson's summation formula** for Schwartz functions. -/
theorem SchwartzMap.tsum_eq_tsum_fourierIntegral (f : SchwartzMap ℝ ℂ) (x : ℝ) :
    ∑' n : ℤ, f (x + n) = ∑' n : ℤ, fourierTransformCLM ℝ f n * fourier n (x : UnitAddCircle) := by
  -- We know that Schwartz functions are `O(‖x ^ (-b)‖)` for *every* `b`; for this argument we take
  -- `b = 2` and work with that.
  apply Real.tsum_eq_tsum_fourierIntegral_of_rpow_decay f.continuous one_lt_two
    (f.isBigO_cocompact_rpow (-2)) ((fourierTransformCLM ℝ f).isBigO_cocompact_rpow (-2))


