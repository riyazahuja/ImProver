/-- The exponential in a Banach algebra `𝔸` over a normed field `𝕂` has strict Fréchet derivative
`1 : 𝔸 →L[𝕂] 𝔸` at zero, as long as it converges on a neighborhood of zero. -/
theorem hasStrictFDerivAt_exp_zero_of_radius_pos (h : 0 < (expSeries 𝕂 𝔸).radius) :
    HasStrictFDerivAt (exp 𝕂) (1 : 𝔸 →L[𝕂] 𝔸) 0 := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕂
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    h : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    ⊢ HasStrictFDerivAt (NormedSpace.exp 𝕂) 1 0
  -/
  convert (hasFPowerSeriesAt_exp_zero_of_radius_pos h).hasStrictFDerivAt
  /-
    case h.e'_12
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕂
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    h : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    ⊢ Eq 1 ((continuousMultilinearCurryFin1 𝕂 𝔸 𝔸) (NormedSpace.expSeries 𝕂 𝔸 1))
  -/
  ext x
  /-
    case h.e'_12.h
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕂
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    h : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    x : 𝔸
    ⊢ Eq (1 x) (((continuousMultilinearCurryFin1 𝕂 𝔸 𝔸) (NormedSpace.expSeries 𝕂 𝔸 …
  -/
  change x = expSeries 𝕂 𝔸 1 fun _ => x
  /-
    case h.e'_12.h
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕂
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    h : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    x : 𝔸
    ⊢ Eq x ((NormedSpace.expSeries 𝕂 𝔸 1) fun x_1 => x)
  -/
  simp [expSeries_apply_eq, Nat.factorial]
  /-
    🎉 no goals
  -/


/-- The exponential in a Banach algebra `𝔸` over a normed field `𝕂` has Fréchet derivative
`1 : 𝔸 →L[𝕂] 𝔸` at zero, as long as it converges on a neighborhood of zero. -/
theorem hasFDerivAt_exp_zero_of_radius_pos (h : 0 < (expSeries 𝕂 𝔸).radius) :
    HasFDerivAt (exp 𝕂) (1 : 𝔸 →L[𝕂] 𝔸) 0 :=
  (hasStrictFDerivAt_exp_zero_of_radius_pos h).hasFDerivAt


/-- The exponential map in a commutative Banach algebra `𝔸` over a normed field `𝕂` of
characteristic zero has Fréchet derivative `NormedSpace.exp 𝕂 x • 1 : 𝔸 →L[𝕂] 𝔸`
at any point `x`in the disk of convergence. -/
theorem hasFDerivAt_exp_of_mem_ball [CharZero 𝕂] {x : 𝔸}
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasFDerivAt (exp 𝕂) (exp 𝕂 x • (1 : 𝔸 →L[𝕂] 𝔸)) x := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : NormedCommRing 𝔸
    inst✝² : NormedAlgebra 𝕂 𝔸
    inst✝¹ : CompleteSpace 𝔸
    inst✝ : CharZero 𝕂
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ HasFDerivAt (NormedSpace.exp 𝕂) (HSMul.hSMul (NormedSpace.exp 𝕂 x) 1) x
  -/
  have hpos : 0 < (expSeries 𝕂 𝔸).radius := (zero_le _).trans_lt hx
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : NormedCommRing 𝔸
    inst✝² : NormedAlgebra 𝕂 𝔸
    inst✝¹ : CompleteSpace 𝔸
    inst✝ : CharZero 𝕂
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    hpos : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    ⊢ HasFDerivAt (NormedSpace.exp 𝕂) (HSMul.hSMul (NormedSpace.exp 𝕂 x) 1) x
  -/
  rw [hasFDerivAt_iff_isLittleO_nhds_zero]
  suffices
    (fun h => exp 𝕂 x * (exp 𝕂 (0 + h) - exp 𝕂 0 - ContinuousLinearMap.id 𝕂 𝔸 h)) =ᶠ[𝓝 0] fun h =>
      exp 𝕂 (x + h) - exp 𝕂 x - exp 𝕂 x • ContinuousLinearMap.id 𝕂 𝔸 h by
    refine (IsLittleO.const_mul_left ?_ _).congr' this (EventuallyEq.refl _ _)
    rw [← hasFDerivAt_iff_isLittleO_nhds_zero]
    exact hasFDerivAt_exp_zero_of_radius_pos hpos
  have : ∀ᶠ h in 𝓝 (0 : 𝔸), h ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius :=
    EMetric.ball_mem_nhds _ hpos
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : NormedCommRing 𝔸
    inst✝² : NormedAlgebra 𝕂 𝔸
    inst✝¹ : CompleteSpace 𝔸
    inst✝ : CharZero 𝕂
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    hpos : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    this : Filter.Eventually (fun h => Membership.mem (EMetric.ball 0 (NormedSpace …
    ⊢ (nhds 0).EventuallyEq (fun h => HMul.hMul (NormedSpace.exp 𝕂 x) (HSub.hSub ( …
  -/
  filter_upwards [this] with _ hh
  /-
    case h
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : NormedCommRing 𝔸
    inst✝² : NormedAlgebra 𝕂 𝔸
    inst✝¹ : CompleteSpace 𝔸
    inst✝ : CharZero 𝕂
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    hpos : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    this : Filter.Eventually (fun h => Membership.mem (EMetric.ball 0 (NormedSpace …
    a✝ : 𝔸
    hh : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) a✝
    ⊢ Eq (HMul.hMul (NormedSpace.exp 𝕂 x) (HSub.hSub (HSub.hSub (NormedSpace.exp 𝕂 …
  -/
  rw [exp_add_of_mem_ball hx hh, exp_zero, zero_add, ContinuousLinearMap.id_apply, smul_eq_mul]
  /-
    case h
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : NormedCommRing 𝔸
    inst✝² : NormedAlgebra 𝕂 𝔸
    inst✝¹ : CompleteSpace 𝔸
    inst✝ : CharZero 𝕂
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    hpos : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    this : Filter.Eventually (fun h => Membership.mem (EMetric.ball 0 (NormedSpace …
    a✝ : 𝔸
    hh : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) a✝
    ⊢ Eq (HMul.hMul (NormedSpace.exp 𝕂 x) (HSub.hSub (HSub.hSub (NormedSpace.exp 𝕂 …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The exponential map in a commutative Banach algebra `𝔸` over a normed field `𝕂` of
characteristic zero has strict Fréchet derivative `NormedSpace.exp 𝕂 x • 1 : 𝔸 →L[𝕂] 𝔸`
at any point `x` in the disk of convergence. -/
theorem hasStrictFDerivAt_exp_of_mem_ball [CharZero 𝕂] {x : 𝔸}
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasStrictFDerivAt (exp 𝕂) (exp 𝕂 x • (1 : 𝔸 →L[𝕂] 𝔸)) x :=
  let ⟨_, hp⟩ := analyticAt_exp_of_mem_ball x hx
  hp.hasFDerivAt.unique (hasFDerivAt_exp_of_mem_ball hx) ▸ hp.hasStrictFDerivAt


/-- The exponential map in a complete normed field `𝕂` of characteristic zero has strict derivative
`NormedSpace.exp 𝕂 x` at any point `x` in the disk of convergence. -/
theorem hasStrictDerivAt_exp_of_mem_ball [CharZero 𝕂] {x : 𝕂}
    (hx : x ∈ EMetric.ball (0 : 𝕂) (expSeries 𝕂 𝕂).radius) :
    HasStrictDerivAt (exp 𝕂) (exp 𝕂 x) x := by
  /-
    𝕂 : Type u_1
    inst✝² : NontriviallyNormedField 𝕂
    inst✝¹ : CompleteSpace 𝕂
    inst✝ : CharZero 𝕂
    x : 𝕂
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝕂).radius) x
    ⊢ HasStrictDerivAt (NormedSpace.exp 𝕂) (NormedSpace.exp 𝕂 x) x
  -/
  simpa using (hasStrictFDerivAt_exp_of_mem_ball hx).hasStrictDerivAt
  /-
    🎉 no goals
  -/


/-- The exponential map in a complete normed field `𝕂` of characteristic zero has derivative
`NormedSpace.exp 𝕂 x` at any point `x` in the disk of convergence. -/
theorem hasDerivAt_exp_of_mem_ball [CharZero 𝕂] {x : 𝕂}
    (hx : x ∈ EMetric.ball (0 : 𝕂) (expSeries 𝕂 𝕂).radius) : HasDerivAt (exp 𝕂) (exp 𝕂 x) x :=
  (hasStrictDerivAt_exp_of_mem_ball hx).hasDerivAt


/-- The exponential map in a complete normed field `𝕂` of characteristic zero has strict derivative
`1` at zero, as long as it converges on a neighborhood of zero. -/
theorem hasStrictDerivAt_exp_zero_of_radius_pos (h : 0 < (expSeries 𝕂 𝕂).radius) :
    HasStrictDerivAt (exp 𝕂) (1 : 𝕂) 0 :=
  (hasStrictFDerivAt_exp_zero_of_radius_pos h).hasStrictDerivAt


/-- The exponential map in a complete normed field `𝕂` of characteristic zero has derivative
`1` at zero, as long as it converges on a neighborhood of zero. -/
theorem hasDerivAt_exp_zero_of_radius_pos (h : 0 < (expSeries 𝕂 𝕂).radius) :
    HasDerivAt (exp 𝕂) (1 : 𝕂) 0 :=
  (hasStrictDerivAt_exp_zero_of_radius_pos h).hasDerivAt


/-- The exponential in a Banach algebra `𝔸` over `𝕂 = ℝ` or `𝕂 = ℂ` has strict Fréchet derivative
`1 : 𝔸 →L[𝕂] 𝔸` at zero. -/
theorem hasStrictFDerivAt_exp_zero : HasStrictFDerivAt (exp 𝕂) (1 : 𝔸 →L[𝕂] 𝔸) 0 :=
  hasStrictFDerivAt_exp_zero_of_radius_pos (expSeries_radius_pos 𝕂 𝔸)


/-- The exponential in a Banach algebra `𝔸` over `𝕂 = ℝ` or `𝕂 = ℂ` has Fréchet derivative
`1 : 𝔸 →L[𝕂] 𝔸` at zero. -/
theorem hasFDerivAt_exp_zero : HasFDerivAt (exp 𝕂) (1 : 𝔸 →L[𝕂] 𝔸) 0 :=
  hasStrictFDerivAt_exp_zero.hasFDerivAt


/-- The exponential map in a commutative Banach algebra `𝔸` over `𝕂 = ℝ` or `𝕂 = ℂ` has strict
Fréchet derivative `NormedSpace.exp 𝕂 x • 1 : 𝔸 →L[𝕂] 𝔸` at any point `x`. -/
theorem hasStrictFDerivAt_exp {x : 𝔸} : HasStrictFDerivAt (exp 𝕂) (exp 𝕂 x • (1 : 𝔸 →L[𝕂] 𝔸)) x :=
  hasStrictFDerivAt_exp_of_mem_ball ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)


/-- The exponential map in a commutative Banach algebra `𝔸` over `𝕂 = ℝ` or `𝕂 = ℂ` has
Fréchet derivative `NormedSpace.exp 𝕂 x • 1 : 𝔸 →L[𝕂] 𝔸` at any point `x`. -/
theorem hasFDerivAt_exp {x : 𝔸} : HasFDerivAt (exp 𝕂) (exp 𝕂 x • (1 : 𝔸 →L[𝕂] 𝔸)) x :=
  hasStrictFDerivAt_exp.hasFDerivAt


/-- The exponential map in `𝕂 = ℝ` or `𝕂 = ℂ` has strict derivative `NormedSpace.exp 𝕂 x`
at any point `x`. -/
theorem hasStrictDerivAt_exp {x : 𝕂} : HasStrictDerivAt (exp 𝕂) (exp 𝕂 x) x :=
  hasStrictDerivAt_exp_of_mem_ball ((expSeries_radius_eq_top 𝕂 𝕂).symm ▸ edist_lt_top _ _)


/-- The exponential map in `𝕂 = ℝ` or `𝕂 = ℂ` has derivative `NormedSpace.exp 𝕂 x`
at any point `x`. -/
theorem hasDerivAt_exp {x : 𝕂} : HasDerivAt (exp 𝕂) (exp 𝕂 x) x :=
  hasStrictDerivAt_exp.hasDerivAt


/-- The exponential map in `𝕂 = ℝ` or `𝕂 = ℂ` has strict derivative `1` at zero. -/
theorem hasStrictDerivAt_exp_zero : HasStrictDerivAt (exp 𝕂) (1 : 𝕂) 0 :=
  hasStrictDerivAt_exp_zero_of_radius_pos (expSeries_radius_pos 𝕂 𝕂)


/-- The exponential map in `𝕂 = ℝ` or `𝕂 = ℂ` has derivative `1` at zero. -/
theorem hasDerivAt_exp_zero : HasDerivAt (exp 𝕂) (1 : 𝕂) 0 :=
  hasStrictDerivAt_exp_zero.hasDerivAt


theorem Complex.exp_eq_exp_ℂ : Complex.exp = NormedSpace.exp ℂ := by
  /-
    ⊢ Eq Complex.exp (NormedSpace.exp Complex)
  -/
  refine funext fun x => ?_
  /-
    x : Complex
    ⊢ Eq (Complex.exp x) (NormedSpace.exp Complex x)
  -/
  rw [Complex.exp, exp_eq_tsum_div]
  /-
    x : Complex
    ⊢ Eq (Complex.exp' x).lim ((fun x => tsum fun n => HDiv.hDiv (HPow.hPow x n) ↑ …
  -/
  have : CauSeq.IsComplete ℂ norm := Complex.instIsComplete
  /-
    x : Complex
    this : CauSeq.IsComplete Complex Norm.norm
    ⊢ Eq (Complex.exp' x).lim ((fun x => tsum fun n => HDiv.hDiv (HPow.hPow x n) ↑ …
  -/
  exact tendsto_nhds_unique x.exp'.tendsto_limit (expSeries_div_summable ℝ x).hasSum.tendsto_sum_nat
  /-
    🎉 no goals
  -/


theorem Real.exp_eq_exp_ℝ : Real.exp = NormedSpace.exp ℝ := by
  /-
    ⊢ Eq Real.exp (NormedSpace.exp Real)
  -/
  ext x; exact mod_cast congr_fun Complex.exp_eq_exp_ℂ x
         /-
           🎉 no goals
         -/


theorem hasFDerivAt_exp_smul_const_of_mem_ball (x : 𝔸) (t : 𝕊)
    (htx : t • x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasFDerivAt (fun u : 𝕊 => exp 𝕂 (u • x)) (exp 𝕂 (t • x) • (1 : 𝕊 →L[𝕂] 𝕊).smulRight x) t := by
  -- TODO: prove this via `hasFDerivAt_exp_of_mem_ball` using the commutative ring
  -- `Algebra.elementalAlgebra 𝕊 x`. See https://github.com/leanprover-community/mathlib3/pull/19062 for discussion.
  /-
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    ⊢ HasFDerivAt (fun u => NormedSpace.exp 𝕂 (HSMul.hSMul u x)) (HSMul.hSMul (Nor …
  -/
  have hpos : 0 < (expSeries 𝕂 𝔸).radius := (zero_le _).trans_lt htx
  /-
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    hpos : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    ⊢ HasFDerivAt (fun u => NormedSpace.exp 𝕂 (HSMul.hSMul u x)) (HSMul.hSMul (Nor …
  -/
  rw [hasFDerivAt_iff_isLittleO_nhds_zero]
  suffices (fun (h : 𝕊) => exp 𝕂 (t • x) *
      (exp 𝕂 ((0 + h) • x) - exp 𝕂 ((0 : 𝕊) • x) - ((1 : 𝕊 →L[𝕂] 𝕊).smulRight x) h)) =ᶠ[𝓝 0]
        fun h =>
          exp 𝕂 ((t + h) • x) - exp 𝕂 (t • x) - (exp 𝕂 (t • x) • (1 : 𝕊 →L[𝕂] 𝕊).smulRight x) h by
    apply (IsLittleO.const_mul_left _ _).congr' this (EventuallyEq.refl _ _)
    rw [← hasFDerivAt_iff_isLittleO_nhds_zero (f := fun u => exp 𝕂 (u • x))
      (f' := (1 : 𝕊 →L[𝕂] 𝕊).smulRight x) (x := 0)]
    have : HasFDerivAt (exp 𝕂) (1 : 𝔸 →L[𝕂] 𝔸) ((1 : 𝕊 →L[𝕂] 𝕊).smulRight x 0) := by
      rw [ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.one_apply, zero_smul]
      exact hasFDerivAt_exp_zero_of_radius_pos hpos
    exact this.comp 0 ((1 : 𝕊 →L[𝕂] 𝕊).smulRight x).hasFDerivAt
  have : Tendsto (fun h : 𝕊 => h • x) (𝓝 0) (𝓝 0) := by
    rw [← zero_smul 𝕊 x]
    exact tendsto_id.smul_const x
  have : ∀ᶠ h in 𝓝 (0 : 𝕊), h • x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius :=
    this.eventually (EMetric.ball_mem_nhds _ hpos)
  /-
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    hpos : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    this✝ : Filter.Tendsto (fun h => HSMul.hSMul h x) (nhds 0) (nhds 0)
    this : Filter.Eventually (fun h => Membership.mem (EMetric.ball 0 (NormedSpace …
    ⊢ (nhds 0).EventuallyEq (fun h => HMul.hMul (NormedSpace.exp 𝕂 (HSMul.hSMul t  …
  -/
  filter_upwards [this] with h hh
  /-
    case h
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    hpos : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
    this✝ : Filter.Tendsto (fun h => HSMul.hSMul h x) (nhds 0) (nhds 0)
    this : Filter.Eventually (fun h => Membership.mem (EMetric.ball 0 (NormedSpace …
    h : 𝕊
    hh : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMul …
    ⊢ Eq (HMul.hMul (NormedSpace.exp 𝕂 (HSMul.hSMul t x)) (HSub.hSub (HSub.hSub (N …
  -/
  have : Commute (t • x) (h • x) := ((Commute.refl x).smul_left t).smul_right h
  rw [add_smul t h, exp_add_of_commute_of_mem_ball this htx hh, zero_add, zero_smul, exp_zero,
    ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.one_apply,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.smulRight_apply,
    ContinuousLinearMap.one_apply, smul_eq_mul, mul_sub_left_distrib, mul_sub_left_distrib, mul_one]


theorem hasFDerivAt_exp_smul_const_of_mem_ball' (x : 𝔸) (t : 𝕊)
    (htx : t • x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasFDerivAt (fun u : 𝕊 => exp 𝕂 (u • x))
      (((1 : 𝕊 →L[𝕂] 𝕊).smulRight x).smulRight (exp 𝕂 (t • x))) t := by
  /-
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    ⊢ HasFDerivAt (fun u => NormedSpace.exp 𝕂 (HSMul.hSMul u x)) ((ContinuousLinea …
  -/
  convert hasFDerivAt_exp_smul_const_of_mem_ball 𝕂 _ _ htx using 1
  /-
    case h.e'_12.h.h
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    e_8✝ : Eq NormedAddCommGroup.toAddCommGroup SeminormedAddCommGroup.toAddCommGr …
    ⊢ Eq ((ContinuousLinearMap.smulRight 1 x).smulRight (NormedSpace.exp 𝕂 (HSMul. …
  -/
  ext t'
  /-
    case h.e'_12.h.h.h
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    e_8✝ : Eq NormedAddCommGroup.toAddCommGroup SeminormedAddCommGroup.toAddCommGr …
    t' : 𝕊
    ⊢ Eq (((ContinuousLinearMap.smulRight 1 x).smulRight (NormedSpace.exp 𝕂 (HSMul …
  -/
  show Commute (t' • x) (exp 𝕂 (t • x))
  /-
    case h.e'_12.h.h.h
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    e_8✝ : Eq NormedAddCommGroup.toAddCommGroup SeminormedAddCommGroup.toAddCommGr …
    t' : 𝕊
    ⊢ Commute (HSMul.hSMul t' x) (NormedSpace.exp 𝕂 (HSMul.hSMul t x))
  -/
  exact (((Commute.refl x).smul_left t').smul_right t).exp_right 𝕂
  /-
    🎉 no goals
  -/


theorem hasStrictFDerivAt_exp_smul_const_of_mem_ball (x : 𝔸) (t : 𝕊)
    (htx : t • x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasStrictFDerivAt (fun u : 𝕊 => exp 𝕂 (u • x))
      (exp 𝕂 (t • x) • (1 : 𝕊 →L[𝕂] 𝕊).smulRight x) t :=
  let ⟨_, hp⟩ := analyticAt_exp_of_mem_ball (t • x) htx
  have deriv₁ : HasStrictFDerivAt (fun u : 𝕊 => exp 𝕂 (u • x)) _ t :=
    hp.hasStrictFDerivAt.comp t ((ContinuousLinearMap.id 𝕂 𝕊).smulRight x).hasStrictFDerivAt
  have deriv₂ : HasFDerivAt (fun u : 𝕊 => exp 𝕂 (u • x)) _ t :=
    hasFDerivAt_exp_smul_const_of_mem_ball 𝕂 x t htx
  deriv₁.hasFDerivAt.unique deriv₂ ▸ deriv₁


theorem hasStrictFDerivAt_exp_smul_const_of_mem_ball' (x : 𝔸) (t : 𝕊)
    (htx : t • x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasStrictFDerivAt (fun u : 𝕊 => exp 𝕂 (u • x))
      (((1 : 𝕊 →L[𝕂] 𝕊).smulRight x).smulRight (exp 𝕂 (t • x))) t := by
  /-
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    ⊢ HasStrictFDerivAt (fun u => NormedSpace.exp 𝕂 (HSMul.hSMul u x)) ((Continuou …
  -/
  let ⟨_, _⟩ := analyticAt_exp_of_mem_ball (t • x) htx
  /-
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    w✝ : FormalMultilinearSeries 𝕂 𝔸 𝔸
    h✝ : HasFPowerSeriesAt (NormedSpace.exp 𝕂) w✝ (HSMul.hSMul t x)
    ⊢ HasStrictFDerivAt (fun u => NormedSpace.exp 𝕂 (HSMul.hSMul u x)) ((Continuou …
  -/
  convert hasStrictFDerivAt_exp_smul_const_of_mem_ball 𝕂 _ _ htx using 1
  /-
    case h.e'_12.h.h
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    w✝ : FormalMultilinearSeries 𝕂 𝔸 𝔸
    h✝ : HasFPowerSeriesAt (NormedSpace.exp 𝕂) w✝ (HSMul.hSMul t x)
    e_8✝ : Eq NormedAddCommGroup.toAddCommGroup SeminormedAddCommGroup.toAddCommGr …
    ⊢ Eq ((ContinuousLinearMap.smulRight 1 x).smulRight (NormedSpace.exp 𝕂 (HSMul. …
  -/
  ext t'
  /-
    case h.e'_12.h.h.h
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    w✝ : FormalMultilinearSeries 𝕂 𝔸 𝔸
    h✝ : HasFPowerSeriesAt (NormedSpace.exp 𝕂) w✝ (HSMul.hSMul t x)
    e_8✝ : Eq NormedAddCommGroup.toAddCommGroup SeminormedAddCommGroup.toAddCommGr …
    t' : 𝕊
    ⊢ Eq (((ContinuousLinearMap.smulRight 1 x).smulRight (NormedSpace.exp 𝕂 (HSMul …
  -/
  show Commute (t' • x) (exp 𝕂 (t • x))
  /-
    case h.e'_12.h.h.h
    𝕂 : Type u_1
    𝕊 : Type u_2
    𝔸 : Type u_3
    inst✝⁹ : NontriviallyNormedField 𝕂
    inst✝⁸ : CharZero 𝕂
    inst✝⁷ : NormedCommRing 𝕊
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedSpace 𝕂 𝕊
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : Algebra 𝕊 𝔸
    inst✝² : ContinuousSMul 𝕊 𝔸
    inst✝¹ : IsScalarTower 𝕂 𝕊 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕊
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    w✝ : FormalMultilinearSeries 𝕂 𝔸 𝔸
    h✝ : HasFPowerSeriesAt (NormedSpace.exp 𝕂) w✝ (HSMul.hSMul t x)
    e_8✝ : Eq NormedAddCommGroup.toAddCommGroup SeminormedAddCommGroup.toAddCommGr …
    t' : 𝕊
    ⊢ Commute (HSMul.hSMul t' x) (NormedSpace.exp 𝕂 (HSMul.hSMul t x))
  -/
  exact (((Commute.refl x).smul_left t').smul_right t).exp_right 𝕂
  /-
    🎉 no goals
  -/


theorem hasStrictDerivAt_exp_smul_const_of_mem_ball (x : 𝔸) (t : 𝕂)
    (htx : t • x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasStrictDerivAt (fun u : 𝕂 => exp 𝕂 (u • x)) (exp 𝕂 (t • x) * x) t := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : CharZero 𝕂
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕂
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    ⊢ HasStrictDerivAt (fun u => NormedSpace.exp 𝕂 (HSMul.hSMul u x)) (HMul.hMul ( …
  -/
  simpa using (hasStrictFDerivAt_exp_smul_const_of_mem_ball 𝕂 x t htx).hasStrictDerivAt
  /-
    🎉 no goals
  -/


theorem hasStrictDerivAt_exp_smul_const_of_mem_ball' (x : 𝔸) (t : 𝕂)
    (htx : t • x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasStrictDerivAt (fun u : 𝕂 => exp 𝕂 (u • x)) (x * exp 𝕂 (t • x)) t := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : CharZero 𝕂
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    t : 𝕂
    htx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) (HSMu …
    ⊢ HasStrictDerivAt (fun u => NormedSpace.exp 𝕂 (HSMul.hSMul u x)) (HMul.hMul x …
  -/
  simpa using (hasStrictFDerivAt_exp_smul_const_of_mem_ball' 𝕂 x t htx).hasStrictDerivAt
  /-
    🎉 no goals
  -/


theorem hasDerivAt_exp_smul_const_of_mem_ball (x : 𝔸) (t : 𝕂)
    (htx : t • x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasDerivAt (fun u : 𝕂 => exp 𝕂 (u • x)) (exp 𝕂 (t • x) * x) t :=
  (hasStrictDerivAt_exp_smul_const_of_mem_ball x t htx).hasDerivAt


theorem hasDerivAt_exp_smul_const_of_mem_ball' (x : 𝔸) (t : 𝕂)
    (htx : t • x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasDerivAt (fun u : 𝕂 => exp 𝕂 (u • x)) (x * exp 𝕂 (t • x)) t :=
  (hasStrictDerivAt_exp_smul_const_of_mem_ball' x t htx).hasDerivAt


theorem hasFDerivAt_exp_smul_const (x : 𝔸) (t : 𝕊) :
    HasFDerivAt (fun u : 𝕊 => exp 𝕂 (u • x)) (exp 𝕂 (t • x) • (1 : 𝕊 →L[𝕂] 𝕊).smulRight x) t :=
  hasFDerivAt_exp_smul_const_of_mem_ball 𝕂 _ _ <|
    (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem hasFDerivAt_exp_smul_const' (x : 𝔸) (t : 𝕊) :
    HasFDerivAt (fun u : 𝕊 => exp 𝕂 (u • x))
      (((1 : 𝕊 →L[𝕂] 𝕊).smulRight x).smulRight (exp 𝕂 (t • x))) t :=
  hasFDerivAt_exp_smul_const_of_mem_ball' 𝕂 _ _ <|
    (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem hasStrictFDerivAt_exp_smul_const (x : 𝔸) (t : 𝕊) :
    HasStrictFDerivAt (fun u : 𝕊 => exp 𝕂 (u • x))
      (exp 𝕂 (t • x) • (1 : 𝕊 →L[𝕂] 𝕊).smulRight x) t :=
  hasStrictFDerivAt_exp_smul_const_of_mem_ball 𝕂 _ _ <|
    (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem hasStrictFDerivAt_exp_smul_const' (x : 𝔸) (t : 𝕊) :
    HasStrictFDerivAt (fun u : 𝕊 => exp 𝕂 (u • x))
      (((1 : 𝕊 →L[𝕂] 𝕊).smulRight x).smulRight (exp 𝕂 (t • x))) t :=
  hasStrictFDerivAt_exp_smul_const_of_mem_ball' 𝕂 _ _ <|
    (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem hasStrictDerivAt_exp_smul_const (x : 𝔸) (t : 𝕂) :
    HasStrictDerivAt (fun u : 𝕂 => exp 𝕂 (u • x)) (exp 𝕂 (t • x) * x) t :=
  hasStrictDerivAt_exp_smul_const_of_mem_ball _ _ <|
    (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem hasStrictDerivAt_exp_smul_const' (x : 𝔸) (t : 𝕂) :
    HasStrictDerivAt (fun u : 𝕂 => exp 𝕂 (u • x)) (x * exp 𝕂 (t • x)) t :=
  hasStrictDerivAt_exp_smul_const_of_mem_ball' _ _ <|
    (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem hasDerivAt_exp_smul_const (x : 𝔸) (t : 𝕂) :
    HasDerivAt (fun u : 𝕂 => exp 𝕂 (u • x)) (exp 𝕂 (t • x) * x) t :=
  hasDerivAt_exp_smul_const_of_mem_ball _ _ <| (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem hasDerivAt_exp_smul_const' (x : 𝔸) (t : 𝕂) :
    HasDerivAt (fun u : 𝕂 => exp 𝕂 (u • x)) (x * exp 𝕂 (t • x)) t :=
  hasDerivAt_exp_smul_const_of_mem_ball' _ _ <|
    (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


/-- If `f` has sum `a`, then `NormedSpace.exp ∘ f` has product `NormedSpace.exp a`. -/
lemma HasSum.exp {ι : Type*} {f : ι → 𝔸} {a : 𝔸} (h : HasSum f a) :
    HasProd (exp 𝕂 ∘ f) (exp 𝕂 a) :=
  Tendsto.congr (fun s ↦ exp_sum s f) <| Tendsto.exp h


