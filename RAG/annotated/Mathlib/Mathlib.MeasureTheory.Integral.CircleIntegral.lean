/-- The exponential map $θ ↦ c + R e^{θi}$. The range of this map is the circle in `ℂ` with center
`c` and radius `|R|`. -/
def circleMap (c : ℂ) (R : ℝ) : ℝ → ℂ := fun θ => c + R * exp (θ * I)


/-- `circleMap` is `2π`-periodic. -/
theorem periodic_circleMap (c : ℂ) (R : ℝ) : Periodic (circleMap c R) (2 * π) := fun θ => by
  /-
    c : Complex
    R θ : Real
    ⊢ Eq (circleMap c R (HAdd.hAdd θ (HMul.hMul 2 Real.pi))) (circleMap c R θ)
  -/
  simp [circleMap, add_mul, exp_periodic _]
  /-
    🎉 no goals
  -/


theorem Set.Countable.preimage_circleMap {s : Set ℂ} (hs : s.Countable) (c : ℂ) {R : ℝ}
    (hR : R ≠ 0) : (circleMap c R ⁻¹' s).Countable :=
  show (((↑) : ℝ → ℂ) ⁻¹' ((· * I) ⁻¹'
      (exp ⁻¹' ((R * ·) ⁻¹' ((c + ·) ⁻¹' s))))).Countable from
    (((hs.preimage (add_right_injective _)).preimage <|
      mul_right_injective₀ <| ofReal_ne_zero.2 hR).preimage_cexp.preimage <|
        mul_left_injective₀ I_ne_zero).preimage ofReal_injective


@[simp]
theorem circleMap_sub_center (c : ℂ) (R : ℝ) (θ : ℝ) : circleMap c R θ - c = circleMap 0 R θ := by
  /-
    c : Complex
    R θ : Real
    ⊢ Eq (HSub.hSub (circleMap c R θ) c) (circleMap 0 R θ)
  -/
  simp [circleMap]
  /-
    🎉 no goals
  -/


theorem circleMap_zero (R θ : ℝ) : circleMap 0 R θ = R * exp (θ * I) :=
  zero_add _


@[simp]
                                                                               /-
                                                                                 R θ : Real
                                                                                 ⊢ Eq (Complex.abs (circleMap 0 R θ)) (abs R)
                                                                               -/
theorem abs_circleMap_zero (R : ℝ) (θ : ℝ) : abs (circleMap 0 R θ) = |R| := by simp [circleMap]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


                                                                                             /-
                                                                                               c : Complex
                                                                                               R θ : Real
                                                                                               ⊢ Membership.mem (Metric.sphere c (abs R)) (circleMap c R θ)
                                                                                             -/
theorem circleMap_mem_sphere' (c : ℂ) (R : ℝ) (θ : ℝ) : circleMap c R θ ∈ sphere c |R| := by simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


theorem circleMap_mem_sphere (c : ℂ) {R : ℝ} (hR : 0 ≤ R) (θ : ℝ) :
    circleMap c R θ ∈ sphere c R := by
  /-
    c : Complex
    R : Real
    hR : LE.le 0 R
    θ : Real
    ⊢ Membership.mem (Metric.sphere c R) (circleMap c R θ)
  -/
  simpa only [_root_.abs_of_nonneg hR] using circleMap_mem_sphere' c R θ
  /-
    🎉 no goals
  -/


theorem circleMap_mem_closedBall (c : ℂ) {R : ℝ} (hR : 0 ≤ R) (θ : ℝ) :
    circleMap c R θ ∈ closedBall c R :=
  sphere_subset_closedBall (circleMap_mem_sphere c hR θ)


theorem circleMap_not_mem_ball (c : ℂ) (R : ℝ) (θ : ℝ) : circleMap c R θ ∉ ball c R := by
  /-
    c : Complex
    R θ : Real
    ⊢ Not (Membership.mem (Metric.ball c R) (circleMap c R θ))
  -/
  simp [dist_eq, le_abs_self]
  /-
    🎉 no goals
  -/


theorem circleMap_ne_mem_ball {c : ℂ} {R : ℝ} {w : ℂ} (hw : w ∈ ball c R) (θ : ℝ) :
    circleMap c R θ ≠ w :=
  (ne_of_mem_of_not_mem hw (circleMap_not_mem_ball _ _ _)).symm


/-- The range of `circleMap c R` is the circle with center `c` and radius `|R|`. -/
@[simp]
theorem range_circleMap (c : ℂ) (R : ℝ) : range (circleMap c R) = sphere c |R| :=
  calc
    range (circleMap c R) = c +ᵥ R • range fun θ : ℝ => exp (θ * I) := by
      simp (config := { unfoldPartialApp := true }) only [← image_vadd, ← image_smul, ← range_comp,
        vadd_eq_add, circleMap, Function.comp_def, real_smul]
    _ = sphere c |R| := by
      /-
        c : Complex
        R : Real
        ⊢ Eq (HVAdd.hVAdd c (HSMul.hSMul R (Set.range fun θ => Complex.exp (HMul.hMul  …
      -/
      rw [Complex.range_exp_mul_I, smul_sphere R 0 zero_le_one]
      /-
        c : Complex
        R : Real
        ⊢ Eq (HVAdd.hVAdd c (Metric.sphere (HSMul.hSMul R 0) (HMul.hMul (Norm.norm R)  …
      -/
      simp
      /-
        🎉 no goals
      -/


/-- The image of `(0, 2π]` under `circleMap c R` is the circle with center `c` and radius `|R|`. -/
@[simp]
theorem image_circleMap_Ioc (c : ℂ) (R : ℝ) : circleMap c R '' Ioc 0 (2 * π) = sphere c |R| := by
  /-
    c : Complex
    R : Real
    ⊢ Eq (Set.image (circleMap c R) (Set.Ioc 0 (HMul.hMul 2 Real.pi))) (Metric.sph …
  -/
  rw [← range_circleMap, ← (periodic_circleMap c R).image_Ioc Real.two_pi_pos 0, zero_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem circleMap_eq_center_iff {c : ℂ} {R : ℝ} {θ : ℝ} : circleMap c R θ = c ↔ R = 0 := by
  /-
    c : Complex
    R θ : Real
    ⊢ Iff (Eq (circleMap c R θ) c) (Eq R 0)
  -/
  simp [circleMap, exp_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem circleMap_zero_radius (c : ℂ) : circleMap c 0 = const ℝ c :=
  funext fun _ => circleMap_eq_center_iff.2 rfl


theorem circleMap_ne_center {c : ℂ} {R : ℝ} (hR : R ≠ 0) {θ : ℝ} : circleMap c R θ ≠ c :=
  mt circleMap_eq_center_iff.1 hR


theorem hasDerivAt_circleMap (c : ℂ) (R : ℝ) (θ : ℝ) :
    HasDerivAt (circleMap c R) (circleMap 0 R θ * I) θ := by
  simpa only [mul_assoc, one_mul, ofRealCLM_apply, circleMap, ofReal_one, zero_add]
    using (((ofRealCLM.hasDerivAt (x := θ)).mul_const I).cexp.const_mul (R : ℂ)).const_add c

/- TODO: prove `ContDiff ℝ (circleMap c R)`. This needs a version of `ContDiff.mul`
for multiplication in a normed algebra over the base field. -/

theorem differentiable_circleMap (c : ℂ) (R : ℝ) : Differentiable ℝ (circleMap c R) := fun θ =>
  (hasDerivAt_circleMap c R θ).differentiableAt


@[continuity, fun_prop]
theorem continuous_circleMap (c : ℂ) (R : ℝ) : Continuous (circleMap c R) :=
  (differentiable_circleMap c R).continuous


@[fun_prop, measurability]
theorem measurable_circleMap (c : ℂ) (R : ℝ) : Measurable (circleMap c R) :=
  (continuous_circleMap c R).measurable


@[simp]
theorem deriv_circleMap (c : ℂ) (R : ℝ) (θ : ℝ) : deriv (circleMap c R) θ = circleMap 0 R θ * I :=
  (hasDerivAt_circleMap _ _ _).deriv


theorem deriv_circleMap_eq_zero_iff {c : ℂ} {R : ℝ} {θ : ℝ} :
                                              /-
                                                c : Complex
                                                R θ : Real
                                                ⊢ Iff (Eq (deriv (circleMap c R) θ) 0) (Eq R 0)
                                              -/
    deriv (circleMap c R) θ = 0 ↔ R = 0 := by simp [I_ne_zero]
                                              /-
                                                🎉 no goals
                                              -/


theorem deriv_circleMap_ne_zero {c : ℂ} {R : ℝ} {θ : ℝ} (hR : R ≠ 0) :
    deriv (circleMap c R) θ ≠ 0 :=
  mt deriv_circleMap_eq_zero_iff.1 hR


theorem lipschitzWith_circleMap (c : ℂ) (R : ℝ) : LipschitzWith (Real.nnabs R) (circleMap c R) :=
  lipschitzWith_of_nnnorm_deriv_le (differentiable_circleMap _ _) fun θ =>
                              /-
                                c : Complex
                                R θ : Real
                                ⊢ LE.le ↑(NNNorm.nnnorm (deriv (circleMap c R) θ)) ↑(Real.nnabs R)
                              -/
    NNReal.coe_le_coe.1 <| by simp
                              /-
                                🎉 no goals
                              -/


theorem continuous_circleMap_inv {R : ℝ} {z w : ℂ} (hw : w ∈ ball z R) :
    Continuous fun θ => (circleMap z R θ - w)⁻¹ := by
  have : ∀ θ, circleMap z R θ - w ≠ 0 := by
    simp_rw [sub_ne_zero]
    exact fun θ => circleMap_ne_mem_ball hw θ
  -- Porting note: was `continuity`
  /-
    R : Real
    z w : Complex
    hw : Membership.mem (Metric.ball z R) w
    this : ∀ (θ : Real), Ne (HSub.hSub (circleMap z R θ) w) 0
    ⊢ Continuous fun θ => Inv.inv (HSub.hSub (circleMap z R θ) w)
  -/
  exact Continuous.inv₀ (by fun_prop) this
  /-
    🎉 no goals
  -/


/-- We say that a function `f : ℂ → E` is integrable on the circle with center `c` and radius `R` if
the function `f ∘ circleMap c R` is integrable on `[0, 2π]`.

Note that the actual function used in the definition of `circleIntegral` is
`(deriv (circleMap c R) θ) • f (circleMap c R θ)`. Integrability of this function is equivalent
to integrability of `f ∘ circleMap c R` whenever `R ≠ 0`. -/
def CircleIntegrable (f : ℂ → E) (c : ℂ) (R : ℝ) : Prop :=
  IntervalIntegrable (fun θ : ℝ => f (circleMap c R θ)) volume 0 (2 * π)


@[simp]
theorem circleIntegrable_const (a : E) (c : ℂ) (R : ℝ) : CircleIntegrable (fun _ => a) c R :=
  intervalIntegrable_const


nonrec theorem add (hf : CircleIntegrable f c R) (hg : CircleIntegrable g c R) :
    CircleIntegrable (f + g) c R :=
  hf.add hg


nonrec theorem neg (hf : CircleIntegrable f c R) : CircleIntegrable (-f) c R :=
  hf.neg


/-- The function we actually integrate over `[0, 2π]` in the definition of `circleIntegral` is
integrable. -/
theorem out [NormedSpace ℂ E] (hf : CircleIntegrable f c R) :
    IntervalIntegrable (fun θ : ℝ => deriv (circleMap c R) θ • f (circleMap c R θ)) volume 0
      (2 * π) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    f : Complex → E
    c : Complex
    R : Real
    inst✝ : NormedSpace Complex E
    hf : CircleIntegrable f c R
    ⊢ IntervalIntegrable (fun θ => HSMul.hSMul (deriv (circleMap c R) θ) (f (circl …
  -/
  simp only [CircleIntegrable, deriv_circleMap, intervalIntegrable_iff] at *
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    f : Complex → E
    c : Complex
    R : Real
    inst✝ : NormedSpace Complex E
    hf : MeasureTheory.IntegrableOn (fun θ => f (circleMap c R θ)) (Set.uIoc 0 (HM …
    ⊢ MeasureTheory.IntegrableOn (fun θ => HSMul.hSMul (HMul.hMul (circleMap 0 R θ …
  -/
  refine (hf.norm.const_mul |R|).mono' ?_ ?_
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      f : Complex → E
      c : Complex
      R : Real
      inst✝ : NormedSpace Complex E
      hf : MeasureTheory.IntegrableOn (fun θ => f (circleMap c R θ)) (Set.uIoc 0 (HM …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun θ => HSMul.hSMul (HMul.hMul (circleM …
    -/
  · exact ((continuous_circleMap _ _).aestronglyMeasurable.mul_const I).smul hf.aestronglyMeasurable
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      f : Complex → E
      c : Complex
      R : Real
      inst✝ : NormedSpace Complex E
      hf : MeasureTheory.IntegrableOn (fun θ => f (circleMap c R θ)) (Set.uIoc 0 (HM …
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (HSMul.hSMul (HMul.hMul (circle …
    -/
  · simp [norm_smul]
    /-
      🎉 no goals
    -/


@[simp]
theorem circleIntegrable_zero_radius {f : ℂ → E} {c : ℂ} : CircleIntegrable f c 0 := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    c : Complex
    ⊢ CircleIntegrable f c 0
  -/
  simp [CircleIntegrable]
  /-
    🎉 no goals
  -/


theorem circleIntegrable_iff [NormedSpace ℂ E] {f : ℂ → E} {c : ℂ} (R : ℝ) :
    CircleIntegrable f c R ↔ IntervalIntegrable (fun θ : ℝ =>
      deriv (circleMap c R) θ • f (circleMap c R θ)) volume 0 (2 * π) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R : Real
    ⊢ Iff (CircleIntegrable f c R) (IntervalIntegrable (fun θ => HSMul.hSMul (deri …
  -/
  by_cases h₀ : R = 0
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      h₀ : Eq R 0
      ⊢ Iff (CircleIntegrable f c R) (IntervalIntegrable (fun θ => HSMul.hSMul (deri …
    -/
  · simp (config := { unfoldPartialApp := true }) [h₀, const]
    /-
      🎉 no goals
    -/
  /-
    case neg
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R : Real
    h₀ : Not (Eq R 0)
    ⊢ Iff (CircleIntegrable f c R) (IntervalIntegrable (fun θ => HSMul.hSMul (deri …
  -/
  refine ⟨fun h => h.out, fun h => ?_⟩
  /-
    case neg
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R : Real
    h₀ : Not (Eq R 0)
    h : IntervalIntegrable (fun θ => HSMul.hSMul (deriv (circleMap c R) θ) (f (cir …
    ⊢ CircleIntegrable f c R
  -/
  simp only [CircleIntegrable, intervalIntegrable_iff, deriv_circleMap] at h ⊢
  /-
    case neg
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R : Real
    h₀ : Not (Eq R 0)
    h : MeasureTheory.IntegrableOn (fun θ => HSMul.hSMul (HMul.hMul (circleMap 0 R …
    ⊢ MeasureTheory.IntegrableOn (fun θ => f (circleMap c R θ)) (Set.uIoc 0 (HMul. …
  -/
  refine (h.norm.const_mul |R|⁻¹).mono' ?_ ?_
    /-
      case neg.refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      h₀ : Not (Eq R 0)
      h : MeasureTheory.IntegrableOn (fun θ => HSMul.hSMul (HMul.hMul (circleMap 0 R …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun θ => f (circleMap c R θ)) (MeasureTh …
    -/
  · have H : ∀ {θ}, circleMap 0 R θ * I ≠ 0 := fun {θ} => by simp [h₀, I_ne_zero]
    simpa only [inv_smul_smul₀ H]
      using ((continuous_circleMap 0 R).aestronglyMeasurable.mul_const
        I).aemeasurable.inv.aestronglyMeasurable.smul h.aestronglyMeasurable
    /-
      case neg.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      h₀ : Not (Eq R 0)
      h : MeasureTheory.IntegrableOn (fun θ => HSMul.hSMul (HMul.hMul (circleMap 0 R …
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (f (circleMap c R a))) (HMul.hM …
    -/
  · simp [norm_smul, h₀]
    /-
      🎉 no goals
    -/


theorem ContinuousOn.circleIntegrable' {f : ℂ → E} {c : ℂ} {R : ℝ}
    (hf : ContinuousOn f (sphere c |R|)) : CircleIntegrable f c R :=
  (hf.comp_continuous (continuous_circleMap _ _) (circleMap_mem_sphere' _ _)).intervalIntegrable _ _


theorem ContinuousOn.circleIntegrable {f : ℂ → E} {c : ℂ} {R : ℝ} (hR : 0 ≤ R)
    (hf : ContinuousOn f (sphere c R)) : CircleIntegrable f c R :=
  ContinuousOn.circleIntegrable' <| (_root_.abs_of_nonneg hR).symm ▸ hf


/-- The function `fun z ↦ (z - w) ^ n`, `n : ℤ`, is circle integrable on the circle with center `c`
and radius `|R|` if and only if `R = 0` or `0 ≤ n`, or `w` does not belong to this circle. -/
@[simp]
theorem circleIntegrable_sub_zpow_iff {c w : ℂ} {R : ℝ} {n : ℤ} :
    CircleIntegrable (fun z => (z - w) ^ n) c R ↔ R = 0 ∨ 0 ≤ n ∨ w ∉ sphere c |R| := by
  /-
    c w : Complex
    R : Real
    n : Int
    ⊢ Iff (CircleIntegrable (fun z => HPow.hPow (HSub.hSub z w) n) c R) (Or (Eq R  …
  -/
  constructor
    /-
      case mp
      c w : Complex
      R : Real
      n : Int
      ⊢ CircleIntegrable (fun z => HPow.hPow (HSub.hSub z w) n) c R → Or (Eq R 0) (O …
    -/
  · intro h; contrapose! h; rcases h with ⟨hR, hn, hw⟩
    /-
      case mp.intro.intro
      c w : Complex
      R : Real
      n : Int
      hR : Ne R 0
      hn : LT.lt n 0
      hw : Membership.mem (Metric.sphere c (abs R)) w
      ⊢ Not (CircleIntegrable (fun z => HPow.hPow (HSub.hSub z w) n) c R)
    -/
    simp only [circleIntegrable_iff R, deriv_circleMap]
    /-
      case mp.intro.intro
      c w : Complex
      R : Real
      n : Int
      hR : Ne R 0
      hn : LT.lt n 0
      hw : Membership.mem (Metric.sphere c (abs R)) w
      ⊢ Not (IntervalIntegrable (fun θ => HSMul.hSMul (HMul.hMul (circleMap 0 R θ) C …
    -/
    rw [← image_circleMap_Ioc] at hw; rcases hw with ⟨θ, hθ, rfl⟩
    /-
      case mp.intro.intro.intro.intro
      c : Complex
      R : Real
      n : Int
      hR : Ne R 0
      hn : LT.lt n 0
      θ : Real
      hθ : Membership.mem (Set.Ioc 0 (HMul.hMul 2 Real.pi)) θ
      ⊢ Not (IntervalIntegrable (fun θ_1 => HSMul.hSMul (HMul.hMul (circleMap 0 R θ_ …
    -/
    replace hθ : θ ∈ [[0, 2 * π]] := Icc_subset_uIcc (Ioc_subset_Icc_self hθ)
    /-
      case mp.intro.intro.intro.intro
      c : Complex
      R : Real
      n : Int
      hR : Ne R 0
      hn : LT.lt n 0
      θ : Real
      hθ : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) θ
      ⊢ Not (IntervalIntegrable (fun θ_1 => HSMul.hSMul (HMul.hMul (circleMap 0 R θ_ …
    -/
    refine not_intervalIntegrable_of_sub_inv_isBigO_punctured ?_ Real.two_pi_pos.ne hθ
    /-
      case mp.intro.intro.intro.intro
      c : Complex
      R : Real
      n : Int
      hR : Ne R 0
      hn : LT.lt n 0
      θ : Real
      hθ : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) θ
      ⊢ Asymptotics.IsBigO (nhdsWithin θ (HasCompl.compl (Singleton.singleton θ))) ( …
    -/
    set f : ℝ → ℂ := fun θ' => circleMap c R θ' - circleMap c R θ
    have : ∀ᶠ θ' in 𝓝[≠] θ, f θ' ∈ ball (0 : ℂ) 1 \ {0} := by
      suffices ∀ᶠ z in 𝓝[≠] circleMap c R θ, z - circleMap c R θ ∈ ball (0 : ℂ) 1 \ {0} from
        ((differentiable_circleMap c R θ).hasDerivAt.tendsto_punctured_nhds
          (deriv_circleMap_ne_zero hR)).eventually this
      filter_upwards [self_mem_nhdsWithin, mem_nhdsWithin_of_mem_nhds (ball_mem_nhds _ zero_lt_one)]
      simp_all [dist_eq, sub_eq_zero]
    refine (((hasDerivAt_circleMap c R θ).isBigO_sub.mono inf_le_left).inv_rev
      (this.mono fun θ' h₁ h₂ => absurd h₂ h₁.2)).trans ?_
    /-
      case mp.intro.intro.intro.intro
      c : Complex
      R : Real
      n : Int
      hR : Ne R 0
      hn : LT.lt n 0
      θ : Real
      hθ : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) θ
      f : Real → Complex := fun θ' => HSub.hSub (circleMap c R θ') (circleMap c R θ)
      this : Filter.Eventually (fun θ' => Membership.mem (SDiff.sdiff (Metric.ball 0 …
      ⊢ Asymptotics.IsBigO (Min.min (nhds θ) (Filter.principal (HasCompl.compl (Sing …
    -/
    refine IsBigO.of_bound |R|⁻¹ (this.mono fun θ' hθ' => ?_)
    /-
      case mp.intro.intro.intro.intro
      c : Complex
      R : Real
      n : Int
      hR : Ne R 0
      hn : LT.lt n 0
      θ : Real
      hθ : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) θ
      f : Real → Complex := fun θ' => HSub.hSub (circleMap c R θ') (circleMap c R θ)
      this : Filter.Eventually (fun θ' => Membership.mem (SDiff.sdiff (Metric.ball 0 …
      θ' : Real
      hθ' : Membership.mem (SDiff.sdiff (Metric.ball 0 1) (Singleton.singleton 0)) ( …
      ⊢ LE.le (Norm.norm (Inv.inv (HSub.hSub (circleMap c R θ') (circleMap c R θ)))) …
    -/
    set x := abs (f θ')
    suffices x⁻¹ ≤ x ^ n by
      simpa only [inv_mul_cancel_left₀, abs_eq_zero.not.2 hR, norm_eq_abs, map_inv₀,
        Algebra.id.smul_eq_mul, map_mul, abs_circleMap_zero, abs_I, mul_one, abs_zpow, Ne,
        not_false_iff] using this
    /-
      case mp.intro.intro.intro.intro
      c : Complex
      R : Real
      n : Int
      hR : Ne R 0
      hn : LT.lt n 0
      θ : Real
      hθ : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) θ
      f : Real → Complex := fun θ' => HSub.hSub (circleMap c R θ') (circleMap c R θ)
      this : Filter.Eventually (fun θ' => Membership.mem (SDiff.sdiff (Metric.ball 0 …
      θ' : Real
      hθ' : Membership.mem (SDiff.sdiff (Metric.ball 0 1) (Singleton.singleton 0)) ( …
      x : Real := Complex.abs (f θ')
      ⊢ LE.le (Inv.inv x) (HPow.hPow x n)
    -/
    have : x ∈ Ioo (0 : ℝ) 1 := by simpa [x, and_comm] using hθ'
    /-
      case mp.intro.intro.intro.intro
      c : Complex
      R : Real
      n : Int
      hR : Ne R 0
      hn : LT.lt n 0
      θ : Real
      hθ : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) θ
      f : Real → Complex := fun θ' => HSub.hSub (circleMap c R θ') (circleMap c R θ)
      this✝ : Filter.Eventually (fun θ' => Membership.mem (SDiff.sdiff (Metric.ball  …
      θ' : Real
      hθ' : Membership.mem (SDiff.sdiff (Metric.ball 0 1) (Singleton.singleton 0)) ( …
      x : Real := Complex.abs (f θ')
      this : Membership.mem (Set.Ioo 0 1) x
      ⊢ LE.le (Inv.inv x) (HPow.hPow x n)
    -/
    rw [← zpow_neg_one]
    /-
      case mp.intro.intro.intro.intro
      c : Complex
      R : Real
      n : Int
      hR : Ne R 0
      hn : LT.lt n 0
      θ : Real
      hθ : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) θ
      f : Real → Complex := fun θ' => HSub.hSub (circleMap c R θ') (circleMap c R θ)
      this✝ : Filter.Eventually (fun θ' => Membership.mem (SDiff.sdiff (Metric.ball  …
      θ' : Real
      hθ' : Membership.mem (SDiff.sdiff (Metric.ball 0 1) (Singleton.singleton 0)) ( …
      x : Real := Complex.abs (f θ')
      this : Membership.mem (Set.Ioo 0 1) x
      ⊢ LE.le (HPow.hPow x (-1)) (HPow.hPow x n)
    -/
    refine (zpow_right_strictAnti₀ this.1 this.2).le_iff_le.2 (Int.lt_add_one_iff.1 ?_); exact hn
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
    /-
      case mpr
      c w : Complex
      R : Real
      n : Int
      ⊢ Or (Eq R 0) (Or (LE.le 0 n) (Not (Membership.mem (Metric.sphere c (abs R)) w …
    -/
  · rintro (rfl | H)
    exacts [circleIntegrable_zero_radius,
      ((continuousOn_id.sub continuousOn_const).zpow₀ _ fun z hz =>
        H.symm.imp_left fun (hw : w ∉ sphere c |R|) =>
          sub_ne_zero.2 <| ne_of_mem_of_not_mem hz hw).circleIntegrable']


@[simp]
theorem circleIntegrable_sub_inv_iff {c w : ℂ} {R : ℝ} :
    CircleIntegrable (fun z => (z - w)⁻¹) c R ↔ R = 0 ∨ w ∉ sphere c |R| := by
  /-
    c w : Complex
    R : Real
    ⊢ Iff (CircleIntegrable (fun z => Inv.inv (HSub.hSub z w)) c R) (Or (Eq R 0) ( …
  -/
  simp only [← zpow_neg_one, circleIntegrable_sub_zpow_iff]; norm_num
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Definition for $\oint_{|z-c|=R} f(z)\,dz$. -/
def circleIntegral (f : ℂ → E) (c : ℂ) (R : ℝ) : E :=
  ∫ θ : ℝ in (0)..2 * π, deriv (circleMap c R) θ • f (circleMap c R θ)


notation3 "∮ "(...)" in ""C("c", "R")"", "r:(scoped f => circleIntegral f c R) => r


theorem circleIntegral_def_Icc (f : ℂ → E) (c : ℂ) (R : ℝ) :
    (∮ z in C(c, R), f z) = ∫ θ in Icc 0 (2 * π),
    deriv (circleMap c R) θ • f (circleMap c R θ) := by
  rw [circleIntegral, intervalIntegral.integral_of_le Real.two_pi_pos.le,
    Measure.restrict_congr_set Ioc_ae_eq_Icc]


@[simp]
theorem integral_radius_zero (f : ℂ → E) (c : ℂ) : (∮ z in C(c, 0), f z) = 0 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    ⊢ Eq (circleIntegral (fun z => f z) c 0) 0
  -/
  simp (config := { unfoldPartialApp := true }) [circleIntegral, const]
  /-
    🎉 no goals
  -/


theorem integral_congr {f g : ℂ → E} {c : ℂ} {R : ℝ} (hR : 0 ≤ R) (h : EqOn f g (sphere c R)) :
    (∮ z in C(c, R), f z) = ∮ z in C(c, R), g z :=
                                                /-
                                                  E : Type u_1
                                                  inst✝¹ : NormedAddCommGroup E
                                                  inst✝ : NormedSpace Complex E
                                                  f g : Complex → E
                                                  c : Complex
                                                  R : Real
                                                  hR : LE.le 0 R
                                                  h : Set.EqOn f g (Metric.sphere c R)
                                                  θ : Real
                                                  x✝ : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) θ
                                                  ⊢ Eq (HSMul.hSMul (deriv (circleMap c R) θ) ((fun z => f z) (circleMap c R θ)) …
                                                -/
  intervalIntegral.integral_congr fun θ _ => by simp only [h (circleMap_mem_sphere _ hR _)]
                                                /-
                                                  🎉 no goals
                                                -/


theorem integral_sub_inv_smul_sub_smul (f : ℂ → E) (c w : ℂ) (R : ℝ) :
    (∮ z in C(c, R), (z - w)⁻¹ • (z - w) • f z) = ∮ z in C(c, R), f z := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c w : Complex
    R : Real
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z w)) (HSMul.hS …
  -/
  rcases eq_or_ne R 0 with (rfl | hR); · simp only [integral_radius_zero]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c w : Complex
    R : Real
    hR : Ne R 0
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z w)) (HSMul.hS …
  -/
  have : (circleMap c R ⁻¹' {w}).Countable := (countable_singleton _).preimage_circleMap c hR
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c w : Complex
    R : Real
    hR : Ne R 0
    this : (Set.preimage (circleMap c R) (Singleton.singleton w)).Countable
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z w)) (HSMul.hS …
  -/
  refine intervalIntegral.integral_congr_ae ((this.ae_not_mem _).mono fun θ hθ _' => ?_)
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c w : Complex
    R : Real
    hR : Ne R 0
    this : (Set.preimage (circleMap c R) (Singleton.singleton w)).Countable
    θ : Real
    hθ : Not (Membership.mem (Set.preimage (circleMap c R) (Singleton.singleton w) …
    _' : Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi)) θ
    ⊢ Eq (HSMul.hSMul (deriv (circleMap c R) θ) ((fun z => HSMul.hSMul (Inv.inv (H …
  -/
  change circleMap c R θ ≠ w at hθ
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c w : Complex
    R : Real
    hR : Ne R 0
    this : (Set.preimage (circleMap c R) (Singleton.singleton w)).Countable
    θ : Real
    _' : Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi)) θ
    hθ : Ne (circleMap c R θ) w
    ⊢ Eq (HSMul.hSMul (deriv (circleMap c R) θ) ((fun z => HSMul.hSMul (Inv.inv (H …
  -/
  simp only [inv_smul_smul₀ (sub_ne_zero.2 <| hθ)]
  /-
    🎉 no goals
  -/


theorem integral_undef {f : ℂ → E} {c : ℂ} {R : ℝ} (hf : ¬CircleIntegrable f c R) :
    (∮ z in C(c, R), f z) = 0 :=
  intervalIntegral.integral_undef (mt (circleIntegrable_iff R).mpr hf)


theorem integral_sub {f g : ℂ → E} {c : ℂ} {R : ℝ} (hf : CircleIntegrable f c R)
    (hg : CircleIntegrable g c R) :
    (∮ z in C(c, R), f z - g z) = (∮ z in C(c, R), f z) - ∮ z in C(c, R), g z := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f g : Complex → E
    c : Complex
    R : Real
    hf : CircleIntegrable f c R
    hg : CircleIntegrable g c R
    ⊢ Eq (circleIntegral (fun z => HSub.hSub (f z) (g z)) c R) (HSub.hSub (circleI …
  -/
  simp only [circleIntegral, smul_sub, intervalIntegral.integral_sub hf.out hg.out]
  /-
    🎉 no goals
  -/


theorem norm_integral_le_of_norm_le_const' {f : ℂ → E} {c : ℂ} {R C : ℝ}
    (hf : ∀ z ∈ sphere c |R|, ‖f z‖ ≤ C) : ‖∮ z in C(c, R), f z‖ ≤ 2 * π * |R| * C :=
  calc
    ‖∮ z in C(c, R), f z‖ ≤ |R| * C * |2 * π - 0| :=
      intervalIntegral.norm_integral_le_of_norm_le_const fun θ _ =>
        calc
          ‖deriv (circleMap c R) θ • f (circleMap c R θ)‖ = |R| * ‖f (circleMap c R θ)‖ := by
            /-
              E : Type u_1
              inst✝¹ : NormedAddCommGroup E
              inst✝ : NormedSpace Complex E
              f : Complex → E
              c : Complex
              R C : Real
              hf : ∀ (z : Complex), Membership.mem (Metric.sphere c (abs R)) z → LE.le (Norm …
              θ : Real
              x✝ : Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi)) θ
              ⊢ Eq (Norm.norm (HSMul.hSMul (deriv (circleMap c R) θ) (f (circleMap c R θ)))) …
            -/
            simp [norm_smul]
            /-
              🎉 no goals
            -/
          _ ≤ |R| * C :=
            mul_le_mul_of_nonneg_left (hf _ <| circleMap_mem_sphere' _ _ _) (abs_nonneg _)
                              /-
                                E : Type u_1
                                inst✝¹ : NormedAddCommGroup E
                                inst✝ : NormedSpace Complex E
                                f : Complex → E
                                c : Complex
                                R C : Real
                                hf : ∀ (z : Complex), Membership.mem (Metric.sphere c (abs R)) z → LE.le (Norm …
                                ⊢ Eq (HMul.hMul (HMul.hMul (abs R) C) (abs (HSub.hSub (HMul.hMul 2 Real.pi) 0) …
                              -/
    _ = 2 * π * |R| * C := by rw [sub_zero, _root_.abs_of_pos Real.two_pi_pos]; ac_rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem norm_integral_le_of_norm_le_const {f : ℂ → E} {c : ℂ} {R C : ℝ} (hR : 0 ≤ R)
    (hf : ∀ z ∈ sphere c R, ‖f z‖ ≤ C) : ‖∮ z in C(c, R), f z‖ ≤ 2 * π * R * C :=
  have : |R| = R := abs_of_nonneg hR
  calc
                                                                                        /-
                                                                                          E : Type u_1
                                                                                          inst✝¹ : NormedAddCommGroup E
                                                                                          inst✝ : NormedSpace Complex E
                                                                                          f : Complex → E
                                                                                          c : Complex
                                                                                          R C : Real
                                                                                          hR : LE.le 0 R
                                                                                          hf : ∀ (z : Complex), Membership.mem (Metric.sphere c R) z → LE.le (Norm.norm  …
                                                                                          this : Eq (abs R) R
                                                                                          ⊢ ∀ (z : Complex), Membership.mem (Metric.sphere c (abs R)) z → LE.le (Norm.no …
                                                                                        -/
    ‖∮ z in C(c, R), f z‖ ≤ 2 * π * |R| * C := norm_integral_le_of_norm_le_const' <| by rwa [this]
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
                            /-
                              E : Type u_1
                              inst✝¹ : NormedAddCommGroup E
                              inst✝ : NormedSpace Complex E
                              f : Complex → E
                              c : Complex
                              R C : Real
                              hR : LE.le 0 R
                              hf : ∀ (z : Complex), Membership.mem (Metric.sphere c R) z → LE.le (Norm.norm  …
                              this : Eq (abs R) R
                              ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) (abs R)) C) (HMul.hMul (HMul. …
                            -/
    _ = 2 * π * R * C := by rw [this]
                            /-
                              🎉 no goals
                            -/


theorem norm_two_pi_i_inv_smul_integral_le_of_norm_le_const {f : ℂ → E} {c : ℂ} {R C : ℝ}
    (hR : 0 ≤ R) (hf : ∀ z ∈ sphere c R, ‖f z‖ ≤ C) :
    ‖(2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), f z‖ ≤ R * C := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R C : Real
    hR : LE.le 0 R
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere c R) z → LE.le (Norm.norm  …
    ⊢ LE.le (Norm.norm (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Com …
  -/
  have : ‖(2 * π * I : ℂ)⁻¹‖ = (2 * π)⁻¹ := by simp [Real.pi_pos.le]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R C : Real
    hR : LE.le 0 R
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere c R) z → LE.le (Norm.norm  …
    this : Eq (Norm.norm (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I))) ( …
    ⊢ LE.le (Norm.norm (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Com …
  -/
  rw [norm_smul, this, ← div_eq_inv_mul, div_le_iff₀ Real.two_pi_pos, mul_comm (R * C), ← mul_assoc]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R C : Real
    hR : LE.le 0 R
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere c R) z → LE.le (Norm.norm  …
    this : Eq (Norm.norm (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I))) ( …
    ⊢ LE.le (Norm.norm (circleIntegral (fun z => f z) c R)) (HMul.hMul (HMul.hMul  …
  -/
  exact norm_integral_le_of_norm_le_const hR hf
  /-
    🎉 no goals
  -/


/-- If `f` is continuous on the circle `|z - c| = R`, `R > 0`, the `‖f z‖` is less than or equal to
`C : ℝ` on this circle, and this norm is strictly less than `C` at some point `z` of the circle,
then `‖∮ z in C(c, R), f z‖ < 2 * π * R * C`. -/
theorem norm_integral_lt_of_norm_le_const_of_lt {f : ℂ → E} {c : ℂ} {R C : ℝ} (hR : 0 < R)
    (hc : ContinuousOn f (sphere c R)) (hf : ∀ z ∈ sphere c R, ‖f z‖ ≤ C)
    (hlt : ∃ z ∈ sphere c R, ‖f z‖ < C) : ‖∮ z in C(c, R), f z‖ < 2 * π * R * C := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R C : Real
    hR : LT.lt 0 R
    hc : ContinuousOn f (Metric.sphere c R)
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere c R) z → LE.le (Norm.norm  …
    hlt : Exists fun z => And (Membership.mem (Metric.sphere c R) z) (LT.lt (Norm. …
    ⊢ LT.lt (Norm.norm (circleIntegral (fun z => f z) c R)) (HMul.hMul (HMul.hMul  …
  -/
  rw [← _root_.abs_of_pos hR, ← image_circleMap_Ioc] at hlt
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R C : Real
    hR : LT.lt 0 R
    hc : ContinuousOn f (Metric.sphere c R)
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere c R) z → LE.le (Norm.norm  …
    hlt : Exists fun z => And (Membership.mem (Set.image (circleMap c R) (Set.Ioc  …
    ⊢ LT.lt (Norm.norm (circleIntegral (fun z => f z) c R)) (HMul.hMul (HMul.hMul  …
  -/
  rcases hlt with ⟨_, ⟨θ₀, hmem, rfl⟩, hlt⟩
  calc
    ‖∮ z in C(c, R), f z‖ ≤ ∫ θ in (0)..2 * π, ‖deriv (circleMap c R) θ • f (circleMap c R θ)‖ :=
      intervalIntegral.norm_integral_le_integral_norm Real.two_pi_pos.le
    _ < ∫ _ in (0)..2 * π, R * C := by
      simp only [norm_smul, deriv_circleMap, norm_eq_abs, map_mul, abs_I, mul_one,
        abs_circleMap_zero, abs_of_pos hR]
      refine intervalIntegral.integral_lt_integral_of_continuousOn_of_le_of_exists_lt
          Real.two_pi_pos ?_ continuousOn_const (fun θ _ => ?_) ⟨θ₀, Ioc_subset_Icc_self hmem, ?_⟩
      · exact continuousOn_const.mul (hc.comp (continuous_circleMap _ _).continuousOn fun θ _ =>
          circleMap_mem_sphere _ hR.le _).norm
      · exact mul_le_mul_of_nonneg_left (hf _ <| circleMap_mem_sphere _ hR.le _) hR.le
      · exact (mul_lt_mul_left hR).2 hlt
    _ = 2 * π * R * C := by simp [mul_assoc]; ring


@[simp]
theorem integral_smul {𝕜 : Type*} [RCLike 𝕜] [NormedSpace 𝕜 E] [SMulCommClass 𝕜 ℂ E] (a : 𝕜)
    (f : ℂ → E) (c : ℂ) (R : ℝ) : (∮ z in C(c, R), a • f z) = a • ∮ z in C(c, R), f z := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : SMulCommClass 𝕜 Complex E
    a : 𝕜
    f : Complex → E
    c : Complex
    R : Real
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul a (f z)) c R) (HSMul.hSMul a (circl …
  -/
  simp only [circleIntegral, ← smul_comm a (_ : ℂ) (_ : E), intervalIntegral.integral_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_smul_const [CompleteSpace E] (f : ℂ → ℂ) (a : E) (c : ℂ) (R : ℝ) :
    (∮ z in C(c, R), f z • a) = (∮ z in C(c, R), f z) • a := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → Complex
    a : E
    c : Complex
    R : Real
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (f z) a) c R) (HSMul.hSMul (circleI …
  -/
  simp only [circleIntegral, intervalIntegral.integral_smul_const, ← smul_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_const_mul (a : ℂ) (f : ℂ → ℂ) (c : ℂ) (R : ℝ) :
    (∮ z in C(c, R), a * f z) = a * ∮ z in C(c, R), f z :=
  integral_smul a f c R


@[simp]
theorem integral_sub_center_inv (c : ℂ) {R : ℝ} (hR : R ≠ 0) :
    (∮ z in C(c, R), (z - c)⁻¹) = 2 * π * I := by
  simp [circleIntegral, ← div_eq_mul_inv, mul_div_cancel_left₀ _ (circleMap_ne_center hR),
    -- Porting note: `simp` didn't need a hint to apply `integral_const` here
    intervalIntegral.integral_const I]


/-- If `f' : ℂ → E` is a derivative of a complex differentiable function on the circle
`Metric.sphere c |R|`, then `∮ z in C(c, R), f' z = 0`. -/
theorem integral_eq_zero_of_hasDerivWithinAt' [CompleteSpace E] {f f' : ℂ → E} {c : ℂ} {R : ℝ}
    (h : ∀ z ∈ sphere c |R|, HasDerivWithinAt f (f' z) (sphere c |R|) z) :
    (∮ z in C(c, R), f' z) = 0 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f f' : Complex → E
    c : Complex
    R : Real
    h : ∀ (z : Complex), Membership.mem (Metric.sphere c (abs R)) z → HasDerivWith …
    ⊢ Eq (circleIntegral (fun z => f' z) c R) 0
  -/
  by_cases hi : CircleIntegrable f' c R
    /-
      case pos
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f f' : Complex → E
      c : Complex
      R : Real
      h : ∀ (z : Complex), Membership.mem (Metric.sphere c (abs R)) z → HasDerivWith …
      hi : CircleIntegrable f' c R
      ⊢ Eq (circleIntegral (fun z => f' z) c R) 0
    -/
  · rw [← sub_eq_zero.2 ((periodic_circleMap c R).comp f).eq]
    /-
      case pos
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f f' : Complex → E
      c : Complex
      R : Real
      h : ∀ (z : Complex), Membership.mem (Metric.sphere c (abs R)) z → HasDerivWith …
      hi : CircleIntegrable f' c R
      ⊢ Eq (circleIntegral (fun z => f' z) c R) (HSub.hSub (Function.comp f (circleM …
    -/
    refine intervalIntegral.integral_eq_sub_of_hasDerivAt (fun θ _ => ?_) hi.out
    exact (h _ (circleMap_mem_sphere' _ _ _)).scomp_hasDerivAt θ
      (differentiable_circleMap _ _ _).hasDerivAt (circleMap_mem_sphere' _ _)
    /-
      case neg
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f f' : Complex → E
      c : Complex
      R : Real
      h : ∀ (z : Complex), Membership.mem (Metric.sphere c (abs R)) z → HasDerivWith …
      hi : Not (CircleIntegrable f' c R)
      ⊢ Eq (circleIntegral (fun z => f' z) c R) 0
    -/
  · exact integral_undef hi
    /-
      🎉 no goals
    -/


/-- If `f' : ℂ → E` is a derivative of a complex differentiable function on the circle
`Metric.sphere c R`, then `∮ z in C(c, R), f' z = 0`. -/
theorem integral_eq_zero_of_hasDerivWithinAt [CompleteSpace E]
    {f f' : ℂ → E} {c : ℂ} {R : ℝ} (hR : 0 ≤ R)
    (h : ∀ z ∈ sphere c R, HasDerivWithinAt f (f' z) (sphere c R) z) : (∮ z in C(c, R), f' z) = 0 :=
  integral_eq_zero_of_hasDerivWithinAt' <| (_root_.abs_of_nonneg hR).symm ▸ h


/-- If `n < 0` and `|w - c| = |R|`, then `(z - w) ^ n` is not circle integrable on the circle with
center `c` and radius `|R|`, so the integral `∮ z in C(c, R), (z - w) ^ n` is equal to zero. -/
theorem integral_sub_zpow_of_undef {n : ℤ} {c w : ℂ} {R : ℝ} (hn : n < 0)
    (hw : w ∈ sphere c |R|) : (∮ z in C(c, R), (z - w) ^ n) = 0 := by
  /-
    n : Int
    c w : Complex
    R : Real
    hn : LT.lt n 0
    hw : Membership.mem (Metric.sphere c (abs R)) w
    ⊢ Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z w) n) c R) 0
  -/
  rcases eq_or_ne R 0 with (rfl | h0)
    /-
      case inl
      n : Int
      c w : Complex
      hn : LT.lt n 0
      hw : Membership.mem (Metric.sphere c (abs 0)) w
      ⊢ Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z w) n) c 0) 0
    -/
  · apply integral_radius_zero
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Int
      c w : Complex
      R : Real
      hn : LT.lt n 0
      hw : Membership.mem (Metric.sphere c (abs R)) w
      h0 : Ne R 0
      ⊢ Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z w) n) c R) 0
    -/
  · apply integral_undef
    /-
      case inr.hf
      n : Int
      c w : Complex
      R : Real
      hn : LT.lt n 0
      hw : Membership.mem (Metric.sphere c (abs R)) w
      h0 : Ne R 0
      ⊢ Not (CircleIntegrable (fun z => HPow.hPow (HSub.hSub z w) n) c R)
    -/
    simpa [circleIntegrable_sub_zpow_iff, *, not_or]
    /-
      🎉 no goals
    -/


/-- If `n ≠ -1` is an integer number, then the integral of `(z - w) ^ n` over the circle equals
zero. -/
theorem integral_sub_zpow_of_ne {n : ℤ} (hn : n ≠ -1) (c w : ℂ) (R : ℝ) :
    (∮ z in C(c, R), (z - w) ^ n) = 0 := by
  /-
    n : Int
    hn : Ne n (-1)
    c w : Complex
    R : Real
    ⊢ Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z w) n) c R) 0
  -/
  rcases em (w ∈ sphere c |R| ∧ n < -1) with (⟨hw, hn⟩ | H)
    /-
      case inl.intro
      n : Int
      hn✝ : Ne n (-1)
      c w : Complex
      R : Real
      hw : Membership.mem (Metric.sphere c (abs R)) w
      hn : LT.lt n (-1)
      ⊢ Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z w) n) c R) 0
    -/
  · exact integral_sub_zpow_of_undef (hn.trans (by decide)) hw
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Int
    hn : Ne n (-1)
    c w : Complex
    R : Real
    H : Not (And (Membership.mem (Metric.sphere c (abs R)) w) (LT.lt n (-1)))
    ⊢ Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z w) n) c R) 0
  -/
  push_neg at H
  have hd : ∀ z, z ≠ w ∨ -1 ≤ n →
      HasDerivAt (fun z => (z - w) ^ (n + 1) / (n + 1)) ((z - w) ^ n) z := by
    intro z hne
    convert ((hasDerivAt_zpow (n + 1) _ (hne.imp _ _)).comp z
      ((hasDerivAt_id z).sub_const w)).div_const _ using 1
    · have hn' : (n + 1 : ℂ) ≠ 0 := by
        rwa [Ne, ← eq_neg_iff_add_eq_zero, ← Int.cast_one, ← Int.cast_neg, Int.cast_inj]
      simp [mul_assoc, mul_div_cancel_left₀ _ hn']
    exacts [sub_ne_zero.2, neg_le_iff_add_nonneg.1]
  /-
    case inr
    n : Int
    hn : Ne n (-1)
    c w : Complex
    R : Real
    H : Membership.mem (Metric.sphere c (abs R)) w → LE.le (-1) n
    hd : ∀ (z : Complex), Or (Ne z w) (LE.le (-1) n) → HasDerivAt (fun z => HDiv.h …
    ⊢ Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z w) n) c R) 0
  -/
  refine integral_eq_zero_of_hasDerivWithinAt' fun z hz => (hd z ?_).hasDerivWithinAt
  /-
    case inr
    n : Int
    hn : Ne n (-1)
    c w : Complex
    R : Real
    H : Membership.mem (Metric.sphere c (abs R)) w → LE.le (-1) n
    hd : ∀ (z : Complex), Or (Ne z w) (LE.le (-1) n) → HasDerivAt (fun z => HDiv.h …
    z : Complex
    hz : Membership.mem (Metric.sphere c (abs R)) z
    ⊢ Or (Ne z w) (LE.le (-1) n)
  -/
  exact (ne_or_eq z w).imp_right fun (h : z = w) => H <| h ▸ hz
  /-
    🎉 no goals
  -/


/-- The power series that is equal to
$\frac{1}{2πi}\sum_{n=0}^{\infty}
  \oint_{|z-c|=R} \left(\frac{w-c}{z - c}\right)^n \frac{1}{z-c}f(z)\,dz$ at
`w - c`. The coefficients of this power series depend only on `f ∘ circleMap c R`, and the power
series converges to `f w` if `f` is differentiable on the closed ball `Metric.closedBall c R` and
`w` belongs to the corresponding open ball. For any circle integrable function `f`, this power
series converges to the Cauchy integral for `f`. -/
def cauchyPowerSeries (f : ℂ → E) (c : ℂ) (R : ℝ) : FormalMultilinearSeries ℂ ℂ E := fun n =>
  ContinuousMultilinearMap.mkPiRing ℂ _ <|
    (2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - c)⁻¹ ^ n • (z - c)⁻¹ • f z


theorem cauchyPowerSeries_apply (f : ℂ → E) (c : ℂ) (R : ℝ) (n : ℕ) (w : ℂ) :
    (cauchyPowerSeries f c R n fun _ => w) =
      (2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (w / (z - c)) ^ n • (z - c)⁻¹ • f z := by
  simp only [cauchyPowerSeries, ContinuousMultilinearMap.mkPiRing_apply, Fin.prod_const,
    div_eq_mul_inv, mul_pow, mul_smul, circleIntegral.integral_smul]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R : Real
    n : Nat
    w : Complex
    ⊢ Eq (HSMul.hSMul (HPow.hPow w n) (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul  …
  -/
  rw [← smul_comm (w ^ n)]
  /-
    🎉 no goals
  -/


theorem norm_cauchyPowerSeries_le (f : ℂ → E) (c : ℂ) (R : ℝ) (n : ℕ) :
    ‖cauchyPowerSeries f c R n‖ ≤
      ((2 * π)⁻¹ * ∫ θ : ℝ in (0)..2 * π, ‖f (circleMap c R θ)‖) * |R|⁻¹ ^ n :=
  calc ‖cauchyPowerSeries f c R n‖
    _ = (2 * π)⁻¹ * ‖∮ z in C(c, R), (z - c)⁻¹ ^ n • (z - c)⁻¹ • f z‖ := by
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        c : Complex
        R : Real
        n : Nat
        ⊢ Eq (Norm.norm (cauchyPowerSeries f c R n)) (HMul.hMul (Inv.inv (HMul.hMul 2  …
      -/
      simp [cauchyPowerSeries, norm_smul, Real.pi_pos.le]
      /-
        🎉 no goals
      -/
    _ ≤ (2 * π)⁻¹ * ∫ θ in (0)..2 * π, ‖deriv (circleMap c R) θ •
        (circleMap c R θ - c)⁻¹ ^ n • (circleMap c R θ - c)⁻¹ • f (circleMap c R θ)‖ :=
      (mul_le_mul_of_nonneg_left
        (intervalIntegral.norm_integral_le_integral_norm Real.two_pi_pos.le)
            /-
              E : Type u_1
              inst✝¹ : NormedAddCommGroup E
              inst✝ : NormedSpace Complex E
              f : Complex → E
              c : Complex
              R : Real
              n : Nat
              ⊢ LE.le 0 (Inv.inv (HMul.hMul 2 Real.pi))
            -/
        (by simp [Real.pi_pos.le]))
            /-
              🎉 no goals
            -/
    _ = (2 * π)⁻¹ *
        (|R|⁻¹ ^ n * (|R| * (|R|⁻¹ * ∫ x : ℝ in (0)..2 * π, ‖f (circleMap c R x)‖))) := by
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        c : Complex
        R : Real
        n : Nat
        ⊢ Eq (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (intervalIntegral (fun θ => No …
      -/
      simp [norm_smul, mul_left_comm |R|]
      /-
        🎉 no goals
      -/
    _ ≤ ((2 * π)⁻¹ * ∫ θ : ℝ in (0)..2 * π, ‖f (circleMap c R θ)‖) * |R|⁻¹ ^ n := by
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        c : Complex
        R : Real
        n : Nat
        ⊢ LE.le (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (HMul.hMul (HPow.hPow (Inv. …
      -/
      rcases eq_or_ne R 0 with (rfl | hR)
        /-
          case inl
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Complex → E
          c : Complex
          n : Nat
          ⊢ LE.le (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (HMul.hMul (HPow.hPow (Inv. …
        -/
      · cases n <;> simp [-mul_inv_rev]
                    /-
                      🎉 no goals
                    -/
        /-
          case inl.zero
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Complex → E
          c : Complex
          ⊢ LE.le 0 (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (HMul.hMul (HMul.hMul 2 R …
        -/
        rw [← mul_assoc, inv_mul_cancel₀ (Real.two_pi_pos.ne.symm), one_mul]
        /-
          case inl.zero
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Complex → E
          c : Complex
          ⊢ LE.le 0 (Norm.norm (f c))
        -/
        apply norm_nonneg
        /-
          🎉 no goals
        -/
        /-
          case inr
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Complex → E
          c : Complex
          R : Real
          n : Nat
          hR : Ne R 0
          ⊢ LE.le (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (HMul.hMul (HPow.hPow (Inv. …
        -/
      · rw [mul_inv_cancel_left₀, mul_assoc, mul_comm (|R|⁻¹ ^ n)]
        /-
          case inr.h
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Complex → E
          c : Complex
          R : Real
          n : Nat
          hR : Ne R 0
          ⊢ Ne (abs R) 0
        -/
        rwa [Ne, _root_.abs_eq_zero]
        /-
          🎉 no goals
        -/


theorem le_radius_cauchyPowerSeries (f : ℂ → E) (c : ℂ) (R : ℝ≥0) :
    ↑R ≤ (cauchyPowerSeries f c R).radius := by
  refine
    (cauchyPowerSeries f c R).le_radius_of_bound
      ((2 * π)⁻¹ * ∫ θ : ℝ in (0)..2 * π, ‖f (circleMap c R θ)‖) fun n => ?_
  refine (mul_le_mul_of_nonneg_right (norm_cauchyPowerSeries_le _ _ _ _)
    (pow_nonneg R.coe_nonneg _)).trans ?_
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R : NNReal
    n : Nat
    ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (inte …
  -/
  rw [_root_.abs_of_nonneg R.coe_nonneg]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R : NNReal
    n : Nat
    ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (inte …
  -/
  rcases eq_or_ne (R ^ n : ℝ) 0 with hR | hR
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : NNReal
      n : Nat
      hR : Eq (HPow.hPow (↑R) n) 0
      ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (inte …
    -/
  · rw_mod_cast [hR, mul_zero]
    exact mul_nonneg (inv_nonneg.2 Real.two_pi_pos.le)
      (intervalIntegral.integral_nonneg Real.two_pi_pos.le fun _ _ => norm_nonneg _)
    /-
      case inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : NNReal
      n : Nat
      hR : Ne (HPow.hPow (↑R) n) 0
      ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (inte …
    -/
  · rw [inv_pow]
    /-
      case inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : NNReal
      n : Nat
      hR : Ne (HPow.hPow (↑R) n) 0
      ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (inte …
    -/
    have : (R : ℝ) ^ n ≠ 0 := by norm_cast at hR ⊢
    /-
      case inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : NNReal
      n : Nat
      hR this : Ne (HPow.hPow (↑R) n) 0
      ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv (HMul.hMul 2 Real.pi)) (inte …
    -/
    rw [inv_mul_cancel_right₀ this]
    /-
      🎉 no goals
    -/


/-- For any circle integrable function `f`, the power series `cauchyPowerSeries f c R` multiplied
by `2πI` converges to the integral `∮ z in C(c, R), (z - w)⁻¹ • f z` on the open disc
`Metric.ball c R`. -/
theorem hasSum_two_pi_I_cauchyPowerSeries_integral {f : ℂ → E} {c : ℂ} {R : ℝ} {w : ℂ}
    (hf : CircleIntegrable f c R) (hw : abs w < R) :
    HasSum (fun n : ℕ => ∮ z in C(c, R), (w / (z - c)) ^ n • (z - c)⁻¹ • f z)
      (∮ z in C(c, R), (z - (c + w))⁻¹ • f z) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R : Real
    w : Complex
    hf : CircleIntegrable f c R
    hw : LT.lt (Complex.abs w) R
    ⊢ HasSum (fun n => circleIntegral (fun z => HSMul.hSMul (HPow.hPow (HDiv.hDiv  …
  -/
  have hR : 0 < R := (Complex.abs.nonneg w).trans_lt hw
  have hwR : abs w / R ∈ Ico (0 : ℝ) 1 :=
    ⟨div_nonneg (Complex.abs.nonneg w) hR.le, (div_lt_one hR).2 hw⟩
  refine intervalIntegral.hasSum_integral_of_dominated_convergence
      (fun n θ => ‖f (circleMap c R θ)‖ * (abs w / R) ^ n) (fun n => ?_) (fun n => ?_) ?_ ?_ ?_
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      w : Complex
      hf : CircleIntegrable f c R
      hw : LT.lt (Complex.abs w) R
      hR : LT.lt 0 R
      hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
      n : Nat
      ⊢ MeasureTheory.AEStronglyMeasurable (fun θ => HSMul.hSMul (deriv (circleMap c …
    -/
  · simp only [deriv_circleMap]
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      w : Complex
      hf : CircleIntegrable f c R
      hw : LT.lt (Complex.abs w) R
      hR : LT.lt 0 R
      hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
      n : Nat
      ⊢ MeasureTheory.AEStronglyMeasurable (fun θ => HSMul.hSMul (HMul.hMul (circleM …
    -/
    apply_rules [AEStronglyMeasurable.smul, hf.def'.1] <;> apply Measurable.aestronglyMeasurable
      /-
        case refine_1.hf.hf
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        c : Complex
        R : Real
        w : Complex
        hf : CircleIntegrable f c R
        hw : LT.lt (Complex.abs w) R
        hR : LT.lt 0 R
        hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
        n : Nat
        ⊢ Measurable fun x => HMul.hMul (circleMap 0 R x) Complex.I
      -/
    · fun_prop
      /-
        🎉 no goals
      -/
      /-
        case refine_1.hg.hf.hf
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        c : Complex
        R : Real
        w : Complex
        hf : CircleIntegrable f c R
        hw : LT.lt (Complex.abs w) R
        hR : LT.lt 0 R
        hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
        n : Nat
        ⊢ Measurable fun x => HPow.hPow (HDiv.hDiv w (HSub.hSub (circleMap c R x) c)) n
      -/
    · fun_prop
      /-
        🎉 no goals
      -/
      /-
        case refine_1.hg.hg.hf.hf
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        c : Complex
        R : Real
        w : Complex
        hf : CircleIntegrable f c R
        hw : LT.lt (Complex.abs w) R
        hR : LT.lt 0 R
        hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
        n : Nat
        ⊢ Measurable fun x => Inv.inv (HSub.hSub (circleMap c R x) c)
      -/
    · fun_prop
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      w : Complex
      hf : CircleIntegrable f c R
      hw : LT.lt (Complex.abs w) R
      hR : LT.lt 0 R
      hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
      n : Nat
      ⊢ Filter.Eventually (fun t => Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi) …
    -/
  · simp [norm_smul, abs_of_pos hR, mul_left_comm R, inv_mul_cancel_left₀ hR.ne', mul_comm ‖_‖]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      w : Complex
      hf : CircleIntegrable f c R
      hw : LT.lt (Complex.abs w) R
      hR : LT.lt 0 R
      hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
      ⊢ Filter.Eventually (fun t => Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi) …
    -/
  · exact Eventually.of_forall fun _ _ => (summable_geometric_of_lt_one hwR.1 hwR.2).mul_left _
    /-
      🎉 no goals
    -/
  · simpa only [tsum_mul_left, tsum_geometric_of_lt_one hwR.1 hwR.2] using
      hf.norm.mul_continuousOn continuousOn_const
    /-
      case refine_5
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      w : Complex
      hf : CircleIntegrable f c R
      hw : LT.lt (Complex.abs w) R
      hR : LT.lt 0 R
      hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
      ⊢ Filter.Eventually (fun t => Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi) …
    -/
  · refine Eventually.of_forall fun θ _ => HasSum.const_smul _ ?_
    /-
      case refine_5
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      w : Complex
      hf : CircleIntegrable f c R
      hw : LT.lt (Complex.abs w) R
      hR : LT.lt 0 R
      hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
      θ : Real
      x✝ : Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi)) θ
      ⊢ HasSum (fun n => (fun z => HSMul.hSMul (HPow.hPow (HDiv.hDiv w (HSub.hSub z  …
    -/
    simp only [smul_smul]
    /-
      case refine_5
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      w : Complex
      hf : CircleIntegrable f c R
      hw : LT.lt (Complex.abs w) R
      hR : LT.lt 0 R
      hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
      θ : Real
      x✝ : Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi)) θ
      ⊢ HasSum (fun n => HSMul.hSMul (HMul.hMul (HPow.hPow (HDiv.hDiv w (HSub.hSub ( …
    -/
    refine HasSum.smul_const ?_ _
    /-
      case refine_5
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      w : Complex
      hf : CircleIntegrable f c R
      hw : LT.lt (Complex.abs w) R
      hR : LT.lt 0 R
      hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
      θ : Real
      x✝ : Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi)) θ
      ⊢ HasSum (fun n => HMul.hMul (HPow.hPow (HDiv.hDiv w (HSub.hSub (circleMap c R …
    -/
    have : ‖w / (circleMap c R θ - c)‖ < 1 := by simpa [abs_of_pos hR] using hwR.2
    /-
      case refine_5
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      w : Complex
      hf : CircleIntegrable f c R
      hw : LT.lt (Complex.abs w) R
      hR : LT.lt 0 R
      hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
      θ : Real
      x✝ : Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi)) θ
      this : LT.lt (Norm.norm (HDiv.hDiv w (HSub.hSub (circleMap c R θ) c))) 1
      ⊢ HasSum (fun n => HMul.hMul (HPow.hPow (HDiv.hDiv w (HSub.hSub (circleMap c R …
    -/
    convert (hasSum_geometric_of_norm_lt_one this).mul_right _ using 1
    /-
      case h.e'_6
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      c : Complex
      R : Real
      w : Complex
      hf : CircleIntegrable f c R
      hw : LT.lt (Complex.abs w) R
      hR : LT.lt 0 R
      hwR : Membership.mem (Set.Ico 0 1) (HDiv.hDiv (Complex.abs w) R)
      θ : Real
      x✝ : Membership.mem (Set.uIoc 0 (HMul.hMul 2 Real.pi)) θ
      this : LT.lt (Norm.norm (HDiv.hDiv w (HSub.hSub (circleMap c R θ) c))) 1
      ⊢ Eq (Inv.inv (HSub.hSub (circleMap c R θ) (HAdd.hAdd c w))) (HMul.hMul (Inv.i …
    -/
    simp [← sub_sub, ← mul_inv, sub_mul, div_mul_cancel₀ _ (circleMap_ne_center hR.ne')]
    /-
      🎉 no goals
    -/


/-- For any circle integrable function `f`, the power series `cauchyPowerSeries f c R`, `R > 0`,
converges to the Cauchy integral `(2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - w)⁻¹ • f z` on the open
disc `Metric.ball c R`. -/
theorem hasSum_cauchyPowerSeries_integral {f : ℂ → E} {c : ℂ} {R : ℝ} {w : ℂ}
    (hf : CircleIntegrable f c R) (hw : abs w < R) :
    HasSum (fun n => cauchyPowerSeries f c R n fun _ => w)
      ((2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - (c + w))⁻¹ • f z) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R : Real
    w : Complex
    hf : CircleIntegrable f c R
    hw : LT.lt (Complex.abs w) R
    ⊢ HasSum (fun n => (cauchyPowerSeries f c R n) fun x => w) (HSMul.hSMul (Inv.i …
  -/
  simp only [cauchyPowerSeries_apply]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    c : Complex
    R : Real
    w : Complex
    hf : CircleIntegrable f c R
    hw : LT.lt (Complex.abs w) R
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comp …
  -/
  exact (hasSum_two_pi_I_cauchyPowerSeries_integral hf hw).const_smul _
  /-
    🎉 no goals
  -/


/-- For any circle integrable function `f`, the power series `cauchyPowerSeries f c R`, `R > 0`,
converges to the Cauchy integral `(2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - w)⁻¹ • f z` on the open
disc `Metric.ball c R`. -/
theorem sum_cauchyPowerSeries_eq_integral {f : ℂ → E} {c : ℂ} {R : ℝ} {w : ℂ}
    (hf : CircleIntegrable f c R) (hw : abs w < R) :
    (cauchyPowerSeries f c R).sum w = (2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - (c + w))⁻¹ • f z :=
  (hasSum_cauchyPowerSeries_integral hf hw).tsum_eq


/-- For any circle integrable function `f`, the power series `cauchyPowerSeries f c R`, `R > 0`,
converges to the Cauchy integral `(2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - w)⁻¹ • f z` on the open
disc `Metric.ball c R`. -/
theorem hasFPowerSeriesOn_cauchy_integral {f : ℂ → E} {c : ℂ} {R : ℝ≥0}
    (hf : CircleIntegrable f c R) (hR : 0 < R) :
    HasFPowerSeriesOnBall (fun w => (2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - w)⁻¹ • f z)
      (cauchyPowerSeries f c R) c R :=
  { r_le := le_radius_cauchyPowerSeries _ _ _
    r_pos := ENNReal.coe_pos.2 hR
                                                                  /-
                                                                    E : Type u_1
                                                                    inst✝¹ : NormedAddCommGroup E
                                                                    inst✝ : NormedSpace Complex E
                                                                    f : Complex → E
                                                                    c : Complex
                                                                    R : NNReal
                                                                    hf : CircleIntegrable f c ↑R
                                                                    hR : LT.lt 0 R
                                                                    y✝ : Complex
                                                                    hy : Membership.mem (EMetric.ball 0 ↑R) y✝
                                                                    ⊢ LT.lt (Complex.abs y✝) ↑R
                                                                  -/
    hasSum := fun hy ↦ hasSum_cauchyPowerSeries_integral hf <| by simpa using hy }
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Integral $\oint_{|z-c|=R} \frac{dz}{z-w} = 2πi$ whenever $|w-c| < R$. -/
theorem integral_sub_inv_of_mem_ball {c w : ℂ} {R : ℝ} (hw : w ∈ ball c R) :
    (∮ z in C(c, R), (z - w)⁻¹) = 2 * π * I := by
  /-
    c w : Complex
    R : Real
    hw : Membership.mem (Metric.ball c R) w
    ⊢ Eq (circleIntegral (fun z => Inv.inv (HSub.hSub z w)) c R) (HMul.hMul (HMul. …
  -/
  have hR : 0 < R := dist_nonneg.trans_lt hw
  suffices H : HasSum (fun n : ℕ => ∮ z in C(c, R), ((w - c) / (z - c)) ^ n * (z - c)⁻¹)
      (2 * π * I) by
    have A : CircleIntegrable (fun _ => (1 : ℂ)) c R := continuousOn_const.circleIntegrable'
    refine (H.unique ?_).symm
    simpa only [smul_eq_mul, mul_one, add_sub_cancel] using
      hasSum_two_pi_I_cauchyPowerSeries_integral A hw
  have H : ∀ n : ℕ, n ≠ 0 → (∮ z in C(c, R), (z - c) ^ (-n - 1 : ℤ)) = 0 := by
    refine fun n hn => integral_sub_zpow_of_ne ?_ _ _ _; simpa
  /-
    c w : Complex
    R : Real
    hw : Membership.mem (Metric.ball c R) w
    hR : LT.lt 0 R
    H : ∀ (n : Nat), Ne n 0 → Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z  …
    ⊢ HasSum (fun n => circleIntegral (fun z => HMul.hMul (HPow.hPow (HDiv.hDiv (H …
  -/
  have : (∮ z in C(c, R), ((w - c) / (z - c)) ^ 0 * (z - c)⁻¹) = 2 * π * I := by simp [hR.ne']
  /-
    c w : Complex
    R : Real
    hw : Membership.mem (Metric.ball c R) w
    hR : LT.lt 0 R
    H : ∀ (n : Nat), Ne n 0 → Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z  …
    this : Eq (circleIntegral (fun z => HMul.hMul (HPow.hPow (HDiv.hDiv (HSub.hSub …
    ⊢ HasSum (fun n => circleIntegral (fun z => HMul.hMul (HPow.hPow (HDiv.hDiv (H …
  -/
  refine this ▸ hasSum_single _ fun n hn => ?_
  /-
    c w : Complex
    R : Real
    hw : Membership.mem (Metric.ball c R) w
    hR : LT.lt 0 R
    H : ∀ (n : Nat), Ne n 0 → Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z  …
    this : Eq (circleIntegral (fun z => HMul.hMul (HPow.hPow (HDiv.hDiv (HSub.hSub …
    n : Nat
    hn : Ne n 0
    ⊢ Eq (circleIntegral (fun z => HMul.hMul (HPow.hPow (HDiv.hDiv (HSub.hSub w c) …
  -/
  simp only [div_eq_mul_inv, mul_pow, integral_const_mul, mul_assoc]
  /-
    c w : Complex
    R : Real
    hw : Membership.mem (Metric.ball c R) w
    hR : LT.lt 0 R
    H : ∀ (n : Nat), Ne n 0 → Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z  …
    this : Eq (circleIntegral (fun z => HMul.hMul (HPow.hPow (HDiv.hDiv (HSub.hSub …
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub w c) n) (circleIntegral (fun z => HMul.h …
  -/
  rw [(integral_congr hR.le fun z hz => _).trans (H n hn), mul_zero]
  /-
    c w : Complex
    R : Real
    hw : Membership.mem (Metric.ball c R) w
    hR : LT.lt 0 R
    H : ∀ (n : Nat), Ne n 0 → Eq (circleIntegral (fun z => HPow.hPow (HSub.hSub z  …
    this : Eq (circleIntegral (fun z => HMul.hMul (HPow.hPow (HDiv.hDiv (HSub.hSub …
    n : Nat
    hn : Ne n 0
    ⊢ ∀ (z : Complex), Membership.mem (Metric.sphere c R) z → Eq (HMul.hMul (HPow. …
  -/
  intro z _
  rw [← pow_succ, ← zpow_natCast, inv_zpow, ← zpow_neg, Int.ofNat_succ, neg_add,
    sub_eq_add_neg _ (1 : ℤ)]


