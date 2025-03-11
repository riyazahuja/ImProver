@[fun_prop]
lemma continuous_mul_log : Continuous fun x ↦ x * log x := by
  /-
    ⊢ Continuous fun x => HMul.hMul x (Real.log x)
  -/
  rw [continuous_iff_continuousAt]
  /-
    ⊢ ∀ (x : Real), ContinuousAt (fun x => HMul.hMul x (Real.log x)) x
  -/
  intro x
  /-
    x : Real
    ⊢ ContinuousAt (fun x => HMul.hMul x (Real.log x)) x
  -/
  obtain hx | rfl := ne_or_eq x 0
    /-
      case inl
      x : Real
      hx : Ne x 0
      ⊢ ContinuousAt (fun x => HMul.hMul x (Real.log x)) x
    -/
  · exact (continuous_id'.continuousAt).mul (continuousAt_log hx)
    /-
      🎉 no goals
    -/
  /-
    case inr
    ⊢ ContinuousAt (fun x => HMul.hMul x (Real.log x)) 0
  -/
  rw [ContinuousAt, zero_mul]
  /-
    case inr
    ⊢ Filter.Tendsto (fun x => HMul.hMul x (Real.log x)) (nhds 0) (nhds 0)
  -/
  simp_rw [mul_comm _ (log _)]
  /-
    case inr
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhds 0) (nhds 0)
  -/
  nth_rewrite 1 [← nhdsWithin_univ]
  /-
    case inr
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 Set.univ) ( …
  -/
  have : (Set.univ : Set ℝ) = Set.Iio 0 ∪ Set.Ioi 0 ∪ {0} := by ext; simp [em]
  /-
    case inr
    this : Eq Set.univ (Union.union (Union.union (Set.Iio 0) (Set.Ioi 0)) (Singlet …
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 Set.univ) ( …
  -/
  rw [this, nhdsWithin_union, nhdsWithin_union]
  /-
    case inr
    this : Eq Set.univ (Union.union (Union.union (Set.Iio 0) (Set.Ioi 0)) (Singlet …
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (Max.max (Max.max (nhdsWi …
  -/
  simp only [nhdsWithin_singleton, sup_le_iff, Filter.nonpos_iff, Filter.tendsto_sup]
  /-
    case inr
    this : Eq Set.univ (Union.union (Union.union (Set.Iio 0) (Set.Ioi 0)) (Singlet …
    ⊢ And (And (Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 ( …
  -/
  refine ⟨⟨tendsto_log_mul_self_nhds_zero_left, ?_⟩, ?_⟩
    /-
      case inr.refine_1
      this : Eq Set.univ (Union.union (Union.union (Set.Iio 0) (Set.Ioi 0)) (Singlet …
      ⊢ Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 (Set.Ioi 0) …
    -/
  · simpa only [rpow_one] using tendsto_log_mul_rpow_nhds_zero zero_lt_one
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      this : Eq Set.univ (Union.union (Union.union (Set.Iio 0) (Set.Ioi 0)) (Singlet …
      ⊢ Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (Pure.pure 0) (nhds 0)
    -/
  · convert tendsto_pure_nhds (fun x ↦ log x * x) 0
    /-
      case h.e'_5.h.e'_3
      this : Eq Set.univ (Union.union (Union.union (Set.Iio 0) (Set.Ioi 0)) (Singlet …
      ⊢ Eq 0 (HMul.hMul (Real.log 0) 0)
    -/
    simp
    /-
      🎉 no goals
    -/


@[fun_prop]
lemma Continuous.mul_log {α : Type*} [TopologicalSpace α] {f : α → ℝ} (hf : Continuous f) :
    Continuous fun a ↦ f a * log (f a) := continuous_mul_log.comp hf


lemma differentiableOn_mul_log : DifferentiableOn ℝ (fun x ↦ x * log x) {0}ᶜ :=
  differentiable_id'.differentiableOn.mul differentiableOn_log


lemma deriv_mul_log {x : ℝ} (hx : x ≠ 0) : deriv (fun x ↦ x * log x) x = log x + 1 := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ Eq (deriv (fun x => HMul.hMul x (Real.log x)) x) (HAdd.hAdd (Real.log x) 1)
  -/
  rw [deriv_mul differentiableAt_id' (differentiableAt_log hx)]
  /-
    x : Real
    hx : Ne x 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (deriv (fun x => x) x) (Real.log x)) (HMul.hMul x ( …
  -/
  simp only [deriv_id'', one_mul, deriv_log', ne_eq, add_right_inj]
  /-
    x : Real
    hx : Ne x 0
    ⊢ Eq (HMul.hMul x (Inv.inv x)) 1
  -/
  exact mul_inv_cancel₀ hx
  /-
    🎉 no goals
  -/


lemma hasDerivAt_mul_log {x : ℝ} (hx : x ≠ 0) : HasDerivAt (fun x ↦ x * log x) (log x + 1) x := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ HasDerivAt (fun x => HMul.hMul x (Real.log x)) (HAdd.hAdd (Real.log x) 1) x
  -/
  rw [← deriv_mul_log hx, hasDerivAt_deriv_iff]
  /-
    x : Real
    hx : Ne x 0
    ⊢ DifferentiableAt Real (fun x => HMul.hMul x (Real.log x)) x
  -/
  refine DifferentiableOn.differentiableAt differentiableOn_mul_log ?_
  /-
    x : Real
    hx : Ne x 0
    ⊢ Membership.mem (nhds x) (HasCompl.compl (Singleton.singleton 0))
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


open Filter in
private lemma tendsto_deriv_mul_log_nhdsWithin_zero :
    Tendsto (deriv (fun x ↦ x * log x)) (𝓝[>] 0) atBot := by
  have : (deriv (fun x ↦ x * log x)) =ᶠ[𝓝[>] 0] (fun x ↦ log x + 1) := by
    apply eventuallyEq_nhdsWithin_of_eqOn
    intro x hx
    rw [Set.mem_Ioi] at hx
    exact deriv_mul_log hx.ne'
  /-
    this : (nhdsWithin 0 (Set.Ioi 0)).EventuallyEq (deriv fun x => HMul.hMul x (Re …
    ⊢ Filter.Tendsto (deriv fun x => HMul.hMul x (Real.log x)) (nhdsWithin 0 (Set. …
  -/
  simp only [tendsto_congr' this, tendsto_atBot_add_const_right, tendsto_log_nhdsWithin_zero_right]
  /-
    🎉 no goals
  -/


/-- At `x=0`, `(fun x ↦ x * log x)` is not differentiable
(but note that it is continuous, see `continuous_mul_log`). -/
lemma not_DifferentiableAt_log_mul_zero :
    ¬ DifferentiableAt ℝ (fun x ↦ x * log x) 0 := fun h ↦
  (not_differentiableWithinAt_of_deriv_tendsto_atBot_Ioi (fun x : ℝ ↦ x * log x) (a := 0))
    tendsto_deriv_mul_log_nhdsWithin_zero
    (h.differentiableWithinAt (s := Set.Ioi 0))


/-- Not differentiable, hence `deriv` has junk value zero. -/
lemma deriv_mul_log_zero : deriv (fun x ↦ x * log x) 0 = 0 :=
  deriv_zero_of_not_differentiableAt not_DifferentiableAt_log_mul_zero


lemma not_continuousAt_deriv_mul_log_zero :
    ¬ ContinuousAt (deriv (fun (x : ℝ) ↦ x * log x)) 0 :=
                                                                                           /-
                                                                                             ⊢ Disjoint (nhds (deriv (fun x => HMul.hMul x (Real.log x)) 0)) Filter.atBot
                                                                                           -/
  not_continuousAt_of_tendsto tendsto_deriv_mul_log_nhdsWithin_zero nhdsWithin_le_nhds (by simp)
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


lemma deriv2_mul_log (x : ℝ) : deriv^[2] (fun x ↦ x * log x) x = x⁻¹ := by
  /-
    x : Real
    ⊢ Eq (Nat.iterate deriv 2 (fun x => HMul.hMul x (Real.log x)) x) (Inv.inv x)
  -/
  simp only [Function.iterate_succ, Function.iterate_zero, Function.id_comp, Function.comp_apply]
  /-
    x : Real
    ⊢ Eq (deriv (deriv fun x => HMul.hMul x (Real.log x)) x) (Inv.inv x)
  -/
  by_cases hx : x = 0
    /-
      case pos
      x : Real
      hx : Eq x 0
      ⊢ Eq (deriv (deriv fun x => HMul.hMul x (Real.log x)) x) (Inv.inv x)
    -/
  · rw [hx, inv_zero]
    exact deriv_zero_of_not_differentiableAt
      (fun h ↦ not_continuousAt_deriv_mul_log_zero h.continuousAt)
  · suffices ∀ᶠ y in (𝓝 x), deriv (fun x ↦ x * log x) y = log y + 1 by
      refine (Filter.EventuallyEq.deriv_eq this).trans ?_
      rw [deriv_add_const, deriv_log x]
    /-
      case neg
      x : Real
      hx : Not (Eq x 0)
      ⊢ Filter.Eventually (fun y => Eq (deriv (fun x => HMul.hMul x (Real.log x)) y) …
    -/
    filter_upwards [eventually_ne_nhds hx] with y hy using deriv_mul_log hy
    /-
      🎉 no goals
    -/


lemma strictConvexOn_mul_log : StrictConvexOn ℝ (Set.Ici (0 : ℝ)) (fun x ↦ x * log x) := by
  /-
    ⊢ StrictConvexOn Real (Set.Ici 0) fun x => HMul.hMul x (Real.log x)
  -/
  refine strictConvexOn_of_deriv2_pos (convex_Ici 0) (continuous_mul_log.continuousOn) ?_
  /-
    ⊢ ∀ (x : Real), Membership.mem (interior (Set.Ici 0)) x → LT.lt 0 (Nat.iterate …
  -/
  intro x hx
  /-
    x : Real
    hx : Membership.mem (interior (Set.Ici 0)) x
    ⊢ LT.lt 0 (Nat.iterate deriv 2 (fun x => HMul.hMul x (Real.log x)) x)
  -/
  simp only [Set.nonempty_Iio, interior_Ici', Set.mem_Ioi] at hx
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ LT.lt 0 (Nat.iterate deriv 2 (fun x => HMul.hMul x (Real.log x)) x)
  -/
  rw [deriv2_mul_log]
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ LT.lt 0 (Inv.inv x)
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma convexOn_mul_log : ConvexOn ℝ (Set.Ici (0 : ℝ)) (fun x ↦ x * log x) :=
  strictConvexOn_mul_log.convexOn


lemma mul_log_nonneg {x : ℝ} (hx : 1 ≤ x) : 0 ≤ x * log x :=
  mul_nonneg (zero_le_one.trans hx) (log_nonneg hx)


lemma mul_log_nonpos {x : ℝ} (hx₀ : 0 ≤ x) (hx₁ : x ≤ 1) : x * log x ≤ 0 :=
  mul_nonpos_of_nonneg_of_nonpos hx₀ (log_nonpos hx₀ hx₁)


/-- The function `x ↦ - x * log x` from `ℝ` to `ℝ`. -/
noncomputable def negMulLog (x : ℝ) : ℝ := - x * log x


lemma negMulLog_def : negMulLog = fun x ↦ - x * log x := rfl


                                                                 /-
                                                                   ⊢ Eq Real.negMulLog fun x => Neg.neg (HMul.hMul x (Real.log x))
                                                                 -/
lemma negMulLog_eq_neg : negMulLog = fun x ↦ - (x * log x) := by simp [negMulLog_def]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                           /-
                                                             ⊢ Eq (Real.negMulLog 0) 0
                                                           -/
@[simp] lemma negMulLog_zero : negMulLog (0 : ℝ) = 0 := by simp [negMulLog]
                                                           /-
                                                             🎉 no goals
                                                           -/


                                                          /-
                                                            ⊢ Eq (Real.negMulLog 1) 0
                                                          -/
@[simp] lemma negMulLog_one : negMulLog (1 : ℝ) = 0 := by simp [negMulLog]
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma negMulLog_nonneg {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 1) : 0 ≤ negMulLog x := by
  /-
    x : Real
    h1 : LE.le 0 x
    h2 : LE.le x 1
    ⊢ LE.le 0 x.negMulLog
  -/
  simpa only [negMulLog_eq_neg, neg_nonneg] using mul_log_nonpos h1 h2
  /-
    🎉 no goals
  -/


lemma negMulLog_mul (x y : ℝ) : negMulLog (x * y) = y * negMulLog x + x * negMulLog y := by
  /-
    x y : Real
    ⊢ Eq (HMul.hMul x y).negMulLog (HAdd.hAdd (HMul.hMul y x.negMulLog) (HMul.hMul …
  -/
  simp only [negMulLog, neg_mul, neg_add_rev]
  /-
    x y : Real
    ⊢ Eq (Neg.neg (HMul.hMul (HMul.hMul x y) (Real.log (HMul.hMul x y)))) (HAdd.hA …
  -/
  by_cases hx : x = 0
    /-
      case pos
      x y : Real
      hx : Eq x 0
      ⊢ Eq (Neg.neg (HMul.hMul (HMul.hMul x y) (Real.log (HMul.hMul x y)))) (HAdd.hA …
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x y : Real
    hx : Not (Eq x 0)
    ⊢ Eq (Neg.neg (HMul.hMul (HMul.hMul x y) (Real.log (HMul.hMul x y)))) (HAdd.hA …
  -/
  by_cases hy : y = 0
    /-
      case pos
      x y : Real
      hx : Not (Eq x 0)
      hy : Eq y 0
      ⊢ Eq (Neg.neg (HMul.hMul (HMul.hMul x y) (Real.log (HMul.hMul x y)))) (HAdd.hA …
    -/
  · simp [hy]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x y : Real
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Eq (Neg.neg (HMul.hMul (HMul.hMul x y) (Real.log (HMul.hMul x y)))) (HAdd.hA …
  -/
  rw [log_mul hx hy]
  /-
    case neg
    x y : Real
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Eq (Neg.neg (HMul.hMul (HMul.hMul x y) (HAdd.hAdd (Real.log x) (Real.log y)) …
  -/
  ring
  /-
    🎉 no goals
  -/


@[fun_prop] lemma continuous_negMulLog : Continuous negMulLog := by
  /-
    ⊢ Continuous Real.negMulLog
  -/
  simpa only [negMulLog_eq_neg] using continuous_mul_log.neg
  /-
    🎉 no goals
  -/


lemma differentiableOn_negMulLog : DifferentiableOn ℝ negMulLog {0}ᶜ := by
  /-
    ⊢ DifferentiableOn Real Real.negMulLog (HasCompl.compl (Singleton.singleton 0))
  -/
  simpa only [negMulLog_eq_neg] using differentiableOn_mul_log.neg
  /-
    🎉 no goals
  -/


lemma differentiableAt_negMulLog_iff {x : ℝ} : DifferentiableAt ℝ negMulLog x ↔ x ≠ 0 := by
  /-
    x : Real
    ⊢ Iff (DifferentiableAt Real Real.negMulLog x) (Ne x 0)
  -/
  constructor
    /-
      case mp
      x : Real
      ⊢ DifferentiableAt Real Real.negMulLog x → Ne x 0
    -/
  · unfold negMulLog
    /-
      case mp
      x : Real
      ⊢ DifferentiableAt Real (fun x => HMul.hMul (Neg.neg x) (Real.log x)) x → Ne x 0
    -/
    intro h eq0
    /-
      case mp
      x : Real
      h : DifferentiableAt Real (fun x => HMul.hMul (Neg.neg x) (Real.log x)) x
      eq0 : Eq x 0
      ⊢ False
    -/
    simp only [neg_mul, differentiableAt_neg_iff, eq0] at h
    /-
      case mp
      x : Real
      eq0 : Eq x 0
      h : DifferentiableAt Real (fun y => HMul.hMul y (Real.log y)) 0
      ⊢ False
    -/
    exact not_DifferentiableAt_log_mul_zero h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x : Real
      ⊢ Ne x 0 → DifferentiableAt Real Real.negMulLog x
    -/
  · intro hx
    have : x ∈ ({0} : Set ℝ)ᶜ := by
      simp_all only [ne_eq, Set.mem_compl_iff, Set.mem_singleton_iff, not_false_eq_true]
    /-
      case mpr
      x : Real
      hx : Ne x 0
      this : Membership.mem (HasCompl.compl (Singleton.singleton 0)) x
      ⊢ DifferentiableAt Real Real.negMulLog x
    -/
    have := differentiableOn_negMulLog x this
    /-
      case mpr
      x : Real
      hx : Ne x 0
      this✝ : Membership.mem (HasCompl.compl (Singleton.singleton 0)) x
      this : DifferentiableWithinAt Real Real.negMulLog (HasCompl.compl (Singleton.s …
      ⊢ DifferentiableAt Real Real.negMulLog x
    -/
    apply DifferentiableWithinAt.differentiableAt (s := {0}ᶜ) <;>
    simp_all only [ne_eq, Set.mem_compl_iff, Set.mem_singleton_iff, not_false_eq_true,
      compl_singleton_mem_nhds_iff]


@[fun_prop] alias ⟨_, differentiableAt_negMulLog⟩ := differentiableAt_negMulLog_iff


lemma deriv_negMulLog {x : ℝ} (hx : x ≠ 0) : deriv negMulLog x = - log x - 1 := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ Eq (deriv Real.negMulLog x) (HSub.hSub (Neg.neg (Real.log x)) 1)
  -/
  rw [negMulLog_eq_neg, deriv.neg, deriv_mul_log hx]
  /-
    x : Real
    hx : Ne x 0
    ⊢ Eq (Neg.neg (HAdd.hAdd (Real.log x) 1)) (HSub.hSub (Neg.neg (Real.log x)) 1)
  -/
  ring
  /-
    🎉 no goals
  -/


lemma hasDerivAt_negMulLog {x : ℝ} (hx : x ≠ 0) : HasDerivAt negMulLog (- log x - 1) x := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ HasDerivAt Real.negMulLog (HSub.hSub (Neg.neg (Real.log x)) 1) x
  -/
  rw [← deriv_negMulLog hx, hasDerivAt_deriv_iff]
  /-
    x : Real
    hx : Ne x 0
    ⊢ DifferentiableAt Real Real.negMulLog x
  -/
  refine DifferentiableOn.differentiableAt differentiableOn_negMulLog ?_
  /-
    x : Real
    hx : Ne x 0
    ⊢ Membership.mem (nhds x) (HasCompl.compl (Singleton.singleton 0))
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


lemma deriv2_negMulLog (x : ℝ) : deriv^[2] negMulLog x = - x⁻¹ := by
  /-
    x : Real
    ⊢ Eq (Nat.iterate deriv 2 Real.negMulLog x) (Neg.neg (Inv.inv x))
  -/
  rw [negMulLog_eq_neg]
  /-
    x : Real
    ⊢ Eq (Nat.iterate deriv 2 (fun x => Neg.neg (HMul.hMul x (Real.log x))) x) (Ne …
  -/
  have h := deriv2_mul_log
  simp only [Function.iterate_succ, Function.iterate_zero, Function.id_comp,
    Function.comp_apply, deriv.neg', differentiableAt_id', differentiableAt_log_iff, ne_eq] at h ⊢
  /-
    x : Real
    h : ∀ (x : Real), Eq (deriv (deriv fun x => HMul.hMul x (Real.log x)) x) (Inv. …
    ⊢ Eq (Neg.neg (deriv (deriv fun y => HMul.hMul y (Real.log y)) x)) (Neg.neg (I …
  -/
  rw [h]
  /-
    🎉 no goals
  -/


lemma strictConcaveOn_negMulLog : StrictConcaveOn ℝ (Set.Ici (0 : ℝ)) negMulLog := by
  /-
    ⊢ StrictConcaveOn Real (Set.Ici 0) Real.negMulLog
  -/
  simpa only [negMulLog_eq_neg] using strictConvexOn_mul_log.neg
  /-
    🎉 no goals
  -/


lemma concaveOn_negMulLog : ConcaveOn ℝ (Set.Ici (0 : ℝ)) negMulLog :=
  strictConcaveOn_negMulLog.concaveOn


