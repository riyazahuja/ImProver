lemma IsIntegralCurveOn.comp_add (hγ : IsIntegralCurveOn γ v s) (dt : ℝ) :
    IsIntegralCurveOn (γ ∘ (· + dt)) v (-dt +ᵥ s) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    dt : Real
    ⊢ IsIntegralCurveOn (Function.comp γ fun x => HAdd.hAdd x dt) v (HVAdd.hVAdd ( …
  -/
  intros t ht
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    dt t : Real
    ht : Membership.mem (HVAdd.hVAdd (Neg.neg dt) s) t
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (Function.comp γ fun x => HA …
  -/
  rw [comp_apply, ← ContinuousLinearMap.comp_id (ContinuousLinearMap.smulRight 1 (v (γ (t + dt))))]
  rw [mem_vadd_set_iff_neg_vadd_mem, neg_neg, vadd_eq_add, add_comm,
    ← mem_setOf (p := fun t ↦ t + dt ∈ s)] at ht
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    dt t : Real
    ht : Membership.mem (setOf fun x => Membership.mem s (HAdd.hAdd x dt)) t
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (Function.comp γ fun x => HA …
  -/
  apply HasMFDerivAt.comp t (hγ (t + dt) ht)
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    dt t : Real
    ht : Membership.mem (setOf fun x => Membership.mem s (HAdd.hAdd x dt)) t
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) (modelWithCornersSelf Real Rea …
  -/
  refine ⟨(continuous_add_right _).continuousAt, ?_⟩
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    dt t : Real
    ht : Membership.mem (setOf fun x => Membership.mem s (HAdd.hAdd x dt)) t
    ⊢ HasFDerivWithinAt (writtenInExtChartAt (modelWithCornersSelf Real Real) (mod …
  -/
  simp only [mfld_simps, hasFDerivWithinAt_univ]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    dt t : Real
    ht : Membership.mem (setOf fun x => Membership.mem s (HAdd.hAdd x dt)) t
    ⊢ HasFDerivAt (fun x => HAdd.hAdd x dt) (ContinuousLinearMap.id Real Real) t
  -/
  exact HasFDerivAt.add_const (hasFDerivAt_id _) _
  /-
    🎉 no goals
  -/


lemma isIntegralCurveOn_comp_add {dt : ℝ} :
    IsIntegralCurveOn γ v s ↔ IsIntegralCurveOn (γ ∘ (· + dt)) v (-dt +ᵥ s) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    dt : Real
    ⊢ Iff (IsIntegralCurveOn γ v s) (IsIntegralCurveOn (Function.comp γ fun x => H …
  -/
  refine ⟨fun hγ ↦ hγ.comp_add _, fun hγ ↦ ?_⟩
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    dt : Real
    hγ : IsIntegralCurveOn (Function.comp γ fun x => HAdd.hAdd x dt) v (HVAdd.hVAd …
    ⊢ IsIntegralCurveOn γ v s
  -/
  convert hγ.comp_add (-dt)
    /-
      case h.e'_10
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      s : Set Real
      dt : Real
      hγ : IsIntegralCurveOn (Function.comp γ fun x => HAdd.hAdd x dt) v (HVAdd.hVAd …
      ⊢ Eq γ (Function.comp (Function.comp γ fun x => HAdd.hAdd x dt) fun x => HAdd. …
    -/
  · ext t
    /-
      case h.e'_10.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      s : Set Real
      dt : Real
      hγ : IsIntegralCurveOn (Function.comp γ fun x => HAdd.hAdd x dt) v (HVAdd.hVAd …
      t : Real
      ⊢ Eq (γ t) (Function.comp (Function.comp γ fun x => HAdd.hAdd x dt) (fun x =>  …
    -/
    simp only [Function.comp_apply, neg_add_cancel_right]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_12
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      s : Set Real
      dt : Real
      hγ : IsIntegralCurveOn (Function.comp γ fun x => HAdd.hAdd x dt) v (HVAdd.hVAd …
      ⊢ Eq s (HVAdd.hVAdd (Neg.neg (Neg.neg dt)) (HVAdd.hVAdd (Neg.neg dt) s))
    -/
  · simp only [neg_neg, vadd_neg_vadd]
    /-
      🎉 no goals
    -/


lemma isIntegralCurveOn_comp_sub {dt : ℝ} :
    IsIntegralCurveOn γ v s ↔ IsIntegralCurveOn (γ ∘ (· - dt)) v (dt +ᵥ s) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    dt : Real
    ⊢ Iff (IsIntegralCurveOn γ v s) (IsIntegralCurveOn (Function.comp γ fun x => H …
  -/
  simpa using isIntegralCurveOn_comp_add (dt := -dt)
  /-
    🎉 no goals
  -/


lemma IsIntegralCurveAt.comp_add (hγ : IsIntegralCurveAt γ v t₀) (dt : ℝ) :
    IsIntegralCurveAt (γ ∘ (· + dt)) v (t₀ - dt) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    hγ : IsIntegralCurveAt γ v t₀
    dt : Real
    ⊢ IsIntegralCurveAt (Function.comp γ fun x => HAdd.hAdd x dt) v (HSub.hSub t₀  …
  -/
  rw [isIntegralCurveAt_iff'] at *
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    hγ : Exists fun ε => And (GT.gt ε 0) (IsIntegralCurveOn γ v (Metric.ball t₀ ε))
    dt : Real
    ⊢ Exists fun ε => And (GT.gt ε 0) (IsIntegralCurveOn (Function.comp γ fun x => …
  -/
  obtain ⟨ε, hε, h⟩ := hγ
  /-
    case intro.intro
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ dt ε : Real
    hε : GT.gt ε 0
    h : IsIntegralCurveOn γ v (Metric.ball t₀ ε)
    ⊢ Exists fun ε => And (GT.gt ε 0) (IsIntegralCurveOn (Function.comp γ fun x => …
  -/
  refine ⟨ε, hε, ?_⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ dt ε : Real
    hε : GT.gt ε 0
    h : IsIntegralCurveOn γ v (Metric.ball t₀ ε)
    ⊢ IsIntegralCurveOn (Function.comp γ fun x => HAdd.hAdd x dt) v (Metric.ball ( …
  -/
  convert h.comp_add dt
  /-
    case h.e'_12
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ dt ε : Real
    hε : GT.gt ε 0
    h : IsIntegralCurveOn γ v (Metric.ball t₀ ε)
    ⊢ Eq (Metric.ball (HSub.hSub t₀ dt) ε) (HVAdd.hVAdd (Neg.neg dt) (Metric.ball  …
  -/
  rw [Metric.vadd_ball, vadd_eq_add, neg_add_eq_sub]
  /-
    🎉 no goals
  -/


lemma isIntegralCurveAt_comp_add {dt : ℝ} :
    IsIntegralCurveAt γ v t₀ ↔ IsIntegralCurveAt (γ ∘ (· + dt)) v (t₀ - dt) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ dt : Real
    ⊢ Iff (IsIntegralCurveAt γ v t₀) (IsIntegralCurveAt (Function.comp γ fun x =>  …
  -/
  refine ⟨fun hγ ↦ hγ.comp_add _, fun hγ ↦ ?_⟩
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ dt : Real
    hγ : IsIntegralCurveAt (Function.comp γ fun x => HAdd.hAdd x dt) v (HSub.hSub  …
    ⊢ IsIntegralCurveAt γ v t₀
  -/
  convert hγ.comp_add (-dt)
    /-
      case h.e'_10
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      t₀ dt : Real
      hγ : IsIntegralCurveAt (Function.comp γ fun x => HAdd.hAdd x dt) v (HSub.hSub  …
      ⊢ Eq γ (Function.comp (Function.comp γ fun x => HAdd.hAdd x dt) fun x => HAdd. …
    -/
  · ext t
    /-
      case h.e'_10.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      t₀ dt : Real
      hγ : IsIntegralCurveAt (Function.comp γ fun x => HAdd.hAdd x dt) v (HSub.hSub  …
      t : Real
      ⊢ Eq (γ t) (Function.comp (Function.comp γ fun x => HAdd.hAdd x dt) (fun x =>  …
    -/
    simp only [Function.comp_apply, neg_add_cancel_right]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_12
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      t₀ dt : Real
      hγ : IsIntegralCurveAt (Function.comp γ fun x => HAdd.hAdd x dt) v (HSub.hSub  …
      ⊢ Eq t₀ (HSub.hSub (HSub.hSub t₀ dt) (Neg.neg dt))
    -/
  · simp only [sub_neg_eq_add, sub_add_cancel]
    /-
      🎉 no goals
    -/


lemma isIntegralCurveAt_comp_sub {dt : ℝ} :
    IsIntegralCurveAt γ v t₀ ↔ IsIntegralCurveAt (γ ∘ (· - dt)) v (t₀ + dt) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ dt : Real
    ⊢ Iff (IsIntegralCurveAt γ v t₀) (IsIntegralCurveAt (Function.comp γ fun x =>  …
  -/
  simpa using isIntegralCurveAt_comp_add (dt := -dt)
  /-
    🎉 no goals
  -/


lemma IsIntegralCurve.comp_add (hγ : IsIntegralCurve γ v) (dt : ℝ) :
    IsIntegralCurve (γ ∘ (· + dt)) v := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    hγ : IsIntegralCurve γ v
    dt : Real
    ⊢ IsIntegralCurve (Function.comp γ fun x => HAdd.hAdd x dt) v
  -/
  rw [isIntegralCurve_iff_isIntegralCurveOn] at *
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    hγ : IsIntegralCurveOn γ v Set.univ
    dt : Real
    ⊢ IsIntegralCurveOn (Function.comp γ fun x => HAdd.hAdd x dt) v Set.univ
  -/
  simpa using hγ.comp_add dt
  /-
    🎉 no goals
  -/


lemma isIntegralCurve_comp_add {dt : ℝ} :
    IsIntegralCurve γ v ↔ IsIntegralCurve (γ ∘ (· + dt)) v := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    dt : Real
    ⊢ Iff (IsIntegralCurve γ v) (IsIntegralCurve (Function.comp γ fun x => HAdd.hA …
  -/
  refine ⟨fun hγ ↦ hγ.comp_add _, fun hγ ↦ ?_⟩
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    dt : Real
    hγ : IsIntegralCurve (Function.comp γ fun x => HAdd.hAdd x dt) v
    ⊢ IsIntegralCurve γ v
  -/
  convert hγ.comp_add (-dt)
  /-
    case h.e'_10
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    dt : Real
    hγ : IsIntegralCurve (Function.comp γ fun x => HAdd.hAdd x dt) v
    ⊢ Eq γ (Function.comp (Function.comp γ fun x => HAdd.hAdd x dt) fun x => HAdd. …
  -/
  ext t
  /-
    case h.e'_10.h
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    dt : Real
    hγ : IsIntegralCurve (Function.comp γ fun x => HAdd.hAdd x dt) v
    t : Real
    ⊢ Eq (γ t) (Function.comp (Function.comp γ fun x => HAdd.hAdd x dt) (fun x =>  …
  -/
  simp only [Function.comp_apply, neg_add_cancel_right]
  /-
    🎉 no goals
  -/


lemma isIntegralCurve_comp_sub {dt : ℝ} :
    IsIntegralCurve γ v ↔ IsIntegralCurve (γ ∘ (· - dt)) v := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    dt : Real
    ⊢ Iff (IsIntegralCurve γ v) (IsIntegralCurve (Function.comp γ fun x => HSub.hS …
  -/
  simpa using isIntegralCurve_comp_add (dt := -dt)
  /-
    🎉 no goals
  -/


lemma IsIntegralCurveOn.comp_mul (hγ : IsIntegralCurveOn γ v s) (a : ℝ) :
    IsIntegralCurveOn (γ ∘ (· * a)) (a • v) { t | t * a ∈ s } := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    a : Real
    ⊢ IsIntegralCurveOn (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a v) …
  -/
  intros t ht
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    a t : Real
    ht : Membership.mem (setOf fun t => Membership.mem s (HMul.hMul t a)) t
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (Function.comp γ fun x => HM …
  -/
  rw [comp_apply, Pi.smul_apply, ← ContinuousLinearMap.smulRight_comp]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    a t : Real
    ht : Membership.mem (setOf fun t => Membership.mem s (HMul.hMul t a)) t
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (Function.comp γ fun x => HM …
  -/
  refine HasMFDerivAt.comp t (hγ (t * a) ht) ⟨(continuous_mul_right _).continuousAt, ?_⟩
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    a t : Real
    ht : Membership.mem (setOf fun t => Membership.mem s (HMul.hMul t a)) t
    ⊢ HasFDerivWithinAt (writtenInExtChartAt (modelWithCornersSelf Real Real) (mod …
  -/
  simp only [mfld_simps, hasFDerivWithinAt_univ]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    hγ : IsIntegralCurveOn γ v s
    a t : Real
    ht : Membership.mem (setOf fun t => Membership.mem s (HMul.hMul t a)) t
    ⊢ HasFDerivAt (fun x => HMul.hMul x a) (ContinuousLinearMap.smulRight 1 a) t
  -/
  exact HasFDerivAt.mul_const' (hasFDerivAt_id _) _
  /-
    🎉 no goals
  -/


lemma isIntegralCurveOn_comp_mul_ne_zero {a : ℝ} (ha : a ≠ 0) :
    IsIntegralCurveOn γ v s ↔ IsIntegralCurveOn (γ ∘ (· * a)) (a • v) { t | t * a ∈ s } := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    a : Real
    ha : Ne a 0
    ⊢ Iff (IsIntegralCurveOn γ v s) (IsIntegralCurveOn (Function.comp γ fun x => H …
  -/
  refine ⟨fun hγ ↦ hγ.comp_mul a, fun hγ ↦ ?_⟩
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    a : Real
    ha : Ne a 0
    hγ : IsIntegralCurveOn (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a …
    ⊢ IsIntegralCurveOn γ v s
  -/
  convert hγ.comp_mul a⁻¹
    /-
      case h.e'_10
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      s : Set Real
      a : Real
      ha : Ne a 0
      hγ : IsIntegralCurveOn (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a …
      ⊢ Eq γ (Function.comp (Function.comp γ fun x => HMul.hMul x a) fun x => HMul.h …
    -/
  · ext t
    /-
      case h.e'_10.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      s : Set Real
      a : Real
      ha : Ne a 0
      hγ : IsIntegralCurveOn (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a …
      t : Real
      ⊢ Eq (γ t) (Function.comp (Function.comp γ fun x => HMul.hMul x a) (fun x => H …
    -/
    simp only [Function.comp_apply, mul_assoc, inv_mul_eq_div, div_self ha, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_11
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      s : Set Real
      a : Real
      ha : Ne a 0
      hγ : IsIntegralCurveOn (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a …
      ⊢ Eq v (HSMul.hSMul (Inv.inv a) (HSMul.hSMul a v))
    -/
  · simp only [smul_smul, inv_mul_eq_div, div_self ha, one_smul]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_12
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      s : Set Real
      a : Real
      ha : Ne a 0
      hγ : IsIntegralCurveOn (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a …
      ⊢ Eq s (setOf fun t => Membership.mem (setOf fun t => Membership.mem s (HMul.h …
    -/
  · simp only [mem_setOf_eq, mul_assoc, inv_mul_eq_div, div_self ha, mul_one, setOf_mem_eq]
    /-
      🎉 no goals
    -/


lemma IsIntegralCurveAt.comp_mul_ne_zero (hγ : IsIntegralCurveAt γ v t₀) {a : ℝ} (ha : a ≠ 0) :
    IsIntegralCurveAt (γ ∘ (· * a)) (a • v) (t₀ / a) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    hγ : IsIntegralCurveAt γ v t₀
    a : Real
    ha : Ne a 0
    ⊢ IsIntegralCurveAt (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a v) …
  -/
  rw [isIntegralCurveAt_iff'] at *
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    hγ : Exists fun ε => And (GT.gt ε 0) (IsIntegralCurveOn γ v (Metric.ball t₀ ε))
    a : Real
    ha : Ne a 0
    ⊢ Exists fun ε => And (GT.gt ε 0) (IsIntegralCurveOn (Function.comp γ fun x => …
  -/
  obtain ⟨ε, hε, h⟩ := hγ
  /-
    case intro.intro
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ a : Real
    ha : Ne a 0
    ε : Real
    hε : GT.gt ε 0
    h : IsIntegralCurveOn γ v (Metric.ball t₀ ε)
    ⊢ Exists fun ε => And (GT.gt ε 0) (IsIntegralCurveOn (Function.comp γ fun x => …
  -/
  refine ⟨ε / |a|, by positivity, ?_⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ a : Real
    ha : Ne a 0
    ε : Real
    hε : GT.gt ε 0
    h : IsIntegralCurveOn γ v (Metric.ball t₀ ε)
    ⊢ IsIntegralCurveOn (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a v) …
  -/
  convert h.comp_mul a
  /-
    case h.e'_12
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ a : Real
    ha : Ne a 0
    ε : Real
    hε : GT.gt ε 0
    h : IsIntegralCurveOn γ v (Metric.ball t₀ ε)
    ⊢ Eq (Metric.ball (HDiv.hDiv t₀ a) (HDiv.hDiv ε (abs a))) (setOf fun t => Memb …
  -/
  ext t
  rw [mem_setOf_eq, Metric.mem_ball, Metric.mem_ball, Real.dist_eq, Real.dist_eq,
    lt_div_iff₀ (abs_pos.mpr ha), ← abs_mul, sub_mul, div_mul_cancel₀ _ ha]


lemma isIntegralCurveAt_comp_mul_ne_zero {a : ℝ} (ha : a ≠ 0) :
    IsIntegralCurveAt γ v t₀ ↔ IsIntegralCurveAt (γ ∘ (· * a)) (a • v) (t₀ / a) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ a : Real
    ha : Ne a 0
    ⊢ Iff (IsIntegralCurveAt γ v t₀) (IsIntegralCurveAt (Function.comp γ fun x =>  …
  -/
  refine ⟨fun hγ ↦ hγ.comp_mul_ne_zero ha, fun hγ ↦ ?_⟩
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ a : Real
    ha : Ne a 0
    hγ : IsIntegralCurveAt (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a …
    ⊢ IsIntegralCurveAt γ v t₀
  -/
  convert hγ.comp_mul_ne_zero (inv_ne_zero ha)
    /-
      case h.e'_10
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      t₀ a : Real
      ha : Ne a 0
      hγ : IsIntegralCurveAt (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a …
      ⊢ Eq γ (Function.comp (Function.comp γ fun x => HMul.hMul x a) fun x => HMul.h …
    -/
  · ext t
    /-
      case h.e'_10.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      t₀ a : Real
      ha : Ne a 0
      hγ : IsIntegralCurveAt (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a …
      t : Real
      ⊢ Eq (γ t) (Function.comp (Function.comp γ fun x => HMul.hMul x a) (fun x => H …
    -/
    simp only [Function.comp_apply, mul_assoc, inv_mul_eq_div, div_self ha, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_11
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      t₀ a : Real
      ha : Ne a 0
      hγ : IsIntegralCurveAt (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a …
      ⊢ Eq v (HSMul.hSMul (Inv.inv a) (HSMul.hSMul a v))
    -/
  · simp only [smul_smul, inv_mul_eq_div, div_self ha, one_smul]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_12
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      t₀ a : Real
      ha : Ne a 0
      hγ : IsIntegralCurveAt (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a …
      ⊢ Eq t₀ (HDiv.hDiv (HDiv.hDiv t₀ a) (Inv.inv a))
    -/
  · simp only [div_inv_eq_mul, div_mul_cancel₀ _ ha]
    /-
      🎉 no goals
    -/


lemma IsIntegralCurve.comp_mul (hγ : IsIntegralCurve γ v) (a : ℝ) :
    IsIntegralCurve (γ ∘ (· * a)) (a • v) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    hγ : IsIntegralCurve γ v
    a : Real
    ⊢ IsIntegralCurve (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a v)
  -/
  rw [isIntegralCurve_iff_isIntegralCurveOn] at *
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    hγ : IsIntegralCurveOn γ v Set.univ
    a : Real
    ⊢ IsIntegralCurveOn (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a v) …
  -/
  exact hγ.comp_mul _
  /-
    🎉 no goals
  -/


lemma isIntegralCurve_comp_mul_ne_zero {a : ℝ} (ha : a ≠ 0) :
    IsIntegralCurve γ v ↔ IsIntegralCurve (γ ∘ (· * a)) (a • v) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    a : Real
    ha : Ne a 0
    ⊢ Iff (IsIntegralCurve γ v) (IsIntegralCurve (Function.comp γ fun x => HMul.hM …
  -/
  refine ⟨fun hγ ↦ hγ.comp_mul _, fun hγ ↦ ?_⟩
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    a : Real
    ha : Ne a 0
    hγ : IsIntegralCurve (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a v)
    ⊢ IsIntegralCurve γ v
  -/
  convert hγ.comp_mul a⁻¹
    /-
      case h.e'_10
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      a : Real
      ha : Ne a 0
      hγ : IsIntegralCurve (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a v)
      ⊢ Eq γ (Function.comp (Function.comp γ fun x => HMul.hMul x a) fun x => HMul.h …
    -/
  · ext t
    /-
      case h.e'_10.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      a : Real
      ha : Ne a 0
      hγ : IsIntegralCurve (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a v)
      t : Real
      ⊢ Eq (γ t) (Function.comp (Function.comp γ fun x => HMul.hMul x a) (fun x => H …
    -/
    simp only [Function.comp_apply, mul_assoc, inv_mul_eq_div, div_self ha, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_11
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      a : Real
      ha : Ne a 0
      hγ : IsIntegralCurve (Function.comp γ fun x => HMul.hMul x a) (HSMul.hSMul a v)
      ⊢ Eq v (HSMul.hSMul (Inv.inv a) (HSMul.hSMul a v))
    -/
  · simp only [smul_smul, inv_mul_eq_div, div_self ha, one_smul]
    /-
      🎉 no goals
    -/


/-- If the vector field `v` vanishes at `x₀`, then the constant curve at `x₀`
is a global integral curve of `v`. -/
lemma isIntegralCurve_const {x : M} (h : v x = 0) : IsIntegralCurve (fun _ ↦ x) v := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    v : (x : M) → TangentSpace I x
    x : M
    h : Eq (v x) 0
    ⊢ IsIntegralCurve (fun x_1 => x) v
  -/
  intro t
  rw [h, ← ContinuousLinearMap.zero_apply (R₁ := ℝ) (R₂ := ℝ) (1 : ℝ),
    ContinuousLinearMap.smulRight_one_one]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    v : (x : M) → TangentSpace I x
    x : M
    h : Eq (v x) 0
    t : Real
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (fun x_1 => x) t 0
  -/
  exact hasMFDerivAt_const ..
  /-
    🎉 no goals
  -/


