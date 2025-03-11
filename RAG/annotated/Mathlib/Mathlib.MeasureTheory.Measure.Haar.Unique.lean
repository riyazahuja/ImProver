/-- In a locally compact regular space with an inner regular measure, the measure of a compact
set `k` is the infimum of the integrals of compactly supported functions equal to `1` on `k`. -/
lemma IsCompact.measure_eq_biInf_integral_hasCompactSupport
    {X : Type*} [TopologicalSpace X] [MeasurableSpace X] [BorelSpace X]
    {k : Set X} (hk : IsCompact k)
    (μ : Measure X) [IsFiniteMeasureOnCompacts μ] [InnerRegularCompactLTTop μ]
    [LocallyCompactSpace X] [RegularSpace X] :
    μ k = ⨅ (f : X → ℝ) (_ : Continuous f) (_ : HasCompactSupport f) (_ : EqOn f 1 k)
      (_ : 0 ≤ f), ENNReal.ofReal (∫ x, f x ∂μ) := by
  /-
    X : Type u_1
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : BorelSpace X
    k : Set X
    hk : IsCompact k
    μ : MeasureTheory.Measure X
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝² : μ.InnerRegularCompactLTTop
    inst✝¹ : LocallyCompactSpace X
    inst✝ : RegularSpace X
    ⊢ Eq (μ k) (iInf fun f => iInf fun x => iInf fun x => iInf fun x => iInf fun x …
  -/
  apply le_antisymm
    /-
      case a
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : BorelSpace X
      k : Set X
      hk : IsCompact k
      μ : MeasureTheory.Measure X
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝² : μ.InnerRegularCompactLTTop
      inst✝¹ : LocallyCompactSpace X
      inst✝ : RegularSpace X
      ⊢ LE.le (μ k) (iInf fun f => iInf fun x => iInf fun x => iInf fun x => iInf fu …
    -/
  · simp only [le_iInf_iff]
    /-
      case a
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : BorelSpace X
      k : Set X
      hk : IsCompact k
      μ : MeasureTheory.Measure X
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝² : μ.InnerRegularCompactLTTop
      inst✝¹ : LocallyCompactSpace X
      inst✝ : RegularSpace X
      ⊢ ∀ (i : X → Real), Continuous i → HasCompactSupport i → Set.EqOn i 1 k → LE.l …
    -/
    intro f f_cont f_comp fk f_nonneg
    /-
      case a
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : BorelSpace X
      k : Set X
      hk : IsCompact k
      μ : MeasureTheory.Measure X
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝² : μ.InnerRegularCompactLTTop
      inst✝¹ : LocallyCompactSpace X
      inst✝ : RegularSpace X
      f : X → Real
      f_cont : Continuous f
      f_comp : HasCompactSupport f
      fk : Set.EqOn f 1 k
      f_nonneg : LE.le 0 f
      ⊢ LE.le (μ k) (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x))
    -/
    apply (f_cont.integrable_of_hasCompactSupport f_comp).measure_le_integral
      /-
        case a.f_nonneg
        X : Type u_1
        inst✝⁶ : TopologicalSpace X
        inst✝⁵ : MeasurableSpace X
        inst✝⁴ : BorelSpace X
        k : Set X
        hk : IsCompact k
        μ : MeasureTheory.Measure X
        inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ
        inst✝² : μ.InnerRegularCompactLTTop
        inst✝¹ : LocallyCompactSpace X
        inst✝ : RegularSpace X
        f : X → Real
        f_cont : Continuous f
        f_comp : HasCompactSupport f
        fk : Set.EqOn f 1 k
        f_nonneg : LE.le 0 f
        ⊢ (MeasureTheory.ae μ).EventuallyLE 0 f
      -/
    · exact Eventually.of_forall f_nonneg
      /-
        🎉 no goals
      -/
      /-
        case a.hs
        X : Type u_1
        inst✝⁶ : TopologicalSpace X
        inst✝⁵ : MeasurableSpace X
        inst✝⁴ : BorelSpace X
        k : Set X
        hk : IsCompact k
        μ : MeasureTheory.Measure X
        inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ
        inst✝² : μ.InnerRegularCompactLTTop
        inst✝¹ : LocallyCompactSpace X
        inst✝ : RegularSpace X
        f : X → Real
        f_cont : Continuous f
        f_comp : HasCompactSupport f
        fk : Set.EqOn f 1 k
        f_nonneg : LE.le 0 f
        ⊢ ∀ (x : X), Membership.mem k x → LE.le 1 (f x)
      -/
    · exact fun x hx ↦ by simp [fk hx]
      /-
        🎉 no goals
      -/
    /-
      case a
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : BorelSpace X
      k : Set X
      hk : IsCompact k
      μ : MeasureTheory.Measure X
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝² : μ.InnerRegularCompactLTTop
      inst✝¹ : LocallyCompactSpace X
      inst✝ : RegularSpace X
      ⊢ LE.le (iInf fun f => iInf fun x => iInf fun x => iInf fun x => iInf fun x => …
    -/
  · apply le_of_forall_lt' (fun r hr ↦ ?_)
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : BorelSpace X
      k : Set X
      hk : IsCompact k
      μ : MeasureTheory.Measure X
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝² : μ.InnerRegularCompactLTTop
      inst✝¹ : LocallyCompactSpace X
      inst✝ : RegularSpace X
      r : ENNReal
      hr : LT.lt (μ k) r
      ⊢ LT.lt (iInf fun f => iInf fun x => iInf fun x => iInf fun x => iInf fun x => …
    -/
    simp only [iInf_lt_iff, exists_prop, exists_and_left]
    obtain ⟨U, kU, U_open, mu_U⟩ : ∃ U, k ⊆ U ∧ IsOpen U ∧ μ U < r :=
      hk.exists_isOpen_lt_of_lt r hr
    obtain ⟨⟨f, f_cont⟩, fk, fU, f_comp, f_range⟩ : ∃ (f : C(X, ℝ)), EqOn f 1 k ∧ EqOn f 0 Uᶜ
        ∧ HasCompactSupport f ∧ ∀ (x : X), f x ∈ Icc 0 1 := exists_continuous_one_zero_of_isCompact
      hk U_open.isClosed_compl (disjoint_compl_right_iff_subset.mpr kU)
    /-
      case intro.intro.intro.intro.mk.intro.intro.intro
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : BorelSpace X
      k : Set X
      hk : IsCompact k
      μ : MeasureTheory.Measure X
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝² : μ.InnerRegularCompactLTTop
      inst✝¹ : LocallyCompactSpace X
      inst✝ : RegularSpace X
      r : ENNReal
      hr : LT.lt (μ k) r
      U : Set X
      kU : HasSubset.Subset k U
      U_open : IsOpen U
      mu_U : LT.lt (μ U) r
      f : X → Real
      f_cont : Continuous f
      fk : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 1 k
      fU : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 0 (HasCompl.compl U)
      f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
      f_range : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_to …
      ⊢ Exists fun i => And (Continuous i) (And (HasCompactSupport i) (And (Set.EqOn …
    -/
    refine ⟨f, f_cont, f_comp, fk, fun x ↦ (f_range x).1, ?_⟩
    /-
      case intro.intro.intro.intro.mk.intro.intro.intro
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : BorelSpace X
      k : Set X
      hk : IsCompact k
      μ : MeasureTheory.Measure X
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝² : μ.InnerRegularCompactLTTop
      inst✝¹ : LocallyCompactSpace X
      inst✝ : RegularSpace X
      r : ENNReal
      hr : LT.lt (μ k) r
      U : Set X
      kU : HasSubset.Subset k U
      U_open : IsOpen U
      mu_U : LT.lt (μ U) r
      f : X → Real
      f_cont : Continuous f
      fk : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 1 k
      fU : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 0 (HasCompl.compl U)
      f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
      f_range : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_to …
      ⊢ LT.lt (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) r
    -/
    exact (integral_le_measure (fun x _hx ↦ (f_range x).2) (fun x hx ↦ (fU hx).le)).trans_lt mu_U
    /-
      🎉 no goals
    -/


/-- The parameterized integral `x ↦ ∫ y, g (y⁻¹ * x) ∂μ` depends continuously on `y` when `g` is a
compactly supported continuous function on a topological group `G`, and `μ` is finite on compact
sets. -/
@[to_additive]
lemma continuous_integral_apply_inv_mul
    {G : Type*} [TopologicalSpace G] [LocallyCompactSpace G] [Group G] [TopologicalGroup G]
    [MeasurableSpace G] [BorelSpace G]
    {μ : Measure G} [IsFiniteMeasureOnCompacts μ] {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] {g : G → E}
    (hg : Continuous g) (h'g : HasCompactSupport g) :
    Continuous (fun (x : G) ↦ ∫ y, g (y⁻¹ * x) ∂μ) := by
  /-
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : LocallyCompactSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : G → E
    hg : Continuous g
    h'g : HasCompactSupport g
    ⊢ Continuous fun x => MeasureTheory.integral μ fun y => g (HMul.hMul (Inv.inv  …
  -/
  let k := tsupport g
  /-
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : LocallyCompactSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : G → E
    hg : Continuous g
    h'g : HasCompactSupport g
    k : Set G := tsupport g
    ⊢ Continuous fun x => MeasureTheory.integral μ fun y => g (HMul.hMul (Inv.inv  …
  -/
  have k_comp : IsCompact k := h'g
  /-
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : LocallyCompactSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : G → E
    hg : Continuous g
    h'g : HasCompactSupport g
    k : Set G := tsupport g
    k_comp : IsCompact k
    ⊢ Continuous fun x => MeasureTheory.integral μ fun y => g (HMul.hMul (Inv.inv  …
  -/
  apply continuous_iff_continuousAt.2 (fun x₀ ↦ ?_)
  /-
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : LocallyCompactSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : G → E
    hg : Continuous g
    h'g : HasCompactSupport g
    k : Set G := tsupport g
    k_comp : IsCompact k
    x₀ : G
    ⊢ ContinuousAt (fun x => MeasureTheory.integral μ fun y => g (HMul.hMul (Inv.i …
  -/
  obtain ⟨t, t_comp, ht⟩ : ∃ t, IsCompact t ∧ t ∈ 𝓝 x₀ := exists_compact_mem_nhds x₀
  /-
    case intro.intro
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : LocallyCompactSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : G → E
    hg : Continuous g
    h'g : HasCompactSupport g
    k : Set G := tsupport g
    k_comp : IsCompact k
    x₀ : G
    t : Set G
    t_comp : IsCompact t
    ht : Membership.mem (nhds x₀) t
    ⊢ ContinuousAt (fun x => MeasureTheory.integral μ fun y => g (HMul.hMul (Inv.i …
  -/
  let k' : Set G := t • k⁻¹
  /-
    case intro.intro
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : LocallyCompactSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : G → E
    hg : Continuous g
    h'g : HasCompactSupport g
    k : Set G := tsupport g
    k_comp : IsCompact k
    x₀ : G
    t : Set G
    t_comp : IsCompact t
    ht : Membership.mem (nhds x₀) t
    k' : Set G := HSMul.hSMul t (Inv.inv k)
    ⊢ ContinuousAt (fun x => MeasureTheory.integral μ fun y => g (HMul.hMul (Inv.i …
  -/
  have k'_comp : IsCompact k' := t_comp.smul_set k_comp.inv
  have A : ContinuousOn (fun (x : G) ↦ ∫ y, g (y⁻¹ * x) ∂μ) t := by
    apply continuousOn_integral_of_compact_support k'_comp
    · exact (hg.comp (continuous_snd.inv.mul continuous_fst)).continuousOn
    · intro p x hp hx
      contrapose! hx
      refine ⟨p, hp, p⁻¹ * x, ?_, by simp⟩
      simpa only [Set.mem_inv, mul_inv_rev, inv_inv] using subset_tsupport _ hx
  /-
    case intro.intro
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : LocallyCompactSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : G → E
    hg : Continuous g
    h'g : HasCompactSupport g
    k : Set G := tsupport g
    k_comp : IsCompact k
    x₀ : G
    t : Set G
    t_comp : IsCompact t
    ht : Membership.mem (nhds x₀) t
    k' : Set G := HSMul.hSMul t (Inv.inv k)
    k'_comp : IsCompact k'
    A : ContinuousOn (fun x => MeasureTheory.integral μ fun y => g (HMul.hMul (Inv …
    ⊢ ContinuousAt (fun x => MeasureTheory.integral μ fun y => g (HMul.hMul (Inv.i …
  -/
  exact A.continuousAt ht
  /-
    🎉 no goals
  -/


/-- In a group with a left invariant measure `μ` and a right invariant measure `ν`, one can express
integrals with respect to `μ` as integrals with respect to `ν` up to a constant scaling factor
(given in the statement as `∫ x, g x ∂μ` where `g` is a fixed reference function) and an
explicit density `y ↦ 1/∫ z, g (z⁻¹ * y) ∂ν`. -/
@[to_additive]
lemma integral_isMulLeftInvariant_isMulRightInvariant_combo
    {μ ν : Measure G} [IsFiniteMeasureOnCompacts μ] [IsFiniteMeasureOnCompacts ν]
    [IsMulLeftInvariant μ] [IsMulRightInvariant ν] [IsOpenPosMeasure ν]
    {f g : G → ℝ} (hf : Continuous f) (h'f : HasCompactSupport f)
    (hg : Continuous g) (h'g : HasCompactSupport g) (g_nonneg : 0 ≤ g) {x₀ : G} (g_pos : g x₀ ≠ 0) :
    ∫ x, f x ∂μ = (∫ y, f y * (∫ z, g (z⁻¹ * y) ∂ν)⁻¹ ∂ν) * ∫ x, g x ∂μ := by
  -- The group has to be locally compact, otherwise all integrals vanish and the result is trivial.
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : ν.IsMulRightInvariant
    inst✝ : ν.IsOpenPosMeasure
    f g : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    hg : Continuous g
    h'g : HasCompactSupport g
    g_nonneg : LE.le 0 g
    x₀ : G
    g_pos : Ne (g x₀) 0
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (HMul.hMul (MeasureTheory.integra …
  -/
  rcases h'f.eq_zero_or_locallyCompactSpace_of_group hf with Hf|Hf
    /-
      case inl
      G : Type u_1
      inst✝⁹ : TopologicalSpace G
      inst✝⁸ : Group G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : MeasurableSpace G
      inst✝⁵ : BorelSpace G
      μ ν : MeasureTheory.Measure G
      inst✝⁴ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts ν
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : ν.IsMulRightInvariant
      inst✝ : ν.IsOpenPosMeasure
      f g : G → Real
      hf : Continuous f
      h'f : HasCompactSupport f
      hg : Continuous g
      h'g : HasCompactSupport g
      g_nonneg : LE.le 0 g
      x₀ : G
      g_pos : Ne (g x₀) 0
      Hf : Eq f 0
      ⊢ Eq (MeasureTheory.integral μ fun x => f x) (HMul.hMul (MeasureTheory.integra …
    -/
  · simp [Hf]
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : ν.IsMulRightInvariant
    inst✝ : ν.IsOpenPosMeasure
    f g : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    hg : Continuous g
    h'g : HasCompactSupport g
    g_nonneg : LE.le 0 g
    x₀ : G
    g_pos : Ne (g x₀) 0
    Hf : LocallyCompactSpace G
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (HMul.hMul (MeasureTheory.integra …
  -/
  let D : G → ℝ := fun (x : G) ↦ ∫ y, g (y⁻¹ * x) ∂ν
  /-
    case inr
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : ν.IsMulRightInvariant
    inst✝ : ν.IsOpenPosMeasure
    f g : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    hg : Continuous g
    h'g : HasCompactSupport g
    g_nonneg : LE.le 0 g
    x₀ : G
    g_pos : Ne (g x₀) 0
    Hf : LocallyCompactSpace G
    D : G → Real := fun x => MeasureTheory.integral ν fun y => g (HMul.hMul (Inv.i …
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (HMul.hMul (MeasureTheory.integra …
  -/
  have D_cont : Continuous D := continuous_integral_apply_inv_mul hg h'g
  have D_pos : ∀ x, 0 < D x := by
    intro x
    have C : Continuous (fun y ↦ g (y⁻¹ * x)) := hg.comp (continuous_inv.mul continuous_const)
    apply (integral_pos_iff_support_of_nonneg _ _).2
    · apply C.isOpen_support.measure_pos ν
      exact ⟨x * x₀⁻¹, by simpa using g_pos⟩
    · exact fun y ↦ g_nonneg (y⁻¹ * x)
    · apply C.integrable_of_hasCompactSupport
      exact h'g.comp_homeomorph ((Homeomorph.inv G).trans (Homeomorph.mulRight x))
  calc
  ∫ x, f x ∂μ = ∫ x, f x * (D x)⁻¹ * D x ∂μ := by
    congr with x; rw [mul_assoc, inv_mul_cancel₀ (D_pos x).ne', mul_one]
  _ = ∫ x, (∫ y, f x * (D x)⁻¹ * g (y⁻¹ * x) ∂ν) ∂μ := by simp_rw [D, integral_mul_left]
  _ = ∫ y, (∫ x, f x * (D x)⁻¹ * g (y⁻¹ * x) ∂μ) ∂ν := by
      apply integral_integral_swap_of_hasCompactSupport
      · apply Continuous.mul
        · exact (hf.comp continuous_fst).mul
            ((D_cont.comp continuous_fst).inv₀ (fun x ↦ (D_pos _).ne'))
        · exact hg.comp (continuous_snd.inv.mul continuous_fst)
      · let K := tsupport f
        have K_comp : IsCompact K := h'f
        let L := tsupport g
        have L_comp : IsCompact L := h'g
        let M := (fun (p : G × G) ↦ p.1 * p.2⁻¹) '' (K ×ˢ L)
        have M_comp : IsCompact M :=
          (K_comp.prod L_comp).image (continuous_fst.mul continuous_snd.inv)
        have M'_comp : IsCompact (closure M) := M_comp.closure
        have : ∀ (p : G × G), p ∉ K ×ˢ closure M → f p.1 * (D p.1)⁻¹ * g (p.2⁻¹ * p.1) = 0 := by
          rintro ⟨x, y⟩ hxy
          by_cases H : x ∈ K; swap
          · simp [image_eq_zero_of_nmem_tsupport H]
          have : g (y⁻¹ * x) = 0 := by
            apply image_eq_zero_of_nmem_tsupport
            contrapose! hxy
            simp only [mem_prod, H, true_and]
            apply subset_closure
            simp only [M, mem_image, mem_prod, Prod.exists]
            exact ⟨x, y⁻¹ * x, ⟨H, hxy⟩, by group⟩
          simp [this]
        apply HasCompactSupport.intro' (K_comp.prod M'_comp) ?_ this
        exact (isClosed_tsupport f).prod isClosed_closure
  _ = ∫ y, (∫ x, f (y * x) * (D (y * x))⁻¹ * g x ∂μ) ∂ν := by
      congr with y
      rw [← integral_mul_left_eq_self _ y]
      simp
  _ = ∫ x, (∫ y, f (y * x) * (D (y * x))⁻¹ * g x ∂ν) ∂μ := by
      apply (integral_integral_swap_of_hasCompactSupport _ _).symm
      · apply Continuous.mul ?_ (hg.comp continuous_fst)
        exact (hf.comp (continuous_snd.mul continuous_fst)).mul
          ((D_cont.comp (continuous_snd.mul continuous_fst)).inv₀ (fun x ↦ (D_pos _).ne'))
      · let K := tsupport f
        have K_comp : IsCompact K := h'f
        let L := tsupport g
        have L_comp : IsCompact L := h'g
        let M := (fun (p : G × G) ↦ p.1 * p.2⁻¹) '' (K ×ˢ L)
        have M_comp : IsCompact M :=
          (K_comp.prod L_comp).image (continuous_fst.mul continuous_snd.inv)
        have M'_comp : IsCompact (closure M) := M_comp.closure
        have : ∀ (p : G × G), p ∉ L ×ˢ closure M →
            f (p.2 * p.1) * (D (p.2 * p.1))⁻¹ * g p.1 = 0 := by
          rintro ⟨x, y⟩ hxy
          by_cases H : x ∈ L; swap
          · simp [image_eq_zero_of_nmem_tsupport H]
          have : f (y * x) = 0 := by
            apply image_eq_zero_of_nmem_tsupport
            contrapose! hxy
            simp only [mem_prod, H, true_and]
            apply subset_closure
            simp only [M, mem_image, mem_prod, Prod.exists]
            exact ⟨y * x, x, ⟨hxy, H⟩, by group⟩
          simp [this]
        apply HasCompactSupport.intro' (L_comp.prod M'_comp) ?_ this
        exact (isClosed_tsupport g).prod isClosed_closure
  _ = ∫ x, (∫ y, f y * (D y)⁻¹ ∂ν) * g x ∂μ := by
      simp_rw [integral_mul_right]
      congr with x
      conv_rhs => rw [← integral_mul_right_eq_self _ x]
  _ = (∫ y, f y * (D y)⁻¹ ∂ν) * ∫ x, g x ∂μ := integral_mul_left _ _


/-- Given two left-invariant measures which are finite on
compacts, they coincide in the following sense: they give the same value to the integral of
continuous compactly supported functions, up to a multiplicative constant. -/
@[to_additive exists_integral_isAddLeftInvariant_eq_smul_of_hasCompactSupport]
lemma exists_integral_isMulLeftInvariant_eq_smul_of_hasCompactSupport (μ' μ : Measure G)
    [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ'] :
    ∃ (c : ℝ≥0), ∀ (f : G → ℝ), Continuous f → HasCompactSupport f →
      ∫ x, f x ∂μ' = ∫ x, f x ∂(c • μ) := by
  -- The group has to be locally compact, otherwise all integrals vanish and the result is trivial.
  /-
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    ⊢ Exists fun c => ∀ (f : G → Real), Continuous f → HasCompactSupport f → Eq (M …
  -/
  by_cases H : LocallyCompactSpace G; swap
    /-
      case neg
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝² : μ.IsHaarMeasure
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
      inst✝ : μ'.IsMulLeftInvariant
      H : Not (LocallyCompactSpace G)
      ⊢ Exists fun c => ∀ (f : G → Real), Continuous f → HasCompactSupport f → Eq (M …
    -/
  · refine ⟨0, fun f f_cont f_comp ↦ ?_⟩
    /-
      case neg
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝² : μ.IsHaarMeasure
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
      inst✝ : μ'.IsMulLeftInvariant
      H : Not (LocallyCompactSpace G)
      f : G → Real
      f_cont : Continuous f
      f_comp : HasCompactSupport f
      ⊢ Eq (MeasureTheory.integral μ' fun x => f x) (MeasureTheory.integral (HSMul.h …
    -/
    rcases f_comp.eq_zero_or_locallyCompactSpace_of_group f_cont with hf|hf
      /-
        case neg.inl
        G : Type u_1
        inst✝⁷ : TopologicalSpace G
        inst✝⁶ : Group G
        inst✝⁵ : TopologicalGroup G
        inst✝⁴ : MeasurableSpace G
        inst✝³ : BorelSpace G
        μ' μ : MeasureTheory.Measure G
        inst✝² : μ.IsHaarMeasure
        inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
        inst✝ : μ'.IsMulLeftInvariant
        H : Not (LocallyCompactSpace G)
        f : G → Real
        f_cont : Continuous f
        f_comp : HasCompactSupport f
        hf : Eq f 0
        ⊢ Eq (MeasureTheory.integral μ' fun x => f x) (MeasureTheory.integral (HSMul.h …
      -/
    · simp [hf]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        G : Type u_1
        inst✝⁷ : TopologicalSpace G
        inst✝⁶ : Group G
        inst✝⁵ : TopologicalGroup G
        inst✝⁴ : MeasurableSpace G
        inst✝³ : BorelSpace G
        μ' μ : MeasureTheory.Measure G
        inst✝² : μ.IsHaarMeasure
        inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
        inst✝ : μ'.IsMulLeftInvariant
        H : Not (LocallyCompactSpace G)
        f : G → Real
        f_cont : Continuous f
        f_comp : HasCompactSupport f
        hf : LocallyCompactSpace G
        ⊢ Eq (MeasureTheory.integral μ' fun x => f x) (MeasureTheory.integral (HSMul.h …
      -/
    · exact (H hf).elim
      /-
        🎉 no goals
      -/
  -- Fix some nonzero continuous function with compact support `g`.
  obtain ⟨⟨g, g_cont⟩, g_comp, g_nonneg, g_one⟩ :
    ∃ (g : C(G, ℝ)), HasCompactSupport g ∧ 0 ≤ g ∧ g 1 ≠ 0 := exists_continuous_nonneg_pos 1
  have int_g_pos : 0 < ∫ x, g x ∂μ :=
    g_cont.integral_pos_of_hasCompactSupport_nonneg_nonzero g_comp g_nonneg g_one
  -- The proportionality constant we are looking for will be the ratio of the integrals of `g`
  -- with respect to `μ'` and `μ`.
  /-
    case pos.intro.mk.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    H : LocallyCompactSpace G
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    int_g_pos : LT.lt 0 (MeasureTheory.integral μ fun x => g x)
    ⊢ Exists fun c => ∀ (f : G → Real), Continuous f → HasCompactSupport f → Eq (M …
  -/
  let c : ℝ := (∫ x, g x ∂μ) ⁻¹ * (∫ x, g x ∂μ')
  have c_nonneg : 0 ≤ c :=
    mul_nonneg (inv_nonneg.2 (integral_nonneg g_nonneg)) (integral_nonneg g_nonneg)
  /-
    case pos.intro.mk.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    H : LocallyCompactSpace G
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    int_g_pos : LT.lt 0 (MeasureTheory.integral μ fun x => g x)
    c : Real := HMul.hMul (Inv.inv (MeasureTheory.integral μ fun x => g x)) (Measu …
    c_nonneg : LE.le 0 c
    ⊢ Exists fun c => ∀ (f : G → Real), Continuous f → HasCompactSupport f → Eq (M …
  -/
  refine ⟨⟨c, c_nonneg⟩, fun f f_cont f_comp ↦ ?_⟩
  /- use the lemma `integral_mulLeftInvariant_mulRightInvariant_combo` for `μ` and then `μ'`
  to reexpress the integral of `f` as the integral of `g` times a factor which only depends
  on a right-invariant measure `ν`. We use `ν = μ.inv` for convenience. -/
  /-
    case pos.intro.mk.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    H : LocallyCompactSpace G
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    int_g_pos : LT.lt 0 (MeasureTheory.integral μ fun x => g x)
    c : Real := HMul.hMul (Inv.inv (MeasureTheory.integral μ fun x => g x)) (Measu …
    c_nonneg : LE.le 0 c
    f : G → Real
    f_cont : Continuous f
    f_comp : HasCompactSupport f
    ⊢ Eq (MeasureTheory.integral μ' fun x => f x) (MeasureTheory.integral (HSMul.h …
  -/
  let ν := μ.inv
  have A : ∫ x, f x ∂μ = (∫ y, f y * (∫ z, g (z⁻¹ * y) ∂ν)⁻¹ ∂ν) * ∫ x, g x ∂μ :=
    integral_isMulLeftInvariant_isMulRightInvariant_combo f_cont f_comp g_cont g_comp g_nonneg g_one
  /-
    case pos.intro.mk.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    H : LocallyCompactSpace G
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    int_g_pos : LT.lt 0 (MeasureTheory.integral μ fun x => g x)
    c : Real := HMul.hMul (Inv.inv (MeasureTheory.integral μ fun x => g x)) (Measu …
    c_nonneg : LE.le 0 c
    f : G → Real
    f_cont : Continuous f
    f_comp : HasCompactSupport f
    ν : MeasureTheory.Measure G := μ.inv
    A : Eq (MeasureTheory.integral μ fun x => f x) (HMul.hMul (MeasureTheory.integ …
    ⊢ Eq (MeasureTheory.integral μ' fun x => f x) (MeasureTheory.integral (HSMul.h …
  -/
  rw [← mul_inv_eq_iff_eq_mul₀ int_g_pos.ne'] at A
  have B : ∫ x, f x ∂μ' = (∫ y, f y * (∫ z, g (z⁻¹ * y) ∂ν)⁻¹ ∂ν) * ∫ x, g x ∂μ' :=
    integral_isMulLeftInvariant_isMulRightInvariant_combo f_cont f_comp g_cont g_comp g_nonneg g_one
  /- Since the `ν`-factor is the same for `μ` and `μ'`, this gives the result. -/
  /-
    case pos.intro.mk.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    H : LocallyCompactSpace G
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    int_g_pos : LT.lt 0 (MeasureTheory.integral μ fun x => g x)
    c : Real := HMul.hMul (Inv.inv (MeasureTheory.integral μ fun x => g x)) (Measu …
    c_nonneg : LE.le 0 c
    f : G → Real
    f_cont : Continuous f
    f_comp : HasCompactSupport f
    ν : MeasureTheory.Measure G := μ.inv
    A : Eq (HMul.hMul (MeasureTheory.integral μ fun x => f x) (Inv.inv (MeasureThe …
    B : Eq (MeasureTheory.integral μ' fun x => f x) (HMul.hMul (MeasureTheory.inte …
    ⊢ Eq (MeasureTheory.integral μ' fun x => f x) (MeasureTheory.integral (HSMul.h …
  -/
  rw [← A, mul_assoc, mul_comm] at B
  /-
    case pos.intro.mk.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    H : LocallyCompactSpace G
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    int_g_pos : LT.lt 0 (MeasureTheory.integral μ fun x => g x)
    c : Real := HMul.hMul (Inv.inv (MeasureTheory.integral μ fun x => g x)) (Measu …
    c_nonneg : LE.le 0 c
    f : G → Real
    f_cont : Continuous f
    f_comp : HasCompactSupport f
    ν : MeasureTheory.Measure G := μ.inv
    A : Eq (HMul.hMul (MeasureTheory.integral μ fun x => f x) (Inv.inv (MeasureThe …
    B : Eq (MeasureTheory.integral μ' fun x => f x) (HMul.hMul (HMul.hMul (Inv.inv …
    ⊢ Eq (MeasureTheory.integral μ' fun x => f x) (MeasureTheory.integral (HSMul.h …
  -/
  simp only [B, integral_smul_nnreal_measure]
  /-
    case pos.intro.mk.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    H : LocallyCompactSpace G
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    int_g_pos : LT.lt 0 (MeasureTheory.integral μ fun x => g x)
    c : Real := HMul.hMul (Inv.inv (MeasureTheory.integral μ fun x => g x)) (Measu …
    c_nonneg : LE.le 0 c
    f : G → Real
    f_cont : Continuous f
    f_comp : HasCompactSupport f
    ν : MeasureTheory.Measure G := μ.inv
    A : Eq (HMul.hMul (MeasureTheory.integral μ fun x => f x) (Inv.inv (MeasureThe …
    B : Eq (MeasureTheory.integral μ' fun x => f x) (HMul.hMul (HMul.hMul (Inv.inv …
    ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv (MeasureTheory.integral μ fun x => g x)) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


open scoped Classical in
/-- Given two left-invariant measures which are finite on compacts, `haarScalarFactor μ' μ` is a
scalar such that `∫ f dμ' = (haarScalarFactor μ' μ) ∫ f dμ` for any compactly supported continuous
function `f`.

Note that there is a dissymmetry in the assumptions between `μ'` and `μ`: the measure `μ'` needs
only be finite on compact sets, while `μ` has to be finite on compact sets and positive on open
sets, i.e., a Haar measure, to exclude for instance the case where `μ = 0`, where the definition
doesn't make sense. -/
@[to_additive "Given two left-invariant measures which are finite on compacts,
`addHaarScalarFactor μ' μ` is a scalar such that `∫ f dμ' = (addHaarScalarFactor μ' μ) ∫ f dμ` for
any compactly supported continuous function `f`.

Note that there is a dissymmetry in the assumptions between `μ'` and `μ`: the measure `μ'` needs
only be finite on compact sets, while `μ` has to be finite on compact sets and positive on open
sets, i.e., an additive Haar measure, to exclude for instance the case where `μ = 0`, where the
definition doesn't make sense."]
noncomputable def haarScalarFactor
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ'] :
    ℝ≥0 :=
  if ¬ LocallyCompactSpace G then 1
  else (exists_integral_isMulLeftInvariant_eq_smul_of_hasCompactSupport μ' μ).choose


/-- Two left invariant measures integrate in the same way continuous compactly supported functions,
up to the scalar `haarScalarFactor μ' μ`. See also
`measure_isMulInvariant_eq_smul_of_isCompact_closure`, which gives the same result for compact
sets, and `measure_isHaarMeasure_eq_smul_of_isOpen` for open sets. -/
@[to_additive integral_isAddLeftInvariant_eq_smul_of_hasCompactSupport
"Two left invariant measures integrate in the same way continuous compactly supported functions,
up to the scalar `addHaarScalarFactor μ' μ`. See also
`measure_isAddInvariant_eq_smul_of_isCompact_closure`, which gives the same result for compact
sets, and `measure_isAddHaarMeasure_eq_smul_of_isOpen` for open sets."]
theorem integral_isMulLeftInvariant_eq_smul_of_hasCompactSupport
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ']
    {f : G → ℝ} (hf : Continuous f) (h'f : HasCompactSupport f) :
    ∫ x, f x ∂μ' = ∫ x, f x ∂(haarScalarFactor μ' μ • μ) := by
  classical
  rcases h'f.eq_zero_or_locallyCompactSpace_of_group hf with Hf|Hf
  · simp [Hf]
  · simp only [haarScalarFactor, Hf, not_true_eq_false, ite_false]
    exact (exists_integral_isMulLeftInvariant_eq_smul_of_hasCompactSupport μ' μ).choose_spec
      f hf h'f


@[to_additive addHaarScalarFactor_eq_integral_div]
lemma haarScalarFactor_eq_integral_div (μ' μ : Measure G) [IsHaarMeasure μ]
    [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ'] {f : G → ℝ} (hf : Continuous f)
    (h'f : HasCompactSupport f) (int_nonzero : ∫ x, f x ∂μ ≠ 0) :
    haarScalarFactor μ' μ = (∫ x, f x ∂μ') / ∫ x, f x ∂μ := by
  /-
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    f : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    int_nonzero : Ne (MeasureTheory.integral μ fun x => f x) 0
    ⊢ Eq (↑(μ'.haarScalarFactor μ)) (HDiv.hDiv (MeasureTheory.integral μ' fun x => …
  -/
  have := integral_isMulLeftInvariant_eq_smul_of_hasCompactSupport μ' μ hf h'f
  /-
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    f : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    int_nonzero : Ne (MeasureTheory.integral μ fun x => f x) 0
    this : Eq (MeasureTheory.integral μ' fun x => f x) (MeasureTheory.integral (HS …
    ⊢ Eq (↑(μ'.haarScalarFactor μ)) (HDiv.hDiv (MeasureTheory.integral μ' fun x => …
  -/
  rw [integral_smul_nnreal_measure] at this
  /-
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    f : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    int_nonzero : Ne (MeasureTheory.integral μ fun x => f x) 0
    this : Eq (MeasureTheory.integral μ' fun x => f x) (HSMul.hSMul (μ'.haarScalar …
    ⊢ Eq (↑(μ'.haarScalarFactor μ)) (HDiv.hDiv (MeasureTheory.integral μ' fun x => …
  -/
  exact EuclideanDomain.eq_div_of_mul_eq_left int_nonzero this.symm
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) addHaarScalarFactor_smul]
lemma haarScalarFactor_smul [LocallyCompactSpace G] (μ' μ : Measure G) [IsHaarMeasure μ]
    [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ'] {c : ℝ≥0} :
    haarScalarFactor (c • μ') μ = c • haarScalarFactor μ' μ := by
  obtain ⟨⟨g, g_cont⟩, g_comp, g_nonneg, g_one⟩ :
    ∃ g : C(G, ℝ), HasCompactSupport g ∧ 0 ≤ g ∧ g 1 ≠ 0 := exists_continuous_nonneg_pos 1
  have int_g_ne_zero : ∫ x, g x ∂μ ≠ 0 :=
    ne_of_gt (g_cont.integral_pos_of_hasCompactSupport_nonneg_nonzero g_comp g_nonneg g_one)
  /-
    case intro.mk.intro.intro
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : BorelSpace G
    inst✝³ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    c : NNReal
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    int_g_ne_zero : Ne (MeasureTheory.integral μ fun x => g x) 0
    ⊢ Eq ((HSMul.hSMul c μ').haarScalarFactor μ) (HSMul.hSMul c (μ'.haarScalarFact …
  -/
  apply NNReal.coe_injective
  calc
    haarScalarFactor (c • μ') μ = (∫ x, g x ∂(c • μ')) / ∫ x, g x ∂μ :=
      haarScalarFactor_eq_integral_div _ _ g_cont g_comp int_g_ne_zero
    _ = (c • (∫ x, g x ∂μ')) / ∫ x, g x ∂μ := by simp
    _ = c • ((∫ x, g x ∂μ') / ∫ x, g x ∂μ) := smul_div_assoc c _ _
    _ = c • haarScalarFactor μ' μ := by
      rw [← haarScalarFactor_eq_integral_div _ _ g_cont g_comp int_g_ne_zero]


@[to_additive (attr := simp)]
lemma haarScalarFactor_self (μ : Measure G) [IsHaarMeasure μ] :
    haarScalarFactor μ μ = 1 := by
  /-
    G : Type u_1
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    inst✝² : MeasurableSpace G
    inst✝¹ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝ : μ.IsHaarMeasure
    ⊢ Eq (μ.haarScalarFactor μ) 1
  -/
  by_cases hG : LocallyCompactSpace G; swap
    /-
      case neg
      G : Type u_1
      inst✝⁵ : TopologicalSpace G
      inst✝⁴ : Group G
      inst✝³ : TopologicalGroup G
      inst✝² : MeasurableSpace G
      inst✝¹ : BorelSpace G
      μ : MeasureTheory.Measure G
      inst✝ : μ.IsHaarMeasure
      hG : Not (LocallyCompactSpace G)
      ⊢ Eq (μ.haarScalarFactor μ) 1
    -/
  · simp [haarScalarFactor, hG]
    /-
      🎉 no goals
    -/
  obtain ⟨⟨g, g_cont⟩, g_comp, g_nonneg, g_one⟩ :
    ∃ g : C(G, ℝ), HasCompactSupport g ∧ 0 ≤ g ∧ g 1 ≠ 0 := exists_continuous_nonneg_pos 1
  have int_g_ne_zero : ∫ x, g x ∂μ ≠ 0 :=
    ne_of_gt (g_cont.integral_pos_of_hasCompactSupport_nonneg_nonzero g_comp g_nonneg g_one)
  /-
    case pos.intro.mk.intro.intro
    G : Type u_1
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    inst✝² : MeasurableSpace G
    inst✝¹ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝ : μ.IsHaarMeasure
    hG : LocallyCompactSpace G
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    int_g_ne_zero : Ne (MeasureTheory.integral μ fun x => g x) 0
    ⊢ Eq (μ.haarScalarFactor μ) 1
  -/
  apply NNReal.coe_injective
  calc
    haarScalarFactor μ μ = (∫ x, g x ∂μ) / ∫ x, g x ∂μ :=
      haarScalarFactor_eq_integral_div _ _ g_cont g_comp int_g_ne_zero
    _ = 1 := div_self int_g_ne_zero


@[to_additive addHaarScalarFactor_eq_mul]
lemma haarScalarFactor_eq_mul (μ' μ ν : Measure G)
    [IsHaarMeasure μ] [IsHaarMeasure ν] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ'] :
    haarScalarFactor μ' ν = haarScalarFactor μ' μ * haarScalarFactor μ ν := by
  -- The group has to be locally compact, otherwise the scalar factor is 1 by definition.
  /-
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : BorelSpace G
    μ' μ ν : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : ν.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    ⊢ Eq (μ'.haarScalarFactor ν) (HMul.hMul (μ'.haarScalarFactor μ) (μ.haarScalarF …
  -/
  by_cases hG : LocallyCompactSpace G; swap
    /-
      case neg
      G : Type u_1
      inst✝⁸ : TopologicalSpace G
      inst✝⁷ : Group G
      inst✝⁶ : TopologicalGroup G
      inst✝⁵ : MeasurableSpace G
      inst✝⁴ : BorelSpace G
      μ' μ ν : MeasureTheory.Measure G
      inst✝³ : μ.IsHaarMeasure
      inst✝² : ν.IsHaarMeasure
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
      inst✝ : μ'.IsMulLeftInvariant
      hG : Not (LocallyCompactSpace G)
      ⊢ Eq (μ'.haarScalarFactor ν) (HMul.hMul (μ'.haarScalarFactor μ) (μ.haarScalarF …
    -/
  · simp [haarScalarFactor, hG]
    /-
      🎉 no goals
    -/
  -- Fix some nonzero continuous function with compact support `g`.
  obtain ⟨⟨g, g_cont⟩, g_comp, g_nonneg, g_one⟩ :
    ∃ (g : C(G, ℝ)), HasCompactSupport g ∧ 0 ≤ g ∧ g 1 ≠ 0 := exists_continuous_nonneg_pos 1
  /-
    case pos.intro.mk.intro.intro
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : BorelSpace G
    μ' μ ν : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : ν.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    hG : LocallyCompactSpace G
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    ⊢ Eq (μ'.haarScalarFactor ν) (HMul.hMul (μ'.haarScalarFactor μ) (μ.haarScalarF …
  -/
  have Z := integral_isMulLeftInvariant_eq_smul_of_hasCompactSupport μ' μ g_cont g_comp
  simp only [integral_smul_nnreal_measure, smul_smul,
    integral_isMulLeftInvariant_eq_smul_of_hasCompactSupport μ' ν g_cont g_comp,
    integral_isMulLeftInvariant_eq_smul_of_hasCompactSupport μ ν g_cont g_comp] at Z
  have int_g_pos : 0 < ∫ x, g x ∂ν := by
    apply (integral_pos_iff_support_of_nonneg g_nonneg _).2
    · exact IsOpen.measure_pos ν g_cont.isOpen_support ⟨1, g_one⟩
    · exact g_cont.integrable_of_hasCompactSupport g_comp
  change (haarScalarFactor μ' ν : ℝ) * ∫ (x : G), g x ∂ν =
    (haarScalarFactor μ' μ * haarScalarFactor μ ν : ℝ≥0) * ∫ (x : G), g x ∂ν at Z
  /-
    case pos.intro.mk.intro.intro
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : BorelSpace G
    μ' μ ν : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : ν.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    hG : LocallyCompactSpace G
    g : G → Real
    g_cont : Continuous g
    g_comp : HasCompactSupport ⇑{ toFun := g, continuous_toFun := g_cont }
    g_nonneg : LE.le 0 { toFun := g, continuous_toFun := g_cont }
    g_one : Ne ({ toFun := g, continuous_toFun := g_cont } 1) 0
    int_g_pos : LT.lt 0 (MeasureTheory.integral ν fun x => g x)
    Z : Eq (HMul.hMul (↑(μ'.haarScalarFactor ν)) (MeasureTheory.integral ν fun x = …
    ⊢ Eq (μ'.haarScalarFactor ν) (HMul.hMul (μ'.haarScalarFactor μ) (μ.haarScalarF …
  -/
  simpa only [mul_eq_mul_right_iff (M₀ := ℝ), int_g_pos.ne', or_false, ← NNReal.eq_iff] using Z
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-05")] alias addHaarScalarFactor_eq_add := addHaarScalarFactor_eq_mul


/-- The scalar factor between two left-invariant measures is non-zero when both measures are
positive on open sets. -/
@[to_additive]
lemma haarScalarFactor_pos_of_isHaarMeasure (μ' μ : Measure G) [IsHaarMeasure μ]
    [IsHaarMeasure μ'] : 0 < haarScalarFactor μ' μ :=
                                /-
                                  G : Type u_1
                                  inst✝⁶ : TopologicalSpace G
                                  inst✝⁵ : Group G
                                  inst✝⁴ : TopologicalGroup G
                                  inst✝³ : MeasurableSpace G
                                  inst✝² : BorelSpace G
                                  μ' μ : MeasureTheory.Measure G
                                  inst✝¹ : μ.IsHaarMeasure
                                  inst✝ : μ'.IsHaarMeasure
                                  H : Eq (μ'.haarScalarFactor μ) 0
                                  ⊢ False
                                -/
  pos_iff_ne_zero.2 (fun H ↦ by simpa [H] using haarScalarFactor_eq_mul μ' μ μ')
                                /-
                                  🎉 no goals
                                -/


@[deprecated (since := "2024-02-12")]
alias haarScalarFactor_pos_of_isOpenPosMeasure := haarScalarFactor_pos_of_isHaarMeasure


@[deprecated (since := "2024-02-12")]
alias addHaarScalarFactor_pos_of_isOpenPosMeasure := addHaarScalarFactor_pos_of_isAddHaarMeasure


/-- Two left invariant measures give the same mass to level sets of continuous compactly supported
functions, up to the scalar `haarScalarFactor μ' μ`.

Auxiliary lemma in the proof of the more general
`measure_isMulInvariant_eq_smul_of_isCompact_closure`, which works for any set with
compact closure. -/
@[to_additive measure_preimage_isAddLeftInvariant_eq_smul_of_hasCompactSupport
"Two left invariant measures give the same mass to level sets of continuous compactly supported
functions, up to the scalar `addHaarScalarFactor μ' μ`.

Auxiliary lemma in the proof of the more general
`measure_isAddInvariant_eq_smul_of_isCompact_closure`, which works for any set with
compact closure."]
lemma measure_preimage_isMulLeftInvariant_eq_smul_of_hasCompactSupport
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ']
    {f : G → ℝ} (hf : Continuous f) (h'f : HasCompactSupport f) :
    μ' (f ⁻¹' {1}) = haarScalarFactor μ' μ • μ (f ⁻¹' {1}) := by
  /- This follows from the fact that the two measures integrate in the same way continuous
  functions, by approximating the indicator function of `f ⁻¹' {1}` by continuous functions
  (namely `vₙ ∘ f` where `vₙ` is equal to `1` at `1`, and `0` outside of a small neighborhood
  `(1 - uₙ, 1 + uₙ)` where `uₙ` is a sequence tending to `0`).
  We use `vₙ = thickenedIndicator uₙ {1}` to take advantage of existing lemmas. -/
  obtain ⟨u, -, u_mem, u_lim⟩ : ∃ u, StrictAnti u ∧ (∀ (n : ℕ), u n ∈ Ioo 0 1)
    ∧ Tendsto u atTop (𝓝 0) := exists_seq_strictAnti_tendsto' (zero_lt_one : (0 : ℝ) < 1)
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    f : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    u : Nat → Real
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    ⊢ Eq (μ' (Set.preimage f (Singleton.singleton 1))) (HSMul.hSMul (μ'.haarScalar …
  -/
  let v : ℕ → ℝ → ℝ := fun n x ↦ thickenedIndicator (u_mem n).1 ({1} : Set ℝ) x
  have vf_cont n : Continuous ((v n) ∘ f) := by
    apply Continuous.comp (continuous_induced_dom.comp ?_) hf
    exact BoundedContinuousFunction.continuous (thickenedIndicator (u_mem n).left {1})
  have I : ∀ (ν : Measure G), IsFiniteMeasureOnCompacts ν →
      Tendsto (fun n ↦ ∫ x, v n (f x) ∂ν) atTop
      (𝓝 (∫ x, Set.indicator ({1} : Set ℝ) (fun _ ↦ 1) (f x) ∂ν)) := by
    intro ν hν
    apply tendsto_integral_of_dominated_convergence
        (bound := (tsupport f).indicator (fun (_ : G) ↦ (1 : ℝ)) )
    · exact fun n ↦ (vf_cont n).aestronglyMeasurable
    · apply IntegrableOn.integrable_indicator _ (isClosed_tsupport f).measurableSet
      simpa using IsCompact.measure_lt_top h'f
    · refine fun n ↦ Eventually.of_forall (fun x ↦ ?_)
      by_cases hx : x ∈ tsupport f
      · simp only [v, Real.norm_eq_abs, NNReal.abs_eq, hx, indicator_of_mem]
        norm_cast
        exact thickenedIndicator_le_one _ _ _
      · simp only [v, Real.norm_eq_abs, NNReal.abs_eq, hx, not_false_eq_true, indicator_of_not_mem]
        rw [thickenedIndicator_zero]
        · simp
        · simpa [image_eq_zero_of_nmem_tsupport hx] using (u_mem n).2.le
    · filter_upwards with x
      have T := tendsto_pi_nhds.1 (thickenedIndicator_tendsto_indicator_closure
        (fun n ↦ (u_mem n).1) u_lim ({1} : Set ℝ)) (f x)
      simp only [thickenedIndicator_apply, closure_singleton] at T
      convert NNReal.tendsto_coe.2 T
      simp
  have M n : ∫ (x : G), v n (f x) ∂μ' = ∫ (x : G), v n (f x) ∂(haarScalarFactor μ' μ • μ) := by
    apply integral_isMulLeftInvariant_eq_smul_of_hasCompactSupport μ' μ (vf_cont n)
    apply h'f.comp_left
    simp only [v, thickenedIndicator_apply, NNReal.coe_eq_zero]
    rw [thickenedIndicatorAux_zero (u_mem n).1]
    · simp only [ENNReal.zero_toNNReal]
    · simpa using (u_mem n).2.le
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    f : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    u : Nat → Real
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Real → Real := fun n x => ↑((thickenedIndicator ⋯ (Singleton.singlet …
    vf_cont : ∀ (n : Nat), Continuous (Function.comp (v n) f)
    I : ∀ (ν : MeasureTheory.Measure G), MeasureTheory.IsFiniteMeasureOnCompacts ν …
    M : ∀ (n : Nat), Eq (MeasureTheory.integral μ' fun x => v n (f x)) (MeasureThe …
    ⊢ Eq (μ' (Set.preimage f (Singleton.singleton 1))) (HSMul.hSMul (μ'.haarScalar …
  -/
  have I1 := I μ' (by infer_instance)
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    f : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    u : Nat → Real
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Real → Real := fun n x => ↑((thickenedIndicator ⋯ (Singleton.singlet …
    vf_cont : ∀ (n : Nat), Continuous (Function.comp (v n) f)
    I : ∀ (ν : MeasureTheory.Measure G), MeasureTheory.IsFiniteMeasureOnCompacts ν …
    M : ∀ (n : Nat), Eq (MeasureTheory.integral μ' fun x => v n (f x)) (MeasureThe …
    I1 : Filter.Tendsto (fun n => MeasureTheory.integral μ' fun x => v n (f x)) Fi …
    ⊢ Eq (μ' (Set.preimage f (Singleton.singleton 1))) (HSMul.hSMul (μ'.haarScalar …
  -/
  simp_rw [M] at I1
  have J1 : ∫ (x : G), indicator {1} (fun _ ↦ (1 : ℝ)) (f x) ∂μ'
      = ∫ (x : G), indicator {1} (fun _ ↦ 1) (f x) ∂(haarScalarFactor μ' μ • μ) :=
    tendsto_nhds_unique I1 (I (haarScalarFactor μ' μ • μ) (by infer_instance))
  have J2 : ENNReal.toReal (μ' (f ⁻¹' {1}))
      = ENNReal.toReal ((haarScalarFactor μ' μ • μ) (f ⁻¹' {1})) := by
    have : (fun x ↦ indicator {1} (fun _ ↦ (1 : ℝ)) (f x)) =
        (fun x ↦ indicator (f ⁻¹' {1}) (fun _ ↦ (1 : ℝ)) x) := by
      ext x
      exact (indicator_comp_right f (s := ({1} : Set ℝ)) (g := (fun _ ↦ (1 : ℝ))) (x := x)).symm
    have mf : MeasurableSet (f ⁻¹' {1}) := (isClosed_singleton.preimage hf).measurableSet
    simpa only [this, mf, integral_indicator_const, smul_eq_mul, mul_one, Pi.smul_apply,
      nnreal_smul_coe_apply, ENNReal.toReal_mul, ENNReal.coe_toReal] using J1
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    f : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    u : Nat → Real
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Real → Real := fun n x => ↑((thickenedIndicator ⋯ (Singleton.singlet …
    vf_cont : ∀ (n : Nat), Continuous (Function.comp (v n) f)
    I : ∀ (ν : MeasureTheory.Measure G), MeasureTheory.IsFiniteMeasureOnCompacts ν …
    M : ∀ (n : Nat), Eq (MeasureTheory.integral μ' fun x => v n (f x)) (MeasureThe …
    I1 : Filter.Tendsto (fun n => MeasureTheory.integral (HSMul.hSMul (μ'.haarScal …
    J1 : Eq (MeasureTheory.integral μ' fun x => (Singleton.singleton 1).indicator  …
    J2 : Eq (μ' (Set.preimage f (Singleton.singleton 1))).toReal ((HSMul.hSMul (μ' …
    ⊢ Eq (μ' (Set.preimage f (Singleton.singleton 1))) (HSMul.hSMul (μ'.haarScalar …
  -/
  have C : IsCompact (f ⁻¹' {1}) := h'f.isCompact_preimage hf isClosed_singleton (by simp)
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    f : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    u : Nat → Real
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Real → Real := fun n x => ↑((thickenedIndicator ⋯ (Singleton.singlet …
    vf_cont : ∀ (n : Nat), Continuous (Function.comp (v n) f)
    I : ∀ (ν : MeasureTheory.Measure G), MeasureTheory.IsFiniteMeasureOnCompacts ν …
    M : ∀ (n : Nat), Eq (MeasureTheory.integral μ' fun x => v n (f x)) (MeasureThe …
    I1 : Filter.Tendsto (fun n => MeasureTheory.integral (HSMul.hSMul (μ'.haarScal …
    J1 : Eq (MeasureTheory.integral μ' fun x => (Singleton.singleton 1).indicator  …
    J2 : Eq (μ' (Set.preimage f (Singleton.singleton 1))).toReal ((HSMul.hSMul (μ' …
    C : IsCompact (Set.preimage f (Singleton.singleton 1))
    ⊢ Eq (μ' (Set.preimage f (Singleton.singleton 1))) (HSMul.hSMul (μ'.haarScalar …
  -/
  rw [ENNReal.toReal_eq_toReal C.measure_lt_top.ne C.measure_lt_top.ne] at J2
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    f : G → Real
    hf : Continuous f
    h'f : HasCompactSupport f
    u : Nat → Real
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Real → Real := fun n x => ↑((thickenedIndicator ⋯ (Singleton.singlet …
    vf_cont : ∀ (n : Nat), Continuous (Function.comp (v n) f)
    I : ∀ (ν : MeasureTheory.Measure G), MeasureTheory.IsFiniteMeasureOnCompacts ν …
    M : ∀ (n : Nat), Eq (MeasureTheory.integral μ' fun x => v n (f x)) (MeasureThe …
    I1 : Filter.Tendsto (fun n => MeasureTheory.integral (HSMul.hSMul (μ'.haarScal …
    J1 : Eq (MeasureTheory.integral μ' fun x => (Singleton.singleton 1).indicator  …
    J2 : Eq (μ' (Set.preimage f (Singleton.singleton 1))) ((HSMul.hSMul (μ'.haarSc …
    C : IsCompact (Set.preimage f (Singleton.singleton 1))
    ⊢ Eq (μ' (Set.preimage f (Singleton.singleton 1))) (HSMul.hSMul (μ'.haarScalar …
  -/
  simpa using J2
  /-
    🎉 no goals
  -/


/-- If an invariant measure is inner regular, then it gives less mass to sets with compact closure
than any other invariant measure, up to the scalar `haarScalarFactor μ' μ`.

Auxiliary lemma in the proof of the more general
`measure_isMulInvariant_eq_smul_of_isCompact_closure`, which gives equality for any
set with compact closure. -/
@[to_additive smul_measure_isAddInvariant_le_of_isCompact_closure
"If an invariant measure is inner regular, then it gives less mass to sets with compact closure
than any other invariant measure, up to the scalar `addHaarScalarFactor μ' μ`.

Auxiliary lemma in the proof of the more general
`measure_isAddInvariant_eq_smul_of_isCompact_closure`, which gives equality for any
set with compact closure."]
lemma smul_measure_isMulInvariant_le_of_isCompact_closure [LocallyCompactSpace G]
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ']
    [InnerRegularCompactLTTop μ]
    {s : Set G} (hs : MeasurableSet s) (h's : IsCompact (closure s)) :
    haarScalarFactor μ' μ • μ s ≤ μ' s := by
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : μ.InnerRegularCompactLTTop
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    ⊢ LE.le (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s)) (μ' s)
  -/
  apply le_of_forall_lt (fun r hr ↦ ?_)
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : μ.InnerRegularCompactLTTop
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    r : ENNReal
    hr : LT.lt r (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
    ⊢ LT.lt r (μ' s)
  -/
  let ν := haarScalarFactor μ' μ • μ
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : μ.InnerRegularCompactLTTop
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    r : ENNReal
    hr : LT.lt r (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    ⊢ LT.lt r (μ' s)
  -/
  have : ν s ≠ ∞ := ((measure_mono subset_closure).trans_lt h's.measure_lt_top).ne
  obtain ⟨-, hf, ⟨f, f_cont, f_comp, rfl⟩, νf⟩ :
      ∃ K ⊆ s, (∃ f, Continuous f ∧ HasCompactSupport f ∧ K = f ⁻¹' {1}) ∧ r < ν K :=
    innerRegularWRT_preimage_one_hasCompactSupport_measure_ne_top_of_group ⟨hs, this⟩ r
      (by convert hr)
  calc
  r < ν (f ⁻¹' {1}) := νf
  _ = μ' (f ⁻¹' {1}) :=
    (measure_preimage_isMulLeftInvariant_eq_smul_of_hasCompactSupport _ _ f_cont f_comp).symm
  _ ≤ μ' s := measure_mono hf


/-- If an invariant measure is inner regular, then it gives the same mass to measurable sets with
compact closure as any other invariant measure, up to the scalar `haarScalarFactor μ' μ`.

Auxiliary lemma in the proof of the more general
`measure_isMulInvariant_eq_smul_of_isCompact_closure`, which works for any set with
compact closure, and removes the inner regularity assumption. -/
@[to_additive measure_isAddInvariant_eq_smul_of_isCompact_closure_of_innerRegularCompactLTTop
" If an invariant measure is inner regular, then it gives the same mass to measurable sets with
compact closure as any other invariant measure, up to the scalar `addHaarScalarFactor μ' μ`.

Auxiliary lemma in the proof of the more general
`measure_isAddInvariant_eq_smul_of_isCompact_closure`, which works for any set with
compact closure, and removes the inner regularity assumption."]
lemma measure_isMulInvariant_eq_smul_of_isCompact_closure_of_innerRegularCompactLTTop
    [LocallyCompactSpace G]
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ']
    [InnerRegularCompactLTTop μ]
    {s : Set G} (hs : MeasurableSet s) (h's : IsCompact (closure s)) :
    μ' s = haarScalarFactor μ' μ • μ s := by
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : μ.InnerRegularCompactLTTop
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    ⊢ Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  apply le_antisymm ?_ (smul_measure_isMulInvariant_le_of_isCompact_closure μ' μ hs h's)
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : μ.InnerRegularCompactLTTop
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    ⊢ LE.le (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  let ν := haarScalarFactor μ' μ • μ
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : μ.InnerRegularCompactLTTop
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    ⊢ LE.le (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  change μ' s ≤ ν s
  obtain ⟨⟨f, f_cont⟩, hf, -, f_comp, -⟩ : ∃ f : C(G, ℝ), EqOn f 1 (closure s) ∧ EqOn f 0 ∅
      ∧ HasCompactSupport f ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1 :=
    exists_continuous_one_zero_of_isCompact h's isClosed_empty (disjoint_empty _)
  /-
    case intro.mk.intro.intro.intro
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : μ.InnerRegularCompactLTTop
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    f : G → Real
    f_cont : Continuous f
    hf : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 1 (closure s)
    f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
    ⊢ LE.le (μ' s) (ν s)
  -/
  let t := f ⁻¹' {1}
  /-
    case intro.mk.intro.intro.intro
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : μ.InnerRegularCompactLTTop
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    f : G → Real
    f_cont : Continuous f
    hf : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 1 (closure s)
    f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
    t : Set G := Set.preimage f (Singleton.singleton 1)
    ⊢ LE.le (μ' s) (ν s)
  -/
  have t_closed : IsClosed t := isClosed_singleton.preimage f_cont
  /-
    case intro.mk.intro.intro.intro
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : μ.InnerRegularCompactLTTop
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    f : G → Real
    f_cont : Continuous f
    hf : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 1 (closure s)
    f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
    t : Set G := Set.preimage f (Singleton.singleton 1)
    t_closed : IsClosed t
    ⊢ LE.le (μ' s) (ν s)
  -/
  have t_comp : IsCompact t := f_comp.isCompact_preimage f_cont isClosed_singleton (by simp)
  /-
    case intro.mk.intro.intro.intro
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : μ.IsHaarMeasure
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : μ.InnerRegularCompactLTTop
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    f : G → Real
    f_cont : Continuous f
    hf : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 1 (closure s)
    f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
    t : Set G := Set.preimage f (Singleton.singleton 1)
    t_closed : IsClosed t
    t_comp : IsCompact t
    ⊢ LE.le (μ' s) (ν s)
  -/
  have st : s ⊆ t := (IsClosed.closure_subset_iff t_closed).mp hf
  have A : ν (t \ s) ≤ μ' (t \ s) := by
    apply smul_measure_isMulInvariant_le_of_isCompact_closure _ _ (t_closed.measurableSet.diff hs)
    exact t_comp.closure_of_subset diff_subset
  have B : μ' t = ν t :=
    measure_preimage_isMulLeftInvariant_eq_smul_of_hasCompactSupport _ _ f_cont f_comp
  rwa [measure_diff st hs.nullMeasurableSet, measure_diff st hs.nullMeasurableSet, ← B,
    ENNReal.sub_le_sub_iff_left] at A
    /-
      case intro.mk.intro.intro.intro.h
      G : Type u_1
      inst✝⁹ : TopologicalSpace G
      inst✝⁸ : Group G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : MeasurableSpace G
      inst✝⁵ : BorelSpace G
      inst✝⁴ : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝³ : μ.IsHaarMeasure
      inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
      inst✝¹ : μ'.IsMulLeftInvariant
      inst✝ : μ.InnerRegularCompactLTTop
      s : Set G
      hs : MeasurableSet s
      h's : IsCompact (closure s)
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      f : G → Real
      f_cont : Continuous f
      hf : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 1 (closure s)
      f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
      t : Set G := Set.preimage f (Singleton.singleton 1)
      t_closed : IsClosed t
      t_comp : IsCompact t
      st : HasSubset.Subset s t
      A : LE.le (HSub.hSub (μ' t) (ν s)) (HSub.hSub (μ' t) (μ' s))
      B : Eq (μ' t) (ν t)
      ⊢ LE.le (μ' s) (μ' t)
    -/
  · exact measure_mono st
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.intro.intro.h'
      G : Type u_1
      inst✝⁹ : TopologicalSpace G
      inst✝⁸ : Group G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : MeasurableSpace G
      inst✝⁵ : BorelSpace G
      inst✝⁴ : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝³ : μ.IsHaarMeasure
      inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
      inst✝¹ : μ'.IsMulLeftInvariant
      inst✝ : μ.InnerRegularCompactLTTop
      s : Set G
      hs : MeasurableSet s
      h's : IsCompact (closure s)
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      f : G → Real
      f_cont : Continuous f
      hf : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 1 (closure s)
      f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
      t : Set G := Set.preimage f (Singleton.singleton 1)
      t_closed : IsClosed t
      t_comp : IsCompact t
      st : HasSubset.Subset s t
      A : LE.le (HSub.hSub (μ' t) (ν s)) (HSub.hSub (μ' t) (μ' s))
      B : Eq (μ' t) (ν t)
      ⊢ Ne (μ' t) Top.top
    -/
  · exact t_comp.measure_lt_top.ne
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.intro.intro
      G : Type u_1
      inst✝⁹ : TopologicalSpace G
      inst✝⁸ : Group G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : MeasurableSpace G
      inst✝⁵ : BorelSpace G
      inst✝⁴ : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝³ : μ.IsHaarMeasure
      inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
      inst✝¹ : μ'.IsMulLeftInvariant
      inst✝ : μ.InnerRegularCompactLTTop
      s : Set G
      hs : MeasurableSet s
      h's : IsCompact (closure s)
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      f : G → Real
      f_cont : Continuous f
      hf : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 1 (closure s)
      f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
      t : Set G := Set.preimage f (Singleton.singleton 1)
      t_closed : IsClosed t
      t_comp : IsCompact t
      st : HasSubset.Subset s t
      A : LE.le (HSub.hSub (ν t) (ν s)) (μ' (SDiff.sdiff t s))
      B : Eq (μ' t) (ν t)
      ⊢ Ne (μ' s) Top.top
    -/
  · exact ((measure_mono st).trans_lt t_comp.measure_lt_top).ne
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.intro.intro
      G : Type u_1
      inst✝⁹ : TopologicalSpace G
      inst✝⁸ : Group G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : MeasurableSpace G
      inst✝⁵ : BorelSpace G
      inst✝⁴ : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝³ : μ.IsHaarMeasure
      inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ'
      inst✝¹ : μ'.IsMulLeftInvariant
      inst✝ : μ.InnerRegularCompactLTTop
      s : Set G
      hs : MeasurableSet s
      h's : IsCompact (closure s)
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      f : G → Real
      f_cont : Continuous f
      hf : Set.EqOn (⇑{ toFun := f, continuous_toFun := f_cont }) 1 (closure s)
      f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
      t : Set G := Set.preimage f (Singleton.singleton 1)
      t_closed : IsClosed t
      t_comp : IsCompact t
      st : HasSubset.Subset s t
      A : LE.le (ν (SDiff.sdiff t s)) (μ' (SDiff.sdiff t s))
      B : Eq (μ' t) (ν t)
      ⊢ Ne (ν s) Top.top
    -/
  · exact ((measure_mono st).trans_lt t_comp.measure_lt_top).ne
    /-
      🎉 no goals
    -/


/-- Given an invariant measure then it gives the same mass to measurable sets with
compact closure as any other invariant measure, up to the scalar `haarScalarFactor μ' μ`.

Auxiliary lemma in the proof of the more general
`measure_isMulInvariant_eq_smul_of_isCompact_closure`, which removes the
measurability assumption. -/
@[to_additive measure_isAddInvariant_eq_smul_of_isCompact_closure_of_measurableSet
"Given an invariant measure then it gives the same mass to measurable sets with
compact closure as any other invariant measure, up to the scalar `addHaarScalarFactor μ' μ`.

Auxiliary lemma in the proof of the more general
`measure_isAddInvariant_eq_smul_of_isCompact_closure`, which removes the
measurability assumption."]
lemma measure_isMulInvariant_eq_smul_of_isCompact_closure_of_measurableSet [LocallyCompactSpace G]
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ']
    {s : Set G} (hs : MeasurableSet s) (h's : IsCompact (closure s)) :
    μ' s = haarScalarFactor μ' μ • μ s := by
  /-
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : BorelSpace G
    inst✝³ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    ⊢ Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  let ν : Measure G := haar
  have A : μ' s = haarScalarFactor μ' ν • ν s :=
    measure_isMulInvariant_eq_smul_of_isCompact_closure_of_innerRegularCompactLTTop μ' ν hs h's
  have B : μ s = haarScalarFactor μ ν • ν s :=
    measure_isMulInvariant_eq_smul_of_isCompact_closure_of_innerRegularCompactLTTop μ ν hs h's
  /-
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : BorelSpace G
    inst✝³ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    s : Set G
    hs : MeasurableSet s
    h's : IsCompact (closure s)
    ν : MeasureTheory.Measure G := MeasureTheory.Measure.haar
    A : Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor ν) (ν s))
    B : Eq (μ s) (HSMul.hSMul (μ.haarScalarFactor ν) (ν s))
    ⊢ Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  rw [A, B, smul_smul, haarScalarFactor_eq_mul μ' μ ν]
  /-
    🎉 no goals
  -/


/-- **Uniqueness of left-invariant measures**:
Given two left-invariant measures which are finite on compacts, they coincide in the following
sense: they give the same value to sets with compact closure, up to the multiplicative
constant `haarScalarFactor μ' μ`. -/
@[to_additive measure_isAddInvariant_eq_smul_of_isCompact_closure
"**Uniqueness of left-invariant measures**:
Given two left-invariant measures which are finite on compacts, they coincide in the following
sense: they give the same value to sets with compact closure, up to the multiplicative
constant `addHaarScalarFactor μ' μ`. "]
theorem measure_isMulInvariant_eq_smul_of_isCompact_closure [LocallyCompactSpace G]
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ']
    {s : Set G} (h's : IsCompact (closure s)) :
    μ' s = haarScalarFactor μ' μ • μ s := by
  /-
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : BorelSpace G
    inst✝³ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    s : Set G
    h's : IsCompact (closure s)
    ⊢ Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  let ν := haarScalarFactor μ' μ • μ
  /-
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : BorelSpace G
    inst✝³ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝ : μ'.IsMulLeftInvariant
    s : Set G
    h's : IsCompact (closure s)
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    ⊢ Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  apply le_antisymm
  · calc
    μ' s ≤ μ' ((toMeasurable ν s) ∩ (closure s)) :=
      measure_mono <| subset_inter (subset_toMeasurable ν s) subset_closure
    _ = ν ((toMeasurable ν s) ∩ (closure s)) := by
      apply measure_isMulInvariant_eq_smul_of_isCompact_closure_of_measurableSet _ _ _ _
      · exact (measurableSet_toMeasurable ν s).inter isClosed_closure.measurableSet
      · exact h's.closure_of_subset inter_subset_right
    _ ≤ ν (toMeasurable ν s) := measure_mono inter_subset_left
    _ = ν s := measure_toMeasurable s
  · calc
    ν s ≤ ν ((toMeasurable μ' s) ∩ (closure s)) :=
      measure_mono <| subset_inter (subset_toMeasurable μ' s) subset_closure
    _ = μ' ((toMeasurable μ' s) ∩ (closure s)) := by
      apply (measure_isMulInvariant_eq_smul_of_isCompact_closure_of_measurableSet _ _ _ _).symm
      · exact (measurableSet_toMeasurable μ' s).inter isClosed_closure.measurableSet
      · exact h's.closure_of_subset inter_subset_right
    _ ≤ μ' (toMeasurable μ' s) := measure_mono inter_subset_left
    _ = μ' s := measure_toMeasurable s


/-- **Uniqueness of Haar measures**:
Two Haar measures on a compact group coincide up to a multiplicative factor. -/
@[to_additive isAddInvariant_eq_smul_of_compactSpace]
lemma isMulInvariant_eq_smul_of_compactSpace [CompactSpace G] (μ' μ : Measure G)
    [IsHaarMeasure μ] [IsMulLeftInvariant μ'] [IsFiniteMeasureOnCompacts μ'] :
    μ' = haarScalarFactor μ' μ • μ := by
  /-
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : BorelSpace G
    inst✝³ : CompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    ⊢ Eq μ' (HSMul.hSMul (μ'.haarScalarFactor μ) μ)
  -/
  ext s _hs
  /-
    case h
    G : Type u_1
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : BorelSpace G
    inst✝³ : CompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    s : Set G
    _hs : MeasurableSet s
    ⊢ Eq (μ' s) ((HSMul.hSMul (μ'.haarScalarFactor μ) μ) s)
  -/
  exact measure_isMulInvariant_eq_smul_of_isCompact_closure _ _ isClosed_closure.isCompact
  /-
    🎉 no goals
  -/


@[to_additive]
instance (priority := 100) instInnerRegularOfIsHaarMeasureOfCompactSpace
    [CompactSpace G] (μ : Measure G) [IsMulLeftInvariant μ] [IsFiniteMeasureOnCompacts μ] :
    InnerRegular μ := by
  /-
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : CompactSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    ⊢ μ.InnerRegular
  -/
  rw [isMulInvariant_eq_smul_of_compactSpace μ haar]
  /-
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : CompactSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    ⊢ (HSMul.hSMul (μ.haarScalarFactor MeasureTheory.Measure.haar) MeasureTheory.M …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[to_additive]
instance (priority := 100) instRegularOfIsHaarMeasureOfCompactSpace
    [CompactSpace G] (μ : Measure G) [IsMulLeftInvariant μ] [IsFiniteMeasureOnCompacts μ] :
    Regular μ := by
  /-
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : CompactSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    ⊢ μ.Regular
  -/
  rw [isMulInvariant_eq_smul_of_compactSpace μ haar]
  /-
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : CompactSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    ⊢ (HSMul.hSMul (μ.haarScalarFactor MeasureTheory.Measure.haar) MeasureTheory.M …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- **Uniqueness of Haar measures**:
Two Haar measures which are probability measures coincide. -/
@[to_additive]
lemma isHaarMeasure_eq_of_isProbabilityMeasure [LocallyCompactSpace G] (μ' μ : Measure G)
    [IsProbabilityMeasure μ] [IsProbabilityMeasure μ'] [IsHaarMeasure μ] [IsHaarMeasure μ'] :
    μ' = μ := by
  have : CompactSpace G := by
    by_contra H
    rw [not_compactSpace_iff] at H
    simpa using measure_univ_of_isMulLeftInvariant μ
  have A s : μ' s = haarScalarFactor μ' μ • μ s :=
    measure_isMulInvariant_eq_smul_of_isCompact_closure _ _ isClosed_closure.isCompact
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.IsProbabilityMeasure μ
    inst✝² : MeasureTheory.IsProbabilityMeasure μ'
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    this : CompactSpace G
    A : ∀ (s : Set G), Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
    ⊢ Eq μ' μ
  -/
  have Z := A univ
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.IsProbabilityMeasure μ
    inst✝² : MeasureTheory.IsProbabilityMeasure μ'
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    this : CompactSpace G
    A : ∀ (s : Set G), Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
    Z : Eq (μ' Set.univ) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ Set.univ))
    ⊢ Eq μ' μ
  -/
  simp only [measure_univ, ENNReal.smul_def, smul_eq_mul, mul_one, ENNReal.one_eq_coe] at Z
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.IsProbabilityMeasure μ
    inst✝² : MeasureTheory.IsProbabilityMeasure μ'
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    this : CompactSpace G
    A : ∀ (s : Set G), Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
    Z : Eq 1 (μ'.haarScalarFactor μ)
    ⊢ Eq μ' μ
  -/
  ext s _hs
  /-
    case h
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.IsProbabilityMeasure μ
    inst✝² : MeasureTheory.IsProbabilityMeasure μ'
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    this : CompactSpace G
    A : ∀ (s : Set G), Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
    Z : Eq 1 (μ'.haarScalarFactor μ)
    s : Set G
    _hs : MeasurableSet s
    ⊢ Eq (μ' s) (μ s)
  -/
  simp [A s, ← Z]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-12")]
alias haarScalarFactor_eq_one_of_isProbabilityMeasure := isHaarMeasure_eq_of_isProbabilityMeasure


@[deprecated (since := "2024-02-12")]
alias addHaarScalarFactor_eq_one_of_isProbabilityMeasure :=
  isAddHaarMeasure_eq_of_isProbabilityMeasure


@[to_additive measure_isAddHaarMeasure_eq_smul_of_isEverywherePos]
theorem measure_isHaarMeasure_eq_smul_of_isEverywherePos [LocallyCompactSpace G]
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsHaarMeasure μ']
    {s : Set G} (hs : MeasurableSet s) (h's : IsEverywherePos μ s) :
    μ' s = haarScalarFactor μ' μ • μ s := by
  /-
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    s : Set G
    hs : MeasurableSet s
    h's : μ.IsEverywherePos s
    ⊢ Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  let ν := haarScalarFactor μ' μ • μ
  /-
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    s : Set G
    hs : MeasurableSet s
    h's : μ.IsEverywherePos s
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    ⊢ Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  change μ' s = ν s
  /- Fix a compact neighborhood `k` of the identity, and consider a maximal disjoint family `m` of
  sets `x • k` centered at points in `s`. Then `s` is covered by the sets `x • (k * k⁻¹)` by
  maximality. If the family is countable, then since `μ'` and `ν` coincide in compact sets, and
  the measure of a countable disjoint union is the sum of the measures, we get `μ' s = ν s`.
  Otherwise, the family is uncountable, and each intersection with `s` has positive measure by
  the everywhere positivity assumption, so `ν s = ∞`, and `μ' s = ∞` in the same way. -/
  obtain ⟨k, k_comp, k_closed, k_mem⟩ : ∃ k, IsCompact k ∧ IsClosed k ∧ k ∈ 𝓝 (1 : G) := by
    rcases exists_compact_mem_nhds (1 : G) with ⟨k, hk, hmem⟩
    exact ⟨closure k, hk.closure, isClosed_closure, mem_of_superset hmem subset_closure⟩
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    s : Set G
    hs : MeasurableSet s
    h's : μ.IsEverywherePos s
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    k : Set G
    k_comp : IsCompact k
    k_closed : IsClosed k
    k_mem : Membership.mem (nhds 1) k
    ⊢ Eq (μ' s) (ν s)
  -/
  have one_k : 1 ∈ k := mem_of_mem_nhds k_mem
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    s : Set G
    hs : MeasurableSet s
    h's : μ.IsEverywherePos s
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    k : Set G
    k_comp : IsCompact k
    k_closed : IsClosed k
    k_mem : Membership.mem (nhds 1) k
    one_k : Membership.mem k 1
    ⊢ Eq (μ' s) (ν s)
  -/
  let A : Set (Set G) := {t | t ⊆ s ∧ PairwiseDisjoint t (fun x ↦ x • k)}
  obtain ⟨m, m_max⟩ : ∃ m, Maximal (· ∈ A) m := by
    apply zorn_subset
    intro c cA hc
    refine ⟨⋃ a ∈ c, a, ⟨?_, ?_⟩, ?_⟩
    · simp only [iUnion_subset_iff]
      intro a ac x hx
      simp only [A, subset_def, mem_setOf_eq] at cA
      exact (cA _ ac).1 x hx
    · rintro x hx y hy hxy
      simp only [mem_iUnion, exists_prop] at hx hy
      rcases hx with ⟨a, ac, xa⟩
      rcases hy with ⟨b, bc, yb⟩
      obtain ⟨m, mc, am, bm⟩ : ∃ m ∈ c, a ⊆ m ∧ b ⊆ m := hc.directedOn _ ac _ bc
      exact (cA mc).2 (am xa) (bm yb) hxy
    · intro a ac
      exact subset_biUnion_of_mem (u := id) ac
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    s : Set G
    hs : MeasurableSet s
    h's : μ.IsEverywherePos s
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    k : Set G
    k_comp : IsCompact k
    k_closed : IsClosed k
    k_mem : Membership.mem (nhds 1) k
    one_k : Membership.mem k 1
    A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
    m : Set G
    m_max : Maximal (fun x => Membership.mem A x) m
    ⊢ Eq (μ' s) (ν s)
  -/
  obtain ⟨hms : m ⊆ s, hdj : PairwiseDisjoint m (fun x ↦ x • k)⟩ := m_max.prop
  have sm : s ⊆ ⋃ x ∈ m, x • (k * k⁻¹) := by
    intro y hy
    by_cases h'y : m ∪ {y} ∈ A
    · have ym : y ∈ m := m_max.mem_of_prop_insert (by simpa using h'y)
      have : y ∈ y • (k * k⁻¹) := by
        simpa using mem_leftCoset y (Set.mul_mem_mul one_k (Set.inv_mem_inv.mpr one_k))
      exact mem_biUnion ym this
    · obtain ⟨x, xm, -, z, zy, zx⟩ : ∃ x ∈ m, y ≠ x ∧ ∃ z, z ∈ y • k ∧ z ∈ x • k := by
        simpa [A, hms, hy, insert_subset_iff, pairwiseDisjoint_insert, hdj, not_disjoint_iff]
          using h'y
      have : y ∈ x • (k * k⁻¹) := by
        rw [show y = x * ((x⁻¹ * z) * (y⁻¹ * z)⁻¹) by group]
        have : (x⁻¹ * z) * (y⁻¹ * z)⁻¹ ∈ k * k⁻¹ := Set.mul_mem_mul ((mem_leftCoset_iff x).mp zx)
          (Set.inv_mem_inv.mpr ((mem_leftCoset_iff y).mp zy))
        exact mem_leftCoset x this
      exact mem_biUnion xm this
  /-
    case intro.intro.intro.intro.intro
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    s : Set G
    hs : MeasurableSet s
    h's : μ.IsEverywherePos s
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    k : Set G
    k_comp : IsCompact k
    k_closed : IsClosed k
    k_mem : Membership.mem (nhds 1) k
    one_k : Membership.mem k 1
    A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
    m : Set G
    m_max : Maximal (fun x => Membership.mem A x) m
    hms : HasSubset.Subset m s
    hdj : m.PairwiseDisjoint fun x => HSMul.hSMul x k
    sm : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => HSMul.hSMul x …
    ⊢ Eq (μ' s) (ν s)
  -/
  rcases eq_empty_or_nonempty m with rfl|hm
    /-
      case intro.intro.intro.intro.intro.inl
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      m_max : Maximal (fun x => Membership.mem A x) EmptyCollection.emptyCollection
      hms : HasSubset.Subset EmptyCollection.emptyCollection s
      hdj : EmptyCollection.emptyCollection.PairwiseDisjoint fun x => HSMul.hSMul x k
      sm : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => HSMul.hSMul x …
      ⊢ Eq (μ' s) (ν s)
    -/
  · simp only [mem_empty_iff_false, iUnion_of_empty, iUnion_empty, subset_empty_iff] at sm
    /-
      case intro.intro.intro.intro.intro.inl
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      m_max : Maximal (fun x => Membership.mem A x) EmptyCollection.emptyCollection
      hms : HasSubset.Subset EmptyCollection.emptyCollection s
      hdj : EmptyCollection.emptyCollection.PairwiseDisjoint fun x => HSMul.hSMul x k
      sm : Eq s EmptyCollection.emptyCollection
      ⊢ Eq (μ' s) (ν s)
    -/
    simp [sm]
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.intro.intro.inr
    G : Type u_1
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsHaarMeasure
    inst✝ : μ'.IsHaarMeasure
    s : Set G
    hs : MeasurableSet s
    h's : μ.IsEverywherePos s
    ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
    k : Set G
    k_comp : IsCompact k
    k_closed : IsClosed k
    k_mem : Membership.mem (nhds 1) k
    one_k : Membership.mem k 1
    A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
    m : Set G
    m_max : Maximal (fun x => Membership.mem A x) m
    hms : HasSubset.Subset m s
    hdj : m.PairwiseDisjoint fun x => HSMul.hSMul x k
    sm : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => HSMul.hSMul x …
    hm : m.Nonempty
    ⊢ Eq (μ' s) (ν s)
  -/
  by_cases h'm : Set.Countable m
    /-
      case pos
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      m : Set G
      m_max : Maximal (fun x => Membership.mem A x) m
      hms : HasSubset.Subset m s
      hdj : m.PairwiseDisjoint fun x => HSMul.hSMul x k
      sm : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => HSMul.hSMul x …
      hm : m.Nonempty
      h'm : m.Countable
      ⊢ Eq (μ' s) (ν s)
    -/
  · rcases h'm.exists_eq_range hm with ⟨f, rfl⟩
    have M i : MeasurableSet (disjointed (fun n ↦ s ∩ f n • (k * k⁻¹)) i) := by
      apply MeasurableSet.disjointed (fun j ↦ hs.inter ?_)
      have : IsClosed (k • k⁻¹) := IsClosed.smul_left_of_isCompact k_closed.inv k_comp
      exact (IsClosed.smul this (f j)).measurableSet
    /-
      case pos.intro
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      f : Nat → G
      m_max : Maximal (fun x => Membership.mem A x) (Set.range f)
      hms : HasSubset.Subset (Set.range f) s
      hdj : (Set.range f).PairwiseDisjoint fun x => HSMul.hSMul x k
      sm : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => HSMul.hSMul x …
      hm : (Set.range f).Nonempty
      h'm : (Set.range f).Countable
      M : ∀ (i : Nat), MeasurableSet (disjointed (fun n => Inter.inter s (HSMul.hSMu …
      ⊢ Eq (μ' s) (ν s)
    -/
    simp only [mem_range, iUnion_exists, iUnion_iUnion_eq'] at sm
    /-
      case pos.intro
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      f : Nat → G
      m_max : Maximal (fun x => Membership.mem A x) (Set.range f)
      hms : HasSubset.Subset (Set.range f) s
      hdj : (Set.range f).PairwiseDisjoint fun x => HSMul.hSMul x k
      hm : (Set.range f).Nonempty
      h'm : (Set.range f).Countable
      M : ∀ (i : Nat), MeasurableSet (disjointed (fun n => Inter.inter s (HSMul.hSMu …
      sm : HasSubset.Subset s (Set.iUnion fun y => HSMul.hSMul (f y) (HMul.hMul k (I …
      ⊢ Eq (μ' s) (ν s)
    -/
    have s_eq : s = ⋃ n, s ∩ (f n • (k * k⁻¹)) := by rwa [← inter_iUnion, eq_comm, inter_eq_left]
    have I : μ' s = ∑' n, μ' (disjointed (fun n ↦ s ∩ f n • (k * k⁻¹)) n) := by
      rw [← measure_iUnion (disjoint_disjointed _) M, iUnion_disjointed, ← s_eq]
    have J : ν s = ∑' n, ν (disjointed (fun n ↦ s ∩ f n • (k * k⁻¹)) n) := by
      rw [← measure_iUnion (disjoint_disjointed _) M, iUnion_disjointed, ← s_eq]
    /-
      case pos.intro
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      f : Nat → G
      m_max : Maximal (fun x => Membership.mem A x) (Set.range f)
      hms : HasSubset.Subset (Set.range f) s
      hdj : (Set.range f).PairwiseDisjoint fun x => HSMul.hSMul x k
      hm : (Set.range f).Nonempty
      h'm : (Set.range f).Countable
      M : ∀ (i : Nat), MeasurableSet (disjointed (fun n => Inter.inter s (HSMul.hSMu …
      sm : HasSubset.Subset s (Set.iUnion fun y => HSMul.hSMul (f y) (HMul.hMul k (I …
      s_eq : Eq s (Set.iUnion fun n => Inter.inter s (HSMul.hSMul (f n) (HMul.hMul k …
      I : Eq (μ' s) (tsum fun n => μ' (disjointed (fun n => Inter.inter s (HSMul.hSM …
      J : Eq (ν s) (tsum fun n => ν (disjointed (fun n => Inter.inter s (HSMul.hSMul …
      ⊢ Eq (μ' s) (ν s)
    -/
    rw [I, J]
    /-
      case pos.intro
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      f : Nat → G
      m_max : Maximal (fun x => Membership.mem A x) (Set.range f)
      hms : HasSubset.Subset (Set.range f) s
      hdj : (Set.range f).PairwiseDisjoint fun x => HSMul.hSMul x k
      hm : (Set.range f).Nonempty
      h'm : (Set.range f).Countable
      M : ∀ (i : Nat), MeasurableSet (disjointed (fun n => Inter.inter s (HSMul.hSMu …
      sm : HasSubset.Subset s (Set.iUnion fun y => HSMul.hSMul (f y) (HMul.hMul k (I …
      s_eq : Eq s (Set.iUnion fun n => Inter.inter s (HSMul.hSMul (f n) (HMul.hMul k …
      I : Eq (μ' s) (tsum fun n => μ' (disjointed (fun n => Inter.inter s (HSMul.hSM …
      J : Eq (ν s) (tsum fun n => ν (disjointed (fun n => Inter.inter s (HSMul.hSMul …
      ⊢ Eq (tsum fun n => μ' (disjointed (fun n => Inter.inter s (HSMul.hSMul (f n)  …
    -/
    congr with n
    /-
      case pos.intro.e_f.h
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      f : Nat → G
      m_max : Maximal (fun x => Membership.mem A x) (Set.range f)
      hms : HasSubset.Subset (Set.range f) s
      hdj : (Set.range f).PairwiseDisjoint fun x => HSMul.hSMul x k
      hm : (Set.range f).Nonempty
      h'm : (Set.range f).Countable
      M : ∀ (i : Nat), MeasurableSet (disjointed (fun n => Inter.inter s (HSMul.hSMu …
      sm : HasSubset.Subset s (Set.iUnion fun y => HSMul.hSMul (f y) (HMul.hMul k (I …
      s_eq : Eq s (Set.iUnion fun n => Inter.inter s (HSMul.hSMul (f n) (HMul.hMul k …
      I : Eq (μ' s) (tsum fun n => μ' (disjointed (fun n => Inter.inter s (HSMul.hSM …
      J : Eq (ν s) (tsum fun n => ν (disjointed (fun n => Inter.inter s (HSMul.hSMul …
      n : Nat
      ⊢ Eq (μ' (disjointed (fun n => Inter.inter s (HSMul.hSMul (f n) (HMul.hMul k ( …
    -/
    apply measure_isMulInvariant_eq_smul_of_isCompact_closure
    /-
      case pos.intro.e_f.h.h's
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      f : Nat → G
      m_max : Maximal (fun x => Membership.mem A x) (Set.range f)
      hms : HasSubset.Subset (Set.range f) s
      hdj : (Set.range f).PairwiseDisjoint fun x => HSMul.hSMul x k
      hm : (Set.range f).Nonempty
      h'm : (Set.range f).Countable
      M : ∀ (i : Nat), MeasurableSet (disjointed (fun n => Inter.inter s (HSMul.hSMu …
      sm : HasSubset.Subset s (Set.iUnion fun y => HSMul.hSMul (f y) (HMul.hMul k (I …
      s_eq : Eq s (Set.iUnion fun n => Inter.inter s (HSMul.hSMul (f n) (HMul.hMul k …
      I : Eq (μ' s) (tsum fun n => μ' (disjointed (fun n => Inter.inter s (HSMul.hSM …
      J : Eq (ν s) (tsum fun n => ν (disjointed (fun n => Inter.inter s (HSMul.hSMul …
      n : Nat
      ⊢ IsCompact (closure (disjointed (fun n => Inter.inter s (HSMul.hSMul (f n) (H …
    -/
    have : IsCompact (f n • (k * k⁻¹)) := IsCompact.smul (f n) (k_comp.mul k_comp.inv)
    /-
      case pos.intro.e_f.h.h's
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      f : Nat → G
      m_max : Maximal (fun x => Membership.mem A x) (Set.range f)
      hms : HasSubset.Subset (Set.range f) s
      hdj : (Set.range f).PairwiseDisjoint fun x => HSMul.hSMul x k
      hm : (Set.range f).Nonempty
      h'm : (Set.range f).Countable
      M : ∀ (i : Nat), MeasurableSet (disjointed (fun n => Inter.inter s (HSMul.hSMu …
      sm : HasSubset.Subset s (Set.iUnion fun y => HSMul.hSMul (f y) (HMul.hMul k (I …
      s_eq : Eq s (Set.iUnion fun n => Inter.inter s (HSMul.hSMul (f n) (HMul.hMul k …
      I : Eq (μ' s) (tsum fun n => μ' (disjointed (fun n => Inter.inter s (HSMul.hSM …
      J : Eq (ν s) (tsum fun n => ν (disjointed (fun n => Inter.inter s (HSMul.hSMul …
      n : Nat
      this : IsCompact (HSMul.hSMul (f n) (HMul.hMul k (Inv.inv k)))
      ⊢ IsCompact (closure (disjointed (fun n => Inter.inter s (HSMul.hSMul (f n) (H …
    -/
    exact this.closure_of_subset <| (disjointed_subset _ _).trans inter_subset_right
    /-
      🎉 no goals
    -/
  · have H : ∀ (ρ : Measure G), IsEverywherePos ρ s → ρ s = ∞ := by
      intro ρ hρ
      have M : ∀ (i : ↑m), MeasurableSet (s ∩ (i : G) • k) :=
        fun i ↦ hs.inter (IsClosed.smul k_closed _).measurableSet
      contrapose! h'm
      have : ∑' (x : m), ρ (s ∩ ((x : G) • k)) < ∞ := by
        apply lt_of_le_of_lt (MeasureTheory.tsum_meas_le_meas_iUnion_of_disjoint _ M _) _
        · have I : PairwiseDisjoint m fun x ↦ s ∩ x • k :=
            hdj.mono (fun x ↦ inter_subset_right)
          exact I.on_injective Subtype.val_injective (fun x ↦ x.2)
        · exact lt_of_le_of_lt (measure_mono (by simp [inter_subset_left])) h'm.lt_top
      have C : Set.Countable (support fun (i : m) ↦ ρ (s ∩ (i : G) • k)) :=
        Summable.countable_support_ennreal this.ne
      have : support (fun (i : m) ↦ ρ (s ∩ (i : G) • k)) = univ := by
        refine eq_univ_iff_forall.2 fun i ↦ ?_
        refine ne_of_gt (hρ (i : G) (hms i.2) _ ?_)
        exact inter_mem_nhdsWithin s (by simpa)
      rw [this] at C
      have : Countable m := countable_univ_iff.mp C
      exact to_countable m
    have Hν : IsEverywherePos ν s :=
      h's.smul_measure_nnreal (haarScalarFactor_pos_of_isHaarMeasure _ _).ne'
    have Hμ' : IsEverywherePos μ' s := by
      apply Hν.of_forall_exists_nhds_eq (fun x _hx ↦ ?_)
      obtain ⟨t, t_comp, t_mem⟩ : ∃ t, IsCompact t ∧ t ∈ 𝓝 x := exists_compact_mem_nhds x
      refine ⟨t, t_mem, fun u hu ↦ ?_⟩
      apply measure_isMulInvariant_eq_smul_of_isCompact_closure
      exact t_comp.closure_of_subset hu
    /-
      case neg
      G : Type u_1
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : Group G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      inst✝² : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      inst✝ : μ'.IsHaarMeasure
      s : Set G
      hs : MeasurableSet s
      h's : μ.IsEverywherePos s
      ν : MeasureTheory.Measure G := HSMul.hSMul (μ'.haarScalarFactor μ) μ
      k : Set G
      k_comp : IsCompact k
      k_closed : IsClosed k
      k_mem : Membership.mem (nhds 1) k
      one_k : Membership.mem k 1
      A : Set (Set G) := setOf fun t => And (HasSubset.Subset t s) (t.PairwiseDisjoi …
      m : Set G
      m_max : Maximal (fun x => Membership.mem A x) m
      hms : HasSubset.Subset m s
      hdj : m.PairwiseDisjoint fun x => HSMul.hSMul x k
      sm : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => HSMul.hSMul x …
      hm : m.Nonempty
      h'm : Not m.Countable
      H : ∀ (ρ : MeasureTheory.Measure G), ρ.IsEverywherePos s → Eq (ρ s) Top.top
      Hν : ν.IsEverywherePos s
      Hμ' : μ'.IsEverywherePos s
      ⊢ Eq (μ' s) (ν s)
    -/
    rw [H ν Hν, H μ' Hμ']
    /-
      🎉 no goals
    -/


/-- **Uniqueness of Haar measures**:
Given two Haar measures, they coincide in the following sense: they give the same value to open
sets, up to the multiplicative constant `haarScalarFactor μ' μ`. -/
@[to_additive measure_isAddHaarMeasure_eq_smul_of_isOpen
"**Uniqueness of Haar measures**:
Given two additive Haar measures, they coincide in the following sense: they give the same value to
open sets, up to the multiplicative constant `addHaarScalarFactor μ' μ`."]
theorem measure_isHaarMeasure_eq_smul_of_isOpen [LocallyCompactSpace G]
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsHaarMeasure μ'] {s : Set G} (hs : IsOpen s) :
    μ' s = haarScalarFactor μ' μ • μ s :=
  measure_isHaarMeasure_eq_smul_of_isEverywherePos μ' μ hs.measurableSet hs.isEverywherePos


/-- **Uniqueness of left-invariant measures**:
Given two left-invariant measures which are finite on
compacts and inner regular for finite measure sets with respect to compact sets,
they coincide in the following sense: they give the same value to finite measure sets,
up to a multiplicative constant. -/
@[to_additive]
lemma measure_isMulLeftInvariant_eq_smul_of_ne_top [LocallyCompactSpace G]
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ']
    [InnerRegularCompactLTTop μ] [InnerRegularCompactLTTop μ'] {s : Set G}
    (hs : μ s ≠ ∞) (h's : μ' s ≠ ∞) : μ' s = haarScalarFactor μ' μ • μ s := by
  /- We know that the measures integrate in the same way continuous compactly supported functions,
  up to the factor `c = haarScalarFactor μ' μ`. -/
  /-
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : μ'.InnerRegularCompactLTTop
    s : Set G
    hs : Ne (μ s) Top.top
    h's : Ne (μ' s) Top.top
    ⊢ Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  let c := haarScalarFactor μ' μ
  /- By regularity, every measurable set of finite measure may be approximated by compact sets.
  Therefore, the measures coincide on measurable sets of finite measure. -/
  have B : ∀ s, MeasurableSet s → μ s < ∞ → μ' s < ∞ → μ' s = (c • μ) s := by
    intro s s_meas hs h's
    have : (c • μ) s ≠ ∞ := by simp [ENNReal.mul_eq_top, hs.ne]
    rw [s_meas.measure_eq_iSup_isCompact_of_ne_top h's.ne,
        s_meas.measure_eq_iSup_isCompact_of_ne_top this]
    congr! 4 with K _Ks K_comp
    exact measure_isMulInvariant_eq_smul_of_isCompact_closure μ' μ K_comp.closure
  /- Finally, replace an arbitrary finite measure set with a measurable version, and use the
  version for measurable sets. -/
  /-
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : μ'.InnerRegularCompactLTTop
    s : Set G
    hs : Ne (μ s) Top.top
    h's : Ne (μ' s) Top.top
    c : NNReal := μ'.haarScalarFactor μ
    B : ∀ (s : Set G), MeasurableSet s → LT.lt (μ s) Top.top → LT.lt (μ' s) Top.to …
    ⊢ Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  let t := toMeasurable μ' s ∩ toMeasurable μ s
  /-
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : μ'.InnerRegularCompactLTTop
    s : Set G
    hs : Ne (μ s) Top.top
    h's : Ne (μ' s) Top.top
    c : NNReal := μ'.haarScalarFactor μ
    B : ∀ (s : Set G), MeasurableSet s → LT.lt (μ s) Top.top → LT.lt (μ' s) Top.to …
    t : Set G := Inter.inter (MeasureTheory.toMeasurable μ' s) (MeasureTheory.toMe …
    ⊢ Eq (μ' s) (HSMul.hSMul (μ'.haarScalarFactor μ) (μ s))
  -/
  have st : s ⊆ t := subset_inter (subset_toMeasurable μ' s) (subset_toMeasurable μ s)
  have mu'_t : μ' t = μ' s := by
    apply le_antisymm
    · exact (measure_mono inter_subset_left).trans (measure_toMeasurable s).le
    · exact measure_mono st
  have mu_t : μ t = μ s := by
    apply le_antisymm
    · exact (measure_mono inter_subset_right).trans (measure_toMeasurable s).le
    · exact measure_mono st
  simp only [← mu'_t, smul_toOuterMeasure, OuterMeasure.coe_smul, Pi.smul_apply, ← mu_t,
    nnreal_smul_coe_apply]
  /-
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : μ'.InnerRegularCompactLTTop
    s : Set G
    hs : Ne (μ s) Top.top
    h's : Ne (μ' s) Top.top
    c : NNReal := μ'.haarScalarFactor μ
    B : ∀ (s : Set G), MeasurableSet s → LT.lt (μ s) Top.top → LT.lt (μ' s) Top.to …
    t : Set G := Inter.inter (MeasureTheory.toMeasurable μ' s) (MeasureTheory.toMe …
    st : HasSubset.Subset s t
    mu'_t : Eq (μ' t) (μ' s)
    mu_t : Eq (μ t) (μ s)
    ⊢ Eq (μ' t) (HMul.hMul (↑(μ'.haarScalarFactor μ)) (μ t))
  -/
  apply B
    /-
      case a
      G : Type u_1
      inst✝¹⁰ : TopologicalSpace G
      inst✝⁹ : Group G
      inst✝⁸ : TopologicalGroup G
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : BorelSpace G
      inst✝⁵ : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝⁴ : μ.IsHaarMeasure
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
      inst✝² : μ'.IsMulLeftInvariant
      inst✝¹ : μ.InnerRegularCompactLTTop
      inst✝ : μ'.InnerRegularCompactLTTop
      s : Set G
      hs : Ne (μ s) Top.top
      h's : Ne (μ' s) Top.top
      c : NNReal := μ'.haarScalarFactor μ
      B : ∀ (s : Set G), MeasurableSet s → LT.lt (μ s) Top.top → LT.lt (μ' s) Top.to …
      t : Set G := Inter.inter (MeasureTheory.toMeasurable μ' s) (MeasureTheory.toMe …
      st : HasSubset.Subset s t
      mu'_t : Eq (μ' t) (μ' s)
      mu_t : Eq (μ t) (μ s)
      ⊢ MeasurableSet t
    -/
  · exact (measurableSet_toMeasurable _ _).inter (measurableSet_toMeasurable _ _)
    /-
      🎉 no goals
    -/
    /-
      case a
      G : Type u_1
      inst✝¹⁰ : TopologicalSpace G
      inst✝⁹ : Group G
      inst✝⁸ : TopologicalGroup G
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : BorelSpace G
      inst✝⁵ : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝⁴ : μ.IsHaarMeasure
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
      inst✝² : μ'.IsMulLeftInvariant
      inst✝¹ : μ.InnerRegularCompactLTTop
      inst✝ : μ'.InnerRegularCompactLTTop
      s : Set G
      hs : Ne (μ s) Top.top
      h's : Ne (μ' s) Top.top
      c : NNReal := μ'.haarScalarFactor μ
      B : ∀ (s : Set G), MeasurableSet s → LT.lt (μ s) Top.top → LT.lt (μ' s) Top.to …
      t : Set G := Inter.inter (MeasureTheory.toMeasurable μ' s) (MeasureTheory.toMe …
      st : HasSubset.Subset s t
      mu'_t : Eq (μ' t) (μ' s)
      mu_t : Eq (μ t) (μ s)
      ⊢ LT.lt (μ t) Top.top
    -/
  · exact mu_t.le.trans_lt hs.lt_top
    /-
      🎉 no goals
    -/
    /-
      case a
      G : Type u_1
      inst✝¹⁰ : TopologicalSpace G
      inst✝⁹ : Group G
      inst✝⁸ : TopologicalGroup G
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : BorelSpace G
      inst✝⁵ : LocallyCompactSpace G
      μ' μ : MeasureTheory.Measure G
      inst✝⁴ : μ.IsHaarMeasure
      inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
      inst✝² : μ'.IsMulLeftInvariant
      inst✝¹ : μ.InnerRegularCompactLTTop
      inst✝ : μ'.InnerRegularCompactLTTop
      s : Set G
      hs : Ne (μ s) Top.top
      h's : Ne (μ' s) Top.top
      c : NNReal := μ'.haarScalarFactor μ
      B : ∀ (s : Set G), MeasurableSet s → LT.lt (μ s) Top.top → LT.lt (μ' s) Top.to …
      t : Set G := Inter.inter (MeasureTheory.toMeasurable μ' s) (MeasureTheory.toMe …
      st : HasSubset.Subset s t
      mu'_t : Eq (μ' t) (μ' s)
      mu_t : Eq (μ t) (μ s)
      ⊢ LT.lt (μ' t) Top.top
    -/
  · exact mu'_t.le.trans_lt h's.lt_top
    /-
      🎉 no goals
    -/


/-- **Uniqueness of left-invariant measures**:
Given two left-invariant measures which are finite
on compacts and inner regular, they coincide up to a multiplicative constant. -/
@[to_additive isAddLeftInvariant_eq_smul_of_innerRegular]
lemma isMulLeftInvariant_eq_smul_of_innerRegular [LocallyCompactSpace G]
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ']
    [InnerRegular μ] [InnerRegular μ'] :
    μ' = haarScalarFactor μ' μ • μ := by
  /-
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.InnerRegular
    inst✝ : μ'.InnerRegular
    ⊢ Eq μ' (HSMul.hSMul (μ'.haarScalarFactor μ) μ)
  -/
  ext s hs
  /-
    case h
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.InnerRegular
    inst✝ : μ'.InnerRegular
    s : Set G
    hs : MeasurableSet s
    ⊢ Eq (μ' s) ((HSMul.hSMul (μ'.haarScalarFactor μ) μ) s)
  -/
  rw [hs.measure_eq_iSup_isCompact, hs.measure_eq_iSup_isCompact]
  /-
    case h
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.InnerRegular
    inst✝ : μ'.InnerRegular
    s : Set G
    hs : MeasurableSet s
    ⊢ Eq (iSup fun K => iSup fun x => iSup fun x => μ' K) (iSup fun K => iSup fun  …
  -/
  congr! 4 with K _Ks K_comp
  exact measure_isMulLeftInvariant_eq_smul_of_ne_top μ' μ K_comp.measure_lt_top.ne
    K_comp.measure_lt_top.ne


/-- **Uniqueness of left-invariant measures**:
Given two left-invariant measures which are finite
on compacts and regular, they coincide up to a multiplicative constant. -/
@[to_additive isAddLeftInvariant_eq_smul_of_regular]
lemma isMulLeftInvariant_eq_smul_of_regular [LocallyCompactSpace G]
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ']
    [Regular μ] [Regular μ'] :
    μ' = haarScalarFactor μ' μ • μ := by
  have A : ∀ U, IsOpen U → μ' U = (haarScalarFactor μ' μ • μ) U := by
    intro U hU
    rw [hU.measure_eq_iSup_isCompact, hU.measure_eq_iSup_isCompact]
    congr! 4 with K _KU K_comp
    exact measure_isMulLeftInvariant_eq_smul_of_ne_top μ' μ K_comp.measure_lt_top.ne
      K_comp.measure_lt_top.ne
  /-
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.Regular
    inst✝ : μ'.Regular
    A : ∀ (U : Set G), IsOpen U → Eq (μ' U) ((HSMul.hSMul (μ'.haarScalarFactor μ)  …
    ⊢ Eq μ' (HSMul.hSMul (μ'.haarScalarFactor μ) μ)
  -/
  ext s _hs
  /-
    case h
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.Regular
    inst✝ : μ'.Regular
    A : ∀ (U : Set G), IsOpen U → Eq (μ' U) ((HSMul.hSMul (μ'.haarScalarFactor μ)  …
    s : Set G
    _hs : MeasurableSet s
    ⊢ Eq (μ' s) ((HSMul.hSMul (μ'.haarScalarFactor μ) μ) s)
  -/
  rw [s.measure_eq_iInf_isOpen, s.measure_eq_iInf_isOpen]
  /-
    case h
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.Regular
    inst✝ : μ'.Regular
    A : ∀ (U : Set G), IsOpen U → Eq (μ' U) ((HSMul.hSMul (μ'.haarScalarFactor μ)  …
    s : Set G
    _hs : MeasurableSet s
    ⊢ Eq (iInf fun U => iInf fun x => iInf fun x => μ' U) (iInf fun U => iInf fun  …
  -/
  congr! 4 with U _sU U_open
  /-
    case h.h.e'_4.h.f.f
    G : Type u_1
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : Group G
    inst✝⁸ : TopologicalGroup G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : BorelSpace G
    inst✝⁵ : LocallyCompactSpace G
    μ' μ : MeasureTheory.Measure G
    inst✝⁴ : μ.IsHaarMeasure
    inst✝³ : MeasureTheory.IsFiniteMeasureOnCompacts μ'
    inst✝² : μ'.IsMulLeftInvariant
    inst✝¹ : μ.Regular
    inst✝ : μ'.Regular
    A : ∀ (U : Set G), IsOpen U → Eq (μ' U) ((HSMul.hSMul (μ'.haarScalarFactor μ)  …
    s : Set G
    _hs : MeasurableSet s
    U : Set G
    _sU : HasSubset.Subset s U
    U_open : IsOpen U
    ⊢ Eq (μ' U) ((HSMul.hSMul (μ'.haarScalarFactor μ) μ) U)
  -/
  exact A U U_open
  /-
    🎉 no goals
  -/


/-- **Uniqueness of left-invariant measures**:
Two Haar measures coincide up to a multiplicative constant in a second countable group. -/
@[to_additive isAddLeftInvariant_eq_smul]
lemma isMulLeftInvariant_eq_smul [LocallyCompactSpace G] [SecondCountableTopology G]
    (μ' μ : Measure G) [IsHaarMeasure μ] [IsFiniteMeasureOnCompacts μ'] [IsMulLeftInvariant μ'] :
    μ' = haarScalarFactor μ' μ • μ :=
  isMulLeftInvariant_eq_smul_of_regular μ' μ
  -- one could use as well `isMulLeftInvariant_eq_smul_of_innerRegular`, as in a
  -- second countable topological space all Haar measures are regular and inner regular


@[deprecated (since := "2024-02-12")] alias isHaarMeasure_eq_smul := isMulLeftInvariant_eq_smul

@[deprecated (since := "2024-02-12")] alias isAddHaarMeasure_eq_smul := isAddLeftInvariant_eq_smul


/-- An invariant σ-finite measure is absolutely continuous with respect to a Haar measure in a
second countable group. -/
@[to_additive
"An invariant measure is absolutely continuous with respect to an additive Haar measure. "]
theorem absolutelyContinuous_isHaarMeasure [LocallyCompactSpace G]
    [SecondCountableTopology G] (μ ν : Measure G)
    [SigmaFinite μ] [IsMulLeftInvariant μ] [IsHaarMeasure ν] : μ ≪ ν := by
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    inst✝³ : SecondCountableTopology G
    μ ν : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsHaarMeasure
    ⊢ μ.AbsolutelyContinuous ν
  -/
  have K : PositiveCompacts G := Classical.arbitrary _
  have h : haarMeasure K = (haarScalarFactor (haarMeasure K) ν : ℝ≥0∞) • ν :=
    isMulLeftInvariant_eq_smul (haarMeasure K) ν
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    inst✝³ : SecondCountableTopology G
    μ ν : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsHaarMeasure
    K : TopologicalSpace.PositiveCompacts G
    h : Eq (MeasureTheory.Measure.haarMeasure K) (HSMul.hSMul (↑((MeasureTheory.Me …
    ⊢ μ.AbsolutelyContinuous ν
  -/
  rw [haarMeasure_unique μ K, h, smul_smul]
  /-
    G : Type u_1
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : LocallyCompactSpace G
    inst✝³ : SecondCountableTopology G
    μ ν : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsHaarMeasure
    K : TopologicalSpace.PositiveCompacts G
    h : Eq (MeasureTheory.Measure.haarMeasure K) (HSMul.hSMul (↑((MeasureTheory.Me …
    ⊢ (HSMul.hSMul (HMul.hMul (μ ↑K) ↑((MeasureTheory.Measure.haarMeasure K).haarS …
  -/
  exact smul_absolutelyContinuous
  /-
    🎉 no goals
  -/


/-- A continuous surjective monoid homomorphism of topological groups with compact codomain
is measure preserving, provided that the Haar measures on the domain and on the codomain
have the same total mass.
-/
@[to_additive
  "A continuous surjective additive monoid homomorphism of topological groups with compact codomain
is measure preserving, provided that the Haar measures on the domain and on the codomain
have the same total mass."]
theorem _root_.MonoidHom.measurePreserving
    {H : Type*} [Group H] [TopologicalSpace H] [TopologicalGroup H] [CompactSpace H]
    [MeasurableSpace H] [BorelSpace H]
    {μ : Measure G} [IsHaarMeasure μ] {ν : Measure H} [IsHaarMeasure ν]
    {f : G →* H} (hcont : Continuous f) (hsurj : Surjective f) (huniv : μ univ = ν univ) :
    MeasurePreserving f μ ν where
  measurable := hcont.measurable
  map_eq := by
    /-
      G : Type u_1
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : Group G
      inst✝¹⁰ : TopologicalGroup G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : BorelSpace G
      H : Type u_2
      inst✝⁷ : Group H
      inst✝⁶ : TopologicalSpace H
      inst✝⁵ : TopologicalGroup H
      inst✝⁴ : CompactSpace H
      inst✝³ : MeasurableSpace H
      inst✝² : BorelSpace H
      μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      ν : MeasureTheory.Measure H
      inst✝ : ν.IsHaarMeasure
      f : MonoidHom G H
      hcont : Continuous ⇑f
      hsurj : Function.Surjective ⇑f
      huniv : Eq (μ Set.univ) (ν Set.univ)
      ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) ν
    -/
    have : IsFiniteMeasure μ := ⟨by rw [huniv]; apply measure_lt_top⟩
    /-
      G : Type u_1
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : Group G
      inst✝¹⁰ : TopologicalGroup G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : BorelSpace G
      H : Type u_2
      inst✝⁷ : Group H
      inst✝⁶ : TopologicalSpace H
      inst✝⁵ : TopologicalGroup H
      inst✝⁴ : CompactSpace H
      inst✝³ : MeasurableSpace H
      inst✝² : BorelSpace H
      μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      ν : MeasureTheory.Measure H
      inst✝ : ν.IsHaarMeasure
      f : MonoidHom G H
      hcont : Continuous ⇑f
      hsurj : Function.Surjective ⇑f
      huniv : Eq (μ Set.univ) (ν Set.univ)
      this : MeasureTheory.IsFiniteMeasure μ
      ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) ν
    -/
    have : (μ.map f).IsHaarMeasure := isHaarMeasure_map_of_isFiniteMeasure μ f hcont hsurj
    /-
      G : Type u_1
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : Group G
      inst✝¹⁰ : TopologicalGroup G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : BorelSpace G
      H : Type u_2
      inst✝⁷ : Group H
      inst✝⁶ : TopologicalSpace H
      inst✝⁵ : TopologicalGroup H
      inst✝⁴ : CompactSpace H
      inst✝³ : MeasurableSpace H
      inst✝² : BorelSpace H
      μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      ν : MeasureTheory.Measure H
      inst✝ : ν.IsHaarMeasure
      f : MonoidHom G H
      hcont : Continuous ⇑f
      hsurj : Function.Surjective ⇑f
      huniv : Eq (μ Set.univ) (ν Set.univ)
      this✝ : MeasureTheory.IsFiniteMeasure μ
      this : (MeasureTheory.Measure.map (⇑f) μ).IsHaarMeasure
      ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) ν
    -/
    set C : ℝ≥0 := haarScalarFactor (μ.map f) ν
    /-
      G : Type u_1
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : Group G
      inst✝¹⁰ : TopologicalGroup G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : BorelSpace G
      H : Type u_2
      inst✝⁷ : Group H
      inst✝⁶ : TopologicalSpace H
      inst✝⁵ : TopologicalGroup H
      inst✝⁴ : CompactSpace H
      inst✝³ : MeasurableSpace H
      inst✝² : BorelSpace H
      μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      ν : MeasureTheory.Measure H
      inst✝ : ν.IsHaarMeasure
      f : MonoidHom G H
      hcont : Continuous ⇑f
      hsurj : Function.Surjective ⇑f
      huniv : Eq (μ Set.univ) (ν Set.univ)
      this✝ : MeasureTheory.IsFiniteMeasure μ
      this : (MeasureTheory.Measure.map (⇑f) μ).IsHaarMeasure
      C : NNReal := (MeasureTheory.Measure.map (⇑f) μ).haarScalarFactor ν
      ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) ν
    -/
    have hC : μ.map f = C • ν := isMulLeftInvariant_eq_smul_of_innerRegular _ _
    /-
      G : Type u_1
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : Group G
      inst✝¹⁰ : TopologicalGroup G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : BorelSpace G
      H : Type u_2
      inst✝⁷ : Group H
      inst✝⁶ : TopologicalSpace H
      inst✝⁵ : TopologicalGroup H
      inst✝⁴ : CompactSpace H
      inst✝³ : MeasurableSpace H
      inst✝² : BorelSpace H
      μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      ν : MeasureTheory.Measure H
      inst✝ : ν.IsHaarMeasure
      f : MonoidHom G H
      hcont : Continuous ⇑f
      hsurj : Function.Surjective ⇑f
      huniv : Eq (μ Set.univ) (ν Set.univ)
      this✝ : MeasureTheory.IsFiniteMeasure μ
      this : (MeasureTheory.Measure.map (⇑f) μ).IsHaarMeasure
      C : NNReal := (MeasureTheory.Measure.map (⇑f) μ).haarScalarFactor ν
      hC : Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul C ν)
      ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) ν
    -/
    suffices C = 1 by rwa [this, one_smul] at hC
    have : C * ν univ = 1 * ν univ := by
      rw [one_mul, ← smul_eq_mul, ← ENNReal.smul_def, ← smul_apply, ← hC,
        map_apply hcont.measurable .univ, preimage_univ, huniv]
    /-
      G : Type u_1
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : Group G
      inst✝¹⁰ : TopologicalGroup G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : BorelSpace G
      H : Type u_2
      inst✝⁷ : Group H
      inst✝⁶ : TopologicalSpace H
      inst✝⁵ : TopologicalGroup H
      inst✝⁴ : CompactSpace H
      inst✝³ : MeasurableSpace H
      inst✝² : BorelSpace H
      μ : MeasureTheory.Measure G
      inst✝¹ : μ.IsHaarMeasure
      ν : MeasureTheory.Measure H
      inst✝ : ν.IsHaarMeasure
      f : MonoidHom G H
      hcont : Continuous ⇑f
      hsurj : Function.Surjective ⇑f
      huniv : Eq (μ Set.univ) (ν Set.univ)
      this✝¹ : MeasureTheory.IsFiniteMeasure μ
      this✝ : (MeasureTheory.Measure.map (⇑f) μ).IsHaarMeasure
      C : NNReal := (MeasureTheory.Measure.map (⇑f) μ).haarScalarFactor ν
      hC : Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul C ν)
      this : Eq (HMul.hMul (↑C) (ν Set.univ)) (HMul.hMul 1 (ν Set.univ))
      ⊢ Eq C 1
    -/
    rwa [ENNReal.mul_eq_mul_right (NeZero.ne _) (measure_ne_top _ _), ENNReal.coe_eq_one] at this
    /-
      🎉 no goals
    -/


/-- Any regular Haar measure is invariant under inversion in an abelian group. -/
@[to_additive "Any regular additive Haar measure is invariant under negation in an abelian group."]
instance (priority := 100) IsHaarMeasure.isInvInvariant_of_regular
    [LocallyCompactSpace G] [Regular μ] : IsInvInvariant μ := by
  -- the image measure is a Haar measure. By uniqueness up to multiplication, it is of the form
  -- `c μ`. Applying again inversion, one gets the measure `c^2 μ`. But since inversion is an
  -- involution, this is also `μ`. Hence, `c^2 = 1`, which implies `c = 1`.
  /-
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.Regular
    ⊢ μ.IsInvInvariant
  -/
  constructor
  /-
    case inv_eq_self
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.Regular
    ⊢ Eq μ.inv μ
  -/
  let c : ℝ≥0∞ := haarScalarFactor μ.inv μ
  /-
    case inv_eq_self
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.Regular
    c : ENNReal := ↑(μ.inv.haarScalarFactor μ)
    ⊢ Eq μ.inv μ
  -/
  have hc : μ.inv = c • μ := isMulLeftInvariant_eq_smul_of_regular μ.inv μ
  have : map Inv.inv (map Inv.inv μ) = c ^ 2 • μ := by
    rw [← inv_def μ, hc, Measure.map_smul, ← inv_def μ, hc, smul_smul, pow_two]
  have μeq : μ = c ^ 2 • μ := by
    rw [map_map continuous_inv.measurable continuous_inv.measurable] at this
    simpa only [inv_involutive, Involutive.comp_self, Measure.map_id]
  /-
    case inv_eq_self
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.Regular
    c : ENNReal := ↑(μ.inv.haarScalarFactor μ)
    hc : Eq μ.inv (HSMul.hSMul c μ)
    this : Eq (MeasureTheory.Measure.map Inv.inv (MeasureTheory.Measure.map Inv.in …
    μeq : Eq μ (HSMul.hSMul (HPow.hPow c 2) μ)
    ⊢ Eq μ.inv μ
  -/
  have K : PositiveCompacts G := Classical.arbitrary _
  have : c ^ 2 * μ K = 1 ^ 2 * μ K := by
    conv_rhs => rw [μeq]
    simp
  have : c ^ 2 = 1 ^ 2 :=
    (ENNReal.mul_eq_mul_right (measure_pos_of_nonempty_interior _ K.interior_nonempty).ne'
          K.isCompact.measure_lt_top.ne).1 this
  /-
    case inv_eq_self
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.Regular
    c : ENNReal := ↑(μ.inv.haarScalarFactor μ)
    hc : Eq μ.inv (HSMul.hSMul c μ)
    this✝¹ : Eq (MeasureTheory.Measure.map Inv.inv (MeasureTheory.Measure.map Inv. …
    μeq : Eq μ (HSMul.hSMul (HPow.hPow c 2) μ)
    K : TopologicalSpace.PositiveCompacts G
    this✝ : Eq (HMul.hMul (HPow.hPow c 2) (μ ↑K)) (HMul.hMul (HPow.hPow 1 2) (μ ↑K))
    this : Eq (HPow.hPow c 2) (HPow.hPow 1 2)
    ⊢ Eq μ.inv μ
  -/
  have : c = 1 := (ENNReal.pow_right_strictMono two_ne_zero).injective this
  /-
    case inv_eq_self
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.Regular
    c : ENNReal := ↑(μ.inv.haarScalarFactor μ)
    hc : Eq μ.inv (HSMul.hSMul c μ)
    this✝² : Eq (MeasureTheory.Measure.map Inv.inv (MeasureTheory.Measure.map Inv. …
    μeq : Eq μ (HSMul.hSMul (HPow.hPow c 2) μ)
    K : TopologicalSpace.PositiveCompacts G
    this✝¹ : Eq (HMul.hMul (HPow.hPow c 2) (μ ↑K)) (HMul.hMul (HPow.hPow 1 2) (μ ↑ …
    this✝ : Eq (HPow.hPow c 2) (HPow.hPow 1 2)
    this : Eq c 1
    ⊢ Eq μ.inv μ
  -/
  rw [hc, this, one_smul]
  /-
    🎉 no goals
  -/


/-- Any inner regular Haar measure is invariant under inversion in an abelian group. -/
@[to_additive "Any regular additive Haar measure is invariant under negation in an abelian group."]
instance (priority := 100) IsHaarMeasure.isInvInvariant_of_innerRegular
    [LocallyCompactSpace G] [InnerRegular μ] : IsInvInvariant μ := by
  -- the image measure is a Haar measure. By uniqueness up to multiplication, it is of the form
  -- `c μ`. Applying again inversion, one gets the measure `c^2 μ`. But since inversion is an
  -- involution, this is also `μ`. Hence, `c^2 = 1`, which implies `c = 1`.
  /-
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.InnerRegular
    ⊢ μ.IsInvInvariant
  -/
  constructor
  /-
    case inv_eq_self
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.InnerRegular
    ⊢ Eq μ.inv μ
  -/
  let c : ℝ≥0∞ := haarScalarFactor μ.inv μ
  /-
    case inv_eq_self
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.InnerRegular
    c : ENNReal := ↑(μ.inv.haarScalarFactor μ)
    ⊢ Eq μ.inv μ
  -/
  have hc : μ.inv = c • μ := isMulLeftInvariant_eq_smul_of_innerRegular μ.inv μ
  have : map Inv.inv (map Inv.inv μ) = c ^ 2 • μ := by
    rw [← inv_def μ, hc, Measure.map_smul, ← inv_def μ, hc, smul_smul, pow_two]
  have μeq : μ = c ^ 2 • μ := by
    rw [map_map continuous_inv.measurable continuous_inv.measurable] at this
    simpa only [inv_involutive, Involutive.comp_self, Measure.map_id]
  /-
    case inv_eq_self
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.InnerRegular
    c : ENNReal := ↑(μ.inv.haarScalarFactor μ)
    hc : Eq μ.inv (HSMul.hSMul c μ)
    this : Eq (MeasureTheory.Measure.map Inv.inv (MeasureTheory.Measure.map Inv.in …
    μeq : Eq μ (HSMul.hSMul (HPow.hPow c 2) μ)
    ⊢ Eq μ.inv μ
  -/
  have K : PositiveCompacts G := Classical.arbitrary _
  have : c ^ 2 * μ K = 1 ^ 2 * μ K := by
    conv_rhs => rw [μeq]
    simp
  have : c ^ 2 = 1 ^ 2 :=
    (ENNReal.mul_eq_mul_right (measure_pos_of_nonempty_interior _ K.interior_nonempty).ne'
          K.isCompact.measure_lt_top.ne).1 this
  /-
    case inv_eq_self
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.InnerRegular
    c : ENNReal := ↑(μ.inv.haarScalarFactor μ)
    hc : Eq μ.inv (HSMul.hSMul c μ)
    this✝¹ : Eq (MeasureTheory.Measure.map Inv.inv (MeasureTheory.Measure.map Inv. …
    μeq : Eq μ (HSMul.hSMul (HPow.hPow c 2) μ)
    K : TopologicalSpace.PositiveCompacts G
    this✝ : Eq (HMul.hMul (HPow.hPow c 2) (μ ↑K)) (HMul.hMul (HPow.hPow 1 2) (μ ↑K))
    this : Eq (HPow.hPow c 2) (HPow.hPow 1 2)
    ⊢ Eq μ.inv μ
  -/
  have : c = 1 := (ENNReal.pow_right_strictMono two_ne_zero).injective this
  /-
    case inv_eq_self
    G : Type u_1
    inst✝⁷ : CommGroup G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.InnerRegular
    c : ENNReal := ↑(μ.inv.haarScalarFactor μ)
    hc : Eq μ.inv (HSMul.hSMul c μ)
    this✝² : Eq (MeasureTheory.Measure.map Inv.inv (MeasureTheory.Measure.map Inv. …
    μeq : Eq μ (HSMul.hSMul (HPow.hPow c 2) μ)
    K : TopologicalSpace.PositiveCompacts G
    this✝¹ : Eq (HMul.hMul (HPow.hPow c 2) (μ ↑K)) (HMul.hMul (HPow.hPow 1 2) (μ ↑ …
    this✝ : Eq (HPow.hPow c 2) (HPow.hPow 1 2)
    this : Eq c 1
    ⊢ Eq μ.inv μ
  -/
  rw [hc, this, one_smul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem measurePreserving_zpow [CompactSpace G] [RootableBy G ℤ] {n : ℤ} (hn : n ≠ 0) :
    MeasurePreserving (fun g : G => g ^ n) μ μ :=
  (zpowGroupHom n).measurePreserving (μ := μ) (continuous_zpow n)
    (RootableBy.surjective_pow G ℤ hn) rfl


@[to_additive]
theorem MeasurePreserving.zpow [CompactSpace G] [RootableBy G ℤ]
    {n : ℤ} (hn : n ≠ 0) {X : Type*}
    [MeasurableSpace X] {μ' : Measure X} {f : X → G} (hf : MeasurePreserving f μ' μ) :
    MeasurePreserving (fun x => f x ^ n) μ' μ :=
  (measurePreserving_zpow μ hn).comp hf


