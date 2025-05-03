lemma CompletelyRegularSpace.exists_BCNN {X : Type*} [TopologicalSpace X] [CompletelyRegularSpace X]
    {K : Set X} (K_closed : IsClosed K) {x : X} (x_notin_K : x ∉ K) :
    ∃ (f : X →ᵇ ℝ≥0), f x = 1 ∧ (∀ y ∈ K, f y = 0) := by
  obtain ⟨g, g_cont, gx_zero, g_one_on_K⟩ :=
    CompletelyRegularSpace.completely_regular x K K_closed x_notin_K
  have g_bdd : ∀ x y, dist (Real.toNNReal (g x)) (Real.toNNReal (g y)) ≤ 1 := by
    refine fun x y ↦ ((Real.lipschitzWith_toNNReal).dist_le_mul (g x) (g y)).trans ?_
    simpa using Real.dist_le_of_mem_Icc_01 (g x).prop (g y).prop
  set g' := BoundedContinuousFunction.mkOfBound
      ⟨fun x ↦ Real.toNNReal (g x), continuous_real_toNNReal.comp g_cont.subtype_val⟩ 1 g_bdd
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompletelyRegularSpace X
    K : Set X
    K_closed : IsClosed K
    x : X
    x_notin_K : Not (Membership.mem K x)
    g : X → ↑unitInterval
    g_cont : Continuous g
    gx_zero : Eq (g x) 0
    g_one_on_K : Set.EqOn g 1 K
    g_bdd : ∀ (x y : X), LE.le (Dist.dist (↑(g x)).toNNReal (↑(g y)).toNNReal) 1
    g' : BoundedContinuousFunction X NNReal := BoundedContinuousFunction.mkOfBound …
    ⊢ Exists fun f => And (Eq (f x) 1) (∀ (y : X), Membership.mem K y → Eq (f y) 0)
  -/
  set f := 1 - g'
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompletelyRegularSpace X
    K : Set X
    K_closed : IsClosed K
    x : X
    x_notin_K : Not (Membership.mem K x)
    g : X → ↑unitInterval
    g_cont : Continuous g
    gx_zero : Eq (g x) 0
    g_one_on_K : Set.EqOn g 1 K
    g_bdd : ∀ (x y : X), LE.le (Dist.dist (↑(g x)).toNNReal (↑(g y)).toNNReal) 1
    g' : BoundedContinuousFunction X NNReal := BoundedContinuousFunction.mkOfBound …
    f : BoundedContinuousFunction X NNReal := HSub.hSub 1 g'
    ⊢ Exists fun f => And (Eq (f x) 1) (∀ (y : X), Membership.mem K y → Eq (f y) 0)
  -/
  refine ⟨f, by simp [f, g', gx_zero], fun y y_in_K ↦ by simp [f, g', g_one_on_K y_in_K, tsub_self]⟩
  /-
    🎉 no goals
  -/


/-- The Dirac delta mass at a point `x : X` as a `ProbabilityMeasure`. -/
noncomputable def diracProba (x : X) : ProbabilityMeasure X :=
  ⟨Measure.dirac x, Measure.dirac.isProbabilityMeasure⟩


/-- The assignment `x ↦ diracProba x` is injective if all singletons are measurable. -/
lemma injective_diracProba {X : Type*} [MeasurableSpace X] [MeasurableSpace.SeparatesPoints X] :
    Function.Injective (fun (x : X) ↦ diracProba x) := by
  /-
    X : Type u_2
    inst✝¹ : MeasurableSpace X
    inst✝ : MeasurableSpace.SeparatesPoints X
    ⊢ Function.Injective fun x => MeasureTheory.diracProba x
  -/
  intro x y x_eq_y
  /-
    X : Type u_2
    inst✝¹ : MeasurableSpace X
    inst✝ : MeasurableSpace.SeparatesPoints X
    x y : X
    x_eq_y : Eq ((fun x => MeasureTheory.diracProba x) x) ((fun x => MeasureTheory …
    ⊢ Eq x y
  -/
  rw [← dirac_eq_dirac_iff]
  /-
    X : Type u_2
    inst✝¹ : MeasurableSpace X
    inst✝ : MeasurableSpace.SeparatesPoints X
    x y : X
    x_eq_y : Eq ((fun x => MeasureTheory.diracProba x) x) ((fun x => MeasureTheory …
    ⊢ Eq (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y)
  -/
  rwa [Subtype.ext_iff] at x_eq_y
  /-
    🎉 no goals
  -/


@[simp] lemma diracProba_toMeasure_apply' (x : X) {A : Set X} (A_mble : MeasurableSet A) :
    (diracProba x).toMeasure A = A.indicator 1 x := Measure.dirac_apply' x A_mble


@[simp] lemma diracProba_toMeasure_apply_of_mem {x : X} {A : Set X} (x_in_A : x ∈ A) :
    (diracProba x).toMeasure A = 1 := Measure.dirac_apply_of_mem x_in_A


@[simp] lemma diracProba_toMeasure_apply [MeasurableSingletonClass X] (x : X) (A : Set X) :
    (diracProba x).toMeasure A = A.indicator 1 x := Measure.dirac_apply _ _


/-- The assignment `x ↦ diracProba x` is continuous `X → ProbabilityMeasure X`. -/
lemma continuous_diracProba : Continuous (fun (x : X) ↦ diracProba x) := by
  /-
    X : Type u_1
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : OpensMeasurableSpace X
    ⊢ Continuous fun x => MeasureTheory.diracProba x
  -/
  rw [continuous_iff_continuousAt]
  /-
    X : Type u_1
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : OpensMeasurableSpace X
    ⊢ ∀ (x : X), ContinuousAt (fun x => MeasureTheory.diracProba x) x
  -/
  apply fun x ↦ ProbabilityMeasure.tendsto_iff_forall_lintegral_tendsto.mpr fun f ↦ ?_
  have f_mble : Measurable (fun X ↦ (f X : ℝ≥0∞)) :=
    measurable_coe_nnreal_ennreal_iff.mpr f.continuous.measurable
  /-
    X : Type u_1
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : OpensMeasurableSpace X
    x : X
    f : BoundedContinuousFunction X NNReal
    f_mble : Measurable fun X_1 => ↑(f X_1)
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral ↑(MeasureTheory.diracProba  …
  -/
  simp only [diracProba, ProbabilityMeasure.coe_mk, lintegral_dirac' _ f_mble]
  /-
    X : Type u_1
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : OpensMeasurableSpace X
    x : X
    f : BoundedContinuousFunction X NNReal
    f_mble : Measurable fun X_1 => ↑(f X_1)
    ⊢ Filter.Tendsto (fun i => ↑(f i)) (nhds x) (nhds ↑(f x))
  -/
  exact (ENNReal.continuous_coe.comp f.continuous).continuousAt
  /-
    🎉 no goals
  -/


/-- In a T0 topological space equipped with a sigma algebra which contains all open sets,
the assignment `x ↦ diracProba x` is injective. -/
lemma injective_diracProba_of_T0 [T0Space X] :
    Function.Injective (fun (x : X) ↦ diracProba x) := by
  /-
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T0Space X
    ⊢ Function.Injective fun x => MeasureTheory.diracProba x
  -/
  intro x y δx_eq_δy
  /-
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T0Space X
    x y : X
    δx_eq_δy : Eq ((fun x => MeasureTheory.diracProba x) x) ((fun x => MeasureTheo …
    ⊢ Eq x y
  -/
  by_contra x_ne_y
  /-
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T0Space X
    x y : X
    δx_eq_δy : Eq ((fun x => MeasureTheory.diracProba x) x) ((fun x => MeasureTheo …
    x_ne_y : Not (Eq x y)
    ⊢ False
  -/
  exact dirac_ne_dirac x_ne_y <| congr_arg Subtype.val δx_eq_δy
  /-
    🎉 no goals
  -/


lemma not_tendsto_diracProba_of_not_tendsto [CompletelyRegularSpace X] {x : X} (L : Filter X)
    (h : ¬ Tendsto id L (𝓝 x)) :
    ¬ Tendsto diracProba L (𝓝 (diracProba x)) := by
  obtain ⟨U, U_nhd, hU⟩ : ∃ U, U ∈ 𝓝 x ∧ ∃ᶠ x in L, x ∉ U := by
    by_contra! con
    apply h
    intro U U_nhd
    simpa only [not_frequently, not_not] using con U U_nhd
  /-
    case intro.intro
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : CompletelyRegularSpace X
    x : X
    L : Filter X
    h : Not (Filter.Tendsto id L (nhds x))
    U : Set X
    U_nhd : Membership.mem (nhds x) U
    hU : Filter.Frequently (fun x => Not (Membership.mem U x)) L
    ⊢ Not (Filter.Tendsto MeasureTheory.diracProba L (nhds (MeasureTheory.diracPro …
  -/
  have Uint_nhd : interior U ∈ 𝓝 x := by simpa only [interior_mem_nhds] using U_nhd
  obtain ⟨f, fx_eq_one, f_vanishes_outside⟩ :=
    CompletelyRegularSpace.exists_BCNN isOpen_interior.isClosed_compl
      (by simpa only [mem_compl_iff, not_not] using mem_of_mem_nhds Uint_nhd)
  /-
    case intro.intro.intro.intro
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : CompletelyRegularSpace X
    x : X
    L : Filter X
    h : Not (Filter.Tendsto id L (nhds x))
    U : Set X
    U_nhd : Membership.mem (nhds x) U
    hU : Filter.Frequently (fun x => Not (Membership.mem U x)) L
    Uint_nhd : Membership.mem (nhds x) (interior U)
    f : BoundedContinuousFunction X NNReal
    fx_eq_one : Eq (f x) 1
    f_vanishes_outside : ∀ (y : X), Membership.mem (HasCompl.compl (interior U)) y …
    ⊢ Not (Filter.Tendsto MeasureTheory.diracProba L (nhds (MeasureTheory.diracPro …
  -/
  rw [ProbabilityMeasure.tendsto_iff_forall_lintegral_tendsto, not_forall]
  /-
    case intro.intro.intro.intro
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : CompletelyRegularSpace X
    x : X
    L : Filter X
    h : Not (Filter.Tendsto id L (nhds x))
    U : Set X
    U_nhd : Membership.mem (nhds x) U
    hU : Filter.Frequently (fun x => Not (Membership.mem U x)) L
    Uint_nhd : Membership.mem (nhds x) (interior U)
    f : BoundedContinuousFunction X NNReal
    fx_eq_one : Eq (f x) 1
    f_vanishes_outside : ∀ (y : X), Membership.mem (HasCompl.compl (interior U)) y …
    ⊢ Exists fun x_1 => Not (Filter.Tendsto (fun i => MeasureTheory.lintegral ↑(Me …
  -/
  use f
  simp only [diracProba, ProbabilityMeasure.coe_mk, fx_eq_one,
             lintegral_dirac' _ (measurable_coe_nnreal_ennreal_iff.mpr f.continuous.measurable)]
  /-
    case h
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : CompletelyRegularSpace X
    x : X
    L : Filter X
    h : Not (Filter.Tendsto id L (nhds x))
    U : Set X
    U_nhd : Membership.mem (nhds x) U
    hU : Filter.Frequently (fun x => Not (Membership.mem U x)) L
    Uint_nhd : Membership.mem (nhds x) (interior U)
    f : BoundedContinuousFunction X NNReal
    fx_eq_one : Eq (f x) 1
    f_vanishes_outside : ∀ (y : X), Membership.mem (HasCompl.compl (interior U)) y …
    ⊢ Not (Filter.Tendsto (fun i => ↑(f i)) L (nhds ↑1))
  -/
  apply not_tendsto_iff_exists_frequently_nmem.mpr
  refine ⟨Ioi 0, Ioi_mem_nhds (by simp only [ENNReal.coe_one, zero_lt_one]),
          hU.mp (Eventually.of_forall ?_)⟩
  /-
    case h
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : CompletelyRegularSpace X
    x : X
    L : Filter X
    h : Not (Filter.Tendsto id L (nhds x))
    U : Set X
    U_nhd : Membership.mem (nhds x) U
    hU : Filter.Frequently (fun x => Not (Membership.mem U x)) L
    Uint_nhd : Membership.mem (nhds x) (interior U)
    f : BoundedContinuousFunction X NNReal
    fx_eq_one : Eq (f x) 1
    f_vanishes_outside : ∀ (y : X), Membership.mem (HasCompl.compl (interior U)) y …
    ⊢ ∀ (x : X), Not (Membership.mem U x) → Not (Membership.mem (Set.Ioi 0) ↑(f x))
  -/
  intro x x_notin_U
  rw [f_vanishes_outside x
        (compl_subset_compl.mpr (show interior U ⊆ U from interior_subset) x_notin_U)]
  /-
    case h
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : CompletelyRegularSpace X
    x✝ : X
    L : Filter X
    h : Not (Filter.Tendsto id L (nhds x✝))
    U : Set X
    U_nhd : Membership.mem (nhds x✝) U
    hU : Filter.Frequently (fun x => Not (Membership.mem U x)) L
    Uint_nhd : Membership.mem (nhds x✝) (interior U)
    f : BoundedContinuousFunction X NNReal
    fx_eq_one : Eq (f x✝) 1
    f_vanishes_outside : ∀ (y : X), Membership.mem (HasCompl.compl (interior U)) y …
    x : X
    x_notin_U : Not (Membership.mem U x)
    ⊢ Not (Membership.mem (Set.Ioi 0) ↑0)
  -/
  simp only [ENNReal.coe_zero, mem_Ioi, lt_self_iff_false, not_false_eq_true]
  /-
    🎉 no goals
  -/


lemma tendsto_diracProba_iff_tendsto [CompletelyRegularSpace X] {x : X} (L : Filter X) :
    Tendsto diracProba L (𝓝 (diracProba x)) ↔ Tendsto id L (𝓝 x) := by
  /-
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : CompletelyRegularSpace X
    x : X
    L : Filter X
    ⊢ Iff (Filter.Tendsto MeasureTheory.diracProba L (nhds (MeasureTheory.diracPro …
  -/
  constructor
    /-
      case mp
      X : Type u_1
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : CompletelyRegularSpace X
      x : X
      L : Filter X
      ⊢ Filter.Tendsto MeasureTheory.diracProba L (nhds (MeasureTheory.diracProba x) …
    -/
  · contrapose
    /-
      case mp
      X : Type u_1
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : CompletelyRegularSpace X
      x : X
      L : Filter X
      ⊢ Not (Filter.Tendsto id L (nhds x)) → Not (Filter.Tendsto MeasureTheory.dirac …
    -/
    exact not_tendsto_diracProba_of_not_tendsto L
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : CompletelyRegularSpace X
      x : X
      L : Filter X
      ⊢ Filter.Tendsto id L (nhds x) → Filter.Tendsto MeasureTheory.diracProba L (nh …
    -/
  · intro h
    /-
      case mpr
      X : Type u_1
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : CompletelyRegularSpace X
      x : X
      L : Filter X
      h : Filter.Tendsto id L (nhds x)
      ⊢ Filter.Tendsto MeasureTheory.diracProba L (nhds (MeasureTheory.diracProba x))
    -/
    have aux := (@continuous_diracProba X _ _ _).continuousAt (x := x)
    /-
      case mpr
      X : Type u_1
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : CompletelyRegularSpace X
      x : X
      L : Filter X
      h : Filter.Tendsto id L (nhds x)
      aux : ContinuousAt (fun x => MeasureTheory.diracProba x) x
      ⊢ Filter.Tendsto MeasureTheory.diracProba L (nhds (MeasureTheory.diracProba x))
    -/
    simp only [ContinuousAt] at aux
    /-
      case mpr
      X : Type u_1
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : CompletelyRegularSpace X
      x : X
      L : Filter X
      h : Filter.Tendsto id L (nhds x)
      aux : Filter.Tendsto (fun x => MeasureTheory.diracProba x) (nhds x) (nhds (Mea …
      ⊢ Filter.Tendsto MeasureTheory.diracProba L (nhds (MeasureTheory.diracProba x))
    -/
    exact aux.comp h
    /-
      🎉 no goals
    -/


/-- An inverse function to `diracProba` (only really an inverse under hypotheses that
guarantee injectivity of `diracProba`). -/
noncomputable def diracProbaInverse : range (diracProba (X := X)) → X :=
  fun μ' ↦ (mem_range.mp μ'.prop).choose

-- We redeclare `X` here to temporarily avoid the `[TopologicalSpace X]` instance.

@[simp] lemma diracProba_diracProbaInverse {X : Type*} [MeasurableSpace X]
    (μ : range (diracProba (X := X))) :
    diracProba (diracProbaInverse μ) = μ := (mem_range.mp μ.prop).choose_spec


lemma diracProbaInverse_eq [T0Space X] {x : X} {μ : range (diracProba (X := X))}
    (h : μ = diracProba x) :
    diracProbaInverse μ = x := by
  /-
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T0Space X
    x : X
    μ : ↑(Set.range MeasureTheory.diracProba)
    h : Eq (↑μ) (MeasureTheory.diracProba x)
    ⊢ Eq (MeasureTheory.diracProbaInverse μ) x
  -/
  apply injective_diracProba_of_T0 (X := X)
  /-
    case a
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T0Space X
    x : X
    μ : ↑(Set.range MeasureTheory.diracProba)
    h : Eq (↑μ) (MeasureTheory.diracProba x)
    ⊢ Eq ((fun x => MeasureTheory.diracProba x) (MeasureTheory.diracProbaInverse μ …
  -/
  simp only [← h]
  /-
    case a
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T0Space X
    x : X
    μ : ↑(Set.range MeasureTheory.diracProba)
    h : Eq (↑μ) (MeasureTheory.diracProba x)
    ⊢ Eq (MeasureTheory.diracProba (MeasureTheory.diracProbaInverse μ)) ↑μ
  -/
  exact (mem_range.mp μ.prop).choose_spec
  /-
    🎉 no goals
  -/


/-- In a T0 topological space `X`, the assignment `x ↦ diracProba x` is a bijection to its
range in `ProbabilityMeasure X`. -/
noncomputable def diracProbaEquiv [T0Space X] : X ≃ range (diracProba (X := X)) where
                                     /-
                                       X : Type u_1
                                       inst✝³ : MeasurableSpace X
                                       inst✝² : TopologicalSpace X
                                       inst✝¹ : OpensMeasurableSpace X
                                       inst✝ : T0Space X
                                       x : X
                                       ⊢ Membership.mem (Set.range MeasureTheory.diracProba) (MeasureTheory.diracProb …
                                     -/
  toFun := fun x ↦ ⟨diracProba x, by exact mem_range_self x⟩
                                     /-
                                       🎉 no goals
                                     -/
  invFun := diracProbaInverse
                   /-
                     X : Type u_1
                     inst✝³ : MeasurableSpace X
                     inst✝² : TopologicalSpace X
                     inst✝¹ : OpensMeasurableSpace X
                     inst✝ : T0Space X
                     x : X
                     ⊢ Eq (MeasureTheory.diracProbaInverse ((fun x => ⟨MeasureTheory.diracProba x,  …
                   -/
  left_inv x := by apply diracProbaInverse_eq; rfl
                                               /-
                                                 🎉 no goals
                                               -/
                                 /-
                                   X : Type u_1
                                   inst✝³ : MeasurableSpace X
                                   inst✝² : TopologicalSpace X
                                   inst✝¹ : OpensMeasurableSpace X
                                   inst✝ : T0Space X
                                   μ : ↑(Set.range MeasureTheory.diracProba)
                                   ⊢ Eq ↑((fun x => ⟨MeasureTheory.diracProba x, ⋯⟩) (MeasureTheory.diracProbaInv …
                                 -/
  right_inv μ := Subtype.ext (by simp only [diracProba_diracProbaInverse])
                                 /-
                                   🎉 no goals
                                 -/


/-- The composition of `diracProbaEquiv.symm` and `diracProba` is the subtype inclusion. -/
lemma diracProba_comp_diracProbaEquiv_symm_eq_val [T0Space X] :
    diracProba ∘ (diracProbaEquiv (X := X)).symm = fun μ ↦ μ.val := by
  /-
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T0Space X
    ⊢ Eq (Function.comp MeasureTheory.diracProba ⇑MeasureTheory.diracProbaEquiv.sy …
  -/
  funext μ; simp [diracProbaEquiv]
            /-
              🎉 no goals
            -/


lemma tendsto_diracProbaEquivSymm_iff_tendsto [T0Space X] [CompletelyRegularSpace X]
    {μ : range (diracProba (X := X))} (F : Filter (range (diracProba (X := X)))) :
    Tendsto diracProbaEquiv.symm F (𝓝 (diracProbaEquiv.symm μ)) ↔ Tendsto id F (𝓝 μ) := by
  have key :=
    tendsto_diracProba_iff_tendsto (F.map diracProbaEquiv.symm) (x := diracProbaEquiv.symm μ)
  /-
    X : Type u_1
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : T0Space X
    inst✝ : CompletelyRegularSpace X
    μ : ↑(Set.range MeasureTheory.diracProba)
    F : Filter ↑(Set.range MeasureTheory.diracProba)
    key : Iff (Filter.Tendsto MeasureTheory.diracProba (Filter.map (⇑MeasureTheory …
    ⊢ Iff (Filter.Tendsto (⇑MeasureTheory.diracProbaEquiv.symm) F (nhds (MeasureTh …
  -/
  rw [← (diracProbaEquiv (X := X)).symm_comp_self, ← tendsto_map'_iff] at key
  /-
    X : Type u_1
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : T0Space X
    inst✝ : CompletelyRegularSpace X
    μ : ↑(Set.range MeasureTheory.diracProba)
    F : Filter ↑(Set.range MeasureTheory.diracProba)
    key : Iff (Filter.Tendsto MeasureTheory.diracProba (Filter.map (⇑MeasureTheory …
    ⊢ Iff (Filter.Tendsto (⇑MeasureTheory.diracProbaEquiv.symm) F (nhds (MeasureTh …
  -/
  simp only [tendsto_map'_iff, map_map, Equiv.self_comp_symm, map_id] at key
  /-
    X : Type u_1
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : T0Space X
    inst✝ : CompletelyRegularSpace X
    μ : ↑(Set.range MeasureTheory.diracProba)
    F : Filter ↑(Set.range MeasureTheory.diracProba)
    key : Iff (Filter.Tendsto (Function.comp MeasureTheory.diracProba ⇑MeasureTheo …
    ⊢ Iff (Filter.Tendsto (⇑MeasureTheory.diracProbaEquiv.symm) F (nhds (MeasureTh …
  -/
  simp only [← key, diracProba_comp_diracProbaEquiv_symm_eq_val]
  /-
    X : Type u_1
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : T0Space X
    inst✝ : CompletelyRegularSpace X
    μ : ↑(Set.range MeasureTheory.diracProba)
    F : Filter ↑(Set.range MeasureTheory.diracProba)
    key : Iff (Filter.Tendsto (Function.comp MeasureTheory.diracProba ⇑MeasureTheo …
    ⊢ Iff (Filter.Tendsto (fun μ => ↑μ) F (nhds (MeasureTheory.diracProba (Measure …
  -/
  convert tendsto_subtype_rng.symm
  /-
    case h.e'_1.h.e'_5.h.e'_3
    X : Type u_1
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : T0Space X
    inst✝ : CompletelyRegularSpace X
    μ : ↑(Set.range MeasureTheory.diracProba)
    F : Filter ↑(Set.range MeasureTheory.diracProba)
    key : Iff (Filter.Tendsto (Function.comp MeasureTheory.diracProba ⇑MeasureTheo …
    ⊢ Eq (MeasureTheory.diracProba (MeasureTheory.diracProbaEquiv.symm μ)) ↑μ
  -/
  exact apply_rangeSplitting (fun x ↦ diracProba x) μ
  /-
    🎉 no goals
  -/


/-- In a T0 topological space, `diracProbaEquiv` is continuous. -/
lemma continuous_diracProbaEquiv [T0Space X] :
    Continuous (diracProbaEquiv (X := X)) :=
  Continuous.subtype_mk continuous_diracProba mem_range_self


/-- In a completely regular T0 topological space, the inverse of `diracProbaEquiv` is continuous. -/
lemma continuous_diracProbaEquivSymm [T0Space X] [CompletelyRegularSpace X] :
    Continuous (diracProbaEquiv (X := X)).symm := by
  /-
    X : Type u_1
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : T0Space X
    inst✝ : CompletelyRegularSpace X
    ⊢ Continuous ⇑MeasureTheory.diracProbaEquiv.symm
  -/
  apply continuous_iff_continuousAt.mpr
  /-
    X : Type u_1
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : T0Space X
    inst✝ : CompletelyRegularSpace X
    ⊢ ∀ (x : ↑(Set.range MeasureTheory.diracProba)), ContinuousAt (⇑MeasureTheory. …
  -/
  intro μ
  /-
    X : Type u_1
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : T0Space X
    inst✝ : CompletelyRegularSpace X
    μ : ↑(Set.range MeasureTheory.diracProba)
    ⊢ ContinuousAt (⇑MeasureTheory.diracProbaEquiv.symm) μ
  -/
  apply continuousAt_of_tendsto_nhds (y := diracProbaInverse μ)
  /-
    X : Type u_1
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : T0Space X
    inst✝ : CompletelyRegularSpace X
    μ : ↑(Set.range MeasureTheory.diracProba)
    ⊢ Filter.Tendsto (⇑MeasureTheory.diracProbaEquiv.symm) (nhds μ) (nhds (Measure …
  -/
  exact (tendsto_diracProbaEquivSymm_iff_tendsto _).mpr fun _ mem_nhd ↦ mem_nhd
  /-
    🎉 no goals
  -/


/-- In a completely regular T0 topological space `X`, `diracProbaEquiv` is a homeomorphism to
its image in `ProbabilityMeasure X`. -/
noncomputable def diracProbaHomeomorph [T0Space X] [CompletelyRegularSpace X] :
    X ≃ₜ range (diracProba (X := X)) :=
  @Homeomorph.mk X _ _ _ diracProbaEquiv continuous_diracProbaEquiv continuous_diracProbaEquivSymm


/-- If `X` is a completely regular T0 space with its Borel sigma algebra, then the mapping
that takes a point `x : X` to the delta-measure `diracProba x` is an embedding
`X → ProbabilityMeasure X`. -/
theorem isEmbedding_diracProba [T0Space X] [CompletelyRegularSpace X] :
    IsEmbedding (fun (x : X) ↦ diracProba x) :=
  IsEmbedding.subtypeVal.comp diracProbaHomeomorph.isEmbedding


@[deprecated (since := "2024-10-26")]
alias embedding_diracProba := isEmbedding_diracProba


