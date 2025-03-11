/-- If a random variable `f` in `ℝ≥0∞` is independent of an event `T`, then if you restrict the
  random variable to `T`, then `E[f * indicator T c 0]=E[f] * E[indicator T c 0]`. It is useful for
  `lintegral_mul_eq_lintegral_mul_lintegral_of_independent_measurableSpace`. -/
theorem lintegral_mul_indicator_eq_lintegral_mul_lintegral_indicator {Mf mΩ : MeasurableSpace Ω}
    {μ : Measure Ω} (hMf : Mf ≤ mΩ) (c : ℝ≥0∞) {T : Set Ω} (h_meas_T : MeasurableSet T)
    (h_ind : IndepSets {s | MeasurableSet[Mf] s} {T} μ) (h_meas_f : Measurable[Mf] f) :
    (∫⁻ ω, f ω * T.indicator (fun _ => c) ω ∂μ) =
      (∫⁻ ω, f ω ∂μ) * ∫⁻ ω, T.indicator (fun _ => c) ω ∂μ := by
  /-
    Ω : Type u_1
    f : Ω → ENNReal
    Mf mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    hMf : LE.le Mf mΩ
    c : ENNReal
    T : Set Ω
    h_meas_T : MeasurableSet T
    h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
    h_meas_f : Measurable f
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) (T.indicator (fun x = …
  -/
  revert f
  have h_mul_indicator : ∀ g, Measurable g → Measurable fun a => g a * T.indicator (fun _ => c) a :=
    fun g h_mg => h_mg.mul (measurable_const.indicator h_meas_T)
  /-
    Ω : Type u_1
    Mf mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    hMf : LE.le Mf mΩ
    c : ENNReal
    T : Set Ω
    h_meas_T : MeasurableSet T
    h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
    h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
    ⊢ ∀ {f : Ω → ENNReal}, Measurable f → Eq (MeasureTheory.lintegral μ fun ω => H …
  -/
  apply @Measurable.ennreal_induction _ Mf
    /-
      case h_ind
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      ⊢ ∀ (c_1 : ENNReal) ⦃s : Set Ω⦄, MeasurableSet s → Eq (MeasureTheory.lintegral …
    -/
  · intro c' s' h_meas_s'
    /-
      case h_ind
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      c' : ENNReal
      s' : Set Ω
      h_meas_s' : MeasurableSet s'
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (s'.indicator (fun x => c') …
    -/
    simp_rw [← inter_indicator_mul]
    rw [lintegral_indicator (MeasurableSet.inter (hMf _ h_meas_s') h_meas_T),
      lintegral_indicator (hMf _ h_meas_s'), lintegral_indicator h_meas_T]
    simp only [measurable_const, lintegral_const, univ_inter, lintegral_const_mul,
      MeasurableSet.univ, Measure.restrict_apply]
    /-
      case h_ind
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      c' : ENNReal
      s' : Set Ω
      h_meas_s' : MeasurableSet s'
      ⊢ Eq (HMul.hMul (HMul.hMul c' c) (μ (Inter.inter s' T))) (HMul.hMul (HMul.hMul …
    -/
    rw [IndepSets_iff] at h_ind
    /-
      case h_ind
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ∀ (t1 t2 : Set Ω), Membership.mem (setOf fun s => MeasurableSet s) t1  …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      c' : ENNReal
      s' : Set Ω
      h_meas_s' : MeasurableSet s'
      ⊢ Eq (HMul.hMul (HMul.hMul c' c) (μ (Inter.inter s' T))) (HMul.hMul (HMul.hMul …
    -/
    rw [mul_mul_mul_comm, h_ind s' T h_meas_s' (Set.mem_singleton _)]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      ⊢ ∀ ⦃f g : Ω → ENNReal⦄, Disjoint (Function.support f) (Function.support g) →  …
    -/
  · intro f' g _ h_meas_f' _ h_ind_f' h_ind_g
    /-
      case h_add
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      f' g : Ω → ENNReal
      a✝¹ : Disjoint (Function.support f') (Function.support g)
      h_meas_f' : Measurable f'
      a✝ : Measurable g
      h_ind_f' : Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f' ω) (T.indicato …
      h_ind_g : Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (g ω) (T.indicator  …
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (HAdd.hAdd f' g ω) (T.indic …
    -/
    have h_measM_f' : Measurable f' := h_meas_f'.mono hMf le_rfl
    /-
      case h_add
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      f' g : Ω → ENNReal
      a✝¹ : Disjoint (Function.support f') (Function.support g)
      h_meas_f' : Measurable f'
      a✝ : Measurable g
      h_ind_f' : Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f' ω) (T.indicato …
      h_ind_g : Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (g ω) (T.indicator  …
      h_measM_f' : Measurable f'
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (HAdd.hAdd f' g ω) (T.indic …
    -/
    simp_rw [Pi.add_apply, right_distrib]
    rw [lintegral_add_left (h_mul_indicator _ h_measM_f'), lintegral_add_left h_measM_f',
      right_distrib, h_ind_f', h_ind_g]
    /-
      case h_iSup
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      ⊢ ∀ ⦃f : Nat → Ω → ENNReal⦄, (∀ (n : Nat), Measurable (f n)) → Monotone f → (∀ …
    -/
  · intro f h_meas_f h_mono_f h_ind_f
    /-
      case h_iSup
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      f : Nat → Ω → ENNReal
      h_meas_f : ∀ (n : Nat), Measurable (f n)
      h_mono_f : Monotone f
      h_ind_f : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f n ω …
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul ((fun x => iSup fun n => f  …
    -/
    have h_measM_f : ∀ n, Measurable (f n) := fun n => (h_meas_f n).mono hMf le_rfl
    /-
      case h_iSup
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      f : Nat → Ω → ENNReal
      h_meas_f : ∀ (n : Nat), Measurable (f n)
      h_mono_f : Monotone f
      h_ind_f : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f n ω …
      h_measM_f : ∀ (n : Nat), Measurable (f n)
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul ((fun x => iSup fun n => f  …
    -/
    simp_rw [ENNReal.iSup_mul]
    /-
      case h_iSup
      Ω : Type u_1
      Mf mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      c : ENNReal
      T : Set Ω
      h_meas_T : MeasurableSet T
      h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
      h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
      f : Nat → Ω → ENNReal
      h_meas_f : ∀ (n : Nat), Measurable (f n)
      h_mono_f : Monotone f
      h_ind_f : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f n ω …
      h_measM_f : ∀ (n : Nat), Measurable (f n)
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => iSup fun i => HMul.hMul (f i ω) (T.in …
    -/
    rw [lintegral_iSup h_measM_f h_mono_f, lintegral_iSup, ENNReal.iSup_mul]
      /-
        case h_iSup
        Ω : Type u_1
        Mf mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        hMf : LE.le Mf mΩ
        c : ENNReal
        T : Set Ω
        h_meas_T : MeasurableSet T
        h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
        h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
        f : Nat → Ω → ENNReal
        h_meas_f : ∀ (n : Nat), Measurable (f n)
        h_mono_f : Monotone f
        h_ind_f : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f n ω …
        h_measM_f : ∀ (n : Nat), Measurable (f n)
        ⊢ Eq (iSup fun n => MeasureTheory.lintegral μ fun a => HMul.hMul (f n a) (T.in …
      -/
    · simp_rw [← h_ind_f]
      /-
        🎉 no goals
      -/
      /-
        case h_iSup.hf
        Ω : Type u_1
        Mf mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        hMf : LE.le Mf mΩ
        c : ENNReal
        T : Set Ω
        h_meas_T : MeasurableSet T
        h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
        h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
        f : Nat → Ω → ENNReal
        h_meas_f : ∀ (n : Nat), Measurable (f n)
        h_mono_f : Monotone f
        h_ind_f : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f n ω …
        h_measM_f : ∀ (n : Nat), Measurable (f n)
        ⊢ ∀ (n : Nat), Measurable fun ω => HMul.hMul (f n ω) (T.indicator (fun x => c) …
      -/
    · exact fun n => h_mul_indicator _ (h_measM_f n)
      /-
        🎉 no goals
      -/
      /-
        case h_iSup.h_mono
        Ω : Type u_1
        Mf mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        hMf : LE.le Mf mΩ
        c : ENNReal
        T : Set Ω
        h_meas_T : MeasurableSet T
        h_ind : ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleto …
        h_mul_indicator : ∀ (g : Ω → ENNReal), Measurable g → Measurable fun a => HMul …
        f : Nat → Ω → ENNReal
        h_meas_f : ∀ (n : Nat), Measurable (f n)
        h_mono_f : Monotone f
        h_ind_f : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f n ω …
        h_measM_f : ∀ (n : Nat), Measurable (f n)
        ⊢ Monotone fun i ω => HMul.hMul (f i ω) (T.indicator (fun x => c) ω)
      -/
    · exact fun m n h_le a => mul_le_mul_right' (h_mono_f h_le a) _
      /-
        🎉 no goals
      -/


/-- If `f` and `g` are independent random variables with values in `ℝ≥0∞`,
   then `E[f * g] = E[f] * E[g]`. However, instead of directly using the independence
   of the random variables, it uses the independence of measurable spaces for the
   domains of `f` and `g`. This is similar to the sigma-algebra approach to
   independence. See `lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun` for
   a more common variant of the product of independent variables. -/
theorem lintegral_mul_eq_lintegral_mul_lintegral_of_independent_measurableSpace
    {Mf Mg mΩ : MeasurableSpace Ω} {μ : Measure Ω} (hMf : Mf ≤ mΩ) (hMg : Mg ≤ mΩ)
    (h_ind : Indep Mf Mg μ) (h_meas_f : Measurable[Mf] f) (h_meas_g : Measurable[Mg] g) :
    ∫⁻ ω, f ω * g ω ∂μ = (∫⁻ ω, f ω ∂μ) * ∫⁻ ω, g ω ∂μ := by
  /-
    Ω : Type u_1
    f g : Ω → ENNReal
    Mf Mg mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    hMf : LE.le Mf mΩ
    hMg : LE.le Mg mΩ
    h_ind : ProbabilityTheory.Indep Mf Mg μ
    h_meas_f : Measurable f
    h_meas_g : Measurable g
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) (g ω)) (HMul.hMul (Me …
  -/
  revert g
  /-
    Ω : Type u_1
    f : Ω → ENNReal
    Mf Mg mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    hMf : LE.le Mf mΩ
    hMg : LE.le Mg mΩ
    h_ind : ProbabilityTheory.Indep Mf Mg μ
    h_meas_f : Measurable f
    ⊢ ∀ {g : Ω → ENNReal}, Measurable g → Eq (MeasureTheory.lintegral μ fun ω => H …
  -/
  have h_measM_f : Measurable f := h_meas_f.mono hMf le_rfl
  /-
    Ω : Type u_1
    f : Ω → ENNReal
    Mf Mg mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    hMf : LE.le Mf mΩ
    hMg : LE.le Mg mΩ
    h_ind : ProbabilityTheory.Indep Mf Mg μ
    h_meas_f : Measurable f
    h_measM_f : Measurable f
    ⊢ ∀ {g : Ω → ENNReal}, Measurable g → Eq (MeasureTheory.lintegral μ fun ω => H …
  -/
  apply @Measurable.ennreal_induction _ Mg
    /-
      case h_ind
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      ⊢ ∀ (c : ENNReal) ⦃s : Set Ω⦄, MeasurableSet s → Eq (MeasureTheory.lintegral μ …
    -/
  · intro c s h_s
    /-
      case h_ind
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      c : ENNReal
      s : Set Ω
      h_s : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) (s.indicator (fun x = …
    -/
    apply lintegral_mul_indicator_eq_lintegral_mul_lintegral_indicator hMf _ (hMg _ h_s) _ h_meas_f
    /-
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      c : ENNReal
      s : Set Ω
      h_s : MeasurableSet s
      ⊢ ProbabilityTheory.IndepSets (setOf fun s => MeasurableSet s) (Singleton.sing …
    -/
    apply indepSets_of_indepSets_of_le_right h_ind
    /-
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      c : ENNReal
      s : Set Ω
      h_s : MeasurableSet s
      ⊢ HasSubset.Subset (Singleton.singleton s) (setOf fun s => MeasurableSet s)
    -/
    rwa [singleton_subset_iff]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      ⊢ ∀ ⦃f_1 g : Ω → ENNReal⦄, Disjoint (Function.support f_1) (Function.support g …
    -/
  · intro f' g _ h_measMg_f' _ h_ind_f' h_ind_g'
    /-
      case h_add
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      f' g : Ω → ENNReal
      a✝¹ : Disjoint (Function.support f') (Function.support g)
      h_measMg_f' : Measurable f'
      a✝ : Measurable g
      h_ind_f' : Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) (f' ω)) (HMu …
      h_ind_g' : Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) (g ω)) (HMul …
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) (HAdd.hAdd f' g ω)) ( …
    -/
    have h_measM_f' : Measurable f' := h_measMg_f'.mono hMg le_rfl
    /-
      case h_add
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      f' g : Ω → ENNReal
      a✝¹ : Disjoint (Function.support f') (Function.support g)
      h_measMg_f' : Measurable f'
      a✝ : Measurable g
      h_ind_f' : Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) (f' ω)) (HMu …
      h_ind_g' : Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) (g ω)) (HMul …
      h_measM_f' : Measurable f'
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) (HAdd.hAdd f' g ω)) ( …
    -/
    simp_rw [Pi.add_apply, left_distrib]
    rw [lintegral_add_left h_measM_f', lintegral_add_left (h_measM_f.mul h_measM_f'), left_distrib,
      h_ind_f', h_ind_g']
    /-
      case h_iSup
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      ⊢ ∀ ⦃f_1 : Nat → Ω → ENNReal⦄, (∀ (n : Nat), Measurable (f_1 n)) → Monotone f_ …
    -/
  · intro f' h_meas_f' h_mono_f' h_ind_f'
    /-
      case h_iSup
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      f' : Nat → Ω → ENNReal
      h_meas_f' : ∀ (n : Nat), Measurable (f' n)
      h_mono_f' : Monotone f'
      h_ind_f' : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) …
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) ((fun x => iSup fun n …
    -/
    have h_measM_f' : ∀ n, Measurable (f' n) := fun n => (h_meas_f' n).mono hMg le_rfl
    /-
      case h_iSup
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      f' : Nat → Ω → ENNReal
      h_meas_f' : ∀ (n : Nat), Measurable (f' n)
      h_mono_f' : Monotone f'
      h_ind_f' : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) …
      h_measM_f' : ∀ (n : Nat), Measurable (f' n)
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) ((fun x => iSup fun n …
    -/
    simp_rw [ENNReal.mul_iSup]
    /-
      case h_iSup
      Ω : Type u_1
      f : Ω → ENNReal
      Mf Mg mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      hMf : LE.le Mf mΩ
      hMg : LE.le Mg mΩ
      h_ind : ProbabilityTheory.Indep Mf Mg μ
      h_meas_f : Measurable f
      h_measM_f : Measurable f
      f' : Nat → Ω → ENNReal
      h_meas_f' : ∀ (n : Nat), Measurable (f' n)
      h_mono_f' : Monotone f'
      h_ind_f' : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) …
      h_measM_f' : ∀ (n : Nat), Measurable (f' n)
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => iSup fun i => HMul.hMul (f ω) (f' i ω …
    -/
    rw [lintegral_iSup, lintegral_iSup h_measM_f' h_mono_f', ENNReal.mul_iSup]
      /-
        case h_iSup
        Ω : Type u_1
        f : Ω → ENNReal
        Mf Mg mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        hMf : LE.le Mf mΩ
        hMg : LE.le Mg mΩ
        h_ind : ProbabilityTheory.Indep Mf Mg μ
        h_meas_f : Measurable f
        h_measM_f : Measurable f
        f' : Nat → Ω → ENNReal
        h_meas_f' : ∀ (n : Nat), Measurable (f' n)
        h_mono_f' : Monotone f'
        h_ind_f' : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) …
        h_measM_f' : ∀ (n : Nat), Measurable (f' n)
        ⊢ Eq (iSup fun n => MeasureTheory.lintegral μ fun a => HMul.hMul (f a) (f' n a …
      -/
    · simp_rw [← h_ind_f']
      /-
        🎉 no goals
      -/
      /-
        case h_iSup.hf
        Ω : Type u_1
        f : Ω → ENNReal
        Mf Mg mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        hMf : LE.le Mf mΩ
        hMg : LE.le Mg mΩ
        h_ind : ProbabilityTheory.Indep Mf Mg μ
        h_meas_f : Measurable f
        h_measM_f : Measurable f
        f' : Nat → Ω → ENNReal
        h_meas_f' : ∀ (n : Nat), Measurable (f' n)
        h_mono_f' : Monotone f'
        h_ind_f' : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) …
        h_measM_f' : ∀ (n : Nat), Measurable (f' n)
        ⊢ ∀ (n : Nat), Measurable fun ω => HMul.hMul (f ω) (f' n ω)
      -/
    · exact fun n => h_measM_f.mul (h_measM_f' n)
      /-
        🎉 no goals
      -/
      /-
        case h_iSup.h_mono
        Ω : Type u_1
        f : Ω → ENNReal
        Mf Mg mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        hMf : LE.le Mf mΩ
        hMg : LE.le Mg mΩ
        h_ind : ProbabilityTheory.Indep Mf Mg μ
        h_meas_f : Measurable f
        h_measM_f : Measurable f
        f' : Nat → Ω → ENNReal
        h_meas_f' : ∀ (n : Nat), Measurable (f' n)
        h_mono_f' : Monotone f'
        h_ind_f' : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul (f ω) …
        h_measM_f' : ∀ (n : Nat), Measurable (f' n)
        ⊢ Monotone fun i ω => HMul.hMul (f ω) (f' i ω)
      -/
    · exact fun n m (h_le : n ≤ m) a => mul_le_mul_left' (h_mono_f' h_le a) _
      /-
        🎉 no goals
      -/


/-- If `f` and `g` are independent random variables with values in `ℝ≥0∞`,
   then `E[f * g] = E[f] * E[g]`. -/
theorem lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun (h_meas_f : Measurable f)
    (h_meas_g : Measurable g) (h_indep_fun : IndepFun f g μ) :
    (∫⁻ ω, (f * g) ω ∂μ) = (∫⁻ ω, f ω ∂μ) * ∫⁻ ω, g ω ∂μ :=
  lintegral_mul_eq_lintegral_mul_lintegral_of_independent_measurableSpace
    (measurable_iff_comap_le.1 h_meas_f) (measurable_iff_comap_le.1 h_meas_g) h_indep_fun
    (Measurable.of_comap_le le_rfl) (Measurable.of_comap_le le_rfl)


/-- If `f` and `g` with values in `ℝ≥0∞` are independent and almost everywhere measurable,
   then `E[f * g] = E[f] * E[g]` (slightly generalizing
   `lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun`). -/
theorem lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun' (h_meas_f : AEMeasurable f μ)
    (h_meas_g : AEMeasurable g μ) (h_indep_fun : IndepFun f g μ) :
    (∫⁻ ω, (f * g) ω ∂μ) = (∫⁻ ω, f ω ∂μ) * ∫⁻ ω, g ω ∂μ := by
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f g : Ω → ENNReal
    h_meas_f : AEMeasurable f μ
    h_meas_g : AEMeasurable g μ
    h_indep_fun : ProbabilityTheory.IndepFun f g μ
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => HMul.hMul f g ω) (HMul.hMul (MeasureT …
  -/
  have fg_ae : f * g =ᵐ[μ] h_meas_f.mk _ * h_meas_g.mk _ := h_meas_f.ae_eq_mk.mul h_meas_g.ae_eq_mk
  rw [lintegral_congr_ae h_meas_f.ae_eq_mk, lintegral_congr_ae h_meas_g.ae_eq_mk,
    lintegral_congr_ae fg_ae]
  apply lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun h_meas_f.measurable_mk
      h_meas_g.measurable_mk
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f g : Ω → ENNReal
    h_meas_f : AEMeasurable f μ
    h_meas_g : AEMeasurable g μ
    h_indep_fun : ProbabilityTheory.IndepFun f g μ
    fg_ae : (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f g) (HMul.hMul (AEMeasur …
    ⊢ ProbabilityTheory.IndepFun (AEMeasurable.mk f h_meas_f) (AEMeasurable.mk g h …
  -/
  exact h_indep_fun.ae_eq h_meas_f.ae_eq_mk h_meas_g.ae_eq_mk
  /-
    🎉 no goals
  -/


theorem lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun'' (h_meas_f : AEMeasurable f μ)
    (h_meas_g : AEMeasurable g μ) (h_indep_fun : IndepFun f g μ) :
    ∫⁻ ω, f ω * g ω ∂μ = (∫⁻ ω, f ω ∂μ) * ∫⁻ ω, g ω ∂μ :=
  lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun' h_meas_f h_meas_g h_indep_fun


/-- The product of two independent, integrable, real-valued random variables is integrable. -/
theorem IndepFun.integrable_mul {β : Type*} [MeasurableSpace β] {X Y : Ω → β}
    [NormedDivisionRing β] [BorelSpace β] (hXY : IndepFun X Y μ) (hX : Integrable X μ)
    (hY : Integrable Y μ) : Integrable (X * Y) μ := by
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    ⊢ MeasureTheory.Integrable (HMul.hMul X Y) μ
  -/
  let nX : Ω → ENNReal := fun a => ‖X a‖₊
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    nX : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (X a))
    ⊢ MeasureTheory.Integrable (HMul.hMul X Y) μ
  -/
  let nY : Ω → ENNReal := fun a => ‖Y a‖₊
  have hXY' : IndepFun (fun a => ‖X a‖₊) (fun a => ‖Y a‖₊) μ :=
    hXY.comp measurable_nnnorm measurable_nnnorm
  have hXY'' : IndepFun nX nY μ :=
    hXY'.comp measurable_coe_nnreal_ennreal measurable_coe_nnreal_ennreal
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    nX : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (X a))
    nY : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (Y a))
    hXY' : ProbabilityTheory.IndepFun (fun a => NNNorm.nnnorm (X a)) (fun a => NNN …
    hXY'' : ProbabilityTheory.IndepFun nX nY μ
    ⊢ MeasureTheory.Integrable (HMul.hMul X Y) μ
  -/
  have hnX : AEMeasurable nX μ := hX.1.aemeasurable.nnnorm.coe_nnreal_ennreal
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    nX : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (X a))
    nY : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (Y a))
    hXY' : ProbabilityTheory.IndepFun (fun a => NNNorm.nnnorm (X a)) (fun a => NNN …
    hXY'' : ProbabilityTheory.IndepFun nX nY μ
    hnX : AEMeasurable nX μ
    ⊢ MeasureTheory.Integrable (HMul.hMul X Y) μ
  -/
  have hnY : AEMeasurable nY μ := hY.1.aemeasurable.nnnorm.coe_nnreal_ennreal
  have hmul : ∫⁻ a, nX a * nY a ∂μ = (∫⁻ a, nX a ∂μ) * ∫⁻ a, nY a ∂μ :=
    lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun' hnX hnY hXY''
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    nX : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (X a))
    nY : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (Y a))
    hXY' : ProbabilityTheory.IndepFun (fun a => NNNorm.nnnorm (X a)) (fun a => NNN …
    hXY'' : ProbabilityTheory.IndepFun nX nY μ
    hnX : AEMeasurable nX μ
    hnY : AEMeasurable nY μ
    hmul : Eq (MeasureTheory.lintegral μ fun a => HMul.hMul (nX a) (nY a)) (HMul.h …
    ⊢ MeasureTheory.Integrable (HMul.hMul X Y) μ
  -/
  refine ⟨hX.1.mul hY.1, ?_⟩
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    nX : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (X a))
    nY : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (Y a))
    hXY' : ProbabilityTheory.IndepFun (fun a => NNNorm.nnnorm (X a)) (fun a => NNN …
    hXY'' : ProbabilityTheory.IndepFun nX nY μ
    hnX : AEMeasurable nX μ
    hnY : AEMeasurable nY μ
    hmul : Eq (MeasureTheory.lintegral μ fun a => HMul.hMul (nX a) (nY a)) (HMul.h …
    ⊢ MeasureTheory.HasFiniteIntegral (HMul.hMul X Y) μ
  -/
  simp only [nX, nY] at hmul
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    nX : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (X a))
    nY : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (Y a))
    hXY' : ProbabilityTheory.IndepFun (fun a => NNNorm.nnnorm (X a)) (fun a => NNN …
    hXY'' : ProbabilityTheory.IndepFun nX nY μ
    hnX : AEMeasurable nX μ
    hnY : AEMeasurable nY μ
    hmul : Eq (MeasureTheory.lintegral μ fun a => HMul.hMul ↑(NNNorm.nnnorm (X a)) …
    ⊢ MeasureTheory.HasFiniteIntegral (HMul.hMul X Y) μ
  -/
  simp_rw [hasFiniteIntegral_iff_nnnorm, Pi.mul_apply, nnnorm_mul, ENNReal.coe_mul, hmul]
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    nX : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (X a))
    nY : Ω → ENNReal := fun a => ↑(NNNorm.nnnorm (Y a))
    hXY' : ProbabilityTheory.IndepFun (fun a => NNNorm.nnnorm (X a)) (fun a => NNN …
    hXY'' : ProbabilityTheory.IndepFun nX nY μ
    hnX : AEMeasurable nX μ
    hnY : AEMeasurable nY μ
    hmul : Eq (MeasureTheory.lintegral μ fun a => HMul.hMul ↑(NNNorm.nnnorm (X a)) …
    ⊢ LT.lt (HMul.hMul (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (X a))) …
  -/
  exact ENNReal.mul_lt_top hX.2 hY.2
  /-
    🎉 no goals
  -/


/-- If the product of two independent real-valued random variables is integrable and
the second one is not almost everywhere zero, then the first one is integrable. -/
theorem IndepFun.integrable_left_of_integrable_mul {β : Type*} [MeasurableSpace β] {X Y : Ω → β}
    [NormedDivisionRing β] [BorelSpace β] (hXY : IndepFun X Y μ) (h'XY : Integrable (X * Y) μ)
    (hX : AEStronglyMeasurable X μ) (hY : AEStronglyMeasurable Y μ) (h'Y : ¬Y =ᵐ[μ] 0) :
    Integrable X μ := by
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
    ⊢ MeasureTheory.Integrable X μ
  -/
  refine ⟨hX, ?_⟩
  have I : (∫⁻ ω, ‖Y ω‖₊ ∂μ) ≠ 0 := fun H ↦ by
    have I : (fun ω => ‖Y ω‖₊ : Ω → ℝ≥0∞) =ᵐ[μ] 0 := (lintegral_eq_zero_iff' hY.ennnorm).1 H
    apply h'Y
    filter_upwards [I] with ω hω
    simpa using hω
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (Y ω))) 0
    ⊢ MeasureTheory.HasFiniteIntegral X μ
  -/
  refine hasFiniteIntegral_iff_nnnorm.mpr <| lt_top_iff_ne_top.2 fun H => ?_
  have J : IndepFun (fun ω => ‖X ω‖₊ : Ω → ℝ≥0∞) (fun ω => ‖Y ω‖₊ : Ω → ℝ≥0∞) μ := by
    have M : Measurable fun x : β => (‖x‖₊ : ℝ≥0∞) := measurable_nnnorm.coe_nnreal_ennreal
    apply IndepFun.comp hXY M M
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (Y ω))) 0
    H : Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (X a))) Top.top
    J : ProbabilityTheory.IndepFun (fun ω => ↑(NNNorm.nnnorm (X ω))) (fun ω => ↑(N …
    ⊢ False
  -/
  have A : (∫⁻ ω, ‖X ω * Y ω‖₊ ∂μ) < ∞ := h'XY.2
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (Y ω))) 0
    H : Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (X a))) Top.top
    J : ProbabilityTheory.IndepFun (fun ω => ↑(NNNorm.nnnorm (X ω))) (fun ω => ↑(N …
    A : LT.lt (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (HMul.hMul (X ω) …
    ⊢ False
  -/
  simp only [nnnorm_mul, ENNReal.coe_mul] at A
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (Y ω))) 0
    H : Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (X a))) Top.top
    J : ProbabilityTheory.IndepFun (fun ω => ↑(NNNorm.nnnorm (X ω))) (fun ω => ↑(N …
    A : LT.lt (MeasureTheory.lintegral μ fun ω => HMul.hMul ↑(NNNorm.nnnorm (X ω)) …
    ⊢ False
  -/
  rw [lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun'' hX.ennnorm hY.ennnorm J, H] at A
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (Y ω))) 0
    H : Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (X a))) Top.top
    J : ProbabilityTheory.IndepFun (fun ω => ↑(NNNorm.nnnorm (X ω))) (fun ω => ↑(N …
    A : LT.lt (HMul.hMul Top.top (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnno …
    ⊢ False
  -/
  simp only [ENNReal.top_mul I, lt_self_iff_false] at A
  /-
    🎉 no goals
  -/


/-- If the product of two independent real-valued random variables is integrable and the
first one is not almost everywhere zero, then the second one is integrable. -/
theorem IndepFun.integrable_right_of_integrable_mul {β : Type*} [MeasurableSpace β] {X Y : Ω → β}
    [NormedDivisionRing β] [BorelSpace β] (hXY : IndepFun X Y μ) (h'XY : Integrable (X * Y) μ)
    (hX : AEStronglyMeasurable X μ) (hY : AEStronglyMeasurable Y μ) (h'X : ¬X =ᵐ[μ] 0) :
    Integrable Y μ := by
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
    ⊢ MeasureTheory.Integrable Y μ
  -/
  refine ⟨hY, ?_⟩
  have I : (∫⁻ ω, ‖X ω‖₊ ∂μ) ≠ 0 := fun H ↦ by
    have I : (fun ω => ‖X ω‖₊ : Ω → ℝ≥0∞) =ᵐ[μ] 0 := (lintegral_eq_zero_iff' hX.ennnorm).1 H
    apply h'X
    filter_upwards [I] with ω hω
    simpa using hω
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (X ω))) 0
    ⊢ MeasureTheory.HasFiniteIntegral Y μ
  -/
  refine lt_top_iff_ne_top.2 fun H => ?_
  have J : IndepFun (fun ω => ‖X ω‖₊ : Ω → ℝ≥0∞) (fun ω => ‖Y ω‖₊ : Ω → ℝ≥0∞) μ := by
    have M : Measurable fun x : β => (‖x‖₊ : ℝ≥0∞) := measurable_nnnorm.coe_nnreal_ennreal
    apply IndepFun.comp hXY M M
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (X ω))) 0
    H : Eq (MeasureTheory.lintegral μ fun a => ENorm.enorm (Y a)) Top.top
    J : ProbabilityTheory.IndepFun (fun ω => ↑(NNNorm.nnnorm (X ω))) (fun ω => ↑(N …
    ⊢ False
  -/
  have A : (∫⁻ ω, ‖X ω * Y ω‖₊ ∂μ) < ∞ := h'XY.2
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (X ω))) 0
    H : Eq (MeasureTheory.lintegral μ fun a => ENorm.enorm (Y a)) Top.top
    J : ProbabilityTheory.IndepFun (fun ω => ↑(NNNorm.nnnorm (X ω))) (fun ω => ↑(N …
    A : LT.lt (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (HMul.hMul (X ω) …
    ⊢ False
  -/
  simp only [nnnorm_mul, ENNReal.coe_mul] at A
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (X ω))) 0
    H : Eq (MeasureTheory.lintegral μ fun a => ENorm.enorm (Y a)) Top.top
    J : ProbabilityTheory.IndepFun (fun ω => ↑(NNNorm.nnnorm (X ω))) (fun ω => ↑(N …
    A : LT.lt (MeasureTheory.lintegral μ fun ω => HMul.hMul ↑(NNNorm.nnnorm (X ω)) …
    ⊢ False
  -/
  simp_rw [enorm_eq_nnnorm] at H
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (X ω))) 0
    H : Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (Y a))) Top.top
    J : ProbabilityTheory.IndepFun (fun ω => ↑(NNNorm.nnnorm (X ω))) (fun ω => ↑(N …
    A : LT.lt (MeasureTheory.lintegral μ fun ω => HMul.hMul ↑(NNNorm.nnnorm (X ω)) …
    ⊢ False
  -/
  rw [lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun'' hX.ennnorm hY.ennnorm J, H] at A
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_2
    inst✝² : MeasurableSpace β
    X Y : Ω → β
    inst✝¹ : NormedDivisionRing β
    inst✝ : BorelSpace β
    hXY : ProbabilityTheory.IndepFun X Y μ
    h'XY : MeasureTheory.Integrable (HMul.hMul X Y) μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
    I : Ne (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (X ω))) 0
    H : Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (Y a))) Top.top
    J : ProbabilityTheory.IndepFun (fun ω => ↑(NNNorm.nnnorm (X ω))) (fun ω => ↑(N …
    A : LT.lt (HMul.hMul (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm (X ω) …
    ⊢ False
  -/
  simp only [ENNReal.mul_top I, lt_self_iff_false] at A
  /-
    🎉 no goals
  -/


/-- The (Bochner) integral of the product of two independent, nonnegative random
  variables is the product of their integrals. The proof is just plumbing around
  `lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun'`. -/
theorem IndepFun.integral_mul_of_nonneg (hXY : IndepFun X Y μ) (hXp : 0 ≤ X) (hYp : 0 ≤ Y)
    (hXm : AEMeasurable X μ) (hYm : AEMeasurable Y μ) :
    integral μ (X * Y) = integral μ X * integral μ Y := by
  have h1 : AEMeasurable (fun a => ENNReal.ofReal (X a)) μ :=
    ENNReal.measurable_ofReal.comp_aemeasurable hXm
  have h2 : AEMeasurable (fun a => ENNReal.ofReal (Y a)) μ :=
    ENNReal.measurable_ofReal.comp_aemeasurable hYm
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hXp : LE.le 0 X
    hYp : LE.le 0 Y
    hXm : AEMeasurable X μ
    hYm : AEMeasurable Y μ
    h1 : AEMeasurable (fun a => ENNReal.ofReal (X a)) μ
    h2 : AEMeasurable (fun a => ENNReal.ofReal (Y a)) μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have h3 : AEMeasurable (X * Y) μ := hXm.mul hYm
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hXp : LE.le 0 X
    hYp : LE.le 0 Y
    hXm : AEMeasurable X μ
    hYm : AEMeasurable Y μ
    h1 : AEMeasurable (fun a => ENNReal.ofReal (X a)) μ
    h2 : AEMeasurable (fun a => ENNReal.ofReal (Y a)) μ
    h3 : AEMeasurable (HMul.hMul X Y) μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have h4 : 0 ≤ᵐ[μ] X * Y := ae_of_all _ fun ω => mul_nonneg (hXp ω) (hYp ω)
  rw [integral_eq_lintegral_of_nonneg_ae (ae_of_all _ hXp) hXm.aestronglyMeasurable,
    integral_eq_lintegral_of_nonneg_ae (ae_of_all _ hYp) hYm.aestronglyMeasurable,
    integral_eq_lintegral_of_nonneg_ae h4 h3.aestronglyMeasurable]
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hXp : LE.le 0 X
    hYp : LE.le 0 Y
    hXm : AEMeasurable X μ
    hYm : AEMeasurable Y μ
    h1 : AEMeasurable (fun a => ENNReal.ofReal (X a)) μ
    h2 : AEMeasurable (fun a => ENNReal.ofReal (Y a)) μ
    h3 : AEMeasurable (HMul.hMul X Y) μ
    h4 : (MeasureTheory.ae μ).EventuallyLE 0 (HMul.hMul X Y)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul X Y a)).toR …
  -/
  simp_rw [← ENNReal.toReal_mul, Pi.mul_apply, ENNReal.ofReal_mul (hXp _)]
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hXp : LE.le 0 X
    hYp : LE.le 0 Y
    hXm : AEMeasurable X μ
    hYm : AEMeasurable Y μ
    h1 : AEMeasurable (fun a => ENNReal.ofReal (X a)) μ
    h2 : AEMeasurable (fun a => ENNReal.ofReal (Y a)) μ
    h3 : AEMeasurable (HMul.hMul X Y) μ
    h4 : (MeasureTheory.ae μ).EventuallyLE 0 (HMul.hMul X Y)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul (ENNReal.ofReal (X a)) (ENN …
  -/
  congr
  /-
    case e_a
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hXp : LE.le 0 X
    hYp : LE.le 0 Y
    hXm : AEMeasurable X μ
    hYm : AEMeasurable Y μ
    h1 : AEMeasurable (fun a => ENNReal.ofReal (X a)) μ
    h2 : AEMeasurable (fun a => ENNReal.ofReal (Y a)) μ
    h3 : AEMeasurable (HMul.hMul X Y) μ
    h4 : (MeasureTheory.ae μ).EventuallyLE 0 (HMul.hMul X Y)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul (ENNReal.ofReal (X a)) (ENN …
  -/
  apply lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun' h1 h2
  /-
    case e_a
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hXp : LE.le 0 X
    hYp : LE.le 0 Y
    hXm : AEMeasurable X μ
    hYm : AEMeasurable Y μ
    h1 : AEMeasurable (fun a => ENNReal.ofReal (X a)) μ
    h2 : AEMeasurable (fun a => ENNReal.ofReal (Y a)) μ
    h3 : AEMeasurable (HMul.hMul X Y) μ
    h4 : (MeasureTheory.ae μ).EventuallyLE 0 (HMul.hMul X Y)
    ⊢ ProbabilityTheory.IndepFun (fun a => ENNReal.ofReal (X a)) (fun a => ENNReal …
  -/
  exact hXY.comp ENNReal.measurable_ofReal ENNReal.measurable_ofReal
  /-
    🎉 no goals
  -/


/-- The (Bochner) integral of the product of two independent, integrable random
  variables is the product of their integrals. The proof is pedestrian decomposition
  into their positive and negative parts in order to apply `IndepFun.integral_mul_of_nonneg`
  four times. -/
theorem IndepFun.integral_mul_of_integrable (hXY : IndepFun X Y μ) (hX : Integrable X μ)
    (hY : Integrable Y μ) : integral μ (X * Y) = integral μ X * integral μ Y := by
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  let pos : ℝ → ℝ := fun x => max x 0
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  let neg : ℝ → ℝ := fun x => max (-x) 0
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have posm : Measurable pos := measurable_id'.max measurable_const
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have negm : Measurable neg := measurable_id'.neg.max measurable_const
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  let Xp := pos ∘ X
  -- `X⁺` would look better but it makes `simp_rw` below fail
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  let Xm := neg ∘ X
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  let Yp := pos ∘ Y
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  let Ym := neg ∘ Y
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hXpm : X = Xp - Xm := funext fun ω => (max_zero_sub_max_neg_zero_eq_self (X ω)).symm
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hYpm : Y = Yp - Ym := funext fun ω => (max_zero_sub_max_neg_zero_eq_self (Y ω)).symm
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hp1 : 0 ≤ Xm := fun ω => le_max_right _ _
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hp2 : 0 ≤ Xp := fun ω => le_max_right _ _
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hp3 : 0 ≤ Ym := fun ω => le_max_right _ _
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hp4 : 0 ≤ Yp := fun ω => le_max_right _ _
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hm1 : AEMeasurable Xm μ := hX.1.aemeasurable.neg.max aemeasurable_const
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hm2 : AEMeasurable Xp μ := hX.1.aemeasurable.max aemeasurable_const
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hm3 : AEMeasurable Ym μ := hY.1.aemeasurable.neg.max aemeasurable_const
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hm4 : AEMeasurable Yp μ := hY.1.aemeasurable.max aemeasurable_const
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hv1 : Integrable Xm μ := hX.neg_part
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hv2 : Integrable Xp μ := hX.pos_part
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hv3 : Integrable Ym μ := hY.neg_part
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hv4 : Integrable Yp μ := hY.pos_part
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hi1 : IndepFun Xm Ym μ := hXY.comp negm negm
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hi2 : IndepFun Xp Ym μ := hXY.comp posm negm
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    hi2 : ProbabilityTheory.IndepFun Xp Ym μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hi3 : IndepFun Xm Yp μ := hXY.comp negm posm
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    hi2 : ProbabilityTheory.IndepFun Xp Ym μ
    hi3 : ProbabilityTheory.IndepFun Xm Yp μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hi4 : IndepFun Xp Yp μ := hXY.comp posm posm
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    hi2 : ProbabilityTheory.IndepFun Xp Ym μ
    hi3 : ProbabilityTheory.IndepFun Xm Yp μ
    hi4 : ProbabilityTheory.IndepFun Xp Yp μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hl1 : Integrable (Xm * Ym) μ := hi1.integrable_mul hv1 hv3
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    hi2 : ProbabilityTheory.IndepFun Xp Ym μ
    hi3 : ProbabilityTheory.IndepFun Xm Yp μ
    hi4 : ProbabilityTheory.IndepFun Xp Yp μ
    hl1 : MeasureTheory.Integrable (HMul.hMul Xm Ym) μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hl2 : Integrable (Xp * Ym) μ := hi2.integrable_mul hv2 hv3
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    hi2 : ProbabilityTheory.IndepFun Xp Ym μ
    hi3 : ProbabilityTheory.IndepFun Xm Yp μ
    hi4 : ProbabilityTheory.IndepFun Xp Yp μ
    hl1 : MeasureTheory.Integrable (HMul.hMul Xm Ym) μ
    hl2 : MeasureTheory.Integrable (HMul.hMul Xp Ym) μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hl3 : Integrable (Xm * Yp) μ := hi3.integrable_mul hv1 hv4
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    hi2 : ProbabilityTheory.IndepFun Xp Ym μ
    hi3 : ProbabilityTheory.IndepFun Xm Yp μ
    hi4 : ProbabilityTheory.IndepFun Xp Yp μ
    hl1 : MeasureTheory.Integrable (HMul.hMul Xm Ym) μ
    hl2 : MeasureTheory.Integrable (HMul.hMul Xp Ym) μ
    hl3 : MeasureTheory.Integrable (HMul.hMul Xm Yp) μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hl4 : Integrable (Xp * Yp) μ := hi4.integrable_mul hv2 hv4
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    hi2 : ProbabilityTheory.IndepFun Xp Ym μ
    hi3 : ProbabilityTheory.IndepFun Xm Yp μ
    hi4 : ProbabilityTheory.IndepFun Xp Yp μ
    hl1 : MeasureTheory.Integrable (HMul.hMul Xm Ym) μ
    hl2 : MeasureTheory.Integrable (HMul.hMul Xp Ym) μ
    hl3 : MeasureTheory.Integrable (HMul.hMul Xm Yp) μ
    hl4 : MeasureTheory.Integrable (HMul.hMul Xp Yp) μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hl5 : Integrable (Xp * Yp - Xm * Yp) μ := hl4.sub hl3
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    hi2 : ProbabilityTheory.IndepFun Xp Ym μ
    hi3 : ProbabilityTheory.IndepFun Xm Yp μ
    hi4 : ProbabilityTheory.IndepFun Xp Yp μ
    hl1 : MeasureTheory.Integrable (HMul.hMul Xm Ym) μ
    hl2 : MeasureTheory.Integrable (HMul.hMul Xp Ym) μ
    hl3 : MeasureTheory.Integrable (HMul.hMul Xm Yp) μ
    hl4 : MeasureTheory.Integrable (HMul.hMul Xp Yp) μ
    hl5 : MeasureTheory.Integrable (HSub.hSub (HMul.hMul Xp Yp) (HMul.hMul Xm Yp)) μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  have hl6 : Integrable (Xp * Ym - Xm * Ym) μ := hl2.sub hl1
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    hi2 : ProbabilityTheory.IndepFun Xp Ym μ
    hi3 : ProbabilityTheory.IndepFun Xm Yp μ
    hi4 : ProbabilityTheory.IndepFun Xp Yp μ
    hl1 : MeasureTheory.Integrable (HMul.hMul Xm Ym) μ
    hl2 : MeasureTheory.Integrable (HMul.hMul Xp Ym) μ
    hl3 : MeasureTheory.Integrable (HMul.hMul Xm Yp) μ
    hl4 : MeasureTheory.Integrable (HMul.hMul Xp Yp) μ
    hl5 : MeasureTheory.Integrable (HSub.hSub (HMul.hMul Xp Yp) (HMul.hMul Xm Yp)) μ
    hl6 : MeasureTheory.Integrable (HSub.hSub (HMul.hMul Xp Ym) (HMul.hMul Xm Ym)) μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  rw [hXpm, hYpm, mul_sub, sub_mul, sub_mul]
  rw [integral_sub' hl5 hl6, integral_sub' hl4 hl3, integral_sub' hl2 hl1, integral_sub' hv2 hv1,
    integral_sub' hv4 hv3, hi1.integral_mul_of_nonneg hp1 hp3 hm1 hm3,
    hi2.integral_mul_of_nonneg hp2 hp3 hm2 hm3, hi3.integral_mul_of_nonneg hp1 hp4 hm1 hm4,
    hi4.integral_mul_of_nonneg hp2 hp4 hm2 hm4]
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.Integrable X μ
    hY : MeasureTheory.Integrable Y μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    Xp : Ω → Real := Function.comp pos X
    Xm : Ω → Real := Function.comp neg X
    Yp : Ω → Real := Function.comp pos Y
    Ym : Ω → Real := Function.comp neg Y
    hXpm : Eq X (HSub.hSub Xp Xm)
    hYpm : Eq Y (HSub.hSub Yp Ym)
    hp1 : LE.le 0 Xm
    hp2 : LE.le 0 Xp
    hp3 : LE.le 0 Ym
    hp4 : LE.le 0 Yp
    hm1 : AEMeasurable Xm μ
    hm2 : AEMeasurable Xp μ
    hm3 : AEMeasurable Ym μ
    hm4 : AEMeasurable Yp μ
    hv1 : MeasureTheory.Integrable Xm μ
    hv2 : MeasureTheory.Integrable Xp μ
    hv3 : MeasureTheory.Integrable Ym μ
    hv4 : MeasureTheory.Integrable Yp μ
    hi1 : ProbabilityTheory.IndepFun Xm Ym μ
    hi2 : ProbabilityTheory.IndepFun Xp Ym μ
    hi3 : ProbabilityTheory.IndepFun Xm Yp μ
    hi4 : ProbabilityTheory.IndepFun Xp Yp μ
    hl1 : MeasureTheory.Integrable (HMul.hMul Xm Ym) μ
    hl2 : MeasureTheory.Integrable (HMul.hMul Xp Ym) μ
    hl3 : MeasureTheory.Integrable (HMul.hMul Xm Yp) μ
    hl4 : MeasureTheory.Integrable (HMul.hMul Xp Yp) μ
    hl5 : MeasureTheory.Integrable (HSub.hSub (HMul.hMul Xp Yp) (HMul.hMul Xm Yp)) μ
    hl6 : MeasureTheory.Integrable (HSub.hSub (HMul.hMul Xp Ym) (HMul.hMul Xm Ym)) μ
    ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul (MeasureTheory.integral μ Xp) (MeasureTh …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The (Bochner) integral of the product of two independent random
  variables is the product of their integrals. -/
theorem IndepFun.integral_mul (hXY : IndepFun X Y μ) (hX : AEStronglyMeasurable X μ)
    (hY : AEStronglyMeasurable Y μ) : integral μ (X * Y) = integral μ X * integral μ Y := by
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  by_cases h'X : X =ᵐ[μ] 0
  · have h' : X * Y =ᵐ[μ] 0 := by
      filter_upwards [h'X] with ω hω
      simp [hω]
    simp only [integral_congr_ae h'X, integral_congr_ae h', Pi.zero_apply, integral_const,
      Algebra.id.smul_eq_mul, mul_zero, zero_mul]
  /-
    case neg
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  by_cases h'Y : Y =ᵐ[μ] 0
  · have h' : X * Y =ᵐ[μ] 0 := by
      filter_upwards [h'Y] with ω hω
      simp [hω]
    simp only [integral_congr_ae h'Y, integral_congr_ae h', Pi.zero_apply, integral_const,
      Algebra.id.smul_eq_mul, mul_zero, zero_mul]
  /-
    case neg
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    hXY : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
    h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
    ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
  -/
  by_cases h : Integrable (X * Y) μ
    /-
      case pos
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X Y : Ω → Real
      hXY : ProbabilityTheory.IndepFun X Y μ
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hY : MeasureTheory.AEStronglyMeasurable Y μ
      h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
      h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
      h : MeasureTheory.Integrable (HMul.hMul X Y) μ
      ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
    -/
  · have HX : Integrable X μ := hXY.integrable_left_of_integrable_mul h hX hY h'Y
    /-
      case pos
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X Y : Ω → Real
      hXY : ProbabilityTheory.IndepFun X Y μ
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hY : MeasureTheory.AEStronglyMeasurable Y μ
      h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
      h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
      h : MeasureTheory.Integrable (HMul.hMul X Y) μ
      HX : MeasureTheory.Integrable X μ
      ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
    -/
    have HY : Integrable Y μ := hXY.integrable_right_of_integrable_mul h hX hY h'X
    /-
      case pos
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X Y : Ω → Real
      hXY : ProbabilityTheory.IndepFun X Y μ
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hY : MeasureTheory.AEStronglyMeasurable Y μ
      h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
      h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
      h : MeasureTheory.Integrable (HMul.hMul X Y) μ
      HX : MeasureTheory.Integrable X μ
      HY : MeasureTheory.Integrable Y μ
      ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
    -/
    exact hXY.integral_mul_of_integrable HX HY
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X Y : Ω → Real
      hXY : ProbabilityTheory.IndepFun X Y μ
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hY : MeasureTheory.AEStronglyMeasurable Y μ
      h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
      h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
      h : Not (MeasureTheory.Integrable (HMul.hMul X Y) μ)
      ⊢ Eq (MeasureTheory.integral μ (HMul.hMul X Y)) (HMul.hMul (MeasureTheory.inte …
    -/
  · rw [integral_undef h]
    have I : ¬(Integrable X μ ∧ Integrable Y μ) := by
      rintro ⟨HX, HY⟩
      exact h (hXY.integrable_mul HX HY)
    /-
      case neg
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X Y : Ω → Real
      hXY : ProbabilityTheory.IndepFun X Y μ
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hY : MeasureTheory.AEStronglyMeasurable Y μ
      h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
      h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
      h : Not (MeasureTheory.Integrable (HMul.hMul X Y) μ)
      I : Not (And (MeasureTheory.Integrable X μ) (MeasureTheory.Integrable Y μ))
      ⊢ Eq 0 (HMul.hMul (MeasureTheory.integral μ X) (MeasureTheory.integral μ Y))
    -/
    rw [not_and_or] at I
    /-
      case neg
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X Y : Ω → Real
      hXY : ProbabilityTheory.IndepFun X Y μ
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hY : MeasureTheory.AEStronglyMeasurable Y μ
      h'X : Not ((MeasureTheory.ae μ).EventuallyEq X 0)
      h'Y : Not ((MeasureTheory.ae μ).EventuallyEq Y 0)
      h : Not (MeasureTheory.Integrable (HMul.hMul X Y) μ)
      I : Or (Not (MeasureTheory.Integrable X μ)) (Not (MeasureTheory.Integrable Y μ))
      ⊢ Eq 0 (HMul.hMul (MeasureTheory.integral μ X) (MeasureTheory.integral μ Y))
    -/
                          /-
                            🎉 no goals
                          -/
    cases' I with I I <;> simp [integral_undef I]
                          /-
                            🎉 no goals
                          -/


theorem IndepFun.integral_mul' (hXY : IndepFun X Y μ) (hX : AEStronglyMeasurable X μ)
    (hY : AEStronglyMeasurable Y μ) :
    (integral μ fun ω => X ω * Y ω) = integral μ X * integral μ Y :=
  hXY.integral_mul hX hY


/-- Independence of functions `f` and `g` into arbitrary types is characterized by the relation
  `E[(φ ∘ f) * (ψ ∘ g)] = E[φ ∘ f] * E[ψ ∘ g]` for all measurable `φ` and `ψ` with values in `ℝ`
  satisfying appropriate integrability conditions. -/
theorem indepFun_iff_integral_comp_mul [IsFiniteMeasure μ] {β β' : Type*} {mβ : MeasurableSpace β}
    {mβ' : MeasurableSpace β'} {f : Ω → β} {g : Ω → β'} {hfm : Measurable f} {hgm : Measurable g} :
    IndepFun f g μ ↔ ∀ {φ : β → ℝ} {ψ : β' → ℝ}, Measurable φ → Measurable ψ →
      Integrable (φ ∘ f) μ → Integrable (ψ ∘ g) μ →
        integral μ (φ ∘ f * ψ ∘ g) = integral μ (φ ∘ f) * integral μ (ψ ∘ g) := by
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    β : Type u_2
    β' : Type u_3
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    f : Ω → β
    g : Ω → β'
    hfm : Measurable f
    hgm : Measurable g
    ⊢ Iff (ProbabilityTheory.IndepFun f g μ) (∀ {φ : β → Real} {ψ : β' → Real}, Me …
  -/
  refine ⟨fun hfg _ _ hφ hψ => IndepFun.integral_mul_of_integrable (hfg.comp hφ hψ), ?_⟩
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    β : Type u_2
    β' : Type u_3
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    f : Ω → β
    g : Ω → β'
    hfm : Measurable f
    hgm : Measurable g
    ⊢ (∀ {φ : β → Real} {ψ : β' → Real}, Measurable φ → Measurable ψ → MeasureTheo …
  -/
  rw [IndepFun_iff]
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    β : Type u_2
    β' : Type u_3
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    f : Ω → β
    g : Ω → β'
    hfm : Measurable f
    hgm : Measurable g
    ⊢ (∀ {φ : β → Real} {ψ : β' → Real}, Measurable φ → Measurable ψ → MeasureTheo …
  -/
  rintro h _ _ ⟨A, hA, rfl⟩ ⟨B, hB, rfl⟩
  specialize
    h (measurable_one.indicator hA) (measurable_one.indicator hB)
      ((integrable_const 1).indicator (hfm.comp measurable_id hA))
      ((integrable_const 1).indicator (hgm.comp measurable_id hB))
  rwa [← ENNReal.toReal_eq_toReal (measure_ne_top μ _), ENNReal.toReal_mul, ←
    integral_indicator_one ((hfm hA).inter (hgm hB)), ← integral_indicator_one (hfm hA), ←
    integral_indicator_one (hgm hB), Set.inter_indicator_one]
  /-
    case intro.intro.intro.intro
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    β : Type u_2
    β' : Type u_3
    mβ : MeasurableSpace β
    mβ' : MeasurableSpace β'
    f : Ω → β
    g : Ω → β'
    hfm : Measurable f
    hgm : Measurable g
    A : Set β
    hA : MeasurableSet A
    B : Set β'
    hB : MeasurableSet B
    h : Eq (MeasureTheory.integral μ (HMul.hMul (Function.comp (A.indicator 1) f)  …
    ⊢ Ne (HMul.hMul (μ (Set.preimage f A)) (μ (Set.preimage g B))) Top.top
  -/
  exact ENNReal.mul_ne_top (measure_ne_top μ _) (measure_ne_top μ _)
  /-
    🎉 no goals
  -/


