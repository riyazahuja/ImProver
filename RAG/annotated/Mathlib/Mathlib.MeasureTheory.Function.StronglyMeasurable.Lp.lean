local infixr:25 " →ₛ " => SimpleFunc


theorem Memℒp.finStronglyMeasurable_of_stronglyMeasurable (hf : Memℒp f p μ)
    (hf_meas : StronglyMeasurable f) (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) :
    FinStronglyMeasurable f μ := by
  /-
    α : Type u_1
    G : Type u_2
    p : ENNReal
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    f : α → G
    hf : MeasureTheory.Memℒp f p μ
    hf_meas : MeasureTheory.StronglyMeasurable f
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    ⊢ MeasureTheory.FinStronglyMeasurable f μ
  -/
  borelize G
  haveI : SeparableSpace (Set.range f ∪ {0} : Set G) :=
    hf_meas.separableSpace_range_union_singleton
  /-
    α : Type u_1
    G : Type u_2
    p : ENNReal
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    f : α → G
    hf : MeasureTheory.Memℒp f p μ
    hf_meas : MeasureTheory.StronglyMeasurable f
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
    ⊢ MeasureTheory.FinStronglyMeasurable f μ
  -/
  let fs := SimpleFunc.approxOn f hf_meas.measurable (Set.range f ∪ {0}) 0 (by simp)
  /-
    α : Type u_1
    G : Type u_2
    p : ENNReal
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    f : α → G
    hf : MeasureTheory.Memℒp f p μ
    hf_meas : MeasureTheory.StronglyMeasurable f
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
    fs : Nat → MeasureTheory.SimpleFunc α G := MeasureTheory.SimpleFunc.approxOn f …
    ⊢ MeasureTheory.FinStronglyMeasurable f μ
  -/
  refine ⟨fs, ?_, ?_⟩
  · have h_fs_Lp : ∀ n, Memℒp (fs n) p μ :=
      SimpleFunc.memℒp_approxOn_range hf_meas.measurable hf
    /-
      case refine_1
      α : Type u_1
      G : Type u_2
      p : ENNReal
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      f : α → G
      hf : MeasureTheory.Memℒp f p μ
      hf_meas : MeasureTheory.StronglyMeasurable f
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      this✝¹ : MeasurableSpace G := borel G
      this✝ : BorelSpace G
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
      fs : Nat → MeasureTheory.SimpleFunc α G := MeasureTheory.SimpleFunc.approxOn f …
      h_fs_Lp : ∀ (n : Nat), MeasureTheory.Memℒp (⇑(fs n)) p μ
      ⊢ ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
    -/
    exact fun n => (fs n).measure_support_lt_top_of_memℒp (h_fs_Lp n) hp_ne_zero hp_ne_top
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      G : Type u_2
      p : ENNReal
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      f : α → G
      hf : MeasureTheory.Memℒp f p μ
      hf_meas : MeasureTheory.StronglyMeasurable f
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      this✝¹ : MeasurableSpace G := borel G
      this✝ : BorelSpace G
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
      fs : Nat → MeasureTheory.SimpleFunc α G := MeasureTheory.SimpleFunc.approxOn f …
      ⊢ ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f x))
    -/
  · intro x
    /-
      case refine_2
      α : Type u_1
      G : Type u_2
      p : ENNReal
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      f : α → G
      hf : MeasureTheory.Memℒp f p μ
      hf_meas : MeasureTheory.StronglyMeasurable f
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      this✝¹ : MeasurableSpace G := borel G
      this✝ : BorelSpace G
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
      fs : Nat → MeasureTheory.SimpleFunc α G := MeasureTheory.SimpleFunc.approxOn f …
      x : α
      ⊢ Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f x))
    -/
    apply SimpleFunc.tendsto_approxOn
    /-
      case refine_2.hx
      α : Type u_1
      G : Type u_2
      p : ENNReal
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      f : α → G
      hf : MeasureTheory.Memℒp f p μ
      hf_meas : MeasureTheory.StronglyMeasurable f
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      this✝¹ : MeasurableSpace G := borel G
      this✝ : BorelSpace G
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
      fs : Nat → MeasureTheory.SimpleFunc α G := MeasureTheory.SimpleFunc.approxOn f …
      x : α
      ⊢ Membership.mem (closure (Union.union (Set.range f) (Singleton.singleton 0))) …
    -/
    apply subset_closure
    /-
      case refine_2.hx.a
      α : Type u_1
      G : Type u_2
      p : ENNReal
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      f : α → G
      hf : MeasureTheory.Memℒp f p μ
      hf_meas : MeasureTheory.StronglyMeasurable f
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      this✝¹ : MeasurableSpace G := borel G
      this✝ : BorelSpace G
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
      fs : Nat → MeasureTheory.SimpleFunc α G := MeasureTheory.SimpleFunc.approxOn f …
      x : α
      ⊢ Membership.mem (Union.union (Set.range f) (Singleton.singleton 0)) (f x)
    -/
    simp
    /-
      🎉 no goals
    -/


theorem Memℒp.aefinStronglyMeasurable (hf : Memℒp f p μ) (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) :
    AEFinStronglyMeasurable f μ :=
  ⟨hf.aestronglyMeasurable.mk f,
    ((memℒp_congr_ae hf.aestronglyMeasurable.ae_eq_mk).mp
          hf).finStronglyMeasurable_of_stronglyMeasurable
      hf.aestronglyMeasurable.stronglyMeasurable_mk hp_ne_zero hp_ne_top,
    hf.aestronglyMeasurable.ae_eq_mk⟩


theorem Integrable.aefinStronglyMeasurable (hf : Integrable f μ) : AEFinStronglyMeasurable f μ :=
  (memℒp_one_iff_integrable.mpr hf).aefinStronglyMeasurable one_ne_zero ENNReal.coe_ne_top


theorem Lp.finStronglyMeasurable (f : Lp G p μ) (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) :
    FinStronglyMeasurable f μ :=
  (Lp.memℒp f).finStronglyMeasurable_of_stronglyMeasurable (Lp.stronglyMeasurable f) hp_ne_zero
    hp_ne_top


