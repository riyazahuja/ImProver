/-- A kernel `ρCond` is a conditional kernel for a measure `ρ` if it disintegrates it in the sense
that `ρ.fst ⊗ₘ ρCond = ρ`. -/
class IsCondKernel : Prop where
  disintegrate : ρ.fst ⊗ₘ ρCond = ρ


lemma disintegrate : ρ.fst ⊗ₘ ρCond = ρ := IsCondKernel.disintegrate


lemma IsCondKernel.isSFiniteKernel (hρ : ρ ≠ 0) : IsSFiniteKernel ρCond := by
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    mΩ : MeasurableSpace Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    ρCond : ProbabilityTheory.Kernel α Ω
    inst✝ : ρ.IsCondKernel ρCond
    hρ : Ne ρ 0
    ⊢ ProbabilityTheory.IsSFiniteKernel ρCond
  -/
  contrapose! hρ; rwa [← ρ.disintegrate ρCond, Measure.compProd_of_not_isSFiniteKernel]
                  /-
                    🎉 no goals
                  -/


/-- Auxiliary lemma for `IsCondKernel.apply_of_ne_zero`. -/
private lemma IsCondKernel.apply_of_ne_zero_of_measurableSet [MeasurableSingletonClass α] {x : α}
    (hx : ρ.fst {x} ≠ 0) {s : Set Ω} (hs : MeasurableSet s) :
    ρCond x s = (ρ.fst {x})⁻¹ * ρ ({x} ×ˢ s) := by
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    mΩ : MeasurableSpace Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    ρCond : ProbabilityTheory.Kernel α Ω
    inst✝² : ρ.IsCondKernel ρCond
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    inst✝ : MeasurableSingletonClass α
    x : α
    hx : Ne (ρ.fst (Singleton.singleton x)) 0
    s : Set Ω
    hs : MeasurableSet s
    ⊢ Eq ((ρCond x) s) (HMul.hMul (Inv.inv (ρ.fst (Singleton.singleton x))) (ρ (SP …
  -/
  have := isSFiniteKernel ρ ρCond (by rintro rfl; simp at hx)
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    mΩ : MeasurableSpace Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    ρCond : ProbabilityTheory.Kernel α Ω
    inst✝² : ρ.IsCondKernel ρCond
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    inst✝ : MeasurableSingletonClass α
    x : α
    hx : Ne (ρ.fst (Singleton.singleton x)) 0
    s : Set Ω
    hs : MeasurableSet s
    this : ProbabilityTheory.IsSFiniteKernel ρCond
    ⊢ Eq ((ρCond x) s) (HMul.hMul (Inv.inv (ρ.fst (Singleton.singleton x))) (ρ (SP …
  -/
  nth_rewrite 2 [← ρ.disintegrate ρCond]
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    mΩ : MeasurableSpace Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    ρCond : ProbabilityTheory.Kernel α Ω
    inst✝² : ρ.IsCondKernel ρCond
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    inst✝ : MeasurableSingletonClass α
    x : α
    hx : Ne (ρ.fst (Singleton.singleton x)) 0
    s : Set Ω
    hs : MeasurableSet s
    this : ProbabilityTheory.IsSFiniteKernel ρCond
    ⊢ Eq ((ρCond x) s) (HMul.hMul (Inv.inv (ρ.fst (Singleton.singleton x))) ((ρ.fs …
  -/
  rw [Measure.compProd_apply (measurableSet_prod.mpr (Or.inl ⟨measurableSet_singleton x, hs⟩))]
  classical
  have (a) : ρCond a (Prod.mk a ⁻¹' {x} ×ˢ s) = ({x} : Set α).indicator (ρCond · s) a := by
    obtain rfl | hax := eq_or_ne a x
    · simp only [singleton_prod, mem_singleton_iff, indicator_of_mem]
      congr with y
      simp
    · simp only [singleton_prod, mem_singleton_iff, hax, not_false_eq_true, indicator_of_not_mem]
      have : Prod.mk a ⁻¹' (Prod.mk x '' s) = ∅ := by ext y; simp [Ne.symm hax]
      simp only [this, measure_empty]
  simp_rw [this]
  rw [MeasureTheory.lintegral_indicator (measurableSet_singleton x)]
  simp only [Measure.restrict_singleton, lintegral_smul_measure, lintegral_dirac]
  rw [← mul_assoc, ENNReal.inv_mul_cancel hx (measure_ne_top _ _), one_mul]


/-- If the singleton `{x}` has non-zero mass for `ρ.fst`, then for all `s : Set Ω`,
`ρCond x s = (ρ.fst {x})⁻¹ * ρ ({x} ×ˢ s)` . -/
lemma IsCondKernel.apply_of_ne_zero [MeasurableSingletonClass α] {x : α}
    (hx : ρ.fst {x} ≠ 0) (s : Set Ω) : ρCond x s = (ρ.fst {x})⁻¹ * ρ ({x} ×ˢ s) := by
  have : ρCond x s = ((ρ.fst {x})⁻¹ • ρ).comap (fun (y : Ω) ↦ (x, y)) s := by
    congr 2 with s hs
    simp [IsCondKernel.apply_of_ne_zero_of_measurableSet _ _ hx hs,
      (measurableEmbedding_prod_mk_left x).comap_apply]
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    mΩ : MeasurableSpace Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    ρCond : ProbabilityTheory.Kernel α Ω
    inst✝² : ρ.IsCondKernel ρCond
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    inst✝ : MeasurableSingletonClass α
    x : α
    hx : Ne (ρ.fst (Singleton.singleton x)) 0
    s : Set Ω
    this : Eq ((ρCond x) s) ((MeasureTheory.Measure.comap (fun y => { fst := x, sn …
    ⊢ Eq ((ρCond x) s) (HMul.hMul (Inv.inv (ρ.fst (Singleton.singleton x))) (ρ (SP …
  -/
  simp [this, (measurableEmbedding_prod_mk_left x).comap_apply, hx]
  /-
    🎉 no goals
  -/


lemma IsCondKernel.isProbabilityMeasure [MeasurableSingletonClass α] {a : α} (ha : ρ.fst {a} ≠ 0) :
    IsProbabilityMeasure (ρCond a) := by
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    mΩ : MeasurableSpace Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    ρCond : ProbabilityTheory.Kernel α Ω
    inst✝² : ρ.IsCondKernel ρCond
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    inst✝ : MeasurableSingletonClass α
    a : α
    ha : Ne (ρ.fst (Singleton.singleton a)) 0
    ⊢ MeasureTheory.IsProbabilityMeasure (ρCond a)
  -/
  constructor
  rw [IsCondKernel.apply_of_ne_zero _ _ ha, prod_univ, ← Measure.fst_apply
    (measurableSet_singleton _), ENNReal.inv_mul_cancel ha (measure_ne_top _ _)]


lemma IsCondKernel.isMarkovKernel [MeasurableSingletonClass α] (hρ : ∀ a, ρ.fst {a} ≠ 0) :
    IsMarkovKernel ρCond := ⟨fun _ ↦ isProbabilityMeasure _ _ (hρ _)⟩


/-- A kernel `κCond` is a conditional kernel for a kernel `κ` if it disintegrates it in the sense
that `κ.fst ⊗ₖ κCond = κ`. -/
class IsCondKernel : Prop where
  protected disintegrate : κ.fst ⊗ₖ κCond = κ


instance instIsCondKernel_zero (κCond : Kernel (α × β) Ω) : IsCondKernel 0 κCond where
                     /-
                       α : Type u_1
                       β : Type u_2
                       Ω : Type u_3
                       mα : MeasurableSpace α
                       mβ : MeasurableSpace β
                       mΩ : MeasurableSpace Ω
                       κ : ProbabilityTheory.Kernel α (Prod β Ω)
                       κCond✝ κCond : ProbabilityTheory.Kernel (Prod α β) Ω
                       ⊢ Eq ((ProbabilityTheory.Kernel.fst 0).compProd κCond) 0
                     -/
  disintegrate := by simp
                     /-
                       🎉 no goals
                     -/


lemma disintegrate : κ.fst ⊗ₖ κCond = κ := IsCondKernel.disintegrate


/-- Auxiliary definition for `ProbabilityTheory.Kernel.condKernel`.

A conditional kernel for `κ : Kernel α (β × Ω)` where `α` is countable and `Ω` is a measurable
space. -/
noncomputable def condKernelCountable (h_atom : ∀ x y, x ∈ measurableAtom y → κCond x = κCond y) :
    Kernel (α × β) Ω where
  toFun p := κCond p.1 p.2
  measurable' := by
    /-
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α (Prod β Ω)
      κCond✝ : ProbabilityTheory.Kernel (Prod α β) Ω
      inst✝¹ : κ.IsCondKernel κCond✝
      inst✝ : Countable α
      κCond : α → ProbabilityTheory.Kernel β Ω
      h_atom : ∀ (x y : α), Membership.mem (measurableAtom y) x → Eq (κCond x) (κCon …
      ⊢ Measurable fun p => (κCond p.1) p.2
    -/
    change Measurable ((fun q : β × α ↦ (κCond q.2) q.1) ∘ Prod.swap)
    /-
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mΩ : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α (Prod β Ω)
      κCond✝ : ProbabilityTheory.Kernel (Prod α β) Ω
      inst✝¹ : κ.IsCondKernel κCond✝
      inst✝ : Countable α
      κCond : α → ProbabilityTheory.Kernel β Ω
      h_atom : ∀ (x y : α), Membership.mem (measurableAtom y) x → Eq (κCond x) (κCon …
      ⊢ Measurable (Function.comp (fun q => (κCond q.2) q.1) Prod.swap)
    -/
    refine (measurable_from_prod_countable' (fun a ↦ (κCond a).measurable) ?_).comp measurable_swap
      /-
        α : Type u_1
        β : Type u_2
        Ω : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mΩ : MeasurableSpace Ω
        κ : ProbabilityTheory.Kernel α (Prod β Ω)
        κCond✝ : ProbabilityTheory.Kernel (Prod α β) Ω
        inst✝¹ : κ.IsCondKernel κCond✝
        inst✝ : Countable α
        κCond : α → ProbabilityTheory.Kernel β Ω
        h_atom : ∀ (x y : α), Membership.mem (measurableAtom y) x → Eq (κCond x) (κCon …
        ⊢ ∀ (y y' : α) (x : β), Membership.mem (measurableAtom y) y' → Eq ((κCond { fs …
      -/
    · intro x y hx hy
      /-
        α : Type u_1
        β : Type u_2
        Ω : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mΩ : MeasurableSpace Ω
        κ : ProbabilityTheory.Kernel α (Prod β Ω)
        κCond✝ : ProbabilityTheory.Kernel (Prod α β) Ω
        inst✝¹ : κ.IsCondKernel κCond✝
        inst✝ : Countable α
        κCond : α → ProbabilityTheory.Kernel β Ω
        h_atom : ∀ (x y : α), Membership.mem (measurableAtom y) x → Eq (κCond x) (κCon …
        x y : α
        hx : β
        hy : Membership.mem (measurableAtom x) y
        ⊢ Eq ((κCond { fst := hx, snd := y }.2) { fst := hx, snd := y }.1) ((κCond { f …
      -/
      simpa using DFunLike.congr (h_atom _ _ hy) rfl
      /-
        🎉 no goals
      -/


lemma condKernelCountable_apply (h_atom : ∀ x y, x ∈ measurableAtom y → κCond x = κCond y)
    (p : α × β) : condKernelCountable κCond h_atom p = κCond p.1 p.2 := rfl


instance condKernelCountable.instIsMarkovKernel [∀ a, IsMarkovKernel (κCond a)]
     (h_atom : ∀ x y, x ∈ measurableAtom y → κCond x = κCond y) :
    IsMarkovKernel (condKernelCountable κCond h_atom) where
  isProbabilityMeasure p := (‹∀ a, IsMarkovKernel (κCond a)› p.1).isProbabilityMeasure p.2


instance condKernelCountable.instIsCondKernel [∀ a, IsMarkovKernel (κCond a)]
    (h_atom : ∀ x y, x ∈ measurableAtom y → κCond x = κCond y) (κ : Kernel α (β × Ω))
    [IsSFiniteKernel κ] [∀ a, (κ a).IsCondKernel (κCond a)] :
    κ.IsCondKernel (condKernelCountable κCond h_atom) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mΩ : MeasurableSpace Ω
    κ✝ : ProbabilityTheory.Kernel α (Prod β Ω)
    κCond✝ : ProbabilityTheory.Kernel (Prod α β) Ω
    inst✝⁴ : κ✝.IsCondKernel κCond✝
    inst✝³ : Countable α
    κCond : α → ProbabilityTheory.Kernel β Ω
    inst✝² : ∀ (a : α), ProbabilityTheory.IsMarkovKernel (κCond a)
    h_atom : ∀ (x y : α), Membership.mem (measurableAtom y) x → Eq (κCond x) (κCon …
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ∀ (a : α), (κ a).IsCondKernel (κCond a)
    ⊢ κ.IsCondKernel (ProbabilityTheory.Kernel.condKernelCountable κCond h_atom)
  -/
  constructor
  /-
    case disintegrate
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mΩ : MeasurableSpace Ω
    κ✝ : ProbabilityTheory.Kernel α (Prod β Ω)
    κCond✝ : ProbabilityTheory.Kernel (Prod α β) Ω
    inst✝⁴ : κ✝.IsCondKernel κCond✝
    inst✝³ : Countable α
    κCond : α → ProbabilityTheory.Kernel β Ω
    inst✝² : ∀ (a : α), ProbabilityTheory.IsMarkovKernel (κCond a)
    h_atom : ∀ (x y : α), Membership.mem (measurableAtom y) x → Eq (κCond x) (κCon …
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ∀ (a : α), (κ a).IsCondKernel (κCond a)
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.condKernelCountable κCond h_ato …
  -/
  ext a s hs
  /-
    case disintegrate.h.h
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mΩ : MeasurableSpace Ω
    κ✝ : ProbabilityTheory.Kernel α (Prod β Ω)
    κCond✝ : ProbabilityTheory.Kernel (Prod α β) Ω
    inst✝⁴ : κ✝.IsCondKernel κCond✝
    inst✝³ : Countable α
    κCond : α → ProbabilityTheory.Kernel β Ω
    inst✝² : ∀ (a : α), ProbabilityTheory.IsMarkovKernel (κCond a)
    h_atom : ∀ (x y : α), Membership.mem (measurableAtom y) x → Eq (κCond x) (κCon …
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ∀ (a : α), (κ a).IsCondKernel (κCond a)
    a : α
    s : Set (Prod β Ω)
    hs : MeasurableSet s
    ⊢ Eq (((κ.fst.compProd (ProbabilityTheory.Kernel.condKernelCountable κCond h_a …
  -/
  conv_rhs => rw [← (κ a).disintegrate (κCond a)]
  /-
    case disintegrate.h.h
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mΩ : MeasurableSpace Ω
    κ✝ : ProbabilityTheory.Kernel α (Prod β Ω)
    κCond✝ : ProbabilityTheory.Kernel (Prod α β) Ω
    inst✝⁴ : κ✝.IsCondKernel κCond✝
    inst✝³ : Countable α
    κCond : α → ProbabilityTheory.Kernel β Ω
    inst✝² : ∀ (a : α), ProbabilityTheory.IsMarkovKernel (κCond a)
    h_atom : ∀ (x y : α), Membership.mem (measurableAtom y) x → Eq (κCond x) (κCon …
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ∀ (a : α), (κ a).IsCondKernel (κCond a)
    a : α
    s : Set (Prod β Ω)
    hs : MeasurableSet s
    ⊢ Eq (((κ.fst.compProd (ProbabilityTheory.Kernel.condKernelCountable κCond h_a …
  -/
  simp_rw [compProd_apply hs, condKernelCountable_apply, Measure.compProd_apply hs]
  /-
    case disintegrate.h.h
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mΩ : MeasurableSpace Ω
    κ✝ : ProbabilityTheory.Kernel α (Prod β Ω)
    κCond✝ : ProbabilityTheory.Kernel (Prod α β) Ω
    inst✝⁴ : κ✝.IsCondKernel κCond✝
    inst✝³ : Countable α
    κCond : α → ProbabilityTheory.Kernel β Ω
    inst✝² : ∀ (a : α), ProbabilityTheory.IsMarkovKernel (κCond a)
    h_atom : ∀ (x y : α), Membership.mem (measurableAtom y) x → Eq (κCond x) (κCon …
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ∀ (a : α), (κ a).IsCondKernel (κCond a)
    a : α
    s : Set (Prod β Ω)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (κ.fst a) fun b => ((κCond a) b) (setOf fun c => …
  -/
  congr
  /-
    🎉 no goals
  -/


