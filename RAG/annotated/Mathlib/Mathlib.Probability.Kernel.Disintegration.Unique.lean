/-- A s-finite kernel which satisfy the disintegration property of the given measure `ρ` is almost
everywhere equal to the disintegration kernel of `ρ` when evaluated on a measurable set.

This theorem in the case of finite kernels is weaker than `eq_condKernel_of_measure_eq_compProd`
which asserts that the kernels are equal almost everywhere and not just on a given measurable
set. -/
theorem eq_condKernel_of_measure_eq_compProd' (κ : Kernel α Ω) [IsSFiniteKernel κ]
    (hκ : ρ = ρ.fst ⊗ₘ κ) {s : Set Ω} (hs : MeasurableSet s) :
    ∀ᵐ x ∂ρ.fst, κ x s = ρ.condKernel x s := by
  refine ae_eq_of_forall_setLIntegral_eq_of_sigmaFinite
    (Kernel.measurable_coe κ hs) (Kernel.measurable_coe ρ.condKernel hs) (fun t ht _ ↦ ?_)
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    κ : ProbabilityTheory.Kernel α Ω
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : Eq ρ (ρ.fst.compProd κ)
    s : Set Ω
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    x✝ : LT.lt (ρ.fst t) Top.top
    ⊢ Eq (MeasureTheory.lintegral (ρ.fst.restrict t) fun x => (κ x) s) (MeasureThe …
  -/
  conv_rhs => rw [Measure.setLIntegral_condKernel_eq_measure_prod ht hs, hκ]
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    κ : ProbabilityTheory.Kernel α Ω
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : Eq ρ (ρ.fst.compProd κ)
    s : Set Ω
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    x✝ : LT.lt (ρ.fst t) Top.top
    ⊢ Eq (MeasureTheory.lintegral (ρ.fst.restrict t) fun x => (κ x) s) ((ρ.fst.com …
  -/
  simp only [Measure.compProd_apply (ht.prod hs), Set.mem_prod, ← lintegral_indicator ht]
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    κ : ProbabilityTheory.Kernel α Ω
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : Eq ρ (ρ.fst.compProd κ)
    s : Set Ω
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    x✝ : LT.lt (ρ.fst t) Top.top
    ⊢ Eq (MeasureTheory.lintegral ρ.fst fun a => t.indicator (fun x => (κ x) s) a) …
  -/
  congr with x
  /-
    case e_f.h
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    κ : ProbabilityTheory.Kernel α Ω
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : Eq ρ (ρ.fst.compProd κ)
    s : Set Ω
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    x✝ : LT.lt (ρ.fst t) Top.top
    x : α
    ⊢ Eq (t.indicator (fun x => (κ x) s) x) ((κ x) (Set.preimage (Prod.mk x) (SPro …
  -/
  by_cases hx : x ∈ t
  /-
    case pos
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    κ : ProbabilityTheory.Kernel α Ω
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : Eq ρ (ρ.fst.compProd κ)
    s : Set Ω
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    x✝ : LT.lt (ρ.fst t) Top.top
    x : α
    hx : Membership.mem t x
    ⊢ Eq (t.indicator (fun x => (κ x) s) x) ((κ x) (Set.preimage (Prod.mk x) (SPro …
  -/
  all_goals simp [hx]
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma for `eq_condKernel_of_measure_eq_compProd`.
Uniqueness of the disintegration kernel on ℝ. -/
lemma eq_condKernel_of_measure_eq_compProd_real {ρ : Measure (α × ℝ)} [IsFiniteMeasure ρ]
    (κ : Kernel α ℝ) [IsFiniteKernel κ] (hκ : ρ = ρ.fst ⊗ₘ κ) :
    ∀ᵐ x ∂ρ.fst, κ x = ρ.condKernel x := by
  have huniv : ∀ᵐ x ∂ρ.fst, κ x Set.univ = ρ.condKernel x Set.univ :=
    eq_condKernel_of_measure_eq_compProd' κ hκ MeasurableSet.univ
  suffices ∀ᵐ x ∂ρ.fst, ∀ ⦃t⦄, MeasurableSet t → κ x t = ρ.condKernel x t by
    filter_upwards [this] with x hx
    ext t ht; exact hx ht
  apply MeasurableSpace.ae_induction_on_inter Real.borel_eq_generateFrom_Iic_rat
    Real.isPiSystem_Iic_rat
    /-
      case h_empty
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      huniv : Filter.Eventually (fun x => Eq ((κ x) Set.univ) ((ρ.condKernel x) Set. …
      ⊢ Filter.Eventually (fun x => Eq ((κ x) EmptyCollection.emptyCollection) ((ρ.c …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h_basic
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      huniv : Filter.Eventually (fun x => Eq ((κ x) Set.univ) ((ρ.condKernel x) Set. …
      ⊢ Filter.Eventually (fun x => ∀ (t : Set Real), Membership.mem (Set.iUnion fun …
    -/
  · simp only [iUnion_singleton_eq_range, mem_range, forall_exists_index, forall_apply_eq_imp_iff]
    /-
      case h_basic
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      huniv : Filter.Eventually (fun x => Eq ((κ x) Set.univ) ((ρ.condKernel x) Set. …
      ⊢ Filter.Eventually (fun x => ∀ (a : Rat), Eq ((κ x) (Set.Iic ↑a)) ((ρ.condKer …
    -/
    exact ae_all_iff.2 fun q ↦ eq_condKernel_of_measure_eq_compProd' κ hκ measurableSet_Iic
    /-
      🎉 no goals
    -/
    /-
      case h_compl
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      huniv : Filter.Eventually (fun x => Eq ((κ x) Set.univ) ((ρ.condKernel x) Set. …
      ⊢ Filter.Eventually (fun x => ∀ (t : Set Real), MeasurableSet t → Eq ((κ x) t) …
    -/
  · filter_upwards [huniv] with x hxuniv t ht heq
    /-
      case h
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      huniv : Filter.Eventually (fun x => Eq ((κ x) Set.univ) ((ρ.condKernel x) Set. …
      x : α
      hxuniv : Eq ((κ x) Set.univ) ((ρ.condKernel x) Set.univ)
      t : Set Real
      ht : MeasurableSet t
      heq : Eq ((κ x) t) ((ρ.condKernel x) t)
      ⊢ Eq ((κ x) (HasCompl.compl t)) ((ρ.condKernel x) (HasCompl.compl t))
    -/
    rw [measure_compl ht <| measure_ne_top _ _, heq, hxuniv, measure_compl ht <| measure_ne_top _ _]
    /-
      🎉 no goals
    -/
    /-
      case h_union
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      huniv : Filter.Eventually (fun x => Eq ((κ x) Set.univ) ((ρ.condKernel x) Set. …
      ⊢ Filter.Eventually (fun x => ∀ (f : Nat → Set Real), Pairwise (Function.onFun …
    -/
  · refine ae_of_all _ (fun x f hdisj hf heq ↦ ?_)
    /-
      case h_union
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      huniv : Filter.Eventually (fun x => Eq ((κ x) Set.univ) ((ρ.condKernel x) Set. …
      x : α
      f : Nat → Set Real
      hdisj : Pairwise (Function.onFun Disjoint f)
      hf : ∀ (i : Nat), MeasurableSet (f i)
      heq : ∀ (i : Nat), Eq ((κ x) (f i)) ((ρ.condKernel x) (f i))
      ⊢ Eq ((κ x) (Set.iUnion fun i => f i)) ((ρ.condKernel x) (Set.iUnion fun i =>  …
    -/
    rw [measure_iUnion hdisj hf, measure_iUnion hdisj hf]
    /-
      case h_union
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      huniv : Filter.Eventually (fun x => Eq ((κ x) Set.univ) ((ρ.condKernel x) Set. …
      x : α
      f : Nat → Set Real
      hdisj : Pairwise (Function.onFun Disjoint f)
      hf : ∀ (i : Nat), MeasurableSet (f i)
      heq : ∀ (i : Nat), Eq ((κ x) (f i)) ((ρ.condKernel x) (f i))
      ⊢ Eq (tsum fun i => (κ x) (f i)) (tsum fun i => (ρ.condKernel x) (f i))
    -/
    exact tsum_congr heq
    /-
      🎉 no goals
    -/


/-- A finite kernel which satisfies the disintegration property is almost everywhere equal to the
disintegration kernel. -/
theorem eq_condKernel_of_measure_eq_compProd (κ : Kernel α Ω) [IsFiniteKernel κ]
    (hκ : ρ = ρ.fst ⊗ₘ κ) :
    ∀ᵐ x ∂ρ.fst, κ x = ρ.condKernel x := by
  -- The idea is to transport the question to `ℝ` from `Ω` using `embeddingReal`
  -- and then construct a measure on `α × ℝ`
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    κ : ProbabilityTheory.Kernel α Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq ρ (ρ.fst.compProd κ)
    ⊢ Filter.Eventually (fun x => Eq (κ x) (ρ.condKernel x)) (MeasureTheory.ae ρ.f …
  -/
  let f := embeddingReal Ω
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    κ : ProbabilityTheory.Kernel α Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq ρ (ρ.fst.compProd κ)
    f : Ω → Real := MeasureTheory.embeddingReal Ω
    ⊢ Filter.Eventually (fun x => Eq (κ x) (ρ.condKernel x)) (MeasureTheory.ae ρ.f …
  -/
  have hf := measurableEmbedding_embeddingReal Ω
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    κ : ProbabilityTheory.Kernel α Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq ρ (ρ.fst.compProd κ)
    f : Ω → Real := MeasureTheory.embeddingReal Ω
    hf : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
    ⊢ Filter.Eventually (fun x => Eq (κ x) (ρ.condKernel x)) (MeasureTheory.ae ρ.f …
  -/
  set ρ' : Measure (α × ℝ) := ρ.map (Prod.map id f) with hρ'def
  have hρ' : ρ'.fst = ρ.fst := by
    ext s hs
    rw [hρ'def, Measure.fst_apply, Measure.fst_apply, Measure.map_apply]
    exacts [rfl, Measurable.prod measurable_fst <| hf.measurable.comp measurable_snd,
      measurable_fst hs, hs, hs]
  have hρ'' : ∀ᵐ x ∂ρ.fst, Kernel.map κ f x = ρ'.condKernel x := by
    rw [← hρ']
    refine eq_condKernel_of_measure_eq_compProd_real (Kernel.map κ f) ?_
    ext s hs
    conv_lhs => rw [hρ'def, hκ]
    rw [Measure.map_apply (measurable_id.prod_map hf.measurable) hs, hρ',
      Measure.compProd_apply hs, Measure.compProd_apply (measurable_id.prod_map hf.measurable hs)]
    congr with a
    rw [Kernel.map_apply' _ hf.measurable]
    exacts [rfl, measurable_prod_mk_left hs]
  suffices ∀ᵐ x ∂ρ.fst, ∀ s, MeasurableSet s → ρ'.condKernel x s = ρ.condKernel x (f ⁻¹' s) by
    filter_upwards [hρ'', this] with x hx h
    rw [Kernel.map_apply _ hf.measurable] at hx
    ext s hs
    rw [← Set.preimage_image_eq s hf.injective,
      ← Measure.map_apply hf.measurable <| hf.measurableSet_image.2 hs, hx,
      h _ <| hf.measurableSet_image.2 hs]
  suffices ρ.map (Prod.map id f) = (ρ.fst ⊗ₘ (Kernel.map ρ.condKernel f)) by
    rw [← hρ'] at this
    have heq := eq_condKernel_of_measure_eq_compProd_real _ this
    rw [hρ'] at heq
    filter_upwards [heq] with x hx s hs
    rw [← hx, Kernel.map_apply _ hf.measurable, Measure.map_apply hf.measurable hs]
  /-
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    κ : ProbabilityTheory.Kernel α Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq ρ (ρ.fst.compProd κ)
    f : Ω → Real := MeasureTheory.embeddingReal Ω
    hf : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
    ρ' : MeasureTheory.Measure (Prod α Real) := MeasureTheory.Measure.map (Prod.ma …
    hρ'def : Eq ρ' (MeasureTheory.Measure.map (Prod.map id f) ρ)
    hρ' : Eq ρ'.fst ρ.fst
    hρ'' : Filter.Eventually (fun x => Eq ((κ.map f) x) (ρ'.condKernel x)) (Measur …
    ⊢ Eq (MeasureTheory.Measure.map (Prod.map id f) ρ) (ρ.fst.compProd (ρ.condKern …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    Ω : Type u_3
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
    κ : ProbabilityTheory.Kernel α Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq ρ (ρ.fst.compProd κ)
    f : Ω → Real := MeasureTheory.embeddingReal Ω
    hf : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
    ρ' : MeasureTheory.Measure (Prod α Real) := MeasureTheory.Measure.map (Prod.ma …
    hρ'def : Eq ρ' (MeasureTheory.Measure.map (Prod.map id f) ρ)
    hρ' : Eq ρ'.fst ρ.fst
    hρ'' : Filter.Eventually (fun x => Eq ((κ.map f) x) (ρ'.condKernel x)) (Measur …
    s : Set (Prod α Real)
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.map (Prod.map id f) ρ) s) ((ρ.fst.compProd (ρ.con …
  -/
  conv_lhs => rw [← ρ.disintegrate ρ.condKernel]
  rw [Measure.compProd_apply hs, Measure.map_apply (measurable_id.prod_map hf.measurable) hs,
    Measure.compProd_apply]
    /-
      case h
      α : Type u_1
      Ω : Type u_3
      mα : MeasurableSpace α
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : StandardBorelSpace Ω
      inst✝² : Nonempty Ω
      ρ : MeasureTheory.Measure (Prod α Ω)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Ω
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      f : Ω → Real := MeasureTheory.embeddingReal Ω
      hf : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
      ρ' : MeasureTheory.Measure (Prod α Real) := MeasureTheory.Measure.map (Prod.ma …
      hρ'def : Eq ρ' (MeasureTheory.Measure.map (Prod.map id f) ρ)
      hρ' : Eq ρ'.fst ρ.fst
      hρ'' : Filter.Eventually (fun x => Eq ((κ.map f) x) (ρ'.condKernel x)) (Measur …
      s : Set (Prod α Real)
      hs : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral ρ.fst fun a => (ρ.condKernel a) (Set.preimage (P …
    -/
  · congr with a
    /-
      case h.e_f.h
      α : Type u_1
      Ω : Type u_3
      mα : MeasurableSpace α
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : StandardBorelSpace Ω
      inst✝² : Nonempty Ω
      ρ : MeasureTheory.Measure (Prod α Ω)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Ω
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      f : Ω → Real := MeasureTheory.embeddingReal Ω
      hf : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
      ρ' : MeasureTheory.Measure (Prod α Real) := MeasureTheory.Measure.map (Prod.ma …
      hρ'def : Eq ρ' (MeasureTheory.Measure.map (Prod.map id f) ρ)
      hρ' : Eq ρ'.fst ρ.fst
      hρ'' : Filter.Eventually (fun x => Eq ((κ.map f) x) (ρ'.condKernel x)) (Measur …
      s : Set (Prod α Real)
      hs : MeasurableSet s
      a : α
      ⊢ Eq ((ρ.condKernel a) (Set.preimage (Prod.mk a) (Set.preimage (Prod.map id (M …
    -/
    rw [Kernel.map_apply' _ hf.measurable]
    /-
      case h.e_f.h
      α : Type u_1
      Ω : Type u_3
      mα : MeasurableSpace α
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : StandardBorelSpace Ω
      inst✝² : Nonempty Ω
      ρ : MeasureTheory.Measure (Prod α Ω)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Ω
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      f : Ω → Real := MeasureTheory.embeddingReal Ω
      hf : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
      ρ' : MeasureTheory.Measure (Prod α Real) := MeasureTheory.Measure.map (Prod.ma …
      hρ'def : Eq ρ' (MeasureTheory.Measure.map (Prod.map id f) ρ)
      hρ' : Eq ρ'.fst ρ.fst
      hρ'' : Filter.Eventually (fun x => Eq ((κ.map f) x) (ρ'.condKernel x)) (Measur …
      s : Set (Prod α Real)
      hs : MeasurableSet s
      a : α
      ⊢ Eq ((ρ.condKernel a) (Set.preimage (Prod.mk a) (Set.preimage (Prod.map id (M …
    -/
    exacts [rfl, measurable_prod_mk_left hs]
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      Ω : Type u_3
      mα : MeasurableSpace α
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : StandardBorelSpace Ω
      inst✝² : Nonempty Ω
      ρ : MeasureTheory.Measure (Prod α Ω)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ
      κ : ProbabilityTheory.Kernel α Ω
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hκ : Eq ρ (ρ.fst.compProd κ)
      f : Ω → Real := MeasureTheory.embeddingReal Ω
      hf : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
      ρ' : MeasureTheory.Measure (Prod α Real) := MeasureTheory.Measure.map (Prod.ma …
      hρ'def : Eq ρ' (MeasureTheory.Measure.map (Prod.map id f) ρ)
      hρ' : Eq ρ'.fst ρ.fst
      hρ'' : Filter.Eventually (fun x => Eq ((κ.map f) x) (ρ'.condKernel x)) (Measur …
      s : Set (Prod α Real)
      hs : MeasurableSet s
      ⊢ MeasurableSet (Set.preimage (Prod.map id (MeasureTheory.embeddingReal Ω)) s)
    -/
  · exact measurable_id.prod_map hf.measurable hs
    /-
      🎉 no goals
    -/


lemma Kernel.apply_eq_measure_condKernel_of_compProd_eq
    {ρ : Kernel α (β × Ω)} [IsFiniteKernel ρ] {κ : Kernel (α × β) Ω} [IsFiniteKernel κ]
    (hκ : Kernel.fst ρ ⊗ₖ κ = ρ) (a : α) :
    (fun b ↦ κ (a, b)) =ᵐ[Kernel.fst ρ a] (ρ a).condKernel := by
  have : ρ a = (ρ a).fst ⊗ₘ Kernel.comap κ (fun b ↦ (a, b)) measurable_prod_mk_left := by
    ext s hs
    conv_lhs => rw [← hκ]
    rw [Measure.compProd_apply hs, Kernel.compProd_apply hs]
    rfl
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsFiniteKernel ρ
    κ : ProbabilityTheory.Kernel (Prod α β) Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq (ρ.fst.compProd κ) ρ
    a : α
    this : Eq (ρ a) ((ρ a).fst.compProd (κ.comap (fun b => { fst := a, snd := b }) …
    ⊢ (MeasureTheory.ae (ρ.fst a)).EventuallyEq (fun b => κ { fst := a, snd := b } …
  -/
  have h := eq_condKernel_of_measure_eq_compProd _ this
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsFiniteKernel ρ
    κ : ProbabilityTheory.Kernel (Prod α β) Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq (ρ.fst.compProd κ) ρ
    a : α
    this : Eq (ρ a) ((ρ a).fst.compProd (κ.comap (fun b => { fst := a, snd := b }) …
    h : Filter.Eventually (fun x => Eq ((κ.comap (fun b => { fst := a, snd := b }) …
    ⊢ (MeasureTheory.ae (ρ.fst a)).EventuallyEq (fun b => κ { fst := a, snd := b } …
  -/
  rw [Kernel.fst_apply]
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsFiniteKernel ρ
    κ : ProbabilityTheory.Kernel (Prod α β) Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq (ρ.fst.compProd κ) ρ
    a : α
    this : Eq (ρ a) ((ρ a).fst.compProd (κ.comap (fun b => { fst := a, snd := b }) …
    h : Filter.Eventually (fun x => Eq ((κ.comap (fun b => { fst := a, snd := b }) …
    ⊢ (MeasureTheory.ae (MeasureTheory.Measure.map Prod.fst (ρ a))).EventuallyEq ( …
  -/
  filter_upwards [h] with b hb
  /-
    case h
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsFiniteKernel ρ
    κ : ProbabilityTheory.Kernel (Prod α β) Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq (ρ.fst.compProd κ) ρ
    a : α
    this : Eq (ρ a) ((ρ a).fst.compProd (κ.comap (fun b => { fst := a, snd := b }) …
    h : Filter.Eventually (fun x => Eq ((κ.comap (fun b => { fst := a, snd := b }) …
    b : β
    hb : Eq ((κ.comap (fun b => { fst := a, snd := b }) ⋯) b) ((ρ a).condKernel b)
    ⊢ Eq (κ { fst := a, snd := b }) ((ρ a).condKernel b)
  -/
  rw [← hb, Kernel.comap_apply]
  /-
    🎉 no goals
  -/


/-- For `fst κ a`-almost all `b`, the conditional kernel `Kernel.condKernel κ` applied to `(a, b)`
is equal to the conditional kernel of the measure `κ a` applied to `b`. -/
lemma Kernel.condKernel_apply_eq_condKernel [CountableOrCountablyGenerated α β]
    (κ : Kernel α (β × Ω)) [IsFiniteKernel κ] (a : α) :
    (fun b ↦ Kernel.condKernel κ (a, b)) =ᵐ[Kernel.fst κ a] (κ a).condKernel :=
  Kernel.apply_eq_measure_condKernel_of_compProd_eq (κ.disintegrate _) a


lemma condKernel_const [CountableOrCountablyGenerated α β] (ρ : Measure (β × Ω)) [IsFiniteMeasure ρ]
    (a : α) :
    (fun b ↦ Kernel.condKernel (Kernel.const α ρ) (a, b)) =ᵐ[ρ.fst] ρ.condKernel := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    a : α
    ⊢ (MeasureTheory.ae ρ.fst).EventuallyEq (fun b => (ProbabilityTheory.Kernel.co …
  -/
  have h := Kernel.condKernel_apply_eq_condKernel (Kernel.const α ρ) a
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    a : α
    h : (MeasureTheory.ae ((ProbabilityTheory.Kernel.const α ρ).fst a)).Eventually …
    ⊢ (MeasureTheory.ae ρ.fst).EventuallyEq (fun b => (ProbabilityTheory.Kernel.co …
  -/
  simp_rw [Kernel.fst_apply, Kernel.const_apply] at h
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    a : α
    h : (MeasureTheory.ae (MeasureTheory.Measure.map Prod.fst ρ)).EventuallyEq (fu …
    ⊢ (MeasureTheory.ae ρ.fst).EventuallyEq (fun b => (ProbabilityTheory.Kernel.co …
  -/
  filter_upwards [h] with b hb using hb
  /-
    🎉 no goals
  -/


/-- A finite kernel which satisfies the disintegration property is almost everywhere equal to the
disintegration kernel. -/
theorem eq_condKernel_of_kernel_eq_compProd [CountableOrCountablyGenerated α β]
    {ρ : Kernel α (β × Ω)} [IsFiniteKernel ρ] {κ : Kernel (α × β) Ω} [IsFiniteKernel κ]
    (hκ : Kernel.fst ρ ⊗ₖ κ = ρ) (a : α) :
    ∀ᵐ x ∂(Kernel.fst ρ a), κ (a, x) = Kernel.condKernel ρ (a, x) := by
  filter_upwards [Kernel.condKernel_apply_eq_condKernel ρ a,
    Kernel.apply_eq_measure_condKernel_of_compProd_eq hκ a] with a h1 h2
  /-
    case h
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    inst✝² : MeasurableSpace.CountableOrCountablyGenerated α β
    ρ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsFiniteKernel ρ
    κ : ProbabilityTheory.Kernel (Prod α β) Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq (ρ.fst.compProd κ) ρ
    a✝ : α
    a : β
    h1 : Eq (ρ.condKernel { fst := a✝, snd := a }) ((ρ a✝).condKernel a)
    h2 : Eq (κ { fst := a✝, snd := a }) ((ρ a✝).condKernel a)
    ⊢ Eq (κ { fst := a✝, snd := a }) (ρ.condKernel { fst := a✝, snd := a })
  -/
  rw [h1, h2]
  /-
    🎉 no goals
  -/


