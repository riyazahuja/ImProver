local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


@[aesop safe 20 apply (rule_sets := [Measurable])]
theorem Measurable.inner {_ : MeasurableSpace α} [MeasurableSpace E] [OpensMeasurableSpace E]
    [SecondCountableTopology E] {f g : α → E} (hf : Measurable f)
    (hg : Measurable g) : Measurable fun t => ⟪f t, g t⟫ :=
  Continuous.measurable2 continuous_inner hf hg


@[measurability]
theorem Measurable.const_inner {_ : MeasurableSpace α} [MeasurableSpace E] [OpensMeasurableSpace E]
    [SecondCountableTopology E] {c : E} {f : α → E} (hf : Measurable f) :
    Measurable fun t => ⟪c, f t⟫ :=
  Measurable.inner measurable_const hf


@[measurability]
theorem Measurable.inner_const {_ : MeasurableSpace α} [MeasurableSpace E] [OpensMeasurableSpace E]
    [SecondCountableTopology E] {c : E} {f : α → E} (hf : Measurable f) :
    Measurable fun t => ⟪f t, c⟫ :=
  Measurable.inner hf measurable_const


@[aesop safe 20 apply (rule_sets := [Measurable])]
theorem AEMeasurable.inner {m : MeasurableSpace α} [MeasurableSpace E] [OpensMeasurableSpace E]
    [SecondCountableTopology E] {μ : MeasureTheory.Measure α} {f g : α → E}
    (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) : AEMeasurable (fun x => ⟪f x, g x⟫) μ := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    m : MeasurableSpace α
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : SecondCountableTopology E
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    ⊢ AEMeasurable (fun x => Inner.inner (f x) (g x)) μ
  -/
  refine ⟨fun x => ⟪hf.mk f x, hg.mk g x⟫, hf.measurable_mk.inner hg.measurable_mk, ?_⟩
  /-
    α : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    m : MeasurableSpace α
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : SecondCountableTopology E
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => Inner.inner (f x) (g x)) fun x = …
  -/
  refine hf.ae_eq_mk.mp (hg.ae_eq_mk.mono fun x hxg hxf => ?_)
  /-
    α : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    m : MeasurableSpace α
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : SecondCountableTopology E
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    x : α
    hxg : Eq (g x) (AEMeasurable.mk g hg x)
    hxf : Eq (f x) (AEMeasurable.mk f hf x)
    ⊢ Eq ((fun x => Inner.inner (f x) (g x)) x) ((fun x => Inner.inner (AEMeasurab …
  -/
  dsimp only
  /-
    α : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    m : MeasurableSpace α
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : SecondCountableTopology E
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    x : α
    hxg : Eq (g x) (AEMeasurable.mk g hg x)
    hxf : Eq (f x) (AEMeasurable.mk f hf x)
    ⊢ Eq (Inner.inner (f x) (g x)) (Inner.inner (AEMeasurable.mk f hf x) (AEMeasur …
  -/
  congr
  /-
    🎉 no goals
  -/


set_option linter.unusedVariables false in
@[measurability]
theorem AEMeasurable.const_inner {m : MeasurableSpace α} [MeasurableSpace E]
    [OpensMeasurableSpace E] [SecondCountableTopology E]
    {μ : MeasureTheory.Measure α} {f : α → E} {c : E} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x => ⟪c, f x⟫) μ :=
  AEMeasurable.inner aemeasurable_const hf


set_option linter.unusedVariables false in
@[measurability]
theorem AEMeasurable.inner_const {m : MeasurableSpace α} [MeasurableSpace E]
    [OpensMeasurableSpace E] [SecondCountableTopology E]
    {μ : MeasureTheory.Measure α} {f : α → E} {c : E} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x => ⟪f x, c⟫) μ :=
  AEMeasurable.inner hf aemeasurable_const

