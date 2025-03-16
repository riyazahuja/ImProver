theorem measurableSet_lineDifferentiableAt (hf : Continuous f) :
    MeasurableSet {x : E | LineDifferentiableAt 𝕜 f x v} := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : OpensMeasurableSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    v : E
    hf : Continuous f
    ⊢ MeasurableSet (setOf fun x => LineDifferentiableAt 𝕜 f x v)
  -/
  borelize 𝕜
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : OpensMeasurableSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    v : E
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    ⊢ MeasurableSet (setOf fun x => LineDifferentiableAt 𝕜 f x v)
  -/
  let g : E → 𝕜 → F := fun x t ↦ f (x + t • v)
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : OpensMeasurableSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    v : E
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    g : E → 𝕜 → F := fun x t => f (HAdd.hAdd x (HSMul.hSMul t v))
    ⊢ MeasurableSet (setOf fun x => LineDifferentiableAt 𝕜 f x v)
  -/
  have hg : Continuous g.uncurry := by fun_prop
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : OpensMeasurableSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    v : E
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    g : E → 𝕜 → F := fun x t => f (HAdd.hAdd x (HSMul.hSMul t v))
    hg : Continuous (Function.uncurry g)
    ⊢ MeasurableSet (setOf fun x => LineDifferentiableAt 𝕜 f x v)
  -/
  exact measurable_prod_mk_right (measurableSet_of_differentiableAt_with_param 𝕜 hg)
  /-
    🎉 no goals
  -/


theorem measurable_lineDeriv [MeasurableSpace F] [BorelSpace F]
    (hf : Continuous f) : Measurable (fun x ↦ lineDeriv 𝕜 f x v) := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : OpensMeasurableSpace E
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace F
    f : E → F
    v : E
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    hf : Continuous f
    ⊢ Measurable fun x => lineDeriv 𝕜 f x v
  -/
  borelize 𝕜
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : OpensMeasurableSpace E
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace F
    f : E → F
    v : E
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    ⊢ Measurable fun x => lineDeriv 𝕜 f x v
  -/
  let g : E → 𝕜 → F := fun x t ↦ f (x + t • v)
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : OpensMeasurableSpace E
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace F
    f : E → F
    v : E
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    g : E → 𝕜 → F := fun x t => f (HAdd.hAdd x (HSMul.hSMul t v))
    ⊢ Measurable fun x => lineDeriv 𝕜 f x v
  -/
  have hg : Continuous g.uncurry := by fun_prop
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : OpensMeasurableSpace E
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace F
    f : E → F
    v : E
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    g : E → 𝕜 → F := fun x t => f (HAdd.hAdd x (HSMul.hSMul t v))
    hg : Continuous (Function.uncurry g)
    ⊢ Measurable fun x => lineDeriv 𝕜 f x v
  -/
  exact (measurable_deriv_with_param hg).comp measurable_prod_mk_right
  /-
    🎉 no goals
  -/


theorem stronglyMeasurable_lineDeriv [SecondCountableTopologyEither E F] (hf : Continuous f) :
    StronglyMeasurable (fun x ↦ lineDeriv 𝕜 f x v) := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : OpensMeasurableSpace E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : E → F
    v : E
    inst✝ : SecondCountableTopologyEither E F
    hf : Continuous f
    ⊢ MeasureTheory.StronglyMeasurable fun x => lineDeriv 𝕜 f x v
  -/
  borelize 𝕜
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : OpensMeasurableSpace E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : E → F
    v : E
    inst✝ : SecondCountableTopologyEither E F
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    ⊢ MeasureTheory.StronglyMeasurable fun x => lineDeriv 𝕜 f x v
  -/
  let g : E → 𝕜 → F := fun x t ↦ f (x + t • v)
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : OpensMeasurableSpace E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : E → F
    v : E
    inst✝ : SecondCountableTopologyEither E F
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    g : E → 𝕜 → F := fun x t => f (HAdd.hAdd x (HSMul.hSMul t v))
    ⊢ MeasureTheory.StronglyMeasurable fun x => lineDeriv 𝕜 f x v
  -/
  have hg : Continuous g.uncurry := by fun_prop
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : OpensMeasurableSpace E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : E → F
    v : E
    inst✝ : SecondCountableTopologyEither E F
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    g : E → 𝕜 → F := fun x t => f (HAdd.hAdd x (HSMul.hSMul t v))
    hg : Continuous (Function.uncurry g)
    ⊢ MeasureTheory.StronglyMeasurable fun x => lineDeriv 𝕜 f x v
  -/
  exact (stronglyMeasurable_deriv_with_param hg).comp_measurable measurable_prod_mk_right
  /-
    🎉 no goals
  -/


theorem aemeasurable_lineDeriv [MeasurableSpace F] [BorelSpace F]
    (hf : Continuous f) (μ : Measure E) :
    AEMeasurable (fun x ↦ lineDeriv 𝕜 f x v) μ :=
  (measurable_lineDeriv hf).aemeasurable


theorem aestronglyMeasurable_lineDeriv [SecondCountableTopologyEither E F]
    (hf : Continuous f) (μ : Measure E) :
    AEStronglyMeasurable (fun x ↦ lineDeriv 𝕜 f x v) μ :=
  (stronglyMeasurable_lineDeriv hf).aestronglyMeasurable


theorem measurableSet_lineDifferentiableAt_uncurry (hf : Continuous f) :
    MeasurableSet {p : E × E | LineDifferentiableAt 𝕜 f p.1 p.2} := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : OpensMeasurableSpace E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : E → F
    inst✝ : SecondCountableTopology E
    hf : Continuous f
    ⊢ MeasurableSet (setOf fun p => LineDifferentiableAt 𝕜 f p.1 p.2)
  -/
  borelize 𝕜
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : OpensMeasurableSpace E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : E → F
    inst✝ : SecondCountableTopology E
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    ⊢ MeasurableSet (setOf fun p => LineDifferentiableAt 𝕜 f p.1 p.2)
  -/
  let g : (E × E) → 𝕜 → F := fun p t ↦ f (p.1 + t • p.2)
  have : Continuous g.uncurry :=
    hf.comp <| (continuous_fst.comp continuous_fst).add
    <| continuous_snd.smul (continuous_snd.comp continuous_fst)
  have M_meas : MeasurableSet {q : (E × E) × 𝕜 | DifferentiableAt 𝕜 (g q.1) q.2} :=
    measurableSet_of_differentiableAt_with_param 𝕜 this
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : OpensMeasurableSpace E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : E → F
    inst✝ : SecondCountableTopology E
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    g : Prod E E → 𝕜 → F := fun p t => f (HAdd.hAdd p.1 (HSMul.hSMul t p.2))
    this : Continuous (Function.uncurry g)
    M_meas : MeasurableSet (setOf fun q => DifferentiableAt 𝕜 (g q.1) q.2)
    ⊢ MeasurableSet (setOf fun p => LineDifferentiableAt 𝕜 f p.1 p.2)
  -/
  exact measurable_prod_mk_right M_meas
  /-
    🎉 no goals
  -/


theorem measurable_lineDeriv_uncurry [MeasurableSpace F] [BorelSpace F]
    (hf : Continuous f) : Measurable (fun (p : E × E) ↦ lineDeriv 𝕜 f p.1 p.2) := by
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : MeasurableSpace E
    inst✝⁶ : OpensMeasurableSpace E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : CompleteSpace F
    f : E → F
    inst✝² : SecondCountableTopology E
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    hf : Continuous f
    ⊢ Measurable fun p => lineDeriv 𝕜 f p.1 p.2
  -/
  borelize 𝕜
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : MeasurableSpace E
    inst✝⁶ : OpensMeasurableSpace E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : CompleteSpace F
    f : E → F
    inst✝² : SecondCountableTopology E
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    ⊢ Measurable fun p => lineDeriv 𝕜 f p.1 p.2
  -/
  let g : (E × E) → 𝕜 → F := fun p t ↦ f (p.1 + t • p.2)
  have : Continuous g.uncurry :=
    hf.comp <| (continuous_fst.comp continuous_fst).add
    <| continuous_snd.smul (continuous_snd.comp continuous_fst)
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : MeasurableSpace E
    inst✝⁶ : OpensMeasurableSpace E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : CompleteSpace F
    f : E → F
    inst✝² : SecondCountableTopology E
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    g : Prod E E → 𝕜 → F := fun p t => f (HAdd.hAdd p.1 (HSMul.hSMul t p.2))
    this : Continuous (Function.uncurry g)
    ⊢ Measurable fun p => lineDeriv 𝕜 f p.1 p.2
  -/
  exact (measurable_deriv_with_param this).comp measurable_prod_mk_right
  /-
    🎉 no goals
  -/


theorem stronglyMeasurable_lineDeriv_uncurry (hf : Continuous f) :
    StronglyMeasurable (fun (p : E × E) ↦ lineDeriv 𝕜 f p.1 p.2) := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : OpensMeasurableSpace E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : E → F
    inst✝ : SecondCountableTopology E
    hf : Continuous f
    ⊢ MeasureTheory.StronglyMeasurable fun p => lineDeriv 𝕜 f p.1 p.2
  -/
  borelize 𝕜
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : OpensMeasurableSpace E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : E → F
    inst✝ : SecondCountableTopology E
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    ⊢ MeasureTheory.StronglyMeasurable fun p => lineDeriv 𝕜 f p.1 p.2
  -/
  let g : (E × E) → 𝕜 → F := fun p t ↦ f (p.1 + t • p.2)
  have : Continuous g.uncurry :=
    hf.comp <| (continuous_fst.comp continuous_fst).add
    <| continuous_snd.smul (continuous_snd.comp continuous_fst)
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : LocallyCompactSpace 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : OpensMeasurableSpace E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : E → F
    inst✝ : SecondCountableTopology E
    hf : Continuous f
    this✝¹ : MeasurableSpace 𝕜 := borel 𝕜
    this✝ : BorelSpace 𝕜
    g : Prod E E → 𝕜 → F := fun p t => f (HAdd.hAdd p.1 (HSMul.hSMul t p.2))
    this : Continuous (Function.uncurry g)
    ⊢ MeasureTheory.StronglyMeasurable fun p => lineDeriv 𝕜 f p.1 p.2
  -/
  exact (stronglyMeasurable_deriv_with_param this).comp_measurable measurable_prod_mk_right
  /-
    🎉 no goals
  -/


theorem aemeasurable_lineDeriv_uncurry [MeasurableSpace F] [BorelSpace F]
    (hf : Continuous f) (μ : Measure (E × E)) :
    AEMeasurable (fun (p : E × E) ↦ lineDeriv 𝕜 f p.1 p.2) μ :=
  (measurable_lineDeriv_uncurry hf).aemeasurable


theorem aestronglyMeasurable_lineDeriv_uncurry (hf : Continuous f) (μ : Measure (E × E)) :
    AEStronglyMeasurable (fun (p : E × E) ↦ lineDeriv 𝕜 f p.1 p.2) μ :=
  (stronglyMeasurable_lineDeriv_uncurry hf).aestronglyMeasurable

