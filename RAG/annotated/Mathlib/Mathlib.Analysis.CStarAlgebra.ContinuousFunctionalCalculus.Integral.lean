lemma cfcL_integral (a : A) (f : X → C(spectrum 𝕜 a, 𝕜)) (hf₁ : Integrable f μ)
    (ha : p a := by cfc_tac) :
    ∫ x, cfcL (a := a) ha (f x) ∂μ = cfcL (a := a) ha (∫ x, f x ∂μ) := by
  /-
    X : Type u_1
    𝕜 : Type u_2
    A : Type u_3
    p : A → Prop
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedAlgebra 𝕜 A
    inst✝² : NormedAlgebra Real A
    inst✝¹ : CompleteSpace A
    inst✝ : ContinuousFunctionalCalculus 𝕜 p
    a : A
    f : X → ContinuousMap (↑(spectrum 𝕜 a)) 𝕜
    hf₁ : MeasureTheory.Integrable f μ
    ha : autoParam (p a) _auto✝
    ⊢ Eq (MeasureTheory.integral μ fun x => (cfcL ha) (f x)) ((cfcL ha) (MeasureTh …
  -/
  rw [ContinuousLinearMap.integral_comp_comm _ hf₁]
  /-
    🎉 no goals
  -/


lemma cfcHom_integral (a : A) (f : X → C(spectrum 𝕜 a, 𝕜)) (hf₁ : Integrable f μ)
    (ha : p a := by cfc_tac) :
    ∫ x, cfcHom (a := a) ha (f x) ∂μ = cfcHom (a := a) ha (∫ x, f x ∂μ) :=
  cfcL_integral a f hf₁ ha


open ContinuousMap in
/-- The continuous functional calculus commutes with integration. -/
lemma cfc_integral [TopologicalSpace X] [OpensMeasurableSpace X] (f : X → 𝕜 → 𝕜)
    (bound : X → ℝ) (a : A) [SecondCountableTopologyEither X C(spectrum 𝕜 a, 𝕜)]
    (hf₁ : ∀ x, ContinuousOn (f x) (spectrum 𝕜 a))
    (hf₂ : Continuous (fun x ↦ (⟨_, hf₁ x |>.restrict⟩ : C(spectrum 𝕜 a, 𝕜))))
    (hbound : ∀ x, ∀ z ∈ spectrum 𝕜 a, ‖f x z‖ ≤ ‖bound x‖)
    (hbound_finite_integral : HasFiniteIntegral bound μ) (ha : p a := by cfc_tac) :
    cfc (fun r => ∫ x, f x r ∂μ) a = ∫ x, cfc (f x) a ∂μ := by
  /-
    X : Type u_1
    𝕜 : Type u_2
    A : Type u_3
    p : A → Prop
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝⁸ : NormedRing A
    inst✝⁷ : StarRing A
    inst✝⁶ : NormedAlgebra 𝕜 A
    inst✝⁵ : NormedAlgebra Real A
    inst✝⁴ : CompleteSpace A
    inst✝³ : ContinuousFunctionalCalculus 𝕜 p
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    f : X → 𝕜 → 𝕜
    bound : X → Real
    a : A
    inst✝ : SecondCountableTopologyEither X (ContinuousMap (↑(spectrum 𝕜 a)) 𝕜)
    hf₁ : ∀ (x : X), ContinuousOn (f x) (spectrum 𝕜 a)
    hf₂ : Continuous fun x => { toFun := (spectrum 𝕜 a).restrict (f x), continuous …
    hbound : ∀ (x : X) (z : 𝕜), Membership.mem (spectrum 𝕜 a) z → LE.le (Norm.norm …
    hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun r => MeasureTheory.integral μ fun x => f x r) a) (MeasureTheory …
  -/
  let fc : X → C(spectrum 𝕜 a, 𝕜) := fun x => ⟨_, (hf₁ x).restrict⟩
  have fc_integrable : Integrable fc μ := by
    refine ⟨hf₂.aestronglyMeasurable, ?_⟩
    refine hbound_finite_integral.mono <| .of_forall fun x ↦ ?_
    rw [norm_le _ (norm_nonneg (bound x))]
    exact fun z ↦ hbound x z.1 z.2
  have h_int_fc : (spectrum 𝕜 a).restrict (∫ x, f x · ∂μ) = ∫ x, fc x ∂μ := by
    ext; simp [integral_apply fc_integrable, fc]
  have hcont₂ : ContinuousOn (fun r => ∫ x, f x r ∂μ) (spectrum 𝕜 a) := by
    rw [continuousOn_iff_continuous_restrict]
    convert map_continuous (∫ x, fc x ∂μ)
  rw [integral_congr_ae (.of_forall fun _ ↦ cfc_apply ..), cfc_apply ..,
    cfcHom_integral _ _ fc_integrable]
  /-
    X : Type u_1
    𝕜 : Type u_2
    A : Type u_3
    p : A → Prop
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝⁸ : NormedRing A
    inst✝⁷ : StarRing A
    inst✝⁶ : NormedAlgebra 𝕜 A
    inst✝⁵ : NormedAlgebra Real A
    inst✝⁴ : CompleteSpace A
    inst✝³ : ContinuousFunctionalCalculus 𝕜 p
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    f : X → 𝕜 → 𝕜
    bound : X → Real
    a : A
    inst✝ : SecondCountableTopologyEither X (ContinuousMap (↑(spectrum 𝕜 a)) 𝕜)
    hf₁ : ∀ (x : X), ContinuousOn (f x) (spectrum 𝕜 a)
    hf₂ : Continuous fun x => { toFun := (spectrum 𝕜 a).restrict (f x), continuous …
    hbound : ∀ (x : X) (z : 𝕜), Membership.mem (spectrum 𝕜 a) z → LE.le (Norm.norm …
    hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
    ha : autoParam (p a) _auto✝
    fc : X → ContinuousMap (↑(spectrum 𝕜 a)) 𝕜 := fun x => { toFun := (spectrum 𝕜  …
    fc_integrable : MeasureTheory.Integrable fc μ
    h_int_fc : Eq ((spectrum 𝕜 a).restrict fun x => MeasureTheory.integral μ fun x …
    hcont₂ : ContinuousOn (fun r => MeasureTheory.integral μ fun x => f x r) (spec …
    ⊢ Eq ((cfcHom ha) { toFun := (spectrum 𝕜 a).restrict fun r => MeasureTheory.in …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- The continuous functional calculus commutes with integration. -/
lemma cfc_integral' [TopologicalSpace X] [OpensMeasurableSpace X] (f : X → 𝕜 → 𝕜)
    (bound : X → ℝ) (a : A) [SecondCountableTopologyEither X C(spectrum 𝕜 a, 𝕜)]
    (hf : Continuous (fun x => (spectrum 𝕜 a).restrict (f x)).uncurry)
    (hbound : ∀ x, ∀ z ∈ spectrum 𝕜 a, ‖f x z‖ ≤ ‖bound x‖)
    (hbound_finite_integral : HasFiniteIntegral bound μ) (ha : p a := by cfc_tac) :
    cfc (fun r => ∫ x, f x r ∂μ) a = ∫ x, cfc (f x) a ∂μ := by
  /-
    X : Type u_1
    𝕜 : Type u_2
    A : Type u_3
    p : A → Prop
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝⁸ : NormedRing A
    inst✝⁷ : StarRing A
    inst✝⁶ : NormedAlgebra 𝕜 A
    inst✝⁵ : NormedAlgebra Real A
    inst✝⁴ : CompleteSpace A
    inst✝³ : ContinuousFunctionalCalculus 𝕜 p
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    f : X → 𝕜 → 𝕜
    bound : X → Real
    a : A
    inst✝ : SecondCountableTopologyEither X (ContinuousMap (↑(spectrum 𝕜 a)) 𝕜)
    hf : Continuous (Function.uncurry fun x => (spectrum 𝕜 a).restrict (f x))
    hbound : ∀ (x : X) (z : 𝕜), Membership.mem (spectrum 𝕜 a) z → LE.le (Norm.norm …
    hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun r => MeasureTheory.integral μ fun x => f x r) a) (MeasureTheory …
  -/
  refine cfc_integral f bound a ?_ ?_ hbound hbound_finite_integral
    /-
      case refine_1
      X : Type u_1
      𝕜 : Type u_2
      A : Type u_3
      p : A → Prop
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝⁸ : NormedRing A
      inst✝⁷ : StarRing A
      inst✝⁶ : NormedAlgebra 𝕜 A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : CompleteSpace A
      inst✝³ : ContinuousFunctionalCalculus 𝕜 p
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      f : X → 𝕜 → 𝕜
      bound : X → Real
      a : A
      inst✝ : SecondCountableTopologyEither X (ContinuousMap (↑(spectrum 𝕜 a)) 𝕜)
      hf : Continuous (Function.uncurry fun x => (spectrum 𝕜 a).restrict (f x))
      hbound : ∀ (x : X) (z : 𝕜), Membership.mem (spectrum 𝕜 a) z → LE.le (Norm.norm …
      hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
      ha : autoParam (p a) _auto✝
      ⊢ ∀ (x : X), ContinuousOn (f x) (spectrum 𝕜 a)
    -/
  · exact (continuousOn_iff_continuous_restrict.mpr <| hf.uncurry_left ·)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      𝕜 : Type u_2
      A : Type u_3
      p : A → Prop
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝⁸ : NormedRing A
      inst✝⁷ : StarRing A
      inst✝⁶ : NormedAlgebra 𝕜 A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : CompleteSpace A
      inst✝³ : ContinuousFunctionalCalculus 𝕜 p
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      f : X → 𝕜 → 𝕜
      bound : X → Real
      a : A
      inst✝ : SecondCountableTopologyEither X (ContinuousMap (↑(spectrum 𝕜 a)) 𝕜)
      hf : Continuous (Function.uncurry fun x => (spectrum 𝕜 a).restrict (f x))
      hbound : ∀ (x : X) (z : 𝕜), Membership.mem (spectrum 𝕜 a) z → LE.le (Norm.norm …
      hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
      ha : autoParam (p a) _auto✝
      ⊢ Continuous fun x => { toFun := (spectrum 𝕜 a).restrict (f x), continuous_toF …
    -/
  · exact ContinuousMap.curry ⟨_, hf⟩ |>.continuous
    /-
      🎉 no goals
    -/


lemma cfcₙL_integral (a : A) (f : X → C(quasispectrum 𝕜 a, 𝕜)₀) (hf₁ : Integrable f μ)
    (ha : p a := by cfc_tac) :
    ∫ x, cfcₙL (a := a) ha (f x) ∂μ = cfcₙL (a := a) ha (∫ x, f x ∂μ) := by
  /-
    X : Type u_1
    𝕜 : Type u_2
    A : Type u_3
    p : A → Prop
    inst✝⁹ : RCLike 𝕜
    inst✝⁸ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝⁷ : NonUnitalNormedRing A
    inst✝⁶ : StarRing A
    inst✝⁵ : CompleteSpace A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : NormedSpace Real A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : NonUnitalContinuousFunctionalCalculus 𝕜 p
    a : A
    f : X → ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
    hf₁ : MeasureTheory.Integrable f μ
    ha : autoParam (p a) _auto✝
    ⊢ Eq (MeasureTheory.integral μ fun x => (cfcₙL ha) (f x)) ((cfcₙL ha) (Measure …
  -/
  rw [ContinuousLinearMap.integral_comp_comm _ hf₁]
  /-
    🎉 no goals
  -/


lemma cfcₙHom_integral (a : A) (f : X → C(quasispectrum 𝕜 a, 𝕜)₀) (hf₁ : Integrable f μ)
    (ha : p a := by cfc_tac) :
    ∫ x, cfcₙHom (a := a) ha (f x) ∂μ = cfcₙHom (a := a) ha (∫ x, f x ∂μ) :=
  cfcₙL_integral a f hf₁ ha


open ContinuousMapZero in
/-- The non-unital continuous functional calculus commutes with integration. -/
lemma cfcₙ_integral [TopologicalSpace X] [OpensMeasurableSpace X] (f : X → 𝕜 → 𝕜)
    (bound : X → ℝ) (a : A) [SecondCountableTopologyEither X C(quasispectrum 𝕜 a, 𝕜)₀]
    (hf₁ : ∀ x, ContinuousOn (f x) (quasispectrum 𝕜 a))
    (hf₂ : ∀ x, f x 0 = 0)
    (hf₃ : Continuous (fun x ↦ (⟨⟨_, hf₁ x |>.restrict⟩, hf₂ x⟩ : C(quasispectrum 𝕜 a, 𝕜)₀)))
    (hbound : ∀ x, ∀ z ∈ quasispectrum 𝕜 a, ‖f x z‖ ≤ ‖bound x‖)
    (hbound_finite_integral : HasFiniteIntegral bound μ) (ha : p a := by cfc_tac) :
    cfcₙ (fun r => ∫ x, f x r ∂μ) a = ∫ x, cfcₙ (f x) a ∂μ := by
  /-
    X : Type u_1
    𝕜 : Type u_2
    A : Type u_3
    p : A → Prop
    inst✝¹² : RCLike 𝕜
    inst✝¹¹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝¹⁰ : NonUnitalNormedRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : CompleteSpace A
    inst✝⁷ : NormedSpace 𝕜 A
    inst✝⁶ : NormedSpace Real A
    inst✝⁵ : IsScalarTower 𝕜 A A
    inst✝⁴ : SMulCommClass 𝕜 A A
    inst✝³ : NonUnitalContinuousFunctionalCalculus 𝕜 p
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    f : X → 𝕜 → 𝕜
    bound : X → Real
    a : A
    inst✝ : SecondCountableTopologyEither X (ContinuousMapZero (↑(quasispectrum 𝕜  …
    hf₁ : ∀ (x : X), ContinuousOn (f x) (quasispectrum 𝕜 a)
    hf₂ : ∀ (x : X), Eq (f x 0) 0
    hf₃ : Continuous fun x => { toFun := (quasispectrum 𝕜 a).restrict (f x), conti …
    hbound : ∀ (x : X) (z : 𝕜), Membership.mem (quasispectrum 𝕜 a) z → LE.le (Norm …
    hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfcₙ (fun r => MeasureTheory.integral μ fun x => f x r) a) (MeasureTheor …
  -/
  let fc : X → C(quasispectrum 𝕜 a, 𝕜)₀ := fun x => ⟨⟨_, (hf₁ x).restrict⟩, hf₂ x⟩
  have fc_integrable : Integrable fc μ := by
    refine ⟨hf₃.aestronglyMeasurable, ?_⟩
    refine hbound_finite_integral.mono <| .of_forall fun x ↦ ?_
    change ‖(fc x : C(quasispectrum  𝕜 a, 𝕜))‖ ≤ ‖bound x‖
    rw [ContinuousMap.norm_le _ (norm_nonneg (bound x))]
    exact fun z ↦ hbound x z.1 z.2
  have h_int_fc : (quasispectrum 𝕜 a).restrict (∫ x, f x · ∂μ) = ∫ x, fc x ∂μ := by
    ext; simp [integral_apply fc_integrable, fc]
  have hcont₂ : ContinuousOn (fun r => ∫ x, f x r ∂μ) (quasispectrum 𝕜 a) := by
    rw [continuousOn_iff_continuous_restrict]
    convert map_continuous (∫ x, fc x ∂μ)
  rw [integral_congr_ae (.of_forall fun _ ↦ cfcₙ_apply ..), cfcₙ_apply ..,
    cfcₙHom_integral _ _ fc_integrable]
  /-
    X : Type u_1
    𝕜 : Type u_2
    A : Type u_3
    p : A → Prop
    inst✝¹² : RCLike 𝕜
    inst✝¹¹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝¹⁰ : NonUnitalNormedRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : CompleteSpace A
    inst✝⁷ : NormedSpace 𝕜 A
    inst✝⁶ : NormedSpace Real A
    inst✝⁵ : IsScalarTower 𝕜 A A
    inst✝⁴ : SMulCommClass 𝕜 A A
    inst✝³ : NonUnitalContinuousFunctionalCalculus 𝕜 p
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    f : X → 𝕜 → 𝕜
    bound : X → Real
    a : A
    inst✝ : SecondCountableTopologyEither X (ContinuousMapZero (↑(quasispectrum 𝕜  …
    hf₁ : ∀ (x : X), ContinuousOn (f x) (quasispectrum 𝕜 a)
    hf₂ : ∀ (x : X), Eq (f x 0) 0
    hf₃ : Continuous fun x => { toFun := (quasispectrum 𝕜 a).restrict (f x), conti …
    hbound : ∀ (x : X) (z : 𝕜), Membership.mem (quasispectrum 𝕜 a) z → LE.le (Norm …
    hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
    ha : autoParam (p a) _auto✝
    fc : X → ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜 := fun x => { toFun := (qu …
    fc_integrable : MeasureTheory.Integrable fc μ
    h_int_fc : Eq ((quasispectrum 𝕜 a).restrict fun x => MeasureTheory.integral μ  …
    hcont₂ : ContinuousOn (fun r => MeasureTheory.integral μ fun x => f x r) (quas …
    ⊢ Eq ((cfcₙHom ha) { toFun := (quasispectrum 𝕜 a).restrict fun r => MeasureThe …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- The non-unital continuous functional calculus commutes with integration. -/
lemma cfcₙ_integral' [TopologicalSpace X] [OpensMeasurableSpace X] (f : X → 𝕜 → 𝕜)
    (bound : X → ℝ) (a : A) [SecondCountableTopologyEither X C(quasispectrum 𝕜 a, 𝕜)₀]
    (hf : Continuous (fun x => (quasispectrum 𝕜 a).restrict (f x)).uncurry)
    (hf₂ : ∀ x, f x 0 = 0)
    (hbound : ∀ x, ∀ z ∈ quasispectrum 𝕜 a, ‖f x z‖ ≤ ‖bound x‖)
    (hbound_finite_integral : HasFiniteIntegral bound μ) (ha : p a := by cfc_tac) :
    cfcₙ (fun r => ∫ x, f x r ∂μ) a = ∫ x, cfcₙ (f x) a ∂μ := by
  /-
    X : Type u_1
    𝕜 : Type u_2
    A : Type u_3
    p : A → Prop
    inst✝¹² : RCLike 𝕜
    inst✝¹¹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝¹⁰ : NonUnitalNormedRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : CompleteSpace A
    inst✝⁷ : NormedSpace 𝕜 A
    inst✝⁶ : NormedSpace Real A
    inst✝⁵ : IsScalarTower 𝕜 A A
    inst✝⁴ : SMulCommClass 𝕜 A A
    inst✝³ : NonUnitalContinuousFunctionalCalculus 𝕜 p
    inst✝² : TopologicalSpace X
    inst✝¹ : OpensMeasurableSpace X
    f : X → 𝕜 → 𝕜
    bound : X → Real
    a : A
    inst✝ : SecondCountableTopologyEither X (ContinuousMapZero (↑(quasispectrum 𝕜  …
    hf : Continuous (Function.uncurry fun x => (quasispectrum 𝕜 a).restrict (f x))
    hf₂ : ∀ (x : X), Eq (f x 0) 0
    hbound : ∀ (x : X) (z : 𝕜), Membership.mem (quasispectrum 𝕜 a) z → LE.le (Norm …
    hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfcₙ (fun r => MeasureTheory.integral μ fun x => f x r) a) (MeasureTheor …
  -/
  refine cfcₙ_integral f bound a ?_ hf₂ ?_ hbound hbound_finite_integral
    /-
      case refine_1
      X : Type u_1
      𝕜 : Type u_2
      A : Type u_3
      p : A → Prop
      inst✝¹² : RCLike 𝕜
      inst✝¹¹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹⁰ : NonUnitalNormedRing A
      inst✝⁹ : StarRing A
      inst✝⁸ : CompleteSpace A
      inst✝⁷ : NormedSpace 𝕜 A
      inst✝⁶ : NormedSpace Real A
      inst✝⁵ : IsScalarTower 𝕜 A A
      inst✝⁴ : SMulCommClass 𝕜 A A
      inst✝³ : NonUnitalContinuousFunctionalCalculus 𝕜 p
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      f : X → 𝕜 → 𝕜
      bound : X → Real
      a : A
      inst✝ : SecondCountableTopologyEither X (ContinuousMapZero (↑(quasispectrum 𝕜  …
      hf : Continuous (Function.uncurry fun x => (quasispectrum 𝕜 a).restrict (f x))
      hf₂ : ∀ (x : X), Eq (f x 0) 0
      hbound : ∀ (x : X) (z : 𝕜), Membership.mem (quasispectrum 𝕜 a) z → LE.le (Norm …
      hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
      ha : autoParam (p a) _auto✝
      ⊢ ∀ (x : X), ContinuousOn (f x) (quasispectrum 𝕜 a)
    -/
  · exact (continuousOn_iff_continuous_restrict.mpr <| hf.uncurry_left ·)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      𝕜 : Type u_2
      A : Type u_3
      p : A → Prop
      inst✝¹² : RCLike 𝕜
      inst✝¹¹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹⁰ : NonUnitalNormedRing A
      inst✝⁹ : StarRing A
      inst✝⁸ : CompleteSpace A
      inst✝⁷ : NormedSpace 𝕜 A
      inst✝⁶ : NormedSpace Real A
      inst✝⁵ : IsScalarTower 𝕜 A A
      inst✝⁴ : SMulCommClass 𝕜 A A
      inst✝³ : NonUnitalContinuousFunctionalCalculus 𝕜 p
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      f : X → 𝕜 → 𝕜
      bound : X → Real
      a : A
      inst✝ : SecondCountableTopologyEither X (ContinuousMapZero (↑(quasispectrum 𝕜  …
      hf : Continuous (Function.uncurry fun x => (quasispectrum 𝕜 a).restrict (f x))
      hf₂ : ∀ (x : X), Eq (f x 0) 0
      hbound : ∀ (x : X) (z : 𝕜), Membership.mem (quasispectrum 𝕜 a) z → LE.le (Norm …
      hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
      ha : autoParam (p a) _auto✝
      ⊢ Continuous fun x => { toFun := (quasispectrum 𝕜 a).restrict (f x), continuou …
    -/
  · let g := ((↑) : C(quasispectrum 𝕜 a, 𝕜)₀ → C(quasispectrum 𝕜 a, 𝕜))
    /-
      case refine_2
      X : Type u_1
      𝕜 : Type u_2
      A : Type u_3
      p : A → Prop
      inst✝¹² : RCLike 𝕜
      inst✝¹¹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹⁰ : NonUnitalNormedRing A
      inst✝⁹ : StarRing A
      inst✝⁸ : CompleteSpace A
      inst✝⁷ : NormedSpace 𝕜 A
      inst✝⁶ : NormedSpace Real A
      inst✝⁵ : IsScalarTower 𝕜 A A
      inst✝⁴ : SMulCommClass 𝕜 A A
      inst✝³ : NonUnitalContinuousFunctionalCalculus 𝕜 p
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      f : X → 𝕜 → 𝕜
      bound : X → Real
      a : A
      inst✝ : SecondCountableTopologyEither X (ContinuousMapZero (↑(quasispectrum 𝕜  …
      hf : Continuous (Function.uncurry fun x => (quasispectrum 𝕜 a).restrict (f x))
      hf₂ : ∀ (x : X), Eq (f x 0) 0
      hbound : ∀ (x : X) (z : 𝕜), Membership.mem (quasispectrum 𝕜 a) z → LE.le (Norm …
      hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
      ha : autoParam (p a) _auto✝
      g : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜 → ContinuousMap (↑(quasispectru …
      ⊢ Continuous fun x => { toFun := (quasispectrum 𝕜 a).restrict (f x), continuou …
    -/
    refine ((isInducing_iff g).mpr rfl).continuous_iff.mpr ?_
    /-
      case refine_2
      X : Type u_1
      𝕜 : Type u_2
      A : Type u_3
      p : A → Prop
      inst✝¹² : RCLike 𝕜
      inst✝¹¹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹⁰ : NonUnitalNormedRing A
      inst✝⁹ : StarRing A
      inst✝⁸ : CompleteSpace A
      inst✝⁷ : NormedSpace 𝕜 A
      inst✝⁶ : NormedSpace Real A
      inst✝⁵ : IsScalarTower 𝕜 A A
      inst✝⁴ : SMulCommClass 𝕜 A A
      inst✝³ : NonUnitalContinuousFunctionalCalculus 𝕜 p
      inst✝² : TopologicalSpace X
      inst✝¹ : OpensMeasurableSpace X
      f : X → 𝕜 → 𝕜
      bound : X → Real
      a : A
      inst✝ : SecondCountableTopologyEither X (ContinuousMapZero (↑(quasispectrum 𝕜  …
      hf : Continuous (Function.uncurry fun x => (quasispectrum 𝕜 a).restrict (f x))
      hf₂ : ∀ (x : X), Eq (f x 0) 0
      hbound : ∀ (x : X) (z : 𝕜), Membership.mem (quasispectrum 𝕜 a) z → LE.le (Norm …
      hbound_finite_integral : MeasureTheory.HasFiniteIntegral bound μ
      ha : autoParam (p a) _auto✝
      g : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜 → ContinuousMap (↑(quasispectru …
      ⊢ Continuous (Function.comp g fun x => { toFun := (quasispectrum 𝕜 a).restrict …
    -/
    exact ContinuousMap.curry ⟨_, hf⟩ |>.continuous
    /-
      🎉 no goals
    -/


