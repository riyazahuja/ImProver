/-- A signed measure `s` is said to `HaveLebesgueDecomposition` with respect to a measure `μ`
if the positive part and the negative part of `s` both `HaveLebesgueDecomposition` with
respect to `μ`. -/
class HaveLebesgueDecomposition (s : SignedMeasure α) (μ : Measure α) : Prop where
  posPart : s.toJordanDecomposition.posPart.HaveLebesgueDecomposition μ
  negPart : s.toJordanDecomposition.negPart.HaveLebesgueDecomposition μ


theorem not_haveLebesgueDecomposition_iff (s : SignedMeasure α) (μ : Measure α) :
    ¬s.HaveLebesgueDecomposition μ ↔
      ¬s.toJordanDecomposition.posPart.HaveLebesgueDecomposition μ ∨
        ¬s.toJordanDecomposition.negPart.HaveLebesgueDecomposition μ :=
  ⟨fun h => not_or_of_imp fun hp hn => h ⟨hp, hn⟩, fun h hl => (not_and_or.2 h) ⟨hl.1, hl.2⟩⟩

-- `inferInstance` directly does not work
-- see Note [lower instance priority]

instance (priority := 100) haveLebesgueDecomposition_of_sigmaFinite (s : SignedMeasure α)
    (μ : Measure α) [SigmaFinite μ] : s.HaveLebesgueDecomposition μ where
  posPart := inferInstance
  negPart := inferInstance


instance haveLebesgueDecomposition_neg (s : SignedMeasure α) (μ : Measure α)
    [s.HaveLebesgueDecomposition μ] : (-s).HaveLebesgueDecomposition μ where
  posPart := by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      ⊢ (Neg.neg s).toJordanDecomposition.posPart.HaveLebesgueDecomposition μ
    -/
    rw [toJordanDecomposition_neg, JordanDecomposition.neg_posPart]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      ⊢ s.toJordanDecomposition.negPart.HaveLebesgueDecomposition μ
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  negPart := by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      ⊢ (Neg.neg s).toJordanDecomposition.negPart.HaveLebesgueDecomposition μ
    -/
    rw [toJordanDecomposition_neg, JordanDecomposition.neg_negPart]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      ⊢ s.toJordanDecomposition.posPart.HaveLebesgueDecomposition μ
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance haveLebesgueDecomposition_smul (s : SignedMeasure α) (μ : Measure α)
    [s.HaveLebesgueDecomposition μ] (r : ℝ≥0) : (r • s).HaveLebesgueDecomposition μ where
  posPart := by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      r : NNReal
      ⊢ (HSMul.hSMul r s).toJordanDecomposition.posPart.HaveLebesgueDecomposition μ
    -/
    rw [toJordanDecomposition_smul, JordanDecomposition.smul_posPart]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      r : NNReal
      ⊢ (HSMul.hSMul r s.toJordanDecomposition.posPart).HaveLebesgueDecomposition μ
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  negPart := by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      r : NNReal
      ⊢ (HSMul.hSMul r s).toJordanDecomposition.negPart.HaveLebesgueDecomposition μ
    -/
    rw [toJordanDecomposition_smul, JordanDecomposition.smul_negPart]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      r : NNReal
      ⊢ (HSMul.hSMul r s.toJordanDecomposition.negPart).HaveLebesgueDecomposition μ
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance haveLebesgueDecomposition_smul_real (s : SignedMeasure α) (μ : Measure α)
    [s.HaveLebesgueDecomposition μ] (r : ℝ) : (r • s).HaveLebesgueDecomposition μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    inst✝ : s.HaveLebesgueDecomposition μ
    r : Real
    ⊢ (HSMul.hSMul r s).HaveLebesgueDecomposition μ
  -/
  by_cases hr : 0 ≤ r
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      r : Real
      hr : LE.le 0 r
      ⊢ (HSMul.hSMul r s).HaveLebesgueDecomposition μ
    -/
  · lift r to ℝ≥0 using hr
    /-
      case pos.intro
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      r : NNReal
      ⊢ (HSMul.hSMul (↑r) s).HaveLebesgueDecomposition μ
    -/
    exact s.haveLebesgueDecomposition_smul μ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : s.HaveLebesgueDecomposition μ
      r : Real
      hr : Not (LE.le 0 r)
      ⊢ (HSMul.hSMul r s).HaveLebesgueDecomposition μ
    -/
  · rw [not_le] at hr
    refine
      { posPart := by
          rw [toJordanDecomposition_smul_real, JordanDecomposition.real_smul_posPart_neg _ _ hr]
          infer_instance
        negPart := by
          rw [toJordanDecomposition_smul_real, JordanDecomposition.real_smul_negPart_neg _ _ hr]
          infer_instance }


/-- Given a signed measure `s` and a measure `μ`, `s.singularPart μ` is the signed measure
such that `s.singularPart μ + μ.withDensityᵥ (s.rnDeriv μ) = s` and
`s.singularPart μ` is mutually singular with respect to `μ`. -/
def singularPart (s : SignedMeasure α) (μ : Measure α) : SignedMeasure α :=
  (s.toJordanDecomposition.posPart.singularPart μ).toSignedMeasure -
    (s.toJordanDecomposition.negPart.singularPart μ).toSignedMeasure


theorem singularPart_mutuallySingular (s : SignedMeasure α) (μ : Measure α) :
    s.toJordanDecomposition.posPart.singularPart μ ⟂ₘ
      s.toJordanDecomposition.negPart.singularPart μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    ⊢ (s.toJordanDecomposition.posPart.singularPart μ).MutuallySingular (s.toJorda …
  -/
  by_cases hl : s.HaveLebesgueDecomposition μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      hl : s.HaveLebesgueDecomposition μ
      ⊢ (s.toJordanDecomposition.posPart.singularPart μ).MutuallySingular (s.toJorda …
    -/
  · obtain ⟨i, hi, hpos, hneg⟩ := s.toJordanDecomposition.mutuallySingular
    /-
      case pos.intro.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      hl : s.HaveLebesgueDecomposition μ
      i : Set α
      hi : MeasurableSet i
      hpos : Eq (s.toJordanDecomposition.posPart i) 0
      hneg : Eq (s.toJordanDecomposition.negPart (HasCompl.compl i)) 0
      ⊢ (s.toJordanDecomposition.posPart.singularPart μ).MutuallySingular (s.toJorda …
    -/
    rw [s.toJordanDecomposition.posPart.haveLebesgueDecomposition_add μ] at hpos
    /-
      case pos.intro.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      hl : s.HaveLebesgueDecomposition μ
      i : Set α
      hi : MeasurableSet i
      hpos : Eq ((HAdd.hAdd (s.toJordanDecomposition.posPart.singularPart μ) (μ.with …
      hneg : Eq (s.toJordanDecomposition.negPart (HasCompl.compl i)) 0
      ⊢ (s.toJordanDecomposition.posPart.singularPart μ).MutuallySingular (s.toJorda …
    -/
    rw [s.toJordanDecomposition.negPart.haveLebesgueDecomposition_add μ] at hneg
    /-
      case pos.intro.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      hl : s.HaveLebesgueDecomposition μ
      i : Set α
      hi : MeasurableSet i
      hpos : Eq ((HAdd.hAdd (s.toJordanDecomposition.posPart.singularPart μ) (μ.with …
      hneg : Eq ((HAdd.hAdd (s.toJordanDecomposition.negPart.singularPart μ) (μ.with …
      ⊢ (s.toJordanDecomposition.posPart.singularPart μ).MutuallySingular (s.toJorda …
    -/
    rw [add_apply, add_eq_zero] at hpos hneg
    /-
      case pos.intro.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      hl : s.HaveLebesgueDecomposition μ
      i : Set α
      hi : MeasurableSet i
      hpos : And (Eq ((s.toJordanDecomposition.posPart.singularPart μ) i) 0) (Eq ((μ …
      hneg : And (Eq ((s.toJordanDecomposition.negPart.singularPart μ) (HasCompl.com …
      ⊢ (s.toJordanDecomposition.posPart.singularPart μ).MutuallySingular (s.toJorda …
    -/
    exact ⟨i, hi, hpos.1, hneg.1⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      hl : Not (s.HaveLebesgueDecomposition μ)
      ⊢ (s.toJordanDecomposition.posPart.singularPart μ).MutuallySingular (s.toJorda …
    -/
  · rw [not_haveLebesgueDecomposition_iff] at hl
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      hl : Or (Not (s.toJordanDecomposition.posPart.HaveLebesgueDecomposition μ)) (N …
      ⊢ (s.toJordanDecomposition.posPart.singularPart μ).MutuallySingular (s.toJorda …
    -/
    cases' hl with hp hn
      /-
        case neg.inl
        α : Type u_1
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.Measure α
        hp : Not (s.toJordanDecomposition.posPart.HaveLebesgueDecomposition μ)
        ⊢ (s.toJordanDecomposition.posPart.singularPart μ).MutuallySingular (s.toJorda …
      -/
    · rw [Measure.singularPart, dif_neg hp]
      /-
        case neg.inl
        α : Type u_1
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.Measure α
        hp : Not (s.toJordanDecomposition.posPart.HaveLebesgueDecomposition μ)
        ⊢ MeasureTheory.Measure.MutuallySingular 0 (s.toJordanDecomposition.negPart.si …
      -/
      exact MutuallySingular.zero_left
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        α : Type u_1
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.Measure α
        hn : Not (s.toJordanDecomposition.negPart.HaveLebesgueDecomposition μ)
        ⊢ (s.toJordanDecomposition.posPart.singularPart μ).MutuallySingular (s.toJorda …
      -/
    · rw [Measure.singularPart, Measure.singularPart, dif_neg hn]
      /-
        case neg.inr
        α : Type u_1
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.Measure α
        hn : Not (s.toJordanDecomposition.negPart.HaveLebesgueDecomposition μ)
        ⊢ (dite (s.toJordanDecomposition.posPart.HaveLebesgueDecomposition μ) (fun h = …
      -/
      exact MutuallySingular.zero_right
      /-
        🎉 no goals
      -/


theorem singularPart_totalVariation (s : SignedMeasure α) (μ : Measure α) :
    (s.singularPart μ).totalVariation =
      s.toJordanDecomposition.posPart.singularPart μ +
        s.toJordanDecomposition.negPart.singularPart μ := by
  have :
    (s.singularPart μ).toJordanDecomposition =
      ⟨s.toJordanDecomposition.posPart.singularPart μ,
        s.toJordanDecomposition.negPart.singularPart μ, singularPart_mutuallySingular s μ⟩ := by
    refine JordanDecomposition.toSignedMeasure_injective ?_
    rw [toSignedMeasure_toJordanDecomposition, singularPart, JordanDecomposition.toSignedMeasure]
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    this : Eq (s.singularPart μ).toJordanDecomposition (MeasureTheory.JordanDecomp …
    ⊢ Eq (s.singularPart μ).totalVariation (HAdd.hAdd (s.toJordanDecomposition.pos …
  -/
  rw [totalVariation, this]
  /-
    🎉 no goals
  -/


nonrec theorem mutuallySingular_singularPart (s : SignedMeasure α) (μ : Measure α) :
    singularPart s μ ⟂ᵥ μ.toENNRealVectorMeasure := by
  rw [mutuallySingular_ennreal_iff, singularPart_totalVariation,
    VectorMeasure.ennrealToMeasure_toENNRealVectorMeasure]
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    ⊢ (HAdd.hAdd (s.toJordanDecomposition.posPart.singularPart μ) (s.toJordanDecom …
  -/
  exact (mutuallySingular_singularPart _ _).add_left (mutuallySingular_singularPart _ _)
  /-
    🎉 no goals
  -/


/-- The Radon-Nikodym derivative between a signed measure and a positive measure.

`rnDeriv s μ` satisfies `μ.withDensityᵥ (s.rnDeriv μ) = s`
if and only if `s` is absolutely continuous with respect to `μ` and this fact is known as
`MeasureTheory.SignedMeasure.absolutelyContinuous_iff_withDensity_rnDeriv_eq`
and can be found in `MeasureTheory.Decomposition.RadonNikodym`. -/
def rnDeriv (s : SignedMeasure α) (μ : Measure α) : α → ℝ := fun x =>
  (s.toJordanDecomposition.posPart.rnDeriv μ x).toReal -
    (s.toJordanDecomposition.negPart.rnDeriv μ x).toReal

-- The generated equation theorem is the form of `rnDeriv s μ x = ...`.

theorem rnDeriv_def (s : SignedMeasure α) (μ : Measure α) : rnDeriv s μ = fun x =>
    (s.toJordanDecomposition.posPart.rnDeriv μ x).toReal -
      (s.toJordanDecomposition.negPart.rnDeriv μ x).toReal :=
  rfl


@[measurability]
theorem measurable_rnDeriv (s : SignedMeasure α) (μ : Measure α) : Measurable (rnDeriv s μ) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    ⊢ Measurable (s.rnDeriv μ)
  -/
  rw [rnDeriv_def]
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    ⊢ Measurable fun x => HSub.hSub (s.toJordanDecomposition.posPart.rnDeriv μ x). …
  -/
  fun_prop
  /-
    🎉 no goals
  -/


theorem integrable_rnDeriv (s : SignedMeasure α) (μ : Measure α) : Integrable (rnDeriv s μ) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    ⊢ MeasureTheory.Integrable (s.rnDeriv μ) μ
  -/
  refine Integrable.sub ?_ ?_ <;>
      /-
        case refine_1
        α : Type u_1
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.Measure α
        ⊢ MeasureTheory.Integrable (fun x => (s.toJordanDecomposition.posPart.rnDeriv  …
      -/
        /-
          case refine_1.left
          α : Type u_1
          m : MeasurableSpace α
          s : MeasureTheory.SignedMeasure α
          μ : MeasureTheory.Measure α
          ⊢ MeasureTheory.AEStronglyMeasurable (fun x => (s.toJordanDecomposition.posPar …
        -/
        /-
          case refine_1.left.hf
          α : Type u_1
          m : MeasurableSpace α
          s : MeasureTheory.SignedMeasure α
          μ : MeasureTheory.Measure α
          ⊢ Measurable fun x => (s.toJordanDecomposition.posPart.rnDeriv μ x).toReal
        -/
        /-
          🎉 no goals
        -/
      /-
        case refine_1.right
        α : Type u_1
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.Measure α
        ⊢ MeasureTheory.HasFiniteIntegral (fun x => (s.toJordanDecomposition.posPart.r …
      -/
      /-
        🎉 no goals
      -/
        /-
          case refine_2.left.hf
          α : Type u_1
          m : MeasurableSpace α
          s : MeasureTheory.SignedMeasure α
          μ : MeasureTheory.Measure α
          ⊢ Measurable fun x => (s.toJordanDecomposition.negPart.rnDeriv μ x).toReal
        -/
        fun_prop
        /-
          🎉 no goals
        -/
      /-
        case refine_2.right
        α : Type u_1
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.Measure α
        ⊢ MeasureTheory.HasFiniteIntegral (fun x => (s.toJordanDecomposition.negPart.r …
      -/
      exact hasFiniteIntegral_toReal_of_lintegral_ne_top (lintegral_rnDeriv_lt_top _ μ).ne
      /-
        🎉 no goals
      -/


/-- **The Lebesgue Decomposition theorem between a signed measure and a measure**:
Given a signed measure `s` and a σ-finite measure `μ`, there exist a signed measure `t` and a
measurable and integrable function `f`, such that `t` is mutually singular with respect to `μ`
and `s = t + μ.withDensityᵥ f`. In this case `t = s.singularPart μ` and
`f = s.rnDeriv μ`. -/
theorem singularPart_add_withDensity_rnDeriv_eq [s.HaveLebesgueDecomposition μ] :
    s.singularPart μ + μ.withDensityᵥ (s.rnDeriv μ) = s := by
  conv_rhs =>
    rw [← toSignedMeasure_toJordanDecomposition s, JordanDecomposition.toSignedMeasure]
  rw [singularPart, rnDeriv_def,
    withDensityᵥ_sub' (integrable_toReal_of_lintegral_ne_top _ _)
      (integrable_toReal_of_lintegral_ne_top _ _),
    withDensityᵥ_toReal, withDensityᵥ_toReal, sub_eq_add_neg, sub_eq_add_neg,
    add_comm (s.toJordanDecomposition.posPart.singularPart μ).toSignedMeasure, ← add_assoc,
    add_assoc (-(s.toJordanDecomposition.negPart.singularPart μ).toSignedMeasure),
    ← toSignedMeasure_add, add_comm, ← add_assoc, ← neg_add, ← toSignedMeasure_add, add_comm,
    ← sub_eq_add_neg]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : MeasureTheory.SignedMeasure α
      inst✝ : s.HaveLebesgueDecomposition μ
      ⊢ Eq (HSub.hSub (HAdd.hAdd (s.toJordanDecomposition.posPart.singularPart μ) (μ …
    -/
  · convert rfl
    -- `convert rfl` much faster than `congr`
      /-
        case h.e'_3.h.e'_5.h.e'_3
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : MeasureTheory.SignedMeasure α
        inst✝ : s.HaveLebesgueDecomposition μ
        ⊢ Eq s.toJordanDecomposition.posPart (HAdd.hAdd (s.toJordanDecomposition.posPa …
      -/
    · exact s.toJordanDecomposition.posPart.haveLebesgueDecomposition_add μ
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.h.e'_6.h.e'_3
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : MeasureTheory.SignedMeasure α
        inst✝ : s.HaveLebesgueDecomposition μ
        ⊢ Eq s.toJordanDecomposition.negPart (HAdd.hAdd (μ.withDensity (s.toJordanDeco …
      -/
    · rw [add_comm]
      /-
        case h.e'_3.h.e'_6.h.e'_3
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : MeasureTheory.SignedMeasure α
        inst✝ : s.HaveLebesgueDecomposition μ
        ⊢ Eq s.toJordanDecomposition.negPart (HAdd.hAdd (s.toJordanDecomposition.negPa …
      -/
      exact s.toJordanDecomposition.negPart.haveLebesgueDecomposition_add μ
      /-
        🎉 no goals
      -/
  all_goals
    first
    | exact (lintegral_rnDeriv_lt_top _ _).ne
    | measurability


theorem jordanDecomposition_add_withDensity_mutuallySingular {f : α → ℝ} (hf : Measurable f)
    (htμ : t ⟂ᵥ μ.toENNRealVectorMeasure) :
    (t.toJordanDecomposition.posPart + μ.withDensity fun x : α => ENNReal.ofReal (f x)) ⟂ₘ
      t.toJordanDecomposition.negPart + μ.withDensity fun x : α => ENNReal.ofReal (-f x) := by
  rw [mutuallySingular_ennreal_iff, totalVariation_mutuallySingular_iff,
    VectorMeasure.ennrealToMeasure_toENNRealVectorMeasure] at htμ
  exact
    ((JordanDecomposition.mutuallySingular _).add_right
          (htμ.1.mono_ac (refl _) (withDensity_absolutelyContinuous _ _))).add_left
      ((htμ.2.symm.mono_ac (withDensity_absolutelyContinuous _ _) (refl _)).add_right
        (withDensity_ofReal_mutuallySingular hf))


theorem toJordanDecomposition_eq_of_eq_add_withDensity {f : α → ℝ} (hf : Measurable f)
    (hfi : Integrable f μ) (htμ : t ⟂ᵥ μ.toENNRealVectorMeasure) (hadd : s = t + μ.withDensityᵥ f) :
    s.toJordanDecomposition =
      @JordanDecomposition.mk α _
        (t.toJordanDecomposition.posPart + μ.withDensity fun x => ENNReal.ofReal (f x))
        (t.toJordanDecomposition.negPart + μ.withDensity fun x => ENNReal.ofReal (-f x))
            /-
              α : Type u_1
              m : MeasurableSpace α
              μ : MeasureTheory.Measure α
              s t : MeasureTheory.SignedMeasure α
              f : α → Real
              hf : Measurable f
              hfi : MeasureTheory.Integrable f μ
              htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
              hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
              ⊢ MeasureTheory.IsFiniteMeasure (HAdd.hAdd t.toJordanDecomposition.posPart (μ. …
            -/
        (by haveI := isFiniteMeasure_withDensity_ofReal hfi.2; infer_instance)
                                                               /-
                                                                 🎉 no goals
                                                               -/
            /-
              α : Type u_1
              m : MeasurableSpace α
              μ : MeasureTheory.Measure α
              s t : MeasureTheory.SignedMeasure α
              f : α → Real
              hf : Measurable f
              hfi : MeasureTheory.Integrable f μ
              htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
              hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
              ⊢ MeasureTheory.IsFiniteMeasure (HAdd.hAdd t.toJordanDecomposition.negPart (μ. …
            -/
        (by haveI := isFiniteMeasure_withDensity_ofReal hfi.neg.2; infer_instance)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
        (jordanDecomposition_add_withDensity_mutuallySingular hf htμ) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    ⊢ Eq s.toJordanDecomposition (MeasureTheory.JordanDecomposition.mk (HAdd.hAdd  …
  -/
  haveI := isFiniteMeasure_withDensity_ofReal hfi.2
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    this : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal (f …
    ⊢ Eq s.toJordanDecomposition (MeasureTheory.JordanDecomposition.mk (HAdd.hAdd  …
  -/
  haveI := isFiniteMeasure_withDensity_ofReal hfi.neg.2
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    this✝ : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal ( …
    this : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal (N …
    ⊢ Eq s.toJordanDecomposition (MeasureTheory.JordanDecomposition.mk (HAdd.hAdd  …
  -/
  refine toJordanDecomposition_eq ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    this✝ : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal ( …
    this : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal (N …
    ⊢ Eq s (MeasureTheory.JordanDecomposition.mk (HAdd.hAdd t.toJordanDecompositio …
  -/
  simp_rw [JordanDecomposition.toSignedMeasure, hadd]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    this✝ : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal ( …
    this : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal (N …
    ⊢ Eq (HAdd.hAdd t (μ.withDensityᵥ f)) (HSub.hSub (HAdd.hAdd t.toJordanDecompos …
  -/
  ext i hi
  rw [VectorMeasure.sub_apply, toSignedMeasure_apply_measurable hi,
      toSignedMeasure_apply_measurable hi, add_apply, add_apply, ENNReal.toReal_add,
      ENNReal.toReal_add, add_sub_add_comm, ← toSignedMeasure_apply_measurable hi,
      ← toSignedMeasure_apply_measurable hi, ← VectorMeasure.sub_apply,
      ← JordanDecomposition.toSignedMeasure, toSignedMeasure_toJordanDecomposition,
      VectorMeasure.add_apply, ← toSignedMeasure_apply_measurable hi,
      ← toSignedMeasure_apply_measurable hi,
      withDensityᵥ_eq_withDensity_pos_part_sub_withDensity_neg_part hfi,
      VectorMeasure.sub_apply] <;>
    /-
      case h.ha
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      hf : Measurable f
      hfi : MeasureTheory.Integrable f μ
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      this✝ : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal ( …
      this : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal (N …
      i : Set α
      hi : MeasurableSet i
      ⊢ Ne (t.toJordanDecomposition.negPart i) Top.top
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    exact (measure_lt_top _ _).ne
    /-
      🎉 no goals
    -/


private theorem haveLebesgueDecomposition_mk' (μ : Measure α) {f : α → ℝ} (hf : Measurable f)
    (hfi : Integrable f μ) (htμ : t ⟂ᵥ μ.toENNRealVectorMeasure) (hadd : s = t + μ.withDensityᵥ f) :
    s.HaveLebesgueDecomposition μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    ⊢ s.HaveLebesgueDecomposition μ
  -/
  have htμ' := htμ
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    htμ' : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    ⊢ s.HaveLebesgueDecomposition μ
  -/
  rw [mutuallySingular_ennreal_iff] at htμ
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    htμ : t.totalVariation.MutuallySingular μ.toENNRealVectorMeasure.ennrealToMeas …
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    htμ' : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    ⊢ s.HaveLebesgueDecomposition μ
  -/
  change _ ⟂ₘ VectorMeasure.equivMeasure.toFun (VectorMeasure.equivMeasure.invFun μ) at htμ
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    htμ' : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    htμ : t.totalVariation.MutuallySingular (MeasureTheory.VectorMeasure.equivMeas …
    ⊢ s.HaveLebesgueDecomposition μ
  -/
  rw [VectorMeasure.equivMeasure.right_inv, totalVariation_mutuallySingular_iff] at htμ
  refine
    { posPart := by
        use ⟨t.toJordanDecomposition.posPart, fun x => ENNReal.ofReal (f x)⟩
        refine ⟨hf.ennreal_ofReal, htμ.1, ?_⟩
        rw [toJordanDecomposition_eq_of_eq_add_withDensity hf hfi htμ' hadd]
      negPart := by
        use ⟨t.toJordanDecomposition.negPart, fun x => ENNReal.ofReal (-f x)⟩
        refine ⟨hf.neg.ennreal_ofReal, htμ.2, ?_⟩
        rw [toJordanDecomposition_eq_of_eq_add_withDensity hf hfi htμ' hadd] }


theorem haveLebesgueDecomposition_mk (μ : Measure α) {f : α → ℝ} (hf : Measurable f)
    (htμ : t ⟂ᵥ μ.toENNRealVectorMeasure) (hadd : s = t + μ.withDensityᵥ f) :
    s.HaveLebesgueDecomposition μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Measurable f
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    ⊢ s.HaveLebesgueDecomposition μ
  -/
  by_cases hfi : Integrable f μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      s t : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Measurable f
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      hfi : MeasureTheory.Integrable f μ
      ⊢ s.HaveLebesgueDecomposition μ
    -/
  · exact haveLebesgueDecomposition_mk' μ hf hfi htμ hadd
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      s t : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Measurable f
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ s.HaveLebesgueDecomposition μ
    -/
  · rw [withDensityᵥ, dif_neg hfi, add_zero] at hadd
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      s t : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Measurable f
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s t
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ s.HaveLebesgueDecomposition μ
    -/
    refine haveLebesgueDecomposition_mk' μ measurable_zero (integrable_zero _ _ μ) htμ ?_
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      s t : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Measurable f
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s t
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq s (HAdd.hAdd t (μ.withDensityᵥ 0))
    -/
    rwa [withDensityᵥ_zero, add_zero]
    /-
      🎉 no goals
    -/


private theorem eq_singularPart' (t : SignedMeasure α) {f : α → ℝ} (hf : Measurable f)
    (hfi : Integrable f μ) (htμ : t ⟂ᵥ μ.toENNRealVectorMeasure) (hadd : s = t + μ.withDensityᵥ f) :
    t = s.singularPart μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    ⊢ Eq t (s.singularPart μ)
  -/
  have htμ' := htμ
  rw [mutuallySingular_ennreal_iff, totalVariation_mutuallySingular_iff,
    VectorMeasure.ennrealToMeasure_toENNRealVectorMeasure] at htμ
  rw [singularPart, ← t.toSignedMeasure_toJordanDecomposition,
    JordanDecomposition.toSignedMeasure]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    hf : Measurable f
    hfi : MeasureTheory.Integrable f μ
    htμ : And (t.toJordanDecomposition.posPart.MutuallySingular μ) (t.toJordanDeco …
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    htμ' : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    ⊢ Eq (HSub.hSub t.toJordanDecomposition.posPart.toSignedMeasure t.toJordanDeco …
  -/
  congr
  -- NB: `measurability` proves this `have`, but is slow.
  -- TODO: make `fun_prop` able to handle this
    /-
      case e_a.e_μ
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      hf : Measurable f
      hfi : MeasureTheory.Integrable f μ
      htμ : And (t.toJordanDecomposition.posPart.MutuallySingular μ) (t.toJordanDeco …
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      htμ' : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      ⊢ Eq t.toJordanDecomposition.posPart (s.toJordanDecomposition.posPart.singular …
    -/
  · have hfpos : Measurable fun x => ENNReal.ofReal (f x) := hf.real_toNNReal.coe_nnreal_ennreal
    /-
      case e_a.e_μ
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      hf : Measurable f
      hfi : MeasureTheory.Integrable f μ
      htμ : And (t.toJordanDecomposition.posPart.MutuallySingular μ) (t.toJordanDeco …
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      htμ' : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hfpos : Measurable fun x => ENNReal.ofReal (f x)
      ⊢ Eq t.toJordanDecomposition.posPart (s.toJordanDecomposition.posPart.singular …
    -/
    refine eq_singularPart hfpos htμ.1 ?_
    /-
      case e_a.e_μ
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      hf : Measurable f
      hfi : MeasureTheory.Integrable f μ
      htμ : And (t.toJordanDecomposition.posPart.MutuallySingular μ) (t.toJordanDeco …
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      htμ' : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hfpos : Measurable fun x => ENNReal.ofReal (f x)
      ⊢ Eq s.toJordanDecomposition.posPart (HAdd.hAdd t.toJordanDecomposition.posPar …
    -/
    rw [toJordanDecomposition_eq_of_eq_add_withDensity hf hfi htμ' hadd]
    /-
      🎉 no goals
    -/
  · have hfneg : Measurable fun x => ENNReal.ofReal (-f x) :=
      -- NB: `measurability` proves this, but is slow.
      -- XXX: `fun_prop` doesn't work here yet
      (measurable_neg_iff.mpr hf).real_toNNReal.coe_nnreal_ennreal
    /-
      case e_a.e_μ
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      hf : Measurable f
      hfi : MeasureTheory.Integrable f μ
      htμ : And (t.toJordanDecomposition.posPart.MutuallySingular μ) (t.toJordanDeco …
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      htμ' : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hfneg : Measurable fun x => ENNReal.ofReal (Neg.neg (f x))
      ⊢ Eq t.toJordanDecomposition.negPart (s.toJordanDecomposition.negPart.singular …
    -/
    refine eq_singularPart hfneg htμ.2 ?_
    /-
      case e_a.e_μ
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      hf : Measurable f
      hfi : MeasureTheory.Integrable f μ
      htμ : And (t.toJordanDecomposition.posPart.MutuallySingular μ) (t.toJordanDeco …
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      htμ' : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hfneg : Measurable fun x => ENNReal.ofReal (Neg.neg (f x))
      ⊢ Eq s.toJordanDecomposition.negPart (HAdd.hAdd t.toJordanDecomposition.negPar …
    -/
    rw [toJordanDecomposition_eq_of_eq_add_withDensity hf hfi htμ' hadd]
    /-
      🎉 no goals
    -/


/-- Given a measure `μ`, signed measures `s` and `t`, and a function `f` such that `t` is
mutually singular with respect to `μ` and `s = t + μ.withDensityᵥ f`, we have
`t = singularPart s μ`, i.e. `t` is the singular part of the Lebesgue decomposition between
`s` and `μ`. -/
theorem eq_singularPart (t : SignedMeasure α) (f : α → ℝ) (htμ : t ⟂ᵥ μ.toENNRealVectorMeasure)
    (hadd : s = t + μ.withDensityᵥ f) : t = s.singularPart μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    ⊢ Eq t (s.singularPart μ)
  -/
  by_cases hfi : Integrable f μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      hfi : MeasureTheory.Integrable f μ
      ⊢ Eq t (s.singularPart μ)
    -/
  · refine eq_singularPart' t hfi.1.measurable_mk (hfi.congr hfi.1.ae_eq_mk) htμ ?_
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      hfi : MeasureTheory.Integrable f μ
      ⊢ Eq s (HAdd.hAdd t (μ.withDensityᵥ (MeasureTheory.AEStronglyMeasurable.mk f ⋯ …
    -/
    convert hadd using 2
    /-
      case h.e'_3.h.e'_6
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      hfi : MeasureTheory.Integrable f μ
      ⊢ Eq (μ.withDensityᵥ (MeasureTheory.AEStronglyMeasurable.mk f ⋯)) (μ.withDensi …
    -/
    exact WithDensityᵥEq.congr_ae hfi.1.ae_eq_mk.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq t (s.singularPart μ)
    -/
  · rw [withDensityᵥ, dif_neg hfi, add_zero] at hadd
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s t
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq t (s.singularPart μ)
    -/
    refine eq_singularPart' t measurable_zero (integrable_zero _ _ μ) htμ ?_
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : MeasureTheory.SignedMeasure α
      f : α → Real
      htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
      hadd : Eq s t
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq s (HAdd.hAdd t (μ.withDensityᵥ 0))
    -/
    rwa [withDensityᵥ_zero, add_zero]
    /-
      🎉 no goals
    -/


theorem singularPart_zero (μ : Measure α) : (0 : SignedMeasure α).singularPart μ = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.SignedMeasure.singularPart 0 μ) 0
  -/
  refine (eq_singularPart 0 0 VectorMeasure.MutuallySingular.zero_left ?_).symm
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq 0 (HAdd.hAdd 0 (μ.withDensityᵥ 0))
  -/
  rw [zero_add, withDensityᵥ_zero]
  /-
    🎉 no goals
  -/


theorem singularPart_neg (s : SignedMeasure α) (μ : Measure α) :
    (-s).singularPart μ = -s.singularPart μ := by
  have h₁ :
    ((-s).toJordanDecomposition.posPart.singularPart μ).toSignedMeasure =
      (s.toJordanDecomposition.negPart.singularPart μ).toSignedMeasure := by
    refine toSignedMeasure_congr ?_
    rw [toJordanDecomposition_neg, JordanDecomposition.neg_posPart]
  have h₂ :
    ((-s).toJordanDecomposition.negPart.singularPart μ).toSignedMeasure =
      (s.toJordanDecomposition.posPart.singularPart μ).toSignedMeasure := by
    refine toSignedMeasure_congr ?_
    rw [toJordanDecomposition_neg, JordanDecomposition.neg_negPart]
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    h₁ : Eq ((Neg.neg s).toJordanDecomposition.posPart.singularPart μ).toSignedMea …
    h₂ : Eq ((Neg.neg s).toJordanDecomposition.negPart.singularPart μ).toSignedMea …
    ⊢ Eq ((Neg.neg s).singularPart μ) (Neg.neg (s.singularPart μ))
  -/
  rw [singularPart, singularPart, neg_sub, h₁, h₂]
  /-
    🎉 no goals
  -/


theorem singularPart_smul_nnreal (s : SignedMeasure α) (μ : Measure α) (r : ℝ≥0) :
    (r • s).singularPart μ = r • s.singularPart μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    r : NNReal
    ⊢ Eq ((HSMul.hSMul r s).singularPart μ) (HSMul.hSMul r (s.singularPart μ))
  -/
  rw [singularPart, singularPart, smul_sub, ← toSignedMeasure_smul, ← toSignedMeasure_smul]
  conv_lhs =>
    congr
    · congr
      · rw [toJordanDecomposition_smul, JordanDecomposition.smul_posPart, singularPart_smul]
    · congr
      rw [toJordanDecomposition_smul, JordanDecomposition.smul_negPart, singularPart_smul]


nonrec theorem singularPart_smul (s : SignedMeasure α) (μ : Measure α) (r : ℝ) :
    (r • s).singularPart μ = r • s.singularPart μ := by
  cases le_or_lt 0 r with
  | inl hr =>
    lift r to ℝ≥0 using hr
    exact singularPart_smul_nnreal s μ r
  | inr hr =>
    rw [singularPart, singularPart]
    conv_lhs =>
      congr
      · congr
        · rw [toJordanDecomposition_smul_real,
            JordanDecomposition.real_smul_posPart_neg _ _ hr, singularPart_smul]
      · congr
        · rw [toJordanDecomposition_smul_real,
            JordanDecomposition.real_smul_negPart_neg _ _ hr, singularPart_smul]
    rw [toSignedMeasure_smul, toSignedMeasure_smul, ← neg_sub, ← smul_sub, NNReal.smul_def,
      ← neg_smul, Real.coe_toNNReal _ (le_of_lt (neg_pos.mpr hr)), neg_neg]


theorem singularPart_add (s t : SignedMeasure α) (μ : Measure α) [s.HaveLebesgueDecomposition μ]
    [t.HaveLebesgueDecomposition μ] :
    (s + t).singularPart μ = s.singularPart μ + t.singularPart μ := by
  refine
    (eq_singularPart _ (s.rnDeriv μ + t.rnDeriv μ)
        ((mutuallySingular_singularPart s μ).add_left (mutuallySingular_singularPart t μ))
        ?_).symm
  rw [withDensityᵥ_add (integrable_rnDeriv s μ) (integrable_rnDeriv t μ), add_assoc,
    add_comm (t.singularPart μ), add_assoc, add_comm _ (t.singularPart μ),
    singularPart_add_withDensity_rnDeriv_eq, ← add_assoc,
    singularPart_add_withDensity_rnDeriv_eq]


theorem singularPart_sub (s t : SignedMeasure α) (μ : Measure α) [s.HaveLebesgueDecomposition μ]
    [t.HaveLebesgueDecomposition μ] :
    (s - t).singularPart μ = s.singularPart μ - t.singularPart μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    inst✝¹ : s.HaveLebesgueDecomposition μ
    inst✝ : t.HaveLebesgueDecomposition μ
    ⊢ Eq ((HSub.hSub s t).singularPart μ) (HSub.hSub (s.singularPart μ) (t.singula …
  -/
  rw [sub_eq_add_neg, sub_eq_add_neg, singularPart_add, singularPart_neg]
  /-
    🎉 no goals
  -/


/-- Given a measure `μ`, signed measures `s` and `t`, and a function `f` such that `t` is
mutually singular with respect to `μ` and `s = t + μ.withDensityᵥ f`, we have
`f = rnDeriv s μ`, i.e. `f` is the Radon-Nikodym derivative of `s` and `μ`. -/
theorem eq_rnDeriv (t : SignedMeasure α) (f : α → ℝ) (hfi : Integrable f μ)
    (htμ : t ⟂ᵥ μ.toENNRealVectorMeasure) (hadd : s = t + μ.withDensityᵥ f) :
    f =ᵐ[μ] s.rnDeriv μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    ⊢ (MeasureTheory.ae μ).EventuallyEq f (s.rnDeriv μ)
  -/
  set f' := hfi.1.mk f
  have hadd' : s = t + μ.withDensityᵥ f' := by
    convert hadd using 2
    exact WithDensityᵥEq.congr_ae hfi.1.ae_eq_mk.symm
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    f' : α → Real := MeasureTheory.AEStronglyMeasurable.mk f ⋯
    hadd' : Eq s (HAdd.hAdd t (μ.withDensityᵥ f'))
    ⊢ (MeasureTheory.ae μ).EventuallyEq f (s.rnDeriv μ)
  -/
  have := haveLebesgueDecomposition_mk μ hfi.1.measurable_mk htμ hadd'
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : MeasureTheory.SignedMeasure α
    f : α → Real
    hfi : MeasureTheory.Integrable f μ
    htμ : MeasureTheory.VectorMeasure.MutuallySingular t μ.toENNRealVectorMeasure
    hadd : Eq s (HAdd.hAdd t (μ.withDensityᵥ f))
    f' : α → Real := MeasureTheory.AEStronglyMeasurable.mk f ⋯
    hadd' : Eq s (HAdd.hAdd t (μ.withDensityᵥ f'))
    this : s.HaveLebesgueDecomposition μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq f (s.rnDeriv μ)
  -/
  refine (Integrable.ae_eq_of_withDensityᵥ_eq (integrable_rnDeriv _ _) hfi ?_).symm
  rw [← add_right_inj t, ← hadd, eq_singularPart _ f htμ hadd,
    singularPart_add_withDensity_rnDeriv_eq]


theorem rnDeriv_neg (s : SignedMeasure α) (μ : Measure α) [s.HaveLebesgueDecomposition μ] :
    (-s).rnDeriv μ =ᵐ[μ] -s.rnDeriv μ := by
  refine
    Integrable.ae_eq_of_withDensityᵥ_eq (integrable_rnDeriv _ _) (integrable_rnDeriv _ _).neg ?_
  rw [withDensityᵥ_neg, ← add_right_inj ((-s).singularPart μ),
    singularPart_add_withDensity_rnDeriv_eq, singularPart_neg, ← neg_add,
    singularPart_add_withDensity_rnDeriv_eq]


theorem rnDeriv_smul (s : SignedMeasure α) (μ : Measure α) [s.HaveLebesgueDecomposition μ] (r : ℝ) :
    (r • s).rnDeriv μ =ᵐ[μ] r • s.rnDeriv μ := by
  refine
    Integrable.ae_eq_of_withDensityᵥ_eq (integrable_rnDeriv _ _)
      ((integrable_rnDeriv _ _).smul r) ?_
  rw [withDensityᵥ_smul (rnDeriv s μ) r, ← add_right_inj ((r • s).singularPart μ),
    singularPart_add_withDensity_rnDeriv_eq, singularPart_smul, ← smul_add,
    singularPart_add_withDensity_rnDeriv_eq]


theorem rnDeriv_add (s t : SignedMeasure α) (μ : Measure α) [s.HaveLebesgueDecomposition μ]
    [t.HaveLebesgueDecomposition μ] [(s + t).HaveLebesgueDecomposition μ] :
    (s + t).rnDeriv μ =ᵐ[μ] s.rnDeriv μ + t.rnDeriv μ := by
  refine
    Integrable.ae_eq_of_withDensityᵥ_eq (integrable_rnDeriv _ _)
      ((integrable_rnDeriv _ _).add (integrable_rnDeriv _ _)) ?_
  rw [← add_right_inj ((s + t).singularPart μ), singularPart_add_withDensity_rnDeriv_eq,
    withDensityᵥ_add (integrable_rnDeriv _ _) (integrable_rnDeriv _ _), singularPart_add,
    add_assoc, add_comm (t.singularPart μ), add_assoc, add_comm _ (t.singularPart μ),
    singularPart_add_withDensity_rnDeriv_eq, ← add_assoc,
    singularPart_add_withDensity_rnDeriv_eq]


theorem rnDeriv_sub (s t : SignedMeasure α) (μ : Measure α) [s.HaveLebesgueDecomposition μ]
    [t.HaveLebesgueDecomposition μ] [hst : (s - t).HaveLebesgueDecomposition μ] :
    (s - t).rnDeriv μ =ᵐ[μ] s.rnDeriv μ - t.rnDeriv μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    inst✝¹ : s.HaveLebesgueDecomposition μ
    inst✝ : t.HaveLebesgueDecomposition μ
    hst : (HSub.hSub s t).HaveLebesgueDecomposition μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HSub.hSub s t).rnDeriv μ) (HSub.hSub (s. …
  -/
  rw [sub_eq_add_neg] at hst
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    inst✝¹ : s.HaveLebesgueDecomposition μ
    inst✝ : t.HaveLebesgueDecomposition μ
    hst : (HAdd.hAdd s (Neg.neg t)).HaveLebesgueDecomposition μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HSub.hSub s t).rnDeriv μ) (HSub.hSub (s. …
  -/
  rw [sub_eq_add_neg, sub_eq_add_neg]
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    inst✝¹ : s.HaveLebesgueDecomposition μ
    inst✝ : t.HaveLebesgueDecomposition μ
    hst : (HAdd.hAdd s (Neg.neg t)).HaveLebesgueDecomposition μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HAdd.hAdd s (Neg.neg t)).rnDeriv μ) (HAd …
  -/
  exact ae_eq_trans (rnDeriv_add _ _ _) (Filter.EventuallyEq.add (ae_eq_refl _) (rnDeriv_neg _ _))
  /-
    🎉 no goals
  -/


/-- A complex measure is said to `HaveLebesgueDecomposition` with respect to a positive measure
if both its real and imaginary part `HaveLebesgueDecomposition` with respect to that measure. -/
class HaveLebesgueDecomposition (c : ComplexMeasure α) (μ : Measure α) : Prop where
  rePart : c.re.HaveLebesgueDecomposition μ
  imPart : c.im.HaveLebesgueDecomposition μ


/-- The singular part between a complex measure `c` and a positive measure `μ` is the complex
measure satisfying `c.singularPart μ + μ.withDensityᵥ (c.rnDeriv μ) = c`. This property is given
by `MeasureTheory.ComplexMeasure.singularPart_add_withDensity_rnDeriv_eq`. -/
def singularPart (c : ComplexMeasure α) (μ : Measure α) : ComplexMeasure α :=
  (c.re.singularPart μ).toComplexMeasure (c.im.singularPart μ)


/-- The Radon-Nikodym derivative between a complex measure and a positive measure. -/
def rnDeriv (c : ComplexMeasure α) (μ : Measure α) : α → ℂ := fun x =>
  ⟨c.re.rnDeriv μ x, c.im.rnDeriv μ x⟩


theorem integrable_rnDeriv (c : ComplexMeasure α) (μ : Measure α) : Integrable (c.rnDeriv μ) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    c : MeasureTheory.ComplexMeasure α
    μ : MeasureTheory.Measure α
    ⊢ MeasureTheory.Integrable (c.rnDeriv μ) μ
  -/
  rw [← memℒp_one_iff_integrable, ← memℒp_re_im_iff]
  exact
    ⟨memℒp_one_iff_integrable.2 (SignedMeasure.integrable_rnDeriv _ _),
      memℒp_one_iff_integrable.2 (SignedMeasure.integrable_rnDeriv _ _)⟩


theorem singularPart_add_withDensity_rnDeriv_eq [c.HaveLebesgueDecomposition μ] :
    c.singularPart μ + μ.withDensityᵥ (c.rnDeriv μ) = c := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : MeasureTheory.ComplexMeasure α
    inst✝ : c.HaveLebesgueDecomposition μ
    ⊢ Eq (HAdd.hAdd (c.singularPart μ) (μ.withDensityᵥ (c.rnDeriv μ))) c
  -/
  conv_rhs => rw [← c.toComplexMeasure_to_signedMeasure]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : MeasureTheory.ComplexMeasure α
    inst✝ : c.HaveLebesgueDecomposition μ
    ⊢ Eq (HAdd.hAdd (c.singularPart μ) (μ.withDensityᵥ (c.rnDeriv μ))) ((MeasureTh …
  -/
  ext i hi : 1
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : MeasureTheory.ComplexMeasure α
    inst✝ : c.HaveLebesgueDecomposition μ
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq (↑(HAdd.hAdd (c.singularPart μ) (μ.withDensityᵥ (c.rnDeriv μ))) i) (↑((Me …
  -/
  rw [VectorMeasure.add_apply, SignedMeasure.toComplexMeasure_apply]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : MeasureTheory.ComplexMeasure α
    inst✝ : c.HaveLebesgueDecomposition μ
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq (HAdd.hAdd (↑(c.singularPart μ) i) (↑(μ.withDensityᵥ (c.rnDeriv μ)) i)) { …
  -/
  apply Complex.ext
  · rw [Complex.add_re, withDensityᵥ_apply (c.integrable_rnDeriv μ) hi, ← RCLike.re_eq_complex_re,
      ← integral_re (c.integrable_rnDeriv μ).integrableOn, RCLike.re_eq_complex_re,
      ← withDensityᵥ_apply _ hi]
      /-
        case h.a
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        c : MeasureTheory.ComplexMeasure α
        inst✝ : c.HaveLebesgueDecomposition μ
        i : Set α
        hi : MeasurableSet i
        ⊢ Eq (HAdd.hAdd (↑(c.singularPart μ) i).re (↑(μ.withDensityᵥ fun x => (c.rnDer …
      -/
    · change (c.re.singularPart μ + μ.withDensityᵥ (c.re.rnDeriv μ)) i = _
      /-
        case h.a
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        c : MeasureTheory.ComplexMeasure α
        inst✝ : c.HaveLebesgueDecomposition μ
        i : Set α
        hi : MeasurableSet i
        ⊢ Eq (↑(HAdd.hAdd ((MeasureTheory.ComplexMeasure.re c).singularPart μ) (μ.with …
      -/
      rw [c.re.singularPart_add_withDensity_rnDeriv_eq μ]
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        c : MeasureTheory.ComplexMeasure α
        inst✝ : c.HaveLebesgueDecomposition μ
        i : Set α
        hi : MeasurableSet i
        ⊢ MeasureTheory.Integrable (fun x => (c.rnDeriv μ x).re) μ
      -/
    · exact SignedMeasure.integrable_rnDeriv _ _
      /-
        🎉 no goals
      -/
  · rw [Complex.add_im, withDensityᵥ_apply (c.integrable_rnDeriv μ) hi, ← RCLike.im_eq_complex_im,
      ← integral_im (c.integrable_rnDeriv μ).integrableOn, RCLike.im_eq_complex_im,
      ← withDensityᵥ_apply _ hi]
      /-
        case h.a
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        c : MeasureTheory.ComplexMeasure α
        inst✝ : c.HaveLebesgueDecomposition μ
        i : Set α
        hi : MeasurableSet i
        ⊢ Eq (HAdd.hAdd (↑(c.singularPart μ) i).im (↑(μ.withDensityᵥ fun x => (c.rnDer …
      -/
    · change (c.im.singularPart μ + μ.withDensityᵥ (c.im.rnDeriv μ)) i = _
      /-
        case h.a
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        c : MeasureTheory.ComplexMeasure α
        inst✝ : c.HaveLebesgueDecomposition μ
        i : Set α
        hi : MeasurableSet i
        ⊢ Eq (↑(HAdd.hAdd ((MeasureTheory.ComplexMeasure.im c).singularPart μ) (μ.with …
      -/
      rw [c.im.singularPart_add_withDensity_rnDeriv_eq μ]
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        c : MeasureTheory.ComplexMeasure α
        inst✝ : c.HaveLebesgueDecomposition μ
        i : Set α
        hi : MeasurableSet i
        ⊢ MeasureTheory.Integrable (fun x => (c.rnDeriv μ x).im) μ
      -/
    · exact SignedMeasure.integrable_rnDeriv _ _
      /-
        🎉 no goals
      -/


