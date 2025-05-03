/-- Probability measures are defined as the subtype of measures that have the property of being
probability measures (i.e., their total mass is one). -/
def ProbabilityMeasure (Ω : Type*) [MeasurableSpace Ω] : Type _ :=
  { μ : Measure Ω // IsProbabilityMeasure μ }


instance [Inhabited Ω] : Inhabited (ProbabilityMeasure Ω) :=
  ⟨⟨Measure.dirac default, Measure.dirac.isProbabilityMeasure⟩⟩

-- Porting note: as with other subtype synonyms (e.g., `ℝ≥0`), we need a new function for the
-- coercion instead of relying on `Subtype.val`.

/-- Coercion from `MeasureTheory.ProbabilityMeasure Ω` to `MeasureTheory.Measure Ω`. -/
@[coe]
def toMeasure : ProbabilityMeasure Ω → Measure Ω := Subtype.val


/-- A probability measure can be interpreted as a measure. -/
instance : Coe (ProbabilityMeasure Ω) (MeasureTheory.Measure Ω) := { coe := toMeasure }


instance (μ : ProbabilityMeasure Ω) : IsProbabilityMeasure (μ : Measure Ω) :=
  μ.prop


@[simp, norm_cast] lemma coe_mk (μ : Measure Ω) (hμ) : toMeasure ⟨μ, hμ⟩ = μ := rfl


@[simp]
theorem val_eq_to_measure (ν : ProbabilityMeasure Ω) : ν.val = (ν : Measure Ω) := rfl


theorem toMeasure_injective : Function.Injective ((↑) : ProbabilityMeasure Ω → Measure Ω) :=
  Subtype.coe_injective


instance instFunLike : FunLike (ProbabilityMeasure Ω) (Set Ω) ℝ≥0 where
  coe μ s := ((μ : Measure Ω) s).toNNReal
  coe_injective' μ ν h := toMeasure_injective <| Measure.ext fun s _ ↦ by
    /-
      Ω : Type u_1
      inst✝ : MeasurableSpace Ω
      μ ν : MeasureTheory.ProbabilityMeasure Ω
      h : Eq ((fun μ s => (↑μ s).toNNReal) μ) ((fun μ s => (↑μ s).toNNReal) ν)
      s : Set Ω
      x✝ : MeasurableSet s
      ⊢ Eq (↑μ s) (↑ν s)
    -/
    simpa [ENNReal.toNNReal_eq_toNNReal_iff, measure_ne_top] using congr_fun h s
    /-
      🎉 no goals
    -/


lemma coeFn_def (μ : ProbabilityMeasure Ω) : μ = fun s ↦ ((μ : Measure Ω) s).toNNReal := rfl


lemma coeFn_mk (μ : Measure Ω) (hμ) :
    DFunLike.coe (F := ProbabilityMeasure Ω) ⟨μ, hμ⟩ = fun s ↦ (μ s).toNNReal := rfl


@[simp, norm_cast]
lemma mk_apply (μ : Measure Ω) (hμ) (s : Set Ω) :
    DFunLike.coe (F := ProbabilityMeasure Ω) ⟨μ, hμ⟩ s = (μ s).toNNReal := rfl


@[simp, norm_cast]
theorem coeFn_univ (ν : ProbabilityMeasure Ω) : ν univ = 1 :=
  congr_arg ENNReal.toNNReal ν.prop.measure_univ


theorem coeFn_univ_ne_zero (ν : ProbabilityMeasure Ω) : ν univ ≠ 0 := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    ν : MeasureTheory.ProbabilityMeasure Ω
    ⊢ Ne (ν Set.univ) 0
  -/
  simp only [coeFn_univ, Ne, one_ne_zero, not_false_iff]
  /-
    🎉 no goals
  -/


/-- A probability measure can be interpreted as a finite measure. -/
def toFiniteMeasure (μ : ProbabilityMeasure Ω) : FiniteMeasure Ω := ⟨μ, inferInstance⟩


@[simp] lemma coeFn_toFiniteMeasure (μ : ProbabilityMeasure Ω) : ⇑μ.toFiniteMeasure = μ := rfl

lemma toFiniteMeasure_apply (μ : ProbabilityMeasure Ω) (s : Set Ω) :
    μ.toFiniteMeasure s = μ s := rfl


@[simp]
theorem toMeasure_comp_toFiniteMeasure_eq_toMeasure (ν : ProbabilityMeasure Ω) :
    (ν.toFiniteMeasure : Measure Ω) = (ν : Measure Ω) := rfl


@[simp]
theorem coeFn_comp_toFiniteMeasure_eq_coeFn (ν : ProbabilityMeasure Ω) :
    (ν.toFiniteMeasure : Set Ω → ℝ≥0) = (ν : Set Ω → ℝ≥0) := rfl


@[simp]
theorem toFiniteMeasure_apply_eq_apply (ν : ProbabilityMeasure Ω) (s : Set Ω) :
    ν.toFiniteMeasure s = ν s := rfl


@[simp]
theorem ennreal_coeFn_eq_coeFn_toMeasure (ν : ProbabilityMeasure Ω) (s : Set Ω) :
    (ν s : ℝ≥0∞) = (ν : Measure Ω) s := by
  rw [← coeFn_comp_toFiniteMeasure_eq_coeFn, FiniteMeasure.ennreal_coeFn_eq_coeFn_toMeasure,
    toMeasure_comp_toFiniteMeasure_eq_toMeasure]


@[simp]
theorem null_iff_toMeasure_null (ν : ProbabilityMeasure Ω) (s : Set Ω) :
    ν s = 0 ↔ (ν : Measure Ω) s = 0 :=
              /-
                Ω : Type u_1
                inst✝ : MeasurableSpace Ω
                ν : MeasureTheory.ProbabilityMeasure Ω
                s : Set Ω
                h : Eq (ν s) 0
                ⊢ Eq (↑ν s) 0
              -/
  ⟨fun h ↦ by rw [← ennreal_coeFn_eq_coeFn_toMeasure, h, ENNReal.coe_zero],
              /-
                🎉 no goals
              -/
   fun h ↦ congrArg ENNReal.toNNReal h⟩


theorem apply_mono (μ : ProbabilityMeasure Ω) {s₁ s₂ : Set Ω} (h : s₁ ⊆ s₂) : μ s₁ ≤ μ s₂ := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    s₁ s₂ : Set Ω
    h : HasSubset.Subset s₁ s₂
    ⊢ LE.le (μ s₁) (μ s₂)
  -/
  rw [← coeFn_comp_toFiniteMeasure_eq_coeFn]
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    s₁ s₂ : Set Ω
    h : HasSubset.Subset s₁ s₂
    ⊢ LE.le (μ.toFiniteMeasure s₁) (μ.toFiniteMeasure s₂)
  -/
  exact MeasureTheory.FiniteMeasure.apply_mono _ h
  /-
    🎉 no goals
  -/


@[simp] theorem apply_le_one (μ : ProbabilityMeasure Ω) (s : Set Ω) : μ s ≤ 1 := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    s : Set Ω
    ⊢ LE.le (μ s) 1
  -/
  simpa using apply_mono μ (subset_univ s)
  /-
    🎉 no goals
  -/


theorem nonempty (μ : ProbabilityMeasure Ω) : Nonempty Ω := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    ⊢ Nonempty Ω
  -/
  by_contra maybe_empty
  have zero : (μ : Measure Ω) univ = 0 := by
    rw [univ_eq_empty_iff.mpr (not_nonempty_iff.mp maybe_empty), measure_empty]
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    maybe_empty : Not (Nonempty Ω)
    zero : Eq (↑μ Set.univ) 0
    ⊢ False
  -/
  rw [measure_univ] at zero
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    maybe_empty : Not (Nonempty Ω)
    zero : Eq 1 0
    ⊢ False
  -/
  exact zero_ne_one zero.symm
  /-
    🎉 no goals
  -/


@[ext]
theorem eq_of_forall_toMeasure_apply_eq (μ ν : ProbabilityMeasure Ω)
    (h : ∀ s : Set Ω, MeasurableSet s → (μ : Measure Ω) s = (ν : Measure Ω) s) : μ = ν := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.ProbabilityMeasure Ω
    h : ∀ (s : Set Ω), MeasurableSet s → Eq (↑μ s) (↑ν s)
    ⊢ Eq μ ν
  -/
  apply toMeasure_injective
  /-
    case a
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.ProbabilityMeasure Ω
    h : ∀ (s : Set Ω), MeasurableSet s → Eq (↑μ s) (↑ν s)
    ⊢ Eq ↑μ ↑ν
  -/
  ext1 s s_mble
  /-
    case a.h
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.ProbabilityMeasure Ω
    h : ∀ (s : Set Ω), MeasurableSet s → Eq (↑μ s) (↑ν s)
    s : Set Ω
    s_mble : MeasurableSet s
    ⊢ Eq (↑μ s) (↑ν s)
  -/
  exact h s s_mble
  /-
    🎉 no goals
  -/


theorem eq_of_forall_apply_eq (μ ν : ProbabilityMeasure Ω)
    (h : ∀ s : Set Ω, MeasurableSet s → μ s = ν s) : μ = ν := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.ProbabilityMeasure Ω
    h : ∀ (s : Set Ω), MeasurableSet s → Eq (μ s) (ν s)
    ⊢ Eq μ ν
  -/
  ext1 s s_mble
  /-
    case h
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.ProbabilityMeasure Ω
    h : ∀ (s : Set Ω), MeasurableSet s → Eq (μ s) (ν s)
    s : Set Ω
    s_mble : MeasurableSet s
    ⊢ Eq (↑μ s) (↑ν s)
  -/
  simpa [ennreal_coeFn_eq_coeFn_toMeasure] using congr_arg ((↑) : ℝ≥0 → ℝ≥0∞) (h s s_mble)
  /-
    🎉 no goals
  -/


@[simp]
theorem mass_toFiniteMeasure (μ : ProbabilityMeasure Ω) : μ.toFiniteMeasure.mass = 1 :=
  μ.coeFn_univ


theorem toFiniteMeasure_nonzero (μ : ProbabilityMeasure Ω) : μ.toFiniteMeasure ≠ 0 := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    ⊢ Ne μ.toFiniteMeasure 0
  -/
  simp [← FiniteMeasure.mass_nonzero_iff]
  /-
    🎉 no goals
  -/


theorem testAgainstNN_lipschitz (μ : ProbabilityMeasure Ω) :
    LipschitzWith 1 fun f : Ω →ᵇ ℝ≥0 ↦ μ.toFiniteMeasure.testAgainstNN f :=
  μ.mass_toFiniteMeasure ▸ μ.toFiniteMeasure.testAgainstNN_lipschitz


/-- The topology of weak convergence on `MeasureTheory.ProbabilityMeasure Ω`. This is inherited
(induced) from the topology of weak convergence of finite measures via the inclusion
`MeasureTheory.ProbabilityMeasure.toFiniteMeasure`. -/
instance : TopologicalSpace (ProbabilityMeasure Ω) :=
  TopologicalSpace.induced toFiniteMeasure inferInstance


theorem toFiniteMeasure_continuous :
    Continuous (toFiniteMeasure : ProbabilityMeasure Ω → FiniteMeasure Ω) :=
  continuous_induced_dom


/-- Probability measures yield elements of the `WeakDual` of bounded continuous nonnegative
functions via `MeasureTheory.FiniteMeasure.testAgainstNN`, i.e., integration. -/
def toWeakDualBCNN : ProbabilityMeasure Ω → WeakDual ℝ≥0 (Ω →ᵇ ℝ≥0) :=
  FiniteMeasure.toWeakDualBCNN ∘ toFiniteMeasure


@[simp]
theorem coe_toWeakDualBCNN (μ : ProbabilityMeasure Ω) :
    ⇑μ.toWeakDualBCNN = μ.toFiniteMeasure.testAgainstNN := rfl


@[simp]
theorem toWeakDualBCNN_apply (μ : ProbabilityMeasure Ω) (f : Ω →ᵇ ℝ≥0) :
    μ.toWeakDualBCNN f = (∫⁻ ω, f ω ∂(μ : Measure Ω)).toNNReal := rfl


theorem toWeakDualBCNN_continuous : Continuous fun μ : ProbabilityMeasure Ω ↦ μ.toWeakDualBCNN :=
  FiniteMeasure.toWeakDualBCNN_continuous.comp toFiniteMeasure_continuous

/- Integration of (nonnegative bounded continuous) test functions against Borel probability
measures depends continuously on the measure. -/

theorem continuous_testAgainstNN_eval (f : Ω →ᵇ ℝ≥0) :
    Continuous fun μ : ProbabilityMeasure Ω ↦ μ.toFiniteMeasure.testAgainstNN f :=
  (FiniteMeasure.continuous_testAgainstNN_eval f).comp toFiniteMeasure_continuous

-- The canonical mapping from probability measures to finite measures is an embedding.

theorem toFiniteMeasure_isEmbedding (Ω : Type*) [MeasurableSpace Ω] [TopologicalSpace Ω]
    [OpensMeasurableSpace Ω] :
    IsEmbedding (toFiniteMeasure : ProbabilityMeasure Ω → FiniteMeasure Ω) where
  eq_induced := rfl
  injective _μ _ν h := Subtype.eq <| congr_arg FiniteMeasure.toMeasure h


@[deprecated (since := "2024-10-26")]
alias toFiniteMeasure_embedding := toFiniteMeasure_isEmbedding


theorem tendsto_nhds_iff_toFiniteMeasure_tendsto_nhds {δ : Type*} (F : Filter δ)
    {μs : δ → ProbabilityMeasure Ω} {μ₀ : ProbabilityMeasure Ω} :
    Tendsto μs F (𝓝 μ₀) ↔ Tendsto (toFiniteMeasure ∘ μs) F (𝓝 μ₀.toFiniteMeasure) :=
  (toFiniteMeasure_isEmbedding Ω).tendsto_nhds_iff


/-- A characterization of weak convergence of probability measures by the condition that the
integrals of every continuous bounded nonnegative function converge to the integral of the function
against the limit measure. -/
theorem tendsto_iff_forall_lintegral_tendsto {γ : Type*} {F : Filter γ}
    {μs : γ → ProbabilityMeasure Ω} {μ : ProbabilityMeasure Ω} :
    Tendsto μs F (𝓝 μ) ↔
      ∀ f : Ω →ᵇ ℝ≥0,
        Tendsto (fun i ↦ ∫⁻ ω, f ω ∂(μs i : Measure Ω)) F (𝓝 (∫⁻ ω, f ω ∂(μ : Measure Ω))) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.ProbabilityMeasure Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    ⊢ Iff (Filter.Tendsto μs F (nhds μ)) (∀ (f : BoundedContinuousFunction Ω NNRea …
  -/
  rw [tendsto_nhds_iff_toFiniteMeasure_tendsto_nhds]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.ProbabilityMeasure Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    ⊢ Iff (Filter.Tendsto (Function.comp MeasureTheory.ProbabilityMeasure.toFinite …
  -/
  exact FiniteMeasure.tendsto_iff_forall_lintegral_tendsto
  /-
    🎉 no goals
  -/


/-- The characterization of weak convergence of probability measures by the usual (defining)
condition that the integrals of every continuous bounded function converge to the integral of the
function against the limit measure. -/
theorem tendsto_iff_forall_integral_tendsto {γ : Type*} {F : Filter γ}
    {μs : γ → ProbabilityMeasure Ω} {μ : ProbabilityMeasure Ω} :
    Tendsto μs F (𝓝 μ) ↔
      ∀ f : Ω →ᵇ ℝ,
        Tendsto (fun i ↦ ∫ ω, f ω ∂(μs i : Measure Ω)) F (𝓝 (∫ ω, f ω ∂(μ : Measure Ω))) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.ProbabilityMeasure Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    ⊢ Iff (Filter.Tendsto μs F (nhds μ)) (∀ (f : BoundedContinuousFunction Ω Real) …
  -/
  rw [tendsto_nhds_iff_toFiniteMeasure_tendsto_nhds]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.ProbabilityMeasure Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    ⊢ Iff (Filter.Tendsto (Function.comp MeasureTheory.ProbabilityMeasure.toFinite …
  -/
  rw [FiniteMeasure.tendsto_iff_forall_integral_tendsto]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.ProbabilityMeasure Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    ⊢ Iff (∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => Meas …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma continuous_integral_boundedContinuousFunction
    {α : Type*} [TopologicalSpace α] [MeasurableSpace α] [OpensMeasurableSpace α] (f : α →ᵇ ℝ) :
    Continuous fun μ : ProbabilityMeasure α ↦ ∫ x, f x ∂μ := by
  /-
    α : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f : BoundedContinuousFunction α Real
    ⊢ Continuous fun μ => MeasureTheory.integral ↑μ fun x => f x
  -/
  rw [continuous_iff_continuousAt]
  /-
    α : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f : BoundedContinuousFunction α Real
    ⊢ ∀ (x : MeasureTheory.ProbabilityMeasure α), ContinuousAt (fun μ => MeasureTh …
  -/
  intro μ
  exact continuousAt_of_tendsto_nhds
    (ProbabilityMeasure.tendsto_iff_forall_integral_tendsto.mp tendsto_id f)


/-- On topological spaces where indicators of closed sets have decreasing approximating sequences of
continuous functions (`HasOuterApproxClosed`), the topology of convergence in distribution of Borel
probability measures is Hausdorff (`T2Space`). -/
instance t2Space : T2Space (ProbabilityMeasure Ω) := (toFiniteMeasure_isEmbedding Ω).t2Space


/-- Normalize a finite measure so that it becomes a probability measure, i.e., divide by the
total mass. -/
def normalize : ProbabilityMeasure Ω :=
  if zero : μ.mass = 0 then ⟨Measure.dirac ‹Nonempty Ω›.some, Measure.dirac.isProbabilityMeasure⟩
  else
    { val := ↑(μ.mass⁻¹ • μ)
      property := by
        /-
          Ω : Type u_1
          inst✝ : Nonempty Ω
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.FiniteMeasure Ω
          zero : Not (Eq μ.mass 0)
          ⊢ MeasureTheory.IsProbabilityMeasure ↑(HSMul.hSMul (Inv.inv μ.mass) μ)
        -/
        refine ⟨?_⟩
        -- Porting note: paying the price that this isn't `simp` lemma now.
        /-
          Ω : Type u_1
          inst✝ : Nonempty Ω
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.FiniteMeasure Ω
          zero : Not (Eq μ.mass 0)
          ⊢ Eq (↑(HSMul.hSMul (Inv.inv μ.mass) μ) Set.univ) 1
        -/
        rw [FiniteMeasure.toMeasure_smul]
        simp only [Measure.coe_smul, Pi.smul_apply, Measure.nnreal_smul_coe_apply, ne_eq,
          mass_zero_iff, ENNReal.coe_inv zero, ennreal_mass]
        /-
          Ω : Type u_1
          inst✝ : Nonempty Ω
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.FiniteMeasure Ω
          zero : Not (Eq μ.mass 0)
          ⊢ Eq (HMul.hMul (Inv.inv (↑μ Set.univ)) (↑μ Set.univ)) 1
        -/
        rw [← Ne, ← ENNReal.coe_ne_zero, ennreal_mass] at zero
        /-
          Ω : Type u_1
          inst✝ : Nonempty Ω
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.FiniteMeasure Ω
          zero : Ne (↑μ Set.univ) 0
          ⊢ Eq (HMul.hMul (Inv.inv (↑μ Set.univ)) (↑μ Set.univ)) 1
        -/
        exact ENNReal.inv_mul_cancel zero μ.prop.measure_univ_lt_top.ne }
        /-
          🎉 no goals
        -/


@[simp]
theorem self_eq_mass_mul_normalize (s : Set Ω) : μ s = μ.mass * μ.normalize s := by
  /-
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    s : Set Ω
    ⊢ Eq (μ s) (HMul.hMul μ.mass (μ.normalize s))
  -/
  obtain rfl | h := eq_or_ne μ 0
    /-
      case inl
      Ω : Type u_1
      inst✝ : Nonempty Ω
      m0 : MeasurableSpace Ω
      s : Set Ω
      ⊢ Eq (0 s) (HMul.hMul (MeasureTheory.FiniteMeasure.mass 0) ((MeasureTheory.Fin …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    s : Set Ω
    h : Ne μ 0
    ⊢ Eq (μ s) (HMul.hMul μ.mass (μ.normalize s))
  -/
  have mass_nonzero : μ.mass ≠ 0 := by rwa [μ.mass_nonzero_iff]
  /-
    case inr
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    s : Set Ω
    h : Ne μ 0
    mass_nonzero : Ne μ.mass 0
    ⊢ Eq (μ s) (HMul.hMul μ.mass (μ.normalize s))
  -/
  simp only [normalize, dif_neg mass_nonzero]
  /-
    case inr
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    s : Set Ω
    h : Ne μ 0
    mass_nonzero : Ne μ.mass 0
    ⊢ Eq (μ s) (HMul.hMul μ.mass (⟨↑(HSMul.hSMul (Inv.inv μ.mass) μ), ⋯⟩ s))
  -/
  simp [ProbabilityMeasure.coe_mk, toMeasure_smul, mul_inv_cancel_left₀ mass_nonzero, coeFn_def]
  /-
    🎉 no goals
  -/


theorem self_eq_mass_smul_normalize : μ = μ.mass • μ.normalize.toFiniteMeasure := by
  /-
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Eq μ (HSMul.hSMul μ.mass μ.normalize.toFiniteMeasure)
  -/
  apply eq_of_forall_apply_eq
  /-
    case h
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ ∀ (s : Set Ω), MeasurableSet s → Eq (μ s) ((HSMul.hSMul μ.mass μ.normalize.t …
  -/
  intro s _s_mble
  rw [μ.self_eq_mass_mul_normalize s, smul_apply, smul_eq_mul,
    ProbabilityMeasure.coeFn_comp_toFiniteMeasure_eq_coeFn]


theorem normalize_eq_of_nonzero (nonzero : μ ≠ 0) (s : Set Ω) : μ.normalize s = μ.mass⁻¹ * μ s := by
  simp only [μ.self_eq_mass_mul_normalize, μ.mass_nonzero_iff.mpr nonzero, inv_mul_cancel_left₀,
    Ne, not_false_iff]


theorem normalize_eq_inv_mass_smul_of_nonzero (nonzero : μ ≠ 0) :
    μ.normalize.toFiniteMeasure = μ.mass⁻¹ • μ := by
  /-
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    nonzero : Ne μ 0
    ⊢ Eq μ.normalize.toFiniteMeasure (HSMul.hSMul (Inv.inv μ.mass) μ)
  -/
  nth_rw 3 [μ.self_eq_mass_smul_normalize]
  /-
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    nonzero : Ne μ 0
    ⊢ Eq μ.normalize.toFiniteMeasure (HSMul.hSMul (Inv.inv μ.mass) (HSMul.hSMul μ. …
  -/
  rw [← smul_assoc]
  simp only [μ.mass_nonzero_iff.mpr nonzero, Algebra.id.smul_eq_mul, inv_mul_cancel₀, Ne,
    not_false_iff, one_smul]


theorem toMeasure_normalize_eq_of_nonzero (nonzero : μ ≠ 0) :
    (μ.normalize : Measure Ω) = μ.mass⁻¹ • μ := by
  /-
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    nonzero : Ne μ 0
    ⊢ Eq (↑μ.normalize) (HSMul.hSMul (Inv.inv μ.mass) ↑μ)
  -/
  ext1 s _s_mble
  rw [← μ.normalize.ennreal_coeFn_eq_coeFn_toMeasure s, μ.normalize_eq_of_nonzero nonzero s,
    ENNReal.coe_mul, ennreal_coeFn_eq_coeFn_toMeasure]
  /-
    case h
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    nonzero : Ne μ 0
    s : Set Ω
    _s_mble : MeasurableSet s
    ⊢ Eq (HMul.hMul (↑(Inv.inv μ.mass)) (↑μ s)) ((HSMul.hSMul (Inv.inv μ.mass) ↑μ) …
  -/
  exact Measure.coe_nnreal_smul_apply _ _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.ProbabilityMeasure.toFiniteMeasure_normalize_eq_self {m0 : MeasurableSpace Ω}
    (μ : ProbabilityMeasure Ω) : μ.toFiniteMeasure.normalize = μ := by
  /-
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    ⊢ Eq μ.toFiniteMeasure.normalize μ
  -/
  apply ProbabilityMeasure.eq_of_forall_apply_eq
  /-
    case h
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    ⊢ ∀ (s : Set Ω), MeasurableSet s → Eq (μ.toFiniteMeasure.normalize s) (μ s)
  -/
  intro s _s_mble
  /-
    case h
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    s : Set Ω
    _s_mble : MeasurableSet s
    ⊢ Eq (μ.toFiniteMeasure.normalize s) (μ s)
  -/
  rw [μ.toFiniteMeasure.normalize_eq_of_nonzero μ.toFiniteMeasure_nonzero s]
  /-
    case h
    Ω : Type u_1
    inst✝ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    s : Set Ω
    _s_mble : MeasurableSet s
    ⊢ Eq (HMul.hMul (Inv.inv μ.toFiniteMeasure.mass) (μ.toFiniteMeasure s)) (μ s)
  -/
  simp only [ProbabilityMeasure.mass_toFiniteMeasure, inv_one, one_mul, μ.coeFn_toFiniteMeasure]
  /-
    🎉 no goals
  -/


/-- Averaging with respect to a finite measure is the same as integrating against
`MeasureTheory.FiniteMeasure.normalize`. -/
theorem average_eq_integral_normalize {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (nonzero : μ ≠ 0) (f : Ω → E) :
    average (μ : Measure Ω) f = ∫ ω, f ω ∂(μ.normalize : Measure Ω) := by
  /-
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    nonzero : Ne μ 0
    f : Ω → E
    ⊢ Eq (MeasureTheory.average (↑μ) f) (MeasureTheory.integral ↑μ.normalize fun ω …
  -/
  rw [μ.toMeasure_normalize_eq_of_nonzero nonzero, average]
  /-
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    nonzero : Ne μ 0
    f : Ω → E
    ⊢ Eq (MeasureTheory.integral (HSMul.hSMul (Inv.inv (↑μ Set.univ)) ↑μ) fun x => …
  -/
  congr
  /-
    case e_μ.e_a
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    nonzero : Ne μ 0
    f : Ω → E
    ⊢ Eq (Inv.inv (↑μ Set.univ)) (↑ENNReal.ofNNRealHom.toMonoidWithZeroHom (Inv.in …
  -/
  simp [ENNReal.coe_inv (μ.mass_nonzero_iff.mpr nonzero), ennreal_mass]
  /-
    🎉 no goals
  -/


theorem testAgainstNN_eq_mass_mul (f : Ω →ᵇ ℝ≥0) :
    μ.testAgainstNN f = μ.mass * μ.normalize.toFiniteMeasure.testAgainstNN f := by
  /-
    Ω : Type u_1
    inst✝¹ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝ : TopologicalSpace Ω
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Eq (μ.testAgainstNN f) (HMul.hMul μ.mass (μ.normalize.toFiniteMeasure.testAg …
  -/
  nth_rw 1 [μ.self_eq_mass_smul_normalize]
  /-
    Ω : Type u_1
    inst✝¹ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝ : TopologicalSpace Ω
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Eq ((HSMul.hSMul μ.mass μ.normalize.toFiniteMeasure).testAgainstNN f) (HMul. …
  -/
  rw [μ.normalize.toFiniteMeasure.smul_testAgainstNN_apply μ.mass f, smul_eq_mul]
  /-
    🎉 no goals
  -/


theorem normalize_testAgainstNN (nonzero : μ ≠ 0) (f : Ω →ᵇ ℝ≥0) :
    μ.normalize.toFiniteMeasure.testAgainstNN f = μ.mass⁻¹ * μ.testAgainstNN f := by
  /-
    Ω : Type u_1
    inst✝¹ : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝ : TopologicalSpace Ω
    nonzero : Ne μ 0
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Eq (μ.normalize.toFiniteMeasure.testAgainstNN f) (HMul.hMul (Inv.inv μ.mass) …
  -/
  simp [μ.testAgainstNN_eq_mass_mul, inv_mul_cancel_left₀ <| μ.mass_nonzero_iff.mpr nonzero]
  /-
    🎉 no goals
  -/


theorem tendsto_testAgainstNN_of_tendsto_normalize_testAgainstNN_of_tendsto_mass {γ : Type*}
    {F : Filter γ} {μs : γ → FiniteMeasure Ω}
    (μs_lim : Tendsto (fun i ↦ (μs i).normalize) F (𝓝 μ.normalize))
    (mass_lim : Tendsto (fun i ↦ (μs i).mass) F (𝓝 μ.mass)) (f : Ω →ᵇ ℝ≥0) :
    Tendsto (fun i ↦ (μs i).testAgainstNN f) F (𝓝 (μ.testAgainstNN f)) := by
  /-
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto (fun i => (μs i).normalize) F (nhds μ.normalize)
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds μ.mass)
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Filter.Tendsto (fun i => (μs i).testAgainstNN f) F (nhds (μ.testAgainstNN f))
  -/
  by_cases h_mass : μ.mass = 0
  · simp only [μ.mass_zero_iff.mp h_mass, zero_testAgainstNN_apply, zero_mass,
      eq_self_iff_true] at mass_lim ⊢
    /-
      case pos
      Ω : Type u_1
      inst✝² : Nonempty Ω
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      γ : Type u_2
      F : Filter γ
      μs : γ → MeasureTheory.FiniteMeasure Ω
      μs_lim : Filter.Tendsto (fun i => (μs i).normalize) F (nhds μ.normalize)
      f : BoundedContinuousFunction Ω NNReal
      h_mass : Eq μ.mass 0
      mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds 0)
      ⊢ Filter.Tendsto (fun i => (μs i).testAgainstNN f) F (nhds 0)
    -/
    exact tendsto_zero_testAgainstNN_of_tendsto_zero_mass mass_lim f
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto (fun i => (μs i).normalize) F (nhds μ.normalize)
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds μ.mass)
    f : BoundedContinuousFunction Ω NNReal
    h_mass : Not (Eq μ.mass 0)
    ⊢ Filter.Tendsto (fun i => (μs i).testAgainstNN f) F (nhds (μ.testAgainstNN f))
  -/
  simp_rw [fun i ↦ (μs i).testAgainstNN_eq_mass_mul f, μ.testAgainstNN_eq_mass_mul f]
  /-
    case neg
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto (fun i => (μs i).normalize) F (nhds μ.normalize)
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds μ.mass)
    f : BoundedContinuousFunction Ω NNReal
    h_mass : Not (Eq μ.mass 0)
    ⊢ Filter.Tendsto (fun i => HMul.hMul (μs i).mass ((μs i).normalize.toFiniteMea …
  -/
  rw [ProbabilityMeasure.tendsto_nhds_iff_toFiniteMeasure_tendsto_nhds] at μs_lim
  /-
    case neg
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto (Function.comp MeasureTheory.ProbabilityMeasure.toFini …
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds μ.mass)
    f : BoundedContinuousFunction Ω NNReal
    h_mass : Not (Eq μ.mass 0)
    ⊢ Filter.Tendsto (fun i => HMul.hMul (μs i).mass ((μs i).normalize.toFiniteMea …
  -/
  rw [tendsto_iff_forall_testAgainstNN_tendsto] at μs_lim
  have lim_pair :
    Tendsto (fun i ↦ (⟨(μs i).mass, (μs i).normalize.toFiniteMeasure.testAgainstNN f⟩ : ℝ≥0 × ℝ≥0))
      F (𝓝 ⟨μ.mass, μ.normalize.toFiniteMeasure.testAgainstNN f⟩) :=
    (Prod.tendsto_iff _ _).mpr ⟨mass_lim, μs_lim f⟩
  /-
    case neg
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μs_lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i =>  …
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds μ.mass)
    f : BoundedContinuousFunction Ω NNReal
    h_mass : Not (Eq μ.mass 0)
    lim_pair : Filter.Tendsto (fun i => { fst := (μs i).mass, snd := (μs i).normal …
    ⊢ Filter.Tendsto (fun i => HMul.hMul (μs i).mass ((μs i).normalize.toFiniteMea …
  -/
  exact tendsto_mul.comp lim_pair
  /-
    🎉 no goals
  -/


theorem tendsto_normalize_testAgainstNN_of_tendsto {γ : Type*} {F : Filter γ}
    {μs : γ → FiniteMeasure Ω} (μs_lim : Tendsto μs F (𝓝 μ)) (nonzero : μ ≠ 0) (f : Ω →ᵇ ℝ≥0) :
    Tendsto (fun i ↦ (μs i).normalize.toFiniteMeasure.testAgainstNN f) F
      (𝓝 (μ.normalize.toFiniteMeasure.testAgainstNN f)) := by
  /-
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs F (nhds μ)
    nonzero : Ne μ 0
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Filter.Tendsto (fun i => (μs i).normalize.toFiniteMeasure.testAgainstNN f) F …
  -/
  have lim_mass := μs_lim.mass
  have aux : {(0 : ℝ≥0)}ᶜ ∈ 𝓝 μ.mass :=
    isOpen_compl_singleton.mem_nhds (μ.mass_nonzero_iff.mpr nonzero)
  have eventually_nonzero : ∀ᶠ i in F, μs i ≠ 0 := by
    simp_rw [← mass_nonzero_iff]
    exact lim_mass aux
  have eve : ∀ᶠ i in F,
      (μs i).normalize.toFiniteMeasure.testAgainstNN f =
        (μs i).mass⁻¹ * (μs i).testAgainstNN f := by
    filter_upwards [eventually_iff.mp eventually_nonzero]
    intro i hi
    apply normalize_testAgainstNN _ hi
  /-
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs F (nhds μ)
    nonzero : Ne μ 0
    f : BoundedContinuousFunction Ω NNReal
    lim_mass : Filter.Tendsto (fun i => (μs i).mass) F (nhds μ.mass)
    aux : Membership.mem (nhds μ.mass) (HasCompl.compl (Singleton.singleton 0))
    eventually_nonzero : Filter.Eventually (fun i => Ne (μs i) 0) F
    eve : Filter.Eventually (fun i => Eq ((μs i).normalize.toFiniteMeasure.testAga …
    ⊢ Filter.Tendsto (fun i => (μs i).normalize.toFiniteMeasure.testAgainstNN f) F …
  -/
  simp_rw [tendsto_congr' eve, μ.normalize_testAgainstNN nonzero]
  have lim_pair :
    Tendsto (fun i ↦ (⟨(μs i).mass⁻¹, (μs i).testAgainstNN f⟩ : ℝ≥0 × ℝ≥0)) F
      (𝓝 ⟨μ.mass⁻¹, μ.testAgainstNN f⟩) := by
    refine (Prod.tendsto_iff _ _).mpr ⟨?_, ?_⟩
    · exact (continuousOn_inv₀.continuousAt aux).tendsto.comp lim_mass
    · exact tendsto_iff_forall_testAgainstNN_tendsto.mp μs_lim f
  /-
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs F (nhds μ)
    nonzero : Ne μ 0
    f : BoundedContinuousFunction Ω NNReal
    lim_mass : Filter.Tendsto (fun i => (μs i).mass) F (nhds μ.mass)
    aux : Membership.mem (nhds μ.mass) (HasCompl.compl (Singleton.singleton 0))
    eventually_nonzero : Filter.Eventually (fun i => Ne (μs i) 0) F
    eve : Filter.Eventually (fun i => Eq ((μs i).normalize.toFiniteMeasure.testAga …
    lim_pair : Filter.Tendsto (fun i => { fst := Inv.inv (μs i).mass, snd := (μs i …
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Inv.inv (μs x).mass) ((μs x).testAgainst …
  -/
  exact tendsto_mul.comp lim_pair
  /-
    🎉 no goals
  -/


/-- If the normalized versions of finite measures converge weakly and their total masses
also converge, then the finite measures themselves converge weakly. -/
theorem tendsto_of_tendsto_normalize_testAgainstNN_of_tendsto_mass {γ : Type*} {F : Filter γ}
    {μs : γ → FiniteMeasure Ω} (μs_lim : Tendsto (fun i ↦ (μs i).normalize) F (𝓝 μ.normalize))
    (mass_lim : Tendsto (fun i ↦ (μs i).mass) F (𝓝 μ.mass)) : Tendsto μs F (𝓝 μ) := by
  /-
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto (fun i => (μs i).normalize) F (nhds μ.normalize)
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds μ.mass)
    ⊢ Filter.Tendsto μs F (nhds μ)
  -/
  rw [tendsto_iff_forall_testAgainstNN_tendsto]
  exact fun f ↦
    tendsto_testAgainstNN_of_tendsto_normalize_testAgainstNN_of_tendsto_mass μs_lim mass_lim f


/-- If finite measures themselves converge weakly to a nonzero limit measure, then their
normalized versions also converge weakly. -/
theorem tendsto_normalize_of_tendsto {γ : Type*} {F : Filter γ} {μs : γ → FiniteMeasure Ω}
    (μs_lim : Tendsto μs F (𝓝 μ)) (nonzero : μ ≠ 0) :
    Tendsto (fun i ↦ (μs i).normalize) F (𝓝 μ.normalize) := by
  rw [ProbabilityMeasure.tendsto_nhds_iff_toFiniteMeasure_tendsto_nhds,
    tendsto_iff_forall_testAgainstNN_tendsto]
  /-
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs F (nhds μ)
    nonzero : Ne μ 0
    ⊢ ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => (Functi …
  -/
  exact fun f ↦ tendsto_normalize_testAgainstNN_of_tendsto μs_lim nonzero f
  /-
    🎉 no goals
  -/


/-- The weak convergence of finite measures to a nonzero limit can be characterized by the weak
convergence of both their normalized versions (probability measures) and their total masses. -/
theorem tendsto_normalize_iff_tendsto {γ : Type*} {F : Filter γ} {μs : γ → FiniteMeasure Ω}
    (nonzero : μ ≠ 0) :
    Tendsto (fun i ↦ (μs i).normalize) F (𝓝 μ.normalize) ∧
        Tendsto (fun i ↦ (μs i).mass) F (𝓝 μ.mass) ↔
      Tendsto μs F (𝓝 μ) := by
  /-
    Ω : Type u_1
    inst✝² : Nonempty Ω
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    nonzero : Ne μ 0
    ⊢ Iff (And (Filter.Tendsto (fun i => (μs i).normalize) F (nhds μ.normalize)) ( …
  -/
  constructor
    /-
      case mp
      Ω : Type u_1
      inst✝² : Nonempty Ω
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      γ : Type u_2
      F : Filter γ
      μs : γ → MeasureTheory.FiniteMeasure Ω
      nonzero : Ne μ 0
      ⊢ And (Filter.Tendsto (fun i => (μs i).normalize) F (nhds μ.normalize)) (Filte …
    -/
  · rintro ⟨normalized_lim, mass_lim⟩
    /-
      case mp.intro
      Ω : Type u_1
      inst✝² : Nonempty Ω
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      γ : Type u_2
      F : Filter γ
      μs : γ → MeasureTheory.FiniteMeasure Ω
      nonzero : Ne μ 0
      normalized_lim : Filter.Tendsto (fun i => (μs i).normalize) F (nhds μ.normalize)
      mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds μ.mass)
      ⊢ Filter.Tendsto μs F (nhds μ)
    -/
    exact tendsto_of_tendsto_normalize_testAgainstNN_of_tendsto_mass normalized_lim mass_lim
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Ω : Type u_1
      inst✝² : Nonempty Ω
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      γ : Type u_2
      F : Filter γ
      μs : γ → MeasureTheory.FiniteMeasure Ω
      nonzero : Ne μ 0
      ⊢ Filter.Tendsto μs F (nhds μ) → And (Filter.Tendsto (fun i => (μs i).normaliz …
    -/
  · intro μs_lim
    /-
      case mpr
      Ω : Type u_1
      inst✝² : Nonempty Ω
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      γ : Type u_2
      F : Filter γ
      μs : γ → MeasureTheory.FiniteMeasure Ω
      nonzero : Ne μ 0
      μs_lim : Filter.Tendsto μs F (nhds μ)
      ⊢ And (Filter.Tendsto (fun i => (μs i).normalize) F (nhds μ.normalize)) (Filte …
    -/
    exact ⟨tendsto_normalize_of_tendsto μs_lim nonzero, μs_lim.mass⟩
    /-
      🎉 no goals
    -/


/-- The push-forward of a probability measure by a measurable function. -/
noncomputable def map (ν : ProbabilityMeasure Ω) {f : Ω → Ω'} (f_aemble : AEMeasurable f ν) :
    ProbabilityMeasure Ω' :=
  ⟨(ν : Measure Ω).map f,
   ⟨by simp only [Measure.map_apply_of_aemeasurable f_aemble MeasurableSet.univ,
                  preimage_univ, measure_univ]⟩⟩


@[simp] lemma toMeasure_map (ν : ProbabilityMeasure Ω) {f : Ω → Ω'} (hf : AEMeasurable f ν) :
    (ν.map hf).toMeasure = ν.toMeasure.map f := rfl


/-- Note that this is an equality of elements of `ℝ≥0∞`. See also
`MeasureTheory.ProbabilityMeasure.map_apply` for the corresponding equality as elements of `ℝ≥0`. -/
lemma map_apply' (ν : ProbabilityMeasure Ω) {f : Ω → Ω'} (f_aemble : AEMeasurable f ν)
    {A : Set Ω'} (A_mble : MeasurableSet A) :
    (ν.map f_aemble : Measure Ω') A = (ν : Measure Ω) (f ⁻¹' A) :=
  Measure.map_apply_of_aemeasurable f_aemble A_mble


lemma map_apply_of_aemeasurable (ν : ProbabilityMeasure Ω) {f : Ω → Ω'}
    (f_aemble : AEMeasurable f ν) {A : Set Ω'} (A_mble : MeasurableSet A) :
    (ν.map f_aemble) A = ν (f ⁻¹' A) := by
  exact (ENNReal.toNNReal_eq_toNNReal_iff' (measure_ne_top _ _) (measure_ne_top _ _)).mpr <|
    ν.map_apply' f_aemble A_mble


lemma map_apply (ν : ProbabilityMeasure Ω) {f : Ω → Ω'} (f_aemble : AEMeasurable f ν)
    {A : Set Ω'} (A_mble : MeasurableSet A) :
    (ν.map f_aemble) A = ν (f ⁻¹' A) :=
  map_apply_of_aemeasurable ν f_aemble A_mble


/-- If `f : X → Y` is continuous and `Y` is equipped with the Borel sigma algebra, then
convergence (in distribution) of `ProbabilityMeasure`s on `X` implies convergence (in
distribution) of the push-forwards of these measures by `f`. -/
lemma tendsto_map_of_tendsto_of_continuous {ι : Type*} {L : Filter ι}
    (νs : ι → ProbabilityMeasure Ω) (ν : ProbabilityMeasure Ω) (lim : Tendsto νs L (𝓝 ν))
    {f : Ω → Ω'} (f_cont : Continuous f) :
    Tendsto (fun i ↦ (νs i).map f_cont.measurable.aemeasurable) L
      (𝓝 (ν.map f_cont.measurable.aemeasurable)) := by
  /-
    Ω : Type u_1
    Ω' : Type u_2
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : MeasurableSpace Ω'
    inst✝³ : TopologicalSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω'
    inst✝ : BorelSpace Ω'
    ι : Type u_3
    L : Filter ι
    νs : ι → MeasureTheory.ProbabilityMeasure Ω
    ν : MeasureTheory.ProbabilityMeasure Ω
    lim : Filter.Tendsto νs L (nhds ν)
    f : Ω → Ω'
    f_cont : Continuous f
    ⊢ Filter.Tendsto (fun i => (νs i).map ⋯) L (nhds (ν.map ⋯))
  -/
  rw [ProbabilityMeasure.tendsto_iff_forall_lintegral_tendsto] at lim ⊢
  /-
    Ω : Type u_1
    Ω' : Type u_2
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : MeasurableSpace Ω'
    inst✝³ : TopologicalSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω'
    inst✝ : BorelSpace Ω'
    ι : Type u_3
    L : Filter ι
    νs : ι → MeasureTheory.ProbabilityMeasure Ω
    ν : MeasureTheory.ProbabilityMeasure Ω
    lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
    f : Ω → Ω'
    f_cont : Continuous f
    ⊢ ∀ (f_1 : BoundedContinuousFunction Ω' NNReal), Filter.Tendsto (fun i => Meas …
  -/
  intro g
  /-
    Ω : Type u_1
    Ω' : Type u_2
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : MeasurableSpace Ω'
    inst✝³ : TopologicalSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω'
    inst✝ : BorelSpace Ω'
    ι : Type u_3
    L : Filter ι
    νs : ι → MeasureTheory.ProbabilityMeasure Ω
    ν : MeasureTheory.ProbabilityMeasure Ω
    lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
    f : Ω → Ω'
    f_cont : Continuous f
    g : BoundedContinuousFunction Ω' NNReal
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral ↑((νs i).map ⋯) fun ω => ↑( …
  -/
  convert lim (g.compContinuous ⟨f, f_cont⟩) <;>
    /-
      case h.e'_3.h
      Ω : Type u_1
      Ω' : Type u_2
      inst✝⁵ : MeasurableSpace Ω
      inst✝⁴ : MeasurableSpace Ω'
      inst✝³ : TopologicalSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω'
      inst✝ : BorelSpace Ω'
      ι : Type u_3
      L : Filter ι
      νs : ι → MeasureTheory.ProbabilityMeasure Ω
      ν : MeasureTheory.ProbabilityMeasure Ω
      lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
      f : Ω → Ω'
      f_cont : Continuous f
      g : BoundedContinuousFunction Ω' NNReal
      x✝ : ι
      ⊢ Eq (MeasureTheory.lintegral ↑((νs x✝).map ⋯) fun ω => ↑(g ω)) (MeasureTheory …
    -/
    /-
      case h.e'_3.h
      Ω : Type u_1
      Ω' : Type u_2
      inst✝⁵ : MeasurableSpace Ω
      inst✝⁴ : MeasurableSpace Ω'
      inst✝³ : TopologicalSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω'
      inst✝ : BorelSpace Ω'
      ι : Type u_3
      L : Filter ι
      νs : ι → MeasureTheory.ProbabilityMeasure Ω
      ν : MeasureTheory.ProbabilityMeasure Ω
      lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
      f : Ω → Ω'
      f_cont : Continuous f
      g : BoundedContinuousFunction Ω' NNReal
      x✝ : ι
      ⊢ Eq (MeasureTheory.lintegral ↑⟨MeasureTheory.Measure.map f ↑(νs x✝), ⋯⟩ fun ω …
    -/
    /-
      case h.e'_3.h
      Ω : Type u_1
      Ω' : Type u_2
      inst✝⁵ : MeasurableSpace Ω
      inst✝⁴ : MeasurableSpace Ω'
      inst✝³ : TopologicalSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω'
      inst✝ : BorelSpace Ω'
      ι : Type u_3
      L : Filter ι
      νs : ι → MeasureTheory.ProbabilityMeasure Ω
      ν : MeasureTheory.ProbabilityMeasure Ω
      lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
      f : Ω → Ω'
      f_cont : Continuous f
      g : BoundedContinuousFunction Ω' NNReal
      x✝ : ι
      ⊢ Measurable fun ω => ↑(g ω)
    -/
    /-
      🎉 no goals
    -/
    refine lintegral_map ?_ f_cont.measurable
    /-
      case h.e'_5.h.e'_3
      Ω : Type u_1
      Ω' : Type u_2
      inst✝⁵ : MeasurableSpace Ω
      inst✝⁴ : MeasurableSpace Ω'
      inst✝³ : TopologicalSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω'
      inst✝ : BorelSpace Ω'
      ι : Type u_3
      L : Filter ι
      νs : ι → MeasureTheory.ProbabilityMeasure Ω
      ν : MeasureTheory.ProbabilityMeasure Ω
      lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
      f : Ω → Ω'
      f_cont : Continuous f
      g : BoundedContinuousFunction Ω' NNReal
      ⊢ Measurable fun ω => ↑(g ω)
    -/
    exact (ENNReal.continuous_coe.comp g.continuous).measurable
    /-
      🎉 no goals
    -/


/-- If `f : X → Y` is continuous and `Y` is equipped with the Borel sigma algebra, then
the push-forward of probability measures `f* : ProbabilityMeasure X → ProbabilityMeasure Y`
is continuous (in the topologies of convergence in distribution). -/
lemma continuous_map {f : Ω → Ω'} (f_cont : Continuous f) :
    Continuous (fun ν ↦ ProbabilityMeasure.map ν f_cont.measurable.aemeasurable) := by
  /-
    Ω : Type u_1
    Ω' : Type u_2
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : MeasurableSpace Ω'
    inst✝³ : TopologicalSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω'
    inst✝ : BorelSpace Ω'
    f : Ω → Ω'
    f_cont : Continuous f
    ⊢ Continuous fun ν => ν.map ⋯
  -/
  rw [continuous_iff_continuousAt]
  /-
    Ω : Type u_1
    Ω' : Type u_2
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : MeasurableSpace Ω'
    inst✝³ : TopologicalSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω'
    inst✝ : BorelSpace Ω'
    f : Ω → Ω'
    f_cont : Continuous f
    ⊢ ∀ (x : MeasureTheory.ProbabilityMeasure Ω), ContinuousAt (fun ν => ν.map ⋯) x
  -/
  exact fun _ ↦ tendsto_map_of_tendsto_of_continuous _ _ continuous_id.continuousAt f_cont
  /-
    🎉 no goals
  -/


