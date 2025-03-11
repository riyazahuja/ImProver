/-- Finite measures are defined as the subtype of measures that have the property of being finite
measures (i.e., their total mass is finite). -/
def _root_.MeasureTheory.FiniteMeasure (Ω : Type*) [MeasurableSpace Ω] : Type _ :=
  { μ : Measure Ω // IsFiniteMeasure μ }

-- Porting note: as with other subtype synonyms (e.g., `ℝ≥0`, we need a new function for the
-- coercion instead of relying on `Subtype.val`.

/-- Coercion from `MeasureTheory.FiniteMeasure Ω` to `MeasureTheory.Measure Ω`. -/
@[coe]
def toMeasure : FiniteMeasure Ω → Measure Ω := Subtype.val


/-- A finite measure can be interpreted as a measure. -/
instance instCoe : Coe (FiniteMeasure Ω) (MeasureTheory.Measure Ω) := { coe := toMeasure }


instance isFiniteMeasure (μ : FiniteMeasure Ω) : IsFiniteMeasure (μ : Measure Ω) := μ.prop


@[simp]
theorem val_eq_toMeasure (ν : FiniteMeasure Ω) : ν.val = (ν : Measure Ω) := rfl


theorem toMeasure_injective : Function.Injective ((↑) : FiniteMeasure Ω → Measure Ω) :=
  Subtype.coe_injective


instance instFunLike : FunLike (FiniteMeasure Ω) (Set Ω) ℝ≥0 where
  coe μ s := ((μ : Measure Ω) s).toNNReal
  coe_injective' μ ν h := toMeasure_injective <| Measure.ext fun s _ ↦ by
    /-
      Ω : Type u_1
      inst✝ : MeasurableSpace Ω
      μ ν : MeasureTheory.FiniteMeasure Ω
      h : Eq ((fun μ s => (↑μ s).toNNReal) μ) ((fun μ s => (↑μ s).toNNReal) ν)
      s : Set Ω
      x✝ : MeasurableSet s
      ⊢ Eq (↑μ s) (↑ν s)
    -/
    simpa [ENNReal.toNNReal_eq_toNNReal_iff, measure_ne_top] using congr_fun h s
    /-
      🎉 no goals
    -/


lemma coeFn_def (μ : FiniteMeasure Ω) : μ = fun s ↦ ((μ : Measure Ω) s).toNNReal := rfl


lemma coeFn_mk (μ : Measure Ω) (hμ) :
    DFunLike.coe (F := FiniteMeasure Ω) ⟨μ, hμ⟩ = fun s ↦ (μ s).toNNReal := rfl


@[simp, norm_cast]
lemma mk_apply (μ : Measure Ω) (hμ) (s : Set Ω) :
    DFunLike.coe (F := FiniteMeasure Ω) ⟨μ, hμ⟩ s = (μ s).toNNReal := rfl


@[simp]
theorem ennreal_coeFn_eq_coeFn_toMeasure (ν : FiniteMeasure Ω) (s : Set Ω) :
    (ν s : ℝ≥0∞) = (ν : Measure Ω) s :=
  ENNReal.coe_toNNReal (measure_lt_top (↑ν) s).ne


@[simp]
theorem null_iff_toMeasure_null (ν : FiniteMeasure Ω) (s : Set Ω) :
    ν s = 0 ↔ (ν : Measure Ω) s = 0 :=
              /-
                Ω : Type u_1
                inst✝ : MeasurableSpace Ω
                ν : MeasureTheory.FiniteMeasure Ω
                s : Set Ω
                h : Eq (ν s) 0
                ⊢ Eq (↑ν s) 0
              -/
  ⟨fun h ↦ by rw [← ennreal_coeFn_eq_coeFn_toMeasure, h, ENNReal.coe_zero],
              /-
                🎉 no goals
              -/
   fun h ↦ congrArg ENNReal.toNNReal h⟩


theorem apply_mono (μ : FiniteMeasure Ω) {s₁ s₂ : Set Ω} (h : s₁ ⊆ s₂) : μ s₁ ≤ μ s₂ :=
  ENNReal.toNNReal_mono (measure_ne_top _ s₂) ((μ : Measure Ω).mono h)


/-- The (total) mass of a finite measure `μ` is `μ univ`, i.e., the cast to `NNReal` of
`(μ : measure Ω) univ`. -/
def mass (μ : FiniteMeasure Ω) : ℝ≥0 := μ univ


@[simp] theorem apply_le_mass (μ : FiniteMeasure Ω) (s : Set Ω) : μ s ≤ μ.mass := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    s : Set Ω
    ⊢ LE.le (μ s) μ.mass
  -/
  simpa using apply_mono μ (subset_univ s)
  /-
    🎉 no goals
  -/


@[simp]
theorem ennreal_mass {μ : FiniteMeasure Ω} : (μ.mass : ℝ≥0∞) = (μ : Measure Ω) univ :=
  ennreal_coeFn_eq_coeFn_toMeasure μ Set.univ


instance instZero : Zero (FiniteMeasure Ω) where zero := ⟨0, MeasureTheory.isFiniteMeasureZero⟩


@[simp, norm_cast] lemma coeFn_zero : ⇑(0 : FiniteMeasure Ω) = 0 := rfl


@[simp]
theorem zero_mass : (0 : FiniteMeasure Ω).mass = 0 := rfl


@[simp]
theorem mass_zero_iff (μ : FiniteMeasure Ω) : μ.mass = 0 ↔ μ = 0 := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Iff (Eq μ.mass 0) (Eq μ 0)
  -/
  refine ⟨fun μ_mass => ?_, fun hμ => by simp only [hμ, zero_mass]⟩
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μ_mass : Eq μ.mass 0
    ⊢ Eq μ 0
  -/
  apply toMeasure_injective
  /-
    case a
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μ_mass : Eq μ.mass 0
    ⊢ Eq ↑μ ↑0
  -/
  apply Measure.measure_univ_eq_zero.mp
  /-
    case a
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μ_mass : Eq μ.mass 0
    ⊢ Eq (↑μ Set.univ) 0
  -/
  rwa [← ennreal_mass, ENNReal.coe_eq_zero]
  /-
    🎉 no goals
  -/


theorem mass_nonzero_iff (μ : FiniteMeasure Ω) : μ.mass ≠ 0 ↔ μ ≠ 0 :=
  not_iff_not.mpr <| FiniteMeasure.mass_zero_iff μ


@[ext]
theorem eq_of_forall_toMeasure_apply_eq (μ ν : FiniteMeasure Ω)
    (h : ∀ s : Set Ω, MeasurableSet s → (μ : Measure Ω) s = (ν : Measure Ω) s) : μ = ν := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    h : ∀ (s : Set Ω), MeasurableSet s → Eq (↑μ s) (↑ν s)
    ⊢ Eq μ ν
  -/
  apply Subtype.ext
  /-
    case a
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    h : ∀ (s : Set Ω), MeasurableSet s → Eq (↑μ s) (↑ν s)
    ⊢ Eq ↑μ ↑ν
  -/
  ext1 s s_mble
  /-
    case a.h
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    h : ∀ (s : Set Ω), MeasurableSet s → Eq (↑μ s) (↑ν s)
    s : Set Ω
    s_mble : MeasurableSet s
    ⊢ Eq (↑μ s) (↑ν s)
  -/
  exact h s s_mble
  /-
    🎉 no goals
  -/


theorem eq_of_forall_apply_eq (μ ν : FiniteMeasure Ω)
    (h : ∀ s : Set Ω, MeasurableSet s → μ s = ν s) : μ = ν := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    h : ∀ (s : Set Ω), MeasurableSet s → Eq (μ s) (ν s)
    ⊢ Eq μ ν
  -/
  ext1 s s_mble
  /-
    case h
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    h : ∀ (s : Set Ω), MeasurableSet s → Eq (μ s) (ν s)
    s : Set Ω
    s_mble : MeasurableSet s
    ⊢ Eq (↑μ s) (↑ν s)
  -/
  simpa [ennreal_coeFn_eq_coeFn_toMeasure] using congr_arg ((↑) : ℝ≥0 → ℝ≥0∞) (h s s_mble)
  /-
    🎉 no goals
  -/


instance instInhabited : Inhabited (FiniteMeasure Ω) := ⟨0⟩


instance instAdd : Add (FiniteMeasure Ω) where add μ ν := ⟨μ + ν, MeasureTheory.isFiniteMeasureAdd⟩


instance instSMul : SMul R (FiniteMeasure Ω) where
  smul (c : R) μ := ⟨c • (μ : Measure Ω), MeasureTheory.isFiniteMeasureSMulOfNNRealTower⟩


@[simp, norm_cast]
theorem toMeasure_zero : ((↑) : FiniteMeasure Ω → Measure Ω) 0 = 0 := rfl

-- Porting note: with `simp` here the `coeFn` lemmas below fall prey to `simpNF`: the LHS simplifies

@[norm_cast]
theorem toMeasure_add (μ ν : FiniteMeasure Ω) : ↑(μ + ν) = (↑μ + ↑ν : Measure Ω) := rfl


@[simp, norm_cast]
theorem toMeasure_smul (c : R) (μ : FiniteMeasure Ω) : ↑(c • μ) = c • (μ : Measure Ω) :=
  rfl


@[simp, norm_cast]
theorem coeFn_add (μ ν : FiniteMeasure Ω) : (⇑(μ + ν) : Set Ω → ℝ≥0) = (⇑μ + ⇑ν : Set Ω → ℝ≥0) := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    ⊢ Eq (⇑(HAdd.hAdd μ ν)) (HAdd.hAdd ⇑μ ⇑ν)
  -/
  funext
  simp only [Pi.add_apply, ← ENNReal.coe_inj, ne_eq, ennreal_coeFn_eq_coeFn_toMeasure,
    ENNReal.coe_add]
  /-
    case h
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    x✝ : Set Ω
    ⊢ Eq (↑(HAdd.hAdd μ ν) x✝) (HAdd.hAdd (↑μ x✝) (↑ν x✝))
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coeFn_smul [IsScalarTower R ℝ≥0 ℝ≥0] (c : R) (μ : FiniteMeasure Ω) :
    (⇑(c • μ) : Set Ω → ℝ≥0) = c • (⇑μ : Set Ω → ℝ≥0) := by
  /-
    Ω : Type u_1
    inst✝⁵ : MeasurableSpace Ω
    R : Type u_2
    inst✝⁴ : SMul R NNReal
    inst✝³ : SMul R ENNReal
    inst✝² : IsScalarTower R NNReal ENNReal
    inst✝¹ : IsScalarTower R ENNReal ENNReal
    inst✝ : IsScalarTower R NNReal NNReal
    c : R
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Eq (⇑(HSMul.hSMul c μ)) (HSMul.hSMul c ⇑μ)
  -/
  funext; simp [← ENNReal.coe_inj, ENNReal.coe_smul]
          /-
            🎉 no goals
          -/


instance instAddCommMonoid : AddCommMonoid (FiniteMeasure Ω) :=
  toMeasure_injective.addCommMonoid _ toMeasure_zero toMeasure_add fun _ _ ↦ toMeasure_smul _ _


/-- Coercion is an `AddMonoidHom`. -/
@[simps]
def toMeasureAddMonoidHom : FiniteMeasure Ω →+ Measure Ω where
  toFun := (↑)
  map_zero' := toMeasure_zero
  map_add' := toMeasure_add


instance {Ω : Type*} [MeasurableSpace Ω] : Module ℝ≥0 (FiniteMeasure Ω) :=
  Function.Injective.module _ toMeasureAddMonoidHom toMeasure_injective toMeasure_smul


@[simp]
theorem smul_apply [IsScalarTower R ℝ≥0 ℝ≥0] (c : R) (μ : FiniteMeasure Ω) (s : Set Ω) :
    (c • μ) s = c • μ s := by
  /-
    Ω : Type u_1
    inst✝⁵ : MeasurableSpace Ω
    R : Type u_2
    inst✝⁴ : SMul R NNReal
    inst✝³ : SMul R ENNReal
    inst✝² : IsScalarTower R NNReal ENNReal
    inst✝¹ : IsScalarTower R ENNReal ENNReal
    inst✝ : IsScalarTower R NNReal NNReal
    c : R
    μ : MeasureTheory.FiniteMeasure Ω
    s : Set Ω
    ⊢ Eq ((HSMul.hSMul c μ) s) (HSMul.hSMul c (μ s))
  -/
  rw [coeFn_smul, Pi.smul_apply]
  /-
    🎉 no goals
  -/


/-- Restrict a finite measure μ to a set A. -/
def restrict (μ : FiniteMeasure Ω) (A : Set Ω) : FiniteMeasure Ω where
  val := (μ : Measure Ω).restrict A
  property := MeasureTheory.isFiniteMeasureRestrict (μ : Measure Ω) A


theorem restrict_measure_eq (μ : FiniteMeasure Ω) (A : Set Ω) :
    (μ.restrict A : Measure Ω) = (μ : Measure Ω).restrict A := rfl


theorem restrict_apply_measure (μ : FiniteMeasure Ω) (A : Set Ω) {s : Set Ω}
    (s_mble : MeasurableSet s) : (μ.restrict A : Measure Ω) s = (μ : Measure Ω) (s ∩ A) :=
  Measure.restrict_apply s_mble


theorem restrict_apply (μ : FiniteMeasure Ω) (A : Set Ω) {s : Set Ω} (s_mble : MeasurableSet s) :
    (μ.restrict A) s = μ (s ∩ A) := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    A s : Set Ω
    s_mble : MeasurableSet s
    ⊢ Eq ((μ.restrict A) s) (μ (Inter.inter s A))
  -/
  apply congr_arg ENNReal.toNNReal
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    A s : Set Ω
    s_mble : MeasurableSet s
    ⊢ Eq (↑(μ.restrict A) s) (↑μ (Inter.inter s A))
  -/
  exact Measure.restrict_apply s_mble
  /-
    🎉 no goals
  -/


theorem restrict_mass (μ : FiniteMeasure Ω) (A : Set Ω) : (μ.restrict A).mass = μ A := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    A : Set Ω
    ⊢ Eq (μ.restrict A).mass (μ A)
  -/
  simp only [mass, restrict_apply μ A MeasurableSet.univ, univ_inter]
  /-
    🎉 no goals
  -/


theorem restrict_eq_zero_iff (μ : FiniteMeasure Ω) (A : Set Ω) : μ.restrict A = 0 ↔ μ A = 0 := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    A : Set Ω
    ⊢ Iff (Eq (μ.restrict A) 0) (Eq (μ A) 0)
  -/
  rw [← mass_zero_iff, restrict_mass]
  /-
    🎉 no goals
  -/


theorem restrict_nonzero_iff (μ : FiniteMeasure Ω) (A : Set Ω) : μ.restrict A ≠ 0 ↔ μ A ≠ 0 := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    A : Set Ω
    ⊢ Iff (Ne (μ.restrict A) 0) (Ne (μ A) 0)
  -/
  rw [← mass_nonzero_iff, restrict_mass]
  /-
    🎉 no goals
  -/


/-- Two finite Borel measures are equal if the integrals of all bounded continuous functions with
respect to both agree. -/
theorem ext_of_forall_lintegral_eq [HasOuterApproxClosed Ω] [BorelSpace Ω]
    {μ ν : FiniteMeasure Ω} (h : ∀ (f : Ω →ᵇ ℝ≥0), ∫⁻ x, f x ∂μ = ∫⁻ x, f x ∂ν) :
    μ = ν := by
  /-
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : BorelSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral ↑μ …
    ⊢ Eq μ ν
  -/
  apply Subtype.ext
  /-
    case a
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : BorelSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral ↑μ …
    ⊢ Eq ↑μ ↑ν
  -/
  change (μ : Measure Ω) = (ν : Measure Ω)
  /-
    case a
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : BorelSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral ↑μ …
    ⊢ Eq ↑μ ↑ν
  -/
  exact ext_of_forall_lintegral_eq_of_IsFiniteMeasure h
  /-
    🎉 no goals
  -/


/-- The pairing of a finite (Borel) measure `μ` with a nonnegative bounded continuous
function is obtained by (Lebesgue) integrating the (test) function against the measure.
This is `MeasureTheory.FiniteMeasure.testAgainstNN`. -/
def testAgainstNN (μ : FiniteMeasure Ω) (f : Ω →ᵇ ℝ≥0) : ℝ≥0 :=
  (∫⁻ ω, f ω ∂(μ : Measure Ω)).toNNReal


@[simp]
theorem testAgainstNN_coe_eq {μ : FiniteMeasure Ω} {f : Ω →ᵇ ℝ≥0} :
    (μ.testAgainstNN f : ℝ≥0∞) = ∫⁻ ω, f ω ∂(μ : Measure Ω) :=
  ENNReal.coe_toNNReal (f.lintegral_lt_top_of_nnreal _).ne


theorem testAgainstNN_const (μ : FiniteMeasure Ω) (c : ℝ≥0) :
    μ.testAgainstNN (BoundedContinuousFunction.const Ω c) = c * μ.mass := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : TopologicalSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    c : NNReal
    ⊢ Eq (μ.testAgainstNN (BoundedContinuousFunction.const Ω c)) (HMul.hMul c μ.ma …
  -/
  simp [← ENNReal.coe_inj]
  /-
    🎉 no goals
  -/


theorem testAgainstNN_mono (μ : FiniteMeasure Ω) {f g : Ω →ᵇ ℝ≥0} (f_le_g : (f : Ω → ℝ≥0) ≤ g) :
    μ.testAgainstNN f ≤ μ.testAgainstNN g := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : TopologicalSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    f g : BoundedContinuousFunction Ω NNReal
    f_le_g : LE.le ⇑f ⇑g
    ⊢ LE.le (μ.testAgainstNN f) (μ.testAgainstNN g)
  -/
  simp only [← ENNReal.coe_le_coe, testAgainstNN_coe_eq]
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : TopologicalSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    f g : BoundedContinuousFunction Ω NNReal
    f_le_g : LE.le ⇑f ⇑g
    ⊢ LE.le (MeasureTheory.lintegral ↑μ fun ω => ↑(f ω)) (MeasureTheory.lintegral  …
  -/
  gcongr
  /-
    case hfg.a
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : TopologicalSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    f g : BoundedContinuousFunction Ω NNReal
    f_le_g : LE.le ⇑f ⇑g
    x✝ : Ω
    ⊢ LE.le (f x✝) (g x✝)
  -/
  apply f_le_g
  /-
    🎉 no goals
  -/


@[simp]
theorem testAgainstNN_zero (μ : FiniteMeasure Ω) : μ.testAgainstNN 0 = 0 := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : TopologicalSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Eq (μ.testAgainstNN 0) 0
  -/
  simpa only [zero_mul] using μ.testAgainstNN_const 0
  /-
    🎉 no goals
  -/


@[simp]
theorem testAgainstNN_one (μ : FiniteMeasure Ω) : μ.testAgainstNN 1 = μ.mass := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : TopologicalSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Eq (μ.testAgainstNN 1) μ.mass
  -/
  simp only [testAgainstNN, coe_one, Pi.one_apply, ENNReal.coe_one, lintegral_one]
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : TopologicalSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Eq (↑μ Set.univ).toNNReal μ.mass
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_testAgainstNN_apply (f : Ω →ᵇ ℝ≥0) : (0 : FiniteMeasure Ω).testAgainstNN f = 0 := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : TopologicalSpace Ω
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Eq (MeasureTheory.FiniteMeasure.testAgainstNN 0 f) 0
  -/
  simp only [testAgainstNN, toMeasure_zero, lintegral_zero_measure, ENNReal.zero_toNNReal]
  /-
    🎉 no goals
  -/


theorem zero_testAgainstNN : (0 : FiniteMeasure Ω).testAgainstNN = 0 := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : TopologicalSpace Ω
    ⊢ Eq (MeasureTheory.FiniteMeasure.testAgainstNN 0) 0
  -/
  funext
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : TopologicalSpace Ω
    x✝ : BoundedContinuousFunction Ω NNReal
    ⊢ Eq (MeasureTheory.FiniteMeasure.testAgainstNN 0 x✝) (0 x✝)
  -/
  simp only [zero_testAgainstNN_apply, Pi.zero_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_testAgainstNN_apply (c : ℝ≥0) (μ : FiniteMeasure Ω) (f : Ω →ᵇ ℝ≥0) :
    (c • μ).testAgainstNN f = c • μ.testAgainstNN f := by
  simp only [testAgainstNN, toMeasure_smul, smul_eq_mul, ← ENNReal.smul_toNNReal, ENNReal.smul_def,
    lintegral_smul_measure]


theorem testAgainstNN_add (μ : FiniteMeasure Ω) (f₁ f₂ : Ω →ᵇ ℝ≥0) :
    μ.testAgainstNN (f₁ + f₂) = μ.testAgainstNN f₁ + μ.testAgainstNN f₂ := by
  simp only [← ENNReal.coe_inj, BoundedContinuousFunction.coe_add, ENNReal.coe_add, Pi.add_apply,
    testAgainstNN_coe_eq]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    f₁ f₂ : BoundedContinuousFunction Ω NNReal
    ⊢ Eq (MeasureTheory.lintegral ↑μ fun ω => HAdd.hAdd ↑(f₁ ω) ↑(f₂ ω)) (HAdd.hAd …
  -/
  exact lintegral_add_left (BoundedContinuousFunction.measurable_coe_ennreal_comp _) _
  /-
    🎉 no goals
  -/


theorem testAgainstNN_smul [IsScalarTower R ℝ≥0 ℝ≥0] [PseudoMetricSpace R] [Zero R]
    [BoundedSMul R ℝ≥0] (μ : FiniteMeasure Ω) (c : R) (f : Ω →ᵇ ℝ≥0) :
    μ.testAgainstNN (c • f) = c • μ.testAgainstNN f := by
  simp only [← ENNReal.coe_inj, BoundedContinuousFunction.coe_smul, testAgainstNN_coe_eq,
    ENNReal.coe_smul]
  simp_rw [← smul_one_smul ℝ≥0∞ c (f _ : ℝ≥0∞), ← smul_one_smul ℝ≥0∞ c (lintegral _ _ : ℝ≥0∞),
    smul_eq_mul]
  /-
    Ω : Type u_1
    inst✝¹⁰ : MeasurableSpace Ω
    R : Type u_2
    inst✝⁹ : SMul R NNReal
    inst✝⁸ : SMul R ENNReal
    inst✝⁷ : IsScalarTower R NNReal ENNReal
    inst✝⁶ : IsScalarTower R ENNReal ENNReal
    inst✝⁵ : TopologicalSpace Ω
    inst✝⁴ : OpensMeasurableSpace Ω
    inst✝³ : IsScalarTower R NNReal NNReal
    inst✝² : PseudoMetricSpace R
    inst✝¹ : Zero R
    inst✝ : BoundedSMul R NNReal
    μ : MeasureTheory.FiniteMeasure Ω
    c : R
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Eq (MeasureTheory.lintegral ↑μ fun ω => HMul.hMul (HSMul.hSMul c 1) ↑(f ω))  …
  -/
  exact lintegral_const_mul (c • (1 : ℝ≥0∞)) f.measurable_coe_ennreal_comp
  /-
    🎉 no goals
  -/


theorem testAgainstNN_lipschitz_estimate (μ : FiniteMeasure Ω) (f g : Ω →ᵇ ℝ≥0) :
    μ.testAgainstNN f ≤ μ.testAgainstNN g + nndist f g * μ.mass := by
  simp only [← μ.testAgainstNN_const (nndist f g), ← testAgainstNN_add, ← ENNReal.coe_le_coe,
    BoundedContinuousFunction.coe_add, const_apply, ENNReal.coe_add, Pi.add_apply,
    coe_nnreal_ennreal_nndist, testAgainstNN_coe_eq]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    f g : BoundedContinuousFunction Ω NNReal
    ⊢ LE.le (MeasureTheory.lintegral ↑μ fun ω => ↑(f ω)) (MeasureTheory.lintegral  …
  -/
  apply lintegral_mono
  /-
    case hfg
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    f g : BoundedContinuousFunction Ω NNReal
    ⊢ LE.le (fun a => ↑(f a)) fun a => HAdd.hAdd (↑(g a)) (EDist.edist f g)
  -/
  have le_dist : ∀ ω, dist (f ω) (g ω) ≤ nndist f g := BoundedContinuousFunction.dist_coe_le_dist
  /-
    case hfg
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    f g : BoundedContinuousFunction Ω NNReal
    le_dist : ∀ (ω : Ω), LE.le (Dist.dist (f ω) (g ω)) ↑(NNDist.nndist f g)
    ⊢ LE.le (fun a => ↑(f a)) fun a => HAdd.hAdd (↑(g a)) (EDist.edist f g)
  -/
  intro ω
  have le' : f ω ≤ g ω + nndist f g := by
    calc f ω
     _ ≤ g ω + nndist (f ω) (g ω)     := NNReal.le_add_nndist (f ω) (g ω)
     _ ≤ g ω + nndist f g             := (add_le_add_iff_left (g ω)).mpr (le_dist ω)
  have le : (f ω : ℝ≥0∞) ≤ (g ω : ℝ≥0∞) + nndist f g := by
    simpa only [← ENNReal.coe_add] using (by exact_mod_cast le')
  /-
    case hfg
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    f g : BoundedContinuousFunction Ω NNReal
    le_dist : ∀ (ω : Ω), LE.le (Dist.dist (f ω) (g ω)) ↑(NNDist.nndist f g)
    ω : Ω
    le' : LE.le (f ω) (HAdd.hAdd (g ω) (NNDist.nndist f g))
    le : LE.le (↑(f ω)) (HAdd.hAdd ↑(g ω) ↑(NNDist.nndist f g))
    ⊢ LE.le ((fun a => ↑(f a)) ω) ((fun a => HAdd.hAdd (↑(g a)) (EDist.edist f g)) …
  -/
  rwa [coe_nnreal_ennreal_nndist] at le
  /-
    🎉 no goals
  -/


theorem testAgainstNN_lipschitz (μ : FiniteMeasure Ω) :
    LipschitzWith μ.mass fun f : Ω →ᵇ ℝ≥0 ↦ μ.testAgainstNN f := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ LipschitzWith μ.mass fun f => μ.testAgainstNN f
  -/
  rw [lipschitzWith_iff_dist_le_mul]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ ∀ (x y : BoundedContinuousFunction Ω NNReal), LE.le (Dist.dist (μ.testAgains …
  -/
  intro f₁ f₂
  suffices abs (μ.testAgainstNN f₁ - μ.testAgainstNN f₂ : ℝ) ≤ μ.mass * dist f₁ f₂ by
    rwa [NNReal.dist_eq]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    f₁ f₂ : BoundedContinuousFunction Ω NNReal
    ⊢ LE.le (abs (HSub.hSub ↑(μ.testAgainstNN f₁) ↑(μ.testAgainstNN f₂))) (HMul.hM …
  -/
  apply abs_le.mpr
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    f₁ f₂ : BoundedContinuousFunction Ω NNReal
    ⊢ And (LE.le (Neg.neg (HMul.hMul (↑μ.mass) (Dist.dist f₁ f₂))) (HSub.hSub ↑(μ. …
  -/
  constructor
    /-
      case left
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      f₁ f₂ : BoundedContinuousFunction Ω NNReal
      ⊢ LE.le (Neg.neg (HMul.hMul (↑μ.mass) (Dist.dist f₁ f₂))) (HSub.hSub ↑(μ.testA …
    -/
  · have key := μ.testAgainstNN_lipschitz_estimate f₂ f₁
    /-
      case left
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      f₁ f₂ : BoundedContinuousFunction Ω NNReal
      key : LE.le (μ.testAgainstNN f₂) (HAdd.hAdd (μ.testAgainstNN f₁) (HMul.hMul (N …
      ⊢ LE.le (Neg.neg (HMul.hMul (↑μ.mass) (Dist.dist f₁ f₂))) (HSub.hSub ↑(μ.testA …
    -/
    rw [mul_comm] at key
    /-
      case left
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      f₁ f₂ : BoundedContinuousFunction Ω NNReal
      key : LE.le (μ.testAgainstNN f₂) (HAdd.hAdd (μ.testAgainstNN f₁) (HMul.hMul μ. …
      ⊢ LE.le (Neg.neg (HMul.hMul (↑μ.mass) (Dist.dist f₁ f₂))) (HSub.hSub ↑(μ.testA …
    -/
    suffices ↑(μ.testAgainstNN f₂) ≤ ↑(μ.testAgainstNN f₁) + ↑μ.mass * dist f₁ f₂ by linarith
    /-
      case left
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      f₁ f₂ : BoundedContinuousFunction Ω NNReal
      key : LE.le (μ.testAgainstNN f₂) (HAdd.hAdd (μ.testAgainstNN f₁) (HMul.hMul μ. …
      ⊢ LE.le (↑(μ.testAgainstNN f₂)) (HAdd.hAdd (↑(μ.testAgainstNN f₁)) (HMul.hMul  …
    -/
    simpa [nndist_comm] using NNReal.coe_mono key
    /-
      🎉 no goals
    -/
    /-
      case right
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      f₁ f₂ : BoundedContinuousFunction Ω NNReal
      ⊢ LE.le (HSub.hSub ↑(μ.testAgainstNN f₁) ↑(μ.testAgainstNN f₂)) (HMul.hMul (↑μ …
    -/
  · have key := μ.testAgainstNN_lipschitz_estimate f₁ f₂
    /-
      case right
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      f₁ f₂ : BoundedContinuousFunction Ω NNReal
      key : LE.le (μ.testAgainstNN f₁) (HAdd.hAdd (μ.testAgainstNN f₂) (HMul.hMul (N …
      ⊢ LE.le (HSub.hSub ↑(μ.testAgainstNN f₁) ↑(μ.testAgainstNN f₂)) (HMul.hMul (↑μ …
    -/
    rw [mul_comm] at key
    /-
      case right
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      f₁ f₂ : BoundedContinuousFunction Ω NNReal
      key : LE.le (μ.testAgainstNN f₁) (HAdd.hAdd (μ.testAgainstNN f₂) (HMul.hMul μ. …
      ⊢ LE.le (HSub.hSub ↑(μ.testAgainstNN f₁) ↑(μ.testAgainstNN f₂)) (HMul.hMul (↑μ …
    -/
    suffices ↑(μ.testAgainstNN f₁) ≤ ↑(μ.testAgainstNN f₂) + ↑μ.mass * dist f₁ f₂ by linarith
    /-
      case right
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : TopologicalSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      f₁ f₂ : BoundedContinuousFunction Ω NNReal
      key : LE.le (μ.testAgainstNN f₁) (HAdd.hAdd (μ.testAgainstNN f₂) (HMul.hMul μ. …
      ⊢ LE.le (↑(μ.testAgainstNN f₁)) (HAdd.hAdd (↑(μ.testAgainstNN f₂)) (HMul.hMul  …
    -/
    simpa using NNReal.coe_mono key
    /-
      🎉 no goals
    -/


/-- Finite measures yield elements of the `WeakDual` of bounded continuous nonnegative
functions via `MeasureTheory.FiniteMeasure.testAgainstNN`, i.e., integration. -/
def toWeakDualBCNN (μ : FiniteMeasure Ω) : WeakDual ℝ≥0 (Ω →ᵇ ℝ≥0) where
  toFun f := μ.testAgainstNN f
  map_add' := testAgainstNN_add μ
  map_smul' := testAgainstNN_smul μ
  cont := μ.testAgainstNN_lipschitz.continuous


@[simp]
theorem coe_toWeakDualBCNN (μ : FiniteMeasure Ω) : ⇑μ.toWeakDualBCNN = μ.testAgainstNN :=
  rfl


@[simp]
theorem toWeakDualBCNN_apply (μ : FiniteMeasure Ω) (f : Ω →ᵇ ℝ≥0) :
    μ.toWeakDualBCNN f = (∫⁻ x, f x ∂(μ : Measure Ω)).toNNReal := rfl


/-- The topology of weak convergence on `MeasureTheory.FiniteMeasure Ω` is inherited (induced)
from the weak-* topology on `WeakDual ℝ≥0 (Ω →ᵇ ℝ≥0)` via the function
`MeasureTheory.FiniteMeasure.toWeakDualBCNN`. -/
instance instTopologicalSpace : TopologicalSpace (FiniteMeasure Ω) :=
  TopologicalSpace.induced toWeakDualBCNN inferInstance


theorem toWeakDualBCNN_continuous : Continuous (@toWeakDualBCNN Ω _ _ _) :=
  continuous_induced_dom


/-- Integration of (nonnegative bounded continuous) test functions against finite Borel measures
depends continuously on the measure. -/
theorem continuous_testAgainstNN_eval (f : Ω →ᵇ ℝ≥0) :
    Continuous fun μ : FiniteMeasure Ω ↦ μ.testAgainstNN f := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Continuous fun μ => μ.testAgainstNN f
  -/
  show Continuous ((fun φ : WeakDual ℝ≥0 (Ω →ᵇ ℝ≥0) ↦ φ f) ∘ toWeakDualBCNN)
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Continuous (Function.comp (fun φ => φ f) MeasureTheory.FiniteMeasure.toWeakD …
  -/
  refine Continuous.comp ?_ (toWeakDualBCNN_continuous (Ω := Ω))
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Continuous fun φ => φ f
  -/
  exact WeakBilin.eval_continuous _ _
  /-
    🎉 no goals
  -/


/-- The total mass of a finite measure depends continuously on the measure. -/
theorem continuous_mass : Continuous fun μ : FiniteMeasure Ω ↦ μ.mass := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    ⊢ Continuous fun μ => μ.mass
  -/
  simp_rw [← testAgainstNN_one]; exact continuous_testAgainstNN_eval 1
                                 /-
                                   🎉 no goals
                                 -/


/-- Convergence of finite measures implies the convergence of their total masses. -/
theorem _root_.Filter.Tendsto.mass {γ : Type*} {F : Filter γ} {μs : γ → FiniteMeasure Ω}
    {μ : FiniteMeasure Ω} (h : Tendsto μs F (𝓝 μ)) : Tendsto (fun i ↦ (μs i).mass) F (𝓝 μ.mass) :=
  (continuous_mass.tendsto μ).comp h


theorem tendsto_iff_weakDual_tendsto {γ : Type*} {F : Filter γ} {μs : γ → FiniteMeasure Ω}
    {μ : FiniteMeasure Ω} :
    Tendsto μs F (𝓝 μ) ↔ Tendsto (fun i ↦ (μs i).toWeakDualBCNN) F (𝓝 μ.toWeakDualBCNN) :=
  IsInducing.tendsto_nhds_iff ⟨rfl⟩


theorem tendsto_iff_forall_toWeakDualBCNN_tendsto {γ : Type*} {F : Filter γ}
    {μs : γ → FiniteMeasure Ω} {μ : FiniteMeasure Ω} :
    Tendsto μs F (𝓝 μ) ↔
      ∀ f : Ω →ᵇ ℝ≥0, Tendsto (fun i ↦ (μs i).toWeakDualBCNN f) F (𝓝 (μ.toWeakDualBCNN f)) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Iff (Filter.Tendsto μs F (nhds μ)) (∀ (f : BoundedContinuousFunction Ω NNRea …
  -/
  rw [tendsto_iff_weakDual_tendsto, tendsto_iff_forall_eval_tendsto_topDualPairing]; rfl
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem tendsto_iff_forall_testAgainstNN_tendsto {γ : Type*} {F : Filter γ}
    {μs : γ → FiniteMeasure Ω} {μ : FiniteMeasure Ω} :
    Tendsto μs F (𝓝 μ) ↔
      ∀ f : Ω →ᵇ ℝ≥0, Tendsto (fun i ↦ (μs i).testAgainstNN f) F (𝓝 (μ.testAgainstNN f)) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Iff (Filter.Tendsto μs F (nhds μ)) (∀ (f : BoundedContinuousFunction Ω NNRea …
  -/
  rw [FiniteMeasure.tendsto_iff_forall_toWeakDualBCNN_tendsto]; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- If the total masses of finite measures tend to zero, then the measures tend to
zero. This formulation concerns the associated functionals on bounded continuous
nonnegative test functions. See `MeasureTheory.FiniteMeasure.tendsto_zero_of_tendsto_zero_mass` for
a formulation stating the weak convergence of measures. -/
theorem tendsto_zero_testAgainstNN_of_tendsto_zero_mass {γ : Type*} {F : Filter γ}
    {μs : γ → FiniteMeasure Ω} (mass_lim : Tendsto (fun i ↦ (μs i).mass) F (𝓝 0)) (f : Ω →ᵇ ℝ≥0) :
    Tendsto (fun i ↦ (μs i).testAgainstNN f) F (𝓝 0) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds 0)
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Filter.Tendsto (fun i => (μs i).testAgainstNN f) F (nhds 0)
  -/
  apply tendsto_iff_dist_tendsto_zero.mpr
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds 0)
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Filter.Tendsto (fun b => Dist.dist ((μs b).testAgainstNN f) 0) F (nhds 0)
  -/
  have obs := fun i ↦ (μs i).testAgainstNN_lipschitz_estimate f 0
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds 0)
    f : BoundedContinuousFunction Ω NNReal
    obs : ∀ (i : γ), LE.le ((μs i).testAgainstNN f) (HAdd.hAdd ((μs i).testAgainst …
    ⊢ Filter.Tendsto (fun b => Dist.dist ((μs b).testAgainstNN f) 0) F (nhds 0)
  -/
  simp_rw [testAgainstNN_zero, zero_add] at obs
  simp_rw [show ∀ i, dist ((μs i).testAgainstNN f) 0 = (μs i).testAgainstNN f by
      simp only [dist_nndist, NNReal.nndist_zero_eq_val', eq_self_iff_true, imp_true_iff]]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds 0)
    f : BoundedContinuousFunction Ω NNReal
    obs : ∀ (i : γ), LE.le ((μs i).testAgainstNN f) (HMul.hMul (NNDist.nndist f 0) …
    ⊢ Filter.Tendsto (fun b => ↑((μs b).testAgainstNN f)) F (nhds 0)
  -/
  apply squeeze_zero (fun i ↦ NNReal.coe_nonneg _) obs
  have lim_pair : Tendsto (fun i ↦ (⟨nndist f 0, (μs i).mass⟩ : ℝ × ℝ)) F (𝓝 ⟨nndist f 0, 0⟩) :=
    (Prod.tendsto_iff _ _).mpr ⟨tendsto_const_nhds, (NNReal.continuous_coe.tendsto 0).comp mass_lim⟩
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds 0)
    f : BoundedContinuousFunction Ω NNReal
    obs : ∀ (i : γ), LE.le ((μs i).testAgainstNN f) (HMul.hMul (NNDist.nndist f 0) …
    lim_pair : Filter.Tendsto (fun i => { fst := ↑(NNDist.nndist f 0), snd := ↑(μs …
    ⊢ Filter.Tendsto (fun t => (fun a => ↑a) (HMul.hMul (NNDist.nndist f 0) (μs t) …
  -/
  simpa using tendsto_mul.comp lim_pair
  /-
    🎉 no goals
  -/


/-- If the total masses of finite measures tend to zero, then the measures tend to zero. -/
theorem tendsto_zero_of_tendsto_zero_mass {γ : Type*} {F : Filter γ} {μs : γ → FiniteMeasure Ω}
    (mass_lim : Tendsto (fun i ↦ (μs i).mass) F (𝓝 0)) : Tendsto μs F (𝓝 0) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds 0)
    ⊢ Filter.Tendsto μs F (nhds 0)
  -/
  rw [tendsto_iff_forall_testAgainstNN_tendsto]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds 0)
    ⊢ ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => (μs i). …
  -/
  intro f
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds 0)
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Filter.Tendsto (fun i => (μs i).testAgainstNN f) F (nhds (MeasureTheory.Fini …
  -/
  convert tendsto_zero_testAgainstNN_of_tendsto_zero_mass mass_lim f
  /-
    case h.e'_5.h.e'_3
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    mass_lim : Filter.Tendsto (fun i => (μs i).mass) F (nhds 0)
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Eq (MeasureTheory.FiniteMeasure.testAgainstNN 0 f) 0
  -/
  rw [zero_testAgainstNN_apply]
  /-
    🎉 no goals
  -/


/-- A characterization of weak convergence in terms of integrals of bounded continuous
nonnegative functions. -/
theorem tendsto_iff_forall_lintegral_tendsto {γ : Type*} {F : Filter γ} {μs : γ → FiniteMeasure Ω}
    {μ : FiniteMeasure Ω} :
    Tendsto μs F (𝓝 μ) ↔
      ∀ f : Ω →ᵇ ℝ≥0,
        Tendsto (fun i ↦ ∫⁻ x, f x ∂(μs i : Measure Ω)) F (𝓝 (∫⁻ x, f x ∂(μ : Measure Ω))) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_3
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Iff (Filter.Tendsto μs F (nhds μ)) (∀ (f : BoundedContinuousFunction Ω NNRea …
  -/
  rw [tendsto_iff_forall_toWeakDualBCNN_tendsto]
  simp_rw [toWeakDualBCNN_apply _ _, ← testAgainstNN_coe_eq, ENNReal.tendsto_coe,
    ENNReal.toNNReal_coe]


/-- The mapping `toWeakDualBCNN` from finite Borel measures to the weak dual of `Ω →ᵇ ℝ≥0` is
injective, if in the underlying space `Ω`, indicator functions of closed sets have decreasing
approximations by sequences of continuous functions (in particular if `Ω` is pseudometrizable). -/
lemma injective_toWeakDualBCNN :
    Injective (toWeakDualBCNN : FiniteMeasure Ω → WeakDual ℝ≥0 (Ω →ᵇ ℝ≥0)) := by
  /-
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : BorelSpace Ω
    ⊢ Function.Injective MeasureTheory.FiniteMeasure.toWeakDualBCNN
  -/
  intro μ ν hμν
  /-
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : BorelSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    hμν : Eq μ.toWeakDualBCNN ν.toWeakDualBCNN
    ⊢ Eq μ ν
  -/
  apply ext_of_forall_lintegral_eq
  /-
    case h
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : BorelSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    hμν : Eq μ.toWeakDualBCNN ν.toWeakDualBCNN
    ⊢ ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral ↑μ f …
  -/
  intro f
  /-
    case h
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : BorelSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    hμν : Eq μ.toWeakDualBCNN ν.toWeakDualBCNN
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Eq (MeasureTheory.lintegral ↑μ fun x => ↑(f x)) (MeasureTheory.lintegral ↑ν  …
  -/
  have key := congr_fun (congrArg DFunLike.coe hμν) f
  /-
    case h
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : BorelSpace Ω
    μ ν : MeasureTheory.FiniteMeasure Ω
    hμν : Eq μ.toWeakDualBCNN ν.toWeakDualBCNN
    f : BoundedContinuousFunction Ω NNReal
    key : Eq (μ.toWeakDualBCNN f) (ν.toWeakDualBCNN f)
    ⊢ Eq (MeasureTheory.lintegral ↑μ fun x => ↑(f x)) (MeasureTheory.lintegral ↑ν  …
  -/
  apply (ENNReal.toNNReal_eq_toNNReal_iff' ?_ ?_).mp key
    /-
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : TopologicalSpace Ω
      inst✝¹ : HasOuterApproxClosed Ω
      inst✝ : BorelSpace Ω
      μ ν : MeasureTheory.FiniteMeasure Ω
      hμν : Eq μ.toWeakDualBCNN ν.toWeakDualBCNN
      f : BoundedContinuousFunction Ω NNReal
      key : Eq (μ.toWeakDualBCNN f) (ν.toWeakDualBCNN f)
      ⊢ Ne (MeasureTheory.lintegral ↑μ fun ω => ↑(f ω)) Top.top
    -/
  · exact (lintegral_lt_top_of_nnreal μ f).ne
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : TopologicalSpace Ω
      inst✝¹ : HasOuterApproxClosed Ω
      inst✝ : BorelSpace Ω
      μ ν : MeasureTheory.FiniteMeasure Ω
      hμν : Eq μ.toWeakDualBCNN ν.toWeakDualBCNN
      f : BoundedContinuousFunction Ω NNReal
      key : Eq (μ.toWeakDualBCNN f) (ν.toWeakDualBCNN f)
      ⊢ Ne (MeasureTheory.lintegral ↑ν fun ω => ↑(f ω)) Top.top
    -/
  · exact (lintegral_lt_top_of_nnreal ν f).ne
    /-
      🎉 no goals
    -/


lemma isEmbedding_toWeakDualBCNN :
    IsEmbedding (toWeakDualBCNN : FiniteMeasure Ω → WeakDual ℝ≥0 (Ω →ᵇ ℝ≥0)) where
  eq_induced := rfl
  injective := injective_toWeakDualBCNN


@[deprecated (since := "2024-10-26")]
alias embedding_toWeakDualBCNN := isEmbedding_toWeakDualBCNN


/-- On topological spaces where indicators of closed sets have decreasing approximating sequences of
continuous functions (`HasOuterApproxClosed`), the topology of weak convergence of finite Borel
measures is Hausdorff (`T2Space`). -/
instance t2Space : T2Space (FiniteMeasure Ω) := (isEmbedding_toWeakDualBCNN Ω).t2Space


/-- A bounded convergence theorem for a finite measure:
If a sequence of bounded continuous non-negative functions are uniformly bounded by a constant
and tend pointwise to a limit, then their integrals (`MeasureTheory.lintegral`) against the finite
measure tend to the integral of the limit.

A related result with more general assumptions is
`MeasureTheory.tendsto_lintegral_nn_filter_of_le_const`.
-/
theorem tendsto_lintegral_nn_of_le_const (μ : FiniteMeasure Ω) {fs : ℕ → Ω →ᵇ ℝ≥0} {c : ℝ≥0}
    (fs_le_const : ∀ n ω, fs n ω ≤ c) {f : Ω → ℝ≥0}
    (fs_lim : ∀ ω, Tendsto (fun n ↦ fs n ω) atTop (𝓝 (f ω))) :
    Tendsto (fun n ↦ ∫⁻ ω, fs n ω ∂(μ : Measure Ω)) atTop (𝓝 (∫⁻ ω, f ω ∂(μ : Measure Ω))) :=
  tendsto_lintegral_nn_filter_of_le_const μ
    (.of_forall fun n ↦ .of_forall (fs_le_const n))
    (.of_forall fs_lim)


/-- A bounded convergence theorem for a finite measure:
If bounded continuous non-negative functions are uniformly bounded by a constant and tend to a
limit, then their integrals against the finite measure tend to the integral of the limit.
This formulation assumes:
 * the functions tend to a limit along a countably generated filter;
 * the limit is in the almost everywhere sense;
 * boundedness holds almost everywhere;
 * integration is the pairing against non-negative continuous test functions
   (`MeasureTheory.FiniteMeasure.testAgainstNN`).

A related result using `MeasureTheory.lintegral` for integration is
`MeasureTheory.FiniteMeasure.tendsto_lintegral_nn_filter_of_le_const`.
-/
theorem tendsto_testAgainstNN_filter_of_le_const {ι : Type*} {L : Filter ι}
    [L.IsCountablyGenerated] {μ : FiniteMeasure Ω} {fs : ι → Ω →ᵇ ℝ≥0} {c : ℝ≥0}
    (fs_le_const : ∀ᶠ i in L, ∀ᵐ ω : Ω ∂(μ : Measure Ω), fs i ω ≤ c) {f : Ω →ᵇ ℝ≥0}
    (fs_lim : ∀ᵐ ω : Ω ∂(μ : Measure Ω), Tendsto (fun i ↦ fs i ω) L (𝓝 (f ω))) :
    Tendsto (fun i ↦ μ.testAgainstNN (fs i)) L (𝓝 (μ.testAgainstNN f)) := by
  /-
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    inst✝ : L.IsCountablyGenerated
    μ : MeasureTheory.FiniteMeasure Ω
    fs : ι → BoundedContinuousFunction Ω NNReal
    c : NNReal
    fs_le_const : Filter.Eventually (fun i => Filter.Eventually (fun ω => LE.le (( …
    f : BoundedContinuousFunction Ω NNReal
    fs_lim : Filter.Eventually (fun ω => Filter.Tendsto (fun i => (fs i) ω) L (nhd …
    ⊢ Filter.Tendsto (fun i => μ.testAgainstNN (fs i)) L (nhds (μ.testAgainstNN f))
  -/
  apply (ENNReal.tendsto_toNNReal (f.lintegral_lt_top_of_nnreal (μ : Measure Ω)).ne).comp
  /-
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    inst✝ : L.IsCountablyGenerated
    μ : MeasureTheory.FiniteMeasure Ω
    fs : ι → BoundedContinuousFunction Ω NNReal
    c : NNReal
    fs_le_const : Filter.Eventually (fun i => Filter.Eventually (fun ω => LE.le (( …
    f : BoundedContinuousFunction Ω NNReal
    fs_lim : Filter.Eventually (fun ω => Filter.Tendsto (fun i => (fs i) ω) L (nhd …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral ↑μ fun ω => ↑((fs i) ω)) L  …
  -/
  exact tendsto_lintegral_nn_filter_of_le_const (Ω := Ω) μ fs_le_const fs_lim
  /-
    🎉 no goals
  -/


/-- A bounded convergence theorem for a finite measure:
If a sequence of bounded continuous non-negative functions are uniformly bounded by a constant and
tend pointwise to a limit, then their integrals (`MeasureTheory.FiniteMeasure.testAgainstNN`)
against the finite measure tend to the integral of the limit.

Related results:
 * `MeasureTheory.FiniteMeasure.tendsto_testAgainstNN_filter_of_le_const`:
   more general assumptions
 * `MeasureTheory.FiniteMeasure.tendsto_lintegral_nn_of_le_const`:
   using `MeasureTheory.lintegral` for integration.
-/
theorem tendsto_testAgainstNN_of_le_const {μ : FiniteMeasure Ω} {fs : ℕ → Ω →ᵇ ℝ≥0} {c : ℝ≥0}
    (fs_le_const : ∀ n ω, fs n ω ≤ c) {f : Ω →ᵇ ℝ≥0}
    (fs_lim : ∀ ω, Tendsto (fun n ↦ fs n ω) atTop (𝓝 (f ω))) :
    Tendsto (fun n ↦ μ.testAgainstNN (fs n)) atTop (𝓝 (μ.testAgainstNN f)) :=
  tendsto_testAgainstNN_filter_of_le_const
    (.of_forall fun n ↦ .of_forall (fs_le_const n))
    (.of_forall fs_lim)


theorem tendsto_of_forall_integral_tendsto {γ : Type*} {F : Filter γ} {μs : γ → FiniteMeasure Ω}
    {μ : FiniteMeasure Ω}
    (h : ∀ f : Ω →ᵇ ℝ,
          Tendsto (fun i ↦ ∫ x, f x ∂(μs i : Measure Ω)) F (𝓝 (∫ x, f x ∂(μ : Measure Ω)))) :
    Tendsto μs F (𝓝 μ) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => Measure …
    ⊢ Filter.Tendsto μs F (nhds μ)
  -/
  apply tendsto_iff_forall_lintegral_tendsto.mpr
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => Measure …
    ⊢ ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Measure …
  -/
  intro f
  apply (ENNReal.tendsto_toReal_iff (fi := F)
      (fun i ↦ (f.lintegral_lt_top_of_nnreal (μs i)).ne) (f.lintegral_lt_top_of_nnreal μ).ne).mp
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => Measure …
    f : BoundedContinuousFunction Ω NNReal
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral ↑(μs n) fun x => ↑(f x)).t …
  -/
  have lip : LipschitzWith 1 ((↑) : ℝ≥0 → ℝ) := isometry_subtype_coe.lipschitz
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => Measure …
    f : BoundedContinuousFunction Ω NNReal
    lip : LipschitzWith 1 NNReal.toReal
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral ↑(μs n) fun x => ↑(f x)).t …
  -/
  set f₀ := BoundedContinuousFunction.comp _ lip f with _def_f₀
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => Measure …
    f : BoundedContinuousFunction Ω NNReal
    lip : LipschitzWith 1 NNReal.toReal
    f₀ : BoundedContinuousFunction Ω Real := BoundedContinuousFunction.comp NNReal …
    _def_f₀ : Eq f₀ (BoundedContinuousFunction.comp NNReal.toReal lip f)
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral ↑(μs n) fun x => ↑(f x)).t …
  -/
  have f₀_eq : ⇑f₀ = ((↑) : ℝ≥0 → ℝ) ∘ ⇑f := rfl
  have f₀_nn : 0 ≤ ⇑f₀ := fun _ ↦ by
    simp only [f₀_eq, Pi.zero_apply, Function.comp_apply, NNReal.zero_le_coe]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => Measure …
    f : BoundedContinuousFunction Ω NNReal
    lip : LipschitzWith 1 NNReal.toReal
    f₀ : BoundedContinuousFunction Ω Real := BoundedContinuousFunction.comp NNReal …
    _def_f₀ : Eq f₀ (BoundedContinuousFunction.comp NNReal.toReal lip f)
    f₀_eq : Eq (⇑f₀) (Function.comp NNReal.toReal ⇑f)
    f₀_nn : LE.le 0 ⇑f₀
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral ↑(μs n) fun x => ↑(f x)).t …
  -/
  have f₀_ae_nn : 0 ≤ᵐ[(μ : Measure Ω)] ⇑f₀ := .of_forall f₀_nn
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => Measure …
    f : BoundedContinuousFunction Ω NNReal
    lip : LipschitzWith 1 NNReal.toReal
    f₀ : BoundedContinuousFunction Ω Real := BoundedContinuousFunction.comp NNReal …
    _def_f₀ : Eq f₀ (BoundedContinuousFunction.comp NNReal.toReal lip f)
    f₀_eq : Eq (⇑f₀) (Function.comp NNReal.toReal ⇑f)
    f₀_nn : LE.le 0 ⇑f₀
    f₀_ae_nn : (MeasureTheory.ae ↑μ).EventuallyLE 0 ⇑f₀
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral ↑(μs n) fun x => ↑(f x)).t …
  -/
  have f₀_ae_nns : ∀ i, 0 ≤ᵐ[(μs i : Measure Ω)] ⇑f₀ := fun i ↦ .of_forall f₀_nn
  have aux :=
    integral_eq_lintegral_of_nonneg_ae f₀_ae_nn f₀.continuous.measurable.aestronglyMeasurable
  have auxs := fun i ↦
    integral_eq_lintegral_of_nonneg_ae (f₀_ae_nns i) f₀.continuous.measurable.aestronglyMeasurable
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => Measure …
    f : BoundedContinuousFunction Ω NNReal
    lip : LipschitzWith 1 NNReal.toReal
    f₀ : BoundedContinuousFunction Ω Real := BoundedContinuousFunction.comp NNReal …
    _def_f₀ : Eq f₀ (BoundedContinuousFunction.comp NNReal.toReal lip f)
    f₀_eq : Eq (⇑f₀) (Function.comp NNReal.toReal ⇑f)
    f₀_nn : LE.le 0 ⇑f₀
    f₀_ae_nn : (MeasureTheory.ae ↑μ).EventuallyLE 0 ⇑f₀
    f₀_ae_nns : ∀ (i : γ), (MeasureTheory.ae ↑(μs i)).EventuallyLE 0 ⇑f₀
    aux : Eq (MeasureTheory.integral ↑μ fun a => f₀ a) (MeasureTheory.lintegral ↑μ …
    auxs : ∀ (i : γ), Eq (MeasureTheory.integral ↑(μs i) fun a => f₀ a) (MeasureTh …
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral ↑(μs n) fun x => ↑(f x)).t …
  -/
  simp_rw [f₀_eq, Function.comp_apply, ENNReal.ofReal_coe_nnreal] at aux auxs
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => Measure …
    f : BoundedContinuousFunction Ω NNReal
    lip : LipschitzWith 1 NNReal.toReal
    f₀ : BoundedContinuousFunction Ω Real := BoundedContinuousFunction.comp NNReal …
    _def_f₀ : Eq f₀ (BoundedContinuousFunction.comp NNReal.toReal lip f)
    f₀_eq : Eq (⇑f₀) (Function.comp NNReal.toReal ⇑f)
    f₀_nn : LE.le 0 ⇑f₀
    f₀_ae_nn : (MeasureTheory.ae ↑μ).EventuallyLE 0 ⇑f₀
    f₀_ae_nns : ∀ (i : γ), (MeasureTheory.ae ↑(μs i)).EventuallyLE 0 ⇑f₀
    aux : Eq (MeasureTheory.integral ↑μ fun a => ↑(f a)) (MeasureTheory.lintegral  …
    auxs : ∀ (i : γ), Eq (MeasureTheory.integral ↑(μs i) fun a => ↑(f a)) (Measure …
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral ↑(μs n) fun x => ↑(f x)).t …
  -/
  simpa only [← aux, ← auxs] using h f₀
  /-
    🎉 no goals
  -/


/-- A characterization of weak convergence in terms of integrals of bounded continuous
real-valued functions. -/
theorem tendsto_iff_forall_integral_tendsto {γ : Type*} {F : Filter γ} {μs : γ → FiniteMeasure Ω}
    {μ : FiniteMeasure Ω} :
    Tendsto μs F (𝓝 μ) ↔
      ∀ f : Ω →ᵇ ℝ,
        Tendsto (fun i ↦ ∫ x, f x ∂(μs i : Measure Ω)) F (𝓝 (∫ x, f x ∂(μ : Measure Ω))) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Iff (Filter.Tendsto μs F (nhds μ)) (∀ (f : BoundedContinuousFunction Ω Real) …
  -/
  refine ⟨?_, tendsto_of_forall_integral_tendsto⟩
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ Filter.Tendsto μs F (nhds μ) → ∀ (f : BoundedContinuousFunction Ω Real), Fil …
  -/
  rw [tendsto_iff_forall_lintegral_tendsto]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    ⊢ (∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Measur …
  -/
  intro h f
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Measu …
    f : BoundedContinuousFunction Ω Real
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral ↑(μs i) fun x => f x) F (nhd …
  -/
  simp_rw [BoundedContinuousFunction.integral_eq_integral_nnrealPart_sub]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Measu …
    f : BoundedContinuousFunction Ω Real
    ⊢ Filter.Tendsto (fun i => HSub.hSub (MeasureTheory.integral ↑(μs i) fun x =>  …
  -/
  set f_pos := f.nnrealPart with _def_f_pos
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Measu …
    f : BoundedContinuousFunction Ω Real
    f_pos : BoundedContinuousFunction Ω NNReal := f.nnrealPart
    _def_f_pos : Eq f_pos f.nnrealPart
    ⊢ Filter.Tendsto (fun i => HSub.hSub (MeasureTheory.integral ↑(μs i) fun x =>  …
  -/
  set f_neg := (-f).nnrealPart with _def_f_neg
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Measu …
    f : BoundedContinuousFunction Ω Real
    f_pos : BoundedContinuousFunction Ω NNReal := f.nnrealPart
    _def_f_pos : Eq f_pos f.nnrealPart
    f_neg : BoundedContinuousFunction Ω NNReal := (Neg.neg f).nnrealPart
    _def_f_neg : Eq f_neg (Neg.neg f).nnrealPart
    ⊢ Filter.Tendsto (fun i => HSub.hSub (MeasureTheory.integral ↑(μs i) fun x =>  …
  -/
  have tends_pos := (ENNReal.tendsto_toReal (f_pos.lintegral_lt_top_of_nnreal μ).ne).comp (h f_pos)
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Measu …
    f : BoundedContinuousFunction Ω Real
    f_pos : BoundedContinuousFunction Ω NNReal := f.nnrealPart
    _def_f_pos : Eq f_pos f.nnrealPart
    f_neg : BoundedContinuousFunction Ω NNReal := (Neg.neg f).nnrealPart
    _def_f_neg : Eq f_neg (Neg.neg f).nnrealPart
    tends_pos : Filter.Tendsto (Function.comp ENNReal.toReal fun i => MeasureTheor …
    ⊢ Filter.Tendsto (fun i => HSub.hSub (MeasureTheory.integral ↑(μs i) fun x =>  …
  -/
  have tends_neg := (ENNReal.tendsto_toReal (f_neg.lintegral_lt_top_of_nnreal μ).ne).comp (h f_neg)
  have aux :
    ∀ g : Ω →ᵇ ℝ≥0,
      (ENNReal.toReal ∘ fun i : γ ↦ ∫⁻ x : Ω, ↑(g x) ∂(μs i : Measure Ω)) =
        fun i : γ ↦ (∫⁻ x : Ω, ↑(g x) ∂(μs i : Measure Ω)).toReal :=
    fun _ ↦ rfl
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Measu …
    f : BoundedContinuousFunction Ω Real
    f_pos : BoundedContinuousFunction Ω NNReal := f.nnrealPart
    _def_f_pos : Eq f_pos f.nnrealPart
    f_neg : BoundedContinuousFunction Ω NNReal := (Neg.neg f).nnrealPart
    _def_f_neg : Eq f_neg (Neg.neg f).nnrealPart
    tends_pos : Filter.Tendsto (Function.comp ENNReal.toReal fun i => MeasureTheor …
    tends_neg : Filter.Tendsto (Function.comp ENNReal.toReal fun i => MeasureTheor …
    aux : ∀ (g : BoundedContinuousFunction Ω NNReal), Eq (Function.comp ENNReal.to …
    ⊢ Filter.Tendsto (fun i => HSub.hSub (MeasureTheory.integral ↑(μs i) fun x =>  …
  -/
  simp_rw [aux, BoundedContinuousFunction.toReal_lintegral_coe_eq_integral] at tends_pos tends_neg
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    γ : Type u_2
    F : Filter γ
    μs : γ → MeasureTheory.FiniteMeasure Ω
    μ : MeasureTheory.FiniteMeasure Ω
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Measu …
    f : BoundedContinuousFunction Ω Real
    f_pos : BoundedContinuousFunction Ω NNReal := f.nnrealPart
    _def_f_pos : Eq f_pos f.nnrealPart
    f_neg : BoundedContinuousFunction Ω NNReal := (Neg.neg f).nnrealPart
    _def_f_neg : Eq f_neg (Neg.neg f).nnrealPart
    aux : ∀ (g : BoundedContinuousFunction Ω NNReal), Eq (Function.comp ENNReal.to …
    tends_pos : Filter.Tendsto (fun i => MeasureTheory.integral ↑(μs i) fun x => ↑ …
    tends_neg : Filter.Tendsto (fun i => MeasureTheory.integral ↑(μs i) fun x => ↑ …
    ⊢ Filter.Tendsto (fun i => HSub.hSub (MeasureTheory.integral ↑(μs i) fun x =>  …
  -/
  exact Tendsto.sub tends_pos tends_neg
  /-
    🎉 no goals
  -/


lemma continuous_integral_boundedContinuousFunction
    {α : Type*} [TopologicalSpace α] [MeasurableSpace α] [OpensMeasurableSpace α] (f : α →ᵇ ℝ) :
    Continuous fun μ : FiniteMeasure α ↦ ∫ x, f x ∂μ := by
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
    ⊢ ∀ (x : MeasureTheory.FiniteMeasure α), ContinuousAt (fun μ => MeasureTheory. …
  -/
  intro μ
  exact continuousAt_of_tendsto_nhds
    (FiniteMeasure.tendsto_iff_forall_integral_tendsto.mp tendsto_id f)


/-- The push-forward of a finite measure by a function between measurable spaces. -/
noncomputable def map (ν : FiniteMeasure Ω) (f : Ω → Ω') : FiniteMeasure Ω' :=
  ⟨(ν : Measure Ω).map f, by
    /-
      Ω : Type u_1
      Ω' : Type u_2
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSpace Ω'
      ν : MeasureTheory.FiniteMeasure Ω
      f : Ω → Ω'
      ⊢ MeasureTheory.IsFiniteMeasure (MeasureTheory.Measure.map f ↑ν)
    -/
    constructor
    /-
      case measure_univ_lt_top
      Ω : Type u_1
      Ω' : Type u_2
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSpace Ω'
      ν : MeasureTheory.FiniteMeasure Ω
      f : Ω → Ω'
      ⊢ LT.lt ((MeasureTheory.Measure.map f ↑ν) Set.univ) Top.top
    -/
    by_cases f_aemble : AEMeasurable f ν
      /-
        case pos
        Ω : Type u_1
        Ω' : Type u_2
        inst✝¹ : MeasurableSpace Ω
        inst✝ : MeasurableSpace Ω'
        ν : MeasureTheory.FiniteMeasure Ω
        f : Ω → Ω'
        f_aemble : AEMeasurable f ↑ν
        ⊢ LT.lt ((MeasureTheory.Measure.map f ↑ν) Set.univ) Top.top
      -/
    · rw [Measure.map_apply_of_aemeasurable f_aemble MeasurableSet.univ]
      /-
        case pos
        Ω : Type u_1
        Ω' : Type u_2
        inst✝¹ : MeasurableSpace Ω
        inst✝ : MeasurableSpace Ω'
        ν : MeasureTheory.FiniteMeasure Ω
        f : Ω → Ω'
        f_aemble : AEMeasurable f ↑ν
        ⊢ LT.lt (↑ν (Set.preimage f Set.univ)) Top.top
      -/
      exact measure_lt_top (↑ν) (f ⁻¹' univ)
      /-
        🎉 no goals
      -/
      /-
        case neg
        Ω : Type u_1
        Ω' : Type u_2
        inst✝¹ : MeasurableSpace Ω
        inst✝ : MeasurableSpace Ω'
        ν : MeasureTheory.FiniteMeasure Ω
        f : Ω → Ω'
        f_aemble : Not (AEMeasurable f ↑ν)
        ⊢ LT.lt ((MeasureTheory.Measure.map f ↑ν) Set.univ) Top.top
      -/
    · simp [Measure.map, f_aemble]⟩
      /-
        🎉 no goals
      -/


@[simp] lemma toMeasure_map (ν : FiniteMeasure Ω) (f : Ω → Ω') :
    (ν.map f).toMeasure = ν.toMeasure.map f := rfl


/-- Note that this is an equality of elements of `ℝ≥0∞`. See also
`MeasureTheory.FiniteMeasure.map_apply` for the corresponding equality as elements of `ℝ≥0`. -/
lemma map_apply' (ν : FiniteMeasure Ω) {f : Ω → Ω'} (f_aemble : AEMeasurable f ν)
    {A : Set Ω'} (A_mble : MeasurableSet A) :
    (ν.map f : Measure Ω') A = (ν : Measure Ω) (f ⁻¹' A) :=
  Measure.map_apply_of_aemeasurable f_aemble A_mble


lemma map_apply_of_aemeasurable (ν : FiniteMeasure Ω) {f : Ω → Ω'} (f_aemble : AEMeasurable f ν)
    {A : Set Ω'} (A_mble : MeasurableSet A) :
    ν.map f A = ν (f ⁻¹' A) := by
  /-
    Ω : Type u_1
    Ω' : Type u_2
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSpace Ω'
    ν : MeasureTheory.FiniteMeasure Ω
    f : Ω → Ω'
    f_aemble : AEMeasurable f ↑ν
    A : Set Ω'
    A_mble : MeasurableSet A
    ⊢ Eq ((ν.map f) A) (ν (Set.preimage f A))
  -/
  have key := ν.map_apply' f_aemble A_mble
  /-
    Ω : Type u_1
    Ω' : Type u_2
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSpace Ω'
    ν : MeasureTheory.FiniteMeasure Ω
    f : Ω → Ω'
    f_aemble : AEMeasurable f ↑ν
    A : Set Ω'
    A_mble : MeasurableSet A
    key : Eq (↑(ν.map f) A) (↑ν (Set.preimage f A))
    ⊢ Eq ((ν.map f) A) (ν (Set.preimage f A))
  -/
  exact (ENNReal.toNNReal_eq_toNNReal_iff' (measure_ne_top _ _) (measure_ne_top _ _)).mpr key
  /-
    🎉 no goals
  -/


lemma map_apply (ν : FiniteMeasure Ω) {f : Ω → Ω'} (f_mble : Measurable f)
    {A : Set Ω'} (A_mble : MeasurableSet A) :
    ν.map f A = ν (f ⁻¹' A) :=
  map_apply_of_aemeasurable ν f_mble.aemeasurable A_mble


@[simp] lemma map_add {f : Ω → Ω'} (f_mble : Measurable f) (ν₁ ν₂ : FiniteMeasure Ω) :
    (ν₁ + ν₂).map f = ν₁.map f + ν₂.map f := by
  /-
    Ω : Type u_1
    Ω' : Type u_2
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSpace Ω'
    f : Ω → Ω'
    f_mble : Measurable f
    ν₁ ν₂ : MeasureTheory.FiniteMeasure Ω
    ⊢ Eq ((HAdd.hAdd ν₁ ν₂).map f) (HAdd.hAdd (ν₁.map f) (ν₂.map f))
  -/
  ext s s_mble
  /-
    case h
    Ω : Type u_1
    Ω' : Type u_2
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSpace Ω'
    f : Ω → Ω'
    f_mble : Measurable f
    ν₁ ν₂ : MeasureTheory.FiniteMeasure Ω
    s : Set Ω'
    s_mble : MeasurableSet s
    ⊢ Eq (↑((HAdd.hAdd ν₁ ν₂).map f) s) (↑(HAdd.hAdd (ν₁.map f) (ν₂.map f)) s)
  -/
  simp only [map_apply' _ f_mble.aemeasurable s_mble, toMeasure_add, Measure.add_apply]
  /-
    🎉 no goals
  -/


@[simp] lemma map_smul {f : Ω → Ω'} (c : ℝ≥0) (ν : FiniteMeasure Ω) :
    (c • ν).map f = c • (ν.map f) := by
  /-
    Ω : Type u_1
    Ω' : Type u_2
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSpace Ω'
    f : Ω → Ω'
    c : NNReal
    ν : MeasureTheory.FiniteMeasure Ω
    ⊢ Eq ((HSMul.hSMul c ν).map f) (HSMul.hSMul c (ν.map f))
  -/
  ext s _
  /-
    case h
    Ω : Type u_1
    Ω' : Type u_2
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSpace Ω'
    f : Ω → Ω'
    c : NNReal
    ν : MeasureTheory.FiniteMeasure Ω
    s : Set Ω'
    a✝ : MeasurableSet s
    ⊢ Eq (↑((HSMul.hSMul c ν).map f) s) (↑(HSMul.hSMul c (ν.map f)) s)
  -/
  simp [toMeasure_smul]
  /-
    🎉 no goals
  -/


/-- The push-forward of a finite measure by a function between measurable spaces as a linear map. -/
noncomputable def mapHom {f : Ω → Ω'} (f_mble : Measurable f) :
    FiniteMeasure Ω →ₗ[ℝ≥0] FiniteMeasure Ω' where
  toFun := fun ν ↦ ν.map f
  map_add' := map_add f_mble
  map_smul' := map_smul


/-- If `f : X → Y` is continuous and `Y` is equipped with the Borel sigma algebra, then
(weak) convergence of `FiniteMeasure`s on `X` implies (weak) convergence of the push-forwards
of these measures by `f`. -/
lemma tendsto_map_of_tendsto_of_continuous {ι : Type*} {L : Filter ι}
    (νs : ι → FiniteMeasure Ω) (ν : FiniteMeasure Ω) (lim : Tendsto νs L (𝓝 ν))
    {f : Ω → Ω'} (f_cont : Continuous f) :
    Tendsto (fun i ↦ (νs i).map f) L (𝓝 (ν.map f)) := by
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
    νs : ι → MeasureTheory.FiniteMeasure Ω
    ν : MeasureTheory.FiniteMeasure Ω
    lim : Filter.Tendsto νs L (nhds ν)
    f : Ω → Ω'
    f_cont : Continuous f
    ⊢ Filter.Tendsto (fun i => (νs i).map f) L (nhds (ν.map f))
  -/
  rw [FiniteMeasure.tendsto_iff_forall_lintegral_tendsto] at lim ⊢
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
    νs : ι → MeasureTheory.FiniteMeasure Ω
    ν : MeasureTheory.FiniteMeasure Ω
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
    νs : ι → MeasureTheory.FiniteMeasure Ω
    ν : MeasureTheory.FiniteMeasure Ω
    lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
    f : Ω → Ω'
    f_cont : Continuous f
    g : BoundedContinuousFunction Ω' NNReal
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral ↑((νs i).map f) fun x => ↑( …
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
      νs : ι → MeasureTheory.FiniteMeasure Ω
      ν : MeasureTheory.FiniteMeasure Ω
      lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
      f : Ω → Ω'
      f_cont : Continuous f
      g : BoundedContinuousFunction Ω' NNReal
      x✝ : ι
      ⊢ Eq (MeasureTheory.lintegral ↑((νs x✝).map f) fun x => ↑(g x)) (MeasureTheory …
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
      νs : ι → MeasureTheory.FiniteMeasure Ω
      ν : MeasureTheory.FiniteMeasure Ω
      lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
      f : Ω → Ω'
      f_cont : Continuous f
      g : BoundedContinuousFunction Ω' NNReal
      x✝ : ι
      ⊢ Eq (MeasureTheory.lintegral ↑⟨MeasureTheory.Measure.map f ↑(νs x✝), ⋯⟩ fun x …
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
      νs : ι → MeasureTheory.FiniteMeasure Ω
      ν : MeasureTheory.FiniteMeasure Ω
      lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
      f : Ω → Ω'
      f_cont : Continuous f
      g : BoundedContinuousFunction Ω' NNReal
      x✝ : ι
      ⊢ Measurable fun x => ↑(g x)
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
      νs : ι → MeasureTheory.FiniteMeasure Ω
      ν : MeasureTheory.FiniteMeasure Ω
      lim : ∀ (f : BoundedContinuousFunction Ω NNReal), Filter.Tendsto (fun i => Mea …
      f : Ω → Ω'
      f_cont : Continuous f
      g : BoundedContinuousFunction Ω' NNReal
      ⊢ Measurable fun x => ↑(g x)
    -/
    exact (ENNReal.continuous_coe.comp g.continuous).measurable
    /-
      🎉 no goals
    -/


/-- If `f : X → Y` is continuous and `Y` is equipped with the Borel sigma algebra, then
the push-forward of finite measures `f* : FiniteMeasure X → FiniteMeasure Y` is continuous
(in the topologies of weak convergence of measures). -/
lemma continuous_map {f : Ω → Ω'} (f_cont : Continuous f) :
    Continuous (fun ν ↦ FiniteMeasure.map ν f) := by
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
    ⊢ Continuous fun ν => ν.map f
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
    ⊢ ∀ (x : MeasureTheory.FiniteMeasure Ω), ContinuousAt (fun ν => ν.map f) x
  -/
  exact fun _ ↦ tendsto_map_of_tendsto_of_continuous _ _ continuous_id.continuousAt f_cont
  /-
    🎉 no goals
  -/


/-- The push-forward of a finite measure by a continuous function between Borel spaces as
a continuous linear map. -/
noncomputable def mapCLM {f : Ω → Ω'} (f_cont : Continuous f) :
    FiniteMeasure Ω →L[ℝ≥0] FiniteMeasure Ω' where
  toFun := fun ν ↦ ν.map f
  map_add' := map_add f_cont.measurable
  map_smul' := map_smul
  cont := continuous_map f_cont


