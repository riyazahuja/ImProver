/-- A `ComplexMeasure` is a `ℂ`-vector measure. -/
abbrev ComplexMeasure (α : Type*) [MeasurableSpace α] :=
  VectorMeasure α ℂ


/-- The real part of a complex measure is a signed measure. -/
@[simps! apply]
def re : ComplexMeasure α →ₗ[ℝ] SignedMeasure α :=
  mapRangeₗ Complex.reCLM Complex.continuous_re


/-- The imaginary part of a complex measure is a signed measure. -/
@[simps! apply]
def im : ComplexMeasure α →ₗ[ℝ] SignedMeasure α :=
  mapRangeₗ Complex.imCLM Complex.continuous_im


/-- Given `s` and `t` signed measures, `s + it` is a complex measure -/
@[simps!]
def _root_.MeasureTheory.SignedMeasure.toComplexMeasure (s t : SignedMeasure α) :
    ComplexMeasure α where
  measureOf' i := ⟨s i, t i⟩
               /-
                 α : Type u_1
                 m : MeasurableSpace α
                 s t : MeasureTheory.SignedMeasure α
                 ⊢ Eq ((fun i => { re := ↑s i, im := ↑t i }) EmptyCollection.emptyCollection) 0
               -/
  empty' := by dsimp only; rw [s.empty, t.empty]; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/
                             /-
                               α : Type u_1
                               m : MeasurableSpace α
                               s t : MeasureTheory.SignedMeasure α
                               i : Set α
                               hi : Not (MeasurableSet i)
                               ⊢ Eq ((fun i => { re := ↑s i, im := ↑t i }) i) 0
                             -/
  not_measurable' i hi := by dsimp only; rw [s.not_measurable hi, t.not_measurable hi]; rfl
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
  m_iUnion' _ hf hfdisj := (Complex.hasSum_iff _ _).2 ⟨s.m_iUnion hf hfdisj, t.m_iUnion hf hfdisj⟩


theorem _root_.MeasureTheory.SignedMeasure.toComplexMeasure_apply
    {s t : SignedMeasure α} {i : Set α} : s.toComplexMeasure t i = ⟨s i, t i⟩ := rfl


theorem toComplexMeasure_to_signedMeasure (c : ComplexMeasure α) :
    SignedMeasure.toComplexMeasure (ComplexMeasure.re c) (ComplexMeasure.im c) = c := rfl


theorem _root_.MeasureTheory.SignedMeasure.re_toComplexMeasure (s t : SignedMeasure α) :
    ComplexMeasure.re (SignedMeasure.toComplexMeasure s t) = s := rfl


theorem _root_.MeasureTheory.SignedMeasure.im_toComplexMeasure (s t : SignedMeasure α) :
    ComplexMeasure.im (SignedMeasure.toComplexMeasure s t) = t := rfl


/-- The complex measures form an equivalence to the type of pairs of signed measures. -/
@[simps]
def equivSignedMeasure : ComplexMeasure α ≃ SignedMeasure α × SignedMeasure α where
  toFun c := ⟨ComplexMeasure.re c, ComplexMeasure.im c⟩
  invFun := fun ⟨s, t⟩ => s.toComplexMeasure t
  left_inv c := c.toComplexMeasure_to_signedMeasure
  right_inv := fun ⟨s, t⟩ => Prod.mk.inj_iff.2 ⟨s.re_toComplexMeasure t, s.im_toComplexMeasure t⟩


/-- The complex measures form a linear isomorphism to the type of pairs of signed measures. -/
@[simps]
def equivSignedMeasureₗ : ComplexMeasure α ≃ₗ[R] SignedMeasure α × SignedMeasure α :=
  { equivSignedMeasure with
                              /-
                                α : Type u_1
                                m : MeasurableSpace α
                                R : Type u_2
                                inst✝³ : Semiring R
                                inst✝² : Module R Real
                                inst✝¹ : ContinuousConstSMul R Real
                                inst✝ : ContinuousConstSMul R Complex
                                c d : MeasureTheory.ComplexMeasure α
                                ⊢ Eq (__src✝.toFun (HAdd.hAdd c d)) (HAdd.hAdd (__src✝.toFun c) (__src✝.toFun  …
                              -/
    map_add' := fun c d => by rfl
                              /-
                                🎉 no goals
                              -/
    map_smul' := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        R : Type u_2
        inst✝³ : Semiring R
        inst✝² : Module R Real
        inst✝¹ : ContinuousConstSMul R Real
        inst✝ : ContinuousConstSMul R Complex
        ⊢ ∀ (m_1 : R) (x : MeasureTheory.ComplexMeasure α), Eq ({ toFun := __src✝.toFu …
      -/
      intro r c
      /-
        α : Type u_1
        m : MeasurableSpace α
        R : Type u_2
        inst✝³ : Semiring R
        inst✝² : Module R Real
        inst✝¹ : ContinuousConstSMul R Real
        inst✝ : ContinuousConstSMul R Complex
        r : R
        c : MeasureTheory.ComplexMeasure α
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul r c)) (HSMul …
      -/
      dsimp
      /-
        α : Type u_1
        m : MeasurableSpace α
        R : Type u_2
        inst✝³ : Semiring R
        inst✝² : Module R Real
        inst✝¹ : ContinuousConstSMul R Real
        inst✝ : ContinuousConstSMul R Complex
        r : R
        c : MeasureTheory.ComplexMeasure α
        ⊢ Eq { fst := MeasureTheory.VectorMeasure.mapRange (HSMul.hSMul r c) Complex.r …
      -/
      ext
        /-
          case fst.h
          α : Type u_1
          m : MeasurableSpace α
          R : Type u_2
          inst✝³ : Semiring R
          inst✝² : Module R Real
          inst✝¹ : ContinuousConstSMul R Real
          inst✝ : ContinuousConstSMul R Complex
          r : R
          c : MeasureTheory.ComplexMeasure α
          i✝ : Set α
          a✝ : MeasurableSet i✝
          ⊢ Eq (↑{ fst := MeasureTheory.VectorMeasure.mapRange (HSMul.hSMul r c) Complex …
        -/
      · simp [Complex.smul_re]
        /-
          🎉 no goals
        -/
        /-
          case snd.h
          α : Type u_1
          m : MeasurableSpace α
          R : Type u_2
          inst✝³ : Semiring R
          inst✝² : Module R Real
          inst✝¹ : ContinuousConstSMul R Real
          inst✝ : ContinuousConstSMul R Complex
          r : R
          c : MeasureTheory.ComplexMeasure α
          i✝ : Set α
          a✝ : MeasurableSet i✝
          ⊢ Eq (↑{ fst := MeasureTheory.VectorMeasure.mapRange (HSMul.hSMul r c) Complex …
        -/
      · simp [Complex.smul_im] }
        /-
          🎉 no goals
        -/


theorem absolutelyContinuous_ennreal_iff (c : ComplexMeasure α) (μ : VectorMeasure α ℝ≥0∞) :
    c ≪ᵥ μ ↔ ComplexMeasure.re c ≪ᵥ μ ∧ ComplexMeasure.im c ≪ᵥ μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    c : MeasureTheory.ComplexMeasure α
    μ : MeasureTheory.VectorMeasure α ENNReal
    ⊢ Iff (MeasureTheory.VectorMeasure.AbsolutelyContinuous c μ) (And (MeasureTheo …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_1
      m : MeasurableSpace α
      c : MeasureTheory.ComplexMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : MeasureTheory.VectorMeasure.AbsolutelyContinuous c μ
      ⊢ And (MeasureTheory.VectorMeasure.AbsolutelyContinuous (MeasureTheory.Complex …
    -/
                                  /-
                                    🎉 no goals
                                  -/
  · constructor <;> · intro i hi; simp [h hi]
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      c : MeasureTheory.ComplexMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : And (MeasureTheory.VectorMeasure.AbsolutelyContinuous (MeasureTheory.Compl …
      ⊢ MeasureTheory.VectorMeasure.AbsolutelyContinuous c μ
    -/
  · intro i hi
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      c : MeasureTheory.ComplexMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : And (MeasureTheory.VectorMeasure.AbsolutelyContinuous (MeasureTheory.Compl …
      i : Set α
      hi : Eq (↑μ i) 0
      ⊢ Eq (↑c i) 0
    -/
    rw [← Complex.re_add_im (c i), (_ : (c i).re = 0), (_ : (c i).im = 0)]
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      c : MeasureTheory.ComplexMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : And (MeasureTheory.VectorMeasure.AbsolutelyContinuous (MeasureTheory.Compl …
      i : Set α
      hi : Eq (↑μ i) 0
      ⊢ Eq (HAdd.hAdd (↑0) (HMul.hMul (↑0) Complex.I)) 0
    -/
    exacts [by simp, h.2 hi, h.1 hi]
    /-
      🎉 no goals
    -/


