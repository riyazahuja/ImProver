/-- A kernel from a measurable space `α` to another measurable space `β` is a measurable function
`κ : α → Measure β`. The measurable space structure on `MeasureTheory.Measure β` is given by
`MeasureTheory.Measure.instMeasurableSpace`. A map `κ : α → MeasureTheory.Measure β` is measurable
iff `∀ s : Set β, MeasurableSet s → Measurable (fun a ↦ κ a s)`. -/
structure Kernel (α β : Type*) [MeasurableSpace α] [MeasurableSpace β] where
  /-- The underlying function of a kernel.

  Do not use this function directly. Instead use the coercion coming from the `DFunLike`
  instance. -/
  toFun : α → Measure β
  /-- A kernel is a measurable map.

  Do not use this lemma directly. Use `Kernel.measurable` instead. -/
  measurable' : Measurable toFun


@[deprecated (since := "2024-07-22")] alias kernel := Kernel


/-- Notation for `Kernel` with respect to a non-standard σ-algebra in the domain. -/
scoped notation "Kernel[" mα "]" α:arg β:arg => @Kernel α β mα _


/-- Notation for `Kernel` with respect to a non-standard σ-algebra in the domain and codomain. -/
scoped notation "Kernel[" mα ", " mβ "]" α:arg β:arg => @Kernel α β mα mβ


instance instFunLike : FunLike (Kernel α β) α (Measure β) where
  coe := toFun
                             /-
                               α : Type u_1
                               β : Type u_2
                               ι : Type u_3
                               mα : MeasurableSpace α
                               mβ : MeasurableSpace β
                               f g : ProbabilityTheory.Kernel α β
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


lemma measurable (κ : Kernel α β) : Measurable κ := κ.measurable'

@[simp, norm_cast] lemma coe_mk (f : α → Measure β) (hf) : mk f hf = f := rfl


instance instZero : Zero (Kernel α β) where zero := ⟨0, measurable_zero⟩

noncomputable instance instAdd : Add (Kernel α β) where add κ η := ⟨κ + η, κ.2.add η.2⟩

noncomputable instance instSMulNat : SMul ℕ (Kernel α β) where
  smul n κ := ⟨n • κ, (measurable_const (a := n)).smul κ.2⟩


@[simp, norm_cast] lemma coe_zero : ⇑(0 : Kernel α β) = 0 := rfl

@[simp, norm_cast] lemma coe_add (κ η : Kernel α β) : ⇑(κ + η) = κ + η := rfl

@[simp, norm_cast] lemma coe_nsmul (n : ℕ) (κ : Kernel α β) : ⇑(n • κ) = n • κ := rfl


@[simp] lemma zero_apply (a : α) : (0 : Kernel α β) a = 0 := rfl

@[simp] lemma add_apply (κ η : Kernel α β) (a : α) : (κ + η) a = κ a + η a := rfl

@[simp] lemma nsmul_apply (n : ℕ) (κ : Kernel α β) (a : α) : (n • κ) a = n • κ a := rfl


noncomputable instance instAddCommMonoid : AddCommMonoid (Kernel α β) :=
                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                ι : Type u_3
                                                                mα : MeasurableSpace α
                                                                mβ : MeasurableSpace β
                                                                ⊢ ∀ (x : ProbabilityTheory.Kernel α β) (n : Nat), Eq (⇑(HSMul.hSMul n x)) (HSM …
                                                              -/
  DFunLike.coe_injective.addCommMonoid _ coe_zero coe_add (by intros; rfl)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


instance instPartialOrder : PartialOrder (Kernel α β) := .lift _ DFunLike.coe_injective


instance instCovariantAddLE {α β : Type*} [MeasurableSpace α] [MeasurableSpace β] :
    CovariantClass (Kernel α β) (Kernel α β) (· + ·) (· ≤ ·) :=
  ⟨fun _ _ _ hμ a ↦ add_le_add_left (hμ a) _⟩


noncomputable
instance instOrderBot {α β : Type*} [MeasurableSpace α] [MeasurableSpace β] :
    OrderBot (Kernel α β) where
  bot := 0
                   /-
                     α✝ : Type u_1
                     β✝ : Type u_2
                     ι : Type u_3
                     mα : MeasurableSpace α✝
                     mβ : MeasurableSpace β✝
                     α : Type u_4
                     β : Type u_5
                     inst✝¹ : MeasurableSpace α
                     inst✝ : MeasurableSpace β
                     κ : ProbabilityTheory.Kernel α β
                     a : α
                     ⊢ LE.le ((fun f => ⇑f) Bot.bot a) ((fun f => ⇑f) κ a)
                   -/
  bot_le κ a := by simp only [coe_zero, Pi.zero_apply, Measure.zero_le]
                   /-
                     🎉 no goals
                   -/


/-- Coercion to a function as an additive monoid homomorphism. -/
def coeAddHom (α β : Type*) [MeasurableSpace α] [MeasurableSpace β] :
    Kernel α β →+ α → Measure β where
  toFun := (⇑)
  map_zero' := coe_zero
  map_add' := coe_add


@[simp]
theorem coe_finset_sum (I : Finset ι) (κ : ι → Kernel α β) : ⇑(∑ i ∈ I, κ i) = ∑ i ∈ I, ⇑(κ i) :=
  map_sum (coeAddHom α β) _ _


theorem finset_sum_apply (I : Finset ι) (κ : ι → Kernel α β) (a : α) :
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              ι : Type u_3
                                              mα : MeasurableSpace α
                                              mβ : MeasurableSpace β
                                              I : Finset ι
                                              κ : ι → ProbabilityTheory.Kernel α β
                                              a : α
                                              ⊢ Eq ((I.sum fun i => κ i) a) (I.sum fun i => (κ i) a)
                                            -/
    (∑ i ∈ I, κ i) a = ∑ i ∈ I, κ i a := by rw [coe_finset_sum, Finset.sum_apply]
                                            /-
                                              🎉 no goals
                                            -/


theorem finset_sum_apply' (I : Finset ι) (κ : ι → Kernel α β) (a : α) (s : Set β) :
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  ι : Type u_3
                                                  mα : MeasurableSpace α
                                                  mβ : MeasurableSpace β
                                                  I : Finset ι
                                                  κ : ι → ProbabilityTheory.Kernel α β
                                                  a : α
                                                  s : Set β
                                                  ⊢ Eq (((I.sum fun i => κ i) a) s) (I.sum fun i => ((κ i) a) s)
                                                -/
    (∑ i ∈ I, κ i) a s = ∑ i ∈ I, κ i a s := by rw [finset_sum_apply, Measure.finset_sum_apply]
                                                /-
                                                  🎉 no goals
                                                -/


/-- A kernel is a Markov kernel if every measure in its image is a probability measure. -/
class IsMarkovKernel (κ : Kernel α β) : Prop where
  isProbabilityMeasure : ∀ a, IsProbabilityMeasure (κ a)


/-- A class for kernels which are zero or a Markov kernel. -/
class IsZeroOrMarkovKernel (κ : Kernel α β) : Prop where
  eq_zero_or_isMarkovKernel' : κ = 0 ∨ IsMarkovKernel κ


/-- A kernel is finite if every measure in its image is finite, with a uniform bound. -/
class IsFiniteKernel (κ : Kernel α β) : Prop where
  exists_univ_le : ∃ C : ℝ≥0∞, C < ∞ ∧ ∀ a, κ a Set.univ ≤ C


theorem eq_zero_or_isMarkovKernel
    (κ : Kernel α β) [h : IsZeroOrMarkovKernel κ] :
    κ = 0 ∨ IsMarkovKernel κ :=
  h.eq_zero_or_isMarkovKernel'


/-- A constant `C : ℝ≥0∞` such that `C < ∞` (`ProbabilityTheory.IsFiniteKernel.bound_lt_top κ`) and
for all `a : α` and `s : Set β`, `κ a s ≤ C` (`ProbabilityTheory.Kernel.measure_le_bound κ a s`).

Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: does it make sense to
-- make `ProbabilityTheory.IsFiniteKernel.bound` the least possible bound?
-- Should it be an `NNReal` number? -/
noncomputable def IsFiniteKernel.bound (κ : Kernel α β) [h : IsFiniteKernel κ] : ℝ≥0∞ :=
  h.exists_univ_le.choose


theorem IsFiniteKernel.bound_lt_top (κ : Kernel α β) [h : IsFiniteKernel κ] :
    IsFiniteKernel.bound κ < ∞ :=
  h.exists_univ_le.choose_spec.1


theorem IsFiniteKernel.bound_ne_top (κ : Kernel α β) [IsFiniteKernel κ] :
    IsFiniteKernel.bound κ ≠ ∞ :=
  (IsFiniteKernel.bound_lt_top κ).ne


theorem Kernel.measure_le_bound (κ : Kernel α β) [h : IsFiniteKernel κ] (a : α) (s : Set β) :
    κ a s ≤ IsFiniteKernel.bound κ :=
  (measure_mono (Set.subset_univ s)).trans (h.exists_univ_le.choose_spec.2 a)


instance isFiniteKernel_zero (α β : Type*) {_ : MeasurableSpace α} {_ : MeasurableSpace β} :
    IsFiniteKernel (0 : Kernel α β) :=
  ⟨⟨0, ENNReal.coe_lt_top, fun _ => by
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        ι : Type u_3
        mα : MeasurableSpace α✝
        mβ : MeasurableSpace β✝
        α : Type u_4
        β : Type u_5
        x✝² : MeasurableSpace α
        x✝¹ : MeasurableSpace β
        x✝ : α
        ⊢ LE.le ((0 x✝) Set.univ) 0
      -/
      simp only [Kernel.zero_apply, Measure.coe_zero, Pi.zero_apply, le_zero_iff]⟩⟩
      /-
        🎉 no goals
      -/


instance IsFiniteKernel.add (κ η : Kernel α β) [IsFiniteKernel κ] [IsFiniteKernel η] :
    IsFiniteKernel (κ + η) := by
  refine ⟨⟨IsFiniteKernel.bound κ + IsFiniteKernel.bound η,
    ENNReal.add_lt_top.mpr ⟨IsFiniteKernel.bound_lt_top κ, IsFiniteKernel.bound_lt_top η⟩,
    fun a => ?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ LE.le (((HAdd.hAdd κ η) a) Set.univ) (HAdd.hAdd (ProbabilityTheory.IsFiniteK …
  -/
  exact add_le_add (Kernel.measure_le_bound _ _ _) (Kernel.measure_le_bound _ _ _)
  /-
    🎉 no goals
  -/


lemma isFiniteKernel_of_le {κ ν : Kernel α β} [hν : IsFiniteKernel ν] (hκν : κ ≤ ν) :
    IsFiniteKernel κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ ν : ProbabilityTheory.Kernel α β
    hν : ProbabilityTheory.IsFiniteKernel ν
    hκν : LE.le κ ν
    ⊢ ProbabilityTheory.IsFiniteKernel κ
  -/
  refine ⟨hν.bound, hν.bound_lt_top, fun a ↦ (hκν _ _).trans (Kernel.measure_le_bound ν a Set.univ)⟩
  /-
    🎉 no goals
  -/


instance IsMarkovKernel.is_probability_measure' [IsMarkovKernel κ] (a : α) :
    IsProbabilityMeasure (κ a) :=
  IsMarkovKernel.isProbabilityMeasure a


instance : IsZeroOrMarkovKernel (0 : Kernel α β) := ⟨Or.inl rfl⟩


instance (priority := 100) IsMarkovKernel.IsZeroOrMarkovKernel [h : IsMarkovKernel κ] :
    IsZeroOrMarkovKernel κ := ⟨Or.inr h⟩


instance (priority := 100) IsZeroOrMarkovKernel.isZeroOrProbabilityMeasure
    [IsZeroOrMarkovKernel κ] (a : α) : IsZeroOrProbabilityMeasure (κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    a : α
    ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (κ a)
  -/
  rcases eq_zero_or_isMarkovKernel κ with rfl | h'
    /-
      case inl
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      η : ProbabilityTheory.Kernel α β
      a : α
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel 0
      ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (0 a)
    -/
  · simp only [Kernel.zero_apply]
    /-
      case inl
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      η : ProbabilityTheory.Kernel α β
      a : α
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel 0
      ⊢ MeasureTheory.IsZeroOrProbabilityMeasure 0
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ η : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel κ
      a : α
      h' : ProbabilityTheory.IsMarkovKernel κ
      ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (κ a)
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


instance IsFiniteKernel.isFiniteMeasure [IsFiniteKernel κ] (a : α) : IsFiniteMeasure (κ a) :=
  ⟨(Kernel.measure_le_bound κ a Set.univ).trans_lt (IsFiniteKernel.bound_lt_top κ)⟩


instance (priority := 100) IsZeroOrMarkovKernel.isFiniteKernel [h : IsZeroOrMarkovKernel κ] :
    IsFiniteKernel κ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    h : ProbabilityTheory.IsZeroOrMarkovKernel κ
    ⊢ ProbabilityTheory.IsFiniteKernel κ
  -/
  rcases eq_zero_or_isMarkovKernel κ with rfl | _h'
    /-
      case inl
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      η : ProbabilityTheory.Kernel α β
      h : ProbabilityTheory.IsZeroOrMarkovKernel 0
      ⊢ ProbabilityTheory.IsFiniteKernel 0
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ η : ProbabilityTheory.Kernel α β
      h : ProbabilityTheory.IsZeroOrMarkovKernel κ
      _h' : ProbabilityTheory.IsMarkovKernel κ
      ⊢ ProbabilityTheory.IsFiniteKernel κ
    -/
  · exact ⟨⟨1, ENNReal.one_lt_top, fun _ => prob_le_one⟩⟩
    /-
      🎉 no goals
    -/


@[ext]
theorem ext (h : ∀ a, κ a = η a) : κ = η := DFunLike.ext _ _ h


theorem ext_iff' : κ = η ↔ ∀ a s, MeasurableSet s → κ a s = η a s := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    ⊢ Iff (Eq κ η) (∀ (a : α) (s : Set β), MeasurableSet s → Eq ((κ a) s) ((η a) s))
  -/
  simp_rw [Kernel.ext_iff, Measure.ext_iff]
  /-
    🎉 no goals
  -/


theorem ext_fun (h : ∀ a f, Measurable f → ∫⁻ b, f b ∂κ a = ∫⁻ b, f b ∂η a) :
    κ = η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    h : ∀ (a : α) (f : β → ENNReal), Measurable f → Eq (MeasureTheory.lintegral (κ …
    ⊢ Eq κ η
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    h : ∀ (a : α) (f : β → ENNReal), Measurable f → Eq (MeasureTheory.lintegral (κ …
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((κ a) s) ((η a) s)
  -/
  specialize h a (s.indicator fun _ => 1) (Measurable.indicator measurable_const hs)
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    a : α
    s : Set β
    hs : MeasurableSet s
    h : Eq (MeasureTheory.lintegral (κ a) fun b => s.indicator (fun x => 1) b) (Me …
    ⊢ Eq ((κ a) s) ((η a) s)
  -/
  simp_rw [lintegral_indicator_const hs, one_mul] at h
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    a : α
    s : Set β
    hs : MeasurableSet s
    h : Eq ((κ a) s) ((η a) s)
    ⊢ Eq ((κ a) s) ((η a) s)
  -/
  rw [h]
  /-
    🎉 no goals
  -/


theorem ext_fun_iff : κ = η ↔ ∀ a f, Measurable f → ∫⁻ b, f b ∂κ a = ∫⁻ b, f b ∂η a :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       mα : MeasurableSpace α
                       mβ : MeasurableSpace β
                       κ η : ProbabilityTheory.Kernel α β
                       h : Eq κ η
                       a : α
                       f : β → ENNReal
                       x✝ : Measurable f
                       ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => f b) (MeasureTheory.lintegral (η  …
                     -/
  ⟨fun h a f _ => by rw [h], ext_fun⟩
                     /-
                       🎉 no goals
                     -/


protected theorem measurable_coe (κ : Kernel α β) {s : Set β} (hs : MeasurableSet s) :
    Measurable fun a => κ a s :=
  (Measure.measurable_coe hs).comp κ.measurable


lemma apply_congr_of_mem_measurableAtom (κ : Kernel α β) {y' y : α} (hy' : y' ∈ measurableAtom y) :
    κ y' = κ y := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    y' y : α
    hy' : Membership.mem (measurableAtom y) y'
    ⊢ Eq (κ y') (κ y)
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    y' y : α
    hy' : Membership.mem (measurableAtom y) y'
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((κ y') s) ((κ y) s)
  -/
  exact mem_of_mem_measurableAtom hy' (κ.measurable_coe hs (measurableSet_singleton (κ y s))) rfl
  /-
    🎉 no goals
  -/


/-- Sum of an indexed family of kernels. -/
protected noncomputable def sum [Countable ι] (κ : ι → Kernel α β) : Kernel α β where
  toFun a := Measure.sum fun n => κ n a
  measurable' := by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ✝ η : ProbabilityTheory.Kernel α β
      inst✝ : Countable ι
      κ : ι → ProbabilityTheory.Kernel α β
      ⊢ Measurable fun a => MeasureTheory.Measure.sum fun n => (κ n) a
    -/
    refine Measure.measurable_of_measurable_coe _ fun s hs => ?_
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ✝ η : ProbabilityTheory.Kernel α β
      inst✝ : Countable ι
      κ : ι → ProbabilityTheory.Kernel α β
      s : Set β
      hs : MeasurableSet s
      ⊢ Measurable fun b => (MeasureTheory.Measure.sum fun n => (κ n) b) s
    -/
    simp_rw [Measure.sum_apply _ hs]
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ✝ η : ProbabilityTheory.Kernel α β
      inst✝ : Countable ι
      κ : ι → ProbabilityTheory.Kernel α β
      s : Set β
      hs : MeasurableSet s
      ⊢ Measurable fun b => tsum fun i => ((κ i) b) s
    -/
    exact Measurable.ennreal_tsum fun n => Kernel.measurable_coe (κ n) hs
    /-
      🎉 no goals
    -/


theorem sum_apply [Countable ι] (κ : ι → Kernel α β) (a : α) :
    Kernel.sum κ a = Measure.sum fun n => κ n a :=
  rfl


theorem sum_apply' [Countable ι] (κ : ι → Kernel α β) (a : α) {s : Set β} (hs : MeasurableSet s) :
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             ι : Type u_3
                                             mα : MeasurableSpace α
                                             mβ : MeasurableSpace β
                                             inst✝ : Countable ι
                                             κ : ι → ProbabilityTheory.Kernel α β
                                             a : α
                                             s : Set β
                                             hs : MeasurableSet s
                                             ⊢ Eq (((ProbabilityTheory.Kernel.sum κ) a) s) (tsum fun n => ((κ n) a) s)
                                           -/
    Kernel.sum κ a s = ∑' n, κ n a s := by rw [sum_apply κ a, Measure.sum_apply _ hs]
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem sum_zero [Countable ι] : (Kernel.sum fun _ : ι => (0 : Kernel α β)) = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    ⊢ Eq (ProbabilityTheory.Kernel.sum fun x => 0) 0
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (((ProbabilityTheory.Kernel.sum fun x => 0) a) s) ((0 a) s)
  -/
  rw [sum_apply' _ a hs]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (tsum fun n => (0 a) s) ((0 a) s)
  -/
  simp only [zero_apply, Measure.coe_zero, Pi.zero_apply, tsum_zero]
  /-
    🎉 no goals
  -/


theorem sum_comm [Countable ι] (κ : ι → ι → Kernel α β) :
    (Kernel.sum fun n => Kernel.sum (κ n)) = Kernel.sum fun m => Kernel.sum fun n => κ n m := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    κ : ι → ι → ProbabilityTheory.Kernel α β
    ⊢ Eq (ProbabilityTheory.Kernel.sum fun n => ProbabilityTheory.Kernel.sum (κ n) …
  -/
  ext a s; simp_rw [sum_apply]; rw [Measure.sum_comm]
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem sum_fintype [Fintype ι] (κ : ι → Kernel α β) : Kernel.sum κ = ∑ i, κ i := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Fintype ι
    κ : ι → ProbabilityTheory.Kernel α β
    ⊢ Eq (ProbabilityTheory.Kernel.sum κ) (Finset.univ.sum fun i => κ i)
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Fintype ι
    κ : ι → ProbabilityTheory.Kernel α β
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (((ProbabilityTheory.Kernel.sum κ) a) s) (((Finset.univ.sum fun i => κ i) …
  -/
  simp only [sum_apply' κ a hs, finset_sum_apply' _ κ a s, tsum_fintype]
  /-
    🎉 no goals
  -/


theorem sum_add [Countable ι] (κ η : ι → Kernel α β) :
    (Kernel.sum fun n => κ n + η n) = Kernel.sum κ + Kernel.sum η := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    κ η : ι → ProbabilityTheory.Kernel α β
    ⊢ Eq (ProbabilityTheory.Kernel.sum fun n => HAdd.hAdd (κ n) (η n)) (HAdd.hAdd  …
  -/
  ext a s hs
  simp only [coe_add, Pi.add_apply, sum_apply, Measure.sum_apply _ hs, Pi.add_apply,
    Measure.coe_add, tsum_add ENNReal.summable ENNReal.summable]


/-- A kernel is s-finite if it can be written as the sum of countably many finite kernels. -/
class _root_.ProbabilityTheory.IsSFiniteKernel (κ : Kernel α β) : Prop where
  tsum_finite : ∃ κs : ℕ → Kernel α β, (∀ n, IsFiniteKernel (κs n)) ∧ κ = Kernel.sum κs


instance (priority := 100) IsFiniteKernel.isSFiniteKernel [h : IsFiniteKernel κ] :
    IsSFiniteKernel κ :=
  ⟨⟨fun n => if n = 0 then κ else 0, fun n => by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        κ η : ProbabilityTheory.Kernel α β
        h : ProbabilityTheory.IsFiniteKernel κ
        n : Nat
        ⊢ ProbabilityTheory.IsFiniteKernel ((fun n => ite (Eq n 0) κ 0) n)
      -/
      simp only; split_ifs
        /-
          case pos
          α : Type u_1
          β : Type u_2
          ι : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          κ η : ProbabilityTheory.Kernel α β
          h : ProbabilityTheory.IsFiniteKernel κ
          n : Nat
          h✝ : Eq n 0
          ⊢ ProbabilityTheory.IsFiniteKernel κ
        -/
      · exact h
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          β : Type u_2
          ι : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          κ η : ProbabilityTheory.Kernel α β
          h : ProbabilityTheory.IsFiniteKernel κ
          n : Nat
          h✝ : Not (Eq n 0)
          ⊢ ProbabilityTheory.IsFiniteKernel 0
        -/
      · infer_instance, by
        /-
          🎉 no goals
        -/
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        κ η : ProbabilityTheory.Kernel α β
        h : ProbabilityTheory.IsFiniteKernel κ
        ⊢ Eq κ (ProbabilityTheory.Kernel.sum fun n => ite (Eq n 0) κ 0)
      -/
      ext a s hs
      /-
        case h.h
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        κ η : ProbabilityTheory.Kernel α β
        h : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Set β
        hs : MeasurableSet s
        ⊢ Eq ((κ a) s) (((ProbabilityTheory.Kernel.sum fun n => ite (Eq n 0) κ 0) a) s)
      -/
      rw [Kernel.sum_apply' _ _ hs]
      have : (fun i => ((ite (i = 0) κ 0) a) s) = fun i => ite (i = 0) (κ a s) 0 := by
        ext1 i; split_ifs <;> rfl
      /-
        case h.h
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        κ η : ProbabilityTheory.Kernel α β
        h : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Set β
        hs : MeasurableSet s
        this : Eq (fun i => ((ite (Eq i 0) κ 0) a) s) fun i => ite (Eq i 0) ((κ a) s) 0
        ⊢ Eq ((κ a) s) (tsum fun n => ((ite (Eq n 0) κ 0) a) s)
      -/
      rw [this, tsum_ite_eq]⟩⟩
      /-
        🎉 no goals
      -/


/-- A sequence of finite kernels such that `κ = ProbabilityTheory.Kernel.sum (seq κ)`. See
`ProbabilityTheory.Kernel.isFiniteKernel_seq` and `ProbabilityTheory.Kernel.kernel_sum_seq`. -/
noncomputable def seq (κ : Kernel α β) [h : IsSFiniteKernel κ] : ℕ → Kernel α β :=
  h.tsum_finite.choose


theorem kernel_sum_seq (κ : Kernel α β) [h : IsSFiniteKernel κ] : Kernel.sum (seq κ) = κ :=
  h.tsum_finite.choose_spec.2.symm


theorem measure_sum_seq (κ : Kernel α β) [h : IsSFiniteKernel κ] (a : α) :
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   mα : MeasurableSpace α
                                                   mβ : MeasurableSpace β
                                                   κ : ProbabilityTheory.Kernel α β
                                                   h : ProbabilityTheory.IsSFiniteKernel κ
                                                   a : α
                                                   ⊢ Eq (MeasureTheory.Measure.sum fun n => (κ.seq n) a) (κ a)
                                                 -/
    (Measure.sum fun n => seq κ n a) = κ a := by rw [← Kernel.sum_apply, kernel_sum_seq κ]
                                                 /-
                                                   🎉 no goals
                                                 -/


instance isFiniteKernel_seq (κ : Kernel α β) [h : IsSFiniteKernel κ] (n : ℕ) :
    IsFiniteKernel (Kernel.seq κ n) :=
  h.tsum_finite.choose_spec.1 n


instance _root_.ProbabilityTheory.IsSFiniteKernel.sFinite [IsSFiniteKernel κ] (a : α) :
    SFinite (κ a) :=
  ⟨⟨fun n ↦ seq κ n a, inferInstance, (measure_sum_seq κ a).symm⟩⟩


instance IsSFiniteKernel.add (κ η : Kernel α β) [IsSFiniteKernel κ] [IsSFiniteKernel η] :
    IsSFiniteKernel (κ + η) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ η✝ κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ ProbabilityTheory.IsSFiniteKernel (HAdd.hAdd κ η)
  -/
  refine ⟨⟨fun n => seq κ n + seq η n, fun n => inferInstance, ?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ η✝ κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (HAdd.hAdd κ η) (ProbabilityTheory.Kernel.sum fun n => HAdd.hAdd (κ.seq n …
  -/
  rw [sum_add, kernel_sum_seq κ, kernel_sum_seq η]
  /-
    🎉 no goals
  -/


theorem IsSFiniteKernel.finset_sum {κs : ι → Kernel α β} (I : Finset ι)
    (h : ∀ i ∈ I, IsSFiniteKernel (κs i)) : IsSFiniteKernel (∑ i ∈ I, κs i) := by
  classical
  induction' I using Finset.induction with i I hi_nmem_I h_ind h
  · rw [Finset.sum_empty]; infer_instance
  · rw [Finset.sum_insert hi_nmem_I]
    haveI : IsSFiniteKernel (κs i) := h i (Finset.mem_insert_self _ _)
    have : IsSFiniteKernel (∑ x ∈ I, κs x) :=
      h_ind fun i hiI => h i (Finset.mem_insert_of_mem hiI)
    exact IsSFiniteKernel.add _ _


theorem isSFiniteKernel_sum_of_denumerable [Denumerable ι] {κs : ι → Kernel α β}
    (hκs : ∀ n, IsSFiniteKernel (κs n)) : IsSFiniteKernel (Kernel.sum κs) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Denumerable ι
    κs : ι → ProbabilityTheory.Kernel α β
    hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
    ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.sum κs)
  -/
  let e : ℕ ≃ ι × ℕ := (Denumerable.eqv (ι × ℕ)).symm
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Denumerable ι
    κs : ι → ProbabilityTheory.Kernel α β
    hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
    e : Equiv Nat (Prod ι Nat) := (Denumerable.eqv (Prod ι Nat)).symm
    ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.sum κs)
  -/
  refine ⟨⟨fun n => seq (κs (e n).1) (e n).2, inferInstance, ?_⟩⟩
  have hκ_eq : Kernel.sum κs = Kernel.sum fun n => Kernel.sum (seq (κs n)) := by
    simp_rw [kernel_sum_seq]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Denumerable ι
    κs : ι → ProbabilityTheory.Kernel α β
    hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
    e : Equiv Nat (Prod ι Nat) := (Denumerable.eqv (Prod ι Nat)).symm
    hκ_eq : Eq (ProbabilityTheory.Kernel.sum κs) (ProbabilityTheory.Kernel.sum fun …
    ⊢ Eq (ProbabilityTheory.Kernel.sum κs) (ProbabilityTheory.Kernel.sum fun n =>  …
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Denumerable ι
    κs : ι → ProbabilityTheory.Kernel α β
    hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
    e : Equiv Nat (Prod ι Nat) := (Denumerable.eqv (Prod ι Nat)).symm
    hκ_eq : Eq (ProbabilityTheory.Kernel.sum κs) (ProbabilityTheory.Kernel.sum fun …
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (((ProbabilityTheory.Kernel.sum κs) a) s) (((ProbabilityTheory.Kernel.sum …
  -/
  rw [hκ_eq]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Denumerable ι
    κs : ι → ProbabilityTheory.Kernel α β
    hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
    e : Equiv Nat (Prod ι Nat) := (Denumerable.eqv (Prod ι Nat)).symm
    hκ_eq : Eq (ProbabilityTheory.Kernel.sum κs) (ProbabilityTheory.Kernel.sum fun …
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (((ProbabilityTheory.Kernel.sum fun n => ProbabilityTheory.Kernel.sum (κs …
  -/
  simp_rw [Kernel.sum_apply' _ _ hs]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Denumerable ι
    κs : ι → ProbabilityTheory.Kernel α β
    hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
    e : Equiv Nat (Prod ι Nat) := (Denumerable.eqv (Prod ι Nat)).symm
    hκ_eq : Eq (ProbabilityTheory.Kernel.sum κs) (ProbabilityTheory.Kernel.sum fun …
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (tsum fun n => tsum fun n_1 => (((κs n).seq n_1) a) s) (tsum fun n => ((( …
  -/
  change (∑' i, ∑' m, seq (κs i) m a s) = ∑' n, (fun im : ι × ℕ => seq (κs im.fst) im.snd a s) (e n)
  rw [e.tsum_eq (fun im : ι × ℕ => seq (κs im.fst) im.snd a s),
    tsum_prod' ENNReal.summable fun _ => ENNReal.summable]


theorem isSFiniteKernel_sum [Countable ι] {κs : ι → Kernel α β}
    (hκs : ∀ n, IsSFiniteKernel (κs n)) : IsSFiniteKernel (Kernel.sum κs) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    κs : ι → ProbabilityTheory.Kernel α β
    hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
    ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.sum κs)
  -/
  cases fintypeOrInfinite ι
    /-
      case inl
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝ : Countable ι
      κs : ι → ProbabilityTheory.Kernel α β
      hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
      val✝ : Fintype ι
      ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.sum κs)
    -/
  · rw [sum_fintype]
    /-
      case inl
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝ : Countable ι
      κs : ι → ProbabilityTheory.Kernel α β
      hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
      val✝ : Fintype ι
      ⊢ ProbabilityTheory.IsSFiniteKernel (Finset.univ.sum fun i => κs i)
    -/
    exact IsSFiniteKernel.finset_sum Finset.univ fun i _ => hκs i
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    κs : ι → ProbabilityTheory.Kernel α β
    hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
    val✝ : Infinite ι
    ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.sum κs)
  -/
  cases nonempty_denumerable ι
  /-
    case inr.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    κs : ι → ProbabilityTheory.Kernel α β
    hκs : ∀ (n : ι), ProbabilityTheory.IsSFiniteKernel (κs n)
    val✝¹ : Infinite ι
    val✝ : Denumerable ι
    ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.sum κs)
  -/
  exact isSFiniteKernel_sum_of_denumerable hκs
  /-
    🎉 no goals
  -/


