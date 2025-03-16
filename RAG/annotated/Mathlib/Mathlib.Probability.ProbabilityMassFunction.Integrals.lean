theorem integral_eq_tsum (p : PMF α) (f : α → E) (hf : Integrable f p.toMeasure) :
    ∫ a, f a ∂(p.toMeasure) = ∑' a, (p a).toReal • f a := calc
                                                 /-
                                                   α : Type u_1
                                                   inst✝⁴ : MeasurableSpace α
                                                   inst✝³ : MeasurableSingletonClass α
                                                   E : Type u_2
                                                   inst✝² : NormedAddCommGroup E
                                                   inst✝¹ : NormedSpace Real E
                                                   inst✝ : CompleteSpace E
                                                   p : PMF α
                                                   f : α → E
                                                   hf : MeasureTheory.Integrable f p.toMeasure
                                                   ⊢ Eq (MeasureTheory.integral p.toMeasure fun a => f a) (MeasureTheory.integral …
                                                 -/
  _ = ∫ a in p.support, f a ∂(p.toMeasure) := by rw [restrict_toMeasure_support p]
                                                 /-
                                                   🎉 no goals
                                                 -/
  _ = ∑' (a : support p), (p.toMeasure {a.val}).toReal • f a := by
    /-
      α : Type u_1
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSingletonClass α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      p : PMF α
      f : α → E
      hf : MeasureTheory.Integrable f p.toMeasure
      ⊢ Eq (MeasureTheory.integral (p.toMeasure.restrict p.support) fun a => f a) (t …
    -/
    apply integral_countable f p.support_countable
    /-
      α : Type u_1
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSingletonClass α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      p : PMF α
      f : α → E
      hf : MeasureTheory.Integrable f p.toMeasure
      ⊢ MeasureTheory.IntegrableOn f p.support p.toMeasure
    -/
    rwa [IntegrableOn, restrict_toMeasure_support p]
    /-
      🎉 no goals
    -/
  _ = ∑' (a : support p), (p a).toReal • f a := by
    /-
      α : Type u_1
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSingletonClass α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      p : PMF α
      f : α → E
      hf : MeasureTheory.Integrable f p.toMeasure
      ⊢ Eq (tsum fun a => HSMul.hSMul (p.toMeasure (Singleton.singleton ↑a)).toReal  …
    -/
    congr with x; congr 2
    /-
      case e_f.h.e_a.e_a
      α : Type u_1
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSingletonClass α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      p : PMF α
      f : α → E
      hf : MeasureTheory.Integrable f p.toMeasure
      x : ↑p.support
      ⊢ Eq (p.toMeasure (Singleton.singleton ↑x)) (p ↑x)
    -/
    apply PMF.toMeasure_apply_singleton p x (MeasurableSet.singleton _)
    /-
      🎉 no goals
    -/
  _ = ∑' a, (p a).toReal • f a :=
    tsum_subtype_eq_of_support_subset <| calc
      (fun a ↦ (p a).toReal • f a).support ⊆ (fun a ↦ (p a).toReal).support :=
        Function.support_smul_subset_left _ _
                                             /-
                                               α : Type u_1
                                               inst✝⁴ : MeasurableSpace α
                                               inst✝³ : MeasurableSingletonClass α
                                               E : Type u_2
                                               inst✝² : NormedAddCommGroup E
                                               inst✝¹ : NormedSpace Real E
                                               inst✝ : CompleteSpace E
                                               p : PMF α
                                               f : α → E
                                               hf : MeasureTheory.Integrable f p.toMeasure
                                               x : α
                                               h1 : Membership.mem (Function.support fun a => (p a).toReal) x
                                               h2 : Eq (p x) 0
                                               ⊢ Eq ((fun a => (p a).toReal) x) 0
                                             -/
      _ ⊆ support p := fun x h1 h2 => h1 (by simp [h2])
                                             /-
                                               🎉 no goals
                                             -/


theorem integral_eq_sum [Fintype α] (p : PMF α) (f : α → E) :
    ∫ a, f a ∂(p.toMeasure) = ∑ a, (p a).toReal • f a := by
  /-
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSingletonClass α
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : Fintype α
    p : PMF α
    f : α → E
    ⊢ Eq (MeasureTheory.integral p.toMeasure fun a => f a) (Finset.univ.sum fun a  …
  -/
  rw [integral_fintype _ .of_finite]
  /-
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSingletonClass α
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : Fintype α
    p : PMF α
    f : α → E
    ⊢ Eq (Finset.univ.sum fun x => HSMul.hSMul (p.toMeasure (Singleton.singleton x …
  -/
  congr with x; congr 2
  /-
    case e_f.h.e_a.e_a
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSingletonClass α
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : Fintype α
    p : PMF α
    f : α → E
    x : α
    ⊢ Eq (p.toMeasure (Singleton.singleton x)) (p x)
  -/
  exact PMF.toMeasure_apply_singleton p x (MeasurableSet.singleton _)
  /-
    🎉 no goals
  -/


theorem bernoulli_expectation {p : ℝ≥0∞} (h : p ≤ 1) :
                                                                  /-
                                                                    p : ENNReal
                                                                    h : LE.le p 1
                                                                    ⊢ Eq (MeasureTheory.integral (PMF.bernoulli p h).toMeasure fun b => cond b 1 0 …
                                                                  -/
    ∫ b, cond b 1 0 ∂((bernoulli p h).toMeasure) = p.toReal := by simp [integral_eq_sum]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


