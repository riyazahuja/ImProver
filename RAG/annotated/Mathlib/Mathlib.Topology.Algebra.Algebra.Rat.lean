/-- The action induced by `DivisionRing.toRatAlgebra` is continuous. -/
instance DivisionRing.continuousConstSMul_rat {A} [DivisionRing A] [TopologicalSpace A]
    [ContinuousMul A] [CharZero A] : ContinuousConstSMul ℚ A :=
               /-
                 A : Type u_1
                 inst✝³ : DivisionRing A
                 inst✝² : TopologicalSpace A
                 inst✝¹ : ContinuousMul A
                 inst✝ : CharZero A
                 r : Rat
                 ⊢ Continuous fun x => HSMul.hSMul r x
               -/
  ⟨fun r => by simpa only [Algebra.smul_def] using continuous_const.mul continuous_id⟩
               /-
                 🎉 no goals
               -/


