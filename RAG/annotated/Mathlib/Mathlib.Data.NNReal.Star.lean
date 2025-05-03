instance : StarRing ℝ≥0 := starRingOfComm


instance : TrivialStar ℝ≥0 where
  star_trivial _ := rfl


instance : StarModule ℝ≥0 ℝ where
                  /-
                    ⊢ ∀ (r : NNReal) (a : Real), Eq (Star.star (HSMul.hSMul r a)) (HSMul.hSMul (St …
                  -/
  star_smul := by simp only [star_trivial, eq_self_iff_true, forall_const]
                  /-
                    🎉 no goals
                  -/


instance {E : Type*} [AddCommMonoid E] [Star E] [Module ℝ E] [StarModule ℝ E] :
    StarModule ℝ≥0 E where
  star_smul _ := star_smul (_ : ℝ)

