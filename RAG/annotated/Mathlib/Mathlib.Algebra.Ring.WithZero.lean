instance instLeftDistribClass [Mul α] [Add α] [LeftDistribClass α] :
    LeftDistribClass (WithZero α) where
  left_distrib a b c := by
    /-
      α : Type u_1
      inst✝² : Mul α
      inst✝¹ : Add α
      inst✝ : LeftDistribClass α
      a b c : WithZero α
      ⊢ Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c))
    -/
    cases' a with a; · rfl
                       /-
                         🎉 no goals
                       -/
    /-
      case h₂
      α : Type u_1
      inst✝² : Mul α
      inst✝¹ : Add α
      inst✝ : LeftDistribClass α
      b c : WithZero α
      a : α
      ⊢ Eq (HMul.hMul (↑a) (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul (↑a) b) (HMul.hMul …
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
    cases' b with b <;> cases' c with c <;> try rfl
    /-
      case h₂.h₂.h₂
      α : Type u_1
      inst✝² : Mul α
      inst✝¹ : Add α
      inst✝ : LeftDistribClass α
      a b c : α
      ⊢ Eq (HMul.hMul (↑a) (HAdd.hAdd ↑b ↑c)) (HAdd.hAdd (HMul.hMul ↑a ↑b) (HMul.hMu …
    -/
    exact congr_arg some (left_distrib _ _ _)
    /-
      🎉 no goals
    -/


instance instRightDistribClass [Mul α] [Add α] [RightDistribClass α] :
    RightDistribClass (WithZero α) where
  right_distrib a b c := by
    /-
      α : Type u_1
      inst✝² : Mul α
      inst✝¹ : Add α
      inst✝ : RightDistribClass α
      a b c : WithZero α
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
    -/
    cases' c with c
      /-
        case h₁
        α : Type u_1
        inst✝² : Mul α
        inst✝¹ : Add α
        inst✝ : RightDistribClass α
        a b : WithZero α
        ⊢ Eq (HMul.hMul (HAdd.hAdd a b) 0) (HAdd.hAdd (HMul.hMul a 0) (HMul.hMul b 0))
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case h₂
      α : Type u_1
      inst✝² : Mul α
      inst✝¹ : Add α
      inst✝ : RightDistribClass α
      a b : WithZero α
      c : α
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) ↑c) (HAdd.hAdd (HMul.hMul a ↑c) (HMul.hMul b ↑ …
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
    cases' a with a <;> cases' b with b <;> try rfl
    /-
      case h₂.h₂.h₂
      α : Type u_1
      inst✝² : Mul α
      inst✝¹ : Add α
      inst✝ : RightDistribClass α
      c a b : α
      ⊢ Eq (HMul.hMul (HAdd.hAdd ↑a ↑b) ↑c) (HAdd.hAdd (HMul.hMul ↑a ↑c) (HMul.hMul  …
    -/
    exact congr_arg some (right_distrib _ _ _)
    /-
      🎉 no goals
    -/


instance instDistrib [Distrib α] : Distrib (WithZero α) where
  left_distrib := left_distrib
  right_distrib := right_distrib


instance instSemiring [Semiring α] : Semiring (WithZero α) :=
  { addMonoidWithOne, addCommMonoid, mulZeroClass, monoidWithZero, instDistrib with }


