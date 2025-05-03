@[bound]
theorem norm_smul_le (r : α) (x : β) : ‖r • x‖ ≤ ‖r‖ * ‖x‖ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : SeminormedAddGroup α
    inst✝² : SeminormedAddGroup β
    inst✝¹ : SMulZeroClass α β
    inst✝ : BoundedSMul α β
    r : α
    x : β
    ⊢ LE.le (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (Norm.norm x))
  -/
  simpa [smul_zero] using dist_smul_pair r 0 x
  /-
    🎉 no goals
  -/


@[bound]
theorem nnnorm_smul_le (r : α) (x : β) : ‖r • x‖₊ ≤ ‖r‖₊ * ‖x‖₊ :=
  norm_smul_le _ _


theorem dist_smul_le (s : α) (x y : β) : dist (s • x) (s • y) ≤ ‖s‖ * dist x y := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : SeminormedAddGroup α
    inst✝² : SeminormedAddGroup β
    inst✝¹ : SMulZeroClass α β
    inst✝ : BoundedSMul α β
    s : α
    x y : β
    ⊢ LE.le (Dist.dist (HSMul.hSMul s x) (HSMul.hSMul s y)) (HMul.hMul (Norm.norm  …
  -/
  simpa only [dist_eq_norm, sub_zero] using dist_smul_pair s x y
  /-
    🎉 no goals
  -/


theorem nndist_smul_le (s : α) (x y : β) : nndist (s • x) (s • y) ≤ ‖s‖₊ * nndist x y :=
  dist_smul_le s x y


theorem lipschitzWith_smul (s : α) : LipschitzWith ‖s‖₊ (s • · : β → β) :=
  lipschitzWith_iff_dist_le_mul.2 <| dist_smul_le _


theorem edist_smul_le (s : α) (x y : β) : edist (s • x) (s • y) ≤ ‖s‖₊ • edist x y :=
  lipschitzWith_smul s x y


/-- Left multiplication is bounded. -/
instance NonUnitalSeminormedRing.to_boundedSMul [NonUnitalSeminormedRing α] : BoundedSMul α α where
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝ : NonUnitalSeminormedRing α
                                  x y₁ y₂ : α
                                  ⊢ LE.le (Dist.dist (HSMul.hSMul x y₁) (HSMul.hSMul x y₂)) (HMul.hMul (Dist.dis …
                                -/
  dist_smul_pair' x y₁ y₂ := by simpa [mul_sub, dist_eq_norm] using norm_mul_le x (y₁ - y₂)
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝ : NonUnitalSeminormedRing α
                                  x₁ x₂ y : α
                                  ⊢ LE.le (Dist.dist (HSMul.hSMul x₁ y) (HSMul.hSMul x₂ y)) (HMul.hMul (Dist.dis …
                                -/
  dist_pair_smul' x₁ x₂ y := by simpa [sub_mul, dist_eq_norm] using norm_mul_le (x₁ - x₂) y
                                /-
                                  🎉 no goals
                                -/


/-- Right multiplication is bounded. -/
instance NonUnitalSeminormedRing.to_has_bounded_op_smul [NonUnitalSeminormedRing α] :
    BoundedSMul αᵐᵒᵖ α where
  dist_smul_pair' x y₁ y₂ := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : NonUnitalSeminormedRing α
      x : MulOpposite α
      y₁ y₂ : α
      ⊢ LE.le (Dist.dist (HSMul.hSMul x y₁) (HSMul.hSMul x y₂)) (HMul.hMul (Dist.dis …
    -/
    simpa [sub_mul, dist_eq_norm, mul_comm] using norm_mul_le (y₁ - y₂) x.unop
    /-
      🎉 no goals
    -/
  dist_pair_smul' x₁ x₂ y := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : NonUnitalSeminormedRing α
      x₁ x₂ : MulOpposite α
      y : α
      ⊢ LE.le (Dist.dist (HSMul.hSMul x₁ y) (HSMul.hSMul x₂ y)) (HMul.hMul (Dist.dis …
    -/
    simpa [mul_sub, dist_eq_norm, mul_comm] using norm_mul_le y (x₁ - x₂).unop
    /-
      🎉 no goals
    -/


theorem BoundedSMul.of_norm_smul_le (h : ∀ (r : α) (x : β), ‖r • x‖ ≤ ‖r‖ * ‖x‖) :
    BoundedSMul α β :=
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           inst✝² : SeminormedRing α
                                           inst✝¹ : SeminormedAddCommGroup β
                                           inst✝ : Module α β
                                           h : ∀ (r : α) (x : β), LE.le (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.no …
                                           a : α
                                           b₁ b₂ : β
                                           ⊢ LE.le (Dist.dist (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)) (HMul.hMul (Dist.dis …
                                         -/
  { dist_smul_pair' := fun a b₁ b₂ => by simpa [smul_sub, dist_eq_norm] using h a (b₁ - b₂)
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           inst✝² : SeminormedRing α
                                           inst✝¹ : SeminormedAddCommGroup β
                                           inst✝ : Module α β
                                           h : ∀ (r : α) (x : β), LE.le (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.no …
                                           a₁ a₂ : α
                                           b : β
                                           ⊢ LE.le (Dist.dist (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)) (HMul.hMul (Dist.dis …
                                         -/
    dist_pair_smul' := fun a₁ a₂ b => by simpa [sub_smul, dist_eq_norm] using h (a₁ - a₂) b }
                                         /-
                                           🎉 no goals
                                         -/


theorem BoundedSMul.of_nnnorm_smul_le (h : ∀ (r : α) (x : β), ‖r • x‖₊ ≤ ‖r‖₊ * ‖x‖₊) :
    BoundedSMul α β := .of_norm_smul_le h


theorem norm_smul (r : α) (x : β) : ‖r • x‖ = ‖r‖ * ‖x‖ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : NormedDivisionRing α
    inst✝² : SeminormedAddGroup β
    inst✝¹ : MulActionWithZero α β
    inst✝ : BoundedSMul α β
    r : α
    x : β
    ⊢ Eq (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (Norm.norm x))
  -/
  by_cases h : r = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝³ : NormedDivisionRing α
      inst✝² : SeminormedAddGroup β
      inst✝¹ : MulActionWithZero α β
      inst✝ : BoundedSMul α β
      r : α
      x : β
      h : Eq r 0
      ⊢ Eq (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (Norm.norm x))
    -/
  · simp [h, zero_smul α x]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝³ : NormedDivisionRing α
      inst✝² : SeminormedAddGroup β
      inst✝¹ : MulActionWithZero α β
      inst✝ : BoundedSMul α β
      r : α
      x : β
      h : Not (Eq r 0)
      ⊢ Eq (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (Norm.norm x))
    -/
  · refine le_antisymm (norm_smul_le r x) ?_
    calc
      ‖r‖ * ‖x‖ = ‖r‖ * ‖r⁻¹ • r • x‖ := by rw [inv_smul_smul₀ h]
      _ ≤ ‖r‖ * (‖r⁻¹‖ * ‖r • x‖) := by gcongr; apply norm_smul_le
      _ = ‖r • x‖ := by rw [norm_inv, ← mul_assoc, mul_inv_cancel₀ (mt norm_eq_zero.1 h), one_mul]


theorem nnnorm_smul (r : α) (x : β) : ‖r • x‖₊ = ‖r‖₊ * ‖x‖₊ :=
  NNReal.eq <| norm_smul r x


theorem dist_smul₀ (s : α) (x y : β) : dist (s • x) (s • y) = ‖s‖ * dist x y := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : NormedDivisionRing α
    inst✝² : SeminormedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : BoundedSMul α β
    s : α
    x y : β
    ⊢ Eq (Dist.dist (HSMul.hSMul s x) (HSMul.hSMul s y)) (HMul.hMul (Norm.norm s)  …
  -/
  simp_rw [dist_eq_norm, (norm_smul s (x - y)).symm, smul_sub]
  /-
    🎉 no goals
  -/


theorem nndist_smul₀ (s : α) (x y : β) : nndist (s • x) (s • y) = ‖s‖₊ * nndist x y :=
  NNReal.eq <| dist_smul₀ s x y


theorem edist_smul₀ (s : α) (x y : β) : edist (s • x) (s • y) = ‖s‖₊ • edist x y := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : NormedDivisionRing α
    inst✝² : SeminormedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : BoundedSMul α β
    s : α
    x y : β
    ⊢ Eq (EDist.edist (HSMul.hSMul s x) (HSMul.hSMul s y)) (HSMul.hSMul (NNNorm.nn …
  -/
  simp only [edist_nndist, nndist_smul₀, ENNReal.coe_mul, ENNReal.smul_def, smul_eq_mul]
  /-
    🎉 no goals
  -/


