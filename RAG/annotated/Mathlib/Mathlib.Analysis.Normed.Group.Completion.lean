instance [UniformSpace E] [Norm E] : Norm (Completion E) where
  norm := Completion.extension Norm.norm


@[simp]
theorem norm_coe {E} [SeminormedAddCommGroup E] (x : E) : ‖(x : Completion E)‖ = ‖x‖ :=
  Completion.extension_coe uniformContinuous_norm x


instance [SeminormedAddCommGroup E] : NormedAddCommGroup (Completion E) where
  dist_eq x y := by
    /-
      E : Type u_1
      inst✝ : SeminormedAddCommGroup E
      x y : UniformSpace.Completion E
      ⊢ Eq (Dist.dist x y) (Norm.norm (HSub.hSub x y))
    -/
    induction x, y using Completion.induction_on₂
      /-
        case hp
        E : Type u_1
        inst✝ : SeminormedAddCommGroup E
        ⊢ IsClosed (setOf fun x => Eq (Dist.dist x.1 x.2) (Norm.norm (HSub.hSub x.1 x. …
      -/
    · refine isClosed_eq (Completion.uniformContinuous_extension₂ _).continuous ?_
      /-
        case hp
        E : Type u_1
        inst✝ : SeminormedAddCommGroup E
        ⊢ Continuous fun x => Norm.norm (HSub.hSub x.1 x.2)
      -/
      exact Continuous.comp Completion.continuous_extension continuous_sub
      /-
        🎉 no goals
      -/
      /-
        case ih
        E : Type u_1
        inst✝ : SeminormedAddCommGroup E
        a✝ b✝ : E
        ⊢ Eq (Dist.dist (↑E a✝) (↑E b✝)) (Norm.norm (HSub.hSub (↑E a✝) (↑E b✝)))
      -/
    · rw [← Completion.coe_sub, norm_coe, Completion.dist_eq, dist_eq_norm]
      /-
        🎉 no goals
      -/


@[simp]
theorem nnnorm_coe {E} [SeminormedAddCommGroup E] (x : E) : ‖(x : Completion E)‖₊ = ‖x‖₊ := by
  /-
    E : Type u_2
    inst✝ : SeminormedAddCommGroup E
    x : E
    ⊢ Eq (NNNorm.nnnorm (↑E x)) (NNNorm.nnnorm x)
  -/
  simp [nnnorm]
  /-
    🎉 no goals
  -/


