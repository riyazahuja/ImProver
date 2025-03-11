theorem eventually_nnnorm_sub_lt (x₀ : E) {ε : ℝ≥0} (ε_pos : 0 < ε) :
    ∀ᶠ x in 𝓝 x₀, ‖x - x₀‖₊ < ε :=
                                                                     /-
                                                                       E : Type u_1
                                                                       inst✝ : SeminormedAddCommGroup E
                                                                       x₀ : E
                                                                       ε : NNReal
                                                                       ε_pos : LT.lt 0 ε
                                                                       ⊢ LT.lt ((fun x => NNNorm.nnnorm (HSub.hSub (id x) x₀)) x₀) ε
                                                                     -/
  (continuousAt_id.sub continuousAt_const).nnnorm (gt_mem_nhds <| by simpa)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem eventually_norm_sub_lt (x₀ : E) {ε : ℝ} (ε_pos : 0 < ε) :
    ∀ᶠ x in 𝓝 x₀, ‖x - x₀‖ < ε :=
                                                                   /-
                                                                     E : Type u_1
                                                                     inst✝ : SeminormedAddCommGroup E
                                                                     x₀ : E
                                                                     ε : Real
                                                                     ε_pos : LT.lt 0 ε
                                                                     ⊢ LT.lt ((fun x => Norm.norm (HSub.hSub (id x) x₀)) x₀) ε
                                                                   -/
  (continuousAt_id.sub continuousAt_const).norm (gt_mem_nhds <| by simpa)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

