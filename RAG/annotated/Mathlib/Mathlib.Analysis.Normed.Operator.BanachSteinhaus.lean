/-- This is the standard Banach-Steinhaus theorem, or Uniform Boundedness Principle.
If a family of continuous linear maps from a Banach space into a normed space is pointwise
bounded, then the norms of these linear maps are uniformly bounded.

See also `WithSeminorms.banach_steinhaus` for the general statement in barrelled spaces. -/
theorem banach_steinhaus {ι : Type*} [CompleteSpace E] {g : ι → E →SL[σ₁₂] F}
    (h : ∀ x, ∃ C, ∀ i, ‖g i x‖ ≤ C) : ∃ C', ∀ i, ‖g i‖ ≤ C' := by
  /-
    E : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    𝕜₂ : Type u_4
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    ι : Type u_5
    inst✝ : CompleteSpace E
    g : ι → ContinuousLinearMap σ₁₂ E F
    h : ∀ (x : E), Exists fun C => ∀ (i : ι), LE.le (Norm.norm ((g i) x)) C
    ⊢ Exists fun C' => ∀ (i : ι), LE.le (Norm.norm (g i)) C'
  -/
  rw [show (∃ C, ∀ i, ‖g i‖ ≤ C) ↔ _ from (NormedSpace.equicontinuous_TFAE g).out 5 2]
  /-
    E : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    𝕜₂ : Type u_4
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    ι : Type u_5
    inst✝ : CompleteSpace E
    g : ι → ContinuousLinearMap σ₁₂ E F
    h : ∀ (x : E), Exists fun C => ∀ (i : ι), LE.le (Norm.norm ((g i) x)) C
    ⊢ UniformEquicontinuous (Function.comp DFunLike.coe g)
  -/
  refine (norm_withSeminorms 𝕜₂ F).banach_steinhaus (fun _ x ↦ ?_)
  /-
    E : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    𝕜₂ : Type u_4
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    ι : Type u_5
    inst✝ : CompleteSpace E
    g : ι → ContinuousLinearMap σ₁₂ E F
    h : ∀ (x : E), Exists fun C => ∀ (i : ι), LE.le (Norm.norm ((g i) x)) C
    x✝ : Fin 1
    x : E
    ⊢ BddAbove (Set.range fun i => (normSeminorm 𝕜₂ F) ((g i) x))
  -/
  simpa [bddAbove_def, forall_mem_range] using h x
  /-
    🎉 no goals
  -/


/-- This version of Banach-Steinhaus is stated in terms of suprema of `↑‖·‖₊ : ℝ≥0∞`
for convenience. -/
theorem banach_steinhaus_iSup_nnnorm {ι : Type*} [CompleteSpace E] {g : ι → E →SL[σ₁₂] F}
    (h : ∀ x, (⨆ i, ↑‖g i x‖₊) < ∞) : (⨆ i, ↑‖g i‖₊) < ∞ := by
  /-
    E : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    𝕜₂ : Type u_4
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    ι : Type u_5
    inst✝ : CompleteSpace E
    g : ι → ContinuousLinearMap σ₁₂ E F
    h : ∀ (x : E), LT.lt (iSup fun i => ↑(NNNorm.nnnorm ((g i) x))) Top.top
    ⊢ LT.lt (iSup fun i => ↑(NNNorm.nnnorm (g i))) Top.top
  -/
  rw [show ((⨆ i, ↑‖g i‖₊) < ∞) ↔ _ from (NormedSpace.equicontinuous_TFAE g).out 8 2]
  /-
    E : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    𝕜₂ : Type u_4
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    ι : Type u_5
    inst✝ : CompleteSpace E
    g : ι → ContinuousLinearMap σ₁₂ E F
    h : ∀ (x : E), LT.lt (iSup fun i => ↑(NNNorm.nnnorm ((g i) x))) Top.top
    ⊢ UniformEquicontinuous (Function.comp DFunLike.coe g)
  -/
  refine (norm_withSeminorms 𝕜₂ F).banach_steinhaus (fun _ x ↦ ?_)
  /-
    E : Type u_1
    F : Type u_2
    𝕜 : Type u_3
    𝕜₂ : Type u_4
    inst✝⁷ : SeminormedAddCommGroup E
    inst✝⁶ : SeminormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    ι : Type u_5
    inst✝ : CompleteSpace E
    g : ι → ContinuousLinearMap σ₁₂ E F
    h : ∀ (x : E), LT.lt (iSup fun i => ↑(NNNorm.nnnorm ((g i) x))) Top.top
    x✝ : Fin 1
    x : E
    ⊢ BddAbove (Set.range fun i => (normSeminorm 𝕜₂ F) ((g i) x))
  -/
  simpa [← NNReal.bddAbove_coe, ← Set.range_comp] using ENNReal.iSup_coe_lt_top.1 (h x)
  /-
    🎉 no goals
  -/


/-- Given a *sequence* of continuous linear maps which converges pointwise and for which the
domain is complete, the Banach-Steinhaus theorem is used to guarantee that the limit map
is a *continuous* linear map as well. -/
abbrev continuousLinearMapOfTendsto {α : Type*} [CompleteSpace E] [T2Space F] {l : Filter α}
    [l.IsCountablyGenerated] [l.NeBot] (g : α → E →SL[σ₁₂] F) {f : E → F}
    (h : Tendsto (fun n x ↦ g n x) l (𝓝 f)) :
    E →SL[σ₁₂] F :=
  (norm_withSeminorms 𝕜₂ F).continuousLinearMapOfTendsto g h

