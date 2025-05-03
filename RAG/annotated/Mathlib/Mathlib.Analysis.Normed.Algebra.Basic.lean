theorem norm_le_norm_one (φ : characterSpace 𝕜 A) : ‖toNormedDual (φ : WeakDual 𝕜 A)‖ ≤ ‖(1 : A)‖ :=
  ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg (1 : A)) fun a =>
    mul_comm ‖a‖ ‖(1 : A)‖ ▸ spectrum.norm_le_norm_mul_of_mem (apply_mem_spectrum φ a)


instance [ProperSpace 𝕜] : CompactSpace (characterSpace 𝕜 A) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ProperSpace 𝕜
    ⊢ CompactSpace ↑(WeakDual.characterSpace 𝕜 A)
  -/
  rw [← isCompact_iff_compactSpace]
  have h : characterSpace 𝕜 A ⊆ toNormedDual ⁻¹' Metric.closedBall 0 ‖(1 : A)‖ := by
    intro φ hφ
    rw [Set.mem_preimage, mem_closedBall_zero_iff]
    exact (norm_le_norm_one ⟨φ, ⟨hφ.1, hφ.2⟩⟩ : _)
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ProperSpace 𝕜
    h : HasSubset.Subset (WeakDual.characterSpace 𝕜 A) (Set.preimage (⇑WeakDual.to …
    ⊢ IsCompact (WeakDual.characterSpace 𝕜 A)
  -/
  exact (isCompact_closedBall 𝕜 0 _).of_isClosed_subset CharacterSpace.isClosed h
  /-
    🎉 no goals
  -/


