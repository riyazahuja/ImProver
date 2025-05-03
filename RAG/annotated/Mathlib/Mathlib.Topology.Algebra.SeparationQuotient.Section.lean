/-- There exists a continuous `K`-linear map from `SeparationQuotient E` to `E`
such that `mk (outCLM x) = x` for all `x`.

Note that continuity of this map comes for free, because `mk` is a topology inducing map.
-/
theorem exists_out_continuousLinearMap :
    ∃ f : SeparationQuotient E →L[K] E, mkCLM K E ∘L f = .id K (SeparationQuotient E) := by
  rcases (mkCLM K E).toLinearMap.exists_rightInverse_of_surjective
    (LinearMap.range_eq_top.mpr surjective_mk) with ⟨f, hf⟩
  /-
    case intro
    K : Type u_1
    E : Type u_2
    inst✝⁵ : DivisionRing K
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module K E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul K E
    f : LinearMap (RingHom.id K) (SeparationQuotient E) E
    hf : Eq ((↑(SeparationQuotient.mkCLM K E)).comp f) LinearMap.id
    ⊢ Exists fun f => Eq ((SeparationQuotient.mkCLM K E).comp f) (ContinuousLinear …
  -/
  replace hf : mk ∘ f = id := congr_arg DFunLike.coe hf
  /-
    case intro
    K : Type u_1
    E : Type u_2
    inst✝⁵ : DivisionRing K
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module K E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul K E
    f : LinearMap (RingHom.id K) (SeparationQuotient E) E
    hf : Eq (Function.comp SeparationQuotient.mk ⇑f) id
    ⊢ Exists fun f => Eq ((SeparationQuotient.mkCLM K E).comp f) (ContinuousLinear …
  -/
  exact ⟨⟨f, isInducing_mk.continuous_iff.2 (by continuity)⟩, DFunLike.ext' hf⟩
  /-
    🎉 no goals
  -/


/-- A continuous `K`-linear map from `SeparationQuotient E` to `E`
such that `mk (outCLM x) = x` for all `x`. -/
noncomputable def outCLM : SeparationQuotient E →L[K] E :=
  (exists_out_continuousLinearMap K E).choose


@[simp]
theorem mkCLM_comp_outCLM : mkCLM K E ∘L outCLM K E = .id K (SeparationQuotient E) :=
  (exists_out_continuousLinearMap K E).choose_spec


variable {E} in
@[simp]
theorem mk_outCLM (x : SeparationQuotient E) : mk (outCLM K E x) = x :=
  DFunLike.congr_fun (mkCLM_comp_outCLM K E) x


@[simp]
theorem mk_comp_outCLM : mk ∘ outCLM K E = id := funext (mk_outCLM K)


variable {K} in
theorem postcomp_mkCLM_surjective {L : Type*} [Semiring L] (σ : L →+* K)
    (F : Type*) [AddCommMonoid F] [Module L F] [TopologicalSpace F] :
    Function.Surjective ((mkCLM K E).comp : (F →SL[σ] E) → (F →SL[σ] SeparationQuotient E)) := by
  /-
    K : Type u_1
    E : Type u_2
    inst✝⁹ : DivisionRing K
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module K E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul K E
    L : Type u_3
    inst✝³ : Semiring L
    σ : RingHom L K
    F : Type u_4
    inst✝² : AddCommMonoid F
    inst✝¹ : Module L F
    inst✝ : TopologicalSpace F
    ⊢ Function.Surjective (SeparationQuotient.mkCLM K E).comp
  -/
  intro f
  /-
    K : Type u_1
    E : Type u_2
    inst✝⁹ : DivisionRing K
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module K E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul K E
    L : Type u_3
    inst✝³ : Semiring L
    σ : RingHom L K
    F : Type u_4
    inst✝² : AddCommMonoid F
    inst✝¹ : Module L F
    inst✝ : TopologicalSpace F
    f : ContinuousLinearMap σ F (SeparationQuotient E)
    ⊢ Exists fun a => Eq ((SeparationQuotient.mkCLM K E).comp a) f
  -/
  use (outCLM K E).comp f
  /-
    case h
    K : Type u_1
    E : Type u_2
    inst✝⁹ : DivisionRing K
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module K E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul K E
    L : Type u_3
    inst✝³ : Semiring L
    σ : RingHom L K
    F : Type u_4
    inst✝² : AddCommMonoid F
    inst✝¹ : Module L F
    inst✝ : TopologicalSpace F
    f : ContinuousLinearMap σ F (SeparationQuotient E)
    ⊢ Eq ((SeparationQuotient.mkCLM K E).comp ((SeparationQuotient.outCLM K E).com …
  -/
  rw [← ContinuousLinearMap.comp_assoc, mkCLM_comp_outCLM, ContinuousLinearMap.id_comp]
  /-
    🎉 no goals
  -/


/-- The `SeparationQuotient.outCLM K E` map is a topological embedding. -/
theorem isEmbedding_outCLM : IsEmbedding (outCLM K E) :=
  Function.LeftInverse.isEmbedding (mk_outCLM K) continuous_mk (map_continuous _)


@[deprecated (since := "2024-10-26")]
alias outCLM_embedding := isEmbedding_outCLM


theorem outCLM_injective : Function.Injective (outCLM K E) :=
  (isEmbedding_outCLM K E).injective


theorem outCLM_isUniformInducing : IsUniformInducing (outCLM K E) := by
  /-
    K : Type u_1
    E : Type u_2
    inst✝⁵ : DivisionRing K
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module K E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousConstSMul K E
    ⊢ IsUniformInducing ⇑(SeparationQuotient.outCLM K E)
  -/
  rw [← isUniformInducing_mk.isUniformInducing_comp_iff, mk_comp_outCLM]
  /-
    K : Type u_1
    E : Type u_2
    inst✝⁵ : DivisionRing K
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module K E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousConstSMul K E
    ⊢ IsUniformInducing id
  -/
  exact .id
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias outCLM_uniformInducing := outCLM_isUniformInducing


theorem outCLM_isUniformEmbedding : IsUniformEmbedding (outCLM K E) where
  injective := outCLM_injective K E
  toIsUniformInducing := outCLM_isUniformInducing K E


@[deprecated (since := "2024-10-01")]
alias outCLM_uniformEmbedding := outCLM_isUniformEmbedding


theorem outCLM_uniformContinuous : UniformContinuous (outCLM K E) :=
  (outCLM_isUniformInducing K E).uniformContinuous


