/-- A reducible type synonym for the bundle of continuous (semi)linear maps. For some reason, it
helps with instance search.

Porting note: after the port is done, we may want to remove this definition.
-/
protected abbrev Bundle.ContinuousLinearMap [∀ x, TopologicalSpace (E₁ x)]
    [∀ x, TopologicalSpace (E₂ x)] : B → Type _ := fun x => E₁ x →SL[σ] E₂ x


/-- Assume `eᵢ` and `eᵢ'` are trivializations of the bundles `Eᵢ` over base `B` with fiber `Fᵢ`
(`i ∈ {1,2}`), then `Pretrivialization.continuousLinearMapCoordChange σ e₁ e₁' e₂ e₂'` is the
coordinate change function between the two induced (pre)trivializations
`Pretrivialization.continuousLinearMap σ e₁ e₂` and
`Pretrivialization.continuousLinearMap σ e₁' e₂'` of `Bundle.ContinuousLinearMap`. -/
def continuousLinearMapCoordChange [e₁.IsLinear 𝕜₁] [e₁'.IsLinear 𝕜₁] [e₂.IsLinear 𝕜₂]
    [e₂'.IsLinear 𝕜₂] (b : B) : (F₁ →SL[σ] F₂) →L[𝕜₂] F₁ →SL[σ] F₂ :=
  ((e₁'.coordChangeL 𝕜₁ e₁ b).symm.arrowCongrSL (e₂.coordChangeL 𝕜₂ e₂' b) :
    (F₁ →SL[σ] F₂) ≃L[𝕜₂] F₁ →SL[σ] F₂)


theorem continuousOn_continuousLinearMapCoordChange [RingHomIsometric σ]
    [VectorBundle 𝕜₁ F₁ E₁] [VectorBundle 𝕜₂ F₂ E₂]
    [MemTrivializationAtlas e₁] [MemTrivializationAtlas e₁'] [MemTrivializationAtlas e₂]
    [MemTrivializationAtlas e₂'] :
    ContinuousOn (continuousLinearMapCoordChange σ e₁ e₁' e₂ e₂')
      (e₁.baseSet ∩ e₂.baseSet ∩ (e₁'.baseSet ∩ e₂'.baseSet)) := by
  /-
    𝕜₁ : Type u_1
    inst✝²³ : NontriviallyNormedField 𝕜₁
    𝕜₂ : Type u_2
    inst✝²² : NontriviallyNormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    B : Type u_3
    F₁ : Type u_4
    inst✝²¹ : NormedAddCommGroup F₁
    inst✝²⁰ : NormedSpace 𝕜₁ F₁
    E₁ : B → Type u_5
    inst✝¹⁹ : (x : B) → AddCommGroup (E₁ x)
    inst✝¹⁸ : (x : B) → Module 𝕜₁ (E₁ x)
    inst✝¹⁷ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_6
    inst✝¹⁶ : NormedAddCommGroup F₂
    inst✝¹⁵ : NormedSpace 𝕜₂ F₂
    E₂ : B → Type u_7
    inst✝¹⁴ : (x : B) → AddCommGroup (E₂ x)
    inst✝¹³ : (x : B) → Module 𝕜₂ (E₂ x)
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹¹ : TopologicalSpace B
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹⁰ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁹ : FiberBundle F₁ E₁
    inst✝⁸ : (x : B) → TopologicalSpace (E₂ x)
    inst✝⁷ : FiberBundle F₂ E₂
    inst✝⁶ : RingHomIsometric σ
    inst✝⁵ : VectorBundle 𝕜₁ F₁ E₁
    inst✝⁴ : VectorBundle 𝕜₂ F₂ E₂
    inst✝³ : MemTrivializationAtlas e₁
    inst✝² : MemTrivializationAtlas e₁'
    inst✝¹ : MemTrivializationAtlas e₂
    inst✝ : MemTrivializationAtlas e₂'
    ⊢ ContinuousOn (Pretrivialization.continuousLinearMapCoordChange σ e₁ e₁' e₂ e …
  -/
  have h₁ := (compSL F₁ F₂ F₂ σ (RingHom.id 𝕜₂)).continuous
  /-
    𝕜₁ : Type u_1
    inst✝²³ : NontriviallyNormedField 𝕜₁
    𝕜₂ : Type u_2
    inst✝²² : NontriviallyNormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    B : Type u_3
    F₁ : Type u_4
    inst✝²¹ : NormedAddCommGroup F₁
    inst✝²⁰ : NormedSpace 𝕜₁ F₁
    E₁ : B → Type u_5
    inst✝¹⁹ : (x : B) → AddCommGroup (E₁ x)
    inst✝¹⁸ : (x : B) → Module 𝕜₁ (E₁ x)
    inst✝¹⁷ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_6
    inst✝¹⁶ : NormedAddCommGroup F₂
    inst✝¹⁵ : NormedSpace 𝕜₂ F₂
    E₂ : B → Type u_7
    inst✝¹⁴ : (x : B) → AddCommGroup (E₂ x)
    inst✝¹³ : (x : B) → Module 𝕜₂ (E₂ x)
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹¹ : TopologicalSpace B
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹⁰ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁹ : FiberBundle F₁ E₁
    inst✝⁸ : (x : B) → TopologicalSpace (E₂ x)
    inst✝⁷ : FiberBundle F₂ E₂
    inst✝⁶ : RingHomIsometric σ
    inst✝⁵ : VectorBundle 𝕜₁ F₁ E₁
    inst✝⁴ : VectorBundle 𝕜₂ F₂ E₂
    inst✝³ : MemTrivializationAtlas e₁
    inst✝² : MemTrivializationAtlas e₁'
    inst✝¹ : MemTrivializationAtlas e₂
    inst✝ : MemTrivializationAtlas e₂'
    h₁ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₂ F₂ σ (RingHom.id 𝕜₂))
    ⊢ ContinuousOn (Pretrivialization.continuousLinearMapCoordChange σ e₁ e₁' e₂ e …
  -/
  have h₂ := (ContinuousLinearMap.flip (compSL F₁ F₁ F₂ (RingHom.id 𝕜₁) σ)).continuous
  /-
    𝕜₁ : Type u_1
    inst✝²³ : NontriviallyNormedField 𝕜₁
    𝕜₂ : Type u_2
    inst✝²² : NontriviallyNormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    B : Type u_3
    F₁ : Type u_4
    inst✝²¹ : NormedAddCommGroup F₁
    inst✝²⁰ : NormedSpace 𝕜₁ F₁
    E₁ : B → Type u_5
    inst✝¹⁹ : (x : B) → AddCommGroup (E₁ x)
    inst✝¹⁸ : (x : B) → Module 𝕜₁ (E₁ x)
    inst✝¹⁷ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_6
    inst✝¹⁶ : NormedAddCommGroup F₂
    inst✝¹⁵ : NormedSpace 𝕜₂ F₂
    E₂ : B → Type u_7
    inst✝¹⁴ : (x : B) → AddCommGroup (E₂ x)
    inst✝¹³ : (x : B) → Module 𝕜₂ (E₂ x)
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹¹ : TopologicalSpace B
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹⁰ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁹ : FiberBundle F₁ E₁
    inst✝⁸ : (x : B) → TopologicalSpace (E₂ x)
    inst✝⁷ : FiberBundle F₂ E₂
    inst✝⁶ : RingHomIsometric σ
    inst✝⁵ : VectorBundle 𝕜₁ F₁ E₁
    inst✝⁴ : VectorBundle 𝕜₂ F₂ E₂
    inst✝³ : MemTrivializationAtlas e₁
    inst✝² : MemTrivializationAtlas e₁'
    inst✝¹ : MemTrivializationAtlas e₂
    inst✝ : MemTrivializationAtlas e₂'
    h₁ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₂ F₂ σ (RingHom.id 𝕜₂))
    h₂ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₁ F₂ (RingHom.id 𝕜₁) σ).flip
    ⊢ ContinuousOn (Pretrivialization.continuousLinearMapCoordChange σ e₁ e₁' e₂ e …
  -/
  have h₃ := continuousOn_coordChange 𝕜₁ e₁' e₁
  /-
    𝕜₁ : Type u_1
    inst✝²³ : NontriviallyNormedField 𝕜₁
    𝕜₂ : Type u_2
    inst✝²² : NontriviallyNormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    B : Type u_3
    F₁ : Type u_4
    inst✝²¹ : NormedAddCommGroup F₁
    inst✝²⁰ : NormedSpace 𝕜₁ F₁
    E₁ : B → Type u_5
    inst✝¹⁹ : (x : B) → AddCommGroup (E₁ x)
    inst✝¹⁸ : (x : B) → Module 𝕜₁ (E₁ x)
    inst✝¹⁷ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_6
    inst✝¹⁶ : NormedAddCommGroup F₂
    inst✝¹⁵ : NormedSpace 𝕜₂ F₂
    E₂ : B → Type u_7
    inst✝¹⁴ : (x : B) → AddCommGroup (E₂ x)
    inst✝¹³ : (x : B) → Module 𝕜₂ (E₂ x)
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹¹ : TopologicalSpace B
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹⁰ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁹ : FiberBundle F₁ E₁
    inst✝⁸ : (x : B) → TopologicalSpace (E₂ x)
    inst✝⁷ : FiberBundle F₂ E₂
    inst✝⁶ : RingHomIsometric σ
    inst✝⁵ : VectorBundle 𝕜₁ F₁ E₁
    inst✝⁴ : VectorBundle 𝕜₂ F₂ E₂
    inst✝³ : MemTrivializationAtlas e₁
    inst✝² : MemTrivializationAtlas e₁'
    inst✝¹ : MemTrivializationAtlas e₂
    inst✝ : MemTrivializationAtlas e₂'
    h₁ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₂ F₂ σ (RingHom.id 𝕜₂))
    h₂ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₁ F₂ (RingHom.id 𝕜₁) σ).flip
    h₃ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₁ e₁' e₁ b)) (Inter …
    ⊢ ContinuousOn (Pretrivialization.continuousLinearMapCoordChange σ e₁ e₁' e₂ e …
  -/
  have h₄ := continuousOn_coordChange 𝕜₂ e₂ e₂'
  /-
    𝕜₁ : Type u_1
    inst✝²³ : NontriviallyNormedField 𝕜₁
    𝕜₂ : Type u_2
    inst✝²² : NontriviallyNormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    B : Type u_3
    F₁ : Type u_4
    inst✝²¹ : NormedAddCommGroup F₁
    inst✝²⁰ : NormedSpace 𝕜₁ F₁
    E₁ : B → Type u_5
    inst✝¹⁹ : (x : B) → AddCommGroup (E₁ x)
    inst✝¹⁸ : (x : B) → Module 𝕜₁ (E₁ x)
    inst✝¹⁷ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_6
    inst✝¹⁶ : NormedAddCommGroup F₂
    inst✝¹⁵ : NormedSpace 𝕜₂ F₂
    E₂ : B → Type u_7
    inst✝¹⁴ : (x : B) → AddCommGroup (E₂ x)
    inst✝¹³ : (x : B) → Module 𝕜₂ (E₂ x)
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹¹ : TopologicalSpace B
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹⁰ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁹ : FiberBundle F₁ E₁
    inst✝⁸ : (x : B) → TopologicalSpace (E₂ x)
    inst✝⁷ : FiberBundle F₂ E₂
    inst✝⁶ : RingHomIsometric σ
    inst✝⁵ : VectorBundle 𝕜₁ F₁ E₁
    inst✝⁴ : VectorBundle 𝕜₂ F₂ E₂
    inst✝³ : MemTrivializationAtlas e₁
    inst✝² : MemTrivializationAtlas e₁'
    inst✝¹ : MemTrivializationAtlas e₂
    inst✝ : MemTrivializationAtlas e₂'
    h₁ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₂ F₂ σ (RingHom.id 𝕜₂))
    h₂ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₁ F₂ (RingHom.id 𝕜₁) σ).flip
    h₃ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₁ e₁' e₁ b)) (Inter …
    h₄ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₂ e₂ e₂' b)) (Inter …
    ⊢ ContinuousOn (Pretrivialization.continuousLinearMapCoordChange σ e₁ e₁' e₂ e …
  -/
  refine ((h₁.comp_continuousOn (h₄.mono ?_)).clm_comp (h₂.comp_continuousOn (h₃.mono ?_))).congr ?_
    /-
      case refine_1
      𝕜₁ : Type u_1
      inst✝²³ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²² : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝²¹ : NormedAddCommGroup F₁
      inst✝²⁰ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁹ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁸ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁷ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁶ : NormedAddCommGroup F₂
      inst✝¹⁵ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹⁴ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹³ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝¹¹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝¹⁰ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁹ : FiberBundle F₁ E₁
      inst✝⁸ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁷ : FiberBundle F₂ E₂
      inst✝⁶ : RingHomIsometric σ
      inst✝⁵ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁴ : VectorBundle 𝕜₂ F₂ E₂
      inst✝³ : MemTrivializationAtlas e₁
      inst✝² : MemTrivializationAtlas e₁'
      inst✝¹ : MemTrivializationAtlas e₂
      inst✝ : MemTrivializationAtlas e₂'
      h₁ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₂ F₂ σ (RingHom.id 𝕜₂))
      h₂ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₁ F₂ (RingHom.id 𝕜₁) σ).flip
      h₃ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₁ e₁' e₁ b)) (Inter …
      h₄ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₂ e₂ e₂' b)) (Inter …
      ⊢ HasSubset.Subset (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.int …
    -/
  · mfld_set_tac
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜₁ : Type u_1
      inst✝²³ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²² : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝²¹ : NormedAddCommGroup F₁
      inst✝²⁰ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁹ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁸ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁷ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁶ : NormedAddCommGroup F₂
      inst✝¹⁵ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹⁴ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹³ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝¹¹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝¹⁰ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁹ : FiberBundle F₁ E₁
      inst✝⁸ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁷ : FiberBundle F₂ E₂
      inst✝⁶ : RingHomIsometric σ
      inst✝⁵ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁴ : VectorBundle 𝕜₂ F₂ E₂
      inst✝³ : MemTrivializationAtlas e₁
      inst✝² : MemTrivializationAtlas e₁'
      inst✝¹ : MemTrivializationAtlas e₂
      inst✝ : MemTrivializationAtlas e₂'
      h₁ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₂ F₂ σ (RingHom.id 𝕜₂))
      h₂ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₁ F₂ (RingHom.id 𝕜₁) σ).flip
      h₃ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₁ e₁' e₁ b)) (Inter …
      h₄ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₂ e₂ e₂' b)) (Inter …
      ⊢ HasSubset.Subset (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.int …
    -/
  · mfld_set_tac
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      𝕜₁ : Type u_1
      inst✝²³ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²² : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝²¹ : NormedAddCommGroup F₁
      inst✝²⁰ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁹ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁸ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁷ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁶ : NormedAddCommGroup F₂
      inst✝¹⁵ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹⁴ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹³ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝¹¹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝¹⁰ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁹ : FiberBundle F₁ E₁
      inst✝⁸ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁷ : FiberBundle F₂ E₂
      inst✝⁶ : RingHomIsometric σ
      inst✝⁵ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁴ : VectorBundle 𝕜₂ F₂ E₂
      inst✝³ : MemTrivializationAtlas e₁
      inst✝² : MemTrivializationAtlas e₁'
      inst✝¹ : MemTrivializationAtlas e₂
      inst✝ : MemTrivializationAtlas e₂'
      h₁ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₂ F₂ σ (RingHom.id 𝕜₂))
      h₂ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₁ F₂ (RingHom.id 𝕜₁) σ).flip
      h₃ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₁ e₁' e₁ b)) (Inter …
      h₄ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₂ e₂ e₂' b)) (Inter …
      ⊢ Set.EqOn (Pretrivialization.continuousLinearMapCoordChange σ e₁ e₁' e₂ e₂')  …
    -/
  · intro b _
    /-
      case refine_3
      𝕜₁ : Type u_1
      inst✝²³ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²² : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝²¹ : NormedAddCommGroup F₁
      inst✝²⁰ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁹ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁸ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁷ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁶ : NormedAddCommGroup F₂
      inst✝¹⁵ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹⁴ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹³ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝¹¹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝¹⁰ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁹ : FiberBundle F₁ E₁
      inst✝⁸ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁷ : FiberBundle F₂ E₂
      inst✝⁶ : RingHomIsometric σ
      inst✝⁵ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁴ : VectorBundle 𝕜₂ F₂ E₂
      inst✝³ : MemTrivializationAtlas e₁
      inst✝² : MemTrivializationAtlas e₁'
      inst✝¹ : MemTrivializationAtlas e₂
      inst✝ : MemTrivializationAtlas e₂'
      h₁ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₂ F₂ σ (RingHom.id 𝕜₂))
      h₂ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₁ F₂ (RingHom.id 𝕜₁) σ).flip
      h₃ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₁ e₁' e₁ b)) (Inter …
      h₄ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₂ e₂ e₂' b)) (Inter …
      b : B
      a✝ : Membership.mem (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.in …
      ⊢ Eq (Pretrivialization.continuousLinearMapCoordChange σ e₁ e₁' e₂ e₂' b) ((fu …
    -/
    ext L v
    /-
      case refine_3.h.h
      𝕜₁ : Type u_1
      inst✝²³ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²² : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝²¹ : NormedAddCommGroup F₁
      inst✝²⁰ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁹ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁸ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁷ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁶ : NormedAddCommGroup F₂
      inst✝¹⁵ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹⁴ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹³ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝¹¹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝¹⁰ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁹ : FiberBundle F₁ E₁
      inst✝⁸ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁷ : FiberBundle F₂ E₂
      inst✝⁶ : RingHomIsometric σ
      inst✝⁵ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁴ : VectorBundle 𝕜₂ F₂ E₂
      inst✝³ : MemTrivializationAtlas e₁
      inst✝² : MemTrivializationAtlas e₁'
      inst✝¹ : MemTrivializationAtlas e₂
      inst✝ : MemTrivializationAtlas e₂'
      h₁ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₂ F₂ σ (RingHom.id 𝕜₂))
      h₂ : Continuous ⇑(ContinuousLinearMap.compSL F₁ F₁ F₂ (RingHom.id 𝕜₁) σ).flip
      h₃ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₁ e₁' e₁ b)) (Inter …
      h₄ : ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜₂ e₂ e₂' b)) (Inter …
      b : B
      a✝ : Membership.mem (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.in …
      L : ContinuousLinearMap σ F₁ F₂
      v : F₁
      ⊢ Eq (((Pretrivialization.continuousLinearMapCoordChange σ e₁ e₁' e₂ e₂' b) L) …
    -/
    dsimp [continuousLinearMapCoordChange]
    /-
      🎉 no goals
    -/


/-- Given trivializations `e₁`, `e₂` for vector bundles `E₁`, `E₂` over a base `B`,
`Pretrivialization.continuousLinearMap σ e₁ e₂` is the induced pretrivialization for the
continuous `σ`-semilinear maps from `E₁` to `E₂`. That is, the map which will later become a
trivialization, after the bundle of continuous semilinear maps is equipped with the right
topological vector bundle structure. -/
def continuousLinearMap :
    Pretrivialization (F₁ →SL[σ] F₂) (π (F₁ →SL[σ] F₂) (Bundle.ContinuousLinearMap σ E₁ E₂)) where
  toFun p := ⟨p.1, .comp (e₂.continuousLinearMapAt 𝕜₂ p.1) (p.2.comp (e₁.symmL 𝕜₁ p.1))⟩
  invFun p := ⟨p.1, .comp (e₂.symmL 𝕜₂ p.1) (p.2.comp (e₁.continuousLinearMapAt 𝕜₁ p.1))⟩
  source := Bundle.TotalSpace.proj ⁻¹' (e₁.baseSet ∩ e₂.baseSet)
  target := (e₁.baseSet ∩ e₂.baseSet) ×ˢ Set.univ
  map_source' := fun ⟨_, _⟩ h => ⟨h, Set.mem_univ _⟩
  map_target' := fun ⟨_, _⟩ h => h.1
  left_inv' := fun ⟨x, L⟩ ⟨h₁, h₂⟩ => by
    /-
      𝕜₁ : Type u_1
      inst✝²⁰ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁹ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁸ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
      inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
      x✝¹ : Bundle.TotalSpace (ContinuousLinearMap σ F₁ F₂) (Bundle.ContinuousLinear …
      x : B
      L : Bundle.ContinuousLinearMap σ E₁ E₂ x
      x✝ : Membership.mem (Set.preimage Bundle.TotalSpace.proj (Inter.inter e₁.baseS …
      h₁ : Membership.mem e₁.baseSet { proj := x, snd := L }.proj
      h₂ : Membership.mem e₂.baseSet { proj := x, snd := L }.proj
      ⊢ Eq ((fun p => { proj := p.1, snd := (Trivialization.symmL 𝕜₂ e₂ p.1).comp (p …
    -/
    simp only [TotalSpace.mk_inj]
    /-
      𝕜₁ : Type u_1
      inst✝²⁰ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁹ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁸ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
      inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
      x✝¹ : Bundle.TotalSpace (ContinuousLinearMap σ F₁ F₂) (Bundle.ContinuousLinear …
      x : B
      L : Bundle.ContinuousLinearMap σ E₁ E₂ x
      x✝ : Membership.mem (Set.preimage Bundle.TotalSpace.proj (Inter.inter e₁.baseS …
      h₁ : Membership.mem e₁.baseSet { proj := x, snd := L }.proj
      h₂ : Membership.mem e₂.baseSet { proj := x, snd := L }.proj
      ⊢ Eq ((Trivialization.symmL 𝕜₂ e₂ x).comp (((Trivialization.continuousLinearMa …
    -/
    ext (v : E₁ x)
    /-
      case h
      𝕜₁ : Type u_1
      inst✝²⁰ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁹ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁸ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
      inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
      x✝¹ : Bundle.TotalSpace (ContinuousLinearMap σ F₁ F₂) (Bundle.ContinuousLinear …
      x : B
      L : Bundle.ContinuousLinearMap σ E₁ E₂ x
      x✝ : Membership.mem (Set.preimage Bundle.TotalSpace.proj (Inter.inter e₁.baseS …
      h₁ : Membership.mem e₁.baseSet { proj := x, snd := L }.proj
      h₂ : Membership.mem e₂.baseSet { proj := x, snd := L }.proj
      v : E₁ x
      ⊢ Eq (((Trivialization.symmL 𝕜₂ e₂ x).comp (((Trivialization.continuousLinearM …
    -/
    dsimp only [comp_apply]
    /-
      case h
      𝕜₁ : Type u_1
      inst✝²⁰ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁹ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁸ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
      inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
      x✝¹ : Bundle.TotalSpace (ContinuousLinearMap σ F₁ F₂) (Bundle.ContinuousLinear …
      x : B
      L : Bundle.ContinuousLinearMap σ E₁ E₂ x
      x✝ : Membership.mem (Set.preimage Bundle.TotalSpace.proj (Inter.inter e₁.baseS …
      h₁ : Membership.mem e₁.baseSet { proj := x, snd := L }.proj
      h₂ : Membership.mem e₂.baseSet { proj := x, snd := L }.proj
      v : E₁ x
      ⊢ Eq ((Trivialization.symmL 𝕜₂ e₂ x) ((Trivialization.continuousLinearMapAt 𝕜₂ …
    -/
    rw [Trivialization.symmL_continuousLinearMapAt, Trivialization.symmL_continuousLinearMapAt]
    /-
      case h.hb
      𝕜₁ : Type u_1
      inst✝²⁰ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁹ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁸ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
      inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
      x✝¹ : Bundle.TotalSpace (ContinuousLinearMap σ F₁ F₂) (Bundle.ContinuousLinear …
      x : B
      L : Bundle.ContinuousLinearMap σ E₁ E₂ x
      x✝ : Membership.mem (Set.preimage Bundle.TotalSpace.proj (Inter.inter e₁.baseS …
      h₁ : Membership.mem e₁.baseSet { proj := x, snd := L }.proj
      h₂ : Membership.mem e₂.baseSet { proj := x, snd := L }.proj
      v : E₁ x
      ⊢ Membership.mem e₁.baseSet x
    -/
    exacts [h₁, h₂]
    /-
      🎉 no goals
    -/
  right_inv' := fun ⟨x, f⟩ ⟨⟨h₁, h₂⟩, _⟩ => by
    /-
      𝕜₁ : Type u_1
      inst✝²⁰ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁹ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁸ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
      inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
      x✝¹ : Prod B (ContinuousLinearMap σ F₁ F₂)
      x : B
      f : ContinuousLinearMap σ F₁ F₂
      x✝ : Membership.mem (SProd.sprod (Inter.inter e₁.baseSet e₂.baseSet) Set.univ) …
      h₁ : Membership.mem e₁.baseSet { fst := x, snd := f }.1
      h₂ : Membership.mem e₂.baseSet { fst := x, snd := f }.1
      right✝ : Membership.mem Set.univ { fst := x, snd := f }.2
      ⊢ Eq ((fun p => { fst := p.proj, snd := (Trivialization.continuousLinearMapAt  …
    -/
    simp only [Prod.mk_inj_left]
    /-
      𝕜₁ : Type u_1
      inst✝²⁰ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁹ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁸ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
      inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
      x✝¹ : Prod B (ContinuousLinearMap σ F₁ F₂)
      x : B
      f : ContinuousLinearMap σ F₁ F₂
      x✝ : Membership.mem (SProd.sprod (Inter.inter e₁.baseSet e₂.baseSet) Set.univ) …
      h₁ : Membership.mem e₁.baseSet { fst := x, snd := f }.1
      h₂ : Membership.mem e₂.baseSet { fst := x, snd := f }.1
      right✝ : Membership.mem Set.univ { fst := x, snd := f }.2
      ⊢ Eq ((Trivialization.continuousLinearMapAt 𝕜₂ e₂ x).comp (((Trivialization.sy …
    -/
    ext v
    /-
      case h
      𝕜₁ : Type u_1
      inst✝²⁰ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁹ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁸ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
      inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
      x✝¹ : Prod B (ContinuousLinearMap σ F₁ F₂)
      x : B
      f : ContinuousLinearMap σ F₁ F₂
      x✝ : Membership.mem (SProd.sprod (Inter.inter e₁.baseSet e₂.baseSet) Set.univ) …
      h₁ : Membership.mem e₁.baseSet { fst := x, snd := f }.1
      h₂ : Membership.mem e₂.baseSet { fst := x, snd := f }.1
      right✝ : Membership.mem Set.univ { fst := x, snd := f }.2
      v : F₁
      ⊢ Eq (((Trivialization.continuousLinearMapAt 𝕜₂ e₂ x).comp (((Trivialization.s …
    -/
    dsimp only [comp_apply]
    /-
      case h
      𝕜₁ : Type u_1
      inst✝²⁰ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁹ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁸ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
      inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
      x✝¹ : Prod B (ContinuousLinearMap σ F₁ F₂)
      x : B
      f : ContinuousLinearMap σ F₁ F₂
      x✝ : Membership.mem (SProd.sprod (Inter.inter e₁.baseSet e₂.baseSet) Set.univ) …
      h₁ : Membership.mem e₁.baseSet { fst := x, snd := f }.1
      h₂ : Membership.mem e₂.baseSet { fst := x, snd := f }.1
      right✝ : Membership.mem Set.univ { fst := x, snd := f }.2
      v : F₁
      ⊢ Eq ((Trivialization.continuousLinearMapAt 𝕜₂ e₂ x) ((Trivialization.symmL 𝕜₂ …
    -/
    rw [Trivialization.continuousLinearMapAt_symmL, Trivialization.continuousLinearMapAt_symmL]
    /-
      case h.hb
      𝕜₁ : Type u_1
      inst✝²⁰ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁹ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
      inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁸ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
      inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
      x✝¹ : Prod B (ContinuousLinearMap σ F₁ F₂)
      x : B
      f : ContinuousLinearMap σ F₁ F₂
      x✝ : Membership.mem (SProd.sprod (Inter.inter e₁.baseSet e₂.baseSet) Set.univ) …
      h₁ : Membership.mem e₁.baseSet { fst := x, snd := f }.1
      h₂ : Membership.mem e₂.baseSet { fst := x, snd := f }.1
      right✝ : Membership.mem Set.univ { fst := x, snd := f }.2
      v : F₁
      ⊢ Membership.mem e₁.baseSet x
    -/
    exacts [h₁, h₂]
    /-
      🎉 no goals
    -/
  open_target := (e₁.open_baseSet.inter e₂.open_baseSet).prod isOpen_univ
  baseSet := e₁.baseSet ∩ e₂.baseSet
  open_baseSet := e₁.open_baseSet.inter e₂.open_baseSet
  source_eq := rfl
  target_eq := rfl
  proj_toFun _ _ := rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: see if Lean 4 can generate this instance without a hint

instance continuousLinearMap.isLinear [∀ x, ContinuousAdd (E₂ x)] [∀ x, ContinuousSMul 𝕜₂ (E₂ x)] :
    (Pretrivialization.continuousLinearMap σ e₁ e₂).IsLinear 𝕜₂ where
  linear x _ :=
    { map_add := fun L L' =>
        show (e₂.continuousLinearMapAt 𝕜₂ x).comp ((L + L').comp (e₁.symmL 𝕜₁ x)) = _ by
          /-
            𝕜₁ : Type u_1
            inst✝²² : NontriviallyNormedField 𝕜₁
            𝕜₂ : Type u_2
            inst✝²¹ : NontriviallyNormedField 𝕜₂
            σ : RingHom 𝕜₁ 𝕜₂
            B : Type u_3
            F₁ : Type u_4
            inst✝²⁰ : NormedAddCommGroup F₁
            inst✝¹⁹ : NormedSpace 𝕜₁ F₁
            E₁ : B → Type u_5
            inst✝¹⁸ : (x : B) → AddCommGroup (E₁ x)
            inst✝¹⁷ : (x : B) → Module 𝕜₁ (E₁ x)
            inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
            F₂ : Type u_6
            inst✝¹⁵ : NormedAddCommGroup F₂
            inst✝¹⁴ : NormedSpace 𝕜₂ F₂
            E₂ : B → Type u_7
            inst✝¹³ : (x : B) → AddCommGroup (E₂ x)
            inst✝¹² : (x : B) → Module 𝕜₂ (E₂ x)
            inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
            inst✝¹⁰ : TopologicalSpace B
            e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
            e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
            inst✝⁹ : (x : B) → TopologicalSpace (E₁ x)
            inst✝⁸ : FiberBundle F₁ E₁
            inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
            inst✝⁶ : FiberBundle F₂ E₂
            inst✝⁵ : Trivialization.IsLinear 𝕜₁ e₁
            inst✝⁴ : Trivialization.IsLinear 𝕜₁ e₁'
            inst✝³ : Trivialization.IsLinear 𝕜₂ e₂
            inst✝² : Trivialization.IsLinear 𝕜₂ e₂'
            inst✝¹ : ∀ (x : B), ContinuousAdd (E₂ x)
            inst✝ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
            x : B
            x✝ : Membership.mem (Pretrivialization.continuousLinearMap σ e₁ e₂).baseSet x
            L L' : Bundle.ContinuousLinearMap σ E₁ E₂ x
            ⊢ Eq ((Trivialization.continuousLinearMapAt 𝕜₂ e₂ x).comp (ContinuousLinearMap …
          -/
          simp_rw [add_comp, comp_add]
          /-
            𝕜₁ : Type u_1
            inst✝²² : NontriviallyNormedField 𝕜₁
            𝕜₂ : Type u_2
            inst✝²¹ : NontriviallyNormedField 𝕜₂
            σ : RingHom 𝕜₁ 𝕜₂
            B : Type u_3
            F₁ : Type u_4
            inst✝²⁰ : NormedAddCommGroup F₁
            inst✝¹⁹ : NormedSpace 𝕜₁ F₁
            E₁ : B → Type u_5
            inst✝¹⁸ : (x : B) → AddCommGroup (E₁ x)
            inst✝¹⁷ : (x : B) → Module 𝕜₁ (E₁ x)
            inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
            F₂ : Type u_6
            inst✝¹⁵ : NormedAddCommGroup F₂
            inst✝¹⁴ : NormedSpace 𝕜₂ F₂
            E₂ : B → Type u_7
            inst✝¹³ : (x : B) → AddCommGroup (E₂ x)
            inst✝¹² : (x : B) → Module 𝕜₂ (E₂ x)
            inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
            inst✝¹⁰ : TopologicalSpace B
            e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
            e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
            inst✝⁹ : (x : B) → TopologicalSpace (E₁ x)
            inst✝⁸ : FiberBundle F₁ E₁
            inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
            inst✝⁶ : FiberBundle F₂ E₂
            inst✝⁵ : Trivialization.IsLinear 𝕜₁ e₁
            inst✝⁴ : Trivialization.IsLinear 𝕜₁ e₁'
            inst✝³ : Trivialization.IsLinear 𝕜₂ e₂
            inst✝² : Trivialization.IsLinear 𝕜₂ e₂'
            inst✝¹ : ∀ (x : B), ContinuousAdd (E₂ x)
            inst✝ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
            x : B
            x✝ : Membership.mem (Pretrivialization.continuousLinearMap σ e₁ e₂).baseSet x
            L L' : Bundle.ContinuousLinearMap σ E₁ E₂ x
            ⊢ Eq (HAdd.hAdd ((Trivialization.continuousLinearMapAt 𝕜₂ e₂ x).comp (Continuo …
          -/
          rfl
          /-
            🎉 no goals
          -/
      map_smul := fun c L =>
        show (e₂.continuousLinearMapAt 𝕜₂ x).comp ((c • L).comp (e₁.symmL 𝕜₁ x)) = _ by
          /-
            𝕜₁ : Type u_1
            inst✝²² : NontriviallyNormedField 𝕜₁
            𝕜₂ : Type u_2
            inst✝²¹ : NontriviallyNormedField 𝕜₂
            σ : RingHom 𝕜₁ 𝕜₂
            B : Type u_3
            F₁ : Type u_4
            inst✝²⁰ : NormedAddCommGroup F₁
            inst✝¹⁹ : NormedSpace 𝕜₁ F₁
            E₁ : B → Type u_5
            inst✝¹⁸ : (x : B) → AddCommGroup (E₁ x)
            inst✝¹⁷ : (x : B) → Module 𝕜₁ (E₁ x)
            inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
            F₂ : Type u_6
            inst✝¹⁵ : NormedAddCommGroup F₂
            inst✝¹⁴ : NormedSpace 𝕜₂ F₂
            E₂ : B → Type u_7
            inst✝¹³ : (x : B) → AddCommGroup (E₂ x)
            inst✝¹² : (x : B) → Module 𝕜₂ (E₂ x)
            inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
            inst✝¹⁰ : TopologicalSpace B
            e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
            e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
            inst✝⁹ : (x : B) → TopologicalSpace (E₁ x)
            inst✝⁸ : FiberBundle F₁ E₁
            inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
            inst✝⁶ : FiberBundle F₂ E₂
            inst✝⁵ : Trivialization.IsLinear 𝕜₁ e₁
            inst✝⁴ : Trivialization.IsLinear 𝕜₁ e₁'
            inst✝³ : Trivialization.IsLinear 𝕜₂ e₂
            inst✝² : Trivialization.IsLinear 𝕜₂ e₂'
            inst✝¹ : ∀ (x : B), ContinuousAdd (E₂ x)
            inst✝ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
            x : B
            x✝ : Membership.mem (Pretrivialization.continuousLinearMap σ e₁ e₂).baseSet x
            c : 𝕜₂
            L : Bundle.ContinuousLinearMap σ E₁ E₂ x
            ⊢ Eq ((Trivialization.continuousLinearMapAt 𝕜₂ e₂ x).comp (ContinuousLinearMap …
          -/
          simp_rw [smul_comp, comp_smulₛₗ, RingHom.id_apply]
          /-
            𝕜₁ : Type u_1
            inst✝²² : NontriviallyNormedField 𝕜₁
            𝕜₂ : Type u_2
            inst✝²¹ : NontriviallyNormedField 𝕜₂
            σ : RingHom 𝕜₁ 𝕜₂
            B : Type u_3
            F₁ : Type u_4
            inst✝²⁰ : NormedAddCommGroup F₁
            inst✝¹⁹ : NormedSpace 𝕜₁ F₁
            E₁ : B → Type u_5
            inst✝¹⁸ : (x : B) → AddCommGroup (E₁ x)
            inst✝¹⁷ : (x : B) → Module 𝕜₁ (E₁ x)
            inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
            F₂ : Type u_6
            inst✝¹⁵ : NormedAddCommGroup F₂
            inst✝¹⁴ : NormedSpace 𝕜₂ F₂
            E₂ : B → Type u_7
            inst✝¹³ : (x : B) → AddCommGroup (E₂ x)
            inst✝¹² : (x : B) → Module 𝕜₂ (E₂ x)
            inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
            inst✝¹⁰ : TopologicalSpace B
            e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
            e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
            inst✝⁹ : (x : B) → TopologicalSpace (E₁ x)
            inst✝⁸ : FiberBundle F₁ E₁
            inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
            inst✝⁶ : FiberBundle F₂ E₂
            inst✝⁵ : Trivialization.IsLinear 𝕜₁ e₁
            inst✝⁴ : Trivialization.IsLinear 𝕜₁ e₁'
            inst✝³ : Trivialization.IsLinear 𝕜₂ e₂
            inst✝² : Trivialization.IsLinear 𝕜₂ e₂'
            inst✝¹ : ∀ (x : B), ContinuousAdd (E₂ x)
            inst✝ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
            x : B
            x✝ : Membership.mem (Pretrivialization.continuousLinearMap σ e₁ e₂).baseSet x
            c : 𝕜₂
            L : Bundle.ContinuousLinearMap σ E₁ E₂ x
            ⊢ Eq (HSMul.hSMul c ((Trivialization.continuousLinearMapAt 𝕜₂ e₂ x).comp (Cont …
          -/
          rfl }
          /-
            🎉 no goals
          -/


theorem continuousLinearMap_apply (p : TotalSpace (F₁ →SL[σ] F₂) fun x => E₁ x →SL[σ] E₂ x) :
    (continuousLinearMap σ e₁ e₂) p =
      ⟨p.1, .comp (e₂.continuousLinearMapAt 𝕜₂ p.1) (p.2.comp (e₁.symmL 𝕜₁ p.1))⟩ :=
  rfl


theorem continuousLinearMap_symm_apply (p : B × (F₁ →SL[σ] F₂)) :
    (continuousLinearMap σ e₁ e₂).toPartialEquiv.symm p =
      ⟨p.1, .comp (e₂.symmL 𝕜₂ p.1) (p.2.comp (e₁.continuousLinearMapAt 𝕜₁ p.1))⟩ :=
  rfl


theorem continuousLinearMap_symm_apply' {b : B} (hb : b ∈ e₁.baseSet ∩ e₂.baseSet)
    (L : F₁ →SL[σ] F₂) :
    (continuousLinearMap σ e₁ e₂).symm b L =
      (e₂.symmL 𝕜₂ b).comp (L.comp <| e₁.continuousLinearMapAt 𝕜₁ b) := by
  /-
    𝕜₁ : Type u_1
    inst✝¹⁸ : NontriviallyNormedField 𝕜₁
    𝕜₂ : Type u_2
    inst✝¹⁷ : NontriviallyNormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    B : Type u_3
    F₁ : Type u_4
    inst✝¹⁶ : NormedAddCommGroup F₁
    inst✝¹⁵ : NormedSpace 𝕜₁ F₁
    E₁ : B → Type u_5
    inst✝¹⁴ : (x : B) → AddCommGroup (E₁ x)
    inst✝¹³ : (x : B) → Module 𝕜₁ (E₁ x)
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_6
    inst✝¹¹ : NormedAddCommGroup F₂
    inst✝¹⁰ : NormedSpace 𝕜₂ F₂
    E₂ : B → Type u_7
    inst✝⁹ : (x : B) → AddCommGroup (E₂ x)
    inst✝⁸ : (x : B) → Module 𝕜₂ (E₂ x)
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁶ : TopologicalSpace B
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁴ : FiberBundle F₁ E₁
    inst✝³ : (x : B) → TopologicalSpace (E₂ x)
    inst✝² : FiberBundle F₂ E₂
    inst✝¹ : Trivialization.IsLinear 𝕜₁ e₁
    inst✝ : Trivialization.IsLinear 𝕜₂ e₂
    b : B
    hb : Membership.mem (Inter.inter e₁.baseSet e₂.baseSet) b
    L : ContinuousLinearMap σ F₁ F₂
    ⊢ Eq ((Pretrivialization.continuousLinearMap σ e₁ e₂).symm b L) ((Trivializati …
  -/
  rw [symm_apply]
    /-
      𝕜₁ : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁷ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁶ : NormedAddCommGroup F₁
      inst✝¹⁵ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁴ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹³ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹¹ : NormedAddCommGroup F₂
      inst✝¹⁰ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝⁹ : (x : B) → AddCommGroup (E₂ x)
      inst✝⁸ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁶ : TopologicalSpace B
      e₁ : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁴ : FiberBundle F₁ E₁
      inst✝³ : (x : B) → TopologicalSpace (E₂ x)
      inst✝² : FiberBundle F₂ E₂
      inst✝¹ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂
      b : B
      hb : Membership.mem (Inter.inter e₁.baseSet e₂.baseSet) b
      L : ContinuousLinearMap σ F₁ F₂
      ⊢ Eq (cast ⋯ (↑(Pretrivialization.continuousLinearMap σ e₁ e₂).symm { fst := b …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case hb
      𝕜₁ : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝¹⁷ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁶ : NormedAddCommGroup F₁
      inst✝¹⁵ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁴ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹³ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹¹ : NormedAddCommGroup F₂
      inst✝¹⁰ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝⁹ : (x : B) → AddCommGroup (E₂ x)
      inst✝⁸ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁶ : TopologicalSpace B
      e₁ : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁴ : FiberBundle F₁ E₁
      inst✝³ : (x : B) → TopologicalSpace (E₂ x)
      inst✝² : FiberBundle F₂ E₂
      inst✝¹ : Trivialization.IsLinear 𝕜₁ e₁
      inst✝ : Trivialization.IsLinear 𝕜₂ e₂
      b : B
      hb : Membership.mem (Inter.inter e₁.baseSet e₂.baseSet) b
      L : ContinuousLinearMap σ F₁ F₂
      ⊢ Membership.mem (Pretrivialization.continuousLinearMap σ e₁ e₂).baseSet b
    -/
  · exact hb
    /-
      🎉 no goals
    -/


theorem continuousLinearMapCoordChange_apply (b : B)
    (hb : b ∈ e₁.baseSet ∩ e₂.baseSet ∩ (e₁'.baseSet ∩ e₂'.baseSet)) (L : F₁ →SL[σ] F₂) :
    continuousLinearMapCoordChange σ e₁ e₁' e₂ e₂' b L =
      (continuousLinearMap σ e₁' e₂' ⟨b, (continuousLinearMap σ e₁ e₂).symm b L⟩).2 := by
  /-
    𝕜₁ : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜₁
    𝕜₂ : Type u_2
    inst✝¹⁹ : NontriviallyNormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    B : Type u_3
    F₁ : Type u_4
    inst✝¹⁸ : NormedAddCommGroup F₁
    inst✝¹⁷ : NormedSpace 𝕜₁ F₁
    E₁ : B → Type u_5
    inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
    inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
    inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_6
    inst✝¹³ : NormedAddCommGroup F₂
    inst✝¹² : NormedSpace 𝕜₂ F₂
    E₂ : B → Type u_7
    inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
    inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
    inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁸ : TopologicalSpace B
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁶ : FiberBundle F₁ E₁
    inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
    inst✝⁴ : FiberBundle F₂ E₂
    inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
    inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
    inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
    inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
    b : B
    hb : Membership.mem (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.in …
    L : ContinuousLinearMap σ F₁ F₂
    ⊢ Eq ((Pretrivialization.continuousLinearMapCoordChange σ e₁ e₁' e₂ e₂' b) L)  …
  -/
  ext v
  simp_rw [continuousLinearMapCoordChange, ContinuousLinearEquiv.coe_coe,
    ContinuousLinearEquiv.arrowCongrSL_apply, continuousLinearMap_apply,
    continuousLinearMap_symm_apply' σ e₁ e₂ hb.1, comp_apply, ContinuousLinearEquiv.coe_coe,
    ContinuousLinearEquiv.symm_symm, Trivialization.continuousLinearMapAt_apply,
    Trivialization.symmL_apply]
  rw [e₂.coordChangeL_apply e₂', e₁'.coordChangeL_apply e₁, e₁.coe_linearMapAt_of_mem hb.1.1,
    e₂'.coe_linearMapAt_of_mem hb.2.2]
  /-
    case h.hb
    𝕜₁ : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜₁
    𝕜₂ : Type u_2
    inst✝¹⁹ : NontriviallyNormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    B : Type u_3
    F₁ : Type u_4
    inst✝¹⁸ : NormedAddCommGroup F₁
    inst✝¹⁷ : NormedSpace 𝕜₁ F₁
    E₁ : B → Type u_5
    inst✝¹⁶ : (x : B) → AddCommGroup (E₁ x)
    inst✝¹⁵ : (x : B) → Module 𝕜₁ (E₁ x)
    inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_6
    inst✝¹³ : NormedAddCommGroup F₂
    inst✝¹² : NormedSpace 𝕜₂ F₂
    E₂ : B → Type u_7
    inst✝¹¹ : (x : B) → AddCommGroup (E₂ x)
    inst✝¹⁰ : (x : B) → Module 𝕜₂ (E₂ x)
    inst✝⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁸ : TopologicalSpace B
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝⁷ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁶ : FiberBundle F₁ E₁
    inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
    inst✝⁴ : FiberBundle F₂ E₂
    inst✝³ : Trivialization.IsLinear 𝕜₁ e₁
    inst✝² : Trivialization.IsLinear 𝕜₁ e₁'
    inst✝¹ : Trivialization.IsLinear 𝕜₂ e₂
    inst✝ : Trivialization.IsLinear 𝕜₂ e₂'
    b : B
    hb : Membership.mem (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.in …
    L : ContinuousLinearMap σ F₁ F₂
    v : F₁
    ⊢ Membership.mem (Inter.inter e₁'.baseSet e₁.baseSet) b
  -/
  exacts [⟨hb.2.1, hb.1.1⟩, ⟨hb.1.2, hb.2.2⟩]
  /-
    🎉 no goals
  -/


/-- The continuous `σ`-semilinear maps between two topological vector bundles form a
`VectorPrebundle` (this is an auxiliary construction for the
`VectorBundle` instance, in which the pretrivializations are collated but no topology
on the total space is yet provided). -/
def Bundle.ContinuousLinearMap.vectorPrebundle :
    VectorPrebundle 𝕜₂ (F₁ →SL[σ] F₂) (Bundle.ContinuousLinearMap σ E₁ E₂) where
  pretrivializationAtlas :=
    {e | ∃ (e₁ : Trivialization F₁ (π F₁ E₁)) (e₂ : Trivialization F₂ (π F₂ E₂))
      (_ : MemTrivializationAtlas e₁) (_ : MemTrivializationAtlas e₂),
        e = Pretrivialization.continuousLinearMap σ e₁ e₂}
  pretrivialization_linear' := by
    /-
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      ⊢ ∀ (e : Pretrivialization (ContinuousLinearMap σ F₁ F₂) Bundle.TotalSpace.pro …
    -/
    rintro _ ⟨e₁, he₁, e₂, he₂, rfl⟩
    /-
      case intro.intro.intro.intro
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁✝ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂✝ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      e₁ : Trivialization F₁ Bundle.TotalSpace.proj
      he₁ : Trivialization F₂ Bundle.TotalSpace.proj
      e₂ : MemTrivializationAtlas e₁
      he₂ : MemTrivializationAtlas he₁
      ⊢ Pretrivialization.IsLinear 𝕜₂ (Pretrivialization.continuousLinearMap σ e₁ he₁)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  pretrivializationAt x :=
    Pretrivialization.continuousLinearMap σ (trivializationAt F₁ E₁ x) (trivializationAt F₂ E₂ x)
  mem_base_pretrivializationAt x :=
    ⟨mem_baseSet_trivializationAt F₁ E₁ x, mem_baseSet_trivializationAt F₂ E₂ x⟩
  pretrivialization_mem_atlas x :=
    ⟨trivializationAt F₁ E₁ x, trivializationAt F₂ E₂ x, inferInstance, inferInstance, rfl⟩
  exists_coordChange := by
    /-
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      ⊢ ∀ (e : Pretrivialization (ContinuousLinearMap σ F₁ F₂) Bundle.TotalSpace.pro …
    -/
    rintro _ ⟨e₁, e₂, he₁, he₂, rfl⟩ _ ⟨e₁', e₂', he₁', he₂', rfl⟩
    exact ⟨continuousLinearMapCoordChange σ e₁ e₁' e₂ e₂',
      continuousOn_continuousLinearMapCoordChange,
      continuousLinearMapCoordChange_apply σ e₁ e₁' e₂ e₂'⟩
  totalSpaceMk_isInducing := by
    /-
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      ⊢ ∀ (b : B), Topology.IsInducing (Function.comp (↑((fun x => Pretrivialization …
    -/
    intro b
    let L₁ : E₁ b ≃L[𝕜₁] F₁ :=
      (trivializationAt F₁ E₁ b).continuousLinearEquivAt 𝕜₁ b
        (mem_baseSet_trivializationAt _ _ _)
    let L₂ : E₂ b ≃L[𝕜₂] F₂ :=
      (trivializationAt F₂ E₂ b).continuousLinearEquivAt 𝕜₂ b
        (mem_baseSet_trivializationAt _ _ _)
    /-
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      b : B
      L₁ : ContinuousLinearEquiv (RingHom.id 𝕜₁) (E₁ b) F₁ := Trivialization.continu …
      L₂ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (E₂ b) F₂ := Trivialization.continu …
      ⊢ Topology.IsInducing (Function.comp (↑((fun x => Pretrivialization.continuous …
    -/
    let φ : (E₁ b →SL[σ] E₂ b) ≃L[𝕜₂] F₁ →SL[σ] F₂ := L₁.arrowCongrSL L₂
    /-
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      b : B
      L₁ : ContinuousLinearEquiv (RingHom.id 𝕜₁) (E₁ b) F₁ := Trivialization.continu …
      L₂ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (E₂ b) F₂ := Trivialization.continu …
      φ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (ContinuousLinearMap σ (E₁ b) (E₂ b) …
      ⊢ Topology.IsInducing (Function.comp (↑((fun x => Pretrivialization.continuous …
    -/
    have : IsInducing fun x => (b, φ x) := isInducing_const_prod.mpr φ.toHomeomorph.isInducing
    /-
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      b : B
      L₁ : ContinuousLinearEquiv (RingHom.id 𝕜₁) (E₁ b) F₁ := Trivialization.continu …
      L₂ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (E₂ b) F₂ := Trivialization.continu …
      φ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (ContinuousLinearMap σ (E₁ b) (E₂ b) …
      this : Topology.IsInducing fun x => { fst := b, snd := φ x }
      ⊢ Topology.IsInducing (Function.comp (↑((fun x => Pretrivialization.continuous …
    -/
    convert this
    /-
      case h.e'_5.h.h.e'_4
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      b : B
      L₁ : ContinuousLinearEquiv (RingHom.id 𝕜₁) (E₁ b) F₁ := Trivialization.continu …
      L₂ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (E₂ b) F₂ := Trivialization.continu …
      φ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (ContinuousLinearMap σ (E₁ b) (E₂ b) …
      this : Topology.IsInducing fun x => { fst := b, snd := φ x }
      x✝ : Bundle.ContinuousLinearMap σ E₁ E₂ b
      ⊢ Eq (Function.comp (↑((fun x => Pretrivialization.continuousLinearMap σ (Fibe …
    -/
    ext f
    /-
      case h.e'_5.h.h.e'_4.h
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      b : B
      L₁ : ContinuousLinearEquiv (RingHom.id 𝕜₁) (E₁ b) F₁ := Trivialization.continu …
      L₂ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (E₂ b) F₂ := Trivialization.continu …
      φ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (ContinuousLinearMap σ (E₁ b) (E₂ b) …
      this : Topology.IsInducing fun x => { fst := b, snd := φ x }
      x✝ : Bundle.ContinuousLinearMap σ E₁ E₂ b
      f : F₁
      ⊢ Eq ((Function.comp (↑((fun x => Pretrivialization.continuousLinearMap σ (Fib …
    -/
    dsimp [Pretrivialization.continuousLinearMap_apply]
    /-
      case h.e'_5.h.h.e'_4.h
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      b : B
      L₁ : ContinuousLinearEquiv (RingHom.id 𝕜₁) (E₁ b) F₁ := Trivialization.continu …
      L₂ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (E₂ b) F₂ := Trivialization.continu …
      φ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (ContinuousLinearMap σ (E₁ b) (E₂ b) …
      this : Topology.IsInducing fun x => { fst := b, snd := φ x }
      x✝ : Bundle.ContinuousLinearMap σ E₁ E₂ b
      f : F₁
      ⊢ Eq ((Trivialization.linearMapAt 𝕜₂ (FiberBundle.trivializationAt F₂ E₂ b) b) …
    -/
    rw [Trivialization.linearMapAt_def_of_mem _ (mem_baseSet_trivializationAt _ _ _)]
    /-
      case h.e'_5.h.h.e'_4.h
      𝕜₁ : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜₁
      𝕜₂ : Type u_2
      inst✝²⁰ : NontriviallyNormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      B : Type u_3
      F₁ : Type u_4
      inst✝¹⁹ : NormedAddCommGroup F₁
      inst✝¹⁸ : NormedSpace 𝕜₁ F₁
      E₁ : B → Type u_5
      inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
      inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
      inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_6
      inst✝¹⁴ : NormedAddCommGroup F₂
      inst✝¹³ : NormedSpace 𝕜₂ F₂
      E₂ : B → Type u_7
      inst✝¹² : (x : B) → AddCommGroup (E₂ x)
      inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : TopologicalSpace B
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
      inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁴ : FiberBundle F₂ E₂
      inst✝³ : VectorBundle 𝕜₂ F₂ E₂
      inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
      inst✝ : RingHomIsometric σ
      b : B
      L₁ : ContinuousLinearEquiv (RingHom.id 𝕜₁) (E₁ b) F₁ := Trivialization.continu …
      L₂ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (E₂ b) F₂ := Trivialization.continu …
      φ : ContinuousLinearEquiv (RingHom.id 𝕜₂) (ContinuousLinearMap σ (E₁ b) (E₂ b) …
      this : Topology.IsInducing fun x => { fst := b, snd := φ x }
      x✝ : Bundle.ContinuousLinearMap σ E₁ E₂ b
      f : F₁
      ⊢ Eq (↑(Trivialization.linearEquivAt 𝕜₂ (FiberBundle.trivializationAt F₂ E₂ b) …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Topology on the total space of the continuous `σ`-semilinear maps between two "normable" vector
bundles over the same base. -/
instance Bundle.ContinuousLinearMap.topologicalSpaceTotalSpace :
    TopologicalSpace (TotalSpace (F₁ →SL[σ] F₂) (Bundle.ContinuousLinearMap σ E₁ E₂)) :=
  (Bundle.ContinuousLinearMap.vectorPrebundle σ F₁ E₁ F₂ E₂).totalSpaceTopology


/-- The continuous `σ`-semilinear maps between two vector bundles form a fiber bundle. -/
instance Bundle.ContinuousLinearMap.fiberBundle :
    FiberBundle (F₁ →SL[σ] F₂) fun x => E₁ x →SL[σ] E₂ x :=
  (Bundle.ContinuousLinearMap.vectorPrebundle σ F₁ E₁ F₂ E₂).toFiberBundle


/-- The continuous `σ`-semilinear maps between two vector bundles form a vector bundle. -/
instance Bundle.ContinuousLinearMap.vectorBundle :
    VectorBundle 𝕜₂ (F₁ →SL[σ] F₂) (Bundle.ContinuousLinearMap σ E₁ E₂) :=
  (Bundle.ContinuousLinearMap.vectorPrebundle σ F₁ E₁ F₂ E₂).toVectorBundle


/-- Given trivializations `e₁`, `e₂` in the atlas for vector bundles `E₁`, `E₂` over a base `B`,
the induced trivialization for the continuous `σ`-semilinear maps from `E₁` to `E₂`,
whose base set is `e₁.baseSet ∩ e₂.baseSet`. -/
def Trivialization.continuousLinearMap :
    Trivialization (F₁ →SL[σ] F₂) (π (F₁ →SL[σ] F₂) (Bundle.ContinuousLinearMap σ E₁ E₂)) :=
  VectorPrebundle.trivializationOfMemPretrivializationAtlas _ ⟨e₁, e₂, he₁, he₂, rfl⟩


instance Bundle.ContinuousLinearMap.memTrivializationAtlas :
    MemTrivializationAtlas
      (e₁.continuousLinearMap σ e₂ :
        Trivialization (F₁ →SL[σ] F₂) (π (F₁ →SL[σ] F₂) (Bundle.ContinuousLinearMap σ E₁ E₂))) where
                         /-
                           𝕜₁ : Type u_1
                           inst✝²¹ : NontriviallyNormedField 𝕜₁
                           𝕜₂ : Type u_2
                           inst✝²⁰ : NontriviallyNormedField 𝕜₂
                           σ : RingHom 𝕜₁ 𝕜₂
                           B : Type u_3
                           F₁ : Type u_4
                           inst✝¹⁹ : NormedAddCommGroup F₁
                           inst✝¹⁸ : NormedSpace 𝕜₁ F₁
                           E₁ : B → Type u_5
                           inst✝¹⁷ : (x : B) → AddCommGroup (E₁ x)
                           inst✝¹⁶ : (x : B) → Module 𝕜₁ (E₁ x)
                           inst✝¹⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
                           F₂ : Type u_6
                           inst✝¹⁴ : NormedAddCommGroup F₂
                           inst✝¹³ : NormedSpace 𝕜₂ F₂
                           E₂ : B → Type u_7
                           inst✝¹² : (x : B) → AddCommGroup (E₂ x)
                           inst✝¹¹ : (x : B) → Module 𝕜₂ (E₂ x)
                           inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
                           inst✝⁹ : TopologicalSpace B
                           e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
                           e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
                           inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
                           inst✝⁷ : FiberBundle F₁ E₁
                           inst✝⁶ : VectorBundle 𝕜₁ F₁ E₁
                           inst✝⁵ : (x : B) → TopologicalSpace (E₂ x)
                           inst✝⁴ : FiberBundle F₂ E₂
                           inst✝³ : VectorBundle 𝕜₂ F₂ E₂
                           inst✝² : ∀ (x : B), TopologicalAddGroup (E₂ x)
                           inst✝¹ : ∀ (x : B), ContinuousSMul 𝕜₂ (E₂ x)
                           inst✝ : RingHomIsometric σ
                           he₁ : MemTrivializationAtlas e₁
                           he₂ : MemTrivializationAtlas e₂
                           ⊢ MemTrivializationAtlas e₁
                         -/
                         /-
                           🎉 no goals
                         -/
  out := ⟨_, ⟨e₁, e₂, by infer_instance, by infer_instance, rfl⟩, rfl⟩
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem Trivialization.baseSet_continuousLinearMap :
    (e₁.continuousLinearMap σ e₂).baseSet = e₁.baseSet ∩ e₂.baseSet :=
  rfl


theorem Trivialization.continuousLinearMap_apply
    (p : TotalSpace (F₁ →SL[σ] F₂) (Bundle.ContinuousLinearMap σ E₁ E₂)) :
    e₁.continuousLinearMap σ e₂ p =
      ⟨p.1, (e₂.continuousLinearMapAt 𝕜₂ p.1 : _ →L[𝕜₂] _).comp
        (p.2.comp (e₁.symmL 𝕜₁ p.1 : F₁ →L[𝕜₁] E₁ p.1) : F₁ →SL[σ] E₂ p.1)⟩ :=
  rfl


theorem hom_trivializationAt_apply (x₀ : B)
    (x : TotalSpace (F₁ →SL[σ] F₂) (Bundle.ContinuousLinearMap σ E₁ E₂)) :
    trivializationAt (F₁ →SL[σ] F₂) (fun x => E₁ x →SL[σ] E₂ x) x₀ x =
      ⟨x.1, inCoordinates F₁ E₁ F₂ E₂ x₀ x.1 x₀ x.1 x.2⟩ :=
  rfl


@[simp, mfld_simps]
theorem hom_trivializationAt_source (x₀ : B) :
    (trivializationAt (F₁ →SL[σ] F₂) (Bundle.ContinuousLinearMap σ E₁ E₂) x₀).source =
      π (F₁ →SL[σ] F₂) (Bundle.ContinuousLinearMap σ E₁ E₂) ⁻¹'
        ((trivializationAt F₁ E₁ x₀).baseSet ∩ (trivializationAt F₂ E₂ x₀).baseSet) :=
  rfl


@[simp, mfld_simps]
theorem hom_trivializationAt_target (x₀ : B) :
    (trivializationAt (F₁ →SL[σ] F₂) (fun x => E₁ x →SL[σ] E₂ x) x₀).target =
      ((trivializationAt F₁ E₁ x₀).baseSet ∩ (trivializationAt F₂ E₂ x₀).baseSet) ×ˢ Set.univ :=
  rfl

