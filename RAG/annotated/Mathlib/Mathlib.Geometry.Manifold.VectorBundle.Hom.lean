local notation "LE₁E₂" => TotalSpace (F₁ →L[𝕜] F₂) (Bundle.ContinuousLinearMap (RingHom.id 𝕜) E₁ E₂)

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11083): moved slow parts to separate lemmas

theorem contMDiffOn_continuousLinearMapCoordChange
    [SmoothVectorBundle F₁ E₁ IB] [SmoothVectorBundle F₂ E₂ IB] [MemTrivializationAtlas e₁]
    [MemTrivializationAtlas e₁'] [MemTrivializationAtlas e₂] [MemTrivializationAtlas e₂'] :
    ContMDiffOn IB 𝓘(𝕜, (F₁ →L[𝕜] F₂) →L[𝕜] F₁ →L[𝕜] F₂) ⊤
      (continuousLinearMapCoordChange (RingHom.id 𝕜) e₁ e₁' e₂ e₂')
      (e₁.baseSet ∩ e₂.baseSet ∩ (e₁'.baseSet ∩ e₂'.baseSet)) := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F₁ : Type u_3
    F₂ : Type u_4
    E₁ : B → Type u_6
    E₂ : B → Type u_7
    inst✝²⁷ : NontriviallyNormedField 𝕜
    inst✝²⁶ : (x : B) → AddCommGroup (E₁ x)
    inst✝²⁵ : (x : B) → Module 𝕜 (E₁ x)
    inst✝²⁴ : NormedAddCommGroup F₁
    inst✝²³ : NormedSpace 𝕜 F₁
    inst✝²² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²¹ : (x : B) → TopologicalSpace (E₁ x)
    inst✝²⁰ : (x : B) → AddCommGroup (E₂ x)
    inst✝¹⁹ : (x : B) → Module 𝕜 (E₂ x)
    inst✝¹⁸ : NormedAddCommGroup F₂
    inst✝¹⁷ : NormedSpace 𝕜 F₂
    inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁵ : (x : B) → TopologicalSpace (E₂ x)
    EB : Type u_8
    inst✝¹⁴ : NormedAddCommGroup EB
    inst✝¹³ : NormedSpace 𝕜 EB
    HB : Type u_9
    inst✝¹² : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹¹ : TopologicalSpace B
    inst✝¹⁰ : ChartedSpace HB B
    inst✝⁹ : FiberBundle F₁ E₁
    inst✝⁸ : VectorBundle 𝕜 F₁ E₁
    inst✝⁷ : FiberBundle F₂ E₂
    inst✝⁶ : VectorBundle 𝕜 F₂ E₂
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝⁵ : SmoothVectorBundle F₁ E₁ IB
    inst✝⁴ : SmoothVectorBundle F₂ E₂ IB
    inst✝³ : MemTrivializationAtlas e₁
    inst✝² : MemTrivializationAtlas e₁'
    inst✝¹ : MemTrivializationAtlas e₂
    inst✝ : MemTrivializationAtlas e₂'
    ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) ( …
  -/
  have h₁ := contMDiffOn_coordChangeL (IB := IB) e₁' e₁ (n := ⊤)
  /-
    𝕜 : Type u_1
    B : Type u_2
    F₁ : Type u_3
    F₂ : Type u_4
    E₁ : B → Type u_6
    E₂ : B → Type u_7
    inst✝²⁷ : NontriviallyNormedField 𝕜
    inst✝²⁶ : (x : B) → AddCommGroup (E₁ x)
    inst✝²⁵ : (x : B) → Module 𝕜 (E₁ x)
    inst✝²⁴ : NormedAddCommGroup F₁
    inst✝²³ : NormedSpace 𝕜 F₁
    inst✝²² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²¹ : (x : B) → TopologicalSpace (E₁ x)
    inst✝²⁰ : (x : B) → AddCommGroup (E₂ x)
    inst✝¹⁹ : (x : B) → Module 𝕜 (E₂ x)
    inst✝¹⁸ : NormedAddCommGroup F₂
    inst✝¹⁷ : NormedSpace 𝕜 F₂
    inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁵ : (x : B) → TopologicalSpace (E₂ x)
    EB : Type u_8
    inst✝¹⁴ : NormedAddCommGroup EB
    inst✝¹³ : NormedSpace 𝕜 EB
    HB : Type u_9
    inst✝¹² : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹¹ : TopologicalSpace B
    inst✝¹⁰ : ChartedSpace HB B
    inst✝⁹ : FiberBundle F₁ E₁
    inst✝⁸ : VectorBundle 𝕜 F₁ E₁
    inst✝⁷ : FiberBundle F₂ E₂
    inst✝⁶ : VectorBundle 𝕜 F₂ E₂
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝⁵ : SmoothVectorBundle F₁ E₁ IB
    inst✝⁴ : SmoothVectorBundle F₂ E₂ IB
    inst✝³ : MemTrivializationAtlas e₁
    inst✝² : MemTrivializationAtlas e₁'
    inst✝¹ : MemTrivializationAtlas e₂
    inst✝ : MemTrivializationAtlas e₂'
    h₁ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
    ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) ( …
  -/
  have h₂ := contMDiffOn_coordChangeL (IB := IB) e₂ e₂' (n := ⊤)
  /-
    𝕜 : Type u_1
    B : Type u_2
    F₁ : Type u_3
    F₂ : Type u_4
    E₁ : B → Type u_6
    E₂ : B → Type u_7
    inst✝²⁷ : NontriviallyNormedField 𝕜
    inst✝²⁶ : (x : B) → AddCommGroup (E₁ x)
    inst✝²⁵ : (x : B) → Module 𝕜 (E₁ x)
    inst✝²⁴ : NormedAddCommGroup F₁
    inst✝²³ : NormedSpace 𝕜 F₁
    inst✝²² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²¹ : (x : B) → TopologicalSpace (E₁ x)
    inst✝²⁰ : (x : B) → AddCommGroup (E₂ x)
    inst✝¹⁹ : (x : B) → Module 𝕜 (E₂ x)
    inst✝¹⁸ : NormedAddCommGroup F₂
    inst✝¹⁷ : NormedSpace 𝕜 F₂
    inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁵ : (x : B) → TopologicalSpace (E₂ x)
    EB : Type u_8
    inst✝¹⁴ : NormedAddCommGroup EB
    inst✝¹³ : NormedSpace 𝕜 EB
    HB : Type u_9
    inst✝¹² : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹¹ : TopologicalSpace B
    inst✝¹⁰ : ChartedSpace HB B
    inst✝⁹ : FiberBundle F₁ E₁
    inst✝⁸ : VectorBundle 𝕜 F₁ E₁
    inst✝⁷ : FiberBundle F₂ E₂
    inst✝⁶ : VectorBundle 𝕜 F₂ E₂
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝⁵ : SmoothVectorBundle F₁ E₁ IB
    inst✝⁴ : SmoothVectorBundle F₂ E₂ IB
    inst✝³ : MemTrivializationAtlas e₁
    inst✝² : MemTrivializationAtlas e₁'
    inst✝¹ : MemTrivializationAtlas e₂
    inst✝ : MemTrivializationAtlas e₂'
    h₁ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
    h₂ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
    ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) ( …
  -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  refine (h₁.mono ?_).cle_arrowCongr (h₂.mono ?_) <;> mfld_set_tac
                                                      /-
                                                        🎉 no goals
                                                      -/


@[deprecated (since := "2024-11-21")]
alias smoothOn_continuousLinearMapCoordChange := contMDiffOn_continuousLinearMapCoordChange


theorem hom_chart (y₀ y : LE₁E₂) :
    chartAt (ModelProd HB (F₁ →L[𝕜] F₂)) y₀ y =
      (chartAt HB y₀.1 y.1, inCoordinates F₁ E₁ F₂ E₂ y₀.1 y.1 y₀.1 y.1 y.2) := by
  rw [FiberBundle.chartedSpace_chartAt, trans_apply, PartialHomeomorph.prod_apply,
    Trivialization.coe_coe, PartialHomeomorph.refl_apply, Function.id_def,
    hom_trivializationAt_apply]


theorem contMDiffAt_hom_bundle (f : M → LE₁E₂) {x₀ : M} {n : ℕ∞} :
    ContMDiffAt IM (IB.prod 𝓘(𝕜, F₁ →L[𝕜] F₂)) n f x₀ ↔
      ContMDiffAt IM IB n (fun x => (f x).1) x₀ ∧
        ContMDiffAt IM 𝓘(𝕜, F₁ →L[𝕜] F₂) n
          (fun x => inCoordinates F₁ E₁ F₂ E₂ (f x₀).1 (f x).1 (f x₀).1 (f x).1 (f x).2) x₀ :=
  contMDiffAt_totalSpace ..


@[deprecated (since := "2024-11-21")] alias smoothAt_hom_bundle := contMDiffAt_hom_bundle



instance Bundle.ContinuousLinearMap.vectorPrebundle.isSmooth :
    (Bundle.ContinuousLinearMap.vectorPrebundle (RingHom.id 𝕜) F₁ E₁ F₂ E₂).IsSmooth IB where
  exists_smoothCoordChange := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      F₁ : Type u_3
      F₂ : Type u_4
      M : Type u_5
      E₁ : B → Type u_6
      E₂ : B → Type u_7
      inst✝³⁰ : NontriviallyNormedField 𝕜
      inst✝²⁹ : (x : B) → AddCommGroup (E₁ x)
      inst✝²⁸ : (x : B) → Module 𝕜 (E₁ x)
      inst✝²⁷ : NormedAddCommGroup F₁
      inst✝²⁶ : NormedSpace 𝕜 F₁
      inst✝²⁵ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      inst✝²⁴ : (x : B) → TopologicalSpace (E₁ x)
      inst✝²³ : (x : B) → AddCommGroup (E₂ x)
      inst✝²² : (x : B) → Module 𝕜 (E₂ x)
      inst✝²¹ : NormedAddCommGroup F₂
      inst✝²⁰ : NormedSpace 𝕜 F₂
      inst✝¹⁹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝¹⁸ : (x : B) → TopologicalSpace (E₂ x)
      EB : Type u_8
      inst✝¹⁷ : NormedAddCommGroup EB
      inst✝¹⁶ : NormedSpace 𝕜 EB
      HB : Type u_9
      inst✝¹⁵ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁴ : TopologicalSpace B
      inst✝¹³ : ChartedSpace HB B
      EM : Type u_10
      inst✝¹² : NormedAddCommGroup EM
      inst✝¹¹ : NormedSpace 𝕜 EM
      HM : Type u_11
      inst✝¹⁰ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : ChartedSpace HM M
      inst✝⁷ : FiberBundle F₁ E₁
      inst✝⁶ : VectorBundle 𝕜 F₁ E₁
      inst✝⁵ : FiberBundle F₂ E₂
      inst✝⁴ : VectorBundle 𝕜 F₂ E₂
      e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝³ : ∀ (x : B), TopologicalAddGroup (E₂ x)
      inst✝² : ∀ (x : B), ContinuousSMul 𝕜 (E₂ x)
      inst✝¹ : SmoothVectorBundle F₁ E₁ IB
      inst✝ : SmoothVectorBundle F₂ E₂ IB
      ⊢ ∀ (e : Pretrivialization (ContinuousLinearMap (RingHom.id 𝕜) F₁ F₂) Bundle.T …
    -/
    rintro _ ⟨e₁, e₂, he₁, he₂, rfl⟩ _ ⟨e₁', e₂', he₁', he₂', rfl⟩
    exact ⟨continuousLinearMapCoordChange (RingHom.id 𝕜) e₁ e₁' e₂ e₂',
      contMDiffOn_continuousLinearMapCoordChange,
      continuousLinearMapCoordChange_apply (RingHom.id 𝕜) e₁ e₁' e₂ e₂'⟩


instance SmoothVectorBundle.continuousLinearMap :
    SmoothVectorBundle (F₁ →L[𝕜] F₂) (Bundle.ContinuousLinearMap (RingHom.id 𝕜) E₁ E₂) IB :=
  (Bundle.ContinuousLinearMap.vectorPrebundle (RingHom.id 𝕜) F₁ E₁ F₂ E₂).smoothVectorBundle IB


/-- Consider a smooth map `v : M → E₁` to a vector bundle, over a basemap `b₁ : M → B₁`, and
another basemap `b₂ : M → B₂`. Given linear maps `ϕ m : E₁ (b₁ m) → E₂ (b₂ m)` depending smoothly
on `m`, one can apply `ϕ m` to `g m`, and the resulting map is smooth.

Note that the smoothness of `ϕ` can not be always be stated as smoothness of a map into a manifold,
as the pullback bundles `b₁ *ᵖ E₁` and `b₂ *ᵖ E₂` only make sense when `b₁` and `b₂` are globally
smooth, but we want to apply this lemma with only local information. Therefore, we formulate it
using smoothness of `ϕ` read in coordinates.

Version for `ContMDiffWithinAt`. We also give a version for `ContMDiffAt`, but no version for
`ContMDiffOn` or `ContMDiff` as our assumption, written in coordinates, only makes sense around
a point.
 -/
lemma ContMDiffWithinAt.clm_apply_of_inCoordinates
    (hϕ : ContMDiffWithinAt IM 𝓘(𝕜, F₁ →L[𝕜] F₂) n
      (fun m ↦ inCoordinates F₁ E₁ F₂ E₂ (b₁ m₀) (b₁ m) (b₂ m₀) (b₂ m) (ϕ m)) s m₀)
    (hv : ContMDiffWithinAt IM (IB₁.prod 𝓘(𝕜, F₁)) n (fun m ↦ (v m : TotalSpace F₁ E₁)) s m₀)
    (hb₂ : ContMDiffWithinAt IM IB₂ n b₂ s m₀) :
    ContMDiffWithinAt IM (IB₂.prod 𝓘(𝕜, F₂)) n (fun m ↦ (ϕ m (v m) : TotalSpace F₂ E₂)) s m₀ := by
  /-
    𝕜 : Type u_1
    F₁ : Type u_2
    F₂ : Type u_3
    B₁ : Type u_4
    B₂ : Type u_5
    M : Type u_6
    E₁ : B₁ → Type u_7
    E₂ : B₂ → Type u_8
    inst✝³¹ : NontriviallyNormedField 𝕜
    inst✝³⁰ : (x : B₁) → AddCommGroup (E₁ x)
    inst✝²⁹ : (x : B₁) → Module 𝕜 (E₁ x)
    inst✝²⁸ : NormedAddCommGroup F₁
    inst✝²⁷ : NormedSpace 𝕜 F₁
    inst✝²⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²⁵ : (x : B₁) → TopologicalSpace (E₁ x)
    inst✝²⁴ : (x : B₂) → AddCommGroup (E₂ x)
    inst✝²³ : (x : B₂) → Module 𝕜 (E₂ x)
    inst✝²² : NormedAddCommGroup F₂
    inst✝²¹ : NormedSpace 𝕜 F₂
    inst✝²⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁹ : (x : B₂) → TopologicalSpace (E₂ x)
    EB₁ : Type u_9
    inst✝¹⁸ : NormedAddCommGroup EB₁
    inst✝¹⁷ : NormedSpace 𝕜 EB₁
    HB₁ : Type u_10
    inst✝¹⁶ : TopologicalSpace HB₁
    IB₁ : ModelWithCorners 𝕜 EB₁ HB₁
    inst✝¹⁵ : TopologicalSpace B₁
    inst✝¹⁴ : ChartedSpace HB₁ B₁
    EB₂ : Type u_11
    inst✝¹³ : NormedAddCommGroup EB₂
    inst✝¹² : NormedSpace 𝕜 EB₂
    HB₂ : Type u_12
    inst✝¹¹ : TopologicalSpace HB₂
    IB₂ : ModelWithCorners 𝕜 EB₂ HB₂
    inst✝¹⁰ : TopologicalSpace B₂
    inst✝⁹ : ChartedSpace HB₂ B₂
    EM : Type u_13
    inst✝⁸ : NormedAddCommGroup EM
    inst✝⁷ : NormedSpace 𝕜 EM
    HM : Type u_14
    inst✝⁶ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    n : ENat
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    s : Set M
    hϕ : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHo …
    hv : ContMDiffWithinAt IM (IB₁.prod (modelWithCornersSelf 𝕜 F₁)) n (fun m => { …
    hb₂ : ContMDiffWithinAt IM IB₂ n b₂ s m₀
    ⊢ ContMDiffWithinAt IM (IB₂.prod (modelWithCornersSelf 𝕜 F₂)) n (fun m => { pr …
  -/
  rw [← contMDiffWithinAt_insert_self] at hϕ hv hb₂ ⊢
  /-
    𝕜 : Type u_1
    F₁ : Type u_2
    F₂ : Type u_3
    B₁ : Type u_4
    B₂ : Type u_5
    M : Type u_6
    E₁ : B₁ → Type u_7
    E₂ : B₂ → Type u_8
    inst✝³¹ : NontriviallyNormedField 𝕜
    inst✝³⁰ : (x : B₁) → AddCommGroup (E₁ x)
    inst✝²⁹ : (x : B₁) → Module 𝕜 (E₁ x)
    inst✝²⁸ : NormedAddCommGroup F₁
    inst✝²⁷ : NormedSpace 𝕜 F₁
    inst✝²⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²⁵ : (x : B₁) → TopologicalSpace (E₁ x)
    inst✝²⁴ : (x : B₂) → AddCommGroup (E₂ x)
    inst✝²³ : (x : B₂) → Module 𝕜 (E₂ x)
    inst✝²² : NormedAddCommGroup F₂
    inst✝²¹ : NormedSpace 𝕜 F₂
    inst✝²⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁹ : (x : B₂) → TopologicalSpace (E₂ x)
    EB₁ : Type u_9
    inst✝¹⁸ : NormedAddCommGroup EB₁
    inst✝¹⁷ : NormedSpace 𝕜 EB₁
    HB₁ : Type u_10
    inst✝¹⁶ : TopologicalSpace HB₁
    IB₁ : ModelWithCorners 𝕜 EB₁ HB₁
    inst✝¹⁵ : TopologicalSpace B₁
    inst✝¹⁴ : ChartedSpace HB₁ B₁
    EB₂ : Type u_11
    inst✝¹³ : NormedAddCommGroup EB₂
    inst✝¹² : NormedSpace 𝕜 EB₂
    HB₂ : Type u_12
    inst✝¹¹ : TopologicalSpace HB₂
    IB₂ : ModelWithCorners 𝕜 EB₂ HB₂
    inst✝¹⁰ : TopologicalSpace B₂
    inst✝⁹ : ChartedSpace HB₂ B₂
    EM : Type u_13
    inst✝⁸ : NormedAddCommGroup EM
    inst✝⁷ : NormedSpace 𝕜 EM
    HM : Type u_14
    inst✝⁶ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    n : ENat
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    s : Set M
    hϕ : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHo …
    hv : ContMDiffWithinAt IM (IB₁.prod (modelWithCornersSelf 𝕜 F₁)) n (fun m => { …
    hb₂ : ContMDiffWithinAt IM IB₂ n b₂ (Insert.insert m₀ s) m₀
    ⊢ ContMDiffWithinAt IM (IB₂.prod (modelWithCornersSelf 𝕜 F₂)) n (fun m => { pr …
  -/
  rw [contMDiffWithinAt_totalSpace] at hv ⊢
  /-
    𝕜 : Type u_1
    F₁ : Type u_2
    F₂ : Type u_3
    B₁ : Type u_4
    B₂ : Type u_5
    M : Type u_6
    E₁ : B₁ → Type u_7
    E₂ : B₂ → Type u_8
    inst✝³¹ : NontriviallyNormedField 𝕜
    inst✝³⁰ : (x : B₁) → AddCommGroup (E₁ x)
    inst✝²⁹ : (x : B₁) → Module 𝕜 (E₁ x)
    inst✝²⁸ : NormedAddCommGroup F₁
    inst✝²⁷ : NormedSpace 𝕜 F₁
    inst✝²⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²⁵ : (x : B₁) → TopologicalSpace (E₁ x)
    inst✝²⁴ : (x : B₂) → AddCommGroup (E₂ x)
    inst✝²³ : (x : B₂) → Module 𝕜 (E₂ x)
    inst✝²² : NormedAddCommGroup F₂
    inst✝²¹ : NormedSpace 𝕜 F₂
    inst✝²⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁹ : (x : B₂) → TopologicalSpace (E₂ x)
    EB₁ : Type u_9
    inst✝¹⁸ : NormedAddCommGroup EB₁
    inst✝¹⁷ : NormedSpace 𝕜 EB₁
    HB₁ : Type u_10
    inst✝¹⁶ : TopologicalSpace HB₁
    IB₁ : ModelWithCorners 𝕜 EB₁ HB₁
    inst✝¹⁵ : TopologicalSpace B₁
    inst✝¹⁴ : ChartedSpace HB₁ B₁
    EB₂ : Type u_11
    inst✝¹³ : NormedAddCommGroup EB₂
    inst✝¹² : NormedSpace 𝕜 EB₂
    HB₂ : Type u_12
    inst✝¹¹ : TopologicalSpace HB₂
    IB₂ : ModelWithCorners 𝕜 EB₂ HB₂
    inst✝¹⁰ : TopologicalSpace B₂
    inst✝⁹ : ChartedSpace HB₂ B₂
    EM : Type u_13
    inst✝⁸ : NormedAddCommGroup EM
    inst✝⁷ : NormedSpace 𝕜 EM
    HM : Type u_14
    inst✝⁶ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    n : ENat
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    s : Set M
    hϕ : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHo …
    hv : And (ContMDiffWithinAt IM IB₁ n (fun x => { proj := b₁ x, snd := v x }.pr …
    hb₂ : ContMDiffWithinAt IM IB₂ n b₂ (Insert.insert m₀ s) m₀
    ⊢ And (ContMDiffWithinAt IM IB₂ n (fun x => { proj := b₂ x, snd := (ϕ x) (v x) …
  -/
  refine ⟨hb₂, ?_⟩
  /-
    𝕜 : Type u_1
    F₁ : Type u_2
    F₂ : Type u_3
    B₁ : Type u_4
    B₂ : Type u_5
    M : Type u_6
    E₁ : B₁ → Type u_7
    E₂ : B₂ → Type u_8
    inst✝³¹ : NontriviallyNormedField 𝕜
    inst✝³⁰ : (x : B₁) → AddCommGroup (E₁ x)
    inst✝²⁹ : (x : B₁) → Module 𝕜 (E₁ x)
    inst✝²⁸ : NormedAddCommGroup F₁
    inst✝²⁷ : NormedSpace 𝕜 F₁
    inst✝²⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²⁵ : (x : B₁) → TopologicalSpace (E₁ x)
    inst✝²⁴ : (x : B₂) → AddCommGroup (E₂ x)
    inst✝²³ : (x : B₂) → Module 𝕜 (E₂ x)
    inst✝²² : NormedAddCommGroup F₂
    inst✝²¹ : NormedSpace 𝕜 F₂
    inst✝²⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁹ : (x : B₂) → TopologicalSpace (E₂ x)
    EB₁ : Type u_9
    inst✝¹⁸ : NormedAddCommGroup EB₁
    inst✝¹⁷ : NormedSpace 𝕜 EB₁
    HB₁ : Type u_10
    inst✝¹⁶ : TopologicalSpace HB₁
    IB₁ : ModelWithCorners 𝕜 EB₁ HB₁
    inst✝¹⁵ : TopologicalSpace B₁
    inst✝¹⁴ : ChartedSpace HB₁ B₁
    EB₂ : Type u_11
    inst✝¹³ : NormedAddCommGroup EB₂
    inst✝¹² : NormedSpace 𝕜 EB₂
    HB₂ : Type u_12
    inst✝¹¹ : TopologicalSpace HB₂
    IB₂ : ModelWithCorners 𝕜 EB₂ HB₂
    inst✝¹⁰ : TopologicalSpace B₂
    inst✝⁹ : ChartedSpace HB₂ B₂
    EM : Type u_13
    inst✝⁸ : NormedAddCommGroup EM
    inst✝⁷ : NormedSpace 𝕜 EM
    HM : Type u_14
    inst✝⁶ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    n : ENat
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    s : Set M
    hϕ : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHo …
    hv : And (ContMDiffWithinAt IM IB₁ n (fun x => { proj := b₁ x, snd := v x }.pr …
    hb₂ : ContMDiffWithinAt IM IB₂ n b₂ (Insert.insert m₀ s) m₀
    ⊢ ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F₂) n (fun x => (↑(FiberBundle. …
  -/
  apply (ContMDiffWithinAt.clm_apply hϕ hv.2).congr_of_eventuallyEq_of_mem ?_ (mem_insert m₀ s)
  have A : ∀ᶠ m in 𝓝[insert m₀ s] m₀, b₁ m ∈ (trivializationAt F₁ E₁ (b₁ m₀)).baseSet := by
    apply hv.1.continuousWithinAt
    apply (trivializationAt F₁ E₁ (b₁ m₀)).open_baseSet.mem_nhds
    exact FiberBundle.mem_baseSet_trivializationAt' (b₁ m₀)
  have A' : ∀ᶠ m in 𝓝[insert m₀ s] m₀, b₂ m ∈ (trivializationAt F₂ E₂ (b₂ m₀)).baseSet := by
    apply hb₂.continuousWithinAt
    apply (trivializationAt F₂ E₂ (b₂ m₀)).open_baseSet.mem_nhds
    exact FiberBundle.mem_baseSet_trivializationAt' (b₂ m₀)
  /-
    𝕜 : Type u_1
    F₁ : Type u_2
    F₂ : Type u_3
    B₁ : Type u_4
    B₂ : Type u_5
    M : Type u_6
    E₁ : B₁ → Type u_7
    E₂ : B₂ → Type u_8
    inst✝³¹ : NontriviallyNormedField 𝕜
    inst✝³⁰ : (x : B₁) → AddCommGroup (E₁ x)
    inst✝²⁹ : (x : B₁) → Module 𝕜 (E₁ x)
    inst✝²⁸ : NormedAddCommGroup F₁
    inst✝²⁷ : NormedSpace 𝕜 F₁
    inst✝²⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²⁵ : (x : B₁) → TopologicalSpace (E₁ x)
    inst✝²⁴ : (x : B₂) → AddCommGroup (E₂ x)
    inst✝²³ : (x : B₂) → Module 𝕜 (E₂ x)
    inst✝²² : NormedAddCommGroup F₂
    inst✝²¹ : NormedSpace 𝕜 F₂
    inst✝²⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁹ : (x : B₂) → TopologicalSpace (E₂ x)
    EB₁ : Type u_9
    inst✝¹⁸ : NormedAddCommGroup EB₁
    inst✝¹⁷ : NormedSpace 𝕜 EB₁
    HB₁ : Type u_10
    inst✝¹⁶ : TopologicalSpace HB₁
    IB₁ : ModelWithCorners 𝕜 EB₁ HB₁
    inst✝¹⁵ : TopologicalSpace B₁
    inst✝¹⁴ : ChartedSpace HB₁ B₁
    EB₂ : Type u_11
    inst✝¹³ : NormedAddCommGroup EB₂
    inst✝¹² : NormedSpace 𝕜 EB₂
    HB₂ : Type u_12
    inst✝¹¹ : TopologicalSpace HB₂
    IB₂ : ModelWithCorners 𝕜 EB₂ HB₂
    inst✝¹⁰ : TopologicalSpace B₂
    inst✝⁹ : ChartedSpace HB₂ B₂
    EM : Type u_13
    inst✝⁸ : NormedAddCommGroup EM
    inst✝⁷ : NormedSpace 𝕜 EM
    HM : Type u_14
    inst✝⁶ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    n : ENat
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    s : Set M
    hϕ : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHo …
    hv : And (ContMDiffWithinAt IM IB₁ n (fun x => { proj := b₁ x, snd := v x }.pr …
    hb₂ : ContMDiffWithinAt IM IB₂ n b₂ (Insert.insert m₀ s) m₀
    A : Filter.Eventually (fun m => Membership.mem (FiberBundle.trivializationAt F …
    A' : Filter.Eventually (fun m => Membership.mem (FiberBundle.trivializationAt  …
    ⊢ (nhdsWithin m₀ (Insert.insert m₀ s)).EventuallyEq (fun x => (↑(FiberBundle.t …
  -/
  filter_upwards [A, A'] with m hm h'm
  /-
    case h
    𝕜 : Type u_1
    F₁ : Type u_2
    F₂ : Type u_3
    B₁ : Type u_4
    B₂ : Type u_5
    M : Type u_6
    E₁ : B₁ → Type u_7
    E₂ : B₂ → Type u_8
    inst✝³¹ : NontriviallyNormedField 𝕜
    inst✝³⁰ : (x : B₁) → AddCommGroup (E₁ x)
    inst✝²⁹ : (x : B₁) → Module 𝕜 (E₁ x)
    inst✝²⁸ : NormedAddCommGroup F₁
    inst✝²⁷ : NormedSpace 𝕜 F₁
    inst✝²⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²⁵ : (x : B₁) → TopologicalSpace (E₁ x)
    inst✝²⁴ : (x : B₂) → AddCommGroup (E₂ x)
    inst✝²³ : (x : B₂) → Module 𝕜 (E₂ x)
    inst✝²² : NormedAddCommGroup F₂
    inst✝²¹ : NormedSpace 𝕜 F₂
    inst✝²⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁹ : (x : B₂) → TopologicalSpace (E₂ x)
    EB₁ : Type u_9
    inst✝¹⁸ : NormedAddCommGroup EB₁
    inst✝¹⁷ : NormedSpace 𝕜 EB₁
    HB₁ : Type u_10
    inst✝¹⁶ : TopologicalSpace HB₁
    IB₁ : ModelWithCorners 𝕜 EB₁ HB₁
    inst✝¹⁵ : TopologicalSpace B₁
    inst✝¹⁴ : ChartedSpace HB₁ B₁
    EB₂ : Type u_11
    inst✝¹³ : NormedAddCommGroup EB₂
    inst✝¹² : NormedSpace 𝕜 EB₂
    HB₂ : Type u_12
    inst✝¹¹ : TopologicalSpace HB₂
    IB₂ : ModelWithCorners 𝕜 EB₂ HB₂
    inst✝¹⁰ : TopologicalSpace B₂
    inst✝⁹ : ChartedSpace HB₂ B₂
    EM : Type u_13
    inst✝⁸ : NormedAddCommGroup EM
    inst✝⁷ : NormedSpace 𝕜 EM
    HM : Type u_14
    inst✝⁶ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    n : ENat
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    s : Set M
    hϕ : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHo …
    hv : And (ContMDiffWithinAt IM IB₁ n (fun x => { proj := b₁ x, snd := v x }.pr …
    hb₂ : ContMDiffWithinAt IM IB₂ n b₂ (Insert.insert m₀ s) m₀
    A : Filter.Eventually (fun m => Membership.mem (FiberBundle.trivializationAt F …
    A' : Filter.Eventually (fun m => Membership.mem (FiberBundle.trivializationAt  …
    m : M
    hm : Membership.mem (FiberBundle.trivializationAt F₁ E₁ (b₁ m₀)).baseSet (b₁ m)
    h'm : Membership.mem (FiberBundle.trivializationAt F₂ E₂ (b₂ m₀)).baseSet (b₂ m)
    ⊢ Eq (↑(FiberBundle.trivializationAt F₂ E₂ (b₂ m₀)) { proj := b₂ m, snd := (ϕ  …
  -/
  rw [inCoordinates_eq hm h'm]
  simp only [coe_comp', ContinuousLinearEquiv.coe_coe, Trivialization.continuousLinearEquivAt_apply,
    Trivialization.continuousLinearEquivAt_symm_apply, Function.comp_apply]
  /-
    case h
    𝕜 : Type u_1
    F₁ : Type u_2
    F₂ : Type u_3
    B₁ : Type u_4
    B₂ : Type u_5
    M : Type u_6
    E₁ : B₁ → Type u_7
    E₂ : B₂ → Type u_8
    inst✝³¹ : NontriviallyNormedField 𝕜
    inst✝³⁰ : (x : B₁) → AddCommGroup (E₁ x)
    inst✝²⁹ : (x : B₁) → Module 𝕜 (E₁ x)
    inst✝²⁸ : NormedAddCommGroup F₁
    inst✝²⁷ : NormedSpace 𝕜 F₁
    inst✝²⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²⁵ : (x : B₁) → TopologicalSpace (E₁ x)
    inst✝²⁴ : (x : B₂) → AddCommGroup (E₂ x)
    inst✝²³ : (x : B₂) → Module 𝕜 (E₂ x)
    inst✝²² : NormedAddCommGroup F₂
    inst✝²¹ : NormedSpace 𝕜 F₂
    inst✝²⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁹ : (x : B₂) → TopologicalSpace (E₂ x)
    EB₁ : Type u_9
    inst✝¹⁸ : NormedAddCommGroup EB₁
    inst✝¹⁷ : NormedSpace 𝕜 EB₁
    HB₁ : Type u_10
    inst✝¹⁶ : TopologicalSpace HB₁
    IB₁ : ModelWithCorners 𝕜 EB₁ HB₁
    inst✝¹⁵ : TopologicalSpace B₁
    inst✝¹⁴ : ChartedSpace HB₁ B₁
    EB₂ : Type u_11
    inst✝¹³ : NormedAddCommGroup EB₂
    inst✝¹² : NormedSpace 𝕜 EB₂
    HB₂ : Type u_12
    inst✝¹¹ : TopologicalSpace HB₂
    IB₂ : ModelWithCorners 𝕜 EB₂ HB₂
    inst✝¹⁰ : TopologicalSpace B₂
    inst✝⁹ : ChartedSpace HB₂ B₂
    EM : Type u_13
    inst✝⁸ : NormedAddCommGroup EM
    inst✝⁷ : NormedSpace 𝕜 EM
    HM : Type u_14
    inst✝⁶ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    n : ENat
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    s : Set M
    hϕ : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHo …
    hv : And (ContMDiffWithinAt IM IB₁ n (fun x => { proj := b₁ x, snd := v x }.pr …
    hb₂ : ContMDiffWithinAt IM IB₂ n b₂ (Insert.insert m₀ s) m₀
    A : Filter.Eventually (fun m => Membership.mem (FiberBundle.trivializationAt F …
    A' : Filter.Eventually (fun m => Membership.mem (FiberBundle.trivializationAt  …
    m : M
    hm : Membership.mem (FiberBundle.trivializationAt F₁ E₁ (b₁ m₀)).baseSet (b₁ m)
    h'm : Membership.mem (FiberBundle.trivializationAt F₂ E₂ (b₂ m₀)).baseSet (b₂ m)
    ⊢ Eq (↑(FiberBundle.trivializationAt F₂ E₂ (b₂ m₀)) { proj := b₂ m, snd := (ϕ  …
  -/
  congr
  /-
    case h.e_self.e_a.e_snd.h.e_6.h
    𝕜 : Type u_1
    F₁ : Type u_2
    F₂ : Type u_3
    B₁ : Type u_4
    B₂ : Type u_5
    M : Type u_6
    E₁ : B₁ → Type u_7
    E₂ : B₂ → Type u_8
    inst✝³¹ : NontriviallyNormedField 𝕜
    inst✝³⁰ : (x : B₁) → AddCommGroup (E₁ x)
    inst✝²⁹ : (x : B₁) → Module 𝕜 (E₁ x)
    inst✝²⁸ : NormedAddCommGroup F₁
    inst✝²⁷ : NormedSpace 𝕜 F₁
    inst✝²⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²⁵ : (x : B₁) → TopologicalSpace (E₁ x)
    inst✝²⁴ : (x : B₂) → AddCommGroup (E₂ x)
    inst✝²³ : (x : B₂) → Module 𝕜 (E₂ x)
    inst✝²² : NormedAddCommGroup F₂
    inst✝²¹ : NormedSpace 𝕜 F₂
    inst✝²⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁹ : (x : B₂) → TopologicalSpace (E₂ x)
    EB₁ : Type u_9
    inst✝¹⁸ : NormedAddCommGroup EB₁
    inst✝¹⁷ : NormedSpace 𝕜 EB₁
    HB₁ : Type u_10
    inst✝¹⁶ : TopologicalSpace HB₁
    IB₁ : ModelWithCorners 𝕜 EB₁ HB₁
    inst✝¹⁵ : TopologicalSpace B₁
    inst✝¹⁴ : ChartedSpace HB₁ B₁
    EB₂ : Type u_11
    inst✝¹³ : NormedAddCommGroup EB₂
    inst✝¹² : NormedSpace 𝕜 EB₂
    HB₂ : Type u_12
    inst✝¹¹ : TopologicalSpace HB₂
    IB₂ : ModelWithCorners 𝕜 EB₂ HB₂
    inst✝¹⁰ : TopologicalSpace B₂
    inst✝⁹ : ChartedSpace HB₂ B₂
    EM : Type u_13
    inst✝⁸ : NormedAddCommGroup EM
    inst✝⁷ : NormedSpace 𝕜 EM
    HM : Type u_14
    inst✝⁶ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    n : ENat
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    s : Set M
    hϕ : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHo …
    hv : And (ContMDiffWithinAt IM IB₁ n (fun x => { proj := b₁ x, snd := v x }.pr …
    hb₂ : ContMDiffWithinAt IM IB₂ n b₂ (Insert.insert m₀ s) m₀
    A : Filter.Eventually (fun m => Membership.mem (FiberBundle.trivializationAt F …
    A' : Filter.Eventually (fun m => Membership.mem (FiberBundle.trivializationAt  …
    m : M
    hm : Membership.mem (FiberBundle.trivializationAt F₁ E₁ (b₁ m₀)).baseSet (b₁ m)
    h'm : Membership.mem (FiberBundle.trivializationAt F₂ E₂ (b₂ m₀)).baseSet (b₂ m)
    ⊢ Eq (v m) ((FiberBundle.trivializationAt F₁ E₁ (b₁ m₀)).symm (b₁ m) (↑(FiberB …
  -/
  rw [Trivialization.symm_apply_apply_mk (trivializationAt F₁ E₁ (b₁ m₀)) hm (v m)]
  /-
    🎉 no goals
  -/


/-- Consider a smooth map `v : M → E₁` to a vector bundle, over a basemap `b₁ : M → B₁`, and
another basemap `b₂ : M → B₂`. Given linear maps `ϕ m : E₁ (b₁ m) → E₂ (b₂ m)` depending smoothly
on `m`, one can apply `ϕ m` to `g m`, and the resulting map is smooth.

Note that the smoothness of `ϕ` can not be always be stated as smoothness of a map into a manifold,
as the pullback bundles `b₁ *ᵖ E₁` and `b₂ *ᵖ E₂` only make sense when `b₁` and `b₂` are globally
smooth, but we want to apply this lemma with only local information. Therefore, we formulate it
using smoothness of `ϕ` read in coordinates.

Version for `ContMDiffAt`. We also give a version for `ContMDiffWithinAt`, but no version for
`ContMDiffOn` or `ContMDiff` as our assumption, written in coordinates, only makes sense around
a point.
 -/
lemma ContMDiffAt.clm_apply_of_inCoordinates
    (hϕ : ContMDiffAt IM 𝓘(𝕜, F₁ →L[𝕜] F₂) n
      (fun m ↦ inCoordinates F₁ E₁ F₂ E₂ (b₁ m₀) (b₁ m) (b₂ m₀) (b₂ m) (ϕ m)) m₀)
    (hv : ContMDiffAt IM (IB₁.prod 𝓘(𝕜, F₁)) n (fun m ↦ (v m : TotalSpace F₁ E₁)) m₀)
    (hb₂ : ContMDiffAt IM IB₂ n b₂ m₀) :
    ContMDiffAt IM (IB₂.prod 𝓘(𝕜, F₂)) n (fun m ↦ (ϕ m (v m) : TotalSpace F₂ E₂)) m₀ := by
  /-
    𝕜 : Type u_1
    F₁ : Type u_2
    F₂ : Type u_3
    B₁ : Type u_4
    B₂ : Type u_5
    M : Type u_6
    E₁ : B₁ → Type u_7
    E₂ : B₂ → Type u_8
    inst✝³¹ : NontriviallyNormedField 𝕜
    inst✝³⁰ : (x : B₁) → AddCommGroup (E₁ x)
    inst✝²⁹ : (x : B₁) → Module 𝕜 (E₁ x)
    inst✝²⁸ : NormedAddCommGroup F₁
    inst✝²⁷ : NormedSpace 𝕜 F₁
    inst✝²⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²⁵ : (x : B₁) → TopologicalSpace (E₁ x)
    inst✝²⁴ : (x : B₂) → AddCommGroup (E₂ x)
    inst✝²³ : (x : B₂) → Module 𝕜 (E₂ x)
    inst✝²² : NormedAddCommGroup F₂
    inst✝²¹ : NormedSpace 𝕜 F₂
    inst✝²⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁹ : (x : B₂) → TopologicalSpace (E₂ x)
    EB₁ : Type u_9
    inst✝¹⁸ : NormedAddCommGroup EB₁
    inst✝¹⁷ : NormedSpace 𝕜 EB₁
    HB₁ : Type u_10
    inst✝¹⁶ : TopologicalSpace HB₁
    IB₁ : ModelWithCorners 𝕜 EB₁ HB₁
    inst✝¹⁵ : TopologicalSpace B₁
    inst✝¹⁴ : ChartedSpace HB₁ B₁
    EB₂ : Type u_11
    inst✝¹³ : NormedAddCommGroup EB₂
    inst✝¹² : NormedSpace 𝕜 EB₂
    HB₂ : Type u_12
    inst✝¹¹ : TopologicalSpace HB₂
    IB₂ : ModelWithCorners 𝕜 EB₂ HB₂
    inst✝¹⁰ : TopologicalSpace B₂
    inst✝⁹ : ChartedSpace HB₂ B₂
    EM : Type u_13
    inst✝⁸ : NormedAddCommGroup EM
    inst✝⁷ : NormedSpace 𝕜 EM
    HM : Type u_14
    inst✝⁶ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    n : ENat
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    hϕ : ContMDiffAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
    hv : ContMDiffAt IM (IB₁.prod (modelWithCornersSelf 𝕜 F₁)) n (fun m => { proj  …
    hb₂ : ContMDiffAt IM IB₂ n b₂ m₀
    ⊢ ContMDiffAt IM (IB₂.prod (modelWithCornersSelf 𝕜 F₂)) n (fun m => { proj :=  …
  -/
  rw [← contMDiffWithinAt_univ] at hϕ hv hb₂ ⊢
  /-
    𝕜 : Type u_1
    F₁ : Type u_2
    F₂ : Type u_3
    B₁ : Type u_4
    B₂ : Type u_5
    M : Type u_6
    E₁ : B₁ → Type u_7
    E₂ : B₂ → Type u_8
    inst✝³¹ : NontriviallyNormedField 𝕜
    inst✝³⁰ : (x : B₁) → AddCommGroup (E₁ x)
    inst✝²⁹ : (x : B₁) → Module 𝕜 (E₁ x)
    inst✝²⁸ : NormedAddCommGroup F₁
    inst✝²⁷ : NormedSpace 𝕜 F₁
    inst✝²⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    inst✝²⁵ : (x : B₁) → TopologicalSpace (E₁ x)
    inst✝²⁴ : (x : B₂) → AddCommGroup (E₂ x)
    inst✝²³ : (x : B₂) → Module 𝕜 (E₂ x)
    inst✝²² : NormedAddCommGroup F₂
    inst✝²¹ : NormedSpace 𝕜 F₂
    inst✝²⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝¹⁹ : (x : B₂) → TopologicalSpace (E₂ x)
    EB₁ : Type u_9
    inst✝¹⁸ : NormedAddCommGroup EB₁
    inst✝¹⁷ : NormedSpace 𝕜 EB₁
    HB₁ : Type u_10
    inst✝¹⁶ : TopologicalSpace HB₁
    IB₁ : ModelWithCorners 𝕜 EB₁ HB₁
    inst✝¹⁵ : TopologicalSpace B₁
    inst✝¹⁴ : ChartedSpace HB₁ B₁
    EB₂ : Type u_11
    inst✝¹³ : NormedAddCommGroup EB₂
    inst✝¹² : NormedSpace 𝕜 EB₂
    HB₂ : Type u_12
    inst✝¹¹ : TopologicalSpace HB₂
    IB₂ : ModelWithCorners 𝕜 EB₂ HB₂
    inst✝¹⁰ : TopologicalSpace B₂
    inst✝⁹ : ChartedSpace HB₂ B₂
    EM : Type u_13
    inst✝⁸ : NormedAddCommGroup EM
    inst✝⁷ : NormedSpace 𝕜 EM
    HM : Type u_14
    inst✝⁶ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    n : ENat
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    hϕ : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHo …
    hv : ContMDiffWithinAt IM (IB₁.prod (modelWithCornersSelf 𝕜 F₁)) n (fun m => { …
    hb₂ : ContMDiffWithinAt IM IB₂ n b₂ Set.univ m₀
    ⊢ ContMDiffWithinAt IM (IB₂.prod (modelWithCornersSelf 𝕜 F₂)) n (fun m => { pr …
  -/
  exact ContMDiffWithinAt.clm_apply_of_inCoordinates hϕ hv hb₂
  /-
    🎉 no goals
  -/


