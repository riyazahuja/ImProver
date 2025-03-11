/-- Characterization of differentiable functions into a smooth vector bundle. -/
theorem mdifferentiableWithinAt_totalSpace (f : M → TotalSpace F E) {s : Set M} {x₀ : M} :
    MDifferentiableWithinAt IM (IB.prod 𝓘(𝕜, F)) f s x₀ ↔
      MDifferentiableWithinAt IM IB (fun x => (f x).proj) s x₀ ∧
      MDifferentiableWithinAt IM 𝓘(𝕜, F)
        (fun x ↦ (trivializationAt F E (f x₀).proj (f x)).2) s x₀ := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    s : Set M
    x₀ : M
    ⊢ Iff (MDifferentiableWithinAt IM (IB.prod (modelWithCornersSelf 𝕜 F)) f s x₀) …
  -/
  simp (config := { singlePass := true }) only [mdifferentiableWithinAt_iff_target]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    s : Set M
    x₀ : M
    ⊢ Iff (And (ContinuousWithinAt f s x₀) (MDifferentiableWithinAt IM (modelWithC …
  -/
  rw [and_and_and_comm, ← FiberBundle.continuousWithinAt_totalSpace, and_congr_right_iff]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    s : Set M
    x₀ : M
    ⊢ ContinuousWithinAt f s x₀ → Iff (MDifferentiableWithinAt IM (modelWithCorner …
  -/
  intro hf
  simp_rw [modelWithCornersSelf_prod, FiberBundle.extChartAt, Function.comp_def,
    PartialEquiv.trans_apply, PartialEquiv.prod_coe, PartialEquiv.refl_coe,
    extChartAt_self_apply, modelWithCornersSelf_coe, Function.id_def, ← chartedSpaceSelf_prod]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    s : Set M
    x₀ : M
    hf : ContinuousWithinAt f s x₀
    ⊢ Iff (MDifferentiableWithinAt IM ((modelWithCornersSelf 𝕜 EB).prod (modelWith …
  -/
  refine (mdifferentiableWithinAt_prod_iff _).trans (and_congr ?_ Iff.rfl)
  have h1 : (fun x => (f x).proj) ⁻¹' (trivializationAt F E (f x₀).proj).baseSet ∈ 𝓝[s] x₀ :=
    ((FiberBundle.continuous_proj F E).continuousWithinAt.comp hf (mapsTo_image f s))
      ((Trivialization.open_baseSet _).mem_nhds (mem_baseSet_trivializationAt F E _))
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    s : Set M
    x₀ : M
    hf : ContinuousWithinAt f s x₀
    h1 : Membership.mem (nhdsWithin x₀ s) (Set.preimage (fun x => (f x).proj) (Fib …
    ⊢ Iff (MDifferentiableWithinAt IM (modelWithCornersSelf 𝕜 EB) (Function.comp P …
  -/
  refine EventuallyEq.mdifferentiableWithinAt_iff (eventually_of_mem h1 fun x hx => ?_) ?_
    /-
      case refine_1
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      inst✝¹⁴ : NormedAddCommGroup F
      inst✝¹³ : NormedSpace 𝕜 F
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹¹ : (x : B) → TopologicalSpace (E x)
      EB : Type u_7
      inst✝¹⁰ : NormedAddCommGroup EB
      inst✝⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      EM : Type u_10
      inst✝⁷ : NormedAddCommGroup EM
      inst✝⁶ : NormedSpace 𝕜 EM
      HM : Type u_11
      inst✝⁵ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace HM M
      inst✝² : TopologicalSpace B
      inst✝¹ : ChartedSpace HB B
      inst✝ : FiberBundle F E
      f : M → Bundle.TotalSpace F E
      s : Set M
      x₀ : M
      hf : ContinuousWithinAt f s x₀
      h1 : Membership.mem (nhdsWithin x₀ s) (Set.preimage (fun x => (f x).proj) (Fib …
      x : M
      hx : Membership.mem (Set.preimage (fun x => (f x).proj) (FiberBundle.trivializ …
      ⊢ Eq ((fun x => ↑(extChartAt IB (f x₀).proj) (f x).proj) x) (Function.comp Pro …
    -/
  · simp_rw [Function.comp, PartialHomeomorph.coe_coe, Trivialization.coe_coe]
    /-
      case refine_1
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      inst✝¹⁴ : NormedAddCommGroup F
      inst✝¹³ : NormedSpace 𝕜 F
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹¹ : (x : B) → TopologicalSpace (E x)
      EB : Type u_7
      inst✝¹⁰ : NormedAddCommGroup EB
      inst✝⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      EM : Type u_10
      inst✝⁷ : NormedAddCommGroup EM
      inst✝⁶ : NormedSpace 𝕜 EM
      HM : Type u_11
      inst✝⁵ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace HM M
      inst✝² : TopologicalSpace B
      inst✝¹ : ChartedSpace HB B
      inst✝ : FiberBundle F E
      f : M → Bundle.TotalSpace F E
      s : Set M
      x₀ : M
      hf : ContinuousWithinAt f s x₀
      h1 : Membership.mem (nhdsWithin x₀ s) (Set.preimage (fun x => (f x).proj) (Fib …
      x : M
      hx : Membership.mem (Set.preimage (fun x => (f x).proj) (FiberBundle.trivializ …
      ⊢ Eq (↑(extChartAt IB (f x₀).proj) (f x).proj) (↑(extChartAt IB (f x₀).proj) ( …
    -/
    rw [Trivialization.coe_fst']
    /-
      case refine_1.ex
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      inst✝¹⁴ : NormedAddCommGroup F
      inst✝¹³ : NormedSpace 𝕜 F
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹¹ : (x : B) → TopologicalSpace (E x)
      EB : Type u_7
      inst✝¹⁰ : NormedAddCommGroup EB
      inst✝⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      EM : Type u_10
      inst✝⁷ : NormedAddCommGroup EM
      inst✝⁶ : NormedSpace 𝕜 EM
      HM : Type u_11
      inst✝⁵ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace HM M
      inst✝² : TopologicalSpace B
      inst✝¹ : ChartedSpace HB B
      inst✝ : FiberBundle F E
      f : M → Bundle.TotalSpace F E
      s : Set M
      x₀ : M
      hf : ContinuousWithinAt f s x₀
      h1 : Membership.mem (nhdsWithin x₀ s) (Set.preimage (fun x => (f x).proj) (Fib …
      x : M
      hx : Membership.mem (Set.preimage (fun x => (f x).proj) (FiberBundle.trivializ …
      ⊢ Membership.mem (FiberBundle.trivializationAt F E (f x₀).proj).baseSet (f x). …
    -/
    exact hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      inst✝¹⁴ : NormedAddCommGroup F
      inst✝¹³ : NormedSpace 𝕜 F
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹¹ : (x : B) → TopologicalSpace (E x)
      EB : Type u_7
      inst✝¹⁰ : NormedAddCommGroup EB
      inst✝⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      EM : Type u_10
      inst✝⁷ : NormedAddCommGroup EM
      inst✝⁶ : NormedSpace 𝕜 EM
      HM : Type u_11
      inst✝⁵ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace HM M
      inst✝² : TopologicalSpace B
      inst✝¹ : ChartedSpace HB B
      inst✝ : FiberBundle F E
      f : M → Bundle.TotalSpace F E
      s : Set M
      x₀ : M
      hf : ContinuousWithinAt f s x₀
      h1 : Membership.mem (nhdsWithin x₀ s) (Set.preimage (fun x => (f x).proj) (Fib …
      ⊢ Eq (↑(extChartAt IB (f x₀).proj) (f x₀).proj) (Function.comp Prod.fst (fun x …
    -/
  · simp only [mfld_simps]
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

Version for `MDifferentiableWithinAt`. We also give a version for `MDifferentiableAt`, but no
version for `MDifferentiableOn` or `MDifferentiable` as our assumption, written in coordinates,
only makes sense around a point.
 -/
lemma MDifferentiableWithinAt.clm_apply_of_inCoordinates
    (hϕ : MDifferentiableWithinAt IM 𝓘(𝕜, F₁ →L[𝕜] F₂)
      (fun m ↦ inCoordinates F₁ E₁ F₂ E₂ (b₁ m₀) (b₁ m) (b₂ m₀) (b₂ m) (ϕ m)) s m₀)
    (hv : MDifferentiableWithinAt IM (IB₁.prod 𝓘(𝕜, F₁)) (fun m ↦ (v m : TotalSpace F₁ E₁)) s m₀)
    (hb₂ : MDifferentiableWithinAt IM IB₂ b₂ s m₀) :
    MDifferentiableWithinAt IM (IB₂.prod 𝓘(𝕜, F₂))
      (fun m ↦ (ϕ m (v m) : TotalSpace F₂ E₂)) s m₀ := by
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
    hϕ : MDifferentiableWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap ( …
    hv : MDifferentiableWithinAt IM (IB₁.prod (modelWithCornersSelf 𝕜 F₁)) (fun m  …
    hb₂ : MDifferentiableWithinAt IM IB₂ b₂ s m₀
    ⊢ MDifferentiableWithinAt IM (IB₂.prod (modelWithCornersSelf 𝕜 F₂)) (fun m =>  …
  -/
  rw [mdifferentiableWithinAt_totalSpace] at hv ⊢
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
    hϕ : MDifferentiableWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap ( …
    hv : And (MDifferentiableWithinAt IM IB₁ (fun x => { proj := b₁ x, snd := v x  …
    hb₂ : MDifferentiableWithinAt IM IB₂ b₂ s m₀
    ⊢ And (MDifferentiableWithinAt IM IB₂ (fun x => { proj := b₂ x, snd := (ϕ x) ( …
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
    hϕ : MDifferentiableWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap ( …
    hv : And (MDifferentiableWithinAt IM IB₁ (fun x => { proj := b₁ x, snd := v x  …
    hb₂ : MDifferentiableWithinAt IM IB₂ b₂ s m₀
    ⊢ MDifferentiableWithinAt IM (modelWithCornersSelf 𝕜 F₂) (fun x => (↑(FiberBun …
  -/
  apply (MDifferentiableWithinAt.clm_apply hϕ hv.2).congr_of_eventuallyEq_insert
  have A : ∀ᶠ m in 𝓝[insert m₀ s] m₀, b₁ m ∈ (trivializationAt F₁ E₁ (b₁ m₀)).baseSet := by
    apply hv.1.insert.continuousWithinAt
    apply (trivializationAt F₁ E₁ (b₁ m₀)).open_baseSet.mem_nhds
    exact FiberBundle.mem_baseSet_trivializationAt' (b₁ m₀)
  have A' : ∀ᶠ m in 𝓝[insert m₀ s] m₀, b₂ m ∈ (trivializationAt F₂ E₂ (b₂ m₀)).baseSet := by
    apply hb₂.insert.continuousWithinAt
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
    hϕ : MDifferentiableWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap ( …
    hv : And (MDifferentiableWithinAt IM IB₁ (fun x => { proj := b₁ x, snd := v x  …
    hb₂ : MDifferentiableWithinAt IM IB₂ b₂ s m₀
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
    hϕ : MDifferentiableWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap ( …
    hv : And (MDifferentiableWithinAt IM IB₁ (fun x => { proj := b₁ x, snd := v x  …
    hb₂ : MDifferentiableWithinAt IM IB₂ b₂ s m₀
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
    hϕ : MDifferentiableWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap ( …
    hv : And (MDifferentiableWithinAt IM IB₁ (fun x => { proj := b₁ x, snd := v x  …
    hb₂ : MDifferentiableWithinAt IM IB₂ b₂ s m₀
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
    hϕ : MDifferentiableWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap ( …
    hv : And (MDifferentiableWithinAt IM IB₁ (fun x => { proj := b₁ x, snd := v x  …
    hb₂ : MDifferentiableWithinAt IM IB₂ b₂ s m₀
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

Version for `MDifferentiableAt`. We also give a version for `MDifferentiableWithinAt`,
but no version for `MDifferentiableOn` or `MDifferentiable` as our assumption, written
in coordinates, only makes sense around a point.
 -/
lemma MDifferentiableAt.clm_apply_of_inCoordinates
    (hϕ : MDifferentiableAt IM 𝓘(𝕜, F₁ →L[𝕜] F₂)
      (fun m ↦ inCoordinates F₁ E₁ F₂ E₂ (b₁ m₀) (b₁ m) (b₂ m₀) (b₂ m) (ϕ m)) m₀)
    (hv : MDifferentiableAt IM (IB₁.prod 𝓘(𝕜, F₁)) (fun m ↦ (v m : TotalSpace F₁ E₁)) m₀)
    (hb₂ : MDifferentiableAt IM IB₂ b₂ m₀) :
    MDifferentiableAt IM (IB₂.prod 𝓘(𝕜, F₂)) (fun m ↦ (ϕ m (v m) : TotalSpace F₂ E₂)) m₀ := by
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
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    hϕ : MDifferentiableAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHo …
    hv : MDifferentiableAt IM (IB₁.prod (modelWithCornersSelf 𝕜 F₁)) (fun m => { p …
    hb₂ : MDifferentiableAt IM IB₂ b₂ m₀
    ⊢ MDifferentiableAt IM (IB₂.prod (modelWithCornersSelf 𝕜 F₂)) (fun m => { proj …
  -/
  rw [← mdifferentiableWithinAt_univ] at hϕ hv hb₂ ⊢
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
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : VectorBundle 𝕜 F₁ E₁
    inst✝¹ : FiberBundle F₂ E₂
    inst✝ : VectorBundle 𝕜 F₂ E₂
    b₁ : M → B₁
    b₂ : M → B₂
    m₀ : M
    ϕ : (m : M) → ContinuousLinearMap (RingHom.id 𝕜) (E₁ (b₁ m)) (E₂ (b₂ m))
    v : (m : M) → E₁ (b₁ m)
    hϕ : MDifferentiableWithinAt IM (modelWithCornersSelf 𝕜 (ContinuousLinearMap ( …
    hv : MDifferentiableWithinAt IM (IB₁.prod (modelWithCornersSelf 𝕜 F₁)) (fun m  …
    hb₂ : MDifferentiableWithinAt IM IB₂ b₂ Set.univ m₀
    ⊢ MDifferentiableWithinAt IM (IB₂.prod (modelWithCornersSelf 𝕜 F₂)) (fun m =>  …
  -/
  exact MDifferentiableWithinAt.clm_apply_of_inCoordinates hϕ hv hb₂
  /-
    🎉 no goals
  -/


