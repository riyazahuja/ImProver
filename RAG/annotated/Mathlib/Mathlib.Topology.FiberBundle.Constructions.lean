instance topologicalSpace [t₁ : TopologicalSpace B]
    [t₂ : TopologicalSpace F] : TopologicalSpace (TotalSpace F (Trivial B F)) :=
  induced TotalSpace.proj t₁ ⊓ induced (TotalSpace.trivialSnd B F) t₂


theorem isInducing_toProd : IsInducing (TotalSpace.toProd B F) :=
      /-
        B : Type u_1
        F : Type u_2
        inst✝¹ : TopologicalSpace B
        inst✝ : TopologicalSpace F
        ⊢ Eq (Bundle.Trivial.topologicalSpace B F) (TopologicalSpace.induced (⇑(Bundle …
      -/
  ⟨by simp only [instTopologicalSpaceProd, induced_inf, induced_compose]; rfl⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[deprecated (since := "2024-10-28")] alias inducing_toProd := isInducing_toProd


/-- Homeomorphism between the total space of the trivial bundle and the Cartesian product. -/
def homeomorphProd : TotalSpace F (Trivial B F) ≃ₜ B × F :=
  (TotalSpace.toProd _ _).toHomeomorphOfIsInducing (isInducing_toProd B F)


/-- Local trivialization for trivial bundle. -/
def trivialization : Trivialization F (π F (Bundle.Trivial B F)) where
  -- Porting note: golfed
  toPartialHomeomorph := (homeomorphProd B F).toPartialHomeomorph
  baseSet := univ
  open_baseSet := isOpen_univ
  source_eq := rfl
  target_eq := univ_prod_univ.symm
  proj_toFun _ _ := rfl


@[simp]
theorem trivialization_source : (trivialization B F).source = univ := rfl


@[simp]
theorem trivialization_target : (trivialization B F).target = univ := rfl


/-- Fiber bundle instance on the trivial bundle. -/
instance fiberBundle : FiberBundle F (Bundle.Trivial B F) where
  trivializationAtlas' := {trivialization B F}
  trivializationAt' _ := trivialization B F
  mem_baseSet_trivializationAt' := mem_univ
  trivialization_mem_atlas' _ := mem_singleton _
  totalSpaceMk_isInducing' _ := (homeomorphProd B F).symm.isInducing.comp
    (isInducing_const_prod.2 .id)


theorem eq_trivialization (e : Trivialization F (π F (Bundle.Trivial B F)))
    [i : MemTrivializationAtlas e] : e = trivialization B F := i.out


/-- Equip the total space of the fiberwise product of two fiber bundles `E₁`, `E₂` with
the induced topology from the diagonal embedding into `TotalSpace F₁ E₁ × TotalSpace F₂ E₂`. -/
instance FiberBundle.Prod.topologicalSpace : TopologicalSpace (TotalSpace (F₁ × F₂) (E₁ ×ᵇ E₂)) :=
  TopologicalSpace.induced
    (fun p ↦ ((⟨p.1, p.2.1⟩ : TotalSpace F₁ E₁), (⟨p.1, p.2.2⟩ : TotalSpace F₂ E₂)))
    inferInstance


/-- The diagonal map from the total space of the fiberwise product of two fiber bundles
`E₁`, `E₂` into `TotalSpace F₁ E₁ × TotalSpace F₂ E₂` is an inducing map. -/
theorem FiberBundle.Prod.isInducing_diag :
    IsInducing (fun p ↦ (⟨p.1, p.2.1⟩, ⟨p.1, p.2.2⟩) :
      TotalSpace (F₁ × F₂) (E₁ ×ᵇ E₂) → TotalSpace F₁ E₁ × TotalSpace F₂ E₂) :=
  ⟨rfl⟩


@[deprecated (since := "2024-10-28")]
alias FiberBundle.Prod.inducing_diag := FiberBundle.Prod.isInducing_diag


/-- Given trivializations `e₁`, `e₂` for fiber bundles `E₁`, `E₂` over a base `B`, the forward
function for the construction `Trivialization.prod`, the induced
trivialization for the fiberwise product of `E₁` and `E₂`. -/
def Prod.toFun' : TotalSpace (F₁ × F₂) (E₁ ×ᵇ E₂) → B × F₁ × F₂ :=
  fun p ↦ ⟨p.1, (e₁ ⟨p.1, p.2.1⟩).2, (e₂ ⟨p.1, p.2.2⟩).2⟩


theorem Prod.continuous_to_fun : ContinuousOn (Prod.toFun' e₁ e₂)
    (π (F₁ × F₂) (E₁ ×ᵇ E₂) ⁻¹' (e₁.baseSet ∩ e₂.baseSet)) := by
  let f₁ : TotalSpace (F₁ × F₂) (E₁ ×ᵇ E₂) → TotalSpace F₁ E₁ × TotalSpace F₂ E₂ :=
    fun p ↦ ((⟨p.1, p.2.1⟩ : TotalSpace F₁ E₁), (⟨p.1, p.2.2⟩ : TotalSpace F₂ E₂))
  /-
    B : Type u_1
    inst✝⁴ : TopologicalSpace B
    F₁ : Type u_2
    inst✝³ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝¹ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
    ⊢ ContinuousOn (Trivialization.Prod.toFun' e₁ e₂) (Set.preimage Bundle.TotalSp …
  -/
  let f₂ : TotalSpace F₁ E₁ × TotalSpace F₂ E₂ → (B × F₁) × B × F₂ := fun p ↦ ⟨e₁ p.1, e₂ p.2⟩
  /-
    B : Type u_1
    inst✝⁴ : TopologicalSpace B
    F₁ : Type u_2
    inst✝³ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝¹ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
    f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
    ⊢ ContinuousOn (Trivialization.Prod.toFun' e₁ e₂) (Set.preimage Bundle.TotalSp …
  -/
  let f₃ : (B × F₁) × B × F₂ → B × F₁ × F₂ := fun p ↦ ⟨p.1.1, p.1.2, p.2.2⟩
  /-
    B : Type u_1
    inst✝⁴ : TopologicalSpace B
    F₁ : Type u_2
    inst✝³ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝¹ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
    f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
    f₃ : Prod (Prod B F₁) (Prod B F₂) → Prod B (Prod F₁ F₂) := fun p => { fst := p …
    ⊢ ContinuousOn (Trivialization.Prod.toFun' e₁ e₂) (Set.preimage Bundle.TotalSp …
  -/
  have hf₁ : Continuous f₁ := (Prod.isInducing_diag F₁ E₁ F₂ E₂).continuous
  have hf₂ : ContinuousOn f₂ (e₁.source ×ˢ e₂.source) :=
    e₁.toPartialHomeomorph.continuousOn.prod_map e₂.toPartialHomeomorph.continuousOn
  /-
    B : Type u_1
    inst✝⁴ : TopologicalSpace B
    F₁ : Type u_2
    inst✝³ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝¹ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
    f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
    f₃ : Prod (Prod B F₁) (Prod B F₂) → Prod B (Prod F₁ F₂) := fun p => { fst := p …
    hf₁ : Continuous f₁
    hf₂ : ContinuousOn f₂ (SProd.sprod e₁.source e₂.source)
    ⊢ ContinuousOn (Trivialization.Prod.toFun' e₁ e₂) (Set.preimage Bundle.TotalSp …
  -/
  have hf₃ : Continuous f₃ := by fun_prop
  /-
    B : Type u_1
    inst✝⁴ : TopologicalSpace B
    F₁ : Type u_2
    inst✝³ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝¹ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
    f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
    f₃ : Prod (Prod B F₁) (Prod B F₂) → Prod B (Prod F₁ F₂) := fun p => { fst := p …
    hf₁ : Continuous f₁
    hf₂ : ContinuousOn f₂ (SProd.sprod e₁.source e₂.source)
    hf₃ : Continuous f₃
    ⊢ ContinuousOn (Trivialization.Prod.toFun' e₁ e₂) (Set.preimage Bundle.TotalSp …
  -/
  refine ((hf₃.comp_continuousOn hf₂).comp hf₁.continuousOn ?_).congr ?_
    /-
      case refine_1
      B : Type u_1
      inst✝⁴ : TopologicalSpace B
      F₁ : Type u_2
      inst✝³ : TopologicalSpace F₁
      E₁ : B → Type u_3
      inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_4
      inst✝¹ : TopologicalSpace F₂
      E₂ : B → Type u_5
      inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      e₁ : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ : Trivialization F₂ Bundle.TotalSpace.proj
      f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
      f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
      f₃ : Prod (Prod B F₁) (Prod B F₂) → Prod B (Prod F₁ F₂) := fun p => { fst := p …
      hf₁ : Continuous f₁
      hf₂ : ContinuousOn f₂ (SProd.sprod e₁.source e₂.source)
      hf₃ : Continuous f₃
      ⊢ Set.MapsTo f₁ (Set.preimage Bundle.TotalSpace.proj (Inter.inter e₁.baseSet e …
    -/
  · rw [e₁.source_eq, e₂.source_eq]
    /-
      case refine_1
      B : Type u_1
      inst✝⁴ : TopologicalSpace B
      F₁ : Type u_2
      inst✝³ : TopologicalSpace F₁
      E₁ : B → Type u_3
      inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_4
      inst✝¹ : TopologicalSpace F₂
      E₂ : B → Type u_5
      inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      e₁ : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ : Trivialization F₂ Bundle.TotalSpace.proj
      f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
      f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
      f₃ : Prod (Prod B F₁) (Prod B F₂) → Prod B (Prod F₁ F₂) := fun p => { fst := p …
      hf₁ : Continuous f₁
      hf₂ : ContinuousOn f₂ (SProd.sprod e₁.source e₂.source)
      hf₃ : Continuous f₃
      ⊢ Set.MapsTo f₁ (Set.preimage Bundle.TotalSpace.proj (Inter.inter e₁.baseSet e …
    -/
    exact mapsTo_preimage _ _
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    B : Type u_1
    inst✝⁴ : TopologicalSpace B
    F₁ : Type u_2
    inst✝³ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝¹ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
    f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
    f₃ : Prod (Prod B F₁) (Prod B F₂) → Prod B (Prod F₁ F₂) := fun p => { fst := p …
    hf₁ : Continuous f₁
    hf₂ : ContinuousOn f₂ (SProd.sprod e₁.source e₂.source)
    hf₃ : Continuous f₃
    ⊢ Set.EqOn (Trivialization.Prod.toFun' e₁ e₂) (Function.comp (Function.comp f₃ …
  -/
  rintro ⟨b, v₁, v₂⟩ ⟨hb₁, _⟩
  /-
    case refine_2.mk.mk.intro
    B : Type u_1
    inst✝⁴ : TopologicalSpace B
    F₁ : Type u_2
    inst✝³ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝¹ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
    f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
    f₃ : Prod (Prod B F₁) (Prod B F₂) → Prod B (Prod F₁ F₂) := fun p => { fst := p …
    hf₁ : Continuous f₁
    hf₂ : ContinuousOn f₂ (SProd.sprod e₁.source e₂.source)
    hf₃ : Continuous f₃
    b : B
    v₁ : E₁ b
    v₂ : E₂ b
    hb₁ : Membership.mem e₁.baseSet { proj := b, snd := { fst := v₁, snd := v₂ } } …
    right✝ : Membership.mem e₂.baseSet { proj := b, snd := { fst := v₁, snd := v₂  …
    ⊢ Eq (Trivialization.Prod.toFun' e₁ e₂ { proj := b, snd := { fst := v₁, snd := …
  -/
  simp only [f₁, f₂, f₃, Prod.toFun', Prod.mk.inj_iff, Function.comp_apply, and_true]
  /-
    case refine_2.mk.mk.intro
    B : Type u_1
    inst✝⁴ : TopologicalSpace B
    F₁ : Type u_2
    inst✝³ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝¹ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
    f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
    f₃ : Prod (Prod B F₁) (Prod B F₂) → Prod B (Prod F₁ F₂) := fun p => { fst := p …
    hf₁ : Continuous f₁
    hf₂ : ContinuousOn f₂ (SProd.sprod e₁.source e₂.source)
    hf₃ : Continuous f₃
    b : B
    v₁ : E₁ b
    v₂ : E₂ b
    hb₁ : Membership.mem e₁.baseSet { proj := b, snd := { fst := v₁, snd := v₂ } } …
    right✝ : Membership.mem e₂.baseSet { proj := b, snd := { fst := v₁, snd := v₂  …
    ⊢ Eq b (↑e₁ { proj := b, snd := v₁ }).1
  -/
  rw [e₁.coe_fst]
  /-
    case refine_2.mk.mk.intro
    B : Type u_1
    inst✝⁴ : TopologicalSpace B
    F₁ : Type u_2
    inst✝³ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝¹ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
    f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
    f₃ : Prod (Prod B F₁) (Prod B F₂) → Prod B (Prod F₁ F₂) := fun p => { fst := p …
    hf₁ : Continuous f₁
    hf₂ : ContinuousOn f₂ (SProd.sprod e₁.source e₂.source)
    hf₃ : Continuous f₃
    b : B
    v₁ : E₁ b
    v₂ : E₂ b
    hb₁ : Membership.mem e₁.baseSet { proj := b, snd := { fst := v₁, snd := v₂ } } …
    right✝ : Membership.mem e₂.baseSet { proj := b, snd := { fst := v₁, snd := v₂  …
    ⊢ Membership.mem e₁.source { proj := b, snd := v₁ }
  -/
  rw [e₁.source_eq, mem_preimage]
  /-
    case refine_2.mk.mk.intro
    B : Type u_1
    inst✝⁴ : TopologicalSpace B
    F₁ : Type u_2
    inst✝³ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝¹ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    f₁ : (Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)) → Prod (Bund …
    f₂ : Prod (Bundle.TotalSpace F₁ E₁) (Bundle.TotalSpace F₂ E₂) → Prod (Prod B F …
    f₃ : Prod (Prod B F₁) (Prod B F₂) → Prod B (Prod F₁ F₂) := fun p => { fst := p …
    hf₁ : Continuous f₁
    hf₂ : ContinuousOn f₂ (SProd.sprod e₁.source e₂.source)
    hf₃ : Continuous f₃
    b : B
    v₁ : E₁ b
    v₂ : E₂ b
    hb₁ : Membership.mem e₁.baseSet { proj := b, snd := { fst := v₁, snd := v₂ } } …
    right✝ : Membership.mem e₂.baseSet { proj := b, snd := { fst := v₁, snd := v₂  …
    ⊢ Membership.mem e₁.baseSet { proj := b, snd := v₁ }.proj
  -/
  exact hb₁
  /-
    🎉 no goals
  -/


/-- Given trivializations `e₁`, `e₂` for fiber bundles `E₁`, `E₂` over a base `B`, the inverse
function for the construction `Trivialization.prod`, the induced
trivialization for the fiberwise product of `E₁` and `E₂`. -/
noncomputable def Prod.invFun' (p : B × F₁ × F₂) : TotalSpace (F₁ × F₂) (E₁ ×ᵇ E₂) :=
  ⟨p.1, e₁.symm p.1 p.2.1, e₂.symm p.1 p.2.2⟩


theorem Prod.left_inv {x : TotalSpace (F₁ × F₂) (E₁ ×ᵇ E₂)}
    (h : x ∈ π (F₁ × F₂) (E₁ ×ᵇ E₂) ⁻¹' (e₁.baseSet ∩ e₂.baseSet)) :
    Prod.invFun' e₁ e₂ (Prod.toFun' e₁ e₂ x) = x := by
  /-
    B : Type u_1
    inst✝⁶ : TopologicalSpace B
    F₁ : Type u_2
    inst✝⁵ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝³ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : (x : B) → Zero (E₁ x)
    inst✝ : (x : B) → Zero (E₂ x)
    x : Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)
    h : Membership.mem (Set.preimage Bundle.TotalSpace.proj (Inter.inter e₁.baseSe …
    ⊢ Eq (Trivialization.Prod.invFun' e₁ e₂ (Trivialization.Prod.toFun' e₁ e₂ x)) x
  -/
  obtain ⟨x, v₁, v₂⟩ := x
  /-
    case mk.mk
    B : Type u_1
    inst✝⁶ : TopologicalSpace B
    F₁ : Type u_2
    inst✝⁵ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝³ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : (x : B) → Zero (E₁ x)
    inst✝ : (x : B) → Zero (E₂ x)
    x : B
    v₁ : E₁ x
    v₂ : E₂ x
    h : Membership.mem (Set.preimage Bundle.TotalSpace.proj (Inter.inter e₁.baseSe …
    ⊢ Eq (Trivialization.Prod.invFun' e₁ e₂ (Trivialization.Prod.toFun' e₁ e₂ { pr …
  -/
  obtain ⟨h₁ : x ∈ e₁.baseSet, h₂ : x ∈ e₂.baseSet⟩ := h
  /-
    case mk.mk.intro
    B : Type u_1
    inst✝⁶ : TopologicalSpace B
    F₁ : Type u_2
    inst✝⁵ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝³ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : (x : B) → Zero (E₁ x)
    inst✝ : (x : B) → Zero (E₂ x)
    x : B
    v₁ : E₁ x
    v₂ : E₂ x
    h₁ : Membership.mem e₁.baseSet x
    h₂ : Membership.mem e₂.baseSet x
    ⊢ Eq (Trivialization.Prod.invFun' e₁ e₂ (Trivialization.Prod.toFun' e₁ e₂ { pr …
  -/
  simp only [Prod.toFun', Prod.invFun', symm_apply_apply_mk, h₁, h₂]
  /-
    🎉 no goals
  -/


theorem Prod.right_inv {x : B × F₁ × F₂}
    (h : x ∈ (e₁.baseSet ∩ e₂.baseSet) ×ˢ (univ : Set (F₁ × F₂))) :
    Prod.toFun' e₁ e₂ (Prod.invFun' e₁ e₂ x) = x := by
  /-
    B : Type u_1
    inst✝⁶ : TopologicalSpace B
    F₁ : Type u_2
    inst✝⁵ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝³ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : (x : B) → Zero (E₁ x)
    inst✝ : (x : B) → Zero (E₂ x)
    x : Prod B (Prod F₁ F₂)
    h : Membership.mem (SProd.sprod (Inter.inter e₁.baseSet e₂.baseSet) Set.univ) x
    ⊢ Eq (Trivialization.Prod.toFun' e₁ e₂ (Trivialization.Prod.invFun' e₁ e₂ x)) x
  -/
  obtain ⟨x, w₁, w₂⟩ := x
  /-
    case mk.mk
    B : Type u_1
    inst✝⁶ : TopologicalSpace B
    F₁ : Type u_2
    inst✝⁵ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝³ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : (x : B) → Zero (E₁ x)
    inst✝ : (x : B) → Zero (E₂ x)
    x : B
    w₁ : F₁
    w₂ : F₂
    h : Membership.mem (SProd.sprod (Inter.inter e₁.baseSet e₂.baseSet) Set.univ)  …
    ⊢ Eq (Trivialization.Prod.toFun' e₁ e₂ (Trivialization.Prod.invFun' e₁ e₂ { fs …
  -/
  obtain ⟨⟨h₁ : x ∈ e₁.baseSet, h₂ : x ∈ e₂.baseSet⟩, -⟩ := h
  /-
    case mk.mk.intro.intro
    B : Type u_1
    inst✝⁶ : TopologicalSpace B
    F₁ : Type u_2
    inst✝⁵ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝³ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : (x : B) → Zero (E₁ x)
    inst✝ : (x : B) → Zero (E₂ x)
    x : B
    w₁ : F₁
    w₂ : F₂
    h₁ : Membership.mem e₁.baseSet x
    h₂ : Membership.mem e₂.baseSet x
    ⊢ Eq (Trivialization.Prod.toFun' e₁ e₂ (Trivialization.Prod.invFun' e₁ e₂ { fs …
  -/
  simp only [Prod.toFun', Prod.invFun', apply_mk_symm, h₁, h₂]
  /-
    🎉 no goals
  -/


theorem Prod.continuous_inv_fun :
    ContinuousOn (Prod.invFun' e₁ e₂) ((e₁.baseSet ∩ e₂.baseSet) ×ˢ univ) := by
  /-
    B : Type u_1
    inst✝⁶ : TopologicalSpace B
    F₁ : Type u_2
    inst✝⁵ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝³ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : (x : B) → Zero (E₁ x)
    inst✝ : (x : B) → Zero (E₂ x)
    ⊢ ContinuousOn (Trivialization.Prod.invFun' e₁ e₂) (SProd.sprod (Inter.inter e …
  -/
  rw [(Prod.isInducing_diag F₁ E₁ F₂ E₂).continuousOn_iff]
  /-
    B : Type u_1
    inst✝⁶ : TopologicalSpace B
    F₁ : Type u_2
    inst✝⁵ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝³ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : (x : B) → Zero (E₁ x)
    inst✝ : (x : B) → Zero (E₂ x)
    ⊢ ContinuousOn (Function.comp (fun p => { fst := { proj := p.proj, snd := p.sn …
  -/
  have H₁ : Continuous fun p : B × F₁ × F₂ ↦ ((p.1, p.2.1), (p.1, p.2.2)) := by fun_prop
  /-
    B : Type u_1
    inst✝⁶ : TopologicalSpace B
    F₁ : Type u_2
    inst✝⁵ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝³ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : (x : B) → Zero (E₁ x)
    inst✝ : (x : B) → Zero (E₂ x)
    H₁ : Continuous fun p => { fst := { fst := p.1, snd := p.2.1 }, snd := { fst : …
    ⊢ ContinuousOn (Function.comp (fun p => { fst := { proj := p.proj, snd := p.sn …
  -/
  refine (e₁.continuousOn_symm.prod_map e₂.continuousOn_symm).comp H₁.continuousOn ?_
  /-
    B : Type u_1
    inst✝⁶ : TopologicalSpace B
    F₁ : Type u_2
    inst✝⁵ : TopologicalSpace F₁
    E₁ : B → Type u_3
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_4
    inst✝³ : TopologicalSpace F₂
    E₂ : B → Type u_5
    inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : (x : B) → Zero (E₁ x)
    inst✝ : (x : B) → Zero (E₂ x)
    H₁ : Continuous fun p => { fst := { fst := p.1, snd := p.2.1 }, snd := { fst : …
    ⊢ Set.MapsTo (fun p => { fst := { fst := p.1, snd := p.2.1 }, snd := { fst :=  …
  -/
  exact fun x h ↦ ⟨⟨h.1.1, mem_univ _⟩, ⟨h.1.2, mem_univ _⟩⟩
  /-
    🎉 no goals
  -/


/-- Given trivializations `e₁`, `e₂` for bundle types `E₁`, `E₂` over a base `B`, the induced
trivialization for the fiberwise product of `E₁` and `E₂`, whose base set is
`e₁.baseSet ∩ e₂.baseSet`. -/
noncomputable def prod : Trivialization (F₁ × F₂) (π (F₁ × F₂) (E₁ ×ᵇ E₂)) where
  toFun := Prod.toFun' e₁ e₂
  invFun := Prod.invFun' e₁ e₂
  source := π (F₁ × F₂) (E₁ ×ᵇ E₂) ⁻¹' (e₁.baseSet ∩ e₂.baseSet)
  target := (e₁.baseSet ∩ e₂.baseSet) ×ˢ Set.univ
  map_source' _ h := ⟨h, Set.mem_univ _⟩
  map_target' _ h := h.1
  left_inv' _ := Prod.left_inv
  right_inv' _ := Prod.right_inv
  open_source := by
    convert (e₁.open_source.prod e₂.open_source).preimage
        (FiberBundle.Prod.isInducing_diag F₁ E₁ F₂ E₂).continuous
    /-
      case h.e'_3
      B : Type u_1
      inst✝⁶ : TopologicalSpace B
      F₁ : Type u_2
      inst✝⁵ : TopologicalSpace F₁
      E₁ : B → Type u_3
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_4
      inst✝³ : TopologicalSpace F₂
      E₂ : B → Type u_5
      inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      e₁ : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝¹ : (x : B) → Zero (E₁ x)
      inst✝ : (x : B) → Zero (E₂ x)
      ⊢ Eq { toFun := Trivialization.Prod.toFun' e₁ e₂, invFun := Trivialization.Pro …
    -/
    ext x
    /-
      case h.e'_3.h
      B : Type u_1
      inst✝⁶ : TopologicalSpace B
      F₁ : Type u_2
      inst✝⁵ : TopologicalSpace F₁
      E₁ : B → Type u_3
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_4
      inst✝³ : TopologicalSpace F₂
      E₂ : B → Type u_5
      inst✝² : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      e₁ : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ : Trivialization F₂ Bundle.TotalSpace.proj
      inst✝¹ : (x : B) → Zero (E₁ x)
      inst✝ : (x : B) → Zero (E₂ x)
      x : Bundle.TotalSpace (Prod F₁ F₂) fun x => Prod (E₁ x) (E₂ x)
      ⊢ Iff (Membership.mem { toFun := Trivialization.Prod.toFun' e₁ e₂, invFun := T …
    -/
    simp only [Trivialization.source_eq, mfld_simps]
    /-
      🎉 no goals
    -/
  open_target := (e₁.open_baseSet.inter e₂.open_baseSet).prod isOpen_univ
  continuousOn_toFun := Prod.continuous_to_fun
  continuousOn_invFun := Prod.continuous_inv_fun
  baseSet := e₁.baseSet ∩ e₂.baseSet
  open_baseSet := e₁.open_baseSet.inter e₂.open_baseSet
  source_eq := rfl
  target_eq := rfl
  proj_toFun _ _ := rfl


@[simp]
theorem baseSet_prod : (prod e₁ e₂).baseSet = e₁.baseSet ∩ e₂.baseSet := rfl


theorem prod_symm_apply (x : B) (w₁ : F₁) (w₂ : F₂) :
    (prod e₁ e₂).toPartialEquiv.symm (x, w₁, w₂) = ⟨x, e₁.symm x w₁, e₂.symm x w₂⟩ := rfl


/-- The product of two fiber bundles is a fiber bundle. -/
noncomputable instance FiberBundle.prod : FiberBundle (F₁ × F₂) (E₁ ×ᵇ E₂) where
  totalSpaceMk_isInducing' b := by
    /-
      B : Type u_1
      inst✝¹⁰ : TopologicalSpace B
      F₁ : Type u_2
      inst✝⁹ : TopologicalSpace F₁
      E₁ : B → Type u_3
      inst✝⁸ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_4
      inst✝⁷ : TopologicalSpace F₂
      E₂ : B → Type u_5
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁵ : (x : B) → Zero (E₁ x)
      inst✝⁴ : (x : B) → Zero (E₂ x)
      inst✝³ : (x : B) → TopologicalSpace (E₁ x)
      inst✝² : (x : B) → TopologicalSpace (E₂ x)
      inst✝¹ : FiberBundle F₁ E₁
      inst✝ : FiberBundle F₂ E₂
      b : B
      ⊢ Topology.IsInducing (Bundle.TotalSpace.mk b)
    -/
    rw [← (Prod.isInducing_diag F₁ E₁ F₂ E₂).of_comp_iff]
    /-
      B : Type u_1
      inst✝¹⁰ : TopologicalSpace B
      F₁ : Type u_2
      inst✝⁹ : TopologicalSpace F₁
      E₁ : B → Type u_3
      inst✝⁸ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_4
      inst✝⁷ : TopologicalSpace F₂
      E₂ : B → Type u_5
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁵ : (x : B) → Zero (E₁ x)
      inst✝⁴ : (x : B) → Zero (E₂ x)
      inst✝³ : (x : B) → TopologicalSpace (E₁ x)
      inst✝² : (x : B) → TopologicalSpace (E₂ x)
      inst✝¹ : FiberBundle F₁ E₁
      inst✝ : FiberBundle F₂ E₂
      b : B
      ⊢ Topology.IsInducing (Function.comp (fun p => { fst := { proj := p.proj, snd  …
    -/
    exact (totalSpaceMk_isInducing F₁ E₁ b).prodMap (totalSpaceMk_isInducing F₂ E₂ b)
    /-
      🎉 no goals
    -/
  trivializationAtlas' := { e |
    ∃ (e₁ : Trivialization F₁ (π F₁ E₁)) (e₂ : Trivialization F₂ (π F₂ E₂))
      (_ : MemTrivializationAtlas e₁) (_ : MemTrivializationAtlas e₂),
      e = Trivialization.prod e₁ e₂ }
  trivializationAt' b := (trivializationAt F₁ E₁ b).prod (trivializationAt F₂ E₂ b)
  mem_baseSet_trivializationAt' b :=
    ⟨mem_baseSet_trivializationAt F₁ E₁ b, mem_baseSet_trivializationAt F₂ E₂ b⟩
  trivialization_mem_atlas' b :=
    ⟨trivializationAt F₁ E₁ b, trivializationAt F₂ E₂ b, inferInstance, inferInstance, rfl⟩


instance {e₁ : Trivialization F₁ (π F₁ E₁)} {e₂ : Trivialization F₂ (π F₂ E₂)}
    [MemTrivializationAtlas e₁] [MemTrivializationAtlas e₂] :
    MemTrivializationAtlas (e₁.prod e₂ : Trivialization (F₁ × F₂) (π (F₁ × F₂) (E₁ ×ᵇ E₂))) where
  out := ⟨e₁, e₂, inferInstance, inferInstance, rfl⟩


instance [∀ x : B, TopologicalSpace (E x)] : ∀ x : B', TopologicalSpace ((f *ᵖ E) x) :=
  inferInstanceAs (∀ x, TopologicalSpace (E (f x)))


/-- Definition of `Pullback.TotalSpace.topologicalSpace`, which we make irreducible. -/
irreducible_def pullbackTopology : TopologicalSpace (TotalSpace F (f *ᵖ E)) :=
  induced TotalSpace.proj ‹TopologicalSpace B'› ⊓
    induced (Pullback.lift f) ‹TopologicalSpace (TotalSpace F E)›


/-- The topology on the total space of a pullback bundle is the coarsest topology for which both
the projections to the base and the map to the original bundle are continuous. -/
instance Pullback.TotalSpace.topologicalSpace : TopologicalSpace (TotalSpace F (f *ᵖ E)) :=
  pullbackTopology F E f


theorem Pullback.continuous_proj (f : B' → B) : Continuous (π F (f *ᵖ E)) := by
  /-
    B : Type u
    F : Type v
    E : B → Type w₁
    B' : Type w₂
    inst✝¹ : TopologicalSpace B'
    inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
    f : B' → B
    ⊢ Continuous Bundle.TotalSpace.proj
  -/
  rw [continuous_iff_le_induced, Pullback.TotalSpace.topologicalSpace, pullbackTopology_def]
  /-
    B : Type u
    F : Type v
    E : B → Type w₁
    B' : Type w₂
    inst✝¹ : TopologicalSpace B'
    inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
    f : B' → B
    ⊢ LE.le (Min.min (TopologicalSpace.induced Bundle.TotalSpace.proj inst✝¹) (Top …
  -/
  exact inf_le_left
  /-
    🎉 no goals
  -/


theorem Pullback.continuous_lift (f : B' → B) : Continuous (@Pullback.lift B F E B' f) := by
  /-
    B : Type u
    F : Type v
    E : B → Type w₁
    B' : Type w₂
    inst✝¹ : TopologicalSpace B'
    inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
    f : B' → B
    ⊢ Continuous (Bundle.Pullback.lift f)
  -/
  rw [continuous_iff_le_induced, Pullback.TotalSpace.topologicalSpace, pullbackTopology_def]
  /-
    B : Type u
    F : Type v
    E : B → Type w₁
    B' : Type w₂
    inst✝¹ : TopologicalSpace B'
    inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
    f : B' → B
    ⊢ LE.le (Min.min (TopologicalSpace.induced Bundle.TotalSpace.proj inst✝¹) (Top …
  -/
  exact inf_le_right
  /-
    🎉 no goals
  -/


theorem inducing_pullbackTotalSpaceEmbedding (f : B' → B) :
    IsInducing (@pullbackTotalSpaceEmbedding B F E B' f) := by
  /-
    B : Type u
    F : Type v
    E : B → Type w₁
    B' : Type w₂
    inst✝¹ : TopologicalSpace B'
    inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
    f : B' → B
    ⊢ Topology.IsInducing (Bundle.pullbackTotalSpaceEmbedding f)
  -/
  constructor
  simp_rw [instTopologicalSpaceProd, induced_inf, induced_compose,
    Pullback.TotalSpace.topologicalSpace, pullbackTopology_def]
  /-
    case eq_induced
    B : Type u
    F : Type v
    E : B → Type w₁
    B' : Type w₂
    inst✝¹ : TopologicalSpace B'
    inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
    f : B' → B
    ⊢ Eq (Min.min (TopologicalSpace.induced Bundle.TotalSpace.proj inst✝¹) (Topolo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Pullback.continuous_totalSpaceMk [∀ x, TopologicalSpace (E x)] [FiberBundle F E]
    {f : B' → B} {x : B'} : Continuous (@TotalSpace.mk _ F (f *ᵖ E) x) := by
  simp only [continuous_iff_le_induced, Pullback.TotalSpace.topologicalSpace, induced_compose,
    induced_inf, Function.comp_def, induced_const, top_inf_eq, pullbackTopology_def]
  /-
    B : Type u
    F : Type v
    E : B → Type w₁
    B' : Type w₂
    inst✝⁵ : TopologicalSpace B'
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : TopologicalSpace F
    inst✝² : TopologicalSpace B
    inst✝¹ : (x : B) → TopologicalSpace (E x)
    inst✝ : FiberBundle F E
    f : B' → B
    x : B'
    ⊢ LE.le (instTopologicalSpacePullback E f x) (TopologicalSpace.induced (fun x_ …
  -/
  exact (FiberBundle.totalSpaceMk_isInducing F E (f x)).eq_induced.le
  /-
    🎉 no goals
  -/


/-- A fiber bundle trivialization can be pulled back to a trivialization on the pullback bundle. -/
noncomputable def Trivialization.pullback (e : Trivialization F (π F E)) (f : K) :
    Trivialization F (π F ((f : B' → B) *ᵖ E)) where
  toFun z := (z.proj, (e (Pullback.lift f z)).2)
  invFun y := @TotalSpace.mk _ F (f *ᵖ E) y.1 (e.symm (f y.1) y.2)
  source := Pullback.lift f ⁻¹' e.source
  baseSet := f ⁻¹' e.baseSet
  target := (f ⁻¹' e.baseSet) ×ˢ univ
  map_source' x h := by
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      x : Bundle.TotalSpace F (Bundle.Pullback (⇑f) E)
      h : Membership.mem (Set.preimage (Bundle.Pullback.lift ⇑f) e.source) x
      ⊢ Membership.mem (SProd.sprod (Set.preimage (⇑f) e.baseSet) Set.univ) ((fun z  …
    -/
    simp_rw [e.source_eq, mem_preimage, Pullback.lift_proj] at h
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      x : Bundle.TotalSpace F (Bundle.Pullback (⇑f) E)
      h : Membership.mem e.baseSet (f x.proj)
      ⊢ Membership.mem (SProd.sprod (Set.preimage (⇑f) e.baseSet) Set.univ) ((fun z  …
    -/
    simp_rw [prod_mk_mem_set_prod_eq, mem_univ, and_true, mem_preimage, h]
    /-
      🎉 no goals
    -/
  map_target' y h := by
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      y : Prod B' F
      h : Membership.mem (SProd.sprod (Set.preimage (⇑f) e.baseSet) Set.univ) y
      ⊢ Membership.mem (Set.preimage (Bundle.Pullback.lift ⇑f) e.source) ((fun y =>  …
    -/
    rw [mem_prod, mem_preimage] at h
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      y : Prod B' F
      h : And (Membership.mem e.baseSet (f y.1)) (Membership.mem Set.univ y.2)
      ⊢ Membership.mem (Set.preimage (Bundle.Pullback.lift ⇑f) e.source) ((fun y =>  …
    -/
    simp_rw [e.source_eq, mem_preimage, Pullback.lift_proj, h.1]
    /-
      🎉 no goals
    -/
  left_inv' x h := by
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      x : Bundle.TotalSpace F (Bundle.Pullback (⇑f) E)
      h : Membership.mem (Set.preimage (Bundle.Pullback.lift ⇑f) e.source) x
      ⊢ Eq ((fun y => { proj := y.1, snd := e.symm (f y.1) y.2 }) ((fun z => { fst : …
    -/
    simp_rw [mem_preimage, e.mem_source, Pullback.lift_proj] at h
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      x : Bundle.TotalSpace F (Bundle.Pullback (⇑f) E)
      h : Membership.mem e.baseSet (f x.proj)
      ⊢ Eq ((fun y => { proj := y.1, snd := e.symm (f y.1) y.2 }) ((fun z => { fst : …
    -/
    simp_rw [Pullback.lift, e.symm_apply_apply_mk h]
    /-
      🎉 no goals
    -/
  right_inv' x h := by
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      x : Prod B' F
      h : Membership.mem (SProd.sprod (Set.preimage (⇑f) e.baseSet) Set.univ) x
      ⊢ Eq ((fun z => { fst := z.proj, snd := (↑e (Bundle.Pullback.lift (⇑f) z)).2 } …
    -/
    simp_rw [mem_prod, mem_preimage, mem_univ, and_true] at h
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      x : Prod B' F
      h : Membership.mem e.baseSet (f x.1)
      ⊢ Eq ((fun z => { fst := z.proj, snd := (↑e (Bundle.Pullback.lift (⇑f) z)).2 } …
    -/
    simp_rw [Pullback.lift_mk, e.apply_mk_symm h]
    /-
      🎉 no goals
    -/
  open_source := by
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      ⊢ IsOpen { toFun := fun z => { fst := z.proj, snd := (↑e (Bundle.Pullback.lift …
    -/
    simp_rw [e.source_eq, ← preimage_comp]
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      ⊢ IsOpen (Set.preimage (Function.comp Bundle.TotalSpace.proj (Bundle.Pullback. …
    -/
    exact e.open_baseSet.preimage ((map_continuous f).comp <| Pullback.continuous_proj F E f)
    /-
      🎉 no goals
    -/
  open_target := ((map_continuous f).isOpen_preimage _ e.open_baseSet).prod isOpen_univ
  open_baseSet := (map_continuous f).isOpen_preimage _ e.open_baseSet
  continuousOn_toFun :=
    (Pullback.continuous_proj F E f).continuousOn.prod
      (continuous_snd.comp_continuousOn <|
        e.continuousOn.comp (Pullback.continuous_lift F E f).continuousOn Subset.rfl)
  continuousOn_invFun := by
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      ⊢ ContinuousOn { toFun := fun z => { fst := z.proj, snd := (↑e (Bundle.Pullbac …
    -/
    dsimp only
    simp_rw [(inducing_pullbackTotalSpaceEmbedding F E f).continuousOn_iff, Function.comp_def,
      pullbackTotalSpaceEmbedding]
    refine
      continuousOn_fst.prod
        (e.continuousOn_symm.comp ((map_continuous f).prodMap continuous_id).continuousOn
          Subset.rfl)
  source_eq := by
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      ⊢ Eq { toFun := fun z => { fst := z.proj, snd := (↑e (Bundle.Pullback.lift (⇑f …
    -/
    dsimp only
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      ⊢ Eq (Set.preimage (Bundle.Pullback.lift ⇑f) e.source) (Set.preimage Bundle.To …
    -/
    rw [e.source_eq]
    /-
      B : Type u
      F : Type v
      E : B → Type w₁
      B' : Type w₂
      f✝ : B' → B
      inst✝⁶ : TopologicalSpace B'
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalSpace B
      inst✝² : (_b : B) → Zero (E _b)
      K : Type U
      inst✝¹ : FunLike K B' B
      inst✝ : ContinuousMapClass K B' B
      e : Trivialization F Bundle.TotalSpace.proj
      f : K
      ⊢ Eq (Set.preimage (Bundle.Pullback.lift ⇑f) (Set.preimage Bundle.TotalSpace.p …
    -/
    rfl
    /-
      🎉 no goals
    -/
  target_eq := rfl
  proj_toFun _ _ := rfl


noncomputable instance FiberBundle.pullback [∀ x, TopologicalSpace (E x)] [FiberBundle F E]
    (f : K) : FiberBundle F ((f : B' → B) *ᵖ E) where
  totalSpaceMk_isInducing' x :=
    (totalSpaceMk_isInducing F E (f x)).of_comp (Pullback.continuous_totalSpaceMk F E)
      (Pullback.continuous_lift F E f)
  trivializationAtlas' :=
    { ef | ∃ (e : Trivialization F (π F E)) (_ : MemTrivializationAtlas e), ef = e.pullback f }
  trivializationAt' x := (trivializationAt F E (f x)).pullback f
  mem_baseSet_trivializationAt' x := mem_baseSet_trivializationAt F E (f x)
  trivialization_mem_atlas' x := ⟨trivializationAt F E (f x), inferInstance, rfl⟩


