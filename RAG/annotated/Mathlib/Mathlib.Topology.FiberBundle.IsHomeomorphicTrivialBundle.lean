/-- A trivial fiber bundle with fiber `F` over a base `B` is a space `Z`
projecting on `B` for which there exists a homeomorphism to `B × F` that sends `proj`
to `Prod.fst`. -/
def IsHomeomorphicTrivialFiberBundle (proj : Z → B) : Prop :=
  ∃ e : Z ≃ₜ B × F, ∀ x, (e x).1 = proj x


protected theorem proj_eq (h : IsHomeomorphicTrivialFiberBundle F proj) :
    ∃ e : Z ≃ₜ B × F, proj = Prod.fst ∘ e :=
  ⟨h.choose, (funext h.choose_spec).symm⟩


/-- The projection from a trivial fiber bundle to its base is surjective. -/
protected theorem surjective_proj [Nonempty F] (h : IsHomeomorphicTrivialFiberBundle F proj) :
    Function.Surjective proj := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_3
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalSpace Z
    proj : Z → B
    inst✝ : Nonempty F
    h : IsHomeomorphicTrivialFiberBundle F proj
    ⊢ Function.Surjective proj
  -/
  obtain ⟨e, rfl⟩ := h.proj_eq
  /-
    case intro
    B : Type u_1
    F : Type u_2
    Z : Type u_3
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalSpace Z
    inst✝ : Nonempty F
    e : Homeomorph Z (Prod B F)
    h : IsHomeomorphicTrivialFiberBundle F (Function.comp Prod.fst ⇑e)
    ⊢ Function.Surjective (Function.comp Prod.fst ⇑e)
  -/
  exact Prod.fst_surjective.comp e.surjective
  /-
    🎉 no goals
  -/


/-- The projection from a trivial fiber bundle to its base is continuous. -/
protected theorem continuous_proj (h : IsHomeomorphicTrivialFiberBundle F proj) :
    Continuous proj := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalSpace Z
    proj : Z → B
    h : IsHomeomorphicTrivialFiberBundle F proj
    ⊢ Continuous proj
  -/
  obtain ⟨e, rfl⟩ := h.proj_eq; exact continuous_fst.comp e.continuous
                                /-
                                  🎉 no goals
                                -/


/-- The projection from a trivial fiber bundle to its base is open. -/
protected theorem isOpenMap_proj (h : IsHomeomorphicTrivialFiberBundle F proj) :
    IsOpenMap proj := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalSpace Z
    proj : Z → B
    h : IsHomeomorphicTrivialFiberBundle F proj
    ⊢ IsOpenMap proj
  -/
  obtain ⟨e, rfl⟩ := h.proj_eq; exact isOpenMap_fst.comp e.isOpenMap
                                /-
                                  🎉 no goals
                                -/


/-- The projection from a trivial fiber bundle to its base is open. -/
protected theorem isQuotientMap_proj [Nonempty F] (h : IsHomeomorphicTrivialFiberBundle F proj) :
    IsQuotientMap proj :=
  h.isOpenMap_proj.isQuotientMap h.continuous_proj h.surjective_proj


@[deprecated (since := "2024-10-22")]
alias quotientMap_proj := IsHomeomorphicTrivialFiberBundle.isQuotientMap_proj


/-- The first projection in a product is a trivial fiber bundle. -/
theorem isHomeomorphicTrivialFiberBundle_fst :
    IsHomeomorphicTrivialFiberBundle F (Prod.fst : B × F → B) :=
  ⟨Homeomorph.refl _, fun _x => rfl⟩


/-- The second projection in a product is a trivial fiber bundle. -/
theorem isHomeomorphicTrivialFiberBundle_snd :
    IsHomeomorphicTrivialFiberBundle F (Prod.snd : F × B → B) :=
  ⟨Homeomorph.prodComm _ _, fun _x => rfl⟩

