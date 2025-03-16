theorem isIndObject_pi (h : ∀ (g : α → C), IsIndObject (∏ᶜ yoneda.obj ∘ g))
    (f : α → Cᵒᵖ ⥤ Type v) (hf : ∀ a, IsIndObject (f a)) : IsIndObject (∏ᶜ f) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    α : Type v
    h : ∀ (g : α → C), CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.pi …
    f : α → CategoryTheory.Functor (Opposite C) (Type v)
    hf : ∀ (a : α), CategoryTheory.Limits.IsIndObject (f a)
    ⊢ CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.piObj f)
  -/
  let F := fun a => (hf a).presentation.F ⋙ yoneda
  suffices (∏ᶜ f ≅ colimit (pointwiseProduct F)) from
    (isIndObject_colimit _ _ (fun i => h _)).map this.inv
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    α : Type v
    h : ∀ (g : α → C), CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.pi …
    f : α → CategoryTheory.Functor (Opposite C) (Type v)
    hf : ∀ (a : α), CategoryTheory.Limits.IsIndObject (f a)
    F : (a : α) → CategoryTheory.Functor ⋯.presentation.I (CategoryTheory.Functor  …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.piObj f) (CategoryTheory.Limits.co …
  -/
  refine Pi.mapIso (fun s => ?_) ≪≫ (asIso (colimitPointwiseProductToProductColimit F)).symm
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    α : Type v
    h : ∀ (g : α → C), CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.pi …
    f : α → CategoryTheory.Functor (Opposite C) (Type v)
    hf : ∀ (a : α), CategoryTheory.Limits.IsIndObject (f a)
    F : (a : α) → CategoryTheory.Functor ⋯.presentation.I (CategoryTheory.Functor  …
    s : α
    ⊢ CategoryTheory.Iso (f s) (CategoryTheory.Limits.colimit (F s))
  -/
  exact IsColimit.coconePointUniqueUpToIso (hf s).presentation.isColimit (colimit.isColimit _)
  /-
    🎉 no goals
  -/


theorem isIndObject_limit_of_discrete (h : ∀ (g : α → C), IsIndObject (∏ᶜ yoneda.obj ∘ g))
    (F : Discrete α ⥤ Cᵒᵖ ⥤ Type v) (hF : ∀ a, IsIndObject (F.obj a)) : IsIndObject (limit F) :=
  IsIndObject.map (Pi.isoLimit _).hom (isIndObject_pi h _ (fun a => hF ⟨a⟩))


theorem isIndObject_limit_of_discrete_of_hasLimitsOfShape [HasLimitsOfShape (Discrete α) C]
    (F : Discrete α ⥤ Cᵒᵖ ⥤ Type v) (hF : ∀ a, IsIndObject (F.obj a)) : IsIndObject (limit F) :=
  isIndObject_limit_of_discrete (fun g => (isIndObject_limit_comp_yoneda (Discrete.functor g)).map
      (HasLimit.isoOfNatIso (Discrete.compNatIsoDiscrete g yoneda)).hom) F hF


