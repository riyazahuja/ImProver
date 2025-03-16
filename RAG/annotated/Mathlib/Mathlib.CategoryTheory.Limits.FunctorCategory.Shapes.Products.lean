/-- Evaluating a product of functors amounts to taking the product of the evaluations. -/
noncomputable def piObjIso (f : α → D ⥤ C) (d : D) : (∏ᶜ f).obj d ≅ ∏ᶜ (fun s => (f s).obj d) :=
  limitObjIsoLimitCompEvaluation (Discrete.functor f) d ≪≫
    HasLimit.isoOfNatIso (Discrete.compNatIsoDiscrete _ _)


@[reassoc (attr := simp)]
theorem piObjIso_hom_comp_π (f : α → D ⥤ C) (d : D) (s : α) :
    (piObjIso f d).hom ≫ Pi.π (fun s => (f s).obj d) s = (Pi.π f s).app d := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    α : Type w
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
    f : α → CategoryTheory.Functor D C
    d : D
    s : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.piObjIso f d). …
  -/
  simp [piObjIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem piObjIso_inv_comp_pi (f : α → D ⥤ C) (d : D) (s : α) :
    (piObjIso f d).inv ≫ (Pi.π f s).app d = Pi.π (fun s => (f s).obj d) s := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    α : Type w
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
    f : α → CategoryTheory.Functor D C
    d : D
    s : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.piObjIso f d). …
  -/
  simp [piObjIso]
  /-
    🎉 no goals
  -/


/-- Evaluating a coproduct of functors amounts to taking the coproduct of the evaluations. -/
noncomputable def sigmaObjIso (f : α → D ⥤ C) (d : D) : (∐ f).obj d ≅ ∐ (fun s => (f s).obj d) :=
  colimitObjIsoColimitCompEvaluation (Discrete.functor f) d ≪≫
    HasColimit.isoOfNatIso (Discrete.compNatIsoDiscrete _ _)


@[reassoc (attr := simp)]
theorem ι_comp_sigmaObjIso_hom (f : α → D ⥤ C) (d : D) (s : α) :
    (Sigma.ι f s).app d ≫ (sigmaObjIso f d).hom = Sigma.ι (fun s => (f s).obj d) s := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    α : Type w
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
    f : α → CategoryTheory.Functor D C
    d : D
    s : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Sigma.ι f s). …
  -/
  simp [sigmaObjIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem ι_comp_sigmaObjIso_inv (f : α → D ⥤ C) (d : D) (s : α) :
    Sigma.ι (fun s => (f s).obj d) s ≫ (sigmaObjIso f d).inv = (Sigma.ι f s).app d := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    α : Type w
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
    f : α → CategoryTheory.Functor D C
    d : D
    s : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun s …
  -/
  simp [sigmaObjIso]
  /-
    🎉 no goals
  -/


