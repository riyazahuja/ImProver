lemma isSheaf_of_isLimit (hc : IsLimit c) (hF : ∀ j, Presheaf.IsSheaf J (F.obj j).presheaf) :
    Presheaf.IsSheaf J (c.pt.presheaf) := by
  let G : D ⥤ Sheaf J AddCommGrp.{v} :=
    { obj := fun j => ⟨(F.obj j).presheaf, hF j⟩
      map := fun φ => ⟨(PresheafOfModules.toPresheaf R).map (F.map φ)⟩ }
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    R : CategoryTheory.Functor (Opposite C) RingCat
    F : CategoryTheory.Functor D (PresheafOfModules R)
    inst✝¹ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules. …
    c : CategoryTheory.Limits.Cone F
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape D AddCommGrp
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (j : D), CategoryTheory.Presheaf.IsSheaf J (F.obj j).presheaf
    G : CategoryTheory.Functor D (CategoryTheory.Sheaf J AddCommGrp) := { obj := f …
    ⊢ CategoryTheory.Presheaf.IsSheaf J c.pt.presheaf
  -/
  exact Sheaf.isSheaf_of_isLimit G _ (isLimitOfPreserves (toPresheaf R) hc)
  /-
    🎉 no goals
  -/


instance (X : Cᵒᵖ) : Small.{v} (((F ⋙ forget _) ⋙ PresheafOfModules.evaluation _ X) ⋙
    CategoryTheory.forget _).sections := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    R : CategoryTheory.Sheaf J RingCat
    F : CategoryTheory.Functor D (SheafOfModules R)
    inst✝¹ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (SheafOfModules.eva …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape D AddCommGrp
    X : Opposite C
    ⊢ Small.{v, max u₂ v} ↑(((F.comp (SheafOfModules.forget R)).comp (PresheafOfMo …
  -/
  change Small.{v} ((F ⋙ evaluation R X) ⋙ CategoryTheory.forget _).sections
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    R : CategoryTheory.Sheaf J RingCat
    F : CategoryTheory.Functor D (SheafOfModules R)
    inst✝¹ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (SheafOfModules.eva …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape D AddCommGrp
    X : Opposite C
    ⊢ Small.{v, max u₂ v} ↑((F.comp (SheafOfModules.evaluation R X)).comp (Categor …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


noncomputable instance createsLimit : CreatesLimit F (forget _) :=
  createsLimitOfFullyFaithfulOfIso' (limit.isLimit (F ⋙ forget _))
    (mk (limit (F ⋙ forget _))
      (PresheafOfModules.isSheaf_of_isLimit (limit.isLimit (F ⋙ forget _))
        (fun j => (F.obj j).isSheaf))) (Iso.refl _)


instance hasLimit : HasLimit F := hasLimit_of_created F (forget _)


noncomputable instance evaluationPreservesLimit (X : Cᵒᵖ) :
    PreservesLimit F (evaluation R X) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    R : CategoryTheory.Sheaf J RingCat
    F : CategoryTheory.Functor D (SheafOfModules R)
    inst✝¹ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (SheafOfModules.eva …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape D AddCommGrp
    X : Opposite C
    ⊢ CategoryTheory.Limits.PreservesLimit F (SheafOfModules.evaluation R X)
  -/
  dsimp [evaluation]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    R : CategoryTheory.Sheaf J RingCat
    F : CategoryTheory.Functor D (SheafOfModules R)
    inst✝¹ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (SheafOfModules.eva …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape D AddCommGrp
    X : Opposite C
    ⊢ CategoryTheory.Limits.PreservesLimit F ((SheafOfModules.forget R).comp (Pres …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance hasLimitsOfShape : HasLimitsOfShape D (SheafOfModules.{v} R) where


noncomputable instance evaluationPreservesLimitsOfShape (X : Cᵒᵖ) :
    PreservesLimitsOfShape D (evaluation R X : SheafOfModules.{v} R ⥤ _) where


noncomputable instance forgetPreservesLimitsOfShape :
    PreservesLimitsOfShape D (forget.{v} R) where


instance hasFiniteLimits : HasFiniteLimits (SheafOfModules.{v} R) :=
  ⟨fun _ => inferInstance⟩


noncomputable instance evaluationPreservesFiniteLimits (X : Cᵒᵖ) :
    PreservesFiniteLimits (evaluation.{v} R X) where


noncomputable instance forgetPreservesFiniteLimits :
    PreservesFiniteLimits (forget.{v} R) where


instance hasLimitsOfSize : HasLimitsOfSize.{v₂, v} (SheafOfModules.{v} R) where


noncomputable instance evaluationPreservesLimitsOfSize (X : Cᵒᵖ) :
    PreservesLimitsOfSize.{v₂, v} (evaluation R X : SheafOfModules.{v} R ⥤ _) where


noncomputable instance forgetPreservesLimitsOfSize :
    PreservesLimitsOfSize.{v₂, v} (forget.{v} R) where


noncomputable instance :
     PreservesFiniteLimits (SheafOfModules.toSheaf.{v} R ⋙ sheafToPresheaf _ _) :=
  comp_preservesFiniteLimits (SheafOfModules.forget.{v} R) (PresheafOfModules.toPresheaf R.val)


noncomputable instance : PreservesFiniteLimits (SheafOfModules.toSheaf.{v} R) :=
  preservesFiniteLimits_of_reflects_of_preserves _ (sheafToPresheaf _ _)


