/-- Naturally in `X`, we have `Hom(YX, colim_i Fi) ≅ colim_i Hom(YX, Fi)`. -/
noncomputable def yonedaYonedaColimit :
    yoneda.op ⋙ yoneda.obj (colimit F) ≅ yoneda.op ⋙ colimit (F ⋙ yoneda) := calc
  yoneda.op ⋙ yoneda.obj (colimit F)
    ≅ colimit F ⋙ uliftFunctor.{u₁} := yonedaOpCompYonedaObj (colimit F)
  _ ≅ F.flip ⋙ colim ⋙ uliftFunctor.{u₁} :=
        isoWhiskerRight (colimitIsoFlipCompColim F) uliftFunctor.{u₁}
  _ ≅ F.flip ⋙ (whiskeringRight _ _ _).obj uliftFunctor.{u₁} ⋙ colim :=
        isoWhiskerLeft F.flip (preservesColimitNatIso uliftFunctor.{u₁})
  _ ≅ (yoneda.op ⋙ coyoneda ⋙ (whiskeringLeft _ _ _).obj F) ⋙ colim := isoWhiskerRight
        (isoWhiskerRight largeCurriedYonedaLemma.symm ((whiskeringLeft _ _ _).obj F)) colim
  _ ≅ yoneda.op ⋙ colimit (F ⋙ yoneda) :=
        isoWhiskerLeft yoneda.op (colimitIsoFlipCompColim (F ⋙ yoneda)).symm


theorem yonedaYonedaColimit_app_inv {X : C} : ((yonedaYonedaColimit F).app (op X)).inv =
    (colimitObjIsoColimitCompEvaluation _ _).hom ≫
      (colimit.post F (coyoneda.obj (op (yoneda.obj X)))) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    ⊢ Eq ((CategoryTheory.yonedaYonedaColimit F).app { unop := X }).inv (CategoryT …
  -/
  dsimp [yonedaYonedaColimit]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitObjIsoC …
  -/
  simp only [Category.id_comp, Iso.cancel_iso_hom_left]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimMap (Cate …
  -/
  apply colimit.hom_ext
  /-
    case w
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.col …
  -/
  intro j
  /-
    case w
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  rw [colimit.ι_post, ι_colimMap_assoc]
  /-
    case w
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.whiskerLeft F (Categ …
  -/
  simp only [← CategoryTheory.Functor.assoc, comp_evaluation]
  /-
    case w
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.whiskerLeft F (Categ …
  -/
  rw [ι_preservesColimitIso_inv_assoc, ← Functor.map_comp_assoc]
  /-
    case w
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.whiskerLeft F (Categ …
  -/
  simp only [← comp_evaluation]
  /-
    case w
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.whiskerLeft F (Categ …
  -/
  rw [colimitObjIsoColimitCompEvaluation_ι_inv, whiskerLeft_app]
  /-
    case w
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.largeCurriedYonedaLe …
  -/
  ext η Y f
  simp [largeCurriedYonedaLemma, yonedaOpCompYonedaObj, FunctorToTypes.colimit.map_ι_apply,
    map_yonedaEquiv]


noncomputable instance {X : C} : PreservesColimit F (coyoneda.obj (op (yoneda.obj X))) := by
  suffices IsIso (colimit.post F (coyoneda.obj (op (yoneda.obj X)))) from
    preservesColimit_of_isIso_post _ _
  suffices colimit.post F (coyoneda.obj (op (yoneda.obj X))) =
      (colimitObjIsoColimitCompEvaluation _ _).inv ≫ ((yonedaYonedaColimit F).app (op X)).inv from
    this ▸ inferInstance
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} J
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J (Type v₁)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J (Type (max u₁ v₁))
    F : CategoryTheory.Functor J (CategoryTheory.Functor (Opposite C) (Type v₁))
    X : C
    ⊢ Eq (CategoryTheory.Limits.colimit.post F (CategoryTheory.coyoneda.obj { unop …
  -/
  rw [yonedaYonedaColimit_app_inv, Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


