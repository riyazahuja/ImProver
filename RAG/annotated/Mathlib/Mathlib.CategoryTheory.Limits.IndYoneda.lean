/-- Hom is functorially cocontinuous: coyoneda of a colimit is the limit
over coyoneda of the diagram. -/
noncomputable def coyonedaOpColimitIsoLimitCoyoneda :
    coyoneda.obj (op <| colimit F) ≅ limit (F.op ⋙ coyoneda) :=
  coyoneda.mapIso (limitOpIsoOpColimit F).symm ≪≫ (preservesLimitIso coyoneda F.op)


@[reassoc (attr := simp)]
lemma coyonedaOpColimitIsoLimitCoyoneda_hom_comp_π (i : I) :
    (coyonedaOpColimitIsoLimitCoyoneda F).hom ≫ limit.π (F.op.comp coyoneda) ⟨i⟩
      = coyoneda.map (colimit.ι F i).op := by
  simp only [coyonedaOpColimitIsoLimitCoyoneda, Functor.mapIso_symm,
    Iso.trans_hom, Iso.symm_hom, Functor.mapIso_inv, Category.assoc, preservesLimitIso_hom_π,
    ← Functor.map_comp, limitOpIsoOpColimit_inv_comp_π]


@[reassoc (attr := simp)]
lemma coyonedaOpColimitIsoLimitCoyoneda_inv_comp_π (i : I) :
    (coyonedaOpColimitIsoLimitCoyoneda F).inv ≫ coyoneda.map (colimit.ι F i).op =
      limit.π (F.op.comp coyoneda) ⟨i⟩ := by
  rw [← coyonedaOpColimitIsoLimitCoyoneda_hom_comp_π, ← Category.assoc,
    Iso.inv_hom_id, Category.id_comp]


/-- Hom is cocontinuous: homomorphisms from a colimit is the limit over yoneda of the diagram. -/
noncomputable def colimitHomIsoLimitYoneda
    [HasLimitsOfShape Iᵒᵖ (Type u₂)] (A : C) :
    (colimit F ⟶ A) ≅ limit (F.op ⋙ yoneda.obj A) :=
  (coyonedaOpColimitIsoLimitCoyoneda F).app A ≪≫ limitObjIsoLimitCompEvaluation _ _


@[reassoc (attr := simp)]
lemma colimitHomIsoLimitYoneda_hom_comp_π [HasLimitsOfShape Iᵒᵖ (Type u₂)] (A : C) (i : I) :
    (colimitHomIsoLimitYoneda F A).hom ≫ limit.π (F.op ⋙ yoneda.obj A) ⟨i⟩ =
      (coyoneda.map (colimit.ι F i).op).app A := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    F : CategoryTheory.Functor I C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type u₂)
    A : C
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitHomIsoL …
  -/
  simp only [colimitHomIsoLimitYoneda, Iso.trans_hom, Iso.app_hom, Category.assoc]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    F : CategoryTheory.Functor I C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type u₂)
    A : C
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.coyonedaOpCol …
  -/
  erw [limitObjIsoLimitCompEvaluation_hom_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    F : CategoryTheory.Functor I C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type u₂)
    A : C
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.coyonedaOpCol …
  -/
  change ((coyonedaOpColimitIsoLimitCoyoneda F).hom ≫ _).app A = _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    F : CategoryTheory.Functor I C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type u₂)
    A : C
    i : I
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coyonedaOpCol …
  -/
  rw [coyonedaOpColimitIsoLimitCoyoneda_hom_comp_π]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma colimitHomIsoLimitYoneda_inv_comp_π [HasLimitsOfShape Iᵒᵖ (Type u₂)] (A : C) (i : I) :
    (colimitHomIsoLimitYoneda F A).inv ≫ (coyoneda.map (colimit.ι F i).op).app A =
      limit.π (F.op ⋙ yoneda.obj A) ⟨i⟩ := by
  rw [← colimitHomIsoLimitYoneda_hom_comp_π, ← Category.assoc,
    Iso.inv_hom_id, Category.id_comp]


/-- Variant of `coyonedaOoColimitIsoLimitCoyoneda` for contravariant `F`. -/
noncomputable def coyonedaOpColimitIsoLimitCoyoneda' :
    coyoneda.obj (op <| colimit F) ≅ limit (F.rightOp ⋙ coyoneda) :=
  coyoneda.mapIso (limitRightOpIsoOpColimit F).symm ≪≫ preservesLimitIso coyoneda F.rightOp


@[reassoc (attr := simp)]
lemma coyonedaOpColimitIsoLimitCoyoneda'_hom_comp_π (i : I) :
    (coyonedaOpColimitIsoLimitCoyoneda' F).hom ≫ limit.π (F.rightOp ⋙ coyoneda) i =
      coyoneda.map (colimit.ι F ⟨i⟩).op := by
  simp only [coyonedaOpColimitIsoLimitCoyoneda', Functor.mapIso_symm, Iso.trans_hom, Iso.symm_hom,
    Functor.mapIso_inv, Category.assoc, preservesLimitIso_hom_π, ← Functor.map_comp,
    limitRightOpIsoOpColimit_inv_comp_π]


@[reassoc (attr := simp)]
lemma coyonedaOpColimitIsoLimitCoyoneda'_inv_comp_π (i : I) :
    (coyonedaOpColimitIsoLimitCoyoneda' F).inv ≫ coyoneda.map (colimit.ι F ⟨i⟩).op =
      limit.π (F.rightOp ⋙ coyoneda) i := by
  rw [← coyonedaOpColimitIsoLimitCoyoneda'_hom_comp_π, ← Category.assoc,
    Iso.inv_hom_id, Category.id_comp]


/-- Variant of `colimitHomIsoLimitYoneda` for contravariant `F`. -/
noncomputable def colimitHomIsoLimitYoneda' [HasLimitsOfShape I (Type u₂)] (A : C) :
    (colimit F ⟶ A) ≅ limit (F.rightOp ⋙ yoneda.obj A) :=
  (coyonedaOpColimitIsoLimitCoyoneda' F).app A ≪≫ limitObjIsoLimitCompEvaluation _ _


@[reassoc (attr := simp)]
lemma colimitHomIsoLimitYoneda'_hom_comp_π [HasLimitsOfShape I (Type u₂)] (A : C) (i : I) :
    (colimitHomIsoLimitYoneda' F A).hom ≫ limit.π (F.rightOp ⋙ yoneda.obj A) i =
      (coyoneda.map (colimit.ι F ⟨i⟩).op).app A := by
  simp only [yoneda_obj_obj, colimitHomIsoLimitYoneda', Iso.trans_hom,
    Iso.app_hom, Category.assoc]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    F : CategoryTheory.Functor (Opposite I) C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type u₂)
    A : C
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.coyonedaOpCol …
  -/
  erw [limitObjIsoLimitCompEvaluation_hom_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    F : CategoryTheory.Functor (Opposite I) C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type u₂)
    A : C
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.coyonedaOpCol …
  -/
  change ((coyonedaOpColimitIsoLimitCoyoneda' F).hom ≫ _).app A = _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    F : CategoryTheory.Functor (Opposite I) C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type u₂)
    A : C
    i : I
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coyonedaOpCol …
  -/
  rw [coyonedaOpColimitIsoLimitCoyoneda'_hom_comp_π]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma colimitHomIsoLimitYoneda'_inv_comp_π [HasLimitsOfShape I (Type u₂)] (A : C) (i : I) :
    (colimitHomIsoLimitYoneda' F A).inv ≫ (coyoneda.map (colimit.ι F ⟨i⟩).op).app A =
      limit.π (F.rightOp ⋙ yoneda.obj A) i := by
  rw [← colimitHomIsoLimitYoneda'_hom_comp_π, ← Category.assoc,
    Iso.inv_hom_id, Category.id_comp]


/-- Pro-Coyoneda lemma: morphisms from colimit of coyoneda of diagram `D` to `F` is limit
of `F` evaluated at `D`. This variant is for contravariant diagrams, see
`colimitCoyonedaHomIsoLimit'` for a covariant version. -/
noncomputable def colimitCoyonedaHomIsoLimit :
    (colimit (D.rightOp ⋙ coyoneda) ⟶ F) ≅ limit (D ⋙ F ⋙ uliftFunctor.{u₁}) :=
  colimitHomIsoLimitYoneda _ F ≪≫
    HasLimit.isoOfNatIso (isoWhiskerLeft (D ⋙ Prod.sectL C F) (coyonedaLemma C))


@[simp]
lemma colimitCoyonedaHomIsoLimit_π_apply (f : colimit (D.rightOp ⋙ coyoneda) ⟶ F) (i : I) :
    limit.π (D ⋙ F ⋙ uliftFunctor.{u₁}) (op i) ((colimitCoyonedaHomIsoLimit D F).hom f) =
      ⟨f.app (D.obj (op i)) ((colimit.ι (D.rightOp ⋙ coyoneda) i).app (D.obj (op i))
          (𝟙 (D.obj (op i))))⟩ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.rightOp.comp CategoryTheory.coyon …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.rightOp.comp CategoryTheory.c …
    i : I
    ⊢ Eq (CategoryTheory.Limits.limit.π (D.comp (F.comp CategoryTheory.uliftFuncto …
  -/
  change ((colimitCoyonedaHomIsoLimit D F).hom ≫ (limit.π (D ⋙ F ⋙ uliftFunctor.{u₁}) (op i))) f = _
  simp only [colimitCoyonedaHomIsoLimit, Iso.trans_hom, Category.assoc,
    HasLimit.isoOfNatIso_hom_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.rightOp.comp CategoryTheory.coyon …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.rightOp.comp CategoryTheory.c …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitHomIsoL …
  -/
  rw [← Category.assoc, colimitHomIsoLimitYoneda_hom_comp_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.rightOp.comp CategoryTheory.coyon …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.rightOp.comp CategoryTheory.c …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.map (Catego …
  -/
  dsimp [coyonedaLemma, types_comp_apply]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.rightOp.comp CategoryTheory.coyon …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.rightOp.comp CategoryTheory.c …
    i : I
    ⊢ Eq (Equiv.ulift.symm (CategoryTheory.coyonedaEquiv (CategoryTheory.CategoryS …
  -/
  erw [coyonedaEquiv_comp, coyonedaEquiv_apply]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.rightOp.comp CategoryTheory.coyon …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.rightOp.comp CategoryTheory.c …
    i : I
    ⊢ Eq (Equiv.ulift.symm (f.app (D.obj { unop := i }) ((CategoryTheory.Limits.co …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Pro-Coyoneda lemma: morphisms from colimit of coyoneda of diagram `D` to `F` is limit
of `F` evaluated at `D`. This variant is for contravariant diagrams, see
`colimitCoyonedaHomIsoLimit'` for a covariant version. -/
noncomputable def colimitCoyonedaHomIsoLimitLeftOp :
    (colimit (D ⋙ coyoneda) ⟶ F) ≅ limit (D.leftOp ⋙ F ⋙ uliftFunctor.{u₁}) :=
  haveI : HasColimit (D.leftOp.rightOp ⋙ coyoneda) :=
    inferInstanceAs <| HasColimit (D ⋙ coyoneda)
  colimitCoyonedaHomIsoLimit D.leftOp F


@[simp]
lemma colimitCoyonedaHomIsoLimitLeftOp_π_apply (f : colimit (D ⋙ coyoneda) ⟶ F) (i : I) :
    limit.π (D.leftOp ⋙ F ⋙ uliftFunctor.{u₁}) (op i)
        ((colimitCoyonedaHomIsoLimitLeftOp D F).hom f) =
      ⟨f.app (D.obj i).unop ((colimit.ι (D ⋙ coyoneda) i).app (D.obj i).unop
          (𝟙 (D.obj i).unop))⟩ :=
  haveI : HasColimit (D.leftOp.rightOp ⋙ coyoneda) :=
    inferInstanceAs <| HasColimit (D ⋙ coyoneda)
  colimitCoyonedaHomIsoLimit_π_apply _ _ _ _


/-- Ind-Yoneda lemma: morphisms from colimit of yoneda of diagram `D` to `F` is limit of `F`
evaluated at `D`. This version is for covariant diagrams, see `colimitYonedaHomIsoLimit'` for a
contravariant version. -/
noncomputable def colimitYonedaHomIsoLimit :
      (colimit (D.unop ⋙ yoneda) ⟶ F) ≅ limit (D ⋙ F ⋙ uliftFunctor.{u₁}) :=
  colimitHomIsoLimitYoneda _ _ ≪≫
    HasLimit.isoOfNatIso (isoWhiskerLeft (D ⋙ Prod.sectL _ _) (yonedaLemma C))


@[simp]
lemma colimitYonedaHomIsoLimit_π_apply (f : colimit (D.unop ⋙ yoneda) ⟶ F) (i : Iᵒᵖ) :
    limit.π (D ⋙ F ⋙ uliftFunctor.{u₁}) i ((colimitYonedaHomIsoLimit D F).hom f) =
      ⟨f.app (D.obj i)
        ((colimit.ι (D.unop ⋙ yoneda) i.unop).app (D.obj i) (𝟙 (D.obj i).unop))⟩ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.unop.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.unop.comp CategoryTheory.yone …
    i : Opposite I
    ⊢ Eq (CategoryTheory.Limits.limit.π (D.comp (F.comp CategoryTheory.uliftFuncto …
  -/
  change ((colimitYonedaHomIsoLimit D F).hom ≫ (limit.π (D ⋙ F ⋙ uliftFunctor.{u₁}) i)) f = _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.unop.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.unop.comp CategoryTheory.yone …
    i : Opposite I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitYonedaH …
  -/
  simp only [colimitYonedaHomIsoLimit, Iso.trans_hom, Category.assoc, HasLimit.isoOfNatIso_hom_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.unop.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.unop.comp CategoryTheory.yone …
    i : Opposite I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitHomIsoL …
  -/
  rw [← Category.assoc, colimitHomIsoLimitYoneda_hom_comp_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.unop.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.unop.comp CategoryTheory.yone …
    i : Opposite I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.map (Catego …
  -/
  dsimp [yonedaLemma]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.unop.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.unop.comp CategoryTheory.yone …
    i : Opposite I
    ⊢ Eq (Equiv.ulift.symm (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStr …
  -/
  erw [yonedaEquiv_comp, yonedaEquiv_apply]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor (Opposite I) (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.unop.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.unop.comp CategoryTheory.yone …
    i : Opposite I
    ⊢ Eq (Equiv.ulift.symm (f.app { unop := Opposite.unop (D.obj i) } ((CategoryTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Ind-Yoneda lemma: morphisms from colimit of yoneda of diagram `D` to `F` is limit of `F`
evaluated at `D`. This version is for covariant diagrams, see `colimitYonedaHomIsoLimit'` for a
contravariant version. -/
noncomputable def colimitYonedaHomIsoLimitOp :
      (colimit (D ⋙ yoneda) ⟶ F) ≅ limit (D.op ⋙ F ⋙ uliftFunctor.{u₁}) :=
  haveI : HasColimit (D.op.unop ⋙ yoneda) :=
    inferInstanceAs <| HasColimit (D ⋙ yoneda)
  colimitYonedaHomIsoLimit D.op F


@[simp]
lemma colimitYonedaHomIsoLimitOp_π_apply (f : colimit (D ⋙ yoneda) ⟶ F) (i : Iᵒᵖ) :
    limit.π (D.op ⋙ F ⋙ uliftFunctor.{u₁}) i ((colimitYonedaHomIsoLimitOp D F).hom f) =
      ⟨f.app (op (D.obj i.unop))
        ((colimit.ι (D ⋙ yoneda) i.unop).app (op (D.obj i.unop)) (𝟙 (D.obj i.unop)))⟩ :=
  haveI : HasColimit (D.op.unop ⋙ yoneda) :=
    inferInstanceAs <| HasColimit (D ⋙ yoneda)
  colimitYonedaHomIsoLimit_π_apply _ _ _ _


/-- Pro-Coyoneda lemma: morphisms from colimit of coyoneda of diagram `D` to `F` is limit
of `F` evaluated at `D`. This variant is for covariant diagrams, see
`colimitCoyonedaHomIsoLimit` for a covariant version. -/
noncomputable def colimitCoyonedaHomIsoLimit' :
    (colimit (D.op ⋙ coyoneda) ⟶ F) ≅ limit (D ⋙ F ⋙ uliftFunctor.{u₁}) :=
  colimitHomIsoLimitYoneda' _ F ≪≫
    HasLimit.isoOfNatIso (isoWhiskerLeft (D ⋙ Prod.sectL C F) (coyonedaLemma C))


@[simp]
lemma colimitCoyonedaHomIsoLimit'_π_apply (f : colimit (D.op ⋙ coyoneda) ⟶ F) (i : I) :
    limit.π (D ⋙ F ⋙ uliftFunctor.{u₁}) i ((colimitCoyonedaHomIsoLimit' D F).hom f) =
      ⟨f.app (D.obj i) ((colimit.ι (D.op ⋙ coyoneda) ⟨i⟩).app (D.obj i) (𝟙 (D.obj i)))⟩ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.op.comp CategoryTheory.coyoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.op.comp CategoryTheory.coyone …
    i : I
    ⊢ Eq (CategoryTheory.Limits.limit.π (D.comp (F.comp CategoryTheory.uliftFuncto …
  -/
  change ((colimitCoyonedaHomIsoLimit' D F).hom ≫ (limit.π (D ⋙ F ⋙ uliftFunctor.{u₁}) i)) f = _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.op.comp CategoryTheory.coyoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.op.comp CategoryTheory.coyone …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitCoyoned …
  -/
  simp only [colimitCoyonedaHomIsoLimit', Iso.trans_hom, Category.assoc, HasLimit.isoOfNatIso_hom_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.op.comp CategoryTheory.coyoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.op.comp CategoryTheory.coyone …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitHomIsoL …
  -/
  rw [← Category.assoc, colimitHomIsoLimitYoneda'_hom_comp_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.op.comp CategoryTheory.coyoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.op.comp CategoryTheory.coyone …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.map (Catego …
  -/
  dsimp [coyonedaLemma]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.op.comp CategoryTheory.coyoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.op.comp CategoryTheory.coyone …
    i : I
    ⊢ Eq (Equiv.ulift.symm (CategoryTheory.coyonedaEquiv (CategoryTheory.CategoryS …
  -/
  erw [coyonedaEquiv_comp, coyonedaEquiv_apply]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I C
    F : CategoryTheory.Functor C (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.op.comp CategoryTheory.coyoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.op.comp CategoryTheory.coyone …
    i : I
    ⊢ Eq (Equiv.ulift.symm (f.app (D.obj i) ((CategoryTheory.Limits.colimit.ι (D.o …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Pro-Coyoneda lemma: morphisms from colimit of coyoneda of diagram `D` to `F` is limit
of `F` evaluated at `D`. This variant is for covariant diagrams, see
`colimitCoyonedaHomIsoLimit` for a covariant version. -/
noncomputable def colimitCoyonedaHomIsoLimitUnop :
    (colimit (D ⋙ coyoneda) ⟶ F) ≅ limit (D.unop ⋙ F ⋙ uliftFunctor.{u₁}) :=
  haveI : HasColimit (D.unop.op ⋙ coyoneda) :=
    inferInstanceAs <| HasColimit (D ⋙ coyoneda)
  colimitCoyonedaHomIsoLimit' D.unop F


@[simp]
lemma colimitCoyonedaHomIsoLimitUnop_π_apply (f : colimit (D ⋙ coyoneda) ⟶ F) (i : I) :
    limit.π (D.unop ⋙ F ⋙ uliftFunctor.{u₁}) i ((colimitCoyonedaHomIsoLimitUnop D F).hom f) =
      ⟨f.app (D.obj (op i)).unop
          ((colimit.ι (D ⋙ coyoneda) ⟨i⟩).app (D.obj (op i)).unop (𝟙 (D.obj (op i)).unop))⟩ :=
  haveI : HasColimit (D.unop.op ⋙ coyoneda) :=
    inferInstanceAs <| HasColimit (D ⋙ coyoneda)
  colimitCoyonedaHomIsoLimit'_π_apply _ _ _ _


/-- Ind-Yoneda lemma: morphisms from colimit of yoneda of diagram `D` to `F` is limit of `F`
evaluated at `D`. This version is for contravariant diagrams, see `colimitYonedaHomIsoLimit` for a
covariant version. -/
noncomputable def colimitYonedaHomIsoLimit' :
    (colimit (D.leftOp ⋙ yoneda) ⟶ F) ≅ limit (D ⋙ F ⋙ uliftFunctor.{u₁}) :=
  colimitHomIsoLimitYoneda' _ F ≪≫
    HasLimit.isoOfNatIso (isoWhiskerLeft (D ⋙ Prod.sectL _ _) (yonedaLemma C))


@[simp]
lemma colimitYonedaHomIsoLimit'_π_apply (f : colimit (D.leftOp ⋙ yoneda) ⟶ F) (i : I) :
    limit.π (D ⋙ F ⋙ uliftFunctor.{u₁}) i ((colimitYonedaHomIsoLimit' D F).hom f) =
      ⟨f.app (D.obj i)
        ((colimit.ι (D.leftOp ⋙ yoneda) (op i)).app (D.obj i) (𝟙 (D.obj i).unop))⟩ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.leftOp.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.leftOp.comp CategoryTheory.yo …
    i : I
    ⊢ Eq (CategoryTheory.Limits.limit.π (D.comp (F.comp CategoryTheory.uliftFuncto …
  -/
  change ((colimitYonedaHomIsoLimit' D F).hom ≫ (limit.π (D ⋙ F ⋙ uliftFunctor.{u₁}) i)) f = _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.leftOp.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.leftOp.comp CategoryTheory.yo …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitYonedaH …
  -/
  simp only [colimitYonedaHomIsoLimit', Iso.trans_hom, Category.assoc, HasLimit.isoOfNatIso_hom_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.leftOp.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.leftOp.comp CategoryTheory.yo …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitHomIsoL …
  -/
  rw [← Category.assoc, colimitHomIsoLimitYoneda'_hom_comp_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.leftOp.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.leftOp.comp CategoryTheory.yo …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.map (Catego …
  -/
  dsimp [yonedaLemma]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.leftOp.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.leftOp.comp CategoryTheory.yo …
    i : I
    ⊢ Eq (Equiv.ulift.symm (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStr …
  -/
  erw [yonedaEquiv_comp, yonedaEquiv_apply]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    I : Type v₁
    inst✝² : CategoryTheory.Category.{v₂, v₁} I
    D : CategoryTheory.Functor I (Opposite C)
    F : CategoryTheory.Functor (Opposite C) (Type u₂)
    inst✝¹ : CategoryTheory.Limits.HasColimit (D.leftOp.comp CategoryTheory.yoneda)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape I (Type (max u₁ u₂))
    f : Quiver.Hom (CategoryTheory.Limits.colimit (D.leftOp.comp CategoryTheory.yo …
    i : I
    ⊢ Eq (Equiv.ulift.symm (f.app { unop := Opposite.unop (D.obj i) } ((CategoryTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Ind-Yoneda lemma: morphisms from colimit of yoneda of diagram `D` to `F` is limit of `F`
evaluated at `D`. This version is for contravariant diagrams, see `colimitYonedaHomIsoLimit` for a
covariant version. -/
noncomputable def colimitYonedaHomIsoLimitRightOp :
    (colimit (D ⋙ yoneda) ⟶ F) ≅ limit (D.rightOp ⋙ F ⋙ uliftFunctor.{u₁}) :=
  haveI : HasColimit (D.rightOp.leftOp ⋙ yoneda) :=
    inferInstanceAs <| HasColimit (D ⋙ yoneda)
  colimitYonedaHomIsoLimit' D.rightOp F


@[simp]
lemma colimitYonedaHomIsoLimitRightOp_π_apply (f : colimit (D ⋙ yoneda) ⟶ F) (i : I) :
    limit.π (D.rightOp ⋙ F ⋙ uliftFunctor.{u₁}) i ((colimitYonedaHomIsoLimitRightOp D F).hom f) =
      ⟨f.app (op (D.obj (op i)))
        ((colimit.ι (D ⋙ yoneda) (op i)).app (op (D.obj (op i))) (𝟙 (D.obj (op i))))⟩ :=
  haveI : HasColimit (D.rightOp.leftOp ⋙ yoneda) :=
    inferInstanceAs <| HasColimit (D ⋙ yoneda)
  colimitYonedaHomIsoLimit'_π_apply _ _ _ _


