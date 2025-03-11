/-- Variant of `colimitYonedaHomIsoLimitOp`: natural transformations with domain
`colimit (F ⋙ yoneda)` are equivalent to a limit in a lower universe. -/
noncomputable def colimitYonedaHomEquiv : (colimit (F ⋙ yoneda) ⟶ G) ≃ limit (F.op ⋙ G) :=
  Equiv.symm <| Equiv.ulift.symm.trans <| Equiv.symm <| Iso.toEquiv <| calc
  (colimit (F ⋙ yoneda) ⟶ G) ≅ limit (F.op ⋙ G ⋙ uliftFunctor.{u}) :=
        colimitYonedaHomIsoLimitOp _ _
  _ ≅ limit ((F.op ⋙ G) ⋙ uliftFunctor.{u}) :=
        HasLimit.isoOfNatIso (Functor.associator _ _ _).symm
  _ ≅ uliftFunctor.{u}.obj (limit (F.op ⋙ G)) :=
        (preservesLimitIso _ _).symm


@[simp]
theorem colimitYonedaHomEquiv_π_apply (η : colimit (F ⋙ yoneda) ⟶ G) (i : Iᵒᵖ) :
    limit.π (F.op ⋙ G) i (colimitYonedaHomEquiv F G η) =
      η.app (op (F.obj i.unop)) ((colimit.ι (F ⋙ yoneda) i.unop).app _ (𝟙 _)) := by
  simp only [Functor.comp_obj, Functor.op_obj, colimitYonedaHomEquiv, uliftFunctor_obj,
    Iso.instTransIso_trans, Iso.trans_assoc, Iso.toEquiv_comp, Equiv.symm_trans_apply,
    Equiv.symm_symm, Equiv.trans_apply, Iso.toEquiv_fun, Iso.symm_hom, Equiv.ulift_apply]
  have (a) := congrArg ULift.down
    (congrFun (preservesLimitIso_inv_π uliftFunctor.{u, v} (F.op ⋙ G) i) a)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    I : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} I
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape I (Type v)
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type v)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u v))
    F : CategoryTheory.Functor I C
    G : CategoryTheory.Functor (Opposite C) (Type v)
    η : Quiver.Hom (CategoryTheory.Limits.colimit (F.comp CategoryTheory.yoneda)) G
    i : Opposite I
    this : ∀ (a : CategoryTheory.Limits.limit ((F.op.comp G).comp CategoryTheory.u …
    ⊢ Eq (CategoryTheory.Limits.limit.π (F.op.comp G) i ((CategoryTheory.preserves …
  -/
  dsimp at this
  rw [this, ← types_comp_apply (HasLimit.isoOfNatIso _).hom (limit.π _ _),
    HasLimit.isoOfNatIso_hom_π]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    I : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} I
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape I (Type v)
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type v)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite I) (Type (max u v))
    F : CategoryTheory.Functor I C
    G : CategoryTheory.Functor (Opposite C) (Type v)
    η : Quiver.Hom (CategoryTheory.Limits.colimit (F.comp CategoryTheory.yoneda)) G
    i : Opposite I
    this : ∀ (a : CategoryTheory.Limits.limit ((F.op.comp G).comp CategoryTheory.u …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.op. …
  -/
  simp
  /-
    🎉 no goals
  -/


instance : Small.{v} (colimit (F ⋙ yoneda) ⟶ G) where
  equiv_small := ⟨_, ⟨colimitYonedaHomEquiv F G⟩⟩


instance : LocallySmall.{v} (FullSubcategory (IsIndObject (C := C))) where
  hom_small X Y := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : CategoryTheory.FullSubcategory CategoryTheory.Limits.IsIndObject
      ⊢ Small.{v, max u v} (Quiver.Hom X Y)
    -/
    obtain ⟨⟨P⟩⟩ := X.2
    /-
      case mk'.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : CategoryTheory.FullSubcategory CategoryTheory.Limits.IsIndObject
      P : CategoryTheory.Limits.IndObjectPresentation X.obj
      ⊢ Small.{v, max u v} (Quiver.Hom X Y)
    -/
    obtain ⟨⟨Q⟩⟩ := Y.2
    /-
      case mk'.intro.mk'.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : CategoryTheory.FullSubcategory CategoryTheory.Limits.IsIndObject
      P : CategoryTheory.Limits.IndObjectPresentation X.obj
      Q : CategoryTheory.Limits.IndObjectPresentation Y.obj
      ⊢ Small.{v, max u v} (Quiver.Hom X Y)
    -/
    let e₁ := IsColimit.coconePointUniqueUpToIso (P.isColimit) (colimit.isColimit _)
    /-
      case mk'.intro.mk'.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : CategoryTheory.FullSubcategory CategoryTheory.Limits.IsIndObject
      P : CategoryTheory.Limits.IndObjectPresentation X.obj
      Q : CategoryTheory.Limits.IndObjectPresentation Y.obj
      e₁ : CategoryTheory.Iso { pt := X.obj, ι := P.ι }.pt (CategoryTheory.Limits.co …
      ⊢ Small.{v, max u v} (Quiver.Hom X Y)
    -/
    let e₂ := IsColimit.coconePointUniqueUpToIso (Q.isColimit) (colimit.isColimit _)
    /-
      case mk'.intro.mk'.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : CategoryTheory.FullSubcategory CategoryTheory.Limits.IsIndObject
      P : CategoryTheory.Limits.IndObjectPresentation X.obj
      Q : CategoryTheory.Limits.IndObjectPresentation Y.obj
      e₁ : CategoryTheory.Iso { pt := X.obj, ι := P.ι }.pt (CategoryTheory.Limits.co …
      e₂ : CategoryTheory.Iso { pt := Y.obj, ι := Q.ι }.pt (CategoryTheory.Limits.co …
      ⊢ Small.{v, max u v} (Quiver.Hom X Y)
    -/
    let e₃ := Iso.homCongr e₁ e₂
    /-
      case mk'.intro.mk'.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : CategoryTheory.FullSubcategory CategoryTheory.Limits.IsIndObject
      P : CategoryTheory.Limits.IndObjectPresentation X.obj
      Q : CategoryTheory.Limits.IndObjectPresentation Y.obj
      e₁ : CategoryTheory.Iso { pt := X.obj, ι := P.ι }.pt (CategoryTheory.Limits.co …
      e₂ : CategoryTheory.Iso { pt := Y.obj, ι := Q.ι }.pt (CategoryTheory.Limits.co …
      e₃ : Equiv (Quiver.Hom { pt := X.obj, ι := P.ι }.pt { pt := Y.obj, ι := Q.ι }. …
      ⊢ Small.{v, max u v} (Quiver.Hom X Y)
    -/
    dsimp only [colimit.cocone_x] at e₃
    /-
      case mk'.intro.mk'.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : CategoryTheory.FullSubcategory CategoryTheory.Limits.IsIndObject
      P : CategoryTheory.Limits.IndObjectPresentation X.obj
      Q : CategoryTheory.Limits.IndObjectPresentation Y.obj
      e₁ : CategoryTheory.Iso { pt := X.obj, ι := P.ι }.pt (CategoryTheory.Limits.co …
      e₂ : CategoryTheory.Iso { pt := Y.obj, ι := Q.ι }.pt (CategoryTheory.Limits.co …
      e₃ : Equiv (Quiver.Hom X.obj Y.obj) (Quiver.Hom (CategoryTheory.Limits.colimit …
      ⊢ Small.{v, max u v} (Quiver.Hom X Y)
    -/
    exact small_map e₃
    /-
      🎉 no goals
    -/


