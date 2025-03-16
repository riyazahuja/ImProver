@[simp]
theorem preserves_lift_mapCone (c₁ c₂ : Cone F) (t : IsLimit c₁) :
    (isLimitOfPreserves G t).lift (G.mapCone c₂) = G.map (t.lift c₂) :=
                                                      /-
                                                        C : Type u₁
                                                        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                        D : Type u₂
                                                        inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                        G : CategoryTheory.Functor C D
                                                        J : Type w
                                                        inst✝¹ : CategoryTheory.Category.{w', w} J
                                                        F : CategoryTheory.Functor J C
                                                        inst✝ : CategoryTheory.Limits.PreservesLimit F G
                                                        c₁ c₂ : CategoryTheory.Limits.Cone F
                                                        t : CategoryTheory.Limits.IsLimit c₁
                                                        ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (G.map (t.lift c₂)) ((G.ma …
                                                      -/
  ((isLimitOfPreserves G t).uniq (G.mapCone c₂) _ (by simp [← G.map_comp])).symm
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- If `G` preserves limits, we have an isomorphism from the image of the limit of a functor `F`
to the limit of the functor `F ⋙ G`.
-/
def preservesLimitIso : G.obj (limit F) ≅ limit (F ⋙ G) :=
  (isLimitOfPreserves G (limit.isLimit _)).conePointUniqueUpToIso (limit.isLimit _)


@[reassoc (attr := simp)]
theorem preservesLimitIso_hom_π (j) :
    (preservesLimitIso G F).hom ≫ limit.π _ j = G.map (limit.π F j) :=
  IsLimit.conePointUniqueUpToIso_hom_comp _ _ j


@[deprecated (since := "2024-10-27")] alias preservesLimitsIso_hom_π := preservesLimitIso_hom_π


@[reassoc (attr := simp)]
theorem preservesLimitIso_inv_π (j) :
    (preservesLimitIso G F).inv ≫ G.map (limit.π F j) = limit.π _ j :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ j


@[deprecated (since := "2024-10-27")] alias preservesLimitsIso_inv_π := preservesLimitIso_inv_π


@[reassoc (attr := simp)]
theorem lift_comp_preservesLimitIso_hom (t : Cone F) :
    G.map (limit.lift _ t) ≫ (preservesLimitIso G F).hom =
    limit.lift (F ⋙ G) (G.mapCone _) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    inst✝² : CategoryTheory.Category.{w', w} J
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.PreservesLimit F G
    inst✝ : CategoryTheory.Limits.HasLimit F
    t : CategoryTheory.Limits.Cone F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.limit.l …
  -/
  ext
  /-
    case w
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    inst✝² : CategoryTheory.Category.{w', w} J
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.PreservesLimit F G
    inst✝ : CategoryTheory.Limits.HasLimit F
    t : CategoryTheory.Limits.Cone F
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [← G.map_comp]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-27")]
alias lift_comp_preservesLimitsIso_hom := lift_comp_preservesLimitIso_hom


instance : IsIso (limit.post F G) :=
  show IsIso (preservesLimitIso G F).hom from inferInstance


/-- If `C, D` has all limits of shape `J`, and `G` preserves them, then `preservesLimitsIso` is
functorial wrt `F`. -/
@[simps!]
def preservesLimitNatIso : lim ⋙ G ≅ (whiskeringRight J C D).obj G ⋙ lim :=
  NatIso.ofComponents (fun F => preservesLimitIso G F)
    (by
      /-
        C : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        J : Type w
        inst✝⁵ : CategoryTheory.Category.{w', w} J
        F : CategoryTheory.Functor J C
        inst✝⁴ : CategoryTheory.Limits.PreservesLimit F G
        inst✝³ : CategoryTheory.Limits.HasLimit F
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape J G
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J D
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
        ⊢ ∀ {X Y : CategoryTheory.Functor J C} (f : Quiver.Hom X Y), Eq (CategoryTheor …
      -/
      intro _ _ f
      /-
        C : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        J : Type w
        inst✝⁵ : CategoryTheory.Category.{w', w} J
        F : CategoryTheory.Functor J C
        inst✝⁴ : CategoryTheory.Limits.PreservesLimit F G
        inst✝³ : CategoryTheory.Limits.HasLimit F
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape J G
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J D
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
        X✝ Y✝ : CategoryTheory.Functor J C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.lim.comp G).m …
      -/
      apply limit.hom_ext; intro j
      /-
        case w
        C : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        J : Type w
        inst✝⁵ : CategoryTheory.Category.{w', w} J
        F : CategoryTheory.Functor J C
        inst✝⁴ : CategoryTheory.Limits.PreservesLimit F G
        inst✝³ : CategoryTheory.Limits.HasLimit F
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape J G
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J D
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
        X✝ Y✝ : CategoryTheory.Functor J C
        f : Quiver.Hom X✝ Y✝
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp
      simp only [preservesLimitIso_hom_π, whiskerRight_app, limMap_π, Category.assoc,
        preservesLimitIso_hom_π_assoc, ← G.map_comp])


/-- If the comparison morphism `G.obj (limit F) ⟶ limit (F ⋙ G)` is an isomorphism, then `G`
    preserves limits of `F`. -/
lemma preservesLimit_of_isIso_post [IsIso (limit.post F G)] : PreservesLimit F G :=
  preservesLimit_of_preserves_limit_cone (limit.isLimit F) (by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      J : Type w
      inst✝³ : CategoryTheory.Category.{w', w} J
      F : CategoryTheory.Functor J C
      inst✝² : CategoryTheory.Limits.HasLimit F
      inst✝¹ : CategoryTheory.Limits.HasLimit (F.comp G)
      inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.limit.post F G)
      ⊢ CategoryTheory.Limits.IsLimit (G.mapCone (CategoryTheory.Limits.limit.cone F))
    -/
    convert IsLimit.ofPointIso (limit.isLimit (F ⋙ G))
    /-
      case convert_2
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      J : Type w
      inst✝³ : CategoryTheory.Category.{w', w} J
      F : CategoryTheory.Functor J C
      inst✝² : CategoryTheory.Limits.HasLimit F
      inst✝¹ : CategoryTheory.Limits.HasLimit (F.comp G)
      inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.limit.post F G)
      ⊢ CategoryTheory.IsIso ((CategoryTheory.Limits.limit.isLimit (F.comp G)).lift  …
    -/
    assumption)
    /-
      🎉 no goals
    -/


@[simp]
theorem preserves_desc_mapCocone (c₁ c₂ : Cocone F) (t : IsColimit c₁) :
    (isColimitOfPreserves G t).desc (G.mapCocone _) = G.map (t.desc c₂) :=
                                                         /-
                                                           C : Type u₁
                                                           inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                           D : Type u₂
                                                           inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                           G : CategoryTheory.Functor C D
                                                           J : Type w
                                                           inst✝¹ : CategoryTheory.Category.{w', w} J
                                                           F : CategoryTheory.Functor J C
                                                           inst✝ : CategoryTheory.Limits.PreservesColimit F G
                                                           c₁ c₂ : CategoryTheory.Limits.Cocone F
                                                           t : CategoryTheory.Limits.IsColimit c₁
                                                           ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((G.mapCocone c₁).ι.app j) …
                                                         -/
  ((isColimitOfPreserves G t).uniq (G.mapCocone _) _ (by simp [← G.map_comp])).symm
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- If `G` preserves colimits, we have an isomorphism from the image of the colimit of a functor `F`
to the colimit of the functor `F ⋙ G`.
-/
def preservesColimitIso : G.obj (colimit F) ≅ colimit (F ⋙ G) :=
  (isColimitOfPreserves G (colimit.isColimit _)).coconePointUniqueUpToIso (colimit.isColimit _)


@[reassoc (attr := simp)]
theorem ι_preservesColimitIso_inv (j : J) :
    colimit.ι _ j ≫ (preservesColimitIso G F).inv = G.map (colimit.ι F j) :=
  IsColimit.comp_coconePointUniqueUpToIso_inv _ (colimit.isColimit (F ⋙ G)) j


@[deprecated (since := "2024-10-27")]
alias ι_preservesColimitsIso_inv := ι_preservesColimitIso_inv


@[reassoc (attr := simp)]
theorem ι_preservesColimitIso_hom (j : J) :
    G.map (colimit.ι F j) ≫ (preservesColimitIso G F).hom = colimit.ι (F ⋙ G) j :=
  (isColimitOfPreserves G (colimit.isColimit _)).comp_coconePointUniqueUpToIso_hom _ j


@[deprecated (since := "2024-10-27")]
alias ι_preservesColimitsIso_hom := ι_preservesColimitIso_hom


@[reassoc (attr := simp)]
theorem preservesColimitIso_inv_comp_desc (t : Cocone F) :
    (preservesColimitIso G F).inv ≫ G.map (colimit.desc _ t) =
    colimit.desc _ (G.mapCocone t) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    inst✝² : CategoryTheory.Category.{w', w} J
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.PreservesColimit F G
    inst✝ : CategoryTheory.Limits.HasColimit F
    t : CategoryTheory.Limits.Cocone F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesColimitIso G …
  -/
  ext
  /-
    case w
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    inst✝² : CategoryTheory.Category.{w', w} J
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.PreservesColimit F G
    inst✝ : CategoryTheory.Limits.HasColimit F
    t : CategoryTheory.Limits.Cocone F
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  simp [← G.map_comp]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-27")]
alias preservesColimitsIso_inv_comp_desc := preservesColimitIso_inv_comp_desc


instance : IsIso (colimit.post F G) :=
  show IsIso (preservesColimitIso G F).inv from inferInstance


/-- If `C, D` has all colimits of shape `J`, and `G` preserves them, then `preservesColimitIso`
is functorial wrt `F`. -/
@[simps!]
def preservesColimitNatIso : colim ⋙ G ≅ (whiskeringRight J C D).obj G ⋙ colim :=
  NatIso.ofComponents (fun F => preservesColimitIso G F)
    (by
      /-
        C : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        J : Type w
        inst✝⁵ : CategoryTheory.Category.{w', w} J
        F : CategoryTheory.Functor J C
        inst✝⁴ : CategoryTheory.Limits.PreservesColimit F G
        inst✝³ : CategoryTheory.Limits.HasColimit F
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape J G
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        ⊢ ∀ {X Y : CategoryTheory.Functor J C} (f : Quiver.Hom X Y), Eq (CategoryTheor …
      -/
      intro _ _ f
      /-
        C : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        J : Type w
        inst✝⁵ : CategoryTheory.Category.{w', w} J
        F : CategoryTheory.Functor J C
        inst✝⁴ : CategoryTheory.Limits.PreservesColimit F G
        inst✝³ : CategoryTheory.Limits.HasColimit F
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape J G
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        X✝ Y✝ : CategoryTheory.Functor J C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colim.comp G) …
      -/
      rw [← Iso.inv_comp_eq, ← Category.assoc, ← Iso.eq_comp_inv]
      /-
        C : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        J : Type w
        inst✝⁵ : CategoryTheory.Category.{w', w} J
        F : CategoryTheory.Functor J C
        inst✝⁴ : CategoryTheory.Limits.PreservesColimit F G
        inst✝³ : CategoryTheory.Limits.HasColimit F
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape J G
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        X✝ Y✝ : CategoryTheory.Functor J C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun F => CategoryTheory.preservesCo …
      -/
      apply colimit.hom_ext; intro j
      /-
        case w
        C : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        J : Type w
        inst✝⁵ : CategoryTheory.Category.{w', w} J
        F : CategoryTheory.Functor J C
        inst✝⁴ : CategoryTheory.Limits.PreservesColimit F G
        inst✝³ : CategoryTheory.Limits.HasColimit F
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape J G
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        X✝ Y✝ : CategoryTheory.Functor J C
        f : Quiver.Hom X✝ Y✝
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (((C …
      -/
      dsimp
      /-
        case w
        C : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        J : Type w
        inst✝⁵ : CategoryTheory.Category.{w', w} J
        F : CategoryTheory.Functor J C
        inst✝⁴ : CategoryTheory.Limits.PreservesColimit F G
        inst✝³ : CategoryTheory.Limits.HasColimit F
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape J G
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        X✝ Y✝ : CategoryTheory.Functor J C
        f : Quiver.Hom X✝ Y✝
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (X✝. …
      -/
      rw [ι_colimMap_assoc]
      simp only [ι_preservesColimitIso_inv, whiskerRight_app, Category.assoc,
        ι_preservesColimitIso_inv_assoc, ← G.map_comp]
      /-
        case w
        C : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        J : Type w
        inst✝⁵ : CategoryTheory.Category.{w', w} J
        F : CategoryTheory.Functor J C
        inst✝⁴ : CategoryTheory.Limits.PreservesColimit F G
        inst✝³ : CategoryTheory.Limits.HasColimit F
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape J G
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J D
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        X✝ Y✝ : CategoryTheory.Functor J C
        f : Quiver.Hom X✝ Y✝
        j : J
        ⊢ Eq (G.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit …
      -/
      rw [ι_colimMap])
      /-
        🎉 no goals
      -/


/-- If the comparison morphism `colimit (F ⋙ G) ⟶ G.obj (colimit F)` is an isomorphism, then `G`
    preserves colimits of `F`. -/
lemma preservesColimit_of_isIso_post [IsIso (colimit.post F G)] : PreservesColimit F G :=
  preservesColimit_of_preserves_colimit_cocone (colimit.isColimit F) (by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      J : Type w
      inst✝³ : CategoryTheory.Category.{w', w} J
      F : CategoryTheory.Functor J C
      inst✝² : CategoryTheory.Limits.HasColimit F
      inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp G)
      inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.colimit.post F G)
      ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone (CategoryTheory.Limits.colimit. …
    -/
    convert IsColimit.ofPointIso (colimit.isColimit (F ⋙ G))
    /-
      case convert_2
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      J : Type w
      inst✝³ : CategoryTheory.Category.{w', w} J
      F : CategoryTheory.Functor J C
      inst✝² : CategoryTheory.Limits.HasColimit F
      inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp G)
      inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.colimit.post F G)
      ⊢ CategoryTheory.IsIso ((CategoryTheory.Limits.colimit.isColimit (F.comp G)).d …
    -/
    assumption)
    /-
      🎉 no goals
    -/


