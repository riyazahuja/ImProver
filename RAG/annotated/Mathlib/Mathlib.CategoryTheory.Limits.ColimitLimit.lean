theorem map_id_left_eq_curry_map {j : J} {k k' : K} {f : k ⟶ k'} :
    F.map ((𝟙 j, f) : (j, k) ⟶ (j, k')) = ((curry.obj F).obj j).map f :=
  rfl


theorem map_id_right_eq_curry_swap_map {j j' : J} {f : j ⟶ j'} {k : K} :
    F.map ((f, 𝟙 k) : (j, k) ⟶ (j', k)) = ((curry.obj (Prod.swap K J ⋙ F)).obj k).map f :=
  rfl


/-- The universal morphism
$\colim_k \lim_j F(j,k) → \lim_j \colim_k F(j, k)$.
-/
noncomputable def colimitLimitToLimitColimit :
    colimit (curry.obj (Prod.swap K J ⋙ F) ⋙ lim) ⟶ limit (curry.obj F ⋙ colim) :=
  limit.lift (curry.obj F ⋙ colim)
    { pt := _
      π :=
        { app := fun j =>
            colimit.desc (curry.obj (Prod.swap K J ⋙ F) ⋙ lim)
              { pt := _
                ι :=
                  { app := fun k =>
                      limit.π ((curry.obj (Prod.swap K J ⋙ F)).obj k) j ≫
                        colimit.ι ((curry.obj F).obj j) k
                    naturality := by
                      /-
                        J : Type u₁
                        K : Type u₂
                        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
                        inst✝³ : CategoryTheory.Category.{v₂, u₂} K
                        C : Type u
                        inst✝² : CategoryTheory.Category.{v, u} C
                        F : CategoryTheory.Functor (Prod J K) C
                        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
                        inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
                        j : J
                        ⊢ ∀ ⦃X Y : K⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                      -/
                      intro k k' f
                      simp only [Functor.comp_obj, lim_obj, colimit.cocone_x,
                        Functor.const_obj_obj, Functor.comp_map, lim_map,
                        curry_obj_obj_obj, Prod.swap_obj, limMap_π_assoc, curry_obj_map_app,
                        Prod.swap_map, Functor.const_obj_map, Category.comp_id]
                      /-
                        J : Type u₁
                        K : Type u₂
                        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
                        inst✝³ : CategoryTheory.Category.{v₂, u₂} K
                        C : Type u
                        inst✝² : CategoryTheory.Category.{v, u} C
                        F : CategoryTheory.Functor (Prod J K) C
                        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
                        inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
                        j : J
                        k k' : K
                        f : Quiver.Hom k k'
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π ((Cate …
                      -/
                      rw [map_id_left_eq_curry_map, colimit.w] } }
                      /-
                        🎉 no goals
                      -/
          naturality := by
            /-
              J : Type u₁
              K : Type u₂
              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
              inst✝³ : CategoryTheory.Category.{v₂, u₂} K
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor (Prod J K) C
              inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
              inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
              ⊢ ∀ ⦃X Y : J⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
            -/
            intro j j' f
            /-
              J : Type u₁
              K : Type u₂
              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
              inst✝³ : CategoryTheory.Category.{v₂, u₂} K
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor (Prod J K) C
              inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
              inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
              j j' : J
              f : Quiver.Hom j j'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
            -/
            dsimp
            /-
              J : Type u₁
              K : Type u₂
              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
              inst✝³ : CategoryTheory.Category.{v₂, u₂} K
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor (Prod J K) C
              inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
              inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
              j j' : J
              f : Quiver.Hom j j'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
            -/
            ext k
            simp only [Functor.comp_obj, lim_obj, Category.id_comp, colimit.ι_desc,
              colimit.ι_desc_assoc, Category.assoc, ι_colimMap,
              curry_obj_obj_obj, curry_obj_map_app]
            /-
              case w
              J : Type u₁
              K : Type u₂
              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
              inst✝³ : CategoryTheory.Category.{v₂, u₂} K
              C : Type u
              inst✝² : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor (Prod J K) C
              inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
              inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
              j j' : J
              f : Quiver.Hom j j'
              k : K
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π ((Cate …
            -/
            rw [map_id_right_eq_curry_swap_map, limit.w_assoc] } }
            /-
              🎉 no goals
            -/


/-- Since `colimit_limit_to_limit_colimit` is a morphism from a colimit to a limit,
this lemma characterises it.
-/
@[reassoc (attr := simp)]
theorem ι_colimitLimitToLimitColimit_π (j) (k) :
    colimit.ι _ k ≫ colimitLimitToLimitColimit F ≫ limit.π _ j =
      limit.π ((curry.obj (Prod.swap K J ⋙ F)).obj k) j ≫ colimit.ι ((curry.obj F).obj j) k := by
  /-
    J : Type u₁
    K : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Prod J K) C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
  -/
  dsimp [colimitLimitToLimitColimit]
  /-
    J : Type u₁
    K : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Prod J K) C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem ι_colimitLimitToLimitColimit_π_apply [Small.{v} J] [Small.{v} K] (F : J × K ⥤ Type v)
    (j : J) (k : K) (f) : limit.π (curry.obj F ⋙ colim) j
        (colimitLimitToLimitColimit F (colimit.ι (curry.obj (Prod.swap K J ⋙ F) ⋙ lim) k f)) =
      colimit.ι ((curry.obj F).obj j) k (limit.π ((curry.obj (Prod.swap K J ⋙ F)).obj k) j f) := by
  /-
    J : Type u₁
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    inst✝¹ : Small.{v, u₁} J
    inst✝ : Small.{v, u₂} K
    F : CategoryTheory.Functor (Prod J K) (Type v)
    j : J
    k : K
    f : ((CategoryTheory.curry.obj ((CategoryTheory.Prod.swap K J).comp F)).comp C …
    ⊢ Eq (CategoryTheory.Limits.limit.π ((CategoryTheory.curry.obj F).comp Categor …
  -/
  dsimp [colimitLimitToLimitColimit]
  /-
    J : Type u₁
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    inst✝¹ : Small.{v, u₁} J
    inst✝ : Small.{v, u₂} K
    F : CategoryTheory.Functor (Prod J K) (Type v)
    j : J
    k : K
    f : ((CategoryTheory.curry.obj ((CategoryTheory.Prod.swap K J).comp F)).comp C …
    ⊢ Eq (CategoryTheory.Limits.limit.π ((CategoryTheory.curry.obj F).comp Categor …
  -/
  rw [Types.Limit.lift_π_apply]
  /-
    J : Type u₁
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    inst✝¹ : Small.{v, u₁} J
    inst✝ : Small.{v, u₂} K
    F : CategoryTheory.Functor (Prod J K) (Type v)
    j : J
    k : K
    f : ((CategoryTheory.curry.obj ((CategoryTheory.Prod.swap K J).comp F)).comp C …
    ⊢ Eq ({ pt := CategoryTheory.Limits.colimit ((CategoryTheory.curry.obj ((Categ …
  -/
  dsimp only
  /-
    J : Type u₁
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    inst✝¹ : Small.{v, u₁} J
    inst✝ : Small.{v, u₂} K
    F : CategoryTheory.Functor (Prod J K) (Type v)
    j : J
    k : K
    f : ((CategoryTheory.curry.obj ((CategoryTheory.Prod.swap K J).comp F)).comp C …
    ⊢ Eq (CategoryTheory.Limits.colimit.desc ((CategoryTheory.curry.obj ((Category …
  -/
  rw [Types.Colimit.ι_desc_apply]
  /-
    J : Type u₁
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    inst✝¹ : Small.{v, u₁} J
    inst✝ : Small.{v, u₂} K
    F : CategoryTheory.Functor (Prod J K) (Type v)
    j : J
    k : K
    f : ((CategoryTheory.curry.obj ((CategoryTheory.Prod.swap K J).comp F)).comp C …
    ⊢ Eq ({ pt := CategoryTheory.Limits.colimit ((CategoryTheory.curry.obj F).obj  …
  -/
  dsimp
  /-
    🎉 no goals
  -/


/-- The map `colimit_limit_to_limit_colimit` realized as a map of cones. -/
@[simps]
noncomputable def colimitLimitToLimitColimitCone (G : J ⥤ K ⥤ C) [HasLimit G] :
    colim.mapCone (limit.cone G) ⟶ limit.cone (G ⋙ colim) where
  hom :=
    colim.map (limitIsoSwapCompLim G).hom ≫
      colimitLimitToLimitColimit (uncurry.obj G : _) ≫
        lim.map (whiskerRight (currying.unitIso.app G).inv colim)
  w j := by
    /-
      J : Type u₁
      K : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor (Prod J K) C
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape K C
      G : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      inst✝ : CategoryTheory.Limits.HasLimit G
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    dsimp
    /-
      J : Type u₁
      K : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor (Prod J K) C
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape K C
      G : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      inst✝ : CategoryTheory.Limits.HasLimit G
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    ext1 k
    simp only [Category.assoc, limMap_π, Functor.comp_obj, colim_obj, whiskerRight_app,
      colim_map, ι_colimMap_assoc, lim_obj, limitIsoSwapCompLim_hom_app,
      ι_colimitLimitToLimitColimit_π_assoc, curry_obj_obj_obj, Prod.swap_obj,
      uncurry_obj_obj, ι_colimMap, currying_unitIso_inv_app_app_app, Category.id_comp,
      limMap_π_assoc, Functor.flip_obj_obj, flipIsoCurrySwapUncurry_hom_app_app]
    /-
      case w
      J : Type u₁
      K : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor (Prod J K) C
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape K C
      G : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      inst✝ : CategoryTheory.Limits.HasLimit G
      j : J
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitObjIsoLim …
    -/
    erw [limitObjIsoLimitCompEvaluation_hom_π_assoc]
    /-
      🎉 no goals
    -/


