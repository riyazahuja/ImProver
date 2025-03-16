/-- The induced scalar multiplication on
`colimit (F ⋙ forget₂ _ AddCommGrp)`. -/
@[simps]
noncomputable def coconePointSMul :
    R →+* End (colimit (F ⋙ forget₂ _ AddCommGrp)) where
  toFun r := colimMap
    { app := fun j => (F.obj j).smul r
      naturality := fun _ _ _ => smul_naturality _ _ }
                                   /-
                                     R : Type w
                                     inst✝² : Ring R
                                     J : Type u
                                     inst✝¹ : CategoryTheory.Category.{v, u} J
                                     F : CategoryTheory.Functor J (ModuleCat R)
                                     inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
                                     ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.col …
                                   -/
                                  /-
                                    R : Type w
                                    inst✝² : Ring R
                                    J : Type u
                                    inst✝¹ : CategoryTheory.Category.{v, u} J
                                    F : CategoryTheory.Functor J (ModuleCat R)
                                    inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
                                    ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.col …
                                  -/
  map_zero' := colimit.hom_ext (by simp)
                                  /-
                                    🎉 no goals
                                  -/
                                   /-
                                     🎉 no goals
                                   -/
  map_one' := colimit.hom_ext (by simp)
  map_add' r s := colimit.hom_ext (fun j => by
    /-
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      r s : R
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
                                               /-
                                                 R : Type w
                                                 inst✝² : Ring R
                                                 J : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} J
                                                 F : CategoryTheory.Functor J (ModuleCat R)
                                                 inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
                                                 r s : R
                                                 j : J
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
                                               -/
    simp only [Functor.comp_obj, forget₂_obj, map_add, ι_colimMap]
                                               /-
                                                 🎉 no goals
                                               -/
    /-
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      r s : R
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd ((F.obj j).smul r) ((F.obj …
    -/
    rw [Preadditive.add_comp, Preadditive.comp_add]
    /-
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      r s : R
      j : J
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((F.obj j).smul r) (Catego …
    -/
    simp only [ι_colimMap, Functor.comp_obj, forget₂_obj])
    /-
      🎉 no goals
    -/
  map_mul' r s := colimit.hom_ext (fun j => by simp)


/-- The cocone for `F` constructed from the colimit of
`(F ⋙ forget₂ (ModuleCat R) AddCommGrp)`. -/
@[simps]
noncomputable def colimitCocone : Cocone F where
  pt := mkOfSMul (coconePointSMul F)
  ι :=
    { app := fun j => homMk (colimit.ι (F ⋙ forget₂ _ AddCommGrp)  j) (fun r => by
        /-
          R : Type w
          inst✝² : Ring R
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          F : CategoryTheory.Functor J (ModuleCat R)
          inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
          j : J
          r : R
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
        -/
        dsimp
        -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
        /-
          R : Type w
          inst✝² : Ring R
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          F : CategoryTheory.Functor J (ModuleCat R)
          inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
          j : J
          r : R
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
        -/
        erw [mkOfSMul_smul]
        /-
          R : Type w
          inst✝² : Ring R
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          F : CategoryTheory.Functor J (ModuleCat R)
          inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
          j : J
          r : R
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
        -/
        simp)
        /-
          🎉 no goals
        -/
      naturality := fun i j f => by
        /-
          R : Type w
          inst✝² : Ring R
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          F : CategoryTheory.Functor J (ModuleCat R)
          inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
          i j : J
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => ModuleCat.homMk  …
        -/
        apply (forget₂ _ AddCommGrp).map_injective
        /-
          case a
          R : Type w
          inst✝² : Ring R
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          F : CategoryTheory.Functor J (ModuleCat R)
          inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
          i j : J
          f : Quiver.Hom i j
          ⊢ Eq ((CategoryTheory.forget₂ (ModuleCat R) AddCommGrp).map (CategoryTheory.Ca …
        -/
        simp only [Functor.map_comp, forget₂_map_homMk]
        /-
          case a
          R : Type w
          inst✝² : Ring R
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          F : CategoryTheory.Functor J (ModuleCat R)
          inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
          i j : J
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.forget₂ (ModuleCat R …
        -/
        dsimp
        /-
          case a
          R : Type w
          inst✝² : Ring R
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          F : CategoryTheory.Functor J (ModuleCat R)
          inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
          i j : J
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f).hom.toAddMonoidHom (Categor …
        -/
        erw [colimit.w (F ⋙ forget₂ _ AddCommGrp), comp_id] }
        /-
          🎉 no goals
        -/


/-- The cocone for `F` constructed from the colimit of
`(F ⋙ forget₂ (ModuleCat R) AddCommGrp)` is a colimit cocone. -/
noncomputable def isColimitColimitCocone : IsColimit (colimitCocone F) where
  desc s := homMk (colimit.desc _ ((forget₂ _ AddCommGrp).mapCocone s)) (fun r => by
    /-
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      r : R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.desc ( …
    -/
    apply colimit.hom_ext
    /-
      case w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      r : R
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.col …
    -/
    intro j
    /-
      case w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      r : R
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    dsimp
    /-
      case w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      r : R
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    rw [colimit.ι_desc_assoc]
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      r : R
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.forget₂ (ModuleCat  …
    -/
    erw [mkOfSMul_smul]
    /-
      case w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      r : R
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.forget₂ (ModuleCat  …
    -/
    dsimp
    simp only [ι_colimMap_assoc, Functor.comp_obj, forget₂_obj, colimit.ι_desc,
      Functor.mapCocone_pt, Functor.mapCocone_ι_app, forget₂_map]
    /-
      case w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      r : R
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j).hom.toAddMonoidHom (s.pt. …
    -/
    exact smul_naturality (s.ι.app j) r)
    /-
      🎉 no goals
    -/
  fac s j := by
    /-
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.HasColimit.colimitCocone  …
    -/
    apply (forget₂ _ AddCommGrp).map_injective
    /-
      case a
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      j : J
      ⊢ Eq ((CategoryTheory.forget₂ (ModuleCat R) AddCommGrp).map (CategoryTheory.Ca …
    -/
    exact colimit.ι_desc ((forget₂ _ AddCommGrp).mapCocone s) j
    /-
      🎉 no goals
    -/
  uniq s m hm := by
    /-
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (ModuleCat.HasColimit.colimitCocone F).pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.HasColimit. …
      ⊢ Eq m ((fun s => ModuleCat.homMk (CategoryTheory.Limits.colimit.desc (F.comp  …
    -/
    apply (forget₂ _ AddCommGrp).map_injective
    /-
      case a
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (ModuleCat.HasColimit.colimitCocone F).pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.HasColimit. …
      ⊢ Eq ((CategoryTheory.forget₂ (ModuleCat R) AddCommGrp).map m) ((CategoryTheor …
    -/
    apply colimit.hom_ext
    /-
      case a.w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (ModuleCat.HasColimit.colimitCocone F).pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.HasColimit. …
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.col …
    -/
    intro j
    /-
      case a.w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (ModuleCat.HasColimit.colimitCocone F).pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.HasColimit. …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    erw [colimit.ι_desc ((forget₂ _ AddCommGrp).mapCocone s) j]
    /-
      case a.w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (ModuleCat.HasColimit.colimitCocone F).pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.HasColimit. …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    dsimp
    /-
      case a.w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (ModuleCat.HasColimit.colimitCocone F).pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.HasColimit. …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    rw [← hm]
    /-
      case a.w
      R : Type w
      inst✝² : Ring R
      J : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (ModuleCat.HasColimit.colimitCocone F).pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.HasColimit. …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    rfl
    /-
      🎉 no goals
    -/


instance : HasColimit F := ⟨_, isColimitColimitCocone F⟩


noncomputable instance : PreservesColimit F (forget₂ _ AddCommGrp) :=
  preservesColimit_of_preserves_colimit_cocone (isColimitColimitCocone F) (colimit.isColimit _)


noncomputable instance reflectsColimit :
    ReflectsColimit F (forget₂ (ModuleCat.{w'} R) AddCommGrp) :=
  reflectsColimit_of_reflectsIsomorphisms _ _


instance hasColimitsOfShape [HasColimitsOfShape J AddCommGrp.{w'}] :
    HasColimitsOfShape J (ModuleCat.{w'} R) where


noncomputable instance reflectsColimitsOfShape [HasColimitsOfShape J AddCommGrp.{w'}] :
    ReflectsColimitsOfShape J (forget₂ (ModuleCat.{w'} R) AddCommGrp) where


instance hasColimitsOfSize [HasColimitsOfSize.{v, u} AddCommGrp.{w'}] :
    HasColimitsOfSize.{v, u} (ModuleCat.{w'} R) where


noncomputable instance forget₂PreservesColimitsOfShape
    [HasColimitsOfShape J AddCommGrp.{w'}] :
    PreservesColimitsOfShape J (forget₂ (ModuleCat.{w'} R) AddCommGrp) where


noncomputable instance forget₂PreservesColimitsOfSize
    [HasColimitsOfSize.{u, v} AddCommGrp.{w'}] :
    PreservesColimitsOfSize.{u, v} (forget₂ (ModuleCat.{w'} R) AddCommGrp) where


noncomputable instance
    [HasColimitsOfSize.{u, v} AddCommGrpMax.{w, w'}] :
    PreservesColimitsOfSize.{u, v} (forget₂ (ModuleCatMax.{w, w'} R) AddCommGrp) where


instance : HasFiniteColimits (ModuleCat.{w'} R) := inferInstance

-- Sanity checks, just to make sure typeclass search can find the instances we want.

instance : HasCoequalizers (ModuleCat.{v} R) where


