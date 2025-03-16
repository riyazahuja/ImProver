theorem isColimit_exists_rep {c : Cocone F} (hc : IsColimit c) (x : c.pt) :
    ∃ (i : J) (y : F.obj i), (c.ι.app i).base y = x :=
  Concrete.isColimit_exists_rep (F ⋙ forget C) (isColimitOfPreserves (forget C) hc) x

-- Porting note: argument `C` of colimit need to be made explicit, odd

theorem colimit_exists_rep (x : colimit (C := SheafedSpace C) F) :
    ∃ (i : J) (y : F.obj i), (colimit.ι F i).base y = x :=
  Concrete.isColimit_exists_rep (F ⋙ SheafedSpace.forget C)
    (isColimitOfPreserves (SheafedSpace.forget _) (colimit.isColimit F)) x


instance {X Y : SheafedSpace C} (f g : X ⟶ Y) : Epi (coequalizer.π f g).base := by
  rw [← show _ = (coequalizer.π f g).base from
      ι_comp_coequalizerComparison f g (SheafedSpace.forget C),
      ← PreservesCoequalizer.iso_hom]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    J : Type v
    inst✝ : CategoryTheory.Category.{v, v} J
    F : CategoryTheory.Functor J (AlgebraicGeometry.SheafedSpace C)
    X Y : AlgebraicGeometry.SheafedSpace C
    f g : Quiver.Hom X Y
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
  -/
  apply epi_comp
  /-
    🎉 no goals
  -/


/-- The explicit coproduct for `F : discrete ι ⥤ LocallyRingedSpace`. -/
noncomputable def coproduct : LocallyRingedSpace where
  toSheafedSpace := colimit (C := SheafedSpace.{u+1, u, u} CommRingCatMax.{u, u})
    (F ⋙ forgetToSheafedSpace)
  isLocalRing x := by
    /-
      ι : Type u
      F : CategoryTheory.Functor (CategoryTheory.Discrete ι) AlgebraicGeometry.Local …
      x : ↑↑(CategoryTheory.Limits.colimit (F.comp AlgebraicGeometry.LocallyRingedSp …
      ⊢ IsLocalRing ↑((CategoryTheory.Limits.colimit (F.comp AlgebraicGeometry.Local …
    -/
    obtain ⟨i, y, ⟨⟩⟩ := SheafedSpace.colimit_exists_rep (F ⋙ forgetToSheafedSpace) x
    haveI : IsLocalRing (((F ⋙ forgetToSheafedSpace).obj i).presheaf.stalk y) :=
      (F.obj i).isLocalRing _
    exact
      (asIso ((colimit.ι (C := SheafedSpace.{u+1, u, u} CommRingCatMax.{u, u})
          (F ⋙ forgetToSheafedSpace) i : _).stalkMap y)).symm.commRingCatIsoToRingEquiv.isLocalRing


/-- The explicit coproduct cofan for `F : discrete ι ⥤ LocallyRingedSpace`. -/
noncomputable def coproductCofan : Cocone F where
  pt := coproduct F
  ι :=
    { app := fun j => ⟨colimit.ι (C := SheafedSpace.{u+1, u, u} CommRingCatMax.{u, u})
        (F ⋙ forgetToSheafedSpace) j, inferInstance⟩
                                                        /-
                                                          ι : Type u
                                                          F : CategoryTheory.Functor (CategoryTheory.Discrete ι) AlgebraicGeometry.Local …
                                                          x✝² x✝¹ : CategoryTheory.Discrete ι
                                                          j j' : ι
                                                          x✝ : Quiver.Hom { as := j } { as := j' }
                                                          f : Eq j j'
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { down := { down := f } }) ((f …
                                                        -/
      naturality := fun ⟨j⟩ ⟨j'⟩ ⟨⟨(f : j = j')⟩⟩ => by subst f; aesop }
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- The explicit coproduct cofan constructed in `coproduct_cofan` is indeed a colimit. -/
noncomputable def coproductCofanIsColimit : IsColimit (coproductCofan F) where
  desc s :=
    ⟨colimit.desc (C := SheafedSpace.{u+1, u, u} CommRingCatMax.{u, u})
      (F ⋙ forgetToSheafedSpace) (forgetToSheafedSpace.mapCocone s), by
      /-
        ι : Type u
        F : CategoryTheory.Functor (CategoryTheory.Discrete ι) AlgebraicGeometry.Local …
        s : CategoryTheory.Limits.Cocone F
        ⊢ ∀ (x : ↑↑(AlgebraicGeometry.LocallyRingedSpace.coproductCofan F).pt.toPreshe …
      -/
      intro x
      /-
        ι : Type u
        F : CategoryTheory.Functor (CategoryTheory.Discrete ι) AlgebraicGeometry.Local …
        s : CategoryTheory.Limits.Cocone F
        x : ↑↑(AlgebraicGeometry.LocallyRingedSpace.coproductCofan F).pt.toPresheafedS …
        ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.L …
      -/
      obtain ⟨i, y, ⟨⟩⟩ := SheafedSpace.colimit_exists_rep (F ⋙ forgetToSheafedSpace) x
      have := PresheafedSpace.stalkMap.comp
        (colimit.ι (C := SheafedSpace.{u+1, u, u} CommRingCatMax.{u, u})
          (F ⋙ forgetToSheafedSpace) i)
        (colimit.desc (C := SheafedSpace.{u+1, u, u} CommRingCatMax.{u, u})
          (F ⋙ forgetToSheafedSpace) (forgetToSheafedSpace.mapCocone s)) y
      /-
        case intro.intro.refl
        ι : Type u
        F : CategoryTheory.Functor (CategoryTheory.Discrete ι) AlgebraicGeometry.Local …
        s : CategoryTheory.Limits.Cocone F
        i : CategoryTheory.Discrete ι
        y : ↑↑((F.comp AlgebraicGeometry.LocallyRingedSpace.forgetToSheafedSpace).obj  …
        this : Eq (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.Cate …
        ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.L …
      -/
      rw [← IsIso.comp_inv_eq] at this
      erw [← this,
        PresheafedSpace.stalkMap.congr_hom _ _
          (colimit.ι_desc (C := SheafedSpace.{u+1, u, u} CommRingCatMax.{u, u})
            (forgetToSheafedSpace.mapCocone s) i : _)]
      haveI :
        IsLocalHom
          (((forgetToSheafedSpace.mapCocone s).ι.app i).stalkMap y).hom :=
        (s.ι.app i).2 y
      /-
        case intro.intro.refl
        ι : Type u
        F : CategoryTheory.Functor (CategoryTheory.Discrete ι) AlgebraicGeometry.Local …
        s : CategoryTheory.Limits.Cocone F
        i : CategoryTheory.Discrete ι
        y : ↑↑((F.comp AlgebraicGeometry.LocallyRingedSpace.forgetToSheafedSpace).obj  …
        this✝ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSp …
        this : IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap ((AlgebraicG …
        ⊢ IsLocalHom (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruc …
      -/
      infer_instance⟩
      /-
        🎉 no goals
      -/
  fac _ _ := LocallyRingedSpace.Hom.ext'
    (colimit.ι_desc (C := SheafedSpace.{u+1, u, u} CommRingCatMax.{u, u}) _ _)
  uniq s f h :=
    LocallyRingedSpace.Hom.ext'
      (IsColimit.uniq _ (forgetToSheafedSpace.mapCocone s) f.toShHom fun j =>
        congr_arg LocallyRingedSpace.Hom.toShHom (h j))


instance : HasCoproducts.{u} LocallyRingedSpace.{u} := fun _ =>
  ⟨fun F => ⟨⟨⟨_, coproductCofanIsColimit F⟩⟩⟩⟩


noncomputable instance (J : Type _) :
    PreservesColimitsOfShape (Discrete.{u} J) forgetToSheafedSpace.{u} :=
  ⟨fun {G} =>
    preservesColimit_of_preserves_colimit_cocone (coproductCofanIsColimit G)
      ((colimit.isColimit (C := SheafedSpace.{u+1, u, u} CommRingCatMax.{u, u}) _).ofIsoColimit
        (Cocones.ext (Iso.refl _) fun _ => Category.comp_id _))⟩


@[instance]
theorem coequalizer_π_app_isLocalHom
    (U : TopologicalSpace.Opens (coequalizer f.toShHom g.toShHom).carrier) :
    IsLocalHom ((coequalizer.π f.toShHom g.toShHom : _).c.app (op U)).hom := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    ⊢ IsLocalHom ((CategoryTheory.Limits.coequalizer.π (AlgebraicGeometry.LocallyR …
  -/
  have := ι_comp_coequalizerComparison f.toShHom g.toShHom SheafedSpace.forgetToPresheafedSpace
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequaliz …
    ⊢ IsLocalHom ((CategoryTheory.Limits.coequalizer.π (AlgebraicGeometry.LocallyR …
  -/
  rw [← PreservesCoequalizer.iso_hom] at this
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequaliz …
    ⊢ IsLocalHom ((CategoryTheory.Limits.coequalizer.π (AlgebraicGeometry.LocallyR …
  -/
  erw [SheafedSpace.congr_app this.symm (op U)]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequaliz …
    ⊢ IsLocalHom (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CategoryStru …
  -/
  rw [PresheafedSpace.comp_c_app, ← PresheafedSpace.colimitPresheafObjIsoComponentwiseLimit_hom_π]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): this instance has to be manually added
  haveI : IsIso (PreservesCoequalizer.iso
      SheafedSpace.forgetToPresheafedSpace f.toShHom g.toShHom).hom.c :=
    PresheafedSpace.c_isIso_of_iso _
  -- Had to add this instance too.
  have := CommRingCat.equalizer_ι_is_local_ring_hom' (PresheafedSpace.componentwiseDiagram _
        ((Opens.map
              (PreservesCoequalizer.iso SheafedSpace.forgetToPresheafedSpace (Hom.toShHom f)
                    (Hom.toShHom g)).hom.base).obj
          (unop (op U))))
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    this✝¹ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequal …
    this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.PreservesCoequalizer.iso A …
    this : IsLocalHom (CategoryTheory.Limits.limit.π (AlgebraicGeometry.Presheafed …
    ⊢ IsLocalHom (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-10")]
alias coequalizer_π_app_isLocalRingHom := coequalizer_π_app_isLocalHom


/-- (Implementation). The basic open set of the section `π꙳ s`. -/
noncomputable def imageBasicOpen : Opens Y :=
  Y.toRingedSpace.basicOpen
    (show Y.presheaf.obj (op (unop _)) from ((coequalizer.π f.toShHom g.toShHom).c.app (op U)) s)


theorem imageBasicOpen_image_preimage :
    (coequalizer.π f.toShHom g.toShHom).base ⁻¹' ((coequalizer.π f.toShHom g.toShHom).base ''
      (imageBasicOpen f g U s).1) = (imageBasicOpen f g U s).1 := by
  fapply Types.coequalizer_preimage_image_eq_of_preimage_eq f.base
    -- Porting note: Type of `f.base` and `g.base` needs to be explicit
    (g.base : X.carrier.1 ⟶ Y.carrier.1)
    /-
      case e
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ⇑f.base ⇑(CategoryTheory.Limits.coequ …
    -/
  · ext
    /-
      case e.h
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      a✝ : ↑↑X.toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (⇑f.base) (⇑(CategoryTheory.Limits.co …
    -/
    simp_rw [types_comp_apply, ← TopCat.comp_app, ← PresheafedSpace.comp_base]
    /-
      case e.h
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      a✝ : ↑↑X.toPresheafedSpace
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp f.toHom (CategoryTheory.Limits.coequ …
    -/
    congr 2
    /-
      case e.h.e_a.e_self
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      a✝ : ↑↑X.toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f.toHom (CategoryTheory.Limits.coequa …
    -/
    exact coequalizer.condition f.toShHom g.toShHom
    /-
      🎉 no goals
    -/
    /-
      case h
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ ⇑(Category …
    -/
  · apply isColimitCoforkMapOfIsColimit (forget TopCat)
    /-
      case h.l
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ (CategoryT …
    -/
    apply isColimitCoforkMapOfIsColimit (SheafedSpace.forget _)
    /-
      case h.l.l
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ (CategoryT …
    -/
    exact coequalizerIsCoequalizer f.toShHom g.toShHom
    /-
      🎉 no goals
    -/
  · suffices
      (TopologicalSpace.Opens.map f.base).obj (imageBasicOpen f g U s) =
        (TopologicalSpace.Opens.map g.base).obj (imageBasicOpen f g U s)
      by injection this
    /-
      case H
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      ⊢ Eq ((TopologicalSpace.Opens.map f.base).obj (AlgebraicGeometry.LocallyRinged …
    -/
    delta imageBasicOpen
    /-
      case H
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      ⊢ Eq ((TopologicalSpace.Opens.map f.base).obj (Y.toRingedSpace.basicOpen (letF …
    -/
    rw [preimage_basicOpen f, preimage_basicOpen g]
    /-
      case H
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      ⊢ Eq (X.toRingedSpace.basicOpen ((f.c.app { unop := Opposite.unop { unop := (T …
    -/
    dsimp only [Functor.op, unop_op]
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): change `rw` to `erw`
    erw [← CommRingCat.comp_apply, ← SheafedSpace.comp_c_app', ← CommRingCat.comp_apply,
      ← SheafedSpace.comp_c_app',
      SheafedSpace.congr_app (coequalizer.condition f.toShHom g.toShHom),
      CommRingCat.comp_apply, X.toRingedSpace.basicOpen_res]
    /-
      case H
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      ⊢ Eq (Min.min (Opposite.unop ((TopologicalSpace.Opens.map (CategoryTheory.Cate …
    -/
    apply inf_eq_right.mpr
    /-
      case H
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      ⊢ LE.le (X.toRingedSpace.basicOpen (((CategoryTheory.CategoryStruct.comp (Alge …
    -/
    refine (RingedSpace.basicOpen_le _ _).trans ?_
    /-
      case H
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
      s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
      ⊢ LE.le (Opposite.unop ((TopologicalSpace.Opens.map (CategoryTheory.CategorySt …
    -/
    rw [coequalizer.condition f.toShHom g.toShHom]
    /-
      🎉 no goals
    -/


theorem imageBasicOpen_image_open :
    IsOpen ((coequalizer.π f.toShHom g.toShHom).base '' (imageBasicOpen f g U s).1) := by
  rw [← (TopCat.homeoOfIso (PreservesCoequalizer.iso (SheafedSpace.forget _) f.toShHom
    g.toShHom)).isOpen_preimage, TopCat.coequalizer_isOpen_iff, ← Set.preimage_comp]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
    ⊢ IsOpen (Set.preimage (Function.comp ⇑(TopCat.homeoOfIso (CategoryTheory.Limi …
  -/
  erw [← TopCat.coe_comp]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
    ⊢ IsOpen (Set.preimage (⇑(CategoryTheory.CategoryStruct.comp (CategoryTheory.L …
  -/
  rw [PreservesCoequalizer.iso_hom, ι_comp_coequalizerComparison]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
    ⊢ IsOpen (Set.preimage (⇑((AlgebraicGeometry.SheafedSpace.forget CommRingCat). …
  -/
  dsimp only [SheafedSpace.forget]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): change `rw` to `erw`
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
    ⊢ IsOpen (Set.preimage (⇑(CategoryTheory.Limits.coequalizer.π (AlgebraicGeomet …
  -/
  erw [imageBasicOpen_image_preimage]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    s : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
    ⊢ IsOpen (AlgebraicGeometry.LocallyRingedSpace.HasCoequalizer.imageBasicOpen f …
  -/
  exact (imageBasicOpen f g U s).2
  /-
    🎉 no goals
  -/


@[instance]
theorem coequalizer_π_stalk_isLocalHom (x : Y) :
    IsLocalHom ((coequalizer.π f.toShHom g.toShHom : _).stalkMap x).hom := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    x : ↑Y.toTopCat
    ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.L …
  -/
  constructor
  /-
    case map_nonunit
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    x : ↑Y.toTopCat
    ⊢ ∀ (a : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRinged …
  -/
  rintro a ha
  /-
    case map_nonunit
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    x : ↑Y.toTopCat
    a : ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRingedSpace …
    ha : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.L …
    ⊢ IsUnit a
  -/
  rcases TopCat.Presheaf.germ_exist (C := CommRingCat) _ _ a with ⟨U, hU, s, rfl⟩
  -- need `erw` to see through `ConcreteCategory.instFunLike`
  rw [← CommRingCat.forget_map_apply, PresheafedSpace.stalkMap_germ_apply
    (coequalizer.π (C := SheafedSpace _) f.toShHom g.toShHom) U _ hU] at ha
  /-
    case map_nonunit.intro.intro.intro
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    x : ↑Y.toTopCat
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    hU : Membership.mem U ((CategoryTheory.Limits.coequalizer.π (AlgebraicGeometry …
    s : (CategoryTheory.forget CommRingCat).obj ((CategoryTheory.Limits.coequalize …
    ha : IsUnit ((Y.presheaf.germ ((TopologicalSpace.Opens.map (CategoryTheory.Lim …
    ⊢ IsUnit (((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRinged …
  -/
  rw [CommRingCat.forget_map_apply]
  /-
    case map_nonunit.intro.intro.intro
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    x : ↑Y.toTopCat
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    hU : Membership.mem U ((CategoryTheory.Limits.coequalizer.π (AlgebraicGeometry …
    s : (CategoryTheory.forget CommRingCat).obj ((CategoryTheory.Limits.coequalize …
    ha : IsUnit ((Y.presheaf.germ ((TopologicalSpace.Opens.map (CategoryTheory.Lim …
    ⊢ IsUnit (((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRinged …
  -/
  let V := imageBasicOpen f g U s
  have hV : (coequalizer.π f.toShHom g.toShHom).base ⁻¹'
      ((coequalizer.π f.toShHom g.toShHom).base '' V.1) = V.1 :=
    imageBasicOpen_image_preimage f g U s
  have hV' : V = ⟨(coequalizer.π f.toShHom g.toShHom).base ⁻¹'
      ((coequalizer.π f.toShHom g.toShHom).base '' V.1), hV.symm ▸ V.2⟩ :=
    SetLike.ext' hV.symm
  have V_open : IsOpen ((coequalizer.π f.toShHom g.toShHom).base '' V.1) :=
    imageBasicOpen_image_open f g U s
  have VleU : (⟨(coequalizer.π f.toShHom g.toShHom).base '' V.1, V_open⟩ : _) ≤ U :=
    Set.image_subset_iff.mpr (Y.toRingedSpace.basicOpen_le _)
  /-
    case map_nonunit.intro.intro.intro
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    x : ↑Y.toTopCat
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    hU : Membership.mem U ((CategoryTheory.Limits.coequalizer.π (AlgebraicGeometry …
    s : (CategoryTheory.forget CommRingCat).obj ((CategoryTheory.Limits.coequalize …
    ha : IsUnit ((Y.presheaf.germ ((TopologicalSpace.Opens.map (CategoryTheory.Lim …
    V : TopologicalSpace.Opens ↑Y.toTopCat := AlgebraicGeometry.LocallyRingedSpace …
    hV : Eq (Set.preimage (⇑(CategoryTheory.Limits.coequalizer.π (AlgebraicGeometr …
    hV' : Eq V { carrier := Set.preimage (⇑(CategoryTheory.Limits.coequalizer.π (A …
    V_open : IsOpen (Set.image (⇑(CategoryTheory.Limits.coequalizer.π (AlgebraicGe …
    VleU : LE.le { carrier := Set.image (⇑(CategoryTheory.Limits.coequalizer.π (Al …
    ⊢ IsUnit (((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRinged …
  -/
  have hxV : x ∈ V := ⟨hU, ha⟩
  rw [← CommRingCat.germ_res_apply (coequalizer f.toShHom g.toShHom).presheaf (homOfLE VleU) _
      (@Set.mem_image_of_mem _ _ (coequalizer.π f.toShHom g.toShHom).base x V.1 hxV) s]
  /-
    case map_nonunit.intro.intro.intro
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    x : ↑Y.toTopCat
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    hU : Membership.mem U ((CategoryTheory.Limits.coequalizer.π (AlgebraicGeometry …
    s : (CategoryTheory.forget CommRingCat).obj ((CategoryTheory.Limits.coequalize …
    ha : IsUnit ((Y.presheaf.germ ((TopologicalSpace.Opens.map (CategoryTheory.Lim …
    V : TopologicalSpace.Opens ↑Y.toTopCat := AlgebraicGeometry.LocallyRingedSpace …
    hV : Eq (Set.preimage (⇑(CategoryTheory.Limits.coequalizer.π (AlgebraicGeometr …
    hV' : Eq V { carrier := Set.preimage (⇑(CategoryTheory.Limits.coequalizer.π (A …
    V_open : IsOpen (Set.image (⇑(CategoryTheory.Limits.coequalizer.π (AlgebraicGe …
    VleU : LE.le { carrier := Set.image (⇑(CategoryTheory.Limits.coequalizer.π (Al …
    hxV : Membership.mem V x
    ⊢ IsUnit (((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyRinged …
  -/
  apply RingHom.isUnit_map
  rw [← isUnit_map_iff ((coequalizer.π f.toShHom g.toShHom : _).c.app _).hom,
    ← CommRingCat.comp_apply, NatTrans.naturality, CommRingCat.comp_apply,
    ← isUnit_map_iff (Y.presheaf.map (eqToHom hV').op).hom]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): change `rw` to `erw`
  /-
    case map_nonunit.intro.intro.intro.a
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    x : ↑Y.toTopCat
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.coequalizer (AlgebraicGeom …
    hU : Membership.mem U ((CategoryTheory.Limits.coequalizer.π (AlgebraicGeometry …
    s : (CategoryTheory.forget CommRingCat).obj ((CategoryTheory.Limits.coequalize …
    ha : IsUnit ((Y.presheaf.germ ((TopologicalSpace.Opens.map (CategoryTheory.Lim …
    V : TopologicalSpace.Opens ↑Y.toTopCat := AlgebraicGeometry.LocallyRingedSpace …
    hV : Eq (Set.preimage (⇑(CategoryTheory.Limits.coequalizer.π (AlgebraicGeometr …
    hV' : Eq V { carrier := Set.preimage (⇑(CategoryTheory.Limits.coequalizer.π (A …
    V_open : IsOpen (Set.image (⇑(CategoryTheory.Limits.coequalizer.π (AlgebraicGe …
    VleU : LE.le { carrier := Set.image (⇑(CategoryTheory.Limits.coequalizer.π (Al …
    hxV : Membership.mem V x
    ⊢ IsUnit ((Y.presheaf.map (CategoryTheory.eqToHom hV').op).hom ((((TopCat.Pres …
  -/
  erw [← CommRingCat.comp_apply, ← CommRingCat.comp_apply, ← Y.presheaf.map_comp]
  convert @RingedSpace.isUnit_res_basicOpen Y.toRingedSpace (unop _)
      (((coequalizer.π f.toShHom g.toShHom).c.app (op U)) s)


@[deprecated (since := "2024-10-10")]
alias coequalizer_π_stalk_isLocalRingHom := coequalizer_π_stalk_isLocalHom


/-- The coequalizer of two locally ringed space in the category of sheafed spaces is a locally
ringed space. -/
noncomputable def coequalizer : LocallyRingedSpace where
  toSheafedSpace := Limits.coequalizer f.toShHom g.toShHom
  isLocalRing x := by
    obtain ⟨y, rfl⟩ :=
      (TopCat.epi_iff_surjective (coequalizer.π f.toShHom g.toShHom).base).mp inferInstance x
    /-
      case intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      y : ↑↑Y.toPresheafedSpace
      ⊢ IsLocalRing ↑((CategoryTheory.Limits.coequalizer (AlgebraicGeometry.LocallyR …
    -/
    exact ((coequalizer.π f.toShHom g.toShHom : _).stalkMap y).hom.domain_isLocalRing
    /-
      🎉 no goals
    -/


/-- The explicit coequalizer cofork of locally ringed spaces. -/
noncomputable def coequalizerCofork : Cofork f g :=
  @Cofork.ofπ _ _ _ _ f g (coequalizer f g) ⟨coequalizer.π f.toShHom g.toShHom,
    -- Porting note: this used to be automatic
    HasCoequalizer.coequalizer_π_stalk_isLocalHom _ _⟩
    (LocallyRingedSpace.Hom.ext' (coequalizer.condition f.toShHom g.toShHom))


theorem isLocalHom_stalkMap_congr {X Y : RingedSpace} (f g : X ⟶ Y) (H : f = g) (x)
    (h : IsLocalHom (f.stalkMap x).hom) :
    IsLocalHom (g.stalkMap x).hom := by
  /-
    X Y : AlgebraicGeometry.RingedSpace
    f g : Quiver.Hom X Y
    H : Eq f g
    x : ↑↑X.toPresheafedSpace
    h : IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap f x).hom
    ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap g x).hom
  -/
  rw [PresheafedSpace.stalkMap.congr_hom _ _ H.symm x]; infer_instance
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The cofork constructed in `coequalizer_cofork` is indeed a colimit cocone. -/
noncomputable def coequalizerCoforkIsColimit : IsColimit (coequalizerCofork f g) := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    ⊢ CategoryTheory.Limits.IsColimit (AlgebraicGeometry.LocallyRingedSpace.coequa …
  -/
  apply Cofork.IsColimit.mk'
  /-
    case create
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    ⊢ (s : CategoryTheory.Limits.Cofork f g) → Subtype fun l => And (Eq (CategoryT …
  -/
  intro s
  /-
    case create
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Cofork f g
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeome …
  -/
  have e : f.toShHom ≫ s.π.toShHom = g.toShHom ≫ s.π.toShHom := by injection s.condition
  /-
    case create
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Cofork f g
    e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeome …
  -/
  refine ⟨⟨coequalizer.desc s.π.toShHom e, ?_⟩, ?_⟩
    /-
      case create.refine_1
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      ⊢ ∀ (x : ↑↑(AlgebraicGeometry.LocallyRingedSpace.coequalizerCofork f g).pt.toP …
    -/
  · intro x
    rcases (TopCat.epi_iff_surjective
      (coequalizer.π f.toShHom g.toShHom).base).mp inferInstance x with ⟨y, rfl⟩
    -- Porting note: was `apply isLocalHom_of_comp _ (PresheafedSpace.stalkMap ...)`, this
    -- used to allow you to provide the proof that `... ≫ ...` is a local ring homomorphism later,
    -- but this is no longer possible
    /-
      case create.refine_1.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      y : ↑↑Y.toPresheafedSpace
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.L …
    -/
    set h := _
    /-
      case create.refine_1.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      y : ↑↑Y.toPresheafedSpace
      h : ?m.176182 := ?m.176183
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.L …
    -/
    change IsLocalHom h
    suffices _ : IsLocalHom (((coequalizerCofork f g).π.1.stalkMap _).hom.comp h) by
      apply isLocalHom_of_comp _ ((coequalizerCofork f g).π.1.stalkMap _).hom
    /-
      case create.refine_1.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      y : ↑↑Y.toPresheafedSpace
      h : RingHom ↑(s.pt.presheaf.stalk ((CategoryTheory.Limits.coequalizer.desc (Al …
      ⊢ IsLocalHom (((AlgebraicGeometry.LocallyRingedSpace.coequalizerCofork f g).π. …
    -/
    rw [← CommRingCat.hom_ofHom h, ← CommRingCat.hom_comp]
    /-
      case create.refine_1.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      y : ↑↑Y.toPresheafedSpace
      h : RingHom ↑(s.pt.presheaf.stalk ((CategoryTheory.Limits.coequalizer.desc (Al …
      ⊢ IsLocalHom (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom h) ((Algeb …
    -/
    erw [← PresheafedSpace.stalkMap.comp]
    /-
      case create.refine_1.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      y : ↑↑Y.toPresheafedSpace
      h : RingHom ↑(s.pt.presheaf.stalk ((CategoryTheory.Limits.coequalizer.desc (Al …
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.C …
    -/
    apply isLocalHom_stalkMap_congr _ _ (coequalizer.π_desc s.π.toShHom e).symm y
    /-
      case create.refine_1.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      y : ↑↑Y.toPresheafedSpace
      h : RingHom ↑(s.pt.presheaf.stalk ((CategoryTheory.Limits.coequalizer.desc (Al …
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometr …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  /-
    case create.refine_2
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Cofork f g
    e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRinged …
  -/
  constructor
    /-
      case create.refine_2.left
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
    -/
  · exact LocallyRingedSpace.Hom.ext' (coequalizer.π_desc _ _)
    /-
      🎉 no goals
    -/
  /-
    case create.refine_2.right
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Cofork f g
    e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    ⊢ ∀ {m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.Walk …
  -/
  intro m h
  /-
    case create.refine_2.right
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Cofork f g
    e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
    h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    ⊢ Eq m { toHom := CategoryTheory.Limits.coequalizer.desc (AlgebraicGeometry.Lo …
  -/
  replace h : (coequalizerCofork f g).π.toShHom ≫ m.1 = s.π.toShHom := by rw [← h]; rfl
  /-
    case create.refine_2.right
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Cofork f g
    e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
    h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    ⊢ Eq m { toHom := CategoryTheory.Limits.coequalizer.desc (AlgebraicGeometry.Lo …
  -/
  apply LocallyRingedSpace.Hom.ext'
  /-
    case create.refine_2.right.h
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Cofork f g
    e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
    h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom m) { toHom := CategoryT …
  -/
  apply (colimit.isColimit (parallelPair f.toShHom g.toShHom)).uniq (Cofork.ofπ s.π.toShHom e) m.1
  /-
    case create.refine_2.right.h
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Cofork f g
    e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
    h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
    ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Catego …
  -/
  rintro ⟨⟩
  · rw [← (colimit.cocone (parallelPair f.toShHom g.toShHom)).w WalkingParallelPairHom.left,
      Category.assoc]
    /-
      case create.refine_2.right.h.zero
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.parallelPair  …
    -/
    change _ ≫ _ ≫ _ = _ ≫ _
    /-
      case create.refine_2.right.h.zero
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.parallelPair  …
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case create.refine_2.right.h.one
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Cofork f g
      e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.cocon …
    -/
  · exact h
    /-
      🎉 no goals
    -/


instance : HasCoequalizer f g :=
  ⟨⟨⟨_, coequalizerCoforkIsColimit f g⟩⟩⟩


instance : HasCoequalizers LocallyRingedSpace :=
  hasCoequalizers_of_hasColimit_parallelPair _


noncomputable instance preservesCoequalizer :
    PreservesColimitsOfShape WalkingParallelPair forgetToSheafedSpace.{v} :=
  ⟨fun {F} => by
    -- Porting note: was `apply preservesColimitOfIsoDiagram ...` and the proof that preservation
    -- of colimit is provided later
    suffices PreservesColimit (parallelPair (F.map WalkingParallelPairHom.left)
        (F.map WalkingParallelPairHom.right)) forgetToSheafedSpace from
      preservesColimit_of_iso_diagram _ (diagramIsoParallelPair F).symm
    /-
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair Algebraic …
      ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair ( …
    -/
    apply preservesColimit_of_preserves_colimit_cocone (coequalizerCoforkIsColimit _ _)
    /-
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair Algebraic …
      ⊢ CategoryTheory.Limits.IsColimit (AlgebraicGeometry.LocallyRingedSpace.forget …
    -/
    apply (isColimitMapCoconeCoforkEquiv _ _).symm _
    /-
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair Algebraic …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ (Algebraic …
    -/
    dsimp only [forgetToSheafedSpace]
    /-
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f g : Quiver.Hom X Y
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair Algebraic …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ (CategoryT …
    -/
    exact coequalizerIsCoequalizer _ _⟩
    /-
      🎉 no goals
    -/


instance : HasColimits LocallyRingedSpace :=
  has_colimits_of_hasCoequalizers_and_coproducts


noncomputable instance preservesColimits_forgetToSheafedSpace :
    PreservesColimits LocallyRingedSpace.forgetToSheafedSpace.{u} :=
  preservesColimits_of_preservesCoequalizers_and_coproducts _


