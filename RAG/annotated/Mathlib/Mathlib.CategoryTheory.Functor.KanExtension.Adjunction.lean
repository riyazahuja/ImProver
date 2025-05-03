/-- The left Kan extension functor `(C ⥤ H) ⥤ (D ⥤ H)` along a functor `C ⥤ D`. -/
noncomputable def lan : (C ⥤ H) ⥤ (D ⥤ H) where
  obj F := leftKanExtension L F
  map {F₁ F₂} φ := descOfIsLeftKanExtension _ (leftKanExtensionUnit L F₁) _
    (φ ≫ leftKanExtensionUnit L F₂)


/-- The natural transformation `F ⟶ L ⋙ (L.lan).obj G`. -/
noncomputable def lanUnit : (𝟭 (C ⥤ H)) ⟶ L.lan ⋙ (whiskeringLeft C D H).obj L where
  app F := leftKanExtensionUnit L F
                             /-
                               C : Type u_1
                               D : Type u_2
                               inst✝³ : CategoryTheory.Category.{?u.5504, u_1} C
                               inst✝² : CategoryTheory.Category.{?u.5508, u_2} D
                               L : CategoryTheory.Functor C D
                               H : Type u_3
                               inst✝¹ : CategoryTheory.Category.{?u.5545, u_3} H
                               inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
                               F₁ F₂ : CategoryTheory.Functor C H
                               φ : Quiver.Hom F₁ F₂
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
                             -/
  naturality {F₁ F₂} φ := by ext; simp [lan]
                                  /-
                                    🎉 no goals
                                  -/


instance (F : C ⥤ H) : (L.lan.obj F).IsLeftKanExtension (L.lanUnit.app F) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
    F : CategoryTheory.Functor C H
    ⊢ (L.lan.obj F).IsLeftKanExtension (L.lanUnit.app F)
  -/
  dsimp [lan, lanUnit]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
    F : CategoryTheory.Functor C H
    ⊢ (L.leftKanExtension F).IsLeftKanExtension (L.leftKanExtensionUnit F)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If there exists a pointwise left Kan extension of `F` along `L`,
then `L.lan.obj G` is a pointwise left Kan extension of `F`. -/
noncomputable def isPointwiseLeftKanExtensionLeftKanExtensionUnit
    (F : C ⥤ H) [HasPointwiseLeftKanExtension L F] :
    (LeftExtension.mk _ (L.leftKanExtensionUnit F)).IsPointwiseLeftKanExtension :=
  isPointwiseLeftKanExtensionOfIsLeftKanExtension (F := F) _ (leftKanExtensionUnit L F)


/-- If a left Kan extension is pointwise, then evaluating it at an object is isomorphic to
taking a colimit. -/
noncomputable def leftKanExtensionObjIsoColimit [HasLeftKanExtension L F] (X : D) :
    (L.leftKanExtension F).obj X ≅ colimit (proj L X ⋙ F) :=
  LeftExtension.IsPointwiseLeftKanExtensionAt.isoColimit (F := F)
   (isPointwiseLeftKanExtensionLeftKanExtensionUnit L F X)


@[reassoc (attr := simp)]
lemma ι_leftKanExtensionObjIsoColimit_inv [HasLeftKanExtension L F] (X : D)
    (f : CostructuredArrow L X) :
    colimit.ι _ f ≫ (L.leftKanExtensionObjIsoColimit F X).inv =
    (L.leftKanExtensionUnit F).app f.left ≫ (L.leftKanExtension F).map f.hom := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_5, u_3} H
    F : CategoryTheory.Functor C H
    inst✝¹ : L.HasPointwiseLeftKanExtension F
    inst✝ : L.HasLeftKanExtension F
    X : D
    f : CategoryTheory.CostructuredArrow L X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
  -/
  simp [leftKanExtensionObjIsoColimit, lanUnit]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_leftKanExtensionObjIsoColimit_hom (X : D) (f : CostructuredArrow L X) :
    (L.leftKanExtensionUnit F).app f.left ≫ (L.leftKanExtension F).map f.hom ≫
      (L.leftKanExtensionObjIsoColimit F X).hom =
    colimit.ι (proj L X ⋙ F) f :=
  LeftExtension.IsPointwiseLeftKanExtensionAt.ι_isoColimit_hom (F := F)
    (isPointwiseLeftKanExtensionLeftKanExtensionUnit L F X) f


lemma leftKanExtensionUnit_leftKanExtension_map_leftKanExtensionObjIsoColimit_hom (X : D)
    (f : CostructuredArrow L X) :
    (leftKanExtensionUnit L F).app f.left ≫ (leftKanExtension L F).map f.hom ≫
       (L.leftKanExtensionObjIsoColimit F X).hom =
    colimit.ι (proj L X ⋙ F) f :=
  LeftExtension.IsPointwiseLeftKanExtensionAt.ι_isoColimit_hom (F := F)
    (isPointwiseLeftKanExtensionLeftKanExtensionUnit L F X) f


@[reassoc (attr := simp)]
lemma leftKanExtensionUnit_leftKanExtensionObjIsoColimit_hom (X : C) :
    (L.leftKanExtensionUnit F).app X ≫ (L.leftKanExtensionObjIsoColimit F (L.obj X)).hom =
    colimit.ι (proj L (L.obj X) ⋙ F) (CostructuredArrow.mk (𝟙 _)) := by
  simpa using leftKanExtensionUnit_leftKanExtension_map_leftKanExtensionObjIsoColimit_hom L F
    (L.obj X) (CostructuredArrow.mk (𝟙 _))


@[instance]
theorem hasColimit_map_comp_ι_comp_grotendieckProj {X Y : D} (f : X ⟶ Y) :
    HasColimit ((functor L).map f ⋙ Grothendieck.ι (functor L) Y ⋙ grothendieckProj L ⋙ F) :=
  hasColimitOfIso (isoWhiskerRight (mapCompιCompGrothendieckProj L f) F)


/-- The left Kan extension of `F : C ⥤ H` along a functor `L : C ⥤ D` is isomorphic to the
fiberwise colimit of the projection functor on the Grothendieck construction of the costructured
arrow category composed with `F`. -/
@[simps!]
noncomputable def leftKanExtensionIsoFiberwiseColimit [HasLeftKanExtension L F] :
    leftKanExtension L F ≅ fiberwiseColimit (grothendieckProj L ⋙ F) :=
  letI : ∀ X, HasColimit (Grothendieck.ι (functor L) X ⋙ grothendieckProj L ⋙ F) :=
      fun X => hasColimitOfIso <| Iso.symm <| isoWhiskerRight (eqToIso ((functor L).map_id X)) _ ≪≫
      Functor.leftUnitor (Grothendieck.ι (functor L) X ⋙ grothendieckProj L ⋙ F)
  Iso.symm <| NatIso.ofComponents
    (fun X => HasColimit.isoOfNatIso (isoWhiskerRight (ιCompGrothendieckProj L X) F) ≪≫
      (leftKanExtensionObjIsoColimit L F X).symm)
                                 /-
                                   C : Type u_1
                                   D : Type u_2
                                   inst✝⁴ : CategoryTheory.Category.{?u.29815, u_1} C
                                   inst✝³ : CategoryTheory.Category.{?u.29819, u_2} D
                                   L : CategoryTheory.Functor C D
                                   H : Type u_3
                                   inst✝² : CategoryTheory.Category.{?u.29856, u_3} H
                                   F : CategoryTheory.Functor C H
                                   inst✝¹ : L.HasPointwiseLeftKanExtension F
                                   inst✝ : L.HasLeftKanExtension F
                                   this : ∀ (X : D), CategoryTheory.Limits.HasColimit ((CategoryTheory.Grothendie …
                                   X✝ Y✝ : D
                                   f : Quiver.Hom X✝ Y✝
                                   ⊢ ∀ (j : ↑((CategoryTheory.CostructuredArrow.functor L).obj X✝)), Eq (Category …
                                 -/
    fun f => colimit.hom_ext (by simp)
                                 /-
                                   🎉 no goals
                                 -/


variable (H) in
/-- The left Kan extension functor `L.Lan` is left adjoint to the precomposition by `L`. -/
noncomputable def lanAdjunction : L.lan ⊣ (whiskeringLeft C D H).obj L :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun F G => homEquivOfIsLeftKanExtension _ (L.lanUnit.app F) G
      homEquiv_naturality_left_symm := fun {F₁ F₂ G} f α =>
        hom_ext_of_isLeftKanExtension _ (L.lanUnit.app F₁) _ _ (by
          /-
            C : Type u_1
            D : Type u_2
            inst✝³ : CategoryTheory.Category.{?u.55291, u_1} C
            inst✝² : CategoryTheory.Category.{?u.55295, u_2} D
            L : CategoryTheory.Functor C D
            H : Type u_3
            inst✝¹ : CategoryTheory.Category.{?u.55332, u_3} H
            inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
            F₁ F₂ : CategoryTheory.Functor C H
            G : CategoryTheory.Functor D H
            f : Quiver.Hom F₁ F₂
            α : Quiver.Hom F₂ (((CategoryTheory.whiskeringLeft C D H).obj L).obj G)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.lanUnit.app F₁) (CategoryTheory.wh …
          -/
          ext X
          /-
            case w.h
            C : Type u_1
            D : Type u_2
            inst✝³ : CategoryTheory.Category.{?u.55291, u_1} C
            inst✝² : CategoryTheory.Category.{?u.55295, u_2} D
            L : CategoryTheory.Functor C D
            H : Type u_3
            inst✝¹ : CategoryTheory.Category.{?u.55332, u_3} H
            inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
            F₁ F₂ : CategoryTheory.Functor C H
            G : CategoryTheory.Functor D H
            f : Quiver.Hom F₁ F₂
            α : Quiver.Hom F₂ (((CategoryTheory.whiskeringLeft C D H).obj L).obj G)
            X : C
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (L.lanUnit.app F₁) (CategoryTheory.w …
          -/
          dsimp [homEquivOfIsLeftKanExtension]
          /-
            case w.h
            C : Type u_1
            D : Type u_2
            inst✝³ : CategoryTheory.Category.{?u.55291, u_1} C
            inst✝² : CategoryTheory.Category.{?u.55295, u_2} D
            L : CategoryTheory.Functor C D
            H : Type u_3
            inst✝¹ : CategoryTheory.Category.{?u.55332, u_3} H
            inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
            F₁ F₂ : CategoryTheory.Functor C H
            G : CategoryTheory.Functor D H
            f : Quiver.Hom F₁ F₂
            α : Quiver.Hom F₂ (((CategoryTheory.whiskeringLeft C D H).obj L).obj G)
            X : C
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((L.lanUnit.app F₁).app X) (((L.lan.o …
          -/
          rw [descOfIsLeftKanExtension_fac_app, NatTrans.comp_app, ← assoc]
          /-
            case w.h
            C : Type u_1
            D : Type u_2
            inst✝³ : CategoryTheory.Category.{?u.55291, u_1} C
            inst✝² : CategoryTheory.Category.{?u.55295, u_2} D
            L : CategoryTheory.Functor C D
            H : Type u_3
            inst✝¹ : CategoryTheory.Category.{?u.55332, u_3} H
            inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
            F₁ F₂ : CategoryTheory.Functor C H
            G : CategoryTheory.Functor D H
            f : Quiver.Hom F₁ F₂
            α : Quiver.Hom F₂ (((CategoryTheory.whiskeringLeft C D H).obj L).obj G)
            X : C
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app X) (α.app X)) (CategoryTheory. …
          -/
          have h := congr_app (L.lanUnit.naturality f) X
          /-
            case w.h
            C : Type u_1
            D : Type u_2
            inst✝³ : CategoryTheory.Category.{?u.55291, u_1} C
            inst✝² : CategoryTheory.Category.{?u.55295, u_2} D
            L : CategoryTheory.Functor C D
            H : Type u_3
            inst✝¹ : CategoryTheory.Category.{?u.55332, u_3} H
            inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
            F₁ F₂ : CategoryTheory.Functor C H
            G : CategoryTheory.Functor D H
            f : Quiver.Hom F₁ F₂
            α : Quiver.Hom F₂ (((CategoryTheory.whiskeringLeft C D H).obj L).obj G)
            X : C
            h : Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Categ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app X) (α.app X)) (CategoryTheory. …
          -/
          dsimp at h ⊢
          /-
            case w.h
            C : Type u_1
            D : Type u_2
            inst✝³ : CategoryTheory.Category.{?u.55291, u_1} C
            inst✝² : CategoryTheory.Category.{?u.55295, u_2} D
            L : CategoryTheory.Functor C D
            H : Type u_3
            inst✝¹ : CategoryTheory.Category.{?u.55332, u_3} H
            inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
            F₁ F₂ : CategoryTheory.Functor C H
            G : CategoryTheory.Functor D H
            f : Quiver.Hom F₁ F₂
            α : Quiver.Hom F₂ (((CategoryTheory.whiskeringLeft C D H).obj L).obj G)
            X : C
            h : Eq (CategoryTheory.CategoryStruct.comp (f.app X) ((L.lanUnit.app F₂).app X …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app X) (α.app X)) (CategoryTheory. …
          -/
          rw [← h, assoc, descOfIsLeftKanExtension_fac_app] )
          /-
            🎉 no goals
          -/
      homEquiv_naturality_right := fun {F G₁ G₂} β f => by
        /-
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{?u.55291, u_1} C
          inst✝² : CategoryTheory.Category.{?u.55295, u_2} D
          L : CategoryTheory.Functor C D
          H : Type u_3
          inst✝¹ : CategoryTheory.Category.{?u.55332, u_3} H
          inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
          F : CategoryTheory.Functor C H
          G₁ G₂ : CategoryTheory.Functor D H
          β : Quiver.Hom (L.lan.obj F) G₁
          f : Quiver.Hom G₁ G₂
          ⊢ Eq (((fun F G => (L.lan.obj F).homEquivOfIsLeftKanExtension (L.lanUnit.app F …
        -/
        dsimp [homEquivOfIsLeftKanExtension]
        /-
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{?u.55291, u_1} C
          inst✝² : CategoryTheory.Category.{?u.55295, u_2} D
          L : CategoryTheory.Functor C D
          H : Type u_3
          inst✝¹ : CategoryTheory.Category.{?u.55332, u_3} H
          inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
          F : CategoryTheory.Functor C H
          G₁ G₂ : CategoryTheory.Functor D H
          β : Quiver.Hom (L.lan.obj F) G₁
          f : Quiver.Hom G₁ G₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.lanUnit.app F) (CategoryTheory.Cat …
        -/
        rw [assoc] }
        /-
          🎉 no goals
        -/


variable (H) in
@[simp]
lemma lanAdjunction_unit : (L.lanAdjunction H).unit = L.lanUnit := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
    ⊢ Eq (L.lanAdjunction H).unit L.lanUnit
  -/
  ext F : 2
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
    F : CategoryTheory.Functor C H
    ⊢ Eq ((L.lanAdjunction H).unit.app F) (L.lanUnit.app F)
  -/
  dsimp [lanAdjunction, homEquivOfIsLeftKanExtension]
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
    F : CategoryTheory.Functor C H
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.lanUnit.app F) (CategoryTheory.Cat …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma lanAdjunction_counit_app (G : D ⥤ H) :
    (L.lanAdjunction H).counit.app G =
      descOfIsLeftKanExtension (L.lan.obj (L ⋙ G)) (L.lanUnit.app (L ⋙ G)) G (𝟙 (L ⋙ G)) :=
  rfl


@[reassoc (attr := simp)]
lemma lanUnit_app_whiskerLeft_lanAdjunction_counit_app (G : D ⥤ H) :
    L.lanUnit.app (L ⋙ G) ≫ whiskerLeft L ((L.lanAdjunction H).counit.app G) = 𝟙 (L ⋙ G) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
    G : CategoryTheory.Functor D H
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.lanUnit.app (L.comp G)) (CategoryT …
  -/
  simp [lanAdjunction_counit_app]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma lanUnit_app_app_lanAdjunction_counit_app_app (G : D ⥤ H) (X : C) :
    (L.lanUnit.app (L ⋙ G)).app X ≫ ((L.lanAdjunction H).counit.app G).app (L.obj X) = 𝟙 _ :=
  congr_app (L.lanUnit_app_whiskerLeft_lanAdjunction_counit_app G) X


lemma isIso_lanAdjunction_counit_app_iff (G : D ⥤ H) :
    IsIso ((L.lanAdjunction H).counit.app G) ↔ G.IsLeftKanExtension (𝟙 (L ⋙ G)) :=
                                                                /-
                                                                  C : Type u_1
                                                                  D : Type u_2
                                                                  inst✝³ : CategoryTheory.Category.{u_6, u_1} C
                                                                  inst✝² : CategoryTheory.Category.{u_4, u_2} D
                                                                  L : CategoryTheory.Functor C D
                                                                  H : Type u_3
                                                                  inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
                                                                  inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
                                                                  G : CategoryTheory.Functor D H
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.lanUnit.app (L.comp G)) (CategoryT …
                                                                -/
  (isLeftKanExtension_iff_isIso _ (L.lanUnit.app (L ⋙ G)) _ (by simp)).symm
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- Composing the left Kan extension of `L : C ⥤ D` with `colim` on shapes `D` is isomorphic
to `colim` on shapes `C`. -/
@[simps!]
noncomputable def lanCompColimIso [HasColimitsOfShape C H] [HasColimitsOfShape D H] :
    L.lan ⋙ colim ≅ colim (C := H) :=
  Iso.symm <| NatIso.ofComponents
    (fun G ↦ (colimitIsoOfIsLeftKanExtension _ (L.lanUnit.app G)).symm)
    (fun f ↦ colimit.hom_ext (fun i ↦ by
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{?u.78610, u_1} C
        inst✝⁴ : CategoryTheory.Category.{?u.78614, u_2} D
        L : CategoryTheory.Functor C D
        H : Type u_3
        inst✝³ : CategoryTheory.Category.{?u.78651, u_3} H
        inst✝² : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape C H
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape D H
        X✝ Y✝ : CategoryTheory.Functor C H
        f : Quiver.Hom X✝ Y✝
        i : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι X✝ i …
      -/
      dsimp
      rw [ι_colimMap_assoc, ι_colimitIsoOfIsLeftKanExtension_inv,
        ι_colimitIsoOfIsLeftKanExtension_inv_assoc, ι_colimMap, ← assoc, ← assoc]
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{?u.78610, u_1} C
        inst✝⁴ : CategoryTheory.Category.{?u.78614, u_2} D
        L : CategoryTheory.Functor C D
        H : Type u_3
        inst✝³ : CategoryTheory.Category.{?u.78651, u_3} H
        inst✝² : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape C H
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape D H
        X✝ Y✝ : CategoryTheory.Functor C H
        f : Quiver.Hom X✝ Y✝
        i : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      congr 1
      /-
        case e_a
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{?u.78610, u_1} C
        inst✝⁴ : CategoryTheory.Category.{?u.78614, u_2} D
        L : CategoryTheory.Functor C D
        H : Type u_3
        inst✝³ : CategoryTheory.Category.{?u.78651, u_3} H
        inst✝² : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape C H
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape D H
        X✝ Y✝ : CategoryTheory.Functor C H
        f : Quiver.Hom X✝ Y✝
        i : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app i) ((L.lanUnit.app Y✝).app i)) …
      -/
      exact congr_app (L.lanUnit.naturality f) i))
      /-
        🎉 no goals
      -/


instance : HasColimit (CostructuredArrow.grothendieckProj L ⋙ G) :=
  hasColimit_of_hasColimit_fiberwiseColimit_of_hasColimit _


/-- If `G : C ⥤ H` admits a left Kan extension along a functor `L : C ⥤ D` and `H` has colimits of
shape `C` and `D`, then the colimit of `G` is isomorphic to the colimit of a canonical functor
`Grothendieck (CostructuredArrow.functor L) ⥤ H` induced by `L` and `G`. -/
noncomputable def colimitIsoColimitGrothendieck :
    colimit G ≅ colimit (CostructuredArrow.grothendieckProj L ⋙ G) := calc
  colimit G
    ≅ colimit (leftKanExtension L G) :=
        (colimitIsoOfIsLeftKanExtension _ (L.leftKanExtensionUnit G)).symm
  _ ≅ colimit (fiberwiseColimit (CostructuredArrow.grothendieckProj L ⋙ G)) :=
        HasColimit.isoOfNatIso (leftKanExtensionIsoFiberwiseColimit L G)
  _ ≅ colimit (CostructuredArrow.grothendieckProj L ⋙ G) :=
        colimitFiberwiseColimitIso _


@[reassoc (attr := simp)]
lemma ι_colimitIsoColimitGrothendieck_inv (X : Grothendieck (CostructuredArrow.functor L)) :
    colimit.ι (CostructuredArrow.grothendieckProj L ⋙ G) X ≫
      (colimitIsoColimitGrothendieck L G).inv =
    colimit.ι G ((CostructuredArrow.proj L X.base).obj X.fiber) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_6, u_3} H
    G : CategoryTheory.Functor C H
    inst✝² : L.HasPointwiseLeftKanExtension G
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape D H
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape C H
    X : CategoryTheory.Grothendieck (CategoryTheory.CostructuredArrow.functor L)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
  -/
  simp [colimitIsoColimitGrothendieck]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_colimitIsoColimitGrothendieck_hom (X : C) :
    colimit.ι G X ≫ (colimitIsoColimitGrothendieck L G).hom =
    colimit.ι (CostructuredArrow.grothendieckProj L ⋙ G) ⟨L.obj X, .mk (𝟙 _)⟩ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} H
    G : CategoryTheory.Functor C H
    inst✝² : L.HasPointwiseLeftKanExtension G
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape D H
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape C H
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι G X) …
  -/
  rw [← Iso.eq_comp_inv]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} H
    G : CategoryTheory.Functor C H
    inst✝² : L.HasPointwiseLeftKanExtension G
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape D H
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape C H
    X : C
    ⊢ Eq (CategoryTheory.Limits.colimit.ι G X) (CategoryTheory.CategoryStruct.comp …
  -/
  exact (ι_colimitIsoColimitGrothendieck_inv L G ⟨L.obj X, .mk (𝟙 _)⟩).symm
  /-
    🎉 no goals
  -/


instance (F : C ⥤ H) (X : C) [HasPointwiseLeftKanExtension L F]
    [∀ (F : C ⥤ H), HasLeftKanExtension L F] :
    IsIso ((L.lanUnit.app F).app X) :=
  (isPointwiseLeftKanExtensionLeftKanExtensionUnit L F (L.obj X)).isIso_hom_app


instance (F : C ⥤ H) [HasPointwiseLeftKanExtension L F]
    [∀ (F : C ⥤ H), HasLeftKanExtension L F] :
    IsIso (L.lanUnit.app F) :=
  NatIso.isIso_of_isIso_app _


instance coreflective [∀ (F : C ⥤ H), HasPointwiseLeftKanExtension L F] :
    IsIso (L.lanUnit (H := H)) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_3} H
    inst✝² : L.Full
    inst✝¹ : L.Faithful
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasPointwiseLeftKanExtension F
    ⊢ CategoryTheory.IsIso L.lanUnit
  -/
  apply NatIso.isIso_of_isIso_app _
  /-
    🎉 no goals
  -/


instance (F : C ⥤ H) [HasPointwiseLeftKanExtension L F]
    [∀ (F : C ⥤ H), HasLeftKanExtension L F] :
    IsIso ((L.lanAdjunction H).unit.app F) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_5, u_3} H
    inst✝³ : L.Full
    inst✝² : L.Faithful
    F : CategoryTheory.Functor C H
    inst✝¹ : L.HasPointwiseLeftKanExtension F
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
    ⊢ CategoryTheory.IsIso ((L.lanAdjunction H).unit.app F)
  -/
  rw [lanAdjunction_unit]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_5, u_3} H
    inst✝³ : L.Full
    inst✝² : L.Faithful
    F : CategoryTheory.Functor C H
    inst✝¹ : L.HasPointwiseLeftKanExtension F
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasLeftKanExtension F
    ⊢ CategoryTheory.IsIso (L.lanUnit.app F)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance coreflective' [∀ (F : C ⥤ H), HasPointwiseLeftKanExtension L F] :
    IsIso (L.lanAdjunction H).unit := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_3} H
    inst✝² : L.Full
    inst✝¹ : L.Faithful
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasPointwiseLeftKanExtension F
    ⊢ CategoryTheory.IsIso (L.lanAdjunction H).unit
  -/
  apply NatIso.isIso_of_isIso_app _
  /-
    🎉 no goals
  -/


/-- The right Kan extension functor `(C ⥤ H) ⥤ (D ⥤ H)` along a functor `C ⥤ D`. -/
noncomputable def ran : (C ⥤ H) ⥤ (D ⥤ H) where
  obj F := rightKanExtension L F
  map {F₁ F₂} φ := liftOfIsRightKanExtension _ (rightKanExtensionCounit L F₂) _
    (rightKanExtensionCounit L F₁ ≫ φ)


/-- The natural transformation `L ⋙ (L.lan).obj G ⟶ L`. -/
noncomputable def ranCounit : L.ran ⋙ (whiskeringLeft C D H).obj L ⟶ (𝟭 (C ⥤ H)) where
  app F := rightKanExtensionCounit L F
                             /-
                               C : Type u_1
                               D : Type u_2
                               inst✝³ : CategoryTheory.Category.{?u.128399, u_1} C
                               inst✝² : CategoryTheory.Category.{?u.128403, u_2} D
                               L : CategoryTheory.Functor C D
                               H : Type u_3
                               inst✝¹ : CategoryTheory.Category.{?u.128440, u_3} H
                               inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
                               F₁ F₂ : CategoryTheory.Functor C H
                               φ : Quiver.Hom F₁ F₂
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((L.ran.comp ((CategoryTheory.whisker …
                             -/
  naturality {F₁ F₂} φ := by ext; simp [ran]
                                  /-
                                    🎉 no goals
                                  -/


instance (F : C ⥤ H) : (L.ran.obj F).IsRightKanExtension (L.ranCounit.app F) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
    F : CategoryTheory.Functor C H
    ⊢ (L.ran.obj F).IsRightKanExtension (L.ranCounit.app F)
  -/
  dsimp [ran, ranCounit]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
    F : CategoryTheory.Functor C H
    ⊢ (L.rightKanExtension F).IsRightKanExtension (L.rightKanExtensionCounit F)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If there exists a pointwise right Kan extension of `F` along `L`,
then `L.ran.obj G` is a pointwise right Kan extension of `F`. -/
noncomputable def isPointwiseRightKanExtensionRanCounit
    (F : C ⥤ H) [HasPointwiseRightKanExtension L F] :
    (RightExtension.mk _ (L.ranCounit.app F)).IsPointwiseRightKanExtension :=
  isPointwiseRightKanExtensionOfIsRightKanExtension (F := F) _ (L.ranCounit.app F)


/-- If a right Kan extension is pointwise, then evaluating it at an object is isomorphic to
taking a limit. -/
noncomputable def ranObjObjIsoLimit (F : C ⥤ H) [HasPointwiseRightKanExtension L F] (X : D) :
    (L.ran.obj F).obj X ≅ limit (StructuredArrow.proj X L ⋙ F) :=
  RightExtension.IsPointwiseRightKanExtensionAt.isoLimit (F := F)
    (isPointwiseRightKanExtensionRanCounit L F X)


@[reassoc (attr := simp)]
lemma ranObjObjIsoLimit_hom_π
    (F : C ⥤ H) [HasPointwiseRightKanExtension L F] (X : D) (f : StructuredArrow X L) :
    (L.ranObjObjIsoLimit F X).hom ≫ limit.π _ f =
    (L.ran.obj F).map f.hom ≫ (L.ranCounit.app F).app f.right := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_5, u_3} H
    inst✝¹ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
    F : CategoryTheory.Functor C H
    inst✝ : L.HasPointwiseRightKanExtension F
    X : D
    f : CategoryTheory.StructuredArrow X L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.ranObjObjIsoLimit F X).hom (Catego …
  -/
  simp [ranObjObjIsoLimit, ran, ranCounit]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ranObjObjIsoLimit_inv_π
    (F : C ⥤ H) [HasPointwiseRightKanExtension L F] (X : D) (f : StructuredArrow X L) :
    (L.ranObjObjIsoLimit F X).inv ≫ (L.ran.obj F).map f.hom ≫ (L.ranCounit.app F).app f.right =
    limit.π _ f :=
  RightExtension.IsPointwiseRightKanExtensionAt.isoLimit_inv_π (F := F)
    (isPointwiseRightKanExtensionRanCounit L F X) f


variable (H) in
/-- The right Kan extension functor `L.ran` is right adjoint to the
precomposition by `L`. -/
noncomputable def ranAdjunction : (whiskeringLeft C D H).obj L ⊣ L.ran :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun F G =>
        (homEquivOfIsRightKanExtension (α := L.ranCounit.app G) _ F).symm
      homEquiv_naturality_right := fun {F G₁ G₂} β f ↦
        hom_ext_of_isRightKanExtension _ (L.ranCounit.app G₂) _ _ (by
        /-
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{?u.148712, u_1} C
          inst✝² : CategoryTheory.Category.{?u.148716, u_2} D
          L : CategoryTheory.Functor C D
          H : Type u_3
          inst✝¹ : CategoryTheory.Category.{?u.148753, u_3} H
          inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
          F : CategoryTheory.Functor D H
          G₁ G₂ : CategoryTheory.Functor C H
          β : Quiver.Hom (((CategoryTheory.whiskeringLeft C D H).obj L).obj F) G₁
          f : Quiver.Hom G₁ G₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L (((fun  …
        -/
        ext X
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{?u.148712, u_1} C
          inst✝² : CategoryTheory.Category.{?u.148716, u_2} D
          L : CategoryTheory.Functor C D
          H : Type u_3
          inst✝¹ : CategoryTheory.Category.{?u.148753, u_3} H
          inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
          F : CategoryTheory.Functor D H
          G₁ G₂ : CategoryTheory.Functor C H
          β : Quiver.Hom (((CategoryTheory.whiskeringLeft C D H).obj L).obj F) G₁
          f : Quiver.Hom G₁ G₂
          X : C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L (((fun …
        -/
        dsimp [homEquivOfIsRightKanExtension]
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{?u.148712, u_1} C
          inst✝² : CategoryTheory.Category.{?u.148716, u_2} D
          L : CategoryTheory.Functor C D
          H : Type u_3
          inst✝¹ : CategoryTheory.Category.{?u.148753, u_3} H
          inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
          F : CategoryTheory.Functor D H
          G₁ G₂ : CategoryTheory.Functor C H
          β : Quiver.Hom (((CategoryTheory.whiskeringLeft C D H).obj L).obj F) G₁
          f : Quiver.Hom G₁ G₂
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((L.ran.obj G₂).liftOfIsRightKanExte …
        -/
        rw [liftOfIsRightKanExtension_fac_app, NatTrans.comp_app, assoc]
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{?u.148712, u_1} C
          inst✝² : CategoryTheory.Category.{?u.148716, u_2} D
          L : CategoryTheory.Functor C D
          H : Type u_3
          inst✝¹ : CategoryTheory.Category.{?u.148753, u_3} H
          inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
          F : CategoryTheory.Functor D H
          G₁ G₂ : CategoryTheory.Functor C H
          β : Quiver.Hom (((CategoryTheory.whiskeringLeft C D H).obj L).obj F) G₁
          f : Quiver.Hom G₁ G₂
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (β.app X) (f.app X)) (CategoryTheory. …
        -/
        /-
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{?u.148712, u_1} C
          inst✝² : CategoryTheory.Category.{?u.148716, u_2} D
          L : CategoryTheory.Functor C D
          H : Type u_3
          inst✝¹ : CategoryTheory.Category.{?u.148753, u_3} H
          inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
          F₁ F₂ : CategoryTheory.Functor D H
          G : CategoryTheory.Functor C H
          β : Quiver.Hom F₁ F₂
          f : Quiver.Hom F₂ (L.ran.obj G)
          ⊢ Eq (((fun F G => ((L.ran.obj G).homEquivOfIsRightKanExtension (L.ranCounit.a …
        -/
        have h := congr_app (L.ranCounit.naturality f) X
        /-
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{?u.148712, u_1} C
          inst✝² : CategoryTheory.Category.{?u.148716, u_2} D
          L : CategoryTheory.Functor C D
          H : Type u_3
          inst✝¹ : CategoryTheory.Category.{?u.148753, u_3} H
          inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
          F₁ F₂ : CategoryTheory.Functor D H
          G : CategoryTheory.Functor C H
          β : Quiver.Hom F₁ F₂
          f : Quiver.Hom F₂ (L.ran.obj G)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{?u.148712, u_1} C
          inst✝² : CategoryTheory.Category.{?u.148716, u_2} D
          L : CategoryTheory.Functor C D
          H : Type u_3
          inst✝¹ : CategoryTheory.Category.{?u.148753, u_3} H
          inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
          F : CategoryTheory.Functor D H
          G₁ G₂ : CategoryTheory.Functor C H
          β : Quiver.Hom (((CategoryTheory.whiskeringLeft C D H).obj L).obj F) G₁
          f : Quiver.Hom G₁ G₂
          X : C
          h : Eq ((CategoryTheory.CategoryStruct.comp ((L.ran.comp ((CategoryTheory.whis …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (β.app X) (f.app X)) (CategoryTheory. …
        -/
        /-
          🎉 no goals
        -/
        dsimp at h ⊢
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{?u.148712, u_1} C
          inst✝² : CategoryTheory.Category.{?u.148716, u_2} D
          L : CategoryTheory.Functor C D
          H : Type u_3
          inst✝¹ : CategoryTheory.Category.{?u.148753, u_3} H
          inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
          F : CategoryTheory.Functor D H
          G₁ G₂ : CategoryTheory.Functor C H
          β : Quiver.Hom (((CategoryTheory.whiskeringLeft C D H).obj L).obj F) G₁
          f : Quiver.Hom G₁ G₂
          X : C
          h : Eq (CategoryTheory.CategoryStruct.comp ((L.ran.map f).app (L.obj X)) ((L.r …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (β.app X) (f.app X)) (CategoryTheory. …
        -/
        rw [h, liftOfIsRightKanExtension_fac_app_assoc])
        /-
          🎉 no goals
        -/
      homEquiv_naturality_left_symm := fun {F₁ F₂ G} β f ↦ by
        dsimp [homEquivOfIsRightKanExtension]
        rw [assoc] }


variable (H) in
@[simp]
lemma ranAdjunction_counit : (L.ranAdjunction H).counit = L.ranCounit := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
    ⊢ Eq (L.ranAdjunction H).counit L.ranCounit
  -/
  ext F : 2
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
    F : CategoryTheory.Functor C H
    ⊢ Eq ((L.ranAdjunction H).counit.app F) (L.ranCounit.app F)
  -/
  dsimp [ranAdjunction, homEquivOfIsRightKanExtension]
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
    F : CategoryTheory.Functor C H
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (L. …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma ranAdjunction_unit_app (G : D ⥤ H) :
    (L.ranAdjunction H).unit.app G =
      liftOfIsRightKanExtension (L.ran.obj (L ⋙ G)) (L.ranCounit.app (L ⋙ G)) G (𝟙 (L ⋙ G)) :=
  rfl


@[reassoc (attr := simp)]
lemma ranCounit_app_whiskerLeft_ranAdjunction_unit_app (G : D ⥤ H) :
    whiskerLeft L ((L.ranAdjunction H).unit.app G) ≫ L.ranCounit.app (L ⋙ G) = 𝟙 (L ⋙ G) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
    G : CategoryTheory.Functor D H
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L ((L.ran …
  -/
  simp [ranAdjunction_unit_app]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ranCounit_app_app_ranAdjunction_unit_app_app (G : D ⥤ H) (X : C) :
    ((L.ranAdjunction H).unit.app G).app (L.obj X) ≫ (L.ranCounit.app (L ⋙ G)).app X = 𝟙 _ :=
  congr_app (L.ranCounit_app_whiskerLeft_ranAdjunction_unit_app G) X


lemma isIso_ranAdjunction_unit_app_iff (G : D ⥤ H) :
    IsIso ((L.ranAdjunction H).unit.app G) ↔ G.IsRightKanExtension (𝟙 (L ⋙ G)) :=
                                                                   /-
                                                                     C : Type u_1
                                                                     D : Type u_2
                                                                     inst✝³ : CategoryTheory.Category.{u_6, u_1} C
                                                                     inst✝² : CategoryTheory.Category.{u_4, u_2} D
                                                                     L : CategoryTheory.Functor C D
                                                                     H : Type u_3
                                                                     inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
                                                                     inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
                                                                     G : CategoryTheory.Functor D H
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L ((L.ran …
                                                                   -/
  (isRightKanExtension_iff_isIso _ (L.ranCounit.app (L ⋙ G)) _ (by simp)).symm
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- Composing the right Kan extension of `L : C ⥤ D` with `lim` on shapes `D` is isomorphic
to `lim` on shapes `C`. -/
@[simps!]
noncomputable def ranCompLimIso (L : C ⥤ D) [∀ (G : C ⥤ H), L.HasRightKanExtension G]
    [HasLimitsOfShape C H] [HasLimitsOfShape D H] : L.ran ⋙ lim ≅ lim (C := H) :=
  NatIso.ofComponents
    (fun G ↦ limitIsoOfIsRightKanExtension _ (L.ranCounit.app G))
    (fun f ↦ limit.hom_ext (fun i ↦ by
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁶ : CategoryTheory.Category.{?u.170733, u_1} C
        inst✝⁵ : CategoryTheory.Category.{?u.170737, u_2} D
        L✝ : CategoryTheory.Functor C D
        H : Type u_3
        inst✝⁴ : CategoryTheory.Category.{?u.170774, u_3} H
        inst✝³ : ∀ (F : CategoryTheory.Functor C H), L✝.HasRightKanExtension F
        L : CategoryTheory.Functor C D
        inst✝² : ∀ (G : CategoryTheory.Functor C H), L.HasRightKanExtension G
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape C H
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape D H
        X✝ Y✝ : CategoryTheory.Functor C H
        f : Quiver.Hom X✝ Y✝
        i : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp
      rw [assoc, assoc, limMap_π, limitIsoOfIsRightKanExtension_hom_π_assoc,
        limitIsoOfIsRightKanExtension_hom_π, limMap_π_assoc]
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁶ : CategoryTheory.Category.{?u.170733, u_1} C
        inst✝⁵ : CategoryTheory.Category.{?u.170737, u_2} D
        L✝ : CategoryTheory.Functor C D
        H : Type u_3
        inst✝⁴ : CategoryTheory.Category.{?u.170774, u_3} H
        inst✝³ : ∀ (F : CategoryTheory.Functor C H), L✝.HasRightKanExtension F
        L : CategoryTheory.Functor C D
        inst✝² : ∀ (G : CategoryTheory.Functor C H), L.HasRightKanExtension G
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape C H
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape D H
        X✝ Y✝ : CategoryTheory.Functor C H
        f : Quiver.Hom X✝ Y✝
        i : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (L.ran …
      -/
      congr 1
      /-
        case e_a
        C : Type u_1
        D : Type u_2
        inst✝⁶ : CategoryTheory.Category.{?u.170733, u_1} C
        inst✝⁵ : CategoryTheory.Category.{?u.170737, u_2} D
        L✝ : CategoryTheory.Functor C D
        H : Type u_3
        inst✝⁴ : CategoryTheory.Category.{?u.170774, u_3} H
        inst✝³ : ∀ (F : CategoryTheory.Functor C H), L✝.HasRightKanExtension F
        L : CategoryTheory.Functor C D
        inst✝² : ∀ (G : CategoryTheory.Functor C H), L.HasRightKanExtension G
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape C H
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape D H
        X✝ Y✝ : CategoryTheory.Functor C H
        f : Quiver.Hom X✝ Y✝
        i : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((L.ran.map f).app (L.obj i)) ((L.ran …
      -/
      exact congr_app (L.ranCounit.naturality f) i))
      /-
        🎉 no goals
      -/


instance (F : C ⥤ H) (X : C) [HasPointwiseRightKanExtension L F]
    [∀ (F : C ⥤ H), HasRightKanExtension L F] :
    IsIso ((L.ranCounit.app F).app X) :=
  (isPointwiseRightKanExtensionRanCounit L F (L.obj X)).isIso_hom_app


instance (F : C ⥤ H) [HasPointwiseRightKanExtension L F]
    [∀ (F : C ⥤ H), HasRightKanExtension L F] :
    IsIso (L.ranCounit.app F) :=
  NatIso.isIso_of_isIso_app _


instance reflective [∀ (F : C ⥤ H), HasPointwiseRightKanExtension L F] :
    IsIso (L.ranCounit (H := H)) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_3} H
    inst✝² : L.Full
    inst✝¹ : L.Faithful
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasPointwiseRightKanExtension F
    ⊢ CategoryTheory.IsIso L.ranCounit
  -/
  apply NatIso.isIso_of_isIso_app _
  /-
    🎉 no goals
  -/


instance (F : C ⥤ H) [HasPointwiseRightKanExtension L F]
    [∀ (F : C ⥤ H), HasRightKanExtension L F] :
    IsIso ((L.ranAdjunction H).counit.app F) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_5, u_3} H
    inst✝³ : L.Full
    inst✝² : L.Faithful
    F : CategoryTheory.Functor C H
    inst✝¹ : L.HasPointwiseRightKanExtension F
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
    ⊢ CategoryTheory.IsIso ((L.ranAdjunction H).counit.app F)
  -/
  rw [ranAdjunction_counit]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_5, u_3} H
    inst✝³ : L.Full
    inst✝² : L.Faithful
    F : CategoryTheory.Functor C H
    inst✝¹ : L.HasPointwiseRightKanExtension F
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasRightKanExtension F
    ⊢ CategoryTheory.IsIso (L.ranCounit.app F)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance reflective' [∀ (F : C ⥤ H), HasPointwiseRightKanExtension L F] :
    IsIso (L.ranAdjunction H).counit := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    L : CategoryTheory.Functor C D
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_3} H
    inst✝² : L.Full
    inst✝¹ : L.Faithful
    inst✝ : ∀ (F : CategoryTheory.Functor C H), L.HasPointwiseRightKanExtension F
    ⊢ CategoryTheory.IsIso (L.ranAdjunction H).counit
  -/
  apply NatIso.isIso_of_isIso_app _
  /-
    🎉 no goals
  -/


