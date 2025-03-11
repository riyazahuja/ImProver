/-- Given a functor `F : D ⥤ C`, this is a predicate on objects `X : C` corresponding
to the domain of definition of the (partial) left adjoint of `F`. -/
def LeftAdjointObjIsDefined (X : C) : Prop := IsCorepresentable (F ⋙ coyoneda.obj (op X))


lemma leftAdjointObjIsDefined_iff (X : C) :
                                                                                    /-
                                                                                      C : Type u₁
                                                                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                      D : Type u₂
                                                                                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                      F : CategoryTheory.Functor D C
                                                                                      X : C
                                                                                      ⊢ Iff (F.LeftAdjointObjIsDefined X) (F.comp (CategoryTheory.coyoneda.obj { uno …
                                                                                    -/
    F.LeftAdjointObjIsDefined X ↔ IsCorepresentable (F ⋙ coyoneda.obj (op X)) := by rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


variable {F} in
lemma leftAdjointObjIsDefined_of_adjunction {G : C ⥤ D} (adj : G ⊣ F) (X : C) :
    F.LeftAdjointObjIsDefined X :=
  (adj.corepresentableBy X).isCorepresentable


/-- The full subcategory where `F.partialLeftAdjoint` shall be defined. -/
abbrev PartialLeftAdjointSource := FullSubcategory F.LeftAdjointObjIsDefined


instance (X : F.PartialLeftAdjointSource) :
    IsCorepresentable (F ⋙ coyoneda.obj (op X.obj)) := X.property


/-- Given `F : D ⥤ C`, this is `F.partialLeftAdjoint` on objects: it sends
`X : C` such that `F.LeftAdjointObjIsDefined X` holds to an object of `D`
which represents the functor `F ⋙ coyoneda.obj (op X.obj)`. -/
noncomputable def partialLeftAdjointObj (X : F.PartialLeftAdjointSource) : D :=
  (F ⋙ coyoneda.obj (op X.obj)).coreprX


/-- Given `F : D ⥤ C`, this is the canonical bijection
`(F.partialLeftAdjointObj X ⟶ Y) ≃ (X.obj ⟶ F.obj Y)`
for all `X : F.PartialLeftAdjointSource` and `Y : D`. -/
noncomputable def partialLeftAdjointHomEquiv {X : F.PartialLeftAdjointSource} {Y : D} :
    (F.partialLeftAdjointObj X ⟶ Y) ≃ (X.obj ⟶ F.obj Y) :=
  (F ⋙ coyoneda.obj (op X.obj)).corepresentableBy.homEquiv


lemma partialLeftAdjointHomEquiv_comp {X : F.PartialLeftAdjointSource} {Y Y' : D}
    (f : F.partialLeftAdjointObj X ⟶ Y) (g : Y ⟶ Y') :
    F.partialLeftAdjointHomEquiv (f ≫ g) =
      F.partialLeftAdjointHomEquiv f ≫ F.map g := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D C
    X : F.PartialLeftAdjointSource
    Y Y' : D
    f : Quiver.Hom (F.partialLeftAdjointObj X) Y
    g : Quiver.Hom Y Y'
    ⊢ Eq (F.partialLeftAdjointHomEquiv (CategoryTheory.CategoryStruct.comp f g)) ( …
  -/
  apply CorepresentableBy.homEquiv_comp
  /-
    🎉 no goals
  -/


/-- Given `F : D ⥤ C`, this is `F.partialLeftAdjoint` on morphisms. -/
noncomputable def partialLeftAdjointMap {X Y : F.PartialLeftAdjointSource}
    (f : X ⟶ Y) : F.partialLeftAdjointObj X ⟶ F.partialLeftAdjointObj Y :=
    F.partialLeftAdjointHomEquiv.symm (f ≫ F.partialLeftAdjointHomEquiv (𝟙 _))


@[simp]
lemma partialLeftAdjointHomEquiv_map {X Y : F.PartialLeftAdjointSource}
    (f : X ⟶ Y) :
    F.partialLeftAdjointHomEquiv (F.partialLeftAdjointMap f) =
         /-
           C : Type u₁
           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
           D : Type u₂
           inst✝ : CategoryTheory.Category.{v₂, u₂} D
           F : CategoryTheory.Functor D C
           X Y : F.PartialLeftAdjointSource
           f : Quiver.Hom X Y
           ⊢ Quiver.Hom X.obj (F.obj (F.partialLeftAdjointObj Y))
         -/
      by exact f ≫ F.partialLeftAdjointHomEquiv (𝟙 _) := by
         /-
           🎉 no goals
         -/
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D C
    X Y : F.PartialLeftAdjointSource
    f : Quiver.Hom X Y
    ⊢ Eq (F.partialLeftAdjointHomEquiv (F.partialLeftAdjointMap f)) (CategoryTheor …
  -/
  simp [partialLeftAdjointMap]
  /-
    🎉 no goals
  -/


lemma partialLeftAdjointHomEquiv_map_comp {X X' : F.PartialLeftAdjointSource} {Y : D}
    (f : X ⟶ X') (g : F.partialLeftAdjointObj X' ⟶ Y) :
    F.partialLeftAdjointHomEquiv (F.partialLeftAdjointMap f ≫ g) =
         /-
           C : Type u₁
           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
           D : Type u₂
           inst✝ : CategoryTheory.Category.{v₂, u₂} D
           F : CategoryTheory.Functor D C
           X X' : F.PartialLeftAdjointSource
           Y : D
           f : Quiver.Hom X X'
           g : Quiver.Hom (F.partialLeftAdjointObj X') Y
           ⊢ Quiver.Hom X.obj (F.obj Y)
         -/
      by exact f ≫ F.partialLeftAdjointHomEquiv g := by
         /-
           🎉 no goals
         -/
  rw [partialLeftAdjointHomEquiv_comp, partialLeftAdjointHomEquiv_map, assoc,
    ← partialLeftAdjointHomEquiv_comp, id_comp]


/-- Given `F : D ⥤ C`, this is the partial adjoint functor `F.PartialLeftAdjointSource ⥤ D`. -/
@[simps]
noncomputable def partialLeftAdjoint : F.PartialLeftAdjointSource ⥤ D where
  obj := F.partialLeftAdjointObj
  map := F.partialLeftAdjointMap
  map_id X := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor D C
      X : F.PartialLeftAdjointSource
      ⊢ Eq ({ obj := F.partialLeftAdjointObj, map := fun {X Y} => F.partialLeftAdjoi …
    -/
    apply F.partialLeftAdjointHomEquiv.injective
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor D C
      X : F.PartialLeftAdjointSource
      ⊢ Eq (F.partialLeftAdjointHomEquiv ({ obj := F.partialLeftAdjointObj, map := f …
    -/
    dsimp
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor D C
      X : F.PartialLeftAdjointSource
      ⊢ Eq (F.partialLeftAdjointHomEquiv (F.partialLeftAdjointMap (CategoryTheory.Ca …
    -/
    rw [partialLeftAdjointHomEquiv_map]
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor D C
      X : F.PartialLeftAdjointSource
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
    -/
    erw [id_comp]
    /-
      🎉 no goals
    -/
  map_comp {X Y Z} f g := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor D C
      X Y Z : F.PartialLeftAdjointSource
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := F.partialLeftAdjointObj, map := fun {X Y} => F.partialLeftAdjoi …
    -/
    apply F.partialLeftAdjointHomEquiv.injective
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor D C
      X Y Z : F.PartialLeftAdjointSource
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (F.partialLeftAdjointHomEquiv ({ obj := F.partialLeftAdjointObj, map := f …
    -/
    dsimp
    rw [partialLeftAdjointHomEquiv_map, partialLeftAdjointHomEquiv_comp,
      partialLeftAdjointHomEquiv_map, assoc]
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor D C
      X Y Z : F.PartialLeftAdjointSource
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    erw [assoc]
    rw [← F.partialLeftAdjointHomEquiv_comp, id_comp,
      partialLeftAdjointHomEquiv_map]


lemma isRightAdjoint_of_leftAdjointObjIsDefined_eq_top
    (h : F.LeftAdjointObjIsDefined = ⊤) : F.IsRightAdjoint := by
  replace h : ∀ X, IsCorepresentable (F ⋙ coyoneda.obj (op X)) := fun X ↦ by
    simp only [← leftAdjointObjIsDefined_iff, h, Pi.top_apply, Prop.top_eq_true]
  exact (Adjunction.adjunctionOfEquivLeft
    (fun X Y ↦ (F ⋙ coyoneda.obj (op X)).corepresentableBy.homEquiv)
    (fun X Y Y' g f ↦ by apply CorepresentableBy.homEquiv_comp)).isRightAdjoint


variable (F) in
lemma isRightAdjoint_iff_leftAdjointObjIsDefined_eq_top :
    F.IsRightAdjoint ↔ F.LeftAdjointObjIsDefined = ⊤ := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D C
    ⊢ Iff F.IsRightAdjoint (Eq F.LeftAdjointObjIsDefined Top.top)
  -/
  refine ⟨fun h ↦ ?_, isRightAdjoint_of_leftAdjointObjIsDefined_eq_top⟩
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D C
    h : F.IsRightAdjoint
    ⊢ Eq F.LeftAdjointObjIsDefined Top.top
  -/
  ext X
  simpa only [Pi.top_apply, Prop.top_eq_true, iff_true]
    using leftAdjointObjIsDefined_of_adjunction (Adjunction.ofIsRightAdjoint F) X


/-- Auxiliary definition for `leftAdjointObjIsDefined_of_isColimit`. -/
noncomputable def corepresentableByCompCoyonedaObjOfIsColimit {J : Type*} [Category J]
    {R : J ⥤ F.PartialLeftAdjointSource}
    {c : Cocone (R ⋙ fullSubcategoryInclusion _)} (hc : IsColimit c)
    {c' : Cocone (R ⋙ F.partialLeftAdjoint)} (hc' : IsColimit c') :
    (F ⋙ coyoneda.obj (op c.pt)).CorepresentableBy c'.pt where
  homEquiv {Y} :=
    { toFun := fun f ↦ hc.desc (Cocone.mk _
        { app := fun j ↦ F.partialLeftAdjointHomEquiv (c'.ι.app j ≫ f)
          naturality := fun j j' φ ↦ by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              F : CategoryTheory.Functor D C
              J : Type u_1
              inst✝ : CategoryTheory.Category.{?u.16473, u_1} J
              R : CategoryTheory.Functor J F.PartialLeftAdjointSource
              c : CategoryTheory.Limits.Cocone (R.comp (CategoryTheory.fullSubcategoryInclus …
              hc : CategoryTheory.Limits.IsColimit c
              c' : CategoryTheory.Limits.Cocone (R.comp F.partialLeftAdjoint)
              hc' : CategoryTheory.Limits.IsColimit c'
              Y : D
              f : Quiver.Hom c'.pt Y
              j j' : J
              φ : Quiver.Hom j j'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((R.comp (CategoryTheory.fullSubcateg …
            -/
            dsimp
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              F : CategoryTheory.Functor D C
              J : Type u_1
              inst✝ : CategoryTheory.Category.{?u.16473, u_1} J
              R : CategoryTheory.Functor J F.PartialLeftAdjointSource
              c : CategoryTheory.Limits.Cocone (R.comp (CategoryTheory.fullSubcategoryInclus …
              hc : CategoryTheory.Limits.IsColimit c
              c' : CategoryTheory.Limits.Cocone (R.comp F.partialLeftAdjoint)
              hc' : CategoryTheory.Limits.IsColimit c'
              Y : D
              f : Quiver.Hom c'.pt Y
              j j' : J
              φ : Quiver.Hom j j'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map φ) (F.partialLeftAdjointHomEqu …
            -/
            rw [comp_id, ← c'.w φ, ← partialLeftAdjointHomEquiv_map_comp, assoc]
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              F : CategoryTheory.Functor D C
              J : Type u_1
              inst✝ : CategoryTheory.Category.{?u.16473, u_1} J
              R : CategoryTheory.Functor J F.PartialLeftAdjointSource
              c : CategoryTheory.Limits.Cocone (R.comp (CategoryTheory.fullSubcategoryInclus …
              hc : CategoryTheory.Limits.IsColimit c
              c' : CategoryTheory.Limits.Cocone (R.comp F.partialLeftAdjoint)
              hc' : CategoryTheory.Limits.IsColimit c'
              Y : D
              f : Quiver.Hom c'.pt Y
              j j' : J
              φ : Quiver.Hom j j'
              ⊢ Eq (F.partialLeftAdjointHomEquiv (CategoryTheory.CategoryStruct.comp (F.part …
            -/
            dsimp })
            /-
              🎉 no goals
            -/
      invFun := fun g ↦ hc'.desc (Cocone.mk _
        { app := fun j ↦ F.partialLeftAdjointHomEquiv.symm (c.ι.app j ≫ g)
          naturality := fun j j' φ ↦ by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              F : CategoryTheory.Functor D C
              J : Type u_1
              inst✝ : CategoryTheory.Category.{?u.16473, u_1} J
              R : CategoryTheory.Functor J F.PartialLeftAdjointSource
              c : CategoryTheory.Limits.Cocone (R.comp (CategoryTheory.fullSubcategoryInclus …
              hc : CategoryTheory.Limits.IsColimit c
              c' : CategoryTheory.Limits.Cocone (R.comp F.partialLeftAdjoint)
              hc' : CategoryTheory.Limits.IsColimit c'
              Y : D
              g : (F.comp (CategoryTheory.coyoneda.obj { unop := c.pt })).obj Y
              j j' : J
              φ : Quiver.Hom j j'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((R.comp F.partialLeftAdjoint).map φ) …
            -/
            apply F.partialLeftAdjointHomEquiv.injective
            /-
              case a
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              F : CategoryTheory.Functor D C
              J : Type u_1
              inst✝ : CategoryTheory.Category.{?u.16473, u_1} J
              R : CategoryTheory.Functor J F.PartialLeftAdjointSource
              c : CategoryTheory.Limits.Cocone (R.comp (CategoryTheory.fullSubcategoryInclus …
              hc : CategoryTheory.Limits.IsColimit c
              c' : CategoryTheory.Limits.Cocone (R.comp F.partialLeftAdjoint)
              hc' : CategoryTheory.Limits.IsColimit c'
              Y : D
              g : (F.comp (CategoryTheory.coyoneda.obj { unop := c.pt })).obj Y
              j j' : J
              φ : Quiver.Hom j j'
              ⊢ Eq (F.partialLeftAdjointHomEquiv (CategoryTheory.CategoryStruct.comp ((R.com …
            -/
            have := c.w φ
            /-
              case a
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              F : CategoryTheory.Functor D C
              J : Type u_1
              inst✝ : CategoryTheory.Category.{?u.16473, u_1} J
              R : CategoryTheory.Functor J F.PartialLeftAdjointSource
              c : CategoryTheory.Limits.Cocone (R.comp (CategoryTheory.fullSubcategoryInclus …
              hc : CategoryTheory.Limits.IsColimit c
              c' : CategoryTheory.Limits.Cocone (R.comp F.partialLeftAdjoint)
              hc' : CategoryTheory.Limits.IsColimit c'
              Y : D
              g : (F.comp (CategoryTheory.coyoneda.obj { unop := c.pt })).obj Y
              j j' : J
              φ : Quiver.Hom j j'
              this : Eq (CategoryTheory.CategoryStruct.comp ((R.comp (CategoryTheory.fullSub …
              ⊢ Eq (F.partialLeftAdjointHomEquiv (CategoryTheory.CategoryStruct.comp ((R.com …
            -/
            dsimp at this ⊢
            rw [comp_id, Equiv.apply_symm_apply, partialLeftAdjointHomEquiv_map_comp,
              Equiv.apply_symm_apply, reassoc_of% this] })
                                                  /-
                                                    C : Type u₁
                                                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                    D : Type u₂
                                                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                    F : CategoryTheory.Functor D C
                                                    J : Type u_1
                                                    inst✝ : CategoryTheory.Category.{?u.16473, u_1} J
                                                    R : CategoryTheory.Functor J F.PartialLeftAdjointSource
                                                    c : CategoryTheory.Limits.Cocone (R.comp (CategoryTheory.fullSubcategoryInclus …
                                                    hc : CategoryTheory.Limits.IsColimit c
                                                    c' : CategoryTheory.Limits.Cocone (R.comp F.partialLeftAdjoint)
                                                    hc' : CategoryTheory.Limits.IsColimit c'
                                                    Y : D
                                                    f : Quiver.Hom c'.pt Y
                                                    j : J
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (c'.ι.app j) ((fun g => hc'.desc { pt …
                                                  -/
      left_inv := fun f ↦ hc'.hom_ext (fun j ↦ by simp)
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    C : Type u₁
                                                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                    D : Type u₂
                                                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                    F : CategoryTheory.Functor D C
                                                    J : Type u_1
                                                    inst✝ : CategoryTheory.Category.{?u.16473, u_1} J
                                                    R : CategoryTheory.Functor J F.PartialLeftAdjointSource
                                                    c : CategoryTheory.Limits.Cocone (R.comp (CategoryTheory.fullSubcategoryInclus …
                                                    hc : CategoryTheory.Limits.IsColimit c
                                                    c' : CategoryTheory.Limits.Cocone (R.comp F.partialLeftAdjoint)
                                                    hc' : CategoryTheory.Limits.IsColimit c'
                                                    Y : D
                                                    g : (F.comp (CategoryTheory.coyoneda.obj { unop := c.pt })).obj Y
                                                    j : J
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) ((fun f => hc.desc { pt : …
                                                  -/
      right_inv := fun g ↦ hc.hom_ext (fun j ↦ by simp) }
                                                  /-
                                                    🎉 no goals
                                                  -/
  homEquiv_comp {Y Y'} g f := hc.hom_ext (fun j ↦ by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor D C
      J : Type u_1
      inst✝ : CategoryTheory.Category.{?u.16473, u_1} J
      R : CategoryTheory.Functor J F.PartialLeftAdjointSource
      c : CategoryTheory.Limits.Cocone (R.comp (CategoryTheory.fullSubcategoryInclus …
      hc : CategoryTheory.Limits.IsColimit c
      c' : CategoryTheory.Limits.Cocone (R.comp F.partialLeftAdjoint)
      hc' : CategoryTheory.Limits.IsColimit c'
      Y Y' : D
      g : Quiver.Hom Y Y'
      f : Quiver.Hom c'.pt Y
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) ((fun {Y} => { toFun := f …
    -/
    dsimp
    simp only [IsColimit.fac, IsColimit.fac_assoc, partialLeftAdjointHomEquiv_comp,
      F.map_comp, assoc] )


lemma leftAdjointObjIsDefined_of_isColimit {J : Type*} [Category J] {R : J ⥤ C} {c : Cocone R}
    (hc : IsColimit c) [HasColimitsOfShape J D]
    (h : ∀ (j : J), F.LeftAdjointObjIsDefined (R.obj j)) :
    F.LeftAdjointObjIsDefined c.pt :=
  (corepresentableByCompCoyonedaObjOfIsColimit
    (R := FullSubcategory.lift _ R h) hc (colimit.isColimit _)).isCorepresentable


lemma leftAdjointObjIsDefined_colimit {J : Type*} [Category J] (R : J ⥤ C)
    [HasColimit R] [HasColimitsOfShape J D]
    (h : ∀ (j : J), F.LeftAdjointObjIsDefined (R.obj j)) :
    F.LeftAdjointObjIsDefined (colimit R) :=
  leftAdjointObjIsDefined_of_isColimit (colimit.isColimit R) h


