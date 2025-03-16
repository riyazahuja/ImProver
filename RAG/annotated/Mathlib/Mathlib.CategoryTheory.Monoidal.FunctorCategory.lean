/-- (An auxiliary definition for `functorCategoryMonoidal`.)
Tensor product of functors `C ⥤ D`, when `D` is monoidal.
 -/
@[simps]
def tensorObj : C ⥤ D where
  obj X := F.obj X ⊗ G.obj X
  map f := F.map f ⊗ G.map f


/-- (An auxiliary definition for `functorCategoryMonoidal`.)
Tensor product of natural transformations into `D`, when `D` is monoidal.
-/
@[simps]
def tensorHom : tensorObj F F' ⟶ tensorObj G G' where
  app X := α.app X ⊗ β.app X
                         /-
                           C : Type u₁
                           inst✝² : CategoryTheory.Category.{v₁, u₁} C
                           D : Type u₂
                           inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                           inst✝ : CategoryTheory.MonoidalCategory D
                           F G F' G' : CategoryTheory.Functor C D
                           α : Quiver.Hom F G
                           β : Quiver.Hom F' G'
                           X Y : C
                           f : Quiver.Hom X Y
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monoidal.FunctorCate …
                         -/
  naturality X Y f := by dsimp; rw [← tensor_comp, α.naturality, β.naturality, tensor_comp]
                                /-
                                  🎉 no goals
                                -/


/-- (An auxiliary definition for `functorCategoryMonoidal`.) -/
@[simps]
def whiskerLeft (F) (β : F' ⟶ G') : tensorObj F F' ⟶ tensorObj F G' where
  app X := F.obj X ◁ β.app X
  naturality X Y f := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.MonoidalCategory D
      F✝ G F' G' : CategoryTheory.Functor C D
      α : Quiver.Hom F✝ G
      β✝ : Quiver.Hom F' G'
      F : CategoryTheory.Functor C D
      β : Quiver.Hom F' G'
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monoidal.FunctorCate …
    -/
    simp only [← id_tensorHom]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.MonoidalCategory D
      F✝ G F' G' : CategoryTheory.Functor C D
      α : Quiver.Hom F✝ G
      β✝ : Quiver.Hom F' G'
      F : CategoryTheory.Functor C D
      β : Quiver.Hom F' G'
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monoidal.FunctorCate …
    -/
    apply (tensorHom (𝟙 F) β).naturality
    /-
      🎉 no goals
    -/


/-- (An auxiliary definition for `functorCategoryMonoidal`.) -/
@[simps]
def whiskerRight (F') : tensorObj F F' ⟶ tensorObj G F' where
  app X := α.app X ▷ F'.obj X
  naturality X Y f := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.MonoidalCategory D
      F G F'✝ G' : CategoryTheory.Functor C D
      α : Quiver.Hom F G
      β : Quiver.Hom F'✝ G'
      F' : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monoidal.FunctorCate …
    -/
    simp only [← tensorHom_id]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.MonoidalCategory D
      F G F'✝ G' : CategoryTheory.Functor C D
      α : Quiver.Hom F G
      β : Quiver.Hom F'✝ G'
      F' : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monoidal.FunctorCate …
    -/
    apply (tensorHom α (𝟙 F')).naturality
    /-
      🎉 no goals
    -/


/-- When `C` is any category, and `D` is a monoidal category,
the functor category `C ⥤ D` has a natural pointwise monoidal structure,
where `(F ⊗ G).obj X = F.obj X ⊗ G.obj X`.
-/
instance functorCategoryMonoidalStruct : MonoidalCategoryStruct (C ⥤ D) where
  tensorObj F G := tensorObj F G
  tensorHom α β := tensorHom α β
  whiskerLeft F _ _ α := FunctorCategory.whiskerLeft F α
  whiskerRight α F := FunctorCategory.whiskerRight α F
  tensorUnit := (CategoryTheory.Functor.const C).obj (𝟙_ D)
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                    inst✝ : CategoryTheory.MonoidalCategory D
                    F : CategoryTheory.Functor C D
                    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                  -/
  leftUnitor F := NatIso.ofComponents fun X => λ_ (F.obj X)
                      /-
                        C : Type u₁
                        inst✝² : CategoryTheory.Category.{v₁, u₁} C
                        D : Type u₂
                        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                        inst✝ : CategoryTheory.MonoidalCategory D
                        F G H : CategoryTheory.Functor C D
                        ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                      -/
                  /-
                    🎉 no goals
                  -/
                      /-
                        🎉 no goals
                      -/
                   /-
                     C : Type u₁
                     inst✝² : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                     inst✝ : CategoryTheory.MonoidalCategory D
                     F : CategoryTheory.Functor C D
                     ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                   -/
  rightUnitor F := NatIso.ofComponents fun X => ρ_ (F.obj X)
                   /-
                     🎉 no goals
                   -/
  associator F G H := NatIso.ofComponents fun X => α_ (F.obj X) (G.obj X) (H.obj X)


@[simp]
theorem tensorUnit_obj {X} : (𝟙_ (C ⥤ D)).obj X = 𝟙_ D :=
  rfl


@[simp]
theorem tensorUnit_map {X Y} {f : X ⟶ Y} : (𝟙_ (C ⥤ D)).map f = 𝟙 (𝟙_ D) :=
  rfl


@[simp]
theorem tensorObj_obj {F G : C ⥤ D} {X} : (F ⊗ G).obj X = F.obj X ⊗ G.obj X :=
  rfl


@[simp]
theorem tensorObj_map {F G : C ⥤ D} {X Y} {f : X ⟶ Y} : (F ⊗ G).map f = F.map f ⊗ G.map f :=
  rfl


@[simp]
theorem tensorHom_app {F G F' G' : C ⥤ D} {α : F ⟶ G} {β : F' ⟶ G'} {X} :
    (α ⊗ β).app X = α.app X ⊗ β.app X :=
  rfl


@[simp]
theorem whiskerLeft_app {F F' G' : C ⥤ D} {β : F' ⟶ G'} {X} :
    (F ◁ β).app X = F.obj X ◁ β.app X :=
  rfl


@[simp]
theorem whiskerRight_app {F G F' : C ⥤ D} {α : F ⟶ G} {X} :
    (α ▷ F').app X = α.app X ▷ F'.obj X :=
  rfl


@[simp]
theorem leftUnitor_hom_app {F : C ⥤ D} {X} :
    ((λ_ F).hom : 𝟙_ _ ⊗ F ⟶ F).app X = (λ_ (F.obj X)).hom :=
  rfl


@[simp]
theorem leftUnitor_inv_app {F : C ⥤ D} {X} :
    ((λ_ F).inv : F ⟶ 𝟙_ _ ⊗ F).app X = (λ_ (F.obj X)).inv :=
  rfl


@[simp]
theorem rightUnitor_hom_app {F : C ⥤ D} {X} :
    ((ρ_ F).hom : F ⊗ 𝟙_ _ ⟶ F).app X = (ρ_ (F.obj X)).hom :=
  rfl


@[simp]
theorem rightUnitor_inv_app {F : C ⥤ D} {X} :
    ((ρ_ F).inv : F ⟶ F ⊗ 𝟙_ _).app X = (ρ_ (F.obj X)).inv :=
  rfl


@[simp]
theorem associator_hom_app {F G H : C ⥤ D} {X} :
    ((α_ F G H).hom : (F ⊗ G) ⊗ H ⟶ F ⊗ G ⊗ H).app X = (α_ (F.obj X) (G.obj X) (H.obj X)).hom :=
  rfl


@[simp]
theorem associator_inv_app {F G H : C ⥤ D} {X} :
    ((α_ F G H).inv : F ⊗ G ⊗ H ⟶ (F ⊗ G) ⊗ H).app X = (α_ (F.obj X) (G.obj X) (H.obj X)).inv :=
  rfl


/-- When `C` is any category, and `D` is a monoidal category,
the functor category `C ⥤ D` has a natural pointwise monoidal structure,
where `(F ⊗ G).obj X = F.obj X ⊗ G.obj X`.
-/
instance functorCategoryMonoidal : MonoidalCategory (C ⥤ D) where
                      /-
                        C : Type u₁
                        inst✝² : CategoryTheory.Category.{v₁, u₁} C
                        D : Type u₂
                        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                        inst✝ : CategoryTheory.MonoidalCategory D
                        ⊢ ∀ {X₁ Y₁ X₂ Y₂ : CategoryTheory.Functor C D} (f : Quiver.Hom X₁ Y₁) (g : Qui …
                      -/
  tensorHom_def := by intros; ext; simp [tensorHom_def]
                                   /-
                                     🎉 no goals
                                   -/
                         /-
                           C : Type u₁
                           inst✝² : CategoryTheory.Category.{v₁, u₁} C
                           D : Type u₂
                           inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                           inst✝ : CategoryTheory.MonoidalCategory D
                           F G H K : CategoryTheory.Functor C D
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                         -/
  pentagon F G H K := by ext X; dsimp; rw [pentagon]
                                       /-
                                         🎉 no goals
                                       -/


/-- When `C` is any category, and `D` is a braided monoidal category,
the natural pointwise monoidal structure on the functor category `C ⥤ D`
is also braided.
-/
instance functorCategoryBraided : BraidedCategory (C ⥤ D) where
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝² : CategoryTheory.Category.{v₂, u₂} D
                    inst✝¹ : CategoryTheory.MonoidalCategory D
                    inst✝ : CategoryTheory.BraidedCategory D
                    F G : CategoryTheory.Functor C D
                    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
                  -/
  braiding F G := NatIso.ofComponents fun _ => β_ _ _
                  /-
                    🎉 no goals
                  -/
                              /-
                                C : Type u₁
                                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                D : Type u₂
                                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                inst✝¹ : CategoryTheory.MonoidalCategory D
                                inst✝ : CategoryTheory.BraidedCategory D
                                F G H : CategoryTheory.Functor C D
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                              -/
  hexagon_forward F G H := by ext X; apply hexagon_forward
                                     /-
                                       🎉 no goals
                                     -/
                              /-
                                C : Type u₁
                                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                D : Type u₂
                                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                inst✝¹ : CategoryTheory.MonoidalCategory D
                                inst✝ : CategoryTheory.BraidedCategory D
                                F G H : CategoryTheory.Functor C D
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                              -/
  hexagon_reverse F G H := by ext X; apply hexagon_reverse
                                     /-
                                       🎉 no goals
                                     -/


/-- When `C` is any category, and `D` is a symmetric monoidal category,
the natural pointwise monoidal structure on the functor category `C ⥤ D`
is also symmetric.
-/
instance functorCategorySymmetric : SymmetricCategory (C ⥤ D) where
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       inst✝¹ : CategoryTheory.MonoidalCategory D
                       inst✝ : CategoryTheory.SymmetricCategory D
                       F G : CategoryTheory.Functor C D
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
                     -/
  symmetry F G := by ext X; apply symmetry
                            /-
                              🎉 no goals
                            -/


