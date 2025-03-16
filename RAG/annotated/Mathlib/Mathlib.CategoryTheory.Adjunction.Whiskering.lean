/-- Given an adjunction `F ⊣ G`, this provides the natural adjunction
  `(whiskeringRight C _ _).obj F ⊣ (whiskeringRight C _ _).obj G`. -/
@[simps! unit_app_app counit_app_app]
protected def whiskerRight (adj : F ⊣ G) :
    (whiskeringRight C D E).obj F ⊣ (whiskeringRight C E D).obj G where
  unit :=
    { app := fun X =>
        (Functor.rightUnitor _).inv ≫ whiskerLeft X adj.unit ≫ (Functor.associator _ _ _).inv
                       /-
                         C : Type u_1
                         D : Type u_2
                         E : Type u_3
                         inst✝² : CategoryTheory.Category.{?u.73, u_1} C
                         inst✝¹ : CategoryTheory.Category.{?u.77, u_2} D
                         inst✝ : CategoryTheory.Category.{?u.81, u_3} E
                         F : CategoryTheory.Functor D E
                         G : CategoryTheory.Functor E D
                         adj : CategoryTheory.Adjunction F G
                         ⊢ ∀ ⦃X Y : CategoryTheory.Functor C D⦄ (f : Quiver.Hom X Y), Eq (CategoryTheor …
                       -/
      naturality := by intros; ext; dsimp; simp }
                                           /-
                                             🎉 no goals
                                           -/
  counit :=
    { app := fun X =>
        (Functor.associator _ _ _).hom ≫ whiskerLeft X adj.counit ≫ (Functor.rightUnitor _).hom
                       /-
                         C : Type u_1
                         D : Type u_2
                         E : Type u_3
                         inst✝² : CategoryTheory.Category.{?u.73, u_1} C
                         inst✝¹ : CategoryTheory.Category.{?u.77, u_2} D
                         inst✝ : CategoryTheory.Category.{?u.81, u_3} E
                         F : CategoryTheory.Functor D E
                         G : CategoryTheory.Functor E D
                         adj : CategoryTheory.Adjunction F G
                         ⊢ ∀ ⦃X Y : CategoryTheory.Functor C E⦄ (f : Quiver.Hom X Y), Eq (CategoryTheor …
                       -/
      naturality := by intros; ext; dsimp; simp }
                                           /-
                                             🎉 no goals
                                           -/


/-- Given an adjunction `F ⊣ G`, this provides the natural adjunction
  `(whiskeringLeft _ _ C).obj G ⊣ (whiskeringLeft _ _ C).obj F`. -/
@[simps! unit_app_app counit_app_app]
protected def whiskerLeft (adj : F ⊣ G) :
    (whiskeringLeft E D C).obj G ⊣ (whiskeringLeft D E C).obj F where
  unit :=
    { app := fun X =>
        (Functor.leftUnitor _).inv ≫ whiskerRight adj.unit X ≫ (Functor.associator _ _ _).hom }
  counit :=
    { app := fun X =>
        (Functor.associator _ _ _).inv ≫ whiskerRight adj.counit X ≫ (Functor.leftUnitor _).hom }
                                   /-
                                     C : Type u_1
                                     D : Type u_2
                                     E : Type u_3
                                     inst✝² : CategoryTheory.Category.{?u.21740, u_1} C
                                     inst✝¹ : CategoryTheory.Category.{?u.21744, u_2} D
                                     inst✝ : CategoryTheory.Category.{?u.21748, u_3} E
                                     F : CategoryTheory.Functor D E
                                     G : CategoryTheory.Functor E D
                                     adj : CategoryTheory.Adjunction F G
                                     X : CategoryTheory.Functor D C
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.whiskeringLeft E D  …
                                   -/
  left_triangle_components X := by ext; simp [← X.map_comp]
                                        /-
                                          🎉 no goals
                                        -/
                                    /-
                                      C : Type u_1
                                      D : Type u_2
                                      E : Type u_3
                                      inst✝² : CategoryTheory.Category.{?u.21740, u_1} C
                                      inst✝¹ : CategoryTheory.Category.{?u.21744, u_2} D
                                      inst✝ : CategoryTheory.Category.{?u.21748, u_3} E
                                      F : CategoryTheory.Functor D E
                                      G : CategoryTheory.Functor E D
                                      adj : CategoryTheory.Adjunction F G
                                      X : CategoryTheory.Functor E C
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ app := fun X => CategoryTheory.Cat …
                                    -/
  right_triangle_components X := by ext; simp [← X.map_comp]
                                         /-
                                           🎉 no goals
                                         -/


