@[simps]
noncomputable instance instMonoidalCategoryStruct :
    MonoidalCategoryStruct.{u} (CoalgebraCat R) where
  tensorObj X Y := of R (X ⊗[R] Y)
  whiskerLeft X _ _ f := ofHom (f.1.lTensor X)
  whiskerRight f X := ofHom (f.1.rTensor X)
  tensorHom f g := ofHom (Coalgebra.TensorProduct.map f.1 g.1)
  tensorUnit := CoalgebraCat.of R R
  associator X Y Z := (Coalgebra.TensorProduct.assoc R X Y Z).toCoalgebraCatIso
  leftUnitor X := (Coalgebra.TensorProduct.lid R X).toCoalgebraCatIso
  rightUnitor X := (Coalgebra.TensorProduct.rid R X).toCoalgebraCatIso


/-- The data needed to induce a `MonoidalCategory` structure via
`CoalgebraCat.instMonoidalCategoryStruct` and the forgetful functor to modules. -/
@[simps]
noncomputable def MonoidalCategory.inducingFunctorData :
    Monoidal.InducingFunctorData (forget₂ (CoalgebraCat R) (ModuleCat R)) where
  μIso _ _ := Iso.refl _
                               /-
                                 R : Type u
                                 inst✝ : CommRing R
                                 X Y Z : CoalgebraCat R
                                 f : Quiver.Hom Y Z
                                 ⊢ Eq ((CategoryTheory.forget₂ (CoalgebraCat R) (ModuleCat R)).map (CategoryThe …
                               -/
  whiskerLeft_eq X Y Z f := by ext; rfl
                                    /-
                                      🎉 no goals
                                    -/
                            /-
                              R : Type u
                              inst✝ : CommRing R
                              X₁✝ X₂✝ : CoalgebraCat R
                              X : Quiver.Hom X₁✝ X₂✝
                              f : CoalgebraCat R
                              ⊢ Eq ((CategoryTheory.forget₂ (CoalgebraCat R) (ModuleCat R)).map (CategoryThe …
                            -/
  whiskerRight_eq X f := by ext; rfl
                                 /-
                                   🎉 no goals
                                 -/
                         /-
                           R : Type u
                           inst✝ : CommRing R
                           X₁✝ Y₁✝ X₂✝ Y₂✝ : CoalgebraCat R
                           f : Quiver.Hom X₁✝ Y₁✝
                           g : Quiver.Hom X₂✝ Y₂✝
                           ⊢ Eq ((CategoryTheory.forget₂ (CoalgebraCat R) (ModuleCat R)).map (CategoryThe …
                         -/
  tensorHom_eq f g := by ext; rfl
                              /-
                                🎉 no goals
                              -/
  εIso := Iso.refl _
                                                                                           /-
                                                                                             R : Type u
                                                                                             inst✝ : CommRing R
                                                                                             X Y Z : CoalgebraCat R
                                                                                             ⊢ Eq ((TensorProduct.mk R ↑X.toModuleCat ↑Y.toModuleCat).compr₂ ((TensorProduc …
                                                                                           -/
  associator_eq X Y Z := ModuleCat.hom_ext <| TensorProduct.ext <| TensorProduct.ext <| by ext; rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/
                                                                  /-
                                                                    R : Type u
                                                                    inst✝ : CommRing R
                                                                    X : CoalgebraCat R
                                                                    ⊢ Eq ((TensorProduct.mk R ↑CategoryTheory.MonoidalCategoryStruct.tensorUnit.to …
                                                                  -/
  leftUnitor_eq X := ModuleCat.hom_ext <| TensorProduct.ext <| by ext; rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                   /-
                                                                     R : Type u
                                                                     inst✝ : CommRing R
                                                                     X : CoalgebraCat R
                                                                     ⊢ Eq ((TensorProduct.mk R ↑X.toModuleCat ↑CategoryTheory.MonoidalCategoryStruc …
                                                                   -/
  rightUnitor_eq X := ModuleCat.hom_ext <| TensorProduct.ext <| by ext; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


noncomputable instance instMonoidalCategory : MonoidalCategory (CoalgebraCat R) :=
  Monoidal.induced (forget₂ _ (ModuleCat R)) (MonoidalCategory.inducingFunctorData R)


