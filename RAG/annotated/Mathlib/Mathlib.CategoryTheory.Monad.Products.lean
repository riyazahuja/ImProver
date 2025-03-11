/-- `X ⨯ -` has a comonad structure. This is sometimes called the writer comonad. -/
@[simps!]
def prodComonad : Comonad C where
  toFunctor := prod.functor.obj X
  ε := { app := fun _ => Limits.prod.snd }
  δ := { app := fun _ => prod.lift Limits.prod.fst (𝟙 _) }


/-- The forward direction of the equivalence from coalgebras for the product comonad to the over
category.
-/
@[simps]
def coalgebraToOver : Coalgebra (prodComonad X) ⥤ Over X where
  obj A := Over.mk (A.a ≫ Limits.prod.fst)
  map f :=
    Over.homMk f.f
      (by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : C
          inst✝ : CategoryTheory.Limits.HasBinaryProducts C
          X✝ Y✝ : (CategoryTheory.prodComonad X).Coalgebra
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f.f ((fun A => CategoryTheory.Over.mk …
        -/
        rw [Over.mk_hom, ← f.h_assoc]
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : C
          inst✝ : CategoryTheory.Limits.HasBinaryProducts C
          X✝ Y✝ : (CategoryTheory.prodComonad X).Coalgebra
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp X✝.a (CategoryTheory.CategoryStruct.c …
        -/
        dsimp
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : C
          inst✝ : CategoryTheory.Limits.HasBinaryProducts C
          X✝ Y✝ : (CategoryTheory.prodComonad X).Coalgebra
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp X✝.a (CategoryTheory.CategoryStruct.c …
        -/
        simp)
        /-
          🎉 no goals
        -/


/-- The backward direction of the equivalence from coalgebras for the product comonad to the over
category.
-/
@[simps]
def overToCoalgebra : Over X ⥤ Coalgebra (prodComonad X) where
  obj f :=
    { A := f.left
      a := prod.lift f.hom (𝟙 _) }
  map g := { f := g.left }


/-- The equivalence from coalgebras for the product comonad to the over category. -/
@[simps]
def coalgebraEquivOver : Coalgebra (prodComonad X) ≌ Over X where
  functor := coalgebraToOver X
  inverse := overToCoalgebra X
             /-
               C : Type u
               inst✝¹ : CategoryTheory.Category.{v, u} C
               X : C
               inst✝ : CategoryTheory.Limits.HasBinaryProducts C
               ⊢ ∀ {X_1 Y : (CategoryTheory.prodComonad X).Coalgebra} (f : Quiver.Hom X_1 Y), …
             -/
                                                          /-
                                                            C : Type u
                                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                                            X : C
                                                            inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                                            A : (CategoryTheory.prodComonad X).Coalgebra
                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  unitIso := NatIso.ofComponents fun A =>
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
             /-
               🎉 no goals
             -/
    Coalgebra.isoMk (Iso.refl _) (Limits.prod.hom_ext (by simp) (by simpa using A.counit))
                                            /-
                                              C : Type u
                                              inst✝¹ : CategoryTheory.Category.{v, u} C
                                              X : C
                                              inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                              f : CategoryTheory.Over X
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (((CategoryT …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  counitIso := NatIso.ofComponents fun f => Over.isoMk (Iso.refl _)
               /-
                 🎉 no goals
               -/


/-- `X ⨿ -` has a monad structure. This is sometimes called the either monad. -/
@[simps!]
def coprodMonad : Monad C where
  toFunctor := coprod.functor.obj X
  η := { app := fun _ => coprod.inr }
  μ := { app := fun _ => coprod.desc coprod.inl (𝟙 _) }


/-- The forward direction of the equivalence from algebras for the coproduct monad to the under
category.
-/
@[simps]
def algebraToUnder : Monad.Algebra (coprodMonad X) ⥤ Under X where
  obj A := Under.mk (coprod.inl ≫ A.a)
  map f :=
    Under.homMk f.f
      (by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : C
          inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
          X✝ Y✝ : (CategoryTheory.coprodMonad X).Algebra
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun A => CategoryTheory.Under.mk (C …
        -/
        rw [Under.mk_hom, Category.assoc, ← f.h]
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : C
          inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
          X✝ Y✝ : (CategoryTheory.coprodMonad X).Algebra
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
        -/
        dsimp
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : C
          inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
          X✝ Y✝ : (CategoryTheory.coprodMonad X).Algebra
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
        -/
        simp)
        /-
          🎉 no goals
        -/


/-- The backward direction of the equivalence from algebras for the coproduct monad to the under
category.
-/
@[simps]
def underToAlgebra : Under X ⥤ Monad.Algebra (coprodMonad X) where
  obj f :=
    { A := f.right
      a := coprod.desc f.hom (𝟙 _) }
  map g := { f := g.right }


/-- The equivalence from algebras for the coproduct monad to the under category.
-/
@[simps]
def algebraEquivUnder : Monad.Algebra (coprodMonad X) ≌ Under X where
  functor := algebraToUnder X
  inverse := underToAlgebra X
             /-
               C : Type u
               inst✝¹ : CategoryTheory.Category.{v, u} C
               X : C
               inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
               ⊢ ∀ {X_1 Y : (CategoryTheory.coprodMonad X).Algebra} (f : Quiver.Hom X_1 Y), E …
             -/
                                                         /-
                                                           C : Type u
                                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                                           X : C
                                                           inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
                                                           A : (CategoryTheory.coprodMonad X).Algebra
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  unitIso := NatIso.ofComponents fun A =>
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
             /-
               🎉 no goals
             -/
    Monad.Algebra.isoMk (Iso.refl _) (coprod.hom_ext (by simp) (by simpa using A.unit.symm))
  counitIso :=
                                 /-
                                   C : Type u
                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                   X : C
                                   inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
                                   f : CategoryTheory.Under X
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.underToAlgebra X).c …
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
    NatIso.ofComponents fun f => Under.isoMk (Iso.refl _)
    /-
      🎉 no goals
    -/


