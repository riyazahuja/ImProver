@[to_additive (attr := simps tensorObj_as leftUnitor rightUnitor associator) Discrete.addMonoidal]
instance Discrete.monoidal : MonoidalCategory (Discrete M) where
  tensorUnit := Discrete.mk 1
  tensorObj X Y := Discrete.mk (X.as * Y.as)
                                     /-
                                       M : Type u
                                       inst✝ : Monoid M
                                       X x✝¹ x✝ : CategoryTheory.Discrete M
                                       f : Quiver.Hom x✝¹ x✝
                                       ⊢ Eq ((fun X Y => { as := HMul.hMul X.as Y.as }) X x✝¹) ((fun X Y => { as := H …
                                     -/
  whiskerLeft X _ _ f := eqToHom (by dsimp; rw [eq_of_hom f])
                                            /-
                                              🎉 no goals
                                            -/
                                  /-
                                    M : Type u
                                    inst✝ : Monoid M
                                    X₁✝ X₂✝ : CategoryTheory.Discrete M
                                    f : Quiver.Hom X₁✝ X₂✝
                                    X : CategoryTheory.Discrete M
                                    ⊢ Eq ((fun X Y => { as := HMul.hMul X.as Y.as }) X₁✝ X) ((fun X Y => { as := H …
                                  -/
  whiskerRight f X := eqToHom (by dsimp; rw [eq_of_hom f])
                                         /-
                                           🎉 no goals
                                         -/
                               /-
                                 M : Type u
                                 inst✝ : Monoid M
                                 X₁✝ Y₁✝ X₂✝ Y₂✝ : CategoryTheory.Discrete M
                                 f : Quiver.Hom X₁✝ Y₁✝
                                 g : Quiver.Hom X₂✝ Y₂✝
                                 ⊢ Eq ((fun X Y => { as := HMul.hMul X.as Y.as }) X₁✝ X₂✝) ((fun X Y => { as := …
                               -/
  tensorHom f g := eqToHom (by dsimp; rw [eq_of_hom f, eq_of_hom g])
                                      /-
                                        🎉 no goals
                                      -/
  leftUnitor X := Discrete.eqToIso (one_mul X.as)
  rightUnitor X := Discrete.eqToIso (mul_one X.as)
  associator _ _ _ := Discrete.eqToIso (mul_assoc _ _ _)


@[to_additive (attr := simp) Discrete.addMonoidal_tensorUnit_as]
lemma Discrete.monoidal_tensorUnit_as : (𝟙_ (Discrete M)).as = 1 := rfl


/-- A multiplicative morphism between monoids gives a monoidal functor between the corresponding
discrete monoidal categories.
-/
@[to_additive Discrete.addMonoidalFunctor]
def Discrete.monoidalFunctor (F : M →* N) : Discrete M ⥤ Discrete N :=
  Discrete.functor (fun X ↦ Discrete.mk (F X))


@[to_additive (attr := simp) Discrete.addMonoidalFunctor_obj]
lemma Discrete.monoidalFunctor_obj (F : M →* N) (m : M) :
    (Discrete.monoidalFunctor F).obj (Discrete.mk m) = Discrete.mk (F m) := rfl


@[to_additive Discrete.addMonoidalFunctorMonoidal]
instance Discrete.monoidalFunctorMonoidal (F : M →* N) :
    (Discrete.monoidalFunctor F).Monoidal :=
    Functor.CoreMonoidal.toMonoidal
      { εIso := Discrete.eqToIso F.map_one.symm
        μIso := fun m₁ m₂ ↦ Discrete.eqToIso (F.map_mul _ _).symm }


@[to_additive Discrete.addMonoidalFunctor_ε]
lemma Discrete.monoidalFunctor_ε (F : M →* N) :
    ε (monoidalFunctor F) = Discrete.eqToHom F.map_one.symm := rfl


@[to_additive Discrete.addMonoidalFunctor_η]
lemma Discrete.monoidalFunctor_η (F : M →* N) :
    η (monoidalFunctor F) = Discrete.eqToHom F.map_one := rfl


@[to_additive Discrete.addMonoidalFunctor_μ]
lemma Discrete.monoidalFunctor_μ (F : M →* N) (m₁ m₂ : Discrete M) :
    μ (monoidalFunctor F) m₁ m₂ = Discrete.eqToHom (F.map_mul _ _).symm := rfl


@[to_additive Discrete.addMonoidalFunctor_δ]
lemma Discrete.monoidalFunctor_δ (F : M →* N) (m₁ m₂ : Discrete M) :
    δ (monoidalFunctor F) m₁ m₂ = Discrete.eqToHom (F.map_mul _ _) := rfl


/-- The monoidal natural isomorphism corresponding to composing two multiplicative morphisms.
-/
@[to_additive Discrete.addMonoidalFunctorComp
      "The monoidal natural isomorphism corresponding to\ncomposing two additive morphisms."]
def Discrete.monoidalFunctorComp (F : M →* N) (G : N →* K) :
    Discrete.monoidalFunctor F ⋙ Discrete.monoidalFunctor G ≅
      Discrete.monoidalFunctor (G.comp F) := Iso.refl _


@[to_additive Discrete.addMonoidalFunctorComp_isMonoidal]
instance Discrete.monoidalFunctorComp_isMonoidal (F : M →* N) (G : N →* K) :
    NatTrans.IsMonoidal (Discrete.monoidalFunctorComp F G).hom where
  unit := by
    /-
      M : Type u
      inst✝² : Monoid M
      N : Type u'
      inst✝¹ : Monoid N
      K : Type u
      inst✝ : Monoid K
      F : MonoidHom M N
      G : MonoidHom N K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
    -/
    dsimp only [comp_ε, monoidalFunctorComp, Iso.refl, Discrete.monoidalFunctor_ε]
    /-
      M : Type u
      inst✝² : Monoid M
      N : Type u'
      inst✝¹ : Monoid N
      K : Type u
      inst✝ : Monoid K
      F : MonoidHom M N
      G : MonoidHom N K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp [eqToHom_map]
    /-
      🎉 no goals
    -/
  tensor _ _ := by
    /-
      M : Type u
      inst✝² : Monoid M
      N : Type u'
      inst✝¹ : Monoid N
      K : Type u
      inst✝ : Monoid K
      F : MonoidHom M N
      G : MonoidHom N K
      x✝¹ x✝ : CategoryTheory.Discrete M
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
    -/
    dsimp only [comp_μ, monoidalFunctorComp, Iso.refl, Discrete.monoidalFunctor_μ]
    /-
      M : Type u
      inst✝² : Monoid M
      N : Type u'
      inst✝¹ : Monoid N
      K : Type u
      inst✝ : Monoid K
      F : MonoidHom M N
      G : MonoidHom N K
      x✝¹ x✝ : CategoryTheory.Discrete M
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp [eqToHom_map]
    /-
      🎉 no goals
    -/


