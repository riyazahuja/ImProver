/-- Left whiskering of an oplax natural transformation and a modification. -/
@[simps]
def whiskerLeft (η : F ⟶ G) {θ ι : G ⟶ H} (Γ : θ ⟶ ι) : η ≫ θ ⟶ η ≫ ι where
  app a := η.app a ◁ Γ.app a
  naturality {a b} f := by
    /-
      B : Type u₁
      inst✝¹ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝ : CategoryTheory.Bicategory C
      F G H I : CategoryTheory.OplaxFunctor B C
      η : Quiver.Hom F G
      θ ι : Quiver.Hom G H
      Γ : Quiver.Hom θ ι
      a b : B
      f : Quiver.Hom a b
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝¹ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝ : CategoryTheory.Bicategory C
      F G H I : CategoryTheory.OplaxFunctor B C
      η : Quiver.Hom F G
      θ ι : Quiver.Hom G H
      Γ : Quiver.Hom θ ι
      a b : B
      f : Quiver.Hom a b
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
    -/
    rw [associator_inv_naturality_right_assoc, whisker_exchange_assoc]
    /-
      B : Type u₁
      inst✝¹ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝ : CategoryTheory.Bicategory C
      F G H I : CategoryTheory.OplaxFunctor B C
      η : Quiver.Hom F G
      θ ι : Quiver.Hom G H
      Γ : Quiver.Hom θ ι
      a b : B
      f : Quiver.Hom a b
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.associator …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Right whiskering of an oplax natural transformation and a modification. -/
@[simps]
def whiskerRight {η θ : F ⟶ G} (Γ : η ⟶ θ) (ι : G ⟶ H) : η ≫ ι ⟶ θ ≫ ι where
  app a := Γ.app a ▷ ι.app a
  naturality {a b} f := by
    /-
      B : Type u₁
      inst✝¹ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝ : CategoryTheory.Bicategory C
      F G H I : CategoryTheory.OplaxFunctor B C
      η θ : Quiver.Hom F G
      Γ : Quiver.Hom η θ
      ι : Quiver.Hom G H
      a b : B
      f : Quiver.Hom a b
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝¹ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝ : CategoryTheory.Bicategory C
      F G H I : CategoryTheory.OplaxFunctor B C
      η θ : Quiver.Hom F G
      Γ : Quiver.Hom η θ
      ι : Quiver.Hom G H
      a b : B
      f : Quiver.Hom a b
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
    -/
    simp_rw [assoc, ← associator_inv_naturality_left, whisker_exchange_assoc]
    /-
      B : Type u₁
      inst✝¹ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝ : CategoryTheory.Bicategory C
      F G H I : CategoryTheory.OplaxFunctor B C
      η θ : Quiver.Hom F G
      Γ : Quiver.Hom η θ
      ι : Quiver.Hom G H
      a b : B
      f : Quiver.Hom a b
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Associator for the vertical composition of oplax natural transformations. -/
-- Porting note: verified that projections are correct and changed @[simps] to @[simps!]
@[simps!]
def associator (η : F ⟶ G) (θ : G ⟶ H) (ι : H ⟶ I) : (η ≫ θ) ≫ ι ≅ η ≫ θ ≫ ι :=
                                                                               /-
                                                                                 B : Type u₁
                                                                                 inst✝¹ : CategoryTheory.Bicategory B
                                                                                 C : Type u₂
                                                                                 inst✝ : CategoryTheory.Bicategory C
                                                                                 F G H I : CategoryTheory.OplaxFunctor B C
                                                                                 η : Quiver.Hom F G
                                                                                 θ : Quiver.Hom G H
                                                                                 ι : Quiver.Hom H I
                                                                                 ⊢ ∀ {a b : B} (f : Quiver.Hom a b), Eq (CategoryTheory.CategoryStruct.comp (Ca …
                                                                               -/
  ModificationIso.ofComponents (fun a => α_ (η.app a) (θ.app a) (ι.app a)) (by aesop_cat)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- Left unitor for the vertical composition of oplax natural transformations. -/
-- Porting note: verified that projections are correct and changed @[simps] to @[simps!]
@[simps!]
def leftUnitor (η : F ⟶ G) : 𝟙 F ≫ η ≅ η :=
                                                           /-
                                                             B : Type u₁
                                                             inst✝¹ : CategoryTheory.Bicategory B
                                                             C : Type u₂
                                                             inst✝ : CategoryTheory.Bicategory C
                                                             F G H I : CategoryTheory.OplaxFunctor B C
                                                             η : Quiver.Hom F G
                                                             ⊢ ∀ {a b : B} (f : Quiver.Hom a b), Eq (CategoryTheory.CategoryStruct.comp (Ca …
                                                           -/
  ModificationIso.ofComponents (fun a => λ_ (η.app a)) (by aesop_cat)
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- Right unitor for the vertical composition of oplax natural transformations. -/
-- Porting note: verified that projections are correct and changed @[simps] to @[simps!]
@[simps!]
def rightUnitor (η : F ⟶ G) : η ≫ 𝟙 G ≅ η :=
                                                           /-
                                                             B : Type u₁
                                                             inst✝¹ : CategoryTheory.Bicategory B
                                                             C : Type u₂
                                                             inst✝ : CategoryTheory.Bicategory C
                                                             F G H I : CategoryTheory.OplaxFunctor B C
                                                             η : Quiver.Hom F G
                                                             ⊢ ∀ {a b : B} (f : Quiver.Hom a b), Eq (CategoryTheory.CategoryStruct.comp (Ca …
                                                           -/
  ModificationIso.ofComponents (fun a => ρ_ (η.app a)) (by aesop_cat)
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- A bicategory structure on the oplax functors between bicategories. -/
-- Porting note: verified that projections are correct and changed @[simps] to @[simps!]
@[simps!]
instance OplaxFunctor.bicategory : Bicategory (OplaxFunctor B C) where
  whiskerLeft {_ _ _} η _ _ Γ := OplaxNatTrans.whiskerLeft η Γ
  whiskerRight {_ _ _} _ _ Γ η := OplaxNatTrans.whiskerRight Γ η
  associator {_ _ _} _ := OplaxNatTrans.associator
  leftUnitor {_ _} := OplaxNatTrans.leftUnitor
  rightUnitor {_ _} := OplaxNatTrans.rightUnitor
  whisker_exchange {a b c f g h i} η θ := by
    /-
      B : Type u₁
      inst✝¹ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝ : CategoryTheory.Bicategory C
      F G H I a b c : CategoryTheory.OplaxFunctor B C
      f g : Quiver.Hom a b
      h i : Quiver.Hom b c
      η : Quiver.Hom f g
      θ : Quiver.Hom h i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {x x_1 x_2} η x_3 x_4 Γ => Cate …
    -/
    ext
    /-
      case w
      B : Type u₁
      inst✝¹ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝ : CategoryTheory.Bicategory C
      F G H I a b c : CategoryTheory.OplaxFunctor B C
      f g : Quiver.Hom a b
      h i : Quiver.Hom b c
      η : Quiver.Hom f g
      θ : Quiver.Hom h i
      b✝ : B
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((fun {x x_1 x_2} η x_3 x_4 Γ => Cat …
    -/
    exact whisker_exchange _ _
    /-
      🎉 no goals
    -/


