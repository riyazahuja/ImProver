/-- Injectiveness (in a concrete category) as a `MorphismProperty` -/
protected def injective : MorphismProperty C := fun _ _ f => Injective f


/-- Surjectiveness (in a concrete category) as a `MorphismProperty` -/
protected def surjective : MorphismProperty C := fun _ _ f => Surjective f


/-- Bijectiveness (in a concrete category) as a `MorphismProperty` -/
protected def bijective : MorphismProperty C := fun _ _ f => Bijective f


theorem bijective_eq_sup :
    MorphismProperty.bijective C = MorphismProperty.injective C ⊓ MorphismProperty.surjective C :=
  rfl


instance : (MorphismProperty.injective C).IsMultiplicative where
  id_mem X := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : C
      ⊢ CategoryTheory.MorphismProperty.injective C (CategoryTheory.CategoryStruct.i …
    -/
    delta MorphismProperty.injective
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : C
      ⊢ Function.Injective ⇑(CategoryTheory.CategoryStruct.id X)
    -/
    convert injective_id
    /-
      case h.e'_3
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : C
      ⊢ Eq (⇑(CategoryTheory.CategoryStruct.id X)) id
    -/
    aesop
    /-
      🎉 no goals
    -/
  comp_mem f g hf hg := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.injective C f
      hg : CategoryTheory.MorphismProperty.injective C g
      ⊢ CategoryTheory.MorphismProperty.injective C (CategoryTheory.CategoryStruct.c …
    -/
    delta MorphismProperty.injective
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.injective C f
      hg : CategoryTheory.MorphismProperty.injective C g
      ⊢ Function.Injective ⇑(CategoryTheory.CategoryStruct.comp f g)
    -/
    rw [coe_comp]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.injective C f
      hg : CategoryTheory.MorphismProperty.injective C g
      ⊢ Function.Injective (Function.comp ⇑g ⇑f)
    -/
    exact hg.comp hf
    /-
      🎉 no goals
    -/


instance : (MorphismProperty.surjective C).IsMultiplicative where
  id_mem X := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : C
      ⊢ CategoryTheory.MorphismProperty.surjective C (CategoryTheory.CategoryStruct. …
    -/
    delta MorphismProperty.surjective
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : C
      ⊢ Function.Surjective ⇑(CategoryTheory.CategoryStruct.id X)
    -/
    convert surjective_id
    /-
      case h.e'_3
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : C
      ⊢ Eq (⇑(CategoryTheory.CategoryStruct.id X)) id
    -/
    aesop
    /-
      🎉 no goals
    -/
  comp_mem f g hf hg := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.surjective C f
      hg : CategoryTheory.MorphismProperty.surjective C g
      ⊢ CategoryTheory.MorphismProperty.surjective C (CategoryTheory.CategoryStruct. …
    -/
    delta MorphismProperty.surjective
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.surjective C f
      hg : CategoryTheory.MorphismProperty.surjective C g
      ⊢ Function.Surjective ⇑(CategoryTheory.CategoryStruct.comp f g)
    -/
    rw [coe_comp]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.surjective C f
      hg : CategoryTheory.MorphismProperty.surjective C g
      ⊢ Function.Surjective (Function.comp ⇑g ⇑f)
    -/
    exact hg.comp hf
    /-
      🎉 no goals
    -/


instance : (MorphismProperty.bijective C).IsMultiplicative where
  id_mem X := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : C
      ⊢ CategoryTheory.MorphismProperty.bijective C (CategoryTheory.CategoryStruct.i …
    -/
    delta MorphismProperty.bijective
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : C
      ⊢ Function.Bijective ⇑(CategoryTheory.CategoryStruct.id X)
    -/
    convert bijective_id
    /-
      case h.e'_3
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : C
      ⊢ Eq (⇑(CategoryTheory.CategoryStruct.id X)) id
    -/
    aesop
    /-
      🎉 no goals
    -/
  comp_mem f g hf hg := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.bijective C f
      hg : CategoryTheory.MorphismProperty.bijective C g
      ⊢ CategoryTheory.MorphismProperty.bijective C (CategoryTheory.CategoryStruct.c …
    -/
    delta MorphismProperty.bijective
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.bijective C f
      hg : CategoryTheory.MorphismProperty.bijective C g
      ⊢ Function.Bijective ⇑(CategoryTheory.CategoryStruct.comp f g)
    -/
    rw [coe_comp]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.bijective C f
      hg : CategoryTheory.MorphismProperty.bijective C g
      ⊢ Function.Bijective (Function.comp ⇑g ⇑f)
    -/
    exact hg.comp hf
    /-
      🎉 no goals
    -/


instance injective_respectsIso : (MorphismProperty.injective C).RespectsIso :=
  respectsIso_of_isStableUnderComposition
    (fun _ _ f (_ : IsIso f) => ((forget C).mapIso (asIso f)).toEquiv.injective)


instance surjective_respectsIso : (MorphismProperty.surjective C).RespectsIso :=
  respectsIso_of_isStableUnderComposition
    (fun _ _ f (_ : IsIso f) => ((forget C).mapIso (asIso f)).toEquiv.surjective)


instance bijective_respectsIso : (MorphismProperty.bijective C).RespectsIso :=
  respectsIso_of_isStableUnderComposition
    (fun _ _ f (_ : IsIso f) => ((forget C).mapIso (asIso f)).toEquiv.bijective)


/-- The property that any morphism in a concrete category can be factored as a surjective
map followed by an injective map. -/
abbrev HasSurjectiveInjectiveFactorization :=
    (MorphismProperty.surjective C).HasFactorization (MorphismProperty.injective C)


/-- The property that any morphism in a concrete category can be functorially
factored as a surjective map followed by an injective map. -/
abbrev HasFunctorialSurjectiveInjectiveFactorization :=
  (MorphismProperty.surjective C).HasFunctorialFactorization (MorphismProperty.injective C)


/-- The structure containing the data of a functorial factorization of morphisms as
a surjective map followed by an injective map in a concrete category. -/
abbrev FunctorialSurjectiveInjectiveFactorizationData :=
  (MorphismProperty.surjective C).FunctorialFactorizationData (MorphismProperty.injective C)


/-- In the category of types, any map can be functorially factored as a surjective
map followed by an injective map. -/
def functorialSurjectiveInjectiveFactorizationData :
    FunctorialSurjectiveInjectiveFactorizationData (Type u) where
  Z :=
    { obj := fun f => Subtype (Set.range f.hom)
      map := fun φ y => ⟨φ.right y.1, by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.ConcreteCategory C
          X✝ Y✝ : CategoryTheory.Arrow (Type u)
          φ : Quiver.Hom X✝ Y✝
          y : (fun f => Subtype (Set.range f.hom)) X✝
          ⊢ Set.range Y✝.hom (φ.right ↑y)
        -/
        obtain ⟨_, x, rfl⟩ := y
        /-
          case mk.intro
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.ConcreteCategory C
          X✝ Y✝ : CategoryTheory.Arrow (Type u)
          φ : Quiver.Hom X✝ Y✝
          x : (CategoryTheory.Functor.id (Type u)).obj X✝.left
          ⊢ Set.range Y✝.hom (φ.right ↑⟨X✝.hom x, ⋯⟩)
        -/
        exact ⟨φ.left x, congr_fun φ.w x⟩ ⟩ }
        /-
          🎉 no goals
        -/
  i :=
    { app := fun f x => ⟨f.hom x, ⟨x, rfl⟩⟩
      naturality := fun f g φ => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.ConcreteCategory C
          f g : CategoryTheory.Arrow (Type u)
          φ : Quiver.Hom f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.leftFunc.map φ) …
        -/
        ext x
        /-
          case h.a
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.ConcreteCategory C
          f g : CategoryTheory.Arrow (Type u)
          φ : Quiver.Hom f g
          x : CategoryTheory.Arrow.leftFunc.obj f
          ⊢ Eq ↑(CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.leftFunc.map φ …
        -/
        exact congr_fun φ.w x }
        /-
          🎉 no goals
        -/
  p :=
    { app := fun _ y => y.1
                       /-
                         C : Type u
                         inst✝¹ : CategoryTheory.Category.{v, u} C
                         inst✝ : CategoryTheory.ConcreteCategory C
                         ⊢ ∀ ⦃X Y : CategoryTheory.Arrow (Type u)⦄ (f : Quiver.Hom X Y), Eq (CategoryTh …
                       -/
      naturality := by intros; rfl; }
                               /-
                                 🎉 no goals
                               -/
  fac := rfl
  hi := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      ⊢ ∀ (f : CategoryTheory.Arrow (Type u)), CategoryTheory.MorphismProperty.surje …
    -/
    rintro f ⟨_, x, rfl⟩
    /-
      case mk.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      f : CategoryTheory.Arrow (Type u)
      x : (CategoryTheory.Functor.id (Type u)).obj f.left
      ⊢ Exists fun a => Eq (({ app := fun f x => ⟨f.hom x, ⋯⟩, naturality := ⋯ }.app …
    -/
    exact ⟨x, rfl⟩
    /-
      🎉 no goals
    -/
  hp f x₁ x₂ h := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      f : CategoryTheory.Arrow (Type u)
      x₁ x₂ : (CategoryTheory.forget (Type u)).obj ({ obj := fun f => Subtype (Set.r …
      h : Eq (({ app := fun x y => ↑y, naturality := ⋯ }.app f) x₁) (({ app := fun x …
      ⊢ Eq x₁ x₂
    -/
    rw [Subtype.ext_iff]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      f : CategoryTheory.Arrow (Type u)
      x₁ x₂ : (CategoryTheory.forget (Type u)).obj ({ obj := fun f => Subtype (Set.r …
      h : Eq (({ app := fun x y => ↑y, naturality := ⋯ }.app f) x₁) (({ app := fun x …
      ⊢ Eq ↑x₁ ↑x₂
    -/
    exact h
    /-
      🎉 no goals
    -/


instance : HasFunctorialSurjectiveInjectiveFactorization (Type u) where
  nonempty_functorialFactorizationData :=
    ⟨functorialSurjectiveInjectiveFactorizationData⟩


