/-- A category with a terminal object and binary products has a natural monoidal structure. -/
def monoidalOfHasFiniteProducts [HasTerminal C] [HasBinaryProducts C] : MonoidalCategory C :=
  letI : MonoidalCategoryStruct C := {
    tensorObj := fun X Y ↦ X ⨯ Y
    whiskerLeft := fun _ _ _ g ↦ Limits.prod.map (𝟙 _) g
    whiskerRight := fun {_ _} f _ ↦ Limits.prod.map f (𝟙 _)
    tensorHom := fun f g ↦ Limits.prod.map f g
    tensorUnit := ⊤_ C
    associator := prod.associator
    leftUnitor := fun P ↦ Limits.prod.leftUnitor P
    rightUnitor := fun P ↦ Limits.prod.rightUnitor P
  }
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasBinaryProducts C
    this : CategoryTheory.MonoidalCategoryStruct C := { tensorObj := fun X Y => Ca …
    ⊢ ∀ (X₁ X₂ : C), Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (Category …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  .ofTensorHom
  /-
    🎉 no goals
  -/
    (pentagon := prod.pentagon)
    (triangle := prod.triangle)
    (associator_naturality := @prod.associator_naturality _ _ _)


@[ext] theorem unit_ext {X : C} (f g : X ⟶ 𝟙_ C) : f = g := terminal.hom_ext f g


@[ext] theorem tensor_ext {X Y Z : C} (f g : X ⟶ Y ⊗ Z)
    (w₁ : f ≫ prod.fst = g ≫ prod.fst) (w₂ : f ≫ prod.snd = g ≫ prod.snd) : f = g :=
  Limits.prod.hom_ext w₁ w₂


@[simp] theorem tensorUnit : 𝟙_ C = ⊤_ C := rfl


@[simp]
theorem tensorObj (X Y : C) : X ⊗ Y = (X ⨯ Y) :=
  rfl


@[simp]
theorem tensorHom {W X Y Z : C} (f : W ⟶ X) (g : Y ⟶ Z) : f ⊗ g = Limits.prod.map f g :=
  rfl


@[simp]
theorem whiskerLeft (X : C) {Y Z : C} (f : Y ⟶ Z) : X ◁ f = Limits.prod.map (𝟙 X) f :=
  rfl


@[simp]
theorem whiskerRight {X Y : C} (f : X ⟶ Y) (Z : C) : f ▷ Z = Limits.prod.map f (𝟙 Z) :=
  rfl


@[simp]
theorem leftUnitor_hom (X : C) : (λ_ X).hom = Limits.prod.snd :=
  rfl


@[simp]
theorem leftUnitor_inv (X : C) : (λ_ X).inv = prod.lift (terminal.from X) (𝟙 _) :=
  rfl


@[simp]
theorem rightUnitor_hom (X : C) : (ρ_ X).hom = Limits.prod.fst :=
  rfl


@[simp]
theorem rightUnitor_inv (X : C) : (ρ_ X).inv = prod.lift (𝟙 _) (terminal.from X) :=
  rfl

-- We don't mark this as a simp lemma, even though in many particular
-- categories the right hand side will simplify significantly further.
-- For now, we'll plan to create specialised simp lemmas in each particular category.

theorem associator_hom (X Y Z : C) :
    (α_ X Y Z).hom =
      prod.lift (Limits.prod.fst ≫ Limits.prod.fst)
        (prod.lift (Limits.prod.fst ≫ Limits.prod.snd) Limits.prod.snd) :=
  rfl


theorem associator_inv (X Y Z : C) :
    (α_ X Y Z).inv =
      prod.lift (prod.lift prod.fst (prod.snd ≫ prod.fst)) (prod.snd ≫ prod.snd) :=
  rfl


@[reassoc] theorem associator_hom_fst (X Y Z : C) :
                                                          /-
                                                            C : Type u
                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                            inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                                            inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                                            X Y Z : C
                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                          -/
    (α_ X Y Z).hom ≫ prod.fst = prod.fst ≫ prod.fst := by simp [associator_hom]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[reassoc] theorem associator_hom_snd_fst (X Y Z : C) :
                                                                     /-
                                                                       C : Type u
                                                                       inst✝² : CategoryTheory.Category.{v, u} C
                                                                       inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                                                       inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                                                       X Y Z : C
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                     -/
    (α_ X Y Z).hom ≫ prod.snd ≫ prod.fst = prod.fst ≫ prod.snd := by simp [associator_hom]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc] theorem associator_hom_snd_snd (X Y Z : C) :
                                                          /-
                                                            C : Type u
                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                            inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                                            inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                                            X Y Z : C
                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                          -/
    (α_ X Y Z).hom ≫ prod.snd ≫ prod.snd = prod.snd := by simp [associator_hom]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[reassoc] theorem associator_inv_fst_fst (X Y Z : C) :
                                                          /-
                                                            C : Type u
                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                            inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                                            inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                                            X Y Z : C
                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                          -/
    (α_ X Y Z).inv ≫ prod.fst ≫ prod.fst = prod.fst := by simp [associator_inv]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[reassoc] theorem associator_inv_fst_snd (X Y Z : C) :
                                                                     /-
                                                                       C : Type u
                                                                       inst✝² : CategoryTheory.Category.{v, u} C
                                                                       inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                                                       inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                                                       X Y Z : C
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                     -/
    (α_ X Y Z).inv ≫ prod.fst ≫ prod.snd = prod.snd ≫ prod.fst := by simp [associator_inv]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc] theorem associator_inv_snd (X Y Z : C) :
                                                          /-
                                                            C : Type u
                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                            inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                                            inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                                            X Y Z : C
                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                          -/
    (α_ X Y Z).inv ≫ prod.snd = prod.snd ≫ prod.snd := by simp [associator_inv]
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- The monoidal structure coming from finite products is symmetric.
-/
@[simps]
def symmetricOfHasFiniteProducts [HasTerminal C] [HasBinaryProducts C] : SymmetricCategory C where
  braiding X Y := Limits.prod.braiding X Y
                                     /-
                                       C : Type u
                                       inst✝² : CategoryTheory.Category.{v, u} C
                                       X✝¹ Y : C
                                       inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                       inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                       X✝ Y✝ : C
                                       f : Quiver.Hom X✝ Y✝
                                       X : C
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                     -/
                                          /-
                                            C : Type u
                                            inst✝² : CategoryTheory.Category.{v, u} C
                                            X✝ Y : C
                                            inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                            inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                            X x✝¹ x✝ : C
                                            f : Quiver.Hom x✝¹ x✝
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                          -/
  braiding_naturality_left f X := by simp
                                          /-
                                            🎉 no goals
                                          -/
                                     /-
                                       🎉 no goals
                                     -/
  braiding_naturality_right X _ _ f := by simp
                              /-
                                C : Type u
                                inst✝² : CategoryTheory.Category.{v, u} C
                                X✝ Y✝ : C
                                inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                X Y Z : C
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                              -/
  hexagon_forward X Y Z := by dsimp [monoidalOfHasFiniteProducts.associator_hom]; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
                              /-
                                C : Type u
                                inst✝² : CategoryTheory.Category.{v, u} C
                                X✝ Y✝ : C
                                inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                X Y Z : C
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                              -/
  hexagon_reverse X Y Z := by dsimp [monoidalOfHasFiniteProducts.associator_inv]; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
                     /-
                       C : Type u
                       inst✝² : CategoryTheory.Category.{v, u} C
                       X✝ Y✝ : C
                       inst✝¹ : CategoryTheory.Limits.HasTerminal C
                       inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                       X Y : C
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
                     -/
  symmetry X Y := by dsimp; simp
                            /-
                              🎉 no goals
                            -/


/-- A category with an initial object and binary coproducts has a natural monoidal structure. -/
def monoidalOfHasFiniteCoproducts [HasInitial C] [HasBinaryCoproducts C] : MonoidalCategory C :=
  letI : MonoidalCategoryStruct C := {
    tensorObj := fun X Y ↦ X ⨿ Y
    whiskerLeft := fun _ _ _ g ↦ Limits.coprod.map (𝟙 _) g
    whiskerRight := fun {_ _} f _ ↦ Limits.coprod.map f (𝟙 _)
    tensorHom := fun f g ↦ Limits.coprod.map f g
    tensorUnit := ⊥_ C
    associator := coprod.associator
    leftUnitor := fun P ↦ coprod.leftUnitor P
    rightUnitor := fun P ↦ coprod.rightUnitor P
  }
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    this : CategoryTheory.MonoidalCategoryStruct C := { tensorObj := fun X Y => Ca …
    ⊢ ∀ (X₁ X₂ : C), Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (Category …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  .ofTensorHom
  /-
    🎉 no goals
  -/
    (pentagon := coprod.pentagon)
    (triangle := coprod.triangle)
    (associator_naturality := @coprod.associator_naturality _ _ _)


@[simp]
theorem tensorObj (X Y : C) : X ⊗ Y = (X ⨿ Y) :=
  rfl


@[simp]
theorem tensorHom {W X Y Z : C} (f : W ⟶ X) (g : Y ⟶ Z) : f ⊗ g = Limits.coprod.map f g :=
  rfl


@[simp]
theorem whiskerLeft (X : C) {Y Z : C} (f : Y ⟶ Z) : X ◁ f = Limits.coprod.map (𝟙 X) f :=
  rfl


@[simp]
theorem whiskerRight {X Y : C} (f : X ⟶ Y) (Z : C) : f ▷ Z = Limits.coprod.map f (𝟙 Z) :=
  rfl


@[simp]
theorem leftUnitor_hom (X : C) : (λ_ X).hom = coprod.desc (initial.to X) (𝟙 _) :=
  rfl


@[simp]
theorem rightUnitor_hom (X : C) : (ρ_ X).hom = coprod.desc (𝟙 _) (initial.to X) :=
  rfl


@[simp]
theorem leftUnitor_inv (X : C) : (λ_ X).inv = Limits.coprod.inr :=
  rfl


@[simp]
theorem rightUnitor_inv (X : C) : (ρ_ X).inv = Limits.coprod.inl :=
  rfl

-- We don't mark this as a simp lemma, even though in many particular
-- categories the right hand side will simplify significantly further.
-- For now, we'll plan to create specialised simp lemmas in each particular category.

theorem associator_hom (X Y Z : C) :
    (α_ X Y Z).hom =
      coprod.desc (coprod.desc coprod.inl (coprod.inl ≫ coprod.inr)) (coprod.inr ≫ coprod.inr) :=
  rfl


theorem associator_inv (X Y Z : C) :
    (α_ X Y Z).inv =
      coprod.desc (coprod.inl ≫ coprod.inl) (coprod.desc (coprod.inr ≫ coprod.inl) coprod.inr) :=
  rfl


/-- The monoidal structure coming from finite coproducts is symmetric.
-/
@[simps]
def symmetricOfHasFiniteCoproducts [HasInitial C] [HasBinaryCoproducts C] :
    SymmetricCategory C where
  braiding := Limits.coprod.braiding
                                     /-
                                       C : Type u
                                       inst✝² : CategoryTheory.Category.{v, u} C
                                       X Y : C
                                       inst✝¹ : CategoryTheory.Limits.HasInitial C
                                       inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
                                       X✝ Y✝ : C
                                       f : Quiver.Hom X✝ Y✝
                                       g : C
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                     -/
                                      /-
                                        C : Type u
                                        inst✝² : CategoryTheory.Category.{v, u} C
                                        X Y : C
                                        inst✝¹ : CategoryTheory.Limits.HasInitial C
                                        inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
                                        f g : C
                                        ⊢ ∀ {Z : C} (f_1 : Quiver.Hom g Z), Eq (CategoryTheory.CategoryStruct.comp (Ca …
                                      -/
  braiding_naturality_left f g := by simp
                                      /-
                                        🎉 no goals
                                      -/
                                     /-
                                       🎉 no goals
                                     -/
  braiding_naturality_right f g := by simp
                              /-
                                C : Type u
                                inst✝² : CategoryTheory.Category.{v, u} C
                                X✝ Y✝ : C
                                inst✝¹ : CategoryTheory.Limits.HasInitial C
                                inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
                                X Y Z : C
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                              -/
  hexagon_forward X Y Z := by dsimp [monoidalOfHasFiniteCoproducts.associator_hom]; simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                              /-
                                C : Type u
                                inst✝² : CategoryTheory.Category.{v, u} C
                                X✝ Y✝ : C
                                inst✝¹ : CategoryTheory.Limits.HasInitial C
                                inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
                                X Y Z : C
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                              -/
  hexagon_reverse X Y Z := by dsimp [monoidalOfHasFiniteCoproducts.associator_inv]; simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                     /-
                       C : Type u
                       inst✝² : CategoryTheory.Category.{v, u} C
                       X✝ Y✝ : C
                       inst✝¹ : CategoryTheory.Limits.HasInitial C
                       inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
                       X Y : C
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
                     -/
  symmetry X Y := by dsimp; simp
                            /-
                              🎉 no goals
                            -/


instance : F.OplaxMonoidal where
  η' := terminalComparison F
  δ' X Y := prodComparison F X Y
                            /-
                              C : Type u
                              inst✝⁵ : CategoryTheory.Category.{v, u} C
                              X Y : C
                              D : Type u_1
                              inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
                              F : CategoryTheory.Functor C D
                              inst✝³ : CategoryTheory.Limits.HasTerminal C
                              inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                              inst✝¹ : CategoryTheory.Limits.HasTerminal D
                              inst✝ : CategoryTheory.Limits.HasBinaryProducts D
                              X✝ Y✝ : C
                              x✝¹ : Quiver.Hom X✝ Y✝
                              x✝ : C
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => CategoryTheory.Limits.pr …
                            -/
  δ'_natural_left _ _ := by simp [prodComparison_natural]
                            /-
                              🎉 no goals
                            -/
                             /-
                               C : Type u
                               inst✝⁵ : CategoryTheory.Category.{v, u} C
                               X Y : C
                               D : Type u_1
                               inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
                               F : CategoryTheory.Functor C D
                               inst✝³ : CategoryTheory.Limits.HasTerminal C
                               inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                               inst✝¹ : CategoryTheory.Limits.HasTerminal D
                               inst✝ : CategoryTheory.Limits.HasBinaryProducts D
                               X✝ Y✝ x✝¹ : C
                               x✝ : Quiver.Hom X✝ Y✝
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => CategoryTheory.Limits.pr …
                             -/
  δ'_natural_right _ _ := by simp [prodComparison_natural]
                             /-
                               🎉 no goals
                             -/
  oplax_associativity' _ _ _ := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      X Y : C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
      F : CategoryTheory.Functor C D
      inst✝³ : CategoryTheory.Limits.HasTerminal C
      inst✝² : CategoryTheory.Limits.HasBinaryProducts C
      inst✝¹ : CategoryTheory.Limits.HasTerminal D
      inst✝ : CategoryTheory.Limits.HasBinaryProducts D
      x✝² x✝¹ x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => CategoryTheory.Limits.pr …
    -/
    dsimp
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      X Y : C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
      F : CategoryTheory.Functor C D
      inst✝³ : CategoryTheory.Limits.HasTerminal C
      inst✝² : CategoryTheory.Limits.HasBinaryProducts C
      inst✝¹ : CategoryTheory.Limits.HasTerminal D
      inst✝ : CategoryTheory.Limits.HasBinaryProducts D
      x✝² x✝¹ x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prodComparison …
    -/
    ext
      /-
        case h₁
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        X Y : C
        D : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
        F : CategoryTheory.Functor C D
        inst✝³ : CategoryTheory.Limits.HasTerminal C
        inst✝² : CategoryTheory.Limits.HasBinaryProducts C
        inst✝¹ : CategoryTheory.Limits.HasTerminal D
        inst✝ : CategoryTheory.Limits.HasBinaryProducts D
        x✝² x✝¹ x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · dsimp
      simp only [Category.assoc, prod.map_fst, Category.comp_id, prodComparison_fst, ←
        Functor.map_comp]
      /-
        case h₁
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        X Y : C
        D : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
        F : CategoryTheory.Functor C D
        inst✝³ : CategoryTheory.Limits.HasTerminal C
        inst✝² : CategoryTheory.Limits.HasBinaryProducts C
        inst✝¹ : CategoryTheory.Limits.HasTerminal D
        inst✝ : CategoryTheory.Limits.HasBinaryProducts D
        x✝² x✝¹ x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prodComparison …
      -/
      erw [associator_hom_fst, associator_hom_fst]
      /-
        case h₁
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        X Y : C
        D : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
        F : CategoryTheory.Functor C D
        inst✝³ : CategoryTheory.Limits.HasTerminal C
        inst✝² : CategoryTheory.Limits.HasBinaryProducts C
        inst✝¹ : CategoryTheory.Limits.HasTerminal D
        inst✝ : CategoryTheory.Limits.HasBinaryProducts D
        x✝² x✝¹ x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prodComparison …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case h₂.h₁
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        X Y : C
        D : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
        F : CategoryTheory.Functor C D
        inst✝³ : CategoryTheory.Limits.HasTerminal C
        inst✝² : CategoryTheory.Limits.HasBinaryProducts C
        inst✝¹ : CategoryTheory.Limits.HasTerminal D
        inst✝ : CategoryTheory.Limits.HasBinaryProducts D
        x✝² x✝¹ x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · dsimp
      simp only [Category.assoc, prod.map_snd, prodComparison_snd_assoc, prodComparison_fst,
        ← Functor.map_comp]
      /-
        case h₂.h₁
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        X Y : C
        D : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
        F : CategoryTheory.Functor C D
        inst✝³ : CategoryTheory.Limits.HasTerminal C
        inst✝² : CategoryTheory.Limits.HasBinaryProducts C
        inst✝¹ : CategoryTheory.Limits.HasTerminal D
        inst✝ : CategoryTheory.Limits.HasBinaryProducts D
        x✝² x✝¹ x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prodComparison …
      -/
      erw [associator_hom_snd_fst, associator_hom_snd_fst]
      /-
        case h₂.h₁
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        X Y : C
        D : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
        F : CategoryTheory.Functor C D
        inst✝³ : CategoryTheory.Limits.HasTerminal C
        inst✝² : CategoryTheory.Limits.HasBinaryProducts C
        inst✝¹ : CategoryTheory.Limits.HasTerminal D
        inst✝ : CategoryTheory.Limits.HasBinaryProducts D
        x✝² x✝¹ x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prodComparison …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case h₂.h₂
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        X Y : C
        D : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
        F : CategoryTheory.Functor C D
        inst✝³ : CategoryTheory.Limits.HasTerminal C
        inst✝² : CategoryTheory.Limits.HasBinaryProducts C
        inst✝¹ : CategoryTheory.Limits.HasTerminal D
        inst✝ : CategoryTheory.Limits.HasBinaryProducts D
        x✝² x✝¹ x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · dsimp
      simp only [Category.assoc, prod.map_snd, prodComparison_snd_assoc, prodComparison_snd, ←
        Functor.map_comp]
      /-
        case h₂.h₂
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        X Y : C
        D : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
        F : CategoryTheory.Functor C D
        inst✝³ : CategoryTheory.Limits.HasTerminal C
        inst✝² : CategoryTheory.Limits.HasBinaryProducts C
        inst✝¹ : CategoryTheory.Limits.HasTerminal D
        inst✝ : CategoryTheory.Limits.HasBinaryProducts D
        x✝² x✝¹ x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prodComparison …
      -/
      erw [associator_hom_snd_snd, associator_hom_snd_snd]
      /-
        case h₂.h₂
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        X Y : C
        D : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
        F : CategoryTheory.Functor C D
        inst✝³ : CategoryTheory.Limits.HasTerminal C
        inst✝² : CategoryTheory.Limits.HasBinaryProducts C
        inst✝¹ : CategoryTheory.Limits.HasTerminal D
        inst✝ : CategoryTheory.Limits.HasBinaryProducts D
        x✝² x✝¹ x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prodComparison …
      -/
      simp
      /-
        🎉 no goals
      -/
                                /-
                                  C : Type u
                                  inst✝⁵ : CategoryTheory.Category.{v, u} C
                                  X Y : C
                                  D : Type u_1
                                  inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
                                  F : CategoryTheory.Functor C D
                                  inst✝³ : CategoryTheory.Limits.HasTerminal C
                                  inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                                  inst✝¹ : CategoryTheory.Limits.HasTerminal D
                                  inst✝ : CategoryTheory.Limits.HasBinaryProducts D
                                  x✝ : C
                                  ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (F.obj x✝)).inv (Catego …
                                -/
  oplax_left_unitality' _ := by ext; simp [← Functor.map_comp]
                                     /-
                                       🎉 no goals
                                     -/
                                 /-
                                   C : Type u
                                   inst✝⁵ : CategoryTheory.Category.{v, u} C
                                   X Y : C
                                   D : Type u_1
                                   inst✝⁴ : CategoryTheory.Category.{?u.221406, u_1} D
                                   F : CategoryTheory.Functor C D
                                   inst✝³ : CategoryTheory.Limits.HasTerminal C
                                   inst✝² : CategoryTheory.Limits.HasBinaryProducts C
                                   inst✝¹ : CategoryTheory.Limits.HasTerminal D
                                   inst✝ : CategoryTheory.Limits.HasBinaryProducts D
                                   x✝ : C
                                   ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (F.obj x✝)).inv (Categ …
                                 -/
  oplax_right_unitality' _ := by ext; simp [← Functor.map_comp]
                                      /-
                                        🎉 no goals
                                      -/


lemma η_eq : η F = terminalComparison F := rfl

lemma δ_eq (X Y : C) : δ F X Y = prodComparison F X Y := rfl


                             /-
                               C : Type u
                               inst✝⁷ : CategoryTheory.Category.{v, u} C
                               X Y : C
                               D : Type u_1
                               inst✝⁶ : CategoryTheory.Category.{u_2, u_1} D
                               F : CategoryTheory.Functor C D
                               inst✝⁵ : CategoryTheory.Limits.HasTerminal C
                               inst✝⁴ : CategoryTheory.Limits.HasBinaryProducts C
                               inst✝³ : CategoryTheory.Limits.HasTerminal D
                               inst✝² : CategoryTheory.Limits.HasBinaryProducts D
                               inst✝¹ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) F
                               inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
                               ⊢ CategoryTheory.IsIso (CategoryTheory.Functor.OplaxMonoidal.η F)
                             -/
instance : IsIso (η F) := by dsimp [η_eq]; infer_instance
                                           /-
                                             🎉 no goals
                                           -/

                                           /-
                                             C : Type u
                                             inst✝⁷ : CategoryTheory.Category.{v, u} C
                                             X✝ Y✝ : C
                                             D : Type u_1
                                             inst✝⁶ : CategoryTheory.Category.{u_2, u_1} D
                                             F : CategoryTheory.Functor C D
                                             inst✝⁵ : CategoryTheory.Limits.HasTerminal C
                                             inst✝⁴ : CategoryTheory.Limits.HasBinaryProducts C
                                             inst✝³ : CategoryTheory.Limits.HasTerminal D
                                             inst✝² : CategoryTheory.Limits.HasBinaryProducts D
                                             inst✝¹ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) F
                                             inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
                                             X Y : C
                                             ⊢ CategoryTheory.IsIso (CategoryTheory.Functor.OplaxMonoidal.δ F X Y)
                                           -/
instance (X Y : C) : IsIso (δ F X Y) := by dsimp [δ_eq]; infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Promote a finite products preserving functor to a monoidal functor between
categories equipped with the monoidal category structure given by finite products. -/
instance : F.Monoidal := Functor.Monoidal.ofOplaxMonoidal F


