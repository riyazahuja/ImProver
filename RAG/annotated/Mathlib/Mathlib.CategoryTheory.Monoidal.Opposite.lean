/-- The type of objects of the opposite (or "reverse") monoidal category.
Use the notation `Cᴹᵒᵖ`. -/
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): This linter does not exist yet.
structure MonoidalOpposite (C : Type u₁) where
  /-- The object of `MonoidalOpposite C` that represents `x : C`. -/ mop ::
  /-- The object of `C` represented by `x : MonoidalOpposite C`. -/ unmop : C


@[inherit_doc]
notation:max C "ᴹᵒᵖ" => MonoidalOpposite C


theorem mop_injective : Function.Injective (mop : C → Cᴹᵒᵖ) := @mop.inj C


theorem unmop_injective : Function.Injective (unmop : Cᴹᵒᵖ → C) :=
  fun _ _ h => congrArg mop h


theorem mop_inj_iff (x y : C) : mop x = mop y ↔ x = y := mop_injective.eq_iff


@[simp]
theorem unmop_inj_iff (x y : Cᴹᵒᵖ) : unmop x = unmop y ↔ x = y := unmop_injective.eq_iff


@[simp]
theorem mop_unmop (X : Cᴹᵒᵖ) : mop (unmop X) = X := rfl

-- can't be simp bc after putting the lhs in whnf it's `X = X`

theorem unmop_mop (X : C) : unmop (mop X) = X := rfl


instance monoidalOppositeCategory [Category.{v₁} C] : Category Cᴹᵒᵖ where
  Hom X Y := (unmop X ⟶ unmop Y)ᴹᵒᵖ
  id X := mop (𝟙 (unmop X))
  comp f g := mop (unmop f ≫ unmop g)


/-- The monoidal opposite of a morphism `f : X ⟶ Y` is just `f`, thought of as `mop X ⟶ mop Y`. -/
def Quiver.Hom.mop {X Y : C} (f : X ⟶ Y) : mop X ⟶ mop Y := MonoidalOpposite.mop f


/-- We can think of a morphism `f : mop X ⟶ mop Y` as a morphism `X ⟶ Y`. -/
def Quiver.Hom.unmop {X Y : Cᴹᵒᵖ} (f : X ⟶ Y) : unmop X ⟶ unmop Y := MonoidalOpposite.unmop f


theorem mop_inj {X Y : C} :
    Function.Injective (Quiver.Hom.mop : (X ⟶ Y) → (mop' X ⟶ mop' Y)) :=
  fun _ _ H => congr_arg Quiver.Hom.unmop H


theorem unmop_inj {X Y : Cᴹᵒᵖ} :
    Function.Injective (Quiver.Hom.unmop : (X ⟶ Y) → (unmop' X ⟶ unmop' Y)) :=
  fun _ _ H => congr_arg Quiver.Hom.mop H


@[simp]
theorem unmop_mop {X Y : C} {f : X ⟶ Y} : f.mop.unmop = f :=
  rfl


@[simp]
theorem mop_unmop {X Y : Cᴹᵒᵖ} {f : X ⟶ Y} : f.unmop.mop = f :=
  rfl


@[simp]
theorem mop_comp {X Y Z : C} {f : X ⟶ Y} {g : Y ⟶ Z} :
    (f ≫ g).mop = f.mop ≫ g.mop := rfl


@[simp]
theorem mop_id {X : C} : (𝟙 X).mop = 𝟙 (mop X) := rfl


@[simp]
theorem unmop_comp {X Y Z : Cᴹᵒᵖ} {f : X ⟶ Y} {g : Y ⟶ Z} :
    (f ≫ g).unmop = f.unmop ≫ g.unmop := rfl


@[simp]
theorem unmop_id {X : Cᴹᵒᵖ} : (𝟙 X).unmop = 𝟙 (unmop X) := rfl


@[simp]
theorem unmop_id_mop {X : C} : (𝟙 (mop X)).unmop = 𝟙 X := rfl


@[simp]
theorem mop_id_unmop {X : Cᴹᵒᵖ} : (𝟙 (unmop X)).mop = 𝟙 X := rfl


/-- The identity functor on `C`, viewed as a functor from `C` to its monoidal opposite. -/
@[simps obj map] -- need to specify `obj, map` or else we generate `mopFunctor_obj_unmop`
                             /-
                               C : Type u₁
                               inst✝ : CategoryTheory.Category.{v₁, u₁} C
                               ⊢ ∀ (X : C), Eq ({ obj := CategoryTheory.MonoidalOpposite.mop, map := fun {X Y …
                             -/
                             /-
                               🎉 no goals
                             -/
def mopFunctor : C ⥤ Cᴹᵒᵖ := Functor.mk ⟨mop, .mop⟩
                             /-
                               🎉 no goals
                             -/

/-- The identity functor on `C`, viewed as a functor from the monoidal opposite of `C` to `C`. -/
@[simps obj map] -- not necessary but the symmetry with `mopFunctor` looks nicer
                               /-
                                 C : Type u₁
                                 inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                 ⊢ ∀ (X : CategoryTheory.MonoidalOpposite C), Eq ({ obj := CategoryTheory.Monoi …
                               -/
                               /-
                                 🎉 no goals
                               -/
def unmopFunctor : Cᴹᵒᵖ ⥤ C := Functor.mk ⟨unmop, .unmop⟩
                               /-
                                 🎉 no goals
                               -/


/-- An isomorphism in `C` gives an isomorphism in `Cᴹᵒᵖ`. -/
abbrev mop {X Y : C} (f : X ≅ Y) : mop X ≅ mop Y := (mopFunctor C).mapIso f


/-- An isomorphism in `Cᴹᵒᵖ` gives an isomorphism in `C`. -/
abbrev unmop {X Y : Cᴹᵒᵖ} (f : X ≅ Y) : unmop X ≅ unmop Y := (unmopFunctor C).mapIso f


instance {X Y : C} (f : X ⟶ Y) [IsIso f] : IsIso f.mop :=
  (mopFunctor C).map_isIso f

instance {X Y : Cᴹᵒᵖ} (f : X ⟶ Y) [IsIso f] : IsIso f.unmop :=
  (unmopFunctor C).map_isIso f


instance monoidalCategoryOp : MonoidalCategory Cᵒᵖ where
  tensorObj X Y := op (unop X ⊗ unop Y)
  whiskerLeft X _ _ f := (X.unop ◁ f.unop).op
  whiskerRight f X := (f.unop ▷ X.unop).op
  tensorHom f g := (f.unop ⊗ g.unop).op
  tensorHom_def _ _ := Quiver.Hom.unop_inj (tensorHom_def' _ _)
  tensorUnit := op (𝟙_ C)
  associator X Y Z := (α_ (unop X) (unop Y) (unop Z)).symm.op
  leftUnitor X := (λ_ (unop X)).symm.op
  rightUnitor X := (ρ_ (unop X)).symm.op
                                                           /-
                                                             C : Type u₁
                                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                             inst✝ : CategoryTheory.MonoidalCategory C
                                                             X₁✝ X₂✝ X₃✝ Y₁✝ Y₂✝ Y₃✝ : Opposite C
                                                             f : Quiver.Hom X₁✝ Y₁✝
                                                             g : Quiver.Hom X₂✝ Y₂✝
                                                             h : Quiver.Hom X₃✝ Y₃✝
                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                           -/
  associator_naturality f g h := Quiver.Hom.unop_inj <| by simp
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                       /-
                                                         C : Type u₁
                                                         inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                         inst✝ : CategoryTheory.MonoidalCategory C
                                                         X✝ Y✝ : Opposite C
                                                         f : Quiver.Hom X✝ Y✝
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                       -/
  leftUnitor_naturality f := Quiver.Hom.unop_inj <| by simp
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                        /-
                                                          C : Type u₁
                                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                          inst✝ : CategoryTheory.MonoidalCategory C
                                                          X✝ Y✝ : Opposite C
                                                          f : Quiver.Hom X✝ Y✝
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                        -/
  rightUnitor_naturality f := Quiver.Hom.unop_inj <| by simp
                                                        /-
                                                          🎉 no goals
                                                        -/
                                            /-
                                              C : Type u₁
                                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                              inst✝ : CategoryTheory.MonoidalCategory C
                                              X Y : Opposite C
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                            -/
                                                /-
                                                  C : Type u₁
                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                  inst✝ : CategoryTheory.MonoidalCategory C
                                                  W X Y Z : Opposite C
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                -/
  triangle X Y := Quiver.Hom.unop_inj <| by dsimp; monoidal_coherence
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  pentagon W X Y Z := Quiver.Hom.unop_inj <| by dsimp; monoidal_coherence


@[simp] lemma op_tensorObj (X Y : C) : op (X ⊗ Y) = op X ⊗ op Y := rfl

@[simp] lemma unop_tensorObj (X Y : Cᵒᵖ) : unop (X ⊗ Y) = unop X ⊗ unop Y := rfl


@[simp] lemma op_tensorUnit : op (𝟙_ C) = 𝟙_ Cᵒᵖ := rfl

@[simp] lemma unop_tensorUnit : unop (𝟙_ Cᵒᵖ) = 𝟙_ C := rfl


@[simp] lemma op_tensorHom {X₁ Y₁ X₂ Y₂ : C} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) :
    (f ⊗ g).op = f.op ⊗ g.op := rfl

@[simp] lemma unop_tensorHom {X₁ Y₁ X₂ Y₂ : Cᵒᵖ} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) :
    (f ⊗ g).unop = f.unop ⊗ g.unop := rfl


@[simp] lemma op_whiskerLeft (X : C) {Y Z : C} (f : Y ⟶ Z) :
    (X ◁ f).op = op X ◁ f.op := rfl

@[simp] lemma unop_whiskerLeft (X : Cᵒᵖ) {Y Z : Cᵒᵖ} (f : Y ⟶ Z) :
    (X ◁ f).unop =  unop X ◁ f.unop := rfl


@[simp] lemma op_whiskerRight {X Y : C} (f : X ⟶ Y) (Z : C) :
    (f ▷ Z).op = f.op ▷ op Z := rfl

@[simp] lemma unop_whiskerRight {X Y : Cᵒᵖ} (f : X ⟶ Y) (Z : Cᵒᵖ) :
    (f ▷ Z).unop = f.unop ▷ unop Z := rfl


@[simp] lemma op_associator (X Y Z : C) :
    (α_ X Y Z).op = (α_ (op X) (op Y) (op Z)).symm := rfl

@[simp] lemma unop_associator (X Y Z : Cᵒᵖ) :
    (α_ X Y Z).unop = (α_ (unop X) (unop Y) (unop Z)).symm := rfl


@[simp] lemma op_hom_associator (X Y Z : C) :
    (α_ X Y Z).hom.op = (α_ (op X) (op Y) (op Z)).inv := rfl

@[simp] lemma unop_hom_associator (X Y Z : Cᵒᵖ) :
    (α_ X Y Z).hom.unop = (α_ (unop X) (unop Y) (unop Z)).inv := rfl


@[simp] lemma op_inv_associator (X Y Z : C) :
    (α_ X Y Z).inv.op = (α_ (op X) (op Y) (op Z)).hom := rfl

@[simp] lemma unop_inv_associator (X Y Z : Cᵒᵖ) :
    (α_ X Y Z).inv.unop = (α_ (unop X) (unop Y) (unop Z)).hom := rfl


@[simp] lemma op_leftUnitor (X : C) : (λ_ X).op = (λ_ (op X)).symm := rfl

@[simp] lemma unop_leftUnitor (X : Cᵒᵖ) : (λ_ X).unop = (λ_ (unop X)).symm := rfl


@[simp] lemma op_hom_leftUnitor (X : C) : (λ_ X).hom.op = (λ_ (op X)).inv := rfl

@[simp] lemma unop_hom_leftUnitor (X : Cᵒᵖ) : (λ_ X).hom.unop = (λ_ (unop X)).inv := rfl


@[simp] lemma op_inv_leftUnitor (X : C) : (λ_ X).inv.op = (λ_ (op X)).hom := rfl

@[simp] lemma unop_inv_leftUnitor (X : Cᵒᵖ) : (λ_ X).inv.unop = (λ_ (unop X)).hom := rfl


@[simp] lemma op_rightUnitor (X : C) : (ρ_ X).op = (ρ_ (op X)).symm := rfl

@[simp] lemma unop_rightUnitor (X : Cᵒᵖ) : (ρ_ X).unop = (ρ_ (unop X)).symm := rfl


@[simp] lemma op_hom_rightUnitor (X : C) : (ρ_ X).hom.op = (ρ_ (op X)).inv := rfl

@[simp] lemma unop_hom_rightUnitor (X : Cᵒᵖ) : (ρ_ X).hom.unop = (ρ_ (unop X)).inv := rfl


@[simp] lemma op_inv_rightUnitor (X : C) : (ρ_ X).inv.op = (ρ_ (op X)).hom := rfl

@[simp] lemma unop_inv_rightUnitor (X : Cᵒᵖ) : (ρ_ X).inv.unop = (ρ_ (unop X)).hom := rfl


theorem op_tensor_op {W X Y Z : C} (f : W ⟶ X) (g : Y ⟶ Z) : f.op ⊗ g.op = (f ⊗ g).op := rfl


theorem unop_tensor_unop {W X Y Z : Cᵒᵖ} (f : W ⟶ X) (g : Y ⟶ Z) :
    f.unop ⊗ g.unop = (f ⊗ g).unop := rfl


instance monoidalCategoryMop : MonoidalCategory Cᴹᵒᵖ where
  tensorObj X Y := mop (unmop Y ⊗ unmop X)
  whiskerLeft X _ _ f := (f.unmop ▷ X.unmop).mop
  whiskerRight f X := (X.unmop ◁ f.unmop).mop
  tensorHom f g := (g.unmop ⊗ f.unmop).mop
  tensorHom_def _ _ := Quiver.Hom.unmop_inj (tensorHom_def' _ _)
  tensorUnit := mop (𝟙_ C)
  associator X Y Z := (α_ (unmop Z) (unmop Y) (unmop X)).symm.mop
  leftUnitor X := (ρ_ (unmop X)).mop
  rightUnitor X := (λ_ (unmop X)).mop
                                                            /-
                                                              C : Type u₁
                                                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                              inst✝ : CategoryTheory.MonoidalCategory C
                                                              X₁✝ X₂✝ X₃✝ Y₁✝ Y₂✝ Y₃✝ : CategoryTheory.MonoidalOpposite C
                                                              f : Quiver.Hom X₁✝ Y₁✝
                                                              g : Quiver.Hom X₂✝ Y₂✝
                                                              h : Quiver.Hom X₃✝ Y₃✝
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                            -/
  associator_naturality f g h := Quiver.Hom.unmop_inj <| by simp
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                        /-
                                                          C : Type u₁
                                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                          inst✝ : CategoryTheory.MonoidalCategory C
                                                          X✝ Y✝ : CategoryTheory.MonoidalOpposite C
                                                          f : Quiver.Hom X✝ Y✝
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                        -/
  leftUnitor_naturality f := Quiver.Hom.unmop_inj <| by simp
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                         /-
                                                           C : Type u₁
                                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                           inst✝ : CategoryTheory.MonoidalCategory C
                                                           X✝ Y✝ : CategoryTheory.MonoidalOpposite C
                                                           f : Quiver.Hom X✝ Y✝
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                         -/
  rightUnitor_naturality f := Quiver.Hom.unmop_inj <| by simp
                                                         /-
                                                           🎉 no goals
                                                         -/
  -- Porting note: Changed `by coherence` to `by simp` below
                                             /-
                                               C : Type u₁
                                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                               inst✝ : CategoryTheory.MonoidalCategory C
                                               X Y : CategoryTheory.MonoidalOpposite C
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                             -/
                                                 /-
                                                   C : Type u₁
                                                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                   inst✝ : CategoryTheory.MonoidalCategory C
                                                   W X Y Z : CategoryTheory.MonoidalOpposite C
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                 -/
  triangle X Y := Quiver.Hom.unmop_inj <| by simp
                                                        /-
                                                          🎉 no goals
                                                        -/
                                             /-
                                               🎉 no goals
                                             -/
  pentagon W X Y Z := Quiver.Hom.unmop_inj <| by dsimp; monoidal_coherence

-- it would be nice if we could autogenerate all of these somehow

@[simp] lemma mop_tensorObj (X Y : C) : mop (X ⊗ Y) = mop Y ⊗ mop X := rfl

@[simp] lemma unmop_tensorObj (X Y : Cᴹᵒᵖ) : unmop (X ⊗ Y) = unmop Y ⊗ unmop X := rfl


@[simp] lemma mop_tensorUnit : mop (𝟙_ C) = 𝟙_ Cᴹᵒᵖ := rfl

@[simp] lemma unmop_tensorUnit : unmop (𝟙_ Cᴹᵒᵖ) = 𝟙_ C := rfl


@[simp] lemma mop_tensorHom {X₁ Y₁ X₂ Y₂ : C} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) :
    (f ⊗ g).mop = g.mop ⊗ f.mop := rfl

@[simp] lemma unmop_tensorHom {X₁ Y₁ X₂ Y₂ : Cᴹᵒᵖ} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) :
    (f ⊗ g).unmop = g.unmop ⊗ f.unmop := rfl


@[simp] lemma mop_whiskerLeft (X : C) {Y Z : C} (f : Y ⟶ Z) :
    (X ◁ f).mop = f.mop ▷ mop X := rfl

@[simp] lemma unmop_whiskerLeft (X : Cᴹᵒᵖ) {Y Z : Cᴹᵒᵖ} (f : Y ⟶ Z) :
    (X ◁ f).unmop = f.unmop ▷ unmop X := rfl


@[simp] lemma mop_whiskerRight {X Y : C} (f : X ⟶ Y) (Z : C) :
    (f ▷ Z).mop = mop Z ◁ f.mop := rfl

@[simp] lemma unmop_whiskerRight {X Y : Cᴹᵒᵖ} (f : X ⟶ Y) (Z : Cᴹᵒᵖ) :
    (f ▷ Z).unmop = unmop Z ◁ f.unmop := rfl


@[simp] lemma mop_associator (X Y Z : C) :
    (α_ X Y Z).mop = (α_ (mop Z) (mop Y) (mop X)).symm := rfl

@[simp] lemma unmop_associator (X Y Z : Cᴹᵒᵖ) :
    (α_ X Y Z).unmop = (α_ (unmop Z) (unmop Y) (unmop X)).symm := rfl


@[simp] lemma mop_hom_associator (X Y Z : C) :
    (α_ X Y Z).hom.mop = (α_ (mop Z) (mop Y) (mop X)).inv := rfl

@[simp] lemma unmop_hom_associator (X Y Z : Cᴹᵒᵖ) :
    (α_ X Y Z).hom.unmop = (α_ (unmop Z) (unmop Y) (unmop X)).inv := rfl


@[simp] lemma mop_inv_associator (X Y Z : C) :
    (α_ X Y Z).inv.mop = (α_ (mop Z) (mop Y) (mop X)).hom := rfl

@[simp] lemma unmop_inv_associator (X Y Z : Cᴹᵒᵖ) :
    (α_ X Y Z).inv.unmop = (α_ (unmop Z) (unmop Y) (unmop X)).hom := rfl


@[simp] lemma mop_leftUnitor (X : C) : (λ_ X).mop = (ρ_ (mop X)) := rfl

@[simp] lemma unmop_leftUnitor (X : Cᴹᵒᵖ) : (λ_ X).unmop = ρ_ (unmop X) := rfl


@[simp] lemma mop_hom_leftUnitor (X : C) : (λ_ X).hom.mop = (ρ_ (mop X)).hom := rfl

@[simp] lemma unmop_hom_leftUnitor (X : Cᴹᵒᵖ) : (λ_ X).hom.unmop = (ρ_ (unmop X)).hom := rfl


@[simp] lemma mop_inv_leftUnitor (X : C) : (λ_ X).inv.mop = (ρ_ (mop X)).inv := rfl

@[simp] lemma unmop_inv_leftUnitor (X : Cᴹᵒᵖ) : (λ_ X).inv.unmop = (ρ_ (unmop X)).inv := rfl


@[simp] lemma mop_rightUnitor (X : C) : (ρ_ X).mop = (λ_ (mop X)) := rfl

@[simp] lemma unmop_rightUnitor (X : Cᴹᵒᵖ) : (ρ_ X).unmop = λ_ (unmop X) := rfl


@[simp] lemma mop_hom_rightUnitor (X : C) : (ρ_ X).hom.mop = (λ_ (mop X)).hom := rfl

@[simp] lemma unmop_hom_rightUnitor (X : Cᴹᵒᵖ) : (ρ_ X).hom.unmop = (λ_ (unmop X)).hom := rfl


@[simp] lemma mop_inv_rightUnitor (X : C) : (ρ_ X).inv.mop = (λ_ (mop X)).inv := rfl

@[simp] lemma unmop_inv_rightUnitor (X : Cᴹᵒᵖ) : (ρ_ X).inv.unmop = (λ_ (unmop X)).inv := rfl


/-- The (identity) equivalence between `C` and its monoidal opposite. -/
@[simps] def MonoidalOpposite.mopEquiv : C ≌ Cᴹᵒᵖ where
  functor   := mopFunctor C
  inverse   := unmopFunctor C
  unitIso   := Iso.refl _
  counitIso := Iso.refl _


/-- The (identity) equivalence between `Cᴹᵒᵖ` and `C`. -/
@[simps!] def MonoidalOpposite.unmopEquiv : Cᴹᵒᵖ ≌ C := (mopEquiv C).symm

-- todo: upgrade to monoidal equivalence

/-- The equivalence between `C` and its monoidal opposite's monoidal opposite. -/
@[simps!] def MonoidalOpposite.mopMopEquivalence : Cᴹᵒᵖᴹᵒᵖ ≌ C :=
  .trans (MonoidalOpposite.unmopEquiv Cᴹᵒᵖ) (MonoidalOpposite.unmopEquiv C)


/-- The identification `mop X ⊗ mop Y = mop (Y ⊗ X)` as a natural isomorphism. -/
@[simps!]
def MonoidalOpposite.tensorIso :
    tensor Cᴹᵒᵖ ≅ (unmopFunctor C).prod (unmopFunctor C) ⋙
      Prod.swap C C ⋙ tensor C ⋙ mopFunctor C :=
  Iso.refl _


/-- The identification `X ⊗ - = mop (- ⊗ unmop X)` as a natural isomorphism. -/
@[simps!]
def MonoidalOpposite.tensorLeftIso (X : Cᴹᵒᵖ) :
    tensorLeft X ≅ unmopFunctor C ⋙ tensorRight (unmop X) ⋙ mopFunctor C :=
  Iso.refl _


/-- The identification `mop X ⊗ - = mop (- ⊗ X)` as a natural isomorphism. -/
@[simps!]
def MonoidalOpposite.tensorLeftMopIso (X : C) :
    tensorLeft (mop X) ≅ unmopFunctor C ⋙ tensorRight X ⋙ mopFunctor C :=
  Iso.refl _


/-- The identification `unmop X ⊗ - = unmop (mop - ⊗ X)` as a natural isomorphism. -/
@[simps!]
def MonoidalOpposite.tensorLeftUnmopIso (X : Cᴹᵒᵖ) :
    tensorLeft (unmop X) ≅ mopFunctor C ⋙ tensorRight X ⋙ unmopFunctor C :=
  Iso.refl _


/-- The identification `- ⊗ X = mop (unmop X ⊗ -)` as a natural isomorphism. -/
@[simps!]
def MonoidalOpposite.tensorRightIso (X : Cᴹᵒᵖ) :
    tensorRight X ≅ unmopFunctor C ⋙ tensorLeft (unmop X) ⋙ mopFunctor C :=
  Iso.refl _


/-- The identification `- ⊗ mop X = mop (- ⊗ unmop X)` as a natural isomorphism. -/
@[simps!]
def MonoidalOpposite.tensorRightMopIso (X : C) :
    tensorRight (mop X) ≅ unmopFunctor C ⋙ tensorLeft X ⋙ mopFunctor C :=
  Iso.refl _


/-- The identification `- ⊗ unmop X = unmop (X ⊗ mop -)` as a natural isomorphism. -/
@[simps!]
def MonoidalOpposite.tensorRightUnmopIso (X : Cᴹᵒᵖ) :
    tensorRight (unmop X) ≅ mopFunctor C ⋙ tensorLeft X ⋙ unmopFunctor C :=
  Iso.refl _


