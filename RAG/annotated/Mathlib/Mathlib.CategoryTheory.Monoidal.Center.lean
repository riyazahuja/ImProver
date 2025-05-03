/-- A half-braiding on `X : C` is a family of isomorphisms `X ⊗ U ≅ U ⊗ X`,
monoidally natural in `U : C`.

Thinking of `C` as a 2-category with a single `0`-morphism, these are the same as natural
transformations (in the pseudo- sense) of the identity 2-functor on `C`, which send the unique
`0`-morphism to `X`.
-/
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): This linter does not exist yet.
structure HalfBraiding (X : C) where
  β : ∀ U, X ⊗ U ≅ U ⊗ X
  monoidal : ∀ U U', (β (U ⊗ U')).hom =
      (α_ _ _ _).inv ≫
        ((β U).hom ▷ U') ≫ (α_ _ _ _).hom ≫ (U ◁ (β U').hom) ≫ (α_ _ _ _).inv := by
    aesop_cat
  naturality : ∀ {U U'} (f : U ⟶ U'), (X ◁ f) ≫ (β U').hom = (β U).hom ≫ (f ▷ X) := by
    aesop_cat


attribute [reassoc, simp] HalfBraiding.monoidal -- the reassoc lemma is redundant as a simp lemma


attribute [simp, reassoc] HalfBraiding.naturality


/-- The Drinfeld center of a monoidal category `C` has as objects pairs `⟨X, b⟩`, where `X : C`
and `b` is a half-braiding on `X`.
-/
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): This linter does not exist yet.
def Center :=
  Σ X : C, HalfBraiding X


/-- A morphism in the Drinfeld center of `C`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet
@[ext] -- @[nolint has_nonempty_instance]
structure Hom (X Y : Center C) where
  f : X.1 ⟶ Y.1
  comm : ∀ U, (f ▷ U) ≫ (Y.2.β U).hom = (X.2.β U).hom ≫ (U ◁ f) := by aesop_cat


attribute [reassoc (attr := simp)] Hom.comm


instance : Quiver (Center C) where
  Hom := Hom


@[ext]
theorem ext {X Y : Center C} (f g : X ⟶ Y) (w : f.f = g.f) : f = g := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : CategoryTheory.Center C
    f g : Quiver.Hom X Y
    w : Eq f.f g.f
    ⊢ Eq f g
  -/
  cases f; cases g; congr
                    /-
                      🎉 no goals
                    -/


instance : Category (Center C) where
  id X := { f := 𝟙 X.1 }
  comp f g := { f := f.f ≫ g.f }


@[simp]
theorem id_f (X : Center C) : Hom.f (𝟙 X) = 𝟙 X.1 :=
  rfl


@[simp]
theorem comp_f {X Y Z : Center C} (f : X ⟶ Y) (g : Y ⟶ Z) : (f ≫ g).f = f.f ≫ g.f :=
  rfl


/-- Construct an isomorphism in the Drinfeld center from
a morphism whose underlying morphism is an isomorphism.
-/
@[simps]
def isoMk {X Y : Center C} (f : X ⟶ Y) [IsIso f.f] : X ≅ Y where
  hom := f
  inv := ⟨inv f.f,
    fun U => by simp [← cancel_epi (f.f ▷ U), ← comp_whiskerRight_assoc,
      ← MonoidalCategory.whiskerLeft_comp] ⟩


instance isIso_of_f_isIso {X Y : Center C} (f : X ⟶ Y) [IsIso f.f] : IsIso f := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y : CategoryTheory.Center C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f.f
    ⊢ CategoryTheory.IsIso f
  -/
  change IsIso (isoMk f).hom
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y : CategoryTheory.Center C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f.f
    ⊢ CategoryTheory.IsIso (CategoryTheory.Center.isoMk f).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for the `MonoidalCategory` instance on `Center C`. -/
@[simps]
def tensorObj (X Y : Center C) : Center C :=
  ⟨X.1 ⊗ Y.1,
    { β := fun U =>
        α_ _ _ _ ≪≫
          (whiskerLeftIso X.1 (Y.2.β U)) ≪≫ (α_ _ _ _).symm ≪≫
            (whiskerRightIso (X.2.β U) Y.1) ≪≫ α_ _ _ _
      monoidal := fun U U' => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          X Y : CategoryTheory.Center C
          U U' : C
          ⊢ Eq ((fun U => (CategoryTheory.MonoidalCategoryStruct.associator X.fst Y.fst  …
        -/
        dsimp only [Iso.trans_hom, whiskerLeftIso_hom, Iso.symm_hom, whiskerRightIso_hom]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          X Y : CategoryTheory.Center C
          U U' : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        simp only [HalfBraiding.monoidal]
        -- We'd like to commute `X.1 ◁ U ◁ (HalfBraiding.β Y.2 U').hom`
        -- and `((HalfBraiding.β X.2 U).hom ▷ U' ▷ Y.1)` past each other.
        -- We do this with the help of the monoidal composition `⊗≫` and the `coherence` tactic.
        calc
          _ = 𝟙 _ ⊗≫
            X.1 ◁ (HalfBraiding.β Y.2 U).hom ▷ U' ⊗≫
              (_ ◁ (HalfBraiding.β Y.2 U').hom ≫
                (HalfBraiding.β X.2 U).hom ▷ _) ⊗≫
                  U ◁ (HalfBraiding.β X.2 U').hom ▷ Y.1 ⊗≫ 𝟙 _ := by monoidal
          _ = _ := by rw [whisker_exchange]; monoidal
      naturality := fun {U U'} f => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          X Y : CategoryTheory.Center C
          U U' : C
          f : Quiver.Hom U U'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        dsimp only [Iso.trans_hom, whiskerLeftIso_hom, Iso.symm_hom, whiskerRightIso_hom]
        calc
          _ = 𝟙 _ ⊗≫
            (X.1 ◁ (Y.1 ◁ f ≫ (HalfBraiding.β Y.2 U').hom)) ⊗≫
              (HalfBraiding.β X.2 U').hom ▷ Y.1 ⊗≫ 𝟙 _ := by monoidal
          _ = 𝟙 _ ⊗≫
            X.1 ◁ (HalfBraiding.β Y.2 U).hom ⊗≫
              (X.1 ◁ f ≫ (HalfBraiding.β X.2 U').hom) ▷ Y.1 ⊗≫ 𝟙 _ := by
            rw [HalfBraiding.naturality]; monoidal
          _ = _ := by rw [HalfBraiding.naturality]; monoidal }⟩


@[reassoc]
theorem whiskerLeft_comm (X : Center C) {Y₁ Y₂ : Center C} (f : Y₁ ⟶ Y₂) (U : C) :
    (X.1 ◁ f.f) ▷ U ≫ ((tensorObj X Y₂).2.β U).hom =
      ((tensorObj X Y₁).2.β U).hom ≫ U ◁ X.1 ◁ f.f := by
  dsimp only [tensorObj_fst, tensorObj_snd_β, Iso.trans_hom, whiskerLeftIso_hom,
    Iso.symm_hom, whiskerRightIso_hom]
  calc
    _ = 𝟙 _ ⊗≫
      X.fst ◁ (f.f ▷ U ≫ (HalfBraiding.β Y₂.snd U).hom) ⊗≫
        (HalfBraiding.β X.snd U).hom ▷ Y₂.fst ⊗≫ 𝟙 _ := by monoidal
    _ = 𝟙 _ ⊗≫
      X.fst ◁ (HalfBraiding.β Y₁.snd U).hom ⊗≫
        ((X.fst ⊗ U) ◁ f.f ≫ (HalfBraiding.β X.snd U).hom ▷ Y₂.fst) ⊗≫ 𝟙 _ := by
      rw [f.comm]; monoidal
    _ = _ := by rw [whisker_exchange]; monoidal


/-- Auxiliary definition for the `MonoidalCategory` instance on `Center C`. -/
def whiskerLeft (X : Center C) {Y₁ Y₂ : Center C} (f : Y₁ ⟶ Y₂) :
    tensorObj X Y₁ ⟶ tensorObj X Y₂ where
  f := X.1 ◁ f.f
  comm U := whiskerLeft_comm X f U


@[reassoc]
theorem whiskerRight_comm {X₁ X₂ : Center C} (f : X₁ ⟶ X₂) (Y : Center C) (U : C) :
    f.f ▷ Y.1 ▷ U ≫ ((tensorObj X₂ Y).2.β U).hom =
      ((tensorObj X₁ Y).2.β U).hom ≫ U ◁ f.f ▷ Y.1 := by
  dsimp only [tensorObj_fst, tensorObj_snd_β, Iso.trans_hom, whiskerLeftIso_hom,
    Iso.symm_hom, whiskerRightIso_hom]
  calc
    _ = 𝟙 _ ⊗≫
      (f.f ▷ (Y.fst ⊗ U) ≫ X₂.fst ◁ (HalfBraiding.β Y.snd U).hom) ⊗≫
        (HalfBraiding.β X₂.snd U).hom ▷ Y.fst ⊗≫ 𝟙 _ := by monoidal
    _ = 𝟙 _ ⊗≫
      X₁.fst ◁ (HalfBraiding.β Y.snd U).hom ⊗≫
        (f.f ▷ U ≫ (HalfBraiding.β X₂.snd U).hom) ▷ Y.fst ⊗≫ 𝟙 _ := by
      rw [← whisker_exchange]; monoidal
    _ = _ := by rw [f.comm]; monoidal


/-- Auxiliary definition for the `MonoidalCategory` instance on `Center C`. -/
def whiskerRight {X₁ X₂ : Center C} (f : X₁ ⟶ X₂) (Y : Center C) :
    tensorObj X₁ Y ⟶ tensorObj X₂ Y where
  f := f.f ▷ Y.1
  comm U := whiskerRight_comm f Y U


/-- Auxiliary definition for the `MonoidalCategory` instance on `Center C`. -/
@[simps]
def tensorHom {X₁ Y₁ X₂ Y₂ : Center C} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) :
    tensorObj X₁ X₂ ⟶ tensorObj Y₁ Y₂ where
  f := f.f ⊗ g.f
  comm U := by
    rw [tensorHom_def, comp_whiskerRight_assoc, whiskerLeft_comm, whiskerRight_comm_assoc,
      MonoidalCategory.whiskerLeft_comp]


/-- Auxiliary definition for the `MonoidalCategory` instance on `Center C`. -/
@[simps]
def tensorUnit : Center C :=
  ⟨𝟙_ C, { β := fun U => λ_ U ≪≫ (ρ_ U).symm }⟩


/-- Auxiliary definition for the `MonoidalCategory` instance on `Center C`. -/
def associator (X Y Z : Center C) : tensorObj (tensorObj X Y) Z ≅ tensorObj X (tensorObj Y Z) :=
                                           /-
                                             C : Type u₁
                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                             inst✝ : CategoryTheory.MonoidalCategory C
                                             X Y Z : CategoryTheory.Center C
                                             U : C
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                           -/
  isoMk ⟨(α_ X.1 Y.1 Z.1).hom, fun U => by simp⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- Auxiliary definition for the `MonoidalCategory` instance on `Center C`. -/
def leftUnitor (X : Center C) : tensorObj tensorUnit X ≅ X :=
                                   /-
                                     C : Type u₁
                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                     inst✝ : CategoryTheory.MonoidalCategory C
                                     X : CategoryTheory.Center C
                                     U : C
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                   -/
  isoMk ⟨(λ_ X.1).hom, fun U => by simp⟩
                                   /-
                                     🎉 no goals
                                   -/


/-- Auxiliary definition for the `MonoidalCategory` instance on `Center C`. -/
def rightUnitor (X : Center C) : tensorObj X tensorUnit ≅ X :=
                                   /-
                                     C : Type u₁
                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                     inst✝ : CategoryTheory.MonoidalCategory C
                                     X : CategoryTheory.Center C
                                     U : C
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                   -/
  isoMk ⟨(ρ_ X.1).hom, fun U => by simp⟩
                                   /-
                                     🎉 no goals
                                   -/


attribute [local simp] Center.associator Center.leftUnitor Center.rightUnitor


attribute [local simp] Center.whiskerLeft Center.whiskerRight Center.tensorHom


instance : MonoidalCategory (Center C) where
  tensorObj X Y := tensorObj X Y
  tensorHom f g := tensorHom f g
                      /-
                        C : Type u₁
                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                        inst✝ : CategoryTheory.MonoidalCategory C
                        ⊢ ∀ {X₁ Y₁ X₂ Y₂ : CategoryTheory.Center C} (f : Quiver.Hom X₁ Y₁) (g : Quiver …
                      -/
  tensorHom_def := by intros; ext; simp [tensorHom_def]
                                   /-
                                     🎉 no goals
                                   -/
  whiskerLeft X _ _ f := whiskerLeft X f
  whiskerRight f Y := whiskerRight f Y
  tensorUnit := tensorUnit
  associator := associator
  leftUnitor := leftUnitor
  rightUnitor := rightUnitor


@[simp]
theorem tensor_fst (X Y : Center C) : (X ⊗ Y).1 = X.1 ⊗ Y.1 :=
  rfl


@[simp]
theorem tensor_β (X Y : Center C) (U : C) :
    (X ⊗ Y).2.β U =
      α_ _ _ _ ≪≫
        (whiskerLeftIso X.1 (Y.2.β U)) ≪≫ (α_ _ _ _).symm ≪≫
          (whiskerRightIso (X.2.β U) Y.1) ≪≫ α_ _ _ _ :=
  rfl


@[simp]
theorem whiskerLeft_f (X : Center C) {Y₁ Y₂ : Center C} (f : Y₁ ⟶ Y₂) : (X ◁ f).f = X.1 ◁ f.f :=
  rfl


@[simp]
theorem whiskerRight_f {X₁ X₂ : Center C} (f : X₁ ⟶ X₂) (Y : Center C) : (f ▷ Y).f = f.f ▷ Y.1 :=
  rfl


@[simp]
theorem tensor_f {X₁ Y₁ X₂ Y₂ : Center C} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) : (f ⊗ g).f = f.f ⊗ g.f :=
  rfl


@[simp]
theorem tensorUnit_β (U : C) : (𝟙_ (Center C)).2.β U = λ_ U ≪≫ (ρ_ U).symm :=
  rfl


@[simp]
theorem associator_hom_f (X Y Z : Center C) : Hom.f (α_ X Y Z).hom = (α_ X.1 Y.1 Z.1).hom :=
  rfl


@[simp]
theorem associator_inv_f (X Y Z : Center C) : Hom.f (α_ X Y Z).inv = (α_ X.1 Y.1 Z.1).inv := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z : CategoryTheory.Center C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.associator X Y Z).inv.f (CategoryT …
  -/
  apply Iso.inv_ext' -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
  /-
    case hom_inv_id
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z : CategoryTheory.Center C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← associator_hom_f, ← comp_f, Iso.hom_inv_id]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem leftUnitor_hom_f (X : Center C) : Hom.f (λ_ X).hom = (λ_ X.1).hom :=
  rfl


@[simp]
theorem leftUnitor_inv_f (X : Center C) : Hom.f (λ_ X).inv = (λ_ X.1).inv := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X : CategoryTheory.Center C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).inv.f (CategoryTheor …
  -/
  apply Iso.inv_ext' -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
  /-
    case hom_inv_id
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X : CategoryTheory.Center C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← leftUnitor_hom_f, ← comp_f, Iso.hom_inv_id]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem rightUnitor_hom_f (X : Center C) : Hom.f (ρ_ X).hom = (ρ_ X.1).hom :=
  rfl


@[simp]
theorem rightUnitor_inv_f (X : Center C) : Hom.f (ρ_ X).inv = (ρ_ X.1).inv := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X : CategoryTheory.Center C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).inv.f (CategoryTheo …
  -/
  apply Iso.inv_ext' -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
  /-
    case hom_inv_id
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X : CategoryTheory.Center C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← rightUnitor_hom_f, ← comp_f, Iso.hom_inv_id]; rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The forgetful monoidal functor from the Drinfeld center to the original category. -/
@[simps]
def forget : Center C ⥤ C where
  obj X := X.1
  map f := f.f


instance : (forget C).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso := Iso.refl _
      μIso := fun _ _ ↦ Iso.refl _}


@[simp] lemma forget_ε : ε (forget C) = 𝟙 _ := rfl

@[simp] lemma forget_η : η (forget C) = 𝟙 _ := rfl


@[simp] lemma forget_μ (X Y : Center C) : μ (forget C) X Y = 𝟙 _ := rfl

@[simp] lemma forget_δ (X Y : Center C) : δ (forget C) X Y = 𝟙 _ := rfl


instance : (forget C).ReflectsIsomorphisms where
                     /-
                       C : Type u₁
                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                       inst✝ : CategoryTheory.MonoidalCategory C
                       A✝ B✝ : CategoryTheory.Center C
                       f : Quiver.Hom A✝ B✝
                       i : CategoryTheory.IsIso ((CategoryTheory.Center.forget C).map f)
                       ⊢ CategoryTheory.IsIso f
                     -/
  reflects f i := by dsimp at i; change IsIso (isoMk f).hom; infer_instance
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Auxiliary definition for the `BraidedCategory` instance on `Center C`. -/
@[simps!]
def braiding (X Y : Center C) : X ⊗ Y ≅ Y ⊗ X :=
  isoMk
    ⟨(X.2.β Y.1).hom, fun U => by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X Y : CategoryTheory.Center C
        U : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X Y : CategoryTheory.Center C
        U : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      simp only [Category.assoc]
      rw [← IsIso.inv_comp_eq, IsIso.Iso.inv_hom, ← HalfBraiding.monoidal_assoc,
        ← HalfBraiding.naturality_assoc, HalfBraiding.monoidal]
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X Y : CategoryTheory.Center C
        U : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      simp⟩
      /-
        🎉 no goals
      -/


instance braidedCategoryCenter : BraidedCategory (Center C) where
  braiding := braiding

-- `aesop_cat` handles the hexagon axioms

/-- Auxiliary construction for `ofBraided`. -/
@[simps]
def ofBraidedObj (X : C) : Center C :=
  ⟨X, { β := fun Y => β_ X Y}⟩


/-- The functor lifting a braided category to its center, using the braiding as the half-braiding.
-/
@[simps]
def ofBraided : C ⥤ Center C where
  obj := ofBraidedObj
  map f :=
    { f
      comm := fun U => braiding_naturality_left f U }


instance : (ofBraided C).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso :=
        { hom := { f := 𝟙 _ }
          inv := { f := 𝟙 _ } }
      μIso := fun _ _ ↦
        { hom := { f := 𝟙 _ }
          inv := { f := 𝟙 _ } } }


@[simp] lemma ofBraided_ε_f : (ε (ofBraided C)).f = 𝟙 _ := rfl

@[simp] lemma ofBraided_η_f : (η (ofBraided C)).f = 𝟙 _ := rfl


@[simp] lemma ofBraided_μ_f (X Y : C) : (μ (ofBraided C) X Y).f = 𝟙 _ := rfl

@[simp] lemma ofBraided_δ_f (X Y : C) : (δ (ofBraided C) X Y).f = 𝟙 _ := rfl


