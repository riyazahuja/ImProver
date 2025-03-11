/-- An object `X` is (right) closed if `(X ⊗ -)` is a left adjoint. -/
class Closed {C : Type u} [Category.{v} C] [MonoidalCategory.{v} C] (X : C) where
  /-- a choice of a right adjoint for `tensorLeft X` -/
  rightAdj : C ⥤ C
  /-- `tensorLeft X` is a left adjoint -/
  adj : tensorLeft X ⊣ rightAdj


/-- A monoidal category `C` is (right) monoidal closed if every object is (right) closed. -/
class MonoidalClosed (C : Type u) [Category.{v} C] [MonoidalCategory.{v} C] where
  closed (X : C) : Closed X := by infer_instance


/-- If `X` and `Y` are closed then `X ⊗ Y` is.
This isn't an instance because it's not usually how we want to construct internal homs,
we'll usually prove all objects are closed uniformly.
-/
def tensorClosed {X Y : C} (hX : Closed X) (hY : Closed Y) : Closed (X ⊗ Y) where
  adj := (hY.adj.comp hX.adj).ofNatIsoLeft (MonoidalCategory.tensorLeftTensor X Y).symm


/-- The unit object is always closed.
This isn't an instance because most of the time we'll prove closedness for all objects at once,
rather than just for this one.
-/
def unitClosed : Closed (𝟙_ C) where
  rightAdj := 𝟭 C
  adj := Adjunction.id.ofNatIsoLeft (MonoidalCategory.leftUnitorNatIso C).symm


/-- This is the internal hom `A ⟶[C] -`.
-/
def ihom : C ⥤ C :=
  Closed.rightAdj (X := A)


/-- The adjunction between `A ⊗ -` and `A ⟹ -`. -/
def adjunction : tensorLeft A ⊣ ihom A :=
  Closed.adj


/-- The evaluation natural transformation. -/
def ev : ihom A ⋙ tensorLeft A ⟶ 𝟭 C :=
  (ihom.adjunction A).counit


/-- The coevaluation natural transformation. -/
def coev : 𝟭 C ⟶ tensorLeft A ⋙ ihom A :=
  (ihom.adjunction A).unit


@[simp]
theorem ihom_adjunction_counit : (ihom.adjunction A).counit = ev A :=
  rfl


@[simp]
theorem ihom_adjunction_unit : (ihom.adjunction A).unit = coev A :=
  rfl


@[reassoc (attr := simp)]
theorem ev_naturality {X Y : C} (f : X ⟶ Y) :
    A ◁ (ihom A).map f ≫ (ev A).app Y = (ev A).app X ≫ f :=
  (ev A).naturality f


@[reassoc (attr := simp)]
theorem coev_naturality {X Y : C} (f : X ⟶ Y) :
    f ≫ (coev A).app Y = (coev A).app X ≫ (ihom A).map (A ◁ f) :=
  (coev A).naturality f


set_option quotPrecheck false in
/-- `A ⟶[C] B` denotes the internal hom from `A` to `B` -/
notation A " ⟶[" C "] " B:10 => (@ihom C _ _ A _).obj B


@[reassoc (attr := simp)]
theorem ev_coev : (A ◁ (coev A).app B) ≫ (ev A).app (A ⊗ B) = 𝟙 (A ⊗ B) :=
  (ihom.adjunction A).left_triangle_components _


@[reassoc (attr := simp)]
theorem coev_ev : (coev A).app (A ⟶[C] B) ≫ (ihom A).map ((ev A).app B) = 𝟙 (A ⟶[C] B) :=
  Adjunction.right_triangle_components (ihom.adjunction A) _


instance : PreservesColimits (tensorLeft A) :=
  (ihom.adjunction A).leftAdjoint_preservesColimits


/-- Currying in a monoidal closed category. -/
def curry : (A ⊗ Y ⟶ X) → (Y ⟶ A ⟶[C] X) :=
  (ihom.adjunction A).homEquiv _ _


/-- Uncurrying in a monoidal closed category. -/
def uncurry : (Y ⟶ A ⟶[C] X) → (A ⊗ Y ⟶ X) :=
  ((ihom.adjunction A).homEquiv _ _).symm

-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[simp, nolint simpNF]
theorem homEquiv_apply_eq (f : A ⊗ Y ⟶ X) : (ihom.adjunction A).homEquiv _ _ f = curry f :=
  rfl

-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[simp, nolint simpNF]
theorem homEquiv_symm_apply_eq (f : Y ⟶ A ⟶[C] X) :
    ((ihom.adjunction A).homEquiv _ _).symm f = uncurry f :=
  rfl


@[reassoc]
theorem curry_natural_left (f : X ⟶ X') (g : A ⊗ X' ⟶ Y) : curry (_ ◁ f ≫ g) = f ≫ curry g :=
  Adjunction.homEquiv_naturality_left _ _ _


@[reassoc]
theorem curry_natural_right (f : A ⊗ X ⟶ Y) (g : Y ⟶ Y') :
    curry (f ≫ g) = curry f ≫ (ihom _).map g :=
  Adjunction.homEquiv_naturality_right _ _ _


@[reassoc]
theorem uncurry_natural_right (f : X ⟶ A ⟶[C] Y) (g : Y ⟶ Y') :
    uncurry (f ≫ (ihom _).map g) = uncurry f ≫ g :=
  Adjunction.homEquiv_naturality_right_symm _ _ _


@[reassoc]
theorem uncurry_natural_left (f : X ⟶ X') (g : X' ⟶ A ⟶[C] Y) :
    uncurry (f ≫ g) = _ ◁ f ≫ uncurry g :=
  Adjunction.homEquiv_naturality_left_symm _ _ _


@[simp]
theorem uncurry_curry (f : A ⊗ X ⟶ Y) : uncurry (curry f) = f :=
  (Closed.adj.homEquiv _ _).left_inv f


@[simp]
theorem curry_uncurry (f : X ⟶ A ⟶[C] Y) : curry (uncurry f) = f :=
  (Closed.adj.homEquiv _ _).right_inv f


theorem curry_eq_iff (f : A ⊗ Y ⟶ X) (g : Y ⟶ A ⟶[C] X) : curry f = g ↔ f = uncurry g :=
  Adjunction.homEquiv_apply_eq (ihom.adjunction A) f g


theorem eq_curry_iff (f : A ⊗ Y ⟶ X) (g : Y ⟶ A ⟶[C] X) : g = curry f ↔ uncurry g = f :=
  Adjunction.eq_homEquiv_apply (ihom.adjunction A) f g

-- I don't think these two should be simp.

theorem uncurry_eq (g : Y ⟶ A ⟶[C] X) : uncurry g = (A ◁ g) ≫ (ihom.ev A).app X := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    A X Y : C
    inst✝ : CategoryTheory.Closed A
    g : Quiver.Hom Y ((CategoryTheory.ihom A).obj X)
    ⊢ Eq (CategoryTheory.MonoidalClosed.uncurry g) (CategoryTheory.CategoryStruct. …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem curry_eq (g : A ⊗ Y ⟶ X) : curry g = (ihom.coev A).app Y ≫ (ihom A).map g :=
  rfl


theorem curry_injective : Function.Injective (curry : (A ⊗ Y ⟶ X) → (Y ⟶ A ⟶[C] X)) :=
  (Closed.adj.homEquiv _ _).injective


theorem uncurry_injective : Function.Injective (uncurry : (Y ⟶ A ⟶[C] X) → (A ⊗ Y ⟶ X)) :=
  (Closed.adj.homEquiv _ _).symm.injective


theorem uncurry_id_eq_ev : uncurry (𝟙 (A ⟶[C] X)) = (ihom.ev A).app X := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    A X : C
    inst✝ : CategoryTheory.Closed A
    ⊢ Eq (CategoryTheory.MonoidalClosed.uncurry (CategoryTheory.CategoryStruct.id  …
  -/
  simp [uncurry_eq]
  /-
    🎉 no goals
  -/


theorem curry_id_eq_coev : curry (𝟙 _) = (ihom.coev A).app X := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    A X : C
    inst✝ : CategoryTheory.Closed A
    ⊢ Eq (CategoryTheory.MonoidalClosed.curry (CategoryTheory.CategoryStruct.id (C …
  -/
  rw [curry_eq, (ihom A).map_id (A ⊗ _)]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    A X : C
    inst✝ : CategoryTheory.Closed A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ihom.coev A).app ((C …
  -/
  apply comp_id
  /-
    🎉 no goals
  -/


/-- The internal hom out of the unit is naturally isomorphic to the identity functor.-/
def unitNatIso [Closed (𝟙_ C)] : 𝟭 C ≅ ihom (𝟙_ C) :=
  conjugateIsoEquiv (Adjunction.id (C := C)) (ihom.adjunction (𝟙_ C))
    (leftUnitorNatIso C)

/-- Pre-compose an internal hom with an external hom. -/
def pre (f : B ⟶ A) : ihom A ⟶ ihom B :=
  conjugateEquiv (ihom.adjunction _) (ihom.adjunction _) ((tensoringLeft C).map f)


@[reassoc (attr := simp)]
theorem id_tensor_pre_app_comp_ev (f : B ⟶ A) (X : C) :
    B ◁ (pre f).app X ≫ (ihom.ev B).app X = f ▷ (A ⟶[C] X) ≫ (ihom.ev A).app X :=
  conjugateEquiv_counit _ _ ((tensoringLeft C).map f) X


@[simp]
theorem uncurry_pre (f : B ⟶ A) (X : C) :
    MonoidalClosed.uncurry ((pre f).app X) = f ▷ _ ≫ (ihom.ev A).app X := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.MonoidalCategory C
    A B : C
    inst✝¹ : CategoryTheory.Closed A
    inst✝ : CategoryTheory.Closed B
    f : Quiver.Hom B A
    X : C
    ⊢ Eq (CategoryTheory.MonoidalClosed.uncurry ((CategoryTheory.MonoidalClosed.pr …
  -/
  simp [uncurry_eq]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem coev_app_comp_pre_app (f : B ⟶ A) :
    (ihom.coev A).app X ≫ (pre f).app (A ⊗ X) = (ihom.coev B).app X ≫ (ihom B).map (f ▷ _) :=
  unit_conjugateEquiv _ _ ((tensoringLeft C).map f) X


@[simp]
theorem pre_id (A : C) [Closed A] : pre (𝟙 A) = 𝟙 _ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    A : C
    inst✝ : CategoryTheory.Closed A
    ⊢ Eq (CategoryTheory.MonoidalClosed.pre (CategoryTheory.CategoryStruct.id A))  …
  -/
  rw [pre, Functor.map_id]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    A : C
    inst✝ : CategoryTheory.Closed A
    ⊢ Eq ((CategoryTheory.conjugateEquiv (CategoryTheory.ihom.adjunction A) (Categ …
  -/
  apply conjugateEquiv_id
  /-
    🎉 no goals
  -/


@[simp]
theorem pre_map {A₁ A₂ A₃ : C} [Closed A₁] [Closed A₂] [Closed A₃] (f : A₁ ⟶ A₂) (g : A₂ ⟶ A₃) :
    pre (f ≫ g) = pre g ≫ pre f := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    A₁ A₂ A₃ : C
    inst✝² : CategoryTheory.Closed A₁
    inst✝¹ : CategoryTheory.Closed A₂
    inst✝ : CategoryTheory.Closed A₃
    f : Quiver.Hom A₁ A₂
    g : Quiver.Hom A₂ A₃
    ⊢ Eq (CategoryTheory.MonoidalClosed.pre (CategoryTheory.CategoryStruct.comp f  …
  -/
  rw [pre, pre, pre, conjugateEquiv_comp, (tensoringLeft C).map_comp]
  /-
    🎉 no goals
  -/


theorem pre_comm_ihom_map {W X Y Z : C} [Closed W] [Closed X] (f : W ⟶ X) (g : Y ⟶ Z) :
                                                                          /-
                                                                            C : Type u
                                                                            inst✝³ : CategoryTheory.Category.{v, u} C
                                                                            inst✝² : CategoryTheory.MonoidalCategory C
                                                                            W X Y Z : C
                                                                            inst✝¹ : CategoryTheory.Closed W
                                                                            inst✝ : CategoryTheory.Closed X
                                                                            f : Quiver.Hom W X
                                                                            g : Quiver.Hom Y Z
                                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalClosed.pre f …
                                                                          -/
    (pre f).app Y ≫ (ihom W).map g = (ihom X).map g ≫ (pre f).app Z := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- The internal hom functor given by the monoidal closed structure. -/
@[simps]
def internalHom [MonoidalClosed C] : Cᵒᵖ ⥤ C ⥤ C where
  obj X := ihom X.unop
  map f := pre f.unop


/-- Transport the property of being monoidal closed across a monoidal equivalence of categories -/
noncomputable def ofEquiv : MonoidalClosed C where
  closed X :=
    { rightAdj := F ⋙ ihom (F.obj X) ⋙ G
      adj := (adj.comp ((ihom.adjunction (F.obj X)).comp
          adj.toEquivalence.symm.toAdjunction)).ofNatIsoLeft
            (Iso.compInverseIso (H := adj.toEquivalence) (Functor.Monoidal.commTensorLeft F X)) }


/-- Suppose we have a monoidal equivalence `F : C ≌ D`, with `D` monoidal closed. We can pull the
monoidal closed instance back along the equivalence. For `X, Y, Z : C`, this lemma describes the
resulting currying map `Hom(X ⊗ Y, Z) → Hom(Y, (X ⟶[C] Z))`. (`X ⟶[C] Z` is defined to be
`F⁻¹(F(X) ⟶[D] F(Z))`, so currying in `C` is given by essentially conjugating currying in
`D` by `F.`) -/
theorem ofEquiv_curry_def {X Y Z : C} (f : X ⊗ Y ⟶ Z) :
    letI := ofEquiv F adj
    MonoidalClosed.curry f =
      adj.homEquiv Y ((ihom (F.obj X)).obj (F.obj Z))
        (MonoidalClosed.curry (adj.toEquivalence.symm.toAdjunction.homEquiv (F.obj X ⊗ F.obj Y) Z
        ((Iso.compInverseIso (H := adj.toEquivalence)
          (Functor.Monoidal.commTensorLeft F X)).hom.app Y ≫ f))) := by
  -- This whole proof used to be `rfl` before https://github.com/leanprover-community/mathlib4/pull/16317.
  change ((adj.comp ((ihom.adjunction (F.obj X)).comp
      adj.toEquivalence.symm.toAdjunction)).ofNatIsoLeft _).homEquiv _ _ _ = _
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj X Y) Z
    ⊢ Eq ((((adj.comp ((CategoryTheory.ihom.adjunction (F.obj X)).comp adj.toEquiv …
  -/
  dsimp only [Adjunction.ofNatIsoLeft]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj X Y) Z
    ⊢ Eq (((CategoryTheory.Adjunction.mkOfHomEquiv { homEquiv := fun X_1 Y => (Cat …
  -/
  rw [Adjunction.mkOfHomEquiv_homEquiv]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj X Y) Z
    ⊢ Eq (({ homEquiv := fun X_1 Y => (CategoryTheory.Adjunction.equivHomsetLeftOf …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj X Y) Z
    ⊢ Eq (((adj.comp ((CategoryTheory.ihom.adjunction (F.obj X)).comp adj.toEquiva …
  -/
  rw [Adjunction.comp_homEquiv, Adjunction.comp_homEquiv]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj X Y) Z
    ⊢ Eq (((fun x x_1 => ((fun x x_2 => (adj.toEquivalence.symm.toAdjunction.homEq …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Suppose we have a monoidal equivalence `F : C ≌ D`, with `D` monoidal closed. We can pull the
monoidal closed instance back along the equivalence. For `X, Y, Z : C`, this lemma describes the
resulting uncurrying map `Hom(Y, (X ⟶[C] Z)) → Hom(X ⊗ Y ⟶ Z)`. (`X ⟶[C] Z` is
defined to be `F⁻¹(F(X) ⟶[D] F(Z))`, so uncurrying in `C` is given by essentially conjugating
uncurrying in `D` by `F.`) -/
theorem ofEquiv_uncurry_def {X Y Z : C} :
    letI := ofEquiv F adj
    ∀ (f : Y ⟶ (ihom X).obj Z), MonoidalClosed.uncurry f =
      ((Iso.compInverseIso (H := adj.toEquivalence)
          (Functor.Monoidal.commTensorLeft F X)).inv.app Y) ≫
            (adj.toEquivalence.symm.toAdjunction.homEquiv _ _).symm
              (MonoidalClosed.uncurry ((adj.homEquiv _ _).symm f)) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    ⊢ ∀ (f : Quiver.Hom Y ((CategoryTheory.ihom X).obj Z)), Eq (CategoryTheory.Mon …
  -/
  intro f
  -- This whole proof used to be `rfl` before https://github.com/leanprover-community/mathlib4/pull/16317.
  change (((adj.comp ((ihom.adjunction (F.obj X)).comp
      adj.toEquivalence.symm.toAdjunction)).ofNatIsoLeft _).homEquiv _ _).symm _ = _
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    f : Quiver.Hom Y ((CategoryTheory.ihom X).obj Z)
    ⊢ Eq ((((adj.comp ((CategoryTheory.ihom.adjunction (F.obj X)).comp adj.toEquiv …
  -/
  dsimp only [Adjunction.ofNatIsoLeft]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    f : Quiver.Hom Y ((CategoryTheory.ihom X).obj Z)
    ⊢ Eq (((CategoryTheory.Adjunction.mkOfHomEquiv { homEquiv := fun X_1 Y => (Cat …
  -/
  rw [Adjunction.mkOfHomEquiv_homEquiv]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    f : Quiver.Hom Y ((CategoryTheory.ihom X).obj Z)
    ⊢ Eq (({ homEquiv := fun X_1 Y => (CategoryTheory.Adjunction.equivHomsetLeftOf …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    f : Quiver.Hom Y ((CategoryTheory.ihom X).obj Z)
    ⊢ Eq ((CategoryTheory.Adjunction.equivHomsetLeftOfNatIso (CategoryTheory.Funct …
  -/
  rw [Adjunction.comp_homEquiv, Adjunction.comp_homEquiv]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.Monoidal
    inst✝¹ : F.IsEquivalence
    inst✝ : CategoryTheory.MonoidalClosed D
    X Y Z : C
    f : Quiver.Hom Y ((CategoryTheory.ihom X).obj Z)
    ⊢ Eq ((CategoryTheory.Adjunction.equivHomsetLeftOfNatIso (CategoryTheory.Funct …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The C-identity morphism
  `𝟙_ C ⟶ hom(x, x)`
used to equip `C` with the structure of a `C`-category -/
def id (x : C) [Closed x] : 𝟙_ C ⟶ (ihom x).obj x := curry (ρ_ x).hom


/-- The *uncurried* composition morphism
  `x ⊗ (hom(x, y) ⊗ hom(y, z)) ⟶ (x ⊗ hom(x, y)) ⊗ hom(y, z) ⟶ y ⊗ hom(y, z) ⟶ z`.
The `C`-composition morphism will be defined as the adjoint transpose of this map. -/
def compTranspose (x y z : C) [Closed x] [Closed y] : x ⊗ (ihom x).obj y ⊗ (ihom y).obj z ⟶ z :=
  (α_ x ((ihom x).obj y) ((ihom y).obj z)).inv ≫
    (ihom.ev x).app y ▷ ((ihom y).obj z) ≫ (ihom.ev y).app z


/-- The `C`-composition morphism
  `hom(x, y) ⊗ hom(y, z) ⟶ hom(x, z)`
used to equip `C` with the structure of a `C`-category -/
def comp (x y z : C) [Closed x] [Closed y] : (ihom x).obj y ⊗ (ihom y).obj z ⟶ (ihom x).obj z :=
  curry (compTranspose x y z)


/-- Unfold the definition of `id`.
This exists to streamline the proofs of `MonoidalClosed.id_comp` and `MonoidalClosed.comp_id` -/
lemma id_eq (x : C) [Closed x] : id x = curry (ρ_ x).hom := rfl


/-- Unfold the definition of `compTranspose`.
This exists to streamline the proof of `MonoidalClosed.assoc` -/
lemma compTranspose_eq (x y z : C) [Closed x] [Closed y] :
    compTranspose x y z = (α_ _ _ _).inv ≫ (ihom.ev x).app y ▷ _ ≫ (ihom.ev y).app z :=
  rfl


/-- Unfold the definition of `comp`.
This exists to streamline the proof of `MonoidalClosed.assoc` -/
lemma comp_eq (x y z : C) [Closed x] [Closed y] : comp x y z = curry (compTranspose x y z) := rfl


/-- Left unitality of the enriched structure -/
@[reassoc (attr := simp)]
lemma id_comp (x y : C) [Closed x] :
    (λ_ ((ihom x).obj y)).inv ≫ id x ▷ _ ≫ comp x x y = 𝟙 _:= by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    x y : C
    inst✝ : CategoryTheory.Closed x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply uncurry_injective
  rw [uncurry_natural_left, uncurry_natural_left, comp_eq, uncurry_curry, id_eq, compTranspose_eq,
      associator_inv_naturality_middle_assoc, ← comp_whiskerRight_assoc, ← uncurry_eq,
      uncurry_curry, triangle_assoc_comp_right_assoc, whiskerLeft_inv_hom_assoc,
      uncurry_id_eq_ev _ _]


/-- Right unitality of the enriched structure -/
@[reassoc (attr := simp)]
lemma comp_id (x y : C) [Closed x] [Closed y] :
    (ρ_ ((ihom x).obj y)).inv ≫ _ ◁ id y ≫ comp x y y = 𝟙 _ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.MonoidalCategory C
    x y : C
    inst✝¹ : CategoryTheory.Closed x
    inst✝ : CategoryTheory.Closed y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply uncurry_injective
  rw [uncurry_natural_left, uncurry_natural_left, comp_eq, uncurry_curry, compTranspose_eq,
    associator_inv_naturality_right_assoc, ← rightUnitor_tensor_inv_assoc,
    whisker_exchange_assoc, ← rightUnitor_inv_naturality_assoc, ← uncurry_id_eq_ev y y]
  /-
    case a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.MonoidalCategory C
    x y : C
    inst✝¹ : CategoryTheory.Closed x
    inst✝ : CategoryTheory.Closed y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ihom.ev x).app y) (C …
  -/
  simp only [Functor.id_obj]
  /-
    case a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.MonoidalCategory C
    x y : C
    inst✝¹ : CategoryTheory.Closed x
    inst✝ : CategoryTheory.Closed y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ihom.ev x).app y) (C …
  -/
  rw [← uncurry_natural_left]
  /-
    case a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.MonoidalCategory C
    x y : C
    inst✝¹ : CategoryTheory.Closed x
    inst✝ : CategoryTheory.Closed y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ihom.ev x).app y) (C …
  -/
  simp [id_eq, uncurry_id_eq_ev]
  /-
    🎉 no goals
  -/


/-- Associativity of the enriched structure -/
@[reassoc]
lemma assoc (w x y z : C) [Closed w] [Closed x] [Closed y] :
    (α_ _ _ _).inv ≫ comp w x y ▷ _ ≫ comp w y z = _ ◁ comp x y z ≫ comp w x z := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    w x y z : C
    inst✝² : CategoryTheory.Closed w
    inst✝¹ : CategoryTheory.Closed x
    inst✝ : CategoryTheory.Closed y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply uncurry_injective
  /-
    case a
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    w x y z : C
    inst✝² : CategoryTheory.Closed w
    inst✝¹ : CategoryTheory.Closed x
    inst✝ : CategoryTheory.Closed y
    ⊢ Eq (CategoryTheory.MonoidalClosed.uncurry (CategoryTheory.CategoryStruct.com …
  -/
  simp only [uncurry_natural_left, comp_eq]
  /-
    case a
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    w x y z : C
    inst✝² : CategoryTheory.Closed w
    inst✝¹ : CategoryTheory.Closed x
    inst✝ : CategoryTheory.Closed y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [uncurry_curry, uncurry_curry]; simp only [compTranspose_eq, Category.assoc]
  /-
    case a
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    w x y z : C
    inst✝² : CategoryTheory.Closed w
    inst✝¹ : CategoryTheory.Closed x
    inst✝ : CategoryTheory.Closed y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [associator_inv_naturality_middle_assoc, ← comp_whiskerRight_assoc]; dsimp
  rw [← uncurry_eq, uncurry_curry, associator_inv_naturality_right_assoc, whisker_exchange_assoc,
    ← uncurry_eq, uncurry_curry]
  simp only [comp_whiskerRight, tensorLeft_obj, Category.assoc, pentagon_inv_assoc,
    whiskerRight_tensor, Iso.hom_inv_id_assoc]


