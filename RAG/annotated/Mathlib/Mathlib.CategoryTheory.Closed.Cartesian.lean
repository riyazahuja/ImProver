/-- An object `X` is *exponentiable* if `(X × -)` is a left adjoint.
We define this as being `Closed` in the cartesian monoidal structure.
-/

abbrev Exponentiable {C : Type u} [Category.{v} C] [ChosenFiniteProducts C] (X : C) :=
  Closed X


/-- Constructor for `Exponentiable X` which takes as an input an adjunction
`MonoidalCategory.tensorLeft X ⊣ exp` for some functor `exp : C ⥤ C`. -/
abbrev Exponentiable.mk {C : Type u} [Category.{v} C] [ChosenFiniteProducts C] (X : C)
    (exp : C ⥤ C) (adj : MonoidalCategory.tensorLeft X ⊣ exp) :
    Exponentiable X where
  adj := adj


/-- If `X` and `Y` are exponentiable then `X ⨯ Y` is.
This isn't an instance because it's not usually how we want to construct exponentials, we'll usually
prove all objects are exponential uniformly.
-/
def binaryProductExponentiable {C : Type u} [Category.{v} C] [ChosenFiniteProducts C] {X Y : C}
    (hX : Exponentiable X) (hY : Exponentiable Y) : Exponentiable (X ⊗ Y) :=
  tensorClosed hX hY


/-- The terminal object is always exponentiable.
This isn't an instance because most of the time we'll prove cartesian closed for all objects
at once, rather than just for this one.
-/
def terminalExponentiable {C : Type u} [Category.{v} C] [ChosenFiniteProducts C] :
    Exponentiable (𝟙_ C) :=
  unitClosed


/-- A category `C` is cartesian closed if it has finite products and every object is exponentiable.
We define this as `monoidal_closed` with respect to the cartesian monoidal structure.
-/
abbrev CartesianClosed (C : Type u) [Category.{v} C] [ChosenFiniteProducts C] :=
  MonoidalClosed C

-- Porting note: added to ease the port of `CategoryTheory.Closed.Types`

/-- Constructor for `CartesianClosed C`. -/
def CartesianClosed.mk (C : Type u) [Category.{v} C] [ChosenFiniteProducts C]
    (exp : ∀ (X : C), Exponentiable X) :
    CartesianClosed C where
  closed X := exp X


/-- This is (-)^A. -/
abbrev exp : C ⥤ C :=
  ihom A


/-- The adjunction between A ⨯ - and (-)^A. -/
abbrev adjunction : tensorLeft A ⊣ exp A :=
  ihom.adjunction A


/-- The evaluation natural transformation. -/
abbrev ev : exp A ⋙ tensorLeft A ⟶ 𝟭 C :=
  ihom.ev A


/-- The coevaluation natural transformation. -/
abbrev coev : 𝟭 C ⟶ tensorLeft A ⋙ exp A :=
  ihom.coev A

-- Porting note: notation fails to elaborate with `quotPrecheck` on.

set_option quotPrecheck false in
/-- Morphisms obtained using an exponentiable object. -/
notation:20 A " ⟹ " B:19 => (exp A).obj B


open Lean PrettyPrinter.Delaborator SubExpr in
/-- Delaborator for `Prefunctor.obj` -/
@[app_delab Prefunctor.obj]
def delabPrefunctorObjExp : Delab := whenPPOption getPPNotation <| withOverApp 6 <| do
  let e ← getExpr
  guard <| e.isAppOfArity' ``Prefunctor.obj 6
  let A ← withNaryArg 4 do
    let e ← getExpr
    guard <| e.isAppOfArity' ``Functor.toPrefunctor 5
    withNaryArg 4 do
      let e ← getExpr
      guard <| e.isAppOfArity' ``exp 5
      withNaryArg 2 delab
  let B ← withNaryArg 5 delab
  `($A ⟹ $B)

-- Porting note: notation fails to elaborate with `quotPrecheck` on.

set_option quotPrecheck false in
/-- Morphisms from an exponentiable object. -/
notation:30 B " ^^ " A:30 => (exp A).obj B

-- Not simp as it can already prove it.

@[reassoc]
theorem ev_coev : (A ◁ (coev A).app B) ≫ (ev A).app (A ⊗ B) = 𝟙 (A ⊗ B : C) :=
  ihom.ev_coev A B


@[reassoc]
theorem coev_ev : (coev A).app (A ⟹ B) ≫ (exp A).map ((ev A).app B) = 𝟙 (A ⟹ B) :=
  ihom.coev_ev A B


instance : PreservesColimits (tensorLeft A) :=
  (ihom.adjunction A).leftAdjoint_preservesColimits


/-- Currying in a cartesian closed category. -/
def curry : (A ⊗ Y ⟶ X) → (Y ⟶ A ⟹ X) :=
  (exp.adjunction A).homEquiv _ _


/-- Uncurrying in a cartesian closed category. -/
def uncurry : (Y ⟶ A ⟹ X) → (A ⊗ Y ⟶ X) :=
  ((exp.adjunction A).homEquiv _ _).symm

-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[simp, nolint simpNF]
theorem homEquiv_apply_eq (f : A ⊗ Y ⟶ X) : (exp.adjunction A).homEquiv _ _ f = curry f :=
  rfl

-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[simp, nolint simpNF]
theorem homEquiv_symm_apply_eq (f : Y ⟶ A ⟹ X) :
    ((exp.adjunction A).homEquiv _ _).symm f = uncurry f :=
  rfl


@[reassoc]
theorem curry_natural_left (f : X ⟶ X') (g : A ⊗ X' ⟶ Y) :
    curry (_ ◁ f ≫ g) = f ≫ curry g :=
  Adjunction.homEquiv_naturality_left _ _ _


@[reassoc]
theorem curry_natural_right (f : A ⊗ X ⟶ Y) (g : Y ⟶ Y') :
    curry (f ≫ g) = curry f ≫ (exp _).map g :=
  Adjunction.homEquiv_naturality_right _ _ _


@[reassoc]
theorem uncurry_natural_right (f : X ⟶ A ⟹ Y) (g : Y ⟶ Y') :
    uncurry (f ≫ (exp _).map g) = uncurry f ≫ g :=
  Adjunction.homEquiv_naturality_right_symm _ _ _


@[reassoc]
theorem uncurry_natural_left (f : X ⟶ X') (g : X' ⟶ A ⟹ Y) :
    uncurry (f ≫ g) = _ ◁ f ≫ uncurry g :=
  Adjunction.homEquiv_naturality_left_symm _ _ _


@[simp]
theorem uncurry_curry (f : A ⊗ X ⟶ Y) : uncurry (curry f) = f :=
  (Closed.adj.homEquiv _ _).left_inv f


@[simp]
theorem curry_uncurry (f : X ⟶ A ⟹ Y) : curry (uncurry f) = f :=
  (Closed.adj.homEquiv _ _).right_inv f

-- Porting note: extra `(exp.adjunction A)` argument was needed for elaboration to succeed.

theorem curry_eq_iff (f : A ⊗ Y ⟶ X) (g : Y ⟶ A ⟹ X) : curry f = g ↔ f = uncurry g :=
  Adjunction.homEquiv_apply_eq (exp.adjunction A) f g

-- Porting note: extra `(exp.adjunction A)` argument was needed for elaboration to succeed.

theorem eq_curry_iff (f : A ⊗ Y ⟶ X) (g : Y ⟶ A ⟹ X) : g = curry f ↔ uncurry g = f :=
  Adjunction.eq_homEquiv_apply (exp.adjunction A) f g

-- I don't think these two should be simp.

theorem uncurry_eq (g : Y ⟶ A ⟹ X) : uncurry g = (A ◁ g) ≫ (exp.ev A).app X :=
  rfl


theorem curry_eq (g : A ⊗ Y ⟶ X) : curry g = (exp.coev A).app Y ≫ (exp A).map g :=
  rfl


theorem uncurry_id_eq_ev (A X : C) [Exponentiable A] : uncurry (𝟙 (A ⟹ X)) = (exp.ev A).app X := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts C
    A X : C
    inst✝ : CategoryTheory.Exponentiable A
    ⊢ Eq (CategoryTheory.CartesianClosed.uncurry (CategoryTheory.CategoryStruct.id …
  -/
  rw [uncurry_eq, whiskerLeft_id_assoc]
  /-
    🎉 no goals
  -/


theorem curry_id_eq_coev (A X : C) [Exponentiable A] : curry (𝟙 _) = (exp.coev A).app X := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts C
    A X : C
    inst✝ : CategoryTheory.Exponentiable A
    ⊢ Eq (CategoryTheory.CartesianClosed.curry (CategoryTheory.CategoryStruct.id ( …
  -/
  rw [curry_eq, (exp A).map_id (A ⊗ _)]; apply comp_id
                                         /-
                                           🎉 no goals
                                         -/


theorem curry_injective : Function.Injective (curry : (A ⊗ Y ⟶ X) → (Y ⟶ A ⟹ X)) :=
  (Closed.adj.homEquiv _ _).injective


theorem uncurry_injective : Function.Injective (uncurry : (Y ⟶ A ⟹ X) → (A ⊗ Y ⟶ X)) :=
  (Closed.adj.homEquiv _ _).symm.injective


/-- The exponential with the terminal object is naturally isomorphic to the identity. The typeclass
argument is explicit: any instance can be used.-/
def expUnitNatIso [Exponentiable (𝟙_ C)] : 𝟭 C ≅ exp (𝟙_ C) :=
  MonoidalClosed.unitNatIso (C := C)


/-- The exponential of any object with the terminal object is isomorphic to itself, i.e. `X^1 ≅ X`.
The typeclass argument is explicit: any instance can be used.-/
def expUnitIsoSelf [Exponentiable (𝟙_ C)] : (𝟙_ C) ⟹ X ≅ X :=
  (expUnitNatIso.app X).symm


/-- The internal element which points at the given morphism. -/
def internalizeHom (f : A ⟶ Y) : ⊤_ C ⟶ A ⟹ Y :=
  CartesianClosed.curry (ChosenFiniteProducts.fst _ _ ≫ f)


/-- Pre-compose an internal hom with an external hom. -/
def pre (f : B ⟶ A) [Exponentiable B] : exp A ⟶ exp B :=
  conjugateEquiv (exp.adjunction _) (exp.adjunction _) ((tensoringLeft _).map f)


theorem prod_map_pre_app_comp_ev (f : B ⟶ A) [Exponentiable B] (X : C) :
    (B ◁ (pre f).app X) ≫ (exp.ev B).app X =
      f ▷ (A ⟹ X) ≫ (exp.ev A).app X :=
  conjugateEquiv_counit _ _ ((tensoringLeft _).map f) X


theorem uncurry_pre (f : B ⟶ A) [Exponentiable B] (X : C) :
    CartesianClosed.uncurry ((pre f).app X) = f ▷ _ ≫ (exp.ev A).app X := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    A B : C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    inst✝¹ : CategoryTheory.Exponentiable A
    f : Quiver.Hom B A
    inst✝ : CategoryTheory.Exponentiable B
    X : C
    ⊢ Eq (CategoryTheory.CartesianClosed.uncurry ((CategoryTheory.pre f).app X)) ( …
  -/
  rw [uncurry_eq, prod_map_pre_app_comp_ev]
  /-
    🎉 no goals
  -/


theorem coev_app_comp_pre_app (f : B ⟶ A) [Exponentiable B] :
    (exp.coev A).app X ≫ (pre f).app (A ⊗ X) =
      (exp.coev B).app X ≫ (exp B).map (f ⊗ 𝟙 _) :=
  unit_conjugateEquiv _ _ ((tensoringLeft _).map f) X


@[simp]
theorem pre_id (A : C) [Exponentiable A] : pre (𝟙 A) = 𝟙 _ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts C
    A : C
    inst✝ : CategoryTheory.Exponentiable A
    ⊢ Eq (CategoryTheory.pre (CategoryTheory.CategoryStruct.id A)) (CategoryTheory …
  -/
  simp only [pre, Functor.map_id]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts C
    A : C
    inst✝ : CategoryTheory.Exponentiable A
    ⊢ Eq ((CategoryTheory.conjugateEquiv (CategoryTheory.exp.adjunction A) (Catego …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
theorem pre_map {A₁ A₂ A₃ : C} [Exponentiable A₁] [Exponentiable A₂] [Exponentiable A₃]
    (f : A₁ ⟶ A₂) (g : A₂ ⟶ A₃) : pre (f ≫ g) = pre g ≫ pre f := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    A₁ A₂ A₃ : C
    inst✝² : CategoryTheory.Exponentiable A₁
    inst✝¹ : CategoryTheory.Exponentiable A₂
    inst✝ : CategoryTheory.Exponentiable A₃
    f : Quiver.Hom A₁ A₂
    g : Quiver.Hom A₂ A₃
    ⊢ Eq (CategoryTheory.pre (CategoryTheory.CategoryStruct.comp f g)) (CategoryTh …
  -/
  rw [pre, pre, pre, conjugateEquiv_comp]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    A₁ A₂ A₃ : C
    inst✝² : CategoryTheory.Exponentiable A₁
    inst✝¹ : CategoryTheory.Exponentiable A₂
    inst✝ : CategoryTheory.Exponentiable A₃
    f : Quiver.Hom A₁ A₂
    g : Quiver.Hom A₂ A₃
    ⊢ Eq ((CategoryTheory.conjugateEquiv (CategoryTheory.exp.adjunction A₃) (Categ …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The internal hom functor given by the cartesian closed structure. -/
def internalHom [CartesianClosed C] : Cᵒᵖ ⥤ C ⥤ C where
  obj X := exp X.unop
  map f := pre f.unop


/-- If an initial object `I` exists in a CCC, then `A ⨯ I ≅ I`. -/
@[simps]
def zeroMul {I : C} (t : IsInitial I) : A ⊗ I ≅ I where
  hom := ChosenFiniteProducts.snd _ _
  inv := t.to _
  hom_inv_id := by
    have : ChosenFiniteProducts.snd A I = CartesianClosed.uncurry (t.to _) := by
      rw [← curry_eq_iff]
      apply t.hom_ext
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      A B X X' Y Y' Z : C
      inst✝¹ : CategoryTheory.ChosenFiniteProducts C
      inst✝ : CategoryTheory.Exponentiable A
      I : C
      t : CategoryTheory.Limits.IsInitial I
      this : Eq (CategoryTheory.ChosenFiniteProducts.snd A I) (CategoryTheory.Cartes …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
    -/
    rw [this, ← uncurry_natural_right, ← eq_curry_iff]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      A B X X' Y Y' Z : C
      inst✝¹ : CategoryTheory.ChosenFiniteProducts C
      inst✝ : CategoryTheory.Exponentiable A
      I : C
      t : CategoryTheory.Limits.IsInitial I
      this : Eq (CategoryTheory.ChosenFiniteProducts.snd A I) (CategoryTheory.Cartes …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.to ((CategoryTheory.exp A).obj I)) …
    -/
    apply t.hom_ext
    /-
      🎉 no goals
    -/
  inv_hom_id := t.hom_ext _ _


/-- If an initial object `0` exists in a CCC, then `0 ⨯ A ≅ 0`. -/
def mulZero {I : C} (t : IsInitial I) : I ⊗ A ≅ I :=
  β_ _ _ ≪≫ zeroMul t


/-- If an initial object `0` exists in a CCC then `0^B ≅ 1` for any `B`. -/
def powZero {I : C} (t : IsInitial I) [CartesianClosed C] : I ⟹ B ≅ ⊤_ C where
  hom := default
  inv := CartesianClosed.curry ((mulZero t).hom ≫ t.to _)
  hom_inv_id := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      A B X X' Y Y' Z : C
      inst✝² : CategoryTheory.ChosenFiniteProducts C
      inst✝¹ : CategoryTheory.Exponentiable A
      I : C
      t : CategoryTheory.Limits.IsInitial I
      inst✝ : CategoryTheory.CartesianClosed C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp Inhabited.default (CategoryTheory.Car …
    -/
    rw [← curry_natural_left, curry_eq_iff, ← cancel_epi (mulZero t).inv]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      A B X X' Y Y' Z : C
      inst✝² : CategoryTheory.ChosenFiniteProducts C
      inst✝¹ : CategoryTheory.Exponentiable A
      I : C
      t : CategoryTheory.Limits.IsInitial I
      inst✝ : CategoryTheory.CartesianClosed C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.mulZero t).inv (Categ …
    -/
    apply t.hom_ext
    /-
      🎉 no goals
    -/

-- TODO: Generalise the below to its commuted variants.
-- TODO: Define a distributive category, so that zero_mul and friends can be derived from this.

/-- In a CCC with binary coproducts, the distribution morphism is an isomorphism. -/
def prodCoprodDistrib [HasBinaryCoproducts C] [CartesianClosed C] (X Y Z : C) :
    (Z ⊗ X) ⨿ Z ⊗ Y ≅ Z ⊗ (X ⨿ Y) where
  hom := coprod.desc (_ ◁ coprod.inl) (_ ◁ coprod.inr)
  inv :=
    CartesianClosed.uncurry
      (coprod.desc (CartesianClosed.curry coprod.inl) (CartesianClosed.curry coprod.inr))
  hom_inv_id := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      A B X✝ X' Y✝ Y' Z✝ : C
      inst✝³ : CategoryTheory.ChosenFiniteProducts C
      inst✝² : CategoryTheory.Exponentiable A
      inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
      inst✝ : CategoryTheory.CartesianClosed C
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc (C …
    -/
    ext
      /-
        case h₁
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        A B X✝ X' Y✝ Y' Z✝ : C
        inst✝³ : CategoryTheory.ChosenFiniteProducts C
        inst✝² : CategoryTheory.Exponentiable A
        inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
        inst✝ : CategoryTheory.CartesianClosed C
        X Y Z : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
      -/
    · rw [coprod.inl_desc_assoc, comp_id, ← uncurry_natural_left, coprod.inl_desc, uncurry_curry]
      /-
        🎉 no goals
      -/
    /-
      case h₂
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      A B X✝ X' Y✝ Y' Z✝ : C
      inst✝³ : CategoryTheory.ChosenFiniteProducts C
      inst✝² : CategoryTheory.Exponentiable A
      inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
      inst✝ : CategoryTheory.CartesianClosed C
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
    -/
    rw [coprod.inr_desc_assoc, comp_id, ← uncurry_natural_left, coprod.inr_desc, uncurry_curry]
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      A B X✝ X' Y✝ Y' Z✝ : C
      inst✝³ : CategoryTheory.ChosenFiniteProducts C
      inst✝² : CategoryTheory.Exponentiable A
      inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
      inst✝ : CategoryTheory.CartesianClosed C
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CartesianClosed.uncur …
    -/
    rw [← uncurry_natural_right, ← eq_curry_iff]
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      A B X✝ X' Y✝ Y' Z✝ : C
      inst✝³ : CategoryTheory.ChosenFiniteProducts C
      inst✝² : CategoryTheory.Exponentiable A
      inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
      inst✝ : CategoryTheory.CartesianClosed C
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc (C …
    -/
    ext
    · rw [coprod.inl_desc_assoc, ← curry_natural_right, coprod.inl_desc, ← curry_natural_left,
        comp_id]
    rw [coprod.inr_desc_assoc, ← curry_natural_right, coprod.inr_desc, ← curry_natural_left,
      comp_id]


/-- If an initial object `I` exists in a CCC then it is a strict initial object,
i.e. any morphism to `I` is an iso.
This actually shows a slightly stronger version: any morphism to an initial object from an
exponentiable object is an isomorphism.
-/
theorem strict_initial {I : C} (t : IsInitial I) (f : A ⟶ I) : IsIso f := by
  haveI : Mono f := by
    rw [← ChosenFiniteProducts.lift_snd (𝟙 A) f, ← zeroMul_hom t]
    exact mono_comp _ _
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts C
    inst✝ : CategoryTheory.Exponentiable A
    I : C
    t : CategoryTheory.Limits.IsInitial I
    f : Quiver.Hom A I
    this : CategoryTheory.Mono f
    ⊢ CategoryTheory.IsIso f
  -/
  haveI : IsSplitEpi f := IsSplitEpi.mk' ⟨t.to _, t.hom_ext _ _⟩
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts C
    inst✝ : CategoryTheory.Exponentiable A
    I : C
    t : CategoryTheory.Limits.IsInitial I
    f : Quiver.Hom A I
    this✝ : CategoryTheory.Mono f
    this : CategoryTheory.IsSplitEpi f
    ⊢ CategoryTheory.IsIso f
  -/
  apply isIso_of_mono_of_isSplitEpi
  /-
    🎉 no goals
  -/


instance to_initial_isIso [HasInitial C] (f : A ⟶ ⊥_ C) : IsIso f :=
  strict_initial initialIsInitial _


/-- If an initial object `0` exists in a CCC then every morphism from it is monic. -/
theorem initial_mono {I : C} (B : C) (t : IsInitial I) [CartesianClosed C] : Mono (t.to B) :=
  ⟨fun g h _ => by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.ChosenFiniteProducts C
      I B : C
      t : CategoryTheory.Limits.IsInitial I
      inst✝ : CategoryTheory.CartesianClosed C
      Z✝ : C
      g h : Quiver.Hom Z✝ I
      x✝ : Eq (CategoryTheory.CategoryStruct.comp g (t.to B)) (CategoryTheory.Catego …
      ⊢ Eq g h
    -/
    haveI := strict_initial t g
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.ChosenFiniteProducts C
      I B : C
      t : CategoryTheory.Limits.IsInitial I
      inst✝ : CategoryTheory.CartesianClosed C
      Z✝ : C
      g h : Quiver.Hom Z✝ I
      x✝ : Eq (CategoryTheory.CategoryStruct.comp g (t.to B)) (CategoryTheory.Catego …
      this : CategoryTheory.IsIso g
      ⊢ Eq g h
    -/
    haveI := strict_initial t h
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.ChosenFiniteProducts C
      I B : C
      t : CategoryTheory.Limits.IsInitial I
      inst✝ : CategoryTheory.CartesianClosed C
      Z✝ : C
      g h : Quiver.Hom Z✝ I
      x✝ : Eq (CategoryTheory.CategoryStruct.comp g (t.to B)) (CategoryTheory.Catego …
      this✝ : CategoryTheory.IsIso g
      this : CategoryTheory.IsIso h
      ⊢ Eq g h
    -/
    exact eq_of_inv_eq_inv (t.hom_ext _ _)⟩
    /-
      🎉 no goals
    -/


instance Initial.mono_to [HasInitial C] (B : C) [CartesianClosed C] : Mono (initial.to B) :=
  initial_mono B initialIsInitial


/-- Transport the property of being cartesian closed across an equivalence of categories.

Note we didn't require any coherence between the choice of finite products here, since we transport
along the `prodComparison` isomorphism.
-/
def cartesianClosedOfEquiv (e : C ≌ D) [CartesianClosed C] : CartesianClosed D :=
  letI := e.inverse.monoidalOfChosenFiniteProducts
  MonoidalClosed.ofEquiv (e.inverse) e.symm.toAdjunction


