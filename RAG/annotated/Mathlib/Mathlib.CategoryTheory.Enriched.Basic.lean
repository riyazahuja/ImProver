/-- A `V`-category is a category enriched in a monoidal category `V`.

Note that we do not assume that `V` is a concrete category,
so there may not be an "honest" underlying category at all!
-/
class EnrichedCategory (C : Type u₁) where
  Hom : C → C → V
  id (X : C) : 𝟙_ V ⟶ Hom X X
  comp (X Y Z : C) : Hom X Y ⊗ Hom Y Z ⟶ Hom X Z
  id_comp (X Y : C) : (λ_ (Hom X Y)).inv ≫ id X ▷ _ ≫ comp X X Y = 𝟙 _ := by aesop_cat
  comp_id (X Y : C) : (ρ_ (Hom X Y)).inv ≫ _ ◁ id Y ≫ comp X Y Y = 𝟙 _ := by aesop_cat
  assoc (W X Y Z : C) : (α_ _ _ _).inv ≫ comp W X Y ▷ _ ≫ comp W Y Z =
    _ ◁ comp X Y Z ≫ comp W X Z := by aesop_cat


notation X " ⟶[" V "] " Y:10 => (EnrichedCategory.Hom X Y : V)


/-- The `𝟙_ V`-shaped generalized element giving the identity in a `V`-enriched category.
-/
def eId (X : C) : 𝟙_ V ⟶ X ⟶[V] X :=
  EnrichedCategory.id X


/-- The composition `V`-morphism for a `V`-enriched category.
-/
def eComp (X Y Z : C) : ((X ⟶[V] Y) ⊗ Y ⟶[V] Z) ⟶ X ⟶[V] Z :=
  EnrichedCategory.comp X Y Z


@[reassoc (attr := simp)]
theorem e_id_comp (X Y : C) :
    (λ_ (X ⟶[V] Y)).inv ≫ eId V X ▷ _ ≫ eComp V X X Y = 𝟙 (X ⟶[V] Y) :=
  EnrichedCategory.id_comp X Y


@[reassoc (attr := simp)]
theorem e_comp_id (X Y : C) :
    (ρ_ (X ⟶[V] Y)).inv ≫ _ ◁ eId V Y ≫ eComp V X Y Y = 𝟙 (X ⟶[V] Y) :=
  EnrichedCategory.comp_id X Y


@[reassoc (attr := simp)]
theorem e_assoc (W X Y Z : C) :
    (α_ _ _ _).inv ≫ eComp V W X Y ▷ _ ≫ eComp V W Y Z =
      _ ◁ eComp V X Y Z ≫ eComp V W X Z :=
  EnrichedCategory.assoc W X Y Z


@[reassoc]
theorem e_assoc' (W X Y Z : C) :
    (α_ _ _ _).hom ≫ _ ◁ eComp V X Y Z ≫ eComp V W X Z =
      eComp V W X Y ▷ _ ≫ eComp V W Y Z := by
  /-
    V : Type v
    inst✝² : CategoryTheory.Category.{w, v} V
    inst✝¹ : CategoryTheory.MonoidalCategory V
    C : Type u₁
    inst✝ : CategoryTheory.EnrichedCategory V C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← e_assoc V W X Y Z, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


/-- A type synonym for `C`, which should come equipped with a `V`-enriched category structure.
In a moment we will equip this with the `W`-enriched category structure
obtained by applying the functor `F : LaxMonoidalFunctor V W` to each hom object.
-/
@[nolint unusedArguments]
def TransportEnrichment (F : V ⥤ W) [F.LaxMonoidal] (C : Type u₁) :=
  C


instance : EnrichedCategory W (TransportEnrichment F C) where
  Hom := fun X Y : C => F.obj (X ⟶[V] Y)
  id := fun X : C => ε F ≫ F.map (eId V X)
  comp := fun X Y Z : C => μ F _ _ ≫ F.map (eComp V X Y Z)
  id_comp X Y := by
    simp only [comp_whiskerRight, Category.assoc, Functor.LaxMonoidal.μ_natural_left_assoc,
      Functor.LaxMonoidal.left_unitality_inv_assoc]
    /-
      V : Type v
      inst✝⁵ : CategoryTheory.Category.{w, v} V
      inst✝⁴ : CategoryTheory.MonoidalCategory V
      C : Type u₁
      inst✝³ : CategoryTheory.EnrichedCategory V C
      W : Type v'
      inst✝² : CategoryTheory.Category.{w', v'} W
      inst✝¹ : CategoryTheory.MonoidalCategory W
      F : CategoryTheory.Functor V W
      inst✝ : F.LaxMonoidal
      X Y : CategoryTheory.TransportEnrichment F C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.MonoidalCatego …
    -/
    simp_rw [← F.map_comp]
    /-
      V : Type v
      inst✝⁵ : CategoryTheory.Category.{w, v} V
      inst✝⁴ : CategoryTheory.MonoidalCategory V
      C : Type u₁
      inst✝³ : CategoryTheory.EnrichedCategory V C
      W : Type v'
      inst✝² : CategoryTheory.Category.{w', v'} W
      inst✝¹ : CategoryTheory.MonoidalCategory W
      F : CategoryTheory.Functor V W
      inst✝ : F.LaxMonoidal
      X Y : CategoryTheory.TransportEnrichment F C
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
    -/
    convert F.map_id _
    /-
      case h.e'_2.h.h.e'_8
      V : Type v
      inst✝⁵ : CategoryTheory.Category.{w, v} V
      inst✝⁴ : CategoryTheory.MonoidalCategory V
      C : Type u₁
      inst✝³ : CategoryTheory.EnrichedCategory V C
      W : Type v'
      inst✝² : CategoryTheory.Category.{w', v'} W
      inst✝¹ : CategoryTheory.MonoidalCategory W
      F : CategoryTheory.Functor V W
      inst✝ : F.LaxMonoidal
      X Y : CategoryTheory.TransportEnrichment F C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp
    /-
      🎉 no goals
    -/
  comp_id X Y := by
    simp only [MonoidalCategory.whiskerLeft_comp, Category.assoc,
      Functor.LaxMonoidal.μ_natural_right_assoc,
      Functor.LaxMonoidal.right_unitality_inv_assoc]
    /-
      V : Type v
      inst✝⁵ : CategoryTheory.Category.{w, v} V
      inst✝⁴ : CategoryTheory.MonoidalCategory V
      C : Type u₁
      inst✝³ : CategoryTheory.EnrichedCategory V C
      W : Type v'
      inst✝² : CategoryTheory.Category.{w', v'} W
      inst✝¹ : CategoryTheory.MonoidalCategory W
      F : CategoryTheory.Functor V W
      inst✝ : F.LaxMonoidal
      X Y : CategoryTheory.TransportEnrichment F C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.MonoidalCatego …
    -/
    simp_rw [← F.map_comp]
    /-
      V : Type v
      inst✝⁵ : CategoryTheory.Category.{w, v} V
      inst✝⁴ : CategoryTheory.MonoidalCategory V
      C : Type u₁
      inst✝³ : CategoryTheory.EnrichedCategory V C
      W : Type v'
      inst✝² : CategoryTheory.Category.{w', v'} W
      inst✝¹ : CategoryTheory.MonoidalCategory W
      F : CategoryTheory.Functor V W
      inst✝ : F.LaxMonoidal
      X Y : CategoryTheory.TransportEnrichment F C
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
    -/
    convert F.map_id _
    /-
      case h.e'_2.h.h.e'_8
      V : Type v
      inst✝⁵ : CategoryTheory.Category.{w, v} V
      inst✝⁴ : CategoryTheory.MonoidalCategory V
      C : Type u₁
      inst✝³ : CategoryTheory.EnrichedCategory V C
      W : Type v'
      inst✝² : CategoryTheory.Category.{w', v'} W
      inst✝¹ : CategoryTheory.MonoidalCategory W
      F : CategoryTheory.Functor V W
      inst✝ : F.LaxMonoidal
      X Y : CategoryTheory.TransportEnrichment F C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp
    /-
      🎉 no goals
    -/
  assoc P Q R S := by
    rw [comp_whiskerRight, Category.assoc, μ_natural_left_assoc,
      ← associativity_inv_assoc, ← F.map_comp, ← F.map_comp, e_assoc,
      F.map_comp, MonoidalCategory.whiskerLeft_comp, Category.assoc,
      Functor.LaxMonoidal.μ_natural_right_assoc]


/-- Construct an honest category from a `Type v`-enriched category.
-/
def categoryOfEnrichedCategoryType (C : Type u₁) [𝒞 : EnrichedCategory (Type v) C] :
    Category.{v} C where
  Hom := 𝒞.Hom
  id X := eId (Type v) X PUnit.unit
  comp f g := eComp (Type v) _ _ _ ⟨f, g⟩
  id_comp f := congr_fun (e_id_comp (Type v) _ _) f
  comp_id f := congr_fun (e_comp_id (Type v) _ _) f
  assoc f g h := (congr_fun (e_assoc (Type v) _ _ _ _) ⟨f, g, h⟩ : _)


/-- Construct a `Type v`-enriched category from an honest category.
-/
def enrichedCategoryTypeOfCategory (C : Type u₁) [𝒞 : Category.{v} C] :
    EnrichedCategory (Type v) C where
  Hom := 𝒞.Hom
  id X _ := 𝟙 X
  comp _ _ _ p := p.1 ≫ p.2
                    /-
                      V : Type v
                      inst✝² : CategoryTheory.Category.{w, v} V
                      inst✝¹ : CategoryTheory.MonoidalCategory V
                      C✝ : Type u₁
                      inst✝ : CategoryTheory.EnrichedCategory V C✝
                      C : Type u₁
                      𝒞 : CategoryTheory.Category.{v, u₁} C
                      X Y : C
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                    -/
  id_comp X Y := by ext; simp
                         /-
                           🎉 no goals
                         -/
                    /-
                      V : Type v
                      inst✝² : CategoryTheory.Category.{w, v} V
                      inst✝¹ : CategoryTheory.MonoidalCategory V
                      C✝ : Type u₁
                      inst✝ : CategoryTheory.EnrichedCategory V C✝
                      C : Type u₁
                      𝒞 : CategoryTheory.Category.{v, u₁} C
                      X Y : C
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                    -/
  comp_id X Y := by ext; simp
                         /-
                           🎉 no goals
                         -/
                      /-
                        V : Type v
                        inst✝² : CategoryTheory.Category.{w, v} V
                        inst✝¹ : CategoryTheory.MonoidalCategory V
                        C✝ : Type u₁
                        inst✝ : CategoryTheory.EnrichedCategory V C✝
                        C : Type u₁
                        𝒞 : CategoryTheory.Category.{v, u₁} C
                        W X Y Z : C
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                      -/
  assoc W X Y Z := by ext ⟨f, g, h⟩; simp
                                     /-
                                       🎉 no goals
                                     -/


/-- We verify that an enriched category in `Type u` is just the same thing as an honest category.
-/
def enrichedCategoryTypeEquivCategory (C : Type u₁) :
    EnrichedCategory (Type v) C ≃ Category.{v} C where
  toFun _ := categoryOfEnrichedCategoryType C
  invFun _ := enrichedCategoryTypeOfCategory C
  left_inv _ := rfl
  right_inv _ := rfl


/-- A type synonym for `C`, which should come equipped with a `V`-enriched category structure.
In a moment we will equip this with the (honest) category structure
so that `X ⟶ Y` is `(𝟙_ W) ⟶ (X ⟶[W] Y)`.

We obtain this category by
transporting the enrichment in `V` along the lax monoidal functor `coyonedaTensorUnit`,
then using the equivalence of `Type`-enriched categories with honest categories.

This is sometimes called the "underlying" category of an enriched category,
although some care is needed as the functor `coyonedaTensorUnit`,
which always exists, does not necessarily coincide with
"the forgetful functor" from `V` to `Type`, if such exists.
When `V` is any of `Type`, `Top`, `AddCommGroup`, or `Module R`,
`coyonedaTensorUnit` is just the usual forgetful functor, however.
For `V = Algebra R`, the usual forgetful functor is coyoneda of `R[X]`, not of `R`.
(Perhaps we should have a typeclass for this situation: `ConcreteMonoidal`?)
-/
@[nolint unusedArguments]
def ForgetEnrichment (W : Type v) [Category.{w} W] [MonoidalCategory W] (C : Type u₁)
    [EnrichedCategory W C] :=
  C


/-- Typecheck an object of `C` as an object of `ForgetEnrichment W C`. -/
def ForgetEnrichment.of (X : C) : ForgetEnrichment W C :=
  X


/-- Typecheck an object of `ForgetEnrichment W C` as an object of `C`. -/
def ForgetEnrichment.to (X : ForgetEnrichment W C) : C :=
  X


@[simp]
theorem ForgetEnrichment.to_of (X : C) : ForgetEnrichment.to W (ForgetEnrichment.of W X) = X :=
  rfl


@[simp]
theorem ForgetEnrichment.of_to (X : ForgetEnrichment W C) :
    ForgetEnrichment.of W (ForgetEnrichment.to W X) = X :=
  rfl


instance categoryForgetEnrichment : Category (ForgetEnrichment W C) :=
  enrichedCategoryTypeEquivCategory C (inferInstanceAs (EnrichedCategory (Type w)
      (TransportEnrichment (coyoneda.obj (op (𝟙_ W))) C)))


/-- Typecheck a `(𝟙_ W)`-shaped `W`-morphism as a morphism in `ForgetEnrichment W C`. -/
def ForgetEnrichment.homOf {X Y : C} (f : 𝟙_ W ⟶ X ⟶[W] Y) :
    ForgetEnrichment.of W X ⟶ ForgetEnrichment.of W Y :=
  f


/-- Typecheck a morphism in `ForgetEnrichment W C` as a `(𝟙_ W)`-shaped `W`-morphism. -/
def ForgetEnrichment.homTo {X Y : ForgetEnrichment W C} (f : X ⟶ Y) :
    𝟙_ W ⟶ ForgetEnrichment.to W X ⟶[W] ForgetEnrichment.to W Y :=
  f


@[simp]
theorem ForgetEnrichment.homTo_homOf {X Y : C} (f : 𝟙_ W ⟶ X ⟶[W] Y) :
    ForgetEnrichment.homTo W (ForgetEnrichment.homOf W f) = f :=
  rfl


@[simp]
theorem ForgetEnrichment.homOf_homTo {X Y : ForgetEnrichment W C} (f : X ⟶ Y) :
    ForgetEnrichment.homOf W (ForgetEnrichment.homTo W f) = f :=
  rfl


/-- The identity in the "underlying" category of an enriched category. -/
@[simp]
theorem forgetEnrichment_id (X : ForgetEnrichment W C) :
    ForgetEnrichment.homTo W (𝟙 X) = eId W (ForgetEnrichment.to W X : C) :=
  Category.id_comp _


@[simp]
theorem forgetEnrichment_id' (X : C) :
    ForgetEnrichment.homOf W (eId W X) = 𝟙 (ForgetEnrichment.of W X : C) :=
  (forgetEnrichment_id W (ForgetEnrichment.of W X)).symm


/-- Composition in the "underlying" category of an enriched category. -/
@[simp]
theorem forgetEnrichment_comp {X Y Z : ForgetEnrichment W C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    ForgetEnrichment.homTo W (f ≫ g) =
      ((λ_ (𝟙_ W)).inv ≫ (ForgetEnrichment.homTo W f ⊗ ForgetEnrichment.homTo W g)) ≫
        eComp W _ _ _ :=
  rfl


/-- A `V`-functor `F` between `V`-enriched categories
has a `V`-morphism from `X ⟶[V] Y` to `F.obj X ⟶[V] F.obj Y`,
satisfying the usual axioms.
-/
structure EnrichedFunctor (C : Type u₁) [EnrichedCategory V C] (D : Type u₂)
    [EnrichedCategory V D] where
  obj : C → D
  map : ∀ X Y : C, (X ⟶[V] Y) ⟶ obj X ⟶[V] obj Y
  map_id : ∀ X : C, eId V X ≫ map X X = eId V (obj X) := by aesop_cat
  map_comp :
    ∀ X Y Z : C,
      eComp V X Y Z ≫ map X Z = (map X Y ⊗ map Y Z) ≫ eComp V (obj X) (obj Y) (obj Z) := by
    aesop_cat


attribute [reassoc (attr := simp)] EnrichedFunctor.map_id


attribute [reassoc (attr := simp)] EnrichedFunctor.map_comp


/-- The identity enriched functor. -/
@[simps]
def EnrichedFunctor.id (C : Type u₁) [EnrichedCategory V C] : EnrichedFunctor V C C where
  obj X := X
  map _ _ := 𝟙 _


instance : Inhabited (EnrichedFunctor V C C) :=
  ⟨EnrichedFunctor.id V C⟩


/-- Composition of enriched functors. -/
@[simps]
def EnrichedFunctor.comp {C : Type u₁} {D : Type u₂} {E : Type u₃} [EnrichedCategory V C]
    [EnrichedCategory V D] [EnrichedCategory V E] (F : EnrichedFunctor V C D)
    (G : EnrichedFunctor V D E) : EnrichedFunctor V C E where
  obj X := G.obj (F.obj X)
  map _ _ := F.map _ _ ≫ G.map _ _


lemma EnrichedFunctor.ext {C : Type u₁} {D : Type u₂} [EnrichedCategory V C]
    [EnrichedCategory V D] {F G : EnrichedFunctor V C D} (h_obj : ∀ X, F.obj X = G.obj X)
                                                  /-
                                                    V : Type v
                                                    inst✝⁴ : CategoryTheory.Category.{w, v} V
                                                    inst✝³ : CategoryTheory.MonoidalCategory V
                                                    C✝ : Type u₁
                                                    inst✝² : CategoryTheory.EnrichedCategory V C✝
                                                    C : Type u₁
                                                    D : Type u₂
                                                    inst✝¹ : CategoryTheory.EnrichedCategory V C
                                                    inst✝ : CategoryTheory.EnrichedCategory V D
                                                    F G : CategoryTheory.EnrichedFunctor V C D
                                                    h_obj : ∀ (X : C), Eq (F.obj X) (G.obj X)
                                                    X Y : C
                                                    ⊢ Eq (CategoryTheory.EnrichedCategory.Hom (F.obj X) (F.obj Y)) (CategoryTheory …
                                                  -/
    (h_map : ∀ (X Y : C), F.map X Y ≫ eqToHom (by rw [h_obj, h_obj]) = G.map X Y) : F = G := by
                                                  /-
                                                    🎉 no goals
                                                  -/
  match F, G with
  | mk F_obj F_map _ _, mk G_obj G_map _ _ =>
    obtain rfl : F_obj = G_obj := funext fun X ↦ h_obj X
    congr
    ext X Y
    simpa using h_map X Y


/-- An enriched functor induces an honest functor of the underlying categories,
by mapping the `(𝟙_ W)`-shaped morphisms.
-/
def EnrichedFunctor.forget {C : Type u₁} {D : Type u₂} [EnrichedCategory W C] [EnrichedCategory W D]
    (F : EnrichedFunctor W C D) : ForgetEnrichment W C ⥤ ForgetEnrichment W D where
  obj X := ForgetEnrichment.of W (F.obj (ForgetEnrichment.to W X))
  map f :=
    ForgetEnrichment.homOf W
      (ForgetEnrichment.homTo W f ≫ F.map (ForgetEnrichment.to W _) (ForgetEnrichment.to W _))
  map_comp f g := by
    /-
      V : Type v
      inst✝⁶ : CategoryTheory.Category.{w, v} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C✝ : Type u₁
      inst✝⁴ : CategoryTheory.EnrichedCategory V C✝
      W : Type (v + 1)
      inst✝³ : CategoryTheory.Category.{v, v + 1} W
      inst✝² : CategoryTheory.MonoidalCategory W
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.EnrichedCategory W C
      inst✝ : CategoryTheory.EnrichedCategory W D
      F : CategoryTheory.EnrichedFunctor W C D
      X✝ Y✝ Z✝ : CategoryTheory.ForgetEnrichment W C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => CategoryTheory.ForgetEnrichment.of W (F.obj (CategoryT …
    -/
    dsimp
    /-
      V : Type v
      inst✝⁶ : CategoryTheory.Category.{w, v} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C✝ : Type u₁
      inst✝⁴ : CategoryTheory.EnrichedCategory V C✝
      W : Type (v + 1)
      inst✝³ : CategoryTheory.Category.{v, v + 1} W
      inst✝² : CategoryTheory.MonoidalCategory W
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.EnrichedCategory W C
      inst✝ : CategoryTheory.EnrichedCategory W D
      F : CategoryTheory.EnrichedFunctor W C D
      X✝ Y✝ Z✝ : CategoryTheory.ForgetEnrichment W C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.ForgetEnrichment.homOf W (CategoryTheory.CategoryStruct.c …
    -/
    apply_fun ForgetEnrichment.homTo W
    · simp only [Iso.cancel_iso_inv_left, Category.assoc, tensor_comp,
        ForgetEnrichment.homTo_homOf, EnrichedFunctor.map_comp, forgetEnrichment_comp]
      /-
        V : Type v
        inst✝⁶ : CategoryTheory.Category.{w, v} V
        inst✝⁵ : CategoryTheory.MonoidalCategory V
        C✝ : Type u₁
        inst✝⁴ : CategoryTheory.EnrichedCategory V C✝
        W : Type (v + 1)
        inst✝³ : CategoryTheory.Category.{v, v + 1} W
        inst✝² : CategoryTheory.MonoidalCategory W
        C : Type u₁
        D : Type u₂
        inst✝¹ : CategoryTheory.EnrichedCategory W C
        inst✝ : CategoryTheory.EnrichedCategory W D
        F : CategoryTheory.EnrichedFunctor W C D
        X✝ Y✝ Z✝ : CategoryTheory.ForgetEnrichment W C
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case inj
        V : Type v
        inst✝⁶ : CategoryTheory.Category.{w, v} V
        inst✝⁵ : CategoryTheory.MonoidalCategory V
        C✝ : Type u₁
        inst✝⁴ : CategoryTheory.EnrichedCategory V C✝
        W : Type (v + 1)
        inst✝³ : CategoryTheory.Category.{v, v + 1} W
        inst✝² : CategoryTheory.MonoidalCategory W
        C : Type u₁
        D : Type u₂
        inst✝¹ : CategoryTheory.EnrichedCategory W C
        inst✝ : CategoryTheory.EnrichedCategory W D
        F : CategoryTheory.EnrichedFunctor W C D
        X✝ Y✝ Z✝ : CategoryTheory.ForgetEnrichment W C
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        ⊢ Function.Injective (CategoryTheory.ForgetEnrichment.homTo W)
      -/
    · intro f g w; apply_fun ForgetEnrichment.homOf W at w; simpa using w
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- The type of `A`-graded natural transformations between `V`-functors `F` and `G`.
This is the type of morphisms in `V` from `A` to the `V`-object of natural transformations.
-/
@[ext]
structure GradedNatTrans (A : Center V) (F G : EnrichedFunctor V C D) where
  app : ∀ X : C, A.1 ⟶ F.obj X ⟶[V] G.obj X
  naturality :
    ∀ X Y : C,
      (A.2.β (X ⟶[V] Y)).hom ≫ (F.map X Y ⊗ app Y) ≫ eComp V _ _ _ =
        (app X ⊗ G.map X Y) ≫ eComp V _ _ _


/-- A presheaf isomorphic to the Yoneda embedding of
the `V`-object of natural transformations from `F` to `G`.
-/
@[simps]
def enrichedNatTransYoneda (F G : EnrichedFunctor V C D) : Vᵒᵖ ⥤ Type max u₁ w where
  obj A := GradedNatTrans ((Center.ofBraided V).obj (unop A)) F G
  map f σ :=
    { app := fun X => f.unop ≫ σ.app X
      naturality := fun X Y => by
        /-
          V : Type v
          inst✝⁴ : CategoryTheory.Category.{w, v} V
          inst✝³ : CategoryTheory.MonoidalCategory V
          C : Type u₁
          inst✝² : CategoryTheory.EnrichedCategory V C
          D : Type u₂
          inst✝¹ : CategoryTheory.EnrichedCategory V D
          inst✝ : CategoryTheory.BraidedCategory V
          F G : CategoryTheory.EnrichedFunctor V C D
          X✝ Y✝ : Opposite V
          f : Quiver.Hom X✝ Y✝
          σ : (fun A => CategoryTheory.GradedNatTrans ((CategoryTheory.Center.ofBraided  …
          X Y : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Center.ofBraided V) …
        -/
        have p := σ.naturality X Y
        /-
          V : Type v
          inst✝⁴ : CategoryTheory.Category.{w, v} V
          inst✝³ : CategoryTheory.MonoidalCategory V
          C : Type u₁
          inst✝² : CategoryTheory.EnrichedCategory V C
          D : Type u₂
          inst✝¹ : CategoryTheory.EnrichedCategory V D
          inst✝ : CategoryTheory.BraidedCategory V
          F G : CategoryTheory.EnrichedFunctor V C D
          X✝ Y✝ : Opposite V
          f : Quiver.Hom X✝ Y✝
          σ : (fun A => CategoryTheory.GradedNatTrans ((CategoryTheory.Center.ofBraided  …
          X Y : C
          p : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Center.ofBraided  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Center.ofBraided V) …
        -/
        dsimp at p ⊢
        rw [← id_tensor_comp_tensor_id (f.unop ≫ σ.app Y) _, id_tensor_comp, Category.assoc,
          Category.assoc, ← braiding_naturality_assoc, id_tensor_comp_tensor_id_assoc, p, ←
          tensor_comp_assoc, Category.id_comp] }

-- TODO assuming `[HasLimits C]` construct the actual object of natural transformations
-- and show that the functor category is `V`-enriched.

/-- We verify that an enriched functor between `Type v` enriched categories
is just the same thing as an honest functor.
-/
@[simps]
def enrichedFunctorTypeEquivFunctor {C : Type u₁} [𝒞 : EnrichedCategory (Type v) C] {D : Type u₂}
    [𝒟 : EnrichedCategory (Type v) D] : EnrichedFunctor (Type v) C D ≃ C ⥤ D where
  toFun F :=
    { obj := fun X => F.obj X
      map := fun f => F.map _ _ f
      map_id := fun X => congr_fun (F.map_id X) PUnit.unit
      map_comp := fun f g => congr_fun (F.map_comp _ _ _) ⟨f, g⟩ }
  invFun F :=
    { obj := fun X => F.obj X
      map := fun _ _ f => F.map f
                            /-
                              V : Type v
                              inst✝² : CategoryTheory.Category.{w, v} V
                              inst✝¹ : CategoryTheory.MonoidalCategory V
                              C✝ : Type u₁
                              inst✝ : CategoryTheory.EnrichedCategory V C✝
                              C : Type u₁
                              𝒞 : CategoryTheory.EnrichedCategory (Type v) C
                              D : Type u₂
                              𝒟 : CategoryTheory.EnrichedCategory (Type v) D
                              F : CategoryTheory.Functor C D
                              X : C
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eId (Type v) X) ((fun …
                            -/
      map_id := fun X => by ext ⟨⟩; exact F.map_id X
                                    /-
                                      🎉 no goals
                                    -/
                                  /-
                                    V : Type v
                                    inst✝² : CategoryTheory.Category.{w, v} V
                                    inst✝¹ : CategoryTheory.MonoidalCategory V
                                    C✝ : Type u₁
                                    inst✝ : CategoryTheory.EnrichedCategory V C✝
                                    C : Type u₁
                                    𝒞 : CategoryTheory.EnrichedCategory (Type v) C
                                    D : Type u₂
                                    𝒟 : CategoryTheory.EnrichedCategory (Type v) D
                                    F : CategoryTheory.Functor C D
                                    X Y Z : C
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eComp (Type v) X Y Z) …
                                  -/
      map_comp := fun X Y Z => by ext ⟨f, g⟩; exact F.map_comp f g }
                                              /-
                                                🎉 no goals
                                              -/
  left_inv _ := rfl
  right_inv _ := rfl


/-- We verify that the presheaf representing natural transformations
between `Type v`-enriched functors is actually represented by
the usual type of natural transformations!
-/
def enrichedNatTransYonedaTypeIsoYonedaNatTrans {C : Type v} [EnrichedCategory (Type v) C]
    {D : Type v} [EnrichedCategory (Type v) D] (F G : EnrichedFunctor (Type v) C D) :
    enrichedNatTransYoneda F G ≅
      yoneda.obj (enrichedFunctorTypeEquivFunctor F ⟶ enrichedFunctorTypeEquivFunctor G) :=
  NatIso.ofComponents
    (fun α =>
      { hom := fun σ x =>
          { app := fun X => σ.app X x
            naturality := fun X Y f => congr_fun (σ.naturality X Y) ⟨x, f⟩ }
        inv := fun σ =>
          { app := fun X x => (σ x).app X
                                        /-
                                          V : Type v
                                          inst✝⁴ : CategoryTheory.Category.{w, v} V
                                          inst✝³ : CategoryTheory.MonoidalCategory V
                                          C✝ : Type u₁
                                          inst✝² : CategoryTheory.EnrichedCategory V C✝
                                          C : Type v
                                          inst✝¹ : CategoryTheory.EnrichedCategory (Type v) C
                                          D : Type v
                                          inst✝ : CategoryTheory.EnrichedCategory (Type v) D
                                          F G : CategoryTheory.EnrichedFunctor (Type v) C D
                                          α : Opposite (Type v)
                                          σ : (CategoryTheory.yoneda.obj (Quiver.Hom (CategoryTheory.enrichedFunctorType …
                                          X Y : C
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Center.ofBraided (T …
                                        -/
            naturality := fun X Y => by ext ⟨x, f⟩; exact (σ x).naturality f } })
                                                    /-
                                                      🎉 no goals
                                                    -/
        /-
          V : Type v
          inst✝⁴ : CategoryTheory.Category.{w, v} V
          inst✝³ : CategoryTheory.MonoidalCategory V
          C✝ : Type u₁
          inst✝² : CategoryTheory.EnrichedCategory V C✝
          C : Type v
          inst✝¹ : CategoryTheory.EnrichedCategory (Type v) C
          D : Type v
          inst✝ : CategoryTheory.EnrichedCategory (Type v) D
          F G : CategoryTheory.EnrichedFunctor (Type v) C D
          ⊢ ∀ {X Y : Opposite (Type v)} (f : Quiver.Hom X Y), Eq (CategoryTheory.Categor …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


