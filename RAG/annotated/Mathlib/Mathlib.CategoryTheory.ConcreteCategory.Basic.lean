/-- A concrete category is a category `C` with a fixed faithful functor `Forget : C ⥤ Type`.

Note that `ConcreteCategory` potentially depends on three independent universe levels,
* the universe level `w` appearing in `Forget : C ⥤ Type w`
* the universe level `v` of the morphisms (i.e. we have a `Category.{v} C`)
* the universe level `u` of the objects (i.e `C : Type u`)
They are specified that order, to avoid unnecessary universe annotations.
-/
class ConcreteCategory (C : Type u) [Category.{v} C] where
  /-- We have a functor to Type -/
  protected forget : C ⥤ Type w
  /-- That functor is faithful -/
  [forget_faithful : forget.Faithful]


/-- The forgetful functor from a concrete category to `Type u`. -/
abbrev forget (C : Type u) [Category.{v} C] [ConcreteCategory.{w} C] : C ⥤ Type w :=
  ConcreteCategory.forget

-- this is reducible because we want `forget (Type u)` to unfold to `𝟭 _`

@[instance] abbrev ConcreteCategory.types : ConcreteCategory.{u, u, u+1} (Type u) where
  forget := 𝟭 _


/-- Provide a coercion to `Type u` for a concrete category. This is not marked as an instance
as it could potentially apply to every type, and so is too expensive in typeclass search.

You can use it on particular examples as:
```
instance : HasCoeToSort X := ConcreteCategory.hasCoeToSort X
```
-/
def ConcreteCategory.hasCoeToSort (C : Type u) [Category.{v} C] [ConcreteCategory.{w} C] :
    CoeSort C (Type w) where
  coe X := (forget C).obj X


/-- In any concrete category, `(forget C).map` is injective. -/
abbrev ConcreteCategory.instFunLike {X Y : C} : FunLike (X ⟶ Y) X Y where
  coe f := (forget C).map f
  coe_injective' _ _ h := (forget C).map_injective h

/-- In any concrete category, we can test equality of morphisms by pointwise evaluations. -/
@[ext low] -- Porting note: lowered priority
theorem ConcreteCategory.hom_ext {X Y : C} (f g : X ⟶ Y) (w : ∀ x : X, f x = g x) : f = g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    X Y : C
    f g : Quiver.Hom X Y
    w : ∀ (x : (CategoryTheory.forget C).obj X), Eq (f x) (g x)
    ⊢ Eq f g
  -/
  apply (forget C).map_injective
  /-
    case a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    X Y : C
    f g : Quiver.Hom X Y
    w : ∀ (x : (CategoryTheory.forget C).obj X), Eq (f x) (g x)
    ⊢ Eq ((CategoryTheory.forget C).map f) ((CategoryTheory.forget C).map g)
  -/
  dsimp [forget]
  /-
    case a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    X Y : C
    f g : Quiver.Hom X Y
    w : ∀ (x : (CategoryTheory.forget C).obj X), Eq (f x) (g x)
    ⊢ Eq (CategoryTheory.ConcreteCategory.forget.map f) (CategoryTheory.ConcreteCa …
  -/
  funext x
  /-
    case a.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    X Y : C
    f g : Quiver.Hom X Y
    w : ∀ (x : (CategoryTheory.forget C).obj X), Eq (f x) (g x)
    x : CategoryTheory.ConcreteCategory.forget.obj X
    ⊢ Eq (CategoryTheory.ConcreteCategory.forget.map f x) (CategoryTheory.Concrete …
  -/
  exact w x
  /-
    🎉 no goals
  -/


theorem forget_map_eq_coe {X Y : C} (f : X ⟶ Y) : (forget C).map f = f := rfl


/-- Analogue of `congr_fun h x`,
when `h : f = g` is an equality between morphisms in a concrete category.
-/
theorem congr_hom {X Y : C} {f g : X ⟶ Y} (h : f = g) (x : X) : f x = g x :=
  congrFun (congrArg (fun k : X ⟶ Y => (k : X → Y)) h) x


theorem coe_id {X : C} : (𝟙 X : X → X) = id :=
  (forget _).map_id X


theorem coe_comp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) : (f ≫ g : X → Z) = g ∘ f :=
  (forget _).map_comp f g


@[simp] theorem id_apply {X : C} (x : X) : (𝟙 X : X → X) x = x :=
  congr_fun ((forget _).map_id X) x


@[simp] theorem comp_apply {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (x : X) : (f ≫ g) x = g (f x) :=
  congr_fun ((forget _).map_comp _ _) x


theorem comp_apply' {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (x : X) :
    (forget C).map (f ≫ g) x = (forget C).map g ((forget C).map f x) := comp_apply f g x


theorem ConcreteCategory.congr_hom {X Y : C} {f g : X ⟶ Y} (h : f = g) (x : X) : f x = g x :=
  congr_fun (congr_arg (fun f : X ⟶ Y => (f : X → Y)) h) x


theorem ConcreteCategory.congr_arg {X Y : C} (f : X ⟶ Y) {x x' : X} (h : x = x') : f x = f x' :=
  congrArg (f : X → Y) h


@[simp]
theorem ConcreteCategory.hasCoeToFun_Type {X Y : Type u} (f : X ⟶ Y) : CoeFun.coe f = f := rfl


/-- `HasForget₂ C D`, where `C` and `D` are both concrete categories, provides a functor
`forget₂ C D : C ⥤ D` and a proof that `forget₂ ⋙ (forget D) = forget C`.
-/
class HasForget₂ (C : Type u) (D : Type u') [Category.{v} C] [ConcreteCategory.{w} C]
  [Category.{v'} D] [ConcreteCategory.{w} D] where
  /-- A functor from `C` to `D` -/
  forget₂ : C ⥤ D
  /-- It covers the `ConcreteCategory.forget` for `C` and `D` -/
  forget_comp : forget₂ ⋙ forget D = forget C := by aesop


/-- The forgetful functor `C ⥤ D` between concrete categories for which we have an instance
`HasForget₂ C`. -/
abbrev forget₂ (C : Type u) (D : Type u') [Category.{v} C] [ConcreteCategory.{w} C]
    [Category.{v'} D] [ConcreteCategory.{w} D] [HasForget₂ C D] : C ⥤ D :=
  HasForget₂.forget₂


lemma forget₂_comp_apply {C : Type u} {D : Type u'} [Category.{v} C] [ConcreteCategory.{w} C]
    [Category.{v'} D] [ConcreteCategory.{w} D] [HasForget₂ C D] {X Y Z : C}
    (f : X ⟶ Y) (g : Y ⟶ Z) (x : (forget₂ C D).obj X) :
    ((forget₂ C D).map (f ≫ g) x) =
      (forget₂ C D).map g ((forget₂ C D).map f x) := by
  /-
    C : Type u
    D : Type u'
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.ConcreteCategory D
    inst✝ : CategoryTheory.HasForget₂ C D
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    x : (CategoryTheory.forget D).obj ((CategoryTheory.forget₂ C D).obj X)
    ⊢ Eq (((CategoryTheory.forget₂ C D).map (CategoryTheory.CategoryStruct.comp f  …
  -/
  rw [Functor.map_comp, comp_apply]
  /-
    🎉 no goals
  -/


instance forget₂_faithful (C : Type u) (D : Type u') [Category.{v} C] [ConcreteCategory.{w} C]
    [Category.{v'} D] [ConcreteCategory.{w} D] [HasForget₂ C D] : (forget₂ C D).Faithful :=
  HasForget₂.forget_comp.faithful_of_comp


instance InducedCategory.concreteCategory {C : Type u} {D : Type u'}
    [Category.{v'} D] [ConcreteCategory.{w} D] (f : C → D) :
      ConcreteCategory (InducedCategory D f) where
  forget := inducedFunctor f ⋙ forget D


instance InducedCategory.hasForget₂ {C : Type u} {D : Type u'} [Category.{v} D]
    [ConcreteCategory.{w} D] (f : C → D) : HasForget₂ (InducedCategory D f) D where
  forget₂ := inducedFunctor f
  forget_comp := rfl


instance FullSubcategory.concreteCategory {C : Type u} [Category.{v} C] [ConcreteCategory.{w} C]
    (Z : C → Prop) : ConcreteCategory (FullSubcategory Z) where
  forget := fullSubcategoryInclusion Z ⋙ forget C


instance FullSubcategory.hasForget₂ {C : Type u} [Category.{v} C] [ConcreteCategory.{w} C]
    (Z : C → Prop) : HasForget₂ (FullSubcategory Z) C where
  forget₂ := fullSubcategoryInclusion Z
  forget_comp := rfl


/-- In order to construct a “partially forgetting” functor, we do not need to verify functor laws;
it suffices to ensure that compositions agree with `forget₂ C D ⋙ forget D = forget C`.
-/
def HasForget₂.mk' {C : Type u} {D : Type u'} [Category.{v} C] [ConcreteCategory.{w} C]
    [Category.{v'} D] [ConcreteCategory.{w} D]
    (obj : C → D) (h_obj : ∀ X, (forget D).obj (obj X) = (forget C).obj X)
    (map : ∀ {X Y}, (X ⟶ Y) → (obj X ⟶ obj Y))
    (h_map : ∀ {X Y} {f : X ⟶ Y}, HEq ((forget D).map (map f)) ((forget C).map f)) :
    HasForget₂ C D where
  forget₂ := Functor.Faithful.div _ _ _ @h_obj _ @h_map
                    /-
                      C : Type u
                      D : Type u'
                      inst✝³ : CategoryTheory.Category.{v, u} C
                      inst✝² : CategoryTheory.ConcreteCategory C
                      inst✝¹ : CategoryTheory.Category.{v', u'} D
                      inst✝ : CategoryTheory.ConcreteCategory D
                      obj : C → D
                      h_obj : ∀ (X : C), Eq ((CategoryTheory.forget D).obj (obj X)) ((CategoryTheory …
                      map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
                      h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq ((CategoryTheory.forget D).map ( …
                      ⊢ Eq ((CategoryTheory.Functor.Faithful.div (CategoryTheory.forget C) (Category …
                    -/
  forget_comp := by apply Functor.Faithful.div_comp
                    /-
                      🎉 no goals
                    -/


/-- Composition of `HasForget₂` instances. -/
@[reducible]
def HasForget₂.trans (C : Type u) [Category.{v} C] [ConcreteCategory.{w} C]
    (D : Type u') [Category.{v'} D] [ConcreteCategory.{w} D]
    (E : Type u'') [Category.{v''} E] [ConcreteCategory.{w} E]
    [HasForget₂ C D] [HasForget₂ D E] : HasForget₂ C E where
  forget₂ := CategoryTheory.forget₂ C D ⋙ CategoryTheory.forget₂ D E
  forget_comp := by
    /-
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      inst✝⁶ : CategoryTheory.ConcreteCategory C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      E : Type u''
      inst✝³ : CategoryTheory.Category.{v'', u''} E
      inst✝² : CategoryTheory.ConcreteCategory E
      inst✝¹ : CategoryTheory.HasForget₂ C D
      inst✝ : CategoryTheory.HasForget₂ D E
      ⊢ Eq (((CategoryTheory.forget₂ C D).comp (CategoryTheory.forget₂ D E)).comp (C …
    -/
    show (CategoryTheory.forget₂ _ D) ⋙ (CategoryTheory.forget₂ D E ⋙ CategoryTheory.forget E) = _
    /-
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      inst✝⁶ : CategoryTheory.ConcreteCategory C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      E : Type u''
      inst✝³ : CategoryTheory.Category.{v'', u''} E
      inst✝² : CategoryTheory.ConcreteCategory E
      inst✝¹ : CategoryTheory.HasForget₂ C D
      inst✝ : CategoryTheory.HasForget₂ D E
      ⊢ Eq ((CategoryTheory.forget₂ C D).comp ((CategoryTheory.forget₂ D E).comp (Ca …
    -/
    simp only [HasForget₂.forget_comp]
    /-
      🎉 no goals
    -/


/-- Every forgetful functor factors through the identity functor. This is not a global instance as
    it is prone to creating type class resolution loops. -/
def hasForgetToType (C : Type u) [Category.{v} C] [ConcreteCategory.{w} C] :
    HasForget₂ C (Type w) where
  forget₂ := forget C
  forget_comp := Functor.comp_id _


@[simp]
lemma NatTrans.naturality_apply {C D : Type*} [Category C] [Category D] [ConcreteCategory D]
    {F G : C ⥤ D} (φ : F ⟶ G) {X Y : C} (f : X ⟶ Y) (x : F.obj X) :
    φ.app Y (F.map f x) = G.map f (φ.app X x) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
    inst✝ : CategoryTheory.ConcreteCategory D
    F G : CategoryTheory.Functor C D
    φ : Quiver.Hom F G
    X Y : C
    f : Quiver.Hom X Y
    x : (CategoryTheory.forget D).obj (F.obj X)
    ⊢ Eq ((φ.app Y) ((F.map f) x)) ((G.map f) ((φ.app X) x))
  -/
  simpa only [Functor.map_comp] using congr_fun ((forget D).congr_map (φ.naturality f)) x
  /-
    🎉 no goals
  -/


