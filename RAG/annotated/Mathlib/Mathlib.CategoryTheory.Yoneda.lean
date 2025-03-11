/-- The Yoneda embedding, as a functor from `C` into presheaves on `C`.

See <https://stacks.math.columbia.edu/tag/001O>.
-/
@[simps]
def yoneda : C ⥤ Cᵒᵖ ⥤ Type v₁ where
  obj X :=
    { obj := fun Y => unop Y ⟶ X
      map := fun f g => f.unop ≫ g }
  map f :=
    { app := fun _ g => g ≫ f }


/-- The co-Yoneda embedding, as a functor from `Cᵒᵖ` into co-presheaves on `C`.
-/
@[simps]
def coyoneda : Cᵒᵖ ⥤ C ⥤ Type v₁ where
  obj X :=
    { obj := fun Y => unop X ⟶ Y
      map := fun f g => g ≫ f }
  map f :=
    { app := fun _ g => f.unop ≫ g }


theorem obj_map_id {X Y : C} (f : op X ⟶ op Y) :
    (yoneda.obj X).map f (𝟙 X) = (yoneda.map f.unop).app (op Y) (𝟙 Y) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom { unop := X } { unop := Y }
    ⊢ Eq ((CategoryTheory.yoneda.obj X).map f (CategoryTheory.CategoryStruct.id X) …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom { unop := X } { unop := Y }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop (CategoryTheory.CategoryStruct …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem naturality {X Y : C} (α : yoneda.obj X ⟶ yoneda.obj Y) {Z Z' : C} (f : Z ⟶ Z')
    (h : Z' ⟶ X) : f ≫ α.app (op Z') h = α.app (op Z) (f ≫ h) :=
  (FunctorToTypes.naturality _ _ α f.op h).symm


/-- The Yoneda embedding is fully faithful. -/
def fullyFaithful : (yoneda (C := C)).FullyFaithful where
  preimage f := f.app _ (𝟙 _)


lemma fullyFaithful_preimage {X Y : C} (f : yoneda.obj X ⟶ yoneda.obj Y) :
    fullyFaithful.preimage f = f.app (op X) (𝟙 X) := rfl


/-- The Yoneda embedding is full.

See <https://stacks.math.columbia.edu/tag/001P>.
-/
instance yoneda_full : (yoneda : C ⥤ Cᵒᵖ ⥤ Type v₁).Full :=
  fullyFaithful.full


/-- The Yoneda embedding is faithful.

See <https://stacks.math.columbia.edu/tag/001P>.
-/
instance yoneda_faithful : (yoneda : C ⥤ Cᵒᵖ ⥤ Type v₁).Faithful :=
  fullyFaithful.faithful


/-- Extensionality via Yoneda. The typical usage would be
```
-- Goal is `X ≅ Y`
apply yoneda.ext,
-- Goals are now functions `(Z ⟶ X) → (Z ⟶ Y)`, `(Z ⟶ Y) → (Z ⟶ X)`, and the fact that these
-- functions are inverses and natural in `Z`.
```
-/
def ext (X Y : C) (p : ∀ {Z : C}, (Z ⟶ X) → (Z ⟶ Y))
    (q : ∀ {Z : C}, (Z ⟶ Y) → (Z ⟶ X))
    (h₁ : ∀ {Z : C} (f : Z ⟶ X), q (p f) = f) (h₂ : ∀ {Z : C} (f : Z ⟶ Y), p (q f) = f)
    (n : ∀ {Z Z' : C} (f : Z' ⟶ Z) (g : Z ⟶ X), p (f ≫ g) = f ≫ p g) : X ≅ Y :=
  fullyFaithful.preimageIso
     /-
       C : Type u₁
       inst✝ : CategoryTheory.Category.{v₁, u₁} C
       X Y : C
       p : {Z : C} → Quiver.Hom Z X → Quiver.Hom Z Y
       q : {Z : C} → Quiver.Hom Z Y → Quiver.Hom Z X
       h₁ : ∀ {Z : C} (f : Quiver.Hom Z X), Eq (q (p f)) f
       h₂ : ∀ {Z : C} (f : Quiver.Hom Z Y), Eq (p (q f)) f
       n : ∀ {Z Z' : C} (f : Quiver.Hom Z' Z) (g : Quiver.Hom Z X), Eq (p (CategoryTh …
       ⊢ ∀ {X_1 Y_1 : Opposite C} (f : Quiver.Hom X_1 Y_1), Eq (CategoryTheory.Catego …
     -/
    (NatIso.ofComponents fun Z =>
     /-
       🎉 no goals
     -/
      { hom := p
        inv := q })


/-- If `yoneda.map f` is an isomorphism, so was `f`.
-/
theorem isIso {X Y : C} (f : X ⟶ Y) [IsIso (yoneda.map f)] : IsIso f :=
  isIso_of_fully_faithful yoneda f


@[simp]
theorem naturality {X Y : Cᵒᵖ} (α : coyoneda.obj X ⟶ coyoneda.obj Y) {Z Z' : C} (f : Z' ⟶ Z)
    (h : unop X ⟶ Z') : α.app Z' h ≫ f = α.app Z (h ≫ f) :=
  (FunctorToTypes.naturality _ _ α f h).symm


/-- The co-Yoneda embedding is fully faithful. -/
def fullyFaithful : (coyoneda (C := C)).FullyFaithful where
  preimage f := (f.app _ (𝟙 _)).op


lemma fullyFaithful_preimage {X Y : Cᵒᵖ} (f : coyoneda.obj X ⟶ coyoneda.obj Y) :
    fullyFaithful.preimage f = (f.app X.unop (𝟙 X.unop)).op := rfl


/-- The morphism `X ⟶ Y` corresponding to a natural transformation
`coyoneda.obj X ⟶ coyoneda.obj Y`. -/
def preimage {X Y : Cᵒᵖ} (f : coyoneda.obj X ⟶ coyoneda.obj Y) : X ⟶ Y :=
  (f.app _ (𝟙 X.unop)).op


instance coyoneda_full : (coyoneda : Cᵒᵖ ⥤ C ⥤ Type v₁).Full :=
  fullyFaithful.full


instance coyoneda_faithful : (coyoneda : Cᵒᵖ ⥤ C ⥤ Type v₁).Faithful :=
  fullyFaithful.faithful


/-- If `coyoneda.map f` is an isomorphism, so was `f`.
-/
theorem isIso {X Y : Cᵒᵖ} (f : X ⟶ Y) [IsIso (coyoneda.map f)] : IsIso f :=
  isIso_of_fully_faithful coyoneda f


/-- The identity functor on `Type` is isomorphic to the coyoneda functor coming from `PUnit`. -/
def punitIso : coyoneda.obj (Opposite.op PUnit) ≅ 𝟭 (Type v₁) :=
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    ⊢ ∀ {X Y : Type v₁} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.co …
  -/
  NatIso.ofComponents fun X =>
  /-
    🎉 no goals
  -/
    { hom := fun f => f ⟨⟩
      inv := fun x _ => x }


/-- Taking the `unop` of morphisms is a natural isomorphism. -/
@[simps!]
def objOpOp (X : C) : coyoneda.obj (op (op X)) ≅ yoneda.obj X :=
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    ⊢ ∀ {X_1 Y : Opposite C} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategorySt …
  -/
  NatIso.ofComponents fun _ => (opEquiv _ _).toIso
  /-
    🎉 no goals
  -/


/-- The data which expresses that a functor `F : Cᵒᵖ ⥤ Type v` is representable by `Y : C`. -/
structure RepresentableBy (F : Cᵒᵖ ⥤ Type v) (Y : C) where
  /-- the natural bijection `(X ⟶ Y) ≃ F.obj (op X)`. -/
  homEquiv {X : C} : (X ⟶ Y) ≃ F.obj (op X)
  homEquiv_comp {X X' : C} (f : X ⟶ X') (g : X' ⟶ Y) :
    homEquiv (f ≫ g) = F.map f.op (homEquiv g)


/-- If `F ≅ F'`, and `F` is representable, then `F'` is representable. -/
def RepresentableBy.ofIso {F F' : Cᵒᵖ ⥤ Type v} {Y : C} (e : F.RepresentableBy Y) (e' : F ≅ F') :
    F'.RepresentableBy Y where
  homEquiv {X} := e.homEquiv.trans (e'.app _).toEquiv
  homEquiv_comp {X X'} f g := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F F' : CategoryTheory.Functor (Opposite C) (Type v)
      Y : C
      e : F.RepresentableBy Y
      e' : CategoryTheory.Iso F F'
      X X' : C
      f : Quiver.Hom X X'
      g : Quiver.Hom X' Y
      ⊢ Eq ((fun {X} => e.homEquiv.trans (e'.app { unop := X }).toEquiv) (CategoryTh …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F F' : CategoryTheory.Functor (Opposite C) (Type v)
      Y : C
      e : F.RepresentableBy Y
      e' : CategoryTheory.Iso F F'
      X X' : C
      f : Quiver.Hom X X'
      g : Quiver.Hom X' Y
      ⊢ Eq (e'.hom.app { unop := X } (e.homEquiv (CategoryTheory.CategoryStruct.comp …
    -/
    rw [e.homEquiv_comp]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F F' : CategoryTheory.Functor (Opposite C) (Type v)
      Y : C
      e : F.RepresentableBy Y
      e' : CategoryTheory.Iso F F'
      X X' : C
      f : Quiver.Hom X X'
      g : Quiver.Hom X' Y
      ⊢ Eq (e'.hom.app { unop := X } (F.map f.op (e.homEquiv g))) (F'.map f.op (e'.h …
    -/
    apply congr_fun (e'.hom.naturality f.op)
    /-
      🎉 no goals
    -/


/-- The data which expresses that a functor `F : C ⥤ Type v` is corepresentable by `X : C`. -/
structure CorepresentableBy (F : C ⥤ Type v) (X : C) where
  /-- the natural bijection `(X ⟶ Y) ≃ F.obj Y`. -/
  homEquiv {Y : C} : (X ⟶ Y) ≃ F.obj Y
  homEquiv_comp {Y Y' : C} (g : Y ⟶ Y') (f : X ⟶ Y) :
    homEquiv (f ≫ g) = F.map g (homEquiv f)


/-- If `F ≅ F'`, and `F` is corepresentable, then `F'` is corepresentable. -/
def CorepresentableBy.ofIso {F F' : C ⥤ Type v} {X : C} (e : F.CorepresentableBy X)
    (e' : F ≅ F') :
    F'.CorepresentableBy X where
  homEquiv {X} := e.homEquiv.trans (e'.app _).toEquiv
  homEquiv_comp {Y Y'} g f := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F F' : CategoryTheory.Functor C (Type v)
      X : C
      e : F.CorepresentableBy X
      e' : CategoryTheory.Iso F F'
      Y Y' : C
      g : Quiver.Hom Y Y'
      f : Quiver.Hom X Y
      ⊢ Eq ((fun {X_1} => e.homEquiv.trans (e'.app X_1).toEquiv) (CategoryTheory.Cat …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F F' : CategoryTheory.Functor C (Type v)
      X : C
      e : F.CorepresentableBy X
      e' : CategoryTheory.Iso F F'
      Y Y' : C
      g : Quiver.Hom Y Y'
      f : Quiver.Hom X Y
      ⊢ Eq (e'.hom.app Y' (e.homEquiv (CategoryTheory.CategoryStruct.comp f g))) (F' …
    -/
    rw [e.homEquiv_comp]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F F' : CategoryTheory.Functor C (Type v)
      X : C
      e : F.CorepresentableBy X
      e' : CategoryTheory.Iso F F'
      Y Y' : C
      g : Quiver.Hom Y Y'
      f : Quiver.Hom X Y
      ⊢ Eq (e'.hom.app Y' (F.map g (e.homEquiv f))) (F'.map g (e'.hom.app Y (e.homEq …
    -/
    apply congr_fun (e'.hom.naturality g)
    /-
      🎉 no goals
    -/


lemma RepresentableBy.homEquiv_eq {F : Cᵒᵖ ⥤ Type v} {Y : C} (e : F.RepresentableBy Y)
    {X : C} (f : X ⟶ Y) :
    e.homEquiv f = F.map f.op (e.homEquiv (𝟙 Y)) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor (Opposite C) (Type v)
    Y : C
    e : F.RepresentableBy Y
    X : C
    f : Quiver.Hom X Y
    ⊢ Eq (e.homEquiv f) (F.map f.op (e.homEquiv (CategoryTheory.CategoryStruct.id  …
  -/
  conv_lhs => rw [← Category.comp_id f, e.homEquiv_comp]
  /-
    🎉 no goals
  -/


lemma CorepresentableBy.homEquiv_eq {F : C ⥤ Type v} {X : C} (e : F.CorepresentableBy X)
    {Y : C} (f : X ⟶ Y) :
    e.homEquiv f = F.map f (e.homEquiv (𝟙 X)) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor C (Type v)
    X : C
    e : F.CorepresentableBy X
    Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (e.homEquiv f) (F.map f (e.homEquiv (CategoryTheory.CategoryStruct.id X)))
  -/
  conv_lhs => rw [← Category.id_comp f, e.homEquiv_comp]
  /-
    🎉 no goals
  -/


@[ext]
lemma RepresentableBy.ext {F : Cᵒᵖ ⥤ Type v} {Y : C} {e e' : F.RepresentableBy Y}
    (h : e.homEquiv (𝟙 Y) = e'.homEquiv (𝟙 Y)) : e = e' := by
  have : ∀ {X : C} (f : X ⟶ Y), e.homEquiv f = e'.homEquiv f := fun {X} f ↦ by
    rw [e.homEquiv_eq, e'.homEquiv_eq, h]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor (Opposite C) (Type v)
    Y : C
    e e' : F.RepresentableBy Y
    h : Eq (e.homEquiv (CategoryTheory.CategoryStruct.id Y)) (e'.homEquiv (Categor …
    this : ∀ {X : C} (f : Quiver.Hom X Y), Eq (e.homEquiv f) (e'.homEquiv f)
    ⊢ Eq e e'
  -/
  obtain ⟨e, he⟩ := e
  /-
    case mk
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor (Opposite C) (Type v)
    Y : C
    e' : F.RepresentableBy Y
    e : {X : C} → Equiv (Quiver.Hom X Y) (F.obj { unop := X })
    he : ∀ {X X' : C} (f : Quiver.Hom X X') (g : Quiver.Hom X' Y), Eq (e (Category …
    h : Eq ({ homEquiv := e, homEquiv_comp := he }.homEquiv (CategoryTheory.Catego …
    this : ∀ {X : C} (f : Quiver.Hom X Y), Eq ({ homEquiv := e, homEquiv_comp := h …
    ⊢ Eq { homEquiv := e, homEquiv_comp := he } e'
  -/
  obtain ⟨e', he'⟩ := e'
  /-
    case mk.mk
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor (Opposite C) (Type v)
    Y : C
    e : {X : C} → Equiv (Quiver.Hom X Y) (F.obj { unop := X })
    he : ∀ {X X' : C} (f : Quiver.Hom X X') (g : Quiver.Hom X' Y), Eq (e (Category …
    e' : {X : C} → Equiv (Quiver.Hom X Y) (F.obj { unop := X })
    he' : ∀ {X X' : C} (f : Quiver.Hom X X') (g : Quiver.Hom X' Y), Eq (e' (Catego …
    h : Eq ({ homEquiv := e, homEquiv_comp := he }.homEquiv (CategoryTheory.Catego …
    this : ∀ {X : C} (f : Quiver.Hom X Y), Eq ({ homEquiv := e, homEquiv_comp := h …
    ⊢ Eq { homEquiv := e, homEquiv_comp := he } { homEquiv := e', homEquiv_comp := …
  -/
  obtain rfl : @e = @e' := by ext; apply this
  /-
    case mk.mk
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor (Opposite C) (Type v)
    Y : C
    e : {X : C} → Equiv (Quiver.Hom X Y) (F.obj { unop := X })
    he he' : ∀ {X X' : C} (f : Quiver.Hom X X') (g : Quiver.Hom X' Y), Eq (e (Cate …
    h : Eq ({ homEquiv := e, homEquiv_comp := he }.homEquiv (CategoryTheory.Catego …
    this : ∀ {X : C} (f : Quiver.Hom X Y), Eq ({ homEquiv := e, homEquiv_comp := h …
    ⊢ Eq { homEquiv := e, homEquiv_comp := he } { homEquiv := e, homEquiv_comp :=  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[ext]
lemma CorepresentableBy.ext {F : C ⥤ Type v} {X : C} {e e' : F.CorepresentableBy X}
    (h : e.homEquiv (𝟙 X) = e'.homEquiv (𝟙 X)) : e = e' := by
  have : ∀ {Y : C} (f : X ⟶ Y), e.homEquiv f = e'.homEquiv f := fun {X} f ↦ by
    rw [e.homEquiv_eq, e'.homEquiv_eq, h]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor C (Type v)
    X : C
    e e' : F.CorepresentableBy X
    h : Eq (e.homEquiv (CategoryTheory.CategoryStruct.id X)) (e'.homEquiv (Categor …
    this : ∀ {Y : C} (f : Quiver.Hom X Y), Eq (e.homEquiv f) (e'.homEquiv f)
    ⊢ Eq e e'
  -/
  obtain ⟨e, he⟩ := e
  /-
    case mk
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor C (Type v)
    X : C
    e' : F.CorepresentableBy X
    e : {Y : C} → Equiv (Quiver.Hom X Y) (F.obj Y)
    he : ∀ {Y Y' : C} (g : Quiver.Hom Y Y') (f : Quiver.Hom X Y), Eq (e (CategoryT …
    h : Eq ({ homEquiv := e, homEquiv_comp := he }.homEquiv (CategoryTheory.Catego …
    this : ∀ {Y : C} (f : Quiver.Hom X Y), Eq ({ homEquiv := e, homEquiv_comp := h …
    ⊢ Eq { homEquiv := e, homEquiv_comp := he } e'
  -/
  obtain ⟨e', he'⟩ := e'
  /-
    case mk.mk
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor C (Type v)
    X : C
    e : {Y : C} → Equiv (Quiver.Hom X Y) (F.obj Y)
    he : ∀ {Y Y' : C} (g : Quiver.Hom Y Y') (f : Quiver.Hom X Y), Eq (e (CategoryT …
    e' : {Y : C} → Equiv (Quiver.Hom X Y) (F.obj Y)
    he' : ∀ {Y Y' : C} (g : Quiver.Hom Y Y') (f : Quiver.Hom X Y), Eq (e' (Categor …
    h : Eq ({ homEquiv := e, homEquiv_comp := he }.homEquiv (CategoryTheory.Catego …
    this : ∀ {Y : C} (f : Quiver.Hom X Y), Eq ({ homEquiv := e, homEquiv_comp := h …
    ⊢ Eq { homEquiv := e, homEquiv_comp := he } { homEquiv := e', homEquiv_comp := …
  -/
  obtain rfl : @e = @e' := by ext; apply this
  /-
    case mk.mk
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor C (Type v)
    X : C
    e : {Y : C} → Equiv (Quiver.Hom X Y) (F.obj Y)
    he he' : ∀ {Y Y' : C} (g : Quiver.Hom Y Y') (f : Quiver.Hom X Y), Eq (e (Categ …
    h : Eq ({ homEquiv := e, homEquiv_comp := he }.homEquiv (CategoryTheory.Catego …
    this : ∀ {Y : C} (f : Quiver.Hom X Y), Eq ({ homEquiv := e, homEquiv_comp := h …
    ⊢ Eq { homEquiv := e, homEquiv_comp := he } { homEquiv := e, homEquiv_comp :=  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The obvious bijection `F.RepresentableBy Y ≃ (yoneda.obj Y ≅ F)`
when `F : Cᵒᵖ ⥤ Type v₁` and `[Category.{v₁} C]`. -/
def representableByEquiv {F : Cᵒᵖ ⥤ Type v₁} {Y : C} :
    F.RepresentableBy Y ≃ (yoneda.obj Y ≅ F) where
  toFun r := NatIso.ofComponents (fun _ ↦ r.homEquiv.toIso) (fun {X X'} f ↦ by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (Opposite C) (Type v₁)
      Y : C
      r : F.RepresentableBy Y
      X X' : Opposite C
      f : Quiver.Hom X X'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.obj Y).map f) …
    -/
    ext g
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (Opposite C) (Type v₁)
      Y : C
      r : F.RepresentableBy Y
      X X' : Opposite C
      f : Quiver.Hom X X'
      g : (CategoryTheory.yoneda.obj Y).obj X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.obj Y).map f) …
    -/
    simp [r.homEquiv_comp])
    /-
      🎉 no goals
    -/
  invFun e :=
    { homEquiv := (e.app _).toEquiv
      homEquiv_comp := fun {X X'} f g ↦ congr_fun (e.hom.naturality f.op) g }
  left_inv _ := rfl
  right_inv _ := rfl


/-- The isomorphism `yoneda.obj Y ≅ F` induced by `e : F.RepresentableBy Y`. -/
def RepresentableBy.toIso {F : Cᵒᵖ ⥤ Type v₁} {Y : C} (e : F.RepresentableBy Y) :
    yoneda.obj Y ≅ F :=
  representableByEquiv e


/-- The obvious bijection `F.CorepresentableBy X ≃ (yoneda.obj Y ≅ F)`
when `F : C ⥤ Type v₁` and `[Category.{v₁} C]`. -/
def corepresentableByEquiv {F : C ⥤ Type v₁} {X : C} :
    F.CorepresentableBy X ≃ (coyoneda.obj (op X) ≅ F) where
  toFun r := NatIso.ofComponents (fun _ ↦ r.homEquiv.toIso) (fun {X X'} f ↦ by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C (Type v₁)
      X✝ : C
      r : F.CorepresentableBy X✝
      X X' : C
      f : Quiver.Hom X X'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.obj { unop  …
    -/
    ext g
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C (Type v₁)
      X✝ : C
      r : F.CorepresentableBy X✝
      X X' : C
      f : Quiver.Hom X X'
      g : (CategoryTheory.coyoneda.obj { unop := X✝ }).obj X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.obj { unop  …
    -/
    simp [r.homEquiv_comp])
    /-
      🎉 no goals
    -/
  invFun e :=
    { homEquiv := (e.app _).toEquiv
      homEquiv_comp := fun {X X'} f g ↦ congr_fun (e.hom.naturality f) g }
  left_inv _ := rfl
  right_inv _ := rfl


/-- The isomorphism `coyoneda.obj (op X) ≅ F` induced by `e : F.CorepresentableBy X`. -/
def CorepresentableBy.toIso {F : C ⥤ Type v₁} {X : C} (e : F.CorepresentableBy X) :
    coyoneda.obj (op X) ≅ F :=
  corepresentableByEquiv e


/-- A functor `F : Cᵒᵖ ⥤ Type v` is representable if there is an object `Y` with a structure
`F.RepresentableBy Y`, i.e. there is a natural bijection `(X ⟶ Y) ≃ F.obj (op X)`,
which may also be rephrased as a natural isomorphism `yoneda.obj X ≅ F` when `Category.{v} C`.

See <https://stacks.math.columbia.edu/tag/001Q>.
-/
class IsRepresentable (F : Cᵒᵖ ⥤ Type v) : Prop where
  has_representation : ∃ (Y : C), Nonempty (F.RepresentableBy Y)


@[deprecated (since := "2024-10-03")] alias Representable := IsRepresentable


lemma RepresentableBy.isRepresentable {F : Cᵒᵖ ⥤ Type v} {Y : C} (e : F.RepresentableBy Y) :
    F.IsRepresentable where
  has_representation := ⟨Y, ⟨e⟩⟩


/-- Alternative constructor for `F.IsRepresentable`, which takes as an input an
isomorphism `yoneda.obj X ≅ F`. -/
lemma IsRepresentable.mk' {F : Cᵒᵖ ⥤ Type v₁} {X : C} (e : yoneda.obj X ≅ F) :
    F.IsRepresentable :=
  (representableByEquiv.symm e).isRepresentable


instance {X : C} : IsRepresentable (yoneda.obj X) :=
  IsRepresentable.mk' (Iso.refl _)


/-- A functor `F : C ⥤ Type v₁` is corepresentable if there is object `X` so `F ≅ coyoneda.obj X`.

See <https://stacks.math.columbia.edu/tag/001Q>.
-/
class IsCorepresentable (F : C ⥤ Type v) : Prop where
  has_corepresentation : ∃ (X : C), Nonempty (F.CorepresentableBy X)


@[deprecated (since := "2024-10-03")] alias Corepresentable := IsCorepresentable


lemma CorepresentableBy.isCorepresentable {F : C ⥤ Type v} {X : C} (e : F.CorepresentableBy X) :
    F.IsCorepresentable where
  has_corepresentation := ⟨X, ⟨e⟩⟩


/-- Alternative constructor for `F.IsCorepresentable`, which takes as an input an
isomorphism `coyoneda.obj (op X) ≅ F`. -/
lemma IsCorepresentable.mk' {F : C ⥤ Type v₁} {X : C} (e : coyoneda.obj (op X) ≅ F) :
    F.IsCorepresentable :=
  (corepresentableByEquiv.symm e).isCorepresentable


instance {X : Cᵒᵖ} : IsCorepresentable (coyoneda.obj X) :=
  IsCorepresentable.mk' (Iso.refl _)

-- instance : corepresentable (𝟭 (Type v₁)) :=
-- corepresentable_of_nat_iso (op punit) coyoneda.punit_iso

/-- The representing object for the representable functor `F`. -/
noncomputable def reprX : C :=
  hF.has_representation.choose


/-- A chosen term in `F.RepresentableBy (reprX F)` when `F.IsRepresentable` holds. -/
noncomputable def representableBy : F.RepresentableBy F.reprX :=
  hF.has_representation.choose_spec.some


/-- The representing element for the representable functor `F`, sometimes called the universal
element of the functor.
-/
noncomputable def reprx : F.obj (op F.reprX) :=
  F.representableBy.homEquiv (𝟙 _)


/-- An isomorphism between a representable `F` and a functor of the
form `C(-, F.reprX)`.  Note the components `F.reprW.app X`
definitionally have type `(X.unop ⟶ F.reprX) ≅ F.obj X`.
-/
noncomputable def reprW (F : Cᵒᵖ ⥤ Type v₁) [F.IsRepresentable] :
    yoneda.obj F.reprX ≅ F := F.representableBy.toIso


theorem reprW_hom_app (F : Cᵒᵖ ⥤ Type v₁) [F.IsRepresentable]
    (X : Cᵒᵖ) (f : unop X ⟶ F.reprX) :
    F.reprW.hom.app X f = F.map f.op F.reprx := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    inst✝ : F.IsRepresentable
    X : Opposite C
    f : Quiver.Hom (Opposite.unop X) F.reprX
    ⊢ Eq (F.reprW.hom.app X f) (F.map f.op F.reprx)
  -/
  apply RepresentableBy.homEquiv_eq
  /-
    🎉 no goals
  -/


/-- The representing object for the corepresentable functor `F`. -/
noncomputable def coreprX : C :=
  hF.has_corepresentation.choose


/-- A chosen term in `F.CorepresentableBy (coreprX F)` when `F.IsCorepresentable` holds. -/
noncomputable def corepresentableBy : F.CorepresentableBy F.coreprX :=
  hF.has_corepresentation.choose_spec.some


/-- The representing element for the corepresentable functor `F`, sometimes called the universal
element of the functor.
-/
noncomputable def coreprx : F.obj F.coreprX :=
  F.corepresentableBy.homEquiv (𝟙 _)


/-- An isomorphism between a corepresentable `F` and a functor of the form
`C(F.corepr X, -)`. Note the components `F.coreprW.app X`
definitionally have type `F.corepr_X ⟶ X ≅ F.obj X`.
-/
noncomputable def coreprW (F : C ⥤ Type v₁) [F.IsCorepresentable] :
    coyoneda.obj (op F.coreprX) ≅ F :=
  F.corepresentableBy.toIso


theorem coreprW_hom_app (F : C ⥤ Type v₁) [F.IsCorepresentable] (X : C) (f : F.coreprX ⟶ X) :
    F.coreprW.hom.app X f = F.map f F.coreprx := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor C (Type v₁)
    inst✝ : F.IsCorepresentable
    X : C
    f : Quiver.Hom F.coreprX X
    ⊢ Eq (F.coreprW.hom.app X f) (F.map f F.coreprx)
  -/
  apply CorepresentableBy.homEquiv_eq
  /-
    🎉 no goals
  -/


theorem isRepresentable_of_natIso (F : Cᵒᵖ ⥤ Type v₁) {G} (i : F ≅ G) [F.IsRepresentable] :
    G.IsRepresentable :=
  (F.representableBy.ofIso i).isRepresentable


theorem corepresentable_of_natIso (F : C ⥤ Type v₁) {G} (i : F ≅ G) [F.IsCorepresentable] :
    G.IsCorepresentable :=
  (F.corepresentableBy.ofIso i).isCorepresentable


instance : Functor.IsCorepresentable (𝟭 (Type v₁)) :=
  corepresentable_of_natIso (coyoneda.obj (op PUnit)) Coyoneda.punitIso


instance prodCategoryInstance1 : Category ((Cᵒᵖ ⥤ Type v₁) × Cᵒᵖ) :=
  CategoryTheory.prod.{max u₁ v₁, v₁} (Cᵒᵖ ⥤ Type v₁) Cᵒᵖ


instance prodCategoryInstance2 : Category (Cᵒᵖ × (Cᵒᵖ ⥤ Type v₁)) :=
  CategoryTheory.prod.{v₁, max u₁ v₁} Cᵒᵖ (Cᵒᵖ ⥤ Type v₁)


/-- We have a type-level equivalence between natural transformations from the yoneda embedding
and elements of `F.obj X`, without any universe switching.
-/
def yonedaEquiv {X : C} {F : Cᵒᵖ ⥤ Type v₁} : (yoneda.obj X ⟶ F) ≃ F.obj (op X) where
  toFun η := η.app (op X) (𝟙 X)
  invFun ξ := { app := fun _ f ↦ F.map f.op ξ }
  left_inv := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F : CategoryTheory.Functor (Opposite C) (Type v₁)
      ⊢ Function.LeftInverse (fun ξ => { app := fun x f => F.map (Quiver.Hom.op f) ξ …
    -/
    intro η
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F : CategoryTheory.Functor (Opposite C) (Type v₁)
      η : Quiver.Hom (CategoryTheory.yoneda.obj X) F
      ⊢ Eq ((fun ξ => { app := fun x f => F.map (Quiver.Hom.op f) ξ, naturality := ⋯ …
    -/
    ext Y f
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F : CategoryTheory.Functor (Opposite C) (Type v₁)
      η : Quiver.Hom (CategoryTheory.yoneda.obj X) F
      Y : Opposite C
      f : (CategoryTheory.yoneda.obj X).obj Y
      ⊢ Eq (((fun ξ => { app := fun x f => F.map (Quiver.Hom.op f) ξ, naturality :=  …
    -/
    dsimp
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F : CategoryTheory.Functor (Opposite C) (Type v₁)
      η : Quiver.Hom (CategoryTheory.yoneda.obj X) F
      Y : Opposite C
      f : (CategoryTheory.yoneda.obj X).obj Y
      ⊢ Eq (F.map (Quiver.Hom.op f) (η.app { unop := X } (CategoryTheory.CategoryStr …
    -/
    rw [← FunctorToTypes.naturality]
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F : CategoryTheory.Functor (Opposite C) (Type v₁)
      η : Quiver.Hom (CategoryTheory.yoneda.obj X) F
      Y : Opposite C
      f : (CategoryTheory.yoneda.obj X).obj Y
      ⊢ Eq (η.app Y ((CategoryTheory.yoneda.obj X).map (Quiver.Hom.op f) (CategoryTh …
    -/
    simp
    /-
      🎉 no goals
    -/
                  /-
                    C : Type u₁
                    inst✝ : CategoryTheory.Category.{v₁, u₁} C
                    X : C
                    F : CategoryTheory.Functor (Opposite C) (Type v₁)
                    ⊢ Function.RightInverse (fun ξ => { app := fun x f => F.map (Quiver.Hom.op f)  …
                  -/
  right_inv := by intro ξ; simp
                           /-
                             🎉 no goals
                           -/


theorem yonedaEquiv_apply {X : C} {F : Cᵒᵖ ⥤ Type v₁} (f : yoneda.obj X ⟶ F) :
    yonedaEquiv f = f.app (op X) (𝟙 X) :=
  rfl


@[simp]
theorem yonedaEquiv_symm_app_apply {X : C} {F : Cᵒᵖ ⥤ Type v₁} (x : F.obj (op X)) (Y : Cᵒᵖ)
    (f : Y.unop ⟶ X) : (yonedaEquiv.symm x).app Y f = F.map f.op x :=
  rfl


/-- See also `yonedaEquiv_naturality'` for a more general version. -/
lemma yonedaEquiv_naturality {X Y : C} {F : Cᵒᵖ ⥤ Type v₁} (f : yoneda.obj X ⟶ F)
    (g : Y ⟶ X) : F.map g.op (yonedaEquiv f) = yonedaEquiv (yoneda.map g ≫ f) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    f : Quiver.Hom (CategoryTheory.yoneda.obj X) F
    g : Quiver.Hom Y X
    ⊢ Eq (F.map g.op (CategoryTheory.yonedaEquiv f)) (CategoryTheory.yonedaEquiv ( …
  -/
  change (f.app (op X) ≫ F.map g.op) (𝟙 X) = f.app (op Y) (𝟙 Y ≫ g)
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    f : Quiver.Hom (CategoryTheory.yoneda.obj X) F
    g : Quiver.Hom Y X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := X }) (F.map g.op) (C …
  -/
  rw [← f.naturality]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    f : Quiver.Hom (CategoryTheory.yoneda.obj X) F
    g : Quiver.Hom Y X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.obj X).map g. …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    f : Quiver.Hom (CategoryTheory.yoneda.obj X) F
    g : Quiver.Hom Y X
    ⊢ Eq (f.app { unop := Y } (CategoryTheory.CategoryStruct.comp g (CategoryTheor …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Variant of `yonedaEquiv_naturality` with general `g`. This is technically strictly more general
    than `yonedaEquiv_naturality`, but `yonedaEquiv_naturality` is sometimes preferable because it
    can avoid the "motive is not type correct" error. -/
lemma yonedaEquiv_naturality' {X Y : Cᵒᵖ} {F : Cᵒᵖ ⥤ Type v₁} (f : yoneda.obj (unop X) ⟶ F)
    (g : X ⟶ Y) : F.map g (yonedaEquiv f) = yonedaEquiv (yoneda.map g.unop ≫ f) :=
  yonedaEquiv_naturality _ _


lemma yonedaEquiv_comp {X : C} {F G : Cᵒᵖ ⥤ Type v₁} (α : yoneda.obj X ⟶ F) (β : F ⟶ G) :
    yonedaEquiv (α ≫ β) = β.app _ (yonedaEquiv α) :=
  rfl


lemma yonedaEquiv_yoneda_map {X Y : C} (f : X ⟶ Y) : yonedaEquiv (yoneda.map f) = f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.yoneda.map f)) f
  -/
  rw [yonedaEquiv_apply]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.yoneda.map f).app { unop := X } (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma yonedaEquiv_symm_naturality_left {X X' : C} (f : X' ⟶ X) (F : Cᵒᵖ ⥤ Type v₁)
    (x : F.obj ⟨X⟩) : yoneda.map f ≫ yonedaEquiv.symm x = yonedaEquiv.symm ((F.map f.op) x) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X X' : C
    f : Quiver.Hom X' X
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : F.obj { unop := X }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map f) (Catego …
  -/
  apply yonedaEquiv.injective
  /-
    case a
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X X' : C
    f : Quiver.Hom X' X
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : F.obj { unop := X }
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp (Category …
  -/
  simp only [yonedaEquiv_comp, yoneda_obj_obj, yonedaEquiv_symm_app_apply, Equiv.apply_symm_apply]
  /-
    case a
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X X' : C
    f : Quiver.Hom X' X
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : F.obj { unop := X }
    ⊢ Eq (F.map (CategoryTheory.yonedaEquiv (CategoryTheory.yoneda.map f)).op x) ( …
  -/
  erw [yonedaEquiv_yoneda_map]
  /-
    🎉 no goals
  -/


lemma yonedaEquiv_symm_naturality_right (X : C) {F F' : Cᵒᵖ ⥤ Type v₁} (f : F ⟶ F')
    (x : F.obj ⟨X⟩) : yonedaEquiv.symm x ≫ f = yonedaEquiv.symm (f.app ⟨X⟩ x) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    F F' : CategoryTheory.Functor (Opposite C) (Type v₁)
    f : Quiver.Hom F F'
    x : F.obj { unop := X }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yonedaEquiv.symm x) f …
  -/
  apply yonedaEquiv.injective
  /-
    case a
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    F F' : CategoryTheory.Functor (Opposite C) (Type v₁)
    f : Quiver.Hom F F'
    x : F.obj { unop := X }
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp (Category …
  -/
  simp [yonedaEquiv_comp]
  /-
    🎉 no goals
  -/


/-- See also `map_yonedaEquiv'` for a more general version. -/
lemma map_yonedaEquiv {X Y : C} {F : Cᵒᵖ ⥤ Type v₁} (f : yoneda.obj X ⟶ F)
    (g : Y ⟶ X) : F.map g.op (yonedaEquiv f) = f.app (op Y) g := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    f : Quiver.Hom (CategoryTheory.yoneda.obj X) F
    g : Quiver.Hom Y X
    ⊢ Eq (F.map g.op (CategoryTheory.yonedaEquiv f)) (f.app { unop := Y } g)
  -/
  rw [yonedaEquiv_naturality, yonedaEquiv_comp, yonedaEquiv_yoneda_map]
  /-
    🎉 no goals
  -/


/-- Variant of `map_yonedaEquiv` with general `g`. This is technically strictly more general
    than `map_yonedaEquiv`, but `map_yonedaEquiv` is sometimes preferable because it
    can avoid the "motive is not type correct" error. -/
lemma map_yonedaEquiv' {X Y : Cᵒᵖ} {F : Cᵒᵖ ⥤ Type v₁} (f : yoneda.obj (unop X) ⟶ F)
    (g : X ⟶ Y) : F.map g (yonedaEquiv f) = f.app Y g.unop := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : Opposite C
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    f : Quiver.Hom (CategoryTheory.yoneda.obj (Opposite.unop X)) F
    g : Quiver.Hom X Y
    ⊢ Eq (F.map g (CategoryTheory.yonedaEquiv f)) (f.app Y g.unop)
  -/
  rw [yonedaEquiv_naturality', yonedaEquiv_comp, yonedaEquiv_yoneda_map]
  /-
    🎉 no goals
  -/


lemma yonedaEquiv_symm_map {X Y : Cᵒᵖ} (f : X ⟶ Y) {F : Cᵒᵖ ⥤ Type v₁} (t : F.obj X) :
    yonedaEquiv.symm (F.map f t) = yoneda.map f.unop ≫ yonedaEquiv.symm t := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : Opposite C
    f : Quiver.Hom X Y
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    t : F.obj X
    ⊢ Eq (CategoryTheory.yonedaEquiv.symm (F.map f t)) (CategoryTheory.CategoryStr …
  -/
  obtain ⟨u, rfl⟩ := yonedaEquiv.surjective t
  /-
    case intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : Opposite C
    f : Quiver.Hom X Y
    F : CategoryTheory.Functor (Opposite C) (Type v₁)
    u : Quiver.Hom (CategoryTheory.yoneda.obj (Opposite.unop X)) F
    ⊢ Eq (CategoryTheory.yonedaEquiv.symm (F.map f (CategoryTheory.yonedaEquiv u)) …
  -/
  rw [yonedaEquiv_naturality', Equiv.symm_apply_apply, Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- Two morphisms of presheaves of types `P ⟶ Q` coincide if the precompositions
with morphisms `yoneda.obj X ⟶ P` agree. -/
lemma hom_ext_yoneda {P Q : Cᵒᵖ ⥤ Type v₁} {f g : P ⟶ Q}
    (h : ∀ (X : C) (p : yoneda.obj X ⟶ P), p ≫ f = p ≫ g) :
    f = g := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
    f g : Quiver.Hom P Q
    h : ∀ (X : C) (p : Quiver.Hom (CategoryTheory.yoneda.obj X) P), Eq (CategoryTh …
    ⊢ Eq f g
  -/
  ext X x
  simpa only [yonedaEquiv_comp, Equiv.apply_symm_apply]
    using congr_arg (yonedaEquiv) (h _ (yonedaEquiv.symm x))


/-- The "Yoneda evaluation" functor, which sends `X : Cᵒᵖ` and `F : Cᵒᵖ ⥤ Type`
to `F.obj X`, functorially in both `X` and `F`.
-/
def yonedaEvaluation : Cᵒᵖ × (Cᵒᵖ ⥤ Type v₁) ⥤ Type max u₁ v₁ :=
  evaluationUncurried Cᵒᵖ (Type v₁) ⋙ uliftFunctor


@[simp]
theorem yonedaEvaluation_map_down (P Q : Cᵒᵖ × (Cᵒᵖ ⥤ Type v₁)) (α : P ⟶ Q)
    (x : (yonedaEvaluation C).obj P) :
    ((yonedaEvaluation C).map α x).down = α.2.app Q.1 (P.2.map α.1 x.down) :=
  rfl


/-- The "Yoneda pairing" functor, which sends `X : Cᵒᵖ` and `F : Cᵒᵖ ⥤ Type`
to `yoneda.op.obj X ⟶ F`, functorially in both `X` and `F`.
-/
def yonedaPairing : Cᵒᵖ × (Cᵒᵖ ⥤ Type v₁) ⥤ Type max u₁ v₁ :=
  Functor.prod yoneda.op (𝟭 (Cᵒᵖ ⥤ Type v₁)) ⋙ Functor.hom (Cᵒᵖ ⥤ Type v₁)


@[ext]
lemma yonedaPairingExt {X : Cᵒᵖ × (Cᵒᵖ ⥤ Type v₁)} {x y : (yonedaPairing C).obj X}
    (w : ∀ Y, x.app Y = y.app Y) : x = y :=
  NatTrans.ext (funext w)


@[simp]
theorem yonedaPairing_map (P Q : Cᵒᵖ × (Cᵒᵖ ⥤ Type v₁)) (α : P ⟶ Q) (β : (yonedaPairing C).obj P) :
    (yonedaPairing C).map α β = yoneda.map α.1.unop ≫ β ≫ α.2 :=
  rfl


universe w in
variable {C} in
/-- A bijection `(yoneda.obj X ⋙ uliftFunctor ⟶ F) ≃ F.obj (op X)` which is a variant
of `yonedaEquiv` with heterogeneous universes. -/
def yonedaCompUliftFunctorEquiv (F : Cᵒᵖ ⥤ Type max v₁ w) (X : C) :
    (yoneda.obj X ⋙ uliftFunctor ⟶ F) ≃ F.obj (op X) where
  toFun φ := φ.app (op X) (ULift.up (𝟙 _))
  invFun f :=
    { app := fun _ x => F.map (ULift.down x).op f }
  left_inv φ := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (Opposite C) (Type (max v₁ w))
      X : C
      φ : Quiver.Hom ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor …
      ⊢ Eq ((fun f => { app := fun x x_1 => F.map (Quiver.Hom.op x_1.down) f, natura …
    -/
    ext Y f
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (Opposite C) (Type (max v₁ w))
      X : C
      φ : Quiver.Hom ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor …
      Y : Opposite C
      f : ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor.{w, v₁}).o …
      ⊢ Eq (((fun f => { app := fun x x_1 => F.map (Quiver.Hom.op x_1.down) f, natur …
    -/
    dsimp
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (Opposite C) (Type (max v₁ w))
      X : C
      φ : Quiver.Hom ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor …
      Y : Opposite C
      f : ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor.{w, v₁}).o …
      ⊢ Eq (F.map f.down.op (φ.app { unop := X } { down := CategoryTheory.CategorySt …
    -/
    rw [← FunctorToTypes.naturality]
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (Opposite C) (Type (max v₁ w))
      X : C
      φ : Quiver.Hom ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor …
      Y : Opposite C
      f : ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor.{w, v₁}).o …
      ⊢ Eq (φ.app Y (((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor …
    -/
    dsimp
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (Opposite C) (Type (max v₁ w))
      X : C
      φ : Quiver.Hom ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor …
      Y : Opposite C
      f : ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor.{w, v₁}).o …
      ⊢ Eq (φ.app Y { down := CategoryTheory.CategoryStruct.comp f.down (CategoryThe …
    -/
    rw [Category.comp_id]
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (Opposite C) (Type (max v₁ w))
      X : C
      φ : Quiver.Hom ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor …
      Y : Opposite C
      f : ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uliftFunctor.{w, v₁}).o …
      ⊢ Eq (φ.app Y { down := f.down }) (φ.app Y f)
    -/
    rfl
    /-
      🎉 no goals
    -/
                    /-
                      C : Type u₁
                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                      F : CategoryTheory.Functor (Opposite C) (Type (max v₁ w))
                      X : C
                      f : F.obj { unop := X }
                      ⊢ Eq ((fun φ => φ.app { unop := X } { down := CategoryTheory.CategoryStruct.id …
                    -/
  right_inv f := by aesop_cat
                    /-
                      🎉 no goals
                    -/


/-- The Yoneda lemma asserts that the Yoneda pairing
`(X : Cᵒᵖ, F : Cᵒᵖ ⥤ Type) ↦ (yoneda.obj (unop X) ⟶ F)`
is naturally isomorphic to the evaluation `(X, F) ↦ F.obj X`.

See <https://stacks.math.columbia.edu/tag/001P>.
-/
def yonedaLemma : yonedaPairing C ≅ yonedaEvaluation C :=
  NatIso.ofComponents
    (fun _ ↦ Equiv.toIso (yonedaEquiv.trans Equiv.ulift.symm))
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          ⊢ ∀ {X Y : Prod (Opposite C) (CategoryTheory.Functor (Opposite C) (Type v₁))}  …
        -/
    (by intro (X, F) (Y, G) f
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : Opposite C
          F : CategoryTheory.Functor (Opposite C) (Type v₁)
          Y : Opposite C
          G : CategoryTheory.Functor (Opposite C) (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yonedaPairing C).map …
        -/
        ext (a : yoneda.obj X.unop ⟶ F)
        /-
          case h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : Opposite C
          F : CategoryTheory.Functor (Opposite C) (Type v₁)
          Y : Opposite C
          G : CategoryTheory.Functor (Opposite C) (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          a : Quiver.Hom (CategoryTheory.yoneda.obj (Opposite.unop X)) F
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yonedaPairing C).map …
        -/
        apply ULift.ext
        /-
          case h.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : Opposite C
          F : CategoryTheory.Functor (Opposite C) (Type v₁)
          Y : Opposite C
          G : CategoryTheory.Functor (Opposite C) (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          a : Quiver.Hom (CategoryTheory.yoneda.obj (Opposite.unop X)) F
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yonedaPairing C).map …
        -/
        simp only [Functor.prod_obj, Functor.id_obj, types_comp_apply, yonedaEvaluation_map_down]
        /-
          case h.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : Opposite C
          F : CategoryTheory.Functor (Opposite C) (Type v₁)
          Y : Opposite C
          G : CategoryTheory.Functor (Opposite C) (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          a : Quiver.Hom (CategoryTheory.yoneda.obj (Opposite.unop X)) F
          ⊢ Eq ((CategoryTheory.yonedaEquiv.trans Equiv.ulift.symm).toIso.hom ((Category …
        -/
        erw [Equiv.ulift_symm_down, Equiv.ulift_symm_down]
        /-
          case h.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : Opposite C
          F : CategoryTheory.Functor (Opposite C) (Type v₁)
          Y : Opposite C
          G : CategoryTheory.Functor (Opposite C) (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          a : Quiver.Hom (CategoryTheory.yoneda.obj (Opposite.unop X)) F
          ⊢ Eq (CategoryTheory.yonedaEquiv ((CategoryTheory.yonedaPairing C).map f a)) ( …
        -/
        dsimp [yonedaEquiv]
        /-
          case h.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : Opposite C
          F : CategoryTheory.Functor (Opposite C) (Type v₁)
          Y : Opposite C
          G : CategoryTheory.Functor (Opposite C) (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          a : Quiver.Hom (CategoryTheory.yoneda.obj (Opposite.unop X)) F
          ⊢ Eq (f.2.app Y (a.app Y (CategoryTheory.CategoryStruct.comp (CategoryTheory.C …
        -/
        simp [← FunctorToTypes.naturality])
        /-
          🎉 no goals
        -/


/-- The curried version of yoneda lemma when `C` is small. -/
def curriedYonedaLemma {C : Type u₁} [SmallCategory C] :
    (yoneda.op ⋙ coyoneda : Cᵒᵖ ⥤ (Cᵒᵖ ⥤ Type u₁) ⥤ Type u₁) ≅ evaluation Cᵒᵖ (Type u₁) :=
                               /-
                                 C✝ : Type u₁
                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
                                 C : Type u₁
                                 inst✝ : CategoryTheory.SmallCategory C
                                 X : Opposite C
                                 ⊢ ∀ {X_1 Y : CategoryTheory.Functor (Opposite C) (Type u₁)} (f : Quiver.Hom X_ …
                               -/
  NatIso.ofComponents (fun X ↦ NatIso.ofComponents (fun _ ↦ Equiv.toIso yonedaEquiv)) (by
                               /-
                                 🎉 no goals
                               -/
    /-
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
    -/
    intro X Y f
    /-
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.op.comp Categ …
    -/
    ext a b
    /-
      case w.h.h
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      X Y : Opposite C
      f : Quiver.Hom X Y
      a : CategoryTheory.Functor (Opposite C) (Type u₁)
      b : ((CategoryTheory.yoneda.op.comp CategoryTheory.coyoneda).obj X).obj a
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.op.comp Cate …
    -/
    dsimp [yonedaEquiv]
    /-
      case w.h.h
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      X Y : Opposite C
      f : Quiver.Hom X Y
      a : CategoryTheory.Functor (Opposite C) (Type u₁)
      b : ((CategoryTheory.yoneda.op.comp CategoryTheory.coyoneda).obj X).obj a
      ⊢ Eq (b.app Y (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStru …
    -/
    simp [← FunctorToTypes.naturality])
    /-
      🎉 no goals
    -/


/-- The curried version of the Yoneda lemma. -/
def largeCurriedYonedaLemma {C : Type u₁} [Category.{v₁} C] :
    yoneda.op ⋙ coyoneda ≅
      evaluation Cᵒᵖ (Type v₁) ⋙ (whiskeringRight _ _ _).obj uliftFunctor.{u₁} :=
  NatIso.ofComponents
    (fun X => NatIso.ofComponents
      (fun _ => Equiv.toIso <| yonedaEquiv.trans Equiv.ulift.symm)
      (by
        /-
          C✝ : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : Opposite C
          ⊢ ∀ {X_1 Y : CategoryTheory.Functor (Opposite C) (Type v₁)} (f : Quiver.Hom X_ …
        -/
        intros Y Z f
        /-
          C✝ : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : Opposite C
          Y Z : CategoryTheory.Functor (Opposite C) (Type v₁)
          f : Quiver.Hom Y Z
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.yoneda.op.comp Cate …
        -/
        ext g
        /-
          case h
          C✝ : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : Opposite C
          Y Z : CategoryTheory.Functor (Opposite C) (Type v₁)
          f : Quiver.Hom Y Z
          g : ((CategoryTheory.yoneda.op.comp CategoryTheory.coyoneda).obj X).obj Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.yoneda.op.comp Cate …
        -/
        rw [← ULift.down_inj]
        /-
          case h
          C✝ : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : Opposite C
          Y Z : CategoryTheory.Functor (Opposite C) (Type v₁)
          f : Quiver.Hom Y Z
          g : ((CategoryTheory.yoneda.op.comp CategoryTheory.coyoneda).obj X).obj Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.yoneda.op.comp Cate …
        -/
        simpa using yonedaEquiv_comp _ _))
        /-
          🎉 no goals
        -/
    (by
      /-
        C✝ : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
      -/
      intros Y Z f
      /-
        C✝ : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        Y Z : Opposite C
        f : Quiver.Hom Y Z
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.op.comp Categ …
      -/
      ext F g
      /-
        case w.h.h
        C✝ : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        Y Z : Opposite C
        f : Quiver.Hom Y Z
        F : CategoryTheory.Functor (Opposite C) (Type v₁)
        g : ((CategoryTheory.yoneda.op.comp CategoryTheory.coyoneda).obj Y).obj F
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.op.comp Cate …
      -/
      rw [← ULift.down_inj]
      /-
        case w.h.h
        C✝ : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        Y Z : Opposite C
        f : Quiver.Hom Y Z
        F : CategoryTheory.Functor (Opposite C) (Type v₁)
        g : ((CategoryTheory.yoneda.op.comp CategoryTheory.coyoneda).obj Y).obj F
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.op.comp Cate …
      -/
      simpa using (yonedaEquiv_naturality _ _).symm)
      /-
        🎉 no goals
      -/


/-- Version of the Yoneda lemma where the presheaf is fixed but the argument varies. -/
def yonedaOpCompYonedaObj {C : Type u₁} [Category.{v₁} C] (P : Cᵒᵖ ⥤ Type v₁) :
    yoneda.op ⋙ yoneda.obj P ≅ P ⋙ uliftFunctor.{u₁} :=
  isoWhiskerRight largeCurriedYonedaLemma ((evaluation _ _).obj P)


/-- The curried version of yoneda lemma when `C` is small. -/
def curriedYonedaLemma' {C : Type u₁} [SmallCategory C] :
    yoneda ⋙ (whiskeringLeft Cᵒᵖ (Cᵒᵖ ⥤ Type u₁)ᵒᵖ (Type u₁)).obj yoneda.op
      ≅ 𝟭 (Cᵒᵖ ⥤ Type u₁) :=
  /-
    C✝ : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
    C : Type u₁
    inst✝ : CategoryTheory.SmallCategory C
    ⊢ ∀ {X Y : CategoryTheory.Functor (Opposite C) (Type u₁)} (f : Quiver.Hom X Y) …
  -/
    /-
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      F : CategoryTheory.Functor (Opposite C) (Type u₁)
      ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
    -/
  NatIso.ofComponents (fun F ↦ NatIso.ofComponents (fun _ ↦ Equiv.toIso yonedaEquiv) (by
    /-
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      F : CategoryTheory.Functor (Opposite C) (Type u₁)
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.yoneda.comp ((Categ …
    -/
  /-
    🎉 no goals
  -/
    /-
      case h
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      F : CategoryTheory.Functor (Opposite C) (Type u₁)
      X Y : Opposite C
      f : Quiver.Hom X Y
      a : ((CategoryTheory.yoneda.comp ((CategoryTheory.whiskeringLeft (Opposite C)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.yoneda.comp ((Categ …
    -/
    intro X Y f
    /-
      case h
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      F : CategoryTheory.Functor (Opposite C) (Type u₁)
      X Y : Opposite C
      f : Quiver.Hom X Y
      a : ((CategoryTheory.yoneda.comp ((CategoryTheory.whiskeringLeft (Opposite C)  …
      ⊢ Eq (a.app Y (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStru …
    -/
    ext a
    /-
      🎉 no goals
    -/
    dsimp [yonedaEquiv]
    simp [← FunctorToTypes.naturality]))


lemma isIso_of_yoneda_map_bijective {X Y : C} (f : X ⟶ Y)
    (hf : ∀ (T : C), Function.Bijective (fun (x : T ⟶ X) => x ≫ f)) :
    IsIso f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    hf : ∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.comp …
    ⊢ CategoryTheory.IsIso f
  -/
  obtain ⟨g, hg : g ≫ f = 𝟙 Y⟩ := (hf Y).2 (𝟙 Y)
  /-
    case intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    hf : ∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.comp …
    g : Quiver.Hom Y X
    hg : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruc …
    ⊢ CategoryTheory.IsIso f
  -/
  exact ⟨g, (hf _).1 (by aesop_cat), hg⟩
  /-
    🎉 no goals
  -/


lemma isIso_iff_yoneda_map_bijective {X Y : C} (f : X ⟶ Y) :
    IsIso f ↔ (∀ (T : C), Function.Bijective (fun (x : T ⟶ X) => x ≫ f)) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso f) (∀ (T : C), Function.Bijective fun x => Categor …
  -/
  refine ⟨fun _ ↦ ?_, fun hf ↦ isIso_of_yoneda_map_bijective f hf⟩
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    x✝ : CategoryTheory.IsIso f
    ⊢ ∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.comp x f
  -/
  have : IsIso (yoneda.map f) := inferInstance
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    x✝ : CategoryTheory.IsIso f
    this : CategoryTheory.IsIso (CategoryTheory.yoneda.map f)
    ⊢ ∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.comp x f
  -/
  intro T
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    x✝ : CategoryTheory.IsIso f
    this : CategoryTheory.IsIso (CategoryTheory.yoneda.map f)
    T : C
    ⊢ Function.Bijective fun x => CategoryTheory.CategoryStruct.comp x f
  -/
  rw [← isIso_iff_bijective]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    x✝ : CategoryTheory.IsIso f
    this : CategoryTheory.IsIso (CategoryTheory.yoneda.map f)
    T : C
    ⊢ CategoryTheory.IsIso fun x => CategoryTheory.CategoryStruct.comp x f
  -/
  exact inferInstanceAs (IsIso ((yoneda.map f).app _))
  /-
    🎉 no goals
  -/


lemma isIso_iff_isIso_yoneda_map {X Y : C} (f : X ⟶ Y) :
    IsIso f ↔ ∀ c : C, IsIso ((yoneda.map f).app ⟨c⟩) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso f) (∀ (c : C), CategoryTheory.IsIso ((CategoryTheo …
  -/
  rw [isIso_iff_yoneda_map_bijective]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.co …
  -/
  exact forall_congr' fun _ ↦ (isIso_iff_bijective _).symm
  /-
    🎉 no goals
  -/


/-- We have a type-level equivalence between natural transformations from the coyoneda embedding
and elements of `F.obj X.unop`, without any universe switching.
-/
def coyonedaEquiv {X : C} {F : C ⥤ Type v₁} : (coyoneda.obj (op X) ⟶ F) ≃ F.obj X where
  toFun η := η.app X (𝟙 X)
  invFun ξ := { app := fun _ x ↦ F.map x ξ }
  left_inv := fun η ↦ by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F : CategoryTheory.Functor C (Type v₁)
      η : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
      ⊢ Eq ((fun ξ => { app := fun x x_1 => F.map x_1 ξ, naturality := ⋯ }) ((fun η  …
    -/
    ext Y (x : X ⟶ Y)
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F : CategoryTheory.Functor C (Type v₁)
      η : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
      Y : C
      x : Quiver.Hom X Y
      ⊢ Eq (((fun ξ => { app := fun x x_1 => F.map x_1 ξ, naturality := ⋯ }) ((fun η …
    -/
    dsimp
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F : CategoryTheory.Functor C (Type v₁)
      η : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
      Y : C
      x : Quiver.Hom X Y
      ⊢ Eq (F.map x (η.app X (CategoryTheory.CategoryStruct.id X))) (η.app Y x)
    -/
    rw [← FunctorToTypes.naturality]
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F : CategoryTheory.Functor C (Type v₁)
      η : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
      Y : C
      x : Quiver.Hom X Y
      ⊢ Eq (η.app Y ((CategoryTheory.coyoneda.obj { unop := X }).map x (CategoryTheo …
    -/
    simp
    /-
      🎉 no goals
    -/
                  /-
                    C : Type u₁
                    inst✝ : CategoryTheory.Category.{v₁, u₁} C
                    X : C
                    F : CategoryTheory.Functor C (Type v₁)
                    ⊢ Function.RightInverse (fun ξ => { app := fun x x_1 => F.map x_1 ξ, naturalit …
                  -/
  right_inv := by intro ξ; simp
                           /-
                             🎉 no goals
                           -/


theorem coyonedaEquiv_apply {X : C} {F : C ⥤ Type v₁} (f : coyoneda.obj (op X) ⟶ F) :
    coyonedaEquiv f = f.app X (𝟙 X) :=
  rfl


@[simp]
theorem coyonedaEquiv_symm_app_apply {X : C} {F : C ⥤ Type v₁} (x : F.obj X) (Y : C)
    (f : X ⟶ Y) : (coyonedaEquiv.symm x).app Y f = F.map f x :=
  rfl


lemma coyonedaEquiv_naturality {X Y : C} {F : C ⥤ Type v₁} (f : coyoneda.obj (op X) ⟶ F)
    (g : X ⟶ Y) : F.map g (coyonedaEquiv f) = coyonedaEquiv (coyoneda.map g.op ≫ f) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    F : CategoryTheory.Functor C (Type v₁)
    f : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
    g : Quiver.Hom X Y
    ⊢ Eq (F.map g (CategoryTheory.coyonedaEquiv f)) (CategoryTheory.coyonedaEquiv  …
  -/
  change (f.app X ≫ F.map g) (𝟙 X) = f.app Y (g ≫ 𝟙 Y)
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    F : CategoryTheory.Functor C (Type v₁)
    f : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
    g : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app X) (F.map g) (CategoryTheory.C …
  -/
  rw [← f.naturality]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    F : CategoryTheory.Functor C (Type v₁)
    f : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
    g : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.obj { unop  …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    F : CategoryTheory.Functor C (Type v₁)
    f : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
    g : Quiver.Hom X Y
    ⊢ Eq (f.app Y (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStru …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma coyonedaEquiv_comp {X : C} {F G : C ⥤ Type v₁} (α : coyoneda.obj (op X) ⟶ F) (β : F ⟶ G) :
    coyonedaEquiv (α ≫ β) = β.app _ (coyonedaEquiv α) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    F G : CategoryTheory.Functor C (Type v₁)
    α : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
    β : Quiver.Hom F G
    ⊢ Eq (CategoryTheory.coyonedaEquiv (CategoryTheory.CategoryStruct.comp α β)) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma coyonedaEquiv_coyoneda_map {X Y : C} (f : X ⟶ Y) :
    coyonedaEquiv (coyoneda.map f.op) = f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.coyonedaEquiv (CategoryTheory.coyoneda.map f.op)) f
  -/
  rw [coyonedaEquiv_apply]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.coyoneda.map f.op).app Y (CategoryTheory.CategoryStruct. …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma map_coyonedaEquiv {X Y : C} {F : C ⥤ Type v₁} (f : coyoneda.obj (op X) ⟶ F)
    (g : X ⟶ Y) : F.map g (coyonedaEquiv f) = f.app Y g := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    F : CategoryTheory.Functor C (Type v₁)
    f : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
    g : Quiver.Hom X Y
    ⊢ Eq (F.map g (CategoryTheory.coyonedaEquiv f)) (f.app Y g)
  -/
  rw [coyonedaEquiv_naturality, coyonedaEquiv_comp, coyonedaEquiv_coyoneda_map]
  /-
    🎉 no goals
  -/


lemma coyonedaEquiv_symm_map {X Y : C} (f : X ⟶ Y) {F : C ⥤ Type v₁} (t : F.obj X) :
    coyonedaEquiv.symm (F.map f t) = coyoneda.map f.op ≫ coyonedaEquiv.symm t := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    F : CategoryTheory.Functor C (Type v₁)
    t : F.obj X
    ⊢ Eq (CategoryTheory.coyonedaEquiv.symm (F.map f t)) (CategoryTheory.CategoryS …
  -/
  obtain ⟨u, rfl⟩ := coyonedaEquiv.surjective t
  /-
    case intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    F : CategoryTheory.Functor C (Type v₁)
    u : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
    ⊢ Eq (CategoryTheory.coyonedaEquiv.symm (F.map f (CategoryTheory.coyonedaEquiv …
  -/
  simp [coyonedaEquiv_naturality u f]
  /-
    🎉 no goals
  -/


/-- The "Coyoneda evaluation" functor, which sends `X : C` and `F : C ⥤ Type`
to `F.obj X`, functorially in both `X` and `F`.
-/
def coyonedaEvaluation : C × (C ⥤ Type v₁) ⥤ Type max u₁ v₁ :=
  evaluationUncurried C (Type v₁) ⋙ uliftFunctor


@[simp]
theorem coyonedaEvaluation_map_down (P Q : C × (C ⥤ Type v₁)) (α : P ⟶ Q)
    (x : (coyonedaEvaluation C).obj P) :
    ((coyonedaEvaluation C).map α x).down = α.2.app Q.1 (P.2.map α.1 x.down) :=
  rfl


/-- The "Coyoneda pairing" functor, which sends `X : C` and `F : C ⥤ Type`
to `coyoneda.rightOp.obj X ⟶ F`, functorially in both `X` and `F`.
-/
def coyonedaPairing : C × (C ⥤ Type v₁) ⥤ Type max u₁ v₁ :=
  Functor.prod coyoneda.rightOp (𝟭 (C ⥤ Type v₁)) ⋙ Functor.hom (C ⥤ Type v₁)


@[ext]
lemma coyonedaPairingExt {X : C × (C ⥤ Type v₁)} {x y : (coyonedaPairing C).obj X}
    (w : ∀ Y, x.app Y = y.app Y) : x = y :=
  NatTrans.ext (funext w)


@[simp]
theorem coyonedaPairing_map (P Q : C × (C ⥤ Type v₁)) (α : P ⟶ Q) (β : (coyonedaPairing C).obj P) :
    (coyonedaPairing C).map α β = coyoneda.map α.1.op ≫ β ≫ α.2 :=
  rfl


universe w in
variable {C} in
/-- A bijection `(coyoneda.obj X ⋙ uliftFunctor ⟶ F) ≃ F.obj (unop X)` which is a variant
of `coyonedaEquiv` with heterogeneous universes. -/
def coyonedaCompUliftFunctorEquiv (F : C ⥤ Type max v₁ w) (X : Cᵒᵖ) :
    (coyoneda.obj X ⋙ uliftFunctor ⟶ F) ≃ F.obj X.unop where
  toFun φ := φ.app X.unop (ULift.up (𝟙 _))
  invFun f :=
    { app := fun _ x => F.map (ULift.down x) f }
  left_inv φ := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C (Type (max v₁ w))
      X : Opposite C
      φ : Quiver.Hom ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunct …
      ⊢ Eq ((fun f => { app := fun x x_1 => F.map x_1.down f, naturality := ⋯ }) ((f …
    -/
    ext Y f
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C (Type (max v₁ w))
      X : Opposite C
      φ : Quiver.Hom ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunct …
      Y : C
      f : ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunctor.{w, v₁}) …
      ⊢ Eq (((fun f => { app := fun x x_1 => F.map x_1.down f, naturality := ⋯ }) (( …
    -/
    dsimp
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C (Type (max v₁ w))
      X : Opposite C
      φ : Quiver.Hom ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunct …
      Y : C
      f : ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunctor.{w, v₁}) …
      ⊢ Eq (F.map f.down (φ.app (Opposite.unop X) { down := CategoryTheory.CategoryS …
    -/
    rw [← FunctorToTypes.naturality]
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C (Type (max v₁ w))
      X : Opposite C
      φ : Quiver.Hom ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunct …
      Y : C
      f : ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunctor.{w, v₁}) …
      ⊢ Eq (φ.app Y (((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunct …
    -/
    dsimp
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C (Type (max v₁ w))
      X : Opposite C
      φ : Quiver.Hom ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunct …
      Y : C
      f : ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunctor.{w, v₁}) …
      ⊢ Eq (φ.app Y { down := CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
    -/
    rw [Category.id_comp]
    /-
      case w.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C (Type (max v₁ w))
      X : Opposite C
      φ : Quiver.Hom ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunct …
      Y : C
      f : ((CategoryTheory.coyoneda.obj X).comp CategoryTheory.uliftFunctor.{w, v₁}) …
      ⊢ Eq (φ.app Y { down := f.down }) (φ.app Y f)
    -/
    rfl
    /-
      🎉 no goals
    -/
                    /-
                      C : Type u₁
                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                      F : CategoryTheory.Functor C (Type (max v₁ w))
                      X : Opposite C
                      f : F.obj (Opposite.unop X)
                      ⊢ Eq ((fun φ => φ.app (Opposite.unop X) { down := CategoryTheory.CategoryStruc …
                    -/
  right_inv f := by aesop_cat
                    /-
                      🎉 no goals
                    -/


/-- The Coyoneda lemma asserts that the Coyoneda pairing
`(X : C, F : C ⥤ Type) ↦ (coyoneda.obj X ⟶ F)`
is naturally isomorphic to the evaluation `(X, F) ↦ F.obj X`.

See <https://stacks.math.columbia.edu/tag/001P>.
-/
def coyonedaLemma : coyonedaPairing C ≅ coyonedaEvaluation C :=
  NatIso.ofComponents
    (fun _ ↦ Equiv.toIso (coyonedaEquiv.trans Equiv.ulift.symm))
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          ⊢ ∀ {X Y : Prod C (CategoryTheory.Functor C (Type v₁))} (f : Quiver.Hom X Y),  …
        -/
    (by intro (X, F) (Y, G) f
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : C
          F : CategoryTheory.Functor C (Type v₁)
          Y : C
          G : CategoryTheory.Functor C (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyonedaPairing C).m …
        -/
        ext (a : coyoneda.obj (op X) ⟶ F)
        /-
          case h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : C
          F : CategoryTheory.Functor C (Type v₁)
          Y : C
          G : CategoryTheory.Functor C (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          a : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyonedaPairing C).m …
        -/
        apply ULift.ext
        /-
          case h.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : C
          F : CategoryTheory.Functor C (Type v₁)
          Y : C
          G : CategoryTheory.Functor C (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          a : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyonedaPairing C).m …
        -/
        simp only [Functor.prod_obj, Functor.id_obj, types_comp_apply, coyonedaEvaluation_map_down]
        /-
          case h.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : C
          F : CategoryTheory.Functor C (Type v₁)
          Y : C
          G : CategoryTheory.Functor C (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          a : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
          ⊢ Eq ((CategoryTheory.coyonedaEquiv.trans Equiv.ulift.symm).toIso.hom ((Catego …
        -/
        erw [Equiv.ulift_symm_down, Equiv.ulift_symm_down]
        /-
          case h.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : C
          F : CategoryTheory.Functor C (Type v₁)
          Y : C
          G : CategoryTheory.Functor C (Type v₁)
          f : Quiver.Hom { fst := X, snd := F } { fst := Y, snd := G }
          a : Quiver.Hom (CategoryTheory.coyoneda.obj { unop := X }) F
          ⊢ Eq (CategoryTheory.coyonedaEquiv ((CategoryTheory.coyonedaPairing C).map f a …
        -/
        simp [coyonedaEquiv, ← FunctorToTypes.naturality])
        /-
          🎉 no goals
        -/


/-- The curried version of coyoneda lemma when `C` is small. -/
def curriedCoyonedaLemma {C : Type u₁} [SmallCategory C] :
    coyoneda.rightOp ⋙ coyoneda ≅ evaluation C (Type u₁) :=
                               /-
                                 C✝ : Type u₁
                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
                                 C : Type u₁
                                 inst✝ : CategoryTheory.SmallCategory C
                                 X : C
                                 ⊢ ∀ {X_1 Y : CategoryTheory.Functor C (Type u₁)} (f : Quiver.Hom X_1 Y), Eq (C …
                               -/
  NatIso.ofComponents (fun X ↦ NatIso.ofComponents (fun _ ↦ Equiv.toIso coyonedaEquiv)) (by
                               /-
                                 🎉 no goals
                               -/
    /-
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
    -/
    intro X Y f
    /-
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.rightOp.com …
    -/
    ext a b
    /-
      case w.h.h
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      X Y : C
      f : Quiver.Hom X Y
      a : CategoryTheory.Functor C (Type u₁)
      b : ((CategoryTheory.coyoneda.rightOp.comp CategoryTheory.coyoneda).obj X).obj a
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.rightOp.co …
    -/
    simp [coyonedaEquiv, ← FunctorToTypes.naturality])
    /-
      🎉 no goals
    -/


/-- The curried version of the Coyoneda lemma. -/
def largeCurriedCoyonedaLemma {C : Type u₁} [Category.{v₁} C] :
    (coyoneda.rightOp ⋙ coyoneda) ≅
      evaluation C (Type v₁) ⋙ (whiskeringRight _ _ _).obj uliftFunctor.{u₁} :=
  NatIso.ofComponents
    (fun X => NatIso.ofComponents
      (fun _ => Equiv.toIso <| coyonedaEquiv.trans Equiv.ulift.symm)
      (by
        /-
          C✝ : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : C
          ⊢ ∀ {X_1 Y : CategoryTheory.Functor C (Type v₁)} (f : Quiver.Hom X_1 Y), Eq (C …
        -/
        intros Y Z f
        /-
          C✝ : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : C
          Y Z : CategoryTheory.Functor C (Type v₁)
          f : Quiver.Hom Y Z
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.coyoneda.rightOp.co …
        -/
        ext g
        /-
          case h
          C✝ : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : C
          Y Z : CategoryTheory.Functor C (Type v₁)
          f : Quiver.Hom Y Z
          g : ((CategoryTheory.coyoneda.rightOp.comp CategoryTheory.coyoneda).obj X).obj Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.coyoneda.rightOp.co …
        -/
        rw [← ULift.down_inj]
        /-
          case h
          C✝ : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X : C
          Y Z : CategoryTheory.Functor C (Type v₁)
          f : Quiver.Hom Y Z
          g : ((CategoryTheory.coyoneda.rightOp.comp CategoryTheory.coyoneda).obj X).obj Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.coyoneda.rightOp.co …
        -/
        simpa using coyonedaEquiv_comp _ _))
        /-
          🎉 no goals
        -/
    (by
      /-
        C✝ : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
      -/
      intro Y Z f
      /-
        C✝ : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        Y Z : C
        f : Quiver.Hom Y Z
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.rightOp.com …
      -/
      ext F g
      /-
        case w.h.h
        C✝ : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        Y Z : C
        f : Quiver.Hom Y Z
        F : CategoryTheory.Functor C (Type v₁)
        g : ((CategoryTheory.coyoneda.rightOp.comp CategoryTheory.coyoneda).obj Y).obj F
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.rightOp.co …
      -/
      rw [← ULift.down_inj]
      /-
        case w.h.h
        C✝ : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        Y Z : C
        f : Quiver.Hom Y Z
        F : CategoryTheory.Functor C (Type v₁)
        g : ((CategoryTheory.coyoneda.rightOp.comp CategoryTheory.coyoneda).obj Y).obj F
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.rightOp.co …
      -/
      simpa using (coyonedaEquiv_naturality _ _).symm)
      /-
        🎉 no goals
      -/


/-- Version of the Coyoneda lemma where the presheaf is fixed but the argument varies. -/
def coyonedaCompYonedaObj {C : Type u₁} [Category.{v₁} C] (P : C ⥤ Type v₁) :
    coyoneda.rightOp ⋙ yoneda.obj P ≅ P ⋙ uliftFunctor.{u₁} :=
  isoWhiskerRight largeCurriedCoyonedaLemma ((evaluation _ _).obj P)


/-- The curried version of coyoneda lemma when `C` is small. -/
def curriedCoyonedaLemma' {C : Type u₁} [SmallCategory C] :
    yoneda ⋙ (whiskeringLeft C (C ⥤ Type u₁)ᵒᵖ (Type u₁)).obj coyoneda.rightOp
      ≅ 𝟭 (C ⥤ Type u₁) :=
  /-
    C✝ : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
    C : Type u₁
    inst✝ : CategoryTheory.SmallCategory C
    ⊢ ∀ {X Y : CategoryTheory.Functor C (Type u₁)} (f : Quiver.Hom X Y), Eq (Categ …
  -/
    /-
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      F : CategoryTheory.Functor C (Type u₁)
      ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
    -/
  NatIso.ofComponents (fun F ↦ NatIso.ofComponents (fun _ ↦ Equiv.toIso coyonedaEquiv) (by
    /-
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      F : CategoryTheory.Functor C (Type u₁)
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.yoneda.comp ((Categ …
    -/
  /-
    🎉 no goals
  -/
    /-
      case h
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C✝
      C : Type u₁
      inst✝ : CategoryTheory.SmallCategory C
      F : CategoryTheory.Functor C (Type u₁)
      X Y : C
      f : Quiver.Hom X Y
      a : ((CategoryTheory.yoneda.comp ((CategoryTheory.whiskeringLeft C (Opposite ( …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.yoneda.comp ((Categ …
    -/
    intro X Y f
    /-
      🎉 no goals
    -/
    ext a
    simp [coyonedaEquiv, ← FunctorToTypes.naturality]))


lemma isIso_of_coyoneda_map_bijective {X Y : C} (f : X ⟶ Y)
    (hf : ∀ (T : C), Function.Bijective (fun (x : Y ⟶ T) => f ≫ x)) :
    IsIso f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    hf : ∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.comp …
    ⊢ CategoryTheory.IsIso f
  -/
  obtain ⟨g, hg : f ≫ g = 𝟙 X⟩ := (hf X).2 (𝟙 X)
  /-
    case intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    hf : ∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.comp …
    g : Quiver.Hom Y X
    hg : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruc …
    ⊢ CategoryTheory.IsIso f
  -/
  refine ⟨g, hg, (hf _).1 ?_⟩
  /-
    case intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    hf : ∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.comp …
    g : Quiver.Hom Y X
    hg : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruc …
    ⊢ Eq ((fun x => CategoryTheory.CategoryStruct.comp f x) (CategoryTheory.Catego …
  -/
  simp only [Category.comp_id, ← Category.assoc, hg, Category.id_comp]
  /-
    🎉 no goals
  -/


lemma isIso_iff_coyoneda_map_bijective {X Y : C} (f : X ⟶ Y) :
    IsIso f ↔ (∀ (T : C), Function.Bijective (fun (x : Y ⟶ T) => f ≫ x)) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso f) (∀ (T : C), Function.Bijective fun x => Categor …
  -/
  refine ⟨fun _ ↦ ?_, fun hf ↦ isIso_of_coyoneda_map_bijective f hf⟩
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    x✝ : CategoryTheory.IsIso f
    ⊢ ∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.comp f x
  -/
  have : IsIso (coyoneda.map f.op) := inferInstance
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    x✝ : CategoryTheory.IsIso f
    this : CategoryTheory.IsIso (CategoryTheory.coyoneda.map f.op)
    ⊢ ∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.comp f x
  -/
  intro T
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    x✝ : CategoryTheory.IsIso f
    this : CategoryTheory.IsIso (CategoryTheory.coyoneda.map f.op)
    T : C
    ⊢ Function.Bijective fun x => CategoryTheory.CategoryStruct.comp f x
  -/
  rw [← isIso_iff_bijective]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    x✝ : CategoryTheory.IsIso f
    this : CategoryTheory.IsIso (CategoryTheory.coyoneda.map f.op)
    T : C
    ⊢ CategoryTheory.IsIso fun x => CategoryTheory.CategoryStruct.comp f x
  -/
  exact inferInstanceAs (IsIso ((coyoneda.map f.op).app _))
  /-
    🎉 no goals
  -/


lemma isIso_iff_isIso_coyoneda_map {X Y : C} (f : X ⟶ Y) :
    IsIso f ↔ ∀ c : C, IsIso ((coyoneda.map f.op).app c) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso f) (∀ (c : C), CategoryTheory.IsIso ((CategoryTheo …
  -/
  rw [isIso_iff_coyoneda_map_bijective]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (∀ (T : C), Function.Bijective fun x => CategoryTheory.CategoryStruct.co …
  -/
  exact forall_congr' fun _ ↦ (isIso_iff_bijective _).symm
  /-
    🎉 no goals
  -/


/-- The natural transformation `yoneda.obj X ⟶ F.op ⋙ yoneda.obj (F.obj X)`
when `F : C ⥤ D` and `X : C`. -/
def yonedaMap (X : C) : yoneda.obj X ⟶ F.op ⋙ yoneda.obj (F.obj X) where
  app _ f := F.map f


@[simp]
lemma yonedaMap_app_apply {Y : C} {X : Cᵒᵖ} (f : X.unop ⟶ Y) :
    (yonedaMap F Y).app X f = F.map f := rfl


/-- `FullyFaithful.homEquiv` as a natural isomorphism. -/
@[simps!]
def homNatIso {D : Type u₂} [Category.{v₂} D] {F : C ⥤ D} (hF : F.FullyFaithful) (X : C) :
    F.op ⋙ yoneda.obj (F.obj X) ⋙ uliftFunctor.{v₁} ≅ yoneda.obj X ⋙ uliftFunctor.{v₂} :=
  NatIso.ofComponents
    (fun Y => Equiv.toIso (Equiv.ulift.trans <| hF.homEquiv.symm.trans Equiv.ulift.symm))
                 /-
                   C✝ : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C✝
                   C : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝ : CategoryTheory.Category.{v₂, u₂} D
                   F : CategoryTheory.Functor C D
                   hF : F.FullyFaithful
                   X : C
                   X✝ Y✝ : Opposite C
                   f : Quiver.Hom X✝ Y✝
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.op.comp ((CategoryTheory.yoneda.o …
                 -/
    (fun f => by ext; exact Equiv.ulift.injective (hF.map_injective (by simp)))
                      /-
                        🎉 no goals
                      -/


/-- `FullyFaithful.homEquiv` as a natural isomorphism. -/
@[simps!]
def homNatIsoMaxRight {D : Type u₂} [Category.{max v₁ v₂} D] {F : C ⥤ D} (hF : F.FullyFaithful)
    (X : C) : F.op ⋙ yoneda.obj (F.obj X) ≅ yoneda.obj X ⋙ uliftFunctor.{v₂} :=
  NatIso.ofComponents
    (fun Y => Equiv.toIso (hF.homEquiv.symm.trans Equiv.ulift.symm))
                 /-
                   C✝ : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C✝
                   C : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝ : CategoryTheory.Category.{max v₁ v₂, u₂} D
                   F : CategoryTheory.Functor C D
                   hF : F.FullyFaithful
                   X : C
                   X✝ Y✝ : Opposite C
                   f : Quiver.Hom X✝ Y✝
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.op.comp (CategoryTheory.yoneda.ob …
                 -/
    (fun f => by ext; exact Equiv.ulift.injective (hF.map_injective (by simp)))
                      /-
                        🎉 no goals
                      -/


/-- `FullyFaithful.homEquiv` as a natural isomorphism. -/
@[simps!]
def compYonedaCompWhiskeringLeft {D : Type u₂} [Category.{v₂} D] {F : C ⥤ D}
    (hF : F.FullyFaithful) : F ⋙ yoneda ⋙ (whiskeringLeft _ _ _).obj F.op ⋙
      (CategoryTheory.whiskeringRight _ _ _).obj uliftFunctor.{v₁} ≅
      yoneda ⋙ (CategoryTheory.whiskeringRight _ _ _).obj uliftFunctor.{v₂} :=
  NatIso.ofComponents (fun X => hF.homNatIso _)
                 /-
                   C✝ : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C✝
                   C : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝ : CategoryTheory.Category.{v₂, u₂} D
                   F : CategoryTheory.Functor C D
                   hF : F.FullyFaithful
                   X✝ Y✝ : C
                   f : Quiver.Hom X✝ Y✝
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp (CategoryTheory.yoneda.comp  …
                 -/
    (fun f => by ext; exact Equiv.ulift.injective (hF.map_injective (by simp)))
                      /-
                        🎉 no goals
                      -/


/-- `FullyFaithful.homEquiv` as a natural isomorphism. -/
@[simps!]
def compYonedaCompWhiskeringLeftMaxRight {D : Type u₂} [Category.{max v₁ v₂} D] {F : C ⥤ D}
    (hF : F.FullyFaithful) : F ⋙ yoneda ⋙ (whiskeringLeft _ _ _).obj F.op ≅
      yoneda ⋙ (CategoryTheory.whiskeringRight _ _ _).obj uliftFunctor.{v₂} :=
  NatIso.ofComponents (fun X => hF.homNatIsoMaxRight _)
                 /-
                   C✝ : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C✝
                   C : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝ : CategoryTheory.Category.{max v₁ v₂, u₂} D
                   F : CategoryTheory.Functor C D
                   hF : F.FullyFaithful
                   X✝ Y✝ : C
                   f : Quiver.Hom X✝ Y✝
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp (CategoryTheory.yoneda.comp  …
                 -/
    (fun f => by ext; exact Equiv.ulift.injective (hF.map_injective (by simp)))
                      /-
                        🎉 no goals
                      -/


