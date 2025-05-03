/-- If `F : J ⥤ C` then `F.cones` is the functor assigning to an object `X : C` the
type of natural transformations from the constant functor with value `X` to `F`.
An object representing this functor is a limit of `F`.
-/
@[simps!]
def cones : Cᵒᵖ ⥤ Type max u₁ v₃ :=
  (const J).op ⋙ yoneda.obj F


/-- If `F : J ⥤ C` then `F.cocones` is the functor assigning to an object `(X : C)`
the type of natural transformations from `F` to the constant functor with value `X`.
An object corepresenting this functor is a colimit of `F`.
-/
@[simps!]
def cocones : C ⥤ Type max u₁ v₃ :=
  const J ⋙ coyoneda.obj (op F)


/-- Functorially associated to each functor `J ⥤ C`, we have the `C`-presheaf consisting of
cones with a given cone point.
-/
@[simps!]
def cones : (J ⥤ C) ⥤ Cᵒᵖ ⥤ Type max u₁ v₃ where
  obj := Functor.cones
  map f := whiskerLeft (const J).op (yoneda.map f)


/-- Contravariantly associated to each functor `J ⥤ C`, we have the `C`-copresheaf consisting of
cocones with a given cocone point.
-/
@[simps!]
def cocones : (J ⥤ C)ᵒᵖ ⥤ C ⥤ Type max u₁ v₃ where
  obj F := Functor.cocones (unop F)
  map f := whiskerLeft (const J) (coyoneda.map f)


/-- A `c : Cone F` is:
* an object `c.pt` and
* a natural transformation `c.π : c.pt ⟶ F` from the constant `c.pt` functor to `F`.

Example: if `J` is a category coming from a poset then the data required to make
a term of type `Cone F` is morphisms `πⱼ : c.pt ⟶ F j` for all `j : J` and,
for all `i ≤ j` in `J`, morphisms `πᵢⱼ : F i ⟶ F j` such that `πᵢ ≫ πᵢⱼ = πᵢ`.

`Cone F` is equivalent, via `cone.equiv` below, to `Σ X, F.cones.obj X`.
-/
structure Cone (F : J ⥤ C) where
  /-- An object of `C` -/
  pt : C
  /-- A natural transformation from the constant functor at `X` to `F` -/
  π : (const J).obj pt ⟶ F


instance inhabitedCone (F : Discrete PUnit ⥤ C) : Inhabited (Cone F) :=
  ⟨{  pt := F.obj ⟨⟨⟩⟩
      π := { app := fun ⟨⟨⟩⟩ => 𝟙 _
             naturality := by
              /-
                J : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                K : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} K
                C : Type u₃
                inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                D : Type u₄
                inst✝ : CategoryTheory.Category.{v₄, u₄} D
                F : CategoryTheory.Functor (CategoryTheory.Discrete PUnit.{?u.9385 + 1}) C
                ⊢ ∀ ⦃X Y : CategoryTheory.Discrete PUnit.{?u.9385 + 1}⦄ (f : Quiver.Hom X Y),  …
              -/
              intro X Y f
              match X, Y, f with
              | .mk A, .mk B, .up g =>
                aesop_cat
           }
  }⟩


@[reassoc (attr := simp)]
theorem Cone.w {F : J ⥤ C} (c : Cone F) {j j' : J} (f : j ⟶ j') :
    c.π.app j ≫ F.map f = c.π.app j' := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cone F
    j j' : J
    f : Quiver.Hom j j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app j) (F.map f)) (c.π.app j')
  -/
  rw [← c.π.naturality f]
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cone F
    j j' : J
    f : Quiver.Hom j j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
  -/
  apply id_comp
  /-
    🎉 no goals
  -/


/-- A `c : Cocone F` is
* an object `c.pt` and
* a natural transformation `c.ι : F ⟶ c.pt` from `F` to the constant `c.pt` functor.

For example, if the source `J` of `F` is a partially ordered set, then to give
`c : Cocone F` is to give a collection of morphisms `ιⱼ : F j ⟶ c.pt` and, for
all `j ≤ k` in `J`, morphisms `ιⱼₖ : F j ⟶ F k` such that `Fⱼₖ ≫ Fₖ = Fⱼ` for all `j ≤ k`.

`Cocone F` is equivalent, via `Cone.equiv` below, to `Σ X, F.cocones.obj X`.
-/
structure Cocone (F : J ⥤ C) where
  /-- An object of `C` -/
  pt : C
  /-- A natural transformation from `F` to the constant functor at `pt` -/
  ι : F ⟶ (const J).obj pt


instance inhabitedCocone (F : Discrete PUnit ⥤ C) : Inhabited (Cocone F) :=
  ⟨{  pt := F.obj ⟨⟨⟩⟩
      ι := { app := fun ⟨⟨⟩⟩ => 𝟙 _
             naturality := by
              /-
                J : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                K : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} K
                C : Type u₃
                inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                D : Type u₄
                inst✝ : CategoryTheory.Category.{v₄, u₄} D
                F : CategoryTheory.Functor (CategoryTheory.Discrete PUnit.{?u.12627 + 1}) C
                ⊢ ∀ ⦃X Y : CategoryTheory.Discrete PUnit.{?u.12627 + 1}⦄ (f : Quiver.Hom X Y), …
              -/
              intro X Y f
              match X, Y, f with
              | .mk A, .mk B, .up g =>
                aesop_cat
           }
  }⟩


@[reassoc]
theorem Cocone.w {F : J ⥤ C} (c : Cocone F) {j j' : J} (f : j ⟶ j') :
    F.map f ≫ c.ι.app j' = c.ι.app j := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    j j' : J
    f : Quiver.Hom j j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (c.ι.app j')) (c.ι.app j)
  -/
  rw [c.ι.naturality f]
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    j j' : J
    f : Quiver.Hom j j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) (((CategoryTheory.Functor …
  -/
  apply comp_id
  /-
    🎉 no goals
  -/


/-- The isomorphism between a cone on `F` and an element of the functor `F.cones`. -/
@[simps!]
def equiv (F : J ⥤ C) : Cone F ≅ ΣX, F.cones.obj X where
  hom c := ⟨op c.pt, c.π⟩
  inv c :=
    { pt := c.1.unop
      π := c.2 }
  hom_inv_id := by
    /-
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => ⟨{ unop := c.pt }, c.π⟩) fu …
    -/
    funext X
    /-
      case h
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      X : CategoryTheory.Limits.Cone F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => ⟨{ unop := c.pt }, c.π⟩) (f …
    -/
    cases X
    /-
      case h.mk
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      pt✝ : C
      π✝ : Quiver.Hom ((CategoryTheory.Functor.const J).obj pt✝) F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => ⟨{ unop := c.pt }, c.π⟩) (f …
    -/
    rfl
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => { pt := Opposite.unop c.fst …
    -/
    funext X
    /-
      case h
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      X : Sigma fun X => F.cones.obj X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => { pt := Opposite.unop c.fst …
    -/
    cases X
    /-
      case h.mk
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      fst✝ : Opposite C
      snd✝ : F.cones.obj fst✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => { pt := Opposite.unop c.fst …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- A map to the vertex of a cone naturally induces a cone by composition. -/
@[simps]
def extensions (c : Cone F) : yoneda.obj c.pt ⋙ uliftFunctor.{u₁} ⟶ F.cones where
  app _ f := (const J).map f.down ≫ c.π


/-- A map to the vertex of a cone induces a cone by composition. -/
@[simps]
def extend (c : Cone F) {X : C} (f : X ⟶ c.pt) : Cone F :=
  { pt := X
    π := c.extensions.app (op X) ⟨f⟩ }


/-- Whisker a cone by precomposition of a functor. -/
@[simps]
def whisker (E : K ⥤ J) (c : Cone F) : Cone (E ⋙ F) where
  pt := c.pt
  π := whiskerLeft E c.π


/-- The isomorphism between a cocone on `F` and an element of the functor `F.cocones`. -/
def equiv (F : J ⥤ C) : Cocone F ≅ ΣX, F.cocones.obj X where
  hom c := ⟨c.pt, c.ι⟩
  inv c :=
    { pt := c.1
      ι := c.2 }
  hom_inv_id := by
    /-
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => ⟨c.pt, c.ι⟩) fun c => { pt  …
    -/
    funext X
    /-
      case h
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      X : CategoryTheory.Limits.Cocone F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => ⟨c.pt, c.ι⟩) (fun c => { pt …
    -/
    cases X
    /-
      case h.mk
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      pt✝ : C
      ι✝ : Quiver.Hom F ((CategoryTheory.Functor.const J).obj pt✝)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => ⟨c.pt, c.ι⟩) (fun c => { pt …
    -/
    rfl
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => { pt := c.fst, ι := c.snd } …
    -/
    funext X
    /-
      case h
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      X : Sigma fun X => F.cocones.obj X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => { pt := c.fst, ι := c.snd } …
    -/
    cases X
    /-
      case h.mk
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F✝ F : CategoryTheory.Functor J C
      fst✝ : C
      snd✝ : F.cocones.obj fst✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun c => { pt := c.fst, ι := c.snd } …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- A map from the vertex of a cocone naturally induces a cocone by composition. -/
@[simps]
def extensions (c : Cocone F) : coyoneda.obj (op c.pt) ⋙ uliftFunctor.{u₁} ⟶ F.cocones where
  app _ f := c.ι ≫ (const J).map f.down


/-- A map from the vertex of a cocone induces a cocone by composition. -/
@[simps]
def extend (c : Cocone F) {Y : C} (f : c.pt ⟶ Y) : Cocone F where
  pt := Y
  ι := c.extensions.app Y ⟨f⟩


/-- Whisker a cocone by precomposition of a functor. See `whiskering` for a functorial
version.
-/
@[simps]
def whisker (E : K ⥤ J) (c : Cocone F) : Cocone (E ⋙ F) where
  pt := c.pt
  ι := whiskerLeft E c.ι


/-- A cone morphism between two cones for the same diagram is a morphism of the cone points which
commutes with the cone legs. -/
structure ConeMorphism (A B : Cone F) where
  /-- A morphism between the two vertex objects of the cones -/
  hom : A.pt ⟶ B.pt
  /-- The triangle consisting of the two natural transformations and `hom` commutes -/
  w : ∀ j : J, hom ≫ B.π.app j = A.π.app j := by aesop_cat


attribute [reassoc (attr := simp)] ConeMorphism.w


instance inhabitedConeMorphism (A : Cone F) : Inhabited (ConeMorphism A A) :=
  ⟨{ hom := 𝟙 _ }⟩


/-- The category of cones on a given diagram. -/
@[simps]
instance Cone.category : Category (Cone F) where
  Hom A B := ConeMorphism A B
  comp f g := { hom := f.hom ≫ g.hom }
  id B := { hom := 𝟙 B.pt }

-- Porting note: if we do not have `simps` automatically generate the lemma for simplifying
-- the hom field of a category, we need to write the `ext` lemma in terms of the categorical
-- morphism, rather than the underlying structure.

@[ext]
theorem ConeMorphism.ext {c c' : Cone F} (f g : c ⟶ c') (w : f.hom = g.hom) : f = g := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cone F
    f g : Quiver.Hom c c'
    w : Eq f.hom g.hom
    ⊢ Eq f g
  -/
  cases f
  /-
    case mk
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cone F
    g : Quiver.Hom c c'
    hom✝ : Quiver.Hom c.pt c'.pt
    w✝ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp hom✝ (c'.π.app j)) (c.π …
    w : Eq { hom := hom✝, w := w✝ }.hom g.hom
    ⊢ Eq { hom := hom✝, w := w✝ } g
  -/
  cases g
  /-
    case mk.mk
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cone F
    hom✝¹ : Quiver.Hom c.pt c'.pt
    w✝¹ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp hom✝¹ (c'.π.app j)) (c …
    hom✝ : Quiver.Hom c.pt c'.pt
    w✝ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp hom✝ (c'.π.app j)) (c.π …
    w : Eq { hom := hom✝¹, w := w✝¹ }.hom { hom := hom✝, w := w✝ }.hom
    ⊢ Eq { hom := hom✝¹, w := w✝¹ } { hom := hom✝, w := w✝ }
  -/
  congr
  /-
    🎉 no goals
  -/


/-- To give an isomorphism between cones, it suffices to give an
  isomorphism between their vertices which commutes with the cone
  maps. -/
@[aesop apply safe (rule_sets := [CategoryTheory]), simps]
def ext {c c' : Cone F} (φ : c.pt ≅ c'.pt)
    (w : ∀ j, c.π.app j = φ.hom ≫ c'.π.app j := by aesop_cat) : c ≅ c' where
  hom := { hom := φ.hom }
  inv :=
    { hom := φ.inv
      w := fun j => φ.inv_comp_eq.mpr (w j) }


/-- Eta rule for cones. -/
@[simps!]
def eta (c : Cone F) : c ≅ ⟨c.pt, c.π⟩ :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cone F
    ⊢ ∀ (j : J), Eq (c.π.app j) (CategoryTheory.CategoryStruct.comp (CategoryTheor …
  -/
  Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Given a cone morphism whose object part is an isomorphism, produce an
isomorphism of cones.
-/
theorem cone_iso_of_hom_iso {K : J ⥤ C} {c d : Cone K} (f : c ⟶ d) [i : IsIso f.hom] : IsIso f :=
  ⟨⟨{   hom := inv f.hom
                                                                     /-
                                                                       J : Type u₁
                                                                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                                                       C : Type u₃
                                                                       inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                                       K : CategoryTheory.Functor J C
                                                                       c d : CategoryTheory.Limits.Cone K
                                                                       f : Quiver.Hom c d
                                                                       i : CategoryTheory.IsIso f.hom
                                                                       ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f { hom := CategoryTheory.inv f. …
                                                                     -/
        w := fun j => (asIso f.hom).inv_comp_eq.2 (f.w j).symm }, by aesop_cat⟩⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- There is a morphism from an extended cone to the original cone. -/
@[simps]
def extend (s : Cone F) {X : C} (f : X ⟶ s.pt) : s.extend f ⟶ s where
  hom := f


/-- Extending a cone by the identity does nothing. -/
@[simps!]
def extendId (s : Cone F) : s.extend (𝟙 s.pt) ≅ s :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F : CategoryTheory.Functor J C
    s : CategoryTheory.Limits.Cone F
    ⊢ ∀ (j : J), Eq ((s.extend (CategoryTheory.CategoryStruct.id s.pt)).π.app j) ( …
  -/
  Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Extending a cone by a composition is the same as extending the cone twice. -/
@[simps!]
def extendComp (s : Cone F) {X Y : C} (f : X ⟶ Y) (g : Y ⟶ s.pt) :
    s.extend (f ≫ g) ≅ (s.extend g).extend f :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F : CategoryTheory.Functor J C
    s : CategoryTheory.Limits.Cone F
    X Y : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y s.pt
    ⊢ ∀ (j : J), Eq ((s.extend (CategoryTheory.CategoryStruct.comp f g)).π.app j)  …
  -/
  Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- A cone extended by an isomorphism is isomorphic to the original cone. -/
@[simps]
def extendIso (s : Cone F) {X : C} (f : X ≅ s.pt) : s.extend f.hom ≅ s where
  hom := { hom := f.hom }
  inv := { hom := f.inv }


instance {s : Cone F} {X : C} (f : X ⟶ s.pt) [IsIso f] : IsIso (Cones.extend s f) :=
                                   /-
                                     J : Type u₁
                                     inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
                                     K : Type u₂
                                     inst✝³ : CategoryTheory.Category.{v₂, u₂} K
                                     C : Type u₃
                                     inst✝² : CategoryTheory.Category.{v₃, u₃} C
                                     D : Type u₄
                                     inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
                                     F : CategoryTheory.Functor J C
                                     s : CategoryTheory.Limits.Cone F
                                     X : C
                                     f : Quiver.Hom X s.pt
                                     inst✝ : CategoryTheory.IsIso f
                                     ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cones.ext …
                                   -/
  ⟨(extendIso s (asIso f)).inv, by aesop_cat⟩
                                   /-
                                     🎉 no goals
                                   -/


/--
Functorially postcompose a cone for `F` by a natural transformation `F ⟶ G` to give a cone for `G`.
-/
@[simps]
def postcompose {G : J ⥤ C} (α : F ⟶ G) : Cone F ⥤ Cone G where
  obj c :=
    { pt := c.pt
      π := c.π ≫ α }
  map f := { hom := f.hom }


/-- Postcomposing a cone by the composite natural transformation `α ≫ β` is the same as
postcomposing by `α` and then by `β`. -/
@[simps!]
def postcomposeComp {G H : J ⥤ C} (α : F ⟶ G) (β : G ⟶ H) :
    postcompose (α ≫ β) ≅ postcompose α ⋙ postcompose β :=
                               /-
                                 J : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                 K : Type u₂
                                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                 C : Type u₃
                                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                 D : Type u₄
                                 inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                 F G H : CategoryTheory.Functor J C
                                 α : Quiver.Hom F G
                                 β : Quiver.Hom G H
                                 s : CategoryTheory.Limits.Cone F
                                 ⊢ ∀ (j : J), Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Cat …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun s => Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Postcomposing by the identity does not change the cone up to isomorphism. -/
@[simps!]
def postcomposeId : postcompose (𝟙 F) ≅ 𝟭 (Cone F) :=
                               /-
                                 J : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                 K : Type u₂
                                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                 C : Type u₃
                                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                 D : Type u₄
                                 inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                 F : CategoryTheory.Functor J C
                                 s : CategoryTheory.Limits.Cone F
                                 ⊢ ∀ (j : J), Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Cat …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun s => Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- If `F` and `G` are naturally isomorphic functors, then they have equivalent categories of
cones.
-/
@[simps]
def postcomposeEquivalence {G : J ⥤ C} (α : F ≅ G) : Cone F ≌ Cone G where
  functor := postcompose α.hom
  inverse := postcompose α.inv
                                          /-
                                            J : Type u₁
                                            inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                            K : Type u₂
                                            inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                            C : Type u₃
                                            inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                            D : Type u₄
                                            inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                            F G : CategoryTheory.Functor J C
                                            α : CategoryTheory.Iso F G
                                            s : CategoryTheory.Limits.Cone F
                                            ⊢ ∀ (j : J), Eq (((CategoryTheory.Functor.id (CategoryTheory.Limits.Cone F)).o …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  unitIso := NatIso.ofComponents fun s => Cones.ext (Iso.refl _)
             /-
               🎉 no goals
             -/
                                            /-
                                              J : Type u₁
                                              inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                              K : Type u₂
                                              inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                              C : Type u₃
                                              inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                              D : Type u₄
                                              inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                              F G : CategoryTheory.Functor J C
                                              α : CategoryTheory.Iso F G
                                              s : CategoryTheory.Limits.Cone G
                                              ⊢ ∀ (j : J), Eq ((((CategoryTheory.Limits.Cones.postcompose α.inv).comp (Categ …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  counitIso := NatIso.ofComponents fun s => Cones.ext (Iso.refl _)
               /-
                 🎉 no goals
               -/


/-- Whiskering on the left by `E : K ⥤ J` gives a functor from `Cone F` to `Cone (E ⋙ F)`.
-/
@[simps]
def whiskering (E : K ⥤ J) : Cone F ⥤ Cone (E ⋙ F) where
  obj c := c.whisker E
  map f := { hom := f.hom }


/-- Whiskering by an equivalence gives an equivalence between categories of cones.
-/
@[simps]
def whiskeringEquivalence (e : K ≌ J) : Cone F ≌ Cone (e.functor ⋙ F) where
  functor := whiskering e.functor
  inverse := whiskering e.inverse ⋙ postcompose (e.invFunIdAssoc F).hom
                                          /-
                                            J : Type u₁
                                            inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                            K : Type u₂
                                            inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                            C : Type u₃
                                            inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                            D : Type u₄
                                            inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                            F : CategoryTheory.Functor J C
                                            e : CategoryTheory.Equivalence K J
                                            s : CategoryTheory.Limits.Cone F
                                            ⊢ ∀ (j : J), Eq (((CategoryTheory.Functor.id (CategoryTheory.Limits.Cone F)).o …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  unitIso := NatIso.ofComponents fun s => Cones.ext (Iso.refl _)
             /-
               🎉 no goals
             -/
  counitIso :=
    /-
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F : CategoryTheory.Functor J C
      e : CategoryTheory.Equivalence K J
      ⊢ ∀ {X Y : CategoryTheory.Limits.Cone (e.functor.comp F)} (f : Quiver.Hom X Y) …
    -/
    NatIso.ofComponents
    /-
      🎉 no goals
    -/
      fun s =>
            /-
              J : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} J
              K : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} K
              C : Type u₃
              inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
              D : Type u₄
              inst✝ : CategoryTheory.Category.{v₄, u₄} D
              F : CategoryTheory.Functor J C
              e : CategoryTheory.Equivalence K J
              s : CategoryTheory.Limits.Cone (e.functor.comp F)
              ⊢ ∀ (j : K), Eq (((((CategoryTheory.Limits.Cones.whiskering e.inverse).comp (C …
            -/
        Cones.ext (Iso.refl _)
            /-
              J : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} J
              K : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} K
              C : Type u₃
              inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
              D : Type u₄
              inst✝ : CategoryTheory.Category.{v₄, u₄} D
              F : CategoryTheory.Functor J C
              e : CategoryTheory.Equivalence K J
              s : CategoryTheory.Limits.Cone (e.functor.comp F)
              k : K
              ⊢ Eq (((((CategoryTheory.Limits.Cones.whiskering e.inverse).comp (CategoryTheo …
            -/
          (by
            /-
              🎉 no goals
            -/
            intro k
            simpa [e.counit_app_functor] using s.w (e.unitInv.app k))


/-- The categories of cones over `F` and `G` are equivalent if `F` and `G` are naturally isomorphic
(possibly after changing the indexing category by an equivalence).
-/
@[simps! functor inverse unitIso counitIso]
def equivalenceOfReindexing {G : K ⥤ C} (e : K ≌ J) (α : e.functor ⋙ F ≅ G) : Cone F ≌ Cone G :=
  (whiskeringEquivalence e).trans (postcomposeEquivalence α)


/-- Forget the cone structure and obtain just the cone point. -/
@[simps]
def forget : Cone F ⥤ C where
  obj t := t.pt
  map f := f.hom


/-- A functor `G : C ⥤ D` sends cones over `F` to cones over `F ⋙ G` functorially. -/
@[simps]
def functoriality : Cone F ⥤ Cone (F ⋙ G) where
  obj A :=
    { pt := G.obj A.pt
      π :=
        { app := fun j => G.map (A.π.app j)
                           /-
                             J : Type u₁
                             inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                             K : Type u₂
                             inst✝² : CategoryTheory.Category.{v₂, u₂} K
                             C : Type u₃
                             inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                             D : Type u₄
                             inst✝ : CategoryTheory.Category.{v₄, u₄} D
                             F : CategoryTheory.Functor J C
                             G : CategoryTheory.Functor C D
                             A : CategoryTheory.Limits.Cone F
                             ⊢ ∀ ⦃X Y : J⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                           -/
          naturality := by intros; erw [← G.map_comp]; aesop_cat } }
                                                       /-
                                                         🎉 no goals
                                                       -/
  map f :=
    { hom := G.map f.hom
                       /-
                         J : Type u₁
                         inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                         K : Type u₂
                         inst✝² : CategoryTheory.Category.{v₂, u₂} K
                         C : Type u₃
                         inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                         D : Type u₄
                         inst✝ : CategoryTheory.Category.{v₄, u₄} D
                         F : CategoryTheory.Functor J C
                         G : CategoryTheory.Functor C D
                         X✝ Y✝ : CategoryTheory.Limits.Cone F
                         f : Quiver.Hom X✝ Y✝
                         j : J
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f.hom) (((fun A => { pt := G.o …
                       -/
      w := fun j => by simp [-ConeMorphism.w, ← f.w j] }
                       /-
                         🎉 no goals
                       -/


instance functoriality_full [G.Full] [G.Faithful] : (functoriality F G).Full where
  map_surjective t :=
    ⟨{ hom := G.preimage t.hom
                                         /-
                                           J : Type u₁
                                           inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
                                           K : Type u₂
                                           inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
                                           C : Type u₃
                                           inst✝³ : CategoryTheory.Category.{v₃, u₃} C
                                           D : Type u₄
                                           inst✝² : CategoryTheory.Category.{v₄, u₄} D
                                           F : CategoryTheory.Functor J C
                                           G : CategoryTheory.Functor C D
                                           inst✝¹ : G.Full
                                           inst✝ : G.Faithful
                                           X✝ Y✝ : CategoryTheory.Limits.Cone F
                                           t : Quiver.Hom ((CategoryTheory.Limits.Cones.functoriality F G).obj X✝) ((Cate …
                                           j : J
                                           ⊢ Eq (G.map (CategoryTheory.CategoryStruct.comp (G.preimage t.hom) (Y✝.π.app j …
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
       w := fun j => G.map_injective (by simpa using t.w j) }, by aesop_cat⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance functoriality_faithful [G.Faithful] : (Cones.functoriality F G).Faithful where
  map_injective {_X} {_Y} f g h :=
    ConeMorphism.ext f g <| G.map_injective <| congr_arg ConeMorphism.hom h


/-- If `e : C ≌ D` is an equivalence of categories, then `functoriality F e.functor` induces an
equivalence between cones over `F` and cones over `F ⋙ e.functor`.
-/
@[simps]
def functorialityEquivalence (e : C ≌ D) : Cone F ≌ Cone (F ⋙ e.functor) :=
  let f : (F ⋙ e.functor) ⋙ e.inverse ≅ F :=
    Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ e.unitIso.symm ≪≫ Functor.rightUnitor _
  { functor := functoriality F e.functor
    inverse := functoriality (F ⋙ e.functor) e.inverse ⋙ (postcomposeEquivalence f).functor
                                            /-
                                              J : Type u₁
                                              inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                              K : Type u₂
                                              inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                              C : Type u₃
                                              inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                              D : Type u₄
                                              inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                              F : CategoryTheory.Functor J C
                                              G : CategoryTheory.Functor C D
                                              e : CategoryTheory.Equivalence C D
                                              f : CategoryTheory.Iso ((F.comp e.functor).comp e.inverse) F := (F.associator  …
                                              c : CategoryTheory.Limits.Cone F
                                              ⊢ ∀ (j : J), Eq (((CategoryTheory.Functor.id (CategoryTheory.Limits.Cone F)).o …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
    unitIso := NatIso.ofComponents fun c => Cones.ext (e.unitIso.app _)
               /-
                 🎉 no goals
               -/
                                              /-
                                                J : Type u₁
                                                inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                                K : Type u₂
                                                inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                                C : Type u₃
                                                inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                                D : Type u₄
                                                inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                                F : CategoryTheory.Functor J C
                                                G : CategoryTheory.Functor C D
                                                e : CategoryTheory.Equivalence C D
                                                f : CategoryTheory.Iso ((F.comp e.functor).comp e.inverse) F := (F.associator  …
                                                c : CategoryTheory.Limits.Cone (F.comp e.functor)
                                                ⊢ ∀ (j : J), Eq (((((CategoryTheory.Limits.Cones.functoriality (F.comp e.funct …
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
    counitIso := NatIso.ofComponents fun c => Cones.ext (e.counitIso.app _) }
                 /-
                   🎉 no goals
                 -/


/-- If `F` reflects isomorphisms, then `Cones.functoriality F` reflects isomorphisms
as well.
-/
instance reflects_cone_isomorphism (F : C ⥤ D) [F.ReflectsIsomorphisms] (K : J ⥤ C) :
    (Cones.functoriality K F).ReflectsIsomorphisms := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
    F✝ : CategoryTheory.Functor J C
    G F : CategoryTheory.Functor C D
    inst✝ : F.ReflectsIsomorphisms
    K : CategoryTheory.Functor J C
    ⊢ (CategoryTheory.Limits.Cones.functoriality K F).ReflectsIsomorphisms
  -/
  constructor
  /-
    case reflects
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
    F✝ : CategoryTheory.Functor J C
    G F : CategoryTheory.Functor C D
    inst✝ : F.ReflectsIsomorphisms
    K : CategoryTheory.Functor J C
    ⊢ ∀ {A B : CategoryTheory.Limits.Cone K} (f : Quiver.Hom A B) [inst : Category …
  -/
  intro A B f _
  haveI : IsIso (F.map f.hom) :=
    (Cones.forget (K ⋙ F)).map_isIso ((Cones.functoriality K F).map f)
  /-
    case reflects
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    F✝ : CategoryTheory.Functor J C
    G F : CategoryTheory.Functor C D
    inst✝¹ : F.ReflectsIsomorphisms
    K : CategoryTheory.Functor J C
    A B : CategoryTheory.Limits.Cone K
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.IsIso ((CategoryTheory.Limits.Cones.functoriality K F). …
    this : CategoryTheory.IsIso (F.map f.hom)
    ⊢ CategoryTheory.IsIso f
  -/
  haveI := ReflectsIsomorphisms.reflects F f.hom
  /-
    case reflects
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    F✝ : CategoryTheory.Functor J C
    G F : CategoryTheory.Functor C D
    inst✝¹ : F.ReflectsIsomorphisms
    K : CategoryTheory.Functor J C
    A B : CategoryTheory.Limits.Cone K
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.IsIso ((CategoryTheory.Limits.Cones.functoriality K F). …
    this✝ : CategoryTheory.IsIso (F.map f.hom)
    this : CategoryTheory.IsIso f.hom
    ⊢ CategoryTheory.IsIso f
  -/
  apply cone_iso_of_hom_iso
  /-
    🎉 no goals
  -/


/-- A cocone morphism between two cocones for the same diagram is a morphism of the cocone points
which commutes with the cocone legs. -/
structure CoconeMorphism (A B : Cocone F) where
  /-- A morphism between the (co)vertex objects in `C` -/
  hom : A.pt ⟶ B.pt
  /-- The triangle made from the two natural transformations and `hom` commutes -/
  w : ∀ j : J, A.ι.app j ≫ hom = B.ι.app j := by aesop_cat


instance inhabitedCoconeMorphism (A : Cocone F) : Inhabited (CoconeMorphism A A) :=
  ⟨{ hom := 𝟙 _ }⟩


attribute [reassoc (attr := simp)] CoconeMorphism.w


@[simps]
instance Cocone.category : Category (Cocone F) where
  Hom A B := CoconeMorphism A B
  comp f g := { hom := f.hom ≫ g.hom }
  id B := { hom := 𝟙 B.pt }

-- Porting note: if we do not have `simps` automatically generate the lemma for simplifying
-- the hom field of a category, we need to write the `ext` lemma in terms of the categorical
-- morphism, rather than the underlying structure.

@[ext]
theorem CoconeMorphism.ext {c c' : Cocone F} (f g : c ⟶ c') (w : f.hom = g.hom) : f = g := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    f g : Quiver.Hom c c'
    w : Eq f.hom g.hom
    ⊢ Eq f g
  -/
  cases f
  /-
    case mk
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    g : Quiver.Hom c c'
    hom✝ : Quiver.Hom c.pt c'.pt
    w✝ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) hom✝) (c'.ι …
    w : Eq { hom := hom✝, w := w✝ }.hom g.hom
    ⊢ Eq { hom := hom✝, w := w✝ } g
  -/
  cases g
  /-
    case mk.mk
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    hom✝¹ : Quiver.Hom c.pt c'.pt
    w✝¹ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) hom✝¹) (c' …
    hom✝ : Quiver.Hom c.pt c'.pt
    w✝ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) hom✝) (c'.ι …
    w : Eq { hom := hom✝¹, w := w✝¹ }.hom { hom := hom✝, w := w✝ }.hom
    ⊢ Eq { hom := hom✝¹, w := w✝¹ } { hom := hom✝, w := w✝ }
  -/
  congr
  /-
    🎉 no goals
  -/


/-- To give an isomorphism between cocones, it suffices to give an
  isomorphism between their vertices which commutes with the cocone
  maps. -/
@[aesop apply safe (rule_sets := [CategoryTheory]), simps]
def ext {c c' : Cocone F} (φ : c.pt ≅ c'.pt)
    (w : ∀ j, c.ι.app j ≫ φ.hom = c'.ι.app j := by aesop_cat) : c ≅ c' where
  hom := { hom := φ.hom }
  inv :=
    { hom := φ.inv
      w := fun j => φ.comp_inv_eq.mpr (w j).symm }


/-- Eta rule for cocones. -/
@[simps!]
def eta (c : Cocone F) : c ≅ ⟨c.pt, c.ι⟩ :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) (CategoryTheor …
  -/
  Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Given a cocone morphism whose object part is an isomorphism, produce an
isomorphism of cocones.
-/
theorem cocone_iso_of_hom_iso {K : J ⥤ C} {c d : Cocone K} (f : c ⟶ d) [i : IsIso f.hom] :
    IsIso f :=
  ⟨⟨{ hom := inv f.hom
                                                                   /-
                                                                     J : Type u₁
                                                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                                                     C : Type u₃
                                                                     inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                                     K : CategoryTheory.Functor J C
                                                                     c d : CategoryTheory.Limits.Cocone K
                                                                     f : Quiver.Hom c d
                                                                     i : CategoryTheory.IsIso f.hom
                                                                     ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f { hom := CategoryTheory.inv f. …
                                                                   -/
      w := fun j => (asIso f.hom).comp_inv_eq.2 (f.w j).symm }, by aesop_cat⟩⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- There is a morphism from a cocone to its extension. -/
@[simps]
def extend (s : Cocone F) {X : C} (f : s.pt ⟶ X) : s ⟶ s.extend f where
  hom := f


/-- Extending a cocone by the identity does nothing. -/
@[simps!]
def extendId (s : Cocone F) : s ≅ s.extend (𝟙 s.pt) :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F : CategoryTheory.Functor J C
    s : CategoryTheory.Limits.Cocone F
    ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) (CategoryTheor …
  -/
  Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Extending a cocone by a composition is the same as extending the cone twice. -/
@[simps!]
def extendComp (s : Cocone F) {X Y : C} (f : s.pt ⟶ X) (g : X ⟶ Y) :
    s.extend (f ≫ g) ≅ (s.extend f).extend g :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F : CategoryTheory.Functor J C
    s : CategoryTheory.Limits.Cocone F
    X Y : C
    f : Quiver.Hom s.pt X
    g : Quiver.Hom X Y
    ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((s.extend (CategoryTheory …
  -/
  Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- A cocone extended by an isomorphism is isomorphic to the original cone. -/
@[simps]
def extendIso (s : Cocone F) {X : C} (f : s.pt ≅ X) : s ≅ s.extend f.hom where
  hom := { hom := f.hom }
  inv := { hom := f.inv }


instance {s : Cocone F} {X : C} (f : s.pt ⟶ X) [IsIso f] : IsIso (Cocones.extend s f) :=
                                   /-
                                     J : Type u₁
                                     inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
                                     K : Type u₂
                                     inst✝³ : CategoryTheory.Category.{v₂, u₂} K
                                     C : Type u₃
                                     inst✝² : CategoryTheory.Category.{v₃, u₃} C
                                     D : Type u₄
                                     inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
                                     F : CategoryTheory.Functor J C
                                     s : CategoryTheory.Limits.Cocone F
                                     X : C
                                     f : Quiver.Hom s.pt X
                                     inst✝ : CategoryTheory.IsIso f
                                     ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cocones.e …
                                   -/
  ⟨(extendIso s (asIso f)).inv, by aesop_cat⟩
                                   /-
                                     🎉 no goals
                                   -/


/-- Functorially precompose a cocone for `F` by a natural transformation `G ⟶ F` to give a cocone
for `G`. -/
@[simps]
def precompose {G : J ⥤ C} (α : G ⟶ F) : Cocone F ⥤ Cocone G where
  obj c :=
    { pt := c.pt
      ι := α ≫ c.ι }
  map f := { hom := f.hom }


/-- Precomposing a cocone by the composite natural transformation `α ≫ β` is the same as
precomposing by `β` and then by `α`. -/
def precomposeComp {G H : J ⥤ C} (α : F ⟶ G) (β : G ⟶ H) :
    precompose (α ≫ β) ≅ precompose β ⋙ precompose α :=
                               /-
                                 J : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                 K : Type u₂
                                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                 C : Type u₃
                                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                 D : Type u₄
                                 inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                 F G H : CategoryTheory.Functor J C
                                 α : Quiver.Hom F G
                                 β : Quiver.Hom G H
                                 s : CategoryTheory.Limits.Cocone H
                                 ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.C …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun s => Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Precomposing by the identity does not change the cocone up to isomorphism. -/
def precomposeId : precompose (𝟙 F) ≅ 𝟭 (Cocone F) :=
                               /-
                                 J : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                 K : Type u₂
                                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                 C : Type u₃
                                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                 D : Type u₄
                                 inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                 F : CategoryTheory.Functor J C
                                 s : CategoryTheory.Limits.Cocone F
                                 ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.C …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun s => Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- If `F` and `G` are naturally isomorphic functors, then they have equivalent categories of
cocones.
-/
@[simps]
def precomposeEquivalence {G : J ⥤ C} (α : G ≅ F) : Cocone F ≌ Cocone G where
  functor := precompose α.hom
  inverse := precompose α.inv
                                          /-
                                            J : Type u₁
                                            inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                            K : Type u₂
                                            inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                            C : Type u₃
                                            inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                            D : Type u₄
                                            inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                            F G : CategoryTheory.Functor J C
                                            α : CategoryTheory.Iso G F
                                            s : CategoryTheory.Limits.Cocone F
                                            ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor. …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  unitIso := NatIso.ofComponents fun s => Cocones.ext (Iso.refl _)
             /-
               🎉 no goals
             -/
                                            /-
                                              J : Type u₁
                                              inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                              K : Type u₂
                                              inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                              C : Type u₃
                                              inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                              D : Type u₄
                                              inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                              F G : CategoryTheory.Functor J C
                                              α : CategoryTheory.Iso G F
                                              s : CategoryTheory.Limits.Cocone G
                                              ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.Limits. …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  counitIso := NatIso.ofComponents fun s => Cocones.ext (Iso.refl _)
               /-
                 🎉 no goals
               -/


/-- Whiskering on the left by `E : K ⥤ J` gives a functor from `Cocone F` to `Cocone (E ⋙ F)`.
-/
@[simps]
def whiskering (E : K ⥤ J) : Cocone F ⥤ Cocone (E ⋙ F) where
  obj c := c.whisker E
  map f := { hom := f.hom }


/-- Whiskering by an equivalence gives an equivalence between categories of cones.
-/
@[simps]
def whiskeringEquivalence (e : K ≌ J) : Cocone F ≌ Cocone (e.functor ⋙ F) where
  functor := whiskering e.functor
  inverse :=
    whiskering e.inverse ⋙
      precompose
        ((Functor.leftUnitor F).inv ≫
          whiskerRight e.counitIso.inv F ≫ (Functor.associator _ _ _).inv)
                                          /-
                                            J : Type u₁
                                            inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                            K : Type u₂
                                            inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                            C : Type u₃
                                            inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                            D : Type u₄
                                            inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                            F : CategoryTheory.Functor J C
                                            e : CategoryTheory.Equivalence K J
                                            s : CategoryTheory.Limits.Cocone F
                                            ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor. …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  unitIso := NatIso.ofComponents fun s => Cocones.ext (Iso.refl _)
             /-
               🎉 no goals
             -/
               /-
                 J : Type u₁
                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                 K : Type u₂
                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                 C : Type u₃
                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                 D : Type u₄
                 inst✝ : CategoryTheory.Category.{v₄, u₄} D
                 F : CategoryTheory.Functor J C
                 e : CategoryTheory.Equivalence K J
                 ⊢ ∀ {X Y : CategoryTheory.Limits.Cocone (e.functor.comp F)} (f : Quiver.Hom X  …
               -/
                                         /-
                                           J : Type u₁
                                           inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                           K : Type u₂
                                           inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                           C : Type u₃
                                           inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                           D : Type u₄
                                           inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                           F : CategoryTheory.Functor J C
                                           e : CategoryTheory.Equivalence K J
                                           s : CategoryTheory.Limits.Cocone (e.functor.comp F)
                                           k : K
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (((((CategoryTheory.Limits.Cocones.wh …
                                         -/
  counitIso := NatIso.ofComponents fun s =>
                                         /-
                                           🎉 no goals
                                         -/
               /-
                 🎉 no goals
               -/
    Cocones.ext (Iso.refl _) fun k => by simpa [e.counitInv_app_functor k] using s.w (e.unit.app k)


/--
The categories of cocones over `F` and `G` are equivalent if `F` and `G` are naturally isomorphic
(possibly after changing the indexing category by an equivalence).
-/
@[simps! functor_obj]
def equivalenceOfReindexing {G : K ⥤ C} (e : K ≌ J) (α : e.functor ⋙ F ≅ G) : Cocone F ≌ Cocone G :=
  (whiskeringEquivalence e).trans (precomposeEquivalence α.symm)


/-- Forget the cocone structure and obtain just the cocone point. -/
@[simps]
def forget : Cocone F ⥤ C where
  obj t := t.pt
  map f := f.hom


/-- A functor `G : C ⥤ D` sends cocones over `F` to cocones over `F ⋙ G` functorially. -/
@[simps]
def functoriality : Cocone F ⥤ Cocone (F ⋙ G) where
  obj A :=
    { pt := G.obj A.pt
      ι :=
        { app := fun j => G.map (A.ι.app j)
                           /-
                             J : Type u₁
                             inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                             K : Type u₂
                             inst✝² : CategoryTheory.Category.{v₂, u₂} K
                             C : Type u₃
                             inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                             D : Type u₄
                             inst✝ : CategoryTheory.Category.{v₄, u₄} D
                             F : CategoryTheory.Functor J C
                             G : CategoryTheory.Functor C D
                             A : CategoryTheory.Limits.Cocone F
                             ⊢ ∀ ⦃X Y : J⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((F …
                           -/
          naturality := by intros; erw [← G.map_comp]; aesop_cat } }
                                                       /-
                                                         🎉 no goals
                                                       -/
  map f :=
    { hom := G.map f.hom
              /-
                J : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                K : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} K
                C : Type u₃
                inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                D : Type u₄
                inst✝ : CategoryTheory.Category.{v₄, u₄} D
                F : CategoryTheory.Functor J C
                G : CategoryTheory.Functor C D
                X✝ Y✝ : CategoryTheory.Limits.Cocone F
                f : Quiver.Hom X✝ Y✝
                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((fun A => { pt := G.obj  …
              -/
      w := by intros; rw [← Functor.map_comp, CoconeMorphism.w] }
                      /-
                        🎉 no goals
                      -/


instance functoriality_full [G.Full] [G.Faithful] : (functoriality F G).Full where
  map_surjective t :=
    ⟨{ hom := G.preimage t.hom
                                         /-
                                           J : Type u₁
                                           inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
                                           K : Type u₂
                                           inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
                                           C : Type u₃
                                           inst✝³ : CategoryTheory.Category.{v₃, u₃} C
                                           D : Type u₄
                                           inst✝² : CategoryTheory.Category.{v₄, u₄} D
                                           F : CategoryTheory.Functor J C
                                           G : CategoryTheory.Functor C D
                                           inst✝¹ : G.Full
                                           inst✝ : G.Faithful
                                           X✝ Y✝ : CategoryTheory.Limits.Cocone F
                                           t : Quiver.Hom ((CategoryTheory.Limits.Cocones.functoriality F G).obj X✝) ((Ca …
                                           j : J
                                           ⊢ Eq (G.map (CategoryTheory.CategoryStruct.comp (X✝.ι.app j) (G.preimage t.hom …
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
       w := fun j => G.map_injective (by simpa using t.w j) }, by aesop_cat⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance functoriality_faithful [G.Faithful] : (functoriality F G).Faithful where
  map_injective {_X} {_Y} f g h :=
    CoconeMorphism.ext f g <| G.map_injective <| congr_arg CoconeMorphism.hom h


/-- If `e : C ≌ D` is an equivalence of categories, then `functoriality F e.functor` induces an
equivalence between cocones over `F` and cocones over `F ⋙ e.functor`.
-/
@[simps]
def functorialityEquivalence (e : C ≌ D) : Cocone F ≌ Cocone (F ⋙ e.functor) :=
  let f : (F ⋙ e.functor) ⋙ e.inverse ≅ F :=
    Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ e.unitIso.symm ≪≫ Functor.rightUnitor _
  { functor := functoriality F e.functor
    inverse := functoriality (F ⋙ e.functor) e.inverse ⋙ (precomposeEquivalence f.symm).functor
                                            /-
                                              J : Type u₁
                                              inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                              K : Type u₂
                                              inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                              C : Type u₃
                                              inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                              D : Type u₄
                                              inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                              F : CategoryTheory.Functor J C
                                              G : CategoryTheory.Functor C D
                                              e : CategoryTheory.Equivalence C D
                                              f : CategoryTheory.Iso ((F.comp e.functor).comp e.inverse) F := (F.associator  …
                                              c : CategoryTheory.Limits.Cocone F
                                              ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor. …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
    unitIso := NatIso.ofComponents fun c => Cocones.ext (e.unitIso.app _)
               /-
                 🎉 no goals
               -/
                                              /-
                                                J : Type u₁
                                                inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                                K : Type u₂
                                                inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                                C : Type u₃
                                                inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                                D : Type u₄
                                                inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                                F : CategoryTheory.Functor J C
                                                G : CategoryTheory.Functor C D
                                                e : CategoryTheory.Equivalence C D
                                                f : CategoryTheory.Iso ((F.comp e.functor).comp e.inverse) F := (F.associator  …
                                                c : CategoryTheory.Limits.Cocone (F.comp e.functor)
                                                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((((CategoryTheory.Limits …
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
    counitIso := NatIso.ofComponents fun c => Cocones.ext (e.counitIso.app _) }
                 /-
                   🎉 no goals
                 -/


/-- If `F` reflects isomorphisms, then `Cocones.functoriality F` reflects isomorphisms
as well.
-/
instance reflects_cocone_isomorphism (F : C ⥤ D) [F.ReflectsIsomorphisms] (K : J ⥤ C) :
    (Cocones.functoriality K F).ReflectsIsomorphisms := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
    F✝ : CategoryTheory.Functor J C
    G F : CategoryTheory.Functor C D
    inst✝ : F.ReflectsIsomorphisms
    K : CategoryTheory.Functor J C
    ⊢ (CategoryTheory.Limits.Cocones.functoriality K F).ReflectsIsomorphisms
  -/
  constructor
  /-
    case reflects
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
    F✝ : CategoryTheory.Functor J C
    G F : CategoryTheory.Functor C D
    inst✝ : F.ReflectsIsomorphisms
    K : CategoryTheory.Functor J C
    ⊢ ∀ {A B : CategoryTheory.Limits.Cocone K} (f : Quiver.Hom A B) [inst : Catego …
  -/
  intro A B f _
  haveI : IsIso (F.map f.hom) :=
    (Cocones.forget (K ⋙ F)).map_isIso ((Cocones.functoriality K F).map f)
  /-
    case reflects
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    F✝ : CategoryTheory.Functor J C
    G F : CategoryTheory.Functor C D
    inst✝¹ : F.ReflectsIsomorphisms
    K : CategoryTheory.Functor J C
    A B : CategoryTheory.Limits.Cocone K
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.IsIso ((CategoryTheory.Limits.Cocones.functoriality K F …
    this : CategoryTheory.IsIso (F.map f.hom)
    ⊢ CategoryTheory.IsIso f
  -/
  haveI := ReflectsIsomorphisms.reflects F f.hom
  /-
    case reflects
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    F✝ : CategoryTheory.Functor J C
    G F : CategoryTheory.Functor C D
    inst✝¹ : F.ReflectsIsomorphisms
    K : CategoryTheory.Functor J C
    A B : CategoryTheory.Limits.Cocone K
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.IsIso ((CategoryTheory.Limits.Cocones.functoriality K F …
    this✝ : CategoryTheory.IsIso (F.map f.hom)
    this : CategoryTheory.IsIso f.hom
    ⊢ CategoryTheory.IsIso f
  -/
  apply cocone_iso_of_hom_iso
  /-
    🎉 no goals
  -/


/-- The image of a cone in C under a functor G : C ⥤ D is a cone in D. -/
@[simps!]
def mapCone (c : Cone F) : Cone (F ⋙ H) :=
  (Cones.functoriality F H).obj c


/-- The image of a cocone in C under a functor G : C ⥤ D is a cocone in D. -/
@[simps!]
def mapCocone (c : Cocone F) : Cocone (F ⋙ H) :=
  (Cocones.functoriality F H).obj c


/-- Given a cone morphism `c ⟶ c'`, construct a cone morphism on the mapped cones functorially. -/
def mapConeMorphism {c c' : Cone F} (f : c ⟶ c') : H.mapCone c ⟶ H.mapCone c' :=
  (Cones.functoriality F H).map f


/-- Given a cocone morphism `c ⟶ c'`, construct a cocone morphism on the mapped cocones
functorially. -/
def mapCoconeMorphism {c c' : Cocone F} (f : c ⟶ c') : H.mapCocone c ⟶ H.mapCocone c' :=
  (Cocones.functoriality F H).map f


/-- If `H` is an equivalence, we invert `H.mapCone` and get a cone for `F` from a cone
for `F ⋙ H`. -/
noncomputable def mapConeInv [IsEquivalence H] (c : Cone (F ⋙ H)) : Cone F :=
  (Limits.Cones.functorialityEquivalence F (asEquivalence H)).inverse.obj c


/-- `mapCone` is the left inverse to `mapConeInv`. -/
noncomputable def mapConeMapConeInv {F : J ⥤ D} (H : D ⥤ C) [IsEquivalence H]
    (c : Cone (F ⋙ H)) :
    mapCone H (mapConeInv H c) ≅ c :=
  (Limits.Cones.functorialityEquivalence F (asEquivalence H)).counitIso.app c


/-- `MapCone` is the right inverse to `mapConeInv`. -/
noncomputable def mapConeInvMapCone {F : J ⥤ D} (H : D ⥤ C) [IsEquivalence H] (c : Cone F) :
    mapConeInv H (mapCone H c) ≅ c :=
  (Limits.Cones.functorialityEquivalence F (asEquivalence H)).unitIso.symm.app c


/-- If `H` is an equivalence, we invert `H.mapCone` and get a cone for `F` from a cone
for `F ⋙ H`. -/
noncomputable def mapCoconeInv [IsEquivalence H] (c : Cocone (F ⋙ H)) : Cocone F :=
  (Limits.Cocones.functorialityEquivalence F (asEquivalence H)).inverse.obj c


/-- `mapCocone` is the left inverse to `mapCoconeInv`. -/
noncomputable def mapCoconeMapCoconeInv {F : J ⥤ D} (H : D ⥤ C) [IsEquivalence H]
    (c : Cocone (F ⋙ H)) :
    mapCocone H (mapCoconeInv H c) ≅ c :=
  (Limits.Cocones.functorialityEquivalence F (asEquivalence H)).counitIso.app c


/-- `mapCocone` is the right inverse to `mapCoconeInv`. -/
noncomputable def mapCoconeInvMapCocone {F : J ⥤ D} (H : D ⥤ C) [IsEquivalence H] (c : Cocone F) :
    mapCoconeInv H (mapCocone H c) ≅ c :=
  (Limits.Cocones.functorialityEquivalence F (asEquivalence H)).unitIso.symm.app c


/-- `functoriality F _ ⋙ postcompose (whisker_left F _)` simplifies to `functoriality F _`. -/
@[simps!]
def functorialityCompPostcompose {H H' : C ⥤ D} (α : H ≅ H') :
    Cones.functoriality F H ⋙ Cones.postcompose (whiskerLeft F α.hom) ≅ Cones.functoriality F H' :=
                               /-
                                 J : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                 K : Type u₂
                                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                 C : Type u₃
                                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                 D : Type u₄
                                 inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                 H✝ : CategoryTheory.Functor C D
                                 F G : CategoryTheory.Functor J C
                                 H H' : CategoryTheory.Functor C D
                                 α : CategoryTheory.Iso H H'
                                 c : CategoryTheory.Limits.Cone F
                                 ⊢ ∀ (j : J), Eq ((((CategoryTheory.Limits.Cones.functoriality F H).comp (Categ …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun c => Cones.ext (α.app _)
  /-
    🎉 no goals
  -/


/-- For `F : J ⥤ C`, given a cone `c : Cone F`, and a natural isomorphism `α : H ≅ H'` for functors
`H H' : C ⥤ D`, the postcomposition of the cone `H.mapCone` using the isomorphism `α` is
isomorphic to the cone `H'.mapCone`.
-/
@[simps!]
def postcomposeWhiskerLeftMapCone {H H' : C ⥤ D} (α : H ≅ H') (c : Cone F) :
    (Cones.postcompose (whiskerLeft F α.hom : _)).obj (mapCone H c) ≅ mapCone H' c :=
  (functorialityCompPostcompose α).app c


/--
`mapCone` commutes with `postcompose`. In particular, for `F : J ⥤ C`, given a cone `c : Cone F`, a
natural transformation `α : F ⟶ G` and a functor `H : C ⥤ D`, we have two obvious ways of producing
a cone over `G ⋙ H`, and they are both isomorphic.
-/
@[simps!]
def mapConePostcompose {α : F ⟶ G} {c} :
    mapCone H ((Cones.postcompose α).obj c) ≅
      (Cones.postcompose (whiskerRight α H : _)).obj (mapCone H c) :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    H : CategoryTheory.Functor C D
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    c : CategoryTheory.Limits.Cone F
    ⊢ ∀ (j : J), Eq ((H.mapCone ((CategoryTheory.Limits.Cones.postcompose α).obj c …
  -/
  Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- `mapCone` commutes with `postcomposeEquivalence`
-/
@[simps!]
def mapConePostcomposeEquivalenceFunctor {α : F ≅ G} {c} :
    mapCone H ((Cones.postcomposeEquivalence α).functor.obj c) ≅
      (Cones.postcomposeEquivalence (isoWhiskerRight α H : _)).functor.obj (mapCone H c) :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    H : CategoryTheory.Functor C D
    F G : CategoryTheory.Functor J C
    α : CategoryTheory.Iso F G
    c : CategoryTheory.Limits.Cone F
    ⊢ ∀ (j : J), Eq ((H.mapCone ((CategoryTheory.Limits.Cones.postcomposeEquivalen …
  -/
  Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- `functoriality F _ ⋙ precompose (whiskerLeft F _)` simplifies to `functoriality F _`. -/
@[simps!]
def functorialityCompPrecompose {H H' : C ⥤ D} (α : H ≅ H') :
    Cocones.functoriality F H ⋙ Cocones.precompose (whiskerLeft F α.inv) ≅
      Cocones.functoriality F H' :=
                               /-
                                 J : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                 K : Type u₂
                                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                 C : Type u₃
                                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                 D : Type u₄
                                 inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                 H✝ : CategoryTheory.Functor C D
                                 F G : CategoryTheory.Functor J C
                                 H H' : CategoryTheory.Functor C D
                                 α : CategoryTheory.Iso H H'
                                 c : CategoryTheory.Limits.Cocone F
                                 ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.Limits. …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun c => Cocones.ext (α.app _)
  /-
    🎉 no goals
  -/


/--
For `F : J ⥤ C`, given a cocone `c : Cocone F`, and a natural isomorphism `α : H ≅ H'` for functors
`H H' : C ⥤ D`, the precomposition of the cocone `H.mapCocone` using the isomorphism `α` is
isomorphic to the cocone `H'.mapCocone`.
-/
@[simps!]
def precomposeWhiskerLeftMapCocone {H H' : C ⥤ D} (α : H ≅ H') (c : Cocone F) :
    (Cocones.precompose (whiskerLeft F α.inv : _)).obj (mapCocone H c) ≅ mapCocone H' c :=
  (functorialityCompPrecompose α).app c


/-- `map_cocone` commutes with `precompose`. In particular, for `F : J ⥤ C`, given a cocone
`c : Cocone F`, a natural transformation `α : F ⟶ G` and a functor `H : C ⥤ D`, we have two obvious
ways of producing a cocone over `G ⋙ H`, and they are both isomorphic.
-/
@[simps!]
def mapCoconePrecompose {α : F ⟶ G} {c} :
    mapCocone H ((Cocones.precompose α).obj c) ≅
      (Cocones.precompose (whiskerRight α H : _)).obj (mapCocone H c) :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    H : CategoryTheory.Functor C D
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    c : CategoryTheory.Limits.Cocone G
    ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((H.mapCocone ((CategoryTh …
  -/
  Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- `mapCocone` commutes with `precomposeEquivalence`
-/
@[simps!]
def mapCoconePrecomposeEquivalenceFunctor {α : F ≅ G} {c} :
    mapCocone H ((Cocones.precomposeEquivalence α).functor.obj c) ≅
      (Cocones.precomposeEquivalence (isoWhiskerRight α H : _)).functor.obj (mapCocone H c) :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    H : CategoryTheory.Functor C D
    F G : CategoryTheory.Functor J C
    α : CategoryTheory.Iso F G
    c : CategoryTheory.Limits.Cocone G
    ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((H.mapCocone ((CategoryTh …
  -/
  Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- `mapCone` commutes with `whisker`
-/
@[simps!]
def mapConeWhisker {E : K ⥤ J} {c : Cone F} : mapCone H (c.whisker E) ≅ (mapCone H c).whisker E :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    H : CategoryTheory.Functor C D
    F G : CategoryTheory.Functor J C
    E : CategoryTheory.Functor K J
    c : CategoryTheory.Limits.Cone F
    ⊢ ∀ (j : K), Eq ((H.mapCone (CategoryTheory.Limits.Cone.whisker E c)).π.app j) …
  -/
  Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- `mapCocone` commutes with `whisker`
-/
@[simps!]
def mapCoconeWhisker {E : K ⥤ J} {c : Cocone F} :
    mapCocone H (c.whisker E) ≅ (mapCocone H c).whisker E :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    H : CategoryTheory.Functor C D
    F G : CategoryTheory.Functor J C
    E : CategoryTheory.Functor K J
    c : CategoryTheory.Limits.Cocone F
    ⊢ ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp ((H.mapCocone (CategoryThe …
  -/
  Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Change a `Cocone F` into a `Cone F.op`. -/
@[simps]
def Cocone.op (c : Cocone F) : Cone F.op where
  pt := Opposite.op c.pt
  π := NatTrans.op c.ι


/-- Change a `Cone F` into a `Cocone F.op`. -/
@[simps]
def Cone.op (c : Cone F) : Cocone F.op where
  pt := Opposite.op c.pt
  ι := NatTrans.op c.π


/-- Change a `Cocone F.op` into a `Cone F`. -/
@[simps]
def Cocone.unop (c : Cocone F.op) : Cone F where
  pt := Opposite.unop c.pt
  π := NatTrans.removeOp c.ι


/-- Change a `Cone F.op` into a `Cocone F`. -/
@[simps]
def Cone.unop (c : Cone F.op) : Cocone F where
  pt := Opposite.unop c.pt
  ι := NatTrans.removeOp c.π


/-- The category of cocones on `F`
is equivalent to the opposite category of
the category of cones on the opposite of `F`.
-/
def coconeEquivalenceOpConeOp : Cocone F ≌ (Cone F.op)ᵒᵖ where
  functor :=
    { obj := fun c => op (Cocone.op c)
      map := fun {X} {Y} f =>
        Quiver.Hom.op
          { hom := f.hom.op
            w := fun j => by
              /-
                J : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                K : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} K
                C : Type u₃
                inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                D : Type u₄
                inst✝ : CategoryTheory.Category.{v₄, u₄} D
                F : CategoryTheory.Functor J C
                X Y : CategoryTheory.Limits.Cocone F
                f : Quiver.Hom X Y
                j : Opposite J
                ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.op (X.op.π.app j)) (Y.op.π.app j)
              -/
              apply Quiver.Hom.unop_inj
              /-
                case a
                J : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                K : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} K
                C : Type u₃
                inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                D : Type u₄
                inst✝ : CategoryTheory.Category.{v₄, u₄} D
                F : CategoryTheory.Functor J C
                X Y : CategoryTheory.Limits.Cocone F
                f : Quiver.Hom X Y
                j : Opposite J
                ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.op (X.op.π.app j)).unop (Y.op.π …
              -/
              dsimp
              /-
                case a
                J : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                K : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} K
                C : Type u₃
                inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                D : Type u₄
                inst✝ : CategoryTheory.Category.{v₄, u₄} D
                F : CategoryTheory.Functor J C
                X Y : CategoryTheory.Limits.Cocone F
                f : Quiver.Hom X Y
                j : Opposite J
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ι.app (Opposite.unop j)) f.hom) (Y …
              -/
              apply CoconeMorphism.w } }
              /-
                🎉 no goals
              -/
  inverse :=
    { obj := fun c => Cone.unop (unop c)
      map := fun {X} {Y} f =>
        { hom := f.unop.hom.unop
          w := fun j => by
            /-
              J : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} J
              K : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} K
              C : Type u₃
              inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
              D : Type u₄
              inst✝ : CategoryTheory.Category.{v₄, u₄} D
              F : CategoryTheory.Functor J C
              X Y : Opposite (CategoryTheory.Limits.Cone F.op)
              f : Quiver.Hom X Y
              j : J
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun c => (Opposite.unop c).unop) X …
            -/
            apply Quiver.Hom.op_inj
            /-
              case a
              J : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} J
              K : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} K
              C : Type u₃
              inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
              D : Type u₄
              inst✝ : CategoryTheory.Category.{v₄, u₄} D
              F : CategoryTheory.Functor J C
              X Y : Opposite (CategoryTheory.Limits.Cone F.op)
              f : Quiver.Hom X Y
              j : J
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun c => (Opposite.unop c).unop) X …
            -/
            dsimp
            /-
              case a
              J : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} J
              K : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} K
              C : Type u₃
              inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
              D : Type u₄
              inst✝ : CategoryTheory.Category.{v₄, u₄} D
              F : CategoryTheory.Functor J C
              X Y : Opposite (CategoryTheory.Limits.Cone F.op)
              f : Quiver.Hom X Y
              j : J
              ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop.hom ((Opposite.unop X).π.app { …
            -/
            apply ConeMorphism.w } }
            /-
              🎉 no goals
            -/
                                           /-
                                             J : Type u₁
                                             inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                             K : Type u₂
                                             inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                             C : Type u₃
                                             inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                             D : Type u₄
                                             inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                             F : CategoryTheory.Functor J C
                                             c : CategoryTheory.Limits.Cocone F
                                             ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor. …
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  unitIso := NatIso.ofComponents (fun c => Cocones.ext (Iso.refl _))
             /-
               🎉 no goals
             -/
  counitIso :=
    NatIso.ofComponents
      (fun c => by
        /-
          J : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} J
          K : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} K
          C : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
          D : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D
          F : CategoryTheory.Functor J C
          c : Opposite (CategoryTheory.Limits.Cone F.op)
          ⊢ CategoryTheory.Iso (({ obj := fun c => (Opposite.unop c).unop, map := fun {X …
        -/
        induction c
        /-
          case h
          J : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} J
          K : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} K
          C : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
          D : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D
          F : CategoryTheory.Functor J C
          X✝ : CategoryTheory.Limits.Cone F.op
          ⊢ CategoryTheory.Iso (({ obj := fun c => (Opposite.unop c).unop, map := fun {X …
        -/
        apply Iso.op
        /-
          case h.α
          J : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} J
          K : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} K
          C : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
          D : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D
          F : CategoryTheory.Functor J C
          X✝ : CategoryTheory.Limits.Cone F.op
          ⊢ CategoryTheory.Iso X✝ ({ obj := fun c => (Opposite.unop c).unop, map := fun  …
        -/
        exact Cones.ext (Iso.refl _))
        /-
          🎉 no goals
        -/
      fun {X} {Y} f =>
                                                    /-
                                                      J : Type u₁
                                                      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                                      K : Type u₂
                                                      inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                                      C : Type u₃
                                                      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                                      D : Type u₄
                                                      inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                                      F : CategoryTheory.Functor J C
                                                      X Y : Opposite (CategoryTheory.Limits.Cone F.op)
                                                      f : Quiver.Hom X Y
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (({ obj := fun c => (Opposite.unop c) …
                                                    -/
      Quiver.Hom.unop_inj (ConeMorphism.ext _ _ (by simp))
                                                    /-
                                                      🎉 no goals
                                                    -/
  functor_unitIso_comp c := by
    /-
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := fun c => { unop := c.op },  …
    -/
    apply Quiver.Hom.unop_inj
    /-
      case a
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := fun c => { unop := c.op },  …
    -/
    apply ConeMorphism.ext
    /-
      case a.w
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := fun c => { unop := c.op },  …
    -/
    dsimp
    /-
      case a.w
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      D : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} D
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id { u …
    -/
    apply comp_id
    /-
      🎉 no goals
    -/


attribute [simps] coconeEquivalenceOpConeOp


/-- Change a cocone on `F.leftOp : Jᵒᵖ ⥤ C` to a cocone on `F : J ⥤ Cᵒᵖ`. -/
@[simps!]
def coneOfCoconeLeftOp (c : Cocone F.leftOp) : Cone F where
  pt := op c.pt
  π := NatTrans.removeLeftOp c.ι


/-- Change a cone on `F : J ⥤ Cᵒᵖ` to a cocone on `F.leftOp : Jᵒᵖ ⥤ C`. -/
@[simps!]
def coconeLeftOpOfCone (c : Cone F) : Cocone F.leftOp where
  pt := unop c.pt
  ι := NatTrans.leftOp c.π

/- When trying use `@[simps]` to generate the `ι_app` field of this definition, `@[simps]` tries to
  reduce the RHS using `expr.dsimp` and `expr.simp`, but for some reason the expression is not
  being simplified properly. -/

/-- Change a cone on `F.leftOp : Jᵒᵖ ⥤ C` to a cocone on `F : J ⥤ Cᵒᵖ`. -/
@[simps pt]
def coconeOfConeLeftOp (c : Cone F.leftOp) : Cocone F where
  pt := op c.pt
  ι := NatTrans.removeLeftOp c.π


@[simp]
theorem coconeOfConeLeftOp_ι_app (c : Cone F.leftOp) (j) :
    (coconeOfConeLeftOp c).ι.app j = (c.π.app (op j)).op := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J (Opposite C)
    c : CategoryTheory.Limits.Cone F.leftOp
    j : J
    ⊢ Eq ((CategoryTheory.Limits.coconeOfConeLeftOp c).ι.app j) (c.π.app { unop := …
  -/
  dsimp only [coconeOfConeLeftOp]
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J (Opposite C)
    c : CategoryTheory.Limits.Cone F.leftOp
    j : J
    ⊢ Eq ((CategoryTheory.NatTrans.removeLeftOp c.π).app j) (c.π.app { unop := j } …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Change a cocone on `F : J ⥤ Cᵒᵖ` to a cone on `F.leftOp : Jᵒᵖ ⥤ C`. -/
@[simps!]
def coneLeftOpOfCocone (c : Cocone F) : Cone F.leftOp where
  pt := unop c.pt
  π := NatTrans.leftOp c.ι


/-- Change a cocone on `F.rightOp : J ⥤ Cᵒᵖ` to a cone on `F : Jᵒᵖ ⥤ C`. -/
@[simps]
def coneOfCoconeRightOp (c : Cocone F.rightOp) : Cone F where
  pt := unop c.pt
  π := NatTrans.removeRightOp c.ι


/-- Change a cone on `F : Jᵒᵖ ⥤ C` to a cocone on `F.rightOp : Jᵒᵖ ⥤ C`. -/
@[simps]
def coconeRightOpOfCone (c : Cone F) : Cocone F.rightOp where
  pt := op c.pt
  ι := NatTrans.rightOp c.π


/-- Change a cone on `F.rightOp : J ⥤ Cᵒᵖ` to a cocone on `F : Jᵒᵖ ⥤ C`. -/
@[simps]
def coconeOfConeRightOp (c : Cone F.rightOp) : Cocone F where
  pt := unop c.pt
  ι := NatTrans.removeRightOp c.π


/-- Change a cocone on `F : Jᵒᵖ ⥤ C` to a cone on `F.rightOp : J ⥤ Cᵒᵖ`. -/
@[simps]
def coneRightOpOfCocone (c : Cocone F) : Cone F.rightOp where
  pt := op c.pt
  π := NatTrans.rightOp c.ι


/-- Change a cocone on `F.unop : J ⥤ C` into a cone on `F : Jᵒᵖ ⥤ Cᵒᵖ`. -/
@[simps]
def coneOfCoconeUnop (c : Cocone F.unop) : Cone F where
  pt := op c.pt
  π := NatTrans.removeUnop c.ι


/-- Change a cone on `F : Jᵒᵖ ⥤ Cᵒᵖ` into a cocone on `F.unop : J ⥤ C`. -/
@[simps]
def coconeUnopOfCone (c : Cone F) : Cocone F.unop where
  pt := unop c.pt
  ι := NatTrans.unop c.π


/-- Change a cone on `F.unop : J ⥤ C` into a cocone on `F : Jᵒᵖ ⥤ Cᵒᵖ`. -/
@[simps]
def coconeOfConeUnop (c : Cone F.unop) : Cocone F where
  pt := op c.pt
  ι := NatTrans.removeUnop c.π


/-- Change a cocone on `F : Jᵒᵖ ⥤ Cᵒᵖ` into a cone on `F.unop : J ⥤ C`. -/
@[simps]
def coneUnopOfCocone (c : Cocone F) : Cone F.unop where
  pt := unop c.pt
  π := NatTrans.unop c.ι


/-- The opposite cocone of the image of a cone is the image of the opposite cocone. -/
-- Porting note: removed @[simps (config := { rhsMd := semireducible })] and replaced with
@[simps!]
def mapConeOp (t : Cone F) : (mapCone G t).op ≅ mapCocone G.op t.op :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F : CategoryTheory.Functor J C
    G : CategoryTheory.Functor C D
    t : CategoryTheory.Limits.Cone F
    ⊢ ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp ((G.mapCone t).op …
  -/
  Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- The opposite cone of the image of a cocone is the image of the opposite cone. -/
-- Porting note: removed @[simps (config := { rhsMd := semireducible })] and replaced with
@[simps!]
def mapCoconeOp {t : Cocone F} : (mapCocone G t).op ≅ mapCone G.op t.op :=
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F : CategoryTheory.Functor J C
    G : CategoryTheory.Functor C D
    t : CategoryTheory.Limits.Cocone F
    ⊢ ∀ (j : Opposite J), Eq ((G.mapCocone t).op.π.app j) (CategoryTheory.Category …
  -/
  Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


