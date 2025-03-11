/-- The type of morphisms of a disjoint union of categories: for `X : C i` and `Y : C j`, a morphism
`(i, X) ⟶ (j, Y)` if `i = j` is just a morphism `X ⟶ Y`, and if `i ≠ j` there are no such morphisms.
-/
inductive SigmaHom : (Σi, C i) → (Σi, C i) → Type max w₁ v₁ u₁
  | mk : ∀ {i : I} {X Y : C i}, (X ⟶ Y) → SigmaHom ⟨i, X⟩ ⟨i, Y⟩


/-- The identity morphism on an object. -/
def id : ∀ X : Σi, C i, SigmaHom X X
  | ⟨_, _⟩ => mk (𝟙 _)
-- Porting note: reordered universes


instance (X : Σi, C i) : Inhabited (SigmaHom X X) :=
  ⟨id X⟩


/-- Composition of sigma homomorphisms. -/
def comp : ∀ {X Y Z : Σi, C i}, SigmaHom X Y → SigmaHom Y Z → SigmaHom X Z
  | _, _, _, mk f, mk g => mk (f ≫ g)
-- Porting note: reordered universes


instance : CategoryStruct (Σi, C i) where
  Hom := SigmaHom
  id := id
  comp f g := comp f g


@[simp]
lemma comp_def (i : I) (X Y Z : C i) (f : X ⟶ Y) (g : Y ⟶ Z) : comp (mk f) (mk g) = mk (f ≫ g) :=
  rfl


lemma assoc : ∀ {X Y Z W : Σi, C i} (f : X ⟶ Y) (g : Y ⟶ Z) (h : Z ⟶ W), (f ≫ g) ≫ h = f ≫ g ≫ h
  | _, _, _, _, mk _, mk _, mk _ => congr_arg mk (Category.assoc _ _ _)


lemma id_comp : ∀ {X Y : Σi, C i} (f : X ⟶ Y), 𝟙 X ≫ f = f
  | _, _, mk _ => congr_arg mk (Category.id_comp _)


lemma comp_id : ∀ {X Y : Σi, C i} (f : X ⟶ Y), f ≫ 𝟙 Y = f
  | _, _, mk _ => congr_arg mk (Category.comp_id _)


instance sigma : Category (Σi, C i) where
  id_comp := SigmaHom.id_comp
  comp_id := SigmaHom.comp_id
  assoc := SigmaHom.assoc


/-- The inclusion functor into the disjoint union of categories. -/
@[simps map]
def incl (i : I) : C i ⥤ Σi, C i where
  obj X := ⟨i, X⟩
  map := SigmaHom.mk


@[simp]
lemma incl_obj {i : I} (X : C i) : (incl i).obj X = ⟨i, X⟩ :=
  rfl


instance (i : I) : Functor.Full (incl i : C i ⥤ Σi, C i) where
  map_surjective := fun ⟨f⟩ => ⟨f, rfl⟩


instance (i : I) : Functor.Faithful (incl i : C i ⥤ Σi, C i) where
                                  /-
                                    I : Type w₁
                                    C : I → Type u₁
                                    inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
                                    i : I
                                    x✝³ x✝² : C i
                                    x✝¹ x✝ : Quiver.Hom x✝³ x✝²
                                    h : Eq ((CategoryTheory.Sigma.incl i).map x✝¹) ((CategoryTheory.Sigma.incl i). …
                                    ⊢ Eq x✝¹ x✝
                                  -/
  map_injective {_ _ _ _} h := by injection h
                                  /-
                                    🎉 no goals
                                  -/


/--
To build a natural transformation over the sigma category, it suffices to specify it restricted to
each subcategory.
-/
def natTrans {F G : (Σi, C i) ⥤ D} (h : ∀ i : I, incl i ⋙ F ⟶ incl i ⋙ G) : F ⟶ G where
  app := fun ⟨j, X⟩ => (h j).app X
  naturality := by
    /-
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F✝ : (i : I) → CategoryTheory.Functor (C i) D
      F G : CategoryTheory.Functor (Sigma fun i => C i) D
      h : (i : I) → Quiver.Hom ((CategoryTheory.Sigma.incl i).comp F) ((CategoryTheo …
      ⊢ ∀ ⦃X Y : Sigma fun i => C i⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.Catego …
    -/
    rintro ⟨j, X⟩ ⟨_, _⟩ ⟨f⟩
    /-
      case mk.mk.mk
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F✝ : (i : I) → CategoryTheory.Functor (C i) D
      F G : CategoryTheory.Functor (Sigma fun i => C i) D
      h : (i : I) → Quiver.Hom ((CategoryTheory.Sigma.incl i).comp F) ((CategoryTheo …
      j : I
      X Y✝ : C j
      f : Quiver.Hom X Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Sigma.SigmaHom …
    -/
    apply (h j).naturality
    /-
      🎉 no goals
    -/


@[simp]
lemma natTrans_app {F G : (Σi, C i) ⥤ D} (h : ∀ i : I, incl i ⋙ F ⟶ incl i ⋙ G) (i : I)
    (X : C i) : (natTrans h).app ⟨i, X⟩ = (h i).app X :=
  rfl


/-- (Implementation). An auxiliary definition to build the functor `desc`. -/
def descMap : ∀ X Y : Σi, C i, (X ⟶ Y) → ((F X.1).obj X.2 ⟶ (F Y.1).obj Y.2)
  | _, _, SigmaHom.mk g => (F _).map g
-- Porting note: reordered universes


/-- Given a collection of functors `F i : C i ⥤ D`, we can produce a functor `(Σ i, C i) ⥤ D`.

The produced functor `desc F` satisfies: `incl i ⋙ desc F ≅ F i`, i.e. restricted to just the
subcategory `C i`, `desc F` agrees with `F i`, and it is unique (up to natural isomorphism) with
this property.

This witnesses that the sigma-type is the coproduct in Cat.
-/
@[simps obj]
def desc : (Σi, C i) ⥤ D where
  obj X := (F X.1).obj X.2
  map g := descMap F _ _ g
  map_id := by
    /-
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : (i : I) → CategoryTheory.Functor (C i) D
      ⊢ ∀ (X : Sigma fun i => C i), Eq ({ obj := fun X => (F X.fst).obj X.snd, map : …
    -/
    rintro ⟨i, X⟩
    /-
      case mk
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : (i : I) → CategoryTheory.Functor (C i) D
      i : I
      X : C i
      ⊢ Eq ({ obj := fun X => (F X.fst).obj X.snd, map := fun {X Y} g => CategoryThe …
    -/
    apply (F i).map_id
    /-
      🎉 no goals
    -/
  map_comp := by
    /-
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : (i : I) → CategoryTheory.Functor (C i) D
      ⊢ ∀ {X Y Z : Sigma fun i => C i} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq …
    -/
    rintro ⟨i, X⟩ ⟨_, Y⟩ ⟨_, Z⟩ ⟨f⟩ ⟨g⟩
    /-
      case mk.mk.mk.mk.mk
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : (i : I) → CategoryTheory.Functor (C i) D
      i : I
      X Y✝¹ : C i
      f : Quiver.Hom X Y✝¹
      Y✝ : C i
      g : Quiver.Hom Y✝¹ Y✝
      ⊢ Eq ({ obj := fun X => (F X.fst).obj X.snd, map := fun {X Y} g => CategoryThe …
    -/
    apply (F i).map_comp
    /-
      🎉 no goals
    -/


@[simp]
lemma desc_map_mk {i : I} (X Y : C i) (f : X ⟶ Y) : (desc F).map (SigmaHom.mk f) = (F i).map f :=
  rfl

-- We hand-generate the simp lemmas about this since they come out cleaner.

/-- This shows that when `desc F` is restricted to just the subcategory `C i`, `desc F` agrees with
`F i`.
-/
def inclDesc (i : I) : incl i ⋙ desc F ≅ F i :=
  /-
    I : Type w₁
    C : I → Type u₁
    inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : (i : I) → CategoryTheory.Functor (C i) D
    i : I
    ⊢ ∀ {X Y : C i} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


@[simp]
lemma inclDesc_hom_app (i : I) (X : C i) : (inclDesc F i).hom.app X = 𝟙 ((F i).obj X) :=
  rfl


@[simp]
lemma inclDesc_inv_app (i : I) (X : C i) : (inclDesc F i).inv.app X = 𝟙 ((F i).obj X) :=
  rfl


/-- If `q` when restricted to each subcategory `C i` agrees with `F i`, then `q` is isomorphic to
`desc F`.
-/
def descUniq (q : (Σi, C i) ⥤ D) (h : ∀ i, incl i ⋙ q ≅ F i) : q ≅ desc F :=
  NatIso.ofComponents (fun ⟨i, X⟩ => (h i).app X) <| by
    /-
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : (i : I) → CategoryTheory.Functor (C i) D
      q : CategoryTheory.Functor (Sigma fun i => C i) D
      h : (i : I) → CategoryTheory.Iso ((CategoryTheory.Sigma.incl i).comp q) (F i)
      ⊢ ∀ {X Y : Sigma fun i => C i} (f : Quiver.Hom X Y), Eq (CategoryTheory.Catego …
    -/
    rintro ⟨i, X⟩ ⟨_, _⟩ ⟨f⟩
    /-
      case mk.mk.mk
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : (i : I) → CategoryTheory.Functor (C i) D
      q : CategoryTheory.Functor (Sigma fun i => C i) D
      h : (i : I) → CategoryTheory.Iso ((CategoryTheory.Sigma.incl i).comp q) (F i)
      i : I
      X Y✝ : C i
      f : Quiver.Hom X Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (q.map (CategoryTheory.Sigma.SigmaHom …
    -/
    apply (h i).hom.naturality f
    /-
      🎉 no goals
    -/


@[simp]
lemma descUniq_hom_app (q : (Σi, C i) ⥤ D) (h : ∀ i, incl i ⋙ q ≅ F i) (i : I) (X : C i) :
    (descUniq F q h).hom.app ⟨i, X⟩ = (h i).hom.app X :=
  rfl


@[simp]
lemma descUniq_inv_app (q : (Σi, C i) ⥤ D) (h : ∀ i, incl i ⋙ q ≅ F i) (i : I) (X : C i) :
    (descUniq F q h).inv.app ⟨i, X⟩ = (h i).inv.app X :=
  rfl


/--
If `q₁` and `q₂` when restricted to each subcategory `C i` agree, then `q₁` and `q₂` are isomorphic.
-/
@[simps]
def natIso {q₁ q₂ : (Σi, C i) ⥤ D} (h : ∀ i, incl i ⋙ q₁ ≅ incl i ⋙ q₂) : q₁ ≅ q₂ where
  hom := natTrans fun i => (h i).hom
  inv := natTrans fun i => (h i).inv


/-- A function `J → I` induces a functor `Σ j, C (g j) ⥤ Σ i, C i`. -/
def map : (Σj : J, C (g j)) ⥤ Σi : I, C i :=
  desc fun j => incl (g j)


@[simp]
lemma map_obj (j : J) (X : C (g j)) : (Sigma.map C g).obj ⟨j, X⟩ = ⟨g j, X⟩ :=
  rfl


@[simp]
lemma map_map {j : J} {X Y : C (g j)} (f : X ⟶ Y) :
    (Sigma.map C g).map (SigmaHom.mk f) = SigmaHom.mk f :=
  rfl


/-- The functor `Sigma.map C g` restricted to the subcategory `C j` acts as the inclusion of `g j`.
-/
@[simps!]
def inclCompMap (j : J) : incl j ⋙ map C g ≅ incl (g j) :=
  Iso.refl _


/-- The functor `Sigma.map` applied to the identity function is just the identity functor. -/
@[simps!]
def mapId : map C (id : I → I) ≅ 𝟭 (Σi, C i) :=
                  /-
                    I : Type w₁
                    C : I → Type u₁
                    inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
                    J : Type w₂
                    g : J → I
                    i : I
                    ⊢ ∀ {X Y : C (id i)} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.c …
                  -/
  natIso fun i => NatIso.ofComponents fun _ => Iso.refl _
                  /-
                    🎉 no goals
                  -/


/-- The functor `Sigma.map` applied to a composition is a composition of functors. -/
@[simps!]
def mapComp (f : K → J) (g : J → I) : map (fun x ↦ C (g x)) f ⋙ (map C g : _) ≅ map C (g ∘ f) :=
  (descUniq _ _) fun k =>
    (isoWhiskerRight (inclCompMap (fun i => C (g i)) f k) (map C g : _) : _) ≪≫ inclCompMap _ _ _


/-- Assemble an `I`-indexed family of functors into a functor between the sigma types.
-/
def sigma (F : ∀ i, C i ⥤ D i) : (Σi, C i) ⥤ Σi, D i :=
  desc fun i => F i ⋙ incl i


/-- Assemble an `I`-indexed family of natural transformations into a single natural transformation.
-/
def sigma (α : ∀ i, F i ⟶ G i) : Functor.sigma F ⟶ Functor.sigma G where
  app f := SigmaHom.mk ((α f.1).app _)
  naturality := by
    /-
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : I → Type u₁
      inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (D i)
      F G : (i : I) → CategoryTheory.Functor (C i) (D i)
      α : (i : I) → Quiver.Hom (F i) (G i)
      ⊢ ∀ ⦃X Y : Sigma fun i => C i⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.Catego …
    -/
    rintro ⟨i, X⟩ ⟨_, _⟩ ⟨f⟩
    /-
      case mk.mk.mk
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : I → Type u₁
      inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (D i)
      F G : (i : I) → CategoryTheory.Functor (C i) (D i)
      α : (i : I) → Quiver.Hom (F i) (G i)
      i : I
      X Y✝ : C i
      f : Quiver.Hom X Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Sigma.Functor.sigma  …
    -/
    change SigmaHom.mk _ = SigmaHom.mk _
    /-
      case mk.mk.mk
      I : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : I → Type u₁
      inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (D i)
      F G : (i : I) → CategoryTheory.Functor (C i) (D i)
      α : (i : I) → Quiver.Hom (F i) (G i)
      i : I
      X Y✝ : C i
      f : Quiver.Hom X Y✝
      ⊢ Eq (CategoryTheory.Sigma.SigmaHom.mk (CategoryTheory.CategoryStruct.comp ((F …
    -/
    rw [(α i).naturality]
    /-
      🎉 no goals
    -/


