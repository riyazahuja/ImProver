/-- An oplax functor `F` between bicategories `B` and `C` consists of a function between objects
`F.obj`, a function between 1-morphisms `F.map`, and a function between 2-morphisms `F.map₂`.

Unlike functors between categories, `F.map` do not need to strictly commute with the composition,
and do not need to strictly preserve the identity. Instead, there are specified 2-morphisms
`F.map (𝟙 a) ⟶ 𝟙 (F.obj a)` and `F.map (f ≫ g) ⟶ F.map f ≫ F.map g`.

`F.map₂` strictly commute with compositions and preserve the identity. They also preserve the
associator, the left unitor, and the right unitor modulo some adjustments of domains and codomains
of 2-morphisms.
-/
structure OplaxFunctor (B : Type u₁) [Bicategory.{w₁, v₁} B] (C : Type u₂)
  [Bicategory.{w₂, v₂} C] extends PrelaxFunctor B C where
  /-- The 2-morphism underlying the oplax unity constraint. -/
  mapId (a : B) : map (𝟙 a) ⟶ 𝟙 (obj a)
  /-- The 2-morphism underlying the oplax functoriality constraint. -/
  mapComp {a b c : B} (f : a ⟶ b) (g : b ⟶ c) : map (f ≫ g) ⟶ map f ≫ map g
  /-- Naturality of the oplax functoriality constraint, on the left. -/
  mapComp_naturality_left :
    ∀ {a b c : B} {f f' : a ⟶ b} (η : f ⟶ f') (g : b ⟶ c),
      map₂ (η ▷ g) ≫ mapComp f' g = mapComp f g ≫ map₂ η ▷ map g := by
    aesop_cat
  /-- Naturality of the lax functoriality constraight, on the right. -/
  mapComp_naturality_right :
    ∀ {a b c : B} (f : a ⟶ b) {g g' : b ⟶ c} (η : g ⟶ g'),
      map₂ (f ◁ η) ≫ mapComp f g' = mapComp f g ≫ map f ◁ map₂ η := by
    aesop_cat
  /-- Oplax associativity. -/
  map₂_associator :
    ∀ {a b c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d),
      map₂ (α_ f g h).hom ≫ mapComp f (g ≫ h) ≫ map f ◁ mapComp g h =
    mapComp (f ≫ g) h ≫ mapComp f g ▷ map h ≫ (α_ (map f) (map g) (map h)).hom := by
    aesop_cat
  /-- Oplax left unity. -/
  map₂_leftUnitor :
    ∀ {a b : B} (f : a ⟶ b),
      map₂ (λ_ f).hom = mapComp (𝟙 a) f ≫ mapId a ▷ map f ≫ (λ_ (map f)).hom := by
    aesop_cat
  /-- Oplax right unity. -/
  map₂_rightUnitor :
    ∀ {a b : B} (f : a ⟶ b),
      map₂ (ρ_ f).hom = mapComp f (𝟙 b) ≫ map f ◁ mapId b ≫ (ρ_ (map f)).hom := by
    aesop_cat


attribute [reassoc (attr := simp), to_app (attr := simp)]
  mapComp_naturality_left mapComp_naturality_right map₂_associator

attribute [simp, reassoc, to_app] map₂_leftUnitor map₂_rightUnitor


@[reassoc, to_app]
lemma mapComp_assoc_right {a b c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) :
    F.mapComp f (g ≫ h) ≫ F.map f ◁ F.mapComp g h = F.map₂ (α_ f g h).inv ≫
    F.mapComp (f ≫ g) h ≫ F.mapComp f g ▷ F.map h ≫
    (α_ (F.map f) (F.map g) (F.map h)).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.OplaxFunctor B C
    a b c d : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.mapComp f (CategoryTheory.Category …
  -/
  rw [← F.map₂_associator, ← F.map₂_comp_assoc]
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.OplaxFunctor B C
    a b c d : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.mapComp f (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc, to_app]
lemma mapComp_assoc_left {a b c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) :
    F.mapComp (f ≫ g) h ≫ F.mapComp f g ▷ F.map h =
    F.map₂ (α_ f g h).hom ≫ F.mapComp f (g ≫ h) ≫ F.map f ◁ F.mapComp g h
    ≫ (α_ (F.map f) (F.map g) (F.map h)).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.OplaxFunctor B C
    a b c d : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.mapComp (CategoryTheory.CategorySt …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The identity oplax functor. -/
@[simps]
def id (B : Type u₁) [Bicategory.{w₁, v₁} B] : OplaxFunctor B B where
  toPrelaxFunctor := PrelaxFunctor.id B
  mapId := fun a => 𝟙 (𝟙 a)
  mapComp := fun f g => 𝟙 (f ≫ g)


instance : Inhabited (OplaxFunctor B B) :=
  ⟨id B⟩


/-- Composition of oplax functors. -/
--@[simps]
def comp (F : OplaxFunctor B C) (G : OplaxFunctor C D) : OplaxFunctor B D where
  toPrelaxFunctor := F.toPrelaxFunctor.comp G.toPrelaxFunctor
  mapId := fun a => (G.mapFunctor _ _).map (F.mapId a) ≫ G.mapId (F.obj a)
  mapComp := fun f g => (G.mapFunctor _ _).map (F.mapComp f g) ≫ G.mapComp (F.map f) (F.map g)
  mapComp_naturality_left := fun η g => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ F : CategoryTheory.OplaxFunctor B C
      G : CategoryTheory.OplaxFunctor C D
      a✝ b✝ c✝ : B
      f✝ f'✝ : Quiver.Hom a✝ b✝
      η : Quiver.Hom f✝ f'✝
      g : Quiver.Hom b✝ c✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp G.toPrelaxFunctor).map₂ (Cat …
    -/
    dsimp
    rw [← G.map₂_comp_assoc, mapComp_naturality_left, G.map₂_comp_assoc, mapComp_naturality_left,
      assoc]
  mapComp_naturality_right := fun η => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ F : CategoryTheory.OplaxFunctor B C
      G : CategoryTheory.OplaxFunctor C D
      a✝ b✝ c✝ : B
      η : Quiver.Hom a✝ b✝
      ⊢ ∀ {g g' : Quiver.Hom b✝ c✝} (η_1 : Quiver.Hom g g'), Eq (CategoryTheory.Cate …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ F : CategoryTheory.OplaxFunctor B C
      G : CategoryTheory.OplaxFunctor C D
      a✝ b✝ c✝ : B
      η : Quiver.Hom a✝ b✝
      ⊢ ∀ {g g' : Quiver.Hom b✝ c✝} (η_1 : Quiver.Hom g g'), Eq (CategoryTheory.Cate …
    -/
    intros
    rw [← G.map₂_comp_assoc, mapComp_naturality_right, G.map₂_comp_assoc,
      mapComp_naturality_right, assoc]
  map₂_associator := fun f g h => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ F : CategoryTheory.OplaxFunctor B C
      G : CategoryTheory.OplaxFunctor C D
      a✝ b✝ c✝ d✝ : B
      f : Quiver.Hom a✝ b✝
      g : Quiver.Hom b✝ c✝
      h : Quiver.Hom c✝ d✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp G.toPrelaxFunctor).map₂ (Cat …
    -/
    dsimp
    simp only [map₂_associator, ← PrelaxFunctor.map₂_comp_assoc, ← mapComp_naturality_right_assoc,
      Bicategory.whiskerLeft_comp, assoc]
    simp only [map₂_associator, PrelaxFunctor.map₂_comp, mapComp_naturality_left_assoc,
      comp_whiskerRight, assoc]
  map₂_leftUnitor := fun f => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ F : CategoryTheory.OplaxFunctor B C
      G : CategoryTheory.OplaxFunctor C D
      a✝ b✝ : B
      f : Quiver.Hom a✝ b✝
      ⊢ Eq ((F.comp G.toPrelaxFunctor).map₂ (CategoryTheory.Bicategory.leftUnitor f) …
    -/
    dsimp
    simp only [map₂_leftUnitor, PrelaxFunctor.map₂_comp, mapComp_naturality_left_assoc,
      comp_whiskerRight, assoc]
  map₂_rightUnitor := fun f => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ F : CategoryTheory.OplaxFunctor B C
      G : CategoryTheory.OplaxFunctor C D
      a✝ b✝ : B
      f : Quiver.Hom a✝ b✝
      ⊢ Eq ((F.comp G.toPrelaxFunctor).map₂ (CategoryTheory.Bicategory.rightUnitor f …
    -/
    dsimp
    simp only [map₂_rightUnitor, PrelaxFunctor.map₂_comp, mapComp_naturality_right_assoc,
      Bicategory.whiskerLeft_comp, assoc]


/-- A structure on an oplax functor that promotes an oplax functor to a pseudofunctor.

See `Pseudofunctor.mkOfOplax`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet
-- @[nolint has_nonempty_instance]
structure PseudoCore (F : OplaxFunctor B C) where
  /-- The isomorphism giving rise to the oplax unity constraint -/
  mapIdIso (a : B) : F.map (𝟙 a) ≅ 𝟙 (F.obj a)
  /-- The isomorphism giving rise to the oplax functoriality constraint -/
  mapCompIso {a b c : B} (f : a ⟶ b) (g : b ⟶ c) : F.map (f ≫ g) ≅ F.map f ≫ F.map g
  /-- `mapIdIso` gives rise to the oplax unity constraint -/
  mapIdIso_hom : ∀ {a : B}, (mapIdIso a).hom = F.mapId a := by aesop_cat
  /-- `mapCompIso` gives rise to the oplax functoriality constraint -/
  mapCompIso_hom :
    ∀ {a b c : B} (f : a ⟶ b) (g : b ⟶ c), (mapCompIso f g).hom = F.mapComp f g := by aesop_cat


