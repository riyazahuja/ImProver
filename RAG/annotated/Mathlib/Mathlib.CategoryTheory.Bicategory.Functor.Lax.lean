/-- A lax functor `F` between bicategories `B` and `C` consists of a function between objects
`F.obj`, a function between 1-morphisms `F.map`, and a function between 2-morphisms `F.map₂`.

Unlike functors between categories, `F.map` do not need to strictly commute with the composition,
and do not need to strictly preserve the identity. Instead, there are specified 2-morphisms
`𝟙 (F.obj a) ⟶ F.map (𝟙 a)` and `F.map f ≫ F.map g ⟶ F.map (f ≫ g)`.

`F.map₂` strictly commute with compositions and preserve the identity. They also preserve the
associator, the left unitor, and the right unitor modulo some adjustments of domains and codomains
of 2-morphisms.
-/
structure LaxFunctor (B: Type u₁) [Bicategory.{w₁, v₁} B] (C : Type u₂) [Bicategory.{w₂, v₂} C]
    extends PrelaxFunctor B C where
  /-- The 2-morphism underlying the lax unity constraint. -/
  mapId (a : B) : 𝟙 (obj a) ⟶ map (𝟙 a)
  /-- The 2-morphism underlying the lax functoriality constraint. -/
  mapComp {a b c : B} (f : a ⟶ b) (g : b ⟶ c) : map f ≫ map g ⟶ map (f ≫ g)
  /-- Naturality of the lax functoriality constraint, on the left. -/
  mapComp_naturality_left :
    ∀ {a b c : B} {f f' : a ⟶ b} (η : f ⟶ f') (g : b ⟶ c),
      mapComp f g ≫ map₂ (η ▷ g) = map₂ η ▷ map g ≫ mapComp f' g:= by aesop_cat
  /-- Naturality of the lax functoriality constraint, on the right. -/
  mapComp_naturality_right :
    ∀ {a b c : B} (f : a ⟶ b) {g g' : b ⟶ c} (η : g ⟶ g'),
     mapComp f g ≫ map₂ (f ◁ η) = map f ◁ map₂ η ≫ mapComp f g' := by aesop_cat
  /-- Lax associativity. -/
  map₂_associator :
    ∀ {a b c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d),
      mapComp f g ▷ map h ≫ mapComp (f ≫ g) h ≫ map₂ (α_ f g h).hom =
      (α_ (map f) (map g) (map h)).hom ≫ map f ◁ mapComp g h ≫ mapComp f (g ≫ h) := by aesop_cat
  /-- Lax left unity. -/
  map₂_leftUnitor :
    ∀ {a b : B} (f : a ⟶ b),
      map₂ (λ_ f).inv = (λ_ (map f)).inv ≫ mapId a ▷ map f ≫ mapComp (𝟙 a) f := by aesop_cat
  /-- Lax right unity. -/
  map₂_rightUnitor :
    ∀ {a b : B} (f : a ⟶ b),
      map₂ (ρ_ f).inv = (ρ_ (map f)).inv ≫ map f ◁ mapId b ≫ mapComp f (𝟙 b) := by aesop_cat


attribute [reassoc (attr := simp), to_app (attr := simp)]
  mapComp_naturality_left mapComp_naturality_right map₂_associator

attribute [simp, reassoc, to_app] map₂_leftUnitor map₂_rightUnitor


@[reassoc, to_app]
lemma mapComp_assoc_left {a b c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) :
    F.mapComp f g ▷ F.map h ≫ F.mapComp (f ≫ g) h = (α_ (F.map f) (F.map g) (F.map h)).hom ≫
      F.map f ◁ F.mapComp g h ≫ F.mapComp f (g ≫ h) ≫ F.map₂ (α_ f g h).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.LaxFunctor B C
    a b c d : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  rw [← F.map₂_associator_assoc, ← F.map₂_comp]
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.LaxFunctor B C
    a b c d : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  simp only [Iso.hom_inv_id, PrelaxFunctor.map₂_id, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc, to_app]
lemma mapComp_assoc_right {a b c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) :
    F.map f ◁ F.mapComp g h ≫ F.mapComp f (g ≫ h) =
      (α_ (F.map f) (F.map g) (F.map h)).inv ≫ F.mapComp f g ▷ F.map h ≫
        F.mapComp (f ≫ g) h ≫ F.map₂ (α_ f g h).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.LaxFunctor B C
    a b c d : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  simp only [map₂_associator, Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


@[reassoc, to_app]
lemma map₂_leftUnitor_hom {a b : B} (f : a ⟶ b) :
    (λ_ (F.map f)).hom = F.mapId a ▷ F.map f ≫ F.mapComp (𝟙 a) f ≫ F.map₂ (λ_ f).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.LaxFunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (CategoryTheory.Bicategory.leftUnitor (F.map f)).hom (CategoryTheory.Cate …
  -/
  rw [← PrelaxFunctor.map₂Iso_hom, ← assoc, ← Iso.comp_inv_eq, ← Iso.eq_inv_comp]
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.LaxFunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (F.map₂Iso (CategoryTheory.Bicategory.leftUnitor f)).inv (CategoryTheory. …
  -/
  simp only [Functor.mapIso_inv, PrelaxFunctor.mapFunctor_map, map₂_leftUnitor]
  /-
    🎉 no goals
  -/


@[reassoc, to_app]
lemma map₂_rightUnitor_hom {a b : B} (f : a ⟶ b) :
    (ρ_ (F.map f)).hom = F.map f ◁ F.mapId b ≫ F.mapComp f (𝟙 b) ≫ F.map₂ (ρ_ f).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.LaxFunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (CategoryTheory.Bicategory.rightUnitor (F.map f)).hom (CategoryTheory.Cat …
  -/
  rw [← PrelaxFunctor.map₂Iso_hom, ← assoc, ← Iso.comp_inv_eq, ← Iso.eq_inv_comp]
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.LaxFunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (F.map₂Iso (CategoryTheory.Bicategory.rightUnitor f)).inv (CategoryTheory …
  -/
  simp only [Functor.mapIso_inv, PrelaxFunctor.mapFunctor_map, map₂_rightUnitor]
  /-
    🎉 no goals
  -/


/-- The identity lax functor. -/
@[simps]
def id (B : Type u₁) [Bicategory.{w₁, v₁} B] : LaxFunctor B B where
  toPrelaxFunctor := PrelaxFunctor.id B
  mapId := fun a => 𝟙 (𝟙 a)
  mapComp := fun f g => 𝟙 (f ≫ g)


instance : Inhabited (LaxFunctor B B) :=
  ⟨id B⟩


/-- Composition of lax functors. -/
@[simps]
def comp {D : Type u₃} [Bicategory.{w₃, v₃} D] (F : LaxFunctor B C) (G : LaxFunctor C D) :
    LaxFunctor B D where
  toPrelaxFunctor := PrelaxFunctor.comp F.toPrelaxFunctor G.toPrelaxFunctor
  mapId := fun a => G.mapId (F.obj a) ≫ G.map₂ (F.mapId a)
  mapComp := fun f g => G.mapComp (F.map f) (F.map g) ≫ G.map₂ (F.mapComp f g)
  mapComp_naturality_left := fun η g => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      F✝ : CategoryTheory.LaxFunctor B C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.LaxFunctor B C
      G : CategoryTheory.LaxFunctor C D
      a✝ b✝ c✝ : B
      f✝ f'✝ : Quiver.Hom a✝ b✝
      η : Quiver.Hom f✝ f'✝
      g : Quiver.Hom b✝ c✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {a b c} f g => CategoryTheory.C …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      F✝ : CategoryTheory.LaxFunctor B C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.LaxFunctor B C
      G : CategoryTheory.LaxFunctor C D
      a✝ b✝ c✝ : B
      f✝ f'✝ : Quiver.Hom a✝ b✝
      η : Quiver.Hom f✝ f'✝
      g : Quiver.Hom b✝ c✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, ← G.map₂_comp, mapComp_naturality_left, G.map₂_comp, mapComp_naturality_left_assoc]
    /-
      🎉 no goals
    -/
  mapComp_naturality_right := fun f _ _ η => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      F✝ : CategoryTheory.LaxFunctor B C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.LaxFunctor B C
      G : CategoryTheory.LaxFunctor C D
      a✝ b✝ c✝ : B
      f : Quiver.Hom a✝ b✝
      x✝¹ x✝ : Quiver.Hom b✝ c✝
      η : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {a b c} f g => CategoryTheory.C …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      F✝ : CategoryTheory.LaxFunctor B C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.LaxFunctor B C
      G : CategoryTheory.LaxFunctor C D
      a✝ b✝ c✝ : B
      f : Quiver.Hom a✝ b✝
      x✝¹ x✝ : Quiver.Hom b✝ c✝
      η : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, ← G.map₂_comp, mapComp_naturality_right, G.map₂_comp, mapComp_naturality_right_assoc]
    /-
      🎉 no goals
    -/
  map₂_associator := fun f g h => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      F✝ : CategoryTheory.LaxFunctor B C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.LaxFunctor B C
      G : CategoryTheory.LaxFunctor C D
      a✝ b✝ c✝ d✝ : B
      f : Quiver.Hom a✝ b✝
      g : Quiver.Hom b✝ c✝
      h : Quiver.Hom c✝ d✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
    -/
    dsimp
    slice_rhs 1 3 =>
      rw [Bicategory.whiskerLeft_comp, assoc, ← mapComp_naturality_right, ← map₂_associator_assoc]
    slice_rhs 3 5 =>
      rw [← G.map₂_comp, ← G.map₂_comp, ← F.map₂_associator, G.map₂_comp, G.map₂_comp]
    slice_lhs 1 3 =>
      rw [comp_whiskerRight, assoc, ← G.mapComp_naturality_left_assoc]
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      F✝ : CategoryTheory.LaxFunctor B C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.LaxFunctor B C
      G : CategoryTheory.LaxFunctor C D
      a✝ b✝ c✝ d✝ : B
      f : Quiver.Hom a✝ b✝
      g : Quiver.Hom b✝ c✝
      h : Quiver.Hom c✝ d✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [assoc]
    /-
      🎉 no goals
    -/
  map₂_leftUnitor := fun f => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      F✝ : CategoryTheory.LaxFunctor B C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.LaxFunctor B C
      G : CategoryTheory.LaxFunctor C D
      a✝ b✝ : B
      f : Quiver.Hom a✝ b✝
      ⊢ Eq ((F.comp G.toPrelaxFunctor).map₂ (CategoryTheory.Bicategory.leftUnitor f) …
    -/
    dsimp
    simp only [map₂_leftUnitor, PrelaxFunctor.map₂_comp, assoc, mapComp_naturality_left_assoc,
      comp_whiskerRight]
  map₂_rightUnitor := fun f => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      F✝ : CategoryTheory.LaxFunctor B C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.LaxFunctor B C
      G : CategoryTheory.LaxFunctor C D
      a✝ b✝ : B
      f : Quiver.Hom a✝ b✝
      ⊢ Eq ((F.comp G.toPrelaxFunctor).map₂ (CategoryTheory.Bicategory.rightUnitor f …
    -/
    dsimp
    simp only [map₂_rightUnitor, PrelaxFunctor.map₂_comp, assoc, mapComp_naturality_right_assoc,
      Bicategory.whiskerLeft_comp]


/-- A structure on an Lax functor that promotes an Lax functor to a pseudofunctor.

See `Pseudofunctor.mkOfLax`. -/
structure PseudoCore (F : LaxFunctor B C) where
  /-- The isomorphism giving rise to the lax unity constraint -/
  mapIdIso (a : B) : F.map (𝟙 a) ≅ 𝟙 (F.obj a)
  /-- The isomorphism giving rise to the lax functoriality constraint -/
  mapCompIso {a b c : B} (f : a ⟶ b) (g : b ⟶ c) : F.map (f ≫ g) ≅ F.map f ≫ F.map g
  /-- `mapIdIso` gives rise to the lax unity constraint -/
  mapIdIso_inv {a : B} : (mapIdIso a).inv = F.mapId a := by aesop_cat
  /-- `mapCompIso` gives rise to the lax functoriality constraint -/
  mapCompIso_inv {a b c : B} (f : a ⟶ b) (g : b ⟶ c) : (mapCompIso f g).inv = F.mapComp f g := by
    aesop_cat


