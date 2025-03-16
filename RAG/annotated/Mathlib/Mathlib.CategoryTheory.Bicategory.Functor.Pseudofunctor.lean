/-- A pseudofunctor `F` between bicategories `B` and `C` consists of a function between objects
`F.obj`, a function between 1-morphisms `F.map`, and a function between 2-morphisms `F.map₂`.

Unlike functors between categories, `F.map` do not need to strictly commute with the compositions,
and do not need to strictly preserve the identity. Instead, there are specified 2-isomorphisms
`F.map (𝟙 a) ≅ 𝟙 (F.obj a)` and `F.map (f ≫ g) ≅ F.map f ≫ F.map g`.

`F.map₂` strictly commute with compositions and preserve the identity. They also preserve the
associator, the left unitor, and the right unitor modulo some adjustments of domains and codomains
of 2-morphisms.
-/
structure Pseudofunctor (B : Type u₁) [Bicategory.{w₁, v₁} B] (C : Type u₂)
    [Bicategory.{w₂, v₂} C] extends PrelaxFunctor B C where
  mapId (a : B) : map (𝟙 a) ≅ 𝟙 (obj a)
  mapComp {a b c : B} (f : a ⟶ b) (g : b ⟶ c) : map (f ≫ g) ≅ map f ≫ map g
  map₂_whisker_left :
    ∀ {a b c : B} (f : a ⟶ b) {g h : b ⟶ c} (η : g ⟶ h),
      map₂ (f ◁ η) = (mapComp f g).hom ≫ map f ◁ map₂ η ≫ (mapComp f h).inv := by
    aesop_cat
  map₂_whisker_right :
    ∀ {a b c : B} {f g : a ⟶ b} (η : f ⟶ g) (h : b ⟶ c),
      map₂ (η ▷ h) = (mapComp f h).hom ≫ map₂ η ▷ map h ≫ (mapComp g h).inv := by
    aesop_cat
  map₂_associator :
    ∀ {a b c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d),
      map₂ (α_ f g h).hom = (mapComp (f ≫ g) h).hom ≫ (mapComp f g).hom ▷ map h ≫
      (α_ (map f) (map g) (map h)).hom ≫ map f ◁ (mapComp g h).inv ≫
      (mapComp f (g ≫ h)).inv := by
    aesop_cat
  map₂_left_unitor :
    ∀ {a b : B} (f : a ⟶ b),
      map₂ (λ_ f).hom = (mapComp (𝟙 a) f).hom ≫ (mapId a).hom ▷ map f ≫ (λ_ (map f)).hom := by
    aesop_cat
  map₂_right_unitor :
    ∀ {a b : B} (f : a ⟶ b),
      map₂ (ρ_ f).hom = (mapComp f (𝟙 b)).hom ≫ map f ◁ (mapId b).hom ≫ (ρ_ (map f)).hom := by
    aesop_cat


attribute [simp, reassoc, to_app]
  map₂_whisker_left map₂_whisker_right map₂_associator map₂_left_unitor map₂_right_unitor


/-- The oplax functor associated with a pseudofunctor. -/
@[simps]
def toOplax : OplaxFunctor B C where
  toPrelaxFunctor := F.toPrelaxFunctor
  mapId := fun a => (F.mapId a).hom
  mapComp := fun f g => (F.mapComp f g).hom


instance hasCoeToOplax : Coe (Pseudofunctor B C) (OplaxFunctor B C) :=
  ⟨toOplax⟩


/-- The Lax functor associated with a pseudofunctor. -/
@[simps]
def toLax : LaxFunctor B C where
  toPrelaxFunctor := F.toPrelaxFunctor
  mapId := fun a => (F.mapId a).inv
  mapComp := fun f g => (F.mapComp f g).inv
  map₂_leftUnitor f := by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.Pseudofunctor B C
      a✝ b✝ : B
      f : Quiver.Hom a✝ b✝
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.leftUnitor f).inv) (CategoryTheory.Cat …
    -/
    rw [← F.map₂Iso_inv, eq_inv_comp, comp_inv_eq]
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.Pseudofunctor B C
      a✝ b✝ : B
      f : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.Bicategory.leftUnitor (F.map f)).hom (CategoryTheory.Cate …
    -/
    simp
    /-
      🎉 no goals
    -/
  map₂_rightUnitor f := by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.Pseudofunctor B C
      a✝ b✝ : B
      f : Quiver.Hom a✝ b✝
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.rightUnitor f).inv) (CategoryTheory.Ca …
    -/
    rw [← F.map₂Iso_inv, eq_inv_comp, comp_inv_eq]
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F : CategoryTheory.Pseudofunctor B C
      a✝ b✝ : B
      f : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.Bicategory.rightUnitor (F.map f)).hom (CategoryTheory.Cat …
    -/
    simp
    /-
      🎉 no goals
    -/


instance hasCoeToLax : Coe (Pseudofunctor B C) (LaxFunctor B C) :=
  ⟨toLax⟩


/-- The identity pseudofunctor. -/
@[simps]
def id (B : Type u₁) [Bicategory.{w₁, v₁} B] : Pseudofunctor B B where
  toPrelaxFunctor := PrelaxFunctor.id B
  mapId := fun a => Iso.refl (𝟙 a)
  mapComp := fun f g => Iso.refl (f ≫ g)


instance : Inhabited (Pseudofunctor B B) :=
  ⟨id B⟩


/-- Composition of pseudofunctors. -/
@[simps]
def comp (F : Pseudofunctor B C) (G : Pseudofunctor C D) : Pseudofunctor B D where
  toPrelaxFunctor := F.toPrelaxFunctor.comp G.toPrelaxFunctor
  mapId := fun a => G.map₂Iso (F.mapId a) ≪≫ G.mapId (F.obj a)
  mapComp := fun f g => (G.map₂Iso (F.mapComp f g)) ≪≫ G.mapComp (F.map f) (F.map g)
  -- Note: whilst these are all provable by `aesop_cat`, the proof is very slow
                              /-
                                B : Type u₁
                                inst✝² : CategoryTheory.Bicategory B
                                C : Type u₂
                                inst✝¹ : CategoryTheory.Bicategory C
                                D : Type u₃
                                inst✝ : CategoryTheory.Bicategory D
                                F✝ F : CategoryTheory.Pseudofunctor B C
                                G : CategoryTheory.Pseudofunctor C D
                                a✝ b✝ c✝ : B
                                f : Quiver.Hom a✝ b✝
                                η : Quiver.Hom b✝ c✝
                                ⊢ ∀ {h : Quiver.Hom b✝ c✝} (η_1 : Quiver.Hom η h), Eq ((F.comp G.toPrelaxFunct …
                              -/
  map₂_whisker_left f η := by dsimp; simp
                                     /-
                                       🎉 no goals
                                     -/
                               /-
                                 B : Type u₁
                                 inst✝² : CategoryTheory.Bicategory B
                                 C : Type u₂
                                 inst✝¹ : CategoryTheory.Bicategory C
                                 D : Type u₃
                                 inst✝ : CategoryTheory.Bicategory D
                                 F✝ F : CategoryTheory.Pseudofunctor B C
                                 G : CategoryTheory.Pseudofunctor C D
                                 a✝ b✝ c✝ : B
                                 f✝ g✝ : Quiver.Hom a✝ b✝
                                 η : Quiver.Hom f✝ g✝
                                 h : Quiver.Hom b✝ c✝
                                 ⊢ Eq ((F.comp G.toPrelaxFunctor).map₂ (CategoryTheory.Bicategory.whiskerRight  …
                               -/
  map₂_whisker_right η h := by dsimp; simp
                                      /-
                                        🎉 no goals
                                      -/
                              /-
                                B : Type u₁
                                inst✝² : CategoryTheory.Bicategory B
                                C : Type u₂
                                inst✝¹ : CategoryTheory.Bicategory C
                                D : Type u₃
                                inst✝ : CategoryTheory.Bicategory D
                                F✝ F : CategoryTheory.Pseudofunctor B C
                                G : CategoryTheory.Pseudofunctor C D
                                a✝ b✝ c✝ d✝ : B
                                f : Quiver.Hom a✝ b✝
                                g : Quiver.Hom b✝ c✝
                                h : Quiver.Hom c✝ d✝
                                ⊢ Eq ((F.comp G.toPrelaxFunctor).map₂ (CategoryTheory.Bicategory.associator f  …
                              -/
  map₂_associator f g h := by dsimp; simp
                                     /-
                                       🎉 no goals
                                     -/
                           /-
                             B : Type u₁
                             inst✝² : CategoryTheory.Bicategory B
                             C : Type u₂
                             inst✝¹ : CategoryTheory.Bicategory C
                             D : Type u₃
                             inst✝ : CategoryTheory.Bicategory D
                             F✝ F : CategoryTheory.Pseudofunctor B C
                             G : CategoryTheory.Pseudofunctor C D
                             a✝ b✝ : B
                             f : Quiver.Hom a✝ b✝
                             ⊢ Eq ((F.comp G.toPrelaxFunctor).map₂ (CategoryTheory.Bicategory.leftUnitor f) …
                           -/
  map₂_left_unitor f := by dsimp; simp
                                  /-
                                    🎉 no goals
                                  -/
                            /-
                              B : Type u₁
                              inst✝² : CategoryTheory.Bicategory B
                              C : Type u₂
                              inst✝¹ : CategoryTheory.Bicategory C
                              D : Type u₃
                              inst✝ : CategoryTheory.Bicategory D
                              F✝ F : CategoryTheory.Pseudofunctor B C
                              G : CategoryTheory.Pseudofunctor C D
                              a✝ b✝ : B
                              f : Quiver.Hom a✝ b✝
                              ⊢ Eq ((F.comp G.toPrelaxFunctor).map₂ (CategoryTheory.Bicategory.rightUnitor f …
                            -/
  map₂_right_unitor f := by dsimp; simp
                                   /-
                                     🎉 no goals
                                   -/


@[reassoc, to_app]
lemma mapComp_assoc_right_hom {c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) :
    (F.mapComp f (g ≫ h)).hom ≫ F.map f ◁ (F.mapComp g h).hom = F.map₂ (α_ f g h).inv ≫
    (F.mapComp (f ≫ g) h).hom ≫ (F.mapComp f g).hom ▷ F.map h ≫
    (α_ (F.map f) (F.map g) (F.map h)).hom :=
  F.toOplax.mapComp_assoc_right _ _ _


@[reassoc, to_app]
lemma mapComp_assoc_left_hom {c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) :
    (F.mapComp (f ≫ g) h).hom ≫ (F.mapComp f g).hom ▷ F.map h =
    F.map₂ (α_ f g h).hom ≫ (F.mapComp f (g ≫ h)).hom ≫ F.map f ◁ (F.mapComp g h).hom
    ≫ (α_ (F.map f) (F.map g) (F.map h)).inv :=
  F.toOplax.mapComp_assoc_left _ _ _


@[reassoc, to_app]
lemma mapComp_assoc_right_inv {c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) :
    F.map f ◁ (F.mapComp g h).inv ≫ (F.mapComp f (g ≫ h)).inv =
    (α_ (F.map f) (F.map g) (F.map h)).inv ≫ (F.mapComp f g).inv ▷ F.map h ≫
    (F.mapComp (f ≫ g) h).inv ≫ F.map₂ (α_ f g h).hom :=
  F.toLax.mapComp_assoc_right _ _ _


@[reassoc, to_app]
lemma mapComp_assoc_left_inv {c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) :
    (F.mapComp f g).inv ▷ F.map h ≫ (F.mapComp (f ≫ g) h).inv =
    (α_ (F.map f) (F.map g) (F.map h)).hom ≫ F.map f ◁ (F.mapComp g h).inv ≫
    (F.mapComp f (g ≫ h)).inv ≫ F.map₂ (α_ f g h).inv :=
  F.toLax.mapComp_assoc_left _ _ _


@[reassoc, to_app]
lemma mapComp_id_left_hom (f : a ⟶ b) : (F.mapComp (𝟙 a) f).hom =
    F.map₂ (λ_ f).hom ≫ (λ_ (F.map f)).inv ≫ (F.mapId a).inv ▷ F.map f := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.Pseudofunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (F.mapComp (CategoryTheory.CategoryStruct.id a) f).hom (CategoryTheory.Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma mapComp_id_left (f : a ⟶ b) : (F.mapComp (𝟙 a) f) = F.map₂Iso (λ_ f) ≪≫
    (λ_ (F.map f)).symm ≪≫ (whiskerRightIso (F.mapId a) (F.map f)).symm :=
  Iso.ext <| F.mapComp_id_left_hom f


@[reassoc, to_app]
lemma mapComp_id_left_inv (f : a ⟶ b) : (F.mapComp (𝟙 a) f).inv =
    (F.mapId a).hom ▷ F.map f ≫ (λ_ (F.map f)).hom ≫ F.map₂ (λ_ f).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.Pseudofunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (F.mapComp (CategoryTheory.CategoryStruct.id a) f).inv (CategoryTheory.Ca …
  -/
  simp [mapComp_id_left]
  /-
    🎉 no goals
  -/


lemma whiskerRightIso_mapId (f : a ⟶ b) : whiskerRightIso (F.mapId a) (F.map f) =
    (F.mapComp (𝟙 a) f).symm ≪≫ F.map₂Iso (λ_ f) ≪≫ (λ_ (F.map f)).symm := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.Pseudofunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (CategoryTheory.Bicategory.whiskerRightIso (F.mapId a) (F.map f)) ((F.map …
  -/
  simp [mapComp_id_left]
  /-
    🎉 no goals
  -/


@[reassoc, to_app]
lemma whiskerRight_mapId_hom (f : a ⟶ b) : (F.mapId a).hom ▷ F.map f =
    (F.mapComp (𝟙 a) f).inv ≫ F.map₂ (λ_ f).hom ≫ (λ_ (F.map f)).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.Pseudofunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (CategoryTheory.Bicategory.whiskerRight (F.mapId a).hom (F.map f)) (Categ …
  -/
  simp [whiskerRightIso_mapId]
  /-
    🎉 no goals
  -/


@[reassoc, to_app]
lemma whiskerRight_mapId_inv (f : a ⟶ b) : (F.mapId a).inv ▷ F.map f =
    (λ_ (F.map f)).hom ≫ F.map₂ (λ_ f).inv ≫ (F.mapComp (𝟙 a) f).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.Pseudofunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (CategoryTheory.Bicategory.whiskerRight (F.mapId a).inv (F.map f)) (Categ …
  -/
  simpa using congrArg (·.inv) (F.whiskerRightIso_mapId f)
  /-
    🎉 no goals
  -/


@[reassoc, to_app]
lemma mapComp_id_right_hom (f : a ⟶ b) : (F.mapComp f (𝟙 b)).hom =
    F.map₂ (ρ_ f).hom ≫ (ρ_ (F.map f)).inv ≫ F.map f ◁ (F.mapId b).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.Pseudofunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (F.mapComp f (CategoryTheory.CategoryStruct.id b)).hom (CategoryTheory.Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma mapComp_id_right (f : a ⟶ b) : (F.mapComp f (𝟙 b)) = F.map₂Iso (ρ_ f) ≪≫
    (ρ_ (F.map f)).symm ≪≫ (whiskerLeftIso (F.map f) (F.mapId b)).symm :=
  Iso.ext <| F.mapComp_id_right_hom f


@[reassoc, to_app]
lemma mapComp_id_right_inv (f : a ⟶ b) : (F.mapComp f (𝟙 b)).inv =
    F.map f ◁ (F.mapId b).hom ≫ (ρ_ (F.map f)).hom ≫ F.map₂ (ρ_ f).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.Pseudofunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (F.mapComp f (CategoryTheory.CategoryStruct.id b)).inv (CategoryTheory.Ca …
  -/
  simp [mapComp_id_right]
  /-
    🎉 no goals
  -/


lemma whiskerLeftIso_mapId (f : a ⟶ b) : whiskerLeftIso (F.map f) (F.mapId b) =
    (F.mapComp f (𝟙 b)).symm ≪≫ F.map₂Iso (ρ_ f) ≪≫ (ρ_ (F.map f)).symm := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.Pseudofunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (CategoryTheory.Bicategory.whiskerLeftIso (F.map f) (F.mapId b)) ((F.mapC …
  -/
  simp [mapComp_id_right]
  /-
    🎉 no goals
  -/


@[reassoc, to_app]
lemma whiskerLeft_mapId_hom (f : a ⟶ b) : F.map f ◁ (F.mapId b).hom =
    (F.mapComp f (𝟙 b)).inv ≫ F.map₂ (ρ_ f).hom ≫ (ρ_ (F.map f)).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.Pseudofunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (CategoryTheory.Bicategory.whiskerLeft (F.map f) (F.mapId b).hom) (Catego …
  -/
  simp [whiskerLeftIso_mapId]
  /-
    🎉 no goals
  -/


@[reassoc, to_app]
lemma whiskerLeft_mapId_inv (f : a ⟶ b) : F.map f ◁ (F.mapId b).inv =
    (ρ_ (F.map f)).hom ≫ F.map₂ (ρ_ f).inv ≫ (F.mapComp f (𝟙 b)).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.Pseudofunctor B C
    a b : B
    f : Quiver.Hom a b
    ⊢ Eq (CategoryTheory.Bicategory.whiskerLeft (F.map f) (F.mapId b).inv) (Catego …
  -/
  simpa using congrArg (·.inv) (F.whiskerLeftIso_mapId f)
  /-
    🎉 no goals
  -/


/-- Construct a pseudofunctor from an oplax functor whose `mapId` and `mapComp` are isomorphisms. -/
@[simps]
def mkOfOplax (F : OplaxFunctor B C) (F' : F.PseudoCore) : Pseudofunctor B C where
  toPrelaxFunctor := F.toPrelaxFunctor
  mapId := F'.mapIdIso
  mapComp := F'.mapCompIso
  map₂_whisker_left := fun f g h η => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      F' : F.PseudoCore
      a✝ b✝ c✝ : B
      f : Quiver.Hom a✝ b✝
      g h : Quiver.Hom b✝ c✝
      η : Quiver.Hom g h
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.whiskerLeft f η)) (CategoryTheory.Cate …
    -/
    dsimp
    rw [F'.mapCompIso_hom f g, ← F.mapComp_naturality_right_assoc, ← F'.mapCompIso_hom f h,
      hom_inv_id, comp_id]
  map₂_whisker_right := fun η h => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      F' : F.PseudoCore
      a✝ b✝ c✝ : B
      f✝ g✝ : Quiver.Hom a✝ b✝
      η : Quiver.Hom f✝ g✝
      h : Quiver.Hom b✝ c✝
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.whiskerRight η h)) (CategoryTheory.Cat …
    -/
    dsimp
    rw [F'.mapCompIso_hom _ h, ← F.mapComp_naturality_left_assoc, ← F'.mapCompIso_hom _ h,
      hom_inv_id, comp_id]
  map₂_associator := fun f g h => by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      F' : F.PseudoCore
      a✝ b✝ c✝ d✝ : B
      f : Quiver.Hom a✝ b✝
      g : Quiver.Hom b✝ c✝
      h : Quiver.Hom c✝ d✝
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.associator f g h).hom) (CategoryTheory …
    -/
    dsimp
    rw [F'.mapCompIso_hom (f ≫ g) h, F'.mapCompIso_hom f g, ← F.map₂_associator_assoc, ←
      F'.mapCompIso_hom f (g ≫ h), ← F'.mapCompIso_hom g h, whiskerLeft_hom_inv_assoc,
      hom_inv_id, comp_id]


/-- Construct a pseudofunctor from an oplax functor whose `mapId` and `mapComp` are isomorphisms. -/
@[simps!]
noncomputable def mkOfOplax' (F : OplaxFunctor B C) [∀ a, IsIso (F.mapId a)]
    [∀ {a b c} (f : a ⟶ b) (g : b ⟶ c), IsIso (F.mapComp f g)] : Pseudofunctor B C where
  toPrelaxFunctor := F.toPrelaxFunctor
  mapId := fun a => asIso (F.mapId a)
  mapComp := fun f g => asIso (F.mapComp f g)
  map₂_whisker_left := fun f g h η => by
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝³ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝² : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      inst✝¹ : ∀ (a : B), CategoryTheory.IsIso (F.mapId a)
      inst✝ : ∀ {a b c : B} (f : Quiver.Hom a b) (g : Quiver.Hom b c), CategoryTheor …
      a✝ b✝ c✝ : B
      f : Quiver.Hom a✝ b✝
      g h : Quiver.Hom b✝ c✝
      η : Quiver.Hom g h
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.whiskerLeft f η)) (CategoryTheory.Cate …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝³ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝² : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      inst✝¹ : ∀ (a : B), CategoryTheory.IsIso (F.mapId a)
      inst✝ : ∀ {a b c : B} (f : Quiver.Hom a b) (g : Quiver.Hom b c), CategoryTheor …
      a✝ b✝ c✝ : B
      f : Quiver.Hom a✝ b✝
      g h : Quiver.Hom b✝ c✝
      η : Quiver.Hom g h
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.whiskerLeft f η)) (CategoryTheory.Cate …
    -/
    rw [← assoc, IsIso.eq_comp_inv, F.mapComp_naturality_right]
    /-
      🎉 no goals
    -/
  map₂_whisker_right := fun η h => by
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝³ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝² : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      inst✝¹ : ∀ (a : B), CategoryTheory.IsIso (F.mapId a)
      inst✝ : ∀ {a b c : B} (f : Quiver.Hom a b) (g : Quiver.Hom b c), CategoryTheor …
      a✝ b✝ c✝ : B
      f✝ g✝ : Quiver.Hom a✝ b✝
      η : Quiver.Hom f✝ g✝
      h : Quiver.Hom b✝ c✝
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.whiskerRight η h)) (CategoryTheory.Cat …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝³ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝² : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      inst✝¹ : ∀ (a : B), CategoryTheory.IsIso (F.mapId a)
      inst✝ : ∀ {a b c : B} (f : Quiver.Hom a b) (g : Quiver.Hom b c), CategoryTheor …
      a✝ b✝ c✝ : B
      f✝ g✝ : Quiver.Hom a✝ b✝
      η : Quiver.Hom f✝ g✝
      h : Quiver.Hom b✝ c✝
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.whiskerRight η h)) (CategoryTheory.Cat …
    -/
    rw [← assoc, IsIso.eq_comp_inv, F.mapComp_naturality_left]
    /-
      🎉 no goals
    -/
  map₂_associator := fun f g h => by
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝³ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝² : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      inst✝¹ : ∀ (a : B), CategoryTheory.IsIso (F.mapId a)
      inst✝ : ∀ {a b c : B} (f : Quiver.Hom a b) (g : Quiver.Hom b c), CategoryTheor …
      a✝ b✝ c✝ d✝ : B
      f : Quiver.Hom a✝ b✝
      g : Quiver.Hom b✝ c✝
      h : Quiver.Hom c✝ d✝
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.associator f g h).hom) (CategoryTheory …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝³ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝² : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      inst✝¹ : ∀ (a : B), CategoryTheory.IsIso (F.mapId a)
      inst✝ : ∀ {a b c : B} (f : Quiver.Hom a b) (g : Quiver.Hom b c), CategoryTheor …
      a✝ b✝ c✝ d✝ : B
      f : Quiver.Hom a✝ b✝
      g : Quiver.Hom b✝ c✝
      h : Quiver.Hom c✝ d✝
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.associator f g h).hom) (CategoryTheory …
    -/
    simp only [← assoc]
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝³ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝² : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      inst✝¹ : ∀ (a : B), CategoryTheory.IsIso (F.mapId a)
      inst✝ : ∀ {a b c : B} (f : Quiver.Hom a b) (g : Quiver.Hom b c), CategoryTheor …
      a✝ b✝ c✝ d✝ : B
      f : Quiver.Hom a✝ b✝
      g : Quiver.Hom b✝ c✝
      h : Quiver.Hom c✝ d✝
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.associator f g h).hom) (CategoryTheory …
    -/
    rw [IsIso.eq_comp_inv, ← inv_whiskerLeft, IsIso.eq_comp_inv]
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝³ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝² : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.OplaxFunctor B C
      inst✝¹ : ∀ (a : B), CategoryTheory.IsIso (F.mapId a)
      inst✝ : ∀ {a b c : B} (f : Quiver.Hom a b) (g : Quiver.Hom b c), CategoryTheor …
      a✝ b✝ c✝ d✝ : B
      f : Quiver.Hom a✝ b✝
      g : Quiver.Hom b✝ c✝
      h : Quiver.Hom c✝ d✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [assoc, F.map₂_associator]
    /-
      🎉 no goals
    -/


/-- Construct a pseudofunctor from a lax functor whose `mapId` and `mapComp` are isomorphisms. -/
@[simps]
def mkOfLax (F : LaxFunctor B C) (F' : F.PseudoCore) : Pseudofunctor B C where
  toPrelaxFunctor := F.toPrelaxFunctor
  mapId := F'.mapIdIso
  mapComp := F'.mapCompIso
  map₂_whisker_left f g h η := by
    rw [F'.mapCompIso_inv, ← LaxFunctor.mapComp_naturality_right, ← F'.mapCompIso_inv,
      hom_inv_id_assoc]
  map₂_whisker_right η h := by
    rw [F'.mapCompIso_inv, ← LaxFunctor.mapComp_naturality_left, ← F'.mapCompIso_inv,
      hom_inv_id_assoc]
  map₂_associator {a b c d} f g h := by
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.LaxFunctor B C
      F' : F.PseudoCore
      a b c d : B
      f : Quiver.Hom a b
      g : Quiver.Hom b c
      h : Quiver.Hom c d
      ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.associator f g h).hom) (CategoryTheory …
    -/
    rw [F'.mapCompIso_inv, F'.mapCompIso_inv, ← inv_comp_eq, ← IsIso.inv_comp_eq]
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Bicategory B
      C : Type u₂
      inst✝¹ : CategoryTheory.Bicategory C
      D : Type u₃
      inst✝ : CategoryTheory.Bicategory D
      F✝ : CategoryTheory.Pseudofunctor B C
      F : CategoryTheory.LaxFunctor B C
      F' : F.PseudoCore
      a b c d : B
      f : Quiver.Hom a b
      g : Quiver.Hom b c
      h : Quiver.Hom c d
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.B …
    -/
    simp
    /-
      🎉 no goals
    -/
                                 /-
                                   B : Type u₁
                                   inst✝² : CategoryTheory.Bicategory B
                                   C : Type u₂
                                   inst✝¹ : CategoryTheory.Bicategory C
                                   D : Type u₃
                                   inst✝ : CategoryTheory.Bicategory D
                                   F✝ : CategoryTheory.Pseudofunctor B C
                                   F : CategoryTheory.LaxFunctor B C
                                   F' : F.PseudoCore
                                   a b : B
                                   f : Quiver.Hom a b
                                   ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.leftUnitor f).hom) (CategoryTheory.Cat …
                                 -/
  map₂_left_unitor {a b} f := by rw [← IsIso.inv_eq_inv, ← F.map₂_inv]; simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                                  /-
                                    B : Type u₁
                                    inst✝² : CategoryTheory.Bicategory B
                                    C : Type u₂
                                    inst✝¹ : CategoryTheory.Bicategory C
                                    D : Type u₃
                                    inst✝ : CategoryTheory.Bicategory D
                                    F✝ : CategoryTheory.Pseudofunctor B C
                                    F : CategoryTheory.LaxFunctor B C
                                    F' : F.PseudoCore
                                    a b : B
                                    f : Quiver.Hom a b
                                    ⊢ Eq (F.map₂ (CategoryTheory.Bicategory.rightUnitor f).hom) (CategoryTheory.Ca …
                                  -/
  map₂_right_unitor {a b} f := by rw [← IsIso.inv_eq_inv, ← F.map₂_inv]; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- Construct a pseudofunctor from a lax functor whose `mapId` and `mapComp` are isomorphisms. -/
@[simps!]
noncomputable def mkOfLax' (F : LaxFunctor B C) [∀ a, IsIso (F.mapId a)]
    [∀ {a b c} (f : a ⟶ b) (g : b ⟶ c), IsIso (F.mapComp f g)] : Pseudofunctor B C :=
  mkOfLax F
  { mapIdIso := fun a => (asIso (F.mapId a)).symm
    mapCompIso := fun f g => (asIso (F.mapComp f g)).symm }


