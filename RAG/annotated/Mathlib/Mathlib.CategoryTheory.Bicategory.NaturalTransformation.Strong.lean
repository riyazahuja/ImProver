/-- A strong natural transformation between oplax functors `F` and `G` is a natural transformation
that is "natural up to 2-isomorphisms".

More precisely, it consists of the following:
* a 1-morphism `η.app a : F.obj a ⟶ G.obj a` for each object `a : B`.
* a 2-isomorphism `η.naturality f : F.map f ≫ app b ⟶ app a ≫ G.map f` for each 1-morphism
`f : a ⟶ b`.
* These 2-isomorphisms satisfy the naturality condition, and preserve the identities and the
compositions modulo some adjustments of domains and codomains of 2-morphisms.
-/
structure StrongOplaxNatTrans (F G : OplaxFunctor B C) where
  app (a : B) : F.obj a ⟶ G.obj a
  naturality {a b : B} (f : a ⟶ b) : F.map f ≫ app b ≅ app a ≫ G.map f
  naturality_naturality :
    ∀ {a b : B} {f g : a ⟶ b} (η : f ⟶ g),
      F.map₂ η ▷ app b ≫ (naturality g).hom = (naturality f).hom ≫ app a ◁ G.map₂ η := by
    aesop_cat
  naturality_id :
    ∀ a : B,
      (naturality (𝟙 a)).hom ≫ app a ◁ G.mapId a =
        F.mapId a ▷ app a ≫ (λ_ (app a)).hom ≫ (ρ_ (app a)).inv := by
    aesop_cat
  naturality_comp :
    ∀ {a b c : B} (f : a ⟶ b) (g : b ⟶ c),
      (naturality (f ≫ g)).hom ≫ app a ◁ G.mapComp f g =
        F.mapComp f g ▷ app c ≫ (α_ _ _ _).hom ≫ F.map f ◁ (naturality g).hom ≫
        (α_ _ _ _).inv ≫ (naturality f).hom ▷ G.map g ≫ (α_ _ _ _).hom := by
    aesop_cat


attribute [reassoc (attr := simp)] StrongOplaxNatTrans.naturality_naturality
  StrongOplaxNatTrans.naturality_id StrongOplaxNatTrans.naturality_comp


/-- The underlying oplax natural transformation of a strong natural transformation. -/
@[simps]
def toOplax {F G : OplaxFunctor B C} (η : StrongOplaxNatTrans F G) : OplaxNatTrans F G where
  app := η.app
  naturality f := (η.naturality f).hom


/-- Construct a strong natural transformation from an oplax natural transformation whose
naturality 2-cell is an isomorphism. -/
def mkOfOplax {F G : OplaxFunctor B C} (η : OplaxNatTrans F G) (η' : OplaxNatTrans.StrongCore η) :
    StrongOplaxNatTrans F G where
  app := η.app
  naturality := η'.naturality


/-- Construct a strong natural transformation from an oplax natural transformation whose
naturality 2-cell is an isomorphism. -/
noncomputable def mkOfOplax' {F G : OplaxFunctor B C} (η : OplaxNatTrans F G)
    [∀ a b (f : a ⟶ b), IsIso (η.naturality f)] : StrongOplaxNatTrans F G where
  app := η.app
  naturality := fun _ => asIso (η.naturality _)


/-- The identity strong natural transformation. -/
@[simps!]
def id : StrongOplaxNatTrans F F :=
  mkOfOplax (OplaxNatTrans.id F) { naturality := fun f ↦ (ρ_ (F.map f)) ≪≫ (λ_ (F.map f)).symm }


@[simp]
lemma id.toOplax : (id F).toOplax = OplaxNatTrans.id F :=
  rfl


instance : Inhabited (StrongOplaxNatTrans F F) :=
  ⟨id F⟩


@[reassoc (attr := simp)]
theorem whiskerLeft_naturality_naturality (f : a' ⟶ G.obj a) {g h : a ⟶ b} (β : g ⟶ h) :
    f ◁ G.map₂ β ▷ θ.app b ≫ f ◁ (θ.naturality h).hom =
      f ◁ (θ.naturality g).hom ≫ f ◁ θ.app a ◁ H.map₂ β := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    G H : CategoryTheory.OplaxFunctor B C
    θ : CategoryTheory.StrongOplaxNatTrans G H
    a b : B
    a' : C
    f : Quiver.Hom a' (G.obj a)
    g h : Quiver.Hom a b
    β : Quiver.Hom g h
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  apply θ.toOplax.whiskerLeft_naturality_naturality
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerRight_naturality_naturality {f g : a ⟶ b} (β : f ⟶ g) (h : G.obj b ⟶ a') :
    F.map₂ β ▷ η.app b ▷ h ≫ (η.naturality g).hom ▷ h =
      (η.naturality f).hom ▷ h ≫ (α_ _ _ _).hom ≫ η.app a ◁ G.map₂ β ▷ h ≫ (α_ _ _ _).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    η : CategoryTheory.StrongOplaxNatTrans F G
    a b : B
    a' : C
    f g : Quiver.Hom a b
    β : Quiver.Hom f g
    h : Quiver.Hom (G.obj b) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  apply η.toOplax.whiskerRight_naturality_naturality
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerLeft_naturality_comp (f : a' ⟶ G.obj a) (g : a ⟶ b) (h : b ⟶ c) :
    f ◁ (θ.naturality (g ≫ h)).hom ≫ f ◁ θ.app a ◁ H.mapComp g h =
      f ◁ G.mapComp g h ▷ θ.app c ≫
        f ◁ (α_ _ _ _).hom ≫
          f ◁ G.map g ◁ (θ.naturality h).hom ≫
            f ◁ (α_ _ _ _).inv ≫ f ◁ (θ.naturality g).hom ▷ H.map h ≫ f ◁ (α_ _ _ _).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    G H : CategoryTheory.OplaxFunctor B C
    θ : CategoryTheory.StrongOplaxNatTrans G H
    a b c : B
    a' : C
    f : Quiver.Hom a' (G.obj a)
    g : Quiver.Hom a b
    h : Quiver.Hom b c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  apply θ.toOplax.whiskerLeft_naturality_comp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerRight_naturality_comp (f : a ⟶ b) (g : b ⟶ c) (h : G.obj c ⟶ a') :
    (η.naturality (f ≫ g)).hom ▷ h ≫ (α_ _ _ _).hom ≫ η.app a ◁ G.mapComp f g ▷ h =
      F.mapComp f g ▷ η.app c ▷ h ≫
        (α_ _ _ _).hom ▷ h ≫
          (α_ _ _ _).hom ≫
            F.map f ◁ (η.naturality g).hom ▷ h ≫
              (α_ _ _ _).inv ≫
                (α_ _ _ _).inv ▷ h ≫
                 (η.naturality f).hom ▷ G.map g ▷ h ≫ (α_ _ _ _).hom ▷ h ≫ (α_ _ _ _).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    η : CategoryTheory.StrongOplaxNatTrans F G
    a b c : B
    a' : C
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom (G.obj c) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  apply η.toOplax.whiskerRight_naturality_comp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerLeft_naturality_id (f : a' ⟶ G.obj a) :
    f ◁ (θ.naturality (𝟙 a)).hom ≫ f ◁ θ.app a ◁ H.mapId a =
      f ◁ G.mapId a ▷ θ.app a ≫ f ◁ (λ_ (θ.app a)).hom ≫ f ◁ (ρ_ (θ.app a)).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    G H : CategoryTheory.OplaxFunctor B C
    θ : CategoryTheory.StrongOplaxNatTrans G H
    a : B
    a' : C
    f : Quiver.Hom a' (G.obj a)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  apply θ.toOplax.whiskerLeft_naturality_id
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerRight_naturality_id (f : G.obj a ⟶ a') :
    (η.naturality (𝟙 a)).hom ▷ f ≫ (α_ _ _ _).hom ≫ η.app a ◁ G.mapId a ▷ f =
    F.mapId a ▷ η.app a ▷ f ≫ (λ_ (η.app a)).hom ▷ f ≫ (ρ_ (η.app a)).inv ▷ f ≫
    (α_ _ _ _).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    η : CategoryTheory.StrongOplaxNatTrans F G
    a : B
    a' : C
    f : Quiver.Hom (G.obj a) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  apply η.toOplax.whiskerRight_naturality_id
  /-
    🎉 no goals
  -/


/-- Vertical composition of strong natural transformations. -/
@[simps!]
def vcomp (η : StrongOplaxNatTrans F G) (θ : StrongOplaxNatTrans G H) : StrongOplaxNatTrans F H :=
  mkOfOplax (OplaxNatTrans.vcomp η.toOplax θ.toOplax)
    { naturality := fun {a b} f ↦
        (α_ _ _ _).symm ≪≫ whiskerRightIso (η.naturality f) (θ.app b) ≪≫
        (α_ _ _ _) ≪≫ whiskerLeftIso (η.app a) (θ.naturality f) ≪≫ (α_ _ _ _).symm }

@[simps id comp]
instance Pseudofunctor.categoryStruct : CategoryStruct (Pseudofunctor B C) where
  Hom F G := StrongOplaxNatTrans F.toOplax G.toOplax
  id F := StrongOplaxNatTrans.id F.toOplax
  comp := StrongOplaxNatTrans.vcomp



