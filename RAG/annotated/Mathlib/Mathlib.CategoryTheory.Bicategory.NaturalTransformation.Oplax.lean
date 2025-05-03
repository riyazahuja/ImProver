/-- If `η` is an oplax natural transformation between `F` and `G`, we have a 1-morphism
`η.app a : F.obj a ⟶ G.obj a` for each object `a : B`. We also have a 2-morphism
`η.naturality f : F.map f ≫ app b ⟶ app a ≫ G.map f` for each 1-morphism `f : a ⟶ b`.
These 2-morphisms satisfies the naturality condition, and preserve the identities and
the compositions modulo some adjustments of domains and codomains of 2-morphisms.
-/
structure OplaxNatTrans (F G : OplaxFunctor B C) where
  app (a : B) : F.obj a ⟶ G.obj a
  naturality {a b : B} (f : a ⟶ b) : F.map f ≫ app b ⟶ app a ≫ G.map f
  naturality_naturality :
    ∀ {a b : B} {f g : a ⟶ b} (η : f ⟶ g),
      F.map₂ η ▷ app b ≫ naturality g = naturality f ≫ app a ◁ G.map₂ η := by
    aesop_cat
  naturality_id :
    ∀ a : B,
      naturality (𝟙 a) ≫ app a ◁ G.mapId a =
        F.mapId a ▷ app a ≫ (λ_ (app a)).hom ≫ (ρ_ (app a)).inv := by
    aesop_cat
  naturality_comp :
    ∀ {a b c : B} (f : a ⟶ b) (g : b ⟶ c),
      naturality (f ≫ g) ≫ app a ◁ G.mapComp f g =
        F.mapComp f g ▷ app c ≫
          (α_ _ _ _).hom ≫
            F.map f ◁ naturality g ≫ (α_ _ _ _).inv ≫ naturality f ▷ G.map g ≫ (α_ _ _ _).hom := by
    aesop_cat


attribute [reassoc (attr := simp)] OplaxNatTrans.naturality_naturality OplaxNatTrans.naturality_id
  OplaxNatTrans.naturality_comp


/-- The identity oplax natural transformation. -/
@[simps]
def id : OplaxNatTrans F F where
  app a := 𝟙 (F.obj a)
  naturality {_ _} f := (ρ_ (F.map f)).hom ≫ (λ_ (F.map f)).inv


instance : Inhabited (OplaxNatTrans F F) :=
  ⟨id F⟩


@[reassoc (attr := simp)]
theorem whiskerLeft_naturality_naturality (f : a' ⟶ G.obj a) {g h : a ⟶ b} (β : g ⟶ h) :
    f ◁ G.map₂ β ▷ θ.app b ≫ f ◁ θ.naturality h =
      f ◁ θ.naturality g ≫ f ◁ θ.app a ◁ H.map₂ β := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    G H : CategoryTheory.OplaxFunctor B C
    θ : CategoryTheory.OplaxNatTrans G H
    a b : B
    a' : C
    f : Quiver.Hom a' (G.obj a)
    g h : Quiver.Hom a b
    β : Quiver.Hom g h
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  simp_rw [← Bicategory.whiskerLeft_comp, naturality_naturality]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerRight_naturality_naturality {f g : a ⟶ b} (β : f ⟶ g) (h : G.obj b ⟶ a') :
    F.map₂ β ▷ η.app b ▷ h ≫ η.naturality g ▷ h =
      η.naturality f ▷ h ≫ (α_ _ _ _).hom ≫ η.app a ◁ G.map₂ β ▷ h ≫ (α_ _ _ _).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    η : CategoryTheory.OplaxNatTrans F G
    a b : B
    a' : C
    f g : Quiver.Hom a b
    β : Quiver.Hom f g
    h : Quiver.Hom (G.obj b) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  rw [← comp_whiskerRight, naturality_naturality, comp_whiskerRight, whisker_assoc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerLeft_naturality_comp (f : a' ⟶ G.obj a) (g : a ⟶ b) (h : b ⟶ c) :
    f ◁ θ.naturality (g ≫ h) ≫ f ◁ θ.app a ◁ H.mapComp g h =
      f ◁ G.mapComp g h ▷ θ.app c ≫
        f ◁ (α_ _ _ _).hom ≫
          f ◁ G.map g ◁ θ.naturality h ≫
            f ◁ (α_ _ _ _).inv ≫ f ◁ θ.naturality g ▷ H.map h ≫ f ◁ (α_ _ _ _).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    G H : CategoryTheory.OplaxFunctor B C
    θ : CategoryTheory.OplaxNatTrans G H
    a b c : B
    a' : C
    f : Quiver.Hom a' (G.obj a)
    g : Quiver.Hom a b
    h : Quiver.Hom b c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  simp_rw [← Bicategory.whiskerLeft_comp, naturality_comp]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerRight_naturality_comp (f : a ⟶ b) (g : b ⟶ c) (h : G.obj c ⟶ a') :
    η.naturality (f ≫ g) ▷ h ≫ (α_ _ _ _).hom ≫ η.app a ◁ G.mapComp f g ▷ h =
      F.mapComp f g ▷ η.app c ▷ h ≫
        (α_ _ _ _).hom ▷ h ≫
          (α_ _ _ _).hom ≫
            F.map f ◁ η.naturality g ▷ h ≫
              (α_ _ _ _).inv ≫
                (α_ _ _ _).inv ▷ h ≫
                  η.naturality f ▷ G.map g ▷ h ≫ (α_ _ _ _).hom ▷ h ≫ (α_ _ _ _).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    η : CategoryTheory.OplaxNatTrans F G
    a b c : B
    a' : C
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom (G.obj c) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  rw [← associator_naturality_middle, ← comp_whiskerRight_assoc, naturality_comp]; simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[reassoc (attr := simp)]
theorem whiskerLeft_naturality_id (f : a' ⟶ G.obj a) :
    f ◁ θ.naturality (𝟙 a) ≫ f ◁ θ.app a ◁ H.mapId a =
      f ◁ G.mapId a ▷ θ.app a ≫ f ◁ (λ_ (θ.app a)).hom ≫ f ◁ (ρ_ (θ.app a)).inv := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    G H : CategoryTheory.OplaxFunctor B C
    θ : CategoryTheory.OplaxNatTrans G H
    a : B
    a' : C
    f : Quiver.Hom a' (G.obj a)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  simp_rw [← Bicategory.whiskerLeft_comp, naturality_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerRight_naturality_id (f : G.obj a ⟶ a') :
    η.naturality (𝟙 a) ▷ f ≫ (α_ _ _ _).hom ≫ η.app a ◁ G.mapId a ▷ f =
    F.mapId a ▷ η.app a ▷ f ≫ (λ_ (η.app a)).hom ▷ f ≫ (ρ_ (η.app a)).inv ▷ f ≫ (α_ _ _ _).hom := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    η : CategoryTheory.OplaxNatTrans F G
    a : B
    a' : C
    f : Quiver.Hom (G.obj a) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  rw [← associator_naturality_middle, ← comp_whiskerRight_assoc, naturality_id]; simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- Vertical composition of oplax natural transformations. -/
@[simps]
def vcomp (η : OplaxNatTrans F G) (θ : OplaxNatTrans G H) : OplaxNatTrans F H where
  app a := η.app a ≫ θ.app a
  naturality {a b} f :=
    (α_ _ _ _).inv ≫
      η.naturality f ▷ θ.app b ≫ (α_ _ _ _).hom ≫ η.app a ◁ θ.naturality f ≫ (α_ _ _ _).inv
  naturality_comp {a b c} f g := by
    calc
      _ =
          ?_ ≫
            F.mapComp f g ▷ η.app c ▷ θ.app c ≫
              ?_ ≫
                F.map f ◁ η.naturality g ▷ θ.app c ≫
                  ?_ ≫
                    (F.map f ≫ η.app b) ◁ θ.naturality g ≫
                      η.naturality f ▷ (θ.app b ≫ H.map g) ≫
                        ?_ ≫ η.app a ◁ θ.naturality f ▷ H.map g ≫ ?_ :=
        ?_
      _ = _ := ?_
      /-
        case calc_1
        B : Type u₁
        inst✝¹ : CategoryTheory.Bicategory B
        C : Type u₂
        inst✝ : CategoryTheory.Bicategory C
        F G H : CategoryTheory.OplaxFunctor B C
        η✝ : CategoryTheory.OplaxNatTrans F G
        θ✝ : CategoryTheory.OplaxNatTrans G H
        η : CategoryTheory.OplaxNatTrans F G
        θ : CategoryTheory.OplaxNatTrans G H
        a b c : B
        f : Quiver.Hom a b
        g : Quiver.Hom b c
        ⊢ Quiver.Hom (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Catego …
      -/
    · exact (α_ _ _ _).inv
      /-
        🎉 no goals
      -/
      /-
        case calc_2
        B : Type u₁
        inst✝¹ : CategoryTheory.Bicategory B
        C : Type u₂
        inst✝ : CategoryTheory.Bicategory C
        F G H : CategoryTheory.OplaxFunctor B C
        η✝ : CategoryTheory.OplaxNatTrans F G
        θ✝ : CategoryTheory.OplaxNatTrans G H
        η : CategoryTheory.OplaxNatTrans F G
        θ : CategoryTheory.OplaxNatTrans G H
        a b c : B
        f : Quiver.Hom a b
        g : Quiver.Hom b c
        ⊢ Quiver.Hom (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruc …
      -/
    · exact (α_ _ _ _).hom ▷ _ ≫ (α_ _ _ _).hom
      /-
        🎉 no goals
      -/
      /-
        case calc_3
        B : Type u₁
        inst✝¹ : CategoryTheory.Bicategory B
        C : Type u₂
        inst✝ : CategoryTheory.Bicategory C
        F G H : CategoryTheory.OplaxFunctor B C
        η✝ : CategoryTheory.OplaxNatTrans F G
        θ✝ : CategoryTheory.OplaxNatTrans G H
        η : CategoryTheory.OplaxNatTrans F G
        θ : CategoryTheory.OplaxNatTrans G H
        a b c : B
        f : Quiver.Hom a b
        g : Quiver.Hom b c
        ⊢ Quiver.Hom (CategoryTheory.CategoryStruct.comp (F.map f) (CategoryTheory.Cat …
      -/
    · exact _ ◁ (α_ _ _ _).hom ≫ (α_ _ _ _).inv
      /-
        🎉 no goals
      -/
      /-
        case calc_4
        B : Type u₁
        inst✝¹ : CategoryTheory.Bicategory B
        C : Type u₂
        inst✝ : CategoryTheory.Bicategory C
        F G H : CategoryTheory.OplaxFunctor B C
        η✝ : CategoryTheory.OplaxNatTrans F G
        θ✝ : CategoryTheory.OplaxNatTrans G H
        η : CategoryTheory.OplaxNatTrans F G
        θ : CategoryTheory.OplaxNatTrans G H
        a b c : B
        f : Quiver.Hom a b
        g : Quiver.Hom b c
        ⊢ Quiver.Hom (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruc …
      -/
    · exact (α_ _ _ _).hom ≫ _ ◁ (α_ _ _ _).inv
      /-
        🎉 no goals
      -/
      /-
        case calc_5
        B : Type u₁
        inst✝¹ : CategoryTheory.Bicategory B
        C : Type u₂
        inst✝ : CategoryTheory.Bicategory C
        F G H : CategoryTheory.OplaxFunctor B C
        η✝ : CategoryTheory.OplaxNatTrans F G
        θ✝ : CategoryTheory.OplaxNatTrans G H
        η : CategoryTheory.OplaxNatTrans F G
        θ : CategoryTheory.OplaxNatTrans G H
        a b c : B
        f : Quiver.Hom a b
        g : Quiver.Hom b c
        ⊢ Quiver.Hom (CategoryTheory.CategoryStruct.comp (η.app a) (CategoryTheory.Cat …
      -/
    · exact _ ◁ (α_ _ _ _).hom ≫ (α_ _ _ _).inv
      /-
        🎉 no goals
      -/
      /-
        case calc_6
        B : Type u₁
        inst✝¹ : CategoryTheory.Bicategory B
        C : Type u₂
        inst✝ : CategoryTheory.Bicategory C
        F G H : CategoryTheory.OplaxFunctor B C
        η✝ : CategoryTheory.OplaxNatTrans F G
        θ✝ : CategoryTheory.OplaxNatTrans G H
        η : CategoryTheory.OplaxNatTrans F G
        θ : CategoryTheory.OplaxNatTrans G H
        a b c : B
        f : Quiver.Hom a b
        g : Quiver.Hom b c
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {a b} f => CategoryTheory.Categ …
      -/
    · rw [whisker_exchange_assoc]
      /-
        case calc_6
        B : Type u₁
        inst✝¹ : CategoryTheory.Bicategory B
        C : Type u₂
        inst✝ : CategoryTheory.Bicategory C
        F G H : CategoryTheory.OplaxFunctor B C
        η✝ : CategoryTheory.OplaxNatTrans F G
        θ✝ : CategoryTheory.OplaxNatTrans G H
        η : CategoryTheory.OplaxNatTrans F G
        θ : CategoryTheory.OplaxNatTrans G H
        a b c : B
        f : Quiver.Hom a b
        g : Quiver.Hom b c
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {a b} f => CategoryTheory.Categ …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case calc_7
        B : Type u₁
        inst✝¹ : CategoryTheory.Bicategory B
        C : Type u₂
        inst✝ : CategoryTheory.Bicategory C
        F G H : CategoryTheory.OplaxFunctor B C
        η✝ : CategoryTheory.OplaxNatTrans F G
        θ✝ : CategoryTheory.OplaxNatTrans G H
        η : CategoryTheory.OplaxNatTrans F G
        θ : CategoryTheory.OplaxNatTrans G H
        a b c : B
        f : Quiver.Hom a b
        g : Quiver.Hom b c
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.associator …
      -/
    · simp
      /-
        🎉 no goals
      -/


@[simps id comp]
instance : CategoryStruct (OplaxFunctor B C) where
  Hom := OplaxNatTrans
  id := OplaxNatTrans.id
  comp := OplaxNatTrans.vcomp


/-- A structure on an Oplax natural transformation that promotes it to a strong natural
transformation.

See `StrongNatTrans.mkOfOplax`. -/
structure StrongCore {F G : OplaxFunctor B C} (η : OplaxNatTrans F G) where
  naturality {a b : B} (f : a ⟶ b) : F.map f ≫ η.app b ≅ η.app a ≫ G.map f
  naturality_hom {a b : B} (f : a ⟶ b) : (naturality f).hom = η.naturality f := by aesop_cat


