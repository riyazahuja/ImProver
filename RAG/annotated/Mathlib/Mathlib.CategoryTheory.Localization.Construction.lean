/-- If `W : MorphismProperty C`, `LocQuiver W` is a quiver with the same objects
as `C`, and whose morphisms are those in `C` and placeholders for formal
inverses of the morphisms in `W`. -/
structure LocQuiver (W : MorphismProperty C) where
  /-- underlying object -/
  obj : C


instance : Quiver (LocQuiver W) where Hom A B := (A.obj ⟶ B.obj) ⊕ { f : B.obj ⟶ A.obj // W f }


/-- The object in the path category of `LocQuiver W` attached to an object in
the category `C` -/
def ιPaths (X : C) : Paths (LocQuiver W) :=
  ⟨X⟩


/-- The morphism in the path category associated to a morphism in the original category. -/
@[simp]
def ψ₁ {X Y : C} (f : X ⟶ Y) : ιPaths W X ⟶ ιPaths W Y :=
  Paths.of.map (Sum.inl f)


/-- The morphism in the path category corresponding to a formal inverse. -/
@[simp]
def ψ₂ {X Y : C} (w : X ⟶ Y) (hw : W w) : ιPaths W Y ⟶ ιPaths W X :=
  Paths.of.map (Sum.inr ⟨w, hw⟩)


/-- The relations by which we take the quotient in order to get the localized category. -/
inductive relations : HomRel (Paths (LocQuiver W))
  | id (X : C) : relations (ψ₁ W (𝟙 X)) (𝟙 _)
  | comp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) : relations (ψ₁ W (f ≫ g)) (ψ₁ W f ≫ ψ₁ W g)
  | Winv₁ {X Y : C} (w : X ⟶ Y) (hw : W w) : relations (ψ₁ W w ≫ ψ₂ W w hw) (𝟙 _)
  | Winv₂ {X Y : C} (w : X ⟶ Y) (hw : W w) : relations (ψ₂ W w hw ≫ ψ₁ W w) (𝟙 _)


/-- The localized category obtained by formally inverting the morphisms
in `W : MorphismProperty C` -/
def Localization :=
  CategoryTheory.Quotient (Localization.Construction.relations W)


instance : Category (Localization W) := by
  /-
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    D : Type uD
    inst✝ : CategoryTheory.Category.{uD', uD} D
    ⊢ CategoryTheory.Category.{?u.3253, uC} W.Localization
  -/
  dsimp only [Localization]
  /-
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    D : Type uD
    inst✝ : CategoryTheory.Category.{uD', uD} D
    ⊢ CategoryTheory.Category.{?u.3253, uC} (CategoryTheory.Quotient (CategoryTheo …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The obvious functor `C ⥤ W.Localization` -/
def Q : C ⥤ W.Localization where
  obj X := (Quotient.functor _).obj (Paths.of.obj ⟨X⟩)
  map f := (Quotient.functor _).map (ψ₁ W f)
  map_id X := Quotient.sound _ (relations.id X)
  map_comp f g := Quotient.sound _ (relations.comp f g)


/-- The isomorphism in `W.Localization` associated to a morphism `w` in W -/
def wIso {X Y : C} (w : X ⟶ Y) (hw : W w) : Iso (W.Q.obj X) (W.Q.obj Y) where
  hom := W.Q.map w
                                      /-
                                        C : Type uC
                                        inst✝¹ : CategoryTheory.Category.{uC', uC} C
                                        W : CategoryTheory.MorphismProperty C
                                        D : Type uD
                                        inst✝ : CategoryTheory.Category.{uD', uD} D
                                        X Y : C
                                        w : Quiver.Hom X Y
                                        hw : W w
                                        ⊢ Quiver.Hom (CategoryTheory.Paths.of.obj { obj := Y }) (CategoryTheory.Paths. …
                                      -/
  inv := (Quotient.functor _).map (by dsimp; exact Paths.of.map (Sum.inr ⟨w, hw⟩))
                                             /-
                                               🎉 no goals
                                             -/
  hom_inv_id := Quotient.sound _ (relations.Winv₁ w hw)
  inv_hom_id := Quotient.sound _ (relations.Winv₂ w hw)


/-- The formal inverse in `W.Localization` of a morphism `w` in `W`. -/
abbrev wInv {X Y : C} (w : X ⟶ Y) (hw : W w) :=
  (wIso w hw).inv


theorem _root_.CategoryTheory.MorphismProperty.Q_inverts : W.IsInvertedBy W.Q := fun _ _ w hw =>
  (Localization.Construction.wIso w hw).isIso_hom


/-- The lifting of a functor to the path category of `LocQuiver W` -/
@[simps!]
def liftToPathCategory : Paths (LocQuiver W) ⥤ D :=
  Quiv.lift
    { obj := fun X => G.obj X.obj
      map := by
        /-
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          ⊢ {X Y : CategoryTheory.Localization.Construction.LocQuiver W} → Quiver.Hom X  …
        -/
        intros X Y
        /-
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          X Y : CategoryTheory.Localization.Construction.LocQuiver W
          ⊢ Quiver.Hom X Y → Quiver.Hom ((fun X => G.obj X.obj) X) ((fun X => G.obj X.ob …
        -/
        rintro (f | ⟨g, hg⟩)
          /-
            case inl
            C : Type uC
            inst✝¹ : CategoryTheory.Category.{uC', uC} C
            W : CategoryTheory.MorphismProperty C
            D : Type uD
            inst✝ : CategoryTheory.Category.{uD', uD} D
            G : CategoryTheory.Functor C D
            hG : W.IsInvertedBy G
            X Y : CategoryTheory.Localization.Construction.LocQuiver W
            f : Quiver.Hom X.obj Y.obj
            ⊢ Quiver.Hom ((fun X => G.obj X.obj) X) ((fun X => G.obj X.obj) Y)
          -/
        · exact G.map f
          /-
            🎉 no goals
          -/
          /-
            case inr.mk
            C : Type uC
            inst✝¹ : CategoryTheory.Category.{uC', uC} C
            W : CategoryTheory.MorphismProperty C
            D : Type uD
            inst✝ : CategoryTheory.Category.{uD', uD} D
            G : CategoryTheory.Functor C D
            hG : W.IsInvertedBy G
            X Y : CategoryTheory.Localization.Construction.LocQuiver W
            g : Quiver.Hom Y.obj X.obj
            hg : W g
            ⊢ Quiver.Hom ((fun X => G.obj X.obj) X) ((fun X => G.obj X.obj) Y)
          -/
        · haveI := hG g hg
          /-
            case inr.mk
            C : Type uC
            inst✝¹ : CategoryTheory.Category.{uC', uC} C
            W : CategoryTheory.MorphismProperty C
            D : Type uD
            inst✝ : CategoryTheory.Category.{uD', uD} D
            G : CategoryTheory.Functor C D
            hG : W.IsInvertedBy G
            X Y : CategoryTheory.Localization.Construction.LocQuiver W
            g : Quiver.Hom Y.obj X.obj
            hg : W g
            this : CategoryTheory.IsIso (G.map g)
            ⊢ Quiver.Hom ((fun X => G.obj X.obj) X) ((fun X => G.obj X.obj) Y)
          -/
          exact inv (G.map g) }
          /-
            🎉 no goals
          -/


/-- The lifting of a functor `C ⥤ D` inverting `W` as a functor `W.Localization ⥤ D` -/
@[simps!]
def lift : W.Localization ⥤ D :=
  Quotient.lift (relations W) (liftToPathCategory G hG)
    (by
      /-
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G : CategoryTheory.Functor C D
        hG : W.IsInvertedBy G
        ⊢ ∀ (x y : CategoryTheory.Paths (CategoryTheory.Localization.Construction.LocQ …
      -/
      rintro ⟨X⟩ ⟨Y⟩ f₁ f₂ r
      -- Porting note: rest of proof was `rcases r with ⟨⟩; tidy`
      /-
        case mk.mk
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G : CategoryTheory.Functor C D
        hG : W.IsInvertedBy G
        X Y : C
        f₁ f₂ : Quiver.Hom { obj := X } { obj := Y }
        r : CategoryTheory.Localization.Construction.relations W f₁ f₂
        ⊢ Eq ((CategoryTheory.Localization.Construction.liftToPathCategory G hG).map f …
      -/
      rcases r with (_|_|⟨f,hf⟩|⟨f,hf⟩)
        /-
          case mk.mk.id
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          X✝ : C
          X Y : CategoryTheory.Paths (CategoryTheory.Localization.Construction.LocQuiver …
          ⊢ Eq ((CategoryTheory.Localization.Construction.liftToPathCategory G hG).map ( …
        -/
      · aesop_cat
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.comp
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          X✝ Y✝¹ : C
          X Y : CategoryTheory.Paths (CategoryTheory.Localization.Construction.LocQuiver …
          Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          g✝ : Quiver.Hom Y✝ Y✝¹
          ⊢ Eq ((CategoryTheory.Localization.Construction.liftToPathCategory G hG).map ( …
        -/
      · aesop_cat
        /-
          🎉 no goals
        -/
      all_goals
        dsimp
        haveI := hG f hf
        simp
        rfl)


@[simp]
theorem fac : W.Q ⋙ lift G hG = G :=
  Functor.ext (fun _ => rfl)
    (by
      /-
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G : CategoryTheory.Functor C D
        hG : W.IsInvertedBy G
        ⊢ ∀ (X Y : C) (f : Quiver.Hom X Y), Eq ((W.Q.comp (CategoryTheory.Localization …
      -/
      intro X Y f
      /-
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G : CategoryTheory.Functor C D
        hG : W.IsInvertedBy G
        X Y : C
        f : Quiver.Hom X Y
        ⊢ Eq ((W.Q.comp (CategoryTheory.Localization.Construction.lift G hG)).map f) ( …
      -/
      simp only [Functor.comp_map, eqToHom_refl, comp_id, id_comp]
      /-
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G : CategoryTheory.Functor C D
        hG : W.IsInvertedBy G
        X Y : C
        f : Quiver.Hom X Y
        ⊢ Eq ((CategoryTheory.Localization.Construction.lift G hG).map (W.Q.map f)) (G …
      -/
      dsimp [MorphismProperty.Q, Quot.liftOn, Quotient.functor]
      /-
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G : CategoryTheory.Functor C D
        hG : W.IsInvertedBy G
        X Y : C
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.composePath (G.map f).toPath) (G.map f)
      -/
      rw [composePath_toPath])
      /-
        🎉 no goals
      -/


theorem uniq (G₁ G₂ : W.Localization ⥤ D) (h : W.Q ⋙ G₁ = W.Q ⋙ G₂) : G₁ = G₂ := by
  suffices h' : Quotient.functor _ ⋙ G₁ = Quotient.functor _ ⋙ G₂ by
    refine Functor.ext ?_ ?_
    · rintro ⟨⟨X⟩⟩
      apply Functor.congr_obj h
    · rintro ⟨⟨X⟩⟩ ⟨⟨Y⟩⟩ ⟨f⟩
      apply Functor.congr_hom h'
  /-
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    D : Type uD
    inst✝ : CategoryTheory.Category.{uD', uD} D
    G₁ G₂ : CategoryTheory.Functor W.Localization D
    h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
    ⊢ Eq ((CategoryTheory.Quotient.functor (CategoryTheory.Localization.Constructi …
  -/
  refine Paths.ext_functor ?_ ?_
    /-
      case refine_1
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G₁ G₂ : CategoryTheory.Functor W.Localization D
      h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
      ⊢ Eq ((CategoryTheory.Quotient.functor (CategoryTheory.Localization.Constructi …
    -/
  · ext X
    /-
      case refine_1.h
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G₁ G₂ : CategoryTheory.Functor W.Localization D
      h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
      X : CategoryTheory.Paths (CategoryTheory.Localization.Construction.LocQuiver W)
      ⊢ Eq (((CategoryTheory.Quotient.functor (CategoryTheory.Localization.Construct …
    -/
    cases X
    /-
      case refine_1.h.mk
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G₁ G₂ : CategoryTheory.Functor W.Localization D
      h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
      obj✝ : C
      ⊢ Eq (((CategoryTheory.Quotient.functor (CategoryTheory.Localization.Construct …
    -/
    apply Functor.congr_obj h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G₁ G₂ : CategoryTheory.Functor W.Localization D
      h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
      ⊢ ∀ (a b : CategoryTheory.Localization.Construction.LocQuiver W) (e : Quiver.H …
    -/
  · rintro ⟨X⟩ ⟨Y⟩ (f | ⟨w, hw⟩)
      /-
        case refine_2.mk.mk.inl
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G₁ G₂ : CategoryTheory.Functor W.Localization D
        h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
        X Y : C
        f : Quiver.Hom { obj := X }.obj { obj := Y }.obj
        ⊢ Eq (((CategoryTheory.Quotient.functor (CategoryTheory.Localization.Construct …
      -/
    · simpa only using Functor.congr_hom h f
      /-
        🎉 no goals
      -/
      /-
        case refine_2.mk.mk.inr.mk
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G₁ G₂ : CategoryTheory.Functor W.Localization D
        h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
        X Y : C
        w : Quiver.Hom { obj := Y }.obj { obj := X }.obj
        hw : W w
        ⊢ Eq (((CategoryTheory.Quotient.functor (CategoryTheory.Localization.Construct …
      -/
    · have hw : W.Q.map w = (wIso w hw).hom := rfl
      /-
        case refine_2.mk.mk.inr.mk
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G₁ G₂ : CategoryTheory.Functor W.Localization D
        h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
        X Y : C
        w : Quiver.Hom { obj := Y }.obj { obj := X }.obj
        hw✝ : W w
        hw : Eq (W.Q.map w) (CategoryTheory.Localization.Construction.wIso w hw✝).hom
        ⊢ Eq (((CategoryTheory.Quotient.functor (CategoryTheory.Localization.Construct …
      -/
      have hw' := Functor.congr_hom h w
      /-
        case refine_2.mk.mk.inr.mk
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G₁ G₂ : CategoryTheory.Functor W.Localization D
        h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
        X Y : C
        w : Quiver.Hom { obj := Y }.obj { obj := X }.obj
        hw✝ : W w
        hw : Eq (W.Q.map w) (CategoryTheory.Localization.Construction.wIso w hw✝).hom
        hw' : Eq ((W.Q.comp G₁).map w) (CategoryTheory.CategoryStruct.comp (CategoryTh …
        ⊢ Eq (((CategoryTheory.Quotient.functor (CategoryTheory.Localization.Construct …
      -/
      simp only [Functor.comp_map, hw] at hw'
      /-
        case refine_2.mk.mk.inr.mk
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G₁ G₂ : CategoryTheory.Functor W.Localization D
        h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
        X Y : C
        w : Quiver.Hom { obj := Y }.obj { obj := X }.obj
        hw✝ : W w
        hw : Eq (W.Q.map w) (CategoryTheory.Localization.Construction.wIso w hw✝).hom
        hw' : Eq (G₁.map (CategoryTheory.Localization.Construction.wIso w hw✝).hom) (C …
        ⊢ Eq (((CategoryTheory.Quotient.functor (CategoryTheory.Localization.Construct …
      -/
      refine Functor.congr_inv_of_congr_hom _ _ _ ?_ ?_ hw'
      /-
        case refine_2.mk.mk.inr.mk.refine_1
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G₁ G₂ : CategoryTheory.Functor W.Localization D
        h : Eq (W.Q.comp G₁) (W.Q.comp G₂)
        X Y : C
        w : Quiver.Hom { obj := Y }.obj { obj := X }.obj
        hw✝ : W w
        hw : Eq (W.Q.map w) (CategoryTheory.Localization.Construction.wIso w hw✝).hom
        hw' : Eq (G₁.map (CategoryTheory.Localization.Construction.wIso w hw✝).hom) (C …
        ⊢ Eq (G₁.obj (W.Q.obj Y)) (G₂.obj (W.Q.obj Y))
      -/
      all_goals apply Functor.congr_obj h
      /-
        🎉 no goals
      -/


/-- The canonical bijection between objects in a category and its
localization with respect to a morphism_property `W` -/
@[simps]
def objEquiv : C ≃ W.Localization where
  toFun := W.Q.obj
  invFun X := X.as.obj
  left_inv _ := rfl
  right_inv := by
    /-
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G : CategoryTheory.Functor C D
      hG : W.IsInvertedBy G
      ⊢ Function.RightInverse (fun X => X.as.obj) W.Q.obj
    -/
    rintro ⟨⟨X⟩⟩
    /-
      case mk.mk
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G : CategoryTheory.Functor C D
      hG : W.IsInvertedBy G
      X : C
      ⊢ Eq (W.Q.obj ((fun X => X.as.obj) { as := { obj := X } })) { as := { obj := X …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- A `MorphismProperty` in `W.Localization` is satisfied by all
morphisms in the localized category if it contains the image of the
morphisms in the original category, the inverses of the morphisms
in `W` and if it is stable under composition -/
theorem morphismProperty_is_top (P : MorphismProperty W.Localization)
    [P.IsStableUnderComposition] (hP₁ : ∀ ⦃X Y : C⦄ (f : X ⟶ Y), P (W.Q.map f))
    (hP₂ : ∀ ⦃X Y : C⦄ (w : X ⟶ Y) (hw : W w), P (wInv w hw)) :
    P = ⊤ := by
  /-
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    P : CategoryTheory.MorphismProperty W.Localization
    inst✝ : P.IsStableUnderComposition
    hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
    hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
    ⊢ Eq P Top.top
  -/
  funext X Y f
  /-
    case h.h.h
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    P : CategoryTheory.MorphismProperty W.Localization
    inst✝ : P.IsStableUnderComposition
    hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
    hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
    X Y : W.Localization
    f : Quiver.Hom X Y
    ⊢ Eq (P f) (Top.top f)
  -/
  ext
  /-
    case h.h.h.a
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    P : CategoryTheory.MorphismProperty W.Localization
    inst✝ : P.IsStableUnderComposition
    hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
    hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
    X Y : W.Localization
    f : Quiver.Hom X Y
    ⊢ Iff (P f) (Top.top f)
  -/
  constructor
    /-
      case h.h.h.a.mp
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      P : CategoryTheory.MorphismProperty W.Localization
      inst✝ : P.IsStableUnderComposition
      hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
      hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
      X Y : W.Localization
      f : Quiver.Hom X Y
      ⊢ P f → Top.top f
    -/
  · intro
    /-
      case h.h.h.a.mp
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      P : CategoryTheory.MorphismProperty W.Localization
      inst✝ : P.IsStableUnderComposition
      hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
      hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
      X Y : W.Localization
      f : Quiver.Hom X Y
      a✝ : P f
      ⊢ Top.top f
    -/
    apply MorphismProperty.top_apply
    /-
      🎉 no goals
    -/
    /-
      case h.h.h.a.mpr
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      P : CategoryTheory.MorphismProperty W.Localization
      inst✝ : P.IsStableUnderComposition
      hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
      hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
      X Y : W.Localization
      f : Quiver.Hom X Y
      ⊢ Top.top f → P f
    -/
  · intro
    /-
      case h.h.h.a.mpr
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      P : CategoryTheory.MorphismProperty W.Localization
      inst✝ : P.IsStableUnderComposition
      hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
      hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
      X Y : W.Localization
      f : Quiver.Hom X Y
      a✝ : Top.top f
      ⊢ P f
    -/
    let G : _ ⥤ W.Localization := Quotient.functor _
    /-
      case h.h.h.a.mpr
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      P : CategoryTheory.MorphismProperty W.Localization
      inst✝ : P.IsStableUnderComposition
      hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
      hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
      X Y : W.Localization
      f : Quiver.Hom X Y
      a✝ : Top.top f
      G : CategoryTheory.Functor (CategoryTheory.Paths (CategoryTheory.Localization. …
      ⊢ P f
    -/
    haveI : G.Full := Quotient.full_functor _
    suffices ∀ (X₁ X₂ : Paths (LocQuiver W)) (f : X₁ ⟶ X₂), P (G.map f) by
      rcases X with ⟨⟨X⟩⟩
      rcases Y with ⟨⟨Y⟩⟩
      simpa only [Functor.map_preimage] using this _ _ (G.preimage f)
    /-
      case h.h.h.a.mpr
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      P : CategoryTheory.MorphismProperty W.Localization
      inst✝ : P.IsStableUnderComposition
      hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
      hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
      X Y : W.Localization
      f : Quiver.Hom X Y
      a✝ : Top.top f
      G : CategoryTheory.Functor (CategoryTheory.Paths (CategoryTheory.Localization. …
      this : G.Full
      ⊢ ∀ (X₁ X₂ : CategoryTheory.Paths (CategoryTheory.Localization.Construction.Lo …
    -/
    intros X₁ X₂ p
    /-
      case h.h.h.a.mpr
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      P : CategoryTheory.MorphismProperty W.Localization
      inst✝ : P.IsStableUnderComposition
      hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
      hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
      X Y : W.Localization
      f : Quiver.Hom X Y
      a✝ : Top.top f
      G : CategoryTheory.Functor (CategoryTheory.Paths (CategoryTheory.Localization. …
      this : G.Full
      X₁ X₂ : CategoryTheory.Paths (CategoryTheory.Localization.Construction.LocQuiv …
      p : Quiver.Hom X₁ X₂
      ⊢ P (G.map p)
    -/
    induction' p with X₂ X₃ p g hp
      /-
        case h.h.h.a.mpr.nil
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        P : CategoryTheory.MorphismProperty W.Localization
        inst✝ : P.IsStableUnderComposition
        hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
        hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
        X Y : W.Localization
        f : Quiver.Hom X Y
        a✝ : Top.top f
        G : CategoryTheory.Functor (CategoryTheory.Paths (CategoryTheory.Localization. …
        this : G.Full
        X₁ X₂ : CategoryTheory.Paths (CategoryTheory.Localization.Construction.LocQuiv …
        ⊢ P (G.map Quiver.Path.nil)
      -/
    · simpa only [Functor.map_id] using hP₁ (𝟙 X₁.obj)
      /-
        🎉 no goals
      -/
      /-
        case h.h.h.a.mpr.cons
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        P : CategoryTheory.MorphismProperty W.Localization
        inst✝ : P.IsStableUnderComposition
        hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
        hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
        X Y : W.Localization
        f : Quiver.Hom X Y
        a✝ : Top.top f
        G : CategoryTheory.Functor (CategoryTheory.Paths (CategoryTheory.Localization. …
        this : G.Full
        X₁ X₂✝ X₂ X₃ : CategoryTheory.Paths (CategoryTheory.Localization.Construction. …
        p : Quiver.Path X₁ X₂
        g : Quiver.Hom X₂ X₃
        hp : P (G.map p)
        ⊢ P (G.map (p.cons g))
      -/
    · let p' : X₁ ⟶X₂ := p
      /-
        case h.h.h.a.mpr.cons
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        P : CategoryTheory.MorphismProperty W.Localization
        inst✝ : P.IsStableUnderComposition
        hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
        hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
        X Y : W.Localization
        f : Quiver.Hom X Y
        a✝ : Top.top f
        G : CategoryTheory.Functor (CategoryTheory.Paths (CategoryTheory.Localization. …
        this : G.Full
        X₁ X₂✝ X₂ X₃ : CategoryTheory.Paths (CategoryTheory.Localization.Construction. …
        p : Quiver.Path X₁ X₂
        g : Quiver.Hom X₂ X₃
        hp : P (G.map p)
        p' : Quiver.Hom X₁ X₂ := p
        ⊢ P (G.map (p.cons g))
      -/
      rw [show p'.cons g = p' ≫ Quiver.Hom.toPath g by rfl, G.map_comp]
      /-
        case h.h.h.a.mpr.cons
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        P : CategoryTheory.MorphismProperty W.Localization
        inst✝ : P.IsStableUnderComposition
        hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
        hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
        X Y : W.Localization
        f : Quiver.Hom X Y
        a✝ : Top.top f
        G : CategoryTheory.Functor (CategoryTheory.Paths (CategoryTheory.Localization. …
        this : G.Full
        X₁ X₂✝ X₂ X₃ : CategoryTheory.Paths (CategoryTheory.Localization.Construction. …
        p : Quiver.Path X₁ X₂
        g : Quiver.Hom X₂ X₃
        hp : P (G.map p)
        p' : Quiver.Hom X₁ X₂ := p
        ⊢ P (CategoryTheory.CategoryStruct.comp (G.map p') (G.map g.toPath))
      -/
      refine P.comp_mem _ _ hp ?_
      /-
        case h.h.h.a.mpr.cons
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        P : CategoryTheory.MorphismProperty W.Localization
        inst✝ : P.IsStableUnderComposition
        hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
        hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
        X Y : W.Localization
        f : Quiver.Hom X Y
        a✝ : Top.top f
        G : CategoryTheory.Functor (CategoryTheory.Paths (CategoryTheory.Localization. …
        this : G.Full
        X₁ X₂✝ X₂ X₃ : CategoryTheory.Paths (CategoryTheory.Localization.Construction. …
        p : Quiver.Path X₁ X₂
        g : Quiver.Hom X₂ X₃
        hp : P (G.map p)
        p' : Quiver.Hom X₁ X₂ := p
        ⊢ P (G.map g.toPath)
      -/
      rcases g with (g | ⟨g, hg⟩)
        /-
          case h.h.h.a.mpr.cons.inl
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          P : CategoryTheory.MorphismProperty W.Localization
          inst✝ : P.IsStableUnderComposition
          hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
          hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
          X Y : W.Localization
          f : Quiver.Hom X Y
          a✝ : Top.top f
          G : CategoryTheory.Functor (CategoryTheory.Paths (CategoryTheory.Localization. …
          this : G.Full
          X₁ X₂✝ X₂ X₃ : CategoryTheory.Paths (CategoryTheory.Localization.Construction. …
          p : Quiver.Path X₁ X₂
          hp : P (G.map p)
          p' : Quiver.Hom X₁ X₂ := p
          g : Quiver.Hom X₂.obj X₃.obj
          ⊢ P (G.map (Quiver.Hom.toPath (Sum.inl g)))
        -/
      · apply hP₁
        /-
          🎉 no goals
        -/
        /-
          case h.h.h.a.mpr.cons.inr.mk
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          P : CategoryTheory.MorphismProperty W.Localization
          inst✝ : P.IsStableUnderComposition
          hP₁ : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), P (W.Q.map f)
          hP₂ : ∀ ⦃X Y : C⦄ (w : Quiver.Hom X Y) (hw : W w), P (CategoryTheory.Localizat …
          X Y : W.Localization
          f : Quiver.Hom X Y
          a✝ : Top.top f
          G : CategoryTheory.Functor (CategoryTheory.Paths (CategoryTheory.Localization. …
          this : G.Full
          X₁ X₂✝ X₂ X₃ : CategoryTheory.Paths (CategoryTheory.Localization.Construction. …
          p : Quiver.Path X₁ X₂
          hp : P (G.map p)
          p' : Quiver.Hom X₁ X₂ := p
          g : Quiver.Hom X₃.obj X₂.obj
          hg : W g
          ⊢ P (G.map (Quiver.Hom.toPath (Sum.inr ⟨g, hg⟩)))
        -/
      · apply hP₂
        /-
          🎉 no goals
        -/


/-- A `MorphismProperty` in `W.Localization` is satisfied by all
morphisms in the localized category if it contains the image of the
morphisms in the original category, if is stable under composition
and if the property is stable by passing to inverses. -/
theorem morphismProperty_is_top' (P : MorphismProperty W.Localization)
    [P.IsStableUnderComposition] (hP₁ : ∀ ⦃X Y : C⦄ (f : X ⟶ Y), P (W.Q.map f))
    (hP₂ : ∀ ⦃X Y : W.Localization⦄ (e : X ≅ Y) (_ : P e.hom), P e.inv) : P = ⊤ :=
  morphismProperty_is_top P hP₁ (fun _ _ w _ => hP₂ _ (hP₁ w))


/-- If `F₁` and `F₂` are functors `W.Localization ⥤ D` and if we have
`τ : W.Q ⋙ F₁ ⟶ W.Q ⋙ F₂`, we shall define a natural transformation `F₁ ⟶ F₂`.
This is the `app` field of this natural transformation. -/
def app (X : W.Localization) : F₁.obj X ⟶ F₂.obj X :=
  eqToHom (congr_arg F₁.obj ((objEquiv W).right_inv X).symm) ≫
    τ.app ((objEquiv W).invFun X) ≫ eqToHom (congr_arg F₂.obj ((objEquiv W).right_inv X))


@[simp]
theorem app_eq (X : C) : (app τ) (W.Q.obj X) = τ.app X := by
  /-
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    D : Type uD
    inst✝ : CategoryTheory.Category.{uD', uD} D
    F₁ F₂ : CategoryTheory.Functor W.Localization D
    τ : Quiver.Hom (W.Q.comp F₁) (W.Q.comp F₂)
    X : C
    ⊢ Eq (CategoryTheory.Localization.Construction.NatTransExtension.app τ (W.Q.ob …
  -/
  simp only [app, eqToHom_refl, comp_id, id_comp]
  /-
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    D : Type uD
    inst✝ : CategoryTheory.Category.{uD', uD} D
    F₁ F₂ : CategoryTheory.Functor W.Localization D
    τ : Quiver.Hom (W.Q.comp F₁) (W.Q.comp F₂)
    X : C
    ⊢ Eq (τ.app ((CategoryTheory.Localization.Construction.objEquiv W).invFun (W.Q …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `F₁` and `F₂` are functors `W.Localization ⥤ D`, a natural transformation `F₁ ⟶ F₂`
can be obtained from a natural transformation `W.Q ⋙ F₁ ⟶ W.Q ⋙ F₂`. -/
@[simps]
def natTransExtension {F₁ F₂ : W.Localization ⥤ D} (τ : W.Q ⋙ F₁ ⟶ W.Q ⋙ F₂) : F₁ ⟶ F₂ where
  app := NatTransExtension.app τ
  naturality := by
    suffices MorphismProperty.naturalityProperty (NatTransExtension.app τ) = ⊤ by
      intro X Y f
      simpa only [← this] using MorphismProperty.top_apply f
    refine morphismProperty_is_top'
      (MorphismProperty.naturalityProperty (NatTransExtension.app τ))
      ?_ (MorphismProperty.naturalityProperty.stableUnderInverse _)
    /-
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G : CategoryTheory.Functor C D
      hG : W.IsInvertedBy G
      F₁ F₂ : CategoryTheory.Functor W.Localization D
      τ : Quiver.Hom (W.Q.comp F₁) (W.Q.comp F₂)
      ⊢ ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), CategoryTheory.MorphismProperty.naturality …
    -/
    intros X Y f
    /-
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G : CategoryTheory.Functor C D
      hG : W.IsInvertedBy G
      F₁ F₂ : CategoryTheory.Functor W.Localization D
      τ : Quiver.Hom (W.Q.comp F₁) (W.Q.comp F₂)
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.MorphismProperty.naturalityProperty (CategoryTheory.Localizat …
    -/
    dsimp
    /-
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G : CategoryTheory.Functor C D
      hG : W.IsInvertedBy G
      F₁ F₂ : CategoryTheory.Functor W.Localization D
      τ : Quiver.Hom (W.Q.comp F₁) (W.Q.comp F₂)
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₁.map (W.Q.map f)) (CategoryTheory. …
    -/
    simpa only [NatTransExtension.app_eq] using τ.naturality f
    /-
      🎉 no goals
    -/


@[simp]
theorem natTransExtension_hcomp {F G : W.Localization ⥤ D} (τ : W.Q ⋙ F ⟶ W.Q ⋙ G) :
                                          /-
                                            C : Type uC
                                            inst✝¹ : CategoryTheory.Category.{uC', uC} C
                                            W : CategoryTheory.MorphismProperty C
                                            D : Type uD
                                            inst✝ : CategoryTheory.Category.{uD', uD} D
                                            F G : CategoryTheory.Functor W.Localization D
                                            τ : Quiver.Hom (W.Q.comp F) (W.Q.comp G)
                                            ⊢ Eq (CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id W.Q) (Ca …
                                          -/
    𝟙 W.Q ◫ natTransExtension τ = τ := by aesop_cat
                                          /-
                                            🎉 no goals
                                          -/


theorem natTrans_hcomp_injective {F G : W.Localization ⥤ D} {τ₁ τ₂ : F ⟶ G}
    (h : 𝟙 W.Q ◫ τ₁ = 𝟙 W.Q ◫ τ₂) : τ₁ = τ₂ := by
  /-
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    D : Type uD
    inst✝ : CategoryTheory.Category.{uD', uD} D
    F G : CategoryTheory.Functor W.Localization D
    τ₁ τ₂ : Quiver.Hom F G
    h : Eq (CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id W.Q) τ …
    ⊢ Eq τ₁ τ₂
  -/
  ext X
  /-
    case w.h
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    D : Type uD
    inst✝ : CategoryTheory.Category.{uD', uD} D
    F G : CategoryTheory.Functor W.Localization D
    τ₁ τ₂ : Quiver.Hom F G
    h : Eq (CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id W.Q) τ …
    X : W.Localization
    ⊢ Eq (τ₁.app X) (τ₂.app X)
  -/
  have eq := (objEquiv W).right_inv X
  /-
    case w.h
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    D : Type uD
    inst✝ : CategoryTheory.Category.{uD', uD} D
    F G : CategoryTheory.Functor W.Localization D
    τ₁ τ₂ : Quiver.Hom F G
    h : Eq (CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id W.Q) τ …
    X : W.Localization
    eq : Eq ((CategoryTheory.Localization.Construction.objEquiv W).toFun ((Categor …
    ⊢ Eq (τ₁.app X) (τ₂.app X)
  -/
  simp only [objEquiv] at eq
  /-
    case w.h
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{uC', uC} C
    W : CategoryTheory.MorphismProperty C
    D : Type uD
    inst✝ : CategoryTheory.Category.{uD', uD} D
    F G : CategoryTheory.Functor W.Localization D
    τ₁ τ₂ : Quiver.Hom F G
    h : Eq (CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id W.Q) τ …
    X : W.Localization
    eq : Eq (W.Q.obj X.as.obj) X
    ⊢ Eq (τ₁.app X) (τ₂.app X)
  -/
  rw [← eq, ← NatTrans.id_hcomp_app, ← NatTrans.id_hcomp_app, h]
  /-
    🎉 no goals
  -/


/-- The functor `(W.Localization ⥤ D) ⥤ (W.FunctorsInverting D)` induced by the
composition with `W.Q : C ⥤ W.Localization`. -/
@[simps!]
def functor : (W.Localization ⥤ D) ⥤ W.FunctorsInverting D :=
  FullSubcategory.lift _ ((whiskeringLeft _ _ D).obj W.Q) fun _ =>
    MorphismProperty.IsInvertedBy.of_comp W W.Q W.Q_inverts _


/-- The function `(W.FunctorsInverting D) ⥤ (W.Localization ⥤ D)` induced by
`Construction.lift`. -/
@[simps!]
def inverse : W.FunctorsInverting D ⥤ W.Localization ⥤ D where
  obj G := lift G.obj G.property
                                          /-
                                            C : Type uC
                                            inst✝¹ : CategoryTheory.Category.{uC', uC} C
                                            W : CategoryTheory.MorphismProperty C
                                            D : Type uD
                                            inst✝ : CategoryTheory.Category.{uD', uD} D
                                            G : CategoryTheory.Functor C D
                                            hG : W.IsInvertedBy G
                                            X✝ Y✝ : W.FunctorsInverting D
                                            τ : Quiver.Hom X✝ Y✝
                                            ⊢ Eq (W.Q.comp ((fun G => CategoryTheory.Localization.Construction.lift G.obj  …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  map τ := natTransExtension (eqToHom (by rw [fac]) ≫ τ ≫ eqToHom (by rw [fac]))
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  map_id G :=
    natTrans_hcomp_injective
      (by
        /-
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G✝ : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G✝
          G : W.FunctorsInverting D
          ⊢ Eq (CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id W.Q) ({  …
        -/
        rw [natTransExtension_hcomp]
        /-
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G✝ : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G✝
          G : W.FunctorsInverting D
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
        -/
        ext X
        simp only [NatTrans.comp_app, eqToHom_app, eqToHom_refl, comp_id, id_comp,
          NatTrans.hcomp_id_app, NatTrans.id_app, Functor.map_id]
        /-
          case w.h
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G✝ : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G✝
          G : W.FunctorsInverting D
          X : C
          ⊢ Eq ((CategoryTheory.CategoryStruct.id G).app X) (CategoryTheory.CategoryStru …
        -/
        rfl)
        /-
          🎉 no goals
        -/
  map_comp τ₁ τ₂ :=
    natTrans_hcomp_injective
      (by
        /-
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          X✝ Y✝ Z✝ : W.FunctorsInverting D
          τ₁ : Quiver.Hom X✝ Y✝
          τ₂ : Quiver.Hom Y✝ Z✝
          ⊢ Eq (CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id W.Q) ({  …
        -/
        ext X
        simp only [natTransExtension_hcomp, NatTrans.comp_app, eqToHom_app, eqToHom_refl,
          id_comp, comp_id, NatTrans.hcomp_app, NatTrans.id_app, Functor.map_id,
          natTransExtension_app, NatTransExtension.app_eq]
        /-
          case w.h
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          X✝ Y✝ Z✝ : W.FunctorsInverting D
          τ₁ : Quiver.Hom X✝ Y✝
          τ₂ : Quiver.Hom Y✝ Z✝
          X : C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp τ₁ τ₂).app X) (CategoryTheory.Catego …
        -/
        rfl)
        /-
          🎉 no goals
        -/


/-- The unit isomorphism of the equivalence of categories `whiskeringLeftEquivalence W D`. -/
@[simps!]
def unitIso : 𝟭 (W.Localization ⥤ D) ≅ functor W D ⋙ inverse W D :=
  eqToIso
    (by
      /-
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G : CategoryTheory.Functor C D
        hG : W.IsInvertedBy G
        ⊢ Eq (CategoryTheory.Functor.id (CategoryTheory.Functor W.Localization D)) ((C …
      -/
      refine Functor.ext (fun G => ?_) fun G₁ G₂ τ => ?_
        /-
          case refine_1
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G✝ : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G✝
          G : CategoryTheory.Functor W.Localization D
          ⊢ Eq ((CategoryTheory.Functor.id (CategoryTheory.Functor W.Localization D)).ob …
        -/
      · apply uniq
        /-
          case refine_1.h
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G✝ : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G✝
          G : CategoryTheory.Functor W.Localization D
          ⊢ Eq (W.Q.comp ((CategoryTheory.Functor.id (CategoryTheory.Functor W.Localizat …
        -/
        dsimp [Functor]
        /-
          case refine_1.h
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G✝ : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G✝
          G : CategoryTheory.Functor W.Localization D
          ⊢ Eq (W.Q.comp G) (W.Q.comp ((CategoryTheory.Localization.Construction.Whisker …
        -/
        erw [fac]
        /-
          case refine_1.h
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G✝ : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G✝
          G : CategoryTheory.Functor W.Localization D
          ⊢ Eq (W.Q.comp G) ((CategoryTheory.Localization.Construction.WhiskeringLeftEqu …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          G₁ G₂ : CategoryTheory.Functor W.Localization D
          τ : Quiver.Hom G₁ G₂
          ⊢ Eq ((CategoryTheory.Functor.id (CategoryTheory.Functor W.Localization D)).ma …
        -/
      · apply natTrans_hcomp_injective
        /-
          case refine_2.h
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          G₁ G₂ : CategoryTheory.Functor W.Localization D
          τ : Quiver.Hom G₁ G₂
          ⊢ Eq (CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id W.Q) ((C …
        -/
        ext X
        /-
          case refine_2.h.w.h
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          G₁ G₂ : CategoryTheory.Functor W.Localization D
          τ : Quiver.Hom G₁ G₂
          X : C
          ⊢ Eq ((CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id W.Q) (( …
        -/
        simp)
        /-
          🎉 no goals
        -/


/-- The counit isomorphism of the equivalence of categories `WhiskeringLeftEquivalence W D`. -/
@[simps!]
def counitIso : inverse W D ⋙ functor W D ≅ 𝟭 (W.FunctorsInverting D) :=
  eqToIso
    (by
      /-
        C : Type uC
        inst✝¹ : CategoryTheory.Category.{uC', uC} C
        W : CategoryTheory.MorphismProperty C
        D : Type uD
        inst✝ : CategoryTheory.Category.{uD', uD} D
        G : CategoryTheory.Functor C D
        hG : W.IsInvertedBy G
        ⊢ Eq ((CategoryTheory.Localization.Construction.WhiskeringLeftEquivalence.inve …
      -/
      refine Functor.ext ?_ ?_
        /-
          case refine_1
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          ⊢ ∀ (X : W.FunctorsInverting D), Eq (((CategoryTheory.Localization.Constructio …
        -/
      · rintro ⟨G, hG⟩
        /-
          case refine_1.mk
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G✝ : CategoryTheory.Functor C D
          hG✝ : W.IsInvertedBy G✝
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          ⊢ Eq (((CategoryTheory.Localization.Construction.WhiskeringLeftEquivalence.inv …
        -/
        ext
        /-
          case refine_1.mk.h
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G✝ : CategoryTheory.Functor C D
          hG✝ : W.IsInvertedBy G✝
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          ⊢ Eq (((CategoryTheory.Localization.Construction.WhiskeringLeftEquivalence.inv …
        -/
        exact fac G hG
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          ⊢ ∀ (X Y : W.FunctorsInverting D) (f : Quiver.Hom X Y), Eq (((CategoryTheory.L …
        -/
      · rintro ⟨G₁, hG₁⟩ ⟨G₂, hG₂⟩ f
        /-
          case refine_2.mk.mk
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          G₁ : CategoryTheory.Functor C D
          hG₁ : W.IsInvertedBy G₁
          G₂ : CategoryTheory.Functor C D
          hG₂ : W.IsInvertedBy G₂
          f : Quiver.Hom { obj := G₁, property := hG₁ } { obj := G₂, property := hG₂ }
          ⊢ Eq (((CategoryTheory.Localization.Construction.WhiskeringLeftEquivalence.inv …
        -/
        ext
        /-
          case refine_2.mk.mk.h.h
          C : Type uC
          inst✝¹ : CategoryTheory.Category.{uC', uC} C
          W : CategoryTheory.MorphismProperty C
          D : Type uD
          inst✝ : CategoryTheory.Category.{uD', uD} D
          G : CategoryTheory.Functor C D
          hG : W.IsInvertedBy G
          G₁ : CategoryTheory.Functor C D
          hG₁ : W.IsInvertedBy G₁
          G₂ : CategoryTheory.Functor C D
          hG₂ : W.IsInvertedBy G₂
          f : Quiver.Hom { obj := G₁, property := hG₁ } { obj := G₂, property := hG₂ }
          x✝ : C
          ⊢ Eq ((((CategoryTheory.Localization.Construction.WhiskeringLeftEquivalence.in …
        -/
        apply NatTransExtension.app_eq)
        /-
          🎉 no goals
        -/


/-- The equivalence of categories `(W.localization ⥤ D) ≌ (W.FunctorsInverting D)`
induced by the composition with `W.Q : C ⥤ W.localization`. -/
def whiskeringLeftEquivalence : W.Localization ⥤ D ≌ W.FunctorsInverting D where
  functor := WhiskeringLeftEquivalence.functor W D
  inverse := WhiskeringLeftEquivalence.inverse W D
  unitIso := WhiskeringLeftEquivalence.unitIso W D
  counitIso := WhiskeringLeftEquivalence.counitIso W D
  functor_unitIso_comp F := by
    /-
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G : CategoryTheory.Functor C D
      hG : W.IsInvertedBy G
      F : CategoryTheory.Functor W.Localization D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Localization.Constru …
    -/
    ext
    simp only [WhiskeringLeftEquivalence.unitIso_hom, eqToHom_app, eqToHom_refl,
      WhiskeringLeftEquivalence.counitIso_hom, eqToHom_map, eqToHom_trans]
    /-
      case h.h
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{uC', uC} C
      W : CategoryTheory.MorphismProperty C
      D : Type uD
      inst✝ : CategoryTheory.Category.{uD', uD} D
      G : CategoryTheory.Functor C D
      hG : W.IsInvertedBy G
      F : CategoryTheory.Functor W.Localization D
      x✝ : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.id ((CategoryTheory.Localization.Construc …
    -/
    rfl
    /-
      🎉 no goals
    -/


