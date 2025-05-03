/-- If `P` is stable under composition and `f : X ⟶ Y` satisfies `P`,
this is the functor `P.Over Q X ⥤ P.Over Q Y` given by composing with `f`. -/
@[simps! obj_left obj_hom map_left]
def Over.map : P.Over Q X ⥤ P.Over Q Y :=
  Comma.mapRight _ (Discrete.natTrans fun _ ↦ f) <| fun X ↦ P.comp_mem _ _ X.prop hPf


lemma Over.map_comp {X Y Z : T} {f : X ⟶ Y} (hf : P f) {g : Y ⟶ Z} (hg : P g) :
    map Q (P.comp_mem f g hf hg) = map Q hf ⋙ map Q hg := by
  /-
    T : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    inst✝¹ : Q.IsMultiplicative
    inst✝ : P.IsStableUnderComposition
    X Y Z : T
    f : Quiver.Hom X Y
    hf : P f
    g : Quiver.Hom Y Z
    hg : P g
    ⊢ Eq (CategoryTheory.MorphismProperty.Over.map Q ⋯) ((CategoryTheory.MorphismP …
  -/
  fapply Functor.ext
    /-
      case h_obj
      T : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} T
      P Q : CategoryTheory.MorphismProperty T
      inst✝¹ : Q.IsMultiplicative
      inst✝ : P.IsStableUnderComposition
      X Y Z : T
      f : Quiver.Hom X Y
      hf : P f
      g : Quiver.Hom Y Z
      hg : P g
      ⊢ ∀ (X_1 : P.Over Q X), Eq ((CategoryTheory.MorphismProperty.Over.map Q ⋯).obj …
    -/
  · simp [map, Comma.mapRight, CategoryTheory.Comma.mapRight, Comma.lift]
    /-
      🎉 no goals
    -/
    /-
      case h_map
      T : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} T
      P Q : CategoryTheory.MorphismProperty T
      inst✝¹ : Q.IsMultiplicative
      inst✝ : P.IsStableUnderComposition
      X Y Z : T
      f : Quiver.Hom X Y
      hf : P f
      g : Quiver.Hom Y Z
      hg : P g
      ⊢ autoParam (∀ (X_1 Y_1 : P.Over Q X) (f_1 : Quiver.Hom X_1 Y_1), Eq ((Categor …
    -/
  · intro U V k
    /-
      case h_map
      T : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} T
      P Q : CategoryTheory.MorphismProperty T
      inst✝¹ : Q.IsMultiplicative
      inst✝ : P.IsStableUnderComposition
      X Y Z : T
      f : Quiver.Hom X Y
      hf : P f
      g : Quiver.Hom Y Z
      hg : P g
      U V : P.Over Q X
      k : Quiver.Hom U V
      ⊢ Eq ((CategoryTheory.MorphismProperty.Over.map Q ⋯).map k) (CategoryTheory.Ca …
    -/
    ext
    /-
      case h_map.h
      T : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} T
      P Q : CategoryTheory.MorphismProperty T
      inst✝¹ : Q.IsMultiplicative
      inst✝ : P.IsStableUnderComposition
      X Y Z : T
      f : Quiver.Hom X Y
      hf : P f
      g : Quiver.Hom Y Z
      hg : P g
      U V : P.Over Q X
      k : Quiver.Hom U V
      ⊢ Eq ((CategoryTheory.MorphismProperty.Over.map Q ⋯).map k).left (CategoryTheo …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- `Over.map` commutes with composition. -/
@[simps! hom_app_left inv_app_left]
def Over.mapComp {X Y Z : T} {f : X ⟶ Y} (hf : P f) {g : Y ⟶ Z} (hg : P g) [Q.RespectsIso] :
    map Q (P.comp_mem f g hf hg) ≅ map Q hf ⋙ map Q hg :=
                               /-
                                 T : Type u_1
                                 inst✝³ : CategoryTheory.Category.{?u.19349, u_1} T
                                 P Q : CategoryTheory.MorphismProperty T
                                 inst✝² : Q.IsMultiplicative
                                 X✝¹ Y✝ : T
                                 f✝ : Quiver.Hom X✝¹ Y✝
                                 inst✝¹ : P.IsStableUnderComposition
                                 hPf : P f✝
                                 X✝ Y Z : T
                                 f : Quiver.Hom X✝ Y
                                 hf : P f
                                 g : Quiver.Hom Y Z
                                 hg : P g
                                 inst✝ : Q.RespectsIso
                                 X : P.Over Q X✝
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl ((CategoryTh …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents (fun X ↦ Over.isoMk (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- If `P` and `Q` are stable under base change and pullbacks exist in `T`,
this is the functor `P.Over Q Y ⥤ P.Over Q X` given by base change along `f`. -/
@[simps! obj_left obj_hom map_left]
noncomputable def Over.pullback : P.Over Q Y ⥤ P.Over Q X where
  obj A :=
    { __ := (CategoryTheory.Over.pullback f).obj A.toComma
      prop := P.pullback_snd _ _ A.prop }
  map {A B} g :=
    { __ := (CategoryTheory.Over.pullback f).map g.toCommaMorphism
      prop_hom_left := Q.baseChange_map f g.toCommaMorphism g.prop_hom_left
      prop_hom_right := trivial }


/-- `Over.pullback` commutes with composition. -/
@[simps! hom_app_left inv_app_left]
noncomputable def Over.pullbackComp [Q.RespectsIso] {X Y Z : T} (f : X ⟶ Y) (g : Y ⟶ Z) :
    Over.pullback P Q (f ≫ g) ≅ Over.pullback P Q g ⋙ Over.pullback P Q f :=
  /-
    T : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.35056, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    inst✝⁴ : Q.IsMultiplicative
    X✝ Y✝ : T
    f✝ : Quiver.Hom X✝ Y✝
    inst✝³ : CategoryTheory.Limits.HasPullbacks T
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : Q.IsStableUnderBaseChange
    inst✝ : Q.RespectsIso
    X Y Z : T
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ ∀ {X_1 Y_1 : P.Over Q Z} (f_1 : Quiver.Hom X_1 Y_1), Eq (CategoryTheory.Cate …
  -/
                                                                          /-
                                                                            T : Type u_1
                                                                            inst✝⁵ : CategoryTheory.Category.{?u.35056, u_1} T
                                                                            P Q : CategoryTheory.MorphismProperty T
                                                                            inst✝⁴ : Q.IsMultiplicative
                                                                            X✝¹ Y✝ : T
                                                                            f✝ : Quiver.Hom X✝¹ Y✝
                                                                            inst✝³ : CategoryTheory.Limits.HasPullbacks T
                                                                            inst✝² : P.IsStableUnderBaseChange
                                                                            inst✝¹ : Q.IsStableUnderBaseChange
                                                                            inst✝ : Q.RespectsIso
                                                                            X✝ Y Z : T
                                                                            f : Quiver.Hom X✝ Y
                                                                            g : Quiver.Hom Y Z
                                                                            X : P.Over Q Z
                                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackLeftPu …
                                                                          -/
  NatIso.ofComponents
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  /-
    🎉 no goals
  -/
    (fun X ↦ Over.isoMk ((pullbackLeftPullbackSndIso X.hom g f).symm) (by simp))


lemma Over.pullbackComp_left_fst_fst [Q.RespectsIso] {X Y Z : T} (f : X ⟶ Y) (g : Y ⟶ Z)
    (A : P.Over Q Z) :
    ((Over.pullbackComp f g).hom.app A).left ≫
      pullback.fst (pullback.snd A.hom g) f ≫ pullback.fst A.hom g =
        pullback.fst A.hom (f ≫ g) := by
  /-
    T : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    inst✝⁴ : Q.IsMultiplicative
    inst✝³ : CategoryTheory.Limits.HasPullbacks T
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : Q.IsStableUnderBaseChange
    inst✝ : Q.RespectsIso
    X Y Z : T
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    A : P.Over Q Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MorphismProperty.Ove …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `f = g`, then base change along `f` is naturally isomorphic to base change along `g`. -/
noncomputable def Over.pullbackCongr {X Y : T} {f g : X ⟶ Y} (h : f = g) :
    Over.pullback P Q f ≅ Over.pullback P Q g :=
                                           /-
                                             T : Type u_1
                                             inst✝⁴ : CategoryTheory.Category.{?u.60087, u_1} T
                                             P Q : CategoryTheory.MorphismProperty T
                                             inst✝³ : Q.IsMultiplicative
                                             X✝¹ Y✝ : T
                                             f✝ : Quiver.Hom X✝¹ Y✝
                                             inst✝² : CategoryTheory.Limits.HasPullbacks T
                                             inst✝¹ : P.IsStableUnderBaseChange
                                             inst✝ : Q.IsStableUnderBaseChange
                                             X✝ Y : T
                                             f g : Quiver.Hom X✝ Y
                                             h : Eq f g
                                             X : P.Over Q Y
                                             ⊢ Eq ((CategoryTheory.MorphismProperty.Over.pullback P Q f).obj X) ((CategoryT …
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  NatIso.ofComponents (fun X ↦ eqToIso (by rw [h]))
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma Over.pullbackCongr_hom_app_left_fst {X Y : T} {f g : X ⟶ Y} (h : f = g) (A : P.Over Q Y) :
    ((Over.pullbackCongr h).hom.app A).left ≫ pullback.fst A.hom g =
      pullback.fst A.hom f := by
  /-
    T : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    inst✝³ : Q.IsMultiplicative
    inst✝² : CategoryTheory.Limits.HasPullbacks T
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : Q.IsStableUnderBaseChange
    X Y : T
    f g : Quiver.Hom X Y
    h : Eq f g
    A : P.Over Q Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MorphismProperty.Ove …
  -/
  subst h
  /-
    T : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    inst✝³ : Q.IsMultiplicative
    inst✝² : CategoryTheory.Limits.HasPullbacks T
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : Q.IsStableUnderBaseChange
    X Y : T
    f : Quiver.Hom X Y
    A : P.Over Q Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MorphismProperty.Ove …
  -/
  simp [pullbackCongr]
  /-
    🎉 no goals
  -/


/-- `P.Over.map` is left adjoint to `P.Over.pullback` if `f` satisfies `P`. -/
noncomputable def Over.mapPullbackAdj [Q.HasOfPostcompProperty Q] (hPf : P f) (hQf : Q f) :
    Over.map Q hPf ⊣ Over.pullback P Q f :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun A B ↦
        { toFun := fun g ↦
                                                         /-
                                                           T : Type u_1
                                                           inst✝⁶ : CategoryTheory.Category.{?u.67008, u_1} T
                                                           P Q : CategoryTheory.MorphismProperty T
                                                           inst✝⁵ : Q.IsMultiplicative
                                                           X Y : T
                                                           f : Quiver.Hom X Y
                                                           inst✝⁴ : P.IsStableUnderComposition
                                                           inst✝³ : P.IsStableUnderBaseChange
                                                           inst✝² : Q.IsStableUnderBaseChange
                                                           inst✝¹ : CategoryTheory.Limits.HasPullbacks T
                                                           inst✝ : Q.HasOfPostcompProperty Q
                                                           hPf : P f
                                                           hQf : Q f
                                                           A : P.Over Q X
                                                           B : P.Over Q Y
                                                           g : Quiver.Hom ((CategoryTheory.MorphismProperty.Over.map Q hPf).obj A) B
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp g.left B.hom) (CategoryTheory.Categor …
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
            Over.homMk (pullback.lift g.left A.hom <| by simp) (by simp) <| by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
              /-
                T : Type u_1
                inst✝⁶ : CategoryTheory.Category.{?u.67008, u_1} T
                P Q : CategoryTheory.MorphismProperty T
                inst✝⁵ : Q.IsMultiplicative
                X Y : T
                f : Quiver.Hom X Y
                inst✝⁴ : P.IsStableUnderComposition
                inst✝³ : P.IsStableUnderBaseChange
                inst✝² : Q.IsStableUnderBaseChange
                inst✝¹ : CategoryTheory.Limits.HasPullbacks T
                inst✝ : Q.HasOfPostcompProperty Q
                hPf : P f
                hQf : Q f
                A : P.Over Q X
                B : P.Over Q Y
                g : Quiver.Hom ((CategoryTheory.MorphismProperty.Over.map Q hPf).obj A) B
                ⊢ Q (CategoryTheory.Limits.pullback.lift g.left A.hom ⋯)
              -/
              apply Q.of_postcomp (W' := Q)
                /-
                  case hg
                  T : Type u_1
                  inst✝⁶ : CategoryTheory.Category.{?u.67008, u_1} T
                  P Q : CategoryTheory.MorphismProperty T
                  inst✝⁵ : Q.IsMultiplicative
                  X Y : T
                  f : Quiver.Hom X Y
                  inst✝⁴ : P.IsStableUnderComposition
                  inst✝³ : P.IsStableUnderBaseChange
                  inst✝² : Q.IsStableUnderBaseChange
                  inst✝¹ : CategoryTheory.Limits.HasPullbacks T
                  inst✝ : Q.HasOfPostcompProperty Q
                  hPf : P f
                  hQf : Q f
                  A : P.Over Q X
                  B : P.Over Q Y
                  g : Quiver.Hom ((CategoryTheory.MorphismProperty.Over.map Q hPf).obj A) B
                  ⊢ Q ?g
                -/
              · exact Q.pullback_fst B.hom f hQf
                /-
                  🎉 no goals
                -/
                /-
                  case hfg
                  T : Type u_1
                  inst✝⁶ : CategoryTheory.Category.{?u.67008, u_1} T
                  P Q : CategoryTheory.MorphismProperty T
                  inst✝⁵ : Q.IsMultiplicative
                  X Y : T
                  f : Quiver.Hom X Y
                  inst✝⁴ : P.IsStableUnderComposition
                  inst✝³ : P.IsStableUnderBaseChange
                  inst✝² : Q.IsStableUnderBaseChange
                  inst✝¹ : CategoryTheory.Limits.HasPullbacks T
                  inst✝ : Q.HasOfPostcompProperty Q
                  hPf : P f
                  hQf : Q f
                  A : P.Over Q X
                  B : P.Over Q Y
                  g : Quiver.Hom ((CategoryTheory.MorphismProperty.Over.map Q hPf).obj A) B
                  ⊢ Q (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift g …
                -/
              · simpa using g.prop_hom_left
                /-
                  🎉 no goals
                -/
          invFun := fun h ↦ Over.homMk (h.left ≫ pullback.fst B.hom f)
            (by
              simp only [map_obj_left, Functor.const_obj_obj, pullback_obj_left, Functor.id_obj,
                Category.assoc, pullback.condition, map_obj_hom, ← pullback_obj_hom, Over.w_assoc])
            (Q.comp_mem _ _ h.prop_hom_left (Q.pullback_fst _ _ hQf))
                         /-
                           T : Type u_1
                           inst✝⁶ : CategoryTheory.Category.{?u.67008, u_1} T
                           P Q : CategoryTheory.MorphismProperty T
                           inst✝⁵ : Q.IsMultiplicative
                           X Y : T
                           f : Quiver.Hom X Y
                           inst✝⁴ : P.IsStableUnderComposition
                           inst✝³ : P.IsStableUnderBaseChange
                           inst✝² : Q.IsStableUnderBaseChange
                           inst✝¹ : CategoryTheory.Limits.HasPullbacks T
                           inst✝ : Q.HasOfPostcompProperty Q
                           hPf : P f
                           hQf : Q f
                           A : P.Over Q X
                           B : P.Over Q Y
                           ⊢ Function.LeftInverse (fun h => CategoryTheory.MorphismProperty.Over.homMk (C …
                         -/
          left_inv := by aesop_cat
                         /-
                           🎉 no goals
                         -/
          right_inv := fun h ↦ by
            /-
              T : Type u_1
              inst✝⁶ : CategoryTheory.Category.{?u.67008, u_1} T
              P Q : CategoryTheory.MorphismProperty T
              inst✝⁵ : Q.IsMultiplicative
              X Y : T
              f : Quiver.Hom X Y
              inst✝⁴ : P.IsStableUnderComposition
              inst✝³ : P.IsStableUnderBaseChange
              inst✝² : Q.IsStableUnderBaseChange
              inst✝¹ : CategoryTheory.Limits.HasPullbacks T
              inst✝ : Q.HasOfPostcompProperty Q
              hPf : P f
              hQf : Q f
              A : P.Over Q X
              B : P.Over Q Y
              h : Quiver.Hom A ((CategoryTheory.MorphismProperty.Over.pullback P Q f).obj B)
              ⊢ Eq ((fun g => CategoryTheory.MorphismProperty.Over.homMk (CategoryTheory.Lim …
            -/
            ext
            /-
              case h
              T : Type u_1
              inst✝⁶ : CategoryTheory.Category.{?u.67008, u_1} T
              P Q : CategoryTheory.MorphismProperty T
              inst✝⁵ : Q.IsMultiplicative
              X Y : T
              f : Quiver.Hom X Y
              inst✝⁴ : P.IsStableUnderComposition
              inst✝³ : P.IsStableUnderBaseChange
              inst✝² : Q.IsStableUnderBaseChange
              inst✝¹ : CategoryTheory.Limits.HasPullbacks T
              inst✝ : Q.HasOfPostcompProperty Q
              hPf : P f
              hQf : Q f
              A : P.Over Q X
              B : P.Over Q Y
              h : Quiver.Hom A ((CategoryTheory.MorphismProperty.Over.pullback P Q f).obj B)
              ⊢ Eq ((fun g => CategoryTheory.MorphismProperty.Over.homMk (CategoryTheory.Lim …
            -/
            dsimp
            /-
              case h
              T : Type u_1
              inst✝⁶ : CategoryTheory.Category.{?u.67008, u_1} T
              P Q : CategoryTheory.MorphismProperty T
              inst✝⁵ : Q.IsMultiplicative
              X Y : T
              f : Quiver.Hom X Y
              inst✝⁴ : P.IsStableUnderComposition
              inst✝³ : P.IsStableUnderBaseChange
              inst✝² : Q.IsStableUnderBaseChange
              inst✝¹ : CategoryTheory.Limits.HasPullbacks T
              inst✝ : Q.HasOfPostcompProperty Q
              hPf : P f
              hQf : Q f
              A : P.Over Q X
              B : P.Over Q Y
              h : Quiver.Hom A ((CategoryTheory.MorphismProperty.Over.pullback P Q f).obj B)
              ⊢ Eq (CategoryTheory.Limits.pullback.lift (CategoryTheory.CategoryStruct.comp  …
            -/
            ext
              /-
                case h.h₀
                T : Type u_1
                inst✝⁶ : CategoryTheory.Category.{?u.67008, u_1} T
                P Q : CategoryTheory.MorphismProperty T
                inst✝⁵ : Q.IsMultiplicative
                X Y : T
                f : Quiver.Hom X Y
                inst✝⁴ : P.IsStableUnderComposition
                inst✝³ : P.IsStableUnderBaseChange
                inst✝² : Q.IsStableUnderBaseChange
                inst✝¹ : CategoryTheory.Limits.HasPullbacks T
                inst✝ : Q.HasOfPostcompProperty Q
                hPf : P f
                hQf : Q f
                A : P.Over Q X
                B : P.Over Q Y
                h : Quiver.Hom A ((CategoryTheory.MorphismProperty.Over.pullback P Q f).obj B)
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
              -/
            · simp
              /-
                🎉 no goals
              -/
              /-
                case h.h₁
                T : Type u_1
                inst✝⁶ : CategoryTheory.Category.{?u.67008, u_1} T
                P Q : CategoryTheory.MorphismProperty T
                inst✝⁵ : Q.IsMultiplicative
                X Y : T
                f : Quiver.Hom X Y
                inst✝⁴ : P.IsStableUnderComposition
                inst✝³ : P.IsStableUnderBaseChange
                inst✝² : Q.IsStableUnderBaseChange
                inst✝¹ : CategoryTheory.Limits.HasPullbacks T
                inst✝ : Q.HasOfPostcompProperty Q
                hPf : P f
                hQf : Q f
                A : P.Over Q X
                B : P.Over Q Y
                h : Quiver.Hom A ((CategoryTheory.MorphismProperty.Over.pullback P Q f).obj B)
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
              -/
            · simpa using h.w.symm } }
              /-
                🎉 no goals
              -/


