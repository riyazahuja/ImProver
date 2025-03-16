/-- Bundle an `S`-scheme with `P` into an object of `P.Over ⊤ S`. -/
abbrev asOverProp (X : Scheme.{u}) (S : Scheme.{u}) [X.Over S] (h : P (X ↘ S)) : P.Over ⊤ S :=
  ⟨X.asOver S, h⟩


/-- Bundle an `S`-morphism of `S`-scheme with `P` into a morphism in `P.Over ⊤ S`. -/
abbrev Hom.asOverProp {X Y : Scheme.{u}} (f : X.Hom Y) (S : Scheme.{u}) [X.Over S] [Y.Over S]
    [f.IsOver S] {hX : P (X ↘ S)} {hY : P (Y ↘ S)} : X.asOverProp S hX ⟶ Y.asOverProp S hY :=
  ⟨f.asOver S, trivial, trivial⟩


/-- A `P`-cover of a scheme `X` over `S` is a cover, where the components are over `S` and the
component maps commute with the structure morphisms. -/
protected class Cover.Over {P : MorphismProperty Scheme.{u}} {X : Scheme.{u}} [X.Over S]
    (𝒰 : X.Cover P) where
  over (j : 𝒰.J) : (𝒰.obj j).Over S := by infer_instance
  isOver_map (j : 𝒰.J) : (𝒰.map j).IsOver S := by infer_instance


instance [P.ContainsIdentities] [P.RespectsIso] {X Y : Scheme.{u}} (f : X ⟶ Y) [X.Over S] [Y.Over S]
    [f.IsOver S] [IsIso f] : (coverOfIsIso (P := P) f).Over S where
  over _ := inferInstanceAs <| X.Over S
  isOver_map _ := inferInstanceAs <| f.IsOver S


/-- The pullback of a cover of `S`-schemes along a morphism of `S`-schemes. This is not
definitionally equal to `AlgebraicGeometry.Scheme.Cover.pullbackCover`, as here we take
the pullback in `Over S`, whose underlying scheme is only isomorphic but not equal to the
pullback in `Scheme`. -/
@[simps]
def Cover.pullbackCoverOver : W.Cover P where
  J := 𝒰.J
  obj x := (pullback (f.asOver S) ((𝒰.map x).asOver S)).left
  map x := (pullback.fst (f.asOver S) ((𝒰.map x).asOver S)).left
  f x := 𝒰.f (f.base x)
  covers x := (mem_range_iff_of_surjective ((𝒰.pullbackCover f).map (𝒰.f (f.base x))) _
    ((PreservesPullback.iso (Over.forget S) (f.asOver S) ((𝒰.map _).asOver S)).inv)
    (PreservesPullback.iso_inv_fst _ _ _) x).mp ((𝒰.pullbackCover f).covers x)
  map_prop j := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁵ : P.IsStableUnderBaseChange
      inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝³ : W.Over S
      inst✝² : X.Over S
      inst✝¹ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      j : 𝒰.J
      ⊢ P ((fun x => (CategoryTheory.Limits.pullback.fst (AlgebraicGeometry.Scheme.H …
    -/
    dsimp only
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁵ : P.IsStableUnderBaseChange
      inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝³ : W.Over S
      inst✝² : X.Over S
      inst✝¹ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      j : 𝒰.J
      ⊢ P (CategoryTheory.Limits.pullback.fst (AlgebraicGeometry.Scheme.Hom.asOver f …
    -/
    rw [← Over.forget_map, ← PreservesPullback.iso_hom_fst, P.cancel_left_of_respectsIso]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁵ : P.IsStableUnderBaseChange
      inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝³ : W.Over S
      inst✝² : X.Over S
      inst✝¹ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      j : 𝒰.J
      ⊢ P (CategoryTheory.Limits.pullback.fst ((CategoryTheory.Over.forget S).map (A …
    -/
    exact P.pullback_fst _ _ (𝒰.map_prop j)
    /-
      🎉 no goals
    -/


instance (j : 𝒰.J) : ((𝒰.pullbackCoverOver S f).obj j).Over S where
  hom := (pullback (f.asOver S) ((𝒰.map j).asOver S)).hom


instance : (𝒰.pullbackCoverOver S f).Over S where
                                    /-
                                      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
                                      S : AlgebraicGeometry.Scheme
                                      inst✝⁵ : P.IsStableUnderBaseChange
                                      inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
                                      X W : AlgebraicGeometry.Scheme
                                      𝒰 : AlgebraicGeometry.Scheme.Cover P X
                                      f : Quiver.Hom W X
                                      inst✝³ : W.Over S
                                      inst✝² : X.Over S
                                      inst✝¹ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
                                      inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver f S
                                      j : (AlgebraicGeometry.Scheme.Cover.pullbackCoverOver S 𝒰 f).J
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.pull …
                                    -/
  isOver_map j := { comp_over := by exact Over.w (pullback.fst (f.asOver S) ((𝒰.map j).asOver S)) }
                                    /-
                                      🎉 no goals
                                    -/


/-- A variant of `AlgebraicGeometry.Scheme.Cover.pullbackCoverOver` with the arguments in the
fiber products flipped. -/
@[simps]
def Cover.pullbackCoverOver' : W.Cover P where
  J := 𝒰.J
  obj x := (pullback ((𝒰.map x).asOver S) (f.asOver S)).left
  map x := (pullback.snd ((𝒰.map x).asOver S) (f.asOver S)).left
  f x := 𝒰.f (f.base x)
  covers x := (mem_range_iff_of_surjective ((𝒰.pullbackCover' f).map (𝒰.f (f.base x))) _
    ((PreservesPullback.iso (Over.forget S) ((𝒰.map _).asOver S) (f.asOver S)).inv)
    (PreservesPullback.iso_inv_snd _ _ _) x).mp ((𝒰.pullbackCover' f).covers x)
  map_prop j := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁵ : P.IsStableUnderBaseChange
      inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝³ : W.Over S
      inst✝² : X.Over S
      inst✝¹ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      j : 𝒰.J
      ⊢ P ((fun x => (CategoryTheory.Limits.pullback.snd (AlgebraicGeometry.Scheme.H …
    -/
    dsimp only
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁵ : P.IsStableUnderBaseChange
      inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝³ : W.Over S
      inst✝² : X.Over S
      inst✝¹ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      j : 𝒰.J
      ⊢ P (CategoryTheory.Limits.pullback.snd (AlgebraicGeometry.Scheme.Hom.asOver ( …
    -/
    rw [← Over.forget_map, ← PreservesPullback.iso_hom_snd, P.cancel_left_of_respectsIso]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁵ : P.IsStableUnderBaseChange
      inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝³ : W.Over S
      inst✝² : X.Over S
      inst✝¹ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      j : 𝒰.J
      ⊢ P (CategoryTheory.Limits.pullback.snd ((CategoryTheory.Over.forget S).map (A …
    -/
    exact P.pullback_snd _ _ (𝒰.map_prop j)
    /-
      🎉 no goals
    -/


instance (j : 𝒰.J) : ((𝒰.pullbackCoverOver' S f).obj j).Over S where
  hom := (pullback ((𝒰.map j).asOver S) (f.asOver S)).hom


instance : (𝒰.pullbackCoverOver' S f).Over S where
                                    /-
                                      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
                                      S : AlgebraicGeometry.Scheme
                                      inst✝⁵ : P.IsStableUnderBaseChange
                                      inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
                                      X W : AlgebraicGeometry.Scheme
                                      𝒰 : AlgebraicGeometry.Scheme.Cover P X
                                      f : Quiver.Hom W X
                                      inst✝³ : W.Over S
                                      inst✝² : X.Over S
                                      inst✝¹ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
                                      inst✝ : AlgebraicGeometry.Scheme.Hom.IsOver f S
                                      j : (AlgebraicGeometry.Scheme.Cover.pullbackCoverOver' S 𝒰 f).J
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.pull …
                                    -/
  isOver_map j := { comp_over := by exact Over.w (pullback.snd ((𝒰.map j).asOver S) (f.asOver S)) }
                                    /-
                                      🎉 no goals
                                    -/


/-- The pullback of a cover of `S`-schemes with `Q` along a morphism of `S`-schemes. This is not
definitionally equal to `AlgebraicGeometry.Scheme.Cover.pullbackCover`, as here we take
the pullback in `Q.Over ⊤ S`, whose underlying scheme is only isomorphic but not equal to the
pullback in `Scheme`. -/
@[simps (config := .lemmasOnly)]
def Cover.pullbackCoverOverProp : W.Cover P where
  J := 𝒰.J
  obj x := (pullback (f.asOverProp (hX := hW) (hY := hX) S)
    ((𝒰.map x).asOverProp (hX := hQ x) (hY := hX) S)).left
  map x := (pullback.fst (f.asOverProp S) ((𝒰.map x).asOverProp S)).left
  f x := 𝒰.f (f.base x)
  covers x := (mem_range_iff_of_surjective ((𝒰.pullbackCover f).map (𝒰.f (f.base x))) _
    ((PreservesPullback.iso (MorphismProperty.Over.forget Q _ _ ⋙ Over.forget S)
      (f.asOverProp S) ((𝒰.map _).asOverProp S)).inv)
    (PreservesPullback.iso_inv_fst _ _ _) x).mp ((𝒰.pullbackCover f).covers x)
  map_prop j := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁸ : P.IsStableUnderBaseChange
      inst✝⁷ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝⁶ : W.Over S
      inst✝⁵ : X.Over S
      inst✝⁴ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝³ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : Q.HasOfPostcompProperty Q
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.IsStableUnderComposition
      hX : Q (CategoryTheory.over X S inferInstance)
      hW : Q (CategoryTheory.over W S inferInstance)
      hQ : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      j : 𝒰.J
      ⊢ P ((fun x => (CategoryTheory.Limits.pullback.fst (AlgebraicGeometry.Scheme.H …
    -/
    dsimp only
    rw [← Over.forget_map, MorphismProperty.Comma.toCommaMorphism_eq_hom,
      ← MorphismProperty.Comma.forget_map, ← Functor.comp_map]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁸ : P.IsStableUnderBaseChange
      inst✝⁷ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝⁶ : W.Over S
      inst✝⁵ : X.Over S
      inst✝⁴ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝³ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : Q.HasOfPostcompProperty Q
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.IsStableUnderComposition
      hX : Q (CategoryTheory.over X S inferInstance)
      hW : Q (CategoryTheory.over W S inferInstance)
      hQ : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      j : 𝒰.J
      ⊢ P (((CategoryTheory.MorphismProperty.Comma.forget (CategoryTheory.Functor.id …
    -/
    rw [← PreservesPullback.iso_hom_fst, P.cancel_left_of_respectsIso]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁸ : P.IsStableUnderBaseChange
      inst✝⁷ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝⁶ : W.Over S
      inst✝⁵ : X.Over S
      inst✝⁴ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝³ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : Q.HasOfPostcompProperty Q
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.IsStableUnderComposition
      hX : Q (CategoryTheory.over X S inferInstance)
      hW : Q (CategoryTheory.over W S inferInstance)
      hQ : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      j : 𝒰.J
      ⊢ P (CategoryTheory.Limits.pullback.fst (((CategoryTheory.MorphismProperty.Com …
    -/
    exact P.pullback_fst _ _ (𝒰.map_prop j)
    /-
      🎉 no goals
    -/


instance (j : 𝒰.J) : ((𝒰.pullbackCoverOverProp S f hX hW hQ).obj j).Over S where
  hom := (pullback (f.asOverProp (hX := hW) (hY := hX) S)
    ((𝒰.map j).asOverProp (hX := hQ j) (hY := hX) S)).hom


instance : (𝒰.pullbackCoverOverProp S f hX hW hQ).Over S where
  isOver_map j :=
                      /-
                        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
                        S : AlgebraicGeometry.Scheme
                        inst✝⁸ : P.IsStableUnderBaseChange
                        inst✝⁷ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
                        X W : AlgebraicGeometry.Scheme
                        𝒰 : AlgebraicGeometry.Scheme.Cover P X
                        f : Quiver.Hom W X
                        inst✝⁶ : W.Over S
                        inst✝⁵ : X.Over S
                        inst✝⁴ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
                        inst✝³ : AlgebraicGeometry.Scheme.Hom.IsOver f S
                        Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
                        inst✝² : Q.HasOfPostcompProperty Q
                        inst✝¹ : Q.IsStableUnderBaseChange
                        inst✝ : Q.IsStableUnderComposition
                        hX : Q (CategoryTheory.over X S inferInstance)
                        hW : Q (CategoryTheory.over W S inferInstance)
                        hQ : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
                        j : (AlgebraicGeometry.Scheme.Cover.pullbackCoverOverProp S 𝒰 f hX hW hQ).J
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.pull …
                      -/
    { comp_over := by exact (pullback.fst (f.asOverProp S) ((𝒰.map j).asOverProp S)).w }
                      /-
                        🎉 no goals
                      -/


/-- A variant of `AlgebraicGeometry.Scheme.Cover.pullbackCoverOverProp` with the arguments in the
fiber products flipped. -/
@[simps (config := .lemmasOnly)]
def Cover.pullbackCoverOverProp' : W.Cover P where
  J := 𝒰.J
  obj x := (pullback ((𝒰.map x).asOverProp (hX := hQ x) (hY := hX) S)
    (f.asOverProp (hX := hW) (hY := hX) S)).left
  map x := (pullback.snd ((𝒰.map x).asOverProp S) (f.asOverProp S)).left
  f x := 𝒰.f (f.base x)
  covers x := (mem_range_iff_of_surjective ((𝒰.pullbackCover' f).map (𝒰.f (f.base x))) _
    ((PreservesPullback.iso (MorphismProperty.Over.forget Q _ _ ⋙ Over.forget S)
      ((𝒰.map _).asOverProp S) (f.asOverProp S)).inv)
    (PreservesPullback.iso_inv_snd _ _ _) x).mp ((𝒰.pullbackCover' f).covers x)
  map_prop j := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁸ : P.IsStableUnderBaseChange
      inst✝⁷ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝⁶ : W.Over S
      inst✝⁵ : X.Over S
      inst✝⁴ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝³ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : Q.HasOfPostcompProperty Q
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.IsStableUnderComposition
      hX : Q (CategoryTheory.over X S inferInstance)
      hW : Q (CategoryTheory.over W S inferInstance)
      hQ : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      j : 𝒰.J
      ⊢ P ((fun x => (CategoryTheory.Limits.pullback.snd (AlgebraicGeometry.Scheme.H …
    -/
    dsimp only
    rw [← Over.forget_map, MorphismProperty.Comma.toCommaMorphism_eq_hom,
      ← MorphismProperty.Comma.forget_map, ← Functor.comp_map]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁸ : P.IsStableUnderBaseChange
      inst✝⁷ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝⁶ : W.Over S
      inst✝⁵ : X.Over S
      inst✝⁴ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝³ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : Q.HasOfPostcompProperty Q
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.IsStableUnderComposition
      hX : Q (CategoryTheory.over X S inferInstance)
      hW : Q (CategoryTheory.over W S inferInstance)
      hQ : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      j : 𝒰.J
      ⊢ P (((CategoryTheory.MorphismProperty.Comma.forget (CategoryTheory.Functor.id …
    -/
    rw [← PreservesPullback.iso_hom_snd, P.cancel_left_of_respectsIso]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁸ : P.IsStableUnderBaseChange
      inst✝⁷ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝⁶ : W.Over S
      inst✝⁵ : X.Over S
      inst✝⁴ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      inst✝³ : AlgebraicGeometry.Scheme.Hom.IsOver f S
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : Q.HasOfPostcompProperty Q
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.IsStableUnderComposition
      hX : Q (CategoryTheory.over X S inferInstance)
      hW : Q (CategoryTheory.over W S inferInstance)
      hQ : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      j : 𝒰.J
      ⊢ P (CategoryTheory.Limits.pullback.snd (((CategoryTheory.MorphismProperty.Com …
    -/
    exact P.pullback_snd _ _ (𝒰.map_prop j)
    /-
      🎉 no goals
    -/


instance (j : 𝒰.J) : ((𝒰.pullbackCoverOverProp' S f hX hW hQ).obj j).Over S where
  hom := (pullback ((𝒰.map j).asOverProp (hX := hQ j) (hY := hX) S)
    (f.asOverProp (hX := hW) (hY := hX) S)).hom


instance : (𝒰.pullbackCoverOverProp' S f hX hW hQ).Over S where
  isOver_map j :=
                      /-
                        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
                        S : AlgebraicGeometry.Scheme
                        inst✝⁸ : P.IsStableUnderBaseChange
                        inst✝⁷ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
                        X W : AlgebraicGeometry.Scheme
                        𝒰 : AlgebraicGeometry.Scheme.Cover P X
                        f : Quiver.Hom W X
                        inst✝⁶ : W.Over S
                        inst✝⁵ : X.Over S
                        inst✝⁴ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
                        inst✝³ : AlgebraicGeometry.Scheme.Hom.IsOver f S
                        Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
                        inst✝² : Q.HasOfPostcompProperty Q
                        inst✝¹ : Q.IsStableUnderBaseChange
                        inst✝ : Q.IsStableUnderComposition
                        hX : Q (CategoryTheory.over X S inferInstance)
                        hW : Q (CategoryTheory.over W S inferInstance)
                        hQ : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
                        j : (AlgebraicGeometry.Scheme.Cover.pullbackCoverOverProp' S 𝒰 f hX hW hQ).J
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.pull …
                      -/
    { comp_over := by exact (pullback.snd ((𝒰.map j).asOverProp S) (f.asOverProp S)).w }
                      /-
                        🎉 no goals
                      -/


instance (j : (𝒰.bind 𝒱).J) : ((𝒰.bind 𝒱).obj j).Over S :=
  inferInstanceAs <| ((𝒱 j.1).obj j.2).Over S


instance {X : Scheme.{u}} (𝒰 : X.Cover P) (𝒱 : ∀ x, (𝒰.obj x).Cover P)
    [X.Over S] [𝒰.Over S] [∀ x, (𝒱 x).Over S] : (𝒰.bind 𝒱).Over S where
  over := fun ⟨i, j⟩ ↦ inferInstanceAs <| ((𝒱 i).obj j).Over S
                                               /-
                                                 P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
                                                 S : AlgebraicGeometry.Scheme
                                                 inst✝⁶ : P.IsStableUnderComposition
                                                 X✝ : AlgebraicGeometry.Scheme
                                                 𝒰✝ : AlgebraicGeometry.Scheme.Cover P X✝
                                                 𝒱✝ : (x : 𝒰✝.J) → AlgebraicGeometry.Scheme.Cover P (𝒰✝.obj x)
                                                 inst✝⁵ : X✝.Over S
                                                 inst✝⁴ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰✝
                                                 inst✝³ : (x : 𝒰✝.J) → AlgebraicGeometry.Scheme.Cover.Over S (𝒱✝ x)
                                                 X : AlgebraicGeometry.Scheme
                                                 𝒰 : AlgebraicGeometry.Scheme.Cover P X
                                                 𝒱 : (x : 𝒰.J) → AlgebraicGeometry.Scheme.Cover P (𝒰.obj x)
                                                 inst✝² : X.Over S
                                                 inst✝¹ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
                                                 inst✝ : (x : 𝒰.J) → AlgebraicGeometry.Scheme.Cover.Over S (𝒱 x)
                                                 x✝ : (𝒰.bind 𝒱).J
                                                 i : 𝒰.J
                                                 j : (𝒱 i).J
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((𝒰.bind 𝒱).map ⟨i, j⟩) (CategoryTheo …
                                               -/
  isOver_map := fun ⟨i, j⟩ ↦ { comp_over := by simp }
                                               /-
                                                 🎉 no goals
                                               -/


