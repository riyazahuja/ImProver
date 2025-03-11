/-- The map of a fork is a limit iff the fork consisting of the mapped morphisms is a limit. This
essentially lets us commute `Fork.ofι` with `Functor.mapCone`.
-/
def isLimitMapConeForkEquiv :
    IsLimit (G.mapCone (Fork.ofι h w)) ≃
                                      /-
                                        C : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                        D : Type u₂
                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                        G : CategoryTheory.Functor C D
                                        X Y Z : C
                                        f g : Quiver.Hom X Y
                                        h : Quiver.Hom Z X
                                        w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) (CategoryTheory. …
                                      -/
      IsLimit (Fork.ofι (G.map h) (by simp only [← G.map_comp, w]) : Fork (G.map f) (G.map g)) :=
                                      /-
                                        🎉 no goals
                                      -/
  (IsLimit.postcomposeHomEquiv (diagramIsoParallelPair _) _).symm.trans
                                                      /-
                                                        C : Type u₁
                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                        D : Type u₂
                                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                        G : CategoryTheory.Functor C D
                                                        X Y Z : C
                                                        f g : Quiver.Hom X Y
                                                        h : Quiver.Hom Z X
                                                        w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl ((CategoryTh …
                                                      -/
    (IsLimit.equivIsoLimit (Fork.ext (Iso.refl _) (by simp [Fork.ι])))
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The property of preserving equalizers expressed in terms of forks. -/
def isLimitForkMapOfIsLimit [PreservesLimit (parallelPair f g) G] (l : IsLimit (Fork.ofι h w)) :
                                    /-
                                      C : Type u₁
                                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                      G : CategoryTheory.Functor C D
                                      X Y Z : C
                                      f g : Quiver.Hom X Y
                                      h : Quiver.Hom Z X
                                      w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
                                      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
                                      l : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι h w)
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) (CategoryTheory. …
                                    -/
    IsLimit (Fork.ofι (G.map h) (by simp only [← G.map_comp, w]) : Fork (G.map f) (G.map g)) :=
                                    /-
                                      🎉 no goals
                                    -/
  isLimitMapConeForkEquiv G w (isLimitOfPreserves G l)


/-- The property of reflecting equalizers expressed in terms of forks. -/
def isLimitOfIsLimitForkMap [ReflectsLimit (parallelPair f g) G]
                                         /-
                                           C : Type u₁
                                           inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                           D : Type u₂
                                           inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                           G : CategoryTheory.Functor C D
                                           X Y Z : C
                                           f g : Quiver.Hom X Y
                                           h : Quiver.Hom Z X
                                           w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
                                           inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.parallelPai …
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) (CategoryTheory. …
                                         -/
    (l : IsLimit (Fork.ofι (G.map h) (by simp only [← G.map_comp, w]) : Fork (G.map f) (G.map g))) :
                                         /-
                                           🎉 no goals
                                         -/
    IsLimit (Fork.ofι h w) :=
  isLimitOfReflects G ((isLimitMapConeForkEquiv G w).symm l)


/--
If `G` preserves equalizers and `C` has them, then the fork constructed of the mapped morphisms of
a fork is a limit.
-/
def isLimitOfHasEqualizerOfPreservesLimit [PreservesLimit (parallelPair f g) G] :
    IsLimit (Fork.ofι
                                    /-
                                      C : Type u₁
                                      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                      G : CategoryTheory.Functor C D
                                      X Y Z : C
                                      f g : Quiver.Hom X Y
                                      h : Quiver.Hom Z X
                                      w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
                                      inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
                                      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.equaliz …
                                    -/
      (G.map (equalizer.ι f g)) (by simp only [← G.map_comp]; rw [equalizer.condition]) :
                                                              /-
                                                                🎉 no goals
                                                              -/
      Fork (G.map f) (G.map g)) :=
  isLimitForkMapOfIsLimit G _ (equalizerIsEqualizer f g)


/-- If the equalizer comparison map for `G` at `(f,g)` is an isomorphism, then `G` preserves the
equalizer of `(f,g)`.
-/
lemma PreservesEqualizer.of_iso_comparison [i : IsIso (equalizerComparison f g G)] :
    PreservesLimit (parallelPair f g) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
    inst✝ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.equalizerComparison f g G)
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair f g …
  -/
  apply preservesLimit_of_preserves_limit_cone (equalizerIsEqualizer f g)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
    inst✝ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.equalizerComparison f g G)
    ⊢ CategoryTheory.Limits.IsLimit (G.mapCone (CategoryTheory.Limits.Fork.ofι (Ca …
  -/
  apply (isLimitMapConeForkEquiv _ _).symm _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
    inst✝ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.equalizerComparison f g G)
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (G.map (Catego …
  -/
  exact @IsLimit.ofPointIso _ _ _ _ _ _ _ (limit.isLimit (parallelPair (G.map f) (G.map g))) i
  /-
    🎉 no goals
  -/


/--
If `G` preserves the equalizer of `(f,g)`, then the equalizer comparison map for `G` at `(f,g)` is
an isomorphism.
-/
def PreservesEqualizer.iso : G.obj (equalizer f g) ≅ equalizer (G.map f) (G.map g) :=
  IsLimit.conePointUniqueUpToIso (isLimitOfHasEqualizerOfPreservesLimit G f g) (limit.isLimit _)


@[simp]
theorem PreservesEqualizer.iso_hom :
    (PreservesEqualizer.iso G f g).hom = equalizerComparison f g G :=
  rfl


@[simp]
theorem PreservesEqualizer.iso_inv_ι :
    (PreservesEqualizer.iso G f g).inv ≫ G.map (equalizer.ι f g) =
      equalizer.ι (G.map f) (G.map g) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasEqualizer f g
    inst✝¹ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesEqual …
  -/
  rw [← Iso.cancel_iso_hom_left (PreservesEqualizer.iso G f g), ← Category.assoc, Iso.hom_inv_id]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasEqualizer f g
    inst✝¹ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (G. …
  -/
  simp
  /-
    🎉 no goals
  -/


instance : IsIso (equalizerComparison f g G) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f g : Quiver.Hom X Y
    h : Quiver.Hom Z X
    w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
    inst✝² : CategoryTheory.Limits.HasEqualizer f g
    inst✝¹ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.equalizerComparison f g G)
  -/
  rw [← PreservesEqualizer.iso_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f g : Quiver.Hom X Y
    h : Quiver.Hom Z X
    w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
    inst✝² : CategoryTheory.Limits.HasEqualizer f g
    inst✝¹ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesEqualizer.iso G f g).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The map of a cofork is a colimit iff the cofork consisting of the mapped morphisms is a colimit.
This essentially lets us commute `Cofork.ofπ` with `Functor.mapCocone`.
-/
def isColimitMapCoconeCoforkEquiv :
    IsColimit (G.mapCocone (Cofork.ofπ h w)) ≃
      IsColimit
                                  /-
                                    C : Type u₁
                                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                    D : Type u₂
                                    inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                    G : CategoryTheory.Functor C D
                                    X Y Z : C
                                    f g : Quiver.Hom X Y
                                    h : Quiver.Hom Y Z
                                    w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) (CategoryTheory. …
                                  -/
        (Cofork.ofπ (G.map h) (by simp only [← G.map_comp, w]) : Cofork (G.map f) (G.map g)) :=
                                  /-
                                    🎉 no goals
                                  -/
  (IsColimit.precomposeInvEquiv (diagramIsoParallelPair _) _).symm.trans <|
    IsColimit.equivIsoColimit <|
      Cofork.ext (Iso.refl _) <| by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          X Y Z : C
          f g : Quiver.Hom X Y
          h : Quiver.Hom Y Z
          w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ((Cat …
        -/
        dsimp only [Cofork.π, Cofork.ofπ_ι_app]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          X Y Z : C
          f g : Quiver.Hom X Y
          h : Quiver.Hom Y Z
          w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
        -/
        dsimp; rw [Category.comp_id, Category.id_comp]
               /-
                 🎉 no goals
               -/


/-- The property of preserving coequalizers expressed in terms of coforks. -/
def isColimitCoforkMapOfIsColimit [PreservesColimit (parallelPair f g) G]
    (l : IsColimit (Cofork.ofπ h w)) :
    IsColimit
                                /-
                                  C : Type u₁
                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                  D : Type u₂
                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                  G : CategoryTheory.Functor C D
                                  X Y Z : C
                                  f g : Quiver.Hom X Y
                                  h : Quiver.Hom Y Z
                                  w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
                                  inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
                                  l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ h w)
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) (CategoryTheory. …
                                -/
      (Cofork.ofπ (G.map h) (by simp only [← G.map_comp, w]) : Cofork (G.map f) (G.map g)) :=
                                /-
                                  🎉 no goals
                                -/
  isColimitMapCoconeCoforkEquiv G w (isColimitOfPreserves G l)


/-- The property of reflecting coequalizers expressed in terms of coforks. -/
def isColimitOfIsColimitCoforkMap [ReflectsColimit (parallelPair f g) G]
    (l :
      IsColimit
                                  /-
                                    C : Type u₁
                                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                    D : Type u₂
                                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                    G : CategoryTheory.Functor C D
                                    X Y Z : C
                                    f g : Quiver.Hom X Y
                                    h : Quiver.Hom Y Z
                                    w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
                                    inst✝ : CategoryTheory.Limits.ReflectsColimit (CategoryTheory.Limits.parallelP …
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) (CategoryTheory. …
                                  -/
        (Cofork.ofπ (G.map h) (by simp only [← G.map_comp, w]) : Cofork (G.map f) (G.map g))) :
                                  /-
                                    🎉 no goals
                                  -/
    IsColimit (Cofork.ofπ h w) :=
  isColimitOfReflects G ((isColimitMapCoconeCoforkEquiv G w).symm l)


/--
If `G` preserves coequalizers and `C` has them, then the cofork constructed of the mapped morphisms
of a cofork is a colimit.
-/
def isColimitOfHasCoequalizerOfPreservesColimit [PreservesColimit (parallelPair f g) G] :
    IsColimit (Cofork.ofπ (G.map (coequalizer.π f g)) (by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        X Y Z : C
        f g : Quiver.Hom X Y
        h : Quiver.Hom Y Z
        w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
        inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
        inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map (CategoryTheory.Limi …
      -/
      simp only [← G.map_comp]; rw [coequalizer.condition]) : Cofork (G.map f) (G.map g)) :=
                                /-
                                  🎉 no goals
                                -/
  isColimitCoforkMapOfIsColimit G _ (coequalizerIsCoequalizer f g)


/-- If the coequalizer comparison map for `G` at `(f,g)` is an isomorphism, then `G` preserves the
coequalizer of `(f,g)`.
-/
lemma of_iso_comparison [i : IsIso (coequalizerComparison f g G)] :
    PreservesColimit (parallelPair f g) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
    inst✝ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.coequalizerComparison f g G)
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair f …
  -/
  apply preservesColimit_of_preserves_colimit_cocone (coequalizerIsCoequalizer f g)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
    inst✝ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.coequalizerComparison f g G)
    ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone (CategoryTheory.Limits.Cofork.o …
  -/
  apply (isColimitMapCoconeCoforkEquiv _ _).symm _
  exact
    @IsColimit.ofPointIso _ _ _ _ _ _ _ (colimit.isColimit (parallelPair (G.map f) (G.map g))) i


/--
If `G` preserves the coequalizer of `(f,g)`, then the coequalizer comparison map for `G` at `(f,g)`
is an isomorphism.
-/
def PreservesCoequalizer.iso : coequalizer (G.map f) (G.map g) ≅ G.obj (coequalizer f g) :=
  IsColimit.coconePointUniqueUpToIso (colimit.isColimit _)
    (isColimitOfHasCoequalizerOfPreservesColimit G f g)


@[simp]
theorem PreservesCoequalizer.iso_hom :
    (PreservesCoequalizer.iso G f g).hom = coequalizerComparison f g G :=
  rfl


instance : IsIso (coequalizerComparison f g G) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
    inst✝² : CategoryTheory.Limits.HasCoequalizer f g
    inst✝¹ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.coequalizerComparison f g G)
  -/
  rw [← PreservesCoequalizer.iso_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
    inst✝² : CategoryTheory.Limits.HasCoequalizer f g
    inst✝¹ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesCoequalizer.iso G f g). …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance map_π_epi : Epi (G.map (coequalizer.π f g)) :=
  ⟨fun {W} h k => by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      X Y Z : C
      f g : Quiver.Hom X Y
      h✝ : Quiver.Hom Y Z
      w : Eq (CategoryTheory.CategoryStruct.comp f h✝) (CategoryTheory.CategoryStruc …
      inst✝² : CategoryTheory.Limits.HasCoequalizer f g
      inst✝¹ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
      W : D
      h k : Quiver.Hom (G.obj (CategoryTheory.Limits.coequalizer f g)) W
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.coequal …
    -/
    rw [← ι_comp_coequalizerComparison]
    haveI : Epi (coequalizer.π (G.map f) (G.map g) ≫ coequalizerComparison f g G) := by
      apply epi_comp
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      X Y Z : C
      f g : Quiver.Hom X Y
      h✝ : Quiver.Hom Y Z
      w : Eq (CategoryTheory.CategoryStruct.comp f h✝) (CategoryTheory.CategoryStruc …
      inst✝² : CategoryTheory.Limits.HasCoequalizer f g
      inst✝¹ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
      W : D
      h k : Quiver.Hom (G.obj (CategoryTheory.Limits.coequalizer f g)) W
      this : CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    apply (cancel_epi _).1⟩
    /-
      🎉 no goals
    -/


@[reassoc]
theorem map_π_preserves_coequalizer_inv :
    G.map (coequalizer.π f g) ≫ (PreservesCoequalizer.iso G f g).inv =
      coequalizer.π (G.map f) (G.map g) := by
  rw [← ι_comp_coequalizerComparison_assoc, ← PreservesCoequalizer.iso_hom, Iso.hom_inv_id,
    comp_id]


@[reassoc]
theorem map_π_preserves_coequalizer_inv_desc {W : D} (k : G.obj Y ⟶ W)
    (wk : G.map f ≫ k = G.map g ≫ k) : G.map (coequalizer.π f g) ≫
      (PreservesCoequalizer.iso G f g).inv ≫ coequalizer.desc k wk = k := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasCoequalizer f g
    inst✝¹ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    W : D
    k : Quiver.Hom (G.obj Y) W
    wk : Eq (CategoryTheory.CategoryStruct.comp (G.map f) k) (CategoryTheory.Categ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.coequal …
  -/
  rw [← Category.assoc, map_π_preserves_coequalizer_inv, coequalizer.π_desc]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem map_π_preserves_coequalizer_inv_colimMap {X' Y' : D} (f' g' : X' ⟶ Y')
    [HasCoequalizer f' g'] (p : G.obj X ⟶ X') (q : G.obj Y ⟶ Y') (wf : G.map f ≫ q = p ≫ f')
    (wg : G.map g ≫ q = p ≫ g') :
    G.map (coequalizer.π f g) ≫
        (PreservesCoequalizer.iso G f g).inv ≫
          colimMap (parallelPairHom (G.map f) (G.map g) f' g' p q wf wg) =
      q ≫ coequalizer.π f' g' := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasCoequalizer f g
    inst✝² : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.paralle …
    X' Y' : D
    f' g' : Quiver.Hom X' Y'
    inst✝ : CategoryTheory.Limits.HasCoequalizer f' g'
    p : Quiver.Hom (G.obj X) X'
    q : Quiver.Hom (G.obj Y) Y'
    wf : Eq (CategoryTheory.CategoryStruct.comp (G.map f) q) (CategoryTheory.Categ …
    wg : Eq (CategoryTheory.CategoryStruct.comp (G.map g) q) (CategoryTheory.Categ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.coequal …
  -/
  rw [← Category.assoc, map_π_preserves_coequalizer_inv, ι_colimMap, parallelPairHom_app_one]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem map_π_preserves_coequalizer_inv_colimMap_desc {X' Y' : D} (f' g' : X' ⟶ Y')
    [HasCoequalizer f' g'] (p : G.obj X ⟶ X') (q : G.obj Y ⟶ Y') (wf : G.map f ≫ q = p ≫ f')
    (wg : G.map g ≫ q = p ≫ g') {Z' : D} (h : Y' ⟶ Z') (wh : f' ≫ h = g' ≫ h) :
    G.map (coequalizer.π f g) ≫
        (PreservesCoequalizer.iso G f g).inv ≫
          colimMap (parallelPairHom (G.map f) (G.map g) f' g' p q wf wg) ≫ coequalizer.desc h wh =
      q ≫ h := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasCoequalizer f g
    inst✝² : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.paralle …
    X' Y' : D
    f' g' : Quiver.Hom X' Y'
    inst✝ : CategoryTheory.Limits.HasCoequalizer f' g'
    p : Quiver.Hom (G.obj X) X'
    q : Quiver.Hom (G.obj Y) Y'
    wf : Eq (CategoryTheory.CategoryStruct.comp (G.map f) q) (CategoryTheory.Categ …
    wg : Eq (CategoryTheory.CategoryStruct.comp (G.map g) q) (CategoryTheory.Categ …
    Z' : D
    h : Quiver.Hom Y' Z'
    wh : Eq (CategoryTheory.CategoryStruct.comp f' h) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.coequal …
  -/
  slice_lhs 1 3 => rw [map_π_preserves_coequalizer_inv_colimMap]
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    f g : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasCoequalizer f g
    inst✝² : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.paralle …
    X' Y' : D
    f' g' : Quiver.Hom X' Y'
    inst✝ : CategoryTheory.Limits.HasCoequalizer f' g'
    p : Quiver.Hom (G.obj X) X'
    q : Quiver.Hom (G.obj Y) Y'
    wf : Eq (CategoryTheory.CategoryStruct.comp (G.map f) q) (CategoryTheory.Categ …
    wg : Eq (CategoryTheory.CategoryStruct.comp (G.map g) q) (CategoryTheory.Categ …
    Z' : D
    h : Quiver.Hom Y' Z'
    wh : Eq (CategoryTheory.CategoryStruct.comp f' h) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp q …
  -/
  slice_lhs 2 3 => rw [coequalizer.π_desc]
  /-
    🎉 no goals
  -/


/-- Any functor preserves coequalizers of split pairs. -/
instance (priority := 1) preservesSplitCoequalizers (f g : X ⟶ Y) [HasSplitCoequalizer f g] :
    PreservesColimit (parallelPair f g) G := by
  apply
    preservesColimit_of_preserves_colimit_cocone
      (HasSplitCoequalizer.isSplitCoequalizer f g).isCoequalizer
  apply
    (isColimitMapCoconeCoforkEquiv G _).symm
      ((HasSplitCoequalizer.isSplitCoequalizer f g).map G).isCoequalizer


instance (priority := 1) preservesSplitEqualizers (f g : X ⟶ Y) [HasSplitEqualizer f g] :
    PreservesLimit (parallelPair f g) G := by
  apply
    preservesLimit_of_preserves_limit_cone
      (HasSplitEqualizer.isSplitEqualizer f g).isEqualizer
  apply
    (isLimitMapConeForkEquiv G _).symm
      ((HasSplitEqualizer.isSplitEqualizer f g).map G).isEqualizer


