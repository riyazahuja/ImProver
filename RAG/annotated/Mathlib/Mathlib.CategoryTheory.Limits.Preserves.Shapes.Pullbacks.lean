/-- The image of a pullback cone by a functor. -/
abbrev map : PullbackCone (G.map f) (G.map g) :=
  PullbackCone.mk (G.map c.fst) (G.map c.snd)
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          c : CategoryTheory.Limits.PullbackCone f g
          G : CategoryTheory.Functor C D
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map c.fst) (G.map f)) (CategoryThe …
        -/
    (by simpa using G.congr_map c.condition)
        /-
          🎉 no goals
        -/


/-- The map (as a cone) of a pullback cone is limit iff
the map (as a pullback cone) is limit. -/
def isLimitMapConeEquiv :
    IsLimit (mapCone G c) ≃ IsLimit (c.map G) :=
  (IsLimit.postcomposeHomEquiv (diagramIsoCospan.{v₂} _) _).symm.trans <|
    IsLimit.equivIsoLimit <| by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        X Y Z : C
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        c : CategoryTheory.Limits.PullbackCone f g
        G : CategoryTheory.Functor C D
        ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cones.postcompose (CategoryTheory …
      -/
      refine PullbackCone.ext (Iso.refl _) ?_ ?_
        /-
          case refine_1
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          c : CategoryTheory.Limits.PullbackCone f g
          G : CategoryTheory.Functor C D
          ⊢ Eq (CategoryTheory.Limits.PullbackCone.fst ((CategoryTheory.Limits.Cones.pos …
        -/
      · dsimp only [fst]
        /-
          case refine_1
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          c : CategoryTheory.Limits.PullbackCone f g
          G : CategoryTheory.Functor C D
          ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Limits.diagram …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          c : CategoryTheory.Limits.PullbackCone f g
          G : CategoryTheory.Functor C D
          ⊢ Eq (CategoryTheory.Limits.PullbackCone.snd ((CategoryTheory.Limits.Cones.pos …
        -/
      · dsimp only [snd]
        /-
          case refine_2
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          c : CategoryTheory.Limits.PullbackCone f g
          G : CategoryTheory.Functor C D
          ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Limits.diagram …
        -/
        simp
        /-
          🎉 no goals
        -/


/-- The map of a pullback cone is a limit iff the fork consisting of the mapped morphisms is a
limit. This essentially lets us commute `PullbackCone.mk` with `Functor.mapCone`. -/
def isLimitMapConePullbackConeEquiv :
    IsLimit (mapCone G (PullbackCone.mk h k comm)) ≃
      IsLimit
                                                 /-
                                                   C : Type u₁
                                                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                   D : Type u₂
                                                   inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                   G : CategoryTheory.Functor C D
                                                   W X Y Z : C
                                                   f : Quiver.Hom X Z
                                                   g : Quiver.Hom Y Z
                                                   h : Quiver.Hom W X
                                                   k : Quiver.Hom W Y
                                                   comm : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStr …
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) (CategoryTheory. …
                                                 -/
        (PullbackCone.mk (G.map h) (G.map k) (by simp only [← G.map_comp, comm]) :
                                                 /-
                                                   🎉 no goals
                                                 -/
          PullbackCone (G.map f) (G.map g)) :=
  (PullbackCone.mk _ _ comm).isLimitMapConeEquiv G


/-- The property of preserving pullbacks expressed in terms of binary fans. -/
def isLimitPullbackConeMapOfIsLimit [PreservesLimit (cospan f g) G]
    (l : IsLimit (PullbackCone.mk h k comm)) :
                                                       /-
                                                         C : Type u₁
                                                         inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                         D : Type u₂
                                                         inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                         G : CategoryTheory.Functor C D
                                                         W X Y Z : C
                                                         f : Quiver.Hom X Z
                                                         g : Quiver.Hom Y Z
                                                         h : Quiver.Hom W X
                                                         k : Quiver.Hom W Y
                                                         comm : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStr …
                                                         inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
                                                         l : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk h k c …
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) (CategoryTheory. …
                                                       -/
    have : G.map h ≫ G.map f = G.map k ≫ G.map g := by rw [← G.map_comp, ← G.map_comp,comm]
                                                       /-
                                                         🎉 no goals
                                                       -/
    IsLimit (PullbackCone.mk (G.map h) (G.map k) this) :=
  (PullbackCone.isLimitMapConeEquiv _ G).1 (isLimitOfPreserves G l)


/-- The property of reflecting pullbacks expressed in terms of binary fans. -/
def isLimitOfIsLimitPullbackConeMap [ReflectsLimit (cospan f g) G]
    (l : IsLimit (PullbackCone.mk (G.map h) (G.map k) (show G.map h ≫ G.map f = G.map k ≫ G.map g
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              G : CategoryTheory.Functor C D
              W X Y Z : C
              f : Quiver.Hom X Z
              g : Quiver.Hom Y Z
              h : Quiver.Hom W X
              k : Quiver.Hom W Y
              comm : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStr …
              inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan f g) G
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) (CategoryTheory. …
            -/
    from by simp only [← G.map_comp,comm]))) : IsLimit (PullbackCone.mk h k comm) :=
            /-
              🎉 no goals
            -/
  isLimitOfReflects G
    ((PullbackCone.isLimitMapConeEquiv (PullbackCone.mk _ _ comm) G).2 l)


/-- If `G` preserves pullbacks and `C` has them, then the pullback cone constructed of the mapped
morphisms of the pullback cone is a limit. -/
def isLimitOfHasPullbackOfPreservesLimit [HasPullback f g] :
    have : G.map (pullback.fst f g) ≫ G.map f = G.map (pullback.snd f g) ≫ G.map g := by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        W X Y Z : C
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        h : Quiver.Hom W X
        k : Quiver.Hom W Y
        comm : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStr …
        inst✝¹ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f  …
        inst✝ : CategoryTheory.Limits.HasPullback f g
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.pullbac …
      -/
      simp only [← G.map_comp, pullback.condition]
      /-
        🎉 no goals
      -/
    IsLimit (PullbackCone.mk (G.map (pullback.fst f g)) (G.map (pullback.snd f g)) this) :=
  isLimitPullbackConeMapOfIsLimit G _ (pullbackIsPullback f g)


/-- If `F` preserves the pullback of `f, g`, it also preserves the pullback of `g, f`. -/
lemma preservesPullback_symmetry : PreservesLimit (cospan g f) G where
  preserves {c} hc := ⟨by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan g f)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit (G.mapCone c)
    -/
    apply (IsLimit.postcomposeHomEquiv (diagramIsoCospan.{v₂} _) _).toFun
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan g f)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.postcompose (Cat …
    -/
    apply IsLimit.ofIsoLimit _ (PullbackCone.isoMk _).symm
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan g f)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk ((G.map …
    -/
    apply PullbackCone.isLimitOfFlip
    /-
      case ht
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan g f)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk ((G.map …
    -/
    apply (isLimitMapConePullbackConeEquiv _ _).toFun
      /-
        case ht
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        X Y Z : C
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
        c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan g f)
        hc : CategoryTheory.Limits.IsLimit c
        ⊢ CategoryTheory.Limits.IsLimit (G.mapCone (CategoryTheory.Limits.PullbackCone …
      -/
    · refine @isLimitOfPreserves _ _ _ _ _ _ _ _ _ ?_ ?_
        /-
          case ht.refine_1
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
          c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan g f)
          hc : CategoryTheory.Limits.IsLimit c
          ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (c.π.ap …
        -/
      · apply PullbackCone.isLimitOfFlip
        /-
          case ht.refine_1.ht
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
          c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan g f)
          hc : CategoryTheory.Limits.IsLimit c
          ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (c.π.ap …
        -/
        apply IsLimit.ofIsoLimit _ (PullbackCone.isoMk _)
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
          c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan g f)
          hc : CategoryTheory.Limits.IsLimit c
          ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.postcompose (Cat …
        -/
        exact (IsLimit.postcomposeHomEquiv (diagramIsoCospan.{v₁} _) _).invFun hc
        /-
          🎉 no goals
        -/
        /-
          case ht.refine_2
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
          c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan g f)
          hc : CategoryTheory.Limits.IsLimit c
          ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan ((Categor …
        -/
      · dsimp
        /-
          case ht.refine_2
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
          c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.cospan g f)
          hc : CategoryTheory.Limits.IsLimit c
          ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g) G
        -/
        infer_instance
        /-
          🎉 no goals
        -/
    · exact
        (c.π.naturality WalkingCospan.Hom.inr).symm.trans
          (c.π.naturality WalkingCospan.Hom.inl : _)⟩


theorem hasPullback_of_preservesPullback [HasPullback f g] : HasPullback (G.map f) (G.map g) :=
  ⟨⟨⟨_, isLimitPullbackConeMapOfIsLimit G _ (pullbackIsPullback _ _)⟩⟩⟩


/-- If `G` preserves the pullback of `(f,g)`, then the pullback comparison map for `G` at `(f,g)` is
an isomorphism. -/
def PreservesPullback.iso : G.obj (pullback f g) ≅ pullback (G.map f) (G.map g) :=
  IsLimit.conePointUniqueUpToIso (isLimitOfHasPullbackOfPreservesLimit G f g) (limit.isLimit _)


@[simp]
theorem PreservesPullback.iso_hom : (PreservesPullback.iso G f g).hom = pullbackComparison G f g :=
  rfl


@[reassoc]
theorem PreservesPullback.iso_hom_fst :
    (PreservesPullback.iso G f g).hom ≫ pullback.fst _ _ = G.map (pullback.fst f g) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f  …
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesPullb …
  -/
  simp [PreservesPullback.iso]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem PreservesPullback.iso_hom_snd :
    (PreservesPullback.iso G f g).hom ≫ pullback.snd _ _ = G.map (pullback.snd f g) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f  …
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesPullb …
  -/
  simp [PreservesPullback.iso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem PreservesPullback.iso_inv_fst :
    (PreservesPullback.iso G f g).inv ≫ G.map (pullback.fst f g) = pullback.fst _ _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f  …
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesPullb …
  -/
  simp [PreservesPullback.iso, Iso.inv_comp_eq]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem PreservesPullback.iso_inv_snd :
    (PreservesPullback.iso G f g).inv ≫ G.map (pullback.snd f g) = pullback.snd _ _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f  …
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesPullb …
  -/
  simp [PreservesPullback.iso, Iso.inv_comp_eq]
  /-
    🎉 no goals
  -/


/-- A pullback cone in `C` is limit iff if it is so after the application
of `coyoneda.obj X` for all `X : Cᵒᵖ`. -/
def PullbackCone.isLimitCoyonedaEquiv (c : PullbackCone f g) :
    IsLimit c ≃ ∀ (X : Cᵒᵖ), IsLimit (c.map (coyoneda.obj X)) :=
  (Cone.isLimitCoyonedaEquiv c).trans
    (Equiv.piCongrRight (fun X ↦ c.isLimitMapConeEquiv (coyoneda.obj X)))


/-- The image of a pullback cone by a functor. -/
abbrev map : PushoutCocone (G.map f) (G.map g) :=
                                                   /-
                                                     C : Type u₁
                                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                     D : Type u₂
                                                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                     W X Y : C
                                                     f : Quiver.Hom W X
                                                     g : Quiver.Hom W Y
                                                     c : CategoryTheory.Limits.PushoutCocone f g
                                                     G : CategoryTheory.Functor C D
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map c.inl)) (CategoryThe …
                                                   -/
  PushoutCocone.mk (G.map c.inl) (G.map c.inr) (by simpa using G.congr_map c.condition)
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- The map (as a cocone) of a pushout cocone is colimit iff
the map (as a pushout cocone) is limit. -/
def isColimitMapCoconeEquiv :
    IsColimit (mapCocone G c) ≃ IsColimit (c.map G) :=
  (IsColimit.precomposeHomEquiv (diagramIsoSpan.{v₂} _).symm _).symm.trans <|
    IsColimit.equivIsoColimit <| by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        W X Y : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        c : CategoryTheory.Limits.PushoutCocone f g
        G : CategoryTheory.Functor C D
        ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cocones.precompose (CategoryTheor …
      -/
      refine PushoutCocone.ext (Iso.refl _) ?_ ?_
        /-
          case refine_1
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          W X Y : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          c : CategoryTheory.Limits.PushoutCocone f g
          G : CategoryTheory.Functor C D
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCocone. …
        -/
      · dsimp only [inl]
        /-
          case refine_1
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          W X Y : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          c : CategoryTheory.Limits.PushoutCocone f g
          G : CategoryTheory.Functor C D
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          W X Y : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          c : CategoryTheory.Limits.PushoutCocone f g
          G : CategoryTheory.Functor C D
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCocone. …
        -/
      · dsimp only [inr]
        /-
          case refine_2
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          W X Y : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          c : CategoryTheory.Limits.PushoutCocone f g
          G : CategoryTheory.Functor C D
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
        -/
        simp
        /-
          🎉 no goals
        -/


/-- The map of a pushout cocone is a colimit iff the cofork consisting of the mapped morphisms is a
colimit. This essentially lets us commute `PushoutCocone.mk` with `Functor.mapCocone`. -/
def isColimitMapCoconePushoutCoconeEquiv :
    IsColimit (mapCocone G (PushoutCocone.mk h k comm)) ≃
      IsColimit
                                                  /-
                                                    C : Type u₁
                                                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                    D : Type u₂
                                                    inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                    G : CategoryTheory.Functor C D
                                                    W X Y Z : C
                                                    h : Quiver.Hom X Z
                                                    k : Quiver.Hom Y Z
                                                    f : Quiver.Hom W X
                                                    g : Quiver.Hom W Y
                                                    comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) (CategoryTheory. …
                                                  -/
        (PushoutCocone.mk (G.map h) (G.map k) (by simp only [← G.map_comp, comm]) :
                                                  /-
                                                    🎉 no goals
                                                  -/
          PushoutCocone (G.map f) (G.map g)) :=
  (IsColimit.precomposeHomEquiv (diagramIsoSpan.{v₂} _).symm _).symm.trans <|
    IsColimit.equivIsoColimit <|
      Cocones.ext (Iso.refl _) <| by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          W X Y Z : C
          h : Quiver.Hom X Z
          k : Quiver.Hom Y Z
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
          ⊢ ∀ (j : CategoryTheory.Limits.WalkingSpan), Eq (CategoryTheory.CategoryStruct …
        -/
        rintro (_ | _ | _) <;> dsimp <;>
          /-
            case none
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            G : CategoryTheory.Functor C D
            W X Y Z : C
            h : Quiver.Hom X Z
            k : Quiver.Hom Y Z
            f : Quiver.Hom W X
            g : Quiver.Hom W Y
            comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          simp only [Category.comp_id, Category.id_comp, ← G.map_comp]
          /-
            🎉 no goals
          -/


/-- The property of preserving pushouts expressed in terms of binary cofans. -/
def isColimitPushoutCoconeMapOfIsColimit [PreservesColimit (span f g) G]
    (l : IsColimit (PushoutCocone.mk h k comm)) :
    IsColimit (PushoutCocone.mk (G.map h) (G.map k) (show G.map f ≫ G.map h = G.map g ≫ G.map k
              /-
                C : Type u₁
                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                G : CategoryTheory.Functor C D
                W X Y Z : C
                h : Quiver.Hom X Z
                k : Quiver.Hom Y Z
                f : Quiver.Hom W X
                g : Quiver.Hom W Y
                comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
                inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
                l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) (CategoryTheory. …
              -/
      from by simp only [← G.map_comp,comm] )) :=
              /-
                🎉 no goals
              -/
  isColimitMapCoconePushoutCoconeEquiv G comm (isColimitOfPreserves G l)


/-- The property of reflecting pushouts expressed in terms of binary cofans. -/
def isColimitOfIsColimitPushoutCoconeMap [ReflectsColimit (span f g) G]
    (l : IsColimit (PushoutCocone.mk (G.map h) (G.map k) (show G.map f ≫ G.map h =
                                /-
                                  C : Type u₁
                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                  D : Type u₂
                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                  G : CategoryTheory.Functor C D
                                  W X Y Z : C
                                  h : Quiver.Hom X Z
                                  k : Quiver.Hom Y Z
                                  f : Quiver.Hom W X
                                  g : Quiver.Hom W Y
                                  comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
                                  inst✝ : CategoryTheory.Limits.ReflectsColimit (CategoryTheory.Limits.span f g) G
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) (CategoryTheory. …
                                -/
      G.map g ≫ G.map k from by simp only [← G.map_comp,comm]))) :
                                /-
                                  🎉 no goals
                                -/
    IsColimit (PushoutCocone.mk h k comm) :=
  isColimitOfReflects G ((isColimitMapCoconePushoutCoconeEquiv G comm).symm l)


/-- If `G` preserves pushouts and `C` has them, then the pushout cocone constructed of the mapped
morphisms of the pushout cocone is a colimit. -/
def isColimitOfHasPushoutOfPreservesColimit [i : HasPushout f g] :
    IsColimit (PushoutCocone.mk (G.map (pushout.inl _ _)) (G.map (@pushout.inr _ _ _ _ _ f g i))
    (show G.map f ≫ G.map (pushout.inl _ _) = G.map g ≫ G.map (pushout.inr _ _) from by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        W X Y Z : C
        h : Quiver.Hom X Z
        k : Quiver.Hom Y Z
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
        i : CategoryTheory.Limits.HasPushout f g
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map (CategoryTheory.Limi …
      -/
      simp only [← G.map_comp, pushout.condition])) :=
      /-
        🎉 no goals
      -/
  isColimitPushoutCoconeMapOfIsColimit G _ (pushoutIsPushout f g)


/-- If `F` preserves the pushout of `f, g`, it also preserves the pushout of `g, f`. -/
lemma preservesPushout_symmetry : PreservesColimit (span g f) G where
  preserves {c} hc := ⟨by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      W X Y : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span g f)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone c)
    -/
    apply (IsColimit.precomposeHomEquiv (diagramIsoSpan.{v₂} _).symm _).toFun
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      W X Y : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span g f)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompose ( …
    -/
    apply IsColimit.ofIsoColimit _ (PushoutCocone.isoMk _).symm
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      W X Y : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span g f)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk ((G. …
    -/
    apply PushoutCocone.isColimitOfFlip
    /-
      case ht
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      W X Y : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span g f)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk ((G. …
    -/
    apply (isColimitMapCoconePushoutCoconeEquiv _ _).toFun
      /-
        case ht
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        G : CategoryTheory.Functor C D
        W X Y : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span g f)
        hc : CategoryTheory.Limits.IsColimit c
        ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone (CategoryTheory.Limits.PushoutC …
      -/
    · refine @isColimitOfPreserves _ _ _ _ _ _ _ _ _ ?_ ?_ -- Porting note: more TC coddling
        /-
          case ht.refine_1
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          W X Y : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
          c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span g f)
          hc : CategoryTheory.Limits.IsColimit c
          ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk (c.ι …
        -/
      · exact PushoutCocone.flipIsColimit hc
        /-
          🎉 no goals
        -/
        /-
          case ht.refine_2
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          W X Y : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
          c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span g f)
          hc : CategoryTheory.Limits.IsColimit c
          ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span ((Categor …
        -/
      · dsimp
        /-
          case ht.refine_2
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          W X Y : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
          c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.span g f)
          hc : CategoryTheory.Limits.IsColimit c
          ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g) G
        -/
        infer_instance⟩
        /-
          🎉 no goals
        -/


theorem hasPushout_of_preservesPushout [HasPushout f g] : HasPushout (G.map f) (G.map g) :=
  ⟨⟨⟨_, isColimitPushoutCoconeMapOfIsColimit G _ (pushoutIsPushout _ _)⟩⟩⟩


/-- If `G` preserves the pushout of `(f,g)`, then the pushout comparison map for `G` at `(f,g)` is
an isomorphism. -/
def PreservesPushout.iso : pushout (G.map f) (G.map g) ≅ G.obj (pushout f g) :=
  IsColimit.coconePointUniqueUpToIso (colimit.isColimit _)
    (isColimitOfHasPushoutOfPreservesColimit G f g)


@[simp]
theorem PreservesPushout.iso_hom : (PreservesPushout.iso G f g).hom = pushoutComparison G f g :=
  rfl


@[reassoc]
theorem PreservesPushout.inl_iso_hom :
    pushout.inl _ _ ≫ (PreservesPushout.iso G f g).hom = G.map (pushout.inl _ _) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    W X Y : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    inst✝² : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f  …
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl (G …
  -/
  delta PreservesPushout.iso
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    W X Y : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    inst✝² : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f  …
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl (G …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem PreservesPushout.inr_iso_hom :
    pushout.inr _ _ ≫ (PreservesPushout.iso G f g).hom = G.map (pushout.inr _ _) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    W X Y : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    inst✝² : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f  …
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr (G …
  -/
  delta PreservesPushout.iso
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    W X Y : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    inst✝² : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f  …
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr (G …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem PreservesPushout.inl_iso_inv :
    G.map (pushout.inl _ _) ≫ (PreservesPushout.iso G f g).inv = pushout.inl _ _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    W X Y : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    inst✝² : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f  …
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.pushout …
  -/
  simp [PreservesPushout.iso, Iso.comp_inv_eq]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem PreservesPushout.inr_iso_inv :
    G.map (pushout.inr _ _) ≫ (PreservesPushout.iso G f g).inv = pushout.inr _ _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    W X Y : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    inst✝² : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f  …
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.pushout …
  -/
  simp [PreservesPushout.iso, Iso.comp_inv_eq]
  /-
    🎉 no goals
  -/


/-- If the pullback comparison map for `G` at `(f,g)` is an isomorphism, then `G` preserves the
pullback of `(f,g)`. -/
lemma PreservesPullback.of_iso_comparison [i : IsIso (pullbackComparison G f g)] :
    PreservesLimit (cospan f g) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.pullbackComparison G f g)
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g) G
  -/
  apply preservesLimit_of_preserves_limit_cone (pullbackIsPullback f g)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.pullbackComparison G f g)
    ⊢ CategoryTheory.Limits.IsLimit (G.mapCone (CategoryTheory.Limits.PullbackCone …
  -/
  apply (isLimitMapConePullbackConeEquiv _ _).symm _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.pullbackComparison G f g)
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (G.map  …
  -/
  exact @IsLimit.ofPointIso _ _ _ _ _ _ _ (limit.isLimit (cospan (G.map f) (G.map g))) i
  /-
    🎉 no goals
  -/


instance : IsIso (pullbackComparison G f g) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullbackComparison G f g)
  -/
  rw [← PreservesPullback.iso_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesPullback.iso G f g).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If the pushout comparison map for `G` at `(f,g)` is an isomorphism, then `G` preserves the
pushout of `(f,g)`. -/
lemma PreservesPushout.of_iso_comparison [i : IsIso (pushoutComparison G f g)] :
    PreservesColimit (span f g) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.pushoutComparison G f g)
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g) G
  -/
  apply preservesColimit_of_preserves_colimit_cocone (pushoutIsPushout f g)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.pushoutComparison G f g)
    ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone (CategoryTheory.Limits.PushoutC …
  -/
  apply (isColimitMapCoconePushoutCoconeEquiv _ _).symm _
  -- Porting note: apply no longer creates goals for instances
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.pushoutComparison G f g)
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk (G.m …
  -/
  exact @IsColimit.ofPointIso _ _ _ _ _ _ _ (colimit.isColimit (span (G.map f) (G.map g))) i
  /-
    🎉 no goals
  -/


instance : IsIso (pushoutComparison G f g) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝² : CategoryTheory.Limits.HasPushout f g
    inst✝¹ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pushoutComparison G f g)
  -/
  rw [← PreservesPushout.iso_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝² : CategoryTheory.Limits.HasPushout f g
    inst✝¹ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesPushout.iso G f g).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A pushout cocone in `C` is colimit iff it becomes limit
after the application of `yoneda.obj X` for all `X : C`. -/
def PushoutCocone.isColimitYonedaEquiv (c : PushoutCocone f g) :
    IsColimit c ≃ ∀ (X : C), IsLimit (c.op.map (yoneda.obj X)) :=
  (Limits.Cocone.isColimitYonedaEquiv c).trans
    (Equiv.piCongrRight (fun X ↦
      (IsLimit.whiskerEquivalenceEquiv walkingSpanOpEquiv.symm).trans
        ((IsLimit.postcomposeHomEquiv
          (isoWhiskerRight (cospanOp f g).symm (yoneda.obj X)) _).symm.trans
            (Equiv.trans (IsLimit.equivIsoLimit
                  /-
                    C : Type u₁
                    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                    G : CategoryTheory.Functor C D
                    X✝ Y Z : C
                    f : Quiver.Hom X✝ Y
                    g : Quiver.Hom X✝ Z
                    inst✝² : CategoryTheory.Limits.HasPushout f g
                    inst✝¹ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
                    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f g …
                    c : CategoryTheory.Limits.PushoutCocone f g
                    X : C
                    ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cones.postcompose (CategoryTheory …
                  -/
              (by exact Cones.ext (Iso.refl _) (by rintro (_|_|_) <;> simp)))
                  /-
                    🎉 no goals
                  -/
                (c.op.isLimitMapConeEquiv (yoneda.obj X))))))


