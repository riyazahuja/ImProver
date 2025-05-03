/-- The first projection from the pullback. -/
abbrev pullbackFst (f : X ⟶ Z) (g : Y ⟶ Z) : TopCat.of { p : X × Y // f p.1 = g p.2 } ⟶ X :=
  ⟨Prod.fst ∘ Subtype.val, by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      ⊢ Continuous (Function.comp Prod.fst Subtype.val)
    -/
                              /-
                                🎉 no goals
                              -/
    apply Continuous.comp <;> set_option tactic.skipAssignedInstances false in continuity⟩
                              /-
                                🎉 no goals
                              -/


lemma pullbackFst_apply (f : X ⟶ Z) (g : Y ⟶ Z) (x) : pullbackFst f g x = x.1.1 := rfl


/-- The second projection from the pullback. -/
abbrev pullbackSnd (f : X ⟶ Z) (g : Y ⟶ Z) : TopCat.of { p : X × Y // f p.1 = g p.2 } ⟶ Y :=
  ⟨Prod.snd ∘ Subtype.val, by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      ⊢ Continuous (Function.comp Prod.snd Subtype.val)
    -/
                              /-
                                🎉 no goals
                              -/
    apply Continuous.comp <;> set_option tactic.skipAssignedInstances false in continuity⟩
                              /-
                                🎉 no goals
                              -/


lemma pullbackSnd_apply (f : X ⟶ Z) (g : Y ⟶ Z) (x) : pullbackSnd f g x = x.1.2 := rfl


/-- The explicit pullback cone of `X, Y` given by `{ p : X × Y // f p.1 = g p.2 }`. -/
def pullbackCone (f : X ⟶ Z) (g : Y ⟶ Z) : PullbackCone f g :=
  PullbackCone.mk (pullbackFst f g) (pullbackSnd f g)
    (by
      /-
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.pullbackFst f g) f) (Category …
      -/
      dsimp [pullbackFst, pullbackSnd, Function.comp_def]
      /-
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => (↑x).1, continuou …
      -/
      ext ⟨x, h⟩
      -- Next 2 lines were
      -- `rw [comp_apply, ContinuousMap.coe_mk, comp_apply, ContinuousMap.coe_mk]`
      -- `exact h` before https://github.com/leanprover/lean4/pull/2644
      /-
        case w.mk
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        x : Prod ↑X ↑Y
        h : Eq (f x.1) (g x.2)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := fun x => (↑x).1, continuo …
      -/
      rw [comp_apply, comp_apply]
      /-
        case w.mk
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        x : Prod ↑X ↑Y
        h : Eq (f x.1) (g x.2)
        ⊢ Eq (f ({ toFun := fun x => (↑x).1, continuous_toFun := ⋯ } ⟨x, h⟩)) (g ({ to …
      -/
      congr!)
      /-
        🎉 no goals
      -/


/-- The constructed cone is a limit. -/
def pullbackConeIsLimit (f : X ⟶ Z) (g : Y ⟶ Z) : IsLimit (pullbackCone f g) :=
  PullbackCone.isLimitAux' _
    (by
      /-
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        ⊢ (s : CategoryTheory.Limits.PullbackCone f g) → Subtype fun l => And (Eq (Cat …
      -/
      intro S
      /-
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        S : CategoryTheory.Limits.PullbackCone f g
        ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (TopCat.pullb …
      -/
      constructor; swap
      · exact
          { toFun := fun x =>
              ⟨⟨S.fst x, S.snd x⟩, by simpa using ConcreteCategory.congr_hom S.condition x⟩
            continuous_toFun := by
              apply Continuous.subtype_mk <| Continuous.prod_mk ?_ ?_
              · exact (PullbackCone.fst S)|>.continuous_toFun
              · exact (PullbackCone.snd S)|>.continuous_toFun
          }
      /-
        case property
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        S : CategoryTheory.Limits.PullbackCone f g
        ⊢ And (Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨{ fst := S. …
      -/
      refine ⟨?_, ?_, ?_⟩
        /-
          case property.refine_1
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨{ fst := S.fst x …
        -/
      · delta pullbackCone
        /-
          case property.refine_1
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨{ fst := S.fst x …
        -/
        ext a
        -- This used to be `rw`, but we need `rw; rfl` after https://github.com/leanprover/lean4/pull/2644
        /-
          case property.refine_1.w
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          a : (CategoryTheory.forget TopCat).obj S.pt
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨{ fst := S.fst  …
        -/
        rw [comp_apply, ContinuousMap.coe_mk]
        /-
          case property.refine_1.w
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          a : (CategoryTheory.forget TopCat).obj S.pt
          ⊢ Eq ((CategoryTheory.Limits.PullbackCone.mk (TopCat.pullbackFst f g) (TopCat. …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case property.refine_2
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨{ fst := S.fst x …
        -/
      · delta pullbackCone
        /-
          case property.refine_2
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨{ fst := S.fst x …
        -/
        ext a
        -- This used to be `rw`, but we need `rw; rfl` after https://github.com/leanprover/lean4/pull/2644
        /-
          case property.refine_2.w
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          a : (CategoryTheory.forget TopCat).obj S.pt
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨{ fst := S.fst  …
        -/
        rw [comp_apply, ContinuousMap.coe_mk]
        /-
          case property.refine_2.w
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          a : (CategoryTheory.forget TopCat).obj S.pt
          ⊢ Eq ((CategoryTheory.Limits.PullbackCone.mk (TopCat.pullbackFst f g) (TopCat. …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case property.refine_3
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          ⊢ ∀ {m : Quiver.Hom S.pt (TopCat.pullbackCone f g).pt}, Eq (CategoryTheory.Cat …
        -/
      · intro m h₁ h₂
        -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): used to be `ext x`.
        /-
          case property.refine_3
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          m : Quiver.Hom S.pt (TopCat.pullbackCone f g).pt
          h₁ : Eq (CategoryTheory.CategoryStruct.comp m (TopCat.pullbackCone f g).fst) S …
          h₂ : Eq (CategoryTheory.CategoryStruct.comp m (TopCat.pullbackCone f g).snd) S …
          ⊢ Eq m { toFun := fun x => ⟨{ fst := S.fst x, snd := S.snd x }, ⋯⟩, continuous …
        -/
        apply ContinuousMap.ext; intro x
        /-
          case property.refine_3.h
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          m : Quiver.Hom S.pt (TopCat.pullbackCone f g).pt
          h₁ : Eq (CategoryTheory.CategoryStruct.comp m (TopCat.pullbackCone f g).fst) S …
          h₂ : Eq (CategoryTheory.CategoryStruct.comp m (TopCat.pullbackCone f g).snd) S …
          x : ↑S.pt
          ⊢ Eq (m x) ({ toFun := fun x => ⟨{ fst := S.fst x, snd := S.snd x }, ⋯⟩, conti …
        -/
        apply Subtype.ext
        /-
          case property.refine_3.h.a
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          S : CategoryTheory.Limits.PullbackCone f g
          m : Quiver.Hom S.pt (TopCat.pullbackCone f g).pt
          h₁ : Eq (CategoryTheory.CategoryStruct.comp m (TopCat.pullbackCone f g).fst) S …
          h₂ : Eq (CategoryTheory.CategoryStruct.comp m (TopCat.pullbackCone f g).snd) S …
          x : ↑S.pt
          ⊢ Eq ↑(m x) ↑({ toFun := fun x => ⟨{ fst := S.fst x, snd := S.snd x }, ⋯⟩, con …
        -/
        apply Prod.ext
          /-
            case property.refine_3.h.a.fst
            J : Type v
            inst✝ : CategoryTheory.Category.{w, v} J
            X Y Z : TopCat
            f : Quiver.Hom X Z
            g : Quiver.Hom Y Z
            S : CategoryTheory.Limits.PullbackCone f g
            m : Quiver.Hom S.pt (TopCat.pullbackCone f g).pt
            h₁ : Eq (CategoryTheory.CategoryStruct.comp m (TopCat.pullbackCone f g).fst) S …
            h₂ : Eq (CategoryTheory.CategoryStruct.comp m (TopCat.pullbackCone f g).snd) S …
            x : ↑S.pt
            ⊢ Eq (↑(m x)).1 (↑({ toFun := fun x => ⟨{ fst := S.fst x, snd := S.snd x }, ⋯⟩ …
          -/
        · simpa using ConcreteCategory.congr_hom h₁ x
          /-
            🎉 no goals
          -/
          /-
            case property.refine_3.h.a.snd
            J : Type v
            inst✝ : CategoryTheory.Category.{w, v} J
            X Y Z : TopCat
            f : Quiver.Hom X Z
            g : Quiver.Hom Y Z
            S : CategoryTheory.Limits.PullbackCone f g
            m : Quiver.Hom S.pt (TopCat.pullbackCone f g).pt
            h₁ : Eq (CategoryTheory.CategoryStruct.comp m (TopCat.pullbackCone f g).fst) S …
            h₂ : Eq (CategoryTheory.CategoryStruct.comp m (TopCat.pullbackCone f g).snd) S …
            x : ↑S.pt
            ⊢ Eq (↑(m x)).2 (↑({ toFun := fun x => ⟨{ fst := S.fst x, snd := S.snd x }, ⋯⟩ …
          -/
        · simpa using ConcreteCategory.congr_hom h₂ x)
          /-
            🎉 no goals
          -/


/-- The pullback of two maps can be identified as a subspace of `X × Y`. -/
def pullbackIsoProdSubtype (f : X ⟶ Z) (g : Y ⟶ Z) :
    pullback f g ≅ TopCat.of { p : X × Y // f p.1 = g p.2 } :=
  (limit.isLimit _).conePointUniqueUpToIso (pullbackConeIsLimit f g)


@[reassoc (attr := simp)]
theorem pullbackIsoProdSubtype_inv_fst (f : X ⟶ Z) (g : Y ⟶ Z) :
    (pullbackIsoProdSubtype f g).inv ≫ pullback.fst _ _ = pullbackFst f g := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.pullbackIsoProdSubtype f g).i …
  -/
  simp [pullbackCone, pullbackIsoProdSubtype]
  /-
    🎉 no goals
  -/


theorem pullbackIsoProdSubtype_inv_fst_apply (f : X ⟶ Z) (g : Y ⟶ Z)
    (x : { p : X × Y // f p.1 = g p.2 }) :
    pullback.fst f g ((pullbackIsoProdSubtype f g).inv x) = (x : X × Y).fst :=
  ConcreteCategory.congr_hom (pullbackIsoProdSubtype_inv_fst f g) x


@[reassoc (attr := simp)]
theorem pullbackIsoProdSubtype_inv_snd (f : X ⟶ Z) (g : Y ⟶ Z) :
    (pullbackIsoProdSubtype f g).inv ≫ pullback.snd _ _ = pullbackSnd f g := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.pullbackIsoProdSubtype f g).i …
  -/
  simp [pullbackCone, pullbackIsoProdSubtype]
  /-
    🎉 no goals
  -/


theorem pullbackIsoProdSubtype_inv_snd_apply (f : X ⟶ Z) (g : Y ⟶ Z)
    (x : { p : X × Y // f p.1 = g p.2 }) :
    pullback.snd f g ((pullbackIsoProdSubtype f g).inv x) = (x : X × Y).snd :=
  ConcreteCategory.congr_hom (pullbackIsoProdSubtype_inv_snd f g) x


theorem pullbackIsoProdSubtype_hom_fst (f : X ⟶ Z) (g : Y ⟶ Z) :
    (pullbackIsoProdSubtype f g).hom ≫ pullbackFst f g = pullback.fst _ _ := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.pullbackIsoProdSubtype f g).h …
  -/
  rw [← Iso.eq_inv_comp, pullbackIsoProdSubtype_inv_fst]
  /-
    🎉 no goals
  -/


theorem pullbackIsoProdSubtype_hom_snd (f : X ⟶ Z) (g : Y ⟶ Z) :
    (pullbackIsoProdSubtype f g).hom ≫ pullbackSnd f g = pullback.snd _ _ := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.pullbackIsoProdSubtype f g).h …
  -/
  rw [← Iso.eq_inv_comp, pullbackIsoProdSubtype_inv_snd]
  /-
    🎉 no goals
  -/

-- Porting note: why do I need to tell Lean to coerce pullback to a type

theorem pullbackIsoProdSubtype_hom_apply {f : X ⟶ Z} {g : Y ⟶ Z}
    (x : ConcreteCategory.forget.obj (pullback f g)) :
    (pullbackIsoProdSubtype f g).hom x =
      ⟨⟨pullback.fst f g x, pullback.snd f g x⟩, by
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          X Y Z : TopCat
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          x : CategoryTheory.ConcreteCategory.forget.obj (CategoryTheory.Limits.pullback …
          ⊢ Eq (f { fst := (CategoryTheory.Limits.pullback.fst f g) x, snd := (CategoryT …
        -/
        simpa using ConcreteCategory.congr_hom pullback.condition x⟩ := by
        /-
          🎉 no goals
        -/
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    x : CategoryTheory.ConcreteCategory.forget.obj (CategoryTheory.Limits.pullback …
    ⊢ Eq ((TopCat.pullbackIsoProdSubtype f g).hom x) ⟨{ fst := (CategoryTheory.Lim …
  -/
  apply Subtype.ext; apply Prod.ext
  exacts [ConcreteCategory.congr_hom (pullbackIsoProdSubtype_hom_fst f g) x,
    ConcreteCategory.congr_hom (pullbackIsoProdSubtype_hom_snd f g) x]


theorem pullback_topology {X Y Z : TopCat.{u}} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (pullback f g).str =
      induced (pullback.fst f g) X.str ⊓
        induced (pullback.snd f g) Y.str := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.Limits.pullback f g).str (Min.min (TopologicalSpace.induc …
  -/
  let homeo := homeoOfIso (pullbackIsoProdSubtype f g)
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    homeo : Homeomorph ↑(CategoryTheory.Limits.pullback f g) ↑(TopCat.of (Subtype  …
    ⊢ Eq (CategoryTheory.Limits.pullback f g).str (Min.min (TopologicalSpace.induc …
  -/
  refine homeo.isInducing.eq_induced.trans ?_
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    homeo : Homeomorph ↑(CategoryTheory.Limits.pullback f g) ↑(TopCat.of (Subtype  …
    ⊢ Eq (TopologicalSpace.induced (⇑homeo) (TopCat.of (Subtype fun p => Eq (f p.1 …
  -/
  change induced homeo (induced _ ( (induced Prod.fst X.str) ⊓ (induced Prod.snd Y.str))) = _
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    homeo : Homeomorph ↑(CategoryTheory.Limits.pullback f g) ↑(TopCat.of (Subtype  …
    ⊢ Eq (TopologicalSpace.induced (⇑homeo) (TopologicalSpace.induced Subtype.val  …
  -/
  simp only [induced_compose, induced_inf]
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    homeo : Homeomorph ↑(CategoryTheory.Limits.pullback f g) ↑(TopCat.of (Subtype  …
    ⊢ Eq (Min.min (TopologicalSpace.induced (Function.comp (Function.comp Prod.fst …
  -/
  congr
  /-
    🎉 no goals
  -/


theorem range_pullback_to_prod {X Y Z : TopCat} (f : X ⟶ Z) (g : Y ⟶ Z) :
    Set.range (prod.lift (pullback.fst f g) (pullback.snd f g)) =
      { x | (Limits.prod.fst ≫ f) x = (Limits.prod.snd ≫ g) x } := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (Set.range ⇑(CategoryTheory.Limits.prod.lift (CategoryTheory.Limits.pullb …
  -/
  ext x
  /-
    case h
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    x : ↑(CategoryTheory.Limits.prod X Y)
    ⊢ Iff (Membership.mem (Set.range ⇑(CategoryTheory.Limits.prod.lift (CategoryTh …
  -/
  constructor
    /-
      case h.mp
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      x : ↑(CategoryTheory.Limits.prod X Y)
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.prod.lift (CategoryTheory. …
    -/
  · rintro ⟨y, rfl⟩
    /-
      case h.mp.intro
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      y : ↑(CategoryTheory.Limits.pullback f g)
      ⊢ Membership.mem (setOf fun x => Eq ((CategoryTheory.CategoryStruct.comp Categ …
    -/
    change (_ ≫ _ ≫ f) _ = (_ ≫ _ ≫ g) _ -- new `change` after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      case h.mp.intro
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      y : ↑(CategoryTheory.Limits.pullback f g)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (Ca …
    -/
    simp [pullback.condition]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      x : ↑(CategoryTheory.Limits.prod X Y)
      ⊢ Membership.mem (setOf fun x => Eq ((CategoryTheory.CategoryStruct.comp Categ …
    -/
  · rintro (h : f (_, _).1 = g (_, _).2)
    /-
      case h.mpr
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      x : ↑(CategoryTheory.Limits.prod X Y)
      h : Eq (f { fst := CategoryTheory.Limits.prod.fst x, snd := ?m.36453 }.1) (g { …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.prod.lift (CategoryTheory. …
    -/
    use (pullbackIsoProdSubtype f g).inv ⟨⟨_, _⟩, h⟩
    /-
      case h
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      x : ↑(CategoryTheory.Limits.prod X Y)
      h : Eq (f { fst := CategoryTheory.Limits.prod.fst x, snd := CategoryTheory.Lim …
      ⊢ Eq ((CategoryTheory.Limits.prod.lift (CategoryTheory.Limits.pullback.fst f g …
    -/
    change (forget TopCat).map _ _ = _ -- new `change` after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      case h
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      x : ↑(CategoryTheory.Limits.prod X Y)
      h : Eq (f { fst := CategoryTheory.Limits.prod.fst x, snd := CategoryTheory.Lim …
      ⊢ Eq ((CategoryTheory.forget TopCat).map (CategoryTheory.Limits.prod.lift (Cat …
    -/
    apply Concrete.limit_ext
    /-
      case h.a
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      x : ↑(CategoryTheory.Limits.prod X Y)
      h : Eq (f { fst := CategoryTheory.Limits.prod.fst x, snd := CategoryTheory.Lim …
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq ((Cate …
    -/
    rintro ⟨⟨⟩⟩ <;>
    /-
      case h.a.mk.left
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      x : ↑(CategoryTheory.Limits.prod X Y)
      h : Eq (f { fst := CategoryTheory.Limits.prod.fst x, snd := CategoryTheory.Lim …
      ⊢ Eq ((CategoryTheory.Limits.limit.π (CategoryTheory.Limits.pair X Y) { as :=  …
    -/
    erw [← comp_apply, ← comp_apply, limit.lift_π] <;> -- now `erw` after https://github.com/leanprover-community/mathlib4/pull/13170
    -- This used to be `simp` before https://github.com/leanprover/lean4/pull/2644
    /-
      case h.a.mk.left
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      x : ↑(CategoryTheory.Limits.prod X Y)
      h : Eq (f { fst := CategoryTheory.Limits.prod.fst x, snd := CategoryTheory.Lim …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (TopCat.pullbackIsoProdSubtype f g). …
    -/
    /-
      🎉 no goals
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


/-- The pullback along an embedding is (isomorphic to) the preimage. -/
noncomputable
def pullbackHomeoPreimage
    {X Y Z : Type*} [TopologicalSpace X] [TopologicalSpace Y] [TopologicalSpace Z]
    (f : X → Z) (hf : Continuous f) (g : Y → Z) (hg : IsEmbedding g) :
    { p : X × Y // f p.1 = g p.2 } ≃ₜ f ⁻¹' Set.range g where
  toFun := fun x ↦ ⟨x.1.1, _, x.2.symm⟩
  invFun := fun x ↦ ⟨⟨x.1, Exists.choose x.2⟩, (Exists.choose_spec x.2).symm⟩
  left_inv := by
    /-
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      ⊢ Function.LeftInverse (fun x => ⟨{ fst := ↑x, snd := Exists.choose ⋯ }, ⋯⟩) f …
    -/
    intro x
    /-
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      x : Subtype fun p => Eq (f p.1) (g p.2)
      ⊢ Eq ((fun x => ⟨{ fst := ↑x, snd := Exists.choose ⋯ }, ⋯⟩) ((fun x => ⟨(↑x).1 …
    -/
            /-
              🎉 no goals
            -/
    ext <;> dsimp
    /-
      case a.snd
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      x : Subtype fun p => Eq (f p.1) (g p.2)
      ⊢ Eq (Exists.choose ⋯) (↑x).2
    -/
    apply hg.injective
    /-
      case a.snd.a
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      x : Subtype fun p => Eq (f p.1) (g p.2)
      ⊢ Eq (g (Exists.choose ⋯)) (g (↑x).2)
    -/
    convert x.prop
    /-
      case h.e'_2
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      x : Subtype fun p => Eq (f p.1) (g p.2)
      ⊢ Eq (g (Exists.choose ⋯)) (f (↑x).1)
    -/
    exact Exists.choose_spec (p := fun y ↦ g y = f (↑x : X × Y).1) _
    /-
      🎉 no goals
    -/
  right_inv := fun _ ↦ rfl
  continuous_toFun := by
    /-
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      ⊢ Continuous { toFun := fun x => ⟨(↑x).1, ⋯⟩, invFun := fun x => ⟨{ fst := ↑x, …
    -/
    apply Continuous.subtype_mk
    /-
      case h
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      ⊢ Continuous fun x => (↑x).1
    -/
    exact continuous_fst.comp continuous_subtype_val
    /-
      🎉 no goals
    -/
  continuous_invFun := by
    /-
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      ⊢ Continuous { toFun := fun x => ⟨(↑x).1, ⋯⟩, invFun := fun x => ⟨{ fst := ↑x, …
    -/
    apply Continuous.subtype_mk
    /-
      case h
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      ⊢ Continuous fun x => { fst := ↑x, snd := Exists.choose ⋯ }
    -/
    refine continuous_prod_mk.mpr ⟨continuous_subtype_val, hg.isInducing.continuous_iff.mpr ?_⟩
    /-
      case h
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      ⊢ Continuous (Function.comp g fun x => Exists.choose ⋯)
    -/
    convert hf.comp continuous_subtype_val
    /-
      case h.e'_5.h
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      e_1✝ : Eq (↑(Set.preimage f (Set.range g))) (Subtype fun x => Membership.mem ( …
      ⊢ Eq (Function.comp g fun x => Exists.choose ⋯) (Function.comp f Subtype.val)
    -/
    ext x
    /-
      case h.e'_5.h.h
      J : Type v
      inst✝³ : CategoryTheory.Category.{w, v} J
      X✝ Y✝ Z✝ : TopCat
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : X → Z
      hf : Continuous f
      g : Y → Z
      hg : Topology.IsEmbedding g
      e_1✝ : Eq (↑(Set.preimage f (Set.range g))) (Subtype fun x => Membership.mem ( …
      x : ↑(Set.preimage f (Set.range g))
      ⊢ Eq (Function.comp g (fun x => Exists.choose ⋯) x) (Function.comp f Subtype.v …
    -/
    exact Exists.choose_spec x.2
    /-
      🎉 no goals
    -/


theorem isInducing_pullback_to_prod {X Y Z : TopCat.{u}} (f : X ⟶ Z) (g : Y ⟶ Z) :
    IsInducing <| ⇑(prod.lift (pullback.fst f g) (pullback.snd f g)) :=
      /-
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        ⊢ Eq (CategoryTheory.Limits.pullback f g).topologicalSpace_coe (TopologicalSpa …
      -/
  ⟨by simp [topologicalSpace_coe, prod_topology, pullback_topology, induced_compose, ← coe_comp]⟩
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-28")] alias inducing_pullback_to_prod := isInducing_pullback_to_prod


theorem isEmbedding_pullback_to_prod {X Y Z : TopCat.{u}} (f : X ⟶ Z) (g : Y ⟶ Z) :
    IsEmbedding <| ⇑(prod.lift (pullback.fst f g) (pullback.snd f g)) :=
  ⟨isInducing_pullback_to_prod f g, (TopCat.mono_iff_injective _).mp inferInstance⟩


@[deprecated (since := "2024-10-26")]
alias embedding_pullback_to_prod := isEmbedding_pullback_to_prod


/-- If the map `S ⟶ T` is mono, then there is a description of the image of `W ×ₛ X ⟶ Y ×ₜ Z`. -/
theorem range_pullback_map {W X Y Z S T : TopCat} (f₁ : W ⟶ S) (f₂ : X ⟶ S) (g₁ : Y ⟶ T)
    (g₂ : Z ⟶ T) (i₁ : W ⟶ Y) (i₂ : X ⟶ Z) (i₃ : S ⟶ T) [H₃ : Mono i₃] (eq₁ : f₁ ≫ i₃ = i₁ ≫ g₁)
    (eq₂ : f₂ ≫ i₃ = i₂ ≫ g₂) :
    Set.range (pullback.map f₁ f₂ g₁ g₂ i₁ i₂ i₃ eq₁ eq₂) =
      (pullback.fst g₁ g₂) ⁻¹' Set.range i₁ ∩ (pullback.snd g₁ g₂) ⁻¹' Set.range i₂ := by
  /-
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    i₃ : Quiver.Hom S T
    H₃ : CategoryTheory.Mono i₃
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    ⊢ Eq (Set.range ⇑(CategoryTheory.Limits.pullback.map f₁ f₂ g₁ g₂ i₁ i₂ i₃ eq₁  …
  -/
  ext
  /-
    case h
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    i₃ : Quiver.Hom S T
    H₃ : CategoryTheory.Mono i₃
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
    ⊢ Iff (Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f₁ f₂ g₁ …
  -/
  constructor
    /-
      case h.mp
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f₁ f₂ g₁ g₂ i …
    -/
  · rintro ⟨y, rfl⟩
    /-
      case h.mp.intro
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      y : ↑(CategoryTheory.Limits.pullback f₁ f₂)
      ⊢ Membership.mem (Inter.inter (Set.preimage (⇑(CategoryTheory.Limits.pullback. …
    -/
    simp only [Set.mem_inter_iff, Set.mem_preimage, Set.mem_range]
    /-
      case h.mp.intro
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      y : ↑(CategoryTheory.Limits.pullback f₁ f₂)
      ⊢ And (Exists fun y_1 => Eq (i₁ y_1) ((CategoryTheory.Limits.pullback.fst g₁ g …
    -/
    rw [← comp_apply, ← comp_apply]
    /-
      case h.mp.intro
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      y : ↑(CategoryTheory.Limits.pullback f₁ f₂)
      ⊢ And (Exists fun y_1 => Eq (i₁ y_1) ((CategoryTheory.CategoryStruct.comp (Cat …
    -/
    simp only [limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app, comp_apply]
    /-
      case h.mp.intro
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      y : ↑(CategoryTheory.Limits.pullback f₁ f₂)
      ⊢ And (Exists fun y_1 => Eq (i₁ y_1) (i₁ ((CategoryTheory.Limits.pullback.fst  …
    -/
    exact ⟨exists_apply_eq_apply _ _, exists_apply_eq_apply _ _⟩
    /-
      🎉 no goals
    -/
  /-
    case h.mpr
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    i₃ : Quiver.Hom S T
    H₃ : CategoryTheory.Mono i₃
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
    ⊢ Membership.mem (Inter.inter (Set.preimage (⇑(CategoryTheory.Limits.pullback. …
  -/
  rintro ⟨⟨x₁, hx₁⟩, ⟨x₂, hx₂⟩⟩
  have : f₁ x₁ = f₂ x₂ := by
    apply (TopCat.mono_iff_injective _).mp H₃
    rw [← comp_apply, eq₁, ← comp_apply, eq₂,
      comp_apply, comp_apply, hx₁, hx₂, ← comp_apply, pullback.condition]
    rfl -- `rfl` was not needed before https://github.com/leanprover-community/mathlib4/pull/13170
  /-
    case h.mpr.intro.intro.intro
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    i₃ : Quiver.Hom S T
    H₃ : CategoryTheory.Mono i₃
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
    x₁ : ↑W
    hx₁ : Eq (i₁ x₁) ((CategoryTheory.Limits.pullback.fst g₁ g₂) x✝)
    x₂ : ↑X
    hx₂ : Eq (i₂ x₂) ((CategoryTheory.Limits.pullback.snd g₁ g₂) x✝)
    this : Eq (f₁ x₁) (f₂ x₂)
    ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map f₁ f₂ g₁ g₂ i …
  -/
  use (pullbackIsoProdSubtype f₁ f₂).inv ⟨⟨x₁, x₂⟩, this⟩
  /-
    case h
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    i₃ : Quiver.Hom S T
    H₃ : CategoryTheory.Mono i₃
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
    x₁ : ↑W
    hx₁ : Eq (i₁ x₁) ((CategoryTheory.Limits.pullback.fst g₁ g₂) x✝)
    x₂ : ↑X
    hx₂ : Eq (i₂ x₂) ((CategoryTheory.Limits.pullback.snd g₁ g₂) x✝)
    this : Eq (f₁ x₁) (f₂ x₂)
    ⊢ Eq ((CategoryTheory.Limits.pullback.map f₁ f₂ g₁ g₂ i₁ i₂ i₃ eq₁ eq₂) ((TopC …
  -/
  change (forget TopCat).map _ _ = _
  /-
    case h
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    i₃ : Quiver.Hom S T
    H₃ : CategoryTheory.Mono i₃
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
    x₁ : ↑W
    hx₁ : Eq (i₁ x₁) ((CategoryTheory.Limits.pullback.fst g₁ g₂) x✝)
    x₂ : ↑X
    hx₂ : Eq (i₂ x₂) ((CategoryTheory.Limits.pullback.snd g₁ g₂) x✝)
    this : Eq (f₁ x₁) (f₂ x₂)
    ⊢ Eq ((CategoryTheory.forget TopCat).map (CategoryTheory.Limits.pullback.map f …
  -/
  apply Concrete.limit_ext
  /-
    case h.a
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    i₃ : Quiver.Hom S T
    H₃ : CategoryTheory.Mono i₃
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
    x₁ : ↑W
    hx₁ : Eq (i₁ x₁) ((CategoryTheory.Limits.pullback.fst g₁ g₂) x✝)
    x₂ : ↑X
    hx₂ : Eq (i₂ x₂) ((CategoryTheory.Limits.pullback.snd g₁ g₂) x✝)
    this : Eq (f₁ x₁) (f₂ x₂)
    ⊢ ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq ((CategoryTheory.Limits.limi …
  -/
  rintro (_ | _ | _) <;>
  /-
    case h.a.none
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    i₃ : Quiver.Hom S T
    H₃ : CategoryTheory.Mono i₃
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
    x₁ : ↑W
    hx₁ : Eq (i₁ x₁) ((CategoryTheory.Limits.pullback.fst g₁ g₂) x✝)
    x₂ : ↑X
    hx₂ : Eq (i₂ x₂) ((CategoryTheory.Limits.pullback.snd g₁ g₂) x✝)
    this : Eq (f₁ x₁) (f₂ x₂)
    ⊢ Eq ((CategoryTheory.Limits.limit.π (CategoryTheory.Limits.cospan g₁ g₂) Opti …
  -/
  erw [← comp_apply, ← comp_apply] -- now `erw` after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      case h.a.none
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
      x₁ : ↑W
      hx₁ : Eq (i₁ x₁) ((CategoryTheory.Limits.pullback.fst g₁ g₂) x✝)
      x₂ : ↑X
      hx₂ : Eq (i₂ x₂) ((CategoryTheory.Limits.pullback.snd g₁ g₂) x✝)
      this : Eq (f₁ x₁) (f₂ x₂)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (TopCat.pullbackIsoProdSubtype f₁ f₂ …
    -/
  · simp only [Category.assoc, limit.lift_π, PullbackCone.mk_π_app_one]
    /-
      case h.a.none
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
      x₁ : ↑W
      hx₁ : Eq (i₁ x₁) ((CategoryTheory.Limits.pullback.fst g₁ g₂) x✝)
      x₂ : ↑X
      hx₂ : Eq (i₂ x₂) ((CategoryTheory.Limits.pullback.snd g₁ g₂) x✝)
      this : Eq (f₁ x₁) (f₂ x₂)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (TopCat.pullbackIsoProdSubtype f₁ f₂ …
    -/
    simp only [cospan_one, pullbackIsoProdSubtype_inv_fst_assoc, comp_apply]
    rw [pullbackFst_apply, hx₁, ← limit.w _ WalkingCospan.Hom.inl, cospan_map_inl,
        comp_apply (g := g₁)]
  · simp only [cospan_left, limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app,
      pullbackIsoProdSubtype_inv_fst_assoc, comp_apply]
    /-
      case h.a.some.left
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
      x₁ : ↑W
      hx₁ : Eq (i₁ x₁) ((CategoryTheory.Limits.pullback.fst g₁ g₂) x✝)
      x₂ : ↑X
      hx₂ : Eq (i₂ x₂) ((CategoryTheory.Limits.pullback.snd g₁ g₂) x✝)
      this : Eq (f₁ x₁) (f₂ x₂)
      ⊢ Eq (i₁ ((TopCat.pullbackFst f₁ f₂) ⟨{ fst := x₁, snd := x₂ }, this⟩)) ((Cate …
    -/
    erw [hx₁] -- now `erw` after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      🎉 no goals
    -/
  · simp only [cospan_right, limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app,
      pullbackIsoProdSubtype_inv_snd_assoc, comp_apply]
    /-
      case h.a.some.right
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      x✝ : ↑(CategoryTheory.Limits.pullback g₁ g₂)
      x₁ : ↑W
      hx₁ : Eq (i₁ x₁) ((CategoryTheory.Limits.pullback.fst g₁ g₂) x✝)
      x₂ : ↑X
      hx₂ : Eq (i₂ x₂) ((CategoryTheory.Limits.pullback.snd g₁ g₂) x✝)
      this : Eq (f₁ x₁) (f₂ x₂)
      ⊢ Eq (i₂ ((TopCat.pullbackSnd f₁ f₂) ⟨{ fst := x₁, snd := x₂ }, this⟩)) ((Cate …
    -/
    erw [hx₂] -- now `erw` after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      🎉 no goals
    -/


theorem pullback_fst_range {X Y S : TopCat} (f : X ⟶ S) (g : Y ⟶ S) :
    Set.range (pullback.fst f g) = { x : X | ∃ y : Y, f x = g y } := by
  /-
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    ⊢ Eq (Set.range ⇑(CategoryTheory.Limits.pullback.fst f g)) (setOf fun x => Exi …
  -/
  ext x
  /-
    case h
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    x : ↑X
    ⊢ Iff (Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst f g)) x) …
  -/
  constructor
    /-
      case h.mp
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : ↑X
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst f g)) x → Mem …
    -/
  · rintro ⟨(y : (forget TopCat).obj _), rfl⟩
    /-
      case h.mp.intro
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      y : (CategoryTheory.forget TopCat).obj (CategoryTheory.Limits.pullback f g)
      ⊢ Membership.mem (setOf fun x => Exists fun y => Eq (f x) (g y)) ((CategoryThe …
    -/
    use (pullback.snd f g) y
    /-
      case h
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      y : (CategoryTheory.forget TopCat).obj (CategoryTheory.Limits.pullback f g)
      ⊢ Eq (f ((CategoryTheory.Limits.pullback.fst f g) y)) (g ((CategoryTheory.Limi …
    -/
    exact ConcreteCategory.congr_hom pullback.condition y
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : ↑X
      ⊢ Membership.mem (setOf fun x => Exists fun y => Eq (f x) (g y)) x → Membershi …
    -/
  · rintro ⟨y, eq⟩
    /-
      case h.mpr.intro
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : ↑X
      y : ↑Y
      eq : Eq (f x) (g y)
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst f g)) x
    -/
    use (TopCat.pullbackIsoProdSubtype f g).inv ⟨⟨x, y⟩, eq⟩
    /-
      case h
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : ↑X
      y : ↑Y
      eq : Eq (f x) (g y)
      ⊢ Eq ((CategoryTheory.Limits.pullback.fst f g) ((TopCat.pullbackIsoProdSubtype …
    -/
    rw [pullbackIsoProdSubtype_inv_fst_apply]
    /-
      🎉 no goals
    -/


theorem pullback_snd_range {X Y S : TopCat} (f : X ⟶ S) (g : Y ⟶ S) :
    Set.range (pullback.snd f g) = { y : Y | ∃ x : X, f x = g y } := by
  /-
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    ⊢ Eq (Set.range ⇑(CategoryTheory.Limits.pullback.snd f g)) (setOf fun y => Exi …
  -/
  ext y
  /-
    case h
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    y : ↑Y
    ⊢ Iff (Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.snd f g)) y) …
  -/
  constructor
    /-
      case h.mp
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      y : ↑Y
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.snd f g)) y → Mem …
    -/
  · rintro ⟨(x : (forget TopCat).obj _), rfl⟩
    /-
      case h.mp.intro
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : (CategoryTheory.forget TopCat).obj (CategoryTheory.Limits.pullback f g)
      ⊢ Membership.mem (setOf fun y => Exists fun x => Eq (f x) (g y)) ((CategoryThe …
    -/
    use (pullback.fst f g) x
    /-
      case h
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x : (CategoryTheory.forget TopCat).obj (CategoryTheory.Limits.pullback f g)
      ⊢ Eq (f ((CategoryTheory.Limits.pullback.fst f g) x)) (g ((CategoryTheory.Limi …
    -/
    exact ConcreteCategory.congr_hom pullback.condition x
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      y : ↑Y
      ⊢ Membership.mem (setOf fun y => Exists fun x => Eq (f x) (g y)) y → Membershi …
    -/
  · rintro ⟨x, eq⟩
    /-
      case h.mpr.intro
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      y : ↑Y
      x : ↑X
      eq : Eq (f x) (g y)
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.snd f g)) y
    -/
    use (TopCat.pullbackIsoProdSubtype f g).inv ⟨⟨x, y⟩, eq⟩
    /-
      case h
      X Y S : TopCat
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      y : ↑Y
      x : ↑X
      eq : Eq (f x) (g y)
      ⊢ Eq ((CategoryTheory.Limits.pullback.snd f g) ((TopCat.pullbackIsoProdSubtype …
    -/
    rw [pullbackIsoProdSubtype_inv_snd_apply]
    /-
      🎉 no goals
    -/


/-- If there is a diagram where the morphisms `W ⟶ Y` and `X ⟶ Z` are embeddings,
then the induced morphism `W ×ₛ X ⟶ Y ×ₜ Z` is also an embedding.

```
W ⟶ Y
 ↘   ↘
  S ⟶ T
 ↗   ↗
X ⟶ Z
```
-/
theorem pullback_map_isEmbedding {W X Y Z S T : TopCat.{u}} (f₁ : W ⟶ S) (f₂ : X ⟶ S)
    (g₁ : Y ⟶ T) (g₂ : Z ⟶ T) {i₁ : W ⟶ Y} {i₂ : X ⟶ Z} (H₁ : IsEmbedding i₁)
    (H₂ : IsEmbedding i₂) (i₃ : S ⟶ T) (eq₁ : f₁ ≫ i₃ = i₁ ≫ g₁) (eq₂ : f₂ ≫ i₃ = i₂ ≫ g₂) :
    IsEmbedding (pullback.map f₁ f₂ g₁ g₂ i₁ i₂ i₃ eq₁ eq₂) := by
  refine .of_comp (ContinuousMap.continuous_toFun _)
    (show Continuous (prod.lift (pullback.fst g₁ g₂) (pullback.snd g₁ g₂)) from
        ContinuousMap.continuous_toFun _)
      ?_
  suffices
    IsEmbedding (prod.lift (pullback.fst f₁ f₂) (pullback.snd f₁ f₂) ≫ Limits.prod.map i₁ i₂) by
    simpa [← coe_comp] using this
  /-
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    H₁ : Topology.IsEmbedding ⇑i₁
    H₂ : Topology.IsEmbedding ⇑i₂
    i₃ : Quiver.Hom S T
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    ⊢ Topology.IsEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheory.Li …
  -/
  rw [coe_comp]
  /-
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    H₁ : Topology.IsEmbedding ⇑i₁
    H₂ : Topology.IsEmbedding ⇑i₂
    i₃ : Quiver.Hom S T
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    ⊢ Topology.IsEmbedding (Function.comp ⇑(CategoryTheory.Limits.prod.map i₁ i₂)  …
  -/
  exact (isEmbedding_prodMap H₁ H₂).comp (isEmbedding_pullback_to_prod _ _)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-26")]
alias pullback_map_embedding_of_embeddings := pullback_map_isEmbedding


/-- If there is a diagram where the morphisms `W ⟶ Y` and `X ⟶ Z` are open embeddings, and `S ⟶ T`
is mono, then the induced morphism `W ×ₛ X ⟶ Y ×ₜ Z` is also an open embedding.

```
W ⟶ Y
 ↘   ↘
  S ⟶ T
 ↗   ↗
X ⟶ Z
```
-/
theorem pullback_map_isOpenEmbedding {W X Y Z S T : TopCat.{u}} (f₁ : W ⟶ S)
    (f₂ : X ⟶ S) (g₁ : Y ⟶ T) (g₂ : Z ⟶ T) {i₁ : W ⟶ Y} {i₂ : X ⟶ Z} (H₁ : IsOpenEmbedding i₁)
    (H₂ : IsOpenEmbedding i₂) (i₃ : S ⟶ T) [H₃ : Mono i₃] (eq₁ : f₁ ≫ i₃ = i₁ ≫ g₁)
    (eq₂ : f₂ ≫ i₃ = i₂ ≫ g₂) : IsOpenEmbedding (pullback.map f₁ f₂ g₁ g₂ i₁ i₂ i₃ eq₁ eq₂) := by
  /-
    W X Y Z S T : TopCat
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    H₁ : Topology.IsOpenEmbedding ⇑i₁
    H₂ : Topology.IsOpenEmbedding ⇑i₂
    i₃ : Quiver.Hom S T
    H₃ : CategoryTheory.Mono i₃
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.Limits.pullback.map f₁ f₂ g₁ g₂ i₁ …
  -/
  constructor
  · apply
      pullback_map_isEmbedding f₁ f₂ g₁ g₂ H₁.isEmbedding H₂.isEmbedding i₃ eq₁ eq₂
    /-
      case isOpen_range
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      H₁ : Topology.IsOpenEmbedding ⇑i₁
      H₂ : Topology.IsOpenEmbedding ⇑i₂
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      ⊢ IsOpen (Set.range ⇑(CategoryTheory.Limits.pullback.map f₁ f₂ g₁ g₂ i₁ i₂ i₃  …
    -/
  · rw [range_pullback_map]
    /-
      case isOpen_range
      W X Y Z S T : TopCat
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      H₁ : Topology.IsOpenEmbedding ⇑i₁
      H₂ : Topology.IsOpenEmbedding ⇑i₂
      i₃ : Quiver.Hom S T
      H₃ : CategoryTheory.Mono i₃
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      ⊢ IsOpen (Inter.inter (Set.preimage (⇑(CategoryTheory.Limits.pullback.fst g₁ g …
    -/
    apply IsOpen.inter <;> apply Continuous.isOpen_preimage
      /-
        case isOpen_range.hs.self
        W X Y Z S T : TopCat
        f₁ : Quiver.Hom W S
        f₂ : Quiver.Hom X S
        g₁ : Quiver.Hom Y T
        g₂ : Quiver.Hom Z T
        i₁ : Quiver.Hom W Y
        i₂ : Quiver.Hom X Z
        H₁ : Topology.IsOpenEmbedding ⇑i₁
        H₂ : Topology.IsOpenEmbedding ⇑i₂
        i₃ : Quiver.Hom S T
        H₃ : CategoryTheory.Mono i₃
        eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
        eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
        ⊢ Continuous ⇑(CategoryTheory.Limits.pullback.fst g₁ g₂)
      -/
    · apply ContinuousMap.continuous_toFun
      /-
        🎉 no goals
      -/
      /-
        case isOpen_range.hs.a
        W X Y Z S T : TopCat
        f₁ : Quiver.Hom W S
        f₂ : Quiver.Hom X S
        g₁ : Quiver.Hom Y T
        g₂ : Quiver.Hom Z T
        i₁ : Quiver.Hom W Y
        i₂ : Quiver.Hom X Z
        H₁ : Topology.IsOpenEmbedding ⇑i₁
        H₂ : Topology.IsOpenEmbedding ⇑i₂
        i₃ : Quiver.Hom S T
        H₃ : CategoryTheory.Mono i₃
        eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
        eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
        ⊢ IsOpen (Set.range ⇑i₁)
      -/
    · exact H₁.isOpen_range
      /-
        🎉 no goals
      -/
      /-
        case isOpen_range.ht.self
        W X Y Z S T : TopCat
        f₁ : Quiver.Hom W S
        f₂ : Quiver.Hom X S
        g₁ : Quiver.Hom Y T
        g₂ : Quiver.Hom Z T
        i₁ : Quiver.Hom W Y
        i₂ : Quiver.Hom X Z
        H₁ : Topology.IsOpenEmbedding ⇑i₁
        H₂ : Topology.IsOpenEmbedding ⇑i₂
        i₃ : Quiver.Hom S T
        H₃ : CategoryTheory.Mono i₃
        eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
        eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
        ⊢ Continuous ⇑(CategoryTheory.Limits.pullback.snd g₁ g₂)
      -/
    · apply ContinuousMap.continuous_toFun
      /-
        🎉 no goals
      -/
      /-
        case isOpen_range.ht.a
        W X Y Z S T : TopCat
        f₁ : Quiver.Hom W S
        f₂ : Quiver.Hom X S
        g₁ : Quiver.Hom Y T
        g₂ : Quiver.Hom Z T
        i₁ : Quiver.Hom W Y
        i₂ : Quiver.Hom X Z
        H₁ : Topology.IsOpenEmbedding ⇑i₁
        H₂ : Topology.IsOpenEmbedding ⇑i₂
        i₃ : Quiver.Hom S T
        H₃ : CategoryTheory.Mono i₃
        eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
        eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
        ⊢ IsOpen (Set.range ⇑i₂)
      -/
    · exact H₂.isOpen_range
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-18")]
alias pullback_map_openEmbedding_of_open_embeddings := pullback_map_isOpenEmbedding



lemma snd_isEmbedding_of_left {X Y S : TopCat} {f : X ⟶ S} (H : IsEmbedding f) (g : Y ⟶ S) :
    IsEmbedding <| ⇑(pullback.snd f g) := by
  convert (homeoOfIso (asIso (pullback.snd (𝟙 S) g))).isEmbedding.comp
      (pullback_map_isEmbedding (i₂ := 𝟙 Y)
        f g (𝟙 S) g H (homeoOfIso (Iso.refl _)).isEmbedding (𝟙 _) rfl (by simp))
  /-
    case h.e'_5
    X Y S : TopCat
    f : Quiver.Hom X S
    H : Topology.IsEmbedding ⇑f
    g : Quiver.Hom Y S
    ⊢ Eq (⇑(CategoryTheory.Limits.pullback.snd f g)) (Function.comp ⇑(TopCat.homeo …
  -/
  erw [← coe_comp]
  /-
    case h.e'_5
    X Y S : TopCat
    f : Quiver.Hom X S
    H : Topology.IsEmbedding ⇑f
    g : Quiver.Hom Y S
    ⊢ Eq ⇑(CategoryTheory.Limits.pullback.snd f g) ⇑(CategoryTheory.CategoryStruct …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-26")]
alias snd_embedding_of_left_embedding := snd_isEmbedding_of_left


theorem fst_isEmbedding_of_right {X Y S : TopCat} (f : X ⟶ S) {g : Y ⟶ S}
    (H : IsEmbedding g) : IsEmbedding <| ⇑(pullback.fst f g) := by
  convert (homeoOfIso (asIso (pullback.fst f (𝟙 S)))).isEmbedding.comp
      (pullback_map_isEmbedding (i₁ := 𝟙 X)
        f g f (𝟙 _) (homeoOfIso (Iso.refl _)).isEmbedding H (𝟙 _) rfl (by simp))
  /-
    case h.e'_5
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H : Topology.IsEmbedding ⇑g
    ⊢ Eq (⇑(CategoryTheory.Limits.pullback.fst f g)) (Function.comp ⇑(TopCat.homeo …
  -/
  erw [← coe_comp]
  /-
    case h.e'_5
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H : Topology.IsEmbedding ⇑g
    ⊢ Eq ⇑(CategoryTheory.Limits.pullback.fst f g) ⇑(CategoryTheory.CategoryStruct …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-26")]
alias fst_embedding_of_right_embedding := fst_isEmbedding_of_right


theorem isEmbedding_of_pullback {X Y S : TopCat} {f : X ⟶ S} {g : Y ⟶ S} (H₁ : IsEmbedding f)
    (H₂ : IsEmbedding g) : IsEmbedding (limit.π (cospan f g) WalkingCospan.one) := by
  /-
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H₁ : Topology.IsEmbedding ⇑f
    H₂ : Topology.IsEmbedding ⇑g
    ⊢ Topology.IsEmbedding ⇑(CategoryTheory.Limits.limit.π (CategoryTheory.Limits. …
  -/
  convert H₂.comp (snd_isEmbedding_of_left H₁ g)
  /-
    case h.e'_5.h
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H₁ : Topology.IsEmbedding ⇑f
    H₂ : Topology.IsEmbedding ⇑g
    e_2✝ : Eq ↑((CategoryTheory.Limits.cospan f g).obj CategoryTheory.Limits.Walki …
    ⊢ Eq (⇑(CategoryTheory.Limits.limit.π (CategoryTheory.Limits.cospan f g) Categ …
  -/
  rw [← coe_comp, ← limit.w _ WalkingCospan.Hom.inr]
  /-
    case h.e'_5.h
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H₁ : Topology.IsEmbedding ⇑f
    H₂ : Topology.IsEmbedding ⇑g
    e_2✝ : Eq ↑((CategoryTheory.Limits.cospan f g).obj CategoryTheory.Limits.Walki …
    ⊢ Eq ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Cate …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-26")]
alias embedding_of_pullback_embeddings := isEmbedding_of_pullback


theorem snd_isOpenEmbedding_of_left {X Y S : TopCat} {f : X ⟶ S} (H : IsOpenEmbedding f)
    (g : Y ⟶ S) : IsOpenEmbedding <| ⇑(pullback.snd f g) := by
  convert (homeoOfIso (asIso (pullback.snd (𝟙 S) g))).isOpenEmbedding.comp
      (pullback_map_isOpenEmbedding (i₂ := 𝟙 Y) f g (𝟙 _) g H
        (homeoOfIso (Iso.refl _)).isOpenEmbedding (𝟙 _) rfl (by simp))
  /-
    case h.e'_5
    X Y S : TopCat
    f : Quiver.Hom X S
    H : Topology.IsOpenEmbedding ⇑f
    g : Quiver.Hom Y S
    ⊢ Eq (⇑(CategoryTheory.Limits.pullback.snd f g)) (Function.comp ⇑(TopCat.homeo …
  -/
  erw [← coe_comp]
  /-
    case h.e'_5
    X Y S : TopCat
    f : Quiver.Hom X S
    H : Topology.IsOpenEmbedding ⇑f
    g : Quiver.Hom Y S
    ⊢ Eq ⇑(CategoryTheory.Limits.pullback.snd f g) ⇑(CategoryTheory.CategoryStruct …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias snd_openEmbedding_of_left_openEmbedding := snd_isOpenEmbedding_of_left


theorem fst_isOpenEmbedding_of_right {X Y S : TopCat} (f : X ⟶ S) {g : Y ⟶ S}
    (H : IsOpenEmbedding g) : IsOpenEmbedding <| ⇑(pullback.fst f g) := by
  convert (homeoOfIso (asIso (pullback.fst f (𝟙 S)))).isOpenEmbedding.comp
      (pullback_map_isOpenEmbedding (i₁ := 𝟙 X) f g f (𝟙 _)
        (homeoOfIso (Iso.refl _)).isOpenEmbedding H (𝟙 _) rfl (by simp))
  /-
    case h.e'_5
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H : Topology.IsOpenEmbedding ⇑g
    ⊢ Eq (⇑(CategoryTheory.Limits.pullback.fst f g)) (Function.comp ⇑(TopCat.homeo …
  -/
  erw [← coe_comp]
  /-
    case h.e'_5
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H : Topology.IsOpenEmbedding ⇑g
    ⊢ Eq ⇑(CategoryTheory.Limits.pullback.fst f g) ⇑(CategoryTheory.CategoryStruct …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias fst_openEmbedding_of_right_openEmbedding := fst_isOpenEmbedding_of_right


/-- If `X ⟶ S`, `Y ⟶ S` are open embeddings, then so is `X ×ₛ Y ⟶ S`. -/
theorem isOpenEmbedding_of_pullback {X Y S : TopCat} {f : X ⟶ S} {g : Y ⟶ S}
    (H₁ : IsOpenEmbedding f) (H₂ : IsOpenEmbedding g) :
    IsOpenEmbedding (limit.π (cospan f g) WalkingCospan.one) := by
  /-
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H₁ : Topology.IsOpenEmbedding ⇑f
    H₂ : Topology.IsOpenEmbedding ⇑g
    ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.Limits.limit.π (CategoryTheory.Lim …
  -/
  convert H₂.comp (snd_isOpenEmbedding_of_left H₁ g)
  /-
    case h.e'_5.h
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H₁ : Topology.IsOpenEmbedding ⇑f
    H₂ : Topology.IsOpenEmbedding ⇑g
    e_2✝ : Eq ↑((CategoryTheory.Limits.cospan f g).obj CategoryTheory.Limits.Walki …
    ⊢ Eq (⇑(CategoryTheory.Limits.limit.π (CategoryTheory.Limits.cospan f g) Categ …
  -/
  rw [← coe_comp, ← limit.w _ WalkingCospan.Hom.inr]
  /-
    case h.e'_5.h
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H₁ : Topology.IsOpenEmbedding ⇑f
    H₂ : Topology.IsOpenEmbedding ⇑g
    e_2✝ : Eq ↑((CategoryTheory.Limits.cospan f g).obj CategoryTheory.Limits.Walki …
    ⊢ Eq ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Cate …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-30")]
alias isOpenEmbedding_of_pullback_open_embeddings := isOpenEmbedding_of_pullback


@[deprecated (since := "2024-10-18")]
alias openEmbedding_of_pullback_open_embeddings := isOpenEmbedding_of_pullback


theorem fst_iso_of_right_embedding_range_subset {X Y S : TopCat} (f : X ⟶ S) {g : Y ⟶ S}
    (hg : IsEmbedding g) (H : Set.range f ⊆ Set.range g) :
    IsIso (pullback.fst f g) := by
  let esto : (pullback f g : TopCat) ≃ₜ X :=
    (Homeomorph.ofIsEmbedding _ (fst_isEmbedding_of_right f hg)).trans
      { toFun := Subtype.val
        invFun := fun x =>
          ⟨x, by
            rw [pullback_fst_range]
            exact ⟨_, (H (Set.mem_range_self x)).choose_spec.symm⟩⟩
        left_inv := fun ⟨_, _⟩ => rfl
        right_inv := fun x => rfl }
  /-
    X Y S : TopCat
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    hg : Topology.IsEmbedding ⇑g
    H : HasSubset.Subset (Set.range ⇑f) (Set.range ⇑g)
    esto : Homeomorph ↑(CategoryTheory.Limits.pullback f g) ↑X := (Homeomorph.ofIs …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.fst f g)
  -/
  convert (isoOfHomeo esto).isIso_hom
  /-
    🎉 no goals
  -/


theorem snd_iso_of_left_embedding_range_subset {X Y S : TopCat} {f : X ⟶ S} (hf : IsEmbedding f)
    (g : Y ⟶ S) (H : Set.range g ⊆ Set.range f) : IsIso (pullback.snd f g) := by
  let esto : (pullback f g : TopCat) ≃ₜ Y :=
    (Homeomorph.ofIsEmbedding _ (snd_isEmbedding_of_left hf g)).trans
      { toFun := Subtype.val
        invFun := fun x =>
          ⟨x, by
            rw [pullback_snd_range]
            exact ⟨_, (H (Set.mem_range_self x)).choose_spec⟩⟩
        left_inv := fun ⟨_, _⟩ => rfl
        right_inv := fun x => rfl }
  /-
    X Y S : TopCat
    f : Quiver.Hom X S
    hf : Topology.IsEmbedding ⇑f
    g : Quiver.Hom Y S
    H : HasSubset.Subset (Set.range ⇑g) (Set.range ⇑f)
    esto : Homeomorph ↑(CategoryTheory.Limits.pullback f g) ↑Y := (Homeomorph.ofIs …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
  -/
  convert (isoOfHomeo esto).isIso_hom
  /-
    🎉 no goals
  -/


theorem pullback_snd_image_fst_preimage (f : X ⟶ Z) (g : Y ⟶ Z) (U : Set X) :
    (pullback.snd f g) '' ((pullback.fst f g) ⁻¹' U) =
      g ⁻¹' (f '' U) := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    U : Set ↑X
    ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.snd f g)) (Set.preimage (⇑(C …
  -/
  ext x
  /-
    case h
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    U : Set ↑X
    x : ↑Y
    ⊢ Iff (Membership.mem (Set.image (⇑(CategoryTheory.Limits.pullback.snd f g)) ( …
  -/
  constructor
    /-
      case h.mp
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      U : Set ↑X
      x : ↑Y
      ⊢ Membership.mem (Set.image (⇑(CategoryTheory.Limits.pullback.snd f g)) (Set.p …
    -/
  · rintro ⟨(y : (forget TopCat).obj _), hy, rfl⟩
    exact
      ⟨(pullback.fst f g) y, hy, ConcreteCategory.congr_hom pullback.condition y⟩
    /-
      case h.mpr
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      U : Set ↑X
      x : ↑Y
      ⊢ Membership.mem (Set.preimage (⇑g) (Set.image (⇑f) U)) x → Membership.mem (Se …
    -/
  · rintro ⟨y, hy, eq⟩
  -- next 5 lines were
  -- `exact ⟨(TopCat.pullbackIsoProdSubtype f g).inv ⟨⟨_, _⟩, eq⟩, by simpa, by simp⟩` before https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      case h.mpr.intro.intro
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      U : Set ↑X
      x : ↑Y
      y : ↑X
      hy : Membership.mem U y
      eq : Eq (f y) (g x)
      ⊢ Membership.mem (Set.image (⇑(CategoryTheory.Limits.pullback.snd f g)) (Set.p …
    -/
    refine ⟨(TopCat.pullbackIsoProdSubtype f g).inv ⟨⟨_, _⟩, eq⟩, ?_, ?_⟩
      /-
        case h.mpr.intro.intro.refine_1
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        U : Set ↑X
        x : ↑Y
        y : ↑X
        hy : Membership.mem U y
        eq : Eq (f y) (g x)
        ⊢ Membership.mem (Set.preimage (⇑(CategoryTheory.Limits.pullback.fst f g)) U)  …
      -/
    · simp only [coe_of, Set.mem_preimage]
      /-
        case h.mpr.intro.intro.refine_1
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        U : Set ↑X
        x : ↑Y
        y : ↑X
        hy : Membership.mem U y
        eq : Eq (f y) (g x)
        ⊢ Membership.mem U ((CategoryTheory.Limits.pullback.fst f g) ((TopCat.pullback …
      -/
      convert hy
      /-
        case h.e'_5
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        U : Set ↑X
        x : ↑Y
        y : ↑X
        hy : Membership.mem U y
        eq : Eq (f y) (g x)
        ⊢ Eq ((CategoryTheory.Limits.pullback.fst f g) ((TopCat.pullbackIsoProdSubtype …
      -/
      erw [pullbackIsoProdSubtype_inv_fst_apply]
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.intro.intro.refine_2
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        U : Set ↑X
        x : ↑Y
        y : ↑X
        hy : Membership.mem U y
        eq : Eq (f y) (g x)
        ⊢ Eq ((CategoryTheory.Limits.pullback.snd f g) ((TopCat.pullbackIsoProdSubtype …
      -/
    · rw [pullbackIsoProdSubtype_inv_snd_apply]
      /-
        🎉 no goals
      -/


theorem pullback_fst_image_snd_preimage (f : X ⟶ Z) (g : Y ⟶ Z) (U : Set Y) :
    (pullback.fst f g) '' ((pullback.snd f g) ⁻¹' U) =
      f ⁻¹' (g '' U) := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    U : Set ↑Y
    ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.fst f g)) (Set.preimage (⇑(C …
  -/
  ext x
  /-
    case h
    X Y Z : TopCat
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    U : Set ↑Y
    x : ↑X
    ⊢ Iff (Membership.mem (Set.image (⇑(CategoryTheory.Limits.pullback.fst f g)) ( …
  -/
  constructor
    /-
      case h.mp
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      U : Set ↑Y
      x : ↑X
      ⊢ Membership.mem (Set.image (⇑(CategoryTheory.Limits.pullback.fst f g)) (Set.p …
    -/
  · rintro ⟨(y : (forget TopCat).obj _), hy, rfl⟩
    exact
      ⟨(pullback.snd f g) y, hy,
        (ConcreteCategory.congr_hom pullback.condition y).symm⟩
    /-
      case h.mpr
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      U : Set ↑Y
      x : ↑X
      ⊢ Membership.mem (Set.preimage (⇑f) (Set.image (⇑g) U)) x → Membership.mem (Se …
    -/
  · rintro ⟨y, hy, eq⟩
    -- next 5 lines were
    -- `exact ⟨(TopCat.pullbackIsoProdSubtype f g).inv ⟨⟨_, _⟩, eq.symm⟩, by simpa, by simp⟩`
    -- before https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      case h.mpr.intro.intro
      X Y Z : TopCat
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      U : Set ↑Y
      x : ↑X
      y : ↑Y
      hy : Membership.mem U y
      eq : Eq (g y) (f x)
      ⊢ Membership.mem (Set.image (⇑(CategoryTheory.Limits.pullback.fst f g)) (Set.p …
    -/
    refine ⟨(TopCat.pullbackIsoProdSubtype f g).inv ⟨⟨_, _⟩, eq.symm⟩, ?_, ?_⟩
      /-
        case h.mpr.intro.intro.refine_1
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        U : Set ↑Y
        x : ↑X
        y : ↑Y
        hy : Membership.mem U y
        eq : Eq (g y) (f x)
        ⊢ Membership.mem (Set.preimage (⇑(CategoryTheory.Limits.pullback.snd f g)) U)  …
      -/
    · simp only [coe_of, Set.mem_preimage]
      /-
        case h.mpr.intro.intro.refine_1
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        U : Set ↑Y
        x : ↑X
        y : ↑Y
        hy : Membership.mem U y
        eq : Eq (g y) (f x)
        ⊢ Membership.mem U ((CategoryTheory.Limits.pullback.snd f g) ((TopCat.pullback …
      -/
      convert hy
      /-
        case h.e'_5
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        U : Set ↑Y
        x : ↑X
        y : ↑Y
        hy : Membership.mem U y
        eq : Eq (g y) (f x)
        ⊢ Eq ((CategoryTheory.Limits.pullback.snd f g) ((TopCat.pullbackIsoProdSubtype …
      -/
      erw [pullbackIsoProdSubtype_inv_snd_apply]
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.intro.intro.refine_2
        X Y Z : TopCat
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        U : Set ↑Y
        x : ↑X
        y : ↑Y
        hy : Membership.mem U y
        eq : Eq (g y) (f x)
        ⊢ Eq ((CategoryTheory.Limits.pullback.fst f g) ((TopCat.pullbackIsoProdSubtype …
      -/
    · rw [pullbackIsoProdSubtype_inv_fst_apply]
      /-
        🎉 no goals
      -/


theorem coinduced_of_isColimit {F : J ⥤ TopCat.{max v u}} (c : Cocone F) (hc : IsColimit c) :
    c.pt.str = ⨆ j, (F.obj j).str.coinduced (c.ι.app j) := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ Eq c.pt.str (iSup fun j => TopologicalSpace.coinduced (⇑(c.ι.app j)) (F.obj  …
  -/
  let homeo := homeoOfIso (hc.coconePointUniqueUpToIso (colimitCoconeIsColimit F))
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    homeo : Homeomorph ↑c.pt ↑(TopCat.colimitCocone F).pt := TopCat.homeoOfIso (hc …
    ⊢ Eq c.pt.str (iSup fun j => TopologicalSpace.coinduced (⇑(c.ι.app j)) (F.obj  …
  -/
  ext
  /-
    case a.h.a
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    homeo : Homeomorph ↑c.pt ↑(TopCat.colimitCocone F).pt := TopCat.homeoOfIso (hc …
    x✝ : Set ↑c.pt
    ⊢ Iff (IsOpen x✝) (IsOpen x✝)
  -/
  refine homeo.symm.isOpen_preimage.symm.trans (Iff.trans ?_ isOpen_iSup_iff.symm)
  /-
    case a.h.a
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    homeo : Homeomorph ↑c.pt ↑(TopCat.colimitCocone F).pt := TopCat.homeoOfIso (hc …
    x✝ : Set ↑c.pt
    ⊢ Iff (IsOpen (Set.preimage (⇑homeo.symm) x✝)) (∀ (i : J), IsOpen x✝)
  -/
  exact isOpen_iSup_iff
  /-
    🎉 no goals
  -/


theorem colimit_topology (F : J ⥤ TopCat.{max v u}) :
    (colimit F).str = ⨆ j, (F.obj j).str.coinduced (colimit.ι F j) :=
  coinduced_of_isColimit _ (colimit.isColimit F)


theorem colimit_isOpen_iff (F : J ⥤ TopCat.{max v u}) (U : Set ((colimit F : _) : Type max v u)) :
    IsOpen U ↔ ∀ j, IsOpen (colimit.ι F j ⁻¹' U) := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    U : Set ↑(CategoryTheory.Limits.colimit F)
    ⊢ Iff (IsOpen U) (∀ (j : J), IsOpen (Set.preimage (⇑(CategoryTheory.Limits.col …
  -/
  dsimp [topologicalSpace_coe]
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    U : Set ↑(CategoryTheory.Limits.colimit F)
    ⊢ Iff (IsOpen U) (∀ (j : J), IsOpen (Set.preimage (⇑(CategoryTheory.Limits.col …
  -/
  conv_lhs => rw [colimit_topology F]
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    U : Set ↑(CategoryTheory.Limits.colimit F)
    ⊢ Iff (IsOpen U) (∀ (j : J), IsOpen (Set.preimage (⇑(CategoryTheory.Limits.col …
  -/
  exact isOpen_iSup_iff
  /-
    🎉 no goals
  -/


theorem coequalizer_isOpen_iff (F : WalkingParallelPair ⥤ TopCat.{u})
    (U : Set ((colimit F : _) : Type u)) :
    IsOpen U ↔ IsOpen (colimit.ι F WalkingParallelPair.one ⁻¹' U) := by
  /-
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat
    U : Set ↑(CategoryTheory.Limits.colimit F)
    ⊢ Iff (IsOpen U) (IsOpen (Set.preimage (⇑(CategoryTheory.Limits.colimit.ι F Ca …
  -/
  rw [colimit_isOpen_iff]
  /-
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat
    U : Set ↑(CategoryTheory.Limits.colimit F)
    ⊢ Iff (∀ (j : CategoryTheory.Limits.WalkingParallelPair), IsOpen (Set.preimage …
  -/
  constructor
    /-
      case mp
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat
      U : Set ↑(CategoryTheory.Limits.colimit F)
      ⊢ (∀ (j : CategoryTheory.Limits.WalkingParallelPair), IsOpen (Set.preimage (⇑( …
    -/
  · intro H
    /-
      case mp
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat
      U : Set ↑(CategoryTheory.Limits.colimit F)
      H : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), IsOpen (Set.preimage (⇑ …
      ⊢ IsOpen (Set.preimage (⇑(CategoryTheory.Limits.colimit.ι F CategoryTheory.Lim …
    -/
    exact H _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat
      U : Set ↑(CategoryTheory.Limits.colimit F)
      ⊢ IsOpen (Set.preimage (⇑(CategoryTheory.Limits.colimit.ι F CategoryTheory.Lim …
    -/
  · intro H j
    /-
      case mpr
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat
      U : Set ↑(CategoryTheory.Limits.colimit F)
      H : IsOpen (Set.preimage (⇑(CategoryTheory.Limits.colimit.ι F CategoryTheory.L …
      j : CategoryTheory.Limits.WalkingParallelPair
      ⊢ IsOpen (Set.preimage (⇑(CategoryTheory.Limits.colimit.ι F j)) U)
    -/
    cases j
      /-
        case mpr.zero
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat
        U : Set ↑(CategoryTheory.Limits.colimit F)
        H : IsOpen (Set.preimage (⇑(CategoryTheory.Limits.colimit.ι F CategoryTheory.L …
        ⊢ IsOpen (Set.preimage (⇑(CategoryTheory.Limits.colimit.ι F CategoryTheory.Lim …
      -/
    · rw [← colimit.w F WalkingParallelPairHom.left]
      /-
        case mpr.zero
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat
        U : Set ↑(CategoryTheory.Limits.colimit F)
        H : IsOpen (Set.preimage (⇑(CategoryTheory.Limits.colimit.ι F CategoryTheory.L …
        ⊢ IsOpen (Set.preimage (⇑(CategoryTheory.CategoryStruct.comp (F.map CategoryTh …
      -/
      exact (F.map WalkingParallelPairHom.left).continuous_toFun.isOpen_preimage _ H
      /-
        🎉 no goals
      -/
      /-
        case mpr.one
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat
        U : Set ↑(CategoryTheory.Limits.colimit F)
        H : IsOpen (Set.preimage (⇑(CategoryTheory.Limits.colimit.ι F CategoryTheory.L …
        ⊢ IsOpen (Set.preimage (⇑(CategoryTheory.Limits.colimit.ι F CategoryTheory.Lim …
      -/
    · exact H
      /-
        🎉 no goals
      -/


