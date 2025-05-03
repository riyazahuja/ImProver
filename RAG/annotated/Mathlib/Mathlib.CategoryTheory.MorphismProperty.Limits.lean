/-- A morphism property is `IsStableUnderBaseChange` if the base change of such a morphism
still falls in the class. -/
class IsStableUnderBaseChange (P : MorphismProperty C) : Prop where
  of_isPullback {X Y Y' S : C} {f : X ⟶ S} {g : Y ⟶ S} {f' : Y' ⟶ Y} {g' : Y' ⟶ X}
    (sq : IsPullback f' g' g f) (hg : P g) : P g'


/-- A morphism property is `IsStableUnderCobaseChange` if the cobase change of such a morphism
still falls in the class. -/
class IsStableUnderCobaseChange (P : MorphismProperty C) : Prop where
  of_isPushout {A A' B B' : C} {f : A ⟶ A'} {g : A ⟶ B} {f' : B ⟶ B'} {g' : A' ⟶ B'}
    (sq : IsPushout g f f' g') (hf : P f) : P f'


lemma of_isPullback {P : MorphismProperty C} [P.IsStableUnderBaseChange]
    {X Y Y' S : C} {f : X ⟶ S} {g : Y ⟶ S} {f' : Y' ⟶ Y} {g' : Y' ⟶ X}
    (sq : IsPullback f' g' g f) (hg : P g) : P g' :=
  IsStableUnderBaseChange.of_isPullback sq hg


/-- Alternative constructor for `IsStableUnderBaseChange`. -/
theorem IsStableUnderBaseChange.mk' {P : MorphismProperty C} [RespectsIso P]
    (hP₂ : ∀ (X Y S : C) (f : X ⟶ S) (g : Y ⟶ S) [HasPullback f g] (_ : P g),
      P (pullback.fst f g)) :
    IsStableUnderBaseChange P where
  of_isPullback {X Y Y' S f g f' g'} sq hg := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      hP₂ : ∀ (X Y S : C) (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : Category …
      X Y Y' S : C
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      sq : CategoryTheory.IsPullback f' g' g f
      hg : P g
      ⊢ P g'
    -/
    haveI : HasPullback f g := sq.flip.hasPullback
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      hP₂ : ∀ (X Y S : C) (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : Category …
      X Y Y' S : C
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      sq : CategoryTheory.IsPullback f' g' g f
      hg : P g
      this : CategoryTheory.Limits.HasPullback f g
      ⊢ P g'
    -/
    let e := sq.flip.isoPullback
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      hP₂ : ∀ (X Y S : C) (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : Category …
      X Y Y' S : C
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      sq : CategoryTheory.IsPullback f' g' g f
      hg : P g
      this : CategoryTheory.Limits.HasPullback f g
      e : CategoryTheory.Iso Y' (CategoryTheory.Limits.pullback f g) := ⋯.isoPullback
      ⊢ P g'
    -/
    rw [← P.cancel_left_of_respectsIso e.inv, sq.flip.isoPullback_inv_fst]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      hP₂ : ∀ (X Y S : C) (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : Category …
      X Y Y' S : C
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      sq : CategoryTheory.IsPullback f' g' g f
      hg : P g
      this : CategoryTheory.Limits.HasPullback f g
      e : CategoryTheory.Iso Y' (CategoryTheory.Limits.pullback f g) := ⋯.isoPullback
      ⊢ P (CategoryTheory.Limits.pullback.fst f g)
    -/
    exact hP₂ _ _ _ f g hg
    /-
      🎉 no goals
    -/


instance IsStableUnderBaseChange.isomorphisms :
    (isomorphisms C).IsStableUnderBaseChange where
  of_isPullback {_ _ _ _ f g _ _} h hg :=
    have : IsIso g := hg
    have := hasPullback_of_left_iso g f
    h.isoPullback_hom_snd ▸ inferInstanceAs (IsIso _)


variable (C) in
instance IsStableUnderBaseChange.monomorphisms :
    (monomorphisms C).IsStableUnderBaseChange where
  of_isPullback {X Y Y' S f g f' g'} h hg := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Y' S : C
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      h : CategoryTheory.IsPullback f' g' g f
      hg : CategoryTheory.MorphismProperty.monomorphisms C g
      ⊢ CategoryTheory.MorphismProperty.monomorphisms C g'
    -/
    have : Mono g := hg
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Y' S : C
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      h : CategoryTheory.IsPullback f' g' g f
      hg : CategoryTheory.MorphismProperty.monomorphisms C g
      this : CategoryTheory.Mono g
      ⊢ CategoryTheory.MorphismProperty.monomorphisms C g'
    -/
    constructor
    /-
      case right_cancellation
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Y' S : C
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      h : CategoryTheory.IsPullback f' g' g f
      hg : CategoryTheory.MorphismProperty.monomorphisms C g
      this : CategoryTheory.Mono g
      ⊢ ∀ {Z : C} (g h : Quiver.Hom Z Y'), Eq (CategoryTheory.CategoryStruct.comp g  …
    -/
    intro Z f₁ f₂ h₁₂
    /-
      case right_cancellation
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Y' S : C
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      h : CategoryTheory.IsPullback f' g' g f
      hg : CategoryTheory.MorphismProperty.monomorphisms C g
      this : CategoryTheory.Mono g
      Z : C
      f₁ f₂ : Quiver.Hom Z Y'
      h₁₂ : Eq (CategoryTheory.CategoryStruct.comp f₁ g') (CategoryTheory.CategorySt …
      ⊢ Eq f₁ f₂
    -/
    apply PullbackCone.IsLimit.hom_ext h.isLimit
      /-
        case right_cancellation.h₀
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Y' S : C
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        f' : Quiver.Hom Y' Y
        g' : Quiver.Hom Y' X
        h : CategoryTheory.IsPullback f' g' g f
        hg : CategoryTheory.MorphismProperty.monomorphisms C g
        this : CategoryTheory.Mono g
        Z : C
        f₁ f₂ : Quiver.Hom Z Y'
        h₁₂ : Eq (CategoryTheory.CategoryStruct.comp f₁ g') (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ h.cone.fst) (CategoryTheory.Catego …
      -/
    · rw [← cancel_mono g]
      /-
        case right_cancellation.h₀
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Y' S : C
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        f' : Quiver.Hom Y' Y
        g' : Quiver.Hom Y' X
        h : CategoryTheory.IsPullback f' g' g f
        hg : CategoryTheory.MorphismProperty.monomorphisms C g
        this : CategoryTheory.Mono g
        Z : C
        f₁ f₂ : Quiver.Hom Z Y'
        h₁₂ : Eq (CategoryTheory.CategoryStruct.comp f₁ g') (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
      dsimp
      /-
        case right_cancellation.h₀
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Y' S : C
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        f' : Quiver.Hom Y' Y
        g' : Quiver.Hom Y' X
        h : CategoryTheory.IsPullback f' g' g f
        hg : CategoryTheory.MorphismProperty.monomorphisms C g
        this : CategoryTheory.Mono g
        Z : C
        f₁ f₂ : Quiver.Hom Z Y'
        h₁₂ : Eq (CategoryTheory.CategoryStruct.comp f₁ g') (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
      simp only [Category.assoc, h.w, reassoc_of% h₁₂]
      /-
        🎉 no goals
      -/
      /-
        case right_cancellation.h₁
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Y' S : C
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        f' : Quiver.Hom Y' Y
        g' : Quiver.Hom Y' X
        h : CategoryTheory.IsPullback f' g' g f
        hg : CategoryTheory.MorphismProperty.monomorphisms C g
        this : CategoryTheory.Mono g
        Z : C
        f₁ f₂ : Quiver.Hom Z Y'
        h₁₂ : Eq (CategoryTheory.CategoryStruct.comp f₁ g') (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ h.cone.snd) (CategoryTheory.Catego …
      -/
    · exact h₁₂
      /-
        🎉 no goals
      -/


instance (priority := 900) IsStableUnderBaseChange.respectsIso {P : MorphismProperty C}
    [IsStableUnderBaseChange P] : RespectsIso P := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    inst✝ : P.IsStableUnderBaseChange
    ⊢ P.RespectsIso
  -/
  apply RespectsIso.of_respects_arrow_iso
  /-
    case hP
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    inst✝ : P.IsStableUnderBaseChange
    ⊢ ∀ (f g : CategoryTheory.Arrow C), CategoryTheory.Iso f g → P f.hom → P g.hom
  -/
  intro f g e
  /-
    case hP
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    inst✝ : P.IsStableUnderBaseChange
    f g : CategoryTheory.Arrow C
    e : CategoryTheory.Iso f g
    ⊢ P f.hom → P g.hom
  -/
  exact of_isPullback (IsPullback.of_horiz_isIso (CommSq.mk e.inv.w))
  /-
    🎉 no goals
  -/


theorem pullback_fst {P : MorphismProperty C} [IsStableUnderBaseChange P]
    {X Y S : C} (f : X ⟶ S) (g : Y ⟶ S) [HasPullback f g] (H : P g) :
    P (pullback.fst f g) :=
  of_isPullback (IsPullback.of_hasPullback f g).flip H


@[deprecated (since := "2024-11-06")] alias IsStableUnderBaseChange.fst := pullback_fst


theorem pullback_snd {P : MorphismProperty C} [IsStableUnderBaseChange P]
    {X Y S : C} (f : X ⟶ S) (g : Y ⟶ S) [HasPullback f g] (H : P f) :
    P (pullback.snd f g) :=
  of_isPullback (IsPullback.of_hasPullback f g) H


@[deprecated (since := "2024-11-06")] alias IsStableUnderBaseChange.snd := pullback_snd


theorem baseChange_obj [HasPullbacks C] {P : MorphismProperty C}
    [IsStableUnderBaseChange P] {S S' : C} (f : S' ⟶ S) (X : Over S) (H : P X.hom) :
    P ((Over.pullback f).obj X).hom :=
  pullback_snd X.hom f H


@[deprecated (since := "2024-11-06")] alias IsStableUnderBaseChange.baseChange_obj := baseChange_obj


theorem baseChange_map [HasPullbacks C] {P : MorphismProperty C}
    [IsStableUnderBaseChange P] {S S' : C} (f : S' ⟶ S) {X Y : Over S} (g : X ⟶ Y)
    (H : P g.left) : P ((Over.pullback f).map g).left := by
  let e :=
    pullbackRightPullbackFstIso Y.hom f g.left ≪≫
      pullback.congrHom (g.w.trans (Category.comp_id _)) rfl
  have : e.inv ≫ (pullback.snd _ _) = ((Over.pullback f).map g).left := by
    ext <;> dsimp [e] <;> simp
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    P : CategoryTheory.MorphismProperty C
    inst✝ : P.IsStableUnderBaseChange
    S S' : C
    f : Quiver.Hom S' S
    X Y : CategoryTheory.Over S
    g : Quiver.Hom X Y
    H : P g.left
    e : CategoryTheory.Iso (CategoryTheory.Limits.pullback g.left (CategoryTheory. …
    this : Eq (CategoryTheory.CategoryStruct.comp e.inv (CategoryTheory.Limits.pul …
    ⊢ P ((CategoryTheory.Over.pullback f).map g).left
  -/
  rw [← this, P.cancel_left_of_respectsIso]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    P : CategoryTheory.MorphismProperty C
    inst✝ : P.IsStableUnderBaseChange
    S S' : C
    f : Quiver.Hom S' S
    X Y : CategoryTheory.Over S
    g : Quiver.Hom X Y
    H : P g.left
    e : CategoryTheory.Iso (CategoryTheory.Limits.pullback g.left (CategoryTheory. …
    this : Eq (CategoryTheory.CategoryStruct.comp e.inv (CategoryTheory.Limits.pul …
    ⊢ P (CategoryTheory.Limits.pullback.snd g.left (CategoryTheory.Limits.pullback …
  -/
  exact pullback_snd _ _ H
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-06")] alias IsStableUnderBaseChange.baseChange_map := baseChange_map


theorem pullback_map [HasPullbacks C] {P : MorphismProperty C}
    [IsStableUnderBaseChange P] [P.IsStableUnderComposition] {S X X' Y Y' : C} {f : X ⟶ S}
    {g : Y ⟶ S} {f' : X' ⟶ S} {g' : Y' ⟶ S} {i₁ : X ⟶ X'} {i₂ : Y ⟶ Y'} (h₁ : P i₁) (h₂ : P i₂)
    (e₁ : f = i₁ ≫ f') (e₂ : g = i₂ ≫ g') :
    P (pullback.map f g f' g' i₁ i₂ (𝟙 _) ((Category.comp_id _).trans e₁)
        ((Category.comp_id _).trans e₂)) := by
  have :
    pullback.map f g f' g' i₁ i₂ (𝟙 _) ((Category.comp_id _).trans e₁)
        ((Category.comp_id _).trans e₂) =
      ((pullbackSymmetry _ _).hom ≫
          ((Over.pullback _).map (Over.homMk _ e₂.symm : Over.mk g ⟶ Over.mk g')).left) ≫
        (pullbackSymmetry _ _).hom ≫
          ((Over.pullback g').map (Over.homMk _ e₁.symm : Over.mk f ⟶ Over.mk f')).left := by
    ext <;> dsimp <;> simp
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    P : CategoryTheory.MorphismProperty C
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : P.IsStableUnderComposition
    S X X' Y Y' : C
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    f' : Quiver.Hom X' S
    g' : Quiver.Hom Y' S
    i₁ : Quiver.Hom X X'
    i₂ : Quiver.Hom Y Y'
    h₁ : P i₁
    h₂ : P i₂
    e₁ : Eq f (CategoryTheory.CategoryStruct.comp i₁ f')
    e₂ : Eq g (CategoryTheory.CategoryStruct.comp i₂ g')
    this : Eq (CategoryTheory.Limits.pullback.map f g f' g' i₁ i₂ (CategoryTheory. …
    ⊢ P (CategoryTheory.Limits.pullback.map f g f' g' i₁ i₂ (CategoryTheory.Catego …
  -/
  rw [this]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    P : CategoryTheory.MorphismProperty C
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : P.IsStableUnderComposition
    S X X' Y Y' : C
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    f' : Quiver.Hom X' S
    g' : Quiver.Hom Y' S
    i₁ : Quiver.Hom X X'
    i₂ : Quiver.Hom Y Y'
    h₁ : P i₁
    h₂ : P i₂
    e₁ : Eq f (CategoryTheory.CategoryStruct.comp i₁ f')
    e₂ : Eq g (CategoryTheory.CategoryStruct.comp i₂ g')
    this : Eq (CategoryTheory.Limits.pullback.map f g f' g' i₁ i₂ (CategoryTheory. …
    ⊢ P (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp (C …
  -/
  apply P.comp_mem <;> rw [P.cancel_left_of_respectsIso]
  exacts [baseChange_map _ (Over.homMk _ e₂.symm : Over.mk g ⟶ Over.mk g') h₂,
    baseChange_map _ (Over.homMk _ e₁.symm : Over.mk f ⟶ Over.mk f') h₁]


@[deprecated (since := "2024-11-06")] alias IsStableUnderBaseChange.pullback_map := pullback_map


lemma of_isPushout {P : MorphismProperty C} [P.IsStableUnderCobaseChange]
    {A A' B B' : C} {f : A ⟶ A'} {g : A ⟶ B} {f' : B ⟶ B'} {g' : A' ⟶ B'}
    (sq : IsPushout g f f' g') (hf : P f) : P f' :=
  IsStableUnderCobaseChange.of_isPushout sq hf


/-- An alternative constructor for `IsStableUnderCobaseChange`. -/
theorem IsStableUnderCobaseChange.mk' {P : MorphismProperty C} [RespectsIso P]
    (hP₂ : ∀ (A B A' : C) (f : A ⟶ A') (g : A ⟶ B) [HasPushout f g] (_ : P f),
      P (pushout.inr f g)) :
    IsStableUnderCobaseChange P where
  of_isPushout {A A' B B' f g f' g'} sq hf := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      hP₂ : ∀ (A B A' : C) (f : Quiver.Hom A A') (g : Quiver.Hom A B) [inst : Catego …
      A A' B B' : C
      f : Quiver.Hom A A'
      g : Quiver.Hom A B
      f' : Quiver.Hom B B'
      g' : Quiver.Hom A' B'
      sq : CategoryTheory.IsPushout g f f' g'
      hf : P f
      ⊢ P f'
    -/
    haveI : HasPushout f g := sq.flip.hasPushout
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      hP₂ : ∀ (A B A' : C) (f : Quiver.Hom A A') (g : Quiver.Hom A B) [inst : Catego …
      A A' B B' : C
      f : Quiver.Hom A A'
      g : Quiver.Hom A B
      f' : Quiver.Hom B B'
      g' : Quiver.Hom A' B'
      sq : CategoryTheory.IsPushout g f f' g'
      hf : P f
      this : CategoryTheory.Limits.HasPushout f g
      ⊢ P f'
    -/
    let e := sq.flip.isoPushout
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      hP₂ : ∀ (A B A' : C) (f : Quiver.Hom A A') (g : Quiver.Hom A B) [inst : Catego …
      A A' B B' : C
      f : Quiver.Hom A A'
      g : Quiver.Hom A B
      f' : Quiver.Hom B B'
      g' : Quiver.Hom A' B'
      sq : CategoryTheory.IsPushout g f f' g'
      hf : P f
      this : CategoryTheory.Limits.HasPushout f g
      e : CategoryTheory.Iso B' (CategoryTheory.Limits.pushout f g) := ⋯.isoPushout
      ⊢ P f'
    -/
    rw [← P.cancel_right_of_respectsIso _ e.hom, sq.flip.inr_isoPushout_hom]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      hP₂ : ∀ (A B A' : C) (f : Quiver.Hom A A') (g : Quiver.Hom A B) [inst : Catego …
      A A' B B' : C
      f : Quiver.Hom A A'
      g : Quiver.Hom A B
      f' : Quiver.Hom B B'
      g' : Quiver.Hom A' B'
      sq : CategoryTheory.IsPushout g f f' g'
      hf : P f
      this : CategoryTheory.Limits.HasPushout f g
      e : CategoryTheory.Iso B' (CategoryTheory.Limits.pushout f g) := ⋯.isoPushout
      ⊢ P (CategoryTheory.Limits.pushout.inr f g)
    -/
    exact hP₂ _ _ _ f g hf
    /-
      🎉 no goals
    -/


instance IsStableUnderCobaseChange.isomorphisms :
    (isomorphisms C).IsStableUnderCobaseChange where
  of_isPushout {_ _ _ _ f g _ _} h (_ : IsIso f) :=
    have := hasPushout_of_right_iso g f
    h.inl_isoPushout_inv ▸ inferInstanceAs (IsIso _)


variable (C) in
instance IsStableUnderCobaseChange.epimorphisms :
    (epimorphisms C).IsStableUnderCobaseChange where
  of_isPushout {X Y Y' S f g f' g'} h hf := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Y' S : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Y'
      f' : Quiver.Hom Y' S
      g' : Quiver.Hom Y S
      h : CategoryTheory.IsPushout g f f' g'
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      ⊢ CategoryTheory.MorphismProperty.epimorphisms C f'
    -/
    have : Epi f := hf
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Y' S : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Y'
      f' : Quiver.Hom Y' S
      g' : Quiver.Hom Y S
      h : CategoryTheory.IsPushout g f f' g'
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      this : CategoryTheory.Epi f
      ⊢ CategoryTheory.MorphismProperty.epimorphisms C f'
    -/
    constructor
    /-
      case left_cancellation
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Y' S : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Y'
      f' : Quiver.Hom Y' S
      g' : Quiver.Hom Y S
      h : CategoryTheory.IsPushout g f f' g'
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      this : CategoryTheory.Epi f
      ⊢ ∀ {Z : C} (g h : Quiver.Hom S Z), Eq (CategoryTheory.CategoryStruct.comp f'  …
    -/
    intro Z f₁ f₂ h₁₂
    /-
      case left_cancellation
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Y' S : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Y'
      f' : Quiver.Hom Y' S
      g' : Quiver.Hom Y S
      h : CategoryTheory.IsPushout g f f' g'
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      this : CategoryTheory.Epi f
      Z : C
      f₁ f₂ : Quiver.Hom S Z
      h₁₂ : Eq (CategoryTheory.CategoryStruct.comp f' f₁) (CategoryTheory.CategorySt …
      ⊢ Eq f₁ f₂
    -/
    apply PushoutCocone.IsColimit.hom_ext h.isColimit
      /-
        case left_cancellation.h₀
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Y' S : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Y'
        f' : Quiver.Hom Y' S
        g' : Quiver.Hom Y S
        h : CategoryTheory.IsPushout g f f' g'
        hf : CategoryTheory.MorphismProperty.epimorphisms C f
        this : CategoryTheory.Epi f
        Z : C
        f₁ f₂ : Quiver.Hom S Z
        h₁₂ : Eq (CategoryTheory.CategoryStruct.comp f' f₁) (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp h.cocone.inl f₁) (CategoryTheory.Cate …
      -/
    · exact h₁₂
      /-
        🎉 no goals
      -/
      /-
        case left_cancellation.h₁
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Y' S : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Y'
        f' : Quiver.Hom Y' S
        g' : Quiver.Hom Y S
        h : CategoryTheory.IsPushout g f f' g'
        hf : CategoryTheory.MorphismProperty.epimorphisms C f
        this : CategoryTheory.Epi f
        Z : C
        f₁ f₂ : Quiver.Hom S Z
        h₁₂ : Eq (CategoryTheory.CategoryStruct.comp f' f₁) (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp h.cocone.inr f₁) (CategoryTheory.Cate …
      -/
    · rw [← cancel_epi f]
      /-
        case left_cancellation.h₁
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Y' S : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Y'
        f' : Quiver.Hom Y' S
        g' : Quiver.Hom Y S
        h : CategoryTheory.IsPushout g f f' g'
        hf : CategoryTheory.MorphismProperty.epimorphisms C f
        this : CategoryTheory.Epi f
        Z : C
        f₁ f₂ : Quiver.Hom S Z
        h₁₂ : Eq (CategoryTheory.CategoryStruct.comp f' f₁) (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      dsimp
      /-
        case left_cancellation.h₁
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Y' S : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Y'
        f' : Quiver.Hom Y' S
        g' : Quiver.Hom Y S
        h : CategoryTheory.IsPushout g f f' g'
        hf : CategoryTheory.MorphismProperty.epimorphisms C f
        this : CategoryTheory.Epi f
        Z : C
        f₁ f₂ : Quiver.Hom S Z
        h₁₂ : Eq (CategoryTheory.CategoryStruct.comp f' f₁) (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      simp only [← reassoc_of% h.w, h₁₂]
      /-
        🎉 no goals
      -/


instance IsStableUnderCobaseChange.respectsIso {P : MorphismProperty C}
    [IsStableUnderCobaseChange P] : RespectsIso P :=
  RespectsIso.of_respects_arrow_iso _ fun _ _ e ↦
    of_isPushout (IsPushout.of_horiz_isIso (CommSq.mk e.hom.w))


theorem pushout_inl {P : MorphismProperty C} [IsStableUnderCobaseChange P]
    {A B A' : C} (f : A ⟶ A') (g : A ⟶ B) [HasPushout f g] (H : P g) :
    P (pushout.inl f g) :=
  of_isPushout (IsPushout.of_hasPushout f g) H


@[deprecated (since := "2024-11-06")] alias IsStableUnderBaseChange.inl := pushout_inl


theorem pushout_inr {P : MorphismProperty C} [IsStableUnderCobaseChange P]
    {A B A' : C} (f : A ⟶ A') (g : A ⟶ B) [HasPushout f g] (H : P f) : P (pushout.inr f g) :=
  of_isPushout (IsPushout.of_hasPushout f g).flip H


@[deprecated (since := "2024-11-06")] alias IsStableUnderBaseChange.inr := pushout_inr


instance IsStableUnderCobaseChange.op {P : MorphismProperty C} [IsStableUnderCobaseChange P] :
    IsStableUnderBaseChange P.op where
  of_isPullback sq hg := P.of_isPushout sq.unop hg


instance IsStableUnderCobaseChange.unop {P : MorphismProperty Cᵒᵖ} [IsStableUnderCobaseChange P] :
    IsStableUnderBaseChange P.unop where
  of_isPullback sq hg := P.of_isPushout sq.op hg


instance IsStableUnderBaseChange.op {P : MorphismProperty C} [IsStableUnderBaseChange P] :
    IsStableUnderCobaseChange P.op where
  of_isPushout sq hf := P.of_isPullback sq.unop hf


instance IsStableUnderBaseChange.unop {P : MorphismProperty Cᵒᵖ} [IsStableUnderBaseChange P] :
    IsStableUnderCobaseChange P.unop where
  of_isPushout sq hf := P.of_isPullback sq.op hf


instance IsStableUnderBaseChange.inf {P Q : MorphismProperty C} [IsStableUnderBaseChange P]
    [IsStableUnderBaseChange Q] :
    IsStableUnderBaseChange (P ⊓ Q) where
  of_isPullback hp hg := ⟨of_isPullback hp hg.left, of_isPullback hp hg.right⟩


instance IsStableUnderCobaseChange.inf {P Q : MorphismProperty C} [IsStableUnderCobaseChange P]
    [IsStableUnderCobaseChange Q] :
    IsStableUnderCobaseChange (P ⊓ Q) where
  of_isPushout hp hg := ⟨of_isPushout hp hg.left, of_isPushout hp hg.right⟩


/-- The property that a morphism property `W` is stable under limits
indexed by a category `J`. -/
def IsStableUnderLimitsOfShape (J : Type*) [Category J] : Prop :=
  ∀ (X₁ X₂ : J ⥤ C) (c₁ : Cone X₁) (c₂ : Cone X₂)
    (_ : IsLimit c₁) (h₂ : IsLimit c₂) (f : X₁ ⟶ X₂) (_ : W.functorCategory J f),
      W (h₂.lift (Cone.mk _ (c₁.π ≫ f)))


/-- The property that a morphism property `W` is stable under colimits
indexed by a category `J`. -/
def IsStableUnderColimitsOfShape (J : Type*) [Category J] : Prop :=
  ∀ (X₁ X₂ : J ⥤ C) (c₁ : Cocone X₁) (c₂ : Cocone X₂)
    (h₁ : IsColimit c₁) (_ : IsColimit c₂) (f : X₁ ⟶ X₂) (_ : W.functorCategory J f),
      W (h₁.desc (Cocone.mk _ (f ≫ c₂.ι)))


lemma IsStableUnderLimitsOfShape.lim_map {J : Type*} [Category J]
    (hW : W.IsStableUnderLimitsOfShape J) {X Y : J ⥤ C}
    (f : X ⟶ Y) [HasLimitsOfShape J C] (hf : W.functorCategory _ f) :
    W (lim.map f) :=
  hW X Y _ _ (limit.isLimit X) (limit.isLimit Y) f hf


lemma IsStableUnderColimitsOfShape.colim_map {J : Type*} [Category J]
    (hW : W.IsStableUnderColimitsOfShape J) {X Y : J ⥤ C}
    (f : X ⟶ Y) [HasColimitsOfShape J C] (hf : W.functorCategory _ f) :
    W (colim.map f) :=
  hW X Y _ _ (colimit.isColimit X) (colimit.isColimit Y) f hf


/-- The property that a morphism property `W` is stable under products indexed by a type `J`. -/
abbrev IsStableUnderProductsOfShape (J : Type*) := W.IsStableUnderLimitsOfShape (Discrete J)


/-- The property that a morphism property `W` is stable under coproducts indexed by a type `J`. -/
abbrev IsStableUnderCoproductsOfShape (J : Type*) := W.IsStableUnderColimitsOfShape (Discrete J)


lemma IsStableUnderProductsOfShape.mk (J : Type*) [W.RespectsIso]
    (hW : ∀ (X₁ X₂ : J → C) [HasProduct X₁] [HasProduct X₂]
      (f : ∀ j, X₁ j ⟶ X₂ j) (_ : ∀ (j : J), W (f j)),
      W (Limits.Pi.map f)) : W.IsStableUnderProductsOfShape J := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasProduct X₁] [inst_1 :  …
    ⊢ W.IsStableUnderProductsOfShape J
  -/
  intro X₁ X₂ c₁ c₂ hc₁ hc₂ f hf
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasProduct X₁] [inst_1 :  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cone X₁
    c₂ : CategoryTheory.Limits.Cone X₂
    hc₁ : CategoryTheory.Limits.IsLimit c₁
    hc₂ : CategoryTheory.Limits.IsLimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    ⊢ W (hc₂.lift { pt := c₁.pt, π := CategoryTheory.CategoryStruct.comp c₁.π f })
  -/
  let φ := fun j => f.app (Discrete.mk j)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasProduct X₁] [inst_1 :  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cone X₁
    c₂ : CategoryTheory.Limits.Cone X₂
    hc₁ : CategoryTheory.Limits.IsLimit c₁
    hc₂ : CategoryTheory.Limits.IsLimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    ⊢ W (hc₂.lift { pt := c₁.pt, π := CategoryTheory.CategoryStruct.comp c₁.π f })
  -/
  have : HasLimit X₁ := ⟨c₁, hc₁⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasProduct X₁] [inst_1 :  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cone X₁
    c₂ : CategoryTheory.Limits.Cone X₂
    hc₁ : CategoryTheory.Limits.IsLimit c₁
    hc₂ : CategoryTheory.Limits.IsLimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this : CategoryTheory.Limits.HasLimit X₁
    ⊢ W (hc₂.lift { pt := c₁.pt, π := CategoryTheory.CategoryStruct.comp c₁.π f })
  -/
  have : HasLimit X₂ := ⟨c₂, hc₂⟩
  have : HasProduct fun j ↦ X₁.obj (Discrete.mk j) :=
    hasLimitOfIso (Discrete.natIso (fun j ↦ Iso.refl (X₁.obj j)))
  have : HasProduct fun j ↦ X₂.obj (Discrete.mk j) :=
    hasLimitOfIso (Discrete.natIso (fun j ↦ Iso.refl (X₂.obj j)))
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasProduct X₁] [inst_1 :  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cone X₁
    c₂ : CategoryTheory.Limits.Cone X₂
    hc₁ : CategoryTheory.Limits.IsLimit c₁
    hc₂ : CategoryTheory.Limits.IsLimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this✝² : CategoryTheory.Limits.HasLimit X₁
    this✝¹ : CategoryTheory.Limits.HasLimit X₂
    this✝ : CategoryTheory.Limits.HasProduct fun j => X₁.obj { as := j }
    this : CategoryTheory.Limits.HasProduct fun j => X₂.obj { as := j }
    ⊢ W (hc₂.lift { pt := c₁.pt, π := CategoryTheory.CategoryStruct.comp c₁.π f })
  -/
  have hf' := hW _ _ φ (fun j => hf (Discrete.mk j))
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasProduct X₁] [inst_1 :  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cone X₁
    c₂ : CategoryTheory.Limits.Cone X₂
    hc₁ : CategoryTheory.Limits.IsLimit c₁
    hc₂ : CategoryTheory.Limits.IsLimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this✝² : CategoryTheory.Limits.HasLimit X₁
    this✝¹ : CategoryTheory.Limits.HasLimit X₂
    this✝ : CategoryTheory.Limits.HasProduct fun j => X₁.obj { as := j }
    this : CategoryTheory.Limits.HasProduct fun j => X₂.obj { as := j }
    hf' : W (CategoryTheory.Limits.Pi.map φ)
    ⊢ W (hc₂.lift { pt := c₁.pt, π := CategoryTheory.CategoryStruct.comp c₁.π f })
  -/
  refine (W.arrow_mk_iso_iff ?_).2 hf'
  refine Arrow.isoMk
    (IsLimit.conePointUniqueUpToIso hc₁ (limit.isLimit X₁) ≪≫ (Pi.isoLimit X₁).symm)
    (IsLimit.conePointUniqueUpToIso hc₂ (limit.isLimit X₂) ≪≫ (Pi.isoLimit _).symm) ?_
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasProduct X₁] [inst_1 :  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cone X₁
    c₂ : CategoryTheory.Limits.Cone X₂
    hc₁ : CategoryTheory.Limits.IsLimit c₁
    hc₂ : CategoryTheory.Limits.IsLimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this✝² : CategoryTheory.Limits.HasLimit X₁
    this✝¹ : CategoryTheory.Limits.HasLimit X₂
    this✝ : CategoryTheory.Limits.HasProduct fun j => X₁.obj { as := j }
    this : CategoryTheory.Limits.HasProduct fun j => X₂.obj { as := j }
    hf' : W (CategoryTheory.Limits.Pi.map φ)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((hc₁.conePointUniqueUpToIso (Categor …
  -/
  apply limit.hom_ext
  /-
    case w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasProduct X₁] [inst_1 :  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cone X₁
    c₂ : CategoryTheory.Limits.Cone X₂
    hc₁ : CategoryTheory.Limits.IsLimit c₁
    hc₂ : CategoryTheory.Limits.IsLimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this✝² : CategoryTheory.Limits.HasLimit X₁
    this✝¹ : CategoryTheory.Limits.HasLimit X₂
    this✝ : CategoryTheory.Limits.HasProduct fun j => X₁.obj { as := j }
    this : CategoryTheory.Limits.HasProduct fun j => X₂.obj { as := j }
    hf' : W (CategoryTheory.Limits.Pi.map φ)
    ⊢ ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp (C …
  -/
  rintro ⟨j⟩
  /-
    case w.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasProduct X₁] [inst_1 :  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cone X₁
    c₂ : CategoryTheory.Limits.Cone X₂
    hc₁ : CategoryTheory.Limits.IsLimit c₁
    hc₂ : CategoryTheory.Limits.IsLimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this✝² : CategoryTheory.Limits.HasLimit X₁
    this✝¹ : CategoryTheory.Limits.HasLimit X₂
    this✝ : CategoryTheory.Limits.HasProduct fun j => X₁.obj { as := j }
    this : CategoryTheory.Limits.HasProduct fun j => X₂.obj { as := j }
    hf' : W (CategoryTheory.Limits.Pi.map φ)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [φ]
  /-
    🎉 no goals
  -/


lemma IsStableUnderCoproductsOfShape.mk (J : Type*) [W.RespectsIso]
    (hW : ∀ (X₁ X₂ : J → C) [HasCoproduct X₁] [HasCoproduct X₂]
      (f : ∀ j, X₁ j ⟶ X₂ j) (_ : ∀ (j : J), W (f j)),
      W (Limits.Sigma.map f)) : W.IsStableUnderCoproductsOfShape J := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasCoproduct X₁] [inst_1  …
    ⊢ W.IsStableUnderCoproductsOfShape J
  -/
  intro X₁ X₂ c₁ c₂ hc₁ hc₂ f hf
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasCoproduct X₁] [inst_1  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cocone X₁
    c₂ : CategoryTheory.Limits.Cocone X₂
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    ⊢ W (hc₁.desc { pt := c₂.pt, ι := CategoryTheory.CategoryStruct.comp f c₂.ι })
  -/
  let φ := fun j => f.app (Discrete.mk j)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasCoproduct X₁] [inst_1  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cocone X₁
    c₂ : CategoryTheory.Limits.Cocone X₂
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    ⊢ W (hc₁.desc { pt := c₂.pt, ι := CategoryTheory.CategoryStruct.comp f c₂.ι })
  -/
  have : HasColimit X₁ := ⟨c₁, hc₁⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasCoproduct X₁] [inst_1  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cocone X₁
    c₂ : CategoryTheory.Limits.Cocone X₂
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this : CategoryTheory.Limits.HasColimit X₁
    ⊢ W (hc₁.desc { pt := c₂.pt, ι := CategoryTheory.CategoryStruct.comp f c₂.ι })
  -/
  have : HasColimit X₂ := ⟨c₂, hc₂⟩
  have : HasCoproduct fun j ↦ X₁.obj (Discrete.mk j) :=
    hasColimitOfIso (Discrete.natIso (fun j ↦ Iso.refl (X₁.obj j)))
  have : HasCoproduct fun j ↦ X₂.obj (Discrete.mk j) :=
    hasColimitOfIso (Discrete.natIso (fun j ↦ Iso.refl (X₂.obj j)))
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasCoproduct X₁] [inst_1  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cocone X₁
    c₂ : CategoryTheory.Limits.Cocone X₂
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this✝² : CategoryTheory.Limits.HasColimit X₁
    this✝¹ : CategoryTheory.Limits.HasColimit X₂
    this✝ : CategoryTheory.Limits.HasCoproduct fun j => X₁.obj { as := j }
    this : CategoryTheory.Limits.HasCoproduct fun j => X₂.obj { as := j }
    ⊢ W (hc₁.desc { pt := c₂.pt, ι := CategoryTheory.CategoryStruct.comp f c₂.ι })
  -/
  have hf' := hW _ _ φ (fun j => hf (Discrete.mk j))
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasCoproduct X₁] [inst_1  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cocone X₁
    c₂ : CategoryTheory.Limits.Cocone X₂
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this✝² : CategoryTheory.Limits.HasColimit X₁
    this✝¹ : CategoryTheory.Limits.HasColimit X₂
    this✝ : CategoryTheory.Limits.HasCoproduct fun j => X₁.obj { as := j }
    this : CategoryTheory.Limits.HasCoproduct fun j => X₂.obj { as := j }
    hf' : W (CategoryTheory.Limits.Sigma.map φ)
    ⊢ W (hc₁.desc { pt := c₂.pt, ι := CategoryTheory.CategoryStruct.comp f c₂.ι })
  -/
  refine (W.arrow_mk_iso_iff ?_).1 hf'
  refine Arrow.isoMk
    ((Sigma.isoColimit _) ≪≫ IsColimit.coconePointUniqueUpToIso (colimit.isColimit X₁) hc₁)
    ((Sigma.isoColimit _) ≪≫ IsColimit.coconePointUniqueUpToIso (colimit.isColimit X₂) hc₂) ?_
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasCoproduct X₁] [inst_1  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cocone X₁
    c₂ : CategoryTheory.Limits.Cocone X₂
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this✝² : CategoryTheory.Limits.HasColimit X₁
    this✝¹ : CategoryTheory.Limits.HasColimit X₂
    this✝ : CategoryTheory.Limits.HasCoproduct fun j => X₁.obj { as := j }
    this : CategoryTheory.Limits.HasCoproduct fun j => X₂.obj { as := j }
    hf' : W (CategoryTheory.Limits.Sigma.map φ)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Sigma.isoColi …
  -/
  apply colimit.hom_ext
  /-
    case w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasCoproduct X₁] [inst_1  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cocone X₁
    c₂ : CategoryTheory.Limits.Cocone X₂
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this✝² : CategoryTheory.Limits.HasColimit X₁
    this✝¹ : CategoryTheory.Limits.HasColimit X₂
    this✝ : CategoryTheory.Limits.HasCoproduct fun j => X₁.obj { as := j }
    this : CategoryTheory.Limits.HasCoproduct fun j => X₂.obj { as := j }
    hf' : W (CategoryTheory.Limits.Sigma.map φ)
    ⊢ ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp (C …
  -/
  rintro ⟨j⟩
  /-
    case w.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W : CategoryTheory.MorphismProperty C
    J : Type u_1
    inst✝ : W.RespectsIso
    hW : ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasCoproduct X₁] [inst_1  …
    X₁ X₂ : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    c₁ : CategoryTheory.Limits.Cocone X₁
    c₂ : CategoryTheory.Limits.Cocone X₂
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    f : Quiver.Hom X₁ X₂
    hf : W.functorCategory (CategoryTheory.Discrete J) f
    φ : (j : J) → Quiver.Hom (X₁.obj { as := j }) (X₂.obj { as := j }) := fun j => …
    this✝² : CategoryTheory.Limits.HasColimit X₁
    this✝¹ : CategoryTheory.Limits.HasColimit X₂
    this✝ : CategoryTheory.Limits.HasCoproduct fun j => X₁.obj { as := j }
    this : CategoryTheory.Limits.HasCoproduct fun j => X₂.obj { as := j }
    hf' : W (CategoryTheory.Limits.Sigma.map φ)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
  -/
  simp [φ]
  /-
    🎉 no goals
  -/


/-- The condition that a property of morphisms is stable by finite products. -/
class IsStableUnderFiniteProducts : Prop where
  isStableUnderProductsOfShape (J : Type) [Finite J] : W.IsStableUnderProductsOfShape J


/-- The condition that a property of morphisms is stable by finite coproducts. -/
class IsStableUnderFiniteCoproducts : Prop where
  isStableUnderCoproductsOfShape (J : Type) [Finite J] : W.IsStableUnderCoproductsOfShape J


lemma isStableUnderProductsOfShape_of_isStableUnderFiniteProducts
    (J : Type) [Finite J] [W.IsStableUnderFiniteProducts] :
    W.IsStableUnderProductsOfShape J :=
  IsStableUnderFiniteProducts.isStableUnderProductsOfShape J


lemma isStableUnderCoproductsOfShape_of_isStableUnderFiniteCoproducts
    (J : Type) [Finite J] [W.IsStableUnderFiniteCoproducts] :
    W.IsStableUnderCoproductsOfShape J :=
  IsStableUnderFiniteCoproducts.isStableUnderCoproductsOfShape J


/-- For `P : MorphismProperty C`, `P.diagonal` is a morphism property that holds for `f : X ⟶ Y`
whenever `P` holds for `X ⟶ Y xₓ Y`. -/
def diagonal (P : MorphismProperty C) : MorphismProperty C := fun _ _ f => P (pullback.diagonal f)


theorem diagonal_iff {X Y : C} {f : X ⟶ Y} : P.diagonal f ↔ P (pullback.diagonal f) :=
  Iff.rfl


instance RespectsIso.diagonal [P.RespectsIso] : P.diagonal.RespectsIso := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    P : CategoryTheory.MorphismProperty C
    inst✝ : P.RespectsIso
    ⊢ P.diagonal.RespectsIso
  -/
  apply RespectsIso.mk
    /-
      case hprecomp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      ⊢ ∀ {X Y Z : C} (e : CategoryTheory.Iso X Y) (f : Quiver.Hom Y Z), P.diagonal  …
    -/
  · introv H
    rwa [diagonal_iff, pullback.diagonal_comp, P.cancel_left_of_respectsIso,
      P.cancel_left_of_respectsIso, ← P.cancel_right_of_respectsIso _
        (pullback.map (e.hom ≫ f) (e.hom ≫ f) f f e.hom e.hom (𝟙 Z) (by simp) (by simp)),
      ← pullback.condition, P.cancel_left_of_respectsIso]
    /-
      case hpostcomp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      ⊢ ∀ {X Y Z : C} (e : CategoryTheory.Iso Y Z) (f : Quiver.Hom X Y), P.diagonal  …
    -/
  · introv H
    /-
      case hpostcomp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      X Y Z : C
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      H : P.diagonal f
      ⊢ P.diagonal (CategoryTheory.CategoryStruct.comp f e.hom)
    -/
    delta diagonal
    /-
      case hpostcomp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      inst✝ : P.RespectsIso
      X Y Z : C
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      H : P.diagonal f
      ⊢ P (CategoryTheory.Limits.pullback.diagonal (CategoryTheory.CategoryStruct.co …
    -/
    rwa [pullback.diagonal_comp, P.cancel_right_of_respectsIso]
    /-
      🎉 no goals
    -/


instance diagonal_isStableUnderComposition [P.IsStableUnderComposition] [RespectsIso P]
    [IsStableUnderBaseChange P] : P.diagonal.IsStableUnderComposition where
  comp_mem _ _ h₁ h₂ := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      inst✝² : P.IsStableUnderComposition
      inst✝¹ : P.RespectsIso
      inst✝ : P.IsStableUnderBaseChange
      X✝ Y✝ Z✝ : C
      x✝¹ : Quiver.Hom X✝ Y✝
      x✝ : Quiver.Hom Y✝ Z✝
      h₁ : P.diagonal x✝¹
      h₂ : P.diagonal x✝
      ⊢ P.diagonal (CategoryTheory.CategoryStruct.comp x✝¹ x✝)
    -/
    rw [diagonal_iff, pullback.diagonal_comp]
    exact P.comp_mem _ _ h₁
      (by simpa only [cancel_left_of_respectsIso] using P.pullback_snd _ _ h₂)


instance IsStableUnderBaseChange.diagonal [IsStableUnderBaseChange P] [P.RespectsIso] :
    P.diagonal.IsStableUnderBaseChange :=
  IsStableUnderBaseChange.mk'
    (by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasPullbacks C
        P : CategoryTheory.MorphismProperty C
        inst✝¹ : P.IsStableUnderBaseChange
        inst✝ : P.RespectsIso
        ⊢ ∀ (X Y S : C) (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : CategoryTheo …
      -/
      introv h
      rw [diagonal_iff, diagonal_pullback_fst, P.cancel_left_of_respectsIso,
        P.cancel_right_of_respectsIso]
      /-
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.Limits.HasPullbacks C
        P : CategoryTheory.MorphismProperty C
        inst✝² : P.IsStableUnderBaseChange
        inst✝¹ : P.RespectsIso
        X Y S : C
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        inst✝ : CategoryTheory.Limits.HasPullback f g
        h : P.diagonal g
        ⊢ P ((CategoryTheory.Over.pullback f).map (CategoryTheory.Over.homMk (Category …
      -/
      exact P.baseChange_map f _ (by simpa))
      /-
        🎉 no goals
      -/


lemma diagonal_isomorphisms : (isomorphisms C).diagonal = monomorphisms C :=
  ext _ _ fun _ _ _ ↦ pullback.isIso_diagonal_iff _


/-- If `P` is multiplicative and stable under base change, having the of-postcomp property
wrt. `Q` is equivalent to `Q` implying `P` on the diagonal. -/
lemma hasOfPostcompProperty_iff_le_diagonal [P.IsStableUnderBaseChange]
    [P.IsMultiplicative] {Q : MorphismProperty C} [Q.IsStableUnderBaseChange] :
    P.HasOfPostcompProperty Q ↔ Q ≤ P.diagonal := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasPullbacks C
    P : CategoryTheory.MorphismProperty C
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : P.IsMultiplicative
    Q : CategoryTheory.MorphismProperty C
    inst✝ : Q.IsStableUnderBaseChange
    ⊢ Iff (P.HasOfPostcompProperty Q) (LE.le Q P.diagonal)
  -/
  refine ⟨fun hP X Y f hf ↦ ?_, fun hP ↦ ⟨fun {Y X S} g f hf hcomp ↦ ?_⟩⟩
    /-
      case refine_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : P.IsMultiplicative
      Q : CategoryTheory.MorphismProperty C
      inst✝ : Q.IsStableUnderBaseChange
      hP : P.HasOfPostcompProperty Q
      X Y : C
      f : Quiver.Hom X Y
      hf : Q f
      ⊢ P.diagonal f
    -/
  · exact hP.of_postcomp _ _ (Q.pullback_fst _ _ hf) (by simpa using P.id_mem X)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : P.IsMultiplicative
      Q : CategoryTheory.MorphismProperty C
      inst✝ : Q.IsStableUnderBaseChange
      hP : LE.le Q P.diagonal
      Y X S : C
      g : Quiver.Hom Y X
      f : Quiver.Hom X S
      hf : Q f
      hcomp : P (CategoryTheory.CategoryStruct.comp g f)
      ⊢ P g
    -/
  · set gr : Y ⟶ pullback (g ≫ f) f := pullback.lift (𝟙 Y) g (by simp)
    /-
      case refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : P.IsMultiplicative
      Q : CategoryTheory.MorphismProperty C
      inst✝ : Q.IsStableUnderBaseChange
      hP : LE.le Q P.diagonal
      Y X S : C
      g : Quiver.Hom Y X
      f : Quiver.Hom X S
      hf : Q f
      hcomp : P (CategoryTheory.CategoryStruct.comp g f)
      gr : Quiver.Hom Y (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStru …
      ⊢ P g
    -/
    have : g = gr ≫ pullback.snd _ _ := by simp [gr]
    /-
      case refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : P.IsMultiplicative
      Q : CategoryTheory.MorphismProperty C
      inst✝ : Q.IsStableUnderBaseChange
      hP : LE.le Q P.diagonal
      Y X S : C
      g : Quiver.Hom Y X
      f : Quiver.Hom X S
      hf : Q f
      hcomp : P (CategoryTheory.CategoryStruct.comp g f)
      gr : Quiver.Hom Y (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStru …
      this : Eq g (CategoryTheory.CategoryStruct.comp gr (CategoryTheory.Limits.pull …
      ⊢ P g
    -/
    rw [this]
    /-
      case refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : P.IsMultiplicative
      Q : CategoryTheory.MorphismProperty C
      inst✝ : Q.IsStableUnderBaseChange
      hP : LE.le Q P.diagonal
      Y X S : C
      g : Quiver.Hom Y X
      f : Quiver.Hom X S
      hf : Q f
      hcomp : P (CategoryTheory.CategoryStruct.comp g f)
      gr : Quiver.Hom Y (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStru …
      this : Eq g (CategoryTheory.CategoryStruct.comp gr (CategoryTheory.Limits.pull …
      ⊢ P (CategoryTheory.CategoryStruct.comp gr (CategoryTheory.Limits.pullback.snd …
    -/
    apply P.comp_mem
      /-
        case refine_2.hf
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.Limits.HasPullbacks C
        P : CategoryTheory.MorphismProperty C
        inst✝² : P.IsStableUnderBaseChange
        inst✝¹ : P.IsMultiplicative
        Q : CategoryTheory.MorphismProperty C
        inst✝ : Q.IsStableUnderBaseChange
        hP : LE.le Q P.diagonal
        Y X S : C
        g : Quiver.Hom Y X
        f : Quiver.Hom X S
        hf : Q f
        hcomp : P (CategoryTheory.CategoryStruct.comp g f)
        gr : Quiver.Hom Y (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStru …
        this : Eq g (CategoryTheory.CategoryStruct.comp gr (CategoryTheory.Limits.pull …
        ⊢ P gr
      -/
    · exact P.of_isPullback (pullback_lift_diagonal_isPullback g f) (hP _ hf)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.hg
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.Limits.HasPullbacks C
        P : CategoryTheory.MorphismProperty C
        inst✝² : P.IsStableUnderBaseChange
        inst✝¹ : P.IsMultiplicative
        Q : CategoryTheory.MorphismProperty C
        inst✝ : Q.IsStableUnderBaseChange
        hP : LE.le Q P.diagonal
        Y X S : C
        g : Quiver.Hom Y X
        f : Quiver.Hom X S
        hf : Q f
        hcomp : P (CategoryTheory.CategoryStruct.comp g f)
        gr : Quiver.Hom Y (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStru …
        this : Eq g (CategoryTheory.CategoryStruct.comp gr (CategoryTheory.Limits.pull …
        ⊢ P (CategoryTheory.Limits.pullback.snd (CategoryTheory.CategoryStruct.comp g  …
      -/
    · exact P.pullback_snd _ _ hcomp
      /-
        🎉 no goals
      -/


/-- `P.universally` holds for a morphism `f : X ⟶ Y` iff `P` holds for all `X ×[Y] Y' ⟶ Y'`. -/
def universally (P : MorphismProperty C) : MorphismProperty C := fun X Y f =>
  ∀ ⦃X' Y' : C⦄ (i₁ : X' ⟶ X) (i₂ : Y' ⟶ Y) (f' : X' ⟶ Y') (_ : IsPullback f' i₁ i₂ f), P f'


instance universally_respectsIso (P : MorphismProperty C) : P.universally.RespectsIso := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    ⊢ P.universally.RespectsIso
  -/
  apply RespectsIso.mk
    /-
      case hprecomp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      ⊢ ∀ {X Y Z : C} (e : CategoryTheory.Iso X Y) (f : Quiver.Hom Y Z), P.universal …
    -/
  · intro X Y Z e f hf X' Z' i₁ i₂ f' H
    have : IsPullback (𝟙 _) (i₁ ≫ e.hom) i₁ e.inv :=
      IsPullback.of_horiz_isIso
        ⟨by rw [Category.id_comp, Category.assoc, e.hom_inv_id, Category.comp_id]⟩
    exact hf _ _ _
      (by simpa only [Iso.inv_hom_id_assoc, Category.id_comp] using this.paste_horiz H)
    /-
      case hpostcomp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      ⊢ ∀ {X Y Z : C} (e : CategoryTheory.Iso Y Z) (f : Quiver.Hom X Y), P.universal …
    -/
  · intro X Y Z e f hf X' Z' i₁ i₂ f' H
    have : IsPullback (𝟙 _) i₂ (i₂ ≫ e.inv) e.inv :=
      IsPullback.of_horiz_isIso ⟨Category.id_comp _⟩
    exact hf _ _ _ (by simpa only [Category.assoc, Iso.hom_inv_id,
      Category.comp_id, Category.comp_id] using H.paste_horiz this)


instance universally_isStableUnderBaseChange (P : MorphismProperty C) :
    P.universally.IsStableUnderBaseChange where
  of_isPullback H h₁ _ _ _ _ _ H' := h₁ _ _ _ (H'.paste_vert H.flip)


instance IsStableUnderComposition.universally [HasPullbacks C] (P : MorphismProperty C)
    [hP : P.IsStableUnderComposition] : P.universally.IsStableUnderComposition where
  comp_mem {X Y Z} f g hf hg X' Z' i₁ i₂ f' H := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      hP : P.IsStableUnderComposition
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P.universally f
      hg : P.universally g
      X' Z' : C
      i₁ : Quiver.Hom X' X
      i₂ : Quiver.Hom Z' Z
      f' : Quiver.Hom X' Z'
      H : CategoryTheory.IsPullback f' i₁ i₂ (CategoryTheory.CategoryStruct.comp f g)
      ⊢ P f'
    -/
    have := pullback.lift_fst _ _ (H.w.trans (Category.assoc _ _ _).symm)
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      hP : P.IsStableUnderComposition
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P.universally f
      hg : P.universally g
      X' Z' : C
      i₁ : Quiver.Hom X' X
      i₂ : Quiver.Hom Z' Z
      f' : Quiver.Hom X' Z'
      H : CategoryTheory.IsPullback f' i₁ i₂ (CategoryTheory.CategoryStruct.comp f g)
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback. …
      ⊢ P f'
    -/
    rw [← this] at H ⊢
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      hP : P.IsStableUnderComposition
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P.universally f
      hg : P.universally g
      X' Z' : C
      i₁ : Quiver.Hom X' X
      i₂ : Quiver.Hom Z' Z
      f' : Quiver.Hom X' Z'
      H✝ : CategoryTheory.IsPullback f' i₁ i₂ (CategoryTheory.CategoryStruct.comp f g)
      H : CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.comp (CategoryThe …
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback. …
      ⊢ P (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift f …
    -/
    apply P.comp_mem _ _ _ (hg _ _ _ <| IsPullback.of_hasPullback _ _)
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      P : CategoryTheory.MorphismProperty C
      hP : P.IsStableUnderComposition
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P.universally f
      hg : P.universally g
      X' Z' : C
      i₁ : Quiver.Hom X' X
      i₂ : Quiver.Hom Z' Z
      f' : Quiver.Hom X' Z'
      H✝ : CategoryTheory.IsPullback f' i₁ i₂ (CategoryTheory.CategoryStruct.comp f g)
      H : CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.comp (CategoryThe …
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback. …
      ⊢ P (CategoryTheory.Limits.pullback.lift f' (CategoryTheory.CategoryStruct.com …
    -/
    exact hf _ _ _ (H.of_right (pullback.lift_snd _ _ _) (IsPullback.of_hasPullback i₂ g))
    /-
      🎉 no goals
    -/


theorem universally_le (P : MorphismProperty C) : P.universally ≤ P := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    ⊢ LE.le P.universally P
  -/
  intro X Y f hf
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    X Y : C
    f : Quiver.Hom X Y
    hf : P.universally f
    ⊢ P f
  -/
  exact hf (𝟙 _) (𝟙 _) _ (IsPullback.of_vert_isIso ⟨by rw [Category.comp_id, Category.id_comp]⟩)
  /-
    🎉 no goals
  -/


theorem universally_inf (P Q : MorphismProperty C) :
    (P ⊓ Q).universally = P.universally ⊓ Q.universally := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P Q : CategoryTheory.MorphismProperty C
    ⊢ Eq (Min.min P Q).universally (Min.min P.universally Q.universally)
  -/
  ext X Y f
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P Q : CategoryTheory.MorphismProperty C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff ((Min.min P Q).universally f) (Min.min P.universally Q.universally f)
  -/
  show _ ↔ _ ∧ _
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P Q : CategoryTheory.MorphismProperty C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff ((Min.min P Q).universally f) (And (P.universally f) (Q.universally f))
  -/
  simp_rw [universally, ← forall_and]
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P Q : CategoryTheory.MorphismProperty C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (∀ ⦃X' Y' : C⦄ (i₁ : Quiver.Hom X' X) (i₂ : Quiver.Hom Y' Y) (f' : Quive …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem universally_eq_iff {P : MorphismProperty C} :
    P.universally = P ↔ P.IsStableUnderBaseChange :=
  ⟨(· ▸ P.universally_isStableUnderBaseChange),
    fun hP ↦ P.universally_le.antisymm fun _ _ _ hf _ _ _ _ _ H => hP.of_isPullback H.flip hf⟩


theorem IsStableUnderBaseChange.universally_eq {P : MorphismProperty C}
    [hP : P.IsStableUnderBaseChange] : P.universally = P := universally_eq_iff.mpr hP


theorem universally_mono : Monotone (universally : MorphismProperty C → MorphismProperty C) :=
  fun _ _ h _ _ _ h₁ _ _ _ _ _ H => h _ (h₁ _ _ _ H)


