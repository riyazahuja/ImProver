/--
The pretopology on the category of schemes defined by covering families where the components
satisfy `P`.

The coverings are defined via existence of a `P`-cover. This is convenient in practice, as one
directly has the cover available. For a pretopology generating the same Grothendieck topology, see
`AlgebraicGeometry.Scheme.grothendieckTopology_eq_inf`.
-/
def pretopology : Pretopology Scheme.{u} where
  coverings Y S := ∃ (U : Cover.{u} P Y), S = Presieve.ofArrows U.obj U.map
  has_isos _ _ f _ := ⟨coverOfIsIso f, (Presieve.ofArrows_pUnit _).symm⟩
  pullbacks := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      ⊢ ∀ ⦃X Y : AlgebraicGeometry.Scheme⦄ (f : Quiver.Hom Y X) (S : CategoryTheory. …
    -/
    rintro Y X f _ ⟨U, rfl⟩
    /-
      case intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      Y X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : AlgebraicGeometry.Scheme.Cover P Y
      ⊢ Membership.mem ((fun Y S => Exists fun U => Eq S (CategoryTheory.Presieve.of …
    -/
    exact ⟨U.pullbackCover' f, (Presieve.ofArrows_pullback _ _ _).symm⟩
    /-
      🎉 no goals
    -/
  transitive := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      ⊢ ∀ ⦃X : AlgebraicGeometry.Scheme⦄ (S : CategoryTheory.Presieve X) (Ti : ⦃Y :  …
    -/
    rintro X _ T ⟨U, rfl⟩ H
    /-
      case intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : AlgebraicGeometry.Scheme
      U : AlgebraicGeometry.Scheme.Cover P X
      T : ⦃Y : AlgebraicGeometry.Scheme⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Pre …
      H : ∀ ⦃Y : AlgebraicGeometry.Scheme⦄ (f : Quiver.Hom Y X) (H : CategoryTheory. …
      ⊢ Membership.mem ((fun Y S => Exists fun U => Eq S (CategoryTheory.Presieve.of …
    -/
    choose V hV using H
    /-
      case intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : AlgebraicGeometry.Scheme
      U : AlgebraicGeometry.Scheme.Cover P X
      T : ⦃Y : AlgebraicGeometry.Scheme⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Pre …
      V : ⦃Y : AlgebraicGeometry.Scheme⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Pre …
      hV : ∀ ⦃Y : AlgebraicGeometry.Scheme⦄ (f : Quiver.Hom Y X) (H : CategoryTheory …
      ⊢ Membership.mem ((fun Y S => Exists fun U => Eq S (CategoryTheory.Presieve.of …
    -/
    use U.bind (fun j => V (U.map j) ⟨j⟩)
    simpa only [Cover.bind, ← hV] using Presieve.ofArrows_bind U.obj U.map _
      (fun _ f H => (V f H).obj) (fun _ f H => (V f H).map)


/-- The Grothendieck topology on the category of schemes induced by the pretopology defined by
`P`-covers. -/
abbrev grothendieckTopology : GrothendieckTopology Scheme.{u} :=
  (pretopology P).toGrothendieck


/-- The pretopology on the category of schemes defined by jointly surjective families.

Note: The assumption `IsJointlySurjectivePreserving ⊤` is mathematically unneeded, and only here
to reduce imports. To satisfy it, use `AlgebraicGeometry.Scheme.isJointlySurjectivePreserving`. -/
def surjectiveFamiliesPretopology [IsJointlySurjectivePreserving ⊤] : Pretopology Scheme.{u} where
  coverings X S :=
    ∀ x : X, ∃ (Y : Scheme.{u}) (y : Y) (f : Y ⟶ X) (hf : S f), f.base y = x
  has_isos X Y f hf x := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom Y X
      hf : CategoryTheory.IsIso f
      x : ↑↑X.toPresheafedSpace
      ⊢ Exists fun Y_1 => Exists fun y => Exists fun f_1 => Exists fun hf => Eq (f_1 …
    -/
    use Y, (inv f).base x, f
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom Y X
      hf : CategoryTheory.IsIso f
      x : ↑↑X.toPresheafedSpace
      ⊢ Exists fun hf_1 => Eq (f.base ((CategoryTheory.inv f).base x)) x
    -/
    simp [← Scheme.comp_base_apply]
    /-
      🎉 no goals
    -/
  pullbacks X Y f S hS x := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      hS : Membership.mem ((fun X S => ∀ (x : ↑↑X.toPresheafedSpace), Exists fun Y = …
      x : ↑↑Y.toPresheafedSpace
      ⊢ Exists fun Y_1 => Exists fun y => Exists fun f_1 => Exists fun hf => Eq (f_1 …
    -/
    obtain ⟨Z, z, g, hg, hz⟩ := hS (f.base x)
    obtain ⟨w, hw⟩ :=
      IsJointlySurjectivePreserving.exists_preimage_snd_triplet_of_prop (P := ⊤) trivial z x hz
    /-
      case intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      hS : Membership.mem ((fun X S => ∀ (x : ↑↑X.toPresheafedSpace), Exists fun Y = …
      x : ↑↑Y.toPresheafedSpace
      Z : AlgebraicGeometry.Scheme
      z : ↑↑Z.toPresheafedSpace
      g : Quiver.Hom Z X
      hg : S g
      hz : Eq (g.base z) (f.base x)
      w : ↑↑(CategoryTheory.Limits.pullback g f).toPresheafedSpace
      hw : Eq ((CategoryTheory.Limits.pullback.snd g f).base w) x
      ⊢ Exists fun Y_1 => Exists fun y => Exists fun f_1 => Exists fun hf => Eq (f_1 …
    -/
    use pullback g f, w, pullback.snd g f
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      hS : Membership.mem ((fun X S => ∀ (x : ↑↑X.toPresheafedSpace), Exists fun Y = …
      x : ↑↑Y.toPresheafedSpace
      Z : AlgebraicGeometry.Scheme
      z : ↑↑Z.toPresheafedSpace
      g : Quiver.Hom Z X
      hg : S g
      hz : Eq (g.base z) (f.base x)
      w : ↑↑(CategoryTheory.Limits.pullback g f).toPresheafedSpace
      hw : Eq ((CategoryTheory.Limits.pullback.snd g f).base w) x
      ⊢ Exists fun hf => Eq ((CategoryTheory.Limits.pullback.snd g f).base w) x
    -/
    simpa [hw] using Presieve.pullbackArrows.mk Z g hg
    /-
      🎉 no goals
    -/
  transitive X S T hS hT x := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
      X : AlgebraicGeometry.Scheme
      S : CategoryTheory.Presieve X
      T : ⦃Y : AlgebraicGeometry.Scheme⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheo …
      hS : Membership.mem ((fun X S => ∀ (x : ↑↑X.toPresheafedSpace), Exists fun Y = …
      hT : ∀ ⦃Y : AlgebraicGeometry.Scheme⦄ (f : Quiver.Hom Y X) (H : S f), Membersh …
      x : ↑↑X.toPresheafedSpace
      ⊢ Exists fun Y => Exists fun y => Exists fun f => Exists fun hf => Eq (f.base  …
    -/
    obtain ⟨Y, y, f, hf, hy⟩ := hS x
    /-
      case intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
      X : AlgebraicGeometry.Scheme
      S : CategoryTheory.Presieve X
      T : ⦃Y : AlgebraicGeometry.Scheme⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheo …
      hS : Membership.mem ((fun X S => ∀ (x : ↑↑X.toPresheafedSpace), Exists fun Y = …
      hT : ∀ ⦃Y : AlgebraicGeometry.Scheme⦄ (f : Quiver.Hom Y X) (H : S f), Membersh …
      x : ↑↑X.toPresheafedSpace
      Y : AlgebraicGeometry.Scheme
      y : ↑↑Y.toPresheafedSpace
      f : Quiver.Hom Y X
      hf : S f
      hy : Eq (f.base y) x
      ⊢ Exists fun Y => Exists fun y => Exists fun f => Exists fun hf => Eq (f.base  …
    -/
    obtain ⟨Z, z, g, hg, hz⟩ := hT f hf y
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
      X : AlgebraicGeometry.Scheme
      S : CategoryTheory.Presieve X
      T : ⦃Y : AlgebraicGeometry.Scheme⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheo …
      hS : Membership.mem ((fun X S => ∀ (x : ↑↑X.toPresheafedSpace), Exists fun Y = …
      hT : ∀ ⦃Y : AlgebraicGeometry.Scheme⦄ (f : Quiver.Hom Y X) (H : S f), Membersh …
      x : ↑↑X.toPresheafedSpace
      Y : AlgebraicGeometry.Scheme
      y : ↑↑Y.toPresheafedSpace
      f : Quiver.Hom Y X
      hf : S f
      hy : Eq (f.base y) x
      Z : AlgebraicGeometry.Scheme
      z : ↑↑Z.toPresheafedSpace
      g : Quiver.Hom Z Y
      hg : T f hf g
      hz : Eq (g.base z) y
      ⊢ Exists fun Y => Exists fun y => Exists fun f => Exists fun hf => Eq (f.base  …
    -/
    use Z, z, g ≫ f
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
      X : AlgebraicGeometry.Scheme
      S : CategoryTheory.Presieve X
      T : ⦃Y : AlgebraicGeometry.Scheme⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheo …
      hS : Membership.mem ((fun X S => ∀ (x : ↑↑X.toPresheafedSpace), Exists fun Y = …
      hT : ∀ ⦃Y : AlgebraicGeometry.Scheme⦄ (f : Quiver.Hom Y X) (H : S f), Membersh …
      x : ↑↑X.toPresheafedSpace
      Y : AlgebraicGeometry.Scheme
      y : ↑↑Y.toPresheafedSpace
      f : Quiver.Hom Y X
      hf : S f
      hy : Eq (f.base y) x
      Z : AlgebraicGeometry.Scheme
      z : ↑↑Z.toPresheafedSpace
      g : Quiver.Hom Z Y
      hg : T f hf g
      hz : Eq (g.base z) y
      ⊢ Exists fun hf => Eq ((CategoryTheory.CategoryStruct.comp g f).base z) x
    -/
    simpa [hz, hy] using Presieve.bind_comp f hf hg
    /-
      🎉 no goals
    -/


lemma pretopology_le_inf [IsJointlySurjectivePreserving ⊤] :
    pretopology P ≤ surjectiveFamiliesPretopology ⊓ P.pretopology := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    ⊢ LE.le (AlgebraicGeometry.Scheme.pretopology P) (Min.min AlgebraicGeometry.Sc …
  -/
  rintro X S ⟨𝒰, rfl⟩
  /-
    case intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    ⊢ Membership.mem ((Min.min AlgebraicGeometry.Scheme.surjectiveFamiliesPretopol …
  -/
  refine ⟨fun x ↦ ?_, fun ⟨i⟩ ↦ 𝒰.map_prop i⟩
  /-
    case intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    x : ↑↑X.toPresheafedSpace
    ⊢ Exists fun Y => Exists fun y => Exists fun f => Exists fun hf => Eq (f.base  …
  -/
  obtain ⟨a, ha⟩ := 𝒰.covers x
  /-
    case intro.intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    x : ↑↑X.toPresheafedSpace
    a : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace
    ha : Eq ((𝒰.map (𝒰.f x)).base a) x
    ⊢ Exists fun Y => Exists fun y => Exists fun f => Exists fun hf => Eq (f.base  …
  -/
  refine ⟨𝒰.obj (𝒰.f x), a, 𝒰.map (𝒰.f x), ⟨_⟩, ha⟩
  /-
    🎉 no goals
  -/


/--
The Grothendieck topology defined by `P`-covers agrees with the Grothendieck
topology induced by the intersection of the pretopology of surjective families with
the pretopology defined by `P`.

Note: Because of size issues, this does not hold on the level of pretopologies: A presieve
in the intersection can have up to `Type (u + 1)` many components, while in the definition
of `AlgebraicGeometry.Scheme.pretopology` we only allow `Type u` many components.
-/
lemma grothendieckTopology_eq_inf [IsJointlySurjectivePreserving ⊤] :
    grothendieckTopology P = (surjectiveFamiliesPretopology ⊓ P.pretopology).toGrothendieck := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    ⊢ Eq (AlgebraicGeometry.Scheme.grothendieckTopology P) (CategoryTheory.Pretopo …
  -/
  apply le_antisymm ((Pretopology.gi Scheme.{u}).gc.monotone_l (pretopology_le_inf P))
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    ⊢ LE.le (CategoryTheory.Pretopology.toGrothendieck AlgebraicGeometry.Scheme (M …
  -/
  intro X S ⟨T, ⟨hs, hP⟩, hle⟩
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    X : AlgebraicGeometry.Scheme
    S : CategoryTheory.Sieve X
    T : CategoryTheory.Presieve X
    hs : Membership.mem (AlgebraicGeometry.Scheme.surjectiveFamiliesPretopology.co …
    hP : Membership.mem (P.pretopology.coverings X) T
    hle : LE.le T S.arrows
    ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck AlgebraicGeometry …
  -/
  let _ : Type (u + 1) := Presieve X
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    X : AlgebraicGeometry.Scheme
    S : CategoryTheory.Sieve X
    T : CategoryTheory.Presieve X
    hs : Membership.mem (AlgebraicGeometry.Scheme.surjectiveFamiliesPretopology.co …
    hP : Membership.mem (P.pretopology.coverings X) T
    hle : LE.le T S.arrows
    x✝ : Type (u + 1) := CategoryTheory.Presieve X
    ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck AlgebraicGeometry …
  -/
  let J := (Y : Scheme.{u}) × (Y ⟶ X)
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    X : AlgebraicGeometry.Scheme
    S : CategoryTheory.Sieve X
    T : CategoryTheory.Presieve X
    hs : Membership.mem (AlgebraicGeometry.Scheme.surjectiveFamiliesPretopology.co …
    hP : Membership.mem (P.pretopology.coverings X) T
    hle : LE.le T S.arrows
    x✝ : Type (u + 1) := CategoryTheory.Presieve X
    J : Type (u + 1) := Sigma fun Y => Quiver.Hom Y X
    ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck AlgebraicGeometry …
  -/
  choose Y y f hf hy using hs
  let 𝒰 : Cover.{u} P X :=
    { J := X
      obj := Y
      map := f
      f := id
      covers := fun x ↦ ⟨y x, hy x⟩
      map_prop := fun x ↦ hP (hf x)
    }
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    X : AlgebraicGeometry.Scheme
    S : CategoryTheory.Sieve X
    T : CategoryTheory.Presieve X
    hP : Membership.mem (P.pretopology.coverings X) T
    hle : LE.le T S.arrows
    x✝ : Type (u + 1) := CategoryTheory.Presieve X
    J : Type (u + 1) := Sigma fun Y => Quiver.Hom Y X
    Y : ↑↑X.toPresheafedSpace → AlgebraicGeometry.Scheme
    y : (x : ↑↑X.toPresheafedSpace) → ↑↑(Y x).toPresheafedSpace
    f : (x : ↑↑X.toPresheafedSpace) → Quiver.Hom (Y x) X
    hf : ∀ (x : ↑↑X.toPresheafedSpace), T (f x)
    hy : ∀ (x : ↑↑X.toPresheafedSpace), Eq ((f x).base (y x)) x
    𝒰 : AlgebraicGeometry.Scheme.Cover P X := { J := ↑↑X.toPresheafedSpace, obj := …
    ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck AlgebraicGeometry …
  -/
  refine ⟨Presieve.ofArrows 𝒰.obj 𝒰.map, ⟨𝒰, rfl⟩, ?_⟩
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    X : AlgebraicGeometry.Scheme
    S : CategoryTheory.Sieve X
    T : CategoryTheory.Presieve X
    hP : Membership.mem (P.pretopology.coverings X) T
    hle : LE.le T S.arrows
    x✝ : Type (u + 1) := CategoryTheory.Presieve X
    J : Type (u + 1) := Sigma fun Y => Quiver.Hom Y X
    Y : ↑↑X.toPresheafedSpace → AlgebraicGeometry.Scheme
    y : (x : ↑↑X.toPresheafedSpace) → ↑↑(Y x).toPresheafedSpace
    f : (x : ↑↑X.toPresheafedSpace) → Quiver.Hom (Y x) X
    hf : ∀ (x : ↑↑X.toPresheafedSpace), T (f x)
    hy : ∀ (x : ↑↑X.toPresheafedSpace), Eq ((f x).base (y x)) x
    𝒰 : AlgebraicGeometry.Scheme.Cover P X := { J := ↑↑X.toPresheafedSpace, obj := …
    ⊢ LE.le (CategoryTheory.Presieve.ofArrows 𝒰.obj 𝒰.map) S.arrows
  -/
  rintro Z g ⟨x⟩
  /-
    case mk
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Top.top
    X : AlgebraicGeometry.Scheme
    S : CategoryTheory.Sieve X
    T : CategoryTheory.Presieve X
    hP : Membership.mem (P.pretopology.coverings X) T
    hle : LE.le T S.arrows
    x✝ : Type (u + 1) := CategoryTheory.Presieve X
    J : Type (u + 1) := Sigma fun Y => Quiver.Hom Y X
    Y✝ : ↑↑X.toPresheafedSpace → AlgebraicGeometry.Scheme
    y : (x : ↑↑X.toPresheafedSpace) → ↑↑(Y✝ x).toPresheafedSpace
    f : (x : ↑↑X.toPresheafedSpace) → Quiver.Hom (Y✝ x) X
    hf : ∀ (x : ↑↑X.toPresheafedSpace), T (f x)
    hy : ∀ (x : ↑↑X.toPresheafedSpace), Eq ((f x).base (y x)) x
    𝒰 : AlgebraicGeometry.Scheme.Cover P X := { J := ↑↑X.toPresheafedSpace, obj := …
    Y : AlgebraicGeometry.Scheme
    x : 𝒰.J
    ⊢ Membership.mem S.arrows (𝒰.map x)
  -/
  exact hle _ (hf x)
  /-
    🎉 no goals
  -/


lemma pretopology_cover {Y : Scheme.{u}} (𝒰 : Cover.{u} P Y) :
    pretopology P Y (Presieve.ofArrows 𝒰.obj 𝒰.map) :=
  ⟨𝒰, rfl⟩


lemma grothendieckTopology_cover {X : Scheme.{u}} (𝒰 : Cover.{v} P X) :
    grothendieckTopology P X (Sieve.generate (Presieve.ofArrows 𝒰.obj 𝒰.map)) := by
  let 𝒱 : Cover.{u} P X :=
    { J := X
      obj := fun x ↦ 𝒰.obj (𝒰.f x)
      map := fun x ↦ 𝒰.map (𝒰.f x)
      f := id
      covers := 𝒰.covers
      map_prop := fun _ ↦ 𝒰.map_prop _
    }
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝³ : P.IsMultiplicative
    inst✝² : P.RespectsIso
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    𝒱 : AlgebraicGeometry.Scheme.Cover P X := { J := ↑↑X.toPresheafedSpace, obj := …
    ⊢ (AlgebraicGeometry.Scheme.grothendieckTopology P) X (CategoryTheory.Sieve.ge …
  -/
  refine ⟨_, pretopology_cover 𝒱, ?_⟩
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝³ : P.IsMultiplicative
    inst✝² : P.RespectsIso
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    𝒱 : AlgebraicGeometry.Scheme.Cover P X := { J := ↑↑X.toPresheafedSpace, obj := …
    ⊢ LE.le (CategoryTheory.Presieve.ofArrows 𝒱.obj 𝒱.map) (CategoryTheory.Sieve.g …
  -/
  rintro _ _ ⟨y⟩
  /-
    case mk
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝³ : P.IsMultiplicative
    inst✝² : P.RespectsIso
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    𝒱 : AlgebraicGeometry.Scheme.Cover P X := { J := ↑↑X.toPresheafedSpace, obj := …
    Y : AlgebraicGeometry.Scheme
    y : 𝒱.J
    ⊢ Membership.mem (CategoryTheory.Sieve.generate (CategoryTheory.Presieve.ofArr …
  -/
  exact ⟨_, 𝟙 _, 𝒰.map (𝒰.f y), ⟨_⟩, by simp [𝒱]⟩
  /-
    🎉 no goals
  -/


lemma pretopology_le_pretopology (hPQ : P ≤ Q) :
    pretopology P ≤ pretopology Q := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁷ : P.IsMultiplicative
    inst✝⁶ : P.RespectsIso
    inst✝⁵ : P.IsStableUnderBaseChange
    inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝³ : Q.IsMultiplicative
    inst✝² : Q.RespectsIso
    inst✝¹ : Q.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Q
    hPQ : LE.le P Q
    ⊢ LE.le (AlgebraicGeometry.Scheme.pretopology P) (AlgebraicGeometry.Scheme.pre …
  -/
  rintro X - ⟨𝒰, rfl⟩
  /-
    case intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁷ : P.IsMultiplicative
    inst✝⁶ : P.RespectsIso
    inst✝⁵ : P.IsStableUnderBaseChange
    inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝³ : Q.IsMultiplicative
    inst✝² : Q.RespectsIso
    inst✝¹ : Q.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Q
    hPQ : LE.le P Q
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    ⊢ Membership.mem ((AlgebraicGeometry.Scheme.pretopology Q).coverings X) (Categ …
  -/
  use 𝒰.changeProp Q (fun j ↦ hPQ _ (𝒰.map_prop j))
  /-
    case h
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝⁷ : P.IsMultiplicative
    inst✝⁶ : P.RespectsIso
    inst✝⁵ : P.IsStableUnderBaseChange
    inst✝⁴ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝³ : Q.IsMultiplicative
    inst✝² : Q.RespectsIso
    inst✝¹ : Q.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving Q
    hPQ : LE.le P Q
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    ⊢ Eq (CategoryTheory.Presieve.ofArrows 𝒰.obj 𝒰.map) (CategoryTheory.Presieve.o …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma grothendieckTopology_le_grothendieckTopology (hPQ : P ≤ Q) :
    grothendieckTopology P ≤ grothendieckTopology Q :=
  (Pretopology.gi Scheme.{u}).gc.monotone_l (pretopology_le_pretopology hPQ)


