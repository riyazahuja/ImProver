/--
A morphism of schemes `f : X ⟶ Y` is smooth if for each `x : X` there
exists an affine open neighborhood `V` of `x` and an affine open neighborhood `U` of
`f.base x` with `V ≤ f ⁻¹ᵁ U` such that the induced map `Γ(Y, U) ⟶ Γ(X, V)` is
standard smooth.
-/
@[mk_iff]
class IsSmooth : Prop where
  exists_isStandardSmooth : ∀ (x : X), ∃ (U : Y.affineOpens) (V : X.affineOpens) (_ : x ∈ V.1)
    (e : V.1 ≤ f ⁻¹ᵁ U.1), IsStandardSmooth.{0, 0} (f.appLE U V e).hom


/-- The property of scheme morphisms `IsSmooth` is associated with the ring
homomorphism property `Locally IsStandardSmooth.{0, 0}`. -/
instance : HasRingHomProperty @IsSmooth (Locally IsStandardSmooth.{0, 0}) := by
  /-
    n m : Nat
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.HasRingHomProperty @AlgebraicGeometry.IsSmooth fun {R S} [ …
  -/
  apply HasRingHomProperty.locally_of_iff
    /-
      case hQl
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => Rin …
    -/
  · exact isStandardSmooth_localizationPreserves.away
    /-
      🎉 no goals
    -/
    /-
      case hQa
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ RingHom.StableUnderCompositionWithLocalizationAway fun {R S} [CommRing R] [C …
    -/
  · exact isStandardSmooth_stableUnderCompositionWithLocalizationAway
    /-
      🎉 no goals
    -/
    /-
      case h
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), Iff (AlgebraicGeome …
    -/
  · intro X Y f
    /-
      case h
      n m : Nat
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (AlgebraicGeometry.IsSmooth f) (∀ (x : ↑↑X.toPresheafedSpace), Exists fu …
    -/
    rw [isSmooth_iff]
    /-
      🎉 no goals
    -/


/-- Being smooth is stable under composition. -/
instance : MorphismProperty.IsStableUnderComposition @IsSmooth :=
  HasRingHomProperty.stableUnderComposition <| locally_stableUnderComposition
    isStandardSmooth_respectsIso isStandardSmooth_localizationPreserves
      isStandardSmooth_stableUnderComposition


/-- The composition of smooth morphisms is smooth. -/
instance isSmooth_comp {Z : Scheme.{u}} (g : Y ⟶ Z) [IsSmooth f] [IsSmooth g] :
    IsSmooth (f ≫ g) :=
  MorphismProperty.comp_mem _ f g ‹IsSmooth f› ‹IsSmooth g›


/-- Smooth of relative dimension `n` is stable under base change. -/
lemma isSmooth_isStableUnderBaseChange : MorphismProperty.IsStableUnderBaseChange @IsSmooth :=
  HasRingHomProperty.isStableUnderBaseChange <| locally_isStableUnderBaseChange
    isStandardSmooth_respectsIso isStandardSmooth_isStableUnderBaseChange


/--
A morphism of schemes `f : X ⟶ Y` is smooth of relative dimension `n` if for each `x : X` there
exists an affine open neighborhood `V` of `x` and an affine open neighborhood `U` of
`f.base x` with `V ≤ f ⁻¹ᵁ U` such that the induced map `Γ(Y, U) ⟶ Γ(X, V)` is
standard smooth of relative dimension `n`.
-/
@[mk_iff]
class IsSmoothOfRelativeDimension : Prop where
  exists_isStandardSmoothOfRelativeDimension : ∀ (x : X), ∃ (U : Y.affineOpens)
    (V : X.affineOpens) (_ : x ∈ V.1) (e : V.1 ≤ f ⁻¹ᵁ U.1),
    IsStandardSmoothOfRelativeDimension.{0, 0} n (f.appLE U V e).hom


/-- If `f` is smooth of any relative dimension, it is smooth. -/
lemma IsSmoothOfRelativeDimension.isSmooth [IsSmoothOfRelativeDimension n f] : IsSmooth f where
  exists_isStandardSmooth x := by
    /-
      n : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsSmoothOfRelativeDimension n f
      x : ↑↑X.toPresheafedSpace
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => (AlgebraicGe …
    -/
    obtain ⟨U, V, hx, e, hf⟩ := exists_isStandardSmoothOfRelativeDimension (n := n) (f := f) x
    /-
      case intro.intro.intro.intro
      n : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsSmoothOfRelativeDimension n f
      x : ↑↑X.toPresheafedSpace
      U : ↑Y.affineOpens
      V : ↑X.affineOpens
      hx : Membership.mem (↑V) x
      e : LE.le (↑V) ((TopologicalSpace.Opens.map f.base).obj ↑U)
      hf : RingHom.IsStandardSmoothOfRelativeDimension n (AlgebraicGeometry.Scheme.H …
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => (AlgebraicGe …
    -/
    exact ⟨U, V, hx, e, hf.isStandardSmooth⟩
    /-
      🎉 no goals
    -/


/-- The property of scheme morphisms `IsSmoothOfRelativeDimension n` is associated with the ring
homomorphism property `Locally (IsStandardSmoothOfRelativeDimension.{0, 0} n)`. -/
instance : HasRingHomProperty (@IsSmoothOfRelativeDimension n)
    (Locally (IsStandardSmoothOfRelativeDimension.{0, 0} n)) := by
  /-
    n m : Nat
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.HasRingHomProperty (@AlgebraicGeometry.IsSmoothOfRelativeD …
  -/
  apply HasRingHomProperty.locally_of_iff
    /-
      case hQl
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => Rin …
    -/
  · exact (isStandardSmoothOfRelativeDimension_localizationPreserves n).away
    /-
      🎉 no goals
    -/
    /-
      case hQa
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ RingHom.StableUnderCompositionWithLocalizationAway fun {R S} [CommRing R] [C …
    -/
  · exact isStandardSmoothOfRelativeDimension_stableUnderCompositionWithLocalizationAway n
    /-
      🎉 no goals
    -/
    /-
      case h
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), Iff (AlgebraicGeome …
    -/
  · intro X Y f
    /-
      case h
      n m : Nat
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (AlgebraicGeometry.IsSmoothOfRelativeDimension n f) (∀ (x : ↑↑X.toPreshe …
    -/
    rw [isSmoothOfRelativeDimension_iff]
    /-
      🎉 no goals
    -/


/-- Smooth of relative dimension `n` is stable under base change. -/
lemma isSmoothOfRelativeDimension_isStableUnderBaseChange :
    MorphismProperty.IsStableUnderBaseChange (@IsSmoothOfRelativeDimension n) :=
  HasRingHomProperty.isStableUnderBaseChange <| locally_isStableUnderBaseChange
    isStandardSmoothOfRelativeDimension_respectsIso
    (isStandardSmoothOfRelativeDimension_isStableUnderBaseChange n)


/-- Open immersions are smooth of relative dimension `0`. -/
instance (priority := 900) [IsOpenImmersion f] : IsSmoothOfRelativeDimension 0 f :=
  HasRingHomProperty.of_isOpenImmersion
    (locally_holdsForLocalizationAway <|
      isStandardSmoothOfRelativeDimension_holdsForLocalizationAway).containsIdentities


/-- Open immersions are smooth. -/
instance (priority := 900) [IsOpenImmersion f] : IsSmooth f :=
  IsSmoothOfRelativeDimension.isSmooth 0 f


/-- If `f` is smooth of relative dimension `n` and `g` is smooth of relative dimension
`m`, then `f ≫ g` is smooth of relative dimension `n + m`. -/
instance isSmoothOfRelativeDimension_comp {Z : Scheme.{u}} (g : Y ⟶ Z)
    [hf : IsSmoothOfRelativeDimension n f] [hg : IsSmoothOfRelativeDimension m g] :
    IsSmoothOfRelativeDimension (n + m) (f ≫ g) where
  exists_isStandardSmoothOfRelativeDimension x := by
    /-
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      Z : AlgebraicGeometry.Scheme
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsSmoothOfRelativeDimension n f
      hg : AlgebraicGeometry.IsSmoothOfRelativeDimension m g
      x : ↑↑X.toPresheafedSpace
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => RingHom.IsSt …
    -/
    obtain ⟨U₂, V₂, hfx₂, e₂, hf₂⟩ := hg.exists_isStandardSmoothOfRelativeDimension (f.base x)
    /-
      case intro.intro.intro.intro
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      Z : AlgebraicGeometry.Scheme
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsSmoothOfRelativeDimension n f
      hg : AlgebraicGeometry.IsSmoothOfRelativeDimension m g
      x : ↑↑X.toPresheafedSpace
      U₂ : ↑Z.affineOpens
      V₂ : ↑Y.affineOpens
      hfx₂ : Membership.mem (↑V₂) (f.base x)
      e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map g.base).obj ↑U₂)
      hf₂ : RingHom.IsStandardSmoothOfRelativeDimension m (AlgebraicGeometry.Scheme. …
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => RingHom.IsSt …
    -/
    obtain ⟨U₁', V₁', hx₁', e₁', hf₁'⟩ := hf.exists_isStandardSmoothOfRelativeDimension x
    obtain ⟨r, s, hx₁, e₁, hf₁⟩ := exists_basicOpen_le_appLE_of_appLE_of_isAffine
      (isStandardSmoothOfRelativeDimension_stableUnderCompositionWithLocalizationAway n).right
      (isStandardSmoothOfRelativeDimension_localizationPreserves n).away
      x V₂ U₁' V₁' V₁' hx₁' hx₁' e₁' hf₁' hfx₂
    have e : X.basicOpen s ≤ (f ≫ g) ⁻¹ᵁ U₂ :=
      le_trans e₁ <| f.preimage_le_preimage_of_le <| le_trans (Y.basicOpen_le r) e₂
    have heq : (f ≫ g).appLE U₂ (X.basicOpen s) e = g.appLE U₂ V₂ e₂ ≫
        CommRingCat.ofHom (algebraMap Γ(Y, V₂) Γ(Y, Y.basicOpen r)) ≫
          f.appLE (Y.basicOpen r) (X.basicOpen s) e₁ := by
      rw [RingHom.algebraMap_toAlgebra, CommRingCat.ofHom_hom,
        g.appLE_map_assoc, Scheme.appLE_comp_appLE]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      Z : AlgebraicGeometry.Scheme
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsSmoothOfRelativeDimension n f
      hg : AlgebraicGeometry.IsSmoothOfRelativeDimension m g
      x : ↑↑X.toPresheafedSpace
      U₂ : ↑Z.affineOpens
      V₂ : ↑Y.affineOpens
      hfx₂ : Membership.mem (↑V₂) (f.base x)
      e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map g.base).obj ↑U₂)
      hf₂ : RingHom.IsStandardSmoothOfRelativeDimension m (AlgebraicGeometry.Scheme. …
      U₁' : ↑Y.affineOpens
      V₁' : ↑X.affineOpens
      hx₁' : Membership.mem (↑V₁') x
      e₁' : LE.le (↑V₁') ((TopologicalSpace.Opens.map f.base).obj ↑U₁')
      hf₁' : RingHom.IsStandardSmoothOfRelativeDimension n (AlgebraicGeometry.Scheme …
      r : ↑(Y.presheaf.obj { unop := ↑V₂ })
      s : ↑(X.presheaf.obj { unop := ↑V₁' })
      hx₁ : Membership.mem (X.basicOpen s) x
      e₁ : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map f.base).obj (Y.basicOp …
      hf₁ : RingHom.IsStandardSmoothOfRelativeDimension n (AlgebraicGeometry.Scheme. …
      e : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map (CategoryTheory.Categor …
      heq : Eq (AlgebraicGeometry.Scheme.Hom.appLE (CategoryTheory.CategoryStruct.co …
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => RingHom.IsSt …
    -/
    refine ⟨U₂, ⟨X.basicOpen s, V₁'.2.basicOpen s⟩, hx₁, e, heq ▸ ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      Z : AlgebraicGeometry.Scheme
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsSmoothOfRelativeDimension n f
      hg : AlgebraicGeometry.IsSmoothOfRelativeDimension m g
      x : ↑↑X.toPresheafedSpace
      U₂ : ↑Z.affineOpens
      V₂ : ↑Y.affineOpens
      hfx₂ : Membership.mem (↑V₂) (f.base x)
      e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map g.base).obj ↑U₂)
      hf₂ : RingHom.IsStandardSmoothOfRelativeDimension m (AlgebraicGeometry.Scheme. …
      U₁' : ↑Y.affineOpens
      V₁' : ↑X.affineOpens
      hx₁' : Membership.mem (↑V₁') x
      e₁' : LE.le (↑V₁') ((TopologicalSpace.Opens.map f.base).obj ↑U₁')
      hf₁' : RingHom.IsStandardSmoothOfRelativeDimension n (AlgebraicGeometry.Scheme …
      r : ↑(Y.presheaf.obj { unop := ↑V₂ })
      s : ↑(X.presheaf.obj { unop := ↑V₁' })
      hx₁ : Membership.mem (X.basicOpen s) x
      e₁ : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map f.base).obj (Y.basicOp …
      hf₁ : RingHom.IsStandardSmoothOfRelativeDimension n (AlgebraicGeometry.Scheme. …
      e : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map (CategoryTheory.Categor …
      heq : Eq (AlgebraicGeometry.Scheme.Hom.appLE (CategoryTheory.CategoryStruct.co …
      ⊢ RingHom.IsStandardSmoothOfRelativeDimension (HAdd.hAdd n m) (CategoryTheory. …
    -/
    apply IsStandardSmoothOfRelativeDimension.comp ?_ hf₂
    /-
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      Z : AlgebraicGeometry.Scheme
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsSmoothOfRelativeDimension n f
      hg : AlgebraicGeometry.IsSmoothOfRelativeDimension m g
      x : ↑↑X.toPresheafedSpace
      U₂ : ↑Z.affineOpens
      V₂ : ↑Y.affineOpens
      hfx₂ : Membership.mem (↑V₂) (f.base x)
      e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map g.base).obj ↑U₂)
      hf₂ : RingHom.IsStandardSmoothOfRelativeDimension m (AlgebraicGeometry.Scheme. …
      U₁' : ↑Y.affineOpens
      V₁' : ↑X.affineOpens
      hx₁' : Membership.mem (↑V₁') x
      e₁' : LE.le (↑V₁') ((TopologicalSpace.Opens.map f.base).obj ↑U₁')
      hf₁' : RingHom.IsStandardSmoothOfRelativeDimension n (AlgebraicGeometry.Scheme …
      r : ↑(Y.presheaf.obj { unop := ↑V₂ })
      s : ↑(X.presheaf.obj { unop := ↑V₁' })
      hx₁ : Membership.mem (X.basicOpen s) x
      e₁ : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map f.base).obj (Y.basicOp …
      hf₁ : RingHom.IsStandardSmoothOfRelativeDimension n (AlgebraicGeometry.Scheme. …
      e : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map (CategoryTheory.Categor …
      heq : Eq (AlgebraicGeometry.Scheme.Hom.appLE (CategoryTheory.CategoryStruct.co …
      ⊢ RingHom.IsStandardSmoothOfRelativeDimension n (CategoryTheory.CategoryStruct …
    -/
    haveI : IsLocalization.Away r Γ(Y, Y.basicOpen r) := V₂.2.isLocalization_basicOpen r
    exact (isStandardSmoothOfRelativeDimension_stableUnderCompositionWithLocalizationAway n).left
      _ r _ hf₁


instance {Z : Scheme.{u}} (g : Y ⟶ Z) [IsSmoothOfRelativeDimension 0 f]
    [IsSmoothOfRelativeDimension 0 g] :
    IsSmoothOfRelativeDimension 0 (f ≫ g) :=
  inferInstanceAs <| IsSmoothOfRelativeDimension (0 + 0) (f ≫ g)


/-- Smooth of relative dimension `0` is multiplicative. -/
instance : MorphismProperty.IsMultiplicative (@IsSmoothOfRelativeDimension 0) where
  id_mem _ := inferInstance
  comp_mem _ _ _ _ := inferInstance


/-- Smooth morphisms are locally of finite presentation. -/
instance (priority := 100) [hf : IsSmooth f] : LocallyOfFinitePresentation f := by
  /-
    n m : Nat
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : AlgebraicGeometry.IsSmooth f
    ⊢ AlgebraicGeometry.LocallyOfFinitePresentation f
  -/
  rw [HasRingHomProperty.eq_affineLocally @LocallyOfFinitePresentation]
  /-
    n m : Nat
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : AlgebraicGeometry.IsSmooth f
    ⊢ AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => Ring …
  -/
  rw [HasRingHomProperty.eq_affineLocally @IsSmooth] at hf
  /-
    n m : Nat
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => R …
    ⊢ AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => Ring …
  -/
  refine affineLocally_le (fun hf ↦ ?_) f hf
  /-
    n m : Nat
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf✝ : AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] =>  …
    R✝ S✝ : Type u
    inst✝¹ : CommRing R✝
    inst✝ : CommRing S✝
    f✝ : RingHom R✝ S✝
    hf : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => RingHom.IsStandar …
    ⊢ f✝.FinitePresentation
  -/
  apply RingHom.locally_of_locally (Q := RingHom.FinitePresentation) at hf
  · rwa [RingHom.locally_iff_of_localizationSpanTarget finitePresentation_respectsIso
      finitePresentation_ofLocalizationSpanTarget] at hf
    /-
      case hPQ
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf✝ : AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] =>  …
      R✝ S✝ : Type u
      inst✝¹ : CommRing R✝
      inst✝ : CommRing S✝
      f✝ : RingHom R✝ S✝
      hf : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => RingHom.IsStandar …
      ⊢ ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] {f : RingHom R S} …
    -/
  · introv hf
    /-
      case hPQ
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f✝¹ : Quiver.Hom X Y
      hf✝¹ : AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => …
      R✝ S✝ : Type u
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      hf✝ : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => RingHom.IsStanda …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : f.IsStandardSmooth
      ⊢ f.FinitePresentation
    -/
    algebraize [f]
    -- TODO: why is `algebraize` not generating the following instance?
    /-
      case hPQ
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f✝¹ : Quiver.Hom X Y
      hf✝¹ : AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => …
      R✝ S✝ : Type u
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      hf✝ : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => RingHom.IsStanda …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : f.IsStandardSmooth
      algInst✝ : Algebra R S := f.toAlgebra
      ⊢ f.FinitePresentation
    -/
    haveI : Algebra.IsStandardSmooth R S := hf
    /-
      case hPQ
      n m : Nat
      X Y : AlgebraicGeometry.Scheme
      f✝¹ : Quiver.Hom X Y
      hf✝¹ : AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => …
      R✝ S✝ : Type u
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      hf✝ : RingHom.Locally (fun {R S} [CommRing R] [CommRing S] => RingHom.IsStanda …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : f.IsStandardSmooth
      algInst✝ : Algebra R S := f.toAlgebra
      this : Algebra.IsStandardSmooth R S
      ⊢ f.FinitePresentation
    -/
    exact this.finitePresentation
    /-
      🎉 no goals
    -/


