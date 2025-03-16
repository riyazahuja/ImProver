/-- The presieve defined by a `P`-cover of `S`-schemes. -/
def Cover.toPresieveOver {X : Over S} (𝒰 : Cover.{u} P X.left) [𝒰.Over S] : Presieve X :=
  Presieve.ofArrows (fun i ↦ (𝒰.obj i).asOver S) (fun i ↦ (𝒰.map i).asOver S)


/-- The presieve defined by a `P`-cover of `S`-schemes with `Q`. -/
def Cover.toPresieveOverProp {X : Q.Over ⊤ S} (𝒰 : Cover.{u} P X.left) [𝒰.Over S]
    (h : ∀ j, Q (𝒰.obj j ↘ S)) : Presieve X :=
  Presieve.ofArrows (fun i ↦ (𝒰.obj i).asOverProp S (h i)) (fun i ↦ (𝒰.map i).asOverProp S)


lemma Cover.overEquiv_generate_toPresieveOver_eq_ofArrows {X : Over S} (𝒰 : Cover.{u} P X.left)
    [𝒰.Over S] : Sieve.overEquiv X (Sieve.generate 𝒰.toPresieveOver) =
      Sieve.ofArrows 𝒰.obj 𝒰.map := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    X : CategoryTheory.Over S
    𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
    inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
    ⊢ Eq ((CategoryTheory.Sieve.overEquiv X) (CategoryTheory.Sieve.generate 𝒰.toPr …
  -/
  ext V f
  /-
    case h
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    X : CategoryTheory.Over S
    𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
    inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
    V : AlgebraicGeometry.Scheme
    f : Quiver.Hom V X.left
    ⊢ Iff (((CategoryTheory.Sieve.overEquiv X) (CategoryTheory.Sieve.generate 𝒰.to …
  -/
  simp only [Sieve.overEquiv_iff, Functor.const_obj_obj, Sieve.generate_apply]
  /-
    case h
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    X : CategoryTheory.Over S
    𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
    inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
    V : AlgebraicGeometry.Scheme
    f : Quiver.Hom V X.left
    ⊢ Iff (Exists fun Y => Exists fun h => Exists fun g => And (𝒰.toPresieveOver g …
  -/
  constructor
    /-
      case h.mp
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      X : CategoryTheory.Over S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      V : AlgebraicGeometry.Scheme
      f : Quiver.Hom V X.left
      ⊢ (Exists fun Y => Exists fun h => Exists fun g => And (𝒰.toPresieveOver g) (E …
    -/
  · rintro ⟨U, h, g, ⟨k⟩, hcomp⟩
    /-
      case h.mp.intro.intro.intro.intro.mk
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      X : CategoryTheory.Over S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      V : AlgebraicGeometry.Scheme
      f : Quiver.Hom V X.left
      Y : CategoryTheory.Over S
      k : 𝒰.J
      h : Quiver.Hom (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp f X …
      hcomp : Eq (CategoryTheory.CategoryStruct.comp h (AlgebraicGeometry.Scheme.Hom …
      ⊢ Exists fun Y => Exists fun h => Exists fun g => And (CategoryTheory.Presieve …
    -/
    exact ⟨𝒰.obj k, h.left, 𝒰.map k, ⟨k⟩, congrArg CommaMorphism.left hcomp⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      X : CategoryTheory.Over S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      V : AlgebraicGeometry.Scheme
      f : Quiver.Hom V X.left
      ⊢ (Exists fun Y => Exists fun h => Exists fun g => And (CategoryTheory.Presiev …
    -/
  · rintro ⟨U, h, g, ⟨k⟩, hcomp⟩
    /-
      case h.mpr.intro.intro.intro.intro.mk
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      X : CategoryTheory.Over S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      V : AlgebraicGeometry.Scheme
      f : Quiver.Hom V X.left
      Y : AlgebraicGeometry.Scheme
      k : 𝒰.J
      h : Quiver.Hom V (𝒰.obj k)
      hcomp : Eq (CategoryTheory.CategoryStruct.comp h (𝒰.map k)) f
      ⊢ Exists fun Y => Exists fun h => Exists fun g => And (𝒰.toPresieveOver g) (Eq …
    -/
    have : 𝒰.map k ≫ X.hom = 𝒰.obj k ↘ S := comp_over (𝒰.map k) S
    /-
      case h.mpr.intro.intro.intro.intro.mk
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      X : CategoryTheory.Over S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      V : AlgebraicGeometry.Scheme
      f : Quiver.Hom V X.left
      Y : AlgebraicGeometry.Scheme
      k : 𝒰.J
      h : Quiver.Hom V (𝒰.obj k)
      hcomp : Eq (CategoryTheory.CategoryStruct.comp h (𝒰.map k)) f
      this : Eq (CategoryTheory.CategoryStruct.comp (𝒰.map k) X.hom) (CategoryTheory …
      ⊢ Exists fun Y => Exists fun h => Exists fun g => And (𝒰.toPresieveOver g) (Eq …
    -/
    refine ⟨(𝒰.obj k).asOver S, Over.homMk h (by simp [← hcomp, this]), (𝒰.map k).asOver S, ⟨k⟩, ?_⟩
    /-
      case h.mpr.intro.intro.intro.intro.mk
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      X : CategoryTheory.Over S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      V : AlgebraicGeometry.Scheme
      f : Quiver.Hom V X.left
      Y : AlgebraicGeometry.Scheme
      k : 𝒰.J
      h : Quiver.Hom V (𝒰.obj k)
      hcomp : Eq (CategoryTheory.CategoryStruct.comp h (𝒰.map k)) f
      this : Eq (CategoryTheory.CategoryStruct.comp (𝒰.map k) X.hom) (CategoryTheory …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Over.homMk h ⋯) (Alge …
    -/
    ext : 1
    /-
      case h.mpr.intro.intro.intro.intro.mk.h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      X : CategoryTheory.Over S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      V : AlgebraicGeometry.Scheme
      f : Quiver.Hom V X.left
      Y : AlgebraicGeometry.Scheme
      k : 𝒰.J
      h : Quiver.Hom V (𝒰.obj k)
      hcomp : Eq (CategoryTheory.CategoryStruct.comp h (𝒰.map k)) f
      this : Eq (CategoryTheory.CategoryStruct.comp (𝒰.map k) X.hom) (CategoryTheory …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Over.homMk h ⋯) (Alge …
    -/
    simpa
    /-
      🎉 no goals
    -/


lemma Cover.toPresieveOver_le_arrows_iff {X : Over S} (R : Sieve X) (𝒰 : Cover.{u} P X.left)
    [𝒰.Over S] :
    𝒰.toPresieveOver ≤ R.arrows ↔
      Presieve.ofArrows 𝒰.obj 𝒰.map ≤ (Sieve.overEquiv X R).arrows := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    X : CategoryTheory.Over S
    R : CategoryTheory.Sieve X
    𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
    inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
    ⊢ Iff (LE.le 𝒰.toPresieveOver R.arrows) (LE.le (CategoryTheory.Presieve.ofArro …
  -/
  simp_rw [← Sieve.giGenerate.gc.le_iff_le, ← Sieve.overEquiv_le_overEquiv_iff]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    X : CategoryTheory.Over S
    R : CategoryTheory.Sieve X
    𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
    inst✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
    ⊢ Iff (LE.le ((CategoryTheory.Sieve.overEquiv X) (CategoryTheory.Sieve.generat …
  -/
  rw [overEquiv_generate_toPresieveOver_eq_ofArrows]
  /-
    🎉 no goals
  -/


/-- The pretopology on `Over S` induced by `P` where coverings are given by `P`-covers
of `S`-schemes. -/
def overPretopology : Pretopology (Over S) where
  coverings Y R := ∃ (𝒰 : Cover.{u} P Y.left) (_ : 𝒰.Over S), R = 𝒰.toPresieveOver
  has_isos {X Y} f _ := ⟨coverOfIsIso f.left, inferInstance, (Presieve.ofArrows_pUnit _).symm⟩
  pullbacks := by
    /-
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      ⊢ ∀ ⦃X Y : CategoryTheory.Over S⦄ (f : Quiver.Hom Y X) (S_1 : CategoryTheory.P …
    -/
    rintro Y X f _ ⟨𝒰, h, rfl⟩
    /-
      case intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      Y X : CategoryTheory.Over S
      f : Quiver.Hom X Y
      𝒰 : AlgebraicGeometry.Scheme.Cover P Y.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      ⊢ Membership.mem ((fun Y R => Exists fun 𝒰 => Exists fun x => Eq R 𝒰.toPresiev …
    -/
    refine ⟨𝒰.pullbackCoverOver' S f.left, inferInstance, ?_⟩
    simpa [Cover.toPresieveOver] using
      (Presieve.ofArrows_pullback f (fun i ↦ (𝒰.obj i).asOver S) (fun i ↦ (𝒰.map i).asOver S)).symm
  transitive := by
    /-
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      ⊢ ∀ ⦃X : CategoryTheory.Over S⦄ (S_1 : CategoryTheory.Presieve X) (Ti : ⦃Y : C …
    -/
    rintro X _ T ⟨𝒰, h, rfl⟩ H
    /-
      case intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      T : ⦃Y : CategoryTheory.Over S⦄ → (f : Quiver.Hom Y X) → 𝒰.toPresieveOver f →  …
      H : ∀ ⦃Y : CategoryTheory.Over S⦄ (f : Quiver.Hom Y X) (H : 𝒰.toPresieveOver f …
      ⊢ Membership.mem ((fun Y R => Exists fun 𝒰 => Exists fun x => Eq R 𝒰.toPresiev …
    -/
    choose V h hV using H
    /-
      case intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      T : ⦃Y : CategoryTheory.Over S⦄ → (f : Quiver.Hom Y X) → 𝒰.toPresieveOver f →  …
      V : ⦃Y : CategoryTheory.Over S⦄ → (f : Quiver.Hom Y X) → 𝒰.toPresieveOver f →  …
      h : ⦃Y : CategoryTheory.Over S⦄ → (f : Quiver.Hom Y X) → (H : 𝒰.toPresieveOver …
      hV : ∀ ⦃Y : CategoryTheory.Over S⦄ (f : Quiver.Hom Y X) (H : 𝒰.toPresieveOver  …
      ⊢ Membership.mem ((fun Y R => Exists fun 𝒰 => Exists fun x => Eq R 𝒰.toPresiev …
    -/
    refine ⟨𝒰.bind (fun j => V ((𝒰.map j).asOver S) ⟨j⟩), inferInstance, ?_⟩
    convert Presieve.ofArrows_bind _ (fun j ↦ (𝒰.map j).asOver S) _
      (fun Y f H j ↦ ((V f H).obj j).asOver S) (fun Y f H j ↦ ((V f H).map j).asOver S)
    /-
      case h.e'_2.h.h.e'_5.h.h.h.h.h
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      T : ⦃Y : CategoryTheory.Over S⦄ → (f : Quiver.Hom Y X) → 𝒰.toPresieveOver f →  …
      V : ⦃Y : CategoryTheory.Over S⦄ → (f : Quiver.Hom Y X) → 𝒰.toPresieveOver f →  …
      h : ⦃Y : CategoryTheory.Over S⦄ → (f : Quiver.Hom Y X) → (H : 𝒰.toPresieveOver …
      hV : ∀ ⦃Y : CategoryTheory.Over S⦄ (f : Quiver.Hom Y X) (H : 𝒰.toPresieveOver  …
      e_1✝ : Eq (CategoryTheory.Presieve X) (CategoryTheory.Presieve (CategoryTheory …
      e_3✝ : Eq X (CategoryTheory.OverClass.asOver X.left S)
      he✝ : Eq 𝒰.toPresieveOver (CategoryTheory.Presieve.ofArrows (fun j => Category …
      x✝² : CategoryTheory.Over S
      x✝¹ : Quiver.Hom x✝² X
      x✝ : 𝒰.toPresieveOver x✝¹
      ⊢ Eq (T x✝¹ x✝) (CategoryTheory.Presieve.ofArrows (fun j => ((V x✝¹ x✝).obj j) …
    -/
    apply hV
    /-
      🎉 no goals
    -/


/-- The topology on `Over S` induced from the topology on `Scheme` defined by `P`.
This agrees with the topology induced by `S.overPretopology P`, see
`AlgebraicGeometry.Scheme.overGrothendieckTopology_eq_toGrothendieck_overPretopology`. -/
abbrev overGrothendieckTopology : GrothendieckTopology (Over S) :=
  (Scheme.grothendieckTopology P).over S


lemma overGrothendieckTopology_eq_toGrothendieck_overPretopology :
    S.overGrothendieckTopology P = (S.overPretopology P).toGrothendieck := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝³ : P.IsMultiplicative
    inst✝² : P.RespectsIso
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    ⊢ Eq (AlgebraicGeometry.Scheme.overGrothendieckTopology P S) (CategoryTheory.P …
  -/
  ext X R
  /-
    case h.h.h
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝³ : P.IsMultiplicative
    inst✝² : P.RespectsIso
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X : CategoryTheory.Over S
    R : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((AlgebraicGeometry.Scheme.overGrothendieckTopology P S) …
  -/
  rw [GrothendieckTopology.mem_over_iff, Pretopology.mem_toGrothendieck]
  /-
    case h.h.h
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝³ : P.IsMultiplicative
    inst✝² : P.RespectsIso
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X : CategoryTheory.Over S
    R : CategoryTheory.Sieve X
    ⊢ Iff (Exists fun R_1 => And (Membership.mem ((AlgebraicGeometry.Scheme.pretop …
  -/
  constructor
    /-
      case h.h.h.mp
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      ⊢ (Exists fun R_1 => And (Membership.mem ((AlgebraicGeometry.Scheme.pretopolog …
    -/
  · rintro ⟨T, ⟨𝒰, rfl⟩, hT⟩
    /-
      case h.h.h.mp.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      hT : LE.le (CategoryTheory.Presieve.ofArrows 𝒰.obj 𝒰.map) ((CategoryTheory.Sie …
      ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck (CategoryTheory.O …
    -/
    letI (i : 𝒰.J) : (𝒰.obj i).Over S := { hom := 𝒰.map i ≫ X.hom }
    letI : 𝒰.Over S :=
      { over := inferInstance
        isOver_map := fun i ↦ ⟨rfl⟩ }
    /-
      case h.h.h.mp.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      hT : LE.le (CategoryTheory.Presieve.ofArrows 𝒰.obj 𝒰.map) ((CategoryTheory.Sie …
      this✝ : (i : 𝒰.J) → (𝒰.obj i).Over S := fun i => { hom := CategoryTheory.Categ …
      this : AlgebraicGeometry.Scheme.Cover.Over S 𝒰 := { over := inferInstance, isO …
      ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck (CategoryTheory.O …
    -/
    use 𝒰.toPresieveOver, ⟨𝒰, inferInstance, rfl⟩
    /-
      case right
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      hT : LE.le (CategoryTheory.Presieve.ofArrows 𝒰.obj 𝒰.map) ((CategoryTheory.Sie …
      this✝ : (i : 𝒰.J) → (𝒰.obj i).Over S := fun i => { hom := CategoryTheory.Categ …
      this : AlgebraicGeometry.Scheme.Cover.Over S 𝒰 := { over := inferInstance, isO …
      ⊢ LE.le 𝒰.toPresieveOver R.arrows
    -/
    rwa [Cover.toPresieveOver_le_arrows_iff]
    /-
      🎉 no goals
    -/
    /-
      case h.h.h.mpr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck (CategoryTheory.O …
    -/
  · rintro ⟨T, ⟨𝒰, h, rfl⟩, hT⟩
    /-
      case h.h.h.mpr.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      hT : LE.le 𝒰.toPresieveOver R.arrows
      ⊢ Exists fun R_1 => And (Membership.mem ((AlgebraicGeometry.Scheme.pretopology …
    -/
    use Presieve.ofArrows 𝒰.obj 𝒰.map, ⟨𝒰, rfl⟩
    /-
      case right
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      hT : LE.le 𝒰.toPresieveOver R.arrows
      ⊢ LE.le (CategoryTheory.Presieve.ofArrows 𝒰.obj 𝒰.map) ((CategoryTheory.Sieve. …
    -/
    rwa [Cover.toPresieveOver_le_arrows_iff] at hT
    /-
      🎉 no goals
    -/


lemma mem_overGrothendieckTopology (X : Over S) (R : Sieve X) :
    R ∈ S.overGrothendieckTopology P X ↔
      ∃ (𝒰 : Cover.{u} P X.left) (_ : 𝒰.Over S), 𝒰.toPresieveOver ≤ R.arrows := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝³ : P.IsMultiplicative
    inst✝² : P.RespectsIso
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X : CategoryTheory.Over S
    R : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((AlgebraicGeometry.Scheme.overGrothendieckTopology P S) …
  -/
  rw [overGrothendieckTopology_eq_toGrothendieck_overPretopology]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝³ : P.IsMultiplicative
    inst✝² : P.RespectsIso
    inst✝¹ : P.IsStableUnderBaseChange
    inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X : CategoryTheory.Over S
    R : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((CategoryTheory.Pretopology.toGrothendieck (CategoryThe …
  -/
  constructor
    /-
      case mp
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck (CategoryTheory.O …
    -/
  · rintro ⟨T, ⟨𝒰, h, rfl⟩, hle⟩
    /-
      case mp.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      hle : LE.le 𝒰.toPresieveOver R.arrows
      ⊢ Exists fun 𝒰 => Exists fun x => LE.le 𝒰.toPresieveOver R.arrows
    -/
    use 𝒰, h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      ⊢ (Exists fun 𝒰 => Exists fun x => LE.le 𝒰.toPresieveOver R.arrows) → Membersh …
    -/
  · rintro ⟨𝒰, h𝒰, hle⟩
    /-
      case mpr.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝³ : P.IsMultiplicative
      inst✝² : P.RespectsIso
      inst✝¹ : P.IsStableUnderBaseChange
      inst✝ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : CategoryTheory.Over S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h𝒰 : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      hle : LE.le 𝒰.toPresieveOver R.arrows
      ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck (CategoryTheory.O …
    -/
    exact ⟨𝒰.toPresieveOver, ⟨𝒰, h𝒰, rfl⟩, hle⟩
    /-
      🎉 no goals
    -/


variable (S) {P Q} in
lemma locallyCoverDense_of_le (hPQ : P ≤ Q) :
    (MorphismProperty.Over.forget Q ⊤ S).LocallyCoverDense (overGrothendieckTopology P S) where
  functorPushforward_functorPullback_mem X := by
    /-
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : Q.IsStableUnderComposition
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      ⊢ ∀ (T : ↑((AlgebraicGeometry.Scheme.overGrothendieckTopology P S) ((CategoryT …
    -/
    intro ⟨T, hT⟩
    /-
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : Q.IsStableUnderComposition
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      T : CategoryTheory.Sieve ((CategoryTheory.MorphismProperty.Over.forget Q Top.t …
      hT : Membership.mem ((AlgebraicGeometry.Scheme.overGrothendieckTopology P S) ( …
      ⊢ Membership.mem ((AlgebraicGeometry.Scheme.overGrothendieckTopology P S) ((Ca …
    -/
    rw [mem_overGrothendieckTopology] at hT ⊢
    /-
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : Q.IsStableUnderComposition
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      T : CategoryTheory.Sieve ((CategoryTheory.MorphismProperty.Over.forget Q Top.t …
      hT✝ : Membership.mem ((AlgebraicGeometry.Scheme.overGrothendieckTopology P S)  …
      hT : Exists fun 𝒰 => Exists fun x => LE.le 𝒰.toPresieveOver T.arrows
      ⊢ Exists fun 𝒰 => Exists fun x => LE.le 𝒰.toPresieveOver (CategoryTheory.Sieve …
    -/
    obtain ⟨𝒰, h, hle⟩ := hT
    /-
      case intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : Q.IsStableUnderComposition
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      T : CategoryTheory.Sieve ((CategoryTheory.MorphismProperty.Over.forget Q Top.t …
      hT : Membership.mem ((AlgebraicGeometry.Scheme.overGrothendieckTopology P S) ( …
      𝒰 : AlgebraicGeometry.Scheme.Cover P ((CategoryTheory.MorphismProperty.Over.fo …
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      hle : LE.le 𝒰.toPresieveOver T.arrows
      ⊢ Exists fun 𝒰 => Exists fun x => LE.le 𝒰.toPresieveOver (CategoryTheory.Sieve …
    -/
    use 𝒰, h
    /-
      case h
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : Q.IsStableUnderComposition
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      T : CategoryTheory.Sieve ((CategoryTheory.MorphismProperty.Over.forget Q Top.t …
      hT : Membership.mem ((AlgebraicGeometry.Scheme.overGrothendieckTopology P S) ( …
      𝒰 : AlgebraicGeometry.Scheme.Cover P ((CategoryTheory.MorphismProperty.Over.fo …
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      hle : LE.le 𝒰.toPresieveOver T.arrows
      ⊢ LE.le 𝒰.toPresieveOver (CategoryTheory.Sieve.functorPushforward (CategoryThe …
    -/
    rintro - - ⟨i⟩
    have p : Q (𝒰.obj i ↘ S) := by
      rw [← comp_over (𝒰.map i) S]
      exact Q.comp_mem _ _ (hPQ _ <| 𝒰.map_prop i) X.prop
    /-
      case h.mk
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : Q.IsStableUnderComposition
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      T : CategoryTheory.Sieve ((CategoryTheory.MorphismProperty.Over.forget Q Top.t …
      hT : Membership.mem ((AlgebraicGeometry.Scheme.overGrothendieckTopology P S) ( …
      𝒰 : AlgebraicGeometry.Scheme.Cover P ((CategoryTheory.MorphismProperty.Over.fo …
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      hle : LE.le 𝒰.toPresieveOver T.arrows
      Y : CategoryTheory.Over S
      i : 𝒰.J
      p : Q (CategoryTheory.over (𝒰.obj i) S inferInstance)
      ⊢ Membership.mem (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Morp …
    -/
    use (𝒰.obj i).asOverProp S p, MorphismProperty.Over.homMk (𝒰.map i) (comp_over (𝒰.map i) S), 𝟙 _
    /-
      case h
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : Q.IsStableUnderComposition
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      T : CategoryTheory.Sieve ((CategoryTheory.MorphismProperty.Over.forget Q Top.t …
      hT : Membership.mem ((AlgebraicGeometry.Scheme.overGrothendieckTopology P S) ( …
      𝒰 : AlgebraicGeometry.Scheme.Cover P ((CategoryTheory.MorphismProperty.Over.fo …
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      hle : LE.le 𝒰.toPresieveOver T.arrows
      Y : CategoryTheory.Over S
      i : 𝒰.J
      p : Q (CategoryTheory.over (𝒰.obj i) S inferInstance)
      ⊢ And ((CategoryTheory.Sieve.functorPullback (CategoryTheory.MorphismProperty. …
    -/
    exact ⟨hle _ ⟨i⟩, rfl⟩
    /-
      🎉 no goals
    -/


instance : (MorphismProperty.Over.forget P ⊤ S).LocallyCoverDense (overGrothendieckTopology P S) :=
  locallyCoverDense_of_le S le_rfl


variable (S) {P Q} in
/-- If `P` and `Q` are morphism properties with `P ≤ Q`, this is the Grothendieck topology
induced via the forgetful functor `Q.Over ⊤ S ⥤ Over S` by the topology defined by `P`. -/
abbrev smallGrothendieckTopologyOfLE (hPQ : P ≤ Q) : GrothendieckTopology (Q.Over ⊤ S) :=
  letI : (MorphismProperty.Over.forget Q ⊤ S).LocallyCoverDense (overGrothendieckTopology P S) :=
    locallyCoverDense_of_le S hPQ
  (MorphismProperty.Over.forget Q ⊤ S).inducedTopology (S.overGrothendieckTopology P)


/-- The Grothendieck topology on the category of schemes over `S` with `P` induced by `P`, i.e.
coverings are simply surjective families. This is the induced topology by the topology on `S`
defined by `P` via the inclusion `P.Over ⊤ S ⥤ Over S`.

This is a special case of `smallGrothendieckTopologyOfLE` for the case `P = Q`. -/
abbrev smallGrothendieckTopology : GrothendieckTopology (P.Over ⊤ S) :=
  (MorphismProperty.Over.forget P ⊤ S).inducedTopology (S.overGrothendieckTopology P)


/-- The pretopology defined on the subcategory of `S`-schemes satisfying `Q` where coverings
are given by `P`-coverings in `S`-schemes satisfying `Q`.
The most common case is `P = Q`. In this case, this is simply surjective families
in `S`-schemes with `P`. -/
def smallPretopology : Pretopology (Q.Over ⊤ S) where
  coverings Y R := ∃ (𝒰 : Cover.{u} P Y.left) (_ : 𝒰.Over S) (h : ∀ j : 𝒰.J, Q (𝒰.obj j ↘ S)),
    R = 𝒰.toPresieveOverProp h
  has_isos {X Y} f := ⟨coverOfIsIso f.left, inferInstance, fun _ ↦ Y.prop,
    (Presieve.ofArrows_pUnit _).symm⟩
  pullbacks := by
    /-
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      ⊢ ∀ ⦃X Y : Q.Over Top.top S⦄ (f : Quiver.Hom Y X) (S_1 : CategoryTheory.Presie …
    -/
    rintro Y X f _ ⟨𝒰, h, p, rfl⟩
    /-
      case intro.intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      Y X : Q.Over Top.top S
      f : Quiver.Hom X Y
      𝒰 : AlgebraicGeometry.Scheme.Cover P Y.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      ⊢ Membership.mem ((fun Y R => Exists fun 𝒰 => Exists fun x => Exists fun h =>  …
    -/
    refine ⟨𝒰.pullbackCoverOverProp' S f.left (Q := Q) Y.prop X.prop p, inferInstance, ?_, ?_⟩
      /-
        case intro.intro.intro.refine_1
        P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        S : AlgebraicGeometry.Scheme
        inst✝⁶ : P.IsMultiplicative
        inst✝⁵ : P.RespectsIso
        inst✝⁴ : P.IsStableUnderBaseChange
        inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
        inst✝² : Q.IsStableUnderComposition
        inst✝¹ : Q.IsStableUnderBaseChange
        inst✝ : Q.HasOfPostcompProperty Q
        Y X : Q.Over Top.top S
        f : Quiver.Hom X Y
        𝒰 : AlgebraicGeometry.Scheme.Cover P Y.left
        h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
        p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
        ⊢ ∀ (j : (AlgebraicGeometry.Scheme.Cover.pullbackCoverOverProp' S 𝒰 f.left ⋯ ⋯ …
      -/
    · intro j
      /-
        case intro.intro.intro.refine_1
        P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        S : AlgebraicGeometry.Scheme
        inst✝⁶ : P.IsMultiplicative
        inst✝⁵ : P.RespectsIso
        inst✝⁴ : P.IsStableUnderBaseChange
        inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
        inst✝² : Q.IsStableUnderComposition
        inst✝¹ : Q.IsStableUnderBaseChange
        inst✝ : Q.HasOfPostcompProperty Q
        Y X : Q.Over Top.top S
        f : Quiver.Hom X Y
        𝒰 : AlgebraicGeometry.Scheme.Cover P Y.left
        h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
        p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
        j : (AlgebraicGeometry.Scheme.Cover.pullbackCoverOverProp' S 𝒰 f.left ⋯ ⋯ p).J
        ⊢ Q (CategoryTheory.over ((AlgebraicGeometry.Scheme.Cover.pullbackCoverOverPro …
      -/
      apply MorphismProperty.Comma.prop
      /-
        🎉 no goals
      -/
    · exact (Presieve.ofArrows_pullback f (fun i ↦ ⟨(𝒰.obj i).asOver S, p i⟩)
        (fun i ↦ ⟨(𝒰.map i).asOver S, trivial, trivial⟩)).symm
  transitive := by
    /-
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      ⊢ ∀ ⦃X : Q.Over Top.top S⦄ (S_1 : CategoryTheory.Presieve X) (Ti : ⦃Y : Q.Over …
    -/
    rintro X _ T ⟨𝒰, h, p, rfl⟩ H
    /-
      case intro.intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      X : Q.Over Top.top S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      T : ⦃Y : Q.Over Top.top S⦄ → (f : Quiver.Hom Y X) → 𝒰.toPresieveOverProp p f → …
      H : ∀ ⦃Y : Q.Over Top.top S⦄ (f : Quiver.Hom Y X) (H : 𝒰.toPresieveOverProp p  …
      ⊢ Membership.mem ((fun Y R => Exists fun 𝒰 => Exists fun x => Exists fun h =>  …
    -/
    choose V h pV hV using H
    let 𝒱j (j : 𝒰.J) : (Cover P ((𝒰.obj j).asOverProp S (p j)).left) :=
      V ((𝒰.map j).asOverProp S) ⟨j⟩
    /-
      case intro.intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      X : Q.Over Top.top S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      T : ⦃Y : Q.Over Top.top S⦄ → (f : Quiver.Hom Y X) → 𝒰.toPresieveOverProp p f → …
      V : ⦃Y : Q.Over Top.top S⦄ → (f : Quiver.Hom Y X) → 𝒰.toPresieveOverProp p f → …
      h : ⦃Y : Q.Over Top.top S⦄ → (f : Quiver.Hom Y X) → (H : 𝒰.toPresieveOverProp  …
      pV : ∀ ⦃Y : Q.Over Top.top S⦄ (f : Quiver.Hom Y X) (H : 𝒰.toPresieveOverProp p …
      hV : ∀ ⦃Y : Q.Over Top.top S⦄ (f : Quiver.Hom Y X) (H : 𝒰.toPresieveOverProp p …
      𝒱j : (j : 𝒰.J) → AlgebraicGeometry.Scheme.Cover P ((𝒰.obj j).asOverProp S ⋯).l …
      ⊢ Membership.mem ((fun Y R => Exists fun 𝒰 => Exists fun x => Exists fun h =>  …
    -/
    refine ⟨𝒰.bind (fun j ↦ 𝒱j j), inferInstance, fun j ↦ pV _ _ _, ?_⟩
    convert Presieve.ofArrows_bind _ (fun j ↦ ((𝒰.map j).asOverProp S)) _
      (fun Y f H j ↦ ((V f H).obj j).asOverProp S (pV _ _ _))
      (fun Y f H j ↦ ((V f H).map j).asOverProp S)
    /-
      case h.e'_2.h.h.e'_5.h.h.h.h.h
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      X : Q.Over Top.top S
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h✝ : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      T : ⦃Y : Q.Over Top.top S⦄ → (f : Quiver.Hom Y X) → 𝒰.toPresieveOverProp p f → …
      V : ⦃Y : Q.Over Top.top S⦄ → (f : Quiver.Hom Y X) → 𝒰.toPresieveOverProp p f → …
      h : ⦃Y : Q.Over Top.top S⦄ → (f : Quiver.Hom Y X) → (H : 𝒰.toPresieveOverProp  …
      pV : ∀ ⦃Y : Q.Over Top.top S⦄ (f : Quiver.Hom Y X) (H : 𝒰.toPresieveOverProp p …
      hV : ∀ ⦃Y : Q.Over Top.top S⦄ (f : Quiver.Hom Y X) (H : 𝒰.toPresieveOverProp p …
      𝒱j : (j : 𝒰.J) → AlgebraicGeometry.Scheme.Cover P ((𝒰.obj j).asOverProp S ⋯).l …
      e_1✝ : Eq (CategoryTheory.Presieve X) (CategoryTheory.Presieve (X.left.asOverP …
      e_3✝ : Eq X (X.left.asOverProp S ⋯)
      he✝ : Eq (𝒰.toPresieveOverProp p) (CategoryTheory.Presieve.ofArrows (fun j =>  …
      x✝² : Q.Over Top.top S
      x✝¹ : Quiver.Hom x✝² X
      x✝ : 𝒰.toPresieveOverProp p x✝¹
      ⊢ Eq (T x✝¹ x✝) (CategoryTheory.Presieve.ofArrows (fun j => ((V x✝¹ x✝).obj j) …
    -/
    apply hV
    /-
      🎉 no goals
    -/


variable (S) {P Q} in
lemma smallGrothendieckTopologyOfLE_eq_toGrothendieck_smallPretopology (hPQ : P ≤ Q) :
    S.smallGrothendieckTopologyOfLE hPQ = (S.smallPretopology P Q).toGrothendieck := by
  /-
    P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝⁶ : P.IsMultiplicative
    inst✝⁵ : P.RespectsIso
    inst✝⁴ : P.IsStableUnderBaseChange
    inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝² : Q.IsStableUnderComposition
    inst✝¹ : Q.IsStableUnderBaseChange
    inst✝ : Q.HasOfPostcompProperty Q
    hPQ : LE.le P Q
    ⊢ Eq (S.smallGrothendieckTopologyOfLE hPQ) (CategoryTheory.Pretopology.toGroth …
  -/
  ext X R
  simp only [Pretopology.mem_toGrothendieck, Functor.mem_inducedTopology_sieves_iff,
    MorphismProperty.Comma.forget_obj, mem_overGrothendieckTopology]
  /-
    case h.h.h
    P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝⁶ : P.IsMultiplicative
    inst✝⁵ : P.RespectsIso
    inst✝⁴ : P.IsStableUnderBaseChange
    inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝² : Q.IsStableUnderComposition
    inst✝¹ : Q.IsStableUnderBaseChange
    inst✝ : Q.HasOfPostcompProperty Q
    hPQ : LE.le P Q
    X : Q.Over Top.top S
    R : CategoryTheory.Sieve X
    ⊢ Iff (Exists fun 𝒰 => Exists fun x => LE.le 𝒰.toPresieveOver (CategoryTheory. …
  -/
  constructor
    /-
      case h.h.h.mp
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      ⊢ (Exists fun 𝒰 => Exists fun x => LE.le 𝒰.toPresieveOver (CategoryTheory.Siev …
    -/
  · intro ⟨𝒰, h, le⟩
    have hj (j : 𝒰.J) : Q (𝒰.obj j ↘ S) := by
      rw [← comp_over (𝒰.map j)]
      exact Q.comp_mem _ _ (hPQ _ <| 𝒰.map_prop _) X.prop
    /-
      case h.h.h.mp
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      le : LE.le 𝒰.toPresieveOver (CategoryTheory.Sieve.functorPushforward (Category …
      hj : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      ⊢ Exists fun R_1 => And (Membership.mem ((AlgebraicGeometry.Scheme.smallPretop …
    -/
    refine ⟨𝒰.toPresieveOverProp hj, ?_, ?_⟩
      /-
        case h.h.h.mp.refine_1
        P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        S : AlgebraicGeometry.Scheme
        inst✝⁶ : P.IsMultiplicative
        inst✝⁵ : P.RespectsIso
        inst✝⁴ : P.IsStableUnderBaseChange
        inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
        inst✝² : Q.IsStableUnderComposition
        inst✝¹ : Q.IsStableUnderBaseChange
        inst✝ : Q.HasOfPostcompProperty Q
        hPQ : LE.le P Q
        X : Q.Over Top.top S
        R : CategoryTheory.Sieve X
        𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
        h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
        le : LE.le 𝒰.toPresieveOver (CategoryTheory.Sieve.functorPushforward (Category …
        hj : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
        ⊢ Membership.mem ((AlgebraicGeometry.Scheme.smallPretopology P Q).coverings X) …
      -/
    · use 𝒰, h, hj
      /-
        🎉 no goals
      -/
      /-
        case h.h.h.mp.refine_2
        P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        S : AlgebraicGeometry.Scheme
        inst✝⁶ : P.IsMultiplicative
        inst✝⁵ : P.RespectsIso
        inst✝⁴ : P.IsStableUnderBaseChange
        inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
        inst✝² : Q.IsStableUnderComposition
        inst✝¹ : Q.IsStableUnderBaseChange
        inst✝ : Q.HasOfPostcompProperty Q
        hPQ : LE.le P Q
        X : Q.Over Top.top S
        R : CategoryTheory.Sieve X
        𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
        h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
        le : LE.le 𝒰.toPresieveOver (CategoryTheory.Sieve.functorPushforward (Category …
        hj : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
        ⊢ LE.le (𝒰.toPresieveOverProp hj) R.arrows
      -/
    · rintro - - ⟨i⟩
      /-
        case h.h.h.mp.refine_2.mk
        P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        S : AlgebraicGeometry.Scheme
        inst✝⁶ : P.IsMultiplicative
        inst✝⁵ : P.RespectsIso
        inst✝⁴ : P.IsStableUnderBaseChange
        inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
        inst✝² : Q.IsStableUnderComposition
        inst✝¹ : Q.IsStableUnderBaseChange
        inst✝ : Q.HasOfPostcompProperty Q
        hPQ : LE.le P Q
        X : Q.Over Top.top S
        R : CategoryTheory.Sieve X
        𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
        h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
        le : LE.le 𝒰.toPresieveOver (CategoryTheory.Sieve.functorPushforward (Category …
        hj : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
        Y : Q.Over Top.top S
        i : 𝒰.J
        ⊢ Membership.mem R.arrows (AlgebraicGeometry.Scheme.Hom.asOverProp (𝒰.map i) S)
      -/
      let fi : (𝒰.obj i).asOverProp S (hj i) ⟶ X := (𝒰.map i).asOverProp S
      /-
        case h.h.h.mp.refine_2.mk
        P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        S : AlgebraicGeometry.Scheme
        inst✝⁶ : P.IsMultiplicative
        inst✝⁵ : P.RespectsIso
        inst✝⁴ : P.IsStableUnderBaseChange
        inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
        inst✝² : Q.IsStableUnderComposition
        inst✝¹ : Q.IsStableUnderBaseChange
        inst✝ : Q.HasOfPostcompProperty Q
        hPQ : LE.le P Q
        X : Q.Over Top.top S
        R : CategoryTheory.Sieve X
        𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
        h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
        le : LE.le 𝒰.toPresieveOver (CategoryTheory.Sieve.functorPushforward (Category …
        hj : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
        Y : Q.Over Top.top S
        i : 𝒰.J
        fi : Quiver.Hom ((𝒰.obj i).asOverProp S ⋯) X := AlgebraicGeometry.Scheme.Hom.a …
        ⊢ Membership.mem R.arrows (AlgebraicGeometry.Scheme.Hom.asOverProp (𝒰.map i) S)
      -/
      have : R.functorPushforward _ ((MorphismProperty.Over.forget Q ⊤ S).map fi) := le _ ⟨i⟩
      rwa [Sieve.functorPushforward_apply,
        Sieve.mem_functorPushforward_iff_of_full_of_faithful] at this
    /-
      case h.h.h.mpr
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      ⊢ (Exists fun R_1 => And (Membership.mem ((AlgebraicGeometry.Scheme.smallPreto …
    -/
  · rintro ⟨T, ⟨𝒰, h, p, rfl⟩, le⟩
    /-
      case h.h.h.mpr.intro.intro.intro.intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      le : LE.le (𝒰.toPresieveOverProp p) R.arrows
      ⊢ Exists fun 𝒰 => Exists fun x => LE.le 𝒰.toPresieveOver (CategoryTheory.Sieve …
    -/
    use 𝒰, h
    /-
      case h
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      le : LE.le (𝒰.toPresieveOverProp p) R.arrows
      ⊢ LE.le 𝒰.toPresieveOver (CategoryTheory.Sieve.functorPushforward (CategoryThe …
    -/
    rintro - - ⟨i⟩
    /-
      case h.mk
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      hPQ : LE.le P Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      le : LE.le (𝒰.toPresieveOverProp p) R.arrows
      Y : CategoryTheory.Over S
      i : 𝒰.J
      ⊢ Membership.mem (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Morp …
    -/
    exact ⟨(𝒰.obj i).asOverProp S (p i), (𝒰.map i).asOverProp S, 𝟙 _, le _ ⟨i⟩, rfl⟩
    /-
      🎉 no goals
    -/


lemma smallGrothendieckTopology_eq_toGrothendieck_smallPretopology [P.HasOfPostcompProperty P] :
    S.smallGrothendieckTopology P = (S.smallPretopology P P).toGrothendieck :=
  S.smallGrothendieckTopologyOfLE_eq_toGrothendieck_smallPretopology le_rfl


lemma mem_toGrothendieck_smallPretopology (X : Q.Over ⊤ S) (R : Sieve X) :
    R ∈ (S.smallPretopology P Q).toGrothendieck _ X ↔
      ∀ x : X.left, ∃ (Y : Q.Over ⊤ S) (f : Y ⟶ X) (y : Y.left),
        R f ∧ P f.left ∧ f.left.base y = x := by
  /-
    P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝⁶ : P.IsMultiplicative
    inst✝⁵ : P.RespectsIso
    inst✝⁴ : P.IsStableUnderBaseChange
    inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝² : Q.IsStableUnderComposition
    inst✝¹ : Q.IsStableUnderBaseChange
    inst✝ : Q.HasOfPostcompProperty Q
    X : Q.Over Top.top S
    R : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((CategoryTheory.Pretopology.toGrothendieck (Q.Over Top. …
  -/
  rw [Pretopology.mem_toGrothendieck]
  /-
    P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝⁶ : P.IsMultiplicative
    inst✝⁵ : P.RespectsIso
    inst✝⁴ : P.IsStableUnderBaseChange
    inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝² : Q.IsStableUnderComposition
    inst✝¹ : Q.IsStableUnderBaseChange
    inst✝ : Q.HasOfPostcompProperty Q
    X : Q.Over Top.top S
    R : CategoryTheory.Sieve X
    ⊢ Iff (Exists fun R_1 => And (Membership.mem ((AlgebraicGeometry.Scheme.smallP …
  -/
  refine ⟨?_, fun h ↦ ?_⟩
    /-
      case refine_1
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      ⊢ (Exists fun R_1 => And (Membership.mem ((AlgebraicGeometry.Scheme.smallPreto …
    -/
  · rintro ⟨T, ⟨𝒰, h, p, rfl⟩, hle⟩
    /-
      case refine_1.intro.intro.intro.intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      hle : LE.le (𝒰.toPresieveOverProp p) R.arrows
      ⊢ ∀ (x : ↑↑X.left.toPresheafedSpace), Exists fun Y => Exists fun f => Exists f …
    -/
    intro x
    /-
      case refine_1.intro.intro.intro.intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      hle : LE.le (𝒰.toPresieveOverProp p) R.arrows
      x : ↑↑X.left.toPresheafedSpace
      ⊢ Exists fun Y => Exists fun f => Exists fun y => And (R.arrows f) (And (P f.l …
    -/
    obtain ⟨y, hy⟩ := 𝒰.covers x
    refine ⟨(𝒰.obj (𝒰.f x)).asOverProp S (p _), (𝒰.map (𝒰.f x)).asOverProp S, y, hle _ ?_,
      𝒰.map_prop _, hy⟩
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), Q (CategoryTheory.over (𝒰.obj j) S inferInstance)
      hle : LE.le (𝒰.toPresieveOverProp p) R.arrows
      x : ↑↑X.left.toPresheafedSpace
      y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace
      hy : Eq ((𝒰.map (𝒰.f x)).base y) x
      ⊢ Membership.mem (𝒰.toPresieveOverProp p) (AlgebraicGeometry.Scheme.Hom.asOver …
    -/
    use 𝒰.f x
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      h : ∀ (x : ↑↑X.left.toPresheafedSpace), Exists fun Y => Exists fun f => Exists …
      ⊢ Exists fun R_1 => And (Membership.mem ((AlgebraicGeometry.Scheme.smallPretop …
    -/
  · choose Y f y hf hP hy using h
    let 𝒰 : X.left.Cover P :=
      { J := X.left,
        obj := fun i ↦ (Y i).left
        map := fun i ↦ (f i).left
        map_prop := hP
        f := id
        covers := fun i ↦ ⟨y i, hy i⟩ }
    letI : 𝒰.Over S :=
      { over := fun i ↦ inferInstance
        isOver_map := fun i ↦ inferInstance }
    /-
      case refine_2
      P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁶ : P.IsMultiplicative
      inst✝⁵ : P.RespectsIso
      inst✝⁴ : P.IsStableUnderBaseChange
      inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝² : Q.IsStableUnderComposition
      inst✝¹ : Q.IsStableUnderBaseChange
      inst✝ : Q.HasOfPostcompProperty Q
      X : Q.Over Top.top S
      R : CategoryTheory.Sieve X
      Y : ↑↑X.left.toPresheafedSpace → Q.Over Top.top S
      f : (x : ↑↑X.left.toPresheafedSpace) → Quiver.Hom (Y x) X
      y : (x : ↑↑X.left.toPresheafedSpace) → ↑↑(Y x).left.toPresheafedSpace
      hf : ∀ (x : ↑↑X.left.toPresheafedSpace), R.arrows (f x)
      hP : ∀ (x : ↑↑X.left.toPresheafedSpace), P (f x).left
      hy : ∀ (x : ↑↑X.left.toPresheafedSpace), Eq ((f x).left.base (y x)) x
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left := { J := ↑↑X.left.toPresheafedSpa …
      this : AlgebraicGeometry.Scheme.Cover.Over S 𝒰 := { over := fun i => inferInst …
      ⊢ Exists fun R_1 => And (Membership.mem ((AlgebraicGeometry.Scheme.smallPretop …
    -/
    refine ⟨𝒰.toPresieveOverProp fun i ↦ MorphismProperty.Comma.prop _, ?_, ?_⟩
      /-
        case refine_2.refine_1
        P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        S : AlgebraicGeometry.Scheme
        inst✝⁶ : P.IsMultiplicative
        inst✝⁵ : P.RespectsIso
        inst✝⁴ : P.IsStableUnderBaseChange
        inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
        inst✝² : Q.IsStableUnderComposition
        inst✝¹ : Q.IsStableUnderBaseChange
        inst✝ : Q.HasOfPostcompProperty Q
        X : Q.Over Top.top S
        R : CategoryTheory.Sieve X
        Y : ↑↑X.left.toPresheafedSpace → Q.Over Top.top S
        f : (x : ↑↑X.left.toPresheafedSpace) → Quiver.Hom (Y x) X
        y : (x : ↑↑X.left.toPresheafedSpace) → ↑↑(Y x).left.toPresheafedSpace
        hf : ∀ (x : ↑↑X.left.toPresheafedSpace), R.arrows (f x)
        hP : ∀ (x : ↑↑X.left.toPresheafedSpace), P (f x).left
        hy : ∀ (x : ↑↑X.left.toPresheafedSpace), Eq ((f x).left.base (y x)) x
        𝒰 : AlgebraicGeometry.Scheme.Cover P X.left := { J := ↑↑X.left.toPresheafedSpa …
        this : AlgebraicGeometry.Scheme.Cover.Over S 𝒰 := { over := fun i => inferInst …
        ⊢ Membership.mem ((AlgebraicGeometry.Scheme.smallPretopology P Q).coverings X) …
      -/
    · use 𝒰, inferInstance, fun i ↦ MorphismProperty.Comma.prop _
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        S : AlgebraicGeometry.Scheme
        inst✝⁶ : P.IsMultiplicative
        inst✝⁵ : P.RespectsIso
        inst✝⁴ : P.IsStableUnderBaseChange
        inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
        inst✝² : Q.IsStableUnderComposition
        inst✝¹ : Q.IsStableUnderBaseChange
        inst✝ : Q.HasOfPostcompProperty Q
        X : Q.Over Top.top S
        R : CategoryTheory.Sieve X
        Y : ↑↑X.left.toPresheafedSpace → Q.Over Top.top S
        f : (x : ↑↑X.left.toPresheafedSpace) → Quiver.Hom (Y x) X
        y : (x : ↑↑X.left.toPresheafedSpace) → ↑↑(Y x).left.toPresheafedSpace
        hf : ∀ (x : ↑↑X.left.toPresheafedSpace), R.arrows (f x)
        hP : ∀ (x : ↑↑X.left.toPresheafedSpace), P (f x).left
        hy : ∀ (x : ↑↑X.left.toPresheafedSpace), Eq ((f x).left.base (y x)) x
        𝒰 : AlgebraicGeometry.Scheme.Cover P X.left := { J := ↑↑X.left.toPresheafedSpa …
        this : AlgebraicGeometry.Scheme.Cover.Over S 𝒰 := { over := fun i => inferInst …
        ⊢ LE.le (𝒰.toPresieveOverProp ⋯) R.arrows
      -/
    · rintro - - ⟨i⟩
      /-
        case refine_2.refine_2.mk
        P Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        S : AlgebraicGeometry.Scheme
        inst✝⁶ : P.IsMultiplicative
        inst✝⁵ : P.RespectsIso
        inst✝⁴ : P.IsStableUnderBaseChange
        inst✝³ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
        inst✝² : Q.IsStableUnderComposition
        inst✝¹ : Q.IsStableUnderBaseChange
        inst✝ : Q.HasOfPostcompProperty Q
        X : Q.Over Top.top S
        R : CategoryTheory.Sieve X
        Y✝ : ↑↑X.left.toPresheafedSpace → Q.Over Top.top S
        f : (x : ↑↑X.left.toPresheafedSpace) → Quiver.Hom (Y✝ x) X
        y : (x : ↑↑X.left.toPresheafedSpace) → ↑↑(Y✝ x).left.toPresheafedSpace
        hf : ∀ (x : ↑↑X.left.toPresheafedSpace), R.arrows (f x)
        hP : ∀ (x : ↑↑X.left.toPresheafedSpace), P (f x).left
        hy : ∀ (x : ↑↑X.left.toPresheafedSpace), Eq ((f x).left.base (y x)) x
        𝒰 : AlgebraicGeometry.Scheme.Cover P X.left := { J := ↑↑X.left.toPresheafedSpa …
        this : AlgebraicGeometry.Scheme.Cover.Over S 𝒰 := { over := fun i => inferInst …
        Y : Q.Over Top.top S
        i : 𝒰.J
        ⊢ Membership.mem R.arrows (AlgebraicGeometry.Scheme.Hom.asOverProp (𝒰.map i) S)
      -/
      exact hf i
      /-
        🎉 no goals
      -/


lemma mem_smallGrothendieckTopology [P.HasOfPostcompProperty P] (X : P.Over ⊤ S) (R : Sieve X) :
    R ∈ S.smallGrothendieckTopology P X ↔
      ∃ (𝒰 : Cover.{u} P X.left) (_ : 𝒰.Over S) (h : ∀ j, P (𝒰.obj j ↘ S)),
          𝒰.toPresieveOverProp h ≤ R.arrows := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : P.HasOfPostcompProperty P
    X : P.Over Top.top S
    R : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((AlgebraicGeometry.Scheme.smallGrothendieckTopology P)  …
  -/
  rw [smallGrothendieckTopology_eq_toGrothendieck_smallPretopology]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    S : AlgebraicGeometry.Scheme
    inst✝⁴ : P.IsMultiplicative
    inst✝³ : P.RespectsIso
    inst✝² : P.IsStableUnderBaseChange
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    inst✝ : P.HasOfPostcompProperty P
    X : P.Over Top.top S
    R : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((CategoryTheory.Pretopology.toGrothendieck (P.Over Top. …
  -/
  constructor
    /-
      case mp
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : P.HasOfPostcompProperty P
      X : P.Over Top.top S
      R : CategoryTheory.Sieve X
      ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck (P.Over Top.top S …
    -/
  · rintro ⟨T, ⟨𝒰, h, p, rfl⟩, hle⟩
    /-
      case mp.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : P.HasOfPostcompProperty P
      X : P.Over Top.top S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), P (CategoryTheory.over (𝒰.obj j) S inferInstance)
      hle : LE.le (𝒰.toPresieveOverProp p) R.arrows
      ⊢ Exists fun 𝒰 => Exists fun x => Exists fun h => LE.le (𝒰.toPresieveOverProp  …
    -/
    use 𝒰, h, p
    /-
      🎉 no goals
    -/
    /-
      case mpr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : P.HasOfPostcompProperty P
      X : P.Over Top.top S
      R : CategoryTheory.Sieve X
      ⊢ (Exists fun 𝒰 => Exists fun x => Exists fun h => LE.le (𝒰.toPresieveOverProp …
    -/
  · rintro ⟨𝒰, h𝒰, p, hle⟩
    /-
      case mpr.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      S : AlgebraicGeometry.Scheme
      inst✝⁴ : P.IsMultiplicative
      inst✝³ : P.RespectsIso
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      inst✝ : P.HasOfPostcompProperty P
      X : P.Over Top.top S
      R : CategoryTheory.Sieve X
      𝒰 : AlgebraicGeometry.Scheme.Cover P X.left
      h𝒰 : AlgebraicGeometry.Scheme.Cover.Over S 𝒰
      p : ∀ (j : 𝒰.J), P (CategoryTheory.over (𝒰.obj j) S inferInstance)
      hle : LE.le (𝒰.toPresieveOverProp p) R.arrows
      ⊢ Membership.mem ((CategoryTheory.Pretopology.toGrothendieck (P.Over Top.top S …
    -/
    exact ⟨𝒰.toPresieveOverProp p, ⟨𝒰, h𝒰, p, rfl⟩, hle⟩
    /-
      🎉 no goals
    -/


