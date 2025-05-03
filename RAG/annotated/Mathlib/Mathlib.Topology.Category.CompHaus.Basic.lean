/-- The category of compact Hausdorff spaces. -/
abbrev CompHaus := CompHausLike (fun _ ↦ True)


instance : Inhabited CompHaus :=
  ⟨{ toTop := { α := PEmpty }, prop := trivial}⟩


instance : CoeSort CompHaus Type* :=
  ⟨fun X => X.toTop⟩


instance {X : CompHaus} : CompactSpace X :=
  X.is_compact


instance {X : CompHaus} : T2Space X :=
  X.is_hausdorff


instance : HasProp (fun _ ↦ True) X := ⟨trivial⟩


/-- A constructor for objects of the category `CompHaus`,
taking a type, and bundling the compact Hausdorff topology
found by typeclass inference. -/
abbrev of : CompHaus := CompHausLike.of _ X


/-- The fully faithful embedding of `CompHaus` in `TopCat`. -/
-- Porting note: `semireducible` -> `.default`.
abbrev compHausToTop : CompHaus.{u} ⥤ TopCat.{u} :=
  CompHausLike.compHausLikeToTop _
  -- deriving Full, Faithful -- Porting note: deriving fails, adding manually.


/-- (Implementation) The object part of the compactification functor from topological spaces to
compact Hausdorff spaces.
-/
@[simps!]
def stoneCechObj (X : TopCat) : CompHaus :=
  CompHaus.of (StoneCech X)


/-- (Implementation) The bijection of homsets to establish the reflective adjunction of compact
Hausdorff spaces in topological spaces.
-/
noncomputable def stoneCechEquivalence (X : TopCat.{u}) (Y : CompHaus.{u}) :
    (stoneCechObj X ⟶ Y) ≃ (X ⟶ compHausToTop.obj Y) where
  toFun f :=
    { toFun := f ∘ stoneCechUnit
      continuous_toFun := f.2.comp (@continuous_stoneCechUnit X _) }
  invFun f :=
    { toFun := stoneCechExtend f.2
      continuous_toFun := continuous_stoneCechExtend f.2 }
  left_inv := by
    /-
      X : TopCat
      Y : CompHaus
      ⊢ Function.LeftInverse (fun f => { toFun := stoneCechExtend ⋯, continuous_toFu …
    -/
    rintro ⟨f : StoneCech X ⟶ Y, hf : Continuous f⟩
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` fails.
    /-
      case mk
      X : TopCat
      Y : CompHaus
      f : Quiver.Hom (StoneCech ↑X) ↑Y.toTop
      hf : Continuous f
      ⊢ Eq ((fun f => { toFun := stoneCechExtend ⋯, continuous_toFun := ⋯ }) ((fun f …
    -/
    apply ContinuousMap.ext
    /-
      case mk.h
      X : TopCat
      Y : CompHaus
      f : Quiver.Hom (StoneCech ↑X) ↑Y.toTop
      hf : Continuous f
      ⊢ ∀ (a : ↑(stoneCechObj X).toTop), Eq (((fun f => { toFun := stoneCechExtend ⋯ …
    -/
    intro (x : StoneCech X)
    /-
      case mk.h
      X : TopCat
      Y : CompHaus
      f : Quiver.Hom (StoneCech ↑X) ↑Y.toTop
      hf : Continuous f
      x : StoneCech ↑X
      ⊢ Eq (((fun f => { toFun := stoneCechExtend ⋯, continuous_toFun := ⋯ }) ((fun  …
    -/
    refine congr_fun ?_ x
    /-
      case mk.h
      X : TopCat
      Y : CompHaus
      f : Quiver.Hom (StoneCech ↑X) ↑Y.toTop
      hf : Continuous f
      x : StoneCech ↑X
      ⊢ Eq ⇑((fun f => { toFun := stoneCechExtend ⋯, continuous_toFun := ⋯ }) ((fun  …
    -/
    apply Continuous.ext_on denseRange_stoneCechUnit (continuous_stoneCechExtend _) hf
      /-
        case mk.h
        X : TopCat
        Y : CompHaus
        f : Quiver.Hom (StoneCech ↑X) ↑Y.toTop
        hf : Continuous f
        x : StoneCech ↑X
        ⊢ Set.EqOn (stoneCechExtend ⋯) f (Set.range stoneCechUnit)
      -/
    · rintro _ ⟨y, rfl⟩
      /-
        case mk.h.intro
        X : TopCat
        Y : CompHaus
        f : Quiver.Hom (StoneCech ↑X) ↑Y.toTop
        hf : Continuous f
        x : StoneCech ↑X
        y : ↑X
        ⊢ Eq (stoneCechExtend ⋯ (stoneCechUnit y)) (f (stoneCechUnit y))
      -/
      apply congr_fun (stoneCechExtend_extends (hf.comp _)) y
      /-
        X : TopCat
        Y : CompHaus
        f : Quiver.Hom (StoneCech ↑X) ↑Y.toTop
        hf : Continuous f
        x : StoneCech ↑X
        y : ↑X
        ⊢ Continuous stoneCechUnit
      -/
      apply continuous_stoneCechUnit
      /-
        🎉 no goals
      -/
  right_inv := by
    /-
      X : TopCat
      Y : CompHaus
      ⊢ Function.RightInverse (fun f => { toFun := stoneCechExtend ⋯, continuous_toF …
    -/
    rintro ⟨f : (X : Type _) ⟶ Y, hf : Continuous f⟩
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` fails.
    /-
      case mk
      X : TopCat
      Y : CompHaus
      f : Quiver.Hom ↑X ↑Y.toTop
      hf : Continuous f
      ⊢ Eq ((fun f => { toFun := Function.comp (⇑f) stoneCechUnit, continuous_toFun  …
    -/
    apply ContinuousMap.ext
    /-
      case mk.h
      X : TopCat
      Y : CompHaus
      f : Quiver.Hom ↑X ↑Y.toTop
      hf : Continuous f
      ⊢ ∀ (a : ↑X), Eq (((fun f => { toFun := Function.comp (⇑f) stoneCechUnit, cont …
    -/
    intro
    /-
      case mk.h
      X : TopCat
      Y : CompHaus
      f : Quiver.Hom ↑X ↑Y.toTop
      hf : Continuous f
      a✝ : ↑X
      ⊢ Eq (((fun f => { toFun := Function.comp (⇑f) stoneCechUnit, continuous_toFun …
    -/
    exact congr_fun (stoneCechExtend_extends hf) _
    /-
      🎉 no goals
    -/


/-- The Stone-Cech compactification functor from topological spaces to compact Hausdorff spaces,
left adjoint to the inclusion functor.
-/
noncomputable def topToCompHaus : TopCat.{u} ⥤ CompHaus.{u} :=
  Adjunction.leftAdjointOfEquiv stoneCechEquivalence.{u} fun _ _ _ _ _ => rfl


theorem topToCompHaus_obj (X : TopCat) : ↥(topToCompHaus.obj X) = StoneCech X :=
  rfl


/-- The category of compact Hausdorff spaces is reflective in the category of topological spaces.
-/
noncomputable instance compHausToTop.reflective : Reflective compHausToTop where
  L := topToCompHaus
  adj := Adjunction.adjunctionOfEquivLeft _ _


noncomputable instance compHausToTop.createsLimits : CreatesLimits compHausToTop :=
  monadicCreatesLimits _


instance CompHaus.hasLimits : Limits.HasLimits CompHaus :=
  hasLimits_of_hasLimits_createsLimits compHausToTop


instance CompHaus.hasColimits : Limits.HasColimits CompHaus :=
  hasColimits_of_reflective compHausToTop


/-- An explicit limit cone for a functor `F : J ⥤ CompHaus`, defined in terms of
`TopCat.limitCone`. -/
def limitCone {J : Type v} [SmallCategory J] (F : J ⥤ CompHaus.{max v u}) : Limits.Cone F :=
  letI FF : J ⥤ TopCat := F ⋙ compHausToTop
  { pt := {
      toTop := (TopCat.limitCone FF).pt
      is_compact := by
        /-
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          ⊢ CompactSpace ↑(TopCat.limitCone FF).pt
        -/
        show CompactSpace { u : ∀ j, F.obj j | ∀ {i j : J} (f : i ⟶ j), (F.map f) (u i) = u j }
        /-
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          ⊢ CompactSpace ↑(setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f …
        -/
        rw [← isCompact_iff_compactSpace]
        /-
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          ⊢ IsCompact (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u …
        -/
        apply IsClosed.isCompact
        have :
          { u : ∀ j, F.obj j | ∀ {i j : J} (f : i ⟶ j), F.map f (u i) = u j } =
            ⋂ (i : J) (j : J) (f : i ⟶ j), { u | F.map f (u i) = u j } := by
          ext1
          simp only [Set.mem_iInter, Set.mem_setOf_eq]
        /-
          case h
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          this : Eq (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u i …
          ⊢ IsClosed (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u  …
        -/
        rw [this]
        /-
          case h
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          this : Eq (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u i …
          ⊢ IsClosed (Set.iInter fun i => Set.iInter fun j => Set.iInter fun f => setOf  …
        -/
        apply isClosed_iInter
        /-
          case h.h
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          this : Eq (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u i …
          ⊢ ∀ (i : J), IsClosed (Set.iInter fun j => Set.iInter fun f => setOf fun u =>  …
        -/
        intro i
        /-
          case h.h
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          this : Eq (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u i …
          i : J
          ⊢ IsClosed (Set.iInter fun j => Set.iInter fun f => setOf fun u => Eq ((F.map  …
        -/
        apply isClosed_iInter
        /-
          case h.h.h
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          this : Eq (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u i …
          i : J
          ⊢ ∀ (i_1 : J), IsClosed (Set.iInter fun f => setOf fun u => Eq ((F.map f) (u i …
        -/
        intro j
        /-
          case h.h.h
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          this : Eq (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u i …
          i j : J
          ⊢ IsClosed (Set.iInter fun f => setOf fun u => Eq ((F.map f) (u i)) (u j))
        -/
        apply isClosed_iInter
        /-
          case h.h.h.h
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          this : Eq (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u i …
          i j : J
          ⊢ ∀ (i_1 : Quiver.Hom i j), IsClosed (setOf fun u => Eq ((F.map i_1) (u i)) (u …
        -/
        intro f
        /-
          case h.h.h.h
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          this : Eq (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u i …
          i j : J
          f : Quiver.Hom i j
          ⊢ IsClosed (setOf fun u => Eq ((F.map f) (u i)) (u j))
        -/
        apply isClosed_eq
          /-
            case h.h.h.h.hf
            J : Type v
            inst✝ : CategoryTheory.SmallCategory J
            F : CategoryTheory.Functor J CompHaus
            FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
            this : Eq (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u i …
            i j : J
            f : Quiver.Hom i j
            ⊢ Continuous fun y => (F.map f) (y i)
          -/
        · exact (ContinuousMap.continuous (F.map f)).comp (continuous_apply i)
          /-
            🎉 no goals
          -/
          /-
            case h.h.h.h.hg
            J : Type v
            inst✝ : CategoryTheory.SmallCategory J
            F : CategoryTheory.Functor J CompHaus
            FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
            this : Eq (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F.map f) (u i …
            i j : J
            f : Quiver.Hom i j
            ⊢ Continuous fun y => y j
          -/
        · exact continuous_apply j
          /-
            🎉 no goals
          -/
      is_hausdorff :=
        show T2Space { u : ∀ j, F.obj j | ∀ {i j : J} (f : i ⟶ j), (F.map f) (u i) = u j } from
          inferInstance
      prop := trivial }
    π := {
      app := fun j => (TopCat.limitCone FF).π.app j
      naturality := by
        /-
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          ⊢ ∀ ⦃X Y : J⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
        -/
        intro _ _ f
        /-
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          X✝ Y✝ : J
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        ext ⟨x, hx⟩
        /-
          case w.mk
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          X✝ Y✝ : J
          f : Quiver.Hom X✝ Y✝
          x : (j : J) → ↑(FF.obj j)
          hx : Membership.mem (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((FF. …
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).o …
        -/
        simp only [comp_apply, Functor.const_obj_map, id_apply]
        /-
          case w.mk
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J CompHaus
          FF : CategoryTheory.Functor J TopCat := F.comp compHausToTop
          X✝ Y✝ : J
          f : Quiver.Hom X✝ Y✝
          x : (j : J) → ↑(FF.obj j)
          hx : Membership.mem (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((FF. …
          ⊢ Eq (((TopCat.limitCone FF).π.app Y✝) ((CategoryTheory.CategoryStruct.id (Com …
        -/
        exact (hx f).symm } }
        /-
          🎉 no goals
        -/


/-- The limit cone `CompHaus.limitCone F` is indeed a limit cone. -/
def limitConeIsLimit {J : Type v} [SmallCategory J] (F : J ⥤ CompHaus.{max v u}) :
    Limits.IsLimit.{v} (limitCone.{v,u} F) :=
  letI FF : J ⥤ TopCat := F ⋙ compHausToTop
  { lift := fun S => (TopCat.limitConeIsLimit FF).lift (compHausToTop.mapCone S)
    fac := fun S => (TopCat.limitConeIsLimit FF).fac (compHausToTop.mapCone S)
    uniq := fun S => (TopCat.limitConeIsLimit FF).uniq (compHausToTop.mapCone S) }


theorem epi_iff_surjective {X Y : CompHaus.{u}} (f : X ⟶ Y) : Epi f ↔ Function.Surjective f := by
  /-
    X Y : CompHaus
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (Function.Surjective ⇑f)
  -/
  constructor
    /-
      case mp
      X Y : CompHaus
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Epi f → Function.Surjective ⇑f
    -/
  · dsimp [Function.Surjective]
    /-
      case mp
      X Y : CompHaus
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Epi f → ∀ (b : (CategoryTheory.forget CompHaus).obj Y), Exist …
    -/
    contrapose!
    /-
      case mp
      X Y : CompHaus
      f : Quiver.Hom X Y
      ⊢ (Exists fun b => ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) b) …
    -/
    rintro ⟨y, hy⟩ hf
    /-
      case mp.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      ⊢ False
    -/
    let C := Set.range f
    /-
      case mp.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      ⊢ False
    -/
    have hC : IsClosed C := (isCompact_range f.continuous).isClosed
    /-
      case mp.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      ⊢ False
    -/
    let D := ({y} : Set Y)
    /-
      case mp.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      ⊢ False
    -/
    have hD : IsClosed D := isClosed_singleton
    have hCD : Disjoint C D := by
      rw [Set.disjoint_singleton_right]
      rintro ⟨y', hy'⟩
      exact hy y' hy'
    /-
      case mp.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      hD : IsClosed D
      hCD : Disjoint C D
      ⊢ False
    -/
    obtain ⟨φ, hφ0, hφ1, hφ01⟩ := exists_continuous_zero_one_of_isClosed hC hD hCD
    /-
      case mp.intro.intro.intro.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      hD : IsClosed D
      hCD : Disjoint C D
      φ : ContinuousMap ((CategoryTheory.forget CompHaus).obj Y) Real
      hφ0 : Set.EqOn (⇑φ) 0 C
      hφ1 : Set.EqOn (⇑φ) 1 D
      hφ01 : ∀ (x : (CategoryTheory.forget CompHaus).obj Y), Membership.mem (Set.Icc …
      ⊢ False
    -/
    haveI : CompactSpace (ULift.{u} <| Set.Icc (0 : ℝ) 1) := Homeomorph.ulift.symm.compactSpace
    /-
      case mp.intro.intro.intro.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      hD : IsClosed D
      hCD : Disjoint C D
      φ : ContinuousMap ((CategoryTheory.forget CompHaus).obj Y) Real
      hφ0 : Set.EqOn (⇑φ) 0 C
      hφ1 : Set.EqOn (⇑φ) 1 D
      hφ01 : ∀ (x : (CategoryTheory.forget CompHaus).obj Y), Membership.mem (Set.Icc …
      this : CompactSpace (ULift.{u, 0} ↑(Set.Icc 0 1))
      ⊢ False
    -/
    haveI : T2Space (ULift.{u} <| Set.Icc (0 : ℝ) 1) := Homeomorph.ulift.symm.t2Space
    /-
      case mp.intro.intro.intro.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      hD : IsClosed D
      hCD : Disjoint C D
      φ : ContinuousMap ((CategoryTheory.forget CompHaus).obj Y) Real
      hφ0 : Set.EqOn (⇑φ) 0 C
      hφ1 : Set.EqOn (⇑φ) 1 D
      hφ01 : ∀ (x : (CategoryTheory.forget CompHaus).obj Y), Membership.mem (Set.Icc …
      this✝ : CompactSpace (ULift.{u, 0} ↑(Set.Icc 0 1))
      this : T2Space (ULift.{u, 0} ↑(Set.Icc 0 1))
      ⊢ False
    -/
    let Z := of (ULift.{u} <| Set.Icc (0 : ℝ) 1)
    let g : Y ⟶ Z :=
      ⟨fun y' => ⟨⟨φ y', hφ01 y'⟩⟩,
        continuous_uLift_up.comp (φ.continuous.subtype_mk fun y' => hφ01 y')⟩
    /-
      case mp.intro.intro.intro.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      hD : IsClosed D
      hCD : Disjoint C D
      φ : ContinuousMap ((CategoryTheory.forget CompHaus).obj Y) Real
      hφ0 : Set.EqOn (⇑φ) 0 C
      hφ1 : Set.EqOn (⇑φ) 1 D
      hφ01 : ∀ (x : (CategoryTheory.forget CompHaus).obj Y), Membership.mem (Set.Icc …
      this✝ : CompactSpace (ULift.{u, 0} ↑(Set.Icc 0 1))
      this : T2Space (ULift.{u, 0} ↑(Set.Icc 0 1))
      Z : CompHaus := CompHaus.of (ULift.{u, 0} ↑(Set.Icc 0 1))
      g : Quiver.Hom Y Z := { toFun := fun y' => { down := ⟨φ y', ⋯⟩ }, continuous_t …
      ⊢ False
    -/
    let h : Y ⟶ Z := ⟨fun _ => ⟨⟨0, Set.left_mem_Icc.mpr zero_le_one⟩⟩, continuous_const⟩
    have H : h = g := by
      rw [← cancel_epi f]
      ext x
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` doesn't apply these two lemmas.
      apply ULift.ext
      apply Subtype.ext
      dsimp
      -- Porting note: This `change` is not ideal.
      -- I think lean is having issues understanding when a `ContinuousMap` should be considered
      -- as a morphism.
      -- TODO(?): Make morphisms in `CompHaus` (and other topological categories)
      -- into a one-field-structure.
      change 0 = φ (f x)
      simp only [hφ0 (Set.mem_range_self x), Pi.zero_apply]
    /-
      case mp.intro.intro.intro.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      hD : IsClosed D
      hCD : Disjoint C D
      φ : ContinuousMap ((CategoryTheory.forget CompHaus).obj Y) Real
      hφ0 : Set.EqOn (⇑φ) 0 C
      hφ1 : Set.EqOn (⇑φ) 1 D
      hφ01 : ∀ (x : (CategoryTheory.forget CompHaus).obj Y), Membership.mem (Set.Icc …
      this✝ : CompactSpace (ULift.{u, 0} ↑(Set.Icc 0 1))
      this : T2Space (ULift.{u, 0} ↑(Set.Icc 0 1))
      Z : CompHaus := CompHaus.of (ULift.{u, 0} ↑(Set.Icc 0 1))
      g : Quiver.Hom Y Z := { toFun := fun y' => { down := ⟨φ y', ⋯⟩ }, continuous_t …
      h : Quiver.Hom Y Z := { toFun := fun x => { down := ⟨0, ⋯⟩ }, continuous_toFun …
      H : Eq h g
      ⊢ False
    -/
    apply_fun fun e => (e y).down.1 at H
    /-
      case mp.intro.intro.intro.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      hD : IsClosed D
      hCD : Disjoint C D
      φ : ContinuousMap ((CategoryTheory.forget CompHaus).obj Y) Real
      hφ0 : Set.EqOn (⇑φ) 0 C
      hφ1 : Set.EqOn (⇑φ) 1 D
      hφ01 : ∀ (x : (CategoryTheory.forget CompHaus).obj Y), Membership.mem (Set.Icc …
      this✝ : CompactSpace (ULift.{u, 0} ↑(Set.Icc 0 1))
      this : T2Space (ULift.{u, 0} ↑(Set.Icc 0 1))
      Z : CompHaus := CompHaus.of (ULift.{u, 0} ↑(Set.Icc 0 1))
      g : Quiver.Hom Y Z := { toFun := fun y' => { down := ⟨φ y', ⋯⟩ }, continuous_t …
      h : Quiver.Hom Y Z := { toFun := fun x => { down := ⟨0, ⋯⟩ }, continuous_toFun …
      H : Eq ↑(h y).down ↑(g y).down
      ⊢ False
    -/
    dsimp [Z] at H
    /-
      case mp.intro.intro.intro.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      hD : IsClosed D
      hCD : Disjoint C D
      φ : ContinuousMap ((CategoryTheory.forget CompHaus).obj Y) Real
      hφ0 : Set.EqOn (⇑φ) 0 C
      hφ1 : Set.EqOn (⇑φ) 1 D
      hφ01 : ∀ (x : (CategoryTheory.forget CompHaus).obj Y), Membership.mem (Set.Icc …
      this✝ : CompactSpace (ULift.{u, 0} ↑(Set.Icc 0 1))
      this : T2Space (ULift.{u, 0} ↑(Set.Icc 0 1))
      Z : CompHaus := CompHaus.of (ULift.{u, 0} ↑(Set.Icc 0 1))
      g : Quiver.Hom Y Z := { toFun := fun y' => { down := ⟨φ y', ⋯⟩ }, continuous_t …
      h : Quiver.Hom Y Z := { toFun := fun x => { down := ⟨0, ⋯⟩ }, continuous_toFun …
      H : Eq ↑(h y).down ↑(g y).down
      ⊢ False
    -/
    change 0 = φ y at H
    /-
      case mp.intro.intro.intro.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      hD : IsClosed D
      hCD : Disjoint C D
      φ : ContinuousMap ((CategoryTheory.forget CompHaus).obj Y) Real
      hφ0 : Set.EqOn (⇑φ) 0 C
      hφ1 : Set.EqOn (⇑φ) 1 D
      hφ01 : ∀ (x : (CategoryTheory.forget CompHaus).obj Y), Membership.mem (Set.Icc …
      this✝ : CompactSpace (ULift.{u, 0} ↑(Set.Icc 0 1))
      this : T2Space (ULift.{u, 0} ↑(Set.Icc 0 1))
      Z : CompHaus := CompHaus.of (ULift.{u, 0} ↑(Set.Icc 0 1))
      g : Quiver.Hom Y Z := { toFun := fun y' => { down := ⟨φ y', ⋯⟩ }, continuous_t …
      h : Quiver.Hom Y Z := { toFun := fun x => { down := ⟨0, ⋯⟩ }, continuous_toFun …
      H : Eq 0 (φ y)
      ⊢ False
    -/
    simp only [hφ1 (Set.mem_singleton y), Pi.one_apply] at H
    /-
      case mp.intro.intro.intro.intro
      X Y : CompHaus
      f : Quiver.Hom X Y
      y : (CategoryTheory.forget CompHaus).obj Y
      hy : ∀ (a : (CategoryTheory.forget CompHaus).obj X), Ne (f a) y
      hf : CategoryTheory.Epi f
      C : Set ((CategoryTheory.forget CompHaus).obj Y) := Set.range ⇑f
      hC : IsClosed C
      D : Set ↑Y.toTop := Singleton.singleton y
      hD : IsClosed D
      hCD : Disjoint C D
      φ : ContinuousMap ((CategoryTheory.forget CompHaus).obj Y) Real
      hφ0 : Set.EqOn (⇑φ) 0 C
      hφ1 : Set.EqOn (⇑φ) 1 D
      hφ01 : ∀ (x : (CategoryTheory.forget CompHaus).obj Y), Membership.mem (Set.Icc …
      this✝ : CompactSpace (ULift.{u, 0} ↑(Set.Icc 0 1))
      this : T2Space (ULift.{u, 0} ↑(Set.Icc 0 1))
      Z : CompHaus := CompHaus.of (ULift.{u, 0} ↑(Set.Icc 0 1))
      g : Quiver.Hom Y Z := { toFun := fun y' => { down := ⟨φ y', ⋯⟩ }, continuous_t …
      h : Quiver.Hom Y Z := { toFun := fun x => { down := ⟨0, ⋯⟩ }, continuous_toFun …
      H : Eq 0 1
      ⊢ False
    -/
    exact zero_ne_one H
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X Y : CompHaus
      f : Quiver.Hom X Y
      ⊢ Function.Surjective ⇑f → CategoryTheory.Epi f
    -/
  · rw [← CategoryTheory.epi_iff_surjective]
    /-
      case mpr
      X Y : CompHaus
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Epi ⇑f → CategoryTheory.Epi f
    -/
    apply (forget CompHaus).epi_of_epi_map
    /-
      🎉 no goals
    -/


/-- Every `CompHausLike` admits a functor to `CompHaus`. -/
abbrev compHausLikeToCompHaus (P : TopCat → Prop) : CompHausLike P ⥤ CompHaus :=
                                  /-
                                    P : TopCat → Prop
                                    ⊢ ∀ (X : CompHausLike P), P X.toTop → True
                                  -/
  CompHausLike.toCompHausLike (by simp only [implies_true])
                                  /-
                                    🎉 no goals
                                  -/

