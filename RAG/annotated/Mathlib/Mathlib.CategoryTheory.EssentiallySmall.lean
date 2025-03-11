/-- A category is `EssentiallySmall.{w}` if there exists
an equivalence to some `S : Type w` with `[SmallCategory S]`. -/
@[pp_with_univ]
class EssentiallySmall (C : Type u) [Category.{v} C] : Prop where
  /-- An essentially small category is equivalent to some small category. -/
  equiv_smallCategory : ∃ (S : Type w) (_ : SmallCategory S), Nonempty (C ≌ S)


/-- Constructor for `EssentiallySmall C` from an explicit small category witness. -/
theorem EssentiallySmall.mk' {C : Type u} [Category.{v} C] {S : Type w} [SmallCategory S]
    (e : C ≌ S) : EssentiallySmall.{w} C :=
  ⟨⟨S, _, ⟨e⟩⟩⟩


/-- An arbitrarily chosen small model for an essentially small category.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171) removed @[nolint has_nonempty_instance]
@[pp_with_univ]
def SmallModel (C : Type u) [Category.{v} C] [EssentiallySmall.{w} C] : Type w :=
  Classical.choose (@EssentiallySmall.equiv_smallCategory C _ _)


noncomputable instance smallCategorySmallModel (C : Type u) [Category.{v} C]
    [EssentiallySmall.{w} C] : SmallCategory (SmallModel C) :=
  Classical.choose (Classical.choose_spec (@EssentiallySmall.equiv_smallCategory C _ _))


/-- The (noncomputable) categorical equivalence between
an essentially small category and its small model.
-/
noncomputable def equivSmallModel (C : Type u) [Category.{v} C] [EssentiallySmall.{w} C] :
    C ≌ SmallModel C :=
  Nonempty.some
    (Classical.choose_spec (Classical.choose_spec (@EssentiallySmall.equiv_smallCategory C _ _)))


instance (C : Type u) [Category.{v} C] [EssentiallySmall.{w} C] : EssentiallySmall.{w} Cᵒᵖ :=
  EssentiallySmall.mk' (equivSmallModel C).op


theorem essentiallySmall_congr {C : Type u} [Category.{v} C] {D : Type u'} [Category.{v'} D]
    (e : C ≌ D) : EssentiallySmall.{w} C ↔ EssentiallySmall.{w} D := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    e : CategoryTheory.Equivalence C D
    ⊢ Iff (CategoryTheory.EssentiallySmall.{w, v, u} C) (CategoryTheory.Essentiall …
  -/
  fconstructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      e : CategoryTheory.Equivalence C D
      ⊢ CategoryTheory.EssentiallySmall.{w, v, u} C → CategoryTheory.EssentiallySmal …
    -/
  · rintro ⟨S, 𝒮, ⟨f⟩⟩
    /-
      case mp.mk.intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      e : CategoryTheory.Equivalence C D
      S : Type w
      𝒮 : CategoryTheory.SmallCategory S
      f : CategoryTheory.Equivalence C S
      ⊢ CategoryTheory.EssentiallySmall.{w, v', u'} D
    -/
    exact EssentiallySmall.mk' (e.symm.trans f)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      e : CategoryTheory.Equivalence C D
      ⊢ CategoryTheory.EssentiallySmall.{w, v', u'} D → CategoryTheory.EssentiallySm …
    -/
  · rintro ⟨S, 𝒮, ⟨f⟩⟩
    /-
      case mpr.mk.intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      e : CategoryTheory.Equivalence C D
      S : Type w
      𝒮 : CategoryTheory.SmallCategory S
      f : CategoryTheory.Equivalence D S
      ⊢ CategoryTheory.EssentiallySmall.{w, v, u} C
    -/
    exact EssentiallySmall.mk' (e.trans f)
    /-
      🎉 no goals
    -/


theorem Discrete.essentiallySmallOfSmall {α : Type u} [Small.{w} α] :
    EssentiallySmall.{w} (Discrete α) :=
  ⟨⟨Discrete (Shrink α), ⟨inferInstance, ⟨Discrete.equivalence (equivShrink _)⟩⟩⟩⟩


theorem essentiallySmallSelf : EssentiallySmall.{max w v u} C :=
  EssentiallySmall.mk' (AsSmall.equiv : C ≌ AsSmall.{w} C)


/-- A category is `w`-locally small if every hom set is `w`-small.

See `ShrinkHoms C` for a category instance where every hom set has been replaced by a small model.
-/
@[pp_with_univ]
class LocallySmall (C : Type u) [Category.{v} C] : Prop where
  /-- A locally small category has small hom-types. -/
  hom_small : ∀ X Y : C, Small.{w} (X ⟶ Y) := by infer_instance


instance (C : Type u) [Category.{v} C] [LocallySmall.{w} C] (X Y : C) : Small.{w, v} (X ⟶ Y) :=
  LocallySmall.hom_small X Y


instance (C : Type u) [Category.{v} C] [LocallySmall.{w} C] : LocallySmall.{w} Cᵒᵖ where
  hom_small X Y := small_of_injective (opEquiv X Y).injective


theorem locallySmall_of_faithful {C : Type u} [Category.{v} C] {D : Type u'} [Category.{v'} D]
    (F : C ⥤ D) [F.Faithful] [LocallySmall.{w} D] : LocallySmall.{w} C where
  hom_small {_ _} := small_of_injective F.map_injective


theorem locallySmall_congr {C : Type u} [Category.{v} C] {D : Type u'} [Category.{v'} D]
    (e : C ≌ D) : LocallySmall.{w} C ↔ LocallySmall.{w} D :=
  ⟨fun _ => locallySmall_of_faithful e.inverse, fun _ => locallySmall_of_faithful e.functor⟩


instance (priority := 100) locallySmall_self (C : Type u) [Category.{v} C] :
    LocallySmall.{v} C where


instance (priority := 100) locallySmall_of_univLE (C : Type u) [Category.{v} C] [UnivLE.{v, w}] :
    LocallySmall.{w} C where


theorem locallySmall_max {C : Type u} [Category.{v} C] : LocallySmall.{max v w} C where
  hom_small _ _ := small_max.{w} _


instance (priority := 100) locallySmall_of_essentiallySmall (C : Type u) [Category.{v} C]
    [EssentiallySmall.{w} C] : LocallySmall.{w} C :=
  (locallySmall_congr (equivSmallModel C)).mpr (CategoryTheory.locallySmall_self _)


/-- We define a type alias `ShrinkHoms C` for `C`. When we have `LocallySmall.{w} C`,
we'll put a `Category.{w}` instance on `ShrinkHoms C`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
@[pp_with_univ]
def ShrinkHoms (C : Type u) :=
  C


/-- Help the typechecker by explicitly translating from `C` to `ShrinkHoms C`. -/
def toShrinkHoms {C' : Type*} (X : C') : ShrinkHoms C' :=
  X


/-- Help the typechecker by explicitly translating from `ShrinkHoms C` to `C`. -/
def fromShrinkHoms {C' : Type*} (X : ShrinkHoms C') : C' :=
  X


@[simp]
theorem to_from (X : C') : fromShrinkHoms (toShrinkHoms X) = X :=
  rfl


@[simp]
theorem from_to (X : ShrinkHoms C') : toShrinkHoms (fromShrinkHoms X) = X :=
  rfl


@[simps]
noncomputable instance : Category.{w} (ShrinkHoms C) where
  Hom X Y := Shrink (fromShrinkHoms X ⟶ fromShrinkHoms Y)
  id X := equivShrink _ (𝟙 (fromShrinkHoms X))
  comp f g := equivShrink _ ((equivShrink _).symm f ≫ (equivShrink _).symm g)


/-- Implementation of `ShrinkHoms.equivalence`. -/
@[simps]
noncomputable def functor : C ⥤ ShrinkHoms C where
  obj X := toShrinkHoms X
  map {X Y} f := equivShrink (X ⟶ Y) f


/-- Implementation of `ShrinkHoms.equivalence`. -/
@[simps]
noncomputable def inverse : ShrinkHoms C ⥤ C where
  obj X := fromShrinkHoms X
  map {X Y} f := (equivShrink (fromShrinkHoms X ⟶ fromShrinkHoms Y)).symm f


/-- The categorical equivalence between `C` and `ShrinkHoms C`, when `C` is locally small.
-/
@[simps]
noncomputable def equivalence : C ≌ ShrinkHoms C where
  functor := functor C
  inverse := inverse C
             /-
               C : Type u
               inst✝¹ : CategoryTheory.Category.{v, u} C
               inst✝ : CategoryTheory.LocallySmall.{w, v, u} C
               ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
             -/
  unitIso := NatIso.ofComponents (fun _ ↦ Iso.refl _)
             /-
               🎉 no goals
             -/
               /-
                 C : Type u
                 inst✝¹ : CategoryTheory.Category.{v, u} C
                 inst✝ : CategoryTheory.LocallySmall.{w, v, u} C
                 ⊢ ∀ {X Y : CategoryTheory.ShrinkHoms.{u} C} (f : Quiver.Hom X Y), Eq (Category …
               -/
  counitIso := NatIso.ofComponents (fun _ ↦ Iso.refl _)
               /-
                 🎉 no goals
               -/


instance : (functor C).IsEquivalence := (equivalence C).isEquivalence_functor

instance : (inverse C).IsEquivalence := (equivalence C).isEquivalence_inverse


noncomputable instance [Small.{w} C] : Category.{v} (Shrink.{w} C) :=
  InducedCategory.category (equivShrink C).symm


/-- The categorical equivalence between `C` and `Shrink C`, when `C` is small. -/
noncomputable def equivalence [Small.{w} C] : C ≌ Shrink.{w} C :=
  (inducedFunctor (equivShrink C).symm).asEquivalence.symm


/-- A category is essentially small if and only if
the underlying type of its skeleton (i.e. the "set" of isomorphism classes) is small,
and it is locally small.
-/
theorem essentiallySmall_iff (C : Type u) [Category.{v} C] :
    EssentiallySmall.{w} C ↔ Small.{w} (Skeleton C) ∧ LocallySmall.{w} C := by
  -- This theorem is the only bit of real work in this file.
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ Iff (CategoryTheory.EssentiallySmall.{w, v, u} C) (And (Small.{w, u} (Catego …
  -/
  fconstructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ CategoryTheory.EssentiallySmall.{w, v, u} C → And (Small.{w, u} (CategoryThe …
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : CategoryTheory.EssentiallySmall.{w, v, u} C
      ⊢ And (Small.{w, u} (CategoryTheory.Skeleton C)) (CategoryTheory.LocallySmall. …
    -/
    fconstructor
      /-
        case mp.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        h : CategoryTheory.EssentiallySmall.{w, v, u} C
        ⊢ Small.{w, u} (CategoryTheory.Skeleton C)
      -/
    · rcases h with ⟨S, 𝒮, ⟨e⟩⟩
      /-
        case mp.left.mk.intro.intro.intro
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        S : Type w
        𝒮 : CategoryTheory.SmallCategory S
        e : CategoryTheory.Equivalence C S
        ⊢ Small.{w, u} (CategoryTheory.Skeleton C)
      -/
      refine ⟨⟨Skeleton S, ⟨?_⟩⟩⟩
      /-
        case mp.left.mk.intro.intro.intro
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        S : Type w
        𝒮 : CategoryTheory.SmallCategory S
        e : CategoryTheory.Equivalence C S
        ⊢ Equiv (CategoryTheory.Skeleton C) (CategoryTheory.Skeleton S)
      -/
      exact e.skeletonEquiv
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        h : CategoryTheory.EssentiallySmall.{w, v, u} C
        ⊢ CategoryTheory.LocallySmall.{w, v, u} C
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ And (Small.{w, u} (CategoryTheory.Skeleton C)) (CategoryTheory.LocallySmall. …
    -/
  · rintro ⟨⟨S, ⟨e⟩⟩, L⟩
    /-
      case mpr.intro.mk.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      L : CategoryTheory.LocallySmall.{w, v, u} C
      S : Type w
      e : Equiv (CategoryTheory.Skeleton C) S
      ⊢ CategoryTheory.EssentiallySmall.{w, v, u} C
    -/
    let e' := (ShrinkHoms.equivalence C).skeletonEquiv.symm
    /-
      case mpr.intro.mk.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      L : CategoryTheory.LocallySmall.{w, v, u} C
      S : Type w
      e : Equiv (CategoryTheory.Skeleton C) S
      e' : Equiv (CategoryTheory.Skeleton (CategoryTheory.ShrinkHoms.{u} C)) (Catego …
      ⊢ CategoryTheory.EssentiallySmall.{w, v, u} C
    -/
    letI : Category S := InducedCategory.category (e'.trans e).symm
    /-
      case mpr.intro.mk.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      L : CategoryTheory.LocallySmall.{w, v, u} C
      S : Type w
      e : Equiv (CategoryTheory.Skeleton C) S
      e' : Equiv (CategoryTheory.Skeleton (CategoryTheory.ShrinkHoms.{u} C)) (Catego …
      this : CategoryTheory.Category.{w, w} S := CategoryTheory.InducedCategory.cate …
      ⊢ CategoryTheory.EssentiallySmall.{w, v, u} C
    -/
    refine ⟨⟨S, this, ⟨?_⟩⟩⟩
    refine (ShrinkHoms.equivalence C).trans <|
      (skeletonEquivalence (ShrinkHoms C)).symm.trans
        ((inducedFunctor (e'.trans e).symm).asEquivalence.symm)


theorem essentiallySmall_of_small_of_locallySmall [Small.{w} C] [LocallySmall.{w} C] :
    EssentiallySmall.{w} C :=
                                                                          /-
                                                                            C : Type u
                                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                                            inst✝¹ : Small.{w, u} C
                                                                            inst✝ : CategoryTheory.LocallySmall.{w, v, u} C
                                                                            ⊢ CategoryTheory.LocallySmall.{w, v, u} C
                                                                          -/
  (essentiallySmall_iff C).2 ⟨small_of_surjective Quotient.exists_rep, by infer_instance⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


instance locallySmall_fullSubcategory [LocallySmall.{w} C] (P : C → Prop) :
    LocallySmall.{w} (FullSubcategory P) :=
  locallySmall_of_faithful <| fullSubcategoryInclusion P


instance essentiallySmall_fullSubcategory_mem (s : Set C) [Small.{w} s] [LocallySmall.{w} C] :
    EssentiallySmall.{w} (FullSubcategory (· ∈ s)) :=
  suffices Small.{w} (FullSubcategory (· ∈ s)) from essentiallySmall_of_small_of_locallySmall _
                                                          /-
                                                            C : Type u
                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                            s : Set C
                                                            inst✝¹ : Small.{w, u} ↑s
                                                            inst✝ : CategoryTheory.LocallySmall.{w, v, u} C
                                                            ⊢ Function.Injective fun x => ⟨x.obj, ⋯⟩
                                                          -/
  small_of_injective (f := fun x => (⟨x.1, x.2⟩ : s)) (by aesop_cat)
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Any thin category is locally small.
-/
instance (priority := 100) locallySmall_of_thin {C : Type u} [Category.{v} C] [Quiver.IsThin C] :
    LocallySmall.{w} C where


/--
A thin category is essentially small if and only if the underlying type of its skeleton is small.
-/
theorem essentiallySmall_iff_of_thin {C : Type u} [Category.{v} C] [Quiver.IsThin C] :
    EssentiallySmall.{w} C ↔ Small.{w} (Skeleton C) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : Quiver.IsThin C
    ⊢ Iff (CategoryTheory.EssentiallySmall.{w, v, u} C) (Small.{w, u} (CategoryThe …
  -/
  simp [essentiallySmall_iff, CategoryTheory.locallySmall_of_thin]
  /-
    🎉 no goals
  -/


instance [Small.{w} C] : Small.{w} (Discrete C) := small_map discreteEquiv


