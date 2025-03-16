/-- A cover of `X` consists of jointly surjective indexed family of scheme morphisms
with target `X` all satisfying `P`.

This is merely a coverage in the pretopology defined by `P`, and it would be optimal
if we could reuse the existing API about pretopologies, However, the definitions of sieves and
grothendieck topologies uses `Prop`s, so that the actual open sets and immersions are hard to
obtain. Also, since such a coverage in the pretopology usually contains a proper class of
immersions, it is quite hard to glue them, reason about finite covers, etc.

Note: The `map_prop` field is equipped with a default argument `by infer_instance`. In general
this causes worse error messages, but in practice `P` is mostly defined via `class`.
-/
structure Cover (P : MorphismProperty Scheme.{u}) (X : Scheme.{u}) where
  /-- index set of a cover of a scheme `X` -/
  J : Type v
  /-- the components of a cover -/
  obj (j : J) : Scheme
  /-- the components map to `X` -/
  map (j : J) : obj j ⟶ X
  /-- given a point of `x : X`, `f x` is the index of the component which contains `x`  -/
  f (x : X) : J
  /-- the components cover `X` -/
  covers (x : X) : x ∈ Set.range (map (f x)).base
  /-- the component maps satisfy `P` -/
  map_prop (j : J) : P (map j) := by infer_instance


theorem Cover.iUnion_range {X : Scheme.{u}} (𝒰 : X.Cover P) :
    ⋃ i, Set.range (𝒰.map i).base = Set.univ := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    ⊢ Eq (Set.iUnion fun i => Set.range ⇑(𝒰.map i).base) Set.univ
  -/
  rw [Set.eq_univ_iff_forall]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    ⊢ ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (Set.iUnion fun i => Set.range …
  -/
  intro x
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    x : ↑↑X.toPresheafedSpace
    ⊢ Membership.mem (Set.iUnion fun i => Set.range ⇑(𝒰.map i).base) x
  -/
  rw [Set.mem_iUnion]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    X : AlgebraicGeometry.Scheme
    𝒰 : AlgebraicGeometry.Scheme.Cover P X
    x : ↑↑X.toPresheafedSpace
    ⊢ Exists fun i => Membership.mem (Set.range ⇑(𝒰.map i).base) x
  -/
  exact ⟨𝒰.f x, 𝒰.covers x⟩
  /-
    🎉 no goals
  -/


lemma Cover.exists_eq (𝒰 : X.Cover P) (x : X) : ∃ i y, (𝒰.map i).base y = x :=
  ⟨_, 𝒰.covers x⟩


/-- Given a family of schemes with morphisms to `X` satisfying `P` that jointly
cover `X`, this an associated `P`-cover of `X`. -/
@[simps]
def Cover.mkOfCovers (J : Type*) (obj : J → Scheme.{u}) (map : (j : J) → obj j ⟶ X)
    (covers : ∀ x, ∃ j y, (map j).base y = x)
    (map_prop : ∀ j, P (map j) := by infer_instance) : X.Cover P where
  J := J
  obj := obj
  map := map
  f x := (covers x).choose
  covers x := (covers x).choose_spec
  map_prop := map_prop


/-- Turn a `P`-cover into a `Q`-cover by showing that the components satisfy `Q`. -/
def Cover.changeProp (Q : MorphismProperty Scheme.{u}) (𝒰 : X.Cover P) (h : ∀ j, Q (𝒰.map j)) :
    X.Cover Q where
  J := 𝒰.J
  obj := 𝒰.obj
  map := 𝒰.map
  f := 𝒰.f
  covers := 𝒰.covers
  map_prop := h


/-- Given a `P`-cover `{ Uᵢ }` of `X`, and for each `Uᵢ` a `P`-cover, we may combine these
covers to form a `P`-cover of `X`. -/
@[simps! J obj map]
def Cover.bind [P.IsStableUnderComposition] (f : ∀ x : 𝒰.J, (𝒰.obj x).Cover P) : X.Cover P where
  J := Σ i : 𝒰.J, (f i).J
  obj x := (f x.1).obj x.2
  map x := (f x.1).map x.2 ≫ 𝒰.map x.1
  f x := ⟨_, (f _).f (𝒰.covers x).choose⟩
  covers x := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝¹ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝ : P.IsStableUnderComposition
      f : (x : 𝒰.J) → AlgebraicGeometry.Scheme.Cover P (𝒰.obj x)
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun x => CategoryTheory.CategoryStruct.comp ((f …
    -/
    let y := (𝒰.covers x).choose
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝¹ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝ : P.IsStableUnderComposition
      f : (x : 𝒰.J) → AlgebraicGeometry.Scheme.Cover P (𝒰.obj x)
      x : ↑↑X.toPresheafedSpace
      y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace := Exists.choose ⋯
      ⊢ Membership.mem (Set.range ⇑((fun x => CategoryTheory.CategoryStruct.comp ((f …
    -/
    have hy : (𝒰.map (𝒰.f x)).base y = x := (𝒰.covers x).choose_spec
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝¹ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝ : P.IsStableUnderComposition
      f : (x : 𝒰.J) → AlgebraicGeometry.Scheme.Cover P (𝒰.obj x)
      x : ↑↑X.toPresheafedSpace
      y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace := Exists.choose ⋯
      hy : Eq ((𝒰.map (𝒰.f x)).base y) x
      ⊢ Membership.mem (Set.range ⇑((fun x => CategoryTheory.CategoryStruct.comp ((f …
    -/
    rcases (f (𝒰.f x)).covers y with ⟨z, hz⟩
    /-
      case intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝¹ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝ : P.IsStableUnderComposition
      f : (x : 𝒰.J) → AlgebraicGeometry.Scheme.Cover P (𝒰.obj x)
      x : ↑↑X.toPresheafedSpace
      y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace := Exists.choose ⋯
      hy : Eq ((𝒰.map (𝒰.f x)).base y) x
      z : ↑↑((f (𝒰.f x)).obj ((f (𝒰.f x)).f y)).toPresheafedSpace
      hz : Eq (((f (𝒰.f x)).map ((f (𝒰.f x)).f y)).base z) y
      ⊢ Membership.mem (Set.range ⇑((fun x => CategoryTheory.CategoryStruct.comp ((f …
    -/
    change x ∈ Set.range ((f (𝒰.f x)).map ((f (𝒰.f x)).f y) ≫ 𝒰.map (𝒰.f x)).base
    /-
      case intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝¹ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝ : P.IsStableUnderComposition
      f : (x : 𝒰.J) → AlgebraicGeometry.Scheme.Cover P (𝒰.obj x)
      x : ↑↑X.toPresheafedSpace
      y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace := Exists.choose ⋯
      hy : Eq ((𝒰.map (𝒰.f x)).base y) x
      z : ↑↑((f (𝒰.f x)).obj ((f (𝒰.f x)).f y)).toPresheafedSpace
      hz : Eq (((f (𝒰.f x)).map ((f (𝒰.f x)).f y)).base z) y
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.CategoryStruct.comp ((f (𝒰.f x)). …
    -/
    use z
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝¹ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝ : P.IsStableUnderComposition
      f : (x : 𝒰.J) → AlgebraicGeometry.Scheme.Cover P (𝒰.obj x)
      x : ↑↑X.toPresheafedSpace
      y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace := Exists.choose ⋯
      hy : Eq ((𝒰.map (𝒰.f x)).base y) x
      z : ↑↑((f (𝒰.f x)).obj ((f (𝒰.f x)).f y)).toPresheafedSpace
      hz : Eq (((f (𝒰.f x)).map ((f (𝒰.f x)).f y)).base z) y
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((f (𝒰.f x)).map ((f (𝒰.f x)).f y))  …
    -/
    erw [comp_apply]
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝¹ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝ : P.IsStableUnderComposition
      f : (x : 𝒰.J) → AlgebraicGeometry.Scheme.Cover P (𝒰.obj x)
      x : ↑↑X.toPresheafedSpace
      y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace := Exists.choose ⋯
      hy : Eq ((𝒰.map (𝒰.f x)).base y) x
      z : ↑↑((f (𝒰.f x)).obj ((f (𝒰.f x)).f y)).toPresheafedSpace
      hz : Eq (((f (𝒰.f x)).map ((f (𝒰.f x)).f y)).base z) y
      ⊢ Eq ((AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.Sch …
    -/
    rw [hz, hy]
    /-
      🎉 no goals
    -/
  map_prop _ := P.comp_mem _ _ ((f _).map_prop _) (𝒰.map_prop _)


/-- An isomorphism `X ⟶ Y` is a `P`-cover of `Y`. -/
@[simps J obj map]
def coverOfIsIso [P.ContainsIdentities] [P.RespectsIso] {X Y : Scheme.{u}} (f : X ⟶ Y)
    [IsIso f] : Cover.{v} P Y where
  J := PUnit.{v + 1}
  obj _ := X
  map _ := f
  f _ := PUnit.unit
  covers x := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y✝ Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f✝ : Quiver.Hom X✝ Z
      g : Quiver.Hom Y✝ Z
      inst✝³ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝² : P.ContainsIdentities
      inst✝¹ : P.RespectsIso
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      x : ↑↑Y.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun x => f) ((fun x => PUnit.unit) x)).base) x
    -/
    rw [Set.range_eq_univ.mpr]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y✝ Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f✝ : Quiver.Hom X✝ Z
      g : Quiver.Hom Y✝ Z
      inst✝³ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝² : P.ContainsIdentities
      inst✝¹ : P.RespectsIso
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      x : ↑↑Y.toPresheafedSpace
      ⊢ Membership.mem Set.univ x
    -/
    all_goals try trivial
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y✝ Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f✝ : Quiver.Hom X✝ Z
      g : Quiver.Hom Y✝ Z
      inst✝³ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝² : P.ContainsIdentities
      inst✝¹ : P.RespectsIso
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      x : ↑↑Y.toPresheafedSpace
      ⊢ Function.Surjective ⇑((fun x => f) ((fun x => PUnit.unit) x)).base
    -/
    rw [← TopCat.epi_iff_surjective]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y✝ Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f✝ : Quiver.Hom X✝ Z
      g : Quiver.Hom Y✝ Z
      inst✝³ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝² : P.ContainsIdentities
      inst✝¹ : P.RespectsIso
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      x : ↑↑Y.toPresheafedSpace
      ⊢ CategoryTheory.Epi ((fun x => f) ((fun x => PUnit.unit) x)).base
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  map_prop _ := P.of_isIso _


/-- We construct a cover from another, by providing the needed fields and showing that the
provided fields are isomorphic with the original cover. -/
@[simps J obj map]
def Cover.copy [P.RespectsIso] {X : Scheme.{u}} (𝒰 : X.Cover P)
    (J : Type*) (obj : J → Scheme)
    (map : ∀ i, obj i ⟶ X) (e₁ : J ≃ 𝒰.J) (e₂ : ∀ i, obj i ≅ 𝒰.obj (e₁ i))
    (h : ∀ i, map i = (e₂ i).hom ≫ 𝒰.map (e₁ i)) : X.Cover P :=
  { J, obj, map
    f := fun x ↦ e₁.symm (𝒰.f x)
    covers := fun x ↦ by
      rw [h, Scheme.comp_base, TopCat.coe_comp, Set.range_comp, Set.range_eq_univ.mpr,
        Set.image_univ, e₁.rightInverse_symm]
        /-
          P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
          X✝ Y Z : AlgebraicGeometry.Scheme
          𝒰✝ : AlgebraicGeometry.Scheme.Cover P X✝
          f : Quiver.Hom X✝ Z
          g : Quiver.Hom Y Z
          inst✝¹ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categ …
          inst✝ : P.RespectsIso
          X : AlgebraicGeometry.Scheme
          𝒰 : AlgebraicGeometry.Scheme.Cover P X
          J : Type u_1
          obj : J → AlgebraicGeometry.Scheme
          map : (i : J) → Quiver.Hom (obj i) X
          e₁ : Equiv J 𝒰.J
          e₂ : (i : J) → CategoryTheory.Iso (obj i) (𝒰.obj (e₁ i))
          h : ∀ (i : J), Eq (map i) (CategoryTheory.CategoryStruct.comp (e₂ i).hom (𝒰.ma …
          x : ↑↑X.toPresheafedSpace
          ⊢ Membership.mem (Set.range ⇑(𝒰.map (𝒰.f x)).base) x
        -/
      · exact 𝒰.covers x
        /-
          🎉 no goals
        -/
        /-
          P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
          X✝ Y Z : AlgebraicGeometry.Scheme
          𝒰✝ : AlgebraicGeometry.Scheme.Cover P X✝
          f : Quiver.Hom X✝ Z
          g : Quiver.Hom Y Z
          inst✝¹ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categ …
          inst✝ : P.RespectsIso
          X : AlgebraicGeometry.Scheme
          𝒰 : AlgebraicGeometry.Scheme.Cover P X
          J : Type u_1
          obj : J → AlgebraicGeometry.Scheme
          map : (i : J) → Quiver.Hom (obj i) X
          e₁ : Equiv J 𝒰.J
          e₂ : (i : J) → CategoryTheory.Iso (obj i) (𝒰.obj (e₁ i))
          h : ∀ (i : J), Eq (map i) (CategoryTheory.CategoryStruct.comp (e₂ i).hom (𝒰.ma …
          x : ↑↑X.toPresheafedSpace
          ⊢ Function.Surjective ⇑(e₂ ((fun x => e₁.symm (𝒰.f x)) x)).hom.base
        -/
      · rw [← TopCat.epi_iff_surjective]; infer_instance
                                          /-
                                            🎉 no goals
                                          -/
    map_prop := fun j ↦ by
      /-
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        X✝ Y Z : AlgebraicGeometry.Scheme
        𝒰✝ : AlgebraicGeometry.Scheme.Cover P X✝
        f : Quiver.Hom X✝ Z
        g : Quiver.Hom Y Z
        inst✝¹ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categ …
        inst✝ : P.RespectsIso
        X : AlgebraicGeometry.Scheme
        𝒰 : AlgebraicGeometry.Scheme.Cover P X
        J : Type u_1
        obj : J → AlgebraicGeometry.Scheme
        map : (i : J) → Quiver.Hom (obj i) X
        e₁ : Equiv J 𝒰.J
        e₂ : (i : J) → CategoryTheory.Iso (obj i) (𝒰.obj (e₁ i))
        h : ∀ (i : J), Eq (map i) (CategoryTheory.CategoryStruct.comp (e₂ i).hom (𝒰.ma …
        j : J
        ⊢ P (map j)
      -/
      rw [h, P.cancel_left_of_respectsIso]
      /-
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        X✝ Y Z : AlgebraicGeometry.Scheme
        𝒰✝ : AlgebraicGeometry.Scheme.Cover P X✝
        f : Quiver.Hom X✝ Z
        g : Quiver.Hom Y Z
        inst✝¹ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categ …
        inst✝ : P.RespectsIso
        X : AlgebraicGeometry.Scheme
        𝒰 : AlgebraicGeometry.Scheme.Cover P X
        J : Type u_1
        obj : J → AlgebraicGeometry.Scheme
        map : (i : J) → Quiver.Hom (obj i) X
        e₁ : Equiv J 𝒰.J
        e₂ : (i : J) → CategoryTheory.Iso (obj i) (𝒰.obj (e₁ i))
        h : ∀ (i : J), Eq (map i) (CategoryTheory.CategoryStruct.comp (e₂ i).hom (𝒰.ma …
        j : J
        ⊢ P (𝒰.map (e₁ j))
      -/
      exact 𝒰.map_prop (e₁ j) }
      /-
        🎉 no goals
      -/


/-- The pushforward of a cover along an isomorphism. -/
@[simps! J obj map]
def Cover.pushforwardIso [P.RespectsIso] [P.ContainsIdentities] [P.IsStableUnderComposition]
    {X Y : Scheme.{u}} (𝒰 : Cover.{v} P X) (f : X ⟶ Y) [IsIso f] :
    Cover.{v} P Y :=
  ((coverOfIsIso.{v, u} f).bind fun _ => 𝒰).copy 𝒰.J _ _
    ((Equiv.punitProd _).symm.trans (Equiv.sigmaEquivProd PUnit 𝒰.J).symm) (fun _ => Iso.refl _)
    fun _ => (Category.id_comp _).symm


/-- Adding map satisfying `P` into a cover gives another cover. -/
@[simps]
def Cover.add {X Y : Scheme.{u}} (𝒰 : X.Cover P) (f : Y ⟶ X) (hf : P f := by infer_instance) :
    X.Cover P where
  J := Option 𝒰.J
  obj i := Option.rec Y 𝒰.obj i
  map i := Option.rec f 𝒰.map i
  f x := some (𝒰.f x)
  covers := 𝒰.covers
  map_prop j := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y✝ Z : AlgebraicGeometry.Scheme
      𝒰✝ : AlgebraicGeometry.Scheme.Cover P X✝
      f✝ : Quiver.Hom X✝ Z
      g : Quiver.Hom Y✝ Z
      inst✝ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      X Y : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom Y X
      hf : autoParam (P f) _auto✝
      j : Option 𝒰.J
      ⊢ P ((fun i => Option.rec f 𝒰.map i) j)
    -/
    obtain ⟨_ | _⟩ := j
      /-
        case none
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        𝒰✝ : AlgebraicGeometry.Scheme.Cover P X✝
        f✝ : Quiver.Hom X✝ Z
        g : Quiver.Hom Y✝ Z
        inst✝ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
        X Y : AlgebraicGeometry.Scheme
        𝒰 : AlgebraicGeometry.Scheme.Cover P X
        f : Quiver.Hom Y X
        hf : autoParam (P f) _auto✝
        ⊢ P ((fun i => Option.rec f 𝒰.map i) Option.none)
      -/
    · exact hf
      /-
        🎉 no goals
      -/
      /-
        case some
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        𝒰✝ : AlgebraicGeometry.Scheme.Cover P X✝
        f✝ : Quiver.Hom X✝ Z
        g : Quiver.Hom Y✝ Z
        inst✝ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
        X Y : AlgebraicGeometry.Scheme
        𝒰 : AlgebraicGeometry.Scheme.Cover P X
        f : Quiver.Hom Y X
        hf : autoParam (P f) _auto✝
        val✝ : 𝒰.J
        ⊢ P ((fun i => Option.rec f 𝒰.map i) (Option.some val✝))
      -/
    · exact 𝒰.map_prop _
      /-
        🎉 no goals
      -/


/-- A morphism property of schemes is said to preserve joint surjectivity, if
for any pair of morphisms `f : X ⟶ S` and `g : Y ⟶ S` where `g` satisfies `P`,
any pair of points `x : X` and `y : Y` with `f x = g y` can be lifted to a point
of `X ×[S] Y`.

In later files, this will be automatic, since this holds for any morphism `g`
(see `AlgebraicGeometry.Scheme.isJointlySurjectivePreserving`). But at
this early stage in the import tree, we only know it for open immersions. -/
class IsJointlySurjectivePreserving (P : MorphismProperty Scheme.{u}) where
  exists_preimage_fst_triplet_of_prop {X Y S : Scheme.{u}} {f : X ⟶ S} {g : Y ⟶ S} [HasPullback f g]
    (hg : P g) (x : X) (y : Y) (h : f.base x = g.base y) :
    ∃ a : ↑(pullback f g), (pullback.fst f g).base a = x


lemma IsJointlySurjectivePreserving.exists_preimage_snd_triplet_of_prop
    [IsJointlySurjectivePreserving P] {X Y S : Scheme.{u}} {f : X ⟶ S} {g : Y ⟶ S} [HasPullback f g]
    (hf : P f) (x : X) (y : Y) (h : f.base x = g.base y) :
    ∃ a : ↑(pullback f g), (pullback.snd f g).base a = y := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : CategoryTheory.Limits.HasPullback f g
    hf : P f
    x : ↑↑X.toPresheafedSpace
    y : ↑↑Y.toPresheafedSpace
    h : Eq (f.base x) (g.base y)
    ⊢ Exists fun a => Eq ((CategoryTheory.Limits.pullback.snd f g).base a) y
  -/
  let iso := pullbackSymmetry f g
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : CategoryTheory.Limits.HasPullback f g
    hf : P f
    x : ↑↑X.toPresheafedSpace
    y : ↑↑Y.toPresheafedSpace
    h : Eq (f.base x) (g.base y)
    iso : CategoryTheory.Iso (CategoryTheory.Limits.pullback f g) (CategoryTheory. …
    ⊢ Exists fun a => Eq ((CategoryTheory.Limits.pullback.snd f g).base a) y
  -/
  haveI : HasPullback g f := hasPullback_symmetry f g
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : CategoryTheory.Limits.HasPullback f g
    hf : P f
    x : ↑↑X.toPresheafedSpace
    y : ↑↑Y.toPresheafedSpace
    h : Eq (f.base x) (g.base y)
    iso : CategoryTheory.Iso (CategoryTheory.Limits.pullback f g) (CategoryTheory. …
    this : CategoryTheory.Limits.HasPullback g f
    ⊢ Exists fun a => Eq ((CategoryTheory.Limits.pullback.snd f g).base a) y
  -/
  obtain ⟨a, ha⟩ := exists_preimage_fst_triplet_of_prop hf y x h.symm
  /-
    case intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : CategoryTheory.Limits.HasPullback f g
    hf : P f
    x : ↑↑X.toPresheafedSpace
    y : ↑↑Y.toPresheafedSpace
    h : Eq (f.base x) (g.base y)
    iso : CategoryTheory.Iso (CategoryTheory.Limits.pullback f g) (CategoryTheory. …
    this : CategoryTheory.Limits.HasPullback g f
    a : ↑↑(CategoryTheory.Limits.pullback g f).toPresheafedSpace
    ha : Eq ((CategoryTheory.Limits.pullback.fst g f).base a) y
    ⊢ Exists fun a => Eq ((CategoryTheory.Limits.pullback.snd f g).base a) y
  -/
  use (pullbackSymmetry f g).inv.base a
  /-
    case h
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : CategoryTheory.Limits.HasPullback f g
    hf : P f
    x : ↑↑X.toPresheafedSpace
    y : ↑↑Y.toPresheafedSpace
    h : Eq (f.base x) (g.base y)
    iso : CategoryTheory.Iso (CategoryTheory.Limits.pullback f g) (CategoryTheory. …
    this : CategoryTheory.Limits.HasPullback g f
    a : ↑↑(CategoryTheory.Limits.pullback g f).toPresheafedSpace
    ha : Eq ((CategoryTheory.Limits.pullback.fst g f).base a) y
    ⊢ Eq ((CategoryTheory.Limits.pullback.snd f g).base ((CategoryTheory.Limits.pu …
  -/
  rwa [← Scheme.comp_base_apply, pullbackSymmetry_inv_comp_snd]
  /-
    🎉 no goals
  -/


instance : IsJointlySurjectivePreserving @IsOpenImmersion where
  exists_preimage_fst_triplet_of_prop {X Y S f g} _ hg x y h := by
    rw [← show _ = (pullback.fst _ _ : pullback f g ⟶ _).base from
        PreservesPullback.iso_hom_fst Scheme.forgetToTop f g]
    have : x ∈ Set.range (pullback.fst f.base g.base) := by
      rw [TopCat.pullback_fst_range f.base g.base]
      use y
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y✝ Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f✝ : Quiver.Hom X✝ Z
      g✝ : Quiver.Hom Y✝ Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x✝ : CategoryTheory.Limits.HasPullback f g
      hg : AlgebraicGeometry.IsOpenImmersion g
      x : ↑↑X.toPresheafedSpace
      y : ↑↑Y.toPresheafedSpace
      h : Eq (f.base x) (g.base y)
      this : Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst f.base g …
      ⊢ Exists fun a => Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limi …
    -/
    obtain ⟨a, ha⟩ := this
    /-
      case intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y✝ Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f✝ : Quiver.Hom X✝ Z
      g✝ : Quiver.Hom Y✝ Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x✝ : CategoryTheory.Limits.HasPullback f g
      hg : AlgebraicGeometry.IsOpenImmersion g
      x : ↑↑X.toPresheafedSpace
      y : ↑↑Y.toPresheafedSpace
      h : Eq (f.base x) (g.base y)
      a : ↑(CategoryTheory.Limits.pullback f.base g.base)
      ha : Eq ((CategoryTheory.Limits.pullback.fst f.base g.base) a) x
      ⊢ Exists fun a => Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limi …
    -/
    use (PreservesPullback.iso forgetToTop f g).inv a
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y✝ Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f✝ : Quiver.Hom X✝ Z
      g✝ : Quiver.Hom Y✝ Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      x✝ : CategoryTheory.Limits.HasPullback f g
      hg : AlgebraicGeometry.IsOpenImmersion g
      x : ↑↑X.toPresheafedSpace
      y : ↑↑Y.toPresheafedSpace
      h : Eq (f.base x) (g.base y)
      a : ↑(CategoryTheory.Limits.pullback f.base g.base)
      ha : Eq ((CategoryTheory.Limits.pullback.fst f.base g.base) a) x
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesPull …
    -/
    rwa [← TopCat.comp_app, Iso.inv_hom_id_assoc]
    /-
      🎉 no goals
    -/


/-- Given a cover on `X`, we may pull them back along a morphism `W ⟶ X` to obtain
a cover of `W`.

Note that this requires the (unnecessary) assumptions that the pullback exists and that `P`
preserves joint surjectivity. This is needed, because we don't know these in general at this
stage of the import tree, but this API is used in the case of `P = IsOpenImmersion` to
obtain these results in the general case. -/
@[simps]
def Cover.pullbackCover [P.IsStableUnderBaseChange] [IsJointlySurjectivePreserving P]
    {X W : Scheme.{u}} (𝒰 : X.Cover P) (f : W ⟶ X) [∀ x, HasPullback f (𝒰.map x)] : W.Cover P where
  J := 𝒰.J
  obj x := pullback f (𝒰.map x)
  map _ := pullback.fst _ _
  f x := 𝒰.f (f.base x)
  covers x := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : AlgebraicGeometry.Scheme.Cover P X✝
      f✝ : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝³ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categ …
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback f (𝒰.map x)
      x : ↑↑W.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun x => CategoryTheory.Limits.pullback.fst f ( …
    -/
    obtain ⟨y, hy⟩ := 𝒰.covers (f.base x)
    exact IsJointlySurjectivePreserving.exists_preimage_fst_triplet_of_prop
      (𝒰.map_prop _) x y hy.symm
  map_prop j := P.pullback_fst _ _ (𝒰.map_prop j)


/-- The family of morphisms from the pullback cover to the original cover. -/
def Cover.pullbackHom [P.IsStableUnderBaseChange] [IsJointlySurjectivePreserving P]
    {X W : Scheme.{u}} (𝒰 : X.Cover P)
    (f : W ⟶ X) (i) [∀ x, HasPullback f (𝒰.map x)] :
    (𝒰.pullbackCover f).obj i ⟶ 𝒰.obj i :=
  pullback.snd f (𝒰.map i)


@[reassoc (attr := simp)]
lemma Cover.pullbackHom_map [P.IsStableUnderBaseChange] [IsJointlySurjectivePreserving P]
    {X W : Scheme.{u}} (𝒰 : X.Cover P) (f : W ⟶ X) [∀ (x : 𝒰.J), HasPullback f (𝒰.map x)] (i) :
    𝒰.pullbackHom f i ≫ 𝒰.map i = (𝒰.pullbackCover f).map i ≫ f := pullback.condition.symm


/-- Given a cover on `X`, we may pull them back along a morphism `f : W ⟶ X` to obtain
a cover of `W`. This is similar to `Scheme.Cover.pullbackCover`, but here we
take `pullback (𝒰.map x) f` instead of `pullback f (𝒰.map x)`. -/
@[simps]
def Cover.pullbackCover' [P.IsStableUnderBaseChange] [IsJointlySurjectivePreserving P]
    {X W : Scheme.{u}} (𝒰 : X.Cover P) (f : W ⟶ X)
    [∀ x, HasPullback (𝒰.map x) f] :
    W.Cover P where
  J := 𝒰.J
  obj x := pullback (𝒰.map x) f
  map _ := pullback.snd _ _
  f x := 𝒰.f (f.base x)
  covers x := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : AlgebraicGeometry.Scheme.Cover P X✝
      f✝ : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝³ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categ …
      inst✝² : P.IsStableUnderBaseChange
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X W : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X
      f : Quiver.Hom W X
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (𝒰.map x) f
      x : ↑↑W.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun x => CategoryTheory.Limits.pullback.snd (𝒰. …
    -/
    obtain ⟨y, hy⟩ := 𝒰.covers (f.base x)
    exact IsJointlySurjectivePreserving.exists_preimage_snd_triplet_of_prop
      (𝒰.map_prop _) y x hy
  map_prop j := P.pullback_snd _ _ (𝒰.map_prop j)


/-- Given covers `{ Uᵢ }` and `{ Uⱼ }`, we may form the cover `{ Uᵢ ×[X] Uⱼ }`. -/
def Cover.inter [P.IsStableUnderBaseChange] [P.IsStableUnderComposition]
    [IsJointlySurjectivePreserving P]
    {X : Scheme.{u}} (𝒰₁ : Scheme.Cover.{v₁} P X)
    (𝒰₂ : Scheme.Cover.{v₂} P X)
    [∀ (i : 𝒰₁.J) (j : 𝒰₂.J), HasPullback (𝒰₁.map i) (𝒰₂.map j)] : X.Cover P where
  J := 𝒰₁.J × 𝒰₂.J
  obj ij := pullback (𝒰₁.map ij.1) (𝒰₂.map ij.2)
  map ij := pullback.fst _ _ ≫ 𝒰₁.map ij.1
  f x := ⟨𝒰₁.f x, 𝒰₂.f x⟩
  covers x := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝⁴ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝³ : P.IsStableUnderBaseChange
      inst✝² : P.IsStableUnderComposition
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : AlgebraicGeometry.Scheme
      𝒰₁ : AlgebraicGeometry.Scheme.Cover P X
      𝒰₂ : AlgebraicGeometry.Scheme.Cover P X
      inst✝ : ∀ (i : 𝒰₁.J) (j : 𝒰₂.J), CategoryTheory.Limits.HasPullback (𝒰₁.map i)  …
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun ij => CategoryTheory.CategoryStruct.comp (C …
    -/
    simp only [comp_coeBase, TopCat.coe_comp, Set.mem_range, Function.comp_apply]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝⁴ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝³ : P.IsStableUnderBaseChange
      inst✝² : P.IsStableUnderComposition
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : AlgebraicGeometry.Scheme
      𝒰₁ : AlgebraicGeometry.Scheme.Cover P X
      𝒰₂ : AlgebraicGeometry.Scheme.Cover P X
      inst✝ : ∀ (i : 𝒰₁.J) (j : 𝒰₂.J), CategoryTheory.Limits.HasPullback (𝒰₁.map i)  …
      x : ↑↑X.toPresheafedSpace
      ⊢ Exists fun y => Eq ((𝒰₁.map (𝒰₁.f x)).base ((CategoryTheory.Limits.pullback. …
    -/
    obtain ⟨y₁, hy₁⟩ := 𝒰₁.covers x
    /-
      case intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝⁴ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝³ : P.IsStableUnderBaseChange
      inst✝² : P.IsStableUnderComposition
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : AlgebraicGeometry.Scheme
      𝒰₁ : AlgebraicGeometry.Scheme.Cover P X
      𝒰₂ : AlgebraicGeometry.Scheme.Cover P X
      inst✝ : ∀ (i : 𝒰₁.J) (j : 𝒰₂.J), CategoryTheory.Limits.HasPullback (𝒰₁.map i)  …
      x : ↑↑X.toPresheafedSpace
      y₁ : ↑↑(𝒰₁.obj (𝒰₁.f x)).toPresheafedSpace
      hy₁ : Eq ((𝒰₁.map (𝒰₁.f x)).base y₁) x
      ⊢ Exists fun y => Eq ((𝒰₁.map (𝒰₁.f x)).base ((CategoryTheory.Limits.pullback. …
    -/
    obtain ⟨y₂, hy₂⟩ := 𝒰₂.covers x
    obtain ⟨z, hz⟩ := IsJointlySurjectivePreserving.exists_preimage_fst_triplet_of_prop
      (𝒰₂.map_prop _) y₁ y₂ (by rw [hy₁, hy₂])
    /-
      case intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝⁴ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝³ : P.IsStableUnderBaseChange
      inst✝² : P.IsStableUnderComposition
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : AlgebraicGeometry.Scheme
      𝒰₁ : AlgebraicGeometry.Scheme.Cover P X
      𝒰₂ : AlgebraicGeometry.Scheme.Cover P X
      inst✝ : ∀ (i : 𝒰₁.J) (j : 𝒰₂.J), CategoryTheory.Limits.HasPullback (𝒰₁.map i)  …
      x : ↑↑X.toPresheafedSpace
      y₁ : ↑↑(𝒰₁.obj (𝒰₁.f x)).toPresheafedSpace
      hy₁ : Eq ((𝒰₁.map (𝒰₁.f x)).base y₁) x
      y₂ : ↑↑(𝒰₂.obj (𝒰₂.f x)).toPresheafedSpace
      hy₂ : Eq ((𝒰₂.map (𝒰₂.f x)).base y₂) x
      z : ↑↑(CategoryTheory.Limits.pullback (𝒰₁.map (𝒰₁.f x)) (𝒰₂.map (𝒰₂.f x))).toP …
      hz : Eq ((CategoryTheory.Limits.pullback.fst (𝒰₁.map (𝒰₁.f x)) (𝒰₂.map (𝒰₂.f x …
      ⊢ Exists fun y => Eq ((𝒰₁.map (𝒰₁.f x)).base ((CategoryTheory.Limits.pullback. …
    -/
    use z
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : AlgebraicGeometry.Scheme.Cover P X✝
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝⁴ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      inst✝³ : P.IsStableUnderBaseChange
      inst✝² : P.IsStableUnderComposition
      inst✝¹ : AlgebraicGeometry.Scheme.IsJointlySurjectivePreserving P
      X : AlgebraicGeometry.Scheme
      𝒰₁ : AlgebraicGeometry.Scheme.Cover P X
      𝒰₂ : AlgebraicGeometry.Scheme.Cover P X
      inst✝ : ∀ (i : 𝒰₁.J) (j : 𝒰₂.J), CategoryTheory.Limits.HasPullback (𝒰₁.map i)  …
      x : ↑↑X.toPresheafedSpace
      y₁ : ↑↑(𝒰₁.obj (𝒰₁.f x)).toPresheafedSpace
      hy₁ : Eq ((𝒰₁.map (𝒰₁.f x)).base y₁) x
      y₂ : ↑↑(𝒰₂.obj (𝒰₂.f x)).toPresheafedSpace
      hy₂ : Eq ((𝒰₂.map (𝒰₂.f x)).base y₂) x
      z : ↑↑(CategoryTheory.Limits.pullback (𝒰₁.map (𝒰₁.f x)) (𝒰₂.map (𝒰₂.f x))).toP …
      hz : Eq ((CategoryTheory.Limits.pullback.fst (𝒰₁.map (𝒰₁.f x)) (𝒰₂.map (𝒰₂.f x …
      ⊢ Eq ((𝒰₁.map (𝒰₁.f x)).base ((CategoryTheory.Limits.pullback.fst (𝒰₁.map (𝒰₁. …
    -/
    rw [hz, hy₁]
    /-
      🎉 no goals
    -/
  map_prop ij := P.comp_mem _ _ (P.pullback_fst _ _ (𝒰₂.map_prop ij.2)) (𝒰₁.map_prop ij.1)


/--
An affine cover of `X` consists of a jointly surjective family of maps into `X` from
spectra of rings.

Note: The `map_prop` field is equipped with a default argument `by infer_instance`. In general
this causes worse error messages, but in practice `P` is mostly defined via `class`.
-/
structure AffineCover (P : MorphismProperty Scheme.{u}) (X : Scheme.{u}) where
  /-- index set of an affine cover of a scheme `X` -/
  J : Type v
  /-- the ring associated to a component of an affine cover -/
  obj (j : J) : CommRingCat.{u}
  /-- the components map to `X` -/
  map (j : J) : Spec (obj j) ⟶ X
  /-- given a point of `x : X`, `f x` is the index of the component which contains `x`  -/
  f (x : X) : J
  /-- the components cover `X` -/
  covers (x : X) : x ∈ Set.range (map (f x)).base
  /-- the component maps satisfy `P` -/
  map_prop (j : J) : P (map j) := by infer_instance


/-- The cover associated to an affine cover. -/
@[simps]
def AffineCover.cover {X : Scheme.{u}} (𝒰 : X.AffineCover P) : X.Cover P where
  J := 𝒰.J
  map := 𝒰.map
  f := 𝒰.f
  covers := 𝒰.covers
  map_prop := 𝒰.map_prop


/--
A morphism between covers `𝒰 ⟶ 𝒱` indicates that `𝒰` is a refinement of `𝒱`.
Since covers of schemes are indexed, the definition also involves a map on the
indexing types.
-/
structure Cover.Hom {X : Scheme.{u}} (𝒰 𝒱 : Cover.{v} P X) where
  /-- The map on indexing types associated to a morphism of covers. -/
  idx : 𝒰.J → 𝒱.J
  /-- The morphism between open subsets associated to a morphism of covers. -/
  app (j : 𝒰.J) : 𝒰.obj j ⟶ 𝒱.obj (idx j)
  app_prop (j : 𝒰.J) : P (app j) := by infer_instance
  w (j : 𝒰.J) : app j ≫ 𝒱.map _ = 𝒰.map _ := by aesop_cat


attribute [reassoc (attr := simp)] Cover.Hom.w


/-- The identity morphism in the category of covers of a scheme. -/
def Cover.Hom.id [P.ContainsIdentities] {X : Scheme.{u}} (𝒰 : Cover.{v} P X) : 𝒰.Hom 𝒰 where
  idx j := j
  app _ := 𝟙 _
  app_prop _ := P.id_mem _


/-- The composition of two morphisms in the category of covers of a scheme. -/
def Cover.Hom.comp [P.IsStableUnderComposition] {X : Scheme.{u}} {𝒰 𝒱 𝒲 : Cover.{v} P X}
    (f : 𝒰.Hom 𝒱) (g : 𝒱.Hom 𝒲) : 𝒰.Hom 𝒲 where
  idx j := g.idx <| f.idx j
  app _ := f.app _ ≫ g.app _
  app_prop _ := P.comp_mem _ _ (f.app_prop _) (g.app_prop _)


instance Cover.category [P.IsMultiplicative] {X : Scheme.{u}} : Category (Cover.{v} P X) where
  Hom 𝒰 𝒱 := 𝒰.Hom 𝒱
  id := Cover.Hom.id
  comp f g := f.comp g


@[simp]
lemma Cover.id_idx_apply {X : Scheme.{u}} (𝒰 : X.Cover P) (j : 𝒰.J) :
    (𝟙 𝒰 : 𝒰 ⟶ 𝒰).idx j = j := rfl


@[simp]
lemma Cover.id_app {X : Scheme.{u}} (𝒰 : X.Cover P) (j : 𝒰.J) :
    (𝟙 𝒰 : 𝒰 ⟶ 𝒰).app j = 𝟙 _ := rfl


@[simp]
lemma Cover.comp_idx_apply {X : Scheme.{u}} {𝒰 𝒱 𝒲 : X.Cover P}
    (f : 𝒰 ⟶ 𝒱) (g : 𝒱 ⟶ 𝒲) (j : 𝒰.J) :
    (f ≫ g).idx j = g.idx (f.idx j) := rfl


@[simp]
lemma Cover.comp_app {X : Scheme.{u}} {𝒰 𝒱 𝒲 : X.Cover P}
    (f : 𝒰 ⟶ 𝒱) (g : 𝒱 ⟶ 𝒲) (j : 𝒰.J) :
    (f ≫ g).app j = f.app j ≫ g.app _ := rfl


