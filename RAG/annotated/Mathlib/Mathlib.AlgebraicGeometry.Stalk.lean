/--
A morphism from `Spec(O_x)` to `X`, which is defined with the help of an affine open
neighborhood `U` of `x`.
-/
noncomputable def IsAffineOpen.fromSpecStalk
    {X : Scheme} {U : X.Opens} (hU : IsAffineOpen U) {x : X} (hxU : x ∈ U) :
    Spec (X.presheaf.stalk x) ⟶ X :=
  Spec.map (X.presheaf.germ _ x hxU) ≫ hU.fromSpec


/--
The morphism from `Spec(O_x)` to `X` given by `IsAffineOpen.fromSpec` does not depend on the affine
open neighborhood of `x` we choose.
-/
theorem IsAffineOpen.fromSpecStalk_eq (x : X) (hxU : x ∈ U) (hxV : x ∈ V) :
    hU.fromSpecStalk hxU = hV.fromSpecStalk hxV := by
  obtain ⟨U', h₁, h₂, h₃ : U' ≤ U ⊓ V⟩ :=
    Opens.isBasis_iff_nbhd.mp (isBasis_affine_open X) (show x ∈ U ⊓ V from ⟨hxU, hxV⟩)
  /-
    case intro.intro.intro
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    hV : AlgebraicGeometry.IsAffineOpen V
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    hxV : Membership.mem V x
    U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    h₁ : Membership.mem X.affineOpens U'
    h₂ : Membership.mem U' x
    h₃ : LE.le U' (Min.min U V)
    ⊢ Eq (hU.fromSpecStalk hxU) (hV.fromSpecStalk hxV)
  -/
  transitivity fromSpecStalk h₁ h₂
    /-
      X : AlgebraicGeometry.Scheme
      U V : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hV : AlgebraicGeometry.IsAffineOpen V
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hxV : Membership.mem V x
      U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      h₁ : Membership.mem X.affineOpens U'
      h₂ : Membership.mem U' x
      h₃ : LE.le U' (Min.min U V)
      ⊢ Eq (hU.fromSpecStalk hxU) (AlgebraicGeometry.IsAffineOpen.fromSpecStalk h₁ h₂)
    -/
  · delta fromSpecStalk
    /-
      X : AlgebraicGeometry.Scheme
      U V : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hV : AlgebraicGeometry.IsAffineOpen V
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hxV : Membership.mem V x
      U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      h₁ : Membership.mem X.affineOpens U'
      h₂ : Membership.mem U' x
      h₃ : LE.le U' (Min.min U V)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (X.preshe …
    -/
    rw [← hU.map_fromSpec h₁ (homOfLE <| h₃.trans inf_le_left).op]
    /-
      X : AlgebraicGeometry.Scheme
      U V : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hV : AlgebraicGeometry.IsAffineOpen V
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hxV : Membership.mem V x
      U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      h₁ : Membership.mem X.affineOpens U'
      h₂ : Membership.mem U' x
      h₃ : LE.le U' (Min.min U V)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (X.preshe …
    -/
    erw [← Scheme.Spec_map (X.presheaf.map _).op, ← Scheme.Spec_map (X.presheaf.germ _ x h₂).op]
    rw [← Functor.map_comp_assoc, ← op_comp, TopCat.Presheaf.germ_res, Scheme.Spec_map,
      Quiver.Hom.unop_op]
    /-
      X : AlgebraicGeometry.Scheme
      U V : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hV : AlgebraicGeometry.IsAffineOpen V
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hxV : Membership.mem V x
      U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      h₁ : Membership.mem X.affineOpens U'
      h₂ : Membership.mem U' x
      h₃ : LE.le U' (Min.min U V)
      ⊢ Eq (AlgebraicGeometry.IsAffineOpen.fromSpecStalk h₁ h₂) (hV.fromSpecStalk hxV)
    -/
  · delta fromSpecStalk
    /-
      X : AlgebraicGeometry.Scheme
      U V : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hV : AlgebraicGeometry.IsAffineOpen V
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hxV : Membership.mem V x
      U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      h₁ : Membership.mem X.affineOpens U'
      h₂ : Membership.mem U' x
      h₃ : LE.le U' (Min.min U V)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (X.preshe …
    -/
    rw [← hV.map_fromSpec h₁ (homOfLE <| h₃.trans inf_le_right).op]
    /-
      X : AlgebraicGeometry.Scheme
      U V : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hV : AlgebraicGeometry.IsAffineOpen V
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hxV : Membership.mem V x
      U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      h₁ : Membership.mem X.affineOpens U'
      h₂ : Membership.mem U' x
      h₃ : LE.le U' (Min.min U V)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (X.preshe …
    -/
    erw [← Scheme.Spec_map (X.presheaf.map _).op, ← Scheme.Spec_map (X.presheaf.germ _ x h₂).op]
    rw [← Functor.map_comp_assoc, ← op_comp, TopCat.Presheaf.germ_res, Scheme.Spec_map,
      Quiver.Hom.unop_op]


/--
If `x` is a point of `X`, this is the canonical morphism from `Spec(O_x)` to `X`.
-/
noncomputable def Scheme.fromSpecStalk (X : Scheme) (x : X) :
    Spec (X.presheaf.stalk x) ⟶ X :=
  (isAffineOpen_opensRange (X.affineOpenCover.map x)).fromSpecStalk (X.affineOpenCover.covers x)


@[simps over] noncomputable
instance (X : Scheme.{u}) (x : X) : (Spec (X.presheaf.stalk x)).Over X := ⟨X.fromSpecStalk x⟩


@[simps! over] noncomputable
instance (X : Scheme.{u}) (x : X) : (Spec (X.presheaf.stalk x)).CanonicallyOver X where


@[simp]
theorem IsAffineOpen.fromSpecStalk_eq_fromSpecStalk {x : X} (hxU : x ∈ U) :
    hU.fromSpecStalk hxU = X.fromSpecStalk x := fromSpecStalk_eq ..


instance IsAffineOpen.fromSpecStalk_isPreimmersion {X : Scheme.{u}} {U : Opens X}
    (hU : IsAffineOpen U) (x : X) (hx : x ∈ U) : IsPreimmersion (hU.fromSpecStalk hx) := by
  /-
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    U✝ V : X✝.Opens
    hU✝ : AlgebraicGeometry.IsAffineOpen U✝
    hV : AlgebraicGeometry.IsAffineOpen V
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ AlgebraicGeometry.IsPreimmersion (hU.fromSpecStalk hx)
  -/
  dsimp [IsAffineOpen.fromSpecStalk]
  haveI : IsPreimmersion (Spec.map (X.presheaf.germ U x hx)) :=
    letI : Algebra Γ(X, U) (X.presheaf.stalk x) := (X.presheaf.germ U x hx).hom.toAlgebra
    haveI := hU.isLocalization_stalk ⟨x, hx⟩
    IsPreimmersion.of_isLocalization (R := Γ(X, U)) (S := X.presheaf.stalk x)
      (hU.primeIdealOf ⟨x, hx⟩).asIdeal.primeCompl
  /-
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    U✝ V : X✝.Opens
    hU✝ : AlgebraicGeometry.IsAffineOpen U✝
    hV : AlgebraicGeometry.IsAffineOpen V
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    this : AlgebraicGeometry.IsPreimmersion (AlgebraicGeometry.Spec.map (X.preshea …
    ⊢ AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp (Algebr …
  -/
  apply IsPreimmersion.comp
  /-
    🎉 no goals
  -/


instance {X : Scheme.{u}} (x : X) : IsPreimmersion (X.fromSpecStalk x) :=
  IsAffineOpen.fromSpecStalk_isPreimmersion _ _ _


lemma IsAffineOpen.fromSpecStalk_closedPoint {U : Opens X} (hU : IsAffineOpen U)
    {x : X} (hxU : x ∈ U) :
    (hU.fromSpecStalk hxU).base (closedPoint (X.presheaf.stalk x)) = x := by
  /-
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    ⊢ Eq ((hU.fromSpecStalk hxU).base (IsLocalRing.closedPoint ↑(X.presheaf.stalk  …
  -/
  rw [IsAffineOpen.fromSpecStalk, Scheme.comp_base_apply]
  /-
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    ⊢ Eq (hU.fromSpec.base ((AlgebraicGeometry.Spec.map (X.presheaf.germ U x hxU)) …
  -/
  rw [← hU.primeIdealOf_eq_map_closedPoint ⟨x, hxU⟩, hU.fromSpec_primeIdealOf ⟨x, hxU⟩]
  /-
    🎉 no goals
  -/


@[simp]
lemma fromSpecStalk_closedPoint {x : X} :
    (X.fromSpecStalk x).base (closedPoint (X.presheaf.stalk x)) = x :=
  IsAffineOpen.fromSpecStalk_closedPoint _ _


lemma fromSpecStalk_app {x : X} (hxU : x ∈ U) :
    (X.fromSpecStalk x).app U =
      X.presheaf.germ U x hxU ≫
        (ΓSpecIso (X.presheaf.stalk x)).inv ≫
          (Spec (X.presheaf.stalk x)).presheaf.map (homOfLE le_top).op := by
  obtain ⟨_, ⟨V : X.Opens, hV, rfl⟩, hxV, hVU⟩ := (isBasis_affine_open X).exists_subset_of_mem_open
    hxU U.2
  rw [← hV.fromSpecStalk_eq_fromSpecStalk hxV, IsAffineOpen.fromSpecStalk, Scheme.comp_app,
    hV.fromSpec_app_of_le _ hVU, ← X.presheaf.germ_res (homOfLE hVU) x hxV]
  /-
    case intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    V : X.Opens
    hV : Membership.mem X.affineOpens V
    hxV : Membership.mem (↑V) x
    hVU : HasSubset.Subset ↑V ↑U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [Category.assoc, ← ΓSpecIso_inv_naturality_assoc]
  /-
    🎉 no goals
  -/


lemma fromSpecStalk_appTop {x : X} :
    (X.fromSpecStalk x).appTop =
      X.presheaf.germ ⊤ x trivial ≫
        (ΓSpecIso (X.presheaf.stalk x)).inv ≫
          (Spec (X.presheaf.stalk x)).presheaf.map (homOfLE le_top).op :=
  fromSpecStalk_app ..


@[reassoc (attr := simp)]
lemma Spec_map_stalkSpecializes_fromSpecStalk {x y : X} (h : x ⤳ y) :
    Spec.map (X.presheaf.stalkSpecializes h) ≫ X.fromSpecStalk y = X.fromSpecStalk x := by
  obtain ⟨_, ⟨U, hU, rfl⟩, hyU, -⟩ :=
    (isBasis_affine_open X).exists_subset_of_mem_open (Set.mem_univ y) isOpen_univ
  /-
    case intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    x y : ↑↑X.toPresheafedSpace
    h : Specializes x y
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hU : Membership.mem X.affineOpens U
    hyU : Membership.mem (↑U) y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (X.preshe …
  -/
  have hxU : x ∈ U := h.mem_open U.2 hyU
  rw [← hU.fromSpecStalk_eq_fromSpecStalk hyU, ← hU.fromSpecStalk_eq_fromSpecStalk hxU,
    IsAffineOpen.fromSpecStalk, IsAffineOpen.fromSpecStalk, ← Category.assoc, ← Spec.map_comp,
    TopCat.Presheaf.germ_stalkSpecializes]


instance {x y : X} (h : x ⤳ y) : (Spec.map (X.presheaf.stalkSpecializes h)).IsOver X where


@[reassoc (attr := simp)]
lemma Spec_map_stalkMap_fromSpecStalk {x} :
    Spec.map (f.stalkMap x) ≫ Y.fromSpecStalk _ = X.fromSpecStalk x ≫ f := by
  obtain ⟨_, ⟨U, hU, rfl⟩, hxU, -⟩ := (isBasis_affine_open Y).exists_subset_of_mem_open
    (Set.mem_univ (f.base x)) isOpen_univ
  obtain ⟨_, ⟨V, hV, rfl⟩, hxV, hVU⟩ := (isBasis_affine_open X).exists_subset_of_mem_open
    hxU (f ⁻¹ᵁ U).2
  rw [← hU.fromSpecStalk_eq_fromSpecStalk hxU, ← hV.fromSpecStalk_eq_fromSpecStalk hxV,
    IsAffineOpen.fromSpecStalk, ← Spec.map_comp_assoc, Scheme.stalkMap_germ f _ x hxU,
    IsAffineOpen.fromSpecStalk, Spec.map_comp_assoc, ← X.presheaf.germ_res (homOfLE hVU) x hxV,
    Spec.map_comp_assoc, Category.assoc, ← Spec.map_comp_assoc (f.app _),
      Hom.app_eq_appLE, Hom.appLE_map, IsAffineOpen.Spec_map_appLE_fromSpec]


instance [X.Over Y] {x} : Spec.map ((X ↘ Y).stalkMap x) |>.IsOver Y where


lemma Spec_fromSpecStalk (R : CommRingCat) (x) :
    (Spec R).fromSpecStalk x =
      Spec.map ((ΓSpecIso R).inv ≫ (Spec R).presheaf.germ ⊤ x trivial) := by
  rw [← (isAffineOpen_top (Spec R)).fromSpecStalk_eq_fromSpecStalk (x := x) trivial,
    IsAffineOpen.fromSpecStalk, IsAffineOpen.fromSpec_top, isoSpec_Spec_inv,
    ← Spec.map_comp]

-- This is not a simp lemma to respect the abstraction boundaries

/-- A variant of `Spec_fromSpecStalk` that breaks abstraction boundaries. -/
lemma Spec_fromSpecStalk' (R : CommRingCat) (x) :
    (Spec R).fromSpecStalk x = Spec.map (StructureSheaf.toStalk R _) :=
  Spec_fromSpecStalk _ _


@[stacks 01J7]
lemma range_fromSpecStalk {x : X} :
    Set.range (X.fromSpecStalk x).base = { y | y ⤳ x } := by
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (Set.range ⇑(X.fromSpecStalk x).base) (setOf fun y => Specializes y x)
  -/
  ext y
  /-
    case h
    X : AlgebraicGeometry.Scheme
    x y : ↑↑X.toPresheafedSpace
    ⊢ Iff (Membership.mem (Set.range ⇑(X.fromSpecStalk x).base) y) (Membership.mem …
  -/
  constructor
    /-
      case h.mp
      X : AlgebraicGeometry.Scheme
      x y : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑(X.fromSpecStalk x).base) y → Membership.mem (set …
    -/
  · rintro ⟨y, rfl⟩
    exact ((IsLocalRing.specializes_closedPoint y).map (X.fromSpecStalk x).base.2).trans
      (specializes_of_eq fromSpecStalk_closedPoint)
    /-
      case h.mpr
      X : AlgebraicGeometry.Scheme
      x y : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (setOf fun y => Specializes y x) y → Membership.mem (Set.rang …
    -/
  · rintro (hy : y ⤳ x)
    /-
      case h.mpr
      X : AlgebraicGeometry.Scheme
      x y : ↑↑X.toPresheafedSpace
      hy : Specializes y x
      ⊢ Membership.mem (Set.range ⇑(X.fromSpecStalk x).base) y
    -/
    have := fromSpecStalk_closedPoint (x := y)
    /-
      case h.mpr
      X : AlgebraicGeometry.Scheme
      x y : ↑↑X.toPresheafedSpace
      hy : Specializes y x
      this : Eq ((X.fromSpecStalk y).base (IsLocalRing.closedPoint ↑(X.presheaf.stal …
      ⊢ Membership.mem (Set.range ⇑(X.fromSpecStalk x).base) y
    -/
    rw [← Spec_map_stalkSpecializes_fromSpecStalk hy] at this
    /-
      case h.mpr
      X : AlgebraicGeometry.Scheme
      x y : ↑↑X.toPresheafedSpace
      hy : Specializes y x
      this : Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (X. …
      ⊢ Membership.mem (Set.range ⇑(X.fromSpecStalk x).base) y
    -/
    exact ⟨_, this⟩
    /-
      🎉 no goals
    -/


/-- The canonical map `Spec 𝒪_{X, x} ⟶ U` given `x ∈ U ⊆ X`. -/
noncomputable
def Opens.fromSpecStalkOfMem {X : Scheme.{u}} (U : X.Opens) (x : X) (hxU : x ∈ U) :
    Spec (X.presheaf.stalk x) ⟶ U :=
  Spec.map (inv (U.ι.stalkMap ⟨x, hxU⟩)) ≫ U.toScheme.fromSpecStalk ⟨x, hxU⟩


@[reassoc (attr := simp)]
lemma Opens.fromSpecStalkOfMem_ι {X : Scheme.{u}} (U : X.Opens) (x : X) (hxU : x ∈ U) :
    U.fromSpecStalkOfMem x hxU ≫ U.ι = X.fromSpecStalk x := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (U.fromSpecStalkOfMem x hxU) U.ι) (X. …
  -/
  simp only [Opens.fromSpecStalkOfMem, Spec.map_inv, Category.assoc, IsIso.inv_comp_eq]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((↑U).fromSpecStalk ⟨x, hxU⟩) U.ι) (C …
  -/
  exact (Scheme.Spec_map_stalkMap_fromSpecStalk U.ι (x := ⟨x, hxU⟩)).symm
  /-
    🎉 no goals
  -/


instance {X : Scheme.{u}} (U : X.Opens) (x : X) (hxU : x ∈ U) :
    (U.fromSpecStalkOfMem x hxU).IsOver X where


@[reassoc]
lemma fromSpecStalk_toSpecΓ (X : Scheme.{u}) (x : X) :
    X.fromSpecStalk x ≫ X.toSpecΓ = Spec.map (X.presheaf.germ ⊤ x trivial) := by
  rw [Scheme.toSpecΓ_naturality, ← SpecMap_ΓSpecIso_hom, ← Spec.map_comp,
    Scheme.fromSpecStalk_appTop]
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (AlgebraicGeometry.Spec.map (CategoryTheory.CategoryStruct.comp (Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma Opens.fromSpecStalkOfMem_toSpecΓ {X : Scheme.{u}} (U : X.Opens) (x : X) (hxU : x ∈ U) :
    U.fromSpecStalkOfMem x hxU ≫ U.toSpecΓ = Spec.map (X.presheaf.germ U x hxU) := by
  rw [fromSpecStalkOfMem, Opens.toSpecΓ, Category.assoc, fromSpecStalk_toSpecΓ_assoc,
    ← Spec.map_comp, ← Spec.map_comp]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    ⊢ Eq (AlgebraicGeometry.Spec.map (CategoryTheory.CategoryStruct.comp (Category …
  -/
  congr 1
  /-
    case e_f
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp U …
  -/
  rw [IsIso.comp_inv_eq, Iso.inv_comp_eq]
  /-
    case e_f
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    ⊢ Eq ((↑U).presheaf.germ Top.top ⟨x, hxU⟩ trivial) (CategoryTheory.CategoryStr …
  -/
  erw [stalkMap_germ U.ι U ⟨x, hxU⟩]
  /-
    case e_f
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    ⊢ Eq ((↑U).presheaf.germ Top.top ⟨x, hxU⟩ trivial) (CategoryTheory.CategoryStr …
  -/
  rw [Opens.ι_app, Opens.topIso_hom, ← Functor.map_comp_assoc]
  /-
    case e_f
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    ⊢ Eq ((↑U).presheaf.germ Top.top ⟨x, hxU⟩ trivial) (CategoryTheory.CategoryStr …
  -/
  exact (U.toScheme.presheaf.germ_res (homOfLE le_top) ⟨x, hxU⟩ (U := U.ι ⁻¹ᵁ U) hxU).symm
  /-
    🎉 no goals
  -/


/-- For a local ring `(R, 𝔪)`,
this is the isomorphism between the stalk of `Spec R` at `𝔪` and `R`. -/
noncomputable
def stalkClosedPointIso :
    (Spec R).presheaf.stalk (closedPoint R) ≅ R :=
  StructureSheaf.stalkIso _ _ ≪≫ (IsLocalization.atUnits R
      (closedPoint R).asIdeal.primeCompl fun _ ↦ not_not.mp).toRingEquiv.toCommRingCatIso.symm


lemma stalkClosedPointIso_inv :
    (stalkClosedPointIso R).inv = StructureSheaf.toStalk R _ := by
  /-
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    ⊢ Eq (AlgebraicGeometry.stalkClosedPointIso R).inv (AlgebraicGeometry.Structur …
  -/
  ext x
  /-
    case hf.a
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    x : ↑R
    ⊢ Eq ((AlgebraicGeometry.stalkClosedPointIso R).inv.hom x) ((AlgebraicGeometry …
  -/
  exact StructureSheaf.localizationToStalk_of _ _ _
  /-
    🎉 no goals
  -/


lemma ΓSpecIso_hom_stalkClosedPointIso_inv :
    (Scheme.ΓSpecIso R).hom ≫ (stalkClosedPointIso R).inv =
      (Spec R).presheaf.germ ⊤ (closedPoint _) trivial := by
  /-
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.ΓSpecIso R) …
  -/
  rw [stalkClosedPointIso_inv, ← Iso.eq_inv_comp]
  /-
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.toStalk (↑R) (IsLocalRing.closedPoint ↑ …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma germ_stalkClosedPointIso_hom :
    (Spec R).presheaf.germ ⊤ (closedPoint _) trivial ≫ (stalkClosedPointIso R).hom =
      (Scheme.ΓSpecIso R).hom := by
  /-
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Spec R).presheaf. …
  -/
  rw [← ΓSpecIso_hom_stalkClosedPointIso_inv, Category.assoc, Iso.inv_hom_id, Category.comp_id]
  /-
    🎉 no goals
  -/


lemma Spec_stalkClosedPointIso :
    Spec.map (stalkClosedPointIso R).inv = (Spec R).fromSpecStalk (closedPoint R) := by
  /-
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    ⊢ Eq (AlgebraicGeometry.Spec.map (AlgebraicGeometry.stalkClosedPointIso R).inv …
  -/
  rw [stalkClosedPointIso_inv, Scheme.Spec_fromSpecStalk']
  /-
    🎉 no goals
  -/


/--
Given a local ring `(R, 𝔪)` and a morphism `f : Spec R ⟶ X`,
they induce a (local) ring homomorphism `φ : 𝒪_{X, f 𝔪} ⟶ R`.

This is inverse to `φ ↦ Spec.map φ ≫ X.fromSpecStalk (f 𝔪)`. See `SpecToEquivOfLocalRing`.
-/
noncomputable
def stalkClosedPointTo :
    X.presheaf.stalk (f.base (closedPoint R)) ⟶ R :=
  f.stalkMap (closedPoint R) ≫ (stalkClosedPointIso R).hom


instance isLocalHom_stalkClosedPointTo :
    IsLocalHom (stalkClosedPointTo f).hom :=
  inferInstanceAs <| IsLocalHom (f.stalkMap (closedPoint R) ≫ (stalkClosedPointIso R).hom).hom


/-- Copy of `isLocalHom_stalkClosedPointTo` which unbundles the comm ring.

Useful for use in combination with `CommRingCat.of K` for a field `K`.
-/
instance isLocalHom_stalkClosedPointTo' {R : Type u} [CommRing R] [IsLocalRing R]
    (f : Spec (.of R) ⟶ X) :
    IsLocalHom (stalkClosedPointTo f).hom :=
  isLocalHom_stalkClosedPointTo f


lemma preimage_eq_top_of_closedPoint_mem
    {U : Opens X} (hU : f.base (closedPoint R) ∈ U) : f ⁻¹ᵁ U = ⊤ :=
  IsLocalRing.closed_point_mem_iff.mp hU


lemma stalkClosedPointTo_comp (g : X ⟶ Y) :
    stalkClosedPointTo (f ≫ g) = g.stalkMap _ ≫ stalkClosedPointTo f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    g : Quiver.Hom X Y
    ⊢ Eq (AlgebraicGeometry.Scheme.stalkClosedPointTo (CategoryTheory.CategoryStru …
  -/
  rw [stalkClosedPointTo, Scheme.stalkMap_comp]
  /-
    X Y : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    g : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  exact Category.assoc _ _ _
  /-
    🎉 no goals
  -/


lemma germ_stalkClosedPointTo_Spec {R S : CommRingCat} [IsLocalRing S] (φ : R ⟶ S):
    (Spec R).presheaf.germ ⊤ _ trivial ≫ stalkClosedPointTo (Spec.map φ) =
      (ΓSpecIso R).hom ≫ φ := by
  rw [stalkClosedPointTo, Scheme.stalkMap_germ_assoc, ← Iso.inv_comp_eq,
    ← ΓSpecIso_inv_naturality_assoc]
  /-
    R S : CommRingCat
    inst✝ : IsLocalRing ↑S
    φ : Quiver.Hom R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
  -/
  simp_rw [Opens.map_top]
  /-
    R S : CommRingCat
    inst✝ : IsLocalRing ↑S
    φ : Quiver.Hom R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
  -/
  rw [germ_stalkClosedPointIso_hom, Iso.inv_hom_id, Category.comp_id]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma germ_stalkClosedPointTo (U : Opens X) (hU : f.base (closedPoint R) ∈ U) :
    X.presheaf.germ U _ hU ≫ stalkClosedPointTo f = f.app U ≫
      ((Spec R).presheaf.mapIso (eqToIso (preimage_eq_top_of_closedPoint_mem f hU).symm).op ≪≫
        ΓSpecIso R).hom := by
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    U : X.Opens
    hU : Membership.mem U (f.base (IsLocalRing.closedPoint ↑R))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.germ U (f.base (IsLocalRi …
  -/
  rw [stalkClosedPointTo, Scheme.stalkMap_germ_assoc, Iso.trans_hom]
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    U : X.Opens
    hU : Membership.mem U (f.base (IsLocalRing.closedPoint ↑R))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app f U …
  -/
  congr 1
  /-
    case e_a
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    U : X.Opens
    hU : Membership.mem U (f.base (IsLocalRing.closedPoint ↑R))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Spec R).presheaf. …
  -/
  rw [← Iso.eq_comp_inv, Category.assoc, ΓSpecIso_hom_stalkClosedPointIso_inv]
  simp only [TopCat.Presheaf.pushforward_obj_obj, Functor.mapIso_hom, Iso.op_hom, eqToIso.hom,
    TopCat.Presheaf.germ_res]


@[reassoc]
lemma germ_stalkClosedPointTo_Spec_fromSpecStalk
    {x : X} (f : X.presheaf.stalk x ⟶ R) [IsLocalHom f.hom] (U : Opens X) (hU) :
    X.presheaf.germ U _ hU ≫ stalkClosedPointTo (Spec.map f ≫ X.fromSpecStalk x) =
                              /-
                                X Y : AlgebraicGeometry.Scheme
                                f✝¹ : Quiver.Hom X Y
                                U✝ V : X.Opens
                                hU✝ : AlgebraicGeometry.IsAffineOpen U✝
                                hV : AlgebraicGeometry.IsAffineOpen V
                                R : CommRingCat
                                inst✝¹ : IsLocalRing ↑R
                                f✝ : Quiver.Hom (AlgebraicGeometry.Spec R) X
                                x : ↑↑X.toPresheafedSpace
                                f : Quiver.Hom (X.presheaf.stalk x) R
                                inst✝ : IsLocalHom f.hom
                                U : X.Opens
                                hU : Membership.mem U ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
                                ⊢ Membership.mem U x
                              -/
      X.presheaf.germ U x (by simpa using hU) ≫ f := by
                              /-
                                🎉 no goals
                              -/
  have : (Spec.map f ≫ X.fromSpecStalk x).base (closedPoint R) = x := by
    rw [comp_base_apply, Spec_closedPoint, fromSpecStalk_closedPoint]
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝¹ : IsLocalRing ↑R
    x : ↑↑X.toPresheafedSpace
    f : Quiver.Hom (X.presheaf.stalk x) R
    inst✝ : IsLocalHom f.hom
    U : X.Opens
    hU : Membership.mem U ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
    this : Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map f)  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.germ U ((CategoryTheory.C …
  -/
  have : x ∈ U := this ▸ hU
  simp only [TopCat.Presheaf.stalkCongr_hom, TopCat.Presheaf.germ_stalkSpecializes_assoc,
    germ_stalkClosedPointTo, comp_app,
    fromSpecStalk_app (X := X) (x := x) this, Category.assoc, Iso.trans_hom,
    Functor.mapIso_hom, Hom.naturality_assoc, ← Functor.map_comp_assoc,
    (Spec.map f).app_eq_appLE, Hom.appLE_map_assoc, Hom.map_appLE_assoc]
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝¹ : IsLocalRing ↑R
    x : ↑↑X.toPresheafedSpace
    f : Quiver.Hom (X.presheaf.stalk x) R
    inst✝ : IsLocalHom f.hom
    U : X.Opens
    hU : Membership.mem U ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
    this✝ : Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map f) …
    this : Membership.mem U x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.germ U x this) (CategoryT …
  -/
  simp_rw [← Opens.map_top (Spec.map f).base]
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝¹ : IsLocalRing ↑R
    x : ↑↑X.toPresheafedSpace
    f : Quiver.Hom (X.presheaf.stalk x) R
    inst✝ : IsLocalHom f.hom
    U : X.Opens
    hU : Membership.mem U ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
    this✝ : Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map f) …
    this : Membership.mem U x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.germ U x this) (CategoryT …
  -/
  rw [← (Spec.map f).app_eq_appLE, ΓSpecIso_naturality, Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


lemma stalkClosedPointTo_fromSpecStalk (x : X) :
    stalkClosedPointTo (X.fromSpecStalk x) =
                                 /-
                                   X Y : AlgebraicGeometry.Scheme
                                   f✝ : Quiver.Hom X Y
                                   U V : X.Opens
                                   hU : AlgebraicGeometry.IsAffineOpen U
                                   hV : AlgebraicGeometry.IsAffineOpen V
                                   R : CommRingCat
                                   inst✝ : IsLocalRing ↑R
                                   f : Quiver.Hom (AlgebraicGeometry.Spec R) X
                                   x : ↑↑X.toPresheafedSpace
                                   ⊢ Inseparable ((X.fromSpecStalk x).base (IsLocalRing.closedPoint ↑(X.presheaf. …
                                 -/
      (X.presheaf.stalkCongr (by rw [fromSpecStalk_closedPoint]; rfl)).hom := by
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (AlgebraicGeometry.Scheme.stalkClosedPointTo (X.fromSpecStalk x)) (X.pres …
  -/
  refine TopCat.Presheaf.stalk_hom_ext _ fun U hxU ↦ ?_
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxU : Membership.mem U ((X.fromSpecStalk x).base (IsLocalRing.closedPoint ↑(X. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.germ U ((X.fromSpecStalk  …
  -/
  simp only [TopCat.Presheaf.stalkCongr_hom, TopCat.Presheaf.germ_stalkSpecializes, id_eq]
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxU : Membership.mem U ((X.fromSpecStalk x).base (IsLocalRing.closedPoint ↑(X. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.germ U ((X.fromSpecStalk  …
  -/
  have : X.fromSpecStalk x = Spec.map (𝟙 (X.presheaf.stalk x)) ≫ X.fromSpecStalk x := by simp
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxU : Membership.mem U ((X.fromSpecStalk x).base (IsLocalRing.closedPoint ↑(X. …
    this : Eq (X.fromSpecStalk x) (CategoryTheory.CategoryStruct.comp (AlgebraicGe …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.germ U ((X.fromSpecStalk  …
  -/
  convert germ_stalkClosedPointTo_Spec_fromSpecStalk (𝟙 (X.presheaf.stalk x)) U hxU
  /-
    🎉 no goals
  -/


@[reassoc]
lemma Spec_stalkClosedPointTo_fromSpecStalk :
    Spec.map (stalkClosedPointTo f) ≫ X.fromSpecStalk _ = f := by
  obtain ⟨_, ⟨U, hU, rfl⟩, hxU, -⟩ := (isBasis_affine_open X).exists_subset_of_mem_open
    (Set.mem_univ (f.base (closedPoint R))) isOpen_univ
  have := IsAffineOpen.Spec_map_appLE_fromSpec f hU (isAffineOpen_top _)
    (preimage_eq_top_of_closedPoint_mem f hxU).ge
  /-
    case intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsLocalRing ↑R
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hU : Membership.mem X.affineOpens U
    hxU : Membership.mem (↑U) (f.base (IsLocalRing.closedPoint ↑R))
    this : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Alg …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  rw [IsAffineOpen.fromSpec_top, Iso.eq_inv_comp, isoSpec_Spec_hom] at this
  rw [← hU.fromSpecStalk_eq_fromSpecStalk hxU, IsAffineOpen.fromSpecStalk, ← Spec.map_comp_assoc,
    germ_stalkClosedPointTo]
  simpa only [Iso.trans_hom, Functor.mapIso_hom, Iso.op_hom, Category.assoc,
    Hom.app_eq_appLE, Hom.appLE_map_assoc, Spec.map_comp_assoc]


omit [IsLocalRing R] in
/-- useful lemma for applications of `SpecToEquivOfLocalRing` -/
lemma SpecToEquivOfLocalRing_eq_iff
    {f₁ f₂ : Σ x, { f : X.presheaf.stalk x ⟶ R // IsLocalHom f.hom }} :
    f₁ = f₂ ↔ ∃ h₁ : f₁.1 = f₂.1, f₁.2.1 =
                                 /-
                                   X Y : AlgebraicGeometry.Scheme
                                   f : Quiver.Hom X Y
                                   U V : X.Opens
                                   hU : AlgebraicGeometry.IsAffineOpen U
                                   hV : AlgebraicGeometry.IsAffineOpen V
                                   R : CommRingCat
                                   inst✝ : IsLocalRing ↑R
                                   f₁ f₂ : Sigma fun x => Subtype fun f => IsLocalHom f.hom
                                   h₁ : Eq f₁.fst f₂.fst
                                   ⊢ Inseparable f₁.fst f₂.fst
                                 -/
      (X.presheaf.stalkCongr (by rw [h₁]; rfl)).hom ≫ f₂.2.1 := by
                                          /-
                                            🎉 no goals
                                          -/
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    f₁ f₂ : Sigma fun x => Subtype fun f => IsLocalHom f.hom
    ⊢ Iff (Eq f₁ f₂) (Exists fun h₁ => Eq (↑f₁.snd) (CategoryTheory.CategoryStruct …
  -/
  constructor
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      R : CommRingCat
      f₁ f₂ : Sigma fun x => Subtype fun f => IsLocalHom f.hom
      ⊢ Eq f₁ f₂ → Exists fun h₁ => Eq (↑f₁.snd) (CategoryTheory.CategoryStruct.comp …
    -/
  · rintro rfl; simp
                /-
                  🎉 no goals
                -/
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      R : CommRingCat
      f₁ f₂ : Sigma fun x => Subtype fun f => IsLocalHom f.hom
      ⊢ (Exists fun h₁ => Eq (↑f₁.snd) (CategoryTheory.CategoryStruct.comp (X.preshe …
    -/
  · obtain ⟨x₁, ⟨f₁, h₁⟩⟩ := f₁
    /-
      case mpr.mk.mk
      X : AlgebraicGeometry.Scheme
      R : CommRingCat
      f₂ : Sigma fun x => Subtype fun f => IsLocalHom f.hom
      x₁ : ↑↑X.toPresheafedSpace
      f₁ : Quiver.Hom (X.presheaf.stalk x₁) R
      h₁ : IsLocalHom f₁.hom
      ⊢ (Exists fun h₁_1 => Eq (↑⟨x₁, ⟨f₁, h₁⟩⟩.snd) (CategoryTheory.CategoryStruct. …
    -/
    obtain ⟨x₂, ⟨f₂, h₂⟩⟩ := f₂
    /-
      case mpr.mk.mk.mk.mk
      X : AlgebraicGeometry.Scheme
      R : CommRingCat
      x₁ : ↑↑X.toPresheafedSpace
      f₁ : Quiver.Hom (X.presheaf.stalk x₁) R
      h₁ : IsLocalHom f₁.hom
      x₂ : ↑↑X.toPresheafedSpace
      f₂ : Quiver.Hom (X.presheaf.stalk x₂) R
      h₂ : IsLocalHom f₂.hom
      ⊢ (Exists fun h₁_1 => Eq (↑⟨x₁, ⟨f₁, h₁⟩⟩.snd) (CategoryTheory.CategoryStruct. …
    -/
    rintro ⟨rfl : x₁ = x₂, e : f₁ = _⟩
    /-
      case mpr.mk.mk.mk.mk.intro
      X : AlgebraicGeometry.Scheme
      R : CommRingCat
      x₁ : ↑↑X.toPresheafedSpace
      f₁ : Quiver.Hom (X.presheaf.stalk x₁) R
      h₁ : IsLocalHom f₁.hom
      f₂ : Quiver.Hom (X.presheaf.stalk x₁) R
      h₂ : IsLocalHom f₂.hom
      e : Eq f₁ (CategoryTheory.CategoryStruct.comp (X.presheaf.stalkCongr ⋯).hom ↑⟨ …
      ⊢ Eq ⟨x₁, ⟨f₁, h₁⟩⟩ ⟨x₁, ⟨f₂, h₂⟩⟩
    -/
    simp [e]
    /-
      🎉 no goals
    -/


/--
Given a local ring `R` and scheme `X`, morphisms `Spec R ⟶ X` corresponds to pairs
`(x, f)` where `x : X` and `f : 𝒪_{X, x} ⟶ R` is a local ring homomorphism.
-/
@[simps]
noncomputable
def SpecToEquivOfLocalRing :
    (Spec R ⟶ X) ≃ Σ x, { f : X.presheaf.stalk x ⟶ R // IsLocalHom f.hom } where
  toFun f := ⟨f.base (closedPoint R), Scheme.stalkClosedPointTo f, inferInstance⟩
  invFun xf := Spec.map xf.2.1 ≫ X.fromSpecStalk xf.1
  left_inv := Scheme.Spec_stalkClosedPointTo_fromSpecStalk
  right_inv xf := by
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U V : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hV : AlgebraicGeometry.IsAffineOpen V
      R : CommRingCat
      inst✝ : IsLocalRing ↑R
      xf : Sigma fun x => Subtype fun f => IsLocalHom f.hom
      ⊢ Eq ((fun f => ⟨f.base (IsLocalRing.closedPoint ↑R), ⟨AlgebraicGeometry.Schem …
    -/
    obtain ⟨x, ⟨f, hf⟩⟩ := xf
    /-
      case mk.mk
      X Y : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X Y
      U V : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hV : AlgebraicGeometry.IsAffineOpen V
      R : CommRingCat
      inst✝ : IsLocalRing ↑R
      x : ↑↑X.toPresheafedSpace
      f : Quiver.Hom (X.presheaf.stalk x) R
      hf : IsLocalHom f.hom
      ⊢ Eq ((fun f => ⟨f.base (IsLocalRing.closedPoint ↑R), ⟨AlgebraicGeometry.Schem …
    -/
    symm
    /-
      case mk.mk
      X Y : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X Y
      U V : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hV : AlgebraicGeometry.IsAffineOpen V
      R : CommRingCat
      inst✝ : IsLocalRing ↑R
      x : ↑↑X.toPresheafedSpace
      f : Quiver.Hom (X.presheaf.stalk x) R
      hf : IsLocalHom f.hom
      ⊢ Eq ⟨x, ⟨f, hf⟩⟩ ((fun f => ⟨f.base (IsLocalRing.closedPoint ↑R), ⟨AlgebraicG …
    -/
    refine SpecToEquivOfLocalRing_eq_iff.mpr ⟨?_, ?_⟩
    · simp only [Scheme.comp_coeBase, TopCat.coe_comp, Function.comp_apply, Spec_closedPoint,
        Scheme.fromSpecStalk_closedPoint]
      /-
        case mk.mk.refine_2
        X Y : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X Y
        U V : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        hV : AlgebraicGeometry.IsAffineOpen V
        R : CommRingCat
        inst✝ : IsLocalRing ↑R
        x : ↑↑X.toPresheafedSpace
        f : Quiver.Hom (X.presheaf.stalk x) R
        hf : IsLocalHom f.hom
        ⊢ Eq (↑⟨x, ⟨f, hf⟩⟩.snd) (CategoryTheory.CategoryStruct.comp (X.presheaf.stalk …
      -/
    · refine TopCat.Presheaf.stalk_hom_ext _ fun U hxU ↦ ?_
      simp only [Scheme.germ_stalkClosedPointTo_Spec_fromSpecStalk,
        TopCat.Presheaf.stalkCongr_hom, TopCat.Presheaf.germ_stalkSpecializes_assoc]


