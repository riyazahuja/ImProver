/-- The canonical map from the underlying set to the prime spectrum of `Γ(X)`. -/
def toΓSpecFun : X → PrimeSpectrum (Γ.obj (op X)) := fun x =>
  comap (X.presheaf.Γgerm x).hom (IsLocalRing.closedPoint (X.presheaf.stalk x))


theorem not_mem_prime_iff_unit_in_stalk (r : Γ.obj (op X)) (x : X) :
    r ∉ (X.toΓSpecFun x).asIdeal ↔ IsUnit (X.presheaf.Γgerm x r) := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    x : ↑X.toTopCat
    ⊢ Iff (Not (Membership.mem (X.toΓSpecFun x).asIdeal r)) (IsUnit ((X.presheaf.Γ …
  -/
  erw [IsLocalRing.mem_maximalIdeal, Classical.not_not]
  /-
    🎉 no goals
  -/


/-- The preimage of a basic open in `Spec Γ(X)` under the unit is the basic
open in `X` defined by the same element (they are equal as sets). -/
theorem toΓSpec_preimage_basicOpen_eq (r : Γ.obj (op X)) :
    X.toΓSpecFun ⁻¹' (basicOpen r).1 = (X.toRingedSpace.basicOpen r).1 := by
      /-
        X : AlgebraicGeometry.LocallyRingedSpace
        r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
        ⊢ Eq (Set.preimage X.toΓSpecFun (PrimeSpectrum.basicOpen r).carrier) (X.toRing …
      -/
      ext
      /-
        case h
        X : AlgebraicGeometry.LocallyRingedSpace
        r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
        x✝ : ↑X.toTopCat
        ⊢ Iff (Membership.mem (Set.preimage X.toΓSpecFun (PrimeSpectrum.basicOpen r).c …
      -/
      dsimp
      /-
        case h
        X : AlgebraicGeometry.LocallyRingedSpace
        r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
        x✝ : ↑X.toTopCat
        ⊢ Iff (Membership.mem (Set.preimage X.toΓSpecFun ↑(PrimeSpectrum.basicOpen r)) …
      -/
      simp only [Set.mem_preimage, SetLike.mem_coe]
      /-
        case h
        X : AlgebraicGeometry.LocallyRingedSpace
        r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
        x✝ : ↑X.toTopCat
        ⊢ Iff (Membership.mem (PrimeSpectrum.basicOpen r) (X.toΓSpecFun x✝)) (Membersh …
      -/
      rw [X.toRingedSpace.mem_top_basicOpen]
      /-
        case h
        X : AlgebraicGeometry.LocallyRingedSpace
        r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
        x✝ : ↑X.toTopCat
        ⊢ Iff (Membership.mem (PrimeSpectrum.basicOpen r) (X.toΓSpecFun x✝)) (IsUnit ( …
      -/
      exact not_mem_prime_iff_unit_in_stalk ..
      /-
        🎉 no goals
      -/


/-- `toΓSpecFun` is continuous. -/
theorem toΓSpec_continuous : Continuous X.toΓSpecFun := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    ⊢ Continuous X.toΓSpecFun
  -/
  rw [isTopologicalBasis_basic_opens.continuous_iff]
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    ⊢ ∀ (s : Set (PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { uno …
  -/
  rintro _ ⟨r, rfl⟩
  /-
    case intro
    X : AlgebraicGeometry.LocallyRingedSpace
    r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    ⊢ IsOpen (Set.preimage X.toΓSpecFun ((fun r => ↑(PrimeSpectrum.basicOpen r)) r))
  -/
  erw [X.toΓSpec_preimage_basicOpen_eq r]
  /-
    case intro
    X : AlgebraicGeometry.LocallyRingedSpace
    r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    ⊢ IsOpen (X.toRingedSpace.basicOpen r).carrier
  -/
  exact (X.toRingedSpace.basicOpen r).2
  /-
    🎉 no goals
  -/


/-- The canonical (bundled) continuous map from the underlying topological
space of `X` to the prime spectrum of its global sections. -/
@[simps]
def toΓSpecBase : X.toTopCat ⟶ Spec.topObj (Γ.obj (op X)) where
  toFun := X.toΓSpecFun
  continuous_toFun := X.toΓSpec_continuous

-- These lemmas have always been bad (https://github.com/leanprover-community/mathlib4/issues/7657), but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing

/-- The preimage in `X` of a basic open in `Spec Γ(X)` (as an open set). -/
abbrev toΓSpecMapBasicOpen : Opens X :=
  (Opens.map X.toΓSpecBase).obj (basicOpen r)


/-- The preimage is the basic open in `X` defined by the same element `r`. -/
theorem toΓSpecMapBasicOpen_eq : X.toΓSpecMapBasicOpen r = X.toRingedSpace.basicOpen r :=
  Opens.ext (X.toΓSpec_preimage_basicOpen_eq r)


/-- The map from the global sections `Γ(X)` to the sections on the (preimage of) a basic open. -/
abbrev toToΓSpecMapBasicOpen :
    X.presheaf.obj (op ⊤) ⟶ X.presheaf.obj (op <| X.toΓSpecMapBasicOpen r) :=
  X.presheaf.map (X.toΓSpecMapBasicOpen r).leTop.op


/-- `r` is a unit as a section on the basic open defined by `r`. -/
theorem isUnit_res_toΓSpecMapBasicOpen : IsUnit (X.toToΓSpecMapBasicOpen r r) := by
  convert
    (X.presheaf.map <| (eqToHom <| X.toΓSpecMapBasicOpen_eq r).op).hom.isUnit_map
      (X.toRingedSpace.isUnit_res_basicOpen r)
  -- Porting note: `rw [comp_apply]` to `erw [comp_apply]`
  /-
    case h.e'_3
    X : AlgebraicGeometry.LocallyRingedSpace
    r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    ⊢ Eq ((X.toToΓSpecMapBasicOpen r).hom r) ((X.presheaf.map (CategoryTheory.eqTo …
  -/
  erw [← CommRingCat.comp_apply, ← Functor.map_comp]
  /-
    case h.e'_3
    X : AlgebraicGeometry.LocallyRingedSpace
    r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    ⊢ Eq ((X.toToΓSpecMapBasicOpen r).hom r) ((X.toRingedSpace.presheaf.map (Categ …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- Define the sheaf hom on individual basic opens for the unit. -/
def toΓSpecCApp :
    (structureSheaf <| Γ.obj <| op X).val.obj (op <| basicOpen r) ⟶
      X.presheaf.obj (op <| X.toΓSpecMapBasicOpen r) :=
  -- note: the explicit type annotations were not needed before
  -- https://github.com/leanprover-community/mathlib4/pull/19757
  CommRingCat.ofHom  <|
    IsLocalization.Away.lift
      (R := Γ.obj (op X))
      (S := (structureSheaf ↑(Γ.obj (op X))).val.obj (op (basicOpen r)))
      r
      (isUnit_res_toΓSpecMapBasicOpen _ r)


/-- Characterization of the sheaf hom on basic opens,
    direction ← (next lemma) is used at various places, but → is not used in this file. -/
theorem toΓSpecCApp_iff
    (f :
      (structureSheaf <| Γ.obj <| op X).val.obj (op <| basicOpen r) ⟶
        X.presheaf.obj (op <| X.toΓSpecMapBasicOpen r)) :
    toOpen _ (basicOpen r) ≫ f = X.toToΓSpecMapBasicOpen r ↔ f = X.toΓSpecCApp r := by
  -- Porting Note: Type class problem got stuck in `IsLocalization.Away.AwayMap.lift_comp`
  -- created instance manually. This replaces the `pick_goal` tactics
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    f : Quiver.Hom ((AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry.Loc …
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureShea …
  -/
  have loc_inst := IsLocalization.to_basicOpen (Γ.obj (op X)) r
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    f : Quiver.Hom ((AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry.Loc …
    loc_inst : IsLocalization.Away r ↑((AlgebraicGeometry.Spec.structureSheaf ↑(Al …
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureShea …
  -/
  refine CommRingCat.hom_ext_iff.trans ?_
  rw [← @IsLocalization.Away.lift_comp _ _ _ _ _ _ _ r loc_inst _
      (X.isUnit_res_toΓSpecMapBasicOpen r)]
  --pick_goal 5; exact is_localization.to_basic_open _ r
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    f : Quiver.Hom ((AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry.Loc …
    loc_inst : IsLocalization.Away r ↑((AlgebraicGeometry.Spec.structureSheaf ↑(Al …
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureShea …
  -/
  constructor
    /-
      case mp
      X : AlgebraicGeometry.LocallyRingedSpace
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      f : Quiver.Hom ((AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry.Loc …
      loc_inst : IsLocalization.Away r ↑((AlgebraicGeometry.Spec.structureSheaf ↑(Al …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toO …
    -/
  · intro h
    /-
      case mp
      X : AlgebraicGeometry.LocallyRingedSpace
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      f : Quiver.Hom ((AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry.Loc …
      loc_inst : IsLocalization.Away r ↑((AlgebraicGeometry.Spec.structureSheaf ↑(Al …
      h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.t …
      ⊢ Eq f (X.toΓSpecCApp r)
    -/
    ext : 1
    /-
      case mp.hf
      X : AlgebraicGeometry.LocallyRingedSpace
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      f : Quiver.Hom ((AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry.Loc …
      loc_inst : IsLocalization.Away r ↑((AlgebraicGeometry.Spec.structureSheaf ↑(Al …
      h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.t …
      ⊢ Eq f.hom (X.toΓSpecCApp r).hom
    -/
    exact IsLocalization.ringHom_ext (Submonoid.powers r) h
    /-
      🎉 no goals
    -/
  /-
    case mpr
    X : AlgebraicGeometry.LocallyRingedSpace
    r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    f : Quiver.Hom ((AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry.Loc …
    loc_inst : IsLocalization.Away r ↑((AlgebraicGeometry.Spec.structureSheaf ↑(Al …
    ⊢ Eq f (X.toΓSpecCApp r) → Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGe …
  -/
  apply congr_arg
  /-
    🎉 no goals
  -/


theorem toΓSpecCApp_spec : toOpen _ (basicOpen r) ≫ X.toΓSpecCApp r = X.toToΓSpecMapBasicOpen r :=
  (X.toΓSpecCApp_iff r _).2 rfl


/-- The sheaf hom on all basic opens, commuting with restrictions. -/
@[simps app]
def toΓSpecCBasicOpens :
    (inducedFunctor basicOpen).op ⋙ (structureSheaf (Γ.obj (op X))).1 ⟶
      (inducedFunctor basicOpen).op ⋙ ((TopCat.Sheaf.pushforward _ X.toΓSpecBase).obj X.𝒪).1 where
  app r := X.toΓSpecCApp r.unop
  naturality r s f := by
    /-
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      r s : Opposite (CategoryTheory.InducedCategory (TopologicalSpace.Opens (PrimeS …
      f : Quiver.Hom r s
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.inducedFunctor Prim …
    -/
    apply (StructureSheaf.to_basicOpen_epi (Γ.obj (op X)) r.unop).1
    /-
      case a
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      r s : Opposite (CategoryTheory.InducedCategory (TopologicalSpace.Opens (PrimeS …
      f : Quiver.Hom r s
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toO …
    -/
    simp only [← Category.assoc]
    /-
      case a
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      r s : Opposite (CategoryTheory.InducedCategory (TopologicalSpace.Opens (PrimeS …
      f : Quiver.Hom r s
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [X.toΓSpecCApp_spec r.unop]
    /-
      case a
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      r s : Opposite (CategoryTheory.InducedCategory (TopologicalSpace.Opens (PrimeS …
      f : Quiver.Hom r s
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    convert X.toΓSpecCApp_spec s.unop
    /-
      case h.e'_3.h
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      r s : Opposite (CategoryTheory.InducedCategory (TopologicalSpace.Opens (PrimeS …
      f : Quiver.Hom r s
      e_1✝ : Eq (Quiver.Hom (CommRingCat.of ↑(AlgebraicGeometry.LocallyRingedSpace.Γ …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.toToΓSpecMapBasicOpen (Opposite.un …
    -/
    symm
    /-
      case h.e'_3.h
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      r s : Opposite (CategoryTheory.InducedCategory (TopologicalSpace.Opens (PrimeS …
      f : Quiver.Hom r s
      e_1✝ : Eq (Quiver.Hom (CommRingCat.of ↑(AlgebraicGeometry.LocallyRingedSpace.Γ …
      ⊢ Eq (X.toToΓSpecMapBasicOpen (Opposite.unop s)) (CategoryTheory.CategoryStruc …
    -/
    apply X.presheaf.map_comp
    /-
      🎉 no goals
    -/


/-- The canonical morphism of sheafed spaces from `X` to the spectrum of its global sections. -/
@[simps]
def toΓSpecSheafedSpace : X.toSheafedSpace ⟶ Spec.toSheafedSpace.obj (op (Γ.obj (op X))) where
  base := X.toΓSpecBase
  c :=
    TopCat.Sheaf.restrictHomEquivHom (structureSheaf (Γ.obj (op X))).1 _ isBasis_basic_opens
      X.toΓSpecCBasicOpens


theorem toΓSpecSheafedSpace_app_eq :
    X.toΓSpecSheafedSpace.c.app (op (basicOpen r)) = X.toΓSpecCApp r := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    ⊢ Eq (X.toΓSpecSheafedSpace.c.app { unop := PrimeSpectrum.basicOpen r }) (X.to …
  -/
  apply TopCat.Sheaf.extend_hom_app _ _ _
  /-
    🎉 no goals
  -/

-- Porting note: need a helper lemma `toΓSpecSheafedSpace_app_spec_assoc` to help compile
-- `toStalk_stalkMap_to_Γ_Spec`

@[reassoc] theorem toΓSpecSheafedSpace_app_spec (r : Γ.obj (op X)) :
    toOpen (Γ.obj (op X)) (basicOpen r) ≫ X.toΓSpecSheafedSpace.c.app (op (basicOpen r)) =
      X.toToΓSpecMapBasicOpen r :=
  (X.toΓSpecSheafedSpace_app_eq r).symm ▸ X.toΓSpecCApp_spec r


/-- The map on stalks induced by the unit commutes with maps from `Γ(X)` to
    stalks (in `Spec Γ(X)` and in `X`). -/
theorem toStalk_stalkMap_toΓSpec (x : X) :
    toStalk _ _ ≫ X.toΓSpecSheafedSpace.stalkMap x = X.presheaf.Γgerm x := by
  rw [PresheafedSpace.Hom.stalkMap,
    ← toOpen_germ _ (basicOpen (1 : Γ.obj (op X))) _ (by rw [basicOpen_one]; trivial),
    ← Category.assoc, Category.assoc (toOpen _ _), stalkFunctor_map_germ, ← Category.assoc,
    toΓSpecSheafedSpace_app_spec, Γgerm]
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    x : ↑X.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [← stalkPushforward_germ _ _ X.presheaf ⊤]
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    x : ↑X.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 1
  /-
    case e_a
    X : AlgebraicGeometry.LocallyRingedSpace
    x : ↑X.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.toToΓSpecMapBasicOpen 1) (((TopCat …
  -/
  exact (X.toΓSpecBase _* X.presheaf).germ_res le_top.hom _ _
  /-
    🎉 no goals
  -/


/-- The canonical morphism from `X` to the spectrum of its global sections. -/
@[simps! base]
def toΓSpec : X ⟶ Spec.locallyRingedSpaceObj (Γ.obj (op X)) where
  __ := X.toΓSpecSheafedSpace
  prop := by
    /-
      X : AlgebraicGeometry.LocallyRingedSpace
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      ⊢ ∀ (x : ↑↑X.toPresheafedSpace), IsLocalHom (AlgebraicGeometry.PresheafedSpace …
    -/
    intro x
    /-
      X : AlgebraicGeometry.LocallyRingedSpace
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).hom
    -/
    let p : PrimeSpectrum (Γ.obj (op X)) := X.toΓSpecFun x
    /-
      X : AlgebraicGeometry.LocallyRingedSpace
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).hom
    -/
    constructor
    -- show stalk map is local hom ↓
    /-
      case map_nonunit
      X : AlgebraicGeometry.LocallyRingedSpace
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      ⊢ ∀ (a : ↑((AlgebraicGeometry.Spec.locallyRingedSpaceObj (AlgebraicGeometry.Lo …
    -/
    let S := (structureSheaf _).presheaf.stalk p
    /-
      case map_nonunit
      X : AlgebraicGeometry.LocallyRingedSpace
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      ⊢ ∀ (a : ↑((AlgebraicGeometry.Spec.locallyRingedSpaceObj (AlgebraicGeometry.Lo …
    -/
    rintro (t : S) ht
    /-
      case map_nonunit
      X : AlgebraicGeometry.LocallyRingedSpace
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      ⊢ IsUnit t
    -/
    obtain ⟨⟨r, s⟩, he⟩ := IsLocalization.surj p.asIdeal.primeCompl t
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      he : Eq (HMul.hMul t ((algebraMap ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj …
      ⊢ IsUnit t
    -/
    dsimp at he
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      he : Eq (HMul.hMul t ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) ↑s …
      ⊢ IsUnit t
    -/
    set t' := _
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      he : Eq (HMul.hMul t ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) ↑s …
      t' : ?m.59246 := ?m.59247
      ⊢ IsUnit t
    -/
    change t * t' = _ at he
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      t' : ↑S := (algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) ↑s
      he : Eq (HMul.hMul t t') ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S …
      ⊢ IsUnit t
    -/
    apply isUnit_of_mul_isUnit_left (y := t')
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      t' : ↑S := (algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) ↑s
      he : Eq (HMul.hMul t t') ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S …
      ⊢ IsUnit (HMul.hMul t t')
    -/
    rw [he]
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      t' : ↑S := (algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) ↑s
      he : Eq (HMul.hMul t t') ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S …
      ⊢ IsUnit ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) r)
    -/
    refine IsLocalization.map_units S (⟨r, ?_⟩ : p.asIdeal.primeCompl)
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      t' : ↑S := (algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) ↑s
      he : Eq (HMul.hMul t t') ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S …
      ⊢ Membership.mem p.asIdeal.primeCompl r
    -/
    apply (not_mem_prime_iff_unit_in_stalk _ _ _).mpr
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      t' : ↑S := (algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) ↑s
      he : Eq (HMul.hMul t t') ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S …
      ⊢ IsUnit ((X.presheaf.Γgerm x).hom r)
    -/
    rw [← toStalk_stalkMap_toΓSpec]
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      t' : ↑S := (algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) ↑s
      he : Eq (HMul.hMul t t') ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S …
      ⊢ IsUnit ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureShea …
    -/
    erw [CommRingCat.comp_apply, ← he]
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      t' : ↑S := (algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) ↑s
      he : Eq (HMul.hMul t t') ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S …
      ⊢ IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap X.toΓSpecSheafedSpac …
    -/
    rw [RingHom.map_mul]
    /-
      case map_nonunit.intro.mk
      X : AlgebraicGeometry.LocallyRingedSpace
      r✝ : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      x : ↑↑X.toPresheafedSpace
      p : PrimeSpectrum ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })  …
      S : CommRingCat := (AlgebraicGeometry.Spec.structureSheaf ↑(AlgebraicGeometry. …
      t : ↑S
      ht : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap __spread✝⁻⁰ x).ho …
      r : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
      s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
      t' : ↑S := (algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S) ↑s
      he : Eq (HMul.hMul t t') ((algebraMap ↑(X.presheaf.obj { unop := Top.top }) ↑S …
      ⊢ IsUnit (HMul.hMul ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap X.toΓSpec …
    -/
    exact ht.mul <| (IsLocalization.map_units (R := Γ.obj (op X)) S s).map _
    /-
      🎉 no goals
    -/


/-- On a locally ringed space `X`, the preimage of the zero locus of the prime spectrum
of `Γ(X, ⊤)` under `toΓSpec` agrees with the associated zero locus on `X`. -/
lemma toΓSpec_preimage_zeroLocus_eq {X : LocallyRingedSpace.{u}}
    (s : Set (X.presheaf.obj (op ⊤))) :
    X.toΓSpec.base ⁻¹' PrimeSpectrum.zeroLocus s = X.toRingedSpace.zeroLocus s := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    ⊢ Eq (Set.preimage (⇑X.toΓSpec.base) (PrimeSpectrum.zeroLocus s)) (X.toRingedS …
  -/
  simp only [RingedSpace.zeroLocus]
  have (i : LocallyRingedSpace.Γ.obj (op X)) (_ : i ∈ s) :
      ((X.toRingedSpace.basicOpen i).carrier)ᶜ =
        X.toΓSpec.base ⁻¹' (PrimeSpectrum.basicOpen i).carrierᶜ := by
    symm
    erw [Set.preimage_compl, X.toΓSpec_preimage_basicOpen_eq i]
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    this : ∀ (i : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })), Me …
    ⊢ Eq (Set.preimage (⇑X.toΓSpec.base) (PrimeSpectrum.zeroLocus s)) (Set.iInter  …
  -/
  erw [Set.iInter₂_congr this]
  simp_rw [← Set.preimage_iInter₂, Opens.carrier_eq_coe, PrimeSpectrum.basicOpen_eq_zeroLocus_compl,
    compl_compl]
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    this : ∀ (i : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })), Me …
    ⊢ Eq (Set.preimage (⇑X.toΓSpec.base) (PrimeSpectrum.zeroLocus s)) (Set.preimag …
  -/
  rw [← PrimeSpectrum.zeroLocus_iUnion₂]
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    this : ∀ (i : ↑(AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })), Me …
    ⊢ Eq (Set.preimage (⇑X.toΓSpec.base) (PrimeSpectrum.zeroLocus s)) (Set.preimag …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem comp_ring_hom_ext {X : LocallyRingedSpace.{u}} {R : CommRingCat.{u}} {f : R ⟶ Γ.obj (op X)}
    {β : X ⟶ Spec.locallyRingedSpaceObj R}
    (w : X.toΓSpec.base ≫ (Spec.locallyRingedSpaceMap f).base = β.base)
    (h :
      ∀ r : R,
        f ≫ X.presheaf.map (homOfLE le_top : (Opens.map β.base).obj (basicOpen r) ⟶ _).op =
          toOpen R (basicOpen r) ≫ β.c.app (op (basicOpen r))) :
    X.toΓSpec ≫ Spec.locallyRingedSpaceMap f = β := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    R : CommRingCat
    f : Quiver.Hom R (AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    β : Quiver.Hom X (AlgebraicGeometry.Spec.locallyRingedSpaceObj R)
    w : Eq (CategoryTheory.CategoryStruct.comp X.toΓSpec.base (AlgebraicGeometry.S …
    h : ∀ (r : ↑R), Eq (CategoryTheory.CategoryStruct.comp f (X.presheaf.map (Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp X.toΓSpec (AlgebraicGeometry.Spec.loc …
  -/
  ext1
  -- Porting note: was `apply Spec.basicOpen_hom_ext`
  /-
    case h
    X : AlgebraicGeometry.LocallyRingedSpace
    R : CommRingCat
    f : Quiver.Hom R (AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    β : Quiver.Hom X (AlgebraicGeometry.Spec.locallyRingedSpaceObj R)
    w : Eq (CategoryTheory.CategoryStruct.comp X.toΓSpec.base (AlgebraicGeometry.S …
    h : ∀ (r : ↑R), Eq (CategoryTheory.CategoryStruct.comp f (X.presheaf.map (Cate …
    ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (CategoryTheory.Categor …
  -/
  refine Spec.basicOpen_hom_ext w ?_
  /-
    case h
    X : AlgebraicGeometry.LocallyRingedSpace
    R : CommRingCat
    f : Quiver.Hom R (AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    β : Quiver.Hom X (AlgebraicGeometry.Spec.locallyRingedSpaceObj R)
    w : Eq (CategoryTheory.CategoryStruct.comp X.toΓSpec.base (AlgebraicGeometry.S …
    h : ∀ (r : ↑R), Eq (CategoryTheory.CategoryStruct.comp f (X.presheaf.map (Cate …
    ⊢ ∀ (r : ↑R),
        let U := PrimeSpectrum.basicOpen r;
        Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
  -/
  intro r U
  /-
    case h
    X : AlgebraicGeometry.LocallyRingedSpace
    R : CommRingCat
    f : Quiver.Hom R (AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    β : Quiver.Hom X (AlgebraicGeometry.Spec.locallyRingedSpaceObj R)
    w : Eq (CategoryTheory.CategoryStruct.comp X.toΓSpec.base (AlgebraicGeometry.S …
    h : ∀ (r : ↑R), Eq (CategoryTheory.CategoryStruct.comp f (X.presheaf.map (Cate …
    r : ↑R
    U : TopologicalSpace.Opens (PrimeSpectrum ↑R) := PrimeSpectrum.basicOpen r
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [LocallyRingedSpace.comp_c_app]
  /-
    case h
    X : AlgebraicGeometry.LocallyRingedSpace
    R : CommRingCat
    f : Quiver.Hom R (AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    β : Quiver.Hom X (AlgebraicGeometry.Spec.locallyRingedSpaceObj R)
    w : Eq (CategoryTheory.CategoryStruct.comp X.toΓSpec.base (AlgebraicGeometry.S …
    h : ∀ (r : ↑R), Eq (CategoryTheory.CategoryStruct.comp f (X.presheaf.map (Cate …
    r : ↑R
    U : TopologicalSpace.Opens (PrimeSpectrum ↑R) := PrimeSpectrum.basicOpen r
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [toOpen_comp_comap_assoc]
  /-
    case h
    X : AlgebraicGeometry.LocallyRingedSpace
    R : CommRingCat
    f : Quiver.Hom R (AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    β : Quiver.Hom X (AlgebraicGeometry.Spec.locallyRingedSpaceObj R)
    w : Eq (CategoryTheory.CategoryStruct.comp X.toΓSpec.base (AlgebraicGeometry.S …
    h : ∀ (r : ↑R), Eq (CategoryTheory.CategoryStruct.comp f (X.presheaf.map (Cate …
    r : ↑R
    U : TopologicalSpace.Opens (PrimeSpectrum ↑R) := PrimeSpectrum.basicOpen r
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc]
  /-
    case h
    X : AlgebraicGeometry.LocallyRingedSpace
    R : CommRingCat
    f : Quiver.Hom R (AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    β : Quiver.Hom X (AlgebraicGeometry.Spec.locallyRingedSpaceObj R)
    w : Eq (CategoryTheory.CategoryStruct.comp X.toΓSpec.base (AlgebraicGeometry.S …
    h : ∀ (r : ↑R), Eq (CategoryTheory.CategoryStruct.comp f (X.presheaf.map (Cate …
    r : ↑R
    U : TopologicalSpace.Opens (PrimeSpectrum ↑R) := PrimeSpectrum.basicOpen r
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom f.hom) (CategoryTh …
  -/
  erw [toΓSpecSheafedSpace_app_spec, ← X.presheaf.map_comp]
  /-
    case h
    X : AlgebraicGeometry.LocallyRingedSpace
    R : CommRingCat
    f : Quiver.Hom R (AlgebraicGeometry.LocallyRingedSpace.Γ.obj { unop := X })
    β : Quiver.Hom X (AlgebraicGeometry.Spec.locallyRingedSpaceObj R)
    w : Eq (CategoryTheory.CategoryStruct.comp X.toΓSpec.base (AlgebraicGeometry.S …
    h : ∀ (r : ↑R), Eq (CategoryTheory.CategoryStruct.comp f (X.presheaf.map (Cate …
    r : ↑R
    U : TopologicalSpace.Opens (PrimeSpectrum ↑R) := PrimeSpectrum.basicOpen r
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom f.hom) (X.presheaf …
  -/
  exact h r
  /-
    🎉 no goals
  -/


/-- `toSpecΓ _` is an isomorphism so these are mutually two-sided inverses. -/
theorem Γ_Spec_left_triangle : toSpecΓ (Γ.obj (op X)) ≫ X.toΓSpec.c.app (op ⊤) = 𝟙 _ := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.toSpecΓ (Algebraic …
  -/
  unfold toSpecΓ
  rw [← toOpen_res _ (basicOpen (1 : Γ.obj (op X))) ⊤ (eqToHom basicOpen_one.symm),
    Category.assoc, NatTrans.naturality, ← Category.assoc]
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [X.toΓSpecSheafedSpace_app_spec 1, ← Functor.map_comp]
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    ⊢ Eq (X.presheaf.map (CategoryTheory.CategoryStruct.comp (X.toΓSpecMapBasicOpe …
  -/
  convert eqToHom_map X.presheaf _; rfl
                                    /-
                                      🎉 no goals
                                    -/


/-- The unit as a natural transformation. -/
def identityToΓSpec : 𝟭 LocallyRingedSpace.{u} ⟶ Γ.rightOp ⋙ Spec.toLocallyRingedSpace where
  app := LocallyRingedSpace.toΓSpec
  naturality X Y f := by
    /-
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id Algebraic …
    -/
    symm
    /-
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp X.toΓSpec ((AlgebraicGeometry.Locally …
    -/
    apply LocallyRingedSpace.comp_ring_hom_ext
      /-
        case w
        X Y : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id Algebraic …
      -/
    · ext1 x
      /-
        case w.w
        X Y : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Y
        x : (CategoryTheory.forget TopCat).obj ↑((CategoryTheory.Functor.id AlgebraicG …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id Algebrai …
      -/
      dsimp
      /-
        case w.w
        X Y : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Y
        x : (CategoryTheory.forget TopCat).obj ↑((CategoryTheory.Functor.id AlgebraicG …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp X.toΓSpecBase (AlgebraicGeometry.Spe …
      -/
      show PrimeSpectrum.comap (f.c.app (op ⊤)).hom (X.toΓSpecFun x) = Y.toΓSpecFun (f.base x)
      /-
        case w.w
        X Y : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Y
        x : (CategoryTheory.forget TopCat).obj ↑((CategoryTheory.Functor.id AlgebraicG …
        ⊢ Eq ((PrimeSpectrum.comap (f.c.app { unop := Top.top }).hom) (X.toΓSpecFun x) …
      -/
      dsimp [toΓSpecFun]
      rw [← IsLocalRing.comap_closedPoint (f.stalkMap x).hom, ←
        PrimeSpectrum.comap_comp_apply, ← PrimeSpectrum.comap_comp_apply,
        ← CommRingCat.hom_comp, ← CommRingCat.hom_comp]
      /-
        case w.w
        X Y : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Y
        x : (CategoryTheory.forget TopCat).obj ↑((CategoryTheory.Functor.id AlgebraicG …
        ⊢ Eq ((PrimeSpectrum.comap (CategoryTheory.CategoryStruct.comp (f.c.app { unop …
      -/
      congr 3
      /-
        case w.w.e_a.e_f.e_self
        X Y : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Y
        x : (CategoryTheory.forget TopCat).obj ↑((CategoryTheory.Functor.id AlgebraicG …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.c.app { unop := Top.top }) (X.pres …
      -/
      exact (PresheafedSpace.stalkMap_germ f.1 ⊤ x trivial).symm
      /-
        🎉 no goals
      -/
      /-
        case h
        X Y : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Y
        ⊢ ∀ (r : ↑(Opposite.unop (AlgebraicGeometry.LocallyRingedSpace.Γ.rightOp.obj Y …
      -/
    · intro r
      /-
        case h
        X Y : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Y
        r : ↑(Opposite.unop (AlgebraicGeometry.LocallyRingedSpace.Γ.rightOp.obj Y))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
      -/
      rw [LocallyRingedSpace.comp_c_app, ← Category.assoc]
      /-
        case h
        X Y : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Y
        r : ↑(Opposite.unop (AlgebraicGeometry.LocallyRingedSpace.Γ.rightOp.obj Y))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
      -/
      erw [Y.toΓSpecSheafedSpace_app_spec, f.c.naturality]
      /-
        case h
        X Y : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Y
        r : ↑(Opposite.unop (AlgebraicGeometry.LocallyRingedSpace.Γ.rightOp.obj Y))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem left_triangle (X : LocallyRingedSpace) :
    SpecΓIdentity.inv.app (Γ.obj (op X)) ≫ (identityToΓSpec.app X).c.app (op ⊤) = 𝟙 _ :=
  X.Γ_Spec_left_triangle


/-- `SpecΓIdentity` is iso so these are mutually two-sided inverses. -/
theorem right_triangle (R : CommRingCat) :
    identityToΓSpec.app (Spec.toLocallyRingedSpace.obj <| op R) ≫
        Spec.toLocallyRingedSpace.map (SpecΓIdentity.inv.app R).op =
      𝟙 _ := by
  /-
    R : CommRingCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.identityToΓSpec.ap …
  -/
  apply LocallyRingedSpace.comp_ring_hom_ext
    /-
      case w
      R : CommRingCat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id Algebraic …
    -/
  · ext (p : PrimeSpectrum R)
    /-
      case w.w
      R : CommRingCat
      p : PrimeSpectrum ↑R
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id Algebrai …
    -/
    dsimp
    /-
      case w.w
      R : CommRingCat
      p : PrimeSpectrum ↑R
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.locallyRinge …
    -/
    ext x
    erw [← IsLocalization.AtPrime.to_map_mem_maximal_iff ((structureSheaf R).presheaf.stalk p)
        p.asIdeal x]
    /-
      case w.w.asIdeal.h
      R : CommRingCat
      p : PrimeSpectrum ↑R
      x : ↑R
      ⊢ Iff (Membership.mem ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h
      R : CommRingCat
      ⊢ ∀ (r : ↑(Opposite.unop { unop := (CategoryTheory.Functor.id CommRingCat).obj …
    -/
  · intro r; apply toOpen_res
             /-
               🎉 no goals
             -/


/-- The adjunction `Γ ⊣ Spec` from `CommRingᵒᵖ` to `LocallyRingedSpace`. -/
-- Porting note: `simps` cause a time out, so `Unit` and `counit` will be added manually
def locallyRingedSpaceAdjunction : Γ.rightOp ⊣ Spec.toLocallyRingedSpace.{u} where
  unit := identityToΓSpec
  counit := (NatIso.op SpecΓIdentity).inv
  left_triangle_components X := by
    simp only [Functor.id_obj, Functor.rightOp_obj, Γ_obj, Functor.comp_obj,
      Spec.toLocallyRingedSpace_obj, Spec.locallyRingedSpaceObj_toSheafedSpace,
      Spec.sheafedSpaceObj_carrier, Spec.sheafedSpaceObj_presheaf, Functor.rightOp_map, Γ_map,
      Quiver.Hom.unop_op, NatIso.op_inv, NatTrans.op_app, SpecΓIdentity_inv_app]
    /-
      X : AlgebraicGeometry.LocallyRingedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.identityToΓSpec.a …
    -/
    exact congr_arg Quiver.Hom.op (left_triangle X)
    /-
      🎉 no goals
    -/
  right_triangle_components R := by
    simp only [Spec.toLocallyRingedSpace_obj, Functor.id_obj, Functor.comp_obj, Functor.rightOp_obj,
      Γ_obj, Spec.locallyRingedSpaceObj_toSheafedSpace, Spec.sheafedSpaceObj_carrier,
      Spec.sheafedSpaceObj_presheaf, NatIso.op_inv, NatTrans.op_app, op_unop, SpecΓIdentity_inv_app,
      Spec.toLocallyRingedSpace_map, Quiver.Hom.unop_op]
    /-
      R : Opposite CommRingCat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.identityToΓSpec.ap …
    -/
    exact right_triangle R.unop
    /-
      🎉 no goals
    -/


lemma locallyRingedSpaceAdjunction_unit :
    locallyRingedSpaceAdjunction.unit = identityToΓSpec := rfl


lemma locallyRingedSpaceAdjunction_counit :
    locallyRingedSpaceAdjunction.counit = (NatIso.op SpecΓIdentity.{u}).inv := rfl


@[simp]
lemma locallyRingedSpaceAdjunction_counit_app (R : CommRingCatᵒᵖ) :
    locallyRingedSpaceAdjunction.counit.app R =
      (toOpen R.unop ⊤).op := rfl


@[simp]
lemma locallyRingedSpaceAdjunction_counit_app' (R : Type u) [CommRing R] :
    locallyRingedSpaceAdjunction.counit.app (op <| CommRingCat.of R) =
      (toOpen R ⊤).op := rfl


lemma locallyRingedSpaceAdjunction_homEquiv_apply
    {X : LocallyRingedSpace} {R : CommRingCatᵒᵖ}
    (f : Γ.rightOp.obj X ⟶ R) :
    locallyRingedSpaceAdjunction.homEquiv X R f =
      identityToΓSpec.app X ≫ Spec.locallyRingedSpaceMap f.unop := rfl


lemma locallyRingedSpaceAdjunction_homEquiv_apply'
    {X : LocallyRingedSpace} {R : Type u} [CommRing R]
    (f : CommRingCat.of R ⟶ Γ.obj <| op X) :
    locallyRingedSpaceAdjunction.homEquiv X (op <| CommRingCat.of R) (op f) =
      identityToΓSpec.app X ≫ Spec.locallyRingedSpaceMap f := rfl


lemma toOpen_comp_locallyRingedSpaceAdjunction_homEquiv_app
    {X : LocallyRingedSpace} {R : Type u} [CommRing R]
    (f : Γ.rightOp.obj X ⟶ op (CommRingCat.of R)) (U) :
    StructureSheaf.toOpen R U.unop ≫
      (locallyRingedSpaceAdjunction.homEquiv X (op <| CommRingCat.of R) f).c.app U =
    f.unop ≫ X.presheaf.map (homOfLE le_top).op := by
  rw [← StructureSheaf.toOpen_res _ _ _ (homOfLE le_top), Category.assoc,
    NatTrans.naturality _ (homOfLE (le_top (a := U.unop))).op,
    show (toOpen R ⊤) = (toOpen R ⊤).op.unop from rfl,
    ← locallyRingedSpaceAdjunction_counit_app']
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    R : Type u
    inst✝ : CommRing R
    f : Quiver.Hom (AlgebraicGeometry.LocallyRingedSpace.Γ.rightOp.obj X) { unop : …
    U : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.toLocallyRinged …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.ΓSpec.locallyRinge …
  -/
  simp_rw [← Γ_map_op]
  rw [← Γ.rightOp_map_unop, ← Category.assoc, ← unop_comp, ← Adjunction.homEquiv_counit,
    Equiv.symm_apply_apply]
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    R : Type u
    inst✝ : CommRing R
    f : Quiver.Hom (AlgebraicGeometry.LocallyRingedSpace.Γ.rightOp.obj X) { unop : …
    U : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.toLocallyRinged …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop (((TopCat.Presheaf.pushforward …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The adjunction `Γ ⊣ Spec` from `CommRingᵒᵖ` to `Scheme`. -/
def adjunction : Scheme.Γ.rightOp ⊣ Scheme.Spec.{u} where
  unit :=
  { app := fun X ↦ ⟨locallyRingedSpaceAdjunction.{u}.unit.app X.toLocallyRingedSpace⟩
    naturality := fun _ _ f ↦
      Scheme.Hom.ext' (locallyRingedSpaceAdjunction.{u}.unit.naturality f.toLRSHom) }
  counit := (NatIso.op Scheme.SpecΓIdentity.{u}).inv
  left_triangle_components Y :=
    locallyRingedSpaceAdjunction.left_triangle_components Y.toLocallyRingedSpace
  right_triangle_components R :=
    Scheme.Hom.ext' <| locallyRingedSpaceAdjunction.right_triangle_components R


theorem adjunction_homEquiv_apply {X : Scheme} {R : CommRingCatᵒᵖ}
    (f : (op <| Scheme.Γ.obj <| op X) ⟶ R) :
    ΓSpec.adjunction.homEquiv X R f = ⟨locallyRingedSpaceAdjunction.homEquiv X.1 R f⟩ := rfl


theorem adjunction_homEquiv_symm_apply {X : Scheme} {R : CommRingCatᵒᵖ}
    (f : X ⟶ Scheme.Spec.obj R) :
    (ΓSpec.adjunction.homEquiv X R).symm f =
      (locallyRingedSpaceAdjunction.homEquiv X.1 R).symm f.toLRSHom := rfl


theorem adjunction_counit_app' {R : CommRingCatᵒᵖ} :
    ΓSpec.adjunction.counit.app R = locallyRingedSpaceAdjunction.counit.app R := rfl


@[simp]
theorem adjunction_counit_app {R : CommRingCatᵒᵖ} :
    ΓSpec.adjunction.counit.app R = (Scheme.ΓSpecIso (unop R)).inv.op := rfl


/-- The canonical map `X ⟶ Spec Γ(X, ⊤)`. This is the unit of the `Γ-Spec` adjunction. -/
def _root_.AlgebraicGeometry.Scheme.toSpecΓ (X : Scheme.{u}) : X ⟶ Spec Γ(X, ⊤) :=
  ΓSpec.adjunction.unit.app X


@[simp]
theorem adjunction_unit_app {X : Scheme} :
    ΓSpec.adjunction.unit.app X = X.toSpecΓ := rfl


instance isIso_locallyRingedSpaceAdjunction_counit :
    IsIso.{u + 1, u + 1} locallyRingedSpaceAdjunction.counit :=
  (NatIso.op SpecΓIdentity).isIso_inv


instance isIso_adjunction_counit : IsIso ΓSpec.adjunction.counit := by
  /-
    ⊢ CategoryTheory.IsIso AlgebraicGeometry.ΓSpec.adjunction.counit
  -/
  apply (config := { allowSynthFailures := true }) NatIso.isIso_of_isIso_app
  /-
    case inst
    ⊢ ∀ (X : Opposite CommRingCat), CategoryTheory.IsIso (AlgebraicGeometry.ΓSpec. …
  -/
  intro R
  /-
    case inst
    R : Opposite CommRingCat
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.ΓSpec.adjunction.counit.app R)
  -/
  rw [adjunction_counit_app]
  /-
    case inst
    R : Opposite CommRingCat
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.ΓSpecIso (Opposite.unop R)).i …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem Scheme.toSpecΓ_base (X : Scheme.{u}) (x) :
    (Scheme.toSpecΓ X).base x =
      (Spec.map (X.presheaf.germ ⊤ x trivial)).base (IsLocalRing.closedPoint _) := rfl


@[reassoc (attr := simp)]
theorem Scheme.toSpecΓ_naturality {X Y : Scheme.{u}} (f : X ⟶ Y) :
    f ≫ Y.toSpecΓ = X.toSpecΓ ≫ Spec.map (f.appTop) :=
  ΓSpec.adjunction.unit.naturality f


@[simp]
theorem Scheme.toSpecΓ_appTop (X : Scheme.{u}) :
    X.toSpecΓ.appTop = (Scheme.ΓSpecIso Γ(X, ⊤)).hom := by
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop X.toSpecΓ) (AlgebraicGeometry.Scheme …
  -/
  have := ΓSpec.adjunction.left_triangle_components X
  /-
    X : AlgebraicGeometry.Scheme
    this : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Γ.righ …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop X.toSpecΓ) (AlgebraicGeometry.Scheme …
  -/
  dsimp at this
  /-
    X : AlgebraicGeometry.Scheme
    this : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.ap …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop X.toSpecΓ) (AlgebraicGeometry.Scheme …
  -/
  rw [← IsIso.eq_comp_inv] at this
  simp only [ΓSpec.adjunction_counit_app, Functor.id_obj, Functor.comp_obj, Functor.rightOp_obj,
    Scheme.Γ_obj, Category.id_comp] at this
  /-
    X : AlgebraicGeometry.Scheme
    this : Eq (AlgebraicGeometry.Scheme.Hom.appTop X.toSpecΓ).op (CategoryTheory.i …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop X.toSpecΓ) (AlgebraicGeometry.Scheme …
  -/
  rw [← Quiver.Hom.op_inj.eq_iff, this, ← op_inv, IsIso.Iso.inv_inv]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-23")] alias Scheme.toSpecΓ_app_top := Scheme.toSpecΓ_appTop


@[simp]
theorem SpecMap_ΓSpecIso_hom (R : CommRingCat.{u}) :
    Spec.map ((Scheme.ΓSpecIso R).hom) = (Spec R).toSpecΓ := by
  /-
    R : CommRingCat
    ⊢ Eq (AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.ΓSpecIso R).hom) (A …
  -/
  have := ΓSpec.adjunction.right_triangle_components (op R)
  /-
    R : CommRingCat
    this : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.ΓSpec.adjunct …
    ⊢ Eq (AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.ΓSpecIso R).hom) (A …
  -/
  dsimp at this
  /-
    R : CommRingCat
    this : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec R).toSpe …
    ⊢ Eq (AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.ΓSpecIso R).hom) (A …
  -/
  rwa [← IsIso.eq_comp_inv, Category.id_comp, ← Spec.map_inv, IsIso.Iso.inv_inv, eq_comm] at this
  /-
    🎉 no goals
  -/


lemma Scheme.toSpecΓ_preimage_basicOpen (X : Scheme.{u}) (r : Γ(X, ⊤)) :
    X.toSpecΓ ⁻¹ᵁ (PrimeSpectrum.basicOpen r) = X.basicOpen r := by
  /-
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ Eq ((TopologicalSpace.Opens.map X.toSpecΓ.base).obj (PrimeSpectrum.basicOpen …
  -/
  rw [← basicOpen_eq_of_affine, Scheme.preimage_basicOpen, ← Scheme.Hom.appTop]
  /-
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ Eq (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appTop X.toSpecΓ).hom ((Algeb …
  -/
  congr
  /-
    case e_f
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.appTop X.toSpecΓ).hom ((AlgebraicGeometry. …
  -/
  rw [Scheme.toSpecΓ_appTop]
  /-
    case e_f
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ Eq ((AlgebraicGeometry.Scheme.ΓSpecIso (X.presheaf.obj { unop := Top.top })) …
  -/
  exact Iso.inv_hom_id_apply (C := CommRingCat) _ _
  /-
    🎉 no goals
  -/

-- Warning: this LHS of this lemma breaks the structure-sheaf abstraction.

@[reassoc (attr := simp)]
theorem toOpen_toSpecΓ_app {X : Scheme.{u}} (U) :
    StructureSheaf.toOpen _ _ ≫ X.toSpecΓ.app U =
                                  /-
                                    X : AlgebraicGeometry.Scheme
                                    U : (AlgebraicGeometry.Spec (X.presheaf.obj { unop := Top.top })).Opens
                                    ⊢ LE.le ((TopologicalSpace.Opens.map X.toSpecΓ.base).obj U) Top.top
                                  -/
      X.presheaf.map (homOfLE (by exact le_top)).op := by
                                  /-
                                    🎉 no goals
                                  -/
  rw [← StructureSheaf.toOpen_res _ _ _ (homOfLE le_top), Category.assoc,
    NatTrans.naturality _ (homOfLE (le_top (a := U))).op]
  show (ΓSpec.adjunction.counit.app (Scheme.Γ.rightOp.obj X)).unop ≫
    (Scheme.Γ.rightOp.map (ΓSpec.adjunction.unit.app X)).unop ≫ _ = _
  /-
    X : AlgebraicGeometry.Scheme
    U : (AlgebraicGeometry.Spec (X.presheaf.obj { unop := Top.top })).Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.ΓSpec.adjunction.c …
  -/
  rw [← Category.assoc, ← unop_comp, ΓSpec.adjunction.left_triangle_components]
  /-
    X : AlgebraicGeometry.Scheme
    U : (AlgebraicGeometry.Spec (X.presheaf.obj { unop := Top.top })).Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Al …
  -/
  dsimp
  /-
    X : AlgebraicGeometry.Scheme
    U : (AlgebraicGeometry.Spec (X.presheaf.obj { unop := Top.top })).Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (X. …
  -/
  exact Category.id_comp _
  /-
    🎉 no goals
  -/


lemma ΓSpecIso_inv_ΓSpec_adjunction_homEquiv {X : Scheme.{u}} {B : CommRingCat} (φ : B ⟶ Γ(X, ⊤)) :
    (Scheme.ΓSpecIso B).inv ≫ ((ΓSpec.adjunction.homEquiv X (op B)) φ.op).appTop = φ := by
  /-
    X : AlgebraicGeometry.Scheme
    B : CommRingCat
    φ : Quiver.Hom B (X.presheaf.obj { unop := Top.top })
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.ΓSpecIso B) …
  -/
  simp only [Adjunction.homEquiv_apply, Scheme.Spec_map, Opens.map_top, Scheme.comp_app]
  /-
    X : AlgebraicGeometry.Scheme
    B : CommRingCat
    φ : Quiver.Hom B (X.presheaf.obj { unop := Top.top })
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.ΓSpecIso B) …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma ΓSpec_adjunction_homEquiv_eq {X : Scheme.{u}} {B : CommRingCat} (φ : B ⟶ Γ(X, ⊤)) :
    ((ΓSpec.adjunction.homEquiv X (op B)) φ.op).appTop = (Scheme.ΓSpecIso B).hom ≫ φ := by
  /-
    X : AlgebraicGeometry.Scheme
    B : CommRingCat
    φ : Quiver.Hom B (X.presheaf.obj { unop := Top.top })
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop ((AlgebraicGeometry.ΓSpec.adjunction …
  -/
  rw [← Iso.inv_comp_eq, ΓSpecIso_inv_ΓSpec_adjunction_homEquiv]
  /-
    🎉 no goals
  -/


theorem ΓSpecIso_obj_hom {X : Scheme.{u}} (U : X.Opens) :
    (Scheme.ΓSpecIso Γ(X, U)).hom = (Spec.map U.topIso.inv).appTop ≫
                                                     /-
                                                       X : AlgebraicGeometry.Scheme
                                                       U : X.Opens
                                                       ⊢ Eq (AlgebraicGeometry.Scheme.ΓSpecIso (X.presheaf.obj { unop := U })).hom (C …
                                                     -/
      U.toScheme.toSpecΓ.appTop ≫ U.topIso.hom := by simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[deprecated (since := "2024-07-24")]
alias ΓSpec.adjunction_unit_naturality := Scheme.toSpecΓ_naturality

@[deprecated (since := "2024-07-24")]
alias ΓSpec.adjunction_unit_naturality_assoc := Scheme.toSpecΓ_naturality_assoc

@[deprecated (since := "2024-07-24")]
alias ΓSpec.adjunction_unit_app_app_top := Scheme.toSpecΓ_appTop

@[deprecated (since := "2024-07-24")]
alias ΓSpec.adjunction_unit_map_basicOpen := Scheme.toSpecΓ_preimage_basicOpen


/-- Spec preserves limits. -/
instance : Limits.PreservesLimits Spec.toLocallyRingedSpace :=
  ΓSpec.locallyRingedSpaceAdjunction.rightAdjoint_preservesLimits


instance Spec.preservesLimits : Limits.PreservesLimits Scheme.Spec :=
  ΓSpec.adjunction.rightAdjoint_preservesLimits


/-- The functor `Spec.toLocallyRingedSpace : CommRingCatᵒᵖ ⥤ LocallyRingedSpace`
is fully faithful.-/
def Spec.fullyFaithfulToLocallyRingedSpace : Spec.toLocallyRingedSpace.FullyFaithful :=
  ΓSpec.locallyRingedSpaceAdjunction.fullyFaithfulROfIsIsoCounit


/-- Spec is a full functor. -/
instance : Spec.toLocallyRingedSpace.Full :=
  Spec.fullyFaithfulToLocallyRingedSpace.full


/-- Spec is a faithful functor. -/
instance : Spec.toLocallyRingedSpace.Faithful :=
  Spec.fullyFaithfulToLocallyRingedSpace.faithful


/-- The functor `Spec : CommRingCatᵒᵖ ⥤ Scheme` is fully faithful.-/
def Spec.fullyFaithful : Scheme.Spec.FullyFaithful :=
  ΓSpec.adjunction.fullyFaithfulROfIsIsoCounit


/-- Spec is a full functor. -/
instance Spec.full : Scheme.Spec.Full :=
  Spec.fullyFaithful.full


/-- Spec is a faithful functor. -/
instance Spec.faithful : Scheme.Spec.Faithful :=
  Spec.fullyFaithful.faithful


lemma Spec.map_inj : Spec.map φ = Spec.map ψ ↔ φ = ψ := by
  /-
    R S : CommRingCat
    φ ψ : Quiver.Hom R S
    ⊢ Iff (Eq (AlgebraicGeometry.Spec.map φ) (AlgebraicGeometry.Spec.map ψ)) (Eq φ …
  -/
  rw [iff_comm, ← Quiver.Hom.op_inj.eq_iff, ← Scheme.Spec.map_injective.eq_iff]
  /-
    R S : CommRingCat
    φ ψ : Quiver.Hom R S
    ⊢ Iff (Eq (AlgebraicGeometry.Scheme.Spec.map φ.op) (AlgebraicGeometry.Scheme.S …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma Spec.map_injective {R S : CommRingCat} : Function.Injective (Spec.map : (R ⟶ S) → _) :=
  fun _ _ ↦ Spec.map_inj.mp


/-- The preimage under Spec. -/
def Spec.preimage : R ⟶ S := (Scheme.Spec.preimage f).unop


@[simp] lemma Spec.map_preimage : Spec.map (Spec.preimage f) = f := Scheme.Spec.map_preimage f


variable (φ) in
@[simp] lemma Spec.preimage_map : Spec.preimage (Spec.map φ) = φ :=
  Spec.map_injective (Spec.map_preimage (Spec.map φ))


/-- Spec is fully faithful -/
@[simps]
def Spec.homEquiv {R S : CommRingCat} : (Spec S ⟶ Spec R) ≃ (R ⟶ S) where
  toFun := Spec.preimage
  invFun := Spec.map
  left_inv := Spec.map_preimage
  right_inv := Spec.preimage_map


instance : Spec.toLocallyRingedSpace.IsRightAdjoint :=
  (ΓSpec.locallyRingedSpaceAdjunction).isRightAdjoint


instance : Scheme.Spec.IsRightAdjoint :=
  (ΓSpec.adjunction).isRightAdjoint


instance : Reflective Spec.toLocallyRingedSpace where
  adj := ΓSpec.locallyRingedSpaceAdjunction


instance Spec.reflective : Reflective Scheme.Spec where
  adj := ΓSpec.adjunction


@[deprecated (since := "2024-07-02")]
alias LocallyRingedSpace.toΓSpec_preim_basicOpen_eq :=
  LocallyRingedSpace.toΓSpec_preimage_basicOpen_eq


