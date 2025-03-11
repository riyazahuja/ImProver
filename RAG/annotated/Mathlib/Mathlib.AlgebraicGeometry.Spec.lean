/-- The spectrum of a commutative ring, as a topological space.
-/
def Spec.topObj (R : CommRingCat.{u}) : TopCat :=
  TopCat.of (PrimeSpectrum R)


@[simp] theorem Spec.topObj_forget {R} : (forget TopCat).obj (Spec.topObj R) = PrimeSpectrum R :=
  rfl


/-- The induced map of a ring homomorphism on the ring spectra, as a morphism of topological spaces.
-/
def Spec.topMap {R S : CommRingCat.{u}} (f : R ⟶ S) : Spec.topObj S ⟶ Spec.topObj R :=
  PrimeSpectrum.comap f.hom


@[simp]
theorem Spec.topMap_id (R : CommRingCat.{u}) : Spec.topMap (𝟙 R) = 𝟙 (Spec.topObj R) :=
  rfl


@[simp]
theorem Spec.topMap_comp {R S T : CommRingCat.{u}} (f : R ⟶ S) (g : S ⟶ T) :
    Spec.topMap (f ≫ g) = Spec.topMap g ≫ Spec.topMap f :=
  rfl

-- Porting note: `simps!` generate some garbage lemmas, so choose manually,
-- if more is needed, add them here

/-- The spectrum, as a contravariant functor from commutative rings to topological spaces.
-/
@[simps! obj map]
def Spec.toTop : CommRingCat.{u}ᵒᵖ ⥤ TopCat where
  obj R := Spec.topObj (unop R)
  map {_ _} f := Spec.topMap f.unop


/-- The spectrum of a commutative ring, as a `SheafedSpace`.
-/
@[simps]
def Spec.sheafedSpaceObj (R : CommRingCat.{u}) : SheafedSpace CommRingCat where
  carrier := Spec.topObj R
  presheaf := (structureSheaf R).1
  IsSheaf := (structureSheaf R).2


/-- The induced map of a ring homomorphism on the ring spectra, as a morphism of sheafed spaces.
-/
@[simps base c_app]
def Spec.sheafedSpaceMap {R S : CommRingCat.{u}} (f : R ⟶ S) :
    Spec.sheafedSpaceObj S ⟶ Spec.sheafedSpaceObj R where
  base := Spec.topMap f
  c :=
    { app := fun U => CommRingCat.ofHom <|
        comap f.hom (unop U) ((TopologicalSpace.Opens.map (Spec.topMap f)).obj (unop U)) fun _ => id
                                      /-
                                        R S : CommRingCat
                                        f : Quiver.Hom R S
                                        x✝² x✝¹ : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.sheafedSp …
                                        x✝ : Quiver.Hom x✝² x✝¹
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Spec.sheafedSpace …
                                      -/
      naturality := fun {_ _} _ => by ext; rfl }
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem Spec.sheafedSpaceMap_id {R : CommRingCat.{u}} :
    Spec.sheafedSpaceMap (𝟙 R) = 𝟙 (Spec.sheafedSpaceObj R) :=
  AlgebraicGeometry.PresheafedSpace.Hom.ext _ _ (Spec.topMap_id R) <| by
    /-
      R : CommRingCat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.sheafedSpaceM …
    -/
    ext
    /-
      case w.hf.a
      R : CommRingCat
      U✝ : TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.sheafedSpaceObj R).toPre …
      x✝ : ↑((AlgebraicGeometry.Spec.sheafedSpaceObj R).presheaf.obj { unop := U✝ })
      ⊢ Eq (((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.sheafedSpac …
    -/
    dsimp
    /-
      case w.hf.a
      R : CommRingCat
      U✝ : TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.sheafedSpaceObj R).toPre …
      x✝ : ↑((AlgebraicGeometry.Spec.sheafedSpaceObj R).presheaf.obj { unop := U✝ })
      ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf ↑R).val.map (CategoryTheory.Cate …
    -/
    rw [comap_id (by simp)]
    /-
      case w.hf.a
      R : CommRingCat
      U✝ : TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.sheafedSpaceObj R).toPre …
      x✝ : ↑((AlgebraicGeometry.Spec.sheafedSpaceObj R).presheaf.obj { unop := U✝ })
      ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf ↑R).val.map (CategoryTheory.Cate …
    -/
    simp
    /-
      case w.hf.a
      R : CommRingCat
      U✝ : TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.sheafedSpaceObj R).toPre …
      x✝ : ↑((AlgebraicGeometry.Spec.sheafedSpaceObj R).presheaf.obj { unop := U✝ })
      ⊢ Eq ((RingHom.id ↑((AlgebraicGeometry.Spec.structureSheaf ↑R).val.obj { unop  …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem Spec.sheafedSpaceMap_comp {R S T : CommRingCat.{u}} (f : R ⟶ S) (g : S ⟶ T) :
    Spec.sheafedSpaceMap (f ≫ g) = Spec.sheafedSpaceMap g ≫ Spec.sheafedSpaceMap f :=
  AlgebraicGeometry.PresheafedSpace.Hom.ext _ _ (Spec.topMap_comp f g) <| by
    /-
      R S T : CommRingCat
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.sheafedSpaceM …
    -/
    ext
    -- Porting note: was one liner
    -- `dsimp, rw category_theory.functor.map_id, rw category.comp_id, erw comap_comp f g, refl`
    /-
      case w.hf.a
      R S T : CommRingCat
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      U✝ : TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.sheafedSpaceObj R).toPre …
      x✝ : ↑((AlgebraicGeometry.Spec.sheafedSpaceObj R).presheaf.obj { unop := U✝ })
      ⊢ Eq (((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.sheafedSpac …
    -/
    rw [NatTrans.comp_app, sheafedSpaceMap_c_app, whiskerRight_app, eqToHom_refl]
    /-
      case w.hf.a
      R S T : CommRingCat
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      U✝ : TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.sheafedSpaceObj R).toPre …
      x✝ : ↑((AlgebraicGeometry.Spec.sheafedSpaceObj R).presheaf.obj { unop := U✝ })
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (AlgebraicGeometr …
    -/
    erw [(sheafedSpaceObj T).presheaf.map_id, Category.comp_id, comap_comp]
    /-
      case w.hf.a
      R S T : CommRingCat
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      U✝ : TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.sheafedSpaceObj R).toPre …
      x✝ : ↑((AlgebraicGeometry.Spec.sheafedSpaceObj R).presheaf.obj { unop := U✝ })
      ⊢ Eq ((CommRingCat.ofHom ((AlgebraicGeometry.StructureSheaf.comap g.hom ?w.hf. …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Spec, as a contravariant functor from commutative rings to sheafed spaces.
-/
@[simps]
def Spec.toSheafedSpace : CommRingCat.{u}ᵒᵖ ⥤ SheafedSpace CommRingCat where
  obj R := Spec.sheafedSpaceObj (unop R)
  map f := Spec.sheafedSpaceMap f.unop
                     /-
                       X✝ Y✝ Z✝ : Opposite CommRingCat
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun R => AlgebraicGeometry.Spec.sheafedSpaceObj (Opposite.unop  …
                     -/
  map_comp f g := by simp [Spec.sheafedSpaceMap_comp]
                     /-
                       🎉 no goals
                     -/


/-- Spec, as a contravariant functor from commutative rings to presheafed spaces.
-/
def Spec.toPresheafedSpace : CommRingCat.{u}ᵒᵖ ⥤ PresheafedSpace CommRingCat :=
  Spec.toSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace


@[simp]
theorem Spec.toPresheafedSpace_obj (R : CommRingCat.{u}ᵒᵖ) :
    Spec.toPresheafedSpace.obj R = (Spec.sheafedSpaceObj (unop R)).toPresheafedSpace :=
  rfl


theorem Spec.toPresheafedSpace_obj_op (R : CommRingCat.{u}) :
    Spec.toPresheafedSpace.obj (op R) = (Spec.sheafedSpaceObj R).toPresheafedSpace :=
  rfl


@[simp]
theorem Spec.toPresheafedSpace_map (R S : CommRingCat.{u}ᵒᵖ) (f : R ⟶ S) :
    Spec.toPresheafedSpace.map f = Spec.sheafedSpaceMap f.unop :=
  rfl


theorem Spec.toPresheafedSpace_map_op (R S : CommRingCat.{u}) (f : R ⟶ S) :
    Spec.toPresheafedSpace.map f.op = Spec.sheafedSpaceMap f :=
  rfl


theorem Spec.basicOpen_hom_ext {X : RingedSpace.{u}} {R : CommRingCat.{u}}
    {α β : X ⟶ Spec.sheafedSpaceObj R} (w : α.base = β.base)
    (h : ∀ r : R,
      let U := PrimeSpectrum.basicOpen r
                                                                  /-
                                                                    X : AlgebraicGeometry.RingedSpace
                                                                    R : CommRingCat
                                                                    α β : Quiver.Hom X (AlgebraicGeometry.Spec.sheafedSpaceObj R)
                                                                    w : Eq α.base β.base
                                                                    r : ↑R
                                                                    U : TopologicalSpace.Opens (PrimeSpectrum ↑R) := PrimeSpectrum.basicOpen r
                                                                    ⊢ Eq ((TopologicalSpace.Opens.map α.base).op.obj { unop := U }) ((TopologicalS …
                                                                  -/
      (toOpen R U ≫ α.c.app (op U)) ≫ X.presheaf.map (eqToHom (by rw [w])) =
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
        toOpen R U ≫ β.c.app (op U)) :
    α = β := by
  /-
    X : AlgebraicGeometry.RingedSpace
    R : CommRingCat
    α β : Quiver.Hom X (AlgebraicGeometry.Spec.sheafedSpaceObj R)
    w : Eq α.base β.base
    h :
      ∀ (r : ↑R),
        let U := PrimeSpectrum.basicOpen r;
        Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq α β
  -/
  ext : 1
    /-
      case w
      X : AlgebraicGeometry.RingedSpace
      R : CommRingCat
      α β : Quiver.Hom X (AlgebraicGeometry.Spec.sheafedSpaceObj R)
      w : Eq α.base β.base
      h :
        ∀ (r : ↑R),
          let U := PrimeSpectrum.basicOpen r;
          Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      ⊢ Eq α.base β.base
    -/
  · exact w
    /-
      🎉 no goals
    -/
  · apply
      ((TopCat.Sheaf.pushforward _ β.base).obj X.sheaf).hom_ext _ PrimeSpectrum.isBasis_basic_opens
    /-
      case h
      X : AlgebraicGeometry.RingedSpace
      R : CommRingCat
      α β : Quiver.Hom X (AlgebraicGeometry.Spec.sheafedSpaceObj R)
      w : Eq α.base β.base
      h :
        ∀ (r : ↑R),
          let U := PrimeSpectrum.basicOpen r;
          Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      ⊢ ∀ (i : ↑R), Eq ((CategoryTheory.CategoryStruct.comp α.c (CategoryTheory.whis …
    -/
    intro r
    /-
      case h
      X : AlgebraicGeometry.RingedSpace
      R : CommRingCat
      α β : Quiver.Hom X (AlgebraicGeometry.Spec.sheafedSpaceObj R)
      w : Eq α.base β.base
      h :
        ∀ (r : ↑R),
          let U := PrimeSpectrum.basicOpen r;
          Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      r : ↑R
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp α.c (CategoryTheory.whiskerRight (Ca …
    -/
    apply (StructureSheaf.to_basicOpen_epi R r).1
    /-
      case h.a
      X : AlgebraicGeometry.RingedSpace
      R : CommRingCat
      α β : Quiver.Hom X (AlgebraicGeometry.Spec.sheafedSpaceObj R)
      w : Eq α.base β.base
      h :
        ∀ (r : ↑R),
          let U := PrimeSpectrum.basicOpen r;
          Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      r : ↑R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toO …
    -/
    simpa using h r
    /-
      🎉 no goals
    -/

-- Porting note: `simps!` generate some garbage lemmas, so choose manually,
-- if more is needed, add them here

/-- The spectrum of a commutative ring, as a `LocallyRingedSpace`.
-/
@[simps! toSheafedSpace presheaf]
def Spec.locallyRingedSpaceObj (R : CommRingCat.{u}) : LocallyRingedSpace :=
  { Spec.sheafedSpaceObj R with
    isLocalRing := fun x =>
      RingEquiv.isLocalRing (A := Localization.AtPrime x.asIdeal)
        (Iso.commRingCatIsoToRingEquiv <| stalkIso R x).symm }


lemma Spec.locallyRingedSpaceObj_sheaf (R : CommRingCat.{u}) :
    (Spec.locallyRingedSpaceObj R).sheaf = structureSheaf R := rfl


lemma Spec.locallyRingedSpaceObj_sheaf' (R : Type u) [CommRing R] :
    (Spec.locallyRingedSpaceObj <| CommRingCat.of R).sheaf = structureSheaf R := rfl


lemma Spec.locallyRingedSpaceObj_presheaf_map (R : CommRingCat.{u}) {U V} (i : U ⟶ V) :
    (Spec.locallyRingedSpaceObj R).presheaf.map i =
    (structureSheaf R).1.map i := rfl


lemma Spec.locallyRingedSpaceObj_presheaf' (R : Type u) [CommRing R] :
    (Spec.locallyRingedSpaceObj <| CommRingCat.of R).presheaf = (structureSheaf R).1 := rfl


lemma Spec.locallyRingedSpaceObj_presheaf_map' (R : Type u) [CommRing R] {U V} (i : U ⟶ V) :
    (Spec.locallyRingedSpaceObj <| CommRingCat.of R).presheaf.map i =
    (structureSheaf R).1.map i := rfl


@[elementwise]
theorem stalkMap_toStalk {R S : CommRingCat.{u}} (f : R ⟶ S) (p : PrimeSpectrum S) :
    toStalk R (PrimeSpectrum.comap f.hom p) ≫ (Spec.sheafedSpaceMap f).stalkMap p =
      f ≫ toStalk S p := by
  rw [← toOpen_germ S ⊤ p trivial, ← toOpen_germ R ⊤ (PrimeSpectrum.comap f.hom p) trivial,
    Category.assoc]
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    p : PrimeSpectrum ↑S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toO …
  -/
  erw [PresheafedSpace.stalkMap_germ (Spec.sheafedSpaceMap f) ⊤ p trivial]
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    p : PrimeSpectrum ↑S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toO …
  -/
  rw [Spec.sheafedSpaceMap_c_app]
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    p : PrimeSpectrum ↑S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toO …
  -/
  erw [toOpen_comp_comap_assoc]
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    p : PrimeSpectrum ↑S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom f.hom) (CategoryTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Under the isomorphisms `stalkIso`, the map `stalkMap (Spec.sheafedSpaceMap f) p` corresponds
to the induced local ring homomorphism `Localization.localRingHom`.
-/
@[elementwise]
theorem localRingHom_comp_stalkIso {R S : CommRingCat.{u}} (f : R ⟶ S) (p : PrimeSpectrum S) :
    (stalkIso R (PrimeSpectrum.comap f.hom p)).hom ≫
      (CommRingCat.ofHom (Localization.localRingHom (PrimeSpectrum.comap f.hom p).asIdeal p.asIdeal
          f.hom rfl)) ≫
        (stalkIso S p).inv =
      (Spec.sheafedSpaceMap f).stalkMap p :=
  (stalkIso R (PrimeSpectrum.comap f.hom p)).eq_inv_comp.mp <|
    (stalkIso S p).comp_inv_eq.mpr <| CommRingCat.hom_ext <|
      Localization.localRingHom_unique _ _ _ (PrimeSpectrum.comap_asIdeal _ _) fun x => by
        -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644 and https://github.com/leanprover-community/mathlib4/pull/8386
        rw [stalkIso_hom, stalkIso_inv, CommRingCat.comp_apply, CommRingCat.comp_apply,
            localizationToStalk_of]
        /-
          R S : CommRingCat
          f : Quiver.Hom R S
          p : PrimeSpectrum ↑S
          x : ↑R
          ⊢ Eq ((AlgebraicGeometry.StructureSheaf.stalkToFiberRingHom (↑S) p).hom ((Alge …
        -/
        erw [stalkMap_toStalk_apply f p x, stalkToFiberRingHom_toStalk]
        /-
          R S : CommRingCat
          f : Quiver.Hom R S
          p : PrimeSpectrum ↑S
          x : ↑R
          ⊢ Eq ((algebraMap ↑S ↑(CommRingCat.of (Localization.AtPrime p.asIdeal))) (f x) …
        -/
        rfl
        /-
          🎉 no goals
        -/


/-- Version of `localRingHom_comp_stalkIso_apply` using `CommRingCat.Hom.hom` -/
theorem localRingHom_comp_stalkIso_apply' {R S : CommRingCat.{u}} (f : R ⟶ S) (p : PrimeSpectrum S)
    (x) :
    (stalkIso S p).inv ((Localization.localRingHom (PrimeSpectrum.comap f.hom p).asIdeal p.asIdeal
          f.hom rfl) ((stalkIso R (PrimeSpectrum.comap f.hom p)).hom x)) =
      (Spec.sheafedSpaceMap f).stalkMap p x :=
  localRingHom_comp_stalkIso_apply _ _ _


/--
The induced map of a ring homomorphism on the prime spectra, as a morphism of locally ringed spaces.
-/
@[simps toShHom]
def Spec.locallyRingedSpaceMap {R S : CommRingCat.{u}} (f : R ⟶ S) :
    Spec.locallyRingedSpaceObj S ⟶ Spec.locallyRingedSpaceObj R :=
  LocallyRingedSpace.Hom.mk (Spec.sheafedSpaceMap f) fun p =>
    IsLocalHom.mk fun a ha => by
      -- Here, we are showing that the map on prime spectra induced by `f` is really a morphism of
      -- *locally* ringed spaces, i.e. that the induced map on the stalks is a local ring
      -- homomorphism.

      #adaptation_note /-- nightly-2024-04-01
      It's this `erw` that is blowing up. The implicit arguments differ significantly. -/
      /-
        R S : CommRingCat
        f : Quiver.Hom R S
        p : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj S).toPresheafedSpace
        a : ↑((AlgebraicGeometry.Spec.locallyRingedSpaceObj R).presheaf.stalk ((Algebr …
        ha : IsUnit ((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometr …
        ⊢ IsUnit a
      -/
      erw [← localRingHom_comp_stalkIso_apply' f p a] at ha

      /-
        R S : CommRingCat
        f : Quiver.Hom R S
        p : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj S).toPresheafedSpace
        a : ↑((AlgebraicGeometry.Spec.locallyRingedSpaceObj R).presheaf.stalk ((Algebr …
        ha : IsUnit ((AlgebraicGeometry.StructureSheaf.stalkIso (↑S) p).inv.hom ((Loca …
        ⊢ IsUnit a
      -/
      have : IsLocalHom (stalkIso (↑S) p).inv.hom := isLocalHom_of_isIso _
      /-
        R S : CommRingCat
        f : Quiver.Hom R S
        p : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj S).toPresheafedSpace
        a : ↑((AlgebraicGeometry.Spec.locallyRingedSpaceObj R).presheaf.stalk ((Algebr …
        ha : IsUnit ((AlgebraicGeometry.StructureSheaf.stalkIso (↑S) p).inv.hom ((Loca …
        this : IsLocalHom (AlgebraicGeometry.StructureSheaf.stalkIso (↑S) p).inv.hom
        ⊢ IsUnit a
      -/
      replace ha := (isUnit_map_iff (stalkIso S p).inv.hom _).mp ha
      replace ha := IsLocalHom.map_nonunit
        ((stalkIso R ((PrimeSpectrum.comap f.hom) p)).hom a) ha
      /-
        R S : CommRingCat
        f : Quiver.Hom R S
        p : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj S).toPresheafedSpace
        a : ↑((AlgebraicGeometry.Spec.locallyRingedSpaceObj R).presheaf.stalk ((Algebr …
        this : IsLocalHom (AlgebraicGeometry.StructureSheaf.stalkIso (↑S) p).inv.hom
        ha : IsUnit ((AlgebraicGeometry.StructureSheaf.stalkIso (↑R) ((PrimeSpectrum.c …
        ⊢ IsUnit a
      -/
      convert RingHom.isUnit_map (stalkIso R (PrimeSpectrum.comap f.hom p)).inv.hom ha
      /-
        case h.e'_3.h
        R S : CommRingCat
        f : Quiver.Hom R S
        p : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj S).toPresheafedSpace
        a : ↑((AlgebraicGeometry.Spec.locallyRingedSpaceObj R).presheaf.stalk ((Algebr …
        this : IsLocalHom (AlgebraicGeometry.StructureSheaf.stalkIso (↑S) p).inv.hom
        ha : IsUnit ((AlgebraicGeometry.StructureSheaf.stalkIso (↑R) ((PrimeSpectrum.c …
        e_1✝ : Eq ↑((AlgebraicGeometry.Spec.locallyRingedSpaceObj R).presheaf.stalk (( …
        ⊢ Eq a ((AlgebraicGeometry.StructureSheaf.stalkIso (↑R) ((PrimeSpectrum.comap  …
      -/
      rw [← CommRingCat.comp_apply, Iso.hom_inv_id, CommRingCat.id_apply]
      /-
        🎉 no goals
      -/


@[simp]
theorem Spec.locallyRingedSpaceMap_id (R : CommRingCat.{u}) :
    Spec.locallyRingedSpaceMap (𝟙 R) = 𝟙 (Spec.locallyRingedSpaceObj R) :=
  LocallyRingedSpace.Hom.ext' <| by
    /-
      R : CommRingCat
      ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.Spec …
    -/
    rw [Spec.locallyRingedSpaceMap_toShHom, Spec.sheafedSpaceMap_id]; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem Spec.locallyRingedSpaceMap_comp {R S T : CommRingCat.{u}} (f : R ⟶ S) (g : S ⟶ T) :
    Spec.locallyRingedSpaceMap (f ≫ g) =
      Spec.locallyRingedSpaceMap g ≫ Spec.locallyRingedSpaceMap f :=
  LocallyRingedSpace.Hom.ext' <| by
    /-
      R S T : CommRingCat
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.Spec …
    -/
    rw [Spec.locallyRingedSpaceMap_toShHom, Spec.sheafedSpaceMap_comp]; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- Spec, as a contravariant functor from commutative rings to locally ringed spaces.
-/
@[simps]
def Spec.toLocallyRingedSpace : CommRingCat.{u}ᵒᵖ ⥤ LocallyRingedSpace where
  obj R := Spec.locallyRingedSpaceObj (unop R)
  map f := Spec.locallyRingedSpaceMap f.unop
                 /-
                   R : Opposite CommRingCat
                   ⊢ Eq ({ obj := fun R => AlgebraicGeometry.Spec.locallyRingedSpaceObj (Opposite …
                 -/
  map_id R := by dsimp; rw [Spec.locallyRingedSpaceMap_id]
                        /-
                          🎉 no goals
                        -/
                     /-
                       X✝ Y✝ Z✝ : Opposite CommRingCat
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun R => AlgebraicGeometry.Spec.locallyRingedSpaceObj (Opposite …
                     -/
  map_comp f g := by dsimp; rw [Spec.locallyRingedSpaceMap_comp]
                            /-
                              🎉 no goals
                            -/


/-- The counit morphism `R ⟶ Γ(Spec R)` given by `AlgebraicGeometry.StructureSheaf.toOpen`. -/
@[simps!]
def toSpecΓ (R : CommRingCat.{u}) : R ⟶ Γ.obj (op (Spec.toLocallyRingedSpace.obj (op R))) :=
  StructureSheaf.toOpen R ⊤

-- These lemmas have always been bad (https://github.com/leanprover-community/mathlib4/issues/7657), but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing

instance isIso_toSpecΓ (R : CommRingCat.{u}) : IsIso (toSpecΓ R) := by
  /-
    R : CommRingCat
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.toSpecΓ R)
  -/
  cases R; apply StructureSheaf.isIso_to_global
           /-
             🎉 no goals
           -/


@[reassoc]
theorem Spec_Γ_naturality {R S : CommRingCat.{u}} (f : R ⟶ S) :
    f ≫ toSpecΓ S = toSpecΓ R ≫ Γ.map (Spec.toLocallyRingedSpace.map f.op).op := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` failed to pick up one of the three lemmas
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicGeometry.toSpecΓ S)) (Cat …
  -/
  ext : 2
  /-
    case hf.a
    R S : CommRingCat
    f : Quiver.Hom R S
    x✝ : ↑R
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (AlgebraicGeometry.toSpecΓ S)).hom …
  -/
  refine Subtype.ext <| funext fun x' => ?_; symm
  /-
    case hf.a
    R S : CommRingCat
    f : Quiver.Hom R S
    x✝ : ↑R
    x' : Subtype fun x => Membership.mem (Opposite.unop { unop := Top.top }) x
    ⊢ Eq (↑((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.toSpecΓ R) (Alg …
  -/
  apply Localization.localRingHom_to_map
  /-
    🎉 no goals
  -/


/-- The counit (`SpecΓIdentity.inv.op`) of the adjunction `Γ ⊣ Spec` is an isomorphism. -/
@[simps! hom_app inv_app]
def LocallyRingedSpace.SpecΓIdentity : Spec.toLocallyRingedSpace.rightOp ⋙ Γ ≅ 𝟭 _ :=
  Iso.symm <| NatIso.ofComponents.{u,u,u+1,u+1} (fun R =>
    -- Porting note: In Lean3, this `IsIso` is synthesized automatically
    letI : IsIso (toSpecΓ R) := StructureSheaf.isIso_to_global _
                                         /-
                                           X Y : CommRingCat
                                           f : Quiver.Hom X Y
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id CommRingC …
                                         -/
    asIso (toSpecΓ R)) fun {X Y} f => by convert Spec_Γ_naturality (R := X) (S := Y) f
                                         /-
                                           🎉 no goals
                                         -/


/-- The stalk map of `Spec M⁻¹R ⟶ Spec R` is an iso for each `p : Spec M⁻¹R`. -/
theorem Spec_map_localization_isIso (R : CommRingCat.{u}) (M : Submonoid R)
    (x : PrimeSpectrum (Localization M)) :
    IsIso
      ((Spec.toPresheafedSpace.map
        (CommRingCat.ofHom (algebraMap R (Localization M))).op).stalkMap x) := by
  /-
    R : CommRingCat
    M : Submonoid ↑R
    x : PrimeSpectrum (Localization M)
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (Algebr …
  -/
  erw [← localRingHom_comp_stalkIso]
  -- Porting note: replaced `apply (config := { instances := false })`.
  -- See https://github.com/leanprover/lean4/issues/2273
  /-
    R : CommRingCat
    M : Submonoid ↑R
    x : PrimeSpectrum (Localization M)
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
  -/
  refine IsIso.comp_isIso' inferInstance (IsIso.comp_isIso' ?_ inferInstance)
  /- I do not know why this is defeq to the goal, but I'm happy to accept that it is. -/
  show
    IsIso (IsLocalization.localizationLocalizationAtPrimeIsoLocalization M
      x.asIdeal).toRingEquiv.toCommRingCatIso.hom
  /-
    R : CommRingCat
    M : Submonoid ↑R
    x : PrimeSpectrum (Localization M)
    ⊢ CategoryTheory.IsIso (IsLocalization.localizationLocalizationAtPrimeIsoLocal …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- For an algebra `f : R →+* S`, this is the ring homomorphism `S →+* (f∗ 𝒪ₛ)ₚ` for a `p : Spec R`.
This is shown to be the localization at `p` in `isLocalizedModule_toPushforwardStalkAlgHom`.
-/
def toPushforwardStalk : S ⟶ (Spec.topMap f _* (structureSheaf S).1).stalk p :=
  StructureSheaf.toOpen S ⊤ ≫
    @TopCat.Presheaf.germ _ _ _ _ (Spec.topMap f _* (structureSheaf S).1) ⊤ p trivial


@[reassoc]
theorem toPushforwardStalk_comp :
    f ≫ StructureSheaf.toPushforwardStalk f p =
      StructureSheaf.toStalk R p ≫
        (TopCat.Presheaf.stalkFunctor _ _).map (Spec.sheafedSpaceMap f).c := by
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    p : PrimeSpectrum ↑R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicGeometry.StructureSheaf.t …
  -/
  rw [StructureSheaf.toStalk, Category.assoc, TopCat.Presheaf.stalkFunctor_map_germ]
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    p : PrimeSpectrum ↑R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicGeometry.StructureSheaf.t …
  -/
  exact Spec_Γ_naturality_assoc f _
  /-
    🎉 no goals
  -/


instance : Algebra R ((Spec.topMap f _* (structureSheaf S).1).stalk p) :=
  (f ≫ StructureSheaf.toPushforwardStalk f p).hom.toAlgebra


theorem algebraMap_pushforward_stalk :
    algebraMap R ((Spec.topMap f _* (structureSheaf S).1).stalk p) =
      (f ≫ StructureSheaf.toPushforwardStalk f p).hom :=
  rfl


/--
This is the `AlgHom` version of `toPushforwardStalk`, which is the map `S ⟶ (f∗ 𝒪ₛ)ₚ` for some
algebra `R ⟶ S` and some `p : Spec R`.
-/
@[simps!]
def toPushforwardStalkAlgHom :
    S →ₐ[R] (Spec.topMap (CommRingCat.ofHom (algebraMap R S)) _* (structureSheaf S).1).stalk p :=
  { (StructureSheaf.toPushforwardStalk (CommRingCat.ofHom (algebraMap R S)) p).hom with
    commutes' := fun _ => rfl }


theorem isLocalizedModule_toPushforwardStalkAlgHom_aux (y) :
    ∃ x : S × p.asIdeal.primeCompl, x.2 • y = toPushforwardStalkAlgHom R S p x.1 := by
  obtain ⟨U, hp, s, e⟩ := TopCat.Presheaf.germ_exist
    -- Porting note: originally the first variable does not need to be explicit
    (Spec.topMap (CommRingCat.ofHom (algebraMap ↑R ↑S)) _* (structureSheaf S).val) _ y
  obtain ⟨_, ⟨r, rfl⟩, hpr : p ∈ PrimeSpectrum.basicOpen r, hrU : PrimeSpectrum.basicOpen r ≤ U⟩ :=
    PrimeSpectrum.isTopologicalBasis_basic_opens.exists_subset_of_mem_open (show p ∈ U from hp) U.2
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R S : CommRingCat
    p : PrimeSpectrum ↑R
    inst✝ : Algebra ↑R ↑S
    y : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMap …
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.Spec.topObj (CommRingCat.of ↑R))
    hp : Membership.mem U p
    s : (CategoryTheory.forget CommRingCat).obj (((TopCat.Presheaf.pushforward Com …
    e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
    r : ↑(CommRingCat.of ↑R)
    hpr : Membership.mem (PrimeSpectrum.basicOpen r) p
    hrU : LE.le (PrimeSpectrum.basicOpen r) U
    ⊢ Exists fun x => Eq (HSMul.hSMul x.2 y) ((AlgebraicGeometry.StructureSheaf.to …
  -/
  change PrimeSpectrum.basicOpen r ≤ U at hrU
  replace e :=
    ((Spec.topMap (CommRingCat.ofHom (algebraMap R S)) _* (structureSheaf S).1).germ_res_apply
      (homOfLE hrU) p hpr _).trans e
  set s' := (Spec.topMap (CommRingCat.ofHom (algebraMap R S)) _* (structureSheaf S).1).map
      (homOfLE hrU).op s with h
  replace e : ((Spec.topMap (CommRingCat.ofHom (algebraMap R S)) _* (structureSheaf S).val).germ _
      p hpr) s' = y := by
    rw [h]; exact e
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R S : CommRingCat
    p : PrimeSpectrum ↑R
    inst✝ : Algebra ↑R ↑S
    y : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMap …
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.Spec.topObj (CommRingCat.of ↑R))
    hp : Membership.mem U p
    s : (CategoryTheory.forget CommRingCat).obj (((TopCat.Presheaf.pushforward Com …
    r : ↑(CommRingCat.of ↑R)
    hpr : Membership.mem (PrimeSpectrum.basicOpen r) p
    hrU : LE.le (PrimeSpectrum.basicOpen r) U
    s' : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMa …
    h : Eq s' ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec. …
    e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
    ⊢ Exists fun x => Eq (HSMul.hSMul x.2 y) ((AlgebraicGeometry.StructureSheaf.to …
  -/
  clear_value s'; clear! U
  obtain ⟨⟨s, ⟨_, n, rfl⟩⟩, hsn⟩ :=
    @IsLocalization.surj _ _ _ _ _ _
      (StructureSheaf.IsLocalization.to_basicOpen S <| algebraMap R S r) s'
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.mk.mk.intro
    R S : CommRingCat
    p : PrimeSpectrum ↑R
    inst✝ : Algebra ↑R ↑S
    y : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMap …
    r : ↑(CommRingCat.of ↑R)
    hpr : Membership.mem (PrimeSpectrum.basicOpen r) p
    s' : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMa …
    e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
    s : ↑S
    n : Nat
    hsn : Eq (HMul.hMul s' ((algebraMap ↑S ↑((AlgebraicGeometry.Spec.structureShea …
    ⊢ Exists fun x => Eq (HSMul.hSMul x.2 y) ((AlgebraicGeometry.StructureSheaf.to …
  -/
  refine ⟨⟨s, ⟨r, hpr⟩ ^ n⟩, ?_⟩
  rw [Submonoid.smul_def, Algebra.smul_def, algebraMap_pushforward_stalk, toPushforwardStalk,
    CommRingCat.comp_apply, CommRingCat.comp_apply]
  iterate 2
    erw [← (Spec.topMap (CommRingCat.ofHom (algebraMap R S)) _* (structureSheaf S).1).germ_res_apply
      (homOfLE le_top) p hpr]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.mk.mk.intro
    R S : CommRingCat
    p : PrimeSpectrum ↑R
    inst✝ : Algebra ↑R ↑S
    y : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMap …
    r : ↑(CommRingCat.of ↑R)
    hpr : Membership.mem (PrimeSpectrum.basicOpen r) p
    s' : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMa …
    e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
    s : ↑S
    n : Nat
    hsn : Eq (HMul.hMul s' ((algebraMap ↑S ↑((AlgebraicGeometry.Spec.structureShea …
    ⊢ Eq (HMul.hMul ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry …
  -/
  rw [← e]
  -- Porting note: without this `change`, Lean doesn't know how to rewrite `map_mul`
  let f := TopCat.Presheaf.germ (Spec.topMap (CommRingCat.ofHom (algebraMap R S)) _*
      (structureSheaf S).val) _ p hpr
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.mk.mk.intro
    R S : CommRingCat
    p : PrimeSpectrum ↑R
    inst✝ : Algebra ↑R ↑S
    y : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMap …
    r : ↑(CommRingCat.of ↑R)
    hpr : Membership.mem (PrimeSpectrum.basicOpen r) p
    s' : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMa …
    e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
    s : ↑S
    n : Nat
    hsn : Eq (HMul.hMul s' ((algebraMap ↑S ↑((AlgebraicGeometry.Spec.structureShea …
    f : Quiver.Hom (((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.S …
    ⊢ Eq (HMul.hMul ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry …
  -/
  change f _ * f _ = f _
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.mk.mk.intro
    R S : CommRingCat
    p : PrimeSpectrum ↑R
    inst✝ : Algebra ↑R ↑S
    y : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMap …
    r : ↑(CommRingCat.of ↑R)
    hpr : Membership.mem (PrimeSpectrum.basicOpen r) p
    s' : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMa …
    e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
    s : ↑S
    n : Nat
    hsn : Eq (HMul.hMul s' ((algebraMap ↑S ↑((AlgebraicGeometry.Spec.structureShea …
    f : Quiver.Hom (((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.S …
    ⊢ Eq (HMul.hMul (f.hom ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicG …
  -/
  rw [← map_mul, mul_comm]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.mk.mk.intro
    R S : CommRingCat
    p : PrimeSpectrum ↑R
    inst✝ : Algebra ↑R ↑S
    y : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMap …
    r : ↑(CommRingCat.of ↑R)
    hpr : Membership.mem (PrimeSpectrum.basicOpen r) p
    s' : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMa …
    e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
    s : ↑S
    n : Nat
    hsn : Eq (HMul.hMul s' ((algebraMap ↑S ↑((AlgebraicGeometry.Spec.structureShea …
    f : Quiver.Hom (((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.S …
    ⊢ Eq (f.hom (HMul.hMul s' ((((TopCat.Presheaf.pushforward CommRingCat (Algebra …
  -/
  dsimp only [Subtype.coe_mk] at hsn
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.mk.mk.intro
    R S : CommRingCat
    p : PrimeSpectrum ↑R
    inst✝ : Algebra ↑R ↑S
    y : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMap …
    r : ↑(CommRingCat.of ↑R)
    hpr : Membership.mem (PrimeSpectrum.basicOpen r) p
    s' : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMa …
    e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
    s : ↑S
    n : Nat
    hsn : Eq (HMul.hMul s' ((algebraMap ↑S ↑((AlgebraicGeometry.Spec.structureShea …
    f : Quiver.Hom (((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.S …
    ⊢ Eq (f.hom (HMul.hMul s' ((((TopCat.Presheaf.pushforward CommRingCat (Algebra …
  -/
  rw [← map_pow (algebraMap R S)] at hsn
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.mk.mk.intro
    R S : CommRingCat
    p : PrimeSpectrum ↑R
    inst✝ : Algebra ↑R ↑S
    y : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMap …
    r : ↑(CommRingCat.of ↑R)
    hpr : Membership.mem (PrimeSpectrum.basicOpen r) p
    s' : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.topMa …
    e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
    s : ↑S
    n : Nat
    hsn : Eq (HMul.hMul s' ((algebraMap ↑S ↑((AlgebraicGeometry.Spec.structureShea …
    f : Quiver.Hom (((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.S …
    ⊢ Eq (f.hom (HMul.hMul s' ((((TopCat.Presheaf.pushforward CommRingCat (Algebra …
  -/
  congr 1
  /-
    🎉 no goals
  -/


instance isLocalizedModule_toPushforwardStalkAlgHom :
    IsLocalizedModule p.asIdeal.primeCompl (toPushforwardStalkAlgHom R S p).toLinearMap := by
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    p : PrimeSpectrum ↑R
    inst✝ : Algebra ↑R ↑S
    ⊢ IsLocalizedModule p.asIdeal.primeCompl (AlgebraicGeometry.StructureSheaf.toP …
  -/
  apply IsLocalizedModule.mkOfAlgebra
    /-
      case h₁
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      ⊢ ∀ (x : ↑R), Membership.mem p.asIdeal.primeCompl x → IsUnit ((algebraMap ↑R ↑ …
    -/
  · intro x hx; rw [algebraMap_pushforward_stalk, toPushforwardStalk_comp]
    change IsUnit ((TopCat.Presheaf.stalkFunctor CommRingCat p).map
      (Spec.sheafedSpaceMap (CommRingCat.ofHom (algebraMap ↑R ↑S))).c _)
    /-
      case h₁
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑R
      hx : Membership.mem p.asIdeal.primeCompl x
      ⊢ IsUnit (((TopCat.Presheaf.stalkFunctor CommRingCat p).map (AlgebraicGeometry …
    -/
    exact (IsLocalization.map_units ((structureSheaf R).presheaf.stalk p) ⟨x, hx⟩).map _
    /-
      🎉 no goals
    -/
    /-
      case h₂
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      ⊢ ∀ (y : ↑(((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.t …
    -/
  · apply isLocalizedModule_toPushforwardStalkAlgHom_aux
    /-
      🎉 no goals
    -/
    /-
      case h₃
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      ⊢ ∀ (x : ↑S), Eq ((AlgebraicGeometry.StructureSheaf.toPushforwardStalkAlgHom R …
    -/
  · intro x hx
    rw [toPushforwardStalkAlgHom_apply,
      ← (toPushforwardStalk (CommRingCat.ofHom (algebraMap ↑R ↑S)) p).hom.map_zero,
      toPushforwardStalk] at hx
    -- Porting note: this `change` is manually rewriting `comp_apply`
    change _ = (TopCat.Presheaf.germ (Spec.topMap (CommRingCat.ofHom (algebraMap ↑R ↑S)) _*
      (structureSheaf ↑S).val) ⊤ p trivial (toOpen S ⊤ 0)) at hx
    /-
      case h₃
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.to …
      ⊢ Exists fun m => Eq (HSMul.hSMul m x) 0
    -/
    rw [map_zero] at hx
    /-
      case h₃
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.to …
      ⊢ Exists fun m => Eq (HSMul.hSMul m x) 0
    -/
    change (forget CommRingCat).map _ _ = (forget _).map _ _ at hx
    /-
      case h₃
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((CategoryTheory.forget CommRingCat).map (((TopCat.Presheaf.pushforwar …
      ⊢ Exists fun m => Eq (HSMul.hSMul m x) 0
    -/
    obtain ⟨U, hpU, i₁, i₂, e⟩ := TopCat.Presheaf.germ_eq _ _ _ _ _ _ hx
    obtain ⟨_, ⟨r, rfl⟩, hpr, hrU⟩ :=
      PrimeSpectrum.isTopologicalBasis_basic_opens.exists_subset_of_mem_open (show p ∈ U.1 from hpU)
        U.2
    /-
      case h₃.intro.intro.intro.intro.intro.intro.intro.intro
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((CategoryTheory.forget CommRingCat).map (((TopCat.Presheaf.pushforwar …
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.Spec.topObj (CommRingCat.of ↑R))
      hpU : Membership.mem U p
      i₁ i₂ : Quiver.Hom U Top.top
      e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
      r : ↑(CommRingCat.of ↑R)
      hpr : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) r) p
      hrU : HasSubset.Subset ((fun r => ↑(PrimeSpectrum.basicOpen r)) r) U.carrier
      ⊢ Exists fun m => Eq (HSMul.hSMul m x) 0
    -/
    change PrimeSpectrum.basicOpen r ≤ U at hrU
    apply_fun (Spec.topMap (CommRingCat.ofHom (algebraMap R S)) _* (structureSheaf S).1).map
        (homOfLE hrU).op at e
    /-
      case h₃.intro.intro.intro.intro.intro.intro.intro.intro
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((CategoryTheory.forget CommRingCat).map (((TopCat.Presheaf.pushforwar …
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.Spec.topObj (CommRingCat.of ↑R))
      hpU : Membership.mem U p
      i₁ i₂ : Quiver.Hom U Top.top
      r : ↑(CommRingCat.of ↑R)
      hpr : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) r) p
      hrU : LE.le (PrimeSpectrum.basicOpen r) U
      e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
      ⊢ Exists fun m => Eq (HSMul.hSMul m x) 0
    -/
    simp only [Functor.op_map, map_zero, ← comp_apply, toOpen_res] at e
    have : toOpen S (PrimeSpectrum.basicOpen <| algebraMap R S r) x = 0 := by
      refine Eq.trans ?_ e; rfl
    have :=
      (@IsLocalization.mk'_one _ _ _ _ _ _
            (StructureSheaf.IsLocalization.to_basicOpen S <| algebraMap R S r) x).trans
        this
    /-
      case h₃.intro.intro.intro.intro.intro.intro.intro.intro
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((CategoryTheory.forget CommRingCat).map (((TopCat.Presheaf.pushforwar …
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.Spec.topObj (CommRingCat.of ↑R))
      hpU : Membership.mem U p
      i₁ i₂ : Quiver.Hom U Top.top
      r : ↑(CommRingCat.of ↑R)
      hpr : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) r) p
      hrU : LE.le (PrimeSpectrum.basicOpen r) U
      e : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.top …
      this✝ : Eq ((AlgebraicGeometry.StructureSheaf.toOpen (↑S) (PrimeSpectrum.basic …
      this : Eq (IsLocalization.mk' (↑((AlgebraicGeometry.Spec.structureSheaf ↑S).va …
      ⊢ Exists fun m => Eq (HSMul.hSMul m x) 0
    -/
    obtain ⟨⟨_, n, rfl⟩, e⟩ := (IsLocalization.mk'_eq_zero_iff _ _).mp this
    /-
      case h₃.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intro
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((CategoryTheory.forget CommRingCat).map (((TopCat.Presheaf.pushforwar …
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.Spec.topObj (CommRingCat.of ↑R))
      hpU : Membership.mem U p
      i₁ i₂ : Quiver.Hom U Top.top
      r : ↑(CommRingCat.of ↑R)
      hpr : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) r) p
      hrU : LE.le (PrimeSpectrum.basicOpen r) U
      e✝ : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.to …
      this✝ : Eq ((AlgebraicGeometry.StructureSheaf.toOpen (↑S) (PrimeSpectrum.basic …
      this : Eq (IsLocalization.mk' (↑((AlgebraicGeometry.Spec.structureSheaf ↑S).va …
      n : Nat
      e : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow ((algebraMap ↑R ↑S) r) x) n, ⋯⟩) x) 0
      ⊢ Exists fun m => Eq (HSMul.hSMul m x) 0
    -/
    refine ⟨⟨r, hpr⟩ ^ n, ?_⟩
    /-
      case h₃.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intro
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((CategoryTheory.forget CommRingCat).map (((TopCat.Presheaf.pushforwar …
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.Spec.topObj (CommRingCat.of ↑R))
      hpU : Membership.mem U p
      i₁ i₂ : Quiver.Hom U Top.top
      r : ↑(CommRingCat.of ↑R)
      hpr : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) r) p
      hrU : LE.le (PrimeSpectrum.basicOpen r) U
      e✝ : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.to …
      this✝ : Eq ((AlgebraicGeometry.StructureSheaf.toOpen (↑S) (PrimeSpectrum.basic …
      this : Eq (IsLocalization.mk' (↑((AlgebraicGeometry.Spec.structureSheaf ↑S).va …
      n : Nat
      e : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow ((algebraMap ↑R ↑S) r) x) n, ⋯⟩) x) 0
      ⊢ Eq (HSMul.hSMul (HPow.hPow ⟨r, hpr⟩ n) x) 0
    -/
    rw [Submonoid.smul_def, Algebra.smul_def]
    -- Porting note: manually rewrite `Submonoid.coe_pow`
    /-
      case h₃.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intro
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((CategoryTheory.forget CommRingCat).map (((TopCat.Presheaf.pushforwar …
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.Spec.topObj (CommRingCat.of ↑R))
      hpU : Membership.mem U p
      i₁ i₂ : Quiver.Hom U Top.top
      r : ↑(CommRingCat.of ↑R)
      hpr : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) r) p
      hrU : LE.le (PrimeSpectrum.basicOpen r) U
      e✝ : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.to …
      this✝ : Eq ((AlgebraicGeometry.StructureSheaf.toOpen (↑S) (PrimeSpectrum.basic …
      this : Eq (IsLocalization.mk' (↑((AlgebraicGeometry.Spec.structureSheaf ↑S).va …
      n : Nat
      e : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow ((algebraMap ↑R ↑S) r) x) n, ⋯⟩) x) 0
      ⊢ Eq (HMul.hMul ((algebraMap ↑R ↑S) ↑(HPow.hPow ⟨r, hpr⟩ n)) x) 0
    -/
    change (algebraMap R S) (r ^ n) * x = 0
    /-
      case h₃.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intro
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((CategoryTheory.forget CommRingCat).map (((TopCat.Presheaf.pushforwar …
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.Spec.topObj (CommRingCat.of ↑R))
      hpU : Membership.mem U p
      i₁ i₂ : Quiver.Hom U Top.top
      r : ↑(CommRingCat.of ↑R)
      hpr : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) r) p
      hrU : LE.le (PrimeSpectrum.basicOpen r) U
      e✝ : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.to …
      this✝ : Eq ((AlgebraicGeometry.StructureSheaf.toOpen (↑S) (PrimeSpectrum.basic …
      this : Eq (IsLocalization.mk' (↑((AlgebraicGeometry.Spec.structureSheaf ↑S).va …
      n : Nat
      e : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow ((algebraMap ↑R ↑S) r) x) n, ⋯⟩) x) 0
      ⊢ Eq (HMul.hMul ((algebraMap ↑R ↑S) (HPow.hPow r n)) x) 0
    -/
    rw [map_pow]
    /-
      case h₃.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk.intro
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑R
      inst✝ : Algebra ↑R ↑S
      x : ↑S
      hx : Eq ((CategoryTheory.forget CommRingCat).map (((TopCat.Presheaf.pushforwar …
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.Spec.topObj (CommRingCat.of ↑R))
      hpU : Membership.mem U p
      i₁ i₂ : Quiver.Hom U Top.top
      r : ↑(CommRingCat.of ↑R)
      hpr : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) r) p
      hrU : LE.le (PrimeSpectrum.basicOpen r) U
      e✝ : Eq ((((TopCat.Presheaf.pushforward CommRingCat (AlgebraicGeometry.Spec.to …
      this✝ : Eq ((AlgebraicGeometry.StructureSheaf.toOpen (↑S) (PrimeSpectrum.basic …
      this : Eq (IsLocalization.mk' (↑((AlgebraicGeometry.Spec.structureSheaf ↑S).va …
      n : Nat
      e : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow ((algebraMap ↑R ↑S) r) x) n, ⋯⟩) x) 0
      ⊢ Eq (HMul.hMul (HPow.hPow ((algebraMap ↑R ↑S) r) n) x) 0
    -/
    exact e
    /-
      🎉 no goals
    -/


