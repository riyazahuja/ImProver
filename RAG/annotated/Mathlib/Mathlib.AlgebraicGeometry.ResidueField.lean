/-- The residue field of `X` at a point `x` is the residue field of the stalk of `X`
at `x`. -/
def residueField (x : X) : CommRingCat :=
  CommRingCat.of <| IsLocalRing.ResidueField (X.presheaf.stalk x)


instance (x : X) : Field (X.residueField x) :=
  inferInstanceAs <| Field (IsLocalRing.ResidueField (X.presheaf.stalk x))


/-- The residue map from the stalk to the residue field. -/
def residue (X : Scheme.{u}) (x) : X.presheaf.stalk x ⟶ X.residueField x :=
  CommRingCat.ofHom (IsLocalRing.residue (X.presheaf.stalk x))


/-- See `AlgebraicGeometry.IsClosedImmersion.Spec_map_residue` for the stronger result that
`Spec.map (X.residue x)` is a closed immersion. -/
instance {X : Scheme.{u}} (x) : IsPreimmersion (Spec.map (X.residue x)) :=
  IsPreimmersion.mk_Spec_map
    (PrimeSpectrum.isClosedEmbedding_comap_of_surjective _ _
      Ideal.Quotient.mk_surjective).isEmbedding
    (RingHom.surjectiveOnStalks_of_surjective (Ideal.Quotient.mk_surjective))


@[simp]
lemma Spec_map_residue_apply {X : Scheme.{u}} (x : X) (s : Spec (X.residueField x)) :
    (Spec.map (X.residue x)).base s = closedPoint (X.presheaf.stalk x) :=
  IsLocalRing.PrimeSpectrum.comap_residue _ s


lemma residue_surjective (X : Scheme.{u}) (x) : Function.Surjective (X.residue x) :=
  Ideal.Quotient.mk_surjective


instance (X : Scheme.{u}) (x) : Epi (X.residue x) :=
  ConcreteCategory.epi_of_surjective _ (X.residue_surjective x)


/-- If `K` is a field and `f : 𝒪_{X, x} ⟶ K` is a ring map, then this is the induced
map `κ(x) ⟶ K`. -/
def descResidueField {K : Type u} [Field K] {X : Scheme.{u}} {x : X}
    (f : X.presheaf.stalk x ⟶ .of K) [IsLocalHom f.hom] :
    X.residueField x ⟶ .of K :=
  CommRingCat.ofHom (IsLocalRing.ResidueField.lift (S := K) f.hom)


@[reassoc (attr := simp)]
lemma residue_descResidueField {K : Type u} [Field K] {X : Scheme.{u}} {x}
    (f : X.presheaf.stalk x ⟶ .of K) [IsLocalHom f.hom] :
    X.residue x ≫ X.descResidueField f = f :=
  CommRingCat.hom_ext <| RingHom.ext fun _ ↦ rfl


/--
If `U` is an open of `X` containing `x`, we have a canonical ring map from the sections
over `U` to the residue field of `x`.

If we interpret sections over `U` as functions of `X` defined on `U`, then this ring map
corresponds to evaluation at `x`.
-/
def evaluation (U : X.Opens) (x : X) (hx : x ∈ U) : Γ(X, U) ⟶ X.residueField x :=
  X.presheaf.germ U x hx ≫ X.residue _


@[reassoc]
lemma germ_residue (x hx) : X.presheaf.germ U x hx ≫ X.residue x = X.evaluation U x hx := rfl


/-- The global evaluation map from `Γ(X, ⊤)` to the residue field at `x`. -/
abbrev Γevaluation (x : X) : Γ(X, ⊤) ⟶ X.residueField x :=
  X.evaluation ⊤ x trivial


@[simp]
lemma evaluation_eq_zero_iff_not_mem_basicOpen (x : X) (hx : x ∈ U) (f : Γ(X, U)) :
    X.evaluation U x hx f = 0 ↔ x ∉ X.basicOpen f :=
  X.toLocallyRingedSpace.evaluation_eq_zero_iff_not_mem_basicOpen ⟨x, hx⟩ f


lemma evaluation_ne_zero_iff_mem_basicOpen (x : X) (hx : x ∈ U) (f : Γ(X, U)) :
    X.evaluation U x hx f ≠ 0 ↔ x ∈ X.basicOpen f := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Iff (Ne ((X.evaluation U x hx).hom f) 0) (Membership.mem (X.basicOpen f) x)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma basicOpen_eq_bot_iff_forall_evaluation_eq_zero (f : X.presheaf.obj (op U)) :
    X.basicOpen f = ⊥ ↔ ∀ (x : U), X.evaluation U x x.property f = 0 :=
  X.toLocallyRingedSpace.basicOpen_eq_bot_iff_forall_evaluation_eq_zero f


/-- If `X ⟶ Y` is a morphism of locally ringed spaces and `x` a point of `X`, we obtain
a morphism of residue fields in the other direction. -/
def Hom.residueFieldMap (f : X.Hom Y) (x : X) :
    Y.residueField (f.base x) ⟶ X.residueField x :=
  CommRingCat.ofHom <| IsLocalRing.ResidueField.map (f.stalkMap x).hom


@[reassoc]
lemma residue_residueFieldMap (x : X) :
    Y.residue (f.base x) ≫ f.residueFieldMap x = f.stalkMap x ≫ X.residue x := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.residue (f.base x)) (AlgebraicGeom …
  -/
  simp [Hom.residueFieldMap]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.residue (f.base x)) (CommRingCat.o …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma residueFieldMap_id (x : X) :
    Hom.residueFieldMap (𝟙 X) x = 𝟙 (X.residueField x) :=
  LocallyRingedSpace.residueFieldMap_id _


@[simp]
lemma residueFieldMap_comp {Z : Scheme.{u}} (g : Y ⟶ Z) (x : X) :
    (f ≫ g).residueFieldMap x = g.residueFieldMap (f.base x) ≫ f.residueFieldMap x :=
  LocallyRingedSpace.residueFieldMap_comp _ _ _


@[reassoc]
lemma evaluation_naturality {V : Opens Y} (x : X) (hx : f.base x ∈ V) :
    Y.evaluation V (f.base x) hx ≫ f.residueFieldMap x =
      f.app V ≫ X.evaluation (f ⁻¹ᵁ V) x hx :=
  LocallyRingedSpace.evaluation_naturality f.1 ⟨x, hx⟩


lemma evaluation_naturality_apply {V : Opens Y} (x : X) (hx : f.base x ∈ V) (s) :
    f.residueFieldMap x (Y.evaluation V (f.base x) hx s) =
      X.evaluation (f ⁻¹ᵁ V) x hx (f.app V s) :=
  LocallyRingedSpace.evaluation_naturality_apply f.1 ⟨x, hx⟩ s


@[reassoc]
lemma Γevaluation_naturality (x : X) :
    Y.Γevaluation (f.base x) ≫ f.residueFieldMap x =
      f.c.app (op ⊤) ≫ X.Γevaluation x :=
  LocallyRingedSpace.Γevaluation_naturality f.toLRSHom x


lemma Γevaluation_naturality_apply (x : X) (a : Y.presheaf.obj (op ⊤)) :
    f.residueFieldMap x (Y.Γevaluation (f.base x) a) =
      X.Γevaluation x (f.c.app (op ⊤) a) :=
  LocallyRingedSpace.Γevaluation_naturality_apply f.toLRSHom x a


instance [IsOpenImmersion f] (x) : IsIso (f.residueFieldMap x) :=
  (IsLocalRing.ResidueField.mapEquiv
    (asIso (f.stalkMap x)).commRingCatIsoToRingEquiv).toCommRingCatIso.isIso_hom


/-- The isomorphism between residue fields of equal points. -/
def residueFieldCongr {x y : X} (h : x = y) :
    X.residueField x ≅ X.residueField y :=
              /-
                X✝ : AlgebraicGeometry.Scheme
                U : X✝.Opens
                X Y : AlgebraicGeometry.Scheme
                f : Quiver.Hom X Y
                x y : ↑↑X.toPresheafedSpace
                h : Eq x y
                ⊢ Eq (X.residueField x) (X.residueField y)
              -/
  eqToIso (by subst h; rfl)
                       /-
                         🎉 no goals
                       -/


@[simp]
lemma residueFieldCongr_refl {x : X} :
    X.residueFieldCongr (refl x) = Iso.refl _ := rfl


@[simp]
lemma residueFieldCongr_symm {x y : X} (e : x = y) :
    (X.residueFieldCongr e).symm = X.residueFieldCongr e.symm := rfl


@[simp]
lemma residueFieldCongr_inv {x y : X} (e : x = y) :
    (X.residueFieldCongr e).inv = (X.residueFieldCongr e.symm).hom := rfl


@[simp]
lemma residueFieldCongr_trans {x y z : X} (e : x = y) (e' : y = z) :
    X.residueFieldCongr e ≪≫ X.residueFieldCongr e' = X.residueFieldCongr (e.trans e') := by
  /-
    X : AlgebraicGeometry.Scheme
    x y z : ↑↑X.toPresheafedSpace
    e : Eq x y
    e' : Eq y z
    ⊢ Eq ((AlgebraicGeometry.Scheme.residueFieldCongr e).trans (AlgebraicGeometry. …
  -/
  subst e e'
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq ((AlgebraicGeometry.Scheme.residueFieldCongr ⋯).trans (AlgebraicGeometry. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma residueFieldCongr_trans_hom (X : Scheme) {x y z : X} (e : x = y) (e' : y = z) :
    (X.residueFieldCongr e).hom ≫ (X.residueFieldCongr e').hom =
      (X.residueFieldCongr (e.trans e')).hom := by
  /-
    X : AlgebraicGeometry.Scheme
    x y z : ↑↑X.toPresheafedSpace
    e : Eq x y
    e' : Eq y z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.residueFiel …
  -/
  subst e e'
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.residueFiel …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma residue_residueFieldCongr (X : Scheme) {x y : X} (h : x = y) :
    X.residue x ≫ (X.residueFieldCongr h).hom =
      (X.presheaf.stalkCongr (.of_eq h)).hom ≫ X.residue y := by
  /-
    X : AlgebraicGeometry.Scheme
    x y : ↑↑X.toPresheafedSpace
    h : Eq x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.residue x) (AlgebraicGeometry.Sche …
  -/
  subst h
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.residue x) (AlgebraicGeometry.Sche …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma Hom.residueFieldMap_congr {f g : X ⟶ Y} (e : f = g) (x : X) :
                                                   /-
                                                     X✝ : AlgebraicGeometry.Scheme
                                                     U : X✝.Opens
                                                     X Y : AlgebraicGeometry.Scheme
                                                     f✝ f g : Quiver.Hom X Y
                                                     e : Eq f g
                                                     x : ↑↑X.toPresheafedSpace
                                                     ⊢ Eq (f.base x) (g.base x)
                                                   -/
    f.residueFieldMap x = (Y.residueFieldCongr (by subst e; rfl)).hom ≫ g.residueFieldMap x := by
                                                            /-
                                                              🎉 no goals
                                                            -/
  /-
    X Y : AlgebraicGeometry.Scheme
    f g : Quiver.Hom X Y
    e : Eq f g
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.residueFieldMap f x) (CategoryTheory.Catego …
  -/
  subst e; simp
           /-
             🎉 no goals
           -/


/-- The canonical map `Spec κ(x) ⟶ X`. -/
def fromSpecResidueField (X : Scheme) (x : X) :
    Spec (X.residueField x) ⟶ X :=
  Spec.map (X.residue x) ≫ X.fromSpecStalk x


instance {X : Scheme.{u}} (x : X) : IsPreimmersion (X.fromSpecResidueField x) := by
  /-
    X✝¹ : AlgebraicGeometry.Scheme
    U : X✝¹.Opens
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ AlgebraicGeometry.IsPreimmersion (X.fromSpecResidueField x)
  -/
  dsimp only [Scheme.fromSpecResidueField]
  /-
    X✝¹ : AlgebraicGeometry.Scheme
    U : X✝¹.Opens
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp (Algebr …
  -/
  rw [IsPreimmersion.comp_iff]
  /-
    X✝¹ : AlgebraicGeometry.Scheme
    U : X✝¹.Opens
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ AlgebraicGeometry.IsPreimmersion (AlgebraicGeometry.Spec.map (X.residue x))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simps] noncomputable
instance (x : X) : (Spec (X.residueField x)).Over X := ⟨X.fromSpecResidueField x⟩


@[simps! over] noncomputable
instance (x : X) : (Spec (X.residueField x)).CanonicallyOver X where


@[reassoc (attr := simp)]
lemma residueFieldCongr_fromSpecResidueField {x y : X} (h : x = y) :
    Spec.map (X.residueFieldCongr h).hom ≫ X.fromSpecResidueField _ =
      X.fromSpecResidueField _ := by
  /-
    X : AlgebraicGeometry.Scheme
    x y : ↑↑X.toPresheafedSpace
    h : Eq x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  subst h; simp
           /-
             🎉 no goals
           -/


instance {x y : X} (h : x = y) : (Spec.map (X.residueFieldCongr h).hom).IsOver X where


@[reassoc (attr := simp)]
lemma Hom.Spec_map_residueFieldMap_fromSpecResidueField (x : X) :
    Spec.map (f.residueFieldMap x) ≫ Y.fromSpecResidueField _ =
      X.fromSpecResidueField x ≫ f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  dsimp only [fromSpecResidueField]
  rw [Category.assoc, ← Spec_map_stalkMap_fromSpecStalk, ← Spec.map_comp_assoc,
    ← Spec.map_comp_assoc]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Category …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance [X.Over Y] (x : X) : Spec.map ((X ↘ Y).residueFieldMap x) |>.IsOver Y where


@[simp]
lemma fromSpecResidueField_apply (x : X.carrier) (s : Spec (X.residueField x)) :
    (X.fromSpecResidueField x).base s = x := by
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    s : ↑↑(AlgebraicGeometry.Spec (X.residueField x)).toPresheafedSpace
    ⊢ Eq ((X.fromSpecResidueField x).base s) x
  -/
  simp [fromSpecResidueField]
  /-
    🎉 no goals
  -/


lemma range_fromSpecResidueField (x : X.carrier) :
    Set.range (X.fromSpecResidueField x).base = {x} := by
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (Set.range ⇑(X.fromSpecResidueField x).base) (Singleton.singleton x)
  -/
  ext s
  /-
    case h
    X : AlgebraicGeometry.Scheme
    x s : ↑↑X.toPresheafedSpace
    ⊢ Iff (Membership.mem (Set.range ⇑(X.fromSpecResidueField x).base) s) (Members …
  -/
  simp only [Set.mem_range, fromSpecResidueField_apply, Set.mem_singleton_iff, eq_comm (a := s)]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    x s : ↑↑X.toPresheafedSpace
    ⊢ Iff (Exists fun y => Eq x s) (Eq x s)
  -/
  constructor
    /-
      case h.mp
      X : AlgebraicGeometry.Scheme
      x s : ↑↑X.toPresheafedSpace
      ⊢ (Exists fun y => Eq x s) → Eq x s
    -/
  · rintro ⟨-, h⟩
    /-
      case h.mp.intro
      X : AlgebraicGeometry.Scheme
      x s : ↑↑X.toPresheafedSpace
      h : Eq x s
      ⊢ Eq x s
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X : AlgebraicGeometry.Scheme
      x s : ↑↑X.toPresheafedSpace
      ⊢ Eq x s → Exists fun y => Eq x s
    -/
  · rintro rfl
    /-
      case h.mpr
      X : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      ⊢ Exists fun y => Eq x x
    -/
    exact ⟨closedPoint (X.residueField x), rfl⟩
    /-
      🎉 no goals
    -/


lemma descResidueField_fromSpecResidueField {K : Type*} [Field K] (X : Scheme) {x}
    (f : X.presheaf.stalk x ⟶ .of K) [IsLocalHom f.hom] :
    Spec.map (X.descResidueField f) ≫
      X.fromSpecResidueField x = Spec.map f ≫ X.fromSpecStalk x := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    f : Quiver.Hom (X.presheaf.stalk x) (CommRingCat.of K)
    inst✝ : IsLocalHom f.hom
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  simp [fromSpecResidueField, ← Spec.map_comp_assoc]
  /-
    🎉 no goals
  -/


lemma descResidueField_stalkClosedPointTo_fromSpecResidueField
    (K : Type u) [Field K] (X : Scheme.{u}) (f : Spec (.of K) ⟶ X) :
    Spec.map (@descResidueField (CommRingCat.of K) _ X _ (Scheme.stalkClosedPointTo f)
        _) ≫
      X.fromSpecResidueField (f.base (closedPoint K)) = f := by
  /-
    K : Type u
    inst✝ : Field K
    X : AlgebraicGeometry.Scheme
    f : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  rw [X.descResidueField_fromSpecResidueField]
  /-
    K : Type u
    inst✝ : Field K
    X : AlgebraicGeometry.Scheme
    f : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  rw [Scheme.Spec_stalkClosedPointTo_fromSpecStalk]
  /-
    🎉 no goals
  -/


/-- A helper lemma to work with `AlgebraicGeometry.Scheme.SpecToEquivOfField`. -/
lemma SpecToEquivOfField_eq_iff {K : Type*} [Field K] {X : Scheme}
    {f₁ f₂ : Σ x : X.carrier, X.residueField x ⟶ .of K} :
    f₁ = f₂ ↔ ∃ e : f₁.1 = f₂.1, f₁.2 = (X.residueFieldCongr e).hom ≫ f₂.2 := by
  /-
    K : Type u_1
    inst✝ : Field K
    X : AlgebraicGeometry.Scheme
    f₁ f₂ : Sigma fun x => Quiver.Hom (X.residueField x) (CommRingCat.of K)
    ⊢ Iff (Eq f₁ f₂) (Exists fun e => Eq f₁.snd (CategoryTheory.CategoryStruct.com …
  -/
  constructor
    /-
      case mp
      K : Type u_1
      inst✝ : Field K
      X : AlgebraicGeometry.Scheme
      f₁ f₂ : Sigma fun x => Quiver.Hom (X.residueField x) (CommRingCat.of K)
      ⊢ Eq f₁ f₂ → Exists fun e => Eq f₁.snd (CategoryTheory.CategoryStruct.comp (Al …
    -/
  · rintro rfl
    /-
      case mp
      K : Type u_1
      inst✝ : Field K
      X : AlgebraicGeometry.Scheme
      f₁ : Sigma fun x => Quiver.Hom (X.residueField x) (CommRingCat.of K)
      ⊢ Exists fun e => Eq f₁.snd (CategoryTheory.CategoryStruct.comp (AlgebraicGeom …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_1
      inst✝ : Field K
      X : AlgebraicGeometry.Scheme
      f₁ f₂ : Sigma fun x => Quiver.Hom (X.residueField x) (CommRingCat.of K)
      ⊢ (Exists fun e => Eq f₁.snd (CategoryTheory.CategoryStruct.comp (AlgebraicGeo …
    -/
  · obtain ⟨f, _⟩ := f₁
    /-
      case mpr.mk
      K : Type u_1
      inst✝ : Field K
      X : AlgebraicGeometry.Scheme
      f₂ : Sigma fun x => Quiver.Hom (X.residueField x) (CommRingCat.of K)
      f : ↑↑X.toPresheafedSpace
      snd✝ : Quiver.Hom (X.residueField f) (CommRingCat.of K)
      ⊢ (Exists fun e => Eq ⟨f, snd✝⟩.snd (CategoryTheory.CategoryStruct.comp (Algeb …
    -/
    obtain ⟨g, _⟩ := f₂
    /-
      case mpr.mk.mk
      K : Type u_1
      inst✝ : Field K
      X : AlgebraicGeometry.Scheme
      f : ↑↑X.toPresheafedSpace
      snd✝¹ : Quiver.Hom (X.residueField f) (CommRingCat.of K)
      g : ↑↑X.toPresheafedSpace
      snd✝ : Quiver.Hom (X.residueField g) (CommRingCat.of K)
      ⊢ (Exists fun e => Eq ⟨f, snd✝¹⟩.snd (CategoryTheory.CategoryStruct.comp (Alge …
    -/
    rintro ⟨(rfl : f = g), h⟩
    /-
      case mpr.mk.mk.intro
      K : Type u_1
      inst✝ : Field K
      X : AlgebraicGeometry.Scheme
      f : ↑↑X.toPresheafedSpace
      snd✝¹ snd✝ : Quiver.Hom (X.residueField f) (CommRingCat.of K)
      h : Eq ⟨f, snd✝¹⟩.snd (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.S …
      ⊢ Eq ⟨f, snd✝¹⟩ ⟨f, snd✝⟩
    -/
    simpa
    /-
      🎉 no goals
    -/


/-- For a field `K` and a scheme `X`, the morphisms `Spec K ⟶ X` bijectively correspond
to pairs of points `x` of `X` and embeddings `κ(x) ⟶ K`. -/
def SpecToEquivOfField (K : Type u) [Field K] (X : Scheme.{u}) :
    (Spec (.of K) ⟶ X) ≃ Σ x, X.residueField x ⟶ .of K where
  toFun f :=
    ⟨_, X.descResidueField (Scheme.stalkClosedPointTo f)⟩
  invFun xf := Spec.map xf.2 ≫ X.fromSpecResidueField xf.1
  left_inv := Scheme.descResidueField_stalkClosedPointTo_fromSpecResidueField K X
  right_inv f := by
    /-
      X✝¹ : AlgebraicGeometry.Scheme
      U : X✝¹.Opens
      X✝ Y : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y
      K : Type u
      inst✝ : Field K
      X : AlgebraicGeometry.Scheme
      f : Sigma fun x => Quiver.Hom (X.residueField x) (CommRingCat.of K)
      ⊢ Eq ((fun f => ⟨f.base (IsLocalRing.closedPoint ↑(CommRingCat.of K)), Algebra …
    -/
    rw [SpecToEquivOfField_eq_iff]
    simp only [CommRingCat.coe_of, Scheme.comp_coeBase, TopCat.coe_comp, Function.comp_apply,
      Scheme.fromSpecResidueField_apply, exists_true_left]
    /-
      X✝¹ : AlgebraicGeometry.Scheme
      U : X✝¹.Opens
      X✝ Y : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y
      K : Type u
      inst✝ : Field K
      X : AlgebraicGeometry.Scheme
      f : Sigma fun x => Quiver.Hom (X.residueField x) (CommRingCat.of K)
      ⊢ Eq (AlgebraicGeometry.Scheme.descResidueField (AlgebraicGeometry.Scheme.stal …
    -/
    rw [← Spec.map_inj, Spec.map_comp, ← cancel_mono (X.fromSpecResidueField _)]
    /-
      X✝¹ : AlgebraicGeometry.Scheme
      U : X✝¹.Opens
      X✝ Y : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y
      K : Type u
      inst✝ : Field K
      X : AlgebraicGeometry.Scheme
      f : Sigma fun x => Quiver.Hom (X.residueField x) (CommRingCat.of K)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
    -/
    erw [Scheme.descResidueField_stalkClosedPointTo_fromSpecResidueField]
    /-
      X✝¹ : AlgebraicGeometry.Scheme
      U : X✝¹.Opens
      X✝ Y : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y
      K : Type u
      inst✝ : Field K
      X : AlgebraicGeometry.Scheme
      f : Sigma fun x => Quiver.Hom (X.residueField x) (CommRingCat.of K)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map f.snd) (X …
    -/
    simp
    /-
      🎉 no goals
    -/


