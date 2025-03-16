/-- The basic open set `D₊(f)` associated to `f : A`. -/
def basicOpen : (Proj 𝒜).Opens :=
  ProjectiveSpectrum.basicOpen 𝒜 f


@[simp]
theorem mem_basicOpen (x : Proj 𝒜) :
    x ∈ basicOpen 𝒜 f ↔ f ∉ x.asHomogeneousIdeal :=
  Iff.rfl


@[simp] theorem basicOpen_one : basicOpen 𝒜 1 = ⊤ := ProjectiveSpectrum.basicOpen_one ..


@[simp] theorem basicOpen_zero : basicOpen 𝒜 0 = ⊥ := ProjectiveSpectrum.basicOpen_zero ..


@[simp] theorem basicOpen_pow (n) (hn : 0 < n) : basicOpen 𝒜 (f ^ n) = basicOpen 𝒜 f :=
  ProjectiveSpectrum.basicOpen_pow 𝒜 f n hn


theorem basicOpen_mul : basicOpen 𝒜 (f * g) = basicOpen 𝒜 f ⊓ basicOpen 𝒜 g :=
  ProjectiveSpectrum.basicOpen_mul ..


theorem basicOpen_mono (hfg : f ∣ g) : basicOpen 𝒜 g ≤ basicOpen 𝒜 f :=
  (hfg.choose_spec ▸ basicOpen_mul 𝒜 f _).trans_le inf_le_left


theorem basicOpen_eq_iSup_proj (f : A) :
    basicOpen 𝒜 f = ⨆ i : ℕ, basicOpen 𝒜 (GradedAlgebra.proj 𝒜 i f) :=
  ProjectiveSpectrum.basicOpen_eq_union_of_projection ..


theorem isBasis_basicOpen :
    TopologicalSpace.Opens.IsBasis (Set.range (basicOpen 𝒜)) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    ⊢ TopologicalSpace.Opens.IsBasis (Set.range (AlgebraicGeometry.Proj.basicOpen  …
  -/
  delta TopologicalSpace.Opens.IsBasis
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    ⊢ TopologicalSpace.IsTopologicalBasis (Set.image SetLike.coe (Set.range (Algeb …
  -/
  convert ProjectiveSpectrum.isTopologicalBasis_basic_opens 𝒜
  /-
    case h.e'_3.h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    e_1✝ : Eq (↑↑(AlgebraicGeometry.Proj 𝒜).toPresheafedSpace) (ProjectiveSpectrum …
    ⊢ Eq (Set.image SetLike.coe (Set.range (AlgebraicGeometry.Proj.basicOpen 𝒜)))  …
  -/
  exact (Set.range_comp _ _).symm
  /-
    🎉 no goals
  -/


lemma iSup_basicOpen_eq_top {ι : Type*} (f : ι → A)
    (hf : (HomogeneousIdeal.irrelevant 𝒜).toIdeal ≤ Ideal.span (Set.range f)) :
    ⨆ i, Proj.basicOpen 𝒜 (f i) = ⊤ := by
  classical
  refine top_le_iff.mp fun x hx ↦ TopologicalSpace.Opens.mem_iSup.mpr ?_
  by_contra! H
  simp only [mem_basicOpen, Decidable.not_not] at H
  refine x.not_irrelevant_le (hf.trans ?_)
  rwa [Ideal.span_le, Set.range_subset_iff]


/-- The canonical map `(A_f)₀ ⟶ Γ(Proj A, D₊(f))`.
This is an isomorphism when `f` is homogeneous of positive degree. See `basicOpenIsoAway` below. -/
def awayToSection : CommRingCat.of (Away 𝒜 f) ⟶ Γ(Proj 𝒜, basicOpen 𝒜 f) :=
  ProjectiveSpectrum.Proj.awayToSection ..


/-- The canonical map `Proj A |_ D₊(f) ⟶ Spec (A_f)₀`.
This is an isomorphism when `f` is homogeneous of positive degree. See `basicOpenIsoSpec` below. -/
noncomputable
def basicOpenToSpec : (basicOpen 𝒜 f).toScheme ⟶ Spec (.of (Away 𝒜 f)) :=
  (basicOpen 𝒜 f).toSpecΓ ≫ Spec.map (awayToSection 𝒜 f)


lemma basicOpenToSpec_app_top :
    (basicOpenToSpec 𝒜 f).app ⊤ = (Scheme.ΓSpecIso _).hom ≫ awayToSection 𝒜 f ≫
      (basicOpen 𝒜 f).topIso.inv := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.Proj.basicOpenToSpec …
  -/
  rw [basicOpenToSpec]
  simp only [Scheme.comp_coeBase, TopologicalSpace.Opens.map_comp_obj,
    TopologicalSpace.Opens.map_top, Scheme.comp_app, Scheme.Opens.topIso_inv, eqToHom_op]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app (Al …
  -/
  erw [Scheme.comp_app]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app (Al …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The structure map `Proj A ⟶ Spec A₀`. -/
noncomputable
def toSpecZero : Proj 𝒜 ⟶ Spec (.of (𝒜 0)) :=
  (Scheme.topIso _).inv ≫ (Scheme.isoOfEq _ (basicOpen_one _)).inv ≫
    basicOpenToSpec 𝒜 1 ≫ Spec.map (CommRingCat.ofHom (fromZeroRingHom 𝒜 _))


/-- The canonical isomorphism `Proj A |_ D₊(f) ≅ Spec (A_f)₀`
when `f` is homogeneous of positive degree. -/
@[simps! (config := .lemmasOnly) hom]
noncomputable
def basicOpenIsoSpec : (basicOpen 𝒜 f).toScheme ≅ Spec (.of (Away 𝒜 f)) :=
  have : IsIso (basicOpenToSpec 𝒜 f) := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f g : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Proj.basicOpenToSpec 𝒜 f)
    -/
    apply (isIso_iff_of_reflects_iso _ Scheme.forgetToLocallyRingedSpace).mp ?_
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f g : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.ma …
    -/
    convert ProjectiveSpectrum.Proj.isIso_toSpec 𝒜 f f_deg hm using 1
    /-
      case h.e'_5.h
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f g : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      e_3✝ : Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.obj ↑(Algebraic …
      e_4✝ : Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.obj (AlgebraicG …
      ⊢ Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.map (AlgebraicGeomet …
    -/
    refine Eq.trans ?_ (ΓSpec.locallyRingedSpaceAdjunction.homEquiv_apply _ _ _).symm
    /-
      case h.e'_5.h
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f g : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      e_3✝ : Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.obj ↑(Algebraic …
      e_4✝ : Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.obj (AlgebraicG …
      ⊢ Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.map (AlgebraicGeomet …
    -/
    dsimp [basicOpenToSpec, Scheme.Opens.toSpecΓ]
    /-
      case h.e'_5.h
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f g : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      e_3✝ : Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.obj ↑(Algebraic …
      e_4✝ : Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.obj (AlgebraicG …
      ⊢ Eq (AlgebraicGeometry.Scheme.Hom.toLRSHom (CategoryTheory.CategoryStruct.com …
    -/
    simp only [eqToHom_op, Category.assoc, ← Spec.map_comp]
    /-
      case h.e'_5.h
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f g : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      e_3✝ : Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.obj ↑(Algebraic …
      e_4✝ : Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.obj (AlgebraicG …
      ⊢ Eq (AlgebraicGeometry.Scheme.Hom.toLRSHom (CategoryTheory.CategoryStruct.com …
    -/
    rfl
    /-
      🎉 no goals
    -/
  asIso (basicOpenToSpec 𝒜 f)


/-- The canonical isomorphism `(A_f)₀ ≅ Γ(Proj A, D₊(f))`
when `f` is homogeneous of positive degree. -/
@[simps! (config := .lemmasOnly) hom]
noncomputable
def basicOpenIsoAway : CommRingCat.of (Away 𝒜 f) ≅ Γ(Proj 𝒜, basicOpen 𝒜 f) :=
  have : IsIso (awayToSection 𝒜 f) := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f g : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Proj.awayToSection 𝒜 f)
    -/
    have := basicOpenToSpec_app_top 𝒜 f
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f g : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Eq (AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.Proj.basicOpenT …
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Proj.awayToSection 𝒜 f)
    -/
    rw [← Iso.inv_comp_eq, Iso.eq_comp_inv] at this
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f g : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Proj.awayToSection 𝒜 f)
    -/
    rw [← this, ← basicOpenIsoSpec_hom 𝒜 f f_deg hm]
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f g : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  asIso (awayToSection 𝒜 f)


/-- The open immersion `Spec (A_f)₀ ⟶ Proj A`. -/
noncomputable
def awayι : Spec (.of (Away 𝒜 f)) ⟶ Proj 𝒜 :=
  (basicOpenIsoSpec 𝒜 f f_deg hm).inv ≫ (Proj.basicOpen 𝒜 f).ι


instance : IsOpenImmersion (Proj.awayι 𝒜 f f_deg hm) :=
  IsOpenImmersion.comp _ _


lemma opensRange_awayι :
    (Proj.awayι 𝒜 f f_deg hm).opensRange = Proj.basicOpen 𝒜 f :=
  (Scheme.Hom.opensRange_comp_of_isIso _ _).trans (basicOpen 𝒜 f).opensRange_ι


include f_deg hm in
lemma isAffineOpen_basicOpen : IsAffineOpen (basicOpen 𝒜 f) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    ⊢ AlgebraicGeometry.IsAffineOpen (AlgebraicGeometry.Proj.basicOpen 𝒜 f)
  -/
  rw [← opensRange_awayι 𝒜 f f_deg hm]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    ⊢ AlgebraicGeometry.IsAffineOpen (AlgebraicGeometry.Scheme.Hom.opensRange (Alg …
  -/
  exact isAffineOpen_opensRange (awayι _ _ _ _)
  /-
    🎉 no goals
  -/


@[reassoc]
lemma awayι_toSpecZero : awayι 𝒜 f f_deg hm ≫ toSpecZero 𝒜 =
    Spec.map (CommRingCat.ofHom (fromZeroRingHom 𝒜 _)) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Proj.awayι 𝒜 f f_d …
  -/
  rw [toSpecZero, basicOpenToSpec, awayι]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc, Iso.inv_comp_eq, basicOpenIsoSpec_hom]
  have (U) (e : U = ⊤) : (basicOpen 𝒜 f).ι ≫ (Scheme.topIso _).inv ≫ (Scheme.isoOfEq _ e).inv =
      Scheme.homOfLE _ (le_top.trans_eq e.symm) := by
    simp only [← Category.assoc, Iso.comp_inv_eq]
    simp only [Scheme.topIso_hom, Category.assoc, Scheme.isoOfEq_hom_ι, Scheme.homOfLE_ι]
  rw [reassoc_of% this, ← Scheme.Opens.toSpecΓ_SpecMap_map_assoc, basicOpenToSpec, Category.assoc,
    ← Spec.map_comp, ← Spec.map_comp, ← Spec.map_comp]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    this : ∀ (U : (AlgebraicGeometry.Proj 𝒜).Opens) (e : Eq U Top.top), Eq (Catego …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Proj.basicOpen 𝒜 f …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma awayMap_awayToSection  :
    CommRingCat.ofHom (awayMap 𝒜 g_deg hx) ≫ awayToSection 𝒜 x =
      awayToSection 𝒜 f ≫ (Proj 𝒜).presheaf.map (homOfLE (basicOpen_mono _ _ _ ⟨_, hx⟩)).op := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (HomogeneousLocali …
  -/
  ext a
  /-
    case hf.a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    a : ↑(CommRingCat.of (HomogeneousLocalization.Away 𝒜 f))
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (HomogeneousLocal …
  -/
  apply Subtype.ext
  /-
    case hf.a.a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    a : ↑(CommRingCat.of (HomogeneousLocalization.Away 𝒜 f))
    ⊢ Eq ↑((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (HomogeneousLoca …
  -/
  ext ⟨i, hi⟩
  /-
    case hf.a.a.h.mk.a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    a : ↑(CommRingCat.of (HomogeneousLocalization.Away 𝒜 f))
    i : ↑(ProjectiveSpectrum.top 𝒜)
    hi : Membership.mem (Opposite.unop { unop := AlgebraicGeometry.Proj.basicOpen  …
    ⊢ Eq (HomogeneousLocalization.val (↑((CategoryTheory.CategoryStruct.comp (Comm …
  -/
  obtain ⟨⟨n, a, ⟨b, hb'⟩, i, rfl : _ = b⟩, rfl⟩ := mk_surjective a
  /-
    case hf.a.a.h.mk.a.intro.mk.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    i✝ : ↑(ProjectiveSpectrum.top 𝒜)
    hi : Membership.mem (Opposite.unop { unop := AlgebraicGeometry.Proj.basicOpen  …
    n : Nat
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) i)
    ⊢ Eq (HomogeneousLocalization.val (↑((CategoryTheory.CategoryStruct.comp (Comm …
  -/
  simp only [homOfLE_leOfHom, CommRingCat.hom_comp, RingHom.coe_comp, Function.comp_apply]
  /-
    case hf.a.a.h.mk.a.intro.mk.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    i✝ : ↑(ProjectiveSpectrum.top 𝒜)
    hi : Membership.mem (Opposite.unop { unop := AlgebraicGeometry.Proj.basicOpen  …
    n : Nat
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) i)
    ⊢ Eq (HomogeneousLocalization.val (↑((AlgebraicGeometry.Proj.awayToSection 𝒜 x …
  -/
  erw [ProjectiveSpectrum.Proj.awayToSection_apply]
  rw [val_awayMap_mk, Localization.mk_eq_mk', IsLocalization.map_mk',
    ← Localization.mk_eq_mk']
  /-
    case hf.a.a.h.mk.a.intro.mk.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    i✝ : ↑(ProjectiveSpectrum.top 𝒜)
    hi : Membership.mem (Opposite.unop { unop := AlgebraicGeometry.Proj.basicOpen  …
    n : Nat
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) i)
    ⊢ Eq (Localization.mk ((RingHom.id A) (HMul.hMul (↑a) (HPow.hPow g i))) ⟨(Ring …
  -/
  refine Localization.mk_eq_mk_iff.mpr ?_
  /-
    case hf.a.a.h.mk.a.intro.mk.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    i✝ : ↑(ProjectiveSpectrum.top 𝒜)
    hi : Membership.mem (Opposite.unop { unop := AlgebraicGeometry.Proj.basicOpen  …
    n : Nat
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) i)
    ⊢ (Localization.r (↑⟨i✝, hi⟩).asHomogeneousIdeal.toIdeal.primeCompl) { fst :=  …
  -/
  rw [Localization.r_iff_exists]
  /-
    case hf.a.a.h.mk.a.intro.mk.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    i✝ : ↑(ProjectiveSpectrum.top 𝒜)
    hi : Membership.mem (Opposite.unop { unop := AlgebraicGeometry.Proj.basicOpen  …
    n : Nat
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) i)
    ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul ↑{ fst := ↑((fun x_1 => { deg  …
  -/
  use 1
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    i✝ : ↑(ProjectiveSpectrum.top 𝒜)
    hi : Membership.mem (Opposite.unop { unop := AlgebraicGeometry.Proj.basicOpen  …
    n : Nat
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) i)
    ⊢ Eq (HMul.hMul (↑1) (HMul.hMul ↑{ fst := ↑((fun x_1 => { deg := x_1.deg, num  …
  -/
  simp only [OneMemClass.coe_one, RingHom.id_apply, one_mul, hx]
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    i✝ : ↑(ProjectiveSpectrum.top 𝒜)
    hi : Membership.mem (Opposite.unop { unop := AlgebraicGeometry.Proj.basicOpen  …
    n : Nat
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) i)
    ⊢ Eq (HMul.hMul (HPow.hPow f i) (HMul.hMul (↑a) (HPow.hPow g i))) (HMul.hMul ( …
  -/
  ring
  /-
    🎉 no goals
  -/


@[reassoc]
lemma basicOpenToSpec_SpecMap_awayMap :
    basicOpenToSpec 𝒜 x ≫ Spec.map (CommRingCat.ofHom (awayMap 𝒜 g_deg hx)) =
      (Proj 𝒜).homOfLE (basicOpen_mono _ _ _ ⟨_, hx⟩) ≫ basicOpenToSpec 𝒜 f := by
  rw [basicOpenToSpec, Category.assoc, ← Spec.map_comp, awayMap_awayToSection,
    Spec.map_comp, Scheme.Opens.toSpecΓ_SpecMap_map_assoc]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    x : A
    hx : Eq x (HMul.hMul f g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Proj 𝒜).homOfLE ⋯ …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma SpecMap_awayMap_awayι :
    Spec.map (CommRingCat.ofHom (awayMap 𝒜 g_deg hx)) ≫ awayι 𝒜 f f_deg hm =
      awayι 𝒜 x (hx ▸ SetLike.mul_mem_graded f_deg g_deg) (hm.trans_le (m.le_add_right m')) := by
  rw [awayι, awayι, Iso.eq_inv_comp, basicOpenIsoSpec_hom, basicOpenToSpec_SpecMap_awayMap_assoc,
  ← basicOpenIsoSpec_hom _ _ f_deg hm, Iso.hom_inv_id_assoc, Scheme.homOfLE_ι]


/-- The isomorphism `D₊(f) ×[Proj 𝒜] D₊(g) ≅ D₊(fg)`. -/
noncomputable
def pullbackAwayιIso :
    Limits.pullback (awayι 𝒜 f f_deg hm) (awayι 𝒜 g g_deg hm') ≅
      Spec (CommRingCat.of (Away 𝒜 x)) :=
    IsOpenImmersion.isoOfRangeEq (Limits.pullback.fst _ _ ≫ awayι 𝒜 f f_deg hm)
      (awayι 𝒜 x (hx ▸ SetLike.mul_mem_graded f_deg g_deg) (hm.trans_le (m.le_add_right m'))) <| by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f g✝ : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    hm' : LT.lt 0 m'
    x : A
    hx : Eq x (HMul.hMul f g)
    ⊢ Eq (Set.range ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pu …
  -/
  rw [IsOpenImmersion.range_pullback_to_base_of_left]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f g✝ : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    hm' : LT.lt 0 m'
    x : A
    hx : Eq x (HMul.hMul f g)
    ⊢ Eq (Inter.inter (Set.range ⇑(AlgebraicGeometry.Proj.awayι 𝒜 f f_deg hm).base …
  -/
  show ((awayι 𝒜 f _ _).opensRange ⊓ (awayι 𝒜 g _ _).opensRange).1 = (awayι 𝒜 _ _ _).opensRange.1
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f g✝ : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    hm' : LT.lt 0 m'
    x : A
    hx : Eq x (HMul.hMul f g)
    ⊢ Eq (Min.min (AlgebraicGeometry.Scheme.Hom.opensRange (AlgebraicGeometry.Proj …
  -/
  rw [opensRange_awayι, opensRange_awayι, opensRange_awayι, ← basicOpen_mul, hx]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma pullbackAwayιIso_hom_awayι :
    (pullbackAwayιIso 𝒜 f_deg hm g_deg hm' hx).hom ≫
      awayι 𝒜 x (hx ▸ SetLike.mul_mem_graded f_deg g_deg) (hm.trans_le (m.le_add_right m')) =
      Limits.pullback.fst _ _ ≫ awayι 𝒜 f f_deg hm :=
  IsOpenImmersion.isoOfRangeEq_hom_fac ..


@[reassoc (attr := simp)]
lemma pullbackAwayιIso_hom_SpecMap_awayMap_left :
    (pullbackAwayιIso 𝒜 f_deg hm g_deg hm' hx).hom ≫
      Spec.map (CommRingCat.ofHom (awayMap 𝒜 g_deg hx)) = Limits.pullback.fst _ _ := by
  rw [← cancel_mono (awayι 𝒜 f f_deg hm), ← pullbackAwayιIso_hom_awayι,
    Category.assoc, SpecMap_awayMap_awayι]


@[reassoc (attr := simp)]
lemma pullbackAwayιIso_hom_SpecMap_awayMap_right :
    (pullbackAwayιIso 𝒜 f_deg hm g_deg hm' hx).hom ≫
      Spec.map (CommRingCat.ofHom (awayMap 𝒜 f_deg (hx.trans (mul_comm _ _)))) =
      Limits.pullback.snd _ _ := by
  rw [← cancel_mono (awayι 𝒜 g g_deg hm'), ← Limits.pullback.condition,
    ← pullbackAwayιIso_hom_awayι 𝒜 f_deg hm g_deg hm' hx,
    Category.assoc, SpecMap_awayMap_awayι]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    hm' : LT.lt 0 m'
    x : A
    hx : Eq x (HMul.hMul f g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Proj.pullbackAwayι …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma pullbackAwayιIso_inv_fst :
    (pullbackAwayιIso 𝒜 f_deg hm g_deg hm' hx).inv ≫ Limits.pullback.fst _ _ =
      Spec.map (CommRingCat.ofHom (awayMap 𝒜 g_deg hx)) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    hm' : LT.lt 0 m'
    x : A
    hx : Eq x (HMul.hMul f g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Proj.pullbackAwayι …
  -/
  rw [← pullbackAwayιIso_hom_SpecMap_awayMap_left, Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma pullbackAwayιIso_inv_snd :
    (pullbackAwayιIso 𝒜 f_deg hm g_deg hm' hx).inv ≫ Limits.pullback.snd _ _ =
      Spec.map (CommRingCat.ofHom (awayMap 𝒜 f_deg (hx.trans (mul_comm _ _)))) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    m' : Nat
    g : A
    g_deg : Membership.mem (𝒜 m') g
    hm' : LT.lt 0 m'
    x : A
    hx : Eq x (HMul.hMul f g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Proj.pullbackAwayι …
  -/
  rw [← pullbackAwayιIso_hom_SpecMap_awayMap_right, Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


open TopologicalSpace.Opens in
/-- Given a family of homogeneous elements `f` of positive degree that spans the irrelevant ideal,
`Spec (A_f)₀ ⟶ Proj A` forms an affine open cover of `Proj A`. -/
noncomputable
def openCoverOfISupEqTop {ι : Type*} (f : ι → A) {m : ι → ℕ}
    (f_deg : ∀ i, f i ∈ 𝒜 (m i)) (hm : ∀ i, 0 < m i)
    (hf : (HomogeneousIdeal.irrelevant 𝒜).toIdeal ≤ Ideal.span (Set.range f)) :
    (Proj 𝒜).AffineOpenCover where
  J := ι
  obj i := .of (Away 𝒜 (f i))
  map i := awayι 𝒜 (f i) (f_deg i) (hm i)
  f x := (mem_iSup.mp ((iSup_basicOpen_eq_top 𝒜 f hf).ge (Set.mem_univ x))).choose
  covers x := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f✝ g✝ : A
      m✝ : Nat
      f_deg✝ : Membership.mem (𝒜 m✝) f✝
      hm✝ : LT.lt 0 m✝
      m' : Nat
      g : A
      g_deg : Membership.mem (𝒜 m') g
      hm' : LT.lt 0 m'
      x✝ : A
      hx : Eq x✝ (HMul.hMul f✝ g)
      ι : Type u_3
      f : ι → A
      m : ι → Nat
      f_deg : ∀ (i : ι), Membership.mem (𝒜 (m i)) (f i)
      hm : ∀ (i : ι), LT.lt 0 (m i)
      hf : LE.le (HomogeneousIdeal.irrelevant 𝒜).toIdeal (Ideal.span (Set.range f))
      x : ↑↑(AlgebraicGeometry.Proj 𝒜).toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun i => AlgebraicGeometry.Proj.awayι 𝒜 (f i) ⋯ …
    -/
    show x ∈ (awayι 𝒜 _ _ _).opensRange
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f✝ g✝ : A
      m✝ : Nat
      f_deg✝ : Membership.mem (𝒜 m✝) f✝
      hm✝ : LT.lt 0 m✝
      m' : Nat
      g : A
      g_deg : Membership.mem (𝒜 m') g
      hm' : LT.lt 0 m'
      x✝ : A
      hx : Eq x✝ (HMul.hMul f✝ g)
      ι : Type u_3
      f : ι → A
      m : ι → Nat
      f_deg : ∀ (i : ι), Membership.mem (𝒜 (m i)) (f i)
      hm : ∀ (i : ι), LT.lt 0 (m i)
      hf : LE.le (HomogeneousIdeal.irrelevant 𝒜).toIdeal (Ideal.span (Set.range f))
      x : ↑↑(AlgebraicGeometry.Proj 𝒜).toPresheafedSpace
      ⊢ Membership.mem (AlgebraicGeometry.Scheme.Hom.opensRange (AlgebraicGeometry.P …
    -/
    rw [opensRange_awayι]
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f✝ g✝ : A
      m✝ : Nat
      f_deg✝ : Membership.mem (𝒜 m✝) f✝
      hm✝ : LT.lt 0 m✝
      m' : Nat
      g : A
      g_deg : Membership.mem (𝒜 m') g
      hm' : LT.lt 0 m'
      x✝ : A
      hx : Eq x✝ (HMul.hMul f✝ g)
      ι : Type u_3
      f : ι → A
      m : ι → Nat
      f_deg : ∀ (i : ι), Membership.mem (𝒜 (m i)) (f i)
      hm : ∀ (i : ι), LT.lt 0 (m i)
      hf : LE.le (HomogeneousIdeal.irrelevant 𝒜).toIdeal (Ideal.span (Set.range f))
      x : ↑↑(AlgebraicGeometry.Proj 𝒜).toPresheafedSpace
      ⊢ Membership.mem (AlgebraicGeometry.Proj.basicOpen 𝒜 (f ((fun x => ⋯.choose) x …
    -/
    exact (mem_iSup.mp ((iSup_basicOpen_eq_top 𝒜 f hf).ge (Set.mem_univ x))).choose_spec
    /-
      🎉 no goals
    -/


/-- `Proj A` is covered by `Spec (A_f)₀` for all homogeneous elements of positive degree. -/
noncomputable
def affineOpenCover : (Proj 𝒜).AffineOpenCover :=
  openCoverOfISupEqTop 𝒜 (ι := Σ i : PNat, 𝒜 i) (m := fun i ↦ i.1) (fun i ↦ i.2) (fun i ↦ i.2.2)
    (fun i ↦ i.1.2) <| by
  classical
  intro z hz
  rw [← DirectSum.sum_support_decompose 𝒜 z]
  refine Ideal.sum_mem _ fun c hc ↦ if hc0 : c = 0 then ?_ else
    Ideal.subset_span ⟨⟨⟨c, Nat.pos_iff_ne_zero.mpr hc0⟩, _⟩, rfl⟩
  convert Ideal.zero_mem _
  subst hc0
  exact hz


/-- The stalk of `Proj A` at `x` is the degree `0` part of the localization of `A` at `x`. -/
noncomputable
def stalkIso (x : Proj 𝒜) :
    (Proj 𝒜).presheaf.stalk x ≅ .of (AtPrime 𝒜 x.asHomogeneousIdeal.toIdeal) :=
  (stalkIso' 𝒜 x).toCommRingCatIso


