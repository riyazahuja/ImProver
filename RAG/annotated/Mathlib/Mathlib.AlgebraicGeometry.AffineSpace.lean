local notation3 "ℤ[" n "]" => CommRingCat.of (MvPolynomial n (ULift ℤ))

local notation3 "ℤ[" n "].{" u "}" => CommRingCat.of (MvPolynomial n (ULift.{u} ℤ))


/-- `𝔸(n; S)` is the affine `n`-space over `S`.
Note that `n` is an arbitrary index type (e.g. `Fin m`). -/
def AffineSpace (n : Type v) (S : Scheme.{max u v}) : Scheme.{max u v} :=
  pullback (terminal.from S) (terminal.from (Spec ℤ[n]))


/-- `𝔸(n; S)` is the affine `n`-space over `S`. -/
scoped [AlgebraicGeometry] notation "𝔸("n"; "S")" => AffineSpace n S


variable {n} in
lemma of_mvPolynomial_int_ext {R} {f g : ℤ[n] ⟶ R} (h : ∀ i, f (.X i) = g (.X i)) : f = g := by
  suffices f.hom.comp (MvPolynomial.mapEquiv _ ULift.ringEquiv.symm).toRingHom =
      g.hom.comp (MvPolynomial.mapEquiv _ ULift.ringEquiv.symm).toRingHom by
    ext x
    · obtain ⟨x⟩ := x
      simpa [-map_intCast, -eq_intCast] using DFunLike.congr_fun this (C x)
    · simpa [-map_intCast, -eq_intCast] using DFunLike.congr_fun this (X x)
  /-
    n : Type v
    R : CommRingCat
    f g : Quiver.Hom (CommRingCat.of (MvPolynomial n (ULift.{u_1, 0} Int))) R
    h : ∀ (i : n), Eq (f.hom (MvPolynomial.X i)) (g.hom (MvPolynomial.X i))
    ⊢ Eq (f.hom.comp (MvPolynomial.mapEquiv n ULift.ringEquiv.symm).toRingHom) (g. …
  -/
  ext1
    /-
      case hC
      n : Type v
      R : CommRingCat
      f g : Quiver.Hom (CommRingCat.of (MvPolynomial n (ULift.{u_1, 0} Int))) R
      h : ∀ (i : n), Eq (f.hom (MvPolynomial.X i)) (g.hom (MvPolynomial.X i))
      ⊢ Eq ((f.hom.comp (MvPolynomial.mapEquiv n ULift.ringEquiv.symm).toRingHom).co …
    -/
  · exact RingHom.ext_int _ _
    /-
      🎉 no goals
    -/
    /-
      case hX
      n : Type v
      R : CommRingCat
      f g : Quiver.Hom (CommRingCat.of (MvPolynomial n (ULift.{u_1, 0} Int))) R
      h : ∀ (i : n), Eq (f.hom (MvPolynomial.X i)) (g.hom (MvPolynomial.X i))
      i✝ : n
      ⊢ Eq ((f.hom.comp (MvPolynomial.mapEquiv n ULift.ringEquiv.symm).toRingHom) (M …
    -/
  · simpa using h _
    /-
      🎉 no goals
    -/



@[simps (config := .lemmasOnly)]
instance over : 𝔸(n; S).CanonicallyOver S where
  hom := pullback.fst _ _


/-- The map from the affine `n`-space over `S` to the integral model `Spec ℤ[n]`. -/
def toSpecMvPoly : 𝔸(n; S) ⟶ Spec ℤ[n] := pullback.snd _ _


/--
Morphisms into `Spec ℤ[n]` are equivalent the choice of `n` global sections.
Use `homOverEquiv` instead.
-/
@[simps]
def toSpecMvPolyIntEquiv : (X ⟶ Spec ℤ[n]) ≃ (n → Γ(X, ⊤)) where
  toFun f i := f.appTop ((Scheme.ΓSpecIso ℤ[n]).inv (.X i))
  invFun v := X.toSpecΓ ≫ Spec.map
    (CommRingCat.ofHom (MvPolynomial.eval₂Hom ((algebraMap ℤ _).comp ULift.ringEquiv.toRingHom) v))
  left_inv f := by
    /-
      n : Type v
      S X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X (AlgebraicGeometry.Spec (CommRingCat.of (MvPolynomial n (ULif …
      ⊢ Eq ((fun v => CategoryTheory.CategoryStruct.comp X.toSpecΓ (AlgebraicGeometr …
    -/
    apply (ΓSpec.adjunction.homEquiv _ _).symm.injective
    /-
      case a
      n : Type v
      S X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X (AlgebraicGeometry.Spec (CommRingCat.of (MvPolynomial n (ULif …
      ⊢ Eq ((AlgebraicGeometry.ΓSpec.adjunction.homEquiv X { unop := CommRingCat.mk✝ …
    -/
    apply Quiver.Hom.unop_inj
    /-
      case a.a
      n : Type v
      S X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X (AlgebraicGeometry.Spec (CommRingCat.of (MvPolynomial n (ULif …
      ⊢ Eq ((AlgebraicGeometry.ΓSpec.adjunction.homEquiv X { unop := CommRingCat.mk✝ …
    -/
    rw [Adjunction.homEquiv_symm_apply, Adjunction.homEquiv_symm_apply]
    simp only [Functor.rightOp_obj, Scheme.Γ_obj, Scheme.Spec_obj, algebraMap_int_eq,
      RingEquiv.toRingHom_eq_coe, TopologicalSpace.Opens.map_top, Functor.rightOp_map, op_comp,
      Scheme.Γ_map, unop_comp, Quiver.Hom.unop_op, Scheme.comp_app, Scheme.toSpecΓ_appTop,
      Scheme.ΓSpecIso_naturality, ΓSpec.adjunction_counit_app, Category.assoc,
      Iso.cancel_iso_inv_left, ← Iso.eq_inv_comp]
    /-
      case a.a
      n : Type v
      S X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X (AlgebraicGeometry.Spec (CommRingCat.of (MvPolynomial n (ULif …
      ⊢ Eq (CommRingCat.ofHom (MvPolynomial.eval₂Hom ((Int.castRingHom ↑(X.presheaf. …
    -/
    apply of_mvPolynomial_int_ext
    /-
      case a.a.h
      n : Type v
      S X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X (AlgebraicGeometry.Spec (CommRingCat.of (MvPolynomial n (ULif …
      ⊢ ∀ (i : n), Eq ((CommRingCat.ofHom (MvPolynomial.eval₂Hom ((Int.castRingHom ↑ …
    -/
    intro i
    /-
      case a.a.h
      n : Type v
      S X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X (AlgebraicGeometry.Spec (CommRingCat.of (MvPolynomial n (ULif …
      i : n
      ⊢ Eq ((CommRingCat.ofHom (MvPolynomial.eval₂Hom ((Int.castRingHom ↑(X.presheaf …
    -/
    rw [coe_eval₂Hom, eval₂_X]
    /-
      case a.a.h
      n : Type v
      S X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X (AlgebraicGeometry.Spec (CommRingCat.of (MvPolynomial n (ULif …
      i : n
      ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.appTop f).hom ((AlgebraicGeometry.Scheme.Γ …
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv v := by
    /-
      n : Type v
      S X : AlgebraicGeometry.Scheme
      v : n → ↑(X.presheaf.obj { unop := Top.top })
      ⊢ Eq ((fun f i => (AlgebraicGeometry.Scheme.Hom.appTop f).hom ((AlgebraicGeome …
    -/
    ext i
    simp only [algebraMap_int_eq, RingEquiv.toRingHom_eq_coe, Scheme.comp_coeBase,
      TopologicalSpace.Opens.map_comp_obj, TopologicalSpace.Opens.map_top, Scheme.comp_app,
      Scheme.toSpecΓ_appTop, Scheme.ΓSpecIso_naturality, CommRingCat.comp_apply,
      CommRingCat.coe_of]
    -- TODO: why does `simp` not apply this lemma?
    /-
      case h
      n : Type v
      S X : AlgebraicGeometry.Scheme
      v : n → ↑(X.presheaf.obj { unop := Top.top })
      i : n
      ⊢ Eq ((MvPolynomial.eval₂Hom ((Int.castRingHom ↑(X.presheaf.toPrefunctor.1 { u …
    -/
    rw [CommRingCat.hom_inv_apply]
    /-
      case h
      n : Type v
      S X : AlgebraicGeometry.Scheme
      v : n → ↑(X.presheaf.obj { unop := Top.top })
      i : n
      ⊢ Eq ((MvPolynomial.eval₂Hom ((Int.castRingHom ↑(X.presheaf.toPrefunctor.1 { u …
    -/
    simp
    /-
      🎉 no goals
    -/


lemma toSpecMvPolyIntEquiv_comp {X Y : Scheme} (f : X ⟶ Y) (g : Y ⟶ Spec ℤ[n]) (i) :
    toSpecMvPolyIntEquiv n (f ≫ g) i = f.appTop (toSpecMvPolyIntEquiv n g i) := rfl


variable {n} in
/-- The standard coordinates of `𝔸(n; S)`. -/
def coord (i : n) : Γ(𝔸(n; S), ⊤) := toSpecMvPolyIntEquiv _ (toSpecMvPoly n S) i


/-- The morphism `X ⟶ 𝔸(n; S)` given by a `X ⟶ S` and a choice of `n`-coordinate functions. -/
def homOfVector (f : X ⟶ S) (v : n → Γ(X, ⊤)) : X ⟶ 𝔸(n; S) :=
                                                        /-
                                                          n : Type v
                                                          S X : AlgebraicGeometry.Scheme
                                                          f : Quiver.Hom X S
                                                          v : n → ↑(X.presheaf.obj { unop := Top.top })
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.terminal.fro …
                                                        -/
  pullback.lift f ((toSpecMvPolyIntEquiv n).symm v) (by simp)
                                                        /-
                                                          🎉 no goals
                                                        -/


@[reassoc (attr := simp)]
lemma homOfVector_over : homOfVector f v ≫ 𝔸(n; S) ↘ S = f :=
  pullback.lift_fst _ _ _


@[reassoc]
lemma homOfVector_toSpecMvPoly :
    homOfVector f v ≫ toSpecMvPoly n S = (toSpecMvPolyIntEquiv n).symm v :=
  pullback.lift_snd _ _ _


@[simp]
lemma homOfVector_appTop_coord (i) :
    (homOfVector f v).appTop (coord S i) = v i := by
  rw [coord, ← toSpecMvPolyIntEquiv_comp, homOfVector_toSpecMvPoly,
    Equiv.apply_symm_apply]


@[ext 1100]
lemma hom_ext {f g : X ⟶ 𝔸(n; S)}
    (h₁ : f ≫ 𝔸(n; S) ↘ S = g ≫ 𝔸(n; S) ↘ S)
    (h₂ : ∀ i, f.appTop (coord S i) = g.appTop (coord S i)) : f = g := by
  /-
    n : Type v
    S X : AlgebraicGeometry.Scheme
    f g : Quiver.Hom X (AlgebraicGeometry.AffineSpace n S)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.over (AlgebraicG …
    h₂ : ∀ (i : n), Eq ((AlgebraicGeometry.Scheme.Hom.appTop f).hom (AlgebraicGeom …
    ⊢ Eq f g
  -/
  apply pullback.hom_ext h₁
  /-
    n : Type v
    S X : AlgebraicGeometry.Scheme
    f g : Quiver.Hom X (AlgebraicGeometry.AffineSpace n S)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.over (AlgebraicG …
    h₂ : ∀ (i : n), Eq ((AlgebraicGeometry.Scheme.Hom.appTop f).hom (AlgebraicGeom …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pullback.snd …
  -/
  show f ≫ toSpecMvPoly _ _ = g ≫ toSpecMvPoly _ _
  /-
    n : Type v
    S X : AlgebraicGeometry.Scheme
    f g : Quiver.Hom X (AlgebraicGeometry.AffineSpace n S)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.over (AlgebraicG …
    h₂ : ∀ (i : n), Eq ((AlgebraicGeometry.Scheme.Hom.appTop f).hom (AlgebraicGeom …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicGeometry.AffineSpace.toSp …
  -/
  apply (toSpecMvPolyIntEquiv n).injective
  /-
    case a
    n : Type v
    S X : AlgebraicGeometry.Scheme
    f g : Quiver.Hom X (AlgebraicGeometry.AffineSpace n S)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.over (AlgebraicG …
    h₂ : ∀ (i : n), Eq ((AlgebraicGeometry.Scheme.Hom.appTop f).hom (AlgebraicGeom …
    ⊢ Eq ((AlgebraicGeometry.AffineSpace.toSpecMvPolyIntEquiv n) (CategoryTheory.C …
  -/
  ext i
  /-
    case a.h
    n : Type v
    S X : AlgebraicGeometry.Scheme
    f g : Quiver.Hom X (AlgebraicGeometry.AffineSpace n S)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.over (AlgebraicG …
    h₂ : ∀ (i : n), Eq ((AlgebraicGeometry.Scheme.Hom.appTop f).hom (AlgebraicGeom …
    i : n
    ⊢ Eq ((AlgebraicGeometry.AffineSpace.toSpecMvPolyIntEquiv n) (CategoryTheory.C …
  -/
  rw [toSpecMvPolyIntEquiv_comp, toSpecMvPolyIntEquiv_comp]
  /-
    case a.h
    n : Type v
    S X : AlgebraicGeometry.Scheme
    f g : Quiver.Hom X (AlgebraicGeometry.AffineSpace n S)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.over (AlgebraicG …
    h₂ : ∀ (i : n), Eq ((AlgebraicGeometry.Scheme.Hom.appTop f).hom (AlgebraicGeom …
    i : n
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.appTop f).hom ((AlgebraicGeometry.AffineSp …
  -/
  exact h₂ i
  /-
    🎉 no goals
  -/


@[reassoc]
lemma comp_homOfVector {X Y : Scheme} (v : n → Γ(Y, ⊤)) (f : X ⟶ Y) (g : Y ⟶ S) :
    f ≫ homOfVector g v = homOfVector (f ≫ g) (f.appTop ∘ v) := by
  /-
    n : Type v
    S X Y : AlgebraicGeometry.Scheme
    v : n → ↑(Y.presheaf.obj { unop := Top.top })
    f : Quiver.Hom X Y
    g : Quiver.Hom Y S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicGeometry.AffineSpace.homO …
  -/
           /-
             🎉 no goals
           -/
  ext1 <;> simp
           /-
             🎉 no goals
           -/


instance (v : n → Γ(X, ⊤)) : (homOfVector (X ↘ S) v).IsOver S where


/-- `S`-morphisms into `Spec 𝔸(n; S)` are equivalent to the choice of `n` global sections. -/
@[simps]
def homOverEquiv : { f : X ⟶ 𝔸(n; S) // f.IsOver S } ≃ (n → Γ(X, ⊤)) where
  toFun f i := f.1.appTop (coord S i)
  invFun v := ⟨homOfVector (X ↘ S) v, inferInstance⟩
  left_inv f := by
    /-
      n : Type v
      S X : AlgebraicGeometry.Scheme
      inst✝ : X.Over S
      f : Subtype fun f => AlgebraicGeometry.Scheme.Hom.IsOver f S
      ⊢ Eq ((fun v => ⟨AlgebraicGeometry.AffineSpace.homOfVector (CategoryTheory.ove …
    -/
    ext : 2
      /-
        case a.h₁
        n : Type v
        S X : AlgebraicGeometry.Scheme
        inst✝ : X.Over S
        f : Subtype fun f => AlgebraicGeometry.Scheme.Hom.IsOver f S
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑((fun v => ⟨AlgebraicGeometry.Affin …
      -/
    · simp [f.2.1]
      /-
        🎉 no goals
      -/
      /-
        case a.h₂
        n : Type v
        S X : AlgebraicGeometry.Scheme
        inst✝ : X.Over S
        f : Subtype fun f => AlgebraicGeometry.Scheme.Hom.IsOver f S
        i✝ : n
        ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.appTop ↑((fun v => ⟨AlgebraicGeometry.Affi …
      -/
    · rw [homOfVector_appTop_coord]
      /-
        🎉 no goals
      -/
                    /-
                      n : Type v
                      S X : AlgebraicGeometry.Scheme
                      inst✝ : X.Over S
                      v : n → ↑(X.presheaf.obj { unop := Top.top })
                      ⊢ Eq ((fun f i => (AlgebraicGeometry.Scheme.Hom.appTop ↑f).hom (AlgebraicGeome …
                    -/
  right_inv v := by ext i; simp [-TopologicalSpace.Opens.map_top, homOfVector_appTop_coord]
                           /-
                             🎉 no goals
                           -/


variable (n) in
/--
The affine space over an affine base is isomorphic to the spectrum of the polynomial ring.
Also see `AffineSpace.SpecIso`.
-/
@[simps (config := .lemmasOnly) hom inv]
def isoOfIsAffine [IsAffine S] :
    𝔸(n; S) ≅ Spec (.of (MvPolynomial n Γ(S, ⊤))) where
      hom := 𝔸(n; S).toSpecΓ ≫ Spec.map (CommRingCat.ofHom
        (eval₂Hom ((𝔸(n; S) ↘ S).appTop).hom (coord S)))
      inv := homOfVector (Spec.map (CommRingCat.ofHom C) ≫ S.isoSpec.inv)
        ((Scheme.ΓSpecIso (.of (MvPolynomial n Γ(S, ⊤)))).inv ∘ MvPolynomial.X)
      hom_inv_id := by
        /-
          n : Type v
          S X : AlgebraicGeometry.Scheme
          inst✝¹ : X.Over S
          inst✝ : AlgebraicGeometry.IsAffine S
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        ext1
          /-
            case h₁
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · simp only [Category.assoc, homOfVector_over, Category.id_comp]
          rw [← Spec.map_comp_assoc, ← CommRingCat.ofHom_comp, eval₂Hom_comp_C,
            CommRingCat.ofHom_hom, ← Scheme.toSpecΓ_naturality_assoc]
          /-
            case h₁
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.over (AlgebraicGeomet …
          -/
          simp [Scheme.isoSpec]
          /-
            🎉 no goals
          -/
        · simp only [Category.assoc, Scheme.comp_app, Scheme.comp_coeBase,
            TopologicalSpace.Opens.map_comp_obj, TopologicalSpace.Opens.map_top,
            Scheme.toSpecΓ_appTop, Scheme.ΓSpecIso_naturality, CommRingCat.comp_apply,
            homOfVector_appTop_coord, Function.comp_apply, CommRingCat.coe_of, Scheme.id_app,
            CommRingCat.id_apply]
          -- TODO: why does `simp` not apply this?
          /-
            case h₂
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            i✝ : n
            ⊢ Eq ((MvPolynomial.eval₂Hom (AlgebraicGeometry.Scheme.Hom.appTop (CategoryThe …
          -/
          rw [CommRingCat.hom_inv_apply]
          /-
            case h₂
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            i✝ : n
            ⊢ Eq ((MvPolynomial.eval₂Hom (AlgebraicGeometry.Scheme.Hom.appTop (CategoryThe …
          -/
          exact eval₂_X _ _ _
          /-
            🎉 no goals
          -/
      inv_hom_id := by
        /-
          n : Type v
          S X : AlgebraicGeometry.Scheme
          inst✝¹ : X.Over S
          inst✝ : AlgebraicGeometry.IsAffine S
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.AffineSpace.homOfV …
        -/
        apply ext_of_isAffine
        simp only [Scheme.comp_coeBase, TopologicalSpace.Opens.map_comp_obj,
          TopologicalSpace.Opens.map_top, Scheme.comp_app, Scheme.toSpecΓ_appTop,
          Scheme.ΓSpecIso_naturality, Category.assoc, Scheme.id_app, ← Iso.eq_inv_comp,
          Category.comp_id]
        /-
          case e
          n : Type v
          S X : AlgebraicGeometry.Scheme
          inst✝¹ : X.Over S
          inst✝ : AlgebraicGeometry.IsAffine S
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (MvPolynomial.eval …
        -/
        ext : 1
        /-
          case e.hf
          n : Type v
          S X : AlgebraicGeometry.Scheme
          inst✝¹ : X.Over S
          inst✝ : AlgebraicGeometry.IsAffine S
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (MvPolynomial.eval …
        -/
        apply ringHom_ext'
          /-
            case e.hf.hC
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (MvPolynomial.eva …
          -/
        · show _ = (CommRingCat.ofHom C ≫ _).hom
          rw [CommRingCat.hom_comp, RingHom.comp_assoc, eval₂Hom_comp_C,
            ← CommRingCat.hom_comp, ← CommRingCat.hom_ext_iff,
            ← cancel_mono (Scheme.ΓSpecIso _).hom]
          /-
            case e.hf.hC
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          rw [← Scheme.comp_appTop, homOfVector_over, Scheme.comp_appTop]
          simp only [Category.assoc, Scheme.ΓSpecIso_naturality, CommRingCat.of_carrier,
            ← Scheme.toSpecΓ_appTop]
          /-
            case e.hf.hC
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.appTop  …
          -/
          rw [← Scheme.comp_appTop_assoc, Scheme.isoSpec, asIso_inv, IsIso.hom_inv_id]
          /-
            case e.hf.hC
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.appTop  …
          -/
          simp
          /-
            🎉 no goals
          -/
          /-
            case e.hf.hX
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            ⊢ ∀ (i : n), Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (MvPol …
          -/
        · intro i
          /-
            case e.hf.hX
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            i : n
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (MvPolynomial.eva …
          -/
          rw [CommRingCat.comp_apply, coe_eval₂Hom]
          /-
            case e.hf.hX
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            i : n
            ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.AffineSpace.homOfVe …
          -/
          simp only [eval₂_X]
          /-
            case e.hf.hX
            n : Type v
            S X : AlgebraicGeometry.Scheme
            inst✝¹ : X.Over S
            inst✝ : AlgebraicGeometry.IsAffine S
            i : n
            ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.AffineSpace.homOfVe …
          -/
          exact homOfVector_appTop_coord _ _ _
          /-
            🎉 no goals
          -/


@[simp]
lemma isoOfIsAffine_hom_appTop [IsAffine S] :
    (isoOfIsAffine n S).hom.appTop =
      (Scheme.ΓSpecIso _).hom ≫ CommRingCat.ofHom
        (eval₂Hom ((𝔸(n; S) ↘ S).appTop).hom (coord S)) := by
  /-
    n : Type v
    S : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine S
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop (AlgebraicGeometry.AffineSpace.isoOf …
  -/
  simp [isoOfIsAffine_hom]
  /-
    🎉 no goals
  -/


@[simp]
lemma isoOfIsAffine_inv_appTop_coord [IsAffine S] (i) :
    (isoOfIsAffine n S).inv.appTop (coord _ i) = (Scheme.ΓSpecIso (.of _)).inv (.X i) :=
  homOfVector_appTop_coord _ _ _


@[reassoc (attr := simp)]
lemma isoOfIsAffine_inv_over [IsAffine S] :
    (isoOfIsAffine n S).inv ≫ 𝔸(n; S) ↘ S = Spec.map (CommRingCat.ofHom C) ≫ S.isoSpec.inv :=
  pullback.lift_fst _ _ _


instance [IsAffine S] : IsAffine 𝔸(n; S) := isAffine_of_isIso (isoOfIsAffine n S).hom


variable (n) in
/-- The affine space over an affine base is isomorphic to the spectrum of the polynomial ring. -/
def SpecIso (R : CommRingCat.{max u v}) :
    𝔸(n; Spec R) ≅ Spec (.of (MvPolynomial n R)) :=
  isoOfIsAffine _ _ ≪≫ Scheme.Spec.mapIso (MvPolynomial.mapEquiv _
    (Scheme.ΓSpecIso R).symm.commRingCatIsoToRingEquiv).toCommRingCatIso.op


@[simp]
lemma SpecIso_hom_appTop (R : CommRingCat.{max u v}) :
    (SpecIso n R).hom.appTop = (Scheme.ΓSpecIso _).hom ≫
      CommRingCat.ofHom (eval₂Hom ((Scheme.ΓSpecIso _).inv ≫
        (𝔸(n; Spec R) ↘ Spec R).appTop).hom (coord (Spec R))) := by
  simp only [SpecIso, Iso.trans_hom, Functor.mapIso_hom, Iso.op_hom,
    RingEquiv.toRingHom_eq_coe, Scheme.Spec_map, Quiver.Hom.unop_op, Scheme.comp_coeBase,
    TopologicalSpace.Opens.map_comp_obj, TopologicalSpace.Opens.map_top, Scheme.comp_app,
    isoOfIsAffine_hom_appTop, Scheme.ΓSpecIso_naturality_assoc]
  /-
    n : Type v
    R : CommRingCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.ΓSpecIso (C …
  -/
  congr 1
  /-
    case e_a
    n : Type v
    R : CommRingCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (MvPolynomial.mapEquiv n (AlgebraicGe …
  -/
  ext : 1
  /-
    case e_a.hf
    n : Type v
    R : CommRingCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (MvPolynomial.mapEquiv n (AlgebraicGe …
  -/
  apply ringHom_ext'
    /-
      case e_a.hf.hC
      n : Type v
      R : CommRingCat
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (MvPolynomial.mapEquiv n (AlgebraicG …
    -/
  · ext; simp
         /-
           🎉 no goals
         -/
    /-
      case e_a.hf.hX
      n : Type v
      R : CommRingCat
      ⊢ ∀ (i : n), Eq ((CategoryTheory.CategoryStruct.comp (MvPolynomial.mapEquiv n  …
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp]
lemma SpecIso_inv_appTop_coord (R : CommRingCat.{max u v}) (i) :
    (SpecIso n R).inv.appTop (coord _ i) = (Scheme.ΓSpecIso (.of _)).inv (.X i) := by
  simp only [SpecIso, Iso.trans_inv, Functor.mapIso_inv, Iso.op_inv,
    mapEquiv_symm, RingEquiv.toRingHom_eq_coe, Scheme.Spec_map, Quiver.Hom.unop_op,
    Scheme.comp_coeBase, TopologicalSpace.Opens.map_comp_obj, TopologicalSpace.Opens.map_top,
    Scheme.comp_app, CommRingCat.comp_apply]
  rw [isoOfIsAffine_inv_appTop_coord, ← CommRingCat.comp_apply, ← Scheme.ΓSpecIso_inv_naturality,
      CommRingCat.comp_apply]
  /-
    n : Type v
    R : CommRingCat
    i : n
    ⊢ Eq ((AlgebraicGeometry.Scheme.ΓSpecIso (CommRingCat.mk✝ (MvPolynomial n ↑R)) …
  -/
  congr 1
  /-
    case h.e_6.h
    n : Type v
    R : CommRingCat
    i : n
    ⊢ Eq ((MvPolynomial.mapEquiv n (AlgebraicGeometry.Scheme.ΓSpecIso R).symm.comm …
  -/
  exact map_X _ _
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma SpecIso_inv_over (R : CommRingCat.{max u v}) :
    (SpecIso n R).inv ≫ 𝔸(n; Spec R) ↘ Spec R = Spec.map (CommRingCat.ofHom C) := by
  simp only [SpecIso, Iso.trans_inv, Functor.mapIso_inv, Iso.op_inv,
    mapEquiv_symm, RingEquiv.toRingHom_eq_coe, Scheme.Spec_map, Quiver.Hom.unop_op, Category.assoc,
    isoOfIsAffine_inv_over, Scheme.isoSpec_Spec_inv, ← Spec.map_comp]
  /-
    n : Type v
    R : CommRingCat
    ⊢ Eq (AlgebraicGeometry.Spec.map (CategoryTheory.CategoryStruct.comp (Algebrai …
  -/
  congr 1
  /-
    case e_f
    n : Type v
    R : CommRingCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.ΓSpecIso R) …
  -/
  rw [Iso.inv_comp_eq]
  /-
    case e_f
    n : Type v
    R : CommRingCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom MvPolynomial.C) (M …
  -/
  ext : 2
  /-
    case e_f.hf.a
    n : Type v
    R : CommRingCat
    x✝ : ↑(CommRingCat.of ↑((AlgebraicGeometry.Spec R).presheaf.toPrefunctor.1 { u …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom MvPolynomial.C) ( …
  -/
  exact map_C _ _
  /-
    🎉 no goals
  -/


variable (n) in
/-- `𝔸(n; S)` is functorial wrt `S`. -/
def map {S T : Scheme.{max u v}} (f : S ⟶ T) : 𝔸(n; S) ⟶ 𝔸(n; T) :=
  homOfVector (𝔸(n; S) ↘ S ≫ f) (coord S)


@[reassoc (attr := simp)]
lemma map_over {S T : Scheme.{max u v}} (f : S ⟶ T) : map n f ≫ 𝔸(n; T) ↘ T = 𝔸(n; S) ↘ S ≫ f :=
  pullback.lift_fst _ _ _


@[simp]
lemma map_appTop_coord {S T : Scheme.{max u v}} (f : S ⟶ T) (i) :
    (map n f).appTop (coord T i) = coord S i :=
  homOfVector_appTop_coord _ _ _


@[simp]
lemma map_id : map n (𝟙 S) = 𝟙 𝔸(n; S) := by
  /-
    n : Type v
    S : AlgebraicGeometry.Scheme
    ⊢ Eq (AlgebraicGeometry.AffineSpace.map n (CategoryTheory.CategoryStruct.id S) …
  -/
           /-
             🎉 no goals
           -/
  ext1 <;> simp
           /-
             🎉 no goals
           -/


@[reassoc, simp]
lemma map_comp {S S' S'' : Scheme} (f : S ⟶ S') (g : S' ⟶ S'') :
    map n (f ≫ g) = map n f ≫ map n g := by
  /-
    n : Type v
    S S' S'' : AlgebraicGeometry.Scheme
    f : Quiver.Hom S S'
    g : Quiver.Hom S' S''
    ⊢ Eq (AlgebraicGeometry.AffineSpace.map n (CategoryTheory.CategoryStruct.comp  …
  -/
  ext1
    /-
      case h₁
      n : Type v
      S S' S'' : AlgebraicGeometry.Scheme
      f : Quiver.Hom S S'
      g : Quiver.Hom S' S''
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.AffineSpace.map n  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h₂
      n : Type v
      S S' S'' : AlgebraicGeometry.Scheme
      f : Quiver.Hom S S'
      g : Quiver.Hom S' S''
      i✝ : n
      ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.appTop (AlgebraicGeometry.AffineSpace.map  …
    -/
  · simp
    /-
      🎉 no goals
    -/


lemma map_Spec_map {R S : CommRingCat.{max u v}} (φ : R ⟶ S) :
    map n (Spec.map φ) =
      (SpecIso n S).hom ≫ Spec.map (CommRingCat.ofHom (MvPolynomial.map φ.hom)) ≫
        (SpecIso n R).inv := by
  /-
    n : Type v
    R S : CommRingCat
    φ : Quiver.Hom R S
    ⊢ Eq (AlgebraicGeometry.AffineSpace.map n (AlgebraicGeometry.Spec.map φ)) (Cat …
  -/
  rw [← Iso.inv_comp_eq]
  /-
    n : Type v
    R S : CommRingCat
    φ : Quiver.Hom R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.AffineSpace.SpecIs …
  -/
  ext1
  · simp only [map_over, Category.assoc, SpecIso_inv_over, SpecIso_inv_over_assoc,
      ← Spec.map_comp, ← CommRingCat.ofHom_comp]
    /-
      case h₁
      n : Type v
      R S : CommRingCat
      φ : Quiver.Hom R S
      ⊢ Eq (AlgebraicGeometry.Spec.map (CategoryTheory.CategoryStruct.comp φ (CommRi …
    -/
    rw [map_comp_C, CommRingCat.ofHom_comp, CommRingCat.ofHom_hom]
    /-
      🎉 no goals
    -/
  · simp only [Scheme.comp_coeBase, TopologicalSpace.Opens.map_comp_obj,
      TopologicalSpace.Opens.map_top, Scheme.comp_app, CommRingCat.comp_apply]
    /-
      case h₂
      n : Type v
      R S : CommRingCat
      φ : Quiver.Hom R S
      i✝ : n
      ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.AffineSpace.SpecIso …
    -/
    conv_lhs => enter[2]; tactic => exact map_appTop_coord _ _
    /-
      case h₂
      n : Type v
      R S : CommRingCat
      φ : Quiver.Hom R S
      i✝ : n
      ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.AffineSpace.SpecIso …
    -/
    conv_rhs => enter[2]; tactic => exact SpecIso_inv_appTop_coord _ _
    rw [SpecIso_inv_appTop_coord, ← CommRingCat.comp_apply, ← Scheme.ΓSpecIso_inv_naturality,
        CommRingCat.comp_apply, map_X]


/-- The map between affine spaces over affine bases is
isomorphic to the natural map between polynomial rings.  -/
def mapSpecMap {R S : CommRingCat.{max u v}} (φ : R ⟶ S) :
    Arrow.mk (map n (Spec.map φ)) ≅
      Arrow.mk (Spec.map (CommRingCat.ofHom (MvPolynomial.map (σ := n) φ.hom))) :=
                                              /-
                                                n : Type v
                                                S✝ X : AlgebraicGeometry.Scheme
                                                inst✝ : X.Over S✝
                                                R S : CommRingCat
                                                φ : Quiver.Hom R S
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.AffineSpace.SpecIs …
                                              -/
  Arrow.isoMk (SpecIso n S) (SpecIso n R) (by simp [map_Spec_map])
                                              /-
                                                🎉 no goals
                                              -/


/-- `𝔸(n; S)` is functorial wrt `n`. -/
def reindex {n m : Type v} (i : m → n) (S : Scheme.{max u v}) : 𝔸(n; S) ⟶ 𝔸(m; S) :=
  homOfVector (𝔸(n; S) ↘ S) (coord S ∘ i)


@[simp, reassoc]
lemma reindex_over {n m : Type v} (i : m → n) (S : Scheme.{max u v}) :
    reindex i S ≫ 𝔸(m; S) ↘ S = 𝔸(n; S) ↘ S :=
  pullback.lift_fst _ _ _


@[simp]
lemma reindex_appTop_coord {n m : Type v} (i : m → n) (S : Scheme.{max u v}) (j : m) :
    (reindex i S).appTop (coord S j) = coord S (i j) :=
  homOfVector_appTop_coord _ _ _


@[simp]
lemma reindex_id : reindex id S = 𝟙 𝔸(n; S) := by
  /-
    n : Type v
    S : AlgebraicGeometry.Scheme
    ⊢ Eq (AlgebraicGeometry.AffineSpace.reindex id S) (CategoryTheory.CategoryStru …
  -/
           /-
             🎉 no goals
           -/
  ext1 <;> simp
           /-
             🎉 no goals
           -/


@[simp, reassoc]
lemma reindex_comp {n₁ n₂ n₃ : Type v} (i : n₁ → n₂) (j : n₂ → n₃) (S : Scheme.{max u v}) :
    reindex (j ∘ i) S = reindex j S ≫ reindex i S := by
  /-
    n₁ n₂ n₃ : Type v
    i : n₁ → n₂
    j : n₂ → n₃
    S : AlgebraicGeometry.Scheme
    ⊢ Eq (AlgebraicGeometry.AffineSpace.reindex (Function.comp j i) S) (CategoryTh …
  -/
  have H₁ : reindex (j ∘ i) S ≫ 𝔸(n₁; S) ↘ S = (reindex j S ≫ reindex i S) ≫ 𝔸(n₁; S) ↘ S := by simp
  have H₂ (k) : (reindex (j ∘ i) S).appTop (coord S k) =
      (reindex j S).appTop ((reindex i S).appTop (coord S k)) := by
    rw [reindex_appTop_coord, reindex_appTop_coord, reindex_appTop_coord]
    rfl
  /-
    n₁ n₂ n₃ : Type v
    i : n₁ → n₂
    j : n₂ → n₃
    S : AlgebraicGeometry.Scheme
    H₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.AffineSpace.rei …
    H₂ : ∀ (k : n₁), Eq ((AlgebraicGeometry.Scheme.Hom.appTop (AlgebraicGeometry.A …
    ⊢ Eq (AlgebraicGeometry.AffineSpace.reindex (Function.comp j i) S) (CategoryTh …
  -/
  exact hom_ext H₁ H₂
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma map_reindex {n₁ n₂ : Type v} (i : n₁ → n₂) {S T : Scheme.{max u v}} (f : S ⟶ T) :
    map n₂ f ≫ reindex i T = reindex i S ≫ map n₁ f := by
  /-
    n₁ n₂ : Type v
    i : n₁ → n₂
    S T : AlgebraicGeometry.Scheme
    f : Quiver.Hom S T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.AffineSpace.map n₂ …
  -/
                    /-
                      🎉 no goals
                    -/
  apply hom_ext <;> simp
                    /-
                      🎉 no goals
                    -/


/-- The affine space as a functor. -/
@[simps]
def functor : (Type v)ᵒᵖ ⥤ Scheme.{max u v} ⥤ Scheme.{max u v} where
  obj n := { obj := AffineSpace n.unop, map := map n.unop, map_id := map_id, map_comp := map_comp }
  map {n m} i := { app := reindex i.unop, naturality := fun _ _ ↦ map_reindex i.unop }
                 /-
                   n✝ : Type v
                   S X : AlgebraicGeometry.Scheme
                   inst✝ : X.Over S
                   n : Opposite (Type v)
                   ⊢ Eq ({ obj := fun n => { obj := AlgebraicGeometry.AffineSpace (Opposite.unop  …
                 -/
  map_id n := by ext: 2; exact reindex_id _
                         /-
                           🎉 no goals
                         -/
                     /-
                       n : Type v
                       S X : AlgebraicGeometry.Scheme
                       inst✝ : X.Over S
                       X✝ Y✝ Z✝ : Opposite (Type v)
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun n => { obj := AlgebraicGeometry.AffineSpace (Opposite.unop  …
                     -/
  map_comp f g := by ext: 2; dsimp; exact reindex_comp _ _ _
                                    /-
                                      🎉 no goals
                                    -/


