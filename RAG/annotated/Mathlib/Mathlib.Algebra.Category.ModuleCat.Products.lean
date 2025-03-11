/-- The product cone induced by the concrete product. -/
def productCone : Fan Z :=
  Fan.mk (ModuleCat.of R (∀ i : ι, Z i)) fun i =>
    ofHom (LinearMap.proj i : (∀ i : ι, Z i) →ₗ[R] Z i)


/-- The concrete product cone is limiting. -/
def productConeIsLimit : IsLimit (productCone Z) where
  lift s := ofHom (LinearMap.pi fun j => (s.π.app ⟨j⟩).hom : s.pt →ₗ[R] ∀ i : ι, Z i)
  uniq s m w := by
    /-
      R : Type u
      inst✝ : Ring R
      ι : Type v
      Z : ι → ModuleCatMax R
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor Z)
      m : Quiver.Hom s.pt (ModuleCat.productCone Z).pt
      w : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      ⊢ Eq m ((fun s => ModuleCat.ofHom (LinearMap.pi fun j => (s.π.app { as := j }) …
    -/
    ext x
    /-
      case hf.h
      R : Type u
      inst✝ : Ring R
      ι : Type v
      Z : ι → ModuleCatMax R
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor Z)
      m : Quiver.Hom s.pt (ModuleCat.productCone Z).pt
      w : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      x : ↑s.pt
      ⊢ Eq (m.hom x) (((fun s => ModuleCat.ofHom (LinearMap.pi fun j => (s.π.app { a …
    -/
    funext i
    /-
      case hf.h.h
      R : Type u
      inst✝ : Ring R
      ι : Type v
      Z : ι → ModuleCatMax R
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor Z)
      m : Quiver.Hom s.pt (ModuleCat.productCone Z).pt
      w : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      x : ↑s.pt
      i : ι
      ⊢ Eq (m.hom x i) (((fun s => ModuleCat.ofHom (LinearMap.pi fun j => (s.π.app { …
    -/
    exact DFunLike.congr_fun (congr_arg Hom.hom (w ⟨i⟩)) x
    /-
      🎉 no goals
    -/

-- While we could use this to construct a `HasProducts (ModuleCat R)` instance,
-- we already have `HasLimits (ModuleCat R)` in `Algebra.Category.ModuleCat.Limits`.

/-- The categorical product of a family of objects in `ModuleCat`
agrees with the usual module-theoretical product.
-/
noncomputable def piIsoPi : ∏ᶜ Z ≅ ModuleCat.of R (∀ i, Z i) :=
  limit.isoLimitCone ⟨_, productConeIsLimit Z⟩

-- We now show this isomorphism commutes with the inclusion of the kernel into the source.

@[simp, elementwise]
theorem piIsoPi_inv_kernel_ι (i : ι) :
    (piIsoPi Z).inv ≫ Pi.π Z i = ofHom (LinearMap.proj i : (∀ i : ι, Z i) →ₗ[R] Z i) :=
  limit.isoLimitCone_inv_π _ _


@[simp, elementwise]
theorem piIsoPi_hom_ker_subtype (i : ι) :
    (piIsoPi Z).hom ≫ ofHom (LinearMap.proj i : (∀ i : ι, Z i) →ₗ[R] Z i) = Pi.π Z i :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ (limit.isLimit _) (Discrete.mk i)


/-- The coproduct cone induced by the concrete product. -/
def coproductCocone : Cofan Z :=
  Cofan.mk (ModuleCat.of R (⨁ i : ι, Z i)) fun i => ofHom (DirectSum.lof R ι (fun i ↦ Z i) i)


/-- The concrete coproduct cone is limiting. -/
def coproductCoconeIsColimit : IsColimit (coproductCocone Z) where
  desc s := ofHom <| DirectSum.toModule R ι _ fun i ↦ (s.ι.app ⟨i⟩).hom
  fac := by
    /-
      R : Type u
      inst✝¹ : Ring R
      ι : Type v
      Z : ι → ModuleCatMax R
      inst✝ : DecidableEq ι
      ⊢ ∀ (s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor Z)) (j  …
    -/
    rintro s ⟨i⟩
    /-
      case mk
      R : Type u
      inst✝¹ : Ring R
      ι : Type v
      Z : ι → ModuleCatMax R
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor Z)
      i : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.coproductCocone Z).ι.app  …
    -/
    ext (x : Z i)
    simpa only [Discrete.functor_obj_eq_as, coproductCocone, Cofan.mk_pt, Functor.const_obj_obj,
      Cofan.mk_ι_app, hom_comp, LinearMap.coe_comp, Function.comp_apply] using
      DirectSum.toModule_lof (ι := ι) R (M := fun i ↦ Z i) i x
  uniq := by
    /-
      R : Type u
      inst✝¹ : Ring R
      ι : Type v
      Z : ι → ModuleCatMax R
      inst✝ : DecidableEq ι
      ⊢ ∀ (s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor Z)) (m  …
    -/
    rintro s f h
    /-
      R : Type u
      inst✝¹ : Ring R
      ι : Type v
      Z : ι → ModuleCatMax R
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor Z)
      f : Quiver.Hom (ModuleCat.coproductCocone Z).pt s.pt
      h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      ⊢ Eq f ((fun s => ModuleCat.ofHom (DirectSum.toModule R ι ↑s.1 fun i => (s.ι.a …
    -/
    ext : 1
    /-
      case hf
      R : Type u
      inst✝¹ : Ring R
      ι : Type v
      Z : ι → ModuleCatMax R
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor Z)
      f : Quiver.Hom (ModuleCat.coproductCocone Z).pt s.pt
      h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      ⊢ Eq f.hom ((fun s => ModuleCat.ofHom (DirectSum.toModule R ι ↑s.1 fun i => (s …
    -/
    refine DirectSum.linearMap_ext _ fun i ↦ ?_
    /-
      case hf
      R : Type u
      inst✝¹ : Ring R
      ι : Type v
      Z : ι → ModuleCatMax R
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor Z)
      f : Quiver.Hom (ModuleCat.coproductCocone Z).pt s.pt
      h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      i : ι
      ⊢ Eq (f.hom.comp (DirectSum.lof R ι (fun i => ↑(Z i)) i)) (((fun s => ModuleCa …
    -/
    ext x
    simpa only [LinearMap.coe_comp, Function.comp_apply, toModule_lof] using
      congr($(h ⟨i⟩) x)


/-- The categorical coproduct of a family of objects in `ModuleCat`
agrees with direct sum.
-/
noncomputable def coprodIsoDirectSum : ∐ Z ≅ ModuleCat.of R (⨁ i, Z i) :=
  colimit.isoColimitCocone ⟨_, coproductCoconeIsColimit Z⟩


@[simp, elementwise]
theorem ι_coprodIsoDirectSum_hom (i : ι) :
    Sigma.ι Z i ≫ (coprodIsoDirectSum Z).hom = ofHom (DirectSum.lof R ι (fun i ↦ Z i) i) :=
  colimit.isoColimitCocone_ι_hom _ _


@[simp, elementwise]
theorem lof_coprodIsoDirectSum_inv (i : ι) :
    ofHom (DirectSum.lof R ι (fun i ↦ Z i) i) ≫ (coprodIsoDirectSum Z).inv = Sigma.ι Z i :=
  (coproductCoconeIsColimit Z).comp_coconePointUniqueUpToIso_hom (colimit.isColimit _) _


