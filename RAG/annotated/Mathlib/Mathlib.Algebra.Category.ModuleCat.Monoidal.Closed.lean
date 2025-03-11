/-- Auxiliary definition for the `MonoidalClosed` instance on `Module R`.
(This is only a separate definition in order to speed up typechecking. )
-/
def monoidalClosedHomEquiv (M N P : ModuleCat.{u} R) :
    ((MonoidalCategory.tensorLeft M).obj N ⟶ P) ≃
      (N ⟶ ((linearCoyoneda R (ModuleCat R)).obj (op M)).obj P) where
  toFun f := ofHom₂ <| LinearMap.compr₂ (TensorProduct.mk R N M) ((β_ N M).hom ≫ f).hom
  invFun f := (β_ M N).hom ≫ ofHom (TensorProduct.lift f.hom₂)
  left_inv f := by
    /-
      R : Type u
      inst✝ : CommRing R
      M N P : ModuleCat R
      f : Quiver.Hom ((CategoryTheory.MonoidalCategory.tensorLeft M).obj N) P
      ⊢ Eq ((fun f => CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCate …
    -/
    ext : 1
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      M N P : ModuleCat R
      f : Quiver.Hom ((CategoryTheory.MonoidalCategory.tensorLeft M).obj N) P
      ⊢ Eq ((fun f => CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCate …
    -/
    apply TensorProduct.ext'
    /-
      case hf.H
      R : Type u
      inst✝ : CommRing R
      M N P : ModuleCat R
      f : Quiver.Hom ((CategoryTheory.MonoidalCategory.tensorLeft M).obj N) P
      ⊢ ∀ (x : ↑M) (y : ↑N), Eq (((fun f => CategoryTheory.CategoryStruct.comp (Cate …
    -/
    intro m n
    /-
      case hf.H
      R : Type u
      inst✝ : CommRing R
      M N P : ModuleCat R
      f : Quiver.Hom ((CategoryTheory.MonoidalCategory.tensorLeft M).obj N) P
      m : ↑M
      n : ↑N
      ⊢ Eq (((fun f => CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCat …
    -/
    simp only [Hom.hom₂_ofHom₂, LinearMap.comp_apply, hom_comp, MonoidalCategory.tensorLeft_obj]
    /-
      case hf.H
      R : Type u
      inst✝ : CommRing R
      M N P : ModuleCat R
      f : Quiver.Hom ((CategoryTheory.MonoidalCategory.tensorLeft M).obj N) P
      m : ↑M
      n : ↑N
      ⊢ Eq ((TensorProduct.lift ((TensorProduct.mk R ↑N ↑M).compr₂ (f.hom.comp (Cate …
    -/
    erw [MonoidalCategory.braiding_hom_apply, TensorProduct.lift.tmul]
    /-
      🎉 no goals
    -/
  right_inv _ := rfl


instance : MonoidalClosed (ModuleCat.{u} R) where
  closed M :=
    { rightAdj := (linearCoyoneda R (ModuleCat.{u} R)).obj (op M)
      adj := Adjunction.mkOfHomEquiv
            { homEquiv := fun N P => monoidalClosedHomEquiv M N P
              -- Porting note: this proof was automatic in mathlib3
              homEquiv_naturality_left_symm := by
                /-
                  R : Type u
                  inst✝ : CommRing R
                  M : ModuleCat R
                  ⊢ ∀ {X' X Y : ModuleCat R} (f : Quiver.Hom X' X) (g : Quiver.Hom X (((Category …
                -/
                intros
                /-
                  R : Type u
                  inst✝ : CommRing R
                  M X'✝ X✝ Y✝ : ModuleCat R
                  f✝ : Quiver.Hom X'✝ X✝
                  g✝ : Quiver.Hom X✝ (((CategoryTheory.linearCoyoneda R (ModuleCat R)).obj { uno …
                  ⊢ Eq (((fun N P => M.monoidalClosedHomEquiv N P) X'✝ Y✝).symm (CategoryTheory. …
                -/
                ext : 1
                /-
                  case hf
                  R : Type u
                  inst✝ : CommRing R
                  M X'✝ X✝ Y✝ : ModuleCat R
                  f✝ : Quiver.Hom X'✝ X✝
                  g✝ : Quiver.Hom X✝ (((CategoryTheory.linearCoyoneda R (ModuleCat R)).obj { uno …
                  ⊢ Eq (((fun N P => M.monoidalClosedHomEquiv N P) X'✝ Y✝).symm (CategoryTheory. …
                -/
                apply TensorProduct.ext'
                /-
                  case hf.H
                  R : Type u
                  inst✝ : CommRing R
                  M X'✝ X✝ Y✝ : ModuleCat R
                  f✝ : Quiver.Hom X'✝ X✝
                  g✝ : Quiver.Hom X✝ (((CategoryTheory.linearCoyoneda R (ModuleCat R)).obj { uno …
                  ⊢ ∀ (x : ↑M) (y : ↑X'✝), Eq ((((fun N P => M.monoidalClosedHomEquiv N P) X'✝ Y …
                -/
                intro m n
                /-
                  case hf.H
                  R : Type u
                  inst✝ : CommRing R
                  M X'✝ X✝ Y✝ : ModuleCat R
                  f✝ : Quiver.Hom X'✝ X✝
                  g✝ : Quiver.Hom X✝ (((CategoryTheory.linearCoyoneda R (ModuleCat R)).obj { uno …
                  m : ↑M
                  n : ↑X'✝
                  ⊢ Eq ((((fun N P => M.monoidalClosedHomEquiv N P) X'✝ Y✝).symm (CategoryTheory …
                -/
                rfl } }
                /-
                  🎉 no goals
                -/


theorem ihom_map_apply {M N P : ModuleCat.{u} R} (f : N ⟶ P) (g : ModuleCat.of R (M ⟶ N)) :
    (ihom M).map f g = g ≫ f :=
  rfl


theorem monoidalClosed_curry {M N P : ModuleCat.{u} R} (f : M ⊗ N ⟶ P) (x : M) (y : N) :
    ((MonoidalClosed.curry f).hom y).hom x = f (x ⊗ₜ[R] y) :=
  rfl


@[simp]
theorem monoidalClosed_uncurry
    {M N P : ModuleCat.{u} R} (f : N ⟶ M ⟶[ModuleCat.{u} R] P) (x : M) (y : N) :
    MonoidalClosed.uncurry f (x ⊗ₜ[R] y) = (f y).hom x :=
  rfl


/-- Describes the counit of the adjunction `M ⊗ - ⊣ Hom(M, -)`. Given an `R`-module `N` this
should give a map `M ⊗ Hom(M, N) ⟶ N`, so we flip the order of the arguments in the identity map
`Hom(M, N) ⟶ (M ⟶ N)` and uncurry the resulting map `M ⟶ Hom(M, N) ⟶ N.` -/
theorem ihom_ev_app (M N : ModuleCat.{u} R) :
    (ihom.ev M).app N = ModuleCat.ofHom (TensorProduct.uncurry R M ((ihom M).obj N) N
      (LinearMap.lcomp _ _ homLinearEquiv.toLinearMap ∘ₗ LinearMap.id.flip)) := by
  /-
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    ⊢ Eq ((CategoryTheory.ihom.ev M).app N) (ModuleCat.ofHom ((TensorProduct.uncur …
  -/
  rw [← MonoidalClosed.uncurry_id_eq_ev]
  /-
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    ⊢ Eq (CategoryTheory.MonoidalClosed.uncurry (CategoryTheory.CategoryStruct.id  …
  -/
  ext : 1
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    ⊢ Eq (CategoryTheory.MonoidalClosed.uncurry (CategoryTheory.CategoryStruct.id  …
  -/
  apply TensorProduct.ext'
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    ⊢ ∀ (x : ↑M) (y : ↑((CategoryTheory.ihom M).obj N)), Eq ((CategoryTheory.Monoi …
  -/
  apply monoidalClosed_uncurry
  /-
    🎉 no goals
  -/


/-- Describes the unit of the adjunction `M ⊗ - ⊣ Hom(M, -)`. Given an `R`-module `N` this should
define a map `N ⟶ Hom(M, M ⊗ N)`, which is given by flipping the arguments in the natural
`R`-bilinear map `M ⟶ N ⟶ M ⊗ N`. -/
theorem ihom_coev_app (M N : ModuleCat.{u} R) :
    (ihom.coev M).app N = ModuleCat.ofHom₂ (TensorProduct.mk _ _ _).flip :=
  rfl


theorem monoidalClosed_pre_app {M N : ModuleCat.{u} R} (P : ModuleCat.{u} R) (f : N ⟶ M) :
    (MonoidalClosed.pre f).app P = ofHom (homLinearEquiv.symm.toLinearMap ∘ₗ
      LinearMap.lcomp _ _ f.hom ∘ₗ homLinearEquiv.toLinearMap) :=
  rfl


