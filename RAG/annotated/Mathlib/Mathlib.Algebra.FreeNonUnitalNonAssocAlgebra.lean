/-- The free non-unital, non-associative algebra on the type `X` with coefficients in `R`. -/
abbrev FreeNonUnitalNonAssocAlgebra :=
  MonoidAlgebra R (FreeMagma X)


/-- The embedding of `X` into the free algebra with coefficients in `R`. -/
def of : X → FreeNonUnitalNonAssocAlgebra R X :=
  MonoidAlgebra.ofMagma R _ ∘ FreeMagma.of


/-- The functor `X ↦ FreeNonUnitalNonAssocAlgebra R X` from the category of types to the
category of non-unital, non-associative algebras over `R` is adjoint to the forgetful functor in the
other direction. -/
def lift : (X → A) ≃ (FreeNonUnitalNonAssocAlgebra R X →ₙₐ[R] A) :=
  FreeMagma.lift.trans (MonoidAlgebra.liftMagma R)


@[simp]
theorem lift_symm_apply (F : FreeNonUnitalNonAssocAlgebra R X →ₙₐ[R] A) :
    (lift R).symm F = F ∘ of R := rfl


@[simp]
theorem of_comp_lift (f : X → A) : lift R f ∘ of R = f :=
  (lift R).left_inv f


@[simp]
theorem lift_unique (f : X → A) (F : FreeNonUnitalNonAssocAlgebra R X →ₙₐ[R] A) :
    F ∘ of R = f ↔ F = lift R f :=
  (lift R).symm_apply_eq


@[simp]
theorem lift_of_apply (f : X → A) (x) : lift R f (of R x) = f x :=
  congr_fun (of_comp_lift _ f) x


@[simp]
theorem lift_comp_of (F : FreeNonUnitalNonAssocAlgebra R X →ₙₐ[R] A) : lift R (F ∘ of R) = F :=
  (lift R).apply_symm_apply F


@[ext]
theorem hom_ext {F₁ F₂ : FreeNonUnitalNonAssocAlgebra R X →ₙₐ[R] A}
    (h : ∀ x, F₁ (of R x) = F₂ (of R x)) : F₁ = F₂ :=
  (lift R).symm.injective <| funext h


