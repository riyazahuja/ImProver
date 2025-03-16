/-- The determinant of a continuous linear map, mainly as a convenience device to be able to
write `A.det` instead of `(A : M →ₗ[R] M).det`. -/
noncomputable abbrev det {R : Type*} [CommRing R] {M : Type*} [TopologicalSpace M] [AddCommGroup M]
    [Module R M] (A : M →L[R] M) : R :=
  LinearMap.det (A : M →ₗ[R] M)


@[simp]
theorem det_coe_symm {R : Type*} [Field R] {M : Type*} [TopologicalSpace M] [AddCommGroup M]
    [Module R M] (A : M ≃L[R] M) : (A.symm : M →L[R] M).det = (A : M →L[R] M).det⁻¹ :=
  LinearEquiv.det_coe_symm A.toLinearEquiv


