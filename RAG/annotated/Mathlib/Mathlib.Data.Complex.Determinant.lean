/-- The determinant of `conjAe`, as a linear map. -/
@[simp]
theorem det_conjAe : LinearMap.det conjAe.toLinearMap = -1 := by
  /-
    ⊢ Eq (LinearMap.det Complex.conjAe.toLinearMap) (-1)
  -/
  rw [← LinearMap.det_toMatrix basisOneI, toMatrix_conjAe, Matrix.det_fin_two_of]
  /-
    ⊢ Eq (HSub.hSub (HMul.hMul 1 (-1)) (HMul.hMul 0 0)) (-1)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The determinant of `conjAe`, as a linear equiv. -/
@[simp]
theorem linearEquiv_det_conjAe : LinearEquiv.det conjAe.toLinearEquiv = -1 := by
  rw [← Units.eq_iff, LinearEquiv.coe_det, AlgEquiv.toLinearEquiv_toLinearMap, det_conjAe,
    Units.coe_neg_one]


