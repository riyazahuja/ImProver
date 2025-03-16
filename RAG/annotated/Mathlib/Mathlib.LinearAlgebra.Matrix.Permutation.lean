variable (R) in
/-- the permutation matrix associated with an `Equiv.Perm` -/
abbrev Equiv.Perm.permMatrix [Zero R] [One R] : Matrix n n R :=
  σ.toPEquiv.toMatrix


/-- The determinant of a permutation matrix equals its sign. -/
@[simp]
theorem det_permutation [CommRing R] : det (σ.permMatrix R) = Perm.sign σ := by
  rw [← Matrix.mul_one (σ.permMatrix R), PEquiv.toPEquiv_mul_matrix,
    det_permute, det_one, mul_one]


/-- The trace of a permutation matrix equals the number of fixed points. -/
theorem trace_permutation [AddCommMonoidWithOne R] :
    trace (σ.permMatrix R) = (Function.fixedPoints σ).ncard := by
  /-
    n : Type u_1
    R : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    σ : Equiv.Perm n
    inst✝ : AddCommMonoidWithOne R
    ⊢ Eq (Equiv.Perm.permMatrix R σ).trace ↑(Function.fixedPoints ⇑σ).ncard
  -/
  delta trace
  /-
    n : Type u_1
    R : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    σ : Equiv.Perm n
    inst✝ : AddCommMonoidWithOne R
    ⊢ Eq (Finset.univ.sum fun i => (Equiv.Perm.permMatrix R σ).diag i) ↑(Function. …
  -/
  simp [toPEquiv_apply, ← Set.ncard_coe_Finset, Function.fixedPoints, Function.IsFixedPt]
  /-
    🎉 no goals
  -/


