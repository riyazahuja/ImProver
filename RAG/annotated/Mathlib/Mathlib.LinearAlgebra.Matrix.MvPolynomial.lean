/-- The matrix with variable `X (i,j)` at location `(i,j)`. -/
noncomputable def mvPolynomialX [CommSemiring R] : Matrix m n (MvPolynomial (m × n) R) :=
  of fun i j => MvPolynomial.X (i, j)

-- TODO: set as an equation lemma for `mv_polynomial_X`, see https://github.com/leanprover-community/mathlib4/pull/3024

@[simp]
theorem mvPolynomialX_apply [CommSemiring R] (i j) :
    mvPolynomialX m n R i j = MvPolynomial.X (i, j) :=
  rfl


/-- Any matrix `A` can be expressed as the evaluation of `Matrix.mvPolynomialX`.

This is of particular use when `MvPolynomial (m × n) R` is an integral domain but `S` is
not, as if the `MvPolynomial.eval₂` can be pulled to the outside of a goal, it can be solved in
under cancellative assumptions. -/
theorem mvPolynomialX_map_eval₂ [CommSemiring R] [CommSemiring S] (f : R →+* S) (A : Matrix m n S) :
    (mvPolynomialX m n R).map (MvPolynomial.eval₂ f fun p : m × n => A p.1 p.2) = A :=
  ext fun i j => MvPolynomial.eval₂_X _ (fun p : m × n => A p.1 p.2) (i, j)


/-- A variant of `Matrix.mvPolynomialX_map_eval₂` with a bundled `RingHom` on the LHS. -/
theorem mvPolynomialX_mapMatrix_eval [Fintype m] [DecidableEq m] [CommSemiring R]
    (A : Matrix m m R) :
    (MvPolynomial.eval fun p : m × m => A p.1 p.2).mapMatrix (mvPolynomialX m m R) = A :=
  mvPolynomialX_map_eval₂ _ A


/-- A variant of `Matrix.mvPolynomialX_map_eval₂` with a bundled `AlgHom` on the LHS. -/
theorem mvPolynomialX_mapMatrix_aeval [Fintype m] [DecidableEq m] [CommSemiring R] [CommSemiring S]
    [Algebra R S] (A : Matrix m m S) :
    (MvPolynomial.aeval fun p : m × m => A p.1 p.2).mapMatrix (mvPolynomialX m m R) = A :=
  mvPolynomialX_map_eval₂ _ A


/-- In a nontrivial ring, `Matrix.mvPolynomialX m m R` has non-zero determinant. -/
theorem det_mvPolynomialX_ne_zero [DecidableEq m] [Fintype m] [CommRing R] [Nontrivial R] :
    det (mvPolynomialX m m R) ≠ 0 := by
  /-
    m : Type u_1
    R : Type u_3
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    ⊢ Ne (Matrix.mvPolynomialX m m R).det 0
  -/
  intro h_det
  /-
    m : Type u_1
    R : Type u_3
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    h_det : Eq (Matrix.mvPolynomialX m m R).det 0
    ⊢ False
  -/
  have := congr_arg Matrix.det (mvPolynomialX_mapMatrix_eval (1 : Matrix m m R))
  /-
    m : Type u_1
    R : Type u_3
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    h_det : Eq (Matrix.mvPolynomialX m m R).det 0
    this : Eq ((MvPolynomial.eval fun p => 1 p.1 p.2).mapMatrix (Matrix.mvPolynomi …
    ⊢ False
  -/
  rw [det_one, ← RingHom.map_det, h_det, RingHom.map_zero] at this
  /-
    m : Type u_1
    R : Type u_3
    inst✝³ : DecidableEq m
    inst✝² : Fintype m
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    h_det : Eq (Matrix.mvPolynomialX m m R).det 0
    this : Eq 0 1
    ⊢ False
  -/
  exact zero_ne_one this
  /-
    🎉 no goals
  -/


