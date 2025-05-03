/-- The standard basis of `Matrix m n M` given a basis on `M`. -/
protected noncomputable def matrix (b : Basis ι R M) :
    Basis (m × n × ι) R (Matrix m n M) :=
  Basis.reindex (Pi.basis fun _ : m => Pi.basis fun _ : n => b)
    ((Equiv.sigmaEquivProd _ _).trans <| .prodCongr (.refl _) (Equiv.sigmaEquivProd _ _))
    |>.map (Matrix.ofLinearEquiv R)


@[simp]
theorem matrix_apply (b : Basis ι R M) (i : m) (j : n) (k : ι) [DecidableEq m] [DecidableEq n] :
    b.matrix m n (i, j, k) = Matrix.stdBasisMatrix i j (b k) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    m : Type u_4
    n : Type u_5
    inst✝⁶ : Fintype m
    inst✝⁵ : Fintype n
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    b : Basis ι R M
    i : m
    j : n
    k : ι
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    ⊢ Eq ((Basis.matrix m n b) { fst := i, snd := { fst := j, snd := k } }) (Matri …
  -/
  simp [Basis.matrix, Matrix.stdBasisMatrix_eq_of_single_single]
  /-
    🎉 no goals
  -/


/-- The standard basis of `Matrix m n R`. -/
noncomputable def stdBasis : Basis (m × n) R (Matrix m n R) :=
  Basis.reindex (Pi.basis fun _ : m => Pi.basisFun R n) (Equiv.sigmaEquivProd _ _)
    |>.map (ofLinearEquiv R)


theorem stdBasis_eq_stdBasisMatrix (i : m) (j : n) [DecidableEq m] [DecidableEq n] :
    stdBasis R m n (i, j) = stdBasisMatrix i j (1 : R) := by
  /-
    R : Type u_1
    m : Type u_2
    n : Type u_3
    inst✝⁴ : Fintype m
    inst✝³ : Finite n
    inst✝² : Semiring R
    i : m
    j : n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    ⊢ Eq ((Matrix.stdBasis R m n) { fst := i, snd := j }) (Matrix.stdBasisMatrix i …
  -/
  simp [stdBasis, stdBasisMatrix_eq_of_single_single]
  /-
    🎉 no goals
  -/


