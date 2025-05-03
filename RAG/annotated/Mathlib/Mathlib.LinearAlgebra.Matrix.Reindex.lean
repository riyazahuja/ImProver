/-- The natural map that reindexes a matrix's rows and columns with equivalent types,
`Matrix.reindex`, is a linear equivalence. -/
def reindexLinearEquiv (eₘ : m ≃ m') (eₙ : n ≃ n') : Matrix m n A ≃ₗ[R] Matrix m' n' A :=
  { reindex eₘ eₙ with
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }


@[simp]
theorem reindexLinearEquiv_apply (eₘ : m ≃ m') (eₙ : n ≃ n') (M : Matrix m n A) :
    reindexLinearEquiv R A eₘ eₙ M = reindex eₘ eₙ M :=
  rfl


@[simp]
theorem reindexLinearEquiv_symm (eₘ : m ≃ m') (eₙ : n ≃ n') :
    (reindexLinearEquiv R A eₘ eₙ).symm = reindexLinearEquiv R A eₘ.symm eₙ.symm :=
  rfl


@[simp]
theorem reindexLinearEquiv_refl_refl :
    reindexLinearEquiv R A (Equiv.refl m) (Equiv.refl n) = LinearEquiv.refl R _ :=
  LinearEquiv.ext fun _ => rfl


theorem reindexLinearEquiv_trans (e₁ : m ≃ m') (e₂ : n ≃ n') (e₁' : m' ≃ m'') (e₂' : n' ≃ n'') :
    (reindexLinearEquiv R A e₁ e₂).trans (reindexLinearEquiv R A e₁' e₂') =
      (reindexLinearEquiv R A (e₁.trans e₁') (e₂.trans e₂') : _ ≃ₗ[R] _) := by
  /-
    m : Type u_2
    n : Type u_3
    m' : Type u_6
    n' : Type u_7
    m'' : Type u_9
    n'' : Type u_10
    R : Type u_11
    A : Type u_12
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid A
    inst✝ : Module R A
    e₁ : Equiv m m'
    e₂ : Equiv n n'
    e₁' : Equiv m' m''
    e₂' : Equiv n' n''
    ⊢ Eq ((Matrix.reindexLinearEquiv R A e₁ e₂).trans (Matrix.reindexLinearEquiv R …
  -/
  ext
  /-
    case h.a
    m : Type u_2
    n : Type u_3
    m' : Type u_6
    n' : Type u_7
    m'' : Type u_9
    n'' : Type u_10
    R : Type u_11
    A : Type u_12
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid A
    inst✝ : Module R A
    e₁ : Equiv m m'
    e₂ : Equiv n n'
    e₁' : Equiv m' m''
    e₂' : Equiv n' n''
    x✝ : Matrix m n A
    i✝ : m''
    j✝ : n''
    ⊢ Eq (((Matrix.reindexLinearEquiv R A e₁ e₂).trans (Matrix.reindexLinearEquiv  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem reindexLinearEquiv_comp (e₁ : m ≃ m') (e₂ : n ≃ n') (e₁' : m' ≃ m'') (e₂' : n' ≃ n'') :
    reindexLinearEquiv R A e₁' e₂' ∘ reindexLinearEquiv R A e₁ e₂ =
      reindexLinearEquiv R A (e₁.trans e₁') (e₂.trans e₂') := by
  /-
    m : Type u_2
    n : Type u_3
    m' : Type u_6
    n' : Type u_7
    m'' : Type u_9
    n'' : Type u_10
    R : Type u_11
    A : Type u_12
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid A
    inst✝ : Module R A
    e₁ : Equiv m m'
    e₂ : Equiv n n'
    e₁' : Equiv m' m''
    e₂' : Equiv n' n''
    ⊢ Eq (Function.comp ⇑(Matrix.reindexLinearEquiv R A e₁' e₂') ⇑(Matrix.reindexL …
  -/
  rw [← reindexLinearEquiv_trans]
  /-
    m : Type u_2
    n : Type u_3
    m' : Type u_6
    n' : Type u_7
    m'' : Type u_9
    n'' : Type u_10
    R : Type u_11
    A : Type u_12
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid A
    inst✝ : Module R A
    e₁ : Equiv m m'
    e₂ : Equiv n n'
    e₁' : Equiv m' m''
    e₂' : Equiv n' n''
    ⊢ Eq (Function.comp ⇑(Matrix.reindexLinearEquiv R A e₁' e₂') ⇑(Matrix.reindexL …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem reindexLinearEquiv_comp_apply (e₁ : m ≃ m') (e₂ : n ≃ n') (e₁' : m' ≃ m'') (e₂' : n' ≃ n'')
    (M : Matrix m n A) :
    (reindexLinearEquiv R A e₁' e₂') (reindexLinearEquiv R A e₁ e₂ M) =
      reindexLinearEquiv R A (e₁.trans e₁') (e₂.trans e₂') M :=
  submatrix_submatrix _ _ _ _ _


theorem reindexLinearEquiv_one [DecidableEq m] [DecidableEq m'] [One A] (e : m ≃ m') :
    reindexLinearEquiv R A e e (1 : Matrix m m A) = 1 :=
  submatrix_one_equiv e.symm


theorem reindexLinearEquiv_mul [Fintype n] [Fintype n'] (eₘ : m ≃ m') (eₙ : n ≃ n') (eₒ : o ≃ o')
    (M : Matrix m n A) (N : Matrix n o A) :
    reindexLinearEquiv R A eₘ eₙ M * reindexLinearEquiv R A eₙ eₒ N =
      reindexLinearEquiv R A eₘ eₒ (M * N) :=
  submatrix_mul_equiv M N _ _ _


theorem mul_reindexLinearEquiv_one [Fintype n] [DecidableEq o] (e₁ : o ≃ n) (e₂ : o ≃ n')
    (M : Matrix m n A) :
    M * (reindexLinearEquiv R A e₁ e₂ 1) =
      reindexLinearEquiv R A (Equiv.refl m) (e₁.symm.trans e₂) M :=
  haveI := Fintype.ofEquiv _ e₁.symm
  mul_submatrix_one _ _ _


/-- For square matrices with coefficients in an algebra over a commutative semiring, the natural
    map that reindexes a matrix's rows and columns with equivalent types,
    `Matrix.reindex`, is an equivalence of algebras. -/
def reindexAlgEquiv (e : m ≃ n) : Matrix m m A ≃ₐ[R] Matrix n n A :=
  { reindexLinearEquiv A A e e with
    toFun := reindex e e
    map_mul' := fun a b => (reindexLinearEquiv_mul A A e e e a b).symm
    -- Porting note: `submatrix_smul` needed help
                             /-
                               l : Type u_1
                               m : Type u_2
                               n : Type u_3
                               o : Type u_4
                               l' : Type u_5
                               m' : Type u_6
                               n' : Type u_7
                               o' : Type u_8
                               m'' : Type u_9
                               n'' : Type u_10
                               R : Type u_11
                               A : Type u_12
                               inst✝⁶ : CommSemiring R
                               inst✝⁵ : Fintype n
                               inst✝⁴ : Fintype m
                               inst✝³ : DecidableEq m
                               inst✝² : DecidableEq n
                               inst✝¹ : Semiring A
                               inst✝ : Algebra R A
                               e : Equiv m n
                               r : R
                               ⊢ Eq ({ toFun := ⇑(Matrix.reindex e e), invFun := __src✝.invFun, left_inv := ⋯ …
                             -/
    commutes' := fun r => by simp [algebraMap, Algebra.toRingHom, submatrix_smul _ 1] }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem reindexAlgEquiv_apply (e : m ≃ n) (M : Matrix m m A) :
    reindexAlgEquiv R A e M = reindex e e M :=
  rfl


@[simp]
theorem reindexAlgEquiv_symm (e : m ≃ n) : (reindexAlgEquiv R A e).symm =
    reindexAlgEquiv R A e.symm :=
  rfl


@[simp]
theorem reindexAlgEquiv_refl : reindexAlgEquiv R A (Equiv.refl m) = AlgEquiv.refl :=
  AlgEquiv.ext fun _ => rfl


theorem reindexAlgEquiv_mul (e : m ≃ n) (M : Matrix m m A) (N : Matrix m m A) :
    reindexAlgEquiv R A e (M * N) = reindexAlgEquiv R A e M * reindexAlgEquiv R A e N :=
  _root_.map_mul ..


/-- Reindexing both indices along the same equivalence preserves the determinant.

For the `simp` version of this lemma, see `det_submatrix_equiv_self`.
-/
theorem det_reindexLinearEquiv_self [CommRing R] [Fintype m] [DecidableEq m] [Fintype n]
    [DecidableEq n] (e : m ≃ n) (M : Matrix m m R) : det (reindexLinearEquiv R R e e M) = det M :=
  det_reindex_self e M


/-- Reindexing both indices along the same equivalence preserves the determinant.

For the `simp` version of this lemma, see `det_submatrix_equiv_self`.
-/
theorem det_reindexAlgEquiv (B : Type*) [CommRing R] [CommRing B] [Algebra R B] [Fintype m]
    [DecidableEq m] [Fintype n] [DecidableEq n] (e : m ≃ n) (A : Matrix m m B) :
    det (reindexAlgEquiv R B e A) = det A :=
  det_reindex_self e A


