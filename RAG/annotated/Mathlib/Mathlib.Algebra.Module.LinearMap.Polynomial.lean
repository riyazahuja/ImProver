/-- Let `M` be an `(m × n)`-matrix over `R`.
Then `Matrix.toMvPolynomial M` is the family (indexed by `i : m`)
of multivariate polynomials in `n` variables over `R` that evaluates on `c : n → R`
to the dot product of the `i`-th row of `M` with `c`:
`Matrix.toMvPolynomial M i` is the sum of the monomials `C (M i j) * X j`. -/
noncomputable
def toMvPolynomial (M : Matrix m n R) (i : m) : MvPolynomial n R :=
  ∑ j, monomial (.single j 1) (M i j)


lemma toMvPolynomial_eval_eq_apply (M : Matrix m n R) (i : m) (c : n → R) :
    eval c (M.toMvPolynomial i) = (M *ᵥ c) i := by
  simp only [toMvPolynomial, map_sum, eval_monomial, pow_zero, Finsupp.prod_single_index, pow_one,
    mulVec, dotProduct]


lemma toMvPolynomial_map (f : R →+* S) (M : Matrix m n R) (i : m) :
    (M.map f).toMvPolynomial i = MvPolynomial.map f (M.toMvPolynomial i) := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    S : Type u_5
    inst✝² : Fintype n
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    M : Matrix m n R
    i : m
    ⊢ Eq ((M.map ⇑f).toMvPolynomial i) ((MvPolynomial.map f) (M.toMvPolynomial i))
  -/
  simp only [toMvPolynomial, map_apply, map_sum, map_monomial]
  /-
    🎉 no goals
  -/


lemma toMvPolynomial_isHomogeneous (M : Matrix m n R) (i : m) :
    (M.toMvPolynomial i).IsHomogeneous 1 := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝¹ : Fintype n
    inst✝ : CommSemiring R
    M : Matrix m n R
    i : m
    ⊢ (M.toMvPolynomial i).IsHomogeneous 1
  -/
  apply MvPolynomial.IsHomogeneous.sum
  /-
    case h
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝¹ : Fintype n
    inst✝ : CommSemiring R
    M : Matrix m n R
    i : m
    ⊢ ∀ (i_1 : n), Membership.mem Finset.univ i_1 → ((MvPolynomial.monomial (Finsu …
  -/
  rintro j -
  /-
    case h
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝¹ : Fintype n
    inst✝ : CommSemiring R
    M : Matrix m n R
    i : m
    j : n
    ⊢ ((MvPolynomial.monomial (Finsupp.single j 1)) (M i j)).IsHomogeneous 1
  -/
  apply MvPolynomial.isHomogeneous_monomial _ _
  simp [Finsupp.degree, Finsupp.support_single_ne_zero _ one_ne_zero, Finset.sum_singleton,
    Finsupp.single_eq_same]


lemma toMvPolynomial_totalDegree_le (M : Matrix m n R) (i : m) :
    (M.toMvPolynomial i).totalDegree ≤ 1 := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝¹ : Fintype n
    inst✝ : CommSemiring R
    M : Matrix m n R
    i : m
    ⊢ LE.le (M.toMvPolynomial i).totalDegree 1
  -/
  apply (toMvPolynomial_isHomogeneous _ _).totalDegree_le
  /-
    🎉 no goals
  -/


@[simp]
lemma toMvPolynomial_constantCoeff (M : Matrix m n R) (i : m) :
    constantCoeff (M.toMvPolynomial i) = 0 := by
  simp only [toMvPolynomial, ← C_mul_X_eq_monomial, map_sum, _root_.map_mul, constantCoeff_X,
    mul_zero, Finset.sum_const_zero]


@[simp]
lemma toMvPolynomial_zero : (0 : Matrix m n R).toMvPolynomial = 0 := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝¹ : Fintype n
    inst✝ : CommSemiring R
    ⊢ Eq (Matrix.toMvPolynomial 0) 0
  -/
  ext; simp only [toMvPolynomial, zero_apply, map_zero, Finset.sum_const_zero, Pi.zero_apply]
       /-
         🎉 no goals
       -/


@[simp]
lemma toMvPolynomial_one [DecidableEq n] : (1 : Matrix n n R).toMvPolynomial = X := by
  /-
    n : Type u_2
    R : Type u_4
    inst✝² : Fintype n
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq n
    ⊢ Eq (Matrix.toMvPolynomial 1) MvPolynomial.X
  -/
  ext i : 1
  /-
    case h
    n : Type u_2
    R : Type u_4
    inst✝² : Fintype n
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq n
    i : n
    ⊢ Eq (Matrix.toMvPolynomial 1 i) (MvPolynomial.X i)
  -/
  rw [toMvPolynomial, Finset.sum_eq_single i]
    /-
      case h
      n : Type u_2
      R : Type u_4
      inst✝² : Fintype n
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq n
      i : n
      ⊢ Eq ((MvPolynomial.monomial (Finsupp.single i 1)) (1 i i)) (MvPolynomial.X i)
    -/
  · simp only [one_apply_eq, ← C_mul_X_eq_monomial, C_1, one_mul]
    /-
      🎉 no goals
    -/
    /-
      case h.h₀
      n : Type u_2
      R : Type u_4
      inst✝² : Fintype n
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq n
      i : n
      ⊢ ∀ (b : n), Membership.mem Finset.univ b → Ne b i → Eq ((MvPolynomial.monomia …
    -/
  · rintro j - hj
    /-
      case h.h₀
      n : Type u_2
      R : Type u_4
      inst✝² : Fintype n
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq n
      i j : n
      hj : Ne j i
      ⊢ Eq ((MvPolynomial.monomial (Finsupp.single j 1)) (1 i j)) 0
    -/
    simp only [one_apply_ne hj.symm, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.h₁
      n : Type u_2
      R : Type u_4
      inst✝² : Fintype n
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq n
      i : n
      ⊢ Not (Membership.mem Finset.univ i) → Eq ((MvPolynomial.monomial (Finsupp.sin …
    -/
  · intro h
    /-
      case h.h₁
      n : Type u_2
      R : Type u_4
      inst✝² : Fintype n
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq n
      i : n
      h : Not (Membership.mem Finset.univ i)
      ⊢ Eq ((MvPolynomial.monomial (Finsupp.single i 1)) (1 i i)) 0
    -/
    exact (h (Finset.mem_univ _)).elim
    /-
      🎉 no goals
    -/


lemma toMvPolynomial_add (M N : Matrix m n R) :
    (M + N).toMvPolynomial = M.toMvPolynomial + N.toMvPolynomial := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝¹ : Fintype n
    inst✝ : CommSemiring R
    M N : Matrix m n R
    ⊢ Eq (HAdd.hAdd M N).toMvPolynomial (HAdd.hAdd M.toMvPolynomial N.toMvPolynomi …
  -/
  ext i : 1
  /-
    case h
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝¹ : Fintype n
    inst✝ : CommSemiring R
    M N : Matrix m n R
    i : m
    ⊢ Eq ((HAdd.hAdd M N).toMvPolynomial i) (HAdd.hAdd M.toMvPolynomial N.toMvPoly …
  -/
  simp only [toMvPolynomial, add_apply, map_add, Finset.sum_add_distrib, Pi.add_apply]
  /-
    🎉 no goals
  -/


lemma toMvPolynomial_mul (M : Matrix m n R) (N : Matrix n o R) (i : m) :
    (M * N).toMvPolynomial i = bind₁ N.toMvPolynomial (M.toMvPolynomial i) := by
  simp only [toMvPolynomial, mul_apply, map_sum, Finset.sum_comm (γ := o), bind₁, aeval,
    AlgHom.coe_mk, coe_eval₂Hom, eval₂_monomial, algebraMap_apply, Algebra.id.map_eq_id,
    RingHom.id_apply, C_apply, pow_zero, Finsupp.prod_single_index, pow_one, Finset.mul_sum,
    monomial_mul, zero_add]


/-- Let `f : M₁ →ₗ[R] M₂` be an `R`-linear map
between modules `M₁` and `M₂` with bases `b₁` and `b₂` respectively.
Then `LinearMap.toMvPolynomial b₁ b₂ f` is the family of multivariate polynomials over `R`
that evaluates on an element `x` of `M₁` (represented on the basis `b₁`)
to the element `f x` of `M₂` (represented on the basis `b₂`). -/
noncomputable
def toMvPolynomial (f : M₁ →ₗ[R] M₂) (i : ι₂) :
    MvPolynomial ι₁ R :=
  (toMatrix b₁ b₂ f).toMvPolynomial i


lemma toMvPolynomial_eval_eq_apply (f : M₁ →ₗ[R] M₂) (i : ι₂) (c : ι₁ →₀ R) :
    eval c (f.toMvPolynomial b₁ b₂ i) = b₂.repr (f (b₁.repr.symm c)) i := by
  rw [toMvPolynomial, Matrix.toMvPolynomial_eval_eq_apply,
    ← LinearMap.toMatrix_mulVec_repr b₁ b₂, LinearEquiv.apply_symm_apply]


open Algebra.TensorProduct in
lemma toMvPolynomial_baseChange (f : M₁ →ₗ[R] M₂) (i : ι₂) (A : Type*) [CommRing A] [Algebra R A] :
    (f.baseChange A).toMvPolynomial (basis A b₁) (basis A b₂) i =
      MvPolynomial.map (algebraMap R A) (f.toMvPolynomial b₁ b₂ i) := by
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    ι₁ : Type u_4
    ι₂ : Type u_5
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₁
    inst✝⁵ : Module R M₂
    inst✝⁴ : Fintype ι₁
    inst✝³ : Finite ι₂
    inst✝² : DecidableEq ι₁
    b₁ : Basis ι₁ R M₁
    b₂ : Basis ι₂ R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    i : ι₂
    A : Type u_6
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq (LinearMap.toMvPolynomial (Algebra.TensorProduct.basis A b₁) (Algebra.Ten …
  -/
  simp only [toMvPolynomial, toMatrix_baseChange, Matrix.toMvPolynomial_map]
  /-
    🎉 no goals
  -/


lemma toMvPolynomial_isHomogeneous (f : M₁ →ₗ[R] M₂) (i : ι₂) :
    (f.toMvPolynomial b₁ b₂ i).IsHomogeneous 1 :=
  Matrix.toMvPolynomial_isHomogeneous _ _


lemma toMvPolynomial_totalDegree_le (f : M₁ →ₗ[R] M₂) (i : ι₂) :
    (f.toMvPolynomial b₁ b₂ i).totalDegree ≤ 1 :=
  Matrix.toMvPolynomial_totalDegree_le _ _


@[simp]
lemma toMvPolynomial_constantCoeff (f : M₁ →ₗ[R] M₂) (i : ι₂) :
    constantCoeff (f.toMvPolynomial b₁ b₂ i) = 0 :=
  Matrix.toMvPolynomial_constantCoeff _ _


@[simp]
lemma toMvPolynomial_zero : (0 : M₁ →ₗ[R] M₂).toMvPolynomial b₁ b₂ = 0 := by
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    ι₁ : Type u_4
    ι₂ : Type u_5
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M₁
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Fintype ι₁
    inst✝¹ : Finite ι₂
    inst✝ : DecidableEq ι₁
    b₁ : Basis ι₁ R M₁
    b₂ : Basis ι₂ R M₂
    ⊢ Eq (LinearMap.toMvPolynomial b₁ b₂ 0) 0
  -/
  unfold toMvPolynomial; simp only [map_zero, Matrix.toMvPolynomial_zero]
                         /-
                           🎉 no goals
                         -/


@[simp]
lemma toMvPolynomial_id : (id : M₁ →ₗ[R] M₁).toMvPolynomial b₁ b₁ = X := by
  /-
    R : Type u_1
    M₁ : Type u_2
    ι₁ : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M₁
    inst✝² : Module R M₁
    inst✝¹ : Fintype ι₁
    inst✝ : DecidableEq ι₁
    b₁ : Basis ι₁ R M₁
    ⊢ Eq (LinearMap.toMvPolynomial b₁ b₁ LinearMap.id) MvPolynomial.X
  -/
  unfold toMvPolynomial; simp only [toMatrix_id, Matrix.toMvPolynomial_one]
                         /-
                           🎉 no goals
                         -/


lemma toMvPolynomial_add (f g : M₁ →ₗ[R] M₂) :
    (f + g).toMvPolynomial b₁ b₂ = f.toMvPolynomial b₁ b₂ + g.toMvPolynomial b₁ b₂ := by
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    ι₁ : Type u_4
    ι₂ : Type u_5
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M₁
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Fintype ι₁
    inst✝¹ : Finite ι₂
    inst✝ : DecidableEq ι₁
    b₁ : Basis ι₁ R M₁
    b₂ : Basis ι₂ R M₂
    f g : LinearMap (RingHom.id R) M₁ M₂
    ⊢ Eq (LinearMap.toMvPolynomial b₁ b₂ (HAdd.hAdd f g)) (HAdd.hAdd (LinearMap.to …
  -/
  unfold toMvPolynomial; simp only [map_add, Matrix.toMvPolynomial_add]
                         /-
                           🎉 no goals
                         -/


lemma toMvPolynomial_comp (g : M₂ →ₗ[R] M₃) (f : M₁ →ₗ[R] M₂) (i : ι₃) :
    (g ∘ₗ f).toMvPolynomial b₁ b₃ i =
      bind₁ (f.toMvPolynomial b₁ b₂) (g.toMvPolynomial b₂ b₃ i) := by
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    M₃ : Type u_4
    ι₁ : Type u_5
    ι₂ : Type u_6
    ι₃ : Type u_7
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup M₁
    inst✝⁹ : AddCommGroup M₂
    inst✝⁸ : AddCommGroup M₃
    inst✝⁷ : Module R M₁
    inst✝⁶ : Module R M₂
    inst✝⁵ : Module R M₃
    inst✝⁴ : Fintype ι₁
    inst✝³ : Fintype ι₂
    inst✝² : Finite ι₃
    inst✝¹ : DecidableEq ι₁
    inst✝ : DecidableEq ι₂
    b₁ : Basis ι₁ R M₁
    b₂ : Basis ι₂ R M₂
    b₃ : Basis ι₃ R M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    f : LinearMap (RingHom.id R) M₁ M₂
    i : ι₃
    ⊢ Eq (LinearMap.toMvPolynomial b₁ b₃ (g.comp f) i) ((MvPolynomial.bind₁ (Linea …
  -/
  simp only [toMvPolynomial, toMatrix_comp b₁ b₂ b₃, Matrix.toMvPolynomial_mul]
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    M₃ : Type u_4
    ι₁ : Type u_5
    ι₂ : Type u_6
    ι₃ : Type u_7
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup M₁
    inst✝⁹ : AddCommGroup M₂
    inst✝⁸ : AddCommGroup M₃
    inst✝⁷ : Module R M₁
    inst✝⁶ : Module R M₂
    inst✝⁵ : Module R M₃
    inst✝⁴ : Fintype ι₁
    inst✝³ : Fintype ι₂
    inst✝² : Finite ι₃
    inst✝¹ : DecidableEq ι₁
    inst✝ : DecidableEq ι₂
    b₁ : Basis ι₁ R M₁
    b₂ : Basis ι₂ R M₂
    b₃ : Basis ι₃ R M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    f : LinearMap (RingHom.id R) M₁ M₂
    i : ι₃
    ⊢ Eq ((MvPolynomial.bind₁ ((LinearMap.toMatrix b₁ b₂) f).toMvPolynomial) (((Li …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- (Implementation detail, see `LinearMap.polyCharpoly`.)

Let `L` and `M` be finite free modules over `R`,
and let `φ : L →ₗ[R] Module.End R M` be a linear map.
Let `b` be a basis of `L` and `bₘ` a basis of `M`.
Then `LinearMap.polyCharpolyAux φ b bₘ` is the polynomial that evaluates on elements `x` of `L`
to the characteristic polynomial of `φ x` acting on `M`.

This definition does not depend on the choice of `bₘ`
(see `LinearMap.polyCharpolyAux_basisIndep`). -/
noncomputable
def polyCharpolyAux : Polynomial (MvPolynomial ι R) :=
  (charpoly.univ R ιM).map <| MvPolynomial.bind₁ (φ.toMvPolynomial b bₘ.end)


open Algebra.TensorProduct MvPolynomial in
lemma polyCharpolyAux_baseChange (A : Type*) [CommRing A] [Algebra R A] :
    polyCharpolyAux (tensorProduct _ _ _ _ ∘ₗ φ.baseChange A) (basis A b) (basis A bₘ) =
      (polyCharpolyAux φ b bₘ).map (MvPolynomial.map (algebraMap R A)) := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    A : Type u_8
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq (((LinearMap.tensorProduct R A M M).comp (LinearMap.baseChange A φ)).poly …
  -/
  simp only [polyCharpolyAux]
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    A : Type u_8
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq (Polynomial.map (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial (Algebra. …
  -/
  rw [← charpoly.univ_map_map _ (algebraMap R A)]
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    A : Type u_8
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq (Polynomial.map (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial (Algebra. …
  -/
  simp only [Polynomial.map_map]
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    A : Type u_8
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq (Polynomial.map ((↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial (Algebra …
  -/
  congr 1
  /-
    case e_f
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    A : Type u_8
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq ((↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial (Algebra.TensorProduct.b …
  -/
  apply ringHom_ext
    /-
      case e_f.hC
      R : Type u_1
      L : Type u_2
      M : Type u_3
      ι : Type u_5
      ιM : Type u_7
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup L
      inst✝⁸ : Module R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ιM
      inst✝³ : DecidableEq ι
      inst✝² : DecidableEq ιM
      b : Basis ι R L
      bₘ : Basis ιM R M
      A : Type u_8
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ⊢ ∀ (r : R), Eq (((↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial (Algebra.Ten …
    -/
  · intro r
    /-
      case e_f.hC
      R : Type u_1
      L : Type u_2
      M : Type u_3
      ι : Type u_5
      ιM : Type u_7
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup L
      inst✝⁸ : Module R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ιM
      inst✝³ : DecidableEq ι
      inst✝² : DecidableEq ιM
      b : Basis ι R L
      bₘ : Basis ιM R M
      A : Type u_8
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      r : R
      ⊢ Eq (((↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial (Algebra.TensorProduct. …
    -/
    simp only [RingHom.coe_comp, RingHom.coe_coe, Function.comp_apply, map_C, bind₁_C_right]
    /-
      🎉 no goals
    -/
    /-
      case e_f.hX
      R : Type u_1
      L : Type u_2
      M : Type u_3
      ι : Type u_5
      ιM : Type u_7
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup L
      inst✝⁸ : Module R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ιM
      inst✝³ : DecidableEq ι
      inst✝² : DecidableEq ιM
      b : Basis ι R L
      bₘ : Basis ιM R M
      A : Type u_8
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ⊢ ∀ (i : Prod ιM ιM), Eq (((↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial (Al …
    -/
  · rintro ij
    /-
      case e_f.hX
      R : Type u_1
      L : Type u_2
      M : Type u_3
      ι : Type u_5
      ιM : Type u_7
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup L
      inst✝⁸ : Module R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ιM
      inst✝³ : DecidableEq ι
      inst✝² : DecidableEq ιM
      b : Basis ι R L
      bₘ : Basis ιM R M
      A : Type u_8
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ij : Prod ιM ιM
      ⊢ Eq (((↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial (Algebra.TensorProduct. …
    -/
    simp only [RingHom.coe_comp, RingHom.coe_coe, Function.comp_apply, map_X, bind₁_X_right]
    classical
    rw [toMvPolynomial_comp _ (basis A (Basis.end bₘ)), ← toMvPolynomial_baseChange]
    #adaptation_note
    /--
    After https://github.com/leanprover/lean4/pull/4119 we either need to specify the `M₂` argument,
    or use `set_option maxSynthPendingDepth 2 in`.
    -/
    suffices toMvPolynomial (M₂ := (Module.End A (TensorProduct R A M)))
        (basis A bₘ.end) (basis A bₘ).end (tensorProduct R A M M) ij = X ij by
      rw [this, bind₁_X_right]
    simp only [toMvPolynomial, Matrix.toMvPolynomial]
    suffices ∀ kl,
        (toMatrix (basis A bₘ.end) (basis A bₘ).end) (tensorProduct R A M M) ij kl =
        if kl = ij then 1 else 0 by
      rw [Finset.sum_eq_single ij]
      · rw [this, if_pos rfl, X]
      · rintro kl - H
        rw [this, if_neg H, map_zero]
      · intro h
        exact (h (Finset.mem_univ _)).elim
    intro kl
    rw [toMatrix_apply, tensorProduct, TensorProduct.AlgebraTensorModule.lift_apply,
      basis_apply, TensorProduct.lift.tmul, coe_restrictScalars]
    dsimp only [coe_mk, AddHom.coe_mk, smul_apply, baseChangeHom_apply]
    rw [one_smul, Basis.baseChange_end, Basis.repr_self_apply]


open LinearMap in
lemma polyCharpolyAux_map_eq_toMatrix_charpoly (x : L) :
    (polyCharpolyAux φ b bₘ).map (MvPolynomial.eval (b.repr x)) =
      (toMatrix bₘ bₘ (φ x)).charpoly := by
  rw [polyCharpolyAux, Polynomial.map_map, ← MvPolynomial.eval₂Hom_C_eq_bind₁,
    MvPolynomial.comp_eval₂Hom, charpoly.univ_map_eval₂Hom]
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup L
    inst✝⁶ : Module R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝³ : Fintype ι
    inst✝² : Fintype ιM
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    x : L
    ⊢ Eq (Matrix.of (Function.curry fun i => (MvPolynomial.eval ⇑(b.repr x)) (Line …
  -/
  congr
  /-
    case e_M
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup L
    inst✝⁶ : Module R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝³ : Fintype ι
    inst✝² : Fintype ιM
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    x : L
    ⊢ Eq (Matrix.of (Function.curry fun i => (MvPolynomial.eval ⇑(b.repr x)) (Line …
  -/
  ext
  /-
    case e_M.a
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup L
    inst✝⁶ : Module R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝³ : Fintype ι
    inst✝² : Fintype ιM
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    x : L
    i✝ j✝ : ιM
    ⊢ Eq (Matrix.of (Function.curry fun i => (MvPolynomial.eval ⇑(b.repr x)) (Line …
  -/
  rw [of_apply, Function.curry_apply, toMvPolynomial_eval_eq_apply, LinearEquiv.symm_apply_apply]
  /-
    case e_M.a
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup L
    inst✝⁶ : Module R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝³ : Fintype ι
    inst✝² : Fintype ιM
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    x : L
    i✝ j✝ : ιM
    ⊢ Eq ((bₘ.end.repr (φ x)) { fst := i✝, snd := j✝ }) ((LinearMap.toMatrix bₘ bₘ …
  -/
  rfl
  /-
    🎉 no goals
  -/


open LinearMap in
lemma polyCharpolyAux_eval_eq_toMatrix_charpoly_coeff (x : L) (i : ℕ) :
    MvPolynomial.eval (b.repr x) ((polyCharpolyAux φ b bₘ).coeff i) =
      (toMatrix bₘ bₘ (φ x)).charpoly.coeff i := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup L
    inst✝⁶ : Module R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝³ : Fintype ι
    inst✝² : Fintype ιM
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    x : L
    i : Nat
    ⊢ Eq ((MvPolynomial.eval ⇑(b.repr x)) ((φ.polyCharpolyAux b bₘ).coeff i)) (((L …
  -/
  simp [← polyCharpolyAux_map_eq_toMatrix_charpoly φ b bₘ x]
  /-
    🎉 no goals
  -/


@[simp]
lemma polyCharpolyAux_map_eq_charpoly [Module.Finite R M] [Module.Free R M]
    (x : L) :
    (polyCharpolyAux φ b bₘ).map (MvPolynomial.eval (b.repr x)) = (φ x).charpoly := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Free R M
    x : L
    ⊢ Eq (Polynomial.map (MvPolynomial.eval ⇑(b.repr x)) (φ.polyCharpolyAux b bₘ)) …
  -/
  nontriviality R
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Free R M
    x : L
    a✝ : Nontrivial R
    ⊢ Eq (Polynomial.map (MvPolynomial.eval ⇑(b.repr x)) (φ.polyCharpolyAux b bₘ)) …
  -/
  rw [polyCharpolyAux_map_eq_toMatrix_charpoly, LinearMap.charpoly_toMatrix]
  /-
    🎉 no goals
  -/


@[simp]
lemma polyCharpolyAux_coeff_eval [Module.Finite R M] [Module.Free R M] (x : L) (i : ℕ) :
    MvPolynomial.eval (b.repr x) ((polyCharpolyAux φ b bₘ).coeff i) = (φ x).charpoly.coeff i := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Free R M
    x : L
    i : Nat
    ⊢ Eq ((MvPolynomial.eval ⇑(b.repr x)) ((φ.polyCharpolyAux b bₘ).coeff i)) ((Li …
  -/
  nontriviality R
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Free R M
    x : L
    i : Nat
    a✝ : Nontrivial R
    ⊢ Eq ((MvPolynomial.eval ⇑(b.repr x)) ((φ.polyCharpolyAux b bₘ).coeff i)) ((Li …
  -/
  rw [← polyCharpolyAux_map_eq_charpoly φ b bₘ x, Polynomial.coeff_map]
  /-
    🎉 no goals
  -/


lemma polyCharpolyAux_map_eval [Module.Finite R M] [Module.Free R M]
    (x : ι → R) :
    (polyCharpolyAux φ b bₘ).map (MvPolynomial.eval x) =
      (φ (b.repr.symm (Finsupp.equivFunOnFinite.symm x))).charpoly := by
  simp only [← polyCharpolyAux_map_eq_charpoly φ b bₘ, LinearEquiv.apply_symm_apply,
    Finsupp.equivFunOnFinite, Equiv.coe_fn_symm_mk, Finsupp.coe_mk]


open Algebra.TensorProduct TensorProduct in
lemma polyCharpolyAux_map_aeval
    (A : Type*) [CommRing A] [Algebra R A] [Module.Finite A (A ⊗[R] M)] [Module.Free A (A ⊗[R] M)]
    (x : ι → A) :
    (polyCharpolyAux φ b bₘ).map (MvPolynomial.aeval x).toRingHom =
      LinearMap.charpoly ((tensorProduct R A M M).comp (baseChange A φ)
        ((basis A b).repr.symm (Finsupp.equivFunOnFinite.symm x))) := by
  rw [← polyCharpolyAux_map_eval (tensorProduct R A M M ∘ₗ baseChange A φ) _ (basis A bₘ),
    polyCharpolyAux_baseChange, Polynomial.map_map]
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup L
    inst✝¹⁰ : Module R L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁷ : Fintype ι
    inst✝⁶ : Fintype ιM
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    A : Type u_8
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Module.Finite A (TensorProduct R A M)
    inst✝ : Module.Free A (TensorProduct R A M)
    x : ι → A
    ⊢ Eq (Polynomial.map (MvPolynomial.aeval x).toRingHom (φ.polyCharpolyAux b bₘ) …
  -/
  congr
  /-
    case e_f
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup L
    inst✝¹⁰ : Module R L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁷ : Fintype ι
    inst✝⁶ : Fintype ιM
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    A : Type u_8
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Module.Finite A (TensorProduct R A M)
    inst✝ : Module.Free A (TensorProduct R A M)
    x : ι → A
    ⊢ Eq (MvPolynomial.aeval x).toRingHom ((MvPolynomial.eval x).comp (MvPolynomia …
  -/
  exact DFunLike.ext _ _ fun f ↦ (MvPolynomial.eval_map (algebraMap R A) x f).symm
  /-
    🎉 no goals
  -/


open Algebra.TensorProduct MvPolynomial in
/-- `LinearMap.polyCharpolyAux` is independent of the choice of basis of the target module.

Proof strategy:
1. Rewrite `polyCharpolyAux` as the (honest, ordinary) characteristic polynomial
   of the basechange of `φ` to the multivariate polynomial ring `MvPolynomial ι R`.
2. Use that the characteristic polynomial of a linear map is independent of the choice of basis.
   This independence result is used transitively via
   `LinearMap.polyCharpolyAux_map_aeval` and `LinearMap.polyCharpolyAux_map_eq_charpoly`. -/
lemma polyCharpolyAux_basisIndep {ιM' : Type*} [Fintype ιM'] [DecidableEq ιM']
    (bₘ' : Basis ιM' R M) :
    polyCharpolyAux φ b bₘ = polyCharpolyAux φ b bₘ' := by
  let f : Polynomial (MvPolynomial ι R) → Polynomial (MvPolynomial ι R) :=
    Polynomial.map (MvPolynomial.aeval X).toRingHom
  have hf : Function.Injective f := by
    simp only [f, aeval_X_left, AlgHom.toRingHom_eq_coe, AlgHom.id_toRingHom, Polynomial.map_id]
    exact Polynomial.map_injective (RingHom.id _) Function.injective_id
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    ιM' : Type u_8
    inst✝¹ : Fintype ιM'
    inst✝ : DecidableEq ιM'
    bₘ' : Basis ιM' R M
    f : Polynomial (MvPolynomial ι R) → Polynomial (MvPolynomial ι R) := Polynomia …
    hf : Function.Injective f
    ⊢ Eq (φ.polyCharpolyAux b bₘ) (φ.polyCharpolyAux b bₘ')
  -/
  apply hf
  let _h1 : Module.Finite (MvPolynomial ι R) (TensorProduct R (MvPolynomial ι R) M) :=
    Module.Finite.of_basis (basis (MvPolynomial ι R) bₘ)
  let _h2 : Module.Free (MvPolynomial ι R) (TensorProduct R (MvPolynomial ι R) M) :=
    Module.Free.of_basis (basis (MvPolynomial ι R) bₘ)
  /-
    case a
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ιM : Type u_7
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ιM
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ιM
    b : Basis ι R L
    bₘ : Basis ιM R M
    ιM' : Type u_8
    inst✝¹ : Fintype ιM'
    inst✝ : DecidableEq ιM'
    bₘ' : Basis ιM' R M
    f : Polynomial (MvPolynomial ι R) → Polynomial (MvPolynomial ι R) := Polynomia …
    hf : Function.Injective f
    _h1 : Module.Finite (MvPolynomial ι R) (TensorProduct R (MvPolynomial ι R) M)  …
    _h2 : Module.Free (MvPolynomial ι R) (TensorProduct R (MvPolynomial ι R) M) := …
    ⊢ Eq (f (φ.polyCharpolyAux b bₘ)) (f (φ.polyCharpolyAux b bₘ'))
  -/
  simp only [f, polyCharpolyAux_map_aeval, polyCharpolyAux_map_aeval]
  /-
    🎉 no goals
  -/


/-- Let `L` and `M` be finite free modules over `R`,
and let `φ : L →ₗ[R] Module.End R M` be a linear family of endomorphisms.
Let `b` be a basis of `L` and `bₘ` a basis of `M`.
Then `LinearMap.polyCharpoly φ b` is the polynomial that evaluates on elements `x` of `L`
to the characteristic polynomial of `φ x` acting on `M`. -/
noncomputable
def polyCharpoly : Polynomial (MvPolynomial ι R) :=
  φ.polyCharpolyAux b (Module.Free.chooseBasis R M)


lemma polyCharpoly_eq_of_basis [DecidableEq ιM] (bₘ : Basis ιM R M) :
    polyCharpoly φ b =
    (charpoly.univ R ιM).map (MvPolynomial.bind₁ (φ.toMvPolynomial b bₘ.end)) := by
  rw [polyCharpoly, φ.polyCharpolyAux_basisIndep b (Module.Free.chooseBasis R M) bₘ,
    polyCharpolyAux]


lemma polyCharpoly_monic : (polyCharpoly φ b).Monic :=
  (charpoly.univ_monic R _).map _


lemma polyCharpoly_ne_zero [Nontrivial R] : (polyCharpoly φ b) ≠ 0 :=
  (polyCharpoly_monic _ _).ne_zero


@[simp]
lemma polyCharpoly_natDegree [Nontrivial R] :
    (polyCharpoly φ b).natDegree = finrank R M := by
  rw [polyCharpoly, polyCharpolyAux, (charpoly.univ_monic _ _).natDegree_map,
    charpoly.univ_natDegree, finrank_eq_card_chooseBasisIndex]


lemma polyCharpoly_coeff_isHomogeneous (i j : ℕ) (hij : i + j = finrank R M) [Nontrivial R] :
    ((polyCharpoly φ b).coeff i).IsHomogeneous j := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Fintype ι
    inst✝³ : DecidableEq ι
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    b : Basis ι R L
    i j : Nat
    hij : Eq (HAdd.hAdd i j) (Module.finrank R M)
    inst✝ : Nontrivial R
    ⊢ ((φ.polyCharpoly b).coeff i).IsHomogeneous j
  -/
  rw [finrank_eq_card_chooseBasisIndex] at hij
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Fintype ι
    inst✝³ : DecidableEq ι
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    b : Basis ι R L
    i j : Nat
    hij : Eq (HAdd.hAdd i j) (Fintype.card (Module.Free.ChooseBasisIndex R M))
    inst✝ : Nontrivial R
    ⊢ ((φ.polyCharpoly b).coeff i).IsHomogeneous j
  -/
  rw [polyCharpoly, polyCharpolyAux, Polynomial.coeff_map, ← one_mul j]
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Fintype ι
    inst✝³ : DecidableEq ι
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    b : Basis ι R L
    i j : Nat
    hij : Eq (HAdd.hAdd i j) (Fintype.card (Module.Free.ChooseBasisIndex R M))
    inst✝ : Nontrivial R
    ⊢ (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial b (Module.Free.chooseBasis R …
  -/
  apply (charpoly.univ_coeff_isHomogeneous _ _ _ _ hij).eval₂
    /-
      case hf
      R : Type u_1
      L : Type u_2
      M : Type u_3
      ι : Type u_5
      inst✝⁹ : CommRing R
      inst✝⁸ : AddCommGroup L
      inst✝⁷ : Module R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁴ : Fintype ι
      inst✝³ : DecidableEq ι
      inst✝² : Module.Free R M
      inst✝¹ : Module.Finite R M
      b : Basis ι R L
      i j : Nat
      hij : Eq (HAdd.hAdd i j) (Fintype.card (Module.Free.ChooseBasisIndex R M))
      inst✝ : Nontrivial R
      ⊢ ∀ (r : R), ((algebraMap R (MvPolynomial ι R)) r).IsHomogeneous 0
    -/
  · exact fun r ↦ MvPolynomial.isHomogeneous_C _ _
    /-
      🎉 no goals
    -/
    /-
      case hg
      R : Type u_1
      L : Type u_2
      M : Type u_3
      ι : Type u_5
      inst✝⁹ : CommRing R
      inst✝⁸ : AddCommGroup L
      inst✝⁷ : Module R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁴ : Fintype ι
      inst✝³ : DecidableEq ι
      inst✝² : Module.Free R M
      inst✝¹ : Module.Finite R M
      b : Basis ι R L
      i j : Nat
      hij : Eq (HAdd.hAdd i j) (Fintype.card (Module.Free.ChooseBasisIndex R M))
      inst✝ : Nontrivial R
      ⊢ ∀ (i : Prod (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisIndex …
    -/
  · exact LinearMap.toMvPolynomial_isHomogeneous _ _ _
    /-
      🎉 no goals
    -/


open Algebra.TensorProduct MvPolynomial in
lemma polyCharpoly_baseChange (A : Type*) [CommRing A] [Algebra R A] :
    polyCharpoly (tensorProduct _ _ _ _ ∘ₗ φ.baseChange A) (basis A b) =
      (polyCharpoly φ b).map (MvPolynomial.map (algebraMap R A)) := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : DecidableEq ι
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    b : Basis ι R L
    A : Type u_8
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq (((LinearMap.tensorProduct R A M M).comp (LinearMap.baseChange A φ)).poly …
  -/
  unfold polyCharpoly
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : DecidableEq ι
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    b : Basis ι R L
    A : Type u_8
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq (((LinearMap.tensorProduct R A M M).comp (LinearMap.baseChange A φ)).poly …
  -/
  rw [← φ.polyCharpolyAux_baseChange]
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : DecidableEq ι
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    b : Basis ι R L
    A : Type u_8
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq (((LinearMap.tensorProduct R A M M).comp (LinearMap.baseChange A φ)).poly …
  -/
  apply polyCharpolyAux_basisIndep
  /-
    🎉 no goals
  -/


@[simp]
lemma polyCharpoly_map_eq_charpoly (x : L) :
    (polyCharpoly φ b).map (MvPolynomial.eval (b.repr x)) = (φ x).charpoly := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup L
    inst✝⁶ : Module R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    b : Basis ι R L
    x : L
    ⊢ Eq (Polynomial.map (MvPolynomial.eval ⇑(b.repr x)) (φ.polyCharpoly b)) (Line …
  -/
  rw [polyCharpoly, polyCharpolyAux_map_eq_charpoly]
  /-
    🎉 no goals
  -/


@[simp]
lemma polyCharpoly_coeff_eval (x : L) (i : ℕ) :
    MvPolynomial.eval (b.repr x) ((polyCharpoly φ b).coeff i) = (φ x).charpoly.coeff i := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup L
    inst✝⁶ : Module R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    b : Basis ι R L
    x : L
    i : Nat
    ⊢ Eq ((MvPolynomial.eval ⇑(b.repr x)) ((φ.polyCharpoly b).coeff i)) ((LinearMa …
  -/
  rw [polyCharpoly, polyCharpolyAux_coeff_eval]
  /-
    🎉 no goals
  -/


lemma polyCharpoly_coeff_eq_zero_of_basis (b : Basis ι R L) (b' : Basis ι' R L) (k : ℕ)
    (H : (polyCharpoly φ b).coeff k = 0) :
    (polyCharpoly φ b').coeff k = 0 := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ι' : Type u_6
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ι'
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    b : Basis ι R L
    b' : Basis ι' R L
    k : Nat
    H : Eq ((φ.polyCharpoly b).coeff k) 0
    ⊢ Eq ((φ.polyCharpoly b').coeff k) 0
  -/
  rw [polyCharpoly, polyCharpolyAux, Polynomial.coeff_map] at H ⊢
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ι' : Type u_6
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ι'
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    b : Basis ι R L
    b' : Basis ι' R L
    k : Nat
    H : Eq (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial b (Module.Free.chooseBa …
    ⊢ Eq (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial b' (Module.Free.chooseBas …
  -/
  set B := (Module.Free.chooseBasis R M).end
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ι' : Type u_6
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ι'
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    b : Basis ι R L
    b' : Basis ι' R L
    k : Nat
    B : Basis (Prod (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisInd …
    H : Eq (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial b B φ)) ((Matrix.charpo …
    ⊢ Eq (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial b' B φ)) ((Matrix.charpol …
  -/
  set g := toMvPolynomial b' b LinearMap.id
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ι' : Type u_6
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ι'
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    b : Basis ι R L
    b' : Basis ι' R L
    k : Nat
    B : Basis (Prod (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisInd …
    H : Eq (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial b B φ)) ((Matrix.charpo …
    g : ι → MvPolynomial ι' R := LinearMap.toMvPolynomial b' b LinearMap.id
    ⊢ Eq (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial b' B φ)) ((Matrix.charpol …
  -/
  apply_fun (MvPolynomial.bind₁ g) at H
  have : toMvPolynomial b' B φ = fun i ↦ (MvPolynomial.bind₁ g) (toMvPolynomial b B φ i) :=
    funext <| toMvPolynomial_comp b' b B φ LinearMap.id
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ι' : Type u_6
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ι'
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    b : Basis ι R L
    b' : Basis ι' R L
    k : Nat
    B : Basis (Prod (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisInd …
    g : ι → MvPolynomial ι' R := LinearMap.toMvPolynomial b' b LinearMap.id
    H : Eq ((MvPolynomial.bind₁ g) (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial …
    this : Eq (LinearMap.toMvPolynomial b' B φ) fun i => (MvPolynomial.bind₁ g) (L …
    ⊢ Eq (↑(MvPolynomial.bind₁ (LinearMap.toMvPolynomial b' B φ)) ((Matrix.charpol …
  -/
  rwa [map_zero, RingHom.coe_coe, MvPolynomial.bind₁_bind₁, ← this] at H
  /-
    🎉 no goals
  -/


lemma polyCharpoly_coeff_eq_zero_iff_of_basis (b : Basis ι R L) (b' : Basis ι' R L) (k : ℕ) :
    (polyCharpoly φ b).coeff k = 0 ↔ (polyCharpoly φ b').coeff k = 0 := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ι' : Type u_6
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq ι'
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    b : Basis ι R L
    b' : Basis ι' R L
    k : Nat
    ⊢ Iff (Eq ((φ.polyCharpoly b).coeff k) 0) (Eq ((φ.polyCharpoly b').coeff k) 0)
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> apply polyCharpoly_coeff_eq_zero_of_basis
                  /-
                    🎉 no goals
                  -/


/-- (Implementation detail, see `LinearMap.nilRank`.)

Let `L` and `M` be finite free modules over `R`,
and let `φ : L →ₗ[R] Module.End R M` be a linear family of endomorphisms.
Then `LinearMap.nilRankAux φ b` is the smallest index
at which `LinearMap.polyCharpoly φ b` has a non-zero coefficient.

This number does not depend on the choice of `b`, see `nilRankAux_basis_indep`. -/
noncomputable
def nilRankAux (φ : L →ₗ[R] Module.End R M) (b : Basis ι R L) : ℕ :=
  (polyCharpoly φ b).natTrailingDegree


lemma polyCharpoly_coeff_nilRankAux_ne_zero [Nontrivial R] :
    (polyCharpoly φ b).coeff (nilRankAux φ b) ≠ 0 := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Fintype ι
    inst✝³ : DecidableEq ι
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    b : Basis ι R L
    inst✝ : Nontrivial R
    ⊢ Ne ((φ.polyCharpoly b).coeff (φ.nilRankAux b)) 0
  -/
  apply Polynomial.trailingCoeff_nonzero_iff_nonzero.mpr
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Fintype ι
    inst✝³ : DecidableEq ι
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    b : Basis ι R L
    inst✝ : Nontrivial R
    ⊢ Ne (φ.polyCharpoly b) 0
  -/
  apply polyCharpoly_ne_zero
  /-
    🎉 no goals
  -/


lemma nilRankAux_le [Nontrivial R] (b : Basis ι R L) (b' : Basis ι' R L) :
    nilRankAux φ b ≤ nilRankAux φ b' := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ι' : Type u_6
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup L
    inst✝⁹ : Module R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype ι'
    inst✝⁴ : DecidableEq ι
    inst✝³ : DecidableEq ι'
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial R
    b : Basis ι R L
    b' : Basis ι' R L
    ⊢ LE.le (φ.nilRankAux b) (φ.nilRankAux b')
  -/
  apply Polynomial.natTrailingDegree_le_of_ne_zero
  /-
    case h
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ι' : Type u_6
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup L
    inst✝⁹ : Module R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype ι'
    inst✝⁴ : DecidableEq ι
    inst✝³ : DecidableEq ι'
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial R
    b : Basis ι R L
    b' : Basis ι' R L
    ⊢ Ne ((φ.polyCharpoly b).coeff (φ.nilRankAux b')) 0
  -/
  rw [Ne, (polyCharpoly_coeff_eq_zero_iff_of_basis φ b b' _).not]
  /-
    case h
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ι' : Type u_6
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup L
    inst✝⁹ : Module R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype ι'
    inst✝⁴ : DecidableEq ι
    inst✝³ : DecidableEq ι'
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial R
    b : Basis ι R L
    b' : Basis ι' R L
    ⊢ Not (Eq ((φ.polyCharpoly b').coeff (φ.nilRankAux b')) 0)
  -/
  apply polyCharpoly_coeff_nilRankAux_ne_zero
  /-
    🎉 no goals
  -/


lemma nilRankAux_basis_indep [Nontrivial R] (b : Basis ι R L) (b' : Basis ι' R L) :
    nilRankAux φ b = (polyCharpoly φ b').natTrailingDegree := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    ι' : Type u_6
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup L
    inst✝⁹ : Module R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype ι'
    inst✝⁴ : DecidableEq ι
    inst✝³ : DecidableEq ι'
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial R
    b : Basis ι R L
    b' : Basis ι' R L
    ⊢ Eq (φ.nilRankAux b) (φ.polyCharpoly b').natTrailingDegree
  -/
                        /-
                          🎉 no goals
                        -/
  apply le_antisymm <;> apply nilRankAux_le
                        /-
                          🎉 no goals
                        -/


/-- Let `L` and `M` be finite free modules over `R`,
and let `φ : L →ₗ[R] Module.End R M` be a linear family of endomorphisms.
Then `LinearMap.nilRank φ b` is the smallest index
at which `LinearMap.polyCharpoly φ b` has a non-zero coefficient.

This number does not depend on the choice of `b`,
see `LinearMap.nilRank_eq_polyCharpoly_natTrailingDegree`. -/
noncomputable
def nilRank (φ : L →ₗ[R] Module.End R M) : ℕ :=
  nilRankAux φ (Module.Free.chooseBasis R L)


lemma nilRank_eq_polyCharpoly_natTrailingDegree (b : Basis ι R L) :
    nilRank φ = (polyCharpoly φ b).natTrailingDegree := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup L
    inst✝⁹ : Module R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁶ : Fintype ι
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : Nontrivial R
    b : Basis ι R L
    ⊢ Eq φ.nilRank (φ.polyCharpoly b).natTrailingDegree
  -/
  apply nilRankAux_basis_indep
  /-
    🎉 no goals
  -/


lemma polyCharpoly_coeff_nilRank_ne_zero :
    (polyCharpoly φ b).coeff (nilRank φ) ≠ 0 := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup L
    inst✝⁹ : Module R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁶ : Fintype ι
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    b : Basis ι R L
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : Nontrivial R
    ⊢ Ne ((φ.polyCharpoly b).coeff φ.nilRank) 0
  -/
  rw [nilRank_eq_polyCharpoly_natTrailingDegree _ b]
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup L
    inst✝⁹ : Module R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁶ : Fintype ι
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    b : Basis ι R L
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : Nontrivial R
    ⊢ Ne ((φ.polyCharpoly b).coeff (φ.polyCharpoly b).natTrailingDegree) 0
  -/
  apply polyCharpoly_coeff_nilRankAux_ne_zero
  /-
    🎉 no goals
  -/


lemma nilRank_le_card {ι : Type*} [Fintype ι] (b : Basis ι R M) : nilRank φ ≤ Fintype.card ι := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Module.Free R M
    inst✝⁴ : Module.Finite R M
    inst✝³ : Module.Finite R L
    inst✝² : Module.Free R L
    inst✝¹ : Nontrivial R
    ι : Type u_8
    inst✝ : Fintype ι
    b : Basis ι R M
    ⊢ LE.le φ.nilRank (Fintype.card ι)
  -/
  apply Polynomial.natTrailingDegree_le_of_ne_zero
  rw [← Module.finrank_eq_card_basis b, ← polyCharpoly_natDegree φ (chooseBasis R L),
    Polynomial.coeff_natDegree, (polyCharpoly_monic _ _).leadingCoeff]
  /-
    case h
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Module.Free R M
    inst✝⁴ : Module.Finite R M
    inst✝³ : Module.Finite R L
    inst✝² : Module.Free R L
    inst✝¹ : Nontrivial R
    ι : Type u_8
    inst✝ : Fintype ι
    b : Basis ι R M
    ⊢ Ne 1 0
  -/
  apply one_ne_zero
  /-
    🎉 no goals
  -/


lemma nilRank_le_finrank : nilRank φ ≤ finrank R M := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : Nontrivial R
    ⊢ LE.le φ.nilRank (Module.finrank R M)
  -/
  simpa only [finrank_eq_card_chooseBasisIndex R M] using nilRank_le_card φ (chooseBasis R M)
  /-
    🎉 no goals
  -/


lemma nilRank_le_natTrailingDegree_charpoly (x : L) :
    nilRank φ ≤ (φ x).charpoly.natTrailingDegree := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : Nontrivial R
    x : L
    ⊢ LE.le φ.nilRank (LinearMap.charpoly (φ x)).natTrailingDegree
  -/
  apply Polynomial.natTrailingDegree_le_of_ne_zero
  /-
    case h
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : Nontrivial R
    x : L
    ⊢ Ne ((φ.polyCharpoly (Module.Free.chooseBasis R L)).coeff (LinearMap.charpoly …
  -/
  intro h
  /-
    case h
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : Nontrivial R
    x : L
    h : Eq ((φ.polyCharpoly (Module.Free.chooseBasis R L)).coeff (LinearMap.charpo …
    ⊢ False
  -/
  apply_fun (MvPolynomial.eval ((chooseBasis R L).repr x)) at h
  /-
    case h
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : Nontrivial R
    x : L
    h : Eq ((MvPolynomial.eval ⇑((Module.Free.chooseBasis R L).repr x)) ((φ.polyCh …
    ⊢ False
  -/
  rw [polyCharpoly_coeff_eval, map_zero] at h
  /-
    case h
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : Nontrivial R
    x : L
    h : Eq ((LinearMap.charpoly (φ x)).coeff (LinearMap.charpoly (φ x)).natTrailin …
    ⊢ False
  -/
  apply Polynomial.trailingCoeff_nonzero_iff_nonzero.mpr _ h
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : Nontrivial R
    x : L
    h : Eq ((LinearMap.charpoly (φ x)).coeff (LinearMap.charpoly (φ x)).natTrailin …
    ⊢ Ne (LinearMap.charpoly (φ x)) 0
  -/
  apply (LinearMap.charpoly_monic _).ne_zero
  /-
    🎉 no goals
  -/


/-- Let `L` and `M` be finite free modules over `R`,
and let `φ : L →ₗ[R] Module.End R M` be a linear family of endomorphisms,
and denote `n := nilRank φ`.

An element `x : L` is *nil-regular* with respect to `φ`
if the `n`-th coefficient of the characteristic polynomial of `φ x` is non-zero. -/
def IsNilRegular (x : L) : Prop :=
  Polynomial.coeff (φ x).charpoly (nilRank φ) ≠ 0


lemma isNilRegular_def :
    IsNilRegular φ x ↔ (Polynomial.coeff (φ x).charpoly (nilRank φ) ≠ 0) := Iff.rfl


lemma isNilRegular_iff_coeff_polyCharpoly_nilRank_ne_zero :
    IsNilRegular φ x ↔
    MvPolynomial.eval (b.repr x)
      ((polyCharpoly φ b).coeff (nilRank φ)) ≠ 0 := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    ι : Type u_5
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Fintype ι
    inst✝⁴ : DecidableEq ι
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    b : Basis ι R L
    inst✝¹ : Module.Finite R L
    inst✝ : Module.Free R L
    x : L
    ⊢ Iff (φ.IsNilRegular x) (Ne ((MvPolynomial.eval ⇑(b.repr x)) ((φ.polyCharpoly …
  -/
  rw [IsNilRegular, polyCharpoly_coeff_eval]
  /-
    🎉 no goals
  -/


lemma isNilRegular_iff_natTrailingDegree_charpoly_eq_nilRank [Nontrivial R] :
    IsNilRegular φ x ↔ (φ x).charpoly.natTrailingDegree = nilRank φ := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    x : L
    inst✝ : Nontrivial R
    ⊢ Iff (φ.IsNilRegular x) (Eq (LinearMap.charpoly (φ x)).natTrailingDegree φ.ni …
  -/
  rw [isNilRegular_def]
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    x : L
    inst✝ : Nontrivial R
    ⊢ Iff (Ne ((LinearMap.charpoly (φ x)).coeff φ.nilRank) 0) (Eq (LinearMap.charp …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁹ : CommRing R
      inst✝⁸ : AddCommGroup L
      inst✝⁷ : Module R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁴ : Module.Free R M
      inst✝³ : Module.Finite R M
      inst✝² : Module.Finite R L
      inst✝¹ : Module.Free R L
      x : L
      inst✝ : Nontrivial R
      ⊢ Ne ((LinearMap.charpoly (φ x)).coeff φ.nilRank) 0 → Eq (LinearMap.charpoly ( …
    -/
  · intro h
    exact le_antisymm
      (Polynomial.natTrailingDegree_le_of_ne_zero h)
      (nilRank_le_natTrailingDegree_charpoly φ x)
    /-
      case mpr
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁹ : CommRing R
      inst✝⁸ : AddCommGroup L
      inst✝⁷ : Module R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁴ : Module.Free R M
      inst✝³ : Module.Finite R M
      inst✝² : Module.Finite R L
      inst✝¹ : Module.Free R L
      x : L
      inst✝ : Nontrivial R
      ⊢ Eq (LinearMap.charpoly (φ x)).natTrailingDegree φ.nilRank → Ne ((LinearMap.c …
    -/
  · intro h
    /-
      case mpr
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁹ : CommRing R
      inst✝⁸ : AddCommGroup L
      inst✝⁷ : Module R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁴ : Module.Free R M
      inst✝³ : Module.Finite R M
      inst✝² : Module.Finite R L
      inst✝¹ : Module.Free R L
      x : L
      inst✝ : Nontrivial R
      h : Eq (LinearMap.charpoly (φ x)).natTrailingDegree φ.nilRank
      ⊢ Ne ((LinearMap.charpoly (φ x)).coeff φ.nilRank) 0
    -/
    rw [← h]
    /-
      case mpr
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁹ : CommRing R
      inst✝⁸ : AddCommGroup L
      inst✝⁷ : Module R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁴ : Module.Free R M
      inst✝³ : Module.Finite R M
      inst✝² : Module.Finite R L
      inst✝¹ : Module.Free R L
      x : L
      inst✝ : Nontrivial R
      h : Eq (LinearMap.charpoly (φ x)).natTrailingDegree φ.nilRank
      ⊢ Ne ((LinearMap.charpoly (φ x)).coeff (LinearMap.charpoly (φ x)).natTrailingD …
    -/
    apply Polynomial.trailingCoeff_nonzero_iff_nonzero.mpr
    /-
      case mpr
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁹ : CommRing R
      inst✝⁸ : AddCommGroup L
      inst✝⁷ : Module R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      φ : LinearMap (RingHom.id R) L (Module.End R M)
      inst✝⁴ : Module.Free R M
      inst✝³ : Module.Finite R M
      inst✝² : Module.Finite R L
      inst✝¹ : Module.Free R L
      x : L
      inst✝ : Nontrivial R
      h : Eq (LinearMap.charpoly (φ x)).natTrailingDegree φ.nilRank
      ⊢ Ne (LinearMap.charpoly (φ x)) 0
    -/
    apply (LinearMap.charpoly_monic _).ne_zero
    /-
      🎉 no goals
    -/


open Cardinal Module MvPolynomial Module.Free in
lemma exists_isNilRegular_of_finrank_le_card (h : finrank R M ≤ #R) :
    ∃ x : L, IsNilRegular φ x := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : IsDomain R
    h : LE.le (↑(Module.finrank R M)) (Cardinal.mk R)
    ⊢ Exists fun x => φ.IsNilRegular x
  -/
  let b := chooseBasis R L
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : IsDomain R
    h : LE.le (↑(Module.finrank R M)) (Cardinal.mk R)
    b : Basis (Module.Free.ChooseBasisIndex R L) R L := Module.Free.chooseBasis R L
    ⊢ Exists fun x => φ.IsNilRegular x
  -/
  let bₘ := chooseBasis R M
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : IsDomain R
    h : LE.le (↑(Module.finrank R M)) (Cardinal.mk R)
    b : Basis (Module.Free.ChooseBasisIndex R L) R L := Module.Free.chooseBasis R L
    bₘ : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ⊢ Exists fun x => φ.IsNilRegular x
  -/
  let n := Fintype.card (ChooseBasisIndex R M)
  have aux :
    ((polyCharpoly φ b).coeff (nilRank φ)).IsHomogeneous (n - nilRank φ) :=
    polyCharpoly_coeff_isHomogeneous _ b (nilRank φ) (n - nilRank φ)
      (by simp [n, nilRank_le_card φ bₘ, finrank_eq_card_chooseBasisIndex])
  obtain ⟨x, hx⟩ : ∃ r, eval r ((polyCharpoly _ b).coeff (nilRank φ)) ≠ 0 := by
    by_contra! h₀
    apply polyCharpoly_coeff_nilRank_ne_zero φ b
    apply aux.eq_zero_of_forall_eval_eq_zero_of_le_card h₀ (le_trans _ h)
    simp only [n, finrank_eq_card_chooseBasisIndex, Nat.cast_le, Nat.sub_le]
  /-
    case intro
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : IsDomain R
    h : LE.le (↑(Module.finrank R M)) (Cardinal.mk R)
    b : Basis (Module.Free.ChooseBasisIndex R L) R L := Module.Free.chooseBasis R L
    bₘ : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    n : Nat := Fintype.card (Module.Free.ChooseBasisIndex R M)
    aux : ((φ.polyCharpoly b).coeff φ.nilRank).IsHomogeneous (HSub.hSub n φ.nilRank)
    x : Module.Free.ChooseBasisIndex R L → R
    hx : Ne ((MvPolynomial.eval x) ((φ.polyCharpoly b).coeff φ.nilRank)) 0
    ⊢ Exists fun x => φ.IsNilRegular x
  -/
  let c := Finsupp.equivFunOnFinite.symm x
  /-
    case intro
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : IsDomain R
    h : LE.le (↑(Module.finrank R M)) (Cardinal.mk R)
    b : Basis (Module.Free.ChooseBasisIndex R L) R L := Module.Free.chooseBasis R L
    bₘ : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    n : Nat := Fintype.card (Module.Free.ChooseBasisIndex R M)
    aux : ((φ.polyCharpoly b).coeff φ.nilRank).IsHomogeneous (HSub.hSub n φ.nilRank)
    x : Module.Free.ChooseBasisIndex R L → R
    hx : Ne ((MvPolynomial.eval x) ((φ.polyCharpoly b).coeff φ.nilRank)) 0
    c : Finsupp (Module.Free.ChooseBasisIndex R L) R := Finsupp.equivFunOnFinite.s …
    ⊢ Exists fun x => φ.IsNilRegular x
  -/
  use b.repr.symm c
  /-
    case h
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup L
    inst✝⁷ : Module R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Finite R L
    inst✝¹ : Module.Free R L
    inst✝ : IsDomain R
    h : LE.le (↑(Module.finrank R M)) (Cardinal.mk R)
    b : Basis (Module.Free.ChooseBasisIndex R L) R L := Module.Free.chooseBasis R L
    bₘ : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    n : Nat := Fintype.card (Module.Free.ChooseBasisIndex R M)
    aux : ((φ.polyCharpoly b).coeff φ.nilRank).IsHomogeneous (HSub.hSub n φ.nilRank)
    x : Module.Free.ChooseBasisIndex R L → R
    hx : Ne ((MvPolynomial.eval x) ((φ.polyCharpoly b).coeff φ.nilRank)) 0
    c : Finsupp (Module.Free.ChooseBasisIndex R L) R := Finsupp.equivFunOnFinite.s …
    ⊢ φ.IsNilRegular (b.repr.symm c)
  -/
  rwa [isNilRegular_iff_coeff_polyCharpoly_nilRank_ne_zero _ b, LinearEquiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


lemma exists_isNilRegular [Infinite R] : ∃ x : L, IsNilRegular φ x := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Module.Free R M
    inst✝⁴ : Module.Finite R M
    inst✝³ : Module.Finite R L
    inst✝² : Module.Free R L
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    ⊢ Exists fun x => φ.IsNilRegular x
  -/
  apply exists_isNilRegular_of_finrank_le_card
  /-
    case h
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup L
    inst✝⁸ : Module R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    φ : LinearMap (RingHom.id R) L (Module.End R M)
    inst✝⁵ : Module.Free R M
    inst✝⁴ : Module.Finite R M
    inst✝³ : Module.Finite R L
    inst✝² : Module.Free R L
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    ⊢ LE.le (↑(Module.finrank R M)) (Cardinal.mk R)
  -/
  exact (Cardinal.nat_lt_aleph0 _).le.trans <| Cardinal.infinite_iff.mp ‹Infinite R›
  /-
    🎉 no goals
  -/


