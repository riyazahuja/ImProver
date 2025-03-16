/-- If the quadratic form of a vector is invertible, then so is that vector. -/
def invertibleιOfInvertible (m : M) [Invertible (Q m)] : Invertible (ι Q m) where
  invOf := ι Q (⅟ (Q m) • m)
  invOf_mul_self := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝³ : CommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      Q : QuadraticForm R M
      m : M
      inst✝ : Invertible (Q m)
      ⊢ Eq (HMul.hMul ((CliffordAlgebra.ι Q) (HSMul.hSMul (Invertible.invOf (Q m)) m …
    -/
    rw [map_smul, smul_mul_assoc, ι_sq_scalar, Algebra.smul_def, ← map_mul, invOf_mul_self, map_one]
    /-
      🎉 no goals
    -/
  mul_invOf_self := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝³ : CommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      Q : QuadraticForm R M
      m : M
      inst✝ : Invertible (Q m)
      ⊢ Eq (HMul.hMul ((CliffordAlgebra.ι Q) m) ((CliffordAlgebra.ι Q) (HSMul.hSMul  …
    -/
    rw [map_smul, mul_smul_comm, ι_sq_scalar, Algebra.smul_def, ← map_mul, invOf_mul_self, map_one]
    /-
      🎉 no goals
    -/


/-- For a vector with invertible quadratic form, $v^{-1} = \frac{v}{Q(v)}$ -/
theorem invOf_ι (m : M) [Invertible (Q m)] [Invertible (ι Q m)] :
    ⅟ (ι Q m) = ι Q (⅟ (Q m) • m) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    m : M
    inst✝¹ : Invertible (Q m)
    inst✝ : Invertible ((CliffordAlgebra.ι Q) m)
    ⊢ Eq (Invertible.invOf ((CliffordAlgebra.ι Q) m)) ((CliffordAlgebra.ι Q) (HSMu …
  -/
  letI := invertibleιOfInvertible Q m
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    m : M
    inst✝¹ : Invertible (Q m)
    inst✝ : Invertible ((CliffordAlgebra.ι Q) m)
    this : Invertible ((CliffordAlgebra.ι Q) m) := CliffordAlgebra.invertibleιOfIn …
    ⊢ Eq (Invertible.invOf ((CliffordAlgebra.ι Q) m)) ((CliffordAlgebra.ι Q) (HSMu …
  -/
  convert (rfl : ⅟ (ι Q m) = _)
  /-
    🎉 no goals
  -/


theorem isUnit_ι_of_isUnit {m : M} (h : IsUnit (Q m)) : IsUnit (ι Q m) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    m : M
    h : IsUnit (Q m)
    ⊢ IsUnit ((CliffordAlgebra.ι Q) m)
  -/
  cases h.nonempty_invertible
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    m : M
    h : IsUnit (Q m)
    val✝ : Invertible (Q m)
    ⊢ IsUnit ((CliffordAlgebra.ι Q) m)
  -/
  letI := invertibleιOfInvertible Q m
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    m : M
    h : IsUnit (Q m)
    val✝ : Invertible (Q m)
    this : Invertible ((CliffordAlgebra.ι Q) m) := CliffordAlgebra.invertibleιOfIn …
    ⊢ IsUnit ((CliffordAlgebra.ι Q) m)
  -/
  exact isUnit_of_invertible (ι Q m)
  /-
    🎉 no goals
  -/


/-- $aba^{-1}$ is a vector. -/
theorem ι_mul_ι_mul_invOf_ι (a b : M) [Invertible (ι Q a)] [Invertible (Q a)] :
    ι Q a * ι Q b * ⅟ (ι Q a) = ι Q ((⅟ (Q a) * QuadraticMap.polar Q a b) • a - b) := by
  rw [invOf_ι, map_smul, mul_smul_comm, ι_mul_ι_mul_ι, ← map_smul, smul_sub, smul_smul, smul_smul,
    invOf_mul_self, one_smul]


/-- $a^{-1}ba$ is a vector. -/
theorem invOf_ι_mul_ι_mul_ι (a b : M) [Invertible (ι Q a)] [Invertible (Q a)] :
    ⅟ (ι Q a) * ι Q b * ι Q a = ι Q ((⅟ (Q a) * QuadraticMap.polar Q a b) • a - b) := by
  rw [invOf_ι, map_smul, smul_mul_assoc, smul_mul_assoc, ι_mul_ι_mul_ι, ← map_smul, smul_sub,
    smul_smul, smul_smul, invOf_mul_self, one_smul]


/-- Over a ring where `2` is invertible, `Q m` is invertible whenever `ι Q m`. -/
def invertibleOfInvertibleι (m : M) [Invertible (ι Q m)] : Invertible (Q m) :=
  ExteriorAlgebra.invertibleAlgebraMapEquiv M (Q m) <|
                                                                        /-
                                                                          R : Type u_1
                                                                          M : Type u_2
                                                                          inst✝⁴ : CommRing R
                                                                          inst✝³ : AddCommGroup M
                                                                          inst✝² : Module R M
                                                                          Q : QuadraticForm R M
                                                                          inst✝¹ : Invertible 2
                                                                          m : M
                                                                          inst✝ : Invertible ((CliffordAlgebra.ι Q) m)
                                                                          ⊢ Eq (↑(CliffordAlgebra.equivExterior Q) 1) 1
                                                                        -/
    .algebraMapOfInvertibleAlgebraMap (equivExterior Q).toLinearMap (by simp) <|
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
      .copy (.mul ‹Invertible (ι Q m)› ‹Invertible (ι Q m)›) _ (ι_sq_scalar _ _).symm


theorem isUnit_of_isUnit_ι {m : M} (h : IsUnit (ι Q m)) : IsUnit (Q m) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    inst✝ : Invertible 2
    m : M
    h : IsUnit ((CliffordAlgebra.ι Q) m)
    ⊢ IsUnit (Q m)
  -/
  cases h.nonempty_invertible
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    inst✝ : Invertible 2
    m : M
    h : IsUnit ((CliffordAlgebra.ι Q) m)
    val✝ : Invertible ((CliffordAlgebra.ι Q) m)
    ⊢ IsUnit (Q m)
  -/
  letI := invertibleOfInvertibleι Q m
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    inst✝ : Invertible 2
    m : M
    h : IsUnit ((CliffordAlgebra.ι Q) m)
    val✝ : Invertible ((CliffordAlgebra.ι Q) m)
    this : Invertible (Q m) := CliffordAlgebra.invertibleOfInvertibleι Q m
    ⊢ IsUnit (Q m)
  -/
  exact isUnit_of_invertible (Q m)
  /-
    🎉 no goals
  -/


@[simp] theorem isUnit_ι_iff {m : M} : IsUnit (ι Q m) ↔ IsUnit (Q m) :=
  ⟨isUnit_of_isUnit_ι Q, isUnit_ι_of_isUnit Q⟩


