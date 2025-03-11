/-- From a basis `e : ι → M` and a family of vectors `v : ι' → M`, make the matrix whose columns
are the vectors `v i` written in the basis `e`. -/
def Basis.toMatrix (e : Basis ι R M) (v : ι' → M) : Matrix ι ι' R := fun i j => e.repr (v j) i


theorem toMatrix_apply : e.toMatrix v i j = e.repr (v j) i :=
  rfl


theorem toMatrix_transpose_apply : (e.toMatrix v)ᵀ j = e.repr (v j) :=
  funext fun _ => rfl


theorem toMatrix_eq_toMatrix_constr [Fintype ι] [DecidableEq ι] (v : ι → M) :
    e.toMatrix v = LinearMap.toMatrix e e (e.constr ℕ v) := by
  /-
    ι : Type u_1
    R : Type u_5
    M : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    e : Basis ι R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    v : ι → M
    ⊢ Eq (e.toMatrix v) ((LinearMap.toMatrix e e) ((e.constr Nat) v))
  -/
  ext
  /-
    case a
    ι : Type u_1
    R : Type u_5
    M : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    e : Basis ι R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    v : ι → M
    i✝ j✝ : ι
    ⊢ Eq (e.toMatrix v i✝ j✝) ((LinearMap.toMatrix e e) ((e.constr Nat) v) i✝ j✝)
  -/
  rw [Basis.toMatrix_apply, LinearMap.toMatrix_apply, Basis.constr_basis]
  /-
    🎉 no goals
  -/

-- TODO (maybe) Adjust the definition of `Basis.toMatrix` to eliminate the transpose.

theorem coePiBasisFun.toMatrix_eq_transpose [Finite ι] :
    ((Pi.basisFun R ι).toMatrix : Matrix ι ι R → Matrix ι ι R) = Matrix.transpose := by
  /-
    ι : Type u_1
    R : Type u_5
    inst✝¹ : CommSemiring R
    inst✝ : Finite ι
    ⊢ Eq (Pi.basisFun R ι).toMatrix Matrix.transpose
  -/
  ext M i j
  /-
    case h.a
    ι : Type u_1
    R : Type u_5
    inst✝¹ : CommSemiring R
    inst✝ : Finite ι
    M : ι → ι → R
    i j : ι
    ⊢ Eq ((Pi.basisFun R ι).toMatrix M i j) (Matrix.transpose M i j)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toMatrix_self [DecidableEq ι] : e.toMatrix e = 1 := by
  /-
    ι : Type u_1
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    e : Basis ι R M
    inst✝ : DecidableEq ι
    ⊢ Eq (e.toMatrix ⇑e) 1
  -/
  unfold Basis.toMatrix
  /-
    ι : Type u_1
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    e : Basis ι R M
    inst✝ : DecidableEq ι
    ⊢ Eq (fun i j => (e.repr (e j)) i) 1
  -/
  ext i j
  /-
    case a
    ι : Type u_1
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    e : Basis ι R M
    inst✝ : DecidableEq ι
    i j : ι
    ⊢ Eq ((e.repr (e j)) i) (1 i j)
  -/
  simp [Basis.equivFun, Matrix.one_apply, Finsupp.single_apply, eq_comm]
  /-
    🎉 no goals
  -/


theorem toMatrix_update [DecidableEq ι'] (x : M) :
    e.toMatrix (Function.update v j x) = Matrix.updateCol (e.toMatrix v) j (e.repr x) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    e : Basis ι R M
    v : ι' → M
    j : ι'
    inst✝ : DecidableEq ι'
    x : M
    ⊢ Eq (e.toMatrix (Function.update v j x)) ((e.toMatrix v).updateCol j ⇑(e.repr …
  -/
  ext i' k
  /-
    case a
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    e : Basis ι R M
    v : ι' → M
    j : ι'
    inst✝ : DecidableEq ι'
    x : M
    i' : ι
    k : ι'
    ⊢ Eq (e.toMatrix (Function.update v j x) i' k) ((e.toMatrix v).updateCol j (⇑( …
  -/
  rw [Basis.toMatrix, Matrix.updateCol_apply, e.toMatrix_apply]
  /-
    case a
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    e : Basis ι R M
    v : ι' → M
    j : ι'
    inst✝ : DecidableEq ι'
    x : M
    i' : ι
    k : ι'
    ⊢ Eq ((e.repr (Function.update v j x k)) i') (ite (Eq k j) ((e.repr x) i') ((e …
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      R : Type u_5
      M : Type u_6
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      e : Basis ι R M
      v : ι' → M
      j : ι'
      inst✝ : DecidableEq ι'
      x : M
      i' : ι
      k : ι'
      h : Eq k j
      ⊢ Eq ((e.repr (Function.update v j x k)) i') ((e.repr x) i')
    -/
  · rw [h, update_self j x v]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      R : Type u_5
      M : Type u_6
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      e : Basis ι R M
      v : ι' → M
      j : ι'
      inst✝ : DecidableEq ι'
      x : M
      i' : ι
      k : ι'
      h : Not (Eq k j)
      ⊢ Eq ((e.repr (Function.update v j x k)) i') ((e.repr (v k)) i')
    -/
  · rw [update_of_ne h]
    /-
      🎉 no goals
    -/


/-- The basis constructed by `unitsSMul` has vectors given by a diagonal matrix. -/
@[simp]
theorem toMatrix_unitsSMul [DecidableEq ι] (e : Basis ι R₂ M₂) (w : ι → R₂ˣ) :
    e.toMatrix (e.unitsSMul w) = diagonal ((↑) ∘ w) := by
  /-
    ι : Type u_1
    R₂ : Type u_7
    M₂ : Type u_8
    inst✝³ : CommRing R₂
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    inst✝ : DecidableEq ι
    e : Basis ι R₂ M₂
    w : ι → Units R₂
    ⊢ Eq (e.toMatrix ⇑(e.unitsSMul w)) (Matrix.diagonal (Function.comp Units.val w))
  -/
  ext i j
  /-
    case a
    ι : Type u_1
    R₂ : Type u_7
    M₂ : Type u_8
    inst✝³ : CommRing R₂
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    inst✝ : DecidableEq ι
    e : Basis ι R₂ M₂
    w : ι → Units R₂
    i j : ι
    ⊢ Eq (e.toMatrix (⇑(e.unitsSMul w)) i j) (Matrix.diagonal (Function.comp Units …
  -/
  by_cases h : i = j
    /-
      case pos
      ι : Type u_1
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      inst✝ : DecidableEq ι
      e : Basis ι R₂ M₂
      w : ι → Units R₂
      i j : ι
      h : Eq i j
      ⊢ Eq (e.toMatrix (⇑(e.unitsSMul w)) i j) (Matrix.diagonal (Function.comp Units …
    -/
  · simp [h, toMatrix_apply, unitsSMul_apply, Units.smul_def]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      inst✝ : DecidableEq ι
      e : Basis ι R₂ M₂
      w : ι → Units R₂
      i j : ι
      h : Not (Eq i j)
      ⊢ Eq (e.toMatrix (⇑(e.unitsSMul w)) i j) (Matrix.diagonal (Function.comp Units …
    -/
  · simp [h, toMatrix_apply, unitsSMul_apply, Units.smul_def, Ne.symm h]
    /-
      🎉 no goals
    -/


/-- The basis constructed by `isUnitSMul` has vectors given by a diagonal matrix. -/
@[simp]
theorem toMatrix_isUnitSMul [DecidableEq ι] (e : Basis ι R₂ M₂) {w : ι → R₂}
    (hw : ∀ i, IsUnit (w i)) : e.toMatrix (e.isUnitSMul hw) = diagonal w :=
  e.toMatrix_unitsSMul _


theorem toMatrix_smul_left {G} [Group G] [DistribMulAction G M] [SMulCommClass G R M] (g : G) :
    (g • e).toMatrix v = e.toMatrix (g⁻¹ • v) := rfl


@[simp]
theorem sum_toMatrix_smul_self [Fintype ι] : ∑ i : ι, e.toMatrix v i j • e i = v j := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    e : Basis ι R M
    v : ι' → M
    j : ι'
    inst✝ : Fintype ι
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (e.toMatrix v i j) (e i)) (v j)
  -/
  simp_rw [e.toMatrix_apply, e.sum_repr]
  /-
    🎉 no goals
  -/


theorem toMatrix_smul {R₁ S : Type*} [CommRing R₁] [Ring S] [Algebra R₁ S] [Fintype ι]
    [DecidableEq ι] (x : S) (b : Basis ι R₁ S) (w : ι → S) :
    (b.toMatrix (x • w)) = (Algebra.leftMulMatrix b x) * (b.toMatrix w) := by
  /-
    ι : Type u_1
    R₁ : Type u_9
    S : Type u_10
    inst✝⁴ : CommRing R₁
    inst✝³ : Ring S
    inst✝² : Algebra R₁ S
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : S
    b : Basis ι R₁ S
    w : ι → S
    ⊢ Eq (b.toMatrix (HSMul.hSMul x w)) (HMul.hMul ((Algebra.leftMulMatrix b) x) ( …
  -/
  ext
  /-
    case a
    ι : Type u_1
    R₁ : Type u_9
    S : Type u_10
    inst✝⁴ : CommRing R₁
    inst✝³ : Ring S
    inst✝² : Algebra R₁ S
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : S
    b : Basis ι R₁ S
    w : ι → S
    i✝ j✝ : ι
    ⊢ Eq (b.toMatrix (HSMul.hSMul x w) i✝ j✝) (HMul.hMul ((Algebra.leftMulMatrix b …
  -/
  rw [Basis.toMatrix_apply, Pi.smul_apply, smul_eq_mul, ← Algebra.leftMulMatrix_mulVec_repr]
  /-
    case a
    ι : Type u_1
    R₁ : Type u_9
    S : Type u_10
    inst✝⁴ : CommRing R₁
    inst✝³ : Ring S
    inst✝² : Algebra R₁ S
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : S
    b : Basis ι R₁ S
    w : ι → S
    i✝ j✝ : ι
    ⊢ Eq (((Algebra.leftMulMatrix b) x).mulVec (⇑(b.repr (w j✝))) i✝) (HMul.hMul ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toMatrix_map_vecMul {S : Type*} [Ring S] [Algebra R S] [Fintype ι] (b : Basis ι R S)
    (v : ι' → S) : b ᵥ* ((b.toMatrix v).map <| algebraMap R S) = v := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    inst✝³ : CommSemiring R
    S : Type u_9
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    inst✝ : Fintype ι
    b : Basis ι R S
    v : ι' → S
    ⊢ Eq (Matrix.vecMul (⇑b) ((b.toMatrix v).map ⇑(algebraMap R S))) v
  -/
  ext i
  simp_rw [vecMul, dotProduct, Matrix.map_apply, ← Algebra.commutes, ← Algebra.smul_def,
    sum_toMatrix_smul_self]


@[simp]
theorem toLin_toMatrix [Finite ι] [Fintype ι'] [DecidableEq ι'] (v : Basis ι' R M) :
    Matrix.toLin v e (e.toMatrix v) = LinearMap.id :=
                    /-
                      ι : Type u_1
                      ι' : Type u_2
                      R : Type u_5
                      M : Type u_6
                      inst✝⁵ : CommSemiring R
                      inst✝⁴ : AddCommMonoid M
                      inst✝³ : Module R M
                      e : Basis ι R M
                      inst✝² : Finite ι
                      inst✝¹ : Fintype ι'
                      inst✝ : DecidableEq ι'
                      v : Basis ι' R M
                      i : ι'
                      ⊢ Eq (((Matrix.toLin v e) (e.toMatrix ⇑v)) (v i)) (LinearMap.id (v i))
                    -/
  v.ext fun i => by cases nonempty_fintype ι; rw [toLin_self, id_apply, e.sum_toMatrix_smul_self]
                                              /-
                                                🎉 no goals
                                              -/


/-- From a basis `e : ι → M`, build a linear equivalence between families of vectors `v : ι → M`,
and matrices, making the matrix whose columns are the vectors `v i` written in the basis `e`. -/
def toMatrixEquiv [Fintype ι] (e : Basis ι R M) : (ι → M) ≃ₗ[R] Matrix ι ι R where
  toFun := e.toMatrix
  map_add' v w := by
    /-
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v✝ : ι' → M
      i : ι
      j : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      v w : ι → M
      ⊢ Eq (e.toMatrix (HAdd.hAdd v w)) (HAdd.hAdd (e.toMatrix v) (e.toMatrix w))
    -/
    ext i j
    /-
      case a
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v✝ : ι' → M
      i✝ : ι
      j✝ : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      v w : ι → M
      i j : ι
      ⊢ Eq (e.toMatrix (HAdd.hAdd v w) i j) (HAdd.hAdd (e.toMatrix v) (e.toMatrix w) …
    -/
    rw [Matrix.add_apply, e.toMatrix_apply, Pi.add_apply, LinearEquiv.map_add]
    /-
      case a
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v✝ : ι' → M
      i✝ : ι
      j✝ : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      v w : ι → M
      i j : ι
      ⊢ Eq ((HAdd.hAdd (e.repr (v j)) (e.repr (w j))) i) (HAdd.hAdd (e.toMatrix v i  …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_smul' := by
    /-
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v : ι' → M
      i : ι
      j : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      ⊢ ∀ (m : R) (x : ι → M), Eq ({ toFun := e.toMatrix, map_add' := ⋯ }.toFun (HSM …
    -/
    intro c v
    /-
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v✝ : ι' → M
      i : ι
      j : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      c : R
      v : ι → M
      ⊢ Eq ({ toFun := e.toMatrix, map_add' := ⋯ }.toFun (HSMul.hSMul c v)) (HSMul.h …
    -/
    ext i j
    /-
      case a
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v✝ : ι' → M
      i✝ : ι
      j✝ : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      c : R
      v : ι → M
      i j : ι
      ⊢ Eq ({ toFun := e.toMatrix, map_add' := ⋯ }.toFun (HSMul.hSMul c v) i j) (HSM …
    -/
    dsimp only []
    /-
      case a
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v✝ : ι' → M
      i✝ : ι
      j✝ : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      c : R
      v : ι → M
      i j : ι
      ⊢ Eq (e.toMatrix (HSMul.hSMul c v) i j) (HSMul.hSMul ((RingHom.id R) c) (e.toM …
    -/
    rw [e.toMatrix_apply, Pi.smul_apply, LinearEquiv.map_smul]
    /-
      case a
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v✝ : ι' → M
      i✝ : ι
      j✝ : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      c : R
      v : ι → M
      i j : ι
      ⊢ Eq ((HSMul.hSMul c (e.repr (v j))) i) (HSMul.hSMul ((RingHom.id R) c) (e.toM …
    -/
    rfl
    /-
      🎉 no goals
    -/
  invFun m j := ∑ i, m i j • e i
  left_inv := by
    /-
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v : ι' → M
      i : ι
      j : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      ⊢ Function.LeftInverse (fun m j => Finset.univ.sum fun i => HSMul.hSMul (m i j …
    -/
    intro v
    /-
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v✝ : ι' → M
      i : ι
      j : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      ⊢ Eq ((fun m j => Finset.univ.sum fun i => HSMul.hSMul (m i j) (e i)) ({ toFun …
    -/
    ext j
    /-
      case h
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v✝ : ι' → M
      i : ι
      j✝ : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      j : ι
      ⊢ Eq ((fun m j => Finset.univ.sum fun i => HSMul.hSMul (m i j) (e i)) ({ toFun …
    -/
    exact e.sum_toMatrix_smul_self v j
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v : ι' → M
      i : ι
      j : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      ⊢ Function.RightInverse (fun m j => Finset.univ.sum fun i => HSMul.hSMul (m i  …
    -/
    intro m
    /-
      ι : Type u_1
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      R₂ : Type u_7
      M₂ : Type u_8
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      e✝ : Basis ι R M
      v : ι' → M
      i : ι
      j : ι'
      inst✝ : Fintype ι
      e : Basis ι R M
      m : Matrix ι ι R
      ⊢ Eq ({ toFun := e.toMatrix, map_add' := ⋯, map_smul' := ⋯ }.toFun ((fun m j = …
    -/
    ext k l
    simp only [e.toMatrix_apply, ← e.equivFun_apply, ← e.equivFun_symm_apply,
      LinearEquiv.apply_symm_apply]


variable (R₂) in
theorem restrictScalars_toMatrix [Fintype ι] [DecidableEq ι] {S : Type*} [CommRing S] [Nontrivial S]
    [Algebra R₂ S] [Module S M₂] [IsScalarTower R₂ S M₂] [NoZeroSMulDivisors R₂ S]
    (b : Basis ι S M₂) (v : ι → span R₂ (Set.range b)) :
    (algebraMap R₂ S).mapMatrix ((b.restrictScalars R₂).toMatrix v) =
      b.toMatrix (fun i ↦ (v i : M₂)) := by
  /-
    ι : Type u_1
    R₂ : Type u_7
    M₂ : Type u_8
    inst✝¹⁰ : CommRing R₂
    inst✝⁹ : AddCommGroup M₂
    inst✝⁸ : Module R₂ M₂
    inst✝⁷ : Fintype ι
    inst✝⁶ : DecidableEq ι
    S : Type u_9
    inst✝⁵ : CommRing S
    inst✝⁴ : Nontrivial S
    inst✝³ : Algebra R₂ S
    inst✝² : Module S M₂
    inst✝¹ : IsScalarTower R₂ S M₂
    inst✝ : NoZeroSMulDivisors R₂ S
    b : Basis ι S M₂
    v : ι → Subtype fun x => Membership.mem (Submodule.span R₂ (Set.range ⇑b)) x
    ⊢ Eq ((algebraMap R₂ S).mapMatrix ((Basis.restrictScalars R₂ b).toMatrix v)) ( …
  -/
  ext
  rw [RingHom.mapMatrix_apply, Matrix.map_apply, Basis.toMatrix_apply,
    Basis.restrictScalars_repr_apply, Basis.toMatrix_apply]


/-- A generalization of `LinearMap.toMatrix_id`. -/
@[simp]
theorem LinearMap.toMatrix_id_eq_basis_toMatrix [Fintype ι] [DecidableEq ι] [Finite ι'] :
    LinearMap.toMatrix b b' id = b'.toMatrix b := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    b : Basis ι R M
    b' : Basis ι' R M
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : Finite ι'
    ⊢ Eq ((LinearMap.toMatrix b b') LinearMap.id) (b'.toMatrix ⇑b)
  -/
  ext i
  /-
    case a
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    b : Basis ι R M
    b' : Basis ι' R M
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : Finite ι'
    i : ι'
    j✝ : ι
    ⊢ Eq ((LinearMap.toMatrix b b') LinearMap.id i j✝) (b'.toMatrix (⇑b) i j✝)
  -/
  apply LinearMap.toMatrix_apply
  /-
    🎉 no goals
  -/


@[simp]
theorem basis_toMatrix_mul_linearMap_toMatrix [Finite κ] [Fintype κ'] [DecidableEq ι'] :
    c.toMatrix c' * LinearMap.toMatrix b' c' f = LinearMap.toMatrix b' c f :=
  (Matrix.toLin b' c).injective <| by
    /-
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      N : Type u_9
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      b' : Basis ι' R M
      c : Basis κ R N
      c' : Basis κ' R N
      f : LinearMap (RingHom.id R) M N
      inst✝³ : Fintype ι'
      inst✝² : Finite κ
      inst✝¹ : Fintype κ'
      inst✝ : DecidableEq ι'
      ⊢ Eq ((Matrix.toLin b' c) (HMul.hMul (c.toMatrix ⇑c') ((LinearMap.toMatrix b'  …
    -/
    haveI := Classical.decEq κ'
    /-
      ι' : Type u_2
      κ : Type u_3
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      N : Type u_9
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      b' : Basis ι' R M
      c : Basis κ R N
      c' : Basis κ' R N
      f : LinearMap (RingHom.id R) M N
      inst✝³ : Fintype ι'
      inst✝² : Finite κ
      inst✝¹ : Fintype κ'
      inst✝ : DecidableEq ι'
      this : DecidableEq κ'
      ⊢ Eq ((Matrix.toLin b' c) (HMul.hMul (c.toMatrix ⇑c') ((LinearMap.toMatrix b'  …
    -/
    rw [toLin_toMatrix, toLin_mul b' c' c, toLin_toMatrix, c.toLin_toMatrix, LinearMap.id_comp]
    /-
      🎉 no goals
    -/


theorem basis_toMatrix_mul [Fintype κ] [Finite ι] [DecidableEq κ]
    (b₁ : Basis ι R M) (b₂ : Basis ι' R M) (b₃ : Basis κ R N) (A : Matrix ι' κ R) :
    b₁.toMatrix b₂ * A = LinearMap.toMatrix b₃ b₁ (toLin b₃ b₂ A) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    κ : Type u_3
    R : Type u_5
    M : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    N : Type u_9
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Fintype ι'
    inst✝² : Fintype κ
    inst✝¹ : Finite ι
    inst✝ : DecidableEq κ
    b₁ : Basis ι R M
    b₂ : Basis ι' R M
    b₃ : Basis κ R N
    A : Matrix ι' κ R
    ⊢ Eq (HMul.hMul (b₁.toMatrix ⇑b₂) A) ((LinearMap.toMatrix b₃ b₁) ((Matrix.toLi …
  -/
  have := basis_toMatrix_mul_linearMap_toMatrix b₃ b₁ b₂ (Matrix.toLin b₃ b₂ A)
  /-
    ι : Type u_1
    ι' : Type u_2
    κ : Type u_3
    R : Type u_5
    M : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    N : Type u_9
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Fintype ι'
    inst✝² : Fintype κ
    inst✝¹ : Finite ι
    inst✝ : DecidableEq κ
    b₁ : Basis ι R M
    b₂ : Basis ι' R M
    b₃ : Basis κ R N
    A : Matrix ι' κ R
    this : Eq (HMul.hMul (b₁.toMatrix ⇑b₂) ((LinearMap.toMatrix b₃ b₂) ((Matrix.to …
    ⊢ Eq (HMul.hMul (b₁.toMatrix ⇑b₂) A) ((LinearMap.toMatrix b₃ b₁) ((Matrix.toLi …
  -/
  rwa [LinearMap.toMatrix_toLin] at this
  /-
    🎉 no goals
  -/


@[simp]
theorem linearMap_toMatrix_mul_basis_toMatrix [Finite κ'] [DecidableEq ι] [DecidableEq ι'] :
    LinearMap.toMatrix b' c' f * b'.toMatrix b = LinearMap.toMatrix b c' f :=
  (Matrix.toLin b c').injective <| by
    /-
      ι : Type u_1
      ι' : Type u_2
      κ' : Type u_4
      R : Type u_5
      M : Type u_6
      inst✝⁹ : CommSemiring R
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : Module R M
      N : Type u_9
      inst✝⁶ : AddCommMonoid N
      inst✝⁵ : Module R N
      b : Basis ι R M
      b' : Basis ι' R M
      c' : Basis κ' R N
      f : LinearMap (RingHom.id R) M N
      inst✝⁴ : Fintype ι'
      inst✝³ : Fintype ι
      inst✝² : Finite κ'
      inst✝¹ : DecidableEq ι
      inst✝ : DecidableEq ι'
      ⊢ Eq ((Matrix.toLin b c') (HMul.hMul ((LinearMap.toMatrix b' c') f) (b'.toMatr …
    -/
    rw [toLin_toMatrix, toLin_mul b b' c', toLin_toMatrix, b'.toLin_toMatrix, LinearMap.comp_id]
    /-
      🎉 no goals
    -/


theorem basis_toMatrix_mul_linearMap_toMatrix_mul_basis_toMatrix
    [Fintype κ'] [DecidableEq ι] [DecidableEq ι'] :
    c.toMatrix c' * LinearMap.toMatrix b' c' f * b'.toMatrix b = LinearMap.toMatrix b c f := by
  /-
    ι : Type u_1
    ι' : Type u_2
    κ : Type u_3
    κ' : Type u_4
    R : Type u_5
    M : Type u_6
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    N : Type u_9
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    b : Basis ι R M
    b' : Basis ι' R M
    c : Basis κ R N
    c' : Basis κ' R N
    f : LinearMap (RingHom.id R) M N
    inst✝⁵ : Fintype ι'
    inst✝⁴ : Finite κ
    inst✝³ : Fintype ι
    inst✝² : Fintype κ'
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ι'
    ⊢ Eq (HMul.hMul (HMul.hMul (c.toMatrix ⇑c') ((LinearMap.toMatrix b' c') f)) (b …
  -/
  cases nonempty_fintype κ
  /-
    case intro
    ι : Type u_1
    ι' : Type u_2
    κ : Type u_3
    κ' : Type u_4
    R : Type u_5
    M : Type u_6
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    N : Type u_9
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    b : Basis ι R M
    b' : Basis ι' R M
    c : Basis κ R N
    c' : Basis κ' R N
    f : LinearMap (RingHom.id R) M N
    inst✝⁵ : Fintype ι'
    inst✝⁴ : Finite κ
    inst✝³ : Fintype ι
    inst✝² : Fintype κ'
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ι'
    val✝ : Fintype κ
    ⊢ Eq (HMul.hMul (HMul.hMul (c.toMatrix ⇑c') ((LinearMap.toMatrix b' c') f)) (b …
  -/
  rw [basis_toMatrix_mul_linearMap_toMatrix, linearMap_toMatrix_mul_basis_toMatrix]
  /-
    🎉 no goals
  -/


theorem mul_basis_toMatrix [DecidableEq ι] [DecidableEq ι'] (b₁ : Basis ι R M) (b₂ : Basis ι' R M)
    (b₃ : Basis κ R N) (A : Matrix κ ι R) :
    A * b₁.toMatrix b₂ = LinearMap.toMatrix b₂ b₃ (toLin b₁ b₃ A) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    κ : Type u_3
    R : Type u_5
    M : Type u_6
    inst✝⁹ : CommSemiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    N : Type u_9
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : Module R N
    inst✝⁴ : Fintype ι'
    inst✝³ : Finite κ
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ι'
    b₁ : Basis ι R M
    b₂ : Basis ι' R M
    b₃ : Basis κ R N
    A : Matrix κ ι R
    ⊢ Eq (HMul.hMul A (b₁.toMatrix ⇑b₂)) ((LinearMap.toMatrix b₂ b₃) ((Matrix.toLi …
  -/
  cases nonempty_fintype κ
  /-
    case intro
    ι : Type u_1
    ι' : Type u_2
    κ : Type u_3
    R : Type u_5
    M : Type u_6
    inst✝⁹ : CommSemiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    N : Type u_9
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : Module R N
    inst✝⁴ : Fintype ι'
    inst✝³ : Finite κ
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ι'
    b₁ : Basis ι R M
    b₂ : Basis ι' R M
    b₃ : Basis κ R N
    A : Matrix κ ι R
    val✝ : Fintype κ
    ⊢ Eq (HMul.hMul A (b₁.toMatrix ⇑b₂)) ((LinearMap.toMatrix b₂ b₃) ((Matrix.toLi …
  -/
  have := linearMap_toMatrix_mul_basis_toMatrix b₂ b₁ b₃ (Matrix.toLin b₁ b₃ A)
  /-
    case intro
    ι : Type u_1
    ι' : Type u_2
    κ : Type u_3
    R : Type u_5
    M : Type u_6
    inst✝⁹ : CommSemiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    N : Type u_9
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : Module R N
    inst✝⁴ : Fintype ι'
    inst✝³ : Finite κ
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ι'
    b₁ : Basis ι R M
    b₂ : Basis ι' R M
    b₃ : Basis κ R N
    A : Matrix κ ι R
    val✝ : Fintype κ
    this : Eq (HMul.hMul ((LinearMap.toMatrix b₁ b₃) ((Matrix.toLin b₁ b₃) A)) (b₁ …
    ⊢ Eq (HMul.hMul A (b₁.toMatrix ⇑b₂)) ((LinearMap.toMatrix b₂ b₃) ((Matrix.toLi …
  -/
  rwa [LinearMap.toMatrix_toLin] at this
  /-
    🎉 no goals
  -/


theorem basis_toMatrix_basisFun_mul (b : Basis ι R (ι → R)) (A : Matrix ι ι R) :
    b.toMatrix (Pi.basisFun R ι) * A = of fun i j => b.repr (Aᵀ j) i := by
  classical
  simp only [basis_toMatrix_mul _ _ (Pi.basisFun R ι), Matrix.toLin_eq_toLin']
  ext i j
  rw [LinearMap.toMatrix_apply, Matrix.toLin'_apply, Pi.basisFun_apply,
    Matrix.mulVec_single_one, Matrix.of_apply]


/-- See also `Basis.toMatrix_reindex` which gives the `simp` normal form of this result. -/
theorem Basis.toMatrix_reindex' [DecidableEq ι] [DecidableEq ι'] (b : Basis ι R M) (v : ι' → M)
    (e : ι ≃ ι') : (b.reindex e).toMatrix v =
    Matrix.reindexAlgEquiv R R e (b.toMatrix (v ∘ e)) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Fintype ι'
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ι'
    b : Basis ι R M
    v : ι' → M
    e : Equiv ι ι'
    ⊢ Eq ((b.reindex e).toMatrix v) ((Matrix.reindexAlgEquiv R R e) (b.toMatrix (F …
  -/
  ext
  simp only [Basis.toMatrix_apply, Basis.repr_reindex, Matrix.reindexAlgEquiv_apply,
    Matrix.reindex_apply, Matrix.submatrix_apply, Function.comp_apply, e.apply_symm_apply,
    Finsupp.mapDomain_equiv_apply]


@[simp]
lemma Basis.toMatrix_mulVec_repr (m : M) :
    b'.toMatrix b *ᵥ b.repr m = b'.repr m := by
  classical
  simp [← LinearMap.toMatrix_id_eq_basis_toMatrix, LinearMap.toMatrix_mulVec_repr]


/-- A generalization of `Basis.toMatrix_self`, in the opposite direction. -/
@[simp]
theorem Basis.toMatrix_mul_toMatrix {ι'' : Type*} [Fintype ι'] (b'' : ι'' → M) :
    b.toMatrix b' * b'.toMatrix b'' = b.toMatrix b'' := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    b' : Basis ι' R M
    ι'' : Type u_10
    inst✝ : Fintype ι'
    b'' : ι'' → M
    ⊢ Eq (HMul.hMul (b.toMatrix ⇑b') (b'.toMatrix b'')) (b.toMatrix b'')
  -/
  haveI := Classical.decEq ι
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    b' : Basis ι' R M
    ι'' : Type u_10
    inst✝ : Fintype ι'
    b'' : ι'' → M
    this : DecidableEq ι
    ⊢ Eq (HMul.hMul (b.toMatrix ⇑b') (b'.toMatrix b'')) (b.toMatrix b'')
  -/
  haveI := Classical.decEq ι'
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    b' : Basis ι' R M
    ι'' : Type u_10
    inst✝ : Fintype ι'
    b'' : ι'' → M
    this✝ : DecidableEq ι
    this : DecidableEq ι'
    ⊢ Eq (HMul.hMul (b.toMatrix ⇑b') (b'.toMatrix b'')) (b.toMatrix b'')
  -/
  haveI := Classical.decEq ι''
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    b' : Basis ι' R M
    ι'' : Type u_10
    inst✝ : Fintype ι'
    b'' : ι'' → M
    this✝¹ : DecidableEq ι
    this✝ : DecidableEq ι'
    this : DecidableEq ι''
    ⊢ Eq (HMul.hMul (b.toMatrix ⇑b') (b'.toMatrix b'')) (b.toMatrix b'')
  -/
  ext i j
  /-
    case a
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    b' : Basis ι' R M
    ι'' : Type u_10
    inst✝ : Fintype ι'
    b'' : ι'' → M
    this✝¹ : DecidableEq ι
    this✝ : DecidableEq ι'
    this : DecidableEq ι''
    i : ι
    j : ι''
    ⊢ Eq (HMul.hMul (b.toMatrix ⇑b') (b'.toMatrix b'') i j) (b.toMatrix b'' i j)
  -/
  simp only [Matrix.mul_apply, Basis.toMatrix_apply, Basis.sum_repr_mul_repr]
  /-
    🎉 no goals
  -/


/-- `b.toMatrix b'` and `b'.toMatrix b` are inverses. -/
theorem Basis.toMatrix_mul_toMatrix_flip [DecidableEq ι] [Fintype ι'] :
                                            /-
                                              ι : Type u_1
                                              ι' : Type u_2
                                              R : Type u_5
                                              M : Type u_6
                                              inst✝⁴ : CommSemiring R
                                              inst✝³ : AddCommMonoid M
                                              inst✝² : Module R M
                                              b : Basis ι R M
                                              b' : Basis ι' R M
                                              inst✝¹ : DecidableEq ι
                                              inst✝ : Fintype ι'
                                              ⊢ Eq (HMul.hMul (b.toMatrix ⇑b') (b'.toMatrix ⇑b)) 1
                                            -/
    b.toMatrix b' * b'.toMatrix b = 1 := by rw [Basis.toMatrix_mul_toMatrix, Basis.toMatrix_self]
                                            /-
                                              🎉 no goals
                                            -/


/-- A matrix whose columns form a basis `b'`, expressed w.r.t. a basis `b`, is invertible. -/
def Basis.invertibleToMatrix [DecidableEq ι] [Fintype ι] (b b' : Basis ι R₂ M₂) :
    Invertible (b.toMatrix b') :=
  ⟨b'.toMatrix b, Basis.toMatrix_mul_toMatrix_flip _ _, Basis.toMatrix_mul_toMatrix_flip _ _⟩


@[simp]
theorem Basis.toMatrix_reindex (b : Basis ι R M) (v : ι' → M) (e : ι ≃ ι') :
    (b.reindex e).toMatrix v = (b.toMatrix v).submatrix e.symm _root_.id := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_5
    M : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    v : ι' → M
    e : Equiv ι ι'
    ⊢ Eq ((b.reindex e).toMatrix v) ((b.toMatrix v).submatrix (⇑e.symm) id)
  -/
  ext
  simp only [Basis.toMatrix_apply, Basis.repr_reindex, Matrix.submatrix_apply, _root_.id,
    Finsupp.mapDomain_equiv_apply]


@[simp]
theorem Basis.toMatrix_map (b : Basis ι R M) (f : M ≃ₗ[R] N) (v : ι → N) :
    (b.map f).toMatrix v = b.toMatrix (f.symm ∘ v) := by
  /-
    ι : Type u_1
    R : Type u_5
    M : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_9
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    b : Basis ι R M
    f : LinearEquiv (RingHom.id R) M N
    v : ι → N
    ⊢ Eq ((b.map f).toMatrix v) (b.toMatrix (Function.comp (⇑f.symm) v))
  -/
  ext
  /-
    case a
    ι : Type u_1
    R : Type u_5
    M : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_9
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    b : Basis ι R M
    f : LinearEquiv (RingHom.id R) M N
    v : ι → N
    i✝ j✝ : ι
    ⊢ Eq ((b.map f).toMatrix v i✝ j✝) (b.toMatrix (Function.comp (⇑f.symm) v) i✝ j✝)
  -/
  simp only [Basis.toMatrix_apply, Basis.map, LinearEquiv.trans_apply, (· ∘ ·)]
  /-
    🎉 no goals
  -/


