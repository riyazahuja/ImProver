/-- I by J matrix where each entry is a K by L matrix is equivalent to
    I × K by J × L matrix -/
@[simps]
def comp : Matrix I J (Matrix K L R) ≃ Matrix (I × K) (J × L) R where
  toFun m ik jl := m ik.1 jl.1 ik.2 jl.2
  invFun n i j k l := n (i, k) (j, l)
  left_inv _ := rfl
  right_inv _ := rfl


/-- `Matrix.comp` as `AddEquiv` -/
def compAddEquiv : Matrix I J (Matrix K L R) ≃+ Matrix (I × K) (J × L) R where
  __ := Matrix.comp I J K L R
  map_add' _ _ := rfl


@[simp]
theorem compAddEquiv_apply (M : Matrix I J (Matrix K L R)) :
    compAddEquiv I J K L R M = comp I J K L R M := rfl


@[simp]
theorem compAddEquiv_symm_apply (M : Matrix (I × K) (J × L) R) :
    (compAddEquiv I J K L R).symm M = (comp I J K L R).symm M := rfl


/-- `Matrix.comp` as `RingEquiv` -/
def compRingEquiv : Matrix I I (Matrix J J R) ≃+* Matrix (I × J) (I × J) R where
  __ := Matrix.compAddEquiv I I J J R
                     /-
                       I : Type u_1
                       J : Type u_2
                       K : Type u_3
                       L : Type u_4
                       R : Type u_5
                       inst✝² : Semiring R
                       inst✝¹ : Fintype I
                       inst✝ : Fintype J
                       x✝¹ x✝ : Matrix I I (Matrix J J R)
                       ⊢ Eq (__spread✝⁻⁰.toFun (HMul.hMul x✝¹ x✝)) (HMul.hMul (__spread✝⁻⁰.toFun x✝¹) …
                     -/
  map_mul' _ _ := by ext; exact (Matrix.sum_apply ..).trans <| .symm <| Fintype.sum_prod_type ..
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem compRingEquiv_apply (M : Matrix I I (Matrix J J R)) :
    compRingEquiv I J R M = comp I I J J R M := rfl


@[simp]
theorem compRingEquiv_symm_apply (M : Matrix (I × J) (I × J) R) :
    (compRingEquiv I J R).symm M = (comp I I J J R).symm M := rfl


/-- `Matrix.comp` as `LinearEquiv` -/
@[simps!]
def compLinearEquiv : Matrix I J (Matrix K L R) ≃ₗ[K] Matrix (I × K) (J × L) R where
  __ := Matrix.compAddEquiv I J K L R
  map_smul' _ _ := rfl


/-- `Matrix.comp` as `AlgEquiv` -/
@[simps!]
def compAlgEquiv : Matrix I I (Matrix J J R) ≃ₐ[K] Matrix (I × J) (I × J) R where
  __ := Matrix.compRingEquiv I J R
  commutes' c := by
    /-
      I : Type u_1
      J : Type u_2
      K✝ : Type u_3
      L : Type u_4
      R : Type u_5
      K : Type u_6
      inst✝⁶ : CommSemiring K
      inst✝⁵ : Semiring R
      inst✝⁴ : Fintype I
      inst✝³ : Fintype J
      inst✝² : Algebra K R
      inst✝¹ : DecidableEq I
      inst✝ : DecidableEq J
      c : K
      ⊢ Eq (__spread✝⁻⁰.toFun ((algebraMap K (Matrix I I (Matrix J J R))) c)) ((alge …
    -/
    ext _ _
    simp only [compRingEquiv, compAddEquiv, comp, AddEquiv.toEquiv_eq_coe, RingEquiv.toEquiv_eq_coe,
      Equiv.toFun_as_coe, EquivLike.coe_coe, RingEquiv.coe_mk, AddEquiv.coe_mk, Equiv.coe_fn_mk,
      algebraMap_eq_diagonal]
    rw [Pi.algebraMap_def, Pi.algebraMap_def, Algebra.algebraMap_eq_smul_one',
      Algebra.algebraMap_eq_smul_one', ← diagonal_one, diagonal_apply, diagonal_apply]
    /-
      case a
      I : Type u_1
      J : Type u_2
      K✝ : Type u_3
      L : Type u_4
      R : Type u_5
      K : Type u_6
      inst✝⁶ : CommSemiring K
      inst✝⁵ : Semiring R
      inst✝⁴ : Fintype I
      inst✝³ : Fintype J
      inst✝² : Algebra K R
      inst✝¹ : DecidableEq I
      inst✝ : DecidableEq J
      c : K
      i✝ j✝ : Prod I J
      ⊢ Eq (ite (Eq i✝.1 j✝.1) ((fun r => HSMul.hSMul r (Matrix.diagonal fun x => 1) …
    -/
    aesop
    /-
      🎉 no goals
    -/


