/-- A Hopf algebra over a commutative (semi)ring `R` is a bialgebra over `R` equipped with an
`R`-linear endomorphism `antipode` satisfying the antipode axioms. -/
class HopfAlgebra (R : Type u) (A : Type v) [CommSemiring R] [Semiring A] extends
    Bialgebra R A where
  /-- The antipode of the Hopf algebra. -/
  antipode : A →ₗ[R] A
  /-- One of the antipode axioms for a Hopf algebra. -/
  mul_antipode_rTensor_comul :
    LinearMap.mul' R A ∘ₗ antipode.rTensor A ∘ₗ comul = (Algebra.linearMap R A) ∘ₗ counit
  /-- One of the antipode axioms for a Hopf algebra. -/
  mul_antipode_lTensor_comul :
    LinearMap.mul' R A ∘ₗ antipode.lTensor A ∘ₗ comul = (Algebra.linearMap R A) ∘ₗ counit


@[simp]
theorem mul_antipode_rTensor_comul_apply (a : A) :
    LinearMap.mul' R A (antipode.rTensor A (Coalgebra.comul a)) =
    algebraMap R A (Coalgebra.counit a) :=
  LinearMap.congr_fun mul_antipode_rTensor_comul a


@[simp]
theorem mul_antipode_lTensor_comul_apply (a : A) :
    LinearMap.mul' R A (antipode.lTensor A (Coalgebra.comul a)) =
    algebraMap R A (Coalgebra.counit a) :=
  LinearMap.congr_fun mul_antipode_lTensor_comul a


@[simp]
lemma sum_antipode_mul_eq {a : A} (repr : Repr R a) :
    ∑ i ∈ repr.index, antipode (R := R) (repr.left i) * repr.right i =
      algebraMap R A (counit a) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : HopfAlgebra R A
    a : A
    repr : Coalgebra.Repr R a
    ⊢ Eq (repr.index.sum fun i => HMul.hMul (HopfAlgebra.antipode (repr.left i)) ( …
  -/
  simpa [← repr.eq, map_sum] using congr($(mul_antipode_rTensor_comul (R := R)) a)
  /-
    🎉 no goals
  -/


@[simp]
lemma sum_mul_antipode_eq {a : A} (repr : Repr R a) :
    ∑ i ∈ repr.index, repr.left i * antipode (R := R) (repr.right i) =
      algebraMap R A (counit a) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : HopfAlgebra R A
    a : A
    repr : Coalgebra.Repr R a
    ⊢ Eq (repr.index.sum fun i => HMul.hMul (repr.left i) (HopfAlgebra.antipode (r …
  -/
  simpa [← repr.eq, map_sum] using congr($(mul_antipode_lTensor_comul (R := R)) a)
  /-
    🎉 no goals
  -/


lemma sum_antipode_mul_eq_smul {a : A} (repr : Repr R a) :
    ∑ i ∈ repr.index, antipode (R := R) (repr.left i) * repr.right i =
      counit (R := R) a • 1 := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : HopfAlgebra R A
    a : A
    repr : Coalgebra.Repr R a
    ⊢ Eq (repr.index.sum fun i => HMul.hMul (HopfAlgebra.antipode (repr.left i)) ( …
  -/
  rw [sum_antipode_mul_eq, Algebra.smul_def, mul_one]
  /-
    🎉 no goals
  -/


lemma sum_mul_antipode_eq_smul {a : A} (repr : Repr R a) :
    ∑ i ∈ repr.index, repr.left i * antipode (R := R) (repr.right i) =
      counit (R := R) a • 1 := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : HopfAlgebra R A
    a : A
    repr : Coalgebra.Repr R a
    ⊢ Eq (repr.index.sum fun i => HMul.hMul (repr.left i) (HopfAlgebra.antipode (r …
  -/
  rw [sum_mul_antipode_eq, Algebra.smul_def, mul_one]
  /-
    🎉 no goals
  -/


/-- Every commutative (semi)ring is a Hopf algebra over itself -/
instance toHopfAlgebra : HopfAlgebra R R where
  antipode := .id
                                   /-
                                     R : Type u
                                     inst✝ : CommSemiring R
                                     ⊢ Eq ((LinearMap.mul' R R).comp ((LinearMap.rTensor R LinearMap.id).comp Coalg …
                                   -/
  mul_antipode_rTensor_comul := by ext; simp
                                        /-
                                          🎉 no goals
                                        -/
                                   /-
                                     R : Type u
                                     inst✝ : CommSemiring R
                                     ⊢ Eq ((LinearMap.mul' R R).comp ((LinearMap.lTensor R LinearMap.id).comp Coalg …
                                   -/
  mul_antipode_lTensor_comul := by ext; simp
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem antipode_eq_id : antipode (R := R) (A := R) = .id := rfl


