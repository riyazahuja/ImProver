/-- The map from the selfadjoint real subspace to the unitary group. This map only makes sense
over ℂ. -/
@[simps]
noncomputable def selfAdjoint.expUnitary (a : selfAdjoint A) : unitary A :=
  ⟨exp ℂ ((I • a.val) : A),
      exp_mem_unitary_of_mem_skewAdjoint _ (a.prop.smul_mem_skewAdjoint conj_I)⟩


theorem Commute.expUnitary_add {a b : selfAdjoint A} (h : Commute (a : A) (b : A)) :
    expUnitary (a + b) = expUnitary a * expUnitary b := by
  /-
    A : Type u_1
    inst✝⁵ : NormedRing A
    inst✝⁴ : NormedAlgebra Complex A
    inst✝³ : StarRing A
    inst✝² : ContinuousStar A
    inst✝¹ : CompleteSpace A
    inst✝ : StarModule Complex A
    a b : Subtype fun x => Membership.mem (selfAdjoint A) x
    h : Commute ↑a ↑b
    ⊢ Eq (selfAdjoint.expUnitary (HAdd.hAdd a b)) (HMul.hMul (selfAdjoint.expUnita …
  -/
  ext
  have hcomm : Commute (I • (a : A)) (I • (b : A)) := by
    unfold Commute SemiconjBy
    simp only [h.eq, Algebra.smul_mul_assoc, Algebra.mul_smul_comm]
  /-
    case a
    A : Type u_1
    inst✝⁵ : NormedRing A
    inst✝⁴ : NormedAlgebra Complex A
    inst✝³ : StarRing A
    inst✝² : ContinuousStar A
    inst✝¹ : CompleteSpace A
    inst✝ : StarModule Complex A
    a b : Subtype fun x => Membership.mem (selfAdjoint A) x
    h : Commute ↑a ↑b
    hcomm : Commute (HSMul.hSMul Complex.I ↑a) (HSMul.hSMul Complex.I ↑b)
    ⊢ Eq ↑(selfAdjoint.expUnitary (HAdd.hAdd a b)) ↑(HMul.hMul (selfAdjoint.expUni …
  -/
  simpa only [expUnitary_coe, AddSubgroup.coe_add, smul_add] using exp_add_of_commute hcomm
  /-
    🎉 no goals
  -/


theorem Commute.expUnitary {a b : selfAdjoint A} (h : Commute (a : A) (b : A)) :
    Commute (expUnitary a) (expUnitary b) :=
  calc
    selfAdjoint.expUnitary a * selfAdjoint.expUnitary b =
        selfAdjoint.expUnitary b * selfAdjoint.expUnitary a := by
      /-
        A : Type u_1
        inst✝⁵ : NormedRing A
        inst✝⁴ : NormedAlgebra Complex A
        inst✝³ : StarRing A
        inst✝² : ContinuousStar A
        inst✝¹ : CompleteSpace A
        inst✝ : StarModule Complex A
        a b : Subtype fun x => Membership.mem (selfAdjoint A) x
        h : Commute ↑a ↑b
        ⊢ Eq (HMul.hMul (selfAdjoint.expUnitary a) (selfAdjoint.expUnitary b)) (HMul.h …
      -/
      rw [← h.expUnitary_add, ← h.symm.expUnitary_add, add_comm]
      /-
        🎉 no goals
      -/


