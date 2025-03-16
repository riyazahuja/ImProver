/-- The decomposition of `(1 : R)` where `1 = e₁ + e₂ + ⬝ ⬝ ⬝ + eₙ` which is induced by
  the decomposition of the ring `R = V1 ⊕ V2 ⊕ ⬝ ⬝ ⬝ ⊕ Vn`.-/
def idempotent (i : I) : R :=
  decompose V 1 i


lemma decompose_eq_mul_idempotent
    (x : R) (i : I) : decompose V x i = x * idempotent V i := by
  rw [← smul_eq_mul (a := x), idempotent, ← Submodule.coe_smul, ← smul_apply, ← decompose_smul,
    smul_eq_mul, mul_one]


lemma isIdempotentElem_idempotent (i : I) : IsIdempotentElem (idempotent V i : R) := by
  /-
    R : Type u_1
    I : Type u_2
    inst✝² : Ring R
    inst✝¹ : DecidableEq I
    V : I → Ideal R
    inst✝ : DirectSum.Decomposition V
    i : I
    ⊢ IsIdempotentElem (DirectSum.idempotent V i)
  -/
  rw [IsIdempotentElem, ← decompose_eq_mul_idempotent, idempotent, decompose_coe, of_eq_same]
  /-
    🎉 no goals
  -/


/-- If a ring can be decomposed into direct sum of finite left ideals `Vᵢ`
  where `1 = e₁ + ... + eₙ` and `eᵢ ∈ Vᵢ`, then `eᵢ` is a family of complete
  orthogonal idempotents.-/
theorem completeOrthogonalIdempotents_idempotent [Fintype I]:
    CompleteOrthogonalIdempotents fun i ↦ idempotent V i where
  idem := isIdempotentElem_idempotent V
  ortho i j hij := by
    /-
      R : Type u_1
      I : Type u_2
      inst✝³ : Ring R
      inst✝² : DecidableEq I
      V : I → Ideal R
      inst✝¹ : DirectSum.Decomposition V
      inst✝ : Fintype I
      i j : I
      hij : Ne i j
      ⊢ (fun x1 x2 => Eq (HMul.hMul (DirectSum.idempotent V x1) (DirectSum.idempoten …
    -/
    simp only
    rw [← decompose_eq_mul_idempotent, idempotent, decompose_coe,
      of_eq_of_ne (h := hij), Submodule.coe_zero]
  complete := by
    /-
      R : Type u_1
      I : Type u_2
      inst✝³ : Ring R
      inst✝² : DecidableEq I
      V : I → Ideal R
      inst✝¹ : DirectSum.Decomposition V
      inst✝ : Fintype I
      ⊢ Eq (Finset.univ.sum fun i => DirectSum.idempotent V i) 1
    -/
    apply (decompose V).injective
    /-
      case a
      R : Type u_1
      I : Type u_2
      inst✝³ : Ring R
      inst✝² : DecidableEq I
      V : I → Ideal R
      inst✝¹ : DirectSum.Decomposition V
      inst✝ : Fintype I
      ⊢ Eq ((DirectSum.decompose V) (Finset.univ.sum fun i => DirectSum.idempotent V …
    -/
    refine DFunLike.ext _ _ fun i ↦ ?_
    /-
      case a
      R : Type u_1
      I : Type u_2
      inst✝³ : Ring R
      inst✝² : DecidableEq I
      V : I → Ideal R
      inst✝¹ : DirectSum.Decomposition V
      inst✝ : Fintype I
      i : I
      ⊢ Eq (((DirectSum.decompose V) (Finset.univ.sum fun i => DirectSum.idempotent  …
    -/
    rw [decompose_sum, DFinsupp.finset_sum_apply]
    /-
      case a
      R : Type u_1
      I : Type u_2
      inst✝³ : Ring R
      inst✝² : DecidableEq I
      V : I → Ideal R
      inst✝¹ : DirectSum.Decomposition V
      inst✝ : Fintype I
      i : I
      ⊢ Eq (Finset.univ.sum fun a => ((DirectSum.decompose V) (DirectSum.idempotent  …
    -/
    simp [idempotent, of_apply]
    /-
      🎉 no goals
    -/


