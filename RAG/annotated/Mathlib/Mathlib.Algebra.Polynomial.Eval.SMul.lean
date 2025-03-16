@[simp]
theorem eval₂_smul (g : R →+* S) (p : R[X]) (x : S) {s : R} :
    eval₂ g x (s • p) = g s * eval₂ g x p := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    g : RingHom R S
    p : Polynomial R
    x : S
    s : R
    ⊢ Eq (Polynomial.eval₂ g x (HSMul.hSMul s p)) (HMul.hMul (g s) (Polynomial.eva …
  -/
  have A : p.natDegree < p.natDegree.succ := Nat.lt_succ_self _
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    g : RingHom R S
    p : Polynomial R
    x : S
    s : R
    A : LT.lt p.natDegree p.natDegree.succ
    ⊢ Eq (Polynomial.eval₂ g x (HSMul.hSMul s p)) (HMul.hMul (g s) (Polynomial.eva …
  -/
  have B : (s • p).natDegree < p.natDegree.succ := (natDegree_smul_le _ _).trans_lt A
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    g : RingHom R S
    p : Polynomial R
    x : S
    s : R
    A : LT.lt p.natDegree p.natDegree.succ
    B : LT.lt (HSMul.hSMul s p).natDegree p.natDegree.succ
    ⊢ Eq (Polynomial.eval₂ g x (HSMul.hSMul s p)) (HMul.hMul (g s) (Polynomial.eva …
  -/
  rw [eval₂_eq_sum, eval₂_eq_sum, sum_over_range' _ _ _ A, sum_over_range' _ _ _ B] <;>
    /-
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      g : RingHom R S
      p : Polynomial R
      x : S
      s : R
      A : LT.lt p.natDegree p.natDegree.succ
      B : LT.lt (HSMul.hSMul s p).natDegree p.natDegree.succ
      ⊢ Eq ((Finset.range p.natDegree.succ).sum fun a => HMul.hMul (g ((HSMul.hSMul  …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp [mul_sum, mul_assoc]
    /-
      🎉 no goals
    -/


@[simp]
theorem eval_smul [Monoid S] [DistribMulAction S R] [IsScalarTower S R R] (s : S) (p : R[X])
    (x : R) : (s • p).eval x = s • p.eval x := by
  /-
    R : Type u
    S : Type v
    inst✝³ : Semiring R
    inst✝² : Monoid S
    inst✝¹ : DistribMulAction S R
    inst✝ : IsScalarTower S R R
    s : S
    p : Polynomial R
    x : R
    ⊢ Eq (Polynomial.eval x (HSMul.hSMul s p)) (HSMul.hSMul s (Polynomial.eval x p))
  -/
  rw [← smul_one_smul R s p, eval, eval₂_smul, RingHom.id_apply, smul_one_mul]
  /-
    🎉 no goals
  -/


/-- `Polynomial.eval` as linear map -/
@[simps]
def leval {R : Type*} [Semiring R] (r : R) : R[X] →ₗ[R] R where
  toFun f := f.eval r
  map_add' _f _g := eval_add
  map_smul' c f := eval_smul c f r


@[simp]
theorem smul_comp [Monoid S] [DistribMulAction S R] [IsScalarTower S R R] (s : S) (p q : R[X]) :
    (s • p).comp q = s • p.comp q := by
  /-
    R : Type u
    S : Type v
    inst✝³ : Semiring R
    inst✝² : Monoid S
    inst✝¹ : DistribMulAction S R
    inst✝ : IsScalarTower S R R
    s : S
    p q : Polynomial R
    ⊢ Eq ((HSMul.hSMul s p).comp q) (HSMul.hSMul s (p.comp q))
  -/
  rw [← smul_one_smul R s p, comp, comp, eval₂_smul, ← smul_eq_C_mul, smul_assoc, one_smul]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem map_smul (r : R) : (r • p).map f = f r • p.map f := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    r : R
    ⊢ Eq (Polynomial.map f (HSMul.hSMul r p)) (HSMul.hSMul (f r) (Polynomial.map f …
  -/
  rw [map, eval₂_smul, RingHom.comp_apply, C_mul']
  /-
    🎉 no goals
  -/


