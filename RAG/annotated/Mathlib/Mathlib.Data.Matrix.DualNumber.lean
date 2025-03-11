/-- Matrices over dual numbers and dual numbers over matrices are isomorphic. -/
@[simps]
def Matrix.dualNumberEquiv : Matrix n n (DualNumber R) ≃ₐ[R] DualNumber (Matrix n n R) where
  toFun A := ⟨of fun i j => (A i j).fst, of fun i j => (A i j).snd⟩
  invFun d := of fun i j => (d.fst i j, d.snd i j)
  left_inv _ := Matrix.ext fun _ _ => TrivSqZeroExt.ext rfl rfl
  right_inv _ := TrivSqZeroExt.ext (Matrix.ext fun _ _ => rfl) (Matrix.ext fun _ _ => rfl)
  map_mul' A B := by
    /-
      R n : Type
      inst✝² : CommSemiring R
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A B : Matrix n n (DualNumber R)
      ⊢ Eq ({ toFun := fun A => { fst := Matrix.of fun i j => TrivSqZeroExt.fst (A i …
    -/
    ext
      /-
        case h1.a
        R n : Type
        inst✝² : CommSemiring R
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        A B : Matrix n n (DualNumber R)
        i✝ j✝ : n
        ⊢ Eq (TrivSqZeroExt.fst ({ toFun := fun A => { fst := Matrix.of fun i j => Tri …
      -/
    · dsimp [mul_apply]
      /-
        case h1.a
        R n : Type
        inst✝² : CommSemiring R
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        A B : Matrix n n (DualNumber R)
        i✝ j✝ : n
        ⊢ Eq (TrivSqZeroExt.fst (Finset.univ.sum fun j => HMul.hMul (A i✝ j) (B j j✝)) …
      -/
      simp_rw [fst_sum]
      /-
        case h1.a
        R n : Type
        inst✝² : CommSemiring R
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        A B : Matrix n n (DualNumber R)
        i✝ j✝ : n
        ⊢ Eq (Finset.univ.sum fun i => TrivSqZeroExt.fst (HMul.hMul (A i✝ i) (B i j✝)) …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case h2.a
        R n : Type
        inst✝² : CommSemiring R
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        A B : Matrix n n (DualNumber R)
        i✝ j✝ : n
        ⊢ Eq (TrivSqZeroExt.snd ({ toFun := fun A => { fst := Matrix.of fun i j => Tri …
      -/
    · simp_rw [snd_mul, smul_eq_mul, op_smul_eq_mul]
      /-
        case h2.a
        R n : Type
        inst✝² : CommSemiring R
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        A B : Matrix n n (DualNumber R)
        i✝ j✝ : n
        ⊢ Eq (TrivSqZeroExt.snd { fst := Matrix.of fun i j => TrivSqZeroExt.fst (HMul. …
      -/
      simp only [mul_apply, snd_sum, DualNumber.snd_mul, snd_mk, of_apply, fst_mk, add_apply]
      /-
        case h2.a
        R n : Type
        inst✝² : CommSemiring R
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        A B : Matrix n n (DualNumber R)
        i✝ j✝ : n
        ⊢ Eq (Finset.univ.sum fun x => HAdd.hAdd (HMul.hMul (TrivSqZeroExt.fst (A i✝ x …
      -/
      rw [← Finset.sum_add_distrib]
      /-
        🎉 no goals
      -/
  map_add' _ _ := TrivSqZeroExt.ext rfl rfl
  commutes' r := by
    simp_rw [algebraMap_eq_inl', algebraMap_eq_diagonal, Pi.algebraMap_def,
      Algebra.id.map_eq_self, algebraMap_eq_inl, ← diagonal_map (inl_zero R), map_apply, fst_inl,
      snd_inl]
    /-
      R n : Type
      inst✝² : CommSemiring R
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      r : R
      ⊢ Eq { fst := Matrix.of fun i j => Matrix.diagonal (fun m => r) i j, snd := Ma …
    -/
    rfl
    /-
      🎉 no goals
    -/

