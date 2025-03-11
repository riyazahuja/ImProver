/-- Given a polynomial in `K[X]` such that all coefficients belong to the subring `R`,
  `Polynomial.int` is the corresponding polynomial in `R[X]`. -/
def Polynomial.int (P : K[X]) (hP : ∀ n : ℕ, P.coeff n ∈ R) : R[X] where
  toFinsupp :=
  { support := P.support
    toFun := fun n => ⟨P.coeff n, hP n⟩
    mem_support_toFun := fun n => by
      /-
        K : Type u_1
        inst✝ : Field K
        R : Subring K
        P : Polynomial K
        hP : ∀ (n : Nat), Membership.mem R (P.coeff n)
        n : Nat
        ⊢ Iff (Membership.mem P.support n) (Ne ((fun n => ⟨P.coeff n, ⋯⟩) n) 0)
      -/
      rw [ne_eq, ← Subring.coe_eq_zero_iff, mem_support_iff] }
      /-
        🎉 no goals
      -/


@[simp]
theorem int_coeff_eq  (n : ℕ) : ↑((P.int R hP).coeff n) = P.coeff n := rfl


@[simp]
theorem int_leadingCoeff_eq : ↑(P.int R hP).leadingCoeff = P.leadingCoeff := rfl


@[simp]
theorem int_monic_iff : (P.int R hP).Monic ↔ P.Monic := by
  /-
    K : Type u_1
    inst✝ : Field K
    R : Subring K
    P : Polynomial K
    hP : ∀ (n : Nat), Membership.mem R (P.coeff n)
    ⊢ Iff (Polynomial.int R P hP).Monic P.Monic
  -/
  rw [Monic, Monic, ← int_leadingCoeff_eq, OneMemClass.coe_eq_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem int_natDegree : (P.int R hP).natDegree = P.natDegree := rfl


@[simp]
theorem int_eval₂_eq (x : L) :
    eval₂ (algebraMap R L) x (P.int R hP) = aeval x P := by
  /-
    K : Type u_1
    inst✝² : Field K
    R : Subring K
    P : Polynomial K
    hP : ∀ (n : Nat), Membership.mem R (P.coeff n)
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    ⊢ Eq (Polynomial.eval₂ (algebraMap (Subtype fun x => Membership.mem R x) L) x  …
  -/
  rw [aeval_eq_sum_range, eval₂_eq_sum_range]
  /-
    K : Type u_1
    inst✝² : Field K
    R : Subring K
    P : Polynomial K
    hP : ∀ (n : Nat), Membership.mem R (P.coeff n)
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    ⊢ Eq ((Finset.range (HAdd.hAdd (Polynomial.int R P hP).natDegree 1)).sum fun i …
  -/
  exact Finset.sum_congr rfl (fun n _ => by rw [Algebra.smul_def]; rfl)
  /-
    🎉 no goals
  -/


