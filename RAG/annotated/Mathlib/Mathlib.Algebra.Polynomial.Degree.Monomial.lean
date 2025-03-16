lemma natDegree_le_pred (hf : p.natDegree ≤ n) (hn : p.coeff n = 0) : p.natDegree ≤ n - 1 := by
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    hf : LE.le p.natDegree n
    hn : Eq (p.coeff n) 0
    ⊢ LE.le p.natDegree (HSub.hSub n 1)
  -/
  obtain _ | n := n
    /-
      case zero
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hf : LE.le p.natDegree 0
      hn : Eq (p.coeff 0) 0
      ⊢ LE.le p.natDegree (HSub.hSub 0 1)
    -/
  · exact hf
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      hf : LE.le p.natDegree (HAdd.hAdd n 1)
      hn : Eq (p.coeff (HAdd.hAdd n 1)) 0
      ⊢ LE.le p.natDegree (HSub.hSub (HAdd.hAdd n 1) 1)
    -/
  · refine (Nat.le_succ_iff_eq_or_le.1 hf).resolve_left fun h ↦ ?_
    /-
      case succ
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      hf : LE.le p.natDegree (HAdd.hAdd n 1)
      hn : Eq (p.coeff (HAdd.hAdd n 1)) 0
      h : Eq p.natDegree n.succ
      ⊢ False
    -/
    rw [← Nat.succ_eq_add_one, ← h, coeff_natDegree, leadingCoeff_eq_zero] at hn
    /-
      case succ
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      hf : LE.le p.natDegree (HAdd.hAdd n 1)
      hn : Eq p 0
      h : Eq p.natDegree n.succ
      ⊢ False
    -/
    aesop
    /-
      🎉 no goals
    -/


theorem monomial_natDegree_leadingCoeff_eq_self (h : #p.support ≤ 1) :
    monomial p.natDegree p.leadingCoeff = p := by
  classical
  rcases card_support_le_one_iff_monomial.1 h with ⟨n, a, rfl⟩
  by_cases ha : a = 0 <;> simp [ha]


theorem C_mul_X_pow_eq_self (h : #p.support ≤ 1) : C p.leadingCoeff * X ^ p.natDegree = p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : LE.le p.support.card 1
    ⊢ Eq (HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Polynomial.X p.natDeg …
  -/
  rw [C_mul_X_pow_eq_monomial, monomial_natDegree_leadingCoeff_eq_self h]
  /-
    🎉 no goals
  -/


