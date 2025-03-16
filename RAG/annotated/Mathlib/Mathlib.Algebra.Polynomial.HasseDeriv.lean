/-- The `k`th Hasse derivative of a polynomial `∑ a_i X^i` is `∑ (i.choose k) a_i X^(i-k)`.
It satisfies `k! * (hasse_deriv k f) = derivative^[k] f`. -/
def hasseDeriv (k : ℕ) : R[X] →ₗ[R] R[X] :=
  lsum fun i => monomial (i - k) ∘ₗ DistribMulAction.toLinearMap R R (i.choose k)


theorem hasseDeriv_apply :
    hasseDeriv k f = f.sum fun i r => monomial (i - k) (↑(i.choose k) * r) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    f : Polynomial R
    ⊢ Eq ((Polynomial.hasseDeriv k) f) (f.sum fun i r => (Polynomial.monomial (HSu …
  -/
  dsimp [hasseDeriv]
  /-
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    f : Polynomial R
    ⊢ Eq (f.sum fun x1 x2 => (Polynomial.monomial (HSub.hSub x1 k)) (HSMul.hSMul ( …
  -/
  congr; ext; congr
  /-
    case e_f.h.h.a.e_a.h.e_6.h
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    f : Polynomial R
    x✝¹ : Nat
    x✝ : R
    n✝ : Nat
    ⊢ Eq (HSMul.hSMul (x✝¹.choose k) x✝) (HMul.hMul (↑(x✝¹.choose k)) x✝)
  -/
  apply nsmul_eq_mul
  /-
    🎉 no goals
  -/


theorem hasseDeriv_coeff (n : ℕ) :
    (hasseDeriv k f).coeff n = (n + k).choose k * f.coeff (n + k) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    f : Polynomial R
    n : Nat
    ⊢ Eq (((Polynomial.hasseDeriv k) f).coeff n) (HMul.hMul (↑((HAdd.hAdd n k).cho …
  -/
  rw [hasseDeriv_apply, coeff_sum, sum_def, Finset.sum_eq_single (n + k), coeff_monomial]
    /-
      R : Type u_1
      inst✝ : Semiring R
      k : Nat
      f : Polynomial R
      n : Nat
      ⊢ Eq (ite (Eq (HSub.hSub (HAdd.hAdd n k) k) n) (HMul.hMul (↑((HAdd.hAdd n k).c …
    -/
  · simp only [if_true, add_tsub_cancel_right, eq_self_iff_true]
    /-
      🎉 no goals
    -/
    /-
      case h₀
      R : Type u_1
      inst✝ : Semiring R
      k : Nat
      f : Polynomial R
      n : Nat
      ⊢ ∀ (b : Nat), Membership.mem f.support b → Ne b (HAdd.hAdd n k) → Eq (((Polyn …
    -/
  · intro i _hi hink
    /-
      case h₀
      R : Type u_1
      inst✝ : Semiring R
      k : Nat
      f : Polynomial R
      n i : Nat
      _hi : Membership.mem f.support i
      hink : Ne i (HAdd.hAdd n k)
      ⊢ Eq (((Polynomial.monomial (HSub.hSub i k)) (HMul.hMul (↑(i.choose k)) (f.coe …
    -/
    rw [coeff_monomial]
    /-
      case h₀
      R : Type u_1
      inst✝ : Semiring R
      k : Nat
      f : Polynomial R
      n i : Nat
      _hi : Membership.mem f.support i
      hink : Ne i (HAdd.hAdd n k)
      ⊢ Eq (ite (Eq (HSub.hSub i k) n) (HMul.hMul (↑(i.choose k)) (f.coeff i)) 0) 0
    -/
    by_cases hik : i < k
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        k : Nat
        f : Polynomial R
        n i : Nat
        _hi : Membership.mem f.support i
        hink : Ne i (HAdd.hAdd n k)
        hik : LT.lt i k
        ⊢ Eq (ite (Eq (HSub.hSub i k) n) (HMul.hMul (↑(i.choose k)) (f.coeff i)) 0) 0
      -/
    · simp only [Nat.choose_eq_zero_of_lt hik, ite_self, Nat.cast_zero, zero_mul]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        k : Nat
        f : Polynomial R
        n i : Nat
        _hi : Membership.mem f.support i
        hink : Ne i (HAdd.hAdd n k)
        hik : Not (LT.lt i k)
        ⊢ Eq (ite (Eq (HSub.hSub i k) n) (HMul.hMul (↑(i.choose k)) (f.coeff i)) 0) 0
      -/
    · push_neg at hik
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        k : Nat
        f : Polynomial R
        n i : Nat
        _hi : Membership.mem f.support i
        hink : Ne i (HAdd.hAdd n k)
        hik : LE.le k i
        ⊢ Eq (ite (Eq (HSub.hSub i k) n) (HMul.hMul (↑(i.choose k)) (f.coeff i)) 0) 0
      -/
      rw [if_neg]
      /-
        case neg.hnc
        R : Type u_1
        inst✝ : Semiring R
        k : Nat
        f : Polynomial R
        n i : Nat
        _hi : Membership.mem f.support i
        hink : Ne i (HAdd.hAdd n k)
        hik : LE.le k i
        ⊢ Not (Eq (HSub.hSub i k) n)
      -/
      contrapose! hink
      /-
        case neg.hnc
        R : Type u_1
        inst✝ : Semiring R
        k : Nat
        f : Polynomial R
        n i : Nat
        _hi : Membership.mem f.support i
        hik : LE.le k i
        hink : Eq (HSub.hSub i k) n
        ⊢ Eq i (HAdd.hAdd n k)
      -/
      exact (tsub_eq_iff_eq_add_of_le hik).mp hink
      /-
        🎉 no goals
      -/
    /-
      case h₁
      R : Type u_1
      inst✝ : Semiring R
      k : Nat
      f : Polynomial R
      n : Nat
      ⊢ Not (Membership.mem f.support (HAdd.hAdd n k)) → Eq (((Polynomial.monomial ( …
    -/
  · intro h
    /-
      case h₁
      R : Type u_1
      inst✝ : Semiring R
      k : Nat
      f : Polynomial R
      n : Nat
      h : Not (Membership.mem f.support (HAdd.hAdd n k))
      ⊢ Eq (((Polynomial.monomial (HSub.hSub (HAdd.hAdd n k) k)) (HMul.hMul (↑((HAdd …
    -/
    simp only [not_mem_support_iff.mp h, monomial_zero_right, mul_zero, coeff_zero]
    /-
      🎉 no goals
    -/


theorem hasseDeriv_zero' : hasseDeriv 0 f = f := by
  simp only [hasseDeriv_apply, tsub_zero, Nat.choose_zero_right, Nat.cast_one, one_mul,
    sum_monomial_eq]


@[simp]
theorem hasseDeriv_zero : @hasseDeriv R _ 0 = LinearMap.id :=
  LinearMap.ext <| hasseDeriv_zero'


theorem hasseDeriv_eq_zero_of_lt_natDegree (p : R[X]) (n : ℕ) (h : p.natDegree < n) :
    hasseDeriv n p = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : LT.lt p.natDegree n
    ⊢ Eq ((Polynomial.hasseDeriv n) p) 0
  -/
  rw [hasseDeriv_apply, sum_def]
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : LT.lt p.natDegree n
    ⊢ Eq (p.support.sum fun n_1 => (Polynomial.monomial (HSub.hSub n_1 n)) (HMul.h …
  -/
  refine Finset.sum_eq_zero fun x hx => ?_
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : LT.lt p.natDegree n
    x : Nat
    hx : Membership.mem p.support x
    ⊢ Eq ((Polynomial.monomial (HSub.hSub x n)) (HMul.hMul (↑(x.choose n)) (p.coef …
  -/
  simp [Nat.choose_eq_zero_of_lt ((le_natDegree_of_mem_supp _ hx).trans_lt h)]
  /-
    🎉 no goals
  -/


theorem hasseDeriv_one' : hasseDeriv 1 f = derivative f := by
  simp only [hasseDeriv_apply, derivative_apply, ← C_mul_X_pow_eq_monomial, Nat.choose_one_right,
    (Nat.cast_commute _ _).eq]


@[simp]
theorem hasseDeriv_one : @hasseDeriv R _ 1 = derivative :=
  LinearMap.ext <| hasseDeriv_one'


@[simp]
theorem hasseDeriv_monomial (n : ℕ) (r : R) :
    hasseDeriv k (monomial n r) = monomial (n - k) (↑(n.choose k) * r) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    k n : Nat
    r : R
    ⊢ Eq ((Polynomial.hasseDeriv k) ((Polynomial.monomial n) r)) ((Polynomial.mono …
  -/
  ext i
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    k n : Nat
    r : R
    i : Nat
    ⊢ Eq (((Polynomial.hasseDeriv k) ((Polynomial.monomial n) r)).coeff i) (((Poly …
  -/
  simp only [hasseDeriv_coeff, coeff_monomial]
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    k n : Nat
    r : R
    i : Nat
    ⊢ Eq (HMul.hMul (↑((HAdd.hAdd i k).choose k)) (ite (Eq n (HAdd.hAdd i k)) r 0) …
  -/
  by_cases hnik : n = i + k
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      k n : Nat
      r : R
      i : Nat
      hnik : Eq n (HAdd.hAdd i k)
      ⊢ Eq (HMul.hMul (↑((HAdd.hAdd i k).choose k)) (ite (Eq n (HAdd.hAdd i k)) r 0) …
    -/
  · rw [if_pos hnik, if_pos, ← hnik]
    /-
      case pos.hc
      R : Type u_1
      inst✝ : Semiring R
      k n : Nat
      r : R
      i : Nat
      hnik : Eq n (HAdd.hAdd i k)
      ⊢ Eq (HSub.hSub n k) i
    -/
    apply tsub_eq_of_eq_add_rev
    /-
      case pos.hc.h
      R : Type u_1
      inst✝ : Semiring R
      k n : Nat
      r : R
      i : Nat
      hnik : Eq n (HAdd.hAdd i k)
      ⊢ Eq n (HAdd.hAdd k i)
    -/
    rwa [add_comm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      k n : Nat
      r : R
      i : Nat
      hnik : Not (Eq n (HAdd.hAdd i k))
      ⊢ Eq (HMul.hMul (↑((HAdd.hAdd i k).choose k)) (ite (Eq n (HAdd.hAdd i k)) r 0) …
    -/
  · rw [if_neg hnik, mul_zero]
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      k n : Nat
      r : R
      i : Nat
      hnik : Not (Eq n (HAdd.hAdd i k))
      ⊢ Eq 0 (ite (Eq (HSub.hSub n k) i) (HMul.hMul (↑(n.choose k)) r) 0)
    -/
    by_cases hkn : k ≤ n
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        k n : Nat
        r : R
        i : Nat
        hnik : Not (Eq n (HAdd.hAdd i k))
        hkn : LE.le k n
        ⊢ Eq 0 (ite (Eq (HSub.hSub n k) i) (HMul.hMul (↑(n.choose k)) r) 0)
      -/
    · rw [← tsub_eq_iff_eq_add_of_le hkn] at hnik
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        k n : Nat
        r : R
        i : Nat
        hnik : Not (Eq (HSub.hSub n k) i)
        hkn : LE.le k n
        ⊢ Eq 0 (ite (Eq (HSub.hSub n k) i) (HMul.hMul (↑(n.choose k)) r) 0)
      -/
      rw [if_neg hnik]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        k n : Nat
        r : R
        i : Nat
        hnik : Not (Eq n (HAdd.hAdd i k))
        hkn : Not (LE.le k n)
        ⊢ Eq 0 (ite (Eq (HSub.hSub n k) i) (HMul.hMul (↑(n.choose k)) r) 0)
      -/
    · push_neg at hkn
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        k n : Nat
        r : R
        i : Nat
        hnik : Not (Eq n (HAdd.hAdd i k))
        hkn : LT.lt n k
        ⊢ Eq 0 (ite (Eq (HSub.hSub n k) i) (HMul.hMul (↑(n.choose k)) r) 0)
      -/
      rw [Nat.choose_eq_zero_of_lt hkn, Nat.cast_zero, zero_mul, ite_self]
      /-
        🎉 no goals
      -/


theorem hasseDeriv_C (r : R) (hk : 0 < k) : hasseDeriv k (C r) = 0 := by
  rw [← monomial_zero_left, hasseDeriv_monomial, Nat.choose_eq_zero_of_lt hk, Nat.cast_zero,
    zero_mul, monomial_zero_right]


theorem hasseDeriv_apply_one (hk : 0 < k) : hasseDeriv k (1 : R[X]) = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    hk : LT.lt 0 k
    ⊢ Eq ((Polynomial.hasseDeriv k) 1) 0
  -/
  rw [← C_1, hasseDeriv_C k _ hk]
  /-
    🎉 no goals
  -/


theorem hasseDeriv_X (hk : 1 < k) : hasseDeriv k (X : R[X]) = 0 := by
  rw [← monomial_one_one_eq_X, hasseDeriv_monomial, Nat.choose_eq_zero_of_lt hk, Nat.cast_zero,
    zero_mul, monomial_zero_right]


theorem factorial_smul_hasseDeriv : ⇑(k ! • @hasseDeriv R _ k) = (@derivative R _)^[k] := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    ⊢ Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑Pol …
  -/
  induction' k with k ih
    /-
      case zero
      R : Type u_1
      inst✝ : Semiring R
      k : Nat
      ⊢ Eq (⇑(HSMul.hSMul (Nat.factorial 0) (Polynomial.hasseDeriv 0))) (Nat.iterate …
    -/
  · rw [hasseDeriv_zero, factorial_zero, iterate_zero, one_smul, LinearMap.id_coe]
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    ⊢ Eq (⇑(HSMul.hSMul (HAdd.hAdd k 1).factorial (Polynomial.hasseDeriv (HAdd.hAd …
  -/
  ext f n : 2
  /-
    case succ.h.a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    ⊢ Eq (((HSMul.hSMul (HAdd.hAdd k 1).factorial (Polynomial.hasseDeriv (HAdd.hAd …
  -/
  rw [iterate_succ_apply', ← ih]
  simp only [LinearMap.smul_apply, coeff_smul, LinearMap.map_smul_of_tower, coeff_derivative,
    hasseDeriv_coeff, ← @choose_symm_add _ k]
  simp only [nsmul_eq_mul, factorial_succ, mul_assoc, succ_eq_add_one, ← add_assoc,
    add_right_comm n 1 k, ← cast_succ]
  /-
    case succ.h.a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    ⊢ Eq (HMul.hMul (↑(HMul.hMul (HAdd.hAdd k 1) k.factorial)) (HMul.hMul (↑((HAdd …
  -/
  rw [← (cast_commute (n + 1) (f.coeff (n + k + 1))).eq]
  /-
    case succ.h.a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    ⊢ Eq (HMul.hMul (↑(HMul.hMul (HAdd.hAdd k 1) k.factorial)) (HMul.hMul (↑((HAdd …
  -/
  simp only [← mul_assoc]
  /-
    case succ.h.a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul ↑(HMul.hMul (HAdd.hAdd k 1) k.factorial) ↑((HAdd.hA …
  -/
  norm_cast
  /-
    case succ.h.a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    ⊢ Eq (HMul.hMul (↑(HMul.hMul (HMul.hMul (HAdd.hAdd k 1) k.factorial) ((HAdd.hA …
  -/
  congr 2
  /-
    case succ.h.a.e_a.e_a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd k 1) k.factorial) ((HAdd.hAdd (HAdd.hAdd …
  -/
  rw [mul_comm (k+1) _, mul_assoc, mul_assoc]
  /-
    case succ.h.a.e_a.e_a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    ⊢ Eq (HMul.hMul k.factorial (HMul.hMul (HAdd.hAdd k 1) ((HAdd.hAdd (HAdd.hAdd  …
  -/
  congr 1
  /-
    case succ.h.a.e_a.e_a.e_a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    ⊢ Eq (HMul.hMul (HAdd.hAdd k 1) ((HAdd.hAdd (HAdd.hAdd n k) 1).choose (HAdd.hA …
  -/
  have : n + k + 1 = n + (k + 1) := by apply add_assoc
  /-
    case succ.h.a.e_a.e_a.e_a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    this : Eq (HAdd.hAdd (HAdd.hAdd n k) 1) (HAdd.hAdd n (HAdd.hAdd k 1))
    ⊢ Eq (HMul.hMul (HAdd.hAdd k 1) ((HAdd.hAdd (HAdd.hAdd n k) 1).choose (HAdd.hA …
  -/
  rw [← choose_symm_of_eq_add this, choose_succ_right_eq, mul_comm]
  /-
    case succ.h.a.e_a.e_a.e_a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    this : Eq (HAdd.hAdd (HAdd.hAdd n k) 1) (HAdd.hAdd n (HAdd.hAdd k 1))
    ⊢ Eq (HMul.hMul ((HAdd.hAdd (HAdd.hAdd n k) 1).choose n) (HAdd.hAdd k 1)) (HMu …
  -/
  congr
  /-
    case succ.h.a.e_a.e_a.e_a.e_a
    R : Type u_1
    inst✝ : Semiring R
    k✝ k : Nat
    ih : Eq (⇑(HSMul.hSMul k.factorial (Polynomial.hasseDeriv k))) (Nat.iterate (⇑ …
    f : Polynomial R
    n : Nat
    this : Eq (HAdd.hAdd (HAdd.hAdd n k) 1) (HAdd.hAdd n (HAdd.hAdd k 1))
    ⊢ Eq (HAdd.hAdd k 1) (HSub.hSub (HAdd.hAdd (HAdd.hAdd n k) 1) n)
  -/
  rw [add_assoc, add_tsub_cancel_left]
  /-
    🎉 no goals
  -/


theorem hasseDeriv_comp (k l : ℕ) :
    (@hasseDeriv R _ k).comp (hasseDeriv l) = (k + l).choose k • hasseDeriv (k + l) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    k l : Nat
    ⊢ Eq ((Polynomial.hasseDeriv k).comp (Polynomial.hasseDeriv l)) (HSMul.hSMul ( …
  -/
  ext i : 2
  simp only [LinearMap.smul_apply, comp_apply, LinearMap.coe_comp, smul_monomial, hasseDeriv_apply,
    mul_one, monomial_eq_zero_iff, sum_monomial_index, mul_zero, ←
    tsub_add_eq_tsub_tsub, add_comm l k]
  /-
    case h.h
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    ⊢ Eq ((Polynomial.monomial (HSub.hSub i (HAdd.hAdd k l))) (HMul.hMul ↑((HSub.h …
  -/
  rw_mod_cast [nsmul_eq_mul]
  /-
    case h.h
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    ⊢ Eq ((Polynomial.monomial (HSub.hSub i (HAdd.hAdd k l))) ↑(HMul.hMul ((HSub.h …
  -/
  rw [← Nat.cast_mul]
  /-
    case h.h
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    ⊢ Eq ((Polynomial.monomial (HSub.hSub i (HAdd.hAdd k l))) ↑(HMul.hMul ((HSub.h …
  -/
  congr 2
  /-
    case h.h.h.e_6.h.e_a
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    ⊢ Eq (HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) (HMul.hMul ((HAdd.hAd …
  -/
  by_cases hikl : i < k + l
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      k l i : Nat
      hikl : LT.lt i (HAdd.hAdd k l)
      ⊢ Eq (HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) (HMul.hMul ((HAdd.hAd …
    -/
  · rw [choose_eq_zero_of_lt hikl, mul_zero]
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      k l i : Nat
      hikl : LT.lt i (HAdd.hAdd k l)
      ⊢ Eq (HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) 0
    -/
    by_cases hil : i < l
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        k l i : Nat
        hikl : LT.lt i (HAdd.hAdd k l)
        hil : LT.lt i l
        ⊢ Eq (HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) 0
      -/
    · rw [choose_eq_zero_of_lt hil, mul_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        k l i : Nat
        hikl : LT.lt i (HAdd.hAdd k l)
        hil : Not (LT.lt i l)
        ⊢ Eq (HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) 0
      -/
    · push_neg at hil
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        k l i : Nat
        hikl : LT.lt i (HAdd.hAdd k l)
        hil : LE.le l i
        ⊢ Eq (HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) 0
      -/
      rw [← tsub_lt_iff_right hil] at hikl
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        k l i : Nat
        hikl : LT.lt (HSub.hSub i l) k
        hil : LE.le l i
        ⊢ Eq (HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) 0
      -/
      rw [choose_eq_zero_of_lt hikl, zero_mul]
      /-
        🎉 no goals
      -/
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    hikl : Not (LT.lt i (HAdd.hAdd k l))
    ⊢ Eq (HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) (HMul.hMul ((HAdd.hAd …
  -/
  push_neg at hikl
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    hikl : LE.le (HAdd.hAdd k l) i
    ⊢ Eq (HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) (HMul.hMul ((HAdd.hAd …
  -/
  apply @cast_injective ℚ
  /-
    case neg.a
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    hikl : LE.le (HAdd.hAdd k l) i
    ⊢ Eq ↑(HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) ↑(HMul.hMul ((HAdd.h …
  -/
  have h1 : l ≤ i := le_of_add_le_right hikl
  /-
    case neg.a
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    hikl : LE.le (HAdd.hAdd k l) i
    h1 : LE.le l i
    ⊢ Eq ↑(HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) ↑(HMul.hMul ((HAdd.h …
  -/
  have h2 : k ≤ i - l := le_tsub_of_add_le_right hikl
  /-
    case neg.a
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    hikl : LE.le (HAdd.hAdd k l) i
    h1 : LE.le l i
    h2 : LE.le k (HSub.hSub i l)
    ⊢ Eq ↑(HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) ↑(HMul.hMul ((HAdd.h …
  -/
  have h3 : k ≤ k + l := le_self_add
  /-
    case neg.a
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    hikl : LE.le (HAdd.hAdd k l) i
    h1 : LE.le l i
    h2 : LE.le k (HSub.hSub i l)
    h3 : LE.le k (HAdd.hAdd k l)
    ⊢ Eq ↑(HMul.hMul ((HSub.hSub i l).choose k) (i.choose l)) ↑(HMul.hMul ((HAdd.h …
  -/
  push_cast
  /-
    case neg.a
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    hikl : LE.le (HAdd.hAdd k l) i
    h1 : LE.le l i
    h2 : LE.le k (HSub.hSub i l)
    h3 : LE.le k (HAdd.hAdd k l)
    ⊢ Eq (HMul.hMul ↑((HSub.hSub i l).choose k) ↑(i.choose l)) (HMul.hMul ↑((HAdd. …
  -/
  rw [cast_choose ℚ h1, cast_choose ℚ h2, cast_choose ℚ h3, cast_choose ℚ hikl]
  /-
    case neg.a
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    hikl : LE.le (HAdd.hAdd k l) i
    h1 : LE.le l i
    h2 : LE.le k (HSub.hSub i l)
    h3 : LE.le k (HAdd.hAdd k l)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (↑(HSub.hSub i l).factorial) (HMul.hMul ↑k.factoria …
  -/
  rw [show i - (k + l) = i - l - k by rw [add_comm]; apply tsub_add_eq_tsub_tsub]
  /-
    case neg.a
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    hikl : LE.le (HAdd.hAdd k l) i
    h1 : LE.le l i
    h2 : LE.le k (HSub.hSub i l)
    h3 : LE.le k (HAdd.hAdd k l)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (↑(HSub.hSub i l).factorial) (HMul.hMul ↑k.factoria …
  -/
  simp only [add_tsub_cancel_left]
  /-
    case neg.a
    R : Type u_1
    inst✝ : Semiring R
    k l i : Nat
    hikl : LE.le (HAdd.hAdd k l) i
    h1 : LE.le l i
    h2 : LE.le k (HSub.hSub i l)
    h3 : LE.le k (HAdd.hAdd k l)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (↑(HSub.hSub i l).factorial) (HMul.hMul ↑k.factoria …
  -/
  field_simp; ring
              /-
                🎉 no goals
              -/


theorem natDegree_hasseDeriv_le (p : R[X]) (n : ℕ) :
    natDegree (hasseDeriv n p) ≤ natDegree p - n := by
  classical
    rw [hasseDeriv_apply, sum_def]
    refine (natDegree_sum_le _ _).trans ?_
    simp_rw [Function.comp, natDegree_monomial]
    rw [Finset.fold_ite, Finset.fold_const]
    · simp only [ite_self, max_eq_right, zero_le', Finset.fold_max_le, true_and, and_imp,
        tsub_le_iff_right, mem_support_iff, Ne, Finset.mem_filter]
      intro x hx hx'
      have hxp : x ≤ p.natDegree := le_natDegree_of_ne_zero hx
      have hxn : n ≤ x := by
        contrapose! hx'
        simp [Nat.choose_eq_zero_of_lt hx']
      rwa [tsub_add_cancel_of_le (hxn.trans hxp)]
    · simp


theorem natDegree_hasseDeriv [NoZeroSMulDivisors ℕ R] (p : R[X]) (n : ℕ) :
    natDegree (hasseDeriv n p) = natDegree p - n := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : NoZeroSMulDivisors Nat R
    p : Polynomial R
    n : Nat
    ⊢ Eq ((Polynomial.hasseDeriv n) p).natDegree (HSub.hSub p.natDegree n)
  -/
  cases' lt_or_le p.natDegree n with hn hn
    /-
      case inl
      R : Type u_1
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      p : Polynomial R
      n : Nat
      hn : LT.lt p.natDegree n
      ⊢ Eq ((Polynomial.hasseDeriv n) p).natDegree (HSub.hSub p.natDegree n)
    -/
  · simpa [hasseDeriv_eq_zero_of_lt_natDegree, hn] using (tsub_eq_zero_of_le hn.le).symm
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝¹ : Semiring R
      inst✝ : NoZeroSMulDivisors Nat R
      p : Polynomial R
      n : Nat
      hn : LE.le n p.natDegree
      ⊢ Eq ((Polynomial.hasseDeriv n) p).natDegree (HSub.hSub p.natDegree n)
    -/
  · refine map_natDegree_eq_sub ?_ ?_
      /-
        case inr.refine_1
        R : Type u_1
        inst✝¹ : Semiring R
        inst✝ : NoZeroSMulDivisors Nat R
        p : Polynomial R
        n : Nat
        hn : LE.le n p.natDegree
        ⊢ ∀ (f : Polynomial R), LT.lt f.natDegree n → Eq ((Polynomial.hasseDeriv n) f) 0
      -/
    · exact fun h => hasseDeriv_eq_zero_of_lt_natDegree _ _
      /-
        🎉 no goals
      -/
    · classical
        simp only [ite_eq_right_iff, Ne, natDegree_monomial, hasseDeriv_monomial]
        intro k c c0 hh
        -- this is where we use the `smul_eq_zero` from `NoZeroSMulDivisors`
        rw [← nsmul_eq_mul, smul_eq_zero, Nat.choose_eq_zero_iff] at hh
        exact (tsub_eq_zero_of_le (Or.resolve_right hh c0).le).symm


theorem hasseDeriv_mul (f g : R[X]) :
    hasseDeriv k (f * g) = ∑ ij ∈ antidiagonal k, hasseDeriv ij.1 f * hasseDeriv ij.2 g := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    f g : Polynomial R
    ⊢ Eq ((Polynomial.hasseDeriv k) (HMul.hMul f g)) ((Finset.HasAntidiagonal.anti …
  -/
  let D k := (@hasseDeriv R _ k).toAddMonoidHom
  /-
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    f g : Polynomial R
    D : Nat → AddMonoidHom (Polynomial R) (Polynomial R) := fun k => (Polynomial.h …
    ⊢ Eq ((Polynomial.hasseDeriv k) (HMul.hMul f g)) ((Finset.HasAntidiagonal.anti …
  -/
  let Φ := @AddMonoidHom.mul R[X] _
  show
    (compHom (D k)).comp Φ f g =
      ∑ ij ∈ antidiagonal k, ((compHom.comp ((compHom Φ) (D ij.1))).flip (D ij.2) f) g
  /-
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    f g : Polynomial R
    D : Nat → AddMonoidHom (Polynomial R) (Polynomial R) := fun k => (Polynomial.h …
    Φ : AddMonoidHom (Polynomial R) (AddMonoidHom (Polynomial R) (Polynomial R)) : …
    ⊢ Eq ((((AddMonoidHom.compHom (D k)).comp Φ) f) g) ((Finset.HasAntidiagonal.an …
  -/
  simp only [← finset_sum_apply]
  /-
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    f g : Polynomial R
    D : Nat → AddMonoidHom (Polynomial R) (Polynomial R) := fun k => (Polynomial.h …
    Φ : AddMonoidHom (Polynomial R) (AddMonoidHom (Polynomial R) (Polynomial R)) : …
    ⊢ Eq ((((AddMonoidHom.compHom (D k)).comp Φ) f) g) ((((Finset.HasAntidiagonal. …
  -/
  congr 2
  /-
    case e_a.e_a
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    f g : Polynomial R
    D : Nat → AddMonoidHom (Polynomial R) (Polynomial R) := fun k => (Polynomial.h …
    Φ : AddMonoidHom (Polynomial R) (AddMonoidHom (Polynomial R) (Polynomial R)) : …
    ⊢ Eq ((AddMonoidHom.compHom (D k)).comp Φ) ((Finset.HasAntidiagonal.antidiagon …
  -/
  clear f g
  /-
    case e_a.e_a
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    D : Nat → AddMonoidHom (Polynomial R) (Polynomial R) := fun k => (Polynomial.h …
    Φ : AddMonoidHom (Polynomial R) (AddMonoidHom (Polynomial R) (Polynomial R)) : …
    ⊢ Eq ((AddMonoidHom.compHom (D k)).comp Φ) ((Finset.HasAntidiagonal.antidiagon …
  -/
  ext m r n s : 4
  simp only [Φ, D, finset_sum_apply, coe_mulLeft, coe_comp, flip_apply, Function.comp_apply,
             hasseDeriv_monomial, LinearMap.toAddMonoidHom_coe, compHom_apply_apply,
             coe_mul, monomial_mul_monomial]
  have aux :
    ∀ x : ℕ × ℕ,
      x ∈ antidiagonal k →
        monomial (m - x.1 + (n - x.2)) (↑(m.choose x.1) * r * (↑(n.choose x.2) * s)) =
          monomial (m + n - k) (↑(m.choose x.1) * ↑(n.choose x.2) * (r * s)) := by
    intro x hx
    rw [mem_antidiagonal] at hx
    subst hx
    by_cases hm : m < x.1
    · simp only [Nat.choose_eq_zero_of_lt hm, Nat.cast_zero, zero_mul,
                 monomial_zero_right]
    by_cases hn : n < x.2
    · simp only [Nat.choose_eq_zero_of_lt hn, Nat.cast_zero, zero_mul,
                 mul_zero, monomial_zero_right]
    push_neg at hm hn
    rw [tsub_add_eq_add_tsub hm, ← add_tsub_assoc_of_le hn, ← tsub_add_eq_tsub_tsub,
      add_comm x.2 x.1, mul_assoc, ← mul_assoc r, ← (Nat.cast_commute _ r).eq, mul_assoc, mul_assoc]
  /-
    case e_a.e_a.h.h.h.h
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    D : Nat → AddMonoidHom (Polynomial R) (Polynomial R) := fun k => (Polynomial.h …
    Φ : AddMonoidHom (Polynomial R) (AddMonoidHom (Polynomial R) (Polynomial R)) : …
    m : Nat
    r : R
    n : Nat
    s : R
    aux : ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagona …
    ⊢ Eq ((Polynomial.monomial (HSub.hSub (HAdd.hAdd m n) k)) (HMul.hMul (↑((HAdd. …
  -/
  rw [Finset.sum_congr rfl aux]
  /-
    case e_a.e_a.h.h.h.h
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    D : Nat → AddMonoidHom (Polynomial R) (Polynomial R) := fun k => (Polynomial.h …
    Φ : AddMonoidHom (Polynomial R) (AddMonoidHom (Polynomial R) (Polynomial R)) : …
    m : Nat
    r : R
    n : Nat
    s : R
    aux : ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagona …
    ⊢ Eq ((Polynomial.monomial (HSub.hSub (HAdd.hAdd m n) k)) (HMul.hMul (↑((HAdd. …
  -/
  rw [← map_sum, ← Finset.sum_mul]
  /-
    case e_a.e_a.h.h.h.h
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    D : Nat → AddMonoidHom (Polynomial R) (Polynomial R) := fun k => (Polynomial.h …
    Φ : AddMonoidHom (Polynomial R) (AddMonoidHom (Polynomial R) (Polynomial R)) : …
    m : Nat
    r : R
    n : Nat
    s : R
    aux : ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagona …
    ⊢ Eq ((Polynomial.monomial (HSub.hSub (HAdd.hAdd m n) k)) (HMul.hMul (↑((HAdd. …
  -/
  congr
  /-
    case e_a.e_a.h.h.h.h.h.e_6.h.e_a
    R : Type u_1
    inst✝ : Semiring R
    k : Nat
    D : Nat → AddMonoidHom (Polynomial R) (Polynomial R) := fun k => (Polynomial.h …
    Φ : AddMonoidHom (Polynomial R) (AddMonoidHom (Polynomial R) (Polynomial R)) : …
    m : Nat
    r : R
    n : Nat
    s : R
    aux : ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagona …
    ⊢ Eq (↑((HAdd.hAdd m n).choose k)) ((Finset.HasAntidiagonal.antidiagonal k).su …
  -/
  rw_mod_cast [← Nat.add_choose_eq]
  /-
    🎉 no goals
  -/


