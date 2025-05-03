/--
Sum of iterated derivatives of a polynomial, as a linear map

This definition does not allow different weights for the derivatives. It is likely that it could be
extended to allow them, but this was not needed for the initial use case (the integration by parts
of the integral $I_i$ in the
[Lindemann-Weierstrass](https://en.wikipedia.org/wiki/Lindemann%E2%80%93Weierstrass_theorem)
theorem).
-/
noncomputable def sumIDeriv : R[X] →ₗ[R] R[X] :=
  Finsupp.lsum ℕ (fun _ ↦ LinearMap.id) ∘ₗ derivativeFinsupp


theorem sumIDeriv_apply (p : R[X]) :
    sumIDeriv p = ∑ i ∈ range (p.natDegree + 1), derivative^[i] p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (Polynomial.sumIDeriv p) ((Finset.range (HAdd.hAdd p.natDegree 1)).sum fu …
  -/
  dsimp [sumIDeriv]
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq ((Polynomial.derivativeFinsupp p).sum fun i => id) ((Finset.range (HAdd.h …
  -/
  exact Finsupp.sum_of_support_subset _ (by simp) _ (by simp)
  /-
    🎉 no goals
  -/


theorem sumIDeriv_apply_of_lt {p : R[X]} {n : ℕ} (hn : p.natDegree < n) :
    sumIDeriv p = ∑ i ∈ range n, derivative^[i] p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    hn : LT.lt p.natDegree n
    ⊢ Eq (Polynomial.sumIDeriv p) ((Finset.range n).sum fun i => Nat.iterate (⇑Pol …
  -/
  dsimp [sumIDeriv]
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    hn : LT.lt p.natDegree n
    ⊢ Eq ((Polynomial.derivativeFinsupp p).sum fun i => id) ((Finset.range n).sum  …
  -/
  exact Finsupp.sum_of_support_subset _ (by simp [hn]) _ (by simp)
  /-
    🎉 no goals
  -/


theorem sumIDeriv_apply_of_le {p : R[X]} {n : ℕ} (hn : p.natDegree ≤ n) :
    sumIDeriv p = ∑ i ∈ range (n + 1), derivative^[i] p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    hn : LE.le p.natDegree n
    ⊢ Eq (Polynomial.sumIDeriv p) ((Finset.range (HAdd.hAdd n 1)).sum fun i => Nat …
  -/
  dsimp [sumIDeriv]
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    hn : LE.le p.natDegree n
    ⊢ Eq ((Polynomial.derivativeFinsupp p).sum fun i => id) ((Finset.range (HAdd.h …
  -/
  exact Finsupp.sum_of_support_subset _ (by simp [Nat.lt_succ, hn]) _ (by simp)
  /-
    🎉 no goals
  -/


@[simp]
theorem sumIDeriv_C (a : R) : sumIDeriv (C a) = C a := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    a : R
    ⊢ Eq (Polynomial.sumIDeriv (Polynomial.C a)) (Polynomial.C a)
  -/
  rw [sumIDeriv_apply, natDegree_C, zero_add, sum_range_one, Function.iterate_zero_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem sumIDeriv_X : sumIDeriv X = X + C 1 := by
  rw [sumIDeriv_apply, natDegree_X, sum_range_succ, sum_range_one, Function.iterate_zero_apply,
    Function.iterate_one, derivative_X, eq_natCast, Nat.cast_one]


@[simp]
theorem sumIDeriv_map (p : R[X]) (f : R →+* S) :
    sumIDeriv (p.map f) = (sumIDeriv p).map f := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    ⊢ Eq (Polynomial.sumIDeriv (Polynomial.map f p)) (Polynomial.map f (Polynomial …
  -/
  let n := max (p.map f).natDegree p.natDegree
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    n : Nat := Max.max (Polynomial.map f p).natDegree p.natDegree
    ⊢ Eq (Polynomial.sumIDeriv (Polynomial.map f p)) (Polynomial.map f (Polynomial …
  -/
  rw [sumIDeriv_apply_of_le (le_max_left _ _ : _ ≤ n)]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    n : Nat := Max.max (Polynomial.map f p).natDegree p.natDegree
    ⊢ Eq ((Finset.range (HAdd.hAdd (Max.max (Polynomial.map f p).natDegree p.natDe …
  -/
  rw [sumIDeriv_apply_of_le (le_max_right _ _ : _ ≤ n)]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    n : Nat := Max.max (Polynomial.map f p).natDegree p.natDegree
    ⊢ Eq ((Finset.range (HAdd.hAdd (Max.max (Polynomial.map f p).natDegree p.natDe …
  -/
  simp_rw [Polynomial.map_sum, iterate_derivative_map p f]
  /-
    🎉 no goals
  -/


theorem sumIDeriv_derivative (p : R[X]) : sumIDeriv (derivative p) = derivative (sumIDeriv p) := by
  rw [sumIDeriv_apply_of_le ((natDegree_derivative_le p).trans tsub_le_self), sumIDeriv_apply,
    derivative_sum]
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq ((Finset.range (HAdd.hAdd p.natDegree 1)).sum fun i => Nat.iterate (⇑Poly …
  -/
  simp_rw [← Function.iterate_succ_apply, Function.iterate_succ_apply']
  /-
    🎉 no goals
  -/


theorem sumIDeriv_eq_self_add (p : R[X]) : sumIDeriv p = p + derivative (sumIDeriv p) := by
  rw [sumIDeriv_apply, derivative_sum, sum_range_succ', sum_range_succ,
    add_comm, ← add_zero (Finset.sum _ _)]
  simp_rw [← Function.iterate_succ_apply' derivative, Nat.succ_eq_add_one,
    Function.iterate_zero_apply, iterate_derivative_eq_zero (Nat.lt_succ_self _)]


theorem exists_iterate_derivative_eq_factorial_smul (p : R[X]) (k : ℕ) :
    ∃ gp : R[X], gp.natDegree ≤ p.natDegree - k ∧ derivative^[k] p = k ! • gp := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    k : Nat
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree k)) (Eq (Nat …
  -/
  refine ⟨_, (natDegree_sum_le _ _).trans ?_, iterate_derivative_eq_factorial_smul_sum p k⟩
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    k : Nat
    ⊢ LE.le (Finset.fold Max.max 0 (Function.comp Polynomial.natDegree fun i => HM …
  -/
  rw [fold_max_le]
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    k : Nat
    ⊢ And (LE.le 0 (HSub.hSub p.natDegree k)) (∀ (x : Nat), Membership.mem (Nat.it …
  -/
  refine ⟨Nat.zero_le _, fun i hi => ?_⟩
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    k i : Nat
    hi : Membership.mem (Nat.iterate (⇑Polynomial.derivative) k p).support i
    ⊢ LE.le (Function.comp Polynomial.natDegree (fun i => HMul.hMul (Polynomial.C  …
  -/
  dsimp only [Function.comp]
  exact (natDegree_C_mul_le _ _).trans <| (natDegree_X_pow_le _).trans <|
    (le_natDegree_of_mem_supp _ hi).trans <| natDegree_iterate_derivative _ _


theorem aeval_iterate_derivative_of_lt (p : R[X]) (q : ℕ) (r : A) {p' : A[X]}
    (hp : p.map (algebraMap R A) = (X - C r) ^ q * p') {k : ℕ} (hk : k < q) :
    aeval r (derivative^[k] p) = 0 := by
  have h (x) : (X - C r) ^ (q - (k - x)) = (X - C r) ^ 1 * (X - C r) ^ (q - (k - x) - 1) := by
    rw [← pow_add, add_tsub_cancel_of_le]
    rw [Nat.lt_iff_add_one_le] at hk
    exact (le_tsub_of_add_le_left hk).trans (tsub_le_tsub_left (tsub_le_self : _ ≤ k) _)
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    k : Nat
    hk : LT.lt k q
    h : ∀ (x : Nat), Eq (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C r)) (HSub …
    ⊢ Eq ((Polynomial.aeval r) (Nat.iterate (⇑Polynomial.derivative) k p)) 0
  -/
  rw [aeval_def, eval₂_eq_eval_map, ← iterate_derivative_map]
  simp_rw [hp, iterate_derivative_mul, iterate_derivative_X_sub_pow, ← smul_mul_assoc, smul_smul,
    h, ← mul_smul_comm, mul_assoc, ← mul_sum, eval_mul, pow_one, eval_sub, eval_X, eval_C, sub_self,
    zero_mul]


theorem aeval_iterate_derivative_self (p : R[X]) (q : ℕ) (r : A) {p' : A[X]}
    (hp : p.map (algebraMap R A) = (X - C r) ^ q * p') :
    aeval r (derivative^[q] p) = q ! • p'.eval r := by
  have h (x) (h : 1 ≤ x) (h' : x ≤ q) :
      (X - C r) ^ (q - (q - x)) = (X - C r) ^ 1 * (X - C r) ^ (q - (q - x) - 1) := by
    rw [← pow_add, add_tsub_cancel_of_le]
    rwa [tsub_tsub_cancel_of_le h']
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    h : ∀ (x : Nat), LE.le 1 x → LE.le x q → Eq (HPow.hPow (HSub.hSub Polynomial.X …
    ⊢ Eq ((Polynomial.aeval r) (Nat.iterate (⇑Polynomial.derivative) q p)) (HSMul. …
  -/
  rw [aeval_def, eval₂_eq_eval_map, ← iterate_derivative_map]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    h : ∀ (x : Nat), LE.le 1 x → LE.le x q → Eq (HPow.hPow (HSub.hSub Polynomial.X …
    ⊢ Eq (Polynomial.eval r (Nat.iterate (⇑Polynomial.derivative) q (Polynomial.ma …
  -/
  simp_rw [hp, iterate_derivative_mul, iterate_derivative_X_sub_pow, ← smul_mul_assoc, smul_smul]
  rw [sum_range_succ', Nat.choose_zero_right, one_mul, tsub_zero, Nat.descFactorial_self, tsub_self,
    pow_zero, smul_mul_assoc, one_mul, Function.iterate_zero_apply, eval_add, eval_smul]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    h : ∀ (x : Nat), LE.le 1 x → LE.le x q → Eq (HPow.hPow (HSub.hSub Polynomial.X …
    ⊢ Eq (HAdd.hAdd (Polynomial.eval r ((Finset.range q).sum fun k => HMul.hMul (H …
  -/
  convert zero_add _
  /-
    case h.e'_2.h.e'_5
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    h : ∀ (x : Nat), LE.le 1 x → LE.le x q → Eq (HPow.hPow (HSub.hSub Polynomial.X …
    ⊢ Eq (Polynomial.eval r ((Finset.range q).sum fun k => HMul.hMul (HSMul.hSMul  …
  -/
  rw [eval_finset_sum]
  /-
    case h.e'_2.h.e'_5
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    h : ∀ (x : Nat), LE.le 1 x → LE.le x q → Eq (HPow.hPow (HSub.hSub Polynomial.X …
    ⊢ Eq ((Finset.range q).sum fun i => Polynomial.eval r (HMul.hMul (HSMul.hSMul  …
  -/
  apply sum_eq_zero
  /-
    case h.e'_2.h.e'_5.h
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    h : ∀ (x : Nat), LE.le 1 x → LE.le x q → Eq (HPow.hPow (HSub.hSub Polynomial.X …
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range q) x → Eq (Polynomial.eval r (HMul …
  -/
  intro x hx
  rw [h (x + 1) le_add_self (Nat.add_one_le_iff.mpr (mem_range.mp hx)), pow_one,
    eval_mul, eval_smul, eval_mul, eval_sub, eval_X, eval_C, sub_self, zero_mul,
    smul_zero, zero_mul]


theorem aeval_iterate_derivative_of_ge (p : R[X]) (q : ℕ) {k : ℕ} (hk : q ≤ k) :
    ∃ gp : R[X], gp.natDegree ≤ p.natDegree - k ∧
      ∀ r : A, aeval r (derivative^[k] p) = q ! • aeval r gp := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q k : Nat
    hk : LE.le q k
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree k)) (∀ (r :  …
  -/
  obtain ⟨p', p'_le, hp'⟩ := exists_iterate_derivative_eq_factorial_smul p k
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q k : Nat
    hk : LE.le q k
    p' : Polynomial R
    p'_le : LE.le p'.natDegree (HSub.hSub p.natDegree k)
    hp' : Eq (Nat.iterate (⇑Polynomial.derivative) k p) (HSMul.hSMul k.factorial p')
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree k)) (∀ (r :  …
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hk
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    p' : Polynomial R
    k : Nat
    hk : LE.le q (HAdd.hAdd q k)
    p'_le : LE.le p'.natDegree (HSub.hSub p.natDegree (HAdd.hAdd q k))
    hp' : Eq (Nat.iterate (⇑Polynomial.derivative) (HAdd.hAdd q k) p) (HSMul.hSMul …
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree (HAdd.hAdd q …
  -/
  refine ⟨((q + k).descFactorial k : R[X]) * p', (natDegree_C_mul_le _ _).trans p'_le, fun r => ?_⟩
  simp_rw [hp', nsmul_eq_mul, map_mul, map_natCast, ← mul_assoc, ← Nat.cast_mul,
    Nat.add_descFactorial_eq_ascFactorial, Nat.factorial_mul_ascFactorial]


theorem aeval_sumIDeriv_eq_eval (p : R[X]) (r : A) :
    aeval r (sumIDeriv p) = eval r (sumIDeriv (map (algebraMap R A) p)) := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    r : A
    ⊢ Eq ((Polynomial.aeval r) (Polynomial.sumIDeriv p)) (Polynomial.eval r (Polyn …
  -/
  rw [aeval_def, eval, sumIDeriv_map, eval₂_map, RingHom.id_comp]
  /-
    🎉 no goals
  -/


theorem aeval_sumIDeriv (p : R[X]) (q : ℕ) :
    ∃ gp : R[X], gp.natDegree ≤ p.natDegree - q ∧
      ∀ (r : A), (X - C r) ^ q ∣ p.map (algebraMap R A) →
        aeval r (sumIDeriv p) = q ! • aeval r gp := by
  have h (k) :
      ∃ gp : R[X], gp.natDegree ≤ p.natDegree - q ∧
        ∀ (r : A), (X - C r) ^ q ∣ p.map (algebraMap R A) →
          aeval r (derivative^[k] p) = q ! • aeval r gp := by
    cases lt_or_ge k q with
    | inl hk =>
      use 0
      rw [natDegree_zero]
      use Nat.zero_le _
      intro r ⟨p', hp⟩
      rw [map_zero, smul_zero, aeval_iterate_derivative_of_lt p q r hp hk]
    | inr hk =>
      obtain ⟨gp, gp_le, h⟩ := aeval_iterate_derivative_of_ge A p q hk
      exact ⟨gp, gp_le.trans (tsub_le_tsub_left hk _), fun r _ => h r⟩
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    h : ∀ (k : Nat), Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegr …
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree q)) (∀ (r :  …
  -/
  choose c h using h
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    c : Nat → Polynomial R
    h : ∀ (k : Nat), And (LE.le (c k).natDegree (HSub.hSub p.natDegree q)) (∀ (r : …
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree q)) (∀ (r :  …
  -/
  choose c_le hc using h
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    c : Nat → Polynomial R
    c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree q)
    hc : ∀ (k : Nat) (r : A), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomi …
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree q)) (∀ (r :  …
  -/
  refine ⟨(range (p.natDegree + 1)).sum c, ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      A : Type u_3
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      p : Polynomial R
      q : Nat
      c : Nat → Polynomial R
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree q)
      hc : ∀ (k : Nat) (r : A), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomi …
      ⊢ LE.le ((Finset.range (HAdd.hAdd p.natDegree 1)).sum c).natDegree (HSub.hSub  …
    -/
  · refine (natDegree_sum_le _ _).trans ?_
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      A : Type u_3
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      p : Polynomial R
      q : Nat
      c : Nat → Polynomial R
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree q)
      hc : ∀ (k : Nat) (r : A), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomi …
      ⊢ LE.le (Finset.fold Max.max 0 (Function.comp Polynomial.natDegree c) (Finset. …
    -/
    rw [fold_max_le]
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      A : Type u_3
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      p : Polynomial R
      q : Nat
      c : Nat → Polynomial R
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree q)
      hc : ∀ (k : Nat) (r : A), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomi …
      ⊢ And (LE.le 0 (HSub.hSub p.natDegree q)) (∀ (x : Nat), Membership.mem (Finset …
    -/
    exact ⟨Nat.zero_le _, fun i _ => c_le i⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    c : Nat → Polynomial R
    c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree q)
    hc : ∀ (k : Nat) (r : A), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomi …
    ⊢ ∀ (r : A), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C r)) q) ( …
  -/
  intro r ⟨p', hp⟩
  /-
    case refine_2
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : Polynomial R
    q : Nat
    c : Nat → Polynomial R
    c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree q)
    hc : ∀ (k : Nat) (r : A), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomi …
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    ⊢ Eq ((Polynomial.aeval r) (Polynomial.sumIDeriv p)) (HSMul.hSMul q.factorial  …
  -/
  rw [sumIDeriv_apply, map_sum]; simp_rw [hc _ r ⟨_, hp⟩, map_sum, smul_sum]
                                 /-
                                   🎉 no goals
                                 -/


theorem aeval_sumIDeriv_of_pos [Nontrivial A] [NoZeroDivisors A] (p : R[X]) {q : ℕ} (hq : 0 < q)
    (inj_amap : Function.Injective (algebraMap R A)) :
    ∃ gp : R[X], gp.natDegree ≤ p.natDegree - q ∧
      ∀ (r : A) {p' : A[X]},
        p.map (algebraMap R A) = (X - C r) ^ (q - 1) * p' →
        aeval r (sumIDeriv p) = (q - 1)! • p'.eval r + q ! • aeval r gp := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Nontrivial A
    inst✝ : NoZeroDivisors A
    p : Polynomial R
    q : Nat
    hq : LT.lt 0 q
    inj_amap : Function.Injective ⇑(algebraMap R A)
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree q)) (∀ (r :  …
  -/
  rcases eq_or_ne p 0 with (rfl | p0)
    /-
      case inl
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub (Polynomial.natDegree 0) …
    -/
  · use 0
    /-
      case h
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      ⊢ And (LE.le (Polynomial.natDegree 0) (HSub.hSub (Polynomial.natDegree 0) q))  …
    -/
    rw [natDegree_zero]
    /-
      case h
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      ⊢ And (LE.le 0 (HSub.hSub 0 q)) (∀ (r : A) {p' : Polynomial A}, Eq (Polynomial …
    -/
    use Nat.zero_le _
    /-
      case right
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      ⊢ ∀ (r : A) {p' : Polynomial A}, Eq (Polynomial.map (algebraMap R A) 0) (HMul. …
    -/
    intro r p' hp
    /-
      case right
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      r : A
      p' : Polynomial A
      hp : Eq (Polynomial.map (algebraMap R A) 0) (HMul.hMul (HPow.hPow (HSub.hSub P …
      ⊢ Eq ((Polynomial.aeval r) (Polynomial.sumIDeriv 0)) (HAdd.hAdd (HSMul.hSMul ( …
    -/
    rw [map_zero, map_zero, smul_zero, add_zero]
    /-
      case right
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      r : A
      p' : Polynomial A
      hp : Eq (Polynomial.map (algebraMap R A) 0) (HMul.hMul (HPow.hPow (HSub.hSub P …
      ⊢ Eq 0 (HSMul.hSMul (HSub.hSub q 1).factorial (Polynomial.eval r p'))
    -/
    rw [Polynomial.map_zero] at hp
    /-
      case right
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      r : A
      p' : Polynomial A
      hp : Eq 0 (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C r)) (HSu …
      ⊢ Eq 0 (HSMul.hSMul (HSub.hSub q 1).factorial (Polynomial.eval r p'))
    -/
    replace hp := (mul_eq_zero.mp hp.symm).resolve_left ?_
      /-
        case right.refine_2
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial A
        inst✝ : NoZeroDivisors A
        q : Nat
        hq : LT.lt 0 q
        inj_amap : Function.Injective ⇑(algebraMap R A)
        r : A
        p' : Polynomial A
        hp : Eq p' 0
        ⊢ Eq 0 (HSMul.hSMul (HSub.hSub q 1).factorial (Polynomial.eval r p'))
      -/
    · rw [hp, eval_zero, smul_zero]
      /-
        🎉 no goals
      -/
    /-
      case right.refine_1
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      r : A
      p' : Polynomial A
      hp : Eq 0 (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C r)) (HSu …
      ⊢ Not (Eq (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C r)) (HSub.hSub q 1) …
    -/
    exact fun h => X_sub_C_ne_zero r (pow_eq_zero h)
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Nontrivial A
    inst✝ : NoZeroDivisors A
    p : Polynomial R
    q : Nat
    hq : LT.lt 0 q
    inj_amap : Function.Injective ⇑(algebraMap R A)
    p0 : Ne p 0
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree q)) (∀ (r :  …
  -/
  let c k := if hk : q ≤ k then (aeval_iterate_derivative_of_ge A p q hk).choose else 0
  have c_le (k) : (c k).natDegree ≤ p.natDegree - k := by
    dsimp only [c]
    split_ifs with h
    · exact (aeval_iterate_derivative_of_ge A p q h).choose_spec.1
    · rw [natDegree_zero]; exact Nat.zero_le _
  have hc (k) (hk : q ≤ k) : ∀ (r : A), aeval r (derivative^[k] p) = q ! • aeval r (c k) := by
    simp_rw [c, dif_pos hk]
    exact (aeval_iterate_derivative_of_ge A p q hk).choose_spec.2
  /-
    case inr
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Nontrivial A
    inst✝ : NoZeroDivisors A
    p : Polynomial R
    q : Nat
    hq : LT.lt 0 q
    inj_amap : Function.Injective ⇑(algebraMap R A)
    p0 : Ne p 0
    c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
    c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
    hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree q)) (∀ (r :  …
  -/
  refine ⟨∑ x ∈ Ico q (p.natDegree + 1), c x, ?_, ?_⟩
    /-
      case inr.refine_1
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      p : Polynomial R
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      p0 : Ne p 0
      c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
      hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
      ⊢ LE.le ((Finset.Ico q (HAdd.hAdd p.natDegree 1)).sum fun x => c x).natDegree  …
    -/
  · refine (natDegree_sum_le _ _).trans ?_
    /-
      case inr.refine_1
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      p : Polynomial R
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      p0 : Ne p 0
      c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
      hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
      ⊢ LE.le (Finset.fold Max.max 0 (Function.comp Polynomial.natDegree c) (Finset. …
    -/
    rw [fold_max_le]
    /-
      case inr.refine_1
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      p : Polynomial R
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      p0 : Ne p 0
      c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
      hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
      ⊢ And (LE.le 0 (HSub.hSub p.natDegree q)) (∀ (x : Nat), Membership.mem (Finset …
    -/
    exact ⟨Nat.zero_le _, fun i hi => (c_le i).trans (tsub_le_tsub_left (mem_Ico.mp hi).1 _)⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.refine_2
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Nontrivial A
    inst✝ : NoZeroDivisors A
    p : Polynomial R
    q : Nat
    hq : LT.lt 0 q
    inj_amap : Function.Injective ⇑(algebraMap R A)
    p0 : Ne p 0
    c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
    c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
    hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
    ⊢ ∀ (r : A) {p' : Polynomial A}, Eq (Polynomial.map (algebraMap R A) p) (HMul. …
  -/
  intro r p' hp
  have : range (p.natDegree + 1) = range q ∪ Ico q (p.natDegree + 1) := by
    rw [range_eq_Ico, Ico_union_Ico_eq_Ico hq.le]
    rw [← tsub_le_iff_right]
    calc
      q - 1 ≤ q - 1 + p'.natDegree := le_self_add
      _ = (p.map <| algebraMap R A).natDegree := by
        rw [hp, natDegree_mul, natDegree_pow, natDegree_X_sub_C, mul_one,
          ← Nat.sub_add_comm (Nat.one_le_of_lt hq)]
        · exact pow_ne_zero _ (X_sub_C_ne_zero r)
        · rintro rfl
          rw [mul_zero, Polynomial.map_eq_zero_iff inj_amap] at hp
          exact p0 hp
      _ ≤ p.natDegree := natDegree_map_le
  /-
    case inr.refine_2
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Nontrivial A
    inst✝ : NoZeroDivisors A
    p : Polynomial R
    q : Nat
    hq : LT.lt 0 q
    inj_amap : Function.Injective ⇑(algebraMap R A)
    p0 : Ne p 0
    c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
    c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
    hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    this : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range  …
    ⊢ Eq ((Polynomial.aeval r) (Polynomial.sumIDeriv p)) (HAdd.hAdd (HSMul.hSMul ( …
  -/
  rw [← zero_add ((q - 1)! • p'.eval r)]
  /-
    case inr.refine_2
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Nontrivial A
    inst✝ : NoZeroDivisors A
    p : Polynomial R
    q : Nat
    hq : LT.lt 0 q
    inj_amap : Function.Injective ⇑(algebraMap R A)
    p0 : Ne p 0
    c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
    c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
    hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    this : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range  …
    ⊢ Eq ((Polynomial.aeval r) (Polynomial.sumIDeriv p)) (HAdd.hAdd (HAdd.hAdd 0 ( …
  -/
  rw [sumIDeriv_apply, map_sum, map_sum, this]
  /-
    case inr.refine_2
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Nontrivial A
    inst✝ : NoZeroDivisors A
    p : Polynomial R
    q : Nat
    hq : LT.lt 0 q
    inj_amap : Function.Injective ⇑(algebraMap R A)
    p0 : Ne p 0
    c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
    c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
    hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    this : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range  …
    ⊢ Eq ((Union.union (Finset.range q) (Finset.Ico q (HAdd.hAdd p.natDegree 1))). …
  -/
  have : range q = range (q - 1 + 1) := by rw [tsub_add_cancel_of_le (Nat.one_le_of_lt hq)]
  /-
    case inr.refine_2
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Nontrivial A
    inst✝ : NoZeroDivisors A
    p : Polynomial R
    q : Nat
    hq : LT.lt 0 q
    inj_amap : Function.Injective ⇑(algebraMap R A)
    p0 : Ne p 0
    c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
    c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
    hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
    r : A
    p' : Polynomial A
    hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
    this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
    this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
    ⊢ Eq ((Union.union (Finset.range q) (Finset.Ico q (HAdd.hAdd p.natDegree 1))). …
  -/
  rw [sum_union, this, sum_range_succ]
    /-
      case inr.refine_2
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      p : Polynomial R
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      p0 : Ne p 0
      c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
      hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
      r : A
      p' : Polynomial A
      hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
      this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
      this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((Finset.range (HSub.hSub q 1)).sum fun x => (Polyn …
    -/
  · congr 2
      /-
        case inr.refine_2.e_a.e_a
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial A
        inst✝ : NoZeroDivisors A
        p : Polynomial R
        q : Nat
        hq : LT.lt 0 q
        inj_amap : Function.Injective ⇑(algebraMap R A)
        p0 : Ne p 0
        c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
        c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
        hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
        r : A
        p' : Polynomial A
        hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
        this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
        this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
        ⊢ Eq ((Finset.range (HSub.hSub q 1)).sum fun x => (Polynomial.aeval r) (Nat.it …
      -/
    · apply sum_eq_zero
      /-
        case inr.refine_2.e_a.e_a.h
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial A
        inst✝ : NoZeroDivisors A
        p : Polynomial R
        q : Nat
        hq : LT.lt 0 q
        inj_amap : Function.Injective ⇑(algebraMap R A)
        p0 : Ne p 0
        c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
        c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
        hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
        r : A
        p' : Polynomial A
        hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
        this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
        this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
        ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HSub.hSub q 1)) x → Eq ((Polynomi …
      -/
      exact fun x hx => aeval_iterate_derivative_of_lt p _ r hp (mem_range.mp hx)
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_2.e_a.e_a
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial A
        inst✝ : NoZeroDivisors A
        p : Polynomial R
        q : Nat
        hq : LT.lt 0 q
        inj_amap : Function.Injective ⇑(algebraMap R A)
        p0 : Ne p 0
        c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
        c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
        hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
        r : A
        p' : Polynomial A
        hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
        this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
        this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
        ⊢ Eq ((Polynomial.aeval r) (Nat.iterate (⇑Polynomial.derivative) (HSub.hSub q  …
      -/
    · rw [← aeval_iterate_derivative_self _ _ _ hp]
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_2.e_a
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial A
        inst✝ : NoZeroDivisors A
        p : Polynomial R
        q : Nat
        hq : LT.lt 0 q
        inj_amap : Function.Injective ⇑(algebraMap R A)
        p0 : Ne p 0
        c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
        c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
        hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
        r : A
        p' : Polynomial A
        hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
        this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
        this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
        ⊢ Eq ((Finset.Ico q (HAdd.hAdd p.natDegree 1)).sum fun x => (Polynomial.aeval  …
      -/
    · rw [smul_sum, sum_congr rfl]
      /-
        case inr.refine_2.e_a
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial A
        inst✝ : NoZeroDivisors A
        p : Polynomial R
        q : Nat
        hq : LT.lt 0 q
        inj_amap : Function.Injective ⇑(algebraMap R A)
        p0 : Ne p 0
        c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
        c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
        hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
        r : A
        p' : Polynomial A
        hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
        this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
        this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
        ⊢ ∀ (x : Nat), Membership.mem (Finset.Ico q (HAdd.hAdd p.natDegree 1)) x → Eq  …
      -/
      intro k hk
      /-
        case inr.refine_2.e_a
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial A
        inst✝ : NoZeroDivisors A
        p : Polynomial R
        q : Nat
        hq : LT.lt 0 q
        inj_amap : Function.Injective ⇑(algebraMap R A)
        p0 : Ne p 0
        c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
        c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
        hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
        r : A
        p' : Polynomial A
        hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
        this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
        this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
        k : Nat
        hk : Membership.mem (Finset.Ico q (HAdd.hAdd p.natDegree 1)) k
        ⊢ Eq ((Polynomial.aeval r) (Nat.iterate (⇑Polynomial.derivative) k p)) (HSMul. …
      -/
      exact hc k (mem_Ico.mp hk).1 r
      /-
        🎉 no goals
      -/
    /-
      case inr.refine_2
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      p : Polynomial R
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      p0 : Ne p 0
      c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
      hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
      r : A
      p' : Polynomial A
      hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
      this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
      this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
      ⊢ Disjoint (Finset.range q) (Finset.Ico q (HAdd.hAdd p.natDegree 1))
    -/
  · rw [range_eq_Ico, disjoint_iff_inter_eq_empty, eq_empty_iff_forall_not_mem]
    /-
      case inr.refine_2
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      p : Polynomial R
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      p0 : Ne p 0
      c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
      hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
      r : A
      p' : Polynomial A
      hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
      this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
      this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
      ⊢ ∀ (x : Nat), Not (Membership.mem (Inter.inter (Finset.Ico 0 q) (Finset.Ico q …
    -/
    intro x hx
    /-
      case inr.refine_2
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      p : Polynomial R
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      p0 : Ne p 0
      c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
      hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
      r : A
      p' : Polynomial A
      hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
      this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
      this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
      x : Nat
      hx : Membership.mem (Inter.inter (Finset.Ico 0 q) (Finset.Ico q (HAdd.hAdd p.n …
      ⊢ False
    -/
    rw [mem_inter, mem_Ico, mem_Ico] at hx
    /-
      case inr.refine_2
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Nontrivial A
      inst✝ : NoZeroDivisors A
      p : Polynomial R
      q : Nat
      hq : LT.lt 0 q
      inj_amap : Function.Injective ⇑(algebraMap R A)
      p0 : Ne p 0
      c : Nat → Polynomial R := fun k => dite (LE.le q k) (fun hk => ⋯.choose) fun h …
      c_le : ∀ (k : Nat), LE.le (c k).natDegree (HSub.hSub p.natDegree k)
      hc : ∀ (k : Nat), LE.le q k → ∀ (r : A), Eq ((Polynomial.aeval r) (Nat.iterate …
      r : A
      p' : Polynomial A
      hp : Eq (Polynomial.map (algebraMap R A) p) (HMul.hMul (HPow.hPow (HSub.hSub P …
      this✝ : Eq (Finset.range (HAdd.hAdd p.natDegree 1)) (Union.union (Finset.range …
      this : Eq (Finset.range q) (Finset.range (HAdd.hAdd (HSub.hSub q 1) 1))
      x : Nat
      hx : And (And (LE.le 0 x) (LT.lt x q)) (And (LE.le q x) (LT.lt x (HAdd.hAdd p. …
      ⊢ False
    -/
    exact hx.1.2.not_le hx.2.1
    /-
      🎉 no goals
    -/


theorem eval_sumIDeriv_of_pos
    [CommRing R] [Nontrivial R] [NoZeroDivisors R] (p : R[X]) {q : ℕ} (hq : 0 < q) :
    ∃ gp : R[X], gp.natDegree ≤ p.natDegree - q ∧
      ∀ (r : R) {p' : R[X]},
        p = ((X : R[X]) - C r) ^ (q - 1) * p' →
        eval r (sumIDeriv p) = (q - 1)! • p'.eval r + q ! • eval r gp := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    q : Nat
    hq : LT.lt 0 q
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub p.natDegree q)) (∀ (r :  …
  -/
  simpa using aeval_sumIDeriv_of_pos R p hq Function.injective_id
  /-
    🎉 no goals
  -/


