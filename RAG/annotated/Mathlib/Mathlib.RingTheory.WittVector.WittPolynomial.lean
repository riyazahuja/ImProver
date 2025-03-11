/-- `wittPolynomial p R n` is the `n`-th Witt polynomial
with respect to a prime `p` with coefficients in a commutative ring `R`.
It is defined as:

`∑_{i ≤ n} p^i X_i^{p^{n-i}} ∈ R[X_0, X_1, X_2, …]`. -/
noncomputable def wittPolynomial (n : ℕ) : MvPolynomial ℕ R :=
  ∑ i ∈ range (n + 1), monomial (single i (p ^ (n - i))) ((p : R) ^ i)


theorem wittPolynomial_eq_sum_C_mul_X_pow (n : ℕ) :
    wittPolynomial p R n = ∑ i ∈ range (n + 1), C ((p : R) ^ i) * X i ^ p ^ (n - i) := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ⊢ Eq (wittPolynomial p R n) ((Finset.range (HAdd.hAdd n 1)).sum fun i => HMul. …
  -/
  apply sum_congr rfl
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Eq ((MvPolyno …
  -/
  rintro i -
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    n i : Nat
    ⊢ Eq ((MvPolynomial.monomial (Finsupp.single i (HPow.hPow p (HSub.hSub n i)))) …
  -/
  rw [monomial_eq, Finsupp.prod_single_index]
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    n i : Nat
    ⊢ Eq (HPow.hPow (MvPolynomial.X i) 0) 1
  -/
  rw [pow_zero]
  /-
    🎉 no goals
  -/


set_option quotPrecheck false in
@[inherit_doc]
scoped[Witt] notation "W_" => wittPolynomial p

-- Notation with ring of coefficients implicit

set_option quotPrecheck false in
@[inherit_doc]
scoped[Witt] notation "W" => wittPolynomial p _


@[simp]
theorem map_wittPolynomial (f : R →+* S) (n : ℕ) : map f (W n) = W n := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    n : Nat
    ⊢ Eq ((MvPolynomial.map f) (wittPolynomial p R n)) (wittPolynomial p S n)
  -/
  rw [wittPolynomial, map_sum, wittPolynomial]
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => (MvPolynomial.map f) ((MvPol …
  -/
  refine sum_congr rfl fun i _ => ?_
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    n i : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ Eq ((MvPolynomial.map f) ((MvPolynomial.monomial (Finsupp.single i (HPow.hPo …
  -/
  rw [map_monomial, RingHom.map_pow, map_natCast]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_wittPolynomial [hp : Fact p.Prime] (n : ℕ) :
    constantCoeff (wittPolynomial p R n) = 0 := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (wittPolynomial p R n)) 0
  -/
  simp only [wittPolynomial, map_sum, constantCoeff_monomial]
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => ite (Eq (Finsupp.single x (H …
  -/
  rw [sum_eq_zero]
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Eq (ite (Eq ( …
  -/
  rintro i _
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    n i : Nat
    a✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ Eq (ite (Eq (Finsupp.single i (HPow.hPow p (HSub.hSub n i))) 0) (HPow.hPow ( …
  -/
  rw [if_neg]
  /-
    case hnc
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    n i : Nat
    a✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ Not (Eq (Finsupp.single i (HPow.hPow p (HSub.hSub n i))) 0)
  -/
  rw [Finsupp.single_eq_zero]
  /-
    case hnc
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    n i : Nat
    a✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ Not (Eq (HPow.hPow p (HSub.hSub n i)) 0)
  -/
  exact ne_of_gt (pow_pos hp.1.pos _)
  /-
    🎉 no goals
  -/


@[simp]
theorem wittPolynomial_zero : wittPolynomial p R 0 = X 0 := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq (wittPolynomial p R 0) (MvPolynomial.X 0)
  -/
  simp only [wittPolynomial, X, sum_singleton, range_one, pow_zero, zero_add, tsub_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem wittPolynomial_one : wittPolynomial p R 1 = C (p : R) * X 1 + X 0 ^ p := by
  simp only [wittPolynomial_eq_sum_C_mul_X_pow, sum_range_succ_comm, range_one, sum_singleton,
    one_mul, pow_one, C_1, pow_zero, tsub_self, tsub_zero]


theorem aeval_wittPolynomial {A : Type*} [CommRing A] [Algebra R A] (f : ℕ → A) (n : ℕ) :
    aeval f (W_ R n) = ∑ i ∈ range (n + 1), (p : A) ^ i * f i ^ p ^ (n - i) := by
  /-
    p : Nat
    R : Type u_1
    inst✝² : CommRing R
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f : Nat → A
    n : Nat
    ⊢ Eq ((MvPolynomial.aeval f) (wittPolynomial p R n)) ((Finset.range (HAdd.hAdd …
  -/
  simp [wittPolynomial, map_sum, aeval_monomial, Finsupp.prod_single_index]
  /-
    🎉 no goals
  -/


/-- Over the ring `ZMod (p^(n+1))`, we produce the `n+1`st Witt polynomial
by expanding the `n`th Witt polynomial by `p`. -/
@[simp]
theorem wittPolynomial_zmod_self (n : ℕ) :
    W_ (ZMod (p ^ (n + 1))) (n + 1) = expand p (W_ (ZMod (p ^ (n + 1))) n) := by
  /-
    p n : Nat
    ⊢ Eq (wittPolynomial p (ZMod (HPow.hPow p (HAdd.hAdd n 1))) (HAdd.hAdd n 1)) ( …
  -/
  simp only [wittPolynomial_eq_sum_C_mul_X_pow]
  rw [sum_range_succ, ← Nat.cast_pow, CharP.cast_eq_zero (ZMod (p ^ (n + 1))) (p ^ (n + 1)), C_0,
    zero_mul, add_zero, map_sum, sum_congr rfl]
  /-
    p n : Nat
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Eq (HMul.hMul …
  -/
  intro k hk
  /-
    p n k : Nat
    hk : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
    ⊢ Eq (HMul.hMul (MvPolynomial.C (HPow.hPow (↑p) k)) (HPow.hPow (MvPolynomial.X …
  -/
  rw [map_mul (expand p), map_pow (expand p), expand_X, algHom_C, ← pow_mul, ← pow_succ']
  /-
    p n k : Nat
    hk : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
    ⊢ Eq (HMul.hMul (MvPolynomial.C (HPow.hPow (↑p) k)) (HPow.hPow (MvPolynomial.X …
  -/
  congr
  /-
    case e_a.e_a.e_a
    p n k : Nat
    hk : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
    ⊢ Eq (HSub.hSub (HAdd.hAdd n 1) k) (HAdd.hAdd (HSub.hSub n k) 1)
  -/
  rw [mem_range] at hk
  /-
    case e_a.e_a.e_a
    p n k : Nat
    hk : LT.lt k (HAdd.hAdd n 1)
    ⊢ Eq (HSub.hSub (HAdd.hAdd n 1) k) (HAdd.hAdd (HSub.hSub n k) 1)
  -/
  rw [add_comm, add_tsub_assoc_of_le (Nat.lt_succ_iff.mp hk), ← add_comm]
  /-
    🎉 no goals
  -/


theorem wittPolynomial_vars [CharZero R] (n : ℕ) : (wittPolynomial p R n).vars = range (n + 1) := by
  have : ∀ i, (monomial (Finsupp.single i (p ^ (n - i))) ((p : R) ^ i)).vars = {i} := by
    intro i
    refine vars_monomial_single i (pow_ne_zero _ hp.1) ?_
    rw [← Nat.cast_pow, Nat.cast_ne_zero]
    exact pow_ne_zero i hp.1
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    hp : NeZero p
    inst✝ : CharZero R
    n : Nat
    this : ∀ (i : Nat), Eq ((MvPolynomial.monomial (Finsupp.single i (HPow.hPow p  …
    ⊢ Eq (wittPolynomial p R n).vars (Finset.range (HAdd.hAdd n 1))
  -/
  rw [wittPolynomial, vars_sum_of_disjoint]
    /-
      p : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      hp : NeZero p
      inst✝ : CharZero R
      n : Nat
      this : ∀ (i : Nat), Eq ((MvPolynomial.monomial (Finsupp.single i (HPow.hPow p  …
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).biUnion fun i => ((MvPolynomial.monomial  …
    -/
  · simp only [this, biUnion_singleton_eq_self]
    /-
      🎉 no goals
    -/
    /-
      case h
      p : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      hp : NeZero p
      inst✝ : CharZero R
      n : Nat
      this : ∀ (i : Nat), Eq ((MvPolynomial.monomial (Finsupp.single i (HPow.hPow p  …
      ⊢ Pairwise (Function.onFun Disjoint fun i => ((MvPolynomial.monomial (Finsupp. …
    -/
  · simp only [this]
    /-
      case h
      p : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      hp : NeZero p
      inst✝ : CharZero R
      n : Nat
      this : ∀ (i : Nat), Eq ((MvPolynomial.monomial (Finsupp.single i (HPow.hPow p  …
      ⊢ Pairwise (Function.onFun Disjoint fun i => Singleton.singleton i)
    -/
    intro a b h
    /-
      case h
      p : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      hp : NeZero p
      inst✝ : CharZero R
      n : Nat
      this : ∀ (i : Nat), Eq ((MvPolynomial.monomial (Finsupp.single i (HPow.hPow p  …
      a b : Nat
      h : Ne a b
      ⊢ Function.onFun Disjoint (fun i => Singleton.singleton i) a b
    -/
    apply disjoint_singleton_left.mpr
    /-
      case h
      p : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      hp : NeZero p
      inst✝ : CharZero R
      n : Nat
      this : ∀ (i : Nat), Eq ((MvPolynomial.monomial (Finsupp.single i (HPow.hPow p  …
      a b : Nat
      h : Ne a b
      ⊢ Not (Membership.mem ((fun i => Singleton.singleton i) b) a)
    -/
    rwa [mem_singleton]
    /-
      🎉 no goals
    -/


theorem wittPolynomial_vars_subset (n : ℕ) : (wittPolynomial p R n).vars ⊆ range (n + 1) := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : NeZero p
    n : Nat
    ⊢ HasSubset.Subset (wittPolynomial p R n).vars (Finset.range (HAdd.hAdd n 1))
  -/
  rw [← map_wittPolynomial p (Int.castRingHom R), ← wittPolynomial_vars p ℤ]
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : NeZero p
    n : Nat
    ⊢ HasSubset.Subset ((MvPolynomial.map (Int.castRingHom R)) (wittPolynomial p I …
  -/
  apply vars_map
  /-
    🎉 no goals
  -/


/-- The `xInTermsOfW p R n` is the polynomial on the basis of Witt polynomials
that corresponds to the ordinary `X n`. -/
noncomputable def xInTermsOfW [Invertible (p : R)] : ℕ → MvPolynomial ℕ R
  | n => (X n - ∑ i : Fin n,
          C ((p : R) ^ (i : ℕ)) * xInTermsOfW i ^ p ^ (n - (i : ℕ))) * C ((⅟ p : R) ^ n)


theorem xInTermsOfW_eq [Invertible (p : R)] {n : ℕ} : xInTermsOfW p R n =
    (X n - ∑ i ∈ range n, C ((p : R) ^ i) *
      xInTermsOfW p R i ^ p ^ (n - i)) * C ((⅟p : R) ^ n) := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    n : Nat
    ⊢ Eq (xInTermsOfW p R n) (HMul.hMul (HSub.hSub (MvPolynomial.X n) ((Finset.ran …
  -/
  rw [xInTermsOfW, ← Fin.sum_univ_eq_sum_range]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_xInTermsOfW [hp : Fact p.Prime] [Invertible (p : R)] (n : ℕ) :
    constantCoeff (xInTermsOfW p R n) = 0 := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    hp : Fact (Nat.Prime p)
    inst✝ : Invertible ↑p
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (xInTermsOfW p R n)) 0
  -/
  induction n using Nat.strongRecOn with | ind n IH => ?_
  rw [xInTermsOfW_eq, mul_comm, RingHom.map_mul, RingHom.map_sub, map_sum, constantCoeff_C,
    constantCoeff_X, zero_sub, mul_neg, neg_eq_zero]
  -- Porting note: here, we should be able to do `rw [sum_eq_zero]`, but the goal that
  -- is created is not what we expect, and the sum is not replaced by zero...
  -- is it a bug in `rw` tactic?
  /-
    case ind
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    hp : Fact (Nat.Prime p)
    inst✝ : Invertible ↑p
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (MvPolynomial.constantCoeff (xInTermsOfW p R  …
    ⊢ Eq (HMul.hMul (HPow.hPow (Invertible.invOf ↑p) n) ((Finset.range n).sum fun  …
  -/
  refine Eq.trans (?_ : _ = ((⅟↑p : R) ^ n)* 0) (mul_zero _)
  /-
    case ind
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    hp : Fact (Nat.Prime p)
    inst✝ : Invertible ↑p
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (MvPolynomial.constantCoeff (xInTermsOfW p R  …
    ⊢ Eq (HMul.hMul (HPow.hPow (Invertible.invOf ↑p) n) ((Finset.range n).sum fun  …
  -/
  congr 1
  /-
    case ind.e_a
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    hp : Fact (Nat.Prime p)
    inst✝ : Invertible ↑p
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (MvPolynomial.constantCoeff (xInTermsOfW p R  …
    ⊢ Eq ((Finset.range n).sum fun x => MvPolynomial.constantCoeff (HMul.hMul (MvP …
  -/
  rw [sum_eq_zero]
  /-
    case ind.e_a
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    hp : Fact (Nat.Prime p)
    inst✝ : Invertible ↑p
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (MvPolynomial.constantCoeff (xInTermsOfW p R  …
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range n) x → Eq (MvPolynomial.constantCo …
  -/
  intro m H
  /-
    case ind.e_a
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    hp : Fact (Nat.Prime p)
    inst✝ : Invertible ↑p
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (MvPolynomial.constantCoeff (xInTermsOfW p R  …
    m : Nat
    H : Membership.mem (Finset.range n) m
    ⊢ Eq (MvPolynomial.constantCoeff (HMul.hMul (MvPolynomial.C (HPow.hPow (↑p) m) …
  -/
  rw [mem_range] at H
  /-
    case ind.e_a
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    hp : Fact (Nat.Prime p)
    inst✝ : Invertible ↑p
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (MvPolynomial.constantCoeff (xInTermsOfW p R  …
    m : Nat
    H : LT.lt m n
    ⊢ Eq (MvPolynomial.constantCoeff (HMul.hMul (MvPolynomial.C (HPow.hPow (↑p) m) …
  -/
  simp only [RingHom.map_mul, RingHom.map_pow, map_natCast, IH m H]
  /-
    case ind.e_a
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    hp : Fact (Nat.Prime p)
    inst✝ : Invertible ↑p
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (MvPolynomial.constantCoeff (xInTermsOfW p R  …
    m : Nat
    H : LT.lt m n
    ⊢ Eq (HMul.hMul (HPow.hPow (↑p) m) (HPow.hPow 0 (HPow.hPow p (HSub.hSub n m))) …
  -/
  rw [zero_pow, mul_zero]
  /-
    case ind.e_a
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    hp : Fact (Nat.Prime p)
    inst✝ : Invertible ↑p
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (MvPolynomial.constantCoeff (xInTermsOfW p R  …
    m : Nat
    H : LT.lt m n
    ⊢ Ne (HPow.hPow p (HSub.hSub n m)) 0
  -/
  exact pow_ne_zero _ hp.1.ne_zero
  /-
    🎉 no goals
  -/


@[simp]
theorem xInTermsOfW_zero [Invertible (p : R)] : xInTermsOfW p R 0 = X 0 := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    ⊢ Eq (xInTermsOfW p R 0) (MvPolynomial.X 0)
  -/
  rw [xInTermsOfW_eq, range_zero, sum_empty, pow_zero, C_1, mul_one, sub_zero]
  /-
    🎉 no goals
  -/


theorem xInTermsOfW_vars_aux (n : ℕ) :
    n ∈ (xInTermsOfW p ℚ n).vars ∧ (xInTermsOfW p ℚ n).vars ⊆ range (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ And (Membership.mem (xInTermsOfW p Rat n).vars n) (HasSubset.Subset (xInTerm …
  -/
  induction n using Nat.strongRecOn with | ind n ih => ?_
  rw [xInTermsOfW_eq, mul_comm, vars_C_mul _ (Invertible.ne_zero _),
    vars_sub_of_disjoint, vars_X, range_succ, insert_eq]
  on_goal 1 =>
    simp only [true_and, true_or, eq_self_iff_true, mem_union, mem_singleton]
    intro i
    rw [mem_union, mem_union]
    apply Or.imp id
  /-
    case ind
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (m : Nat), LT.lt m n → And (Membership.mem (xInTermsOfW p Rat m).vars m …
    i : Nat
    ⊢ Membership.mem ((Finset.range n).sum fun i => HMul.hMul (MvPolynomial.C (HPo …
  -/
  on_goal 2 => rw [vars_X, disjoint_singleton_left]
  all_goals
    intro H
    replace H := vars_sum_subset _ _ H
    rw [mem_biUnion] at H
    rcases H with ⟨j, hj, H⟩
    rw [vars_C_mul] at H
    swap
    · apply pow_ne_zero
      exact mod_cast hp.1.ne_zero
    rw [mem_range] at hj
    replace H := (ih j hj).2 (vars_pow _ _ H)
    rw [mem_range] at H
    /-
      case ind.intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → And (Membership.mem (xInTermsOfW p Rat m).vars m …
      i j : Nat
      hj : LT.lt j n
      H : LT.lt i (HAdd.hAdd j 1)
      ⊢ Membership.mem (Finset.range n) i
    -/
  · rw [mem_range]
    /-
      case ind.intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → And (Membership.mem (xInTermsOfW p Rat m).vars m …
      i j : Nat
      hj : LT.lt j n
      H : LT.lt i (HAdd.hAdd j 1)
      ⊢ LT.lt i n
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case ind.hpq.intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → And (Membership.mem (xInTermsOfW p Rat m).vars m …
      j : Nat
      hj : LT.lt j n
      H : LT.lt n (HAdd.hAdd j 1)
      ⊢ False
    -/
  · omega
    /-
      🎉 no goals
    -/


theorem xInTermsOfW_vars_subset (n : ℕ) : (xInTermsOfW p ℚ n).vars ⊆ range (n + 1) :=
  (xInTermsOfW_vars_aux p n).2


theorem xInTermsOfW_aux [Invertible (p : R)] (n : ℕ) :
    xInTermsOfW p R n * C ((p : R) ^ n) =
      X n - ∑ i ∈ range n, C ((p : R) ^ i) * xInTermsOfW p R i ^ p ^ (n - i) := by
  rw [xInTermsOfW_eq, mul_assoc, ← C_mul, ← mul_pow, invOf_mul_self,
    one_pow, C_1, mul_one]


@[simp]
theorem bind₁_xInTermsOfW_wittPolynomial [Invertible (p : R)] (k : ℕ) :
    bind₁ (xInTermsOfW p R) (W_ R k) = X k := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    k : Nat
    ⊢ Eq ((MvPolynomial.bind₁ (xInTermsOfW p R)) (wittPolynomial p R k)) (MvPolyno …
  -/
  rw [wittPolynomial_eq_sum_C_mul_X_pow, map_sum]
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    k : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd k 1)).sum fun x => (MvPolynomial.bind₁ (xInTerm …
  -/
  simp only [Nat.cast_pow, map_pow, C_pow, map_mul, algHom_C, algebraMap_eq]
  rw [sum_range_succ_comm, tsub_self, pow_zero, pow_one, bind₁_X_right, mul_comm, ← C_pow,
    xInTermsOfW_aux]
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    k : Nat
    ⊢ Eq (HAdd.hAdd (HSub.hSub (MvPolynomial.X k) ((Finset.range k).sum fun i => H …
  -/
  simp only [Nat.cast_pow, C_pow, bind₁_X_right, sub_add_cancel]
  /-
    🎉 no goals
  -/


@[simp]
theorem bind₁_wittPolynomial_xInTermsOfW [Invertible (p : R)] (n : ℕ) :
    bind₁ (W_ R) (xInTermsOfW p R n) = X n := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ (wittPolynomial p R)) (xInTermsOfW p R n)) (MvPolyno …
  -/
  induction n using Nat.strongRecOn with | ind n H => ?_
  rw [xInTermsOfW_eq, map_mul, map_sub, bind₁_X_right, algHom_C, map_sum,
    show X n = (X n * C ((p : R) ^ n)) * C ((⅟p : R) ^ n) by
      rw [mul_assoc, ← C_mul, ← mul_pow, mul_invOf_self, one_pow, map_one, mul_one]]
  /-
    case ind
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    n : Nat
    H : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.bind₁ (wittPolynomial p R)) (xI …
    ⊢ Eq (HMul.hMul (HSub.hSub (wittPolynomial p R n) ((Finset.range n).sum fun x  …
  -/
  congr 1
  rw [wittPolynomial_eq_sum_C_mul_X_pow, sum_range_succ_comm,
    tsub_self, pow_zero, pow_one, mul_comm (X n), add_sub_assoc, add_right_eq_self, sub_eq_zero]
  /-
    case ind.e_a
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    n : Nat
    H : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.bind₁ (wittPolynomial p R)) (xI …
    ⊢ Eq ((Finset.range n).sum fun x => HMul.hMul (MvPolynomial.C (HPow.hPow (↑p)  …
  -/
  apply sum_congr rfl
  /-
    case ind.e_a
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    n : Nat
    H : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.bind₁ (wittPolynomial p R)) (xI …
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range n) x → Eq (HMul.hMul (MvPolynomial …
  -/
  intro i h
  /-
    case ind.e_a
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    n : Nat
    H : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.bind₁ (wittPolynomial p R)) (xI …
    i : Nat
    h : Membership.mem (Finset.range n) i
    ⊢ Eq (HMul.hMul (MvPolynomial.C (HPow.hPow (↑p) i)) (HPow.hPow (MvPolynomial.X …
  -/
  rw [mem_range] at h
  /-
    case ind.e_a
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible ↑p
    n : Nat
    H : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.bind₁ (wittPolynomial p R)) (xI …
    i : Nat
    h : LT.lt i n
    ⊢ Eq (HMul.hMul (MvPolynomial.C (HPow.hPow (↑p) i)) (HPow.hPow (MvPolynomial.X …
  -/
  rw [map_mul, map_pow (bind₁ _), algHom_C, H i h, algebraMap_eq]
  /-
    🎉 no goals
  -/

