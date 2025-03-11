local notation "𝕎" => WittVector p

-- Porting note: new notation

local notation "𝕄" => MvPolynomial (Fin 2 × ℕ) ℤ


/--
```
(∑ i ∈ range n, (y.coeff i)^(p^(n-i)) * p^i.val) *
(∑ i ∈ range n, (y.coeff i)^(p^(n-i)) * p^i.val)
```
-/
def wittPolyProd (n : ℕ) : 𝕄 :=
  rename (Prod.mk (0 : Fin 2)) (wittPolynomial p ℤ n) *
    rename (Prod.mk (1 : Fin 2)) (wittPolynomial p ℤ n)


theorem wittPolyProd_vars (n : ℕ) : (wittPolyProd p n).vars ⊆ univ ×ˢ range (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (WittVector.wittPolyProd p n).vars (SProd.sprod Finset.univ …
  -/
  rw [wittPolyProd]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (HMul.hMul ((MvPolynomial.rename (Prod.mk 0)) (wittPolynomi …
  -/
  apply Subset.trans (vars_mul _ _)
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (Union.union ((MvPolynomial.rename (Prod.mk 0)) (wittPolyno …
  -/
  refine union_subset ?_ ?_ <;>
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ HasSubset.Subset ((MvPolynomial.rename (Prod.mk 0)) (wittPolynomial p Int n) …
    -/
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ HasSubset.Subset (Finset.image (Prod.mk 0) (wittPolynomial p Int n).vars) (S …
    -/
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ HasSubset.Subset (Finset.image (Prod.mk 1) (wittPolynomial p Int n).vars) (S …
    -/
    simp [wittPolynomial_vars, image_subset_iff]
    /-
      🎉 no goals
    -/


/-- The "remainder term" of `WittVector.wittPolyProd`. See `mul_polyOfInterest_aux2`. -/
def wittPolyProdRemainder (n : ℕ) : 𝕄 :=
  ∑ i ∈ range n, (p : 𝕄) ^ i * wittMul p i ^ p ^ (n - i)


theorem wittPolyProdRemainder_vars (n : ℕ) :
    (wittPolyProdRemainder p n).vars ⊆ univ ×ˢ range n := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (WittVector.wittPolyProdRemainder p n).vars (SProd.sprod Fi …
  -/
  rw [wittPolyProdRemainder]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow (↑p) i) …
  -/
  refine Subset.trans (vars_sum_subset _ _) ?_
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset ((Finset.range n).biUnion fun i => (HMul.hMul (HPow.hPow (↑ …
  -/
  rw [biUnion_subset]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range n) x → HasSubset.Subset (HMul.hMul …
  -/
  intro x hx
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n x : Nat
    hx : Membership.mem (Finset.range n) x
    ⊢ HasSubset.Subset (HMul.hMul (HPow.hPow (↑p) x) (HPow.hPow (WittVector.wittMu …
  -/
  apply Subset.trans (vars_mul _ _)
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n x : Nat
    hx : Membership.mem (Finset.range n) x
    ⊢ HasSubset.Subset (Union.union (HPow.hPow (↑p) x).vars (HPow.hPow (WittVector …
  -/
  refine union_subset ?_ ?_
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n x : Nat
      hx : Membership.mem (Finset.range n) x
      ⊢ HasSubset.Subset (HPow.hPow (↑p) x).vars (SProd.sprod Finset.univ (Finset.ra …
    -/
  · apply Subset.trans (vars_pow _ _)
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n x : Nat
      hx : Membership.mem (Finset.range n) x
      ⊢ HasSubset.Subset (↑p).vars (SProd.sprod Finset.univ (Finset.range n))
    -/
    have : (p : 𝕄) = C (p : ℤ) := by simp only [Int.cast_natCast, eq_intCast]
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n x : Nat
      hx : Membership.mem (Finset.range n) x
      this : Eq (↑p) (MvPolynomial.C ↑p)
      ⊢ HasSubset.Subset (↑p).vars (SProd.sprod Finset.univ (Finset.range n))
    -/
    rw [this, vars_C]
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n x : Nat
      hx : Membership.mem (Finset.range n) x
      this : Eq (↑p) (MvPolynomial.C ↑p)
      ⊢ HasSubset.Subset EmptyCollection.emptyCollection (SProd.sprod Finset.univ (F …
    -/
    apply empty_subset
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n x : Nat
      hx : Membership.mem (Finset.range n) x
      ⊢ HasSubset.Subset (HPow.hPow (WittVector.wittMul p x) (HPow.hPow p (HSub.hSub …
    -/
  · apply Subset.trans (vars_pow _ _)
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n x : Nat
      hx : Membership.mem (Finset.range n) x
      ⊢ HasSubset.Subset (WittVector.wittMul p x).vars (SProd.sprod Finset.univ (Fin …
    -/
    apply Subset.trans (wittMul_vars _ _)
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n x : Nat
      hx : Membership.mem (Finset.range n) x
      ⊢ HasSubset.Subset (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd x 1))) (S …
    -/
    apply product_subset_product (Subset.refl _)
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n x : Nat
      hx : Membership.mem (Finset.range n) x
      ⊢ HasSubset.Subset (Finset.range (HAdd.hAdd x 1)) (Finset.range n)
    -/
    simp only [mem_range, range_subset] at hx ⊢
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n x : Nat
      hx : LT.lt x n
      ⊢ LE.le (HAdd.hAdd x 1) n
    -/
    exact hx
    /-
      🎉 no goals
    -/


/-- `remainder p n` represents the remainder term from `mul_polyOfInterest_aux3`.
`wittPolyProd p (n+1)` will have variables up to `n+1`,
but `remainder` will only have variables up to `n`.
-/
def remainder (n : ℕ) : 𝕄 :=
  (∑ x ∈ range (n + 1),
    (rename (Prod.mk 0)) ((monomial (Finsupp.single x (p ^ (n + 1 - x)))) ((p : ℤ) ^ x))) *
   ∑ x ∈ range (n + 1),
    (rename (Prod.mk 1)) ((monomial (Finsupp.single x (p ^ (n + 1 - x)))) ((p : ℤ) ^ x))


theorem remainder_vars (n : ℕ) : (remainder p n).vars ⊆ univ ×ˢ range (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (WittVector.remainder p n).vars (SProd.sprod Finset.univ (F …
  -/
  rw [remainder]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (HMul.hMul ((Finset.range (HAdd.hAdd n 1)).sum fun x => (Mv …
  -/
  apply Subset.trans (vars_mul _ _)
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (Union.union ((Finset.range (HAdd.hAdd n 1)).sum fun x => ( …
  -/
  refine union_subset ?_ ?_ <;>
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ HasSubset.Subset ((Finset.range (HAdd.hAdd n 1)).sum fun x => (MvPolynomial. …
    -/
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ HasSubset.Subset ((Finset.range (HAdd.hAdd n 1)).biUnion fun i => ((MvPolyno …
    -/
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → HasSubset.Sub …
    -/
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n x : Nat
      hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
      ⊢ HasSubset.Subset ((MvPolynomial.rename (Prod.mk 0)) ((MvPolynomial.monomial  …
    -/
      /-
        case refine_1
        p : Nat
        hp : Fact (Nat.Prime p)
        n x : Nat
        hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
        ⊢ HasSubset.Subset (Finsupp.single { fst := 0, snd := x } (HPow.hPow p (HSub.h …
      -/
      /-
        case refine_1
        p : Nat
        hp : Fact (Nat.Prime p)
        n x : Nat
        hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
        ⊢ HasSubset.Subset (Singleton.singleton { fst := 0, snd := x }) (SProd.sprod F …
      -/
      /-
        🎉 no goals
      -/
      /-
        case refine_1
        p : Nat
        hp : Fact (Nat.Prime p)
        n x : Nat
        hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
        ⊢ Ne (HPow.hPow (↑p) x) 0
      -/
      /-
        case refine_1.h
        p : Nat
        hp : Fact (Nat.Prime p)
        n x : Nat
        hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
        ⊢ Ne (↑p) 0
      -/
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        p : Nat
        hp : Fact (Nat.Prime p)
        n x : Nat
        hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
        ⊢ HasSubset.Subset (Finsupp.single { fst := 1, snd := x } (HPow.hPow p (HSub.h …
      -/
    · apply Subset.trans Finsupp.support_single_subset
      /-
        case refine_2
        p : Nat
        hp : Fact (Nat.Prime p)
        n x : Nat
        hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
        ⊢ HasSubset.Subset (Singleton.singleton { fst := 1, snd := x }) (SProd.sprod F …
      -/
      simpa using mem_range.mp hx
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        p : Nat
        hp : Fact (Nat.Prime p)
        n x : Nat
        hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
        ⊢ Ne (HPow.hPow (↑p) x) 0
      -/
    · apply pow_ne_zero
      /-
        case refine_2.h
        p : Nat
        hp : Fact (Nat.Prime p)
        n x : Nat
        hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
        ⊢ Ne (↑p) 0
      -/
      exact mod_cast hp.out.ne_zero
      /-
        🎉 no goals
      -/


/-- This is the polynomial whose degree we want to get a handle on. -/
def polyOfInterest (n : ℕ) : 𝕄 :=
  wittMul p (n + 1) + (p : 𝕄) ^ (n + 1) * X (0, n + 1) * X (1, n + 1) -
    X (0, n + 1) * rename (Prod.mk (1 : Fin 2)) (wittPolynomial p ℤ (n + 1)) -
    X (1, n + 1) * rename (Prod.mk (0 : Fin 2)) (wittPolynomial p ℤ (n + 1))


theorem mul_polyOfInterest_aux1 (n : ℕ) :
    ∑ i ∈ range (n + 1), (p : 𝕄) ^ i * wittMul p i ^ p ^ (n - i) = wittPolyProd p n := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun i => HMul.hMul (HPow.hPow (↑p) i) …
  -/
  simp only [wittPolyProd]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun i => HMul.hMul (HPow.hPow (↑p) i) …
  -/
  convert wittStructureInt_prop p (X (0 : Fin 2) * X 1) n using 1
    /-
      case h.e'_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun i => HMul.hMul (HPow.hPow (↑p) i) …
    -/
  · simp only [wittPolynomial, wittMul]
    /-
      case h.e'_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul (HPow.hPow (↑p) x) …
    -/
    rw [map_sum]
    /-
      case h.e'_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul (HPow.hPow (↑p) x) …
    -/
    congr 1 with i
    /-
      case h.e'_2.e_f.h.a
      p : Nat
      hp : Fact (Nat.Prime p)
      n i : Nat
      m✝ : Finsupp (Prod (Fin 2) Nat) Nat
      ⊢ Eq (MvPolynomial.coeff m✝ (HMul.hMul (HPow.hPow (↑p) i) (HPow.hPow (wittStru …
    -/
    congr 1
    have hsupp : (Finsupp.single i (p ^ (n - i))).support = {i} := by
      rw [Finsupp.support_eq_singleton]
      simp only [and_true, Finsupp.single_eq_same, eq_self_iff_true, Ne]
      exact pow_ne_zero _ hp.out.ne_zero
    simp only [bind₁_monomial, hsupp, Int.cast_natCast, prod_singleton, eq_intCast,
      Finsupp.single_eq_same, C_pow, mul_eq_mul_left_iff, eq_self_iff_true, Int.cast_pow]
    /-
      case h.e'_3
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ Eq (HMul.hMul ((MvPolynomial.rename (Prod.mk 0)) (wittPolynomial p Int n)) ( …
    -/
  · simp only [map_mul, bind₁_X_right]
    /-
      🎉 no goals
    -/


theorem mul_polyOfInterest_aux2 (n : ℕ) :
    (p : 𝕄) ^ n * wittMul p n + wittPolyProdRemainder p n = wittPolyProd p n := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (↑p) n) (WittVector.wittMul p n)) (WittV …
  -/
  convert mul_polyOfInterest_aux1 p n
  /-
    case h.e'_2
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (↑p) n) (WittVector.wittMul p n)) (WittV …
  -/
  rw [sum_range_succ, add_comm, Nat.sub_self, pow_zero, pow_one]
  /-
    case h.e'_2
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HAdd.hAdd (WittVector.wittPolyProdRemainder p n) (HMul.hMul (HPow.hPow ( …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- We redeclare `p` here to locally discard the unneeded `p.Prime` hypothesis.

theorem mul_polyOfInterest_aux3 (p n : ℕ) : wittPolyProd p (n + 1) =
    -((p : 𝕄) ^ (n + 1) * X (0, n + 1)) * ((p : 𝕄) ^ (n + 1) * X (1, n + 1)) +
    (p : 𝕄) ^ (n + 1) * X (0, n + 1) * rename (Prod.mk (1 : Fin 2)) (wittPolynomial p ℤ (n + 1)) +
    (p : 𝕄) ^ (n + 1) * X (1, n + 1) * rename (Prod.mk (0 : Fin 2)) (wittPolynomial p ℤ (n + 1)) +
    remainder p n := by
  -- a useful auxiliary fact
  /-
    p n : Nat
    ⊢ Eq (WittVector.wittPolyProd p (HAdd.hAdd n 1)) (HAdd.hAdd (HAdd.hAdd (HAdd.h …
  -/
  have mvpz : (p : 𝕄) ^ (n + 1) = MvPolynomial.C ((p : ℤ) ^ (n + 1)) := by norm_cast
  -- Porting note: the original proof applies `sum_range_succ` through a non-`conv` rewrite,
  -- but this does not work in Lean 4; the whole proof also times out very badly. The proof has been
  -- nearly totally rewritten here and now finishes quite fast.
  /-
    p n : Nat
    mvpz : Eq (HPow.hPow (↑p) (HAdd.hAdd n 1)) (MvPolynomial.C (HPow.hPow (↑p) (HA …
    ⊢ Eq (WittVector.wittPolyProd p (HAdd.hAdd n 1)) (HAdd.hAdd (HAdd.hAdd (HAdd.h …
  -/
  rw [wittPolyProd, wittPolynomial, map_sum, map_sum]
  conv_lhs =>
    arg 1
    rw [sum_range_succ, ← C_mul_X_pow_eq_monomial, tsub_self, pow_zero, pow_one, map_mul,
      rename_C, rename_X, ← mvpz]
  conv_lhs =>
    arg 2
    rw [sum_range_succ, ← C_mul_X_pow_eq_monomial, tsub_self, pow_zero, pow_one, map_mul,
      rename_C, rename_X, ← mvpz]
  conv_rhs =>
    enter [1, 1, 2, 2]
    rw [sum_range_succ, ← C_mul_X_pow_eq_monomial, tsub_self, pow_zero, pow_one, map_mul,
      rename_C, rename_X, ← mvpz]
  conv_rhs =>
    enter [1, 2, 2]
    rw [sum_range_succ, ← C_mul_X_pow_eq_monomial, tsub_self, pow_zero, pow_one, map_mul,
      rename_C, rename_X, ← mvpz]
  /-
    p n : Nat
    mvpz : Eq (HPow.hPow (↑p) (HAdd.hAdd n 1)) (MvPolynomial.C (HPow.hPow (↑p) (HA …
    ⊢ Eq (HMul.hMul (HAdd.hAdd ((Finset.range (HAdd.hAdd n 1)).sum fun x => (MvPol …
  -/
  simp only [add_mul, mul_add]
  /-
    p n : Nat
    mvpz : Eq (HPow.hPow (↑p) (HAdd.hAdd n 1)) (MvPolynomial.C (HPow.hPow (↑p) (HA …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul ((Finset.range (HAdd.hAdd n 1)).sum fun  …
  -/
  rw [add_comm _ (remainder p n)]
  /-
    p n : Nat
    mvpz : Eq (HPow.hPow (↑p) (HAdd.hAdd n 1)) (MvPolynomial.C (HPow.hPow (↑p) (HA …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul ((Finset.range (HAdd.hAdd n 1)).sum fun  …
  -/
  simp only [add_assoc]
  /-
    p n : Nat
    mvpz : Eq (HPow.hPow (↑p) (HAdd.hAdd n 1)) (MvPolynomial.C (HPow.hPow (↑p) (HA …
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((Finset.range (HAdd.hAdd n 1)).sum fun x => (MvPol …
  -/
  apply congrArg (Add.add _)
  /-
    p n : Nat
    mvpz : Eq (HPow.hPow (↑p) (HAdd.hAdd n 1)) (MvPolynomial.C (HPow.hPow (↑p) (HA …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow (↑p) (HAdd.hAdd n 1)) (MvPoly …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem mul_polyOfInterest_aux4 (n : ℕ) :
    (p : 𝕄) ^ (n + 1) * wittMul p (n + 1) =
    -((p : 𝕄) ^ (n + 1) * X (0, n + 1)) * ((p : 𝕄) ^ (n + 1) * X (1, n + 1)) +
    (p : 𝕄) ^ (n + 1) * X (0, n + 1) * rename (Prod.mk (1 : Fin 2)) (wittPolynomial p ℤ (n + 1)) +
    (p : 𝕄) ^ (n + 1) * X (1, n + 1) * rename (Prod.mk (0 : Fin 2)) (wittPolynomial p ℤ (n + 1)) +
    (remainder p n - wittPolyProdRemainder p (n + 1)) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (↑p) (HAdd.hAdd n 1)) (WittVector.wittMul p (HAdd.h …
  -/
  rw [← add_sub_assoc, eq_sub_iff_add_eq, mul_polyOfInterest_aux2]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (WittVector.wittPolyProd p (HAdd.hAdd n 1)) (HAdd.hAdd (HAdd.hAdd (HAdd.h …
  -/
  exact mul_polyOfInterest_aux3 _ _
  /-
    🎉 no goals
  -/


theorem mul_polyOfInterest_aux5 (n : ℕ) :
    (p : 𝕄) ^ (n + 1) * polyOfInterest p n = remainder p n - wittPolyProdRemainder p (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (↑p) (HAdd.hAdd n 1)) (WittVector.polyOfInterest p  …
  -/
  simp only [polyOfInterest, mul_sub, mul_add, sub_eq_iff_eq_add']
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (↑p) (HAdd.hAdd n 1)) (WittVector.wittMu …
  -/
  rw [mul_polyOfInterest_aux4 p n]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Neg.neg (HMul.hMu …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem mul_polyOfInterest_vars (n : ℕ) :
    ((p : 𝕄) ^ (n + 1) * polyOfInterest p n).vars ⊆ univ ×ˢ range (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (HMul.hMul (HPow.hPow (↑p) (HAdd.hAdd n 1)) (WittVector.pol …
  -/
  rw [mul_polyOfInterest_aux5]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (HSub.hSub (WittVector.remainder p n) (WittVector.wittPolyP …
  -/
  apply Subset.trans (vars_sub_subset _)
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (Union.union (WittVector.remainder p n).vars (WittVector.wi …
  -/
  refine union_subset ?_ ?_
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ HasSubset.Subset (WittVector.remainder p n).vars (SProd.sprod Finset.univ (F …
    -/
  · apply remainder_vars
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ HasSubset.Subset (WittVector.wittPolyProdRemainder p (HAdd.hAdd n 1)).vars ( …
    -/
  · apply wittPolyProdRemainder_vars
    /-
      🎉 no goals
    -/


theorem polyOfInterest_vars_eq (n : ℕ) : (polyOfInterest p n).vars =
    ((p : 𝕄) ^ (n + 1) * (wittMul p (n + 1) + (p : 𝕄) ^ (n + 1) * X (0, n + 1) * X (1, n + 1) -
      X (0, n + 1) * rename (Prod.mk (1 : Fin 2)) (wittPolynomial p ℤ (n + 1)) -
      X (1, n + 1) * rename (Prod.mk (0 : Fin 2)) (wittPolynomial p ℤ (n + 1)))).vars := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (WittVector.polyOfInterest p n).vars (HMul.hMul (HPow.hPow (↑p) (HAdd.hAd …
  -/
  have : (p : 𝕄) ^ (n + 1) = C ((p : ℤ) ^ (n + 1)) := by norm_cast
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    this : Eq (HPow.hPow (↑p) (HAdd.hAdd n 1)) (MvPolynomial.C (HPow.hPow (↑p) (HA …
    ⊢ Eq (WittVector.polyOfInterest p n).vars (HMul.hMul (HPow.hPow (↑p) (HAdd.hAd …
  -/
  rw [polyOfInterest, this, vars_C_mul]
  /-
    case ha
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    this : Eq (HPow.hPow (↑p) (HAdd.hAdd n 1)) (MvPolynomial.C (HPow.hPow (↑p) (HA …
    ⊢ Ne (HPow.hPow (↑p) (HAdd.hAdd n 1)) 0
  -/
  apply pow_ne_zero
  /-
    case ha.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    this : Eq (HPow.hPow (↑p) (HAdd.hAdd n 1)) (MvPolynomial.C (HPow.hPow (↑p) (HA …
    ⊢ Ne (↑p) 0
  -/
  exact mod_cast hp.out.ne_zero
  /-
    🎉 no goals
  -/


theorem polyOfInterest_vars (n : ℕ) : (polyOfInterest p n).vars ⊆ univ ×ˢ range (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ HasSubset.Subset (WittVector.polyOfInterest p n).vars (SProd.sprod Finset.un …
  -/
  rw [polyOfInterest_vars_eq]; apply mul_polyOfInterest_vars
                               /-
                                 🎉 no goals
                               -/


theorem peval_polyOfInterest (n : ℕ) (x y : 𝕎 k) :
    peval (polyOfInterest p n) ![fun i => x.coeff i, fun i => y.coeff i] =
    (x * y).coeff (n + 1) + p ^ (n + 1) * x.coeff (n + 1) * y.coeff (n + 1) -
      y.coeff (n + 1) * ∑ i ∈ range (n + 1 + 1), p ^ i * x.coeff i ^ p ^ (n + 1 - i) -
      x.coeff (n + 1) * ∑ i ∈ range (n + 1 + 1), p ^ i * y.coeff i ^ p ^ (n + 1 - i) := by
  simp only [polyOfInterest, peval, map_natCast, Matrix.head_cons, map_pow,
    Function.uncurry_apply_pair, aeval_X, Matrix.cons_val_one, map_mul, Matrix.cons_val_zero,
    map_sub]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝ : CommRing k
    n : Nat
    x y : WittVector p k
    ⊢ Eq (HSub.hSub (HSub.hSub ((MvPolynomial.aeval (Function.uncurry (Matrix.vecC …
  -/
  rw [sub_sub, add_comm (_ * _), ← sub_sub]
  simp [wittPolynomial_eq_sum_C_mul_X_pow, aeval, eval₂_rename, mul_coeff, peval, map_natCast,
    map_add, map_pow, map_mul]


/-- The characteristic `p` version of `peval_polyOfInterest` -/
theorem peval_polyOfInterest' (n : ℕ) (x y : 𝕎 k) :
    peval (polyOfInterest p n) ![fun i => x.coeff i, fun i => y.coeff i] =
      (x * y).coeff (n + 1) - y.coeff (n + 1) * x.coeff 0 ^ p ^ (n + 1) -
        x.coeff (n + 1) * y.coeff 0 ^ p ^ (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    x y : WittVector p k
    ⊢ Eq (WittVector.peval (WittVector.polyOfInterest p n) (Matrix.vecCons (fun i  …
  -/
  rw [peval_polyOfInterest]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    x y : WittVector p k
    ⊢ Eq (HSub.hSub (HSub.hSub (HAdd.hAdd ((HMul.hMul x y).coeff (HAdd.hAdd n 1))  …
  -/
  have : (p : k) = 0 := CharP.cast_eq_zero k p
  simp only [this, Nat.cast_pow, ne_eq, add_eq_zero, and_false, zero_pow, zero_mul, add_zero,
    not_false_eq_true, reduceCtorEq]
  have sum_zero_pow_mul_pow_p (y : 𝕎 k) : ∑ x ∈ range (n + 1 + 1),
      (0 : k) ^ x * y.coeff x ^ p ^ (n + 1 - x) = y.coeff 0 ^ p ^ (n + 1) := by
    rw [Finset.sum_eq_single_of_mem 0] <;> simp +contextual
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    x y : WittVector p k
    this : Eq (↑p) 0
    sum_zero_pow_mul_pow_p : ∀ (y : WittVector p k), Eq ((Finset.range (HAdd.hAdd  …
    ⊢ Eq (HSub.hSub (HSub.hSub ((HMul.hMul x y).coeff (HAdd.hAdd n 1)) (HMul.hMul  …
  -/
            /-
              🎉 no goals
            -/
  congr <;> apply sum_zero_pow_mul_pow_p
            /-
              🎉 no goals
            -/


theorem nth_mul_coeff' (n : ℕ) :
    ∃ f : TruncatedWittVector p (n + 1) k → TruncatedWittVector p (n + 1) k → k,
    ∀ x y : 𝕎 k, f (truncateFun (n + 1) x) (truncateFun (n + 1) y) =
      (x * y).coeff (n + 1) - y.coeff (n + 1) * x.coeff 0 ^ p ^ (n + 1) -
        x.coeff (n + 1) * y.coeff 0 ^ p ^ (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    ⊢ Exists fun f => ∀ (x y : WittVector p k), Eq (f (WittVector.truncateFun (HAd …
  -/
  simp only [← peval_polyOfInterest']
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    ⊢ Exists fun f => ∀ (x y : WittVector p k), Eq (f (WittVector.truncateFun (HAd …
  -/
  obtain ⟨f₀, hf₀⟩ := exists_restrict_to_vars k (polyOfInterest_vars p n)
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f₀ : (↑(Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1)) …
    hf₀ : ∀ (x : Prod (Fin 2) Nat → k), Eq (f₀ (Function.comp x Subtype.val)) ((Mv …
    ⊢ Exists fun f => ∀ (x y : WittVector p k), Eq (f (WittVector.truncateFun (HAd …
  -/
  have : ∀ (a : Multiset (Fin 2)) (b : Multiset ℕ), a ×ˢ b = a.product b := fun a b => rfl
  let f : TruncatedWittVector p (n + 1) k → TruncatedWittVector p (n + 1) k → k := by
    intro x y
    apply f₀
    rintro ⟨a, ha⟩
    apply Function.uncurry ![x, y]
    simp_rw [product_val, this, range_val, Multiset.range_succ] at ha
    let S : Set (Fin 2 × ℕ) := (fun a => a.2 = n ∨ a.2 < n)
    have ha' : a ∈ S := by
      convert ha
      dsimp [S]
      congr!
      simp
    refine ⟨a.fst, ⟨a.snd, ?_⟩⟩
    cases' ha' with ha ha <;> omega
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f₀ : (↑(Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1)) …
    hf₀ : ∀ (x : Prod (Fin 2) Nat → k), Eq (f₀ (Function.comp x Subtype.val)) ((Mv …
    this : ∀ (a : Multiset (Fin 2)) (b : Multiset Nat), Eq (SProd.sprod a b) (a.pr …
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
      fun x y =>
        f₀ fun a =>
          Subtype.casesOn a fun a ha =>
            Function.uncurry (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
              (let S := fun a => Or (Eq a.2 n) (LT.lt a.2 n);
              letFun ⋯ fun ha' => { fst := a.1, snd := ⟨a.2, ⋯⟩ })
    ⊢ Exists fun f => ∀ (x y : WittVector p k), Eq (f (WittVector.truncateFun (HAd …
  -/
  use f
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f₀ : (↑(Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1)) …
    hf₀ : ∀ (x : Prod (Fin 2) Nat → k), Eq (f₀ (Function.comp x Subtype.val)) ((Mv …
    this : ∀ (a : Multiset (Fin 2)) (b : Multiset Nat), Eq (SProd.sprod a b) (a.pr …
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
      fun x y =>
        f₀ fun a =>
          Subtype.casesOn a fun a ha =>
            Function.uncurry (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
              (let S := fun a => Or (Eq a.2 n) (LT.lt a.2 n);
              letFun ⋯ fun ha' => { fst := a.1, snd := ⟨a.2, ⋯⟩ })
    ⊢ ∀ (x y : WittVector p k), Eq (f (WittVector.truncateFun (HAdd.hAdd n 1) x) ( …
  -/
  intro x y
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f₀ : (↑(Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1)) …
    hf₀ : ∀ (x : Prod (Fin 2) Nat → k), Eq (f₀ (Function.comp x Subtype.val)) ((Mv …
    this : ∀ (a : Multiset (Fin 2)) (b : Multiset Nat), Eq (SProd.sprod a b) (a.pr …
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
      fun x y =>
        f₀ fun a =>
          Subtype.casesOn a fun a ha =>
            Function.uncurry (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
              (let S := fun a => Or (Eq a.2 n) (LT.lt a.2 n);
              letFun ⋯ fun ha' => { fst := a.1, snd := ⟨a.2, ⋯⟩ })
    x y : WittVector p k
    ⊢ Eq (f (WittVector.truncateFun (HAdd.hAdd n 1) x) (WittVector.truncateFun (HA …
  -/
  dsimp [f, peval]
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f₀ : (↑(Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1)) …
    hf₀ : ∀ (x : Prod (Fin 2) Nat → k), Eq (f₀ (Function.comp x Subtype.val)) ((Mv …
    this : ∀ (a : Multiset (Fin 2)) (b : Multiset Nat), Eq (SProd.sprod a b) (a.pr …
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
      fun x y =>
        f₀ fun a =>
          Subtype.casesOn a fun a ha =>
            Function.uncurry (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
              (let S := fun a => Or (Eq a.2 n) (LT.lt a.2 n);
              letFun ⋯ fun ha' => { fst := a.1, snd := ⟨a.2, ⋯⟩ })
    x y : WittVector p k
    ⊢ Eq (f₀ fun a => Matrix.vecCons (WittVector.truncateFun (HAdd.hAdd n 1) x) (M …
  -/
  rw [← hf₀]
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f₀ : (↑(Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1)) …
    hf₀ : ∀ (x : Prod (Fin 2) Nat → k), Eq (f₀ (Function.comp x Subtype.val)) ((Mv …
    this : ∀ (a : Multiset (Fin 2)) (b : Multiset Nat), Eq (SProd.sprod a b) (a.pr …
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
      fun x y =>
        f₀ fun a =>
          Subtype.casesOn a fun a ha =>
            Function.uncurry (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
              (let S := fun a => Or (Eq a.2 n) (LT.lt a.2 n);
              letFun ⋯ fun ha' => { fst := a.1, snd := ⟨a.2, ⋯⟩ })
    x y : WittVector p k
    ⊢ Eq (f₀ fun a => Matrix.vecCons (WittVector.truncateFun (HAdd.hAdd n 1) x) (M …
  -/
  congr
  /-
    case h.e_a
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f₀ : (↑(Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1)) …
    hf₀ : ∀ (x : Prod (Fin 2) Nat → k), Eq (f₀ (Function.comp x Subtype.val)) ((Mv …
    this : ∀ (a : Multiset (Fin 2)) (b : Multiset Nat), Eq (SProd.sprod a b) (a.pr …
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
      fun x y =>
        f₀ fun a =>
          Subtype.casesOn a fun a ha =>
            Function.uncurry (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
              (let S := fun a => Or (Eq a.2 n) (LT.lt a.2 n);
              letFun ⋯ fun ha' => { fst := a.1, snd := ⟨a.2, ⋯⟩ })
    x y : WittVector p k
    ⊢ Eq (fun a => Matrix.vecCons (WittVector.truncateFun (HAdd.hAdd n 1) x) (Matr …
  -/
  ext a
  /-
    case h.e_a.h
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f₀ : (↑(Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1)) …
    hf₀ : ∀ (x : Prod (Fin 2) Nat → k), Eq (f₀ (Function.comp x Subtype.val)) ((Mv …
    this : ∀ (a : Multiset (Fin 2)) (b : Multiset Nat), Eq (SProd.sprod a b) (a.pr …
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
      fun x y =>
        f₀ fun a =>
          Subtype.casesOn a fun a ha =>
            Function.uncurry (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
              (let S := fun a => Or (Eq a.2 n) (LT.lt a.2 n);
              letFun ⋯ fun ha' => { fst := a.1, snd := ⟨a.2, ⋯⟩ })
    x y : WittVector p k
    a : ↑(Membership.mem (SProd.sprod Finset.univ.val (Multiset.range (HAdd.hAdd n …
    ⊢ Eq (Matrix.vecCons (WittVector.truncateFun (HAdd.hAdd n 1) x) (Matrix.vecCon …
  -/
  cases' a with a ha
  /-
    case h.e_a.h.mk
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f₀ : (↑(Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1)) …
    hf₀ : ∀ (x : Prod (Fin 2) Nat → k), Eq (f₀ (Function.comp x Subtype.val)) ((Mv …
    this : ∀ (a : Multiset (Fin 2)) (b : Multiset Nat), Eq (SProd.sprod a b) (a.pr …
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
      fun x y =>
        f₀ fun a =>
          Subtype.casesOn a fun a ha =>
            Function.uncurry (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
              (let S := fun a => Or (Eq a.2 n) (LT.lt a.2 n);
              letFun ⋯ fun ha' => { fst := a.1, snd := ⟨a.2, ⋯⟩ })
    x y : WittVector p k
    a : Prod (Fin 2) Nat
    ha : Membership.mem (Membership.mem (SProd.sprod Finset.univ.val (Multiset.ran …
    ⊢ Eq (Matrix.vecCons (WittVector.truncateFun (HAdd.hAdd n 1) x) (Matrix.vecCon …
  -/
  cases' a with i m
  /-
    case h.e_a.h.mk.mk
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f₀ : (↑(Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1)) …
    hf₀ : ∀ (x : Prod (Fin 2) Nat → k), Eq (f₀ (Function.comp x Subtype.val)) ((Mv …
    this : ∀ (a : Multiset (Fin 2)) (b : Multiset Nat), Eq (SProd.sprod a b) (a.pr …
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
      fun x y =>
        f₀ fun a =>
          Subtype.casesOn a fun a ha =>
            Function.uncurry (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
              (let S := fun a => Or (Eq a.2 n) (LT.lt a.2 n);
              letFun ⋯ fun ha' => { fst := a.1, snd := ⟨a.2, ⋯⟩ })
    x y : WittVector p k
    i : Fin 2
    m : Nat
    ha : Membership.mem (Membership.mem (SProd.sprod Finset.univ.val (Multiset.ran …
    ⊢ Eq (Matrix.vecCons (WittVector.truncateFun (HAdd.hAdd n 1) x) (Matrix.vecCon …
  -/
                  /-
                    🎉 no goals
                  -/
  fin_cases i <;> rfl -- surely this case split is not necessary
                  /-
                    🎉 no goals
                  -/


theorem nth_mul_coeff (n : ℕ) :
    ∃ f : TruncatedWittVector p (n + 1) k → TruncatedWittVector p (n + 1) k → k,
    ∀ x y : 𝕎 k, (x * y).coeff (n + 1) =
      x.coeff (n + 1) * y.coeff 0 ^ p ^ (n + 1) + y.coeff (n + 1) * x.coeff 0 ^ p ^ (n + 1) +
      f (truncateFun (n + 1) x) (truncateFun (n + 1) y) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    ⊢ Exists fun f => ∀ (x y : WittVector p k), Eq ((HMul.hMul x y).coeff (HAdd.hA …
  -/
  obtain ⟨f, hf⟩ := nth_mul_coeff' p k n
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
    hf : ∀ (x y : WittVector p k), Eq (f (WittVector.truncateFun (HAdd.hAdd n 1) x …
    ⊢ Exists fun f => ∀ (x y : WittVector p k), Eq ((HMul.hMul x y).coeff (HAdd.hA …
  -/
  use f
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
    hf : ∀ (x y : WittVector p k), Eq (f (WittVector.truncateFun (HAdd.hAdd n 1) x …
    ⊢ ∀ (x y : WittVector p k), Eq ((HMul.hMul x y).coeff (HAdd.hAdd n 1)) (HAdd.h …
  -/
  intro x y
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
    hf : ∀ (x y : WittVector p k), Eq (f (WittVector.truncateFun (HAdd.hAdd n 1) x …
    x y : WittVector p k
    ⊢ Eq ((HMul.hMul x y).coeff (HAdd.hAdd n 1)) (HAdd.hAdd (HAdd.hAdd (HMul.hMul  …
  -/
  rw [hf x y]
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : CommRing k
    inst✝ : CharP k p
    n : Nat
    f : TruncatedWittVector p (HAdd.hAdd n 1) k → TruncatedWittVector p (HAdd.hAdd …
    hf : ∀ (x y : WittVector p k), Eq (f (WittVector.truncateFun (HAdd.hAdd n 1) x …
    x y : WittVector p k
    ⊢ Eq ((HMul.hMul x y).coeff (HAdd.hAdd n 1)) (HAdd.hAdd (HAdd.hAdd (HMul.hMul  …
  -/
  ring
  /-
    🎉 no goals
  -/


/--
Produces the "remainder function" of the `n+1`st coefficient, which does not depend on the `n+1`st
coefficients of the inputs. -/
def nthRemainder (n : ℕ) : (Fin (n + 1) → k) → (Fin (n + 1) → k) → k :=
  Classical.choose (nth_mul_coeff p k n)


theorem nthRemainder_spec (n : ℕ) (x y : 𝕎 k) : (x * y).coeff (n + 1) =
    x.coeff (n + 1) * y.coeff 0 ^ p ^ (n + 1) + y.coeff (n + 1) * x.coeff 0 ^ p ^ (n + 1) +
    nthRemainder p n (truncateFun (n + 1) x) (truncateFun (n + 1) y) :=
  Classical.choose_spec (nth_mul_coeff p k n) _ _


