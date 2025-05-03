theorem supDegree_eq_natDegree (p : R[X]) : p.toFinsupp.supDegree id = p.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (AddMonoidAlgebra.supDegree id p.toFinsupp) p.natDegree
  -/
  obtain rfl|h := eq_or_ne p 0
    /-
      case inl
      R : Type u
      inst✝ : Semiring R
      ⊢ Eq (AddMonoidAlgebra.supDegree id (Polynomial.toFinsupp 0)) (Polynomial.natD …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne p 0
    ⊢ Eq (AddMonoidAlgebra.supDegree id p.toFinsupp) p.natDegree
  -/
  apply WithBot.coe_injective
  rw [← AddMonoidAlgebra.supDegree_withBot_some_comp, Function.comp_id, supDegree_eq_degree,
    degree_eq_natDegree h, Nat.cast_withBot]
  /-
    case inr.a
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne p 0
    ⊢ p.toFinsupp.support.Nonempty
  -/
  rwa [support_toFinsupp, nonempty_iff_ne_empty, Ne, support_eq_empty]
  /-
    🎉 no goals
  -/


theorem le_natDegree_of_mem_supp (a : ℕ) : a ∈ p.support → a ≤ natDegree p :=
  le_natDegree_of_ne_zero ∘ mem_support_iff.mp


theorem supp_subset_range (h : natDegree p < m) : p.support ⊆ Finset.range m := fun _n hn =>
  mem_range.2 <| (le_natDegree_of_mem_supp _ hn).trans_lt h


theorem supp_subset_range_natDegree_succ : p.support ⊆ Finset.range (natDegree p + 1) :=
  supp_subset_range (Nat.lt_succ_self _)


theorem as_sum_support (p : R[X]) : p = ∑ i ∈ p.support, monomial i (p.coeff i) :=
  (sum_monomial_eq p).symm


theorem as_sum_support_C_mul_X_pow (p : R[X]) : p = ∑ i ∈ p.support, C (p.coeff i) * X ^ i :=
                                      /-
                                        R : Type u
                                        inst✝ : Semiring R
                                        p : Polynomial R
                                        ⊢ Eq (p.support.sum fun i => (Polynomial.monomial i) (p.coeff i)) (p.support.s …
                                      -/
  _root_.trans p.as_sum_support <| by simp only [C_mul_X_pow_eq_monomial]
                                      /-
                                        🎉 no goals
                                      -/


/-- We can reexpress a sum over `p.support` as a sum over `range n`,
for any `n` satisfying `p.natDegree < n`.
-/
theorem sum_over_range' [AddCommMonoid S] (p : R[X]) {f : ℕ → R → S} (h : ∀ n, f n 0 = 0) (n : ℕ)
    (w : p.natDegree < n) : p.sum f = ∑ a ∈ range n, f a (coeff p a) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : AddCommMonoid S
    p : Polynomial R
    f : Nat → R → S
    h : ∀ (n : Nat), Eq (f n 0) 0
    n : Nat
    w : LT.lt p.natDegree n
    ⊢ Eq (p.sum f) ((Finset.range n).sum fun a => f a (p.coeff a))
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : AddCommMonoid S
    f : Nat → R → S
    h : ∀ (n : Nat), Eq (f n 0) 0
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    w : LT.lt { toFinsupp := toFinsupp✝ }.natDegree n
    ⊢ Eq ({ toFinsupp := toFinsupp✝ }.sum f) ((Finset.range n).sum fun a => f a ({ …
  -/
  have := supp_subset_range w
  /-
    case ofFinsupp
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : AddCommMonoid S
    f : Nat → R → S
    h : ∀ (n : Nat), Eq (f n 0) 0
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    w : LT.lt { toFinsupp := toFinsupp✝ }.natDegree n
    this : HasSubset.Subset { toFinsupp := toFinsupp✝ }.support (Finset.range n)
    ⊢ Eq ({ toFinsupp := toFinsupp✝ }.sum f) ((Finset.range n).sum fun a => f a ({ …
  -/
  simp only [Polynomial.sum, support, coeff, natDegree, degree] at this ⊢
  /-
    case ofFinsupp
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : AddCommMonoid S
    f : Nat → R → S
    h : ∀ (n : Nat), Eq (f n 0) 0
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    w : LT.lt { toFinsupp := toFinsupp✝ }.natDegree n
    this : HasSubset.Subset toFinsupp✝.support (Finset.range n)
    ⊢ Eq (toFinsupp✝.support.sum fun x => f x (toFinsupp✝ x)) ((Finset.range n).su …
  -/
  exact Finsupp.sum_of_support_subset _ this _ fun n _hn => h n
  /-
    🎉 no goals
  -/


/-- We can reexpress a sum over `p.support` as a sum over `range (p.natDegree + 1)`.
-/
theorem sum_over_range [AddCommMonoid S] (p : R[X]) {f : ℕ → R → S} (h : ∀ n, f n 0 = 0) :
    p.sum f = ∑ a ∈ range (p.natDegree + 1), f a (coeff p a) :=
  sum_over_range' p h (p.natDegree + 1) (lt_add_one _)

-- TODO this is essentially a duplicate of `sum_over_range`, and should be removed.

theorem sum_fin [AddCommMonoid S] (f : ℕ → R → S) (hf : ∀ i, f i 0 = 0) {n : ℕ} {p : R[X]}
    (hn : p.degree < n) : (∑ i : Fin n, f i (p.coeff i)) = p.sum f := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : AddCommMonoid S
    f : Nat → R → S
    hf : ∀ (i : Nat), Eq (f i 0) 0
    n : Nat
    p : Polynomial R
    hn : LT.lt p.degree ↑n
    ⊢ Eq (Finset.univ.sum fun i => f (↑i) (p.coeff ↑i)) (p.sum f)
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : AddCommMonoid S
      f : Nat → R → S
      hf : ∀ (i : Nat), Eq (f i 0) 0
      n : Nat
      p : Polynomial R
      hn : LT.lt p.degree ↑n
      hp : Eq p 0
      ⊢ Eq (Finset.univ.sum fun i => f (↑i) (p.coeff ↑i)) (p.sum f)
    -/
  · rw [hp, sum_zero_index, Finset.sum_eq_zero]
    /-
      case pos
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : AddCommMonoid S
      f : Nat → R → S
      hf : ∀ (i : Nat), Eq (f i 0) 0
      n : Nat
      p : Polynomial R
      hn : LT.lt p.degree ↑n
      hp : Eq p 0
      ⊢ ∀ (x : Fin n), Membership.mem Finset.univ x → Eq (f (↑x) (Polynomial.coeff 0 …
    -/
    intro i _
    /-
      case pos
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : AddCommMonoid S
      f : Nat → R → S
      hf : ∀ (i : Nat), Eq (f i 0) 0
      n : Nat
      p : Polynomial R
      hn : LT.lt p.degree ↑n
      hp : Eq p 0
      i : Fin n
      a✝ : Membership.mem Finset.univ i
      ⊢ Eq (f (↑i) (Polynomial.coeff 0 ↑i)) 0
    -/
    exact hf i
    /-
      🎉 no goals
    -/
  rw [sum_over_range' _ hf n ((natDegree_lt_iff_degree_lt hp).mpr hn),
    Fin.sum_univ_eq_sum_range fun i => f i (p.coeff i)]


theorem as_sum_range' (p : R[X]) (n : ℕ) (w : p.natDegree < n) :
    p = ∑ i ∈ range n, monomial i (coeff p i) :=
  p.sum_monomial_eq.symm.trans <| p.sum_over_range' monomial_zero_right _ w


theorem as_sum_range (p : R[X]) : p = ∑ i ∈ range (p.natDegree + 1), monomial i (coeff p i) :=
  p.sum_monomial_eq.symm.trans <| p.sum_over_range <| monomial_zero_right


theorem as_sum_range_C_mul_X_pow (p : R[X]) :
    p = ∑ i ∈ range (p.natDegree + 1), C (coeff p i) * X ^ i :=
                             /-
                               R : Type u
                               inst✝ : Semiring R
                               p : Polynomial R
                               ⊢ Eq ((Finset.range (HAdd.hAdd p.natDegree 1)).sum fun i => (Polynomial.monomi …
                             -/
  p.as_sum_range.trans <| by simp only [C_mul_X_pow_eq_monomial]
                             /-
                               🎉 no goals
                             -/


theorem mem_support_C_mul_X_pow {n a : ℕ} {c : R} (h : a ∈ support (C c * X ^ n)) : a = n :=
  mem_singleton.1 <| support_C_mul_X_pow' n c h


theorem card_support_C_mul_X_pow_le_one {c : R} {n : ℕ} : #(support (C c * X ^ n)) ≤ 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    c : R
    n : Nat
    ⊢ LE.le (HMul.hMul (Polynomial.C c) (HPow.hPow Polynomial.X n)).support.card 1
  -/
  rw [← card_singleton n]
  /-
    R : Type u
    inst✝ : Semiring R
    c : R
    n : Nat
    ⊢ LE.le (HMul.hMul (Polynomial.C c) (HPow.hPow Polynomial.X n)).support.card ( …
  -/
  apply card_le_card (support_C_mul_X_pow' n c)
  /-
    🎉 no goals
  -/


theorem card_supp_le_succ_natDegree (p : R[X]) : #p.support ≤ p.natDegree + 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ LE.le p.support.card (HAdd.hAdd p.natDegree 1)
  -/
  rw [← Finset.card_range (p.natDegree + 1)]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ LE.le p.support.card (Finset.range (HAdd.hAdd p.natDegree 1)).card
  -/
  exact Finset.card_le_card supp_subset_range_natDegree_succ
  /-
    🎉 no goals
  -/


theorem le_degree_of_mem_supp (a : ℕ) : a ∈ p.support → ↑a ≤ degree p :=
  le_degree_of_ne_zero ∘ mem_support_iff.mp


theorem nonempty_support_iff : p.support.Nonempty ↔ p ≠ 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff p.support.Nonempty (Ne p 0)
  -/
  rw [Ne, nonempty_iff_ne_empty, Ne, ← support_eq_empty]
  /-
    🎉 no goals
  -/


theorem natDegree_mem_support_of_nonzero (H : p ≠ 0) : p.natDegree ∈ p.support := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    H : Ne p 0
    ⊢ Membership.mem p.support p.natDegree
  -/
  rw [mem_support_iff]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    H : Ne p 0
    ⊢ Ne (p.coeff p.natDegree) 0
  -/
  exact (not_congr leadingCoeff_eq_zero).mpr H
  /-
    🎉 no goals
  -/


theorem natDegree_eq_support_max' (h : p ≠ 0) :
    p.natDegree = p.support.max' (nonempty_support_iff.mpr h) :=
  (le_max' _ _ <| natDegree_mem_support_of_nonzero h).antisymm <|
    max'_le _ _ _ le_natDegree_of_mem_supp


