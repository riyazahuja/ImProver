/-- A variant of the traditional prime counting function which gives the number of primes
*strictly* less than the input. More convenient for avoiding off-by-one errors.

With `open scoped Nat.Prime`, this has notation `π'`. -/
def primeCounting' : ℕ → ℕ :=
  Nat.count Prime


/-- The prime counting function: Returns the number of primes less than or equal to the input.

With `open scoped Nat.Prime`, this has notation `π`. -/
def primeCounting (n : ℕ) : ℕ :=
  primeCounting' (n + 1)


@[inherit_doc] scoped[Nat.Prime] notation "π" => Nat.primeCounting


@[inherit_doc] scoped[Nat.Prime] notation "π'" => Nat.primeCounting'


@[simp]
theorem primeCounting_sub_one (n : ℕ) : π (n - 1) = π' n := by
  /-
    n : Nat
    ⊢ Eq (HSub.hSub n 1).primeCounting n.primeCounting'
  -/
              /-
                🎉 no goals
              -/
  cases n <;> rfl
              /-
                🎉 no goals
              -/


theorem monotone_primeCounting' : Monotone primeCounting' :=
  count_monotone Prime


theorem monotone_primeCounting : Monotone primeCounting :=
  monotone_primeCounting'.comp (monotone_id.add_const _)


@[simp]
theorem primeCounting'_nth_eq (n : ℕ) : π' (nth Prime n) = n :=
  count_nth_of_infinite infinite_setOf_prime _


/-- The `n`th prime is greater or equal to `n + 2`. -/
theorem add_two_le_nth_prime (n : ℕ) : n + 2 ≤ nth Prime n :=
  nth_prime_zero_eq_two ▸ (nth_strictMono infinite_setOf_prime).add_le_nat n 0


theorem surjective_primeCounting' : Function.Surjective π' :=
  Nat.surjective_count_of_infinite_setOf infinite_setOf_prime


theorem surjective_primeCounting : Function.Surjective π := by
  /-
    ⊢ Function.Surjective Nat.primeCounting
  -/
  suffices Function.Surjective (π ∘ fun n => n - 1) from this.of_comp
  /-
    ⊢ Function.Surjective (Function.comp Nat.primeCounting fun n => HSub.hSub n 1)
  -/
  convert surjective_primeCounting'
  /-
    case h.e'_3
    ⊢ Eq (Function.comp Nat.primeCounting fun n => HSub.hSub n 1) Nat.primeCounting'
  -/
  ext
  /-
    case h.e'_3.h
    x✝ : Nat
    ⊢ Eq (Function.comp Nat.primeCounting (fun n => HSub.hSub n 1) x✝) x✝.primeCou …
  -/
  exact primeCounting_sub_one _
  /-
    🎉 no goals
  -/


theorem tendsto_primeCounting' : Tendsto π' atTop atTop := by
  /-
    ⊢ Filter.Tendsto Nat.primeCounting' Filter.atTop Filter.atTop
  -/
  apply tendsto_atTop_atTop_of_monotone' monotone_primeCounting'
  /-
    ⊢ Not (BddAbove (Set.range Nat.primeCounting'))
  -/
  simp [Set.range_eq_univ.mpr surjective_primeCounting']
  /-
    🎉 no goals
  -/


theorem tensto_primeCounting : Tendsto π atTop atTop :=
  (tendsto_add_atTop_iff_nat 1).mpr tendsto_primeCounting'


@[simp]
theorem prime_nth_prime (n : ℕ) : Prime (nth Prime n) :=
  nth_mem_of_infinite infinite_setOf_prime _


@[simp]
lemma primeCounting'_eq_zero_iff {n : ℕ} : n.primeCounting' = 0 ↔ n ≤ 2 := by
  /-
    n : Nat
    ⊢ Iff (Eq n.primeCounting' 0) (LE.le n 2)
  -/
  rw [primeCounting', Nat.count_eq_zero ⟨_, Nat.prime_two⟩, Nat.nth_prime_zero_eq_two]
  /-
    🎉 no goals
  -/


@[simp]
lemma primeCounting_eq_zero_iff {n : ℕ} : n.primeCounting = 0 ↔ n ≤ 1 := by
  /-
    n : Nat
    ⊢ Iff (Eq n.primeCounting 0) (LE.le n 1)
  -/
  simp [primeCounting]
  /-
    🎉 no goals
  -/


@[simp]
lemma primeCounting_zero : primeCounting 0 = 0 :=
  primeCounting_eq_zero_iff.mpr zero_le_one


@[simp]
lemma primeCounting_one : primeCounting 1 = 0 :=
  primeCounting_eq_zero_iff.mpr le_rfl


/-- The cardinality of the finset `primesBelow n` equals the counting function
`primeCounting'` at `n`. -/
theorem primesBelow_card_eq_primeCounting' (n : ℕ) : #n.primesBelow = primeCounting' n := by
  /-
    n : Nat
    ⊢ Eq n.primesBelow.card n.primeCounting'
  -/
  simp only [primesBelow, primeCounting']
  /-
    n : Nat
    ⊢ Eq (Finset.filter (fun p => Nat.Prime p) (Finset.range n)).card (Nat.count N …
  -/
  exact (count_eq_card_filter_range Prime n).symm
  /-
    🎉 no goals
  -/


/-- A linear upper bound on the size of the `primeCounting'` function -/
theorem primeCounting'_add_le {a k : ℕ} (h0 : 0 < a) (h1 : a < k) (n : ℕ) :
    π' (k + n) ≤ π' k + Nat.totient a * (n / a + 1) :=
  calc
    π' (k + n) ≤ #{p ∈ range k | p.Prime} + #{p ∈ Ico k (k + n) | p.Prime} := by
      rw [primeCounting', count_eq_card_filter_range, range_eq_Ico, ←
        Ico_union_Ico_eq_Ico (zero_le k) le_self_add, filter_union]
      /-
        a k : Nat
        h0 : LT.lt 0 a
        h1 : LT.lt a k
        n : Nat
        ⊢ LE.le (Union.union (Finset.filter (fun x => Nat.Prime x) (Finset.Ico 0 k)) ( …
      -/
      apply card_union_le
      /-
        🎉 no goals
      -/
    _ ≤ π' k + #{p ∈ Ico k (k + n) | p.Prime} := by
      /-
        a k : Nat
        h0 : LT.lt 0 a
        h1 : LT.lt a k
        n : Nat
        ⊢ LE.le (HAdd.hAdd (Finset.filter (fun p => Nat.Prime p) (Finset.range k)).car …
      -/
      rw [primeCounting', count_eq_card_filter_range]
      /-
        🎉 no goals
      -/
    _ ≤ π' k + #{b ∈ Ico k (k + n) | a.Coprime b} := by
      /-
        a k : Nat
        h0 : LT.lt 0 a
        h1 : LT.lt a k
        n : Nat
        ⊢ LE.le (HAdd.hAdd k.primeCounting' (Finset.filter (fun p => Nat.Prime p) (Fin …
      -/
      refine add_le_add_left (card_le_card ?_) k.primeCounting'
      /-
        a k : Nat
        h0 : LT.lt 0 a
        h1 : LT.lt a k
        n : Nat
        ⊢ HasSubset.Subset (Finset.filter (fun p => Nat.Prime p) (Finset.Ico k (HAdd.h …
      -/
      simp only [subset_iff, and_imp, mem_filter, mem_Ico]
      /-
        a k : Nat
        h0 : LT.lt 0 a
        h1 : LT.lt a k
        n : Nat
        ⊢ ∀ ⦃x : Nat⦄, LE.le k x → LT.lt x (HAdd.hAdd k n) → Nat.Prime x → And (And (L …
      -/
      intro p succ_k_le_p p_lt_n p_prime
      /-
        a k : Nat
        h0 : LT.lt 0 a
        h1 : LT.lt a k
        n p : Nat
        succ_k_le_p : LE.le k p
        p_lt_n : LT.lt p (HAdd.hAdd k n)
        p_prime : Nat.Prime p
        ⊢ And (And (LE.le k p) (LT.lt p (HAdd.hAdd k n))) (a.Coprime p)
      -/
      constructor
        /-
          case left
          a k : Nat
          h0 : LT.lt 0 a
          h1 : LT.lt a k
          n p : Nat
          succ_k_le_p : LE.le k p
          p_lt_n : LT.lt p (HAdd.hAdd k n)
          p_prime : Nat.Prime p
          ⊢ And (LE.le k p) (LT.lt p (HAdd.hAdd k n))
        -/
      · exact ⟨succ_k_le_p, p_lt_n⟩
        /-
          🎉 no goals
        -/
        /-
          case right
          a k : Nat
          h0 : LT.lt 0 a
          h1 : LT.lt a k
          n p : Nat
          succ_k_le_p : LE.le k p
          p_lt_n : LT.lt p (HAdd.hAdd k n)
          p_prime : Nat.Prime p
          ⊢ a.Coprime p
        -/
      · rw [coprime_comm]
        /-
          case right
          a k : Nat
          h0 : LT.lt 0 a
          h1 : LT.lt a k
          n p : Nat
          succ_k_le_p : LE.le k p
          p_lt_n : LT.lt p (HAdd.hAdd k n)
          p_prime : Nat.Prime p
          ⊢ p.Coprime a
        -/
        exact coprime_of_lt_prime h0 (gt_of_ge_of_gt succ_k_le_p h1) p_prime
        /-
          🎉 no goals
        -/
    _ ≤ π' k + totient a * (n / a + 1) := by
      /-
        a k : Nat
        h0 : LT.lt 0 a
        h1 : LT.lt a k
        n : Nat
        ⊢ LE.le (HAdd.hAdd k.primeCounting' (Finset.filter (fun b => a.Coprime b) (Fin …
      -/
      rw [add_le_add_iff_left]
      /-
        a k : Nat
        h0 : LT.lt 0 a
        h1 : LT.lt a k
        n : Nat
        ⊢ LE.le (Finset.filter (fun b => a.Coprime b) (Finset.Ico k (HAdd.hAdd k n))). …
      -/
      exact Ico_filter_coprime_le k n h0
      /-
        🎉 no goals
      -/


