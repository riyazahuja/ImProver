/-- A version of `Nat.exists_infinite_primes` using the `Set.Infinite` predicate. -/
theorem infinite_setOf_prime : { p | Prime p }.Infinite :=
  Set.infinite_of_not_bddAbove not_bddAbove_setOf_prime


instance Primes.infinite : Infinite Primes := infinite_setOf_prime.to_subtype


instance Primes.countable : Countable Primes := ⟨⟨coeNat.coe, coe_nat_injective⟩⟩


/-- The prime factors of a natural number as a finset. -/
def primeFactors (n : ℕ) : Finset ℕ := n.primeFactorsList.toFinset


@[simp] lemma toFinset_factors (n : ℕ) : n.primeFactorsList.toFinset = n.primeFactors := rfl


@[simp] lemma mem_primeFactors : p ∈ n.primeFactors ↔ p.Prime ∧ p ∣ n ∧ n ≠ 0 := by
  /-
    n p : Nat
    ⊢ Iff (Membership.mem n.primeFactors p) (And (Nat.Prime p) (And (Dvd.dvd p n)  …
  -/
  simp_rw [← toFinset_factors, List.mem_toFinset, mem_primeFactorsList']
  /-
    🎉 no goals
  -/


lemma mem_primeFactors_of_ne_zero (hn : n ≠ 0) : p ∈ n.primeFactors ↔ p.Prime ∧ p ∣ n := by
  /-
    n p : Nat
    hn : Ne n 0
    ⊢ Iff (Membership.mem n.primeFactors p) (And (Nat.Prime p) (Dvd.dvd p n))
  -/
  simp [hn]
  /-
    🎉 no goals
  -/


lemma primeFactors_mono (hmn : m ∣ n) (hn : n ≠ 0) : primeFactors m ⊆ primeFactors n := by
  /-
    m n : Nat
    hmn : Dvd.dvd m n
    hn : Ne n 0
    ⊢ HasSubset.Subset m.primeFactors n.primeFactors
  -/
  simp only [subset_iff, mem_primeFactors, and_imp]
  /-
    m n : Nat
    hmn : Dvd.dvd m n
    hn : Ne n 0
    ⊢ ∀ ⦃x : Nat⦄, Nat.Prime x → Dvd.dvd x m → Ne m 0 → And (Nat.Prime x) (And (Dv …
  -/
  exact fun p hp hpm _ ↦ ⟨hp, hpm.trans hmn, hn⟩
  /-
    🎉 no goals
  -/


lemma mem_primeFactors_iff_mem_primeFactorsList : p ∈ n.primeFactors ↔ p ∈ n.primeFactorsList := by
  /-
    n p : Nat
    ⊢ Iff (Membership.mem n.primeFactors p) (Membership.mem n.primeFactorsList p)
  -/
  simp only [primeFactors, List.mem_toFinset]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-16")]
alias mem_primeFactors_iff_mem_factors := mem_primeFactors_iff_mem_primeFactorsList


lemma prime_of_mem_primeFactors (hp : p ∈ n.primeFactors) : p.Prime := (mem_primeFactors.1 hp).1

lemma dvd_of_mem_primeFactors (hp : p ∈ n.primeFactors) : p ∣ n := (mem_primeFactors.1 hp).2.1


lemma pos_of_mem_primeFactors (hp : p ∈ n.primeFactors) : 0 < p :=
  (prime_of_mem_primeFactors hp).pos


lemma le_of_mem_primeFactors (h : p ∈ n.primeFactors) : p ≤ n :=
  le_of_dvd (mem_primeFactors.1 h).2.2.bot_lt <| dvd_of_mem_primeFactors h


@[simp] lemma primeFactors_zero : primeFactors 0 = ∅ := by
  /-
    ⊢ Eq (Nat.primeFactors 0) EmptyCollection.emptyCollection
  -/
  ext
  /-
    case h
    a✝ : Nat
    ⊢ Iff (Membership.mem (Nat.primeFactors 0) a✝) (Membership.mem EmptyCollection …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp] lemma primeFactors_one : primeFactors 1 = ∅ := by
  /-
    ⊢ Eq (Nat.primeFactors 1) EmptyCollection.emptyCollection
  -/
  ext
  /-
    case h
    a✝ : Nat
    ⊢ Iff (Membership.mem (Nat.primeFactors 1) a✝) (Membership.mem EmptyCollection …
  -/
  simpa using Prime.ne_one
  /-
    🎉 no goals
  -/


@[simp] lemma primeFactors_eq_empty : n.primeFactors = ∅ ↔ n = 0 ∨ n = 1 := by
  /-
    n : Nat
    ⊢ Iff (Eq n.primeFactors EmptyCollection.emptyCollection) (Or (Eq n 0) (Eq n 1))
  -/
  constructor
    /-
      case mp
      n : Nat
      ⊢ Eq n.primeFactors EmptyCollection.emptyCollection → Or (Eq n 0) (Eq n 1)
    -/
  · contrapose!
    /-
      case mp
      n : Nat
      ⊢ And (Ne n 0) (Ne n 1) → Ne n.primeFactors EmptyCollection.emptyCollection
    -/
    rintro hn
    /-
      case mp
      n : Nat
      hn : And (Ne n 0) (Ne n 1)
      ⊢ Ne n.primeFactors EmptyCollection.emptyCollection
    -/
    obtain ⟨p, hp, hpn⟩ := exists_prime_and_dvd hn.2
    /-
      case mp.intro.intro
      n : Nat
      hn : And (Ne n 0) (Ne n 1)
      p : Nat
      hp : Nat.Prime p
      hpn : Dvd.dvd p n
      ⊢ Ne n.primeFactors EmptyCollection.emptyCollection
    -/
    exact Nonempty.ne_empty <| ⟨_, mem_primeFactors.2 ⟨hp, hpn, hn.1⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      ⊢ Or (Eq n 0) (Eq n 1) → Eq n.primeFactors EmptyCollection.emptyCollection
    -/
                           /-
                             🎉 no goals
                           -/
  · rintro (rfl | rfl) <;> simp
                           /-
                             🎉 no goals
                           -/


@[simp]
lemma nonempty_primeFactors {n : ℕ} : n.primeFactors.Nonempty ↔ 1 < n := by
  rw [← not_iff_not, Finset.not_nonempty_iff_eq_empty, primeFactors_eq_empty, not_lt,
    Nat.le_one_iff_eq_zero_or_eq_one]


@[simp] protected lemma Prime.primeFactors (hp : p.Prime) : p.primeFactors = {p} := by
  /-
    p : Nat
    hp : Nat.Prime p
    ⊢ Eq p.primeFactors (Singleton.singleton p)
  -/
  simp [Nat.primeFactors, primeFactorsList_prime hp]
  /-
    🎉 no goals
  -/


lemma primeFactors_mul (ha : a ≠ 0) (hb : b ≠ 0) :
    (a * b).primeFactors = a.primeFactors ∪ b.primeFactors := by
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HMul.hMul a b).primeFactors (Union.union a.primeFactors b.primeFactors)
  -/
  ext; simp only [Finset.mem_union, mem_primeFactors_iff_mem_primeFactorsList,
    mem_primeFactorsList_mul ha hb]


lemma Coprime.primeFactors_mul {a b : ℕ} (hab : Coprime a b) :
    (a * b).primeFactors = a.primeFactors ∪ b.primeFactors :=
  (List.toFinset.ext <| mem_primeFactorsList_mul_of_coprime hab).trans <| List.toFinset_union _ _


lemma primeFactors_gcd (ha : a ≠ 0) (hb : b ≠ 0) :
    (a.gcd b).primeFactors = a.primeFactors ∩ b.primeFactors := by
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (a.gcd b).primeFactors (Inter.inter a.primeFactors b.primeFactors)
  -/
  ext; simp [dvd_gcd_iff, ha, hb, gcd_ne_zero_left ha]; aesop
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp] lemma disjoint_primeFactors (ha : a ≠ 0) (hb : b ≠ 0) :
    Disjoint a.primeFactors b.primeFactors ↔ Coprime a b := by
  simp [disjoint_iff_inter_eq_empty, coprime_iff_gcd_eq_one, ← primeFactors_gcd, gcd_ne_zero_left,
    ha, hb]


protected lemma Coprime.disjoint_primeFactors (hab : Coprime a b) :
    Disjoint a.primeFactors b.primeFactors :=
  List.disjoint_toFinset_iff_disjoint.2 <| coprime_primeFactorsList_disjoint hab


lemma primeFactors_pow_succ (n k : ℕ) : (n ^ (k + 1)).primeFactors = n.primeFactors := by
  /-
    n k : Nat
    ⊢ Eq (HPow.hPow n (HAdd.hAdd k 1)).primeFactors n.primeFactors
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      k : Nat
      ⊢ Eq (HPow.hPow 0 (HAdd.hAdd k 1)).primeFactors (Nat.primeFactors 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n k : Nat
    hn : Ne n 0
    ⊢ Eq (HPow.hPow n (HAdd.hAdd k 1)).primeFactors n.primeFactors
  -/
  induction' k with k ih
    /-
      case inr.zero
      n : Nat
      hn : Ne n 0
      ⊢ Eq (HPow.hPow n (HAdd.hAdd 0 1)).primeFactors n.primeFactors
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.succ
      n : Nat
      hn : Ne n 0
      k : Nat
      ih : Eq (HPow.hPow n (HAdd.hAdd k 1)).primeFactors n.primeFactors
      ⊢ Eq (HPow.hPow n (HAdd.hAdd (HAdd.hAdd k 1) 1)).primeFactors n.primeFactors
    -/
  · rw [pow_succ', primeFactors_mul hn (pow_ne_zero _ hn), ih, Finset.union_idempotent]
    /-
      🎉 no goals
    -/


lemma primeFactors_pow (n : ℕ) (hk : k ≠ 0) : (n ^ k).primeFactors = n.primeFactors := by
  /-
    k n : Nat
    hk : Ne k 0
    ⊢ Eq (HPow.hPow n k).primeFactors n.primeFactors
  -/
  cases k
    /-
      case zero
      n : Nat
      hk : Ne 0 0
      ⊢ Eq (HPow.hPow n 0).primeFactors n.primeFactors
    -/
  · simp at hk
    /-
      🎉 no goals
    -/
  /-
    case succ
    n n✝ : Nat
    hk : Ne (HAdd.hAdd n✝ 1) 0
    ⊢ Eq (HPow.hPow n (HAdd.hAdd n✝ 1)).primeFactors n.primeFactors
  -/
  rw [primeFactors_pow_succ]
  /-
    🎉 no goals
  -/


/-- The only prime divisor of positive prime power `p^k` is `p` itself -/
lemma primeFactors_prime_pow (hk : k ≠ 0) (hp : Prime p) :
                                     /-
                                       k p : Nat
                                       hk : Ne k 0
                                       hp : Nat.Prime p
                                       ⊢ Eq (HPow.hPow p k).primeFactors (Singleton.singleton p)
                                     -/
    (p ^ k).primeFactors = {p} := by simp [primeFactors_pow p hk, hp]
                                     /-
                                       🎉 no goals
                                     -/


