/-- `primesBelow n` is the set of primes less than `n` as a `Finset`. -/
def primesBelow (n : ℕ) : Finset ℕ := (Finset.range n).filter (fun p ↦ p.Prime)


@[simp]
lemma primesBelow_zero : primesBelow 0 = ∅ := by
  /-
    ⊢ Eq (Nat.primesBelow 0) EmptyCollection.emptyCollection
  -/
  rw [primesBelow, Finset.range_zero, Finset.filter_empty]
  /-
    🎉 no goals
  -/


lemma mem_primesBelow {k n : ℕ} :
                                              /-
                                                k n : Nat
                                                ⊢ Iff (Membership.mem k.primesBelow n) (And (LT.lt n k) (Nat.Prime n))
                                              -/
    n ∈ primesBelow k ↔ n < k ∧ n.Prime := by simp [primesBelow]
                                              /-
                                                🎉 no goals
                                              -/


lemma prime_of_mem_primesBelow {p n : ℕ} (h : p ∈ n.primesBelow) : p.Prime :=
  (Finset.mem_filter.mp h).2


lemma lt_of_mem_primesBelow {p n : ℕ} (h : p ∈ n.primesBelow) : p < n :=
  Finset.mem_range.mp <| Finset.mem_of_mem_filter p h


lemma primesBelow_succ (n : ℕ) :
    primesBelow n.succ = if n.Prime then insert n (primesBelow n) else primesBelow n := by
  /-
    n : Nat
    ⊢ Eq n.succ.primesBelow (ite (Nat.Prime n) (Insert.insert n n.primesBelow) n.p …
  -/
  rw [primesBelow, primesBelow, Finset.range_succ, Finset.filter_insert]
  /-
    🎉 no goals
  -/


lemma not_mem_primesBelow (n : ℕ) : n ∉ primesBelow n :=
  fun hn ↦ (lt_of_mem_primesBelow hn).false



/-- `factoredNumbers s`, for a finite set `s` of natural numbers, is the set of positive natural
numbers all of whose prime factors are in `s`. -/
def factoredNumbers (s : Finset ℕ) : Set ℕ := {m | m ≠ 0 ∧ ∀ p ∈ primeFactorsList m, p ∈ s}


lemma mem_factoredNumbers {s : Finset ℕ} {m : ℕ} :
    m ∈ factoredNumbers s ↔ m ≠ 0 ∧ ∀ p ∈ primeFactorsList m, p ∈ s :=
  Iff.rfl


/-- Membership in `Nat.factoredNumbers n` is decidable. -/
instance (s : Finset ℕ) : DecidablePred (· ∈ factoredNumbers s) :=
  inferInstanceAs <| DecidablePred fun x ↦ x ∈ {m | m ≠ 0 ∧ ∀ p ∈ primeFactorsList m, p ∈ s}


/-- A number that divides an `s`-factored number is itself `s`-factored. -/
lemma mem_factoredNumbers_of_dvd {s : Finset ℕ} {m k : ℕ} (h : m ∈ factoredNumbers s)
    (h' : k ∣ m) :
    k ∈ factoredNumbers s := by
  /-
    s : Finset Nat
    m k : Nat
    h : Membership.mem (Nat.factoredNumbers s) m
    h' : Dvd.dvd k m
    ⊢ Membership.mem (Nat.factoredNumbers s) k
  -/
  obtain ⟨h₁, h₂⟩ := h
  /-
    case intro
    s : Finset Nat
    m k : Nat
    h' : Dvd.dvd k m
    h₁ : Ne m 0
    h₂ : ∀ (p : Nat), Membership.mem m.primeFactorsList p → Membership.mem s p
    ⊢ Membership.mem (Nat.factoredNumbers s) k
  -/
  have hk := ne_zero_of_dvd_ne_zero h₁ h'
  /-
    case intro
    s : Finset Nat
    m k : Nat
    h' : Dvd.dvd k m
    h₁ : Ne m 0
    h₂ : ∀ (p : Nat), Membership.mem m.primeFactorsList p → Membership.mem s p
    hk : Ne k 0
    ⊢ Membership.mem (Nat.factoredNumbers s) k
  -/
  refine ⟨hk, fun p hp ↦ h₂ p ?_⟩
  /-
    case intro
    s : Finset Nat
    m k : Nat
    h' : Dvd.dvd k m
    h₁ : Ne m 0
    h₂ : ∀ (p : Nat), Membership.mem m.primeFactorsList p → Membership.mem s p
    hk : Ne k 0
    p : Nat
    hp : Membership.mem k.primeFactorsList p
    ⊢ Membership.mem m.primeFactorsList p
  -/
  rw [mem_primeFactorsList <| by assumption] at hp ⊢
  /-
    case intro
    s : Finset Nat
    m k : Nat
    h' : Dvd.dvd k m
    h₁ : Ne m 0
    h₂ : ∀ (p : Nat), Membership.mem m.primeFactorsList p → Membership.mem s p
    hk : Ne k 0
    p : Nat
    hp : And (Nat.Prime p) (Dvd.dvd p k)
    ⊢ And (Nat.Prime p) (Dvd.dvd p m)
  -/
  exact ⟨hp.1, hp.2.trans h'⟩
  /-
    🎉 no goals
  -/


/-- `m` is `s`-factored if and only if `m` is nonzero and all prime divisors `≤ m` of `m`
are in `s`. -/
lemma mem_factoredNumbers_iff_forall_le {s : Finset ℕ} {m : ℕ} :
    m ∈ factoredNumbers s ↔ m ≠ 0 ∧ ∀ p ≤ m, p.Prime → p ∣ m → p ∈ s := by
  /-
    s : Finset Nat
    m : Nat
    ⊢ Iff (Membership.mem (Nat.factoredNumbers s) m) (And (Ne m 0) (∀ (p : Nat), L …
  -/
  simp_rw [mem_factoredNumbers, mem_primeFactorsList']
  exact ⟨fun ⟨H₀, H₁⟩ ↦ ⟨H₀, fun p _ hp₂ hp₃ ↦ H₁ p ⟨hp₂, hp₃, H₀⟩⟩,
    fun ⟨H₀, H₁⟩ ↦
      ⟨H₀, fun p ⟨hp₁, hp₂, hp₃⟩ ↦ H₁ p (le_of_dvd (Nat.pos_of_ne_zero hp₃) hp₂) hp₁ hp₂⟩⟩


/-- `m` is `s`-factored if and only if all prime divisors of `m` are in `s`. -/
lemma mem_factoredNumbers' {s : Finset ℕ} {m : ℕ} :
    m ∈ factoredNumbers s ↔ ∀ p, p.Prime → p ∣ m → p ∈ s := by
  /-
    s : Finset Nat
    m : Nat
    ⊢ Iff (Membership.mem (Nat.factoredNumbers s) m) (∀ (p : Nat), Nat.Prime p → D …
  -/
  obtain ⟨p, hp₁, hp₂⟩ := exists_infinite_primes (1 + Finset.sup s id)
  /-
    case intro.intro
    s : Finset Nat
    m p : Nat
    hp₁ : LE.le (HAdd.hAdd 1 (s.sup id)) p
    hp₂ : Nat.Prime p
    ⊢ Iff (Membership.mem (Nat.factoredNumbers s) m) (∀ (p : Nat), Nat.Prime p → D …
  -/
  rw [mem_factoredNumbers_iff_forall_le]
  refine ⟨fun ⟨H₀, H₁⟩ ↦ fun p hp₁ hp₂ ↦ H₁ p (le_of_dvd (Nat.pos_of_ne_zero H₀) hp₂) hp₁ hp₂,
         fun H ↦ ⟨fun h ↦ lt_irrefl p ?_, fun p _ ↦ H p⟩⟩
  calc
    p ≤ s.sup id := Finset.le_sup (f := @id ℕ) <| H p hp₂ <| h.symm ▸ dvd_zero p
    _ < 1 + s.sup id := lt_one_add _
    _ ≤ p := hp₁


lemma ne_zero_of_mem_factoredNumbers {s : Finset ℕ} {m : ℕ} (h : m ∈ factoredNumbers s) : m ≠ 0 :=
  h.1


/-- The `Finset` of prime factors of an `s`-factored number is contained in `s`. -/
lemma primeFactors_subset_of_mem_factoredNumbers {s : Finset ℕ} {m : ℕ}
    (hm : m ∈ factoredNumbers s) :
    m.primeFactors ⊆ s := by
  /-
    s : Finset Nat
    m : Nat
    hm : Membership.mem (Nat.factoredNumbers s) m
    ⊢ HasSubset.Subset m.primeFactors s
  -/
  rw [mem_factoredNumbers] at hm
  /-
    s : Finset Nat
    m : Nat
    hm : And (Ne m 0) (∀ (p : Nat), Membership.mem m.primeFactorsList p → Membersh …
    ⊢ HasSubset.Subset m.primeFactors s
  -/
  exact fun n hn ↦ hm.2 n (mem_primeFactors_iff_mem_primeFactorsList.mp hn)
  /-
    🎉 no goals
  -/


/-- If `m ≠ 0` and the `Finset` of prime factors of `m` is contained in `s`, then `m`
is `s`-factored. -/
lemma mem_factoredNumbers_of_primeFactors_subset {s : Finset ℕ} {m : ℕ} (hm : m ≠ 0)
    (hp : m.primeFactors ⊆ s) :
    m ∈ factoredNumbers s := by
  /-
    s : Finset Nat
    m : Nat
    hm : Ne m 0
    hp : HasSubset.Subset m.primeFactors s
    ⊢ Membership.mem (Nat.factoredNumbers s) m
  -/
  rw [mem_factoredNumbers]
  /-
    s : Finset Nat
    m : Nat
    hm : Ne m 0
    hp : HasSubset.Subset m.primeFactors s
    ⊢ And (Ne m 0) (∀ (p : Nat), Membership.mem m.primeFactorsList p → Membership. …
  -/
  exact ⟨hm, fun p hp' ↦ hp <| mem_primeFactors_iff_mem_primeFactorsList.mpr hp'⟩
  /-
    🎉 no goals
  -/


/-- `m` is `s`-factored if and only if `m ≠ 0` and its `Finset` of prime factors
is contained in `s`. -/
lemma mem_factoredNumbers_iff_primeFactors_subset {s : Finset ℕ} {m : ℕ} :
    m ∈ factoredNumbers s ↔ m ≠ 0 ∧ m.primeFactors ⊆ s :=
  ⟨fun h ↦ ⟨ne_zero_of_mem_factoredNumbers h, primeFactors_subset_of_mem_factoredNumbers h⟩,
   fun ⟨h₁, h₂⟩ ↦ mem_factoredNumbers_of_primeFactors_subset h₁ h₂⟩


@[simp]
lemma factoredNumbers_empty : factoredNumbers ∅ = {1} := by
  /-
    ⊢ Eq (Nat.factoredNumbers EmptyCollection.emptyCollection) (Singleton.singleto …
  -/
  ext m
  simp only [mem_factoredNumbers, Finset.not_mem_empty, ← List.eq_nil_iff_forall_not_mem,
    primeFactorsList_eq_nil, and_or_left, not_and_self_iff, ne_and_eq_iff_right zero_ne_one,
    false_or, Set.mem_singleton_iff]


/-- The product of two `s`-factored numbers is again `s`-factored. -/
lemma mul_mem_factoredNumbers {s : Finset ℕ} {m n : ℕ} (hm : m ∈ factoredNumbers s)
    (hn : n ∈ factoredNumbers s) :
    m * n ∈ factoredNumbers s := by
  /-
    s : Finset Nat
    m n : Nat
    hm : Membership.mem (Nat.factoredNumbers s) m
    hn : Membership.mem (Nat.factoredNumbers s) n
    ⊢ Membership.mem (Nat.factoredNumbers s) (HMul.hMul m n)
  -/
  have hm' := primeFactors_subset_of_mem_factoredNumbers hm
  /-
    s : Finset Nat
    m n : Nat
    hm : Membership.mem (Nat.factoredNumbers s) m
    hn : Membership.mem (Nat.factoredNumbers s) n
    hm' : HasSubset.Subset m.primeFactors s
    ⊢ Membership.mem (Nat.factoredNumbers s) (HMul.hMul m n)
  -/
  have hn' := primeFactors_subset_of_mem_factoredNumbers hn
  exact mem_factoredNumbers_of_primeFactors_subset (mul_ne_zero hm.1 hn.1)
    <| primeFactors_mul hm.1 hn.1 ▸ Finset.union_subset hm' hn'


/-- The product of the prime factors of `n` that are in `s` is an `s`-factored number. -/
lemma prod_mem_factoredNumbers (s : Finset ℕ) (n : ℕ) :
    (n.primeFactorsList.filter (· ∈ s)).prod ∈ factoredNumbers s := by
  have h₀ : (n.primeFactorsList.filter (· ∈ s)).prod ≠ 0 :=
    List.prod_ne_zero fun h ↦ (pos_of_mem_primeFactorsList (List.mem_of_mem_filter h)).false
  /-
    s : Finset Nat
    n : Nat
    h₀ : Ne (List.filter (fun x => Decidable.decide (Membership.mem s x)) n.primeF …
    ⊢ Membership.mem (Nat.factoredNumbers s) (List.filter (fun x => Decidable.deci …
  -/
  refine ⟨h₀, fun p hp ↦ ?_⟩
  /-
    s : Finset Nat
    n : Nat
    h₀ : Ne (List.filter (fun x => Decidable.decide (Membership.mem s x)) n.primeF …
    p : Nat
    hp : Membership.mem (List.filter (fun x => Decidable.decide (Membership.mem s  …
    ⊢ Membership.mem s p
  -/
  obtain ⟨H₁, H₂⟩ := (mem_primeFactorsList h₀).mp hp
  simpa only [decide_eq_true_eq] using List.of_mem_filter <| mem_list_primes_of_dvd_prod H₁.prime
    (fun _ hq ↦ (prime_of_mem_primeFactorsList (List.mem_of_mem_filter hq)).prime) H₂


/-- The sets of `s`-factored and of `s ∪ {N}`-factored numbers are the same when `N` is not prime.
See `Nat.equivProdNatFactoredNumbers` for when `N` is prime. -/
lemma factoredNumbers_insert (s : Finset ℕ) {N : ℕ} (hN : ¬ N.Prime) :
    factoredNumbers (insert N s) = factoredNumbers s := by
  /-
    s : Finset Nat
    N : Nat
    hN : Not (Nat.Prime N)
    ⊢ Eq (Nat.factoredNumbers (Insert.insert N s)) (Nat.factoredNumbers s)
  -/
  ext m
  refine ⟨fun hm ↦ ⟨hm.1, fun p hp ↦ ?_⟩,
          fun hm ↦ ⟨hm.1, fun p hp ↦ Finset.mem_insert_of_mem <| hm.2 p hp⟩⟩
  exact Finset.mem_of_mem_insert_of_ne (hm.2 p hp)
    fun h ↦ hN <| h ▸ prime_of_mem_primeFactorsList hp


@[gcongr] lemma factoredNumbers_mono {s t : Finset ℕ} (hst : s ≤ t) :
    factoredNumbers s ⊆ factoredNumbers t :=
  fun _ hx ↦ ⟨hx.1, fun p hp ↦ hst <| hx.2 p hp⟩


/-- The non-zero non-`s`-factored numbers are `≥ N` when `s` contains all primes less than `N`. -/
lemma factoredNumbers_compl {N : ℕ} {s : Finset ℕ} (h : primesBelow N ≤ s) :
    (factoredNumbers s)ᶜ \ {0} ⊆ {n | N ≤ n} := by
  /-
    N : Nat
    s : Finset Nat
    h : LE.le N.primesBelow s
    ⊢ HasSubset.Subset (SDiff.sdiff (HasCompl.compl (Nat.factoredNumbers s)) (Sing …
  -/
  intro n hn
  simp only [Set.mem_compl_iff, mem_factoredNumbers, Set.mem_diff, ne_eq, not_and, not_forall,
    not_lt, exists_prop, Set.mem_singleton_iff] at hn
  /-
    N : Nat
    s : Finset Nat
    h : LE.le N.primesBelow s
    n : Nat
    hn : And (Not (Eq n 0) → Exists fun x => And (Membership.mem n.primeFactorsLis …
    ⊢ Membership.mem (setOf fun n => LE.le N n) n
  -/
  simp only [Set.mem_setOf_eq]
  /-
    N : Nat
    s : Finset Nat
    h : LE.le N.primesBelow s
    n : Nat
    hn : And (Not (Eq n 0) → Exists fun x => And (Membership.mem n.primeFactorsLis …
    ⊢ LE.le N n
  -/
  obtain ⟨p, hp₁, hp₂⟩ := hn.1 hn.2
  have : N ≤ p := by
    contrapose! hp₂
    exact h <| mem_primesBelow.mpr ⟨hp₂, prime_of_mem_primeFactorsList hp₁⟩
  /-
    case intro.intro
    N : Nat
    s : Finset Nat
    h : LE.le N.primesBelow s
    n : Nat
    hn : And (Not (Eq n 0) → Exists fun x => And (Membership.mem n.primeFactorsLis …
    p : Nat
    hp₁ : Membership.mem n.primeFactorsList p
    hp₂ : Not (Membership.mem s p)
    this : LE.le N p
    ⊢ LE.le N n
  -/
  exact this.trans <| le_of_mem_primeFactorsList hp₁
  /-
    🎉 no goals
  -/


/-- If `p` is a prime and `n` is `s`-factored, then every product `p^e * n`
is `s ∪ {p}`-factored. -/
lemma pow_mul_mem_factoredNumbers {s : Finset ℕ} {p n : ℕ} (hp : p.Prime) (e : ℕ)
    (hn : n ∈ factoredNumbers s) :
    p ^ e * n ∈ factoredNumbers (insert p s) := by
  /-
    s : Finset Nat
    p n : Nat
    hp : Nat.Prime p
    e : Nat
    hn : Membership.mem (Nat.factoredNumbers s) n
    ⊢ Membership.mem (Nat.factoredNumbers (Insert.insert p s)) (HMul.hMul (HPow.hP …
  -/
  have hp' := pow_ne_zero e hp.ne_zero
  /-
    s : Finset Nat
    p n : Nat
    hp : Nat.Prime p
    e : Nat
    hn : Membership.mem (Nat.factoredNumbers s) n
    hp' : Ne (HPow.hPow p e) 0
    ⊢ Membership.mem (Nat.factoredNumbers (Insert.insert p s)) (HMul.hMul (HPow.hP …
  -/
  refine ⟨mul_ne_zero hp' hn.1, fun q hq ↦ ?_⟩
  /-
    s : Finset Nat
    p n : Nat
    hp : Nat.Prime p
    e : Nat
    hn : Membership.mem (Nat.factoredNumbers s) n
    hp' : Ne (HPow.hPow p e) 0
    q : Nat
    hq : Membership.mem (HMul.hMul (HPow.hPow p e) n).primeFactorsList q
    ⊢ Membership.mem (Insert.insert p s) q
  -/
  rcases (mem_primeFactorsList_mul hp' hn.1).mp hq with H | H
    /-
      case inl
      s : Finset Nat
      p n : Nat
      hp : Nat.Prime p
      e : Nat
      hn : Membership.mem (Nat.factoredNumbers s) n
      hp' : Ne (HPow.hPow p e) 0
      q : Nat
      hq : Membership.mem (HMul.hMul (HPow.hPow p e) n).primeFactorsList q
      H : Membership.mem (HPow.hPow p e).primeFactorsList q
      ⊢ Membership.mem (Insert.insert p s) q
    -/
  · rw [mem_primeFactorsList hp'] at H
    /-
      case inl
      s : Finset Nat
      p n : Nat
      hp : Nat.Prime p
      e : Nat
      hn : Membership.mem (Nat.factoredNumbers s) n
      hp' : Ne (HPow.hPow p e) 0
      q : Nat
      hq : Membership.mem (HMul.hMul (HPow.hPow p e) n).primeFactorsList q
      H : And (Nat.Prime q) (Dvd.dvd q (HPow.hPow p e))
      ⊢ Membership.mem (Insert.insert p s) q
    -/
    rw [(prime_dvd_prime_iff_eq H.1 hp).mp <| H.1.dvd_of_dvd_pow H.2]
    /-
      case inl
      s : Finset Nat
      p n : Nat
      hp : Nat.Prime p
      e : Nat
      hn : Membership.mem (Nat.factoredNumbers s) n
      hp' : Ne (HPow.hPow p e) 0
      q : Nat
      hq : Membership.mem (HMul.hMul (HPow.hPow p e) n).primeFactorsList q
      H : And (Nat.Prime q) (Dvd.dvd q (HPow.hPow p e))
      ⊢ Membership.mem (Insert.insert p s) p
    -/
    exact Finset.mem_insert_self p s
    /-
      🎉 no goals
    -/
    /-
      case inr
      s : Finset Nat
      p n : Nat
      hp : Nat.Prime p
      e : Nat
      hn : Membership.mem (Nat.factoredNumbers s) n
      hp' : Ne (HPow.hPow p e) 0
      q : Nat
      hq : Membership.mem (HMul.hMul (HPow.hPow p e) n).primeFactorsList q
      H : Membership.mem n.primeFactorsList q
      ⊢ Membership.mem (Insert.insert p s) q
    -/
  · exact Finset.mem_insert_of_mem <| hn.2 _ H
    /-
      🎉 no goals
    -/


/-- If `p ∉ s` is a prime and `n` is `s`-factored, then `p` and `n` are coprime. -/
lemma Prime.factoredNumbers_coprime {s : Finset ℕ} {p n : ℕ} (hp : p.Prime) (hs : p ∉ s)
    (hn : n ∈ factoredNumbers s) :
    Nat.Coprime p n := by
  /-
    s : Finset Nat
    p n : Nat
    hp : Nat.Prime p
    hs : Not (Membership.mem s p)
    hn : Membership.mem (Nat.factoredNumbers s) n
    ⊢ p.Coprime n
  -/
  rw [hp.coprime_iff_not_dvd, ← mem_primeFactorsList_iff_dvd hn.1 hp]
  /-
    s : Finset Nat
    p n : Nat
    hp : Nat.Prime p
    hs : Not (Membership.mem s p)
    hn : Membership.mem (Nat.factoredNumbers s) n
    ⊢ Not (Membership.mem n.primeFactorsList p)
  -/
  exact fun H ↦ hs <| hn.2 p H
  /-
    🎉 no goals
  -/


/-- If `f : ℕ → F` is multiplicative on coprime arguments, `p ∉ s` is a prime and `m`
is `s`-factored, then `f (p^e * m) = f (p^e) * f m`. -/
lemma factoredNumbers.map_prime_pow_mul {F : Type*} [CommSemiring F] {f : ℕ → F}
    (hmul : ∀ {m n}, Coprime m n → f (m * n) = f m * f n) {s : Finset ℕ} {p : ℕ}
    (hp : p.Prime) (hs : p ∉ s) (e : ℕ) {m : factoredNumbers s} :
    f (p ^ e * m) = f (p ^ e) * f m :=
  hmul <| Coprime.pow_left _ <| hp.factoredNumbers_coprime hs <| Subtype.mem m


open List Perm in
/-- We establish the bijection from `ℕ × factoredNumbers s` to `factoredNumbers (s ∪ {p})`
given by `(e, n) ↦ p^e * n` when `p ∉ s` is a prime. See `Nat.factoredNumbers_insert` for
when `p` is not prime. -/
def equivProdNatFactoredNumbers {s : Finset ℕ} {p : ℕ} (hp : p.Prime) (hs : p ∉ s) :
    ℕ × factoredNumbers s ≃ factoredNumbers (insert p s) where
  toFun := fun ⟨e, n⟩ ↦ ⟨p ^ e * n, pow_mul_mem_factoredNumbers hp e n.2⟩
  invFun := fun ⟨m, _⟩  ↦ (m.factorization p,
                            ⟨(m.primeFactorsList.filter (· ∈ s)).prod, prod_mem_factoredNumbers ..⟩)
  left_inv := by
    /-
      s : Finset Nat
      p : Nat
      hp : Nat.Prime p
      hs : Not (Membership.mem s p)
      ⊢ Function.LeftInverse (fun x => Nat.equivProdNatFactoredNumbers.match_2 (fun  …
    -/
    rintro ⟨e, m, hm₀, hm⟩
    simp (config := { etaStruct := .all }) only
      [Set.coe_setOf, Set.mem_setOf_eq, Prod.mk.injEq, Subtype.mk.injEq]
    /-
      case mk.mk.intro
      s : Finset Nat
      p : Nat
      hp : Nat.Prime p
      hs : Not (Membership.mem s p)
      e m : Nat
      hm₀ : Ne m 0
      hm : ∀ (p : Nat), Membership.mem m.primeFactorsList p → Membership.mem s p
      ⊢ And (Eq ((HMul.hMul (HPow.hPow p e) m).factorization p) e) (Eq (List.filter  …
    -/
    constructor
      /-
        case mk.mk.intro.left
        s : Finset Nat
        p : Nat
        hp : Nat.Prime p
        hs : Not (Membership.mem s p)
        e m : Nat
        hm₀ : Ne m 0
        hm : ∀ (p : Nat), Membership.mem m.primeFactorsList p → Membership.mem s p
        ⊢ Eq ((HMul.hMul (HPow.hPow p e) m).factorization p) e
      -/
    · rw [factorization_mul (pos_iff_ne_zero.mp <| pos_pow_of_pos e hp.pos) hm₀]
      simp only [factorization_pow, Finsupp.coe_add, Finsupp.coe_smul, nsmul_eq_mul,
        Pi.natCast_def, cast_id, Pi.add_apply, Pi.mul_apply, hp.factorization_self,
        mul_one, add_right_eq_self]
      /-
        case mk.mk.intro.left
        s : Finset Nat
        p : Nat
        hp : Nat.Prime p
        hs : Not (Membership.mem s p)
        e m : Nat
        hm₀ : Ne m 0
        hm : ∀ (p : Nat), Membership.mem m.primeFactorsList p → Membership.mem s p
        ⊢ Eq (m.factorization p) 0
      -/
      rw [← primeFactorsList_count_eq, count_eq_zero]
      /-
        case mk.mk.intro.left
        s : Finset Nat
        p : Nat
        hp : Nat.Prime p
        hs : Not (Membership.mem s p)
        e m : Nat
        hm₀ : Ne m 0
        hm : ∀ (p : Nat), Membership.mem m.primeFactorsList p → Membership.mem s p
        ⊢ Not (Membership.mem m.primeFactorsList p)
      -/
      exact fun H ↦ hs (hm p H)
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.intro.right
        s : Finset Nat
        p : Nat
        hp : Nat.Prime p
        hs : Not (Membership.mem s p)
        e m : Nat
        hm₀ : Ne m 0
        hm : ∀ (p : Nat), Membership.mem m.primeFactorsList p → Membership.mem s p
        ⊢ Eq (List.filter (fun x => Decidable.decide (Membership.mem s x)) (HMul.hMul  …
      -/
    · nth_rewrite 2 [← prod_primeFactorsList hm₀]
      refine prod_eq <|
        (filter _ <| perm_primeFactorsList_mul (pow_ne_zero e hp.ne_zero) hm₀).trans ?_
      rw [filter_append, hp.primeFactorsList_pow,
          filter_eq_nil_iff.mpr fun q hq ↦ by rw [mem_replicate] at hq; simp [hq.2, hs],
          nil_append, filter_eq_self.mpr fun q hq ↦ by simp only [hm q hq, decide_true]]
  right_inv := by
    /-
      s : Finset Nat
      p : Nat
      hp : Nat.Prime p
      hs : Not (Membership.mem s p)
      ⊢ Function.RightInverse (fun x => Nat.equivProdNatFactoredNumbers.match_2 (fun …
    -/
    rintro ⟨m, hm₀, hm⟩
    /-
      case mk.intro
      s : Finset Nat
      p : Nat
      hp : Nat.Prime p
      hs : Not (Membership.mem s p)
      m : Nat
      hm₀ : Ne m 0
      hm : ∀ (p_1 : Nat), Membership.mem m.primeFactorsList p_1 → Membership.mem (In …
      ⊢ Eq ((fun x => Nat.equivProdNatFactoredNumbers.match_1 (fun x => ↑(Nat.factor …
    -/
    simp only [Set.coe_setOf, Set.mem_setOf_eq, Subtype.mk.injEq]
    /-
      case mk.intro
      s : Finset Nat
      p : Nat
      hp : Nat.Prime p
      hs : Not (Membership.mem s p)
      m : Nat
      hm₀ : Ne m 0
      hm : ∀ (p_1 : Nat), Membership.mem m.primeFactorsList p_1 → Membership.mem (In …
      ⊢ Eq (HMul.hMul (HPow.hPow p (m.factorization p)) (List.filter (fun x => Decid …
    -/
    rw [← primeFactorsList_count_eq, ← prod_replicate, ← prod_append]
    /-
      case mk.intro
      s : Finset Nat
      p : Nat
      hp : Nat.Prime p
      hs : Not (Membership.mem s p)
      m : Nat
      hm₀ : Ne m 0
      hm : ∀ (p_1 : Nat), Membership.mem m.primeFactorsList p_1 → Membership.mem (In …
      ⊢ Eq (HAppend.hAppend (List.replicate (List.count p m.primeFactorsList) p) (Li …
    -/
    nth_rewrite 3 [← prod_primeFactorsList hm₀]
    have : m.primeFactorsList.filter (· = p) = m.primeFactorsList.filter (¬ · ∈ s) := by
      refine (filter_congr fun q hq ↦ ?_).symm
      simp only [decide_not, Bool.not_eq_true', decide_eq_false_iff_not, decide_eq_true_eq]
      rcases Finset.mem_insert.mp <| hm _ hq with h | h
      · simp only [h, hs, decide_false, Bool.not_false, decide_true]
      · simp only [h, decide_true, Bool.not_true, false_eq_decide_iff]
        exact fun H ↦ hs <| H ▸ h
    /-
      case mk.intro
      s : Finset Nat
      p : Nat
      hp : Nat.Prime p
      hs : Not (Membership.mem s p)
      m : Nat
      hm₀ : Ne m 0
      hm : ∀ (p_1 : Nat), Membership.mem m.primeFactorsList p_1 → Membership.mem (In …
      this : Eq (List.filter (fun x => Decidable.decide (Eq x p)) m.primeFactorsList …
      ⊢ Eq (HAppend.hAppend (List.replicate (List.count p m.primeFactorsList) p) (Li …
    -/
    refine prod_eq <| (filter_eq m.primeFactorsList p).symm ▸ this ▸ perm_append_comm.trans ?_
    /-
      case mk.intro
      s : Finset Nat
      p : Nat
      hp : Nat.Prime p
      hs : Not (Membership.mem s p)
      m : Nat
      hm₀ : Ne m 0
      hm : ∀ (p_1 : Nat), Membership.mem m.primeFactorsList p_1 → Membership.mem (In …
      this : Eq (List.filter (fun x => Decidable.decide (Eq x p)) m.primeFactorsList …
      ⊢ (HAppend.hAppend (List.filter (fun x => Decidable.decide (Membership.mem s x …
    -/
    simp only [decide_not]
    /-
      case mk.intro
      s : Finset Nat
      p : Nat
      hp : Nat.Prime p
      hs : Not (Membership.mem s p)
      m : Nat
      hm₀ : Ne m 0
      hm : ∀ (p_1 : Nat), Membership.mem m.primeFactorsList p_1 → Membership.mem (In …
      this : Eq (List.filter (fun x => Decidable.decide (Eq x p)) m.primeFactorsList …
      ⊢ (HAppend.hAppend (List.filter (fun x => Decidable.decide (Membership.mem s x …
    -/
    exact filter_append_perm (· ∈ s) (primeFactorsList m)
    /-
      🎉 no goals
    -/


@[simp]
lemma equivProdNatFactoredNumbers_apply {s : Finset ℕ} {p e m : ℕ} (hp : p.Prime) (hs : p ∉ s)
    (hm : m ∈ factoredNumbers s) :
    equivProdNatFactoredNumbers hp hs (e, ⟨m, hm⟩) = p ^ e * m := rfl


@[simp]
lemma equivProdNatFactoredNumbers_apply' {s : Finset ℕ} {p : ℕ} (hp : p.Prime) (hs : p ∉ s)
    (x : ℕ × factoredNumbers s) :
    equivProdNatFactoredNumbers hp hs x = p ^ x.1 * x.2 := rfl



/-- `smoothNumbers n` is the set of *`n`-smooth positive natural numbers*, i.e., the
positive natural numbers all of whose prime factors are less than `n`. -/
def smoothNumbers (n : ℕ) : Set ℕ := {m | m ≠ 0 ∧ ∀ p ∈ primeFactorsList m, p < n}


lemma mem_smoothNumbers {n m : ℕ} : m ∈ smoothNumbers n ↔ m ≠ 0 ∧ ∀ p ∈ primeFactorsList m, p < n :=
  Iff.rfl


/-- The `n`-smooth numbers agree with the `Finset.range n`-factored numbers. -/
lemma smoothNumbers_eq_factoredNumbers (n : ℕ) :
    smoothNumbers n = factoredNumbers (Finset.range n) := by
  simp only [smoothNumbers, ne_eq, mem_primeFactorsList', and_imp, factoredNumbers,
    Finset.mem_range]


/-- The `n`-smooth numbers agree with the `primesBelow n`-factored numbers. -/
lemma smmoothNumbers_eq_factoredNumbers_primesBelow (n : ℕ) :
    smoothNumbers n = factoredNumbers n.primesBelow := by
  /-
    n : Nat
    ⊢ Eq n.smoothNumbers (Nat.factoredNumbers n.primesBelow)
  -/
  rw [smoothNumbers_eq_factoredNumbers]
  /-
    n : Nat
    ⊢ Eq (Nat.factoredNumbers (Finset.range n)) (Nat.factoredNumbers n.primesBelow)
  -/
  refine Set.Subset.antisymm (fun m hm ↦ ?_) <| factoredNumbers_mono Finset.mem_of_mem_filter
  /-
    n m : Nat
    hm : Membership.mem (Nat.factoredNumbers (Finset.range n)) m
    ⊢ Membership.mem (Nat.factoredNumbers n.primesBelow) m
  -/
  simp_rw [mem_factoredNumbers'] at hm ⊢
  /-
    n m : Nat
    hm : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p m → Membership.mem (Finset.range n) p
    ⊢ ∀ (p : Nat), Nat.Prime p → Dvd.dvd p m → Membership.mem n.primesBelow p
  -/
  exact fun p hp hp' ↦ mem_primesBelow.mpr ⟨Finset.mem_range.mp <| hm p hp hp', hp⟩
  /-
    🎉 no goals
  -/


/-- Membership in `Nat.smoothNumbers n` is decidable. -/
instance (n : ℕ) : DecidablePred (· ∈ smoothNumbers n) :=
  inferInstanceAs <| DecidablePred fun x ↦ x ∈ {m | m ≠ 0 ∧ ∀ p ∈ primeFactorsList m, p < n}


/-- A number that divides an `n`-smooth number is itself `n`-smooth. -/
lemma mem_smoothNumbers_of_dvd {n m k : ℕ} (h : m ∈ smoothNumbers n) (h' : k ∣ m) :
    k ∈ smoothNumbers n := by
  /-
    n m k : Nat
    h : Membership.mem n.smoothNumbers m
    h' : Dvd.dvd k m
    ⊢ Membership.mem n.smoothNumbers k
  -/
  simp only [smoothNumbers_eq_factoredNumbers] at h ⊢
  /-
    n m k : Nat
    h' : Dvd.dvd k m
    h : Membership.mem (Nat.factoredNumbers (Finset.range n)) m
    ⊢ Membership.mem (Nat.factoredNumbers (Finset.range n)) k
  -/
  exact mem_factoredNumbers_of_dvd h h'
  /-
    🎉 no goals
  -/


/-- `m` is `n`-smooth if and only if `m` is nonzero and all prime divisors `≤ m` of `m`
are less than `n`. -/
lemma mem_smoothNumbers_iff_forall_le {n m : ℕ} :
    m ∈ smoothNumbers n ↔ m ≠ 0 ∧ ∀ p ≤ m, p.Prime → p ∣ m → p < n := by
  /-
    n m : Nat
    ⊢ Iff (Membership.mem n.smoothNumbers m) (And (Ne m 0) (∀ (p : Nat), LE.le p m …
  -/
  simp only [smoothNumbers_eq_factoredNumbers, mem_factoredNumbers_iff_forall_le, Finset.mem_range]
  /-
    🎉 no goals
  -/


/-- `m` is `n`-smooth if and only if all prime divisors of `m` are less than `n`. -/
lemma mem_smoothNumbers' {n m : ℕ} : m ∈ smoothNumbers n ↔ ∀ p, p.Prime → p ∣ m → p < n := by
  /-
    n m : Nat
    ⊢ Iff (Membership.mem n.smoothNumbers m) (∀ (p : Nat), Nat.Prime p → Dvd.dvd p …
  -/
  simp only [smoothNumbers_eq_factoredNumbers, mem_factoredNumbers', Finset.mem_range]
  /-
    🎉 no goals
  -/


/-- The `Finset` of prime factors of an `n`-smooth number is contained in the `Finset`
of primes below `n`. -/
lemma primeFactors_subset_of_mem_smoothNumbers {m n : ℕ} (hms : m ∈ n.smoothNumbers) :
    m.primeFactors ⊆ n.primesBelow :=
  primeFactors_subset_of_mem_factoredNumbers <|
    smmoothNumbers_eq_factoredNumbers_primesBelow n ▸ hms


/-- `m` is an `n`-smooth number if the `Finset` of its prime factors consists of numbers `< n`. -/
lemma mem_smoothNumbers_of_primeFactors_subset {m n : ℕ} (hm : m ≠ 0)
    (hp : m.primeFactors ⊆ Finset.range n) : m ∈ n.smoothNumbers :=
  smoothNumbers_eq_factoredNumbers n ▸ mem_factoredNumbers_of_primeFactors_subset hm hp


/-- `m` is an `n`-smooth number if and only if `m ≠ 0` and the `Finset` of its prime factors
is contained in the `Finset` of primes below `n` -/
lemma mem_smoothNumbers_iff_primeFactors_subset {m n : ℕ} :
    m ∈ n.smoothNumbers ↔ m ≠ 0 ∧ m.primeFactors ⊆ n.primesBelow :=
  ⟨fun h ↦ ⟨h.1, primeFactors_subset_of_mem_smoothNumbers h⟩,
   fun h ↦ mem_smoothNumbers_of_primeFactors_subset h.1 <| h.2.trans <| Finset.filter_subset ..⟩


/-- Zero is never a smooth number -/
lemma ne_zero_of_mem_smoothNumbers {n m : ℕ} (h : m ∈ smoothNumbers n) : m ≠ 0 := h.1


@[simp]
lemma smoothNumbers_zero : smoothNumbers 0 = {1} := by
  /-
    ⊢ Eq (Nat.smoothNumbers 0) (Singleton.singleton 1)
  -/
  simp only [smoothNumbers_eq_factoredNumbers, Finset.range_zero, factoredNumbers_empty]
  /-
    🎉 no goals
  -/


/-- The product of two `n`-smooth numbers is an `n`-smooth number. -/
theorem mul_mem_smoothNumbers {m₁ m₂ n : ℕ}
    (hm1 : m₁ ∈ n.smoothNumbers) (hm2 : m₂ ∈ n.smoothNumbers) : m₁ * m₂ ∈ n.smoothNumbers := by
  /-
    m₁ m₂ n : Nat
    hm1 : Membership.mem n.smoothNumbers m₁
    hm2 : Membership.mem n.smoothNumbers m₂
    ⊢ Membership.mem n.smoothNumbers (HMul.hMul m₁ m₂)
  -/
  rw [smoothNumbers_eq_factoredNumbers] at hm1 hm2 ⊢
  /-
    m₁ m₂ n : Nat
    hm1 : Membership.mem (Nat.factoredNumbers (Finset.range n)) m₁
    hm2 : Membership.mem (Nat.factoredNumbers (Finset.range n)) m₂
    ⊢ Membership.mem (Nat.factoredNumbers (Finset.range n)) (HMul.hMul m₁ m₂)
  -/
  exact mul_mem_factoredNumbers hm1 hm2
  /-
    🎉 no goals
  -/


/-- The product of the prime factors of `n` that are less than `N` is an `N`-smooth number. -/
lemma prod_mem_smoothNumbers (n N : ℕ) :
    (n.primeFactorsList.filter (· < N)).prod ∈ smoothNumbers N := by
  /-
    n N : Nat
    ⊢ Membership.mem N.smoothNumbers (List.filter (fun x => Decidable.decide (LT.l …
  -/
  simp only [smoothNumbers_eq_factoredNumbers, ← Finset.mem_range, prod_mem_factoredNumbers]
  /-
    🎉 no goals
  -/


/-- The sets of `N`-smooth and of `(N+1)`-smooth numbers are the same when `N` is not prime.
See `Nat.equivProdNatSmoothNumbers` for when `N` is prime. -/
lemma smoothNumbers_succ {N : ℕ} (hN : ¬ N.Prime) : N.succ.smoothNumbers = N.smoothNumbers := by
  /-
    N : Nat
    hN : Not (Nat.Prime N)
    ⊢ Eq N.succ.smoothNumbers N.smoothNumbers
  -/
  simp only [smoothNumbers_eq_factoredNumbers, Finset.range_succ, factoredNumbers_insert _ hN]
  /-
    🎉 no goals
  -/


@[simp] lemma smoothNumbers_one : smoothNumbers 1 = {1} := by
  /-
    ⊢ Eq (Nat.smoothNumbers 1) (Singleton.singleton 1)
  -/
  simp +decide only [not_false_eq_true, smoothNumbers_succ, smoothNumbers_zero]
  /-
    🎉 no goals
  -/


@[gcongr] lemma smoothNumbers_mono {N M : ℕ} (hNM : N ≤ M) : N.smoothNumbers ⊆ M.smoothNumbers :=
  fun _ hx ↦ ⟨hx.1, fun p hp => (hx.2 p hp).trans_le hNM⟩


/-- All `m`, `0 < m < n` are `n`-smooth numbers -/
lemma mem_smoothNumbers_of_lt {m n : ℕ} (hm : 0 < m) (hmn : m < n) : m ∈ n.smoothNumbers :=
  smoothNumbers_eq_factoredNumbers _ ▸ ⟨not_eq_zero_of_lt hm,
  fun _ h => Finset.mem_range.mpr <| lt_of_le_of_lt (le_of_mem_primeFactorsList h) hmn⟩


/-- The non-zero non-`N`-smooth numbers are `≥ N`. -/
lemma smoothNumbers_compl (N : ℕ) : (N.smoothNumbers)ᶜ \ {0} ⊆ {n | N ≤ n} := by
  simpa only [smoothNumbers_eq_factoredNumbers]
    using factoredNumbers_compl <| Finset.filter_subset _ (Finset.range N)


/-- If `p` is positive and `n` is `p`-smooth, then every product `p^e * n` is `(p+1)`-smooth. -/
lemma pow_mul_mem_smoothNumbers {p n : ℕ} (hp : p ≠ 0) (e : ℕ) (hn : n ∈ smoothNumbers p) :
    p ^ e * n ∈ smoothNumbers (succ p) := by
  -- This cannot be easily reduced to `pow_mul_mem_factoredNumbers`, as there `p.Prime` is needed.
  /-
    p n : Nat
    hp : Ne p 0
    e : Nat
    hn : Membership.mem p.smoothNumbers n
    ⊢ Membership.mem p.succ.smoothNumbers (HMul.hMul (HPow.hPow p e) n)
  -/
  have : NoZeroDivisors ℕ := inferInstance -- this is needed twice --> speed-up
  /-
    p n : Nat
    hp : Ne p 0
    e : Nat
    hn : Membership.mem p.smoothNumbers n
    this : NoZeroDivisors Nat
    ⊢ Membership.mem p.succ.smoothNumbers (HMul.hMul (HPow.hPow p e) n)
  -/
  have hp' := pow_ne_zero e hp
  /-
    p n : Nat
    hp : Ne p 0
    e : Nat
    hn : Membership.mem p.smoothNumbers n
    this : NoZeroDivisors Nat
    hp' : Ne (HPow.hPow p e) 0
    ⊢ Membership.mem p.succ.smoothNumbers (HMul.hMul (HPow.hPow p e) n)
  -/
  refine ⟨mul_ne_zero hp' hn.1, fun q hq ↦ ?_⟩
  /-
    p n : Nat
    hp : Ne p 0
    e : Nat
    hn : Membership.mem p.smoothNumbers n
    this : NoZeroDivisors Nat
    hp' : Ne (HPow.hPow p e) 0
    q : Nat
    hq : Membership.mem (HMul.hMul (HPow.hPow p e) n).primeFactorsList q
    ⊢ LT.lt q p.succ
  -/
  rcases (mem_primeFactorsList_mul hp' hn.1).mp hq with H | H
    /-
      case inl
      p n : Nat
      hp : Ne p 0
      e : Nat
      hn : Membership.mem p.smoothNumbers n
      this : NoZeroDivisors Nat
      hp' : Ne (HPow.hPow p e) 0
      q : Nat
      hq : Membership.mem (HMul.hMul (HPow.hPow p e) n).primeFactorsList q
      H : Membership.mem (HPow.hPow p e).primeFactorsList q
      ⊢ LT.lt q p.succ
    -/
  · rw [mem_primeFactorsList hp'] at H
    /-
      case inl
      p n : Nat
      hp : Ne p 0
      e : Nat
      hn : Membership.mem p.smoothNumbers n
      this : NoZeroDivisors Nat
      hp' : Ne (HPow.hPow p e) 0
      q : Nat
      hq : Membership.mem (HMul.hMul (HPow.hPow p e) n).primeFactorsList q
      H : And (Nat.Prime q) (Dvd.dvd q (HPow.hPow p e))
      ⊢ LT.lt q p.succ
    -/
    exact lt_succ.mpr <| le_of_dvd hp.bot_lt <| H.1.dvd_of_dvd_pow H.2
    /-
      🎉 no goals
    -/
    /-
      case inr
      p n : Nat
      hp : Ne p 0
      e : Nat
      hn : Membership.mem p.smoothNumbers n
      this : NoZeroDivisors Nat
      hp' : Ne (HPow.hPow p e) 0
      q : Nat
      hq : Membership.mem (HMul.hMul (HPow.hPow p e) n).primeFactorsList q
      H : Membership.mem n.primeFactorsList q
      ⊢ LT.lt q p.succ
    -/
  · exact (hn.2 q H).trans <| lt_succ_self p
    /-
      🎉 no goals
    -/


/-- If `p` is a prime and `n` is `p`-smooth, then `p` and `n` are coprime. -/
lemma Prime.smoothNumbers_coprime {p n : ℕ} (hp : p.Prime) (hn : n ∈ smoothNumbers p) :
    Nat.Coprime p n := by
  /-
    p n : Nat
    hp : Nat.Prime p
    hn : Membership.mem p.smoothNumbers n
    ⊢ p.Coprime n
  -/
  simp only [smoothNumbers_eq_factoredNumbers] at hn
  /-
    p n : Nat
    hp : Nat.Prime p
    hn : Membership.mem (Nat.factoredNumbers (Finset.range p)) n
    ⊢ p.Coprime n
  -/
  exact hp.factoredNumbers_coprime Finset.not_mem_range_self hn
  /-
    🎉 no goals
  -/


/-- If `f : ℕ → F` is multiplicative on coprime arguments, `p` is a prime and `m` is `p`-smooth,
then `f (p^e * m) = f (p^e) * f m`. -/
lemma map_prime_pow_mul {F : Type*} [CommSemiring F] {f : ℕ → F}
    (hmul : ∀ {m n}, Nat.Coprime m n → f (m * n) = f m * f n) {p : ℕ} (hp : p.Prime) (e : ℕ)
    {m : p.smoothNumbers} :
    f (p ^ e * m) = f (p ^ e) * f m :=
  hmul <| Coprime.pow_left _ <| hp.smoothNumbers_coprime <| Subtype.mem m


open List Perm Equiv in
/-- We establish the bijection from `ℕ × smoothNumbers p` to `smoothNumbers (p+1)`
given by `(e, n) ↦ p^e * n` when `p` is a prime. See `Nat.smoothNumbers_succ` for
when `p` is not prime. -/
def equivProdNatSmoothNumbers {p : ℕ} (hp : p.Prime) :
    ℕ × smoothNumbers p ≃ smoothNumbers p.succ :=
  ((prodCongrRight fun _ ↦ setCongr <| smoothNumbers_eq_factoredNumbers p).trans <|
    equivProdNatFactoredNumbers hp Finset.not_mem_range_self).trans <|
    setCongr <| (smoothNumbers_eq_factoredNumbers p.succ) ▸ Finset.range_succ ▸ rfl


@[simp]
lemma equivProdNatSmoothNumbers_apply {p e m : ℕ} (hp : p.Prime) (hm : m ∈ p.smoothNumbers) :
    equivProdNatSmoothNumbers hp (e, ⟨m, hm⟩) = p ^ e * m := rfl


@[simp]
lemma equivProdNatSmoothNumbers_apply' {p : ℕ} (hp : p.Prime) (x : ℕ × p.smoothNumbers) :
    equivProdNatSmoothNumbers hp x = p ^ x.1 * x.2 := rfl



/-- The `k`-smooth numbers up to and including `N` as a `Finset` -/
def smoothNumbersUpTo (N k : ℕ) : Finset ℕ :=
    (Finset.range N.succ).filter (· ∈ smoothNumbers k)


lemma mem_smoothNumbersUpTo {N k n : ℕ} :
    n ∈ smoothNumbersUpTo N k ↔ n ≤ N ∧ n ∈ smoothNumbers k := by
  /-
    N k n : Nat
    ⊢ Iff (Membership.mem (N.smoothNumbersUpTo k) n) (And (LE.le n N) (Membership. …
  -/
  simp [smoothNumbersUpTo, lt_succ]
  /-
    🎉 no goals
  -/


/-- The positive non-`k`-smooth (so "`k`-rough") numbers up to and including `N` as a `Finset` -/
def roughNumbersUpTo (N k : ℕ) : Finset ℕ :=
    (Finset.range N.succ).filter (fun n ↦ n ≠ 0 ∧ n ∉ smoothNumbers k)


lemma smoothNumbersUpTo_card_add_roughNumbersUpTo_card (N k : ℕ) :
    (smoothNumbersUpTo N k).card + (roughNumbersUpTo N k).card = N := by
  rw [smoothNumbersUpTo, roughNumbersUpTo,
    ← Finset.card_union_of_disjoint <| Finset.disjoint_filter.mpr fun n _ hn₂ h ↦ h.2 hn₂,
    Finset.filter_union_right]
  suffices Finset.card (Finset.filter (fun x ↦ x ≠ 0) (Finset.range (succ N))) = N by
    have hn' (n) : n ∈ smoothNumbers k ∨ n ≠ 0 ∧ n ∉ smoothNumbers k ↔ n ≠ 0 := by
      have : n ∈ smoothNumbers k → n ≠ 0 := ne_zero_of_mem_smoothNumbers
      refine ⟨fun H ↦ Or.elim H this fun H ↦ H.1, fun H ↦ ?_⟩
      simp only [ne_eq, H, not_false_eq_true, true_and, or_not]
    rwa [Finset.filter_congr (s := Finset.range (succ N)) fun n _ ↦ hn' n]
  /-
    N k : Nat
    ⊢ Eq (Finset.filter (fun x => Ne x 0) (Finset.range N.succ)).card N
  -/
  rw [Finset.filter_ne', Finset.card_erase_of_mem <| Finset.mem_range_succ_iff.mpr <| zero_le N]
  /-
    N k : Nat
    ⊢ Eq (HSub.hSub (Finset.range N.succ).card 1) N
  -/
  simp only [Finset.card_range, succ_sub_succ_eq_sub, tsub_zero]
  /-
    🎉 no goals
  -/


/-- A `k`-smooth number can be written as a square times a product of distinct primes `< k`. -/
lemma eq_prod_primes_mul_sq_of_mem_smoothNumbers {n k : ℕ} (h : n ∈ smoothNumbers k) :
    ∃ s ∈ k.primesBelow.powerset, ∃ m, n = m ^ 2 * (s.prod id) := by
  /-
    n k : Nat
    h : Membership.mem k.smoothNumbers n
    ⊢ Exists fun s => And (Membership.mem k.primesBelow.powerset s) (Exists fun m  …
  -/
  obtain ⟨l, m, H₁, H₂⟩ := sq_mul_squarefree n
  /-
    case intro.intro.intro
    n k : Nat
    h : Membership.mem k.smoothNumbers n
    l m : Nat
    H₁ : Eq (HMul.hMul (HPow.hPow m 2) l) n
    H₂ : Squarefree l
    ⊢ Exists fun s => And (Membership.mem k.primesBelow.powerset s) (Exists fun m  …
  -/
  have hl : l ∈ smoothNumbers k := mem_smoothNumbers_of_dvd h (Dvd.intro_left (m ^ 2) H₁)
  /-
    case intro.intro.intro
    n k : Nat
    h : Membership.mem k.smoothNumbers n
    l m : Nat
    H₁ : Eq (HMul.hMul (HPow.hPow m 2) l) n
    H₂ : Squarefree l
    hl : Membership.mem k.smoothNumbers l
    ⊢ Exists fun s => And (Membership.mem k.primesBelow.powerset s) (Exists fun m  …
  -/
  refine ⟨l.primeFactorsList.toFinset, ?_,  m, ?_⟩
    /-
      case intro.intro.intro.refine_1
      n k : Nat
      h : Membership.mem k.smoothNumbers n
      l m : Nat
      H₁ : Eq (HMul.hMul (HPow.hPow m 2) l) n
      H₂ : Squarefree l
      hl : Membership.mem k.smoothNumbers l
      ⊢ Membership.mem k.primesBelow.powerset l.primeFactorsList.toFinset
    -/
  · simp only [toFinset_factors, Finset.mem_powerset]
    /-
      case intro.intro.intro.refine_1
      n k : Nat
      h : Membership.mem k.smoothNumbers n
      l m : Nat
      H₁ : Eq (HMul.hMul (HPow.hPow m 2) l) n
      H₂ : Squarefree l
      hl : Membership.mem k.smoothNumbers l
      ⊢ HasSubset.Subset l.primeFactors k.primesBelow
    -/
    refine fun p hp ↦ mem_primesBelow.mpr ⟨?_, (mem_primeFactors.mp hp).1⟩
    /-
      case intro.intro.intro.refine_1
      n k : Nat
      h : Membership.mem k.smoothNumbers n
      l m : Nat
      H₁ : Eq (HMul.hMul (HPow.hPow m 2) l) n
      H₂ : Squarefree l
      hl : Membership.mem k.smoothNumbers l
      p : Nat
      hp : Membership.mem l.primeFactors p
      ⊢ LT.lt p k
    -/
    rw [mem_primeFactors] at hp
    /-
      case intro.intro.intro.refine_1
      n k : Nat
      h : Membership.mem k.smoothNumbers n
      l m : Nat
      H₁ : Eq (HMul.hMul (HPow.hPow m 2) l) n
      H₂ : Squarefree l
      hl : Membership.mem k.smoothNumbers l
      p : Nat
      hp : And (Nat.Prime p) (And (Dvd.dvd p l) (Ne l 0))
      ⊢ LT.lt p k
    -/
    exact mem_smoothNumbers'.mp hl p hp.1 hp.2.1
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.refine_2
    n k : Nat
    h : Membership.mem k.smoothNumbers n
    l m : Nat
    H₁ : Eq (HMul.hMul (HPow.hPow m 2) l) n
    H₂ : Squarefree l
    hl : Membership.mem k.smoothNumbers l
    ⊢ Eq n (HMul.hMul (HPow.hPow m 2) (l.primeFactorsList.toFinset.prod id))
  -/
  rw [← H₁]
  /-
    case intro.intro.intro.refine_2
    n k : Nat
    h : Membership.mem k.smoothNumbers n
    l m : Nat
    H₁ : Eq (HMul.hMul (HPow.hPow m 2) l) n
    H₂ : Squarefree l
    hl : Membership.mem k.smoothNumbers l
    ⊢ Eq (HMul.hMul (HPow.hPow m 2) l) (HMul.hMul (HPow.hPow m 2) (l.primeFactorsL …
  -/
  congr
  /-
    case intro.intro.intro.refine_2.e_a
    n k : Nat
    h : Membership.mem k.smoothNumbers n
    l m : Nat
    H₁ : Eq (HMul.hMul (HPow.hPow m 2) l) n
    H₂ : Squarefree l
    hl : Membership.mem k.smoothNumbers l
    ⊢ Eq l (l.primeFactorsList.toFinset.prod id)
  -/
  simp only [toFinset_factors]
  /-
    case intro.intro.intro.refine_2.e_a
    n k : Nat
    h : Membership.mem k.smoothNumbers n
    l m : Nat
    H₁ : Eq (HMul.hMul (HPow.hPow m 2) l) n
    H₂ : Squarefree l
    hl : Membership.mem k.smoothNumbers l
    ⊢ Eq l (l.primeFactors.prod id)
  -/
  exact (prod_primeFactors_of_squarefree H₂).symm
  /-
    🎉 no goals
  -/


/-- The set of `k`-smooth numbers `≤ N` is contained in the set of numbers of the form `m^2 * P`,
where `m ≤ √N` and `P` is a product of distinct primes `< k`. -/
lemma smoothNumbersUpTo_subset_image (N k : ℕ) :
    smoothNumbersUpTo N k ⊆ Finset.image (fun (s, m) ↦ m ^ 2 * (s.prod id))
      (k.primesBelow.powerset ×ˢ (Finset.range N.sqrt.succ).erase 0) := by
  /-
    N k : Nat
    ⊢ HasSubset.Subset (N.smoothNumbersUpTo k) (Finset.image (fun x => Nat.smoothN …
  -/
  intro n hn
  /-
    N k n : Nat
    hn : Membership.mem (N.smoothNumbersUpTo k) n
    ⊢ Membership.mem (Finset.image (fun x => Nat.smoothNumbersUpTo_subset_image.ma …
  -/
  obtain ⟨hn₁, hn₂⟩ := mem_smoothNumbersUpTo.mp hn
  /-
    case intro
    N k n : Nat
    hn : Membership.mem (N.smoothNumbersUpTo k) n
    hn₁ : LE.le n N
    hn₂ : Membership.mem k.smoothNumbers n
    ⊢ Membership.mem (Finset.image (fun x => Nat.smoothNumbersUpTo_subset_image.ma …
  -/
  obtain ⟨s, hs, m, hm⟩ := eq_prod_primes_mul_sq_of_mem_smoothNumbers hn₂
  simp only [id_eq, Finset.mem_range, zero_lt_succ, not_true_eq_false, Finset.mem_image,
    Finset.mem_product, Finset.mem_powerset, Finset.mem_erase, Prod.exists]
  /-
    case intro.intro.intro.intro
    N k n : Nat
    hn : Membership.mem (N.smoothNumbersUpTo k) n
    hn₁ : LE.le n N
    hn₂ : Membership.mem k.smoothNumbers n
    s : Finset Nat
    hs : Membership.mem k.primesBelow.powerset s
    m : Nat
    hm : Eq n (HMul.hMul (HPow.hPow m 2) (s.prod id))
    ⊢ Exists fun a => Exists fun b => And (And (HasSubset.Subset a k.primesBelow)  …
  -/
  refine ⟨s, m, ⟨Finset.mem_powerset.mp hs, ?_, ?_⟩, hm.symm⟩
    /-
      case intro.intro.intro.intro.refine_1
      N k n : Nat
      hn : Membership.mem (N.smoothNumbersUpTo k) n
      hn₁ : LE.le n N
      hn₂ : Membership.mem k.smoothNumbers n
      s : Finset Nat
      hs : Membership.mem k.primesBelow.powerset s
      m : Nat
      hm : Eq n (HMul.hMul (HPow.hPow m 2) (s.prod id))
      ⊢ Ne m 0
    -/
  · have := hm ▸ ne_zero_of_mem_smoothNumbers hn₂
    /-
      case intro.intro.intro.intro.refine_1
      N k n : Nat
      hn : Membership.mem (N.smoothNumbersUpTo k) n
      hn₁ : LE.le n N
      hn₂ : Membership.mem k.smoothNumbers n
      s : Finset Nat
      hs : Membership.mem k.primesBelow.powerset s
      m : Nat
      hm : Eq n (HMul.hMul (HPow.hPow m 2) (s.prod id))
      this : Ne (HMul.hMul (HPow.hPow m 2) (s.prod id)) 0
      ⊢ Ne m 0
    -/
    simp only [ne_eq, _root_.mul_eq_zero, sq_eq_zero_iff, not_or] at this
    /-
      case intro.intro.intro.intro.refine_1
      N k n : Nat
      hn : Membership.mem (N.smoothNumbersUpTo k) n
      hn₁ : LE.le n N
      hn₂ : Membership.mem k.smoothNumbers n
      s : Finset Nat
      hs : Membership.mem k.primesBelow.powerset s
      m : Nat
      hm : Eq n (HMul.hMul (HPow.hPow m 2) (s.prod id))
      this : And (Not (Eq m 0)) (Not (Eq (s.prod id) 0))
      ⊢ Ne m 0
    -/
    exact this.1
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      N k n : Nat
      hn : Membership.mem (N.smoothNumbersUpTo k) n
      hn₁ : LE.le n N
      hn₂ : Membership.mem k.smoothNumbers n
      s : Finset Nat
      hs : Membership.mem k.primesBelow.powerset s
      m : Nat
      hm : Eq n (HMul.hMul (HPow.hPow m 2) (s.prod id))
      ⊢ LT.lt m N.sqrt.succ
    -/
  · rw [lt_succ, le_sqrt']
    /-
      case intro.intro.intro.intro.refine_2
      N k n : Nat
      hn : Membership.mem (N.smoothNumbersUpTo k) n
      hn₁ : LE.le n N
      hn₂ : Membership.mem k.smoothNumbers n
      s : Finset Nat
      hs : Membership.mem k.primesBelow.powerset s
      m : Nat
      hm : Eq n (HMul.hMul (HPow.hPow m 2) (s.prod id))
      ⊢ LE.le (HPow.hPow m 2) N
    -/
    refine LE.le.trans ?_ (hm ▸ hn₁)
    /-
      case intro.intro.intro.intro.refine_2
      N k n : Nat
      hn : Membership.mem (N.smoothNumbersUpTo k) n
      hn₁ : LE.le n N
      hn₂ : Membership.mem k.smoothNumbers n
      s : Finset Nat
      hs : Membership.mem k.primesBelow.powerset s
      m : Nat
      hm : Eq n (HMul.hMul (HPow.hPow m 2) (s.prod id))
      ⊢ LE.le (HPow.hPow m 2) (HMul.hMul (HPow.hPow m 2) (s.prod id))
    -/
    nth_rw 1 [← mul_one (m ^ 2)]
    exact mul_le_mul_left' (Finset.one_le_prod' fun p hp ↦
      (prime_of_mem_primesBelow <| Finset.mem_powerset.mp hs hp).one_lt.le) _


/-- The cardinality of the set of `k`-smooth numbers `≤ N` is bounded by `2^π(k-1) * √N`. -/
lemma smoothNumbersUpTo_card_le (N k : ℕ) :
    (smoothNumbersUpTo N k).card ≤ 2 ^ k.primesBelow.card * N.sqrt := by
  convert (Finset.card_le_card <| smoothNumbersUpTo_subset_image N k).trans <|
    Finset.card_image_le
  simp only [Finset.card_product, Finset.card_powerset, Finset.mem_range, zero_lt_succ,
    Finset.card_erase_of_mem, Finset.card_range, succ_sub_succ_eq_sub, tsub_zero]


/-- The set of `k`-rough numbers `≤ N` can be written as the union of the sets of multiples `≤ N`
of primes `k ≤ p ≤ N`. -/
lemma roughNumbersUpTo_eq_biUnion (N k) :
    roughNumbersUpTo N k =
      (N.succ.primesBelow \ k.primesBelow).biUnion
        fun p ↦ (Finset.range N.succ).filter (fun m ↦ m ≠ 0 ∧ p ∣ m) := by
  /-
    N k : Nat
    ⊢ Eq (N.roughNumbersUpTo k) ((SDiff.sdiff N.succ.primesBelow k.primesBelow).bi …
  -/
  ext m
  simp only [roughNumbersUpTo, mem_smoothNumbers_iff_forall_le, not_and, not_forall,
    not_lt, exists_prop, exists_and_left, Finset.mem_range, not_le, Finset.mem_filter,
    Finset.filter_congr_decidable, Finset.mem_biUnion, Finset.mem_sdiff, mem_primesBelow,
    show ∀ P Q : Prop, P ∧ (P → Q) ↔ P ∧ Q by tauto]
  /-
    case h
    N k m : Nat
    ⊢ Iff (And (LT.lt m N.succ) (And (Ne m 0) (Exists fun x => And (LE.le x m) (An …
  -/
  simp_rw [← exists_and_left, ← not_lt]
  /-
    case h
    N k m : Nat
    ⊢ Iff (Exists fun x => And (LT.lt m N.succ) (And (Ne m 0) (And (Not (LT.lt m x …
  -/
  refine exists_congr fun p ↦ ?_
  have H₁ : m ≠ 0 → p ∣ m → m < N.succ → p < N.succ :=
    fun h₁ h₂ h₃ ↦ (le_of_dvd (Nat.pos_of_ne_zero h₁) h₂).trans_lt h₃
  have H₂ : m ≠ 0 → p ∣ m → ¬ m < p :=
    fun h₁ h₂ ↦ not_lt.mpr <| le_of_dvd (Nat.pos_of_ne_zero h₁) h₂
  /-
    case h
    N k m p : Nat
    H₁ : Ne m 0 → Dvd.dvd p m → LT.lt m N.succ → LT.lt p N.succ
    H₂ : Ne m 0 → Dvd.dvd p m → Not (LT.lt m p)
    ⊢ Iff (And (LT.lt m N.succ) (And (Ne m 0) (And (Not (LT.lt m p)) (And (Nat.Pri …
  -/
  constructor
    /-
      case h.mp
      N k m p : Nat
      H₁ : Ne m 0 → Dvd.dvd p m → LT.lt m N.succ → LT.lt p N.succ
      H₂ : Ne m 0 → Dvd.dvd p m → Not (LT.lt m p)
      ⊢ And (LT.lt m N.succ) (And (Ne m 0) (And (Not (LT.lt m p)) (And (Nat.Prime p) …
    -/
  · rintro ⟨h₁, h₂, _, h₄, h₅, h₆⟩
    /-
      case h.mp.intro.intro.intro.intro.intro
      N k m p : Nat
      H₁ : Ne m 0 → Dvd.dvd p m → LT.lt m N.succ → LT.lt p N.succ
      H₂ : Ne m 0 → Dvd.dvd p m → Not (LT.lt m p)
      h₁ : LT.lt m N.succ
      h₂ : Ne m 0
      left✝ : Not (LT.lt m p)
      h₄ : Nat.Prime p
      h₅ : Dvd.dvd p m
      h₆ : Not (LT.lt p k)
      ⊢ And (And (And (LT.lt p N.succ) (Nat.Prime p)) (LT.lt p k → Not (Nat.Prime p) …
    -/
    exact ⟨⟨⟨H₁ h₂ h₅ h₁, h₄⟩, fun h _ ↦ h₆ h⟩, h₁, h₂, h₅⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      N k m p : Nat
      H₁ : Ne m 0 → Dvd.dvd p m → LT.lt m N.succ → LT.lt p N.succ
      H₂ : Ne m 0 → Dvd.dvd p m → Not (LT.lt m p)
      ⊢ And (And (And (LT.lt p N.succ) (Nat.Prime p)) (LT.lt p k → Not (Nat.Prime p) …
    -/
  · rintro ⟨⟨⟨_, h₂⟩, h₃⟩, h₄, h₅, h₆⟩
    /-
      case h.mpr.intro.intro.intro.intro.intro
      N k m p : Nat
      H₁ : Ne m 0 → Dvd.dvd p m → LT.lt m N.succ → LT.lt p N.succ
      H₂ : Ne m 0 → Dvd.dvd p m → Not (LT.lt m p)
      h₃ : LT.lt p k → Not (Nat.Prime p)
      left✝ : LT.lt p N.succ
      h₂ : Nat.Prime p
      h₄ : LT.lt m N.succ
      h₅ : Ne m 0
      h₆ : Dvd.dvd p m
      ⊢ And (LT.lt m N.succ) (And (Ne m 0) (And (Not (LT.lt m p)) (And (Nat.Prime p) …
    -/
    exact ⟨h₄, h₅, H₂ h₅ h₆, h₂, h₆, fun h ↦ h₃ h h₂⟩
    /-
      🎉 no goals
    -/


/-- The cardinality of the set of `k`-rough numbers `≤ N` is bounded by the sum of `⌊N/p⌋`
over the primes `k ≤ p ≤ N`. -/
lemma roughNumbersUpTo_card_le (N k : ℕ) :
    (roughNumbersUpTo N k).card ≤ (N.succ.primesBelow \ k.primesBelow).sum (fun p ↦ N / p) := by
  /-
    N k : Nat
    ⊢ LE.le (N.roughNumbersUpTo k).card ((SDiff.sdiff N.succ.primesBelow k.primesB …
  -/
  rw [roughNumbersUpTo_eq_biUnion]
  /-
    N k : Nat
    ⊢ LE.le ((SDiff.sdiff N.succ.primesBelow k.primesBelow).biUnion fun p => Finse …
  -/
  exact Finset.card_biUnion_le.trans <| Finset.sum_le_sum fun p _ ↦ (card_multiples' N p).le
  /-
    🎉 no goals
  -/


