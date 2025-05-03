/-- `n : ℕ` is _abundant_ if the sum of the proper divisors of `n` is greater than `n`. -/
def Abundant (n : ℕ) : Prop := n < ∑ i ∈ properDivisors n, i


/-- `n : ℕ` is _deficient_ if the sum of the proper divisors of `n` is less than `n`. -/
def Deficient (n : ℕ) : Prop := ∑ i ∈ properDivisors n, i < n


/-- A positive natural number `n` is _pseudoperfect_ if there exists a subset of the proper
  divisors of `n` such that the sum of that subset is equal to `n`. -/
def Pseudoperfect (n : ℕ) : Prop :=
  0 < n ∧ ∃ s ⊆ properDivisors n, ∑ i ∈ s, i = n


/-- `n : ℕ` is a _weird_ number if and only if it is abundant but not pseudoperfect. -/
def Weird (n : ℕ) : Prop := Abundant n ∧ ¬ Pseudoperfect n


theorem not_pseudoperfect_iff_forall :
    ¬ Pseudoperfect n ↔ n = 0 ∨ ∀ s ⊆ properDivisors n, ∑ i ∈ s, i ≠ n := by
  /-
    n : Nat
    ⊢ Iff (Not n.Pseudoperfect) (Or (Eq n 0) (∀ (s : Finset Nat), HasSubset.Subset …
  -/
  rw [Pseudoperfect, not_and_or]
  /-
    n : Nat
    ⊢ Iff (Or (Not (LT.lt 0 n)) (Not (Exists fun s => And (HasSubset.Subset s n.pr …
  -/
  simp only [not_lt, nonpos_iff_eq_zero, mem_powerset, not_exists, not_and, ne_eq]
  /-
    🎉 no goals
  -/


theorem deficient_one : Deficient 1 := zero_lt_one

theorem deficient_two : Deficient 2 := one_lt_two

                                            /-
                                              ⊢ Nat.Deficient 3
                                            -/
theorem deficient_three : Deficient 3 := by norm_num [Deficient]
                                            /-
                                              🎉 no goals
                                            -/


theorem abundant_twelve : Abundant 12 := by
  /-
    ⊢ Nat.Abundant 12
  -/
  rw [Abundant, show properDivisors 12 = {1,2,3,4,6} by rfl]
  /-
    ⊢ LT.lt 12 ((Insert.insert 1 (Insert.insert 2 (Insert.insert 3 (Insert.insert  …
  -/
  norm_num
  /-
    🎉 no goals
  -/


set_option maxRecDepth 1730 in
theorem weird_seventy : Weird 70 := by
  /-
    ⊢ Nat.Weird 70
  -/
  rw [Weird, Abundant, not_pseudoperfect_iff_forall]
  /-
    ⊢ And (LT.lt 70 ((Nat.properDivisors 70).sum fun i => i)) (Or (Eq 70 0) (∀ (s  …
  -/
  have h : properDivisors 70 = {1, 2, 5, 7, 10, 14, 35} := by rfl
  /-
    h : Eq (Nat.properDivisors 70) (Insert.insert 1 (Insert.insert 2 (Insert.inser …
    ⊢ And (LT.lt 70 ((Nat.properDivisors 70).sum fun i => i)) (Or (Eq 70 0) (∀ (s  …
  -/
  constructor
    /-
      case left
      h : Eq (Nat.properDivisors 70) (Insert.insert 1 (Insert.insert 2 (Insert.inser …
      ⊢ LT.lt 70 ((Nat.properDivisors 70).sum fun i => i)
    -/
  · rw [h]
    /-
      case left
      h : Eq (Nat.properDivisors 70) (Insert.insert 1 (Insert.insert 2 (Insert.inser …
      ⊢ LT.lt 70 ((Insert.insert 1 (Insert.insert 2 (Insert.insert 5 (Insert.insert  …
    -/
    repeat norm_num
    /-
      🎉 no goals
    -/
    /-
      case right
      h : Eq (Nat.properDivisors 70) (Insert.insert 1 (Insert.insert 2 (Insert.inser …
      ⊢ Or (Eq 70 0) (∀ (s : Finset Nat), HasSubset.Subset s (Nat.properDivisors 70) …
    -/
  · rw [h]
    /-
      case right
      h : Eq (Nat.properDivisors 70) (Insert.insert 1 (Insert.insert 2 (Insert.inser …
      ⊢ Or (Eq 70 0) (∀ (s : Finset Nat), HasSubset.Subset s (Insert.insert 1 (Inser …
    -/
    right
    /-
      case right.h
      h : Eq (Nat.properDivisors 70) (Insert.insert 1 (Insert.insert 2 (Insert.inser …
      ⊢ ∀ (s : Finset Nat), HasSubset.Subset s (Insert.insert 1 (Insert.insert 2 (In …
    -/
    intro s hs
    /-
      case right.h
      h : Eq (Nat.properDivisors 70) (Insert.insert 1 (Insert.insert 2 (Insert.inser …
      s : Finset Nat
      hs : HasSubset.Subset s (Insert.insert 1 (Insert.insert 2 (Insert.insert 5 (In …
      ⊢ Ne (s.sum fun i => i) 70
    -/
    have hs' := mem_powerset.mpr hs
    /-
      case right.h
      h : Eq (Nat.properDivisors 70) (Insert.insert 1 (Insert.insert 2 (Insert.inser …
      s : Finset Nat
      hs : HasSubset.Subset s (Insert.insert 1 (Insert.insert 2 (Insert.insert 5 (In …
      hs' : Membership.mem (Insert.insert 1 (Insert.insert 2 (Insert.insert 5 (Inser …
      ⊢ Ne (s.sum fun i => i) 70
    -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
    fin_cases hs' <;> decide
                      /-
                        🎉 no goals
                      -/


lemma deficient_iff_not_abundant_and_not_perfect (hn : n ≠ 0) :
    Deficient n ↔ ¬ Abundant n ∧ ¬ Perfect n := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Iff n.Deficient (And (Not n.Abundant) (Not n.Perfect))
  -/
  dsimp only [Perfect, Abundant, Deficient]
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Iff (LT.lt (n.properDivisors.sum fun i => i) n) (And (Not (LT.lt n (n.proper …
  -/
  omega
  /-
    🎉 no goals
  -/


lemma perfect_iff_not_abundant_and_not_deficient (hn : 0 ≠ n) :
    Perfect n ↔ ¬ Abundant n ∧ ¬ Deficient n := by
  /-
    n : Nat
    hn : Ne 0 n
    ⊢ Iff n.Perfect (And (Not n.Abundant) (Not n.Deficient))
  -/
  dsimp only [Perfect, Abundant, Deficient]
  /-
    n : Nat
    hn : Ne 0 n
    ⊢ Iff (And (Eq (n.properDivisors.sum fun i => i) n) (LT.lt 0 n)) (And (Not (LT …
  -/
  omega
  /-
    🎉 no goals
  -/


lemma abundant_iff_not_perfect_and_not_deficient (hn : 0 ≠ n) :
    Abundant n ↔ ¬ Perfect n ∧ ¬ Deficient n := by
  /-
    n : Nat
    hn : Ne 0 n
    ⊢ Iff n.Abundant (And (Not n.Perfect) (Not n.Deficient))
  -/
  dsimp only [Perfect, Abundant, Deficient]
  /-
    n : Nat
    hn : Ne 0 n
    ⊢ Iff (LT.lt n (n.properDivisors.sum fun i => i)) (And (Not (And (Eq (n.proper …
  -/
  omega
  /-
    🎉 no goals
  -/


/-- A positive natural number is either deficient, perfect, or abundant -/
theorem deficient_or_perfect_or_abundant (hn : 0 ≠ n) :
    Deficient n ∨ Abundant n ∨ Perfect n := by
  /-
    n : Nat
    hn : Ne 0 n
    ⊢ Or n.Deficient (Or n.Abundant n.Perfect)
  -/
  dsimp only [Perfect, Abundant, Deficient]
  /-
    n : Nat
    hn : Ne 0 n
    ⊢ Or (LT.lt (n.properDivisors.sum fun i => i) n) (Or (LT.lt n (n.properDivisor …
  -/
  omega
  /-
    🎉 no goals
  -/


theorem Perfect.pseudoperfect (h : Perfect n) : Pseudoperfect n :=
  ⟨h.2, ⟨properDivisors n, ⟨fun ⦃_⦄ a ↦ a, h.1⟩⟩⟩


theorem Prime.not_abundant (h : Prime n) : ¬ Abundant n :=
  fun h1 ↦ (h.one_lt.trans h1).ne' (sum_properDivisors_eq_one_iff_prime.mpr h)


theorem Prime.not_weird (h : Prime n) : ¬ Weird n := by
  /-
    n : Nat
    h : Nat.Prime n
    ⊢ Not n.Weird
  -/
  simp only [Nat.Weird, not_and_or]
  /-
    n : Nat
    h : Nat.Prime n
    ⊢ Or (Not n.Abundant) (Not (Not n.Pseudoperfect))
  -/
  left
  /-
    case h
    n : Nat
    h : Nat.Prime n
    ⊢ Not n.Abundant
  -/
  exact h.not_abundant
  /-
    🎉 no goals
  -/


theorem Prime.not_pseudoperfect (h : Prime p) : ¬ Pseudoperfect p := by
  simp_rw [not_pseudoperfect_iff_forall, ← mem_powerset,
    show p.properDivisors.powerset = {∅, {1}} by rw [Prime.properDivisors h]; rfl]
  /-
    p : Nat
    h : Nat.Prime p
    ⊢ Or (Eq p 0) (∀ (s : Finset Nat), Membership.mem (Insert.insert EmptyCollecti …
  -/
  refine Or.inr (fun s hs ↦ ?_)
  /-
    p : Nat
    h : Nat.Prime p
    s : Finset Nat
    hs : Membership.mem (Insert.insert EmptyCollection.emptyCollection (Singleton. …
    ⊢ Ne (s.sum fun i => i) p
  -/
  fin_cases hs <;>
  /-
    case «0»
    p : Nat
    h : Nat.Prime p
    ⊢ Ne (EmptyCollection.emptyCollection.sum fun i => i) p
  -/
  simp only [sum_empty, sum_singleton] <;>
  /-
    case «0»
    p : Nat
    h : Nat.Prime p
    ⊢ Ne 0 p
  -/
  /-
    🎉 no goals
  -/
  linarith [Prime.one_lt h]
  /-
    🎉 no goals
  -/


theorem Prime.not_perfect (h : Prime p) : ¬ Perfect p := by
  /-
    p : Nat
    h : Nat.Prime p
    ⊢ Not p.Perfect
  -/
  have h1 := Prime.not_pseudoperfect h
  /-
    p : Nat
    h : Nat.Prime p
    h1 : Not p.Pseudoperfect
    ⊢ Not p.Perfect
  -/
  revert h1
  /-
    p : Nat
    h : Nat.Prime p
    ⊢ Not p.Pseudoperfect → Not p.Perfect
  -/
  exact not_imp_not.mpr (Perfect.pseudoperfect)
  /-
    🎉 no goals
  -/


/-- Any natural number power of a prime is deficient -/
theorem Prime.deficient_pow  (h : Prime n) : Deficient (n ^ m) := by
  /-
    n m : Nat
    h : Nat.Prime n
    ⊢ (HPow.hPow n m).Deficient
  -/
  rcases Nat.eq_zero_or_pos m with (rfl | _)
    /-
      case inl
      n : Nat
      h : Nat.Prime n
      ⊢ (HPow.hPow n 0).Deficient
    -/
  · simpa using deficient_one
    /-
      🎉 no goals
    -/
  · have h1 : (n ^ m).properDivisors = image (n ^ ·) (range m) := by
      apply subset_antisymm <;> intro a
      · simp only [mem_properDivisors, mem_image, mem_range, dvd_prime_pow h]
        rintro ⟨⟨t, ht, rfl⟩, ha'⟩
        exact ⟨t, lt_of_le_of_ne ht (fun ht' ↦ lt_irrefl _ (ht' ▸ ha')), rfl⟩
      · simp only [mem_image, mem_range, mem_properDivisors, forall_exists_index, and_imp]
        intro x hx hy
        constructor
        · rw [← hy, dvd_prime_pow h]
          exact ⟨x, Nat.le_of_succ_le hx, rfl⟩
        · rw [← hy]
          exact (Nat.pow_lt_pow_iff_right (Prime.two_le h)).mpr hx
    have h2 : ∑ i in image (fun x => n ^ x) (range m), i = ∑ i in range m, n^i := by
      rw [Finset.sum_image]
      rintro x _ y _
      apply pow_injective_of_not_isUnit h.not_unit <| Prime.ne_zero h
    /-
      case inr
      n m : Nat
      h : Nat.Prime n
      h✝ : GT.gt m 0
      h1 : Eq (HPow.hPow n m).properDivisors (Finset.image (fun x => HPow.hPow n x)  …
      h2 : Eq ((Finset.image (fun x => HPow.hPow n x) (Finset.range m)).sum fun i => …
      ⊢ (HPow.hPow n m).Deficient
    -/
    rw [Deficient, h1, h2]
    calc
      ∑ i ∈ range m, n ^ i
        = (n ^ m - 1) / (n - 1) := (Nat.geomSum_eq (Prime.two_le h) _)
      _ ≤ (n ^ m - 1) := Nat.div_le_self (n ^ m - 1) (n - 1)
      _ < n ^ m := sub_lt (pow_pos (Prime.pos h) m) (Nat.one_pos)


theorem _root_.IsPrimePow.deficient (h : IsPrimePow n) : Deficient n := by
  /-
    n : Nat
    h : IsPrimePow n
    ⊢ n.Deficient
  -/
  obtain ⟨p, k, hp, -, rfl⟩ := h
  /-
    case intro.intro.intro.intro
    p k : Nat
    hp : _root_.Prime p
    ⊢ (HPow.hPow p k).Deficient
  -/
  exact hp.nat_prime.deficient_pow
  /-
    🎉 no goals
  -/


theorem Prime.deficient (h : Prime n) : Deficient n := by
  /-
    n : Nat
    h : Nat.Prime n
    ⊢ n.Deficient
  -/
  rw [← pow_one n]
  /-
    n : Nat
    h : Nat.Prime n
    ⊢ (HPow.hPow n 1).Deficient
  -/
  exact h.deficient_pow
  /-
    🎉 no goals
  -/


/-- There exists infinitely many deficient numbers -/
theorem infinite_deficient : {n : ℕ | n.Deficient}.Infinite := by
  /-
    ⊢ (setOf fun n => n.Deficient).Infinite
  -/
  rw [Set.infinite_iff_exists_gt]
  /-
    ⊢ ∀ (a : Nat), Exists fun b => And (Membership.mem (setOf fun n => n.Deficient …
  -/
  intro a
  /-
    a : Nat
    ⊢ Exists fun b => And (Membership.mem (setOf fun n => n.Deficient) b) (LT.lt a …
  -/
  obtain ⟨b, h1, h2⟩ := exists_infinite_primes a.succ
  /-
    case intro.intro
    a b : Nat
    h1 : LE.le a.succ b
    h2 : Nat.Prime b
    ⊢ Exists fun b => And (Membership.mem (setOf fun n => n.Deficient) b) (LT.lt a …
  -/
  exact ⟨b, h2.deficient, h1⟩
  /-
    🎉 no goals
  -/


theorem infinite_even_deficient : {n : ℕ | Even n ∧ n.Deficient}.Infinite := by
  /-
    ⊢ (setOf fun n => And (Even n) n.Deficient).Infinite
  -/
  rw [Set.infinite_iff_exists_gt]
  /-
    ⊢ ∀ (a : Nat), Exists fun b => And (Membership.mem (setOf fun n => And (Even n …
  -/
  intro n
  /-
    n : Nat
    ⊢ Exists fun b => And (Membership.mem (setOf fun n => And (Even n) n.Deficient …
  -/
  use 2 ^ (n + 1)
  /-
    case h
    n : Nat
    ⊢ And (Membership.mem (setOf fun n => And (Even n) n.Deficient) (HPow.hPow 2 ( …
  -/
  constructor
    /-
      case h.left
      n : Nat
      ⊢ Membership.mem (setOf fun n => And (Even n) n.Deficient) (HPow.hPow 2 (HAdd. …
    -/
  · exact ⟨⟨2 ^ n, by ring⟩, prime_two.deficient_pow⟩
    /-
      🎉 no goals
    -/
  · calc
      n ≤ 2 ^ n := Nat.le_of_lt n.lt_two_pow_self
      _ < 2 ^ (n + 1) := (Nat.pow_lt_pow_iff_right (Nat.one_lt_two)).mpr (lt_add_one n)


theorem infinite_odd_deficient : {n : ℕ | Odd n ∧ n.Deficient}.Infinite := by
  /-
    ⊢ (setOf fun n => And (Odd n) n.Deficient).Infinite
  -/
  rw [Set.infinite_iff_exists_gt]
  /-
    ⊢ ∀ (a : Nat), Exists fun b => And (Membership.mem (setOf fun n => And (Odd n) …
  -/
  intro n
  /-
    n : Nat
    ⊢ Exists fun b => And (Membership.mem (setOf fun n => And (Odd n) n.Deficient) …
  -/
  obtain ⟨p, ⟨_, h2⟩⟩ := exists_infinite_primes (max (n + 1) 3)
  exact ⟨p, Set.mem_setOf.mpr ⟨Prime.odd_of_ne_two h2 (Ne.symm (ne_of_lt (by omega))),
    Prime.deficient h2⟩, by omega⟩


