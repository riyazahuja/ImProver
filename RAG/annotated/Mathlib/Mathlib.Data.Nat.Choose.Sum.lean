/-- A version of the **binomial theorem** for commuting elements in noncommutative semirings. -/
theorem add_pow (h : Commute x y) (n : ℕ) :
    (x + y) ^ n = ∑ m ∈ range (n + 1), x ^ m * y ^ (n - m) * n.choose m := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    h : Commute x y
    n : Nat
    ⊢ Eq (HPow.hPow (HAdd.hAdd x y) n) ((Finset.range (HAdd.hAdd n 1)).sum fun m = …
  -/
  let t : ℕ → ℕ → R := fun n m ↦ x ^ m * y ^ (n - m) * n.choose m
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    h : Commute x y
    n : Nat
    t : Nat → Nat → R := fun n m => HMul.hMul (HMul.hMul (HPow.hPow x m) (HPow.hPo …
    ⊢ Eq (HPow.hPow (HAdd.hAdd x y) n) ((Finset.range (HAdd.hAdd n 1)).sum fun m = …
  -/
  change (x + y) ^ n = ∑ m ∈ range (n + 1), t n m
  have h_first : ∀ n, t n 0 = y ^ n := fun n ↦ by
    simp only [t, choose_zero_right, pow_zero, cast_one, mul_one, one_mul, tsub_zero]
  have h_last : ∀ n, t n n.succ = 0 := fun n ↦ by
    simp only [t, choose_succ_self, cast_zero, mul_zero]
  have h_middle :
      ∀ n i : ℕ, i ∈ range n.succ → (t n.succ i.succ) = x * t n i + y * t n i.succ := by
    intro n i h_mem
    have h_le : i ≤ n := le_of_lt_succ (mem_range.mp h_mem)
    dsimp only [t]
    rw [choose_succ_succ, cast_add, mul_add]
    congr 1
    · rw [pow_succ' x, succ_sub_succ, mul_assoc, mul_assoc, mul_assoc]
    · rw [← mul_assoc y, ← mul_assoc y, (h.symm.pow_right i.succ).eq]
      by_cases h_eq : i = n
      · rw [h_eq, choose_succ_self, cast_zero, mul_zero, mul_zero]
      · rw [succ_sub (lt_of_le_of_ne h_le h_eq)]
        rw [pow_succ' y, mul_assoc, mul_assoc, mul_assoc, mul_assoc]
  induction n with
  | zero =>
    rw [pow_zero, sum_range_succ, range_zero, sum_empty, zero_add]
    dsimp only [t]
    rw [pow_zero, pow_zero, choose_self, cast_one, mul_one, mul_one]
  | succ n ih =>
    rw [sum_range_succ', h_first, sum_congr rfl (h_middle n), sum_add_distrib, add_assoc,
      pow_succ' (x + y), ih, add_mul, mul_sum, mul_sum]
    congr 1
    rw [sum_range_succ', sum_range_succ, h_first, h_last, mul_zero, add_zero, _root_.pow_succ']


/-- A version of `Commute.add_pow` that avoids ℕ-subtraction by summing over the antidiagonal and
also with the binomial coefficient applied via scalar action of ℕ. -/
theorem add_pow' (h : Commute x y) (n : ℕ) :
    (x + y) ^ n = ∑ m ∈ antidiagonal n, n.choose m.1 • (x ^ m.1 * y ^ m.2) := by
  simp_rw [Nat.sum_antidiagonal_eq_sum_range_succ fun m p ↦ n.choose m • (x ^ m * y ^ p),
    nsmul_eq_mul, cast_comm, h.add_pow]


/-- The **binomial theorem** -/
theorem add_pow [CommSemiring R] (x y : R) (n : ℕ) :
    (x + y) ^ n = ∑ m ∈ range (n + 1), x ^ m * y ^ (n - m) * n.choose m :=
  (Commute.all x y).add_pow n


/-- A special case of the **binomial theorem** -/
theorem sub_pow [CommRing R] (x y : R) (n : ℕ) :
    (x - y) ^ n = ∑ m ∈ range (n + 1), (-1) ^ (m + n) * x ^ m * y ^ (n - m) * n.choose m := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    n : Nat
    ⊢ Eq (HPow.hPow (HSub.hSub x y) n) ((Finset.range (HAdd.hAdd n 1)).sum fun m = …
  -/
  rw [sub_eq_add_neg, add_pow]
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun m => HMul.hMul (HMul.hMul (HPow.h …
  -/
  congr! 1 with m hm
  have : (-1 : R) ^ (n - m) = (-1) ^ (n + m) := by
    rw [mem_range] at hm
    simp [show n + m = n - m + 2 * m by omega, pow_add]
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    n m : Nat
    hm : Membership.mem (Finset.range (HAdd.hAdd n 1)) m
    this : Eq (HPow.hPow (-1) (HSub.hSub n m)) (HPow.hPow (-1) (HAdd.hAdd n m))
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow x m) (HPow.hPow (Neg.neg y) (HSub.hSub n …
  -/
  rw [neg_pow, this]
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    n m : Nat
    hm : Membership.mem (Finset.range (HAdd.hAdd n 1)) m
    this : Eq (HPow.hPow (-1) (HSub.hSub n m)) (HPow.hPow (-1) (HAdd.hAdd n m))
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow x m) (HMul.hMul (HPow.hPow (-1) (HAdd.hA …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The sum of entries in a row of Pascal's triangle -/
theorem sum_range_choose (n : ℕ) : (∑ m ∈ range (n + 1), n.choose m) = 2 ^ n := by
  /-
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun m => n.choose m) (HPow.hPow 2 n)
  -/
  have := (add_pow 1 1 n).symm
  /-
    n : Nat
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun m => HMul.hMul (HMul.hMul (H …
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun m => n.choose m) (HPow.hPow 2 n)
  -/
  simpa [one_add_one_eq_two] using this
  /-
    🎉 no goals
  -/


theorem sum_range_choose_halfway (m : ℕ) : (∑ i ∈ range (m + 1), (2 * m + 1).choose i) = 4 ^ m :=
  have : (∑ i ∈ range (m + 1), (2 * m + 1).choose (2 * m + 1 - i)) =
      ∑ i ∈ range (m + 1), (2 * m + 1).choose i :=
                                               /-
                                                 m i : Nat
                                                 hi : Membership.mem (Finset.range (HAdd.hAdd m 1)) i
                                                 ⊢ LE.le i (HAdd.hAdd (HMul.hMul 2 m) 1)
                                               -/
    sum_congr rfl fun i hi ↦ choose_symm <| by linarith [mem_range.1 hi]
                                               /-
                                                 🎉 no goals
                                               -/
  mul_right_injective₀ two_ne_zero <|
    calc
      (2 * ∑ i ∈ range (m + 1), (2 * m + 1).choose i) =
          (∑ i ∈ range (m + 1), (2 * m + 1).choose i) +
                                                                          /-
                                                                            m : Nat
                                                                            this : Eq ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul.hMul 2 …
                                                                            ⊢ Eq (HMul.hMul 2 ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMu …
                                                                          -/
            ∑ i ∈ range (m + 1), (2 * m + 1).choose (2 * m + 1 - i) := by rw [two_mul, this]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
      _ = (∑ i ∈ range (m + 1), (2 * m + 1).choose i) +
            ∑ i ∈ Ico (m + 1) (2 * m + 2), (2 * m + 1).choose i := by
        /-
          m : Nat
          this : Eq ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul.hMul 2 …
          ⊢ Eq (HAdd.hAdd ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul. …
        -/
        rw [range_eq_Ico, sum_Ico_reflect _ _ (by omega)]
        /-
          m : Nat
          this : Eq ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul.hMul 2 …
          ⊢ Eq (HAdd.hAdd ((Finset.Ico 0 (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul. …
        -/
        congr
        /-
          case e_a.e_s.e_a
          m : Nat
          this : Eq ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul.hMul 2 …
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 m) 1) 1) (HAdd.hAdd m 1)) ( …
        -/
        have A : m + 1 ≤ 2 * m + 1 := by omega
        /-
          case e_a.e_s.e_a
          m : Nat
          this : Eq ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul.hMul 2 …
          A : LE.le (HAdd.hAdd m 1) (HAdd.hAdd (HMul.hMul 2 m) 1)
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 m) 1) 1) (HAdd.hAdd m 1)) ( …
        -/
        rw [add_comm, add_tsub_assoc_of_le A, ← add_comm]
        /-
          case e_a.e_s.e_a
          m : Nat
          this : Eq ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul.hMul 2 …
          A : LE.le (HAdd.hAdd m 1) (HAdd.hAdd (HMul.hMul 2 m) 1)
          ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd m 1)) 1) ( …
        -/
        congr
        /-
          case e_a.e_s.e_a.e_a
          m : Nat
          this : Eq ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul.hMul 2 …
          A : LE.le (HAdd.hAdd m 1) (HAdd.hAdd (HMul.hMul 2 m) 1)
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd m 1)) m
        -/
        rw [tsub_eq_iff_eq_add_of_le A]
        /-
          case e_a.e_s.e_a.e_a
          m : Nat
          this : Eq ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul.hMul 2 …
          A : LE.le (HAdd.hAdd m 1) (HAdd.hAdd (HMul.hMul 2 m) 1)
          ⊢ Eq (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd m (HAdd.hAdd m 1))
        -/
        ring
        /-
          🎉 no goals
        -/
                                                                                       /-
                                                                                         m : Nat
                                                                                         this : Eq ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul.hMul 2 …
                                                                                         ⊢ LE.le (HAdd.hAdd m 1) (HAdd.hAdd (HMul.hMul 2 m) 2)
                                                                                       -/
      _ = ∑ i ∈ range (2 * m + 2), (2 * m + 1).choose i := sum_range_add_sum_Ico _ (by omega)
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
      _ = 2 ^ (2 * m + 1) := sum_range_choose (2 * m + 1)
                          /-
                            m : Nat
                            this : Eq ((Finset.range (HAdd.hAdd m 1)).sum fun i => (HAdd.hAdd (HMul.hMul 2 …
                            ⊢ Eq (HPow.hPow 2 (HAdd.hAdd (HMul.hMul 2 m) 1)) (HMul.hMul 2 (HPow.hPow 4 m))
                          -/
      _ = 2 * 4 ^ m := by rw [pow_succ, pow_mul, mul_comm]; rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem choose_middle_le_pow (n : ℕ) : (2 * n + 1).choose n ≤ 4 ^ n := by
  have t : (2 * n + 1).choose n ≤ ∑ i ∈ range (n + 1), (2 * n + 1).choose i :=
    single_le_sum (fun x _ ↦ by omega) (self_mem_range_succ n)
  /-
    n : Nat
    t : LE.le ((HAdd.hAdd (HMul.hMul 2 n) 1).choose n) ((Finset.range (HAdd.hAdd n …
    ⊢ LE.le ((HAdd.hAdd (HMul.hMul 2 n) 1).choose n) (HPow.hPow 4 n)
  -/
  simpa [sum_range_choose_halfway n] using t
  /-
    🎉 no goals
  -/


theorem four_pow_le_two_mul_add_one_mul_central_binom (n : ℕ) :
    4 ^ n ≤ (2 * n + 1) * (2 * n).choose n :=
  calc
                                    /-
                                      n : Nat
                                      ⊢ Eq (HPow.hPow 4 n) (HPow.hPow (HAdd.hAdd 1 1) (HMul.hMul 2 n))
                                    -/
    4 ^ n = (1 + 1) ^ (2 * n) := by norm_num [pow_mul]
                                    /-
                                      🎉 no goals
                                    -/
                                                        /-
                                                          n : Nat
                                                          ⊢ Eq (HPow.hPow (HAdd.hAdd 1 1) (HMul.hMul 2 n)) ((Finset.range (HAdd.hAdd (HM …
                                                        -/
    _ = ∑ m ∈ range (2 * n + 1), (2 * n).choose m := by set_option simprocs false in simp [add_pow]
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                                  /-
                                                                    n : Nat
                                                                    ⊢ LE.le ((Finset.range (HAdd.hAdd (HMul.hMul 2 n) 1)).sum fun m => (HMul.hMul  …
                                                                  -/
    _ ≤ ∑ _ ∈ range (2 * n + 1), (2 * n).choose (2 * n / 2) := by gcongr; apply choose_le_middle
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                             /-
                                               n : Nat
                                               ⊢ Eq ((Finset.range (HAdd.hAdd (HMul.hMul 2 n) 1)).sum fun x => (HMul.hMul 2 n …
                                             -/
    _ = (2 * n + 1) * choose (2 * n) n := by simp
                                             /-
                                               🎉 no goals
                                             -/


/-- **Zhu Shijie's identity** aka hockey-stick identity, version with `Icc`. -/
theorem sum_Icc_choose (n k : ℕ) : ∑ m ∈ Icc k n, m.choose k = (n + 1).choose (k + 1) := by
  /-
    n k : Nat
    ⊢ Eq ((Finset.Icc k n).sum fun m => m.choose k) ((HAdd.hAdd n 1).choose (HAdd. …
  -/
  rcases lt_or_le n k with h | h
    /-
      case inl
      n k : Nat
      h : LT.lt n k
      ⊢ Eq ((Finset.Icc k n).sum fun m => m.choose k) ((HAdd.hAdd n 1).choose (HAdd. …
    -/
  · rw [choose_eq_zero_of_lt (by omega), Icc_eq_empty_of_lt h, sum_empty]
    /-
      🎉 no goals
    -/
  · induction n, h using le_induction with
    | base => simp
    | succ n _ ih =>
      rw [← Ico_insert_right (by omega), sum_insert (by simp), Ico_succ_right, ih,
        choose_succ_succ' (n + 1)]


/-- **Zhu Shijie's identity** aka hockey-stick identity, version with `range`.
Summing `(i + k).choose k` for `i ∈ [0, n]` gives `(n + k + 1).choose (k + 1)`.

Combinatorial interpretation: `(i + k).choose k` is the number of decompositions of `[0, i)` in
`k + 1` (possibly empty) intervals (this follows from a stars and bars description). In particular,
`(n + k + 1).choose (k + 1)` corresponds to decomposing `[0, n)` into `k + 2` intervals.
By putting away the last interval (of some length `n - i`),
we have to decompose the remaining interval `[0, i)` into `k + 1` intervals, hence the sum. -/
lemma sum_range_add_choose (n k : ℕ) :
    ∑ i ∈ Finset.range (n + 1), (i + k).choose k = (n + k + 1).choose (k + 1) := by
  /-
    n k : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun i => (HAdd.hAdd i k).choose k) (( …
  -/
  rw [← sum_Icc_choose, range_eq_Ico]
  /-
    n k : Nat
    ⊢ Eq ((Finset.Ico 0 (HAdd.hAdd n 1)).sum fun i => (HAdd.hAdd i k).choose k) (( …
  -/
  convert (sum_map _ (addRightEmbedding k) (·.choose k)).symm using 2
  /-
    case h.e'_3.h
    n k : Nat
    ⊢ Eq (Finset.Icc k (HAdd.hAdd n k)) (Finset.map (addRightEmbedding k) (Finset. …
  -/
  rw [map_add_right_Ico, zero_add, add_right_comm, Nat.Ico_succ_right]
  /-
    🎉 no goals
  -/


theorem Int.alternating_sum_range_choose {n : ℕ} :
    (∑ m ∈ range (n + 1), ((-1) ^ m * n.choose m : ℤ)) = if n = 0 then 1 else 0 := by
  cases n with
  | zero => simp
  | succ n =>
    have h := add_pow (-1 : ℤ) 1 n.succ
    simp only [one_pow, mul_one, neg_add_cancel] at h
    rw [← h, zero_pow n.succ_ne_zero, if_neg n.succ_ne_zero]


theorem Int.alternating_sum_range_choose_of_ne {n : ℕ} (h0 : n ≠ 0) :
    (∑ m ∈ range (n + 1), ((-1) ^ m * n.choose m : ℤ)) = 0 := by
  /-
    n : Nat
    h0 : Ne n 0
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun m => HMul.hMul (HPow.hPow (-1) m) …
  -/
  rw [Int.alternating_sum_range_choose, if_neg h0]
  /-
    🎉 no goals
  -/


theorem sum_powerset_apply_card {α β : Type*} [AddCommMonoid α] (f : ℕ → α) {x : Finset β} :
    ∑ m ∈ x.powerset, f #m = ∑ m ∈ range (#x + 1), (#x).choose m • f m := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : AddCommMonoid α
    f : Nat → α
    x : Finset β
    ⊢ Eq (x.powerset.sum fun m => f m.card) ((Finset.range (HAdd.hAdd x.card 1)).s …
  -/
  trans ∑ m ∈ range (#x + 1), ∑ j ∈ x.powerset with #j = m, f #j
    /-
      α : Type u_2
      β : Type u_3
      inst✝ : AddCommMonoid α
      f : Nat → α
      x : Finset β
      ⊢ Eq (x.powerset.sum fun m => f m.card) ((Finset.range (HAdd.hAdd x.card 1)).s …
    -/
  · refine (sum_fiberwise_of_maps_to ?_ _).symm
    /-
      α : Type u_2
      β : Type u_3
      inst✝ : AddCommMonoid α
      f : Nat → α
      x : Finset β
      ⊢ ∀ (i : Finset β), Membership.mem x.powerset i → Membership.mem (Finset.range …
    -/
    intro y hy
    /-
      α : Type u_2
      β : Type u_3
      inst✝ : AddCommMonoid α
      f : Nat → α
      x y : Finset β
      hy : Membership.mem x.powerset y
      ⊢ Membership.mem (Finset.range (HAdd.hAdd x.card 1)) y.card
    -/
    rw [mem_range, Nat.lt_succ_iff]
    /-
      α : Type u_2
      β : Type u_3
      inst✝ : AddCommMonoid α
      f : Nat → α
      x y : Finset β
      hy : Membership.mem x.powerset y
      ⊢ LE.le y.card x.card
    -/
    rw [mem_powerset] at hy
    /-
      α : Type u_2
      β : Type u_3
      inst✝ : AddCommMonoid α
      f : Nat → α
      x y : Finset β
      hy : HasSubset.Subset y x
      ⊢ LE.le y.card x.card
    -/
    exact card_le_card hy
    /-
      🎉 no goals
    -/
    /-
      α : Type u_2
      β : Type u_3
      inst✝ : AddCommMonoid α
      f : Nat → α
      x : Finset β
      ⊢ Eq ((Finset.range (HAdd.hAdd x.card 1)).sum fun m => (Finset.filter (fun j = …
    -/
  · refine sum_congr rfl fun y _ ↦ ?_
    /-
      α : Type u_2
      β : Type u_3
      inst✝ : AddCommMonoid α
      f : Nat → α
      x : Finset β
      y : Nat
      x✝ : Membership.mem (Finset.range (HAdd.hAdd x.card 1)) y
      ⊢ Eq ((Finset.filter (fun j => Eq j.card y) x.powerset).sum fun j => f j.card) …
    -/
    rw [← card_powersetCard, ← sum_const]
    /-
      α : Type u_2
      β : Type u_3
      inst✝ : AddCommMonoid α
      f : Nat → α
      x : Finset β
      y : Nat
      x✝ : Membership.mem (Finset.range (HAdd.hAdd x.card 1)) y
      ⊢ Eq ((Finset.filter (fun j => Eq j.card y) x.powerset).sum fun j => f j.card) …
    -/
    refine sum_congr powersetCard_eq_filter.symm fun z hz ↦ ?_
    /-
      α : Type u_2
      β : Type u_3
      inst✝ : AddCommMonoid α
      f : Nat → α
      x : Finset β
      y : Nat
      x✝ : Membership.mem (Finset.range (HAdd.hAdd x.card 1)) y
      z : Finset β
      hz : Membership.mem (Finset.powersetCard y x) z
      ⊢ Eq (f z.card) (f y)
    -/
    rw [(mem_powersetCard.1 hz).2]
    /-
      🎉 no goals
    -/


theorem sum_powerset_neg_one_pow_card {α : Type*} [DecidableEq α] {x : Finset α} :
    (∑ m ∈ x.powerset, (-1 : ℤ) ^ #m) = if x = ∅ then 1 else 0 := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    x : Finset α
    ⊢ Eq (x.powerset.sum fun m => HPow.hPow (-1) m.card) (ite (Eq x EmptyCollectio …
  -/
  rw [sum_powerset_apply_card]
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    x : Finset α
    ⊢ Eq ((Finset.range (HAdd.hAdd x.card 1)).sum fun m => HSMul.hSMul (x.card.cho …
  -/
  simp only [nsmul_eq_mul', ← card_eq_zero, Int.alternating_sum_range_choose]
  /-
    🎉 no goals
  -/


theorem sum_powerset_neg_one_pow_card_of_nonempty {α : Type*} {x : Finset α} (h0 : x.Nonempty) :
    (∑ m ∈ x.powerset, (-1 : ℤ) ^ #m) = 0 := by
  classical
  rw [sum_powerset_neg_one_pow_card]
  exact if_neg (nonempty_iff_ne_empty.mp h0)


@[to_additive sum_choose_succ_nsmul]
theorem prod_pow_choose_succ {M : Type*} [CommMonoid M] (f : ℕ → ℕ → M) (n : ℕ) :
    (∏ i ∈ range (n + 2), f i (n + 1 - i) ^ (n + 1).choose i) =
      (∏ i ∈ range (n + 1), f i (n + 1 - i) ^ n.choose i) *
        ∏ i ∈ range (n + 1), f (i + 1) (n - i) ^ n.choose i := by
  have A : (∏ i ∈ range (n + 1), f (i + 1) (n - i) ^ (n.choose (i + 1))) * f 0 (n + 1) =
      ∏ i ∈ range (n + 1), f i (n + 1 - i) ^ (n.choose i) := by
    rw [prod_range_succ, prod_range_succ']; simp
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → Nat → M
    n : Nat
    A : Eq (HMul.hMul ((Finset.range (HAdd.hAdd n 1)).prod fun i => HPow.hPow (f ( …
    ⊢ Eq ((Finset.range (HAdd.hAdd n 2)).prod fun i => HPow.hPow (f i (HSub.hSub ( …
  -/
  rw [prod_range_succ']
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → Nat → M
    n : Nat
    A : Eq (HMul.hMul ((Finset.range (HAdd.hAdd n 1)).prod fun i => HPow.hPow (f ( …
    ⊢ Eq (HMul.hMul ((Finset.range (HAdd.hAdd n 1)).prod fun k => HPow.hPow (f (HA …
  -/
  simpa [choose_succ_succ, pow_add, prod_mul_distrib, A, mul_assoc] using mul_comm _ _
  /-
    🎉 no goals
  -/


@[to_additive sum_antidiagonal_choose_succ_nsmul]
theorem prod_antidiagonal_pow_choose_succ {M : Type*} [CommMonoid M] (f : ℕ → ℕ → M) (n : ℕ) :
    (∏ ij ∈ antidiagonal (n + 1), f ij.1 ij.2 ^ (n + 1).choose ij.1) =
      (∏ ij ∈ antidiagonal n, f ij.1 (ij.2 + 1) ^ n.choose ij.1) *
        ∏ ij ∈ antidiagonal n, f (ij.1 + 1) ij.2 ^ n.choose ij.2 := by
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → Nat → M
    n : Nat
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).prod fun ij => HPo …
  -/
  simp only [Nat.prod_antidiagonal_eq_prod_range_succ_mk, prod_pow_choose_succ]
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → Nat → M
    n : Nat
    ⊢ Eq (HMul.hMul ((Finset.range (HAdd.hAdd n 1)).prod fun i => HPow.hPow (f i ( …
  -/
  have : ∀ i ∈ range (n + 1), i ≤ n := fun i hi ↦ by simpa [Nat.lt_succ_iff] using hi
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → Nat → M
    n : Nat
    this : ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) i → LE.le i n
    ⊢ Eq (HMul.hMul ((Finset.range (HAdd.hAdd n 1)).prod fun i => HPow.hPow (f i ( …
  -/
  congr 1
    /-
      case e_a
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → Nat → M
      n : Nat
      this : ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) i → LE.le i n
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).prod fun i => HPow.hPow (f i (HSub.hSub ( …
    -/
  · refine prod_congr rfl fun i hi ↦ ?_
    /-
      case e_a
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → Nat → M
      n : Nat
      this : ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) i → LE.le i n
      i : Nat
      hi : Membership.mem (Finset.range n.succ) i
      ⊢ Eq (HPow.hPow (f i (HSub.hSub (HAdd.hAdd n 1) i)) (n.choose i)) (HPow.hPow ( …
    -/
    rw [tsub_add_eq_add_tsub (this _ hi)]
    /-
      🎉 no goals
    -/
    /-
      case e_a
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → Nat → M
      n : Nat
      this : ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) i → LE.le i n
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).prod fun i => HPow.hPow (f (HAdd.hAdd i 1 …
    -/
  · refine prod_congr rfl fun i hi ↦ ?_
    /-
      case e_a
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → Nat → M
      n : Nat
      this : ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) i → LE.le i n
      i : Nat
      hi : Membership.mem (Finset.range n.succ) i
      ⊢ Eq (HPow.hPow (f (HAdd.hAdd i 1) (HSub.hSub n i)) (n.choose i)) (HPow.hPow ( …
    -/
    rw [choose_symm (this _ hi)]
    /-
      🎉 no goals
    -/


/-- The sum of `(n+1).choose i * f i (n+1-i)` can be split into two sums at rank `n`,
respectively of `n.choose i * f i (n+1-i)` and `n.choose i * f (i+1) (n-i)`. -/
theorem sum_choose_succ_mul (f : ℕ → ℕ → R) (n : ℕ) :
    (∑ i ∈ range (n + 2), ((n + 1).choose i : R) * f i (n + 1 - i)) =
      (∑ i ∈ range (n + 1), (n.choose i : R) * f i (n + 1 - i)) +
        ∑ i ∈ range (n + 1), (n.choose i : R) * f (i + 1) (n - i) := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    f : Nat → Nat → R
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 2)).sum fun i => HMul.hMul (↑((HAdd.hAdd n 1) …
  -/
  simpa only [nsmul_eq_mul] using sum_choose_succ_nsmul f n
  /-
    🎉 no goals
  -/


/-- The sum along the antidiagonal of `(n+1).choose i * f i j` can be split into two sums along the
antidiagonal at rank `n`, respectively of `n.choose i * f i (j+1)` and `n.choose j * f (i+1) j`. -/
theorem sum_antidiagonal_choose_succ_mul (f : ℕ → ℕ → R) (n : ℕ) :
    (∑ ij ∈ antidiagonal (n + 1), ((n + 1).choose ij.1 : R) * f ij.1 ij.2) =
      (∑ ij ∈ antidiagonal n, (n.choose ij.1 : R) * f ij.1 (ij.2 + 1)) +
        ∑ ij ∈ antidiagonal n, (n.choose ij.2 : R) * f (ij.1 + 1) ij.2 := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    f : Nat → Nat → R
    n : Nat
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).sum fun ij => HMul …
  -/
  simpa only [nsmul_eq_mul] using sum_antidiagonal_choose_succ_nsmul f n
  /-
    🎉 no goals
  -/


theorem sum_antidiagonal_choose_add (d n : ℕ) :
    (∑ ij ∈ antidiagonal n, (d + ij.2).choose d) = (d + n).choose d + (d + n).choose (d + 1) := by
  induction n with
  | zero => simp
  | succ n hn => simpa [Nat.sum_antidiagonal_succ] using hn


