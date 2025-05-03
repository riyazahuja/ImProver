/-- The cardinality of the set of `k`-rough numbers `≤ N` is bounded by `N` times the sum
of `1/p` over the primes `k ≤ p ≤ N`. -/
-- This needs `Mathlib.Analysis.RCLike.Basic`, so we put it here
-- instead of in `Mathlib.NumberTheory.SmoothNumbers`.
lemma Nat.roughNumbersUpTo_card_le' (N k : ℕ) :
    (roughNumbersUpTo N k).card ≤
      N * (N.succ.primesBelow \ k.primesBelow).sum (fun p ↦ (1 : ℝ) / p) := by
  /-
    N k : Nat
    ⊢ LE.le (↑(N.roughNumbersUpTo k).card) (HMul.hMul (↑N) ((SDiff.sdiff N.succ.pr …
  -/
  simp_rw [Finset.mul_sum, mul_one_div]
  exact (Nat.cast_le.mpr <| roughNumbersUpTo_card_le N k).trans <|
    (cast_sum (β := ℝ) ..) ▸ Finset.sum_le_sum fun n _ ↦ cast_div_le


/-- The sum over primes `k ≤ p ≤ 4^(π(k-1)+1)` over `1/p` (as a real number) is at least `1/2`. -/
lemma one_half_le_sum_primes_ge_one_div (k : ℕ) :
    1 / 2 ≤ ∑ p ∈ (4 ^ (k.primesBelow.card + 1)).succ.primesBelow \ k.primesBelow,
      (1 / p : ℝ) := by
  /-
    k : Nat
    ⊢ LE.le (1 / 2) ((SDiff.sdiff (HPow.hPow 4 (HAdd.hAdd k.primesBelow.card 1)).s …
  -/
  set m : ℕ := 2 ^ k.primesBelow.card
  /-
    k : Nat
    m : Nat := HPow.hPow 2 k.primesBelow.card
    ⊢ LE.le (1 / 2) ((SDiff.sdiff (HPow.hPow 4 (HAdd.hAdd k.primesBelow.card 1)).s …
  -/
  set N₀ : ℕ := 2 * m ^ 2 with hN₀
  /-
    k : Nat
    m : Nat := HPow.hPow 2 k.primesBelow.card
    N₀ : Nat := HMul.hMul 2 (HPow.hPow m 2)
    hN₀ : Eq N₀ (HMul.hMul 2 (HPow.hPow m 2))
    ⊢ LE.le (1 / 2) ((SDiff.sdiff (HPow.hPow 4 (HAdd.hAdd k.primesBelow.card 1)).s …
  -/
  let S : ℝ := ((2 * N₀).succ.primesBelow \ k.primesBelow).sum (fun p ↦ (1 / p : ℝ))
  suffices 1 / 2 ≤ S by
    convert this using 5
    rw [show 4 = 2 ^ 2 by norm_num, pow_right_comm]
    ring
  suffices 2 * N₀ ≤ m * (2 * N₀).sqrt + 2 * N₀ * S by
    rwa [hN₀, ← mul_assoc, ← pow_two 2, ← mul_pow, sqrt_eq', ← sub_le_iff_le_add',
      cast_mul, cast_mul, cast_pow, cast_two,
      show (2 * (2 * m ^ 2) - m * (2 * m) : ℝ) = 2 * (2 * m ^ 2) * (1 / 2) by ring,
      _root_.mul_le_mul_left <| by positivity] at this
  calc (2 * N₀ : ℝ)
    _ = ((2 * N₀).smoothNumbersUpTo k).card + ((2 * N₀).roughNumbersUpTo k).card := by
        exact_mod_cast ((2 * N₀).smoothNumbersUpTo_card_add_roughNumbersUpTo_card k).symm
    _ ≤ m * (2 * N₀).sqrt + ((2 * N₀).roughNumbersUpTo k).card := by
        exact_mod_cast Nat.add_le_add_right ((2 * N₀).smoothNumbersUpTo_card_le k) _
    _ ≤ m * (2 * N₀).sqrt + 2 * N₀ * S := add_le_add_left ?_ _
  /-
    k : Nat
    m : Nat := HPow.hPow 2 k.primesBelow.card
    N₀ : Nat := HMul.hMul 2 (HPow.hPow m 2)
    hN₀ : Eq N₀ (HMul.hMul 2 (HPow.hPow m 2))
    S : Real := (SDiff.sdiff (HMul.hMul 2 N₀).succ.primesBelow k.primesBelow).sum  …
    ⊢ LE.le (↑((HMul.hMul 2 N₀).roughNumbersUpTo k).card) (HMul.hMul (HMul.hMul 2  …
  -/
  exact_mod_cast roughNumbersUpTo_card_le' (2 * N₀) k
  /-
    🎉 no goals
  -/


/-- The sum over the reciprocals of the primes diverges. -/
theorem not_summable_one_div_on_primes :
    ¬ Summable (indicator {p | p.Prime} (fun n : ℕ ↦ (1 : ℝ) / n)) := by
  /-
    ⊢ Not (Summable ((setOf fun p => Nat.Prime p).indicator fun n => HDiv.hDiv 1 ↑ …
  -/
  intro h
  /-
    h : Summable ((setOf fun p => Nat.Prime p).indicator fun n => HDiv.hDiv 1 ↑n)
    ⊢ False
  -/
  obtain ⟨k, hk⟩ := h.nat_tsum_vanishing (Iio_mem_nhds one_half_pos : Iio (1 / 2 : ℝ) ∈ 𝓝 0)
  /-
    case intro
    h : Summable ((setOf fun p => Nat.Prime p).indicator fun n => HDiv.hDiv 1 ↑n)
    k : Nat
    hk : ∀ (t : Set Nat), HasSubset.Subset t (setOf fun n => LE.le k n) → Membersh …
    ⊢ False
  -/
  specialize hk ({p | Nat.Prime p} ∩ {p | k ≤ p}) inter_subset_right
  /-
    case intro
    h : Summable ((setOf fun p => Nat.Prime p).indicator fun n => HDiv.hDiv 1 ↑n)
    k : Nat
    hk : Membership.mem (Set.Iio (1 / 2)) (tsum fun n => (setOf fun p => Nat.Prime …
    ⊢ False
  -/
  rw [tsum_subtype, indicator_indicator, inter_eq_left.mpr fun n hn ↦ hn.1, mem_Iio] at hk
  have h' : Summable (indicator ({p | Nat.Prime p} ∩ {p | k ≤ p}) fun n ↦ (1 : ℝ) / n) := by
    convert h.indicator {n : ℕ | k ≤ n} using 1
    simp only [indicator_indicator, inter_comm]
  /-
    case intro
    h : Summable ((setOf fun p => Nat.Prime p).indicator fun n => HDiv.hDiv 1 ↑n)
    k : Nat
    hk : LT.lt (tsum fun x => (Inter.inter (setOf fun p => Nat.Prime p) (setOf fun …
    h' : Summable ((Inter.inter (setOf fun p => Nat.Prime p) (setOf fun p => LE.le …
    ⊢ False
  -/
  refine ((one_half_le_sum_primes_ge_one_div k).trans_lt <| LE.le.trans_lt ?_ hk).false
  convert sum_le_tsum (primesBelow ((4 ^ (k.primesBelow.card + 1)).succ) \ primesBelow k)
    (fun n _ ↦ indicator_nonneg (fun p _ ↦ by positivity) _) h' using 2 with p hp
  /-
    case h.e'_3.a
    h : Summable ((setOf fun p => Nat.Prime p).indicator fun n => HDiv.hDiv 1 ↑n)
    k : Nat
    hk : LT.lt (tsum fun x => (Inter.inter (setOf fun p => Nat.Prime p) (setOf fun …
    h' : Summable ((Inter.inter (setOf fun p => Nat.Prime p) (setOf fun p => LE.le …
    p : Nat
    hp : Membership.mem (SDiff.sdiff (HPow.hPow 4 (HAdd.hAdd k.primesBelow.card 1) …
    ⊢ Eq (HDiv.hDiv 1 ↑p) ((Inter.inter (setOf fun p => Nat.Prime p) (setOf fun p  …
  -/
  obtain ⟨hp₁, hp₂⟩ := mem_setOf_eq ▸ Finset.mem_sdiff.mp hp
  /-
    case h.e'_3.a.intro
    h : Summable ((setOf fun p => Nat.Prime p).indicator fun n => HDiv.hDiv 1 ↑n)
    k : Nat
    hk : LT.lt (tsum fun x => (Inter.inter (setOf fun p => Nat.Prime p) (setOf fun …
    h' : Summable ((Inter.inter (setOf fun p => Nat.Prime p) (setOf fun p => LE.le …
    p : Nat
    hp : Membership.mem (SDiff.sdiff (HPow.hPow 4 (HAdd.hAdd k.primesBelow.card 1) …
    hp₁ : Membership.mem (HPow.hPow 4 (HAdd.hAdd k.primesBelow.card 1)).succ.prime …
    hp₂ : Not (Membership.mem k.primesBelow p)
    ⊢ Eq (HDiv.hDiv 1 ↑p) ((Inter.inter (setOf fun p => Nat.Prime p) (setOf fun p  …
  -/
  have hpp := prime_of_mem_primesBelow hp₁
  /-
    case h.e'_3.a.intro
    h : Summable ((setOf fun p => Nat.Prime p).indicator fun n => HDiv.hDiv 1 ↑n)
    k : Nat
    hk : LT.lt (tsum fun x => (Inter.inter (setOf fun p => Nat.Prime p) (setOf fun …
    h' : Summable ((Inter.inter (setOf fun p => Nat.Prime p) (setOf fun p => LE.le …
    p : Nat
    hp : Membership.mem (SDiff.sdiff (HPow.hPow 4 (HAdd.hAdd k.primesBelow.card 1) …
    hp₁ : Membership.mem (HPow.hPow 4 (HAdd.hAdd k.primesBelow.card 1)).succ.prime …
    hp₂ : Not (Membership.mem k.primesBelow p)
    hpp : Nat.Prime p
    ⊢ Eq (HDiv.hDiv 1 ↑p) ((Inter.inter (setOf fun p => Nat.Prime p) (setOf fun p  …
  -/
  refine (indicator_of_mem (mem_def.mpr ⟨hpp, ?_⟩) fun n : ℕ ↦ (1 / n : ℝ)).symm
  /-
    case h.e'_3.a.intro
    h : Summable ((setOf fun p => Nat.Prime p).indicator fun n => HDiv.hDiv 1 ↑n)
    k : Nat
    hk : LT.lt (tsum fun x => (Inter.inter (setOf fun p => Nat.Prime p) (setOf fun …
    h' : Summable ((Inter.inter (setOf fun p => Nat.Prime p) (setOf fun p => LE.le …
    p : Nat
    hp : Membership.mem (SDiff.sdiff (HPow.hPow 4 (HAdd.hAdd k.primesBelow.card 1) …
    hp₁ : Membership.mem (HPow.hPow 4 (HAdd.hAdd k.primesBelow.card 1)).succ.prime …
    hp₂ : Not (Membership.mem k.primesBelow p)
    hpp : Nat.Prime p
    ⊢ Membership.mem (setOf fun p => LE.le k p) p
  -/
  exact not_lt.mp <| (not_and_or.mp <| (not_congr mem_primesBelow).mp hp₂).neg_resolve_right hpp
  /-
    🎉 no goals
  -/


/-- The sum over the reciprocals of the primes diverges. -/
theorem Nat.Primes.not_summable_one_div : ¬ Summable (fun p : Nat.Primes ↦ (1 / p : ℝ)) := by
  /-
    ⊢ Not (Summable fun p => HDiv.hDiv 1 ↑↑p)
  -/
  convert summable_subtype_iff_indicator.mp.mt not_summable_one_div_on_primes
  /-
    🎉 no goals
  -/


/-- The series over `p^r` for primes `p` converges if and only if `r < -1`. -/
theorem Nat.Primes.summable_rpow {r : ℝ} :
    Summable (fun p : Nat.Primes ↦ (p : ℝ) ^ r) ↔ r < -1 := by
  /-
    r : Real
    ⊢ Iff (Summable fun p => HPow.hPow (↑↑p) r) (LT.lt r (-1))
  -/
  by_cases h : r < -1
  · -- case `r < -1`
    /-
      case pos
      r : Real
      h : LT.lt r (-1)
      ⊢ Iff (Summable fun p => HPow.hPow (↑↑p) r) (LT.lt r (-1))
    -/
    simp only [h, iff_true]
    /-
      case pos
      r : Real
      h : LT.lt r (-1)
      ⊢ Summable fun p => HPow.hPow (↑↑p) r
    -/
    exact (Real.summable_nat_rpow.mpr h).subtype _
    /-
      🎉 no goals
    -/
  · -- case `-1 ≤ r`
    /-
      case neg
      r : Real
      h : Not (LT.lt r (-1))
      ⊢ Iff (Summable fun p => HPow.hPow (↑↑p) r) (LT.lt r (-1))
    -/
    simp only [h, iff_false]
    /-
      case neg
      r : Real
      h : Not (LT.lt r (-1))
      ⊢ Not (Summable fun p => HPow.hPow (↑↑p) r)
    -/
    refine fun H ↦ Nat.Primes.not_summable_one_div <| H.of_nonneg_of_le (fun _ ↦ by positivity) ?_
    /-
      case neg
      r : Real
      h : Not (LT.lt r (-1))
      H : Summable fun p => HPow.hPow (↑↑p) r
      ⊢ ∀ (b : Nat.Primes), LE.le (HDiv.hDiv 1 ↑↑b) (HPow.hPow (↑↑b) r)
    -/
    intro p
    /-
      case neg
      r : Real
      h : Not (LT.lt r (-1))
      H : Summable fun p => HPow.hPow (↑↑p) r
      p : Nat.Primes
      ⊢ LE.le (HDiv.hDiv 1 ↑↑p) (HPow.hPow (↑↑p) r)
    -/
    rw [one_div, ← Real.rpow_neg_one]
    /-
      case neg
      r : Real
      h : Not (LT.lt r (-1))
      H : Summable fun p => HPow.hPow (↑↑p) r
      p : Nat.Primes
      ⊢ LE.le (HPow.hPow (↑↑p) (-1)) (HPow.hPow (↑↑p) r)
    -/
    exact Real.rpow_le_rpow_of_exponent_le (by exact_mod_cast p.prop.one_lt.le) <| not_lt.mp h
    /-
      🎉 no goals
    -/

