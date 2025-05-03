/--
A sequence `u` has the property that its ratio of successive differences is bounded
when there is a positive real number `C` such that, for all n ∈ ℕ,
(u (n + 2) - u (n + 1)) ≤ C * (u (n + 1) - u n)
-/
def SuccDiffBounded (C : ℕ) (u : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, u (n + 2) - u (n + 1) ≤ C • (u (n + 1) - u n)


theorem le_sum_schlomilch' (hf : ∀ ⦃m n⦄, 0 < m → m ≤ n → f n ≤ f m) (h_pos : ∀ n, 0 < u n)
    (hu : Monotone u) (n : ℕ) :
    (∑ k ∈ Ico (u 0) (u n), f k) ≤ ∑ k ∈ range n, (u (k + 1) - u k) • f (u k) := by
  induction n with
  | zero => simp
  | succ n ihn =>
    suffices (∑ k ∈ Ico (u n) (u (n + 1)), f k) ≤ (u (n + 1) - u n) • f (u n) by
      rw [sum_range_succ, ← sum_Ico_consecutive]
      · exact add_le_add ihn this
      exacts [hu n.zero_le, hu n.le_succ]
    have : ∀ k ∈ Ico (u n) (u (n + 1)), f k ≤ f (u n) := fun k hk =>
      hf (Nat.succ_le_of_lt (h_pos n)) (mem_Ico.mp hk).1
    convert sum_le_sum this
    simp [pow_succ, mul_two]


theorem le_sum_condensed' (hf : ∀ ⦃m n⦄, 0 < m → m ≤ n → f n ≤ f m) (n : ℕ) :
    (∑ k ∈ Ico 1 (2 ^ n), f k) ≤ ∑ k ∈ range n, 2 ^ k • f (2 ^ k) := by
  convert le_sum_schlomilch' hf (fun n => pow_pos zero_lt_two n)
    (fun m n hm => pow_right_mono₀ one_le_two hm) n using 2
  /-
    case h.e'_4.a
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    n x✝ : Nat
    a✝ : Membership.mem (Finset.range n) x✝
    ⊢ Eq (HSMul.hSMul (HPow.hPow 2 x✝) (f (HPow.hPow 2 x✝))) (HSMul.hSMul (HSub.hS …
  -/
  simp [pow_succ, mul_two, two_mul]
  /-
    🎉 no goals
  -/


theorem le_sum_schlomilch (hf : ∀ ⦃m n⦄, 0 < m → m ≤ n → f n ≤ f m) (h_pos : ∀ n, 0 < u n)
    (hu : Monotone u) (n : ℕ) :
    (∑ k ∈ range (u n), f k) ≤
      ∑ k ∈ range (u 0), f k + ∑ k ∈ range n, (u (k + 1) - u k) • f (u k) := by
  /-
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    u : Nat → Nat
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu : Monotone u
    n : Nat
    ⊢ LE.le ((Finset.range (u n)).sum fun k => f k) (HAdd.hAdd ((Finset.range (u 0 …
  -/
  convert add_le_add_left (le_sum_schlomilch' hf h_pos hu n) (∑ k ∈ range (u 0), f k)
  /-
    case h.e'_3
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    u : Nat → Nat
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu : Monotone u
    n : Nat
    ⊢ Eq ((Finset.range (u n)).sum fun k => f k) (HAdd.hAdd ((Finset.range (u 0)). …
  -/
  rw [← sum_range_add_sum_Ico _ (hu n.zero_le)]
  /-
    🎉 no goals
  -/


theorem le_sum_condensed (hf : ∀ ⦃m n⦄, 0 < m → m ≤ n → f n ≤ f m) (n : ℕ) :
    (∑ k ∈ range (2 ^ n), f k) ≤ f 0 + ∑ k ∈ range n, 2 ^ k • f (2 ^ k) := by
  /-
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    n : Nat
    ⊢ LE.le ((Finset.range (HPow.hPow 2 n)).sum fun k => f k) (HAdd.hAdd (f 0) ((F …
  -/
  convert add_le_add_left (le_sum_condensed' hf n) (f 0)
  /-
    case h.e'_3
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    n : Nat
    ⊢ Eq ((Finset.range (HPow.hPow 2 n)).sum fun k => f k) (HAdd.hAdd (f 0) ((Fins …
  -/
  rw [← sum_range_add_sum_Ico _ n.one_le_two_pow, sum_range_succ, sum_range_zero, zero_add]
  /-
    🎉 no goals
  -/


theorem sum_schlomilch_le' (hf : ∀ ⦃m n⦄, 1 < m → m ≤ n → f n ≤ f m) (h_pos : ∀ n, 0 < u n)
    (hu : Monotone u) (n : ℕ) :
    (∑ k ∈ range n, (u (k + 1) - u k) • f (u (k + 1))) ≤ ∑ k ∈ Ico (u 0 + 1) (u n + 1), f k := by
  induction n with
  | zero => simp
  | succ n ihn =>
    suffices (u (n + 1) - u n) • f (u (n + 1)) ≤ ∑ k ∈ Ico (u n + 1) (u (n + 1) + 1), f k by
      rw [sum_range_succ, ← sum_Ico_consecutive]
      exacts [add_le_add ihn this,
        (add_le_add_right (hu n.zero_le) _ : u 0 + 1 ≤ u n + 1),
        add_le_add_right (hu n.le_succ) _]
    have : ∀ k ∈ Ico (u n + 1) (u (n + 1) + 1), f (u (n + 1)) ≤ f k := fun k hk =>
      hf (Nat.lt_of_le_of_lt (Nat.succ_le_of_lt (h_pos n)) <| (Nat.lt_succ_of_le le_rfl).trans_le
        (mem_Ico.mp hk).1) (Nat.le_of_lt_succ <| (mem_Ico.mp hk).2)
    convert sum_le_sum this
    simp [pow_succ, mul_two]


theorem sum_condensed_le' (hf : ∀ ⦃m n⦄, 1 < m → m ≤ n → f n ≤ f m) (n : ℕ) :
    (∑ k ∈ range n, 2 ^ k • f (2 ^ (k + 1))) ≤ ∑ k ∈ Ico 2 (2 ^ n + 1), f k := by
  convert sum_schlomilch_le' hf (fun n => pow_pos zero_lt_two n)
    (fun m n hm => pow_right_mono₀ one_le_two hm) n using 2
  /-
    case h.e'_3.a
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
    n x✝ : Nat
    a✝ : Membership.mem (Finset.range n) x✝
    ⊢ Eq (HSMul.hSMul (HPow.hPow 2 x✝) (f (HPow.hPow 2 (HAdd.hAdd x✝ 1)))) (HSMul. …
  -/
  simp [pow_succ, mul_two, two_mul]
  /-
    🎉 no goals
  -/


theorem sum_schlomilch_le {C : ℕ} (hf : ∀ ⦃m n⦄, 1 < m → m ≤ n → f n ≤ f m) (h_pos : ∀ n, 0 < u n)
    (h_nonneg : ∀ n, 0 ≤ f n) (hu : Monotone u) (h_succ_diff : SuccDiffBounded C u) (n : ℕ) :
    ∑ k ∈ range (n + 1), (u (k + 1) - u k) • f (u k) ≤
    (u 1 - u 0) • f (u 0) + C • ∑ k ∈ Ico (u 0 + 1) (u n + 1), f k := by
  /-
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    u : Nat → Nat
    C : Nat
    hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    h_nonneg : ∀ (n : Nat), LE.le 0 (f n)
    hu : Monotone u
    h_succ_diff : SuccDiffBounded C u
    n : Nat
    ⊢ LE.le ((Finset.range (HAdd.hAdd n 1)).sum fun k => HSMul.hSMul (HSub.hSub (u …
  -/
  rw [sum_range_succ', add_comm]
  /-
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    u : Nat → Nat
    C : Nat
    hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    h_nonneg : ∀ (n : Nat), LE.le 0 (f n)
    hu : Monotone u
    h_succ_diff : SuccDiffBounded C u
    n : Nat
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul (HSub.hSub (u (HAdd.hAdd 0 1)) (u 0)) (f (u 0) …
  -/
  gcongr
  suffices ∑ k ∈ range n, (u (k + 2) - u (k + 1)) • f (u (k + 1)) ≤
  C • ∑ k ∈ range n, ((u (k + 1) - u k) • f (u (k + 1))) by
    refine this.trans (nsmul_le_nsmul_right ?_ _)
    exact sum_schlomilch_le' hf h_pos hu n
  have : ∀ k ∈ range n, (u (k + 2) - u (k + 1)) • f (u (k + 1)) ≤
    C • ((u (k + 1) - u k) • f (u (k + 1))) := by
    intro k _
    rw [smul_smul]
    gcongr
    · exact h_nonneg (u (k + 1))
    exact mod_cast h_succ_diff k
  /-
    case bc
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    u : Nat → Nat
    C : Nat
    hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    h_nonneg : ∀ (n : Nat), LE.le 0 (f n)
    hu : Monotone u
    h_succ_diff : SuccDiffBounded C u
    n : Nat
    this : ∀ (k : Nat), Membership.mem (Finset.range n) k → LE.le (HSMul.hSMul (HS …
    ⊢ LE.le ((Finset.range n).sum fun k => HSMul.hSMul (HSub.hSub (u (HAdd.hAdd k  …
  -/
  convert sum_le_sum this
  /-
    case h.e'_4
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    u : Nat → Nat
    C : Nat
    hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    h_nonneg : ∀ (n : Nat), LE.le 0 (f n)
    hu : Monotone u
    h_succ_diff : SuccDiffBounded C u
    n : Nat
    this : ∀ (k : Nat), Membership.mem (Finset.range n) k → LE.le (HSMul.hSMul (HS …
    ⊢ Eq (HSMul.hSMul C ((Finset.range n).sum fun k => HSMul.hSMul (HSub.hSub (u ( …
  -/
  simp [smul_sum]
  /-
    🎉 no goals
  -/


theorem sum_condensed_le (hf : ∀ ⦃m n⦄, 1 < m → m ≤ n → f n ≤ f m) (n : ℕ) :
    (∑ k ∈ range (n + 1), 2 ^ k • f (2 ^ k)) ≤ f 1 + 2 • ∑ k ∈ Ico 2 (2 ^ n + 1), f k := by
  /-
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
    n : Nat
    ⊢ LE.le ((Finset.range (HAdd.hAdd n 1)).sum fun k => HSMul.hSMul (HPow.hPow 2  …
  -/
  convert add_le_add_left (nsmul_le_nsmul_right (sum_condensed_le' hf n) 2) (f 1)
  /-
    case h.e'_3
    M : Type u_1
    inst✝ : OrderedAddCommMonoid M
    f : Nat → M
    hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun k => HSMul.hSMul (HPow.hPow 2 k)  …
  -/
  simp [sum_range_succ', add_comm, pow_succ', mul_nsmul', sum_nsmul]
  /-
    🎉 no goals
  -/


open NNReal in
theorem le_tsum_schlomilch (hf : ∀ ⦃m n⦄, 0 < m → m ≤ n → f n ≤ f m) (h_pos : ∀ n, 0 < u n)
    (hu : StrictMono u) :
    ∑' k , f k ≤ ∑ k ∈ range (u 0), f k + ∑' k : ℕ, (u (k + 1) - u k) * f (u k) := by
  /-
    u : Nat → Nat
    f : Nat → ENNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu : StrictMono u
    ⊢ LE.le (tsum fun k => f k) (HAdd.hAdd ((Finset.range (u 0)).sum fun k => f k) …
  -/
  rw [ENNReal.tsum_eq_iSup_nat' hu.tendsto_atTop]
  refine iSup_le fun n =>
    (Finset.le_sum_schlomilch hf h_pos hu.monotone n).trans (add_le_add_left ?_ _)
  have (k : ℕ) : (u (k + 1) - u k : ℝ≥0∞) = (u (k + 1) - (u k : ℕ) : ℕ) := by
    simp [NNReal.coe_sub (Nat.cast_le (α := ℝ≥0).mpr <| (hu k.lt_succ_self).le)]
  /-
    u : Nat → Nat
    f : Nat → ENNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu : StrictMono u
    n : Nat
    this : ∀ (k : Nat), Eq (HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) ↑(HSub.hSub (u  …
    ⊢ LE.le ((Finset.range n).sum fun k => HSMul.hSMul (HSub.hSub (u (HAdd.hAdd k  …
  -/
  simp only [nsmul_eq_mul, this]
  /-
    u : Nat → Nat
    f : Nat → ENNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu : StrictMono u
    n : Nat
    this : ∀ (k : Nat), Eq (HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) ↑(HSub.hSub (u  …
    ⊢ LE.le ((Finset.range n).sum fun x => HMul.hMul (↑(HSub.hSub (u (HAdd.hAdd x  …
  -/
  apply ENNReal.sum_le_tsum
  /-
    🎉 no goals
  -/


theorem le_tsum_condensed (hf : ∀ ⦃m n⦄, 0 < m → m ≤ n → f n ≤ f m) :
    ∑' k, f k ≤ f 0 + ∑' k : ℕ, 2 ^ k * f (2 ^ k) := by
  /-
    f : Nat → ENNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    ⊢ LE.le (tsum fun k => f k) (HAdd.hAdd (f 0) (tsum fun k => HMul.hMul (HPow.hP …
  -/
  rw [ENNReal.tsum_eq_iSup_nat' (Nat.tendsto_pow_atTop_atTop_of_one_lt _root_.one_lt_two)]
  /-
    f : Nat → ENNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    ⊢ LE.le (iSup fun i => (Finset.range (HPow.hPow 2 i)).sum fun a => f a) (HAdd. …
  -/
  refine iSup_le fun n => (Finset.le_sum_condensed hf n).trans (add_le_add_left ?_ _)
  /-
    f : Nat → ENNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    n : Nat
    ⊢ LE.le ((Finset.range n).sum fun k => HSMul.hSMul (HPow.hPow 2 k) (f (HPow.hP …
  -/
  simp only [nsmul_eq_mul, Nat.cast_pow, Nat.cast_two]
  /-
    f : Nat → ENNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    n : Nat
    ⊢ LE.le ((Finset.range n).sum fun x => HMul.hMul (HPow.hPow 2 x) (f (HPow.hPow …
  -/
  apply ENNReal.sum_le_tsum
  /-
    🎉 no goals
  -/


theorem tsum_schlomilch_le {C : ℕ} (hf : ∀ ⦃m n⦄, 1 < m → m ≤ n → f n ≤ f m) (h_pos : ∀ n, 0 < u n)
    (h_nonneg : ∀ n, 0 ≤ f n) (hu : Monotone u) (h_succ_diff : SuccDiffBounded C u) :
    ∑' k : ℕ, (u (k + 1) - u k) * f (u k) ≤ (u 1 - u 0) * f (u 0) + C * ∑' k, f k := by
  /-
    u : Nat → Nat
    f : Nat → ENNReal
    C : Nat
    hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    h_nonneg : ∀ (n : Nat), LE.le 0 (f n)
    hu : Monotone u
    h_succ_diff : SuccDiffBounded C u
    ⊢ LE.le (tsum fun k => HMul.hMul (HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) (f (u …
  -/
  rw [ENNReal.tsum_eq_iSup_nat' (tendsto_atTop_mono Nat.le_succ tendsto_id)]
  refine
    iSup_le fun n =>
      le_trans ?_
        (add_le_add_left
          (mul_le_mul_of_nonneg_left (ENNReal.sum_le_tsum <| Finset.Ico (u 0 + 1) (u n + 1)) ?_) _)
    /-
      case refine_1
      u : Nat → Nat
      f : Nat → ENNReal
      C : Nat
      hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
      h_pos : ∀ (n : Nat), LT.lt 0 (u n)
      h_nonneg : ∀ (n : Nat), LE.le 0 (f n)
      hu : Monotone u
      h_succ_diff : SuccDiffBounded C u
      n : Nat
      ⊢ LE.le ((Finset.range n.succ).sum fun a => HMul.hMul (HSub.hSub ↑(u (HAdd.hAd …
    -/
  · simpa using Finset.sum_schlomilch_le hf h_pos h_nonneg hu h_succ_diff n
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      u : Nat → Nat
      f : Nat → ENNReal
      C : Nat
      hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
      h_pos : ∀ (n : Nat), LT.lt 0 (u n)
      h_nonneg : ∀ (n : Nat), LE.le 0 (f n)
      hu : Monotone u
      h_succ_diff : SuccDiffBounded C u
      n : Nat
      ⊢ LE.le 0 ↑C
    -/
  · exact zero_le _
    /-
      🎉 no goals
    -/


theorem tsum_condensed_le (hf : ∀ ⦃m n⦄, 1 < m → m ≤ n → f n ≤ f m) :
    (∑' k : ℕ, 2 ^ k * f (2 ^ k)) ≤ f 1 + 2 * ∑' k, f k := by
  /-
    f : Nat → ENNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
    ⊢ LE.le (tsum fun k => HMul.hMul (HPow.hPow 2 k) (f (HPow.hPow 2 k))) (HAdd.hA …
  -/
  rw [ENNReal.tsum_eq_iSup_nat' (tendsto_atTop_mono Nat.le_succ tendsto_id), two_mul, ← two_nsmul]
  refine
    iSup_le fun n =>
      le_trans ?_
        (add_le_add_left
          (nsmul_le_nsmul_right (ENNReal.sum_le_tsum <| Finset.Ico 2 (2 ^ n + 1)) _) _)
  /-
    f : Nat → ENNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 1 m → LE.le m n → LE.le (f n) (f m)
    n : Nat
    ⊢ LE.le ((Finset.range n.succ).sum fun a => HMul.hMul (HPow.hPow 2 a) (f (HPow …
  -/
  simpa using Finset.sum_condensed_le hf n
  /-
    🎉 no goals
  -/


open ENNReal in
/-- for a series of `NNReal` version. -/
theorem summable_schlomilch_iff {C : ℕ} {u : ℕ → ℕ} {f : ℕ → ℝ≥0}
    (hf : ∀ ⦃m n⦄, 0 < m → m ≤ n → f n ≤ f m)
    (h_pos : ∀ n, 0 < u n) (hu_strict : StrictMono u)
    (hC_nonzero : C ≠ 0) (h_succ_diff : SuccDiffBounded C u) :
    (Summable fun k : ℕ => (u (k + 1) - (u k : ℝ≥0)) * f (u k)) ↔ Summable f := by
  /-
    C : Nat
    u : Nat → Nat
    f : Nat → NNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu_strict : StrictMono u
    hC_nonzero : Ne C 0
    h_succ_diff : SuccDiffBounded C u
    ⊢ Iff (Summable fun k => HMul.hMul (HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) (f  …
  -/
  simp only [← tsum_coe_ne_top_iff_summable, Ne, not_iff_not, ENNReal.coe_mul]
  /-
    C : Nat
    u : Nat → Nat
    f : Nat → NNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu_strict : StrictMono u
    hC_nonzero : Ne C 0
    h_succ_diff : SuccDiffBounded C u
    ⊢ Iff (Eq (tsum fun b => HMul.hMul ↑(HSub.hSub ↑(u (HAdd.hAdd b 1)) ↑(u b)) ↑( …
  -/
  constructor <;> intro h
  · replace hf : ∀ m n, 1 < m → m ≤ n → (f n : ℝ≥0∞) ≤ f m := fun m n hm hmn =>
      ENNReal.coe_le_coe.2 (hf (zero_lt_one.trans hm) hmn)
    have h_nonneg : ∀ n, 0 ≤ (f n : ℝ≥0∞) := fun n =>
      ENNReal.coe_le_coe.2 (f n).2
    /-
      case mp
      C : Nat
      u : Nat → Nat
      f : Nat → NNReal
      h_pos : ∀ (n : Nat), LT.lt 0 (u n)
      hu_strict : StrictMono u
      hC_nonzero : Ne C 0
      h_succ_diff : SuccDiffBounded C u
      h : Eq (tsum fun b => HMul.hMul ↑(HSub.hSub ↑(u (HAdd.hAdd b 1)) ↑(u b)) ↑(f ( …
      hf : ∀ (m n : Nat), LT.lt 1 m → LE.le m n → LE.le ↑(f n) ↑(f m)
      h_nonneg : ∀ (n : Nat), LE.le 0 ↑(f n)
      ⊢ Eq (tsum fun b => ↑(f b)) Top.top
    -/
    obtain hC := tsum_schlomilch_le hf h_pos h_nonneg hu_strict.monotone h_succ_diff
    /-
      case mp
      C : Nat
      u : Nat → Nat
      f : Nat → NNReal
      h_pos : ∀ (n : Nat), LT.lt 0 (u n)
      hu_strict : StrictMono u
      hC_nonzero : Ne C 0
      h_succ_diff : SuccDiffBounded C u
      h : Eq (tsum fun b => HMul.hMul ↑(HSub.hSub ↑(u (HAdd.hAdd b 1)) ↑(u b)) ↑(f ( …
      hf : ∀ (m n : Nat), LT.lt 1 m → LE.le m n → LE.le ↑(f n) ↑(f m)
      h_nonneg : ∀ (n : Nat), LE.le 0 ↑(f n)
      hC : LE.le (tsum fun k => HMul.hMul (HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) ↑( …
      ⊢ Eq (tsum fun b => ↑(f b)) Top.top
    -/
    simpa [add_eq_top, mul_ne_top, mul_eq_top, hC_nonzero] using eq_top_mono hC h
    /-
      🎉 no goals
    -/
  · replace hf : ∀ m n, 0 < m → m ≤ n → (f n : ℝ≥0∞) ≤ f m := fun m n hm hmn =>
      ENNReal.coe_le_coe.2 (hf hm hmn)
    /-
      case mpr
      C : Nat
      u : Nat → Nat
      f : Nat → NNReal
      h_pos : ∀ (n : Nat), LT.lt 0 (u n)
      hu_strict : StrictMono u
      hC_nonzero : Ne C 0
      h_succ_diff : SuccDiffBounded C u
      h : Eq (tsum fun b => ↑(f b)) Top.top
      hf : ∀ (m n : Nat), LT.lt 0 m → LE.le m n → LE.le ↑(f n) ↑(f m)
      ⊢ Eq (tsum fun b => HMul.hMul ↑(HSub.hSub ↑(u (HAdd.hAdd b 1)) ↑(u b)) ↑(f (u  …
    -/
    have : ∑ k ∈ range (u 0), (f k : ℝ≥0∞) ≠ ∞ := sum_ne_top.2 fun a _ => coe_ne_top
    /-
      case mpr
      C : Nat
      u : Nat → Nat
      f : Nat → NNReal
      h_pos : ∀ (n : Nat), LT.lt 0 (u n)
      hu_strict : StrictMono u
      hC_nonzero : Ne C 0
      h_succ_diff : SuccDiffBounded C u
      h : Eq (tsum fun b => ↑(f b)) Top.top
      hf : ∀ (m n : Nat), LT.lt 0 m → LE.le m n → LE.le ↑(f n) ↑(f m)
      this : Ne ((Finset.range (u 0)).sum fun k => ↑(f k)) Top.top
      ⊢ Eq (tsum fun b => HMul.hMul ↑(HSub.hSub ↑(u (HAdd.hAdd b 1)) ↑(u b)) ↑(f (u  …
    -/
    simpa [h, add_eq_top, this] using le_tsum_schlomilch hf h_pos hu_strict
    /-
      🎉 no goals
    -/


open ENNReal in
theorem summable_condensed_iff {f : ℕ → ℝ≥0} (hf : ∀ ⦃m n⦄, 0 < m → m ≤ n → f n ≤ f m) :
    (Summable fun k : ℕ => (2 : ℝ≥0) ^ k * f (2 ^ k)) ↔ Summable f := by
  have h_succ_diff : SuccDiffBounded 2 (2 ^ ·) := by
    intro n
    simp [pow_succ, mul_two, two_mul]
  convert summable_schlomilch_iff hf (pow_pos zero_lt_two) (pow_right_strictMono₀ _root_.one_lt_two)
    two_ne_zero h_succ_diff
  /-
    case h.e'_1.h.e'_5.h.h.e'_5
    f : Nat → NNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    h_succ_diff : SuccDiffBounded 2 fun x => HPow.hPow 2 x
    x✝ : Nat
    ⊢ Eq (HPow.hPow 2 x✝) (HSub.hSub ↑(HPow.hPow 2 (HAdd.hAdd x✝ 1)) ↑(HPow.hPow 2 …
  -/
  simp [pow_succ, mul_two, two_mul]
  /-
    🎉 no goals
  -/


open NNReal in
/-- for series of nonnegative real numbers. -/
theorem summable_schlomilch_iff_of_nonneg {C : ℕ} {u : ℕ → ℕ} {f : ℕ → ℝ} (h_nonneg : ∀ n, 0 ≤ f n)
    (hf : ∀ ⦃m n⦄, 0 < m → m ≤ n → f n ≤ f m) (h_pos : ∀ n, 0 < u n)
    (hu_strict : StrictMono u) (hC_nonzero : C ≠ 0) (h_succ_diff : SuccDiffBounded C u) :
    (Summable fun k : ℕ => (u (k + 1) - (u k : ℝ)) * f (u k)) ↔ Summable f := by
  /-
    C : Nat
    u : Nat → Nat
    f : Nat → Real
    h_nonneg : ∀ (n : Nat), LE.le 0 (f n)
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu_strict : StrictMono u
    hC_nonzero : Ne C 0
    h_succ_diff : SuccDiffBounded C u
    ⊢ Iff (Summable fun k => HMul.hMul (HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) (f  …
  -/
  lift f to ℕ → ℝ≥0 using h_nonneg
  /-
    case intro
    C : Nat
    u : Nat → Nat
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu_strict : StrictMono u
    hC_nonzero : Ne C 0
    h_succ_diff : SuccDiffBounded C u
    f : Nat → NNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le ((fun i => ↑(f i)) n) ((fun  …
    ⊢ Iff (Summable fun k => HMul.hMul (HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) ((f …
  -/
  simp only [NNReal.coe_le_coe] at *
  have (k : ℕ) : (u (k + 1) - (u k : ℝ)) = ((u (k + 1) : ℝ≥0) - (u k : ℝ≥0) : ℝ≥0) := by
    have := Nat.cast_le (α := ℝ≥0).mpr <| (hu_strict k.lt_succ_self).le
    simp [NNReal.coe_sub this]
  /-
    case intro
    C : Nat
    u : Nat → Nat
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu_strict : StrictMono u
    hC_nonzero : Ne C 0
    h_succ_diff : SuccDiffBounded C u
    f : Nat → NNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    this : ∀ (k : Nat), Eq (HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) ↑(HSub.hSub ↑(u …
    ⊢ Iff (Summable fun k => HMul.hMul (HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) ↑(f …
  -/
  simp_rw [this]
  /-
    case intro
    C : Nat
    u : Nat → Nat
    h_pos : ∀ (n : Nat), LT.lt 0 (u n)
    hu_strict : StrictMono u
    hC_nonzero : Ne C 0
    h_succ_diff : SuccDiffBounded C u
    f : Nat → NNReal
    hf : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    this : ∀ (k : Nat), Eq (HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) ↑(HSub.hSub ↑(u …
    ⊢ Iff (Summable fun k => HMul.hMul ↑(HSub.hSub ↑(u (HAdd.hAdd k 1)) ↑(u k)) ↑( …
  -/
  exact_mod_cast NNReal.summable_schlomilch_iff hf h_pos hu_strict hC_nonzero h_succ_diff
  /-
    🎉 no goals
  -/


/-- Cauchy condensation test for antitone series of nonnegative real numbers. -/
theorem summable_condensed_iff_of_nonneg {f : ℕ → ℝ} (h_nonneg : ∀ n, 0 ≤ f n)
    (h_mono : ∀ ⦃m n⦄, 0 < m → m ≤ n → f n ≤ f m) :
    (Summable fun k : ℕ => (2 : ℝ) ^ k * f (2 ^ k)) ↔ Summable f := by
  have h_succ_diff : SuccDiffBounded 2 (2 ^ ·) := by
    intro n
    simp [pow_succ, mul_two, two_mul]
  convert summable_schlomilch_iff_of_nonneg h_nonneg h_mono (pow_pos zero_lt_two)
    (pow_right_strictMono₀ one_lt_two) two_ne_zero h_succ_diff
  /-
    case h.e'_1.h.e'_5.h.h.e'_5
    f : Nat → Real
    h_nonneg : ∀ (n : Nat), LE.le 0 (f n)
    h_mono : ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (f n) (f m)
    h_succ_diff : SuccDiffBounded 2 fun x => HPow.hPow 2 x
    x✝ : Nat
    ⊢ Eq (HPow.hPow 2 x✝) (HSub.hSub ↑(HPow.hPow 2 (HAdd.hAdd x✝ 1)) ↑(HPow.hPow 2 …
  -/
  simp [pow_succ, mul_two, two_mul]
  /-
    🎉 no goals
  -/


/-- Test for convergence of the `p`-series: the real-valued series `∑' n : ℕ, (n ^ p)⁻¹` converges
if and only if `1 < p`. -/
@[simp]
theorem summable_nat_rpow_inv {p : ℝ} :
    Summable (fun n => ((n : ℝ) ^ p)⁻¹ : ℕ → ℝ) ↔ 1 < p := by
  /-
    p : Real
    ⊢ Iff (Summable fun n => Inv.inv (HPow.hPow (↑n) p)) (LT.lt 1 p)
  -/
  rcases le_or_lt 0 p with hp | hp
  /- Cauchy condensation test applies only to antitone sequences, so we consider the
    cases `0 ≤ p` and `p < 0` separately. -/
    /-
      case inl
      p : Real
      hp : LE.le 0 p
      ⊢ Iff (Summable fun n => Inv.inv (HPow.hPow (↑n) p)) (LT.lt 1 p)
    -/
  · rw [← summable_condensed_iff_of_nonneg]
    · simp_rw [Nat.cast_pow, Nat.cast_two, ← rpow_natCast, ← rpow_mul zero_lt_two.le, mul_comm _ p,
        rpow_mul zero_lt_two.le, rpow_natCast, ← inv_pow, ← mul_pow,
        summable_geometric_iff_norm_lt_one]
      /-
        case inl
        p : Real
        hp : LE.le 0 p
        ⊢ Iff (LT.lt (Norm.norm (HMul.hMul 2 (Inv.inv (HPow.hPow 2 p)))) 1) (LT.lt 1 p)
      -/
      nth_rw 1 [← rpow_one 2]
      rw [← division_def, ← rpow_sub zero_lt_two, norm_eq_abs,
        abs_of_pos (rpow_pos_of_pos zero_lt_two _), rpow_lt_one_iff zero_lt_two.le]
      /-
        case inl
        p : Real
        hp : LE.le 0 p
        ⊢ Iff (Or (And (Eq 2 0) (Ne (HSub.hSub 1 p) 0)) (Or (And (LT.lt 1 2) (LT.lt (H …
      -/
      norm_num
      /-
        🎉 no goals
      -/
      /-
        case inl.h_nonneg
        p : Real
        hp : LE.le 0 p
        ⊢ ∀ (n : Nat), LE.le 0 (Inv.inv (HPow.hPow (↑n) p))
      -/
    · intro n
      /-
        case inl.h_nonneg
        p : Real
        hp : LE.le 0 p
        n : Nat
        ⊢ LE.le 0 (Inv.inv (HPow.hPow (↑n) p))
      -/
      positivity
      /-
        🎉 no goals
      -/
      /-
        case inl.h_mono
        p : Real
        hp : LE.le 0 p
        ⊢ ∀ ⦃m n : Nat⦄, LT.lt 0 m → LE.le m n → LE.le (Inv.inv (HPow.hPow (↑n) p)) (I …
      -/
    · intro m n hm hmn
      /-
        case inl.h_mono
        p : Real
        hp : LE.le 0 p
        m n : Nat
        hm : LT.lt 0 m
        hmn : LE.le m n
        ⊢ LE.le (Inv.inv (HPow.hPow (↑n) p)) (Inv.inv (HPow.hPow (↑m) p))
      -/
      gcongr
      /-
        🎉 no goals
      -/
  -- If `p < 0`, then `1 / n ^ p` tends to infinity, thus the series diverges.
  · suffices ¬Summable (fun n => ((n : ℝ) ^ p)⁻¹ : ℕ → ℝ) by
      have : ¬1 < p := fun hp₁ => hp.not_le (zero_le_one.trans hp₁.le)
      simpa only [this, iff_false]
    /-
      case inr
      p : Real
      hp : LT.lt p 0
      ⊢ Not (Summable fun n => Inv.inv (HPow.hPow (↑n) p))
    -/
    intro h
    obtain ⟨k : ℕ, hk₁ : ((k : ℝ) ^ p)⁻¹ < 1, hk₀ : k ≠ 0⟩ :=
      ((h.tendsto_cofinite_zero.eventually (gt_mem_nhds zero_lt_one)).and
          (eventually_cofinite_ne 0)).exists
    /-
      case inr.intro.intro
      p : Real
      hp : LT.lt p 0
      h : Summable fun n => Inv.inv (HPow.hPow (↑n) p)
      k : Nat
      hk₁ : LT.lt (Inv.inv (HPow.hPow (↑k) p)) 1
      hk₀ : Ne k 0
      ⊢ False
    -/
    apply hk₀
    /-
      case inr.intro.intro
      p : Real
      hp : LT.lt p 0
      h : Summable fun n => Inv.inv (HPow.hPow (↑n) p)
      k : Nat
      hk₁ : LT.lt (Inv.inv (HPow.hPow (↑k) p)) 1
      hk₀ : Ne k 0
      ⊢ Eq k 0
    -/
    rw [← pos_iff_ne_zero, ← @Nat.cast_pos ℝ] at hk₀
    simpa [inv_lt_one₀ (rpow_pos_of_pos hk₀ _), one_lt_rpow_iff_of_pos hk₀, hp,
      hp.not_lt, hk₀] using hk₁


@[simp]
theorem summable_nat_rpow {p : ℝ} : Summable (fun n => (n : ℝ) ^ p : ℕ → ℝ) ↔ p < -1 := by
  /-
    p : Real
    ⊢ Iff (Summable fun n => HPow.hPow (↑n) p) (LT.lt p (-1))
  -/
  rcases neg_surjective p with ⟨p, rfl⟩
  /-
    case intro
    p : Real
    ⊢ Iff (Summable fun n => HPow.hPow (↑n) (Neg.neg p)) (LT.lt (Neg.neg p) (-1))
  -/
  simp [rpow_neg]
  /-
    🎉 no goals
  -/


/-- Test for convergence of the `p`-series: the real-valued series `∑' n : ℕ, 1 / n ^ p` converges
if and only if `1 < p`. -/
theorem summable_one_div_nat_rpow {p : ℝ} :
    Summable (fun n => 1 / (n : ℝ) ^ p : ℕ → ℝ) ↔ 1 < p := by
  /-
    p : Real
    ⊢ Iff (Summable fun n => HDiv.hDiv 1 (HPow.hPow (↑n) p)) (LT.lt 1 p)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Test for convergence of the `p`-series: the real-valued series `∑' n : ℕ, (n ^ p)⁻¹` converges
if and only if `1 < p`. -/
@[simp]
theorem summable_nat_pow_inv {p : ℕ} :
    Summable (fun n => ((n : ℝ) ^ p)⁻¹ : ℕ → ℝ) ↔ 1 < p := by
  /-
    p : Nat
    ⊢ Iff (Summable fun n => Inv.inv (HPow.hPow (↑n) p)) (LT.lt 1 p)
  -/
  simp only [← rpow_natCast, summable_nat_rpow_inv, Nat.one_lt_cast]
  /-
    🎉 no goals
  -/


/-- Test for convergence of the `p`-series: the real-valued series `∑' n : ℕ, 1 / n ^ p` converges
if and only if `1 < p`. -/
theorem summable_one_div_nat_pow {p : ℕ} :
    Summable (fun n => 1 / (n : ℝ) ^ p : ℕ → ℝ) ↔ 1 < p := by
  /-
    p : Nat
    ⊢ Iff (Summable fun n => HDiv.hDiv 1 (HPow.hPow (↑n) p)) (LT.lt 1 p)
  -/
  simp only [one_div, Real.summable_nat_pow_inv]
  /-
    🎉 no goals
  -/


/-- Summability of the `p`-series over `ℤ`. -/
theorem summable_one_div_int_pow {p : ℕ} :
    (Summable fun n : ℤ ↦ 1 / (n : ℝ) ^ p) ↔ 1 < p := by
  refine ⟨fun h ↦ summable_one_div_nat_pow.mp (h.comp_injective Nat.cast_injective),
    fun h ↦ .of_nat_of_neg (summable_one_div_nat_pow.mpr h)
      (((summable_one_div_nat_pow.mpr h).mul_left <| 1 / (-1 : ℝ) ^ p).congr fun n ↦ ?_)⟩
  /-
    p : Nat
    h : LT.lt 1 p
    n : Nat
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (-1) p)) (HDiv.hDiv 1 (HPow.hPow (↑n)  …
  -/
  rw [Int.cast_neg, Int.cast_natCast, neg_eq_neg_one_mul (n : ℝ), mul_pow, mul_one_div, div_div]
  /-
    🎉 no goals
  -/


theorem summable_abs_int_rpow {b : ℝ} (hb : 1 < b) :
    Summable fun n : ℤ => |(n : ℝ)| ^ (-b) := by
  /-
    b : Real
    hb : LT.lt 1 b
    ⊢ Summable fun n => HPow.hPow (abs ↑n) (Neg.neg b)
  -/
  apply Summable.of_nat_of_neg
  /-
    case hf₁
    b : Real
    hb : LT.lt 1 b
    ⊢ Summable fun n => HPow.hPow (abs ↑↑n) (Neg.neg b)
  -/
  on_goal 2 => simp_rw [Int.cast_neg, abs_neg]
  all_goals
    simp_rw [Int.cast_natCast, fun n : ℕ => abs_of_nonneg (n.cast_nonneg : 0 ≤ (n : ℝ))]
    rwa [summable_nat_rpow, neg_lt_neg_iff]


/-- Harmonic series is not unconditionally summable. -/
theorem not_summable_natCast_inv : ¬Summable (fun n => n⁻¹ : ℕ → ℝ) := by
  have : ¬Summable (fun n => ((n : ℝ) ^ 1)⁻¹ : ℕ → ℝ) :=
    mt (summable_nat_pow_inv (p := 1)).1 (lt_irrefl 1)
  /-
    this : Not (Summable fun n => Inv.inv (HPow.hPow (↑n) 1))
    ⊢ Not (Summable fun n => Inv.inv ↑n)
  -/
  simpa
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias not_summable_nat_cast_inv := not_summable_natCast_inv


/-- Harmonic series is not unconditionally summable. -/
theorem not_summable_one_div_natCast : ¬Summable (fun n => 1 / n : ℕ → ℝ) := by
  /-
    ⊢ Not (Summable fun n => HDiv.hDiv 1 ↑n)
  -/
  simpa only [inv_eq_one_div] using not_summable_natCast_inv
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias not_summable_one_div_nat_cast := not_summable_one_div_natCast


/-- **Divergence of the Harmonic Series** -/
theorem tendsto_sum_range_one_div_nat_succ_atTop :
    Tendsto (fun n => ∑ i ∈ Finset.range n, (1 / (i + 1) : ℝ)) atTop atTop := by
  /-
    ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => HDiv.hDiv 1 (HAdd.hAd …
  -/
  rw [← not_summable_iff_tendsto_nat_atTop_of_nonneg]
    /-
      ⊢ Not (Summable fun i => HDiv.hDiv 1 (HAdd.hAdd (↑i) 1))
    -/
  · exact_mod_cast mt (_root_.summable_nat_add_iff 1).1 not_summable_one_div_natCast
    /-
      🎉 no goals
    -/
    /-
      ⊢ ∀ (n : Nat), LE.le 0 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))
    -/
  · exact fun i => by positivity
    /-
      🎉 no goals
    -/


@[simp]
theorem summable_rpow_inv {p : ℝ} :
    Summable (fun n => ((n : ℝ≥0) ^ p)⁻¹ : ℕ → ℝ≥0) ↔ 1 < p := by
  /-
    p : Real
    ⊢ Iff (Summable fun n => Inv.inv (HPow.hPow (↑n) p)) (LT.lt 1 p)
  -/
  simp [← NNReal.summable_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem summable_rpow {p : ℝ} : Summable (fun n => (n : ℝ≥0) ^ p : ℕ → ℝ≥0) ↔ p < -1 := by
  /-
    p : Real
    ⊢ Iff (Summable fun n => HPow.hPow (↑n) p) (LT.lt p (-1))
  -/
  simp [← NNReal.summable_coe]
  /-
    🎉 no goals
  -/


theorem summable_one_div_rpow {p : ℝ} :
    Summable (fun n => 1 / (n : ℝ≥0) ^ p : ℕ → ℝ≥0) ↔ 1 < p := by
  /-
    p : Real
    ⊢ Iff (Summable fun n => HDiv.hDiv 1 (HPow.hPow (↑n) p)) (LT.lt 1 p)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem sum_Ioc_inv_sq_le_sub {k n : ℕ} (hk : k ≠ 0) (h : k ≤ n) :
    (∑ i ∈ Ioc k n, ((i : α) ^ 2)⁻¹) ≤ (k : α)⁻¹ - (n : α)⁻¹ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    k n : Nat
    hk : Ne k 0
    h : LE.le k n
    ⊢ LE.le ((Finset.Ioc k n).sum fun i => Inv.inv (HPow.hPow (↑i) 2)) (HSub.hSub  …
  -/
  refine Nat.le_induction ?_ ?_ n h
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedField α
      k n : Nat
      hk : Ne k 0
      h : LE.le k n
      ⊢ LE.le ((Finset.Ioc k k).sum fun i => Inv.inv (HPow.hPow (↑i) 2)) (HSub.hSub  …
    -/
  · simp only [Ioc_self, sum_empty, sub_self, le_refl]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    inst✝ : LinearOrderedField α
    k n : Nat
    hk : Ne k 0
    h : LE.le k n
    ⊢ ∀ (n : Nat), LE.le k n → LE.le ((Finset.Ioc k n).sum fun i => Inv.inv (HPow. …
  -/
  intro n hn IH
  /-
    case refine_2
    α : Type u_1
    inst✝ : LinearOrderedField α
    k n✝ : Nat
    hk : Ne k 0
    h : LE.le k n✝
    n : Nat
    hn : LE.le k n
    IH : LE.le ((Finset.Ioc k n).sum fun i => Inv.inv (HPow.hPow (↑i) 2)) (HSub.hS …
    ⊢ LE.le ((Finset.Ioc k (HAdd.hAdd n 1)).sum fun i => Inv.inv (HPow.hPow (↑i) 2 …
  -/
  rw [sum_Ioc_succ_top hn]
  /-
    case refine_2
    α : Type u_1
    inst✝ : LinearOrderedField α
    k n✝ : Nat
    hk : Ne k 0
    h : LE.le k n✝
    n : Nat
    hn : LE.le k n
    IH : LE.le ((Finset.Ioc k n).sum fun i => Inv.inv (HPow.hPow (↑i) 2)) (HSub.hS …
    ⊢ LE.le (HAdd.hAdd ((Finset.Ioc k n).sum fun k => Inv.inv (HPow.hPow (↑k) 2))  …
  -/
  apply (add_le_add IH le_rfl).trans
  simp only [sub_eq_add_neg, add_assoc, Nat.cast_add, Nat.cast_one, le_add_neg_iff_add_le,
    add_le_iff_nonpos_right, neg_add_le_iff_le_add, add_zero]
  /-
    case refine_2
    α : Type u_1
    inst✝ : LinearOrderedField α
    k n✝ : Nat
    hk : Ne k 0
    h : LE.le k n✝
    n : Nat
    hn : LE.le k n
    IH : LE.le ((Finset.Ioc k n).sum fun i => Inv.inv (HPow.hPow (↑i) 2)) (HSub.hS …
    ⊢ LE.le (HAdd.hAdd (Inv.inv (HPow.hPow (HAdd.hAdd (↑n) 1) 2)) (Inv.inv (HAdd.h …
  -/
  have A : 0 < (n : α) := by simpa using hk.bot_lt.trans_le hn
  /-
    case refine_2
    α : Type u_1
    inst✝ : LinearOrderedField α
    k n✝ : Nat
    hk : Ne k 0
    h : LE.le k n✝
    n : Nat
    hn : LE.le k n
    IH : LE.le ((Finset.Ioc k n).sum fun i => Inv.inv (HPow.hPow (↑i) 2)) (HSub.hS …
    A : LT.lt 0 ↑n
    ⊢ LE.le (HAdd.hAdd (Inv.inv (HPow.hPow (HAdd.hAdd (↑n) 1) 2)) (Inv.inv (HAdd.h …
  -/
  field_simp
  /-
    case refine_2
    α : Type u_1
    inst✝ : LinearOrderedField α
    k n✝ : Nat
    hk : Ne k 0
    h : LE.le k n✝
    n : Nat
    hn : LE.le k n
    IH : LE.le ((Finset.Ioc k n).sum fun i => Inv.inv (HPow.hPow (↑i) 2)) (HSub.hS …
    A : LT.lt 0 ↑n
    ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (HAdd.hAdd (↑n) 1) (HPow.hPow (HAdd.hAdd (↑n) 1) …
  -/
  rw [div_le_div_iff₀ _ A]
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedField α
      k n✝ : Nat
      hk : Ne k 0
      h : LE.le k n✝
      n : Nat
      hn : LE.le k n
      IH : LE.le ((Finset.Ioc k n).sum fun i => Inv.inv (HPow.hPow (↑i) 2)) (HSub.hS …
      A : LT.lt 0 ↑n
      ⊢ LE.le (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑n) 1) (HPow.hPow (HAdd.hAdd (↑n) 1) …
    -/
  · linarith
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      k n✝ : Nat
      hk : Ne k 0
      h : LE.le k n✝
      n : Nat
      hn : LE.le k n
      IH : LE.le ((Finset.Ioc k n).sum fun i => Inv.inv (HPow.hPow (↑i) 2)) (HSub.hS …
      A : LT.lt 0 ↑n
      ⊢ LT.lt 0 (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) 2) (HAdd.hAdd (↑n) 1))
    -/
  · positivity
    /-
      🎉 no goals
    -/


theorem sum_Ioo_inv_sq_le (k n : ℕ) : (∑ i ∈ Ioo k n, (i ^ 2 : α)⁻¹) ≤ 2 / (k + 1) :=
  calc
    (∑ i ∈ Ioo k n, ((i : α) ^ 2)⁻¹) ≤ ∑ i ∈ Ioc k (max (k + 1) n), ((i : α) ^ 2)⁻¹ := by
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        k n : Nat
        ⊢ LE.le ((Finset.Ioo k n).sum fun i => Inv.inv (HPow.hPow (↑i) 2)) ((Finset.Io …
      -/
      apply sum_le_sum_of_subset_of_nonneg
        /-
          case h
          α : Type u_1
          inst✝ : LinearOrderedField α
          k n : Nat
          ⊢ HasSubset.Subset (Finset.Ioo k n) (Finset.Ioc k (Max.max (HAdd.hAdd k 1) n))
        -/
      · intro x hx
        /-
          case h
          α : Type u_1
          inst✝ : LinearOrderedField α
          k n x : Nat
          hx : Membership.mem (Finset.Ioo k n) x
          ⊢ Membership.mem (Finset.Ioc k (Max.max (HAdd.hAdd k 1) n)) x
        -/
        simp only [mem_Ioo] at hx
        /-
          case h
          α : Type u_1
          inst✝ : LinearOrderedField α
          k n x : Nat
          hx : And (LT.lt k x) (LT.lt x n)
          ⊢ Membership.mem (Finset.Ioc k (Max.max (HAdd.hAdd k 1) n)) x
        -/
        simp only [hx, hx.2.le, mem_Ioc, le_max_iff, or_true, and_self_iff]
        /-
          🎉 no goals
        -/
        /-
          case hf
          α : Type u_1
          inst✝ : LinearOrderedField α
          k n : Nat
          ⊢ ∀ (i : Nat), Membership.mem (Finset.Ioc k (Max.max (HAdd.hAdd k 1) n)) i → N …
        -/
      · intro i _hi _hident
        /-
          case hf
          α : Type u_1
          inst✝ : LinearOrderedField α
          k n i : Nat
          _hi : Membership.mem (Finset.Ioc k (Max.max (HAdd.hAdd k 1) n)) i
          _hident : Not (Membership.mem (Finset.Ioo k n) i)
          ⊢ LE.le 0 (Inv.inv (HPow.hPow (↑i) 2))
        -/
        positivity
        /-
          🎉 no goals
        -/
    _ ≤ ((k + 1 : α) ^ 2)⁻¹ + ∑ i ∈ Ioc k.succ (max (k + 1) n), ((i : α) ^ 2)⁻¹ := by
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        k n : Nat
        ⊢ LE.le ((Finset.Ioc k (Max.max (HAdd.hAdd k 1) n)).sum fun i => Inv.inv (HPow …
      -/
      rw [← Nat.Icc_succ_left, ← Nat.Ico_succ_right, sum_eq_sum_Ico_succ_bot]
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        k n : Nat
        ⊢ LE.le (HAdd.hAdd (Inv.inv (HPow.hPow (↑k.succ) 2)) ((Finset.Ico (HAdd.hAdd k …
      -/
      swap; · exact Nat.succ_lt_succ ((Nat.lt_succ_self k).trans_le (le_max_left _ _))
              /-
                🎉 no goals
              -/
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        k n : Nat
        ⊢ LE.le (HAdd.hAdd (Inv.inv (HPow.hPow (↑k.succ) 2)) ((Finset.Ico (HAdd.hAdd k …
      -/
      rw [Nat.Ico_succ_right, Nat.Icc_succ_left, Nat.cast_succ]
      /-
        🎉 no goals
      -/
    _ ≤ ((k + 1 : α) ^ 2)⁻¹ + (k + 1 : α)⁻¹ := by
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        k n : Nat
        ⊢ LE.le (HAdd.hAdd (Inv.inv (HPow.hPow (HAdd.hAdd (↑k) 1) 2)) ((Finset.Ioc k.s …
      -/
      refine add_le_add le_rfl ((sum_Ioc_inv_sq_le_sub ?_ (le_max_left _ _)).trans ?_)
        /-
          case refine_1
          α : Type u_1
          inst✝ : LinearOrderedField α
          k n : Nat
          ⊢ Ne k.succ 0
        -/
      · simp only [Ne, Nat.succ_ne_zero, not_false_iff]
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          α : Type u_1
          inst✝ : LinearOrderedField α
          k n : Nat
          ⊢ LE.le (HSub.hSub (Inv.inv ↑k.succ) (Inv.inv ↑(Max.max k.succ n))) (Inv.inv ( …
        -/
      · simp only [Nat.cast_succ, one_div, sub_le_self_iff, inv_nonneg, Nat.cast_nonneg]
        /-
          🎉 no goals
        -/
    _ ≤ 1 / (k + 1) + 1 / (k + 1) := by
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        k n : Nat
        ⊢ LE.le (HAdd.hAdd (Inv.inv (HPow.hPow (HAdd.hAdd (↑k) 1) 2)) (Inv.inv (HAdd.h …
      -/
      have A : (1 : α) ≤ k + 1 := by simp only [le_add_iff_nonneg_left, Nat.cast_nonneg]
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        k n : Nat
        A : LE.le 1 (HAdd.hAdd (↑k) 1)
        ⊢ LE.le (HAdd.hAdd (Inv.inv (HPow.hPow (HAdd.hAdd (↑k) 1) 2)) (Inv.inv (HAdd.h …
      -/
      simp_rw [← one_div]
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        k n : Nat
        A : LE.le 1 (HAdd.hAdd (↑k) 1)
        ⊢ LE.le (HAdd.hAdd (HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd (↑k) 1) 2)) (HDiv.hDiv 1 …
      -/
      gcongr
      /-
        case bc.h
        α : Type u_1
        inst✝ : LinearOrderedField α
        k n : Nat
        A : LE.le 1 (HAdd.hAdd (↑k) 1)
        ⊢ LE.le (HAdd.hAdd (↑k) 1) (HPow.hPow (HAdd.hAdd (↑k) 1) 2)
      -/
      simpa using pow_right_mono₀ A one_le_two
      /-
        🎉 no goals
      -/
                          /-
                            α : Type u_1
                            inst✝ : LinearOrderedField α
                            k n : Nat
                            ⊢ Eq (HAdd.hAdd (HDiv.hDiv 1 (HAdd.hAdd (↑k) 1)) (HDiv.hDiv 1 (HAdd.hAdd (↑k)  …
                          -/
    _ = 2 / (k + 1) := by ring
                          /-
                            🎉 no goals
                          -/


open Set Nat in
/-- The harmonic series restricted to a residue class is not summable. -/
lemma Real.not_summable_indicator_one_div_natCast {m : ℕ} (hm : m ≠ 0) (k : ZMod m) :
    ¬ Summable ({n : ℕ | (n : ZMod m) = k}.indicator fun n : ℕ ↦ (1 / n : ℝ)) := by
  /-
    m : Nat
    hm : Ne m 0
    k : ZMod m
    ⊢ Not (Summable ((setOf fun n => Eq (↑n) k).indicator fun n => HDiv.hDiv 1 ↑n))
  -/
  have : NeZero m := ⟨hm⟩ -- instance is needed below
  /-
    m : Nat
    hm : Ne m 0
    k : ZMod m
    this : NeZero m
    ⊢ Not (Summable ((setOf fun n => Eq (↑n) k).indicator fun n => HDiv.hDiv 1 ↑n))
  -/
  rw [← summable_nat_add_iff 1] -- shift by one to avoid non-monotonicity at zero
  have h (n : ℕ) : {n : ℕ | (n : ZMod m) = k - 1}.indicator (fun n : ℕ ↦ (1 / (n + 1 :) : ℝ)) n =
      if (n : ZMod m) = k - 1 then (1 / (n + 1) : ℝ) else (0 : ℝ) := by
    simp only [indicator_apply, mem_setOf_eq, cast_add, cast_one]
  /-
    m : Nat
    hm : Ne m 0
    k : ZMod m
    this : NeZero m
    h : ∀ (n : Nat), Eq ((setOf fun n => Eq (↑n) (HSub.hSub k 1)).indicator (fun n …
    ⊢ Not (Summable fun n => (setOf fun n => Eq (↑n) k).indicator (fun n => HDiv.h …
  -/
  simp_rw [indicator_apply, mem_setOf, cast_add, cast_one, ← eq_sub_iff_add_eq, ← h]
  /-
    m : Nat
    hm : Ne m 0
    k : ZMod m
    this : NeZero m
    h : ∀ (n : Nat), Eq ((setOf fun n => Eq (↑n) (HSub.hSub k 1)).indicator (fun n …
    ⊢ Not (Summable fun n => (setOf fun n => Eq (↑n) (HSub.hSub k 1)).indicator (f …
  -/
  rw [summable_indicator_mod_iff (fun n₁ n₂ h ↦ by gcongr) (k - 1)]
  /-
    m : Nat
    hm : Ne m 0
    k : ZMod m
    this : NeZero m
    h : ∀ (n : Nat), Eq ((setOf fun n => Eq (↑n) (HSub.hSub k 1)).indicator (fun n …
    ⊢ Not (Summable fun n => HDiv.hDiv 1 ↑(HAdd.hAdd n 1))
  -/
  exact mt (summable_nat_add_iff (f := fun n : ℕ ↦ 1 / (n : ℝ)) 1).mp not_summable_one_div_natCast
  /-
    🎉 no goals
  -/


lemma Real.summable_one_div_nat_add_rpow (a : ℝ) (s : ℝ) :
    Summable (fun n : ℕ ↦ 1 / |n + a| ^ s) ↔ 1 < s := by
  suffices ∀ (b c : ℝ), Summable (fun n : ℕ ↦ 1 / |n + b| ^ s) →
      Summable (fun n : ℕ ↦ 1 / |n + c| ^ s) by
    simp_rw [← summable_one_div_nat_rpow, Iff.intro (this a 0) (this 0 a), add_zero, Nat.abs_cast]
  /-
    a s : Real
    ⊢ ∀ (b c : Real), (Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑ …
  -/
  refine fun b c h ↦ summable_of_isBigO_nat h (isBigO_of_div_tendsto_nhds ?_ 1 ?_)
    /-
      case refine_1
      a s b c : Real
      h : Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)
      ⊢ Filter.Eventually (fun x => Eq (HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑x)  …
    -/
  · filter_upwards [eventually_gt_atTop (Nat.ceil |b|)] with n hn hx
    /-
      case h
      a s b c : Real
      h : Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)
      n : Nat
      hn : LT.lt (Nat.ceil (abs b)) n
      hx : Eq (HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)) 0
      ⊢ Eq (HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) c)) s)) 0
    -/
    have hna : 0 < n + b := by linarith [lt_of_abs_lt ((abs_neg b).symm ▸ Nat.lt_of_ceil_lt hn)]
    /-
      case h
      a s b c : Real
      h : Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)
      n : Nat
      hn : LT.lt (Nat.ceil (abs b)) n
      hx : Eq (HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)) 0
      hna : LT.lt 0 (HAdd.hAdd (↑n) b)
      ⊢ Eq (HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) c)) s)) 0
    -/
    exfalso
    /-
      case h
      a s b c : Real
      h : Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)
      n : Nat
      hn : LT.lt (Nat.ceil (abs b)) n
      hx : Eq (HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)) 0
      hna : LT.lt 0 (HAdd.hAdd (↑n) b)
      ⊢ False
    -/
    revert hx
    /-
      case h
      a s b c : Real
      h : Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)
      n : Nat
      hn : LT.lt (Nat.ceil (abs b)) n
      hna : LT.lt 0 (HAdd.hAdd (↑n) b)
      ⊢ Eq (HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)) 0 → False
    -/
    positivity
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a s b c : Real
      h : Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)
      ⊢ Filter.Tendsto (HDiv.hDiv (fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd ( …
    -/
  · simp_rw [Pi.div_def, div_div, mul_one_div, one_div_div]
    refine (?_ : Tendsto (fun x : ℝ ↦ |x + b| ^ s / |x + c| ^ s) atTop (𝓝 1)).comp
      tendsto_natCast_atTop_atTop
    have : Tendsto (fun x : ℝ ↦ 1 + (b - c) / x) atTop (𝓝 1) := by
      simpa using tendsto_const_nhds.add ((tendsto_const_nhds (X := ℝ)).div_atTop tendsto_id)
    have : Tendsto (fun x ↦ (x + b) / (x + c)) atTop (𝓝 1) := by
      refine (this.comp (tendsto_id.atTop_add (tendsto_const_nhds (x := c)))).congr' ?_
      filter_upwards [eventually_gt_atTop (-c)] with x hx
      field_simp [(by linarith : 0 < x + c).ne']
    /-
      case refine_2
      a s b c : Real
      h : Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)
      this✝ : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv (HSub.hSub b c) x)) Fi …
      this : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd x b) (HAdd.hAdd x c)) Fil …
      ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow (abs (HAdd.hAdd x b)) s) (HPow …
    -/
    apply (one_rpow s ▸ (continuousAt_rpow_const _ s (by simp)).tendsto.comp this).congr'
    /-
      case refine_2
      a s b c : Real
      h : Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)
      this✝ : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv (HSub.hSub b c) x)) Fi …
      this : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd x b) (HAdd.hAdd x c)) Fil …
      ⊢ Filter.atTop.EventuallyEq (Function.comp (fun x => HPow.hPow x s) fun x => H …
    -/
    filter_upwards [eventually_gt_atTop (-b), eventually_gt_atTop (-c)] with x hb hc
    /-
      case h
      a s b c : Real
      h : Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)
      this✝ : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv (HSub.hSub b c) x)) Fi …
      this : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd x b) (HAdd.hAdd x c)) Fil …
      x : Real
      hb : LT.lt (Neg.neg b) x
      hc : LT.lt (Neg.neg c) x
      ⊢ Eq (Function.comp (fun x => HPow.hPow x s) (fun x => HDiv.hDiv (HAdd.hAdd x  …
    -/
    rw [neg_lt_iff_pos_add] at hb hc
    /-
      case h
      a s b c : Real
      h : Summable fun n => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑n) b)) s)
      this✝ : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv (HSub.hSub b c) x)) Fi …
      this : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd x b) (HAdd.hAdd x c)) Fil …
      x : Real
      hb : LT.lt 0 (HAdd.hAdd x b)
      hc : LT.lt 0 (HAdd.hAdd x c)
      ⊢ Eq (Function.comp (fun x => HPow.hPow x s) (fun x => HDiv.hDiv (HAdd.hAdd x  …
    -/
    rw [Function.comp_apply, div_rpow hb.le hc.le, abs_of_pos hb, abs_of_pos hc]
    /-
      🎉 no goals
    -/


lemma Real.summable_one_div_int_add_rpow (a : ℝ) (s : ℝ) :
    Summable (fun n : ℤ ↦ 1 / |n + a| ^ s) ↔ 1 < s := by
  simp_rw [summable_int_iff_summable_nat_and_neg, ← abs_neg (↑(-_ : ℤ) + a), neg_add,
    Int.cast_neg, neg_neg, Int.cast_natCast, summable_one_div_nat_add_rpow, and_self]


theorem summable_pow_div_add {α : Type*} (x : α) [RCLike α] (q k : ℕ) (hq : 1 < q) :
    Summable fun n : ℕ => ‖(x / (↑n + k) ^ q)‖ := by
  /-
    α : Type u_1
    x : α
    inst✝ : RCLike α
    q k : Nat
    hq : LT.lt 1 q
    ⊢ Summable fun n => Norm.norm (HDiv.hDiv x (HPow.hPow (HAdd.hAdd ↑n ↑k) q))
  -/
  simp_rw [norm_div]
  /-
    α : Type u_1
    x : α
    inst✝ : RCLike α
    q k : Nat
    hq : LT.lt 1 q
    ⊢ Summable fun n => HDiv.hDiv (Norm.norm x) (Norm.norm (HPow.hPow (HAdd.hAdd ↑ …
  -/
  apply Summable.const_div
  simpa [hq, Nat.cast_add, one_div, norm_inv, norm_pow, Complex.norm_eq_abs,
    RCLike.norm_natCast, Real.summable_nat_pow_inv, iff_true]
    using summable_nat_add_iff (f := fun x => ‖1 / (x ^ q : α)‖) k


