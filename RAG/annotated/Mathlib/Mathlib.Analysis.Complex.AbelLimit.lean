/-- The Stolz set for a given `M`, roughly teardrop-shaped with the tip at 1 but tending to the
open unit disc as `M` tends to infinity. -/
def stolzSet (M : ℝ) : Set ℂ := {z | ‖z‖ < 1 ∧ ‖1 - z‖ < M * (1 - ‖z‖)}


/-- The cone to the left of `1` with angle `2θ` such that `tan θ = s`. -/
def stolzCone (s : ℝ) : Set ℂ := {z | |z.im| < s * (1 - z.re)}


theorem stolzSet_empty {M : ℝ} (hM : M ≤ 1) : stolzSet M = ∅ := by
  /-
    M : Real
    hM : LE.le M 1
    ⊢ Eq (Complex.stolzSet M) EmptyCollection.emptyCollection
  -/
  ext z
  /-
    case h
    M : Real
    hM : LE.le M 1
    z : Complex
    ⊢ Iff (Membership.mem (Complex.stolzSet M) z) (Membership.mem EmptyCollection. …
  -/
  rw [stolzSet, Set.mem_setOf, Set.mem_empty_iff_false, iff_false, not_and, not_lt, ← sub_pos]
  /-
    case h
    M : Real
    hM : LE.le M 1
    z : Complex
    ⊢ LT.lt 0 (HSub.hSub 1 (Norm.norm z)) → LE.le (HMul.hMul M (HSub.hSub 1 (Norm. …
  -/
  intro zn
  calc
    _ ≤ 1 * (1 - ‖z‖) := mul_le_mul_of_nonneg_right hM zn.le
    _ = ‖(1 : ℂ)‖ - ‖z‖ := by rw [one_mul, norm_one]
    _ ≤ _ := norm_sub_norm_le _ _


theorem nhdsWithin_lt_le_nhdsWithin_stolzSet {M : ℝ} (hM : 1 < M) :
    (𝓝[<] 1).map ofReal ≤ 𝓝[stolzSet M] 1 := by
  /-
    M : Real
    hM : LT.lt 1 M
    ⊢ LE.le (Filter.map Complex.ofReal (nhdsWithin 1 (Set.Iio 1))) (nhdsWithin 1 ( …
  -/
  rw [← tendsto_id']
  refine tendsto_map' <| tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within ofReal
    (tendsto_nhdsWithin_of_tendsto_nhds <| ofRealCLM.continuous.tendsto' 1 1 rfl) ?_
  /-
    M : Real
    hM : LT.lt 1 M
    ⊢ Filter.Eventually (fun x => Membership.mem (Complex.stolzSet M) ↑x) (nhdsWit …
  -/
  simp only [eventually_iff, norm_eq_abs, abs_ofReal, abs_lt, mem_nhdsWithin]
  /-
    M : Real
    hM : LT.lt 1 M
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u 1) (HasSubset.Subset ( …
  -/
  refine ⟨Set.Ioo 0 2, isOpen_Ioo, by norm_num, fun x hx ↦ ?_⟩
  /-
    M : Real
    hM : LT.lt 1 M
    x : Real
    hx : Membership.mem (Inter.inter (Set.Ioo 0 2) (Set.Iio 1)) x
    ⊢ Membership.mem (setOf fun x => Membership.mem (Complex.stolzSet M) ↑x) x
  -/
  simp only [Set.mem_inter_iff, Set.mem_Ioo, Set.mem_Iio] at hx
  simp only [Set.mem_setOf_eq, stolzSet, ← ofReal_one, ← ofReal_sub, norm_eq_abs, abs_ofReal,
    abs_of_pos hx.1.1, abs_of_pos <| sub_pos.mpr hx.2]
  /-
    M : Real
    hM : LT.lt 1 M
    x : Real
    hx : And (And (LT.lt 0 x) (LT.lt x 2)) (LT.lt x 1)
    ⊢ And (LT.lt x 1) (LT.lt (HSub.hSub 1 x) (HMul.hMul M (HSub.hSub 1 x)))
  -/
  exact ⟨hx.2, lt_mul_left (sub_pos.mpr hx.2) hM⟩
  /-
    🎉 no goals
  -/

-- An ugly technical lemma

private lemma stolzCone_subset_stolzSet_aux' (s : ℝ) :
    ∃ M ε, 0 < M ∧ 0 < ε ∧ ∀ x y, 0 < x → x < ε → |y| < s * x →
      sqrt (x ^ 2 + y ^ 2) < M * (1 - sqrt ((1 - x) ^ 2 + y ^ 2)) := by
  refine ⟨2 * sqrt (1 + s ^ 2) + 1, 1 / (1 + s ^ 2), by positivity, by positivity,
    fun x y hx₀ hx₁ hy ↦ ?_⟩
  have H : sqrt ((1 - x) ^ 2 + y ^ 2) ≤ 1 - x / 2 := by
    calc sqrt ((1 - x) ^ 2 + y ^ 2)
      _ ≤ sqrt ((1 - x) ^ 2 + (s * x) ^ 2) := sqrt_le_sqrt <| by rw [← _root_.sq_abs y]; gcongr
      _ = sqrt (1 - 2 * x + (1 + s ^ 2) * x * x) := by congr 1; ring
      _ ≤ sqrt (1 - 2 * x + (1 + s ^ 2) * (1 / (1 + s ^ 2)) * x) := sqrt_le_sqrt <| by gcongr
      _ = sqrt (1 - x) := by congr 1; field_simp; ring
      _ ≤ 1 - x / 2 := by
        simp_rw [sub_eq_add_neg, ← neg_div]
        refine sqrt_one_add_le <| neg_le_neg_iff.mpr (hx₁.trans_le ?_).le
        rw [div_le_one (by positivity)]
        exact le_add_of_nonneg_right <| sq_nonneg s
  calc sqrt (x ^ 2 + y ^ 2)
    _ ≤ sqrt (x ^ 2 + (s * x) ^ 2) := sqrt_le_sqrt <| by rw [← _root_.sq_abs y]; gcongr
    _ = sqrt ((1 + s ^ 2) * x ^ 2) := by congr; ring
    _ = sqrt (1 + s ^ 2) * x := by rw [sqrt_mul' _ (sq_nonneg x), sqrt_sq hx₀.le]
    _ = 2 * sqrt (1 + s ^ 2) * (x / 2) := by ring
    _ < (2 * sqrt (1 + s ^ 2) + 1) * (x / 2) := by gcongr; exact lt_add_one _
    _ ≤ _ := by gcongr; exact le_sub_comm.mpr H


lemma stolzCone_subset_stolzSet_aux {s : ℝ} (hs : 0 < s) :
    ∃ M ε, 0 < M ∧ 0 < ε ∧ {z : ℂ | 1 - ε < z.re} ∩ stolzCone s ⊆ stolzSet M := by
  /-
    s : Real
    hs : LT.lt 0 s
    ⊢ Exists fun M => Exists fun ε => And (LT.lt 0 M) (And (LT.lt 0 ε) (HasSubset. …
  -/
  peel stolzCone_subset_stolzSet_aux' s with M ε hM hε H
  /-
    case h.h.h.h
    s : Real
    hs : LT.lt 0 s
    M ε : Real
    hM : LT.lt 0 M
    hε : LT.lt 0 ε
    H : ∀ (x y : Real), LT.lt 0 x → LT.lt x ε → LT.lt (_root_.abs y) (HMul.hMul s  …
    ⊢ HasSubset.Subset (Inter.inter (setOf fun z => LT.lt (HSub.hSub 1 ε) z.re) (C …
  -/
  rintro z ⟨hzl, hzr⟩
  /-
    case h.h.h.h.intro
    s : Real
    hs : LT.lt 0 s
    M ε : Real
    hM : LT.lt 0 M
    hε : LT.lt 0 ε
    H : ∀ (x y : Real), LT.lt 0 x → LT.lt x ε → LT.lt (_root_.abs y) (HMul.hMul s  …
    z : Complex
    hzl : Membership.mem (setOf fun z => LT.lt (HSub.hSub 1 ε) z.re) z
    hzr : Membership.mem (Complex.stolzCone s) z
    ⊢ Membership.mem (Complex.stolzSet M) z
  -/
  rw [Set.mem_setOf_eq, sub_lt_comm, ← one_re, ← sub_re] at hzl
  /-
    case h.h.h.h.intro
    s : Real
    hs : LT.lt 0 s
    M ε : Real
    hM : LT.lt 0 M
    hε : LT.lt 0 ε
    H : ∀ (x y : Real), LT.lt 0 x → LT.lt x ε → LT.lt (_root_.abs y) (HMul.hMul s  …
    z : Complex
    hzl : LT.lt (HSub.hSub 1 z).re ε
    hzr : Membership.mem (Complex.stolzCone s) z
    ⊢ Membership.mem (Complex.stolzSet M) z
  -/
  rw [stolzCone, Set.mem_setOf_eq, ← one_re, ← sub_re] at hzr
  replace H :=
    H (1 - z).re z.im ((mul_pos_iff_of_pos_left hs).mp <| (abs_nonneg z.im).trans_lt hzr) hzl hzr
  have h : z.im ^ 2 = (1 - z).im ^ 2 := by
    simp only [sub_im, one_im, zero_sub, even_two, neg_sq]
  rw [h, ← abs_eq_sqrt_sq_add_sq, ← norm_eq_abs, ← h, sub_re, one_re, sub_sub_cancel,
    ← abs_eq_sqrt_sq_add_sq, ← norm_eq_abs] at H
  /-
    case h.h.h.h.intro
    s : Real
    hs : LT.lt 0 s
    M ε : Real
    hM : LT.lt 0 M
    hε : LT.lt 0 ε
    z : Complex
    hzl : LT.lt (HSub.hSub 1 z).re ε
    hzr : LT.lt (_root_.abs z.im) (HMul.hMul s (HSub.hSub 1 z).re)
    H : LT.lt (Norm.norm (HSub.hSub 1 z)) (HMul.hMul M (HSub.hSub 1 (Norm.norm z)))
    h : Eq (HPow.hPow z.im 2) (HPow.hPow (HSub.hSub 1 z).im 2)
    ⊢ Membership.mem (Complex.stolzSet M) z
  -/
  exact ⟨sub_pos.mp <| (mul_pos_iff_of_pos_left hM).mp <| (norm_nonneg _).trans_lt H, H⟩
  /-
    🎉 no goals
  -/


lemma nhdsWithin_stolzCone_le_nhdsWithin_stolzSet {s : ℝ} (hs : 0 < s) :
    ∃ M, 𝓝[stolzCone s] 1 ≤ 𝓝[stolzSet M] 1 := by
  /-
    s : Real
    hs : LT.lt 0 s
    ⊢ Exists fun M => LE.le (nhdsWithin 1 (Complex.stolzCone s)) (nhdsWithin 1 (Co …
  -/
  obtain ⟨M, ε, _, hε, H⟩ := stolzCone_subset_stolzSet_aux hs
  /-
    case intro.intro.intro.intro
    s : Real
    hs : LT.lt 0 s
    M ε : Real
    left✝ : LT.lt 0 M
    hε : LT.lt 0 ε
    H : HasSubset.Subset (Inter.inter (setOf fun z => LT.lt (HSub.hSub 1 ε) z.re)  …
    ⊢ Exists fun M => LE.le (nhdsWithin 1 (Complex.stolzCone s)) (nhdsWithin 1 (Co …
  -/
  use M
  /-
    case h
    s : Real
    hs : LT.lt 0 s
    M ε : Real
    left✝ : LT.lt 0 M
    hε : LT.lt 0 ε
    H : HasSubset.Subset (Inter.inter (setOf fun z => LT.lt (HSub.hSub 1 ε) z.re)  …
    ⊢ LE.le (nhdsWithin 1 (Complex.stolzCone s)) (nhdsWithin 1 (Complex.stolzSet M))
  -/
  rw [nhdsWithin_le_iff, mem_nhdsWithin]
  /-
    case h
    s : Real
    hs : LT.lt 0 s
    M ε : Real
    left✝ : LT.lt 0 M
    hε : LT.lt 0 ε
    H : HasSubset.Subset (Inter.inter (setOf fun z => LT.lt (HSub.hSub 1 ε) z.re)  …
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u 1) (HasSubset.Subset ( …
  -/
  refine ⟨{w | 1 - ε < w.re}, isOpen_lt continuous_const continuous_re, ?_, H⟩
  /-
    case h
    s : Real
    hs : LT.lt 0 s
    M ε : Real
    left✝ : LT.lt 0 M
    hε : LT.lt 0 ε
    H : HasSubset.Subset (Inter.inter (setOf fun z => LT.lt (HSub.hSub 1 ε) z.re)  …
    ⊢ Membership.mem (setOf fun w => LT.lt (HSub.hSub 1 ε) w.re) 1
  -/
  simp only [Set.mem_setOf_eq, one_re, sub_lt_self_iff, hε]
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma for Abel's limit theorem. The difference between the sum `l` at 1 and the
power series's value at a point `z` away from 1 can be rewritten as `1 - z` times a power series
whose coefficients are tail sums of `l`. -/
lemma abel_aux (h : Tendsto (fun n ↦ ∑ i ∈ range n, f i) atTop (𝓝 l)) {z : ℂ} (hz : ‖z‖ < 1) :
    Tendsto (fun n ↦ (1 - z) * ∑ i ∈ range n, (l - ∑ j ∈ range (i + 1), f j) * z ^ i)
      atTop (𝓝 (l - ∑' n, f n * z ^ n)) := by
  /-
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HSub.hSub 1 z) ((Finset.range n).sum fun …
  -/
  let s := fun n ↦ ∑ i ∈ range n, f i
  /-
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HSub.hSub 1 z) ((Finset.range n).sum fun …
  -/
  have k := h.sub (summable_powerSeries_of_norm_lt_one h.cauchySeq hz).hasSum.tendsto_sum_nat
  simp_rw [← sum_sub_distrib, ← mul_one_sub, ← geom_sum_mul_neg, ← mul_assoc, ← sum_mul,
    mul_comm, mul_sum _ _ (f _), range_eq_Ico, ← sum_Ico_Ico_comm', ← range_eq_Ico,
    ← sum_mul] at k
  conv at k =>
    enter [1, n]
    rw [sum_congr (g := fun j ↦ (∑ k ∈ range n, f k - ∑ k ∈ range (j + 1), f k) * z ^ j)
      rfl (fun j hj ↦ by congr 1; exact sum_Ico_eq_sub _ (mem_range.mp hj))]
  suffices Tendsto (fun n ↦ (l - s n) * ∑ i ∈ range n, z ^ i) atTop (𝓝 0) by
    simp_rw [mul_sum] at this
    replace this := (this.const_mul (1 - z)).add k
    conv at this =>
      enter [1, n]
      rw [← mul_add, ← sum_add_distrib]
      enter [2, 2, i]
      rw [← add_mul, sub_add_sub_cancel]
    rwa [mul_zero, zero_add] at this
  /-
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    k : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub 1 z) ((Finset.range n).sum f …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HSub.hSub l (s n)) ((Finset.range n).sum …
  -/
  rw [← zero_mul (-1 / (z - 1))]
  /-
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    k : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub 1 z) ((Finset.range n).sum f …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HSub.hSub l (s n)) ((Finset.range n).sum …
  -/
  apply Tendsto.mul
    /-
      case hf
      f : Nat → Complex
      l : Complex
      h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
      k : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub 1 z) ((Finset.range n).sum f …
      ⊢ Filter.Tendsto (fun x => HSub.hSub l (s x)) Filter.atTop (nhds 0)
    -/
  · simpa only [neg_zero, neg_sub] using (tendsto_sub_nhds_zero_iff.mpr h).neg
    /-
      🎉 no goals
    -/
  · conv =>
      enter [1, n]
      rw [geom_sum_eq (by contrapose! hz; simp [hz]), sub_div, sub_eq_add_neg, ← neg_div]
    /-
      case hg
      f : Nat → Complex
      l : Complex
      h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
      k : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub 1 z) ((Finset.range n).sum f …
      ⊢ Filter.Tendsto (fun n => HAdd.hAdd (HDiv.hDiv (HPow.hPow z n) (HSub.hSub z 1 …
    -/
    rw [← zero_add (-1 / (z - 1)), ← zero_div (z - 1)]
    /-
      case hg
      f : Nat → Complex
      l : Complex
      h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
      k : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub 1 z) ((Finset.range n).sum f …
      ⊢ Filter.Tendsto (fun n => HAdd.hAdd (HDiv.hDiv (HPow.hPow z n) (HSub.hSub z 1 …
    -/
    apply Tendsto.add (Tendsto.div_const (tendsto_pow_atTop_nhds_zero_of_norm_lt_one hz) (z - 1))
    /-
      case hg
      f : Nat → Complex
      l : Complex
      h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
      k : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub 1 z) ((Finset.range n).sum f …
      ⊢ Filter.Tendsto (fun x => HAdd.hAdd (HDiv.hDiv 0 (HSub.hSub z 1)) (HDiv.hDiv  …
    -/
    simp only [zero_div, zero_add, tendsto_const_nhds_iff]
    /-
      🎉 no goals
    -/


/-- **Abel's limit theorem**. Given a power series converging at 1, the corresponding function
is continuous at 1 when approaching 1 within a fixed Stolz set. -/
theorem tendsto_tsum_powerSeries_nhdsWithin_stolzSet
    (h : Tendsto (fun n ↦ ∑ i ∈ range n, f i) atTop (𝓝 l)) {M : ℝ} :
    Tendsto (fun z ↦ ∑' n, f n * z ^ n) (𝓝[stolzSet M] 1) (𝓝 l) := by
  -- If `M ≤ 1` the Stolz set is empty and the statement is trivial
  /-
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    ⊢ Filter.Tendsto (fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)) (nhd …
  -/
  cases' le_or_lt M 1 with hM hM
    /-
      case inl
      f : Nat → Complex
      l : Complex
      h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
      M : Real
      hM : LE.le M 1
      ⊢ Filter.Tendsto (fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)) (nhd …
    -/
  · simp_rw [stolzSet_empty hM, nhdsWithin_empty, tendsto_bot]
    /-
      🎉 no goals
    -/
  -- Abbreviations
  /-
    case inr
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    ⊢ Filter.Tendsto (fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)) (nhd …
  -/
  let s := fun n ↦ ∑ i ∈ range n, f i
  /-
    case inr
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    ⊢ Filter.Tendsto (fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)) (nhd …
  -/
  let g := fun z ↦ ∑' n, f n * z ^ n
  /-
    case inr
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    ⊢ Filter.Tendsto (fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)) (nhd …
  -/
  have hm := Metric.tendsto_atTop.mp h
  /-
    case inr
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ⊢ Filter.Tendsto (fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)) (nhd …
  -/
  rw [Metric.tendsto_nhdsWithin_nhds]
  /-
    case inr
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : Complex⦄,  …
  -/
  simp only [dist_eq_norm] at hm ⊢
  -- Introduce the "challenge" `ε`
  /-
    case inr
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : Complex⦄,  …
  -/
  intro ε εpos
  -- First bound, handles the tail
  /-
    case inr
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ε : Real
    εpos : GT.gt ε 0
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : Complex⦄, Membership.mem (Complex.st …
  -/
  obtain ⟨B₁, hB₁⟩ := hm (ε / 4 / M) (by positivity)
  -- Second bound, handles the head
  /-
    case inr.intro
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ε : Real
    εpos : GT.gt ε 0
    B₁ : Nat
    hB₁ : ∀ (n : Nat), GE.ge n B₁ → LT.lt (Norm.norm (HSub.hSub ((Finset.range n). …
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : Complex⦄, Membership.mem (Complex.st …
  -/
  let F := ∑ i ∈ range B₁, ‖l - s (i + 1)‖
  /-
    case inr.intro
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ε : Real
    εpos : GT.gt ε 0
    B₁ : Nat
    hB₁ : ∀ (n : Nat), GE.ge n B₁ → LT.lt (Norm.norm (HSub.hSub ((Finset.range n). …
    F : Real := (Finset.range B₁).sum fun i => Norm.norm (HSub.hSub l (s (HAdd.hAd …
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : Complex⦄, Membership.mem (Complex.st …
  -/
  use ε / 4 / (F + 1), by positivity
  /-
    case right
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ε : Real
    εpos : GT.gt ε 0
    B₁ : Nat
    hB₁ : ∀ (n : Nat), GE.ge n B₁ → LT.lt (Norm.norm (HSub.hSub ((Finset.range n). …
    F : Real := (Finset.range B₁).sum fun i => Norm.norm (HSub.hSub l (s (HAdd.hAd …
    ⊢ ∀ ⦃x : Complex⦄, Membership.mem (Complex.stolzSet M) x → LT.lt (Norm.norm (H …
  -/
  intro z ⟨zn, zm⟩ zd
  /-
    case right
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ε : Real
    εpos : GT.gt ε 0
    B₁ : Nat
    hB₁ : ∀ (n : Nat), GE.ge n B₁ → LT.lt (Norm.norm (HSub.hSub ((Finset.range n). …
    F : Real := (Finset.range B₁).sum fun i => Norm.norm (HSub.hSub l (s (HAdd.hAd …
    z : Complex
    zn : LT.lt (Norm.norm z) 1
    zm : LT.lt (Norm.norm (HSub.hSub 1 z)) (HMul.hMul M (HSub.hSub 1 (Norm.norm z)))
    zd : LT.lt (Norm.norm (HSub.hSub z 1)) (HDiv.hDiv (HDiv.hDiv ε 4) (HAdd.hAdd F …
    ⊢ LT.lt (Norm.norm (HSub.hSub (tsum fun n => HMul.hMul (f n) (HPow.hPow z n))  …
  -/
  have p := abel_aux h zn
  /-
    case right
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ε : Real
    εpos : GT.gt ε 0
    B₁ : Nat
    hB₁ : ∀ (n : Nat), GE.ge n B₁ → LT.lt (Norm.norm (HSub.hSub ((Finset.range n). …
    F : Real := (Finset.range B₁).sum fun i => Norm.norm (HSub.hSub l (s (HAdd.hAd …
    z : Complex
    zn : LT.lt (Norm.norm z) 1
    zm : LT.lt (Norm.norm (HSub.hSub 1 z)) (HMul.hMul M (HSub.hSub 1 (Norm.norm z)))
    zd : LT.lt (Norm.norm (HSub.hSub z 1)) (HDiv.hDiv (HDiv.hDiv ε 4) (HAdd.hAdd F …
    p : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub 1 z) ((Finset.range n).sum f …
    ⊢ LT.lt (Norm.norm (HSub.hSub (tsum fun n => HMul.hMul (f n) (HPow.hPow z n))  …
  -/
  simp_rw [Metric.tendsto_atTop, dist_eq_norm, norm_sub_rev] at p
  -- Third bound, regarding the distance between `l - g z` and the rearranged sum
  /-
    case right
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ε : Real
    εpos : GT.gt ε 0
    B₁ : Nat
    hB₁ : ∀ (n : Nat), GE.ge n B₁ → LT.lt (Norm.norm (HSub.hSub ((Finset.range n). …
    F : Real := (Finset.range B₁).sum fun i => Norm.norm (HSub.hSub l (s (HAdd.hAd …
    z : Complex
    zn : LT.lt (Norm.norm z) 1
    zm : LT.lt (Norm.norm (HSub.hSub 1 z)) (HMul.hMul M (HSub.hSub 1 (Norm.norm z)))
    zd : LT.lt (Norm.norm (HSub.hSub z 1)) (HDiv.hDiv (HDiv.hDiv ε 4) (HAdd.hAdd F …
    p : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt ( …
    ⊢ LT.lt (Norm.norm (HSub.hSub (tsum fun n => HMul.hMul (f n) (HPow.hPow z n))  …
  -/
  obtain ⟨B₂, hB₂⟩ := p (ε / 2) (by positivity)
  /-
    case right.intro
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    hm : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt  …
    ε : Real
    εpos : GT.gt ε 0
    B₁ : Nat
    hB₁ : ∀ (n : Nat), GE.ge n B₁ → LT.lt (Norm.norm (HSub.hSub ((Finset.range n). …
    F : Real := (Finset.range B₁).sum fun i => Norm.norm (HSub.hSub l (s (HAdd.hAd …
    z : Complex
    zn : LT.lt (Norm.norm z) 1
    zm : LT.lt (Norm.norm (HSub.hSub 1 z)) (HMul.hMul M (HSub.hSub 1 (Norm.norm z)))
    zd : LT.lt (Norm.norm (HSub.hSub z 1)) (HDiv.hDiv (HDiv.hDiv ε 4) (HAdd.hAdd F …
    p : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt ( …
    B₂ : Nat
    hB₂ : ∀ (n : Nat), GE.ge n B₂ → LT.lt (Norm.norm (HSub.hSub (HSub.hSub l (tsum …
    ⊢ LT.lt (Norm.norm (HSub.hSub (tsum fun n => HMul.hMul (f n) (HPow.hPow z n))  …
  -/
  clear hm p
  /-
    case right.intro
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    ε : Real
    εpos : GT.gt ε 0
    B₁ : Nat
    hB₁ : ∀ (n : Nat), GE.ge n B₁ → LT.lt (Norm.norm (HSub.hSub ((Finset.range n). …
    F : Real := (Finset.range B₁).sum fun i => Norm.norm (HSub.hSub l (s (HAdd.hAd …
    z : Complex
    zn : LT.lt (Norm.norm z) 1
    zm : LT.lt (Norm.norm (HSub.hSub 1 z)) (HMul.hMul M (HSub.hSub 1 (Norm.norm z)))
    zd : LT.lt (Norm.norm (HSub.hSub z 1)) (HDiv.hDiv (HDiv.hDiv ε 4) (HAdd.hAdd F …
    B₂ : Nat
    hB₂ : ∀ (n : Nat), GE.ge n B₂ → LT.lt (Norm.norm (HSub.hSub (HSub.hSub l (tsum …
    ⊢ LT.lt (Norm.norm (HSub.hSub (tsum fun n => HMul.hMul (f n) (HPow.hPow z n))  …
  -/
  replace hB₂ := hB₂ (max B₁ B₂) (by simp)
  suffices ‖(1 - z) * ∑ i ∈ range (max B₁ B₂), (l - s (i + 1)) * z ^ i‖ < ε / 2 by
    calc
      _ = ‖l - g z‖ := by rw [norm_sub_rev]
      _ = ‖l - g z - (1 - z) * ∑ i ∈ range (max B₁ B₂), (l - s (i + 1)) * z ^ i +
          (1 - z) * ∑ i ∈ range (max B₁ B₂), (l - s (i + 1)) * z ^ i‖ := by rw [sub_add_cancel _]
      _ ≤ ‖l - g z - (1 - z) * ∑ i ∈ range (max B₁ B₂), (l - s (i + 1)) * z ^ i‖ +
          ‖(1 - z) * ∑ i ∈ range (max B₁ B₂), (l - s (i + 1)) * z ^ i‖ := norm_add_le _ _
      _ < ε / 2 + ε / 2 := add_lt_add hB₂ this
      _ = _ := add_halves ε
  -- We break the rearranged sum along `B₁`
  calc
    _ = ‖(1 - z) * ∑ i ∈ range B₁, (l - s (i + 1)) * z ^ i +
        (1 - z) * ∑ i ∈ Ico B₁ (max B₁ B₂), (l - s (i + 1)) * z ^ i‖ := by
      rw [← mul_add, sum_range_add_sum_Ico _ (le_max_left B₁ B₂)]
    _ ≤ ‖(1 - z) * ∑ i ∈ range B₁, (l - s (i + 1)) * z ^ i‖ +
        ‖(1 - z) * ∑ i ∈ Ico B₁ (max B₁ B₂), (l - s (i + 1)) * z ^ i‖ := norm_add_le _ _
    _ = ‖1 - z‖ * ‖∑ i ∈ range B₁, (l - s (i + 1)) * z ^ i‖ +
        ‖1 - z‖ * ‖∑ i ∈ Ico B₁ (max B₁ B₂), (l - s (i + 1)) * z ^ i‖ := by
      rw [norm_mul, norm_mul]
    _ ≤ ‖1 - z‖ * ∑ i ∈ range B₁, ‖l - s (i + 1)‖ * ‖z‖ ^ i +
        ‖1 - z‖ * ∑ i ∈ Ico B₁ (max B₁ B₂), ‖l - s (i + 1)‖ * ‖z‖ ^ i := by
      gcongr <;> simp_rw [← norm_pow, ← norm_mul, norm_sum_le]
  -- then prove that the two pieces are each less than `ε / 4`
  have S₁ : ‖1 - z‖ * ∑ i ∈ range B₁, ‖l - s (i + 1)‖ * ‖z‖ ^ i < ε / 4 :=
    calc
      _ ≤ ‖1 - z‖ * ∑ i ∈ range B₁, ‖l - s (i + 1)‖ := by
        gcongr; nth_rw 3 [← mul_one ‖_‖]
        gcongr; exact pow_le_one₀ (norm_nonneg _) zn.le
      _ ≤ ‖1 - z‖ * (F + 1) := by gcongr; linarith only
      _ < _ := by rwa [norm_sub_rev, lt_div_iff₀ (by positivity)] at zd
  have S₂ : ‖1 - z‖ * ∑ i ∈ Ico B₁ (max B₁ B₂), ‖l - s (i + 1)‖ * ‖z‖ ^ i < ε / 4 :=
    calc
      _ ≤ ‖1 - z‖ * ∑ i ∈ Ico B₁ (max B₁ B₂), ε / 4 / M * ‖z‖ ^ i := by
        gcongr with i hi
        have := hB₁ (i + 1) (by linarith only [(mem_Ico.mp hi).1])
        rw [norm_sub_rev] at this
        exact this.le
      _ = ‖1 - z‖ * (ε / 4 / M) * ∑ i ∈ Ico B₁ (max B₁ B₂), ‖z‖ ^ i := by
        rw [← mul_sum, ← mul_assoc]
      _ ≤ ‖1 - z‖ * (ε / 4 / M) * ∑' i, ‖z‖ ^ i := by
        gcongr
        exact sum_le_tsum _ (fun _ _ ↦ by positivity)
          (summable_geometric_of_lt_one (by positivity) zn)
      _ = ‖1 - z‖ * (ε / 4 / M) / (1 - ‖z‖) := by
        rw [tsum_geometric_of_lt_one (by positivity) zn, ← div_eq_mul_inv]
      _ < M * (1 - ‖z‖) * (ε / 4 / M) / (1 - ‖z‖) := by gcongr; linarith only [zn]
      _ = _ := by
        rw [← mul_rotate, mul_div_cancel_right₀ _ (by linarith only [zn]),
          div_mul_cancel₀ _ (by linarith only [hM])]
  /-
    case right.intro.calc.step
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    ε : Real
    εpos : GT.gt ε 0
    B₁ : Nat
    hB₁ : ∀ (n : Nat), GE.ge n B₁ → LT.lt (Norm.norm (HSub.hSub ((Finset.range n). …
    F : Real := (Finset.range B₁).sum fun i => Norm.norm (HSub.hSub l (s (HAdd.hAd …
    z : Complex
    zn : LT.lt (Norm.norm z) 1
    zm : LT.lt (Norm.norm (HSub.hSub 1 z)) (HMul.hMul M (HSub.hSub 1 (Norm.norm z)))
    zd : LT.lt (Norm.norm (HSub.hSub z 1)) (HDiv.hDiv (HDiv.hDiv ε 4) (HAdd.hAdd F …
    B₂ : Nat
    hB₂ : LT.lt (Norm.norm (HSub.hSub (HSub.hSub l (tsum fun n => HMul.hMul (f n)  …
    S₁ : LT.lt (HMul.hMul (Norm.norm (HSub.hSub 1 z)) ((Finset.range B₁).sum fun i …
    S₂ : LT.lt (HMul.hMul (Norm.norm (HSub.hSub 1 z)) ((Finset.Ico B₁ (Max.max B₁  …
    ⊢ LT.lt (HAdd.hAdd (HMul.hMul (Norm.norm (HSub.hSub 1 z)) ((Finset.range B₁).s …
  -/
  convert add_lt_add S₁ S₂ using 1
  /-
    case h.e'_4
    f : Nat → Complex
    l : Complex
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    M : Real
    hM : LT.lt 1 M
    s : Nat → Complex := fun n => (Finset.range n).sum fun i => f i
    g : Complex → Complex := fun z => tsum fun n => HMul.hMul (f n) (HPow.hPow z n)
    ε : Real
    εpos : GT.gt ε 0
    B₁ : Nat
    hB₁ : ∀ (n : Nat), GE.ge n B₁ → LT.lt (Norm.norm (HSub.hSub ((Finset.range n). …
    F : Real := (Finset.range B₁).sum fun i => Norm.norm (HSub.hSub l (s (HAdd.hAd …
    z : Complex
    zn : LT.lt (Norm.norm z) 1
    zm : LT.lt (Norm.norm (HSub.hSub 1 z)) (HMul.hMul M (HSub.hSub 1 (Norm.norm z)))
    zd : LT.lt (Norm.norm (HSub.hSub z 1)) (HDiv.hDiv (HDiv.hDiv ε 4) (HAdd.hAdd F …
    B₂ : Nat
    hB₂ : LT.lt (Norm.norm (HSub.hSub (HSub.hSub l (tsum fun n => HMul.hMul (f n)  …
    S₁ : LT.lt (HMul.hMul (Norm.norm (HSub.hSub 1 z)) ((Finset.range B₁).sum fun i …
    S₂ : LT.lt (HMul.hMul (Norm.norm (HSub.hSub 1 z)) ((Finset.Ico B₁ (Max.max B₁  …
    ⊢ Eq (HDiv.hDiv ε 2) (HAdd.hAdd (HDiv.hDiv ε 4) (HDiv.hDiv ε 4))
  -/
  linarith only
  /-
    🎉 no goals
  -/


/-- **Abel's limit theorem**. Given a power series converging at 1, the corresponding function
is continuous at 1 when approaching 1 within any fixed Stolz cone. -/
theorem tendsto_tsum_powerSeries_nhdsWithin_stolzCone
    (h : Tendsto (fun n ↦ ∑ i ∈ range n, f i) atTop (𝓝 l)) {s : ℝ} (hs : 0 < s) :
    Tendsto (fun z ↦ ∑' n, f n * z ^ n) (𝓝[stolzCone s] 1) (𝓝 l) :=
  (tendsto_tsum_powerSeries_nhdsWithin_stolzSet h).mono_left
    (nhdsWithin_stolzCone_le_nhdsWithin_stolzSet hs).choose_spec


theorem tendsto_tsum_powerSeries_nhdsWithin_lt
    (h : Tendsto (fun n ↦ ∑ i ∈ range n, f i) atTop (𝓝 l)) :
    Tendsto (fun z ↦ ∑' n, f n * z ^ n) ((𝓝[<] 1).map ofReal) (𝓝 l) :=
  (tendsto_tsum_powerSeries_nhdsWithin_stolzSet (M := 2) h).mono_left
    (nhdsWithin_lt_le_nhdsWithin_stolzSet one_lt_two)


/-- **Abel's limit theorem**. Given a real power series converging at 1, the corresponding function
is continuous at 1 when approaching 1 from the left. -/
theorem tendsto_tsum_powerSeries_nhdsWithin_lt
    (h : Tendsto (fun n ↦ ∑ i ∈ range n, f i) atTop (𝓝 l)) :
    Tendsto (fun x ↦ ∑' n, f n * x ^ n) (𝓝[<] 1) (𝓝 l) := by
  /-
    f : Nat → Real
    l : Real
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    ⊢ Filter.Tendsto (fun x => tsum fun n => HMul.hMul (f n) (HPow.hPow x n)) (nhd …
  -/
  have m : (𝓝 l).map ofReal ≤ 𝓝 ↑l := ofRealCLM.continuous.tendsto l
  /-
    f : Nat → Real
    l : Real
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    m : LE.le (Filter.map Complex.ofReal (nhds l)) (nhds ↑l)
    ⊢ Filter.Tendsto (fun x => tsum fun n => HMul.hMul (f n) (HPow.hPow x n)) (nhd …
  -/
  replace h := (tendsto_map.comp h).mono_right m
  /-
    f : Nat → Real
    l : Real
    m : LE.le (Filter.map Complex.ofReal (nhds l)) (nhds ↑l)
    h : Filter.Tendsto (Function.comp Complex.ofReal fun n => (Finset.range n).sum …
    ⊢ Filter.Tendsto (fun x => tsum fun n => HMul.hMul (f n) (HPow.hPow x n)) (nhd …
  -/
  rw [Function.comp_def] at h
  /-
    f : Nat → Real
    l : Real
    m : LE.le (Filter.map Complex.ofReal (nhds l)) (nhds ↑l)
    h : Filter.Tendsto (fun x => ↑((Finset.range x).sum fun i => f i)) Filter.atTo …
    ⊢ Filter.Tendsto (fun x => tsum fun n => HMul.hMul (f n) (HPow.hPow x n)) (nhd …
  -/
  push_cast at h
  /-
    f : Nat → Real
    l : Real
    m : LE.le (Filter.map Complex.ofReal (nhds l)) (nhds ↑l)
    h : Filter.Tendsto (fun x => (Finset.range x).sum fun i => ↑(f i)) Filter.atTo …
    ⊢ Filter.Tendsto (fun x => tsum fun n => HMul.hMul (f n) (HPow.hPow x n)) (nhd …
  -/
  replace h := Complex.tendsto_tsum_powerSeries_nhdsWithin_lt h
  /-
    f : Nat → Real
    l : Real
    m : LE.le (Filter.map Complex.ofReal (nhds l)) (nhds ↑l)
    h : Filter.Tendsto (fun z => tsum fun n => HMul.hMul (↑(f n)) (HPow.hPow z n)) …
    ⊢ Filter.Tendsto (fun x => tsum fun n => HMul.hMul (f n) (HPow.hPow x n)) (nhd …
  -/
  rw [tendsto_map'_iff] at h
  /-
    f : Nat → Real
    l : Real
    m : LE.le (Filter.map Complex.ofReal (nhds l)) (nhds ↑l)
    h : Filter.Tendsto (Function.comp (fun z => tsum fun n => HMul.hMul (↑(f n)) ( …
    ⊢ Filter.Tendsto (fun x => tsum fun n => HMul.hMul (f n) (HPow.hPow x n)) (nhd …
  -/
  rw [Metric.tendsto_nhdsWithin_nhds] at h ⊢
  /-
    f : Nat → Real
    l : Real
    m : LE.le (Filter.map Complex.ofReal (nhds l)) (nhds ↑l)
    h : ∀ (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : Real⦄, M …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : Real⦄, Mem …
  -/
  convert h
  /-
    case h.h'.h.e'_2.h.h.e'_2.h.h'.h'.h.e'_3
    f : Nat → Real
    l : Real
    m : LE.le (Filter.map Complex.ofReal (nhds l)) (nhds ↑l)
    h : ∀ (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : Real⦄, M …
    a✝⁴ : Real
    a✝³ : GT.gt a✝⁴ 0
    x✝ a✝² : Real
    a✝¹ : Membership.mem (Set.Iio 1) a✝²
    a✝ : LT.lt (Dist.dist a✝² 1) x✝
    ⊢ Eq (Dist.dist (tsum fun n => HMul.hMul (f n) (HPow.hPow a✝² n)) l) (Dist.dis …
  -/
  simp_rw [Function.comp_apply, dist_eq_norm]
  /-
    case h.h'.h.e'_2.h.h.e'_2.h.h'.h'.h.e'_3
    f : Nat → Real
    l : Real
    m : LE.le (Filter.map Complex.ofReal (nhds l)) (nhds ↑l)
    h : ∀ (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : Real⦄, M …
    a✝⁴ : Real
    a✝³ : GT.gt a✝⁴ 0
    x✝ a✝² : Real
    a✝¹ : Membership.mem (Set.Iio 1) a✝²
    a✝ : LT.lt (Dist.dist a✝² 1) x✝
    ⊢ Eq (Norm.norm (HSub.hSub (tsum fun n => HMul.hMul (f n) (HPow.hPow a✝² n)) l …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


