theorem AntitoneOn.integral_le_sum (hf : AntitoneOn f (Icc x₀ (x₀ + a))) :
    (∫ x in x₀..x₀ + a, f x) ≤ ∑ i ∈ Finset.range a, f (x₀ + i) := by
  have hint : ∀ k : ℕ, k < a → IntervalIntegrable f volume (x₀ + k) (x₀ + (k + 1 : ℕ)) := by
    intro k hk
    refine (hf.mono ?_).intervalIntegrable
    rw [uIcc_of_le]
    · apply Icc_subset_Icc
      · simp only [le_add_iff_nonneg_right, Nat.cast_nonneg]
      · simp only [add_le_add_iff_left, Nat.cast_le, Nat.succ_le_of_lt hk]
    · simp only [add_le_add_iff_left, Nat.cast_le, Nat.le_succ]
  calc
    ∫ x in x₀..x₀ + a, f x = ∑ i ∈ Finset.range a, ∫ x in x₀ + i..x₀ + (i + 1 : ℕ), f x := by
      convert (intervalIntegral.sum_integral_adjacent_intervals hint).symm
      simp only [Nat.cast_zero, add_zero]
    _ ≤ ∑ i ∈ Finset.range a, ∫ _ in x₀ + i..x₀ + (i + 1 : ℕ), f (x₀ + i) := by
      apply Finset.sum_le_sum fun i hi => ?_
      have ia : i < a := Finset.mem_range.1 hi
      refine intervalIntegral.integral_mono_on (by simp) (hint _ ia) (by simp) fun x hx => ?_
      apply hf _ _ hx.1
      · simp only [ia.le, mem_Icc, le_add_iff_nonneg_right, Nat.cast_nonneg, add_le_add_iff_left,
          Nat.cast_le, and_self_iff]
      · refine mem_Icc.2 ⟨le_trans (by simp) hx.1, le_trans hx.2 ?_⟩
        simp only [add_le_add_iff_left, Nat.cast_le, Nat.succ_le_of_lt ia]
    _ = ∑ i ∈ Finset.range a, f (x₀ + i) := by simp


theorem AntitoneOn.integral_le_sum_Ico (hab : a ≤ b) (hf : AntitoneOn f (Set.Icc a b)) :
    (∫ x in a..b, f x) ≤ ∑ x ∈ Finset.Ico a b, f x := by
  /-
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : AntitoneOn f (Set.Icc ↑a ↑b)
    ⊢ LE.le (intervalIntegral (fun x => f x) (↑a) (↑b) MeasureTheory.MeasureSpace. …
  -/
  rw [(Nat.sub_add_cancel hab).symm, Nat.cast_add]
  conv =>
    congr
    congr
    · skip
    · skip
    rw [add_comm]
    · skip
    · skip
    congr
    congr
    rw [← zero_add a]
  /-
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : AntitoneOn f (Set.Icc ↑a ↑b)
    ⊢ LE.le (intervalIntegral (fun x => f x) (↑a) (HAdd.hAdd ↑a ↑(HSub.hSub b a))  …
  -/
  rw [← Finset.sum_Ico_add, Nat.Ico_zero_eq_range]
  conv =>
    rhs
    congr
    · skip
    ext
    rw [Nat.cast_add]
  /-
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : AntitoneOn f (Set.Icc ↑a ↑b)
    ⊢ LE.le (intervalIntegral (fun x => f x) (↑a) (HAdd.hAdd ↑a ↑(HSub.hSub b a))  …
  -/
  apply AntitoneOn.integral_le_sum
  /-
    case hf
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : AntitoneOn f (Set.Icc ↑a ↑b)
    ⊢ AntitoneOn f (Set.Icc (↑a) (HAdd.hAdd ↑a ↑(HSub.hSub b a)))
  -/
  simp only [hf, hab, Nat.cast_sub, add_sub_cancel]
  /-
    🎉 no goals
  -/


theorem AntitoneOn.sum_le_integral (hf : AntitoneOn f (Icc x₀ (x₀ + a))) :
    (∑ i ∈ Finset.range a, f (x₀ + (i + 1 : ℕ))) ≤ ∫ x in x₀..x₀ + a, f x := by
  have hint : ∀ k : ℕ, k < a → IntervalIntegrable f volume (x₀ + k) (x₀ + (k + 1 : ℕ)) := by
    intro k hk
    refine (hf.mono ?_).intervalIntegrable
    rw [uIcc_of_le]
    · apply Icc_subset_Icc
      · simp only [le_add_iff_nonneg_right, Nat.cast_nonneg]
      · simp only [add_le_add_iff_left, Nat.cast_le, Nat.succ_le_of_lt hk]
    · simp only [add_le_add_iff_left, Nat.cast_le, Nat.le_succ]
  calc
    (∑ i ∈ Finset.range a, f (x₀ + (i + 1 : ℕ))) =
        ∑ i ∈ Finset.range a, ∫ _ in x₀ + i..x₀ + (i + 1 : ℕ), f (x₀ + (i + 1 : ℕ)) := by simp
    _ ≤ ∑ i ∈ Finset.range a, ∫ x in x₀ + i..x₀ + (i + 1 : ℕ), f x := by
      apply Finset.sum_le_sum fun i hi => ?_
      have ia : i + 1 ≤ a := Finset.mem_range.1 hi
      refine intervalIntegral.integral_mono_on (by simp) (by simp) (hint _ ia) fun x hx => ?_
      apply hf _ _ hx.2
      · refine mem_Icc.2 ⟨le_trans (le_add_of_nonneg_right (Nat.cast_nonneg _)) hx.1,
          le_trans hx.2 ?_⟩
        simp only [Nat.cast_le, add_le_add_iff_left, ia]
      · refine mem_Icc.2 ⟨le_add_of_nonneg_right (Nat.cast_nonneg _), ?_⟩
        simp only [add_le_add_iff_left, Nat.cast_le, ia]
    _ = ∫ x in x₀..x₀ + a, f x := by
      convert intervalIntegral.sum_integral_adjacent_intervals hint
      simp only [Nat.cast_zero, add_zero]


theorem AntitoneOn.sum_le_integral_Ico (hab : a ≤ b) (hf : AntitoneOn f (Set.Icc a b)) :
    (∑ i ∈ Finset.Ico a b, f (i + 1 : ℕ)) ≤ ∫ x in a..b, f x := by
  /-
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : AntitoneOn f (Set.Icc ↑a ↑b)
    ⊢ LE.le ((Finset.Ico a b).sum fun i => f ↑(HAdd.hAdd i 1)) (intervalIntegral ( …
  -/
  rw [(Nat.sub_add_cancel hab).symm, Nat.cast_add]
  conv =>
    congr
    congr
    congr
    rw [← zero_add a]
    · skip
    · skip
    · skip
    rw [add_comm]
  /-
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : AntitoneOn f (Set.Icc ↑a ↑b)
    ⊢ LE.le ((Finset.Ico (HAdd.hAdd 0 a) (HAdd.hAdd (HSub.hSub b a) a)).sum fun i  …
  -/
  rw [← Finset.sum_Ico_add, Nat.Ico_zero_eq_range]
  conv =>
    lhs
    congr
    congr
    · skip
    ext
    rw [add_assoc, Nat.cast_add]
  /-
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : AntitoneOn f (Set.Icc ↑a ↑b)
    ⊢ LE.le ((Finset.range (HSub.hSub b a)).sum fun x => f (HAdd.hAdd ↑a ↑(HAdd.hA …
  -/
  apply AntitoneOn.sum_le_integral
  /-
    case hf
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : AntitoneOn f (Set.Icc ↑a ↑b)
    ⊢ AntitoneOn f (Set.Icc (↑a) (HAdd.hAdd ↑a ↑(HSub.hSub b a)))
  -/
  simp only [hf, hab, Nat.cast_sub, add_sub_cancel]
  /-
    🎉 no goals
  -/


theorem MonotoneOn.sum_le_integral (hf : MonotoneOn f (Icc x₀ (x₀ + a))) :
    (∑ i ∈ Finset.range a, f (x₀ + i)) ≤ ∫ x in x₀..x₀ + a, f x := by
  /-
    x₀ : Real
    a : Nat
    f : Real → Real
    hf : MonotoneOn f (Set.Icc x₀ (HAdd.hAdd x₀ ↑a))
    ⊢ LE.le ((Finset.range a).sum fun i => f (HAdd.hAdd x₀ ↑i)) (intervalIntegral  …
  -/
  rw [← neg_le_neg_iff, ← Finset.sum_neg_distrib, ← intervalIntegral.integral_neg]
  /-
    x₀ : Real
    a : Nat
    f : Real → Real
    hf : MonotoneOn f (Set.Icc x₀ (HAdd.hAdd x₀ ↑a))
    ⊢ LE.le (intervalIntegral (fun x => Neg.neg (f x)) x₀ (HAdd.hAdd x₀ ↑a) Measur …
  -/
  exact hf.neg.integral_le_sum
  /-
    🎉 no goals
  -/


theorem MonotoneOn.sum_le_integral_Ico (hab : a ≤ b) (hf : MonotoneOn f (Set.Icc a b)) :
    ∑ x ∈ Finset.Ico a b, f x ≤ ∫ x in a..b, f x := by
  /-
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : MonotoneOn f (Set.Icc ↑a ↑b)
    ⊢ LE.le ((Finset.Ico a b).sum fun x => f ↑x) (intervalIntegral (fun x => f x)  …
  -/
  rw [← neg_le_neg_iff, ← Finset.sum_neg_distrib, ← intervalIntegral.integral_neg]
  /-
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : MonotoneOn f (Set.Icc ↑a ↑b)
    ⊢ LE.le (intervalIntegral (fun x => Neg.neg (f x)) (↑a) (↑b) MeasureTheory.Mea …
  -/
  exact hf.neg.integral_le_sum_Ico hab
  /-
    🎉 no goals
  -/


theorem MonotoneOn.integral_le_sum (hf : MonotoneOn f (Icc x₀ (x₀ + a))) :
    (∫ x in x₀..x₀ + a, f x) ≤ ∑ i ∈ Finset.range a, f (x₀ + (i + 1 : ℕ)) := by
  /-
    x₀ : Real
    a : Nat
    f : Real → Real
    hf : MonotoneOn f (Set.Icc x₀ (HAdd.hAdd x₀ ↑a))
    ⊢ LE.le (intervalIntegral (fun x => f x) x₀ (HAdd.hAdd x₀ ↑a) MeasureTheory.Me …
  -/
  rw [← neg_le_neg_iff, ← Finset.sum_neg_distrib, ← intervalIntegral.integral_neg]
  /-
    x₀ : Real
    a : Nat
    f : Real → Real
    hf : MonotoneOn f (Set.Icc x₀ (HAdd.hAdd x₀ ↑a))
    ⊢ LE.le ((Finset.range a).sum fun x => Neg.neg (f (HAdd.hAdd x₀ ↑(HAdd.hAdd x  …
  -/
  exact hf.neg.sum_le_integral
  /-
    🎉 no goals
  -/


theorem MonotoneOn.integral_le_sum_Ico (hab : a ≤ b) (hf : MonotoneOn f (Set.Icc a b)) :
    (∫ x in a..b, f x) ≤ ∑ i ∈ Finset.Ico a b, f (i + 1 : ℕ) := by
  /-
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : MonotoneOn f (Set.Icc ↑a ↑b)
    ⊢ LE.le (intervalIntegral (fun x => f x) (↑a) (↑b) MeasureTheory.MeasureSpace. …
  -/
  rw [← neg_le_neg_iff, ← Finset.sum_neg_distrib, ← intervalIntegral.integral_neg]
  /-
    a b : Nat
    f : Real → Real
    hab : LE.le a b
    hf : MonotoneOn f (Set.Icc ↑a ↑b)
    ⊢ LE.le ((Finset.Ico a b).sum fun x => Neg.neg (f ↑(HAdd.hAdd x 1))) (interval …
  -/
  exact hf.neg.sum_le_integral_Ico hab
  /-
    🎉 no goals
  -/

