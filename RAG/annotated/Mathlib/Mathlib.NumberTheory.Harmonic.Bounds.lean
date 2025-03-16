theorem log_add_one_le_harmonic (n : ℕ) :
    Real.log ↑(n+1) ≤ harmonic n := by
  calc _ = ∫ x in (1 : ℕ)..↑(n+1), x⁻¹ := ?_
       _ ≤ ∑ d ∈ Finset.Icc 1 n, (d : ℝ)⁻¹ := ?_
       _ = harmonic n := ?_
    /-
      case calc_1
      n : Nat
      ⊢ Eq (Real.log ↑(HAdd.hAdd n 1)) (intervalIntegral (fun x => Inv.inv x) (↑1) ( …
    -/
  · rw [Nat.cast_one, integral_inv (by simp [(show ¬ (1 : ℝ) ≤ 0 by norm_num)]), div_one]
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      n : Nat
      ⊢ LE.le (intervalIntegral (fun x => Inv.inv x) (↑1) (↑(HAdd.hAdd n 1)) Measure …
    -/
  · exact (inv_antitoneOn_Icc_right <| by norm_num).integral_le_sum_Ico (Nat.le_add_left 1 n)
    /-
      🎉 no goals
    -/
    /-
      case calc_3
      n : Nat
      ⊢ Eq ((Finset.Icc 1 n).sum fun d => Inv.inv ↑d) ↑(harmonic n)
    -/
  · simp only [harmonic_eq_sum_Icc, Rat.cast_sum, Rat.cast_inv, Rat.cast_natCast]
    /-
      🎉 no goals
    -/


theorem harmonic_le_one_add_log (n : ℕ) :
    harmonic n ≤ 1 + Real.log n := by
  /-
    n : Nat
    ⊢ LE.le (↑(harmonic n)) (HAdd.hAdd 1 (Real.log ↑n))
  -/
  by_cases hn0 : n = 0
    /-
      case pos
      n : Nat
      hn0 : Eq n 0
      ⊢ LE.le (↑(harmonic n)) (HAdd.hAdd 1 (Real.log ↑n))
    -/
  · simp [hn0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    hn0 : Not (Eq n 0)
    ⊢ LE.le (↑(harmonic n)) (HAdd.hAdd 1 (Real.log ↑n))
  -/
  have hn : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr hn0
  /-
    case neg
    n : Nat
    hn0 : Not (Eq n 0)
    hn : LE.le 1 n
    ⊢ LE.le (↑(harmonic n)) (HAdd.hAdd 1 (Real.log ↑n))
  -/
  simp_rw [harmonic_eq_sum_Icc, Rat.cast_sum, Rat.cast_inv, Rat.cast_natCast]
  rw [← Finset.sum_erase_add (Finset.Icc 1 n) _ (Finset.left_mem_Icc.mpr hn), add_comm,
    Nat.cast_one, inv_one]
  /-
    case neg
    n : Nat
    hn0 : Not (Eq n 0)
    hn : LE.le 1 n
    ⊢ LE.le (HAdd.hAdd 1 (((Finset.Icc 1 n).erase 1).sum fun x => Inv.inv ↑x)) (HA …
  -/
  refine add_le_add_left ?_ 1
  /-
    case neg
    n : Nat
    hn0 : Not (Eq n 0)
    hn : LE.le 1 n
    ⊢ LE.le (((Finset.Icc 1 n).erase 1).sum fun x => Inv.inv ↑x) (Real.log ↑n)
  -/
  simp only [Nat.lt_one_iff, Finset.mem_Icc, Finset.Icc_erase_left]
  calc ∑ d ∈ .Ico 2 (n + 1), (d : ℝ)⁻¹
    _ = ∑ d ∈ .Ico 2 (n + 1), (↑(d + 1) - 1)⁻¹ := ?_
    _ ≤ ∫ x in (2).. ↑(n + 1), (x - 1)⁻¹  := ?_
    _ = ∫ x in (1)..n, x⁻¹ := ?_
    _ = Real.log ↑n := ?_
    /-
      case neg.calc_1
      n : Nat
      hn0 : Not (Eq n 0)
      hn : LE.le 1 n
      ⊢ Eq ((Finset.Ico 2 (HAdd.hAdd n 1)).sum fun d => Inv.inv ↑d) ((Finset.Ico 2 ( …
    -/
  · simp_rw [Nat.cast_add, Nat.cast_one, add_sub_cancel_right]
    /-
      🎉 no goals
    -/
  · exact @AntitoneOn.sum_le_integral_Ico 2 (n + 1) (fun x : ℝ ↦ (x - 1)⁻¹) (by linarith [hn]) <|
      sub_inv_antitoneOn_Icc_right (by norm_num)
    /-
      case neg.calc_3
      n : Nat
      hn0 : Not (Eq n 0)
      hn : LE.le 1 n
      ⊢ Eq (intervalIntegral (fun x => Inv.inv (HSub.hSub x 1)) 2 (↑(HAdd.hAdd n 1)) …
    -/
  · convert intervalIntegral.integral_comp_sub_right _ 1
      /-
        case h.e'_3.h.e'_5
        n : Nat
        hn0 : Not (Eq n 0)
        hn : LE.le 1 n
        ⊢ Eq 1 (HSub.hSub 2 1)
      -/
    · norm_num
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.h.e'_6
        n : Nat
        hn0 : Not (Eq n 0)
        hn : LE.le 1 n
        ⊢ Eq (↑n) (HSub.hSub (↑(HAdd.hAdd n 1)) 1)
      -/
    · simp only [Nat.cast_add, Nat.cast_one, add_sub_cancel_right]
      /-
        🎉 no goals
      -/
    /-
      case neg.calc_4
      n : Nat
      hn0 : Not (Eq n 0)
      hn : LE.le 1 n
      ⊢ Eq (intervalIntegral (fun x => Inv.inv x) 1 (↑n) MeasureTheory.MeasureSpace. …
    -/
  · convert integral_inv _
      /-
        case h.e'_3.h.e'_1
        n : Nat
        hn0 : Not (Eq n 0)
        hn : LE.le 1 n
        ⊢ Eq (↑n) (HDiv.hDiv (↑n) 1)
      -/
    · rw [div_one]
      /-
        🎉 no goals
      -/
    · simp only [Nat.one_le_cast, hn, Set.uIcc_of_le, Set.mem_Icc, Nat.cast_nonneg,
        and_true, not_le, zero_lt_one]


theorem log_le_harmonic_floor (y : ℝ) (hy : 0 ≤ y) :
    Real.log y ≤ harmonic ⌊y⌋₊ := by
  /-
    y : Real
    hy : LE.le 0 y
    ⊢ LE.le (Real.log y) ↑(harmonic (Nat.floor y))
  -/
  by_cases h0 : y = 0
    /-
      case pos
      y : Real
      hy : LE.le 0 y
      h0 : Eq y 0
      ⊢ LE.le (Real.log y) ↑(harmonic (Nat.floor y))
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  · calc
      _ ≤ Real.log ↑(Nat.floor y + 1) := ?_
      _ ≤ _ := log_add_one_le_harmonic _
    /-
      case neg
      y : Real
      hy : LE.le 0 y
      h0 : Not (Eq y 0)
      ⊢ LE.le (Real.log y) (Real.log ↑(HAdd.hAdd (Nat.floor y) 1))
    -/
    gcongr
    /-
      case neg.hxy
      y : Real
      hy : LE.le 0 y
      h0 : Not (Eq y 0)
      ⊢ LE.le y ↑(HAdd.hAdd (Nat.floor y) 1)
    -/
    apply (Nat.le_ceil y).trans
    /-
      case neg.hxy
      y : Real
      hy : LE.le 0 y
      h0 : Not (Eq y 0)
      ⊢ LE.le ↑(Nat.ceil y) ↑(HAdd.hAdd (Nat.floor y) 1)
    -/
    norm_cast
    /-
      case neg.hxy
      y : Real
      hy : LE.le 0 y
      h0 : Not (Eq y 0)
      ⊢ LE.le (Nat.ceil y) (HAdd.hAdd (Nat.floor y) 1)
    -/
    exact Nat.ceil_le_floor_add_one y
    /-
      🎉 no goals
    -/


theorem harmonic_floor_le_one_add_log (y : ℝ) (hy : 1 ≤ y) :
    harmonic ⌊y⌋₊ ≤ 1 + Real.log y := by
  /-
    y : Real
    hy : LE.le 1 y
    ⊢ LE.le (↑(harmonic (Nat.floor y))) (HAdd.hAdd 1 (Real.log y))
  -/
  refine (harmonic_le_one_add_log _).trans ?_
  /-
    y : Real
    hy : LE.le 1 y
    ⊢ LE.le (HAdd.hAdd 1 (Real.log ↑(Nat.floor y))) (HAdd.hAdd 1 (Real.log y))
  -/
  gcongr
    /-
      case bc.hx
      y : Real
      hy : LE.le 1 y
      ⊢ LT.lt 0 ↑(Nat.floor y)
    -/
  · exact_mod_cast Nat.floor_pos.mpr hy
    /-
      🎉 no goals
    -/
    /-
      case bc.hxy
      y : Real
      hy : LE.le 1 y
      ⊢ LE.le (↑(Nat.floor y)) y
    -/
  · exact Nat.floor_le <| zero_le_one.trans hy
    /-
      🎉 no goals
    -/

