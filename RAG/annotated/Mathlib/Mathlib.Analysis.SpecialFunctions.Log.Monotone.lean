theorem log_mul_self_monotoneOn : MonotoneOn (fun x : ℝ => log x * x) { x | 1 ≤ x } := by
  -- TODO: can be strengthened to exp (-1) ≤ x
  /-
    ⊢ MonotoneOn (fun x => HMul.hMul (Real.log x) x) (setOf fun x => LE.le 1 x)
  -/
  simp only [MonotoneOn, mem_setOf_eq]
  /-
    ⊢ ∀ ⦃a : Real⦄, LE.le 1 a → ∀ ⦃b : Real⦄, LE.le 1 b → LE.le a b → LE.le (HMul. …
  -/
  intro x hex y hey hxy
  /-
    x : Real
    hex : LE.le 1 x
    y : Real
    hey : LE.le 1 y
    hxy : LE.le x y
    ⊢ LE.le (HMul.hMul (Real.log x) x) (HMul.hMul (Real.log y) y)
  -/
  have y_pos : 0 < y := lt_of_lt_of_le zero_lt_one hey
  /-
    x : Real
    hex : LE.le 1 x
    y : Real
    hey : LE.le 1 y
    hxy : LE.le x y
    y_pos : LT.lt 0 y
    ⊢ LE.le (HMul.hMul (Real.log x) x) (HMul.hMul (Real.log y) y)
  -/
  gcongr
  /-
    case b0
    x : Real
    hex : LE.le 1 x
    y : Real
    hey : LE.le 1 y
    hxy : LE.le x y
    y_pos : LT.lt 0 y
    ⊢ LE.le 0 (Real.log y)
  -/
  rwa [le_log_iff_exp_le y_pos, Real.exp_zero]
  /-
    🎉 no goals
  -/


theorem log_div_self_antitoneOn : AntitoneOn (fun x : ℝ => log x / x) { x | exp 1 ≤ x } := by
  /-
    ⊢ AntitoneOn (fun x => HDiv.hDiv (Real.log x) x) (setOf fun x => LE.le (Real.e …
  -/
  simp only [AntitoneOn, mem_setOf_eq]
  /-
    ⊢ ∀ ⦃a : Real⦄, LE.le (Real.exp 1) a → ∀ ⦃b : Real⦄, LE.le (Real.exp 1) b → LE …
  -/
  intro x hex y hey hxy
  /-
    x : Real
    hex : LE.le (Real.exp 1) x
    y : Real
    hey : LE.le (Real.exp 1) y
    hxy : LE.le x y
    ⊢ LE.le (HDiv.hDiv (Real.log y) y) (HDiv.hDiv (Real.log x) x)
  -/
  have x_pos : 0 < x := (exp_pos 1).trans_le hex
  /-
    x : Real
    hex : LE.le (Real.exp 1) x
    y : Real
    hey : LE.le (Real.exp 1) y
    hxy : LE.le x y
    x_pos : LT.lt 0 x
    ⊢ LE.le (HDiv.hDiv (Real.log y) y) (HDiv.hDiv (Real.log x) x)
  -/
  have y_pos : 0 < y := (exp_pos 1).trans_le hey
  /-
    x : Real
    hex : LE.le (Real.exp 1) x
    y : Real
    hey : LE.le (Real.exp 1) y
    hxy : LE.le x y
    x_pos : LT.lt 0 x
    y_pos : LT.lt 0 y
    ⊢ LE.le (HDiv.hDiv (Real.log y) y) (HDiv.hDiv (Real.log x) x)
  -/
  have hlogx : 1 ≤ log x := by rwa [le_log_iff_exp_le x_pos]
  /-
    x : Real
    hex : LE.le (Real.exp 1) x
    y : Real
    hey : LE.le (Real.exp 1) y
    hxy : LE.le x y
    x_pos : LT.lt 0 x
    y_pos : LT.lt 0 y
    hlogx : LE.le 1 (Real.log x)
    ⊢ LE.le (HDiv.hDiv (Real.log y) y) (HDiv.hDiv (Real.log x) x)
  -/
  have hyx : 0 ≤ y / x - 1 := by rwa [le_sub_iff_add_le, le_div_iff₀ x_pos, zero_add, one_mul]
  /-
    x : Real
    hex : LE.le (Real.exp 1) x
    y : Real
    hey : LE.le (Real.exp 1) y
    hxy : LE.le x y
    x_pos : LT.lt 0 x
    y_pos : LT.lt 0 y
    hlogx : LE.le 1 (Real.log x)
    hyx : LE.le 0 (HSub.hSub (HDiv.hDiv y x) 1)
    ⊢ LE.le (HDiv.hDiv (Real.log y) y) (HDiv.hDiv (Real.log x) x)
  -/
  rw [div_le_iff₀ y_pos, ← sub_le_sub_iff_right (log x)]
  calc
    log y - log x = log (y / x) := by rw [log_div y_pos.ne' x_pos.ne']
    _ ≤ y / x - 1 := log_le_sub_one_of_pos (div_pos y_pos x_pos)
    _ ≤ log x * (y / x - 1) := le_mul_of_one_le_left hyx hlogx
    _ = log x / x * y - log x := by ring


theorem log_div_self_rpow_antitoneOn {a : ℝ} (ha : 0 < a) :
    AntitoneOn (fun x : ℝ => log x / x ^ a) { x | exp (1 / a) ≤ x } := by
  /-
    a : Real
    ha : LT.lt 0 a
    ⊢ AntitoneOn (fun x => HDiv.hDiv (Real.log x) (HPow.hPow x a)) (setOf fun x => …
  -/
  simp only [AntitoneOn, mem_setOf_eq]
  /-
    a : Real
    ha : LT.lt 0 a
    ⊢ ∀ ⦃a_1 : Real⦄, LE.le (Real.exp (HDiv.hDiv 1 a)) a_1 → ∀ ⦃b : Real⦄, LE.le ( …
  -/
  intro x hex y _ hxy
  /-
    a : Real
    ha : LT.lt 0 a
    x : Real
    hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
    y : Real
    x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
    hxy : LE.le x y
    ⊢ LE.le (HDiv.hDiv (Real.log y) (HPow.hPow y a)) (HDiv.hDiv (Real.log x) (HPow …
  -/
  have x_pos : 0 < x := lt_of_lt_of_le (exp_pos (1 / a)) hex
  /-
    a : Real
    ha : LT.lt 0 a
    x : Real
    hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
    y : Real
    x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
    hxy : LE.le x y
    x_pos : LT.lt 0 x
    ⊢ LE.le (HDiv.hDiv (Real.log y) (HPow.hPow y a)) (HDiv.hDiv (Real.log x) (HPow …
  -/
  have y_pos : 0 < y := by linarith
  /-
    a : Real
    ha : LT.lt 0 a
    x : Real
    hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
    y : Real
    x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
    hxy : LE.le x y
    x_pos : LT.lt 0 x
    y_pos : LT.lt 0 y
    ⊢ LE.le (HDiv.hDiv (Real.log y) (HPow.hPow y a)) (HDiv.hDiv (Real.log x) (HPow …
  -/
  nth_rw 1 [← rpow_one y]
  /-
    a : Real
    ha : LT.lt 0 a
    x : Real
    hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
    y : Real
    x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
    hxy : LE.le x y
    x_pos : LT.lt 0 x
    y_pos : LT.lt 0 y
    ⊢ LE.le (HDiv.hDiv (Real.log (HPow.hPow y 1)) (HPow.hPow y a)) (HDiv.hDiv (Rea …
  -/
  nth_rw 1 [← rpow_one x]
  rw [← div_self (ne_of_lt ha).symm, div_eq_mul_one_div a a, rpow_mul y_pos.le, rpow_mul x_pos.le,
    log_rpow (rpow_pos_of_pos y_pos a), log_rpow (rpow_pos_of_pos x_pos a), mul_div_assoc,
    mul_div_assoc, mul_le_mul_left (one_div_pos.mpr ha)]
  /-
    a : Real
    ha : LT.lt 0 a
    x : Real
    hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
    y : Real
    x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
    hxy : LE.le x y
    x_pos : LT.lt 0 x
    y_pos : LT.lt 0 y
    ⊢ LE.le (HDiv.hDiv (Real.log (HPow.hPow y a)) (HPow.hPow y a)) (HDiv.hDiv (Rea …
  -/
  refine log_div_self_antitoneOn ?_ ?_ ?_
    /-
      case refine_1
      a : Real
      ha : LT.lt 0 a
      x : Real
      hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
      y : Real
      x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
      hxy : LE.le x y
      x_pos : LT.lt 0 x
      y_pos : LT.lt 0 y
      ⊢ Membership.mem (setOf fun x => LE.le (Real.exp 1) x) (HPow.hPow x a)
    -/
  · simp only [Set.mem_setOf_eq]
    /-
      case refine_1
      a : Real
      ha : LT.lt 0 a
      x : Real
      hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
      y : Real
      x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
      hxy : LE.le x y
      x_pos : LT.lt 0 x
      y_pos : LT.lt 0 y
      ⊢ LE.le (Real.exp 1) (HPow.hPow x a)
    -/
    convert rpow_le_rpow _ hex (le_of_lt ha) using 1
      /-
        case h.e'_3
        a : Real
        ha : LT.lt 0 a
        x : Real
        hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
        y : Real
        x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
        hxy : LE.le x y
        x_pos : LT.lt 0 x
        y_pos : LT.lt 0 y
        ⊢ Eq (Real.exp 1) (HPow.hPow (Real.exp (HDiv.hDiv 1 a)) a)
      -/
    · rw [← exp_mul]
      /-
        case h.e'_3
        a : Real
        ha : LT.lt 0 a
        x : Real
        hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
        y : Real
        x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
        hxy : LE.le x y
        x_pos : LT.lt 0 x
        y_pos : LT.lt 0 y
        ⊢ Eq (Real.exp 1) (Real.exp (HMul.hMul (HDiv.hDiv 1 a) a))
      -/
      simp only [Real.exp_eq_exp]
      /-
        case h.e'_3
        a : Real
        ha : LT.lt 0 a
        x : Real
        hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
        y : Real
        x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
        hxy : LE.le x y
        x_pos : LT.lt 0 x
        y_pos : LT.lt 0 y
        ⊢ Eq 1 (HMul.hMul (HDiv.hDiv 1 a) a)
      -/
      field_simp
      /-
        🎉 no goals
      -/
    /-
      case refine_1
      a : Real
      ha : LT.lt 0 a
      x : Real
      hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
      y : Real
      x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
      hxy : LE.le x y
      x_pos : LT.lt 0 x
      y_pos : LT.lt 0 y
      ⊢ LE.le 0 (Real.exp (HDiv.hDiv 1 a))
    -/
    positivity
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a : Real
      ha : LT.lt 0 a
      x : Real
      hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
      y : Real
      x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
      hxy : LE.le x y
      x_pos : LT.lt 0 x
      y_pos : LT.lt 0 y
      ⊢ Membership.mem (setOf fun x => LE.le (Real.exp 1) x) (HPow.hPow y a)
    -/
  · simp only [Set.mem_setOf_eq]
    /-
      case refine_2
      a : Real
      ha : LT.lt 0 a
      x : Real
      hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
      y : Real
      x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
      hxy : LE.le x y
      x_pos : LT.lt 0 x
      y_pos : LT.lt 0 y
      ⊢ LE.le (Real.exp 1) (HPow.hPow y a)
    -/
    convert rpow_le_rpow _ (_root_.trans hex hxy) (le_of_lt ha) using 1
      /-
        case h.e'_3
        a : Real
        ha : LT.lt 0 a
        x : Real
        hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
        y : Real
        x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
        hxy : LE.le x y
        x_pos : LT.lt 0 x
        y_pos : LT.lt 0 y
        ⊢ Eq (Real.exp 1) (HPow.hPow (Real.exp (HDiv.hDiv 1 a)) a)
      -/
    · rw [← exp_mul]
      /-
        case h.e'_3
        a : Real
        ha : LT.lt 0 a
        x : Real
        hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
        y : Real
        x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
        hxy : LE.le x y
        x_pos : LT.lt 0 x
        y_pos : LT.lt 0 y
        ⊢ Eq (Real.exp 1) (Real.exp (HMul.hMul (HDiv.hDiv 1 a) a))
      -/
      simp only [Real.exp_eq_exp]
      /-
        case h.e'_3
        a : Real
        ha : LT.lt 0 a
        x : Real
        hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
        y : Real
        x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
        hxy : LE.le x y
        x_pos : LT.lt 0 x
        y_pos : LT.lt 0 y
        ⊢ Eq 1 (HMul.hMul (HDiv.hDiv 1 a) a)
      -/
      field_simp
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      a : Real
      ha : LT.lt 0 a
      x : Real
      hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
      y : Real
      x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
      hxy : LE.le x y
      x_pos : LT.lt 0 x
      y_pos : LT.lt 0 y
      ⊢ LE.le 0 (Real.exp (HDiv.hDiv 1 a))
    -/
    positivity
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    a : Real
    ha : LT.lt 0 a
    x : Real
    hex : LE.le (Real.exp (HDiv.hDiv 1 a)) x
    y : Real
    x✝ : LE.le (Real.exp (HDiv.hDiv 1 a)) y
    hxy : LE.le x y
    x_pos : LT.lt 0 x
    y_pos : LT.lt 0 y
    ⊢ LE.le (HPow.hPow x a) (HPow.hPow y a)
  -/
  gcongr
  /-
    🎉 no goals
  -/


theorem log_div_sqrt_antitoneOn : AntitoneOn (fun x : ℝ => log x / √x) { x | exp 2 ≤ x } := by
  /-
    ⊢ AntitoneOn (fun x => HDiv.hDiv (Real.log x) x.sqrt) (setOf fun x => LE.le (R …
  -/
  simp_rw [sqrt_eq_rpow]
  /-
    ⊢ AntitoneOn (fun x => HDiv.hDiv (Real.log x) (HPow.hPow x (1 / 2))) (setOf fu …
  -/
  convert @log_div_self_rpow_antitoneOn (1 / 2) (by norm_num)
  /-
    case h.e'_6.h.e'_2.h.h.e'_3.h.e'_1
    x✝ : Real
    ⊢ Eq 2 (HDiv.hDiv 1 (1 / 2))
  -/
  norm_num
  /-
    🎉 no goals
  -/


