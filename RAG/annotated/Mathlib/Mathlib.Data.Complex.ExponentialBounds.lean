theorem exp_one_near_10 : |exp 1 - 2244083 / 825552| ≤ 1 / 10 ^ 10 := by
  /-
    ⊢ LE.le (abs (HSub.hSub (Real.exp 1) (2244083 / 825552))) (HDiv.hDiv 1 (HPow.h …
  -/
  apply exp_approx_start
  /-
    case h
    ⊢ LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear 0 1 (2244083 / 825552)))) ( …
  -/
  iterate 13 refine exp_1_approx_succ_eq (by norm_num1; rfl) (by norm_cast) ?_
  /-
    case h
    ⊢ LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear 13 1 (HMul.hMul (HSub.hSub  …
  -/
  norm_num1
  /-
    case h
    ⊢ LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear 13 1 (5 / 7)))) (HMul.hMul  …
  -/
  refine exp_approx_end' _ (by norm_num1; rfl) _ (by norm_cast) (by simp) ?_
  /-
    case h
    ⊢ LE.le (abs (HSub.hSub 1 (5 / 7))) (HSub.hSub (243243 / 390625) (HMul.hMul (H …
  -/
                                      /-
                                        🎉 no goals
                                      -/
  rw [_root_.abs_one, abs_of_pos] <;> norm_num1
                                      /-
                                        🎉 no goals
                                      -/


theorem exp_one_near_20 : |exp 1 - 363916618873 / 133877442384| ≤ 1 / 10 ^ 20 := by
  /-
    ⊢ LE.le (abs (HSub.hSub (Real.exp 1) (363916618873 / 133877442384))) (HDiv.hDi …
  -/
  apply exp_approx_start
  /-
    case h
    ⊢ LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear 0 1 (363916618873 / 1338774 …
  -/
  iterate 21 refine exp_1_approx_succ_eq (by norm_num1; rfl) (by norm_cast) ?_
  /-
    case h
    ⊢ LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear 21 1 (HMul.hMul (HSub.hSub  …
  -/
  norm_num1
  /-
    case h
    ⊢ LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear 21 1 (36295539 / 44271641)) …
  -/
  refine exp_approx_end' _ (by norm_num1; rfl) _ (by norm_cast) (by simp) ?_
  /-
    case h
    ⊢ LE.le (abs (HSub.hSub 1 (36295539 / 44271641))) (HSub.hSub (311834363841 / 6 …
  -/
                                      /-
                                        🎉 no goals
                                      -/
  rw [_root_.abs_one, abs_of_pos] <;> norm_num1
                                      /-
                                        🎉 no goals
                                      -/


theorem exp_one_gt_d9 : 2.7182818283 < exp 1 :=
                     /-
                       ⊢ LT.lt 2.7182818283 (HSub.hSub (2244083 / 825552) (HDiv.hDiv 1 (HPow.hPow 10  …
                     -/
  lt_of_lt_of_le (by norm_num) (sub_le_comm.1 (abs_sub_le_iff.1 exp_one_near_10).2)
                     /-
                       🎉 no goals
                     -/


theorem exp_one_lt_d9 : exp 1 < 2.7182818286 :=
                                                                                /-
                                                                                  ⊢ LT.lt (HAdd.hAdd (HDiv.hDiv 1 (HPow.hPow 10 10)) (2244083 / 825552)) 2.71828 …
                                                                                -/
  lt_of_le_of_lt (sub_le_iff_le_add.1 (abs_sub_le_iff.1 exp_one_near_10).1) (by norm_num)
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem exp_neg_one_gt_d9 : 0.36787944116 < exp (-1) := by
  /-
    ⊢ LT.lt 0.36787944116 (Real.exp (-1))
  -/
  rw [exp_neg, lt_inv_comm₀ _ (exp_pos _)]
    /-
      ⊢ LT.lt (Real.exp 1) (Inv.inv 0.36787944116)
    -/
  · refine lt_of_le_of_lt (sub_le_iff_le_add.1 (abs_sub_le_iff.1 exp_one_near_10).1) ?_
    /-
      ⊢ LT.lt (HAdd.hAdd (HDiv.hDiv 1 (HPow.hPow 10 10)) (2244083 / 825552)) (Inv.in …
    -/
    norm_num
    /-
      🎉 no goals
    -/
    /-
      ⊢ LT.lt 0 0.36787944116
    -/
  · norm_num
    /-
      🎉 no goals
    -/


theorem exp_neg_one_lt_d9 : exp (-1) < 0.3678794412 := by
  /-
    ⊢ LT.lt (Real.exp (-1)) 0.3678794412
  -/
  rw [exp_neg, inv_lt_comm₀ (exp_pos _) (by norm_num)]
  /-
    ⊢ LT.lt (Inv.inv 0.3678794412) (Real.exp 1)
  -/
  exact lt_of_lt_of_le (by norm_num) (sub_le_comm.1 (abs_sub_le_iff.1 exp_one_near_10).2)
  /-
    🎉 no goals
  -/


theorem log_two_near_10 : |log 2 - 287209 / 414355| ≤ 1 / 10 ^ 10 := by
  suffices |log 2 - 287209 / 414355| ≤ 1 / 17179869184 + (1 / 10 ^ 10 - 1 / 2 ^ 34) by
    norm_num1 at *
    assumption
  /-
    ⊢ LE.le (abs (HSub.hSub (Real.log 2) (287209 / 414355))) (HAdd.hAdd (1 / 17179 …
  -/
  have t : |(2⁻¹ : ℝ)| = 2⁻¹ := by rw [abs_of_pos]; norm_num
  /-
    t : Eq (abs (Inv.inv 2)) (Inv.inv 2)
    ⊢ LE.le (abs (HSub.hSub (Real.log 2) (287209 / 414355))) (HAdd.hAdd (1 / 17179 …
  -/
  have z := Real.abs_log_sub_add_sum_range_le (show |(2⁻¹ : ℝ)| < 1 by rw [t]; norm_num) 34
  /-
    t : Eq (abs (Inv.inv 2)) (Inv.inv 2)
    z : LE.le (abs (HAdd.hAdd ((Finset.range 34).sum fun i => HDiv.hDiv (HPow.hPow …
    ⊢ LE.le (abs (HSub.hSub (Real.log 2) (287209 / 414355))) (HAdd.hAdd (1 / 17179 …
  -/
  rw [t] at z
  /-
    t : Eq (abs (Inv.inv 2)) (Inv.inv 2)
    z : LE.le (abs (HAdd.hAdd ((Finset.range 34).sum fun i => HDiv.hDiv (HPow.hPow …
    ⊢ LE.le (abs (HSub.hSub (Real.log 2) (287209 / 414355))) (HAdd.hAdd (1 / 17179 …
  -/
  norm_num1 at z
  /-
    t : Eq (abs (Inv.inv 2)) (Inv.inv 2)
    z : LE.le (abs (HAdd.hAdd ((Finset.range 34).sum fun x => HDiv.hDiv (HPow.hPow …
    ⊢ LE.le (abs (HSub.hSub (Real.log 2) (287209 / 414355))) (HAdd.hAdd (1 / 17179 …
  -/
  rw [one_div (2 : ℝ), log_inv, ← sub_eq_add_neg, _root_.abs_sub_comm] at z
  /-
    t : Eq (abs (Inv.inv 2)) (Inv.inv 2)
    z : LE.le (abs (HSub.hSub (Real.log 2) ((Finset.range 34).sum fun x => HDiv.hD …
    ⊢ LE.le (abs (HSub.hSub (Real.log 2) (287209 / 414355))) (HAdd.hAdd (1 / 17179 …
  -/
  apply le_trans (_root_.abs_sub_le _ _ _) (add_le_add z _)
  /-
    t : Eq (abs (Inv.inv 2)) (Inv.inv 2)
    z : LE.le (abs (HSub.hSub (Real.log 2) ((Finset.range 34).sum fun x => HDiv.hD …
    ⊢ LE.le (abs (HSub.hSub ((Finset.range 34).sum fun x => HDiv.hDiv (HPow.hPow ( …
  -/
  simp_rw [sum_range_succ]
  /-
    t : Eq (abs (Inv.inv 2)) (Inv.inv 2)
    z : LE.le (abs (HSub.hSub (Real.log 2) ((Finset.range 34).sum fun x => HDiv.hD …
    ⊢ LE.le (abs (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd …
  -/
  norm_num
  /-
    t : Eq (abs (Inv.inv 2)) (Inv.inv 2)
    z : LE.le (abs (HSub.hSub (Real.log 2) ((Finset.range 34).sum fun x => HDiv.hD …
    ⊢ LE.le (abs (30417026706710207 / 51397301678363663775930777600)) (7011591 / 1 …
  -/
                      /-
                        🎉 no goals
                      -/
  rw [abs_of_pos] <;> norm_num
                      /-
                        🎉 no goals
                      -/


theorem log_two_gt_d9 : 0.6931471803 < log 2 :=
                     /-
                       ⊢ LT.lt 0.6931471803 (HSub.hSub (287209 / 414355) (HDiv.hDiv 1 (HPow.hPow 10 1 …
                     -/
  lt_of_lt_of_le (by norm_num1) (sub_le_comm.1 (abs_sub_le_iff.1 log_two_near_10).2)
                     /-
                       🎉 no goals
                     -/


theorem log_two_lt_d9 : log 2 < 0.6931471808 :=
                                                                                /-
                                                                                  ⊢ LT.lt (HAdd.hAdd (HDiv.hDiv 1 (HPow.hPow 10 10)) (287209 / 414355)) 0.693147 …
                                                                                -/
  lt_of_le_of_lt (sub_le_iff_le_add.1 (abs_sub_le_iff.1 log_two_near_10).1) (by norm_num)
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


