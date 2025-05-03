/-- The sequence with `n`-th term `harmonic n - log (n + 1)`. -/
noncomputable def eulerMascheroniSeq (n : ℕ) : ℝ := harmonic n - log (n + 1)


lemma eulerMascheroniSeq_zero : eulerMascheroniSeq 0 = 0 := by
  /-
    ⊢ Eq (Real.eulerMascheroniSeq 0) 0
  -/
  simp [eulerMascheroniSeq, harmonic_zero]
  /-
    🎉 no goals
  -/


lemma strictMono_eulerMascheroniSeq : StrictMono eulerMascheroniSeq := by
  /-
    ⊢ StrictMono Real.eulerMascheroniSeq
  -/
  refine strictMono_nat_of_lt_succ (fun n ↦ ?_)
  rw [eulerMascheroniSeq, eulerMascheroniSeq, ← sub_pos, sub_sub_sub_comm,
    harmonic_succ, add_comm, Rat.cast_add, add_sub_cancel_right,
    ← log_div (by positivity) (by positivity), add_div, Nat.cast_add_one,
    Nat.cast_add_one, div_self (by positivity), sub_pos, one_div, Rat.cast_inv, Rat.cast_add,
    Rat.cast_one, Rat.cast_natCast]
  /-
    n : Nat
    ⊢ LT.lt (Real.log (HAdd.hAdd 1 (Inv.inv (HAdd.hAdd (↑n) 1)))) (Inv.inv (HAdd.h …
  -/
  refine (log_lt_sub_one_of_pos ?_ (ne_of_gt <| lt_add_of_pos_right _ ?_)).trans_le (le_of_eq ?_)
    /-
      case refine_1
      n : Nat
      ⊢ LT.lt 0 (HAdd.hAdd 1 (Inv.inv (HAdd.hAdd (↑n) 1)))
    -/
  · positivity
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      ⊢ LT.lt 0 (Inv.inv (HAdd.hAdd (↑n) 1))
    -/
  · positivity
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      n : Nat
      ⊢ Eq (HSub.hSub (HAdd.hAdd 1 (Inv.inv (HAdd.hAdd (↑n) 1))) 1) (Inv.inv (HAdd.h …
    -/
  · simp only [add_sub_cancel_left]
    /-
      🎉 no goals
    -/


lemma one_half_lt_eulerMascheroniSeq_six : 1 / 2 < eulerMascheroniSeq 6 := by
  have : eulerMascheroniSeq 6 = 49 / 20 - log 7 := by
    rw [eulerMascheroniSeq]
    norm_num
  /-
    this : Eq (Real.eulerMascheroniSeq 6) (HSub.hSub (49 / 20) (Real.log 7))
    ⊢ LT.lt (1 / 2) (Real.eulerMascheroniSeq 6)
  -/
  rw [this, lt_sub_iff_add_lt, ← lt_sub_iff_add_lt', log_lt_iff_lt_exp (by positivity)]
  /-
    this : Eq (Real.eulerMascheroniSeq 6) (HSub.hSub (49 / 20) (Real.log 7))
    ⊢ LT.lt 7 (Real.exp (HSub.hSub (49 / 20) (1 / 2)))
  -/
  refine lt_of_lt_of_le ?_ (Real.sum_le_exp_of_nonneg (by norm_num) 7)
  /-
    this : Eq (Real.eulerMascheroniSeq 6) (HSub.hSub (49 / 20) (Real.log 7))
    ⊢ LT.lt 7 ((Finset.range 7).sum fun i => HDiv.hDiv (HPow.hPow (HSub.hSub (49 / …
  -/
  simp_rw [Finset.sum_range_succ, Nat.factorial_succ]
  /-
    this : Eq (Real.eulerMascheroniSeq 6) (HSub.hSub (49 / 20) (Real.log 7))
    ⊢ LT.lt 7 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (H …
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- The sequence with `n`-th term `harmonic n - log n`. We use a junk value for `n = 0`, in order
to have the sequence be strictly decreasing. -/
noncomputable def eulerMascheroniSeq' (n : ℕ) : ℝ :=
  if n = 0 then 2 else ↑(harmonic n) - log n


lemma eulerMascheroniSeq'_one : eulerMascheroniSeq' 1 = 1 := by
  /-
    ⊢ Eq (Real.eulerMascheroniSeq' 1) 1
  -/
  simp [eulerMascheroniSeq']
  /-
    🎉 no goals
  -/


lemma strictAnti_eulerMascheroniSeq' : StrictAnti eulerMascheroniSeq' := by
  /-
    ⊢ StrictAnti Real.eulerMascheroniSeq'
  -/
  refine strictAnti_nat_of_succ_lt (fun n ↦ ?_)
  /-
    n : Nat
    ⊢ LT.lt (Real.eulerMascheroniSeq' (HAdd.hAdd n 1)) (Real.eulerMascheroniSeq' n)
  -/
  rcases Nat.eq_zero_or_pos n with rfl | hn
    /-
      case inl
      ⊢ LT.lt (Real.eulerMascheroniSeq' (HAdd.hAdd 0 1)) (Real.eulerMascheroniSeq' 0)
    -/
  · simp [eulerMascheroniSeq']
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    hn : GT.gt n 0
    ⊢ LT.lt (Real.eulerMascheroniSeq' (HAdd.hAdd n 1)) (Real.eulerMascheroniSeq' n)
  -/
  simp_rw [eulerMascheroniSeq', eq_false_intro hn.ne', reduceCtorEq, if_false]
  rw [← sub_pos, sub_sub_sub_comm,
    harmonic_succ, Rat.cast_add, ← sub_sub, sub_self, zero_sub, sub_eq_add_neg, neg_sub,
    ← sub_eq_neg_add, sub_pos, ← log_div (by positivity) (by positivity), ← neg_lt_neg_iff,
    ← log_inv]
  /-
    case inr
    n : Nat
    hn : GT.gt n 0
    ⊢ LT.lt (Real.log (Inv.inv (HDiv.hDiv ↑(HAdd.hAdd n 1) ↑n))) (Neg.neg ↑(Inv.in …
  -/
  refine (log_lt_sub_one_of_pos ?_ ?_).trans_le (le_of_eq ?_)
    /-
      case inr.refine_1
      n : Nat
      hn : GT.gt n 0
      ⊢ LT.lt 0 (Inv.inv (HDiv.hDiv ↑(HAdd.hAdd n 1) ↑n))
    -/
  · positivity
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      n : Nat
      hn : GT.gt n 0
      ⊢ Ne (Inv.inv (HDiv.hDiv ↑(HAdd.hAdd n 1) ↑n)) 1
    -/
  · field_simp
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_3
      n : Nat
      hn : GT.gt n 0
      ⊢ Eq (HSub.hSub (Inv.inv (HDiv.hDiv ↑(HAdd.hAdd n 1) ↑n)) 1) (Neg.neg ↑(Inv.in …
    -/
  · field_simp
    /-
      🎉 no goals
    -/


lemma eulerMascheroniSeq'_six_lt_two_thirds : eulerMascheroniSeq' 6 < 2 / 3 := by
  have h1 : eulerMascheroniSeq' 6 = 49 / 20 - log 6 := by
    rw [eulerMascheroniSeq']
    norm_num
  /-
    h1 : Eq (Real.eulerMascheroniSeq' 6) (HSub.hSub (49 / 20) (Real.log 6))
    ⊢ LT.lt (Real.eulerMascheroniSeq' 6) (2 / 3)
  -/
  rw [h1, sub_lt_iff_lt_add, ← sub_lt_iff_lt_add', lt_log_iff_exp_lt (by positivity)]
  /-
    h1 : Eq (Real.eulerMascheroniSeq' 6) (HSub.hSub (49 / 20) (Real.log 6))
    ⊢ LT.lt (Real.exp (HSub.hSub (49 / 20) (2 / 3))) 6
  -/
  norm_num
  /-
    h1 : Eq (Real.eulerMascheroniSeq' 6) (HSub.hSub (49 / 20) (Real.log 6))
    ⊢ LT.lt (Real.exp (107 / 60)) 6
  -/
  have := rpow_lt_rpow (exp_pos _).le exp_one_lt_d9 (by norm_num : (0 : ℝ) < 107 / 60)
  /-
    h1 : Eq (Real.eulerMascheroniSeq' 6) (HSub.hSub (49 / 20) (Real.log 6))
    this : LT.lt (HPow.hPow (Real.exp 1) (107 / 60)) (HPow.hPow 2.7182818286 (107  …
    ⊢ LT.lt (Real.exp (107 / 60)) 6
  -/
  rw [exp_one_rpow] at this
  /-
    h1 : Eq (Real.eulerMascheroniSeq' 6) (HSub.hSub (49 / 20) (Real.log 6))
    this : LT.lt (Real.exp (107 / 60)) (HPow.hPow 2.7182818286 (107 / 60))
    ⊢ LT.lt (Real.exp (107 / 60)) 6
  -/
  refine lt_trans this ?_
  rw [← rpow_lt_rpow_iff (z := 60), ← rpow_mul, div_mul_cancel₀, ← Nat.cast_ofNat,
    ← Nat.cast_ofNat, rpow_natCast, Nat.cast_ofNat, ← Nat.cast_ofNat (n := 60), rpow_natCast]
    /-
      h1 : Eq (Real.eulerMascheroniSeq' 6) (HSub.hSub (49 / 20) (Real.log 6))
      this : LT.lt (Real.exp (107 / 60)) (HPow.hPow 2.7182818286 (107 / 60))
      ⊢ LT.lt (HPow.hPow 2.7182818286 107) (HPow.hPow 6 (OfNat.ofNat 60))
    -/
  · norm_num
    /-
      🎉 no goals
    -/
  /-
    case h
    h1 : Eq (Real.eulerMascheroniSeq' 6) (HSub.hSub (49 / 20) (Real.log 6))
    this : LT.lt (Real.exp (107 / 60)) (HPow.hPow 2.7182818286 (107 / 60))
    ⊢ Ne 60 0
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


lemma eulerMascheroniSeq_lt_eulerMascheroniSeq' (m n : ℕ) :
    eulerMascheroniSeq m < eulerMascheroniSeq' n := by
  have (r : ℕ) : eulerMascheroniSeq r < eulerMascheroniSeq' r := by
    rcases eq_zero_or_pos r with rfl | hr
    · simp [eulerMascheroniSeq, eulerMascheroniSeq']
    simp only [eulerMascheroniSeq, eulerMascheroniSeq', hr.ne', if_false]
    gcongr
    linarith
  /-
    m n : Nat
    this : ∀ (r : Nat), LT.lt (Real.eulerMascheroniSeq r) (Real.eulerMascheroniSeq …
    ⊢ LT.lt (Real.eulerMascheroniSeq m) (Real.eulerMascheroniSeq' n)
  -/
  apply (strictMono_eulerMascheroniSeq.monotone (le_max_left m n)).trans_lt
  /-
    m n : Nat
    this : ∀ (r : Nat), LT.lt (Real.eulerMascheroniSeq r) (Real.eulerMascheroniSeq …
    ⊢ LT.lt (Real.eulerMascheroniSeq (Max.max m n)) (Real.eulerMascheroniSeq' n)
  -/
  exact (this _).trans_le (strictAnti_eulerMascheroniSeq'.antitone (le_max_right m n))
  /-
    🎉 no goals
  -/


/-- The Euler-Mascheroni constant `γ`. -/
noncomputable def eulerMascheroniConstant : ℝ := limUnder atTop eulerMascheroniSeq


lemma tendsto_eulerMascheroniSeq :
    Tendsto eulerMascheroniSeq atTop (𝓝 eulerMascheroniConstant) := by
  /-
    ⊢ Filter.Tendsto Real.eulerMascheroniSeq Filter.atTop (nhds Real.eulerMaschero …
  -/
  have := tendsto_atTop_ciSup strictMono_eulerMascheroniSeq.monotone ?_
    /-
      case refine_2
      this : Filter.Tendsto Real.eulerMascheroniSeq Filter.atTop (nhds (iSup fun i = …
      ⊢ Filter.Tendsto Real.eulerMascheroniSeq Filter.atTop (nhds Real.eulerMaschero …
    -/
  · rwa [eulerMascheroniConstant, this.limUnder_eq]
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      ⊢ BddAbove (Set.range Real.eulerMascheroniSeq)
    -/
  · exact ⟨_, fun _ ⟨_, hn⟩ ↦ hn ▸ (eulerMascheroniSeq_lt_eulerMascheroniSeq' _ 1).le⟩
    /-
      🎉 no goals
    -/


lemma tendsto_harmonic_sub_log_add_one :
    Tendsto (fun n : ℕ ↦ harmonic n - log (n + 1)) atTop (𝓝 eulerMascheroniConstant) :=
  tendsto_eulerMascheroniSeq


lemma tendsto_eulerMascheroniSeq' :
    Tendsto eulerMascheroniSeq' atTop (𝓝 eulerMascheroniConstant) := by
  suffices Tendsto (fun n ↦ eulerMascheroniSeq' n - eulerMascheroniSeq n) atTop (𝓝 0) by
    simpa using this.add tendsto_eulerMascheroniSeq
  suffices Tendsto (fun x : ℝ ↦ log (x + 1) - log x) atTop (𝓝 0) by
    apply (this.comp tendsto_natCast_atTop_atTop).congr'
    filter_upwards [eventually_ne_atTop 0] with n hn
    simp [eulerMascheroniSeq, eulerMascheroniSeq', eq_false_intro hn]
  suffices Tendsto (fun x : ℝ ↦ log (1 + 1 / x)) atTop (𝓝 0) by
    apply this.congr'
    filter_upwards [eventually_gt_atTop 0] with x hx
    rw [← log_div (by positivity) (by positivity), add_div, div_self hx.ne']
  simpa only [add_zero, log_one] using
    ((tendsto_const_nhds.div_atTop tendsto_id).const_add 1).log (by positivity)


lemma tendsto_harmonic_sub_log :
    Tendsto (fun n : ℕ ↦ harmonic n - log n) atTop (𝓝 eulerMascheroniConstant) := by
  /-
    ⊢ Filter.Tendsto (fun n => HSub.hSub (↑(harmonic n)) (Real.log ↑n)) Filter.atT …
  -/
  apply tendsto_eulerMascheroniSeq'.congr'
  /-
    ⊢ Filter.atTop.EventuallyEq Real.eulerMascheroniSeq' fun n => HSub.hSub (↑(har …
  -/
  filter_upwards [eventually_ne_atTop 0] with n hn
  /-
    case h
    n : Nat
    hn : Ne n 0
    ⊢ Eq (Real.eulerMascheroniSeq' n) (HSub.hSub (↑(harmonic n)) (Real.log ↑n))
  -/
  simp_rw [eulerMascheroniSeq', hn, if_false]
  /-
    🎉 no goals
  -/


lemma eulerMascheroniSeq_lt_eulerMascheroniConstant (n : ℕ) :
    eulerMascheroniSeq n < eulerMascheroniConstant := by
  /-
    n : Nat
    ⊢ LT.lt (Real.eulerMascheroniSeq n) Real.eulerMascheroniConstant
  -/
  refine (strictMono_eulerMascheroniSeq (Nat.lt_succ_self n)).trans_le ?_
  /-
    n : Nat
    ⊢ LE.le (Real.eulerMascheroniSeq n.succ) Real.eulerMascheroniConstant
  -/
  apply strictMono_eulerMascheroniSeq.monotone.ge_of_tendsto tendsto_eulerMascheroniSeq
  /-
    🎉 no goals
  -/


lemma eulerMascheroniConstant_lt_eulerMascheroniSeq' (n : ℕ) :
    eulerMascheroniConstant < eulerMascheroniSeq' n := by
  /-
    n : Nat
    ⊢ LT.lt Real.eulerMascheroniConstant (Real.eulerMascheroniSeq' n)
  -/
  refine lt_of_le_of_lt ?_ (strictAnti_eulerMascheroniSeq' (Nat.lt_succ_self n))
  /-
    n : Nat
    ⊢ LE.le Real.eulerMascheroniConstant (Real.eulerMascheroniSeq' n.succ)
  -/
  apply strictAnti_eulerMascheroniSeq'.antitone.le_of_tendsto tendsto_eulerMascheroniSeq'
  /-
    🎉 no goals
  -/


/-- Lower bound for `γ`. (The true value is about 0.57.) -/
lemma one_half_lt_eulerMascheroniConstant : 1 / 2 < eulerMascheroniConstant :=
  one_half_lt_eulerMascheroniSeq_six.trans (eulerMascheroniSeq_lt_eulerMascheroniConstant _)


/-- Upper bound for `γ`. (The true value is about 0.57.) -/
lemma eulerMascheroniConstant_lt_two_thirds : eulerMascheroniConstant < 2 / 3 :=
  (eulerMascheroniConstant_lt_eulerMascheroniSeq' _).trans eulerMascheroniSeq'_six_lt_two_thirds


