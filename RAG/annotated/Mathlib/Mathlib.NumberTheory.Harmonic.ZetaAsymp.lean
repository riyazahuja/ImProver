@[inherit_doc] local notation "γ" => eulerMascheroniConstant


/-- Auxiliary function used in studying zeta-function asymptotics. -/
noncomputable def term (n : ℕ) (s : ℝ) : ℝ := ∫ x : ℝ in n..(n + 1), (x - n) / x ^ (s + 1)


/-- Sum of finitely many `term`s. -/
noncomputable def term_sum (s : ℝ) (N : ℕ) : ℝ := ∑ n ∈ Finset.range N, term (n + 1) s


/-- Topological sum of `term`s. -/
noncomputable def term_tsum (s : ℝ) : ℝ := ∑' n, term (n + 1) s


lemma term_nonneg (n : ℕ) (s : ℝ) : 0 ≤ term n s := by
  /-
    n : Nat
    s : Real
    ⊢ LE.le 0 (ZetaAsymptotics.term n s)
  -/
  rw [term, intervalIntegral.integral_of_le (by simp)]
  /-
    n : Nat
    s : Real
    ⊢ LE.le 0 (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
  -/
  refine setIntegral_nonneg measurableSet_Ioc (fun x hx ↦ ?_)
  /-
    n : Nat
    s x : Real
    hx : Membership.mem (Set.Ioc (↑n) (HAdd.hAdd (↑n) 1)) x
    ⊢ LE.le 0 (HDiv.hDiv (HSub.hSub x ↑n) (HPow.hPow x (HAdd.hAdd s 1)))
  -/
  refine div_nonneg ?_ (rpow_nonneg ?_ _)
  /-
    case refine_1
    n : Nat
    s x : Real
    hx : Membership.mem (Set.Ioc (↑n) (HAdd.hAdd (↑n) 1)) x
    ⊢ LE.le 0 (HSub.hSub x ↑n)
  -/
  all_goals linarith [hx.1]
  /-
    🎉 no goals
  -/


lemma term_welldef {n : ℕ} (hn : 0 < n) {s : ℝ} (hs : 0 < s) :
    IntervalIntegrable (fun x : ℝ ↦ (x - n) / x ^ (s + 1)) volume n (n + 1) := by
  /-
    n : Nat
    hn : LT.lt 0 n
    s : Real
    hs : LT.lt 0 s
    ⊢ IntervalIntegrable (fun x => HDiv.hDiv (HSub.hSub x ↑n) (HPow.hPow x (HAdd.h …
  -/
  rw [intervalIntegrable_iff_integrableOn_Icc_of_le (by linarith)]
  /-
    n : Nat
    hn : LT.lt 0 n
    s : Real
    hs : LT.lt 0 s
    ⊢ MeasureTheory.IntegrableOn (fun x => HDiv.hDiv (HSub.hSub x ↑n) (HPow.hPow x …
  -/
  refine (continuousOn_of_forall_continuousAt fun x hx ↦ ContinuousAt.div ?_ ?_ ?_).integrableOn_Icc
    /-
      case refine_1
      n : Nat
      hn : LT.lt 0 n
      s : Real
      hs : LT.lt 0 s
      x : Real
      hx : Membership.mem (Set.Icc (↑n) (HAdd.hAdd (↑n) 1)) x
      ⊢ ContinuousAt (fun x => HSub.hSub x ↑n) x
    -/
  · fun_prop
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      hn : LT.lt 0 n
      s : Real
      hs : LT.lt 0 s
      x : Real
      hx : Membership.mem (Set.Icc (↑n) (HAdd.hAdd (↑n) 1)) x
      ⊢ ContinuousAt (fun x => HPow.hPow x (HAdd.hAdd s 1)) x
    -/
  · apply continuousAt_id.rpow_const (Or.inr <| by linarith)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      n : Nat
      hn : LT.lt 0 n
      s : Real
      hs : LT.lt 0 s
      x : Real
      hx : Membership.mem (Set.Icc (↑n) (HAdd.hAdd (↑n) 1)) x
      ⊢ Ne (HPow.hPow x (HAdd.hAdd s 1)) 0
    -/
  · exact (rpow_pos_of_pos ((Nat.cast_pos.mpr hn).trans_le hx.1) _).ne'
    /-
      🎉 no goals
    -/


lemma term_one {n : ℕ} (hn : 0 < n) :
    term n 1 = (log (n + 1) - log n) -  1 / (n + 1) := by
  have hv : ∀ x ∈ uIcc (n : ℝ) (n + 1), 0 < x := by
    intro x hx
    rw [uIcc_of_le (by simp only [le_add_iff_nonneg_right, zero_le_one])] at hx
    exact (Nat.cast_pos.mpr hn).trans_le hx.1
  calc term n 1
    _ = ∫ x : ℝ in n..(n + 1), (x - n) / x ^ 2 := by
      simp_rw [term, one_add_one_eq_two, ← Nat.cast_two (R := ℝ), rpow_natCast]
    _ = ∫ x : ℝ in n..(n + 1), (1 / x - n / x ^ 2) := by
      refine intervalIntegral.integral_congr (fun x hx ↦ ?_)
      field_simp [(hv x hx).ne']
      ring
    _ = (∫ x : ℝ in n..(n + 1), 1 / x) - n * ∫ x : ℝ in n..(n + 1), 1 / x ^ 2 := by
      simp_rw [← mul_one_div (n : ℝ)]
      rw [intervalIntegral.integral_sub]
      · simp_rw [intervalIntegral.integral_const_mul]
      · exact intervalIntegral.intervalIntegrable_one_div (fun x hx ↦ (hv x hx).ne') (by fun_prop)
      · exact (intervalIntegral.intervalIntegrable_one_div
          (fun x hx ↦ (sq_pos_of_pos (hv x hx)).ne') (by fun_prop)).const_mul _
    _ = (log (↑n + 1) - log ↑n) - n * ∫ x : ℝ in n..(n + 1), 1 / x ^ 2 := by
      congr 1
      rw [integral_one_div_of_pos, log_div]
      all_goals positivity
    _ = (log (↑n + 1) - log ↑n) - n * ∫ x : ℝ in n..(n + 1), x ^ (-2 : ℝ) := by
      congr 2
      refine intervalIntegral.integral_congr (fun x hx ↦ ?_)
      rw [rpow_neg, one_div, ← Nat.cast_two (R := ℝ), rpow_natCast]
      exact (hv x hx).le
    _ = log (↑n + 1) - log ↑n - n * (1 / n - 1 / (n + 1)) := by
      rw [integral_rpow]
      · simp_rw [sub_div, (by norm_num : (-2 : ℝ) + 1 = -1), div_neg, div_one, neg_sub_neg,
          rpow_neg_one, ← one_div]
      · refine Or.inr ⟨by norm_num, not_mem_uIcc_of_lt ?_ ?_⟩
        all_goals positivity
    _ = log (↑n + 1) - log ↑n - 1 / (↑n + 1) := by
      congr 1
      field_simp


lemma term_sum_one (N : ℕ) : term_sum 1 N = log (N + 1) - harmonic (N + 1) + 1 := by
  /-
    N : Nat
    ⊢ Eq (ZetaAsymptotics.term_sum 1 N) (HAdd.hAdd (HSub.hSub (Real.log (HAdd.hAdd …
  -/
  induction' N with N hN
  · simp_rw [term_sum, Finset.sum_range_zero, harmonic_succ, harmonic_zero,
      Nat.cast_zero, zero_add, Nat.cast_one, inv_one, Rat.cast_one, log_one, sub_add_cancel]
    /-
      case succ
      N : Nat
      hN : Eq (ZetaAsymptotics.term_sum 1 N) (HAdd.hAdd (HSub.hSub (Real.log (HAdd.h …
      ⊢ Eq (ZetaAsymptotics.term_sum 1 (HAdd.hAdd N 1)) (HAdd.hAdd (HSub.hSub (Real. …
    -/
  · unfold term_sum at hN ⊢
    rw [Finset.sum_range_succ, hN, harmonic_succ (N + 1),
      term_one (by positivity : 0 < N + 1)]
    /-
      case succ
      N : Nat
      hN : Eq ((Finset.range N).sum fun n => ZetaAsymptotics.term (HAdd.hAdd n 1) 1) …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSub.hSub (Real.log (HAdd.hAdd (↑N) 1)) ↑(harmonic …
    -/
    push_cast
    /-
      case succ
      N : Nat
      hN : Eq ((Finset.range N).sum fun n => ZetaAsymptotics.term (HAdd.hAdd n 1) 1) …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSub.hSub (Real.log (HAdd.hAdd (↑N) 1)) ↑(harmonic …
    -/
    ring_nf
    /-
      🎉 no goals
    -/


/-- The topological sum of `ZetaAsymptotics.term (n + 1) 1` over all `n : ℕ` is `1 - γ`. This is
proved by directly evaluating the sum of the first `N` terms and using the limit definition of `γ`.
-/
lemma term_tsum_one : HasSum (fun n ↦ term (n + 1) 1) (1 - γ) := by
  /-
    ⊢ HasSum (fun n => ZetaAsymptotics.term (HAdd.hAdd n 1) 1) (HSub.hSub 1 Real.e …
  -/
  rw [hasSum_iff_tendsto_nat_of_nonneg (fun n ↦ term_nonneg (n + 1) 1)]
  /-
    ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => ZetaAsymptotics.term  …
  -/
  show Tendsto (fun N ↦ term_sum 1 N) atTop _
  /-
    ⊢ Filter.Tendsto (fun N => ZetaAsymptotics.term_sum 1 N) Filter.atTop (nhds (H …
  -/
  simp_rw [term_sum_one, sub_eq_neg_add]
  /-
    ⊢ Filter.Tendsto (fun N => HAdd.hAdd (HAdd.hAdd (Neg.neg ↑(harmonic (HAdd.hAdd …
  -/
  refine Tendsto.add ?_ tendsto_const_nhds
  /-
    ⊢ Filter.Tendsto (fun N => HAdd.hAdd (Neg.neg ↑(harmonic (HAdd.hAdd N 1))) (Re …
  -/
  have := (tendsto_eulerMascheroniSeq'.comp (tendsto_add_atTop_nat 1)).neg
  /-
    this : Filter.Tendsto (fun x => Neg.neg (Function.comp Real.eulerMascheroniSeq …
    ⊢ Filter.Tendsto (fun N => HAdd.hAdd (Neg.neg ↑(harmonic (HAdd.hAdd N 1))) (Re …
  -/
  refine this.congr' (Eventually.of_forall (fun n ↦ ?_))
  /-
    this : Filter.Tendsto (fun x => Neg.neg (Function.comp Real.eulerMascheroniSeq …
    n : Nat
    ⊢ Eq (Neg.neg (Function.comp Real.eulerMascheroniSeq' (fun a => HAdd.hAdd a 1) …
  -/
  simp_rw [Function.comp_apply, eulerMascheroniSeq', reduceCtorEq, if_false]
  /-
    this : Filter.Tendsto (fun x => Neg.neg (Function.comp Real.eulerMascheroniSeq …
    n : Nat
    ⊢ Eq (Neg.neg (HSub.hSub (↑(harmonic (HAdd.hAdd n 1))) (Real.log ↑(HAdd.hAdd n …
  -/
  push_cast
  /-
    this : Filter.Tendsto (fun x => Neg.neg (Function.comp Real.eulerMascheroniSeq …
    n : Nat
    ⊢ Eq (Neg.neg (HSub.hSub (↑(harmonic (HAdd.hAdd n 1))) (Real.log (HAdd.hAdd (↑ …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma term_of_lt {n : ℕ} (hn : 0 < n) {s : ℝ} (hs : 1 < s) :
    term n s = 1 / (s - 1) * (1 / n ^ (s - 1) - 1 / (n + 1) ^ (s - 1))
    - n / s * (1 / n ^ s - 1 / (n + 1) ^ s) := by
  have hv : ∀ x ∈ uIcc (n : ℝ) (n + 1), 0 < x := by
    intro x hx
    rw [uIcc_of_le (by simp only [le_add_iff_nonneg_right, zero_le_one])] at hx
    exact (Nat.cast_pos.mpr hn).trans_le hx.1
  calc term n s
    _ = ∫ x : ℝ in n..(n + 1), (x - n) / x ^ (s + 1) := by rfl
    _ = ∫ x : ℝ in n..(n + 1), (x ^ (-s) - n * x ^ (-(s + 1))) := by
      refine intervalIntegral.integral_congr (fun x hx ↦ ?_)
      rw [sub_div, rpow_add_one (hv x hx).ne', mul_comm, ← div_div, div_self (hv x hx).ne',
        rpow_neg (hv x hx).le, rpow_neg (hv x hx).le, one_div, rpow_add_one (hv x hx).ne', mul_comm,
        div_eq_mul_inv]
    _ = (∫ x : ℝ in n..(n + 1), x ^ (-s)) - n * (∫ x : ℝ in n..(n + 1), x ^ (-(s + 1))) := by
      rw [intervalIntegral.integral_sub, intervalIntegral.integral_const_mul] <;>
      [skip; apply IntervalIntegrable.const_mul] <;>
      · refine intervalIntegral.intervalIntegrable_rpow (Or.inr <| not_mem_uIcc_of_lt ?_ ?_)
        · exact_mod_cast hn
        · linarith
    _ = 1 / (s - 1) * (1 / n ^ (s - 1) - 1 / (n + 1) ^ (s - 1))
          - n / s * (1 / n ^ s - 1 / (n + 1) ^ s) := by
      have : 0 ∉ uIcc (n : ℝ) (n + 1) := (lt_irrefl _ <| hv _ ·)
      rw [integral_rpow (Or.inr ⟨by linarith, this⟩), integral_rpow (Or.inr ⟨by linarith, this⟩)]
      congr 1
      · rw [show -s + 1 = -(s - 1) by ring, div_neg, ← neg_div, mul_comm, mul_one_div, neg_sub,
          rpow_neg (Nat.cast_nonneg _), one_div, rpow_neg (by linarith), one_div]
      · rw [show -(s + 1) + 1 = -s by ring, div_neg, ← neg_div, neg_sub, div_mul_eq_mul_div,
          mul_div_assoc, rpow_neg (Nat.cast_nonneg _), one_div, rpow_neg (by linarith), one_div]


lemma term_sum_of_lt (N : ℕ) {s : ℝ} (hs : 1 < s) :
    term_sum s N = 1 / (s - 1) * (1 - 1 / (N + 1) ^ (s - 1))
    - 1 / s * ((∑ n ∈ Finset.range N, 1 / (n + 1 : ℝ) ^ s) - N / (N + 1) ^ s) := by
  /-
    N : Nat
    s : Real
    hs : LT.lt 1 s
    ⊢ Eq (ZetaAsymptotics.term_sum s N) (HSub.hSub (HMul.hMul (HDiv.hDiv 1 (HSub.h …
  -/
  simp only [term_sum]
  /-
    N : Nat
    s : Real
    hs : LT.lt 1 s
    ⊢ Eq ((Finset.range N).sum fun n => ZetaAsymptotics.term (HAdd.hAdd n 1) s) (H …
  -/
  conv => enter [1, 2, n]; rw [term_of_lt (by simp) hs]
  /-
    N : Nat
    s : Real
    hs : LT.lt 1 s
    ⊢ Eq ((Finset.range N).sum fun n => HSub.hSub (HMul.hMul (HDiv.hDiv 1 (HSub.hS …
  -/
  rw [Finset.sum_sub_distrib]
  /-
    N : Nat
    s : Real
    hs : LT.lt 1 s
    ⊢ Eq (HSub.hSub ((Finset.range N).sum fun x => HMul.hMul (HDiv.hDiv 1 (HSub.hS …
  -/
  congr 1
    /-
      case e_a
      N : Nat
      s : Real
      hs : LT.lt 1 s
      ⊢ Eq ((Finset.range N).sum fun x => HMul.hMul (HDiv.hDiv 1 (HSub.hSub s 1)) (H …
    -/
  · induction' N with N hN
      /-
        case e_a.zero
        s : Real
        hs : LT.lt 1 s
        ⊢ Eq ((Finset.range 0).sum fun x => HMul.hMul (HDiv.hDiv 1 (HSub.hSub s 1)) (H …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case e_a.succ
        s : Real
        hs : LT.lt 1 s
        N : Nat
        hN : Eq ((Finset.range N).sum fun x => HMul.hMul (HDiv.hDiv 1 (HSub.hSub s 1)) …
        ⊢ Eq ((Finset.range (HAdd.hAdd N 1)).sum fun x => HMul.hMul (HDiv.hDiv 1 (HSub …
      -/
    · rw [Finset.sum_range_succ, hN, Nat.cast_add_one]
      /-
        case e_a.succ
        s : Real
        hs : LT.lt 1 s
        N : Nat
        hN : Eq ((Finset.range N).sum fun x => HMul.hMul (HDiv.hDiv 1 (HSub.hSub s 1)) …
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv 1 (HSub.hSub s 1)) (HSub.hSub 1 (HDiv.hD …
      -/
      ring_nf
      /-
        🎉 no goals
      -/
    /-
      case e_a
      N : Nat
      s : Real
      hs : LT.lt 1 s
      ⊢ Eq ((Finset.range N).sum fun x => HMul.hMul (HDiv.hDiv (↑(HAdd.hAdd x 1)) s) …
    -/
  · simp_rw [mul_comm (_ / _), ← mul_div_assoc, div_eq_mul_inv _ s, ← Finset.sum_mul, mul_one]
    /-
      case e_a
      N : Nat
      s : Real
      hs : LT.lt 1 s
      ⊢ Eq (HMul.hMul ((Finset.range N).sum fun i => HMul.hMul (HSub.hSub (HDiv.hDiv …
    -/
    congr 1
    /-
      case e_a.e_a
      N : Nat
      s : Real
      hs : LT.lt 1 s
      ⊢ Eq ((Finset.range N).sum fun i => HMul.hMul (HSub.hSub (HDiv.hDiv 1 (HPow.hP …
    -/
    induction' N with N hN
      /-
        case e_a.e_a.zero
        s : Real
        hs : LT.lt 1 s
        ⊢ Eq ((Finset.range 0).sum fun i => HMul.hMul (HSub.hSub (HDiv.hDiv 1 (HPow.hP …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case e_a.e_a.succ
        s : Real
        hs : LT.lt 1 s
        N : Nat
        hN : Eq ((Finset.range N).sum fun i => HMul.hMul (HSub.hSub (HDiv.hDiv 1 (HPow …
        ⊢ Eq ((Finset.range (HAdd.hAdd N 1)).sum fun i => HMul.hMul (HSub.hSub (HDiv.h …
      -/
    · simp_rw [Finset.sum_range_succ, hN, Nat.cast_add_one, sub_eq_add_neg, add_assoc]
      /-
        case e_a.e_a.succ
        s : Real
        hs : LT.lt 1 s
        N : Nat
        hN : Eq ((Finset.range N).sum fun i => HMul.hMul (HSub.hSub (HDiv.hDiv 1 (HPow …
        ⊢ Eq (HAdd.hAdd ((Finset.range N).sum fun n => HDiv.hDiv 1 (HPow.hPow (HAdd.hA …
      -/
      congr 1
      /-
        case e_a.e_a.succ.e_a
        s : Real
        hs : LT.lt 1 s
        N : Nat
        hN : Eq ((Finset.range N).sum fun i => HMul.hMul (HSub.hSub (HDiv.hDiv 1 (HPow …
        ⊢ Eq (HAdd.hAdd (Neg.neg (HDiv.hDiv (↑N) (HPow.hPow (HAdd.hAdd (↑N) 1) s))) (H …
      -/
      ring_nf
      /-
        🎉 no goals
      -/


/-- For `1 < s`, the topological sum of `ZetaAsymptotics.term (n + 1) s` over all `n : ℕ` is
`1 / (s - 1) - ζ s / s`.
-/
lemma term_tsum_of_lt {s : ℝ} (hs : 1 < s) :
    term_tsum s = (1 / (s - 1) - 1 / s * ∑' n : ℕ, 1 / (n + 1 : ℝ) ^ s) := by
  /-
    s : Real
    hs : LT.lt 1 s
    ⊢ Eq (ZetaAsymptotics.term_tsum s) (HSub.hSub (HDiv.hDiv 1 (HSub.hSub s 1)) (H …
  -/
  apply HasSum.tsum_eq
  /-
    case ha
    s : Real
    hs : LT.lt 1 s
    ⊢ HasSum (fun b => ZetaAsymptotics.term (HAdd.hAdd b 1) s) (HSub.hSub (HDiv.hD …
  -/
  rw [hasSum_iff_tendsto_nat_of_nonneg (fun n ↦ term_nonneg (n + 1) s)]
  /-
    case ha
    s : Real
    hs : LT.lt 1 s
    ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => ZetaAsymptotics.term  …
  -/
  change Tendsto (fun N ↦ term_sum s N) atTop _
  /-
    case ha
    s : Real
    hs : LT.lt 1 s
    ⊢ Filter.Tendsto (fun N => ZetaAsymptotics.term_sum s N) Filter.atTop (nhds (H …
  -/
  simp_rw [term_sum_of_lt _ hs]
  /-
    case ha
    s : Real
    hs : LT.lt 1 s
    ⊢ Filter.Tendsto (fun N => HSub.hSub (HMul.hMul (HDiv.hDiv 1 (HSub.hSub s 1))  …
  -/
  apply Tendsto.sub
    /-
      case ha.hf
      s : Real
      hs : LT.lt 1 s
      ⊢ Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv 1 (HSub.hSub s 1)) (HSub.hSub  …
    -/
  · rw [show 𝓝 (1 / (s - 1)) = 𝓝 (1 / (s - 1) - 1 / (s - 1) * 0) by simp]
    /-
      case ha.hf
      s : Real
      hs : LT.lt 1 s
      ⊢ Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv 1 (HSub.hSub s 1)) (HSub.hSub  …
    -/
    simp_rw [mul_sub, mul_one]
    /-
      case ha.hf
      s : Real
      hs : LT.lt 1 s
      ⊢ Filter.Tendsto (fun x => HSub.hSub (HDiv.hDiv 1 (HSub.hSub s 1)) (HMul.hMul  …
    -/
    refine tendsto_const_nhds.sub (Tendsto.const_mul _ ?_)
    /-
      case ha.hf
      s : Real
      hs : LT.lt 1 s
      ⊢ Filter.Tendsto (fun k => HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd (↑k) 1) (HSub.hSu …
    -/
    refine tendsto_const_nhds.div_atTop <| (tendsto_rpow_atTop (by linarith)).comp ?_
    /-
      case ha.hf
      s : Real
      hs : LT.lt 1 s
      ⊢ Filter.Tendsto (fun k => HAdd.hAdd (↑k) 1) Filter.atTop Filter.atTop
    -/
    exact tendsto_atTop_add_const_right _ _ tendsto_natCast_atTop_atTop
    /-
      🎉 no goals
    -/
    /-
      case ha.hg
      s : Real
      hs : LT.lt 1 s
      ⊢ Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv 1 s) (HSub.hSub ((Finset.range …
    -/
  · rw [← sub_zero (tsum _)]
    /-
      case ha.hg
      s : Real
      hs : LT.lt 1 s
      ⊢ Filter.Tendsto (fun x => HMul.hMul (HDiv.hDiv 1 s) (HSub.hSub ((Finset.range …
    -/
    apply (((Summable.hasSum ?_).tendsto_sum_nat).sub ?_).const_mul
      /-
        s : Real
        hs : LT.lt 1 s
        ⊢ Summable fun i => HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd (↑i) 1) s)
      -/
    · exact_mod_cast (summable_nat_add_iff 1).mpr (summable_one_div_nat_rpow.mpr hs)
      /-
        🎉 no goals
      -/
      /-
        s : Real
        hs : LT.lt 1 s
        ⊢ Filter.Tendsto (fun k => HDiv.hDiv (↑k) (HPow.hPow (HAdd.hAdd (↑k) 1) s)) Fi …
      -/
    · apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
        /-
          case hh
          s : Real
          hs : LT.lt 1 s
          ⊢ Filter.Tendsto ?m.104592 Filter.atTop (nhds 0)
        -/
      · change Tendsto (fun n : ℕ ↦ (1 / ↑(n + 1) : ℝ) ^ (s - 1)) ..
        /-
          case hh
          s : Real
          hs : LT.lt 1 s
          ⊢ Filter.Tendsto (fun n => HPow.hPow (HDiv.hDiv 1 ↑(HAdd.hAdd n 1)) (HSub.hSub …
        -/
        rw [show 𝓝 (0 : ℝ) = 𝓝 (0 ^ (s - 1)) by rw [zero_rpow]; linarith]
        /-
          case hh
          s : Real
          hs : LT.lt 1 s
          ⊢ Filter.Tendsto (fun n => HPow.hPow (HDiv.hDiv 1 ↑(HAdd.hAdd n 1)) (HSub.hSub …
        -/
        refine Tendsto.rpow_const ?_ (Or.inr <| by linarith)
        /-
          case hh
          s : Real
          hs : LT.lt 1 s
          ⊢ Filter.Tendsto (fun n => HDiv.hDiv 1 ↑(HAdd.hAdd n 1)) Filter.atTop (nhds 0)
        -/
        exact (tendsto_const_div_atTop_nhds_zero_nat _).comp (tendsto_add_atTop_nat _)
        /-
          🎉 no goals
        -/
        /-
          case hgf
          s : Real
          hs : LT.lt 1 s
          ⊢ LE.le (fun x => 0) fun k => HDiv.hDiv (↑k) (HPow.hPow (HAdd.hAdd (↑k) 1) s)
        -/
      · intro n
        /-
          case hgf
          s : Real
          hs : LT.lt 1 s
          n : Nat
          ⊢ LE.le ((fun x => 0) n) ((fun k => HDiv.hDiv (↑k) (HPow.hPow (HAdd.hAdd (↑k)  …
        -/
        positivity
        /-
          🎉 no goals
        -/
        /-
          case hfh
          s : Real
          hs : LT.lt 1 s
          ⊢ LE.le (fun k => HDiv.hDiv (↑k) (HPow.hPow (HAdd.hAdd (↑k) 1) s)) fun n => HP …
        -/
      · intro n
        /-
          case hfh
          s : Real
          hs : LT.lt 1 s
          n : Nat
          ⊢ LE.le ((fun k => HDiv.hDiv (↑k) (HPow.hPow (HAdd.hAdd (↑k) 1) s)) n) ((fun n …
        -/
        dsimp only
        /-
          case hfh
          s : Real
          hs : LT.lt 1 s
          n : Nat
          ⊢ LE.le (HDiv.hDiv (↑n) (HPow.hPow (HAdd.hAdd (↑n) 1) s)) (HPow.hPow (HDiv.hDi …
        -/
        transitivity (n + 1) / (n + 1) ^ s
          /-
            s : Real
            hs : LT.lt 1 s
            n : Nat
            ⊢ LE.le (HDiv.hDiv (↑n) (HPow.hPow (HAdd.hAdd (↑n) 1) s)) (HDiv.hDiv (HAdd.hAd …
          -/
        · gcongr
          /-
            case hab
            s : Real
            hs : LT.lt 1 s
            n : Nat
            ⊢ LE.le (↑n) (HAdd.hAdd (↑n) 1)
          -/
          linarith
          /-
            🎉 no goals
          -/
          /-
            s : Real
            hs : LT.lt 1 s
            n : Nat
            ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (↑n) 1) (HPow.hPow (HAdd.hAdd (↑n) 1) s)) (HPow. …
          -/
        · apply le_of_eq
          /-
            case hab
            s : Real
            hs : LT.lt 1 s
            n : Nat
            ⊢ Eq (HDiv.hDiv (HAdd.hAdd (↑n) 1) (HPow.hPow (HAdd.hAdd (↑n) 1) s)) (HPow.hPo …
          -/
          rw [rpow_sub_one, ← div_mul, div_one, mul_comm, one_div, inv_rpow, ← div_eq_mul_inv]
            /-
              case hab
              s : Real
              hs : LT.lt 1 s
              n : Nat
              ⊢ Eq (HDiv.hDiv (HAdd.hAdd (↑n) 1) (HPow.hPow (HAdd.hAdd (↑n) 1) s)) (HDiv.hDi …
            -/
          · norm_cast
            /-
              🎉 no goals
            -/
          /-
            case hab.hx
            s : Real
            hs : LT.lt 1 s
            n : Nat
            ⊢ LE.le 0 ↑(HAdd.hAdd n 1)
          -/
          all_goals positivity
          /-
            🎉 no goals
          -/


/-- Reformulation of `ZetaAsymptotics.term_tsum_of_lt` which is useful for some computations
below. -/
lemma zeta_limit_aux1 {s : ℝ} (hs : 1 < s) :
    (∑' n : ℕ, 1 / (n + 1 : ℝ) ^ s) - 1 / (s - 1) = 1 - s * term_tsum s := by
  /-
    s : Real
    hs : LT.lt 1 s
    ⊢ Eq (HSub.hSub (tsum fun n => HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd (↑n) 1) s)) ( …
  -/
  rw [term_tsum_of_lt hs]
  /-
    s : Real
    hs : LT.lt 1 s
    ⊢ Eq (HSub.hSub (tsum fun n => HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd (↑n) 1) s)) ( …
  -/
  generalize (∑' n : ℕ, 1 / (n + 1 : ℝ) ^ s) = Z
  /-
    s : Real
    hs : LT.lt 1 s
    Z : Real
    ⊢ Eq (HSub.hSub Z (HDiv.hDiv 1 (HSub.hSub s 1))) (HSub.hSub 1 (HMul.hMul s (HS …
  -/
  field_simp [(show s - 1 ≠ 0 by linarith)]
  /-
    s : Real
    hs : LT.lt 1 s
    Z : Real
    ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul Z (HSub.hSub s 1)) 1) (HMul.hMul (HSub.h …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


lemma continuousOn_term (n : ℕ) :
    ContinuousOn (fun x ↦ term (n + 1) x) (Ici 1) := by
  -- TODO: can this be shortened using the lemma
  -- `continuous_parametric_intervalIntegral_of_continuous'` from https://github.com/leanprover-community/mathlib4/pull/11185?
  /-
    n : Nat
    ⊢ ContinuousOn (fun x => ZetaAsymptotics.term (HAdd.hAdd n 1) x) (Set.Ici 1)
  -/
  simp only [term, intervalIntegral.integral_of_le (by linarith : (↑(n + 1) : ℝ) ≤ ↑(n + 1) + 1)]
  /-
    n : Nat
    ⊢ ContinuousOn (fun x => MeasureTheory.integral (MeasureTheory.MeasureSpace.vo …
  -/
  apply continuousOn_of_dominated (bound := fun x ↦ (x - ↑(n + 1)) / x ^ (2 : ℝ))
    /-
      case hF_meas
      n : Nat
      ⊢ ∀ (x : Real), Membership.mem (Set.Ici 1) x → MeasureTheory.AEStronglyMeasura …
    -/
  · exact fun s hs ↦ (term_welldef (by simp) (zero_lt_one.trans_le hs)).1.1
    /-
      🎉 no goals
    -/
    /-
      case h_bound
      n : Nat
      ⊢ ∀ (x : Real), Membership.mem (Set.Ici 1) x → Filter.Eventually (fun a => LE. …
    -/
  · intro s (hs : 1 ≤ s)
    /-
      case h_bound
      n : Nat
      s : Real
      hs : LE.le 1 s
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (HDiv.hDiv (HSub.hSub a ↑(HAdd. …
    -/
    rw [ae_restrict_iff' measurableSet_Ioc]
    /-
      case h_bound
      n : Nat
      s : Real
      hs : LE.le 1 s
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd …
    -/
    filter_upwards with x hx
    /-
      case h_bound.h
      n : Nat
      s : Real
      hs : LE.le 1 s
      x : Real
      hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
      ⊢ LE.le (Norm.norm (HDiv.hDiv (HSub.hSub x ↑(HAdd.hAdd n 1)) (HPow.hPow x (HAd …
    -/
    have : 0 < x := lt_trans (by positivity) hx.1
    /-
      case h_bound.h
      n : Nat
      s : Real
      hs : LE.le 1 s
      x : Real
      hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
      this : LT.lt 0 x
      ⊢ LE.le (Norm.norm (HDiv.hDiv (HSub.hSub x ↑(HAdd.hAdd n 1)) (HPow.hPow x (HAd …
    -/
    rw [norm_of_nonneg (div_nonneg (sub_nonneg.mpr hx.1.le) (by positivity)), Nat.cast_add_one]
    /-
      case h_bound.h
      n : Nat
      s : Real
      hs : LE.le 1 s
      x : Real
      hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
      this : LT.lt 0 x
      ⊢ LE.le (HDiv.hDiv (HSub.hSub x (HAdd.hAdd (↑n) 1)) (HPow.hPow x (HAdd.hAdd s  …
    -/
    apply div_le_div_of_nonneg_left
      /-
        case h_bound.h.ha
        n : Nat
        s : Real
        hs : LE.le 1 s
        x : Real
        hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
        this : LT.lt 0 x
        ⊢ LE.le 0 (HSub.hSub x (HAdd.hAdd (↑n) 1))
      -/
    · exact_mod_cast sub_nonneg.mpr hx.1.le
      /-
        🎉 no goals
      -/
      /-
        case h_bound.h.hc
        n : Nat
        s : Real
        hs : LE.le 1 s
        x : Real
        hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
        this : LT.lt 0 x
        ⊢ LT.lt 0 (HPow.hPow x 2)
      -/
    · positivity
      /-
        🎉 no goals
      -/
      /-
        case h_bound.h.h
        n : Nat
        s : Real
        hs : LE.le 1 s
        x : Real
        hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
        this : LT.lt 0 x
        ⊢ LE.le (HPow.hPow x 2) (HPow.hPow x (HAdd.hAdd s 1))
      -/
    · exact rpow_le_rpow_of_exponent_le (le_trans (by simp) hx.1.le) (by linarith)
      /-
        🎉 no goals
      -/
    /-
      case bound_integrable
      n : Nat
      ⊢ MeasureTheory.Integrable (fun x => HDiv.hDiv (HSub.hSub x ↑(HAdd.hAdd n 1))  …
    -/
  · rw [← IntegrableOn, ← intervalIntegrable_iff_integrableOn_Ioc_of_le (by linarith)]
    /-
      case bound_integrable
      n : Nat
      ⊢ IntervalIntegrable (fun x => HDiv.hDiv (HSub.hSub x ↑(HAdd.hAdd n 1)) (HPow. …
    -/
    exact_mod_cast term_welldef (by omega : 0 < (n + 1)) zero_lt_one
    /-
      🎉 no goals
    -/
    /-
      case h_cont
      n : Nat
      ⊢ Filter.Eventually (fun a => ContinuousOn (fun x => HDiv.hDiv (HSub.hSub a ↑( …
    -/
  · rw [ae_restrict_iff' measurableSet_Ioc]
    /-
      case h_cont
      n : Nat
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd …
    -/
    filter_upwards with x hx
    /-
      case h_cont.h
      n : Nat
      x : Real
      hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
      ⊢ ContinuousOn (fun x_1 => HDiv.hDiv (HSub.hSub x ↑(HAdd.hAdd n 1)) (HPow.hPow …
    -/
    refine continuousOn_of_forall_continuousAt (fun s (hs : 1 ≤ s) ↦ continuousAt_const.div ?_ ?_)
      /-
        case h_cont.h.refine_1
        n : Nat
        x : Real
        hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
        s : Real
        hs : LE.le 1 s
        ⊢ ContinuousAt (fun x_1 => HPow.hPow x (HAdd.hAdd x_1 1)) s
      -/
    · exact continuousAt_const.rpow (continuousAt_id.add continuousAt_const) (Or.inr (by linarith))
      /-
        🎉 no goals
      -/
      /-
        case h_cont.h.refine_2
        n : Nat
        x : Real
        hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
        s : Real
        hs : LE.le 1 s
        ⊢ Ne (HPow.hPow x (HAdd.hAdd s 1)) 0
      -/
    · exact (rpow_pos_of_pos ((Nat.cast_pos.mpr (by simp)).trans hx.1) _).ne'
      /-
        🎉 no goals
      -/


lemma continuousOn_term_tsum : ContinuousOn term_tsum (Ici 1) := by
  -- We use dominated convergence, using `fun n ↦ term n 1` as our uniform bound (since `term` is
  -- monotone decreasing in `s`.)
  /-
    ⊢ ContinuousOn ZetaAsymptotics.term_tsum (Set.Ici 1)
  -/
  refine continuousOn_tsum (fun i ↦ continuousOn_term _) term_tsum_one.summable (fun n s hs ↦ ?_)
  /-
    n : Nat
    s : Real
    hs : Membership.mem (Set.Ici 1) s
    ⊢ LE.le (Norm.norm (ZetaAsymptotics.term (HAdd.hAdd n 1) s)) (ZetaAsymptotics. …
  -/
  rw [term, term, norm_of_nonneg]
    /-
      n : Nat
      s : Real
      hs : Membership.mem (Set.Ici 1) s
      ⊢ LE.le (intervalIntegral (fun x => HDiv.hDiv (HSub.hSub x ↑(HAdd.hAdd n 1)) ( …
    -/
  · simp_rw [intervalIntegral.integral_of_le (by linarith : (↑(n + 1) : ℝ) ≤ ↑(n + 1) + 1)]
    /-
      n : Nat
      s : Real
      hs : Membership.mem (Set.Ici 1) s
      ⊢ LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (S …
    -/
    refine setIntegral_mono_on ?_ ?_ measurableSet_Ioc (fun x hx ↦ ?_)
      /-
        case refine_1
        n : Nat
        s : Real
        hs : Membership.mem (Set.Ici 1) s
        ⊢ MeasureTheory.IntegrableOn (fun x => HDiv.hDiv (HSub.hSub x ↑(HAdd.hAdd n 1) …
      -/
    · exact (term_welldef n.succ_pos (zero_lt_one.trans_le hs)).1
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        n : Nat
        s : Real
        hs : Membership.mem (Set.Ici 1) s
        ⊢ MeasureTheory.IntegrableOn (fun x => HDiv.hDiv (HSub.hSub x ↑(HAdd.hAdd n 1) …
      -/
    · exact (term_welldef n.succ_pos zero_lt_one).1
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        n : Nat
        s : Real
        hs : Membership.mem (Set.Ici 1) s
        x : Real
        hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
        ⊢ LE.le (HDiv.hDiv (HSub.hSub x ↑(HAdd.hAdd n 1)) (HPow.hPow x (HAdd.hAdd s 1) …
      -/
    · rw [div_le_div_iff_of_pos_left] -- leave side-goals to end and kill them all together
        /-
          case refine_3
          n : Nat
          s : Real
          hs : Membership.mem (Set.Ici 1) s
          x : Real
          hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
          ⊢ LE.le (HPow.hPow x (HAdd.hAdd 1 1)) (HPow.hPow x (HAdd.hAdd s 1))
        -/
      · apply rpow_le_rpow_of_exponent_le
          /-
            case refine_3.hx
            n : Nat
            s : Real
            hs : Membership.mem (Set.Ici 1) s
            x : Real
            hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
            ⊢ LE.le 1 x
          -/
        · exact (lt_of_le_of_lt (by simp) hx.1).le
          /-
            🎉 no goals
          -/
          /-
            case refine_3.hyz
            n : Nat
            s : Real
            hs : Membership.mem (Set.Ici 1) s
            x : Real
            hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
            ⊢ LE.le (HAdd.hAdd 1 1) (HAdd.hAdd s 1)
          -/
        · linarith [mem_Ici.mp hs]
          /-
            🎉 no goals
          -/
        /-
          case refine_3.ha
          n : Nat
          s : Real
          hs : Membership.mem (Set.Ici 1) s
          x : Real
          hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
          ⊢ LT.lt 0 (HSub.hSub x ↑(HAdd.hAdd n 1))
        -/
      · linarith [hx.1]
        /-
          🎉 no goals
        -/
      /-
        case refine_3.hb
        n : Nat
        s : Real
        hs : Membership.mem (Set.Ici 1) s
        x : Real
        hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
        ⊢ LT.lt 0 (HPow.hPow x (HAdd.hAdd s 1))
      -/
      all_goals apply rpow_pos_of_pos ((Nat.cast_nonneg _).trans_lt hx.1)
      /-
        🎉 no goals
      -/
    /-
      n : Nat
      s : Real
      hs : Membership.mem (Set.Ici 1) s
      ⊢ LE.le 0 (intervalIntegral (fun x => HDiv.hDiv (HSub.hSub x ↑(HAdd.hAdd n 1)) …
    -/
  · rw [intervalIntegral.integral_of_le (by linarith)]
    /-
      n : Nat
      s : Real
      hs : Membership.mem (Set.Ici 1) s
      ⊢ LE.le 0 (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
    -/
    refine setIntegral_nonneg measurableSet_Ioc (fun x hx ↦ div_nonneg ?_ (rpow_nonneg ?_ _))
    /-
      case refine_1
      n : Nat
      s : Real
      hs : Membership.mem (Set.Ici 1) s
      x : Real
      hx : Membership.mem (Set.Ioc (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑(HAdd.hAdd n 1))  …
      ⊢ LE.le 0 (HSub.hSub x ↑(HAdd.hAdd n 1))
    -/
    all_goals linarith [hx.1]
    /-
      🎉 no goals
    -/


/-- First version of the limit formula, with a limit over real numbers tending to 1 from above. -/
lemma tendsto_riemannZeta_sub_one_div_nhds_right :
    Tendsto (fun s : ℝ ↦ riemannZeta s - 1 / (s - 1)) (𝓝[>] 1) (𝓝 γ) := by
  suffices Tendsto (fun s : ℝ ↦ (∑' n : ℕ, 1 / (n + 1 : ℝ) ^ s) - 1 / (s - 1))
    (𝓝[>] 1) (𝓝 γ) by
    apply ((Complex.continuous_ofReal.tendsto _).comp this).congr'
    filter_upwards [self_mem_nhdsWithin] with s hs
    simp only [Function.comp_apply, Complex.ofReal_sub, Complex.ofReal_div,
      Complex.ofReal_one, sub_left_inj, Complex.ofReal_tsum]
    rw [zeta_eq_tsum_one_div_nat_add_one_cpow (by simpa using hs)]
    congr 1 with n
    rw [Complex.ofReal_cpow (by positivity)]
    norm_cast
  suffices aux2 : Tendsto (fun s : ℝ ↦ (∑' n : ℕ, 1 / (n + 1 : ℝ) ^ s) - 1 / (s - 1))
    (𝓝[>] 1) (𝓝 (1 - term_tsum 1)) by
    have := term_tsum_one.tsum_eq
    rw [← term_tsum, eq_sub_iff_add_eq, ← eq_sub_iff_add_eq'] at this
    simpa only [this] using aux2
  /-
    ⊢ Filter.Tendsto (fun s => HSub.hSub (tsum fun n => HDiv.hDiv 1 (HPow.hPow (HA …
  -/
  apply Tendsto.congr'
    /-
      case hl
      ⊢ (nhdsWithin 1 (Set.Ioi 1)).EventuallyEq ?f₁ fun s => HSub.hSub (tsum fun n = …
    -/
  · filter_upwards [self_mem_nhdsWithin] with s hs using (zeta_limit_aux1 hs).symm
    /-
      🎉 no goals
    -/
    /-
      case h
      ⊢ Filter.Tendsto (fun s => HSub.hSub 1 (HMul.hMul s (ZetaAsymptotics.term_tsum …
    -/
  · apply tendsto_const_nhds.sub
    /-
      case h
      ⊢ Filter.Tendsto (fun x => HMul.hMul x (ZetaAsymptotics.term_tsum x)) (nhdsWit …
    -/
    rw [← one_mul (term_tsum 1)]
    /-
      case h
      ⊢ Filter.Tendsto (fun x => HMul.hMul x (ZetaAsymptotics.term_tsum x)) (nhdsWit …
    -/
    apply (tendsto_id.mono_left nhdsWithin_le_nhds).mul
    /-
      case h
      ⊢ Filter.Tendsto ZetaAsymptotics.term_tsum (nhdsWithin 1 (Set.Ioi 1)) (nhds (Z …
    -/
    have := continuousOn_term_tsum.continuousWithinAt left_mem_Ici
    /-
      case h
      this : ContinuousWithinAt ZetaAsymptotics.term_tsum (Set.Ici 1) 1
      ⊢ Filter.Tendsto ZetaAsymptotics.term_tsum (nhdsWithin 1 (Set.Ioi 1)) (nhds (Z …
    -/
    exact Tendsto.mono_left this (nhdsWithin_mono _ Ioi_subset_Ici_self)
    /-
      🎉 no goals
    -/


/-- The function `ζ s - 1 / (s - 1)` tends to `γ` as `s → 1`. -/
theorem _root_.tendsto_riemannZeta_sub_one_div :
    Tendsto (fun s : ℂ ↦ riemannZeta s - 1 / (s - 1)) (𝓝[≠] 1) (𝓝 γ) := by
  -- We use the removable-singularity theorem to show that *some* limit over `𝓝[≠] (1 : ℂ)` exists,
  -- and then use the previous result to deduce that this limit must be `γ`.
  /-
    ⊢ Filter.Tendsto (fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub.hSub s …
  -/
  let f (s : ℂ) := riemannZeta s - 1 / (s - 1)
  suffices ∃ C, Tendsto f (𝓝[≠] 1) (𝓝 C) by
    cases' this with C hC
    suffices Tendsto (fun s : ℝ ↦ f s) _ _
      from (tendsto_nhds_unique this tendsto_riemannZeta_sub_one_div_nhds_right) ▸ hC
    refine hC.comp (tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ ?_ ?_)
    · exact (Complex.continuous_ofReal.tendsto 1).mono_left (nhdsWithin_le_nhds ..)
    · filter_upwards [self_mem_nhdsWithin] with a ha
      rw [mem_compl_singleton_iff, ← Complex.ofReal_one, Ne, Complex.ofReal_inj]
      exact ne_of_gt ha
  /-
    f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
    ⊢ Exists fun C => Filter.Tendsto f (nhdsWithin 1 (HasCompl.compl (Singleton.si …
  -/
  refine ⟨_, Complex.tendsto_limUnder_of_differentiable_on_punctured_nhds_of_isLittleO ?_ ?_⟩
    /-
      case refine_1
      f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
      ⊢ Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhdsWithin 1 (Has …
    -/
  · filter_upwards [self_mem_nhdsWithin] with s hs
    /-
      case h
      f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
      s : Complex
      hs : Membership.mem (HasCompl.compl (Singleton.singleton 1)) s
      ⊢ DifferentiableAt Complex f s
    -/
    refine (differentiableAt_riemannZeta hs).sub ((differentiableAt_const _).div ?_ ?_)
      /-
        case h.refine_1
        f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
        s : Complex
        hs : Membership.mem (HasCompl.compl (Singleton.singleton 1)) s
        ⊢ DifferentiableAt Complex (fun s => HSub.hSub s 1) s
      -/
    · fun_prop
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2
        f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
        s : Complex
        hs : Membership.mem (HasCompl.compl (Singleton.singleton 1)) s
        ⊢ Ne (HSub.hSub s 1) 0
      -/
    · rwa [mem_compl_singleton_iff, ← sub_ne_zero] at hs
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
      ⊢ Asymptotics.IsLittleO (nhdsWithin 1 (HasCompl.compl (Singleton.singleton 1)) …
    -/
  · refine Asymptotics.isLittleO_of_tendsto' ?_ ?_
      /-
        case refine_2.refine_1
        f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
        ⊢ Filter.Eventually (fun x => Eq (Inv.inv (HSub.hSub x 1)) 0 → Eq (HSub.hSub ( …
      -/
    · filter_upwards [self_mem_nhdsWithin] with t ht ht'
      /-
        case h
        f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
        t : Complex
        ht : Membership.mem (HasCompl.compl (Singleton.singleton 1)) t
        ht' : Eq (Inv.inv (HSub.hSub t 1)) 0
        ⊢ Eq (HSub.hSub (f t) (f 1)) 0
      -/
      rw [inv_eq_zero, sub_eq_zero] at ht'
      /-
        case h
        f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
        t : Complex
        ht : Membership.mem (HasCompl.compl (Singleton.singleton 1)) t
        ht' : Eq t 1
        ⊢ Eq (HSub.hSub (f t) (f 1)) 0
      -/
      tauto
      /-
        🎉 no goals
      -/
    · simp_rw [div_eq_mul_inv, inv_inv, sub_mul,
        (by ring_nf : 𝓝 (0 : ℂ) = 𝓝 ((1 - 1) - f 1 * (1 - 1)))]
      /-
        case refine_2.refine_2
        f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
        ⊢ Filter.Tendsto (fun x => HSub.hSub (HMul.hMul (f x) (HSub.hSub x 1)) (HMul.h …
      -/
      apply Tendsto.sub
        /-
          case refine_2.refine_2.hf
          f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
          ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (HSub.hSub x 1)) (nhdsWithin 1 (Has …
        -/
      · simp_rw [mul_comm (f _), f, mul_sub]
        /-
          case refine_2.refine_2.hf
          f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
          ⊢ Filter.Tendsto (fun x => HSub.hSub (HMul.hMul (HSub.hSub x 1) (riemannZeta x …
        -/
        apply riemannZeta_residue_one.sub
        /-
          case refine_2.refine_2.hf
          f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
          ⊢ Filter.Tendsto (fun x => HMul.hMul (HSub.hSub x 1) (HDiv.hDiv 1 (HSub.hSub x …
        -/
        refine Tendsto.congr' ?_ (tendsto_const_nhds.mono_left nhdsWithin_le_nhds)
        /-
          case refine_2.refine_2.hf
          f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
          ⊢ (nhdsWithin 1 (HasCompl.compl (Singleton.singleton 1))).EventuallyEq (fun x  …
        -/
        filter_upwards [self_mem_nhdsWithin] with x hx
        /-
          case h
          f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
          x : Complex
          hx : Membership.mem (HasCompl.compl (Singleton.singleton 1)) x
          ⊢ Eq 1 (HMul.hMul (HSub.hSub x 1) (HDiv.hDiv 1 (HSub.hSub x 1)))
        -/
        field_simp [sub_ne_zero.mpr <| mem_compl_singleton_iff.mp hx]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.refine_2.hg
          f : Complex → Complex := fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv 1 (HSub …
          ⊢ Filter.Tendsto (fun x => HMul.hMul (f 1) (HSub.hSub x 1)) (nhdsWithin 1 (Has …
        -/
      · exact ((tendsto_id.sub tendsto_const_nhds).mono_left nhdsWithin_le_nhds).const_mul _
        /-
          🎉 no goals
        -/


lemma _root_.isBigO_riemannZeta_sub_one_div {F : Type*} [Norm F] [One F] [NormOneClass F] :
    (fun s : ℂ ↦ riemannZeta s - 1 / (s - 1)) =O[𝓝 1] (fun _ ↦ 1 : ℂ → F) := by
  simpa only [Asymptotics.isBigO_one_nhds_ne_iff] using
     tendsto_riemannZeta_sub_one_div.isBigO_one (F := F)


lemma tendsto_Gamma_term_aux : Tendsto (fun s ↦ 1 / (s - 1) - 1 / Gammaℝ s / (s - 1)) (𝓝[≠] 1)
    (𝓝 (-(γ + Complex.log (4 * ↑π)) / 2)) := by
  /-
    ⊢ Filter.Tendsto (fun s => HSub.hSub (HDiv.hDiv 1 (HSub.hSub s 1)) (HDiv.hDiv  …
  -/
  have h := hasDerivAt_Gammaℝ_one
  /-
    h : HasDerivAt Complex.Gammaℝ (HDiv.hDiv (Neg.neg (HAdd.hAdd (↑Real.eulerMasch …
    ⊢ Filter.Tendsto (fun s => HSub.hSub (HDiv.hDiv 1 (HSub.hSub s 1)) (HDiv.hDiv  …
  -/
  rw [hasDerivAt_iff_tendsto_slope, slope_fun_def_field, Gammaℝ_one] at h
  have := h.div (hasDerivAt_Gammaℝ_one.continuousAt.tendsto.mono_left nhdsWithin_le_nhds)
    (Gammaℝ_one.trans_ne one_ne_zero)
  /-
    h : Filter.Tendsto (fun b => HDiv.hDiv (HSub.hSub b.Gammaℝ 1) (HSub.hSub b 1)) …
    this : Filter.Tendsto (HDiv.hDiv (fun b => HDiv.hDiv (HSub.hSub b.Gammaℝ 1) (H …
    ⊢ Filter.Tendsto (fun s => HSub.hSub (HDiv.hDiv 1 (HSub.hSub s 1)) (HDiv.hDiv  …
  -/
  rw [Gammaℝ_one, div_one] at this
  /-
    h : Filter.Tendsto (fun b => HDiv.hDiv (HSub.hSub b.Gammaℝ 1) (HSub.hSub b 1)) …
    this : Filter.Tendsto (HDiv.hDiv (fun b => HDiv.hDiv (HSub.hSub b.Gammaℝ 1) (H …
    ⊢ Filter.Tendsto (fun s => HSub.hSub (HDiv.hDiv 1 (HSub.hSub s 1)) (HDiv.hDiv  …
  -/
  refine this.congr' ?_
  have : {z | 0 < re z} ∈ 𝓝 (1 : ℂ) := by
    apply (continuous_re.isOpen_preimage _ isOpen_Ioi).mem_nhds
    simp only [mem_preimage, one_re, mem_Ioi, zero_lt_one]
  /-
    h : Filter.Tendsto (fun b => HDiv.hDiv (HSub.hSub b.Gammaℝ 1) (HSub.hSub b 1)) …
    this✝ : Filter.Tendsto (HDiv.hDiv (fun b => HDiv.hDiv (HSub.hSub b.Gammaℝ 1) ( …
    this : Membership.mem (nhds 1) (setOf fun z => LT.lt 0 z.re)
    ⊢ (nhdsWithin 1 (HasCompl.compl (Singleton.singleton 1))).EventuallyEq (HDiv.h …
  -/
  rw [EventuallyEq, eventually_nhdsWithin_iff]
  /-
    h : Filter.Tendsto (fun b => HDiv.hDiv (HSub.hSub b.Gammaℝ 1) (HSub.hSub b 1)) …
    this✝ : Filter.Tendsto (HDiv.hDiv (fun b => HDiv.hDiv (HSub.hSub b.Gammaℝ 1) ( …
    this : Membership.mem (nhds 1) (setOf fun z => LT.lt 0 z.re)
    ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl (Singleton.single …
  -/
  filter_upwards [this] with a ha _
  rw [Pi.div_apply, ← sub_div, div_right_comm, sub_div' _ _ _ (Gammaℝ_ne_zero_of_re_pos ha),
    one_mul]


lemma tendsto_riemannZeta_sub_one_div_Gammaℝ :
    Tendsto (fun s ↦ riemannZeta s - 1 / Gammaℝ s / (s - 1)) (𝓝[≠] 1)
    (𝓝 ((γ - Complex.log (4 * ↑π)) / 2)) := by
  /-
    ⊢ Filter.Tendsto (fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv (HDiv.hDiv 1 s …
  -/
  have := tendsto_riemannZeta_sub_one_div.add tendsto_Gamma_term_aux
  /-
    this : Filter.Tendsto (fun x => HAdd.hAdd (HSub.hSub (riemannZeta x) (HDiv.hDi …
    ⊢ Filter.Tendsto (fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv (HDiv.hDiv 1 s …
  -/
  simp_rw [sub_add_sub_cancel] at this
  /-
    this : Filter.Tendsto (fun x => HSub.hSub (riemannZeta x) (HDiv.hDiv (HDiv.hDi …
    ⊢ Filter.Tendsto (fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv (HDiv.hDiv 1 s …
  -/
  convert this using 2
  /-
    case h.e'_5.h.e'_3
    this : Filter.Tendsto (fun x => HSub.hSub (riemannZeta x) (HDiv.hDiv (HDiv.hDi …
    ⊢ Eq (HDiv.hDiv (HSub.hSub (↑Real.eulerMascheroniConstant) (Complex.log (HMul. …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


/-- Formula for `ζ 1`. Note that mathematically `ζ 1` is undefined, but our construction ascribes
this particular value to it. -/
lemma _root_.riemannZeta_one : riemannZeta 1 = (γ - Complex.log (4 * ↑π)) / 2 := by
  have := (HurwitzZeta.tendsto_hurwitzZetaEven_sub_one_div_nhds_one 0).mono_left
    <| nhdsWithin_le_nhds (s := {1}ᶜ)
  /-
    this : Filter.Tendsto (fun s => HSub.hSub (HurwitzZeta.hurwitzZetaEven 0 s) (H …
    ⊢ Eq (riemannZeta 1) (HDiv.hDiv (HSub.hSub (↑Real.eulerMascheroniConstant) (Co …
  -/
  simp only [HurwitzZeta.hurwitzZetaEven_zero, div_right_comm _ _ (Gammaℝ _)] at this
  /-
    this : Filter.Tendsto (fun s => HSub.hSub (riemannZeta s) (HDiv.hDiv (HDiv.hDi …
    ⊢ Eq (riemannZeta 1) (HDiv.hDiv (HSub.hSub (↑Real.eulerMascheroniConstant) (Co …
  -/
  exact tendsto_nhds_unique this tendsto_riemannZeta_sub_one_div_Gammaℝ
  /-
    🎉 no goals
  -/


/-- Formula for `Λ 1`. Note that mathematically `Λ 1` is undefined, but our construction ascribes
this particular value to it. -/
lemma _root_.completedRiemannZeta_one :
    completedRiemannZeta 1 = (γ - Complex.log (4 * ↑π)) / 2 :=
  (riemannZeta_one ▸ div_one (_ : ℂ) ▸ Gammaℝ_one ▸ riemannZeta_def_of_ne_zero one_ne_zero).symm


/-- Formula for `Λ₀ 1`, where `Λ₀` is the entire function satisfying
`Λ₀ s = π ^ (-s / 2) Γ(s / 2) ζ(s) + 1 / s + 1 / (1 - s)` away from `s = 0, 1`.

Note that `s = 1` is _not_ a pole of `Λ₀`, so this statement (unlike `riemannZeta_one`) is
a mathematically meaningful statement and is not dependent on Mathlib's particular conventions for
division by zero. -/
lemma _root_.completedRiemannZeta₀_one :
    completedRiemannZeta₀ 1 = (γ - Complex.log (4 * ↑π)) / 2 + 1 := by
  /-
    ⊢ Eq (completedRiemannZeta₀ 1) (HAdd.hAdd (HDiv.hDiv (HSub.hSub (↑Real.eulerMa …
  -/
  have := completedRiemannZeta_eq 1
  /-
    this : Eq (completedRiemannZeta 1) (HSub.hSub (HSub.hSub (completedRiemannZeta …
    ⊢ Eq (completedRiemannZeta₀ 1) (HAdd.hAdd (HDiv.hDiv (HSub.hSub (↑Real.eulerMa …
  -/
  rw [sub_self, div_zero, div_one, sub_zero, eq_sub_iff_add_eq] at this
  /-
    this : Eq (HAdd.hAdd (completedRiemannZeta 1) 1) (completedRiemannZeta₀ 1)
    ⊢ Eq (completedRiemannZeta₀ 1) (HAdd.hAdd (HDiv.hDiv (HSub.hSub (↑Real.eulerMa …
  -/
  rw [← this, completedRiemannZeta_one]
  /-
    🎉 no goals
  -/


/-- With Mathlib's particular conventions, we have `ζ 1 ≠ 0`. -/
lemma _root_.riemannZeta_one_ne_zero : riemannZeta 1 ≠ 0 := by
  -- This one's for you, Kevin.
  suffices (γ - (4 * π).log) / 2 ≠ 0 by
    simpa only [riemannZeta_one, ← ofReal_ne_zero, ofReal_log (by positivity : 0 ≤ 4 * π),
      push_cast]
  /-
    ⊢ Ne (HDiv.hDiv (HSub.hSub Real.eulerMascheroniConstant (Real.log (HMul.hMul 4 …
  -/
  refine div_ne_zero (sub_lt_zero.mpr (lt_trans ?_ ?_ (b := 1))).ne two_ne_zero
    /-
      case refine_1
      ⊢ LT.lt Real.eulerMascheroniConstant 1
    -/
  · exact Real.eulerMascheroniConstant_lt_two_thirds.trans (by norm_num)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ⊢ LT.lt 1 (Real.log (HMul.hMul 4 Real.pi))
    -/
  · rw [lt_log_iff_exp_lt (by positivity)]
    exact (lt_trans Real.exp_one_lt_d9 (by norm_num)).trans_le
      <| mul_le_mul_of_nonneg_left two_le_pi (by norm_num)


