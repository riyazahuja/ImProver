lemma isBigO_exp_neg_mul_of_le {c d : ℝ} (hcd : c ≤ d) :
    (fun t ↦ exp (-d * t)) =O[atTop] fun t ↦ exp (-c * t) := by
  /-
    c d : Real
    hcd : LE.le c d
    ⊢ Asymptotics.IsBigO Filter.atTop (fun t => Real.exp (HMul.hMul (Neg.neg d) t) …
  -/
  apply Eventually.isBigO
  /-
    case hfg
    c d : Real
    hcd : LE.le c d
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (Real.exp (HMul.hMul (Neg.neg d …
  -/
  filter_upwards [eventually_gt_atTop 0] with t ht
  /-
    case h
    c d : Real
    hcd : LE.le c d
    t : Real
    ht : LT.lt 0 t
    ⊢ LE.le (Norm.norm (Real.exp (HMul.hMul (Neg.neg d) t))) (Real.exp (HMul.hMul  …
  -/
  rwa [norm_of_nonneg (exp_pos _).le, exp_le_exp, mul_le_mul_right ht, neg_le_neg_iff]
  /-
    🎉 no goals
  -/


private lemma exp_lt_aux {t : ℝ} (ht : 0 < t) : rexp (-π * t) < 1 := by
  /-
    t : Real
    ht : LT.lt 0 t
    ⊢ LT.lt (Real.exp (HMul.hMul (Neg.neg Real.pi) t)) 1
  -/
  simpa only [exp_lt_one_iff, neg_mul, neg_lt_zero] using mul_pos pi_pos ht
  /-
    🎉 no goals
  -/


private lemma isBigO_one_aux :
    IsBigO atTop (fun t : ℝ ↦ (1 - rexp (-π * t))⁻¹) (fun _ ↦ (1 : ℝ)) := by
  /-
    ⊢ Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HSub.hSub 1 (Real.exp (HM …
  -/
  refine ((Tendsto.const_sub _ ?_).inv₀ (by norm_num)).isBigO_one ℝ (c := ((1 - 0)⁻¹ : ℝ))
  simpa only [neg_mul, tendsto_exp_comp_nhds_zero, tendsto_neg_atBot_iff]
    using tendsto_id.const_mul_atTop pi_pos


/-- Summand in the sum to be bounded (`ℕ` version). -/
def f_nat (k : ℕ) (a t : ℝ) (n : ℕ) : ℝ := (n + a) ^ k * exp (-π * (n + a) ^ 2 * t)


/-- An upper bound for the summand when `0 ≤ a`. -/
def g_nat (k : ℕ) (a t : ℝ) (n : ℕ) : ℝ := (n + a) ^ k * exp (-π * (n + a ^ 2) * t)


lemma f_le_g_nat (k : ℕ) {a t : ℝ} (ha : 0 ≤ a) (ht : 0 < t) (n : ℕ) :
    ‖f_nat k a t n‖ ≤ g_nat k a t n := by
  /-
    k : Nat
    a t : Real
    ha : LE.le 0 a
    ht : LT.lt 0 t
    n : Nat
    ⊢ LE.le (Norm.norm (HurwitzKernelBounds.f_nat k a t n)) (HurwitzKernelBounds.g …
  -/
  rw [f_nat, norm_of_nonneg (by positivity)]
  /-
    k : Nat
    a t : Real
    ha : LE.le 0 a
    ht : LT.lt 0 t
    n : Nat
    ⊢ LE.le (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) a) k) (Real.exp (HMul.hMul (HMul …
  -/
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  rw [Real.exp_le_exp, mul_le_mul_right ht,
    mul_le_mul_left_of_neg (neg_lt_zero.mpr pi_pos), ← sub_nonneg]
  have u : (n : ℝ) ≤ (n : ℝ) ^ 2 := by
    simpa only [← Nat.cast_pow, Nat.cast_le] using Nat.le_self_pow two_ne_zero _
  /-
    k : Nat
    a t : Real
    ha : LE.le 0 a
    ht : LT.lt 0 t
    n : Nat
    u : LE.le (↑n) (HPow.hPow (↑n) 2)
    ⊢ LE.le 0 (HSub.hSub (HPow.hPow (HAdd.hAdd (↑n) a) 2) (HAdd.hAdd (↑n) (HPow.hP …
  -/
  convert add_nonneg (sub_nonneg.mpr u) (by positivity : 0 ≤ 2 * n * a) using 1
  /-
    case h.e'_4
    k : Nat
    a t : Real
    ha : LE.le 0 a
    ht : LT.lt 0 t
    n : Nat
    u : LE.le (↑n) (HPow.hPow (↑n) 2)
    ⊢ Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (↑n) a) 2) (HAdd.hAdd (↑n) (HPow.hPow a  …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The sum to be bounded (`ℕ` version). -/
def F_nat (k : ℕ) (a t : ℝ) : ℝ := ∑' n, f_nat k a t n


lemma summable_f_nat (k : ℕ) (a : ℝ) {t : ℝ} (ht : 0 < t) : Summable (f_nat k a t) := by
  have : Summable fun n : ℕ ↦ n ^ k * exp (-π * (n + a) ^ 2 * t) := by
    refine (((summable_pow_mul_jacobiTheta₂_term_bound (|a| * t) ht k).mul_right
      (rexp (-π * a ^ 2 * t))).comp_injective Nat.cast_injective).of_norm_bounded _ (fun n ↦ ?_)
    simp_rw [mul_assoc, Function.comp_apply, ← Real.exp_add, norm_mul, norm_pow, Int.cast_abs,
      Int.cast_natCast, norm_eq_abs, Nat.abs_cast, abs_exp]
    refine mul_le_mul_of_nonneg_left ?_ (pow_nonneg (Nat.cast_nonneg _) _)
    rw [exp_le_exp, ← sub_nonneg]
    rw [show -π * (t * n ^ 2 - 2 * (|a| * (t * n))) + -π * (a ^ 2 * t) - -π * ((n + a) ^ 2 * t)
         = π * t * n * (|a| + a) * 2 by ring]
    refine mul_nonneg (mul_nonneg (by positivity) ?_) two_pos.le
    rw [← neg_le_iff_add_nonneg]
    apply neg_le_abs
  /-
    k : Nat
    a t : Real
    ht : LT.lt 0 t
    this : Summable fun n => HMul.hMul (HPow.hPow (↑n) k) (Real.exp (HMul.hMul (HM …
    ⊢ Summable (HurwitzKernelBounds.f_nat k a t)
  -/
  apply (this.mul_left (2 ^ k)).of_norm_bounded_eventually_nat
  simp_rw [← mul_assoc, f_nat, norm_mul, norm_eq_abs, abs_exp,
    mul_le_mul_iff_of_pos_right (exp_pos _), ← mul_pow, abs_pow, two_mul]
  /-
    k : Nat
    a t : Real
    ht : LT.lt 0 t
    this : Summable fun n => HMul.hMul (HPow.hPow (↑n) k) (Real.exp (HMul.hMul (HM …
    ⊢ Filter.Eventually (fun i => LE.le (HPow.hPow (abs (HAdd.hAdd (↑i) a)) k) (HP …
  -/
  filter_upwards [eventually_ge_atTop (Nat.ceil |a|)] with n hn
  /-
    case h
    k : Nat
    a t : Real
    ht : LT.lt 0 t
    this : Summable fun n => HMul.hMul (HPow.hPow (↑n) k) (Real.exp (HMul.hMul (HM …
    n : Nat
    hn : LE.le (Nat.ceil (abs a)) n
    ⊢ LE.le (HPow.hPow (abs (HAdd.hAdd (↑n) a)) k) (HPow.hPow (HAdd.hAdd ↑n ↑n) k)
  -/
  gcongr
  /-
    case h.hab
    k : Nat
    a t : Real
    ht : LT.lt 0 t
    this : Summable fun n => HMul.hMul (HPow.hPow (↑n) k) (Real.exp (HMul.hMul (HM …
    n : Nat
    hn : LE.le (Nat.ceil (abs a)) n
    ⊢ LE.le (abs (HAdd.hAdd (↑n) a)) (HAdd.hAdd ↑n ↑n)
  -/
  exact (abs_add_le ..).trans (add_le_add (Nat.abs_cast _).le (Nat.ceil_le.mp hn))
  /-
    🎉 no goals
  -/


lemma F_nat_zero_le {a : ℝ} (ha : 0 ≤ a) {t : ℝ} (ht : 0 < t) :
    ‖F_nat 0 a t‖ ≤ rexp (-π * a ^ 2 * t) / (1 - rexp (-π * t)) := by
  /-
    a : Real
    ha : LE.le 0 a
    t : Real
    ht : LT.lt 0 t
    ⊢ LE.le (Norm.norm (HurwitzKernelBounds.F_nat 0 a t)) (HDiv.hDiv (Real.exp (HM …
  -/
  refine tsum_of_norm_bounded ?_ (f_le_g_nat 0 ha ht)
  /-
    a : Real
    ha : LE.le 0 a
    t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (HurwitzKernelBounds.g_nat 0 a t) (HDiv.hDiv (Real.exp (HMul.hMul (HM …
  -/
  convert (hasSum_geometric_of_lt_one (exp_pos _).le <| exp_lt_aux ht).mul_left _ using 1
  /-
    case h.e'_5
    a : Real
    ha : LE.le 0 a
    t : Real
    ht : LT.lt 0 t
    ⊢ Eq (HurwitzKernelBounds.g_nat 0 a t) fun i => HMul.hMul (Real.exp (HMul.hMul …
  -/
  ext1 n
  /-
    case h.e'_5.h
    a : Real
    ha : LE.le 0 a
    t : Real
    ht : LT.lt 0 t
    n : Nat
    ⊢ Eq (HurwitzKernelBounds.g_nat 0 a t n) (HMul.hMul (Real.exp (HMul.hMul (HMul …
  -/
  simp only [g_nat]
  /-
    case h.e'_5.h
    a : Real
    ha : LE.le 0 a
    t : Real
    ht : LT.lt 0 t
    n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) a) 0) (Real.exp (HMul.hMul (HMul.hM …
  -/
  rw [← Real.exp_nat_mul, ← Real.exp_add]
  /-
    case h.e'_5.h
    a : Real
    ha : LE.le 0 a
    t : Real
    ht : LT.lt 0 t
    n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) a) 0) (Real.exp (HMul.hMul (HMul.hM …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


lemma F_nat_zero_zero_sub_le {t : ℝ} (ht : 0 < t) :
    ‖F_nat 0 0 t - 1‖ ≤ rexp (-π * t) / (1 - rexp (-π * t)) := by
  /-
    t : Real
    ht : LT.lt 0 t
    ⊢ LE.le (Norm.norm (HSub.hSub (HurwitzKernelBounds.F_nat 0 0 t) 1)) (HDiv.hDiv …
  -/
  convert F_nat_zero_le zero_le_one ht using 2
  · rw [F_nat, tsum_eq_zero_add (summable_f_nat 0 0 ht), f_nat, Nat.cast_zero, add_zero, pow_zero,
      one_mul, pow_two, mul_zero, mul_zero, zero_mul, exp_zero, add_comm, add_sub_cancel_right]
    /-
      case h.e'_3.h.e'_3
      t : Real
      ht : LT.lt 0 t
      ⊢ Eq (tsum fun b => HurwitzKernelBounds.f_nat 0 0 t (HAdd.hAdd b 1)) (HurwitzK …
    -/
    simp_rw [F_nat, f_nat, Nat.cast_add, Nat.cast_one, add_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.h.e'_5
      t : Real
      ht : LT.lt 0 t
      ⊢ Eq (Real.exp (HMul.hMul (Neg.neg Real.pi) t)) (Real.exp (HMul.hMul (HMul.hMu …
    -/
  · rw [one_pow, mul_one]
    /-
      🎉 no goals
    -/


lemma isBigO_atTop_F_nat_zero_sub {a : ℝ} (ha : 0 ≤ a) : ∃ p, 0 < p ∧
    (fun t ↦ F_nat 0 a t - (if a = 0 then 1 else 0)) =O[atTop] fun t ↦ exp (-p * t) := by
  /-
    a : Real
    ha : LE.le 0 a
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
  -/
  split_ifs with h
    /-
      case pos
      a : Real
      ha : LE.le 0 a
      h : Eq a 0
      ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
    -/
  · rw [h]
    have : (fun t ↦ F_nat 0 0 t - 1) =O[atTop] fun t ↦ rexp (-π * t) / (1 - rexp (-π * t)) := by
      apply Eventually.isBigO
      filter_upwards [eventually_gt_atTop 0] with t ht
      exact F_nat_zero_zero_sub_le ht
    /-
      case pos
      a : Real
      ha : LE.le 0 a
      h : Eq a 0
      this : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBound …
      ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
    -/
    refine ⟨_, pi_pos, this.trans ?_⟩
    /-
      case pos
      a : Real
      ha : LE.le 0 a
      h : Eq a 0
      this : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBound …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun t => HDiv.hDiv (Real.exp (HMul.hMul (Ne …
    -/
    simpa using (isBigO_refl (fun t ↦ rexp (-π * t)) _).mul isBigO_one_aux
    /-
      🎉 no goals
    -/
    /-
      case neg
      a : Real
      ha : LE.le 0 a
      h : Not (Eq a 0)
      ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
    -/
  · simp_rw [sub_zero]
    have : (fun t ↦ F_nat 0 a t) =O[atTop] fun t ↦ rexp (-π * a ^ 2 * t) / (1 - rexp (-π * t)) := by
      apply Eventually.isBigO
      filter_upwards [eventually_gt_atTop 0] with t ht
      exact F_nat_zero_le ha ht
    /-
      case neg
      a : Real
      ha : LE.le 0 a
      h : Not (Eq a 0)
      this : Asymptotics.IsBigO Filter.atTop (fun t => HurwitzKernelBounds.F_nat 0 a …
      ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
    -/
    refine ⟨π * a ^ 2, mul_pos pi_pos (sq_pos_of_ne_zero h), this.trans ?_⟩
    /-
      case neg
      a : Real
      ha : LE.le 0 a
      h : Not (Eq a 0)
      this : Asymptotics.IsBigO Filter.atTop (fun t => HurwitzKernelBounds.F_nat 0 a …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun t => HDiv.hDiv (Real.exp (HMul.hMul (HM …
    -/
    simpa only [neg_mul π (a ^ 2), mul_one] using (isBigO_refl _ _).mul isBigO_one_aux
    /-
      🎉 no goals
    -/


lemma F_nat_one_le {a : ℝ} (ha : 0 ≤ a) {t : ℝ} (ht : 0 < t) :
    ‖F_nat 1 a t‖ ≤ rexp (-π * (a ^ 2 + 1) * t) / (1 - rexp (-π * t)) ^ 2
      + a * rexp (-π * a ^ 2 * t) / (1 - rexp (-π * t)) := by
  /-
    a : Real
    ha : LE.le 0 a
    t : Real
    ht : LT.lt 0 t
    ⊢ LE.le (Norm.norm (HurwitzKernelBounds.F_nat 1 a t)) (HAdd.hAdd (HDiv.hDiv (R …
  -/
  refine tsum_of_norm_bounded ?_ (f_le_g_nat 1 ha ht)
  /-
    a : Real
    ha : LE.le 0 a
    t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (HurwitzKernelBounds.g_nat 1 a t) (HAdd.hAdd (HDiv.hDiv (Real.exp (HM …
  -/
  unfold g_nat
  /-
    a : Real
    ha : LE.le 0 a
    t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (fun n => HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) a) 1) (Real.exp (HMul. …
  -/
  simp_rw [pow_one, add_mul]
  /-
    a : Real
    ha : LE.le 0 a
    t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (fun n => HAdd.hAdd (HMul.hMul (↑n) (Real.exp (HMul.hMul (HMul.hMul ( …
  -/
  apply HasSum.add
  · have h0' : ‖rexp (-π * t)‖ < 1 := by
      simpa only [norm_eq_abs, abs_exp] using exp_lt_aux ht
    /-
      case hf
      a : Real
      ha : LE.le 0 a
      t : Real
      ht : LT.lt 0 t
      h0' : LT.lt (Norm.norm (Real.exp (HMul.hMul (Neg.neg Real.pi) t))) 1
      ⊢ HasSum (fun b => HMul.hMul (↑b) (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Rea …
    -/
    convert (hasSum_coe_mul_geometric_of_norm_lt_one h0').mul_left (exp (-π * a ^ 2 * t)) using 1
      /-
        case h.e'_5
        a : Real
        ha : LE.le 0 a
        t : Real
        ht : LT.lt 0 t
        h0' : LT.lt (Norm.norm (Real.exp (HMul.hMul (Neg.neg Real.pi) t))) 1
        ⊢ Eq (fun b => HMul.hMul (↑b) (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi …
      -/
    · ext1 n
      /-
        case h.e'_5.h
        a : Real
        ha : LE.le 0 a
        t : Real
        ht : LT.lt 0 t
        h0' : LT.lt (Norm.norm (Real.exp (HMul.hMul (Neg.neg Real.pi) t))) 1
        n : Nat
        ⊢ Eq (HMul.hMul (↑n) (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HAdd.h …
      -/
      rw [mul_comm (exp _), ← Real.exp_nat_mul, mul_assoc (n : ℝ), ← Real.exp_add]
      /-
        case h.e'_5.h
        a : Real
        ha : LE.le 0 a
        t : Real
        ht : LT.lt 0 t
        h0' : LT.lt (Norm.norm (Real.exp (HMul.hMul (Neg.neg Real.pi) t))) 1
        n : Nat
        ⊢ Eq (HMul.hMul (↑n) (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HAdd.h …
      -/
      ring_nf
      /-
        🎉 no goals
      -/
      /-
        case h.e'_6
        a : Real
        ha : LE.le 0 a
        t : Real
        ht : LT.lt 0 t
        h0' : LT.lt (Norm.norm (Real.exp (HMul.hMul (Neg.neg Real.pi) t))) 1
        ⊢ Eq (HDiv.hDiv (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HAdd.hAdd ( …
      -/
    · rw [mul_add, add_mul, mul_one, exp_add, mul_div_assoc]
      /-
        🎉 no goals
      -/
    /-
      case hg
      a : Real
      ha : LE.le 0 a
      t : Real
      ht : LT.lt 0 t
      ⊢ HasSum (fun b => HMul.hMul a (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.p …
    -/
  · convert (hasSum_geometric_of_lt_one (exp_pos _).le <| exp_lt_aux ht).mul_left _ using 1
    /-
      case h.e'_5
      a : Real
      ha : LE.le 0 a
      t : Real
      ht : LT.lt 0 t
      ⊢ Eq (fun b => HMul.hMul a (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) ( …
    -/
    ext1 n
    /-
      case h.e'_5.h
      a : Real
      ha : LE.le 0 a
      t : Real
      ht : LT.lt 0 t
      n : Nat
      ⊢ Eq (HMul.hMul a (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HAdd.hAdd …
    -/
    rw [← Real.exp_nat_mul, mul_assoc _ (exp _), ← Real.exp_add]
    /-
      case h.e'_5.h
      a : Real
      ha : LE.le 0 a
      t : Real
      ht : LT.lt 0 t
      n : Nat
      ⊢ Eq (HMul.hMul a (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HAdd.hAdd …
    -/
    ring_nf
    /-
      🎉 no goals
    -/


lemma isBigO_atTop_F_nat_one {a : ℝ} (ha : 0 ≤ a) : ∃ p, 0 < p ∧
    F_nat 1 a =O[atTop] fun t ↦ exp (-p * t) := by
  suffices ∃ p, 0 < p ∧ (fun t ↦ rexp (-π * (a ^ 2 + 1) * t) / (1 - rexp (-π * t)) ^ 2
      + a * rexp (-π * a ^ 2 * t) / (1 - rexp (-π * t))) =O[atTop] fun t ↦ exp (-p * t) by
    let ⟨p, hp, hp'⟩ := this
    refine ⟨p, hp, (Eventually.isBigO ?_).trans hp'⟩
    filter_upwards [eventually_gt_atTop 0] with t ht
    exact F_nat_one_le ha ht
  have aux' : IsBigO atTop (fun t : ℝ ↦ ((1 - rexp (-π * t)) ^ 2)⁻¹) (fun _ ↦ (1 : ℝ)) := by
    simpa only [inv_pow, one_pow] using isBigO_one_aux.pow 2
  /-
    a : Real
    ha : LE.le 0 a
    aux' : Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HPow.hPow (HSub.hSub …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
  -/
  rcases eq_or_lt_of_le ha with rfl | ha'
  · exact ⟨_, pi_pos, by simpa only [zero_pow two_ne_zero, zero_add, mul_one, zero_mul, zero_div,
      add_zero] using (isBigO_refl _ _).mul aux'⟩
    /-
      case inr
      a : Real
      ha : LE.le 0 a
      aux' : Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HPow.hPow (HSub.hSub …
      ha' : LT.lt 0 a
      ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
    -/
  · refine ⟨π * a ^ 2, mul_pos pi_pos <| pow_pos ha' _, IsBigO.add ?_ ?_⟩
      /-
        case inr.refine_1
        a : Real
        ha : LE.le 0 a
        aux' : Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HPow.hPow (HSub.hSub …
        ha' : LT.lt 0 a
        ⊢ Asymptotics.IsBigO Filter.atTop (fun t => HDiv.hDiv (Real.exp (HMul.hMul (HM …
      -/
    · conv_rhs => enter [t]; rw [← mul_one (rexp _)]
      /-
        case inr.refine_1
        a : Real
        ha : LE.le 0 a
        aux' : Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HPow.hPow (HSub.hSub …
        ha' : LT.lt 0 a
        ⊢ Asymptotics.IsBigO Filter.atTop (fun t => HDiv.hDiv (Real.exp (HMul.hMul (HM …
      -/
      refine (Eventually.isBigO ?_).mul aux'
      /-
        case inr.refine_1
        a : Real
        ha : LE.le 0 a
        aux' : Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HPow.hPow (HSub.hSub …
        ha' : LT.lt 0 a
        ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (Real.exp (HMul.hMul (HMul.hMul …
      -/
      filter_upwards [eventually_gt_atTop 0] with t ht
      /-
        case h
        a : Real
        ha : LE.le 0 a
        aux' : Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HPow.hPow (HSub.hSub …
        ha' : LT.lt 0 a
        t : Real
        ht : LT.lt 0 t
        ⊢ LE.le (Norm.norm (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HAdd.hAd …
      -/
      rw [norm_of_nonneg (exp_pos _).le, exp_le_exp]
      /-
        case h
        a : Real
        ha : LE.le 0 a
        aux' : Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HPow.hPow (HSub.hSub …
        ha' : LT.lt 0 a
        t : Real
        ht : LT.lt 0 t
        ⊢ LE.le (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HAdd.hAdd (HPow.hPow a 2) 1)) …
      -/
      nlinarith [pi_pos]
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_2
        a : Real
        ha : LE.le 0 a
        aux' : Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HPow.hPow (HSub.hSub …
        ha' : LT.lt 0 a
        ⊢ Asymptotics.IsBigO Filter.atTop (fun t => HDiv.hDiv (HMul.hMul a (Real.exp ( …
      -/
    · simp_rw [mul_div_assoc, ← neg_mul]
      /-
        case inr.refine_2
        a : Real
        ha : LE.le 0 a
        aux' : Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HPow.hPow (HSub.hSub …
        ha' : LT.lt 0 a
        ⊢ Asymptotics.IsBigO Filter.atTop (fun t => HMul.hMul a (HDiv.hDiv (Real.exp ( …
      -/
      apply IsBigO.const_mul_left
      /-
        case inr.refine_2.h
        a : Real
        ha : LE.le 0 a
        aux' : Asymptotics.IsBigO Filter.atTop (fun t => Inv.inv (HPow.hPow (HSub.hSub …
        ha' : LT.lt 0 a
        ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HDiv.hDiv (Real.exp (HMul.hMul (HM …
      -/
      simpa only [mul_one] using (isBigO_refl _ _).mul isBigO_one_aux
      /-
        🎉 no goals
      -/


/-- Summand in the sum to be bounded (`ℤ` version). -/
def f_int (k : ℕ) (a t : ℝ) (n : ℤ) : ℝ := |n + a| ^ k * exp (-π * (n + a) ^ 2 * t)


lemma f_int_ofNat (k : ℕ) {a : ℝ} (ha : 0 ≤ a) (t : ℝ) (n : ℕ) :
    f_int k a t (Int.ofNat n) = f_nat k a t n := by
  /-
    k : Nat
    a : Real
    ha : LE.le 0 a
    t : Real
    n : Nat
    ⊢ Eq (HurwitzKernelBounds.f_int k a t (Int.ofNat n)) (HurwitzKernelBounds.f_na …
  -/
  rw [f_int, f_nat, Int.ofNat_eq_coe, Int.cast_natCast, abs_of_nonneg (by positivity)]
  /-
    🎉 no goals
  -/


lemma f_int_negSucc (k : ℕ) {a : ℝ} (ha : a ≤ 1) (t : ℝ) (n : ℕ) :
    f_int k a t (Int.negSucc n) = f_nat k (1 - a) t n := by
  /-
    k : Nat
    a : Real
    ha : LE.le a 1
    t : Real
    n : Nat
    ⊢ Eq (HurwitzKernelBounds.f_int k a t (Int.negSucc n)) (HurwitzKernelBounds.f_ …
  -/
  have : (Int.negSucc n) + a = -(n + (1 - a)) := by { push_cast; ring }
  /-
    k : Nat
    a : Real
    ha : LE.le a 1
    t : Real
    n : Nat
    this : Eq (HAdd.hAdd (↑(Int.negSucc n)) a) (Neg.neg (HAdd.hAdd (↑n) (HSub.hSub …
    ⊢ Eq (HurwitzKernelBounds.f_int k a t (Int.negSucc n)) (HurwitzKernelBounds.f_ …
  -/
  rw [f_int, f_nat, this, abs_neg, neg_sq, abs_of_nonneg (by linarith)]
  /-
    🎉 no goals
  -/


lemma summable_f_int (k : ℕ) (a : ℝ) {t : ℝ} (ht : 0 < t) : Summable (f_int k a t) := by
  /-
    k : Nat
    a t : Real
    ht : LT.lt 0 t
    ⊢ Summable (HurwitzKernelBounds.f_int k a t)
  -/
  apply Summable.of_norm
  suffices ∀ n, ‖f_int k a t n‖ = ‖(Int.rec (f_nat k a t) (f_nat k (1 - a) t) : ℤ → ℝ) n‖ from
    funext this ▸ (HasSum.int_rec (summable_f_nat k a ht).hasSum
      (summable_f_nat k (1 - a) ht).hasSum).summable.norm
  /-
    case hf
    k : Nat
    a t : Real
    ht : LT.lt 0 t
    ⊢ ∀ (n : Int), Eq (Norm.norm (HurwitzKernelBounds.f_int k a t n)) (Norm.norm ( …
  -/
  intro n
  /-
    case hf
    k : Nat
    a t : Real
    ht : LT.lt 0 t
    n : Int
    ⊢ Eq (Norm.norm (HurwitzKernelBounds.f_int k a t n)) (Norm.norm ((fun t_1 => I …
  -/
  cases' n with m m
  · simp only [f_int, f_nat, Int.ofNat_eq_coe, Int.cast_natCast, norm_mul, norm_eq_abs, abs_pow,
      abs_abs]
  · simp only [f_int, f_nat, Int.cast_negSucc, norm_mul, norm_eq_abs, abs_pow, abs_abs,
      (by { push_cast; ring } : -↑(m + 1) + a = -(m + (1 - a))), abs_neg, neg_sq]


/-- The sum to be bounded (`ℤ` version). -/
def F_int (k : ℕ) (a : UnitAddCircle) (t : ℝ) : ℝ :=
  (show Function.Periodic (fun b ↦ ∑' (n : ℤ), f_int k b t n) 1 by
    /-
      k : Nat
      a : UnitAddCircle
      t : Real
      ⊢ Function.Periodic (fun b => tsum fun n => HurwitzKernelBounds.f_int k b t n) 1
    -/
    intro b
    /-
      k : Nat
      a : UnitAddCircle
      t b : Real
      ⊢ Eq ((fun b => tsum fun n => HurwitzKernelBounds.f_int k b t n) (HAdd.hAdd b  …
    -/
    simp_rw [← (Equiv.addRight (1 : ℤ)).tsum_eq (f := fun n ↦ f_int k b t n)]
    /-
      k : Nat
      a : UnitAddCircle
      t b : Real
      ⊢ Eq (tsum fun n => HurwitzKernelBounds.f_int k (HAdd.hAdd b 1) t n) (tsum fun …
    -/
    simp only [f_int, ← add_assoc, add_comm, Equiv.coe_addRight, Int.cast_add, Int.cast_one]
    /-
      🎉 no goals
    -/
    ).lift a


lemma F_int_eq_of_mem_Icc (k : ℕ) {a : ℝ} (ha : a ∈ Icc 0 1) {t : ℝ} (ht : 0 < t) :
    F_int k a t = (F_nat k a t) + (F_nat k (1 - a) t) := by
  /-
    k : Nat
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    t : Real
    ht : LT.lt 0 t
    ⊢ Eq (HurwitzKernelBounds.F_int k (↑a) t) (HAdd.hAdd (HurwitzKernelBounds.F_na …
  -/
  simp only [F_int, F_nat, Function.Periodic.lift_coe]
  convert ((summable_f_nat k a ht).hasSum.int_rec (summable_f_nat k (1 - a) ht).hasSum).tsum_eq
    using 3 with n
  /-
    case h.e'_2.h.e'_5.h
    k : Nat
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    t : Real
    ht : LT.lt 0 t
    n : Int
    ⊢ Eq (HurwitzKernelBounds.f_int k a t n) (Int.rec (HurwitzKernelBounds.f_nat k …
  -/
  cases' n with m m
    /-
      case h.e'_2.h.e'_5.h.ofNat
      k : Nat
      a : Real
      ha : Membership.mem (Set.Icc 0 1) a
      t : Real
      ht : LT.lt 0 t
      m : Nat
      ⊢ Eq (HurwitzKernelBounds.f_int k a t (Int.ofNat m)) (Int.rec (HurwitzKernelBo …
    -/
  · rw [f_int_ofNat _ ha.1]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_5.h.negSucc
      k : Nat
      a : Real
      ha : Membership.mem (Set.Icc 0 1) a
      t : Real
      ht : LT.lt 0 t
      m : Nat
      ⊢ Eq (HurwitzKernelBounds.f_int k a t (Int.negSucc m)) (Int.rec (HurwitzKernel …
    -/
  · rw [f_int_negSucc _ ha.2]
    /-
      🎉 no goals
    -/


lemma isBigO_atTop_F_int_zero_sub (a : UnitAddCircle) : ∃ p, 0 < p ∧
    (fun t ↦ F_int 0 a t - (if a = 0 then 1 else 0)) =O[atTop] fun t ↦ exp (-p * t) := by
  /-
    a : UnitAddCircle
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
  -/
  obtain ⟨a, ha, rfl⟩ := a.eq_coe_Ico
  /-
    case intro.intro
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
  -/
  obtain ⟨p, hp, hp'⟩ := isBigO_atTop_F_nat_zero_sub ha.1
  /-
    case intro.intro.intro.intro
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
  -/
  obtain ⟨q, hq, hq'⟩ := isBigO_atTop_F_nat_zero_sub (sub_nonneg.mpr ha.2.le)
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    q : Real
    hq : LT.lt 0 q
    hq' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
  -/
  simp_rw [AddCircle.coe_eq_zero_iff_of_mem_Ico ha]
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    q : Real
    hq : LT.lt 0 q
    hq' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
  -/
  simp_rw [eq_false_intro (by linarith [ha.2] : 1 - a ≠ 0), if_false, sub_zero] at hq'
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    q : Real
    hq : LT.lt 0 q
    hq' : Asymptotics.IsBigO Filter.atTop (fun t => HurwitzKernelBounds.F_nat 0 (H …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun t => H …
  -/
  refine ⟨_, lt_min hp hq, ?_⟩
  have : (fun t ↦ F_int 0 a t - (if a = 0 then 1 else 0)) =ᶠ[atTop]
      fun t ↦ (F_nat 0 a t - (if a = 0 then 1 else 0)) + F_nat 0 (1 - a) t := by
    filter_upwards [eventually_gt_atTop 0] with t ht
    rw [F_int_eq_of_mem_Icc 0 (Ico_subset_Icc_self ha) ht]
    ring
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    q : Real
    hq : LT.lt 0 q
    hq' : Asymptotics.IsBigO Filter.atTop (fun t => HurwitzKernelBounds.F_nat 0 (H …
    this : Filter.atTop.EventuallyEq (fun t => HSub.hSub (HurwitzKernelBounds.F_in …
    ⊢ Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds.F_i …
  -/
  refine this.isBigO.trans ((hp'.trans ?_).add (hq'.trans ?_)) <;>
  /-
    case intro.intro.intro.intro.intro.intro.refine_1
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    q : Real
    hq : LT.lt 0 q
    hq' : Asymptotics.IsBigO Filter.atTop (fun t => HurwitzKernelBounds.F_nat 0 (H …
    this : Filter.atTop.EventuallyEq (fun t => HSub.hSub (HurwitzKernelBounds.F_in …
    ⊢ Asymptotics.IsBigO Filter.atTop (fun t => Real.exp (HMul.hMul (Neg.neg p) t) …
  -/
  apply isBigO_exp_neg_mul_of_le
  /-
    case intro.intro.intro.intro.intro.intro.refine_1.hcd
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    q : Real
    hq : LT.lt 0 q
    hq' : Asymptotics.IsBigO Filter.atTop (fun t => HurwitzKernelBounds.F_nat 0 (H …
    this : Filter.atTop.EventuallyEq (fun t => HSub.hSub (HurwitzKernelBounds.F_in …
    ⊢ LE.le (Min.min p q) p
  -/
  exacts [min_le_left .., min_le_right ..]
  /-
    🎉 no goals
  -/


lemma isBigO_atTop_F_int_one (a : UnitAddCircle) : ∃ p, 0 < p ∧
    F_int 1 a =O[atTop] fun t ↦ exp (-p * t) := by
  /-
    a : UnitAddCircle
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (HurwitzKer …
  -/
  obtain ⟨a, ha, rfl⟩ := a.eq_coe_Ico
  /-
    case intro.intro
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (HurwitzKer …
  -/
  obtain ⟨p, hp, hp'⟩ := isBigO_atTop_F_nat_one ha.1
  /-
    case intro.intro.intro.intro
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 a) fun t => …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (HurwitzKer …
  -/
  obtain ⟨q, hq, hq'⟩ := isBigO_atTop_F_nat_one (sub_nonneg.mpr ha.2.le)
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 a) fun t => …
    q : Real
    hq : LT.lt 0 q
    hq' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 (HSub.hSub  …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (HurwitzKer …
  -/
  refine ⟨_, lt_min hp hq, ?_⟩
  have : F_int 1 a =ᶠ[atTop] fun t ↦ F_nat 1 a t + F_nat 1 (1 - a) t := by
    filter_upwards [eventually_gt_atTop 0] with t ht
    exact F_int_eq_of_mem_Icc 1 (Ico_subset_Icc_self ha) ht
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 a) fun t => …
    q : Real
    hq : LT.lt 0 q
    hq' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 (HSub.hSub  …
    this : Filter.atTop.EventuallyEq (HurwitzKernelBounds.F_int 1 ↑a) fun t => HAd …
    ⊢ Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_int 1 ↑a) fun t => Re …
  -/
  refine this.isBigO.trans ((hp'.trans ?_).add (hq'.trans ?_)) <;>
  /-
    case intro.intro.intro.intro.intro.intro.refine_1
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 a) fun t => …
    q : Real
    hq : LT.lt 0 q
    hq' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 (HSub.hSub  …
    this : Filter.atTop.EventuallyEq (HurwitzKernelBounds.F_int 1 ↑a) fun t => HAd …
    ⊢ Asymptotics.IsBigO Filter.atTop (fun t => Real.exp (HMul.hMul (Neg.neg p) t) …
  -/
  apply isBigO_exp_neg_mul_of_le
  /-
    case intro.intro.intro.intro.intro.intro.refine_1.hcd
    a : Real
    ha : Membership.mem (Set.Ico 0 1) a
    p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 a) fun t => …
    q : Real
    hq : LT.lt 0 q
    hq' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 (HSub.hSub  …
    this : Filter.atTop.EventuallyEq (HurwitzKernelBounds.F_int 1 ↑a) fun t => HAd …
    ⊢ LE.le (Min.min p q) p
  -/
  exacts [min_le_left .., min_le_right ..]
  /-
    🎉 no goals
  -/


