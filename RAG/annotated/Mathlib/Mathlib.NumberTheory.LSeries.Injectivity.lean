/-- The abscissa of absolute convergence of `f + g` is at most the maximum of those
of `f` and `g`. -/
lemma LSeries.abscissaOfAbsConv_add_le (f g : ℕ → ℂ) :
    abscissaOfAbsConv (f + g) ≤ max (abscissaOfAbsConv f) (abscissaOfAbsConv g) :=
  abscissaOfAbsConv_binop_le LSeriesSummable.add f g


/-- The abscissa of absolute convergence of `f - g` is at most the maximum of those
of `f` and `g`. -/
lemma LSeries.abscissaOfAbsConv_sub_le (f g : ℕ → ℂ) :
    abscissaOfAbsConv (f - g) ≤ max (abscissaOfAbsConv f) (abscissaOfAbsConv g) :=
  abscissaOfAbsConv_binop_le LSeriesSummable.sub f g


private
lemma cpow_mul_div_cpow_eq_div_div_cpow (m n : ℕ) (z : ℂ) (x : ℝ) :
    (n + 1) ^ (x : ℂ) * (z / m ^ (x : ℂ)) = z / (m / (n + 1)) ^ (x : ℂ) := by
  /-
    m n : Nat
    z : Complex
    x : Real
    ⊢ Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x) (HDiv.hDiv z (HPow.hPow ↑m ↑ …
  -/
  have Hn : (0 : ℝ) ≤ (n + 1 : ℝ)⁻¹ := by positivity
  /-
    m n : Nat
    z : Complex
    x : Real
    Hn : LE.le 0 (Inv.inv (HAdd.hAdd (↑n) 1))
    ⊢ Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x) (HDiv.hDiv z (HPow.hPow ↑m ↑ …
  -/
  rw [← mul_div_assoc, mul_comm, div_eq_mul_inv z, mul_div_assoc]
  /-
    m n : Nat
    z : Complex
    x : Real
    Hn : LE.le 0 (Inv.inv (HAdd.hAdd (↑n) 1))
    ⊢ Eq (HMul.hMul z (HDiv.hDiv (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x) (HPow.hPow ↑m ↑ …
  -/
  congr
  /-
    case e_a
    m n : Nat
    z : Complex
    x : Real
    Hn : LE.le 0 (Inv.inv (HAdd.hAdd (↑n) 1))
    ⊢ Eq (HDiv.hDiv (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x) (HPow.hPow ↑m ↑x)) (Inv.inv  …
  -/
  simp_rw [div_eq_mul_inv]
  rw [show (n + 1 : ℂ)⁻¹ = (n + 1 : ℝ)⁻¹ by simp,
    show (n + 1 : ℂ) = (n + 1 : ℝ) by norm_cast, show (m : ℂ) = (m : ℝ) by norm_cast,
    mul_cpow_ofReal_nonneg m.cast_nonneg Hn, mul_inv, mul_comm]
  /-
    case e_a
    m n : Nat
    z : Complex
    x : Real
    Hn : LE.le 0 (Inv.inv (HAdd.hAdd (↑n) 1))
    ⊢ Eq (HMul.hMul (Inv.inv (HPow.hPow ↑↑m ↑x)) (HPow.hPow ↑(HAdd.hAdd (↑n) 1) ↑x …
  -/
  congr
  rw [← cpow_neg, show (-x : ℂ) = (-1 : ℝ) * x by simp, cpow_mul_ofReal_nonneg Hn,
    Real.rpow_neg_one, inv_inv]


open Filter Real in
/-- If the coefficients `f m` of an L-series are zero for `m ≤ n` and the L-series converges
at some point, then `f (n+1)` is the limit of `(n+1)^x * LSeries f x` as `x → ∞`. -/
lemma LSeries.tendsto_cpow_mul_atTop {f : ℕ → ℂ} {n : ℕ} (h : ∀ m ≤ n, f m = 0)
    (ha : abscissaOfAbsConv f < ⊤):
    Tendsto (fun x : ℝ ↦ (n + 1) ^ (x : ℂ) * LSeries f x) atTop (nhds (f (n + 1))) := by
  /-
    f : Nat → Complex
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x) (LSerie …
  -/
  obtain ⟨y, hay, hyt⟩ := exists_between ha
  /-
    case intro.intro
    f : Nat → Complex
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    y : EReal
    hay : LT.lt (LSeries.abscissaOfAbsConv f) y
    hyt : LT.lt y Top.top
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x) (LSerie …
  -/
  lift y to ℝ using ⟨hyt.ne, ((OrderBot.bot_le _).trans_lt hay).ne'⟩
  -- `F x m` is the `m`th term of `(n+1)^x * LSeries f x`, except that `F x (n+1) = 0`
  /-
    case intro.intro.intro
    f : Nat → Complex
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    y : Real
    hay : LT.lt (LSeries.abscissaOfAbsConv f) ↑y
    hyt : LT.lt (↑y) Top.top
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x) (LSerie …
  -/
  let F := fun (x : ℝ) ↦ {m | n + 1 < m}.indicator (fun m ↦ f m / (m / (n + 1) : ℂ) ^ (x : ℂ))
  /-
    case intro.intro.intro
    f : Nat → Complex
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    y : Real
    hay : LT.lt (LSeries.abscissaOfAbsConv f) ↑y
    hyt : LT.lt (↑y) Top.top
    F : Real → Nat → Complex := fun x => (setOf fun m => LT.lt (HAdd.hAdd n 1) m). …
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x) (LSerie …
  -/
  have hF₀ (x : ℝ) {m : ℕ} (hm : m ≤ n + 1) : F x m = 0 := by simp [F, not_lt_of_le hm]
  have hF (x : ℝ) {m : ℕ} (hm : m ≠ n + 1) : F x m = ((n + 1) ^ (x : ℂ)) * term f x m := by
    rcases lt_trichotomy m (n + 1) with H | rfl | H
    · simp [Nat.not_lt_of_gt H, term, h m <| Nat.lt_succ_iff.mp H, F]
    · exact (hm rfl).elim
    · simp [H, term, (n.zero_lt_succ.trans H).ne', F, cpow_mul_div_cpow_eq_div_div_cpow]
  have hs {x : ℝ} (hx : x ≥ y) : Summable fun m ↦ (n + 1) ^ (x : ℂ) * term f x m := by
    refine (summable_mul_left_iff <| natCast_add_one_cpow_ne_zero n _).mpr <|
       LSeriesSummable_of_abscissaOfAbsConv_lt_re ?_
    simpa only [ofReal_re] using hay.trans_le <| EReal.coe_le_coe_iff.mpr hx
  -- we can write `(n+1)^x * LSeries f x` as `f (n+1)` plus the series over `F x`
  have key : ∀ x ≥ y, (n + 1) ^ (x : ℂ) * LSeries f x = f (n + 1) + ∑' m : ℕ, F x m := by
    intro x hx
    rw [LSeries, ← tsum_mul_left, tsum_eq_add_tsum_ite (hs hx) (n + 1), pow_mul_term_eq f x n]
    congr
    ext1 m
    rcases eq_or_ne m (n + 1) with rfl | hm
    · simp [hF₀ x le_rfl]
    · simp [hm, hF]
  -- reduce to showing that `∑' m, F x m → 0` as `x → ∞`
  /-
    case intro.intro.intro
    f : Nat → Complex
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    y : Real
    hay : LT.lt (LSeries.abscissaOfAbsConv f) ↑y
    hyt : LT.lt (↑y) Top.top
    F : Real → Nat → Complex := fun x => (setOf fun m => LT.lt (HAdd.hAdd n 1) m). …
    hF₀ : ∀ (x : Real) {m : Nat}, LE.le m (HAdd.hAdd n 1) → Eq (F x m) 0
    hF : ∀ (x : Real) {m : Nat}, Ne m (HAdd.hAdd n 1) → Eq (F x m) (HMul.hMul (HPo …
    hs : ∀ {x : Real}, GE.ge x y → Summable fun m => HMul.hMul (HPow.hPow (HAdd.hA …
    key : ∀ (x : Real), GE.ge x y → Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x …
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x) (LSerie …
  -/
  conv => enter [3, 1]; rw [← add_zero (f _)]
  refine Tendsto.congr'
    (eventuallyEq_of_mem (s := {x | y ≤ x}) (mem_atTop y) key).symm <| tendsto_const_nhds.add ?_
  -- get the prerequisites for applying dominated convergence
  have hys : Summable (F y) := by
    refine ((hs le_rfl).indicator {m | n + 1 < m}).congr fun m ↦ ?_
    by_cases hm : n + 1 < m
    · simp [hF, hm, hm.ne']
    · simp [hm, hF₀ _ (le_of_not_lt hm)]
  have hc (k : ℕ) : Tendsto (F · k) atTop (nhds 0) := by
    rcases lt_or_le (n + 1) k with H | H
    · have H₀ : (0 : ℝ) ≤ k / (n + 1) := by positivity
      have H₀' : (0 : ℝ) ≤ (n + 1) / k := by positivity
      have H₁ : (k / (n + 1) : ℂ) = (k / (n + 1) : ℝ) := by push_cast; rfl
      have H₂ : (n + 1) / k < (1 : ℝ) :=
        (div_lt_one <| mod_cast n.succ_pos.trans H).mpr <| mod_cast H
      simp only [Set.mem_setOf_eq, H, Set.indicator_of_mem, F]
      conv =>
        enter [1, x]
        rw [div_eq_mul_inv, H₁, ← ofReal_cpow H₀, ← ofReal_inv, ← Real.inv_rpow H₀, inv_div]
      conv => enter [3, 1]; rw [← mul_zero (f k)]
      exact
        (tendsto_rpow_atTop_of_base_lt_one _ (neg_one_lt_zero.trans_le H₀') H₂).ofReal.const_mul _
    · simp [hF₀ _ H]
  /-
    case intro.intro.intro
    f : Nat → Complex
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    y : Real
    hay : LT.lt (LSeries.abscissaOfAbsConv f) ↑y
    hyt : LT.lt (↑y) Top.top
    F : Real → Nat → Complex := fun x => (setOf fun m => LT.lt (HAdd.hAdd n 1) m). …
    hF₀ : ∀ (x : Real) {m : Nat}, LE.le m (HAdd.hAdd n 1) → Eq (F x m) 0
    hF : ∀ (x : Real) {m : Nat}, Ne m (HAdd.hAdd n 1) → Eq (F x m) (HMul.hMul (HPo …
    hs : ∀ {x : Real}, GE.ge x y → Summable fun m => HMul.hMul (HPow.hPow (HAdd.hA …
    key : ∀ (x : Real), GE.ge x y → Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x …
    hys : Summable (F y)
    hc : ∀ (k : Nat), Filter.Tendsto (fun x => F x k) Filter.atTop (nhds 0)
    ⊢ Filter.Tendsto (fun x => tsum fun m => F x m) Filter.atTop (nhds 0)
  -/
  rw [show (0 : ℂ) = tsum (fun _ : ℕ ↦ 0) from tsum_zero.symm]
  /-
    case intro.intro.intro
    f : Nat → Complex
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    y : Real
    hay : LT.lt (LSeries.abscissaOfAbsConv f) ↑y
    hyt : LT.lt (↑y) Top.top
    F : Real → Nat → Complex := fun x => (setOf fun m => LT.lt (HAdd.hAdd n 1) m). …
    hF₀ : ∀ (x : Real) {m : Nat}, LE.le m (HAdd.hAdd n 1) → Eq (F x m) 0
    hF : ∀ (x : Real) {m : Nat}, Ne m (HAdd.hAdd n 1) → Eq (F x m) (HMul.hMul (HPo …
    hs : ∀ {x : Real}, GE.ge x y → Summable fun m => HMul.hMul (HPow.hPow (HAdd.hA …
    key : ∀ (x : Real), GE.ge x y → Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x …
    hys : Summable (F y)
    hc : ∀ (k : Nat), Filter.Tendsto (fun x => F x k) Filter.atTop (nhds 0)
    ⊢ Filter.Tendsto (fun x => tsum fun m => F x m) Filter.atTop (nhds (tsum fun x …
  -/
  refine tendsto_tsum_of_dominated_convergence hys.norm hc <| eventually_iff.mpr ?_
  /-
    case intro.intro.intro
    f : Nat → Complex
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    y : Real
    hay : LT.lt (LSeries.abscissaOfAbsConv f) ↑y
    hyt : LT.lt (↑y) Top.top
    F : Real → Nat → Complex := fun x => (setOf fun m => LT.lt (HAdd.hAdd n 1) m). …
    hF₀ : ∀ (x : Real) {m : Nat}, LE.le m (HAdd.hAdd n 1) → Eq (F x m) 0
    hF : ∀ (x : Real) {m : Nat}, Ne m (HAdd.hAdd n 1) → Eq (F x m) (HMul.hMul (HPo …
    hs : ∀ {x : Real}, GE.ge x y → Summable fun m => HMul.hMul (HPow.hPow (HAdd.hA …
    key : ∀ (x : Real), GE.ge x y → Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x …
    hys : Summable (F y)
    hc : ∀ (k : Nat), Filter.Tendsto (fun x => F x k) Filter.atTop (nhds 0)
    ⊢ Membership.mem Filter.atTop (setOf fun x => ∀ (k : Nat), LE.le (Norm.norm (F …
  -/
  filter_upwards [mem_atTop y] with y' hy' k
  -- it remains to show that `‖F y' k‖ ≤ ‖F y k‖` (for `y' ≥ y`)
  /-
    case h
    f : Nat → Complex
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    y : Real
    hay : LT.lt (LSeries.abscissaOfAbsConv f) ↑y
    hyt : LT.lt (↑y) Top.top
    F : Real → Nat → Complex := fun x => (setOf fun m => LT.lt (HAdd.hAdd n 1) m). …
    hF₀ : ∀ (x : Real) {m : Nat}, LE.le m (HAdd.hAdd n 1) → Eq (F x m) 0
    hF : ∀ (x : Real) {m : Nat}, Ne m (HAdd.hAdd n 1) → Eq (F x m) (HMul.hMul (HPo …
    hs : ∀ {x : Real}, GE.ge x y → Summable fun m => HMul.hMul (HPow.hPow (HAdd.hA …
    key : ∀ (x : Real), GE.ge x y → Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x …
    hys : Summable (F y)
    hc : ∀ (k : Nat), Filter.Tendsto (fun x => F x k) Filter.atTop (nhds 0)
    y' : Real
    hy' : LE.le y y'
    k : Nat
    ⊢ LE.le (Norm.norm (F y' k)) (Norm.norm (F y k))
  -/
  rcases lt_or_le (n + 1) k with H | H
  · simp only [Set.mem_setOf_eq, H, Set.indicator_of_mem, norm_div, Complex.norm_eq_abs,
      abs_cpow_real, map_div₀, abs_natCast, F]
    /-
      case h.inl
      f : Nat → Complex
      n : Nat
      h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
      ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
      y : Real
      hay : LT.lt (LSeries.abscissaOfAbsConv f) ↑y
      hyt : LT.lt (↑y) Top.top
      F : Real → Nat → Complex := fun x => (setOf fun m => LT.lt (HAdd.hAdd n 1) m). …
      hF₀ : ∀ (x : Real) {m : Nat}, LE.le m (HAdd.hAdd n 1) → Eq (F x m) 0
      hF : ∀ (x : Real) {m : Nat}, Ne m (HAdd.hAdd n 1) → Eq (F x m) (HMul.hMul (HPo …
      hs : ∀ {x : Real}, GE.ge x y → Summable fun m => HMul.hMul (HPow.hPow (HAdd.hA …
      key : ∀ (x : Real), GE.ge x y → Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x …
      hys : Summable (F y)
      hc : ∀ (k : Nat), Filter.Tendsto (fun x => F x k) Filter.atTop (nhds 0)
      y' : Real
      hy' : LE.le y y'
      k : Nat
      H : LT.lt (HAdd.hAdd n 1) k
      ⊢ LE.le (HDiv.hDiv (Complex.abs (f k)) (HPow.hPow (HDiv.hDiv (↑k) (Complex.abs …
    -/
    rw [← Nat.cast_one, ← Nat.cast_add, abs_natCast]
    have hkn : 1 ≤ (k / (n + 1 :) : ℝ) :=
      (one_le_div (by positivity)).mpr <| mod_cast Nat.le_of_succ_le H
    exact div_le_div_of_nonneg_left (Complex.abs.nonneg _)
      (rpow_pos_of_pos (zero_lt_one.trans_le hkn) _) <| rpow_le_rpow_of_exponent_le hkn hy'
    /-
      case h.inr
      f : Nat → Complex
      n : Nat
      h : ∀ (m : Nat), LE.le m n → Eq (f m) 0
      ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
      y : Real
      hay : LT.lt (LSeries.abscissaOfAbsConv f) ↑y
      hyt : LT.lt (↑y) Top.top
      F : Real → Nat → Complex := fun x => (setOf fun m => LT.lt (HAdd.hAdd n 1) m). …
      hF₀ : ∀ (x : Real) {m : Nat}, LE.le m (HAdd.hAdd n 1) → Eq (F x m) 0
      hF : ∀ (x : Real) {m : Nat}, Ne m (HAdd.hAdd n 1) → Eq (F x m) (HMul.hMul (HPo …
      hs : ∀ {x : Real}, GE.ge x y → Summable fun m => HMul.hMul (HPow.hPow (HAdd.hA …
      key : ∀ (x : Real), GE.ge x y → Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) ↑x …
      hys : Summable (F y)
      hc : ∀ (k : Nat), Filter.Tendsto (fun x => F x k) Filter.atTop (nhds 0)
      y' : Real
      hy' : LE.le y y'
      k : Nat
      H : LE.le k (HAdd.hAdd n 1)
      ⊢ LE.le (Norm.norm (F y' k)) (Norm.norm (F y k))
    -/
  · simp [hF₀ _ H]
    /-
      🎉 no goals
    -/


open Filter in
/-- If the L-series of `f` converges at some point, then `f 1` is the limit of `LSeries f x`
as `x → ∞`. -/
lemma LSeries.tendsto_atTop {f : ℕ → ℂ} (ha : abscissaOfAbsConv f < ⊤):
    Tendsto (fun x : ℝ ↦ LSeries f x) atTop (nhds (f 1)) := by
  /-
    f : Nat → Complex
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    ⊢ Filter.Tendsto (fun x => LSeries f ↑x) Filter.atTop (nhds (f 1))
  -/
  let F (n : ℕ) : ℂ := if n = 0 then 0 else f n
  /-
    f : Nat → Complex
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
    ⊢ Filter.Tendsto (fun x => LSeries f ↑x) Filter.atTop (nhds (f 1))
  -/
  have hF₀ : F 0 = 0 := rfl
  /-
    f : Nat → Complex
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
    hF₀ : Eq (F 0) 0
    ⊢ Filter.Tendsto (fun x => LSeries f ↑x) Filter.atTop (nhds (f 1))
  -/
  have hF {n : ℕ} (hn : n ≠ 0) : F n = f n := if_neg hn
  /-
    f : Nat → Complex
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
    hF₀ : Eq (F 0) 0
    hF : ∀ {n : Nat}, Ne n 0 → Eq (F n) (f n)
    ⊢ Filter.Tendsto (fun x => LSeries f ↑x) Filter.atTop (nhds (f 1))
  -/
  have ha' : abscissaOfAbsConv F < ⊤ := (abscissaOfAbsConv_congr hF).symm ▸ ha
  /-
    f : Nat → Complex
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
    hF₀ : Eq (F 0) 0
    hF : ∀ {n : Nat}, Ne n 0 → Eq (F n) (f n)
    ha' : LT.lt (LSeries.abscissaOfAbsConv F) Top.top
    ⊢ Filter.Tendsto (fun x => LSeries f ↑x) Filter.atTop (nhds (f 1))
  -/
  simp_rw [← LSeries_congr _ hF]
  /-
    f : Nat → Complex
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
    hF₀ : Eq (F 0) 0
    hF : ∀ {n : Nat}, Ne n 0 → Eq (F n) (f n)
    ha' : LT.lt (LSeries.abscissaOfAbsConv F) Top.top
    ⊢ Filter.Tendsto (fun x => LSeries F ↑x) Filter.atTop (nhds (f 1))
  -/
  convert LSeries.tendsto_cpow_mul_atTop (n := 0) (fun _ hm ↦ Nat.le_zero.mp hm ▸ hF₀) ha' using 1
  /-
    case h.e'_3
    f : Nat → Complex
    ha : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
    hF₀ : Eq (F 0) 0
    hF : ∀ {n : Nat}, Ne n 0 → Eq (F n) (f n)
    ha' : LT.lt (LSeries.abscissaOfAbsConv F) Top.top
    ⊢ Eq (fun x => LSeries F ↑x) fun x => HMul.hMul (HPow.hPow (HAdd.hAdd (↑0) 1)  …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma LSeries_eq_zero_of_abscissaOfAbsConv_eq_top {f : ℕ → ℂ} (h : abscissaOfAbsConv f = ⊤) :
    LSeries f = 0 := by
  /-
    f : Nat → Complex
    h : Eq (LSeries.abscissaOfAbsConv f) Top.top
    ⊢ Eq (LSeries f) 0
  -/
  ext1 s
  exact LSeries.eq_zero_of_not_LSeriesSummable f s <| mt LSeriesSummable.abscissaOfAbsConv_le <|
    h ▸ fun H ↦ (H.trans_lt <| EReal.coe_lt_top _).false


open Filter Nat in
/-- The `LSeries` of `f` is zero for large real arguments if and only if either `f n = 0`
for all `n ≠ 0` or the L-series converges nowhere. -/
lemma LSeries_eventually_eq_zero_iff' {f : ℕ → ℂ} :
    (fun x : ℝ ↦ LSeries f x) =ᶠ[atTop] 0 ↔ (∀ n ≠ 0, f n = 0) ∨ abscissaOfAbsConv f = ⊤ := by
  /-
    f : Nat → Complex
    ⊢ Iff (Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0) (Or (∀ (n : Nat),  …
  -/
  by_cases h : abscissaOfAbsConv f = ⊤ <;> simp [h]
    /-
      case pos
      f : Nat → Complex
      h : Eq (LSeries.abscissaOfAbsConv f) Top.top
      ⊢ Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0
    -/
  · exact Eventually.of_forall <| by simp [LSeries_eq_zero_of_abscissaOfAbsConv_eq_top h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      f : Nat → Complex
      h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
      ⊢ Iff (Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0) (∀ (n : Nat), Not  …
    -/
  · refine ⟨fun H ↦ ?_, fun H ↦ Eventually.of_forall fun x ↦ ?_⟩
      /-
        case neg.refine_1
        f : Nat → Complex
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0
        ⊢ ∀ (n : Nat), Not (Eq n 0) → Eq (f n) 0
      -/
    · let F (n : ℕ) : ℂ := if n = 0 then 0 else f n
      /-
        case neg.refine_1
        f : Nat → Complex
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0
        F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
        ⊢ ∀ (n : Nat), Not (Eq n 0) → Eq (f n) 0
      -/
      have hF₀ : F 0 = 0 := rfl
      /-
        case neg.refine_1
        f : Nat → Complex
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0
        F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
        hF₀ : Eq (F 0) 0
        ⊢ ∀ (n : Nat), Not (Eq n 0) → Eq (f n) 0
      -/
      have hF {n : ℕ} (hn : n ≠ 0) : F n = f n := if_neg hn
      /-
        case neg.refine_1
        f : Nat → Complex
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0
        F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
        hF₀ : Eq (F 0) 0
        hF : ∀ {n : Nat}, Ne n 0 → Eq (F n) (f n)
        ⊢ ∀ (n : Nat), Not (Eq n 0) → Eq (f n) 0
      -/
      suffices ∀ n, F n = 0 from fun n hn ↦ (hF hn).symm.trans (this n)
      /-
        case neg.refine_1
        f : Nat → Complex
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0
        F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
        hF₀ : Eq (F 0) 0
        hF : ∀ {n : Nat}, Ne n 0 → Eq (F n) (f n)
        ⊢ ∀ (n : Nat), Eq (F n) 0
      -/
      have ha : ¬ abscissaOfAbsConv F = ⊤ := abscissaOfAbsConv_congr hF ▸ h
      /-
        case neg.refine_1
        f : Nat → Complex
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0
        F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
        hF₀ : Eq (F 0) 0
        hF : ∀ {n : Nat}, Ne n 0 → Eq (F n) (f n)
        ha : Not (Eq (LSeries.abscissaOfAbsConv F) Top.top)
        ⊢ ∀ (n : Nat), Eq (F n) 0
      -/
      have h' (x : ℝ) : LSeries F x = LSeries f x := LSeries_congr x hF
      have H' (n : ℕ) : (fun x : ℝ ↦ n ^ (x : ℂ) * LSeries F x) =ᶠ[atTop] fun _ ↦ 0 := by
        simp only [h']
        rw [eventuallyEq_iff_exists_mem] at H ⊢
        obtain ⟨s, hs⟩ := H
        exact ⟨s, hs.1, fun x hx ↦ by simp [hs.2 hx]⟩
      /-
        case neg.refine_1
        f : Nat → Complex
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0
        F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
        hF₀ : Eq (F 0) 0
        hF : ∀ {n : Nat}, Ne n 0 → Eq (F n) (f n)
        ha : Not (Eq (LSeries.abscissaOfAbsConv F) Top.top)
        h' : ∀ (x : Real), Eq (LSeries F ↑x) (LSeries f ↑x)
        H' : ∀ (n : Nat), Filter.atTop.EventuallyEq (fun x => HMul.hMul (HPow.hPow ↑n  …
        ⊢ ∀ (n : Nat), Eq (F n) 0
      -/
      intro n
      /-
        case neg.refine_1
        f : Nat → Complex
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0
        F : Nat → Complex := fun n => ite (Eq n 0) 0 (f n)
        hF₀ : Eq (F 0) 0
        hF : ∀ {n : Nat}, Ne n 0 → Eq (F n) (f n)
        ha : Not (Eq (LSeries.abscissaOfAbsConv F) Top.top)
        h' : ∀ (x : Real), Eq (LSeries F ↑x) (LSeries f ↑x)
        H' : ∀ (n : Nat), Filter.atTop.EventuallyEq (fun x => HMul.hMul (HPow.hPow ↑n  …
        n : Nat
        ⊢ Eq (F n) 0
      -/
      induction' n using Nat.strongRecOn with n ih
      -- it suffices to show that `n ^ x * LSeries F x` tends to `F n` as `x` tends to `∞`
      suffices Tendsto (fun x : ℝ ↦ n ^ (x : ℂ) * LSeries F x) atTop (nhds (F n)) by
        replace this := this.congr' <| H' n
        simp only [tendsto_const_nhds_iff] at this
        exact this.symm
      cases n with
      | zero => exact Tendsto.congr' (H' 0).symm <| by simp [hF₀]
      | succ n =>
          simpa using LSeries.tendsto_cpow_mul_atTop (fun m hm ↦ ih m <| lt_succ_of_le hm) <|
            Ne.lt_top ha
      /-
        case neg.refine_2
        f : Nat → Complex
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : ∀ (n : Nat), Not (Eq n 0) → Eq (f n) 0
        x : Real
        ⊢ Eq ((fun x => LSeries f ↑x) x) (0 x)
      -/
    · simp [LSeries_congr x fun {n} ↦ H n, show (fun _ : ℕ ↦ (0 : ℂ)) = 0 from rfl]
      /-
        🎉 no goals
      -/


open Nat in
/-- Assuming `f 0 = 0`, the `LSeries` of `f` is zero if and only if either `f = 0` or the
L-series converges nowhere. -/
lemma LSeries_eq_zero_iff {f : ℕ → ℂ} (hf : f 0 = 0) :
    LSeries f = 0 ↔ f = 0 ∨ abscissaOfAbsConv f = ⊤ := by
  /-
    f : Nat → Complex
    hf : Eq (f 0) 0
    ⊢ Iff (Eq (LSeries f) 0) (Or (Eq f 0) (Eq (LSeries.abscissaOfAbsConv f) Top.to …
  -/
  by_cases h : abscissaOfAbsConv f = ⊤ <;> simp [h]
    /-
      case pos
      f : Nat → Complex
      hf : Eq (f 0) 0
      h : Eq (LSeries.abscissaOfAbsConv f) Top.top
      ⊢ Eq (LSeries f) 0
    -/
  · exact LSeries_eq_zero_of_abscissaOfAbsConv_eq_top h
    /-
      🎉 no goals
    -/
    /-
      case neg
      f : Nat → Complex
      hf : Eq (f 0) 0
      h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
      ⊢ Iff (Eq (LSeries f) 0) (Eq f 0)
    -/
  · refine ⟨fun H ↦ ?_, fun H ↦ H ▸ LSeries_zero⟩
    /-
      case neg
      f : Nat → Complex
      hf : Eq (f 0) 0
      h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
      H : Eq (LSeries f) 0
      ⊢ Eq f 0
    -/
    convert (LSeries_eventually_eq_zero_iff'.mp ?_).resolve_right h
      /-
        case a
        f : Nat → Complex
        hf : Eq (f 0) 0
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Eq (LSeries f) 0
        ⊢ Iff (Eq f 0) (∀ (n : Nat), Ne n 0 → Eq (f n) 0)
      -/
    · refine ⟨fun H' _ _ ↦ by rw [H', Pi.zero_apply], fun H' ↦ ?_⟩
      /-
        case a
        f : Nat → Complex
        hf : Eq (f 0) 0
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Eq (LSeries f) 0
        H' : ∀ (n : Nat), Ne n 0 → Eq (f n) 0
        ⊢ Eq f 0
      -/
      ext (- | m)
        /-
          case a.h.zero
          f : Nat → Complex
          hf : Eq (f 0) 0
          h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
          H : Eq (LSeries f) 0
          H' : ∀ (n : Nat), Ne n 0 → Eq (f n) 0
          ⊢ Eq (f 0) (0 0)
        -/
      · simp [hf]
        /-
          🎉 no goals
        -/
        /-
          case a.h.succ
          f : Nat → Complex
          hf : Eq (f 0) 0
          h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
          H : Eq (LSeries f) 0
          H' : ∀ (n : Nat), Ne n 0 → Eq (f n) 0
          m : Nat
          ⊢ Eq (f (HAdd.hAdd m 1)) (0 (HAdd.hAdd m 1))
        -/
      · simp [H']
        /-
          🎉 no goals
        -/
      /-
        case neg
        f : Nat → Complex
        hf : Eq (f 0) 0
        h : Not (Eq (LSeries.abscissaOfAbsConv f) Top.top)
        H : Eq (LSeries f) 0
        ⊢ Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) 0
      -/
    · simpa only [H] using Filter.EventuallyEq.rfl
      /-
        🎉 no goals
      -/


open Filter in
/-- If the `LSeries` of `f` and of `g` converge somewhere and agree on large real arguments,
then the L-series of `f - g` is zero for large real arguments. -/
lemma LSeries_sub_eventuallyEq_zero_of_LSeries_eventually_eq {f g : ℕ → ℂ}
    (hf : abscissaOfAbsConv f < ⊤) (hg : abscissaOfAbsConv g < ⊤)
    (h : (fun x : ℝ ↦ LSeries f x) =ᶠ[atTop] fun x ↦ LSeries g x) :
    (fun x : ℝ ↦ LSeries (f - g) x) =ᶠ[atTop] (0 : ℝ → ℂ) := by
  /-
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    h : Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) fun x => LSeries g ↑x
    ⊢ Filter.atTop.EventuallyEq (fun x => LSeries (HSub.hSub f g) ↑x) 0
  -/
  rw [EventuallyEq, eventually_atTop] at h ⊢
  /-
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    h : Exists fun a => ∀ (b : Real), GE.ge b a → Eq (LSeries f ↑b) (LSeries g ↑b)
    ⊢ Exists fun a => ∀ (b : Real), GE.ge b a → Eq (LSeries (HSub.hSub f g) ↑b) (0 …
  -/
  obtain ⟨x₀, hx₀⟩ := h
  /-
    case intro
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    x₀ : Real
    hx₀ : ∀ (b : Real), GE.ge b x₀ → Eq (LSeries f ↑b) (LSeries g ↑b)
    ⊢ Exists fun a => ∀ (b : Real), GE.ge b a → Eq (LSeries (HSub.hSub f g) ↑b) (0 …
  -/
  obtain ⟨yf, hyf₁, hyf₂⟩ := exists_between hf
  /-
    case intro.intro.intro
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    x₀ : Real
    hx₀ : ∀ (b : Real), GE.ge b x₀ → Eq (LSeries f ↑b) (LSeries g ↑b)
    yf : EReal
    hyf₁ : LT.lt (LSeries.abscissaOfAbsConv f) yf
    hyf₂ : LT.lt yf Top.top
    ⊢ Exists fun a => ∀ (b : Real), GE.ge b a → Eq (LSeries (HSub.hSub f g) ↑b) (0 …
  -/
  obtain ⟨yg, hyg₁, hyg₂⟩ := exists_between hg
  /-
    case intro.intro.intro.intro.intro
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    x₀ : Real
    hx₀ : ∀ (b : Real), GE.ge b x₀ → Eq (LSeries f ↑b) (LSeries g ↑b)
    yf : EReal
    hyf₁ : LT.lt (LSeries.abscissaOfAbsConv f) yf
    hyf₂ : LT.lt yf Top.top
    yg : EReal
    hyg₁ : LT.lt (LSeries.abscissaOfAbsConv g) yg
    hyg₂ : LT.lt yg Top.top
    ⊢ Exists fun a => ∀ (b : Real), GE.ge b a → Eq (LSeries (HSub.hSub f g) ↑b) (0 …
  -/
  lift yf to ℝ using ⟨hyf₂.ne, ((OrderBot.bot_le _).trans_lt hyf₁).ne'⟩
  /-
    case intro.intro.intro.intro.intro.intro
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    x₀ : Real
    hx₀ : ∀ (b : Real), GE.ge b x₀ → Eq (LSeries f ↑b) (LSeries g ↑b)
    yg : EReal
    hyg₁ : LT.lt (LSeries.abscissaOfAbsConv g) yg
    hyg₂ : LT.lt yg Top.top
    yf : Real
    hyf₁ : LT.lt (LSeries.abscissaOfAbsConv f) ↑yf
    hyf₂ : LT.lt (↑yf) Top.top
    ⊢ Exists fun a => ∀ (b : Real), GE.ge b a → Eq (LSeries (HSub.hSub f g) ↑b) (0 …
  -/
  lift yg to ℝ using ⟨hyg₂.ne, ((OrderBot.bot_le _).trans_lt hyg₁).ne'⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    x₀ : Real
    hx₀ : ∀ (b : Real), GE.ge b x₀ → Eq (LSeries f ↑b) (LSeries g ↑b)
    yf : Real
    hyf₁ : LT.lt (LSeries.abscissaOfAbsConv f) ↑yf
    hyf₂ : LT.lt (↑yf) Top.top
    yg : Real
    hyg₁ : LT.lt (LSeries.abscissaOfAbsConv g) ↑yg
    hyg₂ : LT.lt (↑yg) Top.top
    ⊢ Exists fun a => ∀ (b : Real), GE.ge b a → Eq (LSeries (HSub.hSub f g) ↑b) (0 …
  -/
  refine ⟨max x₀ (max yf yg), fun x hx ↦ ?_⟩
  have Hf : LSeriesSummable f x := by
    refine LSeriesSummable_of_abscissaOfAbsConv_lt_re <|
      (ofReal_re x).symm ▸ hyf₁.trans_le (EReal.coe_le_coe_iff.mpr ?_)
    exact (le_max_left _ yg).trans <| (le_max_right x₀ _).trans hx
  have Hg : LSeriesSummable g x := by
    refine LSeriesSummable_of_abscissaOfAbsConv_lt_re <|
      (ofReal_re x).symm ▸ hyg₁.trans_le (EReal.coe_le_coe_iff.mpr ?_)
    exact (le_max_right yf _).trans <| (le_max_right x₀ _).trans hx
  /-
    case intro.intro.intro.intro.intro.intro.intro
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    x₀ : Real
    hx₀ : ∀ (b : Real), GE.ge b x₀ → Eq (LSeries f ↑b) (LSeries g ↑b)
    yf : Real
    hyf₁ : LT.lt (LSeries.abscissaOfAbsConv f) ↑yf
    hyf₂ : LT.lt (↑yf) Top.top
    yg : Real
    hyg₁ : LT.lt (LSeries.abscissaOfAbsConv g) ↑yg
    hyg₂ : LT.lt (↑yg) Top.top
    x : Real
    hx : GE.ge x (Max.max x₀ (Max.max yf yg))
    Hf : LSeriesSummable f ↑x
    Hg : LSeriesSummable g ↑x
    ⊢ Eq (LSeries (HSub.hSub f g) ↑x) (0 x)
  -/
  rw [LSeries_sub Hf Hg, hx₀ x <| (le_max_left ..).trans hx, sub_self, Pi.zero_apply]
  /-
    🎉 no goals
  -/


open Filter in
/-- If the `LSeries` of `f` and of `g` converge somewhere and agree on large real arguments,
then `f n = g n` whenever `n ≠ 0`. -/
lemma LSeries.eq_of_LSeries_eventually_eq {f g : ℕ → ℂ} (hf : abscissaOfAbsConv f < ⊤)
    (hg : abscissaOfAbsConv g < ⊤) (h : (fun x : ℝ ↦ LSeries f x) =ᶠ[atTop] fun x ↦ LSeries g x)
    {n : ℕ} (hn : n ≠ 0) :
    f n = g n := by
  have hsub : (fun x : ℝ ↦ LSeries (f - g) x) =ᶠ[atTop] (0 : ℝ → ℂ) :=
    LSeries_sub_eventuallyEq_zero_of_LSeries_eventually_eq hf hg h
  have ha : abscissaOfAbsConv (f - g) ≠ ⊤ :=
    lt_top_iff_ne_top.mp <| (abscissaOfAbsConv_sub_le f g).trans_lt <| max_lt hf hg
  simpa only [Pi.sub_apply, sub_eq_zero]
    using (LSeries_eventually_eq_zero_iff'.mp hsub).resolve_right ha n hn


/-- If the `LSeries` of `f` and of `g` both converge somewhere, then they are equal if and only
if `f n = g n` whenever `n ≠ 0`. -/
lemma LSeries_eq_iff_of_abscissaOfAbsConv_lt_top {f g : ℕ → ℂ} (hf : abscissaOfAbsConv f < ⊤)
    (hg : abscissaOfAbsConv g < ⊤) :
    LSeries f = LSeries g ↔ ∀ n ≠ 0, f n = g n := by
  /-
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    ⊢ Iff (Eq (LSeries f) (LSeries g)) (∀ (n : Nat), Ne n 0 → Eq (f n) (g n))
  -/
  refine ⟨fun H n hn ↦ ?_, fun H ↦ funext (LSeries_congr · fun {n} ↦ H n)⟩
  /-
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    H : Eq (LSeries f) (LSeries g)
    n : Nat
    hn : Ne n 0
    ⊢ Eq (f n) (g n)
  -/
  refine eq_of_LSeries_eventually_eq hf hg ?_ hn
  /-
    f g : Nat → Complex
    hf : LT.lt (LSeries.abscissaOfAbsConv f) Top.top
    hg : LT.lt (LSeries.abscissaOfAbsConv g) Top.top
    H : Eq (LSeries f) (LSeries g)
    n : Nat
    hn : Ne n 0
    ⊢ Filter.atTop.EventuallyEq (fun x => LSeries f ↑x) fun x => LSeries g ↑x
  -/
  exact Filter.Eventually.of_forall fun x ↦ congr_fun H x
  /-
    🎉 no goals
  -/


/-- The map `f ↦ LSeries f` is injective on functions `f` such that `f 0 = 0` and the L-series
of `f` converges somewhere. -/
lemma LSeries_injOn : Set.InjOn LSeries {f | f 0 = 0 ∧ abscissaOfAbsConv f < ⊤} := by
  /-
    ⊢ Set.InjOn LSeries (setOf fun f => And (Eq (f 0) 0) (LT.lt (LSeries.abscissaO …
  -/
  intro f hf g hg h
  /-
    f : Nat → Complex
    hf : Membership.mem (setOf fun f => And (Eq (f 0) 0) (LT.lt (LSeries.abscissaO …
    g : Nat → Complex
    hg : Membership.mem (setOf fun f => And (Eq (f 0) 0) (LT.lt (LSeries.abscissaO …
    h : Eq (LSeries f) (LSeries g)
    ⊢ Eq f g
  -/
  simp only [Set.mem_setOf] at hf hg
  /-
    f g : Nat → Complex
    h : Eq (LSeries f) (LSeries g)
    hf : And (Eq (f 0) 0) (LT.lt (LSeries.abscissaOfAbsConv f) Top.top)
    hg : And (Eq (g 0) 0) (LT.lt (LSeries.abscissaOfAbsConv g) Top.top)
    ⊢ Eq f g
  -/
  replace h := (LSeries_eq_iff_of_abscissaOfAbsConv_lt_top hf.2 hg.2).mp h
  /-
    f g : Nat → Complex
    hf : And (Eq (f 0) 0) (LT.lt (LSeries.abscissaOfAbsConv f) Top.top)
    hg : And (Eq (g 0) 0) (LT.lt (LSeries.abscissaOfAbsConv g) Top.top)
    h : ∀ (n : Nat), Ne n 0 → Eq (f n) (g n)
    ⊢ Eq f g
  -/
  ext1 n
  cases n with
  | zero => exact hf.1.trans hg.1.symm
  | succ n => exact h _ n.zero_ne_add_one.symm

