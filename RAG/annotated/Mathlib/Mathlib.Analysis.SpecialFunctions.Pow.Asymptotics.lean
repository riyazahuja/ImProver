/-- The function `x ^ y` tends to `+∞` at `+∞` for any positive real `y`. -/
theorem tendsto_rpow_atTop {y : ℝ} (hy : 0 < y) : Tendsto (fun x : ℝ => x ^ y) atTop atTop := by
  /-
    y : Real
    hy : LT.lt 0 y
    ⊢ Filter.Tendsto (fun x => HPow.hPow x y) Filter.atTop Filter.atTop
  -/
  rw [tendsto_atTop_atTop]
  /-
    y : Real
    hy : LT.lt 0 y
    ⊢ ∀ (b : Real), Exists fun i => ∀ (a : Real), LE.le i a → LE.le b (HPow.hPow a …
  -/
  intro b
  /-
    y : Real
    hy : LT.lt 0 y
    b : Real
    ⊢ Exists fun i => ∀ (a : Real), LE.le i a → LE.le b (HPow.hPow a y)
  -/
  use max b 0 ^ (1 / y)
  /-
    case h
    y : Real
    hy : LT.lt 0 y
    b : Real
    ⊢ ∀ (a : Real), LE.le (HPow.hPow (Max.max b 0) (HDiv.hDiv 1 y)) a → LE.le b (H …
  -/
  intro x hx
  exact
    le_of_max_le_left
      (by
        convert rpow_le_rpow (rpow_nonneg (le_max_right b 0) (1 / y)) hx (le_of_lt hy)
          using 1
        rw [← rpow_mul (le_max_right b 0), (eq_div_iff (ne_of_gt hy)).mp rfl, Real.rpow_one])


/-- The function `x ^ (-y)` tends to `0` at `+∞` for any positive real `y`. -/
theorem tendsto_rpow_neg_atTop {y : ℝ} (hy : 0 < y) : Tendsto (fun x : ℝ => x ^ (-y)) atTop (𝓝 0) :=
  Tendsto.congr' (eventuallyEq_of_mem (Ioi_mem_atTop 0) fun _ hx => (rpow_neg (le_of_lt hx) y).symm)
    (tendsto_rpow_atTop hy).inv_tendsto_atTop


open Asymptotics in
lemma tendsto_rpow_atTop_of_base_lt_one (b : ℝ) (hb₀ : -1 < b) (hb₁ : b < 1) :
    Tendsto (b ^ · : ℝ → ℝ) atTop (𝓝 (0 : ℝ)) := by
  /-
    b : Real
    hb₀ : LT.lt (-1) b
    hb₁ : LT.lt b 1
    ⊢ Filter.Tendsto (fun x => HPow.hPow b x) Filter.atTop (nhds 0)
  -/
  rcases lt_trichotomy b 0 with hb|rfl|hb
  case inl => -- b < 0
    simp_rw [Real.rpow_def_of_nonpos hb.le, hb.ne, ite_false]
    rw [← isLittleO_const_iff (c := (1 : ℝ)) one_ne_zero, (one_mul (1 : ℝ)).symm]
    refine IsLittleO.mul_isBigO ?exp ?cos
    case exp =>
      rw [isLittleO_const_iff one_ne_zero]
      refine tendsto_exp_atBot.comp <| (tendsto_const_mul_atBot_of_neg ?_).mpr tendsto_id
      rw [← log_neg_eq_log, log_neg_iff (by linarith)]
      linarith
    case cos =>
      rw [isBigO_iff]
      exact ⟨1, Eventually.of_forall fun x => by simp [Real.abs_cos_le_one]⟩
  case inr.inl => -- b = 0
    refine Tendsto.mono_right ?_ (Iff.mpr pure_le_nhds_iff rfl)
    rw [tendsto_pure]
    filter_upwards [eventually_ne_atTop 0] with _ hx
    simp [hx]
  case inr.inr => -- b > 0
    simp_rw [Real.rpow_def_of_pos hb]
    refine tendsto_exp_atBot.comp <| (tendsto_const_mul_atBot_of_neg ?_).mpr tendsto_id
    exact (log_neg_iff hb).mpr hb₁


lemma tendsto_rpow_atTop_of_base_gt_one (b : ℝ) (hb : 1 < b) :
    Tendsto (b ^ · : ℝ → ℝ) atBot (𝓝 (0 : ℝ)) := by
  /-
    b : Real
    hb : LT.lt 1 b
    ⊢ Filter.Tendsto (fun x => HPow.hPow b x) Filter.atBot (nhds 0)
  -/
  simp_rw [Real.rpow_def_of_pos (by positivity : 0 < b)]
  /-
    b : Real
    hb : LT.lt 1 b
    ⊢ Filter.Tendsto (fun x => Real.exp (HMul.hMul (Real.log b) x)) Filter.atBot ( …
  -/
  refine tendsto_exp_atBot.comp <| (tendsto_const_mul_atBot_of_pos ?_).mpr tendsto_id
  /-
    b : Real
    hb : LT.lt 1 b
    ⊢ LT.lt 0 (Real.log b)
  -/
  exact (log_pos_iff (by positivity)).mpr <| by aesop
  /-
    🎉 no goals
  -/


lemma tendsto_rpow_atBot_of_base_lt_one (b : ℝ) (hb₀ : 0 < b) (hb₁ : b < 1) :
    Tendsto (b ^ · : ℝ → ℝ) atBot atTop := by
  /-
    b : Real
    hb₀ : LT.lt 0 b
    hb₁ : LT.lt b 1
    ⊢ Filter.Tendsto (fun x => HPow.hPow b x) Filter.atBot Filter.atTop
  -/
  simp_rw [Real.rpow_def_of_pos (by positivity : 0 < b)]
  /-
    b : Real
    hb₀ : LT.lt 0 b
    hb₁ : LT.lt b 1
    ⊢ Filter.Tendsto (fun x => Real.exp (HMul.hMul (Real.log b) x)) Filter.atBot F …
  -/
  refine tendsto_exp_atTop.comp <| (tendsto_const_mul_atTop_iff_neg <| tendsto_id (α := ℝ)).mpr ?_
  /-
    b : Real
    hb₀ : LT.lt 0 b
    hb₁ : LT.lt b 1
    ⊢ LT.lt (Real.log b) 0
  -/
  exact (log_neg_iff hb₀).mpr hb₁
  /-
    🎉 no goals
  -/


lemma tendsto_rpow_atBot_of_base_gt_one (b : ℝ) (hb : 1 < b) :
    Tendsto (b ^ · : ℝ → ℝ) atBot (𝓝 0) := by
  /-
    b : Real
    hb : LT.lt 1 b
    ⊢ Filter.Tendsto (fun x => HPow.hPow b x) Filter.atBot (nhds 0)
  -/
  simp_rw [Real.rpow_def_of_pos (by positivity : 0 < b)]
  /-
    b : Real
    hb : LT.lt 1 b
    ⊢ Filter.Tendsto (fun x => Real.exp (HMul.hMul (Real.log b) x)) Filter.atBot ( …
  -/
  refine tendsto_exp_atBot.comp <| (tendsto_const_mul_atBot_iff_pos <| tendsto_id (α := ℝ)).mpr ?_
  /-
    b : Real
    hb : LT.lt 1 b
    ⊢ LT.lt 0 (Real.log b)
  -/
  exact (log_pos_iff (by positivity)).mpr <| by aesop
  /-
    🎉 no goals
  -/



/-- The function `x ^ (a / (b * x + c))` tends to `1` at `+∞`, for any real numbers `a`, `b`, and
`c` such that `b` is nonzero. -/
theorem tendsto_rpow_div_mul_add (a b c : ℝ) (hb : 0 ≠ b) :
    Tendsto (fun x => x ^ (a / (b * x + c))) atTop (𝓝 1) := by
  refine
    Tendsto.congr' ?_
      ((tendsto_exp_nhds_zero_nhds_one.comp
            (by
              simpa only [mul_zero, pow_one] using
                (tendsto_const_nhds (x := a)).mul
                  (tendsto_div_pow_mul_exp_add_atTop b c 1 hb))).comp
        tendsto_log_atTop)
  /-
    a b c : Real
    hb : Ne 0 b
    ⊢ Filter.atTop.EventuallyEq (Function.comp (Function.comp Real.exp fun x => HM …
  -/
  apply eventuallyEq_of_mem (Ioi_mem_atTop (0 : ℝ))
  /-
    a b c : Real
    hb : Ne 0 b
    ⊢ Set.EqOn (Function.comp (Function.comp Real.exp fun x => HMul.hMul a (HDiv.h …
  -/
  intro x hx
  /-
    a b c : Real
    hb : Ne 0 b
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (Function.comp (Function.comp Real.exp fun x => HMul.hMul a (HDiv.hDiv x  …
  -/
  simp only [Set.mem_Ioi, Function.comp_apply] at hx ⊢
  /-
    a b c : Real
    hb : Ne 0 b
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (Real.exp (HMul.hMul a (HDiv.hDiv (Real.log x) (HAdd.hAdd (HMul.hMul b (R …
  -/
  rw [exp_log hx, ← exp_log (rpow_pos_of_pos hx (a / (b * x + c))), log_rpow hx (a / (b * x + c))]
  /-
    a b c : Real
    hb : Ne 0 b
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (Real.exp (HMul.hMul a (HDiv.hDiv (Real.log x) (HAdd.hAdd (HMul.hMul b x) …
  -/
  field_simp
  /-
    🎉 no goals
  -/


/-- The function `x ^ (1 / x)` tends to `1` at `+∞`. -/
theorem tendsto_rpow_div : Tendsto (fun x => x ^ ((1 : ℝ) / x)) atTop (𝓝 1) := by
  /-
    ⊢ Filter.Tendsto (fun x => HPow.hPow x (HDiv.hDiv 1 x)) Filter.atTop (nhds 1)
  -/
  convert tendsto_rpow_div_mul_add (1 : ℝ) _ (0 : ℝ) zero_ne_one
  /-
    case h.e'_3.h.h.e'_6.h.e'_6
    x✝ : Real
    ⊢ Eq x✝ (HAdd.hAdd (HMul.hMul 1 x✝) 0)
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The function `x ^ (-1 / x)` tends to `1` at `+∞`. -/
theorem tendsto_rpow_neg_div : Tendsto (fun x => x ^ (-(1 : ℝ) / x)) atTop (𝓝 1) := by
  /-
    ⊢ Filter.Tendsto (fun x => HPow.hPow x (HDiv.hDiv (-1) x)) Filter.atTop (nhds 1)
  -/
  convert tendsto_rpow_div_mul_add (-(1 : ℝ)) _ (0 : ℝ) zero_ne_one
  /-
    case h.e'_3.h.h.e'_6.h.e'_6
    x✝ : Real
    ⊢ Eq x✝ (HAdd.hAdd (HMul.hMul 1 x✝) 0)
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The function `exp(x) / x ^ s` tends to `+∞` at `+∞`, for any real number `s`. -/
theorem tendsto_exp_div_rpow_atTop (s : ℝ) : Tendsto (fun x : ℝ => exp x / x ^ s) atTop atTop := by
  /-
    s : Real
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Real.exp x) (HPow.hPow x s)) Filter.atTo …
  -/
  cases' archimedean_iff_nat_lt.1 Real.instArchimedean s with n hn
  /-
    case intro
    s : Real
    n : Nat
    hn : LT.lt s ↑n
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Real.exp x) (HPow.hPow x s)) Filter.atTo …
  -/
  refine tendsto_atTop_mono' _ ?_ (tendsto_exp_div_pow_atTop n)
  /-
    case intro
    s : Real
    n : Nat
    hn : LT.lt s ↑n
    ⊢ Filter.atTop.EventuallyLE (fun x => HDiv.hDiv (Real.exp x) (HPow.hPow x n))  …
  -/
  filter_upwards [eventually_gt_atTop (0 : ℝ), eventually_ge_atTop (1 : ℝ)] with x hx₀ hx₁
  /-
    case h
    s : Real
    n : Nat
    hn : LT.lt s ↑n
    x : Real
    hx₀ : LT.lt 0 x
    hx₁ : LE.le 1 x
    ⊢ LE.le (HDiv.hDiv (Real.exp x) (HPow.hPow x n)) (HDiv.hDiv (Real.exp x) (HPow …
  -/
  gcongr
  /-
    case h.h
    s : Real
    n : Nat
    hn : LT.lt s ↑n
    x : Real
    hx₀ : LT.lt 0 x
    hx₁ : LE.le 1 x
    ⊢ LE.le (HPow.hPow x s) (HPow.hPow x n)
  -/
  simpa using rpow_le_rpow_of_exponent_le hx₁ hn.le
  /-
    🎉 no goals
  -/


/-- The function `exp (b * x) / x ^ s` tends to `+∞` at `+∞`, for any real `s` and `b > 0`. -/
theorem tendsto_exp_mul_div_rpow_atTop (s : ℝ) (b : ℝ) (hb : 0 < b) :
    Tendsto (fun x : ℝ => exp (b * x) / x ^ s) atTop atTop := by
  /-
    s b : Real
    hb : LT.lt 0 b
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Real.exp (HMul.hMul b x)) (HPow.hPow x s …
  -/
  refine ((tendsto_rpow_atTop hb).comp (tendsto_exp_div_rpow_atTop (s / b))).congr' ?_
  /-
    s b : Real
    hb : LT.lt 0 b
    ⊢ Filter.atTop.EventuallyEq (Function.comp (fun x => HPow.hPow x b) fun x => H …
  -/
  filter_upwards [eventually_ge_atTop (0 : ℝ)] with x hx₀
  simp [Real.div_rpow, (exp_pos x).le, rpow_nonneg, ← Real.rpow_mul, ← exp_mul,
    mul_comm x, hb.ne', *]


/-- The function `x ^ s * exp (-b * x)` tends to `0` at `+∞`, for any real `s` and `b > 0`. -/
theorem tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero (s : ℝ) (b : ℝ) (hb : 0 < b) :
    Tendsto (fun x : ℝ => x ^ s * exp (-b * x)) atTop (𝓝 0) := by
  /-
    s b : Real
    hb : LT.lt 0 b
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow x s) (Real.exp (HMul.hMul (Neg …
  -/
  refine (tendsto_exp_mul_div_rpow_atTop s b hb).inv_tendsto_atTop.congr' ?_
  /-
    s b : Real
    hb : LT.lt 0 b
    ⊢ Filter.atTop.EventuallyEq (Inv.inv fun x => HDiv.hDiv (Real.exp (HMul.hMul b …
  -/
  filter_upwards with x using by simp [exp_neg, inv_div, div_eq_mul_inv _ (exp _)]
  /-
    🎉 no goals
  -/


nonrec theorem NNReal.tendsto_rpow_atTop {y : ℝ} (hy : 0 < y) :
    Tendsto (fun x : ℝ≥0 => x ^ y) atTop atTop := by
  /-
    y : Real
    hy : LT.lt 0 y
    ⊢ Filter.Tendsto (fun x => HPow.hPow x y) Filter.atTop Filter.atTop
  -/
  rw [Filter.tendsto_atTop_atTop]
  /-
    y : Real
    hy : LT.lt 0 y
    ⊢ ∀ (b : NNReal), Exists fun i => ∀ (a : NNReal), LE.le i a → LE.le b (HPow.hP …
  -/
  intro b
  /-
    y : Real
    hy : LT.lt 0 y
    b : NNReal
    ⊢ Exists fun i => ∀ (a : NNReal), LE.le i a → LE.le b (HPow.hPow a y)
  -/
  obtain ⟨c, hc⟩ := tendsto_atTop_atTop.mp (tendsto_rpow_atTop hy) b
  /-
    case intro
    y : Real
    hy : LT.lt 0 y
    b : NNReal
    c : Real
    hc : ∀ (a : Real), LE.le c a → LE.le (↑b) (HPow.hPow a y)
    ⊢ Exists fun i => ∀ (a : NNReal), LE.le i a → LE.le b (HPow.hPow a y)
  -/
  use c.toNNReal
  /-
    case h
    y : Real
    hy : LT.lt 0 y
    b : NNReal
    c : Real
    hc : ∀ (a : Real), LE.le c a → LE.le (↑b) (HPow.hPow a y)
    ⊢ ∀ (a : NNReal), LE.le c.toNNReal a → LE.le b (HPow.hPow a y)
  -/
  intro a ha
  /-
    case h
    y : Real
    hy : LT.lt 0 y
    b : NNReal
    c : Real
    hc : ∀ (a : Real), LE.le c a → LE.le (↑b) (HPow.hPow a y)
    a : NNReal
    ha : LE.le c.toNNReal a
    ⊢ LE.le b (HPow.hPow a y)
  -/
  exact mod_cast hc a (Real.toNNReal_le_iff_le_coe.mp ha)
  /-
    🎉 no goals
  -/


theorem ENNReal.tendsto_rpow_at_top {y : ℝ} (hy : 0 < y) :
    Tendsto (fun x : ℝ≥0∞ => x ^ y) (𝓝 ⊤) (𝓝 ⊤) := by
  /-
    y : Real
    hy : LT.lt 0 y
    ⊢ Filter.Tendsto (fun x => HPow.hPow x y) (nhds Top.top) (nhds Top.top)
  -/
  rw [ENNReal.tendsto_nhds_top_iff_nnreal]
  /-
    y : Real
    hy : LT.lt 0 y
    ⊢ ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) (HPow.hPow a y)) (nhd …
  -/
  intro x
  obtain ⟨c, _, hc⟩ :=
    (atTop_basis_Ioi.tendsto_iff atTop_basis_Ioi).mp (NNReal.tendsto_rpow_atTop hy) x trivial
  /-
    case intro.intro
    y : Real
    hy : LT.lt 0 y
    x c : NNReal
    left✝ : True
    hc : ∀ (x_1 : NNReal), Membership.mem (Set.Ioi c) x_1 → Membership.mem (Set.Io …
    ⊢ Filter.Eventually (fun a => LT.lt (↑x) (HPow.hPow a y)) (nhds Top.top)
  -/
  have hc' : Set.Ioi ↑c ∈ 𝓝 (⊤ : ℝ≥0∞) := Ioi_mem_nhds ENNReal.coe_lt_top
  /-
    case intro.intro
    y : Real
    hy : LT.lt 0 y
    x c : NNReal
    left✝ : True
    hc : ∀ (x_1 : NNReal), Membership.mem (Set.Ioi c) x_1 → Membership.mem (Set.Io …
    hc' : Membership.mem (nhds Top.top) (Set.Ioi ↑c)
    ⊢ Filter.Eventually (fun a => LT.lt (↑x) (HPow.hPow a y)) (nhds Top.top)
  -/
  filter_upwards [hc'] with a ha
  /-
    case h
    y : Real
    hy : LT.lt 0 y
    x c : NNReal
    left✝ : True
    hc : ∀ (x_1 : NNReal), Membership.mem (Set.Ioi c) x_1 → Membership.mem (Set.Io …
    hc' : Membership.mem (nhds Top.top) (Set.Ioi ↑c)
    a : ENNReal
    ha : Membership.mem (Set.Ioi ↑c) a
    ⊢ LT.lt (↑x) (HPow.hPow a y)
  -/
  by_cases ha' : a = ⊤
    /-
      case pos
      y : Real
      hy : LT.lt 0 y
      x c : NNReal
      left✝ : True
      hc : ∀ (x_1 : NNReal), Membership.mem (Set.Ioi c) x_1 → Membership.mem (Set.Io …
      hc' : Membership.mem (nhds Top.top) (Set.Ioi ↑c)
      a : ENNReal
      ha : Membership.mem (Set.Ioi ↑c) a
      ha' : Eq a Top.top
      ⊢ LT.lt (↑x) (HPow.hPow a y)
    -/
  · simp [ha', hy]
    /-
      🎉 no goals
    -/
  /-
    case neg
    y : Real
    hy : LT.lt 0 y
    x c : NNReal
    left✝ : True
    hc : ∀ (x_1 : NNReal), Membership.mem (Set.Ioi c) x_1 → Membership.mem (Set.Io …
    hc' : Membership.mem (nhds Top.top) (Set.Ioi ↑c)
    a : ENNReal
    ha : Membership.mem (Set.Ioi ↑c) a
    ha' : Not (Eq a Top.top)
    ⊢ LT.lt (↑x) (HPow.hPow a y)
  -/
  lift a to ℝ≥0 using ha'
  -- Porting note: reduced defeq abuse
  /-
    case neg.intro
    y : Real
    hy : LT.lt 0 y
    x c : NNReal
    left✝ : True
    hc : ∀ (x_1 : NNReal), Membership.mem (Set.Ioi c) x_1 → Membership.mem (Set.Io …
    hc' : Membership.mem (nhds Top.top) (Set.Ioi ↑c)
    a : NNReal
    ha : Membership.mem (Set.Ioi ↑c) ↑a
    ⊢ LT.lt (↑x) (HPow.hPow (↑a) y)
  -/
  simp only [Set.mem_Ioi, coe_lt_coe] at ha hc
  /-
    case neg.intro
    y : Real
    hy : LT.lt 0 y
    x c : NNReal
    left✝ : True
    hc' : Membership.mem (nhds Top.top) (Set.Ioi ↑c)
    a : NNReal
    ha : LT.lt c a
    hc : ∀ (x_1 : NNReal), LT.lt c x_1 → LT.lt x (HPow.hPow x_1 y)
    ⊢ LT.lt (↑x) (HPow.hPow (↑a) y)
  -/
  rw [← ENNReal.coe_rpow_of_nonneg _ hy.le]
  /-
    case neg.intro
    y : Real
    hy : LT.lt 0 y
    x c : NNReal
    left✝ : True
    hc' : Membership.mem (nhds Top.top) (Set.Ioi ↑c)
    a : NNReal
    ha : LT.lt c a
    hc : ∀ (x_1 : NNReal), LT.lt c x_1 → LT.lt x (HPow.hPow x_1 y)
    ⊢ LT.lt ↑x ↑(HPow.hPow a y)
  -/
  exact mod_cast hc a ha
  /-
    🎉 no goals
  -/


theorem isTheta_exp_arg_mul_im (hl : IsBoundedUnder (· ≤ ·) l fun x => |(g x).im|) :
    (fun x => Real.exp (arg (f x) * im (g x))) =Θ[l] fun _ => (1 : ℝ) := by
  /-
    α : Type u_1
    l : Filter α
    f g : α → Complex
    hl : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => _root_.abs (g …
    ⊢ Asymptotics.IsTheta l (fun x => Real.exp (HMul.hMul (f x).arg (g x).im)) fun …
  -/
  rcases hl with ⟨b, hb⟩
  /-
    case intro
    α : Type u_1
    l : Filter α
    f g : α → Complex
    b : Real
    hb : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x b) (Filter.map ( …
    ⊢ Asymptotics.IsTheta l (fun x => Real.exp (HMul.hMul (f x).arg (g x).im)) fun …
  -/
  refine Real.isTheta_exp_comp_one.2 ⟨π * b, ?_⟩
  /-
    case intro
    α : Type u_1
    l : Filter α
    f g : α → Complex
    b : Real
    hb : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x b) (Filter.map ( …
    ⊢ Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x (HMul.hMul Real.pi  …
  -/
  rw [eventually_map] at hb ⊢
  /-
    case intro
    α : Type u_1
    l : Filter α
    f g : α → Complex
    b : Real
    hb : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (_root_.abs (g a). …
    ⊢ Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (_root_.abs (HMul.hMu …
  -/
  refine hb.mono fun x hx => ?_
  /-
    case intro
    α : Type u_1
    l : Filter α
    f g : α → Complex
    b : Real
    hb : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (_root_.abs (g a). …
    x : α
    hx : (fun x1 x2 => LE.le x1 x2) (_root_.abs (g x).im) b
    ⊢ (fun x1 x2 => LE.le x1 x2) (_root_.abs (HMul.hMul (f x).arg (g x).im)) (HMul …
  -/
  rw [abs_mul]
  /-
    case intro
    α : Type u_1
    l : Filter α
    f g : α → Complex
    b : Real
    hb : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (_root_.abs (g a). …
    x : α
    hx : (fun x1 x2 => LE.le x1 x2) (_root_.abs (g x).im) b
    ⊢ (fun x1 x2 => LE.le x1 x2) (HMul.hMul (_root_.abs (f x).arg) (_root_.abs (g  …
  -/
  exact mul_le_mul (abs_arg_le_pi _) hx (abs_nonneg _) Real.pi_pos.le
  /-
    🎉 no goals
  -/


theorem isBigO_cpow_rpow (hl : IsBoundedUnder (· ≤ ·) l fun x => |(g x).im|) :
    (fun x => f x ^ g x) =O[l] fun x => abs (f x) ^ (g x).re :=
  calc
    (fun x => f x ^ g x) =O[l]
        (show α → ℝ from fun x => abs (f x) ^ (g x).re / Real.exp (arg (f x) * im (g x))) :=
      isBigO_of_le _ fun _ => (abs_cpow_le _ _).trans (le_abs_self _)
    _ =Θ[l] (show α → ℝ from fun x => abs (f x) ^ (g x).re / (1 : ℝ)) :=
      ((isTheta_refl _ _).div (isTheta_exp_arg_mul_im hl))
    _ =ᶠ[l] (show α → ℝ from fun x => abs (f x) ^ (g x).re) := by
      /-
        α : Type u_1
        l : Filter α
        f g : α → Complex
        hl : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => _root_.abs (g …
        ⊢ l.EventuallyEq (letFun (fun x => HDiv.hDiv (HPow.hPow (Complex.abs (f x)) (g …
      -/
      simp only [ofReal_one, div_one]
      /-
        α : Type u_1
        l : Filter α
        f g : α → Complex
        hl : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => _root_.abs (g …
        ⊢ l.EventuallyEq (fun x => HPow.hPow (Complex.abs (f x)) (g x).re) fun x => HP …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem isTheta_cpow_rpow (hl_im : IsBoundedUnder (· ≤ ·) l fun x => |(g x).im|)
    (hl : ∀ᶠ x in l, f x = 0 → re (g x) = 0 → g x = 0) :
    (fun x => f x ^ g x) =Θ[l] fun x => abs (f x) ^ (g x).re :=
  calc
    (fun x => f x ^ g x) =Θ[l]
        (show α → ℝ from fun x => abs (f x) ^ (g x).re / Real.exp (arg (f x) * im (g x))) :=
      isTheta_of_norm_eventuallyEq' <| hl.mono fun _ => abs_cpow_of_imp
    _ =Θ[l] (show α → ℝ from fun x => abs (f x) ^ (g x).re / (1 : ℝ)) :=
      ((isTheta_refl _ _).div (isTheta_exp_arg_mul_im hl_im))
    _ =ᶠ[l] (show α → ℝ from fun x => abs (f x) ^ (g x).re) := by
      /-
        α : Type u_1
        l : Filter α
        f g : α → Complex
        hl_im : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => _root_.abs …
        hl : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x).re 0 → Eq (g x) 0) l
        ⊢ l.EventuallyEq (letFun (fun x => HDiv.hDiv (HPow.hPow (Complex.abs (f x)) (g …
      -/
      simp only [ofReal_one, div_one]
      /-
        α : Type u_1
        l : Filter α
        f g : α → Complex
        hl_im : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => _root_.abs …
        hl : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x).re 0 → Eq (g x) 0) l
        ⊢ l.EventuallyEq (fun x => HPow.hPow (Complex.abs (f x)) (g x).re) fun x => HP …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem isTheta_cpow_const_rpow {b : ℂ} (hl : b.re = 0 → b ≠ 0 → ∀ᶠ x in l, f x ≠ 0) :
    (fun x => f x ^ b) =Θ[l] fun x => abs (f x) ^ b.re :=
  isTheta_cpow_rpow isBoundedUnder_const <| by
    -- Porting note: was
    -- simpa only [eventually_imp_distrib_right, Ne.def, ← not_frequently, not_imp_not, Imp.swap]
    --   using hl
    -- but including `Imp.swap` caused an infinite loop
    /-
      α : Type u_1
      l : Filter α
      f : α → Complex
      b : Complex
      hl : Eq b.re 0 → Ne b 0 → Filter.Eventually (fun x => Ne (f x) 0) l
      ⊢ Filter.Eventually (fun x => Eq (f x) 0 → Eq b.re 0 → Eq b 0) l
    -/
    convert hl
    /-
      case a
      α : Type u_1
      l : Filter α
      f : α → Complex
      b : Complex
      hl : Eq b.re 0 → Ne b 0 → Filter.Eventually (fun x => Ne (f x) 0) l
      ⊢ Iff (Filter.Eventually (fun x => Eq (f x) 0 → Eq b.re 0 → Eq b 0) l) (Eq b.r …
    -/
    rw [eventually_imp_distrib_right]
    /-
      case a
      α : Type u_1
      l : Filter α
      f : α → Complex
      b : Complex
      hl : Eq b.re 0 → Ne b 0 → Filter.Eventually (fun x => Ne (f x) 0) l
      ⊢ Iff (Filter.Frequently (fun x => Eq (f x) 0) l → Eq b.re 0 → Eq b 0) (Eq b.r …
    -/
    tauto
    /-
      🎉 no goals
    -/


theorem IsBigOWith.rpow (h : IsBigOWith c l f g) (hc : 0 ≤ c) (hr : 0 ≤ r) (hg : 0 ≤ᶠ[l] g) :
    IsBigOWith (c ^ r) l (fun x => f x ^ r) fun x => g x ^ r := by
  /-
    α : Type u_1
    r c : Real
    l : Filter α
    f g : α → Real
    h : Asymptotics.IsBigOWith c l f g
    hc : LE.le 0 c
    hr : LE.le 0 r
    hg : l.EventuallyLE 0 g
    ⊢ Asymptotics.IsBigOWith (HPow.hPow c r) l (fun x => HPow.hPow (f x) r) fun x  …
  -/
  apply IsBigOWith.of_bound
  /-
    case a
    α : Type u_1
    r c : Real
    l : Filter α
    f g : α → Real
    h : Asymptotics.IsBigOWith c l f g
    hc : LE.le 0 c
    hr : LE.le 0 r
    hg : l.EventuallyLE 0 g
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HPow.hPow (f x) r)) (HMul.hMul …
  -/
  filter_upwards [hg, h.bound] with x hgx hx
  calc
    |f x ^ r| ≤ |f x| ^ r := abs_rpow_le_abs_rpow _ _
    _ ≤ (c * |g x|) ^ r := rpow_le_rpow (abs_nonneg _) hx hr
    _ = c ^ r * |g x ^ r| := by rw [mul_rpow hc (abs_nonneg _), abs_rpow_of_nonneg hgx]


theorem IsBigO.rpow (hr : 0 ≤ r) (hg : 0 ≤ᶠ[l] g) (h : f =O[l] g) :
    (fun x => f x ^ r) =O[l] fun x => g x ^ r :=
  let ⟨_, hc, h'⟩ := h.exists_nonneg
  (h'.rpow hc hr hg).isBigO


theorem IsTheta.rpow (hr : 0 ≤ r) (hf : 0 ≤ᶠ[l] f) (hg : 0 ≤ᶠ[l] g) (h : f =Θ[l] g) :
    (fun x => f x ^ r) =Θ[l] fun x => g x ^ r :=
  ⟨h.1.rpow hr hg, h.2.rpow hr hf⟩


theorem IsLittleO.rpow (hr : 0 < r) (hg : 0 ≤ᶠ[l] g) (h : f =o[l] g) :
    (fun x => f x ^ r) =o[l] fun x => g x ^ r := by
  /-
    α : Type u_1
    r : Real
    l : Filter α
    f g : α → Real
    hr : LT.lt 0 r
    hg : l.EventuallyLE 0 g
    h : Asymptotics.IsLittleO l f g
    ⊢ Asymptotics.IsLittleO l (fun x => HPow.hPow (f x) r) fun x => HPow.hPow (g x …
  -/
  refine .of_isBigOWith fun c hc ↦ ?_
  /-
    α : Type u_1
    r : Real
    l : Filter α
    f g : α → Real
    hr : LT.lt 0 r
    hg : l.EventuallyLE 0 g
    h : Asymptotics.IsLittleO l f g
    c : Real
    hc : LT.lt 0 c
    ⊢ Asymptotics.IsBigOWith c l (fun x => HPow.hPow (f x) r) fun x => HPow.hPow ( …
  -/
  rw [← rpow_inv_rpow hc.le hr.ne']
  /-
    α : Type u_1
    r : Real
    l : Filter α
    f g : α → Real
    hr : LT.lt 0 r
    hg : l.EventuallyLE 0 g
    h : Asymptotics.IsLittleO l f g
    c : Real
    hc : LT.lt 0 c
    ⊢ Asymptotics.IsBigOWith (HPow.hPow (HPow.hPow c (Inv.inv r)) r) l (fun x => H …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  refine (h.forall_isBigOWith ?_).rpow ?_ ?_ hg <;> positivity
                                                    /-
                                                      🎉 no goals
                                                    -/


protected lemma IsBigO.sqrt (hfg : f =O[l] g) (hg : 0 ≤ᶠ[l] g) :
    (Real.sqrt <| f ·) =O[l] (Real.sqrt <| g ·) := by
  /-
    α : Type u_1
    l : Filter α
    f g : α → Real
    hfg : Asymptotics.IsBigO l f g
    hg : l.EventuallyLE 0 g
    ⊢ Asymptotics.IsBigO l (fun x => (f x).sqrt) fun x => (g x).sqrt
  -/
  simpa [Real.sqrt_eq_rpow] using hfg.rpow one_half_pos.le hg
  /-
    🎉 no goals
  -/


protected lemma IsLittleO.sqrt (hfg : f =o[l] g) (hg : 0 ≤ᶠ[l] g) :
    (Real.sqrt <| f ·) =o[l] (Real.sqrt <| g ·) := by
  /-
    α : Type u_1
    l : Filter α
    f g : α → Real
    hfg : Asymptotics.IsLittleO l f g
    hg : l.EventuallyLE 0 g
    ⊢ Asymptotics.IsLittleO l (fun x => (f x).sqrt) fun x => (g x).sqrt
  -/
  simpa [Real.sqrt_eq_rpow] using hfg.rpow one_half_pos hg
  /-
    🎉 no goals
  -/


protected lemma IsTheta.sqrt (hfg : f =Θ[l] g) (hf : 0 ≤ᶠ[l] f) (hg : 0 ≤ᶠ[l] g) :
    (Real.sqrt <| f ·) =Θ[l] (Real.sqrt <| g ·) :=
  ⟨hfg.1.sqrt hg, hfg.2.sqrt hf⟩


/-- `x ^ s = o(exp(b * x))` as `x → ∞` for any real `s` and positive `b`. -/
theorem isLittleO_rpow_exp_pos_mul_atTop (s : ℝ) {b : ℝ} (hb : 0 < b) :
    (fun x : ℝ => x ^ s) =o[atTop] fun x => exp (b * x) :=
  isLittleO_of_tendsto (fun _ h => absurd h (exp_pos _).ne') <| by
    simpa only [div_eq_mul_inv, exp_neg, neg_mul] using
      tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero s b hb


/-- `x ^ k = o(exp(b * x))` as `x → ∞` for any integer `k` and positive `b`. -/
theorem isLittleO_zpow_exp_pos_mul_atTop (k : ℤ) {b : ℝ} (hb : 0 < b) :
    (fun x : ℝ => x ^ k) =o[atTop] fun x => exp (b * x) := by
  /-
    k : Int
    b : Real
    hb : LT.lt 0 b
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HPow.hPow x k) fun x => Real.ex …
  -/
  simpa only [Real.rpow_intCast] using isLittleO_rpow_exp_pos_mul_atTop k hb
  /-
    🎉 no goals
  -/


/-- `x ^ k = o(exp(b * x))` as `x → ∞` for any natural `k` and positive `b`. -/
theorem isLittleO_pow_exp_pos_mul_atTop (k : ℕ) {b : ℝ} (hb : 0 < b) :
    (fun x : ℝ => x ^ k) =o[atTop] fun x => exp (b * x) := by
  /-
    k : Nat
    b : Real
    hb : LT.lt 0 b
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HPow.hPow x k) fun x => Real.ex …
  -/
  simpa using isLittleO_zpow_exp_pos_mul_atTop k hb
  /-
    🎉 no goals
  -/


/-- `x ^ s = o(exp x)` as `x → ∞` for any real `s`. -/
theorem isLittleO_rpow_exp_atTop (s : ℝ) : (fun x : ℝ => x ^ s) =o[atTop] exp := by
  /-
    s : Real
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HPow.hPow x s) Real.exp
  -/
  simpa only [one_mul] using isLittleO_rpow_exp_pos_mul_atTop s one_pos
  /-
    🎉 no goals
  -/


/-- `exp (-a * x) = o(x ^ s)` as `x → ∞`, for any positive `a` and real `s`. -/
theorem isLittleO_exp_neg_mul_rpow_atTop {a : ℝ} (ha : 0 < a) (b : ℝ) :
    IsLittleO atTop (fun x : ℝ => exp (-a * x)) fun x : ℝ => x ^ b := by
  /-
    a : Real
    ha : LT.lt 0 a
    b : Real
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => Real.exp (HMul.hMul (Neg.neg a) …
  -/
  apply isLittleO_of_tendsto'
    /-
      case hgf
      a : Real
      ha : LT.lt 0 a
      b : Real
      ⊢ Filter.Eventually (fun x => Eq (HPow.hPow x b) 0 → Eq (Real.exp (HMul.hMul ( …
    -/
  · refine (eventually_gt_atTop 0).mono fun t ht h => ?_
    /-
      case hgf
      a : Real
      ha : LT.lt 0 a
      b t : Real
      ht : LT.lt 0 t
      h : Eq (HPow.hPow t b) 0
      ⊢ Eq (Real.exp (HMul.hMul (Neg.neg a) t)) 0
    -/
    rw [rpow_eq_zero_iff_of_nonneg ht.le] at h
    /-
      case hgf
      a : Real
      ha : LT.lt 0 a
      b t : Real
      ht : LT.lt 0 t
      h : And (Eq t 0) (Ne b 0)
      ⊢ Eq (Real.exp (HMul.hMul (Neg.neg a) t)) 0
    -/
    exact (ht.ne' h.1).elim
    /-
      🎉 no goals
    -/
    /-
      case a
      a : Real
      ha : LT.lt 0 a
      b : Real
      ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Real.exp (HMul.hMul (Neg.neg a) x)) (HPo …
    -/
  · refine (tendsto_exp_mul_div_rpow_atTop (-b) a ha).inv_tendsto_atTop.congr' ?_
    /-
      case a
      a : Real
      ha : LT.lt 0 a
      b : Real
      ⊢ Filter.atTop.EventuallyEq (Inv.inv fun x => HDiv.hDiv (Real.exp (HMul.hMul a …
    -/
    refine (eventually_ge_atTop 0).mono fun t ht => ?_
    /-
      case a
      a : Real
      ha : LT.lt 0 a
      b t : Real
      ht : LE.le 0 t
      ⊢ Eq (Inv.inv (fun x => HDiv.hDiv (Real.exp (HMul.hMul a x)) (HPow.hPow x (Neg …
    -/
    field_simp [Real.exp_neg, rpow_neg ht]
    /-
      🎉 no goals
    -/


theorem isLittleO_log_rpow_atTop {r : ℝ} (hr : 0 < r) : log =o[atTop] fun x => x ^ r :=
  calc
    log =O[atTop] fun x => r * log x := isBigO_self_const_mul _ hr.ne' _ _
    _ =ᶠ[atTop] fun x => log (x ^ r) :=
      ((eventually_gt_atTop 0).mono fun _ hx => (log_rpow hx _).symm)
    _ =o[atTop] fun x => x ^ r := isLittleO_log_id_atTop.comp_tendsto (tendsto_rpow_atTop hr)


theorem isLittleO_log_rpow_rpow_atTop {s : ℝ} (r : ℝ) (hs : 0 < s) :
    (fun x => log x ^ r) =o[atTop] fun x => x ^ s :=
  let r' := max r 1
  have hr : 0 < r' := lt_max_iff.2 <| Or.inr one_pos
  have H : 0 < s / r' := div_pos hs hr
  calc
    (fun x => log x ^ r) =O[atTop] fun x => log x ^ r' :=
      IsBigO.of_bound 1 <|
        (tendsto_log_atTop.eventually_ge_atTop 1).mono fun x hx => by
          /-
            s r : Real
            hs : LT.lt 0 s
            r' : Real := Max.max r 1
            hr : LT.lt 0 r'
            H : LT.lt 0 (HDiv.hDiv s r')
            x : Real
            hx : LE.le 1 (Real.log x)
            ⊢ LE.le (Norm.norm (HPow.hPow (Real.log x) r)) (HMul.hMul 1 (Norm.norm (HPow.h …
          -/
          have hx₀ : 0 ≤ log x := zero_le_one.trans hx
          simp [r', norm_eq_abs, abs_rpow_of_nonneg, abs_rpow_of_nonneg hx₀,
            rpow_le_rpow_of_exponent_le (hx.trans (le_abs_self _))]
    _ =o[atTop] fun x => (x ^ (s / r')) ^ r' :=
      ((isLittleO_log_rpow_atTop H).rpow hr <|
        (_root_.tendsto_rpow_atTop H).eventually <| eventually_ge_atTop 0)
    _ =ᶠ[atTop] fun x => x ^ s :=
                                                 /-
                                                   s r : Real
                                                   hs : LT.lt 0 s
                                                   r' : Real := Max.max r 1
                                                   hr : LT.lt 0 r'
                                                   H : LT.lt 0 (HDiv.hDiv s r')
                                                   x : Real
                                                   hx : LE.le 0 x
                                                   ⊢ Eq ((fun x => HPow.hPow (HPow.hPow x (HDiv.hDiv s r')) r') x) ((fun x => HPo …
                                                 -/
      (eventually_ge_atTop 0).mono fun x hx ↦ by simp only [← rpow_mul hx, div_mul_cancel₀ _ hr.ne']
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem isLittleO_abs_log_rpow_rpow_nhds_zero {s : ℝ} (r : ℝ) (hs : s < 0) :
    (fun x => |log x| ^ r) =o[𝓝[>] 0] fun x => x ^ s :=
  ((isLittleO_log_rpow_rpow_atTop r (neg_pos.2 hs)).comp_tendsto tendsto_inv_nhdsGT_zero).congr'
    (mem_of_superset (Icc_mem_nhdsGT one_pos) fun x hx => by
      /-
        s r : Real
        hs : LT.lt s 0
        x : Real
        hx : Membership.mem (Set.Icc 0 1) x
        ⊢ Membership.mem (setOf fun x => (fun x => Eq (Function.comp (fun x => HPow.hP …
      -/
      simp [abs_of_nonpos, log_nonpos hx.1 hx.2])
      /-
        🎉 no goals
      -/
    (eventually_mem_nhdsWithin.mono fun x hx => by
      /-
        s r : Real
        hs : LT.lt s 0
        x : Real
        hx : Membership.mem (Set.Ioi 0) x
        ⊢ Eq (Function.comp (fun x => HPow.hPow x (Neg.neg s)) (fun x => Inv.inv x) x) …
      -/
      rw [Function.comp_apply, inv_rpow hx.out.le, rpow_neg hx.out.le, inv_inv])
      /-
        🎉 no goals
      -/


theorem isLittleO_log_rpow_nhds_zero {r : ℝ} (hr : r < 0) : log =o[𝓝[>] 0] fun x => x ^ r :=
  (isLittleO_abs_log_rpow_rpow_nhds_zero 1 hr).neg_left.congr'
    (mem_of_superset (Icc_mem_nhdsGT one_pos) fun x hx => by
      /-
        r : Real
        hr : LT.lt r 0
        x : Real
        hx : Membership.mem (Set.Icc 0 1) x
        ⊢ Membership.mem (setOf fun x => (fun x => Eq ((fun x => Neg.neg (HPow.hPow (a …
      -/
      simp [abs_of_nonpos (log_nonpos hx.1 hx.2)])
      /-
        🎉 no goals
      -/
    .rfl


theorem tendsto_log_div_rpow_nhds_zero {r : ℝ} (hr : r < 0) :
    Tendsto (fun x => log x / x ^ r) (𝓝[>] 0) (𝓝 0) :=
  (isLittleO_log_rpow_nhds_zero hr).tendsto_div_nhds_zero


theorem tendsto_log_mul_rpow_nhds_zero {r : ℝ} (hr : 0 < r) :
    Tendsto (fun x => log x * x ^ r) (𝓝[>] 0) (𝓝 0) :=
  (tendsto_log_div_rpow_nhds_zero <| neg_lt_zero.2 hr).congr' <|
                                                  /-
                                                    r : Real
                                                    hr : LT.lt 0 r
                                                    x : Real
                                                    hx : Membership.mem (Set.Ioi 0) x
                                                    ⊢ Eq (HDiv.hDiv (Real.log x) (HPow.hPow x (Neg.neg r))) ((fun x => HMul.hMul ( …
                                                  -/
    eventually_mem_nhdsWithin.mono fun x hx => by rw [rpow_neg hx.out.le, div_inv_eq_mul]
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma tendsto_log_mul_self_nhds_zero_left : Filter.Tendsto (fun x ↦ log x * x) (𝓝[<] 0) (𝓝 0) := by
  /-
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 (Set.Iio 0) …
  -/
  have h := tendsto_log_mul_rpow_nhds_zero zero_lt_one
  /-
    h : Filter.Tendsto (fun x => HMul.hMul (Real.log x) (HPow.hPow x 1)) (nhdsWith …
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 (Set.Iio 0) …
  -/
  simp only [Real.rpow_one] at h
  have h_eq : ∀ x ∈ Set.Iio 0, (- (fun x ↦ log x * x) ∘ (fun x ↦ |x|)) x = log x * x := by
    simp only [Set.mem_Iio, Pi.neg_apply, Function.comp_apply, log_abs]
    intro x hx
    simp only [abs_of_nonpos hx.le, mul_neg, neg_neg]
  /-
    h : Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 (Set.Ioi  …
    h_eq : ∀ (x : Real), Membership.mem (Set.Iio 0) x → Eq (Neg.neg (Function.comp …
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 (Set.Iio 0) …
  -/
  refine tendsto_nhdsWithin_congr h_eq ?_
  /-
    h : Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 (Set.Ioi  …
    h_eq : ∀ (x : Real), Membership.mem (Set.Iio 0) x → Eq (Neg.neg (Function.comp …
    ⊢ Filter.Tendsto (Neg.neg (Function.comp (fun x => HMul.hMul (Real.log x) x) f …
  -/
  nth_rewrite 3 [← neg_zero]
  /-
    h : Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 (Set.Ioi  …
    h_eq : ∀ (x : Real), Membership.mem (Set.Iio 0) x → Eq (Neg.neg (Function.comp …
    ⊢ Filter.Tendsto (Neg.neg (Function.comp (fun x => HMul.hMul (Real.log x) x) f …
  -/
  refine (h.comp (tendsto_abs_nhdsWithin_zero.mono_left ?_)).neg
  /-
    h : Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 (Set.Ioi  …
    h_eq : ∀ (x : Real), Membership.mem (Set.Iio 0) x → Eq (Neg.neg (Function.comp …
    ⊢ LE.le (nhdsWithin 0 (Set.Iio 0)) (nhdsWithin 0 (HasCompl.compl (Singleton.si …
  -/
  refine nhdsWithin_mono 0 (fun x hx ↦ ?_)
  /-
    h : Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 (Set.Ioi  …
    h_eq : ∀ (x : Real), Membership.mem (Set.Iio 0) x → Eq (Neg.neg (Function.comp …
    x : Real
    hx : Membership.mem (Set.Iio 0) x
    ⊢ Membership.mem (HasCompl.compl (Singleton.singleton 0)) x
  -/
  simp only [Set.mem_Iio] at hx
  /-
    h : Filter.Tendsto (fun x => HMul.hMul (Real.log x) x) (nhdsWithin 0 (Set.Ioi  …
    h_eq : ∀ (x : Real), Membership.mem (Set.Iio 0) x → Eq (Neg.neg (Function.comp …
    x : Real
    hx : LT.lt x 0
    ⊢ Membership.mem (HasCompl.compl (Singleton.singleton 0)) x
  -/
  simp only [Set.mem_compl_iff, Set.mem_singleton_iff, hx.ne, not_false_eq_true]
  /-
    🎉 no goals
  -/

