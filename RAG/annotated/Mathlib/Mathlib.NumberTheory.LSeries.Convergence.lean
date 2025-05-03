/-- The abscissa `x : EReal` of absolute convergence of the L-series associated to `f`:
the series converges absolutely at `s` when `re s > x` and does not converge absolutely
when `re s < x`. -/
noncomputable def LSeries.abscissaOfAbsConv (f : ℕ → ℂ) : EReal :=
  sInf <| Real.toEReal '' {x : ℝ | LSeriesSummable f x}


lemma LSeries.abscissaOfAbsConv_congr {f g : ℕ → ℂ} (h : ∀ {n}, n ≠ 0 → f n = g n) :
    abscissaOfAbsConv f = abscissaOfAbsConv g :=
  congr_arg sInf <| congr_arg _ <| Set.ext fun x ↦ LSeriesSummable_congr x h


open Filter in
/-- If `f` and `g` agree on large `n : ℕ`, then their `LSeries` have the same
abscissa of absolute convergence. -/
lemma LSeries.abscissaOfAbsConv_congr' {f g : ℕ → ℂ} (h : f =ᶠ[atTop] g) :
    abscissaOfAbsConv f = abscissaOfAbsConv g :=
  congr_arg sInf <| congr_arg _ <| Set.ext fun x ↦ LSeriesSummable_congr' x h


lemma LSeriesSummable_of_abscissaOfAbsConv_lt_re {f : ℕ → ℂ} {s : ℂ}
    (hs : abscissaOfAbsConv f < s.re) : LSeriesSummable f s := by
  simp only [abscissaOfAbsConv, sInf_lt_iff, Set.mem_image, Set.mem_setOf_eq,
    exists_exists_and_eq_and, EReal.coe_lt_coe_iff] at hs
  /-
    f : Nat → Complex
    s : Complex
    hs : Exists fun a => And (LSeriesSummable f ↑a) (LT.lt a s.re)
    ⊢ LSeriesSummable f s
  -/
  obtain ⟨y, hy, hys⟩ := hs
  /-
    case intro.intro
    f : Nat → Complex
    s : Complex
    y : Real
    hy : LSeriesSummable f ↑y
    hys : LT.lt y s.re
    ⊢ LSeriesSummable f s
  -/
  exact hy.of_re_le_re <| ofReal_re y ▸ hys.le
  /-
    🎉 no goals
  -/


lemma LSeriesSummable_lt_re_of_abscissaOfAbsConv_lt_re {f : ℕ → ℂ} {s : ℂ}
    (hs : abscissaOfAbsConv f < s.re) :
    ∃ x : ℝ, x < s.re ∧ LSeriesSummable f x := by
  /-
    f : Nat → Complex
    s : Complex
    hs : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    ⊢ Exists fun x => And (LT.lt x s.re) (LSeriesSummable f ↑x)
  -/
  obtain ⟨x, hx₁, hx₂⟩ := EReal.exists_between_coe_real hs
  /-
    case intro.intro
    f : Nat → Complex
    s : Complex
    hs : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    x : Real
    hx₁ : LT.lt (LSeries.abscissaOfAbsConv f) ↑x
    hx₂ : LT.lt ↑x ↑s.re
    ⊢ Exists fun x => And (LT.lt x s.re) (LSeriesSummable f ↑x)
  -/
  exact ⟨x, EReal.coe_lt_coe_iff.mp hx₂, LSeriesSummable_of_abscissaOfAbsConv_lt_re hx₁⟩
  /-
    🎉 no goals
  -/


lemma LSeriesSummable.abscissaOfAbsConv_le {f : ℕ → ℂ} {s : ℂ} (h : LSeriesSummable f s) :
    abscissaOfAbsConv f ≤ s.re := by
  /-
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    ⊢ LE.le (LSeries.abscissaOfAbsConv f) ↑s.re
  -/
  refine sInf_le <| Membership.mem.out ?_
  /-
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    ⊢ Membership.mem (setOf fun x => Membership.mem (Set.image Real.toEReal (setOf …
  -/
  simp only [Set.mem_setOf_eq, Set.mem_image, EReal.coe_eq_coe_iff, exists_eq_right]
  /-
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    ⊢ LSeriesSummable f ↑s.re
  -/
  exact h.of_re_le_re <| by simp only [ofReal_re, le_refl]
  /-
    🎉 no goals
  -/


lemma LSeries.abscissaOfAbsConv_le_of_forall_lt_LSeriesSummable {f : ℕ → ℂ} {x : ℝ}
    (h : ∀ y : ℝ, x < y → LSeriesSummable f y) :
    abscissaOfAbsConv f ≤ x := by
  /-
    f : Nat → Complex
    x : Real
    h : ∀ (y : Real), LT.lt x y → LSeriesSummable f ↑y
    ⊢ LE.le (LSeries.abscissaOfAbsConv f) ↑x
  -/
  refine sInf_le_iff.mpr fun y hy ↦ ?_
  simp only [mem_lowerBounds, Set.mem_image, Set.mem_setOf_eq, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂] at hy
  have H (a : EReal) : x < a → y ≤ a := by
    induction' a with a₀
    · simp only [not_lt_bot, le_bot_iff, IsEmpty.forall_iff]
    · exact_mod_cast fun ha ↦ hy a₀ (h a₀ ha)
    · simp only [EReal.coe_lt_top, le_top, forall_true_left]
  /-
    f : Nat → Complex
    x : Real
    h : ∀ (y : Real), LT.lt x y → LSeriesSummable f ↑y
    y : EReal
    hy : ∀ (a : Real), LSeriesSummable f ↑a → LE.le y ↑a
    H : ∀ (a : EReal), LT.lt (↑x) a → LE.le y a
    ⊢ LE.le y ↑x
  -/
  exact Set.Ioi_subset_Ici_iff.mp H
  /-
    🎉 no goals
  -/


lemma LSeries.abscissaOfAbsConv_le_of_forall_lt_LSeriesSummable' {f : ℕ → ℂ} {x : EReal}
    (h : ∀ y : ℝ, x < y → LSeriesSummable f y) :
    abscissaOfAbsConv f ≤ x := by
  /-
    f : Nat → Complex
    x : EReal
    h : ∀ (y : Real), LT.lt x ↑y → LSeriesSummable f ↑y
    ⊢ LE.le (LSeries.abscissaOfAbsConv f) x
  -/
  induction' x with y
    /-
      case h_bot
      f : Nat → Complex
      h : ∀ (y : Real), LT.lt Bot.bot ↑y → LSeriesSummable f ↑y
      ⊢ LE.le (LSeries.abscissaOfAbsConv f) Bot.bot
    -/
  · refine le_of_eq <| sInf_eq_bot.mpr fun y hy ↦ ?_
    /-
      case h_bot
      f : Nat → Complex
      h : ∀ (y : Real), LT.lt Bot.bot ↑y → LSeriesSummable f ↑y
      y : EReal
      hy : GT.gt y Bot.bot
      ⊢ Exists fun a => And (Membership.mem (Set.image Real.toEReal (setOf fun x =>  …
    -/
    induction' y with z
      /-
        case h_bot.h_bot
        f : Nat → Complex
        h : ∀ (y : Real), LT.lt Bot.bot ↑y → LSeriesSummable f ↑y
        hy : GT.gt Bot.bot Bot.bot
        ⊢ Exists fun a => And (Membership.mem (Set.image Real.toEReal (setOf fun x =>  …
      -/
    · simp only [gt_iff_lt, lt_self_iff_false] at hy
      /-
        🎉 no goals
      -/
      /-
        case h_bot.h_real
        f : Nat → Complex
        h : ∀ (y : Real), LT.lt Bot.bot ↑y → LSeriesSummable f ↑y
        z : Real
        hy : GT.gt (↑z) Bot.bot
        ⊢ Exists fun a => And (Membership.mem (Set.image Real.toEReal (setOf fun x =>  …
      -/
    · exact ⟨z - 1,  ⟨z-1, h (z - 1) <| EReal.bot_lt_coe _, rfl⟩, by norm_cast; exact sub_one_lt z⟩
      /-
        🎉 no goals
      -/
      /-
        case h_bot.h_top
        f : Nat → Complex
        h : ∀ (y : Real), LT.lt Bot.bot ↑y → LSeriesSummable f ↑y
        hy : GT.gt Top.top Bot.bot
        ⊢ Exists fun a => And (Membership.mem (Set.image Real.toEReal (setOf fun x =>  …
      -/
    · exact ⟨0, ⟨0, h 0 <| EReal.bot_lt_coe 0, rfl⟩, EReal.zero_lt_top⟩
      /-
        🎉 no goals
      -/
    /-
      case h_real
      f : Nat → Complex
      y : Real
      h : ∀ (y_1 : Real), LT.lt ↑y ↑y_1 → LSeriesSummable f ↑y_1
      ⊢ LE.le (LSeries.abscissaOfAbsConv f) ↑y
    -/
  · exact abscissaOfAbsConv_le_of_forall_lt_LSeriesSummable <| by exact_mod_cast h
    /-
      🎉 no goals
    -/
    /-
      case h_top
      f : Nat → Complex
      h : ∀ (y : Real), LT.lt Top.top ↑y → LSeriesSummable f ↑y
      ⊢ LE.le (LSeries.abscissaOfAbsConv f) Top.top
    -/
  · exact le_top
    /-
      🎉 no goals
    -/


/-- If `‖f n‖` is bounded by a constant times `n^x`, then the abscissa of absolute convergence
of `f` is bounded by `x + 1`. -/
lemma LSeries.abscissaOfAbsConv_le_of_le_const_mul_rpow {f : ℕ → ℂ} {x : ℝ}
    (h : ∃ C, ∀ n ≠ 0, ‖f n‖ ≤ C * n ^ x) : abscissaOfAbsConv f ≤ x + 1 := by
  /-
    f : Nat → Complex
    x : Real
    h : Exists fun C => ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C …
    ⊢ LE.le (LSeries.abscissaOfAbsConv f) (HAdd.hAdd (↑x) 1)
  -/
  rw [show x = x + 1 - 1 by ring] at h
  /-
    f : Nat → Complex
    x : Real
    h : Exists fun C => ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C …
    ⊢ LE.le (LSeries.abscissaOfAbsConv f) (HAdd.hAdd (↑x) 1)
  -/
  by_contra! H
  /-
    f : Nat → Complex
    x : Real
    h : Exists fun C => ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C …
    H : LT.lt (HAdd.hAdd (↑x) 1) (LSeries.abscissaOfAbsConv f)
    ⊢ False
  -/
  obtain ⟨y, hy₁, hy₂⟩ := EReal.exists_between_coe_real H
  exact (LSeriesSummable_of_le_const_mul_rpow (s := y) (EReal.coe_lt_coe_iff.mp hy₁) h
    |>.abscissaOfAbsConv_le.trans_lt hy₂).false


open Filter in
/-- If `‖f n‖` is `O(n^x)`, then the abscissa of absolute convergence
of `f` is bounded by `x + 1`. -/
lemma LSeries.abscissaOfAbsConv_le_of_isBigO_rpow {f : ℕ → ℂ} {x : ℝ}
    (h : f =O[atTop] fun n ↦ (n : ℝ) ^ x) :
    abscissaOfAbsConv f ≤ x + 1 := by
  /-
    f : Nat → Complex
    x : Real
    h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) x
    ⊢ LE.le (LSeries.abscissaOfAbsConv f) (HAdd.hAdd (↑x) 1)
  -/
  rw [show x = x + 1 - 1 by ring] at h
  /-
    f : Nat → Complex
    x : Real
    h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub (HAdd …
    ⊢ LE.le (LSeries.abscissaOfAbsConv f) (HAdd.hAdd (↑x) 1)
  -/
  by_contra! H
  /-
    f : Nat → Complex
    x : Real
    h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub (HAdd …
    H : LT.lt (HAdd.hAdd (↑x) 1) (LSeries.abscissaOfAbsConv f)
    ⊢ False
  -/
  obtain ⟨y, hy₁, hy₂⟩ := EReal.exists_between_coe_real H
  exact (LSeriesSummable_of_isBigO_rpow (s := y) (EReal.coe_lt_coe_iff.mp hy₁) h
    |>.abscissaOfAbsConv_le.trans_lt hy₂).false


/-- If `f` is bounded, then the abscissa of absolute convergence of `f` is bounded above by `1`. -/
lemma LSeries.abscissaOfAbsConv_le_of_le_const {f : ℕ → ℂ} (h : ∃ C, ∀ n ≠ 0, ‖f n‖ ≤ C) :
    abscissaOfAbsConv f ≤ 1 := by
  /-
    f : Nat → Complex
    h : Exists fun C => ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) C
    ⊢ LE.le (LSeries.abscissaOfAbsConv f) 1
  -/
  convert abscissaOfAbsConv_le_of_le_const_mul_rpow (x := 0) ?_
    /-
      case h.e'_4
      f : Nat → Complex
      h : Exists fun C => ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) C
      ⊢ Eq 1 (HAdd.hAdd (↑0) 1)
    -/
  · simp only [EReal.coe_zero, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      f : Nat → Complex
      h : Exists fun C => ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) C
      ⊢ Exists fun C => ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C ( …
    -/
  · simpa only [norm_eq_abs, Real.rpow_zero, mul_one] using h
    /-
      🎉 no goals
    -/


open Filter in
/-- If `f` is `O(1)`, then the abscissa of absolute convergence of `f` is bounded above by `1`. -/
lemma LSeries.abscissaOfAbsConv_le_one_of_isBigO_one {f : ℕ → ℂ} (h : f =O[atTop] (1 : ℕ → ℝ)) :
    abscissaOfAbsConv f ≤ 1 := by
  /-
    f : Nat → Complex
    h : Asymptotics.IsBigO Filter.atTop f 1
    ⊢ LE.le (LSeries.abscissaOfAbsConv f) 1
  -/
  convert abscissaOfAbsConv_le_of_isBigO_rpow (x := 0) ?_
    /-
      case h.e'_4
      f : Nat → Complex
      h : Asymptotics.IsBigO Filter.atTop f 1
      ⊢ Eq 1 (HAdd.hAdd (↑0) 1)
    -/
  · simp only [EReal.coe_zero, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      f : Nat → Complex
      h : Asymptotics.IsBigO Filter.atTop f 1
      ⊢ Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) 0
    -/
  · simpa only [Real.rpow_zero] using h
    /-
      🎉 no goals
    -/


/-- If `f` is real-valued and `x` is strictly greater than the abscissa of absolute convergence
of `f`, then the real series `∑' n, f n / n ^ x` converges. -/
lemma LSeries.summable_real_of_abscissaOfAbsConv_lt {f : ℕ → ℝ} {x : ℝ}
    (h : abscissaOfAbsConv (f ·) < x) :
    Summable fun n : ℕ ↦ f n / (n : ℝ) ^ x := by
  /-
    f : Nat → Real
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑x
    ⊢ Summable fun n => HDiv.hDiv (f n) (HPow.hPow (↑n) x)
  -/
  have h' : abscissaOfAbsConv (f ·) < (x : ℂ).re := by simpa only [ofReal_re] using h
  /-
    f : Nat → Real
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑x
    h' : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑(↑x).re
    ⊢ Summable fun n => HDiv.hDiv (f n) (HPow.hPow (↑n) x)
  -/
  have := LSeriesSummable_of_abscissaOfAbsConv_lt_re h'
  /-
    f : Nat → Real
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑x
    h' : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑(↑x).re
    this : LSeriesSummable (fun x => ↑(f x)) ↑x
    ⊢ Summable fun n => HDiv.hDiv (f n) (HPow.hPow (↑n) x)
  -/
  rw [LSeriesSummable, show term _ _ = fun n ↦ _ from rfl] at this
  conv at this =>
    enter [1, n]
    rw [term_def, ← ofReal_natCast, ← ofReal_cpow n.cast_nonneg, ← ofReal_div, ← ofReal_zero,
      ← apply_ite]
  /-
    f : Nat → Real
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑x
    h' : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑(↑x).re
    this : Summable fun n => ↑(ite (Eq n 0) 0 (HDiv.hDiv (f n) (HPow.hPow (↑n) x)))
    ⊢ Summable fun n => HDiv.hDiv (f n) (HPow.hPow (↑n) x)
  -/
  rw [summable_ofReal] at this
  /-
    f : Nat → Real
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑x
    h' : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑(↑x).re
    this : Summable fun n => ite (Eq n 0) 0 (HDiv.hDiv (f n) (HPow.hPow (↑n) x))
    ⊢ Summable fun n => HDiv.hDiv (f n) (HPow.hPow (↑n) x)
  -/
  refine this.congr_cofinite ?_
  /-
    f : Nat → Real
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑x
    h' : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑(↑x).re
    this : Summable fun n => ite (Eq n 0) 0 (HDiv.hDiv (f n) (HPow.hPow (↑n) x))
    ⊢ Filter.cofinite.EventuallyEq (fun n => ite (Eq n 0) 0 (HDiv.hDiv (f n) (HPow …
  -/
  filter_upwards [Set.Finite.compl_mem_cofinite <| Set.finite_singleton 0] with n hn
  /-
    case h
    f : Nat → Real
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑x
    h' : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑(↑x).re
    this : Summable fun n => ite (Eq n 0) 0 (HDiv.hDiv (f n) (HPow.hPow (↑n) x))
    n : Nat
    hn : Membership.mem (HasCompl.compl (Singleton.singleton 0)) n
    ⊢ Eq (ite (Eq n 0) 0 (HDiv.hDiv (f n) (HPow.hPow (↑n) x))) (HDiv.hDiv (f n) (H …
  -/
  simp only [Set.mem_compl_iff, Set.mem_singleton_iff] at hn
  /-
    case h
    f : Nat → Real
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑x
    h' : LT.lt (LSeries.abscissaOfAbsConv fun x => ↑(f x)) ↑(↑x).re
    this : Summable fun n => ite (Eq n 0) 0 (HDiv.hDiv (f n) (HPow.hPow (↑n) x))
    n : Nat
    hn : Not (Eq n 0)
    ⊢ Eq (ite (Eq n 0) 0 (HDiv.hDiv (f n) (HPow.hPow (↑n) x))) (HDiv.hDiv (f n) (H …
  -/
  exact if_neg hn
  /-
    🎉 no goals
  -/


/-- If `F` is a binary operation on `ℕ → ℂ` with the property that the `LSeries` of `F f g`
converges whenever the `LSeries` of `f` and `g` do, then the abscissa of absolute convergence
of `F f g` is at most the maximum of the abscissa of absolute convergence of `f`
and that of `g`. -/
lemma LSeries.abscissaOfAbsConv_binop_le {F : (ℕ → ℂ) → (ℕ → ℂ) → (ℕ → ℂ)}
    (hF : ∀ {f g s}, LSeriesSummable f s → LSeriesSummable g s → LSeriesSummable (F f g) s)
    (f g : ℕ → ℂ) :
    abscissaOfAbsConv (F f g) ≤ max (abscissaOfAbsConv f) (abscissaOfAbsConv g) := by
  /-
    F : (Nat → Complex) → (Nat → Complex) → Nat → Complex
    hF : ∀ {f g : Nat → Complex} {s : Complex}, LSeriesSummable f s → LSeriesSumma …
    f g : Nat → Complex
    ⊢ LE.le (LSeries.abscissaOfAbsConv (F f g)) (Max.max (LSeries.abscissaOfAbsCon …
  -/
  refine abscissaOfAbsConv_le_of_forall_lt_LSeriesSummable' fun x hx ↦  hF ?_ ?_
  · exact LSeriesSummable_of_abscissaOfAbsConv_lt_re <|
      (ofReal_re x).symm ▸ (le_max_left ..).trans_lt hx
  · exact LSeriesSummable_of_abscissaOfAbsConv_lt_re <|
      (ofReal_re x).symm ▸ (le_max_right ..).trans_lt hx

