/-- The logarithm function defined on the extended nonnegative reals `ℝ≥0∞`
to the extended reals `EReal`. Coincides with the usual logarithm function
and with `Real.log` on positive reals, and takes values `log 0 = ⊥` and `log ⊤ = ⊤`.
Conventions about multiplication in `ℝ≥0∞` and addition in `EReal` make the identity
`log (x * y) = log x + log y` unconditional. -/
noncomputable def log (x : ℝ≥0∞) : EReal :=
  if x = 0 then ⊥
    else if x = ⊤ then ⊤
    else Real.log x.toReal


@[simp] lemma log_zero : log 0 = ⊥ := if_pos rfl

                                        /-
                                          ⊢ Eq (ENNReal.log 1) 0
                                        -/
@[simp] lemma log_one : log 1 = 0 := by simp [log]
                                        /-
                                          🎉 no goals
                                        -/

@[simp] lemma log_top : log ⊤ = ⊤ := rfl


@[simp]
lemma log_ofReal (x : ℝ) : log (ENNReal.ofReal x) = if x ≤ 0 then ⊥ else ↑(Real.log x) := by
  simp only [log, ENNReal.none_eq_top, ENNReal.ofReal_ne_top, IsEmpty.forall_iff,
    ENNReal.ofReal_eq_zero, EReal.coe_ennreal_ofReal, if_false]
  /-
    x : Real
    ⊢ Eq (ite (LE.le x 0) Bot.bot ↑(Real.log (ENNReal.ofReal x).toReal)) (ite (LE. …
  -/
  split_ifs with h_nonpos
    /-
      case pos
      x : Real
      h_nonpos : LE.le x 0
      ⊢ Eq Bot.bot Bot.bot
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      x : Real
      h_nonpos : Not (LE.le x 0)
      ⊢ Eq ↑(Real.log (ENNReal.ofReal x).toReal) ↑(Real.log x)
    -/
  · rw [ENNReal.toReal_ofReal (not_le.mp h_nonpos).le]
    /-
      🎉 no goals
    -/


lemma log_ofReal_of_pos {x : ℝ} (hx : 0 < x) : log (ENNReal.ofReal x) = Real.log x := by
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (ENNReal.ofReal x).log ↑(Real.log x)
  -/
  rw [log_ofReal, if_neg hx.not_le]
  /-
    🎉 no goals
  -/


theorem log_pos_real {x : ℝ≥0∞} (h : x ≠ 0) (h' : x ≠ ⊤) :
                                              /-
                                                x : ENNReal
                                                h : Ne x 0
                                                h' : Ne x Top.top
                                                ⊢ Eq x.log ↑(Real.log x.toReal)
                                              -/
    log x = Real.log (ENNReal.toReal x) := by simp [log, h, h']
                                              /-
                                                🎉 no goals
                                              -/


theorem log_pos_real' {x : ℝ≥0∞} (h : 0 < x.toReal) :
    log x = Real.log (ENNReal.toReal x) := by
  /-
    x : ENNReal
    h : LT.lt 0 x.toReal
    ⊢ Eq x.log ↑(Real.log x.toReal)
  -/
  simp [log, (ENNReal.toReal_pos_iff.1 h).1.ne', (ENNReal.toReal_pos_iff.1 h).2.ne]
  /-
    🎉 no goals
  -/


theorem log_of_nnreal {x : ℝ≥0} (h : x ≠ 0) :
                                      /-
                                        x : NNReal
                                        h : Ne x 0
                                        ⊢ Eq (↑x).log ↑(Real.log ↑x)
                                      -/
    log (x : ℝ≥0∞) = Real.log x := by simp [log, h]
                                      /-
                                        🎉 no goals
                                      -/


theorem log_strictMono : StrictMono log := by
  /-
    ⊢ StrictMono ENNReal.log
  -/
  intro x y h
  /-
    x y : ENNReal
    h : LT.lt x y
    ⊢ LT.lt x.log y.log
  -/
  unfold log
  /-
    x y : ENNReal
    h : LT.lt x y
    ⊢ LT.lt (ite (Eq x 0) Bot.bot (ite (Eq x Top.top) Top.top ↑(Real.log x.toReal) …
  -/
  rcases ENNReal.trichotomy x with (rfl | rfl | x_real)
    /-
      case inl
      y : ENNReal
      h : LT.lt 0 y
      ⊢ LT.lt (ite (Eq 0 0) Bot.bot (ite (Eq 0 Top.top) Top.top ↑(Real.log (ENNReal. …
    -/
  · rcases ENNReal.trichotomy y with (rfl | rfl | y_real)
      /-
        case inl.inl
        h : LT.lt 0 0
        ⊢ LT.lt (ite (Eq 0 0) Bot.bot (ite (Eq 0 Top.top) Top.top ↑(Real.log (ENNReal. …
      -/
    · exfalso; exact lt_irrefl 0 h
               /-
                 🎉 no goals
               -/
      /-
        case inl.inr.inl
        h : LT.lt 0 Top.top
        ⊢ LT.lt (ite (Eq 0 0) Bot.bot (ite (Eq 0 Top.top) Top.top ↑(Real.log (ENNReal. …
      -/
    · simp
      /-
        🎉 no goals
      -/
    · simp [(ENNReal.toReal_pos_iff.1 y_real).1.ne',
        (ENNReal.toReal_pos_iff.1 y_real).2.ne, EReal.bot_lt_coe]
    /-
      case inr.inl
      y : ENNReal
      h : LT.lt Top.top y
      ⊢ LT.lt (ite (Eq Top.top 0) Bot.bot (ite (Eq Top.top Top.top) Top.top ↑(Real.l …
    -/
  · exfalso; exact not_top_lt h
             /-
               🎉 no goals
             -/
  · simp only [(ENNReal.toReal_pos_iff.1 x_real).1.ne',
      (ENNReal.toReal_pos_iff.1 x_real).2.ne, if_false]
    /-
      case inr.inr
      x y : ENNReal
      h : LT.lt x y
      x_real : LT.lt 0 x.toReal
      ⊢ LT.lt (↑(Real.log x.toReal)) (ite (Eq y 0) Bot.bot (ite (Eq y Top.top) Top.t …
    -/
    rcases ENNReal.trichotomy y with (rfl | rfl | y_real)
      /-
        case inr.inr.inl
        x : ENNReal
        x_real : LT.lt 0 x.toReal
        h : LT.lt x 0
        ⊢ LT.lt (↑(Real.log x.toReal)) (ite (Eq 0 0) Bot.bot (ite (Eq 0 Top.top) Top.t …
      -/
    · exfalso; rw [← ENNReal.bot_eq_zero] at h; exact not_lt_bot h
                                                /-
                                                  🎉 no goals
                                                -/
      /-
        case inr.inr.inr.inl
        x : ENNReal
        x_real : LT.lt 0 x.toReal
        h : LT.lt x Top.top
        ⊢ LT.lt (↑(Real.log x.toReal)) (ite (Eq Top.top 0) Bot.bot (ite (Eq Top.top To …
      -/
    · simp
      /-
        🎉 no goals
      -/
    · simp only [(ENNReal.toReal_pos_iff.1 y_real).1.ne', ↓reduceIte,
        (ENNReal.toReal_pos_iff.1 y_real).2.ne, EReal.coe_lt_coe_iff]
      /-
        case inr.inr.inr.inr
        x y : ENNReal
        h : LT.lt x y
        x_real : LT.lt 0 x.toReal
        y_real : LT.lt 0 y.toReal
        ⊢ LT.lt (Real.log x.toReal) (Real.log y.toReal)
      -/
      apply Real.log_lt_log x_real
      exact (ENNReal.toReal_lt_toReal (ENNReal.toReal_pos_iff.1 x_real).2.ne
        (ENNReal.toReal_pos_iff.1 y_real).2.ne).2 h


theorem log_monotone : Monotone log := log_strictMono.monotone


theorem log_injective : Function.Injective log := log_strictMono.injective


theorem log_surjective : Function.Surjective log := by
  /-
    ⊢ Function.Surjective ENNReal.log
  -/
  intro y
  /-
    y : EReal
    ⊢ Exists fun a => Eq a.log y
  -/
  cases' eq_bot_or_bot_lt y with y_bot y_nbot
    /-
      case inl
      y : EReal
      y_bot : Eq y Bot.bot
      ⊢ Exists fun a => Eq a.log y
    -/
  · exact y_bot ▸ ⟨⊥, log_zero⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    y : EReal
    y_nbot : LT.lt Bot.bot y
    ⊢ Exists fun a => Eq a.log y
  -/
  cases' eq_top_or_lt_top y with y_top y_ntop
    /-
      case inr.inl
      y : EReal
      y_nbot : LT.lt Bot.bot y
      y_top : Eq y Top.top
      ⊢ Exists fun a => Eq a.log y
    -/
  · exact y_top ▸ ⟨⊤, log_top⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    y : EReal
    y_nbot : LT.lt Bot.bot y
    y_ntop : LT.lt y Top.top
    ⊢ Exists fun a => Eq a.log y
  -/
  use ENNReal.ofReal (Real.exp y.toReal)
  /-
    case h
    y : EReal
    y_nbot : LT.lt Bot.bot y
    y_ntop : LT.lt y Top.top
    ⊢ Eq (ENNReal.ofReal (Real.exp y.toReal)).log y
  -/
  have exp_y_pos := not_le_of_lt (Real.exp_pos y.toReal)
  simp only [log, ofReal_eq_zero, exp_y_pos, ↓reduceIte, ofReal_ne_top,
    ENNReal.toReal_ofReal (Real.exp_pos y.toReal).le, Real.log_exp y.toReal]
  /-
    case h
    y : EReal
    y_nbot : LT.lt Bot.bot y
    y_ntop : LT.lt y Top.top
    exp_y_pos : Not (LE.le (Real.exp y.toReal) 0)
    ⊢ Eq (↑y.toReal) y
  -/
  exact EReal.coe_toReal y_ntop.ne y_nbot.ne'
  /-
    🎉 no goals
  -/


theorem log_bijective : Function.Bijective log := ⟨log_injective, log_surjective⟩


@[simp]
theorem log_eq_iff {x y : ℝ≥0∞} : log x = log y ↔ x = y :=
  log_injective.eq_iff


@[simp] theorem log_eq_bot_iff {x : ℝ≥0∞} : log x = ⊥ ↔ x = 0 := log_zero ▸ @log_eq_iff x 0


@[simp] theorem log_eq_one_iff {x : ℝ≥0∞} : log x = 0 ↔ x = 1 := log_one ▸ @log_eq_iff x 1


@[simp] theorem log_eq_top_iff {x : ℝ≥0∞} : log x = ⊤ ↔ x = ⊤ := log_top ▸ @log_eq_iff x ⊤


@[simp] lemma log_lt_log_iff {x y : ℝ≥0∞} : log x < log y ↔ x < y := log_strictMono.lt_iff_lt


@[simp] lemma bot_lt_log_iff {x : ℝ≥0∞} : ⊥ < log x ↔ 0 < x := log_zero ▸ @log_lt_log_iff 0 x


@[simp] lemma log_lt_top_iff {x : ℝ≥0∞} : log x < ⊤ ↔ x < ⊤ := log_top ▸ @log_lt_log_iff x ⊤


@[simp] lemma log_lt_zero_iff {x : ℝ≥0∞} : log x < 0 ↔ x < 1 := log_one ▸ @log_lt_log_iff x 1


@[simp] lemma zero_lt_log_iff {x : ℝ≥0∞} : 0 < log x ↔ 1 < x := log_one ▸ @log_lt_log_iff 1 x


@[simp] lemma log_le_log_iff {x y : ℝ≥0∞} : log x ≤ log y ↔ x ≤ y := log_strictMono.le_iff_le


@[simp] lemma log_le_zero_iff {x : ℝ≥0∞} : log x ≤ 0 ↔ x ≤ 1 := log_one ▸ @log_le_log_iff x 1


@[simp] lemma zero_le_log_iff {x : ℝ≥0∞} : 0 ≤ log x ↔ 1 ≤ x := log_one ▸ @log_le_log_iff 1 x


theorem log_mul_add {x y : ℝ≥0∞} : log (x * y) = log x + log y := by
  /-
    x y : ENNReal
    ⊢ Eq (HMul.hMul x y).log (HAdd.hAdd x.log y.log)
  -/
  rcases ENNReal.trichotomy x with (rfl | rfl | x_real)
    /-
      case inl
      y : ENNReal
      ⊢ Eq (HMul.hMul 0 y).log (HAdd.hAdd (ENNReal.log 0) y.log)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      y : ENNReal
      ⊢ Eq (HMul.hMul Top.top y).log (HAdd.hAdd Top.top.log y.log)
    -/
  · rw [log_top]
    /-
      case inr.inl
      y : ENNReal
      ⊢ Eq (HMul.hMul Top.top y).log (HAdd.hAdd Top.top y.log)
    -/
    rcases ENNReal.trichotomy y with (rfl | rfl | y_real)
      /-
        case inr.inl.inl
        ⊢ Eq (HMul.hMul Top.top 0).log (HAdd.hAdd Top.top (ENNReal.log 0))
      -/
    · rw [mul_zero, log_zero, EReal.add_bot]
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.inr.inl
        ⊢ Eq (HMul.hMul Top.top Top.top).log (HAdd.hAdd Top.top Top.top.log)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.inr.inr
        y : ENNReal
        y_real : LT.lt 0 y.toReal
        ⊢ Eq (HMul.hMul Top.top y).log (HAdd.hAdd Top.top y.log)
      -/
    · rw [log_pos_real' y_real, ENNReal.top_mul', EReal.top_add_coe, log_eq_top_iff]
      /-
        case inr.inl.inr.inr
        y : ENNReal
        y_real : LT.lt 0 y.toReal
        ⊢ Eq (ite (Eq y 0) 0 Top.top) Top.top
      -/
      simp only [ite_eq_right_iff, zero_ne_top, imp_false]
      /-
        case inr.inl.inr.inr
        y : ENNReal
        y_real : LT.lt 0 y.toReal
        ⊢ Not (Eq y 0)
      -/
      exact (ENNReal.toReal_pos_iff.1 y_real).1.ne'
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      x y : ENNReal
      x_real : LT.lt 0 x.toReal
      ⊢ Eq (HMul.hMul x y).log (HAdd.hAdd x.log y.log)
    -/
  · rw [log_pos_real' x_real]
    /-
      case inr.inr
      x y : ENNReal
      x_real : LT.lt 0 x.toReal
      ⊢ Eq (HMul.hMul x y).log (HAdd.hAdd (↑(Real.log x.toReal)) y.log)
    -/
    rcases ENNReal.trichotomy y with (rfl | rfl | y_real)
      /-
        case inr.inr.inl
        x : ENNReal
        x_real : LT.lt 0 x.toReal
        ⊢ Eq (HMul.hMul x 0).log (HAdd.hAdd (↑(Real.log x.toReal)) (ENNReal.log 0))
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.inr.inl
        x : ENNReal
        x_real : LT.lt 0 x.toReal
        ⊢ Eq (HMul.hMul x Top.top).log (HAdd.hAdd (↑(Real.log x.toReal)) Top.top.log)
      -/
    · simp [(ENNReal.toReal_pos_iff.1 x_real).1.ne']
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.inr.inr
        x y : ENNReal
        x_real : LT.lt 0 x.toReal
        y_real : LT.lt 0 y.toReal
        ⊢ Eq (HMul.hMul x y).log (HAdd.hAdd (↑(Real.log x.toReal)) y.log)
      -/
    · rw_mod_cast [log_pos_real', log_pos_real' y_real, ENNReal.toReal_mul]
        /-
          case inr.inr.inr.inr
          x y : ENNReal
          x_real : LT.lt 0 x.toReal
          y_real : LT.lt 0 y.toReal
          ⊢ Eq (Real.log (HMul.hMul x.toReal y.toReal)) (HAdd.hAdd (Real.log x.toReal) ( …
        -/
      · exact Real.log_mul x_real.ne' y_real.ne'
        /-
          🎉 no goals
        -/
      /-
        case inr.inr.inr.inr
        x y : ENNReal
        x_real : LT.lt 0 x.toReal
        y_real : LT.lt 0 y.toReal
        ⊢ LT.lt 0 (HMul.hMul x y).toReal
      -/
      rw [toReal_mul]
      /-
        case inr.inr.inr.inr
        x y : ENNReal
        x_real : LT.lt 0 x.toReal
        y_real : LT.lt 0 y.toReal
        ⊢ LT.lt 0 (HMul.hMul x.toReal y.toReal)
      -/
      positivity
      /-
        🎉 no goals
      -/


theorem log_pow {x : ℝ≥0∞} {n : ℕ} : log (x ^ n) = n * log x := by
  /-
    x : ENNReal
    n : Nat
    ⊢ Eq (HPow.hPow x n).log (HMul.hMul (↑n) x.log)
  -/
  cases' Nat.eq_zero_or_pos n with n_zero n_pos
    /-
      case inl
      x : ENNReal
      n : Nat
      n_zero : Eq n 0
      ⊢ Eq (HPow.hPow x n).log (HMul.hMul (↑n) x.log)
    -/
  · simp [n_zero, pow_zero x]
    /-
      🎉 no goals
    -/
  /-
    case inr
    x : ENNReal
    n : Nat
    n_pos : GT.gt n 0
    ⊢ Eq (HPow.hPow x n).log (HMul.hMul (↑n) x.log)
  -/
  rcases ENNReal.trichotomy x with (rfl | rfl | x_real)
    /-
      case inr.inl
      n : Nat
      n_pos : GT.gt n 0
      ⊢ Eq (HPow.hPow 0 n).log (HMul.hMul (↑n) (ENNReal.log 0))
    -/
  · rw [zero_pow n_pos.ne', log_zero, EReal.mul_bot_of_pos (Nat.cast_pos'.2 n_pos)]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inl
      n : Nat
      n_pos : GT.gt n 0
      ⊢ Eq (HPow.hPow Top.top n).log (HMul.hMul (↑n) Top.top.log)
    -/
  · rw [ENNReal.top_pow n_pos, log_top, EReal.mul_top_of_pos (Nat.cast_pos'.2 n_pos)]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr
      x : ENNReal
      n : Nat
      n_pos : GT.gt n 0
      x_real : LT.lt 0 x.toReal
      ⊢ Eq (HPow.hPow x n).log (HMul.hMul (↑n) x.log)
    -/
  · replace x_real := ENNReal.toReal_pos_iff.1 x_real
    simp only [log, pow_eq_zero_iff', x_real.1.ne', false_and, ↓reduceIte, pow_eq_top_iff,
      x_real.2.ne, toReal_pow, Real.log_pow, EReal.coe_mul, EReal.coe_coe_eq_natCast]


theorem log_rpow {x : ℝ≥0∞} {y : ℝ} : log (x ^ y) = y * log x := by
  /-
    x : ENNReal
    y : Real
    ⊢ Eq (HPow.hPow x y).log (HMul.hMul (↑y) x.log)
  -/
  rcases lt_trichotomy y 0 with (y_neg | rfl | y_pos)
    /-
      case inl
      x : ENNReal
      y : Real
      y_neg : LT.lt y 0
      ⊢ Eq (HPow.hPow x y).log (HMul.hMul (↑y) x.log)
    -/
  · rcases ENNReal.trichotomy x with (rfl | rfl | x_real)
    · simp only [ENNReal.zero_rpow_def y, not_lt_of_lt y_neg, y_neg.ne, if_false, log_top,
        log_zero, EReal.coe_mul_bot_of_neg y_neg]
      /-
        case inl.inr.inl
        y : Real
        y_neg : LT.lt y 0
        ⊢ Eq (HPow.hPow Top.top y).log (HMul.hMul (↑y) Top.top.log)
      -/
    · rw [ENNReal.top_rpow_of_neg y_neg, log_zero, log_top, EReal.coe_mul_top_of_neg y_neg]
      /-
        🎉 no goals
      -/
      /-
        case inl.inr.inr
        x : ENNReal
        y : Real
        y_neg : LT.lt y 0
        x_real : LT.lt 0 x.toReal
        ⊢ Eq (HPow.hPow x y).log (HMul.hMul (↑y) x.log)
      -/
    · have x_ne_zero := (ENNReal.toReal_pos_iff.1 x_real).1.ne'
      /-
        case inl.inr.inr
        x : ENNReal
        y : Real
        y_neg : LT.lt y 0
        x_real : LT.lt 0 x.toReal
        x_ne_zero : Ne x 0
        ⊢ Eq (HPow.hPow x y).log (HMul.hMul (↑y) x.log)
      -/
      have x_ne_top := (ENNReal.toReal_pos_iff.1 x_real).2.ne
      simp only [log, rpow_eq_zero_iff, x_ne_zero, false_and, x_ne_top, or_self, ↓reduceIte,
        rpow_eq_top_iff]
      /-
        case inl.inr.inr
        x : ENNReal
        y : Real
        y_neg : LT.lt y 0
        x_real : LT.lt 0 x.toReal
        x_ne_zero : Ne x 0
        x_ne_top : Ne x Top.top
        ⊢ Eq (↑(Real.log (HPow.hPow x y).toReal)) (HMul.hMul ↑y ↑(Real.log x.toReal))
      -/
      norm_cast
      /-
        case inl.inr.inr
        x : ENNReal
        y : Real
        y_neg : LT.lt y 0
        x_real : LT.lt 0 x.toReal
        x_ne_zero : Ne x 0
        x_ne_top : Ne x Top.top
        ⊢ Eq (Real.log (HPow.hPow x y).toReal) (HMul.hMul y (Real.log x.toReal))
      -/
      exact ENNReal.toReal_rpow x y ▸ Real.log_rpow x_real y
      /-
        🎉 no goals
      -/
    /-
      case inr.inl
      x : ENNReal
      ⊢ Eq (HPow.hPow x 0).log (HMul.hMul (↑0) x.log)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      x : ENNReal
      y : Real
      y_pos : LT.lt 0 y
      ⊢ Eq (HPow.hPow x y).log (HMul.hMul (↑y) x.log)
    -/
  · rcases ENNReal.trichotomy x with (rfl | rfl | x_real)
      /-
        case inr.inr.inl
        y : Real
        y_pos : LT.lt 0 y
        ⊢ Eq (HPow.hPow 0 y).log (HMul.hMul (↑y) (ENNReal.log 0))
      -/
    · rw [ENNReal.zero_rpow_of_pos y_pos, log_zero, EReal.mul_bot_of_pos]; norm_cast
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
      /-
        case inr.inr.inr.inl
        y : Real
        y_pos : LT.lt 0 y
        ⊢ Eq (HPow.hPow Top.top y).log (HMul.hMul (↑y) Top.top.log)
      -/
    · rw [ENNReal.top_rpow_of_pos y_pos, log_top, EReal.mul_top_of_pos]; norm_cast
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
      /-
        case inr.inr.inr.inr
        x : ENNReal
        y : Real
        y_pos : LT.lt 0 y
        x_real : LT.lt 0 x.toReal
        ⊢ Eq (HPow.hPow x y).log (HMul.hMul (↑y) x.log)
      -/
    · have x_ne_zero := (ENNReal.toReal_pos_iff.1 x_real).1.ne'
      /-
        case inr.inr.inr.inr
        x : ENNReal
        y : Real
        y_pos : LT.lt 0 y
        x_real : LT.lt 0 x.toReal
        x_ne_zero : Ne x 0
        ⊢ Eq (HPow.hPow x y).log (HMul.hMul (↑y) x.log)
      -/
      have x_ne_top := (ENNReal.toReal_pos_iff.1 x_real).2.ne
      simp only [log, rpow_eq_zero_iff, x_ne_zero, false_and, x_ne_top, or_self, ↓reduceIte,
        rpow_eq_top_iff]
      /-
        case inr.inr.inr.inr
        x : ENNReal
        y : Real
        y_pos : LT.lt 0 y
        x_real : LT.lt 0 x.toReal
        x_ne_zero : Ne x 0
        x_ne_top : Ne x Top.top
        ⊢ Eq (↑(Real.log (HPow.hPow x y).toReal)) (HMul.hMul ↑y ↑(Real.log x.toReal))
      -/
      norm_cast
      /-
        case inr.inr.inr.inr
        x : ENNReal
        y : Real
        y_pos : LT.lt 0 y
        x_real : LT.lt 0 x.toReal
        x_ne_zero : Ne x 0
        x_ne_top : Ne x Top.top
        ⊢ Eq (Real.log (HPow.hPow x y).toReal) (HMul.hMul y (Real.log x.toReal))
      -/
      exact ENNReal.toReal_rpow x y ▸ Real.log_rpow x_real y
      /-
        🎉 no goals
      -/


lemma log_inv {x : ℝ≥0∞} : log x⁻¹ = - log x := by
  /-
    x : ENNReal
    ⊢ Eq (Inv.inv x).log (Neg.neg x.log)
  -/
  simp [← rpow_neg_one, log_rpow]
  /-
    🎉 no goals
  -/


