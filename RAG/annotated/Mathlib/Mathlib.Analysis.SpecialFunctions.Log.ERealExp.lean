/-- Exponential as a function from `EReal` to `ℝ≥0∞`. -/
noncomputable
def exp : EReal → ℝ≥0∞
  | ⊥ => 0
  | ⊤ => ∞
  | (x : ℝ) => ENNReal.ofReal (Real.exp x)


@[simp] lemma exp_bot : exp ⊥ = 0 := rfl

                                         /-
                                           ⊢ Eq (EReal.exp 0) 1
                                         -/
@[simp] lemma exp_zero : exp 0 = 1 := by simp [exp]
                                         /-
                                           🎉 no goals
                                         -/

@[simp] lemma exp_top : exp ⊤ = ∞ := rfl

@[simp] lemma exp_coe (x : ℝ) : exp x = ENNReal.ofReal (Real.exp x) := rfl


@[simp] lemma exp_eq_zero_iff {x : EReal} : exp x = 0 ↔ x = ⊥ := by
  /-
    x : EReal
    ⊢ Iff (Eq x.exp 0) (Eq x Bot.bot)
  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> simp [Real.exp_pos]
                  /-
                    🎉 no goals
                  -/


@[simp] lemma exp_eq_top_iff {x : EReal} : exp x = ∞ ↔ x = ⊤ := by
  /-
    x : EReal
    ⊢ Iff (Eq x.exp Top.top) (Eq x Top.top)
  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> simp
                  /-
                    🎉 no goals
                  -/


lemma exp_strictMono : StrictMono exp := by
  /-
    ⊢ StrictMono EReal.exp
  -/
  intro x y h
  /-
    x y : EReal
    h : LT.lt x y
    ⊢ LT.lt x.exp y.exp
  -/
  induction x
    /-
      case h_bot
      y : EReal
      h : LT.lt Bot.bot y
      ⊢ LT.lt Bot.bot.exp y.exp
    -/
  · rw [exp_bot, pos_iff_ne_zero, ne_eq, exp_eq_zero_iff]
    /-
      case h_bot
      y : EReal
      h : LT.lt Bot.bot y
      ⊢ Not (Eq y Bot.bot)
    -/
    exact h.ne'
    /-
      🎉 no goals
    -/
    /-
      case h_real
      y : EReal
      a✝ : Real
      h : LT.lt (↑a✝) y
      ⊢ LT.lt (↑a✝).exp y.exp
    -/
  · induction y
      /-
        case h_real.h_bot
        a✝ : Real
        h : LT.lt (↑a✝) Bot.bot
        ⊢ LT.lt (↑a✝).exp Bot.bot.exp
      -/
    · simp at h
      /-
        🎉 no goals
      -/
      /-
        case h_real.h_real
        a✝¹ a✝ : Real
        h : LT.lt ↑a✝¹ ↑a✝
        ⊢ LT.lt (↑a✝¹).exp (↑a✝).exp
      -/
    · simp_rw [exp_coe]
      /-
        case h_real.h_real
        a✝¹ a✝ : Real
        h : LT.lt ↑a✝¹ ↑a✝
        ⊢ LT.lt (ENNReal.ofReal (Real.exp a✝¹)) (ENNReal.ofReal (Real.exp a✝))
      -/
      exact ENNReal.ofReal_lt_ofReal_iff'.mpr ⟨Real.exp_lt_exp_of_lt (mod_cast h), Real.exp_pos _⟩
      /-
        🎉 no goals
      -/
      /-
        case h_real.h_top
        a✝ : Real
        h : LT.lt (↑a✝) Top.top
        ⊢ LT.lt (↑a✝).exp Top.top.exp
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case h_top
      y : EReal
      h : LT.lt Top.top y
      ⊢ LT.lt Top.top.exp y.exp
    -/
  · exact (not_top_lt h).elim
    /-
      🎉 no goals
    -/


lemma exp_monotone : Monotone exp := exp_strictMono.monotone


@[simp] lemma exp_lt_exp_iff {a b : EReal} : exp a < exp b ↔ a < b := exp_strictMono.lt_iff_lt


@[simp] lemma zero_lt_exp_iff {a : EReal} : 0 < exp a ↔ ⊥ < a := exp_bot ▸ @exp_lt_exp_iff ⊥ a


@[simp] lemma exp_lt_top_iff {a : EReal} : exp a < ⊤ ↔ a < ⊤ := exp_top ▸ @exp_lt_exp_iff a ⊤


@[simp] lemma exp_lt_one_iff {a : EReal} : exp a < 1 ↔ a < 0 := exp_zero ▸ @exp_lt_exp_iff a 0


@[simp] lemma one_lt_exp_iff {a : EReal} : 1 < exp a ↔ 0 < a := exp_zero ▸ @exp_lt_exp_iff 0 a


@[simp] lemma exp_le_exp_iff {a b : EReal} : exp a ≤ exp b ↔ a ≤ b := exp_strictMono.le_iff_le


@[simp] lemma exp_le_one_iff {a : EReal} : exp a ≤ 1 ↔ a ≤ 0 := exp_zero ▸ @exp_le_exp_iff a 0


@[simp] lemma one_le_exp_iff {a : EReal} : 1 ≤ exp a ↔ 0 ≤ a := exp_zero ▸ @exp_le_exp_iff 0 a


lemma exp_neg (x : EReal) : exp (-x) = (exp x)⁻¹ := by
  /-
    x : EReal
    ⊢ Eq (Neg.neg x).exp (Inv.inv x.exp)
  -/
  induction x
    /-
      case h_bot
      ⊢ Eq (Neg.neg Bot.bot).exp (Inv.inv Bot.bot.exp)
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [exp_coe, ← EReal.coe_neg, exp_coe, ← ENNReal.ofReal_inv_of_pos (Real.exp_pos _),
      Real.exp_neg]
    /-
      case h_top
      ⊢ Eq (Neg.neg Top.top).exp (Inv.inv Top.top.exp)
    -/
  · simp
    /-
      🎉 no goals
    -/


lemma exp_add (x y : EReal) : exp (x + y) = exp x * exp y := by
  /-
    x y : EReal
    ⊢ Eq (HAdd.hAdd x y).exp (HMul.hMul x.exp y.exp)
  -/
  induction x
    /-
      case h_bot
      y : EReal
      ⊢ Eq (HAdd.hAdd Bot.bot y).exp (HMul.hMul Bot.bot.exp y.exp)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h_real
      y : EReal
      a✝ : Real
      ⊢ Eq (HAdd.hAdd (↑a✝) y).exp (HMul.hMul (↑a✝).exp y.exp)
    -/
  · induction y
      /-
        case h_real.h_bot
        a✝ : Real
        ⊢ Eq (HAdd.hAdd (↑a✝) Bot.bot).exp (HMul.hMul (↑a✝).exp Bot.bot.exp)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h_real.h_real
        a✝¹ a✝ : Real
        ⊢ Eq (HAdd.hAdd ↑a✝¹ ↑a✝).exp (HMul.hMul (↑a✝¹).exp (↑a✝).exp)
      -/
    · simp only [← EReal.coe_add, exp_coe]
      /-
        case h_real.h_real
        a✝¹ a✝ : Real
        ⊢ Eq (ENNReal.ofReal (Real.exp (HAdd.hAdd a✝¹ a✝))) (HMul.hMul (ENNReal.ofReal …
      -/
      rw [← ENNReal.ofReal_mul (Real.exp_nonneg _), Real.exp_add]
      /-
        🎉 no goals
      -/
      /-
        case h_real.h_top
        a✝ : Real
        ⊢ Eq (HAdd.hAdd (↑a✝) Top.top).exp (HMul.hMul (↑a✝).exp Top.top.exp)
      -/
    · simp only [EReal.coe_add_top, exp_top, exp_coe]
      /-
        case h_real.h_top
        a✝ : Real
        ⊢ Eq Top.top (HMul.hMul (ENNReal.ofReal (Real.exp a✝)) Top.top)
      -/
      rw [ENNReal.mul_top]
      /-
        case h_real.h_top
        a✝ : Real
        ⊢ Ne (ENNReal.ofReal (Real.exp a✝)) 0
      -/
      simp [Real.exp_pos]
      /-
        🎉 no goals
      -/
    /-
      case h_top
      y : EReal
      ⊢ Eq (HAdd.hAdd Top.top y).exp (HMul.hMul Top.top.exp y.exp)
    -/
  · induction y
      /-
        case h_top.h_bot
        ⊢ Eq (HAdd.hAdd Top.top Bot.bot).exp (HMul.hMul Top.top.exp Bot.bot.exp)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h_top.h_real
        a✝ : Real
        ⊢ Eq (HAdd.hAdd Top.top ↑a✝).exp (HMul.hMul Top.top.exp (↑a✝).exp)
      -/
    · simp only [EReal.top_add_coe, exp_top, exp_coe]
      /-
        case h_top.h_real
        a✝ : Real
        ⊢ Eq Top.top (HMul.hMul Top.top (ENNReal.ofReal (Real.exp a✝)))
      -/
      rw [ENNReal.top_mul]
      /-
        case h_top.h_real
        a✝ : Real
        ⊢ Ne (ENNReal.ofReal (Real.exp a✝)) 0
      -/
      simp [Real.exp_pos]
      /-
        🎉 no goals
      -/
      /-
        case h_top.h_top
        ⊢ Eq (HAdd.hAdd Top.top Top.top).exp (HMul.hMul Top.top.exp Top.top.exp)
      -/
    · simp
      /-
        🎉 no goals
      -/


