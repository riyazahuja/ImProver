/-- ereal : The type `[-∞, ∞]` -/
def EReal := WithBot (WithTop ℝ)
  deriving Bot, Zero, One, Nontrivial, AddMonoid, PartialOrder


instance : ZeroLEOneClass EReal := inferInstanceAs (ZeroLEOneClass (WithBot (WithTop ℝ)))

instance : SupSet EReal := inferInstanceAs (SupSet (WithBot (WithTop ℝ)))

instance : InfSet EReal := inferInstanceAs (InfSet (WithBot (WithTop ℝ)))


instance : CompleteLinearOrder EReal :=
  inferInstanceAs (CompleteLinearOrder (WithBot (WithTop ℝ)))


instance : LinearOrderedAddCommMonoid EReal :=
  inferInstanceAs (LinearOrderedAddCommMonoid (WithBot (WithTop ℝ)))


instance : AddCommMonoidWithOne EReal :=
  inferInstanceAs (AddCommMonoidWithOne (WithBot (WithTop ℝ)))


instance : DenselyOrdered EReal :=
  inferInstanceAs (DenselyOrdered (WithBot (WithTop ℝ)))


instance : CharZero EReal := inferInstanceAs (CharZero (WithBot (WithTop ℝ)))


/-- The canonical inclusion from reals to ereals. Registered as a coercion. -/
@[coe] def Real.toEReal : ℝ → EReal := some ∘ some


instance decidableLT : DecidableRel ((· < ·) : EReal → EReal → Prop) :=
  WithBot.decidableLT

-- TODO: Provide explicitly, otherwise it is inferred noncomputably from `CompleteLinearOrder`

instance : Top EReal := ⟨some ⊤⟩


instance : Coe ℝ EReal := ⟨Real.toEReal⟩


theorem coe_strictMono : StrictMono Real.toEReal :=
  WithBot.coe_strictMono.comp WithTop.coe_strictMono


theorem coe_injective : Injective Real.toEReal :=
  coe_strictMono.injective


@[simp, norm_cast]
protected theorem coe_le_coe_iff {x y : ℝ} : (x : EReal) ≤ (y : EReal) ↔ x ≤ y :=
  coe_strictMono.le_iff_le


@[simp, norm_cast]
protected theorem coe_lt_coe_iff {x y : ℝ} : (x : EReal) < (y : EReal) ↔ x < y :=
  coe_strictMono.lt_iff_lt


@[simp, norm_cast]
protected theorem coe_eq_coe_iff {x y : ℝ} : (x : EReal) = (y : EReal) ↔ x = y :=
  coe_injective.eq_iff


protected theorem coe_ne_coe_iff {x y : ℝ} : (x : EReal) ≠ (y : EReal) ↔ x ≠ y :=
  coe_injective.ne_iff


/-- The canonical map from nonnegative extended reals to extended reals. -/
@[coe] def _root_.ENNReal.toEReal : ℝ≥0∞ → EReal
  | ⊤ => ⊤
  | .some x => x.1


instance hasCoeENNReal : Coe ℝ≥0∞ EReal :=
  ⟨ENNReal.toEReal⟩


instance : Inhabited EReal := ⟨0⟩


@[simp, norm_cast]
theorem coe_zero : ((0 : ℝ) : EReal) = 0 := rfl


@[simp, norm_cast]
theorem coe_one : ((1 : ℝ) : EReal) = 1 := rfl


/-- A recursor for `EReal` in terms of the coercion.

When working in term mode, note that pattern matching can be used directly. -/
@[elab_as_elim, induction_eliminator, cases_eliminator]
protected def rec {C : EReal → Sort*} (h_bot : C ⊥) (h_real : ∀ a : ℝ, C a) (h_top : C ⊤) :
    ∀ a : EReal, C a
  | ⊥ => h_bot
  | (a : ℝ) => h_real a
  | ⊤ => h_top


protected lemma «forall» {p : EReal → Prop} : (∀ r, p r) ↔ p ⊥ ∧ p ⊤ ∧ ∀ r : ℝ, p r where
  mp h := ⟨h _, h _, fun _ ↦ h _⟩
  mpr h := EReal.rec h.1 h.2.2 h.2.1


protected lemma «exists» {p : EReal → Prop} : (∃ r, p r) ↔ p ⊥ ∨ p ⊤ ∨ ∃ r : ℝ, p r where
           /-
             p : EReal → Prop
             ⊢ (Exists fun r => p r) → Or (p Bot.bot) (Or (p Top.top) (Exists fun r => p ↑r))
           -/
                                       /-
                                         🎉 no goals
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
  mp := by rintro ⟨r, hr⟩; cases r <;> aesop
                                       /-
                                         🎉 no goals
                                       -/
            /-
              p : EReal → Prop
              ⊢ Or (p Bot.bot) (Or (p Top.top) (Exists fun r => p ↑r)) → Exists fun r => p r
            -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
  mpr := by rintro (h | h | ⟨r, hr⟩) <;> exact ⟨_, ‹_›⟩
                                         /-
                                           🎉 no goals
                                         -/


/-- The multiplication on `EReal`. Our definition satisfies `0 * x = x * 0 = 0` for any `x`, and
picks the only sensible value elsewhere. -/
protected def mul : EReal → EReal → EReal
  | ⊥, ⊥ => ⊤
  | ⊥, ⊤ => ⊥
  | ⊥, (y : ℝ) => if 0 < y then ⊥ else if y = 0 then 0 else ⊤
  | ⊤, ⊥ => ⊥
  | ⊤, ⊤ => ⊤
  | ⊤, (y : ℝ) => if 0 < y then ⊤ else if y = 0 then 0 else ⊥
  | (x : ℝ), ⊤ => if 0 < x then ⊤ else if x = 0 then 0 else ⊥
  | (x : ℝ), ⊥ => if 0 < x then ⊥ else if x = 0 then 0 else ⊤
  | (x : ℝ), (y : ℝ) => (x * y : ℝ)


instance : Mul EReal := ⟨EReal.mul⟩


@[simp, norm_cast]
theorem coe_mul (x y : ℝ) : (↑(x * y) : EReal) = x * y :=
  rfl


/-- Induct on two `EReal`s by performing case splits on the sign of one whenever the other is
infinite. -/
@[elab_as_elim]
theorem induction₂ {P : EReal → EReal → Prop} (top_top : P ⊤ ⊤) (top_pos : ∀ x : ℝ, 0 < x → P ⊤ x)
    (top_zero : P ⊤ 0) (top_neg : ∀ x : ℝ, x < 0 → P ⊤ x) (top_bot : P ⊤ ⊥)
    (pos_top : ∀ x : ℝ, 0 < x → P x ⊤) (pos_bot : ∀ x : ℝ, 0 < x → P x ⊥) (zero_top : P 0 ⊤)
    (coe_coe : ∀ x y : ℝ, P x y) (zero_bot : P 0 ⊥) (neg_top : ∀ x : ℝ, x < 0 → P x ⊤)
    (neg_bot : ∀ x : ℝ, x < 0 → P x ⊥) (bot_top : P ⊥ ⊤) (bot_pos : ∀ x : ℝ, 0 < x → P ⊥ x)
    (bot_zero : P ⊥ 0) (bot_neg : ∀ x : ℝ, x < 0 → P ⊥ x) (bot_bot : P ⊥ ⊥) : ∀ x y, P x y
  | ⊥, ⊥ => bot_bot
  | ⊥, (y : ℝ) => by
    /-
      P : EReal → EReal → Prop
      top_top : P Top.top Top.top
      top_pos : ∀ (x : Real), LT.lt 0 x → P Top.top ↑x
      top_zero : P Top.top 0
      top_neg : ∀ (x : Real), LT.lt x 0 → P Top.top ↑x
      top_bot : P Top.top Bot.bot
      pos_top : ∀ (x : Real), LT.lt 0 x → P (↑x) Top.top
      pos_bot : ∀ (x : Real), LT.lt 0 x → P (↑x) Bot.bot
      zero_top : P 0 Top.top
      coe_coe : ∀ (x y : Real), P ↑x ↑y
      zero_bot : P 0 Bot.bot
      neg_top : ∀ (x : Real), LT.lt x 0 → P (↑x) Top.top
      neg_bot : ∀ (x : Real), LT.lt x 0 → P (↑x) Bot.bot
      bot_top : P Bot.bot Top.top
      bot_pos : ∀ (x : Real), LT.lt 0 x → P Bot.bot ↑x
      bot_zero : P Bot.bot 0
      bot_neg : ∀ (x : Real), LT.lt x 0 → P Bot.bot ↑x
      bot_bot : P Bot.bot Bot.bot
      y : Real
      ⊢ P Bot.bot ↑y
    -/
    rcases lt_trichotomy y 0 with (hy | rfl | hy)
    /-
      case inl
      P : EReal → EReal → Prop
      top_top : P Top.top Top.top
      top_pos : ∀ (x : Real), LT.lt 0 x → P Top.top ↑x
      top_zero : P Top.top 0
      top_neg : ∀ (x : Real), LT.lt x 0 → P Top.top ↑x
      top_bot : P Top.top Bot.bot
      pos_top : ∀ (x : Real), LT.lt 0 x → P (↑x) Top.top
      pos_bot : ∀ (x : Real), LT.lt 0 x → P (↑x) Bot.bot
      zero_top : P 0 Top.top
      coe_coe : ∀ (x y : Real), P ↑x ↑y
      zero_bot : P 0 Bot.bot
      neg_top : ∀ (x : Real), LT.lt x 0 → P (↑x) Top.top
      neg_bot : ∀ (x : Real), LT.lt x 0 → P (↑x) Bot.bot
      bot_top : P Bot.bot Top.top
      bot_pos : ∀ (x : Real), LT.lt 0 x → P Bot.bot ↑x
      bot_zero : P Bot.bot 0
      bot_neg : ∀ (x : Real), LT.lt x 0 → P Bot.bot ↑x
      bot_bot : P Bot.bot Bot.bot
      y : Real
      hy : LT.lt y 0
      ⊢ P Bot.bot ↑y
    -/
    exacts [bot_neg y hy, bot_zero, bot_pos y hy]
    /-
      🎉 no goals
    -/
  | ⊥, ⊤ => bot_top
  | (x : ℝ), ⊥ => by
    /-
      P : EReal → EReal → Prop
      top_top : P Top.top Top.top
      top_pos : ∀ (x : Real), LT.lt 0 x → P Top.top ↑x
      top_zero : P Top.top 0
      top_neg : ∀ (x : Real), LT.lt x 0 → P Top.top ↑x
      top_bot : P Top.top Bot.bot
      pos_top : ∀ (x : Real), LT.lt 0 x → P (↑x) Top.top
      pos_bot : ∀ (x : Real), LT.lt 0 x → P (↑x) Bot.bot
      zero_top : P 0 Top.top
      coe_coe : ∀ (x y : Real), P ↑x ↑y
      zero_bot : P 0 Bot.bot
      neg_top : ∀ (x : Real), LT.lt x 0 → P (↑x) Top.top
      neg_bot : ∀ (x : Real), LT.lt x 0 → P (↑x) Bot.bot
      bot_top : P Bot.bot Top.top
      bot_pos : ∀ (x : Real), LT.lt 0 x → P Bot.bot ↑x
      bot_zero : P Bot.bot 0
      bot_neg : ∀ (x : Real), LT.lt x 0 → P Bot.bot ↑x
      bot_bot : P Bot.bot Bot.bot
      x : Real
      ⊢ P (↑x) Bot.bot
    -/
    rcases lt_trichotomy x 0 with (hx | rfl | hx)
    /-
      case inl
      P : EReal → EReal → Prop
      top_top : P Top.top Top.top
      top_pos : ∀ (x : Real), LT.lt 0 x → P Top.top ↑x
      top_zero : P Top.top 0
      top_neg : ∀ (x : Real), LT.lt x 0 → P Top.top ↑x
      top_bot : P Top.top Bot.bot
      pos_top : ∀ (x : Real), LT.lt 0 x → P (↑x) Top.top
      pos_bot : ∀ (x : Real), LT.lt 0 x → P (↑x) Bot.bot
      zero_top : P 0 Top.top
      coe_coe : ∀ (x y : Real), P ↑x ↑y
      zero_bot : P 0 Bot.bot
      neg_top : ∀ (x : Real), LT.lt x 0 → P (↑x) Top.top
      neg_bot : ∀ (x : Real), LT.lt x 0 → P (↑x) Bot.bot
      bot_top : P Bot.bot Top.top
      bot_pos : ∀ (x : Real), LT.lt 0 x → P Bot.bot ↑x
      bot_zero : P Bot.bot 0
      bot_neg : ∀ (x : Real), LT.lt x 0 → P Bot.bot ↑x
      bot_bot : P Bot.bot Bot.bot
      x : Real
      hx : LT.lt x 0
      ⊢ P (↑x) Bot.bot
    -/
    exacts [neg_bot x hx, zero_bot, pos_bot x hx]
    /-
      🎉 no goals
    -/
  | (x : ℝ), (y : ℝ) => coe_coe _ _
  | (x : ℝ), ⊤ => by
    /-
      P : EReal → EReal → Prop
      top_top : P Top.top Top.top
      top_pos : ∀ (x : Real), LT.lt 0 x → P Top.top ↑x
      top_zero : P Top.top 0
      top_neg : ∀ (x : Real), LT.lt x 0 → P Top.top ↑x
      top_bot : P Top.top Bot.bot
      pos_top : ∀ (x : Real), LT.lt 0 x → P (↑x) Top.top
      pos_bot : ∀ (x : Real), LT.lt 0 x → P (↑x) Bot.bot
      zero_top : P 0 Top.top
      coe_coe : ∀ (x y : Real), P ↑x ↑y
      zero_bot : P 0 Bot.bot
      neg_top : ∀ (x : Real), LT.lt x 0 → P (↑x) Top.top
      neg_bot : ∀ (x : Real), LT.lt x 0 → P (↑x) Bot.bot
      bot_top : P Bot.bot Top.top
      bot_pos : ∀ (x : Real), LT.lt 0 x → P Bot.bot ↑x
      bot_zero : P Bot.bot 0
      bot_neg : ∀ (x : Real), LT.lt x 0 → P Bot.bot ↑x
      bot_bot : P Bot.bot Bot.bot
      x : Real
      ⊢ P (↑x) Top.top
    -/
    rcases lt_trichotomy x 0 with (hx | rfl | hx)
    /-
      case inl
      P : EReal → EReal → Prop
      top_top : P Top.top Top.top
      top_pos : ∀ (x : Real), LT.lt 0 x → P Top.top ↑x
      top_zero : P Top.top 0
      top_neg : ∀ (x : Real), LT.lt x 0 → P Top.top ↑x
      top_bot : P Top.top Bot.bot
      pos_top : ∀ (x : Real), LT.lt 0 x → P (↑x) Top.top
      pos_bot : ∀ (x : Real), LT.lt 0 x → P (↑x) Bot.bot
      zero_top : P 0 Top.top
      coe_coe : ∀ (x y : Real), P ↑x ↑y
      zero_bot : P 0 Bot.bot
      neg_top : ∀ (x : Real), LT.lt x 0 → P (↑x) Top.top
      neg_bot : ∀ (x : Real), LT.lt x 0 → P (↑x) Bot.bot
      bot_top : P Bot.bot Top.top
      bot_pos : ∀ (x : Real), LT.lt 0 x → P Bot.bot ↑x
      bot_zero : P Bot.bot 0
      bot_neg : ∀ (x : Real), LT.lt x 0 → P Bot.bot ↑x
      bot_bot : P Bot.bot Bot.bot
      x : Real
      hx : LT.lt x 0
      ⊢ P (↑x) Top.top
    -/
    exacts [neg_top x hx, zero_top, pos_top x hx]
    /-
      🎉 no goals
    -/
  | ⊤, ⊥ => top_bot
  | ⊤, (y : ℝ) => by
    /-
      P : EReal → EReal → Prop
      top_top : P Top.top Top.top
      top_pos : ∀ (x : Real), LT.lt 0 x → P Top.top ↑x
      top_zero : P Top.top 0
      top_neg : ∀ (x : Real), LT.lt x 0 → P Top.top ↑x
      top_bot : P Top.top Bot.bot
      pos_top : ∀ (x : Real), LT.lt 0 x → P (↑x) Top.top
      pos_bot : ∀ (x : Real), LT.lt 0 x → P (↑x) Bot.bot
      zero_top : P 0 Top.top
      coe_coe : ∀ (x y : Real), P ↑x ↑y
      zero_bot : P 0 Bot.bot
      neg_top : ∀ (x : Real), LT.lt x 0 → P (↑x) Top.top
      neg_bot : ∀ (x : Real), LT.lt x 0 → P (↑x) Bot.bot
      bot_top : P Bot.bot Top.top
      bot_pos : ∀ (x : Real), LT.lt 0 x → P Bot.bot ↑x
      bot_zero : P Bot.bot 0
      bot_neg : ∀ (x : Real), LT.lt x 0 → P Bot.bot ↑x
      bot_bot : P Bot.bot Bot.bot
      y : Real
      ⊢ P Top.top ↑y
    -/
    rcases lt_trichotomy y 0 with (hy | rfl | hy)
    /-
      case inl
      P : EReal → EReal → Prop
      top_top : P Top.top Top.top
      top_pos : ∀ (x : Real), LT.lt 0 x → P Top.top ↑x
      top_zero : P Top.top 0
      top_neg : ∀ (x : Real), LT.lt x 0 → P Top.top ↑x
      top_bot : P Top.top Bot.bot
      pos_top : ∀ (x : Real), LT.lt 0 x → P (↑x) Top.top
      pos_bot : ∀ (x : Real), LT.lt 0 x → P (↑x) Bot.bot
      zero_top : P 0 Top.top
      coe_coe : ∀ (x y : Real), P ↑x ↑y
      zero_bot : P 0 Bot.bot
      neg_top : ∀ (x : Real), LT.lt x 0 → P (↑x) Top.top
      neg_bot : ∀ (x : Real), LT.lt x 0 → P (↑x) Bot.bot
      bot_top : P Bot.bot Top.top
      bot_pos : ∀ (x : Real), LT.lt 0 x → P Bot.bot ↑x
      bot_zero : P Bot.bot 0
      bot_neg : ∀ (x : Real), LT.lt x 0 → P Bot.bot ↑x
      bot_bot : P Bot.bot Bot.bot
      y : Real
      hy : LT.lt y 0
      ⊢ P Top.top ↑y
    -/
    exacts [top_neg y hy, top_zero, top_pos y hy]
    /-
      🎉 no goals
    -/
  | ⊤, ⊤ => top_top


/-- Induct on two `EReal`s by performing case splits on the sign of one whenever the other is
infinite. This version eliminates some cases by assuming that the relation is symmetric. -/
@[elab_as_elim]
theorem induction₂_symm {P : EReal → EReal → Prop} (symm : ∀ {x y}, P x y → P y x)
    (top_top : P ⊤ ⊤) (top_pos : ∀ x : ℝ, 0 < x → P ⊤ x) (top_zero : P ⊤ 0)
    (top_neg : ∀ x : ℝ, x < 0 → P ⊤ x) (top_bot : P ⊤ ⊥) (pos_bot : ∀ x : ℝ, 0 < x → P x ⊥)
    (coe_coe : ∀ x y : ℝ, P x y) (zero_bot : P 0 ⊥) (neg_bot : ∀ x : ℝ, x < 0 → P x ⊥)
    (bot_bot : P ⊥ ⊥) : ∀ x y, P x y :=
  @induction₂ P top_top top_pos top_zero top_neg top_bot (fun _ h => symm <| top_pos _ h)
    pos_bot (symm top_zero) coe_coe zero_bot (fun _ h => symm <| top_neg _ h) neg_bot (symm top_bot)
    (fun _ h => symm <| pos_bot _ h) (symm zero_bot) (fun _ h => symm <| neg_bot _ h) bot_bot


protected theorem mul_comm (x y : EReal) : x * y = y * x := by
  /-
    x y : EReal
    ⊢ Eq (HMul.hMul x y) (HMul.hMul y x)
  -/
  induction x <;> induction y  <;>
    /-
      case h_bot.h_bot
      ⊢ Eq (HMul.hMul Bot.bot Bot.bot) (HMul.hMul Bot.bot Bot.bot)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    try { rfl }
    /-
      🎉 no goals
    -/
  /-
    case h_real.h_real
    a✝¹ a✝ : Real
    ⊢ Eq (HMul.hMul ↑a✝¹ ↑a✝) (HMul.hMul ↑a✝ ↑a✝¹)
  -/
  rw [← coe_mul, ← coe_mul, mul_comm]
  /-
    🎉 no goals
  -/


protected theorem one_mul : ∀ x : EReal, 1 * x = x
  | ⊤ => if_pos one_pos
  | ⊥ => if_pos one_pos
  | (x : ℝ) => congr_arg Real.toEReal (one_mul x)


protected theorem zero_mul : ∀ x : EReal, 0 * x = 0
  | ⊤ => (if_neg (lt_irrefl _)).trans (if_pos rfl)
  | ⊥ => (if_neg (lt_irrefl _)).trans (if_pos rfl)
  | (x : ℝ) => congr_arg Real.toEReal (zero_mul x)


instance : MulZeroOneClass EReal where
  one_mul := EReal.one_mul
                         /-
                           x : EReal
                           ⊢ Eq (HMul.hMul x 1) x
                         -/
  mul_one := fun x => by rw [EReal.mul_comm, EReal.one_mul]
                         /-
                           🎉 no goals
                         -/
  zero_mul := EReal.zero_mul
                          /-
                            x : EReal
                            ⊢ Eq (HMul.hMul x 0) 0
                          -/
  mul_zero := fun x => by rw [EReal.mul_comm, EReal.zero_mul]
                          /-
                            🎉 no goals
                          -/


instance canLift : CanLift EReal ℝ (↑) fun r => r ≠ ⊤ ∧ r ≠ ⊥ where
  prf x hx := by
    /-
      x : EReal
      hx : And (Ne x Top.top) (Ne x Bot.bot)
      ⊢ Exists fun y => Eq (↑y) x
    -/
    induction x
      /-
        case h_bot
        hx : And (Ne Bot.bot Top.top) (Ne Bot.bot Bot.bot)
        ⊢ Exists fun y => Eq (↑y) Bot.bot
      -/
    · simp at hx
      /-
        🎉 no goals
      -/
      /-
        case h_real
        a✝ : Real
        hx : And (Ne (↑a✝) Top.top) (Ne (↑a✝) Bot.bot)
        ⊢ Exists fun y => Eq ↑y ↑a✝
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h_top
        hx : And (Ne Top.top Top.top) (Ne Top.top Bot.bot)
        ⊢ Exists fun y => Eq (↑y) Top.top
      -/
    · simp at hx
      /-
        🎉 no goals
      -/


/-- The map from extended reals to reals sending infinities to zero. -/
def toReal : EReal → ℝ
  | ⊥ => 0
  | ⊤ => 0
  | (x : ℝ) => x


@[simp]
theorem toReal_top : toReal ⊤ = 0 :=
  rfl


@[simp]
theorem toReal_bot : toReal ⊥ = 0 :=
  rfl


@[simp]
theorem toReal_zero : toReal 0 = 0 :=
  rfl


@[simp]
theorem toReal_one : toReal 1 = 1 :=
  rfl


@[simp]
theorem toReal_coe (x : ℝ) : toReal (x : EReal) = x :=
  rfl


@[simp]
theorem bot_lt_coe (x : ℝ) : (⊥ : EReal) < x :=
  WithBot.bot_lt_coe _


@[simp]
theorem coe_ne_bot (x : ℝ) : (x : EReal) ≠ ⊥ :=
  (bot_lt_coe x).ne'


@[simp]
theorem bot_ne_coe (x : ℝ) : (⊥ : EReal) ≠ x :=
  (bot_lt_coe x).ne


@[simp]
theorem coe_lt_top (x : ℝ) : (x : EReal) < ⊤ :=
  WithBot.coe_lt_coe.2 <| WithTop.coe_lt_top _


@[simp]
theorem coe_ne_top (x : ℝ) : (x : EReal) ≠ ⊤ :=
  (coe_lt_top x).ne


@[simp]
theorem top_ne_coe (x : ℝ) : (⊤ : EReal) ≠ x :=
  (coe_lt_top x).ne'


@[simp]
theorem bot_lt_zero : (⊥ : EReal) < 0 :=
  bot_lt_coe 0


@[simp]
theorem bot_ne_zero : (⊥ : EReal) ≠ 0 :=
  (coe_ne_bot 0).symm


@[simp]
theorem zero_ne_bot : (0 : EReal) ≠ ⊥ :=
  coe_ne_bot 0


@[simp]
theorem zero_lt_top : (0 : EReal) < ⊤ :=
  coe_lt_top 0


@[simp]
theorem zero_ne_top : (0 : EReal) ≠ ⊤ :=
  coe_ne_top 0


@[simp]
theorem top_ne_zero : (⊤ : EReal) ≠ 0 :=
  (coe_ne_top 0).symm


theorem range_coe : range Real.toEReal = {⊥, ⊤}ᶜ := by
  /-
    ⊢ Eq (Set.range Real.toEReal) (HasCompl.compl (Insert.insert Bot.bot (Singleto …
  -/
  ext x
  /-
    case h
    x : EReal
    ⊢ Iff (Membership.mem (Set.range Real.toEReal) x) (Membership.mem (HasCompl.co …
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


theorem range_coe_eq_Ioo : range Real.toEReal = Ioo ⊥ ⊤ := by
  /-
    ⊢ Eq (Set.range Real.toEReal) (Set.Ioo Bot.bot Top.top)
  -/
  ext x
  /-
    case h
    x : EReal
    ⊢ Iff (Membership.mem (Set.range Real.toEReal) x) (Membership.mem (Set.Ioo Bot …
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


@[simp, norm_cast]
theorem coe_add (x y : ℝ) : (↑(x + y) : EReal) = x + y :=
  rfl

-- `coe_mul` moved up


@[norm_cast]
theorem coe_nsmul (n : ℕ) (x : ℝ) : (↑(n • x) : EReal) = n • (x : EReal) :=
  map_nsmul (⟨⟨Real.toEReal, coe_zero⟩, coe_add⟩ : ℝ →+ EReal) _ _


@[simp, norm_cast]
theorem coe_eq_zero {x : ℝ} : (x : EReal) = 0 ↔ x = 0 :=
  EReal.coe_eq_coe_iff


@[simp, norm_cast]
theorem coe_eq_one {x : ℝ} : (x : EReal) = 1 ↔ x = 1 :=
  EReal.coe_eq_coe_iff


theorem coe_ne_zero {x : ℝ} : (x : EReal) ≠ 0 ↔ x ≠ 0 :=
  EReal.coe_ne_coe_iff


theorem coe_ne_one {x : ℝ} : (x : EReal) ≠ 1 ↔ x ≠ 1 :=
  EReal.coe_ne_coe_iff


@[simp, norm_cast]
protected theorem coe_nonneg {x : ℝ} : (0 : EReal) ≤ x ↔ 0 ≤ x :=
  EReal.coe_le_coe_iff


@[simp, norm_cast]
protected theorem coe_nonpos {x : ℝ} : (x : EReal) ≤ 0 ↔ x ≤ 0 :=
  EReal.coe_le_coe_iff


@[simp, norm_cast]
protected theorem coe_pos {x : ℝ} : (0 : EReal) < x ↔ 0 < x :=
  EReal.coe_lt_coe_iff


@[simp, norm_cast]
protected theorem coe_neg' {x : ℝ} : (x : EReal) < 0 ↔ x < 0 :=
  EReal.coe_lt_coe_iff


lemma toReal_eq_zero_iff {x : EReal} : x.toReal = 0 ↔ x = 0 ∨ x = ⊤ ∨ x = ⊥ := by
  /-
    x : EReal
    ⊢ Iff (Eq x.toReal 0) (Or (Eq x 0) (Or (Eq x Top.top) (Eq x Bot.bot)))
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases x <;> norm_num
              /-
                🎉 no goals
              -/


lemma toReal_ne_zero_iff {x : EReal} : x.toReal ≠ 0 ↔ x ≠ 0 ∧ x ≠ ⊤ ∧ x ≠ ⊥ := by
  /-
    x : EReal
    ⊢ Iff (Ne x.toReal 0) (And (Ne x 0) (And (Ne x Top.top) (Ne x Bot.bot)))
  -/
  simp only [ne_eq, toReal_eq_zero_iff, not_or]
  /-
    🎉 no goals
  -/


lemma toReal_eq_toReal {x y : EReal} (hx_top : x ≠ ⊤) (hx_bot : x ≠ ⊥)
    (hy_top : y ≠ ⊤) (hy_bot : y ≠ ⊥) :
    x.toReal = y.toReal ↔ x = y := by
  /-
    x y : EReal
    hx_top : Ne x Top.top
    hx_bot : Ne x Bot.bot
    hy_top : Ne y Top.top
    hy_bot : Ne y Bot.bot
    ⊢ Iff (Eq x.toReal y.toReal) (Eq x y)
  -/
  lift x to ℝ using ⟨hx_top, hx_bot⟩
  /-
    case intro
    y : EReal
    hy_top : Ne y Top.top
    hy_bot : Ne y Bot.bot
    x : Real
    hx_top : Ne (↑x) Top.top
    hx_bot : Ne (↑x) Bot.bot
    ⊢ Iff (Eq (↑x).toReal y.toReal) (Eq (↑x) y)
  -/
  lift y to ℝ using ⟨hy_top, hy_bot⟩
  /-
    case intro.intro
    x : Real
    hx_top : Ne (↑x) Top.top
    hx_bot : Ne (↑x) Bot.bot
    y : Real
    hy_top : Ne (↑y) Top.top
    hy_bot : Ne (↑y) Bot.bot
    ⊢ Iff (Eq (↑x).toReal (↑y).toReal) (Eq ↑x ↑y)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma toReal_nonneg {x : EReal} (hx : 0 ≤ x) : 0 ≤ x.toReal := by
  /-
    x : EReal
    hx : LE.le 0 x
    ⊢ LE.le 0 x.toReal
  -/
  cases x
    /-
      case h_bot
      hx : LE.le 0 Bot.bot
      ⊢ LE.le 0 Bot.bot.toReal
    -/
  · norm_num
    /-
      🎉 no goals
    -/
    /-
      case h_real
      a✝ : Real
      hx : LE.le 0 ↑a✝
      ⊢ LE.le 0 (↑a✝).toReal
    -/
  · exact toReal_coe _ ▸ EReal.coe_nonneg.mp hx
    /-
      🎉 no goals
    -/
    /-
      case h_top
      hx : LE.le 0 Top.top
      ⊢ LE.le 0 Top.top.toReal
    -/
  · norm_num
    /-
      🎉 no goals
    -/


lemma toReal_nonpos {x : EReal} (hx : x ≤ 0) : x.toReal ≤ 0 := by
  /-
    x : EReal
    hx : LE.le x 0
    ⊢ LE.le x.toReal 0
  -/
  cases x
    /-
      case h_bot
      hx : LE.le Bot.bot 0
      ⊢ LE.le Bot.bot.toReal 0
    -/
  · norm_num
    /-
      🎉 no goals
    -/
    /-
      case h_real
      a✝ : Real
      hx : LE.le (↑a✝) 0
      ⊢ LE.le (↑a✝).toReal 0
    -/
  · exact toReal_coe _ ▸ EReal.coe_nonpos.mp hx
    /-
      🎉 no goals
    -/
    /-
      case h_top
      hx : LE.le Top.top 0
      ⊢ LE.le Top.top.toReal 0
    -/
  · norm_num
    /-
      🎉 no goals
    -/


theorem toReal_le_toReal {x y : EReal} (h : x ≤ y) (hx : x ≠ ⊥) (hy : y ≠ ⊤) :
    x.toReal ≤ y.toReal := by
  /-
    x y : EReal
    h : LE.le x y
    hx : Ne x Bot.bot
    hy : Ne y Top.top
    ⊢ LE.le x.toReal y.toReal
  -/
  lift x to ℝ using ⟨ne_top_of_le_ne_top hy h, hx⟩
  /-
    case intro
    y : EReal
    hy : Ne y Top.top
    x : Real
    h : LE.le (↑x) y
    hx : Ne (↑x) Bot.bot
    ⊢ LE.le (↑x).toReal y.toReal
  -/
  lift y to ℝ using ⟨hy, ne_bot_of_le_ne_bot hx h⟩
  /-
    case intro.intro
    x : Real
    hx : Ne (↑x) Bot.bot
    y : Real
    hy : Ne (↑y) Top.top
    h : LE.le ↑x ↑y
    ⊢ LE.le (↑x).toReal (↑y).toReal
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem coe_toReal {x : EReal} (hx : x ≠ ⊤) (h'x : x ≠ ⊥) : (x.toReal : EReal) = x := by
  /-
    x : EReal
    hx : Ne x Top.top
    h'x : Ne x Bot.bot
    ⊢ Eq (↑x.toReal) x
  -/
  lift x to ℝ using ⟨hx, h'x⟩
  /-
    case intro
    x : Real
    hx : Ne (↑x) Top.top
    h'x : Ne (↑x) Bot.bot
    ⊢ Eq ↑(↑x).toReal ↑x
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem le_coe_toReal {x : EReal} (h : x ≠ ⊤) : x ≤ x.toReal := by
  /-
    x : EReal
    h : Ne x Top.top
    ⊢ LE.le x ↑x.toReal
  -/
  by_cases h' : x = ⊥
    /-
      case pos
      x : EReal
      h : Ne x Top.top
      h' : Eq x Bot.bot
      ⊢ LE.le x ↑x.toReal
    -/
  · simp only [h', bot_le]
    /-
      🎉 no goals
    -/
    /-
      case neg
      x : EReal
      h : Ne x Top.top
      h' : Not (Eq x Bot.bot)
      ⊢ LE.le x ↑x.toReal
    -/
  · simp only [le_refl, coe_toReal h h']
    /-
      🎉 no goals
    -/


theorem coe_toReal_le {x : EReal} (h : x ≠ ⊥) : ↑x.toReal ≤ x := by
  /-
    x : EReal
    h : Ne x Bot.bot
    ⊢ LE.le (↑x.toReal) x
  -/
  by_cases h' : x = ⊤
    /-
      case pos
      x : EReal
      h : Ne x Bot.bot
      h' : Eq x Top.top
      ⊢ LE.le (↑x.toReal) x
    -/
  · simp only [h', le_top]
    /-
      🎉 no goals
    -/
    /-
      case neg
      x : EReal
      h : Ne x Bot.bot
      h' : Not (Eq x Top.top)
      ⊢ LE.le (↑x.toReal) x
    -/
  · simp only [le_refl, coe_toReal h' h]
    /-
      🎉 no goals
    -/


theorem eq_top_iff_forall_lt (x : EReal) : x = ⊤ ↔ ∀ y : ℝ, (y : EReal) < x := by
  /-
    x : EReal
    ⊢ Iff (Eq x Top.top) (∀ (y : Real), LT.lt (↑y) x)
  -/
  constructor
    /-
      case mp
      x : EReal
      ⊢ Eq x Top.top → ∀ (y : Real), LT.lt (↑y) x
    -/
  · rintro rfl
    /-
      case mp
      ⊢ ∀ (y : Real), LT.lt (↑y) Top.top
    -/
    exact EReal.coe_lt_top
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x : EReal
      ⊢ (∀ (y : Real), LT.lt (↑y) x) → Eq x Top.top
    -/
  · contrapose!
    /-
      case mpr
      x : EReal
      ⊢ Ne x Top.top → Exists fun y => LE.le x ↑y
    -/
    intro h
    /-
      case mpr
      x : EReal
      h : Ne x Top.top
      ⊢ Exists fun y => LE.le x ↑y
    -/
    exact ⟨x.toReal, le_coe_toReal h⟩
    /-
      🎉 no goals
    -/


theorem eq_bot_iff_forall_lt (x : EReal) : x = ⊥ ↔ ∀ y : ℝ, x < (y : EReal) := by
  /-
    x : EReal
    ⊢ Iff (Eq x Bot.bot) (∀ (y : Real), LT.lt x ↑y)
  -/
  constructor
    /-
      case mp
      x : EReal
      ⊢ Eq x Bot.bot → ∀ (y : Real), LT.lt x ↑y
    -/
  · rintro rfl
    /-
      case mp
      ⊢ ∀ (y : Real), LT.lt Bot.bot ↑y
    -/
    exact bot_lt_coe
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x : EReal
      ⊢ (∀ (y : Real), LT.lt x ↑y) → Eq x Bot.bot
    -/
  · contrapose!
    /-
      case mpr
      x : EReal
      ⊢ Ne x Bot.bot → Exists fun y => LE.le (↑y) x
    -/
    intro h
    /-
      case mpr
      x : EReal
      h : Ne x Bot.bot
      ⊢ Exists fun y => LE.le (↑y) x
    -/
    exact ⟨x.toReal, coe_toReal_le h⟩
    /-
      🎉 no goals
    -/


lemma exists_between_coe_real {x z : EReal} (h : x < z) : ∃ y : ℝ, x < y ∧ y < z := by
  /-
    x z : EReal
    h : LT.lt x z
    ⊢ Exists fun y => And (LT.lt x ↑y) (LT.lt (↑y) z)
  -/
  obtain ⟨a, ha₁, ha₂⟩ := exists_between h
  induction a with
  | h_bot => exact (not_lt_bot ha₁).elim
  | h_real a₀ => exact ⟨a₀, ha₁, ha₂⟩
  | h_top => exact (not_top_lt ha₂).elim


@[simp]
lemma image_coe_Icc (x y : ℝ) : Real.toEReal '' Icc x y = Icc ↑x ↑y := by
  /-
    x y : Real
    ⊢ Eq (Set.image Real.toEReal (Set.Icc x y)) (Set.Icc ↑x ↑y)
  -/
  refine (image_comp WithBot.some WithTop.some _).trans ?_
  /-
    x y : Real
    ⊢ Eq (Set.image WithBot.some (Set.image WithTop.some (Set.Icc x y))) (Set.Icc  …
  -/
  rw [WithTop.image_coe_Icc, WithBot.image_coe_Icc]
  /-
    x y : Real
    ⊢ Eq (Set.Icc ↑↑x ↑↑y) (Set.Icc ↑x ↑y)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma image_coe_Ico (x y : ℝ) : Real.toEReal '' Ico x y = Ico ↑x ↑y := by
  /-
    x y : Real
    ⊢ Eq (Set.image Real.toEReal (Set.Ico x y)) (Set.Ico ↑x ↑y)
  -/
  refine (image_comp WithBot.some WithTop.some _).trans ?_
  /-
    x y : Real
    ⊢ Eq (Set.image WithBot.some (Set.image WithTop.some (Set.Ico x y))) (Set.Ico  …
  -/
  rw [WithTop.image_coe_Ico, WithBot.image_coe_Ico]
  /-
    x y : Real
    ⊢ Eq (Set.Ico ↑↑x ↑↑y) (Set.Ico ↑x ↑y)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma image_coe_Ici (x : ℝ) : Real.toEReal '' Ici x = Ico ↑x ⊤ := by
  /-
    x : Real
    ⊢ Eq (Set.image Real.toEReal (Set.Ici x)) (Set.Ico (↑x) Top.top)
  -/
  refine (image_comp WithBot.some WithTop.some _).trans ?_
  /-
    x : Real
    ⊢ Eq (Set.image WithBot.some (Set.image WithTop.some (Set.Ici x))) (Set.Ico (↑ …
  -/
  rw [WithTop.image_coe_Ici, WithBot.image_coe_Ico]
  /-
    x : Real
    ⊢ Eq (Set.Ico ↑↑x ↑Top.top) (Set.Ico (↑x) Top.top)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma image_coe_Ioc (x y : ℝ) : Real.toEReal '' Ioc x y = Ioc ↑x ↑y := by
  /-
    x y : Real
    ⊢ Eq (Set.image Real.toEReal (Set.Ioc x y)) (Set.Ioc ↑x ↑y)
  -/
  refine (image_comp WithBot.some WithTop.some _).trans ?_
  /-
    x y : Real
    ⊢ Eq (Set.image WithBot.some (Set.image WithTop.some (Set.Ioc x y))) (Set.Ioc  …
  -/
  rw [WithTop.image_coe_Ioc, WithBot.image_coe_Ioc]
  /-
    x y : Real
    ⊢ Eq (Set.Ioc ↑↑x ↑↑y) (Set.Ioc ↑x ↑y)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma image_coe_Ioo (x y : ℝ) : Real.toEReal '' Ioo x y = Ioo ↑x ↑y := by
  /-
    x y : Real
    ⊢ Eq (Set.image Real.toEReal (Set.Ioo x y)) (Set.Ioo ↑x ↑y)
  -/
  refine (image_comp WithBot.some WithTop.some _).trans ?_
  /-
    x y : Real
    ⊢ Eq (Set.image WithBot.some (Set.image WithTop.some (Set.Ioo x y))) (Set.Ioo  …
  -/
  rw [WithTop.image_coe_Ioo, WithBot.image_coe_Ioo]
  /-
    x y : Real
    ⊢ Eq (Set.Ioo ↑↑x ↑↑y) (Set.Ioo ↑x ↑y)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma image_coe_Ioi (x : ℝ) : Real.toEReal '' Ioi x = Ioo ↑x ⊤ := by
  /-
    x : Real
    ⊢ Eq (Set.image Real.toEReal (Set.Ioi x)) (Set.Ioo (↑x) Top.top)
  -/
  refine (image_comp WithBot.some WithTop.some _).trans ?_
  /-
    x : Real
    ⊢ Eq (Set.image WithBot.some (Set.image WithTop.some (Set.Ioi x))) (Set.Ioo (↑ …
  -/
  rw [WithTop.image_coe_Ioi, WithBot.image_coe_Ioo]
  /-
    x : Real
    ⊢ Eq (Set.Ioo ↑↑x ↑Top.top) (Set.Ioo (↑x) Top.top)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma image_coe_Iic (x : ℝ) : Real.toEReal '' Iic x = Ioc ⊥ ↑x := by
  /-
    x : Real
    ⊢ Eq (Set.image Real.toEReal (Set.Iic x)) (Set.Ioc Bot.bot ↑x)
  -/
  refine (image_comp WithBot.some WithTop.some _).trans ?_
  /-
    x : Real
    ⊢ Eq (Set.image WithBot.some (Set.image WithTop.some (Set.Iic x))) (Set.Ioc Bo …
  -/
  rw [WithTop.image_coe_Iic, WithBot.image_coe_Iic]
  /-
    x : Real
    ⊢ Eq (Set.Ioc Bot.bot ↑↑x) (Set.Ioc Bot.bot ↑x)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma image_coe_Iio (x : ℝ) : Real.toEReal '' Iio x = Ioo ⊥ ↑x := by
  /-
    x : Real
    ⊢ Eq (Set.image Real.toEReal (Set.Iio x)) (Set.Ioo Bot.bot ↑x)
  -/
  refine (image_comp WithBot.some WithTop.some _).trans ?_
  /-
    x : Real
    ⊢ Eq (Set.image WithBot.some (Set.image WithTop.some (Set.Iio x))) (Set.Ioo Bo …
  -/
  rw [WithTop.image_coe_Iio, WithBot.image_coe_Iio]
  /-
    x : Real
    ⊢ Eq (Set.Ioo Bot.bot ↑↑x) (Set.Ioo Bot.bot ↑x)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ici (x : ℝ) : Real.toEReal ⁻¹' Ici x = Ici x := by
  /-
    x : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ici ↑x)) (Set.Ici x)
  -/
  change (WithBot.some ∘ WithTop.some) ⁻¹' (Ici (WithBot.some (WithTop.some x))) = _
  /-
    x : Real
    ⊢ Eq (Set.preimage (Function.comp WithBot.some WithTop.some) (Set.Ici ↑↑x)) (S …
  -/
  refine preimage_comp.trans ?_
  /-
    x : Real
    ⊢ Eq (Set.preimage WithTop.some (Set.preimage WithBot.some (Set.Ici ↑↑x))) (Se …
  -/
  simp only [WithBot.preimage_coe_Ici, WithTop.preimage_coe_Ici]
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ioi (x : ℝ) : Real.toEReal ⁻¹' Ioi x = Ioi x := by
  /-
    x : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ioi ↑x)) (Set.Ioi x)
  -/
  change (WithBot.some ∘ WithTop.some) ⁻¹' (Ioi (WithBot.some (WithTop.some x))) = _
  /-
    x : Real
    ⊢ Eq (Set.preimage (Function.comp WithBot.some WithTop.some) (Set.Ioi ↑↑x)) (S …
  -/
  refine preimage_comp.trans ?_
  /-
    x : Real
    ⊢ Eq (Set.preimage WithTop.some (Set.preimage WithBot.some (Set.Ioi ↑↑x))) (Se …
  -/
  simp only [WithBot.preimage_coe_Ioi, WithTop.preimage_coe_Ioi]
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ioi_bot : Real.toEReal ⁻¹' Ioi ⊥ = univ := by
  /-
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ioi Bot.bot)) Set.univ
  -/
  change (WithBot.some ∘ WithTop.some) ⁻¹' (Ioi ⊥) = _
  /-
    ⊢ Eq (Set.preimage (Function.comp WithBot.some WithTop.some) (Set.Ioi Bot.bot) …
  -/
  refine preimage_comp.trans ?_
  /-
    ⊢ Eq (Set.preimage WithTop.some (Set.preimage WithBot.some (Set.Ioi Bot.bot))) …
  -/
  simp only [WithBot.preimage_coe_Ioi_bot, preimage_univ]
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Iic (y : ℝ) : Real.toEReal ⁻¹' Iic y = Iic y := by
  /-
    y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Iic ↑y)) (Set.Iic y)
  -/
  change (WithBot.some ∘ WithTop.some) ⁻¹' (Iic (WithBot.some (WithTop.some y))) = _
  /-
    y : Real
    ⊢ Eq (Set.preimage (Function.comp WithBot.some WithTop.some) (Set.Iic ↑↑y)) (S …
  -/
  refine preimage_comp.trans ?_
  /-
    y : Real
    ⊢ Eq (Set.preimage WithTop.some (Set.preimage WithBot.some (Set.Iic ↑↑y))) (Se …
  -/
  simp only [WithBot.preimage_coe_Iic, WithTop.preimage_coe_Iic]
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Iio (y : ℝ) : Real.toEReal ⁻¹' Iio y = Iio y := by
  /-
    y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Iio ↑y)) (Set.Iio y)
  -/
  change (WithBot.some ∘ WithTop.some) ⁻¹' (Iio (WithBot.some (WithTop.some y))) = _
  /-
    y : Real
    ⊢ Eq (Set.preimage (Function.comp WithBot.some WithTop.some) (Set.Iio ↑↑y)) (S …
  -/
  refine preimage_comp.trans ?_
  /-
    y : Real
    ⊢ Eq (Set.preimage WithTop.some (Set.preimage WithBot.some (Set.Iio ↑↑y))) (Se …
  -/
  simp only [WithBot.preimage_coe_Iio, WithTop.preimage_coe_Iio]
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Iio_top : Real.toEReal ⁻¹' Iio ⊤ = univ := by
  /-
    ⊢ Eq (Set.preimage Real.toEReal (Set.Iio Top.top)) Set.univ
  -/
  change (WithBot.some ∘ WithTop.some) ⁻¹' (Iio (WithBot.some ⊤)) = _
  /-
    ⊢ Eq (Set.preimage (Function.comp WithBot.some WithTop.some) (Set.Iio ↑Top.top …
  -/
  refine preimage_comp.trans ?_
  /-
    ⊢ Eq (Set.preimage WithTop.some (Set.preimage WithBot.some (Set.Iio ↑Top.top)) …
  -/
  simp only [WithBot.preimage_coe_Iio, WithTop.preimage_coe_Iio_top]
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Icc (x y : ℝ) : Real.toEReal ⁻¹' Icc x y = Icc x y := by
  /-
    x y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Icc ↑x ↑y)) (Set.Icc x y)
  -/
  simp_rw [← Ici_inter_Iic]
  /-
    x y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Inter.inter (Set.Ici ↑x) (Set.Iic ↑y))) (Inte …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ico (x y : ℝ) : Real.toEReal ⁻¹' Ico x y = Ico x y := by
  /-
    x y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ico ↑x ↑y)) (Set.Ico x y)
  -/
  simp_rw [← Ici_inter_Iio]
  /-
    x y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Inter.inter (Set.Ici ↑x) (Set.Iio ↑y))) (Inte …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ioc (x y : ℝ) : Real.toEReal ⁻¹' Ioc x y = Ioc x y := by
  /-
    x y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ioc ↑x ↑y)) (Set.Ioc x y)
  -/
  simp_rw [← Ioi_inter_Iic]
  /-
    x y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Inter.inter (Set.Ioi ↑x) (Set.Iic ↑y))) (Inte …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ioo (x y : ℝ) : Real.toEReal ⁻¹' Ioo x y = Ioo x y := by
  /-
    x y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ioo ↑x ↑y)) (Set.Ioo x y)
  -/
  simp_rw [← Ioi_inter_Iio]
  /-
    x y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Inter.inter (Set.Ioi ↑x) (Set.Iio ↑y))) (Inte …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ico_top (x : ℝ) : Real.toEReal ⁻¹' Ico x ⊤ = Ici x := by
  /-
    x : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ico (↑x) Top.top)) (Set.Ici x)
  -/
  rw [← Ici_inter_Iio]
  /-
    x : Real
    ⊢ Eq (Set.preimage Real.toEReal (Inter.inter (Set.Ici ↑x) (Set.Iio Top.top)))  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ioo_top (x : ℝ) : Real.toEReal ⁻¹' Ioo x ⊤ = Ioi x := by
  /-
    x : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ioo (↑x) Top.top)) (Set.Ioi x)
  -/
  rw [← Ioi_inter_Iio]
  /-
    x : Real
    ⊢ Eq (Set.preimage Real.toEReal (Inter.inter (Set.Ioi ↑x) (Set.Iio Top.top)))  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ioc_bot (y : ℝ) : Real.toEReal ⁻¹' Ioc ⊥ y = Iic y := by
  /-
    y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ioc Bot.bot ↑y)) (Set.Iic y)
  -/
  rw [← Ioi_inter_Iic]
  /-
    y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Inter.inter (Set.Ioi Bot.bot) (Set.Iic ↑y)))  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ioo_bot (y : ℝ) : Real.toEReal ⁻¹' Ioo ⊥ y = Iio y := by
  /-
    y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ioo Bot.bot ↑y)) (Set.Iio y)
  -/
  rw [← Ioi_inter_Iio]
  /-
    y : Real
    ⊢ Eq (Set.preimage Real.toEReal (Inter.inter (Set.Ioi Bot.bot) (Set.Iio ↑y)))  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_coe_Ioo_bot_top : Real.toEReal ⁻¹' Ioo ⊥ ⊤ = univ := by
  /-
    ⊢ Eq (Set.preimage Real.toEReal (Set.Ioo Bot.bot Top.top)) Set.univ
  -/
  rw [← Ioi_inter_Iio]
  /-
    ⊢ Eq (Set.preimage Real.toEReal (Inter.inter (Set.Ioi Bot.bot) (Set.Iio Top.to …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toReal_coe_ennreal : ∀ {x : ℝ≥0∞}, toReal (x : EReal) = ENNReal.toReal x
  | ⊤ => rfl
  | .some _ => rfl


@[simp]
theorem coe_ennreal_ofReal {x : ℝ} : (ENNReal.ofReal x : EReal) = max x 0 :=
  rfl


lemma coe_ennreal_toReal {x : ℝ≥0∞} (hx : x ≠ ∞) : (x.toReal : EReal) = x := by
  /-
    x : ENNReal
    hx : Ne x Top.top
    ⊢ Eq ↑x.toReal ↑x
  -/
  lift x to ℝ≥0 using hx
  /-
    case intro
    x : NNReal
    ⊢ Eq ↑(↑x).toReal ↑↑x
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem coe_nnreal_eq_coe_real (x : ℝ≥0) : ((x : ℝ≥0∞) : EReal) = (x : ℝ) :=
  rfl


@[simp, norm_cast]
theorem coe_ennreal_zero : ((0 : ℝ≥0∞) : EReal) = 0 :=
  rfl


@[simp, norm_cast]
theorem coe_ennreal_one : ((1 : ℝ≥0∞) : EReal) = 1 :=
  rfl


@[simp, norm_cast]
theorem coe_ennreal_top : ((⊤ : ℝ≥0∞) : EReal) = ⊤ :=
  rfl


theorem coe_ennreal_strictMono : StrictMono ((↑) : ℝ≥0∞ → EReal) :=
  WithTop.strictMono_iff.2 ⟨fun _ _ => EReal.coe_lt_coe_iff.2, fun _ => coe_lt_top _⟩


theorem coe_ennreal_injective : Injective ((↑) : ℝ≥0∞ → EReal) :=
  coe_ennreal_strictMono.injective


@[simp]
theorem coe_ennreal_eq_top_iff {x : ℝ≥0∞} : (x : EReal) = ⊤ ↔ x = ⊤ :=
  coe_ennreal_injective.eq_iff' rfl


theorem coe_nnreal_ne_top (x : ℝ≥0) : ((x : ℝ≥0∞) : EReal) ≠ ⊤ := coe_ne_top x


@[simp]
theorem coe_nnreal_lt_top (x : ℝ≥0) : ((x : ℝ≥0∞) : EReal) < ⊤ := coe_lt_top x


@[simp, norm_cast]
theorem coe_ennreal_le_coe_ennreal_iff {x y : ℝ≥0∞} : (x : EReal) ≤ (y : EReal) ↔ x ≤ y :=
  coe_ennreal_strictMono.le_iff_le


@[simp, norm_cast]
theorem coe_ennreal_lt_coe_ennreal_iff {x y : ℝ≥0∞} : (x : EReal) < (y : EReal) ↔ x < y :=
  coe_ennreal_strictMono.lt_iff_lt


@[simp, norm_cast]
theorem coe_ennreal_eq_coe_ennreal_iff {x y : ℝ≥0∞} : (x : EReal) = (y : EReal) ↔ x = y :=
  coe_ennreal_injective.eq_iff


theorem coe_ennreal_ne_coe_ennreal_iff {x y : ℝ≥0∞} : (x : EReal) ≠ (y : EReal) ↔ x ≠ y :=
  coe_ennreal_injective.ne_iff


@[simp, norm_cast]
theorem coe_ennreal_eq_zero {x : ℝ≥0∞} : (x : EReal) = 0 ↔ x = 0 := by
  /-
    x : ENNReal
    ⊢ Iff (Eq (↑x) 0) (Eq x 0)
  -/
  rw [← coe_ennreal_eq_coe_ennreal_iff, coe_ennreal_zero]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_ennreal_eq_one {x : ℝ≥0∞} : (x : EReal) = 1 ↔ x = 1 := by
  /-
    x : ENNReal
    ⊢ Iff (Eq (↑x) 1) (Eq x 1)
  -/
  rw [← coe_ennreal_eq_coe_ennreal_iff, coe_ennreal_one]
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem coe_ennreal_ne_zero {x : ℝ≥0∞} : (x : EReal) ≠ 0 ↔ x ≠ 0 :=
  coe_ennreal_eq_zero.not


@[norm_cast]
theorem coe_ennreal_ne_one {x : ℝ≥0∞} : (x : EReal) ≠ 1 ↔ x ≠ 1 :=
  coe_ennreal_eq_one.not


theorem coe_ennreal_nonneg (x : ℝ≥0∞) : (0 : EReal) ≤ x :=
  coe_ennreal_le_coe_ennreal_iff.2 (zero_le x)


@[simp] theorem range_coe_ennreal : range ((↑) : ℝ≥0∞ → EReal) = Set.Ici 0 :=
  Subset.antisymm (range_subset_iff.2 coe_ennreal_nonneg) fun x => match x with
    | ⊥ => fun h => absurd h bot_lt_zero.not_le
    | ⊤ => fun _ => ⟨⊤, rfl⟩
    | (x : ℝ) => fun h => ⟨.some ⟨x, EReal.coe_nonneg.1 h⟩, rfl⟩


instance : CanLift EReal ℝ≥0∞ (↑) (0 ≤ ·) := ⟨range_coe_ennreal.ge⟩


@[simp, norm_cast]
theorem coe_ennreal_pos {x : ℝ≥0∞} : (0 : EReal) < x ↔ 0 < x := by
  /-
    x : ENNReal
    ⊢ Iff (LT.lt 0 ↑x) (LT.lt 0 x)
  -/
  rw [← coe_ennreal_zero, coe_ennreal_lt_coe_ennreal_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem bot_lt_coe_ennreal (x : ℝ≥0∞) : (⊥ : EReal) < x :=
  (bot_lt_coe 0).trans_le (coe_ennreal_nonneg _)


@[simp]
theorem coe_ennreal_ne_bot (x : ℝ≥0∞) : (x : EReal) ≠ ⊥ :=
  (bot_lt_coe_ennreal x).ne'


@[simp, norm_cast]
theorem coe_ennreal_add (x y : ENNReal) : ((x + y : ℝ≥0∞) : EReal) = x + y := by
  /-
    x y : ENNReal
    ⊢ Eq (↑(HAdd.hAdd x y)) (HAdd.hAdd ↑x ↑y)
  -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
  cases x <;> cases y <;> rfl
                          /-
                            🎉 no goals
                          -/


private theorem coe_ennreal_top_mul (x : ℝ≥0) : ((⊤ * x : ℝ≥0∞) : EReal) = ⊤ * x := by
  /-
    x : NNReal
    ⊢ Eq (↑(HMul.hMul Top.top ↑x)) (HMul.hMul Top.top ↑↑x)
  -/
  rcases eq_or_ne x 0 with (rfl | h0)
    /-
      case inl
      ⊢ Eq (↑(HMul.hMul Top.top ↑0)) (HMul.hMul Top.top ↑↑0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : NNReal
      h0 : Ne x 0
      ⊢ Eq (↑(HMul.hMul Top.top ↑x)) (HMul.hMul Top.top ↑↑x)
    -/
  · rw [ENNReal.top_mul (ENNReal.coe_ne_zero.2 h0)]
    /-
      case inr
      x : NNReal
      h0 : Ne x 0
      ⊢ Eq (↑Top.top) (HMul.hMul Top.top ↑↑x)
    -/
    exact Eq.symm <| if_pos <| NNReal.coe_pos.2 h0.bot_lt
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem coe_ennreal_mul : ∀ x y : ℝ≥0∞, ((x * y : ℝ≥0∞) : EReal) = (x : EReal) * y
  | ⊤, ⊤ => rfl
  | ⊤, (y : ℝ≥0) => coe_ennreal_top_mul y
  | (x : ℝ≥0), ⊤ => by
    /-
      x : NNReal
      ⊢ Eq (↑(HMul.hMul (↑x) Top.top)) (HMul.hMul ↑↑x ↑Top.top)
    -/
    rw [mul_comm, coe_ennreal_top_mul, EReal.mul_comm, coe_ennreal_top]
    /-
      🎉 no goals
    -/
  | (x : ℝ≥0), (y : ℝ≥0) => by
    /-
      x y : NNReal
      ⊢ Eq (↑(HMul.hMul ↑x ↑y)) (HMul.hMul ↑↑x ↑↑y)
    -/
    simp only [← ENNReal.coe_mul, coe_nnreal_eq_coe_real, NNReal.coe_mul, EReal.coe_mul]
    /-
      🎉 no goals
    -/


@[norm_cast]
theorem coe_ennreal_nsmul (n : ℕ) (x : ℝ≥0∞) : (↑(n • x) : EReal) = n • (x : EReal) :=
  map_nsmul (⟨⟨(↑), coe_ennreal_zero⟩, coe_ennreal_add⟩ : ℝ≥0∞ →+ EReal) _ _


theorem coe_coe_eq_natCast (n : ℕ) : (n : ℝ) = (n : EReal) := rfl


theorem natCast_ne_bot (n : ℕ) : (n : EReal) ≠ ⊥ := Ne.symm (ne_of_beq_false rfl)


theorem natCast_ne_top (n : ℕ) : (n : EReal) ≠ ⊤ := Ne.symm (ne_of_beq_false rfl)


@[norm_cast]
theorem natCast_eq_iff {m n : ℕ} : (m : EReal) = (n : EReal) ↔ m = n := by
  /-
    m n : Nat
    ⊢ Iff (Eq ↑m ↑n) (Eq m n)
  -/
  rw [← coe_coe_eq_natCast n, ← coe_coe_eq_natCast m, EReal.coe_eq_coe_iff, Nat.cast_inj]
  /-
    🎉 no goals
  -/


theorem natCast_ne_iff {m n : ℕ} : (m : EReal) ≠ (n : EReal) ↔ m ≠ n :=
  not_iff_not.2 natCast_eq_iff


@[norm_cast]
theorem natCast_le_iff {m n : ℕ} : (m : EReal) ≤ (n : EReal) ↔ m ≤ n := by
  /-
    m n : Nat
    ⊢ Iff (LE.le ↑m ↑n) (LE.le m n)
  -/
  rw [← coe_coe_eq_natCast n, ← coe_coe_eq_natCast m, EReal.coe_le_coe_iff, Nat.cast_le]
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem natCast_lt_iff {m n : ℕ} : (m : EReal) < (n : EReal) ↔ m < n := by
  /-
    m n : Nat
    ⊢ Iff (LT.lt ↑m ↑n) (LT.lt m n)
  -/
  rw [← coe_coe_eq_natCast n, ← coe_coe_eq_natCast m, EReal.coe_lt_coe_iff, Nat.cast_lt]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem natCast_mul (m n : ℕ) :
    (m * n : ℕ) = (m : EReal) * (n : EReal) := by
  /-
    m n : Nat
    ⊢ Eq (↑(HMul.hMul m n)) (HMul.hMul ↑m ↑n)
  -/
  rw [← coe_coe_eq_natCast, ← coe_coe_eq_natCast, ← coe_coe_eq_natCast, Nat.cast_mul, EReal.coe_mul]
  /-
    🎉 no goals
  -/


theorem exists_rat_btwn_of_lt :
    ∀ {a b : EReal}, a < b → ∃ x : ℚ, a < (x : ℝ) ∧ ((x : ℝ) : EReal) < b
  | ⊤, _, h => (not_top_lt h).elim
  | (a : ℝ), ⊥, h => (lt_irrefl _ ((bot_lt_coe a).trans h)).elim
                              /-
                                a b : Real
                                h : LT.lt ↑a ↑b
                                ⊢ Exists fun x => And (LT.lt ↑a ↑↑x) (LT.lt ↑↑x ↑b)
                              -/
  | (a : ℝ), (b : ℝ), h => by simp [exists_rat_btwn (EReal.coe_lt_coe_iff.1 h)]
                              /-
                                🎉 no goals
                              -/
  | (a : ℝ), ⊤, _ =>
    let ⟨b, hab⟩ := exists_rat_gt a
           /-
             a : Real
             x✝ : LT.lt (↑a) Top.top
             b : Rat
             hab : LT.lt a ↑b
             ⊢ LT.lt ↑a ↑↑b
           -/
    ⟨b, by simpa using hab, coe_lt_top _⟩
           /-
             🎉 no goals
           -/
  | ⊥, ⊥, h => (lt_irrefl _ h).elim
  | ⊥, (a : ℝ), _ =>
    let ⟨b, hab⟩ := exists_rat_lt a
                         /-
                           a : Real
                           x✝ : LT.lt Bot.bot ↑a
                           b : Rat
                           hab : LT.lt (↑b) a
                           ⊢ LT.lt ↑↑b ↑a
                         -/
    ⟨b, bot_lt_coe _, by simpa using hab⟩
                         /-
                           🎉 no goals
                         -/
  | ⊥, ⊤, _ => ⟨0, bot_lt_coe _, coe_lt_top _⟩


theorem lt_iff_exists_rat_btwn {a b : EReal} :
    a < b ↔ ∃ x : ℚ, a < (x : ℝ) ∧ ((x : ℝ) : EReal) < b :=
  ⟨fun hab => exists_rat_btwn_of_lt hab, fun ⟨_x, ax, xb⟩ => ax.trans xb⟩


theorem lt_iff_exists_real_btwn {a b : EReal} : a < b ↔ ∃ x : ℝ, a < x ∧ (x : EReal) < b :=
  ⟨fun hab =>
    let ⟨x, ax, xb⟩ := exists_rat_btwn_of_lt hab
    ⟨(x : ℝ), ax, xb⟩,
    fun ⟨_x, ax, xb⟩ => ax.trans xb⟩


/-- The set of numbers in `EReal` that are not equal to `±∞` is equivalent to `ℝ`. -/
def neTopBotEquivReal : ({⊥, ⊤}ᶜ : Set EReal) ≃ ℝ where
  toFun x := EReal.toReal x
                     /-
                       x : Real
                       ⊢ Membership.mem (HasCompl.compl (Insert.insert Bot.bot (Singleton.singleton T …
                     -/
  invFun x := ⟨x, by simp⟩
                     /-
                       🎉 no goals
                     -/
  left_inv := fun ⟨x, hx⟩ => by
    /-
      x✝ : ↑(HasCompl.compl (Insert.insert Bot.bot (Singleton.singleton Top.top)))
      x : EReal
      hx : Membership.mem (HasCompl.compl (Insert.insert Bot.bot (Singleton.singleto …
      ⊢ Eq ((fun x => ⟨↑x, ⋯⟩) ((fun x => (↑x).toReal) ⟨x, hx⟩)) ⟨x, hx⟩
    -/
    lift x to ℝ
      /-
        x✝ : ↑(HasCompl.compl (Insert.insert Bot.bot (Singleton.singleton Top.top)))
        x : EReal
        hx : Membership.mem (HasCompl.compl (Insert.insert Bot.bot (Singleton.singleto …
        ⊢ And (Ne x Top.top) (Ne x Bot.bot)
      -/
    · simpa [not_or, and_comm] using hx
      /-
        🎉 no goals
      -/
      /-
        case intro
        x✝ : ↑(HasCompl.compl (Insert.insert Bot.bot (Singleton.singleton Top.top)))
        x : Real
        hx : Membership.mem (HasCompl.compl (Insert.insert Bot.bot (Singleton.singleto …
        ⊢ Eq ((fun x => ⟨↑x, ⋯⟩) ((fun x => (↑x).toReal) ⟨↑x, hx⟩)) ⟨↑x, hx⟩
      -/
    · simp
      /-
        🎉 no goals
      -/
                    /-
                      x : Real
                      ⊢ Eq ((fun x => (↑x).toReal) ((fun x => ⟨↑x, ⋯⟩) x)) x
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem add_bot (x : EReal) : x + ⊥ = ⊥ :=
  WithBot.add_bot _


@[simp]
theorem bot_add (x : EReal) : ⊥ + x = ⊥ :=
  WithBot.bot_add _


@[simp]
theorem add_eq_bot_iff {x y : EReal} : x + y = ⊥ ↔ x = ⊥ ∨ y = ⊥ :=
  WithBot.add_eq_bot


lemma add_ne_bot_iff {x y : EReal} : x + y ≠ ⊥ ↔ x ≠ ⊥ ∧ y ≠ ⊥ := WithBot.add_ne_bot


@[simp]
theorem bot_lt_add_iff {x y : EReal} : ⊥ < x + y ↔ ⊥ < x ∧ ⊥ < y := by
  /-
    x y : EReal
    ⊢ Iff (LT.lt Bot.bot (HAdd.hAdd x y)) (And (LT.lt Bot.bot x) (LT.lt Bot.bot y))
  -/
  simp [bot_lt_iff_ne_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem top_add_top : (⊤ : EReal) + ⊤ = ⊤ :=
  rfl


@[simp]
theorem top_add_coe (x : ℝ) : (⊤ : EReal) + x = ⊤ :=
  rfl


/-- For any extended real number `x` which is not `⊥`, the sum of `⊤` and `x` is equal to `⊤`. -/
@[simp]
theorem top_add_of_ne_bot {x : EReal} (h : x ≠ ⊥) : ⊤ + x = ⊤ := by
  /-
    x : EReal
    h : Ne x Bot.bot
    ⊢ Eq (HAdd.hAdd Top.top x) Top.top
  -/
  induction x
    /-
      case h_bot
      h : Ne Bot.bot Bot.bot
      ⊢ Eq (HAdd.hAdd Top.top Bot.bot) Top.top
    -/
  · exfalso; exact h (Eq.refl ⊥)
             /-
               🎉 no goals
             -/
    /-
      case h_real
      a✝ : Real
      h : Ne (↑a✝) Bot.bot
      ⊢ Eq (HAdd.hAdd Top.top ↑a✝) Top.top
    -/
  · exact top_add_coe _
    /-
      🎉 no goals
    -/
    /-
      case h_top
      h : Ne Top.top Bot.bot
      ⊢ Eq (HAdd.hAdd Top.top Top.top) Top.top
    -/
  · exact top_add_top
    /-
      🎉 no goals
    -/


/-- For any extended real number `x`, the sum of `⊤` and `x` is equal to `⊤`
if and only if `x` is not `⊥`. -/
theorem top_add_iff_ne_bot {x : EReal} : ⊤ + x = ⊤ ↔ x ≠ ⊥ := by
  /-
    x : EReal
    ⊢ Iff (Eq (HAdd.hAdd Top.top x) Top.top) (Ne x Bot.bot)
  -/
  constructor <;> intro h
    /-
      case mp
      x : EReal
      h : Eq (HAdd.hAdd Top.top x) Top.top
      ⊢ Ne x Bot.bot
    -/
  · rintro rfl
    /-
      case mp
      h : Eq (HAdd.hAdd Top.top Bot.bot) Top.top
      ⊢ False
    -/
    rw [add_bot] at h
    /-
      case mp
      h : Eq Bot.bot Top.top
      ⊢ False
    -/
    exact bot_ne_top h
    /-
      🎉 no goals
    -/
  · cases x with
    | h_bot => contradiction
    | h_top => rfl
    | h_real r => exact top_add_of_ne_bot h


/-- For any extended real number `x` which is not `⊥`, the sum of `x` and `⊤` is equal to `⊤`. -/
@[simp]
theorem add_top_of_ne_bot {x : EReal} (h : x ≠ ⊥) : x + ⊤ = ⊤ := by
  /-
    x : EReal
    h : Ne x Bot.bot
    ⊢ Eq (HAdd.hAdd x Top.top) Top.top
  -/
  rw [add_comm, top_add_of_ne_bot h]
  /-
    🎉 no goals
  -/


/-- For any extended real number `x`, the sum of `x` and `⊤` is equal to `⊤`
if and only if `x` is not `⊥`. -/
                                                                 /-
                                                                   x : EReal
                                                                   ⊢ Iff (Eq (HAdd.hAdd x Top.top) Top.top) (Ne x Bot.bot)
                                                                 -/
theorem add_top_iff_ne_bot {x : EReal} : x + ⊤ = ⊤ ↔ x ≠ ⊥ := by rw [add_comm, top_add_iff_ne_bot]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- For any two extended real numbers `a` and `b`, if both `a` and `b` are greater than `0`,
then their sum is also greater than `0`. -/
theorem add_pos {a b : EReal} (ha : 0 < a) (hb : 0 < b) : 0 < a + b := by
  /-
    a b : EReal
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ LT.lt 0 (HAdd.hAdd a b)
  -/
  induction a
    /-
      case h_bot
      b : EReal
      hb : LT.lt 0 b
      ha : LT.lt 0 Bot.bot
      ⊢ LT.lt 0 (HAdd.hAdd Bot.bot b)
    -/
  · exfalso; exact not_lt_bot ha
             /-
               🎉 no goals
             -/
    /-
      case h_real
      b : EReal
      hb : LT.lt 0 b
      a✝ : Real
      ha : LT.lt 0 ↑a✝
      ⊢ LT.lt 0 (HAdd.hAdd (↑a✝) b)
    -/
  · induction b
      /-
        case h_real.h_bot
        a✝ : Real
        ha : LT.lt 0 ↑a✝
        hb : LT.lt 0 Bot.bot
        ⊢ LT.lt 0 (HAdd.hAdd (↑a✝) Bot.bot)
      -/
    · exfalso; exact not_lt_bot hb
               /-
                 🎉 no goals
               -/
      /-
        case h_real.h_real
        a✝¹ : Real
        ha : LT.lt 0 ↑a✝¹
        a✝ : Real
        hb : LT.lt 0 ↑a✝
        ⊢ LT.lt 0 (HAdd.hAdd ↑a✝¹ ↑a✝)
      -/
    · norm_cast at *; exact Left.add_pos ha hb
                      /-
                        🎉 no goals
                      -/
      /-
        case h_real.h_top
        a✝ : Real
        ha : LT.lt 0 ↑a✝
        hb : LT.lt 0 Top.top
        ⊢ LT.lt 0 (HAdd.hAdd (↑a✝) Top.top)
      -/
    · exact add_top_of_ne_bot (bot_lt_zero.trans ha).ne' ▸ hb
      /-
        🎉 no goals
      -/
    /-
      case h_top
      b : EReal
      hb : LT.lt 0 b
      ha : LT.lt 0 Top.top
      ⊢ LT.lt 0 (HAdd.hAdd Top.top b)
    -/
  · rw [top_add_of_ne_bot (bot_lt_zero.trans hb).ne']
    /-
      case h_top
      b : EReal
      hb : LT.lt 0 b
      ha : LT.lt 0 Top.top
      ⊢ LT.lt 0 Top.top
    -/
    exact ha
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_add_top (x : ℝ) : (x : EReal) + ⊤ = ⊤ :=
  rfl


theorem toReal_add {x y : EReal} (hx : x ≠ ⊤) (h'x : x ≠ ⊥) (hy : y ≠ ⊤) (h'y : y ≠ ⊥) :
    toReal (x + y) = toReal x + toReal y := by
  /-
    x y : EReal
    hx : Ne x Top.top
    h'x : Ne x Bot.bot
    hy : Ne y Top.top
    h'y : Ne y Bot.bot
    ⊢ Eq (HAdd.hAdd x y).toReal (HAdd.hAdd x.toReal y.toReal)
  -/
  lift x to ℝ using ⟨hx, h'x⟩
  /-
    case intro
    y : EReal
    hy : Ne y Top.top
    h'y : Ne y Bot.bot
    x : Real
    hx : Ne (↑x) Top.top
    h'x : Ne (↑x) Bot.bot
    ⊢ Eq (HAdd.hAdd (↑x) y).toReal (HAdd.hAdd (↑x).toReal y.toReal)
  -/
  lift y to ℝ using ⟨hy, h'y⟩
  /-
    case intro.intro
    x : Real
    hx : Ne (↑x) Top.top
    h'x : Ne (↑x) Bot.bot
    y : Real
    hy : Ne (↑y) Top.top
    h'y : Ne (↑y) Bot.bot
    ⊢ Eq (HAdd.hAdd ↑x ↑y).toReal (HAdd.hAdd (↑x).toReal (↑y).toReal)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem addLECancellable_coe (x : ℝ) : AddLECancellable (x : EReal)
  | _, ⊤, _ => le_top
  | ⊥, _, _ => bot_le
                        /-
                          x z : Real
                          h : LE.le (HAdd.hAdd (↑x) Top.top) (HAdd.hAdd ↑x ↑z)
                          ⊢ LE.le Top.top ↑z
                        -/
  | ⊤, (z : ℝ), h => by simp only [coe_add_top, ← coe_add, top_le_iff, coe_ne_top] at h
                        /-
                          🎉 no goals
                        -/
                  /-
                    x : Real
                    x✝ : EReal
                    h : LE.le (HAdd.hAdd (↑x) x✝) (HAdd.hAdd (↑x) Bot.bot)
                    ⊢ LE.le x✝ Bot.bot
                  -/
  | _, ⊥, h => by simpa using h
                  /-
                    🎉 no goals
                  -/
  | (y : ℝ), (z : ℝ), h => by
    /-
      x y z : Real
      h : LE.le (HAdd.hAdd ↑x ↑y) (HAdd.hAdd ↑x ↑z)
      ⊢ LE.le ↑y ↑z
    -/
    simpa only [← coe_add, EReal.coe_le_coe_iff, add_le_add_iff_left] using h
    /-
      🎉 no goals
    -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: add `MulLECancellable.strictMono*` etc

theorem add_lt_add_right_coe {x y : EReal} (h : x < y) (z : ℝ) : x + z < y + z :=
  not_le.1 <| mt (addLECancellable_coe z).add_le_add_iff_right.1 h.not_le


theorem add_lt_add_left_coe {x y : EReal} (h : x < y) (z : ℝ) : (z : EReal) + x < z + y := by
  /-
    x y : EReal
    h : LT.lt x y
    z : Real
    ⊢ LT.lt (HAdd.hAdd (↑z) x) (HAdd.hAdd (↑z) y)
  -/
  simpa [add_comm] using add_lt_add_right_coe h z
  /-
    🎉 no goals
  -/


theorem add_lt_add {x y z t : EReal} (h1 : x < y) (h2 : z < t) : x + z < y + t := by
  /-
    x y z t : EReal
    h1 : LT.lt x y
    h2 : LT.lt z t
    ⊢ LT.lt (HAdd.hAdd x z) (HAdd.hAdd y t)
  -/
  rcases eq_or_ne x ⊥ with (rfl | hx)
    /-
      case inl
      y z t : EReal
      h2 : LT.lt z t
      h1 : LT.lt Bot.bot y
      ⊢ LT.lt (HAdd.hAdd Bot.bot z) (HAdd.hAdd y t)
    -/
  · simp [h1, bot_le.trans_lt h2]
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z t : EReal
      h1 : LT.lt x y
      h2 : LT.lt z t
      hx : Ne x Bot.bot
      ⊢ LT.lt (HAdd.hAdd x z) (HAdd.hAdd y t)
    -/
  · lift x to ℝ using ⟨h1.ne_top, hx⟩
    calc (x : EReal) + z < x + t := add_lt_add_left_coe h2 _
    _ ≤ y + t := add_le_add_right h1.le _


theorem add_lt_add_of_lt_of_le' {x y z t : EReal} (h : x < y) (h' : z ≤ t) (hbot : t ≠ ⊥)
    (htop : t = ⊤ → z = ⊤ → x = ⊥) : x + z < y + t := by
  /-
    x y z t : EReal
    h : LT.lt x y
    h' : LE.le z t
    hbot : Ne t Bot.bot
    htop : Eq t Top.top → Eq z Top.top → Eq x Bot.bot
    ⊢ LT.lt (HAdd.hAdd x z) (HAdd.hAdd y t)
  -/
  rcases h'.eq_or_lt with (rfl | hlt)
    /-
      case inl
      x y z : EReal
      h : LT.lt x y
      h' : LE.le z z
      hbot : Ne z Bot.bot
      htop : Eq z Top.top → Eq z Top.top → Eq x Bot.bot
      ⊢ LT.lt (HAdd.hAdd x z) (HAdd.hAdd y z)
    -/
  · rcases eq_or_ne z ⊤ with (rfl | hz)
      /-
        case inl.inl
        x y : EReal
        h : LT.lt x y
        h' : LE.le Top.top Top.top
        hbot : Ne Top.top Bot.bot
        htop : Eq Top.top Top.top → Eq Top.top Top.top → Eq x Bot.bot
        ⊢ LT.lt (HAdd.hAdd x Top.top) (HAdd.hAdd y Top.top)
      -/
    · obtain rfl := htop rfl rfl
      /-
        case inl.inl
        y : EReal
        h' : LE.le Top.top Top.top
        hbot : Ne Top.top Bot.bot
        h : LT.lt Bot.bot y
        htop : Eq Top.top Top.top → Eq Top.top Top.top → Eq Bot.bot Bot.bot
        ⊢ LT.lt (HAdd.hAdd Bot.bot Top.top) (HAdd.hAdd y Top.top)
      -/
      simpa
      /-
        🎉 no goals
      -/
    /-
      case inl.inr
      x y z : EReal
      h : LT.lt x y
      h' : LE.le z z
      hbot : Ne z Bot.bot
      htop : Eq z Top.top → Eq z Top.top → Eq x Bot.bot
      hz : Ne z Top.top
      ⊢ LT.lt (HAdd.hAdd x z) (HAdd.hAdd y z)
    -/
    lift z to ℝ using ⟨hz, hbot⟩
    /-
      case inl.inr.intro
      x y : EReal
      h : LT.lt x y
      z : Real
      h' : LE.le ↑z ↑z
      hbot : Ne (↑z) Bot.bot
      htop : Eq (↑z) Top.top → Eq (↑z) Top.top → Eq x Bot.bot
      hz : Ne (↑z) Top.top
      ⊢ LT.lt (HAdd.hAdd x ↑z) (HAdd.hAdd y ↑z)
    -/
    exact add_lt_add_right_coe h z
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z t : EReal
      h : LT.lt x y
      h' : LE.le z t
      hbot : Ne t Bot.bot
      htop : Eq t Top.top → Eq z Top.top → Eq x Bot.bot
      hlt : LT.lt z t
      ⊢ LT.lt (HAdd.hAdd x z) (HAdd.hAdd y t)
    -/
  · exact add_lt_add h hlt
    /-
      🎉 no goals
    -/


/-- See also `EReal.add_lt_add_of_lt_of_le'` for a version with weaker but less convenient
assumptions. -/
theorem add_lt_add_of_lt_of_le {x y z t : EReal} (h : x < y) (h' : z ≤ t) (hz : z ≠ ⊥)
    (ht : t ≠ ⊤) : x + z < y + t :=
  add_lt_add_of_lt_of_le' h h' (ne_bot_of_le_ne_bot hz h') fun ht' => (ht ht').elim


theorem add_lt_top {x y : EReal} (hx : x ≠ ⊤) (hy : y ≠ ⊤) : x + y < ⊤ :=
  add_lt_add hx.lt_top hy.lt_top


lemma add_ne_top {x y : EReal} (hx : x ≠ ⊤) (hy : y ≠ ⊤) : x + y ≠ ⊤ :=
  lt_top_iff_ne_top.mp <| add_lt_top hx hy


lemma add_ne_top_iff_ne_top₂ {x y : EReal} (hx : x ≠ ⊥) (hy : y ≠ ⊥) :
    x + y ≠ ⊤ ↔ x ≠ ⊤ ∧ y ≠ ⊤ := by
  /-
    x y : EReal
    hx : Ne x Bot.bot
    hy : Ne y Bot.bot
    ⊢ Iff (Ne (HAdd.hAdd x y) Top.top) (And (Ne x Top.top) (Ne y Top.top))
  -/
  refine ⟨?_, fun h ↦ add_ne_top h.1 h.2⟩
  /-
    x y : EReal
    hx : Ne x Bot.bot
    hy : Ne y Bot.bot
    ⊢ Ne (HAdd.hAdd x y) Top.top → And (Ne x Top.top) (Ne y Top.top)
  -/
  cases x <;> simp_all only [ne_eq, not_false_eq_true, top_add_of_ne_bot, not_true_eq_false,
    IsEmpty.forall_iff]
  /-
    case h_real
    y : EReal
    a✝ : Real
    hy : Not (Eq y Bot.bot)
    hx : Not (Eq (↑a✝) Bot.bot)
    ⊢ Not (Eq (HAdd.hAdd (↑a✝) y) Top.top) → And (Not (Eq (↑a✝) Top.top)) (Not (Eq …
  -/
  cases y <;> simp_all only [not_false_eq_true, ne_eq, add_top_of_ne_bot, not_true_eq_false,
    coe_ne_top, and_self, implies_true]


lemma add_ne_top_iff_ne_top_left {x y : EReal} (hy : y ≠ ⊥) (hy' : y ≠ ⊤) :
    x + y ≠ ⊤ ↔ x ≠ ⊤ := by
  /-
    x y : EReal
    hy : Ne y Bot.bot
    hy' : Ne y Top.top
    ⊢ Iff (Ne (HAdd.hAdd x y) Top.top) (Ne x Top.top)
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases x <;> simp [add_ne_top_iff_ne_top₂, hy, hy']
              /-
                🎉 no goals
              -/


lemma add_ne_top_iff_ne_top_right {x y : EReal} (hx : x ≠ ⊥) (hx' : x ≠ ⊤) :
    x + y ≠ ⊤ ↔ y ≠ ⊤ := add_comm x y ▸ add_ne_top_iff_ne_top_left hx hx'


/-- We do not have a notion of `LinearOrderedAddCommMonoidWithBot` but we can at least make
the order dual of the extended reals into a `LinearOrderedAddCommMonoidWithTop`. -/
instance : LinearOrderedAddCommMonoidWithTop ERealᵒᵈ where
               /-
                 ⊢ ∀ (a : OrderDual EReal), LE.le a Top.top
               -/
  le_top := by simp
               /-
                 🎉 no goals
               -/
  top_add' := by
    /-
      ⊢ ∀ (x : OrderDual EReal), Eq (HAdd.hAdd Top.top x) Top.top
    -/
    rw [OrderDual.forall]
    /-
      ⊢ ∀ (a : EReal), Eq (HAdd.hAdd Top.top (OrderDual.toDual a)) Top.top
    -/
    intro x
    /-
      x : EReal
      ⊢ Eq (HAdd.hAdd Top.top (OrderDual.toDual x)) Top.top
    -/
    rw [← OrderDual.toDual_bot, ← toDual_add, bot_add, OrderDual.toDual_bot]
    /-
      🎉 no goals
    -/


/-- negation on `EReal` -/
protected def neg : EReal → EReal
  | ⊥ => ⊤
  | ⊤ => ⊥
  | (x : ℝ) => (-x : ℝ)


instance : Neg EReal := ⟨EReal.neg⟩


instance : SubNegZeroMonoid EReal where
  neg_zero := congr_arg Real.toEReal neg_zero
  zsmul := zsmulRec


@[simp]
theorem neg_top : -(⊤ : EReal) = ⊥ :=
  rfl


@[simp]
theorem neg_bot : -(⊥ : EReal) = ⊤ :=
  rfl


@[simp, norm_cast] theorem coe_neg (x : ℝ) : (↑(-x) : EReal) = -↑x := rfl


@[simp, norm_cast] theorem coe_sub (x y : ℝ) : (↑(x - y) : EReal) = x - y := rfl


@[norm_cast]
theorem coe_zsmul (n : ℤ) (x : ℝ) : (↑(n • x) : EReal) = n • (x : EReal) :=
  map_zsmul' (⟨⟨(↑), coe_zero⟩, coe_add⟩ : ℝ →+ EReal) coe_neg _ _


instance : InvolutiveNeg EReal where
  neg_neg a :=
    match a with
    | ⊥ => rfl
    | ⊤ => rfl
    | (a : ℝ) => congr_arg Real.toEReal (neg_neg a)


@[simp]
theorem toReal_neg : ∀ {a : EReal}, toReal (-a) = -toReal a
            /-
              ⊢ Eq (Neg.neg Top.top).toReal (Neg.neg Top.top.toReal)
            -/
  | ⊤ => by simp
            /-
              🎉 no goals
            -/
            /-
              ⊢ Eq (Neg.neg Bot.bot).toReal (Neg.neg Bot.bot.toReal)
            -/
  | ⊥ => by simp
            /-
              🎉 no goals
            -/
  | (x : ℝ) => rfl


@[simp]
theorem neg_eq_top_iff {x : EReal} : -x = ⊤ ↔ x = ⊥ :=
  neg_injective.eq_iff' rfl


@[simp]
theorem neg_eq_bot_iff {x : EReal} : -x = ⊥ ↔ x = ⊤ :=
  neg_injective.eq_iff' rfl


@[simp]
theorem neg_eq_zero_iff {x : EReal} : -x = 0 ↔ x = 0 :=
  neg_injective.eq_iff' neg_zero


theorem neg_strictAnti : StrictAnti (- · : EReal → EReal) :=
  WithBot.strictAnti_iff.2 ⟨WithTop.strictAnti_iff.2
    ⟨coe_strictMono.comp_strictAnti fun _ _ => neg_lt_neg, fun _ => bot_lt_coe _⟩,
      WithTop.forall.2 ⟨bot_lt_top, fun _ => coe_lt_top _⟩⟩


@[simp] theorem neg_le_neg_iff {a b : EReal} : -a ≤ -b ↔ b ≤ a := neg_strictAnti.le_iff_le


@[simp] theorem neg_lt_neg_iff {a b : EReal} : -a < -b ↔ b < a := neg_strictAnti.lt_iff_lt


/-- `-a ≤ b` if and only if `-b ≤ a` on `EReal`. -/
protected theorem neg_le {a b : EReal} : -a ≤ b ↔ -b ≤ a := by
 /-
   a b : EReal
   ⊢ Iff (LE.le (Neg.neg a) b) (LE.le (Neg.neg b) a)
 -/
 rw [← neg_le_neg_iff, neg_neg]
 /-
   🎉 no goals
 -/


/-- If `-a ≤ b` then `-b ≤ a` on `EReal`. -/
protected theorem neg_le_of_neg_le {a b : EReal} (h : -a ≤ b) : -b ≤ a := EReal.neg_le.mp h


/-- `a ≤ -b` if and only if `b ≤ -a` on `EReal`. -/
protected theorem le_neg {a b : EReal} : a ≤ -b ↔ b ≤ -a := by
  /-
    a b : EReal
    ⊢ Iff (LE.le a (Neg.neg b)) (LE.le b (Neg.neg a))
  -/
  rw [← neg_le_neg_iff, neg_neg]
  /-
    🎉 no goals
  -/


/-- If `a ≤ -b` then `b ≤ -a` on `EReal`. -/
protected theorem le_neg_of_le_neg {a b : EReal} (h : a ≤ -b) : b ≤ -a := EReal.le_neg.mp h


/-- `-a < b` if and only if `-b < a` on `EReal`. -/
                                                          /-
                                                            a b : EReal
                                                            ⊢ Iff (LT.lt (Neg.neg a) b) (LT.lt (Neg.neg b) a)
                                                          -/
theorem neg_lt_comm {a b : EReal} : -a < b ↔ -b < a := by rw [← neg_lt_neg_iff, neg_neg]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[deprecated (since := "2024-11-19")] alias neg_lt_iff_neg_lt := neg_lt_comm


/-- If `-a < b` then `-b < a` on `EReal`. -/
protected theorem neg_lt_of_neg_lt {a b : EReal} (h : -a < b) : -b < a := neg_lt_comm.mp h


/-- `-a < b` if and only if `-b < a` on `EReal`. -/
theorem lt_neg_comm {a b : EReal} : a < -b ↔ b < -a := by
  /-
    a b : EReal
    ⊢ Iff (LT.lt a (Neg.neg b)) (LT.lt b (Neg.neg a))
  -/
  rw [← neg_lt_neg_iff, neg_neg]
  /-
    🎉 no goals
  -/


/-- If `a < -b` then `b < -a` on `EReal`. -/
protected theorem lt_neg_of_lt_neg {a b : EReal} (h : a < -b) : b < -a := lt_neg_comm.mp h


/-- Negation as an order reversing isomorphism on `EReal`. -/
def negOrderIso : EReal ≃o ERealᵒᵈ :=
  { Equiv.neg EReal with
    toFun := fun x => OrderDual.toDual (-x)
    invFun := fun x => -OrderDual.ofDual x
    map_rel_iff' := neg_le_neg_iff }


lemma neg_add {x y : EReal} (h1 : x ≠ ⊥ ∨ y ≠ ⊤) (h2 : x ≠ ⊤ ∨ y ≠ ⊥) :
    - (x + y) = - x - y := by
  /-
    x y : EReal
    h1 : Or (Ne x Bot.bot) (Ne y Top.top)
    h2 : Or (Ne x Top.top) (Ne y Bot.bot)
    ⊢ Eq (Neg.neg (HAdd.hAdd x y)) (HSub.hSub (Neg.neg x) y)
  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  induction x <;> induction y <;> try tauto
                                  /-
                                    🎉 no goals
                                  -/
  /-
    case h_real.h_real
    a✝¹ a✝ : Real
    h1 : Or (Ne (↑a✝¹) Bot.bot) (Ne (↑a✝) Top.top)
    h2 : Or (Ne (↑a✝¹) Top.top) (Ne (↑a✝) Bot.bot)
    ⊢ Eq (Neg.neg (HAdd.hAdd ↑a✝¹ ↑a✝)) (HSub.hSub (Neg.neg ↑a✝¹) ↑a✝)
  -/
  rw [← coe_add, ← coe_neg, ← coe_neg, ← coe_sub, neg_add']
  /-
    🎉 no goals
  -/


lemma neg_sub {x y : EReal} (h1 : x ≠ ⊥ ∨ y ≠ ⊥) (h2 : x ≠ ⊤ ∨ y ≠ ⊤) :
    - (x - y) = - x + y := by
  /-
    x y : EReal
    h1 : Or (Ne x Bot.bot) (Ne y Bot.bot)
    h2 : Or (Ne x Top.top) (Ne y Top.top)
    ⊢ Eq (Neg.neg (HSub.hSub x y)) (HAdd.hAdd (Neg.neg x) y)
  -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  rw [sub_eq_add_neg, neg_add _ _, sub_eq_add_neg, neg_neg] <;> simp_all
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem bot_sub (x : EReal) : ⊥ - x = ⊥ :=
  bot_add x


@[simp]
theorem sub_top (x : EReal) : x - ⊤ = ⊥ :=
  add_bot x


@[simp]
theorem top_sub_bot : (⊤ : EReal) - ⊥ = ⊤ :=
  rfl


@[simp]
theorem top_sub_coe (x : ℝ) : (⊤ : EReal) - x = ⊤ :=
  rfl


@[simp]
theorem coe_sub_bot (x : ℝ) : (x : EReal) - ⊥ = ⊤ :=
  rfl


@[simp]
lemma sub_bot {x : EReal} (h : x ≠ ⊥) : x - ⊥ = ⊤ := by
  /-
    x : EReal
    h : Ne x Bot.bot
    ⊢ Eq (HSub.hSub x Bot.bot) Top.top
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases x <;> tauto
              /-
                🎉 no goals
              -/


@[simp]
lemma top_sub {x : EReal} (hx : x ≠ ⊤) : ⊤ - x = ⊤ := by
  /-
    x : EReal
    hx : Ne x Top.top
    ⊢ Eq (HSub.hSub Top.top x) Top.top
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases x <;> tauto
              /-
                🎉 no goals
              -/


@[simp]
lemma sub_self {x : EReal} (h_top : x ≠ ⊤) (h_bot : x ≠ ⊥) : x - x = 0 := by
  /-
    x : EReal
    h_top : Ne x Top.top
    h_bot : Ne x Bot.bot
    ⊢ Eq (HSub.hSub x x) 0
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases x <;> simp_all [← coe_sub]
              /-
                🎉 no goals
              -/


lemma sub_self_le_zero {x : EReal} : x - x ≤ 0 := by
  /-
    x : EReal
    ⊢ LE.le (HSub.hSub x x) 0
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases x <;> simp
              /-
                🎉 no goals
              -/


lemma sub_nonneg {x y : EReal} (h_top : x ≠ ⊤ ∨ y ≠ ⊤) (h_bot : x ≠ ⊥ ∨ y ≠ ⊥) :
    0 ≤ x - y ↔ y ≤ x := by
  /-
    x y : EReal
    h_top : Or (Ne x Top.top) (Ne y Top.top)
    h_bot : Or (Ne x Bot.bot) (Ne y Bot.bot)
    ⊢ Iff (LE.le 0 (HSub.hSub x y)) (LE.le y x)
  -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
  cases x <;> cases y <;> simp_all [← EReal.coe_sub]
                          /-
                            🎉 no goals
                          -/


lemma sub_nonpos {x y : EReal} : x - y ≤ 0 ↔ x ≤ y := by
  /-
    x y : EReal
    ⊢ Iff (LE.le (HSub.hSub x y) 0) (LE.le x y)
  -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
  cases x <;> cases y <;> simp [← EReal.coe_sub]
                          /-
                            🎉 no goals
                          -/


lemma sub_pos {x y : EReal} : 0 < x - y ↔ y < x := by
  /-
    x y : EReal
    ⊢ Iff (LT.lt 0 (HSub.hSub x y)) (LT.lt y x)
  -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
  cases x <;> cases y <;> simp [← EReal.coe_sub]
                          /-
                            🎉 no goals
                          -/


lemma sub_neg {x y : EReal} (h_top : x ≠ ⊤ ∨ y ≠ ⊤) (h_bot : x ≠ ⊥ ∨ y ≠ ⊥) :
    x - y < 0 ↔ x < y := by
  /-
    x y : EReal
    h_top : Or (Ne x Top.top) (Ne y Top.top)
    h_bot : Or (Ne x Bot.bot) (Ne y Bot.bot)
    ⊢ Iff (LT.lt (HSub.hSub x y) 0) (LT.lt x y)
  -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
  cases x <;> cases y <;> simp_all [← EReal.coe_sub]
                          /-
                            🎉 no goals
                          -/


theorem sub_le_sub {x y z t : EReal} (h : x ≤ y) (h' : t ≤ z) : x - z ≤ y - t :=
  add_le_add h (neg_le_neg_iff.2 h')


theorem sub_lt_sub_of_lt_of_le {x y z t : EReal} (h : x < y) (h' : z ≤ t) (hz : z ≠ ⊥)
    (ht : t ≠ ⊤) : x - t < y - z :=
                                                     /-
                                                       x y z t : EReal
                                                       h : LT.lt x y
                                                       h' : LE.le z t
                                                       hz : Ne z Bot.bot
                                                       ht : Ne t Top.top
                                                       ⊢ Ne (Neg.neg t) Bot.bot
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  add_lt_add_of_lt_of_le h (neg_le_neg_iff.2 h') (by simp [ht]) (by simp [hz])
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem coe_real_ereal_eq_coe_toNNReal_sub_coe_toNNReal (x : ℝ) :
    (x : EReal) = Real.toNNReal x - Real.toNNReal (-x) := by
  /-
    x : Real
    ⊢ Eq (↑x) (HSub.hSub ↑↑x.toNNReal ↑↑(Neg.neg x).toNNReal)
  -/
  rcases le_total 0 x with (h | h)
    /-
      case inl
      x : Real
      h : LE.le 0 x
      ⊢ Eq (↑x) (HSub.hSub ↑↑x.toNNReal ↑↑(Neg.neg x).toNNReal)
    -/
  · lift x to ℝ≥0 using h
    rw [Real.toNNReal_of_nonpos (neg_nonpos.mpr x.coe_nonneg), Real.toNNReal_coe, ENNReal.coe_zero,
      coe_ennreal_zero, sub_zero]
    /-
      case inl.intro
      x : NNReal
      ⊢ Eq ↑↑x ↑↑x
    -/
    rfl
    /-
      🎉 no goals
    -/
  · rw [Real.toNNReal_of_nonpos h, ENNReal.coe_zero, coe_ennreal_zero, coe_nnreal_eq_coe_real,
      Real.coe_toNNReal, zero_sub, coe_neg, neg_neg]
    /-
      case inr.hr
      x : Real
      h : LE.le x 0
      ⊢ LE.le 0 (Neg.neg x)
    -/
    exact neg_nonneg.2 h
    /-
      🎉 no goals
    -/


theorem toReal_sub {x y : EReal} (hx : x ≠ ⊤) (h'x : x ≠ ⊥) (hy : y ≠ ⊤) (h'y : y ≠ ⊥) :
    toReal (x - y) = toReal x - toReal y := by
  /-
    x y : EReal
    hx : Ne x Top.top
    h'x : Ne x Bot.bot
    hy : Ne y Top.top
    h'y : Ne y Bot.bot
    ⊢ Eq (HSub.hSub x y).toReal (HSub.hSub x.toReal y.toReal)
  -/
  lift x to ℝ using ⟨hx, h'x⟩
  /-
    case intro
    y : EReal
    hy : Ne y Top.top
    h'y : Ne y Bot.bot
    x : Real
    hx : Ne (↑x) Top.top
    h'x : Ne (↑x) Bot.bot
    ⊢ Eq (HSub.hSub (↑x) y).toReal (HSub.hSub (↑x).toReal y.toReal)
  -/
  lift y to ℝ using ⟨hy, h'y⟩
  /-
    case intro.intro
    x : Real
    hx : Ne (↑x) Top.top
    h'x : Ne (↑x) Bot.bot
    y : Real
    hy : Ne (↑y) Top.top
    h'y : Ne (↑y) Bot.bot
    ⊢ Eq (HSub.hSub ↑x ↑y).toReal (HSub.hSub (↑x).toReal (↑y).toReal)
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma add_sub_cancel_right {a : EReal} {b : Real} : a + b - b = a := by
  /-
    a : EReal
    b : Real
    ⊢ Eq (HSub.hSub (HAdd.hAdd a ↑b) ↑b) a
  -/
              /-
                🎉 no goals
              -/
  cases a <;> norm_cast
              /-
                🎉 no goals
              -/
  /-
    case h_real
    b a✝ : Real
    ⊢ Eq (HSub.hSub (HAdd.hAdd a✝ b) b) a✝
  -/
  exact _root_.add_sub_cancel_right _ _
  /-
    🎉 no goals
  -/


lemma add_sub_cancel_left {a : EReal} {b : Real} : b + a - b = a := by
  /-
    a : EReal
    b : Real
    ⊢ Eq (HSub.hSub (HAdd.hAdd (↑b) a) ↑b) a
  -/
  rw [add_comm, EReal.add_sub_cancel_right]
  /-
    🎉 no goals
  -/


lemma sub_add_cancel {a : EReal} {b : Real} : a - b + b = a := by
  /-
    a : EReal
    b : Real
    ⊢ Eq (HAdd.hAdd (HSub.hSub a ↑b) ↑b) a
  -/
  rw [add_comm, ← add_sub_assoc, add_sub_cancel_left]
  /-
    🎉 no goals
  -/


lemma sub_add_cancel_right {a : EReal} {b : Real} : b - (a + b) = -a := by
  /-
    a : EReal
    b : Real
    ⊢ Eq (HSub.hSub (↑b) (HAdd.hAdd a ↑b)) (Neg.neg a)
  -/
              /-
                🎉 no goals
              -/
  cases a <;> norm_cast
              /-
                🎉 no goals
              -/
  /-
    case h_real
    b a✝ : Real
    ⊢ Eq (HSub.hSub b (HAdd.hAdd a✝ b)) (Neg.neg a✝)
  -/
  exact _root_.sub_add_cancel_right _ _
  /-
    🎉 no goals
  -/


lemma sub_add_cancel_left {a : EReal} {b : Real} : b - (b + a) = -a := by
  /-
    a : EReal
    b : Real
    ⊢ Eq (HSub.hSub (↑b) (HAdd.hAdd (↑b) a)) (Neg.neg a)
  -/
  rw [add_comm, sub_add_cancel_right]
  /-
    🎉 no goals
  -/


lemma le_sub_iff_add_le {a b c : EReal} (hb : b ≠ ⊥ ∨ c ≠ ⊥) (ht : b ≠ ⊤ ∨ c ≠ ⊤) :
    a ≤ c - b ↔ a + b ≤ c := by
  induction b with
  | h_bot =>
    simp only [ne_eq, not_true_eq_false, false_or] at hb
    simp only [sub_bot hb, le_top, add_bot, bot_le]
  | h_real b =>
    rw [← (addLECancellable_coe b).add_le_add_iff_right, sub_add_cancel]
  | h_top =>
    simp only [ne_eq, not_true_eq_false, false_or, sub_top, le_bot_iff] at ht ⊢
    refine ⟨fun h ↦ h ▸ (bot_add ⊤).symm ▸ bot_le, fun h ↦ ?_⟩
    by_contra ha
    exact (h.trans_lt (Ne.lt_top ht)).ne (add_top_iff_ne_bot.2 ha)


lemma sub_le_iff_le_add {a b c : EReal} (h₁ : b ≠ ⊥ ∨ c ≠ ⊤) (h₂ : b ≠ ⊤ ∨ c ≠ ⊥) :
    a - b ≤ c ↔ a ≤ c + b := by
  /-
    a b c : EReal
    h₁ : Or (Ne b Bot.bot) (Ne c Top.top)
    h₂ : Or (Ne b Top.top) (Ne c Bot.bot)
    ⊢ Iff (LE.le (HSub.hSub a b) c) (LE.le a (HAdd.hAdd c b))
  -/
  suffices a + (-b) ≤ c ↔ a ≤ c - (-b) by simpa [sub_eq_add_neg]
  /-
    a b c : EReal
    h₁ : Or (Ne b Bot.bot) (Ne c Top.top)
    h₂ : Or (Ne b Top.top) (Ne c Bot.bot)
    ⊢ Iff (LE.le (HAdd.hAdd a (Neg.neg b)) c) (LE.le a (HSub.hSub c (Neg.neg b)))
  -/
                                            /-
                                              🎉 no goals
                                            -/
  refine (le_sub_iff_add_le ?_ ?_).symm <;> simpa
                                            /-
                                              🎉 no goals
                                            -/


protected theorem lt_sub_iff_add_lt {a b c : EReal} (h₁ : b ≠ ⊥ ∨ c ≠ ⊤) (h₂ : b ≠ ⊤ ∨ c ≠ ⊥) :
    c < a - b ↔ c + b < a :=
  lt_iff_lt_of_le_iff_le (sub_le_iff_le_add h₁ h₂)


theorem sub_le_of_le_add {a b c : EReal} (h : a ≤ b + c) : a - c ≤ b := by
  induction c with
  | h_bot => rw [add_bot, le_bot_iff] at h; simp only [h, bot_sub, bot_le]
  | h_real c => exact (sub_le_iff_le_add (.inl (coe_ne_bot c)) (.inl (coe_ne_top c))).2 h
  | h_top => simp only [sub_top, bot_le]


/-- See also `EReal.sub_le_of_le_add`.-/
theorem sub_le_of_le_add' {a b c : EReal} (h : a ≤ b + c) : a - b ≤ c :=
  sub_le_of_le_add (add_comm b c ▸ h)


lemma add_le_of_le_sub {a b c : EReal} (h : a ≤ b - c) : a + c ≤ b := by
  /-
    a b c : EReal
    h : LE.le a (HSub.hSub b c)
    ⊢ LE.le (HAdd.hAdd a c) b
  -/
  rw [← neg_neg c]
  /-
    a b c : EReal
    h : LE.le a (HSub.hSub b c)
    ⊢ LE.le (HAdd.hAdd a (Neg.neg (Neg.neg c))) b
  -/
  exact sub_le_of_le_add h
  /-
    🎉 no goals
  -/


lemma sub_lt_iff {a b c : EReal} (h₁ : b ≠ ⊥ ∨ c ≠ ⊥) (h₂ : b ≠ ⊤ ∨ c ≠ ⊤) :
    c - b < a ↔ c < a + b :=
  lt_iff_lt_of_le_iff_le (le_sub_iff_add_le h₁ h₂)


lemma add_lt_of_lt_sub {a b c : EReal} (h : a < b - c) : a + c < b := by
  /-
    a b c : EReal
    h : LT.lt a (HSub.hSub b c)
    ⊢ LT.lt (HAdd.hAdd a c) b
  -/
  contrapose! h
  /-
    a b c : EReal
    h : LE.le b (HAdd.hAdd a c)
    ⊢ LE.le (HSub.hSub b c) a
  -/
  exact sub_le_of_le_add h
  /-
    🎉 no goals
  -/


lemma sub_lt_of_lt_add {a b c : EReal} (h : a < b + c) : a - c < b :=
                         /-
                           a b c : EReal
                           h : LT.lt a (HAdd.hAdd b c)
                           ⊢ LT.lt a (HSub.hSub b (Neg.neg c))
                         -/
  add_lt_of_lt_sub <| by rwa [sub_eq_add_neg, neg_neg]
                         /-
                           🎉 no goals
                         -/


/-- See also `EReal.sub_lt_of_lt_add`.-/
lemma sub_lt_of_lt_add' {a b c : EReal} (h : a < b + c) : a - b < c :=
                         /-
                           a b c : EReal
                           h : LT.lt a (HAdd.hAdd b c)
                           ⊢ LT.lt a (HAdd.hAdd c b)
                         -/
  sub_lt_of_lt_add <| by rwa [add_comm]
                         /-
                           🎉 no goals
                         -/


lemma le_of_forall_lt_iff_le {x y : EReal} : (∀ z : ℝ, x < z → y ≤ z) ↔ y ≤ x := by
  /-
    x y : EReal
    ⊢ Iff (∀ (z : Real), LT.lt x ↑z → LE.le y ↑z) (LE.le y x)
  -/
  refine ⟨fun h ↦ WithBot.le_of_forall_lt_iff_le.1 ?_, fun h _ x_z ↦ h.trans x_z.le⟩
  /-
    x y : EReal
    h : ∀ (z : Real), LT.lt x ↑z → LE.le y ↑z
    ⊢ ∀ (z : WithTop Real), LT.lt x ↑z → LE.le y ↑z
  -/
  rw [WithTop.forall]
  /-
    x y : EReal
    h : ∀ (z : Real), LT.lt x ↑z → LE.le y ↑z
    ⊢ And (LT.lt x ↑Top.top → LE.le y ↑Top.top) (∀ (x_1 : Real), LT.lt x ↑↑x_1 → L …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma ge_of_forall_gt_iff_ge {x y : EReal} : (∀ z : ℝ, z < y → z ≤ x) ↔ y ≤ x := by
  /-
    x y : EReal
    ⊢ Iff (∀ (z : Real), LT.lt (↑z) y → LE.le (↑z) x) (LE.le y x)
  -/
  refine ⟨fun h ↦ WithBot.ge_of_forall_gt_iff_ge.1 ?_, fun h _ x_z ↦ x_z.le.trans h⟩
  /-
    x y : EReal
    h : ∀ (z : Real), LT.lt (↑z) y → LE.le (↑z) x
    ⊢ ∀ (z : WithTop Real), LT.lt (↑z) y → LE.le (↑z) x
  -/
  rw [WithTop.forall]
  /-
    x y : EReal
    h : ∀ (z : Real), LT.lt (↑z) y → LE.le (↑z) x
    ⊢ And (LT.lt (↑Top.top) y → LE.le (↑Top.top) x) (∀ (x_1 : Real), LT.lt (↑↑x_1) …
  -/
  aesop
  /-
    🎉 no goals
  -/


private lemma exists_lt_add_left {a b c : EReal} (hc : c < a + b) : ∃ a' < a, c < a' + b := by
  /-
    a b c : EReal
    hc : LT.lt c (HAdd.hAdd a b)
    ⊢ Exists fun a' => And (LT.lt a' a) (LT.lt c (HAdd.hAdd a' b))
  -/
  obtain ⟨a', hc', ha'⟩ := exists_between (sub_lt_of_lt_add hc)
  /-
    case intro.intro
    a b c : EReal
    hc : LT.lt c (HAdd.hAdd a b)
    a' : EReal
    hc' : LT.lt (HSub.hSub c b) a'
    ha' : LT.lt a' a
    ⊢ Exists fun a' => And (LT.lt a' a) (LT.lt c (HAdd.hAdd a' b))
  -/
  refine ⟨a', ha', (sub_lt_iff (.inl ?_) (.inr hc.ne_top)).1 hc'⟩
  /-
    case intro.intro
    a b c : EReal
    hc : LT.lt c (HAdd.hAdd a b)
    a' : EReal
    hc' : LT.lt (HSub.hSub c b) a'
    ha' : LT.lt a' a
    ⊢ Ne b Bot.bot
  -/
  contrapose! hc
  /-
    case intro.intro
    a b c a' : EReal
    hc' : LT.lt (HSub.hSub c b) a'
    ha' : LT.lt a' a
    hc : Eq b Bot.bot
    ⊢ LE.le (HAdd.hAdd a b) c
  -/
  exact hc ▸ (add_bot a).symm ▸ bot_le
  /-
    🎉 no goals
  -/


private lemma exists_lt_add_right {a b c : EReal} (hc : c < a + b) : ∃ b' < b, c < a + b' := by
  /-
    a b c : EReal
    hc : LT.lt c (HAdd.hAdd a b)
    ⊢ Exists fun b' => And (LT.lt b' b) (LT.lt c (HAdd.hAdd a b'))
  -/
  simp_rw [add_comm a] at hc ⊢; exact exists_lt_add_left hc
                                /-
                                  🎉 no goals
                                -/


lemma add_le_of_forall_lt {a b c : EReal} (h : ∀ a' < a, ∀ b' < b, a' + b' ≤ c) : a + b ≤ c := by
  /-
    a b c : EReal
    h : ∀ (a' : EReal), LT.lt a' a → ∀ (b' : EReal), LT.lt b' b → LE.le (HAdd.hAdd …
    ⊢ LE.le (HAdd.hAdd a b) c
  -/
  refine le_of_forall_ge_of_dense fun d hd ↦ ?_
  /-
    a b c : EReal
    h : ∀ (a' : EReal), LT.lt a' a → ∀ (b' : EReal), LT.lt b' b → LE.le (HAdd.hAdd …
    d : EReal
    hd : LT.lt d (HAdd.hAdd a b)
    ⊢ LE.le d c
  -/
  obtain ⟨a', ha', hd⟩ := exists_lt_add_left hd
  /-
    case intro.intro
    a b c : EReal
    h : ∀ (a' : EReal), LT.lt a' a → ∀ (b' : EReal), LT.lt b' b → LE.le (HAdd.hAdd …
    d : EReal
    hd✝ : LT.lt d (HAdd.hAdd a b)
    a' : EReal
    ha' : LT.lt a' a
    hd : LT.lt d (HAdd.hAdd a' b)
    ⊢ LE.le d c
  -/
  obtain ⟨b', hb', hd⟩ := exists_lt_add_right hd
  /-
    case intro.intro.intro.intro
    a b c : EReal
    h : ∀ (a' : EReal), LT.lt a' a → ∀ (b' : EReal), LT.lt b' b → LE.le (HAdd.hAdd …
    d : EReal
    hd✝¹ : LT.lt d (HAdd.hAdd a b)
    a' : EReal
    ha' : LT.lt a' a
    hd✝ : LT.lt d (HAdd.hAdd a' b)
    b' : EReal
    hb' : LT.lt b' b
    hd : LT.lt d (HAdd.hAdd a' b')
    ⊢ LE.le d c
  -/
  exact hd.le.trans (h _ ha' _ hb')
  /-
    🎉 no goals
  -/


lemma le_add_of_forall_gt {a b c : EReal} (h₁ : a ≠ ⊥ ∨ b ≠ ⊤) (h₂ : a ≠ ⊤ ∨ b ≠ ⊥)
    (h : ∀ a' > a, ∀ b' > b, c ≤ a' + b') : c ≤ a + b := by
  /-
    a b c : EReal
    h₁ : Or (Ne a Bot.bot) (Ne b Top.top)
    h₂ : Or (Ne a Top.top) (Ne b Bot.bot)
    h : ∀ (a' : EReal), GT.gt a' a → ∀ (b' : EReal), GT.gt b' b → LE.le c (HAdd.hA …
    ⊢ LE.le c (HAdd.hAdd a b)
  -/
  rw [← neg_le_neg_iff, neg_add h₁ h₂]
  /-
    a b c : EReal
    h₁ : Or (Ne a Bot.bot) (Ne b Top.top)
    h₂ : Or (Ne a Top.top) (Ne b Bot.bot)
    h : ∀ (a' : EReal), GT.gt a' a → ∀ (b' : EReal), GT.gt b' b → LE.le c (HAdd.hA …
    ⊢ LE.le (HSub.hSub (Neg.neg a) b) (Neg.neg c)
  -/
  refine add_le_of_forall_lt fun a' ha' b' hb' ↦ EReal.le_neg_of_le_neg ?_
  /-
    a b c : EReal
    h₁ : Or (Ne a Bot.bot) (Ne b Top.top)
    h₂ : Or (Ne a Top.top) (Ne b Bot.bot)
    h : ∀ (a' : EReal), GT.gt a' a → ∀ (b' : EReal), GT.gt b' b → LE.le c (HAdd.hA …
    a' : EReal
    ha' : LT.lt a' (Neg.neg a)
    b' : EReal
    hb' : LT.lt b' (Neg.neg b)
    ⊢ LE.le c (Neg.neg (HAdd.hAdd a' b'))
  -/
  rw [neg_add (.inr hb'.ne_top) (.inl ha'.ne_top)]
  /-
    a b c : EReal
    h₁ : Or (Ne a Bot.bot) (Ne b Top.top)
    h₂ : Or (Ne a Top.top) (Ne b Bot.bot)
    h : ∀ (a' : EReal), GT.gt a' a → ∀ (b' : EReal), GT.gt b' b → LE.le c (HAdd.hA …
    a' : EReal
    ha' : LT.lt a' (Neg.neg a)
    b' : EReal
    hb' : LT.lt b' (Neg.neg b)
    ⊢ LE.le c (HSub.hSub (Neg.neg a') b')
  -/
  exact h _ (EReal.lt_neg_of_lt_neg ha') _ (EReal.lt_neg_of_lt_neg hb')
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-19")] alias top_add_le_of_forall_add_le := add_le_of_forall_lt

@[deprecated (since := "2024-11-19")] alias add_le_of_forall_add_le := add_le_of_forall_lt

@[deprecated (since := "2024-11-19")] alias le_add_of_forall_le_add := le_add_of_forall_gt


lemma _root_.ENNReal.toEReal_sub {x y : ℝ≥0∞} (hy_top : y ≠ ∞) (h_le : y ≤ x) :
    (x - y).toEReal = x.toEReal - y.toEReal := by
  /-
    x y : ENNReal
    hy_top : Ne y Top.top
    h_le : LE.le y x
    ⊢ Eq (↑(HSub.hSub x y)) (HSub.hSub ↑x ↑y)
  -/
  lift y to ℝ≥0 using hy_top
  cases x with
  | top => simp [coe_nnreal_eq_coe_real]
  | coe x =>
    simp only [coe_nnreal_eq_coe_real, ← ENNReal.coe_sub, NNReal.coe_sub (mod_cast h_le), coe_sub]


@[simp] lemma top_mul_top : (⊤ : EReal) * ⊤ = ⊤ := rfl


@[simp] lemma top_mul_bot : (⊤ : EReal) * ⊥ = ⊥ := rfl


@[simp] lemma bot_mul_top : (⊥ : EReal) * ⊤ = ⊥ := rfl


@[simp] lemma bot_mul_bot : (⊥ : EReal) * ⊥ = ⊤ := rfl


lemma coe_mul_top_of_pos {x : ℝ} (h : 0 < x) : (x : EReal) * ⊤ = ⊤ :=
  if_pos h


lemma coe_mul_top_of_neg {x : ℝ} (h : x < 0) : (x : EReal) * ⊤ = ⊥ :=
  (if_neg h.not_lt).trans (if_neg h.ne)


lemma top_mul_coe_of_pos {x : ℝ} (h : 0 < x) : (⊤ : EReal) * x = ⊤ :=
  if_pos h


lemma top_mul_coe_of_neg {x : ℝ} (h : x < 0) : (⊤ : EReal) * x = ⊥ :=
  (if_neg h.not_lt).trans (if_neg h.ne)


lemma mul_top_of_pos : ∀ {x : EReal}, 0 < x → x * ⊤ = ⊤
  | ⊥, h => absurd h not_lt_bot
  | (x : ℝ), h => coe_mul_top_of_pos (EReal.coe_pos.1 h)
  | ⊤, _ => rfl


lemma mul_top_of_neg : ∀ {x : EReal}, x < 0 → x * ⊤ = ⊥
  | ⊥, _ => rfl
  | (x : ℝ), h => coe_mul_top_of_neg (EReal.coe_neg'.1 h)
  | ⊤, h => absurd h not_top_lt


lemma top_mul_of_pos {x : EReal} (h : 0 < x) : ⊤ * x = ⊤ := by
  /-
    x : EReal
    h : LT.lt 0 x
    ⊢ Eq (HMul.hMul Top.top x) Top.top
  -/
  rw [EReal.mul_comm]
  /-
    x : EReal
    h : LT.lt 0 x
    ⊢ Eq (HMul.hMul x Top.top) Top.top
  -/
  exact mul_top_of_pos h
  /-
    🎉 no goals
  -/


lemma top_mul_of_neg {x : EReal} (h : x < 0) : ⊤ * x = ⊥ := by
  /-
    x : EReal
    h : LT.lt x 0
    ⊢ Eq (HMul.hMul Top.top x) Bot.bot
  -/
  rw [EReal.mul_comm]
  /-
    x : EReal
    h : LT.lt x 0
    ⊢ Eq (HMul.hMul x Top.top) Bot.bot
  -/
  exact mul_top_of_neg h
  /-
    🎉 no goals
  -/


lemma top_mul_coe_ennreal {x : ℝ≥0∞} (hx : x ≠ 0) : ⊤ * (x : EReal) = ⊤ :=
  top_mul_of_pos <| coe_ennreal_pos.mpr <| pos_iff_ne_zero.mpr hx


lemma coe_ennreal_mul_top {x : ℝ≥0∞} (hx : x ≠ 0) : (x : EReal) * ⊤ = ⊤ := by
  /-
    x : ENNReal
    hx : Ne x 0
    ⊢ Eq (HMul.hMul (↑x) Top.top) Top.top
  -/
  rw [EReal.mul_comm, top_mul_coe_ennreal hx]
  /-
    🎉 no goals
  -/


lemma coe_mul_bot_of_pos {x : ℝ} (h : 0 < x) : (x : EReal) * ⊥ = ⊥ :=
  if_pos h


lemma coe_mul_bot_of_neg {x : ℝ} (h : x < 0) : (x : EReal) * ⊥ = ⊤ :=
  (if_neg h.not_lt).trans (if_neg h.ne)


lemma bot_mul_coe_of_pos {x : ℝ} (h : 0 < x) : (⊥ : EReal) * x = ⊥ :=
  if_pos h


lemma bot_mul_coe_of_neg {x : ℝ} (h : x < 0) : (⊥ : EReal) * x = ⊤ :=
  (if_neg h.not_lt).trans (if_neg h.ne)


lemma mul_bot_of_pos : ∀ {x : EReal}, 0 < x → x * ⊥ = ⊥
  | ⊥, h => absurd h not_lt_bot
  | (x : ℝ), h => coe_mul_bot_of_pos (EReal.coe_pos.1 h)
  | ⊤, _ => rfl


lemma mul_bot_of_neg : ∀ {x : EReal}, x < 0 → x * ⊥ = ⊤
  | ⊥, _ => rfl
  | (x : ℝ), h => coe_mul_bot_of_neg (EReal.coe_neg'.1 h)
  | ⊤, h => absurd h not_top_lt


lemma bot_mul_of_pos {x : EReal} (h : 0 < x) : ⊥ * x = ⊥ := by
  /-
    x : EReal
    h : LT.lt 0 x
    ⊢ Eq (HMul.hMul Bot.bot x) Bot.bot
  -/
  rw [EReal.mul_comm]
  /-
    x : EReal
    h : LT.lt 0 x
    ⊢ Eq (HMul.hMul x Bot.bot) Bot.bot
  -/
  exact mul_bot_of_pos h
  /-
    🎉 no goals
  -/


lemma bot_mul_of_neg {x : EReal} (h : x < 0) : ⊥ * x = ⊤ := by
  /-
    x : EReal
    h : LT.lt x 0
    ⊢ Eq (HMul.hMul Bot.bot x) Top.top
  -/
  rw [EReal.mul_comm]
  /-
    x : EReal
    h : LT.lt x 0
    ⊢ Eq (HMul.hMul x Bot.bot) Top.top
  -/
  exact mul_bot_of_neg h
  /-
    🎉 no goals
  -/


lemma toReal_mul {x y : EReal} : toReal (x * y) = toReal x * toReal y := by
  induction x, y using induction₂_symm with
  | top_zero | zero_bot | top_top | top_bot | bot_bot => simp
  | symm h => rwa [mul_comm, EReal.mul_comm]
  | coe_coe => norm_cast
  | top_pos _ h => simp [top_mul_coe_of_pos h]
  | top_neg _ h => simp [top_mul_coe_of_neg h]
  | pos_bot _ h => simp [coe_mul_bot_of_pos h]
  | neg_bot _ h => simp [coe_mul_bot_of_neg h]


instance : NoZeroDivisors EReal where
  eq_zero_or_eq_zero_of_mul_eq_zero := by
    /-
      ⊢ ∀ {a b : EReal}, Eq (HMul.hMul a b) 0 → Or (Eq a 0) (Eq b 0)
    -/
    intro a b h
    /-
      a b : EReal
      h : Eq (HMul.hMul a b) 0
      ⊢ Or (Eq a 0) (Eq b 0)
    -/
    contrapose! h
    /-
      a b : EReal
      h : And (Ne a 0) (Ne b 0)
      ⊢ Ne (HMul.hMul a b) 0
    -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
    cases a <;> cases b <;> try {· simp_all [← EReal.coe_mul]}
                            /-
                              🎉 no goals
                            -/
      /-
        case h_bot.h_real
        a✝ : Real
        h : And (Ne Bot.bot 0) (Ne (↑a✝) 0)
        ⊢ Ne (HMul.hMul Bot.bot ↑a✝) 0
      -/
    · rcases lt_or_gt_of_ne h.2 with (h | h)
            /-
              case h_bot.h_real.inl
              a✝ : Real
              h✝ : And (Ne Bot.bot 0) (Ne (↑a✝) 0)
              h : LT.lt (↑a✝) 0
              ⊢ Ne (HMul.hMul Bot.bot ↑a✝) 0
            -/
            /-
              🎉 no goals
            -/
        <;> simp [EReal.bot_mul_of_neg, EReal.bot_mul_of_pos, h]
            /-
              🎉 no goals
            -/
      /-
        case h_real.h_bot
        a✝ : Real
        h : And (Ne (↑a✝) 0) (Ne Bot.bot 0)
        ⊢ Ne (HMul.hMul (↑a✝) Bot.bot) 0
      -/
    · rcases lt_or_gt_of_ne h.1 with (h | h)
            /-
              case h_real.h_bot.inl
              a✝ : Real
              h✝ : And (Ne (↑a✝) 0) (Ne Bot.bot 0)
              h : LT.lt (↑a✝) 0
              ⊢ Ne (HMul.hMul (↑a✝) Bot.bot) 0
            -/
            /-
              🎉 no goals
            -/
        <;> simp [EReal.mul_bot_of_pos, EReal.mul_bot_of_neg, h]
            /-
              🎉 no goals
            -/
      /-
        case h_real.h_top
        a✝ : Real
        h : And (Ne (↑a✝) 0) (Ne Top.top 0)
        ⊢ Ne (HMul.hMul (↑a✝) Top.top) 0
      -/
    · rcases lt_or_gt_of_ne h.1 with (h | h)
            /-
              case h_real.h_top.inl
              a✝ : Real
              h✝ : And (Ne (↑a✝) 0) (Ne Top.top 0)
              h : LT.lt (↑a✝) 0
              ⊢ Ne (HMul.hMul (↑a✝) Top.top) 0
            -/
            /-
              🎉 no goals
            -/
        <;> simp [EReal.mul_top_of_neg, EReal.mul_top_of_pos, h]
            /-
              🎉 no goals
            -/
      /-
        case h_top.h_real
        a✝ : Real
        h : And (Ne Top.top 0) (Ne (↑a✝) 0)
        ⊢ Ne (HMul.hMul Top.top ↑a✝) 0
      -/
    · rcases lt_or_gt_of_ne h.2 with (h | h)
            /-
              case h_top.h_real.inl
              a✝ : Real
              h✝ : And (Ne Top.top 0) (Ne (↑a✝) 0)
              h : LT.lt (↑a✝) 0
              ⊢ Ne (HMul.hMul Top.top ↑a✝) 0
            -/
            /-
              🎉 no goals
            -/
        <;> simp [EReal.top_mul_of_pos, EReal.top_mul_of_neg, h]
            /-
              🎉 no goals
            -/


lemma mul_pos_iff {a b : EReal} : 0 < a * b ↔ 0 < a ∧ 0 < b ∨ a < 0 ∧ b < 0 := by
  induction a, b using EReal.induction₂_symm with
  | symm h => simp [EReal.mul_comm, h, and_comm]
  | top_top => simp
  | top_pos _ hx => simp [EReal.top_mul_coe_of_pos hx, hx]
  | top_zero => simp
  | top_neg _ hx => simp [hx, EReal.top_mul_coe_of_neg hx, le_of_lt]
  | top_bot => simp
  | pos_bot _ hx => simp [hx, EReal.coe_mul_bot_of_pos hx, le_of_lt]
  | coe_coe x y => simp [← coe_mul, _root_.mul_pos_iff]
  | zero_bot => simp
  | neg_bot _ hx => simp [hx, EReal.coe_mul_bot_of_neg hx]
  | bot_bot => simp


lemma mul_nonneg_iff {a b : EReal} : 0 ≤ a * b ↔ 0 ≤ a ∧ 0 ≤ b ∨ a ≤ 0 ∧ b ≤ 0 := by
  /-
    a b : EReal
    ⊢ Iff (LE.le 0 (HMul.hMul a b)) (Or (And (LE.le 0 a) (LE.le 0 b)) (And (LE.le  …
  -/
  simp_rw [le_iff_lt_or_eq, mul_pos_iff, zero_eq_mul (a := a)]
  /-
    a b : EReal
    ⊢ Iff (Or (Or (And (LT.lt 0 a) (LT.lt 0 b)) (And (LT.lt a 0) (LT.lt b 0))) (Or …
  -/
  rcases lt_trichotomy a 0 with (h | h | h) <;> rcases lt_trichotomy b 0 with (h' | h' | h')
        /-
          case inl.inl
          a b : EReal
          h : LT.lt a 0
          h' : LT.lt b 0
          ⊢ Iff (Or (Or (And (LT.lt 0 a) (LT.lt 0 b)) (And (LT.lt a 0) (LT.lt b 0))) (Or …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    <;> simp only [h, h', true_or, true_and, or_true, and_true] <;> tauto
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- The product of two positive extended real numbers is positive. -/
lemma mul_pos {a b : EReal} (ha : 0 < a) (hb : 0 < b) : 0 < a * b :=
  mul_pos_iff.mpr (Or.inl ⟨ha, hb⟩)


/-- Induct on two ereals by performing case splits on the sign of one whenever the other is
infinite. This version eliminates some cases by assuming that `P x y` implies `P (-x) y` for all
`x`, `y`. -/
@[elab_as_elim]
lemma induction₂_neg_left {P : EReal → EReal → Prop} (neg_left : ∀ {x y}, P x y → P (-x) y)
    (top_top : P ⊤ ⊤) (top_pos : ∀ x : ℝ, 0 < x → P ⊤ x)
    (top_zero : P ⊤ 0) (top_neg : ∀ x : ℝ, x < 0 → P ⊤ x) (top_bot : P ⊤ ⊥)
    (zero_top : P 0 ⊤) (zero_bot : P 0 ⊥)
    (pos_top : ∀ x : ℝ, 0 < x → P x ⊤) (pos_bot : ∀ x : ℝ, 0 < x → P x ⊥)
    (coe_coe : ∀ x y : ℝ, P x y) : ∀ x y, P x y :=
  have : ∀ y, (∀ x : ℝ, 0 < x → P x y) → ∀ x : ℝ, x < 0 → P x y := fun _ h x hx =>
    neg_neg (x : EReal) ▸ neg_left <| h _ (neg_pos_of_neg hx)
  @induction₂ P top_top top_pos top_zero top_neg top_bot pos_top pos_bot zero_top
    coe_coe zero_bot (this _ pos_top) (this _ pos_bot) (neg_left top_top)
    (fun x hx => neg_left <| top_pos x hx) (neg_left top_zero)
    (fun x hx => neg_left <| top_neg x hx) (neg_left top_bot)


/-- Induct on two ereals by performing case splits on the sign of one whenever the other is
infinite. This version eliminates some cases by assuming that `P` is symmetric and `P x y` implies
`P (-x) y` for all `x`, `y`. -/
@[elab_as_elim]
lemma induction₂_symm_neg {P : EReal → EReal → Prop}
    (symm : ∀ {x y}, P x y → P y x)
    (neg_left : ∀ {x y}, P x y → P (-x) y) (top_top : P ⊤ ⊤)
    (top_pos : ∀ x : ℝ, 0 < x → P ⊤ x) (top_zero : P ⊤ 0) (coe_coe : ∀ x y : ℝ, P x y) :
    ∀ x y, P x y :=
  have neg_right : ∀ {x y}, P x y → P x (-y) := fun h => symm <| neg_left <| symm h
  have : ∀ x, (∀ y : ℝ, 0 < y → P x y) → ∀ y : ℝ, y < 0 → P x y := fun _ h y hy =>
    neg_neg (y : EReal) ▸ neg_right (h _ (neg_pos_of_neg hy))
  @induction₂_neg_left P neg_left top_top top_pos top_zero (this _ top_pos) (neg_right top_top)
    (symm top_zero) (symm <| neg_left top_zero) (fun x hx => symm <| top_pos x hx)
    (fun x hx => symm <| neg_left <| top_pos x hx) coe_coe


protected lemma neg_mul (x y : EReal) : -x * y = -(x * y) := by
  induction x, y using induction₂_neg_left with
  | top_zero | zero_top | zero_bot => simp only [zero_mul, mul_zero, neg_zero]
  | top_top | top_bot => rfl
  | neg_left h => rw [h, neg_neg, neg_neg]
  | coe_coe => norm_cast; exact neg_mul _ _
  | top_pos _ h => rw [top_mul_coe_of_pos h, neg_top, bot_mul_coe_of_pos h]
  | pos_top _ h => rw [coe_mul_top_of_pos h, neg_top, ← coe_neg,
    coe_mul_top_of_neg (neg_neg_of_pos h)]
  | top_neg _ h => rw [top_mul_coe_of_neg h, neg_top, bot_mul_coe_of_neg h, neg_bot]
  | pos_bot _ h => rw [coe_mul_bot_of_pos h, neg_bot, ← coe_neg,
    coe_mul_bot_of_neg (neg_neg_of_pos h)]


instance : HasDistribNeg EReal where
  neg_mul := EReal.neg_mul
  mul_neg := fun x y => by
    /-
      x y : EReal
      ⊢ Eq (HMul.hMul x (Neg.neg y)) (Neg.neg (HMul.hMul x y))
    -/
    rw [x.mul_comm, x.mul_comm]
    /-
      x y : EReal
      ⊢ Eq (HMul.hMul (Neg.neg y) x) (Neg.neg (HMul.hMul y x))
    -/
    exact y.neg_mul x
    /-
      🎉 no goals
    -/


lemma mul_neg_iff {a b : EReal} : a * b < 0 ↔ 0 < a ∧ b < 0 ∨ a < 0 ∧ 0 < b := by
  /-
    a b : EReal
    ⊢ Iff (LT.lt (HMul.hMul a b) 0) (Or (And (LT.lt 0 a) (LT.lt b 0)) (And (LT.lt  …
  -/
  nth_rw 1 [← neg_zero]
  /-
    a b : EReal
    ⊢ Iff (LT.lt (HMul.hMul a b) (-0)) (Or (And (LT.lt 0 a) (LT.lt b 0)) (And (LT. …
  -/
  rw [lt_neg_comm, ← mul_neg a, mul_pos_iff, neg_lt_comm, lt_neg_comm, neg_zero]
  /-
    🎉 no goals
  -/


lemma mul_nonpos_iff {a b : EReal} : a * b ≤ 0 ↔ 0 ≤ a ∧ b ≤ 0 ∨ a ≤ 0 ∧ 0 ≤ b := by
  /-
    a b : EReal
    ⊢ Iff (LE.le (HMul.hMul a b) 0) (Or (And (LE.le 0 a) (LE.le b 0)) (And (LE.le  …
  -/
  nth_rw 1 [← neg_zero]
  /-
    a b : EReal
    ⊢ Iff (LE.le (HMul.hMul a b) (-0)) (Or (And (LE.le 0 a) (LE.le b 0)) (And (LE. …
  -/
  rw [EReal.le_neg, ← mul_neg, mul_nonneg_iff, EReal.neg_le, EReal.le_neg, neg_zero]
  /-
    🎉 no goals
  -/


lemma mul_eq_top (a b : EReal) :
    a * b = ⊤ ↔ (a = ⊥ ∧ b < 0) ∨ (a < 0 ∧ b = ⊥) ∨ (a = ⊤ ∧ 0 < b) ∨ (0 < a ∧ b = ⊤) := by
  induction a, b using EReal.induction₂_symm with
  | symm h =>
    rw [EReal.mul_comm, h]
    refine ⟨fun H ↦ ?_, fun H ↦ ?_⟩ <;>
    cases H with
      | inl h => exact Or.inr (Or.inl ⟨h.2, h.1⟩)
      | inr h => cases h with
        | inl h => exact Or.inl ⟨h.2, h.1⟩
        | inr h => cases h with
          | inl h => exact Or.inr (Or.inr (Or.inr ⟨h.2, h.1⟩))
          | inr h => exact Or.inr (Or.inr (Or.inl ⟨h.2, h.1⟩))
  | top_top => simp
  | top_pos _ hx => simp [EReal.top_mul_coe_of_pos hx, hx]
  | top_zero => simp
  | top_neg _ hx => simp [hx.le, EReal.top_mul_coe_of_neg hx]
  | top_bot => simp
  | pos_bot _ hx => simp [hx.le, EReal.coe_mul_bot_of_pos hx]
  | coe_coe x y =>
    simpa only [EReal.coe_ne_bot, EReal.coe_neg', false_and, and_false, EReal.coe_ne_top,
      EReal.coe_pos, or_self, iff_false, EReal.coe_mul] using EReal.coe_ne_top _
  | zero_bot => simp
  | neg_bot _ hx => simp [hx, EReal.coe_mul_bot_of_neg hx]
  | bot_bot => simp


lemma mul_ne_top (a b : EReal) :
    a * b ≠ ⊤ ↔ (a ≠ ⊥ ∨ 0 ≤ b) ∧ (0 ≤ a ∨ b ≠ ⊥) ∧ (a ≠ ⊤ ∨ b ≤ 0) ∧ (a ≤ 0 ∨ b ≠ ⊤) := by
  /-
    a b : EReal
    ⊢ Iff (Ne (HMul.hMul a b) Top.top) (And (Or (Ne a Bot.bot) (LE.le 0 b)) (And ( …
  -/
  rw [ne_eq, mul_eq_top]
  -- push the negation while keeping the disjunctions, that is converting `¬(p ∧ q)` into `¬p ∨ ¬q`
  -- rather than `p → ¬q`, since we already have disjunctions in the rhs
  /-
    a b : EReal
    ⊢ Iff (Not (Or (And (Eq a Bot.bot) (LT.lt b 0)) (Or (And (LT.lt a 0) (Eq b Bot …
  -/
  set_option push_neg.use_distrib true in push_neg
  /-
    a b : EReal
    ⊢ Iff (And (Or (Ne a Bot.bot) (LE.le 0 b)) (And (Or (LE.le 0 a) (Ne b Bot.bot) …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma mul_eq_bot (a b : EReal) :
    a * b = ⊥ ↔ (a = ⊥ ∧ 0 < b) ∨ (0 < a ∧ b = ⊥) ∨ (a = ⊤ ∧ b < 0) ∨ (a < 0 ∧ b = ⊤) := by
  rw [← neg_eq_top_iff, ← EReal.neg_mul, mul_eq_top, neg_eq_bot_iff, neg_eq_top_iff,
    neg_lt_comm, lt_neg_comm, neg_zero]
  /-
    a b : EReal
    ⊢ Iff (Or (And (Eq a Top.top) (LT.lt b 0)) (Or (And (LT.lt 0 a) (Eq b Bot.bot) …
  -/
  tauto
  /-
    🎉 no goals
  -/


lemma mul_ne_bot (a b : EReal) :
    a * b ≠ ⊥ ↔ (a ≠ ⊥ ∨ b ≤ 0) ∧ (a ≤ 0 ∨ b ≠ ⊥) ∧ (a ≠ ⊤ ∨ 0 ≤ b) ∧ (0 ≤ a ∨ b ≠ ⊤) := by
  /-
    a b : EReal
    ⊢ Iff (Ne (HMul.hMul a b) Bot.bot) (And (Or (Ne a Bot.bot) (LE.le b 0)) (And ( …
  -/
  rw [ne_eq, mul_eq_bot]
  /-
    a b : EReal
    ⊢ Iff (Not (Or (And (Eq a Bot.bot) (LT.lt 0 b)) (Or (And (LT.lt 0 a) (Eq b Bot …
  -/
  set_option push_neg.use_distrib true in push_neg
  /-
    a b : EReal
    ⊢ Iff (And (Or (Ne a Bot.bot) (LE.le b 0)) (And (Or (LE.le a 0) (Ne b Bot.bot) …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma right_distrib_of_nonneg {a b c : EReal} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (a + b) * c = a * c + b * c := by
  /-
    a b c : EReal
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
  -/
  rcases eq_or_lt_of_le ha with (rfl | a_pos)
    /-
      case inl
      b c : EReal
      hb : LE.le 0 b
      ha : LE.le 0 0
      ⊢ Eq (HMul.hMul (HAdd.hAdd 0 b) c) (HAdd.hAdd (HMul.hMul 0 c) (HMul.hMul b c))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b c : EReal
    ha : LE.le 0 a
    hb : LE.le 0 b
    a_pos : LT.lt 0 a
    ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
  -/
  rcases eq_or_lt_of_le hb with (rfl | b_pos)
    /-
      case inr.inl
      a c : EReal
      ha : LE.le 0 a
      a_pos : LT.lt 0 a
      hb : LE.le 0 0
      ⊢ Eq (HMul.hMul (HAdd.hAdd a 0) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul 0 c))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b c : EReal
    ha : LE.le 0 a
    hb : LE.le 0 b
    a_pos : LT.lt 0 a
    b_pos : LT.lt 0 b
    ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
  -/
  rcases lt_trichotomy c 0 with (c_neg | rfl | c_pos)
    /-
      case inr.inr.inl
      a b c : EReal
      ha : LE.le 0 a
      hb : LE.le 0 b
      a_pos : LT.lt 0 a
      b_pos : LT.lt 0 b
      c_neg : LT.lt c 0
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
    -/
  · induction c
    · rw [mul_bot_of_pos a_pos, mul_bot_of_pos b_pos, mul_bot_of_pos (add_pos a_pos b_pos),
        add_bot ⊥]
      /-
        case inr.inr.inl.h_real
        a b : EReal
        ha : LE.le 0 a
        hb : LE.le 0 b
        a_pos : LT.lt 0 a
        b_pos : LT.lt 0 b
        a✝ : Real
        c_neg : LT.lt (↑a✝) 0
        ⊢ Eq (HMul.hMul (HAdd.hAdd a b) ↑a✝) (HAdd.hAdd (HMul.hMul a ↑a✝) (HMul.hMul b …
      -/
    · induction a
        /-
          case inr.inr.inl.h_real.h_bot
          b : EReal
          hb : LE.le 0 b
          b_pos : LT.lt 0 b
          a✝ : Real
          c_neg : LT.lt (↑a✝) 0
          ha : LE.le 0 Bot.bot
          a_pos : LT.lt 0 Bot.bot
          ⊢ Eq (HMul.hMul (HAdd.hAdd Bot.bot b) ↑a✝) (HAdd.hAdd (HMul.hMul Bot.bot ↑a✝)  …
        -/
      · exfalso; exact not_lt_bot a_pos
                 /-
                   🎉 no goals
                 -/
        /-
          case inr.inr.inl.h_real.h_real
          b : EReal
          hb : LE.le 0 b
          b_pos : LT.lt 0 b
          a✝¹ : Real
          c_neg : LT.lt (↑a✝¹) 0
          a✝ : Real
          ha : LE.le 0 ↑a✝
          a_pos : LT.lt 0 ↑a✝
          ⊢ Eq (HMul.hMul (HAdd.hAdd (↑a✝) b) ↑a✝¹) (HAdd.hAdd (HMul.hMul ↑a✝ ↑a✝¹) (HMu …
        -/
      · induction b
          /-
            case inr.inr.inl.h_real.h_real.h_bot
            a✝¹ : Real
            c_neg : LT.lt (↑a✝¹) 0
            a✝ : Real
            ha : LE.le 0 ↑a✝
            a_pos : LT.lt 0 ↑a✝
            hb : LE.le 0 Bot.bot
            b_pos : LT.lt 0 Bot.bot
            ⊢ Eq (HMul.hMul (HAdd.hAdd (↑a✝) Bot.bot) ↑a✝¹) (HAdd.hAdd (HMul.hMul ↑a✝ ↑a✝¹ …
          -/
        · norm_cast
          /-
            🎉 no goals
          -/
          /-
            case inr.inr.inl.h_real.h_real.h_real
            a✝² : Real
            c_neg : LT.lt (↑a✝²) 0
            a✝¹ : Real
            ha : LE.le 0 ↑a✝¹
            a_pos : LT.lt 0 ↑a✝¹
            a✝ : Real
            hb : LE.le 0 ↑a✝
            b_pos : LT.lt 0 ↑a✝
            ⊢ Eq (HMul.hMul (HAdd.hAdd ↑a✝¹ ↑a✝) ↑a✝²) (HAdd.hAdd (HMul.hMul ↑a✝¹ ↑a✝²) (H …
          -/
        · norm_cast; exact right_distrib _ _ _
                     /-
                       🎉 no goals
                     -/
          /-
            case inr.inr.inl.h_real.h_real.h_top
            a✝¹ : Real
            c_neg : LT.lt (↑a✝¹) 0
            a✝ : Real
            ha : LE.le 0 ↑a✝
            a_pos : LT.lt 0 ↑a✝
            hb : LE.le 0 Top.top
            b_pos : LT.lt 0 Top.top
            ⊢ Eq (HMul.hMul (HAdd.hAdd (↑a✝) Top.top) ↑a✝¹) (HAdd.hAdd (HMul.hMul ↑a✝ ↑a✝¹ …
          -/
        · norm_cast
          /-
            case inr.inr.inl.h_real.h_real.h_top
            a✝¹ : Real
            c_neg : LT.lt (↑a✝¹) 0
            a✝ : Real
            ha : LE.le 0 ↑a✝
            a_pos : LT.lt 0 ↑a✝
            hb : LE.le 0 Top.top
            b_pos : LT.lt 0 Top.top
            ⊢ Eq (HMul.hMul (HAdd.hAdd (↑a✝) Top.top) ↑a✝¹) (HAdd.hAdd (↑(HMul.hMul a✝ a✝¹ …
          -/
          rw [add_top_of_ne_bot (coe_ne_bot _), top_mul_of_neg c_neg, add_bot]
          /-
            🎉 no goals
          -/
        /-
          case inr.inr.inl.h_real.h_top
          b : EReal
          hb : LE.le 0 b
          b_pos : LT.lt 0 b
          a✝ : Real
          c_neg : LT.lt (↑a✝) 0
          ha : LE.le 0 Top.top
          a_pos : LT.lt 0 Top.top
          ⊢ Eq (HMul.hMul (HAdd.hAdd Top.top b) ↑a✝) (HAdd.hAdd (HMul.hMul Top.top ↑a✝)  …
        -/
      · rw [top_add_of_ne_bot (ne_bot_of_gt b_pos), top_mul_of_neg c_neg, bot_add]
        /-
          🎉 no goals
        -/
      /-
        case inr.inr.inl.h_top
        a b : EReal
        ha : LE.le 0 a
        hb : LE.le 0 b
        a_pos : LT.lt 0 a
        b_pos : LT.lt 0 b
        c_neg : LT.lt Top.top 0
        ⊢ Eq (HMul.hMul (HAdd.hAdd a b) Top.top) (HAdd.hAdd (HMul.hMul a Top.top) (HMu …
      -/
    · exfalso; exact not_top_lt c_neg
               /-
                 🎉 no goals
               -/
    /-
      case inr.inr.inr.inl
      a b : EReal
      ha : LE.le 0 a
      hb : LE.le 0 b
      a_pos : LT.lt 0 a
      b_pos : LT.lt 0 b
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) 0) (HAdd.hAdd (HMul.hMul a 0) (HMul.hMul b 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr.inr
      a b c : EReal
      ha : LE.le 0 a
      hb : LE.le 0 b
      a_pos : LT.lt 0 a
      b_pos : LT.lt 0 b
      c_pos : LT.lt 0 c
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
    -/
  · induction c
      /-
        case inr.inr.inr.inr.h_bot
        a b : EReal
        ha : LE.le 0 a
        hb : LE.le 0 b
        a_pos : LT.lt 0 a
        b_pos : LT.lt 0 b
        c_pos : LT.lt 0 Bot.bot
        ⊢ Eq (HMul.hMul (HAdd.hAdd a b) Bot.bot) (HAdd.hAdd (HMul.hMul a Bot.bot) (HMu …
      -/
    · exfalso; exact not_lt_bot c_pos
               /-
                 🎉 no goals
               -/
      /-
        case inr.inr.inr.inr.h_real
        a b : EReal
        ha : LE.le 0 a
        hb : LE.le 0 b
        a_pos : LT.lt 0 a
        b_pos : LT.lt 0 b
        a✝ : Real
        c_pos : LT.lt 0 ↑a✝
        ⊢ Eq (HMul.hMul (HAdd.hAdd a b) ↑a✝) (HAdd.hAdd (HMul.hMul a ↑a✝) (HMul.hMul b …
      -/
    · induction a
        /-
          case inr.inr.inr.inr.h_real.h_bot
          b : EReal
          hb : LE.le 0 b
          b_pos : LT.lt 0 b
          a✝ : Real
          c_pos : LT.lt 0 ↑a✝
          ha : LE.le 0 Bot.bot
          a_pos : LT.lt 0 Bot.bot
          ⊢ Eq (HMul.hMul (HAdd.hAdd Bot.bot b) ↑a✝) (HAdd.hAdd (HMul.hMul Bot.bot ↑a✝)  …
        -/
      · exfalso; exact not_lt_bot a_pos
                 /-
                   🎉 no goals
                 -/
        /-
          case inr.inr.inr.inr.h_real.h_real
          b : EReal
          hb : LE.le 0 b
          b_pos : LT.lt 0 b
          a✝¹ : Real
          c_pos : LT.lt 0 ↑a✝¹
          a✝ : Real
          ha : LE.le 0 ↑a✝
          a_pos : LT.lt 0 ↑a✝
          ⊢ Eq (HMul.hMul (HAdd.hAdd (↑a✝) b) ↑a✝¹) (HAdd.hAdd (HMul.hMul ↑a✝ ↑a✝¹) (HMu …
        -/
      · induction b
          /-
            case inr.inr.inr.inr.h_real.h_real.h_bot
            a✝¹ : Real
            c_pos : LT.lt 0 ↑a✝¹
            a✝ : Real
            ha : LE.le 0 ↑a✝
            a_pos : LT.lt 0 ↑a✝
            hb : LE.le 0 Bot.bot
            b_pos : LT.lt 0 Bot.bot
            ⊢ Eq (HMul.hMul (HAdd.hAdd (↑a✝) Bot.bot) ↑a✝¹) (HAdd.hAdd (HMul.hMul ↑a✝ ↑a✝¹ …
          -/
        · norm_cast
          /-
            🎉 no goals
          -/
          /-
            case inr.inr.inr.inr.h_real.h_real.h_real
            a✝² : Real
            c_pos : LT.lt 0 ↑a✝²
            a✝¹ : Real
            ha : LE.le 0 ↑a✝¹
            a_pos : LT.lt 0 ↑a✝¹
            a✝ : Real
            hb : LE.le 0 ↑a✝
            b_pos : LT.lt 0 ↑a✝
            ⊢ Eq (HMul.hMul (HAdd.hAdd ↑a✝¹ ↑a✝) ↑a✝²) (HAdd.hAdd (HMul.hMul ↑a✝¹ ↑a✝²) (H …
          -/
        · norm_cast; exact right_distrib _ _ _
                     /-
                       🎉 no goals
                     -/
          /-
            case inr.inr.inr.inr.h_real.h_real.h_top
            a✝¹ : Real
            c_pos : LT.lt 0 ↑a✝¹
            a✝ : Real
            ha : LE.le 0 ↑a✝
            a_pos : LT.lt 0 ↑a✝
            hb : LE.le 0 Top.top
            b_pos : LT.lt 0 Top.top
            ⊢ Eq (HMul.hMul (HAdd.hAdd (↑a✝) Top.top) ↑a✝¹) (HAdd.hAdd (HMul.hMul ↑a✝ ↑a✝¹ …
          -/
        · norm_cast
          rw [add_top_of_ne_bot (coe_ne_bot _), top_mul_of_pos c_pos,
            add_top_of_ne_bot (coe_ne_bot _)]
      · rw [top_add_of_ne_bot (ne_bot_of_gt b_pos), top_mul_of_pos c_pos,
          top_add_of_ne_bot (ne_bot_of_gt (mul_pos b_pos c_pos))]
    · rw [mul_top_of_pos a_pos, mul_top_of_pos b_pos, mul_top_of_pos (add_pos a_pos b_pos),
        top_add_top]


lemma left_distrib_of_nonneg {a b c : EReal} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    c * (a + b) = c * a + c * b := by
  /-
    a b c : EReal
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ Eq (HMul.hMul c (HAdd.hAdd a b)) (HAdd.hAdd (HMul.hMul c a) (HMul.hMul c b))
  -/
  nth_rewrite 1 [EReal.mul_comm]; nth_rewrite 2 [EReal.mul_comm]; nth_rewrite 3 [EReal.mul_comm]
  /-
    a b c : EReal
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
  -/
  exact right_distrib_of_nonneg ha hb
  /-
    🎉 no goals
  -/


lemma left_distrib_of_nonneg_of_ne_top {x : EReal} (hx_nonneg : 0 ≤ x)
    (hx_ne_top : x ≠ ⊤) (y z : EReal) :
    x * (y + z) = x * y + x * z := by
  cases hx_nonneg.eq_or_gt with
  | inl hx0 => simp [hx0]
  | inr hx0 =>
  lift x to ℝ using ⟨hx_ne_top, hx0.ne_bot⟩
  cases y <;> cases z <;>
    simp [mul_bot_of_pos hx0, mul_top_of_pos hx0, ← coe_mul];
    rw_mod_cast [mul_add]


lemma right_distrib_of_nonneg_of_ne_top {x : EReal} (hx_nonneg : 0 ≤ x)
    (hx_ne_top : x ≠ ⊤) (y z : EReal) :
    (y + z) * x = y * x + z * x := by
  /-
    x : EReal
    hx_nonneg : LE.le 0 x
    hx_ne_top : Ne x Top.top
    y z : EReal
    ⊢ Eq (HMul.hMul (HAdd.hAdd y z) x) (HAdd.hAdd (HMul.hMul y x) (HMul.hMul z x))
  -/
  simpa only [EReal.mul_comm] using left_distrib_of_nonneg_of_ne_top hx_nonneg hx_ne_top y z
  /-
    🎉 no goals
  -/


@[simp]
lemma nsmul_eq_mul (n : ℕ) (x : EReal) : n • x = n * x := by
  induction n with
  | zero => rw [zero_smul, Nat.cast_zero, zero_mul]
  | succ n ih =>
    rw [succ_nsmul, ih, Nat.cast_succ]
    convert (EReal.right_distrib_of_nonneg _ _).symm <;> simp


/-- The absolute value from `EReal` to `ℝ≥0∞`, mapping `⊥` and `⊤` to `⊤` and
a real `x` to `|x|`. -/
protected def abs : EReal → ℝ≥0∞
  | ⊥ => ⊤
  | ⊤ => ⊤
  | (x : ℝ) => ENNReal.ofReal |x|


@[simp] theorem abs_top : (⊤ : EReal).abs = ⊤ := rfl


@[simp] theorem abs_bot : (⊥ : EReal).abs = ⊤ := rfl


theorem abs_def (x : ℝ) : (x : EReal).abs = ENNReal.ofReal |x| := rfl


theorem abs_coe_lt_top (x : ℝ) : (x : EReal).abs < ⊤ :=
  ENNReal.ofReal_lt_top


@[simp]
theorem abs_eq_zero_iff {x : EReal} : x.abs = 0 ↔ x = 0 := by
  /-
    x : EReal
    ⊢ Iff (Eq x.abs 0) (Eq x 0)
  -/
  induction x
    /-
      case h_bot
      ⊢ Iff (Eq Bot.bot.abs 0) (Eq Bot.bot 0)
    -/
  · simp only [abs_bot, ENNReal.top_ne_zero, bot_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case h_real
      a✝ : Real
      ⊢ Iff (Eq (↑a✝).abs 0) (Eq (↑a✝) 0)
    -/
  · simp only [abs_def, coe_eq_zero, ENNReal.ofReal_eq_zero, abs_nonpos_iff]
    /-
      🎉 no goals
    -/
    /-
      case h_top
      ⊢ Iff (Eq Top.top.abs 0) (Eq Top.top 0)
    -/
  · simp only [abs_top, ENNReal.top_ne_zero, top_ne_zero]
    /-
      🎉 no goals
    -/


@[simp]
                                             /-
                                               ⊢ Eq (EReal.abs 0) 0
                                             -/
theorem abs_zero : (0 : EReal).abs = 0 := by rw [abs_eq_zero_iff]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem coe_abs (x : ℝ) : ((x : EReal).abs : EReal) = (|x| : ℝ) := by
  /-
    x : Real
    ⊢ Eq ↑(↑x).abs ↑(abs x)
  -/
  rw [abs_def, ← Real.coe_nnabs, ENNReal.ofReal_coe_nnreal]; rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
protected theorem abs_neg : ∀ x : EReal, (-x).abs = x.abs
  | ⊤ => rfl
  | ⊥ => rfl
                  /-
                    x : Real
                    ⊢ Eq (Neg.neg ↑x).abs (↑x).abs
                  -/
  | (x : ℝ) => by rw [abs_def, ← coe_neg, abs_def, abs_neg]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem abs_mul (x y : EReal) : (x * y).abs = x.abs * y.abs := by
  induction x, y using induction₂_symm_neg with
  | top_zero => simp only [zero_mul, mul_zero, abs_zero]
  | top_top => rfl
  | symm h => rwa [mul_comm, EReal.mul_comm]
  | coe_coe => simp only [← coe_mul, abs_def, _root_.abs_mul, ENNReal.ofReal_mul (abs_nonneg _)]
  | top_pos _ h =>
    rw [top_mul_coe_of_pos h, abs_top, ENNReal.top_mul]
    rw [Ne, abs_eq_zero_iff, coe_eq_zero]
    exact h.ne'
  | neg_left h => rwa [neg_mul, EReal.abs_neg, EReal.abs_neg]


theorem sign_top : sign (⊤ : EReal) = 1 := rfl


theorem sign_bot : sign (⊥ : EReal) = -1 := rfl


@[simp]
theorem sign_coe (x : ℝ) : sign (x : EReal) = sign x := by
  /-
    x : Real
    ⊢ Eq (SignType.sign ↑x) (SignType.sign x)
  -/
  simp only [sign, OrderHom.coe_mk, EReal.coe_pos, EReal.coe_neg']
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
                                                                  /-
                                                                    x : SignType
                                                                    ⊢ Eq ↑↑x ↑x
                                                                  -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
theorem coe_coe_sign (x : SignType) : ((x : ℝ) : EReal) = x := by cases x <;> rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp] theorem sign_neg : ∀ x : EReal, sign (-x) = -sign x
  | ⊤ => rfl
  | ⊥ => rfl
                  /-
                    x : Real
                    ⊢ Eq (SignType.sign (Neg.neg ↑x)) (Neg.neg (SignType.sign ↑x))
                  -/
  | (x : ℝ) => by rw [← coe_neg, sign_coe, sign_coe, Left.sign_neg]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem sign_mul (x y : EReal) : sign (x * y) = sign x * sign y := by
  induction x, y using induction₂_symm_neg with
  | top_zero => simp only [zero_mul, mul_zero, sign_zero]
  | top_top => rfl
  | symm h => rwa [mul_comm, EReal.mul_comm]
  | coe_coe => simp only [← coe_mul, sign_coe, _root_.sign_mul, ENNReal.ofReal_mul (abs_nonneg _)]
  | top_pos _ h =>
    rw [top_mul_coe_of_pos h, sign_top, one_mul, sign_pos (EReal.coe_pos.2 h)]
  | neg_left h => rw [neg_mul, sign_neg, sign_neg, h, neg_mul]


@[simp] protected theorem sign_mul_abs : ∀ x : EReal, (sign x * x.abs : EReal) = x
            /-
              ⊢ Eq (HMul.hMul ↑(SignType.sign Bot.bot) ↑Bot.bot.abs) Bot.bot
            -/
  | ⊥ => by simp
            /-
              🎉 no goals
            -/
            /-
              ⊢ Eq (HMul.hMul ↑(SignType.sign Top.top) ↑Top.top.abs) Top.top
            -/
  | ⊤ => by simp
            /-
              🎉 no goals
            -/
                  /-
                    x : Real
                    ⊢ Eq (HMul.hMul ↑(SignType.sign ↑x) ↑(↑x).abs) ↑x
                  -/
  | (x : ℝ) => by rw [sign_coe, coe_abs, ← coe_coe_sign, ← coe_mul, sign_mul_abs]
                  /-
                    🎉 no goals
                  -/


@[simp] protected theorem abs_mul_sign (x : EReal) : (x.abs * sign x : EReal) = x := by
  /-
    x : EReal
    ⊢ Eq (HMul.hMul ↑x.abs ↑(SignType.sign x)) x
  -/
  rw [EReal.mul_comm, EReal.sign_mul_abs]
  /-
    🎉 no goals
  -/


theorem sign_eq_and_abs_eq_iff_eq {x y : EReal} :
    x.abs = y.abs ∧ sign x = sign y ↔ x = y := by
  /-
    x y : EReal
    ⊢ Iff (And (Eq x.abs y.abs) (Eq (SignType.sign x) (SignType.sign y))) (Eq x y)
  -/
  constructor
    /-
      case mp
      x y : EReal
      ⊢ And (Eq x.abs y.abs) (Eq (SignType.sign x) (SignType.sign y)) → Eq x y
    -/
  · rintro ⟨habs, hsign⟩
    /-
      case mp.intro
      x y : EReal
      habs : Eq x.abs y.abs
      hsign : Eq (SignType.sign x) (SignType.sign y)
      ⊢ Eq x y
    -/
    rw [← x.sign_mul_abs, ← y.sign_mul_abs, habs, hsign]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x y : EReal
      ⊢ Eq x y → And (Eq x.abs y.abs) (Eq (SignType.sign x) (SignType.sign y))
    -/
  · rintro rfl
    /-
      case mpr
      x : EReal
      ⊢ And (Eq x.abs x.abs) (Eq (SignType.sign x) (SignType.sign x))
    -/
    exact ⟨rfl, rfl⟩
    /-
      🎉 no goals
    -/


theorem le_iff_sign {x y : EReal} :
    x ≤ y ↔ sign x < sign y ∨
      sign x = SignType.neg ∧ sign y = SignType.neg ∧ y.abs ≤ x.abs ∨
        sign x = SignType.zero ∧ sign y = SignType.zero ∨
          sign x = SignType.pos ∧ sign y = SignType.pos ∧ x.abs ≤ y.abs := by
  /-
    x y : EReal
    ⊢ Iff (LE.le x y) (Or (LT.lt (SignType.sign x) (SignType.sign y)) (Or (And (Eq …
  -/
  constructor
    /-
      case mp
      x y : EReal
      ⊢ LE.le x y → Or (LT.lt (SignType.sign x) (SignType.sign y)) (Or (And (Eq (Sig …
    -/
  · intro h
    /-
      case mp
      x y : EReal
      h : LE.le x y
      ⊢ Or (LT.lt (SignType.sign x) (SignType.sign y)) (Or (And (Eq (SignType.sign x …
    -/
    refine (sign.monotone h).lt_or_eq.imp_right (fun hs => ?_)
    /-
      case mp
      x y : EReal
      h : LE.le x y
      hs : Eq (SignType.sign x) (SignType.sign y)
      ⊢ Or (And (Eq (SignType.sign x) SignType.neg) (And (Eq (SignType.sign y) SignT …
    -/
    rw [← x.sign_mul_abs, ← y.sign_mul_abs] at h
    /-
      case mp
      x y : EReal
      h : LE.le (HMul.hMul ↑(SignType.sign x) ↑x.abs) (HMul.hMul ↑(SignType.sign y)  …
      hs : Eq (SignType.sign x) (SignType.sign y)
      ⊢ Or (And (Eq (SignType.sign x) SignType.neg) (And (Eq (SignType.sign y) SignT …
    -/
    cases hy : sign y <;> rw [hs, hy] at h ⊢
      /-
        case mp.zero
        x y : EReal
        h : LE.le (HMul.hMul ↑SignType.zero ↑x.abs) (HMul.hMul ↑SignType.zero ↑y.abs)
        hs : Eq (SignType.sign x) (SignType.sign y)
        hy : Eq (SignType.sign y) SignType.zero
        ⊢ Or (And (Eq SignType.zero SignType.neg) (And (Eq SignType.zero SignType.neg) …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mp.neg
        x y : EReal
        h : LE.le (HMul.hMul ↑SignType.neg ↑x.abs) (HMul.hMul ↑SignType.neg ↑y.abs)
        hs : Eq (SignType.sign x) (SignType.sign y)
        hy : Eq (SignType.sign y) SignType.neg
        ⊢ Or (And (Eq SignType.neg SignType.neg) (And (Eq SignType.neg SignType.neg) ( …
      -/
    · left; simpa using h
            /-
              🎉 no goals
            -/
      /-
        case mp.pos
        x y : EReal
        h : LE.le (HMul.hMul ↑SignType.pos ↑x.abs) (HMul.hMul ↑SignType.pos ↑y.abs)
        hs : Eq (SignType.sign x) (SignType.sign y)
        hy : Eq (SignType.sign y) SignType.pos
        ⊢ Or (And (Eq SignType.pos SignType.neg) (And (Eq SignType.pos SignType.neg) ( …
      -/
    · right; right; simpa using h
                    /-
                      🎉 no goals
                    -/
    /-
      case mpr
      x y : EReal
      ⊢ Or (LT.lt (SignType.sign x) (SignType.sign y)) (Or (And (Eq (SignType.sign x …
    -/
  · rintro (h | h | h | h)
      /-
        case mpr.inl
        x y : EReal
        h : LT.lt (SignType.sign x) (SignType.sign y)
        ⊢ LE.le x y
      -/
    · exact (sign.monotone.reflect_lt h).le
      /-
        🎉 no goals
      -/
    /-
      case mpr.inr.inl
      x y : EReal
      h : And (Eq (SignType.sign x) SignType.neg) (And (Eq (SignType.sign y) SignTyp …
      ⊢ LE.le x y
    -/
    all_goals rw [← x.sign_mul_abs, ← y.sign_mul_abs]; simp [h]
    /-
      🎉 no goals
    -/


instance : CommMonoidWithZero EReal :=
  { inferInstanceAs (MulZeroOneClass EReal) with
    mul_assoc := fun x y z => by
      /-
        x y z : EReal
        ⊢ Eq (HMul.hMul (HMul.hMul x y) z) (HMul.hMul x (HMul.hMul y z))
      -/
      rw [← sign_eq_and_abs_eq_iff_eq]
      /-
        x y z : EReal
        ⊢ And (Eq (HMul.hMul (HMul.hMul x y) z).abs (HMul.hMul x (HMul.hMul y z)).abs) …
      -/
      simp only [mul_assoc, abs_mul, eq_self_iff_true, sign_mul, and_self_iff]
      /-
        🎉 no goals
      -/
    mul_comm := EReal.mul_comm }


instance : PosMulMono EReal := posMulMono_iff_covariant_pos.2 <| .mk <| by
  /-
    ⊢ Covariant (Subtype fun x => LT.lt 0 x) EReal (fun x y => HMul.hMul (↑x) y) f …
  -/
  rintro ⟨x, x0⟩ a b h
  /-
    case mk
    x : EReal
    x0 : LT.lt 0 x
    a b : EReal
    h : LE.le a b
    ⊢ LE.le ((fun x y => HMul.hMul (↑x) y) ⟨x, x0⟩ a) ((fun x y => HMul.hMul (↑x)  …
  -/
  simp only [le_iff_sign, EReal.sign_mul, sign_pos x0, one_mul, EReal.abs_mul] at h ⊢
  exact h.imp_right <| Or.imp (And.imp_right <| And.imp_right (mul_le_mul_left' · _)) <|
    Or.imp_right <| And.imp_right <| And.imp_right (mul_le_mul_left' · _)


instance : MulPosMono EReal := posMulMono_iff_mulPosMono.1 inferInstance


instance : PosMulReflectLT EReal := PosMulMono.toPosMulReflectLT


instance : MulPosReflectLT EReal :=
  MulPosMono.toMulPosReflectLT


@[simp, norm_cast]
theorem coe_pow (x : ℝ) (n : ℕ) : (↑(x ^ n) : EReal) = (x : EReal) ^ n :=
  map_pow (⟨⟨(↑), coe_one⟩, coe_mul⟩ : ℝ →* EReal) _ _


@[simp, norm_cast]
theorem coe_ennreal_pow (x : ℝ≥0∞) (n : ℕ) : (↑(x ^ n) : EReal) = (x : EReal) ^ n :=
  map_pow (⟨⟨(↑), coe_ennreal_one⟩, coe_ennreal_mul⟩ : ℝ≥0∞ →* EReal) _ _


lemma min_neg_neg (x y : EReal) : min (-x) (-y) = -max x y := by
  /-
    x y : EReal
    ⊢ Eq (Min.min (Neg.neg x) (Neg.neg y)) (Neg.neg (Max.max x y))
  -/
                                       /-
                                         🎉 no goals
                                       -/
  rcases le_total x y with (h | h) <;> simp_all
                                       /-
                                         🎉 no goals
                                       -/


lemma max_neg_neg (x y : EReal) : max (-x) (-y) = -min x y := by
  /-
    x y : EReal
    ⊢ Eq (Max.max (Neg.neg x) (Neg.neg y)) (Neg.neg (Min.min x y))
  -/
                                       /-
                                         🎉 no goals
                                       -/
  rcases le_total x y with (h | h) <;> simp_all
                                       /-
                                         🎉 no goals
                                       -/


/-- Multiplicative inverse of an `EReal`. We choose `0⁻¹ = 0` to guarantee several good properties,
for instance `(a * b)⁻¹ = a⁻¹ * b⁻¹`. -/
protected def inv : EReal → EReal
  | ⊥ => 0
  | ⊤ => 0
  | (x : ℝ) => (x⁻¹ : ℝ)


instance : Inv (EReal) := ⟨EReal.inv⟩


noncomputable instance : DivInvMonoid EReal where inv := EReal.inv


@[simp]
lemma inv_bot : (⊥ : EReal)⁻¹ = 0 := rfl


@[simp]
lemma inv_top : (⊤ : EReal)⁻¹ = 0 := rfl


lemma coe_inv (x : ℝ) : (x⁻¹ : ℝ) = (x : EReal)⁻¹ := rfl


@[simp]
lemma inv_zero : (0 : EReal)⁻¹ = 0 := by
  /-
    ⊢ Eq (Inv.inv 0) 0
  -/
  change (0 : ℝ)⁻¹ = (0 : EReal)
  /-
    ⊢ Eq (↑(Inv.inv 0)) 0
  -/
  rw [GroupWithZero.inv_zero, coe_zero]
  /-
    🎉 no goals
  -/


noncomputable instance : DivInvOneMonoid EReal where
                /-
                  ⊢ Eq (Inv.inv 1) 1
                -/
  inv_one := by nth_rw 1 [← coe_one, ← coe_inv 1, _root_.inv_one, coe_one]
                /-
                  🎉 no goals
                -/


lemma inv_neg (a : EReal) : (-a)⁻¹ = -a⁻¹ := by
  /-
    a : EReal
    ⊢ Eq (Inv.inv (Neg.neg a)) (Neg.neg (Inv.inv a))
  -/
  induction a
    /-
      case h_bot
      ⊢ Eq (Inv.inv (Neg.neg Bot.bot)) (Neg.neg (Inv.inv Bot.bot))
    -/
  · rw [neg_bot, inv_top, inv_bot, neg_zero]
    /-
      🎉 no goals
    -/
    /-
      case h_real
      a✝ : Real
      ⊢ Eq (Inv.inv (Neg.neg ↑a✝)) (Neg.neg (Inv.inv ↑a✝))
    -/
  · rw [← coe_inv _, ← coe_neg _⁻¹, ← coe_neg _, ← coe_inv (-_)]
    /-
      case h_real
      a✝ : Real
      ⊢ Eq ↑(Inv.inv (Neg.neg a✝)) ↑(Neg.neg (Inv.inv a✝))
    -/
    exact EReal.coe_eq_coe_iff.2 _root_.inv_neg
    /-
      🎉 no goals
    -/
    /-
      case h_top
      ⊢ Eq (Inv.inv (Neg.neg Top.top)) (Neg.neg (Inv.inv Top.top))
    -/
  · rw [neg_top, inv_bot, inv_top, neg_zero]
    /-
      🎉 no goals
    -/


lemma inv_inv {a : EReal} (h : a ≠ ⊥) (h' : a ≠ ⊤) : (a⁻¹)⁻¹ = a := by
  /-
    a : EReal
    h : Ne a Bot.bot
    h' : Ne a Top.top
    ⊢ Eq (Inv.inv (Inv.inv a)) a
  -/
  rw [← coe_toReal h' h, ← coe_inv a.toReal, ← coe_inv a.toReal⁻¹, _root_.inv_inv a.toReal]
  /-
    🎉 no goals
  -/


lemma mul_inv (a b : EReal) : (a * b)⁻¹ = a⁻¹ * b⁻¹ := by
  induction a, b using EReal.induction₂_symm with
  | top_top | top_zero | top_bot | zero_bot | bot_bot => simp
  | @symm a b h => rw [mul_comm b a, mul_comm b⁻¹ a⁻¹]; exact h
  | top_pos x x_pos => rw [top_mul_of_pos (EReal.coe_pos.2 x_pos), inv_top, zero_mul]
  | top_neg x x_neg => rw [top_mul_of_neg (EReal.coe_neg'.2 x_neg), inv_bot, inv_top, zero_mul]
  | pos_bot x x_pos => rw [mul_bot_of_pos (EReal.coe_pos.2 x_pos), inv_bot, mul_zero]
  | coe_coe x y => rw [← coe_mul, ← coe_inv, _root_.mul_inv, coe_mul, coe_inv, coe_inv]
  | neg_bot x x_neg => rw [mul_bot_of_neg (EReal.coe_neg'.2 x_neg), inv_top, inv_bot, mul_zero]


lemma sign_mul_inv_abs (a : EReal) : (sign a) * (a.abs : EReal)⁻¹ = a⁻¹ := by
  induction a with
  | h_bot | h_top => simp
  | h_real a =>
    rcases lt_trichotomy a 0 with (a_neg | rfl | a_pos)
    · rw [sign_coe, _root_.sign_neg a_neg, coe_neg_one, neg_one_mul, ← inv_neg, abs_def a,
        coe_ennreal_ofReal, max_eq_left (abs_nonneg a), ← coe_neg |a|, abs_of_neg a_neg, neg_neg]
    · rw [coe_zero, sign_zero, SignType.coe_zero, abs_zero, coe_ennreal_zero, inv_zero, mul_zero]
    · rw [sign_coe, _root_.sign_pos a_pos, SignType.coe_one, one_mul]
      simp only [abs_def a, coe_ennreal_ofReal, abs_nonneg, max_eq_left]
      congr
      exact abs_of_pos a_pos


lemma sign_mul_inv_abs' (a : EReal) : (sign a) * ((a.abs⁻¹ : ℝ≥0∞) : EReal) = a⁻¹ := by
  induction a with
  | h_bot | h_top  => simp
  | h_real a =>
    rcases lt_trichotomy a 0 with (a_neg | rfl | a_pos)
    · rw [sign_coe, _root_.sign_neg a_neg, coe_neg_one, neg_one_mul, abs_def a,
        ← ofReal_inv_of_pos (abs_pos_of_neg a_neg), coe_ennreal_ofReal,
        max_eq_left (inv_nonneg.2 (abs_nonneg a)), ← coe_neg |a|⁻¹, ← coe_inv a, abs_of_neg a_neg,
        ← _root_.inv_neg, neg_neg]
    · simp
    · rw [sign_coe, _root_.sign_pos a_pos, SignType.coe_one, one_mul, abs_def a,
        ← ofReal_inv_of_pos (abs_pos_of_pos a_pos), coe_ennreal_ofReal,
          max_eq_left (inv_nonneg.2 (abs_nonneg a)), ← coe_inv a]
      congr
      exact abs_of_pos a_pos


lemma inv_nonneg_of_nonneg {a : EReal} (h : 0 ≤ a) : 0 ≤ a⁻¹ := by
  induction a with
  | h_bot | h_top => simp
  | h_real a => rw [← coe_inv a, EReal.coe_nonneg, inv_nonneg]; exact EReal.coe_nonneg.1 h


lemma inv_nonpos_of_nonpos {a : EReal} (h : a ≤ 0) : a⁻¹ ≤ 0 := by
  induction a with
  | h_bot | h_top => simp
  | h_real a => rw [← coe_inv a, EReal.coe_nonpos, inv_nonpos]; exact EReal.coe_nonpos.1 h


lemma inv_pos_of_pos_ne_top {a : EReal} (h : 0 < a) (h' : a ≠ ⊤) : 0 < a⁻¹ := by
  induction a with
  | h_bot => exact (not_lt_bot h).rec
  | h_real a =>  rw [← coe_inv a]; norm_cast at *; exact inv_pos_of_pos h
  | h_top => exact (h' (Eq.refl ⊤)).rec


lemma inv_neg_of_neg_ne_bot {a : EReal} (h : a < 0) (h' : a ≠ ⊥) : a⁻¹ < 0 := by
  induction a with
  | h_bot => exact (h' (Eq.refl ⊥)).rec
  | h_real a => rw [← coe_inv a]; norm_cast at *; exact inv_lt_zero.2 h
  | h_top => exact (not_top_lt h).rec


protected lemma div_eq_inv_mul (a b : EReal) : a / b = b⁻¹ * a := EReal.mul_comm a b⁻¹


lemma coe_div (a b : ℝ) : (a / b : ℝ) = (a : EReal) / (b : EReal) := rfl


theorem natCast_div_le (m n : ℕ) :
    (m / n : ℕ) ≤ (m : EReal) / (n : EReal) := by
  rw [← coe_coe_eq_natCast, ← coe_coe_eq_natCast, ← coe_coe_eq_natCast, ← coe_div,
    EReal.coe_le_coe_iff]
  /-
    m n : Nat
    ⊢ LE.le (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
  -/
  exact Nat.cast_div_le
  /-
    🎉 no goals
  -/


@[simp]
lemma div_bot {a : EReal} : a / ⊥ = 0 := inv_bot ▸ mul_zero a


@[simp]
lemma div_top {a : EReal} : a / ⊤ = 0 := inv_top ▸ mul_zero a


@[simp]
lemma div_zero {a : EReal} : a / 0 = 0 := by
  /-
    a : EReal
    ⊢ Eq (HDiv.hDiv a 0) 0
  -/
  change a * 0⁻¹ = 0
  /-
    a : EReal
    ⊢ Eq (HMul.hMul a (Inv.inv 0)) 0
  -/
  rw [inv_zero, mul_zero a]
  /-
    🎉 no goals
  -/


@[simp]
lemma zero_div {a : EReal} : 0 / a = 0 := zero_mul a⁻¹


lemma top_div_of_pos_ne_top {a : EReal} (h : 0 < a) (h' : a ≠ ⊤) : ⊤ / a = ⊤ :=
  top_mul_of_pos (inv_pos_of_pos_ne_top h h')


lemma top_div_of_neg_ne_bot {a : EReal} (h : a < 0) (h' : a ≠ ⊥) : ⊤ / a = ⊥ :=
  top_mul_of_neg (inv_neg_of_neg_ne_bot h h')


lemma bot_div_of_pos_ne_top {a : EReal} (h : 0 < a) (h' : a ≠ ⊤) : ⊥ / a = ⊥ :=
  bot_mul_of_pos (inv_pos_of_pos_ne_top h h')


lemma bot_div_of_neg_ne_bot {a : EReal} (h : a < 0) (h' : a ≠ ⊥) : ⊥ / a = ⊤ :=
  bot_mul_of_neg (inv_neg_of_neg_ne_bot h h')


lemma div_self {a : EReal} (h₁ : a ≠ ⊥) (h₂ : a ≠ ⊤) (h₃ : a ≠ 0) : a / a = 1 := by
  /-
    a : EReal
    h₁ : Ne a Bot.bot
    h₂ : Ne a Top.top
    h₃ : Ne a 0
    ⊢ Eq (HDiv.hDiv a a) 1
  -/
  rw [← coe_toReal h₂ h₁] at h₃ ⊢
  /-
    a : EReal
    h₁ : Ne a Bot.bot
    h₂ : Ne a Top.top
    h₃ : Ne (↑a.toReal) 0
    ⊢ Eq (HDiv.hDiv ↑a.toReal ↑a.toReal) 1
  -/
  rw [← coe_div, _root_.div_self (coe_ne_zero.1 h₃), coe_one]
  /-
    🎉 no goals
  -/


lemma mul_div (a b c : EReal) : a * (b / c) = (a * b) / c := by
  /-
    a b c : EReal
    ⊢ Eq (HMul.hMul a (HDiv.hDiv b c)) (HDiv.hDiv (HMul.hMul a b) c)
  -/
  change a * (b * c⁻¹) = (a * b) * c⁻¹
  /-
    a b c : EReal
    ⊢ Eq (HMul.hMul a (HMul.hMul b (Inv.inv c))) (HMul.hMul (HMul.hMul a b) (Inv.i …
  -/
  rw [mul_assoc]
  /-
    🎉 no goals
  -/


lemma mul_div_right (a b c : EReal) : (a / b) * c = (a * c) / b := by
  /-
    a b c : EReal
    ⊢ Eq (HMul.hMul (HDiv.hDiv a b) c) (HDiv.hDiv (HMul.hMul a c) b)
  -/
  rw [mul_comm, EReal.mul_div, mul_comm]
  /-
    🎉 no goals
  -/


lemma div_div (a b c : EReal) : a / b / c = a / (b * c) := by
  /-
    a b c : EReal
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv a b) c) (HDiv.hDiv a (HMul.hMul b c))
  -/
  change (a * b⁻¹) * c⁻¹ = a * (b * c)⁻¹
  /-
    a b c : EReal
    ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv b)) (Inv.inv c)) (HMul.hMul a (Inv.inv ( …
  -/
  rw [mul_assoc a b⁻¹, mul_inv]
  /-
    🎉 no goals
  -/


lemma div_mul_cancel {a b : EReal} (h₁ : b ≠ ⊥) (h₂ : b ≠ ⊤) (h₃ : b ≠ 0) : (a / b) * b = a := by
  /-
    a b : EReal
    h₁ : Ne b Bot.bot
    h₂ : Ne b Top.top
    h₃ : Ne b 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv a b) b) a
  -/
  change (a * b⁻¹) * b = a
  /-
    a b : EReal
    h₁ : Ne b Bot.bot
    h₂ : Ne b Top.top
    h₃ : Ne b 0
    ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv b)) b) a
  -/
  rw [mul_assoc, mul_comm b⁻¹ b]
  /-
    a b : EReal
    h₁ : Ne b Bot.bot
    h₂ : Ne b Top.top
    h₃ : Ne b 0
    ⊢ Eq (HMul.hMul a (HMul.hMul b (Inv.inv b))) a
  -/
  change a * (b / b) = a
  /-
    a b : EReal
    h₁ : Ne b Bot.bot
    h₂ : Ne b Top.top
    h₃ : Ne b 0
    ⊢ Eq (HMul.hMul a (HDiv.hDiv b b)) a
  -/
  rw [div_self h₁ h₂ h₃, mul_one]
  /-
    🎉 no goals
  -/


lemma mul_div_cancel {a b : EReal} (h₁ : b ≠ ⊥) (h₂ : b ≠ ⊤) (h₃ : b ≠ 0) : b * (a / b) = a := by
  /-
    a b : EReal
    h₁ : Ne b Bot.bot
    h₂ : Ne b Top.top
    h₃ : Ne b 0
    ⊢ Eq (HMul.hMul b (HDiv.hDiv a b)) a
  -/
  rw [mul_comm, div_mul_cancel h₁ h₂ h₃]
  /-
    🎉 no goals
  -/


lemma mul_div_mul_cancel {a b c : EReal} (h₁ : c ≠ ⊥) (h₂ : c ≠ ⊤) (h₃ : c ≠ 0) :
    (a * c) / (b * c) = a / b := by
  /-
    a b c : EReal
    h₁ : Ne c Bot.bot
    h₂ : Ne c Top.top
    h₃ : Ne c 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul a c) (HMul.hMul b c)) (HDiv.hDiv a b)
  -/
  change (a * c) * (b * c)⁻¹ = a * b⁻¹
  /-
    a b c : EReal
    h₁ : Ne c Bot.bot
    h₂ : Ne c Top.top
    h₃ : Ne c 0
    ⊢ Eq (HMul.hMul (HMul.hMul a c) (Inv.inv (HMul.hMul b c))) (HMul.hMul a (Inv.i …
  -/
  rw [mul_assoc, mul_inv b c]
  /-
    a b c : EReal
    h₁ : Ne c Bot.bot
    h₂ : Ne c Top.top
    h₃ : Ne c 0
    ⊢ Eq (HMul.hMul a (HMul.hMul c (HMul.hMul (Inv.inv b) (Inv.inv c)))) (HMul.hMu …
  -/
  congr
  /-
    case e_a
    a b c : EReal
    h₁ : Ne c Bot.bot
    h₂ : Ne c Top.top
    h₃ : Ne c 0
    ⊢ Eq (HMul.hMul c (HMul.hMul (Inv.inv b) (Inv.inv c))) (Inv.inv b)
  -/
  exact mul_div_cancel h₁ h₂ h₃
  /-
    🎉 no goals
  -/


lemma div_right_distrib_of_nonneg {a b c : EReal} (h : 0 ≤ a) (h' : 0 ≤ b) :
    (a + b) / c = (a / c) + (b / c) :=
  EReal.right_distrib_of_nonneg h h'


lemma monotone_div_right_of_nonneg {b : EReal} (h : 0 ≤ b) : Monotone fun a ↦ a / b :=
  fun _ _ h' ↦ mul_le_mul_of_nonneg_right h' (inv_nonneg_of_nonneg h)


lemma div_le_div_right_of_nonneg {a a' b : EReal} (h : 0 ≤ b) (h' : a ≤ a') :
    a / b ≤ a' / b :=
  monotone_div_right_of_nonneg h h'


lemma strictMono_div_right_of_pos {b : EReal} (h : 0 < b) (h' : b ≠ ⊤) :
    StrictMono fun a ↦ a / b := by
  /-
    b : EReal
    h : LT.lt 0 b
    h' : Ne b Top.top
    ⊢ StrictMono fun a => HDiv.hDiv a b
  -/
  intro a a' a_lt_a'
  /-
    b : EReal
    h : LT.lt 0 b
    h' : Ne b Top.top
    a a' : EReal
    a_lt_a' : LT.lt a a'
    ⊢ LT.lt ((fun a => HDiv.hDiv a b) a) ((fun a => HDiv.hDiv a b) a')
  -/
  apply lt_of_le_of_ne <| div_le_div_right_of_nonneg (le_of_lt h) (le_of_lt a_lt_a')
  /-
    b : EReal
    h : LT.lt 0 b
    h' : Ne b Top.top
    a a' : EReal
    a_lt_a' : LT.lt a a'
    ⊢ Ne (HDiv.hDiv a b) (HDiv.hDiv a' b)
  -/
  intro hyp
  /-
    b : EReal
    h : LT.lt 0 b
    h' : Ne b Top.top
    a a' : EReal
    a_lt_a' : LT.lt a a'
    hyp : Eq (HDiv.hDiv a b) (HDiv.hDiv a' b)
    ⊢ False
  -/
  apply ne_of_lt a_lt_a'
  rw [← @EReal.mul_div_cancel a b (ne_bot_of_gt h) h' (ne_of_gt h), hyp,
    @EReal.mul_div_cancel a' b (ne_bot_of_gt h) h' (ne_of_gt h)]


lemma div_lt_div_right_of_pos {a a' b : EReal} (h₁ : 0 < b) (h₂ : b ≠ ⊤)
    (h₃ : a < a') : a / b < a' / b :=
  strictMono_div_right_of_pos h₁ h₂ h₃


lemma antitone_div_right_of_nonpos {b : EReal} (h : b ≤ 0) : Antitone fun a ↦ a / b := by
  /-
    b : EReal
    h : LE.le b 0
    ⊢ Antitone fun a => HDiv.hDiv a b
  -/
  intro a a' h'
  /-
    b : EReal
    h : LE.le b 0
    a a' : EReal
    h' : LE.le a a'
    ⊢ LE.le ((fun a => HDiv.hDiv a b) a') ((fun a => HDiv.hDiv a b) a)
  -/
  change a' * b⁻¹ ≤ a * b⁻¹
  rw [← neg_neg (a * b⁻¹), ← neg_neg (a' * b⁻¹), neg_le_neg_iff, mul_comm a b⁻¹, mul_comm a' b⁻¹,
    ← neg_mul b⁻¹ a, ← neg_mul b⁻¹ a', mul_comm (-b⁻¹) a, mul_comm (-b⁻¹) a', ← inv_neg b]
  /-
    b : EReal
    h : LE.le b 0
    a a' : EReal
    h' : LE.le a a'
    ⊢ LE.le (HMul.hMul a (Inv.inv (Neg.neg b))) (HMul.hMul a' (Inv.inv (Neg.neg b)))
  -/
  have : 0 ≤ -b := by apply EReal.le_neg_of_le_neg; simp [h]
  /-
    b : EReal
    h : LE.le b 0
    a a' : EReal
    h' : LE.le a a'
    this : LE.le 0 (Neg.neg b)
    ⊢ LE.le (HMul.hMul a (Inv.inv (Neg.neg b))) (HMul.hMul a' (Inv.inv (Neg.neg b)))
  -/
  exact div_le_div_right_of_nonneg this h'
  /-
    🎉 no goals
  -/


lemma div_le_div_right_of_nonpos {a a' b : EReal} (h : b ≤ 0) (h' : a ≤ a') :
    a' / b ≤ a / b :=
  antitone_div_right_of_nonpos h h'


lemma strictAnti_div_right_of_neg {b : EReal} (h : b < 0) (h' : b ≠ ⊥) :
    StrictAnti fun a ↦ a / b := by
  /-
    b : EReal
    h : LT.lt b 0
    h' : Ne b Bot.bot
    ⊢ StrictAnti fun a => HDiv.hDiv a b
  -/
  intro a a' a_lt_a'
  /-
    b : EReal
    h : LT.lt b 0
    h' : Ne b Bot.bot
    a a' : EReal
    a_lt_a' : LT.lt a a'
    ⊢ LT.lt ((fun a => HDiv.hDiv a b) a') ((fun a => HDiv.hDiv a b) a)
  -/
  simp only
  /-
    b : EReal
    h : LT.lt b 0
    h' : Ne b Bot.bot
    a a' : EReal
    a_lt_a' : LT.lt a a'
    ⊢ LT.lt (HDiv.hDiv a' b) (HDiv.hDiv a b)
  -/
  apply lt_of_le_of_ne <| div_le_div_right_of_nonpos (le_of_lt h) (le_of_lt a_lt_a')
  /-
    b : EReal
    h : LT.lt b 0
    h' : Ne b Bot.bot
    a a' : EReal
    a_lt_a' : LT.lt a a'
    ⊢ Ne (HDiv.hDiv a' b) (HDiv.hDiv a b)
  -/
  intro hyp
  /-
    b : EReal
    h : LT.lt b 0
    h' : Ne b Bot.bot
    a a' : EReal
    a_lt_a' : LT.lt a a'
    hyp : Eq (HDiv.hDiv a' b) (HDiv.hDiv a b)
    ⊢ False
  -/
  apply ne_of_lt a_lt_a'
  rw [← @EReal.mul_div_cancel a b h' (ne_top_of_lt h) (ne_of_lt h), ← hyp,
    @EReal.mul_div_cancel a' b h' (ne_top_of_lt h) (ne_of_lt h)]


lemma div_lt_div_right_of_neg {a a' b : EReal} (h₁ : b < 0) (h₂ : b ≠ ⊥)
    (h₃ : a < a') : a' / b < a / b :=
  strictAnti_div_right_of_neg h₁ h₂ h₃


lemma le_div_iff_mul_le {a b c : EReal} (h : b > 0) (h' : b ≠ ⊤) :
    a ≤ c / b ↔ a * b ≤ c := by
  /-
    a b c : EReal
    h : GT.gt b 0
    h' : Ne b Top.top
    ⊢ Iff (LE.le a (HDiv.hDiv c b)) (LE.le (HMul.hMul a b) c)
  -/
  nth_rw 1 [← @mul_div_cancel a b (ne_bot_of_gt h) h' (ne_of_gt h)]
  /-
    a b c : EReal
    h : GT.gt b 0
    h' : Ne b Top.top
    ⊢ Iff (LE.le (HMul.hMul b (HDiv.hDiv a b)) (HDiv.hDiv c b)) (LE.le (HMul.hMul  …
  -/
  rw [mul_div b a b, mul_comm a b]
  /-
    a b c : EReal
    h : GT.gt b 0
    h' : Ne b Top.top
    ⊢ Iff (LE.le (HDiv.hDiv (HMul.hMul b a) b) (HDiv.hDiv c b)) (LE.le (HMul.hMul  …
  -/
  exact StrictMono.le_iff_le (strictMono_div_right_of_pos h h')
  /-
    🎉 no goals
  -/


lemma div_le_iff_le_mul {a b c : EReal} (h : 0 < b) (h' : b ≠ ⊤) :
    a / b ≤ c ↔ a ≤ b * c := by
  /-
    a b c : EReal
    h : LT.lt 0 b
    h' : Ne b Top.top
    ⊢ Iff (LE.le (HDiv.hDiv a b) c) (LE.le a (HMul.hMul b c))
  -/
  nth_rw 1 [← @mul_div_cancel c b (ne_bot_of_gt h) h' (ne_of_gt h)]
  /-
    a b c : EReal
    h : LT.lt 0 b
    h' : Ne b Top.top
    ⊢ Iff (LE.le (HDiv.hDiv a b) (HMul.hMul b (HDiv.hDiv c b))) (LE.le a (HMul.hMu …
  -/
  rw [mul_div b c b, mul_comm b]
  /-
    a b c : EReal
    h : LT.lt 0 b
    h' : Ne b Top.top
    ⊢ Iff (LE.le (HDiv.hDiv a b) (HDiv.hDiv (HMul.hMul c b) b)) (LE.le a (HMul.hMu …
  -/
  exact StrictMono.le_iff_le (strictMono_div_right_of_pos h h')
  /-
    🎉 no goals
  -/


lemma div_nonneg {a b : EReal} (h : 0 ≤ a) (h' : 0 ≤ b) : 0 ≤ a / b :=
  mul_nonneg h (inv_nonneg_of_nonneg h')


lemma div_nonpos_of_nonpos_of_nonneg {a b : EReal} (h : a ≤ 0) (h' : 0 ≤ b) : a / b ≤ 0 :=
  mul_nonpos_of_nonpos_of_nonneg h (inv_nonneg_of_nonneg h')


lemma div_nonpos_of_nonneg_of_nonpos {a b : EReal} (h : 0 ≤ a) (h' : b ≤ 0) : a / b ≤ 0 :=
  mul_nonpos_of_nonneg_of_nonpos h (inv_nonpos_of_nonpos h')


lemma div_nonneg_of_nonpos_of_nonpos {a b : EReal} (h : a ≤ 0) (h' : b ≤ 0) : 0 ≤ a / b :=
  le_of_eq_of_le (Eq.symm zero_div) (div_le_div_right_of_nonpos h' h)


