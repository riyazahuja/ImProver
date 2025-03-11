instance : WellFoundedRelation (WithBot ℕ) where
  rel := (· < ·)
  wf := IsWellFounded.wf


theorem add_eq_zero_iff {n m : WithBot ℕ} : n + m = 0 ↔ n = 0 ∧ m = 0 := by
  /-
    n m : WithBot Nat
    ⊢ Iff (Eq (HAdd.hAdd n m) 0) (And (Eq n 0) (Eq m 0))
  -/
  cases n
    /-
      case bot
      m : WithBot Nat
      ⊢ Iff (Eq (HAdd.hAdd Bot.bot m) 0) (And (Eq Bot.bot 0) (Eq m 0))
    -/
  · simp [WithBot.bot_add]
    /-
      🎉 no goals
    -/
  /-
    case coe
    m : WithBot Nat
    a✝ : Nat
    ⊢ Iff (Eq (HAdd.hAdd (↑a✝) m) 0) (And (Eq (↑a✝) 0) (Eq m 0))
  -/
  cases m
    /-
      case coe.bot
      a✝ : Nat
      ⊢ Iff (Eq (HAdd.hAdd (↑a✝) Bot.bot) 0) (And (Eq (↑a✝) 0) (Eq Bot.bot 0))
    -/
  · simp [WithBot.add_bot]
    /-
      🎉 no goals
    -/
  /-
    case coe.coe
    a✝¹ a✝ : Nat
    ⊢ Iff (Eq (HAdd.hAdd ↑a✝¹ ↑a✝) 0) (And (Eq (↑a✝¹) 0) (Eq (↑a✝) 0))
  -/
  simp [← WithBot.coe_add, add_eq_zero_iff_of_nonneg]
  /-
    🎉 no goals
  -/


theorem add_eq_one_iff {n m : WithBot ℕ} : n + m = 1 ↔ n = 0 ∧ m = 1 ∨ n = 1 ∧ m = 0 := by
  /-
    n m : WithBot Nat
    ⊢ Iff (Eq (HAdd.hAdd n m) 1) (Or (And (Eq n 0) (Eq m 1)) (And (Eq n 1) (Eq m 0 …
  -/
  cases n
    /-
      case bot
      m : WithBot Nat
      ⊢ Iff (Eq (HAdd.hAdd Bot.bot m) 1) (Or (And (Eq Bot.bot 0) (Eq m 1)) (And (Eq  …
    -/
  · simp only [WithBot.bot_add, WithBot.bot_ne_one, WithBot.bot_ne_zero, false_and, or_self]
    /-
      🎉 no goals
    -/
  /-
    case coe
    m : WithBot Nat
    a✝ : Nat
    ⊢ Iff (Eq (HAdd.hAdd (↑a✝) m) 1) (Or (And (Eq (↑a✝) 0) (Eq m 1)) (And (Eq (↑a✝ …
  -/
  cases m
    /-
      case coe.bot
      a✝ : Nat
      ⊢ Iff (Eq (HAdd.hAdd (↑a✝) Bot.bot) 1) (Or (And (Eq (↑a✝) 0) (Eq Bot.bot 1)) ( …
    -/
  · simp [WithBot.add_bot]
    /-
      🎉 no goals
    -/
  /-
    case coe.coe
    a✝¹ a✝ : Nat
    ⊢ Iff (Eq (HAdd.hAdd ↑a✝¹ ↑a✝) 1) (Or (And (Eq (↑a✝¹) 0) (Eq (↑a✝) 1)) (And (E …
  -/
  simp [← WithBot.coe_add, Nat.add_eq_one_iff]
  /-
    🎉 no goals
  -/


theorem add_eq_two_iff {n m : WithBot ℕ} :
    n + m = 2 ↔ n = 0 ∧ m = 2 ∨ n = 1 ∧ m = 1 ∨ n = 2 ∧ m = 0 := by
  /-
    n m : WithBot Nat
    ⊢ Iff (Eq (HAdd.hAdd n m) 2) (Or (And (Eq n 0) (Eq m 2)) (Or (And (Eq n 1) (Eq …
  -/
  cases n
    /-
      case bot
      m : WithBot Nat
      ⊢ Iff (Eq (HAdd.hAdd Bot.bot m) 2) (Or (And (Eq Bot.bot 0) (Eq m 2)) (Or (And  …
    -/
  · simp [WithBot.bot_add]
    /-
      🎉 no goals
    -/
  /-
    case coe
    m : WithBot Nat
    a✝ : Nat
    ⊢ Iff (Eq (HAdd.hAdd (↑a✝) m) 2) (Or (And (Eq (↑a✝) 0) (Eq m 2)) (Or (And (Eq  …
  -/
  cases m
    /-
      case coe.bot
      a✝ : Nat
      ⊢ Iff (Eq (HAdd.hAdd (↑a✝) Bot.bot) 2) (Or (And (Eq (↑a✝) 0) (Eq Bot.bot 2)) ( …
    -/
  · simp [WithBot.add_bot]
    /-
      🎉 no goals
    -/
  /-
    case coe.coe
    a✝¹ a✝ : Nat
    ⊢ Iff (Eq (HAdd.hAdd ↑a✝¹ ↑a✝) 2) (Or (And (Eq (↑a✝¹) 0) (Eq (↑a✝) 2)) (Or (An …
  -/
  simp [← WithBot.coe_add, Nat.add_eq_two_iff]
  /-
    🎉 no goals
  -/


theorem add_eq_three_iff {n m : WithBot ℕ} :
    n + m = 3 ↔ n = 0 ∧ m = 3 ∨ n = 1 ∧ m = 2 ∨ n = 2 ∧ m = 1 ∨ n = 3 ∧ m = 0 := by
  /-
    n m : WithBot Nat
    ⊢ Iff (Eq (HAdd.hAdd n m) 3) (Or (And (Eq n 0) (Eq m 3)) (Or (And (Eq n 1) (Eq …
  -/
  cases n
    /-
      case bot
      m : WithBot Nat
      ⊢ Iff (Eq (HAdd.hAdd Bot.bot m) 3) (Or (And (Eq Bot.bot 0) (Eq m 3)) (Or (And  …
    -/
  · simp [WithBot.bot_add]
    /-
      🎉 no goals
    -/
  /-
    case coe
    m : WithBot Nat
    a✝ : Nat
    ⊢ Iff (Eq (HAdd.hAdd (↑a✝) m) 3) (Or (And (Eq (↑a✝) 0) (Eq m 3)) (Or (And (Eq  …
  -/
  cases m
    /-
      case coe.bot
      a✝ : Nat
      ⊢ Iff (Eq (HAdd.hAdd (↑a✝) Bot.bot) 3) (Or (And (Eq (↑a✝) 0) (Eq Bot.bot 3)) ( …
    -/
  · simp [WithBot.add_bot]
    /-
      🎉 no goals
    -/
  /-
    case coe.coe
    a✝¹ a✝ : Nat
    ⊢ Iff (Eq (HAdd.hAdd ↑a✝¹ ↑a✝) 3) (Or (And (Eq (↑a✝¹) 0) (Eq (↑a✝) 3)) (Or (An …
  -/
  simp [← WithBot.coe_add, Nat.add_eq_three_iff]
  /-
    🎉 no goals
  -/


theorem coe_nonneg {n : ℕ} : 0 ≤ (n : WithBot ℕ) := by
  /-
    n : Nat
    ⊢ LE.le 0 ↑n
  -/
  rw [← WithBot.coe_zero, cast_withBot, WithBot.coe_le_coe]
  /-
    n : Nat
    ⊢ LE.le 0 n
  -/
  exact n.zero_le
  /-
    🎉 no goals
  -/


@[simp]
theorem lt_zero_iff {n : WithBot ℕ} : n < 0 ↔ n = ⊥ := WithBot.lt_coe_bot


theorem one_le_iff_zero_lt {x : WithBot ℕ} : 1 ≤ x ↔ 0 < x := by
  /-
    x : WithBot Nat
    ⊢ Iff (LE.le 1 x) (LT.lt 0 x)
  -/
  refine ⟨zero_lt_one.trans_le, fun h => ?_⟩
  /-
    x : WithBot Nat
    h : LT.lt 0 x
    ⊢ LE.le 1 x
  -/
  cases x
    /-
      case bot
      h : LT.lt 0 Bot.bot
      ⊢ LE.le 1 Bot.bot
    -/
  · exact (not_lt_bot h).elim
    /-
      🎉 no goals
    -/
  · rwa [← WithBot.coe_zero, WithBot.coe_lt_coe, ← Nat.add_one_le_iff, zero_add,
      ← WithBot.coe_le_coe, WithBot.coe_one] at h


theorem lt_one_iff_le_zero {x : WithBot ℕ} : x < 1 ↔ x ≤ 0 :=
                     /-
                       x : WithBot Nat
                       ⊢ Iff (Not (LT.lt x 1)) (Not (LE.le x 0))
                     -/
  not_iff_not.mp (by simpa using one_le_iff_zero_lt)
                     /-
                       🎉 no goals
                     -/


theorem add_one_le_of_lt {n m : WithBot ℕ} (h : n < m) : n + 1 ≤ m := by
  /-
    n m : WithBot Nat
    h : LT.lt n m
    ⊢ LE.le (HAdd.hAdd n 1) m
  -/
  cases n
    /-
      case bot
      m : WithBot Nat
      h : LT.lt Bot.bot m
      ⊢ LE.le (HAdd.hAdd Bot.bot 1) m
    -/
  · simp only [WithBot.bot_add, bot_le]
    /-
      🎉 no goals
    -/
  /-
    case coe
    m : WithBot Nat
    a✝ : Nat
    h : LT.lt (↑a✝) m
    ⊢ LE.le (HAdd.hAdd (↑a✝) 1) m
  -/
  cases m
    /-
      case coe.bot
      a✝ : Nat
      h : LT.lt (↑a✝) Bot.bot
      ⊢ LE.le (HAdd.hAdd (↑a✝) 1) Bot.bot
    -/
  · exact (not_lt_bot h).elim
    /-
      🎉 no goals
    -/
  · rwa [WithBot.coe_lt_coe, ← Nat.add_one_le_iff, ← WithBot.coe_le_coe, WithBot.coe_add,
      WithBot.coe_one] at h


