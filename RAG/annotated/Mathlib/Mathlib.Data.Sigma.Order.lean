/-- Disjoint sum of orders. `⟨i, a⟩ ≤ ⟨j, b⟩` iff `i = j` and `a ≤ b`. -/
protected inductive le [∀ i, LE (α i)] : ∀ _a _b : Σ i, α i, Prop
  | fiber (i : ι) (a b : α i) : a ≤ b → Sigma.le ⟨i, a⟩ ⟨i, b⟩


/-- Disjoint sum of orders. `⟨i, a⟩ < ⟨j, b⟩` iff `i = j` and `a < b`. -/
protected inductive lt [∀ i, LT (α i)] : ∀ _a _b : Σi, α i, Prop
  | fiber (i : ι) (a b : α i) : a < b → Sigma.lt ⟨i, a⟩ ⟨i, b⟩


protected instance LE [∀ i, LE (α i)] : LE (Σi, α i) where
  le := Sigma.le


protected instance LT [∀ i, LT (α i)] : LT (Σi, α i) where
  lt := Sigma.lt


@[simp]
theorem mk_le_mk_iff [∀ i, LE (α i)] {i : ι} {a b : α i} : (⟨i, a⟩ : Sigma α) ≤ ⟨i, b⟩ ↔ a ≤ b :=
  ⟨fun ⟨_, _, _, h⟩ => h, Sigma.le.fiber _ _ _⟩


@[simp]
theorem mk_lt_mk_iff [∀ i, LT (α i)] {i : ι} {a b : α i} : (⟨i, a⟩ : Sigma α) < ⟨i, b⟩ ↔ a < b :=
  ⟨fun ⟨_, _, _, h⟩ => h, Sigma.lt.fiber _ _ _⟩


theorem le_def [∀ i, LE (α i)] {a b : Σi, α i} : a ≤ b ↔ ∃ h : a.1 = b.1, h.rec a.2 ≤ b.2 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → LE (α i)
    a b : Sigma fun i => α i
    ⊢ Iff (LE.le a b) (Exists fun h => LE.le (Eq.rec a.snd h) b.snd)
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LE (α i)
      a b : Sigma fun i => α i
      ⊢ LE.le a b → Exists fun h => LE.le (Eq.rec a.snd h) b.snd
    -/
  · rintro ⟨i, a, b, h⟩
    /-
      case mp.fiber
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LE (α i)
      i : ι
      a b : α i
      h : LE.le a b
      ⊢ Exists fun h => LE.le (Eq.rec ⟨i, a⟩.snd h) ⟨i, b⟩.snd
    -/
    exact ⟨rfl, h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LE (α i)
      a b : Sigma fun i => α i
      ⊢ (Exists fun h => LE.le (Eq.rec a.snd h) b.snd) → LE.le a b
    -/
  · obtain ⟨i, a⟩ := a
    /-
      case mpr.mk
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LE (α i)
      b : Sigma fun i => α i
      i : ι
      a : α i
      ⊢ (Exists fun h => LE.le (Eq.rec ⟨i, a⟩.snd h) b.snd) → LE.le ⟨i, a⟩ b
    -/
    obtain ⟨j, b⟩ := b
    /-
      case mpr.mk.mk
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LE (α i)
      i : ι
      a : α i
      j : ι
      b : α j
      ⊢ (Exists fun h => LE.le (Eq.rec ⟨i, a⟩.snd h) ⟨j, b⟩.snd) → LE.le ⟨i, a⟩ ⟨j, b⟩
    -/
    rintro ⟨rfl : i = j, h⟩
    /-
      case mpr.mk.mk.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LE (α i)
      i : ι
      a b : α i
      h : LE.le (Eq.rec ⟨i, a⟩.snd ⋯) ⟨i, b⟩.snd
      ⊢ LE.le ⟨i, a⟩ ⟨i, b⟩
    -/
    exact le.fiber _ _ _ h
    /-
      🎉 no goals
    -/


theorem lt_def [∀ i, LT (α i)] {a b : Σi, α i} : a < b ↔ ∃ h : a.1 = b.1, h.rec a.2 < b.2 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → LT (α i)
    a b : Sigma fun i => α i
    ⊢ Iff (LT.lt a b) (Exists fun h => LT.lt (Eq.rec a.snd h) b.snd)
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LT (α i)
      a b : Sigma fun i => α i
      ⊢ LT.lt a b → Exists fun h => LT.lt (Eq.rec a.snd h) b.snd
    -/
  · rintro ⟨i, a, b, h⟩
    /-
      case mp.fiber
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LT (α i)
      i : ι
      a b : α i
      h : LT.lt a b
      ⊢ Exists fun h => LT.lt (Eq.rec ⟨i, a⟩.snd h) ⟨i, b⟩.snd
    -/
    exact ⟨rfl, h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LT (α i)
      a b : Sigma fun i => α i
      ⊢ (Exists fun h => LT.lt (Eq.rec a.snd h) b.snd) → LT.lt a b
    -/
  · obtain ⟨i, a⟩ := a
    /-
      case mpr.mk
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LT (α i)
      b : Sigma fun i => α i
      i : ι
      a : α i
      ⊢ (Exists fun h => LT.lt (Eq.rec ⟨i, a⟩.snd h) b.snd) → LT.lt ⟨i, a⟩ b
    -/
    obtain ⟨j, b⟩ := b
    /-
      case mpr.mk.mk
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LT (α i)
      i : ι
      a : α i
      j : ι
      b : α j
      ⊢ (Exists fun h => LT.lt (Eq.rec ⟨i, a⟩.snd h) ⟨j, b⟩.snd) → LT.lt ⟨i, a⟩ ⟨j, b⟩
    -/
    rintro ⟨rfl : i = j, h⟩
    /-
      case mpr.mk.mk.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → LT (α i)
      i : ι
      a b : α i
      h : LT.lt (Eq.rec ⟨i, a⟩.snd ⋯) ⟨i, b⟩.snd
      ⊢ LT.lt ⟨i, a⟩ ⟨i, b⟩
    -/
    exact lt.fiber _ _ _ h
    /-
      🎉 no goals
    -/


protected instance preorder [∀ i, Preorder (α i)] : Preorder (Σi, α i) :=
  { Sigma.LE, Sigma.LT with
    le_refl := fun ⟨i, a⟩ => Sigma.le.fiber i a a le_rfl,
    le_trans := by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝ : (i : ι) → Preorder (α i)
        ⊢ ∀ (a b c : Sigma fun i => α i), LE.le a b → LE.le b c → LE.le a c
      -/
      rintro _ _ _ ⟨i, a, b, hab⟩ ⟨_, _, c, hbc⟩
      /-
        case fiber.fiber
        ι : Type u_1
        α : ι → Type u_2
        inst✝ : (i : ι) → Preorder (α i)
        i : ι
        a b : α i
        hab : LE.le a b
        c : α i
        hbc : LE.le b c
        ⊢ LE.le ⟨i, a⟩ ⟨i, c⟩
      -/
      exact le.fiber i a c (hab.trans hbc),
      /-
        🎉 no goals
      -/
    lt_iff_le_not_le := fun _ _ => by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝ : (i : ι) → Preorder (α i)
        x✝¹ x✝ : Sigma fun i => α i
        ⊢ Iff (LT.lt x✝¹ x✝) (And (LE.le x✝¹ x✝) (Not (LE.le x✝ x✝¹)))
      -/
      constructor
        /-
          case mp
          ι : Type u_1
          α : ι → Type u_2
          inst✝ : (i : ι) → Preorder (α i)
          x✝¹ x✝ : Sigma fun i => α i
          ⊢ LT.lt x✝¹ x✝ → And (LE.le x✝¹ x✝) (Not (LE.le x✝ x✝¹))
        -/
      · rintro ⟨i, a, b, hab⟩
        /-
          case mp.fiber
          ι : Type u_1
          α : ι → Type u_2
          inst✝ : (i : ι) → Preorder (α i)
          i : ι
          a b : α i
          hab : LT.lt a b
          ⊢ And (LE.le ⟨i, a⟩ ⟨i, b⟩) (Not (LE.le ⟨i, b⟩ ⟨i, a⟩))
        -/
        rwa [mk_le_mk_iff, mk_le_mk_iff, ← lt_iff_le_not_le]
        /-
          🎉 no goals
        -/
        /-
          case mpr
          ι : Type u_1
          α : ι → Type u_2
          inst✝ : (i : ι) → Preorder (α i)
          x✝¹ x✝ : Sigma fun i => α i
          ⊢ And (LE.le x✝¹ x✝) (Not (LE.le x✝ x✝¹)) → LT.lt x✝¹ x✝
        -/
      · rintro ⟨⟨i, a, b, hab⟩, h⟩
        /-
          case mpr.intro.fiber
          ι : Type u_1
          α : ι → Type u_2
          inst✝ : (i : ι) → Preorder (α i)
          i : ι
          a b : α i
          hab : LE.le a b
          h : Not (LE.le ⟨i, b⟩ ⟨i, a⟩)
          ⊢ LT.lt ⟨i, a⟩ ⟨i, b⟩
        -/
        rw [mk_le_mk_iff] at h
        /-
          case mpr.intro.fiber
          ι : Type u_1
          α : ι → Type u_2
          inst✝ : (i : ι) → Preorder (α i)
          i : ι
          a b : α i
          hab : LE.le a b
          h : Not (LE.le b a)
          ⊢ LT.lt ⟨i, a⟩ ⟨i, b⟩
        -/
        exact mk_lt_mk_iff.2 (hab.lt_of_not_le h) }
        /-
          🎉 no goals
        -/


instance [∀ i, PartialOrder (α i)] : PartialOrder (Σi, α i) :=
  { Sigma.preorder with
    le_antisymm := by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝ : (i : ι) → PartialOrder (α i)
        ⊢ ∀ (a b : Sigma fun i => α i), LE.le a b → LE.le b a → Eq a b
      -/
      rintro _ _ ⟨i, a, b, hab⟩ ⟨_, _, _, hba⟩
      /-
        case fiber.fiber
        ι : Type u_1
        α : ι → Type u_2
        inst✝ : (i : ι) → PartialOrder (α i)
        i : ι
        a b : α i
        hab : LE.le a b
        hba : LE.le b a
        ⊢ Eq ⟨i, a⟩ ⟨i, b⟩
      -/
      exact congr_arg (Sigma.mk _ ·) <| hab.antisymm hba }
      /-
        🎉 no goals
      -/


instance [∀ i, Preorder (α i)] [∀ i, DenselyOrdered (α i)] : DenselyOrdered (Σi, α i) where
  dense := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), DenselyOrdered (α i)
      ⊢ ∀ (a₁ a₂ : Sigma fun i => α i), LT.lt a₁ a₂ → Exists fun a => And (LT.lt a₁  …
    -/
    rintro ⟨i, a⟩ ⟨_, _⟩ ⟨_, _, b, h⟩
    /-
      case mk.mk.fiber
      ι : Type u_1
      α : ι → Type u_2
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), DenselyOrdered (α i)
      i : ι
      a b : α i
      h : LT.lt a b
      ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨i, b⟩)
    -/
    obtain ⟨c, ha, hb⟩ := exists_between h
    /-
      case mk.mk.fiber.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), DenselyOrdered (α i)
      i : ι
      a b : α i
      h : LT.lt a b
      c : α i
      ha : LT.lt a c
      hb : LT.lt c b
      ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨i, b⟩)
    -/
    exact ⟨⟨i, c⟩, lt.fiber i a c ha, lt.fiber i c b hb⟩
    /-
      🎉 no goals
    -/


/-- The notation `Σₗ i, α i` refers to a sigma type equipped with the lexicographic order. -/
notation3 "Σₗ "(...)", "r:(scoped p => _root_.Lex (Sigma p)) => r


/-- The lexicographical `≤` on a sigma type. -/
protected instance LE [LT ι] [∀ i, LE (α i)] : LE (Σₗ i, α i) where
  le := Lex (· < ·) fun _ => (· ≤ ·)


/-- The lexicographical `<` on a sigma type. -/
protected instance LT [LT ι] [∀ i, LT (α i)] : LT (Σₗ i, α i) where
  lt := Lex (· < ·) fun _ => (· < ·)


theorem le_def [LT ι] [∀ i, LE (α i)] {a b : Σₗ i, α i} :
    a ≤ b ↔ a.1 < b.1 ∨ ∃ h : a.1 = b.1, h.rec a.2 ≤ b.2 :=
  Sigma.lex_iff


theorem lt_def [LT ι] [∀ i, LT (α i)] {a b : Σₗ i, α i} :
    a < b ↔ a.1 < b.1 ∨ ∃ h : a.1 = b.1, h.rec a.2 < b.2 :=
  Sigma.lex_iff


/-- The lexicographical preorder on a sigma type. -/
instance preorder [Preorder ι] [∀ i, Preorder (α i)] : Preorder (Σₗ i, α i) :=
  { Sigma.Lex.LE, Sigma.Lex.LT with
    le_refl := fun ⟨_, a⟩ => Lex.right a a le_rfl,
    le_trans := fun _ _ _ => trans_of ((Lex (· < ·)) fun _ => (· ≤ ·)),
    lt_iff_le_not_le := by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : Preorder ι
        inst✝ : (i : ι) → Preorder (α i)
        ⊢ ∀ (a b : _root_.Lex (Sigma fun i => α i)), Iff (LT.lt a b) (And (LE.le a b)  …
      -/
      refine fun a b => ⟨fun hab => ⟨hab.mono_right fun i a b => le_of_lt, ?_⟩, ?_⟩
        /-
          case refine_1
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a b : _root_.Lex (Sigma fun i => α i)
          hab : LT.lt a b
          ⊢ Not (LE.le b a)
        -/
      · rintro (⟨b, a, hji⟩ | ⟨b, a, hba⟩) <;> obtain ⟨_, _, hij⟩ | ⟨_, _, hab⟩ := hab
          /-
            case refine_1.left.left
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : Preorder ι
            inst✝ : (i : ι) → Preorder (α i)
            i✝ j✝ : ι
            b : α i✝
            a : α j✝
            hji : LT.lt i✝ j✝
            hij : LT.lt j✝ i✝
            ⊢ False
          -/
        · exact hij.not_lt hji
          /-
            🎉 no goals
          -/
          /-
            case refine_1.left.right
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : Preorder ι
            inst✝ : (i : ι) → Preorder (α i)
            i✝ : ι
            b a : α i✝
            hji : LT.lt i✝ i✝
            hab : LT.lt a b
            ⊢ False
          -/
        · exact lt_irrefl _ hji
          /-
            🎉 no goals
          -/
          /-
            case refine_1.right.left
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : Preorder ι
            inst✝ : (i : ι) → Preorder (α i)
            i✝ : ι
            b a : α i✝
            hba : LE.le b a
            hij : LT.lt i✝ i✝
            ⊢ False
          -/
        · exact lt_irrefl _ hij
          /-
            🎉 no goals
          -/
          /-
            case refine_1.right.right
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : Preorder ι
            inst✝ : (i : ι) → Preorder (α i)
            i✝ : ι
            b a : α i✝
            hba : LE.le b a
            hab : LT.lt a b
            ⊢ False
          -/
        · exact hab.not_le hba
          /-
            🎉 no goals
          -/
        /-
          case refine_2
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a b : _root_.Lex (Sigma fun i => α i)
          ⊢ And (LE.le a b) (Not (LE.le b a)) → LT.lt a b
        -/
      · rintro ⟨⟨a, b, hij⟩ | ⟨a, b, hab⟩, hba⟩
          /-
            case refine_2.intro.left
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : Preorder ι
            inst✝ : (i : ι) → Preorder (α i)
            i✝ j✝ : ι
            a : α i✝
            b : α j✝
            hij : LT.lt i✝ j✝
            hba : Not (LE.le ⟨j✝, b⟩ ⟨i✝, a⟩)
            ⊢ LT.lt ⟨i✝, a⟩ ⟨j✝, b⟩
          -/
        · exact Sigma.Lex.left _ _ hij
          /-
            🎉 no goals
          -/
          /-
            case refine_2.intro.right
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : Preorder ι
            inst✝ : (i : ι) → Preorder (α i)
            i✝ : ι
            a b : α i✝
            hab : LE.le a b
            hba : Not (LE.le ⟨i✝, b⟩ ⟨i✝, a⟩)
            ⊢ LT.lt ⟨i✝, a⟩ ⟨i✝, b⟩
          -/
        · exact Sigma.Lex.right _ _ (hab.lt_of_not_le fun h => hba <| Sigma.Lex.right _ _ h) }
          /-
            🎉 no goals
          -/


/-- The lexicographical partial order on a sigma type. -/
instance partialOrder [Preorder ι] [∀ i, PartialOrder (α i)] :
    PartialOrder (Σₗ i, α i) :=
  { Lex.preorder with
    le_antisymm := fun _ _ => antisymm_of ((Lex (· < ·)) fun _ => (· ≤ ·)) }




/-- The lexicographical linear order on a sigma type. -/
instance linearOrder [LinearOrder ι] [∀ i, LinearOrder (α i)] :
    LinearOrder (Σₗ i, α i) :=
  { Lex.partialOrder with
    le_total := total_of ((Lex (· < ·)) fun _ => (· ≤ ·)),
    decidableEq := Sigma.instDecidableEqSigma,
    decidableLE := Lex.decidable _ _ }


/-- The lexicographical linear order on a sigma type. -/
instance orderBot [PartialOrder ι] [OrderBot ι] [∀ i, Preorder (α i)] [OrderBot (α ⊥)] :
    OrderBot (Σₗ i, α i) where
  bot := ⟨⊥, ⊥⟩
  bot_le := fun ⟨a, b⟩ => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : PartialOrder ι
      inst✝² : OrderBot ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : OrderBot (α Bot.bot)
      x✝ : _root_.Lex (Sigma fun i => α i)
      a : ι
      b : α a
      ⊢ LE.le Bot.bot ⟨a, b⟩
    -/
    obtain rfl | ha := eq_bot_or_bot_lt a
      /-
        case inl
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : PartialOrder ι
        inst✝² : OrderBot ι
        inst✝¹ : (i : ι) → Preorder (α i)
        inst✝ : OrderBot (α Bot.bot)
        x✝ : _root_.Lex (Sigma fun i => α i)
        b : α Bot.bot
        ⊢ LE.le Bot.bot ⟨Bot.bot, b⟩
      -/
    · exact Lex.right _ _ bot_le
      /-
        🎉 no goals
      -/
      /-
        case inr
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : PartialOrder ι
        inst✝² : OrderBot ι
        inst✝¹ : (i : ι) → Preorder (α i)
        inst✝ : OrderBot (α Bot.bot)
        x✝ : _root_.Lex (Sigma fun i => α i)
        a : ι
        b : α a
        ha : LT.lt Bot.bot a
        ⊢ LE.le Bot.bot ⟨a, b⟩
      -/
    · exact Lex.left _ _ ha
      /-
        🎉 no goals
      -/


/-- The lexicographical linear order on a sigma type. -/
instance orderTop [PartialOrder ι] [OrderTop ι] [∀ i, Preorder (α i)] [OrderTop (α ⊤)] :
    OrderTop (Σₗ i, α i) where
  top := ⟨⊤, ⊤⟩
  le_top := fun ⟨a, b⟩ => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : PartialOrder ι
      inst✝² : OrderTop ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : OrderTop (α Top.top)
      x✝ : _root_.Lex (Sigma fun i => α i)
      a : ι
      b : α a
      ⊢ LE.le ⟨a, b⟩ Top.top
    -/
    obtain rfl | ha := eq_top_or_lt_top a
      /-
        case inl
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : PartialOrder ι
        inst✝² : OrderTop ι
        inst✝¹ : (i : ι) → Preorder (α i)
        inst✝ : OrderTop (α Top.top)
        x✝ : _root_.Lex (Sigma fun i => α i)
        b : α Top.top
        ⊢ LE.le ⟨Top.top, b⟩ Top.top
      -/
    · exact Lex.right _ _ le_top
      /-
        🎉 no goals
      -/
      /-
        case inr
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : PartialOrder ι
        inst✝² : OrderTop ι
        inst✝¹ : (i : ι) → Preorder (α i)
        inst✝ : OrderTop (α Top.top)
        x✝ : _root_.Lex (Sigma fun i => α i)
        a : ι
        b : α a
        ha : LT.lt a Top.top
        ⊢ LE.le ⟨a, b⟩ Top.top
      -/
    · exact Lex.left _ _ ha
      /-
        🎉 no goals
      -/


/-- The lexicographical linear order on a sigma type. -/
instance boundedOrder [PartialOrder ι] [BoundedOrder ι] [∀ i, Preorder (α i)] [OrderBot (α ⊥)]
    [OrderTop (α ⊤)] : BoundedOrder (Σₗ i, α i) :=
  { Lex.orderBot, Lex.orderTop with }


instance denselyOrdered [Preorder ι] [DenselyOrdered ι] [∀ i, Nonempty (α i)] [∀ i, Preorder (α i)]
    [∀ i, DenselyOrdered (α i)] : DenselyOrdered (Σₗ i, α i) where
  dense := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝⁴ : Preorder ι
      inst✝³ : DenselyOrdered ι
      inst✝² : ∀ (i : ι), Nonempty (α i)
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), DenselyOrdered (α i)
      ⊢ ∀ (a₁ a₂ : _root_.Lex (Sigma fun i => α i)), LT.lt a₁ a₂ → Exists fun a => A …
    -/
    rintro ⟨i, a⟩ ⟨j, b⟩ (⟨_, _, h⟩ | ⟨_, b, h⟩)
      /-
        case mk.mk.left
        ι : Type u_1
        α : ι → Type u_2
        inst✝⁴ : Preorder ι
        inst✝³ : DenselyOrdered ι
        inst✝² : ∀ (i : ι), Nonempty (α i)
        inst✝¹ : (i : ι) → Preorder (α i)
        inst✝ : ∀ (i : ι), DenselyOrdered (α i)
        i : ι
        a : α i
        j : ι
        b : α j
        h : LT.lt i j
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨j, b⟩)
      -/
    · obtain ⟨k, hi, hj⟩ := exists_between h
      /-
        case mk.mk.left.intro.intro
        ι : Type u_1
        α : ι → Type u_2
        inst✝⁴ : Preorder ι
        inst✝³ : DenselyOrdered ι
        inst✝² : ∀ (i : ι), Nonempty (α i)
        inst✝¹ : (i : ι) → Preorder (α i)
        inst✝ : ∀ (i : ι), DenselyOrdered (α i)
        i : ι
        a : α i
        j : ι
        b : α j
        h : LT.lt i j
        k : ι
        hi : LT.lt i k
        hj : LT.lt k j
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨j, b⟩)
      -/
      obtain ⟨c⟩ : Nonempty (α k) := inferInstance
      /-
        case mk.mk.left.intro.intro.intro
        ι : Type u_1
        α : ι → Type u_2
        inst✝⁴ : Preorder ι
        inst✝³ : DenselyOrdered ι
        inst✝² : ∀ (i : ι), Nonempty (α i)
        inst✝¹ : (i : ι) → Preorder (α i)
        inst✝ : ∀ (i : ι), DenselyOrdered (α i)
        i : ι
        a : α i
        j : ι
        b : α j
        h : LT.lt i j
        k : ι
        hi : LT.lt i k
        hj : LT.lt k j
        c : α k
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨j, b⟩)
      -/
      exact ⟨⟨k, c⟩, left _ _ hi, left _ _ hj⟩
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.right
        ι : Type u_1
        α : ι → Type u_2
        inst✝⁴ : Preorder ι
        inst✝³ : DenselyOrdered ι
        inst✝² : ∀ (i : ι), Nonempty (α i)
        inst✝¹ : (i : ι) → Preorder (α i)
        inst✝ : ∀ (i : ι), DenselyOrdered (α i)
        i : ι
        a b : α i
        h : LT.lt a b
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨i, b⟩)
      -/
    · obtain ⟨c, ha, hb⟩ := exists_between h
      /-
        case mk.mk.right.intro.intro
        ι : Type u_1
        α : ι → Type u_2
        inst✝⁴ : Preorder ι
        inst✝³ : DenselyOrdered ι
        inst✝² : ∀ (i : ι), Nonempty (α i)
        inst✝¹ : (i : ι) → Preorder (α i)
        inst✝ : ∀ (i : ι), DenselyOrdered (α i)
        i : ι
        a b : α i
        h : LT.lt a b
        c : α i
        ha : LT.lt a c
        hb : LT.lt c b
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨i, b⟩)
      -/
      exact ⟨⟨i, c⟩, right _ _ ha, right _ _ hb⟩
      /-
        🎉 no goals
      -/


instance denselyOrdered_of_noMaxOrder [Preorder ι] [∀ i, Preorder (α i)]
    [∀ i, DenselyOrdered (α i)] [∀ i, NoMaxOrder (α i)] :
    DenselyOrdered (Σₗ i, α i) where
  dense := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
      inst✝ : ∀ (i : ι), NoMaxOrder (α i)
      ⊢ ∀ (a₁ a₂ : _root_.Lex (Sigma fun i => α i)), LT.lt a₁ a₂ → Exists fun a => A …
    -/
    rintro ⟨i, a⟩ ⟨j, b⟩ (⟨_, _, h⟩ | ⟨_, b, h⟩)
      /-
        case mk.mk.left
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : Preorder ι
        inst✝² : (i : ι) → Preorder (α i)
        inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
        inst✝ : ∀ (i : ι), NoMaxOrder (α i)
        i : ι
        a : α i
        j : ι
        b : α j
        h : LT.lt i j
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨j, b⟩)
      -/
    · obtain ⟨c, ha⟩ := exists_gt a
      /-
        case mk.mk.left.intro
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : Preorder ι
        inst✝² : (i : ι) → Preorder (α i)
        inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
        inst✝ : ∀ (i : ι), NoMaxOrder (α i)
        i : ι
        a : α i
        j : ι
        b : α j
        h : LT.lt i j
        c : α i
        ha : LT.lt a c
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨j, b⟩)
      -/
      exact ⟨⟨i, c⟩, right _ _ ha, left _ _ h⟩
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.right
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : Preorder ι
        inst✝² : (i : ι) → Preorder (α i)
        inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
        inst✝ : ∀ (i : ι), NoMaxOrder (α i)
        i : ι
        a b : α i
        h : LT.lt a b
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨i, b⟩)
      -/
    · obtain ⟨c, ha, hb⟩ := exists_between h
      /-
        case mk.mk.right.intro.intro
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : Preorder ι
        inst✝² : (i : ι) → Preorder (α i)
        inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
        inst✝ : ∀ (i : ι), NoMaxOrder (α i)
        i : ι
        a b : α i
        h : LT.lt a b
        c : α i
        ha : LT.lt a c
        hb : LT.lt c b
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨i, b⟩)
      -/
      exact ⟨⟨i, c⟩, right _ _ ha, right _ _ hb⟩
      /-
        🎉 no goals
      -/


instance denselyOrdered_of_noMinOrder [Preorder ι] [∀ i, Preorder (α i)]
    [∀ i, DenselyOrdered (α i)] [∀ i, NoMinOrder (α i)] :
    DenselyOrdered (Σₗ i, α i) where
  dense := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
      inst✝ : ∀ (i : ι), NoMinOrder (α i)
      ⊢ ∀ (a₁ a₂ : _root_.Lex (Sigma fun i => α i)), LT.lt a₁ a₂ → Exists fun a => A …
    -/
    rintro ⟨i, a⟩ ⟨j, b⟩ (⟨_, _, h⟩ | ⟨_, b, h⟩)
      /-
        case mk.mk.left
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : Preorder ι
        inst✝² : (i : ι) → Preorder (α i)
        inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
        inst✝ : ∀ (i : ι), NoMinOrder (α i)
        i : ι
        a : α i
        j : ι
        b : α j
        h : LT.lt i j
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨j, b⟩)
      -/
    · obtain ⟨c, hb⟩ := exists_lt b
      /-
        case mk.mk.left.intro
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : Preorder ι
        inst✝² : (i : ι) → Preorder (α i)
        inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
        inst✝ : ∀ (i : ι), NoMinOrder (α i)
        i : ι
        a : α i
        j : ι
        b : α j
        h : LT.lt i j
        c : α j
        hb : LT.lt c b
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨j, b⟩)
      -/
      exact ⟨⟨j, c⟩, left _ _ h, right _ _ hb⟩
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.right
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : Preorder ι
        inst✝² : (i : ι) → Preorder (α i)
        inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
        inst✝ : ∀ (i : ι), NoMinOrder (α i)
        i : ι
        a b : α i
        h : LT.lt a b
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨i, b⟩)
      -/
    · obtain ⟨c, ha, hb⟩ := exists_between h
      /-
        case mk.mk.right.intro.intro
        ι : Type u_1
        α : ι → Type u_2
        inst✝³ : Preorder ι
        inst✝² : (i : ι) → Preorder (α i)
        inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
        inst✝ : ∀ (i : ι), NoMinOrder (α i)
        i : ι
        a b : α i
        h : LT.lt a b
        c : α i
        ha : LT.lt a c
        hb : LT.lt c b
        ⊢ Exists fun a_1 => And (LT.lt ⟨i, a⟩ a_1) (LT.lt a_1 ⟨i, b⟩)
      -/
      exact ⟨⟨i, c⟩, right _ _ ha, right _ _ hb⟩
      /-
        🎉 no goals
      -/


instance noMaxOrder_of_nonempty [Preorder ι] [∀ i, Preorder (α i)] [NoMaxOrder ι]
    [∀ i, Nonempty (α i)] : NoMaxOrder (Σₗ i, α i) where
  exists_gt := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : NoMaxOrder ι
      inst✝ : ∀ (i : ι), Nonempty (α i)
      ⊢ ∀ (a : _root_.Lex (Sigma fun i => α i)), Exists fun b => LT.lt a b
    -/
    rintro ⟨i, a⟩
    /-
      case mk
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : NoMaxOrder ι
      inst✝ : ∀ (i : ι), Nonempty (α i)
      i : ι
      a : α i
      ⊢ Exists fun b => LT.lt ⟨i, a⟩ b
    -/
    obtain ⟨j, h⟩ := exists_gt i
    /-
      case mk.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : NoMaxOrder ι
      inst✝ : ∀ (i : ι), Nonempty (α i)
      i : ι
      a : α i
      j : ι
      h : LT.lt i j
      ⊢ Exists fun b => LT.lt ⟨i, a⟩ b
    -/
    obtain ⟨b⟩ : Nonempty (α j) := inferInstance
    /-
      case mk.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : NoMaxOrder ι
      inst✝ : ∀ (i : ι), Nonempty (α i)
      i : ι
      a : α i
      j : ι
      h : LT.lt i j
      b : α j
      ⊢ Exists fun b => LT.lt ⟨i, a⟩ b
    -/
    exact ⟨⟨j, b⟩, left _ _ h⟩
    /-
      🎉 no goals
    -/


instance noMinOrder_of_nonempty [Preorder ι] [∀ i, Preorder (α i)] [NoMinOrder ι]
    [∀ i, Nonempty (α i)] : NoMinOrder (Σₗ i, α i) where
  exists_lt := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : NoMinOrder ι
      inst✝ : ∀ (i : ι), Nonempty (α i)
      ⊢ ∀ (a : _root_.Lex (Sigma fun i => α i)), Exists fun b => LT.lt b a
    -/
    rintro ⟨i, a⟩
    /-
      case mk
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : NoMinOrder ι
      inst✝ : ∀ (i : ι), Nonempty (α i)
      i : ι
      a : α i
      ⊢ Exists fun b => LT.lt b ⟨i, a⟩
    -/
    obtain ⟨j, h⟩ := exists_lt i
    /-
      case mk.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : NoMinOrder ι
      inst✝ : ∀ (i : ι), Nonempty (α i)
      i : ι
      a : α i
      j : ι
      h : LT.lt j i
      ⊢ Exists fun b => LT.lt b ⟨i, a⟩
    -/
    obtain ⟨b⟩ : Nonempty (α j) := inferInstance
    /-
      case mk.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : NoMinOrder ι
      inst✝ : ∀ (i : ι), Nonempty (α i)
      i : ι
      a : α i
      j : ι
      h : LT.lt j i
      b : α j
      ⊢ Exists fun b => LT.lt b ⟨i, a⟩
    -/
    exact ⟨⟨j, b⟩, left _ _ h⟩
    /-
      🎉 no goals
    -/


instance noMaxOrder [Preorder ι] [∀ i, Preorder (α i)] [∀ i, NoMaxOrder (α i)] :
    NoMaxOrder (Σₗ i, α i) where
  exists_gt := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), NoMaxOrder (α i)
      ⊢ ∀ (a : _root_.Lex (Sigma fun i => α i)), Exists fun b => LT.lt a b
    -/
    rintro ⟨i, a⟩
    /-
      case mk
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), NoMaxOrder (α i)
      i : ι
      a : α i
      ⊢ Exists fun b => LT.lt ⟨i, a⟩ b
    -/
    obtain ⟨b, h⟩ := exists_gt a
    /-
      case mk.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), NoMaxOrder (α i)
      i : ι
      a b : α i
      h : LT.lt a b
      ⊢ Exists fun b => LT.lt ⟨i, a⟩ b
    -/
    exact ⟨⟨i, b⟩, right _ _ h⟩
    /-
      🎉 no goals
    -/


instance noMinOrder [Preorder ι] [∀ i, Preorder (α i)] [∀ i, NoMinOrder (α i)] :
    NoMinOrder (Σₗ i, α i) where
  exists_lt := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), NoMinOrder (α i)
      ⊢ ∀ (a : _root_.Lex (Sigma fun i => α i)), Exists fun b => LT.lt b a
    -/
    rintro ⟨i, a⟩
    /-
      case mk
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), NoMinOrder (α i)
      i : ι
      a : α i
      ⊢ Exists fun b => LT.lt b ⟨i, a⟩
    -/
    obtain ⟨b, h⟩ := exists_lt a
    /-
      case mk.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), NoMinOrder (α i)
      i : ι
      a b : α i
      h : LT.lt b a
      ⊢ Exists fun b => LT.lt b ⟨i, a⟩
    -/
    exact ⟨⟨i, b⟩, right _ _ h⟩
    /-
      🎉 no goals
    -/


