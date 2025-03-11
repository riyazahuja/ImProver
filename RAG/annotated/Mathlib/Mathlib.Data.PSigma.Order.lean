/-- The notation `Σₗ' i, α i` refers to a sigma type which is locally equipped with the
lexicographic order. -/
-- TODO: make `Lex` be `Sort u -> Sort u` so we can remove `.{_+1, _+1}`
notation3 "Σₗ' "(...)", "r:(scoped p => _root_.Lex (PSigma.{_+1, _+1} p)) => r


/-- The lexicographical `≤` on a sigma type. -/
instance le [LT ι] [∀ i, LE (α i)] : LE (Σₗ' i, α i) :=
  ⟨Lex (· < ·) fun _ => (· ≤ ·)⟩


/-- The lexicographical `<` on a sigma type. -/
instance lt [LT ι] [∀ i, LT (α i)] : LT (Σₗ' i, α i) :=
  ⟨Lex (· < ·) fun _ => (· < ·)⟩


instance preorder [Preorder ι] [∀ i, Preorder (α i)] : Preorder (Σₗ' i, α i) :=
  { Lex.le, Lex.lt with
    le_refl := fun ⟨_, _⟩ => Lex.right _ le_rfl,
    le_trans := by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : Preorder ι
        inst✝ : (i : ι) → Preorder (α i)
        ⊢ ∀ (a b c : _root_.Lex (PSigma fun i => α i)), LE.le a b → LE.le b c → LE.le  …
      -/
      rintro ⟨a₁, b₁⟩ ⟨a₂, b₂⟩ ⟨a₃, b₃⟩ ⟨h₁r⟩ ⟨h₂r⟩
        /-
          case mk.mk.mk.left.left
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a₁ : ι
          b₁ : α a₁
          a₂ : ι
          b₂ : α a₂
          a₃ : ι
          b₃ : α a₃
          a✝¹ : LT.lt a₁ a₂
          a✝ : LT.lt a₂ a₃
          ⊢ LE.le ⟨a₁, b₁⟩ ⟨a₃, b₃⟩
        -/
      · left
        /-
          case mk.mk.mk.left.left.a
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a₁ : ι
          b₁ : α a₁
          a₂ : ι
          b₂ : α a₂
          a₃ : ι
          b₃ : α a₃
          a✝¹ : LT.lt a₁ a₂
          a✝ : LT.lt a₂ a₃
          ⊢ LT.lt a₁ a₃
        -/
        apply lt_trans
        /-
          case mk.mk.mk.left.left.a.hab
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a₁ : ι
          b₁ : α a₁
          a₂ : ι
          b₂ : α a₂
          a₃ : ι
          b₃ : α a₃
          a✝¹ : LT.lt a₁ a₂
          a✝ : LT.lt a₂ a₃
          ⊢ LT.lt a₁ ?mk.mk.mk.left.left.a.b
        -/
        repeat' assumption
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.mk.left.right
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a₁ : ι
          b₁ : α a₁
          a₂ : ι
          b₂ : α a₂
          a✝¹ : LT.lt a₁ a₂
          b₂✝ : α a₂
          a✝ : LE.le b₂ b₂✝
          ⊢ LE.le ⟨a₁, b₁⟩ ⟨a₂, b₂✝⟩
        -/
      · left
        /-
          case mk.mk.mk.left.right.a
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a₁ : ι
          b₁ : α a₁
          a₂ : ι
          b₂ : α a₂
          a✝¹ : LT.lt a₁ a₂
          b₂✝ : α a₂
          a✝ : LE.le b₂ b₂✝
          ⊢ LT.lt a₁ a₂
        -/
        assumption
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.mk.right.left
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a₁ : ι
          b₁ : α a₁
          a₃ : ι
          b₃ : α a₃
          b₂✝ : α a₁
          a✝¹ : LE.le b₁ b₂✝
          a✝ : LT.lt a₁ a₃
          ⊢ LE.le ⟨a₁, b₁⟩ ⟨a₃, b₃⟩
        -/
      · left
        /-
          case mk.mk.mk.right.left.a
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a₁ : ι
          b₁ : α a₁
          a₃ : ι
          b₃ : α a₃
          b₂✝ : α a₁
          a✝¹ : LE.le b₁ b₂✝
          a✝ : LT.lt a₁ a₃
          ⊢ LT.lt a₁ a₃
        -/
        assumption
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.mk.right.right
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a₁ : ι
          b₁ b₂✝¹ : α a₁
          a✝¹ : LE.le b₁ b₂✝¹
          b₂✝ : α a₁
          a✝ : LE.le b₂✝¹ b₂✝
          ⊢ LE.le ⟨a₁, b₁⟩ ⟨a₁, b₂✝⟩
        -/
      · right
        /-
          case mk.mk.mk.right.right.a
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a₁ : ι
          b₁ b₂✝¹ : α a₁
          a✝¹ : LE.le b₁ b₂✝¹
          b₂✝ : α a₁
          a✝ : LE.le b₂✝¹ b₂✝
          ⊢ LE.le b₁ b₂✝
        -/
        apply le_trans
        /-
          case mk.mk.mk.right.right.a.a
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a₁ : ι
          b₁ b₂✝¹ : α a₁
          a✝¹ : LE.le b₁ b₂✝¹
          b₂✝ : α a₁
          a✝ : LE.le b₂✝¹ b₂✝
          ⊢ LE.le b₁ ?mk.mk.mk.right.right.a.b
        -/
        repeat' assumption,
        /-
          🎉 no goals
        -/
    lt_iff_le_not_le := by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : Preorder ι
        inst✝ : (i : ι) → Preorder (α i)
        ⊢ ∀ (a b : _root_.Lex (PSigma fun i => α i)), Iff (LT.lt a b) (And (LE.le a b) …
      -/
      refine fun a b => ⟨fun hab => ⟨hab.mono_right fun i a b => le_of_lt, ?_⟩, ?_⟩
        /-
          case refine_1
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : Preorder ι
          inst✝ : (i : ι) → Preorder (α i)
          a b : _root_.Lex (PSigma fun i => α i)
          hab : LT.lt a b
          ⊢ Not (LE.le b a)
        -/
      · rintro (⟨i, a, hji⟩ | ⟨i, hba⟩) <;> obtain ⟨_, _, hij⟩ | ⟨_, hab⟩ := hab
          /-
            case refine_1.left.left
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : Preorder ι
            inst✝ : (i : ι) → Preorder (α i)
            a₁✝ : ι
            i : α a₁✝
            a₂✝ : ι
            a : α a₂✝
            hji : LT.lt a₁✝ a₂✝
            hij : LT.lt a₂✝ a₁✝
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
            a₁✝ : ι
            i a : α a₁✝
            hji : LT.lt a₁✝ a₁✝
            hab : LT.lt a i
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
            i : ι
            b₁✝ b₂✝ : α i
            hba : LE.le b₁✝ b₂✝
            hij : LT.lt i i
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
            i : ι
            b₁✝ b₂✝ : α i
            hba : LE.le b₁✝ b₂✝
            hab : LT.lt b₂✝ b₁✝
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
          a b : _root_.Lex (PSigma fun i => α i)
          ⊢ And (LE.le a b) (Not (LE.le b a)) → LT.lt a b
        -/
      · rintro ⟨⟨j, b, hij⟩ | ⟨i, hab⟩, hba⟩
          /-
            case refine_2.intro.left
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : Preorder ι
            inst✝ : (i : ι) → Preorder (α i)
            a₁✝ : ι
            j : α a₁✝
            a₂✝ : ι
            b : α a₂✝
            hij : LT.lt a₁✝ a₂✝
            hba : Not (LE.le ⟨a₂✝, b⟩ ⟨a₁✝, j⟩)
            ⊢ LT.lt ⟨a₁✝, j⟩ ⟨a₂✝, b⟩
          -/
        · exact Lex.left _ _ hij
          /-
            🎉 no goals
          -/
          /-
            case refine_2.intro.right
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : Preorder ι
            inst✝ : (i : ι) → Preorder (α i)
            i : ι
            b₁✝ b₂✝ : α i
            hab : LE.le b₁✝ b₂✝
            hba : Not (LE.le ⟨i, b₂✝⟩ ⟨i, b₁✝⟩)
            ⊢ LT.lt ⟨i, b₁✝⟩ ⟨i, b₂✝⟩
          -/
        · exact Lex.right _ (hab.lt_of_not_le fun h => hba <| Lex.right _ h) }
          /-
            🎉 no goals
          -/


/-- Dictionary / lexicographic partial_order for dependent pairs. -/
instance partialOrder [PartialOrder ι] [∀ i, PartialOrder (α i)] : PartialOrder (Σₗ' i, α i) :=
  { Lex.preorder with
    le_antisymm := by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : PartialOrder ι
        inst✝ : (i : ι) → PartialOrder (α i)
        ⊢ ∀ (a b : _root_.Lex (PSigma fun i => α i)), LE.le a b → LE.le b a → Eq a b
      -/
      rintro ⟨a₁, b₁⟩ ⟨a₂, b₂⟩ (⟨_, _, hlt₁⟩ | ⟨_, hlt₁⟩) (⟨_, _, hlt₂⟩ | ⟨_, hlt₂⟩)
        /-
          case mk.mk.left.left
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : PartialOrder ι
          inst✝ : (i : ι) → PartialOrder (α i)
          a₁ : ι
          b₁ : α a₁
          a₂ : ι
          b₂ : α a₂
          hlt₁ : LT.lt a₁ a₂
          hlt₂ : LT.lt a₂ a₁
          ⊢ Eq ⟨a₁, b₁⟩ ⟨a₂, b₂⟩
        -/
      · exact (lt_irrefl a₁ <| hlt₁.trans hlt₂).elim
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.left.right
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : PartialOrder ι
          inst✝ : (i : ι) → PartialOrder (α i)
          a₁ : ι
          b₁ b₂ : α a₁
          hlt₁ : LT.lt a₁ a₁
          hlt₂ : LE.le b₂ b₁
          ⊢ Eq ⟨a₁, b₁⟩ ⟨a₁, b₂⟩
        -/
      · exact (lt_irrefl a₁ hlt₁).elim
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.right.left
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : PartialOrder ι
          inst✝ : (i : ι) → PartialOrder (α i)
          a₁ : ι
          b₁ b₂✝ : α a₁
          hlt₁ : LE.le b₁ b₂✝
          hlt₂ : LT.lt a₁ a₁
          ⊢ Eq ⟨a₁, b₁⟩ ⟨a₁, b₂✝⟩
        -/
      · exact (lt_irrefl a₁ hlt₂).elim
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.right.right
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : PartialOrder ι
          inst✝ : (i : ι) → PartialOrder (α i)
          a₁ : ι
          b₁ b₂✝ : α a₁
          hlt₁ : LE.le b₁ b₂✝
          hlt₂ : LE.le b₂✝ b₁
          ⊢ Eq ⟨a₁, b₁⟩ ⟨a₁, b₂✝⟩
        -/
      · rw [hlt₁.antisymm hlt₂] }
        /-
          🎉 no goals
        -/


/-- Dictionary / lexicographic linear_order for pairs. -/
instance linearOrder [LinearOrder ι] [∀ i, LinearOrder (α i)] : LinearOrder (Σₗ' i, α i) :=
  { Lex.partialOrder with
    le_total := by
      /-
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : LinearOrder ι
        inst✝ : (i : ι) → LinearOrder (α i)
        ⊢ ∀ (a b : _root_.Lex (PSigma fun i => α i)), Or (LE.le a b) (LE.le b a)
      -/
      rintro ⟨i, a⟩ ⟨j, b⟩
      /-
        case mk.mk
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : LinearOrder ι
        inst✝ : (i : ι) → LinearOrder (α i)
        i : ι
        a : α i
        j : ι
        b : α j
        ⊢ Or (LE.le ⟨i, a⟩ ⟨j, b⟩) (LE.le ⟨j, b⟩ ⟨i, a⟩)
      -/
      obtain hij | rfl | hji := lt_trichotomy i j
        /-
          case mk.mk.inl
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : LinearOrder ι
          inst✝ : (i : ι) → LinearOrder (α i)
          i : ι
          a : α i
          j : ι
          b : α j
          hij : LT.lt i j
          ⊢ Or (LE.le ⟨i, a⟩ ⟨j, b⟩) (LE.le ⟨j, b⟩ ⟨i, a⟩)
        -/
      · exact Or.inl (Lex.left _ _ hij)
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.inr.inl
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : LinearOrder ι
          inst✝ : (i : ι) → LinearOrder (α i)
          i : ι
          a b : α i
          ⊢ Or (LE.le ⟨i, a⟩ ⟨i, b⟩) (LE.le ⟨i, b⟩ ⟨i, a⟩)
        -/
      · obtain hab | hba := le_total a b
          /-
            case mk.mk.inr.inl.inl
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : LinearOrder ι
            inst✝ : (i : ι) → LinearOrder (α i)
            i : ι
            a b : α i
            hab : LE.le a b
            ⊢ Or (LE.le ⟨i, a⟩ ⟨i, b⟩) (LE.le ⟨i, b⟩ ⟨i, a⟩)
          -/
        · exact Or.inl (Lex.right _ hab)
          /-
            🎉 no goals
          -/
          /-
            case mk.mk.inr.inl.inr
            ι : Type u_1
            α : ι → Type u_2
            inst✝¹ : LinearOrder ι
            inst✝ : (i : ι) → LinearOrder (α i)
            i : ι
            a b : α i
            hba : LE.le b a
            ⊢ Or (LE.le ⟨i, a⟩ ⟨i, b⟩) (LE.le ⟨i, b⟩ ⟨i, a⟩)
          -/
        · exact Or.inr (Lex.right _ hba)
          /-
            🎉 no goals
          -/
        /-
          case mk.mk.inr.inr
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : LinearOrder ι
          inst✝ : (i : ι) → LinearOrder (α i)
          i : ι
          a : α i
          j : ι
          b : α j
          hji : LT.lt j i
          ⊢ Or (LE.le ⟨i, a⟩ ⟨j, b⟩) (LE.le ⟨j, b⟩ ⟨i, a⟩)
        -/
      · exact Or.inr (Lex.left _ _ hji),
        /-
          🎉 no goals
        -/
    decidableEq := PSigma.decidableEq, decidableLE := Lex.decidable _ _,
    decidableLT := Lex.decidable _ _ }


/-- The lexicographical linear order on a sigma type. -/
instance orderBot [PartialOrder ι] [OrderBot ι] [∀ i, Preorder (α i)] [OrderBot (α ⊥)] :
    OrderBot (Σₗ' i, α i) where
  bot := ⟨⊥, ⊥⟩
  bot_le := fun ⟨a, b⟩ => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : PartialOrder ι
      inst✝² : OrderBot ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : OrderBot (α Bot.bot)
      x✝ : _root_.Lex (PSigma fun i => α i)
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
        x✝ : _root_.Lex (PSigma fun i => α i)
        b : α Bot.bot
        ⊢ LE.le Bot.bot ⟨Bot.bot, b⟩
      -/
    · exact Lex.right _ bot_le
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
        x✝ : _root_.Lex (PSigma fun i => α i)
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
    OrderTop (Σₗ' i, α i) where
  top := ⟨⊤, ⊤⟩
  le_top := fun ⟨a, b⟩ => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : PartialOrder ι
      inst✝² : OrderTop ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : OrderTop (α Top.top)
      x✝ : _root_.Lex (PSigma fun i => α i)
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
        x✝ : _root_.Lex (PSigma fun i => α i)
        b : α Top.top
        ⊢ LE.le ⟨Top.top, b⟩ Top.top
      -/
    · exact Lex.right _ le_top
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
        x✝ : _root_.Lex (PSigma fun i => α i)
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
    [OrderTop (α ⊤)] : BoundedOrder (Σₗ' i, α i) :=
  { Lex.orderBot, Lex.orderTop with }


instance denselyOrdered [Preorder ι] [DenselyOrdered ι] [∀ i, Nonempty (α i)] [∀ i, Preorder (α i)]
    [∀ i, DenselyOrdered (α i)] : DenselyOrdered (Σₗ' i, α i) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝⁴ : Preorder ι
      inst✝³ : DenselyOrdered ι
      inst✝² : ∀ (i : ι), Nonempty (α i)
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), DenselyOrdered (α i)
      ⊢ ∀ (a₁ a₂ : _root_.Lex (PSigma fun i => α i)), LT.lt a₁ a₂ → Exists fun a =>  …
    -/
    rintro ⟨i, a⟩ ⟨j, b⟩ (⟨_, _, h⟩ | @⟨_, _, b, h⟩)
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
      exact ⟨⟨i, c⟩, right _ ha, right _ hb⟩⟩
      /-
        🎉 no goals
      -/


instance denselyOrdered_of_noMaxOrder [Preorder ι] [∀ i, Preorder (α i)]
    [∀ i, DenselyOrdered (α i)] [∀ i, NoMaxOrder (α i)] : DenselyOrdered (Σₗ' i, α i) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
      inst✝ : ∀ (i : ι), NoMaxOrder (α i)
      ⊢ ∀ (a₁ a₂ : _root_.Lex (PSigma fun i => α i)), LT.lt a₁ a₂ → Exists fun a =>  …
    -/
    rintro ⟨i, a⟩ ⟨j, b⟩ (⟨_, _, h⟩ | @⟨_, _, b, h⟩)
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
      exact ⟨⟨i, c⟩, right _ ha, left _ _ h⟩
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
      exact ⟨⟨i, c⟩, right _ ha, right _ hb⟩⟩
      /-
        🎉 no goals
      -/


instance denselyOrdered_of_noMinOrder [Preorder ι] [∀ i, Preorder (α i)]
    [∀ i, DenselyOrdered (α i)] [∀ i, NoMinOrder (α i)] : DenselyOrdered (Σₗ' i, α i) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : ∀ (i : ι), DenselyOrdered (α i)
      inst✝ : ∀ (i : ι), NoMinOrder (α i)
      ⊢ ∀ (a₁ a₂ : _root_.Lex (PSigma fun i => α i)), LT.lt a₁ a₂ → Exists fun a =>  …
    -/
    rintro ⟨i, a⟩ ⟨j, b⟩ (⟨_, _, h⟩ | @⟨_, _, b, h⟩)
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
      exact ⟨⟨j, c⟩, left _ _ h, right _ hb⟩
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
      exact ⟨⟨i, c⟩, right _ ha, right _ hb⟩⟩
      /-
        🎉 no goals
      -/


instance noMaxOrder_of_nonempty [Preorder ι] [∀ i, Preorder (α i)] [NoMaxOrder ι]
    [∀ i, Nonempty (α i)] : NoMaxOrder (Σₗ' i, α i) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : NoMaxOrder ι
      inst✝ : ∀ (i : ι), Nonempty (α i)
      ⊢ ∀ (a : _root_.Lex (PSigma fun i => α i)), Exists fun b => LT.lt a b
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
    exact ⟨⟨j, b⟩, left _ _ h⟩⟩
    /-
      🎉 no goals
    -/


instance noMinOrder_of_nonempty [Preorder ι] [∀ i, Preorder (α i)] [NoMinOrder ι]
    [∀ i, Nonempty (α i)] : NoMinOrder (Σₗ' i, α i) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝³ : Preorder ι
      inst✝² : (i : ι) → Preorder (α i)
      inst✝¹ : NoMinOrder ι
      inst✝ : ∀ (i : ι), Nonempty (α i)
      ⊢ ∀ (a : _root_.Lex (PSigma fun i => α i)), Exists fun b => LT.lt b a
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
    exact ⟨⟨j, b⟩, left _ _ h⟩⟩
    /-
      🎉 no goals
    -/


instance noMaxOrder [Preorder ι] [∀ i, Preorder (α i)] [∀ i, NoMaxOrder (α i)] :
    NoMaxOrder (Σₗ' i, α i) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), NoMaxOrder (α i)
      ⊢ ∀ (a : _root_.Lex (PSigma fun i => α i)), Exists fun b => LT.lt a b
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
    exact ⟨⟨i, b⟩, right _ h⟩⟩
    /-
      🎉 no goals
    -/


instance noMinOrder [Preorder ι] [∀ i, Preorder (α i)] [∀ i, NoMinOrder (α i)] :
    NoMinOrder (Σₗ' i, α i) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), NoMinOrder (α i)
      ⊢ ∀ (a : _root_.Lex (PSigma fun i => α i)), Exists fun b => LT.lt b a
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
    exact ⟨⟨i, b⟩, right _ h⟩⟩
    /-
      🎉 no goals
    -/


