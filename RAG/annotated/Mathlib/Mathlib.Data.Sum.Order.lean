@[refl]
theorem LiftRel.refl [IsRefl α r] [IsRefl β s] : ∀ x, LiftRel r s x x
  | inl a => LiftRel.inl (_root_.refl a)
  | inr a => LiftRel.inr (_root_.refl a)


instance [IsRefl α r] [IsRefl β s] : IsRefl (α ⊕ β) (LiftRel r s) :=
  ⟨LiftRel.refl _ _⟩


instance [IsIrrefl α r] [IsIrrefl β s] : IsIrrefl (α ⊕ β) (LiftRel r s) :=
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsIrrefl α r
        inst✝ : IsIrrefl β s
        ⊢ ∀ (a : Sum α β), Not (Sum.LiftRel r s a a)
      -/
                               /-
                                 🎉 no goals
                               -/
  ⟨by rintro _ (⟨h⟩ | ⟨h⟩) <;> exact irrefl _ h⟩
                               /-
                                 🎉 no goals
                               -/


@[trans]
theorem LiftRel.trans [IsTrans α r] [IsTrans β s] :
    ∀ {a b c}, LiftRel r s a b → LiftRel r s b c → LiftRel r s a c
  | _, _, _, LiftRel.inl hab, LiftRel.inl hbc => LiftRel.inl <| _root_.trans hab hbc
  | _, _, _, LiftRel.inr hab, LiftRel.inr hbc => LiftRel.inr <| _root_.trans hab hbc


instance [IsTrans α r] [IsTrans β s] : IsTrans (α ⊕ β) (LiftRel r s) :=
  ⟨fun _ _ _ => LiftRel.trans _ _⟩


instance [IsAntisymm α r] [IsAntisymm β s] : IsAntisymm (α ⊕ β) (LiftRel r s) :=
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsAntisymm α r
        inst✝ : IsAntisymm β s
        ⊢ ∀ (a b : Sum α β), Sum.LiftRel r s a b → Sum.LiftRel r s b a → Eq a b
      -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  ⟨by rintro _ _ (⟨hab⟩ | ⟨hab⟩) (⟨hba⟩ | ⟨hba⟩) <;> rw [antisymm hab hba]⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


instance [IsRefl α r] [IsRefl β s] : IsRefl (α ⊕ β) (Lex r s) :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsRefl β s
      ⊢ ∀ (a : Sum α β), Sum.Lex r s a a
    -/
    rintro (a | a)
    /-
      case inl
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsRefl β s
      a : α
      ⊢ Sum.Lex r s (Sum.inl a) (Sum.inl a)
    -/
    exacts [Lex.inl (refl _), Lex.inr (refl _)]⟩
    /-
      🎉 no goals
    -/


instance [IsIrrefl α r] [IsIrrefl β s] : IsIrrefl (α ⊕ β) (Lex r s) :=
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsIrrefl α r
        inst✝ : IsIrrefl β s
        ⊢ ∀ (a : Sum α β), Not (Sum.Lex r s a a)
      -/
                               /-
                                 🎉 no goals
                               -/
  ⟨by rintro _ (⟨h⟩ | ⟨h⟩) <;> exact irrefl _ h⟩
                               /-
                                 🎉 no goals
                               -/


instance [IsTrans α r] [IsTrans β s] : IsTrans (α ⊕ β) (Lex r s) :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrans α r
      inst✝ : IsTrans β s
      ⊢ ∀ (a b c : Sum α β), Sum.Lex r s a b → Sum.Lex r s b c → Sum.Lex r s a c
    -/
    rintro _ _ _ (⟨hab⟩ | ⟨hab⟩) (⟨hbc⟩ | ⟨hbc⟩)
    /-
      case inl.inl
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrans α r
      inst✝ : IsTrans β s
      a₁✝ a₂✝¹ : α
      hab : r a₁✝ a₂✝¹
      a₂✝ : α
      hbc : r a₂✝¹ a₂✝
      ⊢ Sum.Lex r s (Sum.inl a₁✝) (Sum.inl a₂✝)
    -/
    exacts [.inl (_root_.trans hab hbc), .sep _ _, .inr (_root_.trans hab hbc), .sep _ _]⟩
    /-
      🎉 no goals
    -/


instance [IsAntisymm α r] [IsAntisymm β s] : IsAntisymm (α ⊕ β) (Lex r s) :=
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsAntisymm α r
        inst✝ : IsAntisymm β s
        ⊢ ∀ (a b : Sum α β), Sum.Lex r s a b → Sum.Lex r s b a → Eq a b
      -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  ⟨by rintro _ _ (⟨hab⟩ | ⟨hab⟩) (⟨hba⟩ | ⟨hba⟩) <;> rw [antisymm hab hba]⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


instance [IsTotal α r] [IsTotal β s] : IsTotal (α ⊕ β) (Lex r s) :=
  ⟨fun a b =>
    match a, b with
    | inl a, inl b => (total_of r a b).imp Lex.inl Lex.inl
    | inl _, inr _ => Or.inl (Lex.sep _ _)
    | inr _, inl _ => Or.inr (Lex.sep _ _)
    | inr a, inr b => (total_of s a b).imp Lex.inr Lex.inr⟩


instance [IsTrichotomous α r] [IsTrichotomous β s] : IsTrichotomous (α ⊕ β) (Lex r s) :=
  ⟨fun a b =>
    match a, b with
    | inl a, inl b => (trichotomous_of r a b).imp3 Lex.inl (congr_arg _) Lex.inl
    | inl _, inr _ => Or.inl (Lex.sep _ _)
    | inr _, inl _ => Or.inr (Or.inr <| Lex.sep _ _)
    | inr a, inr b => (trichotomous_of s a b).imp3 Lex.inr (congr_arg _) Lex.inr⟩


instance [IsWellOrder α r] [IsWellOrder β s] :
    IsWellOrder (α ⊕ β) (Sum.Lex r s) where wf := Sum.lex_wf IsWellFounded.wf IsWellFounded.wf


instance instLESum [LE α] [LE β] : LE (α ⊕ β) :=
  ⟨LiftRel (· ≤ ·) (· ≤ ·)⟩


instance instLTSum [LT α] [LT β] : LT (α ⊕ β) :=
  ⟨LiftRel (· < ·) (· < ·)⟩


theorem le_def [LE α] [LE β] {a b : α ⊕ β} : a ≤ b ↔ LiftRel (· ≤ ·) (· ≤ ·) a b :=
  Iff.rfl


theorem lt_def [LT α] [LT β] {a b : α ⊕ β} : a < b ↔ LiftRel (· < ·) (· < ·) a b :=
  Iff.rfl


@[simp]
theorem inl_le_inl_iff [LE α] [LE β] {a b : α} : (inl a : α ⊕ β) ≤ inl b ↔ a ≤ b :=
  liftRel_inl_inl


@[simp]
theorem inr_le_inr_iff [LE α] [LE β] {a b : β} : (inr a : α ⊕ β) ≤ inr b ↔ a ≤ b :=
  liftRel_inr_inr


@[simp]
theorem inl_lt_inl_iff [LT α] [LT β] {a b : α} : (inl a : α ⊕ β) < inl b ↔ a < b :=
  liftRel_inl_inl


@[simp]
theorem inr_lt_inr_iff [LT α] [LT β] {a b : β} : (inr a : α ⊕ β) < inr b ↔ a < b :=
  liftRel_inr_inr


@[simp]
theorem not_inl_le_inr [LE α] [LE β] {a : α} {b : β} : ¬inl b ≤ inr a :=
  not_liftRel_inl_inr


@[simp]
theorem not_inl_lt_inr [LT α] [LT β] {a : α} {b : β} : ¬inl b < inr a :=
  not_liftRel_inl_inr


@[simp]
theorem not_inr_le_inl [LE α] [LE β] {a : α} {b : β} : ¬inr b ≤ inl a :=
  not_liftRel_inr_inl


@[simp]
theorem not_inr_lt_inl [LT α] [LT β] {a : α} {b : β} : ¬inr b < inl a :=
  not_liftRel_inr_inl


instance instPreorderSum : Preorder (α ⊕ β) :=
  { instLESum, instLTSum with
    le_refl := fun _ => LiftRel.refl _ _ _,
    le_trans := fun _ _ _ => LiftRel.trans _ _,
    lt_iff_le_not_le := fun a b => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        a b : Sum α β
        ⊢ Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
      -/
      refine ⟨fun hab => ⟨hab.mono (fun _ _ => le_of_lt) fun _ _ => le_of_lt, ?_⟩, ?_⟩
        /-
          case refine_1
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝¹ : Preorder α
          inst✝ : Preorder β
          a b : Sum α β
          hab : LT.lt a b
          ⊢ Not (LE.le b a)
        -/
      · rintro (⟨hba⟩ | ⟨hba⟩)
          /-
            case refine_1.inl
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            a✝ c✝ : α
            hba : LE.le a✝ c✝
            hab : LT.lt (Sum.inl c✝) (Sum.inl a✝)
            ⊢ False
          -/
        · exact hba.not_lt (inl_lt_inl_iff.1 hab)
          /-
            🎉 no goals
          -/
          /-
            case refine_1.inr
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            b✝ d✝ : β
            hba : LE.le b✝ d✝
            hab : LT.lt (Sum.inr d✝) (Sum.inr b✝)
            ⊢ False
          -/
        · exact hba.not_lt (inr_lt_inr_iff.1 hab)
          /-
            🎉 no goals
          -/
        /-
          case refine_2
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝¹ : Preorder α
          inst✝ : Preorder β
          a b : Sum α β
          ⊢ And (LE.le a b) (Not (LE.le b a)) → LT.lt a b
        -/
      · rintro ⟨⟨hab⟩ | ⟨hab⟩, hba⟩
          /-
            case refine_2.intro.inl
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            a✝ c✝ : α
            hab : LE.le a✝ c✝
            hba : Not (LE.le (Sum.inl c✝) (Sum.inl a✝))
            ⊢ LT.lt (Sum.inl a✝) (Sum.inl c✝)
          -/
        · exact LiftRel.inl (hab.lt_of_not_le fun h => hba <| LiftRel.inl h)
          /-
            🎉 no goals
          -/
          /-
            case refine_2.intro.inr
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            b✝ d✝ : β
            hab : LE.le b✝ d✝
            hba : Not (LE.le (Sum.inr d✝) (Sum.inr b✝))
            ⊢ LT.lt (Sum.inr b✝) (Sum.inr d✝)
          -/
        · exact LiftRel.inr (hab.lt_of_not_le fun h => hba <| LiftRel.inr h) }
          /-
            🎉 no goals
          -/


theorem inl_mono : Monotone (inl : α → α ⊕ β) := fun _ _ => LiftRel.inl


theorem inr_mono : Monotone (inr : β → α ⊕ β) := fun _ _ => LiftRel.inr


theorem inl_strictMono : StrictMono (inl : α → α ⊕ β) := fun _ _ => LiftRel.inl


theorem inr_strictMono : StrictMono (inr : β → α ⊕ β) := fun _ _ => LiftRel.inr


instance [PartialOrder α] [PartialOrder β] : PartialOrder (α ⊕ β) :=
  { instPreorderSum with
    le_antisymm := fun _ _ => show LiftRel _ _ _ _ → _ from antisymm }


instance noMinOrder [LT α] [LT β] [NoMinOrder α] [NoMinOrder β] : NoMinOrder (α ⊕ β) :=
  ⟨fun a =>
    match a with
    | inl a =>
      let ⟨b, h⟩ := exists_lt a
      ⟨inl b, inl_lt_inl_iff.2 h⟩
    | inr a =>
      let ⟨b, h⟩ := exists_lt a
      ⟨inr b, inr_lt_inr_iff.2 h⟩⟩


instance noMaxOrder [LT α] [LT β] [NoMaxOrder α] [NoMaxOrder β] : NoMaxOrder (α ⊕ β) :=
  ⟨fun a =>
    match a with
    | inl a =>
      let ⟨b, h⟩ := exists_gt a
      ⟨inl b, inl_lt_inl_iff.2 h⟩
    | inr a =>
      let ⟨b, h⟩ := exists_gt a
      ⟨inr b, inr_lt_inr_iff.2 h⟩⟩


@[simp]
theorem noMinOrder_iff [LT α] [LT β] : NoMinOrder (α ⊕ β) ↔ NoMinOrder α ∧ NoMinOrder β :=
  ⟨fun _ =>
    ⟨⟨fun a => by
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : LT α
          inst✝ : LT β
          x✝ : NoMinOrder (Sum α β)
          a : α
          ⊢ Exists fun b => LT.lt b a
        -/
        obtain ⟨b | b, h⟩ := exists_lt (inl a : α ⊕ β)
          /-
            case intro.inl
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : NoMinOrder (Sum α β)
            a b : α
            h : LT.lt (Sum.inl b) (Sum.inl a)
            ⊢ Exists fun b => LT.lt b a
          -/
        · exact ⟨b, inl_lt_inl_iff.1 h⟩
          /-
            🎉 no goals
          -/
          /-
            case intro.inr
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : NoMinOrder (Sum α β)
            a : α
            b : β
            h : LT.lt (Sum.inr b) (Sum.inl a)
            ⊢ Exists fun b => LT.lt b a
          -/
        · exact (not_inr_lt_inl h).elim⟩,
          /-
            🎉 no goals
          -/
      ⟨fun a => by
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : LT α
          inst✝ : LT β
          x✝ : NoMinOrder (Sum α β)
          a : β
          ⊢ Exists fun b => LT.lt b a
        -/
        obtain ⟨b | b, h⟩ := exists_lt (inr a : α ⊕ β)
          /-
            case intro.inl
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : NoMinOrder (Sum α β)
            a : β
            b : α
            h : LT.lt (Sum.inl b) (Sum.inr a)
            ⊢ Exists fun b => LT.lt b a
          -/
        · exact (not_inl_lt_inr h).elim
          /-
            🎉 no goals
          -/
          /-
            case intro.inr
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : NoMinOrder (Sum α β)
            a b : β
            h : LT.lt (Sum.inr b) (Sum.inr a)
            ⊢ Exists fun b => LT.lt b a
          -/
        · exact ⟨b, inr_lt_inr_iff.1 h⟩⟩⟩,
          /-
            🎉 no goals
          -/
    fun h => @Sum.noMinOrder _ _ _ _ h.1 h.2⟩


@[simp]
theorem noMaxOrder_iff [LT α] [LT β] : NoMaxOrder (α ⊕ β) ↔ NoMaxOrder α ∧ NoMaxOrder β :=
  ⟨fun _ =>
    ⟨⟨fun a => by
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : LT α
          inst✝ : LT β
          x✝ : NoMaxOrder (Sum α β)
          a : α
          ⊢ Exists fun b => LT.lt a b
        -/
        obtain ⟨b | b, h⟩ := exists_gt (inl a : α ⊕ β)
          /-
            case intro.inl
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : NoMaxOrder (Sum α β)
            a b : α
            h : LT.lt (Sum.inl a) (Sum.inl b)
            ⊢ Exists fun b => LT.lt a b
          -/
        · exact ⟨b, inl_lt_inl_iff.1 h⟩
          /-
            🎉 no goals
          -/
          /-
            case intro.inr
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : NoMaxOrder (Sum α β)
            a : α
            b : β
            h : LT.lt (Sum.inl a) (Sum.inr b)
            ⊢ Exists fun b => LT.lt a b
          -/
        · exact (not_inl_lt_inr h).elim⟩,
          /-
            🎉 no goals
          -/
      ⟨fun a => by
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : LT α
          inst✝ : LT β
          x✝ : NoMaxOrder (Sum α β)
          a : β
          ⊢ Exists fun b => LT.lt a b
        -/
        obtain ⟨b | b, h⟩ := exists_gt (inr a : α ⊕ β)
          /-
            case intro.inl
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : NoMaxOrder (Sum α β)
            a : β
            b : α
            h : LT.lt (Sum.inr a) (Sum.inl b)
            ⊢ Exists fun b => LT.lt a b
          -/
        · exact (not_inr_lt_inl h).elim
          /-
            🎉 no goals
          -/
          /-
            case intro.inr
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : NoMaxOrder (Sum α β)
            a b : β
            h : LT.lt (Sum.inr a) (Sum.inr b)
            ⊢ Exists fun b => LT.lt a b
          -/
        · exact ⟨b, inr_lt_inr_iff.1 h⟩⟩⟩,
          /-
            🎉 no goals
          -/
    fun h => @Sum.noMaxOrder _ _ _ _ h.1 h.2⟩


instance denselyOrdered [LT α] [LT β] [DenselyOrdered α] [DenselyOrdered β] :
    DenselyOrdered (α ⊕ β) :=
  ⟨fun a b h =>
    match a, b, h with
    | inl _, inl _, LiftRel.inl h =>
      let ⟨c, ha, hb⟩ := exists_between h
      ⟨toLex (inl c), LiftRel.inl ha, LiftRel.inl hb⟩
    | inr _, inr _, LiftRel.inr h =>
      let ⟨c, ha, hb⟩ := exists_between h
      ⟨toLex (inr c), LiftRel.inr ha, LiftRel.inr hb⟩⟩


@[simp]
theorem denselyOrdered_iff [LT α] [LT β] :
    DenselyOrdered (α ⊕ β) ↔ DenselyOrdered α ∧ DenselyOrdered β :=
  ⟨fun _ =>
    ⟨⟨fun a b h => by
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : LT α
          inst✝ : LT β
          x✝ : DenselyOrdered (Sum α β)
          a b : α
          h : LT.lt a b
          ⊢ Exists fun a_1 => And (LT.lt a a_1) (LT.lt a_1 b)
        -/
        obtain ⟨c | c, ha, hb⟩ := @exists_between (α ⊕ β) _ _ _ _ (inl_lt_inl_iff.2 h)
          /-
            case intro.inl.intro
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : DenselyOrdered (Sum α β)
            a b : α
            h : LT.lt a b
            c : α
            ha : LT.lt (Sum.inl a) (Sum.inl c)
            hb : LT.lt (Sum.inl c) (Sum.inl b)
            ⊢ Exists fun a_1 => And (LT.lt a a_1) (LT.lt a_1 b)
          -/
        · exact ⟨c, inl_lt_inl_iff.1 ha, inl_lt_inl_iff.1 hb⟩
          /-
            🎉 no goals
          -/
          /-
            case intro.inr.intro
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : DenselyOrdered (Sum α β)
            a b : α
            h : LT.lt a b
            c : β
            ha : LT.lt (Sum.inl a) (Sum.inr c)
            hb : LT.lt (Sum.inr c) (Sum.inl b)
            ⊢ Exists fun a_1 => And (LT.lt a a_1) (LT.lt a_1 b)
          -/
        · exact (not_inl_lt_inr ha).elim⟩,
          /-
            🎉 no goals
          -/
      ⟨fun a b h => by
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : LT α
          inst✝ : LT β
          x✝ : DenselyOrdered (Sum α β)
          a b : β
          h : LT.lt a b
          ⊢ Exists fun a_1 => And (LT.lt a a_1) (LT.lt a_1 b)
        -/
        obtain ⟨c | c, ha, hb⟩ := @exists_between (α ⊕ β) _ _ _ _ (inr_lt_inr_iff.2 h)
          /-
            case intro.inl.intro
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : DenselyOrdered (Sum α β)
            a b : β
            h : LT.lt a b
            c : α
            ha : LT.lt (Sum.inr a) (Sum.inl c)
            hb : LT.lt (Sum.inl c) (Sum.inr b)
            ⊢ Exists fun a_1 => And (LT.lt a a_1) (LT.lt a_1 b)
          -/
        · exact (not_inl_lt_inr hb).elim
          /-
            🎉 no goals
          -/
          /-
            case intro.inr.intro
            α : Type u_1
            β : Type u_2
            inst✝¹ : LT α
            inst✝ : LT β
            x✝ : DenselyOrdered (Sum α β)
            a b : β
            h : LT.lt a b
            c : β
            ha : LT.lt (Sum.inr a) (Sum.inr c)
            hb : LT.lt (Sum.inr c) (Sum.inr b)
            ⊢ Exists fun a_1 => And (LT.lt a a_1) (LT.lt a_1 b)
          -/
        · exact ⟨c, inr_lt_inr_iff.1 ha, inr_lt_inr_iff.1 hb⟩⟩⟩,
          /-
            🎉 no goals
          -/
    fun h => @Sum.denselyOrdered _ _ _ _ h.1 h.2⟩


@[simp]
theorem swap_le_swap_iff [LE α] [LE β] {a b : α ⊕ β} : a.swap ≤ b.swap ↔ a ≤ b :=
  liftRel_swap_iff


@[simp]
theorem swap_lt_swap_iff [LT α] [LT β] {a b : α ⊕ β} : a.swap < b.swap ↔ a < b :=
  liftRel_swap_iff


/-- The linear sum of two orders -/
notation:30 α " ⊕ₗ " β:29 => _root_.Lex (α ⊕ β)

--TODO: Can we make `inlₗ`, `inrₗ` `local notation`?

/-- Lexicographical `Sum.inl`. Only used for pattern matching. -/
@[match_pattern]
abbrev _root_.Sum.inlₗ (x : α) : α ⊕ₗ β :=
  toLex (Sum.inl x)


/-- Lexicographical `Sum.inr`. Only used for pattern matching. -/
@[match_pattern]
abbrev _root_.Sum.inrₗ (x : β) : α ⊕ₗ β :=
  toLex (Sum.inr x)


/-- The linear/lexicographical `≤` on a sum. -/
protected instance LE [LE α] [LE β] : LE (α ⊕ₗ β) :=
  ⟨Lex (· ≤ ·) (· ≤ ·)⟩


/-- The linear/lexicographical `<` on a sum. -/
protected instance LT [LT α] [LT β] : LT (α ⊕ₗ β) :=
  ⟨Lex (· < ·) (· < ·)⟩


@[simp]
theorem toLex_le_toLex [LE α] [LE β] {a b : α ⊕ β} :
    toLex a ≤ toLex b ↔ Lex (· ≤ ·) (· ≤ ·) a b :=
  Iff.rfl


@[simp]
theorem toLex_lt_toLex [LT α] [LT β] {a b : α ⊕ β} :
    toLex a < toLex b ↔ Lex (· < ·) (· < ·) a b :=
  Iff.rfl


theorem le_def [LE α] [LE β] {a b : α ⊕ₗ β} : a ≤ b ↔ Lex (· ≤ ·) (· ≤ ·) (ofLex a) (ofLex b) :=
  Iff.rfl


theorem lt_def [LT α] [LT β] {a b : α ⊕ₗ β} : a < b ↔ Lex (· < ·) (· < ·) (ofLex a) (ofLex b) :=
  Iff.rfl


theorem inl_le_inl_iff [LE α] [LE β] {a b : α} : toLex (inl a : α ⊕ β) ≤ toLex (inl b) ↔ a ≤ b :=
  lex_inl_inl


theorem inr_le_inr_iff [LE α] [LE β] {a b : β} : toLex (inr a : α ⊕ β) ≤ toLex (inr b) ↔ a ≤ b :=
  lex_inr_inr


theorem inl_lt_inl_iff [LT α] [LT β] {a b : α} : toLex (inl a : α ⊕ β) < toLex (inl b) ↔ a < b :=
  lex_inl_inl


theorem inr_lt_inr_iff [LT α] [LT β] {a b : β} : toLex (inr a : α ⊕ₗ β) < toLex (inr b) ↔ a < b :=
  lex_inr_inr


theorem inl_le_inr [LE α] [LE β] (a : α) (b : β) : toLex (inl a) ≤ toLex (inr b) :=
  Lex.sep _ _


theorem inl_lt_inr [LT α] [LT β] (a : α) (b : β) : toLex (inl a) < toLex (inr b) :=
  Lex.sep _ _


theorem not_inr_le_inl [LE α] [LE β] {a : α} {b : β} : ¬toLex (inr b) ≤ toLex (inl a) :=
  lex_inr_inl


theorem not_inr_lt_inl [LT α] [LT β] {a : α} {b : β} : ¬toLex (inr b) < toLex (inl a) :=
  lex_inr_inl


instance preorder : Preorder (α ⊕ₗ β) :=
  { Lex.LE, Lex.LT with
    le_refl := refl_of (Lex (· ≤ ·) (· ≤ ·)),
    le_trans := fun _ _ _ => trans_of (Lex (· ≤ ·) (· ≤ ·)),
    lt_iff_le_not_le := fun a b => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        a b : _root_.Lex (Sum α β)
        ⊢ Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
      -/
      refine ⟨fun hab => ⟨hab.mono (fun _ _ => le_of_lt) fun _ _ => le_of_lt, ?_⟩, ?_⟩
        /-
          case refine_1
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝¹ : Preorder α
          inst✝ : Preorder β
          a b : _root_.Lex (Sum α β)
          hab : LT.lt a b
          ⊢ Not (LE.le b a)
        -/
      · rintro (⟨hba⟩ | ⟨hba⟩ | ⟨b, a⟩)
          /-
            case refine_1.inl
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            a₁✝ a₂✝ : α
            hba : LE.le a₁✝ a₂✝
            hab : LT.lt (Sum.inl a₂✝) (Sum.inl a₁✝)
            ⊢ False
          -/
        · exact hba.not_lt (inl_lt_inl_iff.1 hab)
          /-
            🎉 no goals
          -/
          /-
            case refine_1.inr
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            b₁✝ b₂✝ : β
            hba : LE.le b₁✝ b₂✝
            hab : LT.lt (Sum.inr b₂✝) (Sum.inr b₁✝)
            ⊢ False
          -/
        · exact hba.not_lt (inr_lt_inr_iff.1 hab)
          /-
            🎉 no goals
          -/
          /-
            case refine_1.sep
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            b : α
            a : β
            hab : LT.lt (Sum.inr a) (Sum.inl b)
            ⊢ False
          -/
        · exact not_inr_lt_inl hab
          /-
            🎉 no goals
          -/
        /-
          case refine_2
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝¹ : Preorder α
          inst✝ : Preorder β
          a b : _root_.Lex (Sum α β)
          ⊢ And (LE.le a b) (Not (LE.le b a)) → LT.lt a b
        -/
      · rintro ⟨⟨hab⟩ | ⟨hab⟩ | ⟨a, b⟩, hba⟩
          /-
            case refine_2.intro.inl
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            a₁✝ a₂✝ : α
            hab : LE.le a₁✝ a₂✝
            hba : Not (LE.le (Sum.inl a₂✝) (Sum.inl a₁✝))
            ⊢ LT.lt (Sum.inl a₁✝) (Sum.inl a₂✝)
          -/
        · exact Lex.inl (hab.lt_of_not_le fun h => hba <| Lex.inl h)
          /-
            🎉 no goals
          -/
          /-
            case refine_2.intro.inr
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            b₁✝ b₂✝ : β
            hab : LE.le b₁✝ b₂✝
            hba : Not (LE.le (Sum.inr b₂✝) (Sum.inr b₁✝))
            ⊢ LT.lt (Sum.inr b₁✝) (Sum.inr b₂✝)
          -/
        · exact Lex.inr (hab.lt_of_not_le fun h => hba <| Lex.inr h)
          /-
            🎉 no goals
          -/
          /-
            case refine_2.intro.sep
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            a : α
            b : β
            hba : Not (LE.le (Sum.inr b) (Sum.inl a))
            ⊢ LT.lt (Sum.inl a) (Sum.inr b)
          -/
        · exact Lex.sep _ _ }
          /-
            🎉 no goals
          -/


theorem toLex_mono : Monotone (@toLex (α ⊕ β)) := fun _ _ h => h.lex


theorem toLex_strictMono : StrictMono (@toLex (α ⊕ β)) := fun _ _ h => h.lex


theorem inl_mono : Monotone (toLex ∘ inl : α → α ⊕ₗ β) :=
  toLex_mono.comp Sum.inl_mono


theorem inr_mono : Monotone (toLex ∘ inr : β → α ⊕ₗ β) :=
  toLex_mono.comp Sum.inr_mono


theorem inl_strictMono : StrictMono (toLex ∘ inl : α → α ⊕ₗ β) :=
  toLex_strictMono.comp Sum.inl_strictMono


theorem inr_strictMono : StrictMono (toLex ∘ inr : β → α ⊕ₗ β) :=
  toLex_strictMono.comp Sum.inr_strictMono


instance partialOrder [PartialOrder α] [PartialOrder β] : PartialOrder (α ⊕ₗ β) :=
  { Lex.preorder with le_antisymm := fun _ _ => antisymm_of (Lex (· ≤ ·) (· ≤ ·)) }


instance linearOrder [LinearOrder α] [LinearOrder β] : LinearOrder (α ⊕ₗ β) :=
  { Lex.partialOrder with
    le_total := total_of (Lex (· ≤ ·) (· ≤ ·)),
    decidableLE := instDecidableRelSumLex,
    decidableEq := instDecidableEqSum }


/-- The lexicographical bottom of a sum is the bottom of the left component. -/
instance orderBot [LE α] [OrderBot α] [LE β] :
    OrderBot (α ⊕ₗ β) where
  bot := inl ⊥
  bot_le := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝² : LE α
      inst✝¹ : OrderBot α
      inst✝ : LE β
      ⊢ ∀ (a : _root_.Lex (Sum α β)), LE.le Bot.bot a
    -/
    rintro (a | b)
      /-
        case inl
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝² : LE α
        inst✝¹ : OrderBot α
        inst✝ : LE β
        a : α
        ⊢ LE.le Bot.bot (Sum.inl a)
      -/
    · exact Lex.inl bot_le
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝² : LE α
        inst✝¹ : OrderBot α
        inst✝ : LE β
        b : β
        ⊢ LE.le Bot.bot (Sum.inr b)
      -/
    · exact Lex.sep _ _
      /-
        🎉 no goals
      -/


@[simp]
theorem inl_bot [LE α] [OrderBot α] [LE β] : toLex (inl ⊥ : α ⊕ β) = ⊥ :=
  rfl


/-- The lexicographical top of a sum is the top of the right component. -/
instance orderTop [LE α] [LE β] [OrderTop β] :
    OrderTop (α ⊕ₗ β) where
  top := inr ⊤
  le_top := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝² : LE α
      inst✝¹ : LE β
      inst✝ : OrderTop β
      ⊢ ∀ (a : _root_.Lex (Sum α β)), LE.le a Top.top
    -/
    rintro (a | b)
      /-
        case inl
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝² : LE α
        inst✝¹ : LE β
        inst✝ : OrderTop β
        a : α
        ⊢ LE.le (Sum.inl a) Top.top
      -/
    · exact Lex.sep _ _
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝² : LE α
        inst✝¹ : LE β
        inst✝ : OrderTop β
        b : β
        ⊢ LE.le (Sum.inr b) Top.top
      -/
    · exact Lex.inr le_top
      /-
        🎉 no goals
      -/


@[simp]
theorem inr_top [LE α] [LE β] [OrderTop β] : toLex (inr ⊤ : α ⊕ β) = ⊤ :=
  rfl


instance boundedOrder [LE α] [LE β] [OrderBot α] [OrderTop β] : BoundedOrder (α ⊕ₗ β) :=
  { Lex.orderBot, Lex.orderTop with }


instance noMinOrder [LT α] [LT β] [NoMinOrder α] [NoMinOrder β] : NoMinOrder (α ⊕ₗ β) :=
  ⟨fun a =>
    match a with
    | inl a =>
      let ⟨b, h⟩ := exists_lt a
      ⟨toLex (inl b), inl_lt_inl_iff.2 h⟩
    | inr a =>
      let ⟨b, h⟩ := exists_lt a
      ⟨toLex (inr b), inr_lt_inr_iff.2 h⟩⟩


instance noMaxOrder [LT α] [LT β] [NoMaxOrder α] [NoMaxOrder β] : NoMaxOrder (α ⊕ₗ β) :=
  ⟨fun a =>
    match a with
    | inl a =>
      let ⟨b, h⟩ := exists_gt a
      ⟨toLex (inl b), inl_lt_inl_iff.2 h⟩
    | inr a =>
      let ⟨b, h⟩ := exists_gt a
      ⟨toLex (inr b), inr_lt_inr_iff.2 h⟩⟩


instance noMinOrder_of_nonempty [LT α] [LT β] [NoMinOrder α] [Nonempty α] : NoMinOrder (α ⊕ₗ β) :=
  ⟨fun a =>
    match a with
    | inl a =>
      let ⟨b, h⟩ := exists_lt a
      ⟨toLex (inl b), inl_lt_inl_iff.2 h⟩
    | inr _ => ⟨toLex (inl <| Classical.arbitrary α), inl_lt_inr _ _⟩⟩


instance noMaxOrder_of_nonempty [LT α] [LT β] [NoMaxOrder β] [Nonempty β] : NoMaxOrder (α ⊕ₗ β) :=
  ⟨fun a =>
    match a with
    | inl _ => ⟨toLex (inr <| Classical.arbitrary β), inl_lt_inr _ _⟩
    | inr a =>
      let ⟨b, h⟩ := exists_gt a
      ⟨toLex (inr b), inr_lt_inr_iff.2 h⟩⟩


instance denselyOrdered_of_noMaxOrder [LT α] [LT β] [DenselyOrdered α] [DenselyOrdered β]
    [NoMaxOrder α] : DenselyOrdered (α ⊕ₗ β) :=
  ⟨fun a b h =>
    match a, b, h with
    | inl _, inl _, Lex.inl h =>
      let ⟨c, ha, hb⟩ := exists_between h
      ⟨toLex (inl c), inl_lt_inl_iff.2 ha, inl_lt_inl_iff.2 hb⟩
    | inl a, inr _, Lex.sep _ _ =>
      let ⟨c, h⟩ := exists_gt a
      ⟨toLex (inl c), inl_lt_inl_iff.2 h, inl_lt_inr _ _⟩
    | inr _, inr _, Lex.inr h =>
      let ⟨c, ha, hb⟩ := exists_between h
      ⟨toLex (inr c), inr_lt_inr_iff.2 ha, inr_lt_inr_iff.2 hb⟩⟩


instance denselyOrdered_of_noMinOrder [LT α] [LT β] [DenselyOrdered α] [DenselyOrdered β]
    [NoMinOrder β] : DenselyOrdered (α ⊕ₗ β) :=
  ⟨fun a b h =>
    match a, b, h with
    | inl _, inl _, Lex.inl h =>
      let ⟨c, ha, hb⟩ := exists_between h
      ⟨toLex (inl c), inl_lt_inl_iff.2 ha, inl_lt_inl_iff.2 hb⟩
    | inl _, inr b, Lex.sep _ _ =>
      let ⟨c, h⟩ := exists_lt b
      ⟨toLex (inr c), inl_lt_inr _ _, inr_lt_inr_iff.2 h⟩
    | inr _, inr _, Lex.inr h =>
      let ⟨c, ha, hb⟩ := exists_between h
      ⟨toLex (inr c), inr_lt_inr_iff.2 ha, inr_lt_inr_iff.2 hb⟩⟩


/-- `Equiv.sumComm` promoted to an order isomorphism. -/
@[simps! apply]
def sumComm (α β : Type*) [LE α] [LE β] : α ⊕ β ≃o β ⊕ α :=
  { Equiv.sumComm α β with map_rel_iff' := swap_le_swap_iff }


@[simp]
theorem sumComm_symm (α β : Type*) [LE α] [LE β] :
    (OrderIso.sumComm α β).symm = OrderIso.sumComm β α :=
  rfl


/-- `Equiv.sumAssoc` promoted to an order isomorphism. -/
def sumAssoc (α β γ : Type*) [LE α] [LE β] [LE γ] : (α ⊕ β) ⊕ γ ≃o α ⊕ (β ⊕ γ) :=
  { Equiv.sumAssoc α β γ with
    map_rel_iff' := fun {a b} => by
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        γ✝ : Type u_3
        inst✝⁵ : LE α✝
        inst✝⁴ : LE β✝
        inst✝³ : LE γ✝
        a✝ : α✝
        b✝ : β✝
        c : γ✝
        α : Type u_4
        β : Type u_5
        γ : Type u_6
        inst✝² : LE α
        inst✝¹ : LE β
        inst✝ : LE γ
        a b : Sum (Sum α β) γ
        ⊢ Iff (LE.le (__src✝ a) (__src✝ b)) (LE.le a b)
      -/
      rcases a with ((_ | _) | _) <;> rcases b with ((_ | _) | _) <;>
      /-
        case inl.inl.inl.inl
        α✝ : Type u_1
        β✝ : Type u_2
        γ✝ : Type u_3
        inst✝⁵ : LE α✝
        inst✝⁴ : LE β✝
        inst✝³ : LE γ✝
        a : α✝
        b : β✝
        c : γ✝
        α : Type u_4
        β : Type u_5
        γ : Type u_6
        inst✝² : LE α
        inst✝¹ : LE β
        inst✝ : LE γ
        val✝¹ val✝ : α
        ⊢ Iff (LE.le (__src✝ (Sum.inl (Sum.inl val✝¹))) (__src✝ (Sum.inl (Sum.inl val✝ …
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
      simp [Equiv.sumAssoc] }
      /-
        🎉 no goals
      -/


@[simp]
theorem sumAssoc_apply_inl_inl : sumAssoc α β γ (inl (inl a)) = inl a :=
  rfl


@[simp]
theorem sumAssoc_apply_inl_inr : sumAssoc α β γ (inl (inr b)) = inr (inl b) :=
  rfl


@[simp]
theorem sumAssoc_apply_inr : sumAssoc α β γ (inr c) = inr (inr c) :=
  rfl


@[simp]
theorem sumAssoc_symm_apply_inl : (sumAssoc α β γ).symm (inl a) = inl (inl a) :=
  rfl


@[simp]
theorem sumAssoc_symm_apply_inr_inl : (sumAssoc α β γ).symm (inr (inl b)) = inl (inr b) :=
  rfl


@[simp]
theorem sumAssoc_symm_apply_inr_inr : (sumAssoc α β γ).symm (inr (inr c)) = inr c :=
  rfl


/-- `orderDual` is distributive over `⊕` up to an order isomorphism. -/
def sumDualDistrib (α β : Type*) [LE α] [LE β] : (α ⊕ β)ᵒᵈ ≃o αᵒᵈ ⊕ βᵒᵈ :=
  { Equiv.refl _ with
    map_rel_iff' := by
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        γ : Type u_3
        inst✝⁴ : LE α✝
        inst✝³ : LE β✝
        inst✝² : LE γ
        a : α✝
        b : β✝
        c : γ
        α : Type u_4
        β : Type u_5
        inst✝¹ : LE α
        inst✝ : LE β
        ⊢ ∀ {a b : OrderDual (Sum α β)}, Iff (LE.le (__src✝ a) (__src✝ b)) (LE.le a b)
      -/
      rintro (a | a) (b | b)
        /-
          case inl.inl
          α✝ : Type u_1
          β✝ : Type u_2
          γ : Type u_3
          inst✝⁴ : LE α✝
          inst✝³ : LE β✝
          inst✝² : LE γ
          a✝ : α✝
          b✝ : β✝
          c : γ
          α : Type u_4
          β : Type u_5
          inst✝¹ : LE α
          inst✝ : LE β
          a b : α
          ⊢ Iff (LE.le (__src✝ (Sum.inl a)) (__src✝ (Sum.inl b))) (LE.le (Sum.inl a) (Su …
        -/
      · change inl (toDual a) ≤ inl (toDual b) ↔ toDual (inl a) ≤ toDual (inl b)
        /-
          case inl.inl
          α✝ : Type u_1
          β✝ : Type u_2
          γ : Type u_3
          inst✝⁴ : LE α✝
          inst✝³ : LE β✝
          inst✝² : LE γ
          a✝ : α✝
          b✝ : β✝
          c : γ
          α : Type u_4
          β : Type u_5
          inst✝¹ : LE α
          inst✝ : LE β
          a b : α
          ⊢ Iff (LE.le (Sum.inl (OrderDual.toDual a)) (Sum.inl (OrderDual.toDual b))) (L …
        -/
        simp [toDual_le_toDual, inl_le_inl_iff]
        /-
          🎉 no goals
        -/
        /-
          case inl.inr
          α✝ : Type u_1
          β✝ : Type u_2
          γ : Type u_3
          inst✝⁴ : LE α✝
          inst✝³ : LE β✝
          inst✝² : LE γ
          a✝ : α✝
          b✝ : β✝
          c : γ
          α : Type u_4
          β : Type u_5
          inst✝¹ : LE α
          inst✝ : LE β
          a : α
          b : β
          ⊢ Iff (LE.le (__src✝ (Sum.inl a)) (__src✝ (Sum.inr b))) (LE.le (Sum.inl a) (Su …
        -/
      · exact iff_of_false (@not_inl_le_inr (OrderDual β) (OrderDual α) _ _ _ _) not_inr_le_inl
        /-
          🎉 no goals
        -/
        /-
          case inr.inl
          α✝ : Type u_1
          β✝ : Type u_2
          γ : Type u_3
          inst✝⁴ : LE α✝
          inst✝³ : LE β✝
          inst✝² : LE γ
          a✝ : α✝
          b✝ : β✝
          c : γ
          α : Type u_4
          β : Type u_5
          inst✝¹ : LE α
          inst✝ : LE β
          a : β
          b : α
          ⊢ Iff (LE.le (__src✝ (Sum.inr a)) (__src✝ (Sum.inl b))) (LE.le (Sum.inr a) (Su …
        -/
      · exact iff_of_false (@not_inr_le_inl (OrderDual α) (OrderDual β) _ _ _ _) not_inl_le_inr
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          α✝ : Type u_1
          β✝ : Type u_2
          γ : Type u_3
          inst✝⁴ : LE α✝
          inst✝³ : LE β✝
          inst✝² : LE γ
          a✝ : α✝
          b✝ : β✝
          c : γ
          α : Type u_4
          β : Type u_5
          inst✝¹ : LE α
          inst✝ : LE β
          a b : β
          ⊢ Iff (LE.le (__src✝ (Sum.inr a)) (__src✝ (Sum.inr b))) (LE.le (Sum.inr a) (Su …
        -/
      · change inr (toDual a) ≤ inr (toDual b) ↔ toDual (inr a) ≤ toDual (inr b)
        /-
          case inr.inr
          α✝ : Type u_1
          β✝ : Type u_2
          γ : Type u_3
          inst✝⁴ : LE α✝
          inst✝³ : LE β✝
          inst✝² : LE γ
          a✝ : α✝
          b✝ : β✝
          c : γ
          α : Type u_4
          β : Type u_5
          inst✝¹ : LE α
          inst✝ : LE β
          a b : β
          ⊢ Iff (LE.le (Sum.inr (OrderDual.toDual a)) (Sum.inr (OrderDual.toDual b))) (L …
        -/
        simp [toDual_le_toDual, inr_le_inr_iff] }
        /-
          🎉 no goals
        -/


@[simp]
theorem sumDualDistrib_inl : sumDualDistrib α β (toDual (inl a)) = inl (toDual a) :=
  rfl


@[simp]
theorem sumDualDistrib_inr : sumDualDistrib α β (toDual (inr b)) = inr (toDual b) :=
  rfl


@[simp]
theorem sumDualDistrib_symm_inl : (sumDualDistrib α β).symm (inl (toDual a)) = toDual (inl a) :=
  rfl


@[simp]
theorem sumDualDistrib_symm_inr : (sumDualDistrib α β).symm (inr (toDual b)) = toDual (inr b) :=
  rfl


/-- `Equiv.SumAssoc` promoted to an order isomorphism. -/
def sumLexAssoc (α β γ : Type*) [LE α] [LE β] [LE γ] : (α ⊕ₗ β) ⊕ₗ γ ≃o α ⊕ₗ β ⊕ₗ γ :=
  { Equiv.sumAssoc α β γ with
    map_rel_iff' := fun {a b} =>
      ⟨fun h =>
        match a, b, h with
        | inlₗ (inlₗ _), inlₗ (inlₗ _), Lex.inl h => Lex.inl <| Lex.inl h
        | inlₗ (inlₗ _), inlₗ (inrₗ _), Lex.sep _ _ => Lex.inl <| Lex.sep _ _
        | inlₗ (inlₗ _), inrₗ _, Lex.sep _ _ => Lex.sep _ _
        | inlₗ (inrₗ _), inlₗ (inrₗ _), Lex.inr (Lex.inl h) => Lex.inl <| Lex.inr h
        | inlₗ (inrₗ _), inrₗ _, Lex.inr (Lex.sep _ _) => Lex.sep _ _
        | inrₗ _, inrₗ _, Lex.inr (Lex.inr h) => Lex.inr h,
        fun h =>
        match a, b, h with
        | inlₗ (inlₗ _), inlₗ (inlₗ _), Lex.inl (Lex.inl h) => Lex.inl h
        | inlₗ (inlₗ _), inlₗ (inrₗ _), Lex.inl (Lex.sep _ _) => Lex.sep _ _
        | inlₗ (inlₗ _), inrₗ _, Lex.sep _ _ => Lex.sep _ _
        | inlₗ (inrₗ _), inlₗ (inrₗ _), Lex.inl (Lex.inr h) => Lex.inr <| Lex.inl h
        | inlₗ (inrₗ _), inrₗ _, Lex.sep _ _ => Lex.inr <| Lex.sep _ _
        | inrₗ _, inrₗ _, Lex.inr h => Lex.inr <| Lex.inr h⟩ }


@[simp]
theorem sumLexAssoc_apply_inl_inl :
    sumLexAssoc α β γ (toLex <| inl <| toLex <| inl a) = toLex (inl a) :=
  rfl


@[simp]
theorem sumLexAssoc_apply_inl_inr :
    sumLexAssoc α β γ (toLex <| inl <| toLex <| inr b) = toLex (inr <| toLex <| inl b) :=
  rfl


@[simp]
theorem sumLexAssoc_apply_inr :
    sumLexAssoc α β γ (toLex <| inr c) = toLex (inr <| toLex <| inr c) :=
  rfl


@[simp]
theorem sumLexAssoc_symm_apply_inl : (sumLexAssoc α β γ).symm (inl a) = inl (inl a) :=
  rfl


@[simp]
theorem sumLexAssoc_symm_apply_inr_inl : (sumLexAssoc α β γ).symm (inr (inl b)) = inl (inr b) :=
  rfl


@[simp]
theorem sumLexAssoc_symm_apply_inr_inr : (sumLexAssoc α β γ).symm (inr (inr c)) = inr c :=
  rfl


/-- `OrderDual` is antidistributive over `⊕ₗ` up to an order isomorphism. -/
def sumLexDualAntidistrib (α β : Type*) [LE α] [LE β] : (α ⊕ₗ β)ᵒᵈ ≃o βᵒᵈ ⊕ₗ αᵒᵈ :=
  { Equiv.sumComm α β with
    map_rel_iff' := fun {a b} => by
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        γ : Type u_3
        inst✝⁴ : LE α✝
        inst✝³ : LE β✝
        inst✝² : LE γ
        a✝ : α✝
        b✝ : β✝
        c : γ
        α : Type u_4
        β : Type u_5
        inst✝¹ : LE α
        inst✝ : LE β
        a b : OrderDual (Lex (Sum α β))
        ⊢ Iff (LE.le (__src✝ a) (__src✝ b)) (LE.le a b)
      -/
      rcases a with (a | a) <;> rcases b with (b | b)
        /-
          case inl.inl
          α✝ : Type u_1
          β✝ : Type u_2
          γ : Type u_3
          inst✝⁴ : LE α✝
          inst✝³ : LE β✝
          inst✝² : LE γ
          a✝ : α✝
          b✝ : β✝
          c : γ
          α : Type u_4
          β : Type u_5
          inst✝¹ : LE α
          inst✝ : LE β
          a b : α
          ⊢ Iff (LE.le (__src✝ (Sum.inl a)) (__src✝ (Sum.inl b))) (LE.le (Sum.inl a) (Su …
        -/
      · simp
        change
          toLex (inr <| toDual a) ≤ toLex (inr <| toDual b) ↔
            toDual (toLex <| inl a) ≤ toDual (toLex <| inl b)
        /-
          case inl.inl
          α✝ : Type u_1
          β✝ : Type u_2
          γ : Type u_3
          inst✝⁴ : LE α✝
          inst✝³ : LE β✝
          inst✝² : LE γ
          a✝ : α✝
          b✝ : β✝
          c : γ
          α : Type u_4
          β : Type u_5
          inst✝¹ : LE α
          inst✝ : LE β
          a b : α
          ⊢ Iff (LE.le (toLex (Sum.inr (OrderDual.toDual a))) (toLex (Sum.inr (OrderDual …
        -/
        simp [toDual_le_toDual, Lex.inl_le_inl_iff, Lex.inr_le_inr_iff]
        /-
          🎉 no goals
        -/
      · exact iff_of_false (@Lex.not_inr_le_inl (OrderDual β) (OrderDual α) _ _ _ _)
          Lex.not_inr_le_inl
      · exact iff_of_true (@Lex.inl_le_inr (OrderDual β) (OrderDual α) _ _ _ _)
          (Lex.inl_le_inr _ _)
      · change
          toLex (inl <| toDual a) ≤ toLex (inl <| toDual b) ↔
            toDual (toLex <| inr a) ≤ toDual (toLex <| inr b)
        /-
          case inr.inr
          α✝ : Type u_1
          β✝ : Type u_2
          γ : Type u_3
          inst✝⁴ : LE α✝
          inst✝³ : LE β✝
          inst✝² : LE γ
          a✝ : α✝
          b✝ : β✝
          c : γ
          α : Type u_4
          β : Type u_5
          inst✝¹ : LE α
          inst✝ : LE β
          a b : β
          ⊢ Iff (LE.le (toLex (Sum.inl (OrderDual.toDual a))) (toLex (Sum.inl (OrderDual …
        -/
        simp [toDual_le_toDual, Lex.inl_le_inl_iff, Lex.inr_le_inr_iff] }
        /-
          🎉 no goals
        -/


@[simp]
theorem sumLexDualAntidistrib_inl :
    sumLexDualAntidistrib α β (toDual (inl a)) = inr (toDual a) :=
  rfl


@[simp]
theorem sumLexDualAntidistrib_inr :
    sumLexDualAntidistrib α β (toDual (inr b)) = inl (toDual b) :=
  rfl


@[simp]
theorem sumLexDualAntidistrib_symm_inl :
    (sumLexDualAntidistrib α β).symm (inl (toDual b)) = toDual (inr b) :=
  rfl


@[simp]
theorem sumLexDualAntidistrib_symm_inr :
    (sumLexDualAntidistrib α β).symm (inr (toDual a)) = toDual (inl a) :=
  rfl


/-- `WithBot α` is order-isomorphic to `PUnit ⊕ₗ α`, by sending `⊥` to `Unit` and `↑a` to
`a`. -/
def orderIsoPUnitSumLex : WithBot α ≃o PUnit ⊕ₗ α :=
  ⟨(Equiv.optionEquivSumPUnit α).trans <| (Equiv.sumComm _ _).trans toLex, fun {a b} => by
    simp only [Equiv.optionEquivSumPUnit, Option.elim, Equiv.trans_apply, Equiv.coe_fn_mk,
      Equiv.sumComm_apply, swap, Lex.toLex_le_toLex, le_refl]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : LE α
      a b : WithBot α
      ⊢ Iff (Sum.Lex (fun x1 x2 => True) (fun x1 x2 => LE.le x1 x2) (Sum.elim Sum.in …
    -/
    cases a <;> cases b
      /-
        case bot.bot
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : LE α
        ⊢ Iff (Sum.Lex (fun x1 x2 => True) (fun x1 x2 => LE.le x1 x2) (Sum.elim Sum.in …
      -/
    · simp only [elim_inr, lex_inl_inl, bot_le, le_rfl]
      /-
        🎉 no goals
      -/
      /-
        case bot.coe
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : LE α
        a✝ : α
        ⊢ Iff (Sum.Lex (fun x1 x2 => True) (fun x1 x2 => LE.le x1 x2) (Sum.elim Sum.in …
      -/
    · simp only [elim_inr, elim_inl, Lex.sep, bot_le]
      /-
        🎉 no goals
      -/
      /-
        case coe.bot
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : LE α
        a✝ : α
        ⊢ Iff (Sum.Lex (fun x1 x2 => True) (fun x1 x2 => LE.le x1 x2) (Sum.elim Sum.in …
      -/
    · simp only [elim_inl, elim_inr, lex_inr_inl, false_iff]
      /-
        case coe.bot
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : LE α
        a✝ : α
        ⊢ Not (LE.le (↑a✝) Bot.bot)
      -/
      exact not_coe_le_bot _
      /-
        🎉 no goals
      -/
      /-
        case coe.coe
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : LE α
        a✝¹ a✝ : α
        ⊢ Iff (Sum.Lex (fun x1 x2 => True) (fun x1 x2 => LE.le x1 x2) (Sum.elim Sum.in …
      -/
    · simp only [elim_inl, lex_inr_inr, coe_le_coe]
      /-
        🎉 no goals
      -/
  ⟩




@[simp]
theorem orderIsoPUnitSumLex_bot : @orderIsoPUnitSumLex α _ ⊥ = toLex (inl PUnit.unit) :=
  rfl


@[simp]
theorem orderIsoPUnitSumLex_toLex (a : α) : orderIsoPUnitSumLex ↑a = toLex (inr a) :=
  rfl


@[simp]
theorem orderIsoPUnitSumLex_symm_inl (x : PUnit) :
    (@orderIsoPUnitSumLex α _).symm (toLex <| inl x) = ⊥ :=
  rfl


@[simp]
theorem orderIsoPUnitSumLex_symm_inr (a : α) : orderIsoPUnitSumLex.symm (toLex <| inr a) = a :=
  rfl


/-- `WithTop α` is order-isomorphic to `α ⊕ₗ PUnit`, by sending `⊤` to `Unit` and `↑a` to
`a`. -/
def orderIsoSumLexPUnit : WithTop α ≃o α ⊕ₗ PUnit :=
  ⟨(Equiv.optionEquivSumPUnit α).trans toLex, fun {a b} => by
    simp only [Equiv.optionEquivSumPUnit, Option.elim, Equiv.trans_apply, Equiv.coe_fn_mk,
      Lex.toLex_le_toLex, le_refl, lex_inr_inr, le_top]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : LE α
      a b : WithTop α
      ⊢ Iff (Sum.Lex (fun x1 x2 => LE.le x1 x2) (fun x1 x2 => True) (Option.elim.mat …
    -/
    cases a <;> cases b
      /-
        case top.top
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : LE α
        ⊢ Iff (Sum.Lex (fun x1 x2 => LE.le x1 x2) (fun x1 x2 => True) (Option.elim.mat …
      -/
    · simp only [lex_inr_inr, le_top]
      /-
        🎉 no goals
      -/
      /-
        case top.coe
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : LE α
        a✝ : α
        ⊢ Iff (Sum.Lex (fun x1 x2 => LE.le x1 x2) (fun x1 x2 => True) (Option.elim.mat …
      -/
    · simp only [lex_inr_inl, false_iff]
      /-
        case top.coe
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : LE α
        a✝ : α
        ⊢ Not (LE.le Top.top ↑a✝)
      -/
      exact not_top_le_coe _
      /-
        🎉 no goals
      -/
      /-
        case coe.top
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : LE α
        a✝ : α
        ⊢ Iff (Sum.Lex (fun x1 x2 => LE.le x1 x2) (fun x1 x2 => True) (Option.elim.mat …
      -/
    · simp only [Lex.sep, le_top]
      /-
        🎉 no goals
      -/
      /-
        case coe.coe
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : LE α
        a✝¹ a✝ : α
        ⊢ Iff (Sum.Lex (fun x1 x2 => LE.le x1 x2) (fun x1 x2 => True) (Option.elim.mat …
      -/
    · simp only [lex_inl_inl, coe_le_coe]⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem orderIsoSumLexPUnit_top : @orderIsoSumLexPUnit α _ ⊤ = toLex (inr PUnit.unit) :=
  rfl


@[simp]
theorem orderIsoSumLexPUnit_toLex (a : α) : orderIsoSumLexPUnit ↑a = toLex (inl a) :=
  rfl


@[simp]
theorem orderIsoSumLexPUnit_symm_inr (x : PUnit) :
    (@orderIsoSumLexPUnit α _).symm (toLex <| inr x) = ⊤ :=
  rfl


@[simp]
theorem orderIsoSumLexPUnit_symm_inl (a : α) : orderIsoSumLexPUnit.symm (toLex <| inl a) = a :=
  rfl


