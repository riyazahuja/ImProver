@[inherit_doc] notation:35 α " ×ₗ " β:34 => Lex (Prod α β)


/-- Dictionary / lexicographic ordering on pairs. -/
instance instLE (α β : Type*) [LT α] [LE β] : LE (α ×ₗ β) where le := Prod.Lex (· < ·) (· ≤ ·)


instance instLT (α β : Type*) [LT α] [LT β] : LT (α ×ₗ β) where lt := Prod.Lex (· < ·) (· < ·)


theorem le_iff [LT α] [LE β] (a b : α × β) :
    toLex a ≤ toLex b ↔ a.1 < b.1 ∨ a.1 = b.1 ∧ a.2 ≤ b.2 :=
  Prod.lex_def


theorem lt_iff [LT α] [LT β] (a b : α × β) :
    toLex a < toLex b ↔ a.1 < b.1 ∨ a.1 = b.1 ∧ a.2 < b.2 :=
  Prod.lex_def


instance [LT α] [LT β] [WellFoundedLT α] [WellFoundedLT β] : WellFoundedLT (α ×ₗ β) :=
  ⟨WellFounded.prod_lex wellFounded_lt wellFounded_lt⟩


instance [LT α] [LT β] [WellFoundedLT α] [WellFoundedLT β] : WellFoundedRelation (α ×ₗ β) :=
  ⟨(· < ·), wellFounded_lt⟩


/-- Dictionary / lexicographic preorder for pairs. -/
instance preorder (α β : Type*) [Preorder α] [Preorder β] : Preorder (α ×ₗ β) :=
  { Prod.Lex.instLE α β, Prod.Lex.instLT α β with
    le_refl := refl_of <| Prod.Lex _ _,
    le_trans := fun _ _ _ => trans_of <| Prod.Lex _ _,
    lt_iff_le_not_le := fun x₁ x₂ =>
      match x₁, x₂ with
      | (a₁, b₁), (a₂, b₂) => by
        /-
          α✝ : Type u_1
          β✝ : Type u_2
          α : Type u_3
          β : Type u_4
          inst✝¹ : Preorder α
          inst✝ : Preorder β
          x₁ x₂ : Lex (Prod α β)
          a₁ : α
          b₁ : β
          a₂ : α
          b₂ : β
          ⊢ Iff (LT.lt { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }) (And (LE.le {  …
        -/
        constructor
          /-
            case mp
            α✝ : Type u_1
            β✝ : Type u_2
            α : Type u_3
            β : Type u_4
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            x₁ x₂ : Lex (Prod α β)
            a₁ : α
            b₁ : β
            a₂ : α
            b₂ : β
            ⊢ LT.lt { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ } → And (LE.le { fst : …
          -/
        · rintro (⟨_, _, hlt⟩ | ⟨_, hlt⟩)
            /-
              case mp.left
              α✝ : Type u_1
              β✝ : Type u_2
              α : Type u_3
              β : Type u_4
              inst✝¹ : Preorder α
              inst✝ : Preorder β
              x₁ x₂ : Lex (Prod α β)
              a₁ : α
              b₁ : β
              a₂ : α
              b₂ : β
              hlt : LT.lt a₁ a₂
              ⊢ And (LE.le { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }) (Not (LE.le {  …
            -/
          · constructor
              /-
                case mp.left.left
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ : β
                a₂ : α
                b₂ : β
                hlt : LT.lt a₁ a₂
                ⊢ LE.le { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
              -/
            · exact left _ _ hlt
              /-
                🎉 no goals
              -/
              /-
                case mp.left.right
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ : β
                a₂ : α
                b₂ : β
                hlt : LT.lt a₁ a₂
                ⊢ Not (LE.le { fst := a₂, snd := b₂ } { fst := a₁, snd := b₁ })
              -/
            · rintro ⟨⟩
                /-
                  case mp.left.right.left
                  α✝ : Type u_1
                  β✝ : Type u_2
                  α : Type u_3
                  β : Type u_4
                  inst✝¹ : Preorder α
                  inst✝ : Preorder β
                  x₁ x₂ : Lex (Prod α β)
                  a₁ : α
                  b₁ : β
                  a₂ : α
                  b₂ : β
                  hlt : LT.lt a₁ a₂
                  h✝ : LT.lt a₂ a₁
                  ⊢ False
                -/
              · apply lt_asymm hlt; assumption
                                    /-
                                      🎉 no goals
                                    -/
                /-
                  case mp.left.right.right
                  α✝ : Type u_1
                  β✝ : Type u_2
                  α : Type u_3
                  β : Type u_4
                  inst✝¹ : Preorder α
                  inst✝ : Preorder β
                  x₁ x₂ : Lex (Prod α β)
                  a₁ : α
                  b₁ b₂ : β
                  hlt : LT.lt a₁ a₁
                  h✝ : LE.le b₂ b₁
                  ⊢ False
                -/
              · exact lt_irrefl _ hlt
                /-
                  🎉 no goals
                -/
            /-
              case mp.right
              α✝ : Type u_1
              β✝ : Type u_2
              α : Type u_3
              β : Type u_4
              inst✝¹ : Preorder α
              inst✝ : Preorder β
              x₁ x₂ : Lex (Prod α β)
              a₁ : α
              b₁ b₂ : β
              hlt : LT.lt b₁ b₂
              ⊢ And (LE.le { fst := a₁, snd := b₁ } { fst := a₁, snd := b₂ }) (Not (LE.le {  …
            -/
          · constructor
              /-
                case mp.right.left
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ b₂ : β
                hlt : LT.lt b₁ b₂
                ⊢ LE.le { fst := a₁, snd := b₁ } { fst := a₁, snd := b₂ }
              -/
            · right
              /-
                case mp.right.left.h
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ b₂ : β
                hlt : LT.lt b₁ b₂
                ⊢ LE.le b₁ b₂
              -/
              rw [lt_iff_le_not_le] at hlt
              /-
                case mp.right.left.h
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ b₂ : β
                hlt : And (LE.le b₁ b₂) (Not (LE.le b₂ b₁))
                ⊢ LE.le b₁ b₂
              -/
              exact hlt.1
              /-
                🎉 no goals
              -/
              /-
                case mp.right.right
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ b₂ : β
                hlt : LT.lt b₁ b₂
                ⊢ Not (LE.le { fst := a₁, snd := b₂ } { fst := a₁, snd := b₁ })
              -/
            · rintro ⟨⟩
                /-
                  case mp.right.right.left
                  α✝ : Type u_1
                  β✝ : Type u_2
                  α : Type u_3
                  β : Type u_4
                  inst✝¹ : Preorder α
                  inst✝ : Preorder β
                  x₁ x₂ : Lex (Prod α β)
                  a₁ : α
                  b₁ b₂ : β
                  hlt : LT.lt b₁ b₂
                  h✝ : LT.lt a₁ a₁
                  ⊢ False
                -/
              · apply lt_irrefl a₁
                /-
                  case mp.right.right.left
                  α✝ : Type u_1
                  β✝ : Type u_2
                  α : Type u_3
                  β : Type u_4
                  inst✝¹ : Preorder α
                  inst✝ : Preorder β
                  x₁ x₂ : Lex (Prod α β)
                  a₁ : α
                  b₁ b₂ : β
                  hlt : LT.lt b₁ b₂
                  h✝ : LT.lt a₁ a₁
                  ⊢ LT.lt a₁ a₁
                -/
                assumption
                /-
                  🎉 no goals
                -/
                /-
                  case mp.right.right.right
                  α✝ : Type u_1
                  β✝ : Type u_2
                  α : Type u_3
                  β : Type u_4
                  inst✝¹ : Preorder α
                  inst✝ : Preorder β
                  x₁ x₂ : Lex (Prod α β)
                  a₁ : α
                  b₁ b₂ : β
                  hlt : LT.lt b₁ b₂
                  h✝ : LE.le b₂ b₁
                  ⊢ False
                -/
              · rw [lt_iff_le_not_le] at hlt
                /-
                  case mp.right.right.right
                  α✝ : Type u_1
                  β✝ : Type u_2
                  α : Type u_3
                  β : Type u_4
                  inst✝¹ : Preorder α
                  inst✝ : Preorder β
                  x₁ x₂ : Lex (Prod α β)
                  a₁ : α
                  b₁ b₂ : β
                  hlt : And (LE.le b₁ b₂) (Not (LE.le b₂ b₁))
                  h✝ : LE.le b₂ b₁
                  ⊢ False
                -/
                apply hlt.2
                /-
                  case mp.right.right.right
                  α✝ : Type u_1
                  β✝ : Type u_2
                  α : Type u_3
                  β : Type u_4
                  inst✝¹ : Preorder α
                  inst✝ : Preorder β
                  x₁ x₂ : Lex (Prod α β)
                  a₁ : α
                  b₁ b₂ : β
                  hlt : And (LE.le b₁ b₂) (Not (LE.le b₂ b₁))
                  h✝ : LE.le b₂ b₁
                  ⊢ LE.le b₂ b₁
                -/
                assumption
                /-
                  🎉 no goals
                -/
          /-
            case mpr
            α✝ : Type u_1
            β✝ : Type u_2
            α : Type u_3
            β : Type u_4
            inst✝¹ : Preorder α
            inst✝ : Preorder β
            x₁ x₂ : Lex (Prod α β)
            a₁ : α
            b₁ : β
            a₂ : α
            b₂ : β
            ⊢ And (LE.le { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }) (Not (LE.le {  …
          -/
        · rintro ⟨⟨⟩, h₂r⟩
            /-
              case mpr.intro.left
              α✝ : Type u_1
              β✝ : Type u_2
              α : Type u_3
              β : Type u_4
              inst✝¹ : Preorder α
              inst✝ : Preorder β
              x₁ x₂ : Lex (Prod α β)
              a₁ : α
              b₁ : β
              a₂ : α
              b₂ : β
              h₂r : Not (LE.le { fst := a₂, snd := b₂ } { fst := a₁, snd := b₁ })
              h✝ : LT.lt a₁ a₂
              ⊢ LT.lt { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
            -/
          · left
            /-
              case mpr.intro.left.h
              α✝ : Type u_1
              β✝ : Type u_2
              α : Type u_3
              β : Type u_4
              inst✝¹ : Preorder α
              inst✝ : Preorder β
              x₁ x₂ : Lex (Prod α β)
              a₁ : α
              b₁ : β
              a₂ : α
              b₂ : β
              h₂r : Not (LE.le { fst := a₂, snd := b₂ } { fst := a₁, snd := b₁ })
              h✝ : LT.lt a₁ a₂
              ⊢ LT.lt a₁ a₂
            -/
            assumption
            /-
              🎉 no goals
            -/
            /-
              case mpr.intro.right
              α✝ : Type u_1
              β✝ : Type u_2
              α : Type u_3
              β : Type u_4
              inst✝¹ : Preorder α
              inst✝ : Preorder β
              x₁ x₂ : Lex (Prod α β)
              a₁ : α
              b₁ b₂ : β
              h₂r : Not (LE.le { fst := a₁, snd := b₂ } { fst := a₁, snd := b₁ })
              h✝ : LE.le b₁ b₂
              ⊢ LT.lt { fst := a₁, snd := b₁ } { fst := a₁, snd := b₂ }
            -/
          · right
            /-
              case mpr.intro.right.h
              α✝ : Type u_1
              β✝ : Type u_2
              α : Type u_3
              β : Type u_4
              inst✝¹ : Preorder α
              inst✝ : Preorder β
              x₁ x₂ : Lex (Prod α β)
              a₁ : α
              b₁ b₂ : β
              h₂r : Not (LE.le { fst := a₁, snd := b₂ } { fst := a₁, snd := b₁ })
              h✝ : LE.le b₁ b₂
              ⊢ LT.lt b₁ b₂
            -/
            rw [lt_iff_le_not_le]
            /-
              case mpr.intro.right.h
              α✝ : Type u_1
              β✝ : Type u_2
              α : Type u_3
              β : Type u_4
              inst✝¹ : Preorder α
              inst✝ : Preorder β
              x₁ x₂ : Lex (Prod α β)
              a₁ : α
              b₁ b₂ : β
              h₂r : Not (LE.le { fst := a₁, snd := b₂ } { fst := a₁, snd := b₁ })
              h✝ : LE.le b₁ b₂
              ⊢ And (LE.le b₁ b₂) (Not (LE.le b₂ b₁))
            -/
            constructor
              /-
                case mpr.intro.right.h.left
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ b₂ : β
                h₂r : Not (LE.le { fst := a₁, snd := b₂ } { fst := a₁, snd := b₁ })
                h✝ : LE.le b₁ b₂
                ⊢ LE.le b₁ b₂
              -/
            · assumption
              /-
                🎉 no goals
              -/
              /-
                case mpr.intro.right.h.right
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ b₂ : β
                h₂r : Not (LE.le { fst := a₁, snd := b₂ } { fst := a₁, snd := b₁ })
                h✝ : LE.le b₁ b₂
                ⊢ Not (LE.le b₂ b₁)
              -/
            · intro h
              /-
                case mpr.intro.right.h.right
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ b₂ : β
                h₂r : Not (LE.le { fst := a₁, snd := b₂ } { fst := a₁, snd := b₁ })
                h✝ : LE.le b₁ b₂
                h : LE.le b₂ b₁
                ⊢ False
              -/
              apply h₂r
              /-
                case mpr.intro.right.h.right
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ b₂ : β
                h₂r : Not (LE.le { fst := a₁, snd := b₂ } { fst := a₁, snd := b₁ })
                h✝ : LE.le b₁ b₂
                h : LE.le b₂ b₁
                ⊢ LE.le { fst := a₁, snd := b₂ } { fst := a₁, snd := b₁ }
              -/
              right
              /-
                case mpr.intro.right.h.right.h
                α✝ : Type u_1
                β✝ : Type u_2
                α : Type u_3
                β : Type u_4
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                x₁ x₂ : Lex (Prod α β)
                a₁ : α
                b₁ b₂ : β
                h₂r : Not (LE.le { fst := a₁, snd := b₂ } { fst := a₁, snd := b₁ })
                h✝ : LE.le b₁ b₂
                h : LE.le b₂ b₁
                ⊢ LE.le b₂ b₁
              -/
              exact h }
              /-
                🎉 no goals
              -/


theorem monotone_fst [Preorder α] [LE β] (t c : α ×ₗ β) (h : t ≤ c) :
    (ofLex t).1 ≤ (ofLex c).1 := by
  cases (Prod.Lex.le_iff t c).mp h with
  | inl h' => exact h'.le
  | inr h' => exact h'.1.le


theorem toLex_mono : Monotone (toLex : α × β → α ×ₗ β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    ⊢ Monotone ⇑toLex
  -/
  rintro ⟨a₁, b₁⟩ ⟨a₂, b₂⟩ ⟨ha, hb⟩
  /-
    case mk.mk.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    a₁ : α
    b₁ : β
    a₂ : α
    b₂ : β
    ha : LE.le { fst := a₁, snd := b₁ }.1 { fst := a₂, snd := b₂ }.1
    hb : LE.le { fst := a₁, snd := b₁ }.2 { fst := a₂, snd := b₂ }.2
    ⊢ LE.le (toLex { fst := a₁, snd := b₁ }) (toLex { fst := a₂, snd := b₂ })
  -/
  obtain rfl | ha : a₁ = a₂ ∨ _ := ha.eq_or_lt
    /-
      case mk.mk.intro.inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      a₁ : α
      b₁ b₂ : β
      ha : LE.le { fst := a₁, snd := b₁ }.1 { fst := a₁, snd := b₂ }.1
      hb : LE.le { fst := a₁, snd := b₁ }.2 { fst := a₁, snd := b₂ }.2
      ⊢ LE.le (toLex { fst := a₁, snd := b₁ }) (toLex { fst := a₁, snd := b₂ })
    -/
  · exact right _ hb
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.intro.inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      a₁ : α
      b₁ : β
      a₂ : α
      b₂ : β
      ha✝ : LE.le { fst := a₁, snd := b₁ }.1 { fst := a₂, snd := b₂ }.1
      hb : LE.le { fst := a₁, snd := b₁ }.2 { fst := a₂, snd := b₂ }.2
      ha : LT.lt { fst := a₁, snd := b₁ }.1 { fst := a₂, snd := b₂ }.1
      ⊢ LE.le (toLex { fst := a₁, snd := b₁ }) (toLex { fst := a₂, snd := b₂ })
    -/
  · exact left _ _ ha
    /-
      🎉 no goals
    -/


theorem toLex_strictMono : StrictMono (toLex : α × β → α ×ₗ β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    ⊢ StrictMono ⇑toLex
  -/
  rintro ⟨a₁, b₁⟩ ⟨a₂, b₂⟩ h
  /-
    case mk.mk
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    a₁ : α
    b₁ : β
    a₂ : α
    b₂ : β
    h : LT.lt { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
    ⊢ LT.lt (toLex { fst := a₁, snd := b₁ }) (toLex { fst := a₂, snd := b₂ })
  -/
  obtain rfl | ha : a₁ = a₂ ∨ _ := h.le.1.eq_or_lt
    /-
      case mk.mk.inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      a₁ : α
      b₁ b₂ : β
      h : LT.lt { fst := a₁, snd := b₁ } { fst := a₁, snd := b₂ }
      ⊢ LT.lt (toLex { fst := a₁, snd := b₁ }) (toLex { fst := a₁, snd := b₂ })
    -/
  · exact right _ (Prod.mk_lt_mk_iff_right.1 h)
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      a₁ : α
      b₁ : β
      a₂ : α
      b₂ : β
      h : LT.lt { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
      ha : LT.lt { fst := a₁, snd := b₁ }.1 { fst := a₂, snd := b₂ }.1
      ⊢ LT.lt (toLex { fst := a₁, snd := b₁ }) (toLex { fst := a₂, snd := b₂ })
    -/
  · exact left _ _ ha
    /-
      🎉 no goals
    -/


/-- Dictionary / lexicographic partial order for pairs. -/
instance partialOrder (α β : Type*) [PartialOrder α] [PartialOrder β] : PartialOrder (α ×ₗ β) where
  le_antisymm _ _ := by
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      x✝¹ x✝ : Lex (Prod α β)
      ⊢ LE.le x✝¹ x✝ → LE.le x✝ x✝¹ → Eq x✝¹ x✝
    -/
    haveI : IsStrictOrder α (· < ·) := { irrefl := lt_irrefl, trans := fun _ _ _ => lt_trans }
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      x✝¹ x✝ : Lex (Prod α β)
      this : IsStrictOrder α fun x1 x2 => LT.lt x1 x2
      ⊢ LE.le x✝¹ x✝ → LE.le x✝ x✝¹ → Eq x✝¹ x✝
    -/
    haveI : IsAntisymm β (· ≤ ·) := ⟨fun _ _ => le_antisymm⟩
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      x✝¹ x✝ : Lex (Prod α β)
      this✝ : IsStrictOrder α fun x1 x2 => LT.lt x1 x2
      this : IsAntisymm β fun x1 x2 => LE.le x1 x2
      ⊢ LE.le x✝¹ x✝ → LE.le x✝ x✝¹ → Eq x✝¹ x✝
    -/
    exact antisymm (r := Prod.Lex _ _)
    /-
      🎉 no goals
    -/


instance instOrdLexProd [Ord α] [Ord β] : Ord (α ×ₗ β) := lexOrd


theorem compare_def [Ord α] [Ord β] : @compare (α ×ₗ β) _ =
    compareLex (compareOn fun x => (ofLex x).1) (compareOn fun x => (ofLex x).2) := rfl


theorem _root_.lexOrd_eq [Ord α] [Ord β] : @lexOrd α β _ _ = instOrdLexProd := rfl


theorem _root_.Ord.lex_eq [oα : Ord α] [oβ : Ord β] : Ord.lex oα oβ = instOrdLexProd := rfl


instance [Ord α] [Ord β] [OrientedOrd α] [OrientedOrd β] : OrientedOrd (α ×ₗ β) :=
  inferInstanceAs (OrientedCmp (compareLex _ _))


instance [Ord α] [Ord β] [TransOrd α] [TransOrd β] : TransOrd (α ×ₗ β) :=
  inferInstanceAs (TransCmp (compareLex _ _))


/-- Dictionary / lexicographic linear order for pairs. -/
instance linearOrder (α β : Type*) [LinearOrder α] [LinearOrder β] : LinearOrder (α ×ₗ β) :=
  { Prod.Lex.partialOrder α β with
    le_total := total_of (Prod.Lex _ _)
    decidableLE := Prod.Lex.decidable _ _
    decidableLT := Prod.Lex.decidable _ _
    decidableEq := instDecidableEqLex _
    compare_eq_compareOfLessAndEq := fun a b => by
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        a b : Lex (Prod α β)
        ⊢ Eq (Ord.compare a b) (compareOfLessAndEq a b)
      -/
      have : DecidableRel (· < · : α ×ₗ β → α ×ₗ β → Prop) := Prod.Lex.decidable _ _
      have : BEqOrd (α ×ₗ β) := ⟨by
        simp [compare_def, compareLex, compareOn, Ordering.then_eq_eq, compare_eq_iff_eq]⟩
      have : LTOrd (α ×ₗ β) := ⟨by
        simp [compare_def, compareLex, compareOn, Ordering.then_eq_lt, lt_iff,
          compare_lt_iff_lt, compare_eq_iff_eq]⟩
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        a b : Lex (Prod α β)
        this✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
        this✝ : Batteries.BEqOrd (Lex (Prod α β))
        this : Batteries.LTOrd (Lex (Prod α β))
        ⊢ Eq (Ord.compare a b) (compareOfLessAndEq a b)
      -/
      convert LTCmp.eq_compareOfLessAndEq (cmp := compare) a b }
      /-
        🎉 no goals
      -/


instance orderBot [PartialOrder α] [Preorder β] [OrderBot α] [OrderBot β] : OrderBot (α ×ₗ β) where
  bot := toLex ⊥
  bot_le _ := toLex_mono bot_le


instance orderTop [PartialOrder α] [Preorder β] [OrderTop α] [OrderTop β] : OrderTop (α ×ₗ β) where
  top := toLex ⊤
  le_top _ := toLex_mono le_top


instance boundedOrder [PartialOrder α] [Preorder β] [BoundedOrder α] [BoundedOrder β] :
    BoundedOrder (α ×ₗ β) :=
  { Lex.orderBot, Lex.orderTop with }


instance [Preorder α] [Preorder β] [DenselyOrdered α] [DenselyOrdered β] :
    DenselyOrdered (α ×ₗ β) where
  dense := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : DenselyOrdered α
      inst✝ : DenselyOrdered β
      ⊢ ∀ (a₁ a₂ : Lex (Prod α β)), LT.lt a₁ a₂ → Exists fun a => And (LT.lt a₁ a) ( …
    -/
    rintro _ _ (@⟨a₁, b₁, a₂, b₂, h⟩ | @⟨a, b₁, b₂, h⟩)
      /-
        case left
        α : Type u_1
        β : Type u_2
        inst✝³ : Preorder α
        inst✝² : Preorder β
        inst✝¹ : DenselyOrdered α
        inst✝ : DenselyOrdered β
        a₁ : α
        b₁ : β
        a₂ : α
        b₂ : β
        h : LT.lt a₁ a₂
        ⊢ Exists fun a => And (LT.lt { fst := a₁, snd := b₁ } a) (LT.lt a { fst := a₂, …
      -/
    · obtain ⟨c, h₁, h₂⟩ := exists_between h
      /-
        case left.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝³ : Preorder α
        inst✝² : Preorder β
        inst✝¹ : DenselyOrdered α
        inst✝ : DenselyOrdered β
        a₁ : α
        b₁ : β
        a₂ : α
        b₂ : β
        h : LT.lt a₁ a₂
        c : α
        h₁ : LT.lt a₁ c
        h₂ : LT.lt c a₂
        ⊢ Exists fun a => And (LT.lt { fst := a₁, snd := b₁ } a) (LT.lt a { fst := a₂, …
      -/
      exact ⟨(c, b₁), left _ _ h₁, left _ _ h₂⟩
      /-
        🎉 no goals
      -/
      /-
        case right
        α : Type u_1
        β : Type u_2
        inst✝³ : Preorder α
        inst✝² : Preorder β
        inst✝¹ : DenselyOrdered α
        inst✝ : DenselyOrdered β
        a : α
        b₁ b₂ : β
        h : LT.lt b₁ b₂
        ⊢ Exists fun a_1 => And (LT.lt { fst := a, snd := b₁ } a_1) (LT.lt a_1 { fst : …
      -/
    · obtain ⟨c, h₁, h₂⟩ := exists_between h
      /-
        case right.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝³ : Preorder α
        inst✝² : Preorder β
        inst✝¹ : DenselyOrdered α
        inst✝ : DenselyOrdered β
        a : α
        b₁ b₂ : β
        h : LT.lt b₁ b₂
        c : β
        h₁ : LT.lt b₁ c
        h₂ : LT.lt c b₂
        ⊢ Exists fun a_1 => And (LT.lt { fst := a, snd := b₁ } a_1) (LT.lt a_1 { fst : …
      -/
      exact ⟨(a, c), right _ h₁, right _ h₂⟩
      /-
        🎉 no goals
      -/


instance noMaxOrder_of_left [Preorder α] [Preorder β] [NoMaxOrder α] : NoMaxOrder (α ×ₗ β) where
  exists_gt := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMaxOrder α
      ⊢ ∀ (a : Lex (Prod α β)), Exists fun b => LT.lt a b
    -/
    rintro ⟨a, b⟩
    /-
      case mk
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMaxOrder α
      a : α
      b : β
      ⊢ Exists fun b_1 => LT.lt { fst := a, snd := b } b_1
    -/
    obtain ⟨c, h⟩ := exists_gt a
    /-
      case mk.intro
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMaxOrder α
      a : α
      b : β
      c : α
      h : LT.lt a c
      ⊢ Exists fun b_1 => LT.lt { fst := a, snd := b } b_1
    -/
    exact ⟨⟨c, b⟩, left _ _ h⟩
    /-
      🎉 no goals
    -/


instance noMinOrder_of_left [Preorder α] [Preorder β] [NoMinOrder α] : NoMinOrder (α ×ₗ β) where
  exists_lt := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMinOrder α
      ⊢ ∀ (a : Lex (Prod α β)), Exists fun b => LT.lt b a
    -/
    rintro ⟨a, b⟩
    /-
      case mk
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMinOrder α
      a : α
      b : β
      ⊢ Exists fun b_1 => LT.lt b_1 { fst := a, snd := b }
    -/
    obtain ⟨c, h⟩ := exists_lt a
    /-
      case mk.intro
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMinOrder α
      a : α
      b : β
      c : α
      h : LT.lt c a
      ⊢ Exists fun b_1 => LT.lt b_1 { fst := a, snd := b }
    -/
    exact ⟨⟨c, b⟩, left _ _ h⟩
    /-
      🎉 no goals
    -/


instance noMaxOrder_of_right [Preorder α] [Preorder β] [NoMaxOrder β] : NoMaxOrder (α ×ₗ β) where
  exists_gt := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMaxOrder β
      ⊢ ∀ (a : Lex (Prod α β)), Exists fun b => LT.lt a b
    -/
    rintro ⟨a, b⟩
    /-
      case mk
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMaxOrder β
      a : α
      b : β
      ⊢ Exists fun b_1 => LT.lt { fst := a, snd := b } b_1
    -/
    obtain ⟨c, h⟩ := exists_gt b
    /-
      case mk.intro
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMaxOrder β
      a : α
      b c : β
      h : LT.lt b c
      ⊢ Exists fun b_1 => LT.lt { fst := a, snd := b } b_1
    -/
    exact ⟨⟨a, c⟩, right _ h⟩
    /-
      🎉 no goals
    -/


instance noMinOrder_of_right [Preorder α] [Preorder β] [NoMinOrder β] : NoMinOrder (α ×ₗ β) where
  exists_lt := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMinOrder β
      ⊢ ∀ (a : Lex (Prod α β)), Exists fun b => LT.lt b a
    -/
    rintro ⟨a, b⟩
    /-
      case mk
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMinOrder β
      a : α
      b : β
      ⊢ Exists fun b_1 => LT.lt b_1 { fst := a, snd := b }
    -/
    obtain ⟨c, h⟩ := exists_lt b
    /-
      case mk.intro
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      inst✝ : NoMinOrder β
      a : α
      b c : β
      h : LT.lt c b
      ⊢ Exists fun b_1 => LT.lt b_1 { fst := a, snd := b }
    -/
    exact ⟨⟨a, c⟩, right _ h⟩
    /-
      🎉 no goals
    -/


