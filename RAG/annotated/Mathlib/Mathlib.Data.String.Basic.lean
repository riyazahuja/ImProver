@[simp] theorem endPos_empty : "".endPos = 0 := rfl


/-- `<` on string iterators. This coincides with `<` on strings as lists. -/
def ltb (s₁ s₂ : Iterator) : Bool :=
  if s₂.hasNext then
    if s₁.hasNext then
      if s₁.curr = s₂.curr then
        ltb s₁.next s₂.next
      else s₁.curr < s₂.curr
    else true
  else false


instance LT' : LT String :=
  ⟨fun s₁ s₂ ↦ ltb s₁.iter s₂.iter⟩


instance decidableLT : DecidableRel (α := String) (· < ·) := by
  /-
    ⊢ DecidableRel fun x1 x2 => LT.lt x1 x2
  -/
  simp only [LT']
  /-
    ⊢ DecidableRel fun x1 x2 => Eq (String.ltb x1.iter x2.iter) Bool.true
  -/
  infer_instance -- short-circuit type class inference
  /-
    🎉 no goals
  -/


/-- Induction on `String.ltb`. -/
def ltb.inductionOn.{u} {motive : Iterator → Iterator → Sort u} (it₁ it₂ : Iterator)
    (ind : ∀ s₁ s₂ i₁ i₂, Iterator.hasNext ⟨s₂, i₂⟩ → Iterator.hasNext ⟨s₁, i₁⟩ →
      get s₁ i₁ = get s₂ i₂ → motive (Iterator.next ⟨s₁, i₁⟩) (Iterator.next ⟨s₂, i₂⟩) →
      motive ⟨s₁, i₁⟩ ⟨s₂, i₂⟩)
    (eq : ∀ s₁ s₂ i₁ i₂, Iterator.hasNext ⟨s₂, i₂⟩ → Iterator.hasNext ⟨s₁, i₁⟩ →
      ¬ get s₁ i₁ = get s₂ i₂ → motive ⟨s₁, i₁⟩ ⟨s₂, i₂⟩)
    (base₁ : ∀ s₁ s₂ i₁ i₂, Iterator.hasNext ⟨s₂, i₂⟩ → ¬ Iterator.hasNext ⟨s₁, i₁⟩ →
      motive ⟨s₁, i₁⟩ ⟨s₂, i₂⟩)
    (base₂ : ∀ s₁ s₂ i₁ i₂, ¬ Iterator.hasNext ⟨s₂, i₂⟩ → motive ⟨s₁, i₁⟩ ⟨s₂, i₂⟩) :
    motive it₁ it₂ :=
  if h₂ : it₂.hasNext then
    if h₁ : it₁.hasNext then
      if heq : it₁.curr = it₂.curr then
        ind it₁.s it₂.s it₁.i it₂.i h₂ h₁ heq (inductionOn it₁.next it₂.next ind eq base₁ base₂)
      else eq it₁.s it₂.s it₁.i it₂.i h₂ h₁ heq
    else base₁ it₁.s it₂.s it₁.i it₂.i h₂ h₁
  else base₂ it₁.s it₂.s it₁.i it₂.i h₂


theorem ltb_cons_addChar (c : Char) (cs₁ cs₂ : List Char) (i₁ i₂ : Pos) :
    ltb ⟨⟨c :: cs₁⟩, i₁ + c⟩ ⟨⟨c :: cs₂⟩, i₂ + c⟩ = ltb ⟨⟨cs₁⟩, i₁⟩ ⟨⟨cs₂⟩, i₂⟩ := by
  apply ltb.inductionOn ⟨⟨cs₁⟩, i₁⟩ ⟨⟨cs₂⟩, i₂⟩ (motive := fun ⟨⟨cs₁⟩, i₁⟩ ⟨⟨cs₂⟩, i₂⟩ ↦
    ltb ⟨⟨c :: cs₁⟩, i₁ + c⟩ ⟨⟨c :: cs₂⟩, i₂ + c⟩ =
                                     /-
                                       case ind
                                       c : Char
                                       cs₁ cs₂ : List Char
                                       i₁ i₂ : String.Pos
                                       ⊢ ∀ (s₁ s₂ : String) (i₁ i₂ : String.Pos), Eq { s := s₂, i := i₂ }.hasNext Boo …
                                     -/
    ltb ⟨⟨cs₁⟩, i₁⟩ ⟨⟨cs₂⟩, i₂⟩) <;> simp only <;>
  /-
    case ind
    c : Char
    cs₁ cs₂ : List Char
    i₁ i₂ : String.Pos
    ⊢ ∀ (s₁ s₂ : String) (i₁ i₂ : String.Pos), Eq { s := s₂, i := i₂ }.hasNext Boo …
  -/
  intro ⟨cs₁⟩ ⟨cs₂⟩ i₁ i₂ <;>
  /-
    case ind
    c : Char
    cs₁✝ cs₂✝ : List Char
    i₁✝ i₂✝ : String.Pos
    cs₁ cs₂ : List Char
    i₁ i₂ : String.Pos
    ⊢ Eq { s := { data := cs₂ }, i := i₂ }.hasNext Bool.true → Eq { s := { data := …
  -/
  intros <;>
   /-
     case ind
     c : Char
     cs₁✝ cs₂✝ : List Char
     i₁✝ i₂✝ : String.Pos
     cs₁ cs₂ : List Char
     i₁ i₂ : String.Pos
     a✝³ : Eq { s := { data := cs₂ }, i := i₂ }.hasNext Bool.true
     a✝² : Eq { s := { data := cs₁ }, i := i₁ }.hasNext Bool.true
     a✝¹ : Eq ({ data := cs₁ }.get i₁) ({ data := cs₂ }.get i₂)
     a✝ : Eq (String.ltb { s := { data := List.cons c { s := { data := cs₁ }, i :=  …
     ⊢ Eq (String.ltb { s := { data := List.cons c { data := cs₁ }.data }, i := HAd …
   -/
  (conv => lhs; unfold ltb) <;> (conv => rhs; unfold ltb) <;>
  /-
    case ind
    c : Char
    cs₁✝ cs₂✝ : List Char
    i₁✝ i₂✝ : String.Pos
    cs₁ cs₂ : List Char
    i₁ i₂ : String.Pos
    a✝³ : Eq { s := { data := cs₂ }, i := i₂ }.hasNext Bool.true
    a✝² : Eq { s := { data := cs₁ }, i := i₁ }.hasNext Bool.true
    a✝¹ : Eq ({ data := cs₁ }.get i₁) ({ data := cs₂ }.get i₂)
    a✝ : Eq (String.ltb { s := { data := List.cons c { s := { data := cs₁ }, i :=  …
    ⊢ Eq (ite (Eq { s := { data := List.cons c { data := cs₂ }.data }, i := HAdd.h …
  -/
  /-
    🎉 no goals
  -/
  simp only [Iterator.hasNext_cons_addChar, ite_false, ite_true, *, reduceCtorEq]
  /-
    🎉 no goals
  -/
    /-
      case ind
      c : Char
      cs₁✝ cs₂✝ : List Char
      i₁✝ i₂✝ : String.Pos
      cs₁ cs₂ : List Char
      i₁ i₂ : String.Pos
      a✝³ : Eq { s := { data := cs₂ }, i := i₂ }.hasNext Bool.true
      a✝² : Eq { s := { data := cs₁ }, i := i₁ }.hasNext Bool.true
      a✝¹ : Eq ({ data := cs₁ }.get i₁) ({ data := cs₂ }.get i₂)
      a✝ : Eq (String.ltb { s := { data := List.cons c { s := { data := cs₁ }, i :=  …
      ⊢ Eq (ite (Eq { s := { data := List.cons c cs₁ }, i := HAdd.hAdd i₁ c }.curr { …
    -/
  · rename_i h₂ h₁ heq ih
    /-
      case ind
      c : Char
      cs₁✝ cs₂✝ : List Char
      i₁✝ i₂✝ : String.Pos
      cs₁ cs₂ : List Char
      i₁ i₂ : String.Pos
      h₂ : Eq { s := { data := cs₂ }, i := i₂ }.hasNext Bool.true
      h₁ : Eq { s := { data := cs₁ }, i := i₁ }.hasNext Bool.true
      heq : Eq ({ data := cs₁ }.get i₁) ({ data := cs₂ }.get i₂)
      ih : Eq (String.ltb { s := { data := List.cons c { s := { data := cs₁ }, i :=  …
      ⊢ Eq (ite (Eq { s := { data := List.cons c cs₁ }, i := HAdd.hAdd i₁ c }.curr { …
    -/
    simp only [Iterator.next, next, heq, Iterator.curr, get_cons_addChar, ite_true] at ih ⊢
    /-
      case ind
      c : Char
      cs₁✝ cs₂✝ : List Char
      i₁✝ i₂✝ : String.Pos
      cs₁ cs₂ : List Char
      i₁ i₂ : String.Pos
      h₂ : Eq { s := { data := cs₂ }, i := i₂ }.hasNext Bool.true
      h₁ : Eq { s := { data := cs₁ }, i := i₁ }.hasNext Bool.true
      heq : Eq ({ data := cs₁ }.get i₁) ({ data := cs₂ }.get i₂)
      ih : Eq (String.ltb { s := { data := List.cons c cs₁ }, i := HAdd.hAdd (HAdd.h …
      ⊢ Eq (String.ltb { s := { data := List.cons c cs₁ }, i := HAdd.hAdd (HAdd.hAdd …
    -/
    repeat rw [Pos.addChar_right_comm _ c]
    /-
      case ind
      c : Char
      cs₁✝ cs₂✝ : List Char
      i₁✝ i₂✝ : String.Pos
      cs₁ cs₂ : List Char
      i₁ i₂ : String.Pos
      h₂ : Eq { s := { data := cs₂ }, i := i₂ }.hasNext Bool.true
      h₁ : Eq { s := { data := cs₁ }, i := i₁ }.hasNext Bool.true
      heq : Eq ({ data := cs₁ }.get i₁) ({ data := cs₂ }.get i₂)
      ih : Eq (String.ltb { s := { data := List.cons c cs₁ }, i := HAdd.hAdd (HAdd.h …
      ⊢ Eq (String.ltb { s := { data := List.cons c cs₁ }, i := HAdd.hAdd (HAdd.hAdd …
    -/
    exact ih
    /-
      🎉 no goals
    -/
    /-
      case eq
      c : Char
      cs₁✝ cs₂✝ : List Char
      i₁✝ i₂✝ : String.Pos
      cs₁ cs₂ : List Char
      i₁ i₂ : String.Pos
      a✝² : Eq { s := { data := cs₂ }, i := i₂ }.hasNext Bool.true
      a✝¹ : Eq { s := { data := cs₁ }, i := i₁ }.hasNext Bool.true
      a✝ : Not (Eq ({ data := cs₁ }.get i₁) ({ data := cs₂ }.get i₂))
      ⊢ Eq (ite (Eq { s := { data := List.cons c cs₁ }, i := HAdd.hAdd i₁ c }.curr { …
    -/
  · rename_i h₂ h₁ hne
    /-
      case eq
      c : Char
      cs₁✝ cs₂✝ : List Char
      i₁✝ i₂✝ : String.Pos
      cs₁ cs₂ : List Char
      i₁ i₂ : String.Pos
      h₂ : Eq { s := { data := cs₂ }, i := i₂ }.hasNext Bool.true
      h₁ : Eq { s := { data := cs₁ }, i := i₁ }.hasNext Bool.true
      hne : Not (Eq ({ data := cs₁ }.get i₁) ({ data := cs₂ }.get i₂))
      ⊢ Eq (ite (Eq { s := { data := List.cons c cs₁ }, i := HAdd.hAdd i₁ c }.curr { …
    -/
    simp [Iterator.curr, get_cons_addChar, hne]
    /-
      🎉 no goals
    -/


@[simp]
theorem lt_iff_toList_lt : ∀ {s₁ s₂ : String}, s₁ < s₂ ↔ s₁.toList < s₂.toList
  | ⟨s₁⟩, ⟨s₂⟩ => show ltb ⟨⟨s₁⟩, 0⟩ ⟨⟨s₂⟩, 0⟩ ↔ s₁ < s₂ by
    /-
      s₁ s₂ : List Char
      ⊢ Iff (Eq (String.ltb { s := { data := s₁ }, i := 0 } { s := { data := s₂ }, i …
    -/
    induction s₁ generalizing s₂ <;> cases s₂
      /-
        case nil.nil
        ⊢ Iff (Eq (String.ltb { s := { data := List.nil }, i := 0 } { s := { data := L …
      -/
    · unfold ltb; decide
                  /-
                    🎉 no goals
                  -/
      /-
        case nil.cons
        head✝ : Char
        tail✝ : List Char
        ⊢ Iff (Eq (String.ltb { s := { data := List.nil }, i := 0 } { s := { data := L …
      -/
    · rename_i c₂ cs₂; apply iff_of_true
        /-
          case nil.cons.ha
          c₂ : Char
          cs₂ : List Char
          ⊢ Eq (String.ltb { s := { data := List.nil }, i := 0 } { s := { data := List.c …
        -/
      · unfold ltb
        /-
          case nil.cons.ha
          c₂ : Char
          cs₂ : List Char
          ⊢ Eq (ite (Eq { s := { data := List.cons c₂ cs₂ }, i := 0 }.hasNext Bool.true) …
        -/
        simp [Iterator.hasNext, Char.utf8Size_pos]
        /-
          🎉 no goals
        -/
        /-
          case nil.cons.hb
          c₂ : Char
          cs₂ : List Char
          ⊢ LT.lt List.nil (List.cons c₂ cs₂)
        -/
      · apply List.nil_lt_cons
        /-
          🎉 no goals
        -/
      /-
        case cons.nil
        head✝ : Char
        tail✝ : List Char
        tail_ih✝ : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := tail✝ }, i …
        ⊢ Iff (Eq (String.ltb { s := { data := List.cons head✝ tail✝ }, i := 0 } { s : …
      -/
    · rename_i c₁ cs₁ ih; apply iff_of_false
        /-
          case cons.nil.ha
          c₁ : Char
          cs₁ : List Char
          ih : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := cs₁ }, i := 0 }  …
          ⊢ Not (Eq (String.ltb { s := { data := List.cons c₁ cs₁ }, i := 0 } { s := { d …
        -/
      · unfold ltb
        /-
          case cons.nil.ha
          c₁ : Char
          cs₁ : List Char
          ih : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := cs₁ }, i := 0 }  …
          ⊢ Not (Eq (ite (Eq { s := { data := List.nil }, i := 0 }.hasNext Bool.true) (i …
        -/
        simp [Iterator.hasNext]
        /-
          🎉 no goals
        -/
        /-
          case cons.nil.hb
          c₁ : Char
          cs₁ : List Char
          ih : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := cs₁ }, i := 0 }  …
          ⊢ Not (LT.lt (List.cons c₁ cs₁) List.nil)
        -/
      · apply not_lt_of_lt; apply List.nil_lt_cons
                            /-
                              🎉 no goals
                            -/
      /-
        case cons.cons
        head✝¹ : Char
        tail✝¹ : List Char
        tail_ih✝ : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := tail✝¹ },  …
        head✝ : Char
        tail✝ : List Char
        ⊢ Iff (Eq (String.ltb { s := { data := List.cons head✝¹ tail✝¹ }, i := 0 } { s …
      -/
    · rename_i c₁ cs₁ ih c₂ cs₂; unfold ltb
      simp only [Iterator.hasNext, Pos.byteIdx_zero, endPos, utf8ByteSize, utf8ByteSize.go,
        add_pos_iff, Char.utf8Size_pos, or_true, decide_eq_true_eq, ↓reduceIte, Iterator.curr, get,
        utf8GetAux, Iterator.next, next, Bool.ite_eq_true_distrib]
      /-
        case cons.cons
        c₁ : Char
        cs₁ : List Char
        ih : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := cs₁ }, i := 0 }  …
        c₂ : Char
        cs₂ : List Char
        ⊢ Iff (ite (Eq c₁ c₂) (Eq (String.ltb { s := { data := List.cons c₁ cs₁ }, i : …
      -/
      split_ifs with h
        /-
          case pos
          c₁ : Char
          cs₁ : List Char
          ih : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := cs₁ }, i := 0 }  …
          c₂ : Char
          cs₂ : List Char
          h : Eq c₁ c₂
          ⊢ Iff (Eq (String.ltb { s := { data := List.cons c₁ cs₁ }, i := HAdd.hAdd 0 c₁ …
        -/
      · subst c₂
        suffices ltb ⟨⟨c₁ :: cs₁⟩, (0 : Pos) + c₁⟩ ⟨⟨c₁ :: cs₂⟩, (0 : Pos) + c₁⟩ =
          ltb ⟨⟨cs₁⟩, 0⟩ ⟨⟨cs₂⟩, 0⟩ by rw [this]; exact (ih cs₂).trans List.Lex.cons_iff.symm
        /-
          case pos
          c₁ : Char
          cs₁ : List Char
          ih : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := cs₁ }, i := 0 }  …
          cs₂ : List Char
          ⊢ Eq (String.ltb { s := { data := List.cons c₁ cs₁ }, i := HAdd.hAdd 0 c₁ } {  …
        -/
        apply ltb_cons_addChar
        /-
          🎉 no goals
        -/
        /-
          case neg
          c₁ : Char
          cs₁ : List Char
          ih : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := cs₁ }, i := 0 }  …
          c₂ : Char
          cs₂ : List Char
          h : Not (Eq c₁ c₂)
          ⊢ Iff (LT.lt c₁ c₂) (LT.lt (List.cons c₁ cs₁) (List.cons c₂ cs₂))
        -/
      · refine ⟨List.Lex.rel, fun e ↦ ?_⟩
        /-
          case neg
          c₁ : Char
          cs₁ : List Char
          ih : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := cs₁ }, i := 0 }  …
          c₂ : Char
          cs₂ : List Char
          h : Not (Eq c₁ c₂)
          e : LT.lt (List.cons c₁ cs₁) (List.cons c₂ cs₂)
          ⊢ LT.lt c₁ c₂
        -/
        cases e <;> rename_i h'
          /-
            case neg.cons
            c₁ : Char
            cs₁ : List Char
            ih : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := cs₁ }, i := 0 }  …
            cs₂ : List Char
            h : Not (Eq c₁ c₁)
            h' : List.Lex (fun x1 x2 => LT.lt x1 x2) cs₁ cs₂
            ⊢ LT.lt c₁ c₁
          -/
        · contradiction
          /-
            🎉 no goals
          -/
          /-
            case neg.rel
            c₁ : Char
            cs₁ : List Char
            ih : ∀ (s₂ : List Char), Iff (Eq (String.ltb { s := { data := cs₁ }, i := 0 }  …
            c₂ : Char
            cs₂ : List Char
            h : Not (Eq c₁ c₂)
            h' : LT.lt c₁ c₂
            ⊢ LT.lt c₁ c₂
          -/
        · assumption
          /-
            🎉 no goals
          -/


instance LE : LE String :=
  ⟨fun s₁ s₂ ↦ ¬s₂ < s₁⟩


instance decidableLE : DecidableRel (α := String) (· ≤ ·) := by
  /-
    ⊢ DecidableRel fun x1 x2 => LE.le x1 x2
  -/
  simp only [LE]
  /-
    ⊢ DecidableRel fun x1 x2 => Not (LT.lt x2 x1)
  -/
  infer_instance -- short-circuit type class inference
  /-
    🎉 no goals
  -/


@[simp]
theorem le_iff_toList_le {s₁ s₂ : String} : s₁ ≤ s₂ ↔ s₁.toList ≤ s₂.toList :=
  (not_congr lt_iff_toList_lt).trans not_lt


theorem toList_inj {s₁ s₂ : String} : s₁.toList = s₂.toList ↔ s₁ = s₂ :=
  ⟨congr_arg mk, congr_arg toList⟩


theorem asString_nil : [].asString = "" :=
  rfl


@[deprecated (since := "2024-06-04")] alias nil_asString_eq_empty := asString_nil


@[simp]
theorem toList_empty : "".toList = [] :=
  rfl


theorem asString_toList (s : String) : s.toList.asString = s :=
  rfl


@[deprecated (since := "2024-06-04")] alias asString_inv_toList := asString_toList


theorem toList_nonempty : ∀ {s : String}, s ≠ "" → s.toList = s.head :: (s.drop 1).toList
  | ⟨s⟩, h => by
    cases s with
    | nil => simp at h
    | cons c cs =>
      simp only [toList, data_drop, List.drop_succ_cons, List.drop_zero, List.cons.injEq, and_true]
      rfl


@[simp]
theorem head_empty : "".data.head! = default :=
  rfl


instance : LinearOrder String where
  le_refl _ := le_iff_toList_le.mpr le_rfl
  le_trans a b c := by
    /-
      a b c : String
      ⊢ LE.le a b → LE.le b c → LE.le a c
    -/
    simp only [le_iff_toList_le]
    /-
      a b c : String
      ⊢ LE.le a.toList b.toList → LE.le b.toList c.toList → LE.le a.toList c.toList
    -/
    apply le_trans
    /-
      🎉 no goals
    -/
  lt_iff_le_not_le a b := by
    /-
      a b : String
      ⊢ Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
    -/
    simp only [lt_iff_toList_lt, le_iff_toList_le, lt_iff_le_not_le]
    /-
      🎉 no goals
    -/
  le_antisymm a b := by
    /-
      a b : String
      ⊢ LE.le a b → LE.le b a → Eq a b
    -/
    simp only [le_iff_toList_le, ← toList_inj]
    /-
      a b : String
      ⊢ LE.le a.toList b.toList → LE.le b.toList a.toList → Eq a.toList b.toList
    -/
    apply le_antisymm
    /-
      🎉 no goals
    -/
  le_total a b := by
    /-
      a b : String
      ⊢ Or (LE.le a b) (LE.le b a)
    -/
    simp only [le_iff_toList_le]
    /-
      a b : String
      ⊢ Or (LE.le a.toList b.toList) (LE.le b.toList a.toList)
    -/
    apply le_total
    /-
      🎉 no goals
    -/
  decidableLE := String.decidableLE
  compare_eq_compareOfLessAndEq a b := by
    /-
      a b : String
      ⊢ Eq (Ord.compare a b) (compareOfLessAndEq a b)
    -/
    simp only [compare, compareOfLessAndEq, instLT, List.instLT, lt_iff_toList_lt, toList]
    /-
      a b : String
      ⊢ Eq (ite (a.data.lt b.data) Ordering.lt (ite (Eq a b) Ordering.eq Ordering.gt …
    -/
    split_ifs <;>
    /-
      case pos
      a b : String
      h✝¹ : a.data.lt b.data
      h✝ : LT.lt a.data b.data
      ⊢ Eq Ordering.lt Ordering.lt
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp only [List.lt_iff_lex_lt] at * <;>
    /-
      🎉 no goals
    -/
    /-
      case pos
      a b : String
      h✝² : Not (LT.lt a.data b.data)
      h✝¹ : Eq a b
      h✝ : List.Lex (fun x1 x2 => LT.lt x1 x2) a.data b.data
      ⊢ False
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
    contradiction
    /-
      🎉 no goals
    -/


theorem toList_asString (l : List Char) : l.asString.toList = l :=
  rfl


@[deprecated (since := "2024-06-04")] alias toList_inv_asString := toList_asString


@[simp]
theorem length_asString (l : List Char) : l.asString.length = l.length :=
  rfl


@[simp]
theorem asString_inj {l l' : List Char} : l.asString = l'.asString ↔ l = l' :=
              /-
                l l' : List Char
                h : Eq l.asString l'.asString
                ⊢ Eq l l'
              -/
  ⟨fun h ↦ by rw [← toList_asString l, ← toList_asString l', toList_inj, h],
              /-
                🎉 no goals
              -/
   fun h ↦ h ▸ rfl⟩


theorem asString_eq {l : List Char} {s : String} : l.asString = s ↔ l = s.toList := by
  /-
    l : List Char
    s : String
    ⊢ Iff (Eq l.asString s) (Eq l s.toList)
  -/
  rw [← asString_toList s, asString_inj, asString_toList s]
  /-
    🎉 no goals
  -/


@[simp]
theorem String.length_data (s : String) : s.data.length = s.length :=
  rfl

