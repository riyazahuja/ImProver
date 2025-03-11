@[to_additive]
instance one : One (WithTop α) :=
  ⟨(1 : α)⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_one : ((1 : α) : WithTop α) = 1 :=
  rfl


@[to_additive (attr := simp, norm_cast)]
lemma coe_eq_one : (a : WithTop α) = 1 ↔ a = 1 := coe_eq_coe


@[to_additive (attr := simp, norm_cast)]
lemma one_eq_coe : 1 = (a : WithTop α) ↔ a = 1 := eq_comm.trans coe_eq_one


@[to_additive (attr := simp)] lemma top_ne_one : (⊤ : WithTop α) ≠ 1 := top_ne_coe


@[to_additive (attr := simp)] lemma one_ne_top : (1 : WithTop α) ≠ ⊤ := coe_ne_top


@[to_additive (attr := simp)]
theorem untop_one : (1 : WithTop α).untop coe_ne_top = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem untop_one' (d : α) : (1 : WithTop α).untop' d = 1 :=
  rfl


@[to_additive (attr := simp, norm_cast) coe_nonneg]
theorem one_le_coe [LE α] {a : α} : 1 ≤ (a : WithTop α) ↔ 1 ≤ a :=
  coe_le_coe


@[to_additive (attr := simp, norm_cast) coe_le_zero]
theorem coe_le_one [LE α] {a : α} : (a : WithTop α) ≤ 1 ↔ a ≤ 1 :=
  coe_le_coe


@[to_additive (attr := simp, norm_cast) coe_pos]
theorem one_lt_coe [LT α] {a : α} : 1 < (a : WithTop α) ↔ 1 < a :=
  coe_lt_coe


@[to_additive (attr := simp, norm_cast) coe_lt_zero]
theorem coe_lt_one [LT α] {a : α} : (a : WithTop α) < 1 ↔ a < 1 :=
  coe_lt_coe


@[to_additive (attr := simp)]
protected theorem map_one {β} (f : α → β) : (1 : WithTop α).map f = (f 1 : WithTop β) :=
  rfl


@[to_additive]
theorem map_eq_one_iff {α} {f : α → β} {v : WithTop α} [One β] :
    WithTop.map f v = 1 ↔ ∃ x, v = .some x ∧ f x = 1 := map_eq_some_iff


@[to_additive]
theorem one_eq_map_iff {α} {f : α → β} {v : WithTop α} [One β] :
    1 = WithTop.map f v ↔ ∃ x, v = .some x ∧ f x = 1 := some_eq_map_iff


instance zeroLEOneClass [Zero α] [LE α] [ZeroLEOneClass α] : ZeroLEOneClass (WithTop α) :=
  ⟨coe_le_coe.2 zero_le_one⟩


instance add : Add (WithTop α) :=
  ⟨Option.map₂ (· + ·)⟩


@[simp, norm_cast] lemma coe_add (a b : α) : ↑(a + b) = (a + b : WithTop α) := rfl


@[simp]
theorem top_add (a : WithTop α) : ⊤ + a = ⊤ :=
  rfl


@[simp]
                                                  /-
                                                    α : Type u
                                                    inst✝ : Add α
                                                    a : WithTop α
                                                    ⊢ Eq (HAdd.hAdd a Top.top) Top.top
                                                  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
theorem add_top (a : WithTop α) : a + ⊤ = ⊤ := by cases a <;> rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem add_eq_top : a + b = ⊤ ↔ a = ⊤ ∨ b = ⊤ := by
  match a, b with
  | ⊤, _ => simp
  | _, ⊤ => simp
  | (a : α), (b : α) => simp only [← coe_add, coe_ne_top, or_false]


theorem add_ne_top : a + b ≠ ⊤ ↔ a ≠ ⊤ ∧ b ≠ ⊤ :=
  add_eq_top.not.trans not_or


theorem add_lt_top [LT α] {a b : WithTop α} : a + b < ⊤ ↔ a < ⊤ ∧ b < ⊤ := by
  /-
    α : Type u
    inst✝¹ : Add α
    inst✝ : LT α
    a b : WithTop α
    ⊢ Iff (LT.lt (HAdd.hAdd a b) Top.top) (And (LT.lt a Top.top) (LT.lt b Top.top))
  -/
  simp_rw [WithTop.lt_top_iff_ne_top, add_ne_top]
  /-
    🎉 no goals
  -/


theorem add_eq_coe :
    ∀ {a b : WithTop α} {c : α}, a + b = c ↔ ∃ a' b' : α, ↑a' = a ∧ ↑b' = b ∧ a' + b' = c
                  /-
                    α : Type u
                    inst✝ : Add α
                    b : WithTop α
                    c : α
                    ⊢ Iff (Eq (HAdd.hAdd Top.top b) ↑c) (Exists fun a' => Exists fun b' => And (Eq …
                  -/
  | ⊤, b, c => by simp
                  /-
                    🎉 no goals
                  -/
                       /-
                         α : Type u
                         inst✝ : Add α
                         a c : α
                         ⊢ Iff (Eq (HAdd.hAdd (↑a) Top.top) ↑c) (Exists fun a' => Exists fun b' => And  …
                       -/
  | some a, ⊤, c => by simp
                       /-
                         🎉 no goals
                       -/
                            /-
                              α : Type u
                              inst✝ : Add α
                              a b c : α
                              ⊢ Iff (Eq (HAdd.hAdd ↑a ↑b) ↑c) (Exists fun a' => Exists fun b' => And (Eq ↑a' …
                            -/
  | some a, some b, c => by norm_cast; simp
                                       /-
                                         🎉 no goals
                                       -/


                                                                             /-
                                                                               α : Type u
                                                                               inst✝ : Add α
                                                                               x : WithTop α
                                                                               y : α
                                                                               ⊢ Iff (Eq (HAdd.hAdd x ↑y) Top.top) (Eq x Top.top)
                                                                             -/
theorem add_coe_eq_top_iff {x : WithTop α} {y : α} : x + y = ⊤ ↔ x = ⊤ := by simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


                                                                      /-
                                                                        α : Type u
                                                                        inst✝ : Add α
                                                                        x : α
                                                                        y : WithTop α
                                                                        ⊢ Iff (Eq (HAdd.hAdd (↑x) y) Top.top) (Eq y Top.top)
                                                                      -/
theorem coe_add_eq_top_iff {y : WithTop α} : ↑x + y = ⊤ ↔ y = ⊤ := by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem add_right_cancel_iff [IsRightCancelAdd α] (ha : a ≠ ⊤) : b + a = c + a ↔ b = c := by
  /-
    α : Type u
    inst✝¹ : Add α
    a b c : WithTop α
    inst✝ : IsRightCancelAdd α
    ha : Ne a Top.top
    ⊢ Iff (Eq (HAdd.hAdd b a) (HAdd.hAdd c a)) (Eq b c)
  -/
  lift a to α using ha
  /-
    case intro
    α : Type u
    inst✝¹ : Add α
    b c : WithTop α
    inst✝ : IsRightCancelAdd α
    a : α
    ⊢ Iff (Eq (HAdd.hAdd b ↑a) (HAdd.hAdd c ↑a)) (Eq b c)
  -/
  obtain rfl | hb := eq_or_ne b ⊤
    /-
      case intro.inl
      α : Type u
      inst✝¹ : Add α
      c : WithTop α
      inst✝ : IsRightCancelAdd α
      a : α
      ⊢ Iff (Eq (HAdd.hAdd Top.top ↑a) (HAdd.hAdd c ↑a)) (Eq Top.top c)
    -/
  · rw [top_add, eq_comm, WithTop.add_coe_eq_top_iff, eq_comm]
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    α : Type u
    inst✝¹ : Add α
    b c : WithTop α
    inst✝ : IsRightCancelAdd α
    a : α
    hb : Ne b Top.top
    ⊢ Iff (Eq (HAdd.hAdd b ↑a) (HAdd.hAdd c ↑a)) (Eq b c)
  -/
  lift b to α using hb
  simp_rw [← WithTop.coe_add, eq_comm, WithTop.add_eq_coe, coe_eq_coe, exists_and_left,
    exists_eq_left, add_left_inj, exists_eq_right, eq_comm]


theorem add_right_cancel [IsRightCancelAdd α] (ha : a ≠ ⊤) (h : b + a = c + a) : b = c :=
  (WithTop.add_right_cancel_iff ha).1 h


theorem add_left_cancel_iff [IsLeftCancelAdd α] (ha : a ≠ ⊤) : a + b = a + c ↔ b = c := by
  /-
    α : Type u
    inst✝¹ : Add α
    a b c : WithTop α
    inst✝ : IsLeftCancelAdd α
    ha : Ne a Top.top
    ⊢ Iff (Eq (HAdd.hAdd a b) (HAdd.hAdd a c)) (Eq b c)
  -/
  lift a to α using ha
  /-
    case intro
    α : Type u
    inst✝¹ : Add α
    b c : WithTop α
    inst✝ : IsLeftCancelAdd α
    a : α
    ⊢ Iff (Eq (HAdd.hAdd (↑a) b) (HAdd.hAdd (↑a) c)) (Eq b c)
  -/
  obtain rfl | hb := eq_or_ne b ⊤
    /-
      case intro.inl
      α : Type u
      inst✝¹ : Add α
      c : WithTop α
      inst✝ : IsLeftCancelAdd α
      a : α
      ⊢ Iff (Eq (HAdd.hAdd (↑a) Top.top) (HAdd.hAdd (↑a) c)) (Eq Top.top c)
    -/
  · rw [add_top, eq_comm, WithTop.coe_add_eq_top_iff, eq_comm]
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    α : Type u
    inst✝¹ : Add α
    b c : WithTop α
    inst✝ : IsLeftCancelAdd α
    a : α
    hb : Ne b Top.top
    ⊢ Iff (Eq (HAdd.hAdd (↑a) b) (HAdd.hAdd (↑a) c)) (Eq b c)
  -/
  lift b to α using hb
  simp_rw [← WithTop.coe_add, eq_comm, WithTop.add_eq_coe, eq_comm, coe_eq_coe,
    exists_and_left, exists_eq_left', add_right_inj, exists_eq_right']


theorem add_left_cancel [IsLeftCancelAdd α] (ha : a ≠ ⊤) (h : a + b = a + c) : b = c :=
  (WithTop.add_left_cancel_iff ha).1 h


instance addLeftMono [LE α] [AddLeftMono α] : AddLeftMono (WithTop α) :=
  ⟨fun a b c h => by
    /-
      α : Type u
      β : Type v
      inst✝² : Add α
      a✝ b✝ c✝ d : WithTop α
      x : α
      inst✝¹ : LE α
      inst✝ : AddLeftMono α
      a b c : WithTop α
      h : LE.le b c
      ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd a c)
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
    cases a <;> cases c <;> try exact le_top
    /-
      case coe.coe
      α : Type u
      β : Type v
      inst✝² : Add α
      a b✝ c d : WithTop α
      x : α
      inst✝¹ : LE α
      inst✝ : AddLeftMono α
      b : WithTop α
      a✝¹ a✝ : α
      h : LE.le b ↑a✝
      ⊢ LE.le (HAdd.hAdd (↑a✝¹) b) (HAdd.hAdd ↑a✝¹ ↑a✝)
    -/
    rcases le_coe_iff.1 h with ⟨b, rfl, _⟩
    /-
      case coe.coe.intro.intro
      α : Type u
      β : Type v
      inst✝² : Add α
      a b✝ c d : WithTop α
      x : α
      inst✝¹ : LE α
      inst✝ : AddLeftMono α
      a✝¹ a✝ b : α
      right✝ : LE.le b a✝
      h : LE.le ↑b ↑a✝
      ⊢ LE.le (HAdd.hAdd ↑a✝¹ ↑b) (HAdd.hAdd ↑a✝¹ ↑a✝)
    -/
    exact coe_le_coe.2 (add_le_add_left (coe_le_coe.1 h) _)⟩
    /-
      🎉 no goals
    -/


instance addRightMono [LE α] [AddRightMono α] : AddRightMono (WithTop α) :=
  ⟨fun a b c h => by
    /-
      α : Type u
      β : Type v
      inst✝² : Add α
      a✝ b✝ c✝ d : WithTop α
      x : α
      inst✝¹ : LE α
      inst✝ : AddRightMono α
      a b c : WithTop α
      h : LE.le b c
      ⊢ LE.le (Function.swap (fun x1 x2 => HAdd.hAdd x1 x2) a b) (Function.swap (fun …
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
    cases a <;> cases c <;> try exact le_top
    /-
      case coe.coe
      α : Type u
      β : Type v
      inst✝² : Add α
      a b✝ c d : WithTop α
      x : α
      inst✝¹ : LE α
      inst✝ : AddRightMono α
      b : WithTop α
      a✝¹ a✝ : α
      h : LE.le b ↑a✝
      ⊢ LE.le (Function.swap (fun x1 x2 => HAdd.hAdd x1 x2) (↑a✝¹) b) (Function.swap …
    -/
    rcases le_coe_iff.1 h with ⟨b, rfl, _⟩
    /-
      case coe.coe.intro.intro
      α : Type u
      β : Type v
      inst✝² : Add α
      a b✝ c d : WithTop α
      x : α
      inst✝¹ : LE α
      inst✝ : AddRightMono α
      a✝¹ a✝ b : α
      right✝ : LE.le b a✝
      h : LE.le ↑b ↑a✝
      ⊢ LE.le (Function.swap (fun x1 x2 => HAdd.hAdd x1 x2) ↑a✝¹ ↑b) (Function.swap  …
    -/
    exact coe_le_coe.2 (add_le_add_right (coe_le_coe.1 h) _)⟩
    /-
      🎉 no goals
    -/


instance addLeftReflectLT [LT α] [AddLeftReflectLT α] : AddLeftReflectLT (WithTop α) :=
  ⟨fun a b c h => by
    /-
      α : Type u
      β : Type v
      inst✝² : Add α
      a✝ b✝ c✝ d : WithTop α
      x : α
      inst✝¹ : LT α
      inst✝ : AddLeftReflectLT α
      a b c : WithTop α
      h : LT.lt (HAdd.hAdd a b) (HAdd.hAdd a c)
      ⊢ LT.lt b c
    -/
    induction a; · exact (WithTop.not_top_lt _ h).elim
                   /-
                     🎉 no goals
                   -/
    /-
      case coe
      α : Type u
      β : Type v
      inst✝² : Add α
      a b✝ c✝ d : WithTop α
      x : α
      inst✝¹ : LT α
      inst✝ : AddLeftReflectLT α
      b c : WithTop α
      a✝ : α
      h : LT.lt (HAdd.hAdd (↑a✝) b) (HAdd.hAdd (↑a✝) c)
      ⊢ LT.lt b c
    -/
    induction b; · exact (WithTop.not_top_lt _ h).elim
                   /-
                     🎉 no goals
                   -/
    /-
      case coe.coe
      α : Type u
      β : Type v
      inst✝² : Add α
      a b c✝ d : WithTop α
      x : α
      inst✝¹ : LT α
      inst✝ : AddLeftReflectLT α
      c : WithTop α
      a✝¹ a✝ : α
      h : LT.lt (HAdd.hAdd ↑a✝¹ ↑a✝) (HAdd.hAdd (↑a✝¹) c)
      ⊢ LT.lt (↑a✝) c
    -/
    induction c
      /-
        case coe.coe.top
        α : Type u
        β : Type v
        inst✝² : Add α
        a b c d : WithTop α
        x : α
        inst✝¹ : LT α
        inst✝ : AddLeftReflectLT α
        a✝¹ a✝ : α
        h : LT.lt (HAdd.hAdd ↑a✝¹ ↑a✝) (HAdd.hAdd (↑a✝¹) Top.top)
        ⊢ LT.lt (↑a✝) Top.top
      -/
    · exact coe_lt_top _
      /-
        🎉 no goals
      -/
      /-
        case coe.coe.coe
        α : Type u
        β : Type v
        inst✝² : Add α
        a b c d : WithTop α
        x : α
        inst✝¹ : LT α
        inst✝ : AddLeftReflectLT α
        a✝² a✝¹ a✝ : α
        h : LT.lt (HAdd.hAdd ↑a✝² ↑a✝¹) (HAdd.hAdd ↑a✝² ↑a✝)
        ⊢ LT.lt ↑a✝¹ ↑a✝
      -/
    · exact coe_lt_coe.2 (lt_of_add_lt_add_left <| coe_lt_coe.1 h)⟩
      /-
        🎉 no goals
      -/


instance addRightReflectLT [LT α] [AddRightReflectLT α] : AddRightReflectLT (WithTop α) :=
  ⟨fun a b c h => by
    /-
      α : Type u
      β : Type v
      inst✝² : Add α
      a✝ b✝ c✝ d : WithTop α
      x : α
      inst✝¹ : LT α
      inst✝ : AddRightReflectLT α
      a b c : WithTop α
      h : LT.lt (Function.swap (fun x1 x2 => HAdd.hAdd x1 x2) a b) (Function.swap (f …
      ⊢ LT.lt b c
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
    cases a <;> cases b <;> try exact (WithTop.not_top_lt _ h).elim
    /-
      case coe.coe
      α : Type u
      β : Type v
      inst✝² : Add α
      a b c✝ d : WithTop α
      x : α
      inst✝¹ : LT α
      inst✝ : AddRightReflectLT α
      c : WithTop α
      a✝¹ a✝ : α
      h : LT.lt (Function.swap (fun x1 x2 => HAdd.hAdd x1 x2) ↑a✝¹ ↑a✝) (Function.sw …
      ⊢ LT.lt (↑a✝) c
    -/
    cases c
      /-
        case coe.coe.top
        α : Type u
        β : Type v
        inst✝² : Add α
        a b c d : WithTop α
        x : α
        inst✝¹ : LT α
        inst✝ : AddRightReflectLT α
        a✝¹ a✝ : α
        h : LT.lt (Function.swap (fun x1 x2 => HAdd.hAdd x1 x2) ↑a✝¹ ↑a✝) (Function.sw …
        ⊢ LT.lt (↑a✝) Top.top
      -/
    · exact coe_lt_top _
      /-
        🎉 no goals
      -/
      /-
        case coe.coe.coe
        α : Type u
        β : Type v
        inst✝² : Add α
        a b c d : WithTop α
        x : α
        inst✝¹ : LT α
        inst✝ : AddRightReflectLT α
        a✝² a✝¹ a✝ : α
        h : LT.lt (Function.swap (fun x1 x2 => HAdd.hAdd x1 x2) ↑a✝² ↑a✝¹) (Function.s …
        ⊢ LT.lt ↑a✝¹ ↑a✝
      -/
    · exact coe_lt_coe.2 (lt_of_add_lt_add_right <| coe_lt_coe.1 h)⟩
      /-
        🎉 no goals
      -/


protected theorem le_of_add_le_add_left [LE α] [AddLeftReflectLE α] (ha : a ≠ ⊤)
    (h : a + b ≤ a + c) : b ≤ c := by
  /-
    α : Type u
    inst✝² : Add α
    a b c : WithTop α
    inst✝¹ : LE α
    inst✝ : AddLeftReflectLE α
    ha : Ne a Top.top
    h : LE.le (HAdd.hAdd a b) (HAdd.hAdd a c)
    ⊢ LE.le b c
  -/
  lift a to α using ha
  /-
    case intro
    α : Type u
    inst✝² : Add α
    b c : WithTop α
    inst✝¹ : LE α
    inst✝ : AddLeftReflectLE α
    a : α
    h : LE.le (HAdd.hAdd (↑a) b) (HAdd.hAdd (↑a) c)
    ⊢ LE.le b c
  -/
  induction c
    /-
      case intro.top
      α : Type u
      inst✝² : Add α
      b : WithTop α
      inst✝¹ : LE α
      inst✝ : AddLeftReflectLE α
      a : α
      h : LE.le (HAdd.hAdd (↑a) b) (HAdd.hAdd (↑a) Top.top)
      ⊢ LE.le b Top.top
    -/
  · exact le_top
    /-
      🎉 no goals
    -/
    /-
      case intro.coe
      α : Type u
      inst✝² : Add α
      b : WithTop α
      inst✝¹ : LE α
      inst✝ : AddLeftReflectLE α
      a a✝ : α
      h : LE.le (HAdd.hAdd (↑a) b) (HAdd.hAdd ↑a ↑a✝)
      ⊢ LE.le b ↑a✝
    -/
  · induction b
      /-
        case intro.coe.top
        α : Type u
        inst✝² : Add α
        inst✝¹ : LE α
        inst✝ : AddLeftReflectLE α
        a a✝ : α
        h : LE.le (HAdd.hAdd (↑a) Top.top) (HAdd.hAdd ↑a ↑a✝)
        ⊢ LE.le Top.top ↑a✝
      -/
    · exact (not_top_le_coe _ h).elim
      /-
        🎉 no goals
      -/
      /-
        case intro.coe.coe
        α : Type u
        inst✝² : Add α
        inst✝¹ : LE α
        inst✝ : AddLeftReflectLE α
        a a✝¹ a✝ : α
        h : LE.le (HAdd.hAdd ↑a ↑a✝) (HAdd.hAdd ↑a ↑a✝¹)
        ⊢ LE.le ↑a✝ ↑a✝¹
      -/
    · simp only [← coe_add, coe_le_coe] at h ⊢
      /-
        case intro.coe.coe
        α : Type u
        inst✝² : Add α
        inst✝¹ : LE α
        inst✝ : AddLeftReflectLE α
        a a✝¹ a✝ : α
        h : LE.le (HAdd.hAdd a a✝) (HAdd.hAdd a a✝¹)
        ⊢ LE.le a✝ a✝¹
      -/
      exact le_of_add_le_add_left h
      /-
        🎉 no goals
      -/


protected theorem le_of_add_le_add_right [LE α] [AddRightReflectLE α]
    (ha : a ≠ ⊤) (h : b + a ≤ c + a) : b ≤ c := by
  /-
    α : Type u
    inst✝² : Add α
    a b c : WithTop α
    inst✝¹ : LE α
    inst✝ : AddRightReflectLE α
    ha : Ne a Top.top
    h : LE.le (HAdd.hAdd b a) (HAdd.hAdd c a)
    ⊢ LE.le b c
  -/
  lift a to α using ha
  /-
    case intro
    α : Type u
    inst✝² : Add α
    b c : WithTop α
    inst✝¹ : LE α
    inst✝ : AddRightReflectLE α
    a : α
    h : LE.le (HAdd.hAdd b ↑a) (HAdd.hAdd c ↑a)
    ⊢ LE.le b c
  -/
  cases c
    /-
      case intro.top
      α : Type u
      inst✝² : Add α
      b : WithTop α
      inst✝¹ : LE α
      inst✝ : AddRightReflectLE α
      a : α
      h : LE.le (HAdd.hAdd b ↑a) (HAdd.hAdd Top.top ↑a)
      ⊢ LE.le b Top.top
    -/
  · exact le_top
    /-
      🎉 no goals
    -/
    /-
      case intro.coe
      α : Type u
      inst✝² : Add α
      b : WithTop α
      inst✝¹ : LE α
      inst✝ : AddRightReflectLE α
      a a✝ : α
      h : LE.le (HAdd.hAdd b ↑a) (HAdd.hAdd ↑a✝ ↑a)
      ⊢ LE.le b ↑a✝
    -/
  · cases b
      /-
        case intro.coe.top
        α : Type u
        inst✝² : Add α
        inst✝¹ : LE α
        inst✝ : AddRightReflectLE α
        a a✝ : α
        h : LE.le (HAdd.hAdd Top.top ↑a) (HAdd.hAdd ↑a✝ ↑a)
        ⊢ LE.le Top.top ↑a✝
      -/
    · exact (not_top_le_coe _ h).elim
      /-
        🎉 no goals
      -/
      /-
        case intro.coe.coe
        α : Type u
        inst✝² : Add α
        inst✝¹ : LE α
        inst✝ : AddRightReflectLE α
        a a✝¹ a✝ : α
        h : LE.le (HAdd.hAdd ↑a✝ ↑a) (HAdd.hAdd ↑a✝¹ ↑a)
        ⊢ LE.le ↑a✝ ↑a✝¹
      -/
    · exact coe_le_coe.2 (le_of_add_le_add_right <| coe_le_coe.1 h)
      /-
        🎉 no goals
      -/


protected theorem add_lt_add_left [LT α] [AddLeftStrictMono α] (ha : a ≠ ⊤)
    (h : b < c) : a + b < a + c := by
  /-
    α : Type u
    inst✝² : Add α
    a b c : WithTop α
    inst✝¹ : LT α
    inst✝ : AddLeftStrictMono α
    ha : Ne a Top.top
    h : LT.lt b c
    ⊢ LT.lt (HAdd.hAdd a b) (HAdd.hAdd a c)
  -/
  lift a to α using ha
  /-
    case intro
    α : Type u
    inst✝² : Add α
    b c : WithTop α
    inst✝¹ : LT α
    inst✝ : AddLeftStrictMono α
    h : LT.lt b c
    a : α
    ⊢ LT.lt (HAdd.hAdd (↑a) b) (HAdd.hAdd (↑a) c)
  -/
  rcases lt_iff_exists_coe.1 h with ⟨b, rfl, h'⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝² : Add α
    c : WithTop α
    inst✝¹ : LT α
    inst✝ : AddLeftStrictMono α
    a b : α
    h' h : LT.lt (↑b) c
    ⊢ LT.lt (HAdd.hAdd ↑a ↑b) (HAdd.hAdd (↑a) c)
  -/
  cases c
    /-
      case intro.intro.intro.top
      α : Type u
      inst✝² : Add α
      inst✝¹ : LT α
      inst✝ : AddLeftStrictMono α
      a b : α
      h' h : LT.lt (↑b) Top.top
      ⊢ LT.lt (HAdd.hAdd ↑a ↑b) (HAdd.hAdd (↑a) Top.top)
    -/
  · exact coe_lt_top _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.coe
      α : Type u
      inst✝² : Add α
      inst✝¹ : LT α
      inst✝ : AddLeftStrictMono α
      a b a✝ : α
      h' h : LT.lt ↑b ↑a✝
      ⊢ LT.lt (HAdd.hAdd ↑a ↑b) (HAdd.hAdd ↑a ↑a✝)
    -/
  · exact coe_lt_coe.2 (add_lt_add_left (coe_lt_coe.1 h) _)
    /-
      🎉 no goals
    -/


protected theorem add_lt_add_right [LT α] [AddRightStrictMono α] (ha : a ≠ ⊤)
    (h : b < c) : b + a < c + a := by
  /-
    α : Type u
    inst✝² : Add α
    a b c : WithTop α
    inst✝¹ : LT α
    inst✝ : AddRightStrictMono α
    ha : Ne a Top.top
    h : LT.lt b c
    ⊢ LT.lt (HAdd.hAdd b a) (HAdd.hAdd c a)
  -/
  lift a to α using ha
  /-
    case intro
    α : Type u
    inst✝² : Add α
    b c : WithTop α
    inst✝¹ : LT α
    inst✝ : AddRightStrictMono α
    h : LT.lt b c
    a : α
    ⊢ LT.lt (HAdd.hAdd b ↑a) (HAdd.hAdd c ↑a)
  -/
  rcases lt_iff_exists_coe.1 h with ⟨b, rfl, h'⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝² : Add α
    c : WithTop α
    inst✝¹ : LT α
    inst✝ : AddRightStrictMono α
    a b : α
    h' h : LT.lt (↑b) c
    ⊢ LT.lt (HAdd.hAdd ↑b ↑a) (HAdd.hAdd c ↑a)
  -/
  cases c
    /-
      case intro.intro.intro.top
      α : Type u
      inst✝² : Add α
      inst✝¹ : LT α
      inst✝ : AddRightStrictMono α
      a b : α
      h' h : LT.lt (↑b) Top.top
      ⊢ LT.lt (HAdd.hAdd ↑b ↑a) (HAdd.hAdd Top.top ↑a)
    -/
  · exact coe_lt_top _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.coe
      α : Type u
      inst✝² : Add α
      inst✝¹ : LT α
      inst✝ : AddRightStrictMono α
      a b a✝ : α
      h' h : LT.lt ↑b ↑a✝
      ⊢ LT.lt (HAdd.hAdd ↑b ↑a) (HAdd.hAdd ↑a✝ ↑a)
    -/
  · exact coe_lt_coe.2 (add_lt_add_right (coe_lt_coe.1 h) _)
    /-
      🎉 no goals
    -/


protected theorem add_le_add_iff_left [LE α] [AddLeftMono α]
    [AddLeftReflectLE α] (ha : a ≠ ⊤) : a + b ≤ a + c ↔ b ≤ c :=
  ⟨WithTop.le_of_add_le_add_left ha, fun h => add_le_add_left h a⟩


protected theorem add_le_add_iff_right [LE α] [AddRightMono α]
    [AddRightReflectLE α] (ha : a ≠ ⊤) : b + a ≤ c + a ↔ b ≤ c :=
  ⟨WithTop.le_of_add_le_add_right ha, fun h => add_le_add_right h a⟩


protected theorem add_lt_add_iff_left [LT α] [AddLeftStrictMono α]
    [AddLeftReflectLT α] (ha : a ≠ ⊤) : a + b < a + c ↔ b < c :=
  ⟨lt_of_add_lt_add_left, WithTop.add_lt_add_left ha⟩


protected theorem add_lt_add_iff_right [LT α] [AddRightStrictMono α]
    [AddRightReflectLT α] (ha : a ≠ ⊤) : b + a < c + a ↔ b < c :=
  ⟨lt_of_add_lt_add_right, WithTop.add_lt_add_right ha⟩


protected theorem add_lt_add_of_le_of_lt [Preorder α] [AddLeftStrictMono α]
    [AddRightMono α] (ha : a ≠ ⊤) (hab : a ≤ b) (hcd : c < d) :
    a + c < b + d :=
  (WithTop.add_lt_add_left ha hcd).trans_le <| add_le_add_right hab _


protected theorem add_lt_add_of_lt_of_le [Preorder α] [AddLeftMono α]
    [AddRightStrictMono α] (hc : c ≠ ⊤) (hab : a < b) (hcd : c ≤ d) :
    a + c < b + d :=
  (WithTop.add_lt_add_right hc hab).trans_le <| add_le_add_left hcd _


lemma addLECancellable_of_ne_top [Preorder α] [ContravariantClass α α (· + ·) (· ≤ ·)]
    (ha : a ≠ ⊤) : AddLECancellable a := fun _b _c ↦ WithTop.le_of_add_le_add_left ha


lemma addLECancellable_of_lt_top [Preorder α] [ContravariantClass α α (· + ·) (· ≤ ·)]
    (ha : a < ⊤) : AddLECancellable a := addLECancellable_of_ne_top ha.ne


lemma addLECancellable_iff_ne_top [Nonempty α] [Preorder α]
    [ContravariantClass α α (· + ·) (· ≤ ·)] : AddLECancellable a ↔ a ≠ ⊤ where
           /-
             α : Type u
             inst✝³ : Add α
             a : WithTop α
             inst✝² : Nonempty α
             inst✝¹ : Preorder α
             inst✝ : ContravariantClass α α (fun x1 x2 => HAdd.hAdd x1 x2) fun x1 x2 => LE. …
             ⊢ AddLECancellable a → Ne a Top.top
           -/
  mp := by rintro h rfl; exact (coe_lt_top <| Classical.arbitrary _).not_le <| h <| by simp
                         /-
                           🎉 no goals
                         -/
  mpr := addLECancellable_of_ne_top

--  There is no `WithTop.map_mul_of_mulHom`, since `WithTop` does not have a multiplication.

@[simp]
protected theorem map_add {F} [Add β] [FunLike F α β] [AddHomClass F α β]
    (f : F) (a b : WithTop α) :
    (a + b).map f = a.map f + b.map f := by
  /-
    α : Type u
    β : Type v
    inst✝³ : Add α
    F : Type u_1
    inst✝² : Add β
    inst✝¹ : FunLike F α β
    inst✝ : AddHomClass F α β
    f : F
    a b : WithTop α
    ⊢ Eq (WithTop.map (⇑f) (HAdd.hAdd a b)) (HAdd.hAdd (WithTop.map (⇑f) a) (WithT …
  -/
  induction a
    /-
      case top
      α : Type u
      β : Type v
      inst✝³ : Add α
      F : Type u_1
      inst✝² : Add β
      inst✝¹ : FunLike F α β
      inst✝ : AddHomClass F α β
      f : F
      b : WithTop α
      ⊢ Eq (WithTop.map (⇑f) (HAdd.hAdd Top.top b)) (HAdd.hAdd (WithTop.map (⇑f) Top …
    -/
  · exact (top_add _).symm
    /-
      🎉 no goals
    -/
    /-
      case coe
      α : Type u
      β : Type v
      inst✝³ : Add α
      F : Type u_1
      inst✝² : Add β
      inst✝¹ : FunLike F α β
      inst✝ : AddHomClass F α β
      f : F
      b : WithTop α
      a✝ : α
      ⊢ Eq (WithTop.map (⇑f) (HAdd.hAdd (↑a✝) b)) (HAdd.hAdd (WithTop.map ⇑f ↑a✝) (W …
    -/
  · induction b
      /-
        case coe.top
        α : Type u
        β : Type v
        inst✝³ : Add α
        F : Type u_1
        inst✝² : Add β
        inst✝¹ : FunLike F α β
        inst✝ : AddHomClass F α β
        f : F
        a✝ : α
        ⊢ Eq (WithTop.map (⇑f) (HAdd.hAdd (↑a✝) Top.top)) (HAdd.hAdd (WithTop.map ⇑f ↑ …
      -/
    · exact (add_top _).symm
      /-
        🎉 no goals
      -/
      /-
        case coe.coe
        α : Type u
        β : Type v
        inst✝³ : Add α
        F : Type u_1
        inst✝² : Add β
        inst✝¹ : FunLike F α β
        inst✝ : AddHomClass F α β
        f : F
        a✝¹ a✝ : α
        ⊢ Eq (WithTop.map (⇑f) (HAdd.hAdd ↑a✝¹ ↑a✝)) (HAdd.hAdd (WithTop.map ⇑f ↑a✝¹)  …
      -/
    · rw [map_coe, map_coe, ← coe_add, ← coe_add, ← map_add]
      /-
        case coe.coe
        α : Type u
        β : Type v
        inst✝³ : Add α
        F : Type u_1
        inst✝² : Add β
        inst✝¹ : FunLike F α β
        inst✝ : AddHomClass F α β
        f : F
        a✝¹ a✝ : α
        ⊢ Eq (WithTop.map ⇑f ↑(HAdd.hAdd a✝¹ a✝)) ↑(f (HAdd.hAdd a✝¹ a✝))
      -/
      rfl
      /-
        🎉 no goals
      -/


instance addSemigroup [AddSemigroup α] : AddSemigroup (WithTop α) :=
  { WithTop.add with
    add_assoc := fun _ _ _ => Option.map₂_assoc add_assoc }


instance addCommSemigroup [AddCommSemigroup α] : AddCommSemigroup (WithTop α) :=
  { WithTop.addSemigroup with
    add_comm := fun _ _ => Option.map₂_comm add_comm }


instance addZeroClass [AddZeroClass α] : AddZeroClass (WithTop α) :=
  { WithTop.zero, WithTop.add with
    zero_add := Option.map₂_left_identity zero_add
    add_zero := Option.map₂_right_identity add_zero }


instance addMonoid : AddMonoid (WithTop α) where
  __ := WithTop.addSemigroup
  __ := WithTop.addZeroClass
  nsmul n a := match a, n with
    | (a : α), n => ↑(n • a)
    | ⊤, 0 => 0
    | ⊤, _n + 1 => ⊤
                     /-
                       α : Type u
                       β : Type v
                       inst✝ : AddMonoid α
                       a : WithTop α
                       ⊢ Eq ((fun n a => WithTop.addMonoid.match_1 (fun a n => WithTop α) a n (fun a  …
                     -/
                                 /-
                                   🎉 no goals
                                 -/
  nsmul_zero a := by cases a <;> simp [zero_nsmul]
                                 /-
                                   🎉 no goals
                                 -/
                       /-
                         α : Type u
                         β : Type v
                         inst✝ : AddMonoid α
                         n : Nat
                         a : WithTop α
                         ⊢ Eq ((fun n a => WithTop.addMonoid.match_1 (fun a n => WithTop α) a n (fun a  …
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
  nsmul_succ n a := by cases a <;> cases n <;> simp [succ_nsmul, coe_add]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp, norm_cast] lemma coe_nsmul (a : α) (n : ℕ) : ↑(n • a) = n • (a : WithTop α) := rfl


/-- Coercion from `α` to `WithTop α` as an `AddMonoidHom`. -/
def addHom : α →+ WithTop α where
  toFun := WithTop.some
  map_zero' := rfl
  map_add' _ _ := rfl


@[simp, norm_cast] lemma coe_addHom : ⇑(addHom : α →+ WithTop α) = WithTop.some := rfl


instance addCommMonoid [AddCommMonoid α] : AddCommMonoid (WithTop α) :=
  { WithTop.addMonoid, WithTop.addCommSemigroup with }


instance addMonoidWithOne : AddMonoidWithOne (WithTop α) :=
  { WithTop.one, WithTop.addMonoid with
    natCast := fun n => ↑(n : α),
    natCast_zero := by
      /-
        α : Type u
        β : Type v
        inst✝ : AddMonoidWithOne α
        ⊢ Eq (NatCast.natCast 0) 0
      -/
      simp only -- Porting note: Had to add this...?
      /-
        α : Type u
        β : Type v
        inst✝ : AddMonoidWithOne α
        ⊢ Eq (↑↑0) 0
      -/
      rw [Nat.cast_zero, WithTop.coe_zero],
      /-
        🎉 no goals
      -/
    natCast_succ := fun n => by
      /-
        α : Type u
        β : Type v
        inst✝ : AddMonoidWithOne α
        n : Nat
        ⊢ Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (NatCast.natCast n) 1)
      -/
      simp only -- Porting note: Had to add this...?
      /-
        α : Type u
        β : Type v
        inst✝ : AddMonoidWithOne α
        n : Nat
        ⊢ Eq (↑↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑↑n) 1)
      -/
      rw [Nat.cast_add_one, WithTop.coe_add, WithTop.coe_one] }
      /-
        🎉 no goals
      -/


@[simp, norm_cast] lemma coe_natCast (n : ℕ) : ((n : α) : WithTop α) = n := rfl


@[simp] lemma top_ne_natCast (n : ℕ) : (⊤ : WithTop α) ≠ n := top_ne_coe

@[simp] lemma natCast_ne_top (n : ℕ) : (n : WithTop α) ≠ ⊤ := coe_ne_top

@[simp] lemma natCast_lt_top [LT α] (n : ℕ) : (n : WithTop α) < ⊤ := coe_lt_top _


@[deprecated (since := "2024-04-05")] alias coe_nat := coe_natCast

@[deprecated (since := "2024-04-05")] alias nat_ne_top := natCast_ne_top

@[deprecated (since := "2024-04-05")] alias top_ne_nat := top_ne_natCast


@[simp] lemma coe_ofNat (n : ℕ) [n.AtLeastTwo] :
    ((ofNat(n) : α) : WithTop α) = ofNat(n) := rfl

@[simp] lemma coe_eq_ofNat (n : ℕ) [n.AtLeastTwo] (m : α) :
    (m : WithTop α) = ofNat(n) ↔ m = ofNat(n) :=
  coe_eq_coe

@[simp] lemma ofNat_eq_coe (n : ℕ) [n.AtLeastTwo] (m : α) :
    ofNat(n) = (m : WithTop α) ↔ ofNat(n) = m :=
  coe_eq_coe

@[simp] lemma ofNat_ne_top (n : ℕ) [n.AtLeastTwo] : (ofNat(n) : WithTop α) ≠ ⊤ :=
  natCast_ne_top n

@[simp] lemma top_ne_ofNat (n : ℕ) [n.AtLeastTwo] : (⊤ : WithTop α) ≠ ofNat(n) :=
  top_ne_natCast n


@[simp] lemma map_ofNat {f : α → β} (n : ℕ) [n.AtLeastTwo] :
    WithTop.map f (ofNat(n) : WithTop α) = f (ofNat(n)) := map_coe f n


@[simp] lemma map_natCast {f : α → β} (n : ℕ) :
    WithTop.map f (n : WithTop α) = f n := map_coe f n


lemma map_eq_ofNat_iff {f : β → α} {n : ℕ} [n.AtLeastTwo] {a : WithTop β} :
    a.map f = ofNat(n) ↔ ∃ x, a = .some x ∧ f x = n := map_eq_some_iff


lemma ofNat_eq_map_iff {f : β → α} {n : ℕ} [n.AtLeastTwo] {a : WithTop β} :
    ofNat(n) = a.map f ↔ ∃ x, a = .some x ∧ f x = n := some_eq_map_iff


lemma map_eq_natCast_iff {f : β → α} {n : ℕ} {a : WithTop β} :
    a.map f = n ↔ ∃ x, a = .some x ∧ f x = n := map_eq_some_iff


lemma natCast_eq_map_iff {f : β → α} {n : ℕ} {a : WithTop β} :
    n = a.map f ↔ ∃ x, a = .some x ∧ f x = n := some_eq_map_iff


instance charZero [AddMonoidWithOne α] [CharZero α] : CharZero (WithTop α) :=
  { cast_injective := Function.Injective.comp (f := Nat.cast (R := α))
      (fun _ _ => WithTop.coe_eq_coe.1) Nat.cast_injective}


instance addCommMonoidWithOne [AddCommMonoidWithOne α] : AddCommMonoidWithOne (WithTop α) :=
  { WithTop.addMonoidWithOne, WithTop.addCommMonoid with }

-- instance orderedAddCommMonoid [OrderedAddCommMonoid α] : OrderedAddCommMonoid (WithTop α) where
--   add_le_add_left _ _ := add_le_add_left
--
-- instance linearOrderedAddCommMonoidWithTop [LinearOrderedAddCommMonoid α] :
--     LinearOrderedAddCommMonoidWithTop (WithTop α) :=
--   { WithTop.orderTop, WithTop.linearOrder, WithTop.orderedAddCommMonoid with
--     top_add' := WithTop.top_add }
--

instance existsAddOfLE [LE α] [Add α] [ExistsAddOfLE α] : ExistsAddOfLE (WithTop α) :=
  ⟨fun {a} {b} =>
    match a, b with
                 /-
                   α : Type u
                   β : Type v
                   inst✝² : LE α
                   inst✝¹ : Add α
                   inst✝ : ExistsAddOfLE α
                   a b : WithTop α
                   ⊢ LE.le Top.top Top.top → Exists fun c => Eq Top.top (HAdd.hAdd Top.top c)
                 -/
    | ⊤, ⊤ => by simp
                 /-
                   🎉 no goals
                 -/
    | (a : α), ⊤ => fun _ => ⟨⊤, rfl⟩
    | (a : α), (b : α) => fun h => by
      /-
        α : Type u
        β : Type v
        inst✝² : LE α
        inst✝¹ : Add α
        inst✝ : ExistsAddOfLE α
        a✝ b✝ : WithTop α
        a b : α
        h : LE.le ↑a ↑b
        ⊢ Exists fun c => Eq (↑b) (HAdd.hAdd (↑a) c)
      -/
      obtain ⟨c, rfl⟩ := exists_add_of_le (WithTop.coe_le_coe.1 h)
      /-
        case intro
        α : Type u
        β : Type v
        inst✝² : LE α
        inst✝¹ : Add α
        inst✝ : ExistsAddOfLE α
        a✝ b : WithTop α
        a c : α
        h : LE.le ↑a ↑(HAdd.hAdd a c)
        ⊢ Exists fun c_1 => Eq (↑(HAdd.hAdd a c)) (HAdd.hAdd (↑a) c_1)
      -/
      exact ⟨c, rfl⟩
      /-
        🎉 no goals
      -/
    | ⊤, (b : α) => fun h => (not_top_le_coe _ h).elim⟩

-- instance canonicallyOrderedAddCommMonoid [CanonicallyOrderedAddCommMonoid α] :
--     CanonicallyOrderedAddCommMonoid (WithTop α) :=
--   { WithTop.orderBot, WithTop.orderedAddCommMonoid, WithTop.existsAddOfLE with
--     le_self_add := fun a b =>
--       match a, b with
--       | ⊤, ⊤ => le_rfl
--       | (a : α), ⊤ => le_top
--       | (a : α), (b : α) => WithTop.coe_le_coe.2 le_self_add
--       | ⊤, (b : α) => le_rfl }
--
-- instance [CanonicallyLinearOrderedAddCommMonoid α] :
--     CanonicallyLinearOrderedAddCommMonoid (WithTop α) :=
--   { WithTop.canonicallyOrderedAddCommMonoid, WithTop.linearOrder with }


@[to_additive (attr := simp) top_pos]
theorem one_lt_top [One α] [LT α] : (1 : WithTop α) < ⊤ := coe_lt_top _


@[deprecated top_pos (since := "2024-10-22")]
alias zero_lt_top := top_pos


@[norm_cast, deprecated coe_pos (since := "2024-10-22")]
theorem zero_lt_coe [Zero α] [LT α] (a : α) : (0 : WithTop α) < a ↔ 0 < a :=
  coe_lt_coe


/-- A version of `WithTop.map` for `OneHom`s. -/
@[to_additive (attr := simps (config := .asFn))
  "A version of `WithTop.map` for `ZeroHom`s"]
protected def _root_.OneHom.withTopMap {M N : Type*} [One M] [One N] (f : OneHom M N) :
    OneHom (WithTop M) (WithTop N) where
  toFun := WithTop.map f
                 /-
                   α : Type u
                   β : Type v
                   M : Type u_1
                   N : Type u_2
                   inst✝¹ : One M
                   inst✝ : One N
                   f : OneHom M N
                   ⊢ Eq (WithTop.map (⇑f) 1) 1
                 -/
  map_one' := by rw [WithTop.map_one, map_one, coe_one]
                 /-
                   🎉 no goals
                 -/


/-- A version of `WithTop.map` for `AddHom`s. -/
@[simps (config := .asFn)]
protected def _root_.AddHom.withTopMap {M N : Type*} [Add M] [Add N] (f : AddHom M N) :
    AddHom (WithTop M) (WithTop N) where
  toFun := WithTop.map f
  map_add' := WithTop.map_add f


/-- A version of `WithTop.map` for `AddMonoidHom`s. -/
@[simps (config := .asFn)]
protected def _root_.AddMonoidHom.withTopMap {M N : Type*} [AddZeroClass M] [AddZeroClass N]
    (f : M →+ N) : WithTop M →+ WithTop N :=
  { ZeroHom.withTopMap f.toZeroHom, AddHom.withTopMap f.toAddHom with toFun := WithTop.map f }


@[to_additive] instance one : One (WithBot α) := WithTop.one


@[to_additive (attr := simp, norm_cast)] lemma coe_one : ((1 : α) : WithBot α) = 1 := rfl


@[to_additive (attr := simp, norm_cast)]
lemma coe_eq_one : (a : WithBot α) = 1 ↔ a = 1 := coe_eq_coe


@[to_additive (attr := simp, norm_cast)]
lemma one_eq_coe : 1 = (a : WithBot α) ↔ a = 1 := eq_comm.trans coe_eq_one


@[to_additive (attr := simp)] lemma bot_ne_one : (⊥ : WithBot α) ≠ 1 := bot_ne_coe

@[to_additive (attr := simp)] lemma one_ne_bot : (1 : WithBot α) ≠ ⊥ := coe_ne_bot


@[to_additive (attr := simp)]
theorem unbot_one : (1 : WithBot α).unbot coe_ne_bot = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem unbot_one' (d : α) : (1 : WithBot α).unbot' d = 1 :=
  rfl


@[to_additive (attr := simp, norm_cast) coe_nonneg]
theorem one_le_coe [LE α] : 1 ≤ (a : WithBot α) ↔ 1 ≤ a := coe_le_coe


@[to_additive (attr := simp, norm_cast) coe_le_zero]
theorem coe_le_one [LE α] : (a : WithBot α) ≤ 1 ↔ a ≤ 1 := coe_le_coe


@[to_additive (attr := simp, norm_cast) coe_pos]
theorem one_lt_coe [LT α] : 1 < (a : WithBot α) ↔ 1 < a := coe_lt_coe


@[to_additive (attr := simp, norm_cast) coe_lt_zero]
theorem coe_lt_one [LT α] : (a : WithBot α) < 1 ↔ a < 1 := coe_lt_coe


@[to_additive (attr := simp)]
protected theorem map_one {β} (f : α → β) : (1 : WithBot α).map f = (f 1 : WithBot β) :=
  rfl


@[to_additive]
theorem map_eq_one_iff {α} {f : α → β} {v : WithBot α} [One β] :
    WithBot.map f v = 1 ↔ ∃ x, v = .some x ∧ f x = 1 := map_eq_some_iff


@[to_additive]
theorem one_eq_map_iff {α} {f : α → β} {v : WithBot α} [One β] :
    1 = WithBot.map f v ↔ ∃ x, v = .some x ∧ f x = 1 := some_eq_map_iff


instance zeroLEOneClass [Zero α] [LE α] [ZeroLEOneClass α] : ZeroLEOneClass (WithBot α) :=
  ⟨coe_le_coe.2 zero_le_one⟩


instance add [Add α] : Add (WithBot α) :=
  WithTop.add


instance AddSemigroup [AddSemigroup α] : AddSemigroup (WithBot α) :=
  WithTop.addSemigroup


instance addCommSemigroup [AddCommSemigroup α] : AddCommSemigroup (WithBot α) :=
  WithTop.addCommSemigroup


instance addZeroClass [AddZeroClass α] : AddZeroClass (WithBot α) :=
  WithTop.addZeroClass


instance addMonoid : AddMonoid (WithBot α) := WithTop.addMonoid


/-- Coercion from `α` to `WithBot α` as an `AddMonoidHom`. -/
def addHom : α →+ WithBot α where
  toFun := WithTop.some
  map_zero' := rfl
  map_add' _ _ := rfl


@[simp, norm_cast] lemma coe_addHom : ⇑(addHom : α →+ WithBot α) = WithBot.some := rfl


@[simp, norm_cast]
lemma coe_nsmul (a : α) (n : ℕ) : ↑(n • a) = n • (a : WithBot α) :=
  (addHom : α →+ WithBot α).map_nsmul _ _


instance addCommMonoid [AddCommMonoid α] : AddCommMonoid (WithBot α) :=
  WithTop.addCommMonoid


instance addMonoidWithOne : AddMonoidWithOne (WithBot α) := WithTop.addMonoidWithOne


@[norm_cast] lemma coe_natCast (n : ℕ) : ((n : α) : WithBot α) = n := rfl


@[simp] lemma natCast_ne_bot (n : ℕ) : (n : WithBot α) ≠ ⊥ := coe_ne_bot


@[simp] lemma bot_ne_natCast (n : ℕ) : (⊥ : WithBot α) ≠ n := bot_ne_coe


@[deprecated (since := "2024-04-05")] alias nat_ne_bot := natCast_ne_bot

@[deprecated (since := "2024-04-05")] alias bot_ne_nat := bot_ne_natCast


@[simp] lemma coe_ofNat (n : ℕ) [n.AtLeastTwo] :
    ((ofNat(n) : α) : WithBot α) = ofNat(n) := rfl

@[simp] lemma coe_eq_ofNat (n : ℕ) [n.AtLeastTwo] (m : α) :
    (m : WithBot α) = ofNat(n) ↔ m = ofNat(n) :=
  coe_eq_coe

@[simp] lemma ofNat_eq_coe (n : ℕ) [n.AtLeastTwo] (m : α) :
    ofNat(n) = (m : WithBot α) ↔ ofNat(n) = m :=
  coe_eq_coe

@[simp] lemma ofNat_ne_bot (n : ℕ) [n.AtLeastTwo] : (ofNat(n) : WithBot α) ≠ ⊥ :=
  natCast_ne_bot n

@[simp] lemma bot_ne_ofNat (n : ℕ) [n.AtLeastTwo] : (⊥ : WithBot α) ≠ ofNat(n) :=
  bot_ne_natCast n


@[simp] lemma map_ofNat {f : α → β} (n : ℕ) [n.AtLeastTwo] :
    WithBot.map f (ofNat(n) : WithBot α) = f ofNat(n) := map_coe f n


@[simp] lemma map_natCast {f : α → β} (n : ℕ) :
    WithBot.map f (n : WithBot α) = f n := map_coe f n


lemma map_eq_ofNat_iff {f : β → α} {n : ℕ} [n.AtLeastTwo] {a : WithBot β} :
    a.map f = ofNat(n) ↔ ∃ x, a = .some x ∧ f x = n := map_eq_some_iff


lemma ofNat_eq_map_iff {f : β → α} {n : ℕ} [n.AtLeastTwo] {a : WithBot β} :
    ofNat(n) = a.map f ↔ ∃ x, a = .some x ∧ f x = n := some_eq_map_iff


lemma map_eq_natCast_iff {f : β → α} {n : ℕ} {a : WithBot β} :
    a.map f = n ↔ ∃ x, a = .some x ∧ f x = n := map_eq_some_iff


lemma natCast_eq_map_iff {f : β → α} {n : ℕ} {a : WithBot β} :
    n = a.map f ↔ ∃ x, a = .some x ∧ f x = n := some_eq_map_iff


instance charZero [AddMonoidWithOne α] [CharZero α] : CharZero (WithBot α) :=
  WithTop.charZero


instance addCommMonoidWithOne [AddCommMonoidWithOne α] : AddCommMonoidWithOne (WithBot α) :=
  WithTop.addCommMonoidWithOne


@[simp, norm_cast]
theorem coe_add (a b : α) : ((a + b : α) : WithBot α) = a + b :=
  rfl


@[simp]
theorem bot_add (a : WithBot α) : ⊥ + a = ⊥ :=
  rfl


@[simp]
                                                  /-
                                                    α : Type u
                                                    inst✝ : Add α
                                                    a : WithBot α
                                                    ⊢ Eq (HAdd.hAdd a Bot.bot) Bot.bot
                                                  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
theorem add_bot (a : WithBot α) : a + ⊥ = ⊥ := by cases a <;> rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem add_eq_bot : a + b = ⊥ ↔ a = ⊥ ∨ b = ⊥ :=
  WithTop.add_eq_top


theorem add_ne_bot : a + b ≠ ⊥ ↔ a ≠ ⊥ ∧ b ≠ ⊥ :=
  WithTop.add_ne_top


theorem bot_lt_add [LT α] {a b : WithBot α} : ⊥ < a + b ↔ ⊥ < a ∧ ⊥ < b :=
  WithTop.add_lt_top (α := αᵒᵈ)


theorem add_eq_coe : a + b = x ↔ ∃ a' b' : α, ↑a' = a ∧ ↑b' = b ∧ a' + b' = x :=
  WithTop.add_eq_coe


theorem add_coe_eq_bot_iff : a + y = ⊥ ↔ a = ⊥ :=
  WithTop.add_coe_eq_top_iff


theorem coe_add_eq_bot_iff : ↑x + b = ⊥ ↔ b = ⊥ :=
  WithTop.coe_add_eq_top_iff


theorem add_right_cancel_iff [IsRightCancelAdd α] (ha : a ≠ ⊥) : b + a = c + a ↔ b = c :=
  WithTop.add_right_cancel_iff ha


theorem add_right_cancel [IsRightCancelAdd α] (ha : a ≠ ⊥) (h : b + a = c + a) : b = c :=
  WithTop.add_right_cancel ha h


theorem add_left_cancel_iff [IsLeftCancelAdd α] (ha : a ≠ ⊥) : a + b = a + c ↔ b = c :=
  WithTop.add_left_cancel_iff ha


theorem add_left_cancel [IsLeftCancelAdd α] (ha : a ≠ ⊥) (h : a + b = a + c) : b = c :=
  WithTop.add_left_cancel ha h

-- There is no `WithBot.map_mul_of_mulHom`, since `WithBot` does not have a multiplication.

@[simp]
protected theorem map_add {F} [Add β] [FunLike F α β] [AddHomClass F α β]
    (f : F) (a b : WithBot α) :
    (a + b).map f = a.map f + b.map f :=
  WithTop.map_add f a b


/-- A version of `WithBot.map` for `OneHom`s. -/
@[to_additive (attr := simps (config := .asFn))
  "A version of `WithBot.map` for `ZeroHom`s"]
protected def _root_.OneHom.withBotMap {M N : Type*} [One M] [One N] (f : OneHom M N) :
    OneHom (WithBot M) (WithBot N) where
  toFun := WithBot.map f
                 /-
                   α : Type u
                   β : Type v
                   inst✝² : Add α
                   a b c d : WithBot α
                   x y : α
                   M : Type u_1
                   N : Type u_2
                   inst✝¹ : One M
                   inst✝ : One N
                   f : OneHom M N
                   ⊢ Eq (WithBot.map (⇑f) 1) 1
                 -/
  map_one' := by rw [WithBot.map_one, map_one, coe_one]
                 /-
                   🎉 no goals
                 -/


/-- A version of `WithBot.map` for `AddHom`s. -/
@[simps (config := .asFn)]
protected def _root_.AddHom.withBotMap {M N : Type*} [Add M] [Add N] (f : AddHom M N) :
    AddHom (WithBot M) (WithBot N) where
  toFun := WithBot.map f
  map_add' := WithBot.map_add f


/-- A version of `WithBot.map` for `AddMonoidHom`s. -/
@[simps (config := .asFn)]
protected def _root_.AddMonoidHom.withBotMap {M N : Type*} [AddZeroClass M] [AddZeroClass N]
    (f : M →+ N) : WithBot M →+ WithBot N :=
  { ZeroHom.withBotMap f.toZeroHom, AddHom.withBotMap f.toAddHom with toFun := WithBot.map f }


instance addLeftMono [AddLeftMono α] : AddLeftMono (WithBot α) :=
  OrderDual.addLeftMono (α := WithTop αᵒᵈ)


instance addRightMono [AddRightMono α] : AddRightMono (WithBot α) :=
  OrderDual.addRightMono (α := WithTop αᵒᵈ)


instance addLeftReflectLT [AddLeftReflectLT α] : AddLeftReflectLT (WithBot α) :=
  OrderDual.addLeftReflectLT (α := WithTop αᵒᵈ)


instance addRightReflectLT [AddRightReflectLT α] : AddRightReflectLT (WithBot α) :=
  OrderDual.addRightReflectLT (α := WithTop αᵒᵈ)


protected theorem le_of_add_le_add_left [AddLeftReflectLE α] (ha : a ≠ ⊥)
    (h : a + b ≤ a + c) : b ≤ c :=
  WithTop.le_of_add_le_add_left (α := αᵒᵈ) ha h


protected theorem le_of_add_le_add_right [AddRightReflectLE α]
    (ha : a ≠ ⊥) (h : b + a ≤ c + a) : b ≤ c :=
  WithTop.le_of_add_le_add_right (α := αᵒᵈ) ha h


protected theorem add_lt_add_left [AddLeftStrictMono α] (ha : a ≠ ⊥) (h : b < c) :
    a + b < a + c :=
  WithTop.add_lt_add_left (α := αᵒᵈ) ha h


protected theorem add_lt_add_right [AddRightStrictMono α] (ha : a ≠ ⊥)
    (h : b < c) : b + a < c + a :=
  WithTop.add_lt_add_right (α := αᵒᵈ) ha h


protected theorem add_le_add_iff_left [AddLeftMono α]
    [AddLeftReflectLE α] (ha : a ≠ ⊥) : a + b ≤ a + c ↔ b ≤ c :=
  ⟨WithBot.le_of_add_le_add_left ha, fun h => add_le_add_left h a⟩


protected theorem add_le_add_iff_right [AddRightMono α]
    [AddRightReflectLE α] (ha : a ≠ ⊥) : b + a ≤ c + a ↔ b ≤ c :=
  ⟨WithBot.le_of_add_le_add_right ha, fun h => add_le_add_right h a⟩


protected theorem add_lt_add_iff_left [AddLeftStrictMono α]
    [AddLeftReflectLT α] (ha : a ≠ ⊥) : a + b < a + c ↔ b < c :=
  ⟨lt_of_add_lt_add_left, WithBot.add_lt_add_left ha⟩


protected theorem add_lt_add_iff_right [AddRightStrictMono α]
    [AddRightReflectLT α] (ha : a ≠ ⊥) : b + a < c + a ↔ b < c :=
  ⟨lt_of_add_lt_add_right, WithBot.add_lt_add_right ha⟩


protected theorem add_lt_add_of_le_of_lt [AddLeftStrictMono α]
    [AddRightMono α] (hb : b ≠ ⊥) (hab : a ≤ b) (hcd : c < d) :
    a + c < b + d :=
  WithTop.add_lt_add_of_le_of_lt (α := αᵒᵈ) hb hab hcd


protected theorem add_lt_add_of_lt_of_le [AddLeftMono α]
    [AddRightStrictMono α] (hd : d ≠ ⊥) (hab : a < b) (hcd : c ≤ d) :
    a + c < b + d :=
  WithTop.add_lt_add_of_lt_of_le (α := αᵒᵈ) hd hab hcd


