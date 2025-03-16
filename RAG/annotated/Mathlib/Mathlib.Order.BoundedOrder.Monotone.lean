theorem StrictMono.apply_eq_top_iff (hf : StrictMono f) : f a = f ⊤ ↔ a = ⊤ :=
  ⟨fun h => not_lt_top_iff.1 fun ha => (hf ha).ne h, congr_arg _⟩


theorem StrictAnti.apply_eq_top_iff (hf : StrictAnti f) : f a = f ⊤ ↔ a = ⊤ :=
  ⟨fun h => not_lt_top_iff.1 fun ha => (hf ha).ne' h, congr_arg _⟩


theorem StrictMono.maximal_preimage_top [LinearOrder α] [Preorder β] [OrderTop β] {f : α → β}
    (H : StrictMono f) {a} (h_top : f a = ⊤) (x : α) : x ≤ a :=
  H.maximal_of_maximal_image
    (fun p => by
      /-
        α : Type u
        β : Type v
        inst✝² : LinearOrder α
        inst✝¹ : Preorder β
        inst✝ : OrderTop β
        f : α → β
        H : StrictMono f
        a : α
        h_top : Eq (f a) Top.top
        x : α
        p : β
        ⊢ LE.le p (f a)
      -/
      rw [h_top]
      /-
        α : Type u
        β : Type v
        inst✝² : LinearOrder α
        inst✝¹ : Preorder β
        inst✝ : OrderTop β
        f : α → β
        H : StrictMono f
        a : α
        h_top : Eq (f a) Top.top
        x : α
        p : β
        ⊢ LE.le p Top.top
      -/
      exact le_top)
      /-
        🎉 no goals
      -/
    x


theorem StrictMono.apply_eq_bot_iff (hf : StrictMono f) : f a = f ⊥ ↔ a = ⊥ :=
  hf.dual.apply_eq_top_iff


theorem StrictAnti.apply_eq_bot_iff (hf : StrictAnti f) : f a = f ⊥ ↔ a = ⊥ :=
  hf.dual.apply_eq_top_iff


theorem StrictMono.minimal_preimage_bot [LinearOrder α] [PartialOrder β] [OrderBot β] {f : α → β}
    (H : StrictMono f) {a} (h_bot : f a = ⊥) (x : α) : a ≤ x :=
  H.minimal_of_minimal_image
    (fun p => by
      /-
        α : Type u
        β : Type v
        inst✝² : LinearOrder α
        inst✝¹ : PartialOrder β
        inst✝ : OrderBot β
        f : α → β
        H : StrictMono f
        a : α
        h_bot : Eq (f a) Bot.bot
        x : α
        p : β
        ⊢ LE.le (f a) p
      -/
      rw [h_bot]
      /-
        α : Type u
        β : Type v
        inst✝² : LinearOrder α
        inst✝¹ : PartialOrder β
        inst✝ : OrderBot β
        f : α → β
        H : StrictMono f
        a : α
        h_bot : Eq (f a) Bot.bot
        x : α
        p : β
        ⊢ LE.le Bot.bot p
      -/
      exact bot_le)
      /-
        🎉 no goals
      -/
    x


theorem monotone_and {p q : α → Prop} (m_p : Monotone p) (m_q : Monotone q) :
    Monotone fun x => p x ∧ q x :=
  fun _ _ h => And.imp (m_p h) (m_q h)

-- Note: by finish [monotone] doesn't work

theorem monotone_or {p q : α → Prop} (m_p : Monotone p) (m_q : Monotone q) :
    Monotone fun x => p x ∨ q x :=
  fun _ _ h => Or.imp (m_p h) (m_q h)


theorem monotone_le {x : α} : Monotone (x ≤ ·) := fun _ _ h' h => h.trans h'


theorem monotone_lt {x : α} : Monotone (x < ·) := fun _ _ h' h => h.trans_le h'


theorem antitone_le {x : α} : Antitone (· ≤ x) := fun _ _ h' h => h'.trans h


theorem antitone_lt {x : α} : Antitone (· < x) := fun _ _ h' h => h'.trans_lt h


theorem Monotone.forall {P : β → α → Prop} (hP : ∀ x, Monotone (P x)) :
    Monotone fun y => ∀ x, P x y :=
  fun _ _ hy h x => hP x hy <| h x


theorem Antitone.forall {P : β → α → Prop} (hP : ∀ x, Antitone (P x)) :
    Antitone fun y => ∀ x, P x y :=
  fun _ _ hy h x => hP x hy (h x)


theorem Monotone.ball {P : β → α → Prop} {s : Set β} (hP : ∀ x ∈ s, Monotone (P x)) :
    Monotone fun y => ∀ x ∈ s, P x y := fun _ _ hy h x hx => hP x hx hy (h x hx)


theorem Antitone.ball {P : β → α → Prop} {s : Set β} (hP : ∀ x ∈ s, Antitone (P x)) :
    Antitone fun y => ∀ x ∈ s, P x y := fun _ _ hy h x hx => hP x hx hy (h x hx)


theorem Monotone.exists {P : β → α → Prop} (hP : ∀ x, Monotone (P x)) :
    Monotone fun y => ∃ x, P x y :=
  fun _ _ hy ⟨x, hx⟩ ↦ ⟨x, hP x hy hx⟩


theorem Antitone.exists {P : β → α → Prop} (hP : ∀ x, Antitone (P x)) :
    Antitone fun y => ∃ x, P x y :=
  fun _ _ hy ⟨x, hx⟩ ↦ ⟨x, hP x hy hx⟩


theorem forall_ge_iff {P : α → Prop} {x₀ : α} (hP : Monotone P) :
    (∀ x ≥ x₀, P x) ↔ P x₀ :=
  ⟨fun H ↦ H x₀ le_rfl, fun H _ hx ↦ hP hx H⟩


theorem forall_le_iff {P : α → Prop} {x₀ : α} (hP : Antitone P) :
    (∀ x ≤ x₀, P x) ↔ P x₀ :=
  ⟨fun H ↦ H x₀ le_rfl, fun H _ hx ↦ hP hx H⟩


