/-- Disjoint sum of multisets. -/
def disjSum : Multiset (α ⊕ β) :=
  s.map inl + t.map inr


@[simp]
theorem zero_disjSum : (0 : Multiset α).disjSum t = t.map inr :=
  zero_add _


@[simp]
theorem disjSum_zero : s.disjSum (0 : Multiset β) = s.map inl :=
  add_zero _


@[simp]
theorem card_disjSum : Multiset.card (s.disjSum t) = Multiset.card s + Multiset.card t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    ⊢ Eq (s.disjSum t).card (HAdd.hAdd s.card t.card)
  -/
  rw [disjSum, card_add, card_map, card_map]
  /-
    🎉 no goals
  -/


theorem mem_disjSum : x ∈ s.disjSum t ↔ (∃ a, a ∈ s ∧ inl a = x) ∨ ∃ b, b ∈ t ∧ inr b = x := by
  /-
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    x : Sum α β
    ⊢ Iff (Membership.mem (s.disjSum t) x) (Or (Exists fun a => And (Membership.me …
  -/
  simp_rw [disjSum, mem_add, mem_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem inl_mem_disjSum : inl a ∈ s.disjSum t ↔ a ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    a : α
    ⊢ Iff (Membership.mem (s.disjSum t) (Sum.inl a)) (Membership.mem s a)
  -/
  rw [mem_disjSum, or_iff_left]
  -- Porting note: Previous code for L62 was: simp only [exists_eq_right]
    /-
      α : Type u_1
      β : Type u_2
      s : Multiset α
      t : Multiset β
      a : α
      ⊢ Iff (Exists fun a_1 => And (Membership.mem s a_1) (Eq (Sum.inl a_1) (Sum.inl …
    -/
  · simp only [inl.injEq, exists_eq_right]
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    a : α
    ⊢ Not (Exists fun b => And (Membership.mem t b) (Eq (Sum.inr b) (Sum.inl a)))
  -/
  rintro ⟨b, _, hb⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    a : α
    b : β
    left✝ : Membership.mem t b
    hb : Eq (Sum.inr b) (Sum.inl a)
    ⊢ False
  -/
  exact inr_ne_inl hb
  /-
    🎉 no goals
  -/


@[simp]
theorem inr_mem_disjSum : inr b ∈ s.disjSum t ↔ b ∈ t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    b : β
    ⊢ Iff (Membership.mem (s.disjSum t) (Sum.inr b)) (Membership.mem t b)
  -/
  rw [mem_disjSum, or_iff_right]
  -- Porting note: Previous code for L72 was: simp only [exists_eq_right]
    /-
      α : Type u_1
      β : Type u_2
      s : Multiset α
      t : Multiset β
      b : β
      ⊢ Iff (Exists fun b_1 => And (Membership.mem t b_1) (Eq (Sum.inr b_1) (Sum.inr …
    -/
  · simp only [inr.injEq, exists_eq_right]
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    b : β
    ⊢ Not (Exists fun a => And (Membership.mem s a) (Eq (Sum.inl a) (Sum.inr b)))
  -/
  rintro ⟨a, _, ha⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    b : β
    a : α
    left✝ : Membership.mem s a
    ha : Eq (Sum.inl a) (Sum.inr b)
    ⊢ False
  -/
  exact inl_ne_inr ha
  /-
    🎉 no goals
  -/


theorem disjSum_mono (hs : s₁ ≤ s₂) (ht : t₁ ≤ t₂) : s₁.disjSum t₁ ≤ s₂.disjSum t₂ :=
  add_le_add (map_le_map hs) (map_le_map ht)


theorem disjSum_mono_left (t : Multiset β) : Monotone fun s : Multiset α => s.disjSum t :=
  fun _ _ hs => add_le_add_right (map_le_map hs) _


theorem disjSum_mono_right (s : Multiset α) :
    Monotone (s.disjSum : Multiset β → Multiset (α ⊕ β)) := fun _ _ ht =>
  add_le_add_left (map_le_map ht) _


theorem disjSum_lt_disjSum_of_lt_of_le (hs : s₁ < s₂) (ht : t₁ ≤ t₂) :
    s₁.disjSum t₁ < s₂.disjSum t₂ :=
  add_lt_add_of_lt_of_le (map_lt_map hs) (map_le_map ht)


theorem disjSum_lt_disjSum_of_le_of_lt (hs : s₁ ≤ s₂) (ht : t₁ < t₂) :
    s₁.disjSum t₁ < s₂.disjSum t₂ :=
  add_lt_add_of_le_of_lt (map_le_map hs) (map_lt_map ht)


theorem disjSum_strictMono_left (t : Multiset β) : StrictMono fun s : Multiset α => s.disjSum t :=
  fun _ _ hs => disjSum_lt_disjSum_of_lt_of_le hs le_rfl


theorem disjSum_strictMono_right (s : Multiset α) :
    StrictMono (s.disjSum : Multiset β → Multiset (α ⊕ β)) := fun _ _ =>
  disjSum_lt_disjSum_of_le_of_lt le_rfl


protected theorem Nodup.disjSum (hs : s.Nodup) (ht : t.Nodup) : (s.disjSum t).Nodup := by
  /-
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    hs : s.Nodup
    ht : t.Nodup
    ⊢ (s.disjSum t).Nodup
  -/
  refine ((hs.map inl_injective).add_iff <| ht.map inr_injective).2 ?_
  /-
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    hs : s.Nodup
    ht : t.Nodup
    ⊢ Disjoint (Multiset.map Sum.inl s) (Multiset.map Sum.inr t)
  -/
  rw [disjoint_map_map]
  /-
    α : Type u_1
    β : Type u_2
    s : Multiset α
    t : Multiset β
    hs : s.Nodup
    ht : t.Nodup
    ⊢ ∀ (a : α), Membership.mem s a → ∀ (b : β), Membership.mem t b → Ne (Sum.inl  …
  -/
  exact fun _ _ _ _ ↦ inr_ne_inl.symm
  /-
    🎉 no goals
  -/


