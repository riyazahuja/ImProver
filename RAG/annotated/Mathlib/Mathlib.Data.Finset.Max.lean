/-- Let `s` be a finset in a linear order. Then `s.max` is the maximum of `s` if `s` is not empty,
and `⊥` otherwise. It belongs to `WithBot α`. If you want to get an element of `α`, see
`s.max'`. -/
protected def max (s : Finset α) : WithBot α :=
  sup s (↑)


theorem max_eq_sup_coe {s : Finset α} : s.max = s.sup (↑) :=
  rfl


theorem max_eq_sup_withBot (s : Finset α) : s.max = sup s (↑) :=
  rfl


@[simp]
theorem max_empty : (∅ : Finset α).max = ⊥ :=
  rfl


@[simp]
theorem max_insert {a : α} {s : Finset α} : (insert a s).max = max ↑a s.max :=
  fold_insert_idem


@[simp]
theorem max_singleton {a : α} : Finset.max {a} = (a : WithBot α) := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    a : α
    ⊢ Eq (Singleton.singleton a).max ↑a
  -/
  rw [← insert_emptyc_eq]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    a : α
    ⊢ Eq (Insert.insert a EmptyCollection.emptyCollection).max ↑a
  -/
  exact max_insert
  /-
    🎉 no goals
  -/


theorem max_of_mem {s : Finset α} {a : α} (h : a ∈ s) : ∃ b : α, s.max = b := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    a : α
    h : Membership.mem s a
    ⊢ Exists fun b => Eq s.max ↑b
  -/
  obtain ⟨b, h, _⟩ := le_sup (α := WithBot α) h _ rfl
  /-
    case intro.intro
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    a : α
    h✝ : Membership.mem s a
    b : α
    h : Eq (s.sup WithBot.some) ↑b
    right✝ : LE.le a b
    ⊢ Exists fun b => Eq s.max ↑b
  -/
  exact ⟨b, h⟩
  /-
    🎉 no goals
  -/


theorem max_of_nonempty {s : Finset α} (h : s.Nonempty) : ∃ a : α, s.max = a :=
  let ⟨_, h⟩ := h
  max_of_mem h


theorem max_eq_bot {s : Finset α} : s.max = ⊥ ↔ s = ∅ :=
  ⟨fun h ↦ s.eq_empty_or_nonempty.elim id fun H ↦ by
      /-
        α : Type u_2
        inst✝ : LinearOrder α
        s : Finset α
        h : Eq s.max Bot.bot
        H : s.Nonempty
        ⊢ Eq s EmptyCollection.emptyCollection
      -/
      obtain ⟨a, ha⟩ := max_of_nonempty H
      /-
        case intro
        α : Type u_2
        inst✝ : LinearOrder α
        s : Finset α
        h : Eq s.max Bot.bot
        H : s.Nonempty
        a : α
        ha : Eq s.max ↑a
        ⊢ Eq s EmptyCollection.emptyCollection
      -/
      rw [h] at ha; cases ha; , -- the `;` is needed since the `cases` syntax allows `cases a, b`
                    /-
                      🎉 no goals
                    -/
    fun h ↦ h.symm ▸ max_empty⟩


theorem mem_of_max {s : Finset α} : ∀ {a : α}, s.max = a → a ∈ s := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    ⊢ ∀ {a : α}, Eq s.max ↑a → Membership.mem s a
  -/
  induction' s using Finset.induction_on with b s _ ih
    /-
      case empty
      α : Type u_2
      inst✝ : LinearOrder α
      ⊢ ∀ {a : α}, Eq EmptyCollection.emptyCollection.max ↑a → Membership.mem EmptyC …
    -/
  · intro _ H; cases H
               /-
                 🎉 no goals
               -/
    /-
      case insert
      α : Type u_2
      inst✝ : LinearOrder α
      b : α
      s : Finset α
      a✝ : Not (Membership.mem s b)
      ih : ∀ {a : α}, Eq s.max ↑a → Membership.mem s a
      ⊢ ∀ {a : α}, Eq (Insert.insert b s).max ↑a → Membership.mem (Insert.insert b s …
    -/
  · intro a h
    /-
      case insert
      α : Type u_2
      inst✝ : LinearOrder α
      b : α
      s : Finset α
      a✝ : Not (Membership.mem s b)
      ih : ∀ {a : α}, Eq s.max ↑a → Membership.mem s a
      a : α
      h : Eq (Insert.insert b s).max ↑a
      ⊢ Membership.mem (Insert.insert b s) a
    -/
    by_cases p : b = a
      /-
        case pos
        α : Type u_2
        inst✝ : LinearOrder α
        b : α
        s : Finset α
        a✝ : Not (Membership.mem s b)
        ih : ∀ {a : α}, Eq s.max ↑a → Membership.mem s a
        a : α
        h : Eq (Insert.insert b s).max ↑a
        p : Eq b a
        ⊢ Membership.mem (Insert.insert b s) a
      -/
    · induction p
      /-
        case pos.refl
        α : Type u_2
        inst✝ : LinearOrder α
        b : α
        s : Finset α
        a✝ : Not (Membership.mem s b)
        ih : ∀ {a : α}, Eq s.max ↑a → Membership.mem s a
        a : α
        h : Eq (Insert.insert b s).max ↑b
        ⊢ Membership.mem (Insert.insert b s) b
      -/
      exact mem_insert_self b s
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_2
        inst✝ : LinearOrder α
        b : α
        s : Finset α
        a✝ : Not (Membership.mem s b)
        ih : ∀ {a : α}, Eq s.max ↑a → Membership.mem s a
        a : α
        h : Eq (Insert.insert b s).max ↑a
        p : Not (Eq b a)
        ⊢ Membership.mem (Insert.insert b s) a
      -/
    · cases' max_choice (↑b) s.max with q q <;> rw [max_insert, q] at h
        /-
          case neg.inl
          α : Type u_2
          inst✝ : LinearOrder α
          b : α
          s : Finset α
          a✝ : Not (Membership.mem s b)
          ih : ∀ {a : α}, Eq s.max ↑a → Membership.mem s a
          a : α
          h : Eq ↑b ↑a
          p : Not (Eq b a)
          q : Eq (Max.max (↑b) s.max) ↑b
          ⊢ Membership.mem (Insert.insert b s) a
        -/
      · cases h
        /-
          case neg.inl.refl
          α : Type u_2
          inst✝ : LinearOrder α
          b : α
          s : Finset α
          a✝ : Not (Membership.mem s b)
          ih : ∀ {a : α}, Eq s.max ↑a → Membership.mem s a
          q : Eq (Max.max (↑b) s.max) ↑b
          p : Not (Eq b b)
          ⊢ Membership.mem (Insert.insert b s) b
        -/
        cases p rfl
        /-
          🎉 no goals
        -/
        /-
          case neg.inr
          α : Type u_2
          inst✝ : LinearOrder α
          b : α
          s : Finset α
          a✝ : Not (Membership.mem s b)
          ih : ∀ {a : α}, Eq s.max ↑a → Membership.mem s a
          a : α
          h : Eq s.max ↑a
          p : Not (Eq b a)
          q : Eq (Max.max (↑b) s.max) s.max
          ⊢ Membership.mem (Insert.insert b s) a
        -/
      · exact mem_insert_of_mem (ih h)
        /-
          🎉 no goals
        -/


theorem le_max {a : α} {s : Finset α} (as : a ∈ s) : ↑a ≤ s.max :=
  le_sup as


theorem not_mem_of_max_lt_coe {a : α} {s : Finset α} (h : s.max < a) : a ∉ s :=
  mt le_max h.not_le


theorem le_max_of_eq {s : Finset α} {a b : α} (h₁ : a ∈ s) (h₂ : s.max = b) : a ≤ b :=
  WithBot.coe_le_coe.mp <| (le_max h₁).trans h₂.le


theorem not_mem_of_max_lt {s : Finset α} {a b : α} (h₁ : b < a) (h₂ : s.max = ↑b) : a ∉ s :=
  Finset.not_mem_of_max_lt_coe <| h₂.trans_lt <| WithBot.coe_lt_coe.mpr h₁


theorem max_union {s t : Finset α} : (s ∪ t).max = s.max ⊔ t.max := sup_union


@[gcongr]
theorem max_mono {s t : Finset α} (st : s ⊆ t) : s.max ≤ t.max :=
  sup_mono st


protected theorem max_le {M : WithBot α} {s : Finset α} (st : ∀ a ∈ s, (a : WithBot α) ≤ M) :
    s.max ≤ M :=
  Finset.sup_le st


@[simp]
protected lemma max_le_iff {m : WithBot α} {s : Finset α} : s.max ≤ m ↔ ∀ a ∈ s, a ≤ m :=
  Finset.sup_le_iff


@[simp]
protected lemma max_eq_top [OrderTop α] {s : Finset α} : s.max = ⊤ ↔ ⊤ ∈ s :=
                                    /-
                                      α : Type u_2
                                      inst✝¹ : LinearOrder α
                                      inst✝ : OrderTop α
                                      s : Finset α
                                      ⊢ Iff (Exists fun b => And (Membership.mem s b) (Eq (↑b) Top.top)) (Membership …
                                    -/
  Finset.sup_eq_top_iff.trans <| by simp
                                    /-
                                      🎉 no goals
                                    -/


/-- Let `s` be a finset in a linear order. Then `s.min` is the minimum of `s` if `s` is not empty,
and `⊤` otherwise. It belongs to `WithTop α`. If you want to get an element of `α`, see
`s.min'`. -/
protected def min (s : Finset α) : WithTop α :=
  inf s (↑)


theorem min_eq_inf_withTop (s : Finset α) : s.min = inf s (↑) :=
  rfl


@[simp]
theorem min_empty : (∅ : Finset α).min = ⊤ :=
  rfl


@[simp]
theorem min_insert {a : α} {s : Finset α} : (insert a s).min = min (↑a) s.min :=
  fold_insert_idem


@[simp]
theorem min_singleton {a : α} : Finset.min {a} = (a : WithTop α) := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    a : α
    ⊢ Eq (Singleton.singleton a).min ↑a
  -/
  rw [← insert_emptyc_eq]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    a : α
    ⊢ Eq (Insert.insert a EmptyCollection.emptyCollection).min ↑a
  -/
  exact min_insert
  /-
    🎉 no goals
  -/


theorem min_of_mem {s : Finset α} {a : α} (h : a ∈ s) : ∃ b : α, s.min = b := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    a : α
    h : Membership.mem s a
    ⊢ Exists fun b => Eq s.min ↑b
  -/
  obtain ⟨b, h, _⟩ := inf_le (α := WithTop α) h _ rfl
  /-
    case intro.intro
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    a : α
    h✝ : Membership.mem s a
    b : α
    h : Eq (s.inf WithTop.some) ↑b
    right✝ : LE.le b a
    ⊢ Exists fun b => Eq s.min ↑b
  -/
  exact ⟨b, h⟩
  /-
    🎉 no goals
  -/


theorem min_of_nonempty {s : Finset α} (h : s.Nonempty) : ∃ a : α, s.min = a :=
  let ⟨_, h⟩ := h
  min_of_mem h


@[simp]
theorem min_eq_top {s : Finset α} : s.min = ⊤ ↔ s = ∅ := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    ⊢ Iff (Eq s.min Top.top) (Eq s EmptyCollection.emptyCollection)
  -/
  simp [Finset.min, eq_empty_iff_forall_not_mem]
  /-
    🎉 no goals
  -/


theorem mem_of_min {s : Finset α} : ∀ {a : α}, s.min = a → a ∈ s :=
  @mem_of_max αᵒᵈ _ s


theorem min_le {a : α} {s : Finset α} (as : a ∈ s) : s.min ≤ a :=
  inf_le as


theorem not_mem_of_coe_lt_min {a : α} {s : Finset α} (h : ↑a < s.min) : a ∉ s :=
  mt min_le h.not_le


theorem min_le_of_eq {s : Finset α} {a b : α} (h₁ : b ∈ s) (h₂ : s.min = a) : a ≤ b :=
  WithTop.coe_le_coe.mp <| h₂.ge.trans (min_le h₁)


theorem not_mem_of_lt_min {s : Finset α} {a b : α} (h₁ : a < b) (h₂ : s.min = ↑b) : a ∉ s :=
  Finset.not_mem_of_coe_lt_min <| (WithTop.coe_lt_coe.mpr h₁).trans_eq h₂.symm


theorem min_union {s t : Finset α} : (s ∪ t).min = s.min ⊓ t.min := inf_union


@[gcongr]
theorem min_mono {s t : Finset α} (st : s ⊆ t) : t.min ≤ s.min :=
  inf_mono st


protected theorem le_min {m : WithTop α} {s : Finset α} (st : ∀ a : α, a ∈ s → m ≤ a) : m ≤ s.min :=
  Finset.le_inf st


@[simp]
protected theorem le_min_iff {m : WithTop α} {s : Finset α} : m ≤ s.min ↔ ∀ a ∈ s, m ≤ a :=
  Finset.le_inf_iff


@[simp]
protected theorem min_eq_bot [OrderBot α] {s : Finset α} : s.min = ⊥ ↔ ⊥ ∈ s :=
  Finset.max_eq_top (α := αᵒᵈ)


/-- Given a nonempty finset `s` in a linear order `α`, then `s.min' H` is its minimum, as an
element of `α`, where `H` is a proof of nonemptiness. Without this assumption, use instead `s.min`,
taking values in `WithTop α`. -/
def min' (s : Finset α) (H : s.Nonempty) : α :=
  inf' s H id


/-- Given a nonempty finset `s` in a linear order `α`, then `s.max' H` is its maximum, as an
element of `α`, where `H` is a proof of nonemptiness. Without this assumption, use instead `s.max`,
taking values in `WithBot α`. -/
def max' (s : Finset α) (H : s.Nonempty) : α :=
  sup' s H id


theorem min'_mem : s.min' H ∈ s :=
                   /-
                     α : Type u_2
                     inst✝ : LinearOrder α
                     s : Finset α
                     H : s.Nonempty
                     ⊢ Eq s.min ↑(s.min' H)
                   -/
  mem_of_min <| by simp only [Finset.min, min', id_eq, coe_inf', Function.comp_def]
                   /-
                     🎉 no goals
                   -/


theorem min'_le (x) (H2 : x ∈ s) : s.min' ⟨x, H2⟩ ≤ x :=
  min_le_of_eq H2 (WithTop.coe_untop _ _).symm


theorem le_min' (x) (H2 : ∀ y ∈ s, x ≤ y) : x ≤ s.min' H :=
  H2 _ <| min'_mem _ _


theorem isLeast_min' : IsLeast (↑s) (s.min' H) :=
  ⟨min'_mem _ _, min'_le _⟩


@[simp]
theorem le_min'_iff {x} : x ≤ s.min' H ↔ ∀ y ∈ s, x ≤ y :=
  le_isGLB_iff (isLeast_min' s H).isGLB


/-- `{a}.min' _` is `a`. -/
@[simp]
                                                                                        /-
                                                                                          α : Type u_2
                                                                                          inst✝ : LinearOrder α
                                                                                          a : α
                                                                                          ⊢ Eq ((Singleton.singleton a).min' ⋯) a
                                                                                        -/
theorem min'_singleton (a : α) : ({a} : Finset α).min' (singleton_nonempty _) = a := by simp [min']
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


theorem max'_mem : s.max' H ∈ s :=
                   /-
                     α : Type u_2
                     inst✝ : LinearOrder α
                     s : Finset α
                     H : s.Nonempty
                     ⊢ Eq s.max ↑(s.max' H)
                   -/
  mem_of_max <| by simp only [max', Finset.max, id_eq, coe_sup', Function.comp_def]
                   /-
                     🎉 no goals
                   -/


theorem le_max' (x) (H2 : x ∈ s) : x ≤ s.max' ⟨x, H2⟩ :=
  le_max_of_eq H2 (WithBot.coe_unbot _ _).symm


theorem max'_le (x) (H2 : ∀ y ∈ s, y ≤ x) : s.max' H ≤ x :=
  H2 _ <| max'_mem _ _


theorem isGreatest_max' : IsGreatest (↑s) (s.max' H) :=
  ⟨max'_mem _ _, le_max' _⟩


@[simp]
theorem max'_le_iff {x} : s.max' H ≤ x ↔ ∀ y ∈ s, y ≤ x :=
  isLUB_le_iff (isGreatest_max' s H).isLUB


@[simp]
theorem max'_lt_iff {x} : s.max' H < x ↔ ∀ y ∈ s, y < x :=
  ⟨fun Hlt y hy => (s.le_max' y hy).trans_lt Hlt, fun H => H _ <| s.max'_mem _⟩


@[simp]
theorem lt_min'_iff : x < s.min' H ↔ ∀ y ∈ s, x < y :=
  @max'_lt_iff αᵒᵈ _ _ H _


theorem max'_eq_sup' : s.max' H = s.sup' H id := rfl


theorem min'_eq_inf' : s.min' H = s.inf' H id := rfl


/-- `{a}.max' _` is `a`. -/
@[simp]
                                                                                        /-
                                                                                          α : Type u_2
                                                                                          inst✝ : LinearOrder α
                                                                                          a : α
                                                                                          ⊢ Eq ((Singleton.singleton a).max' ⋯) a
                                                                                        -/
theorem max'_singleton (a : α) : ({a} : Finset α).max' (singleton_nonempty _) = a := by simp [max']
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


theorem min'_lt_max' {i j} (H1 : i ∈ s) (H2 : j ∈ s) (H3 : i ≠ j) :
    s.min' ⟨i, H1⟩ < s.max' ⟨i, H1⟩ :=
  isGLB_lt_isLUB_of_ne (s.isLeast_min' _).isGLB (s.isGreatest_max' _).isLUB H1 H2 H3


/-- If there's more than 1 element, the min' is less than the max'. An alternate version of
`min'_lt_max'` which is sometimes more convenient.
-/
theorem min'_lt_max'_of_card (h₂ : 1 < card s) :
                                    /-
                                      F : Type u_1
                                      α : Type u_2
                                      β : Type u_3
                                      γ : Type u_4
                                      ι : Type u_5
                                      κ : Type u_6
                                      inst✝ : LinearOrder α
                                      s : Finset α
                                      H : s.Nonempty
                                      x : α
                                      h₂ : LT.lt 1 s.card
                                      ⊢ LT.lt 0 s.card
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
    s.min' (Finset.card_pos.1 <| by omega) < s.max' (Finset.card_pos.1 <| by omega) := by
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    h₂ : LT.lt 1 s.card
    ⊢ LT.lt (s.min' ⋯) (s.max' ⋯)
  -/
  rcases one_lt_card.1 h₂ with ⟨a, ha, b, hb, hab⟩
  /-
    case intro.intro.intro.intro
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    h₂ : LT.lt 1 s.card
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : Ne a b
    ⊢ LT.lt (s.min' ⋯) (s.max' ⋯)
  -/
  exact s.min'_lt_max' ha hb hab
  /-
    🎉 no goals
  -/


theorem max'_union {s₁ s₂ : Finset α} (h₁ : s₁.Nonempty) (h₂ : s₂.Nonempty) :
    (s₁ ∪ s₂).max' (h₁.mono subset_union_left) = s₁.max' h₁ ⊔ s₂.max' h₂ := sup'_union h₁ h₂ id


theorem min'_union {s₁ s₂ : Finset α} (h₁ : s₁.Nonempty) (h₂ : s₂.Nonempty) :
    (s₁ ∪ s₂).min' (h₁.mono subset_union_left) = s₁.min' h₁ ⊓ s₂.min' h₂ := inf'_union h₁ h₂ id


theorem map_ofDual_min (s : Finset αᵒᵈ) : s.min.map ofDual = (s.image ofDual).max := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset (OrderDual α)
    ⊢ Eq (WithTop.map (⇑OrderDual.ofDual) s.min) (Finset.image (⇑OrderDual.ofDual) …
  -/
  rw [max_eq_sup_withBot, sup_image]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset (OrderDual α)
    ⊢ Eq (WithTop.map (⇑OrderDual.ofDual) s.min) (s.sup (Function.comp WithBot.som …
  -/
  exact congr_fun Option.map_id _
  /-
    🎉 no goals
  -/


theorem map_ofDual_max (s : Finset αᵒᵈ) : s.max.map ofDual = (s.image ofDual).min := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset (OrderDual α)
    ⊢ Eq (WithBot.map (⇑OrderDual.ofDual) s.max) (Finset.image (⇑OrderDual.ofDual) …
  -/
  rw [min_eq_inf_withTop, inf_image]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset (OrderDual α)
    ⊢ Eq (WithBot.map (⇑OrderDual.ofDual) s.max) (s.inf (Function.comp WithTop.som …
  -/
  exact congr_fun Option.map_id _
  /-
    🎉 no goals
  -/


theorem map_toDual_min (s : Finset α) : s.min.map toDual = (s.image toDual).max := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    ⊢ Eq (WithTop.map (⇑OrderDual.toDual) s.min) (Finset.image (⇑OrderDual.toDual) …
  -/
  rw [max_eq_sup_withBot, sup_image]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    ⊢ Eq (WithTop.map (⇑OrderDual.toDual) s.min) (s.sup (Function.comp WithBot.som …
  -/
  exact congr_fun Option.map_id _
  /-
    🎉 no goals
  -/


theorem map_toDual_max (s : Finset α) : s.max.map toDual = (s.image toDual).min := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    ⊢ Eq (WithBot.map (⇑OrderDual.toDual) s.max) (Finset.image (⇑OrderDual.toDual) …
  -/
  rw [min_eq_inf_withTop, inf_image]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    ⊢ Eq (WithBot.map (⇑OrderDual.toDual) s.max) (s.inf (Function.comp WithTop.som …
  -/
  exact congr_fun Option.map_id _
  /-
    🎉 no goals
  -/

-- Porting note: new proofs without `convert` for the next four theorems.


theorem ofDual_min' {s : Finset αᵒᵈ} (hs : s.Nonempty) :
    ofDual (min' s hs) = max' (s.image ofDual) (hs.image _) := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset (OrderDual α)
    hs : s.Nonempty
    ⊢ Eq (OrderDual.ofDual (s.min' hs)) ((Finset.image (⇑OrderDual.ofDual) s).max' …
  -/
  rw [← WithBot.coe_eq_coe]
  simp only [min'_eq_inf', id_eq, ofDual_inf', Function.comp_apply, coe_sup', max'_eq_sup',
    sup_image]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset (OrderDual α)
    hs : s.Nonempty
    ⊢ Eq (s.sup (Function.comp WithBot.some fun x => OrderDual.ofDual x)) (s.sup ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ofDual_max' {s : Finset αᵒᵈ} (hs : s.Nonempty) :
    ofDual (max' s hs) = min' (s.image ofDual) (hs.image _) := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset (OrderDual α)
    hs : s.Nonempty
    ⊢ Eq (OrderDual.ofDual (s.max' hs)) ((Finset.image (⇑OrderDual.ofDual) s).min' …
  -/
  rw [← WithTop.coe_eq_coe]
  simp only [max'_eq_sup', id_eq, ofDual_sup', Function.comp_apply, coe_inf', min'_eq_inf',
    inf_image]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset (OrderDual α)
    hs : s.Nonempty
    ⊢ Eq (s.inf (Function.comp WithTop.some fun x => OrderDual.ofDual x)) (s.inf ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toDual_min' {s : Finset α} (hs : s.Nonempty) :
    toDual (min' s hs) = max' (s.image toDual) (hs.image _) := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    hs : s.Nonempty
    ⊢ Eq (OrderDual.toDual (s.min' hs)) ((Finset.image (⇑OrderDual.toDual) s).max' …
  -/
  rw [← WithBot.coe_eq_coe]
  simp only [min'_eq_inf', id_eq, toDual_inf', Function.comp_apply, coe_sup', max'_eq_sup',
    sup_image]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    hs : s.Nonempty
    ⊢ Eq (s.sup (Function.comp WithBot.some fun x => OrderDual.toDual x)) (s.sup ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toDual_max' {s : Finset α} (hs : s.Nonempty) :
    toDual (max' s hs) = min' (s.image toDual) (hs.image _) := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    hs : s.Nonempty
    ⊢ Eq (OrderDual.toDual (s.max' hs)) ((Finset.image (⇑OrderDual.toDual) s).min' …
  -/
  rw [← WithTop.coe_eq_coe]
  simp only [max'_eq_sup', id_eq, toDual_sup', Function.comp_apply, coe_inf', min'_eq_inf',
    inf_image]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Finset α
    hs : s.Nonempty
    ⊢ Eq (s.inf (Function.comp WithTop.some fun x => OrderDual.toDual x)) (s.inf ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem max'_subset {s t : Finset α} (H : s.Nonempty) (hst : s ⊆ t) :
    s.max' H ≤ t.max' (H.mono hst) :=
  le_max' _ _ (hst (s.max'_mem H))


theorem min'_subset {s t : Finset α} (H : s.Nonempty) (hst : s ⊆ t) :
    t.min' (H.mono hst) ≤ s.min' H :=
  min'_le _ _ (hst (s.min'_mem H))


theorem max'_insert (a : α) (s : Finset α) (H : s.Nonempty) :
    (insert a s).max' (s.insert_nonempty a) = max (s.max' H) a :=
  (isGreatest_max' _ _).unique <| by
    /-
      α : Type u_2
      inst✝ : LinearOrder α
      a : α
      s : Finset α
      H : s.Nonempty
      ⊢ IsGreatest (↑(Insert.insert a s)) (Max.max (s.max' H) a)
    -/
    rw [coe_insert, max_comm]
    /-
      α : Type u_2
      inst✝ : LinearOrder α
      a : α
      s : Finset α
      H : s.Nonempty
      ⊢ IsGreatest (Insert.insert a ↑s) (Max.max a (s.max' H))
    -/
    exact (isGreatest_max' _ _).insert _
    /-
      🎉 no goals
    -/


theorem min'_insert (a : α) (s : Finset α) (H : s.Nonempty) :
    (insert a s).min' (s.insert_nonempty a) = min (s.min' H) a :=
  (isLeast_min' _ _).unique <| by
    /-
      α : Type u_2
      inst✝ : LinearOrder α
      a : α
      s : Finset α
      H : s.Nonempty
      ⊢ IsLeast (↑(Insert.insert a s)) (Min.min (s.min' H) a)
    -/
    rw [coe_insert, min_comm]
    /-
      α : Type u_2
      inst✝ : LinearOrder α
      a : α
      s : Finset α
      H : s.Nonempty
      ⊢ IsLeast (Insert.insert a ↑s) (Min.min a (s.min' H))
    -/
    exact (isLeast_min' _ _).insert _
    /-
      🎉 no goals
    -/


theorem lt_max'_of_mem_erase_max' [DecidableEq α] {a : α} (ha : a ∈ s.erase (s.max' H)) :
    a < s.max' H :=
  lt_of_le_of_ne (le_max' _ _ (mem_of_mem_erase ha)) <| ne_of_mem_of_not_mem ha <| not_mem_erase _ _


theorem min'_lt_of_mem_erase_min' [DecidableEq α] {a : α} (ha : a ∈ s.erase (s.min' H)) :
    s.min' H < a :=
  @lt_max'_of_mem_erase_max' αᵒᵈ _ s H _ a ha


/-- To rewrite from right to left, use `Monotone.map_finset_max'`. -/
@[simp]
theorem max'_image [LinearOrder β] {f : α → β} (hf : Monotone f) (s : Finset α)
    (h : (s.image f).Nonempty) : (s.image f).max' h = f (s.max' h.of_image) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    hf : Monotone f
    s : Finset α
    h : (Finset.image f s).Nonempty
    ⊢ Eq ((Finset.image f s).max' h) (f (s.max' ⋯))
  -/
  simp only [max', sup'_image]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    hf : Monotone f
    s : Finset α
    h : (Finset.image f s).Nonempty
    ⊢ Eq (s.sup' ⋯ (Function.comp id f)) (f (s.sup' ⋯ id))
  -/
  exact .symm <| comp_sup'_eq_sup'_comp _ _ fun _ _ ↦ hf.map_max
  /-
    🎉 no goals
  -/


/-- A version of `Finset.max'_image` with LHS and RHS reversed.
Also, this version assumes that `s` is nonempty, not its image. -/
lemma _root_.Monotone.map_finset_max' [LinearOrder β] {f : α → β} (hf : Monotone f) {s : Finset α}
    (h : s.Nonempty) : f (s.max' h) = (s.image f).max' (h.image f) :=
  .symm <| max'_image hf ..


/-- To rewrite from right to left, use `Monotone.map_finset_min'`. -/
@[simp]
theorem min'_image [LinearOrder β] {f : α → β} (hf : Monotone f) (s : Finset α)
    (h : (s.image f).Nonempty) : (s.image f).min' h = f (s.min' h.of_image) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    hf : Monotone f
    s : Finset α
    h : (Finset.image f s).Nonempty
    ⊢ Eq ((Finset.image f s).min' h) (f (s.min' ⋯))
  -/
  simp only [min', inf'_image]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    hf : Monotone f
    s : Finset α
    h : (Finset.image f s).Nonempty
    ⊢ Eq (s.inf' ⋯ (Function.comp id f)) (f (s.inf' ⋯ id))
  -/
  exact .symm <| comp_inf'_eq_inf'_comp _ _ fun _ _ ↦ hf.map_min
  /-
    🎉 no goals
  -/


/-- A version of `Finset.min'_image` with LHS and RHS reversed.
Also, this version assumes that `s` is nonempty, not its image. -/
lemma _root_.Monotone.map_finset_min' [LinearOrder β] {f : α → β} (hf : Monotone f) {s : Finset α}
    (h : s.Nonempty) : f (s.min' h) = (s.image f).min' (h.image f) :=
  .symm <| min'_image hf ..


theorem coe_max' {s : Finset α} (hs : s.Nonempty) : ↑(s.max' hs) = s.max :=
  coe_sup' hs id


theorem coe_min' {s : Finset α} (hs : s.Nonempty) : ↑(s.min' hs) = s.min :=
  coe_inf' hs id


theorem max_mem_image_coe {s : Finset α} (hs : s.Nonempty) :
    s.max ∈ (s.image (↑) : Finset (WithBot α)) :=
  mem_image.2 ⟨max' s hs, max'_mem _ _, coe_max' hs⟩


theorem min_mem_image_coe {s : Finset α} (hs : s.Nonempty) :
    s.min ∈ (s.image (↑) : Finset (WithTop α)) :=
  mem_image.2 ⟨min' s hs, min'_mem _ _, coe_min' hs⟩


theorem max_mem_insert_bot_image_coe (s : Finset α) :
    s.max ∈ (insert ⊥ (s.image (↑)) : Finset (WithBot α)) :=
  mem_insert.2 <| s.eq_empty_or_nonempty.imp max_eq_bot.2 max_mem_image_coe


theorem min_mem_insert_top_image_coe (s : Finset α) :
    s.min ∈ (insert ⊤ (s.image (↑)) : Finset (WithTop α)) :=
  mem_insert.2 <| s.eq_empty_or_nonempty.imp min_eq_top.2 min_mem_image_coe


theorem max'_erase_ne_self {s : Finset α} (s0 : (s.erase x).Nonempty) : (s.erase x).max' s0 ≠ x :=
  ne_of_mem_erase (max'_mem _ s0)


theorem min'_erase_ne_self {s : Finset α} (s0 : (s.erase x).Nonempty) : (s.erase x).min' s0 ≠ x :=
  ne_of_mem_erase (min'_mem _ s0)


theorem max_erase_ne_self {s : Finset α} : (s.erase x).max ≠ x := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    x : α
    s : Finset α
    ⊢ Ne (s.erase x).max ↑x
  -/
  by_cases s0 : (s.erase x).Nonempty
    /-
      case pos
      α : Type u_2
      inst✝ : LinearOrder α
      x : α
      s : Finset α
      s0 : (s.erase x).Nonempty
      ⊢ Ne (s.erase x).max ↑x
    -/
  · refine ne_of_eq_of_ne (coe_max' s0).symm ?_
    /-
      case pos
      α : Type u_2
      inst✝ : LinearOrder α
      x : α
      s : Finset α
      s0 : (s.erase x).Nonempty
      ⊢ Ne ↑((s.erase x).max' s0) ↑x
    -/
    exact WithBot.coe_eq_coe.not.mpr (max'_erase_ne_self _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      inst✝ : LinearOrder α
      x : α
      s : Finset α
      s0 : Not (s.erase x).Nonempty
      ⊢ Ne (s.erase x).max ↑x
    -/
  · rw [not_nonempty_iff_eq_empty.mp s0, max_empty]
    /-
      case neg
      α : Type u_2
      inst✝ : LinearOrder α
      x : α
      s : Finset α
      s0 : Not (s.erase x).Nonempty
      ⊢ Ne Bot.bot ↑x
    -/
    exact WithBot.bot_ne_coe
    /-
      🎉 no goals
    -/


theorem min_erase_ne_self {s : Finset α} : (s.erase x).min ≠ x := by
  -- Porting note: old proof `convert @max_erase_ne_self αᵒᵈ _ _ _`
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    x : α
    s : Finset α
    ⊢ Ne (s.erase x).min ↑x
  -/
  convert @max_erase_ne_self αᵒᵈ _ (toDual x) (s.map toDual.toEmbedding) using 1
  /-
    case h.e'_2.h
    α : Type u_2
    inst✝ : LinearOrder α
    x : α
    s : Finset α
    e_1✝ : Eq (WithTop α) (WithBot (OrderDual α))
    ⊢ Eq (s.erase x).min ((Finset.map OrderDual.toDual.toEmbedding s).erase (Order …
  -/
  apply congr_arg -- Porting note: forces unfolding to see `Finset.min` is `Finset.max`
  /-
    case h.e'_2.h.h
    α : Type u_2
    inst✝ : LinearOrder α
    x : α
    s : Finset α
    e_1✝ : Eq (WithTop α) (WithBot (OrderDual α))
    ⊢ Eq (s.erase x) ((Finset.map OrderDual.toDual.toEmbedding s).erase (OrderDual …
  -/
  congr!
  /-
    case h.e'_2.h.h.h.e'_3.h
    α : Type u_2
    inst✝ : LinearOrder α
    x : α
    s : Finset α
    e_1✝¹ : Eq (WithTop α) (WithBot (OrderDual α))
    e_1✝ : Eq α (OrderDual α)
    ⊢ Eq s (Finset.map OrderDual.toDual.toEmbedding s)
  -/
  ext; simp only [mem_map_equiv]; exact Iff.rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem exists_next_right {x : α} {s : Finset α} (h : ∃ y ∈ s, x < y) :
    ∃ y ∈ s, x < y ∧ ∀ z ∈ s, x < z → y ≤ z :=
                                                                               /-
                                                                                 α : Type u_2
                                                                                 inst✝ : LinearOrder α
                                                                                 x : α
                                                                                 s : Finset α
                                                                                 h : Exists fun y => And (Membership.mem s y) (LT.lt x y)
                                                                                 y : α
                                                                                 hy : And (Membership.mem s y) (LT.lt x y)
                                                                                 ⊢ And (Membership.mem s y) (LT.lt x y)
                                                                               -/
  have Hne : (s.filter (x < ·)).Nonempty := h.imp fun y hy => mem_filter.2 (by simpa)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  have aux := mem_filter.1 (min'_mem _ Hne)
                         /-
                           α : Type u_2
                           inst✝ : LinearOrder α
                           x : α
                           s : Finset α
                           h : Exists fun y => And (Membership.mem s y) (LT.lt x y)
                           Hne : (Finset.filter (fun x_1 => LT.lt x x_1) s).Nonempty
                           aux : And (Membership.mem s ((Finset.filter (fun x_1 => LT.lt x x_1) s).min' H …
                           ⊢ LT.lt x ((Finset.filter (fun x_1 => LT.lt x x_1) s).min' Hne)
                         -/
                         /-
                           🎉 no goals
                         -/
  ⟨min' _ Hne, aux.1, by simp, fun z hzs hz => min'_le _ _ <| mem_filter.2 ⟨hzs, by simpa⟩⟩
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem exists_next_left {x : α} {s : Finset α} (h : ∃ y ∈ s, y < x) :
    ∃ y ∈ s, y < x ∧ ∀ z ∈ s, z < x → z ≤ y :=
  @exists_next_right αᵒᵈ _ x s h


/-- If finsets `s` and `t` are interleaved, then `Finset.card s ≤ Finset.card t + 1`. -/
theorem card_le_of_interleaved {s t : Finset α}
    (h : ∀ᵉ (x ∈ s) (y ∈ s),
        x < y → (∀ z ∈ s, z ∉ Set.Ioo x y) → ∃ z ∈ t, x < z ∧ z < y) :
    s.card ≤ t.card + 1 := by
  replace h : ∀ᵉ (x ∈ s) (y ∈ s), x < y → ∃ z ∈ t, x < z ∧ z < y := by
    intro x hx y hy hxy
    rcases exists_next_right ⟨y, hy, hxy⟩ with ⟨a, has, hxa, ha⟩
    rcases h x hx a has hxa fun z hzs hz => hz.2.not_le <| ha _ hzs hz.1 with ⟨b, hbt, hxb, hba⟩
    exact ⟨b, hbt, hxb, hba.trans_le <| ha _ hy hxy⟩
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s t : Finset α
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LT.lt x y  …
    ⊢ LE.le s.card (HAdd.hAdd t.card 1)
  -/
  set f : α → WithTop α := fun x => (t.filter fun y => x < y).min
  have f_mono : StrictMonoOn f s := by
    intro x hx y hy hxy
    rcases h x hx y hy hxy with ⟨a, hat, hxa, hay⟩
    calc
      f x ≤ a := min_le (mem_filter.2 ⟨hat, by simpa⟩)
      _ < f y :=
        (Finset.lt_inf_iff <| WithTop.coe_lt_top a).2 fun b hb =>
          WithTop.coe_lt_coe.2 <| hay.trans (by simpa using (mem_filter.1 hb).2)

  calc
    s.card = (s.image f).card := (card_image_of_injOn f_mono.injOn).symm
    _ ≤ (insert ⊤ (t.image (↑)) : Finset (WithTop α)).card :=
      card_mono <| image_subset_iff.2 fun x _ =>
          insert_subset_insert _ (image_subset_image <| filter_subset _ _)
            (min_mem_insert_top_image_coe _)
    _ ≤ t.card + 1 := (card_insert_le _ _).trans (Nat.add_le_add_right card_image_le _)


/-- If finsets `s` and `t` are interleaved, then `Finset.card s ≤ Finset.card (t \ s) + 1`. -/
theorem card_le_diff_of_interleaved {s t : Finset α}
    (h :
      ∀ᵉ (x ∈ s) (y ∈ s),
        x < y → (∀ z ∈ s, z ∉ Set.Ioo x y) → ∃ z ∈ t, x < z ∧ z < y) :
    s.card ≤ (t \ s).card + 1 :=
  card_le_of_interleaved fun x hx y hy hxy hs =>
    let ⟨z, hzt, hxz, hzy⟩ := h x hx y hy hxy hs
    ⟨z, mem_sdiff.2 ⟨hzt, fun hzs => hs z hzs ⟨hxz, hzy⟩⟩, hxz, hzy⟩


/-- Induction principle for `Finset`s in a linearly ordered type: a predicate is true on all
`s : Finset α` provided that:

* it is true on the empty `Finset`,
* for every `s : Finset α` and an element `a` strictly greater than all elements of `s`, `p s`
  implies `p (insert a s)`. -/
@[elab_as_elim]
theorem induction_on_max [DecidableEq α] {p : Finset α → Prop} (s : Finset α) (h0 : p ∅)
    (step : ∀ a s, (∀ x ∈ s, x < a) → p s → p (insert a s)) : p s := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : DecidableEq α
    p : Finset α → Prop
    s : Finset α
    h0 : p EmptyCollection.emptyCollection
    step : ∀ (a : α) (s : Finset α), (∀ (x : α), Membership.mem s x → LT.lt x a) → …
    ⊢ p s
  -/
  induction' s using Finset.strongInductionOn with s ihs
  /-
    case a
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : DecidableEq α
    p : Finset α → Prop
    h0 : p EmptyCollection.emptyCollection
    step : ∀ (a : α) (s : Finset α), (∀ (x : α), Membership.mem s x → LT.lt x a) → …
    s : Finset α
    ihs : ∀ (t : Finset α), HasSSubset.SSubset t s → p t
    ⊢ p s
  -/
  rcases s.eq_empty_or_nonempty with (rfl | hne)
    /-
      case a.inl
      α : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq α
      p : Finset α → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : α) (s : Finset α), (∀ (x : α), Membership.mem s x → LT.lt x a) → …
      ihs : ∀ (t : Finset α), HasSSubset.SSubset t EmptyCollection.emptyCollection → …
      ⊢ p EmptyCollection.emptyCollection
    -/
  · exact h0
    /-
      🎉 no goals
    -/
    /-
      case a.inr
      α : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq α
      p : Finset α → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : α) (s : Finset α), (∀ (x : α), Membership.mem s x → LT.lt x a) → …
      s : Finset α
      ihs : ∀ (t : Finset α), HasSSubset.SSubset t s → p t
      hne : s.Nonempty
      ⊢ p s
    -/
  · have H : s.max' hne ∈ s := max'_mem s hne
    /-
      case a.inr
      α : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq α
      p : Finset α → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : α) (s : Finset α), (∀ (x : α), Membership.mem s x → LT.lt x a) → …
      s : Finset α
      ihs : ∀ (t : Finset α), HasSSubset.SSubset t s → p t
      hne : s.Nonempty
      H : Membership.mem s (s.max' hne)
      ⊢ p s
    -/
    rw [← insert_erase H]
    /-
      case a.inr
      α : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq α
      p : Finset α → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : α) (s : Finset α), (∀ (x : α), Membership.mem s x → LT.lt x a) → …
      s : Finset α
      ihs : ∀ (t : Finset α), HasSSubset.SSubset t s → p t
      hne : s.Nonempty
      H : Membership.mem s (s.max' hne)
      ⊢ p (Insert.insert (s.max' hne) (s.erase (s.max' hne)))
    -/
    exact step _ _ (fun x => s.lt_max'_of_mem_erase_max' hne) (ihs _ <| erase_ssubset H)
    /-
      🎉 no goals
    -/


/-- Induction principle for `Finset`s in a linearly ordered type: a predicate is true on all
`s : Finset α` provided that:

* it is true on the empty `Finset`,
* for every `s : Finset α` and an element `a` strictly less than all elements of `s`, `p s`
  implies `p (insert a s)`. -/
@[elab_as_elim]
theorem induction_on_min [DecidableEq α] {p : Finset α → Prop} (s : Finset α) (h0 : p ∅)
    (step : ∀ a s, (∀ x ∈ s, a < x) → p s → p (insert a s)) : p s :=
  @induction_on_max αᵒᵈ _ _ _ s h0 step


/-- Induction principle for `Finset`s in any type from which a given function `f` maps to a linearly
ordered type : a predicate is true on all `s : Finset α` provided that:

* it is true on the empty `Finset`,
* for every `s : Finset α` and an element `a` such that for elements of `s` denoted by `x` we have
  `f x ≤ f a`, `p s` implies `p (insert a s)`. -/
@[elab_as_elim]
theorem induction_on_max_value [DecidableEq ι] (f : ι → α) {p : Finset ι → Prop} (s : Finset ι)
    (h0 : p ∅) (step : ∀ a s, a ∉ s → (∀ x ∈ s, f x ≤ f a) → p s → p (insert a s)) : p s := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝¹ : LinearOrder α
    inst✝ : DecidableEq ι
    f : ι → α
    p : Finset ι → Prop
    s : Finset ι
    h0 : p EmptyCollection.emptyCollection
    step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
    ⊢ p s
  -/
  induction' s using Finset.strongInductionOn with s ihs
  /-
    case a
    α : Type u_2
    ι : Type u_5
    inst✝¹ : LinearOrder α
    inst✝ : DecidableEq ι
    f : ι → α
    p : Finset ι → Prop
    h0 : p EmptyCollection.emptyCollection
    step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
    s : Finset ι
    ihs : ∀ (t : Finset ι), HasSSubset.SSubset t s → p t
    ⊢ p s
  -/
  rcases (s.image f).eq_empty_or_nonempty with (hne | hne)
    /-
      case a.inl
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq ι
      f : ι → α
      p : Finset ι → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
      s : Finset ι
      ihs : ∀ (t : Finset ι), HasSSubset.SSubset t s → p t
      hne : Eq (Finset.image f s) EmptyCollection.emptyCollection
      ⊢ p s
    -/
  · simp only [image_eq_empty] at hne
    /-
      case a.inl
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq ι
      f : ι → α
      p : Finset ι → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
      s : Finset ι
      ihs : ∀ (t : Finset ι), HasSSubset.SSubset t s → p t
      hne : Eq s EmptyCollection.emptyCollection
      ⊢ p s
    -/
    simp only [hne, h0]
    /-
      🎉 no goals
    -/
    /-
      case a.inr
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq ι
      f : ι → α
      p : Finset ι → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
      s : Finset ι
      ihs : ∀ (t : Finset ι), HasSSubset.SSubset t s → p t
      hne : (Finset.image f s).Nonempty
      ⊢ p s
    -/
  · have H : (s.image f).max' hne ∈ s.image f := max'_mem (s.image f) hne
    /-
      case a.inr
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq ι
      f : ι → α
      p : Finset ι → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
      s : Finset ι
      ihs : ∀ (t : Finset ι), HasSSubset.SSubset t s → p t
      hne : (Finset.image f s).Nonempty
      H : Membership.mem (Finset.image f s) ((Finset.image f s).max' hne)
      ⊢ p s
    -/
    simp only [mem_image, exists_prop] at H
    /-
      case a.inr
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq ι
      f : ι → α
      p : Finset ι → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
      s : Finset ι
      ihs : ∀ (t : Finset ι), HasSSubset.SSubset t s → p t
      hne : (Finset.image f s).Nonempty
      H : Exists fun a => And (Membership.mem s a) (Eq (f a) ((Finset.image f s).max …
      ⊢ p s
    -/
    rcases H with ⟨a, has, hfa⟩
    /-
      case a.inr.intro.intro
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq ι
      f : ι → α
      p : Finset ι → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
      s : Finset ι
      ihs : ∀ (t : Finset ι), HasSSubset.SSubset t s → p t
      hne : (Finset.image f s).Nonempty
      a : ι
      has : Membership.mem s a
      hfa : Eq (f a) ((Finset.image f s).max' hne)
      ⊢ p s
    -/
    rw [← insert_erase has]
    /-
      case a.inr.intro.intro
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq ι
      f : ι → α
      p : Finset ι → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
      s : Finset ι
      ihs : ∀ (t : Finset ι), HasSSubset.SSubset t s → p t
      hne : (Finset.image f s).Nonempty
      a : ι
      has : Membership.mem s a
      hfa : Eq (f a) ((Finset.image f s).max' hne)
      ⊢ p (Insert.insert a (s.erase a))
    -/
    refine step _ _ (not_mem_erase a s) (fun x hx => ?_) (ihs _ <| erase_ssubset has)
    /-
      case a.inr.intro.intro
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq ι
      f : ι → α
      p : Finset ι → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
      s : Finset ι
      ihs : ∀ (t : Finset ι), HasSSubset.SSubset t s → p t
      hne : (Finset.image f s).Nonempty
      a : ι
      has : Membership.mem s a
      hfa : Eq (f a) ((Finset.image f s).max' hne)
      x : ι
      hx : Membership.mem (s.erase a) x
      ⊢ LE.le (f x) (f a)
    -/
    rw [hfa]
    /-
      case a.inr.intro.intro
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : DecidableEq ι
      f : ι → α
      p : Finset ι → Prop
      h0 : p EmptyCollection.emptyCollection
      step : ∀ (a : ι) (s : Finset ι), Not (Membership.mem s a) → (∀ (x : ι), Member …
      s : Finset ι
      ihs : ∀ (t : Finset ι), HasSSubset.SSubset t s → p t
      hne : (Finset.image f s).Nonempty
      a : ι
      has : Membership.mem s a
      hfa : Eq (f a) ((Finset.image f s).max' hne)
      x : ι
      hx : Membership.mem (s.erase a) x
      ⊢ LE.le (f x) ((Finset.image f s).max' hne)
    -/
    exact le_max' _ _ (mem_image_of_mem _ <| mem_of_mem_erase hx)
    /-
      🎉 no goals
    -/


/-- Induction principle for `Finset`s in any type from which a given function `f` maps to a linearly
ordered type : a predicate is true on all `s : Finset α` provided that:

* it is true on the empty `Finset`,
* for every `s : Finset α` and an element `a` such that for elements of `s` denoted by `x` we have
  `f a ≤ f x`, `p s` implies `p (insert a s)`. -/
@[elab_as_elim]
theorem induction_on_min_value [DecidableEq ι] (f : ι → α) {p : Finset ι → Prop} (s : Finset ι)
    (h0 : p ∅) (step : ∀ a s, a ∉ s → (∀ x ∈ s, f a ≤ f x) → p s → p (insert a s)) : p s :=
  @induction_on_max_value αᵒᵈ ι _ _ _ _ s h0 step


theorem exists_max_image (s : Finset β) (f : β → α) (h : s.Nonempty) :
    ∃ x ∈ s, ∀ x' ∈ s, f x' ≤ f x := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : LinearOrder α
    s : Finset β
    f : β → α
    h : s.Nonempty
    ⊢ Exists fun x => And (Membership.mem s x) (∀ (x' : β), Membership.mem s x' →  …
  -/
  cases' max_of_nonempty (h.image f) with y hy
  /-
    case intro
    α : Type u_2
    β : Type u_3
    inst✝ : LinearOrder α
    s : Finset β
    f : β → α
    h : s.Nonempty
    y : α
    hy : Eq (Finset.image f s).max ↑y
    ⊢ Exists fun x => And (Membership.mem s x) (∀ (x' : β), Membership.mem s x' →  …
  -/
  rcases mem_image.mp (mem_of_max hy) with ⟨x, hx, rfl⟩
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝ : LinearOrder α
    s : Finset β
    f : β → α
    h : s.Nonempty
    x : β
    hx : Membership.mem s x
    hy : Eq (Finset.image f s).max ↑(f x)
    ⊢ Exists fun x => And (Membership.mem s x) (∀ (x' : β), Membership.mem s x' →  …
  -/
  exact ⟨x, hx, fun x' hx' => le_max_of_eq (mem_image_of_mem f hx') hy⟩
  /-
    🎉 no goals
  -/


theorem exists_min_image (s : Finset β) (f : β → α) (h : s.Nonempty) :
    ∃ x ∈ s, ∀ x' ∈ s, f x ≤ f x' :=
  @exists_max_image αᵒᵈ β _ s f h


theorem isGLB_iff_isLeast [LinearOrder α] (i : α) (s : Finset α) (hs : s.Nonempty) :
    IsGLB (s : Set α) i ↔ IsLeast (↑s) i := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    i : α
    s : Finset α
    hs : s.Nonempty
    ⊢ Iff (IsGLB (↑s) i) (IsLeast (↑s) i)
  -/
  refine ⟨fun his => ?_, IsLeast.isGLB⟩
  suffices i = min' s hs by
    rw [this]
    exact isLeast_min' s hs
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    i : α
    s : Finset α
    hs : s.Nonempty
    his : IsGLB (↑s) i
    ⊢ Eq i (s.min' hs)
  -/
  rw [IsGLB, IsGreatest, mem_lowerBounds, mem_upperBounds] at his
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    i : α
    s : Finset α
    hs : s.Nonempty
    his : And (∀ (x : α), Membership.mem (↑s) x → LE.le i x) (∀ (x : α), Membershi …
    ⊢ Eq i (s.min' hs)
  -/
  exact le_antisymm (his.1 (Finset.min' s hs) (Finset.min'_mem s hs)) (his.2 _ (Finset.min'_le s))
  /-
    🎉 no goals
  -/


theorem isLUB_iff_isGreatest [LinearOrder α] (i : α) (s : Finset α) (hs : s.Nonempty) :
    IsLUB (s : Set α) i ↔ IsGreatest (↑s) i :=
  @isGLB_iff_isLeast αᵒᵈ _ i s hs


theorem isGLB_mem [LinearOrder α] {i : α} (s : Finset α) (his : IsGLB (s : Set α) i)
    (hs : s.Nonempty) : i ∈ s := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    i : α
    s : Finset α
    his : IsGLB (↑s) i
    hs : s.Nonempty
    ⊢ Membership.mem s i
  -/
  rw [← mem_coe]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    i : α
    s : Finset α
    his : IsGLB (↑s) i
    hs : s.Nonempty
    ⊢ Membership.mem (↑s) i
  -/
  exact ((isGLB_iff_isLeast i s hs).mp his).1
  /-
    🎉 no goals
  -/


theorem isLUB_mem [LinearOrder α] {i : α} (s : Finset α) (his : IsLUB (s : Set α) i)
    (hs : s.Nonempty) : i ∈ s :=
  @isGLB_mem αᵒᵈ _ i s his hs


