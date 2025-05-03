/-- Elements of `s` which are "below" `b` according to relation `r`. -/
def bipartiteBelow : Finset α := {a ∈ s | r a b}


/-- Elements of `t` which are "above" `a` according to relation `r`. -/
def bipartiteAbove : Finset β := {b ∈ t | r a b}


theorem bipartiteBelow_swap : t.bipartiteBelow (swap r) a = t.bipartiteAbove r a := rfl


theorem bipartiteAbove_swap : s.bipartiteAbove (swap r) b = s.bipartiteBelow r b := rfl


@[simp, norm_cast]
theorem coe_bipartiteBelow : s.bipartiteBelow r b = ({a ∈ s | r a b} : Set α) := coe_filter _ _


@[simp, norm_cast]
theorem coe_bipartiteAbove : t.bipartiteAbove r a = ({b ∈ t | r a b} : Set β) := coe_filter _ _


@[simp]
theorem mem_bipartiteBelow {a : α} : a ∈ s.bipartiteBelow r b ↔ a ∈ s ∧ r a b := mem_filter


@[simp]
theorem mem_bipartiteAbove {b : β} : b ∈ t.bipartiteAbove r a ↔ b ∈ t ∧ r a b := mem_filter


@[to_additive]
theorem prod_prod_bipartiteAbove_eq_prod_prod_bipartiteBelow
    [CommMonoid R] (f : α → β → R) [∀ a b, Decidable (r a b)] :
    ∏ a ∈ s, ∏ b ∈ t.bipartiteAbove r a, f a b = ∏ b ∈ t, ∏ a ∈ s.bipartiteBelow r b, f a b := by
  /-
    R : Type u_1
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    s : Finset α
    t : Finset β
    inst✝¹ : CommMonoid R
    f : α → β → R
    inst✝ : (a : α) → (b : β) → Decidable (r a b)
    ⊢ Eq (s.prod fun a => (Finset.bipartiteAbove r t a).prod fun b => f a b) (t.pr …
  -/
  simp_rw [bipartiteAbove, bipartiteBelow, prod_filter]
  /-
    R : Type u_1
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    s : Finset α
    t : Finset β
    inst✝¹ : CommMonoid R
    f : α → β → R
    inst✝ : (a : α) → (b : β) → Decidable (r a b)
    ⊢ Eq (s.prod fun x => t.prod fun a => ite (r x a) (f x a) 1) (t.prod fun x =>  …
  -/
  exact prod_comm
  /-
    🎉 no goals
  -/


theorem sum_card_bipartiteAbove_eq_sum_card_bipartiteBelow [∀ a b, Decidable (r a b)] :
    (∑ a ∈ s, #(t.bipartiteAbove r a)) = ∑ b ∈ t, #(s.bipartiteBelow r b) := by
  /-
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    s : Finset α
    t : Finset β
    inst✝ : (a : α) → (b : β) → Decidable (r a b)
    ⊢ Eq (s.sum fun a => (Finset.bipartiteAbove r t a).card) (t.sum fun b => (Fins …
  -/
  simp_rw [card_eq_sum_ones, sum_sum_bipartiteAbove_eq_sum_sum_bipartiteBelow]
  /-
    🎉 no goals
  -/


/-- **Double counting** argument.

Considering `r` as a bipartite graph, the LHS is a lower bound on the number of edges while the RHS
is an upper bound. -/
theorem card_nsmul_le_card_nsmul [∀ a b, Decidable (r a b)]
    (hm : ∀ a ∈ s, m ≤ #(t.bipartiteAbove r a))
    (hn : ∀ b ∈ t, #(s.bipartiteBelow r b) ≤ n) : #s • m ≤ #t • n :=
  calc
    _ ≤ ∑ a in s, (#(t.bipartiteAbove r a) : R) := s.card_nsmul_le_sum _ _ hm
    _ = ∑ b in t, (#(s.bipartiteBelow r b) : R) := by
      /-
        R : Type u_1
        α : Type u_2
        β : Type u_3
        r : α → β → Prop
        s : Finset α
        t : Finset β
        inst✝¹ : OrderedSemiring R
        m n : R
        inst✝ : (a : α) → (b : β) → Decidable (r a b)
        hm : ∀ (a : α), Membership.mem s a → LE.le m ↑(Finset.bipartiteAbove r t a).card
        hn : ∀ (b : β), Membership.mem t b → LE.le (↑(Finset.bipartiteBelow r s b).car …
        ⊢ Eq (s.sum fun a => ↑(Finset.bipartiteAbove r t a).card) (t.sum fun b => ↑(Fi …
      -/
      norm_cast; rw [sum_card_bipartiteAbove_eq_sum_card_bipartiteBelow]
                 /-
                   🎉 no goals
                 -/
    _ ≤ _ := t.sum_le_card_nsmul _ _ hn


/-- **Double counting** argument.

Considering `r` as a bipartite graph, the LHS is a lower bound on the number of edges while the RHS
is an upper bound. -/
theorem card_nsmul_le_card_nsmul' [∀ a b, Decidable (r a b)]
    (hn : ∀ b ∈ t, n ≤ #(s.bipartiteBelow r b))
    (hm : ∀ a ∈ s, #(t.bipartiteAbove r a) ≤ m) : #t • n ≤ #s • m :=
  card_nsmul_le_card_nsmul (swap r) hn hm


/-- **Double counting** argument.

Considering `r` as a bipartite graph, the LHS is a strict lower bound on the number of edges while
the RHS is an upper bound. -/
theorem card_nsmul_lt_card_nsmul_of_lt_of_le [∀ a b, Decidable (r a b)] (hs : s.Nonempty)
    (hm : ∀ a ∈ s, m < #(t.bipartiteAbove r a))
    (hn : ∀ b ∈ t, #(s.bipartiteBelow r b) ≤ n) : #s • m < #t • n :=
  calc
                          /-
                            R : Type u_1
                            α : Type u_2
                            β : Type u_3
                            inst✝¹ : StrictOrderedSemiring R
                            r : α → β → Prop
                            s : Finset α
                            t : Finset β
                            m n : R
                            inst✝ : (a : α) → (b : β) → Decidable (r a b)
                            hs : s.Nonempty
                            hm : ∀ (a : α), Membership.mem s a → LT.lt m ↑(Finset.bipartiteAbove r t a).card
                            hn : ∀ (b : β), Membership.mem t b → LE.le (↑(Finset.bipartiteBelow r s b).car …
                            ⊢ Eq (HSMul.hSMul s.card m) (s.sum fun _a => m)
                          -/
    _ = ∑ _a ∈ s, m := by rw [sum_const]
                          /-
                            🎉 no goals
                          -/
    _ < ∑ a ∈ s, (#(t.bipartiteAbove r a) : R) := sum_lt_sum_of_nonempty hs hm
    _ = ∑ b in t, (#(s.bipartiteBelow r b) : R) := by
      /-
        R : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : StrictOrderedSemiring R
        r : α → β → Prop
        s : Finset α
        t : Finset β
        m n : R
        inst✝ : (a : α) → (b : β) → Decidable (r a b)
        hs : s.Nonempty
        hm : ∀ (a : α), Membership.mem s a → LT.lt m ↑(Finset.bipartiteAbove r t a).card
        hn : ∀ (b : β), Membership.mem t b → LE.le (↑(Finset.bipartiteBelow r s b).car …
        ⊢ Eq (s.sum fun a => ↑(Finset.bipartiteAbove r t a).card) (t.sum fun b => ↑(Fi …
      -/
      norm_cast; rw [sum_card_bipartiteAbove_eq_sum_card_bipartiteBelow]
                 /-
                   🎉 no goals
                 -/
    _ ≤ _ := t.sum_le_card_nsmul _ _ hn


/-- **Double counting** argument.

Considering `r` as a bipartite graph, the LHS is a lower bound on the number of edges while the RHS
is a strict upper bound. -/
theorem card_nsmul_lt_card_nsmul_of_le_of_lt [∀ a b, Decidable (r a b)] (ht : t.Nonempty)
    (hm : ∀ a ∈ s, m ≤ #(t.bipartiteAbove r a))
    (hn : ∀ b ∈ t, #(s.bipartiteBelow r b) < n) : #s • m < #t • n :=
  calc
    _ ≤ ∑ a in s, (#(t.bipartiteAbove r a) : R) := s.card_nsmul_le_sum _ _ hm
    _ = ∑ b in t, (#(s.bipartiteBelow r b) : R) := by
      /-
        R : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : StrictOrderedSemiring R
        r : α → β → Prop
        s : Finset α
        t : Finset β
        m n : R
        inst✝ : (a : α) → (b : β) → Decidable (r a b)
        ht : t.Nonempty
        hm : ∀ (a : α), Membership.mem s a → LE.le m ↑(Finset.bipartiteAbove r t a).card
        hn : ∀ (b : β), Membership.mem t b → LT.lt (↑(Finset.bipartiteBelow r s b).car …
        ⊢ Eq (s.sum fun a => ↑(Finset.bipartiteAbove r t a).card) (t.sum fun b => ↑(Fi …
      -/
      norm_cast; rw [sum_card_bipartiteAbove_eq_sum_card_bipartiteBelow]
                 /-
                   🎉 no goals
                 -/
    _ < ∑ _b ∈ t, n := sum_lt_sum_of_nonempty ht hn
    _ = _ := sum_const _


/-- **Double counting** argument.

Considering `r` as a bipartite graph, the LHS is a strict lower bound on the number of edges while
the RHS is an upper bound. -/
theorem card_nsmul_lt_card_nsmul_of_lt_of_le' [∀ a b, Decidable (r a b)] (ht : t.Nonempty)
    (hn : ∀ b ∈ t, n < #(s.bipartiteBelow r b))
    (hm : ∀ a ∈ s, #(t.bipartiteAbove r a) ≤ m) : #t • n < #s • m :=
  card_nsmul_lt_card_nsmul_of_lt_of_le (swap r) ht hn hm


/-- **Double counting** argument.

Considering `r` as a bipartite graph, the LHS is a lower bound on the number of edges while the RHS
is a strict upper bound. -/
theorem card_nsmul_lt_card_nsmul_of_le_of_lt' [∀ a b, Decidable (r a b)] (hs : s.Nonempty)
    (hn : ∀ b ∈ t, n ≤ #(s.bipartiteBelow r b))
    (hm : ∀ a ∈ s, #(t.bipartiteAbove r a) < m) : #t • n < #s • m :=
  card_nsmul_lt_card_nsmul_of_le_of_lt (swap r) hs hn hm


/-- **Double counting** argument.

Considering `r` as a bipartite graph, the LHS is a lower bound on the number of edges while the RHS
is an upper bound. -/
theorem card_mul_le_card_mul [∀ a b, Decidable (r a b)]
    (hm : ∀ a ∈ s, m ≤ #(t.bipartiteAbove r a))
    (hn : ∀ b ∈ t, #(s.bipartiteBelow r b) ≤ n) : #s * m ≤ #t * n :=
  card_nsmul_le_card_nsmul _ hm hn


theorem card_mul_le_card_mul' [∀ a b, Decidable (r a b)]
    (hn : ∀ b ∈ t, n ≤ #(s.bipartiteBelow r b))
    (hm : ∀ a ∈ s, #(t.bipartiteAbove r a) ≤ m) : #t * n ≤ #s * m :=
  card_nsmul_le_card_nsmul' _ hn hm


theorem card_mul_eq_card_mul [∀ a b, Decidable (r a b)]
    (hm : ∀ a ∈ s, #(t.bipartiteAbove r a) = m)
    (hn : ∀ b ∈ t, #(s.bipartiteBelow r b) = n) : #s * m = #t * n :=
  (card_mul_le_card_mul _ (fun a ha ↦ (hm a ha).ge) fun b hb ↦ (hn b hb).le).antisymm <|
    card_mul_le_card_mul' _ (fun a ha ↦ (hn a ha).ge) fun b hb ↦ (hm b hb).le


theorem card_le_card_of_forall_subsingleton (hs : ∀ a ∈ s, ∃ b, b ∈ t ∧ r a b)
    (ht : ∀ b ∈ t, ({ a ∈ s | r a b } : Set α).Subsingleton) : #s ≤ #t := by
  classical
    rw [← mul_one #s, ← mul_one #t]
    exact card_mul_le_card_mul r
      (fun a h ↦ card_pos.2 (by
        rw [← coe_nonempty, coe_bipartiteAbove]
        exact hs _ h : (t.bipartiteAbove r a).Nonempty))
      (fun b h ↦ card_le_one.2 (by
        simp_rw [mem_bipartiteBelow]
        exact ht _ h))


theorem card_le_card_of_forall_subsingleton' (ht : ∀ b ∈ t, ∃ a, a ∈ s ∧ r a b)
    (hs : ∀ a ∈ s, ({ b ∈ t | r a b } : Set β).Subsingleton) : #t ≤ #s :=
  card_le_card_of_forall_subsingleton (swap r) ht hs


theorem card_le_card_of_leftTotal_unique (h₁ : LeftTotal r) (h₂ : LeftUnique r) :
    Fintype.card α ≤ Fintype.card β :=
                                            /-
                                              α : Type u_2
                                              β : Type u_3
                                              inst✝¹ : Fintype α
                                              inst✝ : Fintype β
                                              r : α → β → Prop
                                              h₁ : Relator.LeftTotal r
                                              h₂ : Relator.LeftUnique r
                                              ⊢ ∀ (a : α), Membership.mem Finset.univ a → Exists fun b => And (Membership.me …
                                            -/
  card_le_card_of_forall_subsingleton r (by simpa using h₁) fun _ _ _ ha₁ _ ha₂ ↦ h₂ ha₁.2 ha₂.2
                                            /-
                                              🎉 no goals
                                            -/


theorem card_le_card_of_rightTotal_unique (h₁ : RightTotal r) (h₂ : RightUnique r) :
    Fintype.card β ≤ Fintype.card α :=
                                             /-
                                               α : Type u_2
                                               β : Type u_3
                                               inst✝¹ : Fintype α
                                               inst✝ : Fintype β
                                               r : α → β → Prop
                                               h₁ : Relator.RightTotal r
                                               h₂ : Relator.RightUnique r
                                               ⊢ ∀ (b : β), Membership.mem Finset.univ b → Exists fun a => And (Membership.me …
                                             -/
  card_le_card_of_forall_subsingleton' r (by simpa using h₁) fun _ _ _ ha₁ _ ha₂ ↦ h₂ ha₁.2 ha₂.2
                                             /-
                                               🎉 no goals
                                             -/


