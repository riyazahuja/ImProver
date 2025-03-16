theorem disjoint_iff_inter_eq_empty : Disjoint s t ↔ s ∩ t = ∅ :=
  disjoint_iff


@[simp]
theorem union_empty (s : Finset α) : s ∪ ∅ = s :=
                                     /-
                                       α : Type u_1
                                       inst✝ : DecidableEq α
                                       s : Finset α
                                       x : α
                                       ⊢ Iff (Or (Membership.mem s x) (Membership.mem EmptyCollection.emptyCollection …
                                     -/
  ext fun x => mem_union.trans <| by simp
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem empty_union (s : Finset α) : ∅ ∪ s = s :=
                                     /-
                                       α : Type u_1
                                       inst✝ : DecidableEq α
                                       s : Finset α
                                       x : α
                                       ⊢ Iff (Or (Membership.mem EmptyCollection.emptyCollection x) (Membership.mem s …
                                     -/
  ext fun x => mem_union.trans <| by simp
                                     /-
                                       🎉 no goals
                                     -/


@[aesop unsafe apply (rule_sets := [finsetNonempty])]
theorem Nonempty.inl {s t : Finset α} (h : s.Nonempty) : (s ∪ t).Nonempty :=
  h.mono subset_union_left


@[aesop unsafe apply (rule_sets := [finsetNonempty])]
theorem Nonempty.inr {s t : Finset α} (h : t.Nonempty) : (s ∪ t).Nonempty :=
  h.mono subset_union_right


theorem insert_eq (a : α) (s : Finset α) : insert a s = {a} ∪ s :=
  rfl


@[simp]
theorem insert_union (a : α) (s t : Finset α) : insert a s ∪ t = insert a (s ∪ t) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Finset α
    ⊢ Eq (Union.union (Insert.insert a s) t) (Insert.insert a (Union.union s t))
  -/
  simp only [insert_eq, union_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem union_insert (a : α) (s t : Finset α) : s ∪ insert a t = insert a (s ∪ t) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Finset α
    ⊢ Eq (Union.union s (Insert.insert a t)) (Insert.insert a (Union.union s t))
  -/
  simp only [insert_eq, union_left_comm]
  /-
    🎉 no goals
  -/


theorem insert_union_distrib (a : α) (s t : Finset α) :
    insert a (s ∪ t) = insert a s ∪ insert a t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Finset α
    ⊢ Eq (Insert.insert a (Union.union s t)) (Union.union (Insert.insert a s) (Ins …
  -/
  simp only [insert_union, union_insert, insert_idem]
  /-
    🎉 no goals
  -/


/-- To prove a relation on pairs of `Finset X`, it suffices to show that it is
  * symmetric,
  * it holds when one of the `Finset`s is empty,
  * it holds for pairs of singletons,
  * if it holds for `[a, c]` and for `[b, c]`, then it holds for `[a ∪ b, c]`.
-/
theorem induction_on_union (P : Finset α → Finset α → Prop) (symm : ∀ {a b}, P a b → P b a)
    (empty_right : ∀ {a}, P a ∅) (singletons : ∀ {a b}, P {a} {b})
    (union_of : ∀ {a b c}, P a c → P b c → P (a ∪ b) c) : ∀ a b, P a b := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    P : Finset α → Finset α → Prop
    symm : ∀ {a b : Finset α}, P a b → P b a
    empty_right : ∀ {a : Finset α}, P a EmptyCollection.emptyCollection
    singletons : ∀ {a b : α}, P (Singleton.singleton a) (Singleton.singleton b)
    union_of : ∀ {a b c : Finset α}, P a c → P b c → P (Union.union a b) c
    ⊢ ∀ (a b : Finset α), P a b
  -/
  intro a b
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    P : Finset α → Finset α → Prop
    symm : ∀ {a b : Finset α}, P a b → P b a
    empty_right : ∀ {a : Finset α}, P a EmptyCollection.emptyCollection
    singletons : ∀ {a b : α}, P (Singleton.singleton a) (Singleton.singleton b)
    union_of : ∀ {a b c : Finset α}, P a c → P b c → P (Union.union a b) c
    a b : Finset α
    ⊢ P a b
  -/
  refine Finset.induction_on b empty_right fun x s _xs hi => symm ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    P : Finset α → Finset α → Prop
    symm : ∀ {a b : Finset α}, P a b → P b a
    empty_right : ∀ {a : Finset α}, P a EmptyCollection.emptyCollection
    singletons : ∀ {a b : α}, P (Singleton.singleton a) (Singleton.singleton b)
    union_of : ∀ {a b c : Finset α}, P a c → P b c → P (Union.union a b) c
    a b : Finset α
    x : α
    s : Finset α
    _xs : Not (Membership.mem s x)
    hi : P a s
    ⊢ P (Insert.insert x s) a
  -/
  rw [Finset.insert_eq]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    P : Finset α → Finset α → Prop
    symm : ∀ {a b : Finset α}, P a b → P b a
    empty_right : ∀ {a : Finset α}, P a EmptyCollection.emptyCollection
    singletons : ∀ {a b : α}, P (Singleton.singleton a) (Singleton.singleton b)
    union_of : ∀ {a b c : Finset α}, P a c → P b c → P (Union.union a b) c
    a b : Finset α
    x : α
    s : Finset α
    _xs : Not (Membership.mem s x)
    hi : P a s
    ⊢ P (Union.union (Singleton.singleton x) s) a
  -/
  apply union_of _ (symm hi)
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    P : Finset α → Finset α → Prop
    symm : ∀ {a b : Finset α}, P a b → P b a
    empty_right : ∀ {a : Finset α}, P a EmptyCollection.emptyCollection
    singletons : ∀ {a b : α}, P (Singleton.singleton a) (Singleton.singleton b)
    union_of : ∀ {a b c : Finset α}, P a c → P b c → P (Union.union a b) c
    a b : Finset α
    x : α
    s : Finset α
    _xs : Not (Membership.mem s x)
    hi : P a s
    ⊢ P (Singleton.singleton x) a
  -/
  refine Finset.induction_on a empty_right fun a t _ta hi => symm ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    P : Finset α → Finset α → Prop
    symm : ∀ {a b : Finset α}, P a b → P b a
    empty_right : ∀ {a : Finset α}, P a EmptyCollection.emptyCollection
    singletons : ∀ {a b : α}, P (Singleton.singleton a) (Singleton.singleton b)
    union_of : ∀ {a b c : Finset α}, P a c → P b c → P (Union.union a b) c
    a✝ b : Finset α
    x : α
    s : Finset α
    _xs : Not (Membership.mem s x)
    hi✝ : P a✝ s
    a : α
    t : Finset α
    _ta : Not (Membership.mem t a)
    hi : P (Singleton.singleton x) t
    ⊢ P (Insert.insert a t) (Singleton.singleton x)
  -/
  rw [Finset.insert_eq]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    P : Finset α → Finset α → Prop
    symm : ∀ {a b : Finset α}, P a b → P b a
    empty_right : ∀ {a : Finset α}, P a EmptyCollection.emptyCollection
    singletons : ∀ {a b : α}, P (Singleton.singleton a) (Singleton.singleton b)
    union_of : ∀ {a b c : Finset α}, P a c → P b c → P (Union.union a b) c
    a✝ b : Finset α
    x : α
    s : Finset α
    _xs : Not (Membership.mem s x)
    hi✝ : P a✝ s
    a : α
    t : Finset α
    _ta : Not (Membership.mem t a)
    hi : P (Singleton.singleton x) t
    ⊢ P (Union.union (Singleton.singleton a) t) (Singleton.singleton x)
  -/
  exact union_of singletons (symm hi)
  /-
    🎉 no goals
  -/


@[simp]
theorem inter_empty (s : Finset α) : s ∩ ∅ = ∅ :=
                                     /-
                                       α : Type u_1
                                       inst✝ : DecidableEq α
                                       s : Finset α
                                       x✝ : α
                                       ⊢ Iff (And (Membership.mem s x✝) (Membership.mem EmptyCollection.emptyCollecti …
                                     -/
  ext fun _ => mem_inter.trans <| by simp
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem empty_inter (s : Finset α) : ∅ ∩ s = ∅ :=
                                     /-
                                       α : Type u_1
                                       inst✝ : DecidableEq α
                                       s : Finset α
                                       x✝ : α
                                       ⊢ Iff (And (Membership.mem EmptyCollection.emptyCollection x✝) (Membership.mem …
                                     -/
  ext fun _ => mem_inter.trans <| by simp
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem insert_inter_of_mem {s₁ s₂ : Finset α} {a : α} (h : a ∈ s₂) :
    insert a s₁ ∩ s₂ = insert a (s₁ ∩ s₂) :=
  ext fun x => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s₁ s₂ : Finset α
      a : α
      h : Membership.mem s₂ a
      x : α
      ⊢ Iff (Membership.mem (Inter.inter (Insert.insert a s₁) s₂) x) (Membership.mem …
    -/
    have : x = a ∨ x ∈ s₂ ↔ x ∈ s₂ := or_iff_right_of_imp <| by rintro rfl; exact h
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s₁ s₂ : Finset α
      a : α
      h : Membership.mem s₂ a
      x : α
      this : Iff (Or (Eq x a) (Membership.mem s₂ x)) (Membership.mem s₂ x)
      ⊢ Iff (Membership.mem (Inter.inter (Insert.insert a s₁) s₂) x) (Membership.mem …
    -/
    simp only [mem_inter, mem_insert, or_and_left, this]
    /-
      🎉 no goals
    -/


@[simp]
theorem inter_insert_of_mem {s₁ s₂ : Finset α} {a : α} (h : a ∈ s₁) :
                                                /-
                                                  α : Type u_1
                                                  inst✝ : DecidableEq α
                                                  s₁ s₂ : Finset α
                                                  a : α
                                                  h : Membership.mem s₁ a
                                                  ⊢ Eq (Inter.inter s₁ (Insert.insert a s₂)) (Insert.insert a (Inter.inter s₁ s₂))
                                                -/
    s₁ ∩ insert a s₂ = insert a (s₁ ∩ s₂) := by rw [inter_comm, insert_inter_of_mem h, inter_comm]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem insert_inter_of_not_mem {s₁ s₂ : Finset α} {a : α} (h : a ∉ s₂) :
    insert a s₁ ∩ s₂ = s₁ ∩ s₂ :=
  ext fun x => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s₁ s₂ : Finset α
      a : α
      h : Not (Membership.mem s₂ a)
      x : α
      ⊢ Iff (Membership.mem (Inter.inter (Insert.insert a s₁) s₂) x) (Membership.mem …
    -/
    have : ¬(x = a ∧ x ∈ s₂) := by rintro ⟨rfl, H⟩; exact h H
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s₁ s₂ : Finset α
      a : α
      h : Not (Membership.mem s₂ a)
      x : α
      this : Not (And (Eq x a) (Membership.mem s₂ x))
      ⊢ Iff (Membership.mem (Inter.inter (Insert.insert a s₁) s₂) x) (Membership.mem …
    -/
    simp only [mem_inter, mem_insert, or_and_right, this, false_or]
    /-
      🎉 no goals
    -/


@[simp]
theorem inter_insert_of_not_mem {s₁ s₂ : Finset α} {a : α} (h : a ∉ s₁) :
                                     /-
                                       α : Type u_1
                                       inst✝ : DecidableEq α
                                       s₁ s₂ : Finset α
                                       a : α
                                       h : Not (Membership.mem s₁ a)
                                       ⊢ Eq (Inter.inter s₁ (Insert.insert a s₂)) (Inter.inter s₁ s₂)
                                     -/
    s₁ ∩ insert a s₂ = s₁ ∩ s₂ := by rw [inter_comm, insert_inter_of_not_mem h, inter_comm]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem singleton_inter_of_mem {a : α} {s : Finset α} (H : a ∈ s) : {a} ∩ s = {a} :=
                                      /-
                                        α : Type u_1
                                        inst✝ : DecidableEq α
                                        a : α
                                        s : Finset α
                                        H : Membership.mem s a
                                        ⊢ Eq (Inter.inter (Insert.insert a EmptyCollection.emptyCollection) s) (Insert …
                                      -/
  show insert a ∅ ∩ s = insert a ∅ by rw [insert_inter_of_mem H, empty_inter]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem singleton_inter_of_not_mem {a : α} {s : Finset α} (H : a ∉ s) : {a} ∩ s = ∅ :=
  eq_empty_of_forall_not_mem <| by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Finset α
      H : Not (Membership.mem s a)
      ⊢ ∀ (x : α), Not (Membership.mem (Inter.inter (Singleton.singleton a) s) x)
    -/
    simp only [mem_inter, mem_singleton]; rintro x ⟨rfl, h⟩; exact H h
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma singleton_inter {a : α} {s : Finset α} :
    {a} ∩ s = if a ∈ s then {a} else ∅ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ⊢ Eq (Inter.inter (Singleton.singleton a) s) (ite (Membership.mem s a) (Single …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem inter_singleton_of_mem {a : α} {s : Finset α} (h : a ∈ s) : s ∩ {a} = {a} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    h : Membership.mem s a
    ⊢ Eq (Inter.inter s (Singleton.singleton a)) (Singleton.singleton a)
  -/
  rw [inter_comm, singleton_inter_of_mem h]
  /-
    🎉 no goals
  -/


@[simp]
theorem inter_singleton_of_not_mem {a : α} {s : Finset α} (h : a ∉ s) : s ∩ {a} = ∅ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    h : Not (Membership.mem s a)
    ⊢ Eq (Inter.inter s (Singleton.singleton a)) EmptyCollection.emptyCollection
  -/
  rw [inter_comm, singleton_inter_of_not_mem h]
  /-
    🎉 no goals
  -/


lemma inter_singleton {a : α} {s : Finset α} :
    s ∩ {a} = if a ∈ s then {a} else ∅ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ⊢ Eq (Inter.inter s (Singleton.singleton a)) (ite (Membership.mem s a) (Single …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


@[simp] lemma union_eq_empty : s ∪ t = ∅ ↔ s = ∅ ∧ t = ∅ := sup_eq_bot_iff

@[simp] lemma union_nonempty : (s ∪ t).Nonempty ↔ s.Nonempty ∨ t.Nonempty :=
  mod_cast Set.union_nonempty (α := α) (s := s) (t := t)


theorem insert_union_comm (s t : Finset α) (a : α) : insert a s ∪ t = s ∪ insert a t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ⊢ Eq (Union.union (Insert.insert a s) t) (Union.union s (Insert.insert a t))
  -/
  rw [insert_union, union_insert]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_append : toFinset (l ++ l') = l.toFinset ∪ l'.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    ⊢ Eq (HAppend.hAppend l l').toFinset (Union.union l.toFinset l'.toFinset)
  -/
  induction' l with hd tl hl
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      l l' : List α
      ⊢ Eq (HAppend.hAppend List.nil l').toFinset (Union.union List.nil.toFinset l'. …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      l l' : List α
      hd : α
      tl : List α
      hl : Eq (HAppend.hAppend tl l').toFinset (Union.union tl.toFinset l'.toFinset)
      ⊢ Eq (HAppend.hAppend (List.cons hd tl) l').toFinset (Union.union (List.cons h …
    -/
  · simp [hl]
    /-
      🎉 no goals
    -/


