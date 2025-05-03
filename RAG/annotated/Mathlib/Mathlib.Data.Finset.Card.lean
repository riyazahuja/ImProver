/-- `s.card` is the number of elements of `s`, aka its cardinality.

The notation `#s` can be accessed in the `Finset` locale. -/
def card (s : Finset α) : ℕ :=
  Multiset.card s.1


@[inherit_doc] scoped prefix:arg "#" => Finset.card


theorem card_def (s : Finset α) : #s = Multiset.card s.1 :=
  rfl


@[simp] lemma card_val (s : Finset α) : Multiset.card s.1 = #s := rfl


@[simp]
theorem card_mk {m nodup} : #(⟨m, nodup⟩ : Finset α) = Multiset.card m :=
  rfl


@[simp]
theorem card_empty : #(∅ : Finset α) = 0 :=
  rfl


@[gcongr]
theorem card_le_card : s ⊆ t → #s ≤ #t :=
  Multiset.card_le_card ∘ val_le_iff.mpr


@[mono]
                                             /-
                                               α : Type u_1
                                               ⊢ Monotone Finset.card
                                             -/
theorem card_mono : Monotone (@card α) := by apply card_le_card
                                             /-
                                               🎉 no goals
                                             -/


@[simp] lemma card_eq_zero : #s = 0 ↔ s = ∅ := Multiset.card_eq_zero.trans val_eq_zero

lemma card_ne_zero : #s ≠ 0 ↔ s.Nonempty := card_eq_zero.ne.trans nonempty_iff_ne_empty.symm

@[simp] lemma card_pos : 0 < #s ↔ s.Nonempty := Nat.pos_iff_ne_zero.trans card_ne_zero

@[simp] lemma one_le_card : 1 ≤ #s ↔ s.Nonempty := card_pos


alias ⟨_, Nonempty.card_pos⟩ := card_pos

alias ⟨_, Nonempty.card_ne_zero⟩ := card_ne_zero


theorem card_ne_zero_of_mem (h : a ∈ s) : #s ≠ 0 :=
  (not_congr card_eq_zero).2 <| ne_empty_of_mem h


@[simp]
theorem card_singleton (a : α) : #{a} = 1 :=
  Multiset.card_singleton _


theorem card_singleton_inter [DecidableEq α] : #({a} ∩ s) ≤ 1 := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    inst✝ : DecidableEq α
    ⊢ LE.le (Inter.inter (Singleton.singleton a) s).card 1
  -/
  cases' Finset.decidableMem a s with h h
    /-
      case isFalse
      α : Type u_1
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      h : Not (Membership.mem s a)
      ⊢ LE.le (Inter.inter (Singleton.singleton a) s).card 1
    -/
  · simp [Finset.singleton_inter_of_not_mem h]
    /-
      🎉 no goals
    -/
    /-
      case isTrue
      α : Type u_1
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      h : Membership.mem s a
      ⊢ LE.le (Inter.inter (Singleton.singleton a) s).card 1
    -/
  · simp [Finset.singleton_inter_of_mem h]
    /-
      🎉 no goals
    -/


@[simp]
theorem card_cons (h : a ∉ s) : #(s.cons a h) = #s + 1 :=
  Multiset.card_cons _ _


@[simp]
theorem card_insert_of_not_mem (h : a ∉ s) : #(insert a s) = #s + 1 := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    inst✝ : DecidableEq α
    h : Not (Membership.mem s a)
    ⊢ Eq (Insert.insert a s).card (HAdd.hAdd s.card 1)
  -/
  rw [← cons_eq_insert _ _ h, card_cons]
  /-
    🎉 no goals
  -/


                                                                  /-
                                                                    α : Type u_1
                                                                    s : Finset α
                                                                    a : α
                                                                    inst✝ : DecidableEq α
                                                                    h : Membership.mem s a
                                                                    ⊢ Eq (Insert.insert a s).card s.card
                                                                  -/
theorem card_insert_of_mem (h : a ∈ s) : #(insert a s) = #s := by rw [insert_eq_of_mem h]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem card_insert_le (a : α) (s : Finset α) : #(insert a s) ≤ #s + 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ⊢ LE.le (Insert.insert a s).card (HAdd.hAdd s.card 1)
  -/
  by_cases h : a ∈ s
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Finset α
      h : Membership.mem s a
      ⊢ LE.le (Insert.insert a s).card (HAdd.hAdd s.card 1)
    -/
  · rw [insert_eq_of_mem h]
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Finset α
      h : Membership.mem s a
      ⊢ LE.le s.card (HAdd.hAdd s.card 1)
    -/
    exact Nat.le_succ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Finset α
      h : Not (Membership.mem s a)
      ⊢ LE.le (Insert.insert a s).card (HAdd.hAdd s.card 1)
    -/
  · rw [card_insert_of_not_mem h]
    /-
      🎉 no goals
    -/


theorem card_le_two : #{a, b} ≤ 2 := card_insert_le _ _


theorem card_le_three : #{a, b, c} ≤ 3 :=
  (card_insert_le _ _).trans (Nat.succ_le_succ card_le_two)


theorem card_le_four : #{a, b, c, d} ≤ 4 :=
  (card_insert_le _ _).trans (Nat.succ_le_succ card_le_three)


theorem card_le_five : #{a, b, c, d, e} ≤ 5 :=
  (card_insert_le _ _).trans (Nat.succ_le_succ card_le_four)


theorem card_le_six : #{a, b, c, d, e, f} ≤ 6 :=
  (card_insert_le _ _).trans (Nat.succ_le_succ card_le_five)


/-- If `a ∈ s` is known, see also `Finset.card_insert_of_mem` and `Finset.card_insert_of_not_mem`.
-/
theorem card_insert_eq_ite : #(insert a s) = if a ∈ s then #s else #s + 1 := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    inst✝ : DecidableEq α
    ⊢ Eq (Insert.insert a s).card (ite (Membership.mem s a) s.card (HAdd.hAdd s.ca …
  -/
  by_cases h : a ∈ s
    /-
      case pos
      α : Type u_1
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      h : Membership.mem s a
      ⊢ Eq (Insert.insert a s).card (ite (Membership.mem s a) s.card (HAdd.hAdd s.ca …
    -/
  · rw [card_insert_of_mem h, if_pos h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      h : Not (Membership.mem s a)
      ⊢ Eq (Insert.insert a s).card (ite (Membership.mem s a) s.card (HAdd.hAdd s.ca …
    -/
  · rw [card_insert_of_not_mem h, if_neg h]
    /-
      🎉 no goals
    -/


@[simp]
theorem card_pair_eq_one_or_two : #{a, b} = 1 ∨ #{a, b} = 2 := by
  /-
    α : Type u_1
    a b : α
    inst✝ : DecidableEq α
    ⊢ Or (Eq (Insert.insert a (Singleton.singleton b)).card 1) (Eq (Insert.insert  …
  -/
  simp [card_insert_eq_ite]
  /-
    α : Type u_1
    a b : α
    inst✝ : DecidableEq α
    ⊢ Or (Eq a b) (Not (Eq a b))
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp]
theorem card_pair (h : a ≠ b) : #{a, b} = 2 := by
  /-
    α : Type u_1
    a b : α
    inst✝ : DecidableEq α
    h : Ne a b
    ⊢ Eq (Insert.insert a (Singleton.singleton b)).card 2
  -/
  rw [card_insert_of_not_mem (not_mem_singleton.2 h), card_singleton]
  /-
    🎉 no goals
  -/


/-- $\#(s \setminus \{a\}) = \#s - 1$ if $a \in s$. -/
@[simp]
theorem card_erase_of_mem : a ∈ s → #(s.erase a) = #s - 1 :=
  Multiset.card_erase_of_mem


/-- $\#(s \setminus \{a\}) = \#s - 1$ if $a \in s$.
  This result is casted to any additive group with 1,
  so that we don't have to work with `ℕ`-subtraction. -/
@[simp]
theorem cast_card_erase_of_mem {R} [AddGroupWithOne R] {s : Finset α} (hs : a ∈ s) :
    (#(s.erase a) : R) = #s - 1 := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : DecidableEq α
    R : Type u_4
    inst✝ : AddGroupWithOne R
    s : Finset α
    hs : Membership.mem s a
    ⊢ Eq (↑(s.erase a).card) (HSub.hSub (↑s.card) 1)
  -/
  rw [card_erase_of_mem hs, Nat.cast_sub, Nat.cast_one]
  /-
    α : Type u_1
    a : α
    inst✝¹ : DecidableEq α
    R : Type u_4
    inst✝ : AddGroupWithOne R
    s : Finset α
    hs : Membership.mem s a
    ⊢ LE.le 1 s.card
  -/
  rw [Nat.add_one_le_iff, Finset.card_pos]
  /-
    α : Type u_1
    a : α
    inst✝¹ : DecidableEq α
    R : Type u_4
    inst✝ : AddGroupWithOne R
    s : Finset α
    hs : Membership.mem s a
    ⊢ s.Nonempty
  -/
  exact ⟨a, hs⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem card_erase_add_one : a ∈ s → #(s.erase a) + 1 = #s :=
  Multiset.card_erase_add_one


theorem card_erase_lt_of_mem : a ∈ s → #(s.erase a) < #s :=
  Multiset.card_erase_lt_of_mem


theorem card_erase_le : #(s.erase a) ≤ #s :=
  Multiset.card_erase_le


theorem pred_card_le_card_erase : #s - 1 ≤ #(s.erase a) := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    inst✝ : DecidableEq α
    ⊢ LE.le (HSub.hSub s.card 1) (s.erase a).card
  -/
  by_cases h : a ∈ s
    /-
      case pos
      α : Type u_1
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      h : Membership.mem s a
      ⊢ LE.le (HSub.hSub s.card 1) (s.erase a).card
    -/
  · exact (card_erase_of_mem h).ge
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      h : Not (Membership.mem s a)
      ⊢ LE.le (HSub.hSub s.card 1) (s.erase a).card
    -/
  · rw [erase_eq_of_not_mem h]
    /-
      case neg
      α : Type u_1
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      h : Not (Membership.mem s a)
      ⊢ LE.le (HSub.hSub s.card 1) s.card
    -/
    exact Nat.sub_le _ _
    /-
      🎉 no goals
    -/


/-- If `a ∈ s` is known, see also `Finset.card_erase_of_mem` and `Finset.erase_eq_of_not_mem`. -/
theorem card_erase_eq_ite : #(s.erase a) = if a ∈ s then #s - 1 else #s :=
  Multiset.card_erase_eq_ite


@[simp]
theorem card_range (n : ℕ) : #(range n) = n :=
  Multiset.card_range n


@[simp]
theorem card_attach : #s.attach = #s :=
  Multiset.card_attach


theorem Multiset.card_toFinset : #m.toFinset = Multiset.card m.dedup :=
  rfl


theorem Multiset.toFinset_card_le : #m.toFinset ≤ Multiset.card m :=
  card_le_card <| dedup_le _


theorem Multiset.toFinset_card_of_nodup {m : Multiset α} (h : m.Nodup) :
    #m.toFinset = Multiset.card m :=
  congr_arg card <| Multiset.dedup_eq_self.mpr h


theorem Multiset.dedup_card_eq_card_iff_nodup {m : Multiset α} :
    card m.dedup = card m ↔ m.Nodup :=
  .trans ⟨fun h ↦ eq_of_le_of_card_le (dedup_le m) h.ge, congr_arg _⟩ dedup_eq_self


theorem Multiset.toFinset_card_eq_card_iff_nodup {m : Multiset α} :
    #m.toFinset = card m ↔ m.Nodup := dedup_card_eq_card_iff_nodup


theorem List.card_toFinset : #l.toFinset = l.dedup.length :=
  rfl


theorem List.toFinset_card_le : #l.toFinset ≤ l.length :=
  Multiset.toFinset_card_le ⟦l⟧


theorem List.toFinset_card_of_nodup {l : List α} (h : l.Nodup) : #l.toFinset = l.length :=
  Multiset.toFinset_card_of_nodup h


@[simp]
theorem length_toList (s : Finset α) : s.toList.length = #s := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq s.toList.length s.card
  -/
  rw [toList, ← Multiset.coe_card, Multiset.coe_toList, card_def]
  /-
    🎉 no goals
  -/


theorem card_image_le [DecidableEq β] : #(s.image f) ≤ #s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    f : α → β
    inst✝ : DecidableEq β
    ⊢ LE.le (Finset.image f s).card s.card
  -/
  simpa only [card_map] using (s.1.map f).toFinset_card_le
  /-
    🎉 no goals
  -/


theorem card_image_of_injOn [DecidableEq β] (H : Set.InjOn f s) : #(s.image f) = #s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    f : α → β
    inst✝ : DecidableEq β
    H : Set.InjOn f ↑s
    ⊢ Eq (Finset.image f s).card s.card
  -/
  simp only [card, image_val_of_injOn H, card_map]
  /-
    🎉 no goals
  -/


theorem injOn_of_card_image_eq [DecidableEq β] (H : #(s.image f) = #s) : Set.InjOn f s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    f : α → β
    inst✝ : DecidableEq β
    H : Eq (Finset.image f s).card s.card
    ⊢ Set.InjOn f ↑s
  -/
  rw [card_def, card_def, image, toFinset] at H
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    f : α → β
    inst✝ : DecidableEq β
    H : Eq { val := (Multiset.map f s.val).dedup, nodup := ⋯ }.val.card s.val.card
    ⊢ Set.InjOn f ↑s
  -/
  dsimp only at H
  have : (s.1.map f).dedup = s.1.map f := by
    refine Multiset.eq_of_le_of_card_le (Multiset.dedup_le _) ?_
    simp only [H, Multiset.card_map, le_rfl]
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    f : α → β
    inst✝ : DecidableEq β
    H : Eq (Multiset.map f s.val).dedup.card s.val.card
    this : Eq (Multiset.map f s.val).dedup (Multiset.map f s.val)
    ⊢ Set.InjOn f ↑s
  -/
  rw [Multiset.dedup_eq_self] at this
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    f : α → β
    inst✝ : DecidableEq β
    H : Eq (Multiset.map f s.val).dedup.card s.val.card
    this : (Multiset.map f s.val).Nodup
    ⊢ Set.InjOn f ↑s
  -/
  exact inj_on_of_nodup_map this
  /-
    🎉 no goals
  -/


theorem card_image_iff [DecidableEq β] : #(s.image f) = #s ↔ Set.InjOn f s :=
  ⟨injOn_of_card_image_eq, card_image_of_injOn⟩


theorem card_image_of_injective [DecidableEq β] (s : Finset α) (H : Injective f) :
    #(s.image f) = #s :=
  card_image_of_injOn fun _ _ _ _ h => H h


theorem fiber_card_ne_zero_iff_mem_image (s : Finset α) (f : α → β) [DecidableEq β] (y : β) :
    #(s.filter fun x ↦ f x = y) ≠ 0 ↔ y ∈ s.image f := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    f : α → β
    inst✝ : DecidableEq β
    y : β
    ⊢ Iff (Ne (Finset.filter (fun x => Eq (f x) y) s).card 0) (Membership.mem (Fin …
  -/
  rw [← Nat.pos_iff_ne_zero, card_pos, fiber_nonempty_iff_mem_image]
  /-
    🎉 no goals
  -/


lemma card_filter_le_iff (s : Finset α) (P : α → Prop) [DecidablePred P] (n : ℕ) :
    #(s.filter P) ≤ n ↔ ∀ s' ⊆ s, n < #s' → ∃ a ∈ s', ¬ P a :=
                                                                  /-
                                                                    α : Type u_1
                                                                    s : Finset α
                                                                    P : α → Prop
                                                                    inst✝ : DecidablePred P
                                                                    n : Nat
                                                                    H : ∀ (s' : Multiset α), LE.le s' s.val → LT.lt n s'.card → Exists fun a => An …
                                                                    s' : Finset α
                                                                    hs' : HasSubset.Subset s' s
                                                                    h : LT.lt n s'.card
                                                                    ⊢ LE.le s'.val s.val
                                                                  -/
  (s.1.card_filter_le_iff P n).trans ⟨fun H s' hs' h ↦ H s'.1 (by aesop) h,
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    fun H s' hs' h ↦ H ⟨s', nodup_of_le hs' s.2⟩ (fun _ hx ↦ Multiset.subset_of_le hs' hx) h⟩


@[simp]
theorem card_map (f : α ↪ β) : #(s.map f) = #s :=
  Multiset.card_map _ _


@[simp]
theorem card_subtype (p : α → Prop) [DecidablePred p] (s : Finset α) :
                                         /-
                                           α : Type u_1
                                           p : α → Prop
                                           inst✝ : DecidablePred p
                                           s : Finset α
                                           ⊢ Eq (Finset.subtype p s).card (Finset.filter p s).card
                                         -/
    #(s.subtype p) = #(s.filter p) := by simp [Finset.subtype]
                                         /-
                                           🎉 no goals
                                         -/


theorem card_filter_le (s : Finset α) (p : α → Prop) [DecidablePred p] :
    #(s.filter p) ≤ #s :=
  card_le_card <| filter_subset _ _


theorem eq_of_subset_of_card_le {s t : Finset α} (h : s ⊆ t) (h₂ : #t ≤ #s) : s = t :=
  eq_of_veq <| Multiset.eq_of_le_of_card_le (val_le_iff.mpr h) h₂


theorem eq_iff_card_le_of_subset (hst : s ⊆ t) : #t ≤ #s ↔ s = t :=
  ⟨eq_of_subset_of_card_le hst, (ge_of_eq <| congr_arg _ ·)⟩


theorem eq_of_superset_of_card_ge (hst : s ⊆ t) (hts : #t ≤ #s) : t = s :=
  (eq_of_subset_of_card_le hst hts).symm


theorem eq_iff_card_ge_of_superset (hst : s ⊆ t) : #t ≤ #s ↔ t = s :=
  (eq_iff_card_le_of_subset hst).trans eq_comm


theorem subset_iff_eq_of_card_le (h : #t ≤ #s) : s ⊆ t ↔ s = t :=
  ⟨fun hst => eq_of_subset_of_card_le hst h, Eq.subset'⟩


theorem map_eq_of_subset {f : α ↪ α} (hs : s.map f ⊆ s) : s.map f = s :=
  eq_of_subset_of_card_le hs (card_map _).ge


theorem card_filter_eq_iff {p : α → Prop} [DecidablePred p] :
    #(s.filter p) = #s ↔ ∀ x ∈ s, p x := by
  rw [(card_filter_le s p).eq_iff_not_lt, not_lt, eq_iff_card_le_of_subset (filter_subset p s),
    filter_eq_self]


alias ⟨filter_card_eq, _⟩ := card_filter_eq_iff


theorem card_filter_eq_zero_iff {p : α → Prop} [DecidablePred p] :
    #(s.filter p) = 0 ↔ ∀ x ∈ s, ¬ p x := by
  /-
    α : Type u_1
    s : Finset α
    p : α → Prop
    inst✝ : DecidablePred p
    ⊢ Iff (Eq (Finset.filter p s).card 0) (∀ (x : α), Membership.mem s x → Not (p  …
  -/
  rw [card_eq_zero, filter_eq_empty_iff]
  /-
    🎉 no goals
  -/


nonrec lemma card_lt_card (h : s ⊂ t) : #s < #t := card_lt_card <| val_lt_iff.2 h


lemma card_strictMono : StrictMono (card : Finset α → ℕ) := fun _ _ ↦ card_lt_card


theorem card_eq_of_bijective (f : ∀ i, i < n → α) (hf : ∀ a ∈ s, ∃ i, ∃ h : i < n, f i h = a)
    (hf' : ∀ i (h : i < n), f i h ∈ s)
    (f_inj : ∀ i j (hi : i < n) (hj : j < n), f i hi = f j hj → i = j) : #s = n := by
  classical
  have : s = (range n).attach.image fun i => f i.1 (mem_range.1 i.2) := by
    ext a
    suffices _ : a ∈ s ↔ ∃ (i : _) (hi : i ∈ range n), f i (mem_range.1 hi) = a by
      simpa only [mem_image, mem_attach, true_and, Subtype.exists]
    constructor
    · intro ha; obtain ⟨i, hi, rfl⟩ := hf a ha; use i, mem_range.2 hi
    · rintro ⟨i, hi, rfl⟩; apply hf'
  calc
    #s = #((range n).attach.image fun i => f i.1 (mem_range.1 i.2)) := by rw [this]
    _ = #(range n).attach := ?_
    _ = #(range n) := card_attach
    _ = n := card_range n
  apply card_image_of_injective
  intro ⟨i, hi⟩ ⟨j, hj⟩ eq
  exact Subtype.eq <| f_inj i j (mem_range.1 hi) (mem_range.1 hj) eq


/-- Reorder a finset.

The difference with `Finset.card_bij'` is that the bijection is specified as a surjective injection,
rather than by an inverse function.

The difference with `Finset.card_nbij` is that the bijection is allowed to use membership of the
domain, rather than being a non-dependent function. -/
lemma card_bij (i : ∀ a ∈ s, β) (hi : ∀ a ha, i a ha ∈ t)
    (i_inj : ∀ a₁ ha₁ a₂ ha₂, i a₁ ha₁ = i a₂ ha₂ → a₁ = a₂)
    (i_surj : ∀ b ∈ t, ∃ a ha, i a ha = b) : #s = #t := by
  classical
  calc
    #s = #s.attach := card_attach.symm
    _ = #(s.attach.image fun a ↦ i a.1 a.2) := Eq.symm ?_
    _ = #t := ?_
  · apply card_image_of_injective
    intro ⟨_, _⟩ ⟨_, _⟩ h
    simpa using i_inj _ _ _ _ h
  · congr 1
    ext b
    constructor <;> intro h
    · obtain ⟨_, _, rfl⟩ := mem_image.1 h; apply hi
    · obtain ⟨a, ha, rfl⟩ := i_surj b h; exact mem_image.2 ⟨⟨a, ha⟩, by simp⟩


@[deprecated (since := "2024-05-04")] alias card_congr := card_bij


/-- Reorder a finset.

The difference with `Finset.card_bij` is that the bijection is specified with an inverse, rather
than as a surjective injection.

The difference with `Finset.card_nbij'` is that the bijection and its inverse are allowed to use
membership of the domains, rather than being non-dependent functions. -/
lemma card_bij' (i : ∀ a ∈ s, β) (j : ∀ a ∈ t, α) (hi : ∀ a ha, i a ha ∈ t)
    (hj : ∀ a ha, j a ha ∈ s) (left_inv : ∀ a ha, j (i a ha) (hi a ha) = a)
    (right_inv : ∀ a ha, i (j a ha) (hj a ha) = a) : #s = #t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    i : (a : α) → Membership.mem s a → β
    j : (a : β) → Membership.mem t a → α
    hi : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (i a ha)
    hj : ∀ (a : β) (ha : Membership.mem t a), Membership.mem s (j a ha)
    left_inv : ∀ (a : α) (ha : Membership.mem s a), Eq (j (i a ha) ⋯) a
    right_inv : ∀ (a : β) (ha : Membership.mem t a), Eq (i (j a ha) ⋯) a
    ⊢ Eq s.card t.card
  -/
  refine card_bij i hi (fun a1 h1 a2 h2 eq ↦ ?_) (fun b hb ↦ ⟨_, hj b hb, right_inv b hb⟩)
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    i : (a : α) → Membership.mem s a → β
    j : (a : β) → Membership.mem t a → α
    hi : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (i a ha)
    hj : ∀ (a : β) (ha : Membership.mem t a), Membership.mem s (j a ha)
    left_inv : ∀ (a : α) (ha : Membership.mem s a), Eq (j (i a ha) ⋯) a
    right_inv : ∀ (a : β) (ha : Membership.mem t a), Eq (i (j a ha) ⋯) a
    a1 : α
    h1 : Membership.mem s a1
    a2 : α
    h2 : Membership.mem s a2
    eq : Eq (i a1 h1) (i a2 h2)
    ⊢ Eq a1 a2
  -/
  rw [← left_inv a1 h1, ← left_inv a2 h2]
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    i : (a : α) → Membership.mem s a → β
    j : (a : β) → Membership.mem t a → α
    hi : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (i a ha)
    hj : ∀ (a : β) (ha : Membership.mem t a), Membership.mem s (j a ha)
    left_inv : ∀ (a : α) (ha : Membership.mem s a), Eq (j (i a ha) ⋯) a
    right_inv : ∀ (a : β) (ha : Membership.mem t a), Eq (i (j a ha) ⋯) a
    a1 : α
    h1 : Membership.mem s a1
    a2 : α
    h2 : Membership.mem s a2
    eq : Eq (i a1 h1) (i a2 h2)
    ⊢ Eq (j (i a1 h1) ⋯) (j (i a2 h2) ⋯)
  -/
  simp only [eq]
  /-
    🎉 no goals
  -/


/-- Reorder a finset.

The difference with `Finset.card_nbij'` is that the bijection is specified as a surjective
injection, rather than by an inverse function.

The difference with `Finset.card_bij` is that the bijection is a non-dependent function, rather than
being allowed to use membership of the domain. -/
lemma card_nbij (i : α → β) (hi : ∀ a ∈ s, i a ∈ t) (i_inj : (s : Set α).InjOn i)
    (i_surj : (s : Set α).SurjOn i t) : #s = #t :=
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          s : Finset α
                                          t : Finset β
                                          i : α → β
                                          hi : ∀ (a : α), Membership.mem s a → Membership.mem t (i a)
                                          i_inj : Set.InjOn i ↑s
                                          i_surj : Set.SurjOn i ↑s ↑t
                                          ⊢ ∀ (b : β), Membership.mem t b → Exists fun a => Exists fun ha => Eq ((fun a  …
                                        -/
  card_bij (fun a _ ↦ i a) hi i_inj (by simpa using i_surj)
                                        /-
                                          🎉 no goals
                                        -/


/-- Reorder a finset.

The difference with `Finset.card_nbij` is that the bijection is specified with an inverse, rather
than as a surjective injection.

The difference with `Finset.card_bij'` is that the bijection and its inverse are non-dependent
functions, rather than being allowed to use membership of the domains.

The difference with `Finset.card_equiv` is that bijectivity is only required to hold on the domains,
rather than on the entire types. -/
lemma card_nbij' (i : α → β) (j : β → α) (hi : ∀ a ∈ s, i a ∈ t) (hj : ∀ a ∈ t, j a ∈ s)
    (left_inv : ∀ a ∈ s, j (i a) = a) (right_inv : ∀ a ∈ t, i (j a) = a) : #s = #t :=
  card_bij' (fun a _ ↦ i a) (fun b _ ↦ j b) hi hj left_inv right_inv


/-- Specialization of `Finset.card_nbij'` that automatically fills in most arguments.

See `Fintype.card_equiv` for the version where `s` and `t` are `univ`. -/
lemma card_equiv (e : α ≃ β) (hst : ∀ i, i ∈ s ↔ e i ∈ t) : #s = #t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    e : Equiv α β
    hst : ∀ (i : α), Iff (Membership.mem s i) (Membership.mem t (e i))
    ⊢ Eq s.card t.card
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
  refine card_nbij' e e.symm ?_ ?_ ?_ ?_ <;> simp [hst]
                                             /-
                                               🎉 no goals
                                             -/


/-- Specialization of `Finset.card_nbij` that automatically fills in most arguments.

See `Fintype.card_bijective` for the version where `s` and `t` are `univ`. -/
lemma card_bijective (e : α → β) (he : e.Bijective) (hst : ∀ i, i ∈ s ↔ e i ∈ t) :
    #s = #t := card_equiv (.ofBijective e he) hst


lemma card_le_card_of_injOn (f : α → β) (hf : ∀ a ∈ s, f a ∈ t) (f_inj : (s : Set α).InjOn f) :
    #s ≤ #t := by
  classical
  calc
    #s = #(s.image f) := (card_image_of_injOn f_inj).symm
    _  ≤ #t           := card_le_card <| image_subset_iff.2 hf

@[deprecated (since := "2024-06-01")] alias card_le_card_of_inj_on := card_le_card_of_injOn


lemma card_le_card_of_surjOn (f : α → β) (hf : Set.SurjOn f s t) : #t ≤ #s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    f : α → β
    hf : Set.SurjOn f ↑s ↑t
    ⊢ LE.le t.card s.card
  -/
  classical unfold Set.SurjOn at hf; exact (card_le_card (mod_cast hf)).trans card_image_le
  /-
    🎉 no goals
  -/


/-- If there are more pigeons than pigeonholes, then there are two pigeons in the same pigeonhole.
-/
theorem exists_ne_map_eq_of_card_lt_of_maps_to {t : Finset β} (hc : #t < #s) {f : α → β}
    (hf : ∀ a ∈ s, f a ∈ t) : ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ f x = f y := by
  classical
  by_contra! hz
  refine hc.not_le (card_le_card_of_injOn f hf ?_)
  intro x hx y hy
  contrapose
  exact hz x hx y hy


lemma le_card_of_inj_on_range (f : ℕ → α) (hf : ∀ i < n, f i ∈ s)
    (f_inj : ∀ i < n, ∀ j < n, f i = f j → i = j) : n ≤ #s :=
  calc
    n = #(range n) := (card_range n).symm
                                          /-
                                            α : Type u_1
                                            s : Finset α
                                            n : Nat
                                            f : Nat → α
                                            hf : ∀ (i : Nat), LT.lt i n → Membership.mem s (f i)
                                            f_inj : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → Eq (f i) (f j) → Eq  …
                                            ⊢ ∀ (a : Nat), Membership.mem (Finset.range n) a → Membership.mem s (f a)
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
    _ ≤ #s := card_le_card_of_injOn f (by simpa only [mem_range]) (by simpa)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma surj_on_of_inj_on_of_card_le (f : ∀ a ∈ s, β) (hf : ∀ a ha, f a ha ∈ t)
    (hinj : ∀ a₁ a₂ ha₁ ha₂, f a₁ ha₁ = f a₂ ha₂ → a₁ = a₂) (hst : #t ≤ #s) :
    ∀ b ∈ t, ∃ a ha, b = f a ha := by
  classical
  have h : #(s.attach.image fun a : s ↦ f a a.2) = #s := by
    rw [← @card_attach _ s, card_image_of_injective]
    intro ⟨_, _⟩ ⟨_, _⟩ h
    exact Subtype.eq <| hinj _ _ _ _ h
  obtain rfl : image (fun a : { a // a ∈ s } => f a a.prop) s.attach = t :=
    eq_of_subset_of_card_le (image_subset_iff.2 <| by simpa) (by simp [hst, h])
  simp only [mem_image, mem_attach, true_and, Subtype.exists, forall_exists_index]
  exact fun b a ha hb ↦ ⟨a, ha, hb.symm⟩


theorem inj_on_of_surj_on_of_card_le (f : ∀ a ∈ s, β) (hf : ∀ a ha, f a ha ∈ t)
    (hsurj : ∀ b ∈ t, ∃ a ha, f a ha = b) (hst : #s ≤ #t) ⦃a₁⦄ (ha₁ : a₁ ∈ s) ⦃a₂⦄
    (ha₂ : a₂ ∈ s) (ha₁a₂ : f a₁ ha₁ = f a₂ ha₂) : a₁ = a₂ :=
  haveI : Inhabited { x // x ∈ s } := ⟨⟨a₁, ha₁⟩⟩
  let f' : { x // x ∈ s } → { x // x ∈ t } := fun x => ⟨f x.1 x.2, hf x.1 x.2⟩
  let g : { x // x ∈ t } → { x // x ∈ s } :=
    @surjInv _ _ f' fun x =>
      let ⟨y, hy₁, hy₂⟩ := hsurj x.1 x.2
      ⟨⟨y, hy₁⟩, Subtype.eq hy₂⟩
  have hg : Injective g := injective_surjInv _
  have hsg : Surjective g := fun x =>
    let ⟨y, hy⟩ :=
      surj_on_of_inj_on_of_card_le (fun (x : { x // x ∈ t }) (_ : x ∈ t.attach) => g x)
                                                                                             /-
                                                                                               α : Type u_1
                                                                                               β : Type u_2
                                                                                               s : Finset α
                                                                                               t : Finset β
                                                                                               f : (a : α) → Membership.mem s a → β
                                                                                               hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
                                                                                               hsurj : ∀ (b : β), Membership.mem t b → Exists fun a => Exists fun ha => Eq (f …
                                                                                               hst : LE.le s.card t.card
                                                                                               a₁ : α
                                                                                               ha₁ : Membership.mem s a₁
                                                                                               a₂ : α
                                                                                               ha₂ : Membership.mem s a₂
                                                                                               ha₁a₂ : Eq (f a₁ ha₁) (f a₂ ha₂)
                                                                                               this : Inhabited (Subtype fun x => Membership.mem s x)
                                                                                               f' : (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership.mem t …
                                                                                               g : (Subtype fun x => Membership.mem t x) → Subtype fun x => Membership.mem s  …
                                                                                               hg : Function.Injective g
                                                                                               x : Subtype fun x => Membership.mem s x
                                                                                               ⊢ LE.le s.attach.card t.attach.card
                                                                                             -/
        (fun x _ => show g x ∈ s.attach from mem_attach _ _) (fun _ _ _ _ hxy => hg hxy) (by simpa)
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/
        x (mem_attach _ _)
    ⟨y, hy.snd.symm⟩
  have hif : Injective f' :=
    (leftInverse_of_surjective_of_rightInverse hsg (rightInverse_surjInv _)).injective
  Subtype.ext_iff_val.1 (@hif ⟨a₁, ha₁⟩ ⟨a₂, ha₂⟩ (Subtype.eq ha₁a₂))


@[simp]
theorem card_disjUnion (s t : Finset α) (h) : #(s.disjUnion t h) = #s + #t :=
  Multiset.card_add _ _


theorem card_union_add_card_inter (s t : Finset α) :
    #(s ∪ t) + #(s ∩ t) = #s + #t :=
                            /-
                              α : Type u_1
                              inst✝ : DecidableEq α
                              s t : Finset α
                              ⊢ Eq (HAdd.hAdd (Union.union s EmptyCollection.emptyCollection).card (Inter.in …
                            -/
                            /-
                              🎉 no goals
                            -/
  Finset.induction_on t (by simp) fun a r har h => by by_cases a ∈ s <;>
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      a : α
      r : Finset α
      har : Not (Membership.mem r a)
      h : Eq (HAdd.hAdd (Union.union s r).card (Inter.inter s r).card) (HAdd.hAdd s. …
      h✝ : Membership.mem s a
      ⊢ Eq (HAdd.hAdd (Union.union s (Insert.insert a r)).card (Inter.inter s (Inser …
    -/
    /-
      🎉 no goals
    -/
    simp [*, ← add_assoc, add_right_comm _ 1]
    /-
      🎉 no goals
    -/


theorem card_inter_add_card_union (s t : Finset α) :
                                        /-
                                          α : Type u_1
                                          inst✝ : DecidableEq α
                                          s t : Finset α
                                          ⊢ Eq (HAdd.hAdd (Inter.inter s t).card (Union.union s t).card) (HAdd.hAdd s.ca …
                                        -/
    #(s ∩ t) + #(s ∪ t) = #s + #t := by rw [add_comm, card_union_add_card_inter]
                                        /-
                                          🎉 no goals
                                        -/


lemma card_union (s t : Finset α) : #(s ∪ t) = #s + #t - #(s ∩ t) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (Union.union s t).card (HSub.hSub (HAdd.hAdd s.card t.card) (Inter.inter  …
  -/
  rw [← card_union_add_card_inter, Nat.add_sub_cancel]
  /-
    🎉 no goals
  -/


lemma card_inter (s t : Finset α) : #(s ∩ t) = #s + #t - #(s ∪ t) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (Inter.inter s t).card (HSub.hSub (HAdd.hAdd s.card t.card) (Union.union  …
  -/
  rw [← card_inter_add_card_union, Nat.add_sub_cancel]
  /-
    🎉 no goals
  -/


theorem card_union_le (s t : Finset α) : #(s ∪ t) ≤ #s + #t :=
  card_union_add_card_inter s t ▸ Nat.le_add_right _ _


lemma card_union_eq_card_add_card : #(s ∪ t) = #s + #t ↔ Disjoint s t := by
  /-
    α : Type u_1
    s t : Finset α
    inst✝ : DecidableEq α
    ⊢ Iff (Eq (Union.union s t).card (HAdd.hAdd s.card t.card)) (Disjoint s t)
  -/
  rw [← card_union_add_card_inter]; simp [disjoint_iff_inter_eq_empty]
                                    /-
                                      🎉 no goals
                                    -/


@[simp] alias ⟨_, card_union_of_disjoint⟩ := card_union_eq_card_add_card


@[deprecated (since := "2024-02-09")] alias card_union_eq := card_union_of_disjoint

@[deprecated (since := "2024-02-09")] alias card_disjoint_union := card_union_of_disjoint


lemma cast_card_inter [AddGroupWithOne R] :
    (#(s ∩ t) : R) = #s + #t - #(s ∪ t) := by
  /-
    α : Type u_1
    R : Type u_3
    s t : Finset α
    inst✝¹ : DecidableEq α
    inst✝ : AddGroupWithOne R
    ⊢ Eq (↑(Inter.inter s t).card) (HSub.hSub (HAdd.hAdd ↑s.card ↑t.card) ↑(Union. …
  -/
  rw [eq_sub_iff_add_eq, ← cast_add, card_inter_add_card_union, cast_add]
  /-
    🎉 no goals
  -/


lemma cast_card_union [AddGroupWithOne R] :
    (#(s ∪ t) : R) = #s + #t - #(s ∩ t) := by
  /-
    α : Type u_1
    R : Type u_3
    s t : Finset α
    inst✝¹ : DecidableEq α
    inst✝ : AddGroupWithOne R
    ⊢ Eq (↑(Union.union s t).card) (HSub.hSub (HAdd.hAdd ↑s.card ↑t.card) ↑(Inter. …
  -/
  rw [eq_sub_iff_add_eq, ← cast_add, card_union_add_card_inter, cast_add]
  /-
    🎉 no goals
  -/


theorem card_sdiff (h : s ⊆ t) : #(t \ s) = #t - #s := by
  /-
    α : Type u_1
    s t : Finset α
    inst✝ : DecidableEq α
    h : HasSubset.Subset s t
    ⊢ Eq (SDiff.sdiff t s).card (HSub.hSub t.card s.card)
  -/
  suffices #(t \ s) = #(t \ s ∪ s) - #s by rwa [sdiff_union_of_subset h] at this
  /-
    α : Type u_1
    s t : Finset α
    inst✝ : DecidableEq α
    h : HasSubset.Subset s t
    ⊢ Eq (SDiff.sdiff t s).card (HSub.hSub (Union.union (SDiff.sdiff t s) s).card  …
  -/
  rw [card_union_of_disjoint sdiff_disjoint, Nat.add_sub_cancel_right]
  /-
    🎉 no goals
  -/


lemma cast_card_sdiff [AddGroupWithOne R] (h : s ⊆ t) : (#(t \ s) : R) = #t - #s := by
  /-
    α : Type u_1
    R : Type u_3
    s t : Finset α
    inst✝¹ : DecidableEq α
    inst✝ : AddGroupWithOne R
    h : HasSubset.Subset s t
    ⊢ Eq (↑(SDiff.sdiff t s).card) (HSub.hSub ↑t.card ↑s.card)
  -/
  rw [card_sdiff h, Nat.cast_sub (card_mono h)]
  /-
    🎉 no goals
  -/


theorem card_sdiff_add_card_eq_card {s t : Finset α} (h : s ⊆ t) : #(t \ s) + #s = #t :=
  ((Nat.sub_eq_iff_eq_add (card_le_card h)).mp (card_sdiff h).symm).symm


theorem le_card_sdiff (s t : Finset α) : #t - #s ≤ #(t \ s) :=
  calc
    #t - #s ≤ #t - #(s ∩ t) :=
      Nat.sub_le_sub_left (card_le_card inter_subset_left) _
    _ = #(t \ (s ∩ t)) := (card_sdiff inter_subset_right).symm
                       /-
                         α : Type u_1
                         inst✝ : DecidableEq α
                         s t : Finset α
                         ⊢ LE.le (SDiff.sdiff t (Inter.inter s t)).card (SDiff.sdiff t s).card
                       -/
    _ ≤ #(t \ s) := by rw [sdiff_inter_self_right t s]
                       /-
                         🎉 no goals
                       -/


theorem card_le_card_sdiff_add_card : #s ≤ #(s \ t) + #t :=
  Nat.sub_le_iff_le_add.1 <| le_card_sdiff _ _


theorem card_sdiff_add_card (s t : Finset α) : #(s \ t) + #t = #(s ∪ t) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff s t).card t.card) (Union.union s t).card
  -/
  rw [← card_union_of_disjoint sdiff_disjoint, sdiff_union_self_eq_union]
  /-
    🎉 no goals
  -/


lemma card_sdiff_comm (h : #s = #t) : #(s \ t) = #(t \ s) :=
  add_left_injective #t <| by
    /-
      α : Type u_1
      s t : Finset α
      inst✝ : DecidableEq α
      h : Eq s.card t.card
      ⊢ Eq ((fun x => HAdd.hAdd x t.card) (SDiff.sdiff s t).card) ((fun x => HAdd.hA …
    -/
    simp_rw [card_sdiff_add_card, ← h, card_sdiff_add_card, union_comm]
    /-
      🎉 no goals
    -/


@[simp]
lemma card_sdiff_add_card_inter (s t : Finset α) :
    #(s \ t) + #(s ∩ t) = #s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff s t).card (Inter.inter s t).card) s.card
  -/
  rw [← card_union_of_disjoint (disjoint_sdiff_inter _ _), sdiff_union_inter]
  /-
    🎉 no goals
  -/


@[simp]
lemma card_inter_add_card_sdiff (s t : Finset α) :
    #(s ∩ t) + #(s \ t) = #s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HAdd.hAdd (Inter.inter s t).card (SDiff.sdiff s t).card) s.card
  -/
  rw [add_comm, card_sdiff_add_card_inter]
  /-
    🎉 no goals
  -/


/-- **Pigeonhole principle** for two finsets inside an ambient finset. -/
theorem inter_nonempty_of_card_lt_card_add_card (hts : t ⊆ s) (hus : u ⊆ s)
    (hstu : #s < #t + #u) : (t ∩ u).Nonempty := by
  /-
    α : Type u_1
    s t u : Finset α
    inst✝ : DecidableEq α
    hts : HasSubset.Subset t s
    hus : HasSubset.Subset u s
    hstu : LT.lt s.card (HAdd.hAdd t.card u.card)
    ⊢ (Inter.inter t u).Nonempty
  -/
  contrapose! hstu
  calc
    _ = #(t ∪ u) := by simp [← card_union_add_card_inter, not_nonempty_iff_eq_empty.1 hstu]
    _ ≤ #s := by gcongr; exact union_subset hts hus


theorem filter_card_add_filter_neg_card_eq_card
    (p : α → Prop) [DecidablePred p] [∀ x, Decidable (¬p x)] :
    #(s.filter p) + #(s.filter fun a ↦ ¬ p a) = #s := by
  classical
  rw [← card_union_of_disjoint (disjoint_filter_filter_neg _ _ _), filter_union_filter_neg_eq]


/-- Given a subset `s` of a set `t`, of sizes at most and at least `n` respectively, there exists a
set `u` of size `n` which is both a superset of `s` and a subset of `t`. -/
lemma exists_subsuperset_card_eq (hst : s ⊆ t) (hsn : #s ≤ n) (hnt : n ≤ #t) :
    ∃ u, s ⊆ u ∧ u ⊆ t ∧ #u = n := by
  classical
  refine Nat.decreasingInduction' ?_ hnt ⟨t, by simp [hst]⟩
  intro k _ hnk ⟨u, hu₁, hu₂, hu₃⟩
  obtain ⟨a, ha⟩ : (u \ s).Nonempty := by rw [← card_pos, card_sdiff hu₁]; omega
  simp only [mem_sdiff] at ha
  exact ⟨u.erase a, by simp [subset_erase, erase_subset_iff_of_mem (hu₂ _), *]⟩


/-- We can shrink a set to any smaller size. -/
lemma exists_subset_card_eq (hns : n ≤ #s) : ∃ t ⊆ s, #t = n := by
  /-
    α : Type u_1
    s : Finset α
    n : Nat
    hns : LE.le n s.card
    ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq t.card n)
  -/
  simpa using exists_subsuperset_card_eq s.empty_subset (by simp) hns
  /-
    🎉 no goals
  -/


/-- Given a set `A` and a set `B` inside it, we can shrink `A` to any appropriate size, and keep `B`
inside it. -/
@[deprecated exists_subsuperset_card_eq (since := "2024-06-23")]
theorem exists_intermediate_set {A B : Finset α} (i : ℕ) (h₁ : i + #B ≤ #A) (h₂ : B ⊆ A) :
    ∃ C : Finset α, B ⊆ C ∧ C ⊆ A ∧ #C = i + #B :=
  exists_subsuperset_card_eq h₂ (Nat.le_add_left ..) h₁


/-- We can shrink `A` to any smaller size. -/
@[deprecated exists_subset_card_eq (since := "2024-06-23")]
theorem exists_smaller_set (A : Finset α) (i : ℕ) (h₁ : i ≤ #A) :
    ∃ B : Finset α, B ⊆ A ∧ #B = i := exists_subset_card_eq h₁


theorem le_card_iff_exists_subset_card : n ≤ #s ↔ ∃ t ⊆ s, #t = n := by
  /-
    α : Type u_1
    s : Finset α
    n : Nat
    ⊢ Iff (LE.le n s.card) (Exists fun t => And (HasSubset.Subset t s) (Eq t.card  …
  -/
  refine ⟨fun h => ?_, fun ⟨t, hst, ht⟩ => ht ▸ card_le_card hst⟩
  /-
    α : Type u_1
    s : Finset α
    n : Nat
    h : LE.le n s.card
    ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq t.card n)
  -/
  exact exists_subset_card_eq h
  /-
    🎉 no goals
  -/


theorem exists_subset_or_subset_of_two_mul_lt_card [DecidableEq α] {X Y : Finset α} {n : ℕ}
    (hXY : 2 * n < #(X ∪ Y)) : ∃ C : Finset α, n < #C ∧ (C ⊆ X ∨ C ⊆ Y) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    X Y : Finset α
    n : Nat
    hXY : LT.lt (HMul.hMul 2 n) (Union.union X Y).card
    ⊢ Exists fun C => And (LT.lt n C.card) (Or (HasSubset.Subset C X) (HasSubset.S …
  -/
  have h₁ : #(X ∩ (Y \ X)) = 0 := Finset.card_eq_zero.mpr (Finset.inter_sdiff_self X Y)
  have h₂ : #(X ∪ Y) = #X + #(Y \ X) := by
    rw [← card_union_add_card_inter X (Y \ X), Finset.union_sdiff_self_eq_union, h₁, add_zero]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    X Y : Finset α
    n : Nat
    hXY : LT.lt (HMul.hMul 2 n) (Union.union X Y).card
    h₁ : Eq (Inter.inter X (SDiff.sdiff Y X)).card 0
    h₂ : Eq (Union.union X Y).card (HAdd.hAdd X.card (SDiff.sdiff Y X).card)
    ⊢ Exists fun C => And (LT.lt n C.card) (Or (HasSubset.Subset C X) (HasSubset.S …
  -/
  rw [h₂, Nat.two_mul] at hXY
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    X Y : Finset α
    n : Nat
    hXY : LT.lt (HAdd.hAdd n n) (HAdd.hAdd X.card (SDiff.sdiff Y X).card)
    h₁ : Eq (Inter.inter X (SDiff.sdiff Y X)).card 0
    h₂ : Eq (Union.union X Y).card (HAdd.hAdd X.card (SDiff.sdiff Y X).card)
    ⊢ Exists fun C => And (LT.lt n C.card) (Or (HasSubset.Subset C X) (HasSubset.S …
  -/
  obtain h | h : n < #X ∨ n < #(Y \ X) := by contrapose! hXY; omega
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      X Y : Finset α
      n : Nat
      hXY : LT.lt (HAdd.hAdd n n) (HAdd.hAdd X.card (SDiff.sdiff Y X).card)
      h₁ : Eq (Inter.inter X (SDiff.sdiff Y X)).card 0
      h₂ : Eq (Union.union X Y).card (HAdd.hAdd X.card (SDiff.sdiff Y X).card)
      h : LT.lt n X.card
      ⊢ Exists fun C => And (LT.lt n C.card) (Or (HasSubset.Subset C X) (HasSubset.S …
    -/
  · exact ⟨X, h, Or.inl (Finset.Subset.refl X)⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : DecidableEq α
      X Y : Finset α
      n : Nat
      hXY : LT.lt (HAdd.hAdd n n) (HAdd.hAdd X.card (SDiff.sdiff Y X).card)
      h₁ : Eq (Inter.inter X (SDiff.sdiff Y X)).card 0
      h₂ : Eq (Union.union X Y).card (HAdd.hAdd X.card (SDiff.sdiff Y X).card)
      h : LT.lt n (SDiff.sdiff Y X).card
      ⊢ Exists fun C => And (LT.lt n C.card) (Or (HasSubset.Subset C X) (HasSubset.S …
    -/
  · exact ⟨Y \ X, h, Or.inr sdiff_subset⟩
    /-
      🎉 no goals
    -/


theorem card_eq_one : #s = 1 ↔ ∃ a, s = {a} := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (Eq s.card 1) (Exists fun a => Eq s (Singleton.singleton a))
  -/
  cases s
  /-
    case mk
    α : Type u_1
    val✝ : Multiset α
    nodup✝ : val✝.Nodup
    ⊢ Iff (Eq { val := val✝, nodup := nodup✝ }.card 1) (Exists fun a => Eq { val : …
  -/
  simp only [Multiset.card_eq_one, Finset.card, ← val_inj, singleton_val]
  /-
    🎉 no goals
  -/


theorem _root_.Multiset.toFinset_card_eq_one_iff [DecidableEq α] (s : Multiset α) :
    #s.toFinset = 1 ↔ Multiset.card s ≠ 0 ∧ ∃ a : α, s = Multiset.card s • {a} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Iff (Eq s.toFinset.card 1) (And (Ne s.card 0) (Exists fun a => Eq s (HSMul.h …
  -/
  simp_rw [card_eq_one, Multiset.toFinset_eq_singleton_iff, exists_and_left]
  /-
    🎉 no goals
  -/


theorem exists_eq_insert_iff [DecidableEq α] {s t : Finset α} :
    (∃ a ∉ s, insert a s = t) ↔ s ⊆ t ∧ #s + 1 = #t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Iff (Exists fun a => And (Not (Membership.mem s a)) (Eq (Insert.insert a s)  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      ⊢ (Exists fun a => And (Not (Membership.mem s a)) (Eq (Insert.insert a s) t))  …
    -/
  · rintro ⟨a, ha, rfl⟩
    /-
      case mp.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      ha : Not (Membership.mem s a)
      ⊢ And (HasSubset.Subset s (Insert.insert a s)) (Eq (HAdd.hAdd s.card 1) (Inser …
    -/
    exact ⟨subset_insert _ _, (card_insert_of_not_mem ha).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      ⊢ And (HasSubset.Subset s t) (Eq (HAdd.hAdd s.card 1) t.card) → Exists fun a = …
    -/
  · rintro ⟨hst, h⟩
    obtain ⟨a, ha⟩ : ∃ a, t \ s = {a} :=
      card_eq_one.1 (by rw [card_sdiff hst, ← h, Nat.add_sub_cancel_left])
    refine
      ⟨a, fun hs => (?_ : a ∉ {a}) <| mem_singleton_self _, by
        rw [insert_eq, ← ha, sdiff_union_of_subset hst]⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      hst : HasSubset.Subset s t
      h : Eq (HAdd.hAdd s.card 1) t.card
      a : α
      ha : Eq (SDiff.sdiff t s) (Singleton.singleton a)
      hs : Membership.mem s a
      ⊢ Not (Membership.mem (Singleton.singleton a) a)
    -/
    rw [← ha]
    /-
      case mpr.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      hst : HasSubset.Subset s t
      h : Eq (HAdd.hAdd s.card 1) t.card
      a : α
      ha : Eq (SDiff.sdiff t s) (Singleton.singleton a)
      hs : Membership.mem s a
      ⊢ Not (Membership.mem (SDiff.sdiff t s) a)
    -/
    exact not_mem_sdiff_of_mem_right hs
    /-
      🎉 no goals
    -/


theorem card_le_one : #s ≤ 1 ↔ ∀ a ∈ s, ∀ b ∈ s, a = b := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (LE.le s.card 1) (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership. …
  -/
  obtain rfl | ⟨x, hx⟩ := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_1
      ⊢ Iff (LE.le EmptyCollection.emptyCollection.card 1) (∀ (a : α), Membership.me …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    α : Type u_1
    s : Finset α
    x : α
    hx : Membership.mem s x
    ⊢ Iff (LE.le s.card 1) (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership. …
  -/
  refine (Nat.succ_le_of_lt (card_pos.2 ⟨x, hx⟩)).le_iff_eq.trans (card_eq_one.trans ⟨?_, ?_⟩)
    /-
      case inr.intro.refine_1
      α : Type u_1
      s : Finset α
      x : α
      hx : Membership.mem s x
      ⊢ (Exists fun a => Eq s (Singleton.singleton a)) → ∀ (a : α), Membership.mem s …
    -/
  · rintro ⟨y, rfl⟩
    /-
      case inr.intro.refine_1.intro
      α : Type u_1
      x y : α
      hx : Membership.mem (Singleton.singleton y) x
      ⊢ ∀ (a : α), Membership.mem (Singleton.singleton y) a → ∀ (b : α), Membership. …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.refine_2
      α : Type u_1
      s : Finset α
      x : α
      hx : Membership.mem s x
      ⊢ (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq a b) → E …
    -/
  · exact fun h => ⟨x, eq_singleton_iff_unique_mem.2 ⟨hx, fun y hy => h _ hy _ hx⟩⟩
    /-
      🎉 no goals
    -/


theorem card_le_one_iff : #s ≤ 1 ↔ ∀ {a b}, a ∈ s → b ∈ s → a = b := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (LE.le s.card 1) (∀ {a b : α}, Membership.mem s a → Membership.mem s b → …
  -/
  rw [card_le_one]
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq a b) …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem card_le_one_iff_subsingleton_coe : #s ≤ 1 ↔ Subsingleton (s : Type _) :=
  card_le_one.trans (s : Set α).subsingleton_coe.symm


theorem card_le_one_iff_subset_singleton [Nonempty α] : #s ≤ 1 ↔ ∃ x : α, s ⊆ {x} := by
  /-
    α : Type u_1
    s : Finset α
    inst✝ : Nonempty α
    ⊢ Iff (LE.le s.card 1) (Exists fun x => HasSubset.Subset s (Singleton.singleto …
  -/
  refine ⟨fun H => ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      s : Finset α
      inst✝ : Nonempty α
      H : LE.le s.card 1
      ⊢ Exists fun x => HasSubset.Subset s (Singleton.singleton x)
    -/
  · obtain rfl | ⟨x, hx⟩ := s.eq_empty_or_nonempty
      /-
        case refine_1.inl
        α : Type u_1
        inst✝ : Nonempty α
        H : LE.le EmptyCollection.emptyCollection.card 1
        ⊢ Exists fun x => HasSubset.Subset EmptyCollection.emptyCollection (Singleton. …
      -/
    · exact ⟨Classical.arbitrary α, empty_subset _⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.intro
        α : Type u_1
        s : Finset α
        inst✝ : Nonempty α
        H : LE.le s.card 1
        x : α
        hx : Membership.mem s x
        ⊢ Exists fun x => HasSubset.Subset s (Singleton.singleton x)
      -/
    · exact ⟨x, fun y hy => by rw [card_le_one.1 H y hy x hx, mem_singleton]⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      s : Finset α
      inst✝ : Nonempty α
      ⊢ (Exists fun x => HasSubset.Subset s (Singleton.singleton x)) → LE.le s.card 1
    -/
  · rintro ⟨x, hx⟩
    /-
      case refine_2.intro
      α : Type u_1
      s : Finset α
      inst✝ : Nonempty α
      x : α
      hx : HasSubset.Subset s (Singleton.singleton x)
      ⊢ LE.le s.card 1
    -/
    rw [← card_singleton x]
    /-
      case refine_2.intro
      α : Type u_1
      s : Finset α
      inst✝ : Nonempty α
      x : α
      hx : HasSubset.Subset s (Singleton.singleton x)
      ⊢ LE.le s.card (Singleton.singleton x).card
    -/
    exact card_le_card hx
    /-
      🎉 no goals
    -/


lemma exists_mem_ne (hs : 1 < #s) (a : α) : ∃ b ∈ s, b ≠ a := by
  /-
    α : Type u_1
    s : Finset α
    hs : LT.lt 1 s.card
    a : α
    ⊢ Exists fun b => And (Membership.mem s b) (Ne b a)
  -/
  have : Nonempty α := ⟨a⟩
  /-
    α : Type u_1
    s : Finset α
    hs : LT.lt 1 s.card
    a : α
    this : Nonempty α
    ⊢ Exists fun b => And (Membership.mem s b) (Ne b a)
  -/
  by_contra!
  /-
    α : Type u_1
    s : Finset α
    hs : LT.lt 1 s.card
    a : α
    this✝ : Nonempty α
    this : ∀ (b : α), Membership.mem s b → Eq b a
    ⊢ False
  -/
  exact hs.not_le (card_le_one_iff_subset_singleton.2 ⟨a, subset_singleton_iff'.2 this⟩)
  /-
    🎉 no goals
  -/


/-- A `Finset` of a subsingleton type has cardinality at most one. -/
theorem card_le_one_of_subsingleton [Subsingleton α] (s : Finset α) : #s ≤ 1 :=
  Finset.card_le_one_iff.2 fun {_ _ _ _} => Subsingleton.elim _ _


theorem one_lt_card : 1 < #s ↔ ∃ a ∈ s, ∃ b ∈ s, a ≠ b := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (LT.lt 1 s.card) (Exists fun a => And (Membership.mem s a) (Exists fun b …
  -/
  rw [← not_iff_not]
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (Not (LT.lt 1 s.card)) (Not (Exists fun a => And (Membership.mem s a) (E …
  -/
  push_neg
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (LE.le s.card 1) (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership. …
  -/
  exact card_le_one
  /-
    🎉 no goals
  -/


theorem one_lt_card_iff : 1 < #s ↔ ∃ a b, a ∈ s ∧ b ∈ s ∧ a ≠ b := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (LT.lt 1 s.card) (Exists fun a => Exists fun b => And (Membership.mem s  …
  -/
  rw [one_lt_card]
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (Exists fun a => And (Membership.mem s a) (Exists fun b => And (Membersh …
  -/
  simp only [exists_prop, exists_and_left]
  /-
    🎉 no goals
  -/


theorem one_lt_card_iff_nontrivial : 1 < #s ↔ s.Nontrivial := by
  rw [← not_iff_not, not_lt, Finset.Nontrivial, ← Set.nontrivial_coe_sort,
    not_nontrivial_iff_subsingleton, card_le_one_iff_subsingleton_coe, coe_sort_coe]


@[deprecated (since := "2024-02-05")]
alias one_lt_card_iff_nontrivial_coe := one_lt_card_iff_nontrivial


theorem exists_ne_of_one_lt_card (hs : 1 < #s) (a : α) : ∃ b, b ∈ s ∧ b ≠ a := by
  /-
    α : Type u_1
    s : Finset α
    hs : LT.lt 1 s.card
    a : α
    ⊢ Exists fun b => And (Membership.mem s b) (Ne b a)
  -/
  obtain ⟨x, hx, y, hy, hxy⟩ := Finset.one_lt_card.mp hs
  /-
    case intro.intro.intro.intro
    α : Type u_1
    s : Finset α
    hs : LT.lt 1 s.card
    a x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    hxy : Ne x y
    ⊢ Exists fun b => And (Membership.mem s b) (Ne b a)
  -/
  by_cases ha : y = a
    /-
      case pos
      α : Type u_1
      s : Finset α
      hs : LT.lt 1 s.card
      a x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      hxy : Ne x y
      ha : Eq y a
      ⊢ Exists fun b => And (Membership.mem s b) (Ne b a)
    -/
  · exact ⟨x, hx, ne_of_ne_of_eq hxy ha⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      s : Finset α
      hs : LT.lt 1 s.card
      a x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      hxy : Ne x y
      ha : Not (Eq y a)
      ⊢ Exists fun b => And (Membership.mem s b) (Ne b a)
    -/
  · exact ⟨y, hy, ha⟩
    /-
      🎉 no goals
    -/


/-- If a Finset in a Pi type is nontrivial (has at least two elements), then
  its projection to some factor is nontrivial, and the fibers of the projection
  are proper subsets. -/
lemma exists_of_one_lt_card_pi {ι : Type*} {α : ι → Type*} [∀ i, DecidableEq (α i)]
    {s : Finset (∀ i, α i)} (h : 1 < #s) :
    ∃ i, 1 < #(s.image (· i)) ∧ ∀ ai, s.filter (· i = ai) ⊂ s := by
  /-
    ι : Type u_4
    α : ι → Type u_5
    inst✝ : (i : ι) → DecidableEq (α i)
    s : Finset ((i : ι) → α i)
    h : LT.lt 1 s.card
    ⊢ Exists fun i => And (LT.lt 1 (Finset.image (fun x => x i) s).card) (∀ (ai :  …
  -/
  simp_rw [one_lt_card_iff, Function.ne_iff] at h ⊢
  /-
    ι : Type u_4
    α : ι → Type u_5
    inst✝ : (i : ι) → DecidableEq (α i)
    s : Finset ((i : ι) → α i)
    h : Exists fun a => Exists fun b => And (Membership.mem s a) (And (Membership. …
    ⊢ Exists fun i => And (Exists fun a => Exists fun b => And (Membership.mem (Fi …
  -/
  obtain ⟨a1, a2, h1, h2, i, hne⟩ := h
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_4
    α : ι → Type u_5
    inst✝ : (i : ι) → DecidableEq (α i)
    s : Finset ((i : ι) → α i)
    a1 a2 : (i : ι) → α i
    h1 : Membership.mem s a1
    h2 : Membership.mem s a2
    i : ι
    hne : Ne (a1 i) (a2 i)
    ⊢ Exists fun i => And (Exists fun a => Exists fun b => And (Membership.mem (Fi …
  -/
  refine ⟨i, ⟨_, _, mem_image_of_mem _ h1, mem_image_of_mem _ h2, hne⟩, fun ai => ?_⟩
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_4
    α : ι → Type u_5
    inst✝ : (i : ι) → DecidableEq (α i)
    s : Finset ((i : ι) → α i)
    a1 a2 : (i : ι) → α i
    h1 : Membership.mem s a1
    h2 : Membership.mem s a2
    i : ι
    hne : Ne (a1 i) (a2 i)
    ai : α i
    ⊢ HasSSubset.SSubset (Finset.filter (fun x => Eq (x i) ai) s) s
  -/
  rw [filter_ssubset]
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_4
    α : ι → Type u_5
    inst✝ : (i : ι) → DecidableEq (α i)
    s : Finset ((i : ι) → α i)
    a1 a2 : (i : ι) → α i
    h1 : Membership.mem s a1
    h2 : Membership.mem s a2
    i : ι
    hne : Ne (a1 i) (a2 i)
    ai : α i
    ⊢ Exists fun x => And (Membership.mem s x) (Not (Eq (x i) ai))
  -/
  obtain rfl | hne := eq_or_ne (a2 i) ai
  /-
    case intro.intro.intro.intro.intro.inl
    ι : Type u_4
    α : ι → Type u_5
    inst✝ : (i : ι) → DecidableEq (α i)
    s : Finset ((i : ι) → α i)
    a1 a2 : (i : ι) → α i
    h1 : Membership.mem s a1
    h2 : Membership.mem s a2
    i : ι
    hne : Ne (a1 i) (a2 i)
    ⊢ Exists fun x => And (Membership.mem s x) (Not (Eq (x i) (a2 i)))
  -/
  exacts [⟨a1, h1, hne⟩, ⟨a2, h2, hne⟩]
  /-
    🎉 no goals
  -/


theorem card_eq_succ_iff_cons :
    #s = n + 1 ↔ ∃ a t, ∃ (h : a ∉ t), cons a t h = s ∧ #t = n :=
                           /-
                             α : Type u_1
                             s : Finset α
                             n : Nat
                             ⊢ Eq EmptyCollection.emptyCollection.card (HAdd.hAdd n 1) → Exists fun a => Ex …
                           -/
                           /-
                             🎉 no goals
                           -/
  ⟨cons_induction_on s (by simp) fun a s _ _ _ => ⟨a, s, by simp_all⟩,
                                                            /-
                                                              🎉 no goals
                                                            -/
                              /-
                                α : Type u_1
                                s : Finset α
                                n : Nat
                                x✝ : Exists fun a => Exists fun t => Exists fun h => And (Eq (Finset.cons a t  …
                                a : α
                                t : Finset α
                                w✝ : Not (Membership.mem t a)
                                hs : Eq (Finset.cons a t w✝) s
                                right✝ : Eq t.card n
                                ⊢ Eq s.card (HAdd.hAdd n 1)
                              -/
   fun ⟨a, t, _, hs, _⟩ => by simpa [← hs]⟩
                              /-
                                🎉 no goals
                              -/


theorem card_eq_succ : #s = n + 1 ↔ ∃ a t, a ∉ t ∧ insert a t = s ∧ #t = n :=
  ⟨fun h =>
    let ⟨a, has⟩ := card_pos.mp (h.symm ▸ Nat.zero_lt_succ _ : 0 < #s)
    ⟨a, s.erase a, s.not_mem_erase a, insert_erase has, by
      /-
        α : Type u_1
        s : Finset α
        n : Nat
        inst✝ : DecidableEq α
        h : Eq s.card (HAdd.hAdd n 1)
        a : α
        has : Membership.mem s a
        ⊢ Eq (s.erase a).card n
      -/
      simp only [h, card_erase_of_mem has, Nat.add_sub_cancel_right]⟩,
      /-
        🎉 no goals
      -/
    fun ⟨_, _, hat, s_eq, n_eq⟩ => s_eq ▸ n_eq ▸ card_insert_of_not_mem hat⟩


theorem card_eq_two : #s = 2 ↔ ∃ x y, x ≠ y ∧ s = {x, y} := by
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    ⊢ Iff (Eq s.card 2) (Exists fun x => Exists fun y => And (Ne x y) (Eq s (Inser …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      ⊢ Eq s.card 2 → Exists fun x => Exists fun y => And (Ne x y) (Eq s (Insert.ins …
    -/
  · rw [card_eq_succ]
    /-
      case mp
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      ⊢ (Exists fun a => Exists fun t => And (Not (Membership.mem t a)) (And (Eq (In …
    -/
    simp_rw [card_eq_one]
    /-
      case mp
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      ⊢ (Exists fun a => Exists fun t => And (Not (Membership.mem t a)) (And (Eq (In …
    -/
    rintro ⟨a, _, hab, rfl, b, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      hab : Not (Membership.mem (Singleton.singleton b) a)
      ⊢ Exists fun x => Exists fun y => And (Ne x y) (Eq (Insert.insert a (Singleton …
    -/
    exact ⟨a, b, not_mem_singleton.1 hab, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      ⊢ (Exists fun x => Exists fun y => And (Ne x y) (Eq s (Insert.insert x (Single …
    -/
  · rintro ⟨x, y, h, rfl⟩
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      x y : α
      h : Ne x y
      ⊢ Eq (Insert.insert x (Singleton.singleton y)).card 2
    -/
    exact card_pair h
    /-
      🎉 no goals
    -/


theorem card_eq_three : #s = 3 ↔ ∃ x y z, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ s = {x, y, z} := by
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    ⊢ Iff (Eq s.card 3) (Exists fun x => Exists fun y => Exists fun z => And (Ne x …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      ⊢ Eq s.card 3 → Exists fun x => Exists fun y => Exists fun z => And (Ne x y) ( …
    -/
  · rw [card_eq_succ]
    /-
      case mp
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      ⊢ (Exists fun a => Exists fun t => And (Not (Membership.mem t a)) (And (Eq (In …
    -/
    simp_rw [card_eq_two]
    /-
      case mp
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      ⊢ (Exists fun a => Exists fun t => And (Not (Membership.mem t a)) (And (Eq (In …
    -/
    rintro ⟨a, _, abc, rfl, b, c, bc, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      a b c : α
      bc : Ne b c
      abc : Not (Membership.mem (Insert.insert b (Singleton.singleton c)) a)
      ⊢ Exists fun x => Exists fun y => Exists fun z => And (Ne x y) (And (Ne x z) ( …
    -/
    rw [mem_insert, mem_singleton, not_or] at abc
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      a b c : α
      bc : Ne b c
      abc : And (Not (Eq a b)) (Not (Eq a c))
      ⊢ Exists fun x => Exists fun y => Exists fun z => And (Ne x y) (And (Ne x z) ( …
    -/
    exact ⟨a, b, c, abc.1, abc.2, bc, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      ⊢ (Exists fun x => Exists fun y => Exists fun z => And (Ne x y) (And (Ne x z)  …
    -/
  · rintro ⟨x, y, z, xy, xz, yz, rfl⟩
    simp only [xy, xz, yz, mem_insert, card_insert_of_not_mem, not_false_iff, mem_singleton,
      or_self_iff, card_singleton]


theorem two_lt_card_iff : 2 < #s ↔ ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c := by
  classical
    simp_rw [lt_iff_add_one_le, le_card_iff_exists_subset_card, reduceAdd, card_eq_three,
      ← exists_and_left, exists_comm (α := Finset α)]
    constructor
    · rintro ⟨a, b, c, t, hsub, hab, hac, hbc, rfl⟩
      exact ⟨a, b, c, by simp_all [insert_subset_iff]⟩
    · rintro ⟨a, b, c, ha, hb, hc, hab, hac, hbc⟩
      exact ⟨a, b, c, {a, b, c}, by simp_all [insert_subset_iff]⟩


theorem two_lt_card : 2 < #s ↔ ∃ a ∈ s, ∃ b ∈ s, ∃ c ∈ s, a ≠ b ∧ a ≠ c ∧ b ≠ c := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (LT.lt 2 s.card) (Exists fun a => And (Membership.mem s a) (Exists fun b …
  -/
  simp_rw [two_lt_card_iff, exists_and_left]
  /-
    🎉 no goals
  -/


/-- Suppose that, given objects defined on all strict subsets of any finset `s`, one knows how to
define an object on `s`. Then one can inductively define an object on all finsets, starting from
the empty set and iterating. This can be used either to define data, or to prove properties. -/
def strongInduction {p : Finset α → Sort*} (H : ∀ s, (∀ t ⊂ s, p t) → p s) :
    ∀ s : Finset α, p s
  | s =>
    H s fun t h =>
      have : #t < #s := card_lt_card h
      strongInduction H t
  termination_by s => #s


@[nolint unusedHavesSuffices] -- Porting note: false positive
theorem strongInduction_eq {p : Finset α → Sort*} (H : ∀ s, (∀ t ⊂ s, p t) → p s)
    (s : Finset α) : strongInduction H s = H s fun t _ => strongInduction H t := by
  /-
    α : Type u_1
    p : Finset α → Sort u_4
    H : (s : Finset α) → ((t : Finset α) → HasSSubset.SSubset t s → p t) → p s
    s : Finset α
    ⊢ Eq (Finset.strongInduction H s) (H s fun t x => Finset.strongInduction H t)
  -/
  rw [strongInduction]
  /-
    🎉 no goals
  -/


/-- Analogue of `strongInduction` with order of arguments swapped. -/
@[elab_as_elim]
def strongInductionOn {p : Finset α → Sort*} (s : Finset α) :
    (∀ s, (∀ t ⊂ s, p t) → p s) → p s := fun H => strongInduction H s


@[nolint unusedHavesSuffices] -- Porting note: false positive
theorem strongInductionOn_eq {p : Finset α → Sort*} (s : Finset α)
    (H : ∀ s, (∀ t ⊂ s, p t) → p s) :
    s.strongInductionOn H = H s fun t _ => t.strongInductionOn H := by
  /-
    α : Type u_1
    p : Finset α → Sort u_4
    s : Finset α
    H : (s : Finset α) → ((t : Finset α) → HasSSubset.SSubset t s → p t) → p s
    ⊢ Eq (s.strongInductionOn H) (H s fun t x => t.strongInductionOn H)
  -/
  dsimp only [strongInductionOn]
  /-
    α : Type u_1
    p : Finset α → Sort u_4
    s : Finset α
    H : (s : Finset α) → ((t : Finset α) → HasSSubset.SSubset t s → p t) → p s
    ⊢ Eq (Finset.strongInduction H s) (H s fun t x => Finset.strongInduction H t)
  -/
  rw [strongInduction]
  /-
    🎉 no goals
  -/


@[elab_as_elim]
theorem case_strong_induction_on [DecidableEq α] {p : Finset α → Prop} (s : Finset α) (h₀ : p ∅)
    (h₁ : ∀ a s, a ∉ s → (∀ t ⊆ s, p t) → p (insert a s)) : p s :=
  Finset.strongInductionOn s fun s =>
    Finset.induction_on s (fun _ => h₀) fun a s n _ ih =>
      (h₁ a s n) fun t ss => ih _ (lt_of_le_of_lt ss (ssubset_insert n) : t < _)


/-- Suppose that, given objects defined on all nonempty strict subsets of any nontrivial finset `s`,
one knows how to define an object on `s`. Then one can inductively define an object on all finsets,
starting from singletons and iterating.

TODO: Currently this can only be used to prove properties.
Replace `Finset.Nonempty.exists_eq_singleton_or_nontrivial` with computational content
in order to let `p` be `Sort`-valued. -/
@[elab_as_elim]
protected lemma Nonempty.strong_induction {p : ∀ s, s.Nonempty → Prop}
    (h₀ : ∀ a, p {a} (singleton_nonempty _))
    (h₁ : ∀ ⦃s⦄ (hs : s.Nontrivial), (∀ t ht, t ⊂ s → p t ht) → p s hs.nonempty) :
    ∀ ⦃s : Finset α⦄ (hs), p s hs
  | s, hs => by
    /-
      α : Type u_1
      p : (s : Finset α) → s.Nonempty → Prop
      h₀ : ∀ (a : α), p (Singleton.singleton a) ⋯
      h₁ : ∀ ⦃s : Finset α⦄ (hs : s.Nontrivial), (∀ (t : Finset α) (ht : t.Nonempty) …
      s : Finset α
      hs : s.Nonempty
      ⊢ p s hs
    -/
    obtain ⟨a, rfl⟩ | hs := hs.exists_eq_singleton_or_nontrivial
      /-
        case inl.intro
        α : Type u_1
        p : (s : Finset α) → s.Nonempty → Prop
        h₀ : ∀ (a : α), p (Singleton.singleton a) ⋯
        h₁ : ∀ ⦃s : Finset α⦄ (hs : s.Nontrivial), (∀ (t : Finset α) (ht : t.Nonempty) …
        a : α
        hs : (Singleton.singleton a).Nonempty
        ⊢ p (Singleton.singleton a) hs
      -/
    · exact h₀ _
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        p : (s : Finset α) → s.Nonempty → Prop
        h₀ : ∀ (a : α), p (Singleton.singleton a) ⋯
        h₁ : ∀ ⦃s : Finset α⦄ (hs : s.Nontrivial), (∀ (t : Finset α) (ht : t.Nonempty) …
        s : Finset α
        hs✝ : s.Nonempty
        hs : s.Nontrivial
        ⊢ p s hs✝
      -/
    · refine h₁ hs fun t ht hts ↦ ?_
      /-
        case inr
        α : Type u_1
        p : (s : Finset α) → s.Nonempty → Prop
        h₀ : ∀ (a : α), p (Singleton.singleton a) ⋯
        h₁ : ∀ ⦃s : Finset α⦄ (hs : s.Nontrivial), (∀ (t : Finset α) (ht : t.Nonempty) …
        s : Finset α
        hs✝ : s.Nonempty
        hs : s.Nontrivial
        t : Finset α
        ht : t.Nonempty
        hts : HasSSubset.SSubset t s
        ⊢ p t ht
      -/
      have := card_lt_card hts
      /-
        case inr
        α : Type u_1
        p : (s : Finset α) → s.Nonempty → Prop
        h₀ : ∀ (a : α), p (Singleton.singleton a) ⋯
        h₁ : ∀ ⦃s : Finset α⦄ (hs : s.Nontrivial), (∀ (t : Finset α) (ht : t.Nonempty) …
        s : Finset α
        hs✝ : s.Nonempty
        hs : s.Nontrivial
        t : Finset α
        ht : t.Nonempty
        hts : HasSSubset.SSubset t s
        this : LT.lt t.card s.card
        ⊢ p t ht
      -/
      exact ht.strong_induction h₀ h₁
      /-
        🎉 no goals
      -/
termination_by s => #s


/-- Suppose that, given that `p t` can be defined on all supersets of `s` of cardinality less than
`n`, one knows how to define `p s`. Then one can inductively define `p s` for all finsets `s` of
cardinality less than `n`, starting from finsets of card `n` and iterating. This
can be used either to define data, or to prove properties. -/
def strongDownwardInduction {p : Finset α → Sort*} {n : ℕ}
    (H : ∀ t₁, (∀ {t₂ : Finset α}, #t₂ ≤ n → t₁ ⊂ t₂ → p t₂) → #t₁ ≤ n → p t₁) :
    ∀ s : Finset α, #s ≤ n → p s
  | s =>
    H s fun {t} ht h =>
      have := Finset.card_lt_card h
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     R : Type u_3
                                     s✝ t✝ u : Finset α
                                     f : α → β
                                     n✝ : Nat
                                     p : Finset α → Sort u_4
                                     n : Nat
                                     H : (t₁ : Finset α) → ({t₂ : Finset α} → LE.le t₂.card n → HasSSubset.SSubset  …
                                     x✝ : Finset α
                                     s : Finset α := x✝
                                     t : Finset α
                                     ht : LE.le t.card n
                                     h : HasSSubset.SSubset s t
                                     this : LT.lt s.card t.card
                                     ⊢ LT.lt (HSub.hSub n t.card) (HSub.hSub n s.card)
                                   -/
      have : n - #t < n - #s := by omega
                                   /-
                                     🎉 no goals
                                   -/
      strongDownwardInduction H t ht
  termination_by s => n - #s


@[nolint unusedHavesSuffices] -- Porting note: false positive
theorem strongDownwardInduction_eq {p : Finset α → Sort*}
    (H : ∀ t₁, (∀ {t₂ : Finset α}, #t₂ ≤ n → t₁ ⊂ t₂ → p t₂) → #t₁ ≤ n → p t₁)
    (s : Finset α) :
    strongDownwardInduction H s = H s fun {t} ht _ => strongDownwardInduction H t ht := by
  /-
    α : Type u_1
    n : Nat
    p : Finset α → Sort u_4
    H : (t₁ : Finset α) → ({t₂ : Finset α} → LE.le t₂.card n → HasSSubset.SSubset  …
    s : Finset α
    ⊢ Eq (Finset.strongDownwardInduction H s) (H s fun {t} ht x => Finset.strongDo …
  -/
  rw [strongDownwardInduction]
  /-
    🎉 no goals
  -/


/-- Analogue of `strongDownwardInduction` with order of arguments swapped. -/
@[elab_as_elim]
def strongDownwardInductionOn {p : Finset α → Sort*} (s : Finset α)
    (H : ∀ t₁, (∀ {t₂ : Finset α}, #t₂ ≤ n → t₁ ⊂ t₂ → p t₂) → #t₁ ≤ n → p t₁) :
    #s ≤ n → p s :=
  strongDownwardInduction H s


@[nolint unusedHavesSuffices] -- Porting note: false positive
theorem strongDownwardInductionOn_eq {p : Finset α → Sort*} (s : Finset α)
    (H : ∀ t₁, (∀ {t₂ : Finset α}, #t₂ ≤ n → t₁ ⊂ t₂ → p t₂) → #t₁ ≤ n → p t₁) :
    s.strongDownwardInductionOn H = H s fun {t} ht _ => t.strongDownwardInductionOn H ht := by
  /-
    α : Type u_1
    n : Nat
    p : Finset α → Sort u_4
    s : Finset α
    H : (t₁ : Finset α) → ({t₂ : Finset α} → LE.le t₂.card n → HasSSubset.SSubset  …
    ⊢ Eq (fun a => s.strongDownwardInductionOn H a) (H s fun {t} ht x => t.strongD …
  -/
  dsimp only [strongDownwardInductionOn]
  /-
    α : Type u_1
    n : Nat
    p : Finset α → Sort u_4
    s : Finset α
    H : (t₁ : Finset α) → ({t₂ : Finset α} → LE.le t₂.card n → HasSSubset.SSubset  …
    ⊢ Eq (fun a => Finset.strongDownwardInduction H s a) (H s fun {t} ht x => Fins …
  -/
  rw [strongDownwardInduction]
  /-
    🎉 no goals
  -/


theorem lt_wf {α} : WellFounded (@LT.lt (Finset α) _) :=
  have H : Subrelation (@LT.lt (Finset α) _) (InvImage (· < ·) card) := fun {_ _} hxy =>
    card_lt_card hxy
  Subrelation.wf H <| InvImage.wf _ <| (Nat.lt_wfRel).2


