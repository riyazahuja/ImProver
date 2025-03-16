@[simp]
theorem disjUnion_eq_union (s t h) : @disjUnion α s t h = s ∪ t :=
                  /-
                    α : Type u_1
                    inst✝ : DecidableEq α
                    s t : Finset α
                    h : Disjoint s t
                    a : α
                    ⊢ Iff (Membership.mem (s.disjUnion t h) a) (Membership.mem (Union.union s t) a)
                  -/
  ext fun a => by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem disjoint_union_left : Disjoint (s ∪ t) u ↔ Disjoint s u ∧ Disjoint t u := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Finset α
    ⊢ Iff (Disjoint (Union.union s t) u) (And (Disjoint s u) (Disjoint t u))
  -/
  simp only [disjoint_left, mem_union, or_imp, forall_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_union_right : Disjoint s (t ∪ u) ↔ Disjoint s t ∧ Disjoint s u := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Finset α
    ⊢ Iff (Disjoint s (Union.union t u)) (And (Disjoint s t) (Disjoint s u))
  -/
  simp only [disjoint_right, mem_union, or_imp, forall_and]
  /-
    🎉 no goals
  -/


theorem not_disjoint_iff_nonempty_inter : ¬Disjoint s t ↔ (s ∩ t).Nonempty :=
                               /-
                                 α : Type u_1
                                 inst✝ : DecidableEq α
                                 s t : Finset α
                                 ⊢ Iff (Exists fun a => And (Membership.mem s a) (Membership.mem t a)) (Inter.i …
                               -/
  not_disjoint_iff.trans <| by simp [Finset.Nonempty]
                               /-
                                 🎉 no goals
                               -/


alias ⟨_, Nonempty.not_disjoint⟩ := not_disjoint_iff_nonempty_inter


theorem disjoint_or_nonempty_inter (s t : Finset α) : Disjoint s t ∨ (s ∩ t).Nonempty := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Or (Disjoint s t) (Inter.inter s t).Nonempty
  -/
  rw [← not_disjoint_iff_nonempty_inter]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Or (Disjoint s t) (Not (Disjoint s t))
  -/
  exact em _
  /-
    🎉 no goals
  -/


omit [DecidableEq α] in
theorem disjoint_of_subset_iff_left_eq_empty (h : s ⊆ t) :
    Disjoint s t ↔ s = ∅ :=
  disjoint_of_le_iff_left_eq_bot h


                                                             /-
                                                               α : Type u_1
                                                               β : Type u_2
                                                               γ : Type u_3
                                                               ⊢ IsDirected (Finset α) fun x1 x2 => LE.le x1 x2
                                                             -/
instance isDirected_le : IsDirected (Finset α) (· ≤ ·) := by classical infer_instance
                                                             /-
                                                               🎉 no goals
                                                             -/

instance isDirected_subset : IsDirected (Finset α) (· ⊆ ·) := isDirected_le


@[simp]
theorem erase_empty (a : α) : erase ∅ a = ∅ :=
  rfl


protected lemma Nontrivial.erase_nonempty (hs : s.Nontrivial) : (s.erase a).Nonempty :=
                             /-
                               α : Type u_1
                               inst✝ : DecidableEq α
                               s : Finset α
                               a : α
                               hs : s.Nontrivial
                               ⊢ ∀ (a_1 : α), And (Membership.mem s a_1) (Ne a_1 a) → Membership.mem (s.erase …
                             -/
  (hs.exists_ne a).imp <| by aesop
                             /-
                               🎉 no goals
                             -/


@[simp] lemma erase_nonempty (ha : a ∈ s) : (s.erase a).Nonempty ↔ s.Nontrivial := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ha : Membership.mem s a
    ⊢ Iff (s.erase a).Nonempty s.Nontrivial
  -/
  simp only [Finset.Nonempty, mem_erase, and_comm (b := _ ∈ _)]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ha : Membership.mem s a
    ⊢ Iff (Exists fun x => And (Membership.mem s x) (Ne x a)) s.Nontrivial
  -/
  refine ⟨?_, fun hs ↦ hs.exists_ne a⟩
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ha : Membership.mem s a
    ⊢ (Exists fun x => And (Membership.mem s x) (Ne x a)) → s.Nontrivial
  -/
  rintro ⟨b, hb, hba⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hba : Ne b a
    ⊢ s.Nontrivial
  -/
  exact ⟨_, hb, _, ha, hba⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem erase_singleton (a : α) : ({a} : Finset α).erase a = ∅ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq ((Singleton.singleton a).erase a) EmptyCollection.emptyCollection
  -/
  ext x
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a x : α
    ⊢ Iff (Membership.mem ((Singleton.singleton a).erase a) x) (Membership.mem Emp …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem erase_insert_eq_erase (s : Finset α) (a : α) : (insert a s).erase a = s.erase a :=
  ext fun x => by
    simp +contextual only [mem_erase, mem_insert, and_congr_right_iff,
      false_or, iff_self, imp_true_iff]


theorem erase_insert {a : α} {s : Finset α} (h : a ∉ s) : erase (insert a s) a = s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    h : Not (Membership.mem s a)
    ⊢ Eq ((Insert.insert a s).erase a) s
  -/
  rw [erase_insert_eq_erase, erase_eq_of_not_mem h]
  /-
    🎉 no goals
  -/


theorem erase_insert_of_ne {a b : α} {s : Finset α} (h : a ≠ b) :
    erase (insert a s) b = insert a (erase s b) :=
  ext fun x => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      s : Finset α
      h : Ne a b
      x : α
      ⊢ Iff (Membership.mem ((Insert.insert a s).erase b) x) (Membership.mem (Insert …
    -/
    have : x ≠ b ∧ x = a ↔ x = a := and_iff_right_of_imp fun hx => hx.symm ▸ h
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      s : Finset α
      h : Ne a b
      x : α
      this : Iff (And (Ne x b) (Eq x a)) (Eq x a)
      ⊢ Iff (Membership.mem ((Insert.insert a s).erase b) x) (Membership.mem (Insert …
    -/
    simp only [mem_erase, mem_insert, and_or_left, this]
    /-
      🎉 no goals
    -/


theorem erase_cons_of_ne {a b : α} {s : Finset α} (ha : a ∉ s) (hb : a ≠ b) :
    erase (cons a s ha) b = cons a (erase s b) fun h => ha <| erase_subset _ _ h := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Finset α
    ha : Not (Membership.mem s a)
    hb : Ne a b
    ⊢ Eq ((Finset.cons a s ha).erase b) (Finset.cons a (s.erase b) ⋯)
  -/
  simp only [cons_eq_insert, erase_insert_of_ne hb]
  /-
    🎉 no goals
  -/


@[simp] theorem insert_erase (h : a ∈ s) : insert a (erase s a) = s :=
  ext fun x => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      h : Membership.mem s a
      x : α
      ⊢ Iff (Membership.mem (Insert.insert a (s.erase a)) x) (Membership.mem s x)
    -/
    simp only [mem_insert, mem_erase, or_and_left, dec_em, true_and]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      h : Membership.mem s a
      x : α
      ⊢ Iff (Or (Eq x a) (Membership.mem s x)) (Membership.mem s x)
    -/
    apply or_iff_right_of_imp
    /-
      case ha
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      h : Membership.mem s a
      x : α
      ⊢ Eq x a → Membership.mem s x
    -/
    rintro rfl
    /-
      case ha
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x : α
      h : Membership.mem s x
      ⊢ Membership.mem s x
    -/
    exact h
    /-
      🎉 no goals
    -/


lemma erase_eq_iff_eq_insert (hs : a ∈ s) (ht : a ∉ t) : erase s a = t ↔ s = insert a t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    hs : Membership.mem s a
    ht : Not (Membership.mem t a)
    ⊢ Iff (Eq (s.erase a) t) (Eq s (Insert.insert a t))
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma insert_erase_invOn :
    Set.InvOn (insert a) (fun s ↦ erase s a) {s : Finset α | a ∈ s} {s : Finset α | a ∉ s} :=
  ⟨fun _s ↦ insert_erase, fun _s ↦ erase_insert⟩


theorem erase_ssubset {a : α} {s : Finset α} (h : a ∈ s) : s.erase a ⊂ s :=
  calc
    s.erase a ⊂ insert a (s.erase a) := ssubset_insert <| not_mem_erase _ _
    _ = _ := insert_erase h


theorem ssubset_iff_exists_subset_erase {s t : Finset α} : s ⊂ t ↔ ∃ a ∈ t, s ⊆ t.erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Iff (HasSSubset.SSubset s t) (Exists fun a => And (Membership.mem t a) (HasS …
  -/
  refine ⟨fun h => ?_, fun ⟨a, ha, h⟩ => ssubset_of_subset_of_ssubset h <| erase_ssubset ha⟩
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSSubset.SSubset s t
    ⊢ Exists fun a => And (Membership.mem t a) (HasSubset.Subset s (t.erase a))
  -/
  obtain ⟨a, ht, hs⟩ := not_subset.1 h.2
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSSubset.SSubset s t
    a : α
    ht : Membership.mem t a
    hs : Not (Membership.mem s a)
    ⊢ Exists fun a => And (Membership.mem t a) (HasSubset.Subset s (t.erase a))
  -/
  exact ⟨a, ht, subset_erase.2 ⟨h.1, hs⟩⟩
  /-
    🎉 no goals
  -/


theorem erase_ssubset_insert (s : Finset α) (a : α) : s.erase a ⊂ insert a s :=
  ssubset_iff_exists_subset_erase.2
    ⟨a, mem_insert_self _ _, erase_subset_erase _ <| subset_insert _ _⟩


theorem erase_cons {s : Finset α} {a : α} (h : a ∉ s) : (s.cons a h).erase a = s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    h : Not (Membership.mem s a)
    ⊢ Eq ((Finset.cons a s h).erase a) s
  -/
  rw [cons_eq_insert, erase_insert_eq_erase, erase_eq_of_not_mem h]
  /-
    🎉 no goals
  -/


theorem subset_insert_iff {a : α} {s t : Finset α} : s ⊆ insert a t ↔ erase s a ⊆ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Finset α
    ⊢ Iff (HasSubset.Subset s (Insert.insert a t)) (HasSubset.Subset (s.erase a) t)
  -/
  simp only [subset_iff, or_iff_not_imp_left, mem_erase, mem_insert, and_imp]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Finset α
    ⊢ Iff (∀ ⦃x : α⦄, Membership.mem s x → Not (Eq x a) → Membership.mem t x) (∀ ⦃ …
  -/
  exact forall_congr' fun x => forall_swap
  /-
    🎉 no goals
  -/


theorem erase_insert_subset (a : α) (s : Finset α) : erase (insert a s) a ⊆ s :=
  subset_insert_iff.1 <| Subset.rfl


theorem insert_erase_subset (a : α) (s : Finset α) : s ⊆ insert a (erase s a) :=
  subset_insert_iff.2 <| Subset.rfl


theorem subset_insert_iff_of_not_mem (h : a ∉ s) : s ⊆ insert a t ↔ s ⊆ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    h : Not (Membership.mem s a)
    ⊢ Iff (HasSubset.Subset s (Insert.insert a t)) (HasSubset.Subset s t)
  -/
  rw [subset_insert_iff, erase_eq_of_not_mem h]
  /-
    🎉 no goals
  -/


theorem erase_subset_iff_of_mem (h : a ∈ t) : s.erase a ⊆ t ↔ s ⊆ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    h : Membership.mem t a
    ⊢ Iff (HasSubset.Subset (s.erase a) t) (HasSubset.Subset s t)
  -/
  rw [← subset_insert_iff, insert_eq_of_mem h]
  /-
    🎉 no goals
  -/


theorem erase_injOn' (a : α) : { s : Finset α | a ∈ s }.InjOn fun s => erase s a :=
                                          /-
                                            α : Type u_1
                                            inst✝ : DecidableEq α
                                            a : α
                                            s : Finset α
                                            hs : Membership.mem (setOf fun s => Membership.mem s a) s
                                            t : Finset α
                                            ht : Membership.mem (setOf fun s => Membership.mem s a) t
                                            h : Eq (s.erase a) ((fun s => s.erase a) t)
                                            ⊢ Eq s t
                                          -/
  fun s hs t ht (h : s.erase a = _) => by rw [← insert_erase hs, ← insert_erase ht, h]
                                          /-
                                            🎉 no goals
                                          -/


lemma Nontrivial.exists_cons_eq {s : Finset α} (hs : s.Nontrivial) :
    ∃ t a ha b hb hab, (cons b t hb).cons a (mem_cons.not.2 <| not_or_intro hab ha) = s := by
  classical
  obtain ⟨a, ha, b, hb, hab⟩ := hs
  have : b ∈ s.erase a := mem_erase.2 ⟨hab.symm, hb⟩
  refine ⟨(s.erase a).erase b, a, ?_, b, ?_, ?_, ?_⟩ <;>
    simp [insert_erase this, insert_erase ha, *]


lemma erase_sdiff_erase (hab : a ≠ b) (hb : b ∈ s) : s.erase a \ s.erase b = {b} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a b : α
    hab : Ne a b
    hb : Membership.mem s b
    ⊢ Eq (SDiff.sdiff (s.erase a) (s.erase b)) (Singleton.singleton b)
  -/
  ext; aesop
       /-
         🎉 no goals
       -/

-- TODO: Do we want to delete this lemma and `Finset.disjUnion_singleton`,
-- or instead add `Finset.union_singleton`/`Finset.singleton_union`?

theorem sdiff_singleton_eq_erase (a : α) (s : Finset α) : s \ {a} = erase s a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ⊢ Eq (SDiff.sdiff s (Singleton.singleton a)) (s.erase a)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    a✝ : α
    ⊢ Iff (Membership.mem (SDiff.sdiff s (Singleton.singleton a)) a✝) (Membership. …
  -/
  rw [mem_erase, mem_sdiff, mem_singleton, and_comm]
  /-
    🎉 no goals
  -/

-- This lemma matches `Finset.insert_eq` in functionality.

theorem erase_eq (s : Finset α) (a : α) : s.erase a = s \ {a} :=
  (sdiff_singleton_eq_erase _ _).symm


theorem disjoint_erase_comm : Disjoint (s.erase a) t ↔ Disjoint s (t.erase a) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ⊢ Iff (Disjoint (s.erase a) t) (Disjoint s (t.erase a))
  -/
  simp_rw [erase_eq, disjoint_sdiff_comm]
  /-
    🎉 no goals
  -/


lemma disjoint_insert_erase (ha : a ∉ t) : Disjoint (s.erase a) (insert a t) ↔ Disjoint s t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ha : Not (Membership.mem t a)
    ⊢ Iff (Disjoint (s.erase a) (Insert.insert a t)) (Disjoint s t)
  -/
  rw [disjoint_erase_comm, erase_insert ha]
  /-
    🎉 no goals
  -/


lemma disjoint_erase_insert (ha : a ∉ s) : Disjoint (insert a s) (t.erase a) ↔ Disjoint s t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ha : Not (Membership.mem s a)
    ⊢ Iff (Disjoint (Insert.insert a s) (t.erase a)) (Disjoint s t)
  -/
  rw [← disjoint_erase_comm, erase_insert ha]
  /-
    🎉 no goals
  -/


theorem disjoint_of_erase_left (ha : a ∉ t) (hst : Disjoint (s.erase a) t) : Disjoint s t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ha : Not (Membership.mem t a)
    hst : Disjoint (s.erase a) t
    ⊢ Disjoint s t
  -/
  rw [← erase_insert ha, ← disjoint_erase_comm, disjoint_insert_right]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ha : Not (Membership.mem t a)
    hst : Disjoint (s.erase a) t
    ⊢ And (Not (Membership.mem (s.erase a) a)) (Disjoint (s.erase a) t)
  -/
  exact ⟨not_mem_erase _ _, hst⟩
  /-
    🎉 no goals
  -/


theorem disjoint_of_erase_right (ha : a ∉ s) (hst : Disjoint s (t.erase a)) : Disjoint s t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ha : Not (Membership.mem s a)
    hst : Disjoint s (t.erase a)
    ⊢ Disjoint s t
  -/
  rw [← erase_insert ha, disjoint_erase_comm, disjoint_insert_left]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ha : Not (Membership.mem s a)
    hst : Disjoint s (t.erase a)
    ⊢ And (Not (Membership.mem (t.erase a) a)) (Disjoint s (t.erase a))
  -/
  exact ⟨not_mem_erase _ _, hst⟩
  /-
    🎉 no goals
  -/


theorem inter_erase (a : α) (s t : Finset α) : s ∩ t.erase a = (s ∩ t).erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Finset α
    ⊢ Eq (Inter.inter s (t.erase a)) ((Inter.inter s t).erase a)
  -/
  simp only [erase_eq, inter_sdiff_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem erase_inter (a : α) (s t : Finset α) : s.erase a ∩ t = (s ∩ t).erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Finset α
    ⊢ Eq (Inter.inter (s.erase a) t) ((Inter.inter s t).erase a)
  -/
  simpa only [inter_comm t] using inter_erase a t s
  /-
    🎉 no goals
  -/


theorem erase_sdiff_comm (s t : Finset α) (a : α) : s.erase a \ t = (s \ t).erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ⊢ Eq (SDiff.sdiff (s.erase a) t) ((SDiff.sdiff s t).erase a)
  -/
  simp_rw [erase_eq, sdiff_right_comm]
  /-
    🎉 no goals
  -/


theorem erase_inter_comm (s t : Finset α) (a : α) : s.erase a ∩ t = s ∩ t.erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ⊢ Eq (Inter.inter (s.erase a) t) (Inter.inter s (t.erase a))
  -/
  rw [erase_inter, inter_erase]
  /-
    🎉 no goals
  -/


theorem erase_union_distrib (s t : Finset α) (a : α) : (s ∪ t).erase a = s.erase a ∪ t.erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ⊢ Eq ((Union.union s t).erase a) (Union.union (s.erase a) (t.erase a))
  -/
  simp_rw [erase_eq, union_sdiff_distrib]
  /-
    🎉 no goals
  -/


theorem insert_inter_distrib (s t : Finset α) (a : α) :
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : DecidableEq α
                                                       s t : Finset α
                                                       a : α
                                                       ⊢ Eq (Insert.insert a (Inter.inter s t)) (Inter.inter (Insert.insert a s) (Ins …
                                                     -/
    insert a (s ∩ t) = insert a s ∩ insert a t := by simp_rw [insert_eq, union_inter_distrib_left]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem erase_sdiff_distrib (s t : Finset α) (a : α) : (s \ t).erase a = s.erase a \ t.erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ⊢ Eq ((SDiff.sdiff s t).erase a) (SDiff.sdiff (s.erase a) (t.erase a))
  -/
  simp_rw [erase_eq, sdiff_sdiff, sup_sdiff_eq_sup le_rfl, sup_comm]
  /-
    🎉 no goals
  -/


theorem erase_union_of_mem (ha : a ∈ t) (s : Finset α) : s.erase a ∪ t = s ∪ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    t : Finset α
    a : α
    ha : Membership.mem t a
    s : Finset α
    ⊢ Eq (Union.union (s.erase a) t) (Union.union s t)
  -/
  rw [← insert_erase (mem_union_right s ha), erase_union_distrib, ← union_insert, insert_erase ha]
  /-
    🎉 no goals
  -/


theorem union_erase_of_mem (ha : a ∈ s) (t : Finset α) : s ∪ t.erase a = s ∪ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ha : Membership.mem s a
    t : Finset α
    ⊢ Eq (Union.union s (t.erase a)) (Union.union s t)
  -/
  rw [← insert_erase (mem_union_left t ha), erase_union_distrib, ← insert_union, insert_erase ha]
  /-
    🎉 no goals
  -/


@[simp, deprecated erase_eq_of_not_mem (since := "2024-10-01")]
theorem sdiff_singleton_eq_self (ha : a ∉ s) : s \ {a} = s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ha : Not (Membership.mem s a)
    ⊢ Eq (SDiff.sdiff s (Singleton.singleton a)) s
  -/
  rw [← erase_eq, erase_eq_of_not_mem ha]
  /-
    🎉 no goals
  -/


theorem sdiff_union_erase_cancel (hts : t ⊆ s) (ha : a ∈ t) : s \ t ∪ t.erase a = s.erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    hts : HasSubset.Subset t s
    ha : Membership.mem t a
    ⊢ Eq (Union.union (SDiff.sdiff s t) (t.erase a)) (s.erase a)
  -/
  simp_rw [erase_eq, sdiff_union_sdiff_cancel hts (singleton_subset_iff.2 ha)]
  /-
    🎉 no goals
  -/


theorem sdiff_insert (s t : Finset α) (x : α) : s \ insert x t = (s \ t).erase x := by
  simp_rw [← sdiff_singleton_eq_erase, insert_eq, sdiff_sdiff_left', sdiff_union_distrib,
    inter_comm]


theorem sdiff_insert_insert_of_mem_of_not_mem {s t : Finset α} {x : α} (hxs : x ∈ s) (hxt : x ∉ t) :
    insert x (s \ insert x t) = s \ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    x : α
    hxs : Membership.mem s x
    hxt : Not (Membership.mem t x)
    ⊢ Eq (Insert.insert x (SDiff.sdiff s (Insert.insert x t))) (SDiff.sdiff s t)
  -/
  rw [sdiff_insert, insert_erase (mem_sdiff.mpr ⟨hxs, hxt⟩)]
  /-
    🎉 no goals
  -/


theorem sdiff_erase (h : a ∈ s) : s \ t.erase a = insert a (s \ t) := by
  rw [← sdiff_singleton_eq_erase, sdiff_sdiff_eq_sdiff_union (singleton_subset_iff.2 h), insert_eq,
    union_comm]


theorem sdiff_erase_self (ha : a ∈ s) : s \ s.erase a = {a} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ha : Membership.mem s a
    ⊢ Eq (SDiff.sdiff s (s.erase a)) (Singleton.singleton a)
  -/
  rw [sdiff_erase ha, Finset.sdiff_self, insert_emptyc_eq]
  /-
    🎉 no goals
  -/


theorem erase_eq_empty_iff (s : Finset α) (a : α) : s.erase a = ∅ ↔ s = ∅ ∨ s = {a} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ⊢ Iff (Eq (s.erase a) EmptyCollection.emptyCollection) (Or (Eq s EmptyCollecti …
  -/
  rw [← sdiff_singleton_eq_erase, sdiff_eq_empty_iff_subset, subset_singleton_iff]
  /-
    🎉 no goals
  -/

--TODO@Yaël: Kill lemmas duplicate with `BooleanAlgebra`

theorem sdiff_disjoint : Disjoint (t \ s) s :=
  disjoint_left.2 fun _a ha => (mem_sdiff.1 ha).2


theorem disjoint_sdiff : Disjoint s (t \ s) :=
  sdiff_disjoint.symm


theorem disjoint_sdiff_inter (s t : Finset α) : Disjoint (s \ t) (s ∩ t) :=
  disjoint_of_subset_right inter_subset_right sdiff_disjoint


@[simp]
theorem attach_empty : attach (∅ : Finset α) = ∅ :=
  rfl


@[simp]
theorem attach_nonempty_iff {s : Finset α} : s.attach.Nonempty ↔ s.Nonempty := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff s.attach.Nonempty s.Nonempty
  -/
  simp [Finset.Nonempty]
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
protected alias ⟨_, Nonempty.attach⟩ := attach_nonempty_iff


@[simp]
theorem attach_eq_empty_iff {s : Finset α} : s.attach = ∅ ↔ s = ∅ := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (Eq s.attach EmptyCollection.emptyCollection) (Eq s EmptyCollection.empt …
  -/
  simp [eq_empty_iff_forall_not_mem]
  /-
    🎉 no goals
  -/


theorem filter_singleton (a : α) : filter p {a} = if p a then {a} else ∅ := by
  classical
    ext x
    simp only [mem_singleton, forall_eq, mem_filter]
    split_ifs with h <;> by_cases h' : x = a <;> simp [h, h']


theorem filter_cons_of_pos (a : α) (s : Finset α) (ha : a ∉ s) (hp : p a) :
    filter p (cons a s ha) = cons a (filter p s) ((mem_of_mem_filter _).mt ha) :=
  eq_of_veq <| Multiset.filter_cons_of_pos s.val hp


theorem filter_cons_of_neg (a : α) (s : Finset α) (ha : a ∉ s) (hp : ¬p a) :
    filter p (cons a s ha) = filter p s :=
  eq_of_veq <| Multiset.filter_cons_of_neg s.val hp


theorem disjoint_filter {s : Finset α} {p q : α → Prop} [DecidablePred p] [DecidablePred q] :
    Disjoint (s.filter p) (s.filter q) ↔ ∀ x ∈ s, p x → ¬q x := by
  /-
    α : Type u_1
    s : Finset α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    ⊢ Iff (Disjoint (Finset.filter p s) (Finset.filter q s)) (∀ (x : α), Membershi …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> simp +contextual [disjoint_left]
                  /-
                    🎉 no goals
                  -/


theorem disjoint_filter_filter' (s t : Finset α)
    {p q : α → Prop} [DecidablePred p] [DecidablePred q] (h : Disjoint p q) :
    Disjoint (s.filter p) (t.filter q) := by
  /-
    α : Type u_1
    s t : Finset α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    h : Disjoint p q
    ⊢ Disjoint (Finset.filter p s) (Finset.filter q t)
  -/
  simp_rw [disjoint_left, mem_filter]
  /-
    α : Type u_1
    s t : Finset α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    h : Disjoint p q
    ⊢ ∀ ⦃a : α⦄, And (Membership.mem s a) (p a) → Not (And (Membership.mem t a) (q …
  -/
  rintro a ⟨_, hp⟩ ⟨_, hq⟩
  /-
    case intro.intro
    α : Type u_1
    s t : Finset α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    h : Disjoint p q
    a : α
    left✝¹ : Membership.mem s a
    hp : p a
    left✝ : Membership.mem t a
    hq : q a
    ⊢ False
  -/
  rw [Pi.disjoint_iff] at h
  /-
    case intro.intro
    α : Type u_1
    s t : Finset α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    h : ∀ (i : α), Disjoint (p i) (q i)
    a : α
    left✝¹ : Membership.mem s a
    hp : p a
    left✝ : Membership.mem t a
    hq : q a
    ⊢ False
  -/
  simpa [hp, hq] using h a
  /-
    🎉 no goals
  -/


theorem disjoint_filter_filter_neg (s t : Finset α) (p : α → Prop)
    [DecidablePred p] [∀ x, Decidable (¬p x)] :
    Disjoint (s.filter p) (t.filter fun a => ¬p a) :=
  disjoint_filter_filter' s t disjoint_compl_right


@[deprecated (since := "2024-10-01")] alias filter_inter_filter_neg_eq := disjoint_filter_filter_neg


theorem filter_disj_union (s : Finset α) (t : Finset α) (h : Disjoint s t) :
    filter p (disjUnion s t h) = (filter p s).disjUnion (filter p t) (disjoint_filter_filter h) :=
  eq_of_veq <| Multiset.filter_add _ _ _


theorem filter_cons {a : α} (s : Finset α) (ha : a ∉ s) :
    filter p (cons a s ha) =
      if p a then cons a (filter p s) ((mem_of_mem_filter _).mt ha) else filter p s := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    ⊢ Eq (Finset.filter p (Finset.cons a s ha)) (ite (p a) (Finset.cons a (Finset. …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      h : p a
      ⊢ Eq (Finset.filter p (Finset.cons a s ha)) (Finset.cons a (Finset.filter p s) …
    -/
  · rw [filter_cons_of_pos _ _ _ ha h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      h : Not (p a)
      ⊢ Eq (Finset.filter p (Finset.cons a s ha)) (Finset.filter p s)
    -/
  · rw [filter_cons_of_neg _ _ _ ha h]
    /-
      🎉 no goals
    -/


theorem filter_union (s₁ s₂ : Finset α) : (s₁ ∪ s₂).filter p = s₁.filter p ∪ s₂.filter p :=
                  /-
                    α : Type u_1
                    p : α → Prop
                    inst✝¹ : DecidablePred p
                    inst✝ : DecidableEq α
                    s₁ s₂ : Finset α
                    x✝ : α
                    ⊢ Iff (Membership.mem (Finset.filter p (Union.union s₁ s₂)) x✝) (Membership.me …
                  -/
  ext fun _ => by simp only [mem_filter, mem_union, or_and_right]
                  /-
                    🎉 no goals
                  -/


theorem filter_union_right (s : Finset α) : s.filter p ∪ s.filter q = s.filter fun x => p x ∨ q x :=
                  /-
                    α : Type u_1
                    p q : α → Prop
                    inst✝² : DecidablePred p
                    inst✝¹ : DecidablePred q
                    inst✝ : DecidableEq α
                    s : Finset α
                    x : α
                    ⊢ Iff (Membership.mem (Union.union (Finset.filter p s) (Finset.filter q s)) x) …
                  -/
  ext fun x => by simp [mem_filter, mem_union, ← and_or_left]
                  /-
                    🎉 no goals
                  -/


theorem filter_mem_eq_inter {s t : Finset α} [∀ i, Decidable (i ∈ t)] :
    (s.filter fun i => i ∈ t) = s ∩ t :=
                  /-
                    α : Type u_1
                    inst✝¹ : DecidableEq α
                    s t : Finset α
                    inst✝ : (i : α) → Decidable (Membership.mem t i)
                    i : α
                    ⊢ Iff (Membership.mem (Finset.filter (fun i => Membership.mem t i) s) i) (Memb …
                  -/
  ext fun i => by simp [mem_filter, mem_inter]
                  /-
                    🎉 no goals
                  -/


theorem filter_inter_distrib (s t : Finset α) : (s ∩ t).filter p = s.filter p ∩ t.filter p := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (Finset.filter p (Inter.inter s t)) (Inter.inter (Finset.filter p s) (Fin …
  -/
  ext
  /-
    case h
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    s t : Finset α
    a✝ : α
    ⊢ Iff (Membership.mem (Finset.filter p (Inter.inter s t)) a✝) (Membership.mem  …
  -/
  simp [mem_filter, mem_inter, and_assoc]
  /-
    🎉 no goals
  -/


theorem filter_inter (s t : Finset α) : filter p s ∩ t = filter p (s ∩ t) := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (Inter.inter (Finset.filter p s) t) (Finset.filter p (Inter.inter s t))
  -/
  ext
  /-
    case h
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    s t : Finset α
    a✝ : α
    ⊢ Iff (Membership.mem (Inter.inter (Finset.filter p s) t) a✝) (Membership.mem  …
  -/
  simp only [mem_inter, mem_filter, and_right_comm]
  /-
    🎉 no goals
  -/


theorem inter_filter (s t : Finset α) : s ∩ filter p t = filter p (s ∩ t) := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (Inter.inter s (Finset.filter p t)) (Finset.filter p (Inter.inter s t))
  -/
  rw [inter_comm, filter_inter, inter_comm]
  /-
    🎉 no goals
  -/


theorem filter_insert (a : α) (s : Finset α) :
    filter p (insert a s) = if p a then insert a (filter p s) else filter p s := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ⊢ Eq (Finset.filter p (Insert.insert a s)) (ite (p a) (Insert.insert a (Finset …
  -/
  ext x
  /-
    case h
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    x : α
    ⊢ Iff (Membership.mem (Finset.filter p (Insert.insert a s)) x) (Membership.mem …
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
  split_ifs with h <;> by_cases h' : x = a <;> simp [h, h']
                                               /-
                                                 🎉 no goals
                                               -/


theorem filter_erase (a : α) (s : Finset α) : filter p (erase s a) = erase (filter p s) a := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ⊢ Eq (Finset.filter p (s.erase a)) ((Finset.filter p s).erase a)
  -/
  ext x
  /-
    case h
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    x : α
    ⊢ Iff (Membership.mem (Finset.filter p (s.erase a)) x) (Membership.mem ((Finse …
  -/
  simp only [and_assoc, mem_filter, iff_self, mem_erase]
  /-
    🎉 no goals
  -/


theorem filter_or (s : Finset α) : (s.filter fun a => p a ∨ q a) = s.filter p ∪ s.filter q :=
                  /-
                    α : Type u_1
                    p q : α → Prop
                    inst✝² : DecidablePred p
                    inst✝¹ : DecidablePred q
                    inst✝ : DecidableEq α
                    s : Finset α
                    x✝ : α
                    ⊢ Iff (Membership.mem (Finset.filter (fun a => Or (p a) (q a)) s) x✝) (Members …
                  -/
  ext fun _ => by simp [mem_filter, mem_union, and_or_left]
                  /-
                    🎉 no goals
                  -/


theorem filter_and (s : Finset α) : (s.filter fun a => p a ∧ q a) = s.filter p ∩ s.filter q :=
                  /-
                    α : Type u_1
                    p q : α → Prop
                    inst✝² : DecidablePred p
                    inst✝¹ : DecidablePred q
                    inst✝ : DecidableEq α
                    s : Finset α
                    x✝ : α
                    ⊢ Iff (Membership.mem (Finset.filter (fun a => And (p a) (q a)) s) x✝) (Member …
                  -/
  ext fun _ => by simp [mem_filter, mem_inter, and_comm, and_left_comm, and_self_iff, and_assoc]
                  /-
                    🎉 no goals
                  -/


theorem filter_not (s : Finset α) : (s.filter fun a => ¬p a) = s \ s.filter p :=
  ext fun a => by
    simp only [Bool.decide_coe, Bool.not_eq_true', mem_filter, and_comm, mem_sdiff, not_and_or,
      Bool.not_eq_true, and_or_left, and_not_self, or_false]


lemma filter_and_not (s : Finset α) (p q : α → Prop) [DecidablePred p] [DecidablePred q] :
    s.filter (fun a ↦ p a ∧ ¬ q a) = s.filter p \ s.filter q := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    s : Finset α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    ⊢ Eq (Finset.filter (fun a => And (p a) (Not (q a))) s) (SDiff.sdiff (Finset.f …
  -/
  rw [filter_and, filter_not, ← inter_sdiff_assoc, inter_eq_left.2 (filter_subset _ _)]
  /-
    🎉 no goals
  -/


theorem sdiff_eq_filter (s₁ s₂ : Finset α) : s₁ \ s₂ = filter (· ∉ s₂) s₁ :=
                  /-
                    α : Type u_1
                    inst✝ : DecidableEq α
                    s₁ s₂ : Finset α
                    x✝ : α
                    ⊢ Iff (Membership.mem (SDiff.sdiff s₁ s₂) x✝) (Membership.mem (Finset.filter ( …
                  -/
  ext fun _ => by simp [mem_sdiff, mem_filter]
                  /-
                    🎉 no goals
                  -/


theorem subset_union_elim {s : Finset α} {t₁ t₂ : Set α} (h : ↑s ⊆ t₁ ∪ t₂) :
    ∃ s₁ s₂ : Finset α, s₁ ∪ s₂ = s ∧ ↑s₁ ⊆ t₁ ∧ ↑s₂ ⊆ t₂ \ t₁ := by
  classical
    refine ⟨s.filter (· ∈ t₁), s.filter (· ∉ t₁), ?_, ?_, ?_⟩
    · simp [filter_union_right, em]
    · intro x
      simp
    · intro x
      simp only [not_not, coe_filter, Set.mem_setOf_eq, Set.mem_diff, and_imp]
      intro hx hx₂
      exact ⟨Or.resolve_left (h hx) hx₂, hx₂⟩


/-- After filtering out everything that does not equal a given value, at most that value remains.

  This is equivalent to `filter_eq'` with the equality the other way.
-/
theorem filter_eq [DecidableEq β] (s : Finset β) (b : β) :
    s.filter (Eq b) = ite (b ∈ s) {b} ∅ := by
  /-
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset β
    b : β
    ⊢ Eq (Finset.filter (Eq b) s) (ite (Membership.mem s b) (Singleton.singleton b …
  -/
  split_ifs with h
    /-
      case pos
      β : Type u_2
      inst✝ : DecidableEq β
      s : Finset β
      b : β
      h : Membership.mem s b
      ⊢ Eq (Finset.filter (Eq b) s) (Singleton.singleton b)
    -/
  · ext
    /-
      case pos.h
      β : Type u_2
      inst✝ : DecidableEq β
      s : Finset β
      b : β
      h : Membership.mem s b
      a✝ : β
      ⊢ Iff (Membership.mem (Finset.filter (Eq b) s) a✝) (Membership.mem (Singleton. …
    -/
    simp only [mem_filter, mem_singleton, decide_eq_true_eq]
    /-
      case pos.h
      β : Type u_2
      inst✝ : DecidableEq β
      s : Finset β
      b : β
      h : Membership.mem s b
      a✝ : β
      ⊢ Iff (And (Membership.mem s a✝) (Eq b a✝)) (Eq a✝ b)
    -/
    refine ⟨fun h => h.2.symm, ?_⟩
    /-
      case pos.h
      β : Type u_2
      inst✝ : DecidableEq β
      s : Finset β
      b : β
      h : Membership.mem s b
      a✝ : β
      ⊢ Eq a✝ b → And (Membership.mem s a✝) (Eq b a✝)
    -/
    rintro rfl
    /-
      case pos.h
      β : Type u_2
      inst✝ : DecidableEq β
      s : Finset β
      a✝ : β
      h : Membership.mem s a✝
      ⊢ And (Membership.mem s a✝) (Eq a✝ a✝)
    -/
    exact ⟨h, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      β : Type u_2
      inst✝ : DecidableEq β
      s : Finset β
      b : β
      h : Not (Membership.mem s b)
      ⊢ Eq (Finset.filter (Eq b) s) EmptyCollection.emptyCollection
    -/
  · ext
    /-
      case neg.h
      β : Type u_2
      inst✝ : DecidableEq β
      s : Finset β
      b : β
      h : Not (Membership.mem s b)
      a✝ : β
      ⊢ Iff (Membership.mem (Finset.filter (Eq b) s) a✝) (Membership.mem EmptyCollec …
    -/
    simp only [mem_filter, not_and, iff_false, not_mem_empty, decide_eq_true_eq]
    /-
      case neg.h
      β : Type u_2
      inst✝ : DecidableEq β
      s : Finset β
      b : β
      h : Not (Membership.mem s b)
      a✝ : β
      ⊢ Membership.mem s a✝ → Not (Eq b a✝)
    -/
    rintro m rfl
    /-
      case neg.h
      β : Type u_2
      inst✝ : DecidableEq β
      s : Finset β
      b : β
      h : Not (Membership.mem s b)
      m : Membership.mem s b
      ⊢ False
    -/
    exact h m
    /-
      🎉 no goals
    -/


/-- After filtering out everything that does not equal a given value, at most that value remains.

  This is equivalent to `filter_eq` with the equality the other way.
-/
theorem filter_eq' [DecidableEq β] (s : Finset β) (b : β) :
    (s.filter fun a => a = b) = ite (b ∈ s) {b} ∅ :=
                                           /-
                                             β : Type u_2
                                             inst✝ : DecidableEq β
                                             s : Finset β
                                             b x✝¹ : β
                                             x✝ : Membership.mem s x✝¹
                                             ⊢ Iff (Eq x✝¹ b) (Eq b x✝¹)
                                           -/
  _root_.trans (filter_congr fun _ _ => by simp_rw [@eq_comm _ b]) (filter_eq s b)
                                           /-
                                             🎉 no goals
                                           -/


theorem filter_ne [DecidableEq β] (s : Finset β) (b : β) :
    (s.filter fun a => b ≠ a) = s.erase b := by
  /-
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset β
    b : β
    ⊢ Eq (Finset.filter (fun a => Ne b a) s) (s.erase b)
  -/
  ext
  /-
    case h
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset β
    b a✝ : β
    ⊢ Iff (Membership.mem (Finset.filter (fun a => Ne b a) s) a✝) (Membership.mem  …
  -/
  simp only [mem_filter, mem_erase, Ne, decide_not, Bool.not_eq_true', decide_eq_false_iff_not]
  /-
    case h
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset β
    b a✝ : β
    ⊢ Iff (And (Membership.mem s a✝) (Not (Eq b a✝))) (And (Not (Eq a✝ b)) (Member …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem filter_ne' [DecidableEq β] (s : Finset β) (b : β) : (s.filter fun a => a ≠ b) = s.erase b :=
                                           /-
                                             β : Type u_2
                                             inst✝ : DecidableEq β
                                             s : Finset β
                                             b x✝¹ : β
                                             x✝ : Membership.mem s x✝¹
                                             ⊢ Iff (Ne x✝¹ b) (Ne b x✝¹)
                                           -/
  _root_.trans (filter_congr fun _ _ => by simp_rw [@ne_comm _ b]) (filter_ne s b)
                                           /-
                                             🎉 no goals
                                           -/


theorem filter_union_filter_of_codisjoint (s : Finset α) (h : Codisjoint p q) :
    s.filter p ∪ s.filter q = s :=
  (filter_or _ _ _).symm.trans <| filter_true_of_mem fun x _ => h.top_le x trivial


theorem filter_union_filter_neg_eq [∀ x, Decidable (¬p x)] (s : Finset α) :
    (s.filter p ∪ s.filter fun a => ¬p a) = s :=
  filter_union_filter_of_codisjoint _ _ _ <| @codisjoint_hnot_right _ _ p


@[simp]
theorem range_filter_eq {n m : ℕ} : (range n).filter (· = m) = if m < n then {m} else ∅ := by
  /-
    n m : Nat
    ⊢ Eq (Finset.filter (fun x => Eq x m) (Finset.range n)) (ite (LT.lt m n) (Sing …
  -/
  convert filter_eq (range n) m using 2
    /-
      case h.e'_2.h.e'_2
      n m : Nat
      ⊢ Eq (fun x => Eq x m) (Eq m)
    -/
  · ext
    /-
      case h.e'_2.h.e'_2.h.a
      n m x✝ : Nat
      ⊢ Iff (Eq x✝ m) (Eq m x✝)
    -/
    rw [eq_comm]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h₁.a
      n m : Nat
      ⊢ Iff (LT.lt m n) (Membership.mem (Finset.range n) m)
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp]
theorem toFinset_add (s t : Multiset α) : toFinset (s + t) = toFinset s ∪ toFinset t :=
                   /-
                     α : Type u_1
                     inst✝ : DecidableEq α
                     s t : Multiset α
                     ⊢ ∀ (a : α), Iff (Membership.mem (HAdd.hAdd s t).toFinset a) (Membership.mem ( …
                   -/
  Finset.ext <| by simp
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem toFinset_nsmul (s : Multiset α) : ∀ n ≠ 0, (n • s).toFinset = s.toFinset
               /-
                 α : Type u_1
                 inst✝ : DecidableEq α
                 s : Multiset α
                 h : Ne 0 0
                 ⊢ Eq (HSMul.hSMul 0 s).toFinset s.toFinset
               -/
  | 0, h => by contradiction
               /-
                 🎉 no goals
               -/
  | n + 1, _ => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      n : Nat
      x✝ : Ne (HAdd.hAdd n 1) 0
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd n 1) s).toFinset s.toFinset
    -/
    by_cases h : n = 0
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        s : Multiset α
        n : Nat
        x✝ : Ne (HAdd.hAdd n 1) 0
        h : Eq n 0
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd n 1) s).toFinset s.toFinset
      -/
    · rw [h, zero_add, one_nsmul]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        s : Multiset α
        n : Nat
        x✝ : Ne (HAdd.hAdd n 1) 0
        h : Not (Eq n 0)
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd n 1) s).toFinset s.toFinset
      -/
    · rw [add_nsmul, toFinset_add, one_nsmul, toFinset_nsmul s n h, Finset.union_idempotent]
      /-
        🎉 no goals
      -/


theorem toFinset_eq_singleton_iff (s : Multiset α) (a : α) :
    s.toFinset = {a} ↔ card s ≠ 0 ∧ s = card s • {a} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    a : α
    ⊢ Iff (Eq s.toFinset (Singleton.singleton a)) (And (Ne s.card 0) (Eq s (HSMul. …
  -/
  refine ⟨fun H ↦ ⟨fun h ↦ ?_, ext' fun x ↦ ?_⟩, fun H ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      a : α
      H : Eq s.toFinset (Singleton.singleton a)
      h : Eq s.card 0
      ⊢ False
    -/
  · rw [card_eq_zero.1 h, toFinset_zero] at H
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      a : α
      H : Eq EmptyCollection.emptyCollection (Singleton.singleton a)
      h : Eq s.card 0
      ⊢ False
    -/
    exact Finset.singleton_ne_empty _ H.symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      a : α
      H : Eq s.toFinset (Singleton.singleton a)
      x : α
      ⊢ Eq (Multiset.count x s) (Multiset.count x (HSMul.hSMul s.card (Singleton.sin …
    -/
  · rw [count_nsmul, count_singleton]
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      a : α
      H : Eq s.toFinset (Singleton.singleton a)
      x : α
      ⊢ Eq (Multiset.count x s) (HMul.hMul s.card (ite (Eq x a) 1 0))
    -/
    by_cases hx : x = a
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        s : Multiset α
        a : α
        H : Eq s.toFinset (Singleton.singleton a)
        x : α
        hx : Eq x a
        ⊢ Eq (Multiset.count x s) (HMul.hMul s.card (ite (Eq x a) 1 0))
      -/
    · simp_rw [hx, ite_true, mul_one, count_eq_card]
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        s : Multiset α
        a : α
        H : Eq s.toFinset (Singleton.singleton a)
        x : α
        hx : Eq x a
        ⊢ ∀ (x : α), Membership.mem s x → Eq a x
      -/
      intro y hy
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        s : Multiset α
        a : α
        H : Eq s.toFinset (Singleton.singleton a)
        x : α
        hx : Eq x a
        y : α
        hy : Membership.mem s y
        ⊢ Eq a y
      -/
      rw [← mem_toFinset, H, Finset.mem_singleton] at hy
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        s : Multiset α
        a : α
        H : Eq s.toFinset (Singleton.singleton a)
        x : α
        hx : Eq x a
        y : α
        hy : Eq y a
        ⊢ Eq a y
      -/
      exact hy.symm
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      a : α
      H : Eq s.toFinset (Singleton.singleton a)
      x : α
      hx : Not (Eq x a)
      ⊢ Eq (Multiset.count x s) (HMul.hMul s.card (ite (Eq x a) 1 0))
    -/
    have hx' : x ∉ s := fun h' ↦ hx <| by rwa [← mem_toFinset, H, Finset.mem_singleton] at h'
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      a : α
      H : Eq s.toFinset (Singleton.singleton a)
      x : α
      hx : Not (Eq x a)
      hx' : Not (Membership.mem s x)
      ⊢ Eq (Multiset.count x s) (HMul.hMul s.card (ite (Eq x a) 1 0))
    -/
    simp_rw [count_eq_zero_of_not_mem hx', hx, ite_false, Nat.mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    a : α
    H : And (Ne s.card 0) (Eq s (HSMul.hSMul s.card (Singleton.singleton a)))
    ⊢ Eq s.toFinset (Singleton.singleton a)
  -/
  simpa only [toFinset_nsmul _ _ H.1, toFinset_singleton] using congr($(H.2).toFinset)
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_inter (s t : Multiset α) : toFinset (s ∩ t) = toFinset s ∩ toFinset t :=
                   /-
                     α : Type u_1
                     inst✝ : DecidableEq α
                     s t : Multiset α
                     ⊢ ∀ (a : α), Iff (Membership.mem (Inter.inter s t).toFinset a) (Membership.mem …
                   -/
  Finset.ext <| by simp
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem toFinset_union (s t : Multiset α) : (s ∪ t).toFinset = s.toFinset ∪ t.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (Union.union s t).toFinset (Union.union s.toFinset t.toFinset)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
theorem toFinset_eq_empty {m : Multiset α} : m.toFinset = ∅ ↔ m = 0 :=
  Finset.val_inj.symm.trans Multiset.dedup_eq_zero


@[simp]
theorem toFinset_nonempty : s.toFinset.Nonempty ↔ s ≠ 0 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Iff s.toFinset.Nonempty (Ne s 0)
  -/
  simp only [toFinset_eq_empty, Ne, Finset.nonempty_iff_ne_empty]
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
protected alias ⟨_, Aesop.toFinset_nonempty_of_ne⟩ := toFinset_nonempty


@[simp]
theorem toFinset_filter (s : Multiset α) (p : α → Prop) [DecidablePred p] :
    Multiset.toFinset (s.filter p) = s.toFinset.filter p := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    s : Multiset α
    p : α → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Multiset.filter p s).toFinset (Finset.filter p s.toFinset)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
theorem toFinset_union (l l' : List α) : (l ∪ l').toFinset = l.toFinset ∪ l'.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    ⊢ Eq (Union.union l l').toFinset (Union.union l.toFinset l'.toFinset)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    a✝ : α
    ⊢ Iff (Membership.mem (Union.union l l').toFinset a✝) (Membership.mem (Union.u …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_inter (l l' : List α) : (l ∩ l').toFinset = l.toFinset ∩ l'.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    ⊢ Eq (Inter.inter l l').toFinset (Inter.inter l.toFinset l'.toFinset)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    a✝ : α
    ⊢ Iff (Membership.mem (Inter.inter l l').toFinset a✝) (Membership.mem (Inter.i …
  -/
  simp
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
alias ⟨_, Aesop.toFinset_nonempty_of_ne⟩ := toFinset_nonempty_iff


@[simp]
theorem toFinset_filter (s : List α) (p : α → Bool) :
    (s.filter p).toFinset = s.toFinset.filter (p ·) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : List α
    p : α → Bool
    ⊢ Eq (List.filter p s).toFinset (Finset.filter (fun x => Eq (p x) Bool.true) s …
  -/
  ext; simp [List.mem_filter]
       /-
         🎉 no goals
       -/


@[simp]
theorem toList_eq_nil {s : Finset α} : s.toList = [] ↔ s = ∅ :=
  Multiset.toList_eq_nil.trans val_eq_zero


                                                                     /-
                                                                       α : Type u_1
                                                                       s : Finset α
                                                                       ⊢ Iff (Eq s.toList.isEmpty Bool.true) (Eq s EmptyCollection.emptyCollection)
                                                                     -/
theorem empty_toList {s : Finset α} : s.toList.isEmpty ↔ s = ∅ := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
theorem toList_empty : (∅ : Finset α).toList = [] :=
  toList_eq_nil.mpr rfl


theorem Nonempty.toList_ne_nil {s : Finset α} (hs : s.Nonempty) : s.toList ≠ [] :=
  mt toList_eq_nil.mp hs.ne_empty


theorem Nonempty.not_empty_toList {s : Finset α} (hs : s.Nonempty) : ¬s.toList.isEmpty :=
  mt empty_toList.mp hs.ne_empty


/-- Given a finset `l` and a predicate `p`, associate to a proof that there is a unique element of
`l` satisfying `p` this unique element, as an element of the corresponding subtype. -/
def chooseX (hp : ∃! a, a ∈ l ∧ p a) : { a // a ∈ l ∧ p a } :=
  Multiset.chooseX p l.val hp


/-- Given a finset `l` and a predicate `p`, associate to a proof that there is a unique element of
`l` satisfying `p` this unique element, as an element of the ambient type. -/
def choose (hp : ∃! a, a ∈ l ∧ p a) : α :=
  chooseX p l hp


theorem choose_spec (hp : ∃! a, a ∈ l ∧ p a) : choose p l hp ∈ l ∧ p (choose p l hp) :=
  (chooseX p l hp).property


theorem choose_mem (hp : ∃! a, a ∈ l ∧ p a) : choose p l hp ∈ l :=
  (choose_spec _ _ _).1


theorem choose_property (hp : ∃! a, a ∈ l ∧ p a) : p (choose p l hp) :=
  (choose_spec _ _ _).2


/-- The disjoint union of finsets is a sum -/
def Finset.union (s t : Finset α) (h : Disjoint s t) :
    s ⊕ t ≃ (s ∪ t : Finset α) :=
  Equiv.Set.ofEq (coe_union _ _) |>.trans (Equiv.Set.union (disjoint_coe.mpr h)) |>.symm


@[simp]
theorem Finset.union_symm_inl (h : Disjoint s t) (x : s) :
    Equiv.Finset.union s t h (Sum.inl x) = ⟨x, Finset.mem_union.mpr <| Or.inl x.2⟩ :=
  rfl


@[simp]
theorem Finset.union_symm_inr (h : Disjoint s t) (y : t) :
    Equiv.Finset.union s t h (Sum.inr y) = ⟨y, Finset.mem_union.mpr <| Or.inr y.2⟩ :=
  rfl


/-- The type of dependent functions on the disjoint union of finsets `s ∪ t` is equivalent to the
  type of pairs of functions on `s` and on `t`. This is similar to `Equiv.sumPiEquivProdPi`. -/
def piFinsetUnion {ι} [DecidableEq ι] (α : ι → Type*) {s t : Finset ι} (h : Disjoint s t) :
    ((∀ i : s, α i) × ∀ i : t, α i) ≃ ∀ i : (s ∪ t : Finset ι), α i :=
  let e := Equiv.Finset.union s t h
  sumPiEquivProdPi (fun b ↦ α (e b)) |>.symm.trans (.piCongrLeft (fun i : ↥(s ∪ t) ↦ α i) e)


@[simp]
lemma toFinset_replicate (n : ℕ) (a : α) :
    (replicate n a).toFinset = if n = 0 then ∅ else {a} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    n : Nat
    a : α
    ⊢ Eq (Multiset.replicate n a).toFinset (ite (Eq n 0) EmptyCollection.emptyColl …
  -/
  ext x
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    n : Nat
    a x : α
    ⊢ Iff (Membership.mem (Multiset.replicate n a).toFinset x) (Membership.mem (it …
  -/
  simp only [mem_toFinset, Finset.mem_singleton, mem_replicate]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    n : Nat
    a x : α
    ⊢ Iff (And (Ne n 0) (Eq x a)) (Membership.mem (ite (Eq n 0) EmptyCollection.em …
  -/
                        /-
                          🎉 no goals
                        -/
  split_ifs with hn <;> simp [hn]
                        /-
                          🎉 no goals
                        -/


