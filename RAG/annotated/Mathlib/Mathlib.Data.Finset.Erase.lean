/-- `erase s a` is the set `s - {a}`, that is, the elements of `s` which are
  not equal to `a`. -/
def erase (s : Finset α) (a : α) : Finset α :=
  ⟨_, s.2.erase a⟩


@[simp]
theorem erase_val (s : Finset α) (a : α) : (erase s a).1 = s.1.erase a :=
  rfl


@[simp]
theorem mem_erase {a b : α} {s : Finset α} : a ∈ erase s b ↔ a ≠ b ∧ a ∈ s :=
  s.2.mem_erase_iff


theorem not_mem_erase (a : α) (s : Finset α) : a ∉ erase s a :=
  s.2.not_mem_erase


theorem ne_of_mem_erase : b ∈ erase s a → b ≠ a := fun h => (mem_erase.1 h).1


theorem mem_of_mem_erase : b ∈ erase s a → b ∈ s :=
  Multiset.mem_of_mem_erase


theorem mem_erase_of_ne_of_mem : a ≠ b → a ∈ s → a ∈ erase s b := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a b : α
    ⊢ Ne a b → Membership.mem s a → Membership.mem (s.erase b) a
  -/
  simp only [mem_erase]; exact And.intro
                         /-
                           🎉 no goals
                         -/


/-- An element of `s` that is not an element of `erase s a` must be`a`. -/
theorem eq_of_mem_of_not_mem_erase (hs : b ∈ s) (hsa : b ∉ s.erase a) : b = a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a b : α
    hs : Membership.mem s b
    hsa : Not (Membership.mem (s.erase a) b)
    ⊢ Eq b a
  -/
  rw [mem_erase, not_and] at hsa
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a b : α
    hs : Membership.mem s b
    hsa : Ne b a → Not (Membership.mem s b)
    ⊢ Eq b a
  -/
  exact not_imp_not.mp hsa hs
  /-
    🎉 no goals
  -/


@[simp]
theorem erase_eq_of_not_mem {a : α} {s : Finset α} (h : a ∉ s) : erase s a = s :=
  eq_of_veq <| erase_of_not_mem h


@[simp]
theorem erase_eq_self : s.erase a = s ↔ a ∉ s :=
  ⟨fun h => h ▸ not_mem_erase _ _, erase_eq_of_not_mem⟩


theorem erase_ne_self : s.erase a ≠ s ↔ a ∈ s :=
  erase_eq_self.not_left


theorem erase_subset_erase (a : α) {s t : Finset α} (h : s ⊆ t) : erase s a ⊆ erase t a :=
  val_le_iff.1 <| erase_le_erase _ <| val_le_iff.2 h


theorem erase_subset (a : α) (s : Finset α) : erase s a ⊆ s :=
  Multiset.erase_subset _ _


theorem subset_erase {a : α} {s t : Finset α} : s ⊆ t.erase a ↔ s ⊆ t ∧ a ∉ s :=
  ⟨fun h => ⟨h.trans (erase_subset _ _), fun ha => not_mem_erase _ _ (h ha)⟩,
   fun h _b hb => mem_erase.2 ⟨ne_of_mem_of_not_mem hb h.2, h.1 hb⟩⟩


@[simp, norm_cast]
theorem coe_erase (a : α) (s : Finset α) : ↑(erase s a) = (s \ {a} : Set α) :=
                                         /-
                                           α : Type u_1
                                           inst✝ : DecidableEq α
                                           a : α
                                           s : Finset α
                                           x✝ : α
                                           ⊢ Iff (And (Ne x✝ a) (Membership.mem s x✝)) (Membership.mem (SDiff.sdiff (↑s)  …
                                         -/
  Set.ext fun _ => mem_erase.trans <| by rw [and_comm, Set.mem_diff, Set.mem_singleton_iff, mem_coe]
                                         /-
                                           🎉 no goals
                                         -/


                                                                                  /-
                                                                                    α : Type u_1
                                                                                    inst✝ : DecidableEq α
                                                                                    a : α
                                                                                    s : Finset α
                                                                                    ⊢ Eq ((s.erase a).erase a) (s.erase a)
                                                                                  -/
theorem erase_idem {a : α} {s : Finset α} : erase (erase s a) a = erase s a := by simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem erase_right_comm {a b : α} {s : Finset α} : erase (erase s a) b = erase (erase s b) a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Finset α
    ⊢ Eq ((s.erase a).erase b) ((s.erase b).erase a)
  -/
  ext x
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Finset α
    x : α
    ⊢ Iff (Membership.mem ((s.erase a).erase b) x) (Membership.mem ((s.erase b).er …
  -/
  simp only [mem_erase, ← and_assoc]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Finset α
    x : α
    ⊢ Iff (And (And (Ne x b) (Ne x a)) (Membership.mem s x)) (And (And (Ne x a) (N …
  -/
  rw [@and_comm (x ≠ a)]
  /-
    🎉 no goals
  -/


theorem erase_inj {x y : α} (s : Finset α) (hx : x ∈ s) : s.erase x = s.erase y ↔ x = y := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y : α
    s : Finset α
    hx : Membership.mem s x
    ⊢ Iff (Eq (s.erase x) (s.erase y)) (Eq x y)
  -/
  refine ⟨fun h => eq_of_mem_of_not_mem_erase hx ?_, congr_arg _⟩
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y : α
    s : Finset α
    hx : Membership.mem s x
    h : Eq (s.erase x) (s.erase y)
    ⊢ Not (Membership.mem (s.erase y) x)
  -/
  rw [← h]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y : α
    s : Finset α
    hx : Membership.mem s x
    h : Eq (s.erase x) (s.erase y)
    ⊢ Not (Membership.mem (s.erase x) x)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem erase_injOn (s : Finset α) : Set.InjOn s.erase s := fun _ _ _ _ => (erase_inj s ‹_›).mp


