/-- `dedup s` removes duplicates from `s`, yielding a `nodup` multiset. -/
def dedup (s : Multiset α) : Multiset α :=
  Quot.liftOn s (fun l => (l.dedup : Multiset α)) fun _ _ p => Quot.sound p.dedup


@[simp]
theorem coe_dedup (l : List α) : @dedup α _ l = l.dedup :=
  rfl


@[simp]
theorem dedup_zero : @dedup α _ 0 = 0 :=
  rfl


@[simp]
theorem mem_dedup {a : α} {s : Multiset α} : a ∈ dedup s ↔ a ∈ s :=
  Quot.induction_on s fun _ => List.mem_dedup


@[simp]
theorem dedup_cons_of_mem {a : α} {s : Multiset α} : a ∈ s → dedup (a ::ₘ s) = dedup s :=
  Quot.induction_on s fun _ m => @congr_arg _ _ _ _ ofList <| List.dedup_cons_of_mem m


@[simp]
theorem dedup_cons_of_not_mem {a : α} {s : Multiset α} : a ∉ s → dedup (a ::ₘ s) = a ::ₘ dedup s :=
  Quot.induction_on s fun _ m => congr_arg ofList <| List.dedup_cons_of_not_mem m


theorem dedup_le (s : Multiset α) : dedup s ≤ s :=
  Quot.induction_on s fun _ => (dedup_sublist _).subperm


theorem dedup_subset (s : Multiset α) : dedup s ⊆ s :=
  subset_of_le <| dedup_le _


theorem subset_dedup (s : Multiset α) : s ⊆ dedup s := fun _ => mem_dedup.2


@[simp]
theorem dedup_subset' {s t : Multiset α} : dedup s ⊆ t ↔ s ⊆ t :=
  ⟨Subset.trans (subset_dedup _), Subset.trans (dedup_subset _)⟩


@[simp]
theorem subset_dedup' {s t : Multiset α} : s ⊆ dedup t ↔ s ⊆ t :=
  ⟨fun h => Subset.trans h (dedup_subset _), fun h => Subset.trans h (subset_dedup _)⟩


@[simp]
theorem nodup_dedup (s : Multiset α) : Nodup (dedup s) :=
  Quot.induction_on s List.nodup_dedup


theorem dedup_eq_self {s : Multiset α} : dedup s = s ↔ Nodup s :=
  ⟨fun e => e ▸ nodup_dedup s, Quot.induction_on s fun _ h => congr_arg ofList h.dedup⟩


alias ⟨_, Nodup.dedup⟩ := dedup_eq_self


theorem count_dedup (m : Multiset α) (a : α) : m.dedup.count a = if a ∈ m then 1 else 0 :=
  Quot.induction_on m fun _ => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      m : Multiset α
      a : α
      x✝ : List α
      ⊢ Eq (Multiset.count a (Multiset.dedup (Quot.mk (⇑(List.isSetoid α)) x✝))) (it …
    -/
    simp only [quot_mk_to_coe'', coe_dedup, mem_coe, List.mem_dedup, coe_nodup, coe_count]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      m : Multiset α
      a : α
      x✝ : List α
      ⊢ Eq (List.count a x✝.dedup) (ite (Membership.mem x✝ a) 1 0)
    -/
    apply List.count_dedup _ _
    /-
      🎉 no goals
    -/


@[simp]
theorem dedup_idem {m : Multiset α} : m.dedup.dedup = m.dedup :=
  Quot.induction_on m fun _ => @congr_arg _ _ _ _ ofList List.dedup_idem


theorem dedup_eq_zero {s : Multiset α} : dedup s = 0 ↔ s = 0 :=
  ⟨fun h => eq_zero_of_subset_zero <| h ▸ subset_dedup _, fun h => h.symm ▸ dedup_zero⟩


@[simp]
theorem dedup_singleton {a : α} : dedup ({a} : Multiset α) = {a} :=
  (nodup_singleton _).dedup


theorem le_dedup {s t : Multiset α} : s ≤ dedup t ↔ s ≤ t ∧ Nodup s :=
  ⟨fun h => ⟨le_trans h (dedup_le _), nodup_of_le h (nodup_dedup _)⟩,
   fun ⟨l, d⟩ => (le_iff_subset d).2 <| Subset.trans (subset_of_le l) (subset_dedup _)⟩


theorem le_dedup_self {s : Multiset α} : s ≤ dedup s ↔ Nodup s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Iff (LE.le s s.dedup) s.Nodup
  -/
  rw [le_dedup, and_iff_right le_rfl]
  /-
    🎉 no goals
  -/


theorem dedup_ext {s t : Multiset α} : dedup s = dedup t ↔ ∀ a, a ∈ s ↔ a ∈ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Iff (Eq s.dedup t.dedup) (∀ (a : α), Iff (Membership.mem s a) (Membership.me …
  -/
  simp [Nodup.ext]
  /-
    🎉 no goals
  -/


theorem dedup_map_of_injective [DecidableEq β] {f : α → β} (hf : Function.Injective f)
    (s : Multiset α) :
    (s.map f).dedup = s.dedup.map f :=
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    inst✝¹ : DecidableEq α
                                    inst✝ : DecidableEq β
                                    f : α → β
                                    hf : Function.Injective f
                                    s : Multiset α
                                    l : List α
                                    ⊢ Eq (Multiset.map f (Quot.mk (⇑(List.isSetoid α)) l)).dedup (Multiset.map f ( …
                                  -/
  Quot.induction_on s fun l => by simp [List.dedup_map_of_injective hf l]
                                  /-
                                    🎉 no goals
                                  -/


theorem dedup_map_dedup_eq [DecidableEq β] (f : α → β) (s : Multiset α) :
    dedup (map f (dedup s)) = dedup (map f s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    s : Multiset α
    ⊢ Eq (Multiset.map f s.dedup).dedup (Multiset.map f s).dedup
  -/
  simp [dedup_ext]
  /-
    🎉 no goals
  -/


@[simp]
theorem dedup_nsmul {s : Multiset α} {n : ℕ} (h0 : n ≠ 0) : (n • s).dedup = s.dedup := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    n : Nat
    h0 : Ne n 0
    ⊢ Eq (HSMul.hSMul n s).dedup s.dedup
  -/
  ext a
  /-
    case a
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    n : Nat
    h0 : Ne n 0
    a : α
    ⊢ Eq (Multiset.count a (HSMul.hSMul n s).dedup) (Multiset.count a s.dedup)
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : a ∈ s <;> simp [h, h0]
                         /-
                           🎉 no goals
                         -/


theorem Nodup.le_dedup_iff_le {s t : Multiset α} (hno : s.Nodup) : s ≤ t.dedup ↔ s ≤ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    hno : s.Nodup
    ⊢ Iff (LE.le s t.dedup) (LE.le s t)
  -/
  simp [le_dedup, hno]
  /-
    🎉 no goals
  -/


theorem Subset.dedup_add_right {s t : Multiset α} (h : s ⊆ t) :
    dedup (s + t) = dedup t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    h : HasSubset.Subset s t
    ⊢ Eq (HAdd.hAdd s t).dedup t.dedup
  -/
  induction s, t using Quot.induction_on₂
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a✝ b✝ : List α
    h : HasSubset.Subset (Quot.mk (⇑(List.isSetoid α)) a✝) (Quot.mk (⇑(List.isSeto …
    ⊢ Eq (HAdd.hAdd (Quot.mk (⇑(List.isSetoid α)) a✝) (Quot.mk (⇑(List.isSetoid α) …
  -/
  exact congr_arg ((↑) : List α → Multiset α) <| List.Subset.dedup_append_right h
  /-
    🎉 no goals
  -/


theorem Subset.dedup_add_left {s t : Multiset α} (h : t ⊆ s) :
    dedup (s + t) = dedup s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    h : HasSubset.Subset t s
    ⊢ Eq (HAdd.hAdd s t).dedup s.dedup
  -/
  rw [add_comm, Subset.dedup_add_right h]
  /-
    🎉 no goals
  -/


theorem Disjoint.dedup_add {s t : Multiset α} (h : Disjoint s t) :
    dedup (s + t) = dedup s + dedup t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    h : Disjoint s t
    ⊢ Eq (HAdd.hAdd s t).dedup (HAdd.hAdd s.dedup t.dedup)
  -/
  induction s, t using Quot.induction_on₂
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a✝ b✝ : List α
    h : Disjoint (Quot.mk (⇑(List.isSetoid α)) a✝) (Quot.mk (⇑(List.isSetoid α)) b✝)
    ⊢ Eq (HAdd.hAdd (Quot.mk (⇑(List.isSetoid α)) a✝) (Quot.mk (⇑(List.isSetoid α) …
  -/
  exact congr_arg ((↑) : List α → Multiset α) <| List.Disjoint.dedup_append (by simpa using h)
  /-
    🎉 no goals
  -/


/-- Note that the stronger `List.Subset.dedup_append_right` is proved earlier. -/
theorem _root_.List.Subset.dedup_append_left {s t : List α} (h : t ⊆ s) :
    List.dedup (s ++ t) ~ List.dedup s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : List α
    h : HasSubset.Subset t s
    ⊢ (HAppend.hAppend s t).dedup.Perm s.dedup
  -/
  rw [← coe_eq_coe, ← coe_dedup, ← coe_add, Subset.dedup_add_left h, coe_dedup]
  /-
    🎉 no goals
  -/


theorem Multiset.Nodup.le_nsmul_iff_le {α : Type*} {s t : Multiset α} {n : ℕ} (h : s.Nodup)
    (hn : n ≠ 0) : s ≤ n • t ↔ s ≤ t := by
  classical
    rw [← h.le_dedup_iff_le, Iff.comm, ← h.le_dedup_iff_le]
    simp [hn]

