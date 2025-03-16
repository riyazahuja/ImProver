@[simp]
theorem dedup_eq_self [DecidableEq α] (s : Finset α) : dedup s.1 = s.1 :=
  s.2.dedup


/-- `toFinset s` removes duplicates from the multiset `s` to produce a finset. -/
def toFinset (s : Multiset α) : Finset α :=
  ⟨_, nodup_dedup s⟩


@[simp]
theorem toFinset_val (s : Multiset α) : s.toFinset.1 = s.dedup :=
  rfl


theorem toFinset_eq {s : Multiset α} (n : Nodup s) : Finset.mk s n = s.toFinset :=
  Finset.val_inj.1 n.dedup.symm


theorem Nodup.toFinset_inj {l l' : Multiset α} (hl : Nodup l) (hl' : Nodup l')
    (h : l.toFinset = l'.toFinset) : l = l' := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : Multiset α
    hl : l.Nodup
    hl' : l'.Nodup
    h : Eq l.toFinset l'.toFinset
    ⊢ Eq l l'
  -/
  simpa [← toFinset_eq hl, ← toFinset_eq hl'] using h
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_toFinset {a : α} {s : Multiset α} : a ∈ s.toFinset ↔ a ∈ s :=
  mem_dedup


@[simp]
theorem toFinset_subset : s.toFinset ⊆ t.toFinset ↔ s ⊆ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Iff (HasSubset.Subset s.toFinset t.toFinset) (HasSubset.Subset s t)
  -/
  simp only [Finset.subset_iff, Multiset.subset_iff, Multiset.mem_toFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_ssubset : s.toFinset ⊂ t.toFinset ↔ s ⊂ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Iff (HasSSubset.SSubset s.toFinset t.toFinset) (HasSSubset.SSubset s t)
  -/
  simp_rw [Finset.ssubset_def, toFinset_subset]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Iff (And (HasSubset.Subset s t) (Not (HasSubset.Subset t s))) (HasSSubset.SS …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_dedup (m : Multiset α) : m.dedup.toFinset = m.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq m.dedup.toFinset m.toFinset
  -/
  simp_rw [toFinset, dedup_idem]
  /-
    🎉 no goals
  -/


instance isWellFounded_ssubset : IsWellFounded (Multiset β) (· ⊂ ·) := by
  classical
  exact Subrelation.isWellFounded (InvImage _ toFinset) toFinset_ssubset.2


@[simp]
theorem val_toFinset [DecidableEq α] (s : Finset α) : s.val.toFinset = s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq s.val.toFinset s
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a✝ : α
    ⊢ Iff (Membership.mem s.val.toFinset a✝) (Membership.mem s a✝)
  -/
  rw [Multiset.mem_toFinset, ← mem_def]
  /-
    🎉 no goals
  -/


theorem val_le_iff_val_subset {a : Finset α} {b : Multiset α} : a.val ≤ b ↔ a.val ⊆ b :=
  Multiset.le_iff_subset a.nodup


/-- `toFinset l` removes duplicates from the list `l` to produce a finset. -/
def toFinset (l : List α) : Finset α :=
  Multiset.toFinset l


@[simp]
theorem toFinset_val (l : List α) : l.toFinset.1 = (l.dedup : Multiset α) :=
  rfl


@[simp]
theorem toFinset_coe (l : List α) : (l : Multiset α).toFinset = l.toFinset :=
  rfl


theorem toFinset_eq (n : Nodup l) : @Finset.mk α l n = l.toFinset :=
                             /-
                               α : Type u_1
                               inst✝ : DecidableEq α
                               l : List α
                               n : l.Nodup
                               ⊢ (↑l).Nodup
                             -/
  Multiset.toFinset_eq <| by rwa [Multiset.coe_nodup]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem mem_toFinset : a ∈ l.toFinset ↔ a ∈ l :=
  mem_dedup


@[simp, norm_cast]
theorem coe_toFinset (l : List α) : (l.toFinset : Set α) = { a | a ∈ l } :=
  Set.ext fun _ => List.mem_toFinset


theorem toFinset_surj_on : Set.SurjOn toFinset { l : List α | l.Nodup } Set.univ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    ⊢ Set.SurjOn List.toFinset (setOf fun l => l.Nodup) Set.univ
  -/
  rintro ⟨⟨l⟩, hl⟩ _
  /-
    case mk.mk
    α : Type u_1
    inst✝ : DecidableEq α
    val✝ : Multiset α
    l : List α
    hl : Multiset.Nodup (Quot.mk (⇑(List.isSetoid α)) l)
    a✝ : Membership.mem Set.univ { val := Quot.mk (⇑(List.isSetoid α)) l, nodup := …
    ⊢ Membership.mem (Set.image List.toFinset (setOf fun l => l.Nodup)) { val := Q …
  -/
  exact ⟨l, hl, (toFinset_eq hl).symm⟩
  /-
    🎉 no goals
  -/


theorem toFinset_surjective : Surjective (toFinset : List α → Finset α) := fun s =>
  let ⟨l, _, hls⟩ := toFinset_surj_on (Set.mem_univ s)
  ⟨l, hls⟩


theorem toFinset_eq_iff_perm_dedup : l.toFinset = l'.toFinset ↔ l.dedup ~ l'.dedup := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    ⊢ Iff (Eq l.toFinset l'.toFinset) (l.dedup.Perm l'.dedup)
  -/
  simp [Finset.ext_iff, perm_ext_iff_of_nodup (nodup_dedup _) (nodup_dedup _)]
  /-
    🎉 no goals
  -/


theorem toFinset.ext_iff {a b : List α} : a.toFinset = b.toFinset ↔ ∀ x, x ∈ a ↔ x ∈ b := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : List α
    ⊢ Iff (Eq a.toFinset b.toFinset) (∀ (x : α), Iff (Membership.mem a x) (Members …
  -/
  simp only [Finset.ext_iff, mem_toFinset]
  /-
    🎉 no goals
  -/


theorem toFinset.ext : (∀ x, x ∈ l ↔ x ∈ l') → l.toFinset = l'.toFinset :=
  toFinset.ext_iff.mpr


theorem toFinset_eq_of_perm (l l' : List α) (h : l ~ l') : l.toFinset = l'.toFinset :=
  toFinset_eq_iff_perm_dedup.mpr h.dedup


theorem perm_of_nodup_nodup_toFinset_eq (hl : Nodup l) (hl' : Nodup l')
    (h : l.toFinset = l'.toFinset) : l ~ l' := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    hl : l.Nodup
    hl' : l'.Nodup
    h : Eq l.toFinset l'.toFinset
    ⊢ l.Perm l'
  -/
  rw [← Multiset.coe_eq_coe]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    hl : l.Nodup
    hl' : l'.Nodup
    h : Eq l.toFinset l'.toFinset
    ⊢ Eq ↑l ↑l'
  -/
  exact Multiset.Nodup.toFinset_inj hl hl' h
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_reverse {l : List α} : toFinset l.reverse = l.toFinset :=
  toFinset_eq_of_perm _ _ (reverse_perm l)


/-- Produce a list of the elements in the finite set using choice. -/
noncomputable def toList (s : Finset α) : List α :=
  s.1.toList


theorem nodup_toList (s : Finset α) : s.toList.Nodup := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ s.toList.Nodup
  -/
  rw [toList, ← Multiset.coe_nodup, Multiset.coe_toList]
  /-
    α : Type u_1
    s : Finset α
    ⊢ s.val.Nodup
  -/
  exact s.nodup
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_toList {a : α} {s : Finset α} : a ∈ s.toList ↔ a ∈ s :=
  Multiset.mem_toList


@[simp, norm_cast]
theorem coe_toList (s : Finset α) : (s.toList : Multiset α) = s.val :=
  s.val.coe_toList


@[simp]
theorem toList_toFinset [DecidableEq α] (s : Finset α) : s.toList.toFinset = s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq s.toList.toFinset s
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a✝ : α
    ⊢ Iff (Membership.mem s.toList.toFinset a✝) (Membership.mem s a✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem _root_.List.toFinset_toList [DecidableEq α] {s : List α} (hs : s.Nodup) :
    s.toFinset.toList.Perm s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : List α
    hs : s.Nodup
    ⊢ s.toFinset.toList.Perm s
  -/
  apply List.perm_of_nodup_nodup_toFinset_eq (nodup_toList _) hs
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : List α
    hs : s.Nodup
    ⊢ Eq s.toFinset.toList.toFinset s.toFinset
  -/
  rw [toList_toFinset]
  /-
    🎉 no goals
  -/


theorem exists_list_nodup_eq [DecidableEq α] (s : Finset α) :
    ∃ l : List α, l.Nodup ∧ l.toFinset = s :=
  ⟨s.toList, s.nodup_toList, s.toList_toFinset⟩


