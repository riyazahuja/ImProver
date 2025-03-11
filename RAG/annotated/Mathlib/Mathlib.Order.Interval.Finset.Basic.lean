@[simp]
theorem nonempty_Icc : (Icc a b).Nonempty ↔ a ≤ b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ Iff (Finset.Icc a b).Nonempty (LE.le a b)
  -/
  rw [← coe_nonempty, coe_Icc, Set.nonempty_Icc]
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
alias ⟨_, Aesop.nonempty_Icc_of_le⟩ := nonempty_Icc


@[simp]
theorem nonempty_Ico : (Ico a b).Nonempty ↔ a < b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ Iff (Finset.Ico a b).Nonempty (LT.lt a b)
  -/
  rw [← coe_nonempty, coe_Ico, Set.nonempty_Ico]
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
alias ⟨_, Aesop.nonempty_Ico_of_lt⟩ := nonempty_Ico


@[simp]
theorem nonempty_Ioc : (Ioc a b).Nonempty ↔ a < b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ Iff (Finset.Ioc a b).Nonempty (LT.lt a b)
  -/
  rw [← coe_nonempty, coe_Ioc, Set.nonempty_Ioc]
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
alias ⟨_, Aesop.nonempty_Ioc_of_lt⟩ := nonempty_Ioc

-- TODO: This is nonsense. A locally finite order is never densely ordered

@[simp]
theorem nonempty_Ioo [DenselyOrdered α] : (Ioo a b).Nonempty ↔ a < b := by
  /-
    α : Type u_2
    a b : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DenselyOrdered α
    ⊢ Iff (Finset.Ioo a b).Nonempty (LT.lt a b)
  -/
  rw [← coe_nonempty, coe_Ioo, Set.nonempty_Ioo]
  /-
    🎉 no goals
  -/


@[simp]
theorem Icc_eq_empty_iff : Icc a b = ∅ ↔ ¬a ≤ b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ Iff (Eq (Finset.Icc a b) EmptyCollection.emptyCollection) (Not (LE.le a b))
  -/
  rw [← coe_eq_empty, coe_Icc, Set.Icc_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ico_eq_empty_iff : Ico a b = ∅ ↔ ¬a < b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ Iff (Eq (Finset.Ico a b) EmptyCollection.emptyCollection) (Not (LT.lt a b))
  -/
  rw [← coe_eq_empty, coe_Ico, Set.Ico_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioc_eq_empty_iff : Ioc a b = ∅ ↔ ¬a < b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ Iff (Eq (Finset.Ioc a b) EmptyCollection.emptyCollection) (Not (LT.lt a b))
  -/
  rw [← coe_eq_empty, coe_Ioc, Set.Ioc_eq_empty_iff]
  /-
    🎉 no goals
  -/

-- TODO: This is nonsense. A locally finite order is never densely ordered

@[simp]
theorem Ioo_eq_empty_iff [DenselyOrdered α] : Ioo a b = ∅ ↔ ¬a < b := by
  /-
    α : Type u_2
    a b : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DenselyOrdered α
    ⊢ Iff (Eq (Finset.Ioo a b) EmptyCollection.emptyCollection) (Not (LT.lt a b))
  -/
  rw [← coe_eq_empty, coe_Ioo, Set.Ioo_eq_empty_iff]
  /-
    🎉 no goals
  -/


alias ⟨_, Icc_eq_empty⟩ := Icc_eq_empty_iff


alias ⟨_, Ico_eq_empty⟩ := Ico_eq_empty_iff


alias ⟨_, Ioc_eq_empty⟩ := Ioc_eq_empty_iff


@[simp]
theorem Ioo_eq_empty (h : ¬a < b) : Ioo a b = ∅ :=
  eq_empty_iff_forall_not_mem.2 fun _ hx => h ((mem_Ioo.1 hx).1.trans (mem_Ioo.1 hx).2)


@[simp]
theorem Icc_eq_empty_of_lt (h : b < a) : Icc a b = ∅ :=
  Icc_eq_empty h.not_le


@[simp]
theorem Ico_eq_empty_of_le (h : b ≤ a) : Ico a b = ∅ :=
  Ico_eq_empty h.not_lt


@[simp]
theorem Ioc_eq_empty_of_le (h : b ≤ a) : Ioc a b = ∅ :=
  Ioc_eq_empty h.not_lt


@[simp]
theorem Ioo_eq_empty_of_le (h : b ≤ a) : Ioo a b = ∅ :=
  Ioo_eq_empty h.not_lt


                                                 /-
                                                   α : Type u_2
                                                   a b : α
                                                   inst✝¹ : Preorder α
                                                   inst✝ : LocallyFiniteOrder α
                                                   ⊢ Iff (Membership.mem (Finset.Icc a b) a) (LE.le a b)
                                                 -/
theorem left_mem_Icc : a ∈ Icc a b ↔ a ≤ b := by simp only [mem_Icc, true_and, le_rfl]
                                                 /-
                                                   🎉 no goals
                                                 -/


                                                 /-
                                                   α : Type u_2
                                                   a b : α
                                                   inst✝¹ : Preorder α
                                                   inst✝ : LocallyFiniteOrder α
                                                   ⊢ Iff (Membership.mem (Finset.Ico a b) a) (LT.lt a b)
                                                 -/
theorem left_mem_Ico : a ∈ Ico a b ↔ a < b := by simp only [mem_Ico, true_and, le_refl]
                                                 /-
                                                   🎉 no goals
                                                 -/


                                                  /-
                                                    α : Type u_2
                                                    a b : α
                                                    inst✝¹ : Preorder α
                                                    inst✝ : LocallyFiniteOrder α
                                                    ⊢ Iff (Membership.mem (Finset.Icc a b) b) (LE.le a b)
                                                  -/
theorem right_mem_Icc : b ∈ Icc a b ↔ a ≤ b := by simp only [mem_Icc, and_true, le_rfl]
                                                  /-
                                                    🎉 no goals
                                                  -/


                                                  /-
                                                    α : Type u_2
                                                    a b : α
                                                    inst✝¹ : Preorder α
                                                    inst✝ : LocallyFiniteOrder α
                                                    ⊢ Iff (Membership.mem (Finset.Ioc a b) b) (LT.lt a b)
                                                  -/
theorem right_mem_Ioc : b ∈ Ioc a b ↔ a < b := by simp only [mem_Ioc, and_true, le_rfl]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem left_not_mem_Ioc : a ∉ Ioc a b := fun h => lt_irrefl _ (mem_Ioc.1 h).1


theorem left_not_mem_Ioo : a ∉ Ioo a b := fun h => lt_irrefl _ (mem_Ioo.1 h).1


theorem right_not_mem_Ico : b ∉ Ico a b := fun h => lt_irrefl _ (mem_Ico.1 h).2


theorem right_not_mem_Ioo : b ∉ Ioo a b := fun h => lt_irrefl _ (mem_Ioo.1 h).2


theorem Icc_subset_Icc (ha : a₂ ≤ a₁) (hb : b₁ ≤ b₂) : Icc a₁ b₁ ⊆ Icc a₂ b₂ := by
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ha : LE.le a₂ a₁
    hb : LE.le b₁ b₂
    ⊢ HasSubset.Subset (Finset.Icc a₁ b₁) (Finset.Icc a₂ b₂)
  -/
  simpa [← coe_subset] using Set.Icc_subset_Icc ha hb
  /-
    🎉 no goals
  -/


theorem Ico_subset_Ico (ha : a₂ ≤ a₁) (hb : b₁ ≤ b₂) : Ico a₁ b₁ ⊆ Ico a₂ b₂ := by
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ha : LE.le a₂ a₁
    hb : LE.le b₁ b₂
    ⊢ HasSubset.Subset (Finset.Ico a₁ b₁) (Finset.Ico a₂ b₂)
  -/
  simpa [← coe_subset] using Set.Ico_subset_Ico ha hb
  /-
    🎉 no goals
  -/


theorem Ioc_subset_Ioc (ha : a₂ ≤ a₁) (hb : b₁ ≤ b₂) : Ioc a₁ b₁ ⊆ Ioc a₂ b₂ := by
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ha : LE.le a₂ a₁
    hb : LE.le b₁ b₂
    ⊢ HasSubset.Subset (Finset.Ioc a₁ b₁) (Finset.Ioc a₂ b₂)
  -/
  simpa [← coe_subset] using Set.Ioc_subset_Ioc ha hb
  /-
    🎉 no goals
  -/


theorem Ioo_subset_Ioo (ha : a₂ ≤ a₁) (hb : b₁ ≤ b₂) : Ioo a₁ b₁ ⊆ Ioo a₂ b₂ := by
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ha : LE.le a₂ a₁
    hb : LE.le b₁ b₂
    ⊢ HasSubset.Subset (Finset.Ioo a₁ b₁) (Finset.Ioo a₂ b₂)
  -/
  simpa [← coe_subset] using Set.Ioo_subset_Ioo ha hb
  /-
    🎉 no goals
  -/


theorem Icc_subset_Icc_left (h : a₁ ≤ a₂) : Icc a₂ b ⊆ Icc a₁ b :=
  Icc_subset_Icc h le_rfl


theorem Ico_subset_Ico_left (h : a₁ ≤ a₂) : Ico a₂ b ⊆ Ico a₁ b :=
  Ico_subset_Ico h le_rfl


theorem Ioc_subset_Ioc_left (h : a₁ ≤ a₂) : Ioc a₂ b ⊆ Ioc a₁ b :=
  Ioc_subset_Ioc h le_rfl


theorem Ioo_subset_Ioo_left (h : a₁ ≤ a₂) : Ioo a₂ b ⊆ Ioo a₁ b :=
  Ioo_subset_Ioo h le_rfl


theorem Icc_subset_Icc_right (h : b₁ ≤ b₂) : Icc a b₁ ⊆ Icc a b₂ :=
  Icc_subset_Icc le_rfl h


theorem Ico_subset_Ico_right (h : b₁ ≤ b₂) : Ico a b₁ ⊆ Ico a b₂ :=
  Ico_subset_Ico le_rfl h


theorem Ioc_subset_Ioc_right (h : b₁ ≤ b₂) : Ioc a b₁ ⊆ Ioc a b₂ :=
  Ioc_subset_Ioc le_rfl h


theorem Ioo_subset_Ioo_right (h : b₁ ≤ b₂) : Ioo a b₁ ⊆ Ioo a b₂ :=
  Ioo_subset_Ioo le_rfl h


theorem Ico_subset_Ioo_left (h : a₁ < a₂) : Ico a₂ b ⊆ Ioo a₁ b := by
  /-
    α : Type u_2
    a₁ a₂ b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    h : LT.lt a₁ a₂
    ⊢ HasSubset.Subset (Finset.Ico a₂ b) (Finset.Ioo a₁ b)
  -/
  rw [← coe_subset, coe_Ico, coe_Ioo]
  /-
    α : Type u_2
    a₁ a₂ b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    h : LT.lt a₁ a₂
    ⊢ HasSubset.Subset (Set.Ico a₂ b) (Set.Ioo a₁ b)
  -/
  exact Set.Ico_subset_Ioo_left h
  /-
    🎉 no goals
  -/


theorem Ioc_subset_Ioo_right (h : b₁ < b₂) : Ioc a b₁ ⊆ Ioo a b₂ := by
  /-
    α : Type u_2
    a b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    h : LT.lt b₁ b₂
    ⊢ HasSubset.Subset (Finset.Ioc a b₁) (Finset.Ioo a b₂)
  -/
  rw [← coe_subset, coe_Ioc, coe_Ioo]
  /-
    α : Type u_2
    a b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    h : LT.lt b₁ b₂
    ⊢ HasSubset.Subset (Set.Ioc a b₁) (Set.Ioo a b₂)
  -/
  exact Set.Ioc_subset_Ioo_right h
  /-
    🎉 no goals
  -/


theorem Icc_subset_Ico_right (h : b₁ < b₂) : Icc a b₁ ⊆ Ico a b₂ := by
  /-
    α : Type u_2
    a b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    h : LT.lt b₁ b₂
    ⊢ HasSubset.Subset (Finset.Icc a b₁) (Finset.Ico a b₂)
  -/
  rw [← coe_subset, coe_Icc, coe_Ico]
  /-
    α : Type u_2
    a b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    h : LT.lt b₁ b₂
    ⊢ HasSubset.Subset (Set.Icc a b₁) (Set.Ico a b₂)
  -/
  exact Set.Icc_subset_Ico_right h
  /-
    🎉 no goals
  -/


theorem Ioo_subset_Ico_self : Ioo a b ⊆ Ico a b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Ioo a b) (Finset.Ico a b)
  -/
  rw [← coe_subset, coe_Ioo, coe_Ico]
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Set.Ioo a b) (Set.Ico a b)
  -/
  exact Set.Ioo_subset_Ico_self
  /-
    🎉 no goals
  -/


theorem Ioo_subset_Ioc_self : Ioo a b ⊆ Ioc a b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Ioo a b) (Finset.Ioc a b)
  -/
  rw [← coe_subset, coe_Ioo, coe_Ioc]
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Set.Ioo a b) (Set.Ioc a b)
  -/
  exact Set.Ioo_subset_Ioc_self
  /-
    🎉 no goals
  -/


theorem Ico_subset_Icc_self : Ico a b ⊆ Icc a b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Ico a b) (Finset.Icc a b)
  -/
  rw [← coe_subset, coe_Ico, coe_Icc]
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Set.Ico a b) (Set.Icc a b)
  -/
  exact Set.Ico_subset_Icc_self
  /-
    🎉 no goals
  -/


theorem Ioc_subset_Icc_self : Ioc a b ⊆ Icc a b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Ioc a b) (Finset.Icc a b)
  -/
  rw [← coe_subset, coe_Ioc, coe_Icc]
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Set.Ioc a b) (Set.Icc a b)
  -/
  exact Set.Ioc_subset_Icc_self
  /-
    🎉 no goals
  -/


theorem Ioo_subset_Icc_self : Ioo a b ⊆ Icc a b :=
  Ioo_subset_Ico_self.trans Ico_subset_Icc_self


theorem Icc_subset_Icc_iff (h₁ : a₁ ≤ b₁) : Icc a₁ b₁ ⊆ Icc a₂ b₂ ↔ a₂ ≤ a₁ ∧ b₁ ≤ b₂ := by
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    h₁ : LE.le a₁ b₁
    ⊢ Iff (HasSubset.Subset (Finset.Icc a₁ b₁) (Finset.Icc a₂ b₂)) (And (LE.le a₂  …
  -/
  rw [← coe_subset, coe_Icc, coe_Icc, Set.Icc_subset_Icc_iff h₁]
  /-
    🎉 no goals
  -/


theorem Icc_subset_Ioo_iff (h₁ : a₁ ≤ b₁) : Icc a₁ b₁ ⊆ Ioo a₂ b₂ ↔ a₂ < a₁ ∧ b₁ < b₂ := by
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    h₁ : LE.le a₁ b₁
    ⊢ Iff (HasSubset.Subset (Finset.Icc a₁ b₁) (Finset.Ioo a₂ b₂)) (And (LT.lt a₂  …
  -/
  rw [← coe_subset, coe_Icc, coe_Ioo, Set.Icc_subset_Ioo_iff h₁]
  /-
    🎉 no goals
  -/


theorem Icc_subset_Ico_iff (h₁ : a₁ ≤ b₁) : Icc a₁ b₁ ⊆ Ico a₂ b₂ ↔ a₂ ≤ a₁ ∧ b₁ < b₂ := by
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    h₁ : LE.le a₁ b₁
    ⊢ Iff (HasSubset.Subset (Finset.Icc a₁ b₁) (Finset.Ico a₂ b₂)) (And (LE.le a₂  …
  -/
  rw [← coe_subset, coe_Icc, coe_Ico, Set.Icc_subset_Ico_iff h₁]
  /-
    🎉 no goals
  -/


theorem Icc_subset_Ioc_iff (h₁ : a₁ ≤ b₁) : Icc a₁ b₁ ⊆ Ioc a₂ b₂ ↔ a₂ < a₁ ∧ b₁ ≤ b₂ :=
  (Icc_subset_Ico_iff h₁.dual).trans and_comm

--TODO: `Ico_subset_Ioo_iff`, `Ioc_subset_Ioo_iff`

theorem Icc_ssubset_Icc_left (hI : a₂ ≤ b₂) (ha : a₂ < a₁) (hb : b₁ ≤ b₂) :
    Icc a₁ b₁ ⊂ Icc a₂ b₂ := by
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    hI : LE.le a₂ b₂
    ha : LT.lt a₂ a₁
    hb : LE.le b₁ b₂
    ⊢ HasSSubset.SSubset (Finset.Icc a₁ b₁) (Finset.Icc a₂ b₂)
  -/
  rw [← coe_ssubset, coe_Icc, coe_Icc]
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    hI : LE.le a₂ b₂
    ha : LT.lt a₂ a₁
    hb : LE.le b₁ b₂
    ⊢ HasSSubset.SSubset (Set.Icc a₁ b₁) (Set.Icc a₂ b₂)
  -/
  exact Set.Icc_ssubset_Icc_left hI ha hb
  /-
    🎉 no goals
  -/


theorem Icc_ssubset_Icc_right (hI : a₂ ≤ b₂) (ha : a₂ ≤ a₁) (hb : b₁ < b₂) :
    Icc a₁ b₁ ⊂ Icc a₂ b₂ := by
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    hI : LE.le a₂ b₂
    ha : LE.le a₂ a₁
    hb : LT.lt b₁ b₂
    ⊢ HasSSubset.SSubset (Finset.Icc a₁ b₁) (Finset.Icc a₂ b₂)
  -/
  rw [← coe_ssubset, coe_Icc, coe_Icc]
  /-
    α : Type u_2
    a₁ a₂ b₁ b₂ : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    hI : LE.le a₂ b₂
    ha : LE.le a₂ a₁
    hb : LT.lt b₁ b₂
    ⊢ HasSSubset.SSubset (Set.Icc a₁ b₁) (Set.Icc a₂ b₂)
  -/
  exact Set.Icc_ssubset_Icc_right hI ha hb
  /-
    🎉 no goals
  -/


theorem Ico_self : Ico a a = ∅ :=
  Ico_eq_empty <| lt_irrefl _


theorem Ioc_self : Ioc a a = ∅ :=
  Ioc_eq_empty <| lt_irrefl _


theorem Ioo_self : Ioo a a = ∅ :=
  Ioo_eq_empty <| lt_irrefl _


/-- A set with upper and lower bounds in a locally finite order is a fintype -/
def _root_.Set.fintypeOfMemBounds {s : Set α} [DecidablePred (· ∈ s)] (ha : a ∈ lowerBounds s)
    (hb : b ∈ upperBounds s) : Fintype s :=
  Set.fintypeSubset (Set.Icc a b) fun _ hx => ⟨ha hx, hb hx⟩


theorem Ico_filter_lt_of_le_left [DecidablePred (· < c)] (hca : c ≤ a) :
    {x ∈ Ico a b | x < c} = ∅ :=
  filter_false_of_mem fun _ hx => (hca.trans (mem_Ico.1 hx).1).not_lt


theorem Ico_filter_lt_of_right_le [DecidablePred (· < c)] (hbc : b ≤ c) :
    {x ∈ Ico a b | x < c} = Ico a b :=
  filter_true_of_mem fun _ hx => (mem_Ico.1 hx).2.trans_le hbc


theorem Ico_filter_lt_of_le_right [DecidablePred (· < c)] (hcb : c ≤ b) :
    {x ∈ Ico a b | x < c} = Ico a c := by
  /-
    α : Type u_2
    a b c : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidablePred fun x => LT.lt x c
    hcb : LE.le c b
    ⊢ Eq (Finset.filter (fun x => LT.lt x c) (Finset.Ico a b)) (Finset.Ico a c)
  -/
  ext x
  /-
    case h
    α : Type u_2
    a b c : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidablePred fun x => LT.lt x c
    hcb : LE.le c b
    x : α
    ⊢ Iff (Membership.mem (Finset.filter (fun x => LT.lt x c) (Finset.Ico a b)) x) …
  -/
  rw [mem_filter, mem_Ico, mem_Ico, and_right_comm]
  /-
    case h
    α : Type u_2
    a b c : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidablePred fun x => LT.lt x c
    hcb : LE.le c b
    x : α
    ⊢ Iff (And (And (LE.le a x) (LT.lt x c)) (LT.lt x b)) (And (LE.le a x) (LT.lt  …
  -/
  exact and_iff_left_of_imp fun h => h.2.trans_le hcb
  /-
    🎉 no goals
  -/


theorem Ico_filter_le_of_le_left {a b c : α} [DecidablePred (c ≤ ·)] (hca : c ≤ a) :
    {x ∈ Ico a b | c ≤ x} = Ico a b :=
  filter_true_of_mem fun _ hx => hca.trans (mem_Ico.1 hx).1


theorem Ico_filter_le_of_right_le {a b : α} [DecidablePred (b ≤ ·)] :
    {x ∈ Ico a b | b ≤ x} = ∅ :=
  filter_false_of_mem fun _ hx => (mem_Ico.1 hx).2.not_le


theorem Ico_filter_le_of_left_le {a b c : α} [DecidablePred (c ≤ ·)] (hac : a ≤ c) :
    {x ∈ Ico a b | c ≤ x} = Ico c b := by
  /-
    α : Type u_2
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LE.le c x
    hac : LE.le a c
    ⊢ Eq (Finset.filter (fun x => LE.le c x) (Finset.Ico a b)) (Finset.Ico c b)
  -/
  ext x
  /-
    case h
    α : Type u_2
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LE.le c x
    hac : LE.le a c
    x : α
    ⊢ Iff (Membership.mem (Finset.filter (fun x => LE.le c x) (Finset.Ico a b)) x) …
  -/
  rw [mem_filter, mem_Ico, mem_Ico, and_comm, and_left_comm]
  /-
    case h
    α : Type u_2
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LE.le c x
    hac : LE.le a c
    x : α
    ⊢ Iff (And (LE.le a x) (And (LE.le c x) (LT.lt x b))) (And (LE.le c x) (LT.lt  …
  -/
  exact and_iff_right_of_imp fun h => hac.trans h.1
  /-
    🎉 no goals
  -/


theorem Icc_filter_lt_of_lt_right {a b c : α} [DecidablePred (· < c)] (h : b < c) :
    {x ∈ Icc a b | x < c} = Icc a b :=
  filter_true_of_mem fun _ hx => lt_of_le_of_lt (mem_Icc.1 hx).2 h


theorem Ioc_filter_lt_of_lt_right {a b c : α} [DecidablePred (· < c)] (h : b < c) :
    {x ∈ Ioc a b | x < c} = Ioc a b :=
  filter_true_of_mem fun _ hx => lt_of_le_of_lt (mem_Ioc.1 hx).2 h


theorem Iic_filter_lt_of_lt_right {α} [Preorder α] [LocallyFiniteOrderBot α] {a c : α}
    [DecidablePred (· < c)] (h : a < c) : {x ∈ Iic a | x < c} = Iic a :=
  filter_true_of_mem fun _ hx => lt_of_le_of_lt (mem_Iic.1 hx) h


theorem filter_lt_lt_eq_Ioo [DecidablePred fun j => a < j ∧ j < b] :
                                                     /-
                                                       α : Type u_2
                                                       a b : α
                                                       inst✝³ : Preorder α
                                                       inst✝² : LocallyFiniteOrder α
                                                       inst✝¹ : Fintype α
                                                       inst✝ : DecidablePred fun j => And (LT.lt a j) (LT.lt j b)
                                                       ⊢ Eq (Finset.filter (fun j => And (LT.lt a j) (LT.lt j b)) Finset.univ) (Finse …
                                                     -/
    ({j | a < j ∧ j < b} : Finset _) = Ioo a b := by ext; simp
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem filter_lt_le_eq_Ioc [DecidablePred fun j => a < j ∧ j ≤ b] :
                                                     /-
                                                       α : Type u_2
                                                       a b : α
                                                       inst✝³ : Preorder α
                                                       inst✝² : LocallyFiniteOrder α
                                                       inst✝¹ : Fintype α
                                                       inst✝ : DecidablePred fun j => And (LT.lt a j) (LE.le j b)
                                                       ⊢ Eq (Finset.filter (fun j => And (LT.lt a j) (LE.le j b)) Finset.univ) (Finse …
                                                     -/
    ({j | a < j ∧ j ≤ b} : Finset _) = Ioc a b := by ext; simp
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem filter_le_lt_eq_Ico [DecidablePred fun j => a ≤ j ∧ j < b] :
                                                     /-
                                                       α : Type u_2
                                                       a b : α
                                                       inst✝³ : Preorder α
                                                       inst✝² : LocallyFiniteOrder α
                                                       inst✝¹ : Fintype α
                                                       inst✝ : DecidablePred fun j => And (LE.le a j) (LT.lt j b)
                                                       ⊢ Eq (Finset.filter (fun j => And (LE.le a j) (LT.lt j b)) Finset.univ) (Finse …
                                                     -/
    ({j | a ≤ j ∧ j < b} : Finset _) = Ico a b := by ext; simp
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem filter_le_le_eq_Icc [DecidablePred fun j => a ≤ j ∧ j ≤ b] :
                                                     /-
                                                       α : Type u_2
                                                       a b : α
                                                       inst✝³ : Preorder α
                                                       inst✝² : LocallyFiniteOrder α
                                                       inst✝¹ : Fintype α
                                                       inst✝ : DecidablePred fun j => And (LE.le a j) (LE.le j b)
                                                       ⊢ Eq (Finset.filter (fun j => And (LE.le a j) (LE.le j b)) Finset.univ) (Finse …
                                                     -/
    ({j | a ≤ j ∧ j ≤ b} : Finset _) = Icc a b := by ext; simp
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem Ioi_eq_empty : Ioi a = ∅ ↔ IsMax a := by
  /-
    α : Type u_2
    a : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderTop α
    ⊢ Iff (Eq (Finset.Ioi a) EmptyCollection.emptyCollection) (IsMax a)
  -/
  rw [← coe_eq_empty, coe_Ioi, Set.Ioi_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioi_top [OrderTop α] : Ioi (⊤ : α) = ∅ := Ioi_eq_empty.mpr isMax_top


@[simp]
theorem Ici_bot [OrderBot α] [Fintype α] : Ici (⊥ : α) = univ := by
  /-
    α : Type u_2
    inst✝³ : Preorder α
    inst✝² : LocallyFiniteOrderTop α
    inst✝¹ : OrderBot α
    inst✝ : Fintype α
    ⊢ Eq (Finset.Ici Bot.bot) Finset.univ
  -/
  ext a; simp only [mem_Ici, bot_le, mem_univ]
         /-
           🎉 no goals
         -/


@[simp, aesop safe apply (rule_sets := [finsetNonempty])]
lemma nonempty_Ici : (Ici a).Nonempty := ⟨a, mem_Ici.2 le_rfl⟩

@[simp]
                                                        /-
                                                          α : Type u_2
                                                          a : α
                                                          inst✝¹ : Preorder α
                                                          inst✝ : LocallyFiniteOrderTop α
                                                          ⊢ Iff (Finset.Ioi a).Nonempty (Not (IsMax a))
                                                        -/
lemma nonempty_Ioi : (Ioi a).Nonempty ↔ ¬ IsMax a := by simp [Finset.Nonempty]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
alias ⟨_, Aesop.nonempty_Ioi_of_not_isMax⟩ := nonempty_Ioi


theorem Ici_subset_Ici : Ici a ⊆ Ici b ↔ b ≤ a := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderTop α
    ⊢ Iff (HasSubset.Subset (Finset.Ici a) (Finset.Ici b)) (LE.le b a)
  -/
  simpa [← coe_subset] using Set.Ici_subset_Ici
  /-
    🎉 no goals
  -/


theorem Ioi_subset_Ioi (h : a ≤ b) : Ioi b ⊆ Ioi a := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderTop α
    h : LE.le a b
    ⊢ HasSubset.Subset (Finset.Ioi b) (Finset.Ioi a)
  -/
  simpa [← coe_subset] using Set.Ioi_subset_Ioi h
  /-
    🎉 no goals
  -/


theorem Icc_subset_Ici_self : Icc a b ⊆ Ici a := by
  /-
    α : Type u_2
    a b : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrderTop α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Icc a b) (Finset.Ici a)
  -/
  simpa [← coe_subset] using Set.Icc_subset_Ici_self
  /-
    🎉 no goals
  -/


theorem Ico_subset_Ici_self : Ico a b ⊆ Ici a := by
  /-
    α : Type u_2
    a b : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrderTop α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Ico a b) (Finset.Ici a)
  -/
  simpa [← coe_subset] using Set.Ico_subset_Ici_self
  /-
    🎉 no goals
  -/


theorem Ioc_subset_Ioi_self : Ioc a b ⊆ Ioi a := by
  /-
    α : Type u_2
    a b : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrderTop α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Ioc a b) (Finset.Ioi a)
  -/
  simpa [← coe_subset] using Set.Ioc_subset_Ioi_self
  /-
    🎉 no goals
  -/


theorem Ioo_subset_Ioi_self : Ioo a b ⊆ Ioi a := by
  /-
    α : Type u_2
    a b : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrderTop α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Ioo a b) (Finset.Ioi a)
  -/
  simpa [← coe_subset] using Set.Ioo_subset_Ioi_self
  /-
    🎉 no goals
  -/


theorem Ioc_subset_Ici_self : Ioc a b ⊆ Ici a :=
  Ioc_subset_Icc_self.trans Icc_subset_Ici_self


theorem Ioo_subset_Ici_self : Ioo a b ⊆ Ici a :=
  Ioo_subset_Ico_self.trans Ico_subset_Ici_self


@[simp]
theorem Iio_eq_empty : Iio a = ∅ ↔ IsMin a := Ioi_eq_empty (α := αᵒᵈ)


@[simp]
theorem Iio_bot [OrderBot α] : Iio (⊥ : α) = ∅ := Iio_eq_empty.mpr isMin_bot


@[simp]
theorem Iic_top [OrderTop α] [Fintype α] : Iic (⊤ : α) = univ := by
  /-
    α : Type u_2
    inst✝³ : Preorder α
    inst✝² : LocallyFiniteOrderBot α
    inst✝¹ : OrderTop α
    inst✝ : Fintype α
    ⊢ Eq (Finset.Iic Top.top) Finset.univ
  -/
  ext a; simp only [mem_Iic, le_top, mem_univ]
         /-
           🎉 no goals
         -/


@[simp, aesop safe apply (rule_sets := [finsetNonempty])]
lemma nonempty_Iic : (Iic a).Nonempty := ⟨a, mem_Iic.2 le_rfl⟩

@[simp]
                                                        /-
                                                          α : Type u_2
                                                          a : α
                                                          inst✝¹ : Preorder α
                                                          inst✝ : LocallyFiniteOrderBot α
                                                          ⊢ Iff (Finset.Iio a).Nonempty (Not (IsMin a))
                                                        -/
lemma nonempty_Iio : (Iio a).Nonempty ↔ ¬ IsMin a := by simp [Finset.Nonempty]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
alias ⟨_, Aesop.nonempty_Iio_of_not_isMin⟩ := nonempty_Iio


theorem Iic_subset_Iic : Iic a ⊆ Iic b ↔ a ≤ b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderBot α
    ⊢ Iff (HasSubset.Subset (Finset.Iic a) (Finset.Iic b)) (LE.le a b)
  -/
  simpa [← coe_subset] using Set.Iic_subset_Iic
  /-
    🎉 no goals
  -/


theorem Iio_subset_Iio (h : a ≤ b) : Iio a ⊆ Iio b := by
  /-
    α : Type u_2
    a b : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderBot α
    h : LE.le a b
    ⊢ HasSubset.Subset (Finset.Iio a) (Finset.Iio b)
  -/
  simpa [← coe_subset] using Set.Iio_subset_Iio h
  /-
    🎉 no goals
  -/


theorem Icc_subset_Iic_self : Icc a b ⊆ Iic b := by
  /-
    α : Type u_2
    a b : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Icc a b) (Finset.Iic b)
  -/
  simpa [← coe_subset] using Set.Icc_subset_Iic_self
  /-
    🎉 no goals
  -/


theorem Ioc_subset_Iic_self : Ioc a b ⊆ Iic b := by
  /-
    α : Type u_2
    a b : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Ioc a b) (Finset.Iic b)
  -/
  simpa [← coe_subset] using Set.Ioc_subset_Iic_self
  /-
    🎉 no goals
  -/


theorem Ico_subset_Iio_self : Ico a b ⊆ Iio b := by
  /-
    α : Type u_2
    a b : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Ico a b) (Finset.Iio b)
  -/
  simpa [← coe_subset] using Set.Ico_subset_Iio_self
  /-
    🎉 no goals
  -/


theorem Ioo_subset_Iio_self : Ioo a b ⊆ Iio b := by
  /-
    α : Type u_2
    a b : α
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : LocallyFiniteOrder α
    ⊢ HasSubset.Subset (Finset.Ioo a b) (Finset.Iio b)
  -/
  simpa [← coe_subset] using Set.Ioo_subset_Iio_self
  /-
    🎉 no goals
  -/


theorem Ico_subset_Iic_self : Ico a b ⊆ Iic b :=
  Ico_subset_Icc_self.trans Icc_subset_Iic_self


theorem Ioo_subset_Iic_self : Ioo a b ⊆ Iic b :=
  Ioo_subset_Ioc_self.trans Ioc_subset_Iic_self


theorem Ioi_subset_Ici_self : Ioi a ⊆ Ici a := by
  /-
    α : Type u_2
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderTop α
    a : α
    ⊢ HasSubset.Subset (Finset.Ioi a) (Finset.Ici a)
  -/
  simpa [← coe_subset] using Set.Ioi_subset_Ici_self
  /-
    🎉 no goals
  -/


theorem _root_.BddBelow.finite {s : Set α} (hs : BddBelow s) : s.Finite :=
  let ⟨a, ha⟩ := hs
  (Ici a).finite_toSet.subset fun _ hx => mem_Ici.2 <| ha hx


theorem _root_.Set.Infinite.not_bddBelow {s : Set α} : s.Infinite → ¬BddBelow s :=
  mt BddBelow.finite


                                                                                          /-
                                                                                            α : Type u_2
                                                                                            inst✝³ : Preorder α
                                                                                            inst✝² : LocallyFiniteOrderTop α
                                                                                            a : α
                                                                                            inst✝¹ : Fintype α
                                                                                            inst✝ : DecidablePred fun x => LT.lt a x
                                                                                            ⊢ Eq (Finset.filter (fun x => LT.lt a x) Finset.univ) (Finset.Ioi a)
                                                                                          -/
theorem filter_lt_eq_Ioi [DecidablePred (a < ·)] : ({x | a < x} : Finset _) = Ioi a := by ext; simp
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/

                                                                                          /-
                                                                                            α : Type u_2
                                                                                            inst✝³ : Preorder α
                                                                                            inst✝² : LocallyFiniteOrderTop α
                                                                                            a : α
                                                                                            inst✝¹ : Fintype α
                                                                                            inst✝ : DecidablePred fun x => LE.le a x
                                                                                            ⊢ Eq (Finset.filter (fun x => LE.le a x) Finset.univ) (Finset.Ici a)
                                                                                          -/
theorem filter_le_eq_Ici [DecidablePred (a ≤ ·)] : ({x | a ≤ x} : Finset _) = Ici a := by ext; simp
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


theorem Iio_subset_Iic_self : Iio a ⊆ Iic a := by
  /-
    α : Type u_2
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderBot α
    a : α
    ⊢ HasSubset.Subset (Finset.Iio a) (Finset.Iic a)
  -/
  simpa [← coe_subset] using Set.Iio_subset_Iic_self
  /-
    🎉 no goals
  -/


theorem _root_.BddAbove.finite {s : Set α} (hs : BddAbove s) : s.Finite :=
  hs.dual.finite


theorem _root_.Set.Infinite.not_bddAbove {s : Set α} : s.Infinite → ¬BddAbove s :=
  mt BddAbove.finite


                                                                                          /-
                                                                                            α : Type u_2
                                                                                            inst✝³ : Preorder α
                                                                                            inst✝² : LocallyFiniteOrderBot α
                                                                                            a : α
                                                                                            inst✝¹ : Fintype α
                                                                                            inst✝ : DecidablePred fun x => LT.lt x a
                                                                                            ⊢ Eq (Finset.filter (fun x => LT.lt x a) Finset.univ) (Finset.Iio a)
                                                                                          -/
theorem filter_gt_eq_Iio [DecidablePred (· < a)] : ({x | x < a} : Finset _) = Iio a := by ext; simp
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/

                                                                                          /-
                                                                                            α : Type u_2
                                                                                            inst✝³ : Preorder α
                                                                                            inst✝² : LocallyFiniteOrderBot α
                                                                                            a : α
                                                                                            inst✝¹ : Fintype α
                                                                                            inst✝ : DecidablePred fun x => LE.le x a
                                                                                            ⊢ Eq (Finset.filter (fun x => LE.le x a) Finset.univ) (Finset.Iic a)
                                                                                          -/
theorem filter_ge_eq_Iic [DecidablePred (· ≤ a)] : ({x | x ≤ a} : Finset _) = Iic a := by ext; simp
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


@[simp]
theorem Icc_bot [OrderBot α] : Icc (⊥ : α) a = Iic a := rfl


@[simp]
theorem Icc_top [OrderTop α] : Icc a (⊤ : α) = Ici a := rfl


@[simp]
theorem Ico_bot [OrderBot α] : Ico (⊥ : α) a = Iio a := rfl


@[simp]
theorem Ioc_top [OrderTop α] : Ioc a (⊤ : α) = Ioi a := rfl


theorem Icc_bot_top [BoundedOrder α] [Fintype α] : Icc (⊥ : α) (⊤ : α) = univ := by
  /-
    α : Type u_2
    inst✝³ : Preorder α
    inst✝² : LocallyFiniteOrder α
    inst✝¹ : BoundedOrder α
    inst✝ : Fintype α
    ⊢ Eq (Finset.Icc Bot.bot Top.top) Finset.univ
  -/
  rw [Icc_bot, Iic_top]
  /-
    🎉 no goals
  -/


theorem disjoint_Ioi_Iio (a : α) : Disjoint (Ioi a) (Iio a) :=
  disjoint_left.2 fun _ hab hba => (mem_Ioi.1 hab).not_lt <| mem_Iio.1 hba


@[simp]
                                               /-
                                                 α : Type u_2
                                                 inst✝¹ : PartialOrder α
                                                 inst✝ : LocallyFiniteOrder α
                                                 a : α
                                                 ⊢ Eq (Finset.Icc a a) (Singleton.singleton a)
                                               -/
theorem Icc_self (a : α) : Icc a a = {a} := by rw [← coe_eq_singleton, coe_Icc, Set.Icc_self]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem Icc_eq_singleton_iff : Icc a b = {c} ↔ a = c ∧ b = c := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Iff (Eq (Finset.Icc a b) (Singleton.singleton c)) (And (Eq a c) (Eq b c))
  -/
  rw [← coe_eq_singleton, coe_Icc, Set.Icc_eq_singleton_iff]
  /-
    🎉 no goals
  -/


theorem Ico_disjoint_Ico_consecutive (a b c : α) : Disjoint (Ico a b) (Ico b c) :=
  disjoint_left.2 fun _ hab hbc => (mem_Ico.mp hab).2.not_le (mem_Ico.mp hbc).1


@[simp]
theorem Ici_top [OrderTop α] : Ici (⊤ : α) = {⊤} := Icc_eq_singleton_iff.2 ⟨rfl, rfl⟩


@[simp]
theorem Iic_bot [OrderBot α] : Iic (⊥ : α) = {⊥} := Icc_eq_singleton_iff.2 ⟨rfl, rfl⟩


@[simp]
                                                                     /-
                                                                       α : Type u_2
                                                                       inst✝² : PartialOrder α
                                                                       inst✝¹ : LocallyFiniteOrder α
                                                                       inst✝ : DecidableEq α
                                                                       a b : α
                                                                       ⊢ Eq ((Finset.Icc a b).erase a) (Finset.Ioc a b)
                                                                     -/
theorem Icc_erase_left (a b : α) : (Icc a b).erase a = Ioc a b := by simp [← coe_inj]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
                                                                      /-
                                                                        α : Type u_2
                                                                        inst✝² : PartialOrder α
                                                                        inst✝¹ : LocallyFiniteOrder α
                                                                        inst✝ : DecidableEq α
                                                                        a b : α
                                                                        ⊢ Eq ((Finset.Icc a b).erase b) (Finset.Ico a b)
                                                                      -/
theorem Icc_erase_right (a b : α) : (Icc a b).erase b = Ico a b := by simp [← coe_inj]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
                                                                     /-
                                                                       α : Type u_2
                                                                       inst✝² : PartialOrder α
                                                                       inst✝¹ : LocallyFiniteOrder α
                                                                       inst✝ : DecidableEq α
                                                                       a b : α
                                                                       ⊢ Eq ((Finset.Ico a b).erase a) (Finset.Ioo a b)
                                                                     -/
theorem Ico_erase_left (a b : α) : (Ico a b).erase a = Ioo a b := by simp [← coe_inj]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
                                                                      /-
                                                                        α : Type u_2
                                                                        inst✝² : PartialOrder α
                                                                        inst✝¹ : LocallyFiniteOrder α
                                                                        inst✝ : DecidableEq α
                                                                        a b : α
                                                                        ⊢ Eq ((Finset.Ioc a b).erase b) (Finset.Ioo a b)
                                                                      -/
theorem Ioc_erase_right (a b : α) : (Ioc a b).erase b = Ioo a b := by simp [← coe_inj]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
                                                                   /-
                                                                     α : Type u_2
                                                                     inst✝² : PartialOrder α
                                                                     inst✝¹ : LocallyFiniteOrder α
                                                                     inst✝ : DecidableEq α
                                                                     a b : α
                                                                     ⊢ Eq (SDiff.sdiff (Finset.Icc a b) (Insert.insert a (Singleton.singleton b)))  …
                                                                   -/
theorem Icc_diff_both (a b : α) : Icc a b \ {a, b} = Ioo a b := by simp [← coe_inj]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem Ico_insert_right (h : a ≤ b) : insert b (Ico a b) = Icc a b := by
  /-
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidableEq α
    h : LE.le a b
    ⊢ Eq (Insert.insert b (Finset.Ico a b)) (Finset.Icc a b)
  -/
  rw [← coe_inj, coe_insert, coe_Icc, coe_Ico, Set.insert_eq, Set.union_comm, Set.Ico_union_right h]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioc_insert_left (h : a ≤ b) : insert a (Ioc a b) = Icc a b := by
  /-
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidableEq α
    h : LE.le a b
    ⊢ Eq (Insert.insert a (Finset.Ioc a b)) (Finset.Icc a b)
  -/
  rw [← coe_inj, coe_insert, coe_Ioc, coe_Icc, Set.insert_eq, Set.union_comm, Set.Ioc_union_left h]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioo_insert_left (h : a < b) : insert a (Ioo a b) = Ico a b := by
  /-
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidableEq α
    h : LT.lt a b
    ⊢ Eq (Insert.insert a (Finset.Ioo a b)) (Finset.Ico a b)
  -/
  rw [← coe_inj, coe_insert, coe_Ioo, coe_Ico, Set.insert_eq, Set.union_comm, Set.Ioo_union_left h]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioo_insert_right (h : a < b) : insert b (Ioo a b) = Ioc a b := by
  /-
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidableEq α
    h : LT.lt a b
    ⊢ Eq (Insert.insert b (Finset.Ioo a b)) (Finset.Ioc a b)
  -/
  rw [← coe_inj, coe_insert, coe_Ioo, coe_Ioc, Set.insert_eq, Set.union_comm, Set.Ioo_union_right h]
  /-
    🎉 no goals
  -/


@[simp]
                                                                      /-
                                                                        α : Type u_2
                                                                        inst✝² : PartialOrder α
                                                                        inst✝¹ : LocallyFiniteOrder α
                                                                        a b : α
                                                                        inst✝ : DecidableEq α
                                                                        h : LE.le a b
                                                                        ⊢ Eq (SDiff.sdiff (Finset.Icc a b) (Finset.Ico a b)) (Singleton.singleton b)
                                                                      -/
theorem Icc_diff_Ico_self (h : a ≤ b) : Icc a b \ Ico a b = {b} := by simp [← coe_inj, h]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
                                                                      /-
                                                                        α : Type u_2
                                                                        inst✝² : PartialOrder α
                                                                        inst✝¹ : LocallyFiniteOrder α
                                                                        a b : α
                                                                        inst✝ : DecidableEq α
                                                                        h : LE.le a b
                                                                        ⊢ Eq (SDiff.sdiff (Finset.Icc a b) (Finset.Ioc a b)) (Singleton.singleton a)
                                                                      -/
theorem Icc_diff_Ioc_self (h : a ≤ b) : Icc a b \ Ioc a b = {a} := by simp [← coe_inj, h]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
                                                                         /-
                                                                           α : Type u_2
                                                                           inst✝² : PartialOrder α
                                                                           inst✝¹ : LocallyFiniteOrder α
                                                                           a b : α
                                                                           inst✝ : DecidableEq α
                                                                           h : LE.le a b
                                                                           ⊢ Eq (SDiff.sdiff (Finset.Icc a b) (Finset.Ioo a b)) (Insert.insert a (Singlet …
                                                                         -/
theorem Icc_diff_Ioo_self (h : a ≤ b) : Icc a b \ Ioo a b = {a, b} := by simp [← coe_inj, h]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
                                                                      /-
                                                                        α : Type u_2
                                                                        inst✝² : PartialOrder α
                                                                        inst✝¹ : LocallyFiniteOrder α
                                                                        a b : α
                                                                        inst✝ : DecidableEq α
                                                                        h : LT.lt a b
                                                                        ⊢ Eq (SDiff.sdiff (Finset.Ico a b) (Finset.Ioo a b)) (Singleton.singleton a)
                                                                      -/
theorem Ico_diff_Ioo_self (h : a < b) : Ico a b \ Ioo a b = {a} := by simp [← coe_inj, h]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
                                                                      /-
                                                                        α : Type u_2
                                                                        inst✝² : PartialOrder α
                                                                        inst✝¹ : LocallyFiniteOrder α
                                                                        a b : α
                                                                        inst✝ : DecidableEq α
                                                                        h : LT.lt a b
                                                                        ⊢ Eq (SDiff.sdiff (Finset.Ioc a b) (Finset.Ioo a b)) (Singleton.singleton b)
                                                                      -/
theorem Ioc_diff_Ioo_self (h : a < b) : Ioc a b \ Ioo a b = {b} := by simp [← coe_inj, h]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem Ico_inter_Ico_consecutive (a b c : α) : Ico a b ∩ Ico b c = ∅ :=
  (Ico_disjoint_Ico_consecutive a b c).eq_bot


/-- `Finset.cons` version of `Finset.Ico_insert_right`. -/
theorem Icc_eq_cons_Ico (h : a ≤ b) : Icc a b = (Ico a b).cons b right_not_mem_Ico := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrder α
    a b : α
    h : LE.le a b
    ⊢ Eq (Finset.Icc a b) (Finset.cons b (Finset.Ico a b) ⋯)
  -/
  classical rw [cons_eq_insert, Ico_insert_right h]
  /-
    🎉 no goals
  -/


/-- `Finset.cons` version of `Finset.Ioc_insert_left`. -/
theorem Icc_eq_cons_Ioc (h : a ≤ b) : Icc a b = (Ioc a b).cons a left_not_mem_Ioc := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrder α
    a b : α
    h : LE.le a b
    ⊢ Eq (Finset.Icc a b) (Finset.cons a (Finset.Ioc a b) ⋯)
  -/
  classical rw [cons_eq_insert, Ioc_insert_left h]
  /-
    🎉 no goals
  -/


/-- `Finset.cons` version of `Finset.Ioo_insert_right`. -/
theorem Ioc_eq_cons_Ioo (h : a < b) : Ioc a b = (Ioo a b).cons b right_not_mem_Ioo := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrder α
    a b : α
    h : LT.lt a b
    ⊢ Eq (Finset.Ioc a b) (Finset.cons b (Finset.Ioo a b) ⋯)
  -/
  classical rw [cons_eq_insert, Ioo_insert_right h]
  /-
    🎉 no goals
  -/


/-- `Finset.cons` version of `Finset.Ioo_insert_left`. -/
theorem Ico_eq_cons_Ioo (h : a < b) : Ico a b = (Ioo a b).cons a left_not_mem_Ioo := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrder α
    a b : α
    h : LT.lt a b
    ⊢ Eq (Finset.Ico a b) (Finset.cons a (Finset.Ioo a b) ⋯)
  -/
  classical rw [cons_eq_insert, Ioo_insert_left h]
  /-
    🎉 no goals
  -/


theorem Ico_filter_le_left {a b : α} [DecidablePred (· ≤ a)] (hab : a < b) :
    {x ∈ Ico a b | x ≤ a} = {a} := by
  /-
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidablePred fun x => LE.le x a
    hab : LT.lt a b
    ⊢ Eq (Finset.filter (fun x => LE.le x a) (Finset.Ico a b)) (Singleton.singleto …
  -/
  ext x
  /-
    case h
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidablePred fun x => LE.le x a
    hab : LT.lt a b
    x : α
    ⊢ Iff (Membership.mem (Finset.filter (fun x => LE.le x a) (Finset.Ico a b)) x) …
  -/
  rw [mem_filter, mem_Ico, mem_singleton, and_right_comm, ← le_antisymm_iff, eq_comm]
  /-
    case h
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidablePred fun x => LE.le x a
    hab : LT.lt a b
    x : α
    ⊢ Iff (And (Eq x a) (LT.lt x b)) (Eq x a)
  -/
  exact and_iff_left_of_imp fun h => h.le.trans_lt hab
  /-
    🎉 no goals
  -/


theorem card_Ico_eq_card_Icc_sub_one (a b : α) : #(Ico a b) = #(Icc a b) - 1 := by
  classical
    by_cases h : a ≤ b
    · rw [Icc_eq_cons_Ico h, card_cons]
      exact (Nat.add_sub_cancel _ _).symm
    · rw [Ico_eq_empty fun h' => h h'.le, Icc_eq_empty h, card_empty, Nat.zero_sub]


theorem card_Ioc_eq_card_Icc_sub_one (a b : α) : #(Ioc a b) = #(Icc a b) - 1 :=
  @card_Ico_eq_card_Icc_sub_one αᵒᵈ _ _ _ _


theorem card_Ioo_eq_card_Ico_sub_one (a b : α) : #(Ioo a b) = #(Ico a b) - 1 := by
  classical
    by_cases h : a < b
    · rw [Ico_eq_cons_Ioo h, card_cons]
      exact (Nat.add_sub_cancel _ _).symm
    · rw [Ioo_eq_empty h, Ico_eq_empty h, card_empty, Nat.zero_sub]


theorem card_Ioo_eq_card_Ioc_sub_one (a b : α) : #(Ioo a b) = #(Ioc a b) - 1 :=
  @card_Ioo_eq_card_Ico_sub_one αᵒᵈ _ _ _ _


theorem card_Ioo_eq_card_Icc_sub_two (a b : α) : #(Ioo a b) = #(Icc a b) - 2 := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrder α
    a b : α
    ⊢ Eq (Finset.Ioo a b).card (HSub.hSub (Finset.Icc a b).card 2)
  -/
  rw [card_Ioo_eq_card_Ico_sub_one, card_Ico_eq_card_Icc_sub_one]
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrder α
    a b : α
    ⊢ Eq (HSub.hSub (HSub.hSub (Finset.Icc a b).card 1) 1) (HSub.hSub (Finset.Icc  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem Ici_erase [DecidableEq α] (a : α) : (Ici a).erase a = Ioi a := by
  /-
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrderTop α
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq ((Finset.Ici a).erase a) (Finset.Ioi a)
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrderTop α
    inst✝ : DecidableEq α
    a a✝ : α
    ⊢ Iff (Membership.mem ((Finset.Ici a).erase a) a✝) (Membership.mem (Finset.Ioi …
  -/
  simp_rw [Finset.mem_erase, mem_Ici, mem_Ioi, lt_iff_le_and_ne, and_comm, ne_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioi_insert [DecidableEq α] (a : α) : insert a (Ioi a) = Ici a := by
  /-
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrderTop α
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (Insert.insert a (Finset.Ioi a)) (Finset.Ici a)
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrderTop α
    inst✝ : DecidableEq α
    a a✝ : α
    ⊢ Iff (Membership.mem (Insert.insert a (Finset.Ioi a)) a✝) (Membership.mem (Fi …
  -/
  simp_rw [Finset.mem_insert, mem_Ici, mem_Ioi, le_iff_lt_or_eq, or_comm, eq_comm]
  /-
    🎉 no goals
  -/


theorem not_mem_Ioi_self {b : α} : b ∉ Ioi b := fun h => lt_irrefl _ (mem_Ioi.1 h)

-- Purposefully written the other way around

/-- `Finset.cons` version of `Finset.Ioi_insert`. -/
theorem Ici_eq_cons_Ioi (a : α) : Ici a = (Ioi a).cons a not_mem_Ioi_self := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrderTop α
    a : α
    ⊢ Eq (Finset.Ici a) (Finset.cons a (Finset.Ioi a) ⋯)
  -/
  classical rw [cons_eq_insert, Ioi_insert]
  /-
    🎉 no goals
  -/


theorem card_Ioi_eq_card_Ici_sub_one (a : α) : #(Ioi a) = #(Ici a) - 1 := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrderTop α
    a : α
    ⊢ Eq (Finset.Ioi a).card (HSub.hSub (Finset.Ici a).card 1)
  -/
  rw [Ici_eq_cons_Ioi, card_cons, Nat.add_sub_cancel_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem Iic_erase [DecidableEq α] (b : α) : (Iic b).erase b = Iio b := by
  /-
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : DecidableEq α
    b : α
    ⊢ Eq ((Finset.Iic b).erase b) (Finset.Iio b)
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : DecidableEq α
    b a✝ : α
    ⊢ Iff (Membership.mem ((Finset.Iic b).erase b) a✝) (Membership.mem (Finset.Iio …
  -/
  simp_rw [Finset.mem_erase, mem_Iic, mem_Iio, lt_iff_le_and_ne, and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem Iio_insert [DecidableEq α] (b : α) : insert b (Iio b) = Iic b := by
  /-
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : DecidableEq α
    b : α
    ⊢ Eq (Insert.insert b (Finset.Iio b)) (Finset.Iic b)
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : DecidableEq α
    b a✝ : α
    ⊢ Iff (Membership.mem (Insert.insert b (Finset.Iio b)) a✝) (Membership.mem (Fi …
  -/
  simp_rw [Finset.mem_insert, mem_Iic, mem_Iio, le_iff_lt_or_eq, or_comm]
  /-
    🎉 no goals
  -/


theorem not_mem_Iio_self {b : α} : b ∉ Iio b := fun h => lt_irrefl _ (mem_Iio.1 h)

-- Purposefully written the other way around

/-- `Finset.cons` version of `Finset.Iio_insert`. -/
theorem Iic_eq_cons_Iio (b : α) : Iic b = (Iio b).cons b not_mem_Iio_self := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrderBot α
    b : α
    ⊢ Eq (Finset.Iic b) (Finset.cons b (Finset.Iio b) ⋯)
  -/
  classical rw [cons_eq_insert, Iio_insert]
  /-
    🎉 no goals
  -/


theorem card_Iio_eq_card_Iic_sub_one (a : α) : #(Iio a) = #(Iic a) - 1 := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrderBot α
    a : α
    ⊢ Eq (Finset.Iio a).card (HSub.hSub (Finset.Iic a).card 1)
  -/
  rw [Iic_eq_cons_Iio, card_cons, Nat.add_sub_cancel_right]
  /-
    🎉 no goals
  -/


lemma sup'_Iic (a : α) : (Iic a).sup' nonempty_Iic id = a :=
  le_antisymm (sup'_le _ _ fun _ ↦ mem_Iic.1) <| le_sup' (f := id) <| mem_Iic.2 <| le_refl a


@[simp] lemma sup_Iic [OrderBot α] (a : α) : (Iic a).sup id = a :=
  le_antisymm (Finset.sup_le fun _ ↦ mem_Iic.1) <| le_sup (f := id) <| mem_Iic.2 <| le_refl a


lemma inf'_Ici (a : α) : (Ici a).inf' nonempty_Ici id = a :=
  ge_antisymm (le_inf' _ _ fun _ ↦ mem_Ici.1) <| inf'_le (f := id) <| mem_Ici.2 <| le_refl a


@[simp] lemma inf_Ici [OrderTop α] (a : α) : (Ici a).inf id = a :=
  le_antisymm (inf_le (f := id) <| mem_Ici.2 <| le_refl a) <| Finset.le_inf fun _ ↦ mem_Ici.1


theorem Ico_subset_Ico_iff {a₁ b₁ a₂ b₂ : α} (h : a₁ < b₁) :
    Ico a₁ b₁ ⊆ Ico a₂ b₂ ↔ a₂ ≤ a₁ ∧ b₁ ≤ b₂ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a₁ b₁ a₂ b₂ : α
    h : LT.lt a₁ b₁
    ⊢ Iff (HasSubset.Subset (Finset.Ico a₁ b₁) (Finset.Ico a₂ b₂)) (And (LE.le a₂  …
  -/
  rw [← coe_subset, coe_Ico, coe_Ico, Set.Ico_subset_Ico_iff h]
  /-
    🎉 no goals
  -/


theorem Ico_union_Ico_eq_Ico {a b c : α} (hab : a ≤ b) (hbc : b ≤ c) :
    Ico a b ∪ Ico b c = Ico a c := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    hab : LE.le a b
    hbc : LE.le b c
    ⊢ Eq (Union.union (Finset.Ico a b) (Finset.Ico b c)) (Finset.Ico a c)
  -/
  rw [← coe_inj, coe_union, coe_Ico, coe_Ico, coe_Ico, Set.Ico_union_Ico_eq_Ico hab hbc]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioc_union_Ioc_eq_Ioc {a b c : α} (h₁ : a ≤ b) (h₂ : b ≤ c) :
    Ioc a b ∪ Ioc b c = Ioc a c := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    h₁ : LE.le a b
    h₂ : LE.le b c
    ⊢ Eq (Union.union (Finset.Ioc a b) (Finset.Ioc b c)) (Finset.Ioc a c)
  -/
  rw [← coe_inj, coe_union, coe_Ioc, coe_Ioc, coe_Ioc, Set.Ioc_union_Ioc_eq_Ioc h₁ h₂]
  /-
    🎉 no goals
  -/


theorem Ico_subset_Ico_union_Ico {a b c : α} : Ico a c ⊆ Ico a b ∪ Ico b c := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ HasSubset.Subset (Finset.Ico a c) (Union.union (Finset.Ico a b) (Finset.Ico  …
  -/
  rw [← coe_subset, coe_union, coe_Ico, coe_Ico, coe_Ico]
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ HasSubset.Subset (Set.Ico a c) (Union.union (Set.Ico a b) (Set.Ico b c))
  -/
  exact Set.Ico_subset_Ico_union_Ico
  /-
    🎉 no goals
  -/


theorem Ico_union_Ico' {a b c d : α} (hcb : c ≤ b) (had : a ≤ d) :
    Ico a b ∪ Ico c d = Ico (min a c) (max b d) := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c d : α
    hcb : LE.le c b
    had : LE.le a d
    ⊢ Eq (Union.union (Finset.Ico a b) (Finset.Ico c d)) (Finset.Ico (Min.min a c) …
  -/
  rw [← coe_inj, coe_union, coe_Ico, coe_Ico, coe_Ico, Set.Ico_union_Ico' hcb had]
  /-
    🎉 no goals
  -/


theorem Ico_union_Ico {a b c d : α} (h₁ : min a b ≤ max c d) (h₂ : min c d ≤ max a b) :
    Ico a b ∪ Ico c d = Ico (min a c) (max b d) := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c d : α
    h₁ : LE.le (Min.min a b) (Max.max c d)
    h₂ : LE.le (Min.min c d) (Max.max a b)
    ⊢ Eq (Union.union (Finset.Ico a b) (Finset.Ico c d)) (Finset.Ico (Min.min a c) …
  -/
  rw [← coe_inj, coe_union, coe_Ico, coe_Ico, coe_Ico, Set.Ico_union_Ico h₁ h₂]
  /-
    🎉 no goals
  -/


theorem Ico_inter_Ico {a b c d : α} : Ico a b ∩ Ico c d = Ico (max a c) (min b d) := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c d : α
    ⊢ Eq (Inter.inter (Finset.Ico a b) (Finset.Ico c d)) (Finset.Ico (Max.max a c) …
  -/
  rw [← coe_inj, coe_inter, coe_Ico, coe_Ico, coe_Ico, Set.Ico_inter_Ico]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ico_filter_lt (a b c : α) : {x ∈ Ico a b | x < c} = Ico a (min b c) := by
  cases le_total b c with
  | inl h => rw [Ico_filter_lt_of_right_le h, min_eq_left h]
  | inr h => rw [Ico_filter_lt_of_le_right h, min_eq_right h]


@[simp]
theorem Ico_filter_le (a b c : α) : {x ∈ Ico a b | c ≤ x} = Ico (max a c) b := by
  cases le_total a c with
  | inl h => rw [Ico_filter_le_of_left_le h, max_eq_right h]
  | inr h => rw [Ico_filter_le_of_le_left h, max_eq_left h]


@[simp]
theorem Ioo_filter_lt (a b c : α) : {x ∈ Ioo a b | x < c} = Ioo a (min b c) := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Finset.filter (fun x => LT.lt x c) (Finset.Ioo a b)) (Finset.Ioo a (Min. …
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c a✝ : α
    ⊢ Iff (Membership.mem (Finset.filter (fun x => LT.lt x c) (Finset.Ioo a b)) a✝ …
  -/
  simp [and_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem Iio_filter_lt {α} [LinearOrder α] [LocallyFiniteOrderBot α] (a b : α) :
    {x ∈ Iio a | x < b} = Iio (min a b) := by
  /-
    α : Type u_3
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrderBot α
    a b : α
    ⊢ Eq (Finset.filter (fun x => LT.lt x b) (Finset.Iio a)) (Finset.Iio (Min.min  …
  -/
  ext
  /-
    case h
    α : Type u_3
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrderBot α
    a b a✝ : α
    ⊢ Iff (Membership.mem (Finset.filter (fun x => LT.lt x b) (Finset.Iio a)) a✝)  …
  -/
  simp [and_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ico_diff_Ico_left (a b c : α) : Ico a b \ Ico a c = Ico (max a c) b := by
  cases le_total a c with
  | inl h =>
    ext x
    rw [mem_sdiff, mem_Ico, mem_Ico, mem_Ico, max_eq_right h, and_right_comm, not_and, not_lt]
    exact and_congr_left' ⟨fun hx => hx.2 hx.1, fun hx => ⟨h.trans hx, fun _ => hx⟩⟩
  | inr h => rw [Ico_eq_empty_of_le h, sdiff_empty, max_eq_left h]


@[simp]
theorem Ico_diff_Ico_right (a b c : α) : Ico a b \ Ico c b = Ico a (min b c) := by
  cases le_total b c with
  | inl h => rw [Ico_eq_empty_of_le h, sdiff_empty, min_eq_left h]
  | inr h =>
    ext x
    rw [mem_sdiff, mem_Ico, mem_Ico, mem_Ico, min_eq_right h, and_assoc, not_and', not_le]
    exact and_congr_right' ⟨fun hx => hx.2 hx.1, fun hx => ⟨hx.trans_le h, fun _ => hx⟩⟩


theorem _root_.Set.Infinite.exists_gt (hs : s.Infinite) : ∀ a, ∃ b ∈ s, a < b :=
  not_bddAbove_iff.1 hs.not_bddAbove


theorem _root_.Set.infinite_iff_exists_gt [Nonempty α] : s.Infinite ↔ ∀ a, ∃ b ∈ s, a < b :=
  ⟨Set.Infinite.exists_gt, Set.infinite_of_forall_exists_gt⟩


theorem _root_.Set.Infinite.exists_lt (hs : s.Infinite) : ∀ a, ∃ b ∈ s, b < a :=
  not_bddBelow_iff.1 hs.not_bddBelow


theorem _root_.Set.infinite_iff_exists_lt [Nonempty α] : s.Infinite ↔ ∀ a, ∃ b ∈ s, b < a :=
  ⟨Set.Infinite.exists_lt, Set.infinite_of_forall_exists_lt⟩


theorem Ioi_disjUnion_Iio (a : α) :
    (Ioi a).disjUnion (Iio a) (disjoint_Ioi_Iio a) = ({a} : Finset α)ᶜ := by
  /-
    α : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : Fintype α
    inst✝¹ : LocallyFiniteOrderTop α
    inst✝ : LocallyFiniteOrderBot α
    a : α
    ⊢ Eq ((Finset.Ioi a).disjUnion (Finset.Iio a) ⋯) (HasCompl.compl (Singleton.si …
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : Fintype α
    inst✝¹ : LocallyFiniteOrderTop α
    inst✝ : LocallyFiniteOrderBot α
    a a✝ : α
    ⊢ Iff (Membership.mem ((Finset.Ioi a).disjUnion (Finset.Iio a) ⋯) a✝) (Members …
  -/
  simp [eq_comm]
  /-
    🎉 no goals
  -/


theorem uIcc_toDual (a b : α) : [[toDual a, toDual b]] = [[a, b]].map toDual.toEmbedding :=
  Icc_toDual (a ⊔ b) (a ⊓ b)


@[simp]
theorem uIcc_of_le (h : a ≤ b) : [[a, b]] = Icc a b := by
  /-
    α : Type u_2
    inst✝¹ : Lattice α
    inst✝ : LocallyFiniteOrder α
    a b : α
    h : LE.le a b
    ⊢ Eq (Finset.uIcc a b) (Finset.Icc a b)
  -/
  rw [uIcc, inf_eq_left.2 h, sup_eq_right.2 h]
  /-
    🎉 no goals
  -/


@[simp]
theorem uIcc_of_ge (h : b ≤ a) : [[a, b]] = Icc b a := by
  /-
    α : Type u_2
    inst✝¹ : Lattice α
    inst✝ : LocallyFiniteOrder α
    a b : α
    h : LE.le b a
    ⊢ Eq (Finset.uIcc a b) (Finset.Icc b a)
  -/
  rw [uIcc, inf_eq_right.2 h, sup_eq_left.2 h]
  /-
    🎉 no goals
  -/


theorem uIcc_comm (a b : α) : [[a, b]] = [[b, a]] := by
  /-
    α : Type u_2
    inst✝¹ : Lattice α
    inst✝ : LocallyFiniteOrder α
    a b : α
    ⊢ Eq (Finset.uIcc a b) (Finset.uIcc b a)
  -/
  rw [uIcc, uIcc, inf_comm, sup_comm]
  /-
    🎉 no goals
  -/


                                         /-
                                           α : Type u_2
                                           inst✝¹ : Lattice α
                                           inst✝ : LocallyFiniteOrder α
                                           a : α
                                           ⊢ Eq (Finset.uIcc a a) (Singleton.singleton a)
                                         -/
theorem uIcc_self : [[a, a]] = {a} := by simp [uIcc]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem nonempty_uIcc : Finset.Nonempty [[a, b]] :=
  nonempty_Icc.2 inf_le_sup


theorem Icc_subset_uIcc : Icc a b ⊆ [[a, b]] :=
  Icc_subset_Icc inf_le_left le_sup_right


theorem Icc_subset_uIcc' : Icc b a ⊆ [[a, b]] :=
  Icc_subset_Icc inf_le_right le_sup_left


theorem left_mem_uIcc : a ∈ [[a, b]] :=
  mem_Icc.2 ⟨inf_le_left, le_sup_left⟩


theorem right_mem_uIcc : b ∈ [[a, b]] :=
  mem_Icc.2 ⟨inf_le_right, le_sup_right⟩


theorem mem_uIcc_of_le (ha : a ≤ x) (hb : x ≤ b) : x ∈ [[a, b]] :=
  Icc_subset_uIcc <| mem_Icc.2 ⟨ha, hb⟩


theorem mem_uIcc_of_ge (hb : b ≤ x) (ha : x ≤ a) : x ∈ [[a, b]] :=
  Icc_subset_uIcc' <| mem_Icc.2 ⟨hb, ha⟩


theorem uIcc_subset_uIcc (h₁ : a₁ ∈ [[a₂, b₂]]) (h₂ : b₁ ∈ [[a₂, b₂]]) :
    [[a₁, b₁]] ⊆ [[a₂, b₂]] := by
  /-
    α : Type u_2
    inst✝¹ : Lattice α
    inst✝ : LocallyFiniteOrder α
    a₁ a₂ b₁ b₂ : α
    h₁ : Membership.mem (Finset.uIcc a₂ b₂) a₁
    h₂ : Membership.mem (Finset.uIcc a₂ b₂) b₁
    ⊢ HasSubset.Subset (Finset.uIcc a₁ b₁) (Finset.uIcc a₂ b₂)
  -/
  rw [mem_uIcc] at h₁ h₂
  /-
    α : Type u_2
    inst✝¹ : Lattice α
    inst✝ : LocallyFiniteOrder α
    a₁ a₂ b₁ b₂ : α
    h₁ : And (LE.le (Min.min a₂ b₂) a₁) (LE.le a₁ (Max.max a₂ b₂))
    h₂ : And (LE.le (Min.min a₂ b₂) b₁) (LE.le b₁ (Max.max a₂ b₂))
    ⊢ HasSubset.Subset (Finset.uIcc a₁ b₁) (Finset.uIcc a₂ b₂)
  -/
  exact Icc_subset_Icc (_root_.le_inf h₁.1 h₂.1) (_root_.sup_le h₁.2 h₂.2)
  /-
    🎉 no goals
  -/


theorem uIcc_subset_Icc (ha : a₁ ∈ Icc a₂ b₂) (hb : b₁ ∈ Icc a₂ b₂) : [[a₁, b₁]] ⊆ Icc a₂ b₂ := by
  /-
    α : Type u_2
    inst✝¹ : Lattice α
    inst✝ : LocallyFiniteOrder α
    a₁ a₂ b₁ b₂ : α
    ha : Membership.mem (Finset.Icc a₂ b₂) a₁
    hb : Membership.mem (Finset.Icc a₂ b₂) b₁
    ⊢ HasSubset.Subset (Finset.uIcc a₁ b₁) (Finset.Icc a₂ b₂)
  -/
  rw [mem_Icc] at ha hb
  /-
    α : Type u_2
    inst✝¹ : Lattice α
    inst✝ : LocallyFiniteOrder α
    a₁ a₂ b₁ b₂ : α
    ha : And (LE.le a₂ a₁) (LE.le a₁ b₂)
    hb : And (LE.le a₂ b₁) (LE.le b₁ b₂)
    ⊢ HasSubset.Subset (Finset.uIcc a₁ b₁) (Finset.Icc a₂ b₂)
  -/
  exact Icc_subset_Icc (_root_.le_inf ha.1 hb.1) (_root_.sup_le ha.2 hb.2)
  /-
    🎉 no goals
  -/


theorem uIcc_subset_uIcc_iff_mem : [[a₁, b₁]] ⊆ [[a₂, b₂]] ↔ a₁ ∈ [[a₂, b₂]] ∧ b₁ ∈ [[a₂, b₂]] :=
  ⟨fun h => ⟨h left_mem_uIcc, h right_mem_uIcc⟩, fun h => uIcc_subset_uIcc h.1 h.2⟩


theorem uIcc_subset_uIcc_iff_le' :
    [[a₁, b₁]] ⊆ [[a₂, b₂]] ↔ a₂ ⊓ b₂ ≤ a₁ ⊓ b₁ ∧ a₁ ⊔ b₁ ≤ a₂ ⊔ b₂ :=
  Icc_subset_Icc_iff inf_le_sup


theorem uIcc_subset_uIcc_right (h : x ∈ [[a, b]]) : [[x, b]] ⊆ [[a, b]] :=
  uIcc_subset_uIcc h right_mem_uIcc


theorem uIcc_subset_uIcc_left (h : x ∈ [[a, b]]) : [[a, x]] ⊆ [[a, b]] :=
  uIcc_subset_uIcc left_mem_uIcc h


theorem eq_of_mem_uIcc_of_mem_uIcc : a ∈ [[b, c]] → b ∈ [[a, c]] → a = b := by
  /-
    α : Type u_2
    inst✝¹ : DistribLattice α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Membership.mem (Finset.uIcc b c) a → Membership.mem (Finset.uIcc a c) b → Eq …
  -/
  simp_rw [mem_uIcc]
  /-
    α : Type u_2
    inst✝¹ : DistribLattice α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ And (LE.le (Min.min b c) a) (LE.le a (Max.max b c)) → And (LE.le (Min.min a  …
  -/
  exact Set.eq_of_mem_uIcc_of_mem_uIcc
  /-
    🎉 no goals
  -/


theorem eq_of_mem_uIcc_of_mem_uIcc' : b ∈ [[a, c]] → c ∈ [[a, b]] → b = c := by
  /-
    α : Type u_2
    inst✝¹ : DistribLattice α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Membership.mem (Finset.uIcc a c) b → Membership.mem (Finset.uIcc a b) c → Eq …
  -/
  simp_rw [mem_uIcc]
  /-
    α : Type u_2
    inst✝¹ : DistribLattice α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ And (LE.le (Min.min a c) b) (LE.le b (Max.max a c)) → And (LE.le (Min.min a  …
  -/
  exact Set.eq_of_mem_uIcc_of_mem_uIcc'
  /-
    🎉 no goals
  -/


theorem uIcc_injective_right (a : α) : Injective fun b => [[b, a]] := fun b c h => by
  /-
    α : Type u_2
    inst✝¹ : DistribLattice α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    h : Eq ((fun b => Finset.uIcc b a) b) ((fun b => Finset.uIcc b a) c)
    ⊢ Eq b c
  -/
  rw [Finset.ext_iff] at h
  /-
    α : Type u_2
    inst✝¹ : DistribLattice α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    h : ∀ (a_1 : α), Iff (Membership.mem ((fun b => Finset.uIcc b a) b) a_1) (Memb …
    ⊢ Eq b c
  -/
  exact eq_of_mem_uIcc_of_mem_uIcc ((h _).1 left_mem_uIcc) ((h _).2 left_mem_uIcc)
  /-
    🎉 no goals
  -/


theorem uIcc_injective_left (a : α) : Injective (uIcc a) := by
  /-
    α : Type u_2
    inst✝¹ : DistribLattice α
    inst✝ : LocallyFiniteOrder α
    a : α
    ⊢ Function.Injective (Finset.uIcc a)
  -/
  simpa only [uIcc_comm] using uIcc_injective_right a
  /-
    🎉 no goals
  -/


theorem Icc_min_max : Icc (min a b) (max a b) = [[a, b]] :=
  rfl


theorem uIcc_of_not_le (h : ¬a ≤ b) : [[a, b]] = Icc b a :=
  uIcc_of_ge <| le_of_not_ge h


theorem uIcc_of_not_ge (h : ¬b ≤ a) : [[a, b]] = Icc a b :=
  uIcc_of_le <| le_of_not_ge h


theorem uIcc_eq_union : [[a, b]] = Icc a b ∪ Icc b a :=
  coe_injective <| by
    /-
      α : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : LocallyFiniteOrder α
      a b : α
      ⊢ Eq ↑(Finset.uIcc a b) ↑(Union.union (Finset.Icc a b) (Finset.Icc b a))
    -/
    push_cast
    /-
      α : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : LocallyFiniteOrder α
      a b : α
      ⊢ Eq (Set.uIcc a b) (Union.union (Set.Icc a b) (Set.Icc b a))
    -/
    exact Set.uIcc_eq_union
    /-
      🎉 no goals
    -/


                                                                       /-
                                                                         α : Type u_2
                                                                         inst✝¹ : LinearOrder α
                                                                         inst✝ : LocallyFiniteOrder α
                                                                         a b c : α
                                                                         ⊢ Iff (Membership.mem (Finset.uIcc b c) a) (Or (And (LE.le b a) (LE.le a c)) ( …
                                                                       -/
theorem mem_uIcc' : a ∈ [[b, c]] ↔ b ≤ a ∧ a ≤ c ∨ c ≤ a ∧ a ≤ b := by simp [uIcc_eq_union]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem not_mem_uIcc_of_lt : c < a → c < b → c ∉ [[a, b]] := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ LT.lt c a → LT.lt c b → Not (Membership.mem (Finset.uIcc a b) c)
  -/
  rw [mem_uIcc]
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ LT.lt c a → LT.lt c b → Not (And (LE.le (Min.min a b) c) (LE.le c (Max.max a …
  -/
  exact Set.not_mem_uIcc_of_lt
  /-
    🎉 no goals
  -/


theorem not_mem_uIcc_of_gt : a < c → b < c → c ∉ [[a, b]] := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ LT.lt a c → LT.lt b c → Not (Membership.mem (Finset.uIcc a b) c)
  -/
  rw [mem_uIcc]
  /-
    α : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ LT.lt a c → LT.lt b c → Not (And (LE.le (Min.min a b) c) (LE.le c (Max.max a …
  -/
  exact Set.not_mem_uIcc_of_gt
  /-
    🎉 no goals
  -/


theorem uIcc_subset_uIcc_iff_le :
    [[a₁, b₁]] ⊆ [[a₂, b₂]] ↔ min a₂ b₂ ≤ min a₁ b₁ ∧ max a₁ b₁ ≤ max a₂ b₂ :=
  uIcc_subset_uIcc_iff_le'


/-- A sort of triangle inequality. -/
theorem uIcc_subset_uIcc_union_uIcc : [[a, c]] ⊆ [[a, b]] ∪ [[b, c]] :=
  coe_subset.1 <| by
    /-
      α : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : LocallyFiniteOrder α
      a b c : α
      ⊢ HasSubset.Subset ↑(Finset.uIcc a c) ↑(Union.union (Finset.uIcc a b) (Finset. …
    -/
    push_cast
    /-
      α : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : LocallyFiniteOrder α
      a b c : α
      ⊢ HasSubset.Subset (Set.uIcc a c) (Union.union (Set.uIcc a b) (Set.uIcc b c))
    -/
    exact Set.uIcc_subset_uIcc_union_uIcc
    /-
      🎉 no goals
    -/


set_option linter.unusedVariables false in -- `have` for wf induction triggers linter
lemma transGen_wcovBy_of_le [Preorder α] [LocallyFiniteOrder α] {x y : α} (hxy : x ≤ y) :
    TransGen (· ⩿ ·) x y := by
  -- We proceed by well-founded induction on the cardinality of `Icc x y`.
  -- It's impossible for the cardinality to be zero since `x ≤ y`
  have : #(Ico x y) < #(Icc x y) := card_lt_card <|
    ⟨Ico_subset_Icc_self, not_subset.mpr ⟨y, ⟨right_mem_Icc.mpr hxy, right_not_mem_Ico⟩⟩⟩
  /-
    α : Type u_2
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    x y : α
    hxy : LE.le x y
    this : LT.lt (Finset.Ico x y).card (Finset.Icc x y).card
    ⊢ Relation.TransGen (fun x1 x2 => WCovBy x1 x2) x y
  -/
  by_cases hxy' : y ≤ x
  -- If `y ≤ x`, then `x ⩿ y`
    /-
      case pos
      α : Type u_2
      inst✝¹ : Preorder α
      inst✝ : LocallyFiniteOrder α
      x y : α
      hxy : LE.le x y
      this : LT.lt (Finset.Ico x y).card (Finset.Icc x y).card
      hxy' : LE.le y x
      ⊢ Relation.TransGen (fun x1 x2 => WCovBy x1 x2) x y
    -/
  · exact .single <| wcovBy_of_le_of_le hxy hxy'
    /-
      🎉 no goals
    -/
  /- and if `¬ y ≤ x`, then `x < y`, not because it is a linear order, but because `x ≤ y`
  already. In that case, since `z` is maximal in `Ico x y`, then `z ⩿ y` and we can use the
  induction hypothesis to show that `Relation.TransGen (· ⩿ ·) x z`. -/
    /-
      case neg
      α : Type u_2
      inst✝¹ : Preorder α
      inst✝ : LocallyFiniteOrder α
      x y : α
      hxy : LE.le x y
      this : LT.lt (Finset.Ico x y).card (Finset.Icc x y).card
      hxy' : Not (LE.le y x)
      ⊢ Relation.TransGen (fun x1 x2 => WCovBy x1 x2) x y
    -/
  · have h_non : (Ico x y).Nonempty := ⟨x, mem_Ico.mpr ⟨le_rfl, lt_of_le_not_le hxy hxy'⟩⟩
    /-
      case neg
      α : Type u_2
      inst✝¹ : Preorder α
      inst✝ : LocallyFiniteOrder α
      x y : α
      hxy : LE.le x y
      this : LT.lt (Finset.Ico x y).card (Finset.Icc x y).card
      hxy' : Not (LE.le y x)
      h_non : (Finset.Ico x y).Nonempty
      ⊢ Relation.TransGen (fun x1 x2 => WCovBy x1 x2) x y
    -/
    obtain ⟨z, z_mem, hz⟩ := (Ico x y).exists_maximal h_non
    have z_card := calc
      #(Icc x z) ≤ #(Ico x y) := card_le_card <| Icc_subset_Ico_right (mem_Ico.mp z_mem).2
      _          < #(Icc x y) := this
    /-
      case neg.intro.intro
      α : Type u_2
      inst✝¹ : Preorder α
      inst✝ : LocallyFiniteOrder α
      x y : α
      hxy : LE.le x y
      this : LT.lt (Finset.Ico x y).card (Finset.Icc x y).card
      hxy' : Not (LE.le y x)
      h_non : (Finset.Ico x y).Nonempty
      z : α
      z_mem : Membership.mem (Finset.Ico x y) z
      hz : ∀ (x_1 : α), Membership.mem (Finset.Ico x y) x_1 → Not (LT.lt z x_1)
      z_card : LT.lt (Finset.Icc x z).card (Finset.Icc x y).card
      ⊢ Relation.TransGen (fun x1 x2 => WCovBy x1 x2) x y
    -/
    have h₁ := transGen_wcovBy_of_le (mem_Ico.mp z_mem).1
    have h₂ : z ⩿ y := by
      refine ⟨(mem_Ico.mp z_mem).2.le, fun c hzc hcy ↦ hz c ?_ hzc⟩
      exact mem_Ico.mpr <| ⟨(mem_Ico.mp z_mem).1.trans hzc.le, hcy⟩
    /-
      case neg.intro.intro
      α : Type u_2
      inst✝¹ : Preorder α
      inst✝ : LocallyFiniteOrder α
      x y : α
      hxy : LE.le x y
      this : LT.lt (Finset.Ico x y).card (Finset.Icc x y).card
      hxy' : Not (LE.le y x)
      h_non : (Finset.Ico x y).Nonempty
      z : α
      z_mem : Membership.mem (Finset.Ico x y) z
      hz : ∀ (x_1 : α), Membership.mem (Finset.Ico x y) x_1 → Not (LT.lt z x_1)
      z_card : LT.lt (Finset.Icc x z).card (Finset.Icc x y).card
      h₁ : Relation.TransGen (fun x1 x2 => WCovBy x1 x2) x z
      h₂ : WCovBy z y
      ⊢ Relation.TransGen (fun x1 x2 => WCovBy x1 x2) x y
    -/
    exact .tail h₁ h₂
    /-
      🎉 no goals
    -/
termination_by #(Icc x y)


/-- In a locally finite preorder, `≤` is the transitive closure of `⩿`. -/
lemma le_iff_transGen_wcovBy [Preorder α] [LocallyFiniteOrder α] {x y : α} :
    x ≤ y ↔ TransGen (· ⩿ ·) x y := by
  /-
    α : Type u_2
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    x y : α
    ⊢ Iff (LE.le x y) (Relation.TransGen (fun x1 x2 => WCovBy x1 x2) x y)
  -/
  refine ⟨transGen_wcovBy_of_le, fun h ↦ ?_⟩
  induction h with
  | single h => exact h.le
  | tail _ h₁ h₂ => exact h₂.trans h₁.le


/-- In a locally finite partial order, `≤` is the reflexive transitive closure of `⋖`. -/
lemma le_iff_reflTransGen_covBy [PartialOrder α] [LocallyFiniteOrder α] {x y : α} :
    x ≤ y ↔ ReflTransGen (· ⋖ ·) x y := by
  /-
    α : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : LocallyFiniteOrder α
    x y : α
    ⊢ Iff (LE.le x y) (Relation.ReflTransGen (fun x1 x2 => CovBy x1 x2) x y)
  -/
  rw [le_iff_transGen_wcovBy, wcovBy_eq_reflGen_covBy, transGen_reflGen]
  /-
    🎉 no goals
  -/


set_option linter.unusedVariables false in -- `have` for wf induction triggers linter
lemma transGen_covBy_of_lt [Preorder α] [LocallyFiniteOrder α] {x y : α} (hxy : x < y) :
    TransGen (· ⋖ ·) x y := by
  -- We proceed by well-founded induction on the cardinality of `Ico x y`.
  -- It's impossible for the cardinality to be zero since `x < y`
  /-
    α : Type u_2
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    x y : α
    hxy : LT.lt x y
    ⊢ Relation.TransGen (fun x1 x2 => CovBy x1 x2) x y
  -/
  have h_non : (Ico x y).Nonempty := ⟨x, mem_Ico.mpr ⟨le_rfl, hxy⟩⟩
  -- `Ico x y` is a nonempty finset and so contains a maximal element `z` and
  -- `Ico x z` has cardinality strictly less than the cardinality of `Ico x y`
  /-
    α : Type u_2
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    x y : α
    hxy : LT.lt x y
    h_non : (Finset.Ico x y).Nonempty
    ⊢ Relation.TransGen (fun x1 x2 => CovBy x1 x2) x y
  -/
  obtain ⟨z, z_mem, hz⟩ := (Ico x y).exists_maximal h_non
  have z_card : #(Ico x z) < #(Ico x y) := card_lt_card <| ssubset_iff_of_subset
    (Ico_subset_Ico le_rfl (mem_Ico.mp z_mem).2.le) |>.mpr ⟨z, z_mem, right_not_mem_Ico⟩
  /- Since `z` is maximal in `Ico x y`, `z ⋖ y`. -/
  have hzy : z ⋖ y := by
    refine ⟨(mem_Ico.mp z_mem).2, fun c hc hcy ↦ ?_⟩
    exact hz _ (mem_Ico.mpr ⟨((mem_Ico.mp z_mem).1.trans_lt hc).le, hcy⟩) hc
  /-
    case intro.intro
    α : Type u_2
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    x y : α
    hxy : LT.lt x y
    h_non : (Finset.Ico x y).Nonempty
    z : α
    z_mem : Membership.mem (Finset.Ico x y) z
    hz : ∀ (x_1 : α), Membership.mem (Finset.Ico x y) x_1 → Not (LT.lt z x_1)
    z_card : LT.lt (Finset.Ico x z).card (Finset.Ico x y).card
    hzy : CovBy z y
    ⊢ Relation.TransGen (fun x1 x2 => CovBy x1 x2) x y
  -/
  by_cases hxz : x < z
  /- when `x < z`, then we may use the induction hypothesis to get a chain
  `Relation.TransGen (· ⋖ ·) x z`, which we can extend with `Relation.TransGen.tail`. -/
    /-
      case pos
      α : Type u_2
      inst✝¹ : Preorder α
      inst✝ : LocallyFiniteOrder α
      x y : α
      hxy : LT.lt x y
      h_non : (Finset.Ico x y).Nonempty
      z : α
      z_mem : Membership.mem (Finset.Ico x y) z
      hz : ∀ (x_1 : α), Membership.mem (Finset.Ico x y) x_1 → Not (LT.lt z x_1)
      z_card : LT.lt (Finset.Ico x z).card (Finset.Ico x y).card
      hzy : CovBy z y
      hxz : LT.lt x z
      ⊢ Relation.TransGen (fun x1 x2 => CovBy x1 x2) x y
    -/
  · exact .tail (transGen_covBy_of_lt hxz) hzy
    /-
      🎉 no goals
    -/
  /- when `¬ x < z`, then actually `z ≤ x` (not because it's a linear order, but because
  `x ≤ z`), and since `z ⋖ y` we conclude that `x ⋖ y` , then `Relation.TransGen.single`. -/
    /-
      case neg
      α : Type u_2
      inst✝¹ : Preorder α
      inst✝ : LocallyFiniteOrder α
      x y : α
      hxy : LT.lt x y
      h_non : (Finset.Ico x y).Nonempty
      z : α
      z_mem : Membership.mem (Finset.Ico x y) z
      hz : ∀ (x_1 : α), Membership.mem (Finset.Ico x y) x_1 → Not (LT.lt z x_1)
      z_card : LT.lt (Finset.Ico x z).card (Finset.Ico x y).card
      hzy : CovBy z y
      hxz : Not (LT.lt x z)
      ⊢ Relation.TransGen (fun x1 x2 => CovBy x1 x2) x y
    -/
  · simp only [lt_iff_le_not_le, not_and, not_not] at hxz
    /-
      case neg
      α : Type u_2
      inst✝¹ : Preorder α
      inst✝ : LocallyFiniteOrder α
      x y : α
      hxy : LT.lt x y
      h_non : (Finset.Ico x y).Nonempty
      z : α
      z_mem : Membership.mem (Finset.Ico x y) z
      hz : ∀ (x_1 : α), Membership.mem (Finset.Ico x y) x_1 → Not (LT.lt z x_1)
      z_card : LT.lt (Finset.Ico x z).card (Finset.Ico x y).card
      hzy : CovBy z y
      hxz : LE.le x z → LE.le z x
      ⊢ Relation.TransGen (fun x1 x2 => CovBy x1 x2) x y
    -/
    exact .single (hzy.of_le_of_lt (hxz (mem_Ico.mp z_mem).1) hxy)
    /-
      🎉 no goals
    -/
termination_by #(Ico x y)


/-- In a locally finite preorder, `<` is the transitive closure of `⋖`. -/
lemma lt_iff_transGen_covBy [Preorder α] [LocallyFiniteOrder α] {x y : α} :
    x < y ↔ TransGen (· ⋖ ·) x y := by
  /-
    α : Type u_2
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    x y : α
    ⊢ Iff (LT.lt x y) (Relation.TransGen (fun x1 x2 => CovBy x1 x2) x y)
  -/
  refine ⟨transGen_covBy_of_lt, fun h ↦ ?_⟩
  induction h with
  | single hx => exact hx.1
  | tail _ hb ih => exact ih.trans hb.1


/-- A function from a locally finite preorder is monotone if and only if it is monotone when
restricted to pairs satisfying `a ⩿ b`. -/
lemma monotone_iff_forall_wcovBy [Preorder α] [LocallyFiniteOrder α] [Preorder β]
    (f : α → β) : Monotone f ↔ ∀ a b : α, a ⩿ b → f a ≤ f b := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (Monotone f) (∀ (a b : α), WCovBy a b → LE.le (f a) (f b))
  -/
  refine ⟨fun hf _ _ h ↦ hf h.le, fun h a b hab ↦ ?_⟩
  simpa [transGen_eq_self (r := ((· : β) ≤ ·)) transitive_le]
    using TransGen.lift f h <| le_iff_transGen_wcovBy.mp hab


/-- A function from a locally finite partial order is monotone if and only if it is monotone when
restricted to pairs satisfying `a ⋖ b`. -/
lemma monotone_iff_forall_covBy [PartialOrder α] [LocallyFiniteOrder α] [Preorder β]
    (f : α → β) : Monotone f ↔ ∀ a b : α, a ⋖ b → f a ≤ f b := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (Monotone f) (∀ (a b : α), CovBy a b → LE.le (f a) (f b))
  -/
  refine ⟨fun hf _ _ h ↦ hf h.le, fun h a b hab ↦ ?_⟩
  simpa [reflTransGen_eq_self (r := ((· : β) ≤ ·)) IsRefl.reflexive transitive_le]
    using ReflTransGen.lift f h <| le_iff_reflTransGen_covBy.mp hab


/-- A function from a locally finite preorder is strictly monotone if and only if it is strictly
monotone when restricted to pairs satisfying `a ⋖ b`. -/
lemma strictMono_iff_forall_covBy [Preorder α] [LocallyFiniteOrder α] [Preorder β]
    (f : α → β) : StrictMono f ↔ ∀ a b : α, a ⋖ b → f a < f b := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (StrictMono f) (∀ (a b : α), CovBy a b → LT.lt (f a) (f b))
  -/
  refine ⟨fun hf _ _ h ↦ hf h.lt, fun h a b hab ↦ ?_⟩
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : Preorder β
    f : α → β
    h : ∀ (a b : α), CovBy a b → LT.lt (f a) (f b)
    a b : α
    hab : LT.lt a b
    ⊢ LT.lt (f a) (f b)
  -/
  have := Relation.TransGen.lift f h (a := a) (b := b)
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : Preorder β
    f : α → β
    h : ∀ (a b : α), CovBy a b → LT.lt (f a) (f b)
    a b : α
    hab : LT.lt a b
    this : Relation.TransGen CovBy a b → Relation.TransGen LT.lt (f a) (f b)
    ⊢ LT.lt (f a) (f b)
  -/
  rw [← lt_iff_transGen_covBy, transGen_eq_self (@lt_trans β _)] at this
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : Preorder β
    f : α → β
    h : ∀ (a b : α), CovBy a b → LT.lt (f a) (f b)
    a b : α
    hab : LT.lt a b
    this : LT.lt a b → LT.lt (f a) (f b)
    ⊢ LT.lt (f a) (f b)
  -/
  exact this hab
  /-
    🎉 no goals
  -/


/-- A function from a locally finite preorder is antitone if and only if it is antitone when
restricted to pairs satisfying `a ⩿ b`. -/
lemma antitone_iff_forall_wcovBy [Preorder α] [LocallyFiniteOrder α] [Preorder β]
    (f : α → β) : Antitone f ↔ ∀ a b : α, a ⩿ b → f b ≤ f a :=
  monotone_iff_forall_wcovBy (β := βᵒᵈ) f


/-- A function from a locally finite partial order is antitone if and only if it is antitone when
restricted to pairs satisfying `a ⋖ b`. -/
lemma antitone_iff_forall_covBy [PartialOrder α] [LocallyFiniteOrder α] [Preorder β]
    (f : α → β) : Antitone f ↔ ∀ a b : α, a ⋖ b → f b ≤ f a :=
  monotone_iff_forall_covBy (β := βᵒᵈ) f


/-- A function from a locally finite preorder is strictly antitone if and only if it is strictly
antitone when restricted to pairs satisfying `a ⋖ b`. -/
lemma strictAnti_iff_forall_covBy [Preorder α] [LocallyFiniteOrder α] [Preorder β]
    (f : α → β) : StrictAnti f ↔ ∀ a b : α, a ⋖ b → f b < f a :=
  strictMono_iff_forall_covBy (β := βᵒᵈ) f


