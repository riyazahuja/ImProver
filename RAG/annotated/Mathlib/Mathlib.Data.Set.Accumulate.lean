/-- `Accumulate s` is the union of `s y` for `y ≤ x`. -/
def Accumulate [LE α] (s : α → Set β) (x : α) : Set β :=
  ⋃ y ≤ x, s y


theorem accumulate_def [LE α] {x : α} : Accumulate s x = ⋃ y ≤ x, s y :=
  rfl


@[simp]
theorem mem_accumulate [LE α] {x : α} {z : β} : z ∈ Accumulate s x ↔ ∃ y ≤ x, z ∈ s y := by
  /-
    α : Type u_1
    β : Type u_2
    s : α → Set β
    inst✝ : LE α
    x : α
    z : β
    ⊢ Iff (Membership.mem (Set.Accumulate s x) z) (Exists fun y => And (LE.le y x) …
  -/
  simp_rw [accumulate_def, mem_iUnion₂, exists_prop]
  /-
    🎉 no goals
  -/


theorem subset_accumulate [Preorder α] {x : α} : s x ⊆ Accumulate s x := fun _ => mem_biUnion le_rfl


theorem accumulate_subset_iUnion [Preorder α] (x : α) : Accumulate s x ⊆ ⋃ i, s i :=
  (biUnion_subset_biUnion_left (subset_univ _)).trans_eq (biUnion_univ _)


theorem monotone_accumulate [Preorder α] : Monotone (Accumulate s) := fun _ _ hxy =>
  biUnion_subset_biUnion_left fun _ hz => le_trans hz hxy


@[gcongr]
theorem accumulate_subset_accumulate [Preorder α] {x y} (h : x ≤ y) :
    Accumulate s x ⊆ Accumulate s y :=
  monotone_accumulate h


theorem biUnion_accumulate [Preorder α] (x : α) : ⋃ y ≤ x, Accumulate s y = ⋃ y ≤ x, s y := by
  /-
    α : Type u_1
    β : Type u_2
    s : α → Set β
    inst✝ : Preorder α
    x : α
    ⊢ Eq (Set.iUnion fun y => Set.iUnion fun h => Set.Accumulate s y) (Set.iUnion  …
  -/
  apply Subset.antisymm
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      s : α → Set β
      inst✝ : Preorder α
      x : α
      ⊢ HasSubset.Subset (Set.iUnion fun y => Set.iUnion fun h => Set.Accumulate s y …
    -/
  · exact iUnion₂_subset fun y hy => monotone_accumulate hy
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      s : α → Set β
      inst✝ : Preorder α
      x : α
      ⊢ HasSubset.Subset (Set.iUnion fun y => Set.iUnion fun h => s y) (Set.iUnion f …
    -/
  · exact iUnion₂_mono fun y _ => subset_accumulate
    /-
      🎉 no goals
    -/


theorem iUnion_accumulate [Preorder α] : ⋃ x, Accumulate s x = ⋃ x, s x := by
  /-
    α : Type u_1
    β : Type u_2
    s : α → Set β
    inst✝ : Preorder α
    ⊢ Eq (Set.iUnion fun x => Set.Accumulate s x) (Set.iUnion fun x => s x)
  -/
  apply Subset.antisymm
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      s : α → Set β
      inst✝ : Preorder α
      ⊢ HasSubset.Subset (Set.iUnion fun x => Set.Accumulate s x) (Set.iUnion fun x  …
    -/
  · simp only [subset_def, mem_iUnion, exists_imp, mem_accumulate]
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      s : α → Set β
      inst✝ : Preorder α
      ⊢ ∀ (x : β) (x_1 x_2 : α), And (LE.le x_2 x_1) (Membership.mem (s x_2) x) → Ex …
    -/
    intro z x x' ⟨_, hz⟩
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      s : α → Set β
      inst✝ : Preorder α
      z : β
      x x' : α
      left✝ : LE.le x' x
      hz : Membership.mem (s x') z
      ⊢ Exists fun i => Membership.mem (s i) z
    -/
    exact ⟨x', hz⟩
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      s : α → Set β
      inst✝ : Preorder α
      ⊢ HasSubset.Subset (Set.iUnion fun x => s x) (Set.iUnion fun x => Set.Accumula …
    -/
  · exact iUnion_mono fun i => subset_accumulate
    /-
      🎉 no goals
    -/


