/-- `boolIndicator` maps `x` to `true` if `x ∈ s`, else to `false` -/
noncomputable def boolIndicator (x : α) :=
  @ite _ (x ∈ s) (Classical.propDecidable _) true false


theorem mem_iff_boolIndicator (x : α) : x ∈ s ↔ s.boolIndicator x = true := by
  /-
    α : Type u_1
    s : Set α
    x : α
    ⊢ Iff (Membership.mem s x) (Eq (s.boolIndicator x) Bool.true)
  -/
  unfold boolIndicator
  /-
    α : Type u_1
    s : Set α
    x : α
    ⊢ Iff (Membership.mem s x) (Eq (ite (Membership.mem s x) Bool.true Bool.false) …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


theorem not_mem_iff_boolIndicator (x : α) : x ∉ s ↔ s.boolIndicator x = false := by
  /-
    α : Type u_1
    s : Set α
    x : α
    ⊢ Iff (Not (Membership.mem s x)) (Eq (s.boolIndicator x) Bool.false)
  -/
  unfold boolIndicator
  /-
    α : Type u_1
    s : Set α
    x : α
    ⊢ Iff (Not (Membership.mem s x)) (Eq (ite (Membership.mem s x) Bool.true Bool. …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


theorem preimage_boolIndicator_true : s.boolIndicator ⁻¹' {true} = s :=
  ext fun x ↦ (s.mem_iff_boolIndicator x).symm


theorem preimage_boolIndicator_false : s.boolIndicator ⁻¹' {false} = sᶜ :=
  ext fun x ↦ (s.not_mem_iff_boolIndicator x).symm


open scoped Classical in
theorem preimage_boolIndicator_eq_union (t : Set Bool) :
    s.boolIndicator ⁻¹' t = (if true ∈ t then s else ∅) ∪ if false ∈ t then sᶜ else ∅ := by
  /-
    α : Type u_1
    s : Set α
    t : Set Bool
    ⊢ Eq (Set.preimage s.boolIndicator t) (Union.union (ite (Membership.mem t Bool …
  -/
  ext x
  /-
    case h
    α : Type u_1
    s : Set α
    t : Set Bool
    x : α
    ⊢ Iff (Membership.mem (Set.preimage s.boolIndicator t) x) (Membership.mem (Uni …
  -/
  simp only [boolIndicator, mem_preimage]
  /-
    case h
    α : Type u_1
    s : Set α
    t : Set Bool
    x : α
    ⊢ Iff (Membership.mem t (ite (Membership.mem s x) Bool.true Bool.false)) (Memb …
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
                /-
                  🎉 no goals
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
  split_ifs <;> simp [*]
                /-
                  🎉 no goals
                -/


theorem preimage_boolIndicator (t : Set Bool) :
    s.boolIndicator ⁻¹' t = univ ∨
      s.boolIndicator ⁻¹' t = s ∨ s.boolIndicator ⁻¹' t = sᶜ ∨ s.boolIndicator ⁻¹' t = ∅ := by
  /-
    α : Type u_1
    s : Set α
    t : Set Bool
    ⊢ Or (Eq (Set.preimage s.boolIndicator t) Set.univ) (Or (Eq (Set.preimage s.bo …
  -/
  simp only [preimage_boolIndicator_eq_union]
  /-
    α : Type u_1
    s : Set α
    t : Set Bool
    ⊢ Or (Eq (Union.union (ite (Membership.mem t Bool.true) s EmptyCollection.empt …
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
  split_ifs <;> simp [s.union_compl_self]
                /-
                  🎉 no goals
                -/


