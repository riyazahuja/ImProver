/-- A Priestley space is an ordered topological space such that any two distinct points can be
separated by a clopen upper set. Compactness is often assumed, but we do not include it here. -/
class PriestleySpace (α : Type*) [Preorder α] [TopologicalSpace α] : Prop where
  priestley {x y : α} : ¬x ≤ y → ∃ U : Set α, IsClopen U ∧ IsUpperSet U ∧ x ∈ U ∧ y ∉ U


theorem exists_isClopen_upper_of_not_le :
    ¬x ≤ y → ∃ U : Set α, IsClopen U ∧ IsUpperSet U ∧ x ∈ U ∧ y ∉ U :=
  PriestleySpace.priestley


theorem exists_isClopen_lower_of_not_le (h : ¬x ≤ y) :
    ∃ U : Set α, IsClopen U ∧ IsLowerSet U ∧ x ∉ U ∧ y ∈ U :=
  let ⟨U, hU, hU', hx, hy⟩ := exists_isClopen_upper_of_not_le h
  ⟨Uᶜ, hU.compl, hU'.compl, Classical.not_not.2 hx, hy⟩


theorem exists_isClopen_upper_or_lower_of_ne (h : x ≠ y) :
    ∃ U : Set α, IsClopen U ∧ (IsUpperSet U ∨ IsLowerSet U) ∧ x ∈ U ∧ y ∉ U := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : PartialOrder α
    inst✝ : PriestleySpace α
    x y : α
    h : Ne x y
    ⊢ Exists fun U => And (IsClopen U) (And (Or (IsUpperSet U) (IsLowerSet U)) (An …
  -/
  obtain h | h := h.not_le_or_not_le
    /-
      case inl
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : PartialOrder α
      inst✝ : PriestleySpace α
      x y : α
      h✝ : Ne x y
      h : Not (LE.le x y)
      ⊢ Exists fun U => And (IsClopen U) (And (Or (IsUpperSet U) (IsLowerSet U)) (An …
    -/
  · exact (exists_isClopen_upper_of_not_le h).imp fun _ ↦ And.imp_right <| And.imp_left Or.inl
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : PartialOrder α
      inst✝ : PriestleySpace α
      x y : α
      h✝ : Ne x y
      h : Not (LE.le y x)
      ⊢ Exists fun U => And (IsClopen U) (And (Or (IsUpperSet U) (IsLowerSet U)) (An …
    -/
  · obtain ⟨U, hU, hU', hy, hx⟩ := exists_isClopen_lower_of_not_le h
    /-
      case inr.intro.intro.intro.intro
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : PartialOrder α
      inst✝ : PriestleySpace α
      x y : α
      h✝ : Ne x y
      h : Not (LE.le y x)
      U : Set α
      hU : IsClopen U
      hU' : IsLowerSet U
      hy : Not (Membership.mem U y)
      hx : Membership.mem U x
      ⊢ Exists fun U => And (IsClopen U) (And (Or (IsUpperSet U) (IsLowerSet U)) (An …
    -/
    exact ⟨U, hU, Or.inr hU', hx, hy⟩
    /-
      🎉 no goals
    -/

-- See note [lower instance priority]

instance (priority := 100) PriestleySpace.toT2Space : T2Space α :=
  ⟨fun _ _ h ↦
    let ⟨U, hU, _, hx, hy⟩ := exists_isClopen_upper_or_lower_of_ne h
    ⟨U, Uᶜ, hU.isOpen, hU.compl.isOpen, hx, hy, disjoint_compl_right⟩⟩


