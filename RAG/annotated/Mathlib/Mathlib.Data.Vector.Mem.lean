@[simp]
theorem get_mem (i : Fin n) (v : Vector α n) : v.get i ∈ v.toList := List.get_mem _ _


theorem mem_iff_get (v : Vector α n) : a ∈ v.toList ↔ ∃ i, v.get i = a := by
  /-
    α : Type u_1
    n : Nat
    a : α
    v : List.Vector α n
    ⊢ Iff (Membership.mem v.toList a) (Exists fun i => Eq (v.get i) a)
  -/
  simp only [List.mem_iff_get, Fin.exists_iff, Vector.get_eq_get_toList]
  exact
    ⟨fun ⟨i, hi, h⟩ => ⟨i, by rwa [toList_length] at hi, h⟩, fun ⟨i, hi, h⟩ =>
      ⟨i, by rwa [toList_length], h⟩⟩


theorem not_mem_nil : a ∉ (Vector.nil : Vector α 0).toList := by
  /-
    α : Type u_1
    a : α
    ⊢ Not (Membership.mem List.Vector.nil.toList a)
  -/
  unfold Vector.nil
  /-
    α : Type u_1
    a : α
    ⊢ Not (Membership.mem (List.Vector.toList ⟨List.nil, ⋯⟩) a)
  -/
  dsimp
  /-
    α : Type u_1
    a : α
    ⊢ Not (Membership.mem List.nil a)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem not_mem_zero (v : Vector α 0) : a ∉ v.toList :=
  (Vector.eq_nil v).symm ▸ not_mem_nil a


theorem mem_cons_iff (v : Vector α n) : a' ∈ (a ::ᵥ v).toList ↔ a' = a ∨ a' ∈ v.toList := by
  /-
    α : Type u_1
    n : Nat
    a a' : α
    v : List.Vector α n
    ⊢ Iff (Membership.mem (List.Vector.cons a v).toList a') (Or (Eq a' a) (Members …
  -/
  rw [Vector.toList_cons, List.mem_cons]
  /-
    🎉 no goals
  -/


theorem mem_succ_iff (v : Vector α (n + 1)) : a ∈ v.toList ↔ a = v.head ∨ a ∈ v.tail.toList := by
  /-
    α : Type u_1
    n : Nat
    a : α
    v : List.Vector α (HAdd.hAdd n 1)
    ⊢ Iff (Membership.mem v.toList a) (Or (Eq a v.head) (Membership.mem v.tail.toL …
  -/
  obtain ⟨a', v', h⟩ := exists_eq_cons v
  /-
    case intro.intro
    α : Type u_1
    n : Nat
    a : α
    v : List.Vector α (HAdd.hAdd n 1)
    a' : α
    v' : List.Vector α n
    h : Eq v (List.Vector.cons a' v')
    ⊢ Iff (Membership.mem v.toList a) (Or (Eq a v.head) (Membership.mem v.tail.toL …
  -/
  simp_rw [h, Vector.mem_cons_iff, Vector.head_cons, Vector.tail_cons]
  /-
    🎉 no goals
  -/


theorem mem_cons_self (v : Vector α n) : a ∈ (a ::ᵥ v).toList :=
  (Vector.mem_iff_get a (a ::ᵥ v)).2 ⟨0, Vector.get_cons_zero a v⟩


@[simp]
theorem head_mem (v : Vector α (n + 1)) : v.head ∈ v.toList :=
  (Vector.mem_iff_get v.head v).2 ⟨0, Vector.get_zero v⟩


theorem mem_cons_of_mem (v : Vector α n) (ha' : a' ∈ v.toList) : a' ∈ (a ::ᵥ v).toList :=
  (Vector.mem_cons_iff a a' v).2 (Or.inr ha')


theorem mem_of_mem_tail (v : Vector α n) (ha : a ∈ v.tail.toList) : a ∈ v.toList := by
  induction n with
  | zero => exact False.elim (Vector.not_mem_zero a v.tail ha)
  | succ n _ => exact (mem_succ_iff a v).2 (Or.inr ha)


theorem mem_map_iff (b : β) (v : Vector α n) (f : α → β) :
    b ∈ (v.map f).toList ↔ ∃ a : α, a ∈ v.toList ∧ f a = b := by
  /-
    α : Type u_1
    β : Type u_2
    n : Nat
    b : β
    v : List.Vector α n
    f : α → β
    ⊢ Iff (Membership.mem (List.Vector.map f v).toList b) (Exists fun a => And (Me …
  -/
  rw [Vector.toList_map, List.mem_map]
  /-
    🎉 no goals
  -/


theorem not_mem_map_zero (b : β) (v : Vector α 0) (f : α → β) : b ∉ (v.map f).toList := by
  /-
    α : Type u_1
    β : Type u_2
    b : β
    v : List.Vector α 0
    f : α → β
    ⊢ Not (Membership.mem (List.Vector.map f v).toList b)
  -/
  simpa only [Vector.eq_nil v, Vector.map_nil, Vector.toList_nil] using List.not_mem_nil b
  /-
    🎉 no goals
  -/


theorem mem_map_succ_iff (b : β) (v : Vector α (n + 1)) (f : α → β) :
    b ∈ (v.map f).toList ↔ f v.head = b ∨ ∃ a : α, a ∈ v.tail.toList ∧ f a = b := by
  /-
    α : Type u_1
    β : Type u_2
    n : Nat
    b : β
    v : List.Vector α (HAdd.hAdd n 1)
    f : α → β
    ⊢ Iff (Membership.mem (List.Vector.map f v).toList b) (Or (Eq (f v.head) b) (E …
  -/
  rw [mem_succ_iff, head_map, tail_map, mem_map_iff, @eq_comm _ b]
  /-
    🎉 no goals
  -/


