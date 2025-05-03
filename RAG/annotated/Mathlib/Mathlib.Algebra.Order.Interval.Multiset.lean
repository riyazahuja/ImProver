lemma map_add_left_Icc (a b c : α) : (Icc a b).map (c + ·) = Icc (c + a) (c + b) := by
  classical rw [Icc, Icc, ← Finset.image_add_left_Icc, Finset.image_val,
      ((Finset.nodup _).map <| add_right_injective c).dedup]


lemma map_add_left_Ico (a b c : α) : (Ico a b).map (c + ·) = Ico (c + a) (c + b) := by
  classical rw [Ico, Ico, ← Finset.image_add_left_Ico, Finset.image_val,
      ((Finset.nodup _).map <| add_right_injective c).dedup]


lemma map_add_left_Ioc (a b c : α) : (Ioc a b).map (c + ·) = Ioc (c + a) (c + b) := by
  classical rw [Ioc, Ioc, ← Finset.image_add_left_Ioc, Finset.image_val,
      ((Finset.nodup _).map <| add_right_injective c).dedup]


lemma map_add_left_Ioo (a b c : α) : (Ioo a b).map (c + ·) = Ioo (c + a) (c + b) := by
  classical rw [Ioo, Ioo, ← Finset.image_add_left_Ioo, Finset.image_val,
      ((Finset.nodup _).map <| add_right_injective c).dedup]


lemma map_add_right_Icc (a b c : α) : ((Icc a b).map fun x => x + c) = Icc (a + c) (b + c) := by
  /-
    α : Type u_1
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Multiset.map (fun x => HAdd.hAdd x c) (Multiset.Icc a b)) (Multiset.Icc  …
  -/
  simp_rw [add_comm _ c]
  /-
    α : Type u_1
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Multiset.map (fun x => HAdd.hAdd c x) (Multiset.Icc a b)) (Multiset.Icc  …
  -/
  exact map_add_left_Icc _ _ _
  /-
    🎉 no goals
  -/


lemma map_add_right_Ico (a b c : α) : ((Ico a b).map fun x => x + c) = Ico (a + c) (b + c) := by
  /-
    α : Type u_1
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Multiset.map (fun x => HAdd.hAdd x c) (Multiset.Ico a b)) (Multiset.Ico  …
  -/
  simp_rw [add_comm _ c]
  /-
    α : Type u_1
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Multiset.map (fun x => HAdd.hAdd c x) (Multiset.Ico a b)) (Multiset.Ico  …
  -/
  exact map_add_left_Ico _ _ _
  /-
    🎉 no goals
  -/


lemma map_add_right_Ioc (a b c : α) : ((Ioc a b).map fun x => x + c) = Ioc (a + c) (b + c) := by
  /-
    α : Type u_1
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Multiset.map (fun x => HAdd.hAdd x c) (Multiset.Ioc a b)) (Multiset.Ioc  …
  -/
  simp_rw [add_comm _ c]
  /-
    α : Type u_1
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Multiset.map (fun x => HAdd.hAdd c x) (Multiset.Ioc a b)) (Multiset.Ioc  …
  -/
  exact map_add_left_Ioc _ _ _
  /-
    🎉 no goals
  -/


lemma map_add_right_Ioo (a b c : α) : ((Ioo a b).map fun x => x + c) = Ioo (a + c) (b + c) := by
  /-
    α : Type u_1
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Multiset.map (fun x => HAdd.hAdd x c) (Multiset.Ioo a b)) (Multiset.Ioo  …
  -/
  simp_rw [add_comm _ c]
  /-
    α : Type u_1
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Multiset.map (fun x => HAdd.hAdd c x) (Multiset.Ioo a b)) (Multiset.Ioo  …
  -/
  exact map_add_left_Ioo _ _ _
  /-
    🎉 no goals
  -/


