theorem range_list_map (f : α → β) : range (map f) = { l | ∀ x ∈ l, x ∈ range f } := by
  refine antisymm (range_subset_iff.2 fun l => forall_mem_map.2 fun y _ => mem_range_self _)
      fun l hl => ?_
  induction l with
  | nil => exact ⟨[], rfl⟩
  | cons a l ihl =>
    rcases ihl fun x hx => hl x <| subset_cons_self _ _ hx with ⟨l, rfl⟩
    rcases hl a (mem_cons_self _ _) with ⟨a, rfl⟩
    exact ⟨a :: l, map_cons _ _ _⟩


theorem range_list_map_coe (s : Set α) : range (map ((↑) : s → α)) = { l | ∀ x ∈ l, x ∈ s } := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (Set.range (List.map Subtype.val)) (setOf fun l => ∀ (x : α), Membership. …
  -/
  rw [range_list_map, Subtype.range_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem range_list_get : range l.get = { x | x ∈ l } := by
  /-
    α : Type u_1
    l : List α
    ⊢ Eq (Set.range l.get) (setOf fun x => Membership.mem l x)
  -/
  ext x
  /-
    case h
    α : Type u_1
    l : List α
    x : α
    ⊢ Iff (Membership.mem (Set.range l.get) x) (Membership.mem (setOf fun x => Mem …
  -/
  rw [mem_setOf_eq, mem_iff_get, mem_range]
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-04-22")] alias range_list_nthLe := range_list_get


theorem range_list_get? : range l.get? = insert none (some '' { x | x ∈ l }) := by
  /-
    α : Type u_1
    l : List α
    ⊢ Eq (Set.range l.get?) (Insert.insert Option.none (Set.image Option.some (set …
  -/
  rw [← range_list_get, ← range_comp]
  /-
    α : Type u_1
    l : List α
    ⊢ Eq (Set.range l.get?) (Insert.insert Option.none (Set.range (Function.comp O …
  -/
  refine (range_subset_iff.2 fun n => ?_).antisymm (insert_subset_iff.2 ⟨?_, ?_⟩)
  · exact (le_or_lt l.length n).imp get?_eq_none_iff.mpr
      (fun hlt => ⟨⟨_, hlt⟩, (get?_eq_get hlt).symm⟩)
    /-
      case refine_2
      α : Type u_1
      l : List α
      ⊢ Membership.mem (Set.range l.get?) Option.none
    -/
  · exact ⟨_, get?_eq_none_iff.mpr le_rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      l : List α
      ⊢ HasSubset.Subset (Set.range (Function.comp Option.some l.get)) (Set.range l. …
    -/
  · exact range_subset_iff.2 fun k => ⟨_, get?_eq_get _⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem range_list_getD (d : α) : (range fun n : Nat => l[n]?.getD d) = insert d { x | x ∈ l } :=
  calc
    (range fun n => l[n]?.getD d) = (fun o : Option α => o.getD d) '' range l.get? := by
      /-
        α : Type u_1
        l : List α
        d : α
        ⊢ Eq (Set.range fun n => (GetElem?.getElem? l n).getD d) (Set.image (fun o =>  …
      -/
      simp [← range_comp, Function.comp_def]
      /-
        🎉 no goals
      -/
    _ = insert d { x | x ∈ l } := by
      /-
        α : Type u_1
        l : List α
        d : α
        ⊢ Eq (Set.image (fun o => o.getD d) (Set.range l.get?)) (Insert.insert d (setO …
      -/
      simp only [range_list_get?, image_insert_eq, Option.getD, image_image, image_id']
      /-
        🎉 no goals
      -/


@[simp]
theorem range_list_getI [Inhabited α] (l : List α) :
    range l.getI = insert default { x | x ∈ l } := by
  /-
    α : Type u_1
    inst✝ : Inhabited α
    l : List α
    ⊢ Eq (Set.range l.getI) (Insert.insert Inhabited.default (setOf fun x => Membe …
  -/
  unfold List.getI
  /-
    α : Type u_1
    inst✝ : Inhabited α
    l : List α
    ⊢ Eq (Set.range fun n => l.getD n Inhabited.default) (Insert.insert Inhabited. …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If each element of a list can be lifted to some type, then the whole list can be
lifted to this type. -/
instance List.canLift (c) (p) [CanLift α β c p] :
    CanLift (List α) (List β) (List.map c) fun l => ∀ x ∈ l, p x where
  prf l H := by
    /-
      α : Type u_1
      β : Type u_2
      l✝ : List α
      c : β → α
      p : α → Prop
      inst✝ : CanLift α β c p
      l : List α
      H : ∀ (x : α), Membership.mem l x → p x
      ⊢ Exists fun y => Eq (List.map c y) l
    -/
    rw [← Set.mem_range, Set.range_list_map]
    /-
      α : Type u_1
      β : Type u_2
      l✝ : List α
      c : β → α
      p : α → Prop
      inst✝ : CanLift α β c p
      l : List α
      H : ∀ (x : α), Membership.mem l x → p x
      ⊢ Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l x → Membership.me …
    -/
    exact fun a ha => CanLift.prf a (H a ha)
    /-
      🎉 no goals
    -/

