@[simp]
theorem dedup_nil : dedup [] = ([] : List α) :=
  rfl


theorem dedup_cons_of_mem' {a : α} {l : List α} (h : a ∈ dedup l) : dedup (a :: l) = dedup l :=
                             /-
                               α : Type u_1
                               inst✝ : DecidableEq α
                               a : α
                               l : List α
                               h : Membership.mem l.dedup a
                               ⊢ Not (∀ (b : α), Membership.mem (List.pwFilter (fun x1 x2 => Ne x1 x2) l) b → …
                             -/
  pwFilter_cons_of_neg <| by simpa only [forall_mem_ne, not_not] using h
                             /-
                               🎉 no goals
                             -/


theorem dedup_cons_of_not_mem' {a : α} {l : List α} (h : a ∉ dedup l) :
    dedup (a :: l) = a :: dedup l :=
                             /-
                               α : Type u_1
                               inst✝ : DecidableEq α
                               a : α
                               l : List α
                               h : Not (Membership.mem l.dedup a)
                               ⊢ ∀ (b : α), Membership.mem (List.pwFilter (fun x1 x2 => Ne x1 x2) l) b → Ne a b
                             -/
  pwFilter_cons_of_pos <| by simpa only [forall_mem_ne] using h
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem mem_dedup {a : α} {l : List α} : a ∈ dedup l ↔ a ∈ l := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    l : List α
    ⊢ Iff (Membership.mem l.dedup a) (Membership.mem l a)
  -/
  have := not_congr (@forall_mem_pwFilter α (· ≠ ·) _ ?_ a l)
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      this : Iff (Not (∀ (b : α), Membership.mem (List.pwFilter (fun x1 x2 => Ne x1  …
      ⊢ Iff (Membership.mem l.dedup a) (Membership.mem l a)
    -/
  · simpa only [dedup, forall_mem_ne, not_not] using this
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      ⊢ ∀ {x y z : α}, (fun x1 x2 => Ne x1 x2) x z → Or ((fun x1 x2 => Ne x1 x2) x y …
    -/
  · intros x y z xz
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      x y z : α
      xz : Ne x z
      ⊢ Or ((fun x1 x2 => Ne x1 x2) x y) ((fun x1 x2 => Ne x1 x2) y z)
    -/
    exact not_and_or.1 <| mt (fun h ↦ h.1.trans h.2) xz
    /-
      🎉 no goals
    -/


@[simp]
theorem dedup_cons_of_mem {a : α} {l : List α} (h : a ∈ l) : dedup (a :: l) = dedup l :=
  dedup_cons_of_mem' <| mem_dedup.2 h


@[simp]
theorem dedup_cons_of_not_mem {a : α} {l : List α} (h : a ∉ l) : dedup (a :: l) = a :: dedup l :=
  dedup_cons_of_not_mem' <| mt mem_dedup.1 h


theorem dedup_sublist : ∀ l : List α, dedup l <+ l :=
  pwFilter_sublist


theorem dedup_subset : ∀ l : List α, dedup l ⊆ l :=
  pwFilter_subset


theorem subset_dedup (l : List α) : l ⊆ dedup l := fun _ => mem_dedup.2


theorem nodup_dedup : ∀ l : List α, Nodup (dedup l) :=
  pairwise_pwFilter


theorem headI_dedup [Inhabited α] (l : List α) :
    l.dedup.headI = if l.headI ∈ l.tail then l.tail.dedup.headI else l.headI :=
  match l with
  | [] => rfl
                 /-
                   α : Type u_1
                   inst✝¹ : DecidableEq α
                   inst✝ : Inhabited α
                   l✝ : List α
                   a : α
                   l : List α
                   ⊢ Eq (List.cons a l).dedup.headI (ite (Membership.mem (List.cons a l).tail (Li …
                 -/
                                         /-
                                           🎉 no goals
                                         -/
  | a :: l => by by_cases ha : a ∈ l <;> simp [ha, List.dedup_cons_of_mem]
                                         /-
                                           🎉 no goals
                                         -/


theorem tail_dedup [Inhabited α] (l : List α) :
    l.dedup.tail = if l.headI ∈ l.tail then l.tail.dedup.tail else l.tail.dedup :=
  match l with
  | [] => rfl
                 /-
                   α : Type u_1
                   inst✝¹ : DecidableEq α
                   inst✝ : Inhabited α
                   l✝ : List α
                   a : α
                   l : List α
                   ⊢ Eq (List.cons a l).dedup.tail (ite (Membership.mem (List.cons a l).tail (Lis …
                 -/
                                         /-
                                           🎉 no goals
                                         -/
  | a :: l => by by_cases ha : a ∈ l <;> simp [ha, List.dedup_cons_of_mem]
                                         /-
                                           🎉 no goals
                                         -/


theorem dedup_eq_self {l : List α} : dedup l = l ↔ Nodup l :=
  pwFilter_eq_self


theorem dedup_eq_cons (l : List α) (a : α) (l' : List α) :
    l.dedup = a :: l' ↔ a ∈ l ∧ a ∉ l' ∧ l.dedup.tail = l' := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    a : α
    l' : List α
    ⊢ Iff (Eq l.dedup (List.cons a l')) (And (Membership.mem l a) (And (Not (Membe …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a : α
      l' : List α
      h : Eq l.dedup (List.cons a l')
      ⊢ And (Membership.mem l a) (And (Not (Membership.mem l' a)) (Eq l.dedup.tail l …
    -/
  · refine ⟨mem_dedup.1 (h.symm ▸ mem_cons_self _ _), fun ha => ?_, by rw [h, tail_cons]⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a : α
      l' : List α
      h : Eq l.dedup (List.cons a l')
      ha : Membership.mem l' a
      ⊢ False
    -/
    have := count_pos_iff.2 ha
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a : α
      l' : List α
      h : Eq l.dedup (List.cons a l')
      ha : Membership.mem l' a
      this : LT.lt 0 (List.count a l')
      ⊢ False
    -/
    have : count a l.dedup ≤ 1 := nodup_iff_count_le_one.1 (nodup_dedup l) a
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a : α
      l' : List α
      h : Eq l.dedup (List.cons a l')
      ha : Membership.mem l' a
      this✝ : LT.lt 0 (List.count a l')
      this : LE.le (List.count a l.dedup) 1
      ⊢ False
    -/
    rw [h, count_cons_self] at this
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a : α
      l' : List α
      h : Eq l.dedup (List.cons a l')
      ha : Membership.mem l' a
      this✝ : LT.lt 0 (List.count a l')
      this : LE.le (HAdd.hAdd (List.count a l') 1) 1
      ⊢ False
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a : α
      l' : List α
      h : And (Membership.mem l a) (And (Not (Membership.mem l' a)) (Eq l.dedup.tail …
      ⊢ Eq l.dedup (List.cons a l')
    -/
  · have := @List.cons_head!_tail α ⟨a⟩ _ (ne_nil_of_mem (mem_dedup.2 h.1))
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a : α
      l' : List α
      h : And (Membership.mem l a) (And (Not (Membership.mem l' a)) (Eq l.dedup.tail …
      this : Eq (List.cons l.dedup.head! l.dedup.tail) l.dedup
      ⊢ Eq l.dedup (List.cons a l')
    -/
    have hal : a ∈ l.dedup := mem_dedup.2 h.1
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a : α
      l' : List α
      h : And (Membership.mem l a) (And (Not (Membership.mem l' a)) (Eq l.dedup.tail …
      this : Eq (List.cons l.dedup.head! l.dedup.tail) l.dedup
      hal : Membership.mem l.dedup a
      ⊢ Eq l.dedup (List.cons a l')
    -/
    rw [← this, mem_cons, or_iff_not_imp_right] at hal
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a : α
      l' : List α
      h : And (Membership.mem l a) (And (Not (Membership.mem l' a)) (Eq l.dedup.tail …
      this : Eq (List.cons l.dedup.head! l.dedup.tail) l.dedup
      hal : Not (Membership.mem l.dedup.tail a) → Eq a l.dedup.head!
      ⊢ Eq l.dedup (List.cons a l')
    -/
    exact this ▸ h.2.2.symm ▸ cons_eq_cons.2 ⟨(hal (h.2.2.symm ▸ h.2.1)).symm, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem dedup_eq_nil (l : List α) : l.dedup = [] ↔ l = [] := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    ⊢ Iff (Eq l.dedup List.nil) (Eq l List.nil)
  -/
  induction' l with a l hl
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      ⊢ Iff (Eq List.nil.dedup List.nil) (Eq List.nil List.nil)
    -/
  · exact Iff.rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      hl : Iff (Eq l.dedup List.nil) (Eq l List.nil)
      ⊢ Iff (Eq (List.cons a l).dedup List.nil) (Eq (List.cons a l) List.nil)
    -/
  · by_cases h : a ∈ l
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : Iff (Eq l.dedup List.nil) (Eq l List.nil)
        h : Membership.mem l a
        ⊢ Iff (Eq (List.cons a l).dedup List.nil) (Eq (List.cons a l) List.nil)
      -/
    · simp only [List.dedup_cons_of_mem h, hl, List.ne_nil_of_mem h, reduceCtorEq]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : Iff (Eq l.dedup List.nil) (Eq l List.nil)
        h : Not (Membership.mem l a)
        ⊢ Iff (Eq (List.cons a l).dedup List.nil) (Eq (List.cons a l) List.nil)
      -/
    · simp only [List.dedup_cons_of_not_mem h, List.cons_ne_nil]
      /-
        🎉 no goals
      -/


protected theorem Nodup.dedup {l : List α} (h : l.Nodup) : l.dedup = l :=
  List.dedup_eq_self.2 h


@[simp]
theorem dedup_idem {l : List α} : dedup (dedup l) = dedup l :=
  pwFilter_idem


theorem dedup_append (l₁ l₂ : List α) : dedup (l₁ ++ l₂) = l₁ ∪ dedup l₂ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l₁ l₂ : List α
    ⊢ Eq (HAppend.hAppend l₁ l₂).dedup (Union.union l₁ l₂.dedup)
  -/
  induction' l₁ with a l₁ IH; · rfl
                                /-
                                  🎉 no goals
                                -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    l₂ : List α
    a : α
    l₁ : List α
    IH : Eq (HAppend.hAppend l₁ l₂).dedup (Union.union l₁ l₂.dedup)
    ⊢ Eq (HAppend.hAppend (List.cons a l₁) l₂).dedup (Union.union (List.cons a l₁) …
  -/
  simp only [cons_union] at *
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    l₂ : List α
    a : α
    l₁ : List α
    IH : Eq (HAppend.hAppend l₁ l₂).dedup (Union.union l₁ l₂.dedup)
    ⊢ Eq (HAppend.hAppend (List.cons a l₁) l₂).dedup (List.insert a (Union.union l …
  -/
  rw [← IH, cons_append]
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    l₂ : List α
    a : α
    l₁ : List α
    IH : Eq (HAppend.hAppend l₁ l₂).dedup (Union.union l₁ l₂.dedup)
    ⊢ Eq (List.cons a (HAppend.hAppend l₁ l₂)).dedup (List.insert a (HAppend.hAppe …
  -/
  by_cases h : a ∈ dedup (l₁ ++ l₂)
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      l₂ : List α
      a : α
      l₁ : List α
      IH : Eq (HAppend.hAppend l₁ l₂).dedup (Union.union l₁ l₂.dedup)
      h : Membership.mem (HAppend.hAppend l₁ l₂).dedup a
      ⊢ Eq (List.cons a (HAppend.hAppend l₁ l₂)).dedup (List.insert a (HAppend.hAppe …
    -/
  · rw [dedup_cons_of_mem' h, insert_of_mem h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      l₂ : List α
      a : α
      l₁ : List α
      IH : Eq (HAppend.hAppend l₁ l₂).dedup (Union.union l₁ l₂.dedup)
      h : Not (Membership.mem (HAppend.hAppend l₁ l₂).dedup a)
      ⊢ Eq (List.cons a (HAppend.hAppend l₁ l₂)).dedup (List.insert a (HAppend.hAppe …
    -/
  · rw [dedup_cons_of_not_mem' h, insert_of_not_mem h]
    /-
      🎉 no goals
    -/


theorem dedup_map_of_injective [DecidableEq β] {f : α → β} (hf : Function.Injective f)
    (xs : List α) :
    (xs.map f).dedup = xs.dedup.map f := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    rw [map_cons]
    by_cases h : x ∈ xs
    · rw [dedup_cons_of_mem h, dedup_cons_of_mem (mem_map_of_mem f h), ih]
    · rw [dedup_cons_of_not_mem h, dedup_cons_of_not_mem <| (mem_map_of_injective hf).not.mpr h, ih,
        map_cons]


/-- Note that the weaker `List.Subset.dedup_append_left` is proved later. -/
theorem Subset.dedup_append_right {xs ys : List α} (h : xs ⊆ ys) :
    dedup (xs ++ ys) = dedup ys := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    h : HasSubset.Subset xs ys
    ⊢ Eq (HAppend.hAppend xs ys).dedup ys.dedup
  -/
  rw [List.dedup_append, Subset.union_eq_right (h.trans <| subset_dedup _)]
  /-
    🎉 no goals
  -/


theorem Disjoint.union_eq {xs ys : List α} (h : Disjoint xs ys) :
    xs ∪ ys = xs.dedup ++ ys := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    rw [cons_union]
    rw [disjoint_cons_left] at h
    by_cases hx : x ∈ xs
    · rw [dedup_cons_of_mem hx, insert_of_mem (mem_union_left hx _), ih h.2]
    · rw [dedup_cons_of_not_mem hx, insert_of_not_mem, ih h.2, cons_append]
      rw [mem_union_iff, not_or]
      exact ⟨hx, h.1⟩


theorem Disjoint.dedup_append {xs ys : List α} (h : Disjoint xs ys) :
    dedup (xs ++ ys) = dedup xs ++ dedup ys := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    h : xs.Disjoint ys
    ⊢ Eq (HAppend.hAppend xs ys).dedup (HAppend.hAppend xs.dedup ys.dedup)
  -/
  rw [List.dedup_append, Disjoint.union_eq]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    h : xs.Disjoint ys
    ⊢ xs.Disjoint ys.dedup
  -/
  intro a hx hy
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    h : xs.Disjoint ys
    a : α
    hx : Membership.mem xs a
    hy : Membership.mem ys.dedup a
    ⊢ False
  -/
  exact h hx (mem_dedup.mp hy)
  /-
    🎉 no goals
  -/


theorem replicate_dedup {x : α} : ∀ {k}, k ≠ 0 → (replicate k x).dedup = [x]
  | 0, h => (h rfl).elim
  | 1, _ => rfl
  | n + 2, _ => by
    rw [replicate_succ, dedup_cons_of_mem (mem_replicate.2 ⟨n.succ_ne_zero, rfl⟩),
      replicate_dedup n.succ_ne_zero]


theorem count_dedup (l : List α) (a : α) : l.dedup.count a = if a ∈ l then 1 else 0 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    a : α
    ⊢ Eq (List.count a l.dedup) (ite (Membership.mem l a) 1 0)
  -/
  simp_rw [count_eq_of_nodup <| nodup_dedup l, mem_dedup]
  /-
    🎉 no goals
  -/


theorem Perm.dedup {l₁ l₂ : List α} (p : l₁ ~ l₂) : dedup l₁ ~ dedup l₂ :=
  perm_iff_count.2 fun a =>
    if h : a ∈ l₁ then by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        l₁ l₂ : List α
        p : l₁.Perm l₂
        a : α
        h : Membership.mem l₁ a
        ⊢ Eq (List.count a l₁.dedup) (List.count a l₂.dedup)
      -/
      simp [h, nodup_dedup, p.subset h]
      /-
        🎉 no goals
      -/
    else by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        l₁ l₂ : List α
        p : l₁.Perm l₂
        a : α
        h : Not (Membership.mem l₁ a)
        ⊢ Eq (List.count a l₁.dedup) (List.count a l₂.dedup)
      -/
      simp [h, count_eq_zero_of_not_mem, mt p.mem_iff.2]
      /-
        🎉 no goals
      -/


