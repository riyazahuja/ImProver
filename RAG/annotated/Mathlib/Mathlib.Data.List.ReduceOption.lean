@[simp]
theorem reduceOption_cons_of_some (x : α) (l : List (Option α)) :
    reduceOption (some x :: l) = x :: l.reduceOption := by
  /-
    α : Type u_1
    x : α
    l : List (Option α)
    ⊢ Eq (List.cons (Option.some x) l).reduceOption (List.cons x l.reduceOption)
  -/
  simp only [reduceOption, filterMap, id, eq_self_iff_true, and_self_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem reduceOption_cons_of_none (l : List (Option α)) :
                                                    /-
                                                      α : Type u_1
                                                      l : List (Option α)
                                                      ⊢ Eq (List.cons Option.none l).reduceOption l.reduceOption
                                                    -/
    reduceOption (none :: l) = l.reduceOption := by simp only [reduceOption, filterMap, id]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem reduceOption_nil : @reduceOption α [] = [] :=
  rfl


@[simp]
theorem reduceOption_map {l : List (Option α)} {f : α → β} :
    reduceOption (map (Option.map f) l) = map f (reduceOption l) := by
  /-
    α : Type u_1
    β : Type u_2
    l : List (Option α)
    f : α → β
    ⊢ Eq (List.map (Option.map f) l).reduceOption (List.map f l.reduceOption)
  -/
  induction' l with hd tl hl
    /-
      case nil
      α : Type u_1
      β : Type u_2
      f : α → β
      ⊢ Eq (List.map (Option.map f) List.nil).reduceOption (List.map f List.nil.redu …
    -/
  · simp only [reduceOption_nil, map_nil]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      β : Type u_2
      f : α → β
      hd : Option α
      tl : List (Option α)
      hl : Eq (List.map (Option.map f) tl).reduceOption (List.map f tl.reduceOption)
      ⊢ Eq (List.map (Option.map f) (List.cons hd tl)).reduceOption (List.map f (Lis …
    -/
  · cases hd <;>
      /-
        case cons.none
        α : Type u_1
        β : Type u_2
        f : α → β
        tl : List (Option α)
        hl : Eq (List.map (Option.map f) tl).reduceOption (List.map f tl.reduceOption)
        ⊢ Eq (List.map (Option.map f) (List.cons Option.none tl)).reduceOption (List.m …
      -/
      /-
        🎉 no goals
      -/
      simpa [Option.map_some', map, eq_self_iff_true, reduceOption_cons_of_some] using hl
      /-
        🎉 no goals
      -/


theorem reduceOption_append (l l' : List (Option α)) :
    (l ++ l').reduceOption = l.reduceOption ++ l'.reduceOption :=
  filterMap_append l l' id


theorem reduceOption_length_eq {l : List (Option α)} :
    l.reduceOption.length = (l.filter Option.isSome).length := by
  /-
    α : Type u_1
    l : List (Option α)
    ⊢ Eq l.reduceOption.length (List.filter Option.isSome l).length
  -/
  induction' l with hd tl hl
    /-
      case nil
      α : Type u_1
      ⊢ Eq List.nil.reduceOption.length (List.filter Option.isSome List.nil).length
    -/
  · simp_rw [reduceOption_nil, filter_nil, length]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      hd : Option α
      tl : List (Option α)
      hl : Eq tl.reduceOption.length (List.filter Option.isSome tl).length
      ⊢ Eq (List.cons hd tl).reduceOption.length (List.filter Option.isSome (List.co …
    -/
                 /-
                   🎉 no goals
                 -/
  · cases hd <;> simp [hl]
                 /-
                   🎉 no goals
                 -/


theorem length_eq_reduceOption_length_add_filter_none {l : List (Option α)} :
    l.length = l.reduceOption.length + (l.filter Option.isNone).length := by
  /-
    α : Type u_1
    l : List (Option α)
    ⊢ Eq l.length (HAdd.hAdd l.reduceOption.length (List.filter Option.isNone l).l …
  -/
  simp_rw [reduceOption_length_eq, l.length_eq_length_filter_add Option.isSome, Option.bnot_isSome]
  /-
    🎉 no goals
  -/


theorem reduceOption_length_le (l : List (Option α)) : l.reduceOption.length ≤ l.length := by
  /-
    α : Type u_1
    l : List (Option α)
    ⊢ LE.le l.reduceOption.length l.length
  -/
  rw [length_eq_reduceOption_length_add_filter_none]
  /-
    α : Type u_1
    l : List (Option α)
    ⊢ LE.le l.reduceOption.length (HAdd.hAdd l.reduceOption.length (List.filter Op …
  -/
  apply Nat.le_add_right
  /-
    🎉 no goals
  -/


theorem reduceOption_length_eq_iff {l : List (Option α)} :
    l.reduceOption.length = l.length ↔ ∀ x ∈ l, Option.isSome x := by
  /-
    α : Type u_1
    l : List (Option α)
    ⊢ Iff (Eq l.reduceOption.length l.length) (∀ (x : Option α), Membership.mem l  …
  -/
  rw [reduceOption_length_eq, List.filter_length_eq_length]
  /-
    🎉 no goals
  -/


theorem reduceOption_length_lt_iff {l : List (Option α)} :
    l.reduceOption.length < l.length ↔ none ∈ l := by
  rw [Nat.lt_iff_le_and_ne, and_iff_right (reduceOption_length_le l), Ne,
    reduceOption_length_eq_iff]
  /-
    α : Type u_1
    l : List (Option α)
    ⊢ Iff (Not (∀ (x : Option α), Membership.mem l x → Eq x.isSome Bool.true)) (Me …
  -/
                  /-
                    🎉 no goals
                  -/
  induction l <;> simp
  /-
    case cons
    α : Type u_1
    head✝ : Option α
    tail✝ : List (Option α)
    tail_ih✝ : Iff (Not (∀ (x : Option α), Membership.mem tail✝ x → Eq x.isSome Bo …
    ⊢ Iff (Eq head✝.isSome Bool.true → Membership.mem tail✝ Option.none) (Or (Eq O …
  -/
  rw [@eq_comm _ none, ← Option.not_isSome_iff_eq_none, Decidable.imp_iff_not_or]
  /-
    🎉 no goals
  -/


                                                                                  /-
                                                                                    α : Type u_1
                                                                                    x : Option α
                                                                                    ⊢ Eq (List.cons x List.nil).reduceOption x.toList
                                                                                  -/
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
theorem reduceOption_singleton (x : Option α) : [x].reduceOption = x.toList := by cases x <;> rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


theorem reduceOption_concat (l : List (Option α)) (x : Option α) :
    (l.concat x).reduceOption = l.reduceOption ++ x.toList := by
  /-
    α : Type u_1
    l : List (Option α)
    x : Option α
    ⊢ Eq (l.concat x).reduceOption (HAppend.hAppend l.reduceOption x.toList)
  -/
  induction' l with hd tl hl generalizing x
    /-
      case nil
      α : Type u_1
      x : Option α
      ⊢ Eq (List.nil.concat x).reduceOption (HAppend.hAppend List.nil.reduceOption x …
    -/
                /-
                  🎉 no goals
                -/
  · cases x <;> simp [Option.toList]
                /-
                  🎉 no goals
                -/
    /-
      case cons
      α : Type u_1
      hd : Option α
      tl : List (Option α)
      hl : ∀ (x : Option α), Eq (tl.concat x).reduceOption (HAppend.hAppend tl.reduc …
      x : Option α
      ⊢ Eq ((List.cons hd tl).concat x).reduceOption (HAppend.hAppend (List.cons hd  …
    -/
  · simp only [concat_eq_append, reduceOption_append] at hl
    /-
      case cons
      α : Type u_1
      hd : Option α
      tl : List (Option α)
      x : Option α
      hl : ∀ (x : Option α), Eq (HAppend.hAppend tl.reduceOption (List.cons x List.n …
      ⊢ Eq ((List.cons hd tl).concat x).reduceOption (HAppend.hAppend (List.cons hd  …
    -/
                 /-
                   🎉 no goals
                 -/
    cases hd <;> simp [hl, reduceOption_append]
                 /-
                   🎉 no goals
                 -/


theorem reduceOption_concat_of_some (l : List (Option α)) (x : α) :
    (l.concat (some x)).reduceOption = l.reduceOption.concat x := by
  /-
    α : Type u_1
    l : List (Option α)
    x : α
    ⊢ Eq (l.concat (Option.some x)).reduceOption (l.reduceOption.concat x)
  -/
  simp only [reduceOption_nil, concat_eq_append, reduceOption_append, reduceOption_cons_of_some]
  /-
    🎉 no goals
  -/


theorem reduceOption_mem_iff {l : List (Option α)} {x : α} : x ∈ l.reduceOption ↔ some x ∈ l := by
  /-
    α : Type u_1
    l : List (Option α)
    x : α
    ⊢ Iff (Membership.mem l.reduceOption x) (Membership.mem l (Option.some x))
  -/
  simp only [reduceOption, id, mem_filterMap, exists_eq_right]
  /-
    🎉 no goals
  -/


theorem reduceOption_get?_iff {l : List (Option α)} {x : α} :
    (∃ i, l.get? i = some (some x)) ↔ ∃ i, l.reduceOption.get? i = some x := by
  /-
    α : Type u_1
    l : List (Option α)
    x : α
    ⊢ Iff (Exists fun i => Eq (l.get? i) (Option.some (Option.some x))) (Exists fu …
  -/
  rw [← mem_iff_get?, ← mem_iff_get?, reduceOption_mem_iff]
  /-
    🎉 no goals
  -/


