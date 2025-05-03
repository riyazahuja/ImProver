set_option linter.deprecated false in
/-- See `List.length_flatten` for the corresponding statement using `List.sum`. -/
@[deprecated length_flatten (since := "2024-10-17")]
lemma length_flatten' (L : List (List α)) : length (flatten L) = Nat.sum (map length L) := by
  /-
    α : Type u_1
    L : List (List α)
    ⊢ Eq L.flatten.length (Nat.sum (List.map List.length L))
  -/
  induction L <;> [rfl; simp only [*, flatten, map, Nat.sum_cons, length_append]]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-25")] alias length_join' := length_flatten'


set_option linter.deprecated false in
/-- See `List.countP_flatten` for the corresponding statement using `List.sum`. -/
@[deprecated countP_flatten (since := "2024-10-17")]
lemma countP_flatten' (p : α → Bool) :
    ∀ L : List (List α), countP p L.flatten = Nat.sum (L.map (countP p))
  | [] => rfl
                 /-
                   α : Type u_1
                   p : α → Bool
                   a : List α
                   l : List (List α)
                   ⊢ Eq (List.countP p (List.cons a l).flatten) (Nat.sum (List.map (List.countP p …
                 -/
  | a :: l => by rw [flatten, countP_append, map_cons, Nat.sum_cons, countP_flatten' _ l]
                 /-
                   🎉 no goals
                 -/


@[deprecated (since := "2024-10-25")] alias countP_join' := countP_flatten'


set_option linter.deprecated false in
/-- See `List.count_flatten` for the corresponding statement using `List.sum`. -/
@[deprecated count_flatten (since := "2024-10-17")]
lemma count_flatten' [BEq α] (L : List (List α)) (a : α) :
    L.flatten.count a = Nat.sum (L.map (count a)) := countP_flatten' _ _


@[deprecated (since := "2024-10-25")] alias count_join' := count_flatten'


set_option linter.deprecated false in
/-- See `List.length_flatMap` for the corresponding statement using `List.sum`. -/
@[deprecated "Use `List.length_flatMap`." (since := "2024-10-17")]
lemma length_flatMap' (l : List α) (f : α → List β) :
    length (l.flatMap f) = Nat.sum (map (length ∘ f) l) := by
  /-
    α : Type u_1
    β : Type u_2
    l : List α
    f : α → List β
    ⊢ Eq (l.flatMap f).length (Nat.sum (List.map (Function.comp List.length f) l))
  -/
  rw [List.flatMap, length_flatten', map_map]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-16")] alias length_bind' := length_flatMap'


set_option linter.deprecated false in
/-- See `List.countP_flatMap` for the corresponding statement using `List.sum`. -/
@[deprecated "Use `List.countP_flatMap`." (since := "2024-10-17")]
lemma countP_flatMap' (p : β → Bool) (l : List α) (f : α → List β) :
    countP p (l.flatMap f) = Nat.sum (map (countP p ∘ f) l) := by
  /-
    α : Type u_1
    β : Type u_2
    p : β → Bool
    l : List α
    f : α → List β
    ⊢ Eq (List.countP p (l.flatMap f)) (Nat.sum (List.map (Function.comp (List.cou …
  -/
  rw [List.flatMap, countP_flatten', map_map]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-16")] alias countP_bind' := countP_flatMap'


set_option linter.deprecated false in
/-- See `List.count_flatMap` for the corresponding statement using `List.sum`. -/
@[deprecated "Use `List.count_flatMap`." (since := "2024-10-17")]
lemma count_flatMap' [BEq β] (l : List α) (f : α → List β) (x : β) :
    count x (l.flatMap f) = Nat.sum (map (count x ∘ f) l) := countP_flatMap' _ _ _


@[deprecated (since := "2024-10-16")] alias count_bind' := count_flatMap'


set_option linter.deprecated false in
/-- In a join, taking the first elements up to an index which is the sum of the lengths of the
first `i` sublists, is the same as taking the join of the first `i` sublists.

See `List.take_sum_flatten` for the corresponding statement using `List.sum`. -/
@[deprecated "Use `List.take_sum_flatten`." (since := "2024-10-17")]
theorem take_sum_flatten' (L : List (List α)) (i : ℕ) :
    L.flatten.take (Nat.sum ((L.map length).take i)) = (L.take i).flatten := by
  /-
    α : Type u_1
    L : List (List α)
    i : Nat
    ⊢ Eq (List.take (Nat.sum (List.take i (List.map List.length L))) L.flatten) (L …
  -/
  induction L generalizing i
    /-
      case nil
      α : Type u_1
      i : Nat
      ⊢ Eq (List.take (Nat.sum (List.take i (List.map List.length List.nil))) List.n …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      head✝ : List α
      tail✝ : List (List α)
      tail_ih✝ : ∀ (i : Nat), Eq (List.take (Nat.sum (List.take i (List.map List.len …
      i : Nat
      ⊢ Eq (List.take (Nat.sum (List.take i (List.map List.length (List.cons head✝ t …
    -/
                /-
                  🎉 no goals
                -/
  · cases i <;> simp [take_append, *]
                /-
                  🎉 no goals
                -/


@[deprecated (since := "2024-10-25")] alias take_sum_join' := take_sum_flatten'


set_option linter.deprecated false in
/-- In a join, dropping all the elements up to an index which is the sum of the lengths of the
first `i` sublists, is the same as taking the join after dropping the first `i` sublists.

See `List.drop_sum_flatten` for the corresponding statement using `List.sum`. -/
@[deprecated "Use `List.drop_sum_flatten`." (since := "2024-10-17")]
theorem drop_sum_flatten' (L : List (List α)) (i : ℕ) :
    L.flatten.drop (Nat.sum ((L.map length).take i)) = (L.drop i).flatten := by
  /-
    α : Type u_1
    L : List (List α)
    i : Nat
    ⊢ Eq (List.drop (Nat.sum (List.take i (List.map List.length L))) L.flatten) (L …
  -/
  induction L generalizing i
    /-
      case nil
      α : Type u_1
      i : Nat
      ⊢ Eq (List.drop (Nat.sum (List.take i (List.map List.length List.nil))) List.n …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      head✝ : List α
      tail✝ : List (List α)
      tail_ih✝ : ∀ (i : Nat), Eq (List.drop (Nat.sum (List.take i (List.map List.len …
      i : Nat
      ⊢ Eq (List.drop (Nat.sum (List.take i (List.map List.length (List.cons head✝ t …
    -/
                /-
                  🎉 no goals
                -/
  · cases i <;> simp [drop_append, *]
                /-
                  🎉 no goals
                -/


@[deprecated (since := "2024-10-25")] alias drop_sum_join' := drop_sum_flatten'


/-- Taking only the first `i+1` elements in a list, and then dropping the first `i` ones, one is
left with a list of length `1` made of the `i`-th element of the original list. -/
theorem drop_take_succ_eq_cons_getElem (L : List α) (i : Nat) (h : i < L.length) :
    (L.take (i + 1)).drop i = [L[i]] := by
  /-
    α : Type u_1
    L : List α
    i : Nat
    h : LT.lt i L.length
    ⊢ Eq (List.drop i (List.take (HAdd.hAdd i 1) L)) (List.cons (GetElem.getElem L …
  -/
  induction' L with head tail ih generalizing i
    /-
      case nil
      α : Type u_1
      i : Nat
      h : LT.lt i List.nil.length
      ⊢ Eq (List.drop i (List.take (HAdd.hAdd i 1) List.nil)) (List.cons (GetElem.ge …
    -/
  · exact (Nat.not_succ_le_zero i h).elim
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    head : α
    tail : List α
    ih : ∀ (i : Nat) (h : LT.lt i tail.length), Eq (List.drop i (List.take (HAdd.h …
    i : Nat
    h : LT.lt i (List.cons head tail).length
    ⊢ Eq (List.drop i (List.take (HAdd.hAdd i 1) (List.cons head tail))) (List.con …
  -/
  rcases i with _ | i
    /-
      case cons.zero
      α : Type u_1
      head : α
      tail : List α
      ih : ∀ (i : Nat) (h : LT.lt i tail.length), Eq (List.drop i (List.take (HAdd.h …
      h : LT.lt 0 (List.cons head tail).length
      ⊢ Eq (List.drop 0 (List.take (HAdd.hAdd 0 1) (List.cons head tail))) (List.con …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons.succ
      α : Type u_1
      head : α
      tail : List α
      ih : ∀ (i : Nat) (h : LT.lt i tail.length), Eq (List.drop i (List.take (HAdd.h …
      i : Nat
      h : LT.lt (HAdd.hAdd i 1) (List.cons head tail).length
      ⊢ Eq (List.drop (HAdd.hAdd i 1) (List.take (HAdd.hAdd (HAdd.hAdd i 1) 1) (List …
    -/
  · simpa using ih _ (by simpa using h)
    /-
      🎉 no goals
    -/


@[deprecated drop_take_succ_eq_cons_getElem (since := "2024-06-11")]
theorem drop_take_succ_eq_cons_get (L : List α) (i : Fin L.length) :
    (L.take (i + 1)).drop i = [get L i] := by
  /-
    α : Type u_1
    L : List α
    i : Fin L.length
    ⊢ Eq (List.drop (↑i) (List.take (HAdd.hAdd (↑i) 1) L)) (List.cons (L.get i) Li …
  -/
  simp [drop_take_succ_eq_cons_getElem]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
/-- In a flatten of sublists, taking the slice between the indices `A` and `B - 1` gives back the
original sublist of index `i` if `A` is the sum of the lengths of sublists of index `< i`, and
`B` is the sum of the lengths of sublists of index `≤ i`.

See `List.drop_take_succ_flatten_eq_getElem` for the corresponding statement using `List.sum`. -/
@[deprecated "Use `List.drop_take_succ_flatten_eq_getElem`." (since := "2024-10-17")]
theorem drop_take_succ_flatten_eq_getElem' (L : List (List α)) (i : Nat) (h : i <  L.length) :
    (L.flatten.take (Nat.sum ((L.map length).take (i + 1)))).drop
      (Nat.sum ((L.map length).take i)) = L[i] := by
  have : (L.map length).take i = ((L.take (i + 1)).map length).take i := by
    simp [map_take, take_take, Nat.min_eq_left]
  simp only [this, length_map, take_sum_flatten', drop_sum_flatten',
    drop_take_succ_eq_cons_getElem, h, flatten, append_nil]


@[deprecated (since := "2024-10-15")]
alias drop_take_succ_join_eq_getElem' := drop_take_succ_flatten_eq_getElem'


set_option linter.deprecated false in
@[deprecated drop_take_succ_flatten_eq_getElem' (since := "2024-06-11")]
theorem drop_take_succ_join_eq_get' (L : List (List α)) (i : Fin L.length) :
    (L.flatten.take (Nat.sum ((L.map length).take (i + 1)))).drop
      (Nat.sum ((L.map length).take i)) = get L i := by
   /-
     α : Type u_1
     L : List (List α)
     i : Fin L.length
     ⊢ Eq (List.drop (Nat.sum (List.take (↑i) (List.map List.length L))) (List.take …
   -/
   simp [drop_take_succ_flatten_eq_getElem']
   /-
     🎉 no goals
   -/


theorem flatten_drop_length_sub_one {L : List (List α)} (h : L ≠ []) :
    (L.drop (L.length - 1)).flatten = L.getLast h := by
  /-
    α : Type u_1
    L : List (List α)
    h : Ne L List.nil
    ⊢ Eq (List.drop (HSub.hSub L.length 1) L).flatten (L.getLast h)
  -/
  induction L using List.reverseRecOn
    /-
      case nil
      α : Type u_1
      h : Ne List.nil List.nil
      ⊢ Eq (List.drop (HSub.hSub List.nil.length 1) List.nil).flatten (List.nil.getL …
    -/
  · cases h rfl
    /-
      🎉 no goals
    -/
    /-
      case append_singleton
      α : Type u_1
      l✝ : List (List α)
      a✝¹ : List α
      a✝ : ∀ (h : Ne l✝ List.nil), Eq (List.drop (HSub.hSub l✝.length 1) l✝).flatten …
      h : Ne (HAppend.hAppend l✝ (List.cons a✝¹ List.nil)) List.nil
      ⊢ Eq (List.drop (HSub.hSub (HAppend.hAppend l✝ (List.cons a✝¹ List.nil)).lengt …
    -/
  · simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-15")] alias join_drop_length_sub_one := flatten_drop_length_sub_one


/-- We can rebracket `x ++ (l₁ ++ x) ++ (l₂ ++ x) ++ ... ++ (lₙ ++ x)` to
`(x ++ l₁) ++ (x ++ l₂) ++ ... ++ (x ++ lₙ) ++ x` where `L = [l₁, l₂, ..., lₙ]`. -/
theorem append_flatten_map_append (L : List (List α)) (x : List α) :
    x ++ (L.map (· ++ x)).flatten = (L.map (x ++ ·)).flatten ++ x := by
  induction L with
  | nil => rw [map_nil, flatten, append_nil, map_nil, flatten, nil_append]
  | cons _ _ ih =>
    rw [map_cons, flatten, map_cons, flatten, append_assoc, ih, append_assoc, append_assoc]


@[deprecated (since := "2024-10-15")] alias append_join_map_append := append_flatten_map_append


@[deprecated "No deprecation message was provided." (since := "2024-08-15")]
theorem sublist_join {l} {L : List (List α)} (h : l ∈ L) :
    l <+ L.flatten :=
  sublist_flatten_of_mem h


