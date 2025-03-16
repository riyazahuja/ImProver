/--
Split a list to chunks of given lengths.
-/
def splitLengths : List ℕ → List α → List (List α)
  | [], _ => []
  | n::ns, x =>
    let (x0, x1) := x.splitAt n
    x0 :: ns.splitLengths x1


@[simp]
theorem length_splitLengths : (sz.splitLengths l).length = sz.length := by
  /-
    α : Type u_1
    l : List α
    sz : List Nat
    ⊢ Eq (sz.splitLengths l).length sz.length
  -/
                                  /-
                                    🎉 no goals
                                  -/
  induction sz generalizing l <;> simp [splitLengths, *]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
lemma splitLengths_nil : [].splitLengths l = [] := rfl


@[simp]
lemma splitLengths_cons (n : ℕ) :
    (n :: sz).splitLengths l = l.take n :: sz.splitLengths (l.drop n) := by
  /-
    α : Type u_1
    l : List α
    sz : List Nat
    n : Nat
    ⊢ Eq ((List.cons n sz).splitLengths l) (List.cons (List.take n l) (sz.splitLen …
  -/
  simp [splitLengths]
  /-
    🎉 no goals
  -/


theorem take_splitLength (i : ℕ) : (sz.splitLengths l).take i = (sz.take i).splitLengths l := by
  /-
    α : Type u_1
    l : List α
    sz : List Nat
    i : Nat
    ⊢ Eq (List.take i (sz.splitLengths l)) ((List.take i sz).splitLengths l)
  -/
  induction i generalizing sz l
  /-
    case zero
    α : Type u_1
    l : List α
    sz : List Nat
    ⊢ Eq (List.take 0 (sz.splitLengths l)) ((List.take 0 sz).splitLengths l)
  -/
  case zero => simp
  case succ i hi =>
    cases sz
    · simp
    · simp only [splitLengths_cons, take_succ_cons, cons.injEq, true_and, hi]


theorem length_splitLengths_getElem_le {i : ℕ} {hi : i < (sz.splitLengths l).length} :
                                              /-
                                                α : Type u_1
                                                l : List α
                                                sz : List Nat
                                                i : Nat
                                                hi : LT.lt i (sz.splitLengths l).length
                                                ⊢ LT.lt i sz.length
                                              -/
    (sz.splitLengths l)[i].length ≤ sz[i]'(by simpa using hi) := by
                                              /-
                                                🎉 no goals
                                              -/
  /-
    α : Type u_1
    l : List α
    sz : List Nat
    i : Nat
    hi : LT.lt i (sz.splitLengths l).length
    ⊢ LE.le (GetElem.getElem (sz.splitLengths l) i hi).length (GetElem.getElem sz  …
  -/
  induction sz generalizing l i
    /-
      case nil
      α : Type u_1
      l : List α
      i : Nat
      hi : LT.lt i (List.nil.splitLengths l).length
      ⊢ LE.le (GetElem.getElem (List.nil.splitLengths l) i hi).length (GetElem.getEl …
    -/
  · simp at hi
    /-
      🎉 no goals
    -/
  case cons head tail tail_ih =>
    simp only [splitLengths_cons]
    cases i
    · simp
    · simp only [getElem_cons_succ, tail_ih]


theorem flatten_splitLengths (h : l.length ≤ sz.sum) : (sz.splitLengths l).flatten = l := by
  /-
    α : Type u_1
    l : List α
    sz : List Nat
    h : LE.le l.length sz.sum
    ⊢ Eq (sz.splitLengths l).flatten l
  -/
  induction sz generalizing l
    /-
      case nil
      α : Type u_1
      l : List α
      h : LE.le l.length List.nil.sum
      ⊢ Eq (List.nil.splitLengths l).flatten l
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  case cons head tail ih =>
    simp only [splitLengths_cons, flatten_cons]
    rw [ih, take_append_drop]
    simpa [add_comm] using h


theorem map_splitLengths_length (h : sz.sum ≤ l.length) :
    (sz.splitLengths l).map length = sz := by
  /-
    α : Type u_1
    l : List α
    sz : List Nat
    h : LE.le sz.sum l.length
    ⊢ Eq (List.map List.length (sz.splitLengths l)) sz
  -/
  induction sz generalizing l
    /-
      case nil
      α : Type u_1
      l : List α
      h : LE.le List.nil.sum l.length
      ⊢ Eq (List.map List.length (List.nil.splitLengths l)) List.nil
    -/
  · simp
    /-
      🎉 no goals
    -/
  case cons head tail ih =>
    simp only [sum_cons] at h
    simp only [splitLengths_cons, map_cons, length_take, cons.injEq, min_eq_left_iff]
    rw [ih]
    · simp [Nat.le_of_add_right_le h]
    · simp [Nat.le_sub_of_add_le' h]


theorem length_splitLengths_getElem_eq {i : ℕ} (hi : i < sz.length)
    (h : (sz.take (i + 1)).sum ≤ l.length) :
                                /-
                                  α : Type u_1
                                  l : List α
                                  sz : List Nat
                                  i : Nat
                                  hi : LT.lt i sz.length
                                  h : LE.le (List.take (HAdd.hAdd i 1) sz).sum l.length
                                  ⊢ LT.lt i (sz.splitLengths l).length
                                -/
    ((sz.splitLengths l)[i]'(by simpa)).length = sz[i] := by
                                /-
                                  🎉 no goals
                                -/
  /-
    α : Type u_1
    l : List α
    sz : List Nat
    i : Nat
    hi : LT.lt i sz.length
    h : LE.le (List.take (HAdd.hAdd i 1) sz).sum l.length
    ⊢ Eq (GetElem.getElem (sz.splitLengths l) i ⋯).length (GetElem.getElem sz i hi)
  -/
  rw [List.getElem_take' (hj := i.lt_add_one)]
  /-
    α : Type u_1
    l : List α
    sz : List Nat
    i : Nat
    hi : LT.lt i sz.length
    h : LE.le (List.take (HAdd.hAdd i 1) sz).sum l.length
    ⊢ Eq (GetElem.getElem (List.take (HAdd.hAdd i 1) (sz.splitLengths l)) i ⋯).len …
  -/
  simp only [take_splitLength]
  conv_rhs =>
    rw [List.getElem_take' (hj := i.lt_add_one)]
    simp (config := {singlePass := true}) only [← map_splitLengths_length l _ h]
    rw [getElem_map]


theorem splitLengths_length_getElem {α : Type*} (l : List α) (sz : List ℕ)
    (h : sz.sum ≤ l.length) (i : ℕ) (hi : i < (sz.splitLengths l).length) :
                                              /-
                                                α✝ : Type u_1
                                                l✝ : List α✝
                                                sz✝ : List Nat
                                                α : Type u_2
                                                l : List α
                                                sz : List Nat
                                                h : LE.le sz.sum l.length
                                                i : Nat
                                                hi : LT.lt i (sz.splitLengths l).length
                                                ⊢ LT.lt i sz.length
                                              -/
    (sz.splitLengths l)[i].length = sz[i]'(by simpa using hi) := by
                                              /-
                                                🎉 no goals
                                              -/
  /-
    α : Type u_2
    l : List α
    sz : List Nat
    h : LE.le sz.sum l.length
    i : Nat
    hi : LT.lt i (sz.splitLengths l).length
    ⊢ Eq (GetElem.getElem (sz.splitLengths l) i hi).length (GetElem.getElem sz i ⋯)
  -/
  have := map_splitLengths_length l sz h
  /-
    α : Type u_2
    l : List α
    sz : List Nat
    h : LE.le sz.sum l.length
    i : Nat
    hi : LT.lt i (sz.splitLengths l).length
    this : Eq (List.map List.length (sz.splitLengths l)) sz
    ⊢ Eq (GetElem.getElem (sz.splitLengths l) i hi).length (GetElem.getElem sz i ⋯)
  -/
  rw [← List.getElem_map List.length]
    /-
      α : Type u_2
      l : List α
      sz : List Nat
      h : LE.le sz.sum l.length
      i : Nat
      hi : LT.lt i (sz.splitLengths l).length
      this : Eq (List.map List.length (sz.splitLengths l)) sz
      ⊢ Eq (GetElem.getElem (List.map List.length (sz.splitLengths l)) i ?m.7430) (G …
    -/
  · simp [this]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_2
      l : List α
      sz : List Nat
      h : LE.le sz.sum l.length
      i : Nat
      hi : LT.lt i (sz.splitLengths l).length
      this : Eq (List.map List.length (sz.splitLengths l)) sz
      ⊢ LT.lt i (List.map List.length (sz.splitLengths l)).length
    -/
  · simpa using hi
    /-
      🎉 no goals
    -/


theorem length_mem_splitLengths {α : Type*} (l : List α) (sz : List ℕ) (b : ℕ)
    (h : ∀ n ∈ sz, n ≤ b) : ∀ l₂ ∈ sz.splitLengths l, l₂.length ≤ b := by
  /-
    α : Type u_2
    l : List α
    sz : List Nat
    b : Nat
    h : ∀ (n : Nat), Membership.mem sz n → LE.le n b
    ⊢ ∀ (l₂ : List α), Membership.mem (sz.splitLengths l) l₂ → LE.le l₂.length b
  -/
  rw [← List.forall_getElem]
  /-
    α : Type u_2
    l : List α
    sz : List Nat
    b : Nat
    h : ∀ (n : Nat), Membership.mem sz n → LE.le n b
    ⊢ ∀ (n : Nat) (h : LT.lt n (sz.splitLengths l).length), LE.le (GetElem.getElem …
  -/
  intro i hi
  /-
    α : Type u_2
    l : List α
    sz : List Nat
    b : Nat
    h : ∀ (n : Nat), Membership.mem sz n → LE.le n b
    i : Nat
    hi : LT.lt i (sz.splitLengths l).length
    ⊢ LE.le (GetElem.getElem (sz.splitLengths l) i hi).length b
  -/
  have := length_splitLengths_getElem_le l sz (hi := hi)
  /-
    α : Type u_2
    l : List α
    sz : List Nat
    b : Nat
    h : ∀ (n : Nat), Membership.mem sz n → LE.le n b
    i : Nat
    hi : LT.lt i (sz.splitLengths l).length
    this : LE.le (GetElem.getElem (sz.splitLengths l) i hi).length (GetElem.getEle …
    ⊢ LE.le (GetElem.getElem (sz.splitLengths l) i hi).length b
  -/
  have := h (sz[i]'(by simpa using hi)) (getElem_mem ..)
  /-
    α : Type u_2
    l : List α
    sz : List Nat
    b : Nat
    h : ∀ (n : Nat), Membership.mem sz n → LE.le n b
    i : Nat
    hi : LT.lt i (sz.splitLengths l).length
    this✝ : LE.le (GetElem.getElem (sz.splitLengths l) i hi).length (GetElem.getEl …
    this : LE.le (GetElem.getElem sz i ⋯) b
    ⊢ LE.le (GetElem.getElem (sz.splitLengths l) i hi).length b
  -/
  omega
  /-
    🎉 no goals
  -/


