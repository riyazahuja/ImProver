theorem getD_eq_getElem {n : ℕ} (hn : n < l.length) : l.getD n d = l[n] := by
  induction l generalizing n with
  | nil => simp at hn
  | cons head tail ih =>
    cases n
    · exact getD_cons_zero
    · exact ih _


@[deprecated getD_eq_getElem (since := "2024-08-02")]
theorem getD_eq_get {n : ℕ} (hn : n < l.length) : l.getD n d = l.get ⟨n, hn⟩ :=
  getD_eq_getElem l d hn


                                                                                     /-
                                                                                       α : Type u
                                                                                       β : Type v
                                                                                       l : List α
                                                                                       d : α
                                                                                       n : Nat
                                                                                       f : α → β
                                                                                       ⊢ Eq ((List.map f l).getD n (f d)) (f (l.getD n d))
                                                                                     -/
theorem getD_map {n : ℕ} (f : α → β) : (map f l).getD n (f d) = f (l.getD n d) := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem getD_eq_default {n : ℕ} (hn : l.length ≤ n) : l.getD n d = d := by
  induction l generalizing n with
  | nil => exact getD_nil
  | cons head tail ih =>
    cases n
    · simp at hn
    · exact ih (Nat.le_of_succ_le_succ hn)


theorem getD_reverse {l : List α} (i) (h : i < length l) :
    getD l.reverse i = getD l (l.length - 1 - i) := by
  /-
    α : Type u
    l : List α
    i : Nat
    h : LT.lt i l.length
    ⊢ Eq (l.reverse.getD i) (l.getD (HSub.hSub (HSub.hSub l.length 1) i))
  -/
  funext a
  /-
    case h
    α : Type u
    l : List α
    i : Nat
    h : LT.lt i l.length
    a : α
    ⊢ Eq (l.reverse.getD i a) (l.getD (HSub.hSub (HSub.hSub l.length 1) i) a)
  -/
  rwa [List.getD_eq_getElem?_getD, List.getElem?_reverse, ← List.getD_eq_getElem?_getD]
  /-
    🎉 no goals
  -/


/-- An empty list can always be decidably checked for the presence of an element.
Not an instance because it would clash with `DecidableEq α`. -/
def decidableGetDNilNe (a : α) : DecidablePred fun i : ℕ => getD ([] : List α) i a ≠ a :=
  fun _ => isFalse fun H => H getD_nil


@[simp]
                                                                              /-
                                                                                α : Type u
                                                                                d : α
                                                                                n : Nat
                                                                                ⊢ Eq ((GetElem?.getElem? (List.cons d List.nil) n).getD d) d
                                                                              -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
theorem getElem?_getD_singleton_default_eq (n : ℕ) : [d][n]?.getD d = d := by cases n <;> simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[deprecated (since := "2024-06-12")]
alias getD_singleton_default_eq := getElem?_getD_singleton_default_eq


@[simp]
theorem getElem?_getD_replicate_default_eq (r n : ℕ) : (replicate r d)[n]?.getD d = d := by
  induction r generalizing n with
  | zero => simp
  | succ n ih => simp at ih; cases n <;> simp [ih, replicate_succ]


@[deprecated (since := "2024-06-12")]
alias getD_replicate_default_eq := getElem?_getD_replicate_default_eq


theorem getD_replicate {y i n} (h : i < n) :
    getD (replicate n x) i y = x := by
  /-
    α : Type u
    x y : α
    i n : Nat
    h : LT.lt i n
    ⊢ Eq ((List.replicate n x).getD i y) x
  -/
  rw [getD_eq_getElem,  getElem_replicate]
  /-
    case h
    α : Type u
    x y : α
    i n : Nat
    h : LT.lt i n
    ⊢ LT.lt i (List.replicate n x).length
  -/
  rwa [length_replicate]
  /-
    🎉 no goals
  -/


theorem getD_append (l l' : List α) (d : α) (n : ℕ) (h : n < l.length) :
    (l ++ l').getD n d = l.getD n d := by
  rw [getD_eq_getElem _ _ (Nat.lt_of_lt_of_le h (length_append _ _ ▸ Nat.le_add_right _ _)),
    getElem_append_left h, getD_eq_getElem]


theorem getD_append_right (l l' : List α) (d : α) (n : ℕ) (h : l.length ≤ n) :
    (l ++ l').getD n d = l'.getD (n - l.length) d := by
  cases Nat.lt_or_ge n (l ++ l').length with
  | inl h' =>
    rw [getD_eq_getElem (l ++ l') d h', getElem_append_right h, getD_eq_getElem]
  | inr h' =>
    rw [getD_eq_default _ _ h', getD_eq_default]
    rwa [Nat.le_sub_iff_add_le' h, ← length_append]


theorem getD_eq_getD_get? (n : ℕ) : l.getD n d = (l.get? n).getD d := by
  cases Nat.lt_or_ge n l.length with
  | inl h => rw [getD_eq_getElem _ _ h, get?_eq_get h, get_eq_getElem, Option.getD_some]
  | inr h => rw [getD_eq_default _ _ h, get?_eq_none_iff.mpr h, Option.getD_none]


@[simp]
theorem getI_nil : getI ([] : List α) n = default :=
  rfl


@[simp]
theorem getI_cons_zero : getI (x :: xs) 0 = x :=
  rfl


@[simp]
theorem getI_cons_succ : getI (x :: xs) (n + 1) = getI xs n :=
  rfl


theorem getI_eq_getElem {n : ℕ} (hn : n < l.length) : l.getI n = l[n] :=
  getD_eq_getElem l default hn


@[deprecated getI_eq_getElem (since := "2024-08-02")]
theorem getI_eq_get {n : ℕ} (hn : n < l.length) : l.getI n = l.get ⟨n, hn⟩ :=
  getD_eq_getElem l default hn


theorem getI_eq_default {n : ℕ} (hn : l.length ≤ n) : l.getI n = default :=
  getD_eq_default _ _ hn


theorem getD_default_eq_getI {n : ℕ} : l.getD n default = l.getI n :=
  rfl


theorem getI_append (l l' : List α) (n : ℕ) (h : n < l.length) :
    (l ++ l').getI n = l.getI n := getD_append _ _ _ _ h


theorem getI_append_right (l l' : List α) (n : ℕ) (h : l.length ≤ n) :
    (l ++ l').getI n = l'.getI (n - l.length) :=
  getD_append_right _ _ _ _ h


theorem getI_eq_iget_get? (n : ℕ) : l.getI n = (l.get? n).iget := by
  /-
    α : Type u
    l : List α
    inst✝ : Inhabited α
    n : Nat
    ⊢ Eq (l.getI n) (l.get? n).iget
  -/
  rw [← getD_default_eq_getI, getD_eq_getD_get?, Option.getD_default_eq_iget]
  /-
    🎉 no goals
  -/


theorem getI_eq_iget_getElem? (n : ℕ) : l.getI n = l[n]?.iget := by
  /-
    α : Type u
    l : List α
    inst✝ : Inhabited α
    n : Nat
    ⊢ Eq (l.getI n) (GetElem?.getElem? l n).iget
  -/
  rw [← getD_default_eq_getI, getD_eq_getElem?_getD, Option.getD_default_eq_iget]
  /-
    🎉 no goals
  -/


                                                      /-
                                                        α : Type u
                                                        l : List α
                                                        inst✝ : Inhabited α
                                                        ⊢ Eq (l.getI 0) l.headI
                                                      -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
theorem getI_zero_eq_headI : l.getI 0 = l.headI := by cases l <;> rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


