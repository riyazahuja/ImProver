@[simp]
theorem length_iterate (f : α → α) (a : α) (n : ℕ) : length (iterate f a n) = n := by
  /-
    α : Type u_1
    f : α → α
    a : α
    n : Nat
    ⊢ Eq (List.iterate f a n).length n
  -/
                                 /-
                                   🎉 no goals
                                 -/
  induction n generalizing a <;> simp [*]
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem iterate_eq_nil {f : α → α} {a : α} {n : ℕ} : iterate f a n = [] ↔ n = 0 := by
  /-
    α : Type u_1
    f : α → α
    a : α
    n : Nat
    ⊢ Iff (Eq (List.iterate f a n) List.nil) (Eq n 0)
  -/
  rw [← length_eq_zero, length_iterate]
  /-
    🎉 no goals
  -/


theorem getElem?_iterate (f : α → α) (a : α) :
    ∀ (n i : ℕ), i < n → (iterate f a n)[i]? = f^[i] a
                          /-
                            α : Type u_1
                            f : α → α
                            a : α
                            n : Nat
                            x✝ : LT.lt 0 (HAdd.hAdd n 1)
                            ⊢ Eq (GetElem?.getElem? (List.iterate f a (HAdd.hAdd n 1)) 0) (Option.some (Na …
                          -/
  | n + 1, 0    , _ => by simp
                          /-
                            🎉 no goals
                          -/
                          /-
                            α : Type u_1
                            f : α → α
                            a : α
                            n i : Nat
                            h : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd n 1)
                            ⊢ Eq (GetElem?.getElem? (List.iterate f a (HAdd.hAdd n 1)) (HAdd.hAdd i 1)) (O …
                          -/
  | n + 1, i + 1, h => by simp [getElem?_iterate f (f a) n i (by simpa using h)]
                          /-
                            🎉 no goals
                          -/


@[deprecated getElem?_iterate (since := "2024-08-23")]
theorem get?_iterate (f : α → α) (a : α) (n i : ℕ) (h : i < n) :
    get? (iterate f a n) i = f^[i] a := by
  /-
    α : Type u_1
    f : α → α
    a : α
    n i : Nat
    h : LT.lt i n
    ⊢ Eq ((List.iterate f a n).get? i) (Option.some (Nat.iterate f i a))
  -/
  simp only [get?_eq_getElem?, getElem?_iterate, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem getElem_iterate (f : α → α) (a : α) (n : ℕ) (i : Nat) (h : i < (iterate f a n).length) :
    (iterate f a n)[i] = f^[i] a :=
                                                     /-
                                                       α : Type u_1
                                                       f : α → α
                                                       a : α
                                                       n i : Nat
                                                       h : LT.lt i (List.iterate f a n).length
                                                       ⊢ LT.lt i n
                                                     -/
  getElem_eq_iff.2 <| getElem?_iterate _ _ _ _ <| by rwa [length_iterate] at h
                                                     /-
                                                       🎉 no goals
                                                     -/


@[deprecated getElem_iterate (since := "2024-08-23")]
theorem get_iterate (f : α → α) (a : α) (n : ℕ) (i : Fin (iterate f a n).length) :
    get (iterate f a n) i = f^[↑i] a := by
  /-
    α : Type u_1
    f : α → α
    a : α
    n : Nat
    i : Fin (List.iterate f a n).length
    ⊢ Eq ((List.iterate f a n).get i) (Nat.iterate f (↑i) a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_iterate {f : α → α} {a : α} {n : ℕ} {b : α} :
    b ∈ iterate f a n ↔ ∃ m < n, b = f^[m] a := by
  /-
    α : Type u_1
    f : α → α
    a : α
    n : Nat
    b : α
    ⊢ Iff (Membership.mem (List.iterate f a n) b) (Exists fun m => And (LT.lt m n) …
  -/
  simp [List.mem_iff_get, Fin.exists_iff, eq_comm (b := b)]
  /-
    🎉 no goals
  -/


@[simp]
theorem range_map_iterate (n : ℕ) (f : α → α) (a : α) :
    (List.range n).map (f^[·] a) = List.iterate f a n := by
  /-
    α : Type u_1
    n : Nat
    f : α → α
    a : α
    ⊢ Eq (List.map (fun x => Nat.iterate f x a) (List.range n)) (List.iterate f a n)
  -/
                             /-
                               🎉 no goals
                             -/
  apply List.ext_getElem <;> simp
                             /-
                               🎉 no goals
                             -/


theorem iterate_add (f : α → α) (a : α) (m n : ℕ) :
    iterate f a (m + n) = iterate f a m ++ iterate f (f^[m] a) n := by
  induction m generalizing a with
  | zero => simp
  | succ n ih => rw [iterate, add_right_comm, iterate, ih, Nat.iterate, cons_append]


theorem take_iterate (f : α → α) (a : α) (m n : ℕ) :
    take m (iterate f a n) = iterate f a (min m n) := by
  /-
    α : Type u_1
    f : α → α
    a : α
    m n : Nat
    ⊢ Eq (List.take m (List.iterate f a n)) (List.iterate f a (Min.min m n))
  -/
  rw [← range_map_iterate, ← range_map_iterate, ← map_take, take_range]
  /-
    🎉 no goals
  -/


