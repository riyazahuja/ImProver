theorem get_insertIdx_of_lt (l : List α) (x : α) (n k : ℕ) (hn : k < n) (hk : k < l.length)
    (hk' : k < (insertIdx n x l).length := hk.trans_le (length_le_length_insertIdx _ _ _)) :
    (insertIdx n x l).get ⟨k, hk'⟩ = l.get ⟨k, hk⟩ := by
  /-
    α : Type u
    l : List α
    x : α
    n k : Nat
    hn : LT.lt k n
    hk : LT.lt k l.length
    hk' : optParam (LT.lt k (List.insertIdx n x l).length) ⋯
    ⊢ Eq ((List.insertIdx n x l).get ⟨k, hk'⟩) (l.get ⟨k, hk⟩)
  -/
  simp_all [getElem_insertIdx_of_lt]
  /-
    🎉 no goals
  -/


theorem get_insertIdx_self (l : List α) (x : α) (n : ℕ) (hn : n ≤ l.length)
    (hn' : n < (insertIdx n x l).length :=
          /-
            α : Type u
            a : α
            l : List α
            x : α
            n : Nat
            hn : LE.le n l.length
            ⊢ LT.lt n (List.insertIdx n x l).length
          -/
      (by rwa [length_insertIdx_of_le_length hn, Nat.lt_succ_iff])) :
          /-
            🎉 no goals
          -/
    (insertIdx n x l).get ⟨n, hn'⟩ = x := by
  /-
    α : Type u
    l : List α
    x : α
    n : Nat
    hn : LE.le n l.length
    hn' : optParam (LT.lt n (List.insertIdx n x l).length) ⋯
    ⊢ Eq ((List.insertIdx n x l).get ⟨n, hn'⟩) x
  -/
  simp [hn, hn']
  /-
    🎉 no goals
  -/


theorem getElem_insertIdx_add_succ (l : List α) (x : α) (n k : ℕ) (hk' : n + k < l.length)
    (hk : n + k + 1 < (insertIdx n x l).length := (by
      /-
        α : Type u
        a : α
        l : List α
        x : α
        n k : Nat
        hk' : LT.lt (HAdd.hAdd n k) l.length
        ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd n k) 1) (List.insertIdx n x l).length
      -/
      rwa [length_insertIdx_of_le_length (by omega), Nat.succ_lt_succ_iff])) :
      /-
        🎉 no goals
      -/
    (insertIdx n x l)[n + k + 1] = l[n + k] := by
  /-
    α : Type u
    l : List α
    x : α
    n k : Nat
    hk' : LT.lt (HAdd.hAdd n k) l.length
    hk : optParam (LT.lt (HAdd.hAdd (HAdd.hAdd n k) 1) (List.insertIdx n x l).leng …
    ⊢ Eq (GetElem.getElem (List.insertIdx n x l) (HAdd.hAdd (HAdd.hAdd n k) 1) hk) …
  -/
  rw [getElem_insertIdx_of_ge (by omega)]
  /-
    α : Type u
    l : List α
    x : α
    n k : Nat
    hk' : LT.lt (HAdd.hAdd n k) l.length
    hk : optParam (LT.lt (HAdd.hAdd (HAdd.hAdd n k) 1) (List.insertIdx n x l).leng …
    ⊢ Eq (GetElem.getElem l (HSub.hSub (HAdd.hAdd (HAdd.hAdd n k) 1) 1) ⋯) (GetEle …
  -/
  simp only [Nat.add_one_sub_one]
  /-
    🎉 no goals
  -/


theorem get_insertIdx_add_succ (l : List α) (x : α) (n k : ℕ) (hk' : n + k < l.length)
    (hk : n + k + 1 < (insertIdx n x l).length := (by
      /-
        α : Type u
        a : α
        l : List α
        x : α
        n k : Nat
        hk' : LT.lt (HAdd.hAdd n k) l.length
        ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd n k) 1) (List.insertIdx n x l).length
      -/
      rwa [length_insertIdx_of_le_length (by omega), Nat.succ_lt_succ_iff])) :
      /-
        🎉 no goals
      -/
    (insertIdx n x l).get ⟨n + k + 1, hk⟩ = get l ⟨n + k, hk'⟩ := by
  /-
    α : Type u
    l : List α
    x : α
    n k : Nat
    hk' : LT.lt (HAdd.hAdd n k) l.length
    hk : optParam (LT.lt (HAdd.hAdd (HAdd.hAdd n k) 1) (List.insertIdx n x l).leng …
    ⊢ Eq ((List.insertIdx n x l).get ⟨HAdd.hAdd (HAdd.hAdd n k) 1, hk⟩) (l.get ⟨HA …
  -/
  simp [getElem_insertIdx_add_succ, hk, hk']
  /-
    🎉 no goals
  -/


set_option linter.unnecessarySimpa false in
theorem insertIdx_injective (n : ℕ) (x : α) : Function.Injective (insertIdx n x) := by
  /-
    α : Type u
    n : Nat
    x : α
    ⊢ Function.Injective (List.insertIdx n x)
  -/
  induction' n with n IH
    /-
      case zero
      α : Type u
      x : α
      ⊢ Function.Injective (List.insertIdx 0 x)
    -/
  · have : insertIdx 0 x = cons x := funext fun _ => rfl
    /-
      case zero
      α : Type u
      x : α
      this : Eq (List.insertIdx 0 x) (List.cons x)
      ⊢ Function.Injective (List.insertIdx 0 x)
    -/
    simp [this]
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      x : α
      n : Nat
      IH : Function.Injective (List.insertIdx n x)
      ⊢ Function.Injective (List.insertIdx (HAdd.hAdd n 1) x)
    -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
  · rintro (_ | ⟨a, as⟩) (_ | ⟨b, bs⟩) h <;> simpa [IH.eq_iff] using h
                                             /-
                                               🎉 no goals
                                             -/


@[deprecated (since := "2024-10-21")] alias insertNth_zero := insertIdx_zero

@[deprecated (since := "2024-10-21")] alias insertNth_succ_nil := insertIdx_succ_nil

@[deprecated (since := "2024-10-21")] alias insertNth_succ_cons := insertIdx_succ_cons

@[deprecated (since := "2024-10-21")] alias length_insertNth := length_insertIdx

@[deprecated (since := "2024-10-21")] alias removeNth_insertIdx := eraseIdx_insertIdx

@[deprecated (since := "2024-05-04")] alias removeNth_insertNth := eraseIdx_insertIdx

@[deprecated (since := "2024-10-21")] alias insertNth_eraseIdx_of_ge := insertIdx_eraseIdx_of_ge

@[deprecated (since := "2024-05-04")] alias insertNth_removeNth_of_ge := insertIdx_eraseIdx_of_ge

@[deprecated (since := "2024-10-21")] alias insertNth_eraseIdx_of_le := insertIdx_eraseIdx_of_le

@[deprecated (since := "2024-05-04")] alias insertIdx_removeNth_of_le := insertIdx_eraseIdx_of_le

@[deprecated (since := "2024-10-21")] alias insertNth_comm := insertIdx_comm

@[deprecated (since := "2024-10-21")] alias mem_insertNth := mem_insertIdx

@[deprecated (since := "2024-10-21")] alias insertNth_of_length_lt := insertIdx_of_length_lt

@[deprecated (since := "2024-10-21")] alias insertNth_length_self := insertIdx_length_self

@[deprecated (since := "2024-10-21")] alias length_le_length_insertNth := length_le_length_insertIdx

@[deprecated (since := "2024-10-21")] alias length_insertNth_le_succ := length_insertIdx_le_succ

@[deprecated (since := "2024-10-21")] alias getElem_insertNth_of_lt := getElem_insertIdx_of_lt

@[deprecated (since := "2024-10-21")] alias get_insertNth_of_lt := get_insertIdx_of_lt

@[deprecated (since := "2024-10-21")] alias getElem_insertNth_self := getElem_insertIdx_self

@[deprecated (since := "2024-10-21")] alias get_insertNth_self := get_insertIdx_self

@[deprecated (since := "2024-10-21")] alias getElem_insertNth_add_succ := getElem_insertIdx_add_succ

@[deprecated (since := "2024-10-21")] alias get_insertNth_add_succ := get_insertIdx_add_succ

@[deprecated (since := "2024-10-21")] alias insertNth_injective := insertIdx_injective


