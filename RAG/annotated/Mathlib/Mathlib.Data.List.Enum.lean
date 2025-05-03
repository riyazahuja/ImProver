@[deprecated getElem?_enumFrom (since := "2024-08-15")]
theorem get?_enumFrom (n) (l : List α) (m) :
    get? (enumFrom n l) m = (get? l m).map fun a => (n + m, a) := by
  /-
    α : Type u_1
    n : Nat
    l : List α
    m : Nat
    ⊢ Eq ((List.enumFrom n l).get? m) (Option.map (fun a => { fst := HAdd.hAdd n m …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-06")] alias enumFrom_get? := get?_enumFrom


@[deprecated getElem?_enum (since := "2024-08-15")]
theorem get?_enum (l : List α) (n) : get? (enum l) n = (get? l n).map fun a => (n, a) := by
  /-
    α : Type u_1
    l : List α
    n : Nat
    ⊢ Eq (l.enum.get? n) (Option.map (fun a => { fst := n, snd := a }) (l.get? n))
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-06")] alias enum_get? := get?_enum


@[deprecated getElem_enumFrom (since := "2024-08-15")]
theorem get_enumFrom (l : List α) (n) (i : Fin (l.enumFrom n).length) :
    (l.enumFrom n).get i = (n + i, l.get (i.cast enumFrom_length)) := by
  /-
    α : Type u_1
    l : List α
    n : Nat
    i : Fin (List.enumFrom n l).length
    ⊢ Eq ((List.enumFrom n l).get i) { fst := HAdd.hAdd n ↑i, snd := l.get (Fin.ca …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated getElem_enum (since := "2024-08-15")]
theorem get_enum (l : List α) (i : Fin l.enum.length) :
    l.enum.get i = (i.1, l.get (i.cast enum_length)) := by
  /-
    α : Type u_1
    l : List α
    i : Fin l.enum.length
    ⊢ Eq (l.enum.get i) { fst := ↑i, snd := l.get (Fin.cast ⋯ i) }
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated mk_add_mem_enumFrom_iff_getElem? (since := "2024-08-12")]
theorem mk_add_mem_enumFrom_iff_get? {n i : ℕ} {x : α} {l : List α} :
    (n + i, x) ∈ enumFrom n l ↔ l.get? i = x := by
  /-
    α : Type u_1
    n i : Nat
    x : α
    l : List α
    ⊢ Iff (Membership.mem (List.enumFrom n l) { fst := HAdd.hAdd n i, snd := x })  …
  -/
  simp [mem_iff_get?]
  /-
    🎉 no goals
  -/


@[deprecated mk_mem_enumFrom_iff_le_and_getElem?_sub (since := "2024-08-12")]
theorem mk_mem_enumFrom_iff_le_and_get?_sub {n i : ℕ} {x : α} {l : List α} :
    (i, x) ∈ enumFrom n l ↔ n ≤ i ∧ l.get? (i - n) = x := by
  /-
    α : Type u_1
    n i : Nat
    x : α
    l : List α
    ⊢ Iff (Membership.mem (List.enumFrom n l) { fst := i, snd := x }) (And (LE.le  …
  -/
  simp [mk_mem_enumFrom_iff_le_and_getElem?_sub]
  /-
    🎉 no goals
  -/


@[deprecated mk_mem_enum_iff_getElem? (since := "2024-08-15")]
theorem mk_mem_enum_iff_get? {i : ℕ} {x : α} {l : List α} : (i, x) ∈ enum l ↔ l.get? i = x := by
  /-
    α : Type u_1
    i : Nat
    x : α
    l : List α
    ⊢ Iff (Membership.mem l.enum { fst := i, snd := x }) (Eq (l.get? i) (Option.so …
  -/
  simp [enum, mk_mem_enumFrom_iff_le_and_getElem?_sub]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated mem_enum_iff_getElem? (since := "2024-08-15")]
theorem mem_enum_iff_get? {x : ℕ × α} {l : List α} : x ∈ enum l ↔ l.get? x.1 = x.2 :=
  mk_mem_enum_iff_get?


theorem forall_mem_enumFrom {l : List α} {n : ℕ} {p : ℕ × α → Prop} :
    (∀ x ∈ l.enumFrom n, p x) ↔ ∀ (i : ℕ) (_ : i < length l), p (n + i, l[i]) := by
  /-
    α : Type u_1
    l : List α
    n : Nat
    p : Prod Nat α → Prop
    ⊢ Iff (∀ (x : Prod Nat α), Membership.mem (List.enumFrom n l) x → p x) (∀ (i : …
  -/
  simp only [forall_mem_iff_getElem, getElem_enumFrom, enumFrom_length]
  /-
    🎉 no goals
  -/


theorem forall_mem_enum {l : List α} {p : ℕ × α → Prop} :
    (∀ x ∈ l.enum, p x) ↔ ∀ (i : ℕ) (_ : i < length l), p (i, l[i]) :=
                                  /-
                                    α : Type u_1
                                    l : List α
                                    p : Prod Nat α → Prop
                                    ⊢ Iff (∀ (i : Nat) (x : LT.lt i l.length), p { fst := HAdd.hAdd 0 i, snd := Ge …
                                  -/
  forall_mem_enumFrom.trans <| by simp
                                  /-
                                    🎉 no goals
                                  -/


theorem exists_mem_enumFrom {l : List α} {n : ℕ} {p : ℕ × α → Prop} :
    (∃ x ∈ l.enumFrom n, p x) ↔ ∃ (i : ℕ) (_ : i < length l), p (n + i, l[i]) := by
  /-
    α : Type u_1
    l : List α
    n : Nat
    p : Prod Nat α → Prop
    ⊢ Iff (Exists fun x => And (Membership.mem (List.enumFrom n l) x) (p x)) (Exis …
  -/
  simp only [exists_mem_iff_getElem, getElem_enumFrom, enumFrom_length]
  /-
    🎉 no goals
  -/


theorem exists_mem_enum {l : List α} {p : ℕ × α → Prop} :
    (∃ x ∈ l.enum, p x) ↔ ∃ (i : ℕ) (_ : i < length l), p (i, l[i]) :=
                                  /-
                                    α : Type u_1
                                    l : List α
                                    p : Prod Nat α → Prop
                                    ⊢ Iff (Exists fun i => Exists fun x => p { fst := HAdd.hAdd 0 i, snd := GetEle …
                                  -/
  exists_mem_enumFrom.trans <| by simp
                                  /-
                                    🎉 no goals
                                  -/


