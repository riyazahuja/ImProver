/-- Drop `n` elements from the tail end of a list. -/
def rdrop : List α :=
  l.take (l.length - n)


@[simp]
                                                     /-
                                                       α : Type u_1
                                                       n : Nat
                                                       ⊢ Eq (List.nil.rdrop n) List.nil
                                                     -/
theorem rdrop_nil : rdrop ([] : List α) n = [] := by simp [rdrop]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
                                         /-
                                           α : Type u_1
                                           l : List α
                                           ⊢ Eq (l.rdrop 0) l
                                         -/
theorem rdrop_zero : rdrop l 0 = l := by simp [rdrop]
                                         /-
                                           🎉 no goals
                                         -/


theorem rdrop_eq_reverse_drop_reverse : l.rdrop n = reverse (l.reverse.drop n) := by
  /-
    α : Type u_1
    l : List α
    n : Nat
    ⊢ Eq (l.rdrop n) (List.drop n l.reverse).reverse
  -/
  rw [rdrop]
  /-
    α : Type u_1
    l : List α
    n : Nat
    ⊢ Eq (List.take (HSub.hSub l.length n) l) (List.drop n l.reverse).reverse
  -/
  induction' l using List.reverseRecOn with xs x IH generalizing n
    /-
      case nil
      α : Type u_1
      n : Nat
      ⊢ Eq (List.take (HSub.hSub List.nil.length n) List.nil) (List.drop n List.nil. …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case append_singleton
      α : Type u_1
      xs : List α
      x : α
      IH : ∀ (n : Nat), Eq (List.take (HSub.hSub xs.length n) xs) (List.drop n xs.re …
      n : Nat
      ⊢ Eq (List.take (HSub.hSub (HAppend.hAppend xs (List.cons x List.nil)).length  …
    -/
  · cases n
      /-
        case append_singleton.zero
        α : Type u_1
        xs : List α
        x : α
        IH : ∀ (n : Nat), Eq (List.take (HSub.hSub xs.length n) xs) (List.drop n xs.re …
        ⊢ Eq (List.take (HSub.hSub (HAppend.hAppend xs (List.cons x List.nil)).length  …
      -/
    · simp [take_append]
      /-
        🎉 no goals
      -/
      /-
        case append_singleton.succ
        α : Type u_1
        xs : List α
        x : α
        IH : ∀ (n : Nat), Eq (List.take (HSub.hSub xs.length n) xs) (List.drop n xs.re …
        n✝ : Nat
        ⊢ Eq (List.take (HSub.hSub (HAppend.hAppend xs (List.cons x List.nil)).length  …
      -/
    · simp [take_append_eq_append_take, IH]
      /-
        🎉 no goals
      -/


@[simp]
theorem rdrop_concat_succ (x : α) : rdrop (l ++ [x]) (n + 1) = rdrop l n := by
  /-
    α : Type u_1
    l : List α
    n : Nat
    x : α
    ⊢ Eq ((HAppend.hAppend l (List.cons x List.nil)).rdrop (HAdd.hAdd n 1)) (l.rdr …
  -/
  simp [rdrop_eq_reverse_drop_reverse]
  /-
    🎉 no goals
  -/


/-- Take `n` elements from the tail end of a list. -/
def rtake : List α :=
  l.drop (l.length - n)


@[simp]
                                                     /-
                                                       α : Type u_1
                                                       n : Nat
                                                       ⊢ Eq (List.nil.rtake n) List.nil
                                                     -/
theorem rtake_nil : rtake ([] : List α) n = [] := by simp [rtake]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
                                          /-
                                            α : Type u_1
                                            l : List α
                                            ⊢ Eq (l.rtake 0) List.nil
                                          -/
theorem rtake_zero : rtake l 0 = [] := by simp [rtake]
                                          /-
                                            🎉 no goals
                                          -/


theorem rtake_eq_reverse_take_reverse : l.rtake n = reverse (l.reverse.take n) := by
  /-
    α : Type u_1
    l : List α
    n : Nat
    ⊢ Eq (l.rtake n) (List.take n l.reverse).reverse
  -/
  rw [rtake]
  /-
    α : Type u_1
    l : List α
    n : Nat
    ⊢ Eq (List.drop (HSub.hSub l.length n) l) (List.take n l.reverse).reverse
  -/
  induction' l using List.reverseRecOn with xs x IH generalizing n
    /-
      case nil
      α : Type u_1
      n : Nat
      ⊢ Eq (List.drop (HSub.hSub List.nil.length n) List.nil) (List.take n List.nil. …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case append_singleton
      α : Type u_1
      xs : List α
      x : α
      IH : ∀ (n : Nat), Eq (List.drop (HSub.hSub xs.length n) xs) (List.take n xs.re …
      n : Nat
      ⊢ Eq (List.drop (HSub.hSub (HAppend.hAppend xs (List.cons x List.nil)).length  …
    -/
  · cases n
      /-
        case append_singleton.zero
        α : Type u_1
        xs : List α
        x : α
        IH : ∀ (n : Nat), Eq (List.drop (HSub.hSub xs.length n) xs) (List.take n xs.re …
        ⊢ Eq (List.drop (HSub.hSub (HAppend.hAppend xs (List.cons x List.nil)).length  …
      -/
    · exact drop_length _
      /-
        🎉 no goals
      -/
      /-
        case append_singleton.succ
        α : Type u_1
        xs : List α
        x : α
        IH : ∀ (n : Nat), Eq (List.drop (HSub.hSub xs.length n) xs) (List.take n xs.re …
        n✝ : Nat
        ⊢ Eq (List.drop (HSub.hSub (HAppend.hAppend xs (List.cons x List.nil)).length  …
      -/
    · simp [drop_append_eq_append_drop, IH]
      /-
        🎉 no goals
      -/


@[simp]
theorem rtake_concat_succ (x : α) : rtake (l ++ [x]) (n + 1) = rtake l n ++ [x] := by
  /-
    α : Type u_1
    l : List α
    n : Nat
    x : α
    ⊢ Eq ((HAppend.hAppend l (List.cons x List.nil)).rtake (HAdd.hAdd n 1)) (HAppe …
  -/
  simp [rtake_eq_reverse_take_reverse]
  /-
    🎉 no goals
  -/


/-- Drop elements from the tail end of a list that satisfy `p : α → Bool`.
Implemented naively via `List.reverse` -/
def rdropWhile : List α :=
  reverse (l.reverse.dropWhile p)


@[simp]
                                                               /-
                                                                 α : Type u_1
                                                                 p : α → Bool
                                                                 ⊢ Eq (List.rdropWhile p List.nil) List.nil
                                                               -/
theorem rdropWhile_nil : rdropWhile p ([] : List α) = [] := by simp [rdropWhile, dropWhile]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem rdropWhile_concat (x : α) :
    rdropWhile p (l ++ [x]) = if p x then rdropWhile p l else l ++ [x] := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    x : α
    ⊢ Eq (List.rdropWhile p (HAppend.hAppend l (List.cons x List.nil))) (ite (Eq ( …
  -/
  simp only [rdropWhile, dropWhile, reverse_append, reverse_singleton, singleton_append]
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    x : α
    ⊢ Eq (List.filter.match_1 (fun x => List α) (p x) (fun _ => List.dropWhile p l …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem rdropWhile_concat_pos (x : α) (h : p x) : rdropWhile p (l ++ [x]) = rdropWhile p l := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    x : α
    h : Eq (p x) Bool.true
    ⊢ Eq (List.rdropWhile p (HAppend.hAppend l (List.cons x List.nil))) (List.rdro …
  -/
  rw [rdropWhile_concat, if_pos h]
  /-
    🎉 no goals
  -/


@[simp]
theorem rdropWhile_concat_neg (x : α) (h : ¬p x) : rdropWhile p (l ++ [x]) = l ++ [x] := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    x : α
    h : Not (Eq (p x) Bool.true)
    ⊢ Eq (List.rdropWhile p (HAppend.hAppend l (List.cons x List.nil))) (HAppend.h …
  -/
  rw [rdropWhile_concat, if_neg h]
  /-
    🎉 no goals
  -/


theorem rdropWhile_singleton (x : α) : rdropWhile p [x] = if p x then [] else [x] := by
  /-
    α : Type u_1
    p : α → Bool
    x : α
    ⊢ Eq (List.rdropWhile p (List.cons x List.nil)) (ite (Eq (p x) Bool.true) List …
  -/
  rw [← nil_append [x], rdropWhile_concat, rdropWhile_nil]
  /-
    🎉 no goals
  -/


theorem rdropWhile_last_not (hl : l.rdropWhile p ≠ []) : ¬p ((rdropWhile p l).getLast hl) := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    hl : Ne (List.rdropWhile p l) List.nil
    ⊢ Not (Eq (p ((List.rdropWhile p l).getLast hl)) Bool.true)
  -/
  simp_rw [rdropWhile]
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    hl : Ne (List.rdropWhile p l) List.nil
    ⊢ Not (Eq (p ((List.dropWhile p l.reverse).reverse.getLast ⋯)) Bool.true)
  -/
  rw [getLast_reverse, head_dropWhile_not p]
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    hl : Ne (List.rdropWhile p l) List.nil
    ⊢ Not (Eq Bool.false Bool.true)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem rdropWhile_prefix : l.rdropWhile p <+: l := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    ⊢ (List.rdropWhile p l).IsPrefix l
  -/
  rw [← reverse_suffix, rdropWhile, reverse_reverse]
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    ⊢ (List.dropWhile p l.reverse).IsSuffix l.reverse
  -/
  exact dropWhile_suffix _
  /-
    🎉 no goals
  -/


@[simp]
                                                                         /-
                                                                           α : Type u_1
                                                                           p : α → Bool
                                                                           l : List α
                                                                           ⊢ Iff (Eq (List.rdropWhile p l) List.nil) (∀ (x : α), Membership.mem l x → Eq  …
                                                                         -/
theorem rdropWhile_eq_nil_iff : rdropWhile p l = [] ↔ ∀ x ∈ l, p x := by simp [rdropWhile]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/

-- it is in this file because it requires `List.Infix`

@[simp]
theorem dropWhile_eq_self_iff : dropWhile p l = l ↔ ∀ hl : 0 < l.length, ¬p (l.get ⟨0, hl⟩) := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    ⊢ Iff (Eq (List.dropWhile p l) l) (∀ (hl : LT.lt 0 l.length), Not (Eq (p (l.ge …
  -/
  cases' l with hd tl
    /-
      case nil
      α : Type u_1
      p : α → Bool
      ⊢ Iff (Eq (List.dropWhile p List.nil) List.nil) (∀ (hl : LT.lt 0 List.nil.leng …
    -/
  · simp only [dropWhile, true_iff]
    /-
      case nil
      α : Type u_1
      p : α → Bool
      ⊢ ∀ (hl : LT.lt 0 List.nil.length), Not (Eq (p (List.nil.get ⟨0, hl⟩)) Bool.tr …
    -/
    intro h
    /-
      case nil
      α : Type u_1
      p : α → Bool
      h : LT.lt 0 List.nil.length
      ⊢ Not (Eq (p (List.nil.get ⟨0, h⟩)) Bool.true)
    -/
    by_contra
    /-
      case nil
      α : Type u_1
      p : α → Bool
      h : LT.lt 0 List.nil.length
      a✝ : Eq (p (List.nil.get ⟨0, h⟩)) Bool.true
      ⊢ False
    -/
    rwa [length_nil, lt_self_iff_false] at h
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      p : α → Bool
      hd : α
      tl : List α
      ⊢ Iff (Eq (List.dropWhile p (List.cons hd tl)) (List.cons hd tl)) (∀ (hl : LT. …
    -/
  · rw [dropWhile]
    /-
      case cons
      α : Type u_1
      p : α → Bool
      hd : α
      tl : List α
      ⊢ Iff (Eq (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhi …
    -/
    refine ⟨fun h => ?_, fun h => ?_⟩
      /-
        case cons.refine_1
        α : Type u_1
        p : α → Bool
        hd : α
        tl : List α
        h : Eq (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhile  …
        ⊢ ∀ (hl : LT.lt 0 (List.cons hd tl).length), Not (Eq (p ((List.cons hd tl).get …
      -/
    · intro _ H
      /-
        case cons.refine_1
        α : Type u_1
        p : α → Bool
        hd : α
        tl : List α
        h : Eq (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhile  …
        hl✝ : LT.lt 0 (List.cons hd tl).length
        H : Eq (p ((List.cons hd tl).get ⟨0, hl✝⟩)) Bool.true
        ⊢ False
      -/
      rw [get] at H
      /-
        case cons.refine_1
        α : Type u_1
        p : α → Bool
        hd : α
        tl : List α
        h : Eq (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhile  …
        hl✝ : LT.lt 0 (List.cons hd tl).length
        H : Eq (p hd) Bool.true
        ⊢ False
      -/
      refine (cons_ne_self hd tl) (Sublist.antisymm ?_ (sublist_cons_self _ _))
      /-
        case cons.refine_1
        α : Type u_1
        p : α → Bool
        hd : α
        tl : List α
        h : Eq (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhile  …
        hl✝ : LT.lt 0 (List.cons hd tl).length
        H : Eq (p hd) Bool.true
        ⊢ (List.cons hd tl).Sublist tl
      -/
      rw [← h]
      /-
        case cons.refine_1
        α : Type u_1
        p : α → Bool
        hd : α
        tl : List α
        h : Eq (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhile  …
        hl✝ : LT.lt 0 (List.cons hd tl).length
        H : Eq (p hd) Bool.true
        ⊢ (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhile p tl) …
      -/
      simp only [H]
      /-
        case cons.refine_1
        α : Type u_1
        p : α → Bool
        hd : α
        tl : List α
        h : Eq (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhile  …
        hl✝ : LT.lt 0 (List.cons hd tl).length
        H : Eq (p hd) Bool.true
        ⊢ (List.dropWhile p tl).Sublist tl
      -/
      exact List.IsSuffix.sublist (dropWhile_suffix p)
      /-
        🎉 no goals
      -/
      /-
        case cons.refine_2
        α : Type u_1
        p : α → Bool
        hd : α
        tl : List α
        h : ∀ (hl : LT.lt 0 (List.cons hd tl).length), Not (Eq (p ((List.cons hd tl).g …
        ⊢ Eq (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhile p  …
      -/
    · have := h (by simp only [length, Nat.succ_pos])
      /-
        case cons.refine_2
        α : Type u_1
        p : α → Bool
        hd : α
        tl : List α
        h : ∀ (hl : LT.lt 0 (List.cons hd tl).length), Not (Eq (p ((List.cons hd tl).g …
        this : Not (Eq (p ((List.cons hd tl).get ⟨0, ⋯⟩)) Bool.true)
        ⊢ Eq (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhile p  …
      -/
      rw [get] at this
      /-
        case cons.refine_2
        α : Type u_1
        p : α → Bool
        hd : α
        tl : List α
        h : ∀ (hl : LT.lt 0 (List.cons hd tl).length), Not (Eq (p ((List.cons hd tl).g …
        this : Not (Eq (p hd) Bool.true)
        ⊢ Eq (List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dropWhile p  …
      -/
      simp_rw [this]
      /-
        🎉 no goals
      -/


@[simp]
theorem rdropWhile_eq_self_iff : rdropWhile p l = l ↔ ∀ hl : l ≠ [], ¬p (l.getLast hl) := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    ⊢ Iff (Eq (List.rdropWhile p l) l) (∀ (hl : Ne l List.nil), Not (Eq (p (l.getL …
  -/
  simp [rdropWhile, reverse_eq_iff, getLast_eq_getElem, Nat.pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem dropWhile_idempotent : dropWhile p (dropWhile p l) = dropWhile p l := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    ⊢ Eq (List.dropWhile p (List.dropWhile p l)) (List.dropWhile p l)
  -/
  simp only [dropWhile_eq_self_iff]
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    ⊢ ∀ (hl : LT.lt 0 (List.dropWhile p l).length), Not (Eq (p ((List.dropWhile p  …
  -/
  exact fun h => dropWhile_get_zero_not p l h
  /-
    🎉 no goals
  -/


theorem rdropWhile_idempotent : rdropWhile p (rdropWhile p l) = rdropWhile p l :=
  rdropWhile_eq_self_iff.mpr (rdropWhile_last_not _ _)


/-- Take elements from the tail end of a list that satisfy `p : α → Bool`.
Implemented naively via `List.reverse` -/
def rtakeWhile : List α :=
  reverse (l.reverse.takeWhile p)


@[simp]
                                                               /-
                                                                 α : Type u_1
                                                                 p : α → Bool
                                                                 ⊢ Eq (List.rtakeWhile p List.nil) List.nil
                                                               -/
theorem rtakeWhile_nil : rtakeWhile p ([] : List α) = [] := by simp [rtakeWhile, takeWhile]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem rtakeWhile_concat (x : α) :
    rtakeWhile p (l ++ [x]) = if p x then rtakeWhile p l ++ [x] else [] := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    x : α
    ⊢ Eq (List.rtakeWhile p (HAppend.hAppend l (List.cons x List.nil))) (ite (Eq ( …
  -/
  simp only [rtakeWhile, takeWhile, reverse_append, reverse_singleton, singleton_append]
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    x : α
    ⊢ Eq (List.filter.match_1 (fun x => List α) (p x) (fun _ => List.cons x (List. …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem rtakeWhile_concat_pos (x : α) (h : p x) :
                                                          /-
                                                            α : Type u_1
                                                            p : α → Bool
                                                            l : List α
                                                            x : α
                                                            h : Eq (p x) Bool.true
                                                            ⊢ Eq (List.rtakeWhile p (HAppend.hAppend l (List.cons x List.nil))) (HAppend.h …
                                                          -/
    rtakeWhile p (l ++ [x]) = rtakeWhile p l ++ [x] := by rw [rtakeWhile_concat, if_pos h]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem rtakeWhile_concat_neg (x : α) (h : ¬p x) : rtakeWhile p (l ++ [x]) = [] := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    x : α
    h : Not (Eq (p x) Bool.true)
    ⊢ Eq (List.rtakeWhile p (HAppend.hAppend l (List.cons x List.nil))) List.nil
  -/
  rw [rtakeWhile_concat, if_neg h]
  /-
    🎉 no goals
  -/


theorem rtakeWhile_suffix : l.rtakeWhile p <:+ l := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    ⊢ (List.rtakeWhile p l).IsSuffix l
  -/
  rw [← reverse_prefix, rtakeWhile, reverse_reverse]
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    ⊢ (List.takeWhile p l.reverse).IsPrefix l.reverse
  -/
  exact takeWhile_prefix _
  /-
    🎉 no goals
  -/


@[simp]
theorem rtakeWhile_eq_self_iff : rtakeWhile p l = l ↔ ∀ x ∈ l, p x := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    ⊢ Iff (Eq (List.rtakeWhile p l) l) (∀ (x : α), Membership.mem l x → Eq (p x) B …
  -/
  simp [rtakeWhile, reverse_eq_iff]
  /-
    🎉 no goals
  -/

-- Porting note: This needed a lot of rewriting.

@[simp]
theorem rtakeWhile_eq_nil_iff : rtakeWhile p l = [] ↔ ∀ hl : l ≠ [], ¬p (l.getLast hl) := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    ⊢ Iff (Eq (List.rtakeWhile p l) List.nil) (∀ (hl : Ne l List.nil), Not (Eq (p  …
  -/
  induction' l using List.reverseRecOn with l a
    /-
      case nil
      α : Type u_1
      p : α → Bool
      l : List α
      ⊢ Iff (Eq (List.rtakeWhile p List.nil) List.nil) (∀ (hl : Ne List.nil List.nil …
    -/
  · simp only [rtakeWhile, takeWhile, reverse_nil, true_iff]
    /-
      case nil
      α : Type u_1
      p : α → Bool
      l : List α
      ⊢ ∀ (hl : Ne List.nil List.nil), Not (Eq (p (List.nil.getLast hl)) Bool.true)
    -/
    intro f; contradiction
             /-
               🎉 no goals
             -/
  · simp only [rtakeWhile, reverse_append, reverse_cons, reverse_nil, nil_append, singleton_append,
      takeWhile, ne_eq, cons_ne_self, not_false_eq_true, getLast_append_of_ne_nil,
      getLast_singleton]
    /-
      case append_singleton
      α : Type u_1
      p : α → Bool
      l✝ l : List α
      a : α
      a✝ : Iff (Eq (List.rtakeWhile p l) List.nil) (∀ (hl : Ne l List.nil), Not (Eq  …
      ⊢ Iff (Eq (List.filter.match_1 (fun x => List α) (p a) (fun _ => List.cons a ( …
    -/
    refine ⟨fun h => ?_ , fun h => ?_⟩
      /-
        case append_singleton.refine_1
        α : Type u_1
        p : α → Bool
        l✝ l : List α
        a : α
        a✝ : Iff (Eq (List.rtakeWhile p l) List.nil) (∀ (hl : Ne l List.nil), Not (Eq  …
        h : Eq (List.filter.match_1 (fun x => List α) (p a) (fun _ => List.cons a (Lis …
        ⊢ Not (Eq (HAppend.hAppend l (List.cons a List.nil)) List.nil) → Not (Eq (p a) …
      -/
                     /-
                       🎉 no goals
                     -/
    · split at h <;> simp_all
                     /-
                       🎉 no goals
                     -/
      /-
        case append_singleton.refine_2
        α : Type u_1
        p : α → Bool
        l✝ l : List α
        a : α
        a✝ : Iff (Eq (List.rtakeWhile p l) List.nil) (∀ (hl : Ne l List.nil), Not (Eq  …
        h : Not (Eq (HAppend.hAppend l (List.cons a List.nil)) List.nil) → Not (Eq (p  …
        ⊢ Eq (List.filter.match_1 (fun x => List α) (p a) (fun _ => List.cons a (List. …
      -/
    · simp [h]
      /-
        🎉 no goals
      -/


theorem mem_rtakeWhile_imp {x : α} (hx : x ∈ rtakeWhile p l) : p x := by
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    x : α
    hx : Membership.mem (List.rtakeWhile p l) x
    ⊢ Eq (p x) Bool.true
  -/
  rw [rtakeWhile, mem_reverse] at hx
  /-
    α : Type u_1
    p : α → Bool
    l : List α
    x : α
    hx : Membership.mem (List.takeWhile p l.reverse) x
    ⊢ Eq (p x) Bool.true
  -/
  exact mem_takeWhile_imp hx
  /-
    🎉 no goals
  -/


theorem rtakeWhile_idempotent (p : α → Bool) (l : List α) :
    rtakeWhile p (rtakeWhile p l) = rtakeWhile p l :=
  rtakeWhile_eq_self_iff.mpr fun _ => mem_rtakeWhile_imp


lemma rdrop_add (i j : ℕ) : (l.rdrop i).rdrop j = l.rdrop (i + j) := by
  /-
    α : Type u_1
    l : List α
    i j : Nat
    ⊢ Eq ((l.rdrop i).rdrop j) (l.rdrop (HAdd.hAdd i j))
  -/
  simp_rw [rdrop_eq_reverse_drop_reverse, reverse_reverse, drop_drop]
  /-
    🎉 no goals
  -/


@[simp]
lemma rdrop_append_length {l₁ l₂ : List α} :
    List.rdrop (l₁ ++ l₂) (List.length l₂) = l₁ := by
  rw [rdrop_eq_reverse_drop_reverse, ← length_reverse l₂,
      reverse_append, drop_left, reverse_reverse]


lemma rdrop_append_of_le_length {l₁ l₂ : List α} (k : ℕ) :
    k ≤ length l₂ → List.rdrop (l₁ ++ l₂) k = l₁ ++ List.rdrop l₂ k := by
  /-
    α : Type u_1
    l₁ l₂ : List α
    k : Nat
    ⊢ LE.le k l₂.length → Eq ((HAppend.hAppend l₁ l₂).rdrop k) (HAppend.hAppend l₁ …
  -/
  intro hk
  /-
    α : Type u_1
    l₁ l₂ : List α
    k : Nat
    hk : LE.le k l₂.length
    ⊢ Eq ((HAppend.hAppend l₁ l₂).rdrop k) (HAppend.hAppend l₁ (l₂.rdrop k))
  -/
  rw [← length_reverse] at hk
  rw [rdrop_eq_reverse_drop_reverse, reverse_append, drop_append_of_le_length hk,
    reverse_append, reverse_reverse, ← rdrop_eq_reverse_drop_reverse]


@[simp]
lemma rdrop_append_length_add {l₁ l₂ : List α} (k : ℕ) :
    List.rdrop (l₁ ++ l₂) (length l₂ + k) = List.rdrop l₁ k := by
  /-
    α : Type u_1
    l₁ l₂ : List α
    k : Nat
    ⊢ Eq ((HAppend.hAppend l₁ l₂).rdrop (HAdd.hAdd l₂.length k)) (l₁.rdrop k)
  -/
  rw [← rdrop_add, rdrop_append_length]
  /-
    🎉 no goals
  -/


