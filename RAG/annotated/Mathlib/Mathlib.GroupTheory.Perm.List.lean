/-- A list `l : List α` can be interpreted as an `Equiv.Perm α` where each element in the list
is permuted to the next one, defined as `formPerm`. When we have that `Nodup l`,
we prove that `Equiv.Perm.support (formPerm l) = l.toFinset`, and that
`formPerm l` is rotationally invariant, in `formPerm_rotate`.
-/
def formPerm : Equiv.Perm α :=
  (zipWith Equiv.swap l l.tail).prod


@[simp]
theorem formPerm_nil : formPerm ([] : List α) = 1 :=
  rfl


@[simp]
theorem formPerm_singleton (x : α) : formPerm [x] = 1 :=
  rfl


@[simp]
theorem formPerm_cons_cons (x y : α) (l : List α) :
    formPerm (x :: y :: l) = swap x y * formPerm (y :: l) :=
  prod_cons


theorem formPerm_pair (x y : α) : formPerm [x, y] = swap x y :=
  rfl


theorem mem_or_mem_of_zipWith_swap_prod_ne : ∀ {l l' : List α} {x : α},
    (zipWith swap l l').prod x ≠ x → x ∈ l ∨ x ∈ l'
                   /-
                     α : Type u_1
                     inst✝ : DecidableEq α
                     x✝¹ : List α
                     x✝ : α
                     ⊢ Ne ((List.zipWith Equiv.swap List.nil x✝¹).prod x✝) x✝ → Or (Membership.mem  …
                   -/
  | [], _, _ => by simp
                   /-
                     🎉 no goals
                   -/
                   /-
                     α : Type u_1
                     inst✝ : DecidableEq α
                     x✝¹ : List α
                     x✝ : α
                     ⊢ Ne ((List.zipWith Equiv.swap x✝¹ List.nil).prod x✝) x✝ → Or (Membership.mem  …
                   -/
  | _, [], _ => by simp
                   /-
                     🎉 no goals
                   -/
  | a::l, b::l', x => fun hx ↦
    if h : (zipWith swap l l').prod x = x then
                                                                     /-
                                                                       α : Type u_1
                                                                       inst✝ : DecidableEq α
                                                                       a : α
                                                                       l : List α
                                                                       b : α
                                                                       l' : List α
                                                                       x : α
                                                                       hx : Ne ((List.zipWith Equiv.swap (List.cons a l) (List.cons b l')).prod x) x
                                                                       h : Eq ((List.zipWith Equiv.swap l l').prod x) x
                                                                       ⊢ Ne ((Equiv.swap a b) x) x
                                                                     -/
      (eq_or_eq_of_swap_apply_ne_self (a := a) (b := b) (x := x) (by simpa [h] using hx)).imp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
            /-
              α : Type u_1
              inst✝ : DecidableEq α
              a : α
              l : List α
              b : α
              l' : List α
              x : α
              hx : Ne ((List.zipWith Equiv.swap (List.cons a l) (List.cons b l')).prod x) x
              h : Eq ((List.zipWith Equiv.swap l l').prod x) x
              ⊢ Eq x a → Membership.mem (List.cons a l) x
            -/
                        /-
                          🎉 no goals
                        -/
        (by rintro rfl; exact .head _) (by rintro rfl; exact .head _)
                                                       /-
                                                         🎉 no goals
                                                       -/
    else
     (mem_or_mem_of_zipWith_swap_prod_ne h).imp (.tail _) (.tail _)


theorem zipWith_swap_prod_support' (l l' : List α) :
    { x | (zipWith swap l l').prod x ≠ x } ≤ l.toFinset ⊔ l'.toFinset := fun _ h ↦ by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    x✝ : α
    h : Membership.mem (setOf fun x => Ne ((List.zipWith Equiv.swap l l').prod x)  …
    ⊢ Membership.mem (↑(Max.max l.toFinset l'.toFinset)) x✝
  -/
  simpa using mem_or_mem_of_zipWith_swap_prod_ne h
  /-
    🎉 no goals
  -/


theorem zipWith_swap_prod_support [Fintype α] (l l' : List α) :
    (zipWith swap l l').prod.support ≤ l.toFinset ⊔ l'.toFinset := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l l' : List α
    ⊢ LE.le (List.zipWith Equiv.swap l l').prod.support (Max.max l.toFinset l'.toF …
  -/
  intro x hx
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l l' : List α
    x : α
    hx : Membership.mem (List.zipWith Equiv.swap l l').prod.support x
    ⊢ Membership.mem (Max.max l.toFinset l'.toFinset) x
  -/
  have hx' : x ∈ { x | (zipWith swap l l').prod x ≠ x } := by simpa using hx
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l l' : List α
    x : α
    hx : Membership.mem (List.zipWith Equiv.swap l l').prod.support x
    hx' : Membership.mem (setOf fun x => Ne ((List.zipWith Equiv.swap l l').prod x …
    ⊢ Membership.mem (Max.max l.toFinset l'.toFinset) x
  -/
  simpa using zipWith_swap_prod_support' _ _ hx'
  /-
    🎉 no goals
  -/


theorem support_formPerm_le' : { x | formPerm l x ≠ x } ≤ l.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    ⊢ LE.le (setOf fun x => Ne (l.formPerm x) x) ↑l.toFinset
  -/
  refine (zipWith_swap_prod_support' l l.tail).trans ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    ⊢ LE.le ↑(Max.max l.toFinset l.tail.toFinset) ↑l.toFinset
  -/
  simpa [Finset.subset_iff] using tail_subset l
  /-
    🎉 no goals
  -/


theorem support_formPerm_le [Fintype α] : support (formPerm l) ≤ l.toFinset := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    l : List α
    inst✝ : Fintype α
    ⊢ LE.le l.formPerm.support l.toFinset
  -/
  intro x hx
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    l : List α
    inst✝ : Fintype α
    x : α
    hx : Membership.mem l.formPerm.support x
    ⊢ Membership.mem l.toFinset x
  -/
  have hx' : x ∈ { x | formPerm l x ≠ x } := by simpa using hx
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    l : List α
    inst✝ : Fintype α
    x : α
    hx : Membership.mem l.formPerm.support x
    hx' : Membership.mem (setOf fun x => Ne (l.formPerm x) x) x
    ⊢ Membership.mem l.toFinset x
  -/
  simpa using support_formPerm_le' _ hx'
  /-
    🎉 no goals
  -/


theorem mem_of_formPerm_apply_ne (h : l.formPerm x ≠ x) : x ∈ l := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    h : Ne (l.formPerm x) x
    ⊢ Membership.mem l x
  -/
  simpa [or_iff_left_of_imp mem_of_mem_tail] using mem_or_mem_of_zipWith_swap_prod_ne h
  /-
    🎉 no goals
  -/


theorem formPerm_apply_of_not_mem (h : x ∉ l) : formPerm l x = x :=
  not_imp_comm.1 mem_of_formPerm_apply_ne h


theorem formPerm_apply_mem_of_mem (h : x ∈ l) : formPerm l x ∈ l := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    h : Membership.mem l x
    ⊢ Membership.mem l (l.formPerm x)
  -/
  cases' l with y l
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      x : α
      h : Membership.mem List.nil x
      ⊢ Membership.mem List.nil (List.nil.formPerm x)
    -/
  · simp at h
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    x y : α
    l : List α
    h : Membership.mem (List.cons y l) x
    ⊢ Membership.mem (List.cons y l) ((List.cons y l).formPerm x)
  -/
  induction' l with z l IH generalizing x y
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      x y : α
      h : Membership.mem (List.cons y List.nil) x
      ⊢ Membership.mem (List.cons y List.nil) ((List.cons y List.nil).formPerm x)
    -/
  · simpa using h
    /-
      🎉 no goals
    -/
    /-
      case cons.cons
      α : Type u_1
      inst✝ : DecidableEq α
      z : α
      l : List α
      IH : ∀ {x : α} (y : α), Membership.mem (List.cons y l) x → Membership.mem (Lis …
      x y : α
      h : Membership.mem (List.cons y (List.cons z l)) x
      ⊢ Membership.mem (List.cons y (List.cons z l)) ((List.cons y (List.cons z l)). …
    -/
  · by_cases hx : x ∈ z :: l
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        z : α
        l : List α
        IH : ∀ {x : α} (y : α), Membership.mem (List.cons y l) x → Membership.mem (Lis …
        x y : α
        h : Membership.mem (List.cons y (List.cons z l)) x
        hx : Membership.mem (List.cons z l) x
        ⊢ Membership.mem (List.cons y (List.cons z l)) ((List.cons y (List.cons z l)). …
      -/
    · rw [formPerm_cons_cons, mul_apply, swap_apply_def]
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        z : α
        l : List α
        IH : ∀ {x : α} (y : α), Membership.mem (List.cons y l) x → Membership.mem (Lis …
        x y : α
        h : Membership.mem (List.cons y (List.cons z l)) x
        hx : Membership.mem (List.cons z l) x
        ⊢ Membership.mem (List.cons y (List.cons z l)) (ite (Eq ((List.cons z l).formP …
      -/
      split_ifs
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          z : α
          l : List α
          IH : ∀ {x : α} (y : α), Membership.mem (List.cons y l) x → Membership.mem (Lis …
          x y : α
          h : Membership.mem (List.cons y (List.cons z l)) x
          hx : Membership.mem (List.cons z l) x
          h✝ : Eq ((List.cons z l).formPerm x) y
          ⊢ Membership.mem (List.cons y (List.cons z l)) z
        -/
      · simp [IH _ hx]
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          z : α
          l : List α
          IH : ∀ {x : α} (y : α), Membership.mem (List.cons y l) x → Membership.mem (Lis …
          x y : α
          h : Membership.mem (List.cons y (List.cons z l)) x
          hx : Membership.mem (List.cons z l) x
          h✝¹ : Not (Eq ((List.cons z l).formPerm x) y)
          h✝ : Eq ((List.cons z l).formPerm x) z
          ⊢ Membership.mem (List.cons y (List.cons z l)) y
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          inst✝ : DecidableEq α
          z : α
          l : List α
          IH : ∀ {x : α} (y : α), Membership.mem (List.cons y l) x → Membership.mem (Lis …
          x y : α
          h : Membership.mem (List.cons y (List.cons z l)) x
          hx : Membership.mem (List.cons z l) x
          h✝¹ : Not (Eq ((List.cons z l).formPerm x) y)
          h✝ : Not (Eq ((List.cons z l).formPerm x) z)
          ⊢ Membership.mem (List.cons y (List.cons z l)) ((List.cons z l).formPerm x)
        -/
      · simp [*]
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        z : α
        l : List α
        IH : ∀ {x : α} (y : α), Membership.mem (List.cons y l) x → Membership.mem (Lis …
        x y : α
        h : Membership.mem (List.cons y (List.cons z l)) x
        hx : Not (Membership.mem (List.cons z l) x)
        ⊢ Membership.mem (List.cons y (List.cons z l)) ((List.cons y (List.cons z l)). …
      -/
    · replace h : x = y := Or.resolve_right (mem_cons.1 h) hx
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        z : α
        l : List α
        IH : ∀ {x : α} (y : α), Membership.mem (List.cons y l) x → Membership.mem (Lis …
        x y : α
        hx : Not (Membership.mem (List.cons z l) x)
        h : Eq x y
        ⊢ Membership.mem (List.cons y (List.cons z l)) ((List.cons y (List.cons z l)). …
      -/
      simp [formPerm_apply_of_not_mem hx, ← h]
      /-
        🎉 no goals
      -/


theorem mem_of_formPerm_apply_mem (h : l.formPerm x ∈ l) : x ∈ l := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    h : Membership.mem l (l.formPerm x)
    ⊢ Membership.mem l x
  -/
  contrapose h
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    h : Not (Membership.mem l x)
    ⊢ Not (Membership.mem l (l.formPerm x))
  -/
  rwa [formPerm_apply_of_not_mem h]
  /-
    🎉 no goals
  -/


@[simp]
theorem formPerm_mem_iff_mem : l.formPerm x ∈ l ↔ x ∈ l :=
  ⟨l.mem_of_formPerm_apply_mem, l.formPerm_apply_mem_of_mem⟩


@[simp]
theorem formPerm_cons_concat_apply_last (x y : α) (xs : List α) :
    formPerm (x :: (xs ++ [y])) y = x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y : α
    xs : List α
    ⊢ Eq ((List.cons x (HAppend.hAppend xs (List.cons y List.nil))).formPerm y) x
  -/
  induction' xs with z xs IH generalizing x y
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      x y : α
      ⊢ Eq ((List.cons x (HAppend.hAppend List.nil (List.cons y List.nil))).formPerm …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      z : α
      xs : List α
      IH : ∀ (x y : α), Eq ((List.cons x (HAppend.hAppend xs (List.cons y List.nil)) …
      x y : α
      ⊢ Eq ((List.cons x (HAppend.hAppend (List.cons z xs) (List.cons y List.nil))). …
    -/
  · simp [IH]
    /-
      🎉 no goals
    -/


@[simp]
theorem formPerm_apply_getLast (x : α) (xs : List α) :
    formPerm (x :: xs) ((x :: xs).getLast (cons_ne_nil x xs)) = x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    xs : List α
    ⊢ Eq ((List.cons x xs).formPerm ((List.cons x xs).getLast ⋯)) x
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  induction' xs using List.reverseRecOn with xs y _ generalizing x <;> simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem formPerm_apply_getElem_length (x : α) (xs : List α) :
    formPerm (x :: xs) (x :: xs)[xs.length] = x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    xs : List α
    ⊢ Eq ((List.cons x xs).formPerm (GetElem.getElem (List.cons x xs) xs.length ⋯) …
  -/
  rw [getElem_cons_length _ _ _ rfl, formPerm_apply_getLast]
  /-
    🎉 no goals
  -/


@[deprecated formPerm_apply_getElem_length (since := "2024-08-03")]
theorem formPerm_apply_get_length (x : α) (xs : List α) :
                                                            /-
                                                              α : Type u_1
                                                              β : Type u_2
                                                              inst✝ : DecidableEq α
                                                              l : List α
                                                              x✝ x : α
                                                              xs : List α
                                                              ⊢ LT.lt xs.length (List.cons x xs).length
                                                            -/
    formPerm (x :: xs) ((x :: xs).get (Fin.mk xs.length (by simp))) = x :=
                                                            /-
                                                              🎉 no goals
                                                            -/
  formPerm_apply_getElem_length ..


theorem formPerm_apply_head (x y : α) (xs : List α) (h : Nodup (x :: y :: xs)) :
                                        /-
                                          α : Type u_1
                                          inst✝ : DecidableEq α
                                          x y : α
                                          xs : List α
                                          h : (List.cons x (List.cons y xs)).Nodup
                                          ⊢ Eq ((List.cons x (List.cons y xs)).formPerm x) y
                                        -/
    formPerm (x :: y :: xs) x = y := by simp [formPerm_apply_of_not_mem h.not_mem]
                                        /-
                                          🎉 no goals
                                        -/


theorem formPerm_apply_getElem_zero (l : List α) (h : Nodup l) (hl : 1 < l.length) :
    formPerm l l[0] = l[1] := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    hl : LT.lt 1 l.length
    ⊢ Eq (l.formPerm (GetElem.getElem l 0 ⋯)) (GetElem.getElem l 1 hl)
  -/
  rcases l with (_ | ⟨x, _ | ⟨y, tl⟩⟩)
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      h : List.nil.Nodup
      hl : LT.lt 1 List.nil.length
      ⊢ Eq (List.nil.formPerm (GetElem.getElem List.nil 0 ⋯)) (GetElem.getElem List. …
    -/
  · simp at hl
    /-
      🎉 no goals
    -/
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      x : α
      h : (List.cons x List.nil).Nodup
      hl : LT.lt 1 (List.cons x List.nil).length
      ⊢ Eq ((List.cons x List.nil).formPerm (GetElem.getElem (List.cons x List.nil)  …
    -/
  · simp at hl
    /-
      🎉 no goals
    -/
    /-
      case cons.cons
      α : Type u_1
      inst✝ : DecidableEq α
      x y : α
      tl : List α
      h : (List.cons x (List.cons y tl)).Nodup
      hl : LT.lt 1 (List.cons x (List.cons y tl)).length
      ⊢ Eq ((List.cons x (List.cons y tl)).formPerm (GetElem.getElem (List.cons x (L …
    -/
  · rw [getElem_cons_zero, formPerm_apply_head _ _ _ h, getElem_cons_succ, getElem_cons_zero]
    /-
      🎉 no goals
    -/


@[deprecated formPerm_apply_getElem_zero (since := "2024-08-03")]
theorem formPerm_apply_get_zero (l : List α) (h : Nodup l) (hl : 1 < l.length) :
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      inst✝ : DecidableEq α
                                      l✝ : List α
                                      x : α
                                      l : List α
                                      h : l.Nodup
                                      hl : LT.lt 1 l.length
                                      ⊢ LT.lt 0 l.length
                                    -/
    formPerm l (l.get (Fin.mk 0 (by omega))) = l.get (Fin.mk 1 hl) :=
                                    /-
                                      🎉 no goals
                                    -/
  formPerm_apply_getElem_zero l h hl


theorem formPerm_eq_head_iff_eq_getLast (x y : α) :
    formPerm (y :: l) x = y ↔ x = getLast (y :: l) (cons_ne_nil _ _) :=
                /-
                  α : Type u_1
                  inst✝ : DecidableEq α
                  l : List α
                  x y : α
                  ⊢ Iff (Eq ((List.cons y l).formPerm x) y) (Eq ((List.cons y l).formPerm x) ((L …
                -/
  Iff.trans (by rw [formPerm_apply_getLast]) (formPerm (y :: l)).injective.eq_iff
                /-
                  🎉 no goals
                -/


theorem formPerm_apply_lt_getElem (xs : List α) (h : Nodup xs) (n : ℕ) (hn : n + 1 < xs.length) :
    formPerm xs xs[n] = xs[n + 1] := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    h : xs.Nodup
    n : Nat
    hn : LT.lt (HAdd.hAdd n 1) xs.length
    ⊢ Eq (xs.formPerm (GetElem.getElem xs n ⋯)) (GetElem.getElem xs (HAdd.hAdd n 1 …
  -/
  induction' n with n IH generalizing xs
    /-
      case zero
      α : Type u_1
      inst✝ : DecidableEq α
      xs : List α
      h : xs.Nodup
      hn : LT.lt (HAdd.hAdd 0 1) xs.length
      ⊢ Eq (xs.formPerm (GetElem.getElem xs 0 ⋯)) (GetElem.getElem xs (HAdd.hAdd 0 1 …
    -/
  · simpa using formPerm_apply_getElem_zero _ h _
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      inst✝ : DecidableEq α
      n : Nat
      IH : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq  …
      xs : List α
      h : xs.Nodup
      hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) xs.length
      ⊢ Eq (xs.formPerm (GetElem.getElem xs (HAdd.hAdd n 1) ⋯)) (GetElem.getElem xs  …
    -/
  · rcases xs with (_ | ⟨x, _ | ⟨y, l⟩⟩)
      /-
        case succ.nil
        α : Type u_1
        inst✝ : DecidableEq α
        n : Nat
        IH : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq  …
        h : List.nil.Nodup
        hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) List.nil.length
        ⊢ Eq (List.nil.formPerm (GetElem.getElem List.nil (HAdd.hAdd n 1) ⋯)) (GetElem …
      -/
    · simp at hn
      /-
        🎉 no goals
      -/
      /-
        case succ.cons.nil
        α : Type u_1
        inst✝ : DecidableEq α
        n : Nat
        IH : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq  …
        x : α
        h : (List.cons x List.nil).Nodup
        hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (List.cons x List.nil).length
        ⊢ Eq ((List.cons x List.nil).formPerm (GetElem.getElem (List.cons x List.nil)  …
      -/
    · rw [formPerm_singleton, getElem_singleton, getElem_singleton, one_apply]
      /-
        🎉 no goals
      -/
      /-
        case succ.cons.cons
        α : Type u_1
        inst✝ : DecidableEq α
        n : Nat
        IH : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq  …
        x y : α
        l : List α
        h : (List.cons x (List.cons y l)).Nodup
        hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (List.cons x (List.cons y l)).length
        ⊢ Eq ((List.cons x (List.cons y l)).formPerm (GetElem.getElem (List.cons x (Li …
      -/
    · specialize IH (y :: l) h.of_cons _
        /-
          case succ.cons.cons
          α : Type u_1
          inst✝ : DecidableEq α
          n : Nat
          IH : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq  …
          x y : α
          l : List α
          h : (List.cons x (List.cons y l)).Nodup
          hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (List.cons x (List.cons y l)).length
          ⊢ LT.lt (HAdd.hAdd n 1) (List.cons y l).length
        -/
      · simpa [Nat.succ_lt_succ_iff] using hn
        /-
          🎉 no goals
        -/
      /-
        case succ.cons.cons
        α : Type u_1
        inst✝ : DecidableEq α
        n : Nat
        IH✝ : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq …
        x y : α
        l : List α
        h : (List.cons x (List.cons y l)).Nodup
        hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (List.cons x (List.cons y l)).length
        IH : Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) (GetE …
        ⊢ Eq ((List.cons x (List.cons y l)).formPerm (GetElem.getElem (List.cons x (Li …
      -/
      simp only [swap_apply_eq_iff, coe_mul, formPerm_cons_cons, Function.comp]
      /-
        case succ.cons.cons
        α : Type u_1
        inst✝ : DecidableEq α
        n : Nat
        IH✝ : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq …
        x y : α
        l : List α
        h : (List.cons x (List.cons y l)).Nodup
        hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (List.cons x (List.cons y l)).length
        IH : Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) (GetE …
        ⊢ Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons x (List.cons y l))  …
      -/
      simp only [getElem_cons_succ] at *
      /-
        case succ.cons.cons
        α : Type u_1
        inst✝ : DecidableEq α
        n : Nat
        IH✝ : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq …
        x y : α
        l : List α
        h : (List.cons x (List.cons y l)).Nodup
        hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (List.cons x (List.cons y l)).length
        IH : Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) (GetE …
        ⊢ Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) ((Equiv. …
      -/
      rw [← IH, swap_apply_of_ne_of_ne] <;>
        /-
          case succ.cons.cons.a
          α : Type u_1
          inst✝ : DecidableEq α
          n : Nat
          IH✝ : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq …
          x y : α
          l : List α
          h : (List.cons x (List.cons y l)).Nodup
          hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (List.cons x (List.cons y l)).length
          IH : Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) (GetE …
          ⊢ Ne ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) x
        -/
        /-
          case succ.cons.cons.a
          α : Type u_1
          inst✝ : DecidableEq α
          n : Nat
          IH✝ : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq …
          x y : α
          l : List α
          h : (List.cons x (List.cons y l)).Nodup
          hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (List.cons x (List.cons y l)).length
          IH : Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) (GetE …
          hx : Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) x
          ⊢ False
        -/
        /-
          case succ.cons.cons.a
          α : Type u_1
          inst✝ : DecidableEq α
          n : Nat
          IH✝ : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq …
          x y : α
          l : List α
          hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (List.cons x (List.cons y l)).length
          h : (List.cons (GetElem.getElem l n ⋯) (List.cons y l)).Nodup
          IH : Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) (GetE …
          hx : Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) x
          ⊢ False
        -/
        /-
          🎉 no goals
        -/
        rw [← hx, IH] at h
        /-
          case succ.cons.cons.a
          α : Type u_1
          inst✝ : DecidableEq α
          n : Nat
          IH✝ : ∀ (xs : List α), xs.Nodup → ∀ (hn : LT.lt (HAdd.hAdd n 1) xs.length), Eq …
          x y : α
          l : List α
          hn : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (List.cons x (List.cons y l)).length
          h : (List.cons x (List.cons (GetElem.getElem l n ⋯) l)).Nodup
          IH : Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) (GetE …
          hx : Eq ((List.cons y l).formPerm (GetElem.getElem (List.cons y l) n ⋯)) y
          ⊢ False
        -/
        simp [getElem_mem] at h
        /-
          🎉 no goals
        -/


@[deprecated formPerm_apply_lt_getElem (since := "2024-08-03")]
theorem formPerm_apply_lt_get (xs : List α) (h : Nodup xs) (n : ℕ) (hn : n + 1 < xs.length) :
    formPerm xs (xs.get (Fin.mk n ((Nat.lt_succ_self n).trans hn))) =
      xs.get (Fin.mk (n + 1) hn) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    h : xs.Nodup
    n : Nat
    hn : LT.lt (HAdd.hAdd n 1) xs.length
    ⊢ Eq (xs.formPerm (xs.get ⟨n, ⋯⟩)) (xs.get ⟨HAdd.hAdd n 1, hn⟩)
  -/
  simp_all [formPerm_apply_lt_getElem]
  /-
    🎉 no goals
  -/


theorem formPerm_apply_getElem (xs : List α) (w : Nodup xs) (i : ℕ) (h : i < xs.length) :
    formPerm xs xs[i] =
      xs[(i + 1) % xs.length]'(Nat.mod_lt _ (i.zero_le.trans_lt h)) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    w : xs.Nodup
    i : Nat
    h : LT.lt i xs.length
    ⊢ Eq (xs.formPerm (GetElem.getElem xs i h)) (GetElem.getElem xs (HMod.hMod (HA …
  -/
  cases' xs with x xs
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      i : Nat
      w : List.nil.Nodup
      h : LT.lt i List.nil.length
      ⊢ Eq (List.nil.formPerm (GetElem.getElem List.nil i h)) (GetElem.getElem List. …
    -/
  · simp at h
    /-
      🎉 no goals
    -/
  · have : i ≤ xs.length := by
      refine Nat.le_of_lt_succ ?_
      simpa using h
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      i : Nat
      x : α
      xs : List α
      w : (List.cons x xs).Nodup
      h : LT.lt i (List.cons x xs).length
      this : LE.le i xs.length
      ⊢ Eq ((List.cons x xs).formPerm (GetElem.getElem (List.cons x xs) i h)) (GetEl …
    -/
    rcases this.eq_or_lt with (rfl | hn')
      /-
        case cons.inl
        α : Type u_1
        inst✝ : DecidableEq α
        x : α
        xs : List α
        w : (List.cons x xs).Nodup
        h : LT.lt xs.length (List.cons x xs).length
        this : LE.le xs.length xs.length
        ⊢ Eq ((List.cons x xs).formPerm (GetElem.getElem (List.cons x xs) xs.length h) …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case cons.inr
        α : Type u_1
        inst✝ : DecidableEq α
        i : Nat
        x : α
        xs : List α
        w : (List.cons x xs).Nodup
        h : LT.lt i (List.cons x xs).length
        this : LE.le i xs.length
        hn' : LT.lt i xs.length
        ⊢ Eq ((List.cons x xs).formPerm (GetElem.getElem (List.cons x xs) i h)) (GetEl …
      -/
    · rw [formPerm_apply_lt_getElem (x :: xs) w _ (Nat.succ_lt_succ hn')]
      /-
        case cons.inr
        α : Type u_1
        inst✝ : DecidableEq α
        i : Nat
        x : α
        xs : List α
        w : (List.cons x xs).Nodup
        h : LT.lt i (List.cons x xs).length
        this : LE.le i xs.length
        hn' : LT.lt i xs.length
        ⊢ Eq (GetElem.getElem (List.cons x xs) (HAdd.hAdd i 1) ⋯) (GetElem.getElem (Li …
      -/
      congr
      /-
        case cons.inr.e_i
        α : Type u_1
        inst✝ : DecidableEq α
        i : Nat
        x : α
        xs : List α
        w : (List.cons x xs).Nodup
        h : LT.lt i (List.cons x xs).length
        this : LE.le i xs.length
        hn' : LT.lt i xs.length
        ⊢ Eq (HAdd.hAdd i 1) (HMod.hMod (HAdd.hAdd i 1) (List.cons x xs).length)
      -/
      rw [Nat.mod_eq_of_lt]; simpa [Nat.succ_eq_add_one]
                             /-
                               🎉 no goals
                             -/


@[deprecated formPerm_apply_getElem (since := "2024-08-03")]
theorem formPerm_apply_get (xs : List α) (h : Nodup xs) (i : Fin xs.length) :
    formPerm xs (xs.get i) =
      xs.get ⟨((i.val + 1) % xs.length), (Nat.mod_lt _ (i.val.zero_le.trans_lt i.isLt))⟩ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    h : xs.Nodup
    i : Fin xs.length
    ⊢ Eq (xs.formPerm (xs.get i)) (xs.get ⟨HMod.hMod (HAdd.hAdd (↑i) 1) xs.length, …
  -/
  simp [formPerm_apply_getElem, h]
  /-
    🎉 no goals
  -/


theorem support_formPerm_of_nodup' (l : List α) (h : Nodup l) (h' : ∀ x : α, l ≠ [x]) :
    { x | formPerm l x ≠ x } = l.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    h' : ∀ (x : α), Ne l (List.cons x List.nil)
    ⊢ Eq (setOf fun x => Ne (l.formPerm x) x) ↑l.toFinset
  -/
  apply _root_.le_antisymm
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : ∀ (x : α), Ne l (List.cons x List.nil)
      ⊢ LE.le (setOf fun x => Ne (l.formPerm x) x) ↑l.toFinset
    -/
  · exact support_formPerm_le' l
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : ∀ (x : α), Ne l (List.cons x List.nil)
      ⊢ LE.le (↑l.toFinset) (setOf fun x => Ne (l.formPerm x) x)
    -/
  · intro x hx
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : ∀ (x : α), Ne l (List.cons x List.nil)
      x : α
      hx : Membership.mem (↑l.toFinset) x
      ⊢ Membership.mem (setOf fun x => Ne (l.formPerm x) x) x
    -/
    simp only [Finset.mem_coe, mem_toFinset] at hx
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : ∀ (x : α), Ne l (List.cons x List.nil)
      x : α
      hx : Membership.mem l x
      ⊢ Membership.mem (setOf fun x => Ne (l.formPerm x) x) x
    -/
    obtain ⟨n, hn, rfl⟩ := getElem_of_mem hx
    /-
      case a.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : ∀ (x : α), Ne l (List.cons x List.nil)
      n : Nat
      hn : LT.lt n l.length
      hx : Membership.mem l (GetElem.getElem l n hn)
      ⊢ Membership.mem (setOf fun x => Ne (l.formPerm x) x) (GetElem.getElem l n hn)
    -/
    rw [Set.mem_setOf_eq, formPerm_apply_getElem _ h]
    /-
      case a.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : ∀ (x : α), Ne l (List.cons x List.nil)
      n : Nat
      hn : LT.lt n l.length
      hx : Membership.mem l (GetElem.getElem l n hn)
      ⊢ Ne (GetElem.getElem l (HMod.hMod (HAdd.hAdd n 1) l.length) ⋯) (GetElem.getEl …
    -/
    intro H
    /-
      case a.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : ∀ (x : α), Ne l (List.cons x List.nil)
      n : Nat
      hn : LT.lt n l.length
      hx : Membership.mem l (GetElem.getElem l n hn)
      H : Eq (GetElem.getElem l (HMod.hMod (HAdd.hAdd n 1) l.length) ⋯) (GetElem.get …
      ⊢ False
    -/
    rw [nodup_iff_injective_get, Function.Injective] at h
    /-
      case a.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : ∀ ⦃a₁ a₂ : Fin l.length⦄, Eq (l.get a₁) (l.get a₂) → Eq a₁ a₂
      h' : ∀ (x : α), Ne l (List.cons x List.nil)
      n : Nat
      hn : LT.lt n l.length
      hx : Membership.mem l (GetElem.getElem l n hn)
      H : Eq (GetElem.getElem l (HMod.hMod (HAdd.hAdd n 1) l.length) ⋯) (GetElem.get …
      ⊢ False
    -/
    specialize h H
    /-
      case a.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h' : ∀ (x : α), Ne l (List.cons x List.nil)
      n : Nat
      hn : LT.lt n l.length
      hx : Membership.mem l (GetElem.getElem l n hn)
      H : Eq (GetElem.getElem l (HMod.hMod (HAdd.hAdd n 1) l.length) ⋯) (GetElem.get …
      h : Eq ⟨HMod.hMod (HAdd.hAdd n 1) l.length, ⋯⟩ ⟨n, hn⟩
      ⊢ False
    -/
    rcases (Nat.succ_le_of_lt hn).eq_or_lt with hn' | hn'
      /-
        case a.intro.intro.inl
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        h' : ∀ (x : α), Ne l (List.cons x List.nil)
        n : Nat
        hn : LT.lt n l.length
        hx : Membership.mem l (GetElem.getElem l n hn)
        H : Eq (GetElem.getElem l (HMod.hMod (HAdd.hAdd n 1) l.length) ⋯) (GetElem.get …
        h : Eq ⟨HMod.hMod (HAdd.hAdd n 1) l.length, ⋯⟩ ⟨n, hn⟩
        hn' : Eq n.succ l.length
        ⊢ False
      -/
    · simp only [← hn', Nat.mod_self] at h
      /-
        case a.intro.intro.inl
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        h' : ∀ (x : α), Ne l (List.cons x List.nil)
        n : Nat
        hn : LT.lt n l.length
        hx : Membership.mem l (GetElem.getElem l n hn)
        H : Eq (GetElem.getElem l (HMod.hMod (HAdd.hAdd n 1) l.length) ⋯) (GetElem.get …
        hn' : Eq n.succ l.length
        h : Eq ⟨0, ⋯⟩ ⟨n, hn⟩
        ⊢ False
      -/
      refine not_exists.mpr h' ?_
      /-
        case a.intro.intro.inl
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        h' : ∀ (x : α), Ne l (List.cons x List.nil)
        n : Nat
        hn : LT.lt n l.length
        hx : Membership.mem l (GetElem.getElem l n hn)
        H : Eq (GetElem.getElem l (HMod.hMod (HAdd.hAdd n 1) l.length) ⋯) (GetElem.get …
        hn' : Eq n.succ l.length
        h : Eq ⟨0, ⋯⟩ ⟨n, hn⟩
        ⊢ Exists fun x => Eq l (List.cons x List.nil)
      -/
      rw [← length_eq_one, ← hn', (Fin.mk.inj_iff.mp h).symm]
      /-
        🎉 no goals
      -/
      /-
        case a.intro.intro.inr
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        h' : ∀ (x : α), Ne l (List.cons x List.nil)
        n : Nat
        hn : LT.lt n l.length
        hx : Membership.mem l (GetElem.getElem l n hn)
        H : Eq (GetElem.getElem l (HMod.hMod (HAdd.hAdd n 1) l.length) ⋯) (GetElem.get …
        h : Eq ⟨HMod.hMod (HAdd.hAdd n 1) l.length, ⋯⟩ ⟨n, hn⟩
        hn' : LT.lt n.succ l.length
        ⊢ False
      -/
    · simp [Nat.mod_eq_of_lt hn'] at h
      /-
        🎉 no goals
      -/


theorem support_formPerm_of_nodup [Fintype α] (l : List α) (h : Nodup l) (h' : ∀ x : α, l ≠ [x]) :
    support (formPerm l) = l.toFinset := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l : List α
    h : l.Nodup
    h' : ∀ (x : α), Ne l (List.cons x List.nil)
    ⊢ Eq l.formPerm.support l.toFinset
  -/
  rw [← Finset.coe_inj]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l : List α
    h : l.Nodup
    h' : ∀ (x : α), Ne l (List.cons x List.nil)
    ⊢ Eq ↑l.formPerm.support ↑l.toFinset
  -/
  convert support_formPerm_of_nodup' _ h h'
  /-
    case h.e'_2
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l : List α
    h : l.Nodup
    h' : ∀ (x : α), Ne l (List.cons x List.nil)
    ⊢ Eq (↑l.formPerm.support) (setOf fun x => Ne (l.formPerm x) x)
  -/
  simp [Set.ext_iff]
  /-
    🎉 no goals
  -/


theorem formPerm_rotate_one (l : List α) (h : Nodup l) : formPerm (l.rotate 1) = formPerm l := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    ⊢ Eq (l.rotate 1).formPerm l.formPerm
  -/
  have h' : Nodup (l.rotate 1) := by simpa using h
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    h' : (l.rotate 1).Nodup
    ⊢ Eq (l.rotate 1).formPerm l.formPerm
  -/
  ext x
  /-
    case H
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    h' : (l.rotate 1).Nodup
    x : α
    ⊢ Eq ((l.rotate 1).formPerm x) (l.formPerm x)
  -/
  by_cases hx : x ∈ l.rotate 1
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : (l.rotate 1).Nodup
      x : α
      hx : Membership.mem (l.rotate 1) x
      ⊢ Eq ((l.rotate 1).formPerm x) (l.formPerm x)
    -/
  · obtain ⟨k, hk, rfl⟩ := getElem_of_mem hx
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : (l.rotate 1).Nodup
      k : Nat
      hk : LT.lt k (l.rotate 1).length
      hx : Membership.mem (l.rotate 1) (GetElem.getElem (l.rotate 1) k hk)
      ⊢ Eq ((l.rotate 1).formPerm (GetElem.getElem (l.rotate 1) k hk)) (l.formPerm ( …
    -/
    rw [formPerm_apply_getElem _ h', getElem_rotate l, getElem_rotate l, formPerm_apply_getElem _ h]
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : (l.rotate 1).Nodup
      k : Nat
      hk : LT.lt k (l.rotate 1).length
      hx : Membership.mem (l.rotate 1) (GetElem.getElem (l.rotate 1) k hk)
      ⊢ Eq (GetElem.getElem l (HMod.hMod (HAdd.hAdd (HMod.hMod (HAdd.hAdd k 1) (l.ro …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : (l.rotate 1).Nodup
      x : α
      hx : Not (Membership.mem (l.rotate 1) x)
      ⊢ Eq ((l.rotate 1).formPerm x) (l.formPerm x)
    -/
  · rw [formPerm_apply_of_not_mem hx, formPerm_apply_of_not_mem]
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      h' : (l.rotate 1).Nodup
      x : α
      hx : Not (Membership.mem (l.rotate 1) x)
      ⊢ Not (Membership.mem l x)
    -/
    simpa using hx
    /-
      🎉 no goals
    -/


theorem formPerm_rotate (l : List α) (h : Nodup l) (n : ℕ) :
    formPerm (l.rotate n) = formPerm l := by
  induction n with
  | zero => simp
  | succ n hn =>
    rw [← rotate_rotate, formPerm_rotate_one, hn]
    rwa [IsRotated.nodup_iff]
    exact IsRotated.forall l n


theorem formPerm_eq_of_isRotated {l l' : List α} (hd : Nodup l) (h : l ~r l') :
    formPerm l = formPerm l' := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    hd : l.Nodup
    h : l.IsRotated l'
    ⊢ Eq l.formPerm l'.formPerm
  -/
  obtain ⟨n, rfl⟩ := h
  /-
    case intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hd : l.Nodup
    n : Nat
    ⊢ Eq l.formPerm (l.rotate n).formPerm
  -/
  exact (formPerm_rotate l hd n).symm
  /-
    🎉 no goals
  -/


theorem formPerm_append_pair : ∀ (l : List α) (a b : α),
    formPerm (l ++ [a, b]) = formPerm (l ++ [a]) * swap a b
  | [], _, _ => rfl
  | [_], _, _ => rfl
  | x::y::l, a, b => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      x y : α
      l : List α
      a b : α
      ⊢ Eq (HAppend.hAppend (List.cons x (List.cons y l)) (List.cons a (List.cons b  …
    -/
    simpa [mul_assoc] using formPerm_append_pair (y::l) a b
    /-
      🎉 no goals
    -/


theorem formPerm_reverse : ∀ l : List α, formPerm l.reverse = (formPerm l)⁻¹
  | [] => rfl
  | [_] => rfl
  | a::b::l => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      l : List α
      ⊢ Eq (List.cons a (List.cons b l)).reverse.formPerm (Inv.inv (List.cons a (Lis …
    -/
    simp [formPerm_append_pair, swap_comm, ← formPerm_reverse (b::l)]
    /-
      🎉 no goals
    -/


theorem formPerm_pow_apply_getElem (l : List α) (w : Nodup l) (n : ℕ) (i : ℕ) (h : i < l.length) :
    (formPerm l ^ n) l[i] =
      l[(i + n) % l.length]'(Nat.mod_lt _ (i.zero_le.trans_lt h)) := by
  induction n with
  | zero => simp [Nat.mod_eq_of_lt h]
  | succ n hn =>
    simp [pow_succ', mul_apply, hn, formPerm_apply_getElem _ w, Nat.succ_eq_add_one,
      ← Nat.add_assoc]


@[deprecated formPerm_pow_apply_getElem (since := "2024-08-03")]
theorem formPerm_pow_apply_get (l : List α) (h : Nodup l) (n : ℕ) (i : Fin l.length) :
    (formPerm l ^ n) (l.get i) =
      l.get ⟨((i.val + n) % l.length), (Nat.mod_lt _ (i.val.zero_le.trans_lt i.isLt))⟩ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    n : Nat
    i : Fin l.length
    ⊢ Eq ((HPow.hPow l.formPerm n) (l.get i)) (l.get ⟨HMod.hMod (HAdd.hAdd (↑i) n) …
  -/
  simp [formPerm_pow_apply_getElem, h]
  /-
    🎉 no goals
  -/


theorem formPerm_pow_apply_head (x : α) (l : List α) (h : Nodup (x :: l)) (n : ℕ) :
    (formPerm (x :: l) ^ n) x =
      (x :: l)[(n % (x :: l).length)]'(Nat.mod_lt _ (Nat.zero_lt_succ _)) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    l : List α
    h : (List.cons x l).Nodup
    n : Nat
    ⊢ Eq ((HPow.hPow (List.cons x l).formPerm n) x) (GetElem.getElem (List.cons x  …
  -/
  convert formPerm_pow_apply_getElem _ h n 0 (Nat.succ_pos _)
  /-
    case h.e'_3.h.e'_7.h.e'_5
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    l : List α
    h : (List.cons x l).Nodup
    n : Nat
    ⊢ Eq n (HAdd.hAdd 0 n)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem formPerm_ext_iff {x y x' y' : α} {l l' : List α} (hd : Nodup (x :: y :: l))
    (hd' : Nodup (x' :: y' :: l')) :
    formPerm (x :: y :: l) = formPerm (x' :: y' :: l') ↔ (x :: y :: l) ~r (x' :: y' :: l') := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y x' y' : α
    l l' : List α
    hd : (List.cons x (List.cons y l)).Nodup
    hd' : (List.cons x' (List.cons y' l')).Nodup
    ⊢ Iff (Eq (List.cons x (List.cons y l)).formPerm (List.cons x' (List.cons y' l …
  -/
  refine ⟨fun h => ?_, fun hr => formPerm_eq_of_isRotated hd hr⟩
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y x' y' : α
    l l' : List α
    hd : (List.cons x (List.cons y l)).Nodup
    hd' : (List.cons x' (List.cons y' l')).Nodup
    h : Eq (List.cons x (List.cons y l)).formPerm (List.cons x' (List.cons y' l')) …
    ⊢ (List.cons x (List.cons y l)).IsRotated (List.cons x' (List.cons y' l'))
  -/
  rw [Equiv.Perm.ext_iff] at h
  have hx : x' ∈ x :: y :: l := by
    have : x' ∈ { z | formPerm (x :: y :: l) z ≠ z } := by
      rw [Set.mem_setOf_eq, h x', formPerm_apply_head _ _ _ hd']
      simp only [mem_cons, nodup_cons] at hd'
      push_neg at hd'
      exact hd'.left.left.symm
    simpa using support_formPerm_le' _ this
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y x' y' : α
    l l' : List α
    hd : (List.cons x (List.cons y l)).Nodup
    hd' : (List.cons x' (List.cons y' l')).Nodup
    h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
    hx : Membership.mem (List.cons x (List.cons y l)) x'
    ⊢ (List.cons x (List.cons y l)).IsRotated (List.cons x' (List.cons y' l'))
  -/
  obtain ⟨⟨n, hn⟩, hx'⟩ := get_of_mem hx
  have hl : (x :: y :: l).length = (x' :: y' :: l').length := by
    rw [← dedup_eq_self.mpr hd, ← dedup_eq_self.mpr hd', ← card_toFinset, ← card_toFinset]
    refine congr_arg Finset.card ?_
    rw [← Finset.coe_inj, ← support_formPerm_of_nodup' _ hd (by simp), ←
      support_formPerm_of_nodup' _ hd' (by simp)]
    simp only [h]
  /-
    case intro.mk
    α : Type u_1
    inst✝ : DecidableEq α
    x y x' y' : α
    l l' : List α
    hd : (List.cons x (List.cons y l)).Nodup
    hd' : (List.cons x' (List.cons y' l')).Nodup
    h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
    hx : Membership.mem (List.cons x (List.cons y l)) x'
    n : Nat
    hn : LT.lt n (List.cons x (List.cons y l)).length
    hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
    hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
    ⊢ (List.cons x (List.cons y l)).IsRotated (List.cons x' (List.cons y' l'))
  -/
  use n
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    x y x' y' : α
    l l' : List α
    hd : (List.cons x (List.cons y l)).Nodup
    hd' : (List.cons x' (List.cons y' l')).Nodup
    h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
    hx : Membership.mem (List.cons x (List.cons y l)) x'
    n : Nat
    hn : LT.lt n (List.cons x (List.cons y l)).length
    hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
    hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
    ⊢ Eq ((List.cons x (List.cons y l)).rotate n) (List.cons x' (List.cons y' l'))
  -/
  apply List.ext_getElem
    /-
      case h.hl
      α : Type u_1
      inst✝ : DecidableEq α
      x y x' y' : α
      l l' : List α
      hd : (List.cons x (List.cons y l)).Nodup
      hd' : (List.cons x' (List.cons y' l')).Nodup
      h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
      hx : Membership.mem (List.cons x (List.cons y l)) x'
      n : Nat
      hn : LT.lt n (List.cons x (List.cons y l)).length
      hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
      hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
      ⊢ Eq ((List.cons x (List.cons y l)).rotate n).length (List.cons x' (List.cons  …
    -/
  · rw [length_rotate, hl]
    /-
      🎉 no goals
    -/
    /-
      case h.h
      α : Type u_1
      inst✝ : DecidableEq α
      x y x' y' : α
      l l' : List α
      hd : (List.cons x (List.cons y l)).Nodup
      hd' : (List.cons x' (List.cons y' l')).Nodup
      h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
      hx : Membership.mem (List.cons x (List.cons y l)) x'
      n : Nat
      hn : LT.lt n (List.cons x (List.cons y l)).length
      hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
      hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
      ⊢ ∀ (n_1 : Nat) (h₁ : LT.lt n_1 ((List.cons x (List.cons y l)).rotate n).lengt …
    -/
  · intro k hk hk'
    /-
      case h.h
      α : Type u_1
      inst✝ : DecidableEq α
      x y x' y' : α
      l l' : List α
      hd : (List.cons x (List.cons y l)).Nodup
      hd' : (List.cons x' (List.cons y' l')).Nodup
      h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
      hx : Membership.mem (List.cons x (List.cons y l)) x'
      n : Nat
      hn : LT.lt n (List.cons x (List.cons y l)).length
      hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
      hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
      k : Nat
      hk : LT.lt k ((List.cons x (List.cons y l)).rotate n).length
      hk' : LT.lt k (List.cons x' (List.cons y' l')).length
      ⊢ Eq (GetElem.getElem ((List.cons x (List.cons y l)).rotate n) k hk) (GetElem. …
    -/
    rw [getElem_rotate]
    /-
      case h.h
      α : Type u_1
      inst✝ : DecidableEq α
      x y x' y' : α
      l l' : List α
      hd : (List.cons x (List.cons y l)).Nodup
      hd' : (List.cons x' (List.cons y' l')).Nodup
      h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
      hx : Membership.mem (List.cons x (List.cons y l)) x'
      n : Nat
      hn : LT.lt n (List.cons x (List.cons y l)).length
      hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
      hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
      k : Nat
      hk : LT.lt k ((List.cons x (List.cons y l)).rotate n).length
      hk' : LT.lt k (List.cons x' (List.cons y' l')).length
      ⊢ Eq (GetElem.getElem (List.cons x (List.cons y l)) (HMod.hMod (HAdd.hAdd k n) …
    -/
    induction' k with k IH
      /-
        case h.h.zero
        α : Type u_1
        inst✝ : DecidableEq α
        x y x' y' : α
        l l' : List α
        hd : (List.cons x (List.cons y l)).Nodup
        hd' : (List.cons x' (List.cons y' l')).Nodup
        h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
        hx : Membership.mem (List.cons x (List.cons y l)) x'
        n : Nat
        hn : LT.lt n (List.cons x (List.cons y l)).length
        hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
        hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
        hk : LT.lt 0 ((List.cons x (List.cons y l)).rotate n).length
        hk' : LT.lt 0 (List.cons x' (List.cons y' l')).length
        ⊢ Eq (GetElem.getElem (List.cons x (List.cons y l)) (HMod.hMod (HAdd.hAdd 0 n) …
      -/
    · refine Eq.trans ?_ hx'
      /-
        case h.h.zero
        α : Type u_1
        inst✝ : DecidableEq α
        x y x' y' : α
        l l' : List α
        hd : (List.cons x (List.cons y l)).Nodup
        hd' : (List.cons x' (List.cons y' l')).Nodup
        h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
        hx : Membership.mem (List.cons x (List.cons y l)) x'
        n : Nat
        hn : LT.lt n (List.cons x (List.cons y l)).length
        hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
        hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
        hk : LT.lt 0 ((List.cons x (List.cons y l)).rotate n).length
        hk' : LT.lt 0 (List.cons x' (List.cons y' l')).length
        ⊢ Eq (GetElem.getElem (List.cons x (List.cons y l)) (HMod.hMod (HAdd.hAdd 0 n) …
      -/
      congr
      /-
        case h.h.zero.e_i
        α : Type u_1
        inst✝ : DecidableEq α
        x y x' y' : α
        l l' : List α
        hd : (List.cons x (List.cons y l)).Nodup
        hd' : (List.cons x' (List.cons y' l')).Nodup
        h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
        hx : Membership.mem (List.cons x (List.cons y l)) x'
        n : Nat
        hn : LT.lt n (List.cons x (List.cons y l)).length
        hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
        hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
        hk : LT.lt 0 ((List.cons x (List.cons y l)).rotate n).length
        hk' : LT.lt 0 (List.cons x' (List.cons y' l')).length
        ⊢ Eq (HMod.hMod (HAdd.hAdd 0 n) (List.cons x (List.cons y l)).length) n
      -/
      simpa using hn
      /-
        🎉 no goals
      -/
      /-
        case h.h.succ
        α : Type u_1
        inst✝ : DecidableEq α
        x y x' y' : α
        l l' : List α
        hd : (List.cons x (List.cons y l)).Nodup
        hd' : (List.cons x' (List.cons y' l')).Nodup
        h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
        hx : Membership.mem (List.cons x (List.cons y l)) x'
        n : Nat
        hn : LT.lt n (List.cons x (List.cons y l)).length
        hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
        hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
        k : Nat
        IH : ∀ (hk : LT.lt k ((List.cons x (List.cons y l)).rotate n).length) (hk' : L …
        hk : LT.lt (HAdd.hAdd k 1) ((List.cons x (List.cons y l)).rotate n).length
        hk' : LT.lt (HAdd.hAdd k 1) (List.cons x' (List.cons y' l')).length
        ⊢ Eq (GetElem.getElem (List.cons x (List.cons y l)) (HMod.hMod (HAdd.hAdd (HAd …
      -/
    · conv => congr <;> · arg 2; (rw [← Nat.mod_eq_of_lt hk'])
      rw [← formPerm_apply_getElem _ hd' k (k.lt_succ_self.trans hk'),
        ← IH (k.lt_succ_self.trans hk), ← h, formPerm_apply_getElem _ hd]
      /-
        case h.h.succ
        α : Type u_1
        inst✝ : DecidableEq α
        x y x' y' : α
        l l' : List α
        hd : (List.cons x (List.cons y l)).Nodup
        hd' : (List.cons x' (List.cons y' l')).Nodup
        h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
        hx : Membership.mem (List.cons x (List.cons y l)) x'
        n : Nat
        hn : LT.lt n (List.cons x (List.cons y l)).length
        hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
        hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
        k : Nat
        IH : ∀ (hk : LT.lt k ((List.cons x (List.cons y l)).rotate n).length) (hk' : L …
        hk : LT.lt (HAdd.hAdd k 1) ((List.cons x (List.cons y l)).rotate n).length
        hk' : LT.lt (HAdd.hAdd k 1) (List.cons x' (List.cons y' l')).length
        ⊢ Eq (GetElem.getElem (List.cons x (List.cons y l)) (HMod.hMod (HAdd.hAdd (HMo …
      -/
      congr 1
      /-
        case h.h.succ.e_i
        α : Type u_1
        inst✝ : DecidableEq α
        x y x' y' : α
        l l' : List α
        hd : (List.cons x (List.cons y l)).Nodup
        hd' : (List.cons x' (List.cons y' l')).Nodup
        h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
        hx : Membership.mem (List.cons x (List.cons y l)) x'
        n : Nat
        hn : LT.lt n (List.cons x (List.cons y l)).length
        hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
        hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
        k : Nat
        IH : ∀ (hk : LT.lt k ((List.cons x (List.cons y l)).rotate n).length) (hk' : L …
        hk : LT.lt (HAdd.hAdd k 1) ((List.cons x (List.cons y l)).rotate n).length
        hk' : LT.lt (HAdd.hAdd k 1) (List.cons x' (List.cons y' l')).length
        ⊢ Eq (HMod.hMod (HAdd.hAdd (HMod.hMod (HAdd.hAdd k 1) (List.cons x' (List.cons …
      -/
      rw [hl, Nat.mod_eq_of_lt hk', add_right_comm]
      /-
        case h.h.succ.e_i
        α : Type u_1
        inst✝ : DecidableEq α
        x y x' y' : α
        l l' : List α
        hd : (List.cons x (List.cons y l)).Nodup
        hd' : (List.cons x' (List.cons y' l')).Nodup
        h : ∀ (x_1 : α), Eq ((List.cons x (List.cons y l)).formPerm x_1) ((List.cons x …
        hx : Membership.mem (List.cons x (List.cons y l)) x'
        n : Nat
        hn : LT.lt n (List.cons x (List.cons y l)).length
        hx' : Eq ((List.cons x (List.cons y l)).get ⟨n, hn⟩) x'
        hl : Eq (List.cons x (List.cons y l)).length (List.cons x' (List.cons y' l')). …
        k : Nat
        IH : ∀ (hk : LT.lt k ((List.cons x (List.cons y l)).rotate n).length) (hk' : L …
        hk : LT.lt (HAdd.hAdd k 1) ((List.cons x (List.cons y l)).rotate n).length
        hk' : LT.lt (HAdd.hAdd k 1) (List.cons x' (List.cons y' l')).length
        ⊢ Eq (HMod.hMod (HAdd.hAdd (HAdd.hAdd k n) 1) (List.cons x' (List.cons y' l')) …
      -/
      apply Nat.add_mod
      /-
        🎉 no goals
      -/


theorem formPerm_apply_mem_eq_self_iff (hl : Nodup l) (x : α) (hx : x ∈ l) :
    formPerm l x = x ↔ length l ≤ 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Iff (Eq (l.formPerm x) x) (LE.le l.length 1)
  -/
  obtain ⟨k, hk, rfl⟩ := getElem_of_mem hx
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    k : Nat
    hk : LT.lt k l.length
    hx : Membership.mem l (GetElem.getElem l k hk)
    ⊢ Iff (Eq (l.formPerm (GetElem.getElem l k hk)) (GetElem.getElem l k hk)) (LE. …
  -/
  rw [formPerm_apply_getElem _ hl k hk, hl.getElem_inj_iff]
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    k : Nat
    hk : LT.lt k l.length
    hx : Membership.mem l (GetElem.getElem l k hk)
    ⊢ Iff (Eq (HMod.hMod (HAdd.hAdd k 1) l.length) k) (LE.le l.length 1)
  -/
  cases hn : l.length
    /-
      case intro.intro.zero
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.Nodup
      k : Nat
      hk : LT.lt k l.length
      hx : Membership.mem l (GetElem.getElem l k hk)
      hn : Eq l.length 0
      ⊢ Iff (Eq (HMod.hMod (HAdd.hAdd k 1) 0) k) (LE.le 0 1)
    -/
  · exact absurd k.zero_le (hk.trans_le hn.le).not_le
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.succ
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.Nodup
      k : Nat
      hk : LT.lt k l.length
      hx : Membership.mem l (GetElem.getElem l k hk)
      n✝ : Nat
      hn : Eq l.length (HAdd.hAdd n✝ 1)
      ⊢ Iff (Eq (HMod.hMod (HAdd.hAdd k 1) (HAdd.hAdd n✝ 1)) k) (LE.le (HAdd.hAdd n✝ …
    -/
  · rw [hn] at hk
    /-
      case intro.intro.succ
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.Nodup
      k : Nat
      hk✝ : LT.lt k l.length
      hx : Membership.mem l (GetElem.getElem l k hk✝)
      n✝ : Nat
      hk : LT.lt k (HAdd.hAdd n✝ 1)
      hn : Eq l.length (HAdd.hAdd n✝ 1)
      ⊢ Iff (Eq (HMod.hMod (HAdd.hAdd k 1) (HAdd.hAdd n✝ 1)) k) (LE.le (HAdd.hAdd n✝ …
    -/
    rcases (Nat.le_of_lt_succ hk).eq_or_lt with hk' | hk'
      /-
        case intro.intro.succ.inl
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        hl : l.Nodup
        k : Nat
        hk✝ : LT.lt k l.length
        hx : Membership.mem l (GetElem.getElem l k hk✝)
        n✝ : Nat
        hk : LT.lt k (HAdd.hAdd n✝ 1)
        hn : Eq l.length (HAdd.hAdd n✝ 1)
        hk' : Eq k n✝
        ⊢ Iff (Eq (HMod.hMod (HAdd.hAdd k 1) (HAdd.hAdd n✝ 1)) k) (LE.le (HAdd.hAdd n✝ …
      -/
    · simp [← hk', Nat.succ_le_succ_iff, eq_comm]
      /-
        🎉 no goals
      -/
    · simpa [Nat.mod_eq_of_lt (Nat.succ_lt_succ hk'), Nat.succ_lt_succ_iff] using
        (k.zero_le.trans_lt hk').ne.symm


theorem formPerm_apply_mem_ne_self_iff (hl : Nodup l) (x : α) (hx : x ∈ l) :
    formPerm l x ≠ x ↔ 2 ≤ l.length := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Iff (Ne (l.formPerm x) x) (LE.le 2 l.length)
  -/
  rw [Ne, formPerm_apply_mem_eq_self_iff _ hl x hx, not_le]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Iff (LT.lt 1 l.length) (LE.le 2 l.length)
  -/
  exact ⟨Nat.succ_le_of_lt, Nat.lt_of_succ_le⟩
  /-
    🎉 no goals
  -/


theorem mem_of_formPerm_ne_self (l : List α) (x : α) (h : formPerm l x ≠ x) : x ∈ l := by
  suffices x ∈ { y | formPerm l y ≠ y } by
    rw [← mem_toFinset]
    exact support_formPerm_le' _ this
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    h : Ne (l.formPerm x) x
    ⊢ Membership.mem (setOf fun y => Ne (l.formPerm y) y) x
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem formPerm_eq_self_of_not_mem (l : List α) (x : α) (h : x ∉ l) : formPerm l x = x :=
  by_contra fun H => h <| mem_of_formPerm_ne_self _ _ H


theorem formPerm_eq_one_iff (hl : Nodup l) : formPerm l = 1 ↔ l.length ≤ 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    ⊢ Iff (Eq l.formPerm 1) (LE.le l.length 1)
  -/
  cases' l with hd tl
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      hl : List.nil.Nodup
      ⊢ Iff (Eq List.nil.formPerm 1) (LE.le List.nil.length 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      hd : α
      tl : List α
      hl : (List.cons hd tl).Nodup
      ⊢ Iff (Eq (List.cons hd tl).formPerm 1) (LE.le (List.cons hd tl).length 1)
    -/
  · rw [← formPerm_apply_mem_eq_self_iff _ hl hd (mem_cons_self _ _)]
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      hd : α
      tl : List α
      hl : (List.cons hd tl).Nodup
      ⊢ Iff (Eq (List.cons hd tl).formPerm 1) (Eq ((List.cons hd tl).formPerm hd) hd)
    -/
    constructor
      /-
        case cons.mp
        α : Type u_1
        inst✝ : DecidableEq α
        hd : α
        tl : List α
        hl : (List.cons hd tl).Nodup
        ⊢ Eq (List.cons hd tl).formPerm 1 → Eq ((List.cons hd tl).formPerm hd) hd
      -/
    · simp +contextual
      /-
        🎉 no goals
      -/
      /-
        case cons.mpr
        α : Type u_1
        inst✝ : DecidableEq α
        hd : α
        tl : List α
        hl : (List.cons hd tl).Nodup
        ⊢ Eq ((List.cons hd tl).formPerm hd) hd → Eq (List.cons hd tl).formPerm 1
      -/
    · intro h
      simp only [(hd :: tl).formPerm_apply_mem_eq_self_iff hl hd (mem_cons_self hd tl),
        add_le_iff_nonpos_left, length, nonpos_iff_eq_zero, length_eq_zero] at h
      /-
        case cons.mpr
        α : Type u_1
        inst✝ : DecidableEq α
        hd : α
        tl : List α
        hl : (List.cons hd tl).Nodup
        h : Eq tl List.nil
        ⊢ Eq (List.cons hd tl).formPerm 1
      -/
      simp [h]
      /-
        🎉 no goals
      -/


theorem formPerm_eq_formPerm_iff {l l' : List α} (hl : l.Nodup) (hl' : l'.Nodup) :
    l.formPerm = l'.formPerm ↔ l ~r l' ∨ l.length ≤ 1 ∧ l'.length ≤ 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    hl : l.Nodup
    hl' : l'.Nodup
    ⊢ Iff (Eq l.formPerm l'.formPerm) (Or (l.IsRotated l') (And (LE.le l.length 1) …
  -/
  rcases l with (_ | ⟨x, _ | ⟨y, l⟩⟩)
  · suffices l'.length ≤ 1 ↔ l' = nil ∨ l'.length ≤ 1 by
      simpa [eq_comm, formPerm_eq_one_iff, hl, hl', length_eq_zero]
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      l' : List α
      hl' : l'.Nodup
      hl : List.nil.Nodup
      ⊢ Iff (LE.le l'.length 1) (Or (Eq l' List.nil) (LE.le l'.length 1))
    -/
    refine ⟨fun h => Or.inr h, ?_⟩
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      l' : List α
      hl' : l'.Nodup
      hl : List.nil.Nodup
      ⊢ Or (Eq l' List.nil) (LE.le l'.length 1) → LE.le l'.length 1
    -/
    rintro (rfl | h)
      /-
        case nil.inl
        α : Type u_1
        inst✝ : DecidableEq α
        hl hl' : List.nil.Nodup
        ⊢ LE.le List.nil.length 1
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case nil.inr
        α : Type u_1
        inst✝ : DecidableEq α
        l' : List α
        hl' : l'.Nodup
        hl : List.nil.Nodup
        h : LE.le l'.length 1
        ⊢ LE.le l'.length 1
      -/
    · exact h
      /-
        🎉 no goals
      -/
  · suffices l'.length ≤ 1 ↔ [x] ~r l' ∨ l'.length ≤ 1 by
      simpa [eq_comm, formPerm_eq_one_iff, hl, hl', length_eq_zero, le_rfl]
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      l' : List α
      hl' : l'.Nodup
      x : α
      hl : (List.cons x List.nil).Nodup
      ⊢ Iff (LE.le l'.length 1) (Or ((List.cons x List.nil).IsRotated l') (LE.le l'. …
    -/
    refine ⟨fun h => Or.inr h, ?_⟩
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      l' : List α
      hl' : l'.Nodup
      x : α
      hl : (List.cons x List.nil).Nodup
      ⊢ Or ((List.cons x List.nil).IsRotated l') (LE.le l'.length 1) → LE.le l'.leng …
    -/
    rintro (h | h)
      /-
        case cons.nil.inl
        α : Type u_1
        inst✝ : DecidableEq α
        l' : List α
        hl' : l'.Nodup
        x : α
        hl : (List.cons x List.nil).Nodup
        h : (List.cons x List.nil).IsRotated l'
        ⊢ LE.le l'.length 1
      -/
    · simp [← h.perm.length_eq]
      /-
        🎉 no goals
      -/
      /-
        case cons.nil.inr
        α : Type u_1
        inst✝ : DecidableEq α
        l' : List α
        hl' : l'.Nodup
        x : α
        hl : (List.cons x List.nil).Nodup
        h : LE.le l'.length 1
        ⊢ LE.le l'.length 1
      -/
    · exact h
      /-
        🎉 no goals
      -/
    /-
      case cons.cons
      α : Type u_1
      inst✝ : DecidableEq α
      l' : List α
      hl' : l'.Nodup
      x y : α
      l : List α
      hl : (List.cons x (List.cons y l)).Nodup
      ⊢ Iff (Eq (List.cons x (List.cons y l)).formPerm l'.formPerm) (Or ((List.cons  …
    -/
  · rcases l' with (_ | ⟨x', _ | ⟨y', l'⟩⟩)
      /-
        case cons.cons.nil
        α : Type u_1
        inst✝ : DecidableEq α
        x y : α
        l : List α
        hl : (List.cons x (List.cons y l)).Nodup
        hl' : List.nil.Nodup
        ⊢ Iff (Eq (List.cons x (List.cons y l)).formPerm List.nil.formPerm) (Or ((List …
      -/
    · simp [formPerm_eq_one_iff _ hl, -formPerm_cons_cons]
      /-
        🎉 no goals
      -/
      /-
        case cons.cons.cons.nil
        α : Type u_1
        inst✝ : DecidableEq α
        x y : α
        l : List α
        hl : (List.cons x (List.cons y l)).Nodup
        x' : α
        hl' : (List.cons x' List.nil).Nodup
        ⊢ Iff (Eq (List.cons x (List.cons y l)).formPerm (List.cons x' List.nil).formP …
      -/
    · simp [formPerm_eq_one_iff _ hl, -formPerm_cons_cons]
      /-
        🎉 no goals
      -/
      /-
        case cons.cons.cons.cons
        α : Type u_1
        inst✝ : DecidableEq α
        x y : α
        l : List α
        hl : (List.cons x (List.cons y l)).Nodup
        x' y' : α
        l' : List α
        hl' : (List.cons x' (List.cons y' l')).Nodup
        ⊢ Iff (Eq (List.cons x (List.cons y l)).formPerm (List.cons x' (List.cons y' l …
      -/
    · simp [-formPerm_cons_cons, formPerm_ext_iff hl hl', Nat.succ_le_succ_iff]
      /-
        🎉 no goals
      -/


theorem form_perm_zpow_apply_mem_imp_mem (l : List α) (x : α) (hx : x ∈ l) (n : ℤ) :
    (formPerm l ^ n) x ∈ l := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    hx : Membership.mem l x
    n : Int
    ⊢ Membership.mem l ((HPow.hPow l.formPerm n) x)
  -/
  by_cases h : (l.formPerm ^ n) x = x
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      x : α
      hx : Membership.mem l x
      n : Int
      h : Eq ((HPow.hPow l.formPerm n) x) x
      ⊢ Membership.mem l ((HPow.hPow l.formPerm n) x)
    -/
  · simpa [h] using hx
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      x : α
      hx : Membership.mem l x
      n : Int
      h : Not (Eq ((HPow.hPow l.formPerm n) x) x)
      ⊢ Membership.mem l ((HPow.hPow l.formPerm n) x)
    -/
  · have h : x ∈ { x | (l.formPerm ^ n) x ≠ x } := h
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      x : α
      hx : Membership.mem l x
      n : Int
      h✝ : Not (Eq ((HPow.hPow l.formPerm n) x) x)
      h : Membership.mem (setOf fun x => Ne ((HPow.hPow l.formPerm n) x) x) x
      ⊢ Membership.mem l ((HPow.hPow l.formPerm n) x)
    -/
    rw [← set_support_apply_mem] at h
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      x : α
      hx : Membership.mem l x
      n : Int
      h✝ : Not (Eq ((HPow.hPow l.formPerm n) x) x)
      h : Membership.mem (setOf fun x => Ne ((HPow.hPow l.formPerm n) x) x) ((HPow.h …
      ⊢ Membership.mem l ((HPow.hPow l.formPerm n) x)
    -/
    replace h := set_support_zpow_subset _ _ h
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      x : α
      hx : Membership.mem l x
      n : Int
      h✝ : Not (Eq ((HPow.hPow l.formPerm n) x) x)
      h : Membership.mem (setOf fun x => Ne (l.formPerm x) x) ((HPow.hPow l.formPerm …
      ⊢ Membership.mem l ((HPow.hPow l.formPerm n) x)
    -/
    simpa using support_formPerm_le' _ h
    /-
      🎉 no goals
    -/


theorem formPerm_pow_length_eq_one_of_nodup (hl : Nodup l) : formPerm l ^ length l = 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    ⊢ Eq (HPow.hPow l.formPerm l.length) 1
  -/
  ext x
  /-
    case H
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    x : α
    ⊢ Eq ((HPow.hPow l.formPerm l.length) x) (1 x)
  -/
  by_cases hx : x ∈ l
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.Nodup
      x : α
      hx : Membership.mem l x
      ⊢ Eq ((HPow.hPow l.formPerm l.length) x) (1 x)
    -/
  · obtain ⟨k, hk, rfl⟩ := getElem_of_mem hx
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.Nodup
      k : Nat
      hk : LT.lt k l.length
      hx : Membership.mem l (GetElem.getElem l k hk)
      ⊢ Eq ((HPow.hPow l.formPerm l.length) (GetElem.getElem l k hk)) (1 (GetElem.ge …
    -/
    simp [formPerm_pow_apply_getElem _ hl, Nat.mod_eq_of_lt hk]
    /-
      🎉 no goals
    -/
  · have : x ∉ { x | (l.formPerm ^ l.length) x ≠ x } := by
      intro H
      refine hx ?_
      replace H := set_support_zpow_subset l.formPerm l.length H
      simpa using support_formPerm_le' _ H
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.Nodup
      x : α
      hx : Not (Membership.mem l x)
      this : Not (Membership.mem (setOf fun x => Ne ((HPow.hPow l.formPerm l.length) …
      ⊢ Eq ((HPow.hPow l.formPerm l.length) x) (1 x)
    -/
    simpa using this
    /-
      🎉 no goals
    -/


