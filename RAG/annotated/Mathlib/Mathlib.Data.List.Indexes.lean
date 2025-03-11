theorem list_reverse_induction (p : List α → Prop) (base : p [])
    (ind : ∀ (l : List α) (e : α), p l → p (l ++ [e])) : (∀ (l : List α), p l) := by
  /-
    α : Type u
    p : List α → Prop
    base : p List.nil
    ind : ∀ (l : List α) (e : α), p l → p (HAppend.hAppend l (List.cons e List.nil))
    ⊢ ∀ (l : List α), p l
  -/
  let q := fun l ↦ p (reverse l)
  /-
    α : Type u
    p : List α → Prop
    base : p List.nil
    ind : ∀ (l : List α) (e : α), p l → p (HAppend.hAppend l (List.cons e List.nil))
    q : List α → Prop := fun l => p l.reverse
    ⊢ ∀ (l : List α), p l
  -/
  have pq : ∀ l, p (reverse l) → q l := by simp only [q, reverse_reverse]; intro; exact id
  /-
    α : Type u
    p : List α → Prop
    base : p List.nil
    ind : ∀ (l : List α) (e : α), p l → p (HAppend.hAppend l (List.cons e List.nil))
    q : List α → Prop := fun l => p l.reverse
    pq : ∀ (l : List α), p l.reverse → q l
    ⊢ ∀ (l : List α), p l
  -/
  have qp : ∀ l, q (reverse l) → p l := by simp only [q, reverse_reverse]; intro; exact id
  /-
    α : Type u
    p : List α → Prop
    base : p List.nil
    ind : ∀ (l : List α) (e : α), p l → p (HAppend.hAppend l (List.cons e List.nil))
    q : List α → Prop := fun l => p l.reverse
    pq : ∀ (l : List α), p l.reverse → q l
    qp : ∀ (l : List α), q l.reverse → p l
    ⊢ ∀ (l : List α), p l
  -/
  intro l
  /-
    α : Type u
    p : List α → Prop
    base : p List.nil
    ind : ∀ (l : List α) (e : α), p l → p (HAppend.hAppend l (List.cons e List.nil))
    q : List α → Prop := fun l => p l.reverse
    pq : ∀ (l : List α), p l.reverse → q l
    qp : ∀ (l : List α), q l.reverse → p l
    l : List α
    ⊢ p l
  -/
  apply qp
  /-
    case a
    α : Type u
    p : List α → Prop
    base : p List.nil
    ind : ∀ (l : List α) (e : α), p l → p (HAppend.hAppend l (List.cons e List.nil))
    q : List α → Prop := fun l => p l.reverse
    pq : ∀ (l : List α), p l.reverse → q l
    qp : ∀ (l : List α), q l.reverse → p l
    l : List α
    ⊢ q l.reverse
  -/
  generalize (reverse l) = l
  /-
    case a
    α : Type u
    p : List α → Prop
    base : p List.nil
    ind : ∀ (l : List α) (e : α), p l → p (HAppend.hAppend l (List.cons e List.nil))
    q : List α → Prop := fun l => p l.reverse
    pq : ∀ (l : List α), p l.reverse → q l
    qp : ∀ (l : List α), q l.reverse → p l
    l✝ l : List α
    ⊢ q l
  -/
  induction' l with head tail ih
    /-
      case a.nil
      α : Type u
      p : List α → Prop
      base : p List.nil
      ind : ∀ (l : List α) (e : α), p l → p (HAppend.hAppend l (List.cons e List.nil))
      q : List α → Prop := fun l => p l.reverse
      pq : ∀ (l : List α), p l.reverse → q l
      qp : ∀ (l : List α), q l.reverse → p l
      l : List α
      ⊢ q List.nil
    -/
  · apply pq; simp only [reverse_nil, base]
              /-
                🎉 no goals
              -/
    /-
      case a.cons
      α : Type u
      p : List α → Prop
      base : p List.nil
      ind : ∀ (l : List α) (e : α), p l → p (HAppend.hAppend l (List.cons e List.nil))
      q : List α → Prop := fun l => p l.reverse
      pq : ∀ (l : List α), p l.reverse → q l
      qp : ∀ (l : List α), q l.reverse → p l
      l : List α
      head : α
      tail : List α
      ih : q tail
      ⊢ q (List.cons head tail)
    -/
  · apply pq; simp only [reverse_cons]; apply ind; apply qp; rw [reverse_reverse]; exact ih
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[deprecated (since := "2024-10-15")] alias mapIdxGo_length := mapIdx_go_length


theorem mapIdx_append_one : ∀ {f : ℕ → α → β} {l : List α} {e : α},
    mapIdx f (l ++ [e]) = mapIdx f l ++ [f l.length e] :=
  mapIdx_concat


@[local simp]
theorem map_enumFrom_eq_zipWith : ∀ (l : List α) (n : ℕ) (f : ℕ → α → β),
    map (uncurry f) (enumFrom n l) = zipWith (fun i ↦ f (i + n)) (range (length l)) l := by
  /-
    α : Type u
    β : Type v
    ⊢ ∀ (l : List α) (n : Nat) (f : Nat → α → β), Eq (List.map (Function.uncurry f …
  -/
  intro l
  /-
    α : Type u
    β : Type v
    l : List α
    ⊢ ∀ (n : Nat) (f : Nat → α → β), Eq (List.map (Function.uncurry f) (List.enumF …
  -/
  generalize e : l.length = len
  /-
    α : Type u
    β : Type v
    l : List α
    len : Nat
    e : Eq l.length len
    ⊢ ∀ (n : Nat) (f : Nat → α → β), Eq (List.map (Function.uncurry f) (List.enumF …
  -/
  revert l
  /-
    α : Type u
    β : Type v
    len : Nat
    ⊢ ∀ (l : List α), Eq l.length len → ∀ (n : Nat) (f : Nat → α → β), Eq (List.ma …
  -/
  induction' len with len ih <;> intros l e n f
  · have : l = [] := by
      cases l
      · rfl
      · contradiction
    /-
      case zero
      α : Type u
      β : Type v
      l : List α
      e : Eq l.length 0
      n : Nat
      f : Nat → α → β
      this : Eq l List.nil
      ⊢ Eq (List.map (Function.uncurry f) (List.enumFrom n l)) (List.zipWith (fun i  …
    -/
    rw [this]; rfl
               /-
                 🎉 no goals
               -/
    /-
      case succ
      α : Type u
      β : Type v
      len : Nat
      ih : ∀ (l : List α), Eq l.length len → ∀ (n : Nat) (f : Nat → α → β), Eq (List …
      l : List α
      e : Eq l.length (HAdd.hAdd len 1)
      n : Nat
      f : Nat → α → β
      ⊢ Eq (List.map (Function.uncurry f) (List.enumFrom n l)) (List.zipWith (fun i  …
    -/
  · cases' l with head tail
      /-
        case succ.nil
        α : Type u
        β : Type v
        len : Nat
        ih : ∀ (l : List α), Eq l.length len → ∀ (n : Nat) (f : Nat → α → β), Eq (List …
        n : Nat
        f : Nat → α → β
        e : Eq List.nil.length (HAdd.hAdd len 1)
        ⊢ Eq (List.map (Function.uncurry f) (List.enumFrom n List.nil)) (List.zipWith  …
      -/
    · contradiction
      /-
        🎉 no goals
      -/
    · simp only [enumFrom_cons, map_cons, range_succ_eq_map, zipWith_cons_cons,
        Nat.zero_add, zipWith_map_left, true_and]
      /-
        case succ.cons
        α : Type u
        β : Type v
        len : Nat
        ih : ∀ (l : List α), Eq l.length len → ∀ (n : Nat) (f : Nat → α → β), Eq (List …
        n : Nat
        f : Nat → α → β
        head : α
        tail : List α
        e : Eq (List.cons head tail).length (HAdd.hAdd len 1)
        ⊢ Eq (List.cons (Function.uncurry f { fst := n, snd := head }) (List.map (Func …
      -/
      rw [ih]
      · suffices (fun i ↦ f (i + (n + 1))) = ((fun i ↦ f (i + n)) ∘ Nat.succ) by
          rw [this]
          rfl
        /-
          case succ.cons
          α : Type u
          β : Type v
          len : Nat
          ih : ∀ (l : List α), Eq l.length len → ∀ (n : Nat) (f : Nat → α → β), Eq (List …
          n : Nat
          f : Nat → α → β
          head : α
          tail : List α
          e : Eq (List.cons head tail).length (HAdd.hAdd len 1)
          ⊢ Eq (fun i => f (HAdd.hAdd i (HAdd.hAdd n 1))) (Function.comp (fun i => f (HA …
        -/
        funext n' a
        /-
          case succ.cons.h.h
          α : Type u
          β : Type v
          len : Nat
          ih : ∀ (l : List α), Eq l.length len → ∀ (n : Nat) (f : Nat → α → β), Eq (List …
          n : Nat
          f : Nat → α → β
          head : α
          tail : List α
          e : Eq (List.cons head tail).length (HAdd.hAdd len 1)
          n' : Nat
          a : α
          ⊢ Eq (f (HAdd.hAdd n' (HAdd.hAdd n 1)) a) (Function.comp (fun i => f (HAdd.hAd …
        -/
        simp only [comp, Nat.add_assoc, Nat.add_comm, Nat.add_succ]
        /-
          🎉 no goals
        -/
      /-
        case succ.cons.e
        α : Type u
        β : Type v
        len : Nat
        ih : ∀ (l : List α), Eq l.length len → ∀ (n : Nat) (f : Nat → α → β), Eq (List …
        n : Nat
        f : Nat → α → β
        head : α
        tail : List α
        e : Eq (List.cons head tail).length (HAdd.hAdd len 1)
        ⊢ Eq tail.length len
      -/
      simp only [length_cons, Nat.succ.injEq] at e; exact e
                                                    /-
                                                      🎉 no goals
                                                    -/


@[deprecated (since := "2024-10-15")] alias mapIdx_eq_nil := mapIdx_eq_nil_iff


theorem get_mapIdx (l : List α) (f : ℕ → α → β) (i : ℕ) (h : i < l.length)
    (h' : i < (l.mapIdx f).length := h.trans_le length_mapIdx.ge) :
    (l.mapIdx f).get ⟨i, h'⟩ = f i (l.get ⟨i, h⟩) := by
  /-
    α : Type u
    β : Type v
    l : List α
    f : Nat → α → β
    i : Nat
    h : LT.lt i l.length
    h' : optParam (LT.lt i (List.mapIdx f l).length) ⋯
    ⊢ Eq ((List.mapIdx f l).get ⟨i, h'⟩) (f i (l.get ⟨i, h⟩))
  -/
  simp [mapIdx_eq_enum_map, enum_eq_zip_range]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-19")] alias nthLe_mapIdx := get_mapIdx


theorem mapIdx_eq_ofFn (l : List α) (f : ℕ → α → β) :
    l.mapIdx f = ofFn fun i : Fin l.length ↦ f (i : ℕ) (l.get i) := by
  induction l generalizing f with
  | nil => simp
  | cons _ _ IH => simp [IH]


/-- Lean3 `map_with_index` helper function -/
@[deprecated "No deprecation message was provided." (since := "2024-08-15")]
protected def oldMapIdxCore (f : ℕ → α → β) : ℕ → List α → List β
  | _, []      => []
  | k, a :: as => f k a :: List.oldMapIdxCore f (k + 1) as


set_option linter.deprecated false in
/-- Given a function `f : ℕ → α → β` and `as : List α`, `as = [a₀, a₁, ...]`, returns the list
`[f 0 a₀, f 1 a₁, ...]`. -/
@[deprecated "No deprecation message was provided." (since := "2024-08-15")]
protected def oldMapIdx (f : ℕ → α → β) (as : List α) : List β :=
  List.oldMapIdxCore f 0 as


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-08-15")]
protected theorem oldMapIdxCore_eq (l : List α) (f : ℕ → α → β) (n : ℕ) :
    l.oldMapIdxCore f n = l.oldMapIdx fun i a ↦ f (i + n) a := by
  /-
    α : Type u
    β : Type v
    l : List α
    f : Nat → α → β
    n : Nat
    ⊢ Eq (List.oldMapIdxCore f n l) (List.oldMapIdx (fun i a => f (HAdd.hAdd i n)  …
  -/
  induction' l with hd tl hl generalizing f n
    /-
      case nil
      α : Type u
      β : Type v
      f : Nat → α → β
      n : Nat
      ⊢ Eq (List.oldMapIdxCore f n List.nil) (List.oldMapIdx (fun i a => f (HAdd.hAd …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type v
      hd : α
      tl : List α
      hl : ∀ (f : Nat → α → β) (n : Nat), Eq (List.oldMapIdxCore f n tl) (List.oldMa …
      f : Nat → α → β
      n : Nat
      ⊢ Eq (List.oldMapIdxCore f n (List.cons hd tl)) (List.oldMapIdx (fun i a => f  …
    -/
  · rw [List.oldMapIdx]
    /-
      case cons
      α : Type u
      β : Type v
      hd : α
      tl : List α
      hl : ∀ (f : Nat → α → β) (n : Nat), Eq (List.oldMapIdxCore f n tl) (List.oldMa …
      f : Nat → α → β
      n : Nat
      ⊢ Eq (List.oldMapIdxCore f n (List.cons hd tl)) (List.oldMapIdxCore (fun i a = …
    -/
    simp only [List.oldMapIdxCore, hl, Nat.add_left_comm, Nat.add_comm, Nat.add_zero]
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-08-15")]
protected theorem oldMapIdxCore_append : ∀ (f : ℕ → α → β) (n : ℕ) (l₁ l₂ : List α),
    List.oldMapIdxCore f n (l₁ ++ l₂) =
    List.oldMapIdxCore f n l₁ ++ List.oldMapIdxCore f (n + l₁.length) l₂ := by
  /-
    α : Type u
    β : Type v
    ⊢ ∀ (f : Nat → α → β) (n : Nat) (l₁ l₂ : List α), Eq (List.oldMapIdxCore f n ( …
  -/
  intros f n l₁ l₂
  /-
    α : Type u
    β : Type v
    f : Nat → α → β
    n : Nat
    l₁ l₂ : List α
    ⊢ Eq (List.oldMapIdxCore f n (HAppend.hAppend l₁ l₂)) (HAppend.hAppend (List.o …
  -/
  generalize e : (l₁ ++ l₂).length = len
  /-
    α : Type u
    β : Type v
    f : Nat → α → β
    n : Nat
    l₁ l₂ : List α
    len : Nat
    e : Eq (HAppend.hAppend l₁ l₂).length len
    ⊢ Eq (List.oldMapIdxCore f n (HAppend.hAppend l₁ l₂)) (HAppend.hAppend (List.o …
  -/
  revert n l₁ l₂
  /-
    α : Type u
    β : Type v
    f : Nat → α → β
    len : Nat
    ⊢ ∀ (n : Nat) (l₁ l₂ : List α), Eq (HAppend.hAppend l₁ l₂).length len → Eq (Li …
  -/
  induction' len with len ih <;> intros n l₁ l₂ h
  · have l₁_nil : l₁ = [] := by
      cases l₁
      · rfl
      · contradiction
    have l₂_nil : l₂ = [] := by
      cases l₂
      · rfl
      · rw [List.length_append] at h; contradiction
    /-
      case zero
      α : Type u
      β : Type v
      f : Nat → α → β
      n : Nat
      l₁ l₂ : List α
      h : Eq (HAppend.hAppend l₁ l₂).length 0
      l₁_nil : Eq l₁ List.nil
      l₂_nil : Eq l₂ List.nil
      ⊢ Eq (List.oldMapIdxCore f n (HAppend.hAppend l₁ l₂)) (HAppend.hAppend (List.o …
    -/
    simp only [l₁_nil, l₂_nil]; rfl
                                /-
                                  🎉 no goals
                                -/
    /-
      case succ
      α : Type u
      β : Type v
      f : Nat → α → β
      len : Nat
      ih : ∀ (n : Nat) (l₁ l₂ : List α), Eq (HAppend.hAppend l₁ l₂).length len → Eq  …
      n : Nat
      l₁ l₂ : List α
      h : Eq (HAppend.hAppend l₁ l₂).length (HAdd.hAdd len 1)
      ⊢ Eq (List.oldMapIdxCore f n (HAppend.hAppend l₁ l₂)) (HAppend.hAppend (List.o …
    -/
  · cases' l₁ with head tail
      /-
        case succ.nil
        α : Type u
        β : Type v
        f : Nat → α → β
        len : Nat
        ih : ∀ (n : Nat) (l₁ l₂ : List α), Eq (HAppend.hAppend l₁ l₂).length len → Eq  …
        n : Nat
        l₂ : List α
        h : Eq (HAppend.hAppend List.nil l₂).length (HAdd.hAdd len 1)
        ⊢ Eq (List.oldMapIdxCore f n (HAppend.hAppend List.nil l₂)) (HAppend.hAppend ( …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case succ.cons
        α : Type u
        β : Type v
        f : Nat → α → β
        len : Nat
        ih : ∀ (n : Nat) (l₁ l₂ : List α), Eq (HAppend.hAppend l₁ l₂).length len → Eq  …
        n : Nat
        l₂ : List α
        head : α
        tail : List α
        h : Eq (HAppend.hAppend (List.cons head tail) l₂).length (HAdd.hAdd len 1)
        ⊢ Eq (List.oldMapIdxCore f n (HAppend.hAppend (List.cons head tail) l₂)) (HApp …
      -/
    · simp only [List.oldMapIdxCore, List.append_eq, length_cons, cons_append,cons.injEq, true_and]
      suffices n + Nat.succ (length tail) = n + 1 + tail.length by
        rw [this]
        apply ih (n + 1) _ _ _
        simp only [cons_append, length_cons, length_append, Nat.succ.injEq] at h
        simp only [length_append, h]
      /-
        case succ.cons
        α : Type u
        β : Type v
        f : Nat → α → β
        len : Nat
        ih : ∀ (n : Nat) (l₁ l₂ : List α), Eq (HAppend.hAppend l₁ l₂).length len → Eq  …
        n : Nat
        l₂ : List α
        head : α
        tail : List α
        h : Eq (HAppend.hAppend (List.cons head tail) l₂).length (HAdd.hAdd len 1)
        ⊢ Eq (HAdd.hAdd n tail.length.succ) (HAdd.hAdd (HAdd.hAdd n 1) tail.length)
      -/
      rw [Nat.add_assoc]; simp only [Nat.add_comm]
                          /-
                            🎉 no goals
                          -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-08-15")]
protected theorem oldMapIdx_append : ∀ (f : ℕ → α → β) (l : List α) (e : α),
    List.oldMapIdx f (l ++ [e]) = List.oldMapIdx f l ++ [f l.length e] := by
  /-
    α : Type u
    β : Type v
    ⊢ ∀ (f : Nat → α → β) (l : List α) (e : α), Eq (List.oldMapIdx f (HAppend.hApp …
  -/
  intros f l e
  /-
    α : Type u
    β : Type v
    f : Nat → α → β
    l : List α
    e : α
    ⊢ Eq (List.oldMapIdx f (HAppend.hAppend l (List.cons e List.nil))) (HAppend.hA …
  -/
  unfold List.oldMapIdx
  /-
    α : Type u
    β : Type v
    f : Nat → α → β
    l : List α
    e : α
    ⊢ Eq (List.oldMapIdxCore f 0 (HAppend.hAppend l (List.cons e List.nil))) (HApp …
  -/
  rw [List.oldMapIdxCore_append f 0 l [e]]
  /-
    α : Type u
    β : Type v
    f : Nat → α → β
    l : List α
    e : α
    ⊢ Eq (HAppend.hAppend (List.oldMapIdxCore f 0 l) (List.oldMapIdxCore f (HAdd.h …
  -/
  simp only [Nat.zero_add]; rfl
                            /-
                              🎉 no goals
                            -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-08-15")]
protected theorem new_def_eq_old_def :
    ∀ (f : ℕ → α → β) (l : List α), l.mapIdx f = List.oldMapIdx f l := by
  /-
    α : Type u
    β : Type v
    ⊢ ∀ (f : Nat → α → β) (l : List α), Eq (List.mapIdx f l) (List.oldMapIdx f l)
  -/
  intro f
  /-
    α : Type u
    β : Type v
    f : Nat → α → β
    ⊢ ∀ (l : List α), Eq (List.mapIdx f l) (List.oldMapIdx f l)
  -/
  apply list_reverse_induction
    /-
      case base
      α : Type u
      β : Type v
      f : Nat → α → β
      ⊢ Eq (List.mapIdx f List.nil) (List.oldMapIdx f List.nil)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case ind
      α : Type u
      β : Type v
      f : Nat → α → β
      ⊢ ∀ (l : List α) (e : α), Eq (List.mapIdx f l) (List.oldMapIdx f l) → Eq (List …
    -/
  · intro l e h
    /-
      case ind
      α : Type u
      β : Type v
      f : Nat → α → β
      l : List α
      e : α
      h : Eq (List.mapIdx f l) (List.oldMapIdx f l)
      ⊢ Eq (List.mapIdx f (HAppend.hAppend l (List.cons e List.nil))) (List.oldMapId …
    -/
    rw [List.oldMapIdx_append, mapIdx_append_one, h]
    /-
      🎉 no goals
    -/


/-- Specification of `foldrIdx`. -/
def foldrIdxSpec (f : ℕ → α → β → β) (b : β) (as : List α) (start : ℕ) : β :=
  foldr (uncurry f) b <| enumFrom start as


theorem foldrIdxSpec_cons (f : ℕ → α → β → β) (b a as start) :
    foldrIdxSpec f b (a :: as) start = f start a (foldrIdxSpec f b as (start + 1)) :=
  rfl


theorem foldrIdx_eq_foldrIdxSpec (f : ℕ → α → β → β) (b as start) :
    foldrIdx f b as start = foldrIdxSpec f b as start := by
  /-
    α : Type u
    β : Type v
    f : Nat → α → β → β
    b : β
    as : List α
    start : Nat
    ⊢ Eq (List.foldrIdx f b as start) (List.foldrIdxSpec f b as start)
  -/
  induction as generalizing start
    /-
      case nil
      α : Type u
      β : Type v
      f : Nat → α → β → β
      b : β
      start : Nat
      ⊢ Eq (List.foldrIdx f b List.nil start) (List.foldrIdxSpec f b List.nil start)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type v
      f : Nat → α → β → β
      b : β
      head✝ : α
      tail✝ : List α
      tail_ih✝ : ∀ (start : Nat), Eq (List.foldrIdx f b tail✝ start) (List.foldrIdxS …
      start : Nat
      ⊢ Eq (List.foldrIdx f b (List.cons head✝ tail✝) start) (List.foldrIdxSpec f b  …
    -/
  · simp only [foldrIdx, foldrIdxSpec_cons, *]
    /-
      🎉 no goals
    -/


theorem foldrIdx_eq_foldr_enum (f : ℕ → α → β → β) (b : β) (as : List α) :
    foldrIdx f b as = foldr (uncurry f) b (enum as) := by
  /-
    α : Type u
    β : Type v
    f : Nat → α → β → β
    b : β
    as : List α
    ⊢ Eq (List.foldrIdx f b as) (List.foldr (Function.uncurry f) b as.enum)
  -/
  simp only [foldrIdx, foldrIdxSpec, foldrIdx_eq_foldrIdxSpec, enum]
  /-
    🎉 no goals
  -/


theorem indexesValues_eq_filter_enum (p : α → Prop) [DecidablePred p] (as : List α) :
    indexesValues p as = filter (p ∘ Prod.snd) (enum as) := by
  simp (config := { unfoldPartialApp := true }) [indexesValues, foldrIdx_eq_foldr_enum, uncurry,
    filter_eq_foldr, cond_eq_if]


theorem findIdxs_eq_map_indexesValues (p : α → Prop) [DecidablePred p] (as : List α) :
    findIdxs p as = map Prod.fst (indexesValues p as) := by
  simp (config := { unfoldPartialApp := true }) only [indexesValues_eq_filter_enum,
    map_filter_eq_foldr, findIdxs, uncurry, foldrIdx_eq_foldr_enum, decide_eq_true_eq, comp_apply,
    Bool.cond_decide]


/-- Specification of `foldlIdx`. -/
def foldlIdxSpec (f : ℕ → α → β → α) (a : α) (bs : List β) (start : ℕ) : α :=
  foldl (fun a p ↦ f p.fst a p.snd) a <| enumFrom start bs


theorem foldlIdxSpec_cons (f : ℕ → α → β → α) (a b bs start) :
    foldlIdxSpec f a (b :: bs) start = foldlIdxSpec f (f start a b) bs (start + 1) :=
  rfl


theorem foldlIdx_eq_foldlIdxSpec (f : ℕ → α → β → α) (a bs start) :
    foldlIdx f a bs start = foldlIdxSpec f a bs start := by
  /-
    α : Type u
    β : Type v
    f : Nat → α → β → α
    a : α
    bs : List β
    start : Nat
    ⊢ Eq (List.foldlIdx f a bs start) (List.foldlIdxSpec f a bs start)
  -/
  induction bs generalizing start a
    /-
      case nil
      α : Type u
      β : Type v
      f : Nat → α → β → α
      a : α
      start : Nat
      ⊢ Eq (List.foldlIdx f a List.nil start) (List.foldlIdxSpec f a List.nil start)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type v
      f : Nat → α → β → α
      head✝ : β
      tail✝ : List β
      tail_ih✝ : ∀ (a : α) (start : Nat), Eq (List.foldlIdx f a tail✝ start) (List.f …
      a : α
      start : Nat
      ⊢ Eq (List.foldlIdx f a (List.cons head✝ tail✝) start) (List.foldlIdxSpec f a  …
    -/
  · simp [foldlIdxSpec, *]
    /-
      🎉 no goals
    -/


theorem foldlIdx_eq_foldl_enum (f : ℕ → α → β → α) (a : α) (bs : List β) :
    foldlIdx f a bs = foldl (fun a p ↦ f p.fst a p.snd) a (enum bs) := by
  /-
    α : Type u
    β : Type v
    f : Nat → α → β → α
    a : α
    bs : List β
    ⊢ Eq (List.foldlIdx f a bs) (List.foldl (fun a p => f p.1 a p.2) a bs.enum)
  -/
  simp only [foldlIdx, foldlIdxSpec, foldlIdx_eq_foldlIdxSpec, enum]
  /-
    🎉 no goals
  -/


theorem foldrIdxM_eq_foldrM_enum {β} (f : ℕ → α → β → m β) (b : β) (as : List α) [LawfulMonad m] :
    foldrIdxM f b as = foldrM (uncurry f) b (enum as) := by
  simp (config := { unfoldPartialApp := true }) only [foldrIdxM, foldrM_eq_foldr,
    foldrIdx_eq_foldr_enum, uncurry]


theorem foldlIdxM_eq_foldlM_enum [LawfulMonad m] {β} (f : ℕ → β → α → m β) (b : β) (as : List α) :
    foldlIdxM f b as = List.foldlM (fun b p ↦ f p.fst b p.snd) b (enum as) := by
  /-
    α : Type u
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    β : Type u
    f : Nat → β → α → m β
    b : β
    as : List α
    ⊢ Eq (List.foldlIdxM f b as) (List.foldlM (fun b p => f p.1 b p.2) b as.enum)
  -/
  rw [foldlIdxM, foldlM_eq_foldl, foldlIdx_eq_foldl_enum]
  /-
    🎉 no goals
  -/


/-- Specification of `mapIdxMAux`. -/
def mapIdxMAuxSpec {β} (f : ℕ → α → m β) (start : ℕ) (as : List α) : m (List β) :=
  List.traverse (uncurry f) <| enumFrom start as

-- Note: `traverse` the class method would require a less universe-polymorphic
-- `m : Type u → Type u`.

theorem mapIdxMAuxSpec_cons {β} (f : ℕ → α → m β) (start : ℕ) (a : α) (as : List α) :
    mapIdxMAuxSpec f start (a :: as) = cons <$> f start a <*> mapIdxMAuxSpec f (start + 1) as :=
  rfl


theorem mapIdxMGo_eq_mapIdxMAuxSpec
    [LawfulMonad m] {β} (f : ℕ → α → m β) (arr : Array β) (as : List α) :
    mapIdxM.go f as arr = (arr.toList ++ ·) <$> mapIdxMAuxSpec f arr.size as := by
  /-
    α : Type u
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    β : Type u
    f : Nat → α → m β
    arr : Array β
    as : List α
    ⊢ Eq (List.mapIdxM.go f as arr) (Functor.map (fun x => HAppend.hAppend arr.toL …
  -/
  generalize e : as.length = len
  /-
    α : Type u
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    β : Type u
    f : Nat → α → m β
    arr : Array β
    as : List α
    len : Nat
    e : Eq as.length len
    ⊢ Eq (List.mapIdxM.go f as arr) (Functor.map (fun x => HAppend.hAppend arr.toL …
  -/
  revert as arr
  /-
    α : Type u
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    β : Type u
    f : Nat → α → m β
    len : Nat
    ⊢ ∀ (arr : Array β) (as : List α), Eq as.length len → Eq (List.mapIdxM.go f as …
  -/
  induction' len with len ih <;> intro arr as h
  · have : as = [] := by
      cases as
      · rfl
      · contradiction
    /-
      case zero
      α : Type u
      m : Type u → Type v
      inst✝¹ : Monad m
      inst✝ : LawfulMonad m
      β : Type u
      f : Nat → α → m β
      arr : Array β
      as : List α
      h : Eq as.length 0
      this : Eq as List.nil
      ⊢ Eq (List.mapIdxM.go f as arr) (Functor.map (fun x => HAppend.hAppend arr.toL …
    -/
    simp only [this, mapIdxM.go, mapIdxMAuxSpec, enumFrom_nil, List.traverse, map_pure, append_nil]
    /-
      🎉 no goals
    -/
  · match as with
    | nil => contradiction
    | cons head tail =>
      simp only [length_cons, Nat.succ.injEq] at h
      simp only [mapIdxM.go, mapIdxMAuxSpec_cons, map_eq_pure_bind, seq_eq_bind_map,
        LawfulMonad.bind_assoc, pure_bind]
      congr
      conv => { lhs; intro x; rw [ih _ _ h]; }
      funext x
      simp only [Array.push_toList, append_assoc, singleton_append, Array.size_push,
        map_eq_pure_bind]


theorem mapIdxM_eq_mmap_enum [LawfulMonad m] {β} (f : ℕ → α → m β) (as : List α) :
    as.mapIdxM f = List.traverse (uncurry f) (enum as) := by
  simp only [mapIdxM, mapIdxMGo_eq_mapIdxMAuxSpec, Array.toList_toArray,
    nil_append, mapIdxMAuxSpec, Array.size_toArray, length_nil, id_map', enum]


theorem mapIdxMAux'_eq_mapIdxMGo {α} (f : ℕ → α → m PUnit) (as : List α) (arr : Array PUnit) :
    mapIdxMAux' f arr.size as = mapIdxM.go f as arr *> pure PUnit.unit := by
  /-
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    α : Type u_1
    f : Nat → α → m PUnit.{u + 1}
    as : List α
    arr : Array PUnit.{u + 1}
    ⊢ Eq (List.mapIdxMAux' f arr.size as) (SeqRight.seqRight (List.mapIdxM.go f as …
  -/
  revert arr
  /-
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    α : Type u_1
    f : Nat → α → m PUnit.{u + 1}
    as : List α
    ⊢ ∀ (arr : Array PUnit.{u + 1}), Eq (List.mapIdxMAux' f arr.size as) (SeqRight …
  -/
  induction' as with head tail ih <;> intro arr
    /-
      case nil
      m : Type u → Type v
      inst✝¹ : Monad m
      inst✝ : LawfulMonad m
      α : Type u_1
      f : Nat → α → m PUnit.{u + 1}
      arr : Array PUnit.{u + 1}
      ⊢ Eq (List.mapIdxMAux' f arr.size List.nil) (SeqRight.seqRight (List.mapIdxM.g …
    -/
  · simp only [mapIdxMAux', mapIdxM.go, seqRight_eq, map_pure, seq_pure]
    /-
      🎉 no goals
    -/
  · simp only [mapIdxMAux', seqRight_eq, map_eq_pure_bind, seq_eq_bind, bind_pure_unit,
      LawfulMonad.bind_assoc, pure_bind, mapIdxM.go, seq_pure]
    /-
      case cons
      m : Type u → Type v
      inst✝¹ : Monad m
      inst✝ : LawfulMonad m
      α : Type u_1
      f : Nat → α → m PUnit.{u + 1}
      head : α
      tail : List α
      ih : ∀ (arr : Array PUnit.{u + 1}), Eq (List.mapIdxMAux' f arr.size tail) (Seq …
      arr : Array PUnit.{u + 1}
      ⊢ Eq (Bind.bind (f arr.size head) fun x => List.mapIdxMAux' f (HAdd.hAdd arr.s …
    -/
    generalize (f (Array.size arr) head) = head
    /-
      case cons
      m : Type u → Type v
      inst✝¹ : Monad m
      inst✝ : LawfulMonad m
      α : Type u_1
      f : Nat → α → m PUnit.{u + 1}
      head✝ : α
      tail : List α
      ih : ∀ (arr : Array PUnit.{u + 1}), Eq (List.mapIdxMAux' f arr.size tail) (Seq …
      arr : Array PUnit.{u + 1}
      head : m PUnit.{u + 1}
      ⊢ Eq (Bind.bind head fun x => List.mapIdxMAux' f (HAdd.hAdd arr.size 1) tail)  …
    -/
    have : (arr.push ⟨⟩).size = arr.size + 1 := Array.size_push arr ⟨⟩
    /-
      case cons
      m : Type u → Type v
      inst✝¹ : Monad m
      inst✝ : LawfulMonad m
      α : Type u_1
      f : Nat → α → m PUnit.{u + 1}
      head✝ : α
      tail : List α
      ih : ∀ (arr : Array PUnit.{u + 1}), Eq (List.mapIdxMAux' f arr.size tail) (Seq …
      arr : Array PUnit.{u + 1}
      head : m PUnit.{u + 1}
      this : Eq (arr.push PUnit.unit).size (HAdd.hAdd arr.size 1)
      ⊢ Eq (Bind.bind head fun x => List.mapIdxMAux' f (HAdd.hAdd arr.size 1) tail)  …
    -/
    rw [← this, ih]
    /-
      case cons
      m : Type u → Type v
      inst✝¹ : Monad m
      inst✝ : LawfulMonad m
      α : Type u_1
      f : Nat → α → m PUnit.{u + 1}
      head✝ : α
      tail : List α
      ih : ∀ (arr : Array PUnit.{u + 1}), Eq (List.mapIdxMAux' f arr.size tail) (Seq …
      arr : Array PUnit.{u + 1}
      head : m PUnit.{u + 1}
      this : Eq (arr.push PUnit.unit).size (HAdd.hAdd arr.size 1)
      ⊢ Eq (Bind.bind head fun x => SeqRight.seqRight (List.mapIdxM.go f tail (arr.p …
    -/
    simp only [seqRight_eq, map_eq_pure_bind, seq_pure, LawfulMonad.bind_assoc, pure_bind]
    /-
      🎉 no goals
    -/


theorem mapIdxM'_eq_mapIdxM {α} (f : ℕ → α → m PUnit) (as : List α) :
    mapIdxM' f as = mapIdxM as f *> pure PUnit.unit :=
  mapIdxMAux'_eq_mapIdxMGo f as #[]


