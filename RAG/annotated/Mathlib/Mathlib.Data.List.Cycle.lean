/-- Return the `z` such that `x :: z :: _` appears in `xs`, or `default` if there is no such `z`. -/
def nextOr : ∀ (_ : List α) (_ _ : α), α
  | [], _, default => default
  | [_], _, default => default
  -- Handles the not-found and the wraparound case
  | y :: z :: xs, x, default => if x = y then z else nextOr (z :: xs) x default


@[simp]
theorem nextOr_nil (x d : α) : nextOr [] x d = d :=
  rfl


@[simp]
theorem nextOr_singleton (x y d : α) : nextOr [y] x d = d :=
  rfl


@[simp]
theorem nextOr_self_cons_cons (xs : List α) (x y d : α) : nextOr (x :: y :: xs) x d = y :=
  if_pos rfl


theorem nextOr_cons_of_ne (xs : List α) (y x d : α) (h : x ≠ y) :
    nextOr (y :: xs) x d = nextOr xs x d := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    y x d : α
    h : Ne x y
    ⊢ Eq ((List.cons y xs).nextOr x d) (xs.nextOr x d)
  -/
  cases' xs with z zs
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      y x d : α
      h : Ne x y
      ⊢ Eq ((List.cons y List.nil).nextOr x d) (List.nil.nextOr x d)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      y x d : α
      h : Ne x y
      z : α
      zs : List α
      ⊢ Eq ((List.cons y (List.cons z zs)).nextOr x d) ((List.cons z zs).nextOr x d)
    -/
  · exact if_neg h
    /-
      🎉 no goals
    -/


/-- `nextOr` does not depend on the default value, if the next value appears. -/
theorem nextOr_eq_nextOr_of_mem_of_ne (xs : List α) (x d d' : α) (x_mem : x ∈ xs)
    (x_ne : x ≠ xs.getLast (ne_nil_of_mem x_mem)) : nextOr xs x d = nextOr xs x d' := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    x d d' : α
    x_mem : Membership.mem xs x
    x_ne : Ne x (xs.getLast ⋯)
    ⊢ Eq (xs.nextOr x d) (xs.nextOr x d')
  -/
  induction' xs with y ys IH
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      x d d' : α
      x_mem : Membership.mem List.nil x
      x_ne : Ne x (List.nil.getLast ⋯)
      ⊢ Eq (List.nil.nextOr x d) (List.nil.nextOr x d')
    -/
  · cases x_mem
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    x d d' y : α
    ys : List α
    IH : ∀ (x_mem : Membership.mem ys x), Ne x (ys.getLast ⋯) → Eq (ys.nextOr x d) …
    x_mem : Membership.mem (List.cons y ys) x
    x_ne : Ne x ((List.cons y ys).getLast ⋯)
    ⊢ Eq ((List.cons y ys).nextOr x d) ((List.cons y ys).nextOr x d')
  -/
  cases' ys with z zs
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      x d d' y : α
      IH : ∀ (x_mem : Membership.mem List.nil x), Ne x (List.nil.getLast ⋯) → Eq (Li …
      x_mem : Membership.mem (List.cons y List.nil) x
      x_ne : Ne x ((List.cons y List.nil).getLast ⋯)
      ⊢ Eq ((List.cons y List.nil).nextOr x d) ((List.cons y List.nil).nextOr x d')
    -/
  · simp at x_mem x_ne
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      x d d' y : α
      IH : ∀ (x_mem : Membership.mem List.nil x), Ne x (List.nil.getLast ⋯) → Eq (Li …
      x_ne : Not (Eq x y)
      x_mem : Eq x y
      ⊢ Eq ((List.cons y List.nil).nextOr x d) ((List.cons y List.nil).nextOr x d')
    -/
    contradiction
    /-
      🎉 no goals
    -/
  /-
    case cons.cons
    α : Type u_1
    inst✝ : DecidableEq α
    x d d' y z : α
    zs : List α
    IH : ∀ (x_mem : Membership.mem (List.cons z zs) x), Ne x ((List.cons z zs).get …
    x_mem : Membership.mem (List.cons y (List.cons z zs)) x
    x_ne : Ne x ((List.cons y (List.cons z zs)).getLast ⋯)
    ⊢ Eq ((List.cons y (List.cons z zs)).nextOr x d) ((List.cons y (List.cons z zs …
  -/
  by_cases h : x = y
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      x d d' y z : α
      zs : List α
      IH : ∀ (x_mem : Membership.mem (List.cons z zs) x), Ne x ((List.cons z zs).get …
      x_mem : Membership.mem (List.cons y (List.cons z zs)) x
      x_ne : Ne x ((List.cons y (List.cons z zs)).getLast ⋯)
      h : Eq x y
      ⊢ Eq ((List.cons y (List.cons z zs)).nextOr x d) ((List.cons y (List.cons z zs …
    -/
  · rw [h, nextOr_self_cons_cons, nextOr_self_cons_cons]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      x d d' y z : α
      zs : List α
      IH : ∀ (x_mem : Membership.mem (List.cons z zs) x), Ne x ((List.cons z zs).get …
      x_mem : Membership.mem (List.cons y (List.cons z zs)) x
      x_ne : Ne x ((List.cons y (List.cons z zs)).getLast ⋯)
      h : Not (Eq x y)
      ⊢ Eq ((List.cons y (List.cons z zs)).nextOr x d) ((List.cons y (List.cons z zs …
    -/
  · rw [nextOr, nextOr, IH]
      /-
        case neg.x_mem
        α : Type u_1
        inst✝ : DecidableEq α
        x d d' y z : α
        zs : List α
        IH : ∀ (x_mem : Membership.mem (List.cons z zs) x), Ne x ((List.cons z zs).get …
        x_mem : Membership.mem (List.cons y (List.cons z zs)) x
        x_ne : Ne x ((List.cons y (List.cons z zs)).getLast ⋯)
        h : Not (Eq x y)
        ⊢ Membership.mem (List.cons z zs) x
      -/
    · simpa [h] using x_mem
      /-
        🎉 no goals
      -/
      /-
        case neg.x_ne
        α : Type u_1
        inst✝ : DecidableEq α
        x d d' y z : α
        zs : List α
        IH : ∀ (x_mem : Membership.mem (List.cons z zs) x), Ne x ((List.cons z zs).get …
        x_mem : Membership.mem (List.cons y (List.cons z zs)) x
        x_ne : Ne x ((List.cons y (List.cons z zs)).getLast ⋯)
        h : Not (Eq x y)
        ⊢ Ne x ((List.cons z zs).getLast ⋯)
      -/
    · simpa using x_ne
      /-
        🎉 no goals
      -/


theorem mem_of_nextOr_ne {xs : List α} {x d : α} (h : nextOr xs x d ≠ d) : x ∈ xs := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    x d : α
    h : Ne (xs.nextOr x d) d
    ⊢ Membership.mem xs x
  -/
  induction' xs with y ys IH
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      x d : α
      h : Ne (List.nil.nextOr x d) d
      ⊢ Membership.mem List.nil x
    -/
  · simp at h
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    x d y : α
    ys : List α
    IH : Ne (ys.nextOr x d) d → Membership.mem ys x
    h : Ne ((List.cons y ys).nextOr x d) d
    ⊢ Membership.mem (List.cons y ys) x
  -/
  cases' ys with z zs
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      x d y : α
      IH : Ne (List.nil.nextOr x d) d → Membership.mem List.nil x
      h : Ne ((List.cons y List.nil).nextOr x d) d
      ⊢ Membership.mem (List.cons y List.nil) x
    -/
  · simp at h
    /-
      🎉 no goals
    -/
    /-
      case cons.cons
      α : Type u_1
      inst✝ : DecidableEq α
      x d y z : α
      zs : List α
      IH : Ne ((List.cons z zs).nextOr x d) d → Membership.mem (List.cons z zs) x
      h : Ne ((List.cons y (List.cons z zs)).nextOr x d) d
      ⊢ Membership.mem (List.cons y (List.cons z zs)) x
    -/
  · by_cases hx : x = y
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        x d y z : α
        zs : List α
        IH : Ne ((List.cons z zs).nextOr x d) d → Membership.mem (List.cons z zs) x
        h : Ne ((List.cons y (List.cons z zs)).nextOr x d) d
        hx : Eq x y
        ⊢ Membership.mem (List.cons y (List.cons z zs)) x
      -/
    · simp [hx]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        x d y z : α
        zs : List α
        IH : Ne ((List.cons z zs).nextOr x d) d → Membership.mem (List.cons z zs) x
        h : Ne ((List.cons y (List.cons z zs)).nextOr x d) d
        hx : Not (Eq x y)
        ⊢ Membership.mem (List.cons y (List.cons z zs)) x
      -/
    · rw [nextOr_cons_of_ne _ _ _ _ hx] at h
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        x d y z : α
        zs : List α
        IH : Ne ((List.cons z zs).nextOr x d) d → Membership.mem (List.cons z zs) x
        h : Ne ((List.cons z zs).nextOr x d) d
        hx : Not (Eq x y)
        ⊢ Membership.mem (List.cons y (List.cons z zs)) x
      -/
      simpa [hx] using IH h
      /-
        🎉 no goals
      -/


theorem nextOr_concat {xs : List α} {x : α} (d : α) (h : x ∉ xs) : nextOr (xs ++ [x]) x d = d := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    x d : α
    h : Not (Membership.mem xs x)
    ⊢ Eq ((HAppend.hAppend xs (List.cons x List.nil)).nextOr x d) d
  -/
  induction' xs with z zs IH
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      x d : α
      h : Not (Membership.mem List.nil x)
      ⊢ Eq ((HAppend.hAppend List.nil (List.cons x List.nil)).nextOr x d) d
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      x d z : α
      zs : List α
      IH : Not (Membership.mem zs x) → Eq ((HAppend.hAppend zs (List.cons x List.nil …
      h : Not (Membership.mem (List.cons z zs) x)
      ⊢ Eq ((HAppend.hAppend (List.cons z zs) (List.cons x List.nil)).nextOr x d) d
    -/
  · obtain ⟨hz, hzs⟩ := not_or.mp (mt mem_cons.2 h)
    /-
      case cons.intro
      α : Type u_1
      inst✝ : DecidableEq α
      x d z : α
      zs : List α
      IH : Not (Membership.mem zs x) → Eq ((HAppend.hAppend zs (List.cons x List.nil …
      h : Not (Membership.mem (List.cons z zs) x)
      hz : Not (Eq x z)
      hzs : Not (Membership.mem zs x)
      ⊢ Eq ((HAppend.hAppend (List.cons z zs) (List.cons x List.nil)).nextOr x d) d
    -/
    rw [cons_append, nextOr_cons_of_ne _ _ _ _ hz, IH hzs]
    /-
      🎉 no goals
    -/


theorem nextOr_mem {xs : List α} {x d : α} (hd : d ∈ xs) : nextOr xs x d ∈ xs := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    x d : α
    hd : Membership.mem xs d
    ⊢ Membership.mem xs (xs.nextOr x d)
  -/
  revert hd
  suffices ∀ xs' : List α, (∀ x ∈ xs, x ∈ xs') → d ∈ xs' → nextOr xs x d ∈ xs' by
    exact this xs fun _ => id
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    x d : α
    ⊢ ∀ (xs' : List α), (∀ (x : α), Membership.mem xs x → Membership.mem xs' x) →  …
  -/
  intro xs' hxs' hd
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs : List α
    x d : α
    xs' : List α
    hxs' : ∀ (x : α), Membership.mem xs x → Membership.mem xs' x
    hd : Membership.mem xs' d
    ⊢ Membership.mem xs' (xs.nextOr x d)
  -/
  induction' xs with y ys ih
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      x d : α
      xs' : List α
      hd : Membership.mem xs' d
      hxs' : ∀ (x : α), Membership.mem List.nil x → Membership.mem xs' x
      ⊢ Membership.mem xs' (List.nil.nextOr x d)
    -/
  · exact hd
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    x d : α
    xs' : List α
    hd : Membership.mem xs' d
    y : α
    ys : List α
    ih : (∀ (x : α), Membership.mem ys x → Membership.mem xs' x) → Membership.mem  …
    hxs' : ∀ (x : α), Membership.mem (List.cons y ys) x → Membership.mem xs' x
    ⊢ Membership.mem xs' ((List.cons y ys).nextOr x d)
  -/
  cases' ys with z zs
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      x d : α
      xs' : List α
      hd : Membership.mem xs' d
      y : α
      ih : (∀ (x : α), Membership.mem List.nil x → Membership.mem xs' x) → Membershi …
      hxs' : ∀ (x : α), Membership.mem (List.cons y List.nil) x → Membership.mem xs' x
      ⊢ Membership.mem xs' ((List.cons y List.nil).nextOr x d)
    -/
  · exact hd
    /-
      🎉 no goals
    -/
  /-
    case cons.cons
    α : Type u_1
    inst✝ : DecidableEq α
    x d : α
    xs' : List α
    hd : Membership.mem xs' d
    y z : α
    zs : List α
    ih : (∀ (x : α), Membership.mem (List.cons z zs) x → Membership.mem xs' x) → M …
    hxs' : ∀ (x : α), Membership.mem (List.cons y (List.cons z zs)) x → Membership …
    ⊢ Membership.mem xs' ((List.cons y (List.cons z zs)).nextOr x d)
  -/
  rw [nextOr]
  /-
    case cons.cons
    α : Type u_1
    inst✝ : DecidableEq α
    x d : α
    xs' : List α
    hd : Membership.mem xs' d
    y z : α
    zs : List α
    ih : (∀ (x : α), Membership.mem (List.cons z zs) x → Membership.mem xs' x) → M …
    hxs' : ∀ (x : α), Membership.mem (List.cons y (List.cons z zs)) x → Membership …
    ⊢ Membership.mem xs' (ite (Eq x y) z ((List.cons z zs).nextOr x d))
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      x d : α
      xs' : List α
      hd : Membership.mem xs' d
      y z : α
      zs : List α
      ih : (∀ (x : α), Membership.mem (List.cons z zs) x → Membership.mem xs' x) → M …
      hxs' : ∀ (x : α), Membership.mem (List.cons y (List.cons z zs)) x → Membership …
      h : Eq x y
      ⊢ Membership.mem xs' z
    -/
  · exact hxs' _ (mem_cons_of_mem _ (mem_cons_self _ _))
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      x d : α
      xs' : List α
      hd : Membership.mem xs' d
      y z : α
      zs : List α
      ih : (∀ (x : α), Membership.mem (List.cons z zs) x → Membership.mem xs' x) → M …
      hxs' : ∀ (x : α), Membership.mem (List.cons y (List.cons z zs)) x → Membership …
      h : Not (Eq x y)
      ⊢ Membership.mem xs' ((List.cons z zs).nextOr x d)
    -/
  · exact ih fun _ h => hxs' _ (mem_cons_of_mem _ h)
    /-
      🎉 no goals
    -/


/-- Given an element `x : α` of `l : List α` such that `x ∈ l`, get the next
element of `l`. This works from head to tail, (including a check for last element)
so it will match on first hit, ignoring later duplicates.

For example:
 * `next [1, 2, 3] 2 _ = 3`
 * `next [1, 2, 3] 3 _ = 1`
 * `next [1, 2, 3, 2, 4] 2 _ = 3`
 * `next [1, 2, 3, 2] 2 _ = 3`
 * `next [1, 1, 2, 3, 2] 1 _ = 1`
-/
def next (l : List α) (x : α) (h : x ∈ l) : α :=
  nextOr l x (l.get ⟨0, length_pos_of_mem h⟩)


/-- Given an element `x : α` of `l : List α` such that `x ∈ l`, get the previous
element of `l`. This works from head to tail, (including a check for last element)
so it will match on first hit, ignoring later duplicates.

 * `prev [1, 2, 3] 2 _ = 1`
 * `prev [1, 2, 3] 1 _ = 3`
 * `prev [1, 2, 3, 2, 4] 2 _ = 1`
 * `prev [1, 2, 3, 4, 2] 2 _ = 1`
 * `prev [1, 1, 2] 1 _ = 2`
-/
def prev : ∀ l : List α, ∀ x ∈ l, α
                   /-
                     α : Type u_1
                     inst✝ : DecidableEq α
                     x✝ : α
                     h : Membership.mem List.nil x✝
                     ⊢ α
                   -/
  | [], _, h => by simp at h
                   /-
                     🎉 no goals
                   -/
  | [y], _, _ => y
  | y :: z :: xs, x, h =>
    if hx : x = y then getLast (z :: xs) (cons_ne_nil _ _)
                                                   /-
                                                     α : Type u_1
                                                     inst✝ : DecidableEq α
                                                     y z : α
                                                     xs : List α
                                                     x : α
                                                     h : Membership.mem (List.cons y (List.cons z xs)) x
                                                     hx : Not (Eq x y)
                                                     ⊢ Membership.mem (List.cons z xs) x
                                                   -/
    else if x = z then y else prev (z :: xs) x (by simpa [hx] using h)
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem next_singleton (x y : α) (h : x ∈ [y]) : next [y] x h = y :=
  rfl


@[simp]
theorem prev_singleton (x y : α) (h : x ∈ [y]) : prev [y] x h = y :=
  rfl


theorem next_cons_cons_eq' (y z : α) (h : x ∈ y :: z :: l) (hx : x = y) :
                                     /-
                                       α : Type u_1
                                       inst✝ : DecidableEq α
                                       l : List α
                                       x y z : α
                                       h : Membership.mem (List.cons y (List.cons z l)) x
                                       hx : Eq x y
                                       ⊢ Eq ((List.cons y (List.cons z l)).next x h) z
                                     -/
    next (y :: z :: l) x h = z := by rw [next, nextOr, if_pos hx]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem next_cons_cons_eq (z : α) (h : x ∈ x :: z :: l) : next (x :: z :: l) x h = z :=
  next_cons_cons_eq' l x x z h rfl


theorem next_ne_head_ne_getLast (h : x ∈ l) (y : α) (h : x ∈ y :: l) (hy : x ≠ y)
    (hx : x ≠ getLast (y :: l) (cons_ne_nil _ _)) :
                                     /-
                                       α : Type u_1
                                       inst✝ : DecidableEq α
                                       l : List α
                                       x : α
                                       h✝ : Membership.mem l x
                                       y : α
                                       h : Membership.mem (List.cons y l) x
                                       hy : Ne x y
                                       hx : Ne x ((List.cons y l).getLast ⋯)
                                       ⊢ Membership.mem l x
                                     -/
    next (y :: l) x h = next l x (by simpa [hy] using h) := by
                                     /-
                                       🎉 no goals
                                     -/
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    h✝ : Membership.mem l x
    y : α
    h : Membership.mem (List.cons y l) x
    hy : Ne x y
    hx : Ne x ((List.cons y l).getLast ⋯)
    ⊢ Eq ((List.cons y l).next x h) (l.next x ⋯)
  -/
  rw [next, next, nextOr_cons_of_ne _ _ _ _ hy, nextOr_eq_nextOr_of_mem_of_ne]
    /-
      case x_mem
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      x : α
      h✝ : Membership.mem l x
      y : α
      h : Membership.mem (List.cons y l) x
      hy : Ne x y
      hx : Ne x ((List.cons y l).getLast ⋯)
      ⊢ Membership.mem l x
    -/
  · rwa [getLast_cons] at hx
    /-
      case x_mem
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      x : α
      h✝ : Membership.mem l x
      y : α
      h : Membership.mem (List.cons y l) x
      hy : Ne x y
      hx : Ne x ((List.cons y l).getLast ⋯)
      ⊢ Ne l List.nil
    -/
    exact ne_nil_of_mem (by assumption)
    /-
      🎉 no goals
    -/
    /-
      case x_ne
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      x : α
      h✝ : Membership.mem l x
      y : α
      h : Membership.mem (List.cons y l) x
      hy : Ne x y
      hx : Ne x ((List.cons y l).getLast ⋯)
      ⊢ Ne x (l.getLast ⋯)
    -/
  · rwa [getLast_cons] at hx
    /-
      🎉 no goals
    -/


theorem next_cons_concat (y : α) (hy : x ≠ y) (hx : x ∉ l)
    (h : x ∈ y :: l ++ [x] := mem_append_right _ (mem_singleton_self x)) :
    next (y :: l ++ [x]) x h = y := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x y : α
    hy : Ne x y
    hx : Not (Membership.mem l x)
    h : optParam (Membership.mem (HAppend.hAppend (List.cons y l) (List.cons x Lis …
    ⊢ Eq ((HAppend.hAppend (List.cons y l) (List.cons x List.nil)).next x h) y
  -/
  rw [next, nextOr_concat]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      x y : α
      hy : Ne x y
      hx : Not (Membership.mem l x)
      h : optParam (Membership.mem (HAppend.hAppend (List.cons y l) (List.cons x Lis …
      ⊢ Eq ((HAppend.hAppend (List.cons y l) (List.cons x List.nil)).get ⟨0, ⋯⟩) y
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      x y : α
      hy : Ne x y
      hx : Not (Membership.mem l x)
      h : optParam (Membership.mem (HAppend.hAppend (List.cons y l) (List.cons x Lis …
      ⊢ Not (Membership.mem (List.cons y l) x)
    -/
  · simp [hy, hx]
    /-
      🎉 no goals
    -/


theorem next_getLast_cons (h : x ∈ l) (y : α) (h : x ∈ y :: l) (hy : x ≠ y)
    (hx : x = getLast (y :: l) (cons_ne_nil _ _)) (hl : Nodup l) : next (y :: l) x h = y := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    h✝ : Membership.mem l x
    y : α
    h : Membership.mem (List.cons y l) x
    hy : Ne x y
    hx : Eq x ((List.cons y l).getLast ⋯)
    hl : l.Nodup
    ⊢ Eq ((List.cons y l).next x h) y
  -/
  rw [next, get, ← dropLast_append_getLast (cons_ne_nil y l), hx, nextOr_concat]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    h✝ : Membership.mem l x
    y : α
    h : Membership.mem (List.cons y l) x
    hy : Ne x y
    hx : Eq x ((List.cons y l).getLast ⋯)
    hl : l.Nodup
    ⊢ Not (Membership.mem (List.cons y l).dropLast ((List.cons y l).getLast ⋯))
  -/
  subst hx
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    y : α
    hl : l.Nodup
    h✝ : Membership.mem l ((List.cons y l).getLast ⋯)
    h : Membership.mem (List.cons y l) ((List.cons y l).getLast ⋯)
    hy : Ne ((List.cons y l).getLast ⋯) y
    ⊢ Not (Membership.mem (List.cons y l).dropLast ((List.cons y l).getLast ⋯))
  -/
  intro H
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    y : α
    hl : l.Nodup
    h✝ : Membership.mem l ((List.cons y l).getLast ⋯)
    h : Membership.mem (List.cons y l) ((List.cons y l).getLast ⋯)
    hy : Ne ((List.cons y l).getLast ⋯) y
    H : Membership.mem (List.cons y l).dropLast ((List.cons y l).getLast ⋯)
    ⊢ False
  -/
  obtain ⟨⟨_ | k, hk⟩, hk'⟩ := get_of_mem H
    /-
      case h.intro.mk.zero
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      y : α
      hl : l.Nodup
      h✝ : Membership.mem l ((List.cons y l).getLast ⋯)
      h : Membership.mem (List.cons y l) ((List.cons y l).getLast ⋯)
      hy : Ne ((List.cons y l).getLast ⋯) y
      H : Membership.mem (List.cons y l).dropLast ((List.cons y l).getLast ⋯)
      hk : LT.lt 0 (List.cons y l).dropLast.length
      hk' : Eq ((List.cons y l).dropLast.get ⟨0, hk⟩) ((List.cons y l).getLast ⋯)
      ⊢ False
    -/
  · rw [← Option.some_inj] at hk'
    rw [← get?_eq_get, dropLast_eq_take, get?_eq_getElem?, getElem?_take_of_lt, getElem?_cons_zero,
      Option.some_inj] at hk'
      /-
        case h.intro.mk.zero
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        y : α
        hl : l.Nodup
        h✝ : Membership.mem l ((List.cons y l).getLast ⋯)
        h : Membership.mem (List.cons y l) ((List.cons y l).getLast ⋯)
        hy : Ne ((List.cons y l).getLast ⋯) y
        H : Membership.mem (List.cons y l).dropLast ((List.cons y l).getLast ⋯)
        hk : LT.lt 0 (List.cons y l).dropLast.length
        hk' : Eq y ((List.cons y l).getLast ⋯)
        ⊢ False
      -/
    · exact hy (Eq.symm hk')
      /-
        🎉 no goals
      -/
    /-
      case h.intro.mk.zero
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      y : α
      hl : l.Nodup
      h✝ : Membership.mem l ((List.cons y l).getLast ⋯)
      h : Membership.mem (List.cons y l) ((List.cons y l).getLast ⋯)
      hy : Ne ((List.cons y l).getLast ⋯) y
      H : Membership.mem (List.cons y l).dropLast ((List.cons y l).getLast ⋯)
      hk : LT.lt 0 (List.cons y l).dropLast.length
      hk' : Eq (GetElem?.getElem? (List.take (HSub.hSub (List.cons y l).length 1) (L …
      ⊢ LT.lt 0 (HSub.hSub (List.cons y l).length 1)
    -/
    rw [length_cons]
    /-
      case h.intro.mk.zero
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      y : α
      hl : l.Nodup
      h✝ : Membership.mem l ((List.cons y l).getLast ⋯)
      h : Membership.mem (List.cons y l) ((List.cons y l).getLast ⋯)
      hy : Ne ((List.cons y l).getLast ⋯) y
      H : Membership.mem (List.cons y l).dropLast ((List.cons y l).getLast ⋯)
      hk : LT.lt 0 (List.cons y l).dropLast.length
      hk' : Eq (GetElem?.getElem? (List.take (HSub.hSub (List.cons y l).length 1) (L …
      ⊢ LT.lt 0 (HSub.hSub (HAdd.hAdd l.length 1) 1)
    -/
    exact length_pos_of_mem (by assumption)
    /-
      🎉 no goals
    -/
  /-
    case h.intro.mk.succ
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    y : α
    hl : l.Nodup
    h✝ : Membership.mem l ((List.cons y l).getLast ⋯)
    h : Membership.mem (List.cons y l) ((List.cons y l).getLast ⋯)
    hy : Ne ((List.cons y l).getLast ⋯) y
    H : Membership.mem (List.cons y l).dropLast ((List.cons y l).getLast ⋯)
    k : Nat
    hk : LT.lt (HAdd.hAdd k 1) (List.cons y l).dropLast.length
    hk' : Eq ((List.cons y l).dropLast.get ⟨HAdd.hAdd k 1, hk⟩) ((List.cons y l).g …
    ⊢ False
  -/
  suffices k + 1 = l.length by simp [this] at hk
  /-
    case h.intro.mk.succ
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    y : α
    hl : l.Nodup
    h✝ : Membership.mem l ((List.cons y l).getLast ⋯)
    h : Membership.mem (List.cons y l) ((List.cons y l).getLast ⋯)
    hy : Ne ((List.cons y l).getLast ⋯) y
    H : Membership.mem (List.cons y l).dropLast ((List.cons y l).getLast ⋯)
    k : Nat
    hk : LT.lt (HAdd.hAdd k 1) (List.cons y l).dropLast.length
    hk' : Eq ((List.cons y l).dropLast.get ⟨HAdd.hAdd k 1, hk⟩) ((List.cons y l).g …
    ⊢ Eq (HAdd.hAdd k 1) l.length
  -/
  cases' l with hd tl
    /-
      case h.intro.mk.succ.nil
      α : Type u_1
      inst✝ : DecidableEq α
      y : α
      k : Nat
      hl : List.nil.Nodup
      h✝ : Membership.mem List.nil ((List.cons y List.nil).getLast ⋯)
      h : Membership.mem (List.cons y List.nil) ((List.cons y List.nil).getLast ⋯)
      hy : Ne ((List.cons y List.nil).getLast ⋯) y
      H : Membership.mem (List.cons y List.nil).dropLast ((List.cons y List.nil).get …
      hk : LT.lt (HAdd.hAdd k 1) (List.cons y List.nil).dropLast.length
      hk' : Eq ((List.cons y List.nil).dropLast.get ⟨HAdd.hAdd k 1, hk⟩) ((List.cons …
      ⊢ Eq (HAdd.hAdd k 1) List.nil.length
    -/
  · simp at hk
    /-
      🎉 no goals
    -/
    /-
      case h.intro.mk.succ.cons
      α : Type u_1
      inst✝ : DecidableEq α
      y : α
      k : Nat
      hd : α
      tl : List α
      hl : (List.cons hd tl).Nodup
      h✝ : Membership.mem (List.cons hd tl) ((List.cons y (List.cons hd tl)).getLast …
      h : Membership.mem (List.cons y (List.cons hd tl)) ((List.cons y (List.cons hd …
      hy : Ne ((List.cons y (List.cons hd tl)).getLast ⋯) y
      H : Membership.mem (List.cons y (List.cons hd tl)).dropLast ((List.cons y (Lis …
      hk : LT.lt (HAdd.hAdd k 1) (List.cons y (List.cons hd tl)).dropLast.length
      hk' : Eq ((List.cons y (List.cons hd tl)).dropLast.get ⟨HAdd.hAdd k 1, hk⟩) (( …
      ⊢ Eq (HAdd.hAdd k 1) (List.cons hd tl).length
    -/
  · rw [nodup_iff_injective_get] at hl
    /-
      case h.intro.mk.succ.cons
      α : Type u_1
      inst✝ : DecidableEq α
      y : α
      k : Nat
      hd : α
      tl : List α
      hl : Function.Injective (List.cons hd tl).get
      h✝ : Membership.mem (List.cons hd tl) ((List.cons y (List.cons hd tl)).getLast …
      h : Membership.mem (List.cons y (List.cons hd tl)) ((List.cons y (List.cons hd …
      hy : Ne ((List.cons y (List.cons hd tl)).getLast ⋯) y
      H : Membership.mem (List.cons y (List.cons hd tl)).dropLast ((List.cons y (Lis …
      hk : LT.lt (HAdd.hAdd k 1) (List.cons y (List.cons hd tl)).dropLast.length
      hk' : Eq ((List.cons y (List.cons hd tl)).dropLast.get ⟨HAdd.hAdd k 1, hk⟩) (( …
      ⊢ Eq (HAdd.hAdd k 1) (List.cons hd tl).length
    -/
    rw [length, Nat.succ_inj']
    refine Fin.val_eq_of_eq <| @hl ⟨k, Nat.lt_of_succ_lt <| by simpa using hk⟩
      ⟨tl.length, by simp⟩ ?_
    /-
      case h.intro.mk.succ.cons
      α : Type u_1
      inst✝ : DecidableEq α
      y : α
      k : Nat
      hd : α
      tl : List α
      hl : Function.Injective (List.cons hd tl).get
      h✝ : Membership.mem (List.cons hd tl) ((List.cons y (List.cons hd tl)).getLast …
      h : Membership.mem (List.cons y (List.cons hd tl)) ((List.cons y (List.cons hd …
      hy : Ne ((List.cons y (List.cons hd tl)).getLast ⋯) y
      H : Membership.mem (List.cons y (List.cons hd tl)).dropLast ((List.cons y (Lis …
      hk : LT.lt (HAdd.hAdd k 1) (List.cons y (List.cons hd tl)).dropLast.length
      hk' : Eq ((List.cons y (List.cons hd tl)).dropLast.get ⟨HAdd.hAdd k 1, hk⟩) (( …
      ⊢ Eq ((List.cons hd tl).get ⟨k, ⋯⟩) ((List.cons hd tl).get ⟨tl.length, ⋯⟩)
    -/
    rw [← Option.some_inj] at hk'
    rw [← get?_eq_get, dropLast_eq_take, get?_eq_getElem?, getElem?_take_of_lt, getElem?_cons_succ,
      getElem?_eq_getElem, Option.some_inj] at hk'
      /-
        case h.intro.mk.succ.cons
        α : Type u_1
        inst✝ : DecidableEq α
        y : α
        k : Nat
        hd : α
        tl : List α
        hl : Function.Injective (List.cons hd tl).get
        h✝ : Membership.mem (List.cons hd tl) ((List.cons y (List.cons hd tl)).getLast …
        h : Membership.mem (List.cons y (List.cons hd tl)) ((List.cons y (List.cons hd …
        hy : Ne ((List.cons y (List.cons hd tl)).getLast ⋯) y
        H : Membership.mem (List.cons y (List.cons hd tl)).dropLast ((List.cons y (Lis …
        hk : LT.lt (HAdd.hAdd k 1) (List.cons y (List.cons hd tl)).dropLast.length
        hk'✝ : Eq (GetElem?.getElem? (List.cons hd tl) k) (Option.some ((List.cons y ( …
        hk' : Eq (GetElem.getElem (List.cons hd tl) k ?h.intro.mk.succ.cons) ((List.co …
        ⊢ Eq ((List.cons hd tl).get ⟨k, ⋯⟩) ((List.cons hd tl).get ⟨tl.length, ⋯⟩)
      -/
    · rw [get_eq_getElem, hk']
      simp only [getLast_eq_getElem, length_cons, Nat.succ_eq_add_one, Nat.succ_sub_succ_eq_sub,
        Nat.sub_zero, get_eq_getElem, getElem_cons_succ]
    /-
      case h.intro.mk.succ.cons
      α : Type u_1
      inst✝ : DecidableEq α
      y : α
      k : Nat
      hd : α
      tl : List α
      hl : Function.Injective (List.cons hd tl).get
      h✝ : Membership.mem (List.cons hd tl) ((List.cons y (List.cons hd tl)).getLast …
      h : Membership.mem (List.cons y (List.cons hd tl)) ((List.cons y (List.cons hd …
      hy : Ne ((List.cons y (List.cons hd tl)).getLast ⋯) y
      H : Membership.mem (List.cons y (List.cons hd tl)).dropLast ((List.cons y (Lis …
      hk : LT.lt (HAdd.hAdd k 1) (List.cons y (List.cons hd tl)).dropLast.length
      hk' : Eq (GetElem?.getElem? (List.take (HSub.hSub (List.cons y (List.cons hd t …
      ⊢ LT.lt (HAdd.hAdd k 1) (HSub.hSub (List.cons y (List.cons hd tl)).length 1)
    -/
    simpa using hk
    /-
      🎉 no goals
    -/


theorem prev_getLast_cons' (y : α) (hxy : x ∈ y :: l) (hx : x = y) :
                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : DecidableEq α
                                                                     l : List α
                                                                     x y : α
                                                                     hxy : Membership.mem (List.cons y l) x
                                                                     hx : Eq x y
                                                                     ⊢ Eq ((List.cons y l).prev x hxy) ((List.cons y l).getLast ⋯)
                                                                   -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    prev (y :: l) x hxy = getLast (y :: l) (cons_ne_nil _ _) := by cases l <;> simp [prev, hx]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
theorem prev_getLast_cons (h : x ∈ x :: l) :
    prev (x :: l) x h = getLast (x :: l) (cons_ne_nil _ _) :=
  prev_getLast_cons' l x x h rfl


theorem prev_cons_cons_eq' (y z : α) (h : x ∈ y :: z :: l) (hx : x = y) :
                                                                      /-
                                                                        α : Type u_1
                                                                        inst✝ : DecidableEq α
                                                                        l : List α
                                                                        x y z : α
                                                                        h : Membership.mem (List.cons y (List.cons z l)) x
                                                                        hx : Eq x y
                                                                        ⊢ Eq ((List.cons y (List.cons z l)).prev x h) ((List.cons z l).getLast ⋯)
                                                                      -/
    prev (y :: z :: l) x h = getLast (z :: l) (cons_ne_nil _ _) := by rw [prev, dif_pos hx]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem prev_cons_cons_eq (z : α) (h : x ∈ x :: z :: l) :
    prev (x :: z :: l) x h = getLast (z :: l) (cons_ne_nil _ _) :=
  prev_cons_cons_eq' l x x z h rfl


theorem prev_cons_cons_of_ne' (y z : α) (h : x ∈ y :: z :: l) (hy : x ≠ y) (hz : x = z) :
    prev (y :: z :: l) x h = y := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x y z : α
    h : Membership.mem (List.cons y (List.cons z l)) x
    hy : Ne x y
    hz : Eq x z
    ⊢ Eq ((List.cons y (List.cons z l)).prev x h) y
  -/
  cases l
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      x y z : α
      hy : Ne x y
      hz : Eq x z
      h : Membership.mem (List.cons y (List.cons z List.nil)) x
      ⊢ Eq ((List.cons y (List.cons z List.nil)).prev x h) y
    -/
  · simp [prev, hy, hz]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      x y z : α
      hy : Ne x y
      hz : Eq x z
      head✝ : α
      tail✝ : List α
      h : Membership.mem (List.cons y (List.cons z (List.cons head✝ tail✝))) x
      ⊢ Eq ((List.cons y (List.cons z (List.cons head✝ tail✝))).prev x h) y
    -/
  · rw [prev, dif_neg hy, if_pos hz]
    /-
      🎉 no goals
    -/


theorem prev_cons_cons_of_ne (y : α) (h : x ∈ y :: x :: l) (hy : x ≠ y) :
    prev (y :: x :: l) x h = y :=
  prev_cons_cons_of_ne' _ _ _ _ _ hy rfl


theorem prev_ne_cons_cons (y z : α) (h : x ∈ y :: z :: l) (hy : x ≠ y) (hz : x ≠ z) :
                                                 /-
                                                   α : Type u_1
                                                   inst✝ : DecidableEq α
                                                   l : List α
                                                   x y z : α
                                                   h : Membership.mem (List.cons y (List.cons z l)) x
                                                   hy : Ne x y
                                                   hz : Ne x z
                                                   ⊢ Membership.mem (List.cons z l) x
                                                 -/
    prev (y :: z :: l) x h = prev (z :: l) x (by simpa [hy] using h) := by
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x y z : α
    h : Membership.mem (List.cons y (List.cons z l)) x
    hy : Ne x y
    hz : Ne x z
    ⊢ Eq ((List.cons y (List.cons z l)).prev x h) ((List.cons z l).prev x ⋯)
  -/
  cases l
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      x y z : α
      hy : Ne x y
      hz : Ne x z
      h : Membership.mem (List.cons y (List.cons z List.nil)) x
      ⊢ Eq ((List.cons y (List.cons z List.nil)).prev x h) ((List.cons z List.nil).p …
    -/
  · simp [hy, hz] at h
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      x y z : α
      hy : Ne x y
      hz : Ne x z
      head✝ : α
      tail✝ : List α
      h : Membership.mem (List.cons y (List.cons z (List.cons head✝ tail✝))) x
      ⊢ Eq ((List.cons y (List.cons z (List.cons head✝ tail✝))).prev x h) ((List.con …
    -/
  · rw [prev, dif_neg hy, if_neg hz]
    /-
      🎉 no goals
    -/


theorem next_mem (h : x ∈ l) : l.next x h ∈ l :=
  nextOr_mem (get_mem _ _)


theorem prev_mem (h : x ∈ l) : l.prev x h ∈ l := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    h : Membership.mem l x
    ⊢ Membership.mem l (l.prev x h)
  -/
  cases' l with hd tl
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      x : α
      h : Membership.mem List.nil x
      ⊢ Membership.mem List.nil (List.nil.prev x h)
    -/
  · simp at h
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    x hd : α
    tl : List α
    h : Membership.mem (List.cons hd tl) x
    ⊢ Membership.mem (List.cons hd tl) ((List.cons hd tl).prev x h)
  -/
  induction' tl with hd' tl hl generalizing hd
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      x hd : α
      h : Membership.mem (List.cons hd List.nil) x
      ⊢ Membership.mem (List.cons hd List.nil) ((List.cons hd List.nil).prev x h)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons.cons
      α : Type u_1
      inst✝ : DecidableEq α
      x hd' : α
      tl : List α
      hl : ∀ (hd : α) (h : Membership.mem (List.cons hd tl) x), Membership.mem (List …
      hd : α
      h : Membership.mem (List.cons hd (List.cons hd' tl)) x
      ⊢ Membership.mem (List.cons hd (List.cons hd' tl)) ((List.cons hd (List.cons h …
    -/
  · by_cases hx : x = hd
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        x hd' : α
        tl : List α
        hl : ∀ (hd : α) (h : Membership.mem (List.cons hd tl) x), Membership.mem (List …
        hd : α
        h : Membership.mem (List.cons hd (List.cons hd' tl)) x
        hx : Eq x hd
        ⊢ Membership.mem (List.cons hd (List.cons hd' tl)) ((List.cons hd (List.cons h …
      -/
    · simp only [hx, prev_cons_cons_eq]
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        x hd' : α
        tl : List α
        hl : ∀ (hd : α) (h : Membership.mem (List.cons hd tl) x), Membership.mem (List …
        hd : α
        h : Membership.mem (List.cons hd (List.cons hd' tl)) x
        hx : Eq x hd
        ⊢ Membership.mem (List.cons hd (List.cons hd' tl)) ((List.cons hd' tl).getLast …
      -/
      exact mem_cons_of_mem _ (getLast_mem _)
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        x hd' : α
        tl : List α
        hl : ∀ (hd : α) (h : Membership.mem (List.cons hd tl) x), Membership.mem (List …
        hd : α
        h : Membership.mem (List.cons hd (List.cons hd' tl)) x
        hx : Not (Eq x hd)
        ⊢ Membership.mem (List.cons hd (List.cons hd' tl)) ((List.cons hd (List.cons h …
      -/
    · rw [prev, dif_neg hx]
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        x hd' : α
        tl : List α
        hl : ∀ (hd : α) (h : Membership.mem (List.cons hd tl) x), Membership.mem (List …
        hd : α
        h : Membership.mem (List.cons hd (List.cons hd' tl)) x
        hx : Not (Eq x hd)
        ⊢ Membership.mem (List.cons hd (List.cons hd' tl)) (ite (Eq x hd') hd ((List.c …
      -/
      split_ifs with hm
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          x hd' : α
          tl : List α
          hl : ∀ (hd : α) (h : Membership.mem (List.cons hd tl) x), Membership.mem (List …
          hd : α
          h : Membership.mem (List.cons hd (List.cons hd' tl)) x
          hx : Not (Eq x hd)
          hm : Eq x hd'
          ⊢ Membership.mem (List.cons hd (List.cons hd' tl)) hd
        -/
      · exact mem_cons_self _ _
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          inst✝ : DecidableEq α
          x hd' : α
          tl : List α
          hl : ∀ (hd : α) (h : Membership.mem (List.cons hd tl) x), Membership.mem (List …
          hd : α
          h : Membership.mem (List.cons hd (List.cons hd' tl)) x
          hx : Not (Eq x hd)
          hm : Not (Eq x hd')
          ⊢ Membership.mem (List.cons hd (List.cons hd' tl)) ((List.cons hd' tl).prev x ⋯)
        -/
      · exact mem_cons_of_mem _ (hl _ _)
        /-
          🎉 no goals
        -/


theorem next_get (l : List α) (h : Nodup l) (i : Fin l.length) :
    next l (l.get i) (get_mem _ _) =
      l.get ⟨(i + 1) % l.length, Nat.mod_lt _ (i.1.zero_le.trans_lt i.2)⟩ :=
  match l, h, i with
                   /-
                     α : Type u_1
                     inst✝ : DecidableEq α
                     l : List α
                     h : l.Nodup
                     i✝ : Fin l.length
                     x✝ : List.nil.Nodup
                     i : Fin List.nil.length
                     ⊢ Eq (List.nil.next (List.nil.get i) ⋯) (List.nil.get ⟨HMod.hMod (HAdd.hAdd (↑ …
                   -/
  | [], _, i => by simpa using i.2
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u_1
                      inst✝ : DecidableEq α
                      l : List α
                      h : l.Nodup
                      i : Fin l.length
                      head✝ : α
                      x✝¹ : (List.cons head✝ List.nil).Nodup
                      x✝ : Fin (List.cons head✝ List.nil).length
                      ⊢ Eq ((List.cons head✝ List.nil).next ((List.cons head✝ List.nil).get x✝) ⋯) ( …
                    -/
  | [_], _, _ => by simp
                    /-
                      🎉 no goals
                    -/
  | x::y::l, _h, ⟨0, h0⟩ => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l✝ : List α
      h : l✝.Nodup
      i : Fin l✝.length
      x y : α
      l : List α
      _h : (List.cons x (List.cons y l)).Nodup
      h0 : LT.lt 0 (List.cons x (List.cons y l)).length
      ⊢ Eq ((List.cons x (List.cons y l)).next ((List.cons x (List.cons y l)).get ⟨0 …
    -/
    have h₁ : get (x :: y :: l) ⟨0, h0⟩ = x := by simp
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l✝ : List α
      h : l✝.Nodup
      i : Fin l✝.length
      x y : α
      l : List α
      _h : (List.cons x (List.cons y l)).Nodup
      h0 : LT.lt 0 (List.cons x (List.cons y l)).length
      h₁ : Eq ((List.cons x (List.cons y l)).get ⟨0, h0⟩) x
      ⊢ Eq ((List.cons x (List.cons y l)).next ((List.cons x (List.cons y l)).get ⟨0 …
    -/
    rw [next_cons_cons_eq' _ _ _ _ _ h₁]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l✝ : List α
      h : l✝.Nodup
      i : Fin l✝.length
      x y : α
      l : List α
      _h : (List.cons x (List.cons y l)).Nodup
      h0 : LT.lt 0 (List.cons x (List.cons y l)).length
      h₁ : Eq ((List.cons x (List.cons y l)).get ⟨0, h0⟩) x
      ⊢ Eq y ((List.cons x (List.cons y l)).get ⟨HMod.hMod (HAdd.hAdd (↑⟨0, h0⟩) 1)  …
    -/
    simp
    /-
      🎉 no goals
    -/
  | x::y::l, hn, ⟨i+1, hi⟩ => by
    have hx' : (x :: y :: l).get ⟨i+1, hi⟩ ≠ x := by
      intro H
      suffices (i + 1 : ℕ) = 0 by simpa
      rw [nodup_iff_injective_get] at hn
      refine Fin.val_eq_of_eq (@hn ⟨i + 1, hi⟩ ⟨0, by simp⟩ ?_)
      simpa using H
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l✝ : List α
      h : l✝.Nodup
      i✝ : Fin l✝.length
      x y : α
      l : List α
      hn : (List.cons x (List.cons y l)).Nodup
      i : Nat
      hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
      hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
      ⊢ Eq ((List.cons x (List.cons y l)).next ((List.cons x (List.cons y l)).get ⟨H …
    -/
    have hi' : i ≤ l.length := Nat.le_of_lt_succ (Nat.succ_lt_succ_iff.1 hi)
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l✝ : List α
      h : l✝.Nodup
      i✝ : Fin l✝.length
      x y : α
      l : List α
      hn : (List.cons x (List.cons y l)).Nodup
      i : Nat
      hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
      hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
      hi' : LE.le i l.length
      ⊢ Eq ((List.cons x (List.cons y l)).next ((List.cons x (List.cons y l)).get ⟨H …
    -/
    rcases hi'.eq_or_lt with (hi' | hi')
      /-
        case inl
        α : Type u_1
        inst✝ : DecidableEq α
        l✝ : List α
        h : l✝.Nodup
        i✝ : Fin l✝.length
        x y : α
        l : List α
        hn : (List.cons x (List.cons y l)).Nodup
        i : Nat
        hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
        hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
        hi'✝ : LE.le i l.length
        hi' : Eq i l.length
        ⊢ Eq ((List.cons x (List.cons y l)).next ((List.cons x (List.cons y l)).get ⟨H …
      -/
    · subst hi'
      /-
        case inl
        α : Type u_1
        inst✝ : DecidableEq α
        l✝ : List α
        h : l✝.Nodup
        i : Fin l✝.length
        x y : α
        l : List α
        hn : (List.cons x (List.cons y l)).Nodup
        hi : LT.lt (HAdd.hAdd l.length 1) (List.cons x (List.cons y l)).length
        hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd l.length 1, hi⟩) x
        hi' : LE.le l.length l.length
        ⊢ Eq ((List.cons x (List.cons y l)).next ((List.cons x (List.cons y l)).get ⟨H …
      -/
      rw [next_getLast_cons]
        /-
          case inl
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h : l✝.Nodup
          i : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          hi : LT.lt (HAdd.hAdd l.length 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd l.length 1, hi⟩) x
          hi' : LE.le l.length l.length
          ⊢ Eq x ((List.cons x (List.cons y l)).get ⟨HMod.hMod (HAdd.hAdd (↑⟨HAdd.hAdd l …
        -/
      · simp [hi', get]
        /-
          🎉 no goals
        -/
        /-
          case inl.h
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h : l✝.Nodup
          i : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          hi : LT.lt (HAdd.hAdd l.length 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd l.length 1, hi⟩) x
          hi' : LE.le l.length l.length
          ⊢ Membership.mem (List.cons y l) ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd …
        -/
      · rw [get_cons_succ]; exact get_mem _ _
                            /-
                              🎉 no goals
                            -/
        /-
          case inl.hy
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h : l✝.Nodup
          i : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          hi : LT.lt (HAdd.hAdd l.length 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd l.length 1, hi⟩) x
          hi' : LE.le l.length l.length
          ⊢ Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd l.length 1, hi⟩) x
        -/
      · exact hx'
        /-
          🎉 no goals
        -/
        /-
          case inl.hx
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h : l✝.Nodup
          i : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          hi : LT.lt (HAdd.hAdd l.length 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd l.length 1, hi⟩) x
          hi' : LE.le l.length l.length
          ⊢ Eq ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd l.length 1, hi⟩) ((List.con …
        -/
      · simp [getLast_eq_getElem]
        /-
          🎉 no goals
        -/
        /-
          case inl.hl
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h : l✝.Nodup
          i : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          hi : LT.lt (HAdd.hAdd l.length 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd l.length 1, hi⟩) x
          hi' : LE.le l.length l.length
          ⊢ (List.cons y l).Nodup
        -/
      · exact hn.of_cons
        /-
          🎉 no goals
        -/
      /-
        case inr
        α : Type u_1
        inst✝ : DecidableEq α
        l✝ : List α
        h : l✝.Nodup
        i✝ : Fin l✝.length
        x y : α
        l : List α
        hn : (List.cons x (List.cons y l)).Nodup
        i : Nat
        hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
        hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
        hi'✝ : LE.le i l.length
        hi' : LT.lt i l.length
        ⊢ Eq ((List.cons x (List.cons y l)).next ((List.cons x (List.cons y l)).get ⟨H …
      -/
    · rw [next_ne_head_ne_getLast _ _ _ _ _ hx']
        /-
          case inr
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h : l✝.Nodup
          i✝ : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          i : Nat
          hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
          hi'✝ : LE.le i l.length
          hi' : LT.lt i l.length
          ⊢ Eq ((List.cons y l).next ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1,  …
        -/
      · simp only [get_cons_succ]
        /-
          case inr
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h : l✝.Nodup
          i✝ : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          i : Nat
          hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
          hi'✝ : LE.le i l.length
          hi' : LT.lt i l.length
          ⊢ Eq ((List.cons y l).next ((List.cons y l).get ⟨i, ⋯⟩) ⋯) ((List.cons x (List …
        -/
        rw [next_get (y::l), ← get_cons_succ (a := x)]
          /-
            case inr
            α : Type u_1
            inst✝ : DecidableEq α
            l✝ : List α
            h : l✝.Nodup
            i✝ : Fin l✝.length
            x y : α
            l : List α
            hn : (List.cons x (List.cons y l)).Nodup
            i : Nat
            hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
            hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
            hi'✝ : LE.le i l.length
            hi' : LT.lt i l.length
            ⊢ Eq ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd (HMod.hMod (HAdd.hAdd (↑⟨i, …
          -/
        · congr
          /-
            case inr.e_a.e_val
            α : Type u_1
            inst✝ : DecidableEq α
            l✝ : List α
            h : l✝.Nodup
            i✝ : Fin l✝.length
            x y : α
            l : List α
            hn : (List.cons x (List.cons y l)).Nodup
            i : Nat
            hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
            hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
            hi'✝ : LE.le i l.length
            hi' : LT.lt i l.length
            ⊢ Eq (HAdd.hAdd (HMod.hMod (HAdd.hAdd (↑⟨i, ⋯⟩) 1) (List.cons y l).length) 1)  …
          -/
          dsimp
          rw [Nat.mod_eq_of_lt (Nat.succ_lt_succ_iff.2 hi'),
            Nat.mod_eq_of_lt (Nat.succ_lt_succ_iff.2 (Nat.succ_lt_succ_iff.2 hi'))]
          /-
            α : Type u_1
            inst✝ : DecidableEq α
            l✝ : List α
            h : l✝.Nodup
            i✝ : Fin l✝.length
            x y : α
            l : List α
            hn : (List.cons x (List.cons y l)).Nodup
            i : Nat
            hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
            hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
            hi'✝ : LE.le i l.length
            hi' : LT.lt i l.length
            ⊢ LT.lt (HAdd.hAdd (HMod.hMod (HAdd.hAdd (↑⟨i, ⋯⟩) 1) (List.cons y l).length)  …
          -/
        · simp [Nat.mod_eq_of_lt (Nat.succ_lt_succ_iff.2 hi'), hi']
          /-
            🎉 no goals
          -/
          /-
            case inr.h
            α : Type u_1
            inst✝ : DecidableEq α
            l✝ : List α
            h : l✝.Nodup
            i✝ : Fin l✝.length
            x y : α
            l : List α
            hn : (List.cons x (List.cons y l)).Nodup
            i : Nat
            hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
            hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
            hi'✝ : LE.le i l.length
            hi' : LT.lt i l.length
            ⊢ (List.cons y l).Nodup
          -/
        · exact hn.of_cons
          /-
            🎉 no goals
          -/
        /-
          case inr
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h : l✝.Nodup
          i✝ : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          i : Nat
          hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
          hi'✝ : LE.le i l.length
          hi' : LT.lt i l.length
          ⊢ Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) ((List.cons x (Li …
        -/
      · rw [getLast_eq_getElem]
        /-
          case inr
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h : l✝.Nodup
          i✝ : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          i : Nat
          hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
          hi'✝ : LE.le i l.length
          hi' : LT.lt i l.length
          ⊢ Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) (GetElem.getElem  …
        -/
        intro h
        /-
          case inr
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h✝ : l✝.Nodup
          i✝ : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          i : Nat
          hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
          hi'✝ : LE.le i l.length
          hi' : LT.lt i l.length
          h : Eq ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) (GetElem.getEle …
          ⊢ False
        -/
        have := nodup_iff_injective_get.1 hn h
        /-
          case inr
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h✝ : l✝.Nodup
          i✝ : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          i : Nat
          hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
          hi'✝ : LE.le i l.length
          hi' : LT.lt i l.length
          h : Eq ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) (GetElem.getEle …
          this : Eq ⟨HAdd.hAdd i 1, hi⟩ ⟨HSub.hSub (List.cons x (List.cons y l)).length  …
          ⊢ False
        -/
        simp at this; simp [this] at hi'
                      /-
                        🎉 no goals
                      -/
        /-
          α : Type u_1
          inst✝ : DecidableEq α
          l✝ : List α
          h : l✝.Nodup
          i✝ : Fin l✝.length
          x y : α
          l : List α
          hn : (List.cons x (List.cons y l)).Nodup
          i : Nat
          hi : LT.lt (HAdd.hAdd i 1) (List.cons x (List.cons y l)).length
          hx' : Ne ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd i 1, hi⟩) x
          hi'✝ : LE.le i l.length
          hi' : LT.lt i l.length
          ⊢ Membership.mem (List.cons y l) ((List.cons x (List.cons y l)).get ⟨HAdd.hAdd …
        -/
      · rw [get_cons_succ]; exact get_mem _ _
                            /-
                              🎉 no goals
                            -/

-- Unused variable linter incorrectly reports that `h` is unused here.

set_option linter.unusedVariables false in
theorem prev_get (l : List α) (h : Nodup l) (i : Fin l.length) :
    prev l (l.get i) (get_mem _ _) =
      l.get ⟨(i + (l.length - 1)) % l.length, Nat.mod_lt _ i.pos⟩ :=
  match l with
             /-
               α : Type u_1
               inst✝ : DecidableEq α
               l : List α
               h : List.nil.Nodup
               i : Fin List.nil.length
               ⊢ Eq (List.nil.prev (List.nil.get i) ⋯) (List.nil.get ⟨HMod.hMod (HAdd.hAdd (↑ …
             -/
  | [] => by simpa using i.2
             /-
               🎉 no goals
             -/
  | x::l => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l✝ : List α
      x : α
      l : List α
      h : (List.cons x l).Nodup
      i : Fin (List.cons x l).length
      ⊢ Eq ((List.cons x l).prev ((List.cons x l).get i) ⋯) ((List.cons x l).get ⟨HM …
    -/
    obtain ⟨n, hn⟩ := i
    induction l generalizing n x with
    | nil => simp
    | cons y l hl =>
      rcases n with (_ | _ | n)
      · simp [getLast_eq_getElem]
      · simp only [mem_cons, nodup_cons] at h
        push_neg at h
        simp only [List.prev_cons_cons_of_ne _ _ _ _ h.left.left.symm, List.length,
          List.get, add_comm, Nat.succ_add_sub_one, Nat.mod_self, zero_add]
      · rw [prev_ne_cons_cons]
        · convert hl y h.of_cons n.succ (Nat.le_of_succ_le_succ hn) using 1
          have : ∀ k hk, (y :: l).get ⟨k, hk⟩ = (x :: y :: l).get ⟨k + 1, Nat.succ_lt_succ hk⟩ := by
            simp [List.get]
          rw [this]
          congr
          simp only [Nat.add_succ_sub_one, add_zero, length]
          simp only [length, Nat.succ_lt_succ_iff] at hn
          set k := l.length
          rw [Nat.succ_add, ← Nat.add_succ, Nat.add_mod_right, Nat.succ_add, ← Nat.add_succ _ k,
            Nat.add_mod_right, Nat.mod_eq_of_lt, Nat.mod_eq_of_lt]
          · exact Nat.lt_succ_of_lt hn
          · exact Nat.succ_lt_succ (Nat.lt_succ_of_lt hn)
        · intro H
          suffices n.succ.succ = 0 by simpa
          suffices Fin.mk _ hn = ⟨0, by omega⟩ by rwa [Fin.mk.inj_iff] at this
          rw [nodup_iff_injective_get] at h
          apply h; rw [← H]; simp
        · intro H
          suffices n.succ.succ = 1 by simpa
          suffices Fin.mk _ hn = ⟨1, by omega⟩ by rwa [Fin.mk.inj_iff] at this
          rw [nodup_iff_injective_get] at h
          apply h; rw [← H]; simp


theorem pmap_next_eq_rotate_one (h : Nodup l) : (l.pmap l.next fun _ h => h) = l.rotate 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    ⊢ Eq (List.pmap l.next l ⋯) (l.rotate 1)
  -/
  apply List.ext_get
    /-
      case hl
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      ⊢ Eq (List.pmap l.next l ⋯).length (l.rotate 1).length
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      ⊢ ∀ (n : Nat) (h₁ : LT.lt n (List.pmap l.next l ⋯).length) (h₂ : LT.lt n (l.ro …
    -/
  · intros
    /-
      case h
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      n✝ : Nat
      h₁✝ : LT.lt n✝ (List.pmap l.next l ⋯).length
      h₂✝ : LT.lt n✝ (l.rotate 1).length
      ⊢ Eq ((List.pmap l.next l ⋯).get ⟨n✝, h₁✝⟩) ((l.rotate 1).get ⟨n✝, h₂✝⟩)
    -/
    rw [get_pmap, get_rotate, next_get _ h]
    /-
      🎉 no goals
    -/


theorem pmap_prev_eq_rotate_length_sub_one (h : Nodup l) :
    (l.pmap l.prev fun _ h => h) = l.rotate (l.length - 1) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    ⊢ Eq (List.pmap l.prev l ⋯) (l.rotate (HSub.hSub l.length 1))
  -/
  apply List.ext_get
    /-
      case hl
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      ⊢ Eq (List.pmap l.prev l ⋯).length (l.rotate (HSub.hSub l.length 1)).length
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      ⊢ ∀ (n : Nat) (h₁ : LT.lt n (List.pmap l.prev l ⋯).length) (h₂ : LT.lt n (l.ro …
    -/
  · intro n hn hn'
    /-
      case h
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      n : Nat
      hn : LT.lt n (List.pmap l.prev l ⋯).length
      hn' : LT.lt n (l.rotate (HSub.hSub l.length 1)).length
      ⊢ Eq ((List.pmap l.prev l ⋯).get ⟨n, hn⟩) ((l.rotate (HSub.hSub l.length 1)).g …
    -/
    rw [get_rotate, get_pmap, prev_get _ h]
    /-
      🎉 no goals
    -/


theorem prev_next (l : List α) (h : Nodup l) (x : α) (hx : x ∈ l) :
    prev l (next l x hx) (next_mem _ _ _) = x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Eq (l.prev (l.next x hx) ⋯) x
  -/
  obtain ⟨⟨n, hn⟩, rfl⟩ := get_of_mem hx
  /-
    case intro.mk
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    n : Nat
    hn : LT.lt n l.length
    hx : Membership.mem l (l.get ⟨n, hn⟩)
    ⊢ Eq (l.prev (l.next (l.get ⟨n, hn⟩) hx) ⋯) (l.get ⟨n, hn⟩)
  -/
  simp only [next_get, prev_get, h, Nat.mod_add_mod]
  /-
    case intro.mk
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    n : Nat
    hn : LT.lt n l.length
    hx : Membership.mem l (l.get ⟨n, hn⟩)
    ⊢ Eq (l.get ⟨HMod.hMod (HAdd.hAdd (HAdd.hAdd n 1) (HSub.hSub l.length 1)) l.le …
  -/
  cases' l with hd tl
    /-
      case intro.mk.nil
      α : Type u_1
      inst✝ : DecidableEq α
      n : Nat
      h : List.nil.Nodup
      hn : LT.lt n List.nil.length
      hx : Membership.mem List.nil (List.nil.get ⟨n, hn⟩)
      ⊢ Eq (List.nil.get ⟨HMod.hMod (HAdd.hAdd (HAdd.hAdd n 1) (HSub.hSub List.nil.l …
    -/
  · simp at hn
    /-
      🎉 no goals
    -/
  · have : (n + 1 + length tl) % (length tl + 1) = n := by
      rw [length_cons] at hn
      rw [add_assoc, add_comm 1, Nat.add_mod_right, Nat.mod_eq_of_lt hn]
    /-
      case intro.mk.cons
      α : Type u_1
      inst✝ : DecidableEq α
      n : Nat
      hd : α
      tl : List α
      h : (List.cons hd tl).Nodup
      hn : LT.lt n (List.cons hd tl).length
      hx : Membership.mem (List.cons hd tl) ((List.cons hd tl).get ⟨n, hn⟩)
      this : Eq (HMod.hMod (HAdd.hAdd (HAdd.hAdd n 1) tl.length) (HAdd.hAdd tl.lengt …
      ⊢ Eq ((List.cons hd tl).get ⟨HMod.hMod (HAdd.hAdd (HAdd.hAdd n 1) (HSub.hSub ( …
    -/
    simp only [length_cons, Nat.succ_sub_succ_eq_sub, Nat.sub_zero, Nat.succ_eq_add_one, this]
    /-
      🎉 no goals
    -/


theorem next_prev (l : List α) (h : Nodup l) (x : α) (hx : x ∈ l) :
    next l (prev l x hx) (prev_mem _ _ _) = x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Eq (l.next (l.prev x hx) ⋯) x
  -/
  obtain ⟨⟨n, hn⟩, rfl⟩ := get_of_mem hx
  /-
    case intro.mk
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    n : Nat
    hn : LT.lt n l.length
    hx : Membership.mem l (l.get ⟨n, hn⟩)
    ⊢ Eq (l.next (l.prev (l.get ⟨n, hn⟩) hx) ⋯) (l.get ⟨n, hn⟩)
  -/
  simp only [next_get, prev_get, h, Nat.mod_add_mod]
  /-
    case intro.mk
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    n : Nat
    hn : LT.lt n l.length
    hx : Membership.mem l (l.get ⟨n, hn⟩)
    ⊢ Eq (l.get ⟨HMod.hMod (HAdd.hAdd (HAdd.hAdd n (HSub.hSub l.length 1)) 1) l.le …
  -/
  cases' l with hd tl
    /-
      case intro.mk.nil
      α : Type u_1
      inst✝ : DecidableEq α
      n : Nat
      h : List.nil.Nodup
      hn : LT.lt n List.nil.length
      hx : Membership.mem List.nil (List.nil.get ⟨n, hn⟩)
      ⊢ Eq (List.nil.get ⟨HMod.hMod (HAdd.hAdd (HAdd.hAdd n (HSub.hSub List.nil.leng …
    -/
  · simp at hn
    /-
      🎉 no goals
    -/
  · have : (n + length tl + 1) % (length tl + 1) = n := by
      rw [length_cons] at hn
      rw [add_assoc, Nat.add_mod_right, Nat.mod_eq_of_lt hn]
    /-
      case intro.mk.cons
      α : Type u_1
      inst✝ : DecidableEq α
      n : Nat
      hd : α
      tl : List α
      h : (List.cons hd tl).Nodup
      hn : LT.lt n (List.cons hd tl).length
      hx : Membership.mem (List.cons hd tl) ((List.cons hd tl).get ⟨n, hn⟩)
      this : Eq (HMod.hMod (HAdd.hAdd (HAdd.hAdd n tl.length) 1) (HAdd.hAdd tl.lengt …
      ⊢ Eq ((List.cons hd tl).get ⟨HMod.hMod (HAdd.hAdd (HAdd.hAdd n (HSub.hSub (Lis …
    -/
    simp [this]
    /-
      🎉 no goals
    -/


theorem prev_reverse_eq_next (l : List α) (h : Nodup l) (x : α) (hx : x ∈ l) :
    prev l.reverse x (mem_reverse.mpr hx) = next l x hx := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Eq (l.reverse.prev x ⋯) (l.next x hx)
  -/
  obtain ⟨k, hk, rfl⟩ := getElem_of_mem hx
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    k : Nat
    hk : LT.lt k l.length
    hx : Membership.mem l (GetElem.getElem l k hk)
    ⊢ Eq (l.reverse.prev (GetElem.getElem l k hk) ⋯) (l.next (GetElem.getElem l k  …
  -/
  have lpos : 0 < l.length := k.zero_le.trans_lt hk
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    k : Nat
    hk : LT.lt k l.length
    hx : Membership.mem l (GetElem.getElem l k hk)
    lpos : LT.lt 0 l.length
    ⊢ Eq (l.reverse.prev (GetElem.getElem l k hk) ⋯) (l.next (GetElem.getElem l k  …
  -/
  have key : l.length - 1 - k < l.length := by omega
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    k : Nat
    hk : LT.lt k l.length
    hx : Membership.mem l (GetElem.getElem l k hk)
    lpos : LT.lt 0 l.length
    key : LT.lt (HSub.hSub (HSub.hSub l.length 1) k) l.length
    ⊢ Eq (l.reverse.prev (GetElem.getElem l k hk) ⋯) (l.next (GetElem.getElem l k  …
  -/
  rw [← getElem_pmap l.next (fun _ h => h) (by simpa using hk)]
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    k : Nat
    hk : LT.lt k l.length
    hx : Membership.mem l (GetElem.getElem l k hk)
    lpos : LT.lt 0 l.length
    key : LT.lt (HSub.hSub (HSub.hSub l.length 1) k) l.length
    ⊢ Eq (l.reverse.prev (GetElem.getElem l k hk) ⋯) (GetElem.getElem (List.pmap l …
  -/
  simp_rw [getElem_eq_getElem_reverse (l := l), pmap_next_eq_rotate_one _ h]
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    k : Nat
    hk : LT.lt k l.length
    hx : Membership.mem l (GetElem.getElem l k hk)
    lpos : LT.lt 0 l.length
    key : LT.lt (HSub.hSub (HSub.hSub l.length 1) k) l.length
    ⊢ Eq (l.reverse.prev (GetElem.getElem l.reverse (HSub.hSub (HSub.hSub l.length …
  -/
  rw [← getElem_pmap l.reverse.prev fun _ h => h]
  · simp_rw [pmap_prev_eq_rotate_length_sub_one _ (nodup_reverse.mpr h), rotate_reverse,
      length_reverse, Nat.mod_eq_of_lt (Nat.sub_lt lpos Nat.succ_pos'),
      Nat.sub_sub_self (Nat.succ_le_of_lt lpos)]
    /-
      case intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      k : Nat
      hk : LT.lt k l.length
      hx : Membership.mem l (GetElem.getElem l k hk)
      lpos : LT.lt 0 l.length
      key : LT.lt (HSub.hSub (HSub.hSub l.length 1) k) l.length
      ⊢ Eq (GetElem.getElem (l.rotate (Nat.succ 0)).reverse (HSub.hSub (HSub.hSub l. …
    -/
    rw [getElem_eq_getElem_reverse]
      /-
        case intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        h : l.Nodup
        k : Nat
        hk : LT.lt k l.length
        hx : Membership.mem l (GetElem.getElem l k hk)
        lpos : LT.lt 0 l.length
        key : LT.lt (HSub.hSub (HSub.hSub l.length 1) k) l.length
        ⊢ Eq (GetElem.getElem (l.rotate (Nat.succ 0)).reverse.reverse (HSub.hSub (HSub …
      -/
    · simp [Nat.sub_sub_self (Nat.le_sub_one_of_lt hk)]
      /-
        🎉 no goals
      -/
    /-
      case intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      h : l.Nodup
      k : Nat
      hk : LT.lt k l.length
      hx : Membership.mem l (GetElem.getElem l k hk)
      lpos : LT.lt 0 l.length
      key : LT.lt (HSub.hSub (HSub.hSub l.length 1) k) l.length
      ⊢ LT.lt (HSub.hSub (HSub.hSub l.length 1) k) (List.pmap l.reverse.prev l.rever …
    -/
  · simpa
    /-
      🎉 no goals
    -/


theorem next_reverse_eq_prev (l : List α) (h : Nodup l) (x : α) (hx : x ∈ l) :
    next l.reverse x (mem_reverse.mpr hx) = prev l x hx := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Eq (l.reverse.next x ⋯) (l.prev x hx)
  -/
  convert (prev_reverse_eq_next l.reverse (nodup_reverse.mpr h) x (mem_reverse.mpr hx)).symm
  /-
    case h.e'_3.h.e'_3
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Eq l l.reverse.reverse
  -/
  exact (reverse_reverse l).symm
  /-
    🎉 no goals
  -/


theorem isRotated_next_eq {l l' : List α} (h : l ~r l') (hn : Nodup l) {x : α} (hx : x ∈ l) :
    l.next x hx = l'.next x (h.mem_iff.mp hx) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    h : l.IsRotated l'
    hn : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Eq (l.next x hx) (l'.next x ⋯)
  -/
  obtain ⟨k, hk, rfl⟩ := get_of_mem hx
  /-
    case intro.refl
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    h : l.IsRotated l'
    hn : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    ⊢ Eq (l.next (l.get k) hx) (l'.next (l.get k) ⋯)
  -/
  obtain ⟨n, rfl⟩ := id h
  /-
    case intro.refl.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hn : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    n : Nat
    h : l.IsRotated (l.rotate n)
    ⊢ Eq (l.next (l.get k) hx) ((l.rotate n).next (l.get k) ⋯)
  -/
  rw [next_get _ hn]
  /-
    case intro.refl.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hn : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    n : Nat
    h : l.IsRotated (l.rotate n)
    ⊢ Eq (l.get ⟨HMod.hMod (HAdd.hAdd (↑k) 1) l.length, ⋯⟩) ((l.rotate n).next (l. …
  -/
  simp_rw [get_eq_get_rotate _ n k]
  /-
    case intro.refl.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hn : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    n : Nat
    h : l.IsRotated (l.rotate n)
    ⊢ Eq (l.get ⟨HMod.hMod (HAdd.hAdd (↑k) 1) l.length, ⋯⟩) ((l.rotate n).next ((l …
  -/
  rw [next_get _ (h.nodup_iff.mp hn), get_eq_get_rotate _ n]
  /-
    case intro.refl.intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hn : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    n : Nat
    h : l.IsRotated (l.rotate n)
    ⊢ Eq ((l.rotate n).get ⟨HMod.hMod (HAdd.hAdd (HSub.hSub l.length (HMod.hMod n  …
  -/
  simp [add_assoc]
  /-
    🎉 no goals
  -/


theorem isRotated_prev_eq {l l' : List α} (h : l ~r l') (hn : Nodup l) {x : α} (hx : x ∈ l) :
    l.prev x hx = l'.prev x (h.mem_iff.mp hx) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    h : l.IsRotated l'
    hn : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Eq (l.prev x hx) (l'.prev x ⋯)
  -/
  rw [← next_reverse_eq_prev _ hn, ← next_reverse_eq_prev _ (h.nodup_iff.mp hn)]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    h : l.IsRotated l'
    hn : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Eq (l.reverse.next x ⋯) (l'.reverse.next x ⋯)
  -/
  exact isRotated_next_eq h.reverse (nodup_reverse.mpr hn) _
  /-
    🎉 no goals
  -/


/-- `Cycle α` is the quotient of `List α` by cyclic permutation.
Duplicates are allowed.
-/
def Cycle (α : Type*) : Type _ :=
  Quotient (IsRotated.setoid α)


/-- The coercion from `List α` to `Cycle α` -/
@[coe] def ofList : List α → Cycle α :=
  Quot.mk _


instance : Coe (List α) (Cycle α) :=
  ⟨ofList⟩


@[simp]
theorem coe_eq_coe {l₁ l₂ : List α} : (l₁ : Cycle α) = (l₂ : Cycle α) ↔ l₁ ~r l₂ :=
  @Quotient.eq _ (IsRotated.setoid _) _ _


@[simp]
theorem mk_eq_coe (l : List α) : Quot.mk _ l = (l : Cycle α) :=
  rfl


@[simp]
theorem mk''_eq_coe (l : List α) : Quotient.mk'' l = (l : Cycle α) :=
  rfl


theorem coe_cons_eq_coe_append (l : List α) (a : α) :
    (↑(a :: l) : Cycle α) = (↑(l ++ [a]) : Cycle α) :=
                    /-
                      α : Type u_1
                      l : List α
                      a : α
                      ⊢ Eq ((List.cons a l).rotate 1) (HAppend.hAppend l (List.cons a List.nil))
                    -/
  Quot.sound ⟨1, by rw [rotate_cons_succ, rotate_zero]⟩
                    /-
                      🎉 no goals
                    -/


/-- The unique empty cycle. -/
def nil : Cycle α :=
  ([] : List α)


@[simp]
theorem coe_nil : ↑([] : List α) = @nil α :=
  rfl


@[simp]
theorem coe_eq_nil (l : List α) : (l : Cycle α) = nil ↔ l = [] :=
  coe_eq_coe.trans isRotated_nil_iff


/-- For consistency with `EmptyCollection (List α)`. -/
instance : EmptyCollection (Cycle α) :=
  ⟨nil⟩


@[simp]
theorem empty_eq : ∅ = @nil α :=
  rfl


instance : Inhabited (Cycle α) :=
  ⟨nil⟩


/-- An induction principle for `Cycle`. Use as `induction s`. -/
@[elab_as_elim, induction_eliminator]
theorem induction_on {C : Cycle α → Prop} (s : Cycle α) (H0 : C nil)
    (HI : ∀ (a) (l : List α), C ↑l → C ↑(a :: l)) : C s :=
  Quotient.inductionOn' s fun l => by
    /-
      α : Type u_1
      C : Cycle α → Prop
      s : Cycle α
      H0 : C Cycle.nil
      HI : ∀ (a : α) (l : List α), C ↑l → C ↑(List.cons a l)
      l : List α
      ⊢ C (Quotient.mk'' l)
    -/
    refine List.recOn l ?_ ?_ <;> simp only [mk''_eq_coe, coe_nil]
    /-
      case refine_1
      α : Type u_1
      C : Cycle α → Prop
      s : Cycle α
      H0 : C Cycle.nil
      HI : ∀ (a : α) (l : List α), C ↑l → C ↑(List.cons a l)
      l : List α
      ⊢ C Cycle.nil
    -/
    assumption'
    /-
      🎉 no goals
    -/


/-- For `x : α`, `s : Cycle α`, `x ∈ s` indicates that `x` occurs at least once in `s`. -/
def Mem (s : Cycle α) (a : α) : Prop :=
  Quot.liftOn s (fun l => a ∈ l) fun _ _ e => propext <| e.mem_iff


instance : Membership α (Cycle α) :=
  ⟨Mem⟩


@[simp]
theorem mem_coe_iff {a : α} {l : List α} : a ∈ (↑l : Cycle α) ↔ a ∈ l :=
  Iff.rfl


@[simp]
theorem not_mem_nil : ∀ a, a ∉ @nil α :=
  List.not_mem_nil


instance [DecidableEq α] : DecidableEq (Cycle α) := fun s₁ s₂ =>
  Quotient.recOnSubsingleton₂' s₁ s₂ fun _ _ => decidable_of_iff' _ Quotient.eq''


instance [DecidableEq α] (x : α) (s : Cycle α) : Decidable (x ∈ s) :=
  Quotient.recOnSubsingleton' s fun l => show Decidable (x ∈ l) from inferInstance


/-- Reverse a `s : Cycle α` by reversing the underlying `List`. -/
nonrec def reverse (s : Cycle α) : Cycle α :=
  Quot.map reverse (fun _ _ => IsRotated.reverse) s


@[simp]
theorem reverse_coe (l : List α) : (l : Cycle α).reverse = l.reverse :=
  rfl


@[simp]
theorem mem_reverse_iff {a : α} {s : Cycle α} : a ∈ s.reverse ↔ a ∈ s :=
  Quot.inductionOn s fun _ => mem_reverse


@[simp]
theorem reverse_reverse (s : Cycle α) : s.reverse.reverse = s :=
                                 /-
                                   α : Type u_1
                                   s : Cycle α
                                   x✝ : List α
                                   ⊢ Eq (Cycle.reverse (Quot.mk (⇑(List.IsRotated.setoid α)) x✝)).reverse (Quot.m …
                                 -/
  Quot.inductionOn s fun _ => by simp
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem reverse_nil : nil.reverse = @nil α :=
  rfl


/-- The length of the `s : Cycle α`, which is the number of elements, counting duplicates. -/
def length (s : Cycle α) : ℕ :=
  Quot.liftOn s List.length fun _ _ e => e.perm.length_eq


@[simp]
theorem length_coe (l : List α) : length (l : Cycle α) = l.length :=
  rfl


@[simp]
theorem length_nil : length (@nil α) = 0 :=
  rfl


@[simp]
theorem length_reverse (s : Cycle α) : s.reverse.length = s.length :=
  Quot.inductionOn s List.length_reverse


/-- A `s : Cycle α` that is at most one element. -/
def Subsingleton (s : Cycle α) : Prop :=
  s.length ≤ 1


theorem subsingleton_nil : Subsingleton (@nil α) := Nat.zero_le _


theorem length_subsingleton_iff {s : Cycle α} : Subsingleton s ↔ length s ≤ 1 :=
  Iff.rfl


@[simp]
theorem subsingleton_reverse_iff {s : Cycle α} : s.reverse.Subsingleton ↔ s.Subsingleton := by
  /-
    α : Type u_1
    s : Cycle α
    ⊢ Iff s.reverse.Subsingleton s.Subsingleton
  -/
  simp [length_subsingleton_iff]
  /-
    🎉 no goals
  -/


theorem Subsingleton.congr {s : Cycle α} (h : Subsingleton s) :
    ∀ ⦃x⦄ (_hx : x ∈ s) ⦃y⦄ (_hy : y ∈ s), x = y := by
  /-
    α : Type u_1
    s : Cycle α
    h : s.Subsingleton
    ⊢ ∀ ⦃x : α⦄, Membership.mem s x → ∀ ⦃y : α⦄, Membership.mem s y → Eq x y
  -/
  induction' s using Quot.inductionOn with l
  simp only [length_subsingleton_iff, length_coe, mk_eq_coe, le_iff_lt_or_eq, Nat.lt_add_one_iff,
    length_eq_zero, length_eq_one, Nat.not_lt_zero, false_or] at h
  /-
    case h
    α : Type u_1
    l : List α
    h : Or (Eq l List.nil) (Exists fun a => Eq l (List.cons a List.nil))
    ⊢ ∀ ⦃x : α⦄, Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) l) x → ∀ ⦃y  …
  -/
                                     /-
                                       🎉 no goals
                                     -/
  rcases h with (rfl | ⟨z, rfl⟩) <;> simp
                                     /-
                                       🎉 no goals
                                     -/


/-- A `s : Cycle α` that is made up of at least two unique elements. -/
def Nontrivial (s : Cycle α) : Prop :=
  ∃ x y : α, x ≠ y ∧ x ∈ s ∧ y ∈ s


@[simp]
theorem nontrivial_coe_nodup_iff {l : List α} (hl : l.Nodup) :
    Nontrivial (l : Cycle α) ↔ 2 ≤ l.length := by
  /-
    α : Type u_1
    l : List α
    hl : l.Nodup
    ⊢ Iff (↑l).Nontrivial (LE.le 2 l.length)
  -/
  rw [Nontrivial]
  /-
    α : Type u_1
    l : List α
    hl : l.Nodup
    ⊢ Iff (Exists fun x => Exists fun y => And (Ne x y) (And (Membership.mem (↑l)  …
  -/
  rcases l with (_ | ⟨hd, _ | ⟨hd', tl⟩⟩)
    /-
      case nil
      α : Type u_1
      hl : List.nil.Nodup
      ⊢ Iff (Exists fun x => Exists fun y => And (Ne x y) (And (Membership.mem (↑Lis …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons.nil
      α : Type u_1
      hd : α
      hl : (List.cons hd List.nil).Nodup
      ⊢ Iff (Exists fun x => Exists fun y => And (Ne x y) (And (Membership.mem (↑(Li …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp only [mem_cons, exists_prop, mem_coe_iff, List.length, Ne, Nat.succ_le_succ_iff,
      Nat.zero_le, iff_true]
    /-
      case cons.cons
      α : Type u_1
      hd hd' : α
      tl : List α
      hl : (List.cons hd (List.cons hd' tl)).Nodup
      ⊢ Exists fun x => Exists fun y => And (Not (Eq x y)) (And (Or (Eq x hd) (Or (E …
    -/
    refine ⟨hd, hd', ?_, by simp⟩
    /-
      case cons.cons
      α : Type u_1
      hd hd' : α
      tl : List α
      hl : (List.cons hd (List.cons hd' tl)).Nodup
      ⊢ Not (Eq hd hd')
    -/
    simp only [not_or, mem_cons, nodup_cons] at hl
    /-
      case cons.cons
      α : Type u_1
      hd hd' : α
      tl : List α
      hl : And (And (Not (Eq hd hd')) (Not (Membership.mem tl hd))) (And (Not (Membe …
      ⊢ Not (Eq hd hd')
    -/
    exact hl.left.left
    /-
      🎉 no goals
    -/


@[simp]
theorem nontrivial_reverse_iff {s : Cycle α} : s.reverse.Nontrivial ↔ s.Nontrivial := by
  /-
    α : Type u_1
    s : Cycle α
    ⊢ Iff s.reverse.Nontrivial s.Nontrivial
  -/
  simp [Nontrivial]
  /-
    🎉 no goals
  -/


theorem length_nontrivial {s : Cycle α} (h : Nontrivial s) : 2 ≤ length s := by
  /-
    α : Type u_1
    s : Cycle α
    h : s.Nontrivial
    ⊢ LE.le 2 s.length
  -/
  obtain ⟨x, y, hxy, hx, hy⟩ := h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    s : Cycle α
    x y : α
    hxy : Ne x y
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ LE.le 2 s.length
  -/
  induction' s using Quot.inductionOn with l
  /-
    case intro.intro.intro.intro.h
    α : Type u_1
    x y : α
    hxy : Ne x y
    l : List α
    hx : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) l) x
    hy : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) l) y
    ⊢ LE.le 2 (Cycle.length (Quot.mk (⇑(List.IsRotated.setoid α)) l))
  -/
  rcases l with (_ | ⟨hd, _ | ⟨hd', tl⟩⟩)
    /-
      case intro.intro.intro.intro.h.nil
      α : Type u_1
      x y : α
      hxy : Ne x y
      hx : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) List.nil) x
      hy : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) List.nil) y
      ⊢ LE.le 2 (Cycle.length (Quot.mk (⇑(List.IsRotated.setoid α)) List.nil))
    -/
  · simp at hx
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.h.cons.nil
      α : Type u_1
      x y : α
      hxy : Ne x y
      hd : α
      hx : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd List.n …
      hy : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd List.n …
      ⊢ LE.le 2 (Cycle.length (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd Li …
    -/
  · simp only [mem_coe_iff, mk_eq_coe, mem_singleton] at hx hy
    /-
      case intro.intro.intro.intro.h.cons.nil
      α : Type u_1
      x y : α
      hxy : Ne x y
      hd : α
      hx : Eq x hd
      hy : Eq y hd
      ⊢ LE.le 2 (Cycle.length (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd Li …
    -/
    simp [hx, hy] at hxy
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.h.cons.cons
      α : Type u_1
      x y : α
      hxy : Ne x y
      hd hd' : α
      tl : List α
      hx : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd (List. …
      hy : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd (List. …
      ⊢ LE.le 2 (Cycle.length (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd (L …
    -/
  · simp [Nat.succ_le_succ_iff]
    /-
      🎉 no goals
    -/


/-- The `s : Cycle α` contains no duplicates. -/
nonrec def Nodup (s : Cycle α) : Prop :=
  Quot.liftOn s Nodup fun _l₁ _l₂ e => propext <| e.nodup_iff


@[simp]
nonrec theorem nodup_nil : Nodup (@nil α) :=
  nodup_nil


@[simp]
theorem nodup_coe_iff {l : List α} : Nodup (l : Cycle α) ↔ l.Nodup :=
  Iff.rfl


@[simp]
theorem nodup_reverse_iff {s : Cycle α} : s.reverse.Nodup ↔ s.Nodup :=
  Quot.inductionOn s fun _ => nodup_reverse


theorem Subsingleton.nodup {s : Cycle α} (h : Subsingleton s) : Nodup s := by
  /-
    α : Type u_1
    s : Cycle α
    h : s.Subsingleton
    ⊢ s.Nodup
  -/
  induction' s using Quot.inductionOn with l
  /-
    case h
    α : Type u_1
    l : List α
    h : Cycle.Subsingleton (Quot.mk (⇑(List.IsRotated.setoid α)) l)
    ⊢ Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l)
  -/
  cases' l with hd tl
    /-
      case h.nil
      α : Type u_1
      h : Cycle.Subsingleton (Quot.mk (⇑(List.IsRotated.setoid α)) List.nil)
      ⊢ Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) List.nil)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.cons
      α : Type u_1
      hd : α
      tl : List α
      h : Cycle.Subsingleton (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd tl))
      ⊢ Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd tl))
    -/
  · have : tl = [] := by simpa [Subsingleton, length_eq_zero, Nat.succ_le_succ_iff] using h
    /-
      case h.cons
      α : Type u_1
      hd : α
      tl : List α
      h : Cycle.Subsingleton (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd tl))
      this : Eq tl List.nil
      ⊢ Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons hd tl))
    -/
    simp [this]
    /-
      🎉 no goals
    -/


theorem Nodup.nontrivial_iff {s : Cycle α} (h : Nodup s) : Nontrivial s ↔ ¬Subsingleton s := by
  /-
    α : Type u_1
    s : Cycle α
    h : s.Nodup
    ⊢ Iff s.Nontrivial (Not s.Subsingleton)
  -/
  rw [length_subsingleton_iff]
  /-
    α : Type u_1
    s : Cycle α
    h : s.Nodup
    ⊢ Iff s.Nontrivial (Not (LE.le s.length 1))
  -/
  induction s using Quotient.inductionOn'
  /-
    case h
    α : Type u_1
    a✝ : List α
    h : Cycle.Nodup (Quotient.mk'' a✝)
    ⊢ Iff (Cycle.Nontrivial (Quotient.mk'' a✝)) (Not (LE.le (Cycle.length (Quotien …
  -/
  simp only [mk''_eq_coe, nodup_coe_iff] at h
  /-
    case h
    α : Type u_1
    a✝ : List α
    h : a✝.Nodup
    ⊢ Iff (Cycle.Nontrivial (Quotient.mk'' a✝)) (Not (LE.le (Cycle.length (Quotien …
  -/
  simp [h, Nat.succ_le_iff]
  /-
    🎉 no goals
  -/


/-- The `s : Cycle α` as a `Multiset α`.
-/
def toMultiset (s : Cycle α) : Multiset α :=
  Quotient.liftOn' s (↑) fun _ _ h => Multiset.coe_eq_coe.mpr h.perm


@[simp]
theorem coe_toMultiset (l : List α) : (l : Cycle α).toMultiset = l :=
  rfl


@[simp]
theorem nil_toMultiset : nil.toMultiset = (0 : Multiset α) :=
  rfl


@[simp]
theorem card_toMultiset (s : Cycle α) : Multiset.card s.toMultiset = s.length :=
                              /-
                                α : Type u_1
                                s : Cycle α
                                ⊢ ∀ (a : List α), Eq (Cycle.toMultiset (Quotient.mk'' a)).card (Cycle.length ( …
                              -/
  Quotient.inductionOn' s (by simp)
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem toMultiset_eq_nil {s : Cycle α} : s.toMultiset = 0 ↔ s = Cycle.nil :=
                              /-
                                α : Type u_1
                                s : Cycle α
                                ⊢ ∀ (a : List α), Iff (Eq (Cycle.toMultiset (Quotient.mk'' a)) 0) (Eq (Quotien …
                              -/
  Quotient.inductionOn' s (by simp)
                              /-
                                🎉 no goals
                              -/


/-- The lift of `list.map`. -/
def map {β : Type*} (f : α → β) : Cycle α → Cycle β :=
  Quotient.map' (List.map f) fun _ _ h => h.map _


@[simp]
theorem map_nil {β : Type*} (f : α → β) : map f nil = nil :=
  rfl


@[simp]
theorem map_coe {β : Type*} (f : α → β) (l : List α) : map f ↑l = List.map f l :=
  rfl


@[simp]
theorem map_eq_nil {β : Type*} (f : α → β) (s : Cycle α) : map f s = nil ↔ s = nil :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                f : α → β
                                s : Cycle α
                                ⊢ ∀ (a : List α), Iff (Eq (Cycle.map f (Quotient.mk'' a)) Cycle.nil) (Eq (Quot …
                              -/
  Quotient.inductionOn' s (by simp)
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem mem_map {β : Type*} {f : α → β} {b : β} {s : Cycle α} :
    b ∈ s.map f ↔ ∃ a, a ∈ s ∧ f a = b :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                f : α → β
                                b : β
                                s : Cycle α
                                ⊢ ∀ (a : List α), Iff (Membership.mem (Cycle.map f (Quotient.mk'' a)) b) (Exis …
                              -/
  Quotient.inductionOn' s (by simp)
                              /-
                                🎉 no goals
                              -/


/-- The `Multiset` of lists that can make the cycle. -/
def lists (s : Cycle α) : Multiset (List α) :=
  Quotient.liftOn' s (fun l => (l.cyclicPermutations : Multiset (List α))) fun l₁ l₂ h => by
    /-
      α : Type u_1
      s : Cycle α
      l₁ l₂ : List α
      h : (List.IsRotated.setoid α) l₁ l₂
      ⊢ Eq ((fun l => ↑l.cyclicPermutations) l₁) ((fun l => ↑l.cyclicPermutations) l₂)
    -/
    simpa using h.cyclicPermutations.perm
    /-
      🎉 no goals
    -/


@[simp]
theorem lists_coe (l : List α) : lists (l : Cycle α) = ↑l.cyclicPermutations :=
  rfl


@[simp]
theorem mem_lists_iff_coe_eq {s : Cycle α} {l : List α} : l ∈ s.lists ↔ (l : Cycle α) = s :=
  Quotient.inductionOn' s fun l => by
    /-
      α : Type u_1
      s : Cycle α
      l✝ l : List α
      ⊢ Iff (Membership.mem (Cycle.lists (Quotient.mk'' l)) l✝) (Eq (↑l✝) (Quotient. …
    -/
    rw [lists, Quotient.liftOn'_mk'']
    /-
      α : Type u_1
      s : Cycle α
      l✝ l : List α
      ⊢ Iff (Membership.mem (↑l.cyclicPermutations) l✝) (Eq (↑l✝) (Quotient.mk'' l))
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem lists_nil : lists (@nil α) = [([] : List α)] := by
  /-
    α : Type u_1
    ⊢ Eq Cycle.nil.lists ↑(List.cons List.nil List.nil)
  -/
  rw [nil, lists_coe, cyclicPermutations_nil]
  /-
    🎉 no goals
  -/


/-- Auxiliary decidability algorithm for lists that contain at least two unique elements.
-/
def decidableNontrivialCoe : ∀ l : List α, Decidable (Nontrivial (l : Cycle α))
                      /-
                        α : Type u_1
                        inst✝ : DecidableEq α
                        ⊢ Not (↑List.nil).Nontrivial
                      -/
  | [] => isFalse (by simp [Nontrivial])
                      /-
                        🎉 no goals
                      -/
                       /-
                         α : Type u_1
                         inst✝ : DecidableEq α
                         x : α
                         ⊢ Not (↑(List.cons x List.nil)).Nontrivial
                       -/
  | [x] => isFalse (by simp [Nontrivial])
                       /-
                         🎉 no goals
                       -/
  | x :: y :: l =>
    if h : x = y then
                                                               /-
                                                                 α : Type u_1
                                                                 inst✝ : DecidableEq α
                                                                 x y : α
                                                                 l : List α
                                                                 h : Eq x y
                                                                 ⊢ Iff (↑(List.cons x (List.cons y l))).Nontrivial (↑(List.cons x l)).Nontrivial
                                                               -/
      @decidable_of_iff' _ (Nontrivial (x :: l : Cycle α)) (by simp [h, Nontrivial])
                                                               /-
                                                                 🎉 no goals
                                                               -/
        (decidableNontrivialCoe (x :: l))
                             /-
                               α : Type u_1
                               inst✝ : DecidableEq α
                               x y : α
                               l : List α
                               h : Not (Eq x y)
                               ⊢ Membership.mem (↑(List.cons x (List.cons y l))) x
                             -/
                             /-
                               🎉 no goals
                             -/
    else isTrue ⟨x, y, h, by simp, by simp⟩
                                      /-
                                        🎉 no goals
                                      -/


instance {s : Cycle α} : Decidable (Nontrivial s) :=
  Quot.recOnSubsingleton s decidableNontrivialCoe


instance {s : Cycle α} : Decidable (Nodup s) :=
  Quot.recOnSubsingleton s List.nodupDecidable


instance fintypeNodupCycle [Fintype α] : Fintype { s : Cycle α // s.Nodup } :=
                                                                        /-
                                                                          α : Type u_1
                                                                          inst✝¹ : DecidableEq α
                                                                          inst✝ : Fintype α
                                                                          l : Subtype fun l => l.Nodup
                                                                          ⊢ (↑↑l).Nodup
                                                                        -/
  Fintype.ofSurjective (fun l : { l : List α // l.Nodup } => ⟨l.val, by simpa using l.prop⟩)
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    fun ⟨s, hs⟩ => by
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x✝ : Subtype fun s => s.Nodup
      s : Cycle α
      hs : s.Nodup
      ⊢ Exists fun a => Eq ((fun l => ⟨↑↑l, ⋯⟩) a) ⟨s, hs⟩
    -/
    induction' s using Quotient.inductionOn' with s hs
    /-
      case h
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x✝ : Subtype fun s => s.Nodup
      s : List α
      hs : Cycle.Nodup (Quotient.mk'' s)
      ⊢ Exists fun a => Eq ((fun l => ⟨↑↑l, ⋯⟩) a) ⟨Quotient.mk'' s, hs⟩
    -/
    exact ⟨⟨s, hs⟩, by simp⟩
    /-
      🎉 no goals
    -/


instance fintypeNodupNontrivialCycle [Fintype α] :
    Fintype { s : Cycle α // s.Nodup ∧ s.Nontrivial } :=
  Fintype.subtype
    (((Finset.univ : Finset { s : Cycle α // s.Nodup }).map (Function.Embedding.subtype _)).filter
      Cycle.Nontrivial)
        /-
          α : Type u_1
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          ⊢ ∀ (x : Cycle α), Iff (Membership.mem (Finset.filter Cycle.Nontrivial (Finset …
        -/
    (by simp)
        /-
          🎉 no goals
        -/


/-- The `s : Cycle α` as a `Finset α`. -/
def toFinset (s : Cycle α) : Finset α :=
  s.toMultiset.toFinset


@[simp]
theorem toFinset_toMultiset (s : Cycle α) : s.toMultiset.toFinset = s.toFinset :=
  rfl


@[simp]
theorem coe_toFinset (l : List α) : (l : Cycle α).toFinset = l.toFinset :=
  rfl


@[simp]
theorem nil_toFinset : (@nil α).toFinset = ∅ :=
  rfl


@[simp]
theorem toFinset_eq_nil {s : Cycle α} : s.toFinset = ∅ ↔ s = Cycle.nil :=
                              /-
                                α : Type u_1
                                inst✝ : DecidableEq α
                                s : Cycle α
                                ⊢ ∀ (a : List α), Iff (Eq (Cycle.toFinset (Quotient.mk'' a)) EmptyCollection.e …
                              -/
  Quotient.inductionOn' s (by simp)
                              /-
                                🎉 no goals
                              -/


/-- Given a `s : Cycle α` such that `Nodup s`, retrieve the next element after `x ∈ s`. -/
nonrec def next : ∀ (s : Cycle α) (_hs : Nodup s) (x : α) (_hx : x ∈ s), α := fun s =>
  Quot.hrecOn (motive := fun (s : Cycle α) => ∀ (_hs : Cycle.Nodup s) (x : α) (_hx : x ∈ s), α) s
  (fun l _hn x hx => next l x hx) fun l₁ l₂ h =>
    Function.hfunext (propext h.nodup_iff) fun h₁ h₂ _he =>
      Function.hfunext rfl fun x y hxy =>
                                      /-
                                        α : Type u_1
                                        inst✝ : DecidableEq α
                                        s : Cycle α
                                        l₁ l₂ : List α
                                        h : (List.IsRotated.setoid α) l₁ l₂
                                        h₁ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l₁)
                                        h₂ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l₂)
                                        _he : HEq h₁ h₂
                                        x y : α
                                        hxy : HEq x y
                                        ⊢ Iff (Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) l₁) x) (Membership …
                                      -/
        Function.hfunext (propext (by rw [eq_of_heq hxy]; simpa [eq_of_heq hxy] using h.mem_iff))
                                                          /-
                                                            🎉 no goals
                                                          -/
  fun hm hm' he' => heq_of_eq
        /-
          α : Type u_1
          inst✝ : DecidableEq α
          s : Cycle α
          l₁ l₂ : List α
          h : (List.IsRotated.setoid α) l₁ l₂
          h₁ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l₁)
          h₂ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l₂)
          _he : HEq h₁ h₂
          x y : α
          hxy : HEq x y
          hm : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) l₁) x
          hm' : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) l₂) y
          he' : HEq hm hm'
          ⊢ Eq ((fun l _hn x hx => l.next x hx) l₁ h₁ x hm) ((fun l _hn x hx => l.next x …
        -/
    (by rw [heq_iff_eq] at hxy; subst x; simpa using isRotated_next_eq h h₁ _)
                                         /-
                                           🎉 no goals
                                         -/


/-- Given a `s : Cycle α` such that `Nodup s`, retrieve the previous element before `x ∈ s`. -/
nonrec def prev : ∀ (s : Cycle α) (_hs : Nodup s) (x : α) (_hx : x ∈ s), α := fun s =>
  Quot.hrecOn (motive := fun (s : Cycle α) => ∀ (_hs : Cycle.Nodup s) (x : α) (_hx : x ∈ s), α) s
  (fun l _hn x hx => prev l x hx) fun l₁ l₂ h =>
    Function.hfunext (propext h.nodup_iff) fun h₁ h₂ _he =>
      Function.hfunext rfl fun x y hxy =>
                                      /-
                                        α : Type u_1
                                        inst✝ : DecidableEq α
                                        s : Cycle α
                                        l₁ l₂ : List α
                                        h : (List.IsRotated.setoid α) l₁ l₂
                                        h₁ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l₁)
                                        h₂ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l₂)
                                        _he : HEq h₁ h₂
                                        x y : α
                                        hxy : HEq x y
                                        ⊢ Iff (Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) l₁) x) (Membership …
                                      -/
        Function.hfunext (propext (by rw [eq_of_heq hxy]; simpa [eq_of_heq hxy] using h.mem_iff))
                                                          /-
                                                            🎉 no goals
                                                          -/
  fun hm hm' he' => heq_of_eq
        /-
          α : Type u_1
          inst✝ : DecidableEq α
          s : Cycle α
          l₁ l₂ : List α
          h : (List.IsRotated.setoid α) l₁ l₂
          h₁ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l₁)
          h₂ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l₂)
          _he : HEq h₁ h₂
          x y : α
          hxy : HEq x y
          hm : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) l₁) x
          hm' : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) l₂) y
          he' : HEq hm hm'
          ⊢ Eq ((fun l _hn x hx => l.prev x hx) l₁ h₁ x hm) ((fun l _hn x hx => l.prev x …
        -/
    (by rw [heq_iff_eq] at hxy; subst x; simpa using isRotated_prev_eq h h₁ _)
                                         /-
                                           🎉 no goals
                                         -/

-- Porting note: removed `simp` and added `prev_reverse_eq_next'` with `simp` attribute

nonrec theorem prev_reverse_eq_next (s : Cycle α) : ∀ (hs : Nodup s) (x : α) (hx : x ∈ s),
    s.reverse.prev (nodup_reverse_iff.mpr hs) x (mem_reverse_iff.mpr hx) = s.next hs x hx :=
  Quotient.inductionOn' s prev_reverse_eq_next


@[simp]
nonrec theorem prev_reverse_eq_next' (s : Cycle α) (hs : Nodup s.reverse) (x : α)
    (hx : x ∈ s.reverse) :
    s.reverse.prev hs x hx = s.next (nodup_reverse_iff.mp hs) x (mem_reverse_iff.mp hx) :=
  prev_reverse_eq_next s (nodup_reverse_iff.mp hs) x (mem_reverse_iff.mp hx)

-- Porting note: removed `simp` and added `next_reverse_eq_prev'` with `simp` attribute

theorem next_reverse_eq_prev (s : Cycle α) (hs : Nodup s) (x : α) (hx : x ∈ s) :
    s.reverse.next (nodup_reverse_iff.mpr hs) x (mem_reverse_iff.mpr hx) = s.prev hs x hx := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Cycle α
    hs : s.Nodup
    x : α
    hx : Membership.mem s x
    ⊢ Eq (s.reverse.next ⋯ x ⋯) (s.prev hs x hx)
  -/
  simp [← prev_reverse_eq_next]
  /-
    🎉 no goals
  -/


@[simp]
theorem next_reverse_eq_prev' (s : Cycle α) (hs : Nodup s.reverse) (x : α) (hx : x ∈ s.reverse) :
    s.reverse.next hs x hx = s.prev (nodup_reverse_iff.mp hs) x (mem_reverse_iff.mp hx) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Cycle α
    hs : s.reverse.Nodup
    x : α
    hx : Membership.mem s.reverse x
    ⊢ Eq (s.reverse.next hs x hx) (s.prev ⋯ x ⋯)
  -/
  simp [← prev_reverse_eq_next]
  /-
    🎉 no goals
  -/


@[simp]
nonrec theorem next_mem (s : Cycle α) (hs : Nodup s) (x : α) (hx : x ∈ s) : s.next hs x hx ∈ s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Cycle α
    hs : s.Nodup
    x : α
    hx : Membership.mem s x
    ⊢ Membership.mem s (s.next hs x hx)
  -/
  induction s using Quot.inductionOn
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    a✝ : List α
    hs : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) a✝)
    hx : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) a✝) x
    ⊢ Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) a✝) (Cycle.next (Quot.m …
  -/
  apply next_mem; assumption
                  /-
                    🎉 no goals
                  -/


theorem prev_mem (s : Cycle α) (hs : Nodup s) (x : α) (hx : x ∈ s) : s.prev hs x hx ∈ s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Cycle α
    hs : s.Nodup
    x : α
    hx : Membership.mem s x
    ⊢ Membership.mem s (s.prev hs x hx)
  -/
  rw [← next_reverse_eq_prev, ← mem_reverse_iff]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Cycle α
    hs : s.Nodup
    x : α
    hx : Membership.mem s x
    ⊢ Membership.mem s.reverse (s.reverse.next ⋯ x ⋯)
  -/
  apply next_mem
  /-
    🎉 no goals
  -/


@[simp]
nonrec theorem prev_next (s : Cycle α) : ∀ (hs : Nodup s) (x : α) (hx : x ∈ s),
    s.prev hs (s.next hs x hx) (next_mem s hs x hx) = x :=
  Quotient.inductionOn' s prev_next


@[simp]
nonrec theorem next_prev (s : Cycle α) : ∀ (hs : Nodup s) (x : α) (hx : x ∈ s),
    s.next hs (s.prev hs x hx) (prev_mem s hs x hx) = x :=
  Quotient.inductionOn' s next_prev


/-- We define a representation of concrete cycles, available when viewing them in a goal state or
via `#eval`, when over representable types. For example, the cycle `(2 1 4 3)` will be shown
as `c[2, 1, 4, 3]`. Two equal cycles may be printed differently if their internal representation
is different.
-/
unsafe instance [Repr α] : Repr (Cycle α) :=
  ⟨fun s _ => "c[" ++ Std.Format.joinSep (s.map repr).lists.unquot.head! ", " ++ "]"⟩


/-- `chain R s` means that `R` holds between adjacent elements of `s`.

`chain R ([a, b, c] : Cycle α) ↔ R a b ∧ R b c ∧ R c a` -/
nonrec def Chain (r : α → α → Prop) (c : Cycle α) : Prop :=
  Quotient.liftOn' c
    (fun l =>
      match l with
      | [] => True
      | a :: m => Chain r a (m ++ [a]))
    fun a b hab =>
    propext <| by
      /-
        α : Type u_1
        r : α → α → Prop
        c : Cycle α
        a b : List α
        hab : (List.IsRotated.setoid α) a b
        ⊢ Iff ((fun l => Cycle.Chain.match_1 (fun l => Prop) l (fun _ => True) fun a m …
      -/
      cases' a with a l <;> cases' b with b m
        /-
          case nil.nil
          α : Type u_1
          r : α → α → Prop
          c : Cycle α
          hab : (List.IsRotated.setoid α) List.nil List.nil
          ⊢ Iff ((fun l => Cycle.Chain.match_1 (fun l => Prop) l (fun _ => True) fun a m …
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case nil.cons
          α : Type u_1
          r : α → α → Prop
          c : Cycle α
          b : α
          m : List α
          hab : (List.IsRotated.setoid α) List.nil (List.cons b m)
          ⊢ Iff ((fun l => Cycle.Chain.match_1 (fun l => Prop) l (fun _ => True) fun a m …
        -/
      · have := isRotated_nil_iff'.1 hab
        /-
          case nil.cons
          α : Type u_1
          r : α → α → Prop
          c : Cycle α
          b : α
          m : List α
          hab : (List.IsRotated.setoid α) List.nil (List.cons b m)
          this : Eq List.nil (List.cons b m)
          ⊢ Iff ((fun l => Cycle.Chain.match_1 (fun l => Prop) l (fun _ => True) fun a m …
        -/
        contradiction
        /-
          🎉 no goals
        -/
        /-
          case cons.nil
          α : Type u_1
          r : α → α → Prop
          c : Cycle α
          a : α
          l : List α
          hab : (List.IsRotated.setoid α) (List.cons a l) List.nil
          ⊢ Iff ((fun l => Cycle.Chain.match_1 (fun l => Prop) l (fun _ => True) fun a m …
        -/
      · have := isRotated_nil_iff.1 hab
        /-
          case cons.nil
          α : Type u_1
          r : α → α → Prop
          c : Cycle α
          a : α
          l : List α
          hab : (List.IsRotated.setoid α) (List.cons a l) List.nil
          this : Eq (List.cons a l) List.nil
          ⊢ Iff ((fun l => Cycle.Chain.match_1 (fun l => Prop) l (fun _ => True) fun a m …
        -/
        contradiction
        /-
          🎉 no goals
        -/
        /-
          case cons.cons
          α : Type u_1
          r : α → α → Prop
          c : Cycle α
          a : α
          l : List α
          b : α
          m : List α
          hab : (List.IsRotated.setoid α) (List.cons a l) (List.cons b m)
          ⊢ Iff ((fun l => Cycle.Chain.match_1 (fun l => Prop) l (fun _ => True) fun a m …
        -/
      · dsimp only
        /-
          case cons.cons
          α : Type u_1
          r : α → α → Prop
          c : Cycle α
          a : α
          l : List α
          b : α
          m : List α
          hab : (List.IsRotated.setoid α) (List.cons a l) (List.cons b m)
          ⊢ Iff (List.Chain r a (HAppend.hAppend l (List.cons a List.nil))) (List.Chain  …
        -/
        cases' hab with n hn
        /-
          case cons.cons.intro
          α : Type u_1
          r : α → α → Prop
          c : Cycle α
          a : α
          l : List α
          b : α
          m : List α
          n : Nat
          hn : Eq ((List.cons a l).rotate n) (List.cons b m)
          ⊢ Iff (List.Chain r a (HAppend.hAppend l (List.cons a List.nil))) (List.Chain  …
        -/
        induction' n with d hd generalizing a b l m
          /-
            case cons.cons.intro.zero
            α : Type u_1
            r : α → α → Prop
            c : Cycle α
            a : α
            l : List α
            b : α
            m : List α
            hn : Eq ((List.cons a l).rotate 0) (List.cons b m)
            ⊢ Iff (List.Chain r a (HAppend.hAppend l (List.cons a List.nil))) (List.Chain  …
          -/
        · simp only [rotate_zero, cons.injEq] at hn
          /-
            case cons.cons.intro.zero
            α : Type u_1
            r : α → α → Prop
            c : Cycle α
            a : α
            l : List α
            b : α
            m : List α
            hn : And (Eq a b) (Eq l m)
            ⊢ Iff (List.Chain r a (HAppend.hAppend l (List.cons a List.nil))) (List.Chain  …
          -/
          rw [hn.1, hn.2]
          /-
            🎉 no goals
          -/
          /-
            case cons.cons.intro.succ
            α : Type u_1
            r : α → α → Prop
            c : Cycle α
            d : Nat
            hd : ∀ (a : α) (l : List α) (b : α) (m : List α), Eq ((List.cons a l).rotate d …
            a : α
            l : List α
            b : α
            m : List α
            hn : Eq ((List.cons a l).rotate (HAdd.hAdd d 1)) (List.cons b m)
            ⊢ Iff (List.Chain r a (HAppend.hAppend l (List.cons a List.nil))) (List.Chain  …
          -/
        · cases' l with c s
            /-
              case cons.cons.intro.succ.nil
              α : Type u_1
              r : α → α → Prop
              c : Cycle α
              d : Nat
              hd : ∀ (a : α) (l : List α) (b : α) (m : List α), Eq ((List.cons a l).rotate d …
              a b : α
              m : List α
              hn : Eq ((List.cons a List.nil).rotate (HAdd.hAdd d 1)) (List.cons b m)
              ⊢ Iff (List.Chain r a (HAppend.hAppend List.nil (List.cons a List.nil))) (List …
            -/
          · simp only [rotate_cons_succ, nil_append, rotate_singleton, cons.injEq] at hn
            /-
              case cons.cons.intro.succ.nil
              α : Type u_1
              r : α → α → Prop
              c : Cycle α
              d : Nat
              hd : ∀ (a : α) (l : List α) (b : α) (m : List α), Eq ((List.cons a l).rotate d …
              a b : α
              m : List α
              hn : And (Eq a b) (Eq List.nil m)
              ⊢ Iff (List.Chain r a (HAppend.hAppend List.nil (List.cons a List.nil))) (List …
            -/
            rw [hn.1, hn.2]
            /-
              🎉 no goals
            -/
            /-
              case cons.cons.intro.succ.cons
              α : Type u_1
              r : α → α → Prop
              c✝ : Cycle α
              d : Nat
              hd : ∀ (a : α) (l : List α) (b : α) (m : List α), Eq ((List.cons a l).rotate d …
              a b : α
              m : List α
              c : α
              s : List α
              hn : Eq ((List.cons a (List.cons c s)).rotate (HAdd.hAdd d 1)) (List.cons b m)
              ⊢ Iff (List.Chain r a (HAppend.hAppend (List.cons c s) (List.cons a List.nil)) …
            -/
          · rw [Nat.add_comm, ← rotate_rotate, rotate_cons_succ, rotate_zero, cons_append] at hn
            /-
              case cons.cons.intro.succ.cons
              α : Type u_1
              r : α → α → Prop
              c✝ : Cycle α
              d : Nat
              hd : ∀ (a : α) (l : List α) (b : α) (m : List α), Eq ((List.cons a l).rotate d …
              a b : α
              m : List α
              c : α
              s : List α
              hn : Eq ((List.cons c (HAppend.hAppend s (List.cons a List.nil))).rotate d) (L …
              ⊢ Iff (List.Chain r a (HAppend.hAppend (List.cons c s) (List.cons a List.nil)) …
            -/
            rw [← hd c _ _ _ hn]
            /-
              case cons.cons.intro.succ.cons
              α : Type u_1
              r : α → α → Prop
              c✝ : Cycle α
              d : Nat
              hd : ∀ (a : α) (l : List α) (b : α) (m : List α), Eq ((List.cons a l).rotate d …
              a b : α
              m : List α
              c : α
              s : List α
              hn : Eq ((List.cons c (HAppend.hAppend s (List.cons a List.nil))).rotate d) (L …
              ⊢ Iff (List.Chain r a (HAppend.hAppend (List.cons c s) (List.cons a List.nil)) …
            -/
            simp [and_comm]
            /-
              🎉 no goals
            -/


@[simp]
                                                                    /-
                                                                      α : Type u_1
                                                                      r : α → α → Prop
                                                                      ⊢ Cycle.Chain r Cycle.nil
                                                                    -/
theorem Chain.nil (r : α → α → Prop) : Cycle.Chain r (@nil α) := by trivial
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem chain_coe_cons (r : α → α → Prop) (a : α) (l : List α) :
    Chain r (a :: l) ↔ List.Chain r a (l ++ [a]) :=
  Iff.rfl


theorem chain_singleton (r : α → α → Prop) (a : α) : Chain r [a] ↔ r a a := by
  /-
    α : Type u_1
    r : α → α → Prop
    a : α
    ⊢ Iff (Cycle.Chain r ↑(List.cons a List.nil)) (r a a)
  -/
  rw [chain_coe_cons, nil_append, List.chain_singleton]
  /-
    🎉 no goals
  -/


theorem chain_ne_nil (r : α → α → Prop) {l : List α} :
    ∀ hl : l ≠ [], Chain r l ↔ List.Chain r (getLast l hl) l :=
  l.reverseRecOn (fun hm => hm.irrefl.elim) (by
    /-
      α : Type u_1
      r : α → α → Prop
      l : List α
      ⊢ ∀ (l : List α) (a : α), (∀ (hl : Ne l List.nil), Iff (Cycle.Chain r ↑l) (Lis …
    -/
    intro m a _H _
    /-
      α : Type u_1
      r : α → α → Prop
      l m : List α
      a : α
      _H : ∀ (hl : Ne m List.nil), Iff (Cycle.Chain r ↑m) (List.Chain r (m.getLast h …
      hl✝ : Ne (HAppend.hAppend m (List.cons a List.nil)) List.nil
      ⊢ Iff (Cycle.Chain r ↑(HAppend.hAppend m (List.cons a List.nil))) (List.Chain  …
    -/
    rw [← coe_cons_eq_coe_append, chain_coe_cons, getLast_append_singleton])
    /-
      🎉 no goals
    -/


theorem chain_map {β : Type*} {r : α → α → Prop} (f : β → α) {s : Cycle β} :
    Chain r (s.map f) ↔ Chain (fun a b => r (f a) (f b)) s :=
  Quotient.inductionOn s fun l => by
    /-
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      f : β → α
      s : Cycle β
      l : List β
      ⊢ Iff (Cycle.Chain r (Cycle.map f (Quotient.mk (List.IsRotated.setoid β) l)))  …
    -/
    cases' l with a l
      /-
        case nil
        α : Type u_1
        β : Type u_2
        r : α → α → Prop
        f : β → α
        s : Cycle β
        ⊢ Iff (Cycle.Chain r (Cycle.map f (Quotient.mk (List.IsRotated.setoid β) List. …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case cons
        α : Type u_1
        β : Type u_2
        r : α → α → Prop
        f : β → α
        s : Cycle β
        a : β
        l : List β
        ⊢ Iff (Cycle.Chain r (Cycle.map f (Quotient.mk (List.IsRotated.setoid β) (List …
      -/
    · simp [← concat_eq_append, ← List.map_concat, List.chain_map f]
      /-
        🎉 no goals
      -/


nonrec theorem chain_range_succ (r : ℕ → ℕ → Prop) (n : ℕ) :
    Chain r (List.range n.succ) ↔ r n 0 ∧ ∀ m < n, r m m.succ := by
  /-
    r : Nat → Nat → Prop
    n : Nat
    ⊢ Iff (Cycle.Chain r ↑(List.range n.succ)) (And (r n 0) (∀ (m : Nat), LT.lt m  …
  -/
  rw [range_succ, ← coe_cons_eq_coe_append, chain_coe_cons, ← range_succ, chain_range_succ]
  /-
    🎉 no goals
  -/


theorem Chain.imp {r₁ r₂ : α → α → Prop} (H : ∀ a b, r₁ a b → r₂ a b) (p : Chain r₁ s) :
    Chain r₂ s := by
  /-
    α : Type u_1
    s : Cycle α
    r₁ r₂ : α → α → Prop
    H : ∀ (a b : α), r₁ a b → r₂ a b
    p : Cycle.Chain r₁ s
    ⊢ Cycle.Chain r₂ s
  -/
  induction s
    /-
      case H0
      α : Type u_1
      s : Cycle α
      r₁ r₂ : α → α → Prop
      H : ∀ (a b : α), r₁ a b → r₂ a b
      p : Cycle.Chain r₁ Cycle.nil
      ⊢ Cycle.Chain r₂ Cycle.nil
    -/
  · trivial
    /-
      🎉 no goals
    -/
    /-
      case HI
      α : Type u_1
      s : Cycle α
      r₁ r₂ : α → α → Prop
      H : ∀ (a b : α), r₁ a b → r₂ a b
      a✝¹ : α
      l✝ : List α
      a✝ : Cycle.Chain r₁ ↑l✝ → Cycle.Chain r₂ ↑l✝
      p : Cycle.Chain r₁ ↑(List.cons a✝¹ l✝)
      ⊢ Cycle.Chain r₂ ↑(List.cons a✝¹ l✝)
    -/
  · rw [chain_coe_cons] at p ⊢
    /-
      case HI
      α : Type u_1
      s : Cycle α
      r₁ r₂ : α → α → Prop
      H : ∀ (a b : α), r₁ a b → r₂ a b
      a✝¹ : α
      l✝ : List α
      a✝ : Cycle.Chain r₁ ↑l✝ → Cycle.Chain r₂ ↑l✝
      p : List.Chain r₁ a✝¹ (HAppend.hAppend l✝ (List.cons a✝¹ List.nil))
      ⊢ List.Chain r₂ a✝¹ (HAppend.hAppend l✝ (List.cons a✝¹ List.nil))
    -/
    exact p.imp H
    /-
      🎉 no goals
    -/


/-- As a function from a relation to a predicate, `chain` is monotonic. -/
theorem chain_mono : Monotone (Chain : (α → α → Prop) → Cycle α → Prop) := fun _a _b hab _s =>
  Chain.imp hab


theorem chain_of_pairwise : (∀ a ∈ s, ∀ b ∈ s, r a b) → Chain r s := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Cycle α
    ⊢ (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → r a b) → Cy …
  -/
  induction' s with a l _
    /-
      case H0
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      ⊢ (∀ (a : α), Membership.mem Cycle.nil a → ∀ (b : α), Membership.mem Cycle.nil …
    -/
  · exact fun _ => Cycle.Chain.nil r
    /-
      🎉 no goals
    -/
  /-
    case HI
    α : Type u_1
    r : α → α → Prop
    s : Cycle α
    a : α
    l : List α
    a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
    ⊢ (∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membership. …
  -/
  intro hs
  /-
    case HI
    α : Type u_1
    r : α → α → Prop
    s : Cycle α
    a : α
    l : List α
    a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
    hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
    ⊢ Cycle.Chain r ↑(List.cons a l)
  -/
  have Ha : a ∈ (a :: l : Cycle α) := by simp
  /-
    case HI
    α : Type u_1
    r : α → α → Prop
    s : Cycle α
    a : α
    l : List α
    a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
    hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
    Ha : Membership.mem (↑(List.cons a l)) a
    ⊢ Cycle.Chain r ↑(List.cons a l)
  -/
  have Hl : ∀ {b} (_hb : b ∈ l), b ∈ (a :: l : Cycle α) := @fun b hb => by simp [hb]
  /-
    case HI
    α : Type u_1
    r : α → α → Prop
    s : Cycle α
    a : α
    l : List α
    a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
    hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
    Ha : Membership.mem (↑(List.cons a l)) a
    Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
    ⊢ Cycle.Chain r ↑(List.cons a l)
  -/
  rw [Cycle.chain_coe_cons]
  /-
    case HI
    α : Type u_1
    r : α → α → Prop
    s : Cycle α
    a : α
    l : List α
    a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
    hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
    Ha : Membership.mem (↑(List.cons a l)) a
    Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
    ⊢ List.Chain r a (HAppend.hAppend l (List.cons a List.nil))
  -/
  apply Pairwise.chain
  /-
    case HI.p
    α : Type u_1
    r : α → α → Prop
    s : Cycle α
    a : α
    l : List α
    a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
    hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
    Ha : Membership.mem (↑(List.cons a l)) a
    Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
    ⊢ List.Pairwise r (List.cons a (HAppend.hAppend l (List.cons a List.nil)))
  -/
  rw [pairwise_cons]
  refine
    ⟨fun b hb => ?_,
      pairwise_append.2
        ⟨pairwise_of_forall_mem_list fun b hb c hc => hs b (Hl hb) c (Hl hc),
          pairwise_singleton r a, fun b hb c hc => ?_⟩⟩
    /-
      case HI.p.refine_1
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      a : α
      l : List α
      a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
      hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
      Ha : Membership.mem (↑(List.cons a l)) a
      Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
      b : α
      hb : Membership.mem (HAppend.hAppend l (List.cons a List.nil)) b
      ⊢ r a b
    -/
  · rw [mem_append] at hb
    /-
      case HI.p.refine_1
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      a : α
      l : List α
      a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
      hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
      Ha : Membership.mem (↑(List.cons a l)) a
      Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
      b : α
      hb : Or (Membership.mem l b) (Membership.mem (List.cons a List.nil) b)
      ⊢ r a b
    -/
    cases' hb with hb hb
      /-
        case HI.p.refine_1.inl
        α : Type u_1
        r : α → α → Prop
        s : Cycle α
        a : α
        l : List α
        a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
        hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
        Ha : Membership.mem (↑(List.cons a l)) a
        Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
        b : α
        hb : Membership.mem l b
        ⊢ r a b
      -/
    · exact hs a Ha b (Hl hb)
      /-
        🎉 no goals
      -/
      /-
        case HI.p.refine_1.inr
        α : Type u_1
        r : α → α → Prop
        s : Cycle α
        a : α
        l : List α
        a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
        hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
        Ha : Membership.mem (↑(List.cons a l)) a
        Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
        b : α
        hb : Membership.mem (List.cons a List.nil) b
        ⊢ r a b
      -/
    · rw [mem_singleton] at hb
      /-
        case HI.p.refine_1.inr
        α : Type u_1
        r : α → α → Prop
        s : Cycle α
        a : α
        l : List α
        a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
        hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
        Ha : Membership.mem (↑(List.cons a l)) a
        Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
        b : α
        hb : Eq b a
        ⊢ r a b
      -/
      rw [hb]
      /-
        case HI.p.refine_1.inr
        α : Type u_1
        r : α → α → Prop
        s : Cycle α
        a : α
        l : List α
        a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
        hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
        Ha : Membership.mem (↑(List.cons a l)) a
        Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
        b : α
        hb : Eq b a
        ⊢ r a a
      -/
      exact hs a Ha a Ha
      /-
        🎉 no goals
      -/
    /-
      case HI.p.refine_2
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      a : α
      l : List α
      a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
      hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
      Ha : Membership.mem (↑(List.cons a l)) a
      Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
      b : α
      hb : Membership.mem l b
      c : α
      hc : Membership.mem (List.cons a List.nil) c
      ⊢ r b c
    -/
  · rw [mem_singleton] at hc
    /-
      case HI.p.refine_2
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      a : α
      l : List α
      a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
      hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
      Ha : Membership.mem (↑(List.cons a l)) a
      Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
      b : α
      hb : Membership.mem l b
      c : α
      hc : Eq c a
      ⊢ r b c
    -/
    rw [hc]
    /-
      case HI.p.refine_2
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      a : α
      l : List α
      a✝ : (∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membership.mem (↑l) b → r  …
      hs : ∀ (a_1 : α), Membership.mem (↑(List.cons a l)) a_1 → ∀ (b : α), Membershi …
      Ha : Membership.mem (↑(List.cons a l)) a
      Hl : ∀ {b : α}, Membership.mem l b → Membership.mem (↑(List.cons a l)) b
      b : α
      hb : Membership.mem l b
      c : α
      hc : Eq c a
      ⊢ r b a
    -/
    exact hs b (Hl hb) a Ha
    /-
      🎉 no goals
    -/


theorem chain_iff_pairwise [IsTrans α r] : Chain r s ↔ ∀ a ∈ s, ∀ b ∈ s, r a b :=
  ⟨by
    /-
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      inst✝ : IsTrans α r
      ⊢ Cycle.Chain r s → ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem  …
    -/
    induction' s with a l _
      /-
        case H0
        α : Type u_1
        r : α → α → Prop
        s : Cycle α
        inst✝ : IsTrans α r
        ⊢ Cycle.Chain r Cycle.nil → ∀ (a : α), Membership.mem Cycle.nil a → ∀ (b : α), …
      -/
    · exact fun _ b hb => (not_mem_nil _ hb).elim
      /-
        🎉 no goals
      -/
    /-
      case HI
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      inst✝ : IsTrans α r
      a : α
      l : List α
      a✝ : Cycle.Chain r ↑l → ∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membersh …
      ⊢ Cycle.Chain r ↑(List.cons a l) → ∀ (a_2 : α), Membership.mem (↑(List.cons a  …
    -/
    intro hs b hb c hc
    /-
      case HI
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      inst✝ : IsTrans α r
      a : α
      l : List α
      a✝ : Cycle.Chain r ↑l → ∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membersh …
      hs : Cycle.Chain r ↑(List.cons a l)
      b : α
      hb : Membership.mem (↑(List.cons a l)) b
      c : α
      hc : Membership.mem (↑(List.cons a l)) c
      ⊢ r b c
    -/
    rw [Cycle.chain_coe_cons, List.chain_iff_pairwise] at hs
    simp only [pairwise_append, pairwise_cons, mem_append, mem_singleton, List.not_mem_nil,
      IsEmpty.forall_iff, imp_true_iff, Pairwise.nil, forall_eq, true_and] at hs
    /-
      case HI
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      inst✝ : IsTrans α r
      a : α
      l : List α
      a✝ : Cycle.Chain r ↑l → ∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membersh …
      b : α
      hb : Membership.mem (↑(List.cons a l)) b
      c : α
      hc : Membership.mem (↑(List.cons a l)) c
      hs : And (∀ (a' : α), Or (Membership.mem l a') (Eq a' a) → r a a') (And (List. …
      ⊢ r b c
    -/
    simp only [mem_coe_iff, mem_cons] at hb hc
    /-
      case HI
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      inst✝ : IsTrans α r
      a : α
      l : List α
      a✝ : Cycle.Chain r ↑l → ∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membersh …
      b c : α
      hs : And (∀ (a' : α), Or (Membership.mem l a') (Eq a' a) → r a a') (And (List. …
      hb : Or (Eq b a) (Membership.mem l b)
      hc : Or (Eq c a) (Membership.mem l c)
      ⊢ r b c
    -/
    rcases hb with (rfl | hb) <;> rcases hc with (rfl | hc)
      /-
        case HI.inl.inl
        α : Type u_1
        r : α → α → Prop
        s : Cycle α
        inst✝ : IsTrans α r
        l : List α
        a✝ : Cycle.Chain r ↑l → ∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membersh …
        c : α
        hs : And (∀ (a' : α), Or (Membership.mem l a') (Eq a' c) → r c a') (And (List. …
        ⊢ r c c
      -/
    · exact hs.1 c (Or.inr rfl)
      /-
        🎉 no goals
      -/
      /-
        case HI.inl.inr
        α : Type u_1
        r : α → α → Prop
        s : Cycle α
        inst✝ : IsTrans α r
        l : List α
        a✝ : Cycle.Chain r ↑l → ∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membersh …
        b c : α
        hs : And (∀ (a' : α), Or (Membership.mem l a') (Eq a' b) → r b a') (And (List. …
        hc : Membership.mem l c
        ⊢ r b c
      -/
    · exact hs.1 c (Or.inl hc)
      /-
        🎉 no goals
      -/
      /-
        case HI.inr.inl
        α : Type u_1
        r : α → α → Prop
        s : Cycle α
        inst✝ : IsTrans α r
        l : List α
        a✝ : Cycle.Chain r ↑l → ∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membersh …
        b c : α
        hb : Membership.mem l b
        hs : And (∀ (a' : α), Or (Membership.mem l a') (Eq a' c) → r c a') (And (List. …
        ⊢ r b c
      -/
    · exact hs.2.2 b hb
      /-
        🎉 no goals
      -/
      /-
        case HI.inr.inr
        α : Type u_1
        r : α → α → Prop
        s : Cycle α
        inst✝ : IsTrans α r
        a : α
        l : List α
        a✝ : Cycle.Chain r ↑l → ∀ (a : α), Membership.mem (↑l) a → ∀ (b : α), Membersh …
        b c : α
        hs : And (∀ (a' : α), Or (Membership.mem l a') (Eq a' a) → r a a') (And (List. …
        hb : Membership.mem l b
        hc : Membership.mem l c
        ⊢ r b c
      -/
    · exact _root_.trans (hs.2.2 b hb) (hs.1 c (Or.inl hc)), Cycle.chain_of_pairwise⟩
      /-
        🎉 no goals
      -/


theorem Chain.eq_nil_of_irrefl [IsTrans α r] [IsIrrefl α r] (h : Chain r s) : s = Cycle.nil := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Cycle α
    inst✝¹ : IsTrans α r
    inst✝ : IsIrrefl α r
    h : Cycle.Chain r s
    ⊢ Eq s Cycle.nil
  -/
  induction' s with a l _ h
    /-
      case H0
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      inst✝¹ : IsTrans α r
      inst✝ : IsIrrefl α r
      h : Cycle.Chain r Cycle.nil
      ⊢ Eq Cycle.nil Cycle.nil
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case HI
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      inst✝¹ : IsTrans α r
      inst✝ : IsIrrefl α r
      a : α
      l : List α
      a✝ : Cycle.Chain r ↑l → Eq (↑l) Cycle.nil
      h : Cycle.Chain r ↑(List.cons a l)
      ⊢ Eq (↑(List.cons a l)) Cycle.nil
    -/
  · have ha := mem_cons_self a l
    /-
      case HI
      α : Type u_1
      r : α → α → Prop
      s : Cycle α
      inst✝¹ : IsTrans α r
      inst✝ : IsIrrefl α r
      a : α
      l : List α
      a✝ : Cycle.Chain r ↑l → Eq (↑l) Cycle.nil
      h : Cycle.Chain r ↑(List.cons a l)
      ha : Membership.mem (List.cons a l) a
      ⊢ Eq (↑(List.cons a l)) Cycle.nil
    -/
    exact (irrefl_of r a <| chain_iff_pairwise.1 h a ha a ha).elim
    /-
      🎉 no goals
    -/


theorem Chain.eq_nil_of_well_founded [IsWellFounded α r] (h : Chain r s) : s = Cycle.nil :=
  Chain.eq_nil_of_irrefl <| h.imp fun _ _ => Relation.TransGen.single


theorem forall_eq_of_chain [IsTrans α r] [IsAntisymm α r] (hs : Chain r s) {a b : α} (ha : a ∈ s)
    (hb : b ∈ s) : a = b := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Cycle α
    inst✝¹ : IsTrans α r
    inst✝ : IsAntisymm α r
    hs : Cycle.Chain r s
    a b : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    ⊢ Eq a b
  -/
  rw [chain_iff_pairwise] at hs
  /-
    α : Type u_1
    r : α → α → Prop
    s : Cycle α
    inst✝¹ : IsTrans α r
    inst✝ : IsAntisymm α r
    hs : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → r a b
    a b : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    ⊢ Eq a b
  -/
  exact antisymm (hs a ha b hb) (hs b hb a ha)
  /-
    🎉 no goals
  -/


