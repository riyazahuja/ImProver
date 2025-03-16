theorem permutationsAux2_fst (t : α) (ts : List α) (r : List β) :
    ∀ (ys : List α) (f : List α → β), (permutationsAux2 t ts r ys f).1 = ys ++ ts
  | [], _ => rfl
                     /-
                       α : Type u_1
                       β : Type u_2
                       t : α
                       ts : List α
                       r : List β
                       y : α
                       ys : List α
                       f : List α → β
                       ⊢ Eq (List.permutationsAux2 t ts r (List.cons y ys) f).1 (HAppend.hAppend (Lis …
                     -/
  | y :: ys, f => by simp [permutationsAux2, permutationsAux2_fst t _ _ ys]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem permutationsAux2_snd_nil (t : α) (ts : List α) (r : List β) (f : List α → β) :
    (permutationsAux2 t ts r [] f).2 = r :=
  rfl


@[simp]
theorem permutationsAux2_snd_cons (t : α) (ts : List α) (r : List β) (y : α) (ys : List α)
    (f : List α → β) :
    (permutationsAux2 t ts r (y :: ys) f).2 =
      f (t :: y :: ys ++ ts) :: (permutationsAux2 t ts r ys fun x : List α => f (y :: x)).2 := by
  /-
    α : Type u_1
    β : Type u_2
    t : α
    ts : List α
    r : List β
    y : α
    ys : List α
    f : List α → β
    ⊢ Eq (List.permutationsAux2 t ts r (List.cons y ys) f).2 (List.cons (f (HAppen …
  -/
  simp [permutationsAux2, permutationsAux2_fst t _ _ ys]
  /-
    🎉 no goals
  -/


/-- The `r` argument to `permutationsAux2` is the same as appending. -/
theorem permutationsAux2_append (t : α) (ts : List α) (r : List β) (ys : List α) (f : List α → β) :
    (permutationsAux2 t ts nil ys f).2 ++ r = (permutationsAux2 t ts r ys f).2 := by
  /-
    α : Type u_1
    β : Type u_2
    t : α
    ts : List α
    r : List β
    ys : List α
    f : List α → β
    ⊢ Eq (HAppend.hAppend (List.permutationsAux2 t ts List.nil ys f).2 r) (List.pe …
  -/
                                  /-
                                    🎉 no goals
                                  -/
  induction ys generalizing f <;> simp [*]
                                  /-
                                    🎉 no goals
                                  -/


/-- The `ts` argument to `permutationsAux2` can be folded into the `f` argument. -/
theorem permutationsAux2_comp_append {t : α} {ts ys : List α} {r : List β} (f : List α → β) :
    ((permutationsAux2 t [] r ys) fun x => f (x ++ ts)).2 = (permutationsAux2 t ts r ys f).2 := by
  /-
    α : Type u_1
    β : Type u_2
    t : α
    ts ys : List α
    r : List β
    f : List α → β
    ⊢ Eq (List.permutationsAux2 t List.nil r ys fun x => f (HAppend.hAppend x ts)) …
  -/
  induction' ys with ys_hd _ ys_ih generalizing f
    /-
      case nil
      α : Type u_1
      β : Type u_2
      t : α
      ts : List α
      r : List β
      f : List α → β
      ⊢ Eq (List.permutationsAux2 t List.nil r List.nil fun x => f (HAppend.hAppend  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      β : Type u_2
      t : α
      ts : List α
      r : List β
      ys_hd : α
      tail✝ : List α
      ys_ih : ∀ (f : List α → β), Eq (List.permutationsAux2 t List.nil r tail✝ fun x …
      f : List α → β
      ⊢ Eq (List.permutationsAux2 t List.nil r (List.cons ys_hd tail✝) fun x => f (H …
    -/
  · simp [ys_ih fun xs => f (ys_hd :: xs)]
    /-
      🎉 no goals
    -/


theorem map_permutationsAux2' {α' β'} (g : α → α') (g' : β → β') (t : α) (ts ys : List α)
    (r : List β) (f : List α → β) (f' : List α' → β') (H : ∀ a, g' (f a) = f' (map g a)) :
    map g' (permutationsAux2 t ts r ys f).2 =
      (permutationsAux2 (g t) (map g ts) (map g' r) (map g ys) f').2 := by
  /-
    α : Type u_1
    β : Type u_2
    α' : Type u_3
    β' : Type u_4
    g : α → α'
    g' : β → β'
    t : α
    ts ys : List α
    r : List β
    f : List α → β
    f' : List α' → β'
    H : ∀ (a : List α), Eq (g' (f a)) (f' (List.map g a))
    ⊢ Eq (List.map g' (List.permutationsAux2 t ts r ys f).2) (List.permutationsAux …
  -/
  induction' ys with ys_hd _ ys_ih generalizing f f'
    /-
      case nil
      α : Type u_1
      β : Type u_2
      α' : Type u_3
      β' : Type u_4
      g : α → α'
      g' : β → β'
      t : α
      ts : List α
      r : List β
      f : List α → β
      f' : List α' → β'
      H : ∀ (a : List α), Eq (g' (f a)) (f' (List.map g a))
      ⊢ Eq (List.map g' (List.permutationsAux2 t ts r List.nil f).2) (List.permutati …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      β : Type u_2
      α' : Type u_3
      β' : Type u_4
      g : α → α'
      g' : β → β'
      t : α
      ts : List α
      r : List β
      ys_hd : α
      tail✝ : List α
      ys_ih : ∀ (f : List α → β) (f' : List α' → β'), (∀ (a : List α), Eq (g' (f a)) …
      f : List α → β
      f' : List α' → β'
      H : ∀ (a : List α), Eq (g' (f a)) (f' (List.map g a))
      ⊢ Eq (List.map g' (List.permutationsAux2 t ts r (List.cons ys_hd tail✝) f).2)  …
    -/
  · simp only [map, permutationsAux2_snd_cons, cons_append, cons.injEq]
    /-
      case cons
      α : Type u_1
      β : Type u_2
      α' : Type u_3
      β' : Type u_4
      g : α → α'
      g' : β → β'
      t : α
      ts : List α
      r : List β
      ys_hd : α
      tail✝ : List α
      ys_ih : ∀ (f : List α → β) (f' : List α' → β'), (∀ (a : List α), Eq (g' (f a)) …
      f : List α → β
      f' : List α' → β'
      H : ∀ (a : List α), Eq (g' (f a)) (f' (List.map g a))
      ⊢ And (Eq (g' (f (List.cons t (List.cons ys_hd (List.permutationsAux2 t ts r t …
    -/
    rw [ys_ih, permutationsAux2_fst]
      /-
        case cons
        α : Type u_1
        β : Type u_2
        α' : Type u_3
        β' : Type u_4
        g : α → α'
        g' : β → β'
        t : α
        ts : List α
        r : List β
        ys_hd : α
        tail✝ : List α
        ys_ih : ∀ (f : List α → β) (f' : List α' → β'), (∀ (a : List α), Eq (g' (f a)) …
        f : List α → β
        f' : List α' → β'
        H : ∀ (a : List α), Eq (g' (f a)) (f' (List.map g a))
        ⊢ And (Eq (g' (f (List.cons t (List.cons ys_hd (HAppend.hAppend tail✝ ts)))))  …
      -/
    · refine ⟨?_, rfl⟩
      /-
        case cons
        α : Type u_1
        β : Type u_2
        α' : Type u_3
        β' : Type u_4
        g : α → α'
        g' : β → β'
        t : α
        ts : List α
        r : List β
        ys_hd : α
        tail✝ : List α
        ys_ih : ∀ (f : List α → β) (f' : List α' → β'), (∀ (a : List α), Eq (g' (f a)) …
        f : List α → β
        f' : List α' → β'
        H : ∀ (a : List α), Eq (g' (f a)) (f' (List.map g a))
        ⊢ Eq (g' (f (List.cons t (List.cons ys_hd (HAppend.hAppend tail✝ ts))))) (f' ( …
      -/
      simp only [← map_cons, ← map_append]; apply H
                                            /-
                                              🎉 no goals
                                            -/
      /-
        case cons.H
        α : Type u_1
        β : Type u_2
        α' : Type u_3
        β' : Type u_4
        g : α → α'
        g' : β → β'
        t : α
        ts : List α
        r : List β
        ys_hd : α
        tail✝ : List α
        ys_ih : ∀ (f : List α → β) (f' : List α' → β'), (∀ (a : List α), Eq (g' (f a)) …
        f : List α → β
        f' : List α' → β'
        H : ∀ (a : List α), Eq (g' (f a)) (f' (List.map g a))
        ⊢ ∀ (a : List α), Eq (g' (f (List.cons ys_hd a))) (f' (List.cons (g ys_hd) (Li …
      -/
    · intro a; apply H
               /-
                 🎉 no goals
               -/


/-- The `f` argument to `permutationsAux2` when `r = []` can be eliminated. -/
theorem map_permutationsAux2 (t : α) (ts : List α) (ys : List α) (f : List α → β) :
    (permutationsAux2 t ts [] ys id).2.map f = (permutationsAux2 t ts [] ys f).2 := by
  /-
    α : Type u_1
    β : Type u_2
    t : α
    ts ys : List α
    f : List α → β
    ⊢ Eq (List.map f (List.permutationsAux2 t ts List.nil ys id).2) (List.permutat …
  -/
  rw [map_permutationsAux2' id, map_id, map_id]
    /-
      α : Type u_1
      β : Type u_2
      t : α
      ts ys : List α
      f : List α → β
      ⊢ Eq (List.permutationsAux2 (id t) ts (List.map f List.nil) ys ?f').2 (List.pe …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case H
    α : Type u_1
    β : Type u_2
    t : α
    ts ys : List α
    f : List α → β
    ⊢ ∀ (a : List α), Eq (f (id a)) (f (List.map id a))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An expository lemma to show how all of `ts`, `r`, and `f` can be eliminated from
`permutationsAux2`.

`(permutationsAux2 t [] [] ys id).2`, which appears on the RHS, is a list whose elements are
produced by inserting `t` into every non-terminal position of `ys` in order. As an example:
```lean
#eval permutationsAux2 1 [] [] [2, 3, 4] id
-- [[1, 2, 3, 4], [2, 1, 3, 4], [2, 3, 1, 4]]
```
-/
theorem permutationsAux2_snd_eq (t : α) (ts : List α) (r : List β) (ys : List α) (f : List α → β) :
    (permutationsAux2 t ts r ys f).2 =
      ((permutationsAux2 t [] [] ys id).2.map fun x => f (x ++ ts)) ++ r := by
  /-
    α : Type u_1
    β : Type u_2
    t : α
    ts : List α
    r : List β
    ys : List α
    f : List α → β
    ⊢ Eq (List.permutationsAux2 t ts r ys f).2 (HAppend.hAppend (List.map (fun x = …
  -/
  rw [← permutationsAux2_append, map_permutationsAux2, permutationsAux2_comp_append]
  /-
    🎉 no goals
  -/


theorem map_map_permutationsAux2 {α'} (g : α → α') (t : α) (ts ys : List α) :
    map (map g) (permutationsAux2 t ts [] ys id).2 =
      (permutationsAux2 (g t) (map g ts) [] (map g ys) id).2 :=
  map_permutationsAux2' _ _ _ _ _ _ _ _ fun _ => rfl


theorem map_map_permutations'Aux (f : α → β) (t : α) (ts : List α) :
    map (map f) (permutations'Aux t ts) = permutations'Aux (f t) (map f ts) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    t : α
    ts : List α
    ⊢ Eq (List.map (List.map f) (List.permutations'Aux t ts)) (List.permutations'A …
  -/
  induction' ts with a ts ih
    /-
      case nil
      α : Type u_1
      β : Type u_2
      f : α → β
      t : α
      ⊢ Eq (List.map (List.map f) (List.permutations'Aux t List.nil)) (List.permutat …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      β : Type u_2
      f : α → β
      t a : α
      ts : List α
      ih : Eq (List.map (List.map f) (List.permutations'Aux t ts)) (List.permutation …
      ⊢ Eq (List.map (List.map f) (List.permutations'Aux t (List.cons a ts))) (List. …
    -/
  · simp only [permutations'Aux, map_cons, map_map, ← ih, cons.injEq, true_and, Function.comp_def]
    /-
      🎉 no goals
    -/


theorem permutations'Aux_eq_permutationsAux2 (t : α) (ts : List α) :
    permutations'Aux t ts = (permutationsAux2 t [] [ts ++ [t]] ts id).2 := by
  /-
    α : Type u_1
    t : α
    ts : List α
    ⊢ Eq (List.permutations'Aux t ts) (List.permutationsAux2 t List.nil (List.cons …
  -/
  induction' ts with a ts ih; · rfl
                                /-
                                  🎉 no goals
                                -/
  simp only [permutations'Aux, ih, cons_append, permutationsAux2_snd_cons, append_nil, id_eq,
    cons.injEq, true_and]
  /-
    case cons
    α : Type u_1
    t a : α
    ts : List α
    ih : Eq (List.permutations'Aux t ts) (List.permutationsAux2 t List.nil (List.c …
    ⊢ Eq (List.map (List.cons a) (List.permutationsAux2 t List.nil (List.cons (HAp …
  -/
  simp (config := { singlePass := true }) only [← permutationsAux2_append]
  /-
    case cons
    α : Type u_1
    t a : α
    ts : List α
    ih : Eq (List.permutations'Aux t ts) (List.permutationsAux2 t List.nil (List.c …
    ⊢ Eq (List.map (List.cons a) (HAppend.hAppend (List.permutationsAux2 t List.ni …
  -/
  simp [map_permutationsAux2]
  /-
    🎉 no goals
  -/


theorem mem_permutationsAux2 {t : α} {ts : List α} {ys : List α} {l l' : List α} :
    l' ∈ (permutationsAux2 t ts [] ys (l ++ ·)).2 ↔
      ∃ l₁ l₂, l₂ ≠ [] ∧ ys = l₁ ++ l₂ ∧ l' = l ++ l₁ ++ t :: l₂ ++ ts := by
  /-
    α : Type u_1
    t : α
    ts ys l l' : List α
    ⊢ Iff (Membership.mem (List.permutationsAux2 t ts List.nil ys fun x => HAppend …
  -/
  induction' ys with y ys ih generalizing l
    /-
      case nil
      α : Type u_1
      t : α
      ts l' l : List α
      ⊢ Iff (Membership.mem (List.permutationsAux2 t ts List.nil List.nil fun x => H …
    -/
  · simp +contextual
    /-
      🎉 no goals
    -/
  rw [permutationsAux2_snd_cons,
    show (fun x : List α => l ++ y :: x) = (l ++ [y] ++ ·) by funext _; simp, mem_cons, ih]
  /-
    case cons
    α : Type u_1
    t : α
    ts l' : List α
    y : α
    ys : List α
    ih : ∀ {l : List α}, Iff (Membership.mem (List.permutationsAux2 t ts List.nil  …
    l : List α
    ⊢ Iff (Or (Eq l' (HAppend.hAppend l (HAppend.hAppend (List.cons t (List.cons y …
  -/
  constructor
    /-
      case cons.mp
      α : Type u_1
      t : α
      ts l' : List α
      y : α
      ys : List α
      ih : ∀ {l : List α}, Iff (Membership.mem (List.permutationsAux2 t ts List.nil  …
      l : List α
      ⊢ Or (Eq l' (HAppend.hAppend l (HAppend.hAppend (List.cons t (List.cons y ys)) …
    -/
  · rintro (rfl | ⟨l₁, l₂, l0, rfl, rfl⟩)
      /-
        case cons.mp.inl
        α : Type u_1
        t : α
        ts : List α
        y : α
        ys l : List α
        ih : ∀ {l_1 : List α}, Iff (Membership.mem (List.permutationsAux2 t ts List.ni …
        ⊢ Exists fun l₁ => Exists fun l₂ => And (Ne l₂ List.nil) (And (Eq (List.cons y …
      -/
    · exact ⟨[], y :: ys, by simp⟩
      /-
        🎉 no goals
      -/
      /-
        case cons.mp.inr.intro.intro.intro.intro
        α : Type u_1
        t : α
        ts : List α
        y : α
        l l₁ l₂ : List α
        l0 : Ne l₂ List.nil
        ih : ∀ {l_1 : List α}, Iff (Membership.mem (List.permutationsAux2 t ts List.ni …
        ⊢ Exists fun l₁_1 => Exists fun l₂_1 => And (Ne l₂_1 List.nil) (And (Eq (List. …
      -/
    · exact ⟨y :: l₁, l₂, l0, by simp⟩
      /-
        🎉 no goals
      -/
    /-
      case cons.mpr
      α : Type u_1
      t : α
      ts l' : List α
      y : α
      ys : List α
      ih : ∀ {l : List α}, Iff (Membership.mem (List.permutationsAux2 t ts List.nil  …
      l : List α
      ⊢ (Exists fun l₁ => Exists fun l₂ => And (Ne l₂ List.nil) (And (Eq (List.cons  …
    -/
  · rintro ⟨_ | ⟨y', l₁⟩, l₂, l0, ye, rfl⟩
      /-
        case cons.mpr.intro.nil.intro.intro.intro
        α : Type u_1
        t : α
        ts : List α
        y : α
        ys l l₂ : List α
        l0 : Ne l₂ List.nil
        ye : Eq (List.cons y ys) (HAppend.hAppend List.nil l₂)
        ih : ∀ {l_1 : List α}, Iff (Membership.mem (List.permutationsAux2 t ts List.ni …
        ⊢ Or (Eq (HAppend.hAppend (HAppend.hAppend (HAppend.hAppend l List.nil) (List. …
      -/
    · simp [ye]
      /-
        🎉 no goals
      -/
      /-
        case cons.mpr.intro.cons.intro.intro.intro
        α : Type u_1
        t : α
        ts : List α
        y : α
        ys l : List α
        y' : α
        l₁ l₂ : List α
        l0 : Ne l₂ List.nil
        ye : Eq (List.cons y ys) (HAppend.hAppend (List.cons y' l₁) l₂)
        ih : ∀ {l_1 : List α}, Iff (Membership.mem (List.permutationsAux2 t ts List.ni …
        ⊢ Or (Eq (HAppend.hAppend (HAppend.hAppend (HAppend.hAppend l (List.cons y' l₁ …
      -/
    · simp only [cons_append] at ye
      /-
        case cons.mpr.intro.cons.intro.intro.intro
        α : Type u_1
        t : α
        ts : List α
        y : α
        ys l : List α
        y' : α
        l₁ l₂ : List α
        l0 : Ne l₂ List.nil
        ye : Eq (List.cons y ys) (List.cons y' (HAppend.hAppend l₁ l₂))
        ih : ∀ {l_1 : List α}, Iff (Membership.mem (List.permutationsAux2 t ts List.ni …
        ⊢ Or (Eq (HAppend.hAppend (HAppend.hAppend (HAppend.hAppend l (List.cons y' l₁ …
      -/
      rcases ye with ⟨rfl, rfl⟩
      /-
        case cons.mpr.intro.cons.intro.intro.intro.refl
        α : Type u_1
        t : α
        ts : List α
        y : α
        l l₁ l₂ : List α
        l0 : Ne l₂ List.nil
        ih : ∀ {l_1 : List α}, Iff (Membership.mem (List.permutationsAux2 t ts List.ni …
        ⊢ Or (Eq (HAppend.hAppend (HAppend.hAppend (HAppend.hAppend l (List.cons y l₁) …
      -/
      exact Or.inr ⟨l₁, l₂, l0, by simp⟩
      /-
        🎉 no goals
      -/


theorem mem_permutationsAux2' {t : α} {ts : List α} {ys : List α} {l : List α} :
    l ∈ (permutationsAux2 t ts [] ys id).2 ↔
      ∃ l₁ l₂, l₂ ≠ [] ∧ ys = l₁ ++ l₂ ∧ l = l₁ ++ t :: l₂ ++ ts := by
  /-
    α : Type u_1
    t : α
    ts ys l : List α
    ⊢ Iff (Membership.mem (List.permutationsAux2 t ts List.nil ys id).2 l) (Exists …
  -/
  rw [show @id (List α) = ([] ++ ·) by funext _; rfl]; apply mem_permutationsAux2
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem length_permutationsAux2 (t : α) (ts : List α) (ys : List α) (f : List α → β) :
    length (permutationsAux2 t ts [] ys f).2 = length ys := by
  /-
    α : Type u_1
    β : Type u_2
    t : α
    ts ys : List α
    f : List α → β
    ⊢ Eq (List.permutationsAux2 t ts List.nil ys f).2.length ys.length
  -/
                                  /-
                                    🎉 no goals
                                  -/
  induction ys generalizing f <;> simp [*]
                                  /-
                                    🎉 no goals
                                  -/


theorem foldr_permutationsAux2 (t : α) (ts : List α) (r L : List (List α)) :
    foldr (fun y r => (permutationsAux2 t ts r y id).2) r L =
      (L.flatMap fun y => (permutationsAux2 t ts [] y id).2) ++ r := by
  /-
    α : Type u_1
    t : α
    ts : List α
    r L : List (List α)
    ⊢ Eq (List.foldr (fun y r => (List.permutationsAux2 t ts r y id).2) r L) (HApp …
  -/
  induction' L with l L ih
    /-
      case nil
      α : Type u_1
      t : α
      ts : List α
      r : List (List α)
      ⊢ Eq (List.foldr (fun y r => (List.permutationsAux2 t ts r y id).2) r List.nil …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      t : α
      ts : List α
      r : List (List α)
      l : List α
      L : List (List α)
      ih : Eq (List.foldr (fun y r => (List.permutationsAux2 t ts r y id).2) r L) (H …
      ⊢ Eq (List.foldr (fun y r => (List.permutationsAux2 t ts r y id).2) r (List.co …
    -/
  · simp_rw [foldr_cons, ih, flatMap_cons, append_assoc, permutationsAux2_append]
    /-
      🎉 no goals
    -/


theorem mem_foldr_permutationsAux2 {t : α} {ts : List α} {r L : List (List α)} {l' : List α} :
    l' ∈ foldr (fun y r => (permutationsAux2 t ts r y id).2) r L ↔
      l' ∈ r ∨ ∃ l₁ l₂, l₁ ++ l₂ ∈ L ∧ l₂ ≠ [] ∧ l' = l₁ ++ t :: l₂ ++ ts := by
  have :
    (∃ a : List α,
        a ∈ L ∧ ∃ l₁ l₂ : List α, ¬l₂ = nil ∧ a = l₁ ++ l₂ ∧ l' = l₁ ++ t :: (l₂ ++ ts)) ↔
      ∃ l₁ l₂ : List α, ¬l₂ = nil ∧ l₁ ++ l₂ ∈ L ∧ l' = l₁ ++ t :: (l₂ ++ ts) :=
    ⟨fun ⟨_, aL, l₁, l₂, l0, e, h⟩ => ⟨l₁, l₂, l0, e ▸ aL, h⟩, fun ⟨l₁, l₂, l0, aL, h⟩ =>
      ⟨_, aL, l₁, l₂, l0, rfl, h⟩⟩
  /-
    α : Type u_1
    t : α
    ts : List α
    r L : List (List α)
    l' : List α
    this : Iff (Exists fun a => And (Membership.mem L a) (Exists fun l₁ => Exists  …
    ⊢ Iff (Membership.mem (List.foldr (fun y r => (List.permutationsAux2 t ts r y  …
  -/
  rw [foldr_permutationsAux2]
  simp only [mem_permutationsAux2', ← this, or_comm, and_left_comm, mem_append, mem_flatMap,
    append_assoc, cons_append, exists_prop]


theorem length_foldr_permutationsAux2 (t : α) (ts : List α) (r L : List (List α)) :
    length (foldr (fun y r => (permutationsAux2 t ts r y id).2) r L) =
      (map length L).sum + length r := by
  /-
    α : Type u_1
    t : α
    ts : List α
    r L : List (List α)
    ⊢ Eq (List.foldr (fun y r => (List.permutationsAux2 t ts r y id).2) r L).lengt …
  -/
  simp [foldr_permutationsAux2, Function.comp_def, length_permutationsAux2, length_flatMap]
  /-
    🎉 no goals
  -/


theorem length_foldr_permutationsAux2' (t : α) (ts : List α) (r L : List (List α)) (n)
    (H : ∀ l ∈ L, length l = n) :
    length (foldr (fun y r => (permutationsAux2 t ts r y id).2) r L) = n * length L + length r := by
  /-
    α : Type u_1
    t : α
    ts : List α
    r L : List (List α)
    n : Nat
    H : ∀ (l : List α), Membership.mem L l → Eq l.length n
    ⊢ Eq (List.foldr (fun y r => (List.permutationsAux2 t ts r y id).2) r L).lengt …
  -/
  rw [length_foldr_permutationsAux2, (_ : (map length L).sum = n * length L)]
  /-
    α : Type u_1
    t : α
    ts : List α
    r L : List (List α)
    n : Nat
    H : ∀ (l : List α), Membership.mem L l → Eq l.length n
    ⊢ Eq (List.map List.length L).sum (HMul.hMul n L.length)
  -/
  induction' L with l L ih
    /-
      case nil
      α : Type u_1
      t : α
      ts : List α
      r : List (List α)
      n : Nat
      H : ∀ (l : List α), Membership.mem List.nil l → Eq l.length n
      ⊢ Eq (List.map List.length List.nil).sum (HMul.hMul n List.nil.length)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    t : α
    ts : List α
    r : List (List α)
    n : Nat
    l : List α
    L : List (List α)
    ih : (∀ (l : List α), Membership.mem L l → Eq l.length n) → Eq (List.map List. …
    H : ∀ (l_1 : List α), Membership.mem (List.cons l L) l_1 → Eq l_1.length n
    ⊢ Eq (List.map List.length (List.cons l L)).sum (HMul.hMul n (List.cons l L).l …
  -/
  have sum_map : (map length L).sum = n * length L := ih fun l m => H l (mem_cons_of_mem _ m)
  /-
    case cons
    α : Type u_1
    t : α
    ts : List α
    r : List (List α)
    n : Nat
    l : List α
    L : List (List α)
    ih : (∀ (l : List α), Membership.mem L l → Eq l.length n) → Eq (List.map List. …
    H : ∀ (l_1 : List α), Membership.mem (List.cons l L) l_1 → Eq l_1.length n
    sum_map : Eq (List.map List.length L).sum (HMul.hMul n L.length)
    ⊢ Eq (List.map List.length (List.cons l L)).sum (HMul.hMul n (List.cons l L).l …
  -/
  have length_l : length l = n := H _ (mem_cons_self _ _)
  /-
    case cons
    α : Type u_1
    t : α
    ts : List α
    r : List (List α)
    n : Nat
    l : List α
    L : List (List α)
    ih : (∀ (l : List α), Membership.mem L l → Eq l.length n) → Eq (List.map List. …
    H : ∀ (l_1 : List α), Membership.mem (List.cons l L) l_1 → Eq l_1.length n
    sum_map : Eq (List.map List.length L).sum (HMul.hMul n L.length)
    length_l : Eq l.length n
    ⊢ Eq (List.map List.length (List.cons l L)).sum (HMul.hMul n (List.cons l L).l …
  -/
  simp [sum_map, length_l, Nat.mul_add, Nat.add_comm, mul_succ]
  /-
    🎉 no goals
  -/


@[simp]
theorem permutationsAux_nil (is : List α) : permutationsAux [] is = [] := by
  /-
    α : Type u_1
    is : List α
    ⊢ Eq (List.nil.permutationsAux is) List.nil
  -/
  rw [permutationsAux, permutationsAux.rec]
  /-
    🎉 no goals
  -/


@[simp]
theorem permutationsAux_cons (t : α) (ts is : List α) :
    permutationsAux (t :: ts) is =
      foldr (fun y r => (permutationsAux2 t ts r y id).2) (permutationsAux ts (t :: is))
        (permutations is) := by
  /-
    α : Type u_1
    t : α
    ts is : List α
    ⊢ Eq ((List.cons t ts).permutationsAux is) (List.foldr (fun y r => (List.permu …
  -/
  rw [permutationsAux, permutationsAux.rec]; rfl
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem permutations_nil : permutations ([] : List α) = [[]] := by
  /-
    α : Type u_1
    ⊢ Eq List.nil.permutations (List.cons List.nil List.nil)
  -/
  rw [permutations, permutationsAux_nil]
  /-
    🎉 no goals
  -/


theorem map_permutationsAux (f : α → β) :
    ∀ ts is :
    List α, map (map f) (permutationsAux ts is) = permutationsAux (map f ts) (map f is) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ⊢ ∀ (ts is : List α), Eq (List.map (List.map f) (ts.permutationsAux is)) ((Lis …
  -/
  refine permutationsAux.rec (by simp) ?_
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ⊢ ∀ (t : α) (ts is : List α), Eq (List.map (List.map f) (ts.permutationsAux (L …
  -/
  introv IH1 IH2; rw [map] at IH2
  simp only [foldr_permutationsAux2, map_append, map, map_map_permutationsAux2, permutations,
    flatMap_map, IH1, append_assoc, permutationsAux_cons, flatMap_cons, ← IH2, map_flatMap]


theorem map_permutations (f : α → β) (ts : List α) :
    map (map f) (permutations ts) = permutations (map f ts) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ts : List α
    ⊢ Eq (List.map (List.map f) ts.permutations) (List.map f ts).permutations
  -/
  rw [permutations, permutations, map, map_permutationsAux, map]
  /-
    🎉 no goals
  -/


theorem map_permutations' (f : α → β) (ts : List α) :
    map (map f) (permutations' ts) = permutations' (map f ts) := by
  induction' ts with t ts ih <;>
    [rfl; simp [← ih, map_flatMap, ← map_map_permutations'Aux, flatMap_map]]


theorem permutationsAux_append (is is' ts : List α) :
    permutationsAux (is ++ ts) is' =
      (permutationsAux is is').map (· ++ ts) ++ permutationsAux ts (is.reverse ++ is') := by
  /-
    α : Type u_1
    is is' ts : List α
    ⊢ Eq ((HAppend.hAppend is ts).permutationsAux is') (HAppend.hAppend (List.map  …
  -/
  induction' is with t is ih generalizing is'; · simp
                                                 /-
                                                   🎉 no goals
                                                 -/
  simp only [foldr_permutationsAux2, ih, map_flatMap, cons_append, permutationsAux_cons, map_append,
    reverse_cons, append_assoc, singleton_append]
  /-
    case cons
    α : Type u_1
    ts : List α
    t : α
    is : List α
    ih : ∀ (is' : List α), Eq ((HAppend.hAppend is ts).permutationsAux is') (HAppe …
    is' : List α
    ⊢ Eq (HAppend.hAppend (is'.permutations.flatMap fun y => (List.permutationsAux …
  -/
  congr 2
  /-
    case cons.e_a.e_b
    α : Type u_1
    ts : List α
    t : α
    is : List α
    ih : ∀ (is' : List α), Eq ((HAppend.hAppend is ts).permutationsAux is') (HAppe …
    is' : List α
    ⊢ Eq (fun y => (List.permutationsAux2 t (HAppend.hAppend is ts) List.nil y id) …
  -/
  funext _
  /-
    case cons.e_a.e_b.h
    α : Type u_1
    ts : List α
    t : α
    is : List α
    ih : ∀ (is' : List α), Eq ((HAppend.hAppend is ts).permutationsAux is') (HAppe …
    is' x✝ : List α
    ⊢ Eq (List.permutationsAux2 t (HAppend.hAppend is ts) List.nil x✝ id).2 (List. …
  -/
  rw [map_permutationsAux2]
  /-
    case cons.e_a.e_b.h
    α : Type u_1
    ts : List α
    t : α
    is : List α
    ih : ∀ (is' : List α), Eq ((HAppend.hAppend is ts).permutationsAux is') (HAppe …
    is' x✝ : List α
    ⊢ Eq (List.permutationsAux2 t (HAppend.hAppend is ts) List.nil x✝ id).2 (List. …
  -/
  simp (config := { singlePass := true }) only [← permutationsAux2_comp_append]
  /-
    case cons.e_a.e_b.h
    α : Type u_1
    ts : List α
    t : α
    is : List α
    ih : ∀ (is' : List α), Eq ((HAppend.hAppend is ts).permutationsAux is') (HAppe …
    is' x✝ : List α
    ⊢ Eq (List.permutationsAux2 t List.nil List.nil x✝ fun x => id (HAppend.hAppen …
  -/
  simp only [id, append_assoc]
  /-
    🎉 no goals
  -/


theorem permutations_append (is ts : List α) :
    permutations (is ++ ts) = (permutations is).map (· ++ ts) ++ permutationsAux ts is.reverse := by
  /-
    α : Type u_1
    is ts : List α
    ⊢ Eq (HAppend.hAppend is ts).permutations (HAppend.hAppend (List.map (fun x => …
  -/
  simp [permutations, permutationsAux_append]
  /-
    🎉 no goals
  -/


theorem perm_of_mem_permutationsAux :
    ∀ {ts is l : List α}, l ∈ permutationsAux ts is → l ~ ts ++ is := by
  /-
    α : Type u_1
    ⊢ ∀ {ts is l : List α}, Membership.mem (ts.permutationsAux is) l → l.Perm (HAp …
  -/
  show ∀ (ts is l : List α), l ∈ permutationsAux ts is → l ~ ts ++ is
  /-
    α : Type u_1
    ⊢ ∀ (ts is l : List α), Membership.mem (ts.permutationsAux is) l → l.Perm (HAp …
  -/
  refine permutationsAux.rec (by simp) ?_
  /-
    α : Type u_1
    ⊢ ∀ (t : α) (ts is : List α), (∀ (l : List α), Membership.mem (ts.permutations …
  -/
  introv IH1 IH2 m
  /-
    α : Type u_1
    t : α
    ts is : List α
    IH1 : ∀ (l : List α), Membership.mem (ts.permutationsAux (List.cons t is)) l → …
    IH2 : ∀ (l : List α), Membership.mem (is.permutationsAux List.nil) l → l.Perm  …
    l : List α
    m : Membership.mem ((List.cons t ts).permutationsAux is) l
    ⊢ l.Perm (HAppend.hAppend (List.cons t ts) is)
  -/
  rw [permutationsAux_cons, permutations, mem_foldr_permutationsAux2] at m
  /-
    α : Type u_1
    t : α
    ts is : List α
    IH1 : ∀ (l : List α), Membership.mem (ts.permutationsAux (List.cons t is)) l → …
    IH2 : ∀ (l : List α), Membership.mem (is.permutationsAux List.nil) l → l.Perm  …
    l : List α
    m : Or (Membership.mem (ts.permutationsAux (List.cons t is)) l) (Exists fun l₁ …
    ⊢ l.Perm (HAppend.hAppend (List.cons t ts) is)
  -/
  rcases m with (m | ⟨l₁, l₂, m, _, rfl⟩)
    /-
      case inl
      α : Type u_1
      t : α
      ts is : List α
      IH1 : ∀ (l : List α), Membership.mem (ts.permutationsAux (List.cons t is)) l → …
      IH2 : ∀ (l : List α), Membership.mem (is.permutationsAux List.nil) l → l.Perm  …
      l : List α
      m : Membership.mem (ts.permutationsAux (List.cons t is)) l
      ⊢ l.Perm (HAppend.hAppend (List.cons t ts) is)
    -/
  · exact (IH1 _ m).trans perm_middle
    /-
      🎉 no goals
    -/
  · have p : l₁ ++ l₂ ~ is := by
      simp only [mem_cons] at m
      cases' m with e m
      · simp [e]
      exact is.append_nil ▸ IH2 _ m
    /-
      case inr.intro.intro.intro.intro
      α : Type u_1
      t : α
      ts is : List α
      IH1 : ∀ (l : List α), Membership.mem (ts.permutationsAux (List.cons t is)) l → …
      IH2 : ∀ (l : List α), Membership.mem (is.permutationsAux List.nil) l → l.Perm  …
      l₁ l₂ : List α
      m : Membership.mem (List.cons is (is.permutationsAux List.nil)) (HAppend.hAppe …
      left✝ : Ne l₂ List.nil
      p : (HAppend.hAppend l₁ l₂).Perm is
      ⊢ (HAppend.hAppend (HAppend.hAppend l₁ (List.cons t l₂)) ts).Perm (HAppend.hAp …
    -/
    exact ((perm_middle.trans (p.cons _)).append_right _).trans (perm_append_comm.cons _)
    /-
      🎉 no goals
    -/


theorem perm_of_mem_permutations {l₁ l₂ : List α} (h : l₁ ∈ permutations l₂) : l₁ ~ l₂ :=
  (eq_or_mem_of_mem_cons h).elim (fun e => e ▸ Perm.refl _) fun m =>
    append_nil l₂ ▸ perm_of_mem_permutationsAux m


theorem length_permutationsAux :
    ∀ ts is : List α, length (permutationsAux ts is) + is.length ! = (length ts + length is)! := by
  /-
    α : Type u_1
    ⊢ ∀ (ts is : List α), Eq (HAdd.hAdd (ts.permutationsAux is).length is.length.f …
  -/
  refine permutationsAux.rec (by simp) ?_
  /-
    α : Type u_1
    ⊢ ∀ (t : α) (ts is : List α), Eq (HAdd.hAdd (ts.permutationsAux (List.cons t i …
  -/
  intro t ts is IH1 IH2
  /-
    α : Type u_1
    t : α
    ts is : List α
    IH1 : Eq (HAdd.hAdd (ts.permutationsAux (List.cons t is)).length (List.cons t  …
    IH2 : Eq (HAdd.hAdd (is.permutationsAux List.nil).length List.nil.length.facto …
    ⊢ Eq (HAdd.hAdd ((List.cons t ts).permutationsAux is).length is.length.factori …
  -/
  have IH2 : length (permutationsAux is nil) + 1 = is.length ! := by simpa using IH2
  /-
    α : Type u_1
    t : α
    ts is : List α
    IH1 : Eq (HAdd.hAdd (ts.permutationsAux (List.cons t is)).length (List.cons t  …
    IH2✝ : Eq (HAdd.hAdd (is.permutationsAux List.nil).length List.nil.length.fact …
    IH2 : Eq (HAdd.hAdd (is.permutationsAux List.nil).length 1) is.length.factorial
    ⊢ Eq (HAdd.hAdd ((List.cons t ts).permutationsAux is).length is.length.factori …
  -/
  simp only [factorial, Nat.mul_comm, add_eq] at IH1
  rw [permutationsAux_cons,
    length_foldr_permutationsAux2' _ _ _ _ _ fun l m => (perm_of_mem_permutations m).length_eq,
    permutations, length, length, IH2, Nat.succ_add, Nat.factorial_succ, Nat.mul_comm (_ + 1),
    ← Nat.succ_eq_add_one, ← IH1, Nat.add_comm (_ * _), Nat.add_assoc, Nat.mul_succ, Nat.mul_comm]


theorem length_permutations (l : List α) : length (permutations l) = (length l)! :=
  length_permutationsAux l []


theorem mem_permutations_of_perm_lemma {is l : List α}
    (H : l ~ [] ++ is → (∃ (ts' : _) (_ : ts' ~ []), l = ts' ++ is) ∨ l ∈ permutationsAux is []) :
                                       /-
                                         α : Type u_1
                                         is l : List α
                                         H : l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun ts' => Exists fun x  …
                                         ⊢ l.Perm is → Membership.mem is.permutations l
                                       -/
    l ~ is → l ∈ permutations is := by simpa [permutations, perm_nil] using H
                                       /-
                                         🎉 no goals
                                       -/


theorem mem_permutationsAux_of_perm :
    ∀ {ts is l : List α},
      l ~ is ++ ts → (∃ (is' : _) (_ : is' ~ is), l = is' ++ ts) ∨ l ∈ permutationsAux ts is := by
  show ∀ (ts is l : List α),
      l ~ is ++ ts → (∃ (is' : _) (_ : is' ~ is), l = is' ++ ts) ∨ l ∈ permutationsAux ts is
  /-
    α : Type u_1
    ⊢ ∀ (ts is l : List α), l.Perm (HAppend.hAppend is ts) → Or (Exists fun is' => …
  -/
  refine permutationsAux.rec (by simp) ?_
  /-
    α : Type u_1
    ⊢ ∀ (t : α) (ts is : List α), (∀ (l : List α), l.Perm (HAppend.hAppend (List.c …
  -/
  intro t ts is IH1 IH2 l p
  /-
    α : Type u_1
    t : α
    ts is : List α
    IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
    IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
    l : List α
    p : l.Perm (HAppend.hAppend is (List.cons t ts))
    ⊢ Or (Exists fun is' => Exists fun x => Eq l (HAppend.hAppend is' (List.cons t …
  -/
  rw [permutationsAux_cons, mem_foldr_permutationsAux2]
  /-
    α : Type u_1
    t : α
    ts is : List α
    IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
    IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
    l : List α
    p : l.Perm (HAppend.hAppend is (List.cons t ts))
    ⊢ Or (Exists fun is' => Exists fun x => Eq l (HAppend.hAppend is' (List.cons t …
  -/
  rcases IH1 _ (p.trans perm_middle) with (⟨is', p', e⟩ | m)
    /-
      case inl.intro.intro
      α : Type u_1
      t : α
      ts is : List α
      IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
      IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
      l : List α
      p : l.Perm (HAppend.hAppend is (List.cons t ts))
      is' : List α
      p' : is'.Perm (List.cons t is)
      e : Eq l (HAppend.hAppend is' ts)
      ⊢ Or (Exists fun is' => Exists fun x => Eq l (HAppend.hAppend is' (List.cons t …
    -/
  · clear p
    /-
      case inl.intro.intro
      α : Type u_1
      t : α
      ts is : List α
      IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
      IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
      l is' : List α
      p' : is'.Perm (List.cons t is)
      e : Eq l (HAppend.hAppend is' ts)
      ⊢ Or (Exists fun is' => Exists fun x => Eq l (HAppend.hAppend is' (List.cons t …
    -/
    subst e
    /-
      case inl.intro.intro
      α : Type u_1
      t : α
      ts is : List α
      IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
      IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
      is' : List α
      p' : is'.Perm (List.cons t is)
      ⊢ Or (Exists fun is'_1 => Exists fun x => Eq (HAppend.hAppend is' ts) (HAppend …
    -/
    rcases append_of_mem (p'.symm.subset (mem_cons_self _ _)) with ⟨l₁, l₂, e⟩
    /-
      case inl.intro.intro.intro.intro
      α : Type u_1
      t : α
      ts is : List α
      IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
      IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
      is' : List α
      p' : is'.Perm (List.cons t is)
      l₁ l₂ : List α
      e : Eq is' (HAppend.hAppend l₁ (List.cons t l₂))
      ⊢ Or (Exists fun is'_1 => Exists fun x => Eq (HAppend.hAppend is' ts) (HAppend …
    -/
    subst is'
    /-
      case inl.intro.intro.intro.intro
      α : Type u_1
      t : α
      ts is : List α
      IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
      IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
      l₁ l₂ : List α
      p' : (HAppend.hAppend l₁ (List.cons t l₂)).Perm (List.cons t is)
      ⊢ Or (Exists fun is' => Exists fun x => Eq (HAppend.hAppend (HAppend.hAppend l …
    -/
    have p := (perm_middle.symm.trans p').cons_inv
    /-
      case inl.intro.intro.intro.intro
      α : Type u_1
      t : α
      ts is : List α
      IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
      IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
      l₁ l₂ : List α
      p' : (HAppend.hAppend l₁ (List.cons t l₂)).Perm (List.cons t is)
      p : (HAppend.hAppend l₁ l₂).Perm is
      ⊢ Or (Exists fun is' => Exists fun x => Eq (HAppend.hAppend (HAppend.hAppend l …
    -/
    cases' l₂ with a l₂'
      /-
        case inl.intro.intro.intro.intro.nil
        α : Type u_1
        t : α
        ts is : List α
        IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
        IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
        l₁ : List α
        p' : (HAppend.hAppend l₁ (List.cons t List.nil)).Perm (List.cons t is)
        p : (HAppend.hAppend l₁ List.nil).Perm is
        ⊢ Or (Exists fun is' => Exists fun x => Eq (HAppend.hAppend (HAppend.hAppend l …
      -/
    · exact Or.inl ⟨l₁, by simpa using p⟩
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.intro.intro.intro.cons
        α : Type u_1
        t : α
        ts is : List α
        IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
        IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
        l₁ : List α
        a : α
        l₂' : List α
        p' : (HAppend.hAppend l₁ (List.cons t (List.cons a l₂'))).Perm (List.cons t is)
        p : (HAppend.hAppend l₁ (List.cons a l₂')).Perm is
        ⊢ Or (Exists fun is' => Exists fun x => Eq (HAppend.hAppend (HAppend.hAppend l …
      -/
    · exact Or.inr (Or.inr ⟨l₁, a :: l₂', mem_permutations_of_perm_lemma (IH2 _) p, by simp⟩)
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      t : α
      ts is : List α
      IH1 : ∀ (l : List α), l.Perm (HAppend.hAppend (List.cons t is) ts) → Or (Exist …
      IH2 : ∀ (l : List α), l.Perm (HAppend.hAppend List.nil is) → Or (Exists fun is …
      l : List α
      p : l.Perm (HAppend.hAppend is (List.cons t ts))
      m : Membership.mem (ts.permutationsAux (List.cons t is)) l
      ⊢ Or (Exists fun is' => Exists fun x => Eq l (HAppend.hAppend is' (List.cons t …
    -/
  · exact Or.inr (Or.inl m)
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_permutations {s t : List α} : s ∈ permutations t ↔ s ~ t :=
  ⟨perm_of_mem_permutations, mem_permutations_of_perm_lemma mem_permutationsAux_of_perm⟩

-- Porting note: temporary theorem to solve diamond issue

private theorem DecEq_eq [DecidableEq α] :
    List.instBEq = @instBEqOfDecidableEq (List α) instDecidableEqList :=
  congr_arg BEq.mk <| by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      ⊢ Eq List.beq fun a b => Decidable.decide (Eq a b)
    -/
    funext l₁ l₂
    /-
      case h.h
      α : Type u_1
      inst✝ : DecidableEq α
      l₁ l₂ : List α
      ⊢ Eq (l₁.beq l₂) (Decidable.decide (Eq l₁ l₂))
    -/
    show (l₁ == l₂) = _
    /-
      case h.h
      α : Type u_1
      inst✝ : DecidableEq α
      l₁ l₂ : List α
      ⊢ Eq (BEq.beq l₁ l₂) (Decidable.decide (Eq l₁ l₂))
    -/
    rw [Bool.eq_iff_iff, @beq_iff_eq _ (_), decide_eq_true_iff]
    /-
      🎉 no goals
    -/


theorem perm_permutations'Aux_comm (a b : α) (l : List α) :
    (permutations'Aux a l).flatMap (permutations'Aux b) ~
      (permutations'Aux b l).flatMap (permutations'Aux a) := by
  /-
    α : Type u_1
    a b : α
    l : List α
    ⊢ ((List.permutations'Aux a l).flatMap (List.permutations'Aux b)).Perm ((List. …
  -/
  induction' l with c l ih
    /-
      case nil
      α : Type u_1
      a b : α
      ⊢ ((List.permutations'Aux a List.nil).flatMap (List.permutations'Aux b)).Perm  …
    -/
  · exact Perm.swap [a, b] [b, a] []
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    a b c : α
    l : List α
    ih : ((List.permutations'Aux a l).flatMap (List.permutations'Aux b)).Perm ((Li …
    ⊢ ((List.permutations'Aux a (List.cons c l)).flatMap (List.permutations'Aux b) …
  -/
  simp only [permutations'Aux, flatMap_cons, map_cons, map_map, cons_append]
  /-
    case cons
    α : Type u_1
    a b c : α
    l : List α
    ih : ((List.permutations'Aux a l).flatMap (List.permutations'Aux b)).Perm ((Li …
    ⊢ (List.cons (List.cons b (List.cons a (List.cons c l))) (List.cons (List.cons …
  -/
  apply Perm.swap'
  have :
    ∀ a b,
      (map (cons c) (permutations'Aux a l)).flatMap (permutations'Aux b) ~
        map (cons b ∘ cons c) (permutations'Aux a l) ++
          map (cons c) ((permutations'Aux a l).flatMap (permutations'Aux b)) := by
    intros a' b'
    simp only [flatMap_map, permutations'Aux]
    show List.flatMap (permutations'Aux _ l) (fun a => ([b' :: c :: a] ++
      map (cons c) (permutations'Aux _ a))) ~ _
    refine (flatMap_append_perm _ (fun x => [b' :: c :: x]) _).symm.trans ?_
    rw [← map_eq_flatMap, ← map_flatMap]
    exact Perm.refl _
  /-
    case cons.p
    α : Type u_1
    a b c : α
    l : List α
    ih : ((List.permutations'Aux a l).flatMap (List.permutations'Aux b)).Perm ((Li …
    this : ∀ (a b : α), ((List.map (List.cons c) (List.permutations'Aux a l)).flat …
    ⊢ (HAppend.hAppend (List.map (Function.comp (List.cons a) (List.cons c)) (List …
  -/
  refine (((this _ _).append_left _).trans ?_).trans ((this _ _).append_left _).symm
  /-
    case cons.p
    α : Type u_1
    a b c : α
    l : List α
    ih : ((List.permutations'Aux a l).flatMap (List.permutations'Aux b)).Perm ((Li …
    this : ∀ (a b : α), ((List.map (List.cons c) (List.permutations'Aux a l)).flat …
    ⊢ (HAppend.hAppend (List.map (Function.comp (List.cons a) (List.cons c)) (List …
  -/
  rw [← append_assoc, ← append_assoc]
  /-
    case cons.p
    α : Type u_1
    a b c : α
    l : List α
    ih : ((List.permutations'Aux a l).flatMap (List.permutations'Aux b)).Perm ((Li …
    this : ∀ (a b : α), ((List.map (List.cons c) (List.permutations'Aux a l)).flat …
    ⊢ (HAppend.hAppend (HAppend.hAppend (List.map (Function.comp (List.cons a) (Li …
  -/
  exact perm_append_comm.append (ih.map _)
  /-
    🎉 no goals
  -/


theorem Perm.permutations' {s t : List α} (p : s ~ t) : permutations' s ~ permutations' t := by
  induction p with
  | nil => simp
  | cons _ _ IH => exact IH.flatMap_right _
  | swap =>
    dsimp
    rw [flatMap_assoc, flatMap_assoc]
    apply Perm.flatMap_left
    intro l' _
    apply perm_permutations'Aux_comm
  | trans _ _ IH₁ IH₂ => exact IH₁.trans IH₂


theorem permutations_perm_permutations' (ts : List α) : ts.permutations ~ ts.permutations' := by
  /-
    α : Type u_1
    ts : List α
    ⊢ ts.permutations.Perm ts.permutations'
  -/
  obtain ⟨n, h⟩ : ∃ n, length ts < n := ⟨_, Nat.lt_succ_self _⟩
  /-
    case intro
    α : Type u_1
    ts : List α
    n : Nat
    h : LT.lt ts.length n
    ⊢ ts.permutations.Perm ts.permutations'
  -/
  induction' n with n IH generalizing ts; · cases h
                                            /-
                                              🎉 no goals
                                            -/
  /-
    case intro.succ
    α : Type u_1
    n : Nat
    IH : ∀ (ts : List α), LT.lt ts.length n → ts.permutations.Perm ts.permutations'
    ts : List α
    h : LT.lt ts.length (HAdd.hAdd n 1)
    ⊢ ts.permutations.Perm ts.permutations'
  -/
  refine List.reverseRecOn ts (fun _ => ?_) (fun ts t _ h => ?_) h; · simp [permutations]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  /-
    case intro.succ.refine_2
    α : Type u_1
    n : Nat
    IH : ∀ (ts : List α), LT.lt ts.length n → ts.permutations.Perm ts.permutations'
    ts✝ : List α
    h✝ : LT.lt ts✝.length (HAdd.hAdd n 1)
    ts : List α
    t : α
    x✝ : LT.lt ts.length (HAdd.hAdd n 1) → ts.permutations.Perm ts.permutations'
    h : LT.lt (HAppend.hAppend ts (List.cons t List.nil)).length (HAdd.hAdd n 1)
    ⊢ (HAppend.hAppend ts (List.cons t List.nil)).permutations.Perm (HAppend.hAppe …
  -/
  rw [← concat_eq_append, length_concat, Nat.succ_lt_succ_iff] at h
  /-
    case intro.succ.refine_2
    α : Type u_1
    n : Nat
    IH : ∀ (ts : List α), LT.lt ts.length n → ts.permutations.Perm ts.permutations'
    ts✝ : List α
    h✝ : LT.lt ts✝.length (HAdd.hAdd n 1)
    ts : List α
    t : α
    x✝ : LT.lt ts.length (HAdd.hAdd n 1) → ts.permutations.Perm ts.permutations'
    h : LT.lt ts.length n
    ⊢ (HAppend.hAppend ts (List.cons t List.nil)).permutations.Perm (HAppend.hAppe …
  -/
  have IH₂ := (IH ts.reverse (by rwa [length_reverse])).trans (reverse_perm _).permutations'
  simp only [permutations_append, foldr_permutationsAux2, permutationsAux_nil,
    permutationsAux_cons, append_nil]
  refine
    (perm_append_comm.trans ((IH₂.flatMap_right _).append ((IH _ h).map _))).trans
      (Perm.trans ?_ perm_append_comm.permutations')
  /-
    case intro.succ.refine_2
    α : Type u_1
    n : Nat
    IH : ∀ (ts : List α), LT.lt ts.length n → ts.permutations.Perm ts.permutations'
    ts✝ : List α
    h✝ : LT.lt ts✝.length (HAdd.hAdd n 1)
    ts : List α
    t : α
    x✝ : LT.lt ts.length (HAdd.hAdd n 1) → ts.permutations.Perm ts.permutations'
    h : LT.lt ts.length n
    IH₂ : ts.reverse.permutations.Perm ts.permutations'
    ⊢ (HAppend.hAppend (ts.permutations'.flatMap fun y => (List.permutationsAux2 t …
  -/
  rw [map_eq_flatMap, singleton_append, permutations']
  /-
    case intro.succ.refine_2
    α : Type u_1
    n : Nat
    IH : ∀ (ts : List α), LT.lt ts.length n → ts.permutations.Perm ts.permutations'
    ts✝ : List α
    h✝ : LT.lt ts✝.length (HAdd.hAdd n 1)
    ts : List α
    t : α
    x✝ : LT.lt ts.length (HAdd.hAdd n 1) → ts.permutations.Perm ts.permutations'
    h : LT.lt ts.length n
    IH₂ : ts.reverse.permutations.Perm ts.permutations'
    ⊢ (HAppend.hAppend (ts.permutations'.flatMap fun y => (List.permutationsAux2 t …
  -/
  refine (flatMap_append_perm _ _ _).trans ?_
  /-
    case intro.succ.refine_2
    α : Type u_1
    n : Nat
    IH : ∀ (ts : List α), LT.lt ts.length n → ts.permutations.Perm ts.permutations'
    ts✝ : List α
    h✝ : LT.lt ts✝.length (HAdd.hAdd n 1)
    ts : List α
    t : α
    x✝ : LT.lt ts.length (HAdd.hAdd n 1) → ts.permutations.Perm ts.permutations'
    h : LT.lt ts.length n
    IH₂ : ts.reverse.permutations.Perm ts.permutations'
    ⊢ (ts.permutations'.flatMap fun x => HAppend.hAppend (List.permutationsAux2 t  …
  -/
  refine Perm.of_eq ?_
  /-
    case intro.succ.refine_2
    α : Type u_1
    n : Nat
    IH : ∀ (ts : List α), LT.lt ts.length n → ts.permutations.Perm ts.permutations'
    ts✝ : List α
    h✝ : LT.lt ts✝.length (HAdd.hAdd n 1)
    ts : List α
    t : α
    x✝ : LT.lt ts.length (HAdd.hAdd n 1) → ts.permutations.Perm ts.permutations'
    h : LT.lt ts.length n
    IH₂ : ts.reverse.permutations.Perm ts.permutations'
    ⊢ Eq (ts.permutations'.flatMap fun x => HAppend.hAppend (List.permutationsAux2 …
  -/
  congr
  /-
    case intro.succ.refine_2.e_b
    α : Type u_1
    n : Nat
    IH : ∀ (ts : List α), LT.lt ts.length n → ts.permutations.Perm ts.permutations'
    ts✝ : List α
    h✝ : LT.lt ts✝.length (HAdd.hAdd n 1)
    ts : List α
    t : α
    x✝ : LT.lt ts.length (HAdd.hAdd n 1) → ts.permutations.Perm ts.permutations'
    h : LT.lt ts.length n
    IH₂ : ts.reverse.permutations.Perm ts.permutations'
    ⊢ Eq (fun x => HAppend.hAppend (List.permutationsAux2 t List.nil List.nil x id …
  -/
  funext _
  /-
    case intro.succ.refine_2.e_b.h
    α : Type u_1
    n : Nat
    IH : ∀ (ts : List α), LT.lt ts.length n → ts.permutations.Perm ts.permutations'
    ts✝ : List α
    h✝ : LT.lt ts✝.length (HAdd.hAdd n 1)
    ts : List α
    t : α
    x✝¹ : LT.lt ts.length (HAdd.hAdd n 1) → ts.permutations.Perm ts.permutations'
    h : LT.lt ts.length n
    IH₂ : ts.reverse.permutations.Perm ts.permutations'
    x✝ : List α
    ⊢ Eq (HAppend.hAppend (List.permutationsAux2 t List.nil List.nil x✝ id).2 (Lis …
  -/
  rw [permutations'Aux_eq_permutationsAux2, permutationsAux2_append]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_permutations' {s t : List α} : s ∈ permutations' t ↔ s ~ t :=
  (permutations_perm_permutations' _).symm.mem_iff.trans mem_permutations


theorem Perm.permutations {s t : List α} (h : s ~ t) : permutations s ~ permutations t :=
  (permutations_perm_permutations' _).trans <|
    h.permutations'.trans (permutations_perm_permutations' _).symm


@[simp]
theorem perm_permutations_iff {s t : List α} : permutations s ~ permutations t ↔ s ~ t :=
  ⟨fun h => mem_permutations.1 <| h.mem_iff.1 <| mem_permutations.2 (Perm.refl _),
    Perm.permutations⟩


@[simp]
theorem perm_permutations'_iff {s t : List α} : permutations' s ~ permutations' t ↔ s ~ t :=
  ⟨fun h => mem_permutations'.1 <| h.mem_iff.1 <| mem_permutations'.2 (Perm.refl _),
    Perm.permutations'⟩


theorem getElem_permutations'Aux (s : List α) (x : α) (n : ℕ)
    (hn : n < length (permutations'Aux x s)) :
    (permutations'Aux x s)[n] = s.insertIdx n x := by
  /-
    α : Type u_1
    s : List α
    x : α
    n : Nat
    hn : LT.lt n (List.permutations'Aux x s).length
    ⊢ Eq (GetElem.getElem (List.permutations'Aux x s) n hn) (List.insertIdx n x s)
  -/
  induction' s with y s IH generalizing n
    /-
      case nil
      α : Type u_1
      x : α
      n : Nat
      hn : LT.lt n (List.permutations'Aux x List.nil).length
      ⊢ Eq (GetElem.getElem (List.permutations'Aux x List.nil) n hn) (List.insertIdx …
    -/
  · simp only [length, Nat.zero_add, Nat.lt_one_iff] at hn
    /-
      case nil
      α : Type u_1
      x : α
      n : Nat
      hn✝ : LT.lt n (List.permutations'Aux x List.nil).length
      hn : Eq n 0
      ⊢ Eq (GetElem.getElem (List.permutations'Aux x List.nil) n hn✝) (List.insertId …
    -/
    simp [hn]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      x y : α
      s : List α
      IH : ∀ (n : Nat) (hn : LT.lt n (List.permutations'Aux x s).length), Eq (GetEle …
      n : Nat
      hn : LT.lt n (List.permutations'Aux x (List.cons y s)).length
      ⊢ Eq (GetElem.getElem (List.permutations'Aux x (List.cons y s)) n hn) (List.in …
    -/
  · cases n
      /-
        case cons.zero
        α : Type u_1
        x y : α
        s : List α
        IH : ∀ (n : Nat) (hn : LT.lt n (List.permutations'Aux x s).length), Eq (GetEle …
        hn : LT.lt 0 (List.permutations'Aux x (List.cons y s)).length
        ⊢ Eq (GetElem.getElem (List.permutations'Aux x (List.cons y s)) 0 hn) (List.in …
      -/
    · simp [get]
      /-
        🎉 no goals
      -/
      /-
        case cons.succ
        α : Type u_1
        x y : α
        s : List α
        IH : ∀ (n : Nat) (hn : LT.lt n (List.permutations'Aux x s).length), Eq (GetEle …
        n✝ : Nat
        hn : LT.lt (HAdd.hAdd n✝ 1) (List.permutations'Aux x (List.cons y s)).length
        ⊢ Eq (GetElem.getElem (List.permutations'Aux x (List.cons y s)) (HAdd.hAdd n✝  …
      -/
    · simpa [get] using IH _ _
      /-
        🎉 no goals
      -/


theorem get_permutations'Aux (s : List α) (x : α) (n : ℕ)
    (hn : n < length (permutations'Aux x s)) :
    (permutations'Aux x s).get ⟨n, hn⟩ = s.insertIdx n x := by
  /-
    α : Type u_1
    s : List α
    x : α
    n : Nat
    hn : LT.lt n (List.permutations'Aux x s).length
    ⊢ Eq ((List.permutations'Aux x s).get ⟨n, hn⟩) (List.insertIdx n x s)
  -/
  simp [getElem_permutations'Aux]
  /-
    🎉 no goals
  -/


theorem count_permutations'Aux_self [DecidableEq α] (l : List α) (x : α) :
    count (x :: l) (permutations'Aux x l) = length (takeWhile (x = ·) l) + 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    x : α
    ⊢ Eq (List.count (List.cons x l) (List.permutations'Aux x l)) (HAdd.hAdd (List …
  -/
  induction' l with y l IH generalizing x
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      x : α
      ⊢ Eq (List.count (List.cons x List.nil) (List.permutations'Aux x List.nil)) (H …
    -/
  · simp [takeWhile, count]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      y : α
      l : List α
      IH : ∀ (x : α), Eq (List.count (List.cons x l) (List.permutations'Aux x l)) (H …
      x : α
      ⊢ Eq (List.count (List.cons x (List.cons y l)) (List.permutations'Aux x (List. …
    -/
  · rw [permutations'Aux, count_cons_self]
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      y : α
      l : List α
      IH : ∀ (x : α), Eq (List.count (List.cons x l) (List.permutations'Aux x l)) (H …
      x : α
      ⊢ Eq (HAdd.hAdd (List.count (List.cons x (List.cons y l)) (List.map (List.cons …
    -/
    by_cases hx : x = y
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        y : α
        l : List α
        IH : ∀ (x : α), Eq (List.count (List.cons x l) (List.permutations'Aux x l)) (H …
        x : α
        hx : Eq x y
        ⊢ Eq (HAdd.hAdd (List.count (List.cons x (List.cons y l)) (List.map (List.cons …
      -/
    · subst hx
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        IH : ∀ (x : α), Eq (List.count (List.cons x l) (List.permutations'Aux x l)) (H …
        x : α
        ⊢ Eq (HAdd.hAdd (List.count (List.cons x (List.cons x l)) (List.map (List.cons …
      -/
      simpa [takeWhile, Nat.succ_inj', DecEq_eq] using IH _
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        y : α
        l : List α
        IH : ∀ (x : α), Eq (List.count (List.cons x l) (List.permutations'Aux x l)) (H …
        x : α
        hx : Not (Eq x y)
        ⊢ Eq (HAdd.hAdd (List.count (List.cons x (List.cons y l)) (List.map (List.cons …
      -/
    · rw [takeWhile]
      simp only [mem_map, cons.injEq, Ne.symm hx, false_and, and_false, exists_false,
        not_false_iff, count_eq_zero_of_not_mem, Nat.zero_add, hx, decide_false, length_nil]


@[simp]
theorem length_permutations'Aux (s : List α) (x : α) :
    length (permutations'Aux x s) = length s + 1 := by
  /-
    α : Type u_1
    s : List α
    x : α
    ⊢ Eq (List.permutations'Aux x s).length (HAdd.hAdd s.length 1)
  -/
  induction' s with y s IH
    /-
      case nil
      α : Type u_1
      x : α
      ⊢ Eq (List.permutations'Aux x List.nil).length (HAdd.hAdd List.nil.length 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      x y : α
      s : List α
      IH : Eq (List.permutations'Aux x s).length (HAdd.hAdd s.length 1)
      ⊢ Eq (List.permutations'Aux x (List.cons y s)).length (HAdd.hAdd (List.cons y  …
    -/
  · simpa using IH
    /-
      🎉 no goals
    -/


@[deprecated "No deprecation message was provided." (since := "2024-06-12")]
theorem permutations'Aux_get_zero (s : List α) (x : α)
                                                   /-
                                                     α : Type u_1
                                                     β : Type u_2
                                                     s : List α
                                                     x : α
                                                     ⊢ LT.lt 0 (List.permutations'Aux x s).length
                                                   -/
    (hn : 0 < length (permutations'Aux x s) := (by simp)) :
                                                   /-
                                                     🎉 no goals
                                                   -/
    (permutations'Aux x s).get ⟨0, hn⟩ = x :: s :=
  get_permutations'Aux _ _ _ _


theorem injective_permutations'Aux (x : α) : Function.Injective (permutations'Aux x) := by
  /-
    α : Type u_1
    x : α
    ⊢ Function.Injective (List.permutations'Aux x)
  -/
  intro s t h
  /-
    α : Type u_1
    x : α
    s t : List α
    h : Eq (List.permutations'Aux x s) (List.permutations'Aux x t)
    ⊢ Eq s t
  -/
  apply insertIdx_injective s.length x
  /-
    case a
    α : Type u_1
    x : α
    s t : List α
    h : Eq (List.permutations'Aux x s) (List.permutations'Aux x t)
    ⊢ Eq (List.insertIdx s.length x s) (List.insertIdx s.length x t)
  -/
  have hl : s.length = t.length := by simpa using congr_arg length h
  rw [← get_permutations'Aux s x s.length (by simp),
    ← get_permutations'Aux t x s.length (by simp [hl])]
  /-
    case a
    α : Type u_1
    x : α
    s t : List α
    h : Eq (List.permutations'Aux x s) (List.permutations'Aux x t)
    hl : Eq s.length t.length
    ⊢ Eq ((List.permutations'Aux x s).get ⟨s.length, ⋯⟩) ((List.permutations'Aux x …
  -/
  simp only [get_eq_getElem, h, hl]
  /-
    🎉 no goals
  -/


theorem nodup_permutations'Aux_of_not_mem (s : List α) (x : α) (hx : x ∉ s) :
    Nodup (permutations'Aux x s) := by
  /-
    α : Type u_1
    s : List α
    x : α
    hx : Not (Membership.mem s x)
    ⊢ (List.permutations'Aux x s).Nodup
  -/
  induction' s with y s IH
    /-
      case nil
      α : Type u_1
      x : α
      hx : Not (Membership.mem List.nil x)
      ⊢ (List.permutations'Aux x List.nil).Nodup
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      x y : α
      s : List α
      IH : Not (Membership.mem s x) → (List.permutations'Aux x s).Nodup
      hx : Not (Membership.mem (List.cons y s) x)
      ⊢ (List.permutations'Aux x (List.cons y s)).Nodup
    -/
  · simp only [not_or, mem_cons] at hx
    /-
      case cons
      α : Type u_1
      x y : α
      s : List α
      IH : Not (Membership.mem s x) → (List.permutations'Aux x s).Nodup
      hx : And (Not (Eq x y)) (Not (Membership.mem s x))
      ⊢ (List.permutations'Aux x (List.cons y s)).Nodup
    -/
    simp only [permutations'Aux, nodup_cons, mem_map, cons.injEq, exists_eq_right_right, not_and]
    /-
      case cons
      α : Type u_1
      x y : α
      s : List α
      IH : Not (Membership.mem s x) → (List.permutations'Aux x s).Nodup
      hx : And (Not (Eq x y)) (Not (Membership.mem s x))
      ⊢ And (Membership.mem (List.permutations'Aux x s) (List.cons y s) → Not (Eq y  …
    -/
    refine ⟨fun _ => Ne.symm hx.left, ?_⟩
    /-
      case cons
      α : Type u_1
      x y : α
      s : List α
      IH : Not (Membership.mem s x) → (List.permutations'Aux x s).Nodup
      hx : And (Not (Eq x y)) (Not (Membership.mem s x))
      ⊢ (List.map (List.cons y) (List.permutations'Aux x s)).Nodup
    -/
    rw [nodup_map_iff]
      /-
        case cons
        α : Type u_1
        x y : α
        s : List α
        IH : Not (Membership.mem s x) → (List.permutations'Aux x s).Nodup
        hx : And (Not (Eq x y)) (Not (Membership.mem s x))
        ⊢ (List.permutations'Aux x s).Nodup
      -/
    · exact IH hx.right
      /-
        🎉 no goals
      -/
      /-
        case cons
        α : Type u_1
        x y : α
        s : List α
        IH : Not (Membership.mem s x) → (List.permutations'Aux x s).Nodup
        hx : And (Not (Eq x y)) (Not (Membership.mem s x))
        ⊢ Function.Injective (List.cons y)
      -/
    · simp
      /-
        🎉 no goals
      -/


theorem nodup_permutations'Aux_iff {s : List α} {x : α} : Nodup (permutations'Aux x s) ↔ x ∉ s := by
  /-
    α : Type u_1
    s : List α
    x : α
    ⊢ Iff (List.permutations'Aux x s).Nodup (Not (Membership.mem s x))
  -/
  refine ⟨fun h H ↦ ?_, nodup_permutations'Aux_of_not_mem _ _⟩
  /-
    α : Type u_1
    s : List α
    x : α
    h : (List.permutations'Aux x s).Nodup
    H : Membership.mem s x
    ⊢ False
  -/
  obtain ⟨⟨k, hk⟩, hk'⟩ := get_of_mem H
  /-
    case intro.mk
    α : Type u_1
    s : List α
    x : α
    h : (List.permutations'Aux x s).Nodup
    H : Membership.mem s x
    k : Nat
    hk : LT.lt k s.length
    hk' : Eq (s.get ⟨k, hk⟩) x
    ⊢ False
  -/
  rw [nodup_iff_injective_get] at h
  /-
    case intro.mk
    α : Type u_1
    s : List α
    x : α
    h : Function.Injective (List.permutations'Aux x s).get
    H : Membership.mem s x
    k : Nat
    hk : LT.lt k s.length
    hk' : Eq (s.get ⟨k, hk⟩) x
    ⊢ False
  -/
  apply k.succ_ne_self.symm
  /-
    case intro.mk
    α : Type u_1
    s : List α
    x : α
    h : Function.Injective (List.permutations'Aux x s).get
    H : Membership.mem s x
    k : Nat
    hk : LT.lt k s.length
    hk' : Eq (s.get ⟨k, hk⟩) x
    ⊢ Eq k k.succ
  -/
  have kl : k < (permutations'Aux x s).length := by simpa [Nat.lt_succ_iff] using hk.le
  /-
    case intro.mk
    α : Type u_1
    s : List α
    x : α
    h : Function.Injective (List.permutations'Aux x s).get
    H : Membership.mem s x
    k : Nat
    hk : LT.lt k s.length
    hk' : Eq (s.get ⟨k, hk⟩) x
    kl : LT.lt k (List.permutations'Aux x s).length
    ⊢ Eq k k.succ
  -/
  have k1l : k + 1 < (permutations'Aux x s).length := by simpa using hk
  /-
    case intro.mk
    α : Type u_1
    s : List α
    x : α
    h : Function.Injective (List.permutations'Aux x s).get
    H : Membership.mem s x
    k : Nat
    hk : LT.lt k s.length
    hk' : Eq (s.get ⟨k, hk⟩) x
    kl : LT.lt k (List.permutations'Aux x s).length
    k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
    ⊢ Eq k k.succ
  -/
  rw [← @Fin.mk.inj_iff _ _ _ kl k1l]; apply h
  /-
    case intro.mk.a
    α : Type u_1
    s : List α
    x : α
    h : Function.Injective (List.permutations'Aux x s).get
    H : Membership.mem s x
    k : Nat
    hk : LT.lt k s.length
    hk' : Eq (s.get ⟨k, hk⟩) x
    kl : LT.lt k (List.permutations'Aux x s).length
    k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
    ⊢ Eq ((List.permutations'Aux x s).get ⟨k, kl⟩) ((List.permutations'Aux x s).ge …
  -/
  rw [get_permutations'Aux, get_permutations'Aux]
  have hl : length (insertIdx k x s) = length (insertIdx (k + 1) x s) := by
    rw [length_insertIdx_of_le_length hk.le, length_insertIdx_of_le_length (Nat.succ_le_of_lt hk)]
  /-
    case intro.mk.a
    α : Type u_1
    s : List α
    x : α
    h : Function.Injective (List.permutations'Aux x s).get
    H : Membership.mem s x
    k : Nat
    hk : LT.lt k s.length
    hk' : Eq (s.get ⟨k, hk⟩) x
    kl : LT.lt k (List.permutations'Aux x s).length
    k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
    hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
    ⊢ Eq (List.insertIdx k x s) (List.insertIdx (HAdd.hAdd k 1) x s)
  -/
  refine ext_get hl fun n hn hn' => ?_
  /-
    case intro.mk.a
    α : Type u_1
    s : List α
    x : α
    h : Function.Injective (List.permutations'Aux x s).get
    H : Membership.mem s x
    k : Nat
    hk : LT.lt k s.length
    hk' : Eq (s.get ⟨k, hk⟩) x
    kl : LT.lt k (List.permutations'Aux x s).length
    k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
    hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
    n : Nat
    hn : LT.lt n (List.insertIdx k x s).length
    hn' : LT.lt n (List.insertIdx (HAdd.hAdd k 1) x s).length
    ⊢ Eq ((List.insertIdx k x s).get ⟨n, hn⟩) ((List.insertIdx (HAdd.hAdd k 1) x s …
  -/
  rcases lt_trichotomy n k with (H | rfl | H)
  · rw [get_insertIdx_of_lt _ _ _ _ H (H.trans hk),
      get_insertIdx_of_lt _ _ _ _ (H.trans (Nat.lt_succ_self _))]
    /-
      case intro.mk.a.inr.inl
      α : Type u_1
      s : List α
      x : α
      h : Function.Injective (List.permutations'Aux x s).get
      H : Membership.mem s x
      n : Nat
      hk : LT.lt n s.length
      hk' : Eq (s.get ⟨n, hk⟩) x
      kl : LT.lt n (List.permutations'Aux x s).length
      k1l : LT.lt (HAdd.hAdd n 1) (List.permutations'Aux x s).length
      hl : Eq (List.insertIdx n x s).length (List.insertIdx (HAdd.hAdd n 1) x s).len …
      hn : LT.lt n (List.insertIdx n x s).length
      hn' : LT.lt n (List.insertIdx (HAdd.hAdd n 1) x s).length
      ⊢ Eq ((List.insertIdx n x s).get ⟨n, hn⟩) ((List.insertIdx (HAdd.hAdd n 1) x s …
    -/
  · rw [get_insertIdx_self _ _ _ hk.le, get_insertIdx_of_lt _ _ _ _ (Nat.lt_succ_self _) hk, hk']
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.a.inr.inr
      α : Type u_1
      s : List α
      x : α
      h : Function.Injective (List.permutations'Aux x s).get
      H✝ : Membership.mem s x
      k : Nat
      hk : LT.lt k s.length
      hk' : Eq (s.get ⟨k, hk⟩) x
      kl : LT.lt k (List.permutations'Aux x s).length
      k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
      hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
      n : Nat
      hn : LT.lt n (List.insertIdx k x s).length
      hn' : LT.lt n (List.insertIdx (HAdd.hAdd k 1) x s).length
      H : LT.lt k n
      ⊢ Eq ((List.insertIdx k x s).get ⟨n, hn⟩) ((List.insertIdx (HAdd.hAdd k 1) x s …
    -/
  · rcases (Nat.succ_le_of_lt H).eq_or_lt with (rfl | H')
      /-
        case intro.mk.a.inr.inr.inl
        α : Type u_1
        s : List α
        x : α
        h : Function.Injective (List.permutations'Aux x s).get
        H✝ : Membership.mem s x
        k : Nat
        hk : LT.lt k s.length
        hk' : Eq (s.get ⟨k, hk⟩) x
        kl : LT.lt k (List.permutations'Aux x s).length
        k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
        hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
        hn : LT.lt k.succ (List.insertIdx k x s).length
        hn' : LT.lt k.succ (List.insertIdx (HAdd.hAdd k 1) x s).length
        H : LT.lt k k.succ
        ⊢ Eq ((List.insertIdx k x s).get ⟨k.succ, hn⟩) ((List.insertIdx (HAdd.hAdd k 1 …
      -/
    · rw [get_insertIdx_self _ _ _ (Nat.succ_le_of_lt hk)]
      /-
        case intro.mk.a.inr.inr.inl
        α : Type u_1
        s : List α
        x : α
        h : Function.Injective (List.permutations'Aux x s).get
        H✝ : Membership.mem s x
        k : Nat
        hk : LT.lt k s.length
        hk' : Eq (s.get ⟨k, hk⟩) x
        kl : LT.lt k (List.permutations'Aux x s).length
        k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
        hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
        hn : LT.lt k.succ (List.insertIdx k x s).length
        hn' : LT.lt k.succ (List.insertIdx (HAdd.hAdd k 1) x s).length
        H : LT.lt k k.succ
        ⊢ Eq ((List.insertIdx k x s).get ⟨k.succ, hn⟩) x
      -/
      convert hk' using 1
      /-
        case h.e'_2
        α : Type u_1
        s : List α
        x : α
        h : Function.Injective (List.permutations'Aux x s).get
        H✝ : Membership.mem s x
        k : Nat
        hk : LT.lt k s.length
        hk' : Eq (s.get ⟨k, hk⟩) x
        kl : LT.lt k (List.permutations'Aux x s).length
        k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
        hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
        hn : LT.lt k.succ (List.insertIdx k x s).length
        hn' : LT.lt k.succ (List.insertIdx (HAdd.hAdd k 1) x s).length
        H : LT.lt k k.succ
        ⊢ Eq ((List.insertIdx k x s).get ⟨k.succ, hn⟩) (s.get ⟨k, hk⟩)
      -/
      exact get_insertIdx_add_succ _ _ _ 0 _
      /-
        🎉 no goals
      -/
      /-
        case intro.mk.a.inr.inr.inr
        α : Type u_1
        s : List α
        x : α
        h : Function.Injective (List.permutations'Aux x s).get
        H✝ : Membership.mem s x
        k : Nat
        hk : LT.lt k s.length
        hk' : Eq (s.get ⟨k, hk⟩) x
        kl : LT.lt k (List.permutations'Aux x s).length
        k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
        hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
        n : Nat
        hn : LT.lt n (List.insertIdx k x s).length
        hn' : LT.lt n (List.insertIdx (HAdd.hAdd k 1) x s).length
        H : LT.lt k n
        H' : LT.lt k.succ n
        ⊢ Eq ((List.insertIdx k x s).get ⟨n, hn⟩) ((List.insertIdx (HAdd.hAdd k 1) x s …
      -/
    · obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_lt H'
      /-
        case intro.mk.a.inr.inr.inr.intro
        α : Type u_1
        s : List α
        x : α
        h : Function.Injective (List.permutations'Aux x s).get
        H✝ : Membership.mem s x
        k : Nat
        hk : LT.lt k s.length
        hk' : Eq (s.get ⟨k, hk⟩) x
        kl : LT.lt k (List.permutations'Aux x s).length
        k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
        hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
        m : Nat
        hn : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx k x s).length
        hn' : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx (HAdd.hAdd k 1) …
        H : LT.lt k (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
        H' : LT.lt k.succ (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
        ⊢ Eq ((List.insertIdx k x s).get ⟨HAdd.hAdd (HAdd.hAdd k.succ m) 1, hn⟩) ((Lis …
      -/
      rw [length_insertIdx_of_le_length hk.le, Nat.succ_lt_succ_iff, Nat.succ_add] at hn
      /-
        case intro.mk.a.inr.inr.inr.intro
        α : Type u_1
        s : List α
        x : α
        h : Function.Injective (List.permutations'Aux x s).get
        H✝ : Membership.mem s x
        k : Nat
        hk : LT.lt k s.length
        hk' : Eq (s.get ⟨k, hk⟩) x
        kl : LT.lt k (List.permutations'Aux x s).length
        k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
        hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
        m : Nat
        hn✝ : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx k x s).length
        hn : LT.lt (HAdd.hAdd k m).succ s.length
        hn' : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx (HAdd.hAdd k 1) …
        H : LT.lt k (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
        H' : LT.lt k.succ (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
        ⊢ Eq ((List.insertIdx k x s).get ⟨HAdd.hAdd (HAdd.hAdd k.succ m) 1, hn✝⟩) ((Li …
      -/
      rw [get_insertIdx_add_succ]
        /-
          case intro.mk.a.inr.inr.inr.intro
          α : Type u_1
          s : List α
          x : α
          h : Function.Injective (List.permutations'Aux x s).get
          H✝ : Membership.mem s x
          k : Nat
          hk : LT.lt k s.length
          hk' : Eq (s.get ⟨k, hk⟩) x
          kl : LT.lt k (List.permutations'Aux x s).length
          k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
          hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
          m : Nat
          hn✝ : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx k x s).length
          hn : LT.lt (HAdd.hAdd k m).succ s.length
          hn' : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx (HAdd.hAdd k 1) …
          H : LT.lt k (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
          H' : LT.lt k.succ (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
          ⊢ Eq ((List.insertIdx k x s).get ⟨HAdd.hAdd (HAdd.hAdd k.succ m) 1, hn✝⟩) (s.g …
        -/
      · convert get_insertIdx_add_succ s x k m.succ (by simpa using hn) using 2
          /-
            case h.e'_2.h.e'_3
            α : Type u_1
            s : List α
            x : α
            h : Function.Injective (List.permutations'Aux x s).get
            H✝ : Membership.mem s x
            k : Nat
            hk : LT.lt k s.length
            hk' : Eq (s.get ⟨k, hk⟩) x
            kl : LT.lt k (List.permutations'Aux x s).length
            k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
            hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
            m : Nat
            hn✝ : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx k x s).length
            hn : LT.lt (HAdd.hAdd k m).succ s.length
            hn' : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx (HAdd.hAdd k 1) …
            H : LT.lt k (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
            H' : LT.lt k.succ (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
            ⊢ Eq ⟨HAdd.hAdd (HAdd.hAdd k.succ m) 1, hn✝⟩ ⟨HAdd.hAdd (HAdd.hAdd k m.succ) 1 …
          -/
        · simp [Nat.add_assoc, Nat.add_left_comm]
          /-
            🎉 no goals
          -/
          /-
            case h.e'_3.h.e'_3
            α : Type u_1
            s : List α
            x : α
            h : Function.Injective (List.permutations'Aux x s).get
            H✝ : Membership.mem s x
            k : Nat
            hk : LT.lt k s.length
            hk' : Eq (s.get ⟨k, hk⟩) x
            kl : LT.lt k (List.permutations'Aux x s).length
            k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
            hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
            m : Nat
            hn✝ : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx k x s).length
            hn : LT.lt (HAdd.hAdd k m).succ s.length
            hn' : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx (HAdd.hAdd k 1) …
            H : LT.lt k (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
            H' : LT.lt k.succ (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
            ⊢ Eq ⟨HAdd.hAdd (HAdd.hAdd k 1) m, ?intro.mk.a.inr.inr.inr.intro.hk'⟩ ⟨HAdd.hA …
          -/
        · simp [Nat.add_left_comm, Nat.add_comm]
          /-
            🎉 no goals
          -/
        /-
          case intro.mk.a.inr.inr.inr.intro.hk'
          α : Type u_1
          s : List α
          x : α
          h : Function.Injective (List.permutations'Aux x s).get
          H✝ : Membership.mem s x
          k : Nat
          hk : LT.lt k s.length
          hk' : Eq (s.get ⟨k, hk⟩) x
          kl : LT.lt k (List.permutations'Aux x s).length
          k1l : LT.lt (HAdd.hAdd k 1) (List.permutations'Aux x s).length
          hl : Eq (List.insertIdx k x s).length (List.insertIdx (HAdd.hAdd k 1) x s).len …
          m : Nat
          hn✝ : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx k x s).length
          hn : LT.lt (HAdd.hAdd k m).succ s.length
          hn' : LT.lt (HAdd.hAdd (HAdd.hAdd k.succ m) 1) (List.insertIdx (HAdd.hAdd k 1) …
          H : LT.lt k (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
          H' : LT.lt k.succ (HAdd.hAdd (HAdd.hAdd k.succ m) 1)
          ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd k 1) m) s.length
        -/
      · simpa [Nat.succ_add] using hn
        /-
          🎉 no goals
        -/


theorem nodup_permutations (s : List α) (hs : Nodup s) : Nodup s.permutations := by
  /-
    α : Type u_1
    s : List α
    hs : s.Nodup
    ⊢ s.permutations.Nodup
  -/
  rw [(permutations_perm_permutations' s).nodup_iff]
  /-
    α : Type u_1
    s : List α
    hs : s.Nodup
    ⊢ s.permutations'.Nodup
  -/
  induction' hs with x l h h' IH
    /-
      case nil
      α : Type u_1
      s : List α
      ⊢ List.nil.permutations'.Nodup
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      s : List α
      x : α
      l : List α
      h : ∀ (a' : α), Membership.mem l a' → Ne x a'
      h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
      IH : l.permutations'.Nodup
      ⊢ (List.cons x l).permutations'.Nodup
    -/
  · rw [permutations']
    /-
      case cons
      α : Type u_1
      s : List α
      x : α
      l : List α
      h : ∀ (a' : α), Membership.mem l a' → Ne x a'
      h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
      IH : l.permutations'.Nodup
      ⊢ (l.permutations'.flatMap (List.permutations'Aux x)).Nodup
    -/
    rw [nodup_flatMap]
    /-
      case cons
      α : Type u_1
      s : List α
      x : α
      l : List α
      h : ∀ (a' : α), Membership.mem l a' → Ne x a'
      h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
      IH : l.permutations'.Nodup
      ⊢ And (∀ (x_1 : List α), Membership.mem l.permutations' x_1 → (List.permutatio …
    -/
    constructor
      /-
        case cons.left
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        ⊢ ∀ (x_1 : List α), Membership.mem l.permutations' x_1 → (List.permutations'Au …
      -/
    · intro ys hy
      /-
        case cons.left
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        ys : List α
        hy : Membership.mem l.permutations' ys
        ⊢ (List.permutations'Aux x ys).Nodup
      -/
      rw [mem_permutations'] at hy
      /-
        case cons.left
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        ys : List α
        hy : ys.Perm l
        ⊢ (List.permutations'Aux x ys).Nodup
      -/
      rw [nodup_permutations'Aux_iff, hy.mem_iff]
      /-
        case cons.left
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        ys : List α
        hy : ys.Perm l
        ⊢ Not (Membership.mem l x)
      -/
      exact fun H => h x H rfl
      /-
        🎉 no goals
      -/
      /-
        case cons.right
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        ⊢ List.Pairwise (Function.onFun List.Disjoint (List.permutations'Aux x)) l.per …
      -/
    · refine IH.pairwise_of_forall_ne fun as ha bs hb H => ?_
      /-
        case cons.right
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        as : List α
        ha : Membership.mem l.permutations' as
        bs : List α
        hb : Membership.mem l.permutations' bs
        H : Ne as bs
        ⊢ Function.onFun List.Disjoint (List.permutations'Aux x) as bs
      -/
      rw [Function.onFun, disjoint_iff_ne]
      /-
        case cons.right
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        as : List α
        ha : Membership.mem l.permutations' as
        bs : List α
        hb : Membership.mem l.permutations' bs
        H : Ne as bs
        ⊢ ∀ (a : List α), Membership.mem (List.permutations'Aux x as) a → ∀ (b : List  …
      -/
      rintro a ha' b hb' rfl
      /-
        case cons.right
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        as : List α
        ha : Membership.mem l.permutations' as
        bs : List α
        hb : Membership.mem l.permutations' bs
        H : Ne as bs
        a : List α
        ha' : Membership.mem (List.permutations'Aux x as) a
        hb' : Membership.mem (List.permutations'Aux x bs) a
        ⊢ False
      -/
      obtain ⟨⟨n, hn⟩, hn'⟩ := get_of_mem ha'
      /-
        case cons.right.intro.mk
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        as : List α
        ha : Membership.mem l.permutations' as
        bs : List α
        hb : Membership.mem l.permutations' bs
        H : Ne as bs
        a : List α
        ha' : Membership.mem (List.permutations'Aux x as) a
        hb' : Membership.mem (List.permutations'Aux x bs) a
        n : Nat
        hn : LT.lt n (List.permutations'Aux x as).length
        hn' : Eq ((List.permutations'Aux x as).get ⟨n, hn⟩) a
        ⊢ False
      -/
      obtain ⟨⟨m, hm⟩, hm'⟩ := get_of_mem hb'
      /-
        case cons.right.intro.mk.intro.mk
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        as : List α
        ha : Membership.mem l.permutations' as
        bs : List α
        hb : Membership.mem l.permutations' bs
        H : Ne as bs
        a : List α
        ha' : Membership.mem (List.permutations'Aux x as) a
        hb' : Membership.mem (List.permutations'Aux x bs) a
        n : Nat
        hn : LT.lt n (List.permutations'Aux x as).length
        hn' : Eq ((List.permutations'Aux x as).get ⟨n, hn⟩) a
        m : Nat
        hm : LT.lt m (List.permutations'Aux x bs).length
        hm' : Eq ((List.permutations'Aux x bs).get ⟨m, hm⟩) a
        ⊢ False
      -/
      rw [mem_permutations'] at ha hb
      /-
        case cons.right.intro.mk.intro.mk
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        as : List α
        ha : as.Perm l
        bs : List α
        hb : bs.Perm l
        H : Ne as bs
        a : List α
        ha' : Membership.mem (List.permutations'Aux x as) a
        hb' : Membership.mem (List.permutations'Aux x bs) a
        n : Nat
        hn : LT.lt n (List.permutations'Aux x as).length
        hn' : Eq ((List.permutations'Aux x as).get ⟨n, hn⟩) a
        m : Nat
        hm : LT.lt m (List.permutations'Aux x bs).length
        hm' : Eq ((List.permutations'Aux x bs).get ⟨m, hm⟩) a
        ⊢ False
      -/
      have hl : as.length = bs.length := (ha.trans hb.symm).length_eq
      /-
        case cons.right.intro.mk.intro.mk
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        as : List α
        ha : as.Perm l
        bs : List α
        hb : bs.Perm l
        H : Ne as bs
        a : List α
        ha' : Membership.mem (List.permutations'Aux x as) a
        hb' : Membership.mem (List.permutations'Aux x bs) a
        n : Nat
        hn : LT.lt n (List.permutations'Aux x as).length
        hn' : Eq ((List.permutations'Aux x as).get ⟨n, hn⟩) a
        m : Nat
        hm : LT.lt m (List.permutations'Aux x bs).length
        hm' : Eq ((List.permutations'Aux x bs).get ⟨m, hm⟩) a
        hl : Eq as.length bs.length
        ⊢ False
      -/
      simp only [Nat.lt_succ_iff, length_permutations'Aux] at hn hm
      /-
        case cons.right.intro.mk.intro.mk
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        as : List α
        ha : as.Perm l
        bs : List α
        hb : bs.Perm l
        H : Ne as bs
        a : List α
        ha' : Membership.mem (List.permutations'Aux x as) a
        hb' : Membership.mem (List.permutations'Aux x bs) a
        n : Nat
        hn✝ : LT.lt n (List.permutations'Aux x as).length
        hn' : Eq ((List.permutations'Aux x as).get ⟨n, hn✝⟩) a
        m : Nat
        hm✝ : LT.lt m (List.permutations'Aux x bs).length
        hm' : Eq ((List.permutations'Aux x bs).get ⟨m, hm✝⟩) a
        hl : Eq as.length bs.length
        hn : LE.le n as.length
        hm : LE.le m bs.length
        ⊢ False
      -/
      rw [get_permutations'Aux] at hn' hm'
      have hx : (insertIdx n x as)[m]'(by
          rwa [length_insertIdx_of_le_length hn, Nat.lt_succ_iff, hl]) = x := by
        simp [hn', ← hm', hm]
      have hx' : (insertIdx m x bs)[n]'(by
          rwa [length_insertIdx_of_le_length hm, Nat.lt_succ_iff, ← hl]) = x := by
        simp [hm', ← hn', hn]
      /-
        case cons.right.intro.mk.intro.mk
        α : Type u_1
        s : List α
        x : α
        l : List α
        h : ∀ (a' : α), Membership.mem l a' → Ne x a'
        h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
        IH : l.permutations'.Nodup
        as : List α
        ha : as.Perm l
        bs : List α
        hb : bs.Perm l
        H : Ne as bs
        a : List α
        ha' : Membership.mem (List.permutations'Aux x as) a
        hb' : Membership.mem (List.permutations'Aux x bs) a
        n : Nat
        hn✝ : LT.lt n (List.permutations'Aux x as).length
        hn' : Eq (List.insertIdx n x as) a
        m : Nat
        hm✝ : LT.lt m (List.permutations'Aux x bs).length
        hm' : Eq (List.insertIdx m x bs) a
        hl : Eq as.length bs.length
        hn : LE.le n as.length
        hm : LE.le m bs.length
        hx : Eq (GetElem.getElem (List.insertIdx n x as) m ⋯) x
        hx' : Eq (GetElem.getElem (List.insertIdx m x bs) n ⋯) x
        ⊢ False
      -/
      rcases lt_trichotomy n m with (ht | ht | ht)
        /-
          case cons.right.intro.mk.intro.mk.inl
          α : Type u_1
          s : List α
          x : α
          l : List α
          h : ∀ (a' : α), Membership.mem l a' → Ne x a'
          h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
          IH : l.permutations'.Nodup
          as : List α
          ha : as.Perm l
          bs : List α
          hb : bs.Perm l
          H : Ne as bs
          a : List α
          ha' : Membership.mem (List.permutations'Aux x as) a
          hb' : Membership.mem (List.permutations'Aux x bs) a
          n : Nat
          hn✝ : LT.lt n (List.permutations'Aux x as).length
          hn' : Eq (List.insertIdx n x as) a
          m : Nat
          hm✝ : LT.lt m (List.permutations'Aux x bs).length
          hm' : Eq (List.insertIdx m x bs) a
          hl : Eq as.length bs.length
          hn : LE.le n as.length
          hm : LE.le m bs.length
          hx : Eq (GetElem.getElem (List.insertIdx n x as) m ⋯) x
          hx' : Eq (GetElem.getElem (List.insertIdx m x bs) n ⋯) x
          ht : LT.lt n m
          ⊢ False
        -/
      · suffices x ∈ bs by exact h x (hb.subset this) rfl
        /-
          case cons.right.intro.mk.intro.mk.inl
          α : Type u_1
          s : List α
          x : α
          l : List α
          h : ∀ (a' : α), Membership.mem l a' → Ne x a'
          h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
          IH : l.permutations'.Nodup
          as : List α
          ha : as.Perm l
          bs : List α
          hb : bs.Perm l
          H : Ne as bs
          a : List α
          ha' : Membership.mem (List.permutations'Aux x as) a
          hb' : Membership.mem (List.permutations'Aux x bs) a
          n : Nat
          hn✝ : LT.lt n (List.permutations'Aux x as).length
          hn' : Eq (List.insertIdx n x as) a
          m : Nat
          hm✝ : LT.lt m (List.permutations'Aux x bs).length
          hm' : Eq (List.insertIdx m x bs) a
          hl : Eq as.length bs.length
          hn : LE.le n as.length
          hm : LE.le m bs.length
          hx : Eq (GetElem.getElem (List.insertIdx n x as) m ⋯) x
          hx' : Eq (GetElem.getElem (List.insertIdx m x bs) n ⋯) x
          ht : LT.lt n m
          ⊢ Membership.mem bs x
        -/
        rw [← hx', getElem_insertIdx_of_lt ht]
        /-
          case cons.right.intro.mk.intro.mk.inl
          α : Type u_1
          s : List α
          x : α
          l : List α
          h : ∀ (a' : α), Membership.mem l a' → Ne x a'
          h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
          IH : l.permutations'.Nodup
          as : List α
          ha : as.Perm l
          bs : List α
          hb : bs.Perm l
          H : Ne as bs
          a : List α
          ha' : Membership.mem (List.permutations'Aux x as) a
          hb' : Membership.mem (List.permutations'Aux x bs) a
          n : Nat
          hn✝ : LT.lt n (List.permutations'Aux x as).length
          hn' : Eq (List.insertIdx n x as) a
          m : Nat
          hm✝ : LT.lt m (List.permutations'Aux x bs).length
          hm' : Eq (List.insertIdx m x bs) a
          hl : Eq as.length bs.length
          hn : LE.le n as.length
          hm : LE.le m bs.length
          hx : Eq (GetElem.getElem (List.insertIdx n x as) m ⋯) x
          hx' : Eq (GetElem.getElem (List.insertIdx m x bs) n ⋯) x
          ht : LT.lt n m
          ⊢ Membership.mem bs (GetElem.getElem bs n ⋯)
        -/
        exact getElem_mem _
        /-
          🎉 no goals
        -/
        /-
          case cons.right.intro.mk.intro.mk.inr.inl
          α : Type u_1
          s : List α
          x : α
          l : List α
          h : ∀ (a' : α), Membership.mem l a' → Ne x a'
          h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
          IH : l.permutations'.Nodup
          as : List α
          ha : as.Perm l
          bs : List α
          hb : bs.Perm l
          H : Ne as bs
          a : List α
          ha' : Membership.mem (List.permutations'Aux x as) a
          hb' : Membership.mem (List.permutations'Aux x bs) a
          n : Nat
          hn✝ : LT.lt n (List.permutations'Aux x as).length
          hn' : Eq (List.insertIdx n x as) a
          m : Nat
          hm✝ : LT.lt m (List.permutations'Aux x bs).length
          hm' : Eq (List.insertIdx m x bs) a
          hl : Eq as.length bs.length
          hn : LE.le n as.length
          hm : LE.le m bs.length
          hx : Eq (GetElem.getElem (List.insertIdx n x as) m ⋯) x
          hx' : Eq (GetElem.getElem (List.insertIdx m x bs) n ⋯) x
          ht : Eq n m
          ⊢ False
        -/
      · simp only [ht] at hm' hn'
        /-
          case cons.right.intro.mk.intro.mk.inr.inl
          α : Type u_1
          s : List α
          x : α
          l : List α
          h : ∀ (a' : α), Membership.mem l a' → Ne x a'
          h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
          IH : l.permutations'.Nodup
          as : List α
          ha : as.Perm l
          bs : List α
          hb : bs.Perm l
          H : Ne as bs
          a : List α
          ha' : Membership.mem (List.permutations'Aux x as) a
          hb' : Membership.mem (List.permutations'Aux x bs) a
          n : Nat
          hn✝ : LT.lt n (List.permutations'Aux x as).length
          m : Nat
          hm✝ : LT.lt m (List.permutations'Aux x bs).length
          hm' : Eq (List.insertIdx m x bs) a
          hl : Eq as.length bs.length
          hn : LE.le n as.length
          hm : LE.le m bs.length
          hx : Eq (GetElem.getElem (List.insertIdx n x as) m ⋯) x
          hx' : Eq (GetElem.getElem (List.insertIdx m x bs) n ⋯) x
          ht : Eq n m
          hn' : Eq (List.insertIdx m x as) a
          ⊢ False
        -/
        rw [← hm'] at hn'
        /-
          case cons.right.intro.mk.intro.mk.inr.inl
          α : Type u_1
          s : List α
          x : α
          l : List α
          h : ∀ (a' : α), Membership.mem l a' → Ne x a'
          h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
          IH : l.permutations'.Nodup
          as : List α
          ha : as.Perm l
          bs : List α
          hb : bs.Perm l
          H : Ne as bs
          a : List α
          ha' : Membership.mem (List.permutations'Aux x as) a
          hb' : Membership.mem (List.permutations'Aux x bs) a
          n : Nat
          hn✝ : LT.lt n (List.permutations'Aux x as).length
          m : Nat
          hm✝ : LT.lt m (List.permutations'Aux x bs).length
          hm' : Eq (List.insertIdx m x bs) a
          hl : Eq as.length bs.length
          hn : LE.le n as.length
          hm : LE.le m bs.length
          hx : Eq (GetElem.getElem (List.insertIdx n x as) m ⋯) x
          hx' : Eq (GetElem.getElem (List.insertIdx m x bs) n ⋯) x
          ht : Eq n m
          hn' : Eq (List.insertIdx m x as) (List.insertIdx m x bs)
          ⊢ False
        -/
        exact H (insertIdx_injective _ _ hn')
        /-
          🎉 no goals
        -/
        /-
          case cons.right.intro.mk.intro.mk.inr.inr
          α : Type u_1
          s : List α
          x : α
          l : List α
          h : ∀ (a' : α), Membership.mem l a' → Ne x a'
          h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
          IH : l.permutations'.Nodup
          as : List α
          ha : as.Perm l
          bs : List α
          hb : bs.Perm l
          H : Ne as bs
          a : List α
          ha' : Membership.mem (List.permutations'Aux x as) a
          hb' : Membership.mem (List.permutations'Aux x bs) a
          n : Nat
          hn✝ : LT.lt n (List.permutations'Aux x as).length
          hn' : Eq (List.insertIdx n x as) a
          m : Nat
          hm✝ : LT.lt m (List.permutations'Aux x bs).length
          hm' : Eq (List.insertIdx m x bs) a
          hl : Eq as.length bs.length
          hn : LE.le n as.length
          hm : LE.le m bs.length
          hx : Eq (GetElem.getElem (List.insertIdx n x as) m ⋯) x
          hx' : Eq (GetElem.getElem (List.insertIdx m x bs) n ⋯) x
          ht : LT.lt m n
          ⊢ False
        -/
      · suffices x ∈ as by exact h x (ha.subset this) rfl
        /-
          case cons.right.intro.mk.intro.mk.inr.inr
          α : Type u_1
          s : List α
          x : α
          l : List α
          h : ∀ (a' : α), Membership.mem l a' → Ne x a'
          h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
          IH : l.permutations'.Nodup
          as : List α
          ha : as.Perm l
          bs : List α
          hb : bs.Perm l
          H : Ne as bs
          a : List α
          ha' : Membership.mem (List.permutations'Aux x as) a
          hb' : Membership.mem (List.permutations'Aux x bs) a
          n : Nat
          hn✝ : LT.lt n (List.permutations'Aux x as).length
          hn' : Eq (List.insertIdx n x as) a
          m : Nat
          hm✝ : LT.lt m (List.permutations'Aux x bs).length
          hm' : Eq (List.insertIdx m x bs) a
          hl : Eq as.length bs.length
          hn : LE.le n as.length
          hm : LE.le m bs.length
          hx : Eq (GetElem.getElem (List.insertIdx n x as) m ⋯) x
          hx' : Eq (GetElem.getElem (List.insertIdx m x bs) n ⋯) x
          ht : LT.lt m n
          ⊢ Membership.mem as x
        -/
        rw [← hx, getElem_insertIdx_of_lt ht]
        /-
          case cons.right.intro.mk.intro.mk.inr.inr
          α : Type u_1
          s : List α
          x : α
          l : List α
          h : ∀ (a' : α), Membership.mem l a' → Ne x a'
          h' : List.Pairwise (fun x1 x2 => Ne x1 x2) l
          IH : l.permutations'.Nodup
          as : List α
          ha : as.Perm l
          bs : List α
          hb : bs.Perm l
          H : Ne as bs
          a : List α
          ha' : Membership.mem (List.permutations'Aux x as) a
          hb' : Membership.mem (List.permutations'Aux x bs) a
          n : Nat
          hn✝ : LT.lt n (List.permutations'Aux x as).length
          hn' : Eq (List.insertIdx n x as) a
          m : Nat
          hm✝ : LT.lt m (List.permutations'Aux x bs).length
          hm' : Eq (List.insertIdx m x bs) a
          hl : Eq as.length bs.length
          hn : LE.le n as.length
          hm : LE.le m bs.length
          hx : Eq (GetElem.getElem (List.insertIdx n x as) m ⋯) x
          hx' : Eq (GetElem.getElem (List.insertIdx m x bs) n ⋯) x
          ht : LT.lt m n
          ⊢ Membership.mem as (GetElem.getElem as m ⋯)
        -/
        exact getElem_mem _
        /-
          🎉 no goals
        -/


lemma permutations_take_two (x y : α) (s : List α) :
    (x :: y :: s).permutations.take 2 = [x :: y :: s, y :: x :: s] := by
  /-
    α : Type u_1
    x y : α
    s : List α
    ⊢ Eq (List.take 2 (List.cons x (List.cons y s)).permutations) (List.cons (List …
  -/
                  /-
                    🎉 no goals
                  -/
  induction s <;> simp only [take, permutationsAux, permutationsAux.rec, permutationsAux2, id_eq]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem nodup_permutations_iff {s : List α} : Nodup s.permutations ↔ Nodup s := by
  /-
    α : Type u_1
    s : List α
    ⊢ Iff s.permutations.Nodup s.Nodup
  -/
  refine ⟨?_, nodup_permutations s⟩
  /-
    α : Type u_1
    s : List α
    ⊢ s.permutations.Nodup → s.Nodup
  -/
  contrapose
  /-
    α : Type u_1
    s : List α
    ⊢ Not s.Nodup → Not s.permutations.Nodup
  -/
  rw [← exists_duplicate_iff_not_nodup]
  /-
    α : Type u_1
    s : List α
    ⊢ (Exists fun x => List.Duplicate x s) → Not s.permutations.Nodup
  -/
  intro ⟨x, hs⟩
  /-
    α : Type u_1
    s : List α
    x : α
    hs : List.Duplicate x s
    ⊢ Not s.permutations.Nodup
  -/
  rw [duplicate_iff_sublist] at hs
  /-
    α : Type u_1
    s : List α
    x : α
    hs : (List.cons x (List.cons x List.nil)).Sublist s
    ⊢ Not s.permutations.Nodup
  -/
  obtain ⟨l, ht⟩ := List.Sublist.exists_perm_append hs
  /-
    case intro
    α : Type u_1
    s : List α
    x : α
    hs : (List.cons x (List.cons x List.nil)).Sublist s
    l : List α
    ht : s.Perm (HAppend.hAppend (List.cons x (List.cons x List.nil)) l)
    ⊢ Not s.permutations.Nodup
  -/
  rw [List.Perm.nodup_iff (List.Perm.permutations ht), ← exists_duplicate_iff_not_nodup]
  /-
    case intro
    α : Type u_1
    s : List α
    x : α
    hs : (List.cons x (List.cons x List.nil)).Sublist s
    l : List α
    ht : s.Perm (HAppend.hAppend (List.cons x (List.cons x List.nil)) l)
    ⊢ Exists fun x_1 => List.Duplicate x_1 (HAppend.hAppend (List.cons x (List.con …
  -/
  use x :: x :: l
  /-
    case h
    α : Type u_1
    s : List α
    x : α
    hs : (List.cons x (List.cons x List.nil)).Sublist s
    l : List α
    ht : s.Perm (HAppend.hAppend (List.cons x (List.cons x List.nil)) l)
    ⊢ List.Duplicate (List.cons x (List.cons x l)) (HAppend.hAppend (List.cons x ( …
  -/
  rw [List.duplicate_iff_sublist, ← permutations_take_two]
  /-
    case h
    α : Type u_1
    s : List α
    x : α
    hs : (List.cons x (List.cons x List.nil)).Sublist s
    l : List α
    ht : s.Perm (HAppend.hAppend (List.cons x (List.cons x List.nil)) l)
    ⊢ (List.take 2 (List.cons x (List.cons x l)).permutations).Sublist (HAppend.hA …
  -/
  exact take_sublist 2 _
  /-
    🎉 no goals
  -/

-- TODO: `count s s.permutations = (zipWith count s s.tails).prod`


