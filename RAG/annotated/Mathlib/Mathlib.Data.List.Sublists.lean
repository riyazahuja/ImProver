@[simp]
theorem sublists'_nil : sublists' (@nil α) = [[]] :=
  rfl


@[simp]
theorem sublists'_singleton (a : α) : sublists' [a] = [[], [a]] :=
  rfl

-- Porting note: Not the same as `sublists'_aux` from Lean3

/-- Auxiliary helper definition for `sublists'` -/
def sublists'Aux (a : α) (r₁ r₂ : List (List α)) : List (List α) :=
  r₁.foldl (init := r₂) fun r l => r ++ [a :: l]


theorem sublists'Aux_eq_array_foldl (a : α) : ∀ (r₁ r₂ : List (List α)),
    sublists'Aux a r₁ r₂ = ((r₁.toArray).foldl (init := r₂.toArray)
      (fun r l => r.push (a :: l))).toList := by
  /-
    α : Type u
    a : α
    ⊢ ∀ (r₁ r₂ : List (List α)), Eq (List.sublists'Aux a r₁ r₂) (Array.foldl (fun  …
  -/
  intro r₁ r₂
  /-
    α : Type u
    a : α
    r₁ r₂ : List (List α)
    ⊢ Eq (List.sublists'Aux a r₁ r₂) (Array.foldl (fun r l => r.push (List.cons a  …
  -/
  rw [sublists'Aux, Array.foldl_toList]
  have := List.foldl_hom Array.toList (fun r l => r.push (a :: l))
    (fun r l => r ++ [a :: l]) r₁ r₂.toArray (by simp)
  /-
    α : Type u
    a : α
    r₁ r₂ : List (List α)
    this : Eq (List.foldl (fun r l => HAppend.hAppend r (List.cons (List.cons a l) …
    ⊢ Eq (Array.foldl (fun r l => HAppend.hAppend r (List.cons (List.cons a l) Lis …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


theorem sublists'_eq_sublists'Aux (l : List α) :
    sublists' l = l.foldr (fun a r => sublists'Aux a r r) [[]] := by
  /-
    α : Type u
    l : List α
    ⊢ Eq l.sublists' (List.foldr (fun a r => List.sublists'Aux a r r) (List.cons L …
  -/
  simp only [sublists', sublists'Aux_eq_array_foldl]
  /-
    α : Type u
    l : List α
    ⊢ Eq (List.foldr (fun a arr => Array.foldl (fun r l => r.push (List.cons a l)) …
  -/
  rw [← List.foldr_hom Array.toList]
    /-
      case H
      α : Type u
      l : List α
      ⊢ ∀ (x : α) (y : Array (List α)), Eq (Array.foldl (fun r l => r.push (List.con …
    -/
  · intros _ _; congr
                /-
                  🎉 no goals
                -/


theorem sublists'Aux_eq_map (a : α) (r₁ : List (List α)) : ∀ (r₂ : List (List α)),
    sublists'Aux a r₁ r₂ = r₂ ++ map (cons a) r₁ :=
                                    /-
                                      α : Type u
                                      a : α
                                      r₁ x✝ : List (List α)
                                      ⊢ Eq (List.sublists'Aux a List.nil x✝) (HAppend.hAppend x✝ (List.map (List.con …
                                    -/
  List.reverseRecOn r₁ (fun _ => by simp [sublists'Aux]) fun r₁ l ih r₂ => by
                                    /-
                                      🎉 no goals
                                    -/
    /-
      α : Type u
      a : α
      r₁✝ r₁ : List (List α)
      l : List α
      ih : ∀ (r₂ : List (List α)), Eq (List.sublists'Aux a r₁ r₂) (HAppend.hAppend r …
      r₂ : List (List α)
      ⊢ Eq (List.sublists'Aux a (HAppend.hAppend r₁ (List.cons l List.nil)) r₂) (HAp …
    -/
    rw [map_append, map_singleton, ← append_assoc, ← ih, sublists'Aux, foldl_append, foldl]
    /-
      α : Type u
      a : α
      r₁✝ r₁ : List (List α)
      l : List α
      ih : ∀ (r₂ : List (List α)), Eq (List.sublists'Aux a r₁ r₂) (HAppend.hAppend r …
      r₂ : List (List α)
      ⊢ Eq (List.foldl (fun r l => HAppend.hAppend r (List.cons (List.cons a l) List …
    -/
    simp [sublists'Aux]
    /-
      🎉 no goals
    -/

-- Porting note: simp can prove `sublists'_singleton`

@[simp 900]
theorem sublists'_cons (a : α) (l : List α) :
    sublists' (a :: l) = sublists' l ++ map (cons a) (sublists' l) := by
  /-
    α : Type u
    a : α
    l : List α
    ⊢ Eq (List.cons a l).sublists' (HAppend.hAppend l.sublists' (List.map (List.co …
  -/
  simp [sublists'_eq_sublists'Aux, foldr_cons, sublists'Aux_eq_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_sublists' {s t : List α} : s ∈ sublists' t ↔ s <+ t := by
  /-
    α : Type u
    s t : List α
    ⊢ Iff (Membership.mem t.sublists' s) (s.Sublist t)
  -/
  induction' t with a t IH generalizing s
    /-
      case nil
      α : Type u
      s : List α
      ⊢ Iff (Membership.mem List.nil.sublists' s) (s.Sublist List.nil)
    -/
  · simp only [sublists'_nil, mem_singleton]
    /-
      case nil
      α : Type u
      s : List α
      ⊢ Iff (Eq s List.nil) (s.Sublist List.nil)
    -/
    exact ⟨fun h => by rw [h], eq_nil_of_sublist_nil⟩
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u
    a : α
    t : List α
    IH : ∀ {s : List α}, Iff (Membership.mem t.sublists' s) (s.Sublist t)
    s : List α
    ⊢ Iff (Membership.mem (List.cons a t).sublists' s) (s.Sublist (List.cons a t))
  -/
  simp only [sublists'_cons, mem_append, IH, mem_map]
  /-
    case cons
    α : Type u
    a : α
    t : List α
    IH : ∀ {s : List α}, Iff (Membership.mem t.sublists' s) (s.Sublist t)
    s : List α
    ⊢ Iff (Or (s.Sublist t) (Exists fun a_1 => And (a_1.Sublist t) (Eq (List.cons  …
  -/
  constructor <;> intro h
    /-
      case cons.mp
      α : Type u
      a : α
      t : List α
      IH : ∀ {s : List α}, Iff (Membership.mem t.sublists' s) (s.Sublist t)
      s : List α
      h : Or (s.Sublist t) (Exists fun a_1 => And (a_1.Sublist t) (Eq (List.cons a a …
      ⊢ s.Sublist (List.cons a t)
    -/
  · rcases h with (h | ⟨s, h, rfl⟩)
      /-
        case cons.mp.inl
        α : Type u
        a : α
        t : List α
        IH : ∀ {s : List α}, Iff (Membership.mem t.sublists' s) (s.Sublist t)
        s : List α
        h : s.Sublist t
        ⊢ s.Sublist (List.cons a t)
      -/
    · exact sublist_cons_of_sublist _ h
      /-
        🎉 no goals
      -/
      /-
        case cons.mp.inr.intro.intro
        α : Type u
        a : α
        t : List α
        IH : ∀ {s : List α}, Iff (Membership.mem t.sublists' s) (s.Sublist t)
        s : List α
        h : s.Sublist t
        ⊢ (List.cons a s).Sublist (List.cons a t)
      -/
    · exact h.cons_cons _
      /-
        🎉 no goals
      -/
    /-
      case cons.mpr
      α : Type u
      a : α
      t : List α
      IH : ∀ {s : List α}, Iff (Membership.mem t.sublists' s) (s.Sublist t)
      s : List α
      h : s.Sublist (List.cons a t)
      ⊢ Or (s.Sublist t) (Exists fun a_1 => And (a_1.Sublist t) (Eq (List.cons a a_1 …
    -/
  · cases' h with _ _ _ h s _ _ h
      /-
        case cons.mpr.cons
        α : Type u
        a : α
        t : List α
        IH : ∀ {s : List α}, Iff (Membership.mem t.sublists' s) (s.Sublist t)
        s : List α
        h : s.Sublist t
        ⊢ Or (s.Sublist t) (Exists fun a_1 => And (a_1.Sublist t) (Eq (List.cons a a_1 …
      -/
    · exact Or.inl h
      /-
        🎉 no goals
      -/
      /-
        case cons.mpr.cons₂
        α : Type u
        a : α
        t : List α
        IH : ∀ {s : List α}, Iff (Membership.mem t.sublists' s) (s.Sublist t)
        s : List α
        h : s.Sublist t
        ⊢ Or ((List.cons a s).Sublist t) (Exists fun a_1 => And (a_1.Sublist t) (Eq (L …
      -/
    · exact Or.inr ⟨s, h, rfl⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem length_sublists' : ∀ l : List α, length (sublists' l) = 2 ^ length l
  | [] => rfl
  | a :: l => by
    simp_arith only [sublists'_cons, length_append, length_sublists' l,
      length_map, length, Nat.pow_succ']


@[simp]
theorem sublists_nil : sublists (@nil α) = [[]] :=
  rfl


@[simp]
theorem sublists_singleton (a : α) : sublists [a] = [[], [a]] :=
  rfl

-- Porting note: Not the same as `sublists_aux` from Lean3

/-- Auxiliary helper function for `sublists` -/
def sublistsAux (a : α) (r : List (List α)) : List (List α) :=
  r.foldl (init := []) fun r l => r ++ [l, a :: l]


theorem sublistsAux_eq_array_foldl :
    sublistsAux = fun (a : α) (r : List (List α)) =>
      (r.toArray.foldl (init := #[])
        fun r l => (r.push l).push (a :: l)).toList := by
  /-
    α : Type u
    ⊢ Eq List.sublistsAux fun a r => (Array.foldl (fun r l => (r.push l).push (Lis …
  -/
  funext a r
  /-
    case h.h
    α : Type u
    a : α
    r : List (List α)
    ⊢ Eq (List.sublistsAux a r) (Array.foldl (fun r l => (r.push l).push (List.con …
  -/
  simp only [sublistsAux, Array.foldl_toList, Array.mkEmpty]
  have := foldl_hom Array.toList (fun r l => (r.push l).push (a :: l))
    (fun (r : List (List α)) l => r ++ [l, a :: l]) r #[]
    (by simp)
  /-
    case h.h
    α : Type u
    a : α
    r : List (List α)
    this : Eq (List.foldl (fun r l => HAppend.hAppend r (List.cons l (List.cons (L …
    ⊢ Eq (List.foldl (fun r l => HAppend.hAppend r (List.cons l (List.cons (List.c …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


theorem sublistsAux_eq_flatMap :
    sublistsAux = fun (a : α) (r : List (List α)) => r.flatMap fun l => [l, a :: l] :=
  funext fun a => funext fun r =>
  List.reverseRecOn r
        /-
          α : Type u
          a : α
          r : List (List α)
          ⊢ Eq (List.sublistsAux a List.nil) (List.nil.flatMap fun l => List.cons l (Lis …
        -/
    (by simp [sublistsAux])
        /-
          🎉 no goals
        -/
    (fun r l ih => by
      /-
        α : Type u
        a : α
        r✝ r : List (List α)
        l : List α
        ih : Eq (List.sublistsAux a r) (r.flatMap fun l => List.cons l (List.cons (Lis …
        ⊢ Eq (List.sublistsAux a (HAppend.hAppend r (List.cons l List.nil))) ((HAppend …
      -/
      rw [flatMap_append, ← ih, flatMap_singleton, sublistsAux, foldl_append]
      /-
        α : Type u
        a : α
        r✝ r : List (List α)
        l : List α
        ih : Eq (List.sublistsAux a r) (r.flatMap fun l => List.cons l (List.cons (Lis …
        ⊢ Eq (List.foldl (fun r l => HAppend.hAppend r (List.cons l (List.cons (List.c …
      -/
      simp [sublistsAux])
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-16")] alias sublistsAux_eq_bind := sublistsAux_eq_flatMap


@[csimp] theorem sublists_eq_sublistsFast : @sublists = @sublistsFast := by
  /-
    ⊢ Eq @List.sublists @List.sublistsFast
  -/
  ext α l : 2
  /-
    case h.h
    α : Type u_1
    l : List α
    ⊢ Eq l.sublists l.sublistsFast
  -/
  trans l.foldr sublistsAux [[]]
    /-
      α : Type u_1
      l : List α
      ⊢ Eq l.sublists (List.foldr List.sublistsAux (List.cons List.nil List.nil) l)
    -/
  · rw [sublistsAux_eq_flatMap, sublists]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      l : List α
      ⊢ Eq (List.foldr List.sublistsAux (List.cons List.nil List.nil) l) l.sublistsF …
    -/
  · simp only [sublistsFast, sublistsAux_eq_array_foldl, Array.foldr_toList]
    /-
      α : Type u_1
      l : List α
      ⊢ Eq (List.foldr (fun a r => (Array.foldl (fun r l => (r.push l).push (List.co …
    -/
    rw [← foldr_hom Array.toList]
      /-
        case H
        α : Type u_1
        l : List α
        ⊢ ∀ (x : α) (y : Array (List α)), Eq (Array.foldl (fun r l => (r.push l).push  …
      -/
    · intros _ _; congr
                  /-
                    🎉 no goals
                  -/


theorem sublists_append (l₁ l₂ : List α) :
    sublists (l₁ ++ l₂) = (sublists l₂) >>= (fun x => (sublists l₁).map (· ++ x)) := by
  /-
    α : Type u
    l₁ l₂ : List α
    ⊢ Eq (HAppend.hAppend l₁ l₂).sublists (Bind.bind l₂.sublists fun x => List.map …
  -/
  simp only [sublists, foldr_append]
  induction l₁ with
  | nil => simp
  | cons a l₁ ih =>
    rw [foldr_cons, ih]
    simp [List.flatMap, flatten_flatten, Function.comp_def]


theorem sublists_cons (a : α) (l : List α) :
    sublists (a :: l) = sublists l >>= (fun x => [x, a :: x]) :=
  show sublists ([a] ++ l) = _ by
  /-
    α : Type u
    a : α
    l : List α
    ⊢ Eq (HAppend.hAppend (List.cons a List.nil) l).sublists (Bind.bind l.sublists …
  -/
  rw [sublists_append]
  /-
    α : Type u
    a : α
    l : List α
    ⊢ Eq (Bind.bind l.sublists fun x => List.map (fun x_1 => HAppend.hAppend x_1 x …
  -/
  simp only [sublists_singleton, map_cons, bind_eq_flatMap, nil_append, cons_append, map_nil]
  /-
    🎉 no goals
  -/


@[simp]
theorem sublists_concat (l : List α) (a : α) :
    sublists (l ++ [a]) = sublists l ++ map (fun x => x ++ [a]) (sublists l) := by
  rw [sublists_append, sublists_singleton, bind_eq_flatMap, flatMap_cons, flatMap_cons, flatMap_nil,
     map_id'' append_nil, append_nil]


theorem sublists_reverse (l : List α) : sublists (reverse l) = map reverse (sublists' l) := by
  induction' l with hd tl ih <;> [rfl;
    simp only [reverse_cons, sublists_append, sublists'_cons, map_append, ih, sublists_singleton,
      map_eq_map, bind_eq_flatMap, map_map, flatMap_cons, append_nil, flatMap_nil,
      Function.comp_def]]


theorem sublists_eq_sublists' (l : List α) : sublists l = map reverse (sublists' (reverse l)) := by
  /-
    α : Type u
    l : List α
    ⊢ Eq l.sublists (List.map List.reverse l.reverse.sublists')
  -/
  rw [← sublists_reverse, reverse_reverse]
  /-
    🎉 no goals
  -/


theorem sublists'_reverse (l : List α) : sublists' (reverse l) = map reverse (sublists l) := by
  /-
    α : Type u
    l : List α
    ⊢ Eq l.reverse.sublists' (List.map List.reverse l.sublists)
  -/
  simp only [sublists_eq_sublists', map_map, map_id'' reverse_reverse, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem sublists'_eq_sublists (l : List α) : sublists' l = map reverse (sublists (reverse l)) := by
  /-
    α : Type u
    l : List α
    ⊢ Eq l.sublists' (List.map List.reverse l.reverse.sublists)
  -/
  rw [← sublists'_reverse, reverse_reverse]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_sublists {s t : List α} : s ∈ sublists t ↔ s <+ t := by
  rw [← reverse_sublist, ← mem_sublists', sublists'_reverse,
    mem_map_of_injective reverse_injective]


@[simp]
theorem length_sublists (l : List α) : length (sublists l) = 2 ^ length l := by
  /-
    α : Type u
    l : List α
    ⊢ Eq l.sublists.length (HPow.hPow 2 l.length)
  -/
  simp only [sublists_eq_sublists', length_map, length_sublists', length_reverse]
  /-
    🎉 no goals
  -/


theorem map_pure_sublist_sublists (l : List α) : map pure l <+ sublists l := by
  /-
    α : Type u
    l : List α
    ⊢ (List.map Pure.pure l).Sublist l.sublists
  -/
  induction' l using reverseRecOn with l a ih <;> simp only [map, map_append, sublists_concat]
    /-
      case nil
      α : Type u
      ⊢ List.nil.Sublist List.nil.sublists
    -/
  · simp only [sublists_nil, sublist_cons_self]
    /-
      🎉 no goals
    -/
  exact ((append_sublist_append_left _).2 <|
              singleton_sublist.2 <| mem_map.2 ⟨[], mem_sublists.2 (nil_sublist _), by rfl⟩).trans
          ((append_sublist_append_right _).2 ih)


set_option linter.deprecated false in
@[deprecated map_pure_sublist_sublists (since := "2024-03-24")]
theorem map_ret_sublist_sublists (l : List α) : map List.ret l <+ sublists l :=
  map_pure_sublist_sublists l


/-- Auxiliary function to construct the list of all sublists of a given length. Given an
integer `n`, a list `l`, a function `f` and an auxiliary list `L`, it returns the list made of
`f` applied to all sublists of `l` of length `n`, concatenated with `L`. -/
def sublistsLenAux : ℕ → List α → (List α → β) → List β → List β
  | 0, _, f, r => f [] :: r
  | _ + 1, [], _, r => r
  | n + 1, a :: l, f, r => sublistsLenAux (n + 1) l f (sublistsLenAux n l (f ∘ List.cons a) r)


/-- The list of all sublists of a list `l` that are of length `n`. For instance, for
`l = [0, 1, 2, 3]` and `n = 2`, one gets
`[[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]`. -/
def sublistsLen (n : ℕ) (l : List α) : List (List α) :=
  sublistsLenAux n l id []


theorem sublistsLenAux_append :
    ∀ (n : ℕ) (l : List α) (f : List α → β) (g : β → γ) (r : List β) (s : List γ),
      sublistsLenAux n l (g ∘ f) (r.map g ++ s) = (sublistsLenAux n l f r).map g ++ s
                           /-
                             α : Type u
                             β : Type v
                             γ : Type w
                             l : List α
                             f : List α → β
                             g : β → γ
                             r : List β
                             s : List γ
                             ⊢ Eq (List.sublistsLenAux 0 l (Function.comp g f) (HAppend.hAppend (List.map g …
                           -/
  | 0, l, f, g, r, s => by unfold sublistsLenAux; simp
                                                  /-
                                                    🎉 no goals
                                                  -/
  | _ + 1, [], _, _, _, _ => rfl
  | n + 1, a :: l, f, g, r, s => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      n : Nat
      a : α
      l : List α
      f : List α → β
      g : β → γ
      r : List β
      s : List γ
      ⊢ Eq (List.sublistsLenAux (HAdd.hAdd n 1) (List.cons a l) (Function.comp g f)  …
    -/
    unfold sublistsLenAux
    simp only [show (g ∘ f) ∘ List.cons a = g ∘ f ∘ List.cons a by rfl, sublistsLenAux_append,
      sublistsLenAux_append]


theorem sublistsLenAux_eq (l : List α) (n) (f : List α → β) (r) :
    sublistsLenAux n l f r = (sublistsLen n l).map f ++ r := by
  /-
    α : Type u
    β : Type v
    l : List α
    n : Nat
    f : List α → β
    r : List β
    ⊢ Eq (List.sublistsLenAux n l f r) (HAppend.hAppend (List.map f (List.sublists …
  -/
  rw [sublistsLen, ← sublistsLenAux_append]; rfl
                                             /-
                                               🎉 no goals
                                             -/


theorem sublistsLenAux_zero (l : List α) (f : List α → β) (r) :
                                             /-
                                               α : Type u
                                               β : Type v
                                               l : List α
                                               f : List α → β
                                               r : List β
                                               ⊢ Eq (List.sublistsLenAux 0 l f r) (List.cons (f List.nil) r)
                                             -/
                                                         /-
                                                           🎉 no goals
                                                         -/
    sublistsLenAux 0 l f r = f [] :: r := by cases l <;> rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem sublistsLen_zero (l : List α) : sublistsLen 0 l = [[]] :=
  sublistsLenAux_zero _ _ _


@[simp]
theorem sublistsLen_succ_nil (n) : sublistsLen (n + 1) (@nil α) = [] :=
  rfl


@[simp]
theorem sublistsLen_succ_cons (n) (a : α) (l) :
    sublistsLen (n + 1) (a :: l) = sublistsLen (n + 1) l ++ (sublistsLen n l).map (cons a) := by
  rw [sublistsLen, sublistsLenAux, sublistsLenAux_eq, sublistsLenAux_eq, map_id,
                   /-
                     α : Type u
                     n : Nat
                     a : α
                     l : List α
                     ⊢ Eq (HAppend.hAppend (List.sublistsLen (HAdd.hAdd n 1) l) (List.map (Function …
                   -/
      append_nil]; rfl
                   /-
                     🎉 no goals
                   -/


theorem sublistsLen_one (l : List α) : sublistsLen 1 l = l.reverse.map ([·]) :=
            /-
              α : Type u
              l : List α
              ⊢ Eq (List.sublistsLen 1 List.nil) (List.map (fun x => List.cons x List.nil) L …
            -/
  l.rec (by rw [sublistsLen_succ_nil, reverse_nil, map_nil]) fun a s ih ↦ by
            /-
              🎉 no goals
            -/
    /-
      α : Type u
      l : List α
      a : α
      s : List α
      ih : Eq (List.sublistsLen 1 s) (List.map (fun x => List.cons x List.nil) s.rev …
      ⊢ Eq (List.sublistsLen 1 (List.cons a s)) (List.map (fun x => List.cons x List …
    -/
    rw [sublistsLen_succ_cons, ih, reverse_cons, map_append, sublistsLen_zero]; rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
theorem length_sublistsLen :
    ∀ (n) (l : List α), length (sublistsLen n l) = Nat.choose (length l) n
               /-
                 α : Type u
                 l : List α
                 ⊢ Eq (List.sublistsLen 0 l).length (l.length.choose 0)
               -/
  | 0, l => by simp
               /-
                 🎉 no goals
               -/
                    /-
                      α : Type u
                      n✝ : Nat
                      ⊢ Eq (List.sublistsLen (HAdd.hAdd n✝ 1) List.nil).length (List.nil.length.choo …
                    -/
  | _ + 1, [] => by simp
                    /-
                      🎉 no goals
                    -/
  | n + 1, a :: l => by
    rw [sublistsLen_succ_cons, length_append, length_sublistsLen (n+1) l,
      length_map, length_sublistsLen n l, length_cons, Nat.choose_succ_succ, Nat.add_comm]


theorem sublistsLen_sublist_sublists' :
    ∀ (n) (l : List α), sublistsLen n l <+ sublists' l
               /-
                 α : Type u
                 l : List α
                 ⊢ (List.sublistsLen 0 l).Sublist l.sublists'
               -/
  | 0, l => by simp
               /-
                 🎉 no goals
               -/
  | _ + 1, [] => nil_sublist _
  | n + 1, a :: l => by
    /-
      α : Type u
      n : Nat
      a : α
      l : List α
      ⊢ (List.sublistsLen (HAdd.hAdd n 1) (List.cons a l)).Sublist (List.cons a l).s …
    -/
    rw [sublistsLen_succ_cons, sublists'_cons]
    /-
      α : Type u
      n : Nat
      a : α
      l : List α
      ⊢ (HAppend.hAppend (List.sublistsLen (HAdd.hAdd n 1) l) (List.map (List.cons a …
    -/
    exact (sublistsLen_sublist_sublists' _ _).append ((sublistsLen_sublist_sublists' _ _).map _)
    /-
      🎉 no goals
    -/


theorem sublistsLen_sublist_of_sublist (n) {l₁ l₂ : List α} (h : l₁ <+ l₂) :
    sublistsLen n l₁ <+ sublistsLen n l₂ := by
  /-
    α : Type u
    n : Nat
    l₁ l₂ : List α
    h : l₁.Sublist l₂
    ⊢ (List.sublistsLen n l₁).Sublist (List.sublistsLen n l₂)
  -/
  induction' n with n IHn generalizing l₁ l₂; · simp
                                                /-
                                                  🎉 no goals
                                                -/
  /-
    case succ
    α : Type u
    n : Nat
    IHn : ∀ {l₁ l₂ : List α}, l₁.Sublist l₂ → (List.sublistsLen n l₁).Sublist (Lis …
    l₁ l₂ : List α
    h : l₁.Sublist l₂
    ⊢ (List.sublistsLen (HAdd.hAdd n 1) l₁).Sublist (List.sublistsLen (HAdd.hAdd n …
  -/
  induction' h with l₁ l₂ a _ IH l₁ l₂ a s IH; · rfl
                                                 /-
                                                   🎉 no goals
                                                 -/
    /-
      case succ.cons
      α : Type u
      n : Nat
      IHn : ∀ {l₁ l₂ : List α}, l₁.Sublist l₂ → (List.sublistsLen n l₁).Sublist (Lis …
      l₁✝ l₂✝ l₁ l₂ : List α
      a : α
      a✝ : l₁.Sublist l₂
      IH : (List.sublistsLen (HAdd.hAdd n 1) l₁).Sublist (List.sublistsLen (HAdd.hAd …
      ⊢ (List.sublistsLen (HAdd.hAdd n 1) l₁).Sublist (List.sublistsLen (HAdd.hAdd n …
    -/
  · refine IH.trans ?_
    /-
      case succ.cons
      α : Type u
      n : Nat
      IHn : ∀ {l₁ l₂ : List α}, l₁.Sublist l₂ → (List.sublistsLen n l₁).Sublist (Lis …
      l₁✝ l₂✝ l₁ l₂ : List α
      a : α
      a✝ : l₁.Sublist l₂
      IH : (List.sublistsLen (HAdd.hAdd n 1) l₁).Sublist (List.sublistsLen (HAdd.hAd …
      ⊢ (List.sublistsLen (HAdd.hAdd n 1) l₂).Sublist (List.sublistsLen (HAdd.hAdd n …
    -/
    rw [sublistsLen_succ_cons]
    /-
      case succ.cons
      α : Type u
      n : Nat
      IHn : ∀ {l₁ l₂ : List α}, l₁.Sublist l₂ → (List.sublistsLen n l₁).Sublist (Lis …
      l₁✝ l₂✝ l₁ l₂ : List α
      a : α
      a✝ : l₁.Sublist l₂
      IH : (List.sublistsLen (HAdd.hAdd n 1) l₁).Sublist (List.sublistsLen (HAdd.hAd …
      ⊢ (List.sublistsLen (HAdd.hAdd n 1) l₂).Sublist (HAppend.hAppend (List.sublist …
    -/
    apply sublist_append_left
    /-
      🎉 no goals
    -/
    /-
      case succ.cons₂
      α : Type u
      n : Nat
      IHn : ∀ {l₁ l₂ : List α}, l₁.Sublist l₂ → (List.sublistsLen n l₁).Sublist (Lis …
      l₁✝ l₂✝ l₁ l₂ : List α
      a : α
      s : l₁.Sublist l₂
      IH : (List.sublistsLen (HAdd.hAdd n 1) l₁).Sublist (List.sublistsLen (HAdd.hAd …
      ⊢ (List.sublistsLen (HAdd.hAdd n 1) (List.cons a l₁)).Sublist (List.sublistsLe …
    -/
  · simpa only [sublistsLen_succ_cons] using IH.append ((IHn s).map _)
    /-
      🎉 no goals
    -/


theorem length_of_sublistsLen :
    ∀ {n} {l l' : List α}, l' ∈ sublistsLen n l → length l' = n
                      /-
                        α : Type u
                        l l' : List α
                        h : Membership.mem (List.sublistsLen 0 l) l'
                        ⊢ Eq l'.length 0
                      -/
  | 0, l, l', h => by simp_all
                      /-
                        🎉 no goals
                      -/
  | n + 1, a :: l, l', h => by
    /-
      α : Type u
      n : Nat
      a : α
      l l' : List α
      h : Membership.mem (List.sublistsLen (HAdd.hAdd n 1) (List.cons a l)) l'
      ⊢ Eq l'.length (HAdd.hAdd n 1)
    -/
    rw [sublistsLen_succ_cons, mem_append, mem_map] at h
    /-
      α : Type u
      n : Nat
      a : α
      l l' : List α
      h : Or (Membership.mem (List.sublistsLen (HAdd.hAdd n 1) l) l') (Exists fun a_ …
      ⊢ Eq l'.length (HAdd.hAdd n 1)
    -/
    rcases h with (h | ⟨l', h, rfl⟩)
      /-
        case inl
        α : Type u
        n : Nat
        a : α
        l l' : List α
        h : Membership.mem (List.sublistsLen (HAdd.hAdd n 1) l) l'
        ⊢ Eq l'.length (HAdd.hAdd n 1)
      -/
    · exact length_of_sublistsLen h
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro
        α : Type u
        n : Nat
        a : α
        l l' : List α
        h : Membership.mem (List.sublistsLen n l) l'
        ⊢ Eq (List.cons a l').length (HAdd.hAdd n 1)
      -/
    · exact congr_arg (· + 1) (length_of_sublistsLen h)
      /-
        🎉 no goals
      -/


theorem mem_sublistsLen_self {l l' : List α} (h : l' <+ l) :
    l' ∈ sublistsLen (length l') l := by
  /-
    α : Type u
    l l' : List α
    h : l'.Sublist l
    ⊢ Membership.mem (List.sublistsLen l'.length l) l'
  -/
  induction' h with l₁ l₂ a s IH l₁ l₂ a s IH
    /-
      case slnil
      α : Type u
      l l' : List α
      ⊢ Membership.mem (List.sublistsLen List.nil.length List.nil) List.nil
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      l l' l₁ l₂ : List α
      a : α
      s : l₁.Sublist l₂
      IH : Membership.mem (List.sublistsLen l₁.length l₂) l₁
      ⊢ Membership.mem (List.sublistsLen l₁.length (List.cons a l₂)) l₁
    -/
  · cases' l₁ with b l₁
      /-
        case cons.nil
        α : Type u
        l l' l₂ : List α
        a : α
        s : List.nil.Sublist l₂
        IH : Membership.mem (List.sublistsLen List.nil.length l₂) List.nil
        ⊢ Membership.mem (List.sublistsLen List.nil.length (List.cons a l₂)) List.nil
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case cons.cons
        α : Type u
        l l' l₂ : List α
        a b : α
        l₁ : List α
        s : (List.cons b l₁).Sublist l₂
        IH : Membership.mem (List.sublistsLen (List.cons b l₁).length l₂) (List.cons b …
        ⊢ Membership.mem (List.sublistsLen (List.cons b l₁).length (List.cons a l₂)) ( …
      -/
    · rw [length, sublistsLen_succ_cons]
      /-
        case cons.cons
        α : Type u
        l l' l₂ : List α
        a b : α
        l₁ : List α
        s : (List.cons b l₁).Sublist l₂
        IH : Membership.mem (List.sublistsLen (List.cons b l₁).length l₂) (List.cons b …
        ⊢ Membership.mem (HAppend.hAppend (List.sublistsLen (HAdd.hAdd l₁.length 1) l₂ …
      -/
      exact mem_append_left _ IH
      /-
        🎉 no goals
      -/
    /-
      case cons₂
      α : Type u
      l l' l₁ l₂ : List α
      a : α
      s : l₁.Sublist l₂
      IH : Membership.mem (List.sublistsLen l₁.length l₂) l₁
      ⊢ Membership.mem (List.sublistsLen (List.cons a l₁).length (List.cons a l₂)) ( …
    -/
  · rw [length, sublistsLen_succ_cons]
    /-
      case cons₂
      α : Type u
      l l' l₁ l₂ : List α
      a : α
      s : l₁.Sublist l₂
      IH : Membership.mem (List.sublistsLen l₁.length l₂) l₁
      ⊢ Membership.mem (HAppend.hAppend (List.sublistsLen (HAdd.hAdd l₁.length 1) l₂ …
    -/
    exact mem_append_right _ (mem_map.2 ⟨_, IH, rfl⟩)
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_sublistsLen {n} {l l' : List α} :
    l' ∈ sublistsLen n l ↔ l' <+ l ∧ length l' = n :=
  ⟨fun h =>
    ⟨mem_sublists'.1 ((sublistsLen_sublist_sublists' _ _).subset h), length_of_sublistsLen h⟩,
    fun ⟨h₁, h₂⟩ => h₂ ▸ mem_sublistsLen_self h₁⟩


theorem sublistsLen_of_length_lt {n} {l : List α} (h : l.length < n) : sublistsLen n l = [] :=
  eq_nil_iff_forall_not_mem.mpr fun _ =>
    mem_sublistsLen.not.mpr fun ⟨hs, hl⟩ => (h.trans_eq hl.symm).not_le (Sublist.length_le hs)


@[simp]
theorem sublistsLen_length : ∀ l : List α, sublistsLen l.length l = [l]
  | [] => rfl
  | a :: l => by
    simp only [length, sublistsLen_succ_cons, sublistsLen_length, map,
      sublistsLen_of_length_lt (lt_succ_self _), nil_append]


theorem Pairwise.sublists' {R} :
    ∀ {l : List α}, Pairwise R l → Pairwise (Lex (swap R)) (sublists' l)
  | _, Pairwise.nil => pairwise_singleton _ _
  | _, @Pairwise.cons _ _ a l H₁ H₂ => by
    simp only [sublists'_cons, pairwise_append, pairwise_map, mem_sublists', mem_map, exists_imp,
      and_imp]
    /-
      α : Type u
      R : α → α → Prop
      a : α
      l : List α
      H₁ : ∀ (a' : α), Membership.mem l a' → R a a'
      H₂ : List.Pairwise R l
      ⊢ And (List.Pairwise (List.Lex (Function.swap R)) l.sublists') (And (List.Pair …
    -/
    refine ⟨H₂.sublists', H₂.sublists'.imp fun l₁ => Lex.cons l₁, ?_⟩
    /-
      α : Type u
      R : α → α → Prop
      a : α
      l : List α
      H₁ : ∀ (a' : α), Membership.mem l a' → R a a'
      H₂ : List.Pairwise R l
      ⊢ ∀ (a_1 : List α), a_1.Sublist l → ∀ (b x : List α), x.Sublist l → Eq (List.c …
    -/
    rintro l₁ sl₁ x l₂ _ rfl
    /-
      α : Type u
      R : α → α → Prop
      a : α
      l : List α
      H₁ : ∀ (a' : α), Membership.mem l a' → R a a'
      H₂ : List.Pairwise R l
      l₁ : List α
      sl₁ : l₁.Sublist l
      l₂ : List α
      a✝ : l₂.Sublist l
      ⊢ List.Lex (Function.swap R) l₁ (List.cons a l₂)
    -/
    cases' l₁ with b l₁; · constructor
                           /-
                             🎉 no goals
                           -/
    /-
      case cons
      α : Type u
      R : α → α → Prop
      a : α
      l : List α
      H₁ : ∀ (a' : α), Membership.mem l a' → R a a'
      H₂ : List.Pairwise R l
      l₂ : List α
      a✝ : l₂.Sublist l
      b : α
      l₁ : List α
      sl₁ : (List.cons b l₁).Sublist l
      ⊢ List.Lex (Function.swap R) (List.cons b l₁) (List.cons a l₂)
    -/
    exact Lex.rel (H₁ _ <| sl₁.subset <| mem_cons_self _ _)
    /-
      🎉 no goals
    -/


theorem pairwise_sublists {R} {l : List α} (H : Pairwise R l) :
    Pairwise (Lex R on reverse) (sublists l) := by
  /-
    α : Type u
    R : α → α → Prop
    l : List α
    H : List.Pairwise R l
    ⊢ List.Pairwise (Function.onFun (List.Lex R) List.reverse) l.sublists
  -/
  have := (pairwise_reverse.2 H).sublists'
  /-
    α : Type u
    R : α → α → Prop
    l : List α
    H : List.Pairwise R l
    this : List.Pairwise (List.Lex (Function.swap fun b a => R a b)) l.reverse.sub …
    ⊢ List.Pairwise (Function.onFun (List.Lex R) List.reverse) l.sublists
  -/
  rwa [sublists'_reverse, pairwise_map] at this
  /-
    🎉 no goals
  -/


@[simp]
theorem nodup_sublists {l : List α} : Nodup (sublists l) ↔ Nodup l :=
  ⟨fun h => (h.sublist (map_pure_sublist_sublists _)).of_map _, fun h =>
                                                 /-
                                                   α : Type u
                                                   l : List α
                                                   h✝ : l.Nodup
                                                   l₁ l₂ : List α
                                                   h : Function.onFun (List.Lex fun x1 x2 => Ne x1 x2) List.reverse l₁ l₂
                                                   ⊢ Ne l₁ l₂
                                                 -/
    (pairwise_sublists h).imp @fun l₁ l₂ h => by simpa using h.to_ne⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem nodup_sublists' {l : List α} : Nodup (sublists' l) ↔ Nodup l := by
  /-
    α : Type u
    l : List α
    ⊢ Iff l.sublists'.Nodup l.Nodup
  -/
  rw [sublists'_eq_sublists, nodup_map_iff reverse_injective, nodup_sublists, nodup_reverse]
  /-
    🎉 no goals
  -/


protected alias ⟨Nodup.of_sublists, Nodup.sublists⟩ := nodup_sublists


protected alias ⟨Nodup.of_sublists', _⟩ := nodup_sublists'


theorem nodup_sublistsLen (n : ℕ) {l : List α} (h : Nodup l) : (sublistsLen n l).Nodup := by
  have : Pairwise (· ≠ ·) l.sublists' := Pairwise.imp
    (fun h => Lex.to_ne (by convert h using 3; simp [swap, eq_comm])) h.sublists'
  /-
    α : Type u
    n : Nat
    l : List α
    h : l.Nodup
    this : List.Pairwise (fun x1 x2 => Ne x1 x2) l.sublists'
    ⊢ (List.sublistsLen n l).Nodup
  -/
  exact this.sublist (sublistsLen_sublist_sublists' _ _)
  /-
    🎉 no goals
  -/


theorem sublists_map (f : α → β) : ∀ (l : List α),
    sublists (map f l) = map (map f) (sublists l)
             /-
               α : Type u
               β : Type v
               f : α → β
               ⊢ Eq (List.map f List.nil).sublists (List.map (List.map f) List.nil.sublists)
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | a::l => by
    rw [map_cons, sublists_cons, bind_eq_flatMap, sublists_map f l, sublists_cons,
      bind_eq_flatMap, map_eq_flatMap, map_eq_flatMap]
    /-
      α : Type u
      β : Type v
      f : α → β
      a : α
      l : List α
      ⊢ Eq ((l.sublists.flatMap fun x => List.cons (List.map f x) List.nil).flatMap  …
    -/
                             /-
                               🎉 no goals
                             -/
    induction sublists l <;> simp [*]
                             /-
                               🎉 no goals
                             -/


theorem sublists'_map (f : α → β) : ∀ (l : List α),
    sublists' (map f l) = map (map f) (sublists' l)
             /-
               α : Type u
               β : Type v
               f : α → β
               ⊢ Eq (List.map f List.nil).sublists' (List.map (List.map f) List.nil.sublists')
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
               /-
                 α : Type u
                 β : Type v
                 f : α → β
                 a : α
                 l : List α
                 ⊢ Eq (List.map f (List.cons a l)).sublists' (List.map (List.map f) (List.cons  …
               -/
  | a::l => by simp [map_cons, sublists'_cons, sublists'_map f l, Function.comp]
               /-
                 🎉 no goals
               -/

-- Porting note: moved because it is now used to prove `sublists_cons_perm_append`

theorem sublists_perm_sublists' (l : List α) : sublists l ~ sublists' l := by
  /-
    α : Type u
    l : List α
    ⊢ l.sublists.Perm l.sublists'
  -/
  rw [← finRange_map_get l, sublists_map, sublists'_map]
  /-
    α : Type u
    l : List α
    ⊢ (List.map (List.map l.get) (List.finRange l.length).sublists).Perm (List.map …
  -/
  apply Perm.map
  /-
    case p
    α : Type u
    l : List α
    ⊢ (List.finRange l.length).sublists.Perm (List.finRange l.length).sublists'
  -/
  apply (perm_ext_iff_of_nodup _ _).mpr
    /-
      case p
      α : Type u
      l : List α
      ⊢ ∀ (a : List (Fin l.length)), Iff (Membership.mem (List.finRange l.length).su …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      l : List α
      ⊢ (List.finRange l.length).sublists.Nodup
    -/
  · exact nodup_sublists.mpr (nodup_finRange _)
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      l : List α
      ⊢ (List.finRange l.length).sublists'.Nodup
    -/
  · exact (nodup_sublists'.mpr (nodup_finRange _))
    /-
      🎉 no goals
    -/


theorem sublists_cons_perm_append (a : α) (l : List α) :
    sublists (a :: l) ~ sublists l ++ map (cons a) (sublists l) :=
  Perm.trans (sublists_perm_sublists' _) <| by
  /-
    α : Type u
    a : α
    l : List α
    ⊢ (List.cons a l).sublists'.Perm (HAppend.hAppend l.sublists (List.map (List.c …
  -/
  rw [sublists'_cons]
  /-
    α : Type u
    a : α
    l : List α
    ⊢ (HAppend.hAppend l.sublists' (List.map (List.cons a) l.sublists')).Perm (HAp …
  -/
  exact Perm.append (sublists_perm_sublists' _).symm (Perm.map _ (sublists_perm_sublists' _).symm)
  /-
    🎉 no goals
  -/


theorem revzip_sublists (l : List α) : ∀ l₁ l₂, (l₁, l₂) ∈ revzip l.sublists → l₁ ++ l₂ ~ l := by
  /-
    α : Type u
    l : List α
    ⊢ ∀ (l₁ l₂ : List α), Membership.mem l.sublists.revzip { fst := l₁, snd := l₂  …
  -/
  rw [revzip]
  /-
    α : Type u
    l : List α
    ⊢ ∀ (l₁ l₂ : List α), Membership.mem (l.sublists.zip l.sublists.reverse) { fst …
  -/
  induction' l using List.reverseRecOn with l' a ih
    /-
      case nil
      α : Type u
      ⊢ ∀ (l₁ l₂ : List α), Membership.mem (List.nil.sublists.zip List.nil.sublists. …
    -/
  · intro l₁ l₂ h
    simp? at h says
      simp only [sublists_nil, reverse_cons, reverse_nil, nil_append, zip_cons_cons, zip_nil_right,
        mem_singleton, Prod.mk.injEq] at h
    /-
      case nil
      α : Type u
      l₁ l₂ : List α
      h : And (Eq l₁ List.nil) (Eq l₂ List.nil)
      ⊢ (HAppend.hAppend l₁ l₂).Perm List.nil
    -/
    simp [h]
    /-
      🎉 no goals
    -/
    /-
      case append_singleton
      α : Type u
      l' : List α
      a : α
      ih : ∀ (l₁ l₂ : List α), Membership.mem (l'.sublists.zip l'.sublists.reverse)  …
      ⊢ ∀ (l₁ l₂ : List α), Membership.mem ((HAppend.hAppend l' (List.cons a List.ni …
    -/
  · intro l₁ l₂ h
    rw [sublists_concat, reverse_append, zip_append (by simp), ← map_reverse, zip_map_right,
      zip_map_left] at *
    /-
      case append_singleton
      α : Type u
      l' : List α
      a : α
      ih : ∀ (l₁ l₂ : List α), Membership.mem (l'.sublists.zip l'.sublists.reverse)  …
      l₁ l₂ : List α
      h : Membership.mem (HAppend.hAppend (List.map (Prod.map id fun x => HAppend.hA …
      ⊢ (HAppend.hAppend l₁ l₂).Perm (HAppend.hAppend l' (List.cons a List.nil))
    -/
    simp only [Prod.mk.inj_iff, mem_map, mem_append, Prod.map_apply, Prod.exists] at h
    /-
      case append_singleton
      α : Type u
      l' : List α
      a : α
      ih : ∀ (l₁ l₂ : List α), Membership.mem (l'.sublists.zip l'.sublists.reverse)  …
      l₁ l₂ : List α
      h : Or (Exists fun a_1 => Exists fun b => And (Membership.mem (l'.sublists.zip …
      ⊢ (HAppend.hAppend l₁ l₂).Perm (HAppend.hAppend l' (List.cons a List.nil))
    -/
    rcases h with (⟨l₁, l₂', h, rfl, rfl⟩ | ⟨l₁', l₂, h, rfl, rfl⟩)
      /-
        case append_singleton.inl.intro.intro.intro.intro
        α : Type u
        l' : List α
        a : α
        ih : ∀ (l₁ l₂ : List α), Membership.mem (l'.sublists.zip l'.sublists.reverse)  …
        l₁ l₂' : List α
        h : Membership.mem (l'.sublists.zip l'.sublists.reverse) { fst := l₁, snd := l …
        ⊢ (HAppend.hAppend (id l₁) (HAppend.hAppend l₂' (List.cons a List.nil))).Perm  …
      -/
    · rw [← append_assoc]
      /-
        case append_singleton.inl.intro.intro.intro.intro
        α : Type u
        l' : List α
        a : α
        ih : ∀ (l₁ l₂ : List α), Membership.mem (l'.sublists.zip l'.sublists.reverse)  …
        l₁ l₂' : List α
        h : Membership.mem (l'.sublists.zip l'.sublists.reverse) { fst := l₁, snd := l …
        ⊢ (HAppend.hAppend (HAppend.hAppend (id l₁) l₂') (List.cons a List.nil)).Perm  …
      -/
      exact (ih _ _ h).append_right _
      /-
        🎉 no goals
      -/
      /-
        case append_singleton.inr.intro.intro.intro.intro
        α : Type u
        l' : List α
        a : α
        ih : ∀ (l₁ l₂ : List α), Membership.mem (l'.sublists.zip l'.sublists.reverse)  …
        l₁' l₂ : List α
        h : Membership.mem (l'.sublists.zip l'.sublists.reverse) { fst := l₁', snd :=  …
        ⊢ (HAppend.hAppend (HAppend.hAppend l₁' (List.cons a List.nil)) (id l₂)).Perm  …
      -/
    · rw [append_assoc]
      /-
        case append_singleton.inr.intro.intro.intro.intro
        α : Type u
        l' : List α
        a : α
        ih : ∀ (l₁ l₂ : List α), Membership.mem (l'.sublists.zip l'.sublists.reverse)  …
        l₁' l₂ : List α
        h : Membership.mem (l'.sublists.zip l'.sublists.reverse) { fst := l₁', snd :=  …
        ⊢ (HAppend.hAppend l₁' (HAppend.hAppend (List.cons a List.nil) (id l₂))).Perm  …
      -/
      apply (perm_append_comm.append_left _).trans
      /-
        case append_singleton.inr.intro.intro.intro.intro
        α : Type u
        l' : List α
        a : α
        ih : ∀ (l₁ l₂ : List α), Membership.mem (l'.sublists.zip l'.sublists.reverse)  …
        l₁' l₂ : List α
        h : Membership.mem (l'.sublists.zip l'.sublists.reverse) { fst := l₁', snd :=  …
        ⊢ (HAppend.hAppend l₁' (HAppend.hAppend (id l₂) (List.cons a List.nil))).Perm  …
      -/
      rw [← append_assoc]
      /-
        case append_singleton.inr.intro.intro.intro.intro
        α : Type u
        l' : List α
        a : α
        ih : ∀ (l₁ l₂ : List α), Membership.mem (l'.sublists.zip l'.sublists.reverse)  …
        l₁' l₂ : List α
        h : Membership.mem (l'.sublists.zip l'.sublists.reverse) { fst := l₁', snd :=  …
        ⊢ (HAppend.hAppend (HAppend.hAppend l₁' (id l₂)) (List.cons a List.nil)).Perm  …
      -/
      exact (ih _ _ h).append_right _
      /-
        🎉 no goals
      -/


theorem revzip_sublists' (l : List α) : ∀ l₁ l₂, (l₁, l₂) ∈ revzip l.sublists' → l₁ ++ l₂ ~ l := by
  /-
    α : Type u
    l : List α
    ⊢ ∀ (l₁ l₂ : List α), Membership.mem l.sublists'.revzip { fst := l₁, snd := l₂ …
  -/
  rw [revzip]
  /-
    α : Type u
    l : List α
    ⊢ ∀ (l₁ l₂ : List α), Membership.mem (l.sublists'.zip l.sublists'.reverse) { f …
  -/
  induction' l with a l IH <;> intro l₁ l₂ h
  · simp_all only [sublists'_nil, reverse_cons, reverse_nil, nil_append, zip_cons_cons,
      zip_nil_right, mem_singleton, Prod.mk.injEq, append_nil, Perm.refl]
  · rw [sublists'_cons, reverse_append, zip_append, ← map_reverse, zip_map_right, zip_map_left] at *
      <;> [simp only [mem_append, mem_map, Prod.map_apply, id_eq, Prod.mk.injEq, Prod.exists,
        exists_eq_right_right] at h; simp]
    /-
      case cons
      α : Type u
      a : α
      l : List α
      IH : ∀ (l₁ l₂ : List α), Membership.mem (l.sublists'.zip l.sublists'.reverse)  …
      l₁ l₂ : List α
      h : Or (Exists fun a_1 => Exists fun b => And (Membership.mem (l.sublists'.zip …
      ⊢ (HAppend.hAppend l₁ l₂).Perm (List.cons a l)
    -/
    rcases h with (⟨l₁, l₂', h, rfl, rfl⟩ | ⟨l₁', h, rfl⟩)
      /-
        case cons.inl.intro.intro.intro.intro
        α : Type u
        a : α
        l : List α
        IH : ∀ (l₁ l₂ : List α), Membership.mem (l.sublists'.zip l.sublists'.reverse)  …
        l₁ l₂' : List α
        h : Membership.mem (l.sublists'.zip l.sublists'.reverse) { fst := l₁, snd := l …
        ⊢ (HAppend.hAppend l₁ (List.cons a l₂')).Perm (List.cons a l)
      -/
    · exact perm_middle.trans ((IH _ _ h).cons _)
      /-
        🎉 no goals
      -/
      /-
        case cons.inr.intro.intro
        α : Type u
        a : α
        l : List α
        IH : ∀ (l₁ l₂ : List α), Membership.mem (l.sublists'.zip l.sublists'.reverse)  …
        l₂ l₁' : List α
        h : Membership.mem (l.sublists'.zip l.sublists'.reverse) { fst := l₁', snd :=  …
        ⊢ (HAppend.hAppend (List.cons a l₁') l₂).Perm (List.cons a l)
      -/
    · exact (IH _ _ h).cons _
      /-
        🎉 no goals
      -/


theorem range_bind_sublistsLen_perm (l : List α) :
    ((List.range (l.length + 1)).flatMap fun n => sublistsLen n l) ~ sublists' l := by
  /-
    α : Type u
    l : List α
    ⊢ ((List.range (HAdd.hAdd l.length 1)).flatMap fun n => List.sublistsLen n l). …
  -/
  induction' l with h tl l_ih
    /-
      case nil
      α : Type u
      ⊢ ((List.range (HAdd.hAdd List.nil.length 1)).flatMap fun n => List.sublistsLe …
    -/
  · simp [range_succ]
    /-
      🎉 no goals
    -/
  · simp_rw [range_succ_eq_map, length, flatMap_cons, flatMap_map, sublistsLen_succ_cons,
      sublists'_cons, List.sublistsLen_zero, List.singleton_append]
    /-
      case cons
      α : Type u
      h : α
      tl : List α
      l_ih : ((List.range (HAdd.hAdd tl.length 1)).flatMap fun n => List.sublistsLen …
      ⊢ (List.cons List.nil ((List.range (HAdd.hAdd tl.length 1)).flatMap fun a => H …
    -/
    refine ((flatMap_append_perm (range (tl.length + 1)) _ _).symm.cons _).trans ?_
    /-
      case cons
      α : Type u
      h : α
      tl : List α
      l_ih : ((List.range (HAdd.hAdd tl.length 1)).flatMap fun n => List.sublistsLen …
      ⊢ (List.cons List.nil (HAppend.hAppend ((List.range (HAdd.hAdd tl.length 1)).f …
    -/
    simp_rw [← List.map_flatMap, ← cons_append]
    /-
      case cons
      α : Type u
      h : α
      tl : List α
      l_ih : ((List.range (HAdd.hAdd tl.length 1)).flatMap fun n => List.sublistsLen …
      ⊢ (HAppend.hAppend (List.cons List.nil ((List.range (HAdd.hAdd tl.length 1)).f …
    -/
    rw [← List.singleton_append, ← List.sublistsLen_zero tl]
    /-
      case cons
      α : Type u
      h : α
      tl : List α
      l_ih : ((List.range (HAdd.hAdd tl.length 1)).flatMap fun n => List.sublistsLen …
      ⊢ (HAppend.hAppend (HAppend.hAppend (List.sublistsLen 0 tl) ((List.range (HAdd …
    -/
    refine Perm.append ?_ (l_ih.map _)
    rw [List.range_succ, flatMap_append, flatMap_singleton,
      sublistsLen_of_length_lt (Nat.lt_succ_self _), append_nil,
      ← List.flatMap_map Nat.succ fun n => sublistsLen n tl,
      ← flatMap_cons 0 _ fun n => sublistsLen n tl, ← range_succ_eq_map]
    /-
      case cons
      α : Type u
      h : α
      tl : List α
      l_ih : ((List.range (HAdd.hAdd tl.length 1)).flatMap fun n => List.sublistsLen …
      ⊢ ((List.range (HAdd.hAdd tl.length 1)).flatMap fun n => List.sublistsLen n tl …
    -/
    exact l_ih
    /-
      🎉 no goals
    -/


