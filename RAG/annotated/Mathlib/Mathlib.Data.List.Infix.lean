@[deprecated IsSuffix.reverse (since := "2024-08-12")] alias isSuffix.reverse := IsSuffix.reverse

@[deprecated IsPrefix.reverse (since := "2024-08-12")] alias isPrefix.reverse := IsPrefix.reverse

@[deprecated IsInfix.reverse (since := "2024-08-12")] alias isInfix.reverse := IsInfix.reverse


@[deprecated IsInfix.eq_of_length (since := "2024-08-12")]
theorem eq_of_infix_of_length_eq (h : l₁ <:+: l₂) : l₁.length = l₂.length → l₁ = l₂ :=
  h.eq_of_length


@[deprecated IsPrefix.eq_of_length (since := "2024-08-12")]
theorem eq_of_prefix_of_length_eq (h : l₁ <+: l₂) : l₁.length = l₂.length → l₁ = l₂ :=
  h.eq_of_length


@[deprecated IsSuffix.eq_of_length (since := "2024-08-12")]
theorem eq_of_suffix_of_length_eq (h : l₁ <:+ l₂) : l₁.length = l₂.length → l₁ = l₂ :=
  h.eq_of_length


@[gcongr] lemma IsPrefix.take (h : l₁ <+: l₂) (n : ℕ) : l₁.take n <+: l₂.take n := by
  /-
    α : Type u_1
    l₁ l₂ : List α
    h : l₁.IsPrefix l₂
    n : Nat
    ⊢ (List.take n l₁).IsPrefix (List.take n l₂)
  -/
  simpa [prefix_take_iff, Nat.min_le_left] using (take_prefix n l₁).trans h
  /-
    🎉 no goals
  -/


@[gcongr] lemma IsPrefix.drop (h : l₁ <+: l₂) (n : ℕ) : l₁.drop n <+: l₂.drop n := by
  /-
    α : Type u_1
    l₁ l₂ : List α
    h : l₁.IsPrefix l₂
    n : Nat
    ⊢ (List.drop n l₁).IsPrefix (List.drop n l₂)
  -/
  rw [prefix_iff_eq_take.mp h, drop_take]; apply take_prefix
                                           /-
                                             🎉 no goals
                                           -/


lemma isPrefix_append_of_length (h : l₁.length ≤ l₂.length) : l₁ <+: l₂ ++ l₃ ↔ l₁ <+: l₂ :=
              /-
                α : Type u_1
                l₁ l₂ l₃ : List α
                h✝ : LE.le l₁.length l₂.length
                h : l₁.IsPrefix (HAppend.hAppend l₂ l₃)
                ⊢ l₁.IsPrefix l₂
              -/
  ⟨fun h ↦ by rw [prefix_iff_eq_take] at *; nth_rw 1 [h, take_eq_left_iff]; tauto,
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
   fun h ↦ h.trans <| l₂.prefix_append l₃⟩


@[simp] lemma take_isPrefix_take {m n : ℕ} : l.take m <+: l.take n ↔ m ≤ n ∨ l.length ≤ n := by
  /-
    α : Type u_1
    l : List α
    m n : Nat
    ⊢ Iff ((List.take m l).IsPrefix (List.take n l)) (Or (LE.le m n) (LE.le l.leng …
  -/
  simp [prefix_take_iff, take_prefix]; omega
                                       /-
                                         🎉 no goals
                                       -/


lemma dropSlice_sublist (n m : ℕ) (l : List α) : l.dropSlice n m <+ l :=
  calc
                                                          /-
                                                            α : Type u_1
                                                            n m : Nat
                                                            l : List α
                                                            ⊢ Eq (List.dropSlice n m l) (HAppend.hAppend (List.take n l) (List.drop m (Lis …
                                                          -/
    l.dropSlice n m = take n l ++ drop m (drop n l) := by rw [dropSlice_eq, drop_drop, Nat.add_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/
  _ <+ take n l ++ drop n l := (Sublist.refl _).append (drop_sublist _ _)
  _ = _ := take_append_drop _ _


lemma dropSlice_subset (n m : ℕ) (l : List α) : l.dropSlice n m ⊆ l :=
  (dropSlice_sublist n m l).subset


lemma mem_of_mem_dropSlice {n m : ℕ} {l : List α} {a : α} (h : a ∈ l.dropSlice n m) : a ∈ l :=
  dropSlice_subset n m l h


theorem tail_subset (l : List α) : tail l ⊆ l :=
  (tail_sublist l).subset


theorem mem_of_mem_dropLast (h : a ∈ l.dropLast) : a ∈ l :=
  dropLast_subset l h


theorem concat_get_prefix {x y : List α} (h : x <+: y) (hl : x.length < y.length) :
    x ++ [y.get ⟨x.length, hl⟩] <+: y := by
  /-
    α : Type u_1
    x y : List α
    h : x.IsPrefix y
    hl : LT.lt x.length y.length
    ⊢ (HAppend.hAppend x (List.cons (y.get ⟨x.length, hl⟩) List.nil)).IsPrefix y
  -/
  use y.drop (x.length + 1)
  /-
    case h
    α : Type u_1
    x y : List α
    h : x.IsPrefix y
    hl : LT.lt x.length y.length
    ⊢ Eq (HAppend.hAppend (HAppend.hAppend x (List.cons (y.get ⟨x.length, hl⟩) Lis …
  -/
  nth_rw 1 [List.prefix_iff_eq_take.mp h]
  /-
    case h
    α : Type u_1
    x y : List α
    h : x.IsPrefix y
    hl : LT.lt x.length y.length
    ⊢ Eq (HAppend.hAppend (HAppend.hAppend (List.take x.length y) (List.cons (y.ge …
  -/
  convert List.take_append_drop (x.length + 1) y using 2
  /-
    case h.e'_2.h.e'_5
    α : Type u_1
    x y : List α
    h : x.IsPrefix y
    hl : LT.lt x.length y.length
    ⊢ Eq (HAppend.hAppend (List.take x.length y) (List.cons (y.get ⟨x.length, hl⟩) …
  -/
  rw [← List.take_concat_get, List.concat_eq_append]; rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


instance decidableInfix [DecidableEq α] : ∀ l₁ l₂ : List α, Decidable (l₁ <:+: l₂)
  | [], l₂ => isTrue ⟨[], l₂, rfl⟩
                                                /-
                                                  α : Type u_1
                                                  l l₁✝ l₂ l₃ : List α
                                                  a✝ b : α
                                                  inst✝ : DecidableEq α
                                                  a : α
                                                  l₁ : List α
                                                  x✝ : (List.cons a l₁).IsInfix List.nil
                                                  s t : List α
                                                  te : Eq (HAppend.hAppend (HAppend.hAppend s (List.cons a l₁)) t) List.nil
                                                  ⊢ False
                                                -/
  | a :: l₁, [] => isFalse fun ⟨s, t, te⟩ => by simp at te
                                                /-
                                                  🎉 no goals
                                                -/
  | l₁, b :: l₂ =>
    letI := l₁.decidableInfix l₂
    @decidable_of_decidable_of_iff (l₁ <+: b :: l₂ ∨ l₁ <:+: l₂) _ _
      infix_cons_iff.symm


@[deprecated cons_prefix_cons (since := "2024-08-14")]
theorem cons_prefix_iff : a :: l₁ <+: b :: l₂ ↔ a = b ∧ l₁ <+: l₂ := by
  /-
    α : Type u_1
    l₁ l₂ : List α
    a b : α
    ⊢ Iff ((List.cons a l₁).IsPrefix (List.cons b l₂)) (And (Eq a b) (l₁.IsPrefix  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-26")] alias IsPrefix.filter_map := IsPrefix.filterMap


protected theorem IsPrefix.reduceOption {l₁ l₂ : List (Option α)} (h : l₁ <+: l₂) :
    l₁.reduceOption <+: l₂.reduceOption :=
  h.filterMap id


instance : IsPartialOrder (List α) (· <+: ·) where
  refl _ := prefix_rfl
  trans _ _ _ := IsPrefix.trans
  antisymm _ _ h₁ h₂ := h₁.eq_of_length <| h₁.length_le.antisymm h₂.length_le


instance : IsPartialOrder (List α) (· <:+ ·) where
  refl _ := suffix_rfl
  trans _ _ _ := IsSuffix.trans
  antisymm _ _ h₁ h₂ := h₁.eq_of_length <| h₁.length_le.antisymm h₂.length_le


instance : IsPartialOrder (List α) (· <:+: ·) where
  refl _ := infix_rfl
  trans _ _ _ := IsInfix.trans
  antisymm _ _ h₁ h₂ := h₁.eq_of_length <| h₁.length_le.antisymm h₂.length_le


@[simp]
theorem mem_inits : ∀ s t : List α, s ∈ inits t ↔ s <+: t
  | s, [] =>
                                    /-
                                      α : Type u_1
                                      s : List α
                                      this : Iff (Eq s List.nil) (s.IsPrefix List.nil)
                                      ⊢ Iff (Membership.mem List.nil.inits s) (s.IsPrefix List.nil)
                                    -/
    suffices s = nil ↔ s <+: nil by simpa only [inits, mem_singleton]
                                    /-
                                      🎉 no goals
                                    -/
    ⟨fun h => h.symm ▸ prefix_rfl, eq_nil_of_prefix_nil⟩
  | s, a :: t =>
                                                                     /-
                                                                       α : Type u_1
                                                                       s : List α
                                                                       a : α
                                                                       t : List α
                                                                       this : Iff (Or (Eq s List.nil) (Exists fun l => And (Membership.mem t.inits l) …
                                                                       ⊢ Iff (Membership.mem (List.cons a t).inits s) (s.IsPrefix (List.cons a t))
                                                                     -/
    suffices (s = nil ∨ ∃ l ∈ inits t, a :: l = s) ↔ s <+: a :: t by simpa
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    ⟨fun o =>
      match s, o with
        /-
          α : Type u_1
          s✝ : List α
          a : α
          t : List α
          o : Or (Eq s✝ List.nil) (Exists fun l => And (Membership.mem t.inits l) (Eq (L …
          s r : List α
          hr : Membership.mem t.inits r
          hs : Eq (List.cons a r) s
          ⊢ s.IsPrefix (List.cons a t)
        -/
      | _, Or.inl rfl => ⟨_, rfl⟩
        /-
          α : Type u_1
          s✝¹ : List α
          a : α
          t : List α
          o : Or (Eq s✝¹ List.nil) (Exists fun l => And (Membership.mem t.inits l) (Eq ( …
          s✝ r : List α
          hr : Membership.mem t.inits r
          hs : Eq (List.cons a r) s✝
          s : List α
          ht : Eq (HAppend.hAppend r s) t
          ⊢ s✝.IsPrefix (List.cons a t)
        -/
      | s, Or.inr ⟨r, hr, hs⟩ => by
                         /-
                           🎉 no goals
                         -/
        let ⟨s, ht⟩ := (mem_inits _ _).1 hr
        rw [← hs, ← ht]; exact ⟨s, rfl⟩,
      fun mi =>
      match s, mi with
      | [], ⟨_, rfl⟩ => Or.inl rfl
                       /-
                         α : Type u_1
                         s✝ : List α
                         a : α
                         t : List α
                         mi : s✝.IsPrefix (List.cons a t)
                         b : α
                         s r : List α
                         hr : Eq (HAppend.hAppend (List.cons b s) r) (List.cons a t)
                         ba : Eq b a
                         st : Eq (HAppend.hAppend s r) t
                         ⊢ Exists fun l => And (Membership.mem t.inits l) (Eq (List.cons a l) (List.con …
                       -/
      | b :: s, ⟨r, hr⟩ =>
                                /-
                                  🎉 no goals
                                -/
        (List.noConfusion hr) fun ba (st : s ++ r = t) =>
          Or.inr <| by rw [ba]; exact ⟨_, (mem_inits _ _).2 ⟨_, st⟩, rfl⟩⟩


@[simp]
theorem mem_tails : ∀ s t : List α, s ∈ tails t ↔ s <:+ t
  | s, [] => by
    /-
      α : Type u_1
      s : List α
      ⊢ Iff (Membership.mem List.nil.tails s) (s.IsSuffix List.nil)
    -/
    simp only [tails, mem_singleton, suffix_nil]
    /-
      🎉 no goals
    -/
  | s, a :: t => by
    /-
      α : Type u_1
      s : List α
      a : α
      t : List α
      ⊢ Iff (Membership.mem (List.cons a t).tails s) (s.IsSuffix (List.cons a t))
    -/
    simp only [tails, mem_cons, mem_tails s t]
    exact
      show s = a :: t ∨ s <:+ t ↔ s <:+ a :: t from
        ⟨fun o =>
          match s, t, o with
          | _, t, Or.inl rfl => suffix_rfl
          | s, _, Or.inr ⟨l, rfl⟩ => ⟨a :: l, rfl⟩,
          fun e =>
          match s, t, e with
          | _, t, ⟨[], rfl⟩ => Or.inl rfl
          | s, t, ⟨b :: l, he⟩ => List.noConfusion he fun _ lt => Or.inr ⟨l, lt⟩⟩


theorem inits_cons (a : α) (l : List α) : inits (a :: l) = [] :: l.inits.map fun t => a :: t := by
  /-
    α : Type u_1
    a : α
    l : List α
    ⊢ Eq (List.cons a l).inits (List.cons List.nil (List.map (fun t => List.cons a …
  -/
  simp
  /-
    🎉 no goals
  -/


                                                                                     /-
                                                                                       α : Type u_1
                                                                                       a : α
                                                                                       l : List α
                                                                                       ⊢ Eq (List.cons a l).tails (List.cons (List.cons a l) l.tails)
                                                                                     -/
theorem tails_cons (a : α) (l : List α) : tails (a :: l) = (a :: l) :: l.tails := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem inits_append : ∀ s t : List α, inits (s ++ t) = s.inits ++ t.inits.tail.map fun l => s ++ l
                 /-
                   α : Type u_1
                   ⊢ Eq (HAppend.hAppend List.nil List.nil).inits (HAppend.hAppend List.nil.inits …
                 -/
  | [], [] => by simp
                 /-
                   🎉 no goals
                 -/
                     /-
                       α : Type u_1
                       a : α
                       t : List α
                       ⊢ Eq (HAppend.hAppend List.nil (List.cons a t)).inits (HAppend.hAppend List.ni …
                     -/
  | [], a :: t => by simp
                     /-
                       🎉 no goals
                     -/
                    /-
                      α : Type u_1
                      a : α
                      s t : List α
                      ⊢ Eq (HAppend.hAppend (List.cons a s) t).inits (HAppend.hAppend (List.cons a s …
                    -/
  | a :: s, t => by simp [inits_append s t, Function.comp_def]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem tails_append :
    ∀ s t : List α, tails (s ++ t) = (s.tails.map fun l => l ++ t) ++ t.tails.tail
                 /-
                   α : Type u_1
                   ⊢ Eq (HAppend.hAppend List.nil List.nil).tails (HAppend.hAppend (List.map (fun …
                 -/
  | [], [] => by simp
                 /-
                   🎉 no goals
                 -/
                     /-
                       α : Type u_1
                       a : α
                       t : List α
                       ⊢ Eq (HAppend.hAppend List.nil (List.cons a t)).tails (HAppend.hAppend (List.m …
                     -/
  | [], a :: t => by simp
                     /-
                       🎉 no goals
                     -/
                    /-
                      α : Type u_1
                      a : α
                      s t : List α
                      ⊢ Eq (HAppend.hAppend (List.cons a s) t).tails (HAppend.hAppend (List.map (fun …
                    -/
  | a :: s, t => by simp [tails_append s t]
                    /-
                      🎉 no goals
                    -/

-- the lemma names `inits_eq_tails` and `tails_eq_inits` are like `sublists_eq_sublists'`

theorem inits_eq_tails : ∀ l : List α, l.inits = (reverse <| map reverse <| tails <| reverse l)
             /-
               α : Type u_1
               ⊢ Eq List.nil.inits (List.map List.reverse List.nil.reverse.tails).reverse
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                 /-
                   α : Type u_1
                   a : α
                   l : List α
                   ⊢ Eq (List.cons a l).inits (List.map List.reverse (List.cons a l).reverse.tail …
                 -/
  | a :: l => by simp [inits_eq_tails l, map_inj_left, ← map_reverse]
                 /-
                   🎉 no goals
                 -/


theorem tails_eq_inits : ∀ l : List α, l.tails = (reverse <| map reverse <| inits <| reverse l)
             /-
               α : Type u_1
               ⊢ Eq List.nil.tails (List.map List.reverse List.nil.reverse.inits).reverse
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                 /-
                   α : Type u_1
                   a : α
                   l : List α
                   ⊢ Eq (List.cons a l).tails (List.map List.reverse (List.cons a l).reverse.init …
                 -/
  | a :: l => by simp [tails_eq_inits l, append_left_inj]
                 /-
                   🎉 no goals
                 -/


theorem inits_reverse (l : List α) : inits (reverse l) = reverse (map reverse l.tails) := by
  /-
    α : Type u_1
    l : List α
    ⊢ Eq l.reverse.inits (List.map List.reverse l.tails).reverse
  -/
  rw [tails_eq_inits l]
  /-
    α : Type u_1
    l : List α
    ⊢ Eq l.reverse.inits (List.map List.reverse (List.map List.reverse l.reverse.i …
  -/
  simp [reverse_involutive.comp_self, ← map_reverse]
  /-
    🎉 no goals
  -/


theorem tails_reverse (l : List α) : tails (reverse l) = reverse (map reverse l.inits) := by
  /-
    α : Type u_1
    l : List α
    ⊢ Eq l.reverse.tails (List.map List.reverse l.inits).reverse
  -/
  rw [inits_eq_tails l]
  /-
    α : Type u_1
    l : List α
    ⊢ Eq l.reverse.tails (List.map List.reverse (List.map List.reverse l.reverse.t …
  -/
  simp [reverse_involutive.comp_self, ← map_reverse]
  /-
    🎉 no goals
  -/


theorem map_reverse_inits (l : List α) : map reverse l.inits = (reverse <| tails <| reverse l) := by
  /-
    α : Type u_1
    l : List α
    ⊢ Eq (List.map List.reverse l.inits) l.reverse.tails.reverse
  -/
  rw [inits_eq_tails l]
  /-
    α : Type u_1
    l : List α
    ⊢ Eq (List.map List.reverse (List.map List.reverse l.reverse.tails).reverse) l …
  -/
  simp [reverse_involutive.comp_self, ← map_reverse]
  /-
    🎉 no goals
  -/


theorem map_reverse_tails (l : List α) : map reverse l.tails = (reverse <| inits <| reverse l) := by
  /-
    α : Type u_1
    l : List α
    ⊢ Eq (List.map List.reverse l.tails) l.reverse.inits.reverse
  -/
  rw [tails_eq_inits l]
  /-
    α : Type u_1
    l : List α
    ⊢ Eq (List.map List.reverse (List.map List.reverse l.reverse.inits).reverse) l …
  -/
  simp [reverse_involutive.comp_self, ← map_reverse]
  /-
    🎉 no goals
  -/


@[simp]
theorem length_tails (l : List α) : length (tails l) = length l + 1 := by
  /-
    α : Type u_1
    l : List α
    ⊢ Eq l.tails.length (HAdd.hAdd l.length 1)
  -/
  induction' l with x l IH
    /-
      case nil
      α : Type u_1
      ⊢ Eq List.nil.tails.length (HAdd.hAdd List.nil.length 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      x : α
      l : List α
      IH : Eq l.tails.length (HAdd.hAdd l.length 1)
      ⊢ Eq (List.cons x l).tails.length (HAdd.hAdd (List.cons x l).length 1)
    -/
  · simpa using IH
    /-
      🎉 no goals
    -/


@[simp]
                                                                          /-
                                                                            α : Type u_1
                                                                            l : List α
                                                                            ⊢ Eq l.inits.length (HAdd.hAdd l.length 1)
                                                                          -/
theorem length_inits (l : List α) : length (inits l) = length l + 1 := by simp [inits_eq_tails]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem getElem_tails (l : List α) (n : Nat) (h : n < (tails l).length) :
    (tails l)[n] = l.drop n := by
  induction l generalizing n with
  | nil => simp
  | cons a l ihl =>
    cases n with
    | zero => simp
    | succ n => simp [ihl]


theorem get_tails (l : List α) (n : Fin (length (tails l))) : (tails l).get n = l.drop n := by
  /-
    α : Type u_1
    l : List α
    n : Fin l.tails.length
    ⊢ Eq (l.tails.get n) (List.drop (↑n) l)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem getElem_inits (l : List α) (n : Nat) (h : n < length (inits l)) :
    (inits l)[n] = l.take n := by
  induction l generalizing n with
  | nil => simp
  | cons a l ihl =>
    cases n with
    | zero => simp
    | succ n => simp [ihl]


theorem get_inits (l : List α) (n : Fin (length (inits l))) : (inits l).get n = l.take n := by
  /-
    α : Type u_1
    l : List α
    n : Fin l.inits.length
    ⊢ Eq (l.inits.get n) (List.take (↑n) l)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma map_inits {β : Type*} (g : α → β) : (l.map g).inits = l.inits.map (map g) := by
  /-
    α : Type u_1
    l : List α
    β : Type u_2
    g : α → β
    ⊢ Eq (List.map g l).inits (List.map (List.map g) l.inits)
  -/
                                      /-
                                        🎉 no goals
                                      -/
  induction' l using reverseRecOn <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


lemma map_tails {β : Type*} (g : α → β) : (l.map g).tails = l.tails.map (map g) := by
  /-
    α : Type u_1
    l : List α
    β : Type u_2
    g : α → β
    ⊢ Eq (List.map g l).tails (List.map (List.map g) l.tails)
  -/
                                      /-
                                        🎉 no goals
                                      -/
  induction' l using reverseRecOn <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


lemma take_inits {n} : (l.take n).inits = l.inits.take (n + 1) := by
  /-
    α : Type u_1
    l : List α
    n : Nat
    ⊢ Eq (List.take n l).inits (List.take (HAdd.hAdd n 1) l.inits)
  -/
                                           /-
                                             🎉 no goals
                                           -/
  apply ext_getElem <;> (simp [take_take]; omega)
                                           /-
                                             🎉 no goals
                                           -/


theorem insert_eq_ite (a : α) (l : List α) : insert a l = if a ∈ l then l else a :: l := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    l : List α
    ⊢ Eq (Insert.insert a l) (ite (Membership.mem l a) l (List.cons a l))
  -/
  simp only [← elem_iff]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    l : List α
    ⊢ Eq (Insert.insert a l) (ite (Eq (List.elem a l) Bool.true) l (List.cons a l))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem suffix_insert (a : α) (l : List α) : l <:+ l.insert a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    l : List α
    ⊢ l.IsSuffix (List.insert a l)
  -/
  by_cases h : a ∈ l
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      h : Membership.mem l a
      ⊢ l.IsSuffix (List.insert a l)
    -/
  · simp only [insert_of_mem h, insert, suffix_refl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      h : Not (Membership.mem l a)
      ⊢ l.IsSuffix (List.insert a l)
    -/
  · simp only [insert_of_not_mem h, suffix_cons, insert]
    /-
      🎉 no goals
    -/


theorem infix_insert (a : α) (l : List α) : l <:+: l.insert a :=
  (suffix_insert a l).isInfix


theorem sublist_insert (a : α) (l : List α) : l <+ l.insert a :=
  (suffix_insert a l).sublist


theorem subset_insert (a : α) (l : List α) : l ⊆ l.insert a :=
  (sublist_insert a l).subset


@[deprecated (since := "2024-08-15")] alias mem_of_mem_suffix := IsSuffix.mem


@[deprecated IsPrefix.getElem (since := "2024-08-15")]
theorem IsPrefix.get_eq {x y : List α} (h : x <+: y) {n} (hn : n < x.length) :
    x.get ⟨n, hn⟩ = y.get ⟨n, hn.trans_le h.length_le⟩ := by
  /-
    α : Type u_1
    x y : List α
    h : x.IsPrefix y
    n : Nat
    hn : LT.lt n x.length
    ⊢ Eq (x.get ⟨n, hn⟩) (y.get ⟨n, ⋯⟩)
  -/
  simp only [get_eq_getElem, IsPrefix.getElem h hn]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-15")] alias IsPrefix.head_eq := IsPrefix.head


