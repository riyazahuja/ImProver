/-- `≤` implies not `>` for lists. -/
@[deprecated "No deprecation message was provided." (since := "2024-07-27")]
theorem le_eq_not_gt [LT α] : ∀ l₁ l₂ : List α, (l₁ ≤ l₂) = ¬l₂ < l₁ := fun _ _ => rfl

-- Porting note: Delete this attribute
-- attribute [inline] List.head!


/-- There is only one list of an empty type -/
instance uniqueOfIsEmpty [IsEmpty α] : Unique (List α) :=
  { instInhabitedList with
    uniq := fun l =>
      match l with
      | [] => rfl
      | a :: _ => isEmptyElim a }


instance : Std.LawfulIdentity (α := List α) Append.append [] where
  left_id := nil_append
  right_id := append_nil


instance : Std.Associative (α := List α) Append.append where
  assoc := append_assoc


@[simp] theorem cons_injective {a : α} : Injective (cons a) := fun _ _ => tail_eq_of_cons_eq


theorem singleton_injective : Injective fun a : α => [a] := fun _ _ h => (cons_eq_cons.1 h).1


theorem set_of_mem_cons (l : List α) (a : α) : { x | x ∈ a :: l } = insert a { x | x ∈ l } :=
  Set.ext fun _ => mem_cons


theorem _root_.Decidable.List.eq_or_ne_mem_of_mem [DecidableEq α]
    {a b : α} {l : List α} (h : a ∈ b :: l) : a = b ∨ a ≠ b ∧ a ∈ l := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    a b : α
    l : List α
    h : Membership.mem (List.cons b l) a
    ⊢ Or (Eq a b) (And (Ne a b) (Membership.mem l a))
  -/
  by_cases hab : a = b
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      a b : α
      l : List α
      h : Membership.mem (List.cons b l) a
      hab : Eq a b
      ⊢ Or (Eq a b) (And (Ne a b) (Membership.mem l a))
    -/
  · exact Or.inl hab
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      a b : α
      l : List α
      h : Membership.mem (List.cons b l) a
      hab : Not (Eq a b)
      ⊢ Or (Eq a b) (And (Ne a b) (Membership.mem l a))
    -/
  · exact ((List.mem_cons.1 h).elim Or.inl (fun h => Or.inr ⟨hab, h⟩))
    /-
      🎉 no goals
    -/


lemma mem_pair {a b c : α} : a ∈ [b, c] ↔ a = b ∨ a = c := by
  /-
    α : Type u
    a b c : α
    ⊢ Iff (Membership.mem (List.cons b (List.cons c List.nil)) a) (Or (Eq a b) (Eq …
  -/
  rw [mem_cons, mem_singleton]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-23")] alias mem_split := append_of_mem

-- The simpNF linter says that the LHS can be simplified via `List.mem_map`.
-- However this is a higher priority lemma.
-- https://github.com/leanprover/std4/issues/207

@[simp 1100, nolint simpNF]
theorem mem_map_of_injective {f : α → β} (H : Injective f) {a : α} {l : List α} :
    f a ∈ map f l ↔ a ∈ l :=
  ⟨fun m => let ⟨_, m', e⟩ := exists_of_mem_map m; H e ▸ m', mem_map_of_mem _⟩


@[simp]
theorem _root_.Function.Involutive.exists_mem_and_apply_eq_iff {f : α → α}
    (hf : Function.Involutive f) (x : α) (l : List α) : (∃ y : α, y ∈ l ∧ f y = x) ↔ f x ∈ l :=
      /-
        α : Type u
        f : α → α
        hf : Function.Involutive f
        x : α
        l : List α
        ⊢ (Exists fun y => And (Membership.mem l y) (Eq (f y) x)) → Membership.mem l ( …
      -/
  ⟨by rintro ⟨y, h, rfl⟩; rwa [hf y], fun h => ⟨f x, h, hf _⟩⟩
                          /-
                            🎉 no goals
                          -/


theorem mem_map_of_involutive {f : α → α} (hf : Involutive f) {a : α} {l : List α} :
                                /-
                                  α : Type u
                                  f : α → α
                                  hf : Function.Involutive f
                                  a : α
                                  l : List α
                                  ⊢ Iff (Membership.mem (List.map f l) a) (Membership.mem l (f a))
                                -/
    a ∈ map f l ↔ f a ∈ l := by rw [mem_map, hf.exists_mem_and_apply_eq_iff]
                                /-
                                  🎉 no goals
                                -/


alias ⟨_, length_pos_of_ne_nil⟩ := length_pos


theorem length_pos_iff_ne_nil {l : List α} : 0 < length l ↔ l ≠ [] :=
  ⟨ne_nil_of_length_pos, length_pos_of_ne_nil⟩


theorem exists_of_length_succ {n} : ∀ l : List α, l.length = n + 1 → ∃ h t, l = h :: t
  | [], H => absurd H.symm <| succ_ne_zero n
  | h :: t, _ => ⟨h, t, rfl⟩


@[simp] lemma length_injective_iff : Injective (List.length : List α → ℕ) ↔ Subsingleton α := by
  /-
    α : Type u
    ⊢ Iff (Function.Injective List.length) (Subsingleton α)
  -/
  constructor
    /-
      case mp
      α : Type u
      ⊢ Function.Injective List.length → Subsingleton α
    -/
  · intro h; refine ⟨fun x y => ?_⟩; (suffices [x] = [y] by simpa using this); apply h; rfl
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
    /-
      case mpr
      α : Type u
      ⊢ Subsingleton α → Function.Injective List.length
    -/
  · intros hα l1 l2 hl
    /-
      case mpr
      α : Type u
      hα : Subsingleton α
      l1 l2 : List α
      hl : Eq l1.length l2.length
      ⊢ Eq l1 l2
    -/
    induction l1 generalizing l2 <;> cases l2
      /-
        case mpr.nil.nil
        α : Type u
        hα : Subsingleton α
        hl : Eq List.nil.length List.nil.length
        ⊢ Eq List.nil List.nil
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case mpr.nil.cons
        α : Type u
        hα : Subsingleton α
        head✝ : α
        tail✝ : List α
        hl : Eq List.nil.length (List.cons head✝ tail✝).length
        ⊢ Eq List.nil (List.cons head✝ tail✝)
      -/
    · cases hl
      /-
        🎉 no goals
      -/
      /-
        case mpr.cons.nil
        α : Type u
        hα : Subsingleton α
        head✝ : α
        tail✝ : List α
        tail_ih✝ : ∀ ⦃l2 : List α⦄, Eq tail✝.length l2.length → Eq tail✝ l2
        hl : Eq (List.cons head✝ tail✝).length List.nil.length
        ⊢ Eq (List.cons head✝ tail✝) List.nil
      -/
    · cases hl
      /-
        🎉 no goals
      -/
    · next ih _ _ =>
      congr
      · subsingleton
      · apply ih; simpa using hl


@[simp default+1] -- Porting note: this used to be just @[simp]
lemma length_injective [Subsingleton α] : Injective (length : List α → ℕ) :=
  length_injective_iff.mpr inferInstance


theorem length_eq_two {l : List α} : l.length = 2 ↔ ∃ a b, l = [a, b] :=
  ⟨fun _ => let [a, b] := l; ⟨a, b, rfl⟩, fun ⟨_, _, e⟩ => e ▸ rfl⟩


theorem length_eq_three {l : List α} : l.length = 3 ↔ ∃ a b c, l = [a, b, c] :=
  ⟨fun _ => let [a, b, c] := l; ⟨a, b, c, rfl⟩, fun ⟨_, _, _, e⟩ => e ▸ rfl⟩


instance instSingletonList : Singleton α (List α) := ⟨fun x => [x]⟩


instance [DecidableEq α] : Insert α (List α) := ⟨List.insert⟩


instance [DecidableEq α] : LawfulSingleton α (List α) :=
  { insert_emptyc_eq := fun x =>
      show (if x ∈ ([] : List α) then [] else [x]) = [x] from if_neg (not_mem_nil _) }


theorem singleton_eq (x : α) : ({x} : List α) = [x] :=
  rfl


theorem insert_neg [DecidableEq α] {x : α} {l : List α} (h : x ∉ l) :
    Insert.insert x l = x :: l :=
  insert_of_not_mem h


theorem insert_pos [DecidableEq α] {x : α} {l : List α} (h : x ∈ l) : Insert.insert x l = l :=
  insert_of_mem h


theorem doubleton_eq [DecidableEq α] {x y : α} (h : x ≠ y) : ({x, y} : List α) = [x, y] := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    x y : α
    h : Ne x y
    ⊢ Eq (Insert.insert x (Singleton.singleton y)) (List.cons x (List.cons y List. …
  -/
  rw [insert_neg, singleton_eq]
  /-
    α : Type u
    inst✝ : DecidableEq α
    x y : α
    h : Ne x y
    ⊢ Not (Membership.mem (Singleton.singleton y) x)
  -/
  rwa [singleton_eq, mem_singleton]
  /-
    🎉 no goals
  -/


theorem forall_mem_of_forall_mem_cons {p : α → Prop} {a : α} {l : List α} (h : ∀ x ∈ a :: l, p x) :
    ∀ x ∈ l, p x := (forall_mem_cons.1 h).2

-- Porting note: bExists in Lean3 and And in Lean4

theorem exists_mem_cons_of {p : α → Prop} {a : α} (l : List α) (h : p a) : ∃ x ∈ a :: l, p x :=
  ⟨a, mem_cons_self _ _, h⟩

-- Porting note: bExists in Lean3 and And in Lean4

theorem exists_mem_cons_of_exists {p : α → Prop} {a : α} {l : List α} : (∃ x ∈ l, p x) →
    ∃ x ∈ a :: l, p x :=
  fun ⟨x, xl, px⟩ => ⟨x, mem_cons_of_mem _ xl, px⟩

-- Porting note: bExists in Lean3 and And in Lean4

theorem or_exists_of_exists_mem_cons {p : α → Prop} {a : α} {l : List α} : (∃ x ∈ a :: l, p x) →
    p a ∨ ∃ x ∈ l, p x :=
  fun ⟨x, xal, px⟩ =>
                                                             /-
                                                               α : Type u
                                                               p : α → Prop
                                                               a : α
                                                               l : List α
                                                               x✝ : Exists fun x => And (Membership.mem (List.cons a l) x) (p x)
                                                               x : α
                                                               xal : Membership.mem (List.cons a l) x
                                                               px : p x
                                                               h : Eq x a
                                                               ⊢ Or (p a) (Exists fun x => And (Membership.mem l x) (p x))
                                                             -/
    Or.elim (eq_or_mem_of_mem_cons xal) (fun h : x = a => by rw [← h]; left; exact px)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
      fun h : x ∈ l => Or.inr ⟨x, h, px⟩


theorem exists_mem_cons_iff (p : α → Prop) (a : α) (l : List α) :
    (∃ x ∈ a :: l, p x) ↔ p a ∨ ∃ x ∈ l, p x :=
  Iff.intro or_exists_of_exists_mem_cons fun h =>
    Or.elim h (exists_mem_cons_of l) exists_mem_cons_of_exists


theorem cons_subset_of_subset_of_mem {a : α} {l m : List α}
    (ainm : a ∈ m) (lsubm : l ⊆ m) : a::l ⊆ m :=
  cons_subset.2 ⟨ainm, lsubm⟩


theorem append_subset_of_subset_of_subset {l₁ l₂ l : List α} (l₁subl : l₁ ⊆ l) (l₂subl : l₂ ⊆ l) :
    l₁ ++ l₂ ⊆ l :=
  fun _ h ↦ (mem_append.1 h).elim (@l₁subl _) (@l₂subl _)


theorem map_subset_iff {l₁ l₂ : List α} (f : α → β) (h : Injective f) :
    map f l₁ ⊆ map f l₂ ↔ l₁ ⊆ l₂ := by
  /-
    α : Type u
    β : Type v
    l₁ l₂ : List α
    f : α → β
    h : Function.Injective f
    ⊢ Iff (HasSubset.Subset (List.map f l₁) (List.map f l₂)) (HasSubset.Subset l₁  …
  -/
  refine ⟨?_, map_subset f⟩; intro h2 x hx
  /-
    α : Type u
    β : Type v
    l₁ l₂ : List α
    f : α → β
    h : Function.Injective f
    h2 : HasSubset.Subset (List.map f l₁) (List.map f l₂)
    x : α
    hx : Membership.mem l₁ x
    ⊢ Membership.mem l₂ x
  -/
  rcases mem_map.1 (h2 (mem_map_of_mem f hx)) with ⟨x', hx', hxx'⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    l₁ l₂ : List α
    f : α → β
    h : Function.Injective f
    h2 : HasSubset.Subset (List.map f l₁) (List.map f l₂)
    x : α
    hx : Membership.mem l₁ x
    x' : α
    hx' : Membership.mem l₂ x'
    hxx' : Eq (f x') (f x)
    ⊢ Membership.mem l₂ x
  -/
  cases h hxx'; exact hx'
                /-
                  🎉 no goals
                -/


theorem append_eq_has_append {L₁ L₂ : List α} : List.append L₁ L₂ = L₁ ++ L₂ :=
  rfl


theorem append_right_injective (s : List α) : Injective fun t ↦ s ++ t :=
  fun _ _ ↦ append_cancel_left


theorem append_left_injective (t : List α) : Injective fun s ↦ s ++ t :=
  fun _ _ ↦ append_cancel_right


theorem eq_replicate_length {a : α} : ∀ {l : List α}, l = replicate l.length a ↔ ∀ b ∈ l, b = a
             /-
               α : Type u
               a : α
               ⊢ Iff (Eq List.nil (List.replicate List.nil.length a)) (∀ (b : α), Membership. …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                   /-
                     α : Type u
                     a b : α
                     l : List α
                     ⊢ Iff (Eq (List.cons b l) (List.replicate (List.cons b l).length a)) (∀ (b_1 : …
                   -/
  | (b :: l) => by simp [eq_replicate_length, replicate_succ]
                   /-
                     🎉 no goals
                   -/


theorem replicate_add (m n) (a : α) : replicate (m + n) a = replicate m a ++ replicate n a := by
  /-
    α : Type u
    m n : Nat
    a : α
    ⊢ Eq (List.replicate (HAdd.hAdd m n) a) (HAppend.hAppend (List.replicate m a)  …
  -/
  rw [append_replicate_replicate]
  /-
    🎉 no goals
  -/


theorem replicate_succ' (n) (a : α) : replicate (n + 1) a = replicate n a ++ [a] :=
  replicate_add n 1 a


theorem replicate_subset_singleton (n) (a : α) : replicate n a ⊆ [a] := fun _ h =>
  mem_singleton.2 (eq_of_mem_replicate h)


theorem subset_singleton_iff {a : α} {L : List α} : L ⊆ [a] ↔ ∃ n, L = replicate n a := by
  /-
    α : Type u
    a : α
    L : List α
    ⊢ Iff (HasSubset.Subset L (List.cons a List.nil)) (Exists fun n => Eq L (List. …
  -/
  simp only [eq_replicate_iff, subset_def, mem_singleton, exists_eq_left']
  /-
    🎉 no goals
  -/


theorem replicate_right_injective {n : ℕ} (hn : n ≠ 0) : Injective (@replicate α n) :=
  fun _ _ h => (eq_replicate_iff.1 h).2 _ <| mem_replicate.2 ⟨hn, rfl⟩


theorem replicate_right_inj {a b : α} {n : ℕ} (hn : n ≠ 0) :
    replicate n a = replicate n b ↔ a = b :=
  (replicate_right_injective hn).eq_iff


theorem replicate_right_inj' {a b : α} : ∀ {n},
    replicate n a = replicate n b ↔ n = 0 ∨ a = b
            /-
              α : Type u
              a b : α
              ⊢ Iff (Eq (List.replicate 0 a) (List.replicate 0 b)) (Or (Eq 0 0) (Eq a b))
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                                                              /-
                                                                α : Type u
                                                                a b : α
                                                                n : Nat
                                                                ⊢ Iff (Eq a b) (Or (Eq (HAdd.hAdd n 1) 0) (Eq a b))
                                                              -/
  | n + 1 => (replicate_right_inj n.succ_ne_zero).trans <| by simp only [n.succ_ne_zero, false_or]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem replicate_left_injective (a : α) : Injective (replicate · a) :=
  LeftInverse.injective (length_replicate · a)


theorem replicate_left_inj {a : α} {n m : ℕ} : replicate n a = replicate m a ↔ n = m :=
  (replicate_left_injective a).eq_iff


                                                                 /-
                                                                   α : Type u
                                                                   x y : α
                                                                   ⊢ Iff (Membership.mem (Pure.pure y) x) (Eq x y)
                                                                 -/
theorem mem_pure (x y : α) : x ∈ (pure y : List α) ↔ x = y := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem bind_eq_flatMap {α β} (f : α → List β) (l : List α) : l >>= f = l.flatMap f :=
  rfl


@[deprecated (since := "2024-10-16")] alias bind_eq_bind := bind_eq_flatMap


theorem reverse_cons' (a : α) (l : List α) : reverse (a :: l) = concat (reverse l) a := by
  /-
    α : Type u
    a : α
    l : List α
    ⊢ Eq (List.cons a l).reverse (l.reverse.concat a)
  -/
  simp only [reverse_cons, concat_eq_append]
  /-
    🎉 no goals
  -/


theorem reverse_concat' (l : List α) (a : α) : (l ++ [a]).reverse = a :: l.reverse := by
  /-
    α : Type u
    l : List α
    a : α
    ⊢ Eq (HAppend.hAppend l (List.cons a List.nil)).reverse (List.cons a l.reverse)
  -/
  rw [reverse_append]; rfl
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem reverse_singleton (a : α) : reverse [a] = [a] :=
  rfl


@[simp]
theorem reverse_involutive : Involutive (@reverse α) :=
  reverse_reverse


@[simp]
theorem reverse_injective : Injective (@reverse α) :=
  reverse_involutive.injective


theorem reverse_surjective : Surjective (@reverse α) :=
  reverse_involutive.surjective


theorem reverse_bijective : Bijective (@reverse α) :=
  reverse_involutive.bijective


theorem concat_eq_reverse_cons (a : α) (l : List α) : concat l a = reverse (a :: reverse l) := by
  /-
    α : Type u
    a : α
    l : List α
    ⊢ Eq (l.concat a) (List.cons a l.reverse).reverse
  -/
  simp only [concat_eq_append, reverse_cons, reverse_reverse]
  /-
    🎉 no goals
  -/


theorem map_reverseAux (f : α → β) (l₁ l₂ : List α) :
    map f (reverseAux l₁ l₂) = reverseAux (map f l₁) (map f l₂) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    l₁ l₂ : List α
    ⊢ Eq (List.map f (l₁.reverseAux l₂)) ((List.map f l₁).reverseAux (List.map f l …
  -/
  simp only [reverseAux_eq, map_append, map_reverse]
  /-
    🎉 no goals
  -/


theorem getLast_append_singleton {a : α} (l : List α) :
    getLast (l ++ [a]) (append_ne_nil_of_right_ne_nil l (cons_ne_nil a _)) = a := by
  /-
    α : Type u
    a : α
    l : List α
    ⊢ Eq ((HAppend.hAppend l (List.cons a List.nil)).getLast ⋯) a
  -/
  simp [getLast_append]
  /-
    🎉 no goals
  -/

-- Porting note: name should be fixed upstream

theorem getLast_append' (l₁ l₂ : List α) (h : l₂ ≠ []) :
    getLast (l₁ ++ l₂) (append_ne_nil_of_right_ne_nil l₁ h) = getLast l₂ h := by
  induction l₁ with
  | nil => simp
  | cons _ _ ih => simp only [cons_append]; rw [List.getLast_cons]; exact ih


theorem getLast_concat' {a : α} (l : List α) : getLast (concat l a) (concat_ne_nil a l) = a := by
  /-
    α : Type u
    a : α
    l : List α
    ⊢ Eq ((l.concat a).getLast ⋯) a
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem getLast_singleton' (a : α) : getLast [a] (cons_ne_nil a []) = a := rfl


@[simp]
theorem getLast_cons_cons (a₁ a₂ : α) (l : List α) :
    getLast (a₁ :: a₂ :: l) (cons_ne_nil _ _) = getLast (a₂ :: l) (cons_ne_nil a₂ l) :=
  rfl


theorem dropLast_append_getLast : ∀ {l : List α} (h : l ≠ []), dropLast l ++ [getLast l h] = l
  | [], h => absurd rfl h
  | [_], _ => rfl
  | a :: b :: l, h => by
    /-
      α : Type u
      a b : α
      l : List α
      h : Ne (List.cons a (List.cons b l)) List.nil
      ⊢ Eq (HAppend.hAppend (List.cons a (List.cons b l)).dropLast (List.cons ((List …
    -/
    rw [dropLast_cons₂, cons_append, getLast_cons (cons_ne_nil _ _)]
    /-
      α : Type u
      a b : α
      l : List α
      h : Ne (List.cons a (List.cons b l)) List.nil
      ⊢ Eq (List.cons a (HAppend.hAppend (List.cons b l).dropLast (List.cons ((List. …
    -/
    congr
    /-
      case e_tail
      α : Type u
      a b : α
      l : List α
      h : Ne (List.cons a (List.cons b l)) List.nil
      ⊢ Eq (HAppend.hAppend (List.cons b l).dropLast (List.cons ((List.cons b l).get …
    -/
    exact dropLast_append_getLast (cons_ne_nil b l)
    /-
      🎉 no goals
    -/


theorem getLast_congr {l₁ l₂ : List α} (h₁ : l₁ ≠ []) (h₂ : l₂ ≠ []) (h₃ : l₁ = l₂) :
                                        /-
                                          α : Type u
                                          l₁ l₂ : List α
                                          h₁ : Ne l₁ List.nil
                                          h₂ : Ne l₂ List.nil
                                          h₃ : Eq l₁ l₂
                                          ⊢ Eq (l₁.getLast h₁) (l₂.getLast h₂)
                                        -/
    getLast l₁ h₁ = getLast l₂ h₂ := by subst l₁; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem getLast_replicate_succ (m : ℕ) (a : α) :
    (replicate (m + 1) a).getLast (ne_nil_of_length_eq_add_one (length_replicate _ _)) = a := by
  /-
    α : Type u
    m : Nat
    a : α
    ⊢ Eq ((List.replicate (HAdd.hAdd m 1) a).getLast ⋯) a
  -/
  simp only [replicate_succ']
  /-
    α : Type u
    m : Nat
    a : α
    ⊢ Eq ((HAppend.hAppend (List.replicate m a) (List.cons a List.nil)).getLast ⋯) a
  -/
  exact getLast_append_singleton _
  /-
    🎉 no goals
  -/


/-- If the last element of `l` does not satisfy `p`, then it is also the last element of
`l.filter p`. -/
lemma getLast_filter' {p : α → Bool} :
    ∀ (l : List α) (hlp : l.filter p ≠ []), p (l.getLast (hlp <| ·.symm ▸ rfl)) = true →
      (l.filter p).getLast hlp = l.getLast (hlp <| ·.symm ▸ rfl)
                     /-
                       α : Type u
                       p : α → Bool
                       a : α
                       h : Ne (List.filter p (List.cons a List.nil)) List.nil
                       h' : Eq (p ((List.cons a List.nil).getLast ⋯)) Bool.true
                       ⊢ Eq ((List.filter p (List.cons a List.nil)).getLast h) ((List.cons a List.nil …
                     -/
  | [a], h, h' => by rw [List.getLast_singleton'] at h'; simp [List.filter_cons, h']
                                                         /-
                                                           🎉 no goals
                                                         -/
  | a :: b :: as, h, h' => by
    /-
      α : Type u
      p : α → Bool
      a b : α
      as : List α
      h : Ne (List.filter p (List.cons a (List.cons b as))) List.nil
      h' : Eq (p ((List.cons a (List.cons b as)).getLast ⋯)) Bool.true
      ⊢ Eq ((List.filter p (List.cons a (List.cons b as))).getLast h) ((List.cons a  …
    -/
    rw [List.getLast_cons_cons] at h' ⊢
    /-
      α : Type u
      p : α → Bool
      a b : α
      as : List α
      h : Ne (List.filter p (List.cons a (List.cons b as))) List.nil
      h' : Eq (p ((List.cons b as).getLast ⋯)) Bool.true
      ⊢ Eq ((List.filter p (List.cons a (List.cons b as))).getLast h) ((List.cons b  …
    -/
    simp only [List.filter_cons (x := a)] at h ⊢
    /-
      α : Type u
      p : α → Bool
      a b : α
      as : List α
      h✝ : Ne (List.filter p (List.cons a (List.cons b as))) List.nil
      h' : Eq (p ((List.cons b as).getLast ⋯)) Bool.true
      h : Ne (ite (Eq (p a) Bool.true) (List.cons a (List.filter p (List.cons b as)) …
      ⊢ Eq ((ite (Eq (p a) Bool.true) (List.cons a (List.filter p (List.cons b as))) …
    -/
    obtain ha | ha := Bool.eq_false_or_eq_true (p a)
      /-
        case inl
        α : Type u
        p : α → Bool
        a b : α
        as : List α
        h✝ : Ne (List.filter p (List.cons a (List.cons b as))) List.nil
        h' : Eq (p ((List.cons b as).getLast ⋯)) Bool.true
        h : Ne (ite (Eq (p a) Bool.true) (List.cons a (List.filter p (List.cons b as)) …
        ha : Eq (p a) Bool.true
        ⊢ Eq ((ite (Eq (p a) Bool.true) (List.cons a (List.filter p (List.cons b as))) …
      -/
    · simp only [ha, ite_true]
      /-
        case inl
        α : Type u
        p : α → Bool
        a b : α
        as : List α
        h✝ : Ne (List.filter p (List.cons a (List.cons b as))) List.nil
        h' : Eq (p ((List.cons b as).getLast ⋯)) Bool.true
        h : Ne (ite (Eq (p a) Bool.true) (List.cons a (List.filter p (List.cons b as)) …
        ha : Eq (p a) Bool.true
        ⊢ Eq ((List.cons a (List.filter p (List.cons b as))).getLast ⋯) ((List.cons b  …
      -/
      rw [getLast_cons, getLast_filter' (b :: as) _ h']
      /-
        α : Type u
        p : α → Bool
        a b : α
        as : List α
        h✝ : Ne (List.filter p (List.cons a (List.cons b as))) List.nil
        h' : Eq (p ((List.cons b as).getLast ⋯)) Bool.true
        h : Ne (ite (Eq (p a) Bool.true) (List.cons a (List.filter p (List.cons b as)) …
        ha : Eq (p a) Bool.true
        ⊢ Ne (List.filter p (List.cons b as)) List.nil
      -/
      exact ne_nil_of_mem <| mem_filter.2 ⟨getLast_mem _, h'⟩
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u
        p : α → Bool
        a b : α
        as : List α
        h✝ : Ne (List.filter p (List.cons a (List.cons b as))) List.nil
        h' : Eq (p ((List.cons b as).getLast ⋯)) Bool.true
        h : Ne (ite (Eq (p a) Bool.true) (List.cons a (List.filter p (List.cons b as)) …
        ha : Eq (p a) Bool.false
        ⊢ Eq ((ite (Eq (p a) Bool.true) (List.cons a (List.filter p (List.cons b as))) …
      -/
    · simp only [ha, cond_false] at h ⊢
      /-
        case inr
        α : Type u
        p : α → Bool
        a b : α
        as : List α
        h✝ : Ne (List.filter p (List.cons a (List.cons b as))) List.nil
        h' : Eq (p ((List.cons b as).getLast ⋯)) Bool.true
        ha : Eq (p a) Bool.false
        h : Ne (ite (Eq Bool.false Bool.true) (List.cons a (List.filter p (List.cons b …
        ⊢ Eq ((ite (Eq Bool.false Bool.true) (List.cons a (List.filter p (List.cons b  …
      -/
      exact getLast_filter' (b :: as) h h'
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-09-06")] alias getLast?_eq_none := getLast?_eq_none_iff


@[deprecated (since := "2024-06-20")] alias getLast?_isNone := getLast?_eq_none


theorem mem_getLast?_eq_getLast : ∀ {l : List α} {x : α}, x ∈ l.getLast? → ∃ h, x = getLast l h
                                  /-
                                    α : Type u
                                    x : α
                                    hx : Membership.mem List.nil.getLast? x
                                    ⊢ False
                                  -/
  | [], x, hx => False.elim <| by simp at hx
                                  /-
                                    🎉 no goals
                                  -/
  | [a], x, hx =>
                       /-
                         α : Type u
                         a x : α
                         hx : Membership.mem (List.cons a List.nil).getLast? x
                         ⊢ Eq a x
                       -/
    have : a = x := by simpa using hx
                       /-
                         🎉 no goals
                       -/
    this ▸ ⟨cons_ne_nil a [], rfl⟩
  | a :: b :: l, x, hx => by
    /-
      α : Type u
      a b : α
      l : List α
      x : α
      hx : Membership.mem (List.cons a (List.cons b l)).getLast? x
      ⊢ Exists fun h => Eq x ((List.cons a (List.cons b l)).getLast h)
    -/
    rw [getLast?_cons_cons] at hx
    /-
      α : Type u
      a b : α
      l : List α
      x : α
      hx : Membership.mem (List.cons b l).getLast? x
      ⊢ Exists fun h => Eq x ((List.cons a (List.cons b l)).getLast h)
    -/
    rcases mem_getLast?_eq_getLast hx with ⟨_, h₂⟩
    /-
      case intro
      α : Type u
      a b : α
      l : List α
      x : α
      hx : Membership.mem (List.cons b l).getLast? x
      w✝ : Ne (List.cons b l) List.nil
      h₂ : Eq x ((List.cons b l).getLast w✝)
      ⊢ Exists fun h => Eq x ((List.cons a (List.cons b l)).getLast h)
    -/
    use cons_ne_nil _ _
    /-
      case h
      α : Type u
      a b : α
      l : List α
      x : α
      hx : Membership.mem (List.cons b l).getLast? x
      w✝ : Ne (List.cons b l) List.nil
      h₂ : Eq x ((List.cons b l).getLast w✝)
      ⊢ Eq x ((List.cons a (List.cons b l)).getLast ⋯)
    -/
    assumption
    /-
      🎉 no goals
    -/


theorem getLast?_eq_getLast_of_ne_nil : ∀ {l : List α} (h : l ≠ []), l.getLast? = some (l.getLast h)
  | [], h => (h rfl).elim
  | [_], _ => rfl
  | _ :: b :: l, _ => @getLast?_eq_getLast_of_ne_nil (b :: l) (cons_ne_nil _ _)


theorem mem_getLast?_cons {x y : α} : ∀ {l : List α}, x ∈ l.getLast? → x ∈ (y :: l).getLast?
                /-
                  α : Type u
                  x y : α
                  x✝ : Membership.mem List.nil.getLast? x
                  ⊢ Membership.mem (List.cons y List.nil).getLast? x
                -/
  | [], _ => by contradiction
                /-
                  🎉 no goals
                -/
  | _ :: _, h => h


theorem dropLast_append_getLast? : ∀ {l : List α}, ∀ a ∈ l.getLast?, dropLast l ++ [a] = l
  | [], a, ha => (Option.not_mem_none a ha).elim
  | [a], _, rfl => rfl
  | a :: b :: l, c, hc => by
    /-
      α : Type u
      a b : α
      l : List α
      c : α
      hc : Membership.mem (List.cons a (List.cons b l)).getLast? c
      ⊢ Eq (HAppend.hAppend (List.cons a (List.cons b l)).dropLast (List.cons c List …
    -/
    rw [getLast?_cons_cons] at hc
    /-
      α : Type u
      a b : α
      l : List α
      c : α
      hc : Membership.mem (List.cons b l).getLast? c
      ⊢ Eq (HAppend.hAppend (List.cons a (List.cons b l)).dropLast (List.cons c List …
    -/
    rw [dropLast_cons₂, cons_append, dropLast_append_getLast? _ hc]
    /-
      🎉 no goals
    -/


theorem getLastI_eq_getLast? [Inhabited α] : ∀ l : List α, l.getLastI = l.getLast?.iget
             /-
               α : Type u
               inst✝ : Inhabited α
               ⊢ Eq List.nil.getLastI List.nil.getLast?.iget
             -/
  | [] => by simp [getLastI, Inhabited.default]
             /-
               🎉 no goals
             -/
  | [_] => rfl
  | [_, _] => rfl
  | [_, _, _] => rfl
                           /-
                             α : Type u
                             inst✝ : Inhabited α
                             head✝¹ head✝ c : α
                             l : List α
                             ⊢ Eq (List.cons head✝¹ (List.cons head✝ (List.cons c l))).getLastI (List.cons  …
                           -/
  | _ :: _ :: c :: l => by simp [getLastI, getLastI_eq_getLast? (c :: l)]
                           /-
                             🎉 no goals
                           -/


theorem getLast?_append_cons :
    ∀ (l₁ : List α) (a : α) (l₂ : List α), getLast? (l₁ ++ a :: l₂) = getLast? (a :: l₂)
  | [], _, _ => rfl
  | [_], _, _ => rfl
  | b :: c :: l₁, a, l₂ => by rw [cons_append, cons_append, getLast?_cons_cons,
    ← cons_append, getLast?_append_cons (c :: l₁)]


theorem getLast?_append_of_ne_nil (l₁ : List α) :
    ∀ {l₂ : List α} (_ : l₂ ≠ []), getLast? (l₁ ++ l₂) = getLast? l₂
                  /-
                    α : Type u
                    l₁ : List α
                    hl₂ : Ne List.nil List.nil
                    ⊢ Eq (HAppend.hAppend l₁ List.nil).getLast? List.nil.getLast?
                  -/
  | [], hl₂ => by contradiction
                  /-
                    🎉 no goals
                  -/
  | b :: l₂, _ => getLast?_append_cons l₁ b l₂


theorem mem_getLast?_append_of_mem_getLast? {l₁ l₂ : List α} {x : α} (h : x ∈ l₂.getLast?) :
    x ∈ (l₁ ++ l₂).getLast? := by
  /-
    α : Type u
    l₁ l₂ : List α
    x : α
    h : Membership.mem l₂.getLast? x
    ⊢ Membership.mem (HAppend.hAppend l₁ l₂).getLast? x
  -/
  cases l₂
    /-
      case nil
      α : Type u
      l₁ : List α
      x : α
      h : Membership.mem List.nil.getLast? x
      ⊢ Membership.mem (HAppend.hAppend l₁ List.nil).getLast? x
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      l₁ : List α
      x head✝ : α
      tail✝ : List α
      h : Membership.mem (List.cons head✝ tail✝).getLast? x
      ⊢ Membership.mem (HAppend.hAppend l₁ (List.cons head✝ tail✝)).getLast? x
    -/
  · rw [List.getLast?_append_cons]
    /-
      case cons
      α : Type u
      l₁ : List α
      x head✝ : α
      tail✝ : List α
      h : Membership.mem (List.cons head✝ tail✝).getLast? x
      ⊢ Membership.mem (List.cons head✝ tail✝).getLast? x
    -/
    exact h
    /-
      🎉 no goals
    -/


@[simp]
theorem head!_nil [Inhabited α] : ([] : List α).head! = default := rfl


@[simp] theorem head_cons_tail (x : List α) (h : x ≠ []) : x.head h :: x.tail = x := by
  /-
    α : Type u
    x : List α
    h : Ne x List.nil
    ⊢ Eq (List.cons (x.head h) x.tail) x
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp at h ⊢
              /-
                🎉 no goals
              -/


theorem head_eq_getElem_zero {l : List α} (hl : l ≠ []) :
    l.head hl = l[0]'(length_pos.2 hl) :=
  (getElem_zero _).symm


                                                                                   /-
                                                                                     α : Type u
                                                                                     inst✝ : Inhabited α
                                                                                     l : List α
                                                                                     ⊢ Eq l.head! l.head?.iget
                                                                                   -/
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/
theorem head!_eq_head? [Inhabited α] (l : List α) : head! l = (head? l).iget := by cases l <;> rfl
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


theorem surjective_head! [Inhabited α] : Surjective (@head! α _) := fun x => ⟨[x], rfl⟩


theorem surjective_head? : Surjective (@head? α) :=
  Option.forall.2 ⟨⟨[], rfl⟩, fun x => ⟨[x], rfl⟩⟩


theorem surjective_tail : Surjective (@tail α)
  | [] => ⟨[], rfl⟩
  | a :: l => ⟨a :: a :: l, rfl⟩


theorem eq_cons_of_mem_head? {x : α} : ∀ {l : List α}, x ∈ l.head? → l = x :: tail l
  | [], h => (Option.not_mem_none _ h).elim
  | a :: l, h => by
    /-
      α : Type u
      x a : α
      l : List α
      h : Membership.mem (List.cons a l).head? x
      ⊢ Eq (List.cons a l) (List.cons x (List.cons a l).tail)
    -/
    simp only [head?, Option.mem_def, Option.some_inj] at h
    /-
      α : Type u
      x a : α
      l : List α
      h : Eq a x
      ⊢ Eq (List.cons a l) (List.cons x (List.cons a l).tail)
    -/
    exact h ▸ rfl
    /-
      🎉 no goals
    -/


@[simp] theorem head!_cons [Inhabited α] (a : α) (l : List α) : head! (a :: l) = a := rfl


@[simp]
theorem head!_append [Inhabited α] (t : List α) {s : List α} (h : s ≠ []) :
    head! (s ++ t) = head! s := by
  /-
    α : Type u
    inst✝ : Inhabited α
    t s : List α
    h : Ne s List.nil
    ⊢ Eq (HAppend.hAppend s t).head! s.head!
  -/
  induction s
    /-
      case nil
      α : Type u
      inst✝ : Inhabited α
      t : List α
      h : Ne List.nil List.nil
      ⊢ Eq (HAppend.hAppend List.nil t).head! List.nil.head!
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      inst✝ : Inhabited α
      t : List α
      head✝ : α
      tail✝ : List α
      tail_ih✝ : Ne tail✝ List.nil → Eq (HAppend.hAppend tail✝ t).head! tail✝.head!
      h : Ne (List.cons head✝ tail✝) List.nil
      ⊢ Eq (HAppend.hAppend (List.cons head✝ tail✝) t).head! (List.cons head✝ tail✝) …
    -/
  · rfl
    /-
      🎉 no goals
    -/


theorem mem_head?_append_of_mem_head? {s t : List α} {x : α} (h : x ∈ s.head?) :
    x ∈ (s ++ t).head? := by
  /-
    α : Type u
    s t : List α
    x : α
    h : Membership.mem s.head? x
    ⊢ Membership.mem (HAppend.hAppend s t).head? x
  -/
  cases s
    /-
      case nil
      α : Type u
      t : List α
      x : α
      h : Membership.mem List.nil.head? x
      ⊢ Membership.mem (HAppend.hAppend List.nil t).head? x
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      t : List α
      x head✝ : α
      tail✝ : List α
      h : Membership.mem (List.cons head✝ tail✝).head? x
      ⊢ Membership.mem (HAppend.hAppend (List.cons head✝ tail✝) t).head? x
    -/
  · exact h
    /-
      🎉 no goals
    -/


theorem head?_append_of_ne_nil :
    ∀ (l₁ : List α) {l₂ : List α} (_ : l₁ ≠ []), head? (l₁ ++ l₂) = head? l₁
  | _ :: _, _, _ => rfl


theorem tail_append_singleton_of_ne_nil {a : α} {l : List α} (h : l ≠ nil) :
    tail (l ++ [a]) = tail l ++ [a] := by
  /-
    α : Type u
    a : α
    l : List α
    h : Ne l List.nil
    ⊢ Eq (HAppend.hAppend l (List.cons a List.nil)).tail (HAppend.hAppend l.tail ( …
  -/
  induction l
    /-
      case nil
      α : Type u
      a : α
      h : Ne List.nil List.nil
      ⊢ Eq (HAppend.hAppend List.nil (List.cons a List.nil)).tail (HAppend.hAppend L …
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      a head✝ : α
      tail✝ : List α
      tail_ih✝ : Ne tail✝ List.nil → Eq (HAppend.hAppend tail✝ (List.cons a List.nil …
      h : Ne (List.cons head✝ tail✝) List.nil
      ⊢ Eq (HAppend.hAppend (List.cons head✝ tail✝) (List.cons a List.nil)).tail (HA …
    -/
  · rw [tail, cons_append, tail]
    /-
      🎉 no goals
    -/


theorem cons_head?_tail : ∀ {l : List α} {a : α}, a ∈ head? l → a :: tail l = l
                   /-
                     α : Type u
                     a : α
                     h : Membership.mem List.nil.head? a
                     ⊢ Eq (List.cons a List.nil.tail) List.nil
                   -/
  | [], a, h => by contradiction
                   /-
                     🎉 no goals
                   -/
  | b :: l, a, h => by
    /-
      α : Type u
      b : α
      l : List α
      a : α
      h : Membership.mem (List.cons b l).head? a
      ⊢ Eq (List.cons a (List.cons b l).tail) (List.cons b l)
    -/
    simp? at h says simp only [head?_cons, Option.mem_def, Option.some.injEq] at h
    /-
      α : Type u
      b : α
      l : List α
      a : α
      h : Eq b a
      ⊢ Eq (List.cons a (List.cons b l).tail) (List.cons b l)
    -/
    simp [h]
    /-
      🎉 no goals
    -/


theorem head!_mem_head? [Inhabited α] : ∀ {l : List α}, l ≠ [] → head! l ∈ head? l
                /-
                  α : Type u
                  inst✝ : Inhabited α
                  h : Ne List.nil List.nil
                  ⊢ Membership.mem List.nil.head? List.nil.head!
                -/
  | [], h => by contradiction
                /-
                  🎉 no goals
                -/
  | _ :: _, _ => rfl


theorem cons_head!_tail [Inhabited α] {l : List α} (h : l ≠ []) : head! l :: tail l = l :=
  cons_head?_tail (head!_mem_head? h)


theorem head!_mem_self [Inhabited α] {l : List α} (h : l ≠ nil) : l.head! ∈ l := by
  /-
    α : Type u
    inst✝ : Inhabited α
    l : List α
    h : Ne l List.nil
    ⊢ Membership.mem l l.head!
  -/
  have h' := mem_cons_self l.head! l.tail
  /-
    α : Type u
    inst✝ : Inhabited α
    l : List α
    h : Ne l List.nil
    h' : Membership.mem (List.cons l.head! l.tail) l.head!
    ⊢ Membership.mem l l.head!
  -/
  rwa [cons_head!_tail h] at h'
  /-
    🎉 no goals
  -/


theorem get_eq_get? (l : List α) (i : Fin l.length) :
                                 /-
                                   ι : Type u_1
                                   α : Type u
                                   β : Type v
                                   γ : Type w
                                   l₁ l₂ l : List α
                                   i : Fin l.length
                                   ⊢ Eq (l.get? ↑i).isSome Bool.true
                                 -/
    l.get i = (l.get? i).get (by simp [getElem?_eq_getElem]) := by
                                 /-
                                   🎉 no goals
                                 -/
  /-
    α : Type u
    l : List α
    i : Fin l.length
    ⊢ Eq (l.get i) ((l.get? ↑i).get ⋯)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem exists_mem_iff_getElem {l : List α} {p : α → Prop} :
    (∃ x ∈ l, p x) ↔ ∃ (i : ℕ) (_ : i < l.length), p l[i] := by
  /-
    α : Type u
    l : List α
    p : α → Prop
    ⊢ Iff (Exists fun x => And (Membership.mem l x) (p x)) (Exists fun i => Exists …
  -/
  simp only [mem_iff_getElem]
  /-
    α : Type u
    l : List α
    p : α → Prop
    ⊢ Iff (Exists fun x => And (Exists fun n => Exists fun h => Eq (GetElem.getEle …
  -/
  exact ⟨fun ⟨_x, ⟨i, hi, hix⟩, hxp⟩ ↦ ⟨i, hi, hix ▸ hxp⟩, fun ⟨i, hi, hp⟩ ↦ ⟨_, ⟨i, hi, rfl⟩, hp⟩⟩
  /-
    🎉 no goals
  -/


theorem forall_mem_iff_getElem {l : List α} {p : α → Prop} :
    (∀ x ∈ l, p x) ↔ ∀ (i : ℕ) (_ : i < l.length), p l[i] := by
  /-
    α : Type u
    l : List α
    p : α → Prop
    ⊢ Iff (∀ (x : α), Membership.mem l x → p x) (∀ (i : Nat) (x : LT.lt i l.length …
  -/
  simp [mem_iff_getElem, @forall_swap α]
  /-
    🎉 no goals
  -/


theorem getElem_cons {l : List α} {a : α} {n : ℕ} (h : n < (a :: l).length) :
                                                         /-
                                                           ι : Type u_1
                                                           α : Type u
                                                           β : Type v
                                                           γ : Type w
                                                           l₁ l₂ l : List α
                                                           a : α
                                                           n : Nat
                                                           h : LT.lt n (List.cons a l).length
                                                           hn : Not (Eq n 0)
                                                           ⊢ LT.lt (HSub.hSub n 1) l.length
                                                         -/
    (a :: l)[n] = if hn : n = 0 then a else l[n - 1]'(by rw [length_cons] at h; omega) := by
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  /-
    α : Type u
    l : List α
    a : α
    n : Nat
    h : LT.lt n (List.cons a l).length
    ⊢ Eq (GetElem.getElem (List.cons a l) n h) (dite (Eq n 0) (fun hn => a) fun hn …
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp
              /-
                🎉 no goals
              -/


theorem get_tail (l : List α) (i) (h : i < l.tail.length)
                                  /-
                                    ι : Type u_1
                                    α : Type u
                                    β : Type v
                                    γ : Type w
                                    l₁ l₂ l : List α
                                    i : Nat
                                    h : LT.lt i l.tail.length
                                    ⊢ LT.lt (HAdd.hAdd i 1) l.length
                                  -/
    (h' : i + 1 < l.length := (by simp only [length_tail] at h; omega)) :
                                                                /-
                                                                  🎉 no goals
                                                                -/
    l.tail.get ⟨i, h⟩ = l.get ⟨i + 1, h'⟩ := by
  /-
    α : Type u
    l : List α
    i : Nat
    h : LT.lt i l.tail.length
    h' : optParam (LT.lt (HAdd.hAdd i 1) l.length) ⋯
    ⊢ Eq (l.tail.get ⟨i, h⟩) (l.get ⟨HAdd.hAdd i 1, h'⟩)
  -/
  cases l <;> [cases h; rfl]
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided." (since := "2024-08-22")]
theorem get_cons {l : List α} {a : α} {n} (hl) :
    (a :: l).get ⟨n, hl⟩ = if hn : n = 0 then a else
                       /-
                         ι : Type u_1
                         α : Type u
                         β : Type v
                         γ : Type w
                         l₁ l₂ l : List α
                         a : α
                         n : Nat
                         hl : LT.lt n (List.cons a l).length
                         hn : Not (Eq n 0)
                         ⊢ LT.lt (HSub.hSub n 1) l.length
                       -/
      l.get ⟨n - 1, by contrapose! hl; rw [length_cons]; omega⟩ :=
                                                         /-
                                                           🎉 no goals
                                                         -/
  getElem_cons hl


/-- Induction principle from the right for lists: if a property holds for the empty list, and
for `l ++ [a]` if it holds for `l`, then it holds for all lists. The principle is given for
a `Sort`-valued predicate, i.e., it can also be used to construct data. -/
@[elab_as_elim]
def reverseRecOn {motive : List α → Sort*} (l : List α) (nil : motive [])
    (append_singleton : ∀ (l : List α) (a : α), motive l → motive (l ++ [a])) : motive l :=
  match h : reverse l with
                                       /-
                                         ι : Type u_1
                                         α : Type u
                                         β : Type v
                                         γ : Type w
                                         l₁ l₂ : List α
                                         motive : List α → Sort u_2
                                         l : List α
                                         nil : motive List.nil
                                         append_singleton : (l : List α) → (a : α) → motive l → motive (HAppend.hAppend …
                                         h : Eq l.reverse List.nil
                                         ⊢ Eq List.nil l
                                       -/
  | [] => cast (congr_arg motive <| by simpa using congr(reverse $h.symm)) <|
                                       /-
                                         🎉 no goals
                                       -/
      nil
  | head :: tail =>
                                 /-
                                   ι : Type u_1
                                   α : Type u
                                   β : Type v
                                   γ : Type w
                                   l₁ l₂ : List α
                                   motive : List α → Sort u_2
                                   l : List α
                                   nil : motive List.nil
                                   append_singleton : (l : List α) → (a : α) → motive l → motive (HAppend.hAppend …
                                   head : α
                                   tail : List α
                                   h : Eq l.reverse (List.cons head tail)
                                   ⊢ Eq (HAppend.hAppend tail.reverse (List.cons head List.nil)) l
                                 -/
    cast (congr_arg motive <| by simpa using congr(reverse $h.symm)) <|
                                 /-
                                   🎉 no goals
                                 -/
      append_singleton _ head <| reverseRecOn (reverse tail) nil append_singleton
termination_by l.length
decreasing_by
  simp_wf
  rw [← length_reverse l, h, length_cons]
  simp [Nat.lt_succ]


@[simp]
theorem reverseRecOn_nil {motive : List α → Sort*} (nil : motive [])
    (append_singleton : ∀ (l : List α) (a : α), motive l → motive (l ++ [a])) :
    reverseRecOn [] nil append_singleton = nil := reverseRecOn.eq_1 ..

-- `unusedHavesSuffices` is getting confused by the unfolding of `reverseRecOn`

@[simp, nolint unusedHavesSuffices]
theorem reverseRecOn_concat {motive : List α → Sort*} (x : α) (xs : List α) (nil : motive [])
    (append_singleton : ∀ (l : List α) (a : α), motive l → motive (l ++ [a])) :
    reverseRecOn (motive := motive) (xs ++ [x]) nil append_singleton =
      append_singleton _ _ (reverseRecOn (motive := motive) xs nil append_singleton) := by
  suffices ∀ ys (h : reverse (reverse xs) = ys),
      reverseRecOn (motive := motive) (xs ++ [x]) nil append_singleton =
        cast (by simp [(reverse_reverse _).symm.trans h])
          (append_singleton _ x (reverseRecOn (motive := motive) ys nil append_singleton)) by
    exact this _ (reverse_reverse xs)
  /-
    α : Type u
    motive : List α → Sort u_2
    x : α
    xs : List α
    nil : motive List.nil
    append_singleton : (l : List α) → (a : α) → motive l → motive (HAppend.hAppend …
    ⊢ ∀ (ys : List α) (h : Eq xs.reverse.reverse ys), Eq (List.reverseRecOn (HAppe …
  -/
  intros ys hy
  /-
    α : Type u
    motive : List α → Sort u_2
    x : α
    xs : List α
    nil : motive List.nil
    append_singleton : (l : List α) → (a : α) → motive l → motive (HAppend.hAppend …
    ys : List α
    hy : Eq xs.reverse.reverse ys
    ⊢ Eq (List.reverseRecOn (HAppend.hAppend xs (List.cons x List.nil)) nil append …
  -/
  conv_lhs => unfold reverseRecOn
  /-
    α : Type u
    motive : List α → Sort u_2
    x : α
    xs : List α
    nil : motive List.nil
    append_singleton : (l : List α) → (a : α) → motive l → motive (HAppend.hAppend …
    ys : List α
    hy : Eq xs.reverse.reverse ys
    ⊢ Eq (List.reverseRecOn.match_1 (fun x_1 => motive (HAppend.hAppend xs (List.c …
  -/
  split
  /-
    case h_1
    α : Type u
    motive : List α → Sort u_2
    x : α
    xs : List α
    nil : motive List.nil
    append_singleton : (l : List α) → (a : α) → motive l → motive (HAppend.hAppend …
    ys : List α
    hy : Eq xs.reverse.reverse ys
    heq✝ : Eq (HAppend.hAppend xs (List.cons x List.nil)).reverse List.nil
    ⊢ Eq (cast ⋯ nil) (cast ⋯ (append_singleton ys x (List.reverseRecOn ys nil app …
  -/
  next h => simp at h
  next heq =>
    revert heq
    simp only [reverse_append, reverse_cons, reverse_nil, nil_append, singleton_append, cons.injEq]
    rintro ⟨rfl, rfl⟩
    subst ys
    rfl


/-- Bidirectional induction principle for lists: if a property holds for the empty list, the
singleton list, and `a :: (l ++ [b])` from `l`, then it holds for all lists. This can be used to
prove statements about palindromes. The principle is given for a `Sort`-valued predicate, i.e., it
can also be used to construct data. -/
@[elab_as_elim]
def bidirectionalRec {motive : List α → Sort*} (nil : motive []) (singleton : ∀ a : α, motive [a])
    (cons_append : ∀ (a : α) (l : List α) (b : α), motive l → motive (a :: (l ++ [b]))) :
    ∀ l, motive l
  | [] => nil
  | [a] => singleton a
  | a :: b :: l =>
    let l' := dropLast (b :: l)
    let b' := getLast (b :: l) (cons_ne_nil _ _)
             /-
               ι : Type u_1
               α : Type u
               β : Type v
               γ : Type w
               l₁ l₂ : List α
               motive : List α → Sort u_2
               nil : motive List.nil
               singleton : (a : α) → motive (List.cons a List.nil)
               cons_append : (a : α) → (l : List α) → (b : α) → motive l → motive (List.cons  …
               a b : α
               l : List α
               l' : List α := (List.cons b l).dropLast
               b' : α := (List.cons b l).getLast ⋯
               ⊢ Eq (motive (List.cons a (HAppend.hAppend l' (List.cons b' List.nil)))) (moti …
             -/
    cast (by rw [← dropLast_append_getLast (cons_ne_nil b l)]) <|
             /-
               🎉 no goals
             -/
      cons_append a l' b' (bidirectionalRec nil singleton cons_append l')
termination_by l => l.length


@[simp]
theorem bidirectionalRec_nil {motive : List α → Sort*}
    (nil : motive []) (singleton : ∀ a : α, motive [a])
    (cons_append : ∀ (a : α) (l : List α) (b : α), motive l → motive (a :: (l ++ [b]))) :
    bidirectionalRec nil singleton cons_append [] = nil := bidirectionalRec.eq_1 ..



@[simp]
theorem bidirectionalRec_singleton {motive : List α → Sort*}
    (nil : motive []) (singleton : ∀ a : α, motive [a])
    (cons_append : ∀ (a : α) (l : List α) (b : α), motive l → motive (a :: (l ++ [b]))) (a : α) :
    bidirectionalRec nil singleton cons_append [a] = singleton a := by
  /-
    α : Type u
    motive : List α → Sort u_2
    nil : motive List.nil
    singleton : (a : α) → motive (List.cons a List.nil)
    cons_append : (a : α) → (l : List α) → (b : α) → motive l → motive (List.cons  …
    a : α
    ⊢ Eq (List.bidirectionalRec nil singleton cons_append (List.cons a List.nil))  …
  -/
  simp [bidirectionalRec]
  /-
    🎉 no goals
  -/


@[simp]
theorem bidirectionalRec_cons_append {motive : List α → Sort*}
    (nil : motive []) (singleton : ∀ a : α, motive [a])
    (cons_append : ∀ (a : α) (l : List α) (b : α), motive l → motive (a :: (l ++ [b])))
    (a : α) (l : List α) (b : α) :
    bidirectionalRec nil singleton cons_append (a :: (l ++ [b])) =
      cons_append a l b (bidirectionalRec nil singleton cons_append l) := by
  /-
    α : Type u
    motive : List α → Sort u_2
    nil : motive List.nil
    singleton : (a : α) → motive (List.cons a List.nil)
    cons_append : (a : α) → (l : List α) → (b : α) → motive l → motive (List.cons  …
    a : α
    l : List α
    b : α
    ⊢ Eq (List.bidirectionalRec nil singleton cons_append (List.cons a (HAppend.hA …
  -/
  conv_lhs => unfold bidirectionalRec
  cases l with
  | nil => rfl
  | cons x xs =>
  simp only [List.cons_append]
  dsimp only [← List.cons_append]
  suffices ∀ (ys init : List α) (hinit : init = ys) (last : α) (hlast : last = b),
      (cons_append a init last
        (bidirectionalRec nil singleton cons_append init)) =
      cast (congr_arg motive <| by simp [hinit, hlast])
        (cons_append a ys b (bidirectionalRec nil singleton cons_append ys)) by
    rw [this (x :: xs) _ (by rw [dropLast_append_cons, dropLast_single, append_nil]) _ (by simp)]
    simp
  rintro ys init rfl last rfl
  rfl


/-- Like `bidirectionalRec`, but with the list parameter placed first. -/
@[elab_as_elim]
abbrev bidirectionalRecOn {C : List α → Sort*} (l : List α) (H0 : C []) (H1 : ∀ a : α, C [a])
    (Hn : ∀ (a : α) (l : List α) (b : α), C l → C (a :: (l ++ [b]))) : C l :=
  bidirectionalRec H0 H1 Hn l


theorem Sublist.cons_cons {l₁ l₂ : List α} (a : α) (s : l₁ <+ l₂) : a :: l₁ <+ a :: l₂ :=
  Sublist.cons₂ _ s


lemma cons_sublist_cons' {a b : α} : a :: l₁ <+ b :: l₂ ↔ a :: l₁ <+ l₂ ∨ a = b ∧ l₁ <+ l₂ := by
  /-
    α : Type u
    l₁ l₂ : List α
    a b : α
    ⊢ Iff ((List.cons a l₁).Sublist (List.cons b l₂)) (Or ((List.cons a l₁).Sublis …
  -/
  constructor
    /-
      case mp
      α : Type u
      l₁ l₂ : List α
      a b : α
      ⊢ (List.cons a l₁).Sublist (List.cons b l₂) → Or ((List.cons a l₁).Sublist l₂) …
    -/
  · rintro (_ | _)
      /-
        case mp.cons
        α : Type u
        l₁ l₂ : List α
        a b : α
        a✝ : (List.cons a l₁).Sublist l₂
        ⊢ Or ((List.cons a l₁).Sublist l₂) (And (Eq a b) (l₁.Sublist l₂))
      -/
    · exact Or.inl ‹_›
      /-
        🎉 no goals
      -/
      /-
        case mp.cons₂
        α : Type u
        l₁ l₂ : List α
        a : α
        a✝ : l₁.Sublist l₂
        ⊢ Or ((List.cons a l₁).Sublist l₂) (And (Eq a a) (l₁.Sublist l₂))
      -/
    · exact Or.inr ⟨rfl, ‹_›⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u
      l₁ l₂ : List α
      a b : α
      ⊢ Or ((List.cons a l₁).Sublist l₂) (And (Eq a b) (l₁.Sublist l₂)) → (List.cons …
    -/
  · rintro (h | ⟨rfl, h⟩)
      /-
        case mpr.inl
        α : Type u
        l₁ l₂ : List α
        a b : α
        h : (List.cons a l₁).Sublist l₂
        ⊢ (List.cons a l₁).Sublist (List.cons b l₂)
      -/
    · exact h.cons _
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro
        α : Type u
        l₁ l₂ : List α
        a : α
        h : l₁.Sublist l₂
        ⊢ (List.cons a l₁).Sublist (List.cons a l₂)
      -/
    · rwa [cons_sublist_cons]
      /-
        🎉 no goals
      -/


theorem sublist_cons_of_sublist (a : α) (h : l₁ <+ l₂) : l₁ <+ a :: l₂ := h.cons _


@[deprecated "No deprecation message was provided." (since := "2024-04-07")]
theorem sublist_of_cons_sublist_cons {a} (h : a :: l₁ <+ a :: l₂) : l₁ <+ l₂ := h.of_cons_cons


@[deprecated (since := "2024-04-07")] alias cons_sublist_cons_iff := cons_sublist_cons

-- Porting note: this lemma seems to have been renamed on the occasion of its move to Batteries

alias sublist_nil_iff_eq_nil := sublist_nil


@[simp] lemma sublist_singleton {l : List α} {a : α} : l <+ [a] ↔ l = [] ∨ l = [a] := by
  /-
    α : Type u
    l : List α
    a : α
    ⊢ Iff (l.Sublist (List.cons a List.nil)) (Or (Eq l List.nil) (Eq l (List.cons  …
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
  constructor <;> rintro (_ | _) <;> aesop
                                     /-
                                       🎉 no goals
                                     -/


theorem Sublist.antisymm (s₁ : l₁ <+ l₂) (s₂ : l₂ <+ l₁) : l₁ = l₂ :=
  s₁.eq_of_length_le s₂.length_le


instance decidableSublist [DecidableEq α] : ∀ l₁ l₂ : List α, Decidable (l₁ <+ l₂)
  | [], _ => isTrue <| nil_sublist _
  | _ :: _, [] => isFalse fun h => List.noConfusion <| eq_nil_of_sublist_nil h
  | a :: l₁, b :: l₂ =>
    if h : a = b then
      @decidable_of_decidable_of_iff _ _ (decidableSublist l₁ l₂) <| h ▸ cons_sublist_cons.symm
    else
      @decidable_of_decidable_of_iff _ _ (decidableSublist (a :: l₁) l₂)
        ⟨sublist_cons_of_sublist _, fun s =>
          match a, l₁, s, h with
          | _, _, Sublist.cons _ s', h => s'
          | _, _, Sublist.cons₂ t _, h => absurd rfl h⟩


/-- If the first element of two lists are different, then a sublist relation can be reduced. -/
theorem Sublist.of_cons_of_ne {a b} (h₁ : a ≠ b) (h₂ : a :: l₁ <+ b :: l₂) : a :: l₁ <+ l₂ :=
  match h₁, h₂ with
  | _, .cons _ h =>  h


@[simp]
theorem indexOf_cons_self (a : α) (l : List α) : indexOf a (a :: l) = 0 := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    a : α
    l : List α
    ⊢ Eq (List.indexOf a (List.cons a l)) 0
  -/
  rw [indexOf, findIdx_cons, beq_self_eq_true, cond]
  /-
    🎉 no goals
  -/

-- fun e => if_pos e

theorem indexOf_cons_eq {a b : α} (l : List α) : b = a → indexOf a (b :: l) = 0
            /-
              α : Type u
              inst✝ : DecidableEq α
              a b : α
              l : List α
              x✝ : Eq b a
              e : Eq b a := x✝
              ⊢ Eq (List.indexOf a (List.cons b l)) 0
            -/
  | e => by rw [← e]; exact indexOf_cons_self b l
                      /-
                        🎉 no goals
                      -/

-- fun n => if_neg n

@[simp]
theorem indexOf_cons_ne {a b : α} (l : List α) : b ≠ a → indexOf a (b :: l) = succ (indexOf a l)
            /-
              α : Type u
              inst✝ : DecidableEq α
              a b : α
              l : List α
              x✝ : Ne b a
              h : Ne b a := x✝
              ⊢ Eq (List.indexOf a (List.cons b l)) (List.indexOf a l).succ
            -/
  | h => by simp only [indexOf, findIdx_cons, Bool.cond_eq_ite, beq_iff_eq, h, ite_false]
            /-
              🎉 no goals
            -/


theorem indexOf_eq_length {a : α} {l : List α} : indexOf a l = length l ↔ a ∉ l := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    a : α
    l : List α
    ⊢ Iff (Eq (List.indexOf a l) l.length) (Not (Membership.mem l a))
  -/
  induction' l with b l ih
    /-
      case nil
      α : Type u
      inst✝ : DecidableEq α
      a : α
      ⊢ Iff (Eq (List.indexOf a List.nil) List.nil.length) (Not (Membership.mem List …
    -/
  · exact iff_of_true rfl (not_mem_nil _)
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u
    inst✝ : DecidableEq α
    a b : α
    l : List α
    ih : Iff (Eq (List.indexOf a l) l.length) (Not (Membership.mem l a))
    ⊢ Iff (Eq (List.indexOf a (List.cons b l)) (List.cons b l).length) (Not (Membe …
  -/
  simp only [length, mem_cons, indexOf_cons, eq_comm]
  /-
    case cons
    α : Type u
    inst✝ : DecidableEq α
    a b : α
    l : List α
    ih : Iff (Eq (List.indexOf a l) l.length) (Not (Membership.mem l a))
    ⊢ Iff (Eq (cond (BEq.beq b a) 0 (HAdd.hAdd (List.indexOf a l) 1)) (HAdd.hAdd l …
  -/
  rw [cond_eq_if]
  /-
    case cons
    α : Type u
    inst✝ : DecidableEq α
    a b : α
    l : List α
    ih : Iff (Eq (List.indexOf a l) l.length) (Not (Membership.mem l a))
    ⊢ Iff (Eq (ite (Eq (BEq.beq b a) Bool.true) 0 (HAdd.hAdd (List.indexOf a l) 1) …
  -/
  split_ifs with h <;> simp at h
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      a b : α
      l : List α
      ih : Iff (Eq (List.indexOf a l) l.length) (Not (Membership.mem l a))
      h : Eq b a
      ⊢ Iff False (Not (Or (Eq a b) (Membership.mem l a)))
    -/
  · exact iff_of_false (by rintro ⟨⟩) fun H => H <| Or.inl h.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      a b : α
      l : List α
      ih : Iff (Eq (List.indexOf a l) l.length) (Not (Membership.mem l a))
      h : Not (Eq b a)
      ⊢ Iff (Eq (HAdd.hAdd (List.indexOf a l) 1) (HAdd.hAdd l.length 1)) (Not (Or (E …
    -/
  · simp only [Ne.symm h, false_or]
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      a b : α
      l : List α
      ih : Iff (Eq (List.indexOf a l) l.length) (Not (Membership.mem l a))
      h : Not (Eq b a)
      ⊢ Iff (Eq (HAdd.hAdd (List.indexOf a l) 1) (HAdd.hAdd l.length 1)) (Not (Membe …
    -/
    rw [← ih]
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      a b : α
      l : List α
      ih : Iff (Eq (List.indexOf a l) l.length) (Not (Membership.mem l a))
      h : Not (Eq b a)
      ⊢ Iff (Eq (HAdd.hAdd (List.indexOf a l) 1) (HAdd.hAdd l.length 1)) (Eq (List.i …
    -/
    exact succ_inj'
    /-
      🎉 no goals
    -/


@[simp]
theorem indexOf_of_not_mem {l : List α} {a : α} : a ∉ l → indexOf a l = length l :=
  indexOf_eq_length.2


theorem indexOf_le_length {a : α} {l : List α} : indexOf a l ≤ length l := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    a : α
    l : List α
    ⊢ LE.le (List.indexOf a l) l.length
  -/
  induction' l with b l ih; · rfl
                              /-
                                🎉 no goals
                              -/
  /-
    case cons
    α : Type u
    inst✝ : DecidableEq α
    a b : α
    l : List α
    ih : LE.le (List.indexOf a l) l.length
    ⊢ LE.le (List.indexOf a (List.cons b l)) (List.cons b l).length
  -/
  simp only [length, indexOf_cons, cond_eq_if, beq_iff_eq]
  /-
    case cons
    α : Type u
    inst✝ : DecidableEq α
    a b : α
    l : List α
    ih : LE.le (List.indexOf a l) l.length
    ⊢ LE.le (ite (Eq b a) 0 (HAdd.hAdd (List.indexOf a l) 1)) (HAdd.hAdd l.length 1)
  -/
  by_cases h : b = a
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      a b : α
      l : List α
      ih : LE.le (List.indexOf a l) l.length
      h : Eq b a
      ⊢ LE.le (ite (Eq b a) 0 (HAdd.hAdd (List.indexOf a l) 1)) (HAdd.hAdd l.length 1)
    -/
  · rw [if_pos h]; exact Nat.zero_le _
                   /-
                     🎉 no goals
                   -/
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      a b : α
      l : List α
      ih : LE.le (List.indexOf a l) l.length
      h : Not (Eq b a)
      ⊢ LE.le (ite (Eq b a) 0 (HAdd.hAdd (List.indexOf a l) 1)) (HAdd.hAdd l.length 1)
    -/
  · rw [if_neg h]; exact succ_le_succ ih
                   /-
                     🎉 no goals
                   -/


theorem indexOf_lt_length {a} {l : List α} : indexOf a l < length l ↔ a ∈ l :=
  ⟨fun h => Decidable.byContradiction fun al => Nat.ne_of_lt h <| indexOf_eq_length.2 al,
   fun al => (lt_of_le_of_ne indexOf_le_length) fun h => indexOf_eq_length.1 h al⟩


theorem indexOf_append_of_mem {a : α} (h : a ∈ l₁) : indexOf a (l₁ ++ l₂) = indexOf a l₁ := by
  /-
    α : Type u
    l₁ l₂ : List α
    inst✝ : DecidableEq α
    a : α
    h : Membership.mem l₁ a
    ⊢ Eq (List.indexOf a (HAppend.hAppend l₁ l₂)) (List.indexOf a l₁)
  -/
  induction' l₁ with d₁ t₁ ih
    /-
      case nil
      α : Type u
      l₁ l₂ : List α
      inst✝ : DecidableEq α
      a : α
      h : Membership.mem List.nil a
      ⊢ Eq (List.indexOf a (HAppend.hAppend List.nil l₂)) (List.indexOf a List.nil)
    -/
  · exfalso
    /-
      case nil
      α : Type u
      l₁ l₂ : List α
      inst✝ : DecidableEq α
      a : α
      h : Membership.mem List.nil a
      ⊢ False
    -/
    exact not_mem_nil a h
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u
    l₁ l₂ : List α
    inst✝ : DecidableEq α
    a d₁ : α
    t₁ : List α
    ih : Membership.mem t₁ a → Eq (List.indexOf a (HAppend.hAppend t₁ l₂)) (List.i …
    h : Membership.mem (List.cons d₁ t₁) a
    ⊢ Eq (List.indexOf a (HAppend.hAppend (List.cons d₁ t₁) l₂)) (List.indexOf a ( …
  -/
  rw [List.cons_append]
  /-
    case cons
    α : Type u
    l₁ l₂ : List α
    inst✝ : DecidableEq α
    a d₁ : α
    t₁ : List α
    ih : Membership.mem t₁ a → Eq (List.indexOf a (HAppend.hAppend t₁ l₂)) (List.i …
    h : Membership.mem (List.cons d₁ t₁) a
    ⊢ Eq (List.indexOf a (List.cons d₁ (HAppend.hAppend t₁ l₂))) (List.indexOf a ( …
  -/
  by_cases hh : d₁ = a
    /-
      case pos
      α : Type u
      l₁ l₂ : List α
      inst✝ : DecidableEq α
      a d₁ : α
      t₁ : List α
      ih : Membership.mem t₁ a → Eq (List.indexOf a (HAppend.hAppend t₁ l₂)) (List.i …
      h : Membership.mem (List.cons d₁ t₁) a
      hh : Eq d₁ a
      ⊢ Eq (List.indexOf a (List.cons d₁ (HAppend.hAppend t₁ l₂))) (List.indexOf a ( …
    -/
  · iterate 2 rw [indexOf_cons_eq _ hh]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u
    l₁ l₂ : List α
    inst✝ : DecidableEq α
    a d₁ : α
    t₁ : List α
    ih : Membership.mem t₁ a → Eq (List.indexOf a (HAppend.hAppend t₁ l₂)) (List.i …
    h : Membership.mem (List.cons d₁ t₁) a
    hh : Not (Eq d₁ a)
    ⊢ Eq (List.indexOf a (List.cons d₁ (HAppend.hAppend t₁ l₂))) (List.indexOf a ( …
  -/
  rw [indexOf_cons_ne _ hh, indexOf_cons_ne _ hh, ih (mem_of_ne_of_mem (Ne.symm hh) h)]
  /-
    🎉 no goals
  -/


theorem indexOf_append_of_not_mem {a : α} (h : a ∉ l₁) :
    indexOf a (l₁ ++ l₂) = l₁.length + indexOf a l₂ := by
  /-
    α : Type u
    l₁ l₂ : List α
    inst✝ : DecidableEq α
    a : α
    h : Not (Membership.mem l₁ a)
    ⊢ Eq (List.indexOf a (HAppend.hAppend l₁ l₂)) (HAdd.hAdd l₁.length (List.index …
  -/
  induction' l₁ with d₁ t₁ ih
    /-
      case nil
      α : Type u
      l₁ l₂ : List α
      inst✝ : DecidableEq α
      a : α
      h : Not (Membership.mem List.nil a)
      ⊢ Eq (List.indexOf a (HAppend.hAppend List.nil l₂)) (HAdd.hAdd List.nil.length …
    -/
  · rw [List.nil_append, List.length, Nat.zero_add]
    /-
      🎉 no goals
    -/
  rw [List.cons_append, indexOf_cons_ne _ (ne_of_not_mem_cons h).symm, List.length,
    ih (not_mem_of_not_mem_cons h), Nat.succ_add]


@[simp]
theorem getElem?_length (l : List α) : l[l.length]? = none := getElem?_eq_none le_rfl


@[deprecated getElem?_length (since := "2024-06-12")]
theorem get?_length (l : List α) : l.get? l.length = none := get?_eq_none le_rfl


@[deprecated (since := "2024-05-03")] alias get?_injective := get?_inj


/-- A version of `getElem_map` that can be used for rewriting. -/
theorem getElem_map_rev (f : α → β) {l} {n : Nat} {h : n < l.length} :
    f l[n] = (map f l)[n]'((l.length_map f).symm ▸ h) := Eq.symm (getElem_map _)


/-- A version of `get_map` that can be used for rewriting. -/
@[deprecated getElem_map_rev (since := "2024-06-12")]
theorem get_map_rev (f : α → β) {l n} :
    f (get l n) = get (map f l) ⟨n.1, (l.length_map f).symm ▸ n.2⟩ := Eq.symm (getElem_map _)


theorem get_length_sub_one {l : List α} (h : l.length - 1 < l.length) :
                                            /-
                                              ι : Type u_1
                                              α : Type u
                                              β : Type v
                                              γ : Type w
                                              l₁ l₂ l : List α
                                              h : LT.lt (HSub.hSub l.length 1) l.length
                                              ⊢ Ne l List.nil
                                            -/
    l.get ⟨l.length - 1, h⟩ = l.getLast (by rintro rfl; exact Nat.lt_irrefl 0 h) :=
                                                        /-
                                                          🎉 no goals
                                                        -/
  (getLast_eq_getElem l _).symm


theorem take_one_drop_eq_of_lt_length {l : List α} {n : ℕ} (h : n < l.length) :
    (l.drop n).take 1 = [l.get ⟨n, h⟩] := by
  /-
    α : Type u
    l : List α
    n : Nat
    h : LT.lt n l.length
    ⊢ Eq (List.take 1 (List.drop n l)) (List.cons (l.get ⟨n, h⟩) List.nil)
  -/
  rw [drop_eq_getElem_cons h, take, take]
  /-
    α : Type u
    l : List α
    n : Nat
    h : LT.lt n l.length
    ⊢ Eq (List.cons (GetElem.getElem l n h) List.nil) (List.cons (l.get ⟨n, h⟩) Li …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem ext_get?' {l₁ l₂ : List α} (h' : ∀ n < max l₁.length l₂.length, l₁.get? n = l₂.get? n) :
    l₁ = l₂ := by
  /-
    α : Type u
    l₁ l₂ : List α
    h' : ∀ (n : Nat), LT.lt n (Max.max l₁.length l₂.length) → Eq (l₁.get? n) (l₂.g …
    ⊢ Eq l₁ l₂
  -/
  apply ext_get?
  /-
    case a
    α : Type u
    l₁ l₂ : List α
    h' : ∀ (n : Nat), LT.lt n (Max.max l₁.length l₂.length) → Eq (l₁.get? n) (l₂.g …
    ⊢ ∀ (n : Nat), Eq (l₁.get? n) (l₂.get? n)
  -/
  intro n
  /-
    case a
    α : Type u
    l₁ l₂ : List α
    h' : ∀ (n : Nat), LT.lt n (Max.max l₁.length l₂.length) → Eq (l₁.get? n) (l₂.g …
    n : Nat
    ⊢ Eq (l₁.get? n) (l₂.get? n)
  -/
  rcases Nat.lt_or_ge n <| max l₁.length l₂.length with hn | hn
    /-
      case a.inl
      α : Type u
      l₁ l₂ : List α
      h' : ∀ (n : Nat), LT.lt n (Max.max l₁.length l₂.length) → Eq (l₁.get? n) (l₂.g …
      n : Nat
      hn : LT.lt n (Max.max l₁.length l₂.length)
      ⊢ Eq (l₁.get? n) (l₂.get? n)
    -/
  · exact h' n hn
    /-
      🎉 no goals
    -/
    /-
      case a.inr
      α : Type u
      l₁ l₂ : List α
      h' : ∀ (n : Nat), LT.lt n (Max.max l₁.length l₂.length) → Eq (l₁.get? n) (l₂.g …
      n : Nat
      hn : GE.ge n (Max.max l₁.length l₂.length)
      ⊢ Eq (l₁.get? n) (l₂.get? n)
    -/
  · simp_all [Nat.max_le, getElem?_eq_none]
    /-
      🎉 no goals
    -/


theorem ext_get?_iff {l₁ l₂ : List α} : l₁ = l₂ ↔ ∀ n, l₁.get? n = l₂.get? n :=
      /-
        α : Type u
        l₁ l₂ : List α
        ⊢ Eq l₁ l₂ → ∀ (n : Nat), Eq (l₁.get? n) (l₂.get? n)
      -/
  ⟨by rintro rfl _; rfl, ext_get?⟩
                    /-
                      🎉 no goals
                    -/


theorem ext_get_iff {l₁ l₂ : List α} :
    l₁ = l₂ ↔ l₁.length = l₂.length ∧ ∀ n h₁ h₂, get l₁ ⟨n, h₁⟩ = get l₂ ⟨n, h₂⟩ := by
  /-
    α : Type u
    l₁ l₂ : List α
    ⊢ Iff (Eq l₁ l₂) (And (Eq l₁.length l₂.length) (∀ (n : Nat) (h₁ : LT.lt n l₁.l …
  -/
  constructor
    /-
      case mp
      α : Type u
      l₁ l₂ : List α
      ⊢ Eq l₁ l₂ → And (Eq l₁.length l₂.length) (∀ (n : Nat) (h₁ : LT.lt n l₁.length …
    -/
  · rintro rfl
    /-
      case mp
      α : Type u
      l₁ : List α
      ⊢ And (Eq l₁.length l₁.length) (∀ (n : Nat) (h₁ h₂ : LT.lt n l₁.length), Eq (l …
    -/
    exact ⟨rfl, fun _ _ _ ↦ rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      l₁ l₂ : List α
      ⊢ And (Eq l₁.length l₂.length) (∀ (n : Nat) (h₁ : LT.lt n l₁.length) (h₂ : LT. …
    -/
  · intro ⟨h₁, h₂⟩
    /-
      case mpr
      α : Type u
      l₁ l₂ : List α
      h₁ : Eq l₁.length l₂.length
      h₂ : ∀ (n : Nat) (h₁ : LT.lt n l₁.length) (h₂ : LT.lt n l₂.length), Eq (l₁.get …
      ⊢ Eq l₁ l₂
    -/
    exact ext_get h₁ h₂
    /-
      🎉 no goals
    -/


theorem ext_get?_iff' {l₁ l₂ : List α} : l₁ = l₂ ↔
    ∀ n < max l₁.length l₂.length, l₁.get? n = l₂.get? n :=
      /-
        α : Type u
        l₁ l₂ : List α
        ⊢ Eq l₁ l₂ → ∀ (n : Nat), LT.lt n (Max.max l₁.length l₂.length) → Eq (l₁.get?  …
      -/
  ⟨by rintro rfl _ _; rfl, ext_get?'⟩
                      /-
                        🎉 no goals
                      -/


/-- If two lists `l₁` and `l₂` are the same length and `l₁[n]! = l₂[n]!` for all `n`,
then the lists are equal. -/
theorem ext_getElem! [Inhabited α] (hl : length l₁ = length l₂) (h : ∀ n : ℕ, l₁[n]! = l₂[n]!) :
    l₁ = l₂ :=
                                  /-
                                    α : Type u
                                    l₁ l₂ : List α
                                    inst✝ : Inhabited α
                                    hl : Eq l₁.length l₂.length
                                    h : ∀ (n : Nat), Eq (GetElem?.getElem! l₁ n) (GetElem?.getElem! l₂ n)
                                    n : Nat
                                    h₁ : LT.lt n l₁.length
                                    h₂ : LT.lt n l₂.length
                                    ⊢ Eq (GetElem.getElem l₁ n h₁) (GetElem.getElem l₂ n h₂)
                                  -/
  ext_getElem hl fun n h₁ h₂ ↦ by simpa only [← getElem!_pos] using h n
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem getElem_indexOf [DecidableEq α] {a : α} : ∀ {l : List α} (h : indexOf a l < l.length),
    l[indexOf a l] = a
  | b :: l, h => by
    /-
      α : Type u
      inst✝ : DecidableEq α
      a b : α
      l : List α
      h : LT.lt (List.indexOf a (List.cons b l)) (List.cons b l).length
      ⊢ Eq (GetElem.getElem (List.cons b l) (List.indexOf a (List.cons b l)) h) a
    -/
    by_cases h' : b = a <;>
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      a b : α
      l : List α
      h : LT.lt (List.indexOf a (List.cons b l)) (List.cons b l).length
      h' : Eq b a
      ⊢ Eq (GetElem.getElem (List.cons b l) (List.indexOf a (List.cons b l)) h) a
    -/
    /-
      🎉 no goals
    -/
    simp [h', if_pos, if_false, getElem_indexOf]
    /-
      🎉 no goals
    -/

-- This is incorrectly named and should be `get_indexOf`;
-- this already exists, so will require a deprecation dance.

theorem indexOf_get [DecidableEq α] {a : α} {l : List α} (h) : get l ⟨indexOf a l, h⟩ = a := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    a : α
    l : List α
    h : LT.lt (List.indexOf a l) l.length
    ⊢ Eq (l.get ⟨List.indexOf a l, h⟩) a
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem getElem?_indexOf [DecidableEq α] {a : α} {l : List α} (h : a ∈ l) :
                                   /-
                                     α : Type u
                                     inst✝ : DecidableEq α
                                     a : α
                                     l : List α
                                     h : Membership.mem l a
                                     ⊢ Eq (GetElem?.getElem? l (List.indexOf a l)) (Option.some a)
                                   -/
    l[indexOf a l]? = some a := by rw [getElem?_eq_getElem, getElem_indexOf (indexOf_lt_length.2 h)]
                                   /-
                                     🎉 no goals
                                   -/

-- This is incorrectly named and should be `get?_indexOf`;
-- this already exists, so will require a deprecation dance.

theorem indexOf_get? [DecidableEq α] {a : α} {l : List α} (h : a ∈ l) :
                                        /-
                                          α : Type u
                                          inst✝ : DecidableEq α
                                          a : α
                                          l : List α
                                          h : Membership.mem l a
                                          ⊢ Eq (l.get? (List.indexOf a l)) (Option.some a)
                                        -/
    get? l (indexOf a l) = some a := by simp [h]
                                        /-
                                          🎉 no goals
                                        -/


theorem indexOf_inj [DecidableEq α] {l : List α} {x y : α} (hx : x ∈ l) (hy : y ∈ l) :
    indexOf x l = indexOf y l ↔ x = y :=
  ⟨fun h => by
    have x_eq_y :
        get l ⟨indexOf x l, indexOf_lt_length.2 hx⟩ =
        get l ⟨indexOf y l, indexOf_lt_length.2 hy⟩ := by
      simp only [h]
    /-
      α : Type u
      inst✝ : DecidableEq α
      l : List α
      x y : α
      hx : Membership.mem l x
      hy : Membership.mem l y
      h : Eq (List.indexOf x l) (List.indexOf y l)
      x_eq_y : Eq (l.get ⟨List.indexOf x l, ⋯⟩) (l.get ⟨List.indexOf y l, ⋯⟩)
      ⊢ Eq x y
    -/
                                       /-
                                         🎉 no goals
                                       -/
    simp only [indexOf_get] at x_eq_y; exact x_eq_y, fun h => by subst h; rfl⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[deprecated getElem_reverse (since := "2024-06-12")]
theorem get_reverse (l : List α) (i : Nat) (h1 h2) :
    get (reverse l) ⟨length l - 1 - i, h1⟩ = get l ⟨i, h2⟩ := by
  /-
    α : Type u
    l : List α
    i : Nat
    h1 : LT.lt (HSub.hSub (HSub.hSub l.length 1) i) l.reverse.length
    h2 : LT.lt i l.length
    ⊢ Eq (l.reverse.get ⟨HSub.hSub (HSub.hSub l.length 1) i, h1⟩) (l.get ⟨i, h2⟩)
  -/
  rw [get_eq_getElem, get_eq_getElem, getElem_reverse]
  /-
    α : Type u
    l : List α
    i : Nat
    h1 : LT.lt (HSub.hSub (HSub.hSub l.length 1) i) l.reverse.length
    h2 : LT.lt i l.length
    ⊢ Eq (GetElem.getElem l (HSub.hSub (HSub.hSub l.length 1) ↑⟨HSub.hSub (HSub.hS …
  -/
  congr
  /-
    case e_i
    α : Type u
    l : List α
    i : Nat
    h1 : LT.lt (HSub.hSub (HSub.hSub l.length 1) i) l.reverse.length
    h2 : LT.lt i l.length
    ⊢ Eq (HSub.hSub (HSub.hSub l.length 1) ↑⟨HSub.hSub (HSub.hSub l.length 1) i, h …
  -/
  dsimp
  /-
    case e_i
    α : Type u
    l : List α
    i : Nat
    h1 : LT.lt (HSub.hSub (HSub.hSub l.length 1) i) l.reverse.length
    h2 : LT.lt i l.length
    ⊢ Eq (HSub.hSub (HSub.hSub l.length 1) (HSub.hSub (HSub.hSub l.length 1) i)) i
  -/
  omega
  /-
    🎉 no goals
  -/


theorem get_reverse' (l : List α) (n) (hn') :
    l.reverse.get n = l.get ⟨l.length - 1 - n, hn'⟩ := by
  /-
    α : Type u
    l : List α
    n : Fin l.reverse.length
    hn' : LT.lt (HSub.hSub (HSub.hSub l.length 1) ↑n) l.length
    ⊢ Eq (l.reverse.get n) (l.get ⟨HSub.hSub (HSub.hSub l.length 1) ↑n, hn'⟩)
  -/
  simp
  /-
    🎉 no goals
  -/


                                                                                  /-
                                                                                    ι : Type u_1
                                                                                    α : Type u
                                                                                    β : Type v
                                                                                    γ : Type w
                                                                                    l₁ l₂ l : List α
                                                                                    h : Eq l.length 1
                                                                                    ⊢ LT.lt 0 l.length
                                                                                  -/
theorem eq_cons_of_length_one {l : List α} (h : l.length = 1) : l = [l.get ⟨0, by omega⟩] := by
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
  /-
    α : Type u
    l : List α
    h : Eq l.length 1
    ⊢ Eq l (List.cons (l.get ⟨0, ⋯⟩) List.nil)
  -/
  refine ext_get (by convert h) fun n h₁ h₂ => ?_
  /-
    α : Type u
    l : List α
    h : Eq l.length 1
    n : Nat
    h₁ : LT.lt n l.length
    h₂ : LT.lt n (List.cons (l.get ⟨0, ⋯⟩) List.nil).length
    ⊢ Eq (l.get ⟨n, h₁⟩) ((List.cons (l.get ⟨0, ⋯⟩) List.nil).get ⟨n, h₂⟩)
  -/
  simp
  /-
    α : Type u
    l : List α
    h : Eq l.length 1
    n : Nat
    h₁ : LT.lt n l.length
    h₂ : LT.lt n (List.cons (l.get ⟨0, ⋯⟩) List.nil).length
    ⊢ Eq (GetElem.getElem l n ⋯) (GetElem.getElem l 0 ⋯)
  -/
  congr
  /-
    case e_i
    α : Type u
    l : List α
    h : Eq l.length 1
    n : Nat
    h₁ : LT.lt n l.length
    h₂ : LT.lt n (List.cons (l.get ⟨0, ⋯⟩) List.nil).length
    ⊢ Eq n 0
  -/
  omega
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-21")]
alias modifyNthTail_modifyNthTail_le := modifyTailIdx_modifyTailIdx_le


@[deprecated (since := "2024-10-21")]
alias modifyNthTail_modifyNthTail_same := modifyTailIdx_modifyTailIdx_self

@[deprecated (since := "2024-05-04")] alias removeNth_eq_nthTail := eraseIdx_eq_modifyTailIdx


@[deprecated (since := "2024-10-21")] alias modifyNth_eq_set := modify_eq_set


@[simp]
theorem getElem_set_of_ne {l : List α} {i j : ℕ} (h : i ≠ j) (a : α)
    (hj : j < (l.set i a).length) :
                              /-
                                ι : Type u_1
                                α : Type u
                                β : Type v
                                γ : Type w
                                l₁ l₂ l : List α
                                i j : Nat
                                h : Ne i j
                                a : α
                                hj : LT.lt j (l.set i a).length
                                ⊢ LT.lt j l.length
                              -/
    (l.set i a)[j] = l[j]'(by simpa using hj) := by
                              /-
                                🎉 no goals
                              -/
  rw [← Option.some_inj, ← List.getElem?_eq_getElem, List.getElem?_set_ne h,
    List.getElem?_eq_getElem]


@[deprecated getElem_set_of_ne (since := "2024-06-12")]
theorem get_set_of_ne {l : List α} {i j : ℕ} (h : i ≠ j) (a : α)
    (hj : j < (l.set i a).length) :
                                           /-
                                             ι : Type u_1
                                             α : Type u
                                             β : Type v
                                             γ : Type w
                                             l₁ l₂ l : List α
                                             i j : Nat
                                             h : Ne i j
                                             a : α
                                             hj : LT.lt j (l.set i a).length
                                             ⊢ LT.lt j l.length
                                           -/
    (l.set i a).get ⟨j, hj⟩ = l.get ⟨j, by simpa using hj⟩ := by
                                           /-
                                             🎉 no goals
                                           -/
  /-
    α : Type u
    l : List α
    i j : Nat
    h : Ne i j
    a : α
    hj : LT.lt j (l.set i a).length
    ⊢ Eq ((l.set i a).get ⟨j, hj⟩) (l.get ⟨j, ⋯⟩)
  -/
  simp [getElem_set_of_ne, h]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-21")] alias map_congr := map_congr_left


theorem flatMap_pure_eq_map (f : α → β) (l : List α) : l.flatMap (pure ∘ f) = map f l :=
  .symm <| map_eq_flatMap ..


@[deprecated (since := "2024-10-16")] alias bind_pure_eq_map := flatMap_pure_eq_map


set_option linter.deprecated false in
@[deprecated flatMap_pure_eq_map (since := "2024-03-24")]
theorem bind_ret_eq_map (f : α → β) (l : List α) : l.bind (List.ret ∘ f) = map f l :=
  bind_pure_eq_map f l


theorem flatMap_congr {l : List α} {f g : α → List β} (h : ∀ x ∈ l, f x = g x) :
    List.flatMap l f = List.flatMap l g :=
  (congr_arg List.flatten <| map_congr_left h : _)


@[deprecated (since := "2024-10-16")] alias bind_congr := flatMap_congr


theorem infix_flatMap_of_mem {a : α} {as : List α} (h : a ∈ as) (f : α → List α) :
    f a <:+: as.flatMap f :=
  List.infix_of_mem_flatten (List.mem_map_of_mem f h)


@[deprecated (since := "2024-10-16")] alias infix_bind_of_mem := infix_flatMap_of_mem


@[simp]
theorem map_eq_map {α β} (f : α → β) (l : List α) : f <$> l = map f l :=
  rfl


/-- A single `List.map` of a composition of functions is equal to
composing a `List.map` with another `List.map`, fully applied.
This is the reverse direction of `List.map_map`.
-/
theorem comp_map (h : β → γ) (g : α → β) (l : List α) : map (h ∘ g) l = map h (map g l) :=
  (map_map _ _ _).symm


/-- Composing a `List.map` with another `List.map` is equal to
a single `List.map` of composed functions.
-/
@[simp]
theorem map_comp_map (g : β → γ) (f : α → β) : map g ∘ map f = map (g ∘ f) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    g : β → γ
    f : α → β
    ⊢ Eq (Function.comp (List.map g) (List.map f)) (List.map (Function.comp g f))
  -/
  ext l; rw [comp_map, Function.comp_apply]
         /-
           🎉 no goals
         -/


theorem _root_.Function.LeftInverse.list_map {f : α → β} {g : β → α} (h : LeftInverse f g) :
    LeftInverse (map f) (map g)
             /-
               α : Type u
               β : Type v
               f : α → β
               g : β → α
               h : Function.LeftInverse f g
               ⊢ Eq (List.map f (List.map g List.nil)) List.nil
             -/
  | [] => by simp_rw [map_nil]
             /-
               🎉 no goals
             -/
                  /-
                    α : Type u
                    β : Type v
                    f : α → β
                    g : β → α
                    h : Function.LeftInverse f g
                    x : β
                    xs : List β
                    ⊢ Eq (List.map f (List.map g (List.cons x xs))) (List.cons x xs)
                  -/
  | x :: xs => by simp_rw [map_cons, h x, h.list_map xs]
                  /-
                    🎉 no goals
                  -/


nonrec theorem _root_.Function.RightInverse.list_map {f : α → β} {g : β → α}
    (h : RightInverse f g) : RightInverse (map f) (map g) :=
  h.list_map


nonrec theorem _root_.Function.Involutive.list_map {f : α → α}
    (h : Involutive f) : Involutive (map f) :=
  Function.LeftInverse.list_map h


@[simp]
theorem map_leftInverse_iff {f : α → β} {g : β → α} :
    LeftInverse (map f) (map g) ↔ LeftInverse f g :=
                 /-
                   α : Type u
                   β : Type v
                   f : α → β
                   g : β → α
                   h : Function.LeftInverse (List.map f) (List.map g)
                   x : β
                   ⊢ Eq (f (g x)) x
                 -/
  ⟨fun h x => by injection h [x], (·.list_map)⟩
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem map_rightInverse_iff {f : α → β} {g : β → α} :
    RightInverse (map f) (map g) ↔ RightInverse f g := map_leftInverse_iff


@[simp]
theorem map_involutive_iff {f : α → α} :
    Involutive (map f) ↔ Involutive f := map_leftInverse_iff


theorem _root_.Function.Injective.list_map {f : α → β} (h : Injective f) :
    Injective (map f)
  | [], [], _ => rfl
  | x :: xs, y :: ys, hxy => by
    /-
      α : Type u
      β : Type v
      f : α → β
      h : Function.Injective f
      x : α
      xs : List α
      y : α
      ys : List α
      hxy : Eq (List.map f (List.cons x xs)) (List.map f (List.cons y ys))
      ⊢ Eq (List.cons x xs) (List.cons y ys)
    -/
    injection hxy with hxy hxys
    /-
      α : Type u
      β : Type v
      f : α → β
      h : Function.Injective f
      x : α
      xs : List α
      y : α
      ys : List α
      hxy : Eq (f x) (f y)
      hxys : Eq (List.map f xs) (List.map f ys)
      ⊢ Eq (List.cons x xs) (List.cons y ys)
    -/
    rw [h hxy, h.list_map hxys]
    /-
      🎉 no goals
    -/


@[simp]
theorem map_injective_iff {f : α → β} : Injective (map f) ↔ Injective f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ⊢ Iff (Function.Injective (List.map f)) (Function.Injective f)
  -/
  refine ⟨fun h x y hxy => ?_, (·.list_map)⟩
  /-
    α : Type u
    β : Type v
    f : α → β
    h : Function.Injective (List.map f)
    x y : α
    hxy : Eq (f x) (f y)
    ⊢ Eq x y
  -/
  suffices [x] = [y] by simpa using this
  /-
    α : Type u
    β : Type v
    f : α → β
    h : Function.Injective (List.map f)
    x y : α
    hxy : Eq (f x) (f y)
    ⊢ Eq (List.cons x List.nil) (List.cons y List.nil)
  -/
  apply h
  /-
    case a
    α : Type u
    β : Type v
    f : α → β
    h : Function.Injective (List.map f)
    x y : α
    hxy : Eq (f x) (f y)
    ⊢ Eq (List.map f (List.cons x List.nil)) (List.map f (List.cons y List.nil))
  -/
  simp [hxy]
  /-
    🎉 no goals
  -/


theorem _root_.Function.Surjective.list_map {f : α → β} (h : Surjective f) :
    Surjective (map f) :=
  let ⟨_, h⟩ := h.hasRightInverse; h.list_map.surjective


@[simp]
theorem map_surjective_iff {f : α → β} : Surjective (map f) ↔ Surjective f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ⊢ Iff (Function.Surjective (List.map f)) (Function.Surjective f)
  -/
  refine ⟨fun h x => ?_, (·.list_map)⟩
  /-
    α : Type u
    β : Type v
    f : α → β
    h : Function.Surjective (List.map f)
    x : β
    ⊢ Exists fun a => Eq (f a) x
  -/
  let ⟨[y], hxy⟩ := h [x]
  /-
    α : Type u
    β : Type v
    f : α → β
    h : Function.Surjective (List.map f)
    x : β
    y : α
    hxy : Eq (List.map f (List.cons y List.nil)) (List.cons x List.nil)
    ⊢ Exists fun a => Eq (f a) x
  -/
  exact ⟨_, List.singleton_injective hxy⟩
  /-
    🎉 no goals
  -/


theorem _root_.Function.Bijective.list_map {f : α → β} (h : Bijective f) : Bijective (map f) :=
  ⟨h.1.list_map, h.2.list_map⟩


@[simp]
theorem map_bijective_iff {f : α → β} : Bijective (map f) ↔ Bijective f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ⊢ Iff (Function.Bijective (List.map f)) (Function.Bijective f)
  -/
  simp_rw [Function.Bijective, map_injective_iff, map_surjective_iff]
  /-
    🎉 no goals
  -/


theorem eq_of_mem_map_const {b₁ b₂ : β} {l : List α} (h : b₁ ∈ map (const α b₂) l) :
                  /-
                    α : Type u
                    β : Type v
                    b₁ b₂ : β
                    l : List α
                    h : Membership.mem (List.map (Function.const α b₂) l) b₁
                    ⊢ Eq b₁ b₂
                  -/
    b₁ = b₂ := by rw [map_const] at h; exact eq_of_mem_replicate h
                                       /-
                                         🎉 no goals
                                       -/


                                                                             /-
                                                                               α : Type u
                                                                               β : Type v
                                                                               γ : Type w
                                                                               f : α → β → γ
                                                                               l : List β
                                                                               ⊢ Eq (List.zipWith f List.nil l) List.nil
                                                                             -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
theorem nil_zipWith (f : α → β → γ) (l : List β) : zipWith f [] l = [] := by cases l <;> rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


                                                                             /-
                                                                               α : Type u
                                                                               β : Type v
                                                                               γ : Type w
                                                                               f : α → β → γ
                                                                               l : List α
                                                                               ⊢ Eq (List.zipWith f l List.nil) List.nil
                                                                             -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
theorem zipWith_nil (f : α → β → γ) (l : List α) : zipWith f l [] = [] := by cases l <;> rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[simp]
theorem zipWith_flip (f : α → β → γ) : ∀ as bs, zipWith (flip f) bs as = zipWith f as bs
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | a :: as, b :: bs => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      f : α → β → γ
      a : α
      as : List α
      b : β
      bs : List β
      ⊢ Eq (List.zipWith (flip f) (List.cons b bs) (List.cons a as)) (List.zipWith f …
    -/
    simp! [zipWith_flip]
    /-
      α : Type u
      β : Type v
      γ : Type w
      f : α → β → γ
      a : α
      as : List α
      b : β
      bs : List β
      ⊢ Eq (flip f b a) (f a b)
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp] lemma take_eq_self_iff (x : List α) {n : ℕ} : x.take n = x ↔ x.length ≤ n :=
              /-
                α : Type u
                x : List α
                n : Nat
                h : Eq (List.take n x) x
                ⊢ LE.le x.length n
              -/
  ⟨fun h ↦ by rw [← h]; simp; omega, take_of_length_le⟩
                              /-
                                🎉 no goals
                              -/


@[simp] lemma take_self_eq_iff (x : List α) {n : ℕ} : x = x.take n ↔ x.length ≤ n := by
  /-
    α : Type u
    x : List α
    n : Nat
    ⊢ Iff (Eq x (List.take n x)) (LE.le x.length n)
  -/
  rw [Eq.comm, take_eq_self_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma take_eq_left_iff {x y : List α} {n : ℕ} :
    (x ++ y).take n = x.take n ↔ y = [] ∨ n ≤ x.length := by
  /-
    α : Type u
    x y : List α
    n : Nat
    ⊢ Iff (Eq (List.take n (HAppend.hAppend x y)) (List.take n x)) (Or (Eq y List. …
  -/
  simp [take_append_eq_append_take, Nat.sub_eq_zero_iff_le, Or.comm]
  /-
    🎉 no goals
  -/


@[simp] lemma left_eq_take_iff {x y : List α} {n : ℕ} :
    x.take n = (x ++ y).take n ↔ y = [] ∨ n ≤ x.length := by
  /-
    α : Type u
    x y : List α
    n : Nat
    ⊢ Iff (Eq (List.take n x) (List.take n (HAppend.hAppend x y))) (Or (Eq y List. …
  -/
  rw [Eq.comm]; apply take_eq_left_iff
                /-
                  🎉 no goals
                -/


@[simp] lemma drop_take_append_drop (x : List α) (m n : ℕ) :
                                                         /-
                                                           α : Type u
                                                           x : List α
                                                           m n : Nat
                                                           ⊢ Eq (HAppend.hAppend (List.take n (List.drop m x)) (List.drop (HAdd.hAdd m n) …
                                                         -/
    (x.drop m).take n ++ x.drop (m + n) = x.drop m := by rw [← drop_drop, take_append_drop]
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Compared to `drop_take_append_drop`, the order of summands is swapped. -/
@[simp] lemma drop_take_append_drop' (x : List α) (m n : ℕ) :
                                                         /-
                                                           α : Type u
                                                           x : List α
                                                           m n : Nat
                                                           ⊢ Eq (HAppend.hAppend (List.take n (List.drop m x)) (List.drop (HAdd.hAdd n m) …
                                                         -/
    (x.drop m).take n ++ x.drop (n + m) = x.drop m := by rw [Nat.add_comm, drop_take_append_drop]
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- `take_concat_get` in simp normal form -/
lemma take_concat_get' (l : List α) (i : ℕ) (h : i < l.length) :
                                            /-
                                              α : Type u
                                              l : List α
                                              i : Nat
                                              h : LT.lt i l.length
                                              ⊢ Eq (HAppend.hAppend (List.take i l) (List.cons (GetElem.getElem l i h) List. …
                                            -/
  l.take i ++ [l[i]] = l.take (i + 1) := by simp
                                            /-
                                              🎉 no goals
                                            -/


/-- `eq_nil_or_concat` in simp normal form -/
lemma eq_nil_or_concat' (l : List α) : l = [] ∨ ∃ L b, l = L ++ [b] := by
  /-
    α : Type u
    l : List α
    ⊢ Or (Eq l List.nil) (Exists fun L => Exists fun b => Eq l (HAppend.hAppend L  …
  -/
  simpa using l.eq_nil_or_concat
  /-
    🎉 no goals
  -/


theorem cons_getElem_drop_succ {l : List α} {n : Nat} {h : n < l.length} :
    l[n] :: l.drop (n + 1) = l.drop n :=
  (drop_eq_getElem_cons h).symm


theorem cons_get_drop_succ {l : List α} {n} :
    l.get n :: l.drop (n.1 + 1) = l.drop n.1 :=
  (drop_eq_getElem_cons n.2).symm


lemma drop_length_sub_one {l : List α} (h : l ≠ []) : l.drop (l.length - 1) = [l.getLast h] := by
  induction l with
  | nil => aesop
  | cons a l ih =>
    by_cases hl : l = []
    · aesop
    rw [length_cons, Nat.add_one_sub_one, List.drop_length_cons hl a]
    aesop


@[simp]
theorem takeI_length : ∀ n l, length (@takeI α _ n l) = n
  | 0, _ => rfl
  | _ + 1, _ => congr_arg succ (takeI_length _ _)


@[simp]
theorem takeI_nil : ∀ n, takeI n (@nil α) = replicate n default
  | 0 => rfl
  | _ + 1 => congr_arg (cons _) (takeI_nil _)


theorem takeI_eq_take : ∀ {n} {l : List α}, n ≤ length l → takeI n l = take n l
  | 0, _, _ => rfl
  | _ + 1, _ :: _, h => congr_arg (cons _) <| takeI_eq_take <| le_of_succ_le_succ h


@[simp]
theorem takeI_left (l₁ l₂ : List α) : takeI (length l₁) (l₁ ++ l₂) = l₁ :=
                     /-
                       α : Type u
                       inst✝ : Inhabited α
                       l₁ l₂ : List α
                       ⊢ LE.le l₁.length (HAppend.hAppend l₁ l₂).length
                     -/
  (takeI_eq_take (by simp only [length_append, Nat.le_add_right])).trans (take_left _ _)
                     /-
                       🎉 no goals
                     -/


theorem takeI_left' {l₁ l₂ : List α} {n} (h : length l₁ = n) : takeI n (l₁ ++ l₂) = l₁ := by
  /-
    α : Type u
    inst✝ : Inhabited α
    l₁ l₂ : List α
    n : Nat
    h : Eq l₁.length n
    ⊢ Eq (List.takeI n (HAppend.hAppend l₁ l₂)) l₁
  -/
  rw [← h]; apply takeI_left
            /-
              🎉 no goals
            -/


@[simp]
theorem takeD_length : ∀ n l a, length (@takeD α n l a) = n
  | 0, _, _ => rfl
  | _ + 1, _, _ => congr_arg succ (takeD_length _ _ _)

-- `takeD_nil` is already in batteries


theorem takeD_eq_take : ∀ {n} {l : List α} a, n ≤ length l → takeD n l a = take n l
  | 0, _, _, _ => rfl
  | _ + 1, _ :: _, a, h => congr_arg (cons _) <| takeD_eq_take a <| le_of_succ_le_succ h


@[simp]
theorem takeD_left (l₁ l₂ : List α) (a : α) : takeD (length l₁) (l₁ ++ l₂) a = l₁ :=
                       /-
                         α : Type u
                         l₁ l₂ : List α
                         a : α
                         ⊢ LE.le l₁.length (HAppend.hAppend l₁ l₂).length
                       -/
  (takeD_eq_take a (by simp only [length_append, Nat.le_add_right])).trans (take_left _ _)
                       /-
                         🎉 no goals
                       -/


theorem takeD_left' {l₁ l₂ : List α} {n} {a} (h : length l₁ = n) : takeD n (l₁ ++ l₂) a = l₁ := by
  /-
    α : Type u
    l₁ l₂ : List α
    n : Nat
    a : α
    h : Eq l₁.length n
    ⊢ Eq (List.takeD n (HAppend.hAppend l₁ l₂) a) l₁
  -/
  rw [← h]; apply takeD_left
            /-
              🎉 no goals
            -/


theorem foldl_ext (f g : α → β → α) (a : α) {l : List β} (H : ∀ a : α, ∀ b ∈ l, f a b = g a b) :
    foldl f a l = foldl g a l := by
  induction l generalizing a with
  | nil => rfl
  | cons hd tl ih =>
    unfold foldl
    rw [ih _ fun a b bin => H a b <| mem_cons_of_mem _ bin, H a hd (mem_cons_self _ _)]


theorem foldr_ext (f g : α → β → β) (b : β) {l : List α} (H : ∀ a ∈ l, ∀ b : β, f a b = g a b) :
    foldr f b l = foldr g b l := by
  /-
    α : Type u
    β : Type v
    f g : α → β → β
    b : β
    l : List α
    H : ∀ (a : α), Membership.mem l a → ∀ (b : β), Eq (f a b) (g a b)
    ⊢ Eq (List.foldr f b l) (List.foldr g b l)
  -/
  induction' l with hd tl ih; · rfl
                                /-
                                  🎉 no goals
                                -/
  /-
    case cons
    α : Type u
    β : Type v
    f g : α → β → β
    b : β
    hd : α
    tl : List α
    ih : (∀ (a : α), Membership.mem tl a → ∀ (b : β), Eq (f a b) (g a b)) → Eq (Li …
    H : ∀ (a : α), Membership.mem (List.cons hd tl) a → ∀ (b : β), Eq (f a b) (g a …
    ⊢ Eq (List.foldr f b (List.cons hd tl)) (List.foldr g b (List.cons hd tl))
  -/
  simp only [mem_cons, or_imp, forall_and, forall_eq] at H
  /-
    case cons
    α : Type u
    β : Type v
    f g : α → β → β
    b : β
    hd : α
    tl : List α
    ih : (∀ (a : α), Membership.mem tl a → ∀ (b : β), Eq (f a b) (g a b)) → Eq (Li …
    H : And (∀ (b : β), Eq (f hd b) (g hd b)) (∀ (x : α), Membership.mem tl x → ∀  …
    ⊢ Eq (List.foldr f b (List.cons hd tl)) (List.foldr g b (List.cons hd tl))
  -/
  simp only [foldr, ih H.2, H.1]
  /-
    🎉 no goals
  -/


theorem foldl_concat
    (f : β → α → β) (b : β) (x : α) (xs : List α) :
    List.foldl f b (xs ++ [x]) = f (List.foldl f b xs) x := by
  /-
    α : Type u
    β : Type v
    f : β → α → β
    b : β
    x : α
    xs : List α
    ⊢ Eq (List.foldl f b (HAppend.hAppend xs (List.cons x List.nil))) (f (List.fol …
  -/
  simp only [List.foldl_append, List.foldl]
  /-
    🎉 no goals
  -/


theorem foldr_concat
    (f : α → β → β) (b : β) (x : α) (xs : List α) :
    List.foldr f b (xs ++ [x]) = (List.foldr f (f x b) xs) := by
  /-
    α : Type u
    β : Type v
    f : α → β → β
    b : β
    x : α
    xs : List α
    ⊢ Eq (List.foldr f b (HAppend.hAppend xs (List.cons x List.nil))) (List.foldr  …
  -/
  simp only [List.foldr_append, List.foldr]
  /-
    🎉 no goals
  -/


theorem foldl_fixed' {f : α → β → α} {a : α} (hf : ∀ b, f a b = a) : ∀ l : List β, foldl f a l = a
  | [] => rfl
                 /-
                   α : Type u
                   β : Type v
                   f : α → β → α
                   a : α
                   hf : ∀ (b : β), Eq (f a b) a
                   b : β
                   l : List β
                   ⊢ Eq (List.foldl f a (List.cons b l)) a
                 -/
  | b :: l => by rw [foldl_cons, hf b, foldl_fixed' hf l]
                 /-
                   🎉 no goals
                 -/


theorem foldr_fixed' {f : α → β → β} {b : β} (hf : ∀ a, f a b = b) : ∀ l : List α, foldr f b l = b
  | [] => rfl
                 /-
                   α : Type u
                   β : Type v
                   f : α → β → β
                   b : β
                   hf : ∀ (a : α), Eq (f a b) b
                   a : α
                   l : List α
                   ⊢ Eq (List.foldr f b (List.cons a l)) b
                 -/
  | a :: l => by rw [foldr_cons, foldr_fixed' hf l, hf a]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem foldl_fixed {a : α} : ∀ l : List β, foldl (fun a _ => a) a l = a :=
  foldl_fixed' fun _ => rfl


@[simp]
theorem foldr_fixed {b : β} : ∀ l : List α, foldr (fun _ b => b) b l = b :=
  foldr_fixed' fun _ => rfl


theorem foldr_eta : ∀ l : List α, foldr cons [] l = l := by
  /-
    α : Type u
    ⊢ ∀ (l : List α), Eq (List.foldr List.cons List.nil l) l
  -/
  simp only [foldr_cons_eq_append, append_nil, forall_const]
  /-
    🎉 no goals
  -/


theorem reverse_foldl {l : List α} : reverse (foldl (fun t h => h :: t) [] l) = l := by
  /-
    α : Type u
    l : List α
    ⊢ Eq (List.foldl (fun t h => List.cons h t) List.nil l).reverse l
  -/
  rw [← foldr_reverse]; simp only [foldr_cons_eq_append, append_nil, reverse_reverse]
                        /-
                          🎉 no goals
                        -/


theorem foldl_hom₂ (l : List ι) (f : α → β → γ) (op₁ : α → ι → α) (op₂ : β → ι → β)
    (op₃ : γ → ι → γ) (a : α) (b : β) (h : ∀ a b i, f (op₁ a i) (op₂ b i) = op₃ (f a b) i) :
    foldl op₃ (f a b) l = f (foldl op₁ a l) (foldl op₂ b l) :=
  Eq.symm <| by
    /-
      ι : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      l : List ι
      f : α → β → γ
      op₁ : α → ι → α
      op₂ : β → ι → β
      op₃ : γ → ι → γ
      a : α
      b : β
      h : ∀ (a : α) (b : β) (i : ι), Eq (f (op₁ a i) (op₂ b i)) (op₃ (f a b) i)
      ⊢ Eq (f (List.foldl op₁ a l) (List.foldl op₂ b l)) (List.foldl op₃ (f a b) l)
    -/
    revert a b
    /-
      ι : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      l : List ι
      f : α → β → γ
      op₁ : α → ι → α
      op₂ : β → ι → β
      op₃ : γ → ι → γ
      h : ∀ (a : α) (b : β) (i : ι), Eq (f (op₁ a i) (op₂ b i)) (op₃ (f a b) i)
      ⊢ ∀ (a : α) (b : β), Eq (f (List.foldl op₁ a l) (List.foldl op₂ b l)) (List.fo …
    -/
    induction l <;> intros <;> [rfl; simp only [*, foldl]]
    /-
      🎉 no goals
    -/


theorem foldr_hom₂ (l : List ι) (f : α → β → γ) (op₁ : ι → α → α) (op₂ : ι → β → β)
    (op₃ : ι → γ → γ) (a : α) (b : β) (h : ∀ a b i, f (op₁ i a) (op₂ i b) = op₃ i (f a b)) :
    foldr op₃ (f a b) l = f (foldr op₁ a l) (foldr op₂ b l) := by
  /-
    ι : Type u_1
    α : Type u
    β : Type v
    γ : Type w
    l : List ι
    f : α → β → γ
    op₁ : ι → α → α
    op₂ : ι → β → β
    op₃ : ι → γ → γ
    a : α
    b : β
    h : ∀ (a : α) (b : β) (i : ι), Eq (f (op₁ i a) (op₂ i b)) (op₃ i (f a b))
    ⊢ Eq (List.foldr op₃ (f a b) l) (f (List.foldr op₁ a l) (List.foldr op₂ b l))
  -/
  revert a
  /-
    ι : Type u_1
    α : Type u
    β : Type v
    γ : Type w
    l : List ι
    f : α → β → γ
    op₁ : ι → α → α
    op₂ : ι → β → β
    op₃ : ι → γ → γ
    b : β
    h : ∀ (a : α) (b : β) (i : ι), Eq (f (op₁ i a) (op₂ i b)) (op₃ i (f a b))
    ⊢ ∀ (a : α), Eq (List.foldr op₃ (f a b) l) (f (List.foldr op₁ a l) (List.foldr …
  -/
  induction l <;> intros <;> [rfl; simp only [*, foldr]]
  /-
    🎉 no goals
  -/


theorem injective_foldl_comp {l : List (α → α)} {f : α → α}
    (hl : ∀ f ∈ l, Function.Injective f) (hf : Function.Injective f) :
    Function.Injective (@List.foldl (α → α) (α → α) Function.comp f l) := by
  /-
    α : Type u
    l : List (α → α)
    f : α → α
    hl : ∀ (f : α → α), Membership.mem l f → Function.Injective f
    hf : Function.Injective f
    ⊢ Function.Injective (List.foldl Function.comp f l)
  -/
  induction' l with lh lt l_ih generalizing f
    /-
      case nil
      α : Type u
      f : α → α
      hl : ∀ (f : α → α), Membership.mem List.nil f → Function.Injective f
      hf : Function.Injective f
      ⊢ Function.Injective (List.foldl Function.comp f List.nil)
    -/
  · exact hf
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      lh : α → α
      lt : List (α → α)
      l_ih : ∀ {f : α → α}, (∀ (f : α → α), Membership.mem lt f → Function.Injective …
      f : α → α
      hl : ∀ (f : α → α), Membership.mem (List.cons lh lt) f → Function.Injective f
      hf : Function.Injective f
      ⊢ Function.Injective (List.foldl Function.comp f (List.cons lh lt))
    -/
  · apply l_ih fun _ h => hl _ (List.mem_cons_of_mem _ h)
    /-
      case cons
      α : Type u
      lh : α → α
      lt : List (α → α)
      l_ih : ∀ {f : α → α}, (∀ (f : α → α), Membership.mem lt f → Function.Injective …
      f : α → α
      hl : ∀ (f : α → α), Membership.mem (List.cons lh lt) f → Function.Injective f
      hf : Function.Injective f
      ⊢ Function.Injective (Function.comp f lh)
    -/
    apply Function.Injective.comp hf
    /-
      case cons
      α : Type u
      lh : α → α
      lt : List (α → α)
      l_ih : ∀ {f : α → α}, (∀ (f : α → α), Membership.mem lt f → Function.Injective …
      f : α → α
      hl : ∀ (f : α → α), Membership.mem (List.cons lh lt) f → Function.Injective f
      hf : Function.Injective f
      ⊢ Function.Injective lh
    -/
    apply hl _ (List.mem_cons_self _ _)
    /-
      🎉 no goals
    -/


/-- Consider two lists `l₁` and `l₂` with designated elements `a₁` and `a₂` somewhere in them:
`l₁ = x₁ ++ [a₁] ++ z₁` and `l₂ = x₂ ++ [a₂] ++ z₂`.
Assume the designated element `a₂` is present in neither `x₁` nor `z₁`.
We conclude that the lists are equal (`l₁ = l₂`) if and only if their respective parts are equal
(`x₁ = x₂ ∧ a₁ = a₂ ∧ z₁ = z₂`). -/
lemma append_cons_inj_of_not_mem {x₁ x₂ z₁ z₂ : List α} {a₁ a₂ : α}
    (notin_x : a₂ ∉ x₁) (notin_z : a₂ ∉ z₁) :
    x₁ ++ a₁ :: z₁ = x₂ ++ a₂ :: z₂ ↔ x₁ = x₂ ∧ a₁ = a₂ ∧ z₁ = z₂ := by
  /-
    α : Type u
    x₁ x₂ z₁ z₂ : List α
    a₁ a₂ : α
    notin_x : Not (Membership.mem x₁ a₂)
    notin_z : Not (Membership.mem z₁ a₂)
    ⊢ Iff (Eq (HAppend.hAppend x₁ (List.cons a₁ z₁)) (HAppend.hAppend x₂ (List.con …
  -/
  constructor
    /-
      case mp
      α : Type u
      x₁ x₂ z₁ z₂ : List α
      a₁ a₂ : α
      notin_x : Not (Membership.mem x₁ a₂)
      notin_z : Not (Membership.mem z₁ a₂)
      ⊢ Eq (HAppend.hAppend x₁ (List.cons a₁ z₁)) (HAppend.hAppend x₂ (List.cons a₂  …
    -/
  · simp only [append_eq_append_iff, cons_eq_append_iff, cons_eq_cons]
    rintro (⟨c, rfl, ⟨rfl, rfl, rfl⟩ | ⟨d, rfl, rfl⟩⟩ |
                                                     /-
                                                       case mp.inl.intro.intro.inl.intro.intro
                                                       α : Type u
                                                       x₁ z₂ : List α
                                                       a₂ : α
                                                       notin_x : Not (Membership.mem x₁ a₂)
                                                       notin_z : Not (Membership.mem z₂ a₂)
                                                       ⊢ And (Eq x₁ (HAppend.hAppend x₁ List.nil)) (And (Eq a₂ a₂) (Eq z₂ z₂))
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
      ⟨c, rfl, ⟨rfl, rfl, rfl⟩ | ⟨d, rfl, rfl⟩⟩) <;> simp_all
                                                     /-
                                                       🎉 no goals
                                                     -/
    /-
      case mpr
      α : Type u
      x₁ x₂ z₁ z₂ : List α
      a₁ a₂ : α
      notin_x : Not (Membership.mem x₁ a₂)
      notin_z : Not (Membership.mem z₁ a₂)
      ⊢ And (Eq x₁ x₂) (And (Eq a₁ a₂) (Eq z₁ z₂)) → Eq (HAppend.hAppend x₁ (List.co …
    -/
  · rintro ⟨rfl, rfl, rfl⟩
    /-
      case mpr.intro.intro
      α : Type u
      x₁ z₁ : List α
      a₁ : α
      notin_x : Not (Membership.mem x₁ a₁)
      notin_z : Not (Membership.mem z₁ a₁)
      ⊢ Eq (HAppend.hAppend x₁ (List.cons a₁ z₁)) (HAppend.hAppend x₁ (List.cons a₁  …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem length_scanl : ∀ a l, length (scanl f a l) = l.length + 1
  | _, [] => rfl
  | a, x :: l => by
    /-
      α : Type u
      β : Type v
      f : β → α → β
      a : β
      x : α
      l : List α
      ⊢ Eq (List.scanl f a (List.cons x l)).length (HAdd.hAdd (List.cons x l).length …
    -/
    rw [scanl, length_cons, length_cons, ← succ_eq_add_one, congr_arg succ]
    /-
      α : Type u
      β : Type v
      f : β → α → β
      a : β
      x : α
      l : List α
      ⊢ Eq (List.scanl f (f a x) l).length (HAdd.hAdd l.length 1)
    -/
    exact length_scanl _ _
    /-
      🎉 no goals
    -/


@[simp]
theorem scanl_nil (b : β) : scanl f b nil = [b] :=
  rfl


@[simp]
theorem scanl_cons : scanl f b (a :: l) = [b] ++ scanl f (f b a) l := by
  /-
    α : Type u
    β : Type v
    f : β → α → β
    b : β
    a : α
    l : List α
    ⊢ Eq (List.scanl f b (List.cons a l)) (HAppend.hAppend (List.cons b List.nil)  …
  -/
  simp only [scanl, eq_self_iff_true, singleton_append, and_self_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem getElem?_scanl_zero : (scanl f b l)[0]? = some b := by
  /-
    α : Type u
    β : Type v
    f : β → α → β
    b : β
    l : List α
    ⊢ Eq (GetElem?.getElem? (List.scanl f b l) 0) (Option.some b)
  -/
  cases l
    /-
      case nil
      α : Type u
      β : Type v
      f : β → α → β
      b : β
      ⊢ Eq (GetElem?.getElem? (List.scanl f b List.nil) 0) (Option.some b)
    -/
  · simp [scanl_nil]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type v
      f : β → α → β
      b : β
      head✝ : α
      tail✝ : List α
      ⊢ Eq (GetElem?.getElem? (List.scanl f b (List.cons head✝ tail✝)) 0) (Option.so …
    -/
  · simp [scanl_cons, singleton_append]
    /-
      🎉 no goals
    -/


@[deprecated getElem?_scanl_zero (since := "2024-06-12")]
theorem get?_zero_scanl : (scanl f b l).get? 0 = some b := by
  /-
    α : Type u
    β : Type v
    f : β → α → β
    b : β
    l : List α
    ⊢ Eq ((List.scanl f b l).get? 0) (Option.some b)
  -/
  simp [getElem?_scanl_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem getElem_scanl_zero {h : 0 < (scanl f b l).length} : (scanl f b l)[0] = b := by
  /-
    α : Type u
    β : Type v
    f : β → α → β
    b : β
    l : List α
    h : LT.lt 0 (List.scanl f b l).length
    ⊢ Eq (GetElem.getElem (List.scanl f b l) 0 h) b
  -/
  cases l
    /-
      case nil
      α : Type u
      β : Type v
      f : β → α → β
      b : β
      h : LT.lt 0 (List.scanl f b List.nil).length
      ⊢ Eq (GetElem.getElem (List.scanl f b List.nil) 0 h) b
    -/
  · simp [scanl_nil]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type v
      f : β → α → β
      b : β
      head✝ : α
      tail✝ : List α
      h : LT.lt 0 (List.scanl f b (List.cons head✝ tail✝)).length
      ⊢ Eq (GetElem.getElem (List.scanl f b (List.cons head✝ tail✝)) 0 h) b
    -/
  · simp [scanl_cons, singleton_append]
    /-
      🎉 no goals
    -/


@[deprecated getElem_scanl_zero (since := "2024-06-12")]
theorem get_zero_scanl {h : 0 < (scanl f b l).length} : (scanl f b l).get ⟨0, h⟩ = b := by
  /-
    α : Type u
    β : Type v
    f : β → α → β
    b : β
    l : List α
    h : LT.lt 0 (List.scanl f b l).length
    ⊢ Eq ((List.scanl f b l).get ⟨0, h⟩) b
  -/
  simp [getElem_scanl_zero]
  /-
    🎉 no goals
  -/


theorem get?_succ_scanl {i : ℕ} : (scanl f b l).get? (i + 1) =
    ((scanl f b l).get? i).bind fun x => (l.get? i).map fun y => f x y := by
  /-
    α : Type u
    β : Type v
    f : β → α → β
    b : β
    l : List α
    i : Nat
    ⊢ Eq ((List.scanl f b l).get? (HAdd.hAdd i 1)) (((List.scanl f b l).get? i).bi …
  -/
  induction' l with hd tl hl generalizing b i
    /-
      case nil
      α : Type u
      β : Type v
      f : β → α → β
      b : β
      i : Nat
      ⊢ Eq ((List.scanl f b List.nil).get? (HAdd.hAdd i 1)) (((List.scanl f b List.n …
    -/
  · symm
    simp only [Option.bind_eq_none', get?, forall₂_true_iff, not_false_iff, Option.map_none',
      scanl_nil, Option.not_mem_none, forall_true_iff]
    /-
      case cons
      α : Type u
      β : Type v
      f : β → α → β
      hd : α
      tl : List α
      hl : ∀ {b : β} {i : Nat}, Eq ((List.scanl f b tl).get? (HAdd.hAdd i 1)) (((Lis …
      b : β
      i : Nat
      ⊢ Eq ((List.scanl f b (List.cons hd tl)).get? (HAdd.hAdd i 1)) (((List.scanl f …
    -/
  · simp only [scanl_cons, singleton_append]
    /-
      case cons
      α : Type u
      β : Type v
      f : β → α → β
      hd : α
      tl : List α
      hl : ∀ {b : β} {i : Nat}, Eq ((List.scanl f b tl).get? (HAdd.hAdd i 1)) (((Lis …
      b : β
      i : Nat
      ⊢ Eq ((List.cons b (List.scanl f (f b hd) tl)).get? (HAdd.hAdd i 1)) (((List.c …
    -/
    cases i
      /-
        case cons.zero
        α : Type u
        β : Type v
        f : β → α → β
        hd : α
        tl : List α
        hl : ∀ {b : β} {i : Nat}, Eq ((List.scanl f b tl).get? (HAdd.hAdd i 1)) (((Lis …
        b : β
        ⊢ Eq ((List.cons b (List.scanl f (f b hd) tl)).get? (HAdd.hAdd 0 1)) (((List.c …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case cons.succ
        α : Type u
        β : Type v
        f : β → α → β
        hd : α
        tl : List α
        hl : ∀ {b : β} {i : Nat}, Eq ((List.scanl f b tl).get? (HAdd.hAdd i 1)) (((Lis …
        b : β
        n✝ : Nat
        ⊢ Eq ((List.cons b (List.scanl f (f b hd) tl)).get? (HAdd.hAdd (HAdd.hAdd n✝ 1 …
      -/
    · simp only [hl, get?]
      /-
        🎉 no goals
      -/


theorem getElem_succ_scanl {i : ℕ} (h : i + 1 < (scanl f b l).length) :
    (scanl f b l)[i + 1] =
      f ((scanl f b l)[i]'(Nat.lt_of_succ_lt h))
        (l[i]'(Nat.lt_of_succ_lt_succ (h.trans_eq (length_scanl b l)))) := by
  induction i generalizing b l with
  | zero =>
    cases l
    · simp only [scanl, length, zero_eq, lt_self_iff_false] at h
    · simp
  | succ i hi =>
    cases l
    · simp only [scanl, length] at h
      exact absurd h (by omega)
    · simp_rw [scanl_cons]
      rw [getElem_append_right]
      · simp only [length, Nat.zero_add 1, succ_add_sub_one, hi]; rfl
      · simp only [length_singleton]; omega


@[deprecated getElem_succ_scanl (since := "2024-08-22")]
theorem get_succ_scanl {i : ℕ} {h : i + 1 < (scanl f b l).length} :
    (scanl f b l).get ⟨i + 1, h⟩ =
      f ((scanl f b l).get ⟨i, Nat.lt_of_succ_lt h⟩)
        (l.get ⟨i, Nat.lt_of_succ_lt_succ (lt_of_lt_of_le h (le_of_eq (length_scanl b l)))⟩) :=
  getElem_succ_scanl h


@[simp]
theorem scanr_nil (f : α → β → β) (b : β) : scanr f b [] = [b] :=
  rfl


@[simp]
theorem scanr_cons (f : α → β → β) (b : β) (a : α) (l : List α) :
    scanr f b (a :: l) = foldr f b (a :: l) :: scanr f b l := by
  /-
    α : Type u
    β : Type v
    f : α → β → β
    b : β
    a : α
    l : List α
    ⊢ Eq (List.scanr f b (List.cons a l)) (List.cons (List.foldr f b (List.cons a  …
  -/
  simp only [scanr, foldr, cons.injEq, and_true]
  induction l generalizing a with
  | nil => rfl
  | cons hd tl ih => simp only [foldr, ih]


theorem foldl1_eq_foldr1 [hassoc : Std.Associative f] :
    ∀ a b l, foldl f a (l ++ [b]) = foldr f b (a :: l)
  | _, _, nil => rfl
  | a, b, c :: l => by
    /-
      α : Type u
      f : α → α → α
      hassoc : Std.Associative f
      a b c : α
      l : List α
      ⊢ Eq (List.foldl f a (HAppend.hAppend (List.cons c l) (List.cons b List.nil))) …
    -/
    simp only [cons_append, foldl_cons, foldr_cons, foldl1_eq_foldr1 _ _ l]
    /-
      α : Type u
      f : α → α → α
      hassoc : Std.Associative f
      a b c : α
      l : List α
      ⊢ Eq (f (f a c) (List.foldr f b l)) (f a (f c (List.foldr f b l)))
    -/
    rw [hassoc.assoc]
    /-
      🎉 no goals
    -/


theorem foldl_eq_of_comm_of_assoc [hcomm : Std.Commutative f] [hassoc : Std.Associative f] :
    ∀ a b l, foldl f a (b :: l) = f b (foldl f a l)
  | a, b, nil => hcomm.comm a b
  | a, b, c :: l => by
    /-
      α : Type u
      f : α → α → α
      hcomm : Std.Commutative f
      hassoc : Std.Associative f
      a b c : α
      l : List α
      ⊢ Eq (List.foldl f a (List.cons b (List.cons c l))) (f b (List.foldl f a (List …
    -/
    simp only [foldl_cons]
    /-
      α : Type u
      f : α → α → α
      hcomm : Std.Commutative f
      hassoc : Std.Associative f
      a b c : α
      l : List α
      ⊢ Eq (List.foldl f (f (f a b) c) l) (f b (List.foldl f (f a c) l))
    -/
    have : RightCommutative f := inferInstance
    /-
      α : Type u
      f : α → α → α
      hcomm : Std.Commutative f
      hassoc : Std.Associative f
      a b c : α
      l : List α
      this : RightCommutative f
      ⊢ Eq (List.foldl f (f (f a b) c) l) (f b (List.foldl f (f a c) l))
    -/
    rw [← foldl_eq_of_comm_of_assoc .., this.right_comm, foldl_cons]
    /-
      🎉 no goals
    -/


theorem foldl_eq_foldr [Std.Commutative f] [Std.Associative f] :
    ∀ a l, foldl f a l = foldr f a l
  | _, nil => rfl
  | a, b :: l => by
    /-
      α : Type u
      f : α → α → α
      inst✝¹ : Std.Commutative f
      inst✝ : Std.Associative f
      a b : α
      l : List α
      ⊢ Eq (List.foldl f a (List.cons b l)) (List.foldr f a (List.cons b l))
    -/
    simp only [foldr_cons, foldl_eq_of_comm_of_assoc]
    /-
      α : Type u
      f : α → α → α
      inst✝¹ : Std.Commutative f
      inst✝ : Std.Associative f
      a b : α
      l : List α
      ⊢ Eq (f b (List.foldl f a l)) (f b (List.foldr f a l))
    -/
    rw [foldl_eq_foldr a l]
    /-
      🎉 no goals
    -/


theorem foldl_eq_of_comm' : ∀ a b l, foldl f a (b :: l) = f (foldl f a l) b
  | _, _, [] => rfl
                       /-
                         α : Type u
                         β : Type v
                         f : α → β → α
                         hf : ∀ (a : α) (b c : β), Eq (f (f a b) c) (f (f a c) b)
                         a : α
                         b c : β
                         l : List β
                         ⊢ Eq (List.foldl f a (List.cons b (List.cons c l))) (f (List.foldl f a (List.c …
                       -/
  | a, b, c :: l => by rw [foldl, foldl, foldl, ← foldl_eq_of_comm' .., foldl, hf]
                       /-
                         🎉 no goals
                       -/


theorem foldl_eq_foldr' : ∀ a l, foldl f a l = foldr (flip f) a l
  | _, [] => rfl
                    /-
                      α : Type u
                      β : Type v
                      f : α → β → α
                      hf : ∀ (a : α) (b c : β), Eq (f (f a b) c) (f (f a c) b)
                      a : α
                      b : β
                      l : List β
                      ⊢ Eq (List.foldl f a (List.cons b l)) (List.foldr (flip f) a (List.cons b l))
                    -/
  | a, b :: l => by rw [foldl_eq_of_comm' hf, foldr, foldl_eq_foldr' ..]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem foldr_eq_of_comm' (hf : ∀ a b c, f a (f b c) = f b (f a c)) :
    ∀ a b l, foldr f a (b :: l) = foldr f (f b a) l
  | _, _, [] => rfl
                       /-
                         α : Type u
                         β : Type v
                         f : α → β → β
                         hf : ∀ (a b : α) (c : β), Eq (f a (f b c)) (f b (f a c))
                         a : β
                         b c : α
                         l : List α
                         ⊢ Eq (List.foldr f a (List.cons b (List.cons c l))) (List.foldr f (f b a) (Lis …
                       -/
  | a, b, c :: l => by rw [foldr, foldr, foldr, hf, ← foldr_eq_of_comm' hf ..]; rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- Notation for `op a b`. -/
local notation a " ⋆ " b => op a b


/-- Notation for `foldl op a l`. -/
local notation l " <*> " a => foldl op a l


theorem foldl_op_eq_op_foldr_assoc :
    ∀ {l : List α} {a₁ a₂}, ((l <*> a₁) ⋆ a₂) = a₁ ⋆ l.foldr (· ⋆ ·) a₂
  | [], _, _ => rfl
  | a :: l, a₁, a₂ => by
    /-
      α : Type u
      op : α → α → α
      ha : Std.Associative op
      a : α
      l : List α
      a₁ a₂ : α
      ⊢ Eq (op (List.foldl op a₁ (List.cons a l)) a₂) (op a₁ (List.foldr (fun x1 x2  …
    -/
    simp only [foldl_cons, foldr_cons, foldl_assoc, ha.assoc]; rw [foldl_op_eq_op_foldr_assoc]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem foldl_assoc_comm_cons {l : List α} {a₁ a₂} : ((a₁ :: l) <*> a₂) = a₁ ⋆ l <*> a₂ := by
  /-
    α : Type u
    op : α → α → α
    ha : Std.Associative op
    hc : Std.Commutative op
    l : List α
    a₁ a₂ : α
    ⊢ Eq (List.foldl op a₂ (List.cons a₁ l)) (op a₁ (List.foldl op a₂ l))
  -/
  rw [foldl_cons, hc.comm, foldl_assoc]
  /-
    🎉 no goals
  -/


theorem foldrM_eq_foldr (f : α → β → m β) (b l) :
                                                                   /-
                                                                     α : Type u
                                                                     β : Type v
                                                                     m : Type v → Type w
                                                                     inst✝¹ : Monad m
                                                                     inst✝ : LawfulMonad m
                                                                     f : α → β → m β
                                                                     b : β
                                                                     l : List α
                                                                     ⊢ Eq (List.foldrM f b l) (List.foldr (fun a mb => Bind.bind mb (f a)) (Pure.pu …
                                                                   -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    foldrM f b l = foldr (fun a mb => mb >>= f a) (pure b) l := by induction l <;> simp [*]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem foldlM_eq_foldl (f : β → α → m β) (b l) :
    List.foldlM f b l = foldl (fun mb a => mb >>= fun b => f b a) (pure b) l := by
  suffices h :
    ∀ mb : m β, (mb >>= fun b => List.foldlM f b l) = foldl (fun mb a => mb >>= fun b => f b a) mb l
    by simp [← h (pure b)]
  induction l with
  | nil => intro; simp
  | cons _ _ l_ih => intro; simp only [List.foldlM, foldl, ← l_ih, functor_norm]


@[simp]
theorem intersperse_singleton (a b : α) : intersperse a [b] = [b] :=
  rfl


@[simp]
theorem intersperse_cons_cons (a b c : α) (tl : List α) :
    intersperse a (b :: c :: tl) = b :: a :: intersperse a (c :: tl) :=
  rfl


@[deprecated (since := "2024-08-17")] alias splitAt_eq_take_drop := splitAt_eq


@[simp]
theorem splitOn_nil [DecidableEq α] (a : α) : [].splitOn a = [[]] :=
  rfl


@[simp]
theorem splitOnP_nil : [].splitOnP p = [[]] :=
  rfl


theorem splitOnP.go_ne_nil (xs acc : List α) : splitOnP.go p xs acc ≠ [] := by
  /-
    α : Type u
    p : α → Bool
    xs acc : List α
    ⊢ Ne (List.splitOnP.go p xs acc) List.nil
  -/
                                    /-
                                      🎉 no goals
                                    -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  induction xs generalizing acc <;> simp [go]; split <;> simp [*]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem splitOnP.go_acc (xs acc : List α) :
    splitOnP.go p xs acc = modifyHead (acc.reverse ++ ·) (splitOnP p xs) := by
  induction xs generalizing acc with
  | nil => simp only [go, modifyHead, splitOnP_nil, append_nil]
  | cons hd tl ih =>
    simp only [splitOnP, go]; split
    · simp only [modifyHead, reverse_nil, append_nil]
    · rw [ih [hd], modifyHead_modifyHead, ih]
      congr; funext x; simp only [reverse_cons, append_assoc]; rfl


theorem splitOnP_ne_nil (xs : List α) : xs.splitOnP p ≠ [] := splitOnP.go_ne_nil _ _ _


@[simp]
theorem splitOnP_cons (x : α) (xs : List α) :
    (x :: xs).splitOnP p =
      if p x then [] :: xs.splitOnP p else (xs.splitOnP p).modifyHead (cons x) := by
  /-
    α : Type u
    p : α → Bool
    x : α
    xs : List α
    ⊢ Eq (List.splitOnP p (List.cons x xs)) (ite (Eq (p x) Bool.true) (List.cons L …
  -/
  rw [splitOnP, splitOnP.go]; split <;> [rfl; simp [splitOnP.go_acc]]
                              /-
                                🎉 no goals
                              -/


/-- The original list `L` can be recovered by flattening the lists produced by `splitOnP p L`,
interspersed with the elements `L.filter p`. -/
theorem splitOnP_spec (as : List α) :
    flatten (zipWith (· ++ ·) (splitOnP p as) (((as.filter p).map fun x => [x]) ++ [[]])) = as := by
  induction as with
  | nil => rfl
  | cons a as' ih =>
    rw [splitOnP_cons, filter]
    by_cases h : p a
    · rw [if_pos h, h, map, cons_append, zipWith, nil_append, flatten, cons_append, cons_inj_right]
      exact ih
    · rw [if_neg h, eq_false_of_ne_true h, flatten_zipWith (splitOnP_ne_nil _ _)
        (append_ne_nil_of_right_ne_nil _ (cons_ne_nil [] [])), cons_inj_right]
      exact ih
where
  flatten_zipWith {xs ys : List (List α)} {a : α} (hxs : xs ≠ []) (hys : ys ≠ []) :
      flatten (zipWith (fun x x_1 ↦ x ++ x_1) (modifyHead (cons a) xs) ys) =
        a :: flatten (zipWith (fun x x_1 ↦ x ++ x_1) xs ys) := by
    cases xs with | nil => contradiction | cons =>
      cases ys with | nil => contradiction | cons => rfl


/-- If no element satisfies `p` in the list `xs`, then `xs.splitOnP p = [xs]` -/
theorem splitOnP_eq_single (h : ∀ x ∈ xs, ¬p x) : xs.splitOnP p = [xs] := by
  induction xs with
  | nil => rfl
  | cons hd tl ih =>
    simp only [splitOnP_cons, h hd (mem_cons_self hd tl), if_neg]
    rw [ih <| forall_mem_of_forall_mem_cons h]
    rfl


/-- When a list of the form `[...xs, sep, ...as]` is split on `p`, the first element is `xs`,
  assuming no element in `xs` satisfies `p` but `sep` does satisfy `p` -/
theorem splitOnP_first (h : ∀ x ∈ xs, ¬p x) (sep : α) (hsep : p sep) (as : List α) :
    (xs ++ sep :: as).splitOnP p = xs :: as.splitOnP p := by
  induction xs with
  | nil => simp [hsep]
  | cons hd tl ih => simp [h hd _, ih <| forall_mem_of_forall_mem_cons h]


/-- `intercalate [x]` is the left inverse of `splitOn x`  -/
theorem intercalate_splitOn (x : α) [DecidableEq α] : [x].intercalate (xs.splitOn x) = xs := by
  /-
    α : Type u
    xs : List α
    x : α
    inst✝ : DecidableEq α
    ⊢ Eq ((List.cons x List.nil).intercalate (List.splitOn x xs)) xs
  -/
  simp only [intercalate, splitOn]
  /-
    α : Type u
    xs : List α
    x : α
    inst✝ : DecidableEq α
    ⊢ Eq (List.intersperse (List.cons x List.nil) (List.splitOnP (fun x_1 => BEq.b …
  -/
  induction' xs with hd tl ih; · simp [flatten]
                                 /-
                                   🎉 no goals
                                 -/
  /-
    case cons
    α : Type u
    xs : List α
    x : α
    inst✝ : DecidableEq α
    hd : α
    tl : List α
    ih : Eq (List.intersperse (List.cons x List.nil) (List.splitOnP (fun x_1 => BE …
    ⊢ Eq (List.intersperse (List.cons x List.nil) (List.splitOnP (fun x_1 => BEq.b …
  -/
  cases' h' : splitOnP (· == x) tl with hd' tl'; · exact (splitOnP_ne_nil _ tl h').elim
                                                   /-
                                                     🎉 no goals
                                                   -/
  /-
    case cons.cons
    α : Type u
    xs : List α
    x : α
    inst✝ : DecidableEq α
    hd : α
    tl : List α
    ih : Eq (List.intersperse (List.cons x List.nil) (List.splitOnP (fun x_1 => BE …
    hd' : List α
    tl' : List (List α)
    h' : Eq (List.splitOnP (fun x_1 => BEq.beq x_1 x) tl) (List.cons hd' tl')
    ⊢ Eq (List.intersperse (List.cons x List.nil) (List.splitOnP (fun x_1 => BEq.b …
  -/
  rw [h'] at ih
  /-
    case cons.cons
    α : Type u
    xs : List α
    x : α
    inst✝ : DecidableEq α
    hd : α
    tl hd' : List α
    tl' : List (List α)
    ih : Eq (List.intersperse (List.cons x List.nil) (List.cons hd' tl')).flatten tl
    h' : Eq (List.splitOnP (fun x_1 => BEq.beq x_1 x) tl) (List.cons hd' tl')
    ⊢ Eq (List.intersperse (List.cons x List.nil) (List.splitOnP (fun x_1 => BEq.b …
  -/
  rw [splitOnP_cons]
  /-
    case cons.cons
    α : Type u
    xs : List α
    x : α
    inst✝ : DecidableEq α
    hd : α
    tl hd' : List α
    tl' : List (List α)
    ih : Eq (List.intersperse (List.cons x List.nil) (List.cons hd' tl')).flatten tl
    h' : Eq (List.splitOnP (fun x_1 => BEq.beq x_1 x) tl) (List.cons hd' tl')
    ⊢ Eq (List.intersperse (List.cons x List.nil) (ite (Eq (BEq.beq hd x) Bool.tru …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u
      xs : List α
      x : α
      inst✝ : DecidableEq α
      hd : α
      tl hd' : List α
      tl' : List (List α)
      ih : Eq (List.intersperse (List.cons x List.nil) (List.cons hd' tl')).flatten tl
      h' : Eq (List.splitOnP (fun x_1 => BEq.beq x_1 x) tl) (List.cons hd' tl')
      h : Eq (BEq.beq hd x) Bool.true
      ⊢ Eq (List.intersperse (List.cons x List.nil) (List.cons List.nil (List.splitO …
    -/
  · rw [beq_iff_eq] at h
    /-
      case pos
      α : Type u
      xs : List α
      x : α
      inst✝ : DecidableEq α
      hd : α
      tl hd' : List α
      tl' : List (List α)
      ih : Eq (List.intersperse (List.cons x List.nil) (List.cons hd' tl')).flatten tl
      h' : Eq (List.splitOnP (fun x_1 => BEq.beq x_1 x) tl) (List.cons hd' tl')
      h : Eq hd x
      ⊢ Eq (List.intersperse (List.cons x List.nil) (List.cons List.nil (List.splitO …
    -/
    subst h
    /-
      case pos
      α : Type u
      xs : List α
      inst✝ : DecidableEq α
      hd : α
      tl hd' : List α
      tl' : List (List α)
      ih : Eq (List.intersperse (List.cons hd List.nil) (List.cons hd' tl')).flatten …
      h' : Eq (List.splitOnP (fun x => BEq.beq x hd) tl) (List.cons hd' tl')
      ⊢ Eq (List.intersperse (List.cons hd List.nil) (List.cons List.nil (List.split …
    -/
    simp [ih, flatten, h']
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u
    xs : List α
    x : α
    inst✝ : DecidableEq α
    hd : α
    tl hd' : List α
    tl' : List (List α)
    ih : Eq (List.intersperse (List.cons x List.nil) (List.cons hd' tl')).flatten tl
    h' : Eq (List.splitOnP (fun x_1 => BEq.beq x_1 x) tl) (List.cons hd' tl')
    h : Not (Eq (BEq.beq hd x) Bool.true)
    ⊢ Eq (List.intersperse (List.cons x List.nil) (List.modifyHead (List.cons hd)  …
  -/
                /-
                  🎉 no goals
                -/
  cases tl' <;> simpa [flatten, h'] using ih
                /-
                  🎉 no goals
                -/


/-- `splitOn x` is the left inverse of `intercalate [x]`, on the domain
  consisting of each nonempty list of lists `ls` whose elements do not contain `x`  -/
theorem splitOn_intercalate [DecidableEq α] (x : α) (hx : ∀ l ∈ ls, x ∉ l) (hls : ls ≠ []) :
    ([x].intercalate ls).splitOn x = ls := by
  /-
    α : Type u
    ls : List (List α)
    inst✝ : DecidableEq α
    x : α
    hx : ∀ (l : List α), Membership.mem ls l → Not (Membership.mem l x)
    hls : Ne ls List.nil
    ⊢ Eq (List.splitOn x ((List.cons x List.nil).intercalate ls)) ls
  -/
  simp only [intercalate]
  /-
    α : Type u
    ls : List (List α)
    inst✝ : DecidableEq α
    x : α
    hx : ∀ (l : List α), Membership.mem ls l → Not (Membership.mem l x)
    hls : Ne ls List.nil
    ⊢ Eq (List.splitOn x (List.intersperse (List.cons x List.nil) ls).flatten) ls
  -/
  induction' ls with hd tl ih; · contradiction
                                 /-
                                   🎉 no goals
                                 -/
  /-
    case cons
    α : Type u
    ls : List (List α)
    inst✝ : DecidableEq α
    x : α
    hd : List α
    tl : List (List α)
    ih : (∀ (l : List α), Membership.mem tl l → Not (Membership.mem l x)) → Ne tl  …
    hx : ∀ (l : List α), Membership.mem (List.cons hd tl) l → Not (Membership.mem  …
    hls : Ne (List.cons hd tl) List.nil
    ⊢ Eq (List.splitOn x (List.intersperse (List.cons x List.nil) (List.cons hd tl …
  -/
  cases tl
    /-
      case cons.nil
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd : List α
      ih : (∀ (l : List α), Membership.mem List.nil l → Not (Membership.mem l x)) →  …
      hx : ∀ (l : List α), Membership.mem (List.cons hd List.nil) l → Not (Membershi …
      hls : Ne (List.cons hd List.nil) List.nil
      ⊢ Eq (List.splitOn x (List.intersperse (List.cons x List.nil) (List.cons hd Li …
    -/
  · suffices hd.splitOn x = [hd] by simpa [flatten]
    /-
      case cons.nil
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd : List α
      ih : (∀ (l : List α), Membership.mem List.nil l → Not (Membership.mem l x)) →  …
      hx : ∀ (l : List α), Membership.mem (List.cons hd List.nil) l → Not (Membershi …
      hls : Ne (List.cons hd List.nil) List.nil
      ⊢ Eq (List.splitOn x hd) (List.cons hd List.nil)
    -/
    refine splitOnP_eq_single _ _ ?_
    /-
      case cons.nil
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd : List α
      ih : (∀ (l : List α), Membership.mem List.nil l → Not (Membership.mem l x)) →  …
      hx : ∀ (l : List α), Membership.mem (List.cons hd List.nil) l → Not (Membershi …
      hls : Ne (List.cons hd List.nil) List.nil
      ⊢ ∀ (x_1 : α), Membership.mem hd x_1 → Not (Eq (BEq.beq x_1 x) Bool.true)
    -/
    intro y hy H
    /-
      case cons.nil
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd : List α
      ih : (∀ (l : List α), Membership.mem List.nil l → Not (Membership.mem l x)) →  …
      hx : ∀ (l : List α), Membership.mem (List.cons hd List.nil) l → Not (Membershi …
      hls : Ne (List.cons hd List.nil) List.nil
      y : α
      hy : Membership.mem hd y
      H : Eq (BEq.beq y x) Bool.true
      ⊢ False
    -/
    rw [eq_of_beq H] at hy
    /-
      case cons.nil
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd : List α
      ih : (∀ (l : List α), Membership.mem List.nil l → Not (Membership.mem l x)) →  …
      hx : ∀ (l : List α), Membership.mem (List.cons hd List.nil) l → Not (Membershi …
      hls : Ne (List.cons hd List.nil) List.nil
      y : α
      hy : Membership.mem hd x
      H : Eq (BEq.beq y x) Bool.true
      ⊢ False
    -/
    refine hx hd ?_ hy
    /-
      case cons.nil
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd : List α
      ih : (∀ (l : List α), Membership.mem List.nil l → Not (Membership.mem l x)) →  …
      hx : ∀ (l : List α), Membership.mem (List.cons hd List.nil) l → Not (Membershi …
      hls : Ne (List.cons hd List.nil) List.nil
      y : α
      hy : Membership.mem hd x
      H : Eq (BEq.beq y x) Bool.true
      ⊢ Membership.mem (List.cons hd List.nil) hd
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case cons.cons
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd head✝ : List α
      tail✝ : List (List α)
      ih : (∀ (l : List α), Membership.mem (List.cons head✝ tail✝) l → Not (Membersh …
      hx : ∀ (l : List α), Membership.mem (List.cons hd (List.cons head✝ tail✝)) l → …
      hls : Ne (List.cons hd (List.cons head✝ tail✝)) List.nil
      ⊢ Eq (List.splitOn x (List.intersperse (List.cons x List.nil) (List.cons hd (L …
    -/
  · simp only [intersperse_cons_cons, singleton_append, flatten]
    /-
      case cons.cons
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd head✝ : List α
      tail✝ : List (List α)
      ih : (∀ (l : List α), Membership.mem (List.cons head✝ tail✝) l → Not (Membersh …
      hx : ∀ (l : List α), Membership.mem (List.cons hd (List.cons head✝ tail✝)) l → …
      hls : Ne (List.cons hd (List.cons head✝ tail✝)) List.nil
      ⊢ Eq (List.splitOn x (HAppend.hAppend hd (List.cons x (List.intersperse (List. …
    -/
    specialize ih _ _
      /-
        case cons.cons.specialize_1
        α : Type u
        ls : List (List α)
        inst✝ : DecidableEq α
        x : α
        hd head✝ : List α
        tail✝ : List (List α)
        ih : (∀ (l : List α), Membership.mem (List.cons head✝ tail✝) l → Not (Membersh …
        hx : ∀ (l : List α), Membership.mem (List.cons hd (List.cons head✝ tail✝)) l → …
        hls : Ne (List.cons hd (List.cons head✝ tail✝)) List.nil
        ⊢ ∀ (l : List α), Membership.mem (List.cons head✝ tail✝) l → Not (Membership.m …
      -/
    · intro l hl
      /-
        case cons.cons.specialize_1
        α : Type u
        ls : List (List α)
        inst✝ : DecidableEq α
        x : α
        hd head✝ : List α
        tail✝ : List (List α)
        ih : (∀ (l : List α), Membership.mem (List.cons head✝ tail✝) l → Not (Membersh …
        hx : ∀ (l : List α), Membership.mem (List.cons hd (List.cons head✝ tail✝)) l → …
        hls : Ne (List.cons hd (List.cons head✝ tail✝)) List.nil
        l : List α
        hl : Membership.mem (List.cons head✝ tail✝) l
        ⊢ Not (Membership.mem l x)
      -/
      apply hx l
      /-
        case cons.cons.specialize_1
        α : Type u
        ls : List (List α)
        inst✝ : DecidableEq α
        x : α
        hd head✝ : List α
        tail✝ : List (List α)
        ih : (∀ (l : List α), Membership.mem (List.cons head✝ tail✝) l → Not (Membersh …
        hx : ∀ (l : List α), Membership.mem (List.cons hd (List.cons head✝ tail✝)) l → …
        hls : Ne (List.cons hd (List.cons head✝ tail✝)) List.nil
        l : List α
        hl : Membership.mem (List.cons head✝ tail✝) l
        ⊢ Membership.mem (List.cons hd (List.cons head✝ tail✝)) l
      -/
      simp only [mem_cons] at hl ⊢
      /-
        case cons.cons.specialize_1
        α : Type u
        ls : List (List α)
        inst✝ : DecidableEq α
        x : α
        hd head✝ : List α
        tail✝ : List (List α)
        ih : (∀ (l : List α), Membership.mem (List.cons head✝ tail✝) l → Not (Membersh …
        hx : ∀ (l : List α), Membership.mem (List.cons hd (List.cons head✝ tail✝)) l → …
        hls : Ne (List.cons hd (List.cons head✝ tail✝)) List.nil
        l : List α
        hl : Or (Eq l head✝) (Membership.mem tail✝ l)
        ⊢ Or (Eq l hd) (Or (Eq l head✝) (Membership.mem tail✝ l))
      -/
      exact Or.inr hl
      /-
        🎉 no goals
      -/
      /-
        case cons.cons.specialize_2
        α : Type u
        ls : List (List α)
        inst✝ : DecidableEq α
        x : α
        hd head✝ : List α
        tail✝ : List (List α)
        ih : (∀ (l : List α), Membership.mem (List.cons head✝ tail✝) l → Not (Membersh …
        hx : ∀ (l : List α), Membership.mem (List.cons hd (List.cons head✝ tail✝)) l → …
        hls : Ne (List.cons hd (List.cons head✝ tail✝)) List.nil
        ⊢ Ne (List.cons head✝ tail✝) List.nil
      -/
    · exact List.noConfusion
      /-
        🎉 no goals
      -/
    /-
      case cons.cons
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd head✝ : List α
      tail✝ : List (List α)
      hx : ∀ (l : List α), Membership.mem (List.cons hd (List.cons head✝ tail✝)) l → …
      hls : Ne (List.cons hd (List.cons head✝ tail✝)) List.nil
      ih : Eq (List.splitOn x (List.intersperse (List.cons x List.nil) (List.cons he …
      ⊢ Eq (List.splitOn x (HAppend.hAppend hd (List.cons x (List.intersperse (List. …
    -/
    have := splitOnP_first (· == x) hd ?h x (beq_self_eq_true _)
    case h =>
      intro y hy H
      rw [eq_of_beq H] at hy
      exact hx hd (.head _) hy
    /-
      case cons.cons
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd head✝ : List α
      tail✝ : List (List α)
      hx : ∀ (l : List α), Membership.mem (List.cons hd (List.cons head✝ tail✝)) l → …
      hls : Ne (List.cons hd (List.cons head✝ tail✝)) List.nil
      ih : Eq (List.splitOn x (List.intersperse (List.cons x List.nil) (List.cons he …
      this : ∀ (as : List α), Eq (List.splitOnP (fun x_1 => BEq.beq x_1 x) (HAppend. …
      ⊢ Eq (List.splitOn x (HAppend.hAppend hd (List.cons x (List.intersperse (List. …
    -/
    simp only [splitOn] at ih ⊢
    /-
      case cons.cons
      α : Type u
      ls : List (List α)
      inst✝ : DecidableEq α
      x : α
      hd head✝ : List α
      tail✝ : List (List α)
      hx : ∀ (l : List α), Membership.mem (List.cons hd (List.cons head✝ tail✝)) l → …
      hls : Ne (List.cons hd (List.cons head✝ tail✝)) List.nil
      ih : Eq (List.splitOnP (fun x_1 => BEq.beq x_1 x) (List.intersperse (List.cons …
      this : ∀ (as : List α), Eq (List.splitOnP (fun x_1 => BEq.beq x_1 x) (HAppend. …
      ⊢ Eq (List.splitOnP (fun x_1 => BEq.beq x_1 x) (HAppend.hAppend hd (List.cons  …
    -/
    rw [this, ih]
    /-
      🎉 no goals
    -/


theorem modifyLast.go_append_one (f : α → α) (a : α) (tl : List α) (r : Array α) :
    modifyLast.go f (tl ++ [a]) r = (r.toListAppend <| modifyLast.go f (tl ++ [a]) #[]) := by
  cases tl with
  | nil =>
    simp only [nil_append, modifyLast.go]; rfl
  | cons hd tl =>
    simp only [cons_append]
    rw [modifyLast.go, modifyLast.go]
    case x_3 | x_3 => exact append_ne_nil_of_right_ne_nil tl (cons_ne_nil a [])
    rw [modifyLast.go_append_one _ _ tl _, modifyLast.go_append_one _ _ tl (Array.push #[] hd)]
    simp only [Array.toListAppend_eq, Array.push_toList, Array.toList_toArray, nil_append,
      append_assoc]


theorem modifyLast_append_one (f : α → α) (a : α) (l : List α) :
    modifyLast f (l ++ [a]) = l ++ [f a] := by
  cases l with
  | nil =>
    simp only [nil_append, modifyLast, modifyLast.go, Array.toListAppend_eq, Array.toList_toArray]
  | cons _ tl =>
    simp only [cons_append, modifyLast]
    rw [modifyLast.go]
    case x_3 => exact append_ne_nil_of_right_ne_nil tl (cons_ne_nil a [])
    rw [modifyLast.go_append_one, Array.toListAppend_eq, Array.push_toList, Array.toList_toArray,
      nil_append, cons_append, nil_append, cons_inj_right]
    exact modifyLast_append_one _ _ tl


theorem modifyLast_append (f : α → α) (l₁ l₂ : List α) (_ : l₂ ≠ []) :
    modifyLast f (l₁ ++ l₂) = l₁ ++ modifyLast f l₂ := by
  cases l₂ with
  | nil => contradiction
  | cons hd tl =>
    cases tl with
    | nil => exact modifyLast_append_one _ hd _
    | cons hd' tl' =>
      rw [append_cons, ← nil_append (hd :: hd' :: tl'), append_cons [], nil_append,
        modifyLast_append _ (l₁ ++ [hd]) (hd' :: tl') _, modifyLast_append _ [hd] (hd' :: tl') _,
        append_assoc]
      all_goals { exact cons_ne_nil _ _ }


theorem sizeOf_lt_sizeOf_of_mem [SizeOf α] {x : α} {l : List α} (hx : x ∈ l) :
    SizeOf.sizeOf x < SizeOf.sizeOf l := by
  /-
    α : Type u
    inst✝ : SizeOf α
    x : α
    l : List α
    hx : Membership.mem l x
    ⊢ LT.lt (SizeOf.sizeOf x) (SizeOf.sizeOf l)
  -/
                               /-
                                 🎉 no goals
                               -/
  induction' l with h t ih <;> cases hx <;> rw [cons.sizeOf_spec]
    /-
      case cons.head
      α : Type u
      inst✝ : SizeOf α
      x : α
      t : List α
      ih : Membership.mem t x → LT.lt (SizeOf.sizeOf x) (SizeOf.sizeOf t)
      ⊢ LT.lt (SizeOf.sizeOf x) (HAdd.hAdd (HAdd.hAdd 1 (SizeOf.sizeOf x)) (SizeOf.s …
    -/
  · omega
    /-
      🎉 no goals
    -/
    /-
      case cons.tail
      α : Type u
      inst✝ : SizeOf α
      x h : α
      t : List α
      ih : Membership.mem t x → LT.lt (SizeOf.sizeOf x) (SizeOf.sizeOf t)
      a✝ : List.Mem x t
      ⊢ LT.lt (SizeOf.sizeOf x) (HAdd.hAdd (HAdd.hAdd 1 (SizeOf.sizeOf h)) (SizeOf.s …
    -/
  · specialize ih ‹_›
    /-
      case cons.tail
      α : Type u
      inst✝ : SizeOf α
      x h : α
      t : List α
      a✝ : List.Mem x t
      ih : LT.lt (SizeOf.sizeOf x) (SizeOf.sizeOf t)
      ⊢ LT.lt (SizeOf.sizeOf x) (HAdd.hAdd (HAdd.hAdd 1 (SizeOf.sizeOf h)) (SizeOf.s …
    -/
    omega
    /-
      🎉 no goals
    -/


@[deprecated attach_map_coe (since := "2024-07-29")] alias attach_map_coe' := attach_map_coe

@[deprecated attach_map_val (since := "2024-07-29")] alias attach_map_val' := attach_map_val


@[deprecated (since := "2024-05-05")] alias find?_mem := mem_of_find?_eq_some


theorem lookmap.go_append (l : List α) (acc : Array α) :
    lookmap.go f l acc = acc.toListAppend (lookmap f l) := by
  cases l with
  | nil => simp [go, lookmap]
  | cons hd tl =>
    rw [lookmap, go, go]
    cases f hd with
    | none =>
      simp only [go_append tl _, Array.toListAppend_eq, append_assoc, Array.push_toList]
      rfl
    | some a => rfl


@[simp]
theorem lookmap_nil : [].lookmap f = [] :=
  rfl


@[simp]
theorem lookmap_cons_none {a : α} (l : List α) (h : f a = none) :
    (a :: l).lookmap f = a :: l.lookmap f := by
  /-
    α : Type u
    f : α → Option α
    a : α
    l : List α
    h : Eq (f a) Option.none
    ⊢ Eq (List.lookmap f (List.cons a l)) (List.cons a (List.lookmap f l))
  -/
  simp only [lookmap, lookmap.go, Array.toListAppend_eq, Array.toList_toArray, nil_append]
  /-
    α : Type u
    f : α → Option α
    a : α
    l : List α
    h : Eq (f a) Option.none
    ⊢ Eq (List.lookmap.go.match_1 (fun x => List α) (f a) (fun b => List.cons b l) …
  -/
  rw [lookmap.go_append, h]; rfl
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem lookmap_cons_some {a b : α} (l : List α) (h : f a = some b) :
    (a :: l).lookmap f = b :: l := by
  /-
    α : Type u
    f : α → Option α
    a b : α
    l : List α
    h : Eq (f a) (Option.some b)
    ⊢ Eq (List.lookmap f (List.cons a l)) (List.cons b l)
  -/
  simp only [lookmap, lookmap.go, Array.toListAppend_eq, Array.toList_toArray, nil_append]
  /-
    α : Type u
    f : α → Option α
    a b : α
    l : List α
    h : Eq (f a) (Option.some b)
    ⊢ Eq (List.lookmap.go.match_1 (fun x => List α) (f a) (fun b => List.cons b l) …
  -/
  rw [h]
  /-
    🎉 no goals
  -/


theorem lookmap_some : ∀ l : List α, l.lookmap some = l
  | [] => rfl
  | _ :: _ => rfl


theorem lookmap_none : ∀ l : List α, (l.lookmap fun _ => none) = l
  | [] => rfl
  | a :: l => (lookmap_cons_none _ l rfl).trans (congr_arg (cons a) (lookmap_none l))


theorem lookmap_congr {f g : α → Option α} :
    ∀ {l : List α}, (∀ a ∈ l, f a = g a) → l.lookmap f = l.lookmap g
  | [], _ => rfl
  | a :: l, H => by
    /-
      α : Type u
      f g : α → Option α
      a : α
      l : List α
      H : ∀ (a_1 : α), Membership.mem (List.cons a l) a_1 → Eq (f a_1) (g a_1)
      ⊢ Eq (List.lookmap f (List.cons a l)) (List.lookmap g (List.cons a l))
    -/
    cases' forall_mem_cons.1 H with H₁ H₂
    /-
      case intro
      α : Type u
      f g : α → Option α
      a : α
      l : List α
      H : ∀ (a_1 : α), Membership.mem (List.cons a l) a_1 → Eq (f a_1) (g a_1)
      H₁ : Eq (f a) (g a)
      H₂ : ∀ (x : α), Membership.mem l x → Eq (f x) (g x)
      ⊢ Eq (List.lookmap f (List.cons a l)) (List.lookmap g (List.cons a l))
    -/
    cases' h : g a with b
      /-
        case intro.none
        α : Type u
        f g : α → Option α
        a : α
        l : List α
        H : ∀ (a_1 : α), Membership.mem (List.cons a l) a_1 → Eq (f a_1) (g a_1)
        H₁ : Eq (f a) (g a)
        H₂ : ∀ (x : α), Membership.mem l x → Eq (f x) (g x)
        h : Eq (g a) Option.none
        ⊢ Eq (List.lookmap f (List.cons a l)) (List.lookmap g (List.cons a l))
      -/
    · simp [h, H₁.trans h, lookmap_congr H₂]
      /-
        🎉 no goals
      -/
      /-
        case intro.some
        α : Type u
        f g : α → Option α
        a : α
        l : List α
        H : ∀ (a_1 : α), Membership.mem (List.cons a l) a_1 → Eq (f a_1) (g a_1)
        H₁ : Eq (f a) (g a)
        H₂ : ∀ (x : α), Membership.mem l x → Eq (f x) (g x)
        b : α
        h : Eq (g a) (Option.some b)
        ⊢ Eq (List.lookmap f (List.cons a l)) (List.lookmap g (List.cons a l))
      -/
    · simp [lookmap_cons_some _ _ h, lookmap_cons_some _ _ (H₁.trans h)]
      /-
        🎉 no goals
      -/


theorem lookmap_of_forall_not {l : List α} (H : ∀ a ∈ l, f a = none) : l.lookmap f = l :=
  (lookmap_congr H).trans (lookmap_none l)


theorem lookmap_map_eq (g : α → β) (h : ∀ (a), ∀ b ∈ f a, g a = g b) :
    ∀ l : List α, map g (l.lookmap f) = map g l
  | [] => rfl
  | a :: l => by
    /-
      α : Type u
      β : Type v
      f : α → Option α
      g : α → β
      h : ∀ (a b : α), Membership.mem (f a) b → Eq (g a) (g b)
      a : α
      l : List α
      ⊢ Eq (List.map g (List.lookmap f (List.cons a l))) (List.map g (List.cons a l))
    -/
    cases' h' : f a with b
      /-
        case none
        α : Type u
        β : Type v
        f : α → Option α
        g : α → β
        h : ∀ (a b : α), Membership.mem (f a) b → Eq (g a) (g b)
        a : α
        l : List α
        h' : Eq (f a) Option.none
        ⊢ Eq (List.map g (List.lookmap f (List.cons a l))) (List.map g (List.cons a l))
      -/
    · simpa [h'] using lookmap_map_eq _ h l
      /-
        🎉 no goals
      -/
      /-
        case some
        α : Type u
        β : Type v
        f : α → Option α
        g : α → β
        h : ∀ (a b : α), Membership.mem (f a) b → Eq (g a) (g b)
        a : α
        l : List α
        b : α
        h' : Eq (f a) (Option.some b)
        ⊢ Eq (List.map g (List.lookmap f (List.cons a l))) (List.map g (List.cons a l))
      -/
    · simp [lookmap_cons_some _ _ h', h _ _ h']
      /-
        🎉 no goals
      -/


theorem lookmap_id' (h : ∀ (a), ∀ b ∈ f a, a = b) (l : List α) : l.lookmap f = l := by
  /-
    α : Type u
    f : α → Option α
    h : ∀ (a b : α), Membership.mem (f a) b → Eq a b
    l : List α
    ⊢ Eq (List.lookmap f l) l
  -/
  rw [← map_id (l.lookmap f), lookmap_map_eq, map_id]; exact h
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem length_lookmap (l : List α) : length (l.lookmap f) = length l := by
  /-
    α : Type u
    f : α → Option α
    l : List α
    ⊢ Eq (List.lookmap f l).length l.length
  -/
  rw [← length_map, lookmap_map_eq _ fun _ => (), length_map]; simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem length_eq_length_filter_add {l : List (α)} (f : α → Bool) :
    l.length = (l.filter f).length + (l.filter (! f ·)).length := by
  simp_rw [← List.countP_eq_length_filter, l.length_eq_countP_add_countP f, Bool.not_eq_true,
    Bool.decide_eq_false]


theorem filterMap_eq_flatMap_toList (f : α → Option β) (l : List α) :
    l.filterMap f = l.flatMap fun a ↦ (f a).toList := by
  /-
    α : Type u
    β : Type v
    f : α → Option β
    l : List α
    ⊢ Eq (List.filterMap f l) (l.flatMap fun a => (f a).toList)
  -/
                               /-
                                 🎉 no goals
                               -/
  induction' l with a l ih <;> simp [filterMap_cons]
  /-
    case cons
    α : Type u
    β : Type v
    f : α → Option β
    a : α
    l : List α
    ih : Eq (List.filterMap f l) (l.flatMap fun a => (f a).toList)
    ⊢ Eq (List.filterMap.match_1 (fun x => List β) (f a) (fun _ => List.filterMap  …
  -/
                 /-
                   🎉 no goals
                 -/
  rcases f a <;> simp [ih]
                 /-
                   🎉 no goals
                 -/


@[deprecated (since := "2024-10-16")] alias filterMap_eq_bind_toList := filterMap_eq_flatMap_toList


theorem filterMap_congr {f g : α → Option β} {l : List α}
    (h : ∀ x ∈ l, f x = g x) : l.filterMap f = l.filterMap g := by
  /-
    α : Type u
    β : Type v
    f g : α → Option β
    l : List α
    h : ∀ (x : α), Membership.mem l x → Eq (f x) (g x)
    ⊢ Eq (List.filterMap f l) (List.filterMap g l)
  -/
                               /-
                                 🎉 no goals
                               -/
  induction' l with a l ih <;> simp [filterMap_cons]
  /-
    case cons
    α : Type u
    β : Type v
    f g : α → Option β
    a : α
    l : List α
    ih : (∀ (x : α), Membership.mem l x → Eq (f x) (g x)) → Eq (List.filterMap f l …
    h : ∀ (x : α), Membership.mem (List.cons a l) x → Eq (f x) (g x)
    ⊢ Eq (List.filterMap.match_1 (fun x => List β) (f a) (fun _ => List.filterMap  …
  -/
  simp [ih (fun x hx ↦ h x (List.mem_cons_of_mem a hx))]
  /-
    case cons
    α : Type u
    β : Type v
    f g : α → Option β
    a : α
    l : List α
    ih : (∀ (x : α), Membership.mem l x → Eq (f x) (g x)) → Eq (List.filterMap f l …
    h : ∀ (x : α), Membership.mem (List.cons a l) x → Eq (f x) (g x)
    ⊢ Eq (List.filterMap.match_1 (fun x => List β) (f a) (fun _ => List.filterMap  …
  -/
  cases' hfa : f a with b
    /-
      case cons.none
      α : Type u
      β : Type v
      f g : α → Option β
      a : α
      l : List α
      ih : (∀ (x : α), Membership.mem l x → Eq (f x) (g x)) → Eq (List.filterMap f l …
      h : ∀ (x : α), Membership.mem (List.cons a l) x → Eq (f x) (g x)
      hfa : Eq (f a) Option.none
      ⊢ Eq (List.filterMap.match_1 (fun x => List β) Option.none (fun _ => List.filt …
    -/
  · have : g a = none := Eq.symm (by simpa [hfa] using h a (by simp))
    /-
      case cons.none
      α : Type u
      β : Type v
      f g : α → Option β
      a : α
      l : List α
      ih : (∀ (x : α), Membership.mem l x → Eq (f x) (g x)) → Eq (List.filterMap f l …
      h : ∀ (x : α), Membership.mem (List.cons a l) x → Eq (f x) (g x)
      hfa : Eq (f a) Option.none
      this : Eq (g a) Option.none
      ⊢ Eq (List.filterMap.match_1 (fun x => List β) Option.none (fun _ => List.filt …
    -/
    simp [this]
    /-
      🎉 no goals
    -/
    /-
      case cons.some
      α : Type u
      β : Type v
      f g : α → Option β
      a : α
      l : List α
      ih : (∀ (x : α), Membership.mem l x → Eq (f x) (g x)) → Eq (List.filterMap f l …
      h : ∀ (x : α), Membership.mem (List.cons a l) x → Eq (f x) (g x)
      b : β
      hfa : Eq (f a) (Option.some b)
      ⊢ Eq (List.filterMap.match_1 (fun x => List β) (Option.some b) (fun _ => List. …
    -/
  · have : g a = some b := Eq.symm (by simpa [hfa] using h a (by simp))
    /-
      case cons.some
      α : Type u
      β : Type v
      f g : α → Option β
      a : α
      l : List α
      ih : (∀ (x : α), Membership.mem l x → Eq (f x) (g x)) → Eq (List.filterMap f l …
      h : ∀ (x : α), Membership.mem (List.cons a l) x → Eq (f x) (g x)
      b : β
      hfa : Eq (f a) (Option.some b)
      this : Eq (g a) (Option.some b)
      ⊢ Eq (List.filterMap.match_1 (fun x => List β) (Option.some b) (fun _ => List. …
    -/
    simp [this]
    /-
      🎉 no goals
    -/


theorem filterMap_eq_map_iff_forall_eq_some {f : α → Option β} {g : α → β} {l : List α} :
    l.filterMap f = l.map g ↔ ∀ x ∈ l, f x = some (g x) where
  mp := by
    /-
      α : Type u
      β : Type v
      f : α → Option β
      g : α → β
      l : List α
      ⊢ Eq (List.filterMap f l) (List.map g l) → ∀ (x : α), Membership.mem l x → Eq  …
    -/
    induction' l with a l ih
      /-
        case nil
        α : Type u
        β : Type v
        f : α → Option β
        g : α → β
        ⊢ Eq (List.filterMap f List.nil) (List.map g List.nil) → ∀ (x : α), Membership …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case cons
      α : Type u
      β : Type v
      f : α → Option β
      g : α → β
      a : α
      l : List α
      ih : Eq (List.filterMap f l) (List.map g l) → ∀ (x : α), Membership.mem l x →  …
      ⊢ Eq (List.filterMap f (List.cons a l)) (List.map g (List.cons a l)) → ∀ (x :  …
    -/
    cases' ha : f a with b <;> simp [ha, filterMap_cons]
      /-
        case cons.none
        α : Type u
        β : Type v
        f : α → Option β
        g : α → β
        a : α
        l : List α
        ih : Eq (List.filterMap f l) (List.map g l) → ∀ (x : α), Membership.mem l x →  …
        ha : Eq (f a) Option.none
        ⊢ Not (Eq (List.filterMap f l) (List.cons (g a) (List.map g l)))
      -/
    · intro h
      simpa [show (filterMap f l).length = l.length + 1 from by simp[h], Nat.add_one_le_iff]
        using List.length_filterMap_le f l
      /-
        case cons.some
        α : Type u
        β : Type v
        f : α → Option β
        g : α → β
        a : α
        l : List α
        ih : Eq (List.filterMap f l) (List.map g l) → ∀ (x : α), Membership.mem l x →  …
        b : β
        ha : Eq (f a) (Option.some b)
        ⊢ Eq b (g a) → Eq (List.filterMap f l) (List.map g l) → And (Eq b (g a)) (∀ (a …
      -/
    · rintro rfl h
      /-
        case cons.some
        α : Type u
        β : Type v
        f : α → Option β
        g : α → β
        a : α
        l : List α
        ih : Eq (List.filterMap f l) (List.map g l) → ∀ (x : α), Membership.mem l x →  …
        ha : Eq (f a) (Option.some (g a))
        h : Eq (List.filterMap f l) (List.map g l)
        ⊢ And (Eq (g a) (g a)) (∀ (a : α), Membership.mem l a → Eq (f a) (Option.some  …
      -/
      exact ⟨rfl, ih h⟩
      /-
        🎉 no goals
      -/
                                           /-
                                             α : Type u
                                             β : Type v
                                             f : α → Option β
                                             g : α → β
                                             l : List α
                                             h : ∀ (x : α), Membership.mem l x → Eq (f x) (Option.some (g x))
                                             ⊢ ∀ (x : α), Membership.mem l x → Eq (f x) (Function.comp Option.some g x)
                                           -/
  mpr h := Eq.trans (filterMap_congr <| by simpa) (congr_fun (List.filterMap_eq_map _) _)
                                           /-
                                             🎉 no goals
                                           -/


theorem filter_singleton {a : α} : [a].filter p = bif p a then [a] else [] :=
  rfl


theorem filter_eq_foldr (p : α → Bool) (l : List α) :
    filter p l = foldr (fun a out => bif p a then a :: out else out) [] l := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    ⊢ Eq (List.filter p l) (List.foldr (fun a out => cond (p a) (List.cons a out)  …
  -/
                  /-
                    🎉 no goals
                  -/
  induction l <;> simp [*, filter]; rfl
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem filter_subset' (l : List α) : filter p l ⊆ l :=
  (filter_sublist l).subset


theorem of_mem_filter {a : α} {l} (h : a ∈ filter p l) : p a := (mem_filter.1 h).2


theorem mem_of_mem_filter {a : α} {l} (h : a ∈ filter p l) : a ∈ l :=
  filter_subset' l h


theorem mem_filter_of_mem {a : α} {l} (h₁ : a ∈ l) (h₂ : p a) : a ∈ filter p l :=
  mem_filter.2 ⟨h₁, h₂⟩


theorem monotone_filter_left (p : α → Bool) ⦃l l' : List α⦄ (h : l ⊆ l') :
    filter p l ⊆ filter p l' := by
  /-
    α : Type u
    p : α → Bool
    l l' : List α
    h : HasSubset.Subset l l'
    ⊢ HasSubset.Subset (List.filter p l) (List.filter p l')
  -/
  intro x hx
  /-
    α : Type u
    p : α → Bool
    l l' : List α
    h : HasSubset.Subset l l'
    x : α
    hx : Membership.mem (List.filter p l) x
    ⊢ Membership.mem (List.filter p l') x
  -/
  rw [mem_filter] at hx ⊢
  /-
    α : Type u
    p : α → Bool
    l l' : List α
    h : HasSubset.Subset l l'
    x : α
    hx : And (Membership.mem l x) (Eq (p x) Bool.true)
    ⊢ And (Membership.mem l' x) (Eq (p x) Bool.true)
  -/
  exact ⟨h hx.left, hx.right⟩
  /-
    🎉 no goals
  -/


theorem monotone_filter_right (l : List α) ⦃p q : α → Bool⦄
    (h : ∀ a, p a → q a) : l.filter p <+ l.filter q := by
  /-
    α : Type u
    l : List α
    p q : α → Bool
    h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
    ⊢ (List.filter p l).Sublist (List.filter q l)
  -/
  induction' l with hd tl IH
    /-
      case nil
      α : Type u
      p q : α → Bool
      h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
      ⊢ (List.filter p List.nil).Sublist (List.filter q List.nil)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      p q : α → Bool
      h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
      hd : α
      tl : List α
      IH : (List.filter p tl).Sublist (List.filter q tl)
      ⊢ (List.filter p (List.cons hd tl)).Sublist (List.filter q (List.cons hd tl))
    -/
  · by_cases hp : p hd
      /-
        case pos
        α : Type u
        p q : α → Bool
        h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
        hd : α
        tl : List α
        IH : (List.filter p tl).Sublist (List.filter q tl)
        hp : Eq (p hd) Bool.true
        ⊢ (List.filter p (List.cons hd tl)).Sublist (List.filter q (List.cons hd tl))
      -/
    · rw [filter_cons_of_pos hp, filter_cons_of_pos (h _ hp)]
      /-
        case pos
        α : Type u
        p q : α → Bool
        h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
        hd : α
        tl : List α
        IH : (List.filter p tl).Sublist (List.filter q tl)
        hp : Eq (p hd) Bool.true
        ⊢ (List.cons hd (List.filter p tl)).Sublist (List.cons hd (List.filter q tl))
      -/
      exact IH.cons_cons hd
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        p q : α → Bool
        h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
        hd : α
        tl : List α
        IH : (List.filter p tl).Sublist (List.filter q tl)
        hp : Not (Eq (p hd) Bool.true)
        ⊢ (List.filter p (List.cons hd tl)).Sublist (List.filter q (List.cons hd tl))
      -/
    · rw [filter_cons_of_neg hp]
      /-
        case neg
        α : Type u
        p q : α → Bool
        h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
        hd : α
        tl : List α
        IH : (List.filter p tl).Sublist (List.filter q tl)
        hp : Not (Eq (p hd) Bool.true)
        ⊢ (List.filter p tl).Sublist (List.filter q (List.cons hd tl))
      -/
      by_cases hq : q hd
        /-
          case pos
          α : Type u
          p q : α → Bool
          h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
          hd : α
          tl : List α
          IH : (List.filter p tl).Sublist (List.filter q tl)
          hp : Not (Eq (p hd) Bool.true)
          hq : Eq (q hd) Bool.true
          ⊢ (List.filter p tl).Sublist (List.filter q (List.cons hd tl))
        -/
      · rw [filter_cons_of_pos hq]
        /-
          case pos
          α : Type u
          p q : α → Bool
          h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
          hd : α
          tl : List α
          IH : (List.filter p tl).Sublist (List.filter q tl)
          hp : Not (Eq (p hd) Bool.true)
          hq : Eq (q hd) Bool.true
          ⊢ (List.filter p tl).Sublist (List.cons hd (List.filter q tl))
        -/
        exact sublist_cons_of_sublist hd IH
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u
          p q : α → Bool
          h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
          hd : α
          tl : List α
          IH : (List.filter p tl).Sublist (List.filter q tl)
          hp : Not (Eq (p hd) Bool.true)
          hq : Not (Eq (q hd) Bool.true)
          ⊢ (List.filter p tl).Sublist (List.filter q (List.cons hd tl))
        -/
      · rw [filter_cons_of_neg hq]
        /-
          case neg
          α : Type u
          p q : α → Bool
          h : ∀ (a : α), Eq (p a) Bool.true → Eq (q a) Bool.true
          hd : α
          tl : List α
          IH : (List.filter p tl).Sublist (List.filter q tl)
          hp : Not (Eq (p hd) Bool.true)
          hq : Not (Eq (q hd) Bool.true)
          ⊢ (List.filter p tl).Sublist (List.filter q tl)
        -/
        exact IH
        /-
          🎉 no goals
        -/

-- TODO rename to `map_filter` when the deprecated `map_filter` is removed from Lean.

lemma map_filter' {f : α → β} (hf : Injective f) (l : List α)
    [DecidablePred fun b => ∃ a, p a ∧ f a = b] :
    (l.filter p).map f = (l.map f).filter fun b => ∃ a, p a ∧ f a = b := by
  /-
    α : Type u
    β : Type v
    p : α → Bool
    f : α → β
    hf : Function.Injective f
    l : List α
    inst✝ : DecidablePred fun b => Exists fun a => And (Eq (p a) Bool.true) (Eq (f …
    ⊢ Eq (List.map f (List.filter p l)) (List.filter (fun b => Decidable.decide (E …
  -/
  simp [comp_def, filter_map, hf.eq_iff]
  /-
    🎉 no goals
  -/


lemma filter_attach' (l : List α) (p : {a // a ∈ l} → Bool) [DecidableEq α] :
    l.attach.filter p =
      (l.filter fun x => ∃ h, p ⟨x, h⟩).attach.map (Subtype.map id fun _ => mem_of_mem_filter) := by
  classical
  refine map_injective_iff.2 Subtype.coe_injective ?_
  simp [comp_def, map_filter' _ Subtype.coe_injective]

-- Porting note: `Lean.Internal.coeM` forces us to type-ascript `{x // x ∈ l}`

lemma filter_attach (l : List α) (p : α → Bool) :
    (l.attach.filter fun x => p x : List {x // x ∈ l}) =
      (l.filter p).attach.map (Subtype.map id fun _ => mem_of_mem_filter) :=
  map_injective_iff.2 Subtype.coe_injective <| by
    simp_rw [map_map, comp_def, Subtype.map, id, ← Function.comp_apply (g := Subtype.val),
      ← filter_map, attach_map_subtype_val]


lemma filter_comm (q) (l : List α) : filter p (filter q l) = filter q (filter p l) := by
  /-
    α : Type u
    p q : α → Bool
    l : List α
    ⊢ Eq (List.filter p (List.filter q l)) (List.filter q (List.filter p l))
  -/
  simp [Bool.and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem filter_true (l : List α) :
                                       /-
                                         α : Type u
                                         l : List α
                                         ⊢ Eq (List.filter (fun x => Bool.true) l) l
                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
    filter (fun _ => true) l = l := by induction l <;> simp [*, filter]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem filter_false (l : List α) :
                                         /-
                                           α : Type u
                                           l : List α
                                           ⊢ Eq (List.filter (fun x => Bool.false) l) List.nil
                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
    filter (fun _ => false) l = [] := by induction l <;> simp [*, filter]
                                                         /-
                                                           🎉 no goals
                                                         -/

/- Porting note: need a helper theorem for span.loop. -/

theorem span.loop_eq_take_drop :
    ∀ l₁ l₂ : List α, span.loop p l₁ l₂ = (l₂.reverse ++ takeWhile p l₁, dropWhile p l₁)
                 /-
                   α : Type u
                   p : α → Bool
                   l₂ : List α
                   ⊢ Eq (List.span.loop p List.nil l₂) { fst := HAppend.hAppend l₂.reverse (List. …
                 -/
  | [], l₂ => by simp [span.loop, takeWhile, dropWhile]
                 /-
                   🎉 no goals
                 -/
  | (a :: l), l₂ => by
    /-
      α : Type u
      p : α → Bool
      a : α
      l l₂ : List α
      ⊢ Eq (List.span.loop p (List.cons a l) l₂) { fst := HAppend.hAppend l₂.reverse …
    -/
                       /-
                         🎉 no goals
                       -/
    cases hp : p a <;> simp [hp, span.loop, span.loop_eq_take_drop, takeWhile, dropWhile]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem span_eq_take_drop (l : List α) : span p l = (takeWhile p l, dropWhile p l) := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    ⊢ Eq (List.span p l) { fst := List.takeWhile p l, snd := List.dropWhile p l }
  -/
  simpa using span.loop_eq_take_drop p l []
  /-
    🎉 no goals
  -/


theorem dropWhile_get_zero_not (l : List α) (hl : 0 < (l.dropWhile p).length) :
    ¬p ((l.dropWhile p).get ⟨0, hl⟩) := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    hl : LT.lt 0 (List.dropWhile p l).length
    ⊢ Not (Eq (p ((List.dropWhile p l).get ⟨0, hl⟩)) Bool.true)
  -/
  induction' l with hd tl IH
    /-
      case nil
      α : Type u
      p : α → Bool
      hl : LT.lt 0 (List.dropWhile p List.nil).length
      ⊢ Not (Eq (p ((List.dropWhile p List.nil).get ⟨0, hl⟩)) Bool.true)
    -/
  · cases hl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      p : α → Bool
      hd : α
      tl : List α
      IH : ∀ (hl : LT.lt 0 (List.dropWhile p tl).length), Not (Eq (p ((List.dropWhil …
      hl : LT.lt 0 (List.dropWhile p (List.cons hd tl)).length
      ⊢ Not (Eq (p ((List.dropWhile p (List.cons hd tl)).get ⟨0, hl⟩)) Bool.true)
    -/
  · simp only [dropWhile]
    /-
      case cons
      α : Type u
      p : α → Bool
      hd : α
      tl : List α
      IH : ∀ (hl : LT.lt 0 (List.dropWhile p tl).length), Not (Eq (p ((List.dropWhil …
      hl : LT.lt 0 (List.dropWhile p (List.cons hd tl)).length
      ⊢ Not (Eq (p ((List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dro …
    -/
    by_cases hp : p hd
      /-
        case pos
        α : Type u
        p : α → Bool
        hd : α
        tl : List α
        IH : ∀ (hl : LT.lt 0 (List.dropWhile p tl).length), Not (Eq (p ((List.dropWhil …
        hl : LT.lt 0 (List.dropWhile p (List.cons hd tl)).length
        hp : Eq (p hd) Bool.true
        ⊢ Not (Eq (p ((List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dro …
      -/
    · simp_all only [get_eq_getElem]
      /-
        case pos
        α : Type u
        p : α → Bool
        hd : α
        tl : List α
        hl : LT.lt 0 (List.dropWhile p (List.cons hd tl)).length
        hp✝ : Eq (p hd) Bool.true
        IH : ∀ (hl : LT.lt 0 (List.dropWhile p tl).length), Not (Eq (p (GetElem.getEle …
        hp : Eq (p hd) Bool.true
        ⊢ Not (Eq (p (GetElem.getElem (List.dropWhile p tl) 0 ⋯)) Bool.true)
      -/
      apply IH
      /-
        case pos
        α : Type u
        p : α → Bool
        hd : α
        tl : List α
        hl : LT.lt 0 (List.dropWhile p (List.cons hd tl)).length
        hp✝ : Eq (p hd) Bool.true
        IH : ∀ (hl : LT.lt 0 (List.dropWhile p tl).length), Not (Eq (p (GetElem.getEle …
        hp : Eq (p hd) Bool.true
        ⊢ LT.lt 0 (List.dropWhile p tl).length
      -/
      simp_all only [dropWhile_cons_of_pos]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        p : α → Bool
        hd : α
        tl : List α
        IH : ∀ (hl : LT.lt 0 (List.dropWhile p tl).length), Not (Eq (p ((List.dropWhil …
        hl : LT.lt 0 (List.dropWhile p (List.cons hd tl)).length
        hp : Not (Eq (p hd) Bool.true)
        ⊢ Not (Eq (p ((List.filter.match_1 (fun x => List α) (p hd) (fun _ => List.dro …
      -/
    · simp [hp]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-08-19")] alias nthLe_cons := getElem_cons

@[deprecated (since := "2024-08-19")] alias dropWhile_nthLe_zero_not := dropWhile_get_zero_not


@[simp]
theorem dropWhile_eq_nil_iff : dropWhile p l = [] ↔ ∀ x ∈ l, p x := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    ⊢ Iff (Eq (List.dropWhile p l) List.nil) (∀ (x : α), Membership.mem l x → Eq ( …
  -/
  induction' l with x xs IH
    /-
      case nil
      α : Type u
      p : α → Bool
      l : List α
      ⊢ Iff (Eq (List.dropWhile p List.nil) List.nil) (∀ (x : α), Membership.mem Lis …
    -/
  · simp [dropWhile]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      p : α → Bool
      l : List α
      x : α
      xs : List α
      IH : Iff (Eq (List.dropWhile p xs) List.nil) (∀ (x : α), Membership.mem xs x → …
      ⊢ Iff (Eq (List.dropWhile p (List.cons x xs)) List.nil) (∀ (x_1 : α), Membersh …
    -/
                          /-
                            🎉 no goals
                          -/
  · by_cases hp : p x <;> simp [hp, IH]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem takeWhile_eq_self_iff : takeWhile p l = l ↔ ∀ x ∈ l, p x := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    ⊢ Iff (Eq (List.takeWhile p l) l) (∀ (x : α), Membership.mem l x → Eq (p x) Bo …
  -/
  induction' l with x xs IH
    /-
      case nil
      α : Type u
      p : α → Bool
      l : List α
      ⊢ Iff (Eq (List.takeWhile p List.nil) List.nil) (∀ (x : α), Membership.mem Lis …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      p : α → Bool
      l : List α
      x : α
      xs : List α
      IH : Iff (Eq (List.takeWhile p xs) xs) (∀ (x : α), Membership.mem xs x → Eq (p …
      ⊢ Iff (Eq (List.takeWhile p (List.cons x xs)) (List.cons x xs)) (∀ (x_1 : α),  …
    -/
                          /-
                            🎉 no goals
                          -/
  · by_cases hp : p x <;> simp [hp, IH]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem takeWhile_eq_nil_iff : takeWhile p l = [] ↔ ∀ hl : 0 < l.length, ¬p (l.get ⟨0, hl⟩) := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    ⊢ Iff (Eq (List.takeWhile p l) List.nil) (∀ (hl : LT.lt 0 l.length), Not (Eq ( …
  -/
  induction' l with x xs IH
    /-
      case nil
      α : Type u
      p : α → Bool
      l : List α
      ⊢ Iff (Eq (List.takeWhile p List.nil) List.nil) (∀ (hl : LT.lt 0 List.nil.leng …
    -/
  · simp only [takeWhile_nil, Bool.not_eq_true, true_iff]
    /-
      case nil
      α : Type u
      p : α → Bool
      l : List α
      ⊢ ∀ (hl : LT.lt 0 List.nil.length), Eq (p (List.nil.get ⟨0, hl⟩)) Bool.false
    -/
    intro h
    /-
      case nil
      α : Type u
      p : α → Bool
      l : List α
      h : LT.lt 0 List.nil.length
      ⊢ Eq (p (List.nil.get ⟨0, h⟩)) Bool.false
    -/
    simp at h
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      p : α → Bool
      l : List α
      x : α
      xs : List α
      IH : Iff (Eq (List.takeWhile p xs) List.nil) (∀ (hl : LT.lt 0 xs.length), Not  …
      ⊢ Iff (Eq (List.takeWhile p (List.cons x xs)) List.nil) (∀ (hl : LT.lt 0 (List …
    -/
                          /-
                            🎉 no goals
                          -/
  · by_cases hp : p x <;> simp [hp, IH]
                          /-
                            🎉 no goals
                          -/


theorem mem_takeWhile_imp {x : α} (hx : x ∈ takeWhile p l) : p x := by
  induction l with simp [takeWhile] at hx
  | cons hd tl IH =>
    cases hp : p hd
    · simp [hp] at hx
    · rw [hp, mem_cons] at hx
      rcases hx with (rfl | hx)
      · exact hp
      · exact IH hx


theorem takeWhile_takeWhile (p q : α → Bool) (l : List α) :
    takeWhile p (takeWhile q l) = takeWhile (fun a => p a ∧ q a) l := by
  /-
    α : Type u
    p q : α → Bool
    l : List α
    ⊢ Eq (List.takeWhile p (List.takeWhile q l)) (List.takeWhile (fun a => Decidab …
  -/
  induction' l with hd tl IH
    /-
      case nil
      α : Type u
      p q : α → Bool
      ⊢ Eq (List.takeWhile p (List.takeWhile q List.nil)) (List.takeWhile (fun a =>  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      p q : α → Bool
      hd : α
      tl : List α
      IH : Eq (List.takeWhile p (List.takeWhile q tl)) (List.takeWhile (fun a => Dec …
      ⊢ Eq (List.takeWhile p (List.takeWhile q (List.cons hd tl))) (List.takeWhile ( …
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
  · by_cases hp : p hd <;> by_cases hq : q hd <;> simp [takeWhile, hp, hq, IH]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem takeWhile_idem : takeWhile p (takeWhile p l) = takeWhile p l := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    ⊢ Eq (List.takeWhile p (List.takeWhile p l)) (List.takeWhile p l)
  -/
  simp_rw [takeWhile_takeWhile, and_self_iff, Bool.decide_coe]
  /-
    🎉 no goals
  -/


lemma find?_eq_head?_dropWhile_not :
    l.find? p = (l.dropWhile (fun x ↦ ! (p x))).head? := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    ⊢ Eq (List.find? p l) (List.dropWhile (fun x => (p x).not) l).head?
  -/
  induction l
  /-
    case nil
    α : Type u
    p : α → Bool
    l : List α
    ⊢ Eq (List.find? p List.nil) (List.dropWhile (fun x => (p x).not) List.nil).he …
  -/
  case nil => simp
  case cons head tail hi =>
    set ph := p head with phh
    rcases ph with rfl | rfl
    · have phh' : ¬(p head = true) := by simp [phh.symm]
      rw [find?_cons_of_neg _ phh', dropWhile_cons_of_pos]
      · exact hi
      · simpa using phh
    · rw [find?_cons_of_pos _ phh.symm, dropWhile_cons_of_neg]
      · simp
      · simpa using phh


lemma find?_not_eq_head?_dropWhile :
    l.find? (fun x ↦ ! (p x)) = (l.dropWhile p).head? := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    ⊢ Eq (List.find? (fun x => (p x).not) l) (List.dropWhile p l).head?
  -/
  convert l.find?_eq_head?_dropWhile_not ?_
  /-
    case h.e'_3.h.e'_2.h.e'_2.h
    α : Type u
    p : α → Bool
    l : List α
    x✝ : α
    ⊢ Eq (p x✝) (p x✝).not.not
  -/
  simp
  /-
    🎉 no goals
  -/


lemma find?_eq_head_dropWhile_not (h : ∃ x ∈ l, p x) :
                                                               /-
                                                                 ι : Type u_1
                                                                 α : Type u
                                                                 β : Type v
                                                                 γ : Type w
                                                                 l₁ l₂ : List α
                                                                 p : α → Bool
                                                                 l : List α
                                                                 h : Exists fun x => And (Membership.mem l x) (Eq (p x) Bool.true)
                                                                 ⊢ Ne (List.dropWhile (fun x => (p x).not) l) List.nil
                                                               -/
    l.find? p = some ((l.dropWhile (fun x ↦ ! (p x))).head (by simpa using h)) := by
                                                               /-
                                                                 🎉 no goals
                                                               -/
  /-
    α : Type u
    p : α → Bool
    l : List α
    h : Exists fun x => And (Membership.mem l x) (Eq (p x) Bool.true)
    ⊢ Eq (List.find? p l) (Option.some ((List.dropWhile (fun x => (p x).not) l).he …
  -/
  rw [l.find?_eq_head?_dropWhile_not p, ← head_eq_iff_head?_eq_some]
  /-
    🎉 no goals
  -/


lemma find?_not_eq_head_dropWhile (h : ∃ x ∈ l, ¬p x) :
                                                               /-
                                                                 ι : Type u_1
                                                                 α : Type u
                                                                 β : Type v
                                                                 γ : Type w
                                                                 l₁ l₂ : List α
                                                                 p : α → Bool
                                                                 l : List α
                                                                 h : Exists fun x => And (Membership.mem l x) (Not (Eq (p x) Bool.true))
                                                                 ⊢ Ne (List.dropWhile p l) List.nil
                                                               -/
    l.find? (fun x ↦ ! (p x)) = some ((l.dropWhile p).head (by simpa using h)) := by
                                                               /-
                                                                 🎉 no goals
                                                               -/
  /-
    α : Type u
    p : α → Bool
    l : List α
    h : Exists fun x => And (Membership.mem l x) (Not (Eq (p x) Bool.true))
    ⊢ Eq (List.find? (fun x => (p x).not) l) (Option.some ((List.dropWhile p l).he …
  -/
  convert l.find?_eq_head_dropWhile_not ?_
    /-
      case h.e'_3.h.e'_2.h.e'_2.h.e'_2.h
      α : Type u
      p : α → Bool
      l : List α
      h : Exists fun x => And (Membership.mem l x) (Not (Eq (p x) Bool.true))
      x✝ : α
      ⊢ Eq (p x✝) (p x✝).not.not
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      α : Type u
      p : α → Bool
      l : List α
      h : Exists fun x => And (Membership.mem l x) (Not (Eq (p x) Bool.true))
      ⊢ Exists fun x => And (Membership.mem l x) (Eq (p x).not Bool.true)
    -/
  · simpa using h
    /-
      🎉 no goals
    -/


@[simp]
theorem length_eraseP_add_one {l : List α} {a} (al : a ∈ l) (pa : p a) :
    (l.eraseP p).length + 1 = l.length := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    a : α
    al : Membership.mem l a
    pa : Eq (p a) Bool.true
    ⊢ Eq (HAdd.hAdd (List.eraseP p l).length 1) l.length
  -/
  let ⟨_, l₁, l₂, _, _, h₁, h₂⟩ := exists_of_eraseP al pa
  /-
    α : Type u
    p : α → Bool
    l : List α
    a : α
    al : Membership.mem l a
    pa : Eq (p a) Bool.true
    w✝ : α
    l₁ l₂ : List α
    left✝¹ : ∀ (b : α), Membership.mem l₁ b → Not (Eq (p b) Bool.true)
    left✝ : Eq (p w✝) Bool.true
    h₁ : Eq l (HAppend.hAppend l₁ (List.cons w✝ l₂))
    h₂ : Eq (List.eraseP p l) (HAppend.hAppend l₁ l₂)
    ⊢ Eq (HAdd.hAdd (List.eraseP p l).length 1) l.length
  -/
  rw [h₂, h₁, length_append, length_append]
  /-
    α : Type u
    p : α → Bool
    l : List α
    a : α
    al : Membership.mem l a
    pa : Eq (p a) Bool.true
    w✝ : α
    l₁ l₂ : List α
    left✝¹ : ∀ (b : α), Membership.mem l₁ b → Not (Eq (p b) Bool.true)
    left✝ : Eq (p w✝) Bool.true
    h₁ : Eq l (HAppend.hAppend l₁ (List.cons w✝ l₂))
    h₂ : Eq (List.eraseP p l) (HAppend.hAppend l₁ l₂)
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd l₁.length l₂.length) 1) (HAdd.hAdd l₁.length (List. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] theorem length_erase_add_one {a : α} {l : List α} (h : a ∈ l) :
    (l.erase a).length + 1 = l.length := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    a : α
    l : List α
    h : Membership.mem l a
    ⊢ Eq (HAdd.hAdd (l.erase a).length 1) l.length
  -/
  rw [erase_eq_eraseP, length_eraseP_add_one h (decide_eq_true rfl)]
  /-
    🎉 no goals
  -/


theorem map_erase [DecidableEq β] {f : α → β} (finj : Injective f) {a : α} (l : List α) :
    map f (l.erase a) = (map f l).erase (f a) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    finj : Function.Injective f
    a : α
    l : List α
    ⊢ Eq (List.map f (l.erase a)) ((List.map f l).erase (f a))
  -/
  have this : (a == ·) = (f a == f ·) := by ext b; simp [beq_eq_decide, finj.eq_iff]
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    finj : Function.Injective f
    a : α
    l : List α
    this : Eq (fun x => BEq.beq a x) fun x => BEq.beq (f a) (f x)
    ⊢ Eq (List.map f (l.erase a)) ((List.map f l).erase (f a))
  -/
  rw [erase_eq_eraseP, erase_eq_eraseP, eraseP_map, this]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem map_foldl_erase [DecidableEq β] {f : α → β} (finj : Injective f) {l₁ l₂ : List α} :
    map f (foldl List.erase l₁ l₂) = foldl (fun l a => l.erase (f a)) (map f l₁) l₂ := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    finj : Function.Injective f
    l₁ l₂ : List α
    ⊢ Eq (List.map f (List.foldl List.erase l₁ l₂)) (List.foldl (fun l a => l.eras …
  -/
  induction l₂ generalizing l₁ <;> [rfl; simp only [foldl_cons, map_erase finj, *]]
  /-
    🎉 no goals
  -/


theorem erase_getElem [DecidableEq ι] {l : List ι} {i : ℕ} (hi : i < l.length) :
    Perm (l.erase l[i]) (l.eraseIdx i) := by
  induction l generalizing i with
  | nil => simp
  | cons a l IH =>
    cases i with
    | zero => simp
    | succ i =>
      have hi' : i < l.length := by simpa using hi
      if ha : a = l[i] then
        simpa [ha] using .trans (perm_cons_erase (getElem_mem _)) (.cons _ (IH hi'))
      else
        simpa [ha] using IH hi'


@[deprecated erase_getElem (since := "2024-08-03")]
theorem erase_get [DecidableEq ι] {l : List ι} (i : Fin l.length) :
    Perm (l.erase (l.get i)) (l.eraseIdx ↑i) :=
  erase_getElem i.isLt


theorem length_eraseIdx_add_one {l : List ι} {i : ℕ} (h : i < l.length) :
    (l.eraseIdx i).length + 1 = l.length := calc
  (l.eraseIdx i).length + 1
                                                            /-
                                                              ι : Type u_1
                                                              l : List ι
                                                              i : Nat
                                                              h : LT.lt i l.length
                                                              ⊢ Eq (HAdd.hAdd (l.eraseIdx i).length 1) (HAdd.hAdd (HAppend.hAppend (List.tak …
                                                            -/
  _ = (l.take i ++ l.drop (i + 1)).length + 1         := by rw [eraseIdx_eq_take_drop_succ]
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              ι : Type u_1
                                                              l : List ι
                                                              i : Nat
                                                              h : LT.lt i l.length
                                                              ⊢ Eq (HAdd.hAdd (HAppend.hAppend (List.take i l) (List.drop (HAdd.hAdd i 1) l) …
                                                            -/
  _ = (l.take i).length + (l.drop (i + 1)).length + 1 := by rw [length_append]
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              ι : Type u_1
                                                              l : List ι
                                                              i : Nat
                                                              h : LT.lt i l.length
                                                              ⊢ Eq (HAdd.hAdd (HAdd.hAdd (List.take i l).length (List.drop (HAdd.hAdd i 1) l …
                                                            -/
  _ = i + (l.drop (i + 1)).length + 1                 := by rw [length_take_of_le (le_of_lt h)]
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              ι : Type u_1
                                                              l : List ι
                                                              i : Nat
                                                              h : LT.lt i l.length
                                                              ⊢ Eq (HAdd.hAdd (HAdd.hAdd i (List.drop (HAdd.hAdd i 1) l).length) 1) (HAdd.hA …
                                                            -/
  _ = i + (l.length - (i + 1)) + 1                    := by rw [length_drop]
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              ι : Type u_1
                                                              l : List ι
                                                              i : Nat
                                                              h : LT.lt i l.length
                                                              ⊢ Eq (HAdd.hAdd (HAdd.hAdd i (HSub.hSub l.length (HAdd.hAdd i 1))) 1) (HAdd.hA …
                                                            -/
  _ = (i + 1) + (l.length - (i + 1))                  := by omega
                                                            /-
                                                              🎉 no goals
                                                            -/
  _ = l.length                                        := Nat.add_sub_cancel' (succ_le_of_lt h)



@[simp]
theorem map_diff [DecidableEq β] {f : α → β} (finj : Injective f) {l₁ l₂ : List α} :
    map f (l₁.diff l₂) = (map f l₁).diff (map f l₂) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    finj : Function.Injective f
    l₁ l₂ : List α
    ⊢ Eq (List.map f (l₁.diff l₂)) ((List.map f l₁).diff (List.map f l₂))
  -/
  simp only [diff_eq_foldl, foldl_map, map_foldl_erase finj]
  /-
    🎉 no goals
  -/


theorem erase_diff_erase_sublist_of_sublist {a : α} :
    ∀ {l₁ l₂ : List α}, l₁ <+ l₂ → (l₂.erase a).diff (l₁.erase a) <+ l₂.diff l₁
  | [], _, _ => erase_sublist _ _
  | b :: l₁, l₂, h =>
                           /-
                             α : Type u
                             inst✝ : DecidableEq α
                             a b : α
                             l₁ l₂ : List α
                             h : (List.cons b l₁).Sublist l₂
                             heq : Eq b a
                             ⊢ ((l₂.erase a).diff ((List.cons b l₁).erase a)).Sublist (l₂.diff (List.cons b …
                           -/
    if heq : b = a then by simp only [heq, erase_cons_head, diff_cons]; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    else by
      simp only [erase_cons_head b l₁, erase_cons_tail (not_beq_of_ne heq),
        diff_cons ((List.erase l₂ a)) (List.erase l₁ a) b, diff_cons l₂ l₁ b, erase_comm a b l₂]
      /-
        α : Type u
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        h : (List.cons b l₁).Sublist l₂
        heq : Not (Eq b a)
        ⊢ (((l₂.erase b).erase a).diff (l₁.erase a)).Sublist ((l₂.erase b).diff l₁)
      -/
      have h' := h.erase b
      /-
        α : Type u
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        h : (List.cons b l₁).Sublist l₂
        heq : Not (Eq b a)
        h' : ((List.cons b l₁).erase b).Sublist (l₂.erase b)
        ⊢ (((l₂.erase b).erase a).diff (l₁.erase a)).Sublist ((l₂.erase b).diff l₁)
      -/
      rw [erase_cons_head] at h'
      /-
        α : Type u
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        h : (List.cons b l₁).Sublist l₂
        heq : Not (Eq b a)
        h' : l₁.Sublist (l₂.erase b)
        ⊢ (((l₂.erase b).erase a).diff (l₁.erase a)).Sublist ((l₂.erase b).diff l₁)
      -/
      exact @erase_diff_erase_sublist_of_sublist _ l₁ (l₂.erase b) h'
      /-
        🎉 no goals
      -/


theorem choose_spec (hp : ∃ a, a ∈ l ∧ p a) : choose p l hp ∈ l ∧ p (choose p l hp) :=
  (chooseX p l hp).property


theorem choose_mem (hp : ∃ a, a ∈ l ∧ p a) : choose p l hp ∈ l :=
  (choose_spec _ _ _).1


theorem choose_property (hp : ∃ a, a ∈ l ∧ p a) : p (choose p l hp) :=
  (choose_spec _ _ _).2


@[simp]
theorem map₂Left'_nil_right (f : α → Option β → γ) (as) :
                                                             /-
                                                               α : Type u
                                                               β : Type v
                                                               γ : Type w
                                                               f : α → Option β → γ
                                                               as : List α
                                                               ⊢ Eq (List.map₂Left' f as List.nil) { fst := List.map (fun a => f a Option.non …
                                                             -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    map₂Left' f as [] = (as.map fun a => f a none, []) := by cases as <;> rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
                                                                               /-
                                                                                 α : Type u
                                                                                 β : Type v
                                                                                 γ : Type w
                                                                                 f : Option α → β → γ
                                                                                 bs : List β
                                                                                 ⊢ Eq (List.map₂Right' f List.nil bs) { fst := List.map (f Option.none) bs, snd …
                                                                               -/
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
theorem map₂Right'_nil_left : map₂Right' f [] bs = (bs.map (f none), []) := by cases bs <;> rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[simp]
theorem map₂Right'_nil_right : map₂Right' f as [] = ([], as) :=
  rfl


@[simp]
theorem map₂Right'_nil_cons : map₂Right' f [] (b :: bs) = (f none b :: bs.map (f none), []) :=
  rfl


@[simp]
theorem map₂Right'_cons_cons :
    map₂Right' f (a :: as) (b :: bs) =
      let r := map₂Right' f as bs
      (f (some a) b :: r.fst, r.snd) :=
  rfl


@[simp]
theorem zipLeft'_nil_right : zipLeft' as ([] : List β) = (as.map fun a => (a, none), []) := by
  /-
    α : Type u
    β : Type v
    as : List α
    ⊢ Eq (as.zipLeft' List.nil) { fst := List.map (fun a => { fst := a, snd := Opt …
  -/
               /-
                 🎉 no goals
               -/
  cases as <;> rfl
               /-
                 🎉 no goals
               -/


@[simp]
theorem zipLeft'_nil_left : zipLeft' ([] : List α) bs = ([], bs) :=
  rfl


@[simp]
theorem zipLeft'_cons_nil :
    zipLeft' (a :: as) ([] : List β) = ((a, none) :: as.map fun a => (a, none), []) :=
  rfl


@[simp]
theorem zipLeft'_cons_cons :
    zipLeft' (a :: as) (b :: bs) =
      let r := zipLeft' as bs
      ((a, some b) :: r.fst, r.snd) :=
  rfl


@[simp]
theorem zipRight'_nil_left : zipRight' ([] : List α) bs = (bs.map fun b => (none, b), []) := by
  /-
    α : Type u
    β : Type v
    bs : List β
    ⊢ Eq (List.nil.zipRight' bs) { fst := List.map (fun b => { fst := Option.none, …
  -/
               /-
                 🎉 no goals
               -/
  cases bs <;> rfl
               /-
                 🎉 no goals
               -/


@[simp]
theorem zipRight'_nil_right : zipRight' as ([] : List β) = ([], as) :=
  rfl


@[simp]
theorem zipRight'_nil_cons :
    zipRight' ([] : List α) (b :: bs) = ((none, b) :: bs.map fun b => (none, b), []) :=
  rfl


@[simp]
theorem zipRight'_cons_cons :
    zipRight' (a :: as) (b :: bs) =
      let r := zipRight' as bs
      ((some a, b) :: r.fst, r.snd) :=
  rfl


@[simp]
                                                                               /-
                                                                                 α : Type u
                                                                                 β : Type v
                                                                                 γ : Type w
                                                                                 f : α → Option β → γ
                                                                                 as : List α
                                                                                 ⊢ Eq (List.map₂Left f as List.nil) (List.map (fun a => f a Option.none) as)
                                                                               -/
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
theorem map₂Left_nil_right : map₂Left f as [] = as.map fun a => f a none := by cases as <;> rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


theorem map₂Left_eq_map₂Left' : ∀ as bs, map₂Left f as bs = (map₂Left' f as bs).fst
                /-
                  α : Type u
                  β : Type v
                  γ : Type w
                  f : α → Option β → γ
                  x✝ : List β
                  ⊢ Eq (List.map₂Left f List.nil x✝) (List.map₂Left' f List.nil x✝).1
                -/
  | [], _ => by simp
                /-
                  🎉 no goals
                -/
                      /-
                        α : Type u
                        β : Type v
                        γ : Type w
                        f : α → Option β → γ
                        a : α
                        as : List α
                        ⊢ Eq (List.map₂Left f (List.cons a as) List.nil) (List.map₂Left' f (List.cons  …
                      -/
  | a :: as, [] => by simp
                      /-
                        🎉 no goals
                      -/
                           /-
                             α : Type u
                             β : Type v
                             γ : Type w
                             f : α → Option β → γ
                             a : α
                             as : List α
                             b : β
                             bs : List β
                             ⊢ Eq (List.map₂Left f (List.cons a as) (List.cons b bs)) (List.map₂Left' f (Li …
                           -/
  | a :: as, b :: bs => by simp [map₂Left_eq_map₂Left']
                           /-
                             🎉 no goals
                           -/


theorem map₂Left_eq_zipWith :
    ∀ as bs, length as ≤ length bs → map₂Left f as bs = zipWith (fun a b => f a (some b)) as bs
                    /-
                      α : Type u
                      β : Type v
                      γ : Type w
                      f : α → Option β → γ
                      x✝ : LE.le List.nil.length List.nil.length
                      ⊢ Eq (List.map₂Left f List.nil List.nil) (List.zipWith (fun a b => f a (Option …
                    -/
  | [], [], _ => by simp
                    /-
                      🎉 no goals
                    -/
                        /-
                          α : Type u
                          β : Type v
                          γ : Type w
                          f : α → Option β → γ
                          head✝ : β
                          tail✝ : List β
                          x✝ : LE.le List.nil.length (List.cons head✝ tail✝).length
                          ⊢ Eq (List.map₂Left f List.nil (List.cons head✝ tail✝)) (List.zipWith (fun a b …
                        -/
  | [], _ :: _, _ => by simp
                        /-
                          🎉 no goals
                        -/
  | a :: as, [], h => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      f : α → Option β → γ
      a : α
      as : List α
      h : LE.le (List.cons a as).length List.nil.length
      ⊢ Eq (List.map₂Left f (List.cons a as) List.nil) (List.zipWith (fun a b => f a …
    -/
    simp at h
    /-
      🎉 no goals
    -/
  | a :: as, b :: bs, h => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      f : α → Option β → γ
      a : α
      as : List α
      b : β
      bs : List β
      h : LE.le (List.cons a as).length (List.cons b bs).length
      ⊢ Eq (List.map₂Left f (List.cons a as) (List.cons b bs)) (List.zipWith (fun a  …
    -/
    simp only [length_cons, succ_le_succ_iff] at h
    /-
      α : Type u
      β : Type v
      γ : Type w
      f : α → Option β → γ
      a : α
      as : List α
      b : β
      bs : List β
      h : LE.le as.length bs.length
      ⊢ Eq (List.map₂Left f (List.cons a as) (List.cons b bs)) (List.zipWith (fun a  …
    -/
    simp [h, map₂Left_eq_zipWith]
    /-
      🎉 no goals
    -/


@[simp]
                                                                       /-
                                                                         α : Type u
                                                                         β : Type v
                                                                         γ : Type w
                                                                         f : Option α → β → γ
                                                                         bs : List β
                                                                         ⊢ Eq (List.map₂Right f List.nil bs) (List.map (f Option.none) bs)
                                                                       -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
theorem map₂Right_nil_left : map₂Right f [] bs = bs.map (f none) := by cases bs <;> rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[simp]
theorem map₂Right_nil_right : map₂Right f as [] = [] :=
  rfl


@[simp]
theorem map₂Right_nil_cons : map₂Right f [] (b :: bs) = f none b :: bs.map (f none) :=
  rfl


@[simp]
theorem map₂Right_cons_cons :
    map₂Right f (a :: as) (b :: bs) = f (some a) b :: map₂Right f as bs :=
  rfl


theorem map₂Right_eq_map₂Right' : map₂Right f as bs = (map₂Right' f as bs).fst := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : Option α → β → γ
    as : List α
    bs : List β
    ⊢ Eq (List.map₂Right f as bs) (List.map₂Right' f as bs).1
  -/
  simp only [map₂Right, map₂Right', map₂Left_eq_map₂Left']
  /-
    🎉 no goals
  -/


theorem map₂Right_eq_zipWith (h : length bs ≤ length as) :
    map₂Right f as bs = zipWith (fun a b => f (some a) b) as bs := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : Option α → β → γ
    as : List α
    bs : List β
    h : LE.le bs.length as.length
    ⊢ Eq (List.map₂Right f as bs) (List.zipWith (fun a b => f (Option.some a) b) a …
  -/
  have : (fun a b => flip f a (some b)) = flip fun a b => f (some a) b := rfl
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : Option α → β → γ
    as : List α
    bs : List β
    h : LE.le bs.length as.length
    this : Eq (fun a b => flip f a (Option.some b)) (flip fun a b => f (Option.som …
    ⊢ Eq (List.map₂Right f as bs) (List.zipWith (fun a b => f (Option.some a) b) a …
  -/
  simp only [map₂Right, map₂Left_eq_zipWith, zipWith_flip, *]
  /-
    🎉 no goals
  -/


@[simp]
theorem zipLeft_nil_right : zipLeft as ([] : List β) = as.map fun a => (a, none) := by
  /-
    α : Type u
    β : Type v
    as : List α
    ⊢ Eq (as.zipLeft List.nil) (List.map (fun a => { fst := a, snd := Option.none  …
  -/
               /-
                 🎉 no goals
               -/
  cases as <;> rfl
               /-
                 🎉 no goals
               -/


@[simp]
theorem zipLeft_nil_left : zipLeft ([] : List α) bs = [] :=
  rfl


@[simp]
theorem zipLeft_cons_nil :
    zipLeft (a :: as) ([] : List β) = (a, none) :: as.map fun a => (a, none) :=
  rfl


@[simp]
theorem zipLeft_cons_cons : zipLeft (a :: as) (b :: bs) = (a, some b) :: zipLeft as bs :=
  rfl

-- Porting note: arguments explicit for recursion

theorem zipLeft_eq_zipLeft' (as : List α) (bs : List β) : zipLeft as bs = (zipLeft' as bs).fst := by
  /-
    α : Type u
    β : Type v
    as : List α
    bs : List β
    ⊢ Eq (as.zipLeft bs) (as.zipLeft' bs).1
  -/
  rw [zipLeft, zipLeft']
  cases as with
  | nil => rfl
  | cons _ atl =>
    cases bs with
    | nil => rfl
    | cons _ btl =>
      rw [zipWithLeft, zipWithLeft', cons_inj_right]
      exact @zipLeft_eq_zipLeft' atl btl


@[simp]
theorem zipRight_nil_left : zipRight ([] : List α) bs = bs.map fun b => (none, b) := by
  /-
    α : Type u
    β : Type v
    bs : List β
    ⊢ Eq (List.nil.zipRight bs) (List.map (fun b => { fst := Option.none, snd := b …
  -/
               /-
                 🎉 no goals
               -/
  cases bs <;> rfl
               /-
                 🎉 no goals
               -/


@[simp]
theorem zipRight_nil_right : zipRight as ([] : List β) = [] :=
  rfl


@[simp]
theorem zipRight_nil_cons :
    zipRight ([] : List α) (b :: bs) = (none, b) :: bs.map fun b => (none, b) :=
  rfl


@[simp]
theorem zipRight_cons_cons : zipRight (a :: as) (b :: bs) = (some a, b) :: zipRight as bs :=
  rfl


theorem zipRight_eq_zipRight' : zipRight as bs = (zipRight' as bs).fst := by
  /-
    α : Type u
    β : Type v
    as : List α
    bs : List β
    ⊢ Eq (as.zipRight bs) (as.zipRight' bs).1
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
  induction as generalizing bs <;> cases bs <;> simp [*]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem forall_cons (p : α → Prop) (x : α) : ∀ l : List α, Forall p (x :: l) ↔ p x ∧ Forall p l
  | [] => (and_iff_left_of_imp fun _ ↦ trivial).symm
  | _ :: _ => Iff.rfl


theorem forall_iff_forall_mem : ∀ {l : List α}, Forall p l ↔ ∀ x ∈ l, p x
  | [] => (iff_true_intro <| forall_mem_nil _).symm
                 /-
                   α : Type u
                   p : α → Prop
                   x : α
                   l : List α
                   ⊢ Iff (List.Forall p (List.cons x l)) (∀ (x_1 : α), Membership.mem (List.cons  …
                 -/
  | x :: l => by rw [forall_mem_cons, forall_cons, forall_iff_forall_mem]
                 /-
                   🎉 no goals
                 -/


theorem Forall.imp (h : ∀ x, p x → q x) : ∀ {l : List α}, Forall p l → Forall q l
  | [] => id
  | x :: l => by
    /-
      α : Type u
      p q : α → Prop
      h : ∀ (x : α), p x → q x
      x : α
      l : List α
      ⊢ List.Forall p (List.cons x l) → List.Forall q (List.cons x l)
    -/
    simp only [forall_cons, and_imp]
    /-
      α : Type u
      p q : α → Prop
      h : ∀ (x : α), p x → q x
      x : α
      l : List α
      ⊢ p x → List.Forall p l → And (q x) (List.Forall q l)
    -/
    rw [← and_imp]
    /-
      α : Type u
      p q : α → Prop
      h : ∀ (x : α), p x → q x
      x : α
      l : List α
      ⊢ And (p x) (List.Forall p l) → And (q x) (List.Forall q l)
    -/
    exact And.imp (h x) (Forall.imp h)
    /-
      🎉 no goals
    -/


@[simp]
theorem forall_map_iff {p : β → Prop} (f : α → β) : Forall p (l.map f) ↔ Forall (p ∘ f) l := by
  /-
    α : Type u
    β : Type v
    l : List α
    p : β → Prop
    f : α → β
    ⊢ Iff (List.Forall p (List.map f l)) (List.Forall (Function.comp p f) l)
  -/
                  /-
                    🎉 no goals
                  -/
  induction l <;> simp [*]
                  /-
                    🎉 no goals
                  -/


instance (p : α → Prop) [DecidablePred p] : DecidablePred (Forall p) := fun _ =>
  decidable_of_iff' _ forall_iff_forall_mem


theorem get_attach (L : List α) (i) :
                                                                       /-
                                                                         α : Type u
                                                                         L : List α
                                                                         i : Fin L.attach.length
                                                                         ⊢ Eq (↑(L.attach.get i)) (L.get ⟨↑i, ⋯⟩)
                                                                       -/
    (L.attach.get i).1 = L.get ⟨i, length_attach (L := L) ▸ i.2⟩ := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp 1100]
theorem mem_map_swap (x : α) (y : β) (xs : List (α × β)) :
    (y, x) ∈ map Prod.swap xs ↔ (x, y) ∈ xs := by
  /-
    α : Type u
    β : Type v
    x : α
    y : β
    xs : List (Prod α β)
    ⊢ Iff (Membership.mem (List.map Prod.swap xs) { fst := y, snd := x }) (Members …
  -/
  induction' xs with x xs xs_ih
    /-
      case nil
      α : Type u
      β : Type v
      x : α
      y : β
      ⊢ Iff (Membership.mem (List.map Prod.swap List.nil) { fst := y, snd := x }) (M …
    -/
  · simp only [not_mem_nil, map_nil]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type v
      x✝ : α
      y : β
      x : Prod α β
      xs : List (Prod α β)
      xs_ih : Iff (Membership.mem (List.map Prod.swap xs) { fst := y, snd := x✝ }) ( …
      ⊢ Iff (Membership.mem (List.map Prod.swap (List.cons x xs)) { fst := y, snd := …
    -/
  · cases' x with a b
    /-
      case cons.mk
      α : Type u
      β : Type v
      x : α
      y : β
      xs : List (Prod α β)
      xs_ih : Iff (Membership.mem (List.map Prod.swap xs) { fst := y, snd := x }) (M …
      a : α
      b : β
      ⊢ Iff (Membership.mem (List.map Prod.swap (List.cons { fst := a, snd := b } xs …
    -/
    simp only [mem_cons, Prod.mk.inj_iff, map, Prod.swap_prod_mk, Prod.exists, xs_ih, and_comm]
    /-
      🎉 no goals
    -/


theorem dropSlice_eq (xs : List α) (n m : ℕ) : dropSlice n m xs = xs.take n ++ xs.drop (n + m) := by
  /-
    α : Type u
    xs : List α
    n m : Nat
    ⊢ Eq (List.dropSlice n m xs) (HAppend.hAppend (List.take n xs) (List.drop (HAd …
  -/
  induction n generalizing xs
    /-
      case zero
      α : Type u
      m : Nat
      xs : List α
      ⊢ Eq (List.dropSlice 0 m xs) (HAppend.hAppend (List.take 0 xs) (List.drop (HAd …
    -/
                 /-
                   🎉 no goals
                 -/
  · cases xs <;> simp [dropSlice]
                 /-
                   🎉 no goals
                 -/
    /-
      case succ
      α : Type u
      m n✝ : Nat
      a✝ : ∀ (xs : List α), Eq (List.dropSlice n✝ m xs) (HAppend.hAppend (List.take  …
      xs : List α
      ⊢ Eq (List.dropSlice (HAdd.hAdd n✝ 1) m xs) (HAppend.hAppend (List.take (HAdd. …
    -/
                 /-
                   🎉 no goals
                 -/
  · cases xs <;> simp [dropSlice, *, Nat.succ_add]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem length_dropSlice (i j : ℕ) (xs : List α) :
    (List.dropSlice i j xs).length = xs.length - min j (xs.length - i) := by
  induction xs generalizing i j with
  | nil => simp
  | cons x xs xs_ih =>
    cases i <;> simp only [List.dropSlice]
    · cases j with
      | zero => simp
      | succ n => simp_all [xs_ih]; omega
    · simp [xs_ih]; omega


theorem length_dropSlice_lt (i j : ℕ) (hj : 0 < j) (xs : List α) (hi : i < xs.length) :
    (List.dropSlice i j xs).length < xs.length := by
  /-
    α : Type u
    i j : Nat
    hj : LT.lt 0 j
    xs : List α
    hi : LT.lt i xs.length
    ⊢ LT.lt (List.dropSlice i j xs).length xs.length
  -/
  simp; omega
        /-
          🎉 no goals
        -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-07-25")]
theorem sizeOf_dropSlice_lt [SizeOf α] (i j : ℕ) (hj : 0 < j) (xs : List α) (hi : i < xs.length) :
    SizeOf.sizeOf (List.dropSlice i j xs) < SizeOf.sizeOf xs := by
  induction xs generalizing i j hj with
  | nil => cases hi
  | cons x xs xs_ih =>
    cases i <;> simp only [List.dropSlice]
    · cases j with
      | zero => contradiction
      | succ n =>
        dsimp only [drop]; apply lt_of_le_of_lt (drop_sizeOf_le xs n)
        simp only [cons.sizeOf_spec]; omega
    · simp only [cons.sizeOf_spec, Nat.add_lt_add_iff_left]
      apply xs_ih _ j hj
      apply lt_of_succ_lt_succ hi


/-- The images of disjoint lists under a partially defined map are disjoint -/
theorem disjoint_pmap {p : α → Prop} {f : ∀ a : α, p a → β} {s t : List α}
    (hs : ∀ a ∈ s, p a) (ht : ∀ a ∈ t, p a)
    (hf : ∀ (a a' : α) (ha : p a) (ha' : p a'), f a ha = f a' ha' → a = a')
    (h : Disjoint s t) :
    Disjoint (s.pmap f hs) (t.pmap f ht) := by
  /-
    α : Type u
    β : Type v
    p : α → Prop
    f : (a : α) → p a → β
    s t : List α
    hs : ∀ (a : α), Membership.mem s a → p a
    ht : ∀ (a : α), Membership.mem t a → p a
    hf : ∀ (a a' : α) (ha : p a) (ha' : p a'), Eq (f a ha) (f a' ha') → Eq a a'
    h : s.Disjoint t
    ⊢ (List.pmap f s hs).Disjoint (List.pmap f t ht)
  -/
  simp only [Disjoint, mem_pmap]
  /-
    α : Type u
    β : Type v
    p : α → Prop
    f : (a : α) → p a → β
    s t : List α
    hs : ∀ (a : α), Membership.mem s a → p a
    ht : ∀ (a : α), Membership.mem t a → p a
    hf : ∀ (a a' : α) (ha : p a) (ha' : p a'), Eq (f a ha) (f a' ha') → Eq a a'
    h : s.Disjoint t
    ⊢ ∀ ⦃a : β⦄, (Exists fun a_1 => Exists fun h => Eq (f a_1 ⋯) a) → (Exists fun  …
  -/
  rintro b ⟨a, ha, rfl⟩ ⟨a', ha', ha''⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    p : α → Prop
    f : (a : α) → p a → β
    s t : List α
    hs : ∀ (a : α), Membership.mem s a → p a
    ht : ∀ (a : α), Membership.mem t a → p a
    hf : ∀ (a a' : α) (ha : p a) (ha' : p a'), Eq (f a ha) (f a' ha') → Eq a a'
    h : s.Disjoint t
    a : α
    ha : Membership.mem s a
    a' : α
    ha' : Membership.mem t a'
    ha'' : Eq (f a' ⋯) (f a ⋯)
    ⊢ False
  -/
  apply h ha
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    p : α → Prop
    f : (a : α) → p a → β
    s t : List α
    hs : ∀ (a : α), Membership.mem s a → p a
    ht : ∀ (a : α), Membership.mem t a → p a
    hf : ∀ (a a' : α) (ha : p a) (ha' : p a'), Eq (f a ha) (f a' ha') → Eq a a'
    h : s.Disjoint t
    a : α
    ha : Membership.mem s a
    a' : α
    ha' : Membership.mem t a'
    ha'' : Eq (f a' ⋯) (f a ⋯)
    ⊢ Membership.mem t a
  -/
  rwa [hf a a' (hs a ha) (ht a' ha') ha''.symm]
  /-
    🎉 no goals
  -/


/-- The images of disjoint lists under an injective map are disjoint -/
theorem disjoint_map {f : α → β} {s t : List α} (hf : Function.Injective f)
    (h : Disjoint s t) : Disjoint (s.map f) (t.map f) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s t : List α
    hf : Function.Injective f
    h : s.Disjoint t
    ⊢ (List.map f s).Disjoint (List.map f t)
  -/
  rw [← pmap_eq_map _ _ _ (fun _ _ ↦ trivial), ← pmap_eq_map _ _ _ (fun _ _ ↦ trivial)]
  /-
    α : Type u
    β : Type v
    f : α → β
    s t : List α
    hf : Function.Injective f
    h : s.Disjoint t
    ⊢ (List.pmap (fun a x => f a) s ⋯).Disjoint (List.pmap (fun a x => f a) t ⋯)
  -/
  exact disjoint_pmap _ _ (fun _ _ _ _ h' ↦ hf h') h
  /-
    🎉 no goals
  -/


alias Disjoint.map := disjoint_map


theorem Disjoint.of_map {f : α → β} {s t : List α} (h : Disjoint (s.map f) (t.map f)) :
    Disjoint s t := fun _a has hat ↦
  h (mem_map_of_mem f has) (mem_map_of_mem f hat)


theorem Disjoint.map_iff {f : α → β} {s t : List α} (hf : Function.Injective f) :
    Disjoint (s.map f) (t.map f) ↔ Disjoint s t :=
  ⟨fun h ↦ h.of_map, fun h ↦ h.map hf⟩


theorem Perm.disjoint_left {l₁ l₂ l : List α} (p : List.Perm l₁ l₂) :
    Disjoint l₁ l ↔ Disjoint l₂ l := by
  /-
    α : Type u
    l₁ l₂ l : List α
    p : l₁.Perm l₂
    ⊢ Iff (l₁.Disjoint l) (l₂.Disjoint l)
  -/
  simp_rw [List.disjoint_left, p.mem_iff]
  /-
    🎉 no goals
  -/


theorem Perm.disjoint_right {l₁ l₂ l : List α} (p : List.Perm l₁ l₂) :
    Disjoint l l₁ ↔ Disjoint l l₂ := by
  /-
    α : Type u
    l₁ l₂ l : List α
    p : l₁.Perm l₂
    ⊢ Iff (l.Disjoint l₁) (l.Disjoint l₂)
  -/
  simp_rw [List.disjoint_right, p.mem_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_reverse_left {l₁ l₂ : List α} : Disjoint l₁.reverse l₂ ↔ Disjoint l₁ l₂ :=
  reverse_perm _ |>.disjoint_left


@[simp]
theorem disjoint_reverse_right {l₁ l₂ : List α} : Disjoint l₁ l₂.reverse ↔ Disjoint l₁ l₂ :=
  reverse_perm _ |>.disjoint_right


lemma lookup_graph (f : α → β) {a : α} {as : List α} (h : a ∈ as) :
    lookup a (as.map fun x => (x, f x)) = some (f a) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : BEq α
    inst✝ : LawfulBEq α
    f : α → β
    a : α
    as : List α
    h : Membership.mem as a
    ⊢ Eq (List.lookup a (List.map (fun x => { fst := x, snd := f x }) as)) (Option …
  -/
  induction' as with a' as ih
    /-
      case nil
      α : Type u
      β : Type v
      inst✝¹ : BEq α
      inst✝ : LawfulBEq α
      f : α → β
      a : α
      h : Membership.mem List.nil a
      ⊢ Eq (List.lookup a (List.map (fun x => { fst := x, snd := f x }) List.nil)) ( …
    -/
  · exact (List.not_mem_nil _ h).elim
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type v
      inst✝¹ : BEq α
      inst✝ : LawfulBEq α
      f : α → β
      a a' : α
      as : List α
      ih : Membership.mem as a → Eq (List.lookup a (List.map (fun x => { fst := x, s …
      h : Membership.mem (List.cons a' as) a
      ⊢ Eq (List.lookup a (List.map (fun x => { fst := x, snd := f x }) (List.cons a …
    -/
  · by_cases ha : a = a'
      /-
        case pos
        α : Type u
        β : Type v
        inst✝¹ : BEq α
        inst✝ : LawfulBEq α
        f : α → β
        a a' : α
        as : List α
        ih : Membership.mem as a → Eq (List.lookup a (List.map (fun x => { fst := x, s …
        h : Membership.mem (List.cons a' as) a
        ha : Eq a a'
        ⊢ Eq (List.lookup a (List.map (fun x => { fst := x, snd := f x }) (List.cons a …
      -/
    · simp [ha, lookup_cons]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : Type v
        inst✝¹ : BEq α
        inst✝ : LawfulBEq α
        f : α → β
        a a' : α
        as : List α
        ih : Membership.mem as a → Eq (List.lookup a (List.map (fun x => { fst := x, s …
        h : Membership.mem (List.cons a' as) a
        ha : Not (Eq a a')
        ⊢ Eq (List.lookup a (List.map (fun x => { fst := x, snd := f x }) (List.cons a …
      -/
    · simpa [lookup_cons, beq_false_of_ne ha] using ih (List.mem_of_ne_of_mem ha h)
      /-
        🎉 no goals
      -/


