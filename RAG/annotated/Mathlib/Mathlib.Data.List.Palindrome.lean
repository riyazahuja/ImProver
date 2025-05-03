/-- `Palindrome l` asserts that `l` is a palindrome. This is defined inductively:

* The empty list is a palindrome;
* A list with one element is a palindrome;
* Adding the same element to both ends of a palindrome results in a bigger palindrome.
-/
inductive Palindrome : List α → Prop
  | nil : Palindrome []
  | singleton : ∀ x, Palindrome [x]
  | cons_concat : ∀ (x) {l}, Palindrome l → Palindrome (x :: (l ++ [x]))


theorem reverse_eq {l : List α} (p : Palindrome l) : reverse l = l := by
  /-
    α : Type u_1
    l : List α
    p : l.Palindrome
    ⊢ Eq l.reverse l
  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> try (exact rfl)
  /-
    case cons_concat
    α : Type u_1
    l : List α
    x✝ : α
    l✝ : List α
    a✝ : l✝.Palindrome
    a_ih✝ : Eq l✝.reverse l✝
    ⊢ Eq (List.cons x✝ (HAppend.hAppend l✝ (List.cons x✝ List.nil))).reverse (List …
  -/
  simpa
  /-
    🎉 no goals
  -/


theorem of_reverse_eq {l : List α} : reverse l = l → Palindrome l := by
  /-
    α : Type u_1
    l : List α
    ⊢ Eq l.reverse l → l.Palindrome
  -/
  refine bidirectionalRecOn l (fun _ => Palindrome.nil) (fun a _ => Palindrome.singleton a) ?_
  /-
    α : Type u_1
    l : List α
    ⊢ ∀ (a : α) (l : List α) (b : α), (Eq l.reverse l → l.Palindrome) → Eq (List.c …
  -/
  intro x l y hp hr
  /-
    α : Type u_1
    l✝ : List α
    x : α
    l : List α
    y : α
    hp : Eq l.reverse l → l.Palindrome
    hr : Eq (List.cons x (HAppend.hAppend l (List.cons y List.nil))).reverse (List …
    ⊢ (List.cons x (HAppend.hAppend l (List.cons y List.nil))).Palindrome
  -/
  rw [reverse_cons, reverse_append] at hr
  /-
    α : Type u_1
    l✝ : List α
    x : α
    l : List α
    y : α
    hp : Eq l.reverse l → l.Palindrome
    hr : Eq (HAppend.hAppend (HAppend.hAppend (List.cons y List.nil).reverse l.rev …
    ⊢ (List.cons x (HAppend.hAppend l (List.cons y List.nil))).Palindrome
  -/
  rw [head_eq_of_cons_eq hr]
  /-
    α : Type u_1
    l✝ : List α
    x : α
    l : List α
    y : α
    hp : Eq l.reverse l → l.Palindrome
    hr : Eq (HAppend.hAppend (HAppend.hAppend (List.cons y List.nil).reverse l.rev …
    ⊢ (List.cons x (HAppend.hAppend l (List.cons x List.nil))).Palindrome
  -/
  have : Palindrome l := hp (append_inj_left' (tail_eq_of_cons_eq hr) rfl)
  /-
    α : Type u_1
    l✝ : List α
    x : α
    l : List α
    y : α
    hp : Eq l.reverse l → l.Palindrome
    hr : Eq (HAppend.hAppend (HAppend.hAppend (List.cons y List.nil).reverse l.rev …
    this : l.Palindrome
    ⊢ (List.cons x (HAppend.hAppend l (List.cons x List.nil))).Palindrome
  -/
  exact Palindrome.cons_concat x this
  /-
    🎉 no goals
  -/


theorem iff_reverse_eq {l : List α} : Palindrome l ↔ reverse l = l :=
  Iff.intro reverse_eq of_reverse_eq


theorem append_reverse (l : List α) : Palindrome (l ++ reverse l) := by
  /-
    α : Type u_1
    l : List α
    ⊢ (HAppend.hAppend l l.reverse).Palindrome
  -/
  apply of_reverse_eq
  /-
    case a
    α : Type u_1
    l : List α
    ⊢ Eq (HAppend.hAppend l l.reverse).reverse (HAppend.hAppend l l.reverse)
  -/
  rw [reverse_append, reverse_reverse]
  /-
    🎉 no goals
  -/


protected theorem map (f : α → β) (p : Palindrome l) : Palindrome (map f l) :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        l : List α
                        f : α → β
                        p : l.Palindrome
                        ⊢ Eq (List.map f l).reverse (List.map f l)
                      -/
  of_reverse_eq <| by rw [← map_reverse, p.reverse_eq]
                      /-
                        🎉 no goals
                      -/


instance [DecidableEq α] (l : List α) : Decidable (Palindrome l) :=
  decidable_of_iff' _ iff_reverse_eq


