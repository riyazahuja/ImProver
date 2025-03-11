/-- `sort s` constructs a sorted list from the multiset `s`.
  (Uses merge sort algorithm.) -/
def sort (s : Multiset α) : List α :=
  Quot.liftOn s (mergeSort · (r · ·)) fun _ _ h =>
    eq_of_perm_of_sorted ((mergeSort_perm _ _).trans <| h.trans (mergeSort_perm _ _).symm)
      (sorted_mergeSort IsTrans.trans
                       /-
                         α : Type u_1
                         β : Type u_2
                         r : α → α → Prop
                         inst✝⁷ : DecidableRel r
                         inst✝⁶ : IsTrans α r
                         inst✝⁵ : IsAntisymm α r
                         inst✝⁴ : IsTotal α r
                         r' : β → β → Prop
                         inst✝³ : DecidableRel r'
                         inst✝² : IsTrans β r'
                         inst✝¹ : IsAntisymm β r'
                         inst✝ : IsTotal β r'
                         s : Multiset α
                         x✝¹ x✝ : List α
                         h : (List.isSetoid α) x✝¹ x✝
                         a b : α
                         ⊢ Eq ((Decidable.decide (r a b)).or (Decidable.decide (r b a))) Bool.true
                       -/
        (fun a b => by simpa using IsTotal.total a b) _)
                       /-
                         🎉 no goals
                       -/
      (sorted_mergeSort IsTrans.trans
                       /-
                         α : Type u_1
                         β : Type u_2
                         r : α → α → Prop
                         inst✝⁷ : DecidableRel r
                         inst✝⁶ : IsTrans α r
                         inst✝⁵ : IsAntisymm α r
                         inst✝⁴ : IsTotal α r
                         r' : β → β → Prop
                         inst✝³ : DecidableRel r'
                         inst✝² : IsTrans β r'
                         inst✝¹ : IsAntisymm β r'
                         inst✝ : IsTotal β r'
                         s : Multiset α
                         x✝¹ x✝ : List α
                         h : (List.isSetoid α) x✝¹ x✝
                         a b : α
                         ⊢ Eq ((Decidable.decide (r a b)).or (Decidable.decide (r b a))) Bool.true
                       -/
        (fun a b => by simpa using IsTotal.total a b) _)
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem coe_sort (l : List α) : sort r l = mergeSort l (r · ·) :=
  rfl


@[simp]
theorem sort_sorted (s : Multiset α) : Sorted r (sort r s) :=
  Quot.inductionOn s fun l => by
    simpa using sorted_mergeSort (le := (r · ·)) IsTrans.trans
      (fun a b => by simpa using IsTotal.total a b) l


@[simp]
theorem sort_eq (s : Multiset α) : ↑(sort r s) = s :=
  Quot.inductionOn s fun _ => Quot.sound <| mergeSort_perm _ _


@[simp]
                                                                       /-
                                                                         α : Type u_1
                                                                         r : α → α → Prop
                                                                         inst✝³ : DecidableRel r
                                                                         inst✝² : IsTrans α r
                                                                         inst✝¹ : IsAntisymm α r
                                                                         inst✝ : IsTotal α r
                                                                         s : Multiset α
                                                                         a : α
                                                                         ⊢ Iff (Membership.mem (Multiset.sort r s) a) (Membership.mem s a)
                                                                       -/
theorem mem_sort {s : Multiset α} {a : α} : a ∈ sort r s ↔ a ∈ s := by rw [← mem_coe, sort_eq]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem length_sort {s : Multiset α} : (sort r s).length = card s :=
  Quot.inductionOn s <| length_mergeSort


@[simp]
theorem sort_zero : sort r 0 = [] :=
  List.mergeSort_nil


@[simp]
theorem sort_singleton (a : α) : sort r {a} = [a] :=
  List.mergeSort_singleton a


theorem map_sort (f : α → β) (s : Multiset α)
    (hs : ∀ a ∈ s, ∀ b ∈ s, r a b ↔ r' (f a) (f b)) :
    (s.sort r).map f = (s.map f).sort r' := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    inst✝⁷ : DecidableRel r
    inst✝⁶ : IsTrans α r
    inst✝⁵ : IsAntisymm α r
    inst✝⁴ : IsTotal α r
    r' : β → β → Prop
    inst✝³ : DecidableRel r'
    inst✝² : IsTrans β r'
    inst✝¹ : IsAntisymm β r'
    inst✝ : IsTotal β r'
    f : α → β
    s : Multiset α
    hs : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Iff (r a  …
    ⊢ Eq (List.map f (Multiset.sort r s)) (Multiset.sort r' (Multiset.map f s))
  -/
  revert s
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    inst✝⁷ : DecidableRel r
    inst✝⁶ : IsTrans α r
    inst✝⁵ : IsAntisymm α r
    inst✝⁴ : IsTotal α r
    r' : β → β → Prop
    inst✝³ : DecidableRel r'
    inst✝² : IsTrans β r'
    inst✝¹ : IsAntisymm β r'
    inst✝ : IsTotal β r'
    f : α → β
    ⊢ ∀ (s : Multiset α), (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.m …
  -/
  exact Quot.ind fun l h => map_mergeSort (l := l) (by simpa using h)
  /-
    🎉 no goals
  -/


theorem sort_cons (a : α) (s : Multiset α) :
    (∀ b ∈ s, r a b) → sort r (a ::ₘ s) = a :: sort r s := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝³ : DecidableRel r
    inst✝² : IsTrans α r
    inst✝¹ : IsAntisymm α r
    inst✝ : IsTotal α r
    a : α
    s : Multiset α
    ⊢ (∀ (b : α), Membership.mem s b → r a b) → Eq (Multiset.sort r (Multiset.cons …
  -/
  refine Quot.inductionOn s fun l => ?_
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝³ : DecidableRel r
    inst✝² : IsTrans α r
    inst✝¹ : IsAntisymm α r
    inst✝ : IsTotal α r
    a : α
    s : Multiset α
    l : List α
    ⊢ (∀ (b : α), Membership.mem (Quot.mk (⇑(List.isSetoid α)) l) b → r a b) → Eq  …
  -/
  simpa [mergeSort_eq_insertionSort] using insertionSort_cons r (a := a) (l := l)
  /-
    🎉 no goals
  -/


@[simp]
theorem sort_range (n : ℕ) : sort (· ≤ ·) (range n) = List.range n :=
  List.mergeSort_eq_self (sorted_le_range n)


unsafe instance [Repr α] : Repr (Multiset α) where
  reprPrec s _ :=
    if Multiset.card s = 0 then
      "0"
    else
      Std.Format.bracket "{" (Std.Format.joinSep (s.unquot.map repr) ("," ++ Std.Format.line)) "}"


