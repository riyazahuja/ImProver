/-- `sort s` constructs a sorted list from the unordered set `s`.
  (Uses merge sort algorithm.) -/
def sort (s : Finset α) : List α :=
  Multiset.sort r s.1


@[simp]
theorem sort_val (s : Finset α) : Multiset.sort r s.val = sort r s :=
  rfl


@[simp]
theorem sort_sorted (s : Finset α) : List.Sorted r (sort r s) :=
  Multiset.sort_sorted _ _


@[simp]
theorem sort_eq (s : Finset α) : ↑(sort r s) = s.1 :=
  Multiset.sort_eq _ _


@[simp]
theorem sort_nodup (s : Finset α) : (sort r s).Nodup :=
      /-
        α : Type u_1
        r : α → α → Prop
        inst✝³ : DecidableRel r
        inst✝² : IsTrans α r
        inst✝¹ : IsAntisymm α r
        inst✝ : IsTotal α r
        s : Finset α
        ⊢ (↑(Finset.sort r s)).Nodup
      -/
  (by rw [sort_eq]; exact s.2 : @Multiset.Nodup α (sort r s))
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem sort_toFinset [DecidableEq α] (s : Finset α) : (sort r s).toFinset = s :=
  List.toFinset_eq (sort_nodup r s) ▸ eq_of_veq (sort_eq r s)


@[simp]
theorem mem_sort {s : Finset α} {a : α} : a ∈ sort r s ↔ a ∈ s :=
  Multiset.mem_sort _


@[simp]
theorem length_sort {s : Finset α} : (sort r s).length = s.card :=
  Multiset.length_sort _


@[simp]
theorem sort_empty : sort r ∅ = [] :=
  Multiset.sort_zero r


@[simp]
theorem sort_singleton (a : α) : sort r {a} = [a] :=
  Multiset.sort_singleton r a


theorem sort_cons {a : α} {s : Finset α} (h₁ : ∀ b ∈ s, r a b) (h₂ : a ∉ s) :
    sort r (cons a s h₂) = a :: sort r s := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝³ : DecidableRel r
    inst✝² : IsTrans α r
    inst✝¹ : IsAntisymm α r
    inst✝ : IsTotal α r
    a : α
    s : Finset α
    h₁ : ∀ (b : α), Membership.mem s b → r a b
    h₂ : Not (Membership.mem s a)
    ⊢ Eq (Finset.sort r (Finset.cons a s h₂)) (List.cons a (Finset.sort r s))
  -/
  rw [sort, cons_val, Multiset.sort_cons r a _ h₁, sort_val]
  /-
    🎉 no goals
  -/


theorem sort_insert [DecidableEq α] {a : α} {s : Finset α} (h₁ : ∀ b ∈ s, r a b) (h₂ : a ∉ s) :
    sort r (insert a s) = a :: sort r s := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝⁴ : DecidableRel r
    inst✝³ : IsTrans α r
    inst✝² : IsAntisymm α r
    inst✝¹ : IsTotal α r
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    h₁ : ∀ (b : α), Membership.mem s b → r a b
    h₂ : Not (Membership.mem s a)
    ⊢ Eq (Finset.sort r (Insert.insert a s)) (List.cons a (Finset.sort r s))
  -/
  rw [← cons_eq_insert _ _ h₂, sort_cons r h₁]
  /-
    🎉 no goals
  -/


@[simp]
theorem sort_range (n : ℕ) : sort (· ≤ ·) (range n) = List.range n :=
  Multiset.sort_range n


open scoped List in
theorem sort_perm_toList (s : Finset α) : sort r s ~ s.toList := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝³ : DecidableRel r
    inst✝² : IsTrans α r
    inst✝¹ : IsAntisymm α r
    inst✝ : IsTotal α r
    s : Finset α
    ⊢ (Finset.sort r s).Perm s.toList
  -/
  rw [← Multiset.coe_eq_coe]
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝³ : DecidableRel r
    inst✝² : IsTrans α r
    inst✝¹ : IsAntisymm α r
    inst✝ : IsTotal α r
    s : Finset α
    ⊢ Eq ↑(Finset.sort r s) ↑s.toList
  -/
  simp only [coe_toList, sort_eq]
  /-
    🎉 no goals
  -/


theorem _root_.List.toFinset_sort [DecidableEq α] {l : List α} (hl : l.Nodup) :
    sort r l.toFinset = l ↔ l.Sorted r := by
  refine ⟨?_, List.eq_of_perm_of_sorted ((sort_perm_toList r _).trans (List.toFinset_toList hl))
    (sort_sorted r _)⟩
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝⁴ : DecidableRel r
    inst✝³ : IsTrans α r
    inst✝² : IsAntisymm α r
    inst✝¹ : IsTotal α r
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    ⊢ Eq (Finset.sort r l.toFinset) l → List.Sorted r l
  -/
  intro h
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝⁴ : DecidableRel r
    inst✝³ : IsTrans α r
    inst✝² : IsAntisymm α r
    inst✝¹ : IsTotal α r
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    h : Eq (Finset.sort r l.toFinset) l
    ⊢ List.Sorted r l
  -/
  rw [← h]
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝⁴ : DecidableRel r
    inst✝³ : IsTrans α r
    inst✝² : IsAntisymm α r
    inst✝¹ : IsTotal α r
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    h : Eq (Finset.sort r l.toFinset) l
    ⊢ List.Sorted r (Finset.sort r l.toFinset)
  -/
  exact sort_sorted r _
  /-
    🎉 no goals
  -/


theorem sort_sorted_lt (s : Finset α) : List.Sorted (· < ·) (sort (· ≤ ·) s) :=
  (sort_sorted _ _).lt_of_le (sort_nodup _ _)


theorem sort_sorted_gt (s : Finset α) : List.Sorted (· > ·) (sort (· ≥ ·) s) :=
  (sort_sorted _ _).gt_of_ge (sort_nodup _ _)


theorem sorted_zero_eq_min'_aux (s : Finset α) (h : 0 < (s.sort (· ≤ ·)).length) (H : s.Nonempty) :
    (s.sort (· ≤ ·)).get ⟨0, h⟩ = s.min' H := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Finset α
    h : LT.lt 0 (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
    H : s.Nonempty
    ⊢ Eq ((Finset.sort (fun x1 x2 => LE.le x1 x2) s).get ⟨0, h⟩) (s.min' H)
  -/
  let l := s.sort (· ≤ ·)
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Finset α
    h : LT.lt 0 (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
    H : s.Nonempty
    l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
    ⊢ Eq ((Finset.sort (fun x1 x2 => LE.le x1 x2) s).get ⟨0, h⟩) (s.min' H)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt 0 (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      ⊢ LE.le ((Finset.sort (fun x1 x2 => LE.le x1 x2) s).get ⟨0, h⟩) (s.min' H)
    -/
  · have : s.min' H ∈ l := (Finset.mem_sort (α := α) (· ≤ ·)).mpr (s.min'_mem H)
    /-
      case a
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt 0 (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      this : Membership.mem l (s.min' H)
      ⊢ LE.le ((Finset.sort (fun x1 x2 => LE.le x1 x2) s).get ⟨0, h⟩) (s.min' H)
    -/
    obtain ⟨i, hi⟩ : ∃ i, l.get i = s.min' H := List.mem_iff_get.1 this
    /-
      case a.intro
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt 0 (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      this : Membership.mem l (s.min' H)
      i : Fin l.length
      hi : Eq (l.get i) (s.min' H)
      ⊢ LE.le ((Finset.sort (fun x1 x2 => LE.le x1 x2) s).get ⟨0, h⟩) (s.min' H)
    -/
    rw [← hi]
    /-
      case a.intro
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt 0 (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      this : Membership.mem l (s.min' H)
      i : Fin l.length
      hi : Eq (l.get i) (s.min' H)
      ⊢ LE.le ((Finset.sort (fun x1 x2 => LE.le x1 x2) s).get ⟨0, h⟩) (l.get i)
    -/
    exact (s.sort_sorted (· ≤ ·)).rel_get_of_le (Nat.zero_le i)
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt 0 (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      ⊢ LE.le (s.min' H) ((Finset.sort (fun x1 x2 => LE.le x1 x2) s).get ⟨0, h⟩)
    -/
  · have : l.get ⟨0, h⟩ ∈ s := (Finset.mem_sort (α := α) (· ≤ ·)).1 (List.get_mem l _)
    /-
      case a
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt 0 (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      this : Membership.mem s (l.get ⟨0, h⟩)
      ⊢ LE.le (s.min' H) ((Finset.sort (fun x1 x2 => LE.le x1 x2) s).get ⟨0, h⟩)
    -/
    exact s.min'_le _ this
    /-
      🎉 no goals
    -/


theorem sorted_zero_eq_min' {s : Finset α} {h : 0 < (s.sort (· ≤ ·)).length} :
                                                   /-
                                                     α : Type u_1
                                                     β : Type u_2
                                                     inst✝ : LinearOrder α
                                                     s : Finset α
                                                     h : LT.lt 0 (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
                                                     ⊢ LT.lt 0 s.card
                                                   -/
    (s.sort (· ≤ ·))[0] = s.min' (card_pos.1 <| by rwa [length_sort] at h) :=
                                                   /-
                                                     🎉 no goals
                                                   -/
  sorted_zero_eq_min'_aux _ _ _


theorem min'_eq_sorted_zero {s : Finset α} {h : s.Nonempty} :
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         inst✝ : LinearOrder α
                                         s : Finset α
                                         h : s.Nonempty
                                         ⊢ LT.lt 0 (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
                                       -/
    s.min' h = (s.sort (· ≤ ·))[0]'(by rw [length_sort]; exact card_pos.2 h) :=
                                                         /-
                                                           🎉 no goals
                                                         -/
  (sorted_zero_eq_min'_aux _ _ _).symm


theorem sorted_last_eq_max'_aux (s : Finset α)
    (h : (s.sort (· ≤ ·)).length - 1 < (s.sort (· ≤ ·)).length) (H : s.Nonempty) :
    (s.sort (· ≤ ·))[(s.sort (· ≤ ·)).length - 1] = s.max' H := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Finset α
    h : LT.lt (HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length 1) (Fin …
    H : s.Nonempty
    ⊢ Eq (GetElem.getElem (Finset.sort (fun x1 x2 => LE.le x1 x2) s) (HSub.hSub (F …
  -/
  let l := s.sort (· ≤ ·)
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Finset α
    h : LT.lt (HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length 1) (Fin …
    H : s.Nonempty
    l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
    ⊢ Eq (GetElem.getElem (Finset.sort (fun x1 x2 => LE.le x1 x2) s) (HSub.hSub (F …
  -/
  apply le_antisymm
  · have : l.get ⟨(s.sort (· ≤ ·)).length - 1, h⟩ ∈ s :=
      (Finset.mem_sort (α := α) (· ≤ ·)).1 (List.get_mem l _)
    /-
      case a
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt (HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length 1) (Fin …
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      this : Membership.mem s (l.get ⟨HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1  …
      ⊢ LE.le (GetElem.getElem (Finset.sort (fun x1 x2 => LE.le x1 x2) s) (HSub.hSub …
    -/
    exact s.le_max' _ this
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt (HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length 1) (Fin …
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      ⊢ LE.le (s.max' H) (GetElem.getElem (Finset.sort (fun x1 x2 => LE.le x1 x2) s) …
    -/
  · have : s.max' H ∈ l := (Finset.mem_sort (α := α) (· ≤ ·)).mpr (s.max'_mem H)
    /-
      case a
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt (HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length 1) (Fin …
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      this : Membership.mem l (s.max' H)
      ⊢ LE.le (s.max' H) (GetElem.getElem (Finset.sort (fun x1 x2 => LE.le x1 x2) s) …
    -/
    obtain ⟨i, hi⟩ : ∃ i, l.get i = s.max' H := List.mem_iff_get.1 this
    /-
      case a.intro
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt (HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length 1) (Fin …
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      this : Membership.mem l (s.max' H)
      i : Fin l.length
      hi : Eq (l.get i) (s.max' H)
      ⊢ LE.le (s.max' H) (GetElem.getElem (Finset.sort (fun x1 x2 => LE.le x1 x2) s) …
    -/
    rw [← hi]
    /-
      case a.intro
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      h : LT.lt (HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length 1) (Fin …
      H : s.Nonempty
      l : List α := Finset.sort (fun x1 x2 => LE.le x1 x2) s
      this : Membership.mem l (s.max' H)
      i : Fin l.length
      hi : Eq (l.get i) (s.max' H)
      ⊢ LE.le (l.get i) (GetElem.getElem (Finset.sort (fun x1 x2 => LE.le x1 x2) s)  …
    -/
    exact (s.sort_sorted (· ≤ ·)).rel_get_of_le (Nat.le_sub_one_of_lt i.prop)
    /-
      🎉 no goals
    -/


theorem sorted_last_eq_max' {s : Finset α}
    {h : (s.sort (· ≤ ·)).length - 1 < (s.sort (· ≤ ·)).length} :
    (s.sort (· ≤ ·))[(s.sort (· ≤ ·)).length - 1] =
                 /-
                   α : Type u_1
                   β : Type u_2
                   inst✝ : LinearOrder α
                   s : Finset α
                   h : LT.lt (HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length 1) (Fin …
                   ⊢ s.Nonempty
                 -/
      s.max' (by rw [length_sort] at h; exact card_pos.1 (lt_of_le_of_lt bot_le h)) :=
                                        /-
                                          🎉 no goals
                                        -/
  sorted_last_eq_max'_aux _ h _


theorem max'_eq_sorted_last {s : Finset α} {h : s.Nonempty} :
    s.max' h =
      (s.sort (· ≤ ·))[(s.sort (· ≤ ·)).length - 1]'
            /-
              α : Type u_1
              β : Type u_2
              inst✝ : LinearOrder α
              s : Finset α
              h : s.Nonempty
              ⊢ LT.lt (HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length 1) (Finse …
            -/
        (by simpa using Nat.sub_lt (card_pos.mpr h) Nat.zero_lt_one) :=
            /-
              🎉 no goals
            -/
                                 /-
                                   α : Type u_1
                                   inst✝ : LinearOrder α
                                   s : Finset α
                                   h : s.Nonempty
                                   ⊢ LT.lt (HSub.hSub (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length 1) (Finse …
                                 -/
  (sorted_last_eq_max'_aux _ (by simpa using Nat.sub_lt (card_pos.mpr h) Nat.zero_lt_one) _).symm
                                 /-
                                   🎉 no goals
                                 -/


/-- Given a finset `s` of cardinality `k` in a linear order `α`, the map `orderIsoOfFin s h`
is the increasing bijection between `Fin k` and `s` as an `OrderIso`. Here, `h` is a proof that
the cardinality of `s` is `k`. We use this instead of an iso `Fin s.card ≃o s` to avoid
casting issues in further uses of this function. -/
def orderIsoOfFin (s : Finset α) {k : ℕ} (h : s.card = k) : Fin k ≃o s :=
  OrderIso.trans (Fin.castOrderIso ((length_sort (α := α) (· ≤ ·)).trans h).symm) <|
    (s.sort_sorted_lt.getIso _).trans <| OrderIso.setCongr _ _ <| Set.ext fun _ => mem_sort _


/-- Given a finset `s` of cardinality `k` in a linear order `α`, the map `orderEmbOfFin s h` is
the increasing bijection between `Fin k` and `s` as an order embedding into `α`. Here, `h` is a
proof that the cardinality of `s` is `k`. We use this instead of an embedding `Fin s.card ↪o α` to
avoid casting issues in further uses of this function. -/
def orderEmbOfFin (s : Finset α) {k : ℕ} (h : s.card = k) : Fin k ↪o α :=
  (orderIsoOfFin s h).toOrderEmbedding.trans (OrderEmbedding.subtype _)


@[simp]
theorem coe_orderIsoOfFin_apply (s : Finset α) {k : ℕ} (h : s.card = k) (i : Fin k) :
    ↑(orderIsoOfFin s h i) = orderEmbOfFin s h i :=
  rfl


theorem orderIsoOfFin_symm_apply (s : Finset α) {k : ℕ} (h : s.card = k) (x : s) :
    ↑((s.orderIsoOfFin h).symm x) = (s.sort (· ≤ ·)).indexOf ↑x :=
  rfl


theorem orderEmbOfFin_apply (s : Finset α) {k : ℕ} (h : s.card = k) (i : Fin k) :
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    inst✝ : LinearOrder α
                                                    s : Finset α
                                                    k : Nat
                                                    h : Eq s.card k
                                                    i : Fin k
                                                    ⊢ LT.lt (↑i) (Finset.sort (fun x1 x2 => LE.le x1 x2) s).length
                                                  -/
    s.orderEmbOfFin h i = (s.sort (· ≤ ·))[i]'(by rw [length_sort, h]; exact i.2) :=
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  rfl


@[simp]
theorem orderEmbOfFin_mem (s : Finset α) {k : ℕ} (h : s.card = k) (i : Fin k) :
    s.orderEmbOfFin h i ∈ s :=
  (s.orderIsoOfFin h i).2


@[simp]
theorem range_orderEmbOfFin (s : Finset α) {k : ℕ} (h : s.card = k) :
    Set.range (s.orderEmbOfFin h) = s := by
  simp only [orderEmbOfFin, Set.range_comp ((↑) : _ → α) (s.orderIsoOfFin h),
  RelEmbedding.coe_trans, Set.image_univ, Finset.orderEmbOfFin, RelIso.range_eq,
    OrderEmbedding.subtype_apply, OrderIso.coe_toOrderEmbedding, eq_self_iff_true,
    Subtype.range_coe_subtype, Finset.setOf_mem, Finset.coe_inj]


/-- The bijection `orderEmbOfFin s h` sends `0` to the minimum of `s`. -/
theorem orderEmbOfFin_zero {s : Finset α} {k : ℕ} (h : s.card = k) (hz : 0 < k) :
    orderEmbOfFin s h ⟨0, hz⟩ = s.min' (card_pos.mp (h.symm ▸ hz)) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Finset α
    k : Nat
    h : Eq s.card k
    hz : LT.lt 0 k
    ⊢ Eq ((s.orderEmbOfFin h) ⟨0, hz⟩) (s.min' ⋯)
  -/
  simp only [orderEmbOfFin_apply, Fin.getElem_fin, sorted_zero_eq_min']
  /-
    🎉 no goals
  -/


/-- The bijection `orderEmbOfFin s h` sends `k-1` to the maximum of `s`. -/
theorem orderEmbOfFin_last {s : Finset α} {k : ℕ} (h : s.card = k) (hz : 0 < k) :
    orderEmbOfFin s h ⟨k - 1, Nat.sub_lt hz (Nat.succ_pos 0)⟩ =
      s.max' (card_pos.mp (h.symm ▸ hz)) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Finset α
    k : Nat
    h : Eq s.card k
    hz : LT.lt 0 k
    ⊢ Eq ((s.orderEmbOfFin h) ⟨HSub.hSub k 1, ⋯⟩) (s.max' ⋯)
  -/
  simp [orderEmbOfFin_apply, max'_eq_sorted_last, h]
  /-
    🎉 no goals
  -/


/-- `orderEmbOfFin {a} h` sends any argument to `a`. -/
@[simp]
theorem orderEmbOfFin_singleton (a : α) (i : Fin 1) :
    orderEmbOfFin {a} (card_singleton a) i = a := by
  rw [Subsingleton.elim i ⟨0, Nat.zero_lt_one⟩, orderEmbOfFin_zero _ Nat.zero_lt_one,
    min'_singleton]


/-- Any increasing map `f` from `Fin k` to a finset of cardinality `k` has to coincide with
the increasing bijection `orderEmbOfFin s h`. -/
theorem orderEmbOfFin_unique {s : Finset α} {k : ℕ} (h : s.card = k) {f : Fin k → α}
    (hfs : ∀ x, f x ∈ s) (hmono : StrictMono f) : f = s.orderEmbOfFin h := by
  rw [← hmono.range_inj (s.orderEmbOfFin h).strictMono, range_orderEmbOfFin, ← Set.image_univ,
    ← coe_univ, ← coe_image, coe_inj]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Finset α
    k : Nat
    h : Eq s.card k
    f : Fin k → α
    hfs : ∀ (x : Fin k), Membership.mem s (f x)
    hmono : StrictMono f
    ⊢ Eq (Finset.image f Finset.univ) s
  -/
  refine eq_of_subset_of_card_le (fun x hx => ?_) ?_
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      k : Nat
      h : Eq s.card k
      f : Fin k → α
      hfs : ∀ (x : Fin k), Membership.mem s (f x)
      hmono : StrictMono f
      x : α
      hx : Membership.mem (Finset.image f Finset.univ) x
      ⊢ Membership.mem s x
    -/
  · rcases mem_image.1 hx with ⟨x, _, rfl⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      k : Nat
      h : Eq s.card k
      f : Fin k → α
      hfs : ∀ (x : Fin k), Membership.mem s (f x)
      hmono : StrictMono f
      x : Fin k
      left✝ : Membership.mem Finset.univ x
      hx : Membership.mem (Finset.image f Finset.univ) (f x)
      ⊢ Membership.mem s (f x)
    -/
    exact hfs x
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      k : Nat
      h : Eq s.card k
      f : Fin k → α
      hfs : ∀ (x : Fin k), Membership.mem s (f x)
      hmono : StrictMono f
      ⊢ LE.le s.card (Finset.image f Finset.univ).card
    -/
  · rw [h, card_image_of_injective _ hmono.injective, card_univ, Fintype.card_fin]
    /-
      🎉 no goals
    -/


/-- An order embedding `f` from `Fin k` to a finset of cardinality `k` has to coincide with
the increasing bijection `orderEmbOfFin s h`. -/
theorem orderEmbOfFin_unique' {s : Finset α} {k : ℕ} (h : s.card = k) {f : Fin k ↪o α}
    (hfs : ∀ x, f x ∈ s) : f = s.orderEmbOfFin h :=
  RelEmbedding.ext <| funext_iff.1 <| orderEmbOfFin_unique h hfs f.strictMono


/-- Two parametrizations `orderEmbOfFin` of the same set take the same value on `i` and `j` if
and only if `i = j`. Since they can be defined on a priori not defeq types `Fin k` and `Fin l`
(although necessarily `k = l`), the conclusion is rather written `(i : ℕ) = (j : ℕ)`. -/
@[simp]
theorem orderEmbOfFin_eq_orderEmbOfFin_iff {k l : ℕ} {s : Finset α} {i : Fin k} {j : Fin l}
    {h : s.card = k} {h' : s.card = l} :
    s.orderEmbOfFin h i = s.orderEmbOfFin h' j ↔ (i : ℕ) = (j : ℕ) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    k l : Nat
    s : Finset α
    i : Fin k
    j : Fin l
    h : Eq s.card k
    h' : Eq s.card l
    ⊢ Iff (Eq ((s.orderEmbOfFin h) i) ((s.orderEmbOfFin h') j)) (Eq ↑i ↑j)
  -/
  substs k l
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Finset α
    i j : Fin s.card
    ⊢ Iff (Eq ((s.orderEmbOfFin ⋯) i) ((s.orderEmbOfFin ⋯) j)) (Eq ↑i ↑j)
  -/
  exact (s.orderEmbOfFin rfl).eq_iff_eq.trans Fin.ext_iff
  /-
    🎉 no goals
  -/


/-- Given a finset `s` of size at least `k` in a linear order `α`, the map `orderEmbOfCardLe`
is an order embedding from `Fin k` to `α` whose image is contained in `s`. Specifically, it maps
`Fin k` to an initial segment of `s`. -/
def orderEmbOfCardLe (s : Finset α) {k : ℕ} (h : k ≤ s.card) : Fin k ↪o α :=
  (Fin.castLEOrderEmb h).trans (s.orderEmbOfFin rfl)


theorem orderEmbOfCardLe_mem (s : Finset α) {k : ℕ} (h : k ≤ s.card) (a) :
    orderEmbOfCardLe s h a ∈ s := by
  simp only [orderEmbOfCardLe, RelEmbedding.coe_trans, Finset.orderEmbOfFin_mem,
    Function.comp_apply]


unsafe instance [Repr α] : Repr (Finset α) where
  reprPrec s _ :=
    -- multiset uses `0` not `∅` for empty sets
    if s.card = 0 then "∅" else repr s.1


theorem sort_univ (n : ℕ) : Finset.univ.sort (fun x y : Fin n => x ≤ y) = List.finRange n :=
  List.eq_of_perm_of_sorted
    (List.perm_of_nodup_nodup_toFinset_eq
                                                             /-
                                                               n : Nat
                                                               ⊢ Eq (Finset.sort (fun x y => LE.le x y) Finset.univ).toFinset (List.finRange  …
                                                             -/
      (Finset.univ.sort_nodup _) (List.nodup_finRange n) (by simp))
                                                             /-
                                                               🎉 no goals
                                                             -/
    (Finset.univ.sort_sorted LE.le)
    (List.pairwise_le_finRange n)


/-- Given a `Fintype` `α` of cardinality `k`, the map `orderIsoFinOfCardEq s h` is the increasing
bijection between `Fin k` and `α` as an `OrderIso`. Here, `h` is a proof that the cardinality of `α`
is `k`. We use this instead of an iso `Fin (Fintype.card α) ≃o α` to avoid casting issues in further
uses of this function. -/
def Fintype.orderIsoFinOfCardEq
    (α : Type*) [LinearOrder α] [Fintype α] {k : ℕ} (h : Fintype.card α = k) :
    Fin k ≃o α :=
  (Finset.univ.orderIsoOfFin h).trans
    ((OrderIso.setCongr _ _ Finset.coe_univ).trans OrderIso.Set.univ)


/-- Any finite linear order order-embeds into any infinite linear order. -/
lemma nonempty_orderEmbedding_of_finite_infinite
    (α : Type*) [LinearOrder α] [hα : Finite α]
    (β : Type*) [LinearOrder β] [hβ : Infinite β] : Nonempty (α ↪o β) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    hα : Finite α
    β : Type u_2
    inst✝ : LinearOrder β
    hβ : Infinite β
    ⊢ Nonempty (OrderEmbedding α β)
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    hα : Finite α
    β : Type u_2
    inst✝ : LinearOrder β
    hβ : Infinite β
    this : Fintype α
    ⊢ Nonempty (OrderEmbedding α β)
  -/
  obtain ⟨s, hs⟩ := Infinite.exists_subset_card_eq β (Fintype.card α)
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    hα : Finite α
    β : Type u_2
    inst✝ : LinearOrder β
    hβ : Infinite β
    this : Fintype α
    s : Finset β
    hs : Eq s.card (Fintype.card α)
    ⊢ Nonempty (OrderEmbedding α β)
  -/
  exact ⟨((Fintype.orderIsoFinOfCardEq α rfl).symm.toOrderEmbedding).trans (s.orderEmbOfFin hs)⟩
  /-
    🎉 no goals
  -/

