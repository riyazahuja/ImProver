/-- The pigeonhole principle for finitely many pigeons counted by weight, strict inequality version:
if the total weight of a finite set of pigeons is greater than `n • b`, and they are sorted into
`n` pigeonholes, then for some pigeonhole, the total weight of the pigeons in this pigeonhole is
greater than `b`. -/
theorem exists_lt_sum_fiber_of_maps_to_of_nsmul_lt_sum (hf : ∀ a ∈ s, f a ∈ t)
    (hb : #t • b < ∑ x ∈ s, w x) : ∃ y ∈ t, b < ∑ x ∈ s with f x = y, w x :=
                            /-
                              α : Type u
                              β : Type v
                              M : Type w
                              inst✝¹ : DecidableEq β
                              s : Finset α
                              t : Finset β
                              f : α → β
                              w : α → M
                              b : M
                              inst✝ : LinearOrderedCancelAddCommMonoid M
                              hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
                              hb : LT.lt (HSMul.hSMul t.card b) (s.sum fun x => w x)
                              ⊢ LT.lt (t.sum fun i => b) (t.sum fun i => (Finset.filter (fun x => Eq (f x) i …
                            -/
  exists_lt_of_sum_lt <| by simpa only [sum_fiberwise_of_maps_to hf, sum_const]
                            /-
                              🎉 no goals
                            -/


/-- The pigeonhole principle for finitely many pigeons counted by weight, strict inequality version:
if the total weight of a finite set of pigeons is less than `n • b`, and they are sorted into `n`
pigeonholes, then for some pigeonhole, the total weight of the pigeons in this pigeonhole is less
than `b`. -/
theorem exists_sum_fiber_lt_of_maps_to_of_sum_lt_nsmul (hf : ∀ a ∈ s, f a ∈ t)
    (hb : ∑ x ∈ s, w x < #t • b) : ∃ y ∈ t, ∑ x ∈ s with f x = y, w x < b :=
  exists_lt_sum_fiber_of_maps_to_of_nsmul_lt_sum (M := Mᵒᵈ) hf hb


/-- The pigeonhole principle for finitely many pigeons counted by weight, strict inequality version:
if the total weight of a finite set of pigeons is greater than `n • b`, they are sorted into some
pigeonholes, and for all but `n` pigeonholes the total weight of the pigeons there is nonpositive,
then for at least one of these `n` pigeonholes, the total weight of the pigeons in this pigeonhole
is greater than `b`. -/
theorem exists_lt_sum_fiber_of_sum_fiber_nonpos_of_nsmul_lt_sum
    (ht : ∀ y ∉ t, ∑ x ∈ s with f x = y, w x ≤ 0)
    (hb : #t • b < ∑ x ∈ s, w x) : ∃ y ∈ t, b < ∑ x ∈ s with f x = y, w x :=
  exists_lt_of_sum_lt <|
    calc
                                       /-
                                         α : Type u
                                         β : Type v
                                         M : Type w
                                         inst✝¹ : DecidableEq β
                                         s : Finset α
                                         t : Finset β
                                         f : α → β
                                         w : α → M
                                         b : M
                                         inst✝ : LinearOrderedCancelAddCommMonoid M
                                         ht : ∀ (y : β), Not (Membership.mem t y) → LE.le ((Finset.filter (fun x => Eq  …
                                         hb : LT.lt (HSMul.hSMul t.card b) (s.sum fun x => w x)
                                         ⊢ LT.lt (t.sum fun _y => b) (s.sum fun x => w x)
                                       -/
      ∑ _y ∈ t, b < ∑ x ∈ s, w x := by simpa
                                       /-
                                         🎉 no goals
                                       -/
      _ ≤ ∑ y ∈ t, ∑ x ∈ s with f x = y, w x := sum_le_sum_fiberwise_of_sum_fiber_nonpos ht


/-- The pigeonhole principle for finitely many pigeons counted by weight, strict inequality version:
if the total weight of a finite set of pigeons is less than `n • b`, they are sorted into some
pigeonholes, and for all but `n` pigeonholes the total weight of the pigeons there is nonnegative,
then for at least one of these `n` pigeonholes, the total weight of the pigeons in this pigeonhole
is less than `b`. -/
theorem exists_sum_fiber_lt_of_sum_fiber_nonneg_of_sum_lt_nsmul
    (ht : ∀ y ∉ t, (0 : M) ≤ ∑ x ∈ s with f x = y, w x) (hb : ∑ x ∈ s, w x < #t • b) :
    ∃ y ∈ t, ∑ x ∈ s with f x = y, w x < b :=
  exists_lt_sum_fiber_of_sum_fiber_nonpos_of_nsmul_lt_sum (M := Mᵒᵈ) ht hb


/-- The pigeonhole principle for finitely many pigeons counted by weight, non-strict inequality
version: if the total weight of a finite set of pigeons is greater than or equal to `n • b`, and
they are sorted into `n > 0` pigeonholes, then for some pigeonhole, the total weight of the pigeons
in this pigeonhole is greater than or equal to `b`. -/
theorem exists_le_sum_fiber_of_maps_to_of_nsmul_le_sum (hf : ∀ a ∈ s, f a ∈ t) (ht : t.Nonempty)
    (hb : #t • b ≤ ∑ x ∈ s, w x) : ∃ y ∈ t, b ≤ ∑ x ∈ s with f x = y, w x :=
                               /-
                                 α : Type u
                                 β : Type v
                                 M : Type w
                                 inst✝¹ : DecidableEq β
                                 s : Finset α
                                 t : Finset β
                                 f : α → β
                                 w : α → M
                                 b : M
                                 inst✝ : LinearOrderedCancelAddCommMonoid M
                                 hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
                                 ht : t.Nonempty
                                 hb : LE.le (HSMul.hSMul t.card b) (s.sum fun x => w x)
                                 ⊢ LE.le (t.sum fun i => b) (t.sum fun i => (Finset.filter (fun x => Eq (f x) i …
                               -/
  exists_le_of_sum_le ht <| by simpa only [sum_fiberwise_of_maps_to hf, sum_const]
                               /-
                                 🎉 no goals
                               -/


/-- The pigeonhole principle for finitely many pigeons counted by weight, non-strict inequality
version: if the total weight of a finite set of pigeons is less than or equal to `n • b`, and they
are sorted into `n > 0` pigeonholes, then for some pigeonhole, the total weight of the pigeons in
this pigeonhole is less than or equal to `b`. -/
theorem exists_sum_fiber_le_of_maps_to_of_sum_le_nsmul (hf : ∀ a ∈ s, f a ∈ t) (ht : t.Nonempty)
    (hb : ∑ x ∈ s, w x ≤ #t • b) : ∃ y ∈ t, ∑ x ∈ s with f x = y, w x ≤ b :=
  exists_le_sum_fiber_of_maps_to_of_nsmul_le_sum (M := Mᵒᵈ) hf ht hb


/-- The pigeonhole principle for finitely many pigeons counted by weight, non-strict inequality
version: if the total weight of a finite set of pigeons is greater than or equal to `n • b`, they
are sorted into some pigeonholes, and for all but `n > 0` pigeonholes the total weight of the
pigeons there is nonpositive, then for at least one of these `n` pigeonholes, the total weight of
the pigeons in this pigeonhole is greater than or equal to `b`. -/
theorem exists_le_sum_fiber_of_sum_fiber_nonpos_of_nsmul_le_sum
    (hf : ∀ y ∉ t, ∑ x ∈ s with f x = y, w x ≤ 0) (ht : t.Nonempty)
    (hb : #t • b ≤ ∑ x ∈ s, w x) : ∃ y ∈ t, b ≤ ∑ x ∈ s with f x = y, w x :=
  exists_le_of_sum_le ht <|
    calc
                                       /-
                                         α : Type u
                                         β : Type v
                                         M : Type w
                                         inst✝¹ : DecidableEq β
                                         s : Finset α
                                         t : Finset β
                                         f : α → β
                                         w : α → M
                                         b : M
                                         inst✝ : LinearOrderedCancelAddCommMonoid M
                                         hf : ∀ (y : β), Not (Membership.mem t y) → LE.le ((Finset.filter (fun x => Eq  …
                                         ht : t.Nonempty
                                         hb : LE.le (HSMul.hSMul t.card b) (s.sum fun x => w x)
                                         ⊢ LE.le (t.sum fun _y => b) (s.sum fun x => w x)
                                       -/
      ∑ _y ∈ t, b ≤ ∑ x ∈ s, w x := by simpa
                                       /-
                                         🎉 no goals
                                       -/
      _ ≤ ∑ y ∈ t, ∑ x ∈ s with f x = y, w x :=
        sum_le_sum_fiberwise_of_sum_fiber_nonpos hf


/-- The pigeonhole principle for finitely many pigeons counted by weight, non-strict inequality
version: if the total weight of a finite set of pigeons is less than or equal to `n • b`, they are
sorted into some pigeonholes, and for all but `n > 0` pigeonholes the total weight of the pigeons
there is nonnegative, then for at least one of these `n` pigeonholes, the total weight of the
pigeons in this pigeonhole is less than or equal to `b`. -/
theorem exists_sum_fiber_le_of_sum_fiber_nonneg_of_sum_le_nsmul
    (hf : ∀ y ∉ t, (0 : M) ≤ ∑ x ∈ s with f x = y, w x) (ht : t.Nonempty)
    (hb : ∑ x ∈ s, w x ≤ #t • b) : ∃ y ∈ t, ∑ x ∈ s with f x = y, w x ≤ b :=
  exists_le_sum_fiber_of_sum_fiber_nonpos_of_nsmul_le_sum (M := Mᵒᵈ) hf ht hb


/-- The pigeonhole principle for finitely many pigeons counted by heads: there is a pigeonhole with
at least as many pigeons as the ceiling of the average number of pigeons across all pigeonholes. -/
theorem exists_lt_card_fiber_of_nsmul_lt_card_of_maps_to (hf : ∀ a ∈ s, f a ∈ t)
    (ht : #t • b < #s) : ∃ y ∈ t, b < #{x ∈ s | f x = y} := by
  /-
    α : Type u
    β : Type v
    M : Type w
    inst✝¹ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    b : M
    inst✝ : LinearOrderedCommSemiring M
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
    ht : LT.lt (HSMul.hSMul t.card b) ↑s.card
    ⊢ Exists fun y => And (Membership.mem t y) (LT.lt b ↑(Finset.filter (fun x =>  …
  -/
  simp_rw [cast_card] at ht ⊢
  /-
    α : Type u
    β : Type v
    M : Type w
    inst✝¹ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    b : M
    inst✝ : LinearOrderedCommSemiring M
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
    ht : LT.lt (HSMul.hSMul t.card b) (s.sum fun x => 1)
    ⊢ Exists fun y => And (Membership.mem t y) (LT.lt b ((Finset.filter (fun x =>  …
  -/
  exact exists_lt_sum_fiber_of_maps_to_of_nsmul_lt_sum hf ht
  /-
    🎉 no goals
  -/


/-- The pigeonhole principle for finitely many pigeons counted by heads: there is a pigeonhole with
at least as many pigeons as the ceiling of the average number of pigeons across all pigeonholes.
("The maximum is at least the mean" specialized to integers.)

More formally, given a function between finite sets `s` and `t` and a natural number `n` such that
`#t * n < #s`, there exists `y ∈ t` such that its preimage in `s` has more than `n`
elements. -/
theorem exists_lt_card_fiber_of_mul_lt_card_of_maps_to (hf : ∀ a ∈ s, f a ∈ t)
    (hn : #t * n < #s) : ∃ y ∈ t, n < #{x ∈ s | f x = y} :=
  exists_lt_card_fiber_of_nsmul_lt_card_of_maps_to hf hn


/-- The pigeonhole principle for finitely many pigeons counted by heads: there is a pigeonhole with
at most as many pigeons as the floor of the average number of pigeons across all pigeonholes. -/
theorem exists_card_fiber_lt_of_card_lt_nsmul (ht : #s < #t • b) :
    ∃ y ∈ t, #{x ∈ s | f x = y} < b := by
  /-
    α : Type u
    β : Type v
    M : Type w
    inst✝¹ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    b : M
    inst✝ : LinearOrderedCommSemiring M
    ht : LT.lt (↑s.card) (HSMul.hSMul t.card b)
    ⊢ Exists fun y => And (Membership.mem t y) (LT.lt (↑(Finset.filter (fun x => E …
  -/
  simp_rw [cast_card] at ht ⊢
  exact
    exists_sum_fiber_lt_of_sum_fiber_nonneg_of_sum_lt_nsmul
      (fun _ _ => sum_nonneg fun _ _ => zero_le_one) ht


/-- The pigeonhole principle for finitely many pigeons counted by heads: there is a pigeonhole with
at most as many pigeons as the floor of the average number of pigeons across all pigeonholes.  ("The
minimum is at most the mean" specialized to integers.)

More formally, given a function `f`, a finite sets `s` in its domain, a finite set `t` in its
codomain, and a natural number `n` such that `#s < #t * n`, there exists `y ∈ t` such that
its preimage in `s` has less than `n` elements. -/
theorem exists_card_fiber_lt_of_card_lt_mul (hn : #s < #t * n) : ∃ y ∈ t, #{x ∈ s | f x = y} < n :=
  exists_card_fiber_lt_of_card_lt_nsmul hn


/-- The pigeonhole principle for finitely many pigeons counted by heads: given a function between
finite sets `s` and `t` and a number `b` such that `#t • b ≤ #s`, there exists `y ∈ t` such
that its preimage in `s` has at least `b` elements.
See also `Finset.exists_lt_card_fiber_of_nsmul_lt_card_of_maps_to` for a stronger statement. -/
theorem exists_le_card_fiber_of_nsmul_le_card_of_maps_to (hf : ∀ a ∈ s, f a ∈ t) (ht : t.Nonempty)
    (hb : #t • b ≤ #s) : ∃ y ∈ t, b ≤ #{x ∈ s | f x = y} := by
  /-
    α : Type u
    β : Type v
    M : Type w
    inst✝¹ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    b : M
    inst✝ : LinearOrderedCommSemiring M
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
    ht : t.Nonempty
    hb : LE.le (HSMul.hSMul t.card b) ↑s.card
    ⊢ Exists fun y => And (Membership.mem t y) (LE.le b ↑(Finset.filter (fun x =>  …
  -/
  simp_rw [cast_card] at hb ⊢
  /-
    α : Type u
    β : Type v
    M : Type w
    inst✝¹ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    b : M
    inst✝ : LinearOrderedCommSemiring M
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
    ht : t.Nonempty
    hb : LE.le (HSMul.hSMul t.card b) (s.sum fun x => 1)
    ⊢ Exists fun y => And (Membership.mem t y) (LE.le b ((Finset.filter (fun x =>  …
  -/
  exact exists_le_sum_fiber_of_maps_to_of_nsmul_le_sum hf ht hb
  /-
    🎉 no goals
  -/


/-- The pigeonhole principle for finitely many pigeons counted by heads: given a function between
finite sets `s` and `t` and a natural number `b` such that `#t * n ≤ #s`, there exists
`y ∈ t` such that its preimage in `s` has at least `n` elements. See also
`Finset.exists_lt_card_fiber_of_mul_lt_card_of_maps_to` for a stronger statement. -/
theorem exists_le_card_fiber_of_mul_le_card_of_maps_to (hf : ∀ a ∈ s, f a ∈ t) (ht : t.Nonempty)
    (hn : #t * n ≤ #s) : ∃ y ∈ t, n ≤ #{x ∈ s | f x = y} :=
  exists_le_card_fiber_of_nsmul_le_card_of_maps_to hf ht hn


/-- The pigeonhole principle for finitely many pigeons counted by heads: given a function `f`, a
finite sets `s` and `t`, and a number `b` such that `#s ≤ #t • b`, there exists `y ∈ t` such
that its preimage in `s` has no more than `b` elements.
See also `Finset.exists_card_fiber_lt_of_card_lt_nsmul` for a stronger statement. -/
theorem exists_card_fiber_le_of_card_le_nsmul (ht : t.Nonempty) (hb : #s ≤ #t • b) :
    ∃ y ∈ t, #{x ∈ s | f x = y} ≤ b := by
  /-
    α : Type u
    β : Type v
    M : Type w
    inst✝¹ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    b : M
    inst✝ : LinearOrderedCommSemiring M
    ht : t.Nonempty
    hb : LE.le (↑s.card) (HSMul.hSMul t.card b)
    ⊢ Exists fun y => And (Membership.mem t y) (LE.le (↑(Finset.filter (fun x => E …
  -/
  simp_rw [cast_card] at hb ⊢
  refine
    exists_sum_fiber_le_of_sum_fiber_nonneg_of_sum_le_nsmul
      (fun _ _ => sum_nonneg fun _ _ => zero_le_one) ht hb


/-- The pigeonhole principle for finitely many pigeons counted by heads: given a function `f`, a
finite sets `s` in its domain, a finite set `t` in its codomain, and a natural number `n` such that
`#s ≤ #t * n`, there exists `y ∈ t` such that its preimage in `s` has no more than `n`
elements. See also `Finset.exists_card_fiber_lt_of_card_lt_mul` for a stronger statement. -/
theorem exists_card_fiber_le_of_card_le_mul (ht : t.Nonempty) (hn : #s ≤ #t * n) :
    ∃ y ∈ t, #{x ∈ s | f x = y} ≤ n :=
  exists_card_fiber_le_of_card_le_nsmul ht hn


/-- The pigeonhole principle for finitely many pigeons of different weights, strict inequality
version: there is a pigeonhole with the total weight of pigeons in it greater than `b` provided that
the total number of pigeonholes times `b` is less than the total weight of all pigeons. -/
theorem exists_lt_sum_fiber_of_nsmul_lt_sum (hb : card β • b < ∑ x, w x) :
    ∃ y, b < ∑ x with f x = y, w x :=
  let ⟨y, _, hy⟩ := exists_lt_sum_fiber_of_maps_to_of_nsmul_lt_sum (fun _ _ => mem_univ _) hb
  ⟨y, hy⟩


/-- The pigeonhole principle for finitely many pigeons of different weights, non-strict inequality
version: there is a pigeonhole with the total weight of pigeons in it greater than or equal to `b`
provided that the total number of pigeonholes times `b` is less than or equal to the total weight of
all pigeons. -/
theorem exists_le_sum_fiber_of_nsmul_le_sum [Nonempty β] (hb : card β • b ≤ ∑ x, w x) :
    ∃ y, b ≤ ∑ x with f x = y, w x :=
  let ⟨y, _, hy⟩ :=
    exists_le_sum_fiber_of_maps_to_of_nsmul_le_sum (fun _ _ => mem_univ _) univ_nonempty hb
  ⟨y, hy⟩


/-- The pigeonhole principle for finitely many pigeons of different weights, strict inequality
version: there is a pigeonhole with the total weight of pigeons in it less than `b` provided that
the total number of pigeonholes times `b` is greater than the total weight of all pigeons. -/
theorem exists_sum_fiber_lt_of_sum_lt_nsmul (hb : ∑ x, w x < card β • b) :
    ∃ y, ∑ x with f x = y, w x < b :=
  exists_lt_sum_fiber_of_nsmul_lt_sum (M := Mᵒᵈ) _ hb


/-- The pigeonhole principle for finitely many pigeons of different weights, non-strict inequality
version: there is a pigeonhole with the total weight of pigeons in it less than or equal to `b`
provided that the total number of pigeonholes times `b` is greater than or equal to the total weight
of all pigeons. -/
theorem exists_sum_fiber_le_of_sum_le_nsmul [Nonempty β] (hb : ∑ x, w x ≤ card β • b) :
    ∃ y, ∑ x with f x = y, w x ≤ b :=
  exists_le_sum_fiber_of_nsmul_le_sum (M := Mᵒᵈ) _ hb


/-- The strong pigeonhole principle for finitely many pigeons and pigeonholes. There is a pigeonhole
with at least as many pigeons as the ceiling of the average number of pigeons across all
pigeonholes. -/
theorem exists_lt_card_fiber_of_nsmul_lt_card (hb : card β • b < card α) :
    ∃ y : β, b < #{x | f x = y} :=
  let ⟨y, _, h⟩ := exists_lt_card_fiber_of_nsmul_lt_card_of_maps_to (fun _ _ => mem_univ _) hb
  ⟨y, h⟩


/-- The strong pigeonhole principle for finitely many pigeons and pigeonholes.
There is a pigeonhole with at least as many pigeons as
the ceiling of the average number of pigeons across all pigeonholes.
("The maximum is at least the mean" specialized to integers.)

More formally, given a function `f` between finite types `α` and `β` and a number `n` such that
`card β * n < card α`, there exists an element `y : β` such that its preimage has more than `n`
elements. -/
theorem exists_lt_card_fiber_of_mul_lt_card (hn : card β * n < card α) :
    ∃ y : β, n < #{x | f x = y} :=
  exists_lt_card_fiber_of_nsmul_lt_card _ hn


/-- The strong pigeonhole principle for finitely many pigeons and pigeonholes. There is a pigeonhole
with at most as many pigeons as the floor of the average number of pigeons across all pigeonholes.
-/
theorem exists_card_fiber_lt_of_card_lt_nsmul (hb : ↑(card α) < card β • b) :
    ∃ y : β, #{x | f x = y} < b :=
  let ⟨y, _, h⟩ := Finset.exists_card_fiber_lt_of_card_lt_nsmul (f := f) hb
  ⟨y, h⟩


/-- The strong pigeonhole principle for finitely many pigeons and pigeonholes.
There is a pigeonhole with at most as many pigeons as
the floor of the average number of pigeons across all pigeonholes.
("The minimum is at most the mean" specialized to integers.)

More formally, given a function `f` between finite types `α` and `β` and a number `n` such that
`card α < card β * n`, there exists an element `y : β` such that its preimage has less than `n`
elements. -/
theorem exists_card_fiber_lt_of_card_lt_mul (hn : card α < card β * n) :
    ∃ y : β, #{x | f x = y} < n :=
  exists_card_fiber_lt_of_card_lt_nsmul _ hn


/-- The strong pigeonhole principle for finitely many pigeons and pigeonholes.  Given a function `f`
between finite types `α` and `β` and a number `b` such that `card β • b ≤ card α`, there exists an
element `y : β` such that its preimage has at least `b` elements.
See also `Fintype.exists_lt_card_fiber_of_nsmul_lt_card` for a stronger statement. -/
theorem exists_le_card_fiber_of_nsmul_le_card [Nonempty β] (hb : card β • b ≤ card α) :
    ∃ y : β, b ≤ #{x | f x = y} :=
  let ⟨y, _, h⟩ :=
    exists_le_card_fiber_of_nsmul_le_card_of_maps_to (fun _ _ => mem_univ _) univ_nonempty hb
  ⟨y, h⟩


/-- The strong pigeonhole principle for finitely many pigeons and pigeonholes.  Given a function `f`
between finite types `α` and `β` and a number `n` such that `card β * n ≤ card α`, there exists an
element `y : β` such that its preimage has at least `n` elements. See also
`Fintype.exists_lt_card_fiber_of_mul_lt_card` for a stronger statement. -/
theorem exists_le_card_fiber_of_mul_le_card [Nonempty β] (hn : card β * n ≤ card α) :
    ∃ y : β, n ≤ #{x | f x = y} :=
  exists_le_card_fiber_of_nsmul_le_card _ hn


/-- The strong pigeonhole principle for finitely many pigeons and pigeonholes.  Given a function `f`
between finite types `α` and `β` and a number `b` such that `card α ≤ card β • b`, there exists an
element `y : β` such that its preimage has at most `b` elements.
See also `Fintype.exists_card_fiber_lt_of_card_lt_nsmul` for a stronger statement. -/
theorem exists_card_fiber_le_of_card_le_nsmul [Nonempty β] (hb : ↑(card α) ≤ card β • b) :
    ∃ y : β, #{x | f x = y} ≤ b :=
  let ⟨y, _, h⟩ := Finset.exists_card_fiber_le_of_card_le_nsmul univ_nonempty hb
  ⟨y, h⟩


/-- The strong pigeonhole principle for finitely many pigeons and pigeonholes.  Given a function `f`
between finite types `α` and `β` and a number `n` such that `card α ≤ card β * n`, there exists an
element `y : β` such that its preimage has at most `n` elements. See also
`Fintype.exists_card_fiber_lt_of_card_lt_mul` for a stronger statement. -/
theorem exists_card_fiber_le_of_card_le_mul [Nonempty β] (hn : card α ≤ card β * n) :
    ∃ y : β, #{x | f x = y} ≤ n :=
  exists_card_fiber_le_of_card_le_nsmul _ hn


/-- If `s` is an infinite set of natural numbers and `k > 0`, then `s` contains two elements `m < n`
that are equal mod `k`. -/
theorem exists_lt_modEq_of_infinite {s : Set ℕ} (hs : s.Infinite) {k : ℕ} (hk : 0 < k) :
    ∃ m ∈ s, ∃ n ∈ s, m < n ∧ m ≡ n [MOD k] :=
  (hs.exists_lt_map_eq_of_mapsTo fun n _ => show n % k ∈ Iio k from Nat.mod_lt n hk) <|
    finite_lt_nat k


