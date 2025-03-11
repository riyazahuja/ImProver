/-- `Sorted r l` is the same as `List.Pairwise r l`, preferred in the case that `r`
  is a `<` or `≤`-like relation (transitive and antisymmetric or asymmetric) -/
def Sorted :=
  @Pairwise


instance decidableSorted [DecidableRel r] (l : List α) : Decidable (Sorted r l) :=
  List.instDecidablePairwise _


protected theorem Sorted.le_of_lt [Preorder α] {l : List α} (h : l.Sorted (· < ·)) :
    l.Sorted (· ≤ ·) :=
  h.imp le_of_lt


protected theorem Sorted.lt_of_le [PartialOrder α] {l : List α} (h₁ : l.Sorted (· ≤ ·))
    (h₂ : l.Nodup) : l.Sorted (· < ·) :=
  h₁.imp₂ (fun _ _ => lt_of_le_of_ne) h₂


protected theorem Sorted.ge_of_gt [Preorder α] {l : List α} (h : l.Sorted (· > ·)) :
    l.Sorted (· ≥ ·) :=
  h.imp le_of_lt


protected theorem Sorted.gt_of_ge [PartialOrder α] {l : List α} (h₁ : l.Sorted (· ≥ ·))
    (h₂ : l.Nodup) : l.Sorted (· > ·) :=
                                            /-
                                              α : Type u
                                              inst✝ : PartialOrder α
                                              l : List α
                                              h₁ : List.Sorted (fun x1 x2 => GE.ge x1 x2) l
                                              h₂ : l.Nodup
                                              ⊢ List.Pairwise (fun x x_1 => Ne x_1 x) l
                                            -/
  h₁.imp₂ (fun _ _ => lt_of_le_of_ne) <| by simp_rw [ne_comm]; exact h₂
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem sorted_nil : Sorted r [] :=
  Pairwise.nil


theorem Sorted.of_cons : Sorted r (a :: l) → Sorted r l :=
  Pairwise.of_cons


theorem Sorted.tail {r : α → α → Prop} {l : List α} (h : Sorted r l) : Sorted r l.tail :=
  Pairwise.tail h


theorem rel_of_sorted_cons {a : α} {l : List α} : Sorted r (a :: l) → ∀ b ∈ l, r a b :=
  rel_of_pairwise_cons


nonrec theorem Sorted.cons {r : α → α → Prop} [IsTrans α r] {l : List α} {a b : α}
    (hab : r a b) (h : Sorted r (b :: l)) : Sorted r (a :: b :: l) :=
  h.cons <| forall_mem_cons.2 ⟨hab, fun _ hx => _root_.trans hab <| rel_of_sorted_cons h _ hx⟩


theorem sorted_cons_cons {r : α → α → Prop} [IsTrans α r] {l : List α} {a b : α} :
    Sorted r (b :: a :: l) ↔ r b a ∧ Sorted r (a :: l) := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsTrans α r
    l : List α
    a b : α
    ⊢ Iff (List.Sorted r (List.cons b (List.cons a l))) (And (r b a) (List.Sorted  …
  -/
  constructor
    /-
      case mp
      α : Type u
      r : α → α → Prop
      inst✝ : IsTrans α r
      l : List α
      a b : α
      ⊢ List.Sorted r (List.cons b (List.cons a l)) → And (r b a) (List.Sorted r (Li …
    -/
  · intro h
    /-
      case mp
      α : Type u
      r : α → α → Prop
      inst✝ : IsTrans α r
      l : List α
      a b : α
      h : List.Sorted r (List.cons b (List.cons a l))
      ⊢ And (r b a) (List.Sorted r (List.cons a l))
    -/
    exact ⟨rel_of_sorted_cons h _ (mem_cons_self a _), h.of_cons⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      r : α → α → Prop
      inst✝ : IsTrans α r
      l : List α
      a b : α
      ⊢ And (r b a) (List.Sorted r (List.cons a l)) → List.Sorted r (List.cons b (Li …
    -/
  · rintro ⟨h, ha⟩
    /-
      case mpr.intro
      α : Type u
      r : α → α → Prop
      inst✝ : IsTrans α r
      l : List α
      a b : α
      h : r b a
      ha : List.Sorted r (List.cons a l)
      ⊢ List.Sorted r (List.cons b (List.cons a l))
    -/
    exact ha.cons h
    /-
      🎉 no goals
    -/


theorem Sorted.head!_le [Inhabited α] [Preorder α] {a : α} {l : List α} (h : Sorted (· < ·) l)
    (ha : a ∈ l) : l.head! ≤ a := by
  /-
    α : Type u
    inst✝¹ : Inhabited α
    inst✝ : Preorder α
    a : α
    l : List α
    h : List.Sorted (fun x1 x2 => LT.lt x1 x2) l
    ha : Membership.mem l a
    ⊢ LE.le l.head! a
  -/
  rw [← List.cons_head!_tail (List.ne_nil_of_mem ha)] at h ha
  /-
    α : Type u
    inst✝¹ : Inhabited α
    inst✝ : Preorder α
    a : α
    l : List α
    h : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.cons l.head! l.tail)
    ha : Membership.mem (List.cons l.head! l.tail) a
    ⊢ LE.le l.head! a
  -/
  cases ha
    /-
      case head
      α : Type u
      inst✝¹ : Inhabited α
      inst✝ : Preorder α
      l : List α
      h : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.cons l.head! l.tail)
      ⊢ LE.le l.head! l.head!
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case tail
      α : Type u
      inst✝¹ : Inhabited α
      inst✝ : Preorder α
      a : α
      l : List α
      h : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.cons l.head! l.tail)
      a✝ : List.Mem a l.tail
      ⊢ LE.le l.head! a
    -/
  · exact le_of_lt (rel_of_sorted_cons h a (by assumption))
    /-
      🎉 no goals
    -/


theorem Sorted.le_head! [Inhabited α] [Preorder α] {a : α} {l : List α} (h : Sorted (· > ·) l)
    (ha : a ∈ l) : a ≤ l.head! := by
  /-
    α : Type u
    inst✝¹ : Inhabited α
    inst✝ : Preorder α
    a : α
    l : List α
    h : List.Sorted (fun x1 x2 => GT.gt x1 x2) l
    ha : Membership.mem l a
    ⊢ LE.le a l.head!
  -/
  rw [← List.cons_head!_tail (List.ne_nil_of_mem ha)] at h ha
  /-
    α : Type u
    inst✝¹ : Inhabited α
    inst✝ : Preorder α
    a : α
    l : List α
    h : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.cons l.head! l.tail)
    ha : Membership.mem (List.cons l.head! l.tail) a
    ⊢ LE.le a l.head!
  -/
  cases ha
    /-
      case head
      α : Type u
      inst✝¹ : Inhabited α
      inst✝ : Preorder α
      l : List α
      h : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.cons l.head! l.tail)
      ⊢ LE.le l.head! l.head!
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case tail
      α : Type u
      inst✝¹ : Inhabited α
      inst✝ : Preorder α
      a : α
      l : List α
      h : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.cons l.head! l.tail)
      a✝ : List.Mem a l.tail
      ⊢ LE.le a l.head!
    -/
  · exact le_of_lt (rel_of_sorted_cons h a (by assumption))
    /-
      🎉 no goals
    -/


@[simp]
theorem sorted_cons {a : α} {l : List α} : Sorted r (a :: l) ↔ (∀ b ∈ l, r a b) ∧ Sorted r l :=
  pairwise_cons


protected theorem Sorted.nodup {r : α → α → Prop} [IsIrrefl α r] {l : List α} (h : Sorted r l) :
    Nodup l :=
  Pairwise.nodup h


protected theorem Sorted.filter {l : List α} (f : α → Bool) (h : Sorted r l) :
    Sorted r (filter f l) :=
  h.sublist (filter_sublist l)


theorem eq_of_perm_of_sorted [IsAntisymm α r] {l₁ l₂ : List α} (hp : l₁ ~ l₂) (hs₁ : Sorted r l₁)
    (hs₂ : Sorted r l₂) : l₁ = l₂ := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsAntisymm α r
    l₁ l₂ : List α
    hp : l₁.Perm l₂
    hs₁ : List.Sorted r l₁
    hs₂ : List.Sorted r l₂
    ⊢ Eq l₁ l₂
  -/
  induction' hs₁ with a l₁ h₁ hs₁ IH generalizing l₂
    /-
      case nil
      α : Type u
      r : α → α → Prop
      inst✝ : IsAntisymm α r
      l₁ l₂ : List α
      hp : List.nil.Perm l₂
      hs₂ : List.Sorted r l₂
      ⊢ Eq List.nil l₂
    -/
  · exact hp.nil_eq
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      r : α → α → Prop
      inst✝ : IsAntisymm α r
      l₁✝ : List α
      a : α
      l₁ : List α
      h₁ : ∀ (a' : α), Membership.mem l₁ a' → r a a'
      hs₁ : List.Pairwise r l₁
      IH : ∀ {l₂ : List α}, l₁.Perm l₂ → List.Sorted r l₂ → Eq l₁ l₂
      l₂ : List α
      hp : (List.cons a l₁).Perm l₂
      hs₂ : List.Sorted r l₂
      ⊢ Eq (List.cons a l₁) l₂
    -/
  · have : a ∈ l₂ := hp.subset (mem_cons_self _ _)
    /-
      case cons
      α : Type u
      r : α → α → Prop
      inst✝ : IsAntisymm α r
      l₁✝ : List α
      a : α
      l₁ : List α
      h₁ : ∀ (a' : α), Membership.mem l₁ a' → r a a'
      hs₁ : List.Pairwise r l₁
      IH : ∀ {l₂ : List α}, l₁.Perm l₂ → List.Sorted r l₂ → Eq l₁ l₂
      l₂ : List α
      hp : (List.cons a l₁).Perm l₂
      hs₂ : List.Sorted r l₂
      this : Membership.mem l₂ a
      ⊢ Eq (List.cons a l₁) l₂
    -/
    rcases append_of_mem this with ⟨u₂, v₂, rfl⟩
    /-
      case cons.intro.intro
      α : Type u
      r : α → α → Prop
      inst✝ : IsAntisymm α r
      l₁✝ : List α
      a : α
      l₁ : List α
      h₁ : ∀ (a' : α), Membership.mem l₁ a' → r a a'
      hs₁ : List.Pairwise r l₁
      IH : ∀ {l₂ : List α}, l₁.Perm l₂ → List.Sorted r l₂ → Eq l₁ l₂
      u₂ v₂ : List α
      hp : (List.cons a l₁).Perm (HAppend.hAppend u₂ (List.cons a v₂))
      hs₂ : List.Sorted r (HAppend.hAppend u₂ (List.cons a v₂))
      this : Membership.mem (HAppend.hAppend u₂ (List.cons a v₂)) a
      ⊢ Eq (List.cons a l₁) (HAppend.hAppend u₂ (List.cons a v₂))
    -/
    have hp' := (perm_cons a).1 (hp.trans perm_middle)
    /-
      case cons.intro.intro
      α : Type u
      r : α → α → Prop
      inst✝ : IsAntisymm α r
      l₁✝ : List α
      a : α
      l₁ : List α
      h₁ : ∀ (a' : α), Membership.mem l₁ a' → r a a'
      hs₁ : List.Pairwise r l₁
      IH : ∀ {l₂ : List α}, l₁.Perm l₂ → List.Sorted r l₂ → Eq l₁ l₂
      u₂ v₂ : List α
      hp : (List.cons a l₁).Perm (HAppend.hAppend u₂ (List.cons a v₂))
      hs₂ : List.Sorted r (HAppend.hAppend u₂ (List.cons a v₂))
      this : Membership.mem (HAppend.hAppend u₂ (List.cons a v₂)) a
      hp' : l₁.Perm (HAppend.hAppend u₂ v₂)
      ⊢ Eq (List.cons a l₁) (HAppend.hAppend u₂ (List.cons a v₂))
    -/
    obtain rfl := IH hp' (hs₂.sublist <| by simp)
    /-
      case cons.intro.intro
      α : Type u
      r : α → α → Prop
      inst✝ : IsAntisymm α r
      l₁ : List α
      a : α
      u₂ v₂ : List α
      hs₂ : List.Sorted r (HAppend.hAppend u₂ (List.cons a v₂))
      this : Membership.mem (HAppend.hAppend u₂ (List.cons a v₂)) a
      h₁ : ∀ (a' : α), Membership.mem (HAppend.hAppend u₂ v₂) a' → r a a'
      hs₁ : List.Pairwise r (HAppend.hAppend u₂ v₂)
      IH : ∀ {l₂ : List α}, (HAppend.hAppend u₂ v₂).Perm l₂ → List.Sorted r l₂ → Eq  …
      hp : (List.cons a (HAppend.hAppend u₂ v₂)).Perm (HAppend.hAppend u₂ (List.cons …
      hp' : (HAppend.hAppend u₂ v₂).Perm (HAppend.hAppend u₂ v₂)
      ⊢ Eq (List.cons a (HAppend.hAppend u₂ v₂)) (HAppend.hAppend u₂ (List.cons a v₂))
    -/
    change a :: u₂ ++ v₂ = u₂ ++ ([a] ++ v₂)
    /-
      case cons.intro.intro
      α : Type u
      r : α → α → Prop
      inst✝ : IsAntisymm α r
      l₁ : List α
      a : α
      u₂ v₂ : List α
      hs₂ : List.Sorted r (HAppend.hAppend u₂ (List.cons a v₂))
      this : Membership.mem (HAppend.hAppend u₂ (List.cons a v₂)) a
      h₁ : ∀ (a' : α), Membership.mem (HAppend.hAppend u₂ v₂) a' → r a a'
      hs₁ : List.Pairwise r (HAppend.hAppend u₂ v₂)
      IH : ∀ {l₂ : List α}, (HAppend.hAppend u₂ v₂).Perm l₂ → List.Sorted r l₂ → Eq  …
      hp : (List.cons a (HAppend.hAppend u₂ v₂)).Perm (HAppend.hAppend u₂ (List.cons …
      hp' : (HAppend.hAppend u₂ v₂).Perm (HAppend.hAppend u₂ v₂)
      ⊢ Eq (HAppend.hAppend (List.cons a u₂) v₂) (HAppend.hAppend u₂ (HAppend.hAppen …
    -/
    rw [← append_assoc]
    /-
      case cons.intro.intro
      α : Type u
      r : α → α → Prop
      inst✝ : IsAntisymm α r
      l₁ : List α
      a : α
      u₂ v₂ : List α
      hs₂ : List.Sorted r (HAppend.hAppend u₂ (List.cons a v₂))
      this : Membership.mem (HAppend.hAppend u₂ (List.cons a v₂)) a
      h₁ : ∀ (a' : α), Membership.mem (HAppend.hAppend u₂ v₂) a' → r a a'
      hs₁ : List.Pairwise r (HAppend.hAppend u₂ v₂)
      IH : ∀ {l₂ : List α}, (HAppend.hAppend u₂ v₂).Perm l₂ → List.Sorted r l₂ → Eq  …
      hp : (List.cons a (HAppend.hAppend u₂ v₂)).Perm (HAppend.hAppend u₂ (List.cons …
      hp' : (HAppend.hAppend u₂ v₂).Perm (HAppend.hAppend u₂ v₂)
      ⊢ Eq (HAppend.hAppend (List.cons a u₂) v₂) (HAppend.hAppend (HAppend.hAppend u …
    -/
    congr
    have : ∀ x ∈ u₂, x = a := fun x m =>
      antisymm ((pairwise_append.1 hs₂).2.2 _ m a (mem_cons_self _ _)) (h₁ _ (by simp [m]))
    rw [(@eq_replicate_iff _ a (length u₂ + 1) (a :: u₂)).2,
        (@eq_replicate_iff _ a (length u₂ + 1) (u₂ ++ [a])).2] <;>
        /-
          case cons.intro.intro.e_a
          α : Type u
          r : α → α → Prop
          inst✝ : IsAntisymm α r
          l₁ : List α
          a : α
          u₂ v₂ : List α
          hs₂ : List.Sorted r (HAppend.hAppend u₂ (List.cons a v₂))
          this✝ : Membership.mem (HAppend.hAppend u₂ (List.cons a v₂)) a
          h₁ : ∀ (a' : α), Membership.mem (HAppend.hAppend u₂ v₂) a' → r a a'
          hs₁ : List.Pairwise r (HAppend.hAppend u₂ v₂)
          IH : ∀ {l₂ : List α}, (HAppend.hAppend u₂ v₂).Perm l₂ → List.Sorted r l₂ → Eq  …
          hp : (List.cons a (HAppend.hAppend u₂ v₂)).Perm (HAppend.hAppend u₂ (List.cons …
          hp' : (HAppend.hAppend u₂ v₂).Perm (HAppend.hAppend u₂ v₂)
          this : ∀ (x : α), Membership.mem u₂ x → Eq x a
          ⊢ And (Eq (HAppend.hAppend u₂ (List.cons a List.nil)).length (HAdd.hAdd u₂.len …
        -/
        constructor <;>
      /-
        case cons.intro.intro.e_a.left
        α : Type u
        r : α → α → Prop
        inst✝ : IsAntisymm α r
        l₁ : List α
        a : α
        u₂ v₂ : List α
        hs₂ : List.Sorted r (HAppend.hAppend u₂ (List.cons a v₂))
        this✝ : Membership.mem (HAppend.hAppend u₂ (List.cons a v₂)) a
        h₁ : ∀ (a' : α), Membership.mem (HAppend.hAppend u₂ v₂) a' → r a a'
        hs₁ : List.Pairwise r (HAppend.hAppend u₂ v₂)
        IH : ∀ {l₂ : List α}, (HAppend.hAppend u₂ v₂).Perm l₂ → List.Sorted r l₂ → Eq  …
        hp : (List.cons a (HAppend.hAppend u₂ v₂)).Perm (HAppend.hAppend u₂ (List.cons …
        hp' : (HAppend.hAppend u₂ v₂).Perm (HAppend.hAppend u₂ v₂)
        this : ∀ (x : α), Membership.mem u₂ x → Eq x a
        ⊢ Eq (HAppend.hAppend u₂ (List.cons a List.nil)).length (HAdd.hAdd u₂.length 1)
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
      simp [iff_true_intro this, or_comm]
      /-
        🎉 no goals
      -/


theorem sublist_of_subperm_of_sorted [IsAntisymm α r] {l₁ l₂ : List α} (hp : l₁ <+~ l₂)
    (hs₁ : l₁.Sorted r) (hs₂ : l₂.Sorted r) : l₁ <+ l₂ := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsAntisymm α r
    l₁ l₂ : List α
    hp : l₁.Subperm l₂
    hs₁ : List.Sorted r l₁
    hs₂ : List.Sorted r l₂
    ⊢ l₁.Sublist l₂
  -/
  let ⟨_, h, h'⟩ := hp
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsAntisymm α r
    l₁ l₂ : List α
    hp : l₁.Subperm l₂
    hs₁ : List.Sorted r l₁
    hs₂ : List.Sorted r l₂
    w✝ : List α
    h : w✝.Perm l₁
    h' : w✝.Sublist l₂
    ⊢ l₁.Sublist l₂
  -/
  rwa [← eq_of_perm_of_sorted h (hs₂.sublist h') hs₁]
  /-
    🎉 no goals
  -/


@[simp 1100] -- Porting note: higher priority for linter
theorem sorted_singleton (a : α) : Sorted r [a] :=
  pairwise_singleton _ _


theorem sorted_lt_range (n : ℕ) : Sorted (· < ·) (range n) := by
  /-
    n : Nat
    ⊢ List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.range n)
  -/
  rw [Sorted, pairwise_iff_get]
  /-
    n : Nat
    ⊢ ∀ (i j : Fin (List.range n).length), LT.lt i j → LT.lt ((List.range n).get i …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem sorted_le_range (n : ℕ) : Sorted (· ≤ ·) (range n) :=
  (sorted_lt_range n).le_of_lt


theorem Sorted.rel_get_of_lt {l : List α} (h : l.Sorted r) {a b : Fin l.length} (hab : a < b) :
    r (l.get a) (l.get b) :=
  List.pairwise_iff_get.1 h _ _ hab


theorem Sorted.rel_get_of_le [IsRefl α r] {l : List α} (h : l.Sorted r) {a b : Fin l.length}
    (hab : a ≤ b) : r (l.get a) (l.get b) := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsRefl α r
    l : List α
    h : List.Sorted r l
    a b : Fin l.length
    hab : LE.le a b
    ⊢ r (l.get a) (l.get b)
  -/
  obtain rfl | hlt := Fin.eq_or_lt_of_le hab; exacts [refl _, h.rel_get_of_lt hlt]
                                              /-
                                                🎉 no goals
                                              -/


theorem Sorted.rel_of_mem_take_of_mem_drop {l : List α} (h : List.Sorted r l) {k : ℕ} {x y : α}
    (hx : x ∈ List.take k l) (hy : y ∈ List.drop k l) : r x y := by
  /-
    α : Type u
    r : α → α → Prop
    l : List α
    h : List.Sorted r l
    k : Nat
    x y : α
    hx : Membership.mem (List.take k l) x
    hy : Membership.mem (List.drop k l) y
    ⊢ r x y
  -/
  obtain ⟨iy, hiy, rfl⟩ := getElem_of_mem hy
  /-
    case intro.intro
    α : Type u
    r : α → α → Prop
    l : List α
    h : List.Sorted r l
    k : Nat
    x : α
    hx : Membership.mem (List.take k l) x
    iy : Nat
    hiy : LT.lt iy (List.drop k l).length
    hy : Membership.mem (List.drop k l) (GetElem.getElem (List.drop k l) iy hiy)
    ⊢ r x (GetElem.getElem (List.drop k l) iy hiy)
  -/
  obtain ⟨ix, hix, rfl⟩ := getElem_of_mem hx
  /-
    case intro.intro.intro.intro
    α : Type u
    r : α → α → Prop
    l : List α
    h : List.Sorted r l
    k iy : Nat
    hiy : LT.lt iy (List.drop k l).length
    hy : Membership.mem (List.drop k l) (GetElem.getElem (List.drop k l) iy hiy)
    ix : Nat
    hix : LT.lt ix (List.take k l).length
    hx : Membership.mem (List.take k l) (GetElem.getElem (List.take k l) ix hix)
    ⊢ r (GetElem.getElem (List.take k l) ix hix) (GetElem.getElem (List.drop k l)  …
  -/
  rw [getElem_take, getElem_drop]
  /-
    case intro.intro.intro.intro
    α : Type u
    r : α → α → Prop
    l : List α
    h : List.Sorted r l
    k iy : Nat
    hiy : LT.lt iy (List.drop k l).length
    hy : Membership.mem (List.drop k l) (GetElem.getElem (List.drop k l) iy hiy)
    ix : Nat
    hix : LT.lt ix (List.take k l).length
    hx : Membership.mem (List.take k l) (GetElem.getElem (List.take k l) ix hix)
    ⊢ r (GetElem.getElem l ix ⋯) (GetElem.getElem l (HAdd.hAdd k iy) ⋯)
  -/
  rw [length_take] at hix
  /-
    case intro.intro.intro.intro
    α : Type u
    r : α → α → Prop
    l : List α
    h : List.Sorted r l
    k iy : Nat
    hiy : LT.lt iy (List.drop k l).length
    hy : Membership.mem (List.drop k l) (GetElem.getElem (List.drop k l) iy hiy)
    ix : Nat
    hix✝ : LT.lt ix (List.take k l).length
    hix : LT.lt ix (Min.min k l.length)
    hx : Membership.mem (List.take k l) (GetElem.getElem (List.take k l) ix hix✝)
    ⊢ r (GetElem.getElem l ix ⋯) (GetElem.getElem l (HAdd.hAdd k iy) ⋯)
  -/
  exact h.rel_get_of_lt (Nat.lt_add_right _ (Nat.lt_min.mp hix).left)
  /-
    🎉 no goals
  -/


/--
If a list is sorted with respect to a decidable relation,
then it is sorted with respect to the corresponding Bool-valued relation.
-/
theorem Sorted.decide [DecidableRel r] (l : List α) (h : Sorted r l) :
    Sorted (fun a b => decide (r a b) = true) l := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : DecidableRel r
    l : List α
    h : List.Sorted r l
    ⊢ List.Sorted (fun a b => Eq (Decidable.decide (r a b)) Bool.true) l
  -/
  refine h.imp fun {a b} h => by simpa using h
  /-
    🎉 no goals
  -/


theorem sorted_ofFn_iff {r : α → α → Prop} : (ofFn f).Sorted r ↔ ((· < ·) ⇒ r) f f := by
  /-
    n : Nat
    α : Type u
    f : Fin n → α
    r : α → α → Prop
    ⊢ Iff (List.Sorted r (List.ofFn f)) (Relator.LiftFun (fun x1 x2 => LT.lt x1 x2 …
  -/
  simp_rw [Sorted, pairwise_iff_get, get_ofFn, Relator.LiftFun]
  /-
    n : Nat
    α : Type u
    f : Fin n → α
    r : α → α → Prop
    ⊢ Iff (∀ (i j : Fin (List.ofFn f).length), LT.lt i j → r (f (Fin.cast ⋯ i)) (f …
  -/
  exact Iff.symm (Fin.rightInverse_cast _).surjective.forall₂
  /-
    🎉 no goals
  -/


/-- The list `List.ofFn f` is strictly sorted with respect to `(· ≤ ·)` if and only if `f` is
strictly monotone. -/
@[simp] theorem sorted_lt_ofFn_iff : (ofFn f).Sorted (· < ·) ↔ StrictMono f := sorted_ofFn_iff


/-- The list `List.ofFn f` is sorted with respect to `(· ≤ ·)` if and only if `f` is monotone. -/
@[simp] theorem sorted_le_ofFn_iff : (ofFn f).Sorted (· ≤ ·) ↔ Monotone f :=
  sorted_ofFn_iff.trans monotone_iff_forall_lt.symm


/-- The list obtained from a monotone tuple is sorted. -/
alias ⟨_, _root_.Monotone.ofFn_sorted⟩ := sorted_le_ofFn_iff


local infixl:50 " ≼ " => r

local infixl:50 " ≼ " => s


/-- `orderedInsert a l` inserts `a` into `l` at such that
  `orderedInsert a l` is sorted if `l` is. -/
@[simp]
def orderedInsert (a : α) : List α → List α
  | [] => [a]
  | b :: l => if a ≼ b then a :: b :: l else b :: orderedInsert a l


theorem orderedInsert_of_le {a b : α} (l : List α) (h : a ≼ b) :
    orderedInsert r a (b :: l) = a :: b :: l :=
  dif_pos h


/-- `insertionSort l` returns `l` sorted using the insertion sort algorithm. -/
@[simp]
def insertionSort : List α → List α
  | [] => []
  | b :: l => orderedInsert r b (insertionSort l)

-- A quick check that insertionSort is stable:

@[simp]
theorem orderedInsert_nil (a : α) : [].orderedInsert r a = [a] :=
  rfl


theorem orderedInsert_length : ∀ (L : List α) (a : α), (L.orderedInsert r a).length = L.length + 1
  | [], _ => rfl
  | hd :: tl, a => by
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : DecidableRel r
      hd : α
      tl : List α
      a : α
      ⊢ Eq (List.orderedInsert r a (List.cons hd tl)).length (HAdd.hAdd (List.cons h …
    -/
    dsimp [orderedInsert]
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : DecidableRel r
      hd : α
      tl : List α
      a : α
      ⊢ Eq (ite (r a hd) (List.cons a (List.cons hd tl)) (List.cons hd (List.ordered …
    -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> simp [orderedInsert_length tl]
                  /-
                    🎉 no goals
                  -/


/-- An alternative definition of `orderedInsert` using `takeWhile` and `dropWhile`. -/
theorem orderedInsert_eq_take_drop (a : α) :
    ∀ l : List α,
      l.orderedInsert r a = (l.takeWhile fun b => ¬a ≼ b) ++ a :: l.dropWhile fun b => ¬a ≼ b
  | [] => rfl
  | b :: l => by
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : DecidableRel r
      a b : α
      l : List α
      ⊢ Eq (List.orderedInsert r a (List.cons b l)) (HAppend.hAppend (List.takeWhile …
    -/
    dsimp only [orderedInsert]
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : DecidableRel r
      a b : α
      l : List α
      ⊢ Eq (ite (r a b) (List.cons a (List.cons b l)) (List.cons b (List.orderedInse …
    -/
                         /-
                           🎉 no goals
                         -/
    split_ifs with h <;> simp [takeWhile, dropWhile, *, orderedInsert_eq_take_drop a l]
                         /-
                           🎉 no goals
                         -/


theorem insertionSort_cons_eq_take_drop (a : α) (l : List α) :
    insertionSort r (a :: l) =
      ((insertionSort r l).takeWhile fun b => ¬a ≼ b) ++
        a :: (insertionSort r l).dropWhile fun b => ¬a ≼ b :=
  orderedInsert_eq_take_drop r a _


@[simp]
theorem mem_orderedInsert {a b : α} {l : List α} :
    a ∈ orderedInsert r b l ↔ a = b ∨ a ∈ l :=
  match l with
             /-
               α : Type u
               r : α → α → Prop
               inst✝ : DecidableRel r
               a b : α
               l : List α
               ⊢ Iff (Membership.mem (List.orderedInsert r b List.nil) a) (Or (Eq a b) (Membe …
             -/
  | [] => by simp [orderedInsert]
             /-
               🎉 no goals
             -/
  | x :: xs => by
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : DecidableRel r
      a b : α
      l : List α
      x : α
      xs : List α
      ⊢ Iff (Membership.mem (List.orderedInsert r b (List.cons x xs)) a) (Or (Eq a b …
    -/
    rw [orderedInsert]
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : DecidableRel r
      a b : α
      l : List α
      x : α
      xs : List α
      ⊢ Iff (Membership.mem (ite (r b x) (List.cons b (List.cons x xs)) (List.cons x …
    -/
    split_ifs
      /-
        case pos
        α : Type u
        r : α → α → Prop
        inst✝ : DecidableRel r
        a b : α
        l : List α
        x : α
        xs : List α
        h✝ : r b x
        ⊢ Iff (Membership.mem (List.cons b (List.cons x xs)) a) (Or (Eq a b) (Membersh …
      -/
    · simp [orderedInsert]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        r : α → α → Prop
        inst✝ : DecidableRel r
        a b : α
        l : List α
        x : α
        xs : List α
        h✝ : Not (r b x)
        ⊢ Iff (Membership.mem (List.cons x (List.orderedInsert r b xs)) a) (Or (Eq a b …
      -/
    · rw [mem_cons, mem_cons, mem_orderedInsert, or_left_comm]
      /-
        🎉 no goals
      -/


theorem map_orderedInsert (f : α → β) (l : List α) (x : α)
    (hl₁ : ∀ a ∈ l, a ≼ x ↔ f a ≼ f x) (hl₂ : ∀ a ∈ l, x ≼ a ↔ f x ≼ f a) :
    (l.orderedInsert r x).map f = (l.map f).orderedInsert s (f x) := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    rw [List.forall_mem_cons] at hl₁ hl₂
    simp only [List.map, List.orderedInsert, ← hl₁.1, ← hl₂.1]
    split_ifs
    · rw [List.map, List.map]
    · rw [List.map, ih (fun _ ha => hl₁.2 _ ha) (fun _ ha => hl₂.2 _ ha)]


theorem perm_orderedInsert (a) : ∀ l : List α, orderedInsert r a l ~ a :: l
  | [] => Perm.refl _
  | b :: l => by
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : DecidableRel r
      a b : α
      l : List α
      ⊢ (List.orderedInsert r a (List.cons b l)).Perm (List.cons a (List.cons b l))
    -/
    by_cases h : a ≼ b
      /-
        case pos
        α : Type u
        r : α → α → Prop
        inst✝ : DecidableRel r
        a b : α
        l : List α
        h : r a b
        ⊢ (List.orderedInsert r a (List.cons b l)).Perm (List.cons a (List.cons b l))
      -/
    · simp [orderedInsert, h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        r : α → α → Prop
        inst✝ : DecidableRel r
        a b : α
        l : List α
        h : Not (r a b)
        ⊢ (List.orderedInsert r a (List.cons b l)).Perm (List.cons a (List.cons b l))
      -/
    · simpa [orderedInsert, h] using ((perm_orderedInsert a l).cons _).trans (Perm.swap _ _ _)
      /-
        🎉 no goals
      -/


theorem orderedInsert_count [DecidableEq α] (L : List α) (a b : α) :
    count a (L.orderedInsert r b) = count a L + if b = a then 1 else 0 := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝¹ : DecidableRel r
    inst✝ : DecidableEq α
    L : List α
    a b : α
    ⊢ Eq (List.count a (List.orderedInsert r b L)) (HAdd.hAdd (List.count a L) (it …
  -/
  rw [(L.perm_orderedInsert r b).count_eq, count_cons]
  /-
    α : Type u
    r : α → α → Prop
    inst✝¹ : DecidableRel r
    inst✝ : DecidableEq α
    L : List α
    a b : α
    ⊢ Eq (HAdd.hAdd (List.count a L) (ite (Eq (BEq.beq b a) Bool.true) 1 0)) (HAdd …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem perm_insertionSort : ∀ l : List α, insertionSort r l ~ l
  | [] => Perm.nil
  | b :: l => by
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : DecidableRel r
      b : α
      l : List α
      ⊢ (List.insertionSort r (List.cons b l)).Perm (List.cons b l)
    -/
    simpa [insertionSort] using (perm_orderedInsert _ _ _).trans ((perm_insertionSort l).cons b)
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_insertionSort {l : List α} {x : α} : x ∈ l.insertionSort r ↔ x ∈ l :=
  (perm_insertionSort r l).mem_iff


@[simp]
theorem length_insertionSort (l : List α) : (insertionSort r l).length = l.length :=
  (perm_insertionSort r _).length_eq


theorem insertionSort_cons {a : α} {l : List α} (h : ∀ b ∈ l, r a b) :
    insertionSort r (a :: l) = a :: insertionSort r l := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : DecidableRel r
    a : α
    l : List α
    h : ∀ (b : α), Membership.mem l b → r a b
    ⊢ Eq (List.insertionSort r (List.cons a l)) (List.cons a (List.insertionSort r …
  -/
  rw [insertionSort]
  cases hi : insertionSort r l with
  | nil => rfl
  | cons b m =>
    rw [orderedInsert_of_le]
    apply h b <| (mem_insertionSort r).1 _
    rw [hi]
    exact mem_cons_self b m


theorem map_insertionSort (f : α → β) (l : List α) (hl : ∀ a ∈ l, ∀ b ∈ l, a ≼ b ↔ f a ≼ f b) :
    (l.insertionSort r).map f = (l.map f).insertionSort s := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp_rw [List.forall_mem_cons, forall_and] at hl
    simp_rw [List.map, List.insertionSort]
    rw [List.map_orderedInsert _ s, ih hl.2.2]
    · simpa only [mem_insertionSort] using hl.2.1
    · simpa only [mem_insertionSort] using hl.1.2


/-- If `l` is already `List.Sorted` with respect to `r`, then `insertionSort` does not change
it. -/
theorem Sorted.insertionSort_eq : ∀ {l : List α}, Sorted r l → insertionSort r l = l
  | [], _ => rfl
  | [_], _ => rfl
  | a :: b :: l, h => by
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : DecidableRel r
      a b : α
      l : List α
      h : List.Sorted r (List.cons a (List.cons b l))
      ⊢ Eq (List.insertionSort r (List.cons a (List.cons b l))) (List.cons a (List.c …
    -/
    rw [insertionSort, Sorted.insertionSort_eq, orderedInsert, if_pos]
    /-
      case hc
      α : Type u
      r : α → α → Prop
      inst✝ : DecidableRel r
      a b : α
      l : List α
      h : List.Sorted r (List.cons a (List.cons b l))
      ⊢ r a b
    -/
    exacts [rel_of_sorted_cons h _ (mem_cons_self _ _), h.tail]
    /-
      🎉 no goals
    -/


/-- For a reflexive relation, insert then erasing is the identity. -/
theorem erase_orderedInsert [DecidableEq α] [IsRefl α r] (x : α) (xs : List α) :
    (xs.orderedInsert r x).erase x = xs := by
  rw [orderedInsert_eq_take_drop, erase_append_right, List.erase_cons_head,
    takeWhile_append_dropWhile]
  /-
    case h
    α : Type u
    r : α → α → Prop
    inst✝² : DecidableRel r
    inst✝¹ : DecidableEq α
    inst✝ : IsRefl α r
    x : α
    xs : List α
    ⊢ Not (Membership.mem (List.takeWhile (fun b => Decidable.decide (Not (r x b)) …
  -/
  intro h
  /-
    case h
    α : Type u
    r : α → α → Prop
    inst✝² : DecidableRel r
    inst✝¹ : DecidableEq α
    inst✝ : IsRefl α r
    x : α
    xs : List α
    h : Membership.mem (List.takeWhile (fun b => Decidable.decide (Not (r x b))) x …
    ⊢ False
  -/
  replace h := mem_takeWhile_imp h
  /-
    case h
    α : Type u
    r : α → α → Prop
    inst✝² : DecidableRel r
    inst✝¹ : DecidableEq α
    inst✝ : IsRefl α r
    x : α
    xs : List α
    h : Eq (Decidable.decide (Not (r x x))) Bool.true
    ⊢ False
  -/
  simp [refl x] at h
  /-
    🎉 no goals
  -/


/-- Inserting then erasing an element that is absent is the identity. -/
theorem erase_orderedInsert_of_not_mem [DecidableEq α]
    {x : α} {xs : List α} (hx : x ∉ xs) :
    (xs.orderedInsert r x).erase x = xs := by
  rw [orderedInsert_eq_take_drop, erase_append_right, List.erase_cons_head,
    takeWhile_append_dropWhile]
  /-
    case h
    α : Type u
    r : α → α → Prop
    inst✝¹ : DecidableRel r
    inst✝ : DecidableEq α
    x : α
    xs : List α
    hx : Not (Membership.mem xs x)
    ⊢ Not (Membership.mem (List.takeWhile (fun b => Decidable.decide (Not (r x b)) …
  -/
  exact mt ((takeWhile_prefix _).sublist.subset ·) hx
  /-
    🎉 no goals
  -/


/-- For an antisymmetric relation, erasing then inserting is the identity. -/
theorem orderedInsert_erase [DecidableEq α] [IsAntisymm α r] (x : α) (xs : List α) (hx : x ∈ xs)
    (hxs : Sorted r xs) :
    (xs.erase x).orderedInsert r x = xs := by
  induction xs generalizing x with
  | nil => cases hx
  | cons y ys ih =>
    rw [sorted_cons] at hxs
    obtain rfl | hxy := Decidable.eq_or_ne x y
    · rw [erase_cons_head]
      cases ys with
      | nil => rfl
      | cons z zs =>
        rw [orderedInsert, if_pos (hxs.1 _ (.head zs))]
    · rw [mem_cons] at hx
      replace hx := hx.resolve_left hxy
      rw [erase_cons_tail (not_beq_of_ne hxy.symm), orderedInsert, ih _ hx hxs.2, if_neg]
      refine mt (fun hrxy => ?_) hxy
      exact antisymm hrxy (hxs.1 _ hx)


theorem sublist_orderedInsert (x : α) (xs : List α) : xs <+ xs.orderedInsert r x := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : DecidableRel r
    x : α
    xs : List α
    ⊢ xs.Sublist (List.orderedInsert r x xs)
  -/
  rw [orderedInsert_eq_take_drop]
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : DecidableRel r
    x : α
    xs : List α
    ⊢ xs.Sublist (HAppend.hAppend (List.takeWhile (fun b => Decidable.decide (Not  …
  -/
  refine Sublist.trans ?_ (.append_left (.cons _ (.refl _)) _)
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : DecidableRel r
    x : α
    xs : List α
    ⊢ xs.Sublist (HAppend.hAppend (List.takeWhile (fun b => Decidable.decide (Not  …
  -/
  rw [takeWhile_append_dropWhile]
  /-
    🎉 no goals
  -/


theorem cons_sublist_orderedInsert {l c : List α} {a : α} (hl : c <+ l) (ha : ∀ a' ∈ c, a ≼ a') :
    a :: c <+ orderedInsert r a l := by
  induction l with
  | nil         => simp_all only [sublist_nil, orderedInsert, Sublist.refl]
  | cons _ _ ih =>
    unfold orderedInsert
    split_ifs with hr
    · exact .cons₂ _ hl
    · cases hl with
      | cons _ h => exact .cons _ <| ih h
      | cons₂    => exact absurd (ha _ <| mem_cons_self ..) hr


theorem Sublist.orderedInsert_sublist [IsTrans α r] {as bs} (x) (hs : as <+ bs) (hb : bs.Sorted r) :
    orderedInsert r x as <+ orderedInsert r x bs := by
  cases as with
  | nil => simp
  | cons a as =>
    cases bs with
    | nil => contradiction
    | cons b bs =>
      unfold orderedInsert
      cases hs <;> split_ifs with hr
      · exact .cons₂ _ <| .cons _ ‹a :: as <+ bs›
      · have ih := orderedInsert_sublist x ‹a :: as <+ bs›  hb.of_cons
        simp only [hr, orderedInsert, ite_true] at ih
        exact .trans ih <| .cons _ (.refl _)
      · have hba := pairwise_cons.mp hb |>.left _ (mem_of_cons_sublist ‹a :: as <+ bs›)
        exact absurd (trans_of _ ‹r x b› hba) hr
      · have ih := orderedInsert_sublist x ‹a :: as <+ bs› hb.of_cons
        rw [orderedInsert, if_neg hr] at ih
        exact .cons _ ih
      · simp_all only [sorted_cons, cons_sublist_cons]
      · exact .cons₂ _ <| orderedInsert_sublist x ‹as <+ bs› hb.of_cons


theorem Sorted.orderedInsert (a : α) : ∀ l, Sorted r l → Sorted r (orderedInsert r a l)
  | [], _ => sorted_singleton a
  | b :: l, h => by
    /-
      α : Type u
      r : α → α → Prop
      inst✝² : DecidableRel r
      inst✝¹ : IsTotal α r
      inst✝ : IsTrans α r
      a b : α
      l : List α
      h : List.Sorted r (List.cons b l)
      ⊢ List.Sorted r (List.orderedInsert r a (List.cons b l))
    -/
    by_cases h' : a ≼ b
    · -- Porting note: was
      -- `simpa [orderedInsert, h', h] using fun b' bm => trans h' (rel_of_sorted_cons h _ bm)`
      /-
        case pos
        α : Type u
        r : α → α → Prop
        inst✝² : DecidableRel r
        inst✝¹ : IsTotal α r
        inst✝ : IsTrans α r
        a b : α
        l : List α
        h : List.Sorted r (List.cons b l)
        h' : r a b
        ⊢ List.Sorted r (List.orderedInsert r a (List.cons b l))
      -/
      rw [List.orderedInsert, if_pos h', sorted_cons]
      /-
        case pos
        α : Type u
        r : α → α → Prop
        inst✝² : DecidableRel r
        inst✝¹ : IsTotal α r
        inst✝ : IsTrans α r
        a b : α
        l : List α
        h : List.Sorted r (List.cons b l)
        h' : r a b
        ⊢ And (∀ (b_1 : α), Membership.mem (List.cons b l) b_1 → r a b_1) (List.Sorted …
      -/
      exact ⟨forall_mem_cons.2 ⟨h', fun c hc => _root_.trans h' (rel_of_sorted_cons h _ hc)⟩, h⟩
      /-
        🎉 no goals
      -/
    · suffices ∀ b' : α, b' ∈ List.orderedInsert r a l → r b b' by
        simpa [orderedInsert, h', h.of_cons.orderedInsert a l]
      /-
        case neg
        α : Type u
        r : α → α → Prop
        inst✝² : DecidableRel r
        inst✝¹ : IsTotal α r
        inst✝ : IsTrans α r
        a b : α
        l : List α
        h : List.Sorted r (List.cons b l)
        h' : Not (r a b)
        ⊢ ∀ (b' : α), Membership.mem (List.orderedInsert r a l) b' → r b b'
      -/
      intro b' bm
      /-
        case neg
        α : Type u
        r : α → α → Prop
        inst✝² : DecidableRel r
        inst✝¹ : IsTotal α r
        inst✝ : IsTrans α r
        a b : α
        l : List α
        h : List.Sorted r (List.cons b l)
        h' : Not (r a b)
        b' : α
        bm : Membership.mem (List.orderedInsert r a l) b'
        ⊢ r b b'
      -/
      cases' (mem_orderedInsert r).mp bm with be bm
        /-
          case neg.inl
          α : Type u
          r : α → α → Prop
          inst✝² : DecidableRel r
          inst✝¹ : IsTotal α r
          inst✝ : IsTrans α r
          a b : α
          l : List α
          h : List.Sorted r (List.cons b l)
          h' : Not (r a b)
          b' : α
          bm : Membership.mem (List.orderedInsert r a l) b'
          be : Eq b' a
          ⊢ r b b'
        -/
      · subst b'
        /-
          case neg.inl
          α : Type u
          r : α → α → Prop
          inst✝² : DecidableRel r
          inst✝¹ : IsTotal α r
          inst✝ : IsTrans α r
          a b : α
          l : List α
          h : List.Sorted r (List.cons b l)
          h' : Not (r a b)
          bm : Membership.mem (List.orderedInsert r a l) a
          ⊢ r b a
        -/
        exact (total_of r _ _).resolve_left h'
        /-
          🎉 no goals
        -/
        /-
          case neg.inr
          α : Type u
          r : α → α → Prop
          inst✝² : DecidableRel r
          inst✝¹ : IsTotal α r
          inst✝ : IsTrans α r
          a b : α
          l : List α
          h : List.Sorted r (List.cons b l)
          h' : Not (r a b)
          b' : α
          bm✝ : Membership.mem (List.orderedInsert r a l) b'
          bm : Membership.mem l b'
          ⊢ r b b'
        -/
      · exact rel_of_sorted_cons h _ bm
        /-
          🎉 no goals
        -/


/-- The list `List.insertionSort r l` is `List.Sorted` with respect to `r`. -/
theorem sorted_insertionSort : ∀ l, Sorted r (insertionSort r l)
  | [] => sorted_nil
  | a :: l => (sorted_insertionSort l).orderedInsert a _


/--
If `c` is a sorted sublist of `l`, then `c` is still a sublist of `insertionSort r l`.
-/
theorem sublist_insertionSort {l c : List α} (hr : c.Pairwise r) (hc : c <+ l) :
    c <+ insertionSort r l := by
  induction l generalizing c with
  | nil         => simp_all only [sublist_nil, insertionSort, Sublist.refl]
  | cons _ _ ih =>
    cases hc with
    | cons  _ h => exact ih hr h |>.trans (sublist_orderedInsert ..)
    | cons₂ _ h =>
      obtain ⟨hr, hp⟩ := pairwise_cons.mp hr
      exact cons_sublist_orderedInsert (ih hp h) hr


/--
Another statement of stability of insertion sort.
If a pair `[a, b]` is a sublist of `l` and `r a b`,
then `[a, b]` is still a sublist of `insertionSort r l`.
-/
theorem pair_sublist_insertionSort {a b : α} {l : List α} (hab : r a b) (h : [a, b] <+ l) :
    [a, b] <+ insertionSort r l :=
  sublist_insertionSort (pairwise_pair.mpr hab) h


/--
A version of `insertionSort_stable` which only assumes `c <+~ l` (instead of `c <+ l`), but
additionally requires `IsAntisymm α r`, `IsTotal α r` and `IsTrans α r`.
-/
theorem sublist_insertionSort' {l c : List α} (hs : c.Sorted r) (hc : c <+~ l) :
    c <+ insertionSort r l := by
  classical
  obtain ⟨d, hc, hd⟩ := hc
  induction l generalizing c d with
  | nil         => simp_all only [sublist_nil, insertionSort, nil_perm]
  | cons a _ ih =>
    cases hd with
    | cons  _ h => exact ih hs _ hc h |>.trans (sublist_orderedInsert ..)
    | cons₂ _ h =>
      specialize ih (hs.erase _) _ (erase_cons_head a ‹List _› ▸ hc.erase a) h
      have hm := hc.mem_iff.mp <| mem_cons_self ..
      have he := orderedInsert_erase _ _ hm hs
      exact he ▸ Sublist.orderedInsert_sublist _ ih (sorted_insertionSort ..)


/--
Another statement of stability of insertion sort.
If a pair `[a, b]` is a sublist of a permutation of `l` and `a ≼ b`,
then `[a, b]` is still a sublist of `insertionSort r l`.
-/
theorem pair_sublist_insertionSort' {a b : α} {l : List α} (hab : a ≼ b) (h : [a, b] <+~ l) :
    [a, b] <+ insertionSort r l :=
  sublist_insertionSort' (pairwise_pair.mpr hab) h


theorem Sorted.merge {l l' : List α} (h : Sorted r l) (h' : Sorted r l') :
    Sorted r (merge l l' (r · ·)) := by
  simpa using sorted_merge (le := (r · ·))
    (fun a b c h₁ h₂ => by simpa using _root_.trans (by simpa using h₁) (by simpa using h₂))
    (fun a b => by simpa using IsTotal.total a b)
    l l' (by simpa using h) (by simpa using h')


/-- Variant of `sorted_mergeSort` using order typeclasses. -/
theorem sorted_mergeSort' [Preorder α] [DecidableRel ((· : α) ≤ ·)] [IsTotal α (· ≤ ·)]
                                   /-
                                     α : Type u
                                     β : Type v
                                     r : α → α → Prop
                                     s : β → β → Prop
                                     inst✝⁶ : DecidableRel r
                                     inst✝⁵ : DecidableRel s
                                     inst✝⁴ : IsTotal α r
                                     inst✝³ : IsTrans α r
                                     inst✝² : Preorder α
                                     inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
                                     inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
                                     l : List α
                                     ⊢ α → α → Bool
                                   -/
    (l : List α) : Sorted (· ≤ ·) (mergeSort l) := by
                                   /-
                                     🎉 no goals
                                   -/
  simpa using sorted_mergeSort (le := fun a b => a ≤ b)
    (fun a b c h₁ h₂ => by simpa using le_trans (by simpa using h₁) (by simpa using h₂))
    (fun a b => by simpa using IsTotal.total a b)
    l


                                                                            /-
                                                                              α : Type u
                                                                              β : Type v
                                                                              r : α → α → Prop
                                                                              s : β → β → Prop
                                                                              inst✝⁴ : DecidableRel r
                                                                              inst✝³ : DecidableRel s
                                                                              inst✝² : IsTotal α r
                                                                              inst✝¹ : IsTrans α r
                                                                              inst✝ : LinearOrder α
                                                                              l : List α
                                                                              ⊢ α → α → Bool
                                                                            -/
theorem mergeSort_eq_self [LinearOrder α] {l : List α} : Sorted (· ≤ ·) l → mergeSort l = l :=
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  eq_of_perm_of_sorted (mergeSort_perm _ _) (sorted_mergeSort' l)


theorem mergeSort_eq_insertionSort [IsAntisymm α r] (l : List α) :
    mergeSort l (r · ·) = insertionSort r l :=
  eq_of_perm_of_sorted ((mergeSort_perm l _).trans (perm_insertionSort r l).symm)
    (sorted_mergeSort (le := (r · ·))
                             /-
                               α : Type u
                               r : α → α → Prop
                               inst✝³ : DecidableRel r
                               inst✝² : IsTotal α r
                               inst✝¹ : IsTrans α r
                               inst✝ : IsAntisymm α r
                               l : List α
                               a b c : α
                               h₁ : Eq ((fun x1 x2 => Decidable.decide (r x1 x2)) a b) Bool.true
                               h₂ : Eq ((fun x1 x2 => Decidable.decide (r x1 x2)) b c) Bool.true
                               ⊢ Eq ((fun x1 x2 => Decidable.decide (r x1 x2)) a c) Bool.true
                             -/
      (fun a b c h₁ h₂ => by simpa using _root_.trans (by simpa using h₁) (by simpa using h₂))
                             /-
                               🎉 no goals
                             -/
                     /-
                       α : Type u
                       r : α → α → Prop
                       inst✝³ : DecidableRel r
                       inst✝² : IsTotal α r
                       inst✝¹ : IsTrans α r
                       inst✝ : IsAntisymm α r
                       l : List α
                       a b : α
                       ⊢ Eq (((fun x1 x2 => Decidable.decide (r x1 x2)) a b).or ((fun x1 x2 => Decida …
                     -/
      (fun a b => by simpa using IsTotal.total a b)
                     /-
                       🎉 no goals
                     -/
      l)
    (sorted_insertionSort r l).decide


