/-- Fold of a `s : Multiset α` with `f : α → β → β`, given a proof that `LeftCommutative f`
on all elements `x ∈ s`. -/
def noncommFoldr (s : Multiset α)
    (comm : { x | x ∈ s }.Pairwise fun x y => ∀ b, f x (f y b) = f y (f x b)) (b : β) : β :=
  letI : LeftCommutative (α := { x // x ∈ s }) (f ∘ Subtype.val) :=
    ⟨fun ⟨_, hx⟩ ⟨_, hy⟩ =>
      haveI : IsRefl α fun x y => ∀ b, f x (f y b) = f y (f x b) := ⟨fun _ _ => rfl⟩
      comm.of_refl hx hy⟩
  s.attach.foldr (f ∘ Subtype.val) b


@[simp]
theorem noncommFoldr_coe (l : List α) (comm) (b : β) :
    noncommFoldr f (l : Multiset α) comm b = l.foldr f b := by
  /-
    α : Type u_3
    β : Type u_4
    f : α → β → β
    l : List α
    comm : (setOf fun x => Membership.mem (↑l) x).Pairwise fun x y => ∀ (b : β), E …
    b : β
    ⊢ Eq (Multiset.noncommFoldr f (↑l) comm b) (List.foldr f b l)
  -/
  simp only [noncommFoldr, coe_foldr, coe_attach, List.attach, List.attachWith, Function.comp_def]
  /-
    α : Type u_3
    β : Type u_4
    f : α → β → β
    l : List α
    comm : (setOf fun x => Membership.mem (↑l) x).Pairwise fun x y => ∀ (b : β), E …
    b : β
    ⊢ Eq (List.foldr (fun x => f ↑x) b (List.pmap Subtype.mk l ⋯)) (List.foldr f b …
  -/
  rw [← List.foldr_map]
  /-
    α : Type u_3
    β : Type u_4
    f : α → β → β
    l : List α
    comm : (setOf fun x => Membership.mem (↑l) x).Pairwise fun x y => ∀ (b : β), E …
    b : β
    ⊢ Eq (List.foldr f b (List.map Subtype.val (List.pmap Subtype.mk l ⋯))) (List. …
  -/
  simp [List.map_pmap]
  /-
    🎉 no goals
  -/


@[simp]
theorem noncommFoldr_empty (h) (b : β) : noncommFoldr f (0 : Multiset α) h b = b :=
  rfl


theorem noncommFoldr_cons (s : Multiset α) (a : α) (h h') (b : β) :
    noncommFoldr f (a ::ₘ s) h b = f a (noncommFoldr f s h' b) := by
  /-
    α : Type u_3
    β : Type u_4
    f : α → β → β
    s : Multiset α
    a : α
    h : (setOf fun x => Membership.mem (Multiset.cons a s) x).Pairwise fun x y =>  …
    h' : (setOf fun x => Membership.mem s x).Pairwise fun x y => ∀ (b : β), Eq (f  …
    b : β
    ⊢ Eq (Multiset.noncommFoldr f (Multiset.cons a s) h b) (f a (Multiset.noncommF …
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_3
    β : Type u_4
    f : α → β → β
    a : α
    b : β
    a✝ : List α
    h : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSetoi …
    h' : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) a✝) x).Pair …
    ⊢ Eq (Multiset.noncommFoldr f (Multiset.cons a (Quotient.mk (List.isSetoid α)  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem noncommFoldr_eq_foldr (s : Multiset α) [h : LeftCommutative f] (b : β) :
    noncommFoldr f s (fun x _ y _ _ => h.left_comm x y) b = foldr f b s := by
  /-
    α : Type u_3
    β : Type u_4
    f : α → β → β
    s : Multiset α
    h : LeftCommutative f
    b : β
    ⊢ Eq (Multiset.noncommFoldr f s ⋯ b) (Multiset.foldr f b s)
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_3
    β : Type u_4
    f : α → β → β
    h : LeftCommutative f
    b : β
    a✝ : List α
    ⊢ Eq (Multiset.noncommFoldr f (Quotient.mk (List.isSetoid α) a✝) ⋯ b) (Multise …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Fold of a `s : Multiset α` with an associative `op : α → α → α`, given a proofs that `op`
is commutative on all elements `x ∈ s`. -/
def noncommFold (s : Multiset α) (comm : { x | x ∈ s }.Pairwise fun x y => op x y = op y x) :
    α → α :=
                                            /-
                                              F : Type u_1
                                              ι : Type u_2
                                              α : Type u_3
                                              β : Type u_4
                                              γ : Type u_5
                                              f : α → β → β
                                              op : α → α → α
                                              assoc : Std.Associative op
                                              s : Multiset α
                                              comm : (setOf fun x => Membership.mem s x).Pairwise fun x y => Eq (op x y) (op …
                                              x : α
                                              hx : Membership.mem (setOf fun x => Membership.mem s x) x
                                              y : α
                                              hy : Membership.mem (setOf fun x => Membership.mem s x) y
                                              h : Ne x y
                                              b : α
                                              ⊢ Eq (op x (op y b)) (op y (op x b))
                                            -/
  noncommFoldr op s fun x hx y hy h b => by rw [← assoc.assoc, comm hx hy h, assoc.assoc]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem noncommFold_coe (l : List α) (comm) (a : α) :
                                                                /-
                                                                  α : Type u_3
                                                                  op : α → α → α
                                                                  assoc : Std.Associative op
                                                                  l : List α
                                                                  comm : (setOf fun x => Membership.mem (↑l) x).Pairwise fun x y => Eq (op x y)  …
                                                                  a : α
                                                                  ⊢ Eq (Multiset.noncommFold op (↑l) comm a) (List.foldr op a l)
                                                                -/
    noncommFold op (l : Multiset α) comm a = l.foldr op a := by simp [noncommFold]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem noncommFold_empty (h) (a : α) : noncommFold op (0 : Multiset α) h a = a :=
  rfl


theorem noncommFold_cons (s : Multiset α) (a : α) (h h') (x : α) :
    noncommFold op (a ::ₘ s) h x = op a (noncommFold op s h' x) := by
  /-
    α : Type u_3
    op : α → α → α
    assoc : Std.Associative op
    s : Multiset α
    a : α
    h : (setOf fun x => Membership.mem (Multiset.cons a s) x).Pairwise fun x y =>  …
    h' : (setOf fun x => Membership.mem s x).Pairwise fun x y => Eq (op x y) (op y …
    x : α
    ⊢ Eq (Multiset.noncommFold op (Multiset.cons a s) h x) (op a (Multiset.noncomm …
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_3
    op : α → α → α
    assoc : Std.Associative op
    a x : α
    a✝ : List α
    h : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSetoi …
    h' : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) a✝) x).Pair …
    ⊢ Eq (Multiset.noncommFold op (Multiset.cons a (Quotient.mk (List.isSetoid α)  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem noncommFold_eq_fold (s : Multiset α) [Std.Commutative op] (a : α) :
    noncommFold op s (fun x _ y _ _ => Std.Commutative.comm x y) a = fold op a s := by
  /-
    α : Type u_3
    op : α → α → α
    assoc : Std.Associative op
    s : Multiset α
    inst✝ : Std.Commutative op
    a : α
    ⊢ Eq (Multiset.noncommFold op s ⋯ a) (Multiset.fold op a s)
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_3
    op : α → α → α
    assoc : Std.Associative op
    inst✝ : Std.Commutative op
    a : α
    a✝ : List α
    ⊢ Eq (Multiset.noncommFold op (Quotient.mk (List.isSetoid α) a✝) ⋯ a) (Multise …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Product of a `s : Multiset α` with `[Monoid α]`, given a proof that `*` commutes
on all elements `x ∈ s`. -/
@[to_additive
      "Sum of a `s : Multiset α` with `[AddMonoid α]`, given a proof that `+` commutes
      on all elements `x ∈ s`."]
def noncommProd (s : Multiset α) (comm : { x | x ∈ s }.Pairwise Commute) : α :=
  s.noncommFold (· * ·) comm 1


@[to_additive (attr := simp)]
theorem noncommProd_coe (l : List α) (comm) : noncommProd (l : Multiset α) comm = l.prod := by
  /-
    α : Type u_3
    inst✝ : Monoid α
    l : List α
    comm : (setOf fun x => Membership.mem (↑l) x).Pairwise Commute
    ⊢ Eq ((↑l).noncommProd comm) l.prod
  -/
  rw [noncommProd]
  /-
    α : Type u_3
    inst✝ : Monoid α
    l : List α
    comm : (setOf fun x => Membership.mem (↑l) x).Pairwise Commute
    ⊢ Eq (Multiset.noncommFold (fun x1 x2 => HMul.hMul x1 x2) (↑l) comm 1) l.prod
  -/
  simp only [noncommFold_coe]
  /-
    α : Type u_3
    inst✝ : Monoid α
    l : List α
    comm : (setOf fun x => Membership.mem (↑l) x).Pairwise Commute
    ⊢ Eq (List.foldr (fun x1 x2 => HMul.hMul x1 x2) 1 l) l.prod
  -/
  induction' l with hd tl hl
    /-
      case nil
      α : Type u_3
      inst✝ : Monoid α
      comm : (setOf fun x => Membership.mem (↑List.nil) x).Pairwise Commute
      ⊢ Eq (List.foldr (fun x1 x2 => HMul.hMul x1 x2) 1 List.nil) List.nil.prod
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_3
      inst✝ : Monoid α
      hd : α
      tl : List α
      hl : (setOf fun x => Membership.mem (↑tl) x).Pairwise Commute → Eq (List.foldr …
      comm : (setOf fun x => Membership.mem (↑(List.cons hd tl)) x).Pairwise Commute
      ⊢ Eq (List.foldr (fun x1 x2 => HMul.hMul x1 x2) 1 (List.cons hd tl)) (List.con …
    -/
  · rw [List.prod_cons, List.foldr, hl]
    /-
      case cons
      α : Type u_3
      inst✝ : Monoid α
      hd : α
      tl : List α
      hl : (setOf fun x => Membership.mem (↑tl) x).Pairwise Commute → Eq (List.foldr …
      comm : (setOf fun x => Membership.mem (↑(List.cons hd tl)) x).Pairwise Commute
      ⊢ (setOf fun x => Membership.mem (↑tl) x).Pairwise Commute
    -/
    intro x hx y hy
    /-
      case cons
      α : Type u_3
      inst✝ : Monoid α
      hd : α
      tl : List α
      hl : (setOf fun x => Membership.mem (↑tl) x).Pairwise Commute → Eq (List.foldr …
      comm : (setOf fun x => Membership.mem (↑(List.cons hd tl)) x).Pairwise Commute
      x : α
      hx : Membership.mem (setOf fun x => Membership.mem (↑tl) x) x
      y : α
      hy : Membership.mem (setOf fun x => Membership.mem (↑tl) x) y
      ⊢ Ne x y → Commute x y
    -/
    exact comm (List.mem_cons_of_mem _ hx) (List.mem_cons_of_mem _ hy)
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem noncommProd_empty (h) : noncommProd (0 : Multiset α) h = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem noncommProd_cons (s : Multiset α) (a : α) (comm) :
    noncommProd (a ::ₘ s) comm = a * noncommProd s (comm.mono fun _ => mem_cons_of_mem) := by
  /-
    α : Type u_3
    inst✝ : Monoid α
    s : Multiset α
    a : α
    comm : (setOf fun x => Membership.mem (Multiset.cons a s) x).Pairwise Commute
    ⊢ Eq ((Multiset.cons a s).noncommProd comm) (HMul.hMul a (s.noncommProd ⋯))
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_3
    inst✝ : Monoid α
    a : α
    a✝ : List α
    comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
    ⊢ Eq ((Multiset.cons a (Quotient.mk (List.isSetoid α) a✝)).noncommProd comm) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive]
theorem noncommProd_cons' (s : Multiset α) (a : α) (comm) :
    noncommProd (a ::ₘ s) comm = noncommProd s (comm.mono fun _ => mem_cons_of_mem) * a := by
  /-
    α : Type u_3
    inst✝ : Monoid α
    s : Multiset α
    a : α
    comm : (setOf fun x => Membership.mem (Multiset.cons a s) x).Pairwise Commute
    ⊢ Eq ((Multiset.cons a s).noncommProd comm) (HMul.hMul (s.noncommProd ⋯) a)
  -/
  induction' s using Quotient.inductionOn with s
  /-
    case h
    α : Type u_3
    inst✝ : Monoid α
    a : α
    s : List α
    comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
    ⊢ Eq ((Multiset.cons a (Quotient.mk (List.isSetoid α) s)).noncommProd comm) (H …
  -/
  simp only [quot_mk_to_coe, cons_coe, noncommProd_coe, List.prod_cons]
  /-
    case h
    α : Type u_3
    inst✝ : Monoid α
    a : α
    s : List α
    comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
    ⊢ Eq (HMul.hMul a s.prod) (HMul.hMul s.prod a)
  -/
  induction' s with hd tl IH
    /-
      case h.nil
      α : Type u_3
      inst✝ : Monoid α
      a : α
      comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
      ⊢ Eq (HMul.hMul a List.nil.prod) (HMul.hMul List.nil.prod a)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.cons
      α : Type u_3
      inst✝ : Monoid α
      a hd : α
      tl : List α
      IH : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSeto …
      comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
      ⊢ Eq (HMul.hMul a (List.cons hd tl).prod) (HMul.hMul (List.cons hd tl).prod a)
    -/
  · rw [List.prod_cons, mul_assoc, ← IH, ← mul_assoc, ← mul_assoc]
      /-
        case h.cons
        α : Type u_3
        inst✝ : Monoid α
        a hd : α
        tl : List α
        IH : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSeto …
        comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
        ⊢ Eq (HMul.hMul (HMul.hMul a hd) tl.prod) (HMul.hMul (HMul.hMul hd a) tl.prod)
      -/
    · congr 1
      /-
        case h.cons.e_a
        α : Type u_3
        inst✝ : Monoid α
        a hd : α
        tl : List α
        IH : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSeto …
        comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
        ⊢ Eq (HMul.hMul a hd) (HMul.hMul hd a)
      -/
                             /-
                               🎉 no goals
                             -/
      apply comm.of_refl <;> simp
                             /-
                               🎉 no goals
                             -/
      /-
        case h.cons
        α : Type u_3
        inst✝ : Monoid α
        a hd : α
        tl : List α
        IH : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSeto …
        comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
        ⊢ (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSetoid  …
      -/
    · intro x hx y hy
      /-
        case h.cons
        α : Type u_3
        inst✝ : Monoid α
        a hd : α
        tl : List α
        IH : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSeto …
        comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
        x : α
        hx : Membership.mem (setOf fun x => Membership.mem (Multiset.cons a (Quotient. …
        y : α
        hy : Membership.mem (setOf fun x => Membership.mem (Multiset.cons a (Quotient. …
        ⊢ Ne x y → Commute x y
      -/
      simp only [quot_mk_to_coe, List.mem_cons, mem_coe, cons_coe] at hx hy
      /-
        case h.cons
        α : Type u_3
        inst✝ : Monoid α
        a hd : α
        tl : List α
        IH : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSeto …
        comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
        x y : α
        hx : Membership.mem (setOf fun x => Or (Eq x a) (Membership.mem tl x)) x
        hy : Membership.mem (setOf fun x => Or (Eq x a) (Membership.mem tl x)) y
        ⊢ Ne x y → Commute x y
      -/
      apply comm
        /-
          case h.cons.a
          α : Type u_3
          inst✝ : Monoid α
          a hd : α
          tl : List α
          IH : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSeto …
          comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
          x y : α
          hx : Membership.mem (setOf fun x => Or (Eq x a) (Membership.mem tl x)) x
          hy : Membership.mem (setOf fun x => Or (Eq x a) (Membership.mem tl x)) y
          ⊢ Membership.mem (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk  …
        -/
                     /-
                       🎉 no goals
                     -/
      · cases hx <;> simp [*]
                     /-
                       🎉 no goals
                     -/
        /-
          case h.cons.a
          α : Type u_3
          inst✝ : Monoid α
          a hd : α
          tl : List α
          IH : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSeto …
          comm : (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk (List.isSe …
          x y : α
          hx : Membership.mem (setOf fun x => Or (Eq x a) (Membership.mem tl x)) x
          hy : Membership.mem (setOf fun x => Or (Eq x a) (Membership.mem tl x)) y
          ⊢ Membership.mem (setOf fun x => Membership.mem (Multiset.cons a (Quotient.mk  …
        -/
                     /-
                       🎉 no goals
                     -/
      · cases hy <;> simp [*]
                     /-
                       🎉 no goals
                     -/


@[to_additive]
theorem noncommProd_add (s t : Multiset α) (comm) :
    noncommProd (s + t) comm =
      noncommProd s (comm.mono <| subset_of_le <| s.le_add_right t) *
        noncommProd t (comm.mono <| subset_of_le <| t.le_add_left s) := by
  /-
    α : Type u_3
    inst✝ : Monoid α
    s t : Multiset α
    comm : (setOf fun x => Membership.mem (HAdd.hAdd s t) x).Pairwise Commute
    ⊢ Eq ((HAdd.hAdd s t).noncommProd comm) (HMul.hMul (s.noncommProd ⋯) (t.noncom …
  -/
  rcases s with ⟨⟩
  /-
    case mk
    α : Type u_3
    inst✝ : Monoid α
    s t : Multiset α
    a✝ : List α
    comm : (setOf fun x => Membership.mem (HAdd.hAdd (Quot.mk (⇑(List.isSetoid α)) …
    ⊢ Eq ((HAdd.hAdd (Quot.mk (⇑(List.isSetoid α)) a✝) t).noncommProd comm) (HMul. …
  -/
  rcases t with ⟨⟩
  /-
    case mk.mk
    α : Type u_3
    inst✝ : Monoid α
    s t : Multiset α
    a✝¹ a✝ : List α
    comm : (setOf fun x => Membership.mem (HAdd.hAdd (Quot.mk (⇑(List.isSetoid α)) …
    ⊢ Eq ((HAdd.hAdd (Quot.mk (⇑(List.isSetoid α)) a✝¹) (Quot.mk (⇑(List.isSetoid  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive]
lemma noncommProd_induction (s : Multiset α) (comm)
    (p : α → Prop) (hom : ∀ a b, p a → p b → p (a * b)) (unit : p 1) (base : ∀ x ∈ s, p x) :
    p (s.noncommProd comm) := by
  /-
    α : Type u_3
    inst✝ : Monoid α
    s : Multiset α
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    p : α → Prop
    hom : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    unit : p 1
    base : ∀ (x : α), Membership.mem s x → p x
    ⊢ p (s.noncommProd comm)
  -/
  induction' s using Quotient.inductionOn with l
  /-
    case h
    α : Type u_3
    inst✝ : Monoid α
    p : α → Prop
    hom : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    unit : p 1
    l : List α
    comm : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) l) x).Pai …
    base : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) l) x → p x
    ⊢ p (Multiset.noncommProd (Quotient.mk (List.isSetoid α) l) comm)
  -/
  simp only [quot_mk_to_coe, noncommProd_coe, mem_coe] at base ⊢
  /-
    case h
    α : Type u_3
    inst✝ : Monoid α
    p : α → Prop
    hom : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    unit : p 1
    l : List α
    comm : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) l) x).Pai …
    base : ∀ (x : α), Membership.mem l x → p x
    ⊢ p l.prod
  -/
  exact l.prod_induction p hom unit base
  /-
    🎉 no goals
  -/


@[to_additive]
protected theorem map_noncommProd_aux [MonoidHomClass F α β] (s : Multiset α)
    (comm : { x | x ∈ s }.Pairwise Commute) (f : F) : { x | x ∈ s.map f }.Pairwise Commute := by
  /-
    F : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝³ : Monoid α
    inst✝² : Monoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    s : Multiset α
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    f : F
    ⊢ (setOf fun x => Membership.mem (Multiset.map (⇑f) s) x).Pairwise Commute
  -/
  simp only [Multiset.mem_map]
  /-
    F : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝³ : Monoid α
    inst✝² : Monoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    s : Multiset α
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    f : F
    ⊢ (setOf fun x => Exists fun a => And (Membership.mem s a) (Eq (f a) x)).Pairw …
  -/
  rintro _ ⟨x, hx, rfl⟩ _ ⟨y, hy, rfl⟩ _
  /-
    case intro.intro.intro.intro
    F : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝³ : Monoid α
    inst✝² : Monoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    s : Multiset α
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    f : F
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    a✝ : Ne (f x) (f y)
    ⊢ Commute (f x) (f y)
  -/
  exact (comm.of_refl hx hy).map f
  /-
    🎉 no goals
  -/


@[to_additive]
theorem map_noncommProd [MonoidHomClass F α β] (s : Multiset α) (comm) (f : F) :
    f (s.noncommProd comm) = (s.map f).noncommProd (Multiset.map_noncommProd_aux s comm f) := by
  /-
    F : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝³ : Monoid α
    inst✝² : Monoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    s : Multiset α
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    f : F
    ⊢ Eq (f (s.noncommProd comm)) ((Multiset.map (⇑f) s).noncommProd ⋯)
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    F : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝³ : Monoid α
    inst✝² : Monoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    f : F
    a✝ : List α
    comm : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) a✝) x).Pa …
    ⊢ Eq (f (Multiset.noncommProd (Quotient.mk (List.isSetoid α) a✝) comm)) ((Mult …
  -/
  simpa using map_list_prod f _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-23")] alias noncommProd_map := map_noncommProd

@[deprecated (since := "2024-07-23")] alias noncommSum_map := map_noncommSum

@[deprecated (since := "2024-07-23")]
protected alias noncommProd_map_aux := Multiset.map_noncommProd_aux

@[deprecated (since := "2024-07-23")]
protected alias noncommSum_map_aux := Multiset.map_noncommSum_aux


@[to_additive noncommSum_eq_card_nsmul]
theorem noncommProd_eq_pow_card (s : Multiset α) (comm) (m : α) (h : ∀ x ∈ s, x = m) :
    s.noncommProd comm = m ^ Multiset.card s := by
  /-
    α : Type u_3
    inst✝ : Monoid α
    s : Multiset α
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    m : α
    h : ∀ (x : α), Membership.mem s x → Eq x m
    ⊢ Eq (s.noncommProd comm) (HPow.hPow m s.card)
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_3
    inst✝ : Monoid α
    m : α
    a✝ : List α
    comm : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) a✝) x).Pa …
    h : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) a✝) x → Eq x m
    ⊢ Eq (Multiset.noncommProd (Quotient.mk (List.isSetoid α) a✝) comm) (HPow.hPow …
  -/
  simp only [quot_mk_to_coe, noncommProd_coe, coe_card, mem_coe] at *
  /-
    case h
    α : Type u_3
    inst✝ : Monoid α
    m : α
    a✝ : List α
    comm : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) a✝) x).Pa …
    h : ∀ (x : α), Membership.mem a✝ x → Eq x m
    ⊢ Eq a✝.prod (HPow.hPow m a✝.length)
  -/
  exact List.prod_eq_pow_card _ m h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem noncommProd_eq_prod {α : Type*} [CommMonoid α] (s : Multiset α) :
    (noncommProd s fun _ _ _ _ _ => Commute.all _ _) = prod s := by
  /-
    α : Type u_6
    inst✝ : CommMonoid α
    s : Multiset α
    ⊢ Eq (s.noncommProd ⋯) s.prod
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_6
    inst✝ : CommMonoid α
    a✝ : List α
    ⊢ Eq (Multiset.noncommProd (Quotient.mk (List.isSetoid α) a✝) ⋯) (Multiset.pro …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive]
theorem noncommProd_commute (s : Multiset α) (comm) (y : α) (h : ∀ x ∈ s, Commute y x) :
    Commute y (s.noncommProd comm) := by
  /-
    α : Type u_3
    inst✝ : Monoid α
    s : Multiset α
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    y : α
    h : ∀ (x : α), Membership.mem s x → Commute y x
    ⊢ Commute y (s.noncommProd comm)
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_3
    inst✝ : Monoid α
    y : α
    a✝ : List α
    comm : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) a✝) x).Pa …
    h : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) a✝) x → Commute y x
    ⊢ Commute y (Multiset.noncommProd (Quotient.mk (List.isSetoid α) a✝) comm)
  -/
  simp only [quot_mk_to_coe, noncommProd_coe]
  /-
    case h
    α : Type u_3
    inst✝ : Monoid α
    y : α
    a✝ : List α
    comm : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) a✝) x).Pa …
    h : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) a✝) x → Commute y x
    ⊢ Commute y a✝.prod
  -/
  exact Commute.list_prod_right _ _ h
  /-
    🎉 no goals
  -/


theorem mul_noncommProd_erase [DecidableEq α] (s : Multiset α) {a : α} (h : a ∈ s) (comm)
    (comm' := fun _ hx _ hy hxy ↦ comm (s.mem_of_mem_erase hx) (s.mem_of_mem_erase hy) hxy) :
    a * (s.erase a).noncommProd comm' = s.noncommProd comm := by
  /-
    α : Type u_3
    inst✝¹ : Monoid α
    inst✝ : DecidableEq α
    s : Multiset α
    a : α
    h : Membership.mem s a
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (s. …
    ⊢ Eq (HMul.hMul a ((s.erase a).noncommProd comm')) (s.noncommProd comm)
  -/
  induction' s using Quotient.inductionOn with l
  /-
    case h
    α : Type u_3
    inst✝¹ : Monoid α
    inst✝ : DecidableEq α
    a : α
    l : List α
    h : Membership.mem (Quotient.mk (List.isSetoid α) l) a
    comm : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) l) x).Pai …
    comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (Mu …
    ⊢ Eq (HMul.hMul a ((Multiset.erase (Quotient.mk (List.isSetoid α) l) a).noncom …
  -/
  simp only [quot_mk_to_coe, mem_coe, coe_erase, noncommProd_coe] at comm h ⊢
  /-
    case h
    α : Type u_3
    inst✝¹ : Monoid α
    inst✝ : DecidableEq α
    a : α
    l : List α
    comm✝ : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) l) x).Pa …
    comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (Mu …
    comm : (setOf fun x => Membership.mem l x).Pairwise Commute
    h : Membership.mem l a
    ⊢ Eq (HMul.hMul a (l.erase a).prod) l.prod
  -/
  suffices ∀ x ∈ l, ∀ y ∈ l, x * y = y * x by rw [List.prod_erase_of_comm h this]
  /-
    case h
    α : Type u_3
    inst✝¹ : Monoid α
    inst✝ : DecidableEq α
    a : α
    l : List α
    comm✝ : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) l) x).Pa …
    comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (Mu …
    comm : (setOf fun x => Membership.mem l x).Pairwise Commute
    h : Membership.mem l a
    ⊢ ∀ (x : α), Membership.mem l x → ∀ (y : α), Membership.mem l y → Eq (HMul.hMu …
  -/
  intro x hx y hy
  /-
    case h
    α : Type u_3
    inst✝¹ : Monoid α
    inst✝ : DecidableEq α
    a : α
    l : List α
    comm✝ : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) l) x).Pa …
    comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (Mu …
    comm : (setOf fun x => Membership.mem l x).Pairwise Commute
    h : Membership.mem l a
    x : α
    hx : Membership.mem l x
    y : α
    hy : Membership.mem l y
    ⊢ Eq (HMul.hMul x y) (HMul.hMul y x)
  -/
  rcases eq_or_ne x y with rfl | hxy
    /-
      case h.inl
      α : Type u_3
      inst✝¹ : Monoid α
      inst✝ : DecidableEq α
      a : α
      l : List α
      comm✝ : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) l) x).Pa …
      comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (Mu …
      comm : (setOf fun x => Membership.mem l x).Pairwise Commute
      h : Membership.mem l a
      x : α
      hx hy : Membership.mem l x
      ⊢ Eq (HMul.hMul x x) (HMul.hMul x x)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case h.inr
    α : Type u_3
    inst✝¹ : Monoid α
    inst✝ : DecidableEq α
    a : α
    l : List α
    comm✝ : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid α) l) x).Pa …
    comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (Mu …
    comm : (setOf fun x => Membership.mem l x).Pairwise Commute
    h : Membership.mem l a
    x : α
    hx : Membership.mem l x
    y : α
    hy : Membership.mem l y
    hxy : Ne x y
    ⊢ Eq (HMul.hMul x y) (HMul.hMul y x)
  -/
  exact comm hx hy hxy
  /-
    🎉 no goals
  -/


theorem noncommProd_erase_mul [DecidableEq α] (s : Multiset α) {a : α} (h : a ∈ s) (comm)
    (comm' := fun _ hx _ hy hxy ↦ comm (s.mem_of_mem_erase hx) (s.mem_of_mem_erase hy) hxy) :
    (s.erase a).noncommProd comm' * a = s.noncommProd comm := by
  suffices ∀ b ∈ erase s a, Commute a b by
    rw [← (noncommProd_commute (s.erase a) comm' a this).eq, mul_noncommProd_erase s h comm comm']
  /-
    α : Type u_3
    inst✝¹ : Monoid α
    inst✝ : DecidableEq α
    s : Multiset α
    a : α
    h : Membership.mem s a
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (s. …
    ⊢ ∀ (b : α), Membership.mem (s.erase a) b → Commute a b
  -/
  intro b hb
  /-
    α : Type u_3
    inst✝¹ : Monoid α
    inst✝ : DecidableEq α
    s : Multiset α
    a : α
    h : Membership.mem s a
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (s. …
    b : α
    hb : Membership.mem (s.erase a) b
    ⊢ Commute a b
  -/
  rcases eq_or_ne a b with rfl | hab
    /-
      case inl
      α : Type u_3
      inst✝¹ : Monoid α
      inst✝ : DecidableEq α
      s : Multiset α
      a : α
      h : Membership.mem s a
      comm : (setOf fun x => Membership.mem s x).Pairwise Commute
      comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (s. …
      hb : Membership.mem (s.erase a) a
      ⊢ Commute a a
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_3
    inst✝¹ : Monoid α
    inst✝ : DecidableEq α
    s : Multiset α
    a : α
    h : Membership.mem s a
    comm : (setOf fun x => Membership.mem s x).Pairwise Commute
    comm' : optParam (∀ (x : α), Membership.mem (setOf fun x => Membership.mem (s. …
    b : α
    hb : Membership.mem (s.erase a) b
    hab : Ne a b
    ⊢ Commute a b
  -/
  exact comm h (mem_of_mem_erase hb) hab
  /-
    🎉 no goals
  -/


/-- Proof used in definition of `Finset.noncommProd` -/
@[to_additive]
theorem noncommProd_lemma (s : Finset α) (f : α → β)
    (comm : (s : Set α).Pairwise (Commute on f)) :
    Set.Pairwise { x | x ∈ Multiset.map f s.val } Commute := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    ⊢ (setOf fun x => Membership.mem (Multiset.map f s.val) x).Pairwise Commute
  -/
  simp_rw [Multiset.mem_map]
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    ⊢ (setOf fun x => Exists fun a => And (Membership.mem s.val a) (Eq (f a) x)).P …
  -/
  rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩ _
  /-
    case intro.intro.intro.intro
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    a : α
    ha : Membership.mem s.val a
    b : α
    hb : Membership.mem s.val b
    a✝ : Ne (f a) (f b)
    ⊢ Commute (f a) (f b)
  -/
  exact comm.of_refl ha hb
  /-
    🎉 no goals
  -/


/-- Product of a `s : Finset α` mapped with `f : α → β` with `[Monoid β]`,
given a proof that `*` commutes on all elements `f x` for `x ∈ s`. -/
@[to_additive
      "Sum of a `s : Finset α` mapped with `f : α → β` with `[AddMonoid β]`,
given a proof that `+` commutes on all elements `f x` for `x ∈ s`."]
def noncommProd (s : Finset α) (f : α → β)
    (comm : (s : Set α).Pairwise (Commute on f)) : β :=
  (s.1.map f).noncommProd <| noncommProd_lemma s f comm


@[to_additive]
lemma noncommProd_induction (s : Finset α) (f : α → β) (comm)
    (p : β → Prop) (hom : ∀ a b, p a → p b → p (a * b)) (unit : p 1) (base : ∀ x ∈ s, p (f x)) :
    p (s.noncommProd f comm) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    p : β → Prop
    hom : ∀ (a b : β), p a → p b → p (HMul.hMul a b)
    unit : p 1
    base : ∀ (x : α), Membership.mem s x → p (f x)
    ⊢ p (s.noncommProd f comm)
  -/
  refine Multiset.noncommProd_induction _ _ _ hom unit fun b hb ↦ ?_
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    p : β → Prop
    hom : ∀ (a b : β), p a → p b → p (HMul.hMul a b)
    unit : p 1
    base : ∀ (x : α), Membership.mem s x → p (f x)
    b : β
    hb : Membership.mem (Multiset.map f s.val) b
    ⊢ p b
  -/
  obtain (⟨a, ha : a ∈ s, rfl : f a = b⟩) := by simpa using hb
  /-
    case intro.intro
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    p : β → Prop
    hom : ∀ (a b : β), p a → p b → p (HMul.hMul a b)
    unit : p 1
    base : ∀ (x : α), Membership.mem s x → p (f x)
    a : α
    ha : Membership.mem s a
    hb : Membership.mem (Multiset.map f s.val) (f a)
    ⊢ p (f a)
  -/
  exact base a ha
  /-
    🎉 no goals
  -/


@[to_additive (attr := congr)]
theorem noncommProd_congr {s₁ s₂ : Finset α} {f g : α → β} (h₁ : s₁ = s₂)
    (h₂ : ∀ x ∈ s₂, f x = g x) (comm) :
    noncommProd s₁ f comm =
      noncommProd s₂ g fun x hx y hy h => by
        /-
          F : Type u_1
          ι : Type u_2
          α : Type u_3
          β : Type u_4
          γ : Type u_5
          f✝ : α → β → β
          op : α → α → α
          inst✝¹ : Monoid β
          inst✝ : Monoid γ
          s₁ s₂ : Finset α
          f g : α → β
          h₁ : Eq s₁ s₂
          h₂ : ∀ (x : α), Membership.mem s₂ x → Eq (f x) (g x)
          comm : (↑s₁).Pairwise (Function.onFun Commute f)
          x : α
          hx : Membership.mem (↑s₂) x
          y : α
          hy : Membership.mem (↑s₂) y
          h : Ne x y
          ⊢ Function.onFun Commute g x y
        -/
        dsimp only [Function.onFun]
        /-
          F : Type u_1
          ι : Type u_2
          α : Type u_3
          β : Type u_4
          γ : Type u_5
          f✝ : α → β → β
          op : α → α → α
          inst✝¹ : Monoid β
          inst✝ : Monoid γ
          s₁ s₂ : Finset α
          f g : α → β
          h₁ : Eq s₁ s₂
          h₂ : ∀ (x : α), Membership.mem s₂ x → Eq (f x) (g x)
          comm : (↑s₁).Pairwise (Function.onFun Commute f)
          x : α
          hx : Membership.mem (↑s₂) x
          y : α
          hy : Membership.mem (↑s₂) y
          h : Ne x y
          ⊢ Commute (g x) (g y)
        -/
        rw [← h₂ _ hx, ← h₂ _ hy]
        /-
          F : Type u_1
          ι : Type u_2
          α : Type u_3
          β : Type u_4
          γ : Type u_5
          f✝ : α → β → β
          op : α → α → α
          inst✝¹ : Monoid β
          inst✝ : Monoid γ
          s₁ s₂ : Finset α
          f g : α → β
          h₁ : Eq s₁ s₂
          h₂ : ∀ (x : α), Membership.mem s₂ x → Eq (f x) (g x)
          comm : (↑s₁).Pairwise (Function.onFun Commute f)
          x : α
          hx : Membership.mem (↑s₂) x
          y : α
          hy : Membership.mem (↑s₂) y
          h : Ne x y
          ⊢ Commute (f x) (f y)
        -/
        subst h₁
        /-
          F : Type u_1
          ι : Type u_2
          α : Type u_3
          β : Type u_4
          γ : Type u_5
          f✝ : α → β → β
          op : α → α → α
          inst✝¹ : Monoid β
          inst✝ : Monoid γ
          s₁ : Finset α
          f g : α → β
          comm : (↑s₁).Pairwise (Function.onFun Commute f)
          x y : α
          h : Ne x y
          h₂ : ∀ (x : α), Membership.mem s₁ x → Eq (f x) (g x)
          hx : Membership.mem (↑s₁) x
          hy : Membership.mem (↑s₁) y
          ⊢ Commute (f x) (f y)
        -/
        exact comm hx hy h := by
        /-
          🎉 no goals
        -/
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s₁ s₂ : Finset α
    f g : α → β
    h₁ : Eq s₁ s₂
    h₂ : ∀ (x : α), Membership.mem s₂ x → Eq (f x) (g x)
    comm : (↑s₁).Pairwise (Function.onFun Commute f)
    ⊢ Eq (s₁.noncommProd f comm) (s₂.noncommProd g ⋯)
  -/
  simp_rw [noncommProd, Multiset.map_congr (congr_arg _ h₁) h₂]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem noncommProd_toFinset [DecidableEq α] (l : List α) (f : α → β) (comm) (hl : l.Nodup) :
    noncommProd l.toFinset f comm = (l.map f).prod := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : Monoid β
    inst✝ : DecidableEq α
    l : List α
    f : α → β
    comm : (↑l.toFinset).Pairwise (Function.onFun Commute f)
    hl : l.Nodup
    ⊢ Eq (l.toFinset.noncommProd f comm) (List.map f l).prod
  -/
  rw [← List.dedup_eq_self] at hl
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : Monoid β
    inst✝ : DecidableEq α
    l : List α
    f : α → β
    comm : (↑l.toFinset).Pairwise (Function.onFun Commute f)
    hl : Eq l.dedup l
    ⊢ Eq (l.toFinset.noncommProd f comm) (List.map f l).prod
  -/
  simp [noncommProd, hl]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem noncommProd_empty (f : α → β) (h) : noncommProd (∅ : Finset α) f h = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem noncommProd_cons (s : Finset α) (a : α) (f : α → β)
    (ha : a ∉ s) (comm) :
    noncommProd (cons a s ha) f comm =
      f a * noncommProd s f (comm.mono fun _ => Finset.mem_cons.2 ∘ .inr) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    a : α
    f : α → β
    ha : Not (Membership.mem s a)
    comm : (↑(Finset.cons a s ha)).Pairwise (Function.onFun Commute f)
    ⊢ Eq ((Finset.cons a s ha).noncommProd f comm) (HMul.hMul (f a) (s.noncommProd …
  -/
  simp_rw [noncommProd, Finset.cons_val, Multiset.map_cons, Multiset.noncommProd_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem noncommProd_cons' (s : Finset α) (a : α) (f : α → β)
    (ha : a ∉ s) (comm) :
    noncommProd (cons a s ha) f comm =
      noncommProd s f (comm.mono fun _ => Finset.mem_cons.2 ∘ .inr) * f a := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    a : α
    f : α → β
    ha : Not (Membership.mem s a)
    comm : (↑(Finset.cons a s ha)).Pairwise (Function.onFun Commute f)
    ⊢ Eq ((Finset.cons a s ha).noncommProd f comm) (HMul.hMul (s.noncommProd f ⋯)  …
  -/
  simp_rw [noncommProd, Finset.cons_val, Multiset.map_cons, Multiset.noncommProd_cons']
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem noncommProd_insert_of_not_mem [DecidableEq α] (s : Finset α) (a : α) (f : α → β) (comm)
    (ha : a ∉ s) :
    noncommProd (insert a s) f comm =
      f a * noncommProd s f (comm.mono fun _ => mem_insert_of_mem) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : Monoid β
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    f : α → β
    comm : (↑(Insert.insert a s)).Pairwise (Function.onFun Commute f)
    ha : Not (Membership.mem s a)
    ⊢ Eq ((Insert.insert a s).noncommProd f comm) (HMul.hMul (f a) (s.noncommProd  …
  -/
  simp only [← cons_eq_insert _ _ ha, noncommProd_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem noncommProd_insert_of_not_mem' [DecidableEq α] (s : Finset α) (a : α) (f : α → β) (comm)
    (ha : a ∉ s) :
    noncommProd (insert a s) f comm =
      noncommProd s f (comm.mono fun _ => mem_insert_of_mem) * f a := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : Monoid β
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    f : α → β
    comm : (↑(Insert.insert a s)).Pairwise (Function.onFun Commute f)
    ha : Not (Membership.mem s a)
    ⊢ Eq ((Insert.insert a s).noncommProd f comm) (HMul.hMul (s.noncommProd f ⋯) ( …
  -/
  simp only [← cons_eq_insert _ _ ha, noncommProd_cons']
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem noncommProd_singleton (a : α) (f : α → β) :
    noncommProd ({a} : Finset α) f
        (by
          /-
            F : Type u_1
            ι : Type u_2
            α : Type u_3
            β : Type u_4
            γ : Type u_5
            f✝ : α → β → β
            op : α → α → α
            inst✝¹ : Monoid β
            inst✝ : Monoid γ
            a : α
            f : α → β
            ⊢ (↑(Singleton.singleton a)).Pairwise (Function.onFun Commute f)
          -/
          norm_cast
          /-
            F : Type u_1
            ι : Type u_2
            α : Type u_3
            β : Type u_4
            γ : Type u_5
            f✝ : α → β → β
            op : α → α → α
            inst✝¹ : Monoid β
            inst✝ : Monoid γ
            a : α
            f : α → β
            ⊢ (Singleton.singleton a).Pairwise (Function.onFun Commute f)
          -/
          exact Set.pairwise_singleton _ _) =
          /-
            🎉 no goals
          -/
      f a := mul_one _


@[to_additive]
theorem map_noncommProd [MonoidHomClass F β γ] (s : Finset α) (f : α → β) (comm) (g : F) :
    g (s.noncommProd f comm) =
      s.noncommProd (fun i => g (f i)) fun _ hx _ hy _ => (comm.of_refl hx hy).map g := by
  /-
    F : Type u_1
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝³ : Monoid β
    inst✝² : Monoid γ
    inst✝¹ : FunLike F β γ
    inst✝ : MonoidHomClass F β γ
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    g : F
    ⊢ Eq (g (s.noncommProd f comm)) (s.noncommProd (fun i => g (f i)) ⋯)
  -/
  simp [noncommProd, Multiset.map_noncommProd]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-23")] alias noncommSum_map := map_noncommSum


@[to_additive noncommSum_eq_card_nsmul]
theorem noncommProd_eq_pow_card (s : Finset α) (f : α → β) (comm) (m : β) (h : ∀ x ∈ s, f x = m) :
    s.noncommProd f comm = m ^ s.card := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    m : β
    h : ∀ (x : α), Membership.mem s x → Eq (f x) m
    ⊢ Eq (s.noncommProd f comm) (HPow.hPow m s.card)
  -/
  rw [noncommProd, Multiset.noncommProd_eq_pow_card _ _ m]
    /-
      α : Type u_3
      β : Type u_4
      inst✝ : Monoid β
      s : Finset α
      f : α → β
      comm : (↑s).Pairwise (Function.onFun Commute f)
      m : β
      h : ∀ (x : α), Membership.mem s x → Eq (f x) m
      ⊢ Eq (HPow.hPow m (Multiset.map f s.val).card) (HPow.hPow m s.card)
    -/
  · simp only [Finset.card_def, Multiset.card_map]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_3
      β : Type u_4
      inst✝ : Monoid β
      s : Finset α
      f : α → β
      comm : (↑s).Pairwise (Function.onFun Commute f)
      m : β
      h : ∀ (x : α), Membership.mem s x → Eq (f x) m
      ⊢ ∀ (x : β), Membership.mem (Multiset.map f s.val) x → Eq x m
    -/
  · simpa using h
    /-
      🎉 no goals
    -/


@[to_additive]
theorem noncommProd_commute (s : Finset α) (f : α → β) (comm) (y : β)
    (h : ∀ x ∈ s, Commute y (f x)) : Commute y (s.noncommProd f comm) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    y : β
    h : ∀ (x : α), Membership.mem s x → Commute y (f x)
    ⊢ Commute y (s.noncommProd f comm)
  -/
  apply Multiset.noncommProd_commute
  /-
    case h
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    y : β
    h : ∀ (x : α), Membership.mem s x → Commute y (f x)
    ⊢ ∀ (x : β), Membership.mem (Multiset.map f s.val) x → Commute y x
  -/
  intro y
  /-
    case h
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    y✝ : β
    h : ∀ (x : α), Membership.mem s x → Commute y✝ (f x)
    y : β
    ⊢ Membership.mem (Multiset.map f s.val) y → Commute y✝ y
  -/
  rw [Multiset.mem_map]
  /-
    case h
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    y✝ : β
    h : ∀ (x : α), Membership.mem s x → Commute y✝ (f x)
    y : β
    ⊢ (Exists fun a => And (Membership.mem s.val a) (Eq (f a) y)) → Commute y✝ y
  -/
  rintro ⟨x, ⟨hx, rfl⟩⟩
  /-
    case h.intro.intro
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f : α → β
    comm : (↑s).Pairwise (Function.onFun Commute f)
    y : β
    h : ∀ (x : α), Membership.mem s x → Commute y (f x)
    x : α
    hx : Membership.mem s.val x
    ⊢ Commute y (f x)
  -/
  exact h x hx
  /-
    🎉 no goals
  -/


theorem mul_noncommProd_erase [DecidableEq α] (s : Finset α) {a : α} (h : a ∈ s) (f : α → β) (comm)
    (comm' := fun _ hx _ hy hxy ↦ comm (s.mem_of_mem_erase hx) (s.mem_of_mem_erase hy) hxy) :
    f a * (s.erase a).noncommProd f comm' = s.noncommProd f comm := by
  classical
  simpa only [← Multiset.map_erase_of_mem _ _ h] using
    Multiset.mul_noncommProd_erase (s.1.map f) (Multiset.mem_map_of_mem f h) _


theorem noncommProd_erase_mul [DecidableEq α] (s : Finset α) {a : α} (h : a ∈ s) (f : α → β) (comm)
    (comm' := fun _ hx _ hy hxy ↦ comm (s.mem_of_mem_erase hx) (s.mem_of_mem_erase hy) hxy) :
    (s.erase a).noncommProd f comm' * f a = s.noncommProd f comm := by
  classical
  simpa only [← Multiset.map_erase_of_mem _ _ h] using
    Multiset.noncommProd_erase_mul (s.1.map f) (Multiset.mem_map_of_mem f h) _


@[to_additive]
theorem noncommProd_eq_prod {β : Type*} [CommMonoid β] (s : Finset α) (f : α → β) :
    (noncommProd s f fun _ _ _ _ _ => Commute.all _ _) = s.prod f := by
  /-
    α : Type u_3
    β : Type u_6
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    ⊢ Eq (s.noncommProd f ⋯) (s.prod f)
  -/
  induction' s using Finset.cons_induction_on with a s ha IH
    /-
      case h₁
      α : Type u_3
      β : Type u_6
      inst✝ : CommMonoid β
      f : α → β
      ⊢ Eq (EmptyCollection.emptyCollection.noncommProd f ⋯) (EmptyCollection.emptyC …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_3
      β : Type u_6
      inst✝ : CommMonoid β
      f : α → β
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      IH : Eq (s.noncommProd f ⋯) (s.prod f)
      ⊢ Eq ((Finset.cons a s ha).noncommProd f ⋯) ((Finset.cons a s ha).prod f)
    -/
  · simp [ha, IH]
    /-
      🎉 no goals
    -/


/-- The non-commutative version of `Finset.prod_union` -/
@[to_additive "The non-commutative version of `Finset.sum_union`"]
theorem noncommProd_union_of_disjoint [DecidableEq α] {s t : Finset α} (h : Disjoint s t)
    (f : α → β) (comm : { x | x ∈ s ∪ t }.Pairwise (Commute on f)) :
    noncommProd (s ∪ t) f comm =
      noncommProd s f (comm.mono <| coe_subset.2 subset_union_left) *
        noncommProd t f (comm.mono <| coe_subset.2 subset_union_right) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : Monoid β
    inst✝ : DecidableEq α
    s t : Finset α
    h : Disjoint s t
    f : α → β
    comm : (setOf fun x => Membership.mem (Union.union s t) x).Pairwise (Function. …
    ⊢ Eq ((Union.union s t).noncommProd f comm) (HMul.hMul (s.noncommProd f ⋯) (t. …
  -/
  obtain ⟨sl, sl', rfl⟩ := exists_list_nodup_eq s
  /-
    case intro.intro
    α : Type u_3
    β : Type u_4
    inst✝¹ : Monoid β
    inst✝ : DecidableEq α
    t : Finset α
    f : α → β
    sl : List α
    sl' : sl.Nodup
    h : Disjoint sl.toFinset t
    comm : (setOf fun x => Membership.mem (Union.union sl.toFinset t) x).Pairwise  …
    ⊢ Eq ((Union.union sl.toFinset t).noncommProd f comm) (HMul.hMul (sl.toFinset. …
  -/
  obtain ⟨tl, tl', rfl⟩ := exists_list_nodup_eq t
  /-
    case intro.intro.intro.intro
    α : Type u_3
    β : Type u_4
    inst✝¹ : Monoid β
    inst✝ : DecidableEq α
    f : α → β
    sl : List α
    sl' : sl.Nodup
    tl : List α
    tl' : tl.Nodup
    h : Disjoint sl.toFinset tl.toFinset
    comm : (setOf fun x => Membership.mem (Union.union sl.toFinset tl.toFinset) x) …
    ⊢ Eq ((Union.union sl.toFinset tl.toFinset).noncommProd f comm) (HMul.hMul (sl …
  -/
  rw [List.disjoint_toFinset_iff_disjoint] at h
  calc noncommProd (List.toFinset sl ∪ List.toFinset tl) f comm
     = noncommProd ⟨↑(sl ++ tl), Multiset.coe_nodup.2 (sl'.append tl' h)⟩ f
         (by convert comm; simp [Set.ext_iff]) := noncommProd_congr (by ext; simp) (by simp) _
   _ = noncommProd (List.toFinset sl) f (comm.mono <| coe_subset.2 subset_union_left) *
         noncommProd (List.toFinset tl) f (comm.mono <| coe_subset.2 subset_union_right) := by
    simp [noncommProd, List.dedup_eq_self.2 sl', List.dedup_eq_self.2 tl', h]


@[to_additive]
theorem noncommProd_mul_distrib_aux {s : Finset α} {f : α → β} {g : α → β}
    (comm_ff : (s : Set α).Pairwise (Commute on f))
    (comm_gg : (s : Set α).Pairwise (Commute on g))
    (comm_gf : (s : Set α).Pairwise fun x y => Commute (g x) (f y)) :
    (s : Set α).Pairwise fun x y => Commute ((f * g) x) ((f * g) y) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f g : α → β
    comm_ff : (↑s).Pairwise (Function.onFun Commute f)
    comm_gg : (↑s).Pairwise (Function.onFun Commute g)
    comm_gf : (↑s).Pairwise fun x y => Commute (g x) (f y)
    ⊢ (↑s).Pairwise fun x y => Commute (HMul.hMul f g x) (HMul.hMul f g y)
  -/
  intro x hx y hy h
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f g : α → β
    comm_ff : (↑s).Pairwise (Function.onFun Commute f)
    comm_gg : (↑s).Pairwise (Function.onFun Commute g)
    comm_gf : (↑s).Pairwise fun x y => Commute (g x) (f y)
    x : α
    hx : Membership.mem (↑s) x
    y : α
    hy : Membership.mem (↑s) y
    h : Ne x y
    ⊢ Commute (HMul.hMul f g x) (HMul.hMul f g y)
  -/
  apply Commute.mul_left <;> apply Commute.mul_right
    /-
      case hac.hab
      α : Type u_3
      β : Type u_4
      inst✝ : Monoid β
      s : Finset α
      f g : α → β
      comm_ff : (↑s).Pairwise (Function.onFun Commute f)
      comm_gg : (↑s).Pairwise (Function.onFun Commute g)
      comm_gf : (↑s).Pairwise fun x y => Commute (g x) (f y)
      x : α
      hx : Membership.mem (↑s) x
      y : α
      hy : Membership.mem (↑s) y
      h : Ne x y
      ⊢ Commute (f x) (f y)
    -/
  · exact comm_ff.of_refl hx hy
    /-
      🎉 no goals
    -/
    /-
      case hac.hac
      α : Type u_3
      β : Type u_4
      inst✝ : Monoid β
      s : Finset α
      f g : α → β
      comm_ff : (↑s).Pairwise (Function.onFun Commute f)
      comm_gg : (↑s).Pairwise (Function.onFun Commute g)
      comm_gf : (↑s).Pairwise fun x y => Commute (g x) (f y)
      x : α
      hx : Membership.mem (↑s) x
      y : α
      hy : Membership.mem (↑s) y
      h : Ne x y
      ⊢ Commute (f x) (g y)
    -/
  · exact (comm_gf hy hx h.symm).symm
    /-
      🎉 no goals
    -/
    /-
      case hbc.hab
      α : Type u_3
      β : Type u_4
      inst✝ : Monoid β
      s : Finset α
      f g : α → β
      comm_ff : (↑s).Pairwise (Function.onFun Commute f)
      comm_gg : (↑s).Pairwise (Function.onFun Commute g)
      comm_gf : (↑s).Pairwise fun x y => Commute (g x) (f y)
      x : α
      hx : Membership.mem (↑s) x
      y : α
      hy : Membership.mem (↑s) y
      h : Ne x y
      ⊢ Commute (g x) (f y)
    -/
  · exact comm_gf hx hy h
    /-
      🎉 no goals
    -/
    /-
      case hbc.hac
      α : Type u_3
      β : Type u_4
      inst✝ : Monoid β
      s : Finset α
      f g : α → β
      comm_ff : (↑s).Pairwise (Function.onFun Commute f)
      comm_gg : (↑s).Pairwise (Function.onFun Commute g)
      comm_gf : (↑s).Pairwise fun x y => Commute (g x) (f y)
      x : α
      hx : Membership.mem (↑s) x
      y : α
      hy : Membership.mem (↑s) y
      h : Ne x y
      ⊢ Commute (g x) (g y)
    -/
  · exact comm_gg.of_refl hx hy
    /-
      🎉 no goals
    -/


/-- The non-commutative version of `Finset.prod_mul_distrib` -/
@[to_additive "The non-commutative version of `Finset.sum_add_distrib`"]
theorem noncommProd_mul_distrib {s : Finset α} (f : α → β) (g : α → β) (comm_ff comm_gg comm_gf) :
    noncommProd s (f * g) (noncommProd_mul_distrib_aux comm_ff comm_gg comm_gf) =
      noncommProd s f comm_ff * noncommProd s g comm_gg := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    s : Finset α
    f g : α → β
    comm_ff : (↑s).Pairwise (Function.onFun Commute f)
    comm_gg : (↑s).Pairwise (Function.onFun Commute g)
    comm_gf : (↑s).Pairwise fun x y => Commute (g x) (f y)
    ⊢ Eq (s.noncommProd (HMul.hMul f g) ⋯) (HMul.hMul (s.noncommProd f comm_ff) (s …
  -/
  induction' s using Finset.cons_induction_on with x s hnmem ih
    /-
      case h₁
      α : Type u_3
      β : Type u_4
      inst✝ : Monoid β
      f g : α → β
      comm_ff : (↑EmptyCollection.emptyCollection).Pairwise (Function.onFun Commute f)
      comm_gg : (↑EmptyCollection.emptyCollection).Pairwise (Function.onFun Commute g)
      comm_gf : (↑EmptyCollection.emptyCollection).Pairwise fun x y => Commute (g x) …
      ⊢ Eq (EmptyCollection.emptyCollection.noncommProd (HMul.hMul f g) ⋯) (HMul.hMu …
    -/
  · simp
    /-
      🎉 no goals
    -/
  rw [Finset.noncommProd_cons, Finset.noncommProd_cons, Finset.noncommProd_cons, Pi.mul_apply,
    ih (comm_ff.mono fun _ => mem_cons_of_mem) (comm_gg.mono fun _ => mem_cons_of_mem)
      (comm_gf.mono fun _ => mem_cons_of_mem),
    (noncommProd_commute _ _ _ _ fun y hy => ?_).mul_mul_mul_comm]
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Monoid β
    f g : α → β
    x : α
    s : Finset α
    hnmem : Not (Membership.mem s x)
    ih : ∀ (comm_ff : (↑s).Pairwise (Function.onFun Commute f)) (comm_gg : (↑s).Pa …
    comm_ff : (↑(Finset.cons x s hnmem)).Pairwise (Function.onFun Commute f)
    comm_gg : (↑(Finset.cons x s hnmem)).Pairwise (Function.onFun Commute g)
    comm_gf : (↑(Finset.cons x s hnmem)).Pairwise fun x y => Commute (g x) (f y)
    y : α
    hy : Membership.mem s y
    ⊢ Commute (g x) (f y)
  -/
  exact comm_gf (mem_cons_self x s) (mem_cons_of_mem hy) (ne_of_mem_of_not_mem hy hnmem).symm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem noncommProd_mul_single [Fintype ι] [DecidableEq ι] (x : ∀ i, M i) :
    (univ.noncommProd (fun i => Pi.mulSingle i (x i)) fun i _ j _ _ =>
        Pi.mulSingle_apply_commute x i j) = x := by
  /-
    ι : Type u_2
    M : ι → Type u_6
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : (i : ι) → M i
    ⊢ Eq (Finset.univ.noncommProd (fun i => Pi.mulSingle i (x i)) ⋯) x
  -/
  ext i
  apply (univ.map_noncommProd (fun i ↦ MonoidHom.mulSingle M i (x i)) ?a
    (Pi.evalMonoidHom M i)).trans
  case a =>
    intro i _ j _ _
    exact Pi.mulSingle_apply_commute x i j
  /-
    case h
    ι : Type u_2
    M : ι → Type u_6
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : (i : ι) → M i
    i : ι
    ⊢ Eq (Finset.univ.noncommProd (fun i_1 => (Pi.evalMonoidHom M i) ((MonoidHom.m …
  -/
  convert (noncommProd_congr (insert_erase (mem_univ i)).symm _ _).trans _
    /-
      case h.convert_3
      ι : Type u_2
      M : ι → Type u_6
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      x : (i : ι) → M i
      i : ι
      ⊢ ι → M i
    -/
  · intro j
    /-
      case h.convert_3
      ι : Type u_2
      M : ι → Type u_6
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      x : (i : ι) → M i
      i j : ι
      ⊢ M i
    -/
    exact Pi.mulSingle j (x j) i
    /-
      🎉 no goals
    -/
    /-
      case h.convert_4
      ι : Type u_2
      M : ι → Type u_6
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      x : (i : ι) → M i
      i : ι
      ⊢ ∀ (x_1 : ι), Membership.mem (Insert.insert i (Finset.univ.erase i)) x_1 → Eq …
    -/
  · intro j _; dsimp
               /-
                 🎉 no goals
               -/
  · rw [noncommProd_insert_of_not_mem _ _ _ _ (not_mem_erase _ _),
      noncommProd_eq_pow_card (univ.erase i), one_pow, mul_one]
      /-
        case h.convert_8
        ι : Type u_2
        M : ι → Type u_6
        inst✝² : (i : ι) → Monoid (M i)
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        x : (i : ι) → M i
        i : ι
        ⊢ Eq (Pi.mulSingle i (x i) i) (x i)
      -/
    · simp only [MonoidHom.mulSingle_apply, ne_eq, Pi.mulSingle_eq_same]
      /-
        🎉 no goals
      -/
      /-
        case h.convert_8.h
        ι : Type u_2
        M : ι → Type u_6
        inst✝² : (i : ι) → Monoid (M i)
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        x : (i : ι) → M i
        i : ι
        ⊢ ∀ (x_1 : ι), Membership.mem (Finset.univ.erase i) x_1 → Eq (Pi.mulSingle x_1 …
      -/
    · intro j hj
      /-
        case h.convert_8.h
        ι : Type u_2
        M : ι → Type u_6
        inst✝² : (i : ι) → Monoid (M i)
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        x : (i : ι) → M i
        i j : ι
        hj : Membership.mem (Finset.univ.erase i) j
        ⊢ Eq (Pi.mulSingle j (x j) i) 1
      -/
      simp? at hj says simp only [mem_erase, ne_eq, mem_univ, and_true] at hj
      simp only [MonoidHom.mulSingle_apply, Pi.mulSingle, Function.update, Eq.ndrec, Pi.one_apply,
        ne_eq, dite_eq_right_iff]
      /-
        case h.convert_8.h
        ι : Type u_2
        M : ι → Type u_6
        inst✝² : (i : ι) → Monoid (M i)
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        x : (i : ι) → M i
        i j : ι
        hj : Not (Eq j i)
        ⊢ ∀ (h : Eq i j), Eq (Eq.rec (x j) ⋯) 1
      -/
      intro h
      /-
        case h.convert_8.h
        ι : Type u_2
        M : ι → Type u_6
        inst✝² : (i : ι) → Monoid (M i)
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        x : (i : ι) → M i
        i j : ι
        hj : Not (Eq j i)
        h : Eq i j
        ⊢ Eq (Eq.rec (x j) ⋯) 1
      -/
      simp [*] at *
      /-
        🎉 no goals
      -/


@[to_additive]
theorem _root_.MonoidHom.pi_ext [Finite ι] [DecidableEq ι] {f g : (∀ i, M i) →* γ}
    (h : ∀ i x, f (Pi.mulSingle i x) = g (Pi.mulSingle i x)) : f = g := by
  /-
    ι : Type u_2
    γ : Type u_5
    inst✝³ : Monoid γ
    M : ι → Type u_6
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : Finite ι
    inst✝ : DecidableEq ι
    f g : MonoidHom ((i : ι) → M i) γ
    h : ∀ (i : ι) (x : M i), Eq (f (Pi.mulSingle i x)) (g (Pi.mulSingle i x))
    ⊢ Eq f g
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_2
    γ : Type u_5
    inst✝³ : Monoid γ
    M : ι → Type u_6
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : Finite ι
    inst✝ : DecidableEq ι
    f g : MonoidHom ((i : ι) → M i) γ
    h : ∀ (i : ι) (x : M i), Eq (f (Pi.mulSingle i x)) (g (Pi.mulSingle i x))
    val✝ : Fintype ι
    ⊢ Eq f g
  -/
  ext x
  /-
    case intro.h
    ι : Type u_2
    γ : Type u_5
    inst✝³ : Monoid γ
    M : ι → Type u_6
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : Finite ι
    inst✝ : DecidableEq ι
    f g : MonoidHom ((i : ι) → M i) γ
    h : ∀ (i : ι) (x : M i), Eq (f (Pi.mulSingle i x)) (g (Pi.mulSingle i x))
    val✝ : Fintype ι
    x : (i : ι) → M i
    ⊢ Eq (f x) (g x)
  -/
  rw [← noncommProd_mul_single x, univ.map_noncommProd, univ.map_noncommProd]
  /-
    case intro.h
    ι : Type u_2
    γ : Type u_5
    inst✝³ : Monoid γ
    M : ι → Type u_6
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : Finite ι
    inst✝ : DecidableEq ι
    f g : MonoidHom ((i : ι) → M i) γ
    h : ∀ (i : ι) (x : M i), Eq (f (Pi.mulSingle i x)) (g (Pi.mulSingle i x))
    val✝ : Fintype ι
    x : (i : ι) → M i
    ⊢ Eq (Finset.univ.noncommProd (fun i => f (Pi.mulSingle i (x i))) ⋯) (Finset.u …
  -/
  congr 1 with i; exact h i (x i)
                  /-
                    🎉 no goals
                  -/


