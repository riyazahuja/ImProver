/-- Isomorphism between strict and lazy lists. -/
@[deprecated "No deprecation message was provided."  (since := "2024-07-22")]
def listEquivLazyList (α : Type*) : List α ≃ LazyList α where
  toFun := LazyList.ofList
  invFun := LazyList.toList
  right_inv := by
    /-
      α : Type u_1
      ⊢ Function.RightInverse LazyList.toList LazyList.ofList
    -/
    intro xs
    /-
      α : Type u_1
      xs : LazyList α
      ⊢ Eq (LazyList.ofList xs.toList) xs
    -/
    induction xs using toList.induct
      /-
        case case1
        α : Type u_1
        ⊢ Eq (LazyList.ofList LazyList.nil.toList) LazyList.nil
      -/
    /-
      α : Type u_1
      ⊢ Function.LeftInverse LazyList.toList LazyList.ofList
    -/
    · simp [toList, ofList]
    /-
      α : Type u_1
      xs : List α
      ⊢ Eq (LazyList.ofList xs).toList xs
    -/
      /-
        🎉 no goals
      -/
      /-
        case nil
        α : Type u_1
        ⊢ Eq (LazyList.ofList List.nil).toList List.nil
      -/
      /-
        case case2
        α : Type u_1
        h✝ : α
        t✝ : Thunk (LazyList α)
        ih1✝ : Eq (LazyList.ofList t✝.get.toList) t✝.get
        ⊢ Eq (LazyList.ofList (LazyList.cons h✝ t✝).toList) (LazyList.cons h✝ t✝)
      -/
      /-
        🎉 no goals
      -/
      /-
        case cons
        α : Type u_1
        head✝ : α
        tail✝ : List α
        tail_ih✝ : Eq (LazyList.ofList tail✝).toList tail✝
        ⊢ Eq (LazyList.ofList (List.cons head✝ tail✝)).toList (List.cons head✝ tail✝)
      -/
    · simp [toList, ofList, *]; rfl
      /-
        🎉 no goals
      -/
                                /-
                                  🎉 no goals
                                -/
  left_inv := by
    intro xs
    induction xs
    · simp [toList, ofList]
    · simpa [ofList, toList]


@[deprecated "No deprecation message was provided."  (since := "2024-07-22")]
instance : Traversable LazyList where
  map := @LazyList.traverse Id _
  traverse := @LazyList.traverse


@[deprecated "No deprecation message was provided."  (since := "2024-07-22")]
instance : LawfulTraversable LazyList := by
  /-
    ⊢ LawfulTraversable LazyList
  -/
  apply Equiv.isLawfulTraversable' listEquivLazyList <;> intros <;> ext <;> rename_i f xs
  · induction xs using LazyList.rec with
    | nil =>
      simp only [Functor.map, LazyList.traverse, pure, Equiv.map, listEquivLazyList,
        Equiv.coe_fn_symm_mk, toList, List.map_nil, Equiv.coe_fn_mk, ofList]
    | cons =>
      simpa only [Functor.map, LazyList.traverse, Seq.seq, Equiv.map, listEquivLazyList,
        Equiv.coe_fn_symm_mk, toList, List.map_cons, Equiv.coe_fn_mk, ofList, cons.injEq, true_and]
    | mk _ ih => ext; apply ih
  · simp only [Functor.mapConst, comp, Equiv.map, listEquivLazyList, Equiv.coe_fn_symm_mk,
      List.map_eq_map, List.map_const, Equiv.coe_fn_mk]
    induction xs using LazyList.rec with
    | nil => simp [LazyList.traverse, pure, Functor.map, toList, ofList]
    | cons =>
      simpa [toList, ofList, LazyList.traverse, Seq.seq, Functor.map, cons.injEq, true_and]
    | mk _ ih => congr; apply ih
    /-
      case h₂.h
      F✝ : Type u_1 → Type u_1
      inst✝¹ : Applicative F✝
      inst✝ : LawfulApplicative F✝
      α✝ β✝ : Type u_1
      f : α✝ → F✝ β✝
      xs : LazyList α✝
      ⊢ Eq (Traversable.traverse f xs) (Equiv.traverse LazyList.listEquivLazyList f  …
    -/
  · simp only [traverse, Equiv.traverse, listEquivLazyList, Equiv.coe_fn_mk, Equiv.coe_fn_symm_mk]
    induction xs using LazyList.rec with
    | nil => simp only [LazyList.traverse, toList, List.traverse, map_pure, ofList]
    | cons _ tl ih =>
      replace ih : tl.get.traverse f = ofList <$> tl.get.toList.traverse f := ih
      simp [traverse.eq_2, ih, Functor.map_map, seq_map_assoc, toList, List.traverse, map_seq,
        Function.comp_def, Thunk.pure, ofList]
    | mk _ ih => apply ih


@[deprecated "No deprecation message was provided."  (since := "2024-07-22"), simp]
theorem bind_singleton {α} (x : LazyList α) : x.bind singleton = x := by
  induction x using LazyList.rec (motive_2 := fun xs => xs.get.bind singleton = xs.get) with
  | nil => simp [LazyList.bind]
  | cons h t ih =>
    simp only [LazyList.bind, singleton, append, Thunk.get_pure, Thunk.get_mk, cons.injEq, true_and]
    ext
    simp [ih]
  | mk f ih => simp_all


@[deprecated "No deprecation message was provided."  (since := "2024-07-22")]
                                   /-
                                     ⊢ ∀ {α β : Type u_1} (x : α) (y : LazyList β), Eq (Functor.mapConst x y) (Func …
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
    /-
      ⊢ ∀ {α : Type u_1} (x : LazyList α), Eq (Functor.map id x) x
    -/
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
instance : LawfulMonad LazyList := LawfulMonad.mk'
                                   /-
                                     🎉 no goals
                                   -/
  (id_map := by
    intro α xs
    /-
      ⊢ ∀ {α β : Type u_1} (x : α) (f : α → LazyList β), Eq (Bind.bind (Pure.pure x) …
    -/
    induction xs using LazyList.rec (motive_2 := fun xs => id <$> xs.get = xs) with
    /-
      α✝ β✝ : Type u_1
      x✝ : α✝
      f✝ : α✝ → LazyList β✝
      ⊢ Eq (Bind.bind (Pure.pure x✝) f✝) (f✝ x✝)
    -/
    | nil => simp only [Functor.map, comp_id, LazyList.bind]
    /-
      α✝ β✝ : Type u_1
      x✝ : α✝
      f✝ : α✝ → LazyList β✝
      ⊢ Eq ((f✝ x✝).append { fn := fun x => LazyList.nil }) (f✝ x✝)
    -/
    | cons h t _ => simp only [Functor.map, comp_id, bind_singleton]
    /-
      🎉 no goals
    -/
    | mk f _ => ext; simp_all)
    /-
      ⊢ ∀ {α β γ : Type u_1} (x : LazyList α) (f : α → LazyList β) (g : β → LazyList …
    -/
  (pure_bind := by
    intros
    simp only [bind, pure, singleton, LazyList.bind, append, Thunk.pure, Thunk.get]
    apply append_nil)
  (bind_assoc := by
    intro _ _ _ xs _ _
    induction xs using LazyList.rec with
    | nil => simp only [bind, LazyList.bind]
    | cons => simp only [bind, LazyList.bind, append_bind]; congr
    /-
      ⊢ ∀ {α β : Type u_1} (f : α → β) (x : LazyList α), Eq (Bind.bind x fun y => Pu …
    -/
    | mk _ ih => congr; funext; apply ih)
    /-
      α✝ β✝ : Type u_1
      f : α✝ → β✝
      xs : LazyList α✝
      ⊢ Eq (Bind.bind xs fun y => Pure.pure (f y)) (Functor.map f xs)
    -/
  (bind_pure_comp := by
    intro _ _ f xs
    simp only [bind, Functor.map, pure, singleton]
    induction xs using LazyList.traverse.induct (m := @Id) (f := f) with
    | case1 =>
      simp only [Thunk.pure, LazyList.bind, LazyList.traverse, Id.pure_eq]
    | case2 _ _ ih =>
      simp only [Thunk.pure, LazyList.bind, append, Thunk.get_mk, comp_apply, ← ih]
      simp only [Thunk.get, append, singleton, Thunk.pure])


