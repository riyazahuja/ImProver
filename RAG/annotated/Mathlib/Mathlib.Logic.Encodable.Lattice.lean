theorem iSup_decode₂ [CompleteLattice α] (f : β → α) :
    ⨆ (i : ℕ) (b ∈ decode₂ β i), f b = (⨆ b, f b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Encodable β
    inst✝ : CompleteLattice α
    f : β → α
    ⊢ Eq (iSup fun i => iSup fun b => iSup fun h => f b) (iSup fun b => f b)
  -/
  rw [iSup_comm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Encodable β
    inst✝ : CompleteLattice α
    f : β → α
    ⊢ Eq (iSup fun j => iSup fun i => iSup fun h => f j) (iSup fun b => f b)
  -/
  simp only [mem_decode₂, iSup_iSup_eq_right]
  /-
    🎉 no goals
  -/


theorem iUnion_decode₂ (f : β → Set α) : ⋃ (i : ℕ) (b ∈ decode₂ β i), f b = ⋃ b, f b :=
  iSup_decode₂ f


@[elab_as_elim]
theorem iUnion_decode₂_cases {f : β → Set α} {C : Set α → Prop} (H0 : C ∅) (H1 : ∀ b, C (f b)) {n} :
    C (⋃ b ∈ decode₂ β n, f b) :=
  match decode₂ β n with
  | none => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : Encodable β
      f : β → Set α
      C : Set α → Prop
      H0 : C EmptyCollection.emptyCollection
      H1 : ∀ (b : β), C (f b)
      n : Nat
      ⊢ C (Set.iUnion fun b => Set.iUnion fun h => f b)
    -/
    simp only [Option.mem_def, iUnion_of_empty, iUnion_empty, reduceCtorEq]
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : Encodable β
      f : β → Set α
      C : Set α → Prop
      H0 : C EmptyCollection.emptyCollection
      H1 : ∀ (b : β), C (f b)
      n : Nat
      ⊢ C EmptyCollection.emptyCollection
    -/
    apply H0
    /-
      🎉 no goals
    -/
  | some b => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : Encodable β
      f : β → Set α
      C : Set α → Prop
      H0 : C EmptyCollection.emptyCollection
      H1 : ∀ (b : β), C (f b)
      n : Nat
      b : β
      ⊢ C (Set.iUnion fun b_1 => Set.iUnion fun h => f b_1)
    -/
    convert H1 b
    /-
      case h.e'_1
      α : Type u_1
      β : Type u_2
      inst✝ : Encodable β
      f : β → Set α
      C : Set α → Prop
      H0 : C EmptyCollection.emptyCollection
      H1 : ∀ (b : β), C (f b)
      n : Nat
      b : β
      ⊢ Eq (Set.iUnion fun b_1 => Set.iUnion fun h => f b_1) (f b)
    -/
    simp [Set.ext_iff]
    /-
      🎉 no goals
    -/


theorem iUnion_decode₂_disjoint_on {f : β → Set α} (hd : Pairwise (Disjoint on f)) :
    Pairwise (Disjoint on fun i => ⋃ b ∈ decode₂ β i, f b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Encodable β
    f : β → Set α
    hd : Pairwise (Function.onFun Disjoint f)
    ⊢ Pairwise (Function.onFun Disjoint fun i => Set.iUnion fun b => Set.iUnion fu …
  -/
  rintro i j ij
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Encodable β
    f : β → Set α
    hd : Pairwise (Function.onFun Disjoint f)
    i j : Nat
    ij : Ne i j
    ⊢ Function.onFun Disjoint (fun i => Set.iUnion fun b => Set.iUnion fun h => f  …
  -/
  refine disjoint_left.mpr fun x => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Encodable β
    f : β → Set α
    hd : Pairwise (Function.onFun Disjoint f)
    i j : Nat
    ij : Ne i j
    x : α
    ⊢ Membership.mem ((fun i => Set.iUnion fun b => Set.iUnion fun h => f b) i) x  …
  -/
  suffices ∀ a, encode a = i → x ∈ f a → ∀ b, encode b = j → x ∉ f b by simpa [decode₂_eq_some]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Encodable β
    f : β → Set α
    hd : Pairwise (Function.onFun Disjoint f)
    i j : Nat
    ij : Ne i j
    x : α
    ⊢ ∀ (a : β), Eq (Encodable.encode a) i → Membership.mem (f a) x → ∀ (b : β), E …
  -/
  rintro a rfl ha b rfl hb
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Encodable β
    f : β → Set α
    hd : Pairwise (Function.onFun Disjoint f)
    x : α
    a : β
    ha : Membership.mem (f a) x
    b : β
    ij : Ne (Encodable.encode a) (Encodable.encode b)
    hb : Membership.mem (f b) x
    ⊢ False
  -/
  exact (hd (mt (congr_arg encode) ij)).le_bot ⟨ha, hb⟩
  /-
    🎉 no goals
  -/


