/-- Inserting an `Option.none` anywhere in an enumeration yields another enumeration. -/
def insertNone (α : Type u) [FinEnum α] (i : Fin (card α + 1)) : FinEnum (Option α) where
  card := card α + 1
  equiv := equiv.optionCongr.trans <| finSuccEquiv' i |>.symm


/-- This is an arbitrary choice of insertion rank for a default instance.
It keeps the mapping of the existing `α`-inhabitants intact, modulo `Fin.castSucc`. -/
instance instFinEnumOptionLast (α : Type u) [FinEnum α] : FinEnum (Option α) :=
  insertNone α (Fin.last _)


/-- A recursor principle for finite-and-enumerable types, analogous to `Nat.rec`.
It effectively says that every `FinEnum` is either `Empty` or `Option α`, up to an `Equiv` mediated
by `Fin`s of equal cardinality.
In contrast to the `Fintype` case, data can be transported along such an `Equiv`.
Also, since order matters, the choice of element that gets replaced by `Option.none` has
to be provided for every step.

Since every `FinEnum` instance implies a `Fintype` instance and `Prop` is squashed already,
`Fintype.induction_empty_option` can be used if a `Prop` needs to be constructed.
Cf. `Data.Fintype.Option`
-/
def recEmptyOption {P : Type u → Sort v}
    (finChoice : (n : ℕ) → Fin (n + 1))
    (congr : {α β : Type u} → (_ : FinEnum α) → (_ : FinEnum β) → card β = card α → P α → P β)
    (empty : P PEmpty.{u + 1})
    (option : {α : Type u} → FinEnum α → P α → P (Option α))
    (α : Type u) [FinEnum α] :
    P α :=
  match cardeq : card α with
  | 0 => congr _ _ cardeq empty
  | n + 1 =>
    let fN := ULift.instFinEnum (α := Fin n)
    have : card (ULift.{u} <| Fin n) = n := card_ulift.trans card_fin
    congr (insertNone _ <| finChoice n) _
      (cardeq.trans <| congrArg Nat.succ this.symm) <|
        option fN (recEmptyOption finChoice congr empty option _)
termination_by card α


/--
For an empty type, the recursion principle evaluates to whatever `congr`
makes of the base case.
-/
theorem recEmptyOption_of_card_eq_zero {P : Type u → Sort v}
    (finChoice : (n : ℕ) → Fin (n + 1))
    (congr : {α β : Type u} → (_ : FinEnum α) → (_ : FinEnum β) → card β = card α → P α → P β)
    (empty : P PEmpty.{u + 1})
    (option : {α : Type u} → FinEnum α → P α → P (Option α))
    (α : Type u) [FinEnum α] (h : card α = 0) (_ : FinEnum PEmpty.{u + 1}) :
    recEmptyOption finChoice congr empty option α =
      congr _ _ (h.trans card_eq_zero.symm) empty := by
  /-
    P : Type u → Sort v
    finChoice : (n : Nat) → Fin (HAdd.hAdd n 1)
    congr : {α β : Type u} → (x : FinEnum α) → (x_1 : FinEnum β) → Eq (FinEnum.car …
    empty : P PEmpty.{u + 1}
    option : {α : Type u} → FinEnum α → P α → P (Option α)
    α : Type u
    inst✝ : FinEnum α
    h : Eq (FinEnum.card α) 0
    x✝ : FinEnum PEmpty.{u + 1}
    ⊢ Eq (FinEnum.recEmptyOption finChoice (fun {α β} => congr) empty (fun {α} =>  …
  -/
  unfold recEmptyOption
  /-
    P : Type u → Sort v
    finChoice : (n : Nat) → Fin (HAdd.hAdd n 1)
    congr : {α β : Type u} → (x : FinEnum α) → (x_1 : FinEnum β) → Eq (FinEnum.car …
    empty : P PEmpty.{u + 1}
    option : {α : Type u} → FinEnum α → P α → P (Option α)
    α : Type u
    inst✝ : FinEnum α
    h : Eq (FinEnum.card α) 0
    x✝ : FinEnum PEmpty.{u + 1}
    ⊢ Eq
        (FinEnum.recEmptyOption.match_1 (fun x => P α) (FinEnum.card α) (fun carde …
          let fN := ULift.instFinEnum;
          letFun ⋯ fun this => congr (FinEnum.insertNone (ULift.{u, 0} (Fin n)) ↑↑ …
        (congr x✝ inst✝ ⋯ empty)
  -/
  split
    /-
      case h_1
      P : Type u → Sort v
      finChoice : (n : Nat) → Fin (HAdd.hAdd n 1)
      congr : {α β : Type u} → (x : FinEnum α) → (x_1 : FinEnum β) → Eq (FinEnum.car …
      empty : P PEmpty.{u + 1}
      option : {α : Type u} → FinEnum α → P α → P (Option α)
      α : Type u
      inst✝ : FinEnum α
      h : Eq (FinEnum.card α) 0
      x✝ : FinEnum PEmpty.{u + 1}
      heq✝ : Eq (FinEnum.card α) 0
      ⊢ Eq (congr FinEnum.pempty inst✝ ⋯ empty) (congr x✝ inst✝ ⋯ empty)
    -/
  · congr 1; exact Subsingleton.allEq _ _
             /-
               🎉 no goals
             -/
    /-
      case h_2
      P : Type u → Sort v
      finChoice : (n : Nat) → Fin (HAdd.hAdd n 1)
      congr : {α β : Type u} → (x : FinEnum α) → (x_1 : FinEnum β) → Eq (FinEnum.car …
      empty : P PEmpty.{u + 1}
      option : {α : Type u} → FinEnum α → P α → P (Option α)
      α : Type u
      inst✝ : FinEnum α
      h : Eq (FinEnum.card α) 0
      x✝ : FinEnum PEmpty.{u + 1}
      n✝ : Nat
      heq✝ : Eq (FinEnum.card α) n✝.succ
      ⊢ Eq
          (let fN := ULift.instFinEnum;
          letFun ⋯ fun this => congr (FinEnum.insertNone (ULift.{u, 0} (Fin n✝)) ↑↑( …
          (congr x✝ inst✝ ⋯ empty)
    -/
  · exact Nat.noConfusion <| h.symm.trans ‹_›
    /-
      🎉 no goals
    -/


/--
For a type with positive `card`, the recursion principle evaluates to whatever
`congr` makes of the step result, where `Option.none` has been inserted into the
`(finChoice (card α - 1))`th rank of the enumeration.
-/
theorem recEmptyOption_of_card_pos {P : Type u → Sort v}
    (finChoice : (n : ℕ) → Fin (n + 1))
    (congr : {α β : Type u} → (_ : FinEnum α) → (_ : FinEnum β) → card β = card α → P α → P β)
    (empty : P PEmpty.{u + 1})
    (option : {α : Type u} → FinEnum α → P α → P (Option α))
    (α : Type u) [FinEnum α] (h : 0 < card α) :
    recEmptyOption finChoice congr empty option α =
      congr (insertNone _ <| finChoice (card α - 1)) ‹_›
        (congrArg (· + 1) card_fin |>.trans <| (card α).succ_pred_eq_of_pos h).symm
        (option ULift.instFinEnum <|
          recEmptyOption finChoice congr empty option (ULift.{u} <| Fin (card α - 1))) := by
  /-
    P : Type u → Sort v
    finChoice : (n : Nat) → Fin (HAdd.hAdd n 1)
    congr : {α β : Type u} → (x : FinEnum α) → (x_1 : FinEnum β) → Eq (FinEnum.car …
    empty : P PEmpty.{u + 1}
    option : {α : Type u} → FinEnum α → P α → P (Option α)
    α : Type u
    inst✝ : FinEnum α
    h : LT.lt 0 (FinEnum.card α)
    ⊢ Eq (FinEnum.recEmptyOption finChoice (fun {α β} => congr) empty (fun {α} =>  …
  -/
  conv => lhs; unfold recEmptyOption
  /-
    P : Type u → Sort v
    finChoice : (n : Nat) → Fin (HAdd.hAdd n 1)
    congr : {α β : Type u} → (x : FinEnum α) → (x_1 : FinEnum β) → Eq (FinEnum.car …
    empty : P PEmpty.{u + 1}
    option : {α : Type u} → FinEnum α → P α → P (Option α)
    α : Type u
    inst✝ : FinEnum α
    h : LT.lt 0 (FinEnum.card α)
    ⊢ Eq
        (FinEnum.recEmptyOption.match_1 (fun x => P α) (FinEnum.card α) (fun carde …
          let fN := ULift.instFinEnum;
          letFun ⋯ fun this => congr (FinEnum.insertNone (ULift.{u, 0} (Fin n)) ↑↑ …
        (congr (FinEnum.insertNone (ULift.{u, 0} (Fin (HSub.hSub (FinEnum.card α)  …
  -/
  split
    /-
      case h_1
      P : Type u → Sort v
      finChoice : (n : Nat) → Fin (HAdd.hAdd n 1)
      congr : {α β : Type u} → (x : FinEnum α) → (x_1 : FinEnum β) → Eq (FinEnum.car …
      empty : P PEmpty.{u + 1}
      option : {α : Type u} → FinEnum α → P α → P (Option α)
      α : Type u
      inst✝ : FinEnum α
      h : LT.lt 0 (FinEnum.card α)
      heq✝ : Eq (FinEnum.card α) 0
      ⊢ Eq (congr FinEnum.pempty inst✝ ⋯ empty) (congr (FinEnum.insertNone (ULift.{u …
    -/
  · exact absurd (‹_› ▸ h) (card α).lt_irrefl
    /-
      🎉 no goals
    -/
    /-
      case h_2
      P : Type u → Sort v
      finChoice : (n : Nat) → Fin (HAdd.hAdd n 1)
      congr : {α β : Type u} → (x : FinEnum α) → (x_1 : FinEnum β) → Eq (FinEnum.car …
      empty : P PEmpty.{u + 1}
      option : {α : Type u} → FinEnum α → P α → P (Option α)
      α : Type u
      inst✝ : FinEnum α
      h : LT.lt 0 (FinEnum.card α)
      n✝ : Nat
      heq✝ : Eq (FinEnum.card α) n✝.succ
      ⊢ Eq
          (let fN := ULift.instFinEnum;
          letFun ⋯ fun this => congr (FinEnum.insertNone (ULift.{u, 0} (Fin n✝)) ↑↑( …
          (congr (FinEnum.insertNone (ULift.{u, 0} (Fin (HSub.hSub (FinEnum.card α)  …
    -/
  · rcases Nat.succ.inj <| (card α).succ_pred_eq_of_pos h |>.trans ‹_› with rfl; rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- A recursor principle for finite-and-enumerable types, analogous to `Nat.recOn`.
It effectively says that every `FinEnum` is either `Empty` or `Option α`, up to an `Equiv` mediated
by `Fin`s of equal cardinality.
In contrast to the `Fintype` case, data can be transported along such an `Equiv`.
Also, since order matters, the choice of element that gets replaced by `Option.none` has
to be provided for every step.
-/
abbrev recOnEmptyOption {P : Type u → Sort v}
    {α : Type u} (aenum : FinEnum α)
    (finChoice : (n : ℕ) → Fin (n + 1))
    (congr : {α β : Type u} → (_ : FinEnum α) → (_ : FinEnum β) → card β = card α → P α → P β)
    (empty : P PEmpty.{u + 1})
    (option : {α : Type u} → FinEnum α → P α → P (Option α)) :
    P α :=
  @recEmptyOption P finChoice congr empty option α aenum


