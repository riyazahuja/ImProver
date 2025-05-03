/-- The constructors for the naturals -/
inductive Natα : Type
  | zero : Natα
  | succ : Natα


instance : Inhabited Natα :=
  ⟨Natα.zero⟩


/-- The arity of the constructors for the naturals, `zero` takes no arguments, `succ` takes one -/
def Natβ : Natα → Type
  | Natα.zero => Empty
  | Natα.succ => Unit


instance : Inhabited (Natβ Natα.succ) :=
  ⟨()⟩


/-- The isomorphism from the naturals to its corresponding `WType` -/
@[simp]
def ofNat : ℕ → WType Natβ
  | Nat.zero => ⟨Natα.zero, Empty.elim⟩
  | Nat.succ n => ⟨Natα.succ, fun _ ↦ ofNat n⟩


/-- The isomorphism from the `WType` of the naturals to the naturals -/
@[simp]
def toNat : WType Natβ → ℕ
  | WType.mk Natα.zero _ => 0
  | WType.mk Natα.succ f => (f ()).toNat.succ


theorem leftInverse_nat : Function.LeftInverse ofNat toNat
  | WType.mk Natα.zero f => by
    /-
      f : WType.Natβ WType.Natα.zero → WType WType.Natβ
      ⊢ Eq (WType.ofNat (WType.mk WType.Natα.zero f).toNat) (WType.mk WType.Natα.zer …
    -/
    rw [toNat, ofNat]
    /-
      f : WType.Natβ WType.Natα.zero → WType WType.Natβ
      ⊢ Eq (WType.mk WType.Natα.zero Empty.elim) (WType.mk WType.Natα.zero f)
    -/
    congr
    /-
      case e_f
      f : WType.Natβ WType.Natα.zero → WType WType.Natβ
      ⊢ Eq Empty.elim f
    -/
    ext x
    /-
      case e_f.h
      f : WType.Natβ WType.Natα.zero → WType WType.Natβ
      x : Empty
      ⊢ Eq x.elim (f x)
    -/
    cases x
    /-
      🎉 no goals
    -/
  | WType.mk Natα.succ f => by
    /-
      f : WType.Natβ WType.Natα.succ → WType WType.Natβ
      ⊢ Eq (WType.ofNat (WType.mk WType.Natα.succ f).toNat) (WType.mk WType.Natα.suc …
    -/
    simp only [toNat, ofNat, leftInverse_nat (f ()), mk.injEq, heq_eq_eq, true_and]
    /-
      f : WType.Natβ WType.Natα.succ → WType WType.Natβ
      ⊢ Eq (fun x => f Unit.unit) f
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem rightInverse_nat : Function.RightInverse ofNat toNat
  | Nat.zero => rfl
                     /-
                       n : Nat
                       ⊢ Eq (WType.ofNat n.succ).toNat n.succ
                     -/
  | Nat.succ n => by rw [ofNat, toNat, rightInverse_nat n]
                     /-
                       🎉 no goals
                     -/


/-- The naturals are equivalent to their associated `WType` -/
def equivNat : WType Natβ ≃ ℕ where
  toFun := toNat
  invFun := ofNat
  left_inv := leftInverse_nat
  right_inv := rightInverse_nat


/-- `WType.Natα` is equivalent to `PUnit ⊕ PUnit`.
This is useful when considering the associated polynomial endofunctor.
-/
@[simps]
def NatαEquivPUnitSumPUnit : Natα ≃ PUnit.{u + 1} ⊕ PUnit where
  toFun c :=
    match c with
    | Natα.zero => inl unit
    | Natα.succ => inr unit
  invFun b :=
    match b with
    | inl _ => Natα.zero
    | inr _ => Natα.succ
  left_inv c :=
    match c with
    | Natα.zero => rfl
    | Natα.succ => rfl
  right_inv b :=
    match b with
    | inl _ => rfl
    | inr _ => rfl


/-- The constructors for lists.
There is "one constructor `cons x` for each `x : γ`",
since we view `List γ` as
```
| nil : List γ
| cons x₀ : List γ → List γ
| cons x₁ : List γ → List γ
|   ⋮      γ many times
```
-/
inductive Listα : Type u
  | nil : Listα
  | cons : γ → Listα


instance : Inhabited (Listα γ) :=
  ⟨Listα.nil⟩


/-- The arities of each constructor for lists, `nil` takes no arguments, `cons hd` takes one -/
def Listβ : Listα γ → Type u
  | Listα.nil => PEmpty
  | Listα.cons _ => PUnit


instance (hd : γ) : Inhabited (Listβ γ (Listα.cons hd)) :=
  ⟨PUnit.unit⟩


/-- The isomorphism from lists to the `WType` construction of lists -/
@[simp]
def ofList : List γ → WType (Listβ γ)
  | List.nil => ⟨Listα.nil, PEmpty.elim⟩
  | List.cons hd tl => ⟨Listα.cons hd, fun _ ↦ ofList tl⟩


/-- The isomorphism from the `WType` construction of lists to lists -/
@[simp]
def toList : WType (Listβ γ) → List γ
  | WType.mk Listα.nil _ => []
  | WType.mk (Listα.cons hd) f => hd :: (f PUnit.unit).toList


theorem leftInverse_list : Function.LeftInverse (ofList γ) (toList _)
  | WType.mk Listα.nil f => by
    /-
      γ : Type u
      f : WType.Listβ γ WType.Listα.nil → WType (WType.Listβ γ)
      ⊢ Eq (WType.ofList γ (WType.toList γ (WType.mk WType.Listα.nil f))) (WType.mk  …
    -/
    simp only [toList, ofList, mk.injEq, heq_eq_eq, true_and]
    /-
      γ : Type u
      f : WType.Listβ γ WType.Listα.nil → WType (WType.Listβ γ)
      ⊢ Eq PEmpty.elim f
    -/
    ext x
    /-
      case h
      γ : Type u
      f : WType.Listβ γ WType.Listα.nil → WType (WType.Listβ γ)
      x : PEmpty.{u + 1}
      ⊢ Eq x.elim (f x)
    -/
    cases x
    /-
      🎉 no goals
    -/
  | WType.mk (Listα.cons x) f => by
    /-
      γ : Type u
      x : γ
      f : WType.Listβ γ (WType.Listα.cons x) → WType (WType.Listβ γ)
      ⊢ Eq (WType.ofList γ (WType.toList γ (WType.mk (WType.Listα.cons x) f))) (WTyp …
    -/
    simp only [toList, ofList, leftInverse_list (f PUnit.unit), mk.injEq, heq_eq_eq, true_and]
    /-
      γ : Type u
      x : γ
      f : WType.Listβ γ (WType.Listα.cons x) → WType (WType.Listβ γ)
      ⊢ Eq (fun x => f PUnit.unit) f
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem rightInverse_list : Function.RightInverse (ofList γ) (toList _)
  | List.nil => rfl
                          /-
                            γ : Type u
                            hd : γ
                            tl : List γ
                            ⊢ Eq (WType.toList γ (WType.ofList γ (List.cons hd tl))) (List.cons hd tl)
                          -/
  | List.cons hd tl => by simp [rightInverse_list tl]
                          /-
                            🎉 no goals
                          -/


/-- Lists are equivalent to their associated `WType` -/
def equivList : WType (Listβ γ) ≃ List γ where
  toFun := toList _
  invFun := ofList _
  left_inv := leftInverse_list _
  right_inv := rightInverse_list _


/-- `WType.Listα` is equivalent to `γ` with an extra point.
This is useful when considering the associated polynomial endofunctor
-/
def ListαEquivPUnitSum : Listα γ ≃ PUnit.{v + 1} ⊕ γ where
  toFun c :=
    match c with
    | Listα.nil => Sum.inl PUnit.unit
    | Listα.cons x => Sum.inr x
  invFun := Sum.elim (fun _ ↦ Listα.nil) Listα.cons
  left_inv c :=
    match c with
    | Listα.nil => rfl
    | Listα.cons _ => rfl
  right_inv x :=
    match x with
    | Sum.inl PUnit.unit => rfl
    | Sum.inr _ => rfl


