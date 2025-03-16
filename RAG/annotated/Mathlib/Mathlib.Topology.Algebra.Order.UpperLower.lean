/-- Ad hoc class stating that the closure of an upper set is an upper set. This is used to state
lemmas that do not mention algebraic operations for both the additive and multiplicative versions
simultaneously. If you find a satisfying replacement for this typeclass, please remove it! -/
class HasUpperLowerClosure (α : Type*) [TopologicalSpace α] [Preorder α] : Prop where
  isUpperSet_closure : ∀ s : Set α, IsUpperSet s → IsUpperSet (closure s)
  isLowerSet_closure : ∀ s : Set α, IsLowerSet s → IsLowerSet (closure s)
  isOpen_upperClosure : ∀ s : Set α, IsOpen s → IsOpen (upperClosure s : Set α)
  isOpen_lowerClosure : ∀ s : Set α, IsOpen s → IsOpen (lowerClosure s : Set α)


@[to_additive]
instance (priority := 100) OrderedCommGroup.to_hasUpperLowerClosure [OrderedCommGroup α]
    [ContinuousConstSMul α α] : HasUpperLowerClosure α where
  isUpperSet_closure s h x y hxy hx :=
    closure_mono (h.smul_subset <| one_le_div'.2 hxy) <| by
      /-
        α : Type u_1
        inst✝² : TopologicalSpace α
        inst✝¹ : OrderedCommGroup α
        inst✝ : ContinuousConstSMul α α
        s : Set α
        h : IsUpperSet s
        x y : α
        hxy : LE.le x y
        hx : Membership.mem (closure s) x
        ⊢ Membership.mem (closure (HSMul.hSMul (HDiv.hDiv y x) s)) y
      -/
      rw [closure_smul]
      /-
        α : Type u_1
        inst✝² : TopologicalSpace α
        inst✝¹ : OrderedCommGroup α
        inst✝ : ContinuousConstSMul α α
        s : Set α
        h : IsUpperSet s
        x y : α
        hxy : LE.le x y
        hx : Membership.mem (closure s) x
        ⊢ Membership.mem (HSMul.hSMul (HDiv.hDiv y x) (closure s)) y
      -/
      exact ⟨x, hx, div_mul_cancel _ _⟩
      /-
        🎉 no goals
      -/
  isLowerSet_closure s h x y hxy hx :=
    closure_mono (h.smul_subset <| div_le_one'.2 hxy) <| by
      /-
        α : Type u_1
        inst✝² : TopologicalSpace α
        inst✝¹ : OrderedCommGroup α
        inst✝ : ContinuousConstSMul α α
        s : Set α
        h : IsLowerSet s
        x y : α
        hxy : LE.le y x
        hx : Membership.mem (closure s) x
        ⊢ Membership.mem (closure (HSMul.hSMul (HDiv.hDiv y x) s)) y
      -/
      rw [closure_smul]
      /-
        α : Type u_1
        inst✝² : TopologicalSpace α
        inst✝¹ : OrderedCommGroup α
        inst✝ : ContinuousConstSMul α α
        s : Set α
        h : IsLowerSet s
        x y : α
        hxy : LE.le y x
        hx : Membership.mem (closure s) x
        ⊢ Membership.mem (HSMul.hSMul (HDiv.hDiv y x) (closure s)) y
      -/
      exact ⟨x, hx, div_mul_cancel _ _⟩
      /-
        🎉 no goals
      -/
  isOpen_upperClosure s hs := by
    /-
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderedCommGroup α
      inst✝ : ContinuousConstSMul α α
      s : Set α
      hs : IsOpen s
      ⊢ IsOpen ↑(upperClosure s)
    -/
    rw [← mul_one s, ← mul_upperClosure]
    /-
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderedCommGroup α
      inst✝ : ContinuousConstSMul α α
      s : Set α
      hs : IsOpen s
      ⊢ IsOpen (HMul.hMul s ↑(upperClosure 1))
    -/
    exact hs.mul_right
    /-
      🎉 no goals
    -/
  isOpen_lowerClosure s hs := by
    /-
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderedCommGroup α
      inst✝ : ContinuousConstSMul α α
      s : Set α
      hs : IsOpen s
      ⊢ IsOpen ↑(lowerClosure s)
    -/
    rw [← mul_one s, ← mul_lowerClosure]
    /-
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderedCommGroup α
      inst✝ : ContinuousConstSMul α α
      s : Set α
      hs : IsOpen s
      ⊢ IsOpen (HMul.hMul s ↑(lowerClosure 1))
    -/
    exact hs.mul_right
    /-
      🎉 no goals
    -/


protected theorem IsUpperSet.closure : IsUpperSet s → IsUpperSet (closure s) :=
  HasUpperLowerClosure.isUpperSet_closure _


protected theorem IsLowerSet.closure : IsLowerSet s → IsLowerSet (closure s) :=
  HasUpperLowerClosure.isLowerSet_closure _


protected theorem IsOpen.upperClosure : IsOpen s → IsOpen (upperClosure s : Set α) :=
  HasUpperLowerClosure.isOpen_upperClosure _


protected theorem IsOpen.lowerClosure : IsOpen s → IsOpen (lowerClosure s : Set α) :=
  HasUpperLowerClosure.isOpen_lowerClosure _


instance : HasUpperLowerClosure αᵒᵈ where
  isUpperSet_closure := @IsLowerSet.closure α _ _ _
  isLowerSet_closure := @IsUpperSet.closure α _ _ _
  isOpen_upperClosure := @IsOpen.lowerClosure α _ _ _
  isOpen_lowerClosure := @IsOpen.upperClosure α _ _ _

/-
Note: `s.OrdConnected` does not imply `(closure s).OrdConnected`, as we can see by taking
`s := Ioo 0 1 × Ioo 1 2 ∪ Ioo 2 3 × Ioo 0 1` because then
`closure s = Icc 0 1 × Icc 1 2 ∪ Icc 2 3 × Icc 0 1` is not order-connected as
`(1, 1) ∈ closure s`, `(2, 1) ∈ closure s` but `Icc (1, 1) (2, 1) ⊈ closure s`.

`s` looks like
```
xxooooo
xxooooo
oooooxx
oooooxx
```
-/

protected theorem IsUpperSet.interior (h : IsUpperSet s) : IsUpperSet (interior s) := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : HasUpperLowerClosure α
    s : Set α
    h : IsUpperSet s
    ⊢ IsUpperSet (interior s)
  -/
  rw [← isLowerSet_compl, ← closure_compl]
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : HasUpperLowerClosure α
    s : Set α
    h : IsUpperSet s
    ⊢ IsLowerSet (closure (HasCompl.compl s))
  -/
  exact h.compl.closure
  /-
    🎉 no goals
  -/


protected theorem IsLowerSet.interior (h : IsLowerSet s) : IsLowerSet (interior s) :=
  h.toDual.interior


protected theorem Set.OrdConnected.interior (h : s.OrdConnected) : (interior s).OrdConnected := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : HasUpperLowerClosure α
    s : Set α
    h : s.OrdConnected
    ⊢ (interior s).OrdConnected
  -/
  rw [← h.upperClosure_inter_lowerClosure, interior_inter]
  exact
    (upperClosure s).upper.interior.ordConnected.inter (lowerClosure s).lower.interior.ordConnected

