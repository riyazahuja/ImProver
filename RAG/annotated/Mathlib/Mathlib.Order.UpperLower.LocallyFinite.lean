protected theorem Finite.upperClosure [LocallyFiniteOrderTop α] (hs : s.Finite) :
    (upperClosure s : Set α).Finite := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    s : Set α
    inst✝ : LocallyFiniteOrderTop α
    hs : s.Finite
    ⊢ (↑(upperClosure s)).Finite
  -/
  rw [coe_upperClosure]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    s : Set α
    inst✝ : LocallyFiniteOrderTop α
    hs : s.Finite
    ⊢ (Set.iUnion fun a => Set.iUnion fun h => Set.Ici a).Finite
  -/
  exact hs.biUnion fun _ _ => finite_Ici _
  /-
    🎉 no goals
  -/


protected theorem Finite.lowerClosure [LocallyFiniteOrderBot α] (hs : s.Finite) :
    (lowerClosure s : Set α).Finite := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    s : Set α
    inst✝ : LocallyFiniteOrderBot α
    hs : s.Finite
    ⊢ (↑(lowerClosure s)).Finite
  -/
  rw [coe_lowerClosure]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    s : Set α
    inst✝ : LocallyFiniteOrderBot α
    hs : s.Finite
    ⊢ (Set.iUnion fun a => Set.iUnion fun h => Set.Iic a).Finite
  -/
  exact hs.biUnion fun _ _ => finite_Iic _
  /-
    🎉 no goals
  -/


