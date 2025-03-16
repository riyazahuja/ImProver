instance {α β : Type*} [Encodable α] [Encodable β] [Zero β] [∀ x : β, Decidable (x ≠ 0)] :
    Encodable (α →₀ β) :=
  letI : DecidableEq α := Encodable.decidableEqOfEncodable _
  .ofEquiv _ finsuppEquivDFinsupp


instance {α β : Type*} [Countable α] [Countable β] [Zero β] : Countable (α →₀ β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Countable α
    inst✝¹ : Countable β
    inst✝ : Zero β
    ⊢ Countable (Finsupp α β)
  -/
  classical exact .of_equiv _ finsuppEquivDFinsupp.symm
  /-
    🎉 no goals
  -/

