instance UnitsInt.fintype : Fintype ℤˣ :=
                       /-
                         α : Type u_1
                         x : Units Int
                         ⊢ Membership.mem (Insert.insert 1 (Singleton.singleton (-1))) x
                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  ⟨{1, -1}, fun x ↦ by cases Int.units_eq_one_or x <;> simp [*]⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem UnitsInt.univ : (Finset.univ : Finset ℤˣ) = {1, -1} := rfl


@[simp]
theorem Fintype.card_units_int : Fintype.card ℤˣ = 2 := rfl


instance [Monoid α] [Fintype α] [DecidableEq α] : Fintype αˣ :=
  Fintype.ofEquiv _ (unitsEquivProdSubtype α).symm


instance [Monoid α] [Finite α] : Finite αˣ := Finite.of_injective _ Units.ext


theorem Nat.card_units [GroupWithZero α] :
    Nat.card αˣ = Nat.card α - 1 := by
  classical
  rw [Nat.card_congr unitsEquivNeZero, eq_comm, ← Nat.card_congr (Equiv.sumCompl (· = (0 : α)))]
  rcases finite_or_infinite {a : α // a ≠ 0}
  · rw [Nat.card_sum, Nat.card_unique, add_tsub_cancel_left]
  · rw [Nat.card_eq_zero_of_infinite, Nat.card_eq_zero_of_infinite, zero_tsub]


theorem Nat.card_eq_card_units_add_one [GroupWithZero α] [Finite α] :
    Nat.card α = Nat.card αˣ + 1 := by
  /-
    α : Type u_1
    inst✝¹ : GroupWithZero α
    inst✝ : Finite α
    ⊢ Eq (Nat.card α) (HAdd.hAdd (Nat.card (Units α)) 1)
  -/
  rw [Nat.card_units, tsub_add_cancel_of_le Nat.card_pos]
  /-
    🎉 no goals
  -/


theorem Fintype.card_units [GroupWithZero α] [Fintype α] [DecidableEq α] :
    Fintype.card αˣ = Fintype.card α - 1 := by
  /-
    α : Type u_1
    inst✝² : GroupWithZero α
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    ⊢ Eq (Fintype.card (Units α)) (HSub.hSub (Fintype.card α) 1)
  -/
  rw [← Nat.card_eq_fintype_card, Nat.card_units, Nat.card_eq_fintype_card]
  /-
    🎉 no goals
  -/


theorem Fintype.card_eq_card_units_add_one [GroupWithZero α] [Fintype α] [DecidableEq α] :
    Fintype.card α = Fintype.card αˣ + 1 := by
  /-
    α : Type u_1
    inst✝² : GroupWithZero α
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    ⊢ Eq (Fintype.card α) (HAdd.hAdd (Fintype.card (Units α)) 1)
  -/
  rw [Fintype.card_units, tsub_add_cancel_of_le Fintype.card_pos]
  /-
    🎉 no goals
  -/

