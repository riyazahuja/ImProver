@[simp]
theorem AbsoluteValue.map_units_int (abv : AbsoluteValue ℤ S) (x : ℤˣ) : abv x = 1 := by
  /-
    S : Type u_2
    inst✝ : LinearOrderedCommRing S
    abv : AbsoluteValue Int S
    x : Units Int
    ⊢ Eq (abv ↑x) 1
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  rcases Int.units_eq_one_or x with (rfl | rfl) <;> simp
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem AbsoluteValue.map_units_intCast [Nontrivial R] (abv : AbsoluteValue R S) (x : ℤˣ) :
                                /-
                                  R : Type u_1
                                  S : Type u_2
                                  inst✝² : Ring R
                                  inst✝¹ : LinearOrderedCommRing S
                                  inst✝ : Nontrivial R
                                  abv : AbsoluteValue R S
                                  x : Units Int
                                  ⊢ Eq (abv ↑↑x) 1
                                -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
    abv ((x : ℤ) : R) = 1 := by rcases Int.units_eq_one_or x with (rfl | rfl) <;> simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[deprecated (since := "2024-04-17")]
alias AbsoluteValue.map_units_int_cast := AbsoluteValue.map_units_intCast


@[simp]
theorem AbsoluteValue.map_units_int_smul (abv : AbsoluteValue R S) (x : ℤˣ) (y : R) :
                              /-
                                R : Type u_1
                                S : Type u_2
                                inst✝¹ : Ring R
                                inst✝ : LinearOrderedCommRing S
                                abv : AbsoluteValue R S
                                x : Units Int
                                y : R
                                ⊢ Eq (abv (HSMul.hSMul x y)) (abv y)
                              -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
    abv (x • y) = abv y := by rcases Int.units_eq_one_or x with (rfl | rfl) <;> simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- `Int.natAbs` as a bundled monoid with zero hom. -/
@[simps]
def Int.natAbsHom : ℤ →*₀ ℕ where
  toFun := Int.natAbs
  map_mul' := Int.natAbs_mul
  map_one' := Int.natAbs_one
  map_zero' := Int.natAbs_zero

