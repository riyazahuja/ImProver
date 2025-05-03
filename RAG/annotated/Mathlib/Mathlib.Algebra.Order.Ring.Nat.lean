instance instLinearOrderedCommSemiring : LinearOrderedCommSemiring ℕ where
  __ := instCommSemiring
  __ := instLinearOrder
  add_le_add_left := @Nat.add_le_add_left
  le_of_add_le_add_left := @Nat.le_of_add_le_add_left
  zero_le_one := Nat.le_of_lt (Nat.zero_lt_succ 0)
  mul_lt_mul_of_pos_left := @Nat.mul_lt_mul_of_pos_left
  mul_lt_mul_of_pos_right := @Nat.mul_lt_mul_of_pos_right
  exists_pair_ne := ⟨0, 1, ne_of_lt Nat.zero_lt_one⟩


instance instLinearOrderedCommMonoidWithZero : LinearOrderedCommMonoidWithZero ℕ where
  __ := instLinearOrderedCommSemiring
  __ : CommMonoidWithZero ℕ := inferInstance
  mul_le_mul_left _ _ h c := Nat.mul_le_mul_left c h


instance instCanonicallyOrderedCommSemiring : CanonicallyOrderedCommSemiring ℕ where
  __ := instLinearOrderedCommSemiring
  exists_add_of_le h := (Nat.le.dest h).imp fun _ => Eq.symm
  le_self_add := Nat.le_add_right
  eq_zero_or_eq_zero_of_mul_eq_zero := Nat.mul_eq_zero.mp


instance instLinearOrderedSemiring : LinearOrderedSemiring ℕ := inferInstance

instance instStrictOrderedSemiring : StrictOrderedSemiring ℕ := inferInstance

instance instStrictOrderedCommSemiring : StrictOrderedCommSemiring ℕ := inferInstance

instance instOrderedSemiring : OrderedSemiring ℕ := StrictOrderedSemiring.toOrderedSemiring'

instance instOrderedCommSemiring : OrderedCommSemiring ℕ :=
  StrictOrderedCommSemiring.toOrderedCommSemiring'


lemma isCompl_even_odd : IsCompl { n : ℕ | Even n } { n | Odd n } := by
  /-
    ⊢ IsCompl (setOf fun n => Even n) (setOf fun n => Odd n)
  -/
  simp only [← Set.compl_setOf, isCompl_compl, ← not_even_iff_odd]
  /-
    🎉 no goals
  -/


