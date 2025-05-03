instance instLinearOrderedCommRing : LinearOrderedCommRing ℤ where
  __ := instCommRing
  __ := instLinearOrder
  add_le_add_left := @Int.add_le_add_left
  mul_pos := @Int.mul_pos
  zero_le_one := le_of_lt Int.zero_lt_one


instance instOrderedCommRing : OrderedCommRing ℤ := StrictOrderedCommRing.toOrderedCommRing'

instance instOrderedRing : OrderedRing ℤ := StrictOrderedRing.toOrderedRing'


lemma isCompl_even_odd : IsCompl { n : ℤ | Even n } { n | Odd n } := by
  /-
    ⊢ IsCompl (setOf fun n => Even n) (setOf fun n => Odd n)
  -/
  simp [← not_even_iff_odd, ← Set.compl_setOf, isCompl_compl]
  /-
    🎉 no goals
  -/


lemma _root_.Nat.cast_natAbs {α : Type*} [AddGroupWithOne α] (n : ℤ) : (n.natAbs : α) = |n| := by
  /-
    α : Type u_1
    inst✝ : AddGroupWithOne α
    n : Int
    ⊢ Eq ↑n.natAbs ↑(abs n)
  -/
  rw [← natCast_natAbs, Int.cast_natCast]
  /-
    🎉 no goals
  -/


/-- Note this holds in marginally more generality than `Int.cast_mul` -/
lemma cast_mul_eq_zsmul_cast {α : Type*} [AddCommGroupWithOne α] :
    ∀ m n : ℤ, ↑(m * n) = m • (n : α) :=
                                 /-
                                   α : Type u_1
                                   inst✝ : AddCommGroupWithOne α
                                   m : Int
                                   ⊢ ∀ (n : Int), Eq (↑(HMul.hMul 0 n)) (HSMul.hSMul 0 ↑n)
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  fun m ↦ Int.induction_on m (by simp) (fun _ ih ↦ by simp [add_mul, add_zsmul, ih]) fun _ ih ↦ by
                                                      /-
                                                        🎉 no goals
                                                      -/
    /-
      α : Type u_1
      inst✝ : AddCommGroupWithOne α
      m : Int
      x✝ : Nat
      ih : ∀ (n : Int), Eq (↑(HMul.hMul (Neg.neg ↑x✝) n)) (HSMul.hSMul (Neg.neg ↑x✝) …
      ⊢ ∀ (n : Int), Eq (↑(HMul.hMul (HSub.hSub (Neg.neg ↑x✝) 1) n)) (HSMul.hSMul (H …
    -/
    simp only [sub_mul, one_mul, cast_sub, ih, sub_zsmul, one_zsmul, ← sub_eq_add_neg, forall_const]
    /-
      🎉 no goals
    -/


lemma two_le_iff_pos_of_even {m : ℤ} (even : Even m) : 2 ≤ m ↔ 0 < m :=
                        /-
                          m : Int
                          even : Even m
                          ⊢ LT.lt 0 2
                        -/
  le_iff_pos_of_dvd (by decide) even.two_dvd
                        /-
                          🎉 no goals
                        -/


lemma add_two_le_iff_lt_of_even_sub {m n : ℤ} (even : Even (n - m)) : m + 2 ≤ n ↔ m < n := by
  /-
    m n : Int
    even : Even (HSub.hSub n m)
    ⊢ Iff (LE.le (HAdd.hAdd m 2) n) (LT.lt m n)
  -/
  rw [add_comm]; exact le_add_iff_lt_of_dvd_sub (by decide) even.two_dvd
                 /-
                   🎉 no goals
                 -/


