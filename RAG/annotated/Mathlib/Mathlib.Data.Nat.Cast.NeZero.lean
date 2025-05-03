                                                /-
                                                  n : Nat
                                                  inst✝ : NeZero n
                                                  ⊢ LE.le 1 n
                                                -/
theorem one_le {n : ℕ} [NeZero n] : 1 ≤ n := by have := NeZero.ne n; omega
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


lemma natCast_ne (n : ℕ) (R) [AddMonoidWithOne R] [h : NeZero (n : R)] : (n : R) ≠ 0 := h.out


lemma of_neZero_natCast (R) [AddMonoidWithOne R] {n : ℕ} [h : NeZero (n : R)] : NeZero n :=
      /-
        R : Type u_1
        inst✝ : AddMonoidWithOne R
        n : Nat
        h : NeZero ↑n
        ⊢ Ne n 0
      -/
  ⟨by rintro rfl; exact h.out Nat.cast_zero⟩
                  /-
                    🎉 no goals
                  -/


lemma pos_of_neZero_natCast (R) [AddMonoidWithOne R] {n : ℕ} [NeZero (n : R)] : 0 < n :=
  Nat.pos_of_ne_zero (of_neZero_natCast R).out


