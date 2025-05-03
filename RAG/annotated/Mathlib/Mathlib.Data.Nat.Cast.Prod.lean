instance instAddMonoidWithOne : AddMonoidWithOne (α × β) :=
  { Prod.instAddMonoid, @Prod.instOne α β _ _ with
    natCast := fun n => (n, n)
    natCast_zero := congr_arg₂ Prod.mk Nat.cast_zero Nat.cast_zero
    natCast_succ := fun _ => congr_arg₂ Prod.mk (Nat.cast_succ _) (Nat.cast_succ _) }


@[simp]
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          inst✝¹ : AddMonoidWithOne α
                                                          inst✝ : AddMonoidWithOne β
                                                          n : Nat
                                                          ⊢ Eq (↑n).1 ↑n
                                                        -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
theorem fst_natCast (n : ℕ) : (n : α × β).fst = n := by induction n <;> simp [*]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem fst_ofNat (n : ℕ) [n.AtLeastTwo] :
    (ofNat(n) : α × β).1 = (ofNat(n) : α) :=
  rfl


@[simp]
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          inst✝¹ : AddMonoidWithOne α
                                                          inst✝ : AddMonoidWithOne β
                                                          n : Nat
                                                          ⊢ Eq (↑n).2 ↑n
                                                        -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
theorem snd_natCast (n : ℕ) : (n : α × β).snd = n := by induction n <;> simp [*]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem snd_ofNat (n : ℕ) [n.AtLeastTwo] :
    (ofNat(n) : α × β).2 = (ofNat(n) : β) :=
  rfl


