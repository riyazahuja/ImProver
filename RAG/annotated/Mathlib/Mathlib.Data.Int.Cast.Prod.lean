instance : AddGroupWithOne (α × β) :=
  { Prod.instAddMonoidWithOne, Prod.instAddGroup with
    intCast := fun n => (n, n)
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   inst✝¹ : AddGroupWithOne α
                                   inst✝ : AddGroupWithOne β
                                   x✝ : Nat
                                   ⊢ Eq (IntCast.intCast ↑x✝) ↑x✝
                                 -/
                                         /-
                                           🎉 no goals
                                         -/
    intCast_ofNat := fun _ => by ext <;> simp
                                         /-
                                           🎉 no goals
                                         -/
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     inst✝¹ : AddGroupWithOne α
                                     inst✝ : AddGroupWithOne β
                                     x✝ : Nat
                                     ⊢ Eq (IntCast.intCast (Int.negSucc x✝)) (Neg.neg ↑(HAdd.hAdd x✝ 1))
                                   -/
                                           /-
                                             🎉 no goals
                                           -/
    intCast_negSucc := fun _ => by ext <;> simp }
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem fst_intCast (n : ℤ) : (n : α × β).fst = n :=
  rfl


@[simp]
theorem snd_intCast (n : ℤ) : (n : α × β).snd = n :=
  rfl


