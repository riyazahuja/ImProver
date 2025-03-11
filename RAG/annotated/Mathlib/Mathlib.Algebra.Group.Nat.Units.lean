lemma units_eq_one (u : ℕˣ) : u = 1 := Units.ext <| Nat.eq_one_of_dvd_one ⟨u.inv, u.val_inv.symm⟩


lemma addUnits_eq_zero (u : AddUnits ℕ) : u = 0 :=
  AddUnits.ext <| (Nat.eq_zero_of_add_eq_zero u.val_neg).1


@[simp] protected lemma isUnit_iff {n : ℕ} : IsUnit n ↔ n = 1 where
           /-
             n : Nat
             ⊢ IsUnit n → Eq n 1
           -/
  mp := by rintro ⟨u, rfl⟩; obtain rfl := Nat.units_eq_one u; rfl
                                                              /-
                                                                🎉 no goals
                                                              -/
  mpr h := h.symm ▸ ⟨1, rfl⟩


instance unique_units : Unique ℕˣ where
  default := 1
  uniq := Nat.units_eq_one


instance unique_addUnits : Unique (AddUnits ℕ) where
  default := 0
  uniq := Nat.addUnits_eq_zero


