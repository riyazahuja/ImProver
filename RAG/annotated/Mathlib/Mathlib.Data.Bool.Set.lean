@[simp]
theorem univ_eq : (univ : Set Bool) = {false, true} :=
  (eq_univ_of_forall Bool.dichotomy).symm


@[simp]
theorem range_eq {α : Type*} (f : Bool → α) : range f = {f false, f true} := by
  /-
    α : Type u_1
    f : Bool → α
    ⊢ Eq (Set.range f) (Insert.insert (f Bool.false) (Singleton.singleton (f Bool. …
  -/
  rw [← image_univ, univ_eq, image_pair]
  /-
    🎉 no goals
  -/


@[simp] theorem compl_singleton (b : Bool) : ({b}ᶜ : Set Bool) = {!b} :=
  Set.ext fun _ => eq_not_iff.symm


