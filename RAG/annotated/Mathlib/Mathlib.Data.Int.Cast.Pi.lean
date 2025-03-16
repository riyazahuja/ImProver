instance instIntCast : IntCast (∀ i, π i) where intCast n _ := n


theorem intCast_apply (n : ℤ) (i : ι) : (n : ∀ i, π i) i = n :=
  rfl


@[simp]
theorem intCast_def (n : ℤ) : (n : ∀ i, π i) = fun _ => ↑n :=
  rfl


@[deprecated (since := "2024-04-05")] alias int_apply := intCast_apply

@[deprecated (since := "2024-04-05")] alias coe_int := intCast_def


theorem Sum.elim_intCast_intCast {α β γ : Type*} [IntCast γ] (n : ℤ) :
    Sum.elim (n : α → γ) (n : β → γ) = n :=
  Sum.elim_lam_const_lam_const (γ := γ) n

