open Units in
/-- The monoid equivalence between units of a product,
and the product of the units of each monoid. -/
@[to_additive (attr := simps)
  "The additive-monoid equivalence between (additive) units of a product,
  and the product of the (additive) units of each monoid."]
def MulEquiv.piUnits : (Π i, M i)ˣ ≃* Π i, (M i)ˣ where
  toFun f i := ⟨f.val i, f.inv i, congr_fun f.val_inv i, congr_fun f.inv_val i⟩
  invFun f := ⟨(val <| f ·), (inv <| f ·), funext (val_inv <| f ·), funext (inv_val <| f ·)⟩
  left_inv _ := rfl
  right_inv _ := rfl
  map_mul' _ _ := rfl


@[to_additive]
lemma Pi.isUnit_iff :
    IsUnit x ↔ ∀ i, IsUnit (x i) := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    x : (i : ι) → M i
    ⊢ Iff (IsUnit x) (∀ (i : ι), IsUnit (x i))
  -/
  simp_rw [isUnit_iff_exists, funext_iff, ← forall_and]
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    x : (i : ι) → M i
    ⊢ Iff (Exists fun b => ∀ (x_1 : ι), And (Eq (HMul.hMul x b x_1) (1 x_1)) (Eq ( …
  -/
  exact Classical.skolem (p := fun i y ↦ x i * y = 1 ∧ y * x i = 1).symm
  /-
    🎉 no goals
  -/


@[to_additive]
alias ⟨IsUnit.apply, _⟩ := Pi.isUnit_iff


@[to_additive]
lemma IsUnit.val_inv_apply (hx : IsUnit x) (i : ι) : (hx.unit⁻¹).1 i = (hx.apply i).unit⁻¹ := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    x : (i : ι) → M i
    hx : IsUnit x
    i : ι
    ⊢ Eq (↑(Inv.inv hx.unit) i) ↑(Inv.inv ⋯.unit)
  -/
  rw [← Units.inv_eq_val_inv, ← MulEquiv.val_inv_piUnits_apply]; congr; ext; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/

