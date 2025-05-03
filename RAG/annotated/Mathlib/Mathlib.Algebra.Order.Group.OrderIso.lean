/-- `x ↦ x⁻¹` as an order-reversing equivalence. -/
@[to_additive (attr := simps!) "`x ↦ -x` as an order-reversing equivalence."]
def OrderIso.inv : α ≃o αᵒᵈ where
  toEquiv := (Equiv.inv α).trans OrderDual.toDual
  map_rel_iff' {_ _} := inv_le_inv_iff (α := α)


@[to_additive neg_le]
theorem inv_le' : a⁻¹ ≤ b ↔ b⁻¹ ≤ a :=
  (OrderIso.inv α).symm_apply_le


alias ⟨inv_le_of_inv_le', _⟩ := inv_le'


attribute [to_additive neg_le_of_neg_le] inv_le_of_inv_le'


@[to_additive le_neg]
theorem le_inv' : a ≤ b⁻¹ ↔ b ≤ a⁻¹ :=
  (OrderIso.inv α).le_symm_apply


/-- `x ↦ a / x` as an order-reversing equivalence. -/
@[to_additive (attr := simps!) "`x ↦ a - x` as an order-reversing equivalence."]
def OrderIso.divLeft (a : α) : α ≃o αᵒᵈ where
  toEquiv := (Equiv.divLeft a).trans OrderDual.toDual
  map_rel_iff' {_ _} := div_le_div_iff_left (α := α) _


alias ⟨le_inv_of_le_inv, _⟩ := le_inv'


attribute [to_additive] le_inv_of_le_inv


/-- `Equiv.mulRight` as an `OrderIso`. See also `OrderEmbedding.mulRight`. -/
@[to_additive (attr := simps! (config := { simpRhs := true }) toEquiv apply)
  "`Equiv.addRight` as an `OrderIso`. See also `OrderEmbedding.addRight`."]
def OrderIso.mulRight (a : α) : α ≃o α where
  map_rel_iff' {_ _} := mul_le_mul_iff_right a
  toEquiv := Equiv.mulRight a


@[to_additive (attr := simp)]
theorem OrderIso.mulRight_symm (a : α) : (OrderIso.mulRight a).symm = OrderIso.mulRight a⁻¹ := by
  /-
    α : Type u
    inst✝² : Group α
    inst✝¹ : LE α
    inst✝ : MulRightMono α
    a : α
    ⊢ Eq (OrderIso.mulRight a).symm (OrderIso.mulRight (Inv.inv a))
  -/
  ext x
  /-
    case h.h
    α : Type u
    inst✝² : Group α
    inst✝¹ : LE α
    inst✝ : MulRightMono α
    a x : α
    ⊢ Eq ((OrderIso.mulRight a).symm x) ((OrderIso.mulRight (Inv.inv a)) x)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `x ↦ x / a` as an order isomorphism. -/
@[to_additive (attr := simps!) "`x ↦ x - a` as an order isomorphism."]
def OrderIso.divRight (a : α) : α ≃o α where
  toEquiv := Equiv.divRight a
  map_rel_iff' {_ _} := div_le_div_iff_right a


/-- `Equiv.mulLeft` as an `OrderIso`. See also `OrderEmbedding.mulLeft`. -/
@[to_additive (attr := simps! (config := { simpRhs := true }) toEquiv apply)
  "`Equiv.addLeft` as an `OrderIso`. See also `OrderEmbedding.addLeft`."]
def OrderIso.mulLeft (a : α) : α ≃o α where
  map_rel_iff' {_ _} := mul_le_mul_iff_left a
  toEquiv := Equiv.mulLeft a


@[to_additive (attr := simp)]
theorem OrderIso.mulLeft_symm (a : α) : (OrderIso.mulLeft a).symm = OrderIso.mulLeft a⁻¹ := by
  /-
    α : Type u
    inst✝² : Group α
    inst✝¹ : LE α
    inst✝ : MulLeftMono α
    a : α
    ⊢ Eq (OrderIso.mulLeft a).symm (OrderIso.mulLeft (Inv.inv a))
  -/
  ext x
  /-
    case h.h
    α : Type u
    inst✝² : Group α
    inst✝¹ : LE α
    inst✝ : MulLeftMono α
    a x : α
    ⊢ Eq ((OrderIso.mulLeft a).symm x) ((OrderIso.mulLeft (Inv.inv a)) x)
  -/
  rfl
  /-
    🎉 no goals
  -/


