/-- The **Dyson e-transform**. Turns `(s, t)` into `(s ∪ e • t, t ∩ e⁻¹ • s)`. This reduces the
product of the two sets. -/
@[to_additive (attr := simps) "The **Dyson e-transform**.
Turns `(s, t)` into `(s ∪ e +ᵥ t, t ∩ -e +ᵥ s)`. This reduces the sum of the two sets."]
def mulDysonETransform : Finset α × Finset α :=
  (x.1 ∪ e • x.2, x.2 ∩ e⁻¹ • x.1)


@[to_additive]
theorem mulDysonETransform.subset :
    (mulDysonETransform e x).1 * (mulDysonETransform e x).2 ⊆ x.1 * x.2 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommGroup α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ HasSubset.Subset (HMul.hMul (Finset.mulDysonETransform e x).1 (Finset.mulDys …
  -/
  refine union_mul_inter_subset_union.trans (union_subset Subset.rfl ?_)
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommGroup α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ HasSubset.Subset (HMul.hMul (HSMul.hSMul e x.2) (HSMul.hSMul (Inv.inv e) x.1 …
  -/
  rw [mul_smul_comm, smul_mul_assoc, inv_smul_smul, mul_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulDysonETransform.card :
    (mulDysonETransform e x).1.card + (mulDysonETransform e x).2.card = x.1.card + x.2.card := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommGroup α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ Eq (HAdd.hAdd (Finset.mulDysonETransform e x).1.card (Finset.mulDysonETransf …
  -/
  dsimp
  rw [← card_smul_finset e (_ ∩ _), smul_finset_inter, smul_inv_smul, inter_comm,
    card_union_add_card_inter, card_smul_finset]


@[to_additive (attr := simp)]
theorem mulDysonETransform_idem :
    mulDysonETransform e (mulDysonETransform e x) = mulDysonETransform e x := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommGroup α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ Eq (Finset.mulDysonETransform e (Finset.mulDysonETransform e x)) (Finset.mul …
  -/
  ext : 1 <;> dsimp
    /-
      case fst
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : CommGroup α
      e : α
      x : Prod (Finset α) (Finset α)
      ⊢ Eq (Union.union (Union.union x.1 (HSMul.hSMul e x.2)) (HSMul.hSMul e (Inter. …
    -/
  · rw [smul_finset_inter, smul_inv_smul, inter_comm, union_eq_left]
    /-
      case fst
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : CommGroup α
      e : α
      x : Prod (Finset α) (Finset α)
      ⊢ HasSubset.Subset (Inter.inter x.1 (HSMul.hSMul e x.2)) (Union.union x.1 (HSM …
    -/
    exact inter_subset_union
    /-
      🎉 no goals
    -/
    /-
      case snd
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : CommGroup α
      e : α
      x : Prod (Finset α) (Finset α)
      ⊢ Eq (Inter.inter (Inter.inter x.2 (HSMul.hSMul (Inv.inv e) x.1)) (HSMul.hSMul …
    -/
  · rw [smul_finset_union, inv_smul_smul, union_comm, inter_eq_left]
    /-
      case snd
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : CommGroup α
      e : α
      x : Prod (Finset α) (Finset α)
      ⊢ HasSubset.Subset (Inter.inter x.2 (HSMul.hSMul (Inv.inv e) x.1)) (Union.unio …
    -/
    exact inter_subset_union
    /-
      🎉 no goals
    -/


@[to_additive]
theorem mulDysonETransform.smul_finset_snd_subset_fst :
    e • (mulDysonETransform e x).2 ⊆ (mulDysonETransform e x).1 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommGroup α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ HasSubset.Subset (HSMul.hSMul e (Finset.mulDysonETransform e x).2) (Finset.m …
  -/
  dsimp
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommGroup α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ HasSubset.Subset (HSMul.hSMul e (Inter.inter x.2 (HSMul.hSMul (Inv.inv e) x. …
  -/
  rw [smul_finset_inter, smul_inv_smul, inter_comm]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommGroup α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ HasSubset.Subset (Inter.inter x.1 (HSMul.hSMul e x.2)) (Union.union x.1 (HSM …
  -/
  exact inter_subset_union
  /-
    🎉 no goals
  -/


/-- An **e-transform**. Turns `(s, t)` into `(s ∩ s • e, t ∪ e⁻¹ • t)`. This reduces the
product of the two sets. -/
@[to_additive (attr := simps) "An **e-transform**.
Turns `(s, t)` into `(s ∩ s +ᵥ e, t ∪ -e +ᵥ t)`. This reduces the sum of the two sets."]
def mulETransformLeft : Finset α × Finset α :=
  (x.1 ∩ op e • x.1, x.2 ∪ e⁻¹ • x.2)


/-- An **e-transform**. Turns `(s, t)` into `(s ∪ s • e, t ∩ e⁻¹ • t)`. This reduces the
product of the two sets. -/
@[to_additive (attr := simps) "An **e-transform**.
Turns `(s, t)` into `(s ∪ s +ᵥ e, t ∩ -e +ᵥ t)`. This reduces the sum of the two sets."]
def mulETransformRight : Finset α × Finset α :=
  (x.1 ∪ op e • x.1, x.2 ∩ e⁻¹ • x.2)


@[to_additive (attr := simp)]
                                                                /-
                                                                  α : Type u_1
                                                                  inst✝¹ : DecidableEq α
                                                                  inst✝ : Group α
                                                                  x : Prod (Finset α) (Finset α)
                                                                  ⊢ Eq (Finset.mulETransformLeft 1 x) x
                                                                -/
theorem mulETransformLeft_one : mulETransformLeft 1 x = x := by simp [mulETransformLeft]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[to_additive (attr := simp)]
                                                                  /-
                                                                    α : Type u_1
                                                                    inst✝¹ : DecidableEq α
                                                                    inst✝ : Group α
                                                                    x : Prod (Finset α) (Finset α)
                                                                    ⊢ Eq (Finset.mulETransformRight 1 x) x
                                                                  -/
theorem mulETransformRight_one : mulETransformRight 1 x = x := by simp [mulETransformRight]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[to_additive]
theorem mulETransformLeft.fst_mul_snd_subset :
    (mulETransformLeft e x).1 * (mulETransformLeft e x).2 ⊆ x.1 * x.2 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Group α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ HasSubset.Subset (HMul.hMul (Finset.mulETransformLeft e x).1 (Finset.mulETra …
  -/
  refine inter_mul_union_subset_union.trans (union_subset Subset.rfl ?_)
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Group α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ HasSubset.Subset (HMul.hMul (HSMul.hSMul (MulOpposite.op e) x.1) (HSMul.hSMu …
  -/
  rw [op_smul_finset_mul_eq_mul_smul_finset, smul_inv_smul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulETransformRight.fst_mul_snd_subset :
    (mulETransformRight e x).1 * (mulETransformRight e x).2 ⊆ x.1 * x.2 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Group α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ HasSubset.Subset (HMul.hMul (Finset.mulETransformRight e x).1 (Finset.mulETr …
  -/
  refine union_mul_inter_subset_union.trans (union_subset Subset.rfl ?_)
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Group α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ HasSubset.Subset (HMul.hMul (HSMul.hSMul (MulOpposite.op e) x.1) (HSMul.hSMu …
  -/
  rw [op_smul_finset_mul_eq_mul_smul_finset, smul_inv_smul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulETransformLeft.card :
    (mulETransformLeft e x).1.card + (mulETransformRight e x).1.card = 2 * x.1.card :=
                                              /-
                                                α : Type u_1
                                                inst✝¹ : DecidableEq α
                                                inst✝ : Group α
                                                e : α
                                                x : Prod (Finset α) (Finset α)
                                                ⊢ Eq (HAdd.hAdd x.1.card (HSMul.hSMul (MulOpposite.op e) x.1).card) (HMul.hMul …
                                              -/
  (card_inter_add_card_union _ _).trans <| by rw [card_smul_finset, two_mul]
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive]
theorem mulETransformRight.card :
    (mulETransformLeft e x).2.card + (mulETransformRight e x).2.card = 2 * x.2.card :=
                                              /-
                                                α : Type u_1
                                                inst✝¹ : DecidableEq α
                                                inst✝ : Group α
                                                e : α
                                                x : Prod (Finset α) (Finset α)
                                                ⊢ Eq (HAdd.hAdd x.2.card (HSMul.hSMul (Inv.inv e) x.2).card) (HMul.hMul 2 x.2. …
                                              -/
  (card_union_add_card_inter _ _).trans <| by rw [card_smul_finset, two_mul]
                                              /-
                                                🎉 no goals
                                              -/


/-- This statement is meant to be combined with `le_or_lt_of_add_le_add` and similar lemmas. -/
@[to_additive AddETransform.card "This statement is meant to be combined with
`le_or_lt_of_add_le_add` and similar lemmas."]
protected theorem MulETransform.card :
    (mulETransformLeft e x).1.card + (mulETransformLeft e x).2.card +
        ((mulETransformRight e x).1.card + (mulETransformRight e x).2.card) =
      x.1.card + x.2.card + (x.1.card + x.2.card) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Group α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Finset.mulETransformLeft e x).1.card (Finset.mulET …
  -/
  rw [add_add_add_comm, mulETransformLeft.card, mulETransformRight.card, ← mul_add, two_mul]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mulETransformLeft_inv : mulETransformLeft e⁻¹ x = (mulETransformRight e x.swap).swap := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommGroup α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ Eq (Finset.mulETransformLeft (Inv.inv e) x) (Finset.mulETransformRight e x.s …
  -/
  simp [-op_inv, op_smul_eq_smul, mulETransformLeft, mulETransformRight]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mulETransformRight_inv : mulETransformRight e⁻¹ x = (mulETransformLeft e x.swap).swap := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommGroup α
    e : α
    x : Prod (Finset α) (Finset α)
    ⊢ Eq (Finset.mulETransformRight (Inv.inv e) x) (Finset.mulETransformLeft e x.s …
  -/
  simp [-op_inv, op_smul_eq_smul, mulETransformLeft, mulETransformRight]
  /-
    🎉 no goals
  -/


