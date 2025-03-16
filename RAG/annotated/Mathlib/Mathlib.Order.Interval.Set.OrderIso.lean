@[simp]
theorem preimage_Iic (e : α ≃o β) (b : β) : e ⁻¹' Iic b = Iic (e.symm b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    b : β
    ⊢ Eq (Set.preimage (⇑e) (Set.Iic b)) (Set.Iic (e.symm b))
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    b : β
    x : α
    ⊢ Iff (Membership.mem (Set.preimage (⇑e) (Set.Iic b)) x) (Membership.mem (Set. …
  -/
  simp [← e.le_iff_le]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Ici (e : α ≃o β) (b : β) : e ⁻¹' Ici b = Ici (e.symm b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    b : β
    ⊢ Eq (Set.preimage (⇑e) (Set.Ici b)) (Set.Ici (e.symm b))
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    b : β
    x : α
    ⊢ Iff (Membership.mem (Set.preimage (⇑e) (Set.Ici b)) x) (Membership.mem (Set. …
  -/
  simp [← e.le_iff_le]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Iio (e : α ≃o β) (b : β) : e ⁻¹' Iio b = Iio (e.symm b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    b : β
    ⊢ Eq (Set.preimage (⇑e) (Set.Iio b)) (Set.Iio (e.symm b))
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    b : β
    x : α
    ⊢ Iff (Membership.mem (Set.preimage (⇑e) (Set.Iio b)) x) (Membership.mem (Set. …
  -/
  simp [← e.lt_iff_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Ioi (e : α ≃o β) (b : β) : e ⁻¹' Ioi b = Ioi (e.symm b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    b : β
    ⊢ Eq (Set.preimage (⇑e) (Set.Ioi b)) (Set.Ioi (e.symm b))
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    b : β
    x : α
    ⊢ Iff (Membership.mem (Set.preimage (⇑e) (Set.Ioi b)) x) (Membership.mem (Set. …
  -/
  simp [← e.lt_iff_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Icc (e : α ≃o β) (a b : β) : e ⁻¹' Icc a b = Icc (e.symm a) (e.symm b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    a b : β
    ⊢ Eq (Set.preimage (⇑e) (Set.Icc a b)) (Set.Icc (e.symm a) (e.symm b))
  -/
  simp [← Ici_inter_Iic]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Ico (e : α ≃o β) (a b : β) : e ⁻¹' Ico a b = Ico (e.symm a) (e.symm b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    a b : β
    ⊢ Eq (Set.preimage (⇑e) (Set.Ico a b)) (Set.Ico (e.symm a) (e.symm b))
  -/
  simp [← Ici_inter_Iio]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Ioc (e : α ≃o β) (a b : β) : e ⁻¹' Ioc a b = Ioc (e.symm a) (e.symm b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    a b : β
    ⊢ Eq (Set.preimage (⇑e) (Set.Ioc a b)) (Set.Ioc (e.symm a) (e.symm b))
  -/
  simp [← Ioi_inter_Iic]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Ioo (e : α ≃o β) (a b : β) : e ⁻¹' Ioo a b = Ioo (e.symm a) (e.symm b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    a b : β
    ⊢ Eq (Set.preimage (⇑e) (Set.Ioo a b)) (Set.Ioo (e.symm a) (e.symm b))
  -/
  simp [← Ioi_inter_Iio]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_Iic (e : α ≃o β) (a : α) : e '' Iic a = Iic (e a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    a : α
    ⊢ Eq (Set.image (⇑e) (Set.Iic a)) (Set.Iic (e a))
  -/
  rw [e.image_eq_preimage, e.symm.preimage_Iic, e.symm_symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_Ici (e : α ≃o β) (a : α) : e '' Ici a = Ici (e a) :=
  e.dual.image_Iic a


@[simp]
theorem image_Iio (e : α ≃o β) (a : α) : e '' Iio a = Iio (e a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    a : α
    ⊢ Eq (Set.image (⇑e) (Set.Iio a)) (Set.Iio (e a))
  -/
  rw [e.image_eq_preimage, e.symm.preimage_Iio, e.symm_symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_Ioi (e : α ≃o β) (a : α) : e '' Ioi a = Ioi (e a) :=
  e.dual.image_Iio a


@[simp]
theorem image_Ioo (e : α ≃o β) (a b : α) : e '' Ioo a b = Ioo (e a) (e b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    a b : α
    ⊢ Eq (Set.image (⇑e) (Set.Ioo a b)) (Set.Ioo (e a) (e b))
  -/
  rw [e.image_eq_preimage, e.symm.preimage_Ioo, e.symm_symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_Ioc (e : α ≃o β) (a b : α) : e '' Ioc a b = Ioc (e a) (e b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    a b : α
    ⊢ Eq (Set.image (⇑e) (Set.Ioc a b)) (Set.Ioc (e a) (e b))
  -/
  rw [e.image_eq_preimage, e.symm.preimage_Ioc, e.symm_symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_Ico (e : α ≃o β) (a b : α) : e '' Ico a b = Ico (e a) (e b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    a b : α
    ⊢ Eq (Set.image (⇑e) (Set.Ico a b)) (Set.Ico (e a) (e b))
  -/
  rw [e.image_eq_preimage, e.symm.preimage_Ico, e.symm_symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_Icc (e : α ≃o β) (a b : α) : e '' Icc a b = Icc (e a) (e b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderIso α β
    a b : α
    ⊢ Eq (Set.image (⇑e) (Set.Icc a b)) (Set.Icc (e a) (e b))
  -/
  rw [e.image_eq_preimage, e.symm.preimage_Icc, e.symm_symm]
  /-
    🎉 no goals
  -/


/-- Order isomorphism between `Iic (⊤ : α)` and `α` when `α` has a top element -/
def IicTop {α : Type*} [Preorder α] [OrderTop α] : Iic (⊤ : α) ≃o α :=
  { @Equiv.subtypeUnivEquiv α (Iic (⊤ : α)) fun _ => le_top with
                                   /-
                                     α : Type u_1
                                     inst✝¹ : Preorder α
                                     inst✝ : OrderTop α
                                     x y : ↑(Set.Iic Top.top)
                                     ⊢ Iff (LE.le (__src✝ x) (__src✝ y)) (LE.le x y)
                                   -/
    map_rel_iff' := @fun x y => by rfl }
                                   /-
                                     🎉 no goals
                                   -/


/-- Order isomorphism between `Ici (⊥ : α)` and `α` when `α` has a bottom element -/
def IciBot {α : Type*} [Preorder α] [OrderBot α] : Ici (⊥ : α) ≃o α :=
  { @Equiv.subtypeUnivEquiv α (Ici (⊥ : α)) fun _ => bot_le with
                                   /-
                                     α : Type u_1
                                     inst✝¹ : Preorder α
                                     inst✝ : OrderBot α
                                     x y : ↑(Set.Ici Bot.bot)
                                     ⊢ Iff (LE.le (__src✝ x) (__src✝ y)) (LE.le x y)
                                   -/
    map_rel_iff' := @fun x y => by rfl }
                                   /-
                                     🎉 no goals
                                   -/


