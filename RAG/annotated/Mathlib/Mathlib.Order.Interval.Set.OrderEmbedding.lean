@[simp] theorem preimage_Ici : e ⁻¹' Ici (e x) = Ici x := ext fun _ ↦ e.le_iff_le

@[simp] theorem preimage_Iic : e ⁻¹' Iic (e x) = Iic x := ext fun _ ↦ e.le_iff_le

@[simp] theorem preimage_Ioi : e ⁻¹' Ioi (e x) = Ioi x := ext fun _ ↦ e.lt_iff_lt

@[simp] theorem preimage_Iio : e ⁻¹' Iio (e x) = Iio x := ext fun _ ↦ e.lt_iff_lt


                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       inst✝¹ : Preorder α
                                                                       inst✝ : Preorder β
                                                                       e : OrderEmbedding α β
                                                                       x y : α
                                                                       ⊢ Eq (Set.preimage (⇑e) (Set.Icc (e x) (e y))) (Set.Icc x y)
                                                                     -/
@[simp] theorem preimage_Icc : e ⁻¹' Icc (e x) (e y) = Icc x y := by ext; simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       inst✝¹ : Preorder α
                                                                       inst✝ : Preorder β
                                                                       e : OrderEmbedding α β
                                                                       x y : α
                                                                       ⊢ Eq (Set.preimage (⇑e) (Set.Ico (e x) (e y))) (Set.Ico x y)
                                                                     -/
@[simp] theorem preimage_Ico : e ⁻¹' Ico (e x) (e y) = Ico x y := by ext; simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       inst✝¹ : Preorder α
                                                                       inst✝ : Preorder β
                                                                       e : OrderEmbedding α β
                                                                       x y : α
                                                                       ⊢ Eq (Set.preimage (⇑e) (Set.Ioc (e x) (e y))) (Set.Ioc x y)
                                                                     -/
@[simp] theorem preimage_Ioc : e ⁻¹' Ioc (e x) (e y) = Ioc x y := by ext; simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       inst✝¹ : Preorder α
                                                                       inst✝ : Preorder β
                                                                       e : OrderEmbedding α β
                                                                       x y : α
                                                                       ⊢ Eq (Set.preimage (⇑e) (Set.Ioo (e x) (e y))) (Set.Ioo x y)
                                                                     -/
@[simp] theorem preimage_Ioo : e ⁻¹' Ioo (e x) (e y) = Ioo x y := by ext; simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp] theorem preimage_uIcc [Lattice β] (e : α ↪o β) (x y : α) :
    e ⁻¹' (uIcc (e x) (e y)) = uIcc x y := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Lattice β
    e : OrderEmbedding α β
    x y : α
    ⊢ Eq (Set.preimage (⇑e) (Set.uIcc (e x) (e y))) (Set.uIcc x y)
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total x y <;> simp [*]
                         /-
                           🎉 no goals
                         -/


@[simp] theorem preimage_uIoc [LinearOrder β] (e : α ↪o β) (x y : α) :
    e ⁻¹' (uIoc (e x) (e y)) = uIoc x y := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    e : OrderEmbedding α β
    x y : α
    ⊢ Eq (Set.preimage (⇑e) (Set.uIoc (e x) (e y))) (Set.uIoc x y)
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total x y <;> simp [*]
                         /-
                           🎉 no goals
                         -/


