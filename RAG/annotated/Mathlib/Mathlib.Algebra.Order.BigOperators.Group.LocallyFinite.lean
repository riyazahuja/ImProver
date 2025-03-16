@[to_additive]
lemma left_mul_prod_Ioc (h : a ≤ b) : f a * ∏ x ∈ Ioc a b, f x = ∏ x ∈ Icc a b, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LE.le a b
    ⊢ Eq (HMul.hMul (f a) ((Finset.Ioc a b).prod fun x => f x)) ((Finset.Icc a b). …
  -/
  rw [Icc_eq_cons_Ioc h, prod_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_Ioc_mul_left (h : a ≤ b) : (∏ x ∈ Ioc a b, f x) * f a = ∏ x ∈ Icc a b, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LE.le a b
    ⊢ Eq (HMul.hMul ((Finset.Ioc a b).prod fun x => f x) (f a)) ((Finset.Icc a b). …
  -/
  rw [mul_comm, left_mul_prod_Ioc h]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma right_mul_prod_Ico (h : a ≤ b) : f b * ∏ x ∈ Ico a b, f x = ∏ x ∈ Icc a b, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LE.le a b
    ⊢ Eq (HMul.hMul (f b) ((Finset.Ico a b).prod fun x => f x)) ((Finset.Icc a b). …
  -/
  rw [Icc_eq_cons_Ico h, prod_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_Ico_mul_right (h : a ≤ b) : (∏ x ∈ Ico a b, f x) * f b = ∏ x ∈ Icc a b, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LE.le a b
    ⊢ Eq (HMul.hMul ((Finset.Ico a b).prod fun x => f x) (f b)) ((Finset.Icc a b). …
  -/
  rw [mul_comm, right_mul_prod_Ico h]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma left_mul_prod_Ioo (h : a < b) : f a * ∏ x ∈ Ioo a b, f x = ∏ x ∈ Ico a b, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LT.lt a b
    ⊢ Eq (HMul.hMul (f a) ((Finset.Ioo a b).prod fun x => f x)) ((Finset.Ico a b). …
  -/
  rw [Ico_eq_cons_Ioo h, prod_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_Ioo_mul_left (h : a < b) : (∏ x ∈ Ioo a b, f x) * f a = ∏ x ∈ Ico a b, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LT.lt a b
    ⊢ Eq (HMul.hMul ((Finset.Ioo a b).prod fun x => f x) (f a)) ((Finset.Ico a b). …
  -/
  rw [mul_comm, left_mul_prod_Ioo h]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma right_mul_prod_Ioo (h : a < b) : f b * ∏ x ∈ Ioo a b, f x = ∏ x ∈ Ioc a b, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LT.lt a b
    ⊢ Eq (HMul.hMul (f b) ((Finset.Ioo a b).prod fun x => f x)) ((Finset.Ioc a b). …
  -/
  rw [Ioc_eq_cons_Ioo h, prod_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_Ioo_mul_right (h : a < b) : (∏ x ∈ Ioo a b, f x) * f b = ∏ x ∈ Ioc a b, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LT.lt a b
    ⊢ Eq (HMul.hMul ((Finset.Ioo a b).prod fun x => f x) (f b)) ((Finset.Ioc a b). …
  -/
  rw [mul_comm, right_mul_prod_Ioo h]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma left_mul_prod_Ioi (a : α) : f a * ∏ x ∈ Ioi a, f x = ∏ x ∈ Ici a, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    inst✝ : LocallyFiniteOrderTop α
    a : α
    ⊢ Eq (HMul.hMul (f a) ((Finset.Ioi a).prod fun x => f x)) ((Finset.Ici a).prod …
  -/
  rw [Ici_eq_cons_Ioi, prod_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_Ioi_mul_left (a : α) : (∏ x ∈ Ioi a, f x) * f a = ∏ x ∈ Ici a, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    inst✝ : LocallyFiniteOrderTop α
    a : α
    ⊢ Eq (HMul.hMul ((Finset.Ioi a).prod fun x => f x) (f a)) ((Finset.Ici a).prod …
  -/
  rw [mul_comm, left_mul_prod_Ioi]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma right_mul_prod_Iio (a : α) : f a * ∏ x ∈ Iio a, f x = ∏ x ∈ Iic a, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    inst✝ : LocallyFiniteOrderBot α
    a : α
    ⊢ Eq (HMul.hMul (f a) ((Finset.Iio a).prod fun x => f x)) ((Finset.Iic a).prod …
  -/
  rw [Iic_eq_cons_Iio, prod_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_Iio_mul_right (a : α) : (∏ x ∈ Iio a, f x) * f a = ∏ x ∈ Iic a, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid β
    f : α → β
    inst✝ : LocallyFiniteOrderBot α
    a : α
    ⊢ Eq (HMul.hMul ((Finset.Iio a).prod fun x => f x) (f a)) ((Finset.Iic a).prod …
  -/
  rw [mul_comm, right_mul_prod_Iio]
  /-
    🎉 no goals
  -/


