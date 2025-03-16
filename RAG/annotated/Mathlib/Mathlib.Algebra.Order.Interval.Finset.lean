@[simp] lemma map_add_left_Icc (a b c : α) :
    (Icc a b).map (addLeftEmbedding c) = Icc (c + a) (c + b) := by
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Finset.map (addLeftEmbedding c) (Finset.Icc a b)) (Finset.Icc (HAdd.hAdd …
  -/
  rw [← coe_inj, coe_map, coe_Icc, coe_Icc]
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Set.image (⇑(addLeftEmbedding c)) (Set.Icc a b)) (Set.Icc (HAdd.hAdd c a …
  -/
  exact Set.image_const_add_Icc _ _ _
  /-
    🎉 no goals
  -/


@[simp] lemma map_add_right_Icc (a b c : α) :
    (Icc a b).map (addRightEmbedding c) = Icc (a + c) (b + c) := by
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Finset.map (addRightEmbedding c) (Finset.Icc a b)) (Finset.Icc (HAdd.hAd …
  -/
  rw [← coe_inj, coe_map, coe_Icc, coe_Icc]
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Set.image (⇑(addRightEmbedding c)) (Set.Icc a b)) (Set.Icc (HAdd.hAdd a  …
  -/
  exact Set.image_add_const_Icc _ _ _
  /-
    🎉 no goals
  -/


@[simp] lemma map_add_left_Ico (a b c : α) :
    (Ico a b).map (addLeftEmbedding c) = Ico (c + a) (c + b) := by
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Finset.map (addLeftEmbedding c) (Finset.Ico a b)) (Finset.Ico (HAdd.hAdd …
  -/
  rw [← coe_inj, coe_map, coe_Ico, coe_Ico]
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Set.image (⇑(addLeftEmbedding c)) (Set.Ico a b)) (Set.Ico (HAdd.hAdd c a …
  -/
  exact Set.image_const_add_Ico _ _ _
  /-
    🎉 no goals
  -/


@[simp] lemma map_add_right_Ico (a b c : α) :
    (Ico a b).map (addRightEmbedding c) = Ico (a + c) (b + c) := by
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Finset.map (addRightEmbedding c) (Finset.Ico a b)) (Finset.Ico (HAdd.hAd …
  -/
  rw [← coe_inj, coe_map, coe_Ico, coe_Ico]
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Set.image (⇑(addRightEmbedding c)) (Set.Ico a b)) (Set.Ico (HAdd.hAdd a  …
  -/
  exact Set.image_add_const_Ico _ _ _
  /-
    🎉 no goals
  -/


@[simp] lemma map_add_left_Ioc (a b c : α) :
    (Ioc a b).map (addLeftEmbedding c) = Ioc (c + a) (c + b) := by
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Finset.map (addLeftEmbedding c) (Finset.Ioc a b)) (Finset.Ioc (HAdd.hAdd …
  -/
  rw [← coe_inj, coe_map, coe_Ioc, coe_Ioc]
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Set.image (⇑(addLeftEmbedding c)) (Set.Ioc a b)) (Set.Ioc (HAdd.hAdd c a …
  -/
  exact Set.image_const_add_Ioc _ _ _
  /-
    🎉 no goals
  -/


@[simp] lemma map_add_right_Ioc (a b c : α) :
    (Ioc a b).map (addRightEmbedding c) = Ioc (a + c) (b + c) := by
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Finset.map (addRightEmbedding c) (Finset.Ioc a b)) (Finset.Ioc (HAdd.hAd …
  -/
  rw [← coe_inj, coe_map, coe_Ioc, coe_Ioc]
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Set.image (⇑(addRightEmbedding c)) (Set.Ioc a b)) (Set.Ioc (HAdd.hAdd a  …
  -/
  exact Set.image_add_const_Ioc _ _ _
  /-
    🎉 no goals
  -/


@[simp] lemma map_add_left_Ioo (a b c : α) :
    (Ioo a b).map (addLeftEmbedding c) = Ioo (c + a) (c + b) := by
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Finset.map (addLeftEmbedding c) (Finset.Ioo a b)) (Finset.Ioo (HAdd.hAdd …
  -/
  rw [← coe_inj, coe_map, coe_Ioo, coe_Ioo]
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Set.image (⇑(addLeftEmbedding c)) (Set.Ioo a b)) (Set.Ioo (HAdd.hAdd c a …
  -/
  exact Set.image_const_add_Ioo _ _ _
  /-
    🎉 no goals
  -/


@[simp] lemma map_add_right_Ioo (a b c : α) :
    (Ioo a b).map (addRightEmbedding c) = Ioo (a + c) (b + c) := by
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Finset.map (addRightEmbedding c) (Finset.Ioo a b)) (Finset.Ioo (HAdd.hAd …
  -/
  rw [← coe_inj, coe_map, coe_Ioo, coe_Ioo]
  /-
    α : Type u_2
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Set.image (⇑(addRightEmbedding c)) (Set.Ioo a b)) (Set.Ioo (HAdd.hAdd a  …
  -/
  exact Set.image_add_const_Ioo _ _ _
  /-
    🎉 no goals
  -/


@[simp] lemma image_add_left_Icc (a b c : α) : (Icc a b).image (c + ·) = Icc (c + a) (c + b) := by
  /-
    α : Type u_2
    inst✝³ : OrderedCancelAddCommMonoid α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b c : α
    ⊢ Eq (Finset.image (fun x => HAdd.hAdd c x) (Finset.Icc a b)) (Finset.Icc (HAd …
  -/
  rw [← map_add_left_Icc, map_eq_image, addLeftEmbedding, Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


@[simp] lemma image_add_left_Ico (a b c : α) : (Ico a b).image (c + ·) = Ico (c + a) (c + b) := by
  /-
    α : Type u_2
    inst✝³ : OrderedCancelAddCommMonoid α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b c : α
    ⊢ Eq (Finset.image (fun x => HAdd.hAdd c x) (Finset.Ico a b)) (Finset.Ico (HAd …
  -/
  rw [← map_add_left_Ico, map_eq_image, addLeftEmbedding, Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


@[simp] lemma image_add_left_Ioc (a b c : α) : (Ioc a b).image (c + ·) = Ioc (c + a) (c + b) := by
  /-
    α : Type u_2
    inst✝³ : OrderedCancelAddCommMonoid α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b c : α
    ⊢ Eq (Finset.image (fun x => HAdd.hAdd c x) (Finset.Ioc a b)) (Finset.Ioc (HAd …
  -/
  rw [← map_add_left_Ioc, map_eq_image, addLeftEmbedding, Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


@[simp] lemma image_add_left_Ioo (a b c : α) : (Ioo a b).image (c + ·) = Ioo (c + a) (c + b) := by
  /-
    α : Type u_2
    inst✝³ : OrderedCancelAddCommMonoid α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b c : α
    ⊢ Eq (Finset.image (fun x => HAdd.hAdd c x) (Finset.Ioo a b)) (Finset.Ioo (HAd …
  -/
  rw [← map_add_left_Ioo, map_eq_image, addLeftEmbedding, Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


@[simp] lemma image_add_right_Icc (a b c : α) : (Icc a b).image (· + c) = Icc (a + c) (b + c) := by
  /-
    α : Type u_2
    inst✝³ : OrderedCancelAddCommMonoid α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b c : α
    ⊢ Eq (Finset.image (fun x => HAdd.hAdd x c) (Finset.Icc a b)) (Finset.Icc (HAd …
  -/
  rw [← map_add_right_Icc, map_eq_image, addRightEmbedding, Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


@[simp] lemma image_add_right_Ico (a b c : α) : (Ico a b).image (· + c) = Ico (a + c) (b + c) := by
  /-
    α : Type u_2
    inst✝³ : OrderedCancelAddCommMonoid α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b c : α
    ⊢ Eq (Finset.image (fun x => HAdd.hAdd x c) (Finset.Ico a b)) (Finset.Ico (HAd …
  -/
  rw [← map_add_right_Ico, map_eq_image, addRightEmbedding, Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


@[simp] lemma image_add_right_Ioc (a b c : α) : (Ioc a b).image (· + c) = Ioc (a + c) (b + c) := by
  /-
    α : Type u_2
    inst✝³ : OrderedCancelAddCommMonoid α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b c : α
    ⊢ Eq (Finset.image (fun x => HAdd.hAdd x c) (Finset.Ioc a b)) (Finset.Ioc (HAd …
  -/
  rw [← map_add_right_Ioc, map_eq_image, addRightEmbedding, Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


@[simp] lemma image_add_right_Ioo (a b c : α) : (Ioo a b).image (· + c) = Ioo (a + c) (b + c) := by
  /-
    α : Type u_2
    inst✝³ : OrderedCancelAddCommMonoid α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b c : α
    ⊢ Eq (Finset.image (fun x => HAdd.hAdd x c) (Finset.Ioo a b)) (Finset.Ioo (HAd …
  -/
  rw [← map_add_right_Ioo, map_eq_image, addRightEmbedding, Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


