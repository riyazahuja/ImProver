/-- The multiset of elements `x` such that `a ≤ x` and `x ≤ b`. Basically `Set.Icc a b` as a
multiset. -/
def Icc (a b : α) : Multiset α := (Finset.Icc a b).val


/-- The multiset of elements `x` such that `a ≤ x` and `x < b`. Basically `Set.Ico a b` as a
multiset. -/
def Ico (a b : α) : Multiset α := (Finset.Ico a b).val


/-- The multiset of elements `x` such that `a < x` and `x ≤ b`. Basically `Set.Ioc a b` as a
multiset. -/
def Ioc (a b : α) : Multiset α := (Finset.Ioc a b).val


/-- The multiset of elements `x` such that `a < x` and `x < b`. Basically `Set.Ioo a b` as a
multiset. -/
def Ioo (a b : α) : Multiset α := (Finset.Ioo a b).val


                                                          /-
                                                            α : Type u_1
                                                            inst✝¹ : Preorder α
                                                            inst✝ : LocallyFiniteOrder α
                                                            a b x : α
                                                            ⊢ Iff (Membership.mem (Multiset.Icc a b) x) (And (LE.le a x) (LE.le x b))
                                                          -/
@[simp] lemma mem_Icc : x ∈ Icc a b ↔ a ≤ x ∧ x ≤ b := by rw [Icc, ← Finset.mem_def, Finset.mem_Icc]
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                          /-
                                                            α : Type u_1
                                                            inst✝¹ : Preorder α
                                                            inst✝ : LocallyFiniteOrder α
                                                            a b x : α
                                                            ⊢ Iff (Membership.mem (Multiset.Ico a b) x) (And (LE.le a x) (LT.lt x b))
                                                          -/
@[simp] lemma mem_Ico : x ∈ Ico a b ↔ a ≤ x ∧ x < b := by rw [Ico, ← Finset.mem_def, Finset.mem_Ico]
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                          /-
                                                            α : Type u_1
                                                            inst✝¹ : Preorder α
                                                            inst✝ : LocallyFiniteOrder α
                                                            a b x : α
                                                            ⊢ Iff (Membership.mem (Multiset.Ioc a b) x) (And (LT.lt a x) (LE.le x b))
                                                          -/
@[simp] lemma mem_Ioc : x ∈ Ioc a b ↔ a < x ∧ x ≤ b := by rw [Ioc, ← Finset.mem_def, Finset.mem_Ioc]
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                          /-
                                                            α : Type u_1
                                                            inst✝¹ : Preorder α
                                                            inst✝ : LocallyFiniteOrder α
                                                            a b x : α
                                                            ⊢ Iff (Membership.mem (Multiset.Ioo a b) x) (And (LT.lt a x) (LT.lt x b))
                                                          -/
@[simp] lemma mem_Ioo : x ∈ Ioo a b ↔ a < x ∧ x < b := by rw [Ioo, ← Finset.mem_def, Finset.mem_Ioo]
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- The multiset of elements `x` such that `a ≤ x`. Basically `Set.Ici a` as a multiset. -/
def Ici (a : α) : Multiset α := (Finset.Ici a).val


/-- The multiset of elements `x` such that `a < x`. Basically `Set.Ioi a` as a multiset. -/
def Ioi (a : α) : Multiset α := (Finset.Ioi a).val


                                                /-
                                                  α : Type u_1
                                                  inst✝¹ : Preorder α
                                                  inst✝ : LocallyFiniteOrderTop α
                                                  a x : α
                                                  ⊢ Iff (Membership.mem (Multiset.Ici a) x) (LE.le a x)
                                                -/
@[simp] lemma mem_Ici : x ∈ Ici a ↔ a ≤ x := by rw [Ici, ← Finset.mem_def, Finset.mem_Ici]
                                                /-
                                                  🎉 no goals
                                                -/


                                                /-
                                                  α : Type u_1
                                                  inst✝¹ : Preorder α
                                                  inst✝ : LocallyFiniteOrderTop α
                                                  a x : α
                                                  ⊢ Iff (Membership.mem (Multiset.Ioi a) x) (LT.lt a x)
                                                -/
@[simp] lemma mem_Ioi : x ∈ Ioi a ↔ a < x := by rw [Ioi, ← Finset.mem_def, Finset.mem_Ioi]
                                                /-
                                                  🎉 no goals
                                                -/


/-- The multiset of elements `x` such that `x ≤ b`. Basically `Set.Iic b` as a multiset. -/
def Iic (b : α) : Multiset α := (Finset.Iic b).val


/-- The multiset of elements `x` such that `x < b`. Basically `Set.Iio b` as a multiset. -/
def Iio (b : α) : Multiset α := (Finset.Iio b).val


                                                /-
                                                  α : Type u_1
                                                  inst✝¹ : Preorder α
                                                  inst✝ : LocallyFiniteOrderBot α
                                                  b x : α
                                                  ⊢ Iff (Membership.mem (Multiset.Iic b) x) (LE.le x b)
                                                -/
@[simp] lemma mem_Iic : x ∈ Iic b ↔ x ≤ b := by rw [Iic, ← Finset.mem_def, Finset.mem_Iic]
                                                /-
                                                  🎉 no goals
                                                -/


                                                /-
                                                  α : Type u_1
                                                  inst✝¹ : Preorder α
                                                  inst✝ : LocallyFiniteOrderBot α
                                                  b x : α
                                                  ⊢ Iff (Membership.mem (Multiset.Iio b) x) (LT.lt x b)
                                                -/
@[simp] lemma mem_Iio : x ∈ Iio b ↔ x < b := by rw [Iio, ← Finset.mem_def, Finset.mem_Iio]
                                                /-
                                                  🎉 no goals
                                                -/


theorem nodup_Icc : (Icc a b).Nodup :=
  Finset.nodup _


theorem nodup_Ico : (Ico a b).Nodup :=
  Finset.nodup _


theorem nodup_Ioc : (Ioc a b).Nodup :=
  Finset.nodup _


theorem nodup_Ioo : (Ioo a b).Nodup :=
  Finset.nodup _


@[simp]
theorem Icc_eq_zero_iff : Icc a b = 0 ↔ ¬a ≤ b := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    a b : α
    ⊢ Iff (Eq (Multiset.Icc a b) 0) (Not (LE.le a b))
  -/
  rw [Icc, Finset.val_eq_zero, Finset.Icc_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ico_eq_zero_iff : Ico a b = 0 ↔ ¬a < b := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    a b : α
    ⊢ Iff (Eq (Multiset.Ico a b) 0) (Not (LT.lt a b))
  -/
  rw [Ico, Finset.val_eq_zero, Finset.Ico_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioc_eq_zero_iff : Ioc a b = 0 ↔ ¬a < b := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    a b : α
    ⊢ Iff (Eq (Multiset.Ioc a b) 0) (Not (LT.lt a b))
  -/
  rw [Ioc, Finset.val_eq_zero, Finset.Ioc_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioo_eq_zero_iff [DenselyOrdered α] : Ioo a b = 0 ↔ ¬a < b := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DenselyOrdered α
    ⊢ Iff (Eq (Multiset.Ioo a b) 0) (Not (LT.lt a b))
  -/
  rw [Ioo, Finset.val_eq_zero, Finset.Ioo_eq_empty_iff]
  /-
    🎉 no goals
  -/


alias ⟨_, Icc_eq_zero⟩ := Icc_eq_zero_iff


alias ⟨_, Ico_eq_zero⟩ := Ico_eq_zero_iff


alias ⟨_, Ioc_eq_zero⟩ := Ioc_eq_zero_iff


@[simp]
theorem Ioo_eq_zero (h : ¬a < b) : Ioo a b = 0 :=
  eq_zero_iff_forall_not_mem.2 fun _x hx => h ((mem_Ioo.1 hx).1.trans (mem_Ioo.1 hx).2)


@[simp]
theorem Icc_eq_zero_of_lt (h : b < a) : Icc a b = 0 :=
  Icc_eq_zero h.not_le


@[simp]
theorem Ico_eq_zero_of_le (h : b ≤ a) : Ico a b = 0 :=
  Ico_eq_zero h.not_lt


@[simp]
theorem Ioc_eq_zero_of_le (h : b ≤ a) : Ioc a b = 0 :=
  Ioc_eq_zero h.not_lt


@[simp]
theorem Ioo_eq_zero_of_le (h : b ≤ a) : Ioo a b = 0 :=
  Ioo_eq_zero h.not_lt


                                     /-
                                       α : Type u_1
                                       inst✝¹ : Preorder α
                                       inst✝ : LocallyFiniteOrder α
                                       a : α
                                       ⊢ Eq (Multiset.Ico a a) 0
                                     -/
theorem Ico_self : Ico a a = 0 := by rw [Ico, Finset.Ico_self, Finset.empty_val]
                                     /-
                                       🎉 no goals
                                     -/


                                     /-
                                       α : Type u_1
                                       inst✝¹ : Preorder α
                                       inst✝ : LocallyFiniteOrder α
                                       a : α
                                       ⊢ Eq (Multiset.Ioc a a) 0
                                     -/
theorem Ioc_self : Ioc a a = 0 := by rw [Ioc, Finset.Ioc_self, Finset.empty_val]
                                     /-
                                       🎉 no goals
                                     -/


                                     /-
                                       α : Type u_1
                                       inst✝¹ : Preorder α
                                       inst✝ : LocallyFiniteOrder α
                                       a : α
                                       ⊢ Eq (Multiset.Ioo a a) 0
                                     -/
theorem Ioo_self : Ioo a a = 0 := by rw [Ioo, Finset.Ioo_self, Finset.empty_val]
                                     /-
                                       🎉 no goals
                                     -/


theorem left_mem_Icc : a ∈ Icc a b ↔ a ≤ b :=
  Finset.left_mem_Icc


theorem left_mem_Ico : a ∈ Ico a b ↔ a < b :=
  Finset.left_mem_Ico


theorem right_mem_Icc : b ∈ Icc a b ↔ a ≤ b :=
  Finset.right_mem_Icc


theorem right_mem_Ioc : b ∈ Ioc a b ↔ a < b :=
  Finset.right_mem_Ioc


theorem left_not_mem_Ioc : a ∉ Ioc a b :=
  Finset.left_not_mem_Ioc


theorem left_not_mem_Ioo : a ∉ Ioo a b :=
  Finset.left_not_mem_Ioo


theorem right_not_mem_Ico : b ∉ Ico a b :=
  Finset.right_not_mem_Ico


theorem right_not_mem_Ioo : b ∉ Ioo a b :=
  Finset.right_not_mem_Ioo


theorem Ico_filter_lt_of_le_left [DecidablePred (· < c)] (hca : c ≤ a) :
    ((Ico a b).filter fun x => x < c) = ∅ := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LT.lt x c
    hca : LE.le c a
    ⊢ Eq (Multiset.filter (fun x => LT.lt x c) (Multiset.Ico a b)) EmptyCollection …
  -/
  rw [Ico, ← Finset.filter_val, Finset.Ico_filter_lt_of_le_left hca]
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LT.lt x c
    hca : LE.le c a
    ⊢ Eq EmptyCollection.emptyCollection.val EmptyCollection.emptyCollection
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Ico_filter_lt_of_right_le [DecidablePred (· < c)] (hbc : b ≤ c) :
    ((Ico a b).filter fun x => x < c) = Ico a b := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LT.lt x c
    hbc : LE.le b c
    ⊢ Eq (Multiset.filter (fun x => LT.lt x c) (Multiset.Ico a b)) (Multiset.Ico a …
  -/
  rw [Ico, ← Finset.filter_val, Finset.Ico_filter_lt_of_right_le hbc]
  /-
    🎉 no goals
  -/


theorem Ico_filter_lt_of_le_right [DecidablePred (· < c)] (hcb : c ≤ b) :
    ((Ico a b).filter fun x => x < c) = Ico a c := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LT.lt x c
    hcb : LE.le c b
    ⊢ Eq (Multiset.filter (fun x => LT.lt x c) (Multiset.Ico a b)) (Multiset.Ico a …
  -/
  rw [Ico, ← Finset.filter_val, Finset.Ico_filter_lt_of_le_right hcb]
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LT.lt x c
    hcb : LE.le c b
    ⊢ Eq (Finset.Ico a c).val (Multiset.Ico a c)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Ico_filter_le_of_le_left [DecidablePred (c ≤ ·)] (hca : c ≤ a) :
    ((Ico a b).filter fun x => c ≤ x) = Ico a b := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LE.le c x
    hca : LE.le c a
    ⊢ Eq (Multiset.filter (fun x => LE.le c x) (Multiset.Ico a b)) (Multiset.Ico a …
  -/
  rw [Ico, ← Finset.filter_val, Finset.Ico_filter_le_of_le_left hca]
  /-
    🎉 no goals
  -/


theorem Ico_filter_le_of_right_le [DecidablePred (b ≤ ·)] :
    ((Ico a b).filter fun x => b ≤ x) = ∅ := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidablePred fun x => LE.le b x
    ⊢ Eq (Multiset.filter (fun x => LE.le b x) (Multiset.Ico a b)) EmptyCollection …
  -/
  rw [Ico, ← Finset.filter_val, Finset.Ico_filter_le_of_right_le]
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidablePred fun x => LE.le b x
    ⊢ Eq EmptyCollection.emptyCollection.val EmptyCollection.emptyCollection
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Ico_filter_le_of_left_le [DecidablePred (c ≤ ·)] (hac : a ≤ c) :
    ((Ico a b).filter fun x => c ≤ x) = Ico c b := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LE.le c x
    hac : LE.le a c
    ⊢ Eq (Multiset.filter (fun x => LE.le c x) (Multiset.Ico a b)) (Multiset.Ico c …
  -/
  rw [Ico, ← Finset.filter_val, Finset.Ico_filter_le_of_left_le hac]
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b c : α
    inst✝ : DecidablePred fun x => LE.le c x
    hac : LE.le a c
    ⊢ Eq (Finset.Ico c b).val (Multiset.Ico c b)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
                                               /-
                                                 α : Type u_1
                                                 inst✝¹ : PartialOrder α
                                                 inst✝ : LocallyFiniteOrder α
                                                 a : α
                                                 ⊢ Eq (Multiset.Icc a a) (Singleton.singleton a)
                                               -/
theorem Icc_self (a : α) : Icc a a = {a} := by rw [Icc, Finset.Icc_self, Finset.singleton_val]
                                               /-
                                                 🎉 no goals
                                               -/


theorem Ico_cons_right (h : a ≤ b) : b ::ₘ Ico a b = Icc a b := by
  classical
    rw [Ico, ← Finset.insert_val_of_not_mem right_not_mem_Ico, Finset.Ico_insert_right h]
    rfl


theorem Ioo_cons_left (h : a < b) : a ::ₘ Ioo a b = Ico a b := by
  classical
    rw [Ioo, ← Finset.insert_val_of_not_mem left_not_mem_Ioo, Finset.Ioo_insert_left h]
    rfl


theorem Ico_disjoint_Ico {a b c d : α} (h : b ≤ c) : Disjoint (Ico a b) (Ico c d) :=
  disjoint_left.mpr fun hab hbc => by
    /-
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : LocallyFiniteOrder α
      a b c d : α
      h : LE.le b c
      a✝ : α
      hab : Membership.mem (Multiset.Ico a b) a✝
      hbc : Membership.mem (Multiset.Ico c d) a✝
      ⊢ False
    -/
    rw [mem_Ico] at hab hbc
    /-
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : LocallyFiniteOrder α
      a b c d : α
      h : LE.le b c
      a✝ : α
      hab : And (LE.le a a✝) (LT.lt a✝ b)
      hbc : And (LE.le c a✝) (LT.lt a✝ d)
      ⊢ False
    -/
    exact hab.2.not_le (h.trans hbc.1)
    /-
      🎉 no goals
    -/


@[simp]
theorem Ico_inter_Ico_of_le [DecidableEq α] {a b c d : α} (h : b ≤ c) : Ico a b ∩ Ico c d = 0 :=
  Multiset.inter_eq_zero_iff_disjoint.2 <| Ico_disjoint_Ico h


theorem Ico_filter_le_left {a b : α} [DecidablePred (· ≤ a)] (hab : a < b) :
    ((Ico a b).filter fun x => x ≤ a) = {a} := by
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidablePred fun x => LE.le x a
    hab : LT.lt a b
    ⊢ Eq (Multiset.filter (fun x => LE.le x a) (Multiset.Ico a b)) (Singleton.sing …
  -/
  rw [Ico, ← Finset.filter_val, Finset.Ico_filter_le_left hab]
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : DecidablePred fun x => LE.le x a
    hab : LT.lt a b
    ⊢ Eq (Singleton.singleton a).val (Singleton.singleton a)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem card_Ico_eq_card_Icc_sub_one (a b : α) : card (Ico a b) = card (Icc a b) - 1 :=
  Finset.card_Ico_eq_card_Icc_sub_one _ _


theorem card_Ioc_eq_card_Icc_sub_one (a b : α) : card (Ioc a b) = card (Icc a b) - 1 :=
  Finset.card_Ioc_eq_card_Icc_sub_one _ _


theorem card_Ioo_eq_card_Ico_sub_one (a b : α) : card (Ioo a b) = card (Ico a b) - 1 :=
  Finset.card_Ioo_eq_card_Ico_sub_one _ _


theorem card_Ioo_eq_card_Icc_sub_two (a b : α) : card (Ioo a b) = card (Icc a b) - 2 :=
  Finset.card_Ioo_eq_card_Icc_sub_two _ _


theorem Ico_subset_Ico_iff {a₁ b₁ a₂ b₂ : α} (h : a₁ < b₁) :
    Ico a₁ b₁ ⊆ Ico a₂ b₂ ↔ a₂ ≤ a₁ ∧ b₁ ≤ b₂ :=
  Finset.Ico_subset_Ico_iff h


theorem Ico_add_Ico_eq_Ico {a b c : α} (hab : a ≤ b) (hbc : b ≤ c) :
    Ico a b + Ico b c = Ico a c := by
  rw [add_eq_union_iff_disjoint.2 (Ico_disjoint_Ico le_rfl), Ico, Ico, Ico, ← Finset.union_val,
    Finset.Ico_union_Ico_eq_Ico hab hbc]


theorem Ico_inter_Ico : Ico a b ∩ Ico c d = Ico (max a c) (min b d) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c d : α
    ⊢ Eq (Inter.inter (Multiset.Ico a b) (Multiset.Ico c d)) (Multiset.Ico (Max.ma …
  -/
  rw [Ico, Ico, Ico, ← Finset.inter_val, Finset.Ico_inter_Ico]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ico_filter_lt (a b c : α) : ((Ico a b).filter fun x => x < c) = Ico a (min b c) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Multiset.filter (fun x => LT.lt x c) (Multiset.Ico a b)) (Multiset.Ico a …
  -/
  rw [Ico, Ico, ← Finset.filter_val, Finset.Ico_filter_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ico_filter_le (a b c : α) : ((Ico a b).filter fun x => c ≤ x) = Ico (max a c) b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (Multiset.filter (fun x => LE.le c x) (Multiset.Ico a b)) (Multiset.Ico ( …
  -/
  rw [Ico, Ico, ← Finset.filter_val, Finset.Ico_filter_le]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ico_sub_Ico_left (a b c : α) : Ico a b - Ico a c = Ico (max a c) b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (HSub.hSub (Multiset.Ico a b) (Multiset.Ico a c)) (Multiset.Ico (Max.max  …
  -/
  rw [Ico, Ico, Ico, ← Finset.sdiff_val, Finset.Ico_diff_Ico_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ico_sub_Ico_right (a b c : α) : Ico a b - Ico c b = Ico a (min b c) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : LocallyFiniteOrder α
    a b c : α
    ⊢ Eq (HSub.hSub (Multiset.Ico a b) (Multiset.Ico c b)) (Multiset.Ico a (Min.mi …
  -/
  rw [Ico, Ico, Ico, ← Finset.sdiff_val, Finset.Ico_diff_Ico_right]
  /-
    🎉 no goals
  -/


