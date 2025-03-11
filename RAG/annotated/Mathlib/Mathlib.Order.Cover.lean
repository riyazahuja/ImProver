/-- `WCovBy a b` means that `a = b` or `b` covers `a`.
This means that `a ≤ b` and there is no element in between.
-/
def WCovBy (a b : α) : Prop :=
  a ≤ b ∧ ∀ ⦃c⦄, a < c → ¬c < b


/-- Notation for `WCovBy a b`. -/
infixl:50 " ⩿ " => WCovBy


theorem WCovBy.le (h : a ⩿ b) : a ≤ b :=
  h.1


theorem WCovBy.refl (a : α) : a ⩿ a :=
  ⟨le_rfl, fun _ hc => hc.not_lt⟩


@[simp] lemma WCovBy.rfl : a ⩿ a := WCovBy.refl a


protected theorem Eq.wcovBy (h : a = b) : a ⩿ b :=
  h ▸ WCovBy.rfl


theorem wcovBy_of_le_of_le (h1 : a ≤ b) (h2 : b ≤ a) : a ⩿ b :=
  ⟨h1, fun _ hac hcb => (hac.trans hcb).not_le h2⟩


alias LE.le.wcovBy_of_le := wcovBy_of_le_of_le


theorem AntisymmRel.wcovBy (h : AntisymmRel (· ≤ ·) a b) : a ⩿ b :=
  wcovBy_of_le_of_le h.1 h.2


theorem WCovBy.wcovBy_iff_le (hab : a ⩿ b) : b ⩿ a ↔ b ≤ a :=
  ⟨fun h => h.le, fun h => h.wcovBy_of_le hab.le⟩


theorem wcovBy_of_eq_or_eq (hab : a ≤ b) (h : ∀ c, a ≤ c → c ≤ b → c = a ∨ c = b) : a ⩿ b :=
  ⟨hab, fun c ha hb => (h c ha.le hb.le).elim ha.ne' hb.ne⟩


theorem AntisymmRel.trans_wcovBy (hab : AntisymmRel (· ≤ ·) a b) (hbc : b ⩿ c) : a ⩿ c :=
  ⟨hab.1.trans hbc.le, fun _ had hdc => hbc.2 (hab.2.trans_lt had) hdc⟩


theorem wcovBy_congr_left (hab : AntisymmRel (· ≤ ·) a b) : a ⩿ c ↔ b ⩿ c :=
  ⟨hab.symm.trans_wcovBy, hab.trans_wcovBy⟩


theorem WCovBy.trans_antisymm_rel (hab : a ⩿ b) (hbc : AntisymmRel (· ≤ ·) b c) : a ⩿ c :=
  ⟨hab.le.trans hbc.1, fun _ had hdc => hab.2 had <| hdc.trans_le hbc.2⟩


theorem wcovBy_congr_right (hab : AntisymmRel (· ≤ ·) a b) : c ⩿ a ↔ c ⩿ b :=
  ⟨fun h => h.trans_antisymm_rel hab, fun h => h.trans_antisymm_rel hab.symm⟩


/-- If `a ≤ b`, then `b` does not cover `a` iff there's an element in between. -/
theorem not_wcovBy_iff (h : a ≤ b) : ¬a ⩿ b ↔ ∃ c, a < c ∧ c < b := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a b : α
    h : LE.le a b
    ⊢ Iff (Not (WCovBy a b)) (Exists fun c => And (LT.lt a c) (LT.lt c b))
  -/
  simp_rw [WCovBy, h, true_and, not_forall, exists_prop, not_not]
  /-
    🎉 no goals
  -/


instance WCovBy.isRefl : IsRefl α (· ⩿ ·) :=
  ⟨WCovBy.refl⟩


theorem WCovBy.Ioo_eq (h : a ⩿ b) : Ioo a b = ∅ :=
  eq_empty_iff_forall_not_mem.2 fun _ hx => h.2 hx.1 hx.2


theorem wcovBy_iff_Ioo_eq : a ⩿ b ↔ a ≤ b ∧ Ioo a b = ∅ :=
                         /-
                           α : Type u_1
                           inst✝ : Preorder α
                           a b : α
                           ⊢ Iff (∀ ⦃c : α⦄, LT.lt a c → Not (LT.lt c b)) (Eq (Set.Ioo a b) EmptyCollecti …
                         -/
  and_congr_right' <| by simp [eq_empty_iff_forall_not_mem]
                         /-
                           🎉 no goals
                         -/


lemma WCovBy.of_le_of_le (hac : a ⩿ c) (hab : a ≤ b) (hbc : b ≤ c) : b ⩿ c :=
  ⟨hbc, fun _x hbx hxc ↦ hac.2 (hab.trans_lt hbx) hxc⟩


lemma WCovBy.of_le_of_le' (hac : a ⩿ c) (hab : a ≤ b) (hbc : b ≤ c) : a ⩿ b :=
  ⟨hab, fun _x hax hxb ↦ hac.2 hax <| hxb.trans_le hbc⟩


theorem WCovBy.of_image (f : α ↪o β) (h : f a ⩿ f b) : a ⩿ b :=
  ⟨f.le_iff_le.mp h.le, fun _ hac hcb => h.2 (f.lt_iff_lt.mpr hac) (f.lt_iff_lt.mpr hcb)⟩


theorem WCovBy.image (f : α ↪o β) (hab : a ⩿ b) (h : (range f).OrdConnected) : f a ⩿ f b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    a b : α
    f : OrderEmbedding α β
    hab : WCovBy a b
    h : (Set.range ⇑f).OrdConnected
    ⊢ WCovBy (f a) (f b)
  -/
  refine ⟨f.monotone hab.le, fun c ha hb => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    a b : α
    f : OrderEmbedding α β
    hab : WCovBy a b
    h : (Set.range ⇑f).OrdConnected
    c : β
    ha : LT.lt (f a) c
    hb : LT.lt c (f b)
    ⊢ False
  -/
  obtain ⟨c, rfl⟩ := h.out (mem_range_self _) (mem_range_self _) ⟨ha.le, hb.le⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    a b : α
    f : OrderEmbedding α β
    hab : WCovBy a b
    h : (Set.range ⇑f).OrdConnected
    c : α
    ha : LT.lt (f a) (f c)
    hb : LT.lt (f c) (f b)
    ⊢ False
  -/
  rw [f.lt_iff_lt] at ha hb
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    a b : α
    f : OrderEmbedding α β
    hab : WCovBy a b
    h : (Set.range ⇑f).OrdConnected
    c : α
    ha : LT.lt a c
    hb : LT.lt c b
    ⊢ False
  -/
  exact hab.2 ha hb
  /-
    🎉 no goals
  -/


theorem Set.OrdConnected.apply_wcovBy_apply_iff (f : α ↪o β) (h : (range f).OrdConnected) :
    f a ⩿ f b ↔ a ⩿ b :=
  ⟨fun h2 => h2.of_image f, fun hab => hab.image f h⟩


@[simp]
theorem apply_wcovBy_apply_iff {E : Type*} [EquivLike E α β] [OrderIsoClass E α β] (e : E) :
    e a ⩿ e b ↔ a ⩿ b :=
  (ordConnected_range (e : α ≃o β)).apply_wcovBy_apply_iff ((e : α ≃o β) : α ↪o β)


@[simp]
theorem toDual_wcovBy_toDual_iff : toDual b ⩿ toDual a ↔ a ⩿ b :=
  and_congr_right' <| forall_congr' fun _ => forall_swap


@[simp]
theorem ofDual_wcovBy_ofDual_iff {a b : αᵒᵈ} : ofDual a ⩿ ofDual b ↔ b ⩿ a :=
  and_congr_right' <| forall_congr' fun _ => forall_swap


alias ⟨_, WCovBy.toDual⟩ := toDual_wcovBy_toDual_iff


alias ⟨_, WCovBy.ofDual⟩ := ofDual_wcovBy_ofDual_iff


theorem OrderEmbedding.wcovBy_of_apply {α β : Type*} [Preorder α] [Preorder β]
    (f : α ↪o β) {x y : α} (h : f x ⩿ f y) : x ⩿ y := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderEmbedding α β
    x y : α
    h : WCovBy (f x) (f y)
    ⊢ WCovBy x y
  -/
  use f.le_iff_le.1 h.1
  /-
    case right
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderEmbedding α β
    x y : α
    h : WCovBy (f x) (f y)
    ⊢ ∀ ⦃c : α⦄, LT.lt x c → Not (LT.lt c y)
  -/
  intro a
  /-
    case right
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderEmbedding α β
    x y : α
    h : WCovBy (f x) (f y)
    a : α
    ⊢ LT.lt x a → Not (LT.lt a y)
  -/
  rw [← f.lt_iff_lt, ← f.lt_iff_lt]
  /-
    case right
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderEmbedding α β
    x y : α
    h : WCovBy (f x) (f y)
    a : α
    ⊢ LT.lt (f x) (f a) → Not (LT.lt (f a) (f y))
  -/
  apply h.2
  /-
    🎉 no goals
  -/


theorem OrderIso.map_wcovBy {α β : Type*} [Preorder α] [Preorder β]
    (f : α ≃o β) {x y : α} : f x ⩿ f y ↔ x ⩿ y := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x y : α
    ⊢ Iff (WCovBy (f x) (f y)) (WCovBy x y)
  -/
  use f.toOrderEmbedding.wcovBy_of_apply
  /-
    case mpr
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x y : α
    ⊢ WCovBy x y → WCovBy (f x) (f y)
  -/
  conv_lhs => rw [← f.symm_apply_apply x, ← f.symm_apply_apply y]
  /-
    case mpr
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x y : α
    ⊢ WCovBy (f.symm (f x)) (f.symm (f y)) → WCovBy (f x) (f y)
  -/
  exact f.symm.toOrderEmbedding.wcovBy_of_apply
  /-
    🎉 no goals
  -/


theorem WCovBy.eq_or_eq (h : a ⩿ b) (h2 : a ≤ c) (h3 : c ≤ b) : c = a ∨ c = b := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b c : α
    h : WCovBy a b
    h2 : LE.le a c
    h3 : LE.le c b
    ⊢ Or (Eq c a) (Eq c b)
  -/
  rcases h2.eq_or_lt with (h2 | h2); · exact Or.inl h2.symm
                                       /-
                                         🎉 no goals
                                       -/
  /-
    case inr
    α : Type u_1
    inst✝ : PartialOrder α
    a b c : α
    h : WCovBy a b
    h2✝ : LE.le a c
    h3 : LE.le c b
    h2 : LT.lt a c
    ⊢ Or (Eq c a) (Eq c b)
  -/
  rcases h3.eq_or_lt with (h3 | h3); · exact Or.inr h3
                                       /-
                                         🎉 no goals
                                       -/
  /-
    case inr.inr
    α : Type u_1
    inst✝ : PartialOrder α
    a b c : α
    h : WCovBy a b
    h2✝ : LE.le a c
    h3✝ : LE.le c b
    h2 : LT.lt a c
    h3 : LT.lt c b
    ⊢ Or (Eq c a) (Eq c b)
  -/
  exact (h.2 h2 h3).elim
  /-
    🎉 no goals
  -/


/-- An `iff` version of `WCovBy.eq_or_eq` and `wcovBy_of_eq_or_eq`. -/
theorem wcovBy_iff_le_and_eq_or_eq : a ⩿ b ↔ a ≤ b ∧ ∀ c, a ≤ c → c ≤ b → c = a ∨ c = b :=
  ⟨fun h => ⟨h.le, fun _ => h.eq_or_eq⟩, And.rec wcovBy_of_eq_or_eq⟩


theorem WCovBy.le_and_le_iff (h : a ⩿ b) : a ≤ c ∧ c ≤ b ↔ c = a ∨ c = b := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b c : α
    h : WCovBy a b
    ⊢ Iff (And (LE.le a c) (LE.le c b)) (Or (Eq c a) (Eq c b))
  -/
  refine ⟨fun h2 => h.eq_or_eq h2.1 h2.2, ?_⟩; rintro (rfl | rfl)
  /-
    case inl
    α : Type u_1
    inst✝ : PartialOrder α
    b c : α
    h : WCovBy c b
    ⊢ And (LE.le c c) (LE.le c b)
  -/
  exacts [⟨le_rfl, h.le⟩, ⟨h.le, le_rfl⟩]
  /-
    🎉 no goals
  -/


theorem WCovBy.Icc_eq (h : a ⩿ b) : Icc a b = {a, b} := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b : α
    h : WCovBy a b
    ⊢ Eq (Set.Icc a b) (Insert.insert a (Singleton.singleton b))
  -/
  ext c
  /-
    case h
    α : Type u_1
    inst✝ : PartialOrder α
    a b : α
    h : WCovBy a b
    c : α
    ⊢ Iff (Membership.mem (Set.Icc a b) c) (Membership.mem (Insert.insert a (Singl …
  -/
  exact h.le_and_le_iff
  /-
    🎉 no goals
  -/


theorem WCovBy.Ico_subset (h : a ⩿ b) : Ico a b ⊆ {a} := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b : α
    h : WCovBy a b
    ⊢ HasSubset.Subset (Set.Ico a b) (Singleton.singleton a)
  -/
  rw [← Icc_diff_right, h.Icc_eq, diff_singleton_subset_iff, pair_comm]
  /-
    🎉 no goals
  -/


theorem WCovBy.Ioc_subset (h : a ⩿ b) : Ioc a b ⊆ {b} := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b : α
    h : WCovBy a b
    ⊢ HasSubset.Subset (Set.Ioc a b) (Singleton.singleton b)
  -/
  rw [← Icc_diff_left, h.Icc_eq, diff_singleton_subset_iff]
  /-
    🎉 no goals
  -/


theorem WCovBy.sup_eq (hac : a ⩿ c) (hbc : b ⩿ c) (hab : a ≠ b) : a ⊔ b = c :=
  (sup_le hac.le hbc.le).eq_of_not_lt fun h =>
    hab.lt_sup_or_lt_sup.elim (fun h' => hac.2 h' h) fun h' => hbc.2 h' h


theorem WCovBy.inf_eq (hca : c ⩿ a) (hcb : c ⩿ b) (hab : a ≠ b) : a ⊓ b = c :=
  (le_inf hca.le hcb.le).eq_of_not_gt fun h => hab.inf_lt_or_inf_lt.elim (hca.2 h) (hcb.2 h)


/-- `CovBy a b` means that `b` covers `a`: `a < b` and there is no element in between. -/
def CovBy (a b : α) : Prop :=
  a < b ∧ ∀ ⦃c⦄, a < c → ¬c < b


/-- Notation for `CovBy a b`. -/
infixl:50 " ⋖ " => CovBy


theorem CovBy.lt (h : a ⋖ b) : a < b :=
  h.1


/-- If `a < b`, then `b` does not cover `a` iff there's an element in between. -/
theorem not_covBy_iff (h : a < b) : ¬a ⋖ b ↔ ∃ c, a < c ∧ c < b := by
  /-
    α : Type u_1
    inst✝ : LT α
    a b : α
    h : LT.lt a b
    ⊢ Iff (Not (CovBy a b)) (Exists fun c => And (LT.lt a c) (LT.lt c b))
  -/
  simp_rw [CovBy, h, true_and, not_forall, exists_prop, not_not]
  /-
    🎉 no goals
  -/


alias ⟨exists_lt_lt_of_not_covBy, _⟩ := not_covBy_iff


alias LT.lt.exists_lt_lt := exists_lt_lt_of_not_covBy


/-- In a dense order, nothing covers anything. -/
theorem not_covBy [DenselyOrdered α] : ¬a ⋖ b := fun h =>
  let ⟨_, hc⟩ := exists_between h.1
  h.2 hc.1 hc.2


theorem denselyOrdered_iff_forall_not_covBy : DenselyOrdered α ↔ ∀ a b : α, ¬a ⋖ b :=
  ⟨fun h _ _ => @not_covBy _ _ _ _ h, fun h =>
    ⟨fun _ _ hab => exists_lt_lt_of_not_covBy hab <| h _ _⟩⟩


@[deprecated (since := "2024-04-04")]
alias densely_ordered_iff_forall_not_covBy := denselyOrdered_iff_forall_not_covBy


@[simp]
theorem toDual_covBy_toDual_iff : toDual b ⋖ toDual a ↔ a ⋖ b :=
  and_congr_right' <| forall_congr' fun _ => forall_swap


@[simp]
theorem ofDual_covBy_ofDual_iff {a b : αᵒᵈ} : ofDual a ⋖ ofDual b ↔ b ⋖ a :=
  and_congr_right' <| forall_congr' fun _ => forall_swap


alias ⟨_, CovBy.toDual⟩ := toDual_covBy_toDual_iff


alias ⟨_, CovBy.ofDual⟩ := ofDual_covBy_ofDual_iff


theorem CovBy.le (h : a ⋖ b) : a ≤ b :=
  h.1.le


protected theorem CovBy.ne (h : a ⋖ b) : a ≠ b :=
  h.lt.ne


theorem CovBy.ne' (h : a ⋖ b) : b ≠ a :=
  h.lt.ne'


protected theorem CovBy.wcovBy (h : a ⋖ b) : a ⩿ b :=
  ⟨h.le, h.2⟩


theorem WCovBy.covBy_of_not_le (h : a ⩿ b) (h2 : ¬b ≤ a) : a ⋖ b :=
  ⟨h.le.lt_of_not_le h2, h.2⟩


theorem WCovBy.covBy_of_lt (h : a ⩿ b) (h2 : a < b) : a ⋖ b :=
  ⟨h2, h.2⟩


lemma CovBy.of_le_of_lt (hac : a ⋖ c) (hab : a ≤ b) (hbc : b < c) : b ⋖ c :=
  ⟨hbc, fun _x hbx hxc ↦ hac.2 (hab.trans_lt hbx) hxc⟩


lemma CovBy.of_lt_of_le (hac : a ⋖ c) (hab : a < b) (hbc : b ≤ c) : a ⋖ b :=
  ⟨hab, fun _x hax hxb ↦ hac.2 hax <| hxb.trans_le hbc⟩


theorem not_covBy_of_lt_of_lt (h₁ : a < b) (h₂ : b < c) : ¬a ⋖ c :=
  (not_covBy_iff (h₁.trans h₂)).2 ⟨b, h₁, h₂⟩


theorem covBy_iff_wcovBy_and_lt : a ⋖ b ↔ a ⩿ b ∧ a < b :=
  ⟨fun h => ⟨h.wcovBy, h.lt⟩, fun h => h.1.covBy_of_lt h.2⟩


theorem covBy_iff_wcovBy_and_not_le : a ⋖ b ↔ a ⩿ b ∧ ¬b ≤ a :=
  ⟨fun h => ⟨h.wcovBy, h.lt.not_le⟩, fun h => h.1.covBy_of_not_le h.2⟩


theorem wcovBy_iff_covBy_or_le_and_le : a ⩿ b ↔ a ⋖ b ∨ a ≤ b ∧ b ≤ a :=
  ⟨fun h => or_iff_not_imp_right.mpr fun h' => h.covBy_of_not_le fun hba => h' ⟨h.le, hba⟩,
    fun h' => h'.elim (fun h => h.wcovBy) fun h => h.1.wcovBy_of_le h.2⟩


alias ⟨WCovBy.covBy_or_le_and_le, _⟩ := wcovBy_iff_covBy_or_le_and_le


theorem AntisymmRel.trans_covBy (hab : AntisymmRel (· ≤ ·) a b) (hbc : b ⋖ c) : a ⋖ c :=
  ⟨hab.1.trans_lt hbc.lt, fun _ had hdc => hbc.2 (hab.2.trans_lt had) hdc⟩


theorem covBy_congr_left (hab : AntisymmRel (· ≤ ·) a b) : a ⋖ c ↔ b ⋖ c :=
  ⟨hab.symm.trans_covBy, hab.trans_covBy⟩


theorem CovBy.trans_antisymmRel (hab : a ⋖ b) (hbc : AntisymmRel (· ≤ ·) b c) : a ⋖ c :=
  ⟨hab.lt.trans_le hbc.1, fun _ had hdb => hab.2 had <| hdb.trans_le hbc.2⟩


theorem covBy_congr_right (hab : AntisymmRel (· ≤ ·) a b) : c ⋖ a ↔ c ⋖ b :=
  ⟨fun h => h.trans_antisymmRel hab, fun h => h.trans_antisymmRel hab.symm⟩


instance : IsNonstrictStrictOrder α (· ⩿ ·) (· ⋖ ·) :=
  ⟨fun _ _ =>
    covBy_iff_wcovBy_and_not_le.trans <| and_congr_right fun h => h.wcovBy_iff_le.not.symm⟩


instance CovBy.isIrrefl : IsIrrefl α (· ⋖ ·) :=
  ⟨fun _ ha => ha.ne rfl⟩


theorem CovBy.Ioo_eq (h : a ⋖ b) : Ioo a b = ∅ :=
  h.wcovBy.Ioo_eq


theorem covBy_iff_Ioo_eq : a ⋖ b ↔ a < b ∧ Ioo a b = ∅ :=
                         /-
                           α : Type u_1
                           inst✝ : Preorder α
                           a b : α
                           ⊢ Iff (∀ ⦃c : α⦄, LT.lt a c → Not (LT.lt c b)) (Eq (Set.Ioo a b) EmptyCollecti …
                         -/
  and_congr_right' <| by simp [eq_empty_iff_forall_not_mem]
                         /-
                           🎉 no goals
                         -/


theorem CovBy.of_image (f : α ↪o β) (h : f a ⋖ f b) : a ⋖ b :=
  ⟨f.lt_iff_lt.mp h.lt, fun _ hac hcb => h.2 (f.lt_iff_lt.mpr hac) (f.lt_iff_lt.mpr hcb)⟩


theorem CovBy.image (f : α ↪o β) (hab : a ⋖ b) (h : (range f).OrdConnected) : f a ⋖ f b :=
  (hab.wcovBy.image f h).covBy_of_lt <| f.strictMono hab.lt


theorem Set.OrdConnected.apply_covBy_apply_iff (f : α ↪o β) (h : (range f).OrdConnected) :
    f a ⋖ f b ↔ a ⋖ b :=
  ⟨CovBy.of_image f, fun hab => hab.image f h⟩


@[simp]
theorem apply_covBy_apply_iff {E : Type*} [EquivLike E α β] [OrderIsoClass E α β] (e : E) :
    e a ⋖ e b ↔ a ⋖ b :=
  (ordConnected_range (e : α ≃o β)).apply_covBy_apply_iff ((e : α ≃o β) : α ↪o β)


theorem covBy_of_eq_or_eq (hab : a < b) (h : ∀ c, a ≤ c → c ≤ b → c = a ∨ c = b) : a ⋖ b :=
  ⟨hab, fun c ha hb => (h c ha.le hb.le).elim ha.ne' hb.ne⟩


theorem OrderEmbedding.covBy_of_apply {α β : Type*} [Preorder α] [Preorder β]
    (f : α ↪o β) {x y : α} (h : f x ⋖ f y) : x ⋖ y := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderEmbedding α β
    x y : α
    h : CovBy (f x) (f y)
    ⊢ CovBy x y
  -/
  use f.lt_iff_lt.1 h.1
  /-
    case right
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderEmbedding α β
    x y : α
    h : CovBy (f x) (f y)
    ⊢ ∀ ⦃c : α⦄, LT.lt x c → Not (LT.lt c y)
  -/
  intro a
  /-
    case right
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderEmbedding α β
    x y : α
    h : CovBy (f x) (f y)
    a : α
    ⊢ LT.lt x a → Not (LT.lt a y)
  -/
  rw [← f.lt_iff_lt, ← f.lt_iff_lt]
  /-
    case right
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderEmbedding α β
    x y : α
    h : CovBy (f x) (f y)
    a : α
    ⊢ LT.lt (f x) (f a) → Not (LT.lt (f a) (f y))
  -/
  apply h.2
  /-
    🎉 no goals
  -/


theorem OrderIso.map_covBy {α β : Type*} [Preorder α] [Preorder β]
    (f : α ≃o β) {x y : α} : f x ⋖ f y ↔ x ⋖ y := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x y : α
    ⊢ Iff (CovBy (f x) (f y)) (CovBy x y)
  -/
  use f.toOrderEmbedding.covBy_of_apply
  /-
    case mpr
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x y : α
    ⊢ CovBy x y → CovBy (f x) (f y)
  -/
  conv_lhs => rw [← f.symm_apply_apply x, ← f.symm_apply_apply y]
  /-
    case mpr
    α : Type u_3
    β : Type u_4
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x y : α
    ⊢ CovBy (f.symm (f x)) (f.symm (f y)) → CovBy (f x) (f y)
  -/
  exact f.symm.toOrderEmbedding.covBy_of_apply
  /-
    🎉 no goals
  -/


theorem WCovBy.covBy_of_ne (h : a ⩿ b) (h2 : a ≠ b) : a ⋖ b :=
  ⟨h.le.lt_of_ne h2, h.2⟩


theorem covBy_iff_wcovBy_and_ne : a ⋖ b ↔ a ⩿ b ∧ a ≠ b :=
  ⟨fun h => ⟨h.wcovBy, h.ne⟩, fun h => h.1.covBy_of_ne h.2⟩


theorem wcovBy_iff_covBy_or_eq : a ⩿ b ↔ a ⋖ b ∨ a = b := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b : α
    ⊢ Iff (WCovBy a b) (Or (CovBy a b) (Eq a b))
  -/
  rw [le_antisymm_iff, wcovBy_iff_covBy_or_le_and_le]
  /-
    🎉 no goals
  -/


theorem wcovBy_iff_eq_or_covBy : a ⩿ b ↔ a = b ∨ a ⋖ b :=
  wcovBy_iff_covBy_or_eq.trans or_comm


alias ⟨WCovBy.covBy_or_eq, _⟩ := wcovBy_iff_covBy_or_eq


alias ⟨WCovBy.eq_or_covBy, _⟩ := wcovBy_iff_eq_or_covBy


theorem CovBy.eq_or_eq (h : a ⋖ b) (h2 : a ≤ c) (h3 : c ≤ b) : c = a ∨ c = b :=
  h.wcovBy.eq_or_eq h2 h3


/-- An `iff` version of `CovBy.eq_or_eq` and `covBy_of_eq_or_eq`. -/
theorem covBy_iff_lt_and_eq_or_eq : a ⋖ b ↔ a < b ∧ ∀ c, a ≤ c → c ≤ b → c = a ∨ c = b :=
  ⟨fun h => ⟨h.lt, fun _ => h.eq_or_eq⟩, And.rec covBy_of_eq_or_eq⟩


theorem CovBy.Ico_eq (h : a ⋖ b) : Ico a b = {a} := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b : α
    h : CovBy a b
    ⊢ Eq (Set.Ico a b) (Singleton.singleton a)
  -/
  rw [← Ioo_union_left h.lt, h.Ioo_eq, empty_union]
  /-
    🎉 no goals
  -/


theorem CovBy.Ioc_eq (h : a ⋖ b) : Ioc a b = {b} := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b : α
    h : CovBy a b
    ⊢ Eq (Set.Ioc a b) (Singleton.singleton b)
  -/
  rw [← Ioo_union_right h.lt, h.Ioo_eq, empty_union]
  /-
    🎉 no goals
  -/


theorem CovBy.Icc_eq (h : a ⋖ b) : Icc a b = {a, b} :=
  h.wcovBy.Icc_eq


theorem CovBy.Ioi_eq (h : a ⋖ b) : Ioi a = Ici b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : CovBy a b
    ⊢ Eq (Set.Ioi a) (Set.Ici b)
  -/
  rw [← Ioo_union_Ici_eq_Ioi h.lt, h.Ioo_eq, empty_union]
  /-
    🎉 no goals
  -/


theorem CovBy.Iio_eq (h : a ⋖ b) : Iio b = Iic a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : CovBy a b
    ⊢ Eq (Set.Iio b) (Set.Iic a)
  -/
  rw [← Iic_union_Ioo_eq_Iio h.lt, h.Ioo_eq, union_empty]
  /-
    🎉 no goals
  -/


theorem WCovBy.le_of_lt (hab : a ⩿ b) (hcb : c < b) : c ≤ a :=
  not_lt.1 fun hac => hab.2 hac hcb


theorem WCovBy.ge_of_gt (hab : a ⩿ b) (hac : a < c) : b ≤ c :=
  not_lt.1 <| hab.2 hac


theorem CovBy.le_of_lt (hab : a ⋖ b) : c < b → c ≤ a :=
  hab.wcovBy.le_of_lt


theorem CovBy.ge_of_gt (hab : a ⋖ b) : a < c → b ≤ c :=
  hab.wcovBy.ge_of_gt


theorem CovBy.unique_left (ha : a ⋖ c) (hb : b ⋖ c) : a = b :=
  (hb.le_of_lt ha.lt).antisymm <| ha.le_of_lt hb.lt


theorem CovBy.unique_right (hb : a ⋖ b) (hc : a ⋖ c) : b = c :=
  (hb.ge_of_gt hc.lt).antisymm <| hc.ge_of_gt hb.lt


/-- If `a`, `b`, `c` are consecutive and `a < x < c` then `x = b`. -/
theorem CovBy.eq_of_between {x : α} (hab : a ⋖ b) (hbc : b ⋖ c) (hax : a < x) (hxc : x < c) :
    x = b :=
  le_antisymm (le_of_not_lt fun h => hbc.2 h hxc) (le_of_not_lt <| hab.2 hax)


theorem covBy_iff_lt_iff_le_left {x y : α} : x ⋖ y ↔ ∀ {z}, z < y ↔ z ≤ x where
  mp := fun hx _z ↦ ⟨hx.le_of_lt, fun hz ↦ hz.trans_lt hx.lt⟩
  mpr := fun H ↦ ⟨H.2 le_rfl, fun _z hx hz ↦ (H.1 hz).not_lt hx⟩


theorem covBy_iff_le_iff_lt_left {x y : α} : x ⋖ y ↔ ∀ {z}, z ≤ x ↔ z < y := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    x y : α
    ⊢ Iff (CovBy x y) (∀ {z : α}, Iff (LE.le z x) (LT.lt z y))
  -/
  simp_rw [covBy_iff_lt_iff_le_left, iff_comm]
  /-
    🎉 no goals
  -/


theorem covBy_iff_lt_iff_le_right {x y : α} : x ⋖ y ↔ ∀ {z}, x < z ↔ y ≤ z := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    x y : α
    ⊢ Iff (CovBy x y) (∀ {z : α}, Iff (LT.lt x z) (LE.le y z))
  -/
  trans ∀ {z}, ¬ z ≤ x ↔ ¬ z < y
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      x y : α
      ⊢ Iff (CovBy x y) (∀ {z : α}, Iff (Not (LE.le z x)) (Not (LT.lt z y)))
    -/
  · simp_rw [covBy_iff_le_iff_lt_left, not_iff_not]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      x y : α
      ⊢ Iff (∀ {z : α}, Iff (Not (LE.le z x)) (Not (LT.lt z y))) (∀ {z : α}, Iff (LT …
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem covBy_iff_le_iff_lt_right {x y : α} : x ⋖ y ↔ ∀ {z}, y ≤ z ↔ x < z := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    x y : α
    ⊢ Iff (CovBy x y) (∀ {z : α}, Iff (LE.le y z) (LT.lt x z))
  -/
  simp_rw [covBy_iff_lt_iff_le_right, iff_comm]
  /-
    🎉 no goals
  -/


alias ⟨CovBy.lt_iff_le_left, _⟩ := covBy_iff_lt_iff_le_left

alias ⟨CovBy.le_iff_lt_left, _⟩ := covBy_iff_le_iff_lt_left

alias ⟨CovBy.lt_iff_le_right, _⟩ := covBy_iff_lt_iff_le_right

alias ⟨CovBy.le_iff_lt_right, _⟩ := covBy_iff_le_iff_lt_right


/-- If `a < b` then there exist `a' > a` and `b' < b` such that `Set.Iio a'` is strictly to the left
of `Set.Ioi b'`. -/
lemma LT.lt.exists_disjoint_Iio_Ioi (h : a < b) :
    ∃ a' > a, ∃ b' < b, ∀ x < a', ∀ y > b', x < y := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LT.lt a b
    ⊢ Exists fun a' => And (GT.gt a' a) (Exists fun b' => And (LT.lt b' b) (∀ (x : …
  -/
  by_cases h' : a ⋖ b
    /-
      case pos
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : LT.lt a b
      h' : CovBy a b
      ⊢ Exists fun a' => And (GT.gt a' a) (Exists fun b' => And (LT.lt b' b) (∀ (x : …
    -/
  · exact ⟨b, h, a, h, fun x hx y hy => hx.trans_le <| h'.ge_of_gt hy⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : LT.lt a b
      h' : Not (CovBy a b)
      ⊢ Exists fun a' => And (GT.gt a' a) (Exists fun b' => And (LT.lt b' b) (∀ (x : …
    -/
  · rcases h.exists_lt_lt h' with ⟨c, ha, hb⟩
    /-
      case neg.intro.intro
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : LT.lt a b
      h' : Not (CovBy a b)
      c : α
      ha : LT.lt a c
      hb : LT.lt c b
      ⊢ Exists fun a' => And (GT.gt a' a) (Exists fun b' => And (LT.lt b' b) (∀ (x : …
    -/
    exact ⟨c, ha, c, hb, fun _ h₁ _ => lt_trans h₁⟩
    /-
      🎉 no goals
    -/


@[simp] lemma wcovBy_insert (x : α) (s : Set α) : s ⩿ insert x s := by
  /-
    α : Type u_1
    x : α
    s : Set α
    ⊢ WCovBy s (Insert.insert x s)
  -/
  refine wcovBy_of_eq_or_eq (subset_insert x s) fun t hst h2t => ?_
  /-
    α : Type u_1
    x : α
    s t : Set α
    hst : LE.le s t
    h2t : LE.le t (Insert.insert x s)
    ⊢ Or (Eq t s) (Eq t (Insert.insert x s))
  -/
  by_cases h : x ∈ t
    /-
      case pos
      α : Type u_1
      x : α
      s t : Set α
      hst : LE.le s t
      h2t : LE.le t (Insert.insert x s)
      h : Membership.mem t x
      ⊢ Or (Eq t s) (Eq t (Insert.insert x s))
    -/
  · exact Or.inr (subset_antisymm h2t <| insert_subset_iff.mpr ⟨h, hst⟩)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      x : α
      s t : Set α
      hst : LE.le s t
      h2t : LE.le t (Insert.insert x s)
      h : Not (Membership.mem t x)
      ⊢ Or (Eq t s) (Eq t (Insert.insert x s))
    -/
  · refine Or.inl (subset_antisymm ?_ hst)
    /-
      case neg
      α : Type u_1
      x : α
      s t : Set α
      hst : LE.le s t
      h2t : LE.le t (Insert.insert x s)
      h : Not (Membership.mem t x)
      ⊢ HasSubset.Subset t s
    -/
    rwa [← diff_singleton_eq_self h, diff_singleton_subset_iff]
    /-
      🎉 no goals
    -/


@[simp] lemma sdiff_singleton_wcovBy (s : Set α) (a : α) : s \ {a} ⩿ s := by
  /-
    α : Type u_1
    s : Set α
    a : α
    ⊢ WCovBy (SDiff.sdiff s (Singleton.singleton a)) s
  -/
  by_cases ha : a ∈ s
    /-
      case pos
      α : Type u_1
      s : Set α
      a : α
      ha : Membership.mem s a
      ⊢ WCovBy (SDiff.sdiff s (Singleton.singleton a)) s
    -/
  · convert wcovBy_insert a _
    /-
      case h.e'_4
      α : Type u_1
      s : Set α
      a : α
      ha : Membership.mem s a
      ⊢ Eq s (Insert.insert a (SDiff.sdiff s (Singleton.singleton a)))
    -/
    ext
    /-
      case h.e'_4.h
      α : Type u_1
      s : Set α
      a : α
      ha : Membership.mem s a
      x✝ : α
      ⊢ Iff (Membership.mem s x✝) (Membership.mem (Insert.insert a (SDiff.sdiff s (S …
    -/
    simp [ha]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      s : Set α
      a : α
      ha : Not (Membership.mem s a)
      ⊢ WCovBy (SDiff.sdiff s (Singleton.singleton a)) s
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/


@[simp] lemma covBy_insert (ha : a ∉ s) : s ⋖ insert a s :=
  (wcovBy_insert _ _).covBy_of_lt <| ssubset_insert ha


@[simp] lemma sdiff_singleton_covBy (ha : a ∈ s) : s \ {a} ⋖ s :=
  ⟨sdiff_lt (singleton_subset_iff.2 ha) <| singleton_ne_empty _, (sdiff_singleton_wcovBy _ _).2⟩


lemma _root_.CovBy.exists_set_insert (h : s ⋖ t) : ∃ a ∉ s, insert a s = t :=
  let ⟨a, ha, hst⟩ := ssubset_iff_insert.1 h.lt
  ⟨a, ha, (hst.eq_of_not_ssuperset <| h.2 <| ssubset_insert ha).symm⟩


lemma _root_.CovBy.exists_set_sdiff_singleton (h : s ⋖ t) : ∃ a ∈ t, t \ {a} =  s :=
  let ⟨a, ha, hst⟩ := ssubset_iff_sdiff_singleton.1 h.lt
  ⟨a, ha, (hst.eq_of_not_ssubset fun h' ↦ h.2 h' <|
    sdiff_lt (singleton_subset_iff.2 ha) <| singleton_ne_empty _).symm⟩


lemma covBy_iff_exists_insert : s ⋖ t ↔ ∃ a ∉ s, insert a s = t :=
                               /-
                                 α : Type u_1
                                 s t : Set α
                                 ⊢ (Exists fun a => And (Not (Membership.mem s a)) (Eq (Insert.insert a s) t))  …
                               -/
  ⟨CovBy.exists_set_insert, by rintro ⟨a, ha, rfl⟩; exact covBy_insert ha⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma covBy_iff_exists_sdiff_singleton : s ⋖ t ↔ ∃ a ∈ t, t \ {a} = s :=
                                        /-
                                          α : Type u_1
                                          s t : Set α
                                          ⊢ (Exists fun a => And (Membership.mem t a) (Eq (SDiff.sdiff t (Singleton.sing …
                                        -/
  ⟨CovBy.exists_set_sdiff_singleton, by rintro ⟨a, ha, rfl⟩; exact sdiff_singleton_covBy ha⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma wcovBy_eq_reflGen_covBy [PartialOrder α] : ((· : α) ⩿ ·) = ReflGen (· ⋖ ·) := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    ⊢ Eq (fun x1 x2 => WCovBy x1 x2) (Relation.ReflGen fun x1 x2 => CovBy x1 x2)
  -/
  ext x y; simp_rw [wcovBy_iff_eq_or_covBy, @eq_comm _ x, reflGen_iff]
           /-
             🎉 no goals
           -/


lemma transGen_wcovBy_eq_reflTransGen_covBy [PartialOrder α] :
    TransGen ((· : α) ⩿ ·) = ReflTransGen (· ⋖ ·) := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    ⊢ Eq (Relation.TransGen fun x1 x2 => WCovBy x1 x2) (Relation.ReflTransGen fun  …
  -/
  rw [wcovBy_eq_reflGen_covBy, transGen_reflGen]
  /-
    🎉 no goals
  -/


lemma reflTransGen_wcovBy_eq_reflTransGen_covBy [PartialOrder α] :
    ReflTransGen ((· : α) ⩿ ·) = ReflTransGen (· ⋖ ·) := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    ⊢ Eq (Relation.ReflTransGen fun x1 x2 => WCovBy x1 x2) (Relation.ReflTransGen  …
  -/
  rw [wcovBy_eq_reflGen_covBy, reflTransGen_reflGen]
  /-
    🎉 no goals
  -/


@[simp]
theorem swap_wcovBy_swap : x.swap ⩿ y.swap ↔ x ⩿ y :=
  apply_wcovBy_apply_iff (OrderIso.prodComm : α × β ≃o β × α)


@[simp]
theorem swap_covBy_swap : x.swap ⋖ y.swap ↔ x ⋖ y :=
  apply_covBy_apply_iff (OrderIso.prodComm : α × β ≃o β × α)


theorem fst_eq_or_snd_eq_of_wcovBy : x ⩿ y → x.1 = y.1 ∨ x.2 = y.2 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    x y : Prod α β
    ⊢ WCovBy x y → Or (Eq x.1 y.1) (Eq x.2 y.2)
  -/
  refine fun h => of_not_not fun hab => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    x y : Prod α β
    h : WCovBy x y
    hab : Not (Or (Eq x.1 y.1) (Eq x.2 y.2))
    ⊢ False
  -/
  push_neg at hab
  exact
    h.2 (mk_lt_mk.2 <| Or.inl ⟨hab.1.lt_of_le h.1.1, le_rfl⟩)
      (mk_lt_mk.2 <| Or.inr ⟨le_rfl, hab.2.lt_of_le h.1.2⟩)


theorem _root_.WCovBy.fst (h : x ⩿ y) : x.1 ⩿ y.1 :=
  ⟨h.1.1, fun _ h₁ h₂ => h.2 (mk_lt_mk_iff_left.2 h₁) ⟨⟨h₂.le, h.1.2⟩, fun hc => h₂.not_le hc.1⟩⟩


theorem _root_.WCovBy.snd (h : x ⩿ y) : x.2 ⩿ y.2 :=
  ⟨h.1.2, fun _ h₁ h₂ => h.2 (mk_lt_mk_iff_right.2 h₁) ⟨⟨h.1.1, h₂.le⟩, fun hc => h₂.not_le hc.2⟩⟩


theorem mk_wcovBy_mk_iff_left : (a₁, b) ⩿ (a₂, b) ↔ a₁ ⩿ a₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    a₁ a₂ : α
    b : β
    ⊢ Iff (WCovBy { fst := a₁, snd := b } { fst := a₂, snd := b }) (WCovBy a₁ a₂)
  -/
  refine ⟨WCovBy.fst, (And.imp mk_le_mk_iff_left.2) fun h c h₁ h₂ => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    a₁ a₂ : α
    b : β
    h : ∀ ⦃c : α⦄, LT.lt a₁ c → Not (LT.lt c a₂)
    c : Prod α β
    h₁ : LT.lt { fst := a₁, snd := b } c
    h₂ : LT.lt c { fst := a₂, snd := b }
    ⊢ False
  -/
  have : c.2 = b := h₂.le.2.antisymm h₁.le.2
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    a₁ a₂ : α
    b : β
    h : ∀ ⦃c : α⦄, LT.lt a₁ c → Not (LT.lt c a₂)
    c : Prod α β
    h₁ : LT.lt { fst := a₁, snd := b } c
    h₂ : LT.lt c { fst := a₂, snd := b }
    this : Eq c.2 b
    ⊢ False
  -/
  rw [← @Prod.mk.eta _ _ c, this, mk_lt_mk_iff_left] at h₁ h₂
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    a₁ a₂ : α
    b : β
    h : ∀ ⦃c : α⦄, LT.lt a₁ c → Not (LT.lt c a₂)
    c : Prod α β
    h₁ : LT.lt a₁ c.1
    h₂ : LT.lt c.1 a₂
    this : Eq c.2 b
    ⊢ False
  -/
  exact h h₁ h₂
  /-
    🎉 no goals
  -/


theorem mk_wcovBy_mk_iff_right : (a, b₁) ⩿ (a, b₂) ↔ b₁ ⩿ b₂ :=
  swap_wcovBy_swap.trans mk_wcovBy_mk_iff_left


theorem mk_covBy_mk_iff_left : (a₁, b) ⋖ (a₂, b) ↔ a₁ ⋖ a₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    a₁ a₂ : α
    b : β
    ⊢ Iff (CovBy { fst := a₁, snd := b } { fst := a₂, snd := b }) (CovBy a₁ a₂)
  -/
  simp_rw [covBy_iff_wcovBy_and_lt, mk_wcovBy_mk_iff_left, mk_lt_mk_iff_left]
  /-
    🎉 no goals
  -/


theorem mk_covBy_mk_iff_right : (a, b₁) ⋖ (a, b₂) ↔ b₁ ⋖ b₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    a : α
    b₁ b₂ : β
    ⊢ Iff (CovBy { fst := a, snd := b₁ } { fst := a, snd := b₂ }) (CovBy b₁ b₂)
  -/
  simp_rw [covBy_iff_wcovBy_and_lt, mk_wcovBy_mk_iff_right, mk_lt_mk_iff_right]
  /-
    🎉 no goals
  -/


theorem mk_wcovBy_mk_iff : (a₁, b₁) ⩿ (a₂, b₂) ↔ a₁ ⩿ a₂ ∧ b₁ = b₂ ∨ b₁ ⩿ b₂ ∧ a₁ = a₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    a₁ a₂ : α
    b₁ b₂ : β
    ⊢ Iff (WCovBy { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }) (Or (And (WCo …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      a₁ a₂ : α
      b₁ b₂ : β
      h : WCovBy { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
      ⊢ Or (And (WCovBy a₁ a₂) (Eq b₁ b₂)) (And (WCovBy b₁ b₂) (Eq a₁ a₂))
    -/
  · obtain rfl | rfl : a₁ = a₂ ∨ b₁ = b₂ := fst_eq_or_snd_eq_of_wcovBy h
      /-
        case refine_1.inl
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        a₁ : α
        b₁ b₂ : β
        h : WCovBy { fst := a₁, snd := b₁ } { fst := a₁, snd := b₂ }
        ⊢ Or (And (WCovBy a₁ a₁) (Eq b₁ b₂)) (And (WCovBy b₁ b₂) (Eq a₁ a₁))
      -/
    · exact Or.inr ⟨mk_wcovBy_mk_iff_right.1 h, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        a₁ a₂ : α
        b₁ : β
        h : WCovBy { fst := a₁, snd := b₁ } { fst := a₂, snd := b₁ }
        ⊢ Or (And (WCovBy a₁ a₂) (Eq b₁ b₁)) (And (WCovBy b₁ b₁) (Eq a₁ a₂))
      -/
    · exact Or.inl ⟨mk_wcovBy_mk_iff_left.1 h, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      a₁ a₂ : α
      b₁ b₂ : β
      ⊢ Or (And (WCovBy a₁ a₂) (Eq b₁ b₂)) (And (WCovBy b₁ b₂) (Eq a₁ a₂)) → WCovBy  …
    -/
  · rintro (⟨h, rfl⟩ | ⟨h, rfl⟩)
      /-
        case refine_2.inl.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        a₁ a₂ : α
        b₁ : β
        h : WCovBy a₁ a₂
        ⊢ WCovBy { fst := a₁, snd := b₁ } { fst := a₂, snd := b₁ }
      -/
    · exact mk_wcovBy_mk_iff_left.2 h
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        a₁ : α
        b₁ b₂ : β
        h : WCovBy b₁ b₂
        ⊢ WCovBy { fst := a₁, snd := b₁ } { fst := a₁, snd := b₂ }
      -/
    · exact mk_wcovBy_mk_iff_right.2 h
      /-
        🎉 no goals
      -/


theorem mk_covBy_mk_iff : (a₁, b₁) ⋖ (a₂, b₂) ↔ a₁ ⋖ a₂ ∧ b₁ = b₂ ∨ b₁ ⋖ b₂ ∧ a₁ = a₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    a₁ a₂ : α
    b₁ b₂ : β
    ⊢ Iff (CovBy { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }) (Or (And (CovB …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      a₁ a₂ : α
      b₁ b₂ : β
      h : CovBy { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
      ⊢ Or (And (CovBy a₁ a₂) (Eq b₁ b₂)) (And (CovBy b₁ b₂) (Eq a₁ a₂))
    -/
  · obtain rfl | rfl : a₁ = a₂ ∨ b₁ = b₂ := fst_eq_or_snd_eq_of_wcovBy h.wcovBy
      /-
        case refine_1.inl
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        a₁ : α
        b₁ b₂ : β
        h : CovBy { fst := a₁, snd := b₁ } { fst := a₁, snd := b₂ }
        ⊢ Or (And (CovBy a₁ a₁) (Eq b₁ b₂)) (And (CovBy b₁ b₂) (Eq a₁ a₁))
      -/
    · exact Or.inr ⟨mk_covBy_mk_iff_right.1 h, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        a₁ a₂ : α
        b₁ : β
        h : CovBy { fst := a₁, snd := b₁ } { fst := a₂, snd := b₁ }
        ⊢ Or (And (CovBy a₁ a₂) (Eq b₁ b₁)) (And (CovBy b₁ b₁) (Eq a₁ a₂))
      -/
    · exact Or.inl ⟨mk_covBy_mk_iff_left.1 h, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      a₁ a₂ : α
      b₁ b₂ : β
      ⊢ Or (And (CovBy a₁ a₂) (Eq b₁ b₂)) (And (CovBy b₁ b₂) (Eq a₁ a₂)) → CovBy { f …
    -/
  · rintro (⟨h, rfl⟩ | ⟨h, rfl⟩)
      /-
        case refine_2.inl.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        a₁ a₂ : α
        b₁ : β
        h : CovBy a₁ a₂
        ⊢ CovBy { fst := a₁, snd := b₁ } { fst := a₂, snd := b₁ }
      -/
    · exact mk_covBy_mk_iff_left.2 h
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        a₁ : α
        b₁ b₂ : β
        h : CovBy b₁ b₂
        ⊢ CovBy { fst := a₁, snd := b₁ } { fst := a₁, snd := b₂ }
      -/
    · exact mk_covBy_mk_iff_right.2 h
      /-
        🎉 no goals
      -/


theorem wcovBy_iff : x ⩿ y ↔ x.1 ⩿ y.1 ∧ x.2 = y.2 ∨ x.2 ⩿ y.2 ∧ x.1 = y.1 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    x y : Prod α β
    ⊢ Iff (WCovBy x y) (Or (And (WCovBy x.1 y.1) (Eq x.2 y.2)) (And (WCovBy x.2 y. …
  -/
  cases x
  /-
    case mk
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    y : Prod α β
    fst✝ : α
    snd✝ : β
    ⊢ Iff (WCovBy { fst := fst✝, snd := snd✝ } y) (Or (And (WCovBy { fst := fst✝,  …
  -/
  cases y
  /-
    case mk.mk
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    fst✝¹ : α
    snd✝¹ : β
    fst✝ : α
    snd✝ : β
    ⊢ Iff (WCovBy { fst := fst✝¹, snd := snd✝¹ } { fst := fst✝, snd := snd✝ }) (Or …
  -/
  exact mk_wcovBy_mk_iff
  /-
    🎉 no goals
  -/


theorem covBy_iff : x ⋖ y ↔ x.1 ⋖ y.1 ∧ x.2 = y.2 ∨ x.2 ⋖ y.2 ∧ x.1 = y.1 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    x y : Prod α β
    ⊢ Iff (CovBy x y) (Or (And (CovBy x.1 y.1) (Eq x.2 y.2)) (And (CovBy x.2 y.2)  …
  -/
  cases x
  /-
    case mk
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    y : Prod α β
    fst✝ : α
    snd✝ : β
    ⊢ Iff (CovBy { fst := fst✝, snd := snd✝ } y) (Or (And (CovBy { fst := fst✝, sn …
  -/
  cases y
  /-
    case mk.mk
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    fst✝¹ : α
    snd✝¹ : β
    fst✝ : α
    snd✝ : β
    ⊢ Iff (CovBy { fst := fst✝¹, snd := snd✝¹ } { fst := fst✝, snd := snd✝ }) (Or  …
  -/
  exact mk_covBy_mk_iff
  /-
    🎉 no goals
  -/


@[simp, norm_cast] lemma coe_wcovBy_coe : (a : WithTop α) ⩿ b ↔ a ⩿ b :=
  Set.OrdConnected.apply_wcovBy_apply_iff OrderEmbedding.withTopCoe <| by
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b : α
      ⊢ (Set.range ⇑OrderEmbedding.withTopCoe).OrdConnected
    -/
    simp [WithTop.range_coe, ordConnected_Iio]
    /-
      🎉 no goals
    -/


@[simp, norm_cast] lemma coe_covBy_coe : (a : WithTop α) ⋖ b ↔ a ⋖ b :=
  Set.OrdConnected.apply_covBy_apply_iff OrderEmbedding.withTopCoe <| by
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b : α
      ⊢ (Set.range ⇑OrderEmbedding.withTopCoe).OrdConnected
    -/
    simp [WithTop.range_coe, ordConnected_Iio]
    /-
      🎉 no goals
    -/


@[simp] lemma coe_covBy_top : (a : WithTop α) ⋖ ⊤ ↔ IsMax a := by
  simp only [covBy_iff_Ioo_eq, ← image_coe_Ioi, coe_lt_top, image_eq_empty,
    true_and, Ioi_eq_empty_iff]


@[simp] lemma coe_wcovBy_top : (a : WithTop α) ⩿ ⊤ ↔ IsMax a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ Iff (WCovBy (↑a) Top.top) (IsMax a)
  -/
  simp only [wcovBy_iff_Ioo_eq, ← image_coe_Ioi, le_top, image_eq_empty, true_and, Ioi_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp, norm_cast] lemma coe_wcovBy_coe : (a : WithBot α) ⩿ b ↔ a ⩿ b :=
  Set.OrdConnected.apply_wcovBy_apply_iff OrderEmbedding.withBotCoe <| by
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b : α
      ⊢ (Set.range ⇑OrderEmbedding.withBotCoe).OrdConnected
    -/
    simp [WithBot.range_coe, ordConnected_Ioi]
    /-
      🎉 no goals
    -/


@[simp, norm_cast] lemma coe_covBy_coe : (a : WithBot α) ⋖ b ↔ a ⋖ b :=
  Set.OrdConnected.apply_covBy_apply_iff OrderEmbedding.withBotCoe <| by
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b : α
      ⊢ (Set.range ⇑OrderEmbedding.withBotCoe).OrdConnected
    -/
    simp [WithBot.range_coe, ordConnected_Ioi]
    /-
      🎉 no goals
    -/


@[simp] lemma bot_covBy_coe : ⊥ ⋖ (a : WithBot α) ↔ IsMin a := by
  simp only [covBy_iff_Ioo_eq, ← image_coe_Iio, bot_lt_coe, image_eq_empty,
    true_and, Iio_eq_empty_iff]


@[simp] lemma bot_wcovBy_coe : ⊥ ⩿ (a : WithBot α) ↔ IsMin a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ Iff (WCovBy Bot.bot ↑a) (IsMin a)
  -/
  simp only [wcovBy_iff_Ioo_eq, ← image_coe_Iio, bot_le, image_eq_empty, true_and, Iio_eq_empty_iff]
  /-
    🎉 no goals
  -/


lemma exists_covBy_of_wellFoundedLT [wf : WellFoundedLT α] ⦃a : α⦄ (h : ¬ IsMax a) :
    ∃ a', a ⋖ a' := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    wf : WellFoundedLT α
    a : α
    h : Not (IsMax a)
    ⊢ Exists fun a' => CovBy a a'
  -/
  rw [not_isMax_iff] at h
  /-
    α : Type u_1
    inst✝ : Preorder α
    wf : WellFoundedLT α
    a : α
    h : Exists fun b => LT.lt a b
    ⊢ Exists fun a' => CovBy a a'
  -/
  exact ⟨_, wellFounded_lt.min_mem _ h, fun a' ↦ wf.wf.not_lt_min _ h⟩
  /-
    🎉 no goals
  -/


lemma exists_covBy_of_wellFoundedGT [wf : WellFoundedGT α] ⦃a : α⦄ (h : ¬ IsMin a) :
    ∃ a', a' ⋖ a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    wf : WellFoundedGT α
    a : α
    h : Not (IsMin a)
    ⊢ Exists fun a' => CovBy a' a
  -/
  rw [not_isMin_iff] at h
  /-
    α : Type u_1
    inst✝ : Preorder α
    wf : WellFoundedGT α
    a : α
    h : Exists fun b => LT.lt b a
    ⊢ Exists fun a' => CovBy a' a
  -/
  exact ⟨_, wf.wf.min_mem _ h, fun a' h₁ h₂ ↦ wf.wf.not_lt_min _ h h₂ h₁⟩
  /-
    🎉 no goals
  -/


