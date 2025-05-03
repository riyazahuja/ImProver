theorem Bounded.mono (hst : s ⊆ t) (hs : Bounded r t) : Bounded r s :=
  hs.imp fun _ ha b hb => ha b (hst hb)


theorem Unbounded.mono (hst : s ⊆ t) (hs : Unbounded r s) : Unbounded r t := fun a =>
  let ⟨b, hb, hb'⟩ := hs a
  ⟨b, hst hb, hb'⟩


theorem unbounded_le_of_forall_exists_lt [Preorder α] (h : ∀ a, ∃ b ∈ s, a < b) :
    Unbounded (· ≤ ·) s := fun a =>
  let ⟨b, hb, hb'⟩ := h a
  ⟨b, hb, fun hba => hba.not_lt hb'⟩


theorem unbounded_le_iff [LinearOrder α] : Unbounded (· ≤ ·) s ↔ ∀ a, ∃ b ∈ s, a < b := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    ⊢ Iff (Set.Unbounded (fun x1 x2 => LE.le x1 x2) s) (∀ (a : α), Exists fun b => …
  -/
  simp only [Unbounded, not_le]
  /-
    🎉 no goals
  -/


theorem unbounded_lt_of_forall_exists_le [Preorder α] (h : ∀ a, ∃ b ∈ s, a ≤ b) :
    Unbounded (· < ·) s := fun a =>
  let ⟨b, hb, hb'⟩ := h a
  ⟨b, hb, fun hba => hba.not_le hb'⟩


theorem unbounded_lt_iff [LinearOrder α] : Unbounded (· < ·) s ↔ ∀ a, ∃ b ∈ s, a ≤ b := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    ⊢ Iff (Set.Unbounded (fun x1 x2 => LT.lt x1 x2) s) (∀ (a : α), Exists fun b => …
  -/
  simp only [Unbounded, not_lt]
  /-
    🎉 no goals
  -/


theorem unbounded_ge_of_forall_exists_gt [Preorder α] (h : ∀ a, ∃ b ∈ s, b < a) :
    Unbounded (· ≥ ·) s :=
  @unbounded_le_of_forall_exists_lt αᵒᵈ _ _ h


theorem unbounded_ge_iff [LinearOrder α] : Unbounded (· ≥ ·) s ↔ ∀ a, ∃ b ∈ s, b < a :=
  ⟨fun h a =>
    let ⟨b, hb, hba⟩ := h a
    ⟨b, hb, lt_of_not_ge hba⟩,
    unbounded_ge_of_forall_exists_gt⟩


theorem unbounded_gt_of_forall_exists_ge [Preorder α] (h : ∀ a, ∃ b ∈ s, b ≤ a) :
    Unbounded (· > ·) s := fun a =>
  let ⟨b, hb, hb'⟩ := h a
  ⟨b, hb, fun hba => not_le_of_gt hba hb'⟩


theorem unbounded_gt_iff [LinearOrder α] : Unbounded (· > ·) s ↔ ∀ a, ∃ b ∈ s, b ≤ a :=
  ⟨fun h a =>
    let ⟨b, hb, hba⟩ := h a
    ⟨b, hb, le_of_not_gt hba⟩,
    unbounded_gt_of_forall_exists_ge⟩


theorem Bounded.rel_mono {r' : α → α → Prop} (h : Bounded r s) (hrr' : r ≤ r') : Bounded r' s :=
  let ⟨a, ha⟩ := h
  ⟨a, fun b hb => hrr' b a (ha b hb)⟩


theorem bounded_le_of_bounded_lt [Preorder α] (h : Bounded (· < ·) s) : Bounded (· ≤ ·) s :=
  h.rel_mono fun _ _ => le_of_lt


theorem Unbounded.rel_mono {r' : α → α → Prop} (hr : r' ≤ r) (h : Unbounded r s) : Unbounded r' s :=
  fun a =>
  let ⟨b, hb, hba⟩ := h a
  ⟨b, hb, fun hba' => hba (hr b a hba')⟩


theorem unbounded_lt_of_unbounded_le [Preorder α] (h : Unbounded (· ≤ ·) s) : Unbounded (· < ·) s :=
  h.rel_mono fun _ _ => le_of_lt


theorem bounded_le_iff_bounded_lt [Preorder α] [NoMaxOrder α] :
    Bounded (· ≤ ·) s ↔ Bounded (· < ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    ⊢ Iff (Set.Bounded (fun x1 x2 => LE.le x1 x2) s) (Set.Bounded (fun x1 x2 => LT …
  -/
  refine ⟨fun h => ?_, bounded_le_of_bounded_lt⟩
  /-
    α : Type u_1
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    h : Set.Bounded (fun x1 x2 => LE.le x1 x2) s
    ⊢ Set.Bounded (fun x1 x2 => LT.lt x1 x2) s
  -/
  obtain ⟨a, ha⟩ := h
  /-
    case intro
    α : Type u_1
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    ha : ∀ (b : α), Membership.mem s b → (fun x1 x2 => LE.le x1 x2) b a
    ⊢ Set.Bounded (fun x1 x2 => LT.lt x1 x2) s
  -/
  obtain ⟨b, hb⟩ := exists_gt a
  /-
    case intro.intro
    α : Type u_1
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    ha : ∀ (b : α), Membership.mem s b → (fun x1 x2 => LE.le x1 x2) b a
    b : α
    hb : LT.lt a b
    ⊢ Set.Bounded (fun x1 x2 => LT.lt x1 x2) s
  -/
  exact ⟨b, fun c hc => lt_of_le_of_lt (ha c hc) hb⟩
  /-
    🎉 no goals
  -/


theorem unbounded_lt_iff_unbounded_le [Preorder α] [NoMaxOrder α] :
    Unbounded (· < ·) s ↔ Unbounded (· ≤ ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    ⊢ Iff (Set.Unbounded (fun x1 x2 => LT.lt x1 x2) s) (Set.Unbounded (fun x1 x2 = …
  -/
  simp_rw [← not_bounded_iff, bounded_le_iff_bounded_lt]
  /-
    🎉 no goals
  -/


theorem bounded_ge_of_bounded_gt [Preorder α] (h : Bounded (· > ·) s) : Bounded (· ≥ ·) s :=
  let ⟨a, ha⟩ := h
  ⟨a, fun b hb => le_of_lt (ha b hb)⟩


theorem unbounded_gt_of_unbounded_ge [Preorder α] (h : Unbounded (· ≥ ·) s) : Unbounded (· > ·) s :=
  fun a =>
  let ⟨b, hb, hba⟩ := h a
  ⟨b, hb, fun hba' => hba (le_of_lt hba')⟩


theorem bounded_ge_iff_bounded_gt [Preorder α] [NoMinOrder α] :
    Bounded (· ≥ ·) s ↔ Bounded (· > ·) s :=
  @bounded_le_iff_bounded_lt αᵒᵈ _ _ _


theorem unbounded_gt_iff_unbounded_ge [Preorder α] [NoMinOrder α] :
    Unbounded (· > ·) s ↔ Unbounded (· ≥ ·) s :=
  @unbounded_lt_iff_unbounded_le αᵒᵈ _ _ _


theorem unbounded_le_univ [LE α] [NoTopOrder α] : Unbounded (· ≤ ·) (@Set.univ α) := fun a =>
  let ⟨b, hb⟩ := exists_not_le a
  ⟨b, ⟨⟩, hb⟩


theorem unbounded_lt_univ [Preorder α] [NoTopOrder α] : Unbounded (· < ·) (@Set.univ α) :=
  unbounded_lt_of_unbounded_le unbounded_le_univ


theorem unbounded_ge_univ [LE α] [NoBotOrder α] : Unbounded (· ≥ ·) (@Set.univ α) := fun a =>
  let ⟨b, hb⟩ := exists_not_ge a
  ⟨b, ⟨⟩, hb⟩


theorem unbounded_gt_univ [Preorder α] [NoBotOrder α] : Unbounded (· > ·) (@Set.univ α) :=
  unbounded_gt_of_unbounded_ge unbounded_ge_univ


theorem bounded_self (a : α) : Bounded r { b | r b a } :=
  ⟨a, fun _ => id⟩


theorem bounded_lt_Iio [Preorder α] (a : α) : Bounded (· < ·) (Iio a) :=
  bounded_self a


theorem bounded_le_Iio [Preorder α] (a : α) : Bounded (· ≤ ·) (Iio a) :=
  bounded_le_of_bounded_lt (bounded_lt_Iio a)


theorem bounded_le_Iic [Preorder α] (a : α) : Bounded (· ≤ ·) (Iic a) :=
  bounded_self a


theorem bounded_lt_Iic [Preorder α] [NoMaxOrder α] (a : α) : Bounded (· < ·) (Iic a) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ Set.Bounded (fun x1 x2 => LT.lt x1 x2) (Set.Iic a)
  -/
  simp only [← bounded_le_iff_bounded_lt, bounded_le_Iic]
  /-
    🎉 no goals
  -/


theorem bounded_gt_Ioi [Preorder α] (a : α) : Bounded (· > ·) (Ioi a) :=
  bounded_self a


theorem bounded_ge_Ioi [Preorder α] (a : α) : Bounded (· ≥ ·) (Ioi a) :=
  bounded_ge_of_bounded_gt (bounded_gt_Ioi a)


theorem bounded_ge_Ici [Preorder α] (a : α) : Bounded (· ≥ ·) (Ici a) :=
  bounded_self a


theorem bounded_gt_Ici [Preorder α] [NoMinOrder α] (a : α) : Bounded (· > ·) (Ici a) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : NoMinOrder α
    a : α
    ⊢ Set.Bounded (fun x1 x2 => GT.gt x1 x2) (Set.Ici a)
  -/
  simp only [← bounded_ge_iff_bounded_gt, bounded_ge_Ici]
  /-
    🎉 no goals
  -/


theorem bounded_lt_Ioo [Preorder α] (a b : α) : Bounded (· < ·) (Ioo a b) :=
  (bounded_lt_Iio b).mono Set.Ioo_subset_Iio_self


theorem bounded_lt_Ico [Preorder α] (a b : α) : Bounded (· < ·) (Ico a b) :=
  (bounded_lt_Iio b).mono Set.Ico_subset_Iio_self


theorem bounded_lt_Ioc [Preorder α] [NoMaxOrder α] (a b : α) : Bounded (· < ·) (Ioc a b) :=
  (bounded_lt_Iic b).mono Set.Ioc_subset_Iic_self


theorem bounded_lt_Icc [Preorder α] [NoMaxOrder α] (a b : α) : Bounded (· < ·) (Icc a b) :=
  (bounded_lt_Iic b).mono Set.Icc_subset_Iic_self


theorem bounded_le_Ioo [Preorder α] (a b : α) : Bounded (· ≤ ·) (Ioo a b) :=
  (bounded_le_Iio b).mono Set.Ioo_subset_Iio_self


theorem bounded_le_Ico [Preorder α] (a b : α) : Bounded (· ≤ ·) (Ico a b) :=
  (bounded_le_Iio b).mono Set.Ico_subset_Iio_self


theorem bounded_le_Ioc [Preorder α] (a b : α) : Bounded (· ≤ ·) (Ioc a b) :=
  (bounded_le_Iic b).mono Set.Ioc_subset_Iic_self


theorem bounded_le_Icc [Preorder α] (a b : α) : Bounded (· ≤ ·) (Icc a b) :=
  (bounded_le_Iic b).mono Set.Icc_subset_Iic_self


theorem bounded_gt_Ioo [Preorder α] (a b : α) : Bounded (· > ·) (Ioo a b) :=
  (bounded_gt_Ioi a).mono Set.Ioo_subset_Ioi_self


theorem bounded_gt_Ioc [Preorder α] (a b : α) : Bounded (· > ·) (Ioc a b) :=
  (bounded_gt_Ioi a).mono Set.Ioc_subset_Ioi_self


theorem bounded_gt_Ico [Preorder α] [NoMinOrder α] (a b : α) : Bounded (· > ·) (Ico a b) :=
  (bounded_gt_Ici a).mono Set.Ico_subset_Ici_self


theorem bounded_gt_Icc [Preorder α] [NoMinOrder α] (a b : α) : Bounded (· > ·) (Icc a b) :=
  (bounded_gt_Ici a).mono Set.Icc_subset_Ici_self


theorem bounded_ge_Ioo [Preorder α] (a b : α) : Bounded (· ≥ ·) (Ioo a b) :=
  (bounded_ge_Ioi a).mono Set.Ioo_subset_Ioi_self


theorem bounded_ge_Ioc [Preorder α] (a b : α) : Bounded (· ≥ ·) (Ioc a b) :=
  (bounded_ge_Ioi a).mono Set.Ioc_subset_Ioi_self


theorem bounded_ge_Ico [Preorder α] (a b : α) : Bounded (· ≥ ·) (Ico a b) :=
  (bounded_ge_Ici a).mono Set.Ico_subset_Ici_self


theorem bounded_ge_Icc [Preorder α] (a b : α) : Bounded (· ≥ ·) (Icc a b) :=
  (bounded_ge_Ici a).mono Set.Icc_subset_Ici_self


theorem unbounded_le_Ioi [SemilatticeSup α] [NoMaxOrder α] (a : α) :
    Unbounded (· ≤ ·) (Ioi a) := fun b =>
  let ⟨c, hc⟩ := exists_gt (a ⊔ b)
  ⟨c, le_sup_left.trans_lt hc, (le_sup_right.trans_lt hc).not_le⟩


theorem unbounded_le_Ici [SemilatticeSup α] [NoMaxOrder α] (a : α) :
    Unbounded (· ≤ ·) (Ici a) :=
  (unbounded_le_Ioi a).mono Set.Ioi_subset_Ici_self


theorem unbounded_lt_Ioi [SemilatticeSup α] [NoMaxOrder α] (a : α) :
    Unbounded (· < ·) (Ioi a) :=
  unbounded_lt_of_unbounded_le (unbounded_le_Ioi a)


theorem unbounded_lt_Ici [SemilatticeSup α] (a : α) : Unbounded (· < ·) (Ici a) := fun b =>
  ⟨a ⊔ b, le_sup_left, le_sup_right.not_lt⟩


theorem bounded_inter_not (H : ∀ a b, ∃ m, ∀ c, r c a ∨ r c b → r c m) (a : α) :
    Bounded r (s ∩ { b | ¬r b a }) ↔ Bounded r s := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Set α
    H : ∀ (a b : α), Exists fun m => ∀ (c : α), Or (r c a) (r c b) → r c m
    a : α
    ⊢ Iff (Set.Bounded r (Inter.inter s (setOf fun b => Not (r b a)))) (Set.Bounde …
  -/
  refine ⟨?_, Bounded.mono inter_subset_left⟩
  /-
    α : Type u_1
    r : α → α → Prop
    s : Set α
    H : ∀ (a b : α), Exists fun m => ∀ (c : α), Or (r c a) (r c b) → r c m
    a : α
    ⊢ Set.Bounded r (Inter.inter s (setOf fun b => Not (r b a))) → Set.Bounded r s
  -/
  rintro ⟨b, hb⟩
  /-
    case intro
    α : Type u_1
    r : α → α → Prop
    s : Set α
    H : ∀ (a b : α), Exists fun m => ∀ (c : α), Or (r c a) (r c b) → r c m
    a b : α
    hb : ∀ (b_1 : α), Membership.mem (Inter.inter s (setOf fun b => Not (r b a)))  …
    ⊢ Set.Bounded r s
  -/
  obtain ⟨m, hm⟩ := H a b
  /-
    case intro.intro
    α : Type u_1
    r : α → α → Prop
    s : Set α
    H : ∀ (a b : α), Exists fun m => ∀ (c : α), Or (r c a) (r c b) → r c m
    a b : α
    hb : ∀ (b_1 : α), Membership.mem (Inter.inter s (setOf fun b => Not (r b a)))  …
    m : α
    hm : ∀ (c : α), Or (r c a) (r c b) → r c m
    ⊢ Set.Bounded r s
  -/
  exact ⟨m, fun c hc => hm c (or_iff_not_imp_left.2 fun hca => hb c ⟨hc, hca⟩)⟩
  /-
    🎉 no goals
  -/


theorem unbounded_inter_not (H : ∀ a b, ∃ m, ∀ c, r c a ∨ r c b → r c m) (a : α) :
    Unbounded r (s ∩ { b | ¬r b a }) ↔ Unbounded r s := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Set α
    H : ∀ (a b : α), Exists fun m => ∀ (c : α), Or (r c a) (r c b) → r c m
    a : α
    ⊢ Iff (Set.Unbounded r (Inter.inter s (setOf fun b => Not (r b a)))) (Set.Unbo …
  -/
  simp_rw [← not_bounded_iff, bounded_inter_not H]
  /-
    🎉 no goals
  -/


theorem bounded_le_inter_not_le [SemilatticeSup α] (a : α) :
    Bounded (· ≤ ·) (s ∩ { b | ¬b ≤ a }) ↔ Bounded (· ≤ ·) s :=
  bounded_inter_not (fun x y => ⟨x ⊔ y, fun _ h => h.elim le_sup_of_le_left le_sup_of_le_right⟩) a


theorem unbounded_le_inter_not_le [SemilatticeSup α] (a : α) :
    Unbounded (· ≤ ·) (s ∩ { b | ¬b ≤ a }) ↔ Unbounded (· ≤ ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : SemilatticeSup α
    a : α
    ⊢ Iff (Set.Unbounded (fun x1 x2 => LE.le x1 x2) (Inter.inter s (setOf fun b => …
  -/
  rw [← not_bounded_iff, ← not_bounded_iff, not_iff_not]
  /-
    α : Type u_1
    s : Set α
    inst✝ : SemilatticeSup α
    a : α
    ⊢ Iff (Set.Bounded (fun x1 x2 => LE.le x1 x2) (Inter.inter s (setOf fun b => N …
  -/
  exact bounded_le_inter_not_le a
  /-
    🎉 no goals
  -/


theorem bounded_le_inter_lt [LinearOrder α] (a : α) :
    Bounded (· ≤ ·) (s ∩ { b | a < b }) ↔ Bounded (· ≤ ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a : α
    ⊢ Iff (Set.Bounded (fun x1 x2 => LE.le x1 x2) (Inter.inter s (setOf fun b => L …
  -/
  simp_rw [← not_le, bounded_le_inter_not_le]
  /-
    🎉 no goals
  -/


theorem unbounded_le_inter_lt [LinearOrder α] (a : α) :
    Unbounded (· ≤ ·) (s ∩ { b | a < b }) ↔ Unbounded (· ≤ ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a : α
    ⊢ Iff (Set.Unbounded (fun x1 x2 => LE.le x1 x2) (Inter.inter s (setOf fun b => …
  -/
  convert @unbounded_le_inter_not_le _ s _ a
  /-
    case h.e'_1.h.e'_3.h.e'_4.h.e'_2.h.a
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a x✝ : α
    ⊢ Iff (LT.lt a x✝) (Not (LE.le x✝ a))
  -/
  exact lt_iff_not_le
  /-
    🎉 no goals
  -/


theorem bounded_le_inter_le [LinearOrder α] (a : α) :
    Bounded (· ≤ ·) (s ∩ { b | a ≤ b }) ↔ Bounded (· ≤ ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a : α
    ⊢ Iff (Set.Bounded (fun x1 x2 => LE.le x1 x2) (Inter.inter s (setOf fun b => L …
  -/
  refine ⟨?_, Bounded.mono Set.inter_subset_left⟩
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a : α
    ⊢ Set.Bounded (fun x1 x2 => LE.le x1 x2) (Inter.inter s (setOf fun b => LE.le  …
  -/
  rw [← @bounded_le_inter_lt _ s _ a]
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a : α
    ⊢ Set.Bounded (fun x1 x2 => LE.le x1 x2) (Inter.inter s (setOf fun b => LE.le  …
  -/
  exact Bounded.mono fun x ⟨hx, hx'⟩ => ⟨hx, le_of_lt hx'⟩
  /-
    🎉 no goals
  -/


theorem unbounded_le_inter_le [LinearOrder α] (a : α) :
    Unbounded (· ≤ ·) (s ∩ { b | a ≤ b }) ↔ Unbounded (· ≤ ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a : α
    ⊢ Iff (Set.Unbounded (fun x1 x2 => LE.le x1 x2) (Inter.inter s (setOf fun b => …
  -/
  rw [← not_bounded_iff, ← not_bounded_iff, not_iff_not]
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a : α
    ⊢ Iff (Set.Bounded (fun x1 x2 => LE.le x1 x2) (Inter.inter s (setOf fun b => L …
  -/
  exact bounded_le_inter_le a
  /-
    🎉 no goals
  -/


theorem bounded_lt_inter_not_lt [SemilatticeSup α] (a : α) :
    Bounded (· < ·) (s ∩ { b | ¬b < a }) ↔ Bounded (· < ·) s :=
  bounded_inter_not (fun x y => ⟨x ⊔ y, fun _ h => h.elim lt_sup_of_lt_left lt_sup_of_lt_right⟩) a


theorem unbounded_lt_inter_not_lt [SemilatticeSup α] (a : α) :
    Unbounded (· < ·) (s ∩ { b | ¬b < a }) ↔ Unbounded (· < ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : SemilatticeSup α
    a : α
    ⊢ Iff (Set.Unbounded (fun x1 x2 => LT.lt x1 x2) (Inter.inter s (setOf fun b => …
  -/
  rw [← not_bounded_iff, ← not_bounded_iff, not_iff_not]
  /-
    α : Type u_1
    s : Set α
    inst✝ : SemilatticeSup α
    a : α
    ⊢ Iff (Set.Bounded (fun x1 x2 => LT.lt x1 x2) (Inter.inter s (setOf fun b => N …
  -/
  exact bounded_lt_inter_not_lt a
  /-
    🎉 no goals
  -/


theorem bounded_lt_inter_le [LinearOrder α] (a : α) :
    Bounded (· < ·) (s ∩ { b | a ≤ b }) ↔ Bounded (· < ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a : α
    ⊢ Iff (Set.Bounded (fun x1 x2 => LT.lt x1 x2) (Inter.inter s (setOf fun b => L …
  -/
  convert @bounded_lt_inter_not_lt _ s _ a
  /-
    case h.e'_1.h.e'_3.h.e'_4.h.e'_2.h.a
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a x✝ : α
    ⊢ Iff (LE.le a x✝) (Not (LT.lt x✝ a))
  -/
  exact not_lt.symm
  /-
    🎉 no goals
  -/


theorem unbounded_lt_inter_le [LinearOrder α] (a : α) :
    Unbounded (· < ·) (s ∩ { b | a ≤ b }) ↔ Unbounded (· < ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a : α
    ⊢ Iff (Set.Unbounded (fun x1 x2 => LT.lt x1 x2) (Inter.inter s (setOf fun b => …
  -/
  convert @unbounded_lt_inter_not_lt _ s _ a
  /-
    case h.e'_1.h.e'_3.h.e'_4.h.e'_2.h.a
    α : Type u_1
    s : Set α
    inst✝ : LinearOrder α
    a x✝ : α
    ⊢ Iff (LE.le a x✝) (Not (LT.lt x✝ a))
  -/
  exact not_lt.symm
  /-
    🎉 no goals
  -/


theorem bounded_lt_inter_lt [LinearOrder α] [NoMaxOrder α] (a : α) :
    Bounded (· < ·) (s ∩ { b | a < b }) ↔ Bounded (· < ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝¹ : LinearOrder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ Iff (Set.Bounded (fun x1 x2 => LT.lt x1 x2) (Inter.inter s (setOf fun b => L …
  -/
  rw [← bounded_le_iff_bounded_lt, ← bounded_le_iff_bounded_lt]
  /-
    α : Type u_1
    s : Set α
    inst✝¹ : LinearOrder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ Iff (Set.Bounded (fun x1 x2 => LE.le x1 x2) (Inter.inter s (setOf fun b => L …
  -/
  exact bounded_le_inter_lt a
  /-
    🎉 no goals
  -/


theorem unbounded_lt_inter_lt [LinearOrder α] [NoMaxOrder α] (a : α) :
    Unbounded (· < ·) (s ∩ { b | a < b }) ↔ Unbounded (· < ·) s := by
  /-
    α : Type u_1
    s : Set α
    inst✝¹ : LinearOrder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ Iff (Set.Unbounded (fun x1 x2 => LT.lt x1 x2) (Inter.inter s (setOf fun b => …
  -/
  rw [← not_bounded_iff, ← not_bounded_iff, not_iff_not]
  /-
    α : Type u_1
    s : Set α
    inst✝¹ : LinearOrder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ Iff (Set.Bounded (fun x1 x2 => LT.lt x1 x2) (Inter.inter s (setOf fun b => L …
  -/
  exact bounded_lt_inter_lt a
  /-
    🎉 no goals
  -/


theorem bounded_ge_inter_not_ge [SemilatticeInf α] (a : α) :
    Bounded (· ≥ ·) (s ∩ { b | ¬a ≤ b }) ↔ Bounded (· ≥ ·) s :=
  @bounded_le_inter_not_le αᵒᵈ s _ a


theorem unbounded_ge_inter_not_ge [SemilatticeInf α] (a : α) :
    Unbounded (· ≥ ·) (s ∩ { b | ¬a ≤ b }) ↔ Unbounded (· ≥ ·) s :=
  @unbounded_le_inter_not_le αᵒᵈ s _ a


theorem bounded_ge_inter_gt [LinearOrder α] (a : α) :
    Bounded (· ≥ ·) (s ∩ { b | b < a }) ↔ Bounded (· ≥ ·) s :=
  @bounded_le_inter_lt αᵒᵈ s _ a


theorem unbounded_ge_inter_gt [LinearOrder α] (a : α) :
    Unbounded (· ≥ ·) (s ∩ { b | b < a }) ↔ Unbounded (· ≥ ·) s :=
  @unbounded_le_inter_lt αᵒᵈ s _ a


theorem bounded_ge_inter_ge [LinearOrder α] (a : α) :
    Bounded (· ≥ ·) (s ∩ { b | b ≤ a }) ↔ Bounded (· ≥ ·) s :=
  @bounded_le_inter_le αᵒᵈ s _ a


theorem unbounded_ge_iff_unbounded_inter_ge [LinearOrder α] (a : α) :
    Unbounded (· ≥ ·) (s ∩ { b | b ≤ a }) ↔ Unbounded (· ≥ ·) s :=
  @unbounded_le_inter_le αᵒᵈ s _ a


theorem bounded_gt_inter_not_gt [SemilatticeInf α] (a : α) :
    Bounded (· > ·) (s ∩ { b | ¬a < b }) ↔ Bounded (· > ·) s :=
  @bounded_lt_inter_not_lt αᵒᵈ s _ a


theorem unbounded_gt_inter_not_gt [SemilatticeInf α] (a : α) :
    Unbounded (· > ·) (s ∩ { b | ¬a < b }) ↔ Unbounded (· > ·) s :=
  @unbounded_lt_inter_not_lt αᵒᵈ s _ a


theorem bounded_gt_inter_ge [LinearOrder α] (a : α) :
    Bounded (· > ·) (s ∩ { b | b ≤ a }) ↔ Bounded (· > ·) s :=
  @bounded_lt_inter_le αᵒᵈ s _ a


theorem unbounded_inter_ge [LinearOrder α] (a : α) :
    Unbounded (· > ·) (s ∩ { b | b ≤ a }) ↔ Unbounded (· > ·) s :=
  @unbounded_lt_inter_le αᵒᵈ s _ a


theorem bounded_gt_inter_gt [LinearOrder α] [NoMinOrder α] (a : α) :
    Bounded (· > ·) (s ∩ { b | b < a }) ↔ Bounded (· > ·) s :=
  @bounded_lt_inter_lt αᵒᵈ s _ _ a


theorem unbounded_gt_inter_gt [LinearOrder α] [NoMinOrder α] (a : α) :
    Unbounded (· > ·) (s ∩ { b | b < a }) ↔ Unbounded (· > ·) s :=
  @unbounded_lt_inter_lt αᵒᵈ s _ _ a


