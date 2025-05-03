theorem isUpperSet_empty : IsUpperSet (∅ : Set α) := fun _ _ _ => id


theorem isLowerSet_empty : IsLowerSet (∅ : Set α) := fun _ _ _ => id


theorem isUpperSet_univ : IsUpperSet (univ : Set α) := fun _ _ _ => id


theorem isLowerSet_univ : IsLowerSet (univ : Set α) := fun _ _ _ => id


theorem IsUpperSet.compl (hs : IsUpperSet s) : IsLowerSet sᶜ := fun _a _b h hb ha => hb <| hs h ha


theorem IsLowerSet.compl (hs : IsLowerSet s) : IsUpperSet sᶜ := fun _a _b h hb ha => hb <| hs h ha


@[simp]
theorem isUpperSet_compl : IsUpperSet sᶜ ↔ IsLowerSet s :=
  ⟨fun h => by
    /-
      α : Type u_1
      inst✝ : LE α
      s : Set α
      h : IsUpperSet (HasCompl.compl s)
      ⊢ IsLowerSet s
    -/
    convert h.compl
    /-
      case h.e'_3
      α : Type u_1
      inst✝ : LE α
      s : Set α
      h : IsUpperSet (HasCompl.compl s)
      ⊢ Eq s (HasCompl.compl (HasCompl.compl s))
    -/
    rw [compl_compl], IsLowerSet.compl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem isLowerSet_compl : IsLowerSet sᶜ ↔ IsUpperSet s :=
  ⟨fun h => by
    /-
      α : Type u_1
      inst✝ : LE α
      s : Set α
      h : IsLowerSet (HasCompl.compl s)
      ⊢ IsUpperSet s
    -/
    convert h.compl
    /-
      case h.e'_3
      α : Type u_1
      inst✝ : LE α
      s : Set α
      h : IsLowerSet (HasCompl.compl s)
      ⊢ Eq s (HasCompl.compl (HasCompl.compl s))
    -/
    rw [compl_compl], IsUpperSet.compl⟩
    /-
      🎉 no goals
    -/


theorem IsUpperSet.union (hs : IsUpperSet s) (ht : IsUpperSet t) : IsUpperSet (s ∪ t) :=
  fun _ _ h => Or.imp (hs h) (ht h)


theorem IsLowerSet.union (hs : IsLowerSet s) (ht : IsLowerSet t) : IsLowerSet (s ∪ t) :=
  fun _ _ h => Or.imp (hs h) (ht h)


theorem IsUpperSet.inter (hs : IsUpperSet s) (ht : IsUpperSet t) : IsUpperSet (s ∩ t) :=
  fun _ _ h => And.imp (hs h) (ht h)


theorem IsLowerSet.inter (hs : IsLowerSet s) (ht : IsLowerSet t) : IsLowerSet (s ∩ t) :=
  fun _ _ h => And.imp (hs h) (ht h)


theorem isUpperSet_sUnion {S : Set (Set α)} (hf : ∀ s ∈ S, IsUpperSet s) : IsUpperSet (⋃₀ S) :=
  fun _ _ h => Exists.imp fun _ hs => ⟨hs.1, hf _ hs.1 h hs.2⟩


theorem isLowerSet_sUnion {S : Set (Set α)} (hf : ∀ s ∈ S, IsLowerSet s) : IsLowerSet (⋃₀ S) :=
  fun _ _ h => Exists.imp fun _ hs => ⟨hs.1, hf _ hs.1 h hs.2⟩


theorem isUpperSet_iUnion {f : ι → Set α} (hf : ∀ i, IsUpperSet (f i)) : IsUpperSet (⋃ i, f i) :=
  isUpperSet_sUnion <| forall_mem_range.2 hf


theorem isLowerSet_iUnion {f : ι → Set α} (hf : ∀ i, IsLowerSet (f i)) : IsLowerSet (⋃ i, f i) :=
  isLowerSet_sUnion <| forall_mem_range.2 hf


theorem isUpperSet_iUnion₂ {f : ∀ i, κ i → Set α} (hf : ∀ i j, IsUpperSet (f i j)) :
    IsUpperSet (⋃ (i) (j), f i j) :=
  isUpperSet_iUnion fun i => isUpperSet_iUnion <| hf i


theorem isLowerSet_iUnion₂ {f : ∀ i, κ i → Set α} (hf : ∀ i j, IsLowerSet (f i j)) :
    IsLowerSet (⋃ (i) (j), f i j) :=
  isLowerSet_iUnion fun i => isLowerSet_iUnion <| hf i


theorem isUpperSet_sInter {S : Set (Set α)} (hf : ∀ s ∈ S, IsUpperSet s) : IsUpperSet (⋂₀ S) :=
  fun _ _ h => forall₂_imp fun s hs => hf s hs h


theorem isLowerSet_sInter {S : Set (Set α)} (hf : ∀ s ∈ S, IsLowerSet s) : IsLowerSet (⋂₀ S) :=
  fun _ _ h => forall₂_imp fun s hs => hf s hs h


theorem isUpperSet_iInter {f : ι → Set α} (hf : ∀ i, IsUpperSet (f i)) : IsUpperSet (⋂ i, f i) :=
  isUpperSet_sInter <| forall_mem_range.2 hf


theorem isLowerSet_iInter {f : ι → Set α} (hf : ∀ i, IsLowerSet (f i)) : IsLowerSet (⋂ i, f i) :=
  isLowerSet_sInter <| forall_mem_range.2 hf


theorem isUpperSet_iInter₂ {f : ∀ i, κ i → Set α} (hf : ∀ i j, IsUpperSet (f i j)) :
    IsUpperSet (⋂ (i) (j), f i j) :=
  isUpperSet_iInter fun i => isUpperSet_iInter <| hf i


theorem isLowerSet_iInter₂ {f : ∀ i, κ i → Set α} (hf : ∀ i j, IsLowerSet (f i j)) :
    IsLowerSet (⋂ (i) (j), f i j) :=
  isLowerSet_iInter fun i => isLowerSet_iInter <| hf i


@[simp]
theorem isLowerSet_preimage_ofDual_iff : IsLowerSet (ofDual ⁻¹' s) ↔ IsUpperSet s :=
  Iff.rfl


@[simp]
theorem isUpperSet_preimage_ofDual_iff : IsUpperSet (ofDual ⁻¹' s) ↔ IsLowerSet s :=
  Iff.rfl


@[simp]
theorem isLowerSet_preimage_toDual_iff {s : Set αᵒᵈ} : IsLowerSet (toDual ⁻¹' s) ↔ IsUpperSet s :=
  Iff.rfl


@[simp]
theorem isUpperSet_preimage_toDual_iff {s : Set αᵒᵈ} : IsUpperSet (toDual ⁻¹' s) ↔ IsLowerSet s :=
  Iff.rfl


alias ⟨_, IsUpperSet.toDual⟩ := isLowerSet_preimage_ofDual_iff


alias ⟨_, IsLowerSet.toDual⟩ := isUpperSet_preimage_ofDual_iff


alias ⟨_, IsUpperSet.ofDual⟩ := isLowerSet_preimage_toDual_iff


alias ⟨_, IsLowerSet.ofDual⟩ := isUpperSet_preimage_toDual_iff


lemma IsUpperSet.isLowerSet_preimage_coe (hs : IsUpperSet s) :
                                                                           /-
                                                                             α : Type u_1
                                                                             inst✝ : LE α
                                                                             s t : Set α
                                                                             hs : IsUpperSet s
                                                                             ⊢ Iff (IsLowerSet (Set.preimage Subtype.val t)) (∀ (b : α), Membership.mem s b …
                                                                           -/
    IsLowerSet ((↑) ⁻¹' t : Set s) ↔ ∀ b ∈ s, ∀ c ∈ t, b ≤ c → b ∈ t := by aesop
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


lemma IsLowerSet.isUpperSet_preimage_coe (hs : IsLowerSet s) :
                                                                           /-
                                                                             α : Type u_1
                                                                             inst✝ : LE α
                                                                             s t : Set α
                                                                             hs : IsLowerSet s
                                                                             ⊢ Iff (IsUpperSet (Set.preimage Subtype.val t)) (∀ (b : α), Membership.mem s b …
                                                                           -/
    IsUpperSet ((↑) ⁻¹' t : Set s) ↔ ∀ b ∈ s, ∀ c ∈ t, c ≤ b → b ∈ t := by aesop
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


lemma IsUpperSet.sdiff (hs : IsUpperSet s) (ht : ∀ b ∈ s, ∀ c ∈ t, b ≤ c → b ∈ t) :
    IsUpperSet (s \ t) :=
  fun _b _c hbc hb ↦ ⟨hs hbc hb.1, fun hc ↦ hb.2 <| ht _ hb.1 _ hc hbc⟩


lemma IsLowerSet.sdiff (hs : IsLowerSet s) (ht : ∀ b ∈ s, ∀ c ∈ t, c ≤ b → b ∈ t) :
    IsLowerSet (s \ t) :=
  fun _b _c hcb hb ↦ ⟨hs hcb hb.1, fun hc ↦ hb.2 <| ht _ hb.1 _ hc hcb⟩


lemma IsUpperSet.sdiff_of_isLowerSet (hs : IsUpperSet s) (ht : IsLowerSet t) : IsUpperSet (s \ t) :=
                 /-
                   α : Type u_1
                   inst✝ : LE α
                   s t : Set α
                   hs : IsUpperSet s
                   ht : IsLowerSet t
                   ⊢ ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le b c →  …
                 -/
  hs.sdiff <| by aesop
                 /-
                   🎉 no goals
                 -/


lemma IsLowerSet.sdiff_of_isUpperSet (hs : IsLowerSet s) (ht : IsUpperSet t) : IsLowerSet (s \ t) :=
                 /-
                   α : Type u_1
                   inst✝ : LE α
                   s t : Set α
                   hs : IsLowerSet s
                   ht : IsUpperSet t
                   ⊢ ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le c b →  …
                 -/
  hs.sdiff <| by aesop
                 /-
                   🎉 no goals
                 -/


lemma IsUpperSet.erase (hs : IsUpperSet s) (has : ∀ b ∈ s, b ≤ a → b = a) : IsUpperSet (s \ {a}) :=
                 /-
                   α : Type u_1
                   inst✝ : LE α
                   s : Set α
                   a : α
                   hs : IsUpperSet s
                   has : ∀ (b : α), Membership.mem s b → LE.le b a → Eq b a
                   ⊢ ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem (Singleton.singlet …
                 -/
  hs.sdiff <| by simpa using has
                 /-
                   🎉 no goals
                 -/


lemma IsLowerSet.erase (hs : IsLowerSet s) (has : ∀ b ∈ s, a ≤ b → b = a) : IsLowerSet (s \ {a}) :=
                 /-
                   α : Type u_1
                   inst✝ : LE α
                   s : Set α
                   a : α
                   hs : IsLowerSet s
                   has : ∀ (b : α), Membership.mem s b → LE.le a b → Eq b a
                   ⊢ ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem (Singleton.singlet …
                 -/
  hs.sdiff <| by simpa using has
                 /-
                   🎉 no goals
                 -/


theorem isUpperSet_Ici : IsUpperSet (Ici a) := fun _ _ => ge_trans


theorem isLowerSet_Iic : IsLowerSet (Iic a) := fun _ _ => le_trans


theorem isUpperSet_Ioi : IsUpperSet (Ioi a) := fun _ _ => flip lt_of_lt_of_le


theorem isLowerSet_Iio : IsLowerSet (Iio a) := fun _ _ => lt_of_le_of_lt


theorem isUpperSet_iff_Ici_subset : IsUpperSet s ↔ ∀ ⦃a⦄, a ∈ s → Ici a ⊆ s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Iff (IsUpperSet s) (∀ ⦃a : α⦄, Membership.mem s a → HasSubset.Subset (Set.Ic …
  -/
  simp [IsUpperSet, subset_def, @forall_swap (_ ∈ s)]
  /-
    🎉 no goals
  -/


theorem isLowerSet_iff_Iic_subset : IsLowerSet s ↔ ∀ ⦃a⦄, a ∈ s → Iic a ⊆ s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Iff (IsLowerSet s) (∀ ⦃a : α⦄, Membership.mem s a → HasSubset.Subset (Set.Ii …
  -/
  simp [IsLowerSet, subset_def, @forall_swap (_ ∈ s)]
  /-
    🎉 no goals
  -/


alias ⟨IsUpperSet.Ici_subset, _⟩ := isUpperSet_iff_Ici_subset


alias ⟨IsLowerSet.Iic_subset, _⟩ := isLowerSet_iff_Iic_subset


theorem IsUpperSet.Ioi_subset (h : IsUpperSet s) ⦃a⦄ (ha : a ∈ s) : Ioi a ⊆ s :=
  Ioi_subset_Ici_self.trans <| h.Ici_subset ha


theorem IsLowerSet.Iio_subset (h : IsLowerSet s) ⦃a⦄ (ha : a ∈ s) : Iio a ⊆ s :=
  h.toDual.Ioi_subset ha


theorem IsUpperSet.ordConnected (h : IsUpperSet s) : s.OrdConnected :=
  ⟨fun _ ha _ _ => Icc_subset_Ici_self.trans <| h.Ici_subset ha⟩


theorem IsLowerSet.ordConnected (h : IsLowerSet s) : s.OrdConnected :=
  ⟨fun _ _ _ hb => Icc_subset_Iic_self.trans <| h.Iic_subset hb⟩


theorem IsUpperSet.preimage (hs : IsUpperSet s) {f : β → α} (hf : Monotone f) :
    IsUpperSet (f ⁻¹' s : Set β) := fun _ _ h => hs <| hf h


theorem IsLowerSet.preimage (hs : IsLowerSet s) {f : β → α} (hf : Monotone f) :
    IsLowerSet (f ⁻¹' s : Set β) := fun _ _ h => hs <| hf h


theorem IsUpperSet.image (hs : IsUpperSet s) (f : α ≃o β) : IsUpperSet (f '' s : Set β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    hs : IsUpperSet s
    f : OrderIso α β
    ⊢ IsUpperSet (Set.image (⇑f) s)
  -/
  change IsUpperSet ((f : α ≃ β) '' s)
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    hs : IsUpperSet s
    f : OrderIso α β
    ⊢ IsUpperSet (Set.image (⇑↑f) s)
  -/
  rw [Set.image_equiv_eq_preimage_symm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    hs : IsUpperSet s
    f : OrderIso α β
    ⊢ IsUpperSet (Set.preimage (⇑(↑f).symm) s)
  -/
  exact hs.preimage f.symm.monotone
  /-
    🎉 no goals
  -/


theorem IsLowerSet.image (hs : IsLowerSet s) (f : α ≃o β) : IsLowerSet (f '' s : Set β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    hs : IsLowerSet s
    f : OrderIso α β
    ⊢ IsLowerSet (Set.image (⇑f) s)
  -/
  change IsLowerSet ((f : α ≃ β) '' s)
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    hs : IsLowerSet s
    f : OrderIso α β
    ⊢ IsLowerSet (Set.image (⇑↑f) s)
  -/
  rw [Set.image_equiv_eq_preimage_symm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    hs : IsLowerSet s
    f : OrderIso α β
    ⊢ IsLowerSet (Set.preimage (⇑(↑f).symm) s)
  -/
  exact hs.preimage f.symm.monotone
  /-
    🎉 no goals
  -/


theorem OrderEmbedding.image_Ici (e : α ↪o β) (he : IsUpperSet (range e)) (a : α) :
    e '' Ici a = Ici (e a) := by
  rw [← e.preimage_Ici, image_preimage_eq_inter_range,
    inter_eq_left.2 <| he.Ici_subset (mem_range_self _)]


theorem OrderEmbedding.image_Iic (e : α ↪o β) (he : IsLowerSet (range e)) (a : α) :
    e '' Iic a = Iic (e a) :=
  e.dual.image_Ici he a


theorem OrderEmbedding.image_Ioi (e : α ↪o β) (he : IsUpperSet (range e)) (a : α) :
    e '' Ioi a = Ioi (e a) := by
  rw [← e.preimage_Ioi, image_preimage_eq_inter_range,
    inter_eq_left.2 <| he.Ioi_subset (mem_range_self _)]


theorem OrderEmbedding.image_Iio (e : α ↪o β) (he : IsLowerSet (range e)) (a : α) :
    e '' Iio a = Iio (e a) :=
  e.dual.image_Ioi he a


@[simp]
theorem Set.monotone_mem : Monotone (· ∈ s) ↔ IsUpperSet s :=
  Iff.rfl


@[simp]
theorem Set.antitone_mem : Antitone (· ∈ s) ↔ IsLowerSet s :=
  forall_swap


@[simp]
theorem isUpperSet_setOf : IsUpperSet { a | p a } ↔ Monotone p :=
  Iff.rfl


@[simp]
theorem isLowerSet_setOf : IsLowerSet { a | p a } ↔ Antitone p :=
  forall_swap


lemma IsUpperSet.upperBounds_subset (hs : IsUpperSet s) : s.Nonempty → upperBounds s ⊆ s :=
  fun ⟨_a, ha⟩ _b hb ↦ hs (hb ha) ha


lemma IsLowerSet.lowerBounds_subset (hs : IsLowerSet s) : s.Nonempty → lowerBounds s ⊆ s :=
  fun ⟨_a, ha⟩ _b hb ↦ hs (hb ha) ha


theorem IsLowerSet.top_mem (hs : IsLowerSet s) : ⊤ ∈ s ↔ s = univ :=
  ⟨fun h => eq_univ_of_forall fun _ => hs le_top h, fun h => h.symm ▸ mem_univ _⟩


theorem IsUpperSet.top_mem (hs : IsUpperSet s) : ⊤ ∈ s ↔ s.Nonempty :=
  ⟨fun h => ⟨_, h⟩, fun ⟨_a, ha⟩ => hs le_top ha⟩


theorem IsUpperSet.not_top_mem (hs : IsUpperSet s) : ⊤ ∉ s ↔ s = ∅ :=
  hs.top_mem.not.trans not_nonempty_iff_eq_empty


theorem IsUpperSet.bot_mem (hs : IsUpperSet s) : ⊥ ∈ s ↔ s = univ :=
  ⟨fun h => eq_univ_of_forall fun _ => hs bot_le h, fun h => h.symm ▸ mem_univ _⟩


theorem IsLowerSet.bot_mem (hs : IsLowerSet s) : ⊥ ∈ s ↔ s.Nonempty :=
  ⟨fun h => ⟨_, h⟩, fun ⟨_a, ha⟩ => hs bot_le ha⟩


theorem IsLowerSet.not_bot_mem (hs : IsLowerSet s) : ⊥ ∉ s ↔ s = ∅ :=
  hs.bot_mem.not.trans not_nonempty_iff_eq_empty


theorem IsUpperSet.not_bddAbove (hs : IsUpperSet s) : s.Nonempty → ¬BddAbove s := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    s : Set α
    inst✝ : NoMaxOrder α
    hs : IsUpperSet s
    ⊢ s.Nonempty → Not (BddAbove s)
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Preorder α
    s : Set α
    inst✝ : NoMaxOrder α
    hs : IsUpperSet s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem (upperBounds s) b
    ⊢ False
  -/
  obtain ⟨c, hc⟩ := exists_gt b
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : Preorder α
    s : Set α
    inst✝ : NoMaxOrder α
    hs : IsUpperSet s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem (upperBounds s) b
    c : α
    hc : LT.lt b c
    ⊢ False
  -/
  exact hc.not_le (hb <| hs ((hb ha).trans hc.le) ha)
  /-
    🎉 no goals
  -/


theorem not_bddAbove_Ici : ¬BddAbove (Ici a) :=
  (isUpperSet_Ici _).not_bddAbove nonempty_Ici


theorem not_bddAbove_Ioi : ¬BddAbove (Ioi a) :=
  (isUpperSet_Ioi _).not_bddAbove nonempty_Ioi


theorem IsLowerSet.not_bddBelow (hs : IsLowerSet s) : s.Nonempty → ¬BddBelow s := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    s : Set α
    inst✝ : NoMinOrder α
    hs : IsLowerSet s
    ⊢ s.Nonempty → Not (BddBelow s)
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Preorder α
    s : Set α
    inst✝ : NoMinOrder α
    hs : IsLowerSet s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem (lowerBounds s) b
    ⊢ False
  -/
  obtain ⟨c, hc⟩ := exists_lt b
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : Preorder α
    s : Set α
    inst✝ : NoMinOrder α
    hs : IsLowerSet s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem (lowerBounds s) b
    c : α
    hc : LT.lt c b
    ⊢ False
  -/
  exact hc.not_le (hb <| hs (hc.le.trans <| hb ha) ha)
  /-
    🎉 no goals
  -/


theorem not_bddBelow_Iic : ¬BddBelow (Iic a) :=
  (isLowerSet_Iic _).not_bddBelow nonempty_Iic


theorem not_bddBelow_Iio : ¬BddBelow (Iio a) :=
  (isLowerSet_Iio _).not_bddBelow nonempty_Iio


theorem isUpperSet_iff_forall_lt : IsUpperSet s ↔ ∀ ⦃a b : α⦄, a < b → a ∈ s → b ∈ s :=
                            /-
                              α : Type u_1
                              inst✝ : PartialOrder α
                              s : Set α
                              a : α
                              ⊢ Iff (∀ ⦃b : α⦄, LE.le a b → Membership.mem s a → Membership.mem s b) (∀ ⦃b : …
                            -/
  forall_congr' fun a => by simp [le_iff_eq_or_lt, or_imp, forall_and]
                            /-
                              🎉 no goals
                            -/


theorem isLowerSet_iff_forall_lt : IsLowerSet s ↔ ∀ ⦃a b : α⦄, b < a → a ∈ s → b ∈ s :=
                            /-
                              α : Type u_1
                              inst✝ : PartialOrder α
                              s : Set α
                              a : α
                              ⊢ Iff (∀ ⦃b : α⦄, LE.le b a → Membership.mem s a → Membership.mem s b) (∀ ⦃b : …
                            -/
  forall_congr' fun a => by simp [le_iff_eq_or_lt, or_imp, forall_and]
                            /-
                              🎉 no goals
                            -/


theorem isUpperSet_iff_Ioi_subset : IsUpperSet s ↔ ∀ ⦃a⦄, a ∈ s → Ioi a ⊆ s := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Set α
    ⊢ Iff (IsUpperSet s) (∀ ⦃a : α⦄, Membership.mem s a → HasSubset.Subset (Set.Io …
  -/
  simp [isUpperSet_iff_forall_lt, subset_def, @forall_swap (_ ∈ s)]
  /-
    🎉 no goals
  -/


theorem isLowerSet_iff_Iio_subset : IsLowerSet s ↔ ∀ ⦃a⦄, a ∈ s → Iio a ⊆ s := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Set α
    ⊢ Iff (IsLowerSet s) (∀ ⦃a : α⦄, Membership.mem s a → HasSubset.Subset (Set.Ii …
  -/
  simp [isLowerSet_iff_forall_lt, subset_def, @forall_swap (_ ∈ s)]
  /-
    🎉 no goals
  -/


lemma Fibration.isLowerSet_image [LE α] [LE β] (hf : Fibration (· ≤ ·) (· ≤ ·) f)
    {s : Set α} (hs : IsLowerSet s) : IsLowerSet (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : LE α
    inst✝ : LE β
    hf : Relation.Fibration (fun x1 x2 => LE.le x1 x2) (fun x1 x2 => LE.le x1 x2) f
    s : Set α
    hs : IsLowerSet s
    ⊢ IsLowerSet (Set.image f s)
  -/
  rintro _ y' e ⟨x, hx, rfl⟩; obtain ⟨y, e', rfl⟩ := hf e; exact ⟨_, hs e' hx, rfl⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


alias _root_.IsLowerSet.image_fibration := Fibration.isLowerSet_image


lemma fibration_iff_isLowerSet_image_Iic [Preorder α] [LE β] :
    Fibration (· ≤ ·) (· ≤ ·) f ↔ ∀ x, IsLowerSet (f '' Iic x) :=
  ⟨fun h x ↦ (isLowerSet_Iic x).image_fibration h, fun H x _ e ↦ H x e ⟨x, le_rfl, rfl⟩⟩


lemma fibration_iff_isLowerSet_image [Preorder α] [LE β] :
    Fibration (· ≤ ·) (· ≤ ·) f ↔ ∀ s, IsLowerSet s → IsLowerSet (f '' s) :=
  ⟨Fibration.isLowerSet_image,
    fun H ↦ fibration_iff_isLowerSet_image_Iic.mpr (H _ <| isLowerSet_Iic ·)⟩


lemma fibration_iff_image_Iic [Preorder α] [Preorder β] (hf : Monotone f) :
    Fibration (· ≤ ·) (· ≤ ·) f ↔ ∀ x, f '' Iic x = Iic (f x) :=
  ⟨fun H x ↦ le_antisymm (fun _ ⟨_, hy, e⟩ ↦ e ▸ hf hy)
    ((H.isLowerSet_image (isLowerSet_Iic x)).Iic_subset ⟨x, le_rfl, rfl⟩),
    fun H ↦ fibration_iff_isLowerSet_image_Iic.mpr (fun x ↦ (H x).symm ▸ isLowerSet_Iic (f x))⟩


lemma Fibration.isUpperSet_image [LE α] [LE β] (hf : Fibration (· ≥ ·) (· ≥ ·) f)
    {s : Set α} (hs : IsUpperSet s) : IsUpperSet (f '' s) :=
  @Fibration.isLowerSet_image αᵒᵈ βᵒᵈ _ _ _ hf s hs


alias _root_.IsUpperSet.image_fibration := Fibration.isUpperSet_image


lemma fibration_iff_isUpperSet_image_Ici [Preorder α] [LE β] :
    Fibration (· ≥ ·) (· ≥ ·) f ↔ ∀ x, IsUpperSet (f '' Ici x) :=
  @fibration_iff_isLowerSet_image_Iic αᵒᵈ βᵒᵈ _ _ _


lemma fibration_iff_isUpperSet_image [Preorder α] [LE β] :
    Fibration (· ≥ ·) (· ≥ ·) f ↔ ∀ s, IsUpperSet s → IsUpperSet (f '' s) :=
  @fibration_iff_isLowerSet_image αᵒᵈ βᵒᵈ _ _ _


lemma fibration_iff_image_Ici [Preorder α] [Preorder β] (hf : Monotone f) :
    Fibration (· ≥ ·) (· ≥ ·) f ↔ ∀ x, f '' Ici x = Ici (f x) :=
  fibration_iff_image_Iic hf.dual


theorem IsUpperSet.total (hs : IsUpperSet s) (ht : IsUpperSet t) : s ⊆ t ∨ t ⊆ s := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    hs : IsUpperSet s
    ht : IsUpperSet t
    ⊢ Or (HasSubset.Subset s t) (HasSubset.Subset t s)
  -/
  by_contra! h
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    hs : IsUpperSet s
    ht : IsUpperSet t
    h : And (Not (HasSubset.Subset s t)) (Not (HasSubset.Subset t s))
    ⊢ False
  -/
  simp_rw [Set.not_subset] at h
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    hs : IsUpperSet s
    ht : IsUpperSet t
    h : And (Exists fun a => And (Membership.mem s a) (Not (Membership.mem t a)))  …
    ⊢ False
  -/
  obtain ⟨⟨a, has, hat⟩, b, hbt, hbs⟩ := h
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Set α
    hs : IsUpperSet s
    ht : IsUpperSet t
    a : α
    has : Membership.mem s a
    hat : Not (Membership.mem t a)
    b : α
    hbt : Membership.mem t b
    hbs : Not (Membership.mem s b)
    ⊢ False
  -/
  obtain hab | hba := le_total a b
    /-
      case intro.intro.intro.intro.intro.inl
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Set α
      hs : IsUpperSet s
      ht : IsUpperSet t
      a : α
      has : Membership.mem s a
      hat : Not (Membership.mem t a)
      b : α
      hbt : Membership.mem t b
      hbs : Not (Membership.mem s b)
      hab : LE.le a b
      ⊢ False
    -/
  · exact hbs (hs hab has)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.inr
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Set α
      hs : IsUpperSet s
      ht : IsUpperSet t
      a : α
      has : Membership.mem s a
      hat : Not (Membership.mem t a)
      b : α
      hbt : Membership.mem t b
      hbs : Not (Membership.mem s b)
      hba : LE.le b a
      ⊢ False
    -/
  · exact hat (ht hba hbt)
    /-
      🎉 no goals
    -/


theorem IsLowerSet.total (hs : IsLowerSet s) (ht : IsLowerSet t) : s ⊆ t ∨ t ⊆ s :=
  hs.toDual.total ht.toDual


instance : SetLike (UpperSet α) α where
  coe := UpperSet.carrier
                             /-
                               α : Type u_1
                               β : Type u_2
                               γ : Type u_3
                               ι : Sort u_4
                               κ : ι → Sort u_5
                               inst✝ : LE α
                               s t : UpperSet α
                               h : Eq s.carrier t.carrier
                               ⊢ Eq s t
                             -/
  coe_injective' s t h := by cases s; cases t; congr
                                               /-
                                                 🎉 no goals
                                               -/


/-- See Note [custom simps projection]. -/
def Simps.coe (s : UpperSet α) : Set α := s


@[ext]
theorem ext {s t : UpperSet α} : (s : Set α) = t → s = t :=
  SetLike.ext'


@[simp]
theorem carrier_eq_coe (s : UpperSet α) : s.carrier = s :=
  rfl


@[simp] protected lemma upper (s : UpperSet α) : IsUpperSet (s : Set α) := s.upper'


@[simp, norm_cast] lemma coe_mk (s : Set α) (hs) : mk s hs = s := rfl

@[simp] lemma mem_mk {s : Set α} (hs) {a : α} : a ∈ mk s hs ↔ a ∈ s := Iff.rfl


instance : SetLike (LowerSet α) α where
  coe := LowerSet.carrier
                             /-
                               α : Type u_1
                               β : Type u_2
                               γ : Type u_3
                               ι : Sort u_4
                               κ : ι → Sort u_5
                               inst✝ : LE α
                               s t : LowerSet α
                               h : Eq s.carrier t.carrier
                               ⊢ Eq s t
                             -/
  coe_injective' s t h := by cases s; cases t; congr
                                               /-
                                                 🎉 no goals
                                               -/


/-- See Note [custom simps projection]. -/
def Simps.coe (s : LowerSet α) : Set α := s


@[ext]
theorem ext {s t : LowerSet α} : (s : Set α) = t → s = t :=
  SetLike.ext'


@[simp]
theorem carrier_eq_coe (s : LowerSet α) : s.carrier = s :=
  rfl


@[simp] protected lemma lower (s : LowerSet α) : IsLowerSet (s : Set α) := s.lower'


instance : Max (UpperSet α) :=
  ⟨fun s t => ⟨s ∩ t, s.upper.inter t.upper⟩⟩


instance : Min (UpperSet α) :=
  ⟨fun s t => ⟨s ∪ t, s.upper.union t.upper⟩⟩


instance : Top (UpperSet α) :=
  ⟨⟨∅, isUpperSet_empty⟩⟩


instance : Bot (UpperSet α) :=
  ⟨⟨univ, isUpperSet_univ⟩⟩


instance : SupSet (UpperSet α) :=
  ⟨fun S => ⟨⋂ s ∈ S, ↑s, isUpperSet_iInter₂ fun s _ => s.upper⟩⟩


instance : InfSet (UpperSet α) :=
  ⟨fun S => ⟨⋃ s ∈ S, ↑s, isUpperSet_iUnion₂ fun s _ => s.upper⟩⟩


instance completeLattice : CompleteLattice (UpperSet α) :=
  (toDual.injective.comp SetLike.coe_injective).completeLattice _ (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) (fun _ => rfl) rfl rfl


instance completelyDistribLattice : CompletelyDistribLattice (UpperSet α) :=
  .ofMinimalAxioms <|
    (toDual.injective.comp SetLike.coe_injective).completelyDistribLatticeMinimalAxioms .of _
      (fun _ _ => rfl) (fun _ _ => rfl) (fun _ => rfl) (fun _ => rfl) rfl rfl


instance : Inhabited (UpperSet α) :=
  ⟨⊥⟩


@[simp 1100, norm_cast]
theorem coe_subset_coe : (s : Set α) ⊆ t ↔ t ≤ s :=
  Iff.rfl


@[simp 1100, norm_cast] lemma coe_ssubset_coe : (s : Set α) ⊂ t ↔ t < s := Iff.rfl


@[simp, norm_cast]
theorem coe_top : ((⊤ : UpperSet α) : Set α) = ∅ :=
  rfl


@[simp, norm_cast]
theorem coe_bot : ((⊥ : UpperSet α) : Set α) = univ :=
  rfl


@[simp, norm_cast]
                                                       /-
                                                         α : Type u_1
                                                         inst✝ : LE α
                                                         s : UpperSet α
                                                         ⊢ Iff (Eq (↑s) Set.univ) (Eq s Bot.bot)
                                                       -/
theorem coe_eq_univ : (s : Set α) = univ ↔ s = ⊥ := by simp [SetLike.ext'_iff]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp, norm_cast]
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : LE α
                                                       s : UpperSet α
                                                       ⊢ Iff (Eq (↑s) EmptyCollection.emptyCollection) (Eq s Top.top)
                                                     -/
theorem coe_eq_empty : (s : Set α) = ∅ ↔ s = ⊤ := by simp [SetLike.ext'_iff]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp, norm_cast] lemma coe_nonempty : (s : Set α).Nonempty ↔ s ≠ ⊤ :=
  nonempty_iff_ne_empty.trans coe_eq_empty.not


@[simp, norm_cast]
theorem coe_sup (s t : UpperSet α) : (↑(s ⊔ t) : Set α) = (s : Set α) ∩ t :=
  rfl


@[simp, norm_cast]
theorem coe_inf (s t : UpperSet α) : (↑(s ⊓ t) : Set α) = (s : Set α) ∪ t :=
  rfl


@[simp, norm_cast]
theorem coe_sSup (S : Set (UpperSet α)) : (↑(sSup S) : Set α) = ⋂ s ∈ S, ↑s :=
  rfl


@[simp, norm_cast]
theorem coe_sInf (S : Set (UpperSet α)) : (↑(sInf S) : Set α) = ⋃ s ∈ S, ↑s :=
  rfl


@[simp, norm_cast]
                                                                               /-
                                                                                 α : Type u_1
                                                                                 ι : Sort u_4
                                                                                 inst✝ : LE α
                                                                                 f : ι → UpperSet α
                                                                                 ⊢ Eq (↑(iSup fun i => f i)) (Set.iInter fun i => ↑(f i))
                                                                               -/
theorem coe_iSup (f : ι → UpperSet α) : (↑(⨆ i, f i) : Set α) = ⋂ i, f i := by simp [iSup]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp, norm_cast]
                                                                               /-
                                                                                 α : Type u_1
                                                                                 ι : Sort u_4
                                                                                 inst✝ : LE α
                                                                                 f : ι → UpperSet α
                                                                                 ⊢ Eq (↑(iInf fun i => f i)) (Set.iUnion fun i => ↑(f i))
                                                                               -/
theorem coe_iInf (f : ι → UpperSet α) : (↑(⨅ i, f i) : Set α) = ⋃ i, f i := by simp [iInf]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[norm_cast] -- Porting note: no longer a `simp`
theorem coe_iSup₂ (f : ∀ i, κ i → UpperSet α) :
                                                           /-
                                                             α : Type u_1
                                                             ι : Sort u_4
                                                             κ : ι → Sort u_5
                                                             inst✝ : LE α
                                                             f : (i : ι) → κ i → UpperSet α
                                                             ⊢ Eq (↑(iSup fun i => iSup fun j => f i j)) (Set.iInter fun i => Set.iInter fu …
                                                           -/
    (↑(⨆ (i) (j), f i j) : Set α) = ⋂ (i) (j), f i j := by simp_rw [coe_iSup]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[norm_cast] -- Porting note: no longer a `simp`
theorem coe_iInf₂ (f : ∀ i, κ i → UpperSet α) :
                                                           /-
                                                             α : Type u_1
                                                             ι : Sort u_4
                                                             κ : ι → Sort u_5
                                                             inst✝ : LE α
                                                             f : (i : ι) → κ i → UpperSet α
                                                             ⊢ Eq (↑(iInf fun i => iInf fun j => f i j)) (Set.iUnion fun i => Set.iUnion fu …
                                                           -/
    (↑(⨅ (i) (j), f i j) : Set α) = ⋃ (i) (j), f i j := by simp_rw [coe_iInf]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem not_mem_top : a ∉ (⊤ : UpperSet α) :=
  id


@[simp]
theorem mem_bot : a ∈ (⊥ : UpperSet α) :=
  trivial


@[simp]
theorem mem_sup_iff : a ∈ s ⊔ t ↔ a ∈ s ∧ a ∈ t :=
  Iff.rfl


@[simp]
theorem mem_inf_iff : a ∈ s ⊓ t ↔ a ∈ s ∨ a ∈ t :=
  Iff.rfl


@[simp]
theorem mem_sSup_iff : a ∈ sSup S ↔ ∀ s ∈ S, a ∈ s :=
  mem_iInter₂


@[simp]
theorem mem_sInf_iff : a ∈ sInf S ↔ ∃ s ∈ S, a ∈ s :=
                          /-
                            α : Type u_1
                            inst✝ : LE α
                            S : Set (UpperSet α)
                            a : α
                            ⊢ Iff (Exists fun i => Exists fun j => Membership.mem (↑i) a) (Exists fun s => …
                          -/
  mem_iUnion₂.trans <| by simp only [exists_prop, SetLike.mem_coe]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem mem_iSup_iff {f : ι → UpperSet α} : (a ∈ ⨆ i, f i) ↔ ∀ i, a ∈ f i := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : LE α
    a : α
    f : ι → UpperSet α
    ⊢ Iff (Membership.mem (iSup fun i => f i) a) (∀ (i : ι), Membership.mem (f i) a)
  -/
  rw [← SetLike.mem_coe, coe_iSup]
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : LE α
    a : α
    f : ι → UpperSet α
    ⊢ Iff (Membership.mem (Set.iInter fun i => ↑(f i)) a) (∀ (i : ι), Membership.m …
  -/
  exact mem_iInter
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_iInf_iff {f : ι → UpperSet α} : (a ∈ ⨅ i, f i) ↔ ∃ i, a ∈ f i := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : LE α
    a : α
    f : ι → UpperSet α
    ⊢ Iff (Membership.mem (iInf fun i => f i) a) (Exists fun i => Membership.mem ( …
  -/
  rw [← SetLike.mem_coe, coe_iInf]
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : LE α
    a : α
    f : ι → UpperSet α
    ⊢ Iff (Membership.mem (Set.iUnion fun i => ↑(f i)) a) (Exists fun i => Members …
  -/
  exact mem_iUnion
  /-
    🎉 no goals
  -/

-- Porting note: no longer a @[simp]

theorem mem_iSup₂_iff {f : ∀ i, κ i → UpperSet α} : (a ∈ ⨆ (i) (j), f i j) ↔ ∀ i j, a ∈ f i j := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_5
    inst✝ : LE α
    a : α
    f : (i : ι) → κ i → UpperSet α
    ⊢ Iff (Membership.mem (iSup fun i => iSup fun j => f i j) a) (∀ (i : ι) (j : κ …
  -/
  simp_rw [mem_iSup_iff]
  /-
    🎉 no goals
  -/

-- Porting note: no longer a @[simp]

theorem mem_iInf₂_iff {f : ∀ i, κ i → UpperSet α} : (a ∈ ⨅ (i) (j), f i j) ↔ ∃ i j, a ∈ f i j := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_5
    inst✝ : LE α
    a : α
    f : (i : ι) → κ i → UpperSet α
    ⊢ Iff (Membership.mem (iInf fun i => iInf fun j => f i j) a) (Exists fun i =>  …
  -/
  simp_rw [mem_iInf_iff]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem codisjoint_coe : Codisjoint (s : Set α) t ↔ Disjoint s t := by
  /-
    α : Type u_1
    inst✝ : LE α
    s t : UpperSet α
    ⊢ Iff (Codisjoint ↑s ↑t) (Disjoint s t)
  -/
  simp [disjoint_iff, codisjoint_iff, SetLike.ext'_iff]
  /-
    🎉 no goals
  -/


instance : Max (LowerSet α) :=
  ⟨fun s t => ⟨s ∪ t, fun _ _ h => Or.imp (s.lower h) (t.lower h)⟩⟩


instance : Min (LowerSet α) :=
  ⟨fun s t => ⟨s ∩ t, fun _ _ h => And.imp (s.lower h) (t.lower h)⟩⟩


instance : Top (LowerSet α) :=
  ⟨⟨univ, fun _ _ _ => id⟩⟩


instance : Bot (LowerSet α) :=
  ⟨⟨∅, fun _ _ _ => id⟩⟩


instance : SupSet (LowerSet α) :=
  ⟨fun S => ⟨⋃ s ∈ S, ↑s, isLowerSet_iUnion₂ fun s _ => s.lower⟩⟩


instance : InfSet (LowerSet α) :=
  ⟨fun S => ⟨⋂ s ∈ S, ↑s, isLowerSet_iInter₂ fun s _ => s.lower⟩⟩


instance completeLattice : CompleteLattice (LowerSet α) :=
  SetLike.coe_injective.completeLattice _ (fun _ _ => rfl) (fun _ _ => rfl) (fun _ => rfl)
    (fun _ => rfl) rfl rfl


instance completelyDistribLattice : CompletelyDistribLattice (LowerSet α) :=
  .ofMinimalAxioms <| SetLike.coe_injective.completelyDistribLatticeMinimalAxioms .of _
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ => rfl) (fun _ => rfl) rfl rfl


instance : Inhabited (LowerSet α) :=
  ⟨⊥⟩


@[norm_cast] lemma coe_subset_coe : (s : Set α) ⊆ t ↔ s ≤ t := Iff.rfl


@[norm_cast] lemma coe_ssubset_coe : (s : Set α) ⊂ t ↔ s < t := Iff.rfl


@[simp, norm_cast]
theorem coe_top : ((⊤ : LowerSet α) : Set α) = univ :=
  rfl


@[simp, norm_cast]
theorem coe_bot : ((⊥ : LowerSet α) : Set α) = ∅ :=
  rfl


@[simp, norm_cast]
                                                       /-
                                                         α : Type u_1
                                                         inst✝ : LE α
                                                         s : LowerSet α
                                                         ⊢ Iff (Eq (↑s) Set.univ) (Eq s Top.top)
                                                       -/
theorem coe_eq_univ : (s : Set α) = univ ↔ s = ⊤ := by simp [SetLike.ext'_iff]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp, norm_cast]
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : LE α
                                                       s : LowerSet α
                                                       ⊢ Iff (Eq (↑s) EmptyCollection.emptyCollection) (Eq s Bot.bot)
                                                     -/
theorem coe_eq_empty : (s : Set α) = ∅ ↔ s = ⊥ := by simp [SetLike.ext'_iff]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp, norm_cast] lemma coe_nonempty : (s : Set α).Nonempty ↔ s ≠ ⊥ :=
  nonempty_iff_ne_empty.trans coe_eq_empty.not


@[simp, norm_cast]
theorem coe_sup (s t : LowerSet α) : (↑(s ⊔ t) : Set α) = (s : Set α) ∪ t :=
  rfl


@[simp, norm_cast]
theorem coe_inf (s t : LowerSet α) : (↑(s ⊓ t) : Set α) = (s : Set α) ∩ t :=
  rfl


@[simp, norm_cast]
theorem coe_sSup (S : Set (LowerSet α)) : (↑(sSup S) : Set α) = ⋃ s ∈ S, ↑s :=
  rfl


@[simp, norm_cast]
theorem coe_sInf (S : Set (LowerSet α)) : (↑(sInf S) : Set α) = ⋂ s ∈ S, ↑s :=
  rfl


@[simp, norm_cast]
theorem coe_iSup (f : ι → LowerSet α) : (↑(⨆ i, f i) : Set α) = ⋃ i, f i := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : LE α
    f : ι → LowerSet α
    ⊢ Eq (↑(iSup fun i => f i)) (Set.iUnion fun i => ↑(f i))
  -/
  simp_rw [iSup, coe_sSup, mem_range, iUnion_exists, iUnion_iUnion_eq']
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_iInf (f : ι → LowerSet α) : (↑(⨅ i, f i) : Set α) = ⋂ i, f i := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : LE α
    f : ι → LowerSet α
    ⊢ Eq (↑(iInf fun i => f i)) (Set.iInter fun i => ↑(f i))
  -/
  simp_rw [iInf, coe_sInf, mem_range, iInter_exists, iInter_iInter_eq']
  /-
    🎉 no goals
  -/


@[norm_cast] -- Porting note: no longer a `simp`
theorem coe_iSup₂ (f : ∀ i, κ i → LowerSet α) :
                                                           /-
                                                             α : Type u_1
                                                             ι : Sort u_4
                                                             κ : ι → Sort u_5
                                                             inst✝ : LE α
                                                             f : (i : ι) → κ i → LowerSet α
                                                             ⊢ Eq (↑(iSup fun i => iSup fun j => f i j)) (Set.iUnion fun i => Set.iUnion fu …
                                                           -/
    (↑(⨆ (i) (j), f i j) : Set α) = ⋃ (i) (j), f i j := by simp_rw [coe_iSup]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[norm_cast] -- Porting note: no longer a `simp`
theorem coe_iInf₂ (f : ∀ i, κ i → LowerSet α) :
                                                           /-
                                                             α : Type u_1
                                                             ι : Sort u_4
                                                             κ : ι → Sort u_5
                                                             inst✝ : LE α
                                                             f : (i : ι) → κ i → LowerSet α
                                                             ⊢ Eq (↑(iInf fun i => iInf fun j => f i j)) (Set.iInter fun i => Set.iInter fu …
                                                           -/
    (↑(⨅ (i) (j), f i j) : Set α) = ⋂ (i) (j), f i j := by simp_rw [coe_iInf]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem mem_top : a ∈ (⊤ : LowerSet α) :=
  trivial


@[simp]
theorem not_mem_bot : a ∉ (⊥ : LowerSet α) :=
  id


@[simp]
theorem mem_sup_iff : a ∈ s ⊔ t ↔ a ∈ s ∨ a ∈ t :=
  Iff.rfl


@[simp]
theorem mem_inf_iff : a ∈ s ⊓ t ↔ a ∈ s ∧ a ∈ t :=
  Iff.rfl


@[simp]
theorem mem_sSup_iff : a ∈ sSup S ↔ ∃ s ∈ S, a ∈ s :=
                          /-
                            α : Type u_1
                            inst✝ : LE α
                            S : Set (LowerSet α)
                            a : α
                            ⊢ Iff (Exists fun i => Exists fun j => Membership.mem (↑i) a) (Exists fun s => …
                          -/
  mem_iUnion₂.trans <| by simp only [exists_prop, SetLike.mem_coe]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem mem_sInf_iff : a ∈ sInf S ↔ ∀ s ∈ S, a ∈ s :=
  mem_iInter₂


@[simp]
theorem mem_iSup_iff {f : ι → LowerSet α} : (a ∈ ⨆ i, f i) ↔ ∃ i, a ∈ f i := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : LE α
    a : α
    f : ι → LowerSet α
    ⊢ Iff (Membership.mem (iSup fun i => f i) a) (Exists fun i => Membership.mem ( …
  -/
  rw [← SetLike.mem_coe, coe_iSup]
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : LE α
    a : α
    f : ι → LowerSet α
    ⊢ Iff (Membership.mem (Set.iUnion fun i => ↑(f i)) a) (Exists fun i => Members …
  -/
  exact mem_iUnion
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_iInf_iff {f : ι → LowerSet α} : (a ∈ ⨅ i, f i) ↔ ∀ i, a ∈ f i := by
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : LE α
    a : α
    f : ι → LowerSet α
    ⊢ Iff (Membership.mem (iInf fun i => f i) a) (∀ (i : ι), Membership.mem (f i) a)
  -/
  rw [← SetLike.mem_coe, coe_iInf]
  /-
    α : Type u_1
    ι : Sort u_4
    inst✝ : LE α
    a : α
    f : ι → LowerSet α
    ⊢ Iff (Membership.mem (Set.iInter fun i => ↑(f i)) a) (∀ (i : ι), Membership.m …
  -/
  exact mem_iInter
  /-
    🎉 no goals
  -/

-- Porting note: no longer a @[simp]

theorem mem_iSup₂_iff {f : ∀ i, κ i → LowerSet α} : (a ∈ ⨆ (i) (j), f i j) ↔ ∃ i j, a ∈ f i j := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_5
    inst✝ : LE α
    a : α
    f : (i : ι) → κ i → LowerSet α
    ⊢ Iff (Membership.mem (iSup fun i => iSup fun j => f i j) a) (Exists fun i =>  …
  -/
  simp_rw [mem_iSup_iff]
  /-
    🎉 no goals
  -/

-- Porting note: no longer a @[simp]

theorem mem_iInf₂_iff {f : ∀ i, κ i → LowerSet α} : (a ∈ ⨅ (i) (j), f i j) ↔ ∀ i j, a ∈ f i j := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_5
    inst✝ : LE α
    a : α
    f : (i : ι) → κ i → LowerSet α
    ⊢ Iff (Membership.mem (iInf fun i => iInf fun j => f i j) a) (∀ (i : ι) (j : κ …
  -/
  simp_rw [mem_iInf_iff]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem disjoint_coe : Disjoint (s : Set α) t ↔ Disjoint s t := by
  /-
    α : Type u_1
    inst✝ : LE α
    s t : LowerSet α
    ⊢ Iff (Disjoint ↑s ↑t) (Disjoint s t)
  -/
  simp [disjoint_iff, SetLike.ext'_iff]
  /-
    🎉 no goals
  -/


/-- The complement of a lower set as an upper set. -/
def UpperSet.compl (s : UpperSet α) : LowerSet α :=
  ⟨sᶜ, s.upper.compl⟩


/-- The complement of a lower set as an upper set. -/
def LowerSet.compl (s : LowerSet α) : UpperSet α :=
  ⟨sᶜ, s.lower.compl⟩


@[simp]
theorem coe_compl (s : UpperSet α) : (s.compl : Set α) = (↑s)ᶜ :=
  rfl


@[simp]
theorem mem_compl_iff : a ∈ s.compl ↔ a ∉ s :=
  Iff.rfl


@[simp]
nonrec theorem compl_compl (s : UpperSet α) : s.compl.compl = s :=
  UpperSet.ext <| compl_compl _


@[simp]
theorem compl_le_compl : s.compl ≤ t.compl ↔ s ≤ t :=
  compl_subset_compl


@[simp]
protected theorem compl_sup (s t : UpperSet α) : (s ⊔ t).compl = s.compl ⊔ t.compl :=
  LowerSet.ext compl_inf


@[simp]
protected theorem compl_inf (s t : UpperSet α) : (s ⊓ t).compl = s.compl ⊓ t.compl :=
  LowerSet.ext compl_sup


@[simp]
protected theorem compl_top : (⊤ : UpperSet α).compl = ⊤ :=
  LowerSet.ext compl_empty


@[simp]
protected theorem compl_bot : (⊥ : UpperSet α).compl = ⊥ :=
  LowerSet.ext compl_univ


@[simp]
protected theorem compl_sSup (S : Set (UpperSet α)) : (sSup S).compl = ⨆ s ∈ S, UpperSet.compl s :=
                     /-
                       α : Type u_1
                       inst✝ : LE α
                       S : Set (UpperSet α)
                       ⊢ Eq ↑(SupSet.sSup S).compl ↑(iSup fun s => iSup fun h => s.compl)
                     -/
  LowerSet.ext <| by simp only [coe_compl, coe_sSup, compl_iInter₂, LowerSet.coe_iSup₂]
                     /-
                       🎉 no goals
                     -/


@[simp]
protected theorem compl_sInf (S : Set (UpperSet α)) : (sInf S).compl = ⨅ s ∈ S, UpperSet.compl s :=
                     /-
                       α : Type u_1
                       inst✝ : LE α
                       S : Set (UpperSet α)
                       ⊢ Eq ↑(InfSet.sInf S).compl ↑(iInf fun s => iInf fun h => s.compl)
                     -/
  LowerSet.ext <| by simp only [coe_compl, coe_sInf, compl_iUnion₂, LowerSet.coe_iInf₂]
                     /-
                       🎉 no goals
                     -/


@[simp]
protected theorem compl_iSup (f : ι → UpperSet α) : (⨆ i, f i).compl = ⨆ i, (f i).compl :=
                     /-
                       α : Type u_1
                       ι : Sort u_4
                       inst✝ : LE α
                       f : ι → UpperSet α
                       ⊢ Eq ↑(iSup fun i => f i).compl ↑(iSup fun i => (f i).compl)
                     -/
  LowerSet.ext <| by simp only [coe_compl, coe_iSup, compl_iInter, LowerSet.coe_iSup]
                     /-
                       🎉 no goals
                     -/


@[simp]
protected theorem compl_iInf (f : ι → UpperSet α) : (⨅ i, f i).compl = ⨅ i, (f i).compl :=
                     /-
                       α : Type u_1
                       ι : Sort u_4
                       inst✝ : LE α
                       f : ι → UpperSet α
                       ⊢ Eq ↑(iInf fun i => f i).compl ↑(iInf fun i => (f i).compl)
                     -/
  LowerSet.ext <| by simp only [coe_compl, coe_iInf, compl_iUnion, LowerSet.coe_iInf]
                     /-
                       🎉 no goals
                     -/

-- Porting note: no longer a @[simp]

theorem compl_iSup₂ (f : ∀ i, κ i → UpperSet α) :
                                                              /-
                                                                α : Type u_1
                                                                ι : Sort u_4
                                                                κ : ι → Sort u_5
                                                                inst✝ : LE α
                                                                f : (i : ι) → κ i → UpperSet α
                                                                ⊢ Eq (iSup fun i => iSup fun j => f i j).compl (iSup fun i => iSup fun j => (f …
                                                              -/
    (⨆ (i) (j), f i j).compl = ⨆ (i) (j), (f i j).compl := by simp_rw [UpperSet.compl_iSup]
                                                              /-
                                                                🎉 no goals
                                                              -/

-- Porting note: no longer a @[simp]

theorem compl_iInf₂ (f : ∀ i, κ i → UpperSet α) :
                                                              /-
                                                                α : Type u_1
                                                                ι : Sort u_4
                                                                κ : ι → Sort u_5
                                                                inst✝ : LE α
                                                                f : (i : ι) → κ i → UpperSet α
                                                                ⊢ Eq (iInf fun i => iInf fun j => f i j).compl (iInf fun i => iInf fun j => (f …
                                                              -/
    (⨅ (i) (j), f i j).compl = ⨅ (i) (j), (f i j).compl := by simp_rw [UpperSet.compl_iInf]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem coe_compl (s : LowerSet α) : (s.compl : Set α) = (↑s)ᶜ :=
  rfl


@[simp]
nonrec theorem compl_compl (s : LowerSet α) : s.compl.compl = s :=
  LowerSet.ext <| compl_compl _


protected theorem compl_sup (s t : LowerSet α) : (s ⊔ t).compl = s.compl ⊔ t.compl :=
  UpperSet.ext compl_sup


protected theorem compl_inf (s t : LowerSet α) : (s ⊓ t).compl = s.compl ⊓ t.compl :=
  UpperSet.ext compl_inf


protected theorem compl_top : (⊤ : LowerSet α).compl = ⊤ :=
  UpperSet.ext compl_univ


protected theorem compl_bot : (⊥ : LowerSet α).compl = ⊥ :=
  UpperSet.ext compl_empty


protected theorem compl_sSup (S : Set (LowerSet α)) : (sSup S).compl = ⨆ s ∈ S, LowerSet.compl s :=
                     /-
                       α : Type u_1
                       inst✝ : LE α
                       S : Set (LowerSet α)
                       ⊢ Eq ↑(SupSet.sSup S).compl ↑(iSup fun s => iSup fun h => s.compl)
                     -/
  UpperSet.ext <| by simp only [coe_compl, coe_sSup, compl_iUnion₂, UpperSet.coe_iSup₂]
                     /-
                       🎉 no goals
                     -/


protected theorem compl_sInf (S : Set (LowerSet α)) : (sInf S).compl = ⨅ s ∈ S, LowerSet.compl s :=
                     /-
                       α : Type u_1
                       inst✝ : LE α
                       S : Set (LowerSet α)
                       ⊢ Eq ↑(InfSet.sInf S).compl ↑(iInf fun s => iInf fun h => s.compl)
                     -/
  UpperSet.ext <| by simp only [coe_compl, coe_sInf, compl_iInter₂, UpperSet.coe_iInf₂]
                     /-
                       🎉 no goals
                     -/


protected theorem compl_iSup (f : ι → LowerSet α) : (⨆ i, f i).compl = ⨆ i, (f i).compl :=
                     /-
                       α : Type u_1
                       ι : Sort u_4
                       inst✝ : LE α
                       f : ι → LowerSet α
                       ⊢ Eq ↑(iSup fun i => f i).compl ↑(iSup fun i => (f i).compl)
                     -/
  UpperSet.ext <| by simp only [coe_compl, coe_iSup, compl_iUnion, UpperSet.coe_iSup]
                     /-
                       🎉 no goals
                     -/


protected theorem compl_iInf (f : ι → LowerSet α) : (⨅ i, f i).compl = ⨅ i, (f i).compl :=
                     /-
                       α : Type u_1
                       ι : Sort u_4
                       inst✝ : LE α
                       f : ι → LowerSet α
                       ⊢ Eq ↑(iInf fun i => f i).compl ↑(iInf fun i => (f i).compl)
                     -/
  UpperSet.ext <| by simp only [coe_compl, coe_iInf, compl_iInter, UpperSet.coe_iInf]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem compl_iSup₂ (f : ∀ i, κ i → LowerSet α) :
                                                              /-
                                                                α : Type u_1
                                                                ι : Sort u_4
                                                                κ : ι → Sort u_5
                                                                inst✝ : LE α
                                                                f : (i : ι) → κ i → LowerSet α
                                                                ⊢ Eq (iSup fun i => iSup fun j => f i j).compl (iSup fun i => iSup fun j => (f …
                                                              -/
    (⨆ (i) (j), f i j).compl = ⨆ (i) (j), (f i j).compl := by simp_rw [LowerSet.compl_iSup]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem compl_iInf₂ (f : ∀ i, κ i → LowerSet α) :
                                                              /-
                                                                α : Type u_1
                                                                ι : Sort u_4
                                                                κ : ι → Sort u_5
                                                                inst✝ : LE α
                                                                f : (i : ι) → κ i → LowerSet α
                                                                ⊢ Eq (iInf fun i => iInf fun j => f i j).compl (iInf fun i => iInf fun j => (f …
                                                              -/
    (⨅ (i) (j), f i j).compl = ⨅ (i) (j), (f i j).compl := by simp_rw [LowerSet.compl_iInf]
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- Upper sets are order-isomorphic to lower sets under complementation. -/
@[simps]
def upperSetIsoLowerSet : UpperSet α ≃o LowerSet α where
  toFun := UpperSet.compl
  invFun := LowerSet.compl
  left_inv := UpperSet.compl_compl
  right_inv := LowerSet.compl_compl
  map_rel_iff' := UpperSet.compl_le_compl


instance UpperSet.isTotal_le : IsTotal (UpperSet α) (· ≤ ·) := ⟨fun s t => t.upper.total s.upper⟩


instance LowerSet.isTotal_le : IsTotal (LowerSet α) (· ≤ ·) := ⟨fun s t => s.lower.total t.lower⟩


noncomputable instance UpperSet.instLinearOrder : LinearOrder (UpperSet α) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Sort u_4
    κ : ι → Sort u_5
    inst✝ : LinearOrder α
    ⊢ LinearOrder (UpperSet α)
  -/
  classical exact Lattice.toLinearOrder _
  /-
    🎉 no goals
  -/


noncomputable instance LowerSet.instLinearOrder : LinearOrder (LowerSet α) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Sort u_4
    κ : ι → Sort u_5
    inst✝ : LinearOrder α
    ⊢ LinearOrder (LowerSet α)
  -/
  classical exact Lattice.toLinearOrder _
  /-
    🎉 no goals
  -/


noncomputable instance UpperSet.instCompleteLinearOrder : CompleteLinearOrder (UpperSet α) :=
  { completelyDistribLattice, instLinearOrder with }


noncomputable instance LowerSet.instCompleteLinearOrder : CompleteLinearOrder (LowerSet α) :=
  { completelyDistribLattice, instLinearOrder with }


/-- An order isomorphism of Preorders induces an order isomorphism of their upper sets. -/
def map (f : α ≃o β) : UpperSet α ≃o UpperSet β where
  toFun s := ⟨f '' s, s.upper.image f⟩
  invFun t := ⟨f ⁻¹' t, t.upper.preimage f.monotone⟩
  left_inv _ := ext <| f.preimage_image _
  right_inv _ := ext <| f.image_preimage _
  map_rel_iff' := image_subset_image_iff f.injective


@[simp]
theorem symm_map (f : α ≃o β) : (map f).symm = map f.symm :=
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        inst✝¹ : Preorder α
                                        inst✝ : Preorder β
                                        f : OrderIso α β
                                        s : UpperSet β
                                        ⊢ Eq ↑((UpperSet.map f).symm s) ↑((UpperSet.map f.symm) s)
                                      -/
  DFunLike.ext _ _ fun s => ext <| by convert Set.preimage_equiv_eq_image_symm s f.toEquiv
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem mem_map : b ∈ map f s ↔ f.symm b ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    s : UpperSet α
    b : β
    ⊢ Iff (Membership.mem ((UpperSet.map f) s) b) (Membership.mem s (f.symm b))
  -/
  rw [← f.symm_symm, ← symm_map, f.symm_symm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    s : UpperSet α
    b : β
    ⊢ Iff (Membership.mem ((UpperSet.map f.symm).symm s) b) (Membership.mem s (f.s …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem map_refl : map (OrderIso.refl α) = OrderIso.refl _ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Eq (UpperSet.map (OrderIso.refl α)) (OrderIso.refl (UpperSet α))
  -/
  ext
  /-
    case h.h.a.h
    α : Type u_1
    inst✝ : Preorder α
    x✝¹ : UpperSet α
    x✝ : α
    ⊢ Iff (Membership.mem (↑((UpperSet.map (OrderIso.refl α)) x✝¹)) x✝) (Membershi …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_map (g : β ≃o γ) (f : α ≃o β) : map g (map f s) = map (f.trans g) s := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    s : UpperSet α
    g : OrderIso β γ
    f : OrderIso α β
    ⊢ Eq ((UpperSet.map g) ((UpperSet.map f) s)) ((UpperSet.map (f.trans g)) s)
  -/
  ext
  /-
    case a.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    s : UpperSet α
    g : OrderIso β γ
    f : OrderIso α β
    x✝ : γ
    ⊢ Iff (Membership.mem (↑((UpperSet.map g) ((UpperSet.map f) s))) x✝) (Membersh …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_map : (map f s : Set β) = f '' s :=
  rfl


/-- An order isomorphism of Preorders induces an order isomorphism of their lower sets. -/
def map (f : α ≃o β) : LowerSet α ≃o LowerSet β where
  toFun s := ⟨f '' s, s.lower.image f⟩
  invFun t := ⟨f ⁻¹' t, t.lower.preimage f.monotone⟩
  left_inv _ := SetLike.coe_injective <| f.preimage_image _
  right_inv _ := SetLike.coe_injective <| f.image_preimage _
  map_rel_iff' := image_subset_image_iff f.injective


@[simp]
theorem symm_map (f : α ≃o β) : (map f).symm = map f.symm :=
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        inst✝¹ : Preorder α
                                        inst✝ : Preorder β
                                        f : OrderIso α β
                                        s : LowerSet β
                                        ⊢ Eq ↑((LowerSet.map f).symm s) ↑((LowerSet.map f.symm) s)
                                      -/
  DFunLike.ext _ _ fun s => ext <| by convert Set.preimage_equiv_eq_image_symm s f.toEquiv
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem mem_map {f : α ≃o β} {b : β} : b ∈ map f s ↔ f.symm b ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : LowerSet α
    f : OrderIso α β
    b : β
    ⊢ Iff (Membership.mem ((LowerSet.map f) s) b) (Membership.mem s (f.symm b))
  -/
  rw [← f.symm_symm, ← symm_map, f.symm_symm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : LowerSet α
    f : OrderIso α β
    b : β
    ⊢ Iff (Membership.mem ((LowerSet.map f.symm).symm s) b) (Membership.mem s (f.s …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem map_refl : map (OrderIso.refl α) = OrderIso.refl _ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Eq (LowerSet.map (OrderIso.refl α)) (OrderIso.refl (LowerSet α))
  -/
  ext
  /-
    case h.h.a.h
    α : Type u_1
    inst✝ : Preorder α
    x✝¹ : LowerSet α
    x✝ : α
    ⊢ Iff (Membership.mem (↑((LowerSet.map (OrderIso.refl α)) x✝¹)) x✝) (Membershi …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_map (g : β ≃o γ) (f : α ≃o β) : map g (map f s) = map (f.trans g) s := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    s : LowerSet α
    g : OrderIso β γ
    f : OrderIso α β
    ⊢ Eq ((LowerSet.map g) ((LowerSet.map f) s)) ((LowerSet.map (f.trans g)) s)
  -/
  ext
  /-
    case a.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    s : LowerSet α
    g : OrderIso β γ
    f : OrderIso α β
    x✝ : γ
    ⊢ Iff (Membership.mem (↑((LowerSet.map g) ((LowerSet.map f) s))) x✝) (Membersh …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_map (f : α ≃o β) (s : UpperSet α) : (map f s).compl = LowerSet.map f s.compl :=
  SetLike.coe_injective (Set.image_compl_eq f.bijective).symm


@[simp]
theorem compl_map (f : α ≃o β) (s : LowerSet α) : (map f s).compl = UpperSet.map f s.compl :=
  SetLike.coe_injective (Set.image_compl_eq f.bijective).symm


/-- The smallest upper set containing a given element. -/
nonrec def Ici (a : α) : UpperSet α :=
  ⟨Ici a, isUpperSet_Ici a⟩


/-- The smallest upper set containing a given element. -/
nonrec def Ioi (a : α) : UpperSet α :=
  ⟨Ioi a, isUpperSet_Ioi a⟩


@[simp]
theorem coe_Ici (a : α) : ↑(Ici a) = Set.Ici a :=
  rfl


@[simp]
theorem coe_Ioi (a : α) : ↑(Ioi a) = Set.Ioi a :=
  rfl


@[simp]
theorem mem_Ici_iff : b ∈ Ici a ↔ a ≤ b :=
  Iff.rfl


@[simp]
theorem mem_Ioi_iff : b ∈ Ioi a ↔ a < b :=
  Iff.rfl


@[simp]
theorem map_Ici (f : α ≃o β) (a : α) : map f (Ici a) = Ici (f a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    a : α
    ⊢ Eq ((UpperSet.map f) (UpperSet.Ici a)) (UpperSet.Ici (f a))
  -/
  ext
  /-
    case a.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    a : α
    x✝ : β
    ⊢ Iff (Membership.mem (↑((UpperSet.map f) (UpperSet.Ici a))) x✝) (Membership.m …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_Ioi (f : α ≃o β) (a : α) : map f (Ioi a) = Ioi (f a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    a : α
    ⊢ Eq ((UpperSet.map f) (UpperSet.Ioi a)) (UpperSet.Ioi (f a))
  -/
  ext
  /-
    case a.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    a : α
    x✝ : β
    ⊢ Iff (Membership.mem (↑((UpperSet.map f) (UpperSet.Ioi a))) x✝) (Membership.m …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Ici_le_Ioi (a : α) : Ici a ≤ Ioi a :=
  Ioi_subset_Ici_self


@[simp]
nonrec theorem Ici_bot [OrderBot α] : Ici (⊥ : α) = ⊥ :=
  SetLike.coe_injective Ici_bot


@[simp]
nonrec theorem Ioi_top [OrderTop α] : Ioi (⊤ : α) = ⊤ :=
  SetLike.coe_injective Ioi_top


@[simp] lemma Ici_ne_top : Ici a ≠ ⊤ := SetLike.coe_ne_coe.1 nonempty_Ici.ne_empty

@[simp] lemma Ici_lt_top : Ici a < ⊤ := lt_top_iff_ne_top.2 Ici_ne_top

@[simp] lemma le_Ici : s ≤ Ici a ↔ a ∈ s := ⟨fun h ↦ h le_rfl, fun ha ↦ s.upper.Ici_subset ha⟩


nonrec lemma Ici_injective : Injective (Ici : α → UpperSet α) := fun _a _b hab ↦
  Ici_injective <| congr_arg ((↑) : _ → Set α) hab


@[simp] lemma Ici_inj : Ici a = Ici b ↔ a = b := Ici_injective.eq_iff


lemma Ici_ne_Ici : Ici a ≠ Ici b ↔ a ≠ b := Ici_inj.not


@[simp]
theorem Ici_sup [SemilatticeSup α] (a b : α) : Ici (a ⊔ b) = Ici a ⊔ Ici b :=
  ext Ici_inter_Ici.symm


@[simp]
theorem Ici_sSup (S : Set α) : Ici (sSup S) = ⨆ a ∈ S, Ici a :=
                          /-
                            α : Type u_1
                            inst✝ : CompleteLattice α
                            S : Set α
                            c : α
                            ⊢ Iff (Membership.mem (UpperSet.Ici (SupSet.sSup S)) c) (Membership.mem (iSup  …
                          -/
  SetLike.ext fun c => by simp only [mem_Ici_iff, mem_iSup_iff, sSup_le_iff]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem Ici_iSup (f : ι → α) : Ici (⨆ i, f i) = ⨆ i, Ici (f i) :=
                          /-
                            α : Type u_1
                            ι : Sort u_4
                            inst✝ : CompleteLattice α
                            f : ι → α
                            c : α
                            ⊢ Iff (Membership.mem (UpperSet.Ici (iSup fun i => f i)) c) (Membership.mem (i …
                          -/
  SetLike.ext fun c => by simp only [mem_Ici_iff, mem_iSup_iff, iSup_le_iff]
                          /-
                            🎉 no goals
                          -/

-- Porting note: no longer a @[simp]

theorem Ici_iSup₂ (f : ∀ i, κ i → α) : Ici (⨆ (i) (j), f i j) = ⨆ (i) (j), Ici (f i j) := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_5
    inst✝ : CompleteLattice α
    f : (i : ι) → κ i → α
    ⊢ Eq (UpperSet.Ici (iSup fun i => iSup fun j => f i j)) (iSup fun i => iSup fu …
  -/
  simp_rw [Ici_iSup]
  /-
    🎉 no goals
  -/


/-- Principal lower set. `Set.Iic` as a lower set. The smallest lower set containing a given
element. -/
nonrec def Iic (a : α) : LowerSet α :=
  ⟨Iic a, isLowerSet_Iic a⟩


/-- Strict principal lower set. `Set.Iio` as a lower set. -/
nonrec def Iio (a : α) : LowerSet α :=
  ⟨Iio a, isLowerSet_Iio a⟩


@[simp]
theorem coe_Iic (a : α) : ↑(Iic a) = Set.Iic a :=
  rfl


@[simp]
theorem coe_Iio (a : α) : ↑(Iio a) = Set.Iio a :=
  rfl


@[simp]
theorem mem_Iic_iff : b ∈ Iic a ↔ b ≤ a :=
  Iff.rfl


@[simp]
theorem mem_Iio_iff : b ∈ Iio a ↔ b < a :=
  Iff.rfl


@[simp]
theorem map_Iic (f : α ≃o β) (a : α) : map f (Iic a) = Iic (f a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    a : α
    ⊢ Eq ((LowerSet.map f) (LowerSet.Iic a)) (LowerSet.Iic (f a))
  -/
  ext
  /-
    case a.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    a : α
    x✝ : β
    ⊢ Iff (Membership.mem (↑((LowerSet.map f) (LowerSet.Iic a))) x✝) (Membership.m …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_Iio (f : α ≃o β) (a : α) : map f (Iio a) = Iio (f a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    a : α
    ⊢ Eq ((LowerSet.map f) (LowerSet.Iio a)) (LowerSet.Iio (f a))
  -/
  ext
  /-
    case a.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    a : α
    x✝ : β
    ⊢ Iff (Membership.mem (↑((LowerSet.map f) (LowerSet.Iio a))) x✝) (Membership.m …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Ioi_le_Ici (a : α) : Ioi a ≤ Ici a :=
  Ioi_subset_Ici_self


@[simp]
nonrec theorem Iic_top [OrderTop α] : Iic (⊤ : α) = ⊤ :=
  SetLike.coe_injective Iic_top


@[simp]
nonrec theorem Iio_bot [OrderBot α] : Iio (⊥ : α) = ⊥ :=
  SetLike.coe_injective Iio_bot


@[simp] lemma Iic_ne_bot : Iic a ≠ ⊥ := SetLike.coe_ne_coe.1 nonempty_Iic.ne_empty

@[simp] lemma bot_lt_Iic : ⊥ < Iic a := bot_lt_iff_ne_bot.2 Iic_ne_bot

@[simp] lemma Iic_le : Iic a ≤ s ↔ a ∈ s := ⟨fun h ↦ h le_rfl, fun ha ↦ s.lower.Iic_subset ha⟩


nonrec lemma Iic_injective : Injective (Iic : α → LowerSet α) := fun _a _b hab ↦
  Iic_injective <| congr_arg ((↑) : _ → Set α) hab


@[simp] lemma Iic_inj : Iic a = Iic b ↔ a = b := Iic_injective.eq_iff


lemma Iic_ne_Iic : Iic a ≠ Iic b ↔ a ≠ b := Iic_inj.not


@[simp]
theorem Iic_inf [SemilatticeInf α] (a b : α) : Iic (a ⊓ b) = Iic a ⊓ Iic b :=
  SetLike.coe_injective Iic_inter_Iic.symm


@[simp]
theorem Iic_sInf (S : Set α) : Iic (sInf S) = ⨅ a ∈ S, Iic a :=
                          /-
                            α : Type u_1
                            inst✝ : CompleteLattice α
                            S : Set α
                            c : α
                            ⊢ Iff (Membership.mem (LowerSet.Iic (InfSet.sInf S)) c) (Membership.mem (iInf  …
                          -/
  SetLike.ext fun c => by simp only [mem_Iic_iff, mem_iInf₂_iff, le_sInf_iff]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem Iic_iInf (f : ι → α) : Iic (⨅ i, f i) = ⨅ i, Iic (f i) :=
                          /-
                            α : Type u_1
                            ι : Sort u_4
                            inst✝ : CompleteLattice α
                            f : ι → α
                            c : α
                            ⊢ Iff (Membership.mem (LowerSet.Iic (iInf fun i => f i)) c) (Membership.mem (i …
                          -/
  SetLike.ext fun c => by simp only [mem_Iic_iff, mem_iInf_iff, le_iInf_iff]
                          /-
                            🎉 no goals
                          -/

-- Porting note: no longer a @[simp]

theorem Iic_iInf₂ (f : ∀ i, κ i → α) : Iic (⨅ (i) (j), f i j) = ⨅ (i) (j), Iic (f i j) := by
  /-
    α : Type u_1
    ι : Sort u_4
    κ : ι → Sort u_5
    inst✝ : CompleteLattice α
    f : (i : ι) → κ i → α
    ⊢ Eq (LowerSet.Iic (iInf fun i => iInf fun j => f i j)) (iInf fun i => iInf fu …
  -/
  simp_rw [Iic_iInf]
  /-
    🎉 no goals
  -/


/-- The greatest upper set containing a given set. -/
def upperClosure (s : Set α) : UpperSet α :=
  ⟨{ x | ∃ a ∈ s, a ≤ x }, fun _ _ hle h => h.imp fun _x hx => ⟨hx.1, hx.2.trans hle⟩⟩


/-- The least lower set containing a given set. -/
def lowerClosure (s : Set α) : LowerSet α :=
  ⟨{ x | ∃ a ∈ s, x ≤ a }, fun _ _ hle h => h.imp fun _x hx => ⟨hx.1, hle.trans hx.2⟩⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: move `GaloisInsertion`s up, use them to prove lemmas


@[simp]
theorem mem_upperClosure : x ∈ upperClosure s ↔ ∃ a ∈ s, a ≤ x :=
  Iff.rfl


@[simp]
theorem mem_lowerClosure : x ∈ lowerClosure s ↔ ∃ a ∈ s, x ≤ a :=
  Iff.rfl

-- We do not tag those two as `simp` to respect the abstraction.

@[norm_cast]
theorem coe_upperClosure (s : Set α) : ↑(upperClosure s) = ⋃ a ∈ s, Ici a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Eq (↑(upperClosure s)) (Set.iUnion fun a => Set.iUnion fun h => Set.Ici a)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    x✝ : α
    ⊢ Iff (Membership.mem (↑(upperClosure s)) x✝) (Membership.mem (Set.iUnion fun  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem coe_lowerClosure (s : Set α) : ↑(lowerClosure s) = ⋃ a ∈ s, Iic a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Eq (↑(lowerClosure s)) (Set.iUnion fun a => Set.iUnion fun h => Set.Iic a)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    x✝ : α
    ⊢ Iff (Membership.mem (↑(lowerClosure s)) x✝) (Membership.mem (Set.iUnion fun  …
  -/
  simp
  /-
    🎉 no goals
  -/


instance instDecidablePredMemUpperClosure [DecidablePred (∃ a ∈ s, a ≤ ·)] :
    DecidablePred (· ∈ upperClosure s) := ‹DecidablePred _›


instance instDecidablePredMemLowerClosure [DecidablePred (∃ a ∈ s, · ≤ a)] :
    DecidablePred (· ∈ lowerClosure s) := ‹DecidablePred _›


theorem subset_upperClosure : s ⊆ upperClosure s := fun x hx => ⟨x, hx, le_rfl⟩


theorem subset_lowerClosure : s ⊆ lowerClosure s := fun x hx => ⟨x, hx, le_rfl⟩


theorem upperClosure_min (h : s ⊆ t) (ht : IsUpperSet t) : ↑(upperClosure s) ⊆ t :=
  fun _a ⟨_b, hb, hba⟩ => ht hba <| h hb


theorem lowerClosure_min (h : s ⊆ t) (ht : IsLowerSet t) : ↑(lowerClosure s) ⊆ t :=
  fun _a ⟨_b, hb, hab⟩ => ht hab <| h hb


protected theorem IsUpperSet.upperClosure (hs : IsUpperSet s) : ↑(upperClosure s) = s :=
  (upperClosure_min Subset.rfl hs).antisymm subset_upperClosure


protected theorem IsLowerSet.lowerClosure (hs : IsLowerSet s) : ↑(lowerClosure s) = s :=
  (lowerClosure_min Subset.rfl hs).antisymm subset_lowerClosure


@[simp]
protected theorem UpperSet.upperClosure (s : UpperSet α) : upperClosure (s : Set α) = s :=
  SetLike.coe_injective s.2.upperClosure


@[simp]
protected theorem LowerSet.lowerClosure (s : LowerSet α) : lowerClosure (s : Set α) = s :=
  SetLike.coe_injective s.2.lowerClosure


@[simp]
theorem upperClosure_image (f : α ≃o β) :
    upperClosure (f '' s) = UpperSet.map f (upperClosure s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    f : OrderIso α β
    ⊢ Eq (upperClosure (Set.image (⇑f) s)) ((UpperSet.map f) (upperClosure s))
  -/
  rw [← f.symm_symm, ← UpperSet.symm_map, f.symm_symm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    f : OrderIso α β
    ⊢ Eq (upperClosure (Set.image (⇑f) s)) ((UpperSet.map f.symm).symm (upperClosu …
  -/
  ext
  /-
    case a.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    f : OrderIso α β
    x✝ : β
    ⊢ Iff (Membership.mem (↑(upperClosure (Set.image (⇑f) s))) x✝) (Membership.mem …
  -/
  simp [-UpperSet.symm_map, UpperSet.map, OrderIso.symm, ← f.le_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem lowerClosure_image (f : α ≃o β) :
    lowerClosure (f '' s) = LowerSet.map f (lowerClosure s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    f : OrderIso α β
    ⊢ Eq (lowerClosure (Set.image (⇑f) s)) ((LowerSet.map f) (lowerClosure s))
  -/
  rw [← f.symm_symm, ← LowerSet.symm_map, f.symm_symm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    f : OrderIso α β
    ⊢ Eq (lowerClosure (Set.image (⇑f) s)) ((LowerSet.map f.symm).symm (lowerClosu …
  -/
  ext
  /-
    case a.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    f : OrderIso α β
    x✝ : β
    ⊢ Iff (Membership.mem (↑(lowerClosure (Set.image (⇑f) s))) x✝) (Membership.mem …
  -/
  simp [-LowerSet.symm_map, LowerSet.map, OrderIso.symm, ← f.symm_apply_le]
  /-
    🎉 no goals
  -/


@[simp]
theorem UpperSet.iInf_Ici (s : Set α) : ⨅ a ∈ s, UpperSet.Ici a = upperClosure s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Eq (iInf fun a => iInf fun h => UpperSet.Ici a) (upperClosure s)
  -/
  ext
  /-
    case a.h
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    x✝ : α
    ⊢ Iff (Membership.mem (↑(iInf fun a => iInf fun h => UpperSet.Ici a)) x✝) (Mem …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem LowerSet.iSup_Iic (s : Set α) : ⨆ a ∈ s, LowerSet.Iic a = lowerClosure s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Eq (iSup fun a => iSup fun h => LowerSet.Iic a) (lowerClosure s)
  -/
  ext
  /-
    case a.h
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    x✝ : α
    ⊢ Iff (Membership.mem (↑(iSup fun a => iSup fun h => LowerSet.Iic a)) x✝) (Mem …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp] lemma lowerClosure_le {t : LowerSet α} : lowerClosure s ≤ t ↔ s ⊆ t :=
  ⟨fun h ↦ subset_lowerClosure.trans <| LowerSet.coe_subset_coe.2 h,
    fun h ↦ lowerClosure_min h t.lower⟩


@[simp] lemma le_upperClosure {s : UpperSet α} : s ≤ upperClosure t ↔ t ⊆ s :=
  ⟨fun h ↦ subset_upperClosure.trans <| UpperSet.coe_subset_coe.2 h,
    fun h ↦ upperClosure_min h s.upper⟩


theorem gc_upperClosure_coe :
    GaloisConnection (toDual ∘ upperClosure : Set α → (UpperSet α)ᵒᵈ) ((↑) ∘ ofDual) :=
  fun _s _t ↦ le_upperClosure


theorem gc_lowerClosure_coe :
    GaloisConnection (lowerClosure : Set α → LowerSet α) (↑) := fun _s _t ↦ lowerClosure_le


/-- `upperClosure` forms a reversed Galois insertion with the coercion from upper sets to sets. -/
def giUpperClosureCoe :
    GaloisInsertion (toDual ∘ upperClosure : Set α → (UpperSet α)ᵒᵈ) ((↑) ∘ ofDual) where
  choice s hs := toDual (⟨s, fun a _b hab ha => hs ⟨a, ha, hab⟩⟩ : UpperSet α)
  gc := gc_upperClosure_coe
  le_l_u _ := subset_upperClosure
  choice_eq _s hs := ofDual.injective <| SetLike.coe_injective <| subset_upperClosure.antisymm hs


/-- `lowerClosure` forms a Galois insertion with the coercion from lower sets to sets. -/
def giLowerClosureCoe : GaloisInsertion (lowerClosure : Set α → LowerSet α) (↑) where
  choice s hs := ⟨s, fun a _b hba ha => hs ⟨a, ha, hba⟩⟩
  gc := gc_lowerClosure_coe
  le_l_u _ := subset_lowerClosure
  choice_eq _s hs := SetLike.coe_injective <| subset_lowerClosure.antisymm hs


theorem upperClosure_anti : Antitone (upperClosure : Set α → UpperSet α) :=
  gc_upperClosure_coe.monotone_l


theorem lowerClosure_mono : Monotone (lowerClosure : Set α → LowerSet α) :=
  gc_lowerClosure_coe.monotone_l


@[simp]
theorem upperClosure_empty : upperClosure (∅ : Set α) = ⊤ :=
  (@gc_upperClosure_coe α).l_bot


@[simp]
theorem lowerClosure_empty : lowerClosure (∅ : Set α) = ⊥ :=
  (@gc_lowerClosure_coe α).l_bot


@[simp]
theorem upperClosure_singleton (a : α) : upperClosure ({a} : Set α) = UpperSet.Ici a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ Eq (upperClosure (Singleton.singleton a)) (UpperSet.Ici a)
  -/
  ext
  /-
    case a.h
    α : Type u_1
    inst✝ : Preorder α
    a x✝ : α
    ⊢ Iff (Membership.mem (↑(upperClosure (Singleton.singleton a))) x✝) (Membershi …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem lowerClosure_singleton (a : α) : lowerClosure ({a} : Set α) = LowerSet.Iic a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ Eq (lowerClosure (Singleton.singleton a)) (LowerSet.Iic a)
  -/
  ext
  /-
    case a.h
    α : Type u_1
    inst✝ : Preorder α
    a x✝ : α
    ⊢ Iff (Membership.mem (↑(lowerClosure (Singleton.singleton a))) x✝) (Membershi …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem upperClosure_univ : upperClosure (univ : Set α) = ⊥ :=
  bot_unique subset_upperClosure


@[simp]
theorem lowerClosure_univ : lowerClosure (univ : Set α) = ⊤ :=
  top_unique subset_lowerClosure


@[simp]
theorem upperClosure_eq_top_iff : upperClosure s = ⊤ ↔ s = ∅ :=
  (@gc_upperClosure_coe α _).l_eq_bot.trans subset_empty_iff


@[simp]
theorem lowerClosure_eq_bot_iff : lowerClosure s = ⊥ ↔ s = ∅ :=
  (@gc_lowerClosure_coe α _).l_eq_bot.trans subset_empty_iff


@[simp]
theorem upperClosure_union (s t : Set α) : upperClosure (s ∪ t) = upperClosure s ⊓ upperClosure t :=
  (@gc_upperClosure_coe α _).l_sup


@[simp]
theorem lowerClosure_union (s t : Set α) : lowerClosure (s ∪ t) = lowerClosure s ⊔ lowerClosure t :=
  (@gc_lowerClosure_coe α _).l_sup


@[simp]
theorem upperClosure_iUnion (f : ι → Set α) : upperClosure (⋃ i, f i) = ⨅ i, upperClosure (f i) :=
  (@gc_upperClosure_coe α _).l_iSup


@[simp]
theorem lowerClosure_iUnion (f : ι → Set α) : lowerClosure (⋃ i, f i) = ⨆ i, lowerClosure (f i) :=
  (@gc_lowerClosure_coe α _).l_iSup


@[simp]
theorem upperClosure_sUnion (S : Set (Set α)) : upperClosure (⋃₀ S) = ⨅ s ∈ S, upperClosure s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    S : Set (Set α)
    ⊢ Eq (upperClosure S.sUnion) (iInf fun s => iInf fun h => upperClosure s)
  -/
  simp_rw [sUnion_eq_biUnion, upperClosure_iUnion]
  /-
    🎉 no goals
  -/


@[simp]
theorem lowerClosure_sUnion (S : Set (Set α)) : lowerClosure (⋃₀ S) = ⨆ s ∈ S, lowerClosure s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    S : Set (Set α)
    ⊢ Eq (lowerClosure S.sUnion) (iSup fun s => iSup fun h => lowerClosure s)
  -/
  simp_rw [sUnion_eq_biUnion, lowerClosure_iUnion]
  /-
    🎉 no goals
  -/


theorem Set.OrdConnected.upperClosure_inter_lowerClosure (h : s.OrdConnected) :
    ↑(upperClosure s) ∩ ↑(lowerClosure s) = s :=
  (subset_inter subset_upperClosure subset_lowerClosure).antisymm'
    fun _a ⟨⟨_b, hb, hba⟩, _c, hc, hac⟩ => h.out hb hc ⟨hba, hac⟩


theorem ordConnected_iff_upperClosure_inter_lowerClosure :
    s.OrdConnected ↔ ↑(upperClosure s) ∩ ↑(lowerClosure s) = s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Iff s.OrdConnected (Eq (Inter.inter ↑(upperClosure s) ↑(lowerClosure s)) s)
  -/
  refine ⟨Set.OrdConnected.upperClosure_inter_lowerClosure, fun h => ?_⟩
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    h : Eq (Inter.inter ↑(upperClosure s) ↑(lowerClosure s)) s
    ⊢ s.OrdConnected
  -/
  rw [← h]
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    h : Eq (Inter.inter ↑(upperClosure s) ↑(lowerClosure s)) s
    ⊢ (Inter.inter ↑(upperClosure s) ↑(lowerClosure s)).OrdConnected
  -/
  exact (UpperSet.upper _).ordConnected.inter (LowerSet.lower _).ordConnected
  /-
    🎉 no goals
  -/


@[simp]
theorem upperBounds_lowerClosure : upperBounds (lowerClosure s : Set α) = upperBounds s :=
  (upperBounds_mono_set subset_lowerClosure).antisymm
    fun _a ha _b ⟨_c, hc, hcb⟩ ↦ hcb.trans <| ha hc


@[simp]
theorem lowerBounds_upperClosure : lowerBounds (upperClosure s : Set α) = lowerBounds s :=
  (lowerBounds_mono_set subset_upperClosure).antisymm
    fun _a ha _b ⟨_c, hc, hcb⟩ ↦ (ha hc).trans hcb


@[simp]
theorem bddAbove_lowerClosure : BddAbove (lowerClosure s : Set α) ↔ BddAbove s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Iff (BddAbove ↑(lowerClosure s)) (BddAbove s)
  -/
  simp_rw [BddAbove, upperBounds_lowerClosure]
  /-
    🎉 no goals
  -/


@[simp]
theorem bddBelow_upperClosure : BddBelow (upperClosure s : Set α) ↔ BddBelow s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Iff (BddBelow ↑(upperClosure s)) (BddBelow s)
  -/
  simp_rw [BddBelow, lowerBounds_upperClosure]
  /-
    🎉 no goals
  -/


protected alias ⟨BddAbove.of_lowerClosure, BddAbove.lowerClosure⟩ := bddAbove_lowerClosure


protected alias ⟨BddBelow.of_upperClosure, BddBelow.upperClosure⟩ := bddBelow_upperClosure


@[simp] lemma IsLowerSet.disjoint_upperClosure_left (ht : IsLowerSet t) :
    Disjoint ↑(upperClosure s) t ↔ Disjoint s t := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s t : Set α
    ht : IsLowerSet t
    ⊢ Iff (Disjoint (↑(upperClosure s)) t) (Disjoint s t)
  -/
  refine ⟨Disjoint.mono_left subset_upperClosure, ?_⟩
  /-
    α : Type u_1
    inst✝ : Preorder α
    s t : Set α
    ht : IsLowerSet t
    ⊢ Disjoint s t → Disjoint (↑(upperClosure s)) t
  -/
  simp only [disjoint_left, SetLike.mem_coe, mem_upperClosure, forall_exists_index, and_imp]
  /-
    α : Type u_1
    inst✝ : Preorder α
    s t : Set α
    ht : IsLowerSet t
    ⊢ (∀ ⦃a : α⦄, Membership.mem s a → Not (Membership.mem t a)) → ∀ ⦃a : α⦄ (x :  …
  -/
  exact fun h a b hb hba ha ↦ h hb <| ht hba ha
  /-
    🎉 no goals
  -/


@[simp] lemma IsLowerSet.disjoint_upperClosure_right (hs : IsLowerSet s) :
    Disjoint s (upperClosure t) ↔ Disjoint s t := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s t : Set α
    hs : IsLowerSet s
    ⊢ Iff (Disjoint s ↑(upperClosure t)) (Disjoint s t)
  -/
  simpa only [disjoint_comm] using hs.disjoint_upperClosure_left
  /-
    🎉 no goals
  -/


@[simp] lemma IsUpperSet.disjoint_lowerClosure_left (ht : IsUpperSet t) :
    Disjoint ↑(lowerClosure s) t ↔ Disjoint s t := ht.toDual.disjoint_upperClosure_left


@[simp] lemma IsUpperSet.disjoint_lowerClosure_right (hs : IsUpperSet s) :
    Disjoint s (lowerClosure t) ↔ Disjoint s t := hs.toDual.disjoint_upperClosure_right


@[simp] lemma upperClosure_eq :
    ↑(upperClosure s) = s ↔ IsUpperSet s :=
  ⟨(· ▸ UpperSet.upper _), IsUpperSet.upperClosure⟩


@[simp] lemma lowerClosure_eq :
    ↑(lowerClosure s) = s ↔ IsLowerSet s :=
  @upperClosure_eq αᵒᵈ _ _


/-- The biggest lower subset of a lower set `s` disjoint from a set `t`. -/
def sdiff (s : LowerSet α) (t : Set α) : LowerSet α where
  carrier := s \ upperClosure t
  lower' := s.lower.sdiff_of_isUpperSet (upperClosure t).upper


/-- The biggest lower subset of a lower set `s` not containing an element `a`. -/
def erase (s : LowerSet α) (a : α) : LowerSet α where
  carrier := s \ UpperSet.Ici a
  lower' := s.lower.sdiff_of_isUpperSet (UpperSet.Ici a).upper


@[simp, norm_cast]
lemma coe_sdiff (s : LowerSet α) (t : Set α) : s.sdiff t = (s : Set α) \ upperClosure t := rfl


@[simp, norm_cast]
lemma coe_erase (s : LowerSet α) (a : α) : s.erase a = (s : Set α) \ UpperSet.Ici a := rfl


@[simp] lemma sdiff_singleton (s : LowerSet α) (a : α) : s.sdiff {a} = s.erase a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : LowerSet α
    a : α
    ⊢ Eq (s.sdiff (Singleton.singleton a)) (s.erase a)
  -/
  simp [sdiff, erase]
  /-
    🎉 no goals
  -/


lemma sdiff_le_left : s.sdiff t ≤ s := diff_subset

lemma erase_le : s.erase a ≤ s := diff_subset


@[simp] protected lemma sdiff_eq_left : s.sdiff t = s ↔ Disjoint ↑s t := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : LowerSet α
    t : Set α
    ⊢ Iff (Eq (s.sdiff t) s) (Disjoint (↑s) t)
  -/
  simp [← SetLike.coe_set_eq]
  /-
    🎉 no goals
  -/


                                                     /-
                                                       α : Type u_1
                                                       inst✝ : Preorder α
                                                       s : LowerSet α
                                                       a : α
                                                       ⊢ Iff (Eq (s.erase a) s) (Not (Membership.mem s a))
                                                     -/
@[simp] lemma erase_eq : s.erase a = s ↔ a ∉ s := by rw [← sdiff_singleton]; simp [-sdiff_singleton]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp] lemma sdiff_lt_left : s.sdiff t < s ↔ ¬ Disjoint ↑s t :=
  sdiff_le_left.lt_iff_ne.trans LowerSet.sdiff_eq_left.not


@[simp] lemma erase_lt : s.erase a < s ↔ a ∈ s := erase_le.lt_iff_ne.trans erase_eq.not_left


@[simp] protected lemma sdiff_idem (s : LowerSet α) (t : Set α) : (s.sdiff t).sdiff t = s.sdiff t :=
  SetLike.coe_injective sdiff_idem


@[simp] lemma erase_idem (s : LowerSet α) (a : α) : (s.erase a).erase a = s.erase a :=
  SetLike.coe_injective sdiff_idem


lemma sdiff_sup_lowerClosure (hts : t ⊆ s) (hst : ∀ b ∈ s, ∀ c ∈ t, c ≤ b → b ∈ t) :
    s.sdiff t ⊔ lowerClosure t = s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : LowerSet α
    t : Set α
    hts : HasSubset.Subset t ↑s
    hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le c  …
    ⊢ Eq (Max.max (s.sdiff t) (lowerClosure t)) s
  -/
  refine le_antisymm (sup_le sdiff_le_left <| lowerClosure_le.2 hts) fun a ha ↦ ?_
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : LowerSet α
    t : Set α
    hts : HasSubset.Subset t ↑s
    hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le c  …
    a : α
    ha : Membership.mem (↑s) a
    ⊢ Membership.mem (↑(Max.max (s.sdiff t) (lowerClosure t))) a
  -/
  obtain hat | hat := em (a ∈ t)
    /-
      case inl
      α : Type u_1
      inst✝ : Preorder α
      s : LowerSet α
      t : Set α
      hts : HasSubset.Subset t ↑s
      hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le c  …
      a : α
      ha : Membership.mem (↑s) a
      hat : Membership.mem t a
      ⊢ Membership.mem (↑(Max.max (s.sdiff t) (lowerClosure t))) a
    -/
  · exact subset_union_right (subset_lowerClosure hat)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : Preorder α
      s : LowerSet α
      t : Set α
      hts : HasSubset.Subset t ↑s
      hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le c  …
      a : α
      ha : Membership.mem (↑s) a
      hat : Not (Membership.mem t a)
      ⊢ Membership.mem (↑(Max.max (s.sdiff t) (lowerClosure t))) a
    -/
  · refine subset_union_left ⟨ha, ?_⟩
    /-
      case inr
      α : Type u_1
      inst✝ : Preorder α
      s : LowerSet α
      t : Set α
      hts : HasSubset.Subset t ↑s
      hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le c  …
      a : α
      ha : Membership.mem (↑s) a
      hat : Not (Membership.mem t a)
      ⊢ Not (Membership.mem (↑(upperClosure t)) a)
    -/
    rintro ⟨b, hb, hba⟩
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      s : LowerSet α
      t : Set α
      hts : HasSubset.Subset t ↑s
      hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le c  …
      a : α
      ha : Membership.mem (↑s) a
      hat : Not (Membership.mem t a)
      b : α
      hb : Membership.mem t b
      hba : LE.le b a
      ⊢ False
    -/
    exact hat <| hst _ ha _ hb hba
    /-
      🎉 no goals
    -/


lemma lowerClosure_sup_sdiff (hts : t ⊆ s) (hst : ∀ b ∈ s, ∀ c ∈ t, c ≤ b → b ∈ t) :
                                         /-
                                           α : Type u_1
                                           inst✝ : Preorder α
                                           s : LowerSet α
                                           t : Set α
                                           hts : HasSubset.Subset t ↑s
                                           hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le c  …
                                           ⊢ Eq (Max.max (lowerClosure t) (s.sdiff t)) s
                                         -/
    lowerClosure t ⊔ s.sdiff t = s := by rw [sup_comm, sdiff_sup_lowerClosure hts hst]
                                         /-
                                           🎉 no goals
                                         -/


lemma erase_sup_Iic (ha : a ∈ s) (has : ∀ b ∈ s, a ≤ b → b = a) : s.erase a ⊔ Iic a = s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : LowerSet α
    a : α
    ha : Membership.mem s a
    has : ∀ (b : α), Membership.mem s b → LE.le a b → Eq b a
    ⊢ Eq (Max.max (s.erase a) (LowerSet.Iic a)) s
  -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  rw [← lowerClosure_singleton, ← sdiff_singleton, sdiff_sup_lowerClosure] <;> simpa
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


lemma Iic_sup_erase (ha : a ∈ s) (has : ∀ b ∈ s, a ≤ b → b = a) : Iic a ⊔ s.erase a = s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : LowerSet α
    a : α
    ha : Membership.mem s a
    has : ∀ (b : α), Membership.mem s b → LE.le a b → Eq b a
    ⊢ Eq (Max.max (LowerSet.Iic a) (s.erase a)) s
  -/
  rw [sup_comm, erase_sup_Iic ha has]
  /-
    🎉 no goals
  -/


/-- The biggest upper subset of a upper set `s` disjoint from a set `t`. -/
def sdiff (s : UpperSet α) (t : Set α) : UpperSet α where
  carrier := s \ lowerClosure t
  upper' := s.upper.sdiff_of_isLowerSet (lowerClosure t).lower


/-- The biggest upper subset of a upper set `s` not containing an element `a`. -/
def erase (s : UpperSet α) (a : α) : UpperSet α where
  carrier := s \ LowerSet.Iic a
  upper' := s.upper.sdiff_of_isLowerSet (LowerSet.Iic a).lower


@[simp, norm_cast]
lemma coe_sdiff (s : UpperSet α) (t : Set α) : s.sdiff t = (s : Set α) \ lowerClosure t := rfl


@[simp, norm_cast]
lemma coe_erase (s : UpperSet α) (a : α) : s.erase a = (s : Set α) \ LowerSet.Iic a := rfl


@[simp] lemma sdiff_singleton (s : UpperSet α) (a : α) : s.sdiff {a} = s.erase a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : UpperSet α
    a : α
    ⊢ Eq (s.sdiff (Singleton.singleton a)) (s.erase a)
  -/
  simp [sdiff, erase]
  /-
    🎉 no goals
  -/


lemma le_sdiff_left : s ≤ s.sdiff t := diff_subset

lemma le_erase : s ≤ s.erase a := diff_subset


@[simp] protected lemma sdiff_eq_left : s.sdiff t = s ↔ Disjoint ↑s t := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : UpperSet α
    t : Set α
    ⊢ Iff (Eq (s.sdiff t) s) (Disjoint (↑s) t)
  -/
  simp [← SetLike.coe_set_eq]
  /-
    🎉 no goals
  -/


                                                     /-
                                                       α : Type u_1
                                                       inst✝ : Preorder α
                                                       s : UpperSet α
                                                       a : α
                                                       ⊢ Iff (Eq (s.erase a) s) (Not (Membership.mem s a))
                                                     -/
@[simp] lemma erase_eq : s.erase a = s ↔ a ∉ s := by rw [← sdiff_singleton]; simp [-sdiff_singleton]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp] lemma lt_sdiff_left : s < s.sdiff t ↔ ¬ Disjoint ↑s t :=
  le_sdiff_left.gt_iff_ne.trans UpperSet.sdiff_eq_left.not


@[simp] lemma lt_erase : s < s.erase a ↔ a ∈ s := le_erase.gt_iff_ne.trans erase_eq.not_left


@[simp] protected lemma sdiff_idem (s : UpperSet α) (t : Set α) : (s.sdiff t).sdiff t = s.sdiff t :=
  SetLike.coe_injective sdiff_idem


@[simp] lemma erase_idem (s : UpperSet α) (a : α) : (s.erase a).erase a = s.erase a :=
  SetLike.coe_injective sdiff_idem


lemma sdiff_inf_upperClosure (hts : t ⊆ s) (hst : ∀ b ∈ s, ∀ c ∈ t, b ≤ c → b ∈ t) :
    s.sdiff t ⊓ upperClosure t = s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : UpperSet α
    t : Set α
    hts : HasSubset.Subset t ↑s
    hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le b  …
    ⊢ Eq (Min.min (s.sdiff t) (upperClosure t)) s
  -/
  refine ge_antisymm (le_inf le_sdiff_left <| le_upperClosure.2 hts) fun a ha ↦ ?_
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : UpperSet α
    t : Set α
    hts : HasSubset.Subset t ↑s
    hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le b  …
    a : α
    ha : Membership.mem (Function.comp (⇑OrderDual.toDual) SetLike.coe s) a
    ⊢ Membership.mem (Function.comp (⇑OrderDual.toDual) SetLike.coe (Min.min (s.sd …
  -/
  obtain hat | hat := em (a ∈ t)
    /-
      case inl
      α : Type u_1
      inst✝ : Preorder α
      s : UpperSet α
      t : Set α
      hts : HasSubset.Subset t ↑s
      hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le b  …
      a : α
      ha : Membership.mem (Function.comp (⇑OrderDual.toDual) SetLike.coe s) a
      hat : Membership.mem t a
      ⊢ Membership.mem (Function.comp (⇑OrderDual.toDual) SetLike.coe (Min.min (s.sd …
    -/
  · exact subset_union_right (subset_upperClosure hat)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : Preorder α
      s : UpperSet α
      t : Set α
      hts : HasSubset.Subset t ↑s
      hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le b  …
      a : α
      ha : Membership.mem (Function.comp (⇑OrderDual.toDual) SetLike.coe s) a
      hat : Not (Membership.mem t a)
      ⊢ Membership.mem (Function.comp (⇑OrderDual.toDual) SetLike.coe (Min.min (s.sd …
    -/
  · refine subset_union_left ⟨ha, ?_⟩
    /-
      case inr
      α : Type u_1
      inst✝ : Preorder α
      s : UpperSet α
      t : Set α
      hts : HasSubset.Subset t ↑s
      hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le b  …
      a : α
      ha : Membership.mem (Function.comp (⇑OrderDual.toDual) SetLike.coe s) a
      hat : Not (Membership.mem t a)
      ⊢ Not (Membership.mem (↑(lowerClosure t)) a)
    -/
    rintro ⟨b, hb, hab⟩
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      s : UpperSet α
      t : Set α
      hts : HasSubset.Subset t ↑s
      hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le b  …
      a : α
      ha : Membership.mem (Function.comp (⇑OrderDual.toDual) SetLike.coe s) a
      hat : Not (Membership.mem t a)
      b : α
      hb : Membership.mem t b
      hab : LE.le a b
      ⊢ False
    -/
    exact hat <| hst _ ha _ hb hab
    /-
      🎉 no goals
    -/


lemma upperClosure_inf_sdiff (hts : t ⊆ s) (hst : ∀ b ∈ s, ∀ c ∈ t, b ≤ c → b ∈ t) :
                                         /-
                                           α : Type u_1
                                           inst✝ : Preorder α
                                           s : UpperSet α
                                           t : Set α
                                           hts : HasSubset.Subset t ↑s
                                           hst : ∀ (b : α), Membership.mem s b → ∀ (c : α), Membership.mem t c → LE.le b  …
                                           ⊢ Eq (Min.min (upperClosure t) (s.sdiff t)) s
                                         -/
    upperClosure t ⊓ s.sdiff t = s := by rw [inf_comm, sdiff_inf_upperClosure hts hst]
                                         /-
                                           🎉 no goals
                                         -/


lemma erase_inf_Ici (ha : a ∈ s) (has : ∀ b ∈ s, b ≤ a → b = a) : s.erase a ⊓ Ici a = s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : UpperSet α
    a : α
    ha : Membership.mem s a
    has : ∀ (b : α), Membership.mem s b → LE.le b a → Eq b a
    ⊢ Eq (Min.min (s.erase a) (UpperSet.Ici a)) s
  -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  rw [← upperClosure_singleton, ← sdiff_singleton, sdiff_inf_upperClosure] <;> simpa
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


lemma Ici_inf_erase (ha : a ∈ s) (has : ∀ b ∈ s, b ≤ a → b = a) : Ici a ⊓ s.erase a = s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : UpperSet α
    a : α
    ha : Membership.mem s a
    has : ∀ (b : α), Membership.mem s b → LE.le b a → Eq b a
    ⊢ Eq (Min.min (UpperSet.Ici a) (s.erase a)) s
  -/
  rw [inf_comm, erase_inf_Ici ha has]
  /-
    🎉 no goals
  -/


theorem IsUpperSet.prod (hs : IsUpperSet s) (ht : IsUpperSet t) : IsUpperSet (s ×ˢ t) :=
  fun _ _ h ha => ⟨hs h.1 ha.1, ht h.2 ha.2⟩


theorem IsLowerSet.prod (hs : IsLowerSet s) (ht : IsLowerSet t) : IsLowerSet (s ×ˢ t) :=
  fun _ _ h ha => ⟨hs h.1 ha.1, ht h.2 ha.2⟩


/-- The product of two upper sets as an upper set. -/
def prod : UpperSet (α × β) :=
  ⟨s ×ˢ t, s.2.prod t.2⟩


instance instSProd : SProd (UpperSet α) (UpperSet β) (UpperSet (α × β)) where
  sprod := UpperSet.prod


@[simp, norm_cast]
theorem coe_prod : ((s ×ˢ t : UpperSet (α × β)) : Set (α × β)) = (s : Set α) ×ˢ t :=
  rfl


@[simp]
theorem mem_prod {s : UpperSet α} {t : UpperSet β} : x ∈ s ×ˢ t ↔ x.1 ∈ s ∧ x.2 ∈ t :=
  Iff.rfl


theorem Ici_prod (x : α × β) : Ici x = Ici x.1 ×ˢ Ici x.2 :=
  rfl


@[simp]
theorem Ici_prod_Ici (a : α) (b : β) : Ici a ×ˢ Ici b = Ici (a, b) :=
  rfl


@[simp]
theorem prod_top : s ×ˢ (⊤ : UpperSet β) = ⊤ :=
  ext prod_empty


@[simp]
theorem top_prod : (⊤ : UpperSet α) ×ˢ t = ⊤ :=
  ext empty_prod


@[simp]
theorem bot_prod_bot : (⊥ : UpperSet α) ×ˢ (⊥ : UpperSet β) = ⊥ :=
  ext univ_prod_univ


@[simp]
theorem sup_prod : (s₁ ⊔ s₂) ×ˢ t = s₁ ×ˢ t ⊔ s₂ ×ˢ t :=
  ext inter_prod


@[simp]
theorem prod_sup : s ×ˢ (t₁ ⊔ t₂) = s ×ˢ t₁ ⊔ s ×ˢ t₂ :=
  ext prod_inter


@[simp]
theorem inf_prod : (s₁ ⊓ s₂) ×ˢ t = s₁ ×ˢ t ⊓ s₂ ×ˢ t :=
  ext union_prod


@[simp]
theorem prod_inf : s ×ˢ (t₁ ⊓ t₂) = s ×ˢ t₁ ⊓ s ×ˢ t₂ :=
  ext prod_union


theorem prod_sup_prod : s₁ ×ˢ t₁ ⊔ s₂ ×ˢ t₂ = (s₁ ⊔ s₂) ×ˢ (t₁ ⊔ t₂) :=
  ext prod_inter_prod


@[mono]
theorem prod_mono : s₁ ≤ s₂ → t₁ ≤ t₂ → s₁ ×ˢ t₁ ≤ s₂ ×ˢ t₂ :=
  Set.prod_mono


theorem prod_mono_left : s₁ ≤ s₂ → s₁ ×ˢ t ≤ s₂ ×ˢ t :=
  Set.prod_mono_left


theorem prod_mono_right : t₁ ≤ t₂ → s ×ˢ t₁ ≤ s ×ˢ t₂ :=
  Set.prod_mono_right


@[simp]
theorem prod_self_le_prod_self : s₁ ×ˢ s₁ ≤ s₂ ×ˢ s₂ ↔ s₁ ≤ s₂ :=
  prod_self_subset_prod_self


@[simp]
theorem prod_self_lt_prod_self : s₁ ×ˢ s₁ < s₂ ×ˢ s₂ ↔ s₁ < s₂ :=
  prod_self_ssubset_prod_self


theorem prod_le_prod_iff : s₁ ×ˢ t₁ ≤ s₂ ×ˢ t₂ ↔ s₁ ≤ s₂ ∧ t₁ ≤ t₂ ∨ s₂ = ⊤ ∨ t₂ = ⊤ :=
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     inst✝¹ : Preorder α
                                     inst✝ : Preorder β
                                     s₁ s₂ : UpperSet α
                                     t₁ t₂ : UpperSet β
                                     ⊢ Iff (Or (And (HasSubset.Subset ↑s₂ ↑s₁) (HasSubset.Subset ↑t₂ ↑t₁)) (Or (Eq  …
                                   -/
  prod_subset_prod_iff.trans <| by simp
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem prod_eq_top : s ×ˢ t = ⊤ ↔ s = ⊤ ∨ t = ⊤ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : UpperSet α
    t : UpperSet β
    ⊢ Iff (Eq (SProd.sprod s t) Top.top) (Or (Eq s Top.top) (Eq t Top.top))
  -/
  simp_rw [SetLike.ext'_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : UpperSet α
    t : UpperSet β
    ⊢ Iff (Eq ↑(SProd.sprod s t) ↑Top.top) (Or (Eq ↑s ↑Top.top) (Eq ↑t ↑Top.top))
  -/
  exact prod_eq_empty_iff
  /-
    🎉 no goals
  -/


@[simp]
theorem codisjoint_prod :
    Codisjoint (s₁ ×ˢ t₁) (s₂ ×ˢ t₂) ↔ Codisjoint s₁ s₂ ∨ Codisjoint t₁ t₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s₁ s₂ : UpperSet α
    t₁ t₂ : UpperSet β
    ⊢ Iff (Codisjoint (SProd.sprod s₁ t₁) (SProd.sprod s₂ t₂)) (Or (Codisjoint s₁  …
  -/
  simp_rw [codisjoint_iff, prod_sup_prod, prod_eq_top]
  /-
    🎉 no goals
  -/


/-- The product of two lower sets as a lower set. -/
def prod : LowerSet (α × β) := ⟨s ×ˢ t, s.2.prod t.2⟩


instance instSProd : SProd (LowerSet α) (LowerSet β) (LowerSet (α × β)) where
  sprod := LowerSet.prod


@[simp, norm_cast]
theorem coe_prod : ((s ×ˢ t : LowerSet (α × β)) : Set (α × β)) = (s : Set α) ×ˢ t := rfl


@[simp]
theorem mem_prod {s : LowerSet α} {t : LowerSet β} : x ∈ s ×ˢ t ↔ x.1 ∈ s ∧ x.2 ∈ t :=
  Iff.rfl


theorem Iic_prod (x : α × β) : Iic x = Iic x.1 ×ˢ Iic x.2 :=
  rfl


@[simp]
theorem Ici_prod_Ici (a : α) (b : β) : Iic a ×ˢ Iic b = Iic (a, b) :=
  rfl


@[simp]
theorem prod_bot : s ×ˢ (⊥ : LowerSet β) = ⊥ :=
  ext prod_empty


@[simp]
theorem bot_prod : (⊥ : LowerSet α) ×ˢ t = ⊥ :=
  ext empty_prod


@[simp]
theorem top_prod_top : (⊤ : LowerSet α) ×ˢ (⊤ : LowerSet β) = ⊤ :=
  ext univ_prod_univ


@[simp]
theorem inf_prod : (s₁ ⊓ s₂) ×ˢ t = s₁ ×ˢ t ⊓ s₂ ×ˢ t :=
  ext inter_prod


@[simp]
theorem prod_inf : s ×ˢ (t₁ ⊓ t₂) = s ×ˢ t₁ ⊓ s ×ˢ t₂ :=
  ext prod_inter


@[simp]
theorem sup_prod : (s₁ ⊔ s₂) ×ˢ t = s₁ ×ˢ t ⊔ s₂ ×ˢ t :=
  ext union_prod


@[simp]
theorem prod_sup : s ×ˢ (t₁ ⊔ t₂) = s ×ˢ t₁ ⊔ s ×ˢ t₂ :=
  ext prod_union


theorem prod_inf_prod : s₁ ×ˢ t₁ ⊓ s₂ ×ˢ t₂ = (s₁ ⊓ s₂) ×ˢ (t₁ ⊓ t₂) :=
  ext prod_inter_prod


theorem prod_mono : s₁ ≤ s₂ → t₁ ≤ t₂ → s₁ ×ˢ t₁ ≤ s₂ ×ˢ t₂ := Set.prod_mono


theorem prod_mono_left : s₁ ≤ s₂ → s₁ ×ˢ t ≤ s₂ ×ˢ t := Set.prod_mono_left


theorem prod_mono_right : t₁ ≤ t₂ → s ×ˢ t₁ ≤ s ×ˢ t₂ := Set.prod_mono_right


theorem prod_le_prod_iff : s₁ ×ˢ t₁ ≤ s₂ ×ˢ t₂ ↔ s₁ ≤ s₂ ∧ t₁ ≤ t₂ ∨ s₁ = ⊥ ∨ t₁ = ⊥ :=
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     inst✝¹ : Preorder α
                                     inst✝ : Preorder β
                                     s₁ s₂ : LowerSet α
                                     t₁ t₂ : LowerSet β
                                     ⊢ Iff (Or (And (HasSubset.Subset ↑s₁ ↑s₂) (HasSubset.Subset ↑t₁ ↑t₂)) (Or (Eq  …
                                   -/
  prod_subset_prod_iff.trans <| by simp
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem prod_eq_bot : s ×ˢ t = ⊥ ↔ s = ⊥ ∨ t = ⊥ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : LowerSet α
    t : LowerSet β
    ⊢ Iff (Eq (SProd.sprod s t) Bot.bot) (Or (Eq s Bot.bot) (Eq t Bot.bot))
  -/
  simp_rw [SetLike.ext'_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : LowerSet α
    t : LowerSet β
    ⊢ Iff (Eq ↑(SProd.sprod s t) ↑Bot.bot) (Or (Eq ↑s ↑Bot.bot) (Eq ↑t ↑Bot.bot))
  -/
  exact prod_eq_empty_iff
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_prod : Disjoint (s₁ ×ˢ t₁) (s₂ ×ˢ t₂) ↔ Disjoint s₁ s₂ ∨ Disjoint t₁ t₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s₁ s₂ : LowerSet α
    t₁ t₂ : LowerSet β
    ⊢ Iff (Disjoint (SProd.sprod s₁ t₁) (SProd.sprod s₂ t₂)) (Or (Disjoint s₁ s₂)  …
  -/
  simp_rw [disjoint_iff, prod_inf_prod, prod_eq_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem upperClosure_prod (s : Set α) (t : Set β) :
    upperClosure (s ×ˢ t) = upperClosure s ×ˢ upperClosure t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    t : Set β
    ⊢ Eq (upperClosure (SProd.sprod s t)) (SProd.sprod (upperClosure s) (upperClos …
  -/
  ext
  /-
    case a.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    t : Set β
    x✝ : Prod α β
    ⊢ Iff (Membership.mem (↑(upperClosure (SProd.sprod s t))) x✝) (Membership.mem  …
  -/
  simp [Prod.le_def, @and_and_and_comm _ (_ ∈ t)]
  /-
    🎉 no goals
  -/


@[simp]
theorem lowerClosure_prod (s : Set α) (t : Set β) :
    lowerClosure (s ×ˢ t) = lowerClosure s ×ˢ lowerClosure t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    t : Set β
    ⊢ Eq (lowerClosure (SProd.sprod s t)) (SProd.sprod (lowerClosure s) (lowerClos …
  -/
  ext
  /-
    case a.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    s : Set α
    t : Set β
    x✝ : Prod α β
    ⊢ Iff (Membership.mem (↑(lowerClosure (SProd.sprod s t))) x✝) (Membership.mem  …
  -/
  simp [Prod.le_def, @and_and_and_comm _ (_ ∈ t)]
  /-
    🎉 no goals
  -/


