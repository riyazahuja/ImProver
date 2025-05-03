instance instLocallyFiniteOrder : LocallyFiniteOrder (Finset α) where
  finsetIcc s t := t.powerset.filter (s ⊆ ·)
  finsetIco s t := t.ssubsets.filter (s ⊆ ·)
  finsetIoc s t := t.powerset.filter (s ⊂ ·)
  finsetIoo s t := t.ssubsets.filter (s ⊂ ·)
  finset_mem_Icc s t u := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq α
      s✝ t✝ s t u : Finset α
      ⊢ Iff (Membership.mem ((fun s t => Finset.filter (fun x => HasSubset.Subset s  …
    -/
    rw [mem_filter, mem_powerset]
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq α
      s✝ t✝ s t u : Finset α
      ⊢ Iff (And (HasSubset.Subset u t) (HasSubset.Subset s u)) (And (LE.le s u) (LE …
    -/
    exact and_comm
    /-
      🎉 no goals
    -/
  finset_mem_Ico s t u := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq α
      s✝ t✝ s t u : Finset α
      ⊢ Iff (Membership.mem ((fun s t => Finset.filter (fun x => HasSubset.Subset s  …
    -/
    rw [mem_filter, mem_ssubsets]
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq α
      s✝ t✝ s t u : Finset α
      ⊢ Iff (And (HasSSubset.SSubset u t) (HasSubset.Subset s u)) (And (LE.le s u) ( …
    -/
    exact and_comm
    /-
      🎉 no goals
    -/
  finset_mem_Ioc s t u := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq α
      s✝ t✝ s t u : Finset α
      ⊢ Iff (Membership.mem ((fun s t => Finset.filter (fun x => HasSSubset.SSubset  …
    -/
    rw [mem_filter, mem_powerset]
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq α
      s✝ t✝ s t u : Finset α
      ⊢ Iff (And (HasSubset.Subset u t) (HasSSubset.SSubset s u)) (And (LT.lt s u) ( …
    -/
    exact and_comm
    /-
      🎉 no goals
    -/
  finset_mem_Ioo s t u := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq α
      s✝ t✝ s t u : Finset α
      ⊢ Iff (Membership.mem ((fun s t => Finset.filter (fun x => HasSSubset.SSubset  …
    -/
    rw [mem_filter, mem_ssubsets]
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq α
      s✝ t✝ s t u : Finset α
      ⊢ Iff (And (HasSSubset.SSubset u t) (HasSSubset.SSubset s u)) (And (LT.lt s u) …
    -/
    exact and_comm
    /-
      🎉 no goals
    -/


theorem Icc_eq_filter_powerset : Icc s t = t.powerset.filter (s ⊆ ·) :=
  rfl


theorem Ico_eq_filter_ssubsets : Ico s t = t.ssubsets.filter (s ⊆ ·) :=
  rfl


theorem Ioc_eq_filter_powerset : Ioc s t = t.powerset.filter (s ⊂ ·) :=
  rfl


theorem Ioo_eq_filter_ssubsets : Ioo s t = t.ssubsets.filter (s ⊂ ·) :=
  rfl


theorem Iic_eq_powerset : Iic s = s.powerset :=
  filter_true_of_mem fun t _ => empty_subset t


theorem Iio_eq_ssubsets : Iio s = s.ssubsets :=
  filter_true_of_mem fun t _ => empty_subset t


theorem Icc_eq_image_powerset (h : s ⊆ t) : Icc s t = (t \ s).powerset.image (s ∪ ·) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    ⊢ Eq (Finset.Icc s t) (Finset.image (fun x => Union.union s x) (SDiff.sdiff t  …
  -/
  ext u
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    u : Finset α
    ⊢ Iff (Membership.mem (Finset.Icc s t) u) (Membership.mem (Finset.image (fun x …
  -/
  simp_rw [mem_Icc, mem_image, mem_powerset]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    u : Finset α
    ⊢ Iff (And (LE.le s u) (LE.le u t)) (Exists fun a => And (HasSubset.Subset a ( …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      h : HasSubset.Subset s t
      u : Finset α
      ⊢ And (LE.le s u) (LE.le u t) → Exists fun a => And (HasSubset.Subset a (SDiff …
    -/
  · rintro ⟨hs, ht⟩
    /-
      case h.mp.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      h : HasSubset.Subset s t
      u : Finset α
      hs : LE.le s u
      ht : LE.le u t
      ⊢ Exists fun a => And (HasSubset.Subset a (SDiff.sdiff t s)) (Eq (Union.union  …
    -/
    exact ⟨u \ s, sdiff_le_sdiff_right ht, sup_sdiff_cancel_right hs⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      h : HasSubset.Subset s t
      u : Finset α
      ⊢ (Exists fun a => And (HasSubset.Subset a (SDiff.sdiff t s)) (Eq (Union.union …
    -/
  · rintro ⟨v, hv, rfl⟩
    /-
      case h.mpr.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      h : HasSubset.Subset s t
      v : Finset α
      hv : HasSubset.Subset v (SDiff.sdiff t s)
      ⊢ And (LE.le s (Union.union s v)) (LE.le (Union.union s v) t)
    -/
    exact ⟨le_sup_left, union_subset h <| hv.trans sdiff_subset⟩
    /-
      🎉 no goals
    -/


theorem Ico_eq_image_ssubsets (h : s ⊆ t) : Ico s t = (t \ s).ssubsets.image (s ∪ ·) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    ⊢ Eq (Finset.Ico s t) (Finset.image (fun x => Union.union s x) (SDiff.sdiff t  …
  -/
  ext u
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    u : Finset α
    ⊢ Iff (Membership.mem (Finset.Ico s t) u) (Membership.mem (Finset.image (fun x …
  -/
  simp_rw [mem_Ico, mem_image, mem_ssubsets]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    u : Finset α
    ⊢ Iff (And (LE.le s u) (LT.lt u t)) (Exists fun a => And (HasSSubset.SSubset a …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      h : HasSubset.Subset s t
      u : Finset α
      ⊢ And (LE.le s u) (LT.lt u t) → Exists fun a => And (HasSSubset.SSubset a (SDi …
    -/
  · rintro ⟨hs, ht⟩
    /-
      case h.mp.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      h : HasSubset.Subset s t
      u : Finset α
      hs : LE.le s u
      ht : LT.lt u t
      ⊢ Exists fun a => And (HasSSubset.SSubset a (SDiff.sdiff t s)) (Eq (Union.unio …
    -/
    exact ⟨u \ s, sdiff_lt_sdiff_right ht hs, sup_sdiff_cancel_right hs⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      h : HasSubset.Subset s t
      u : Finset α
      ⊢ (Exists fun a => And (HasSSubset.SSubset a (SDiff.sdiff t s)) (Eq (Union.uni …
    -/
  · rintro ⟨v, hv, rfl⟩
    /-
      case h.mpr.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      h : HasSubset.Subset s t
      v : Finset α
      hv : HasSSubset.SSubset v (SDiff.sdiff t s)
      ⊢ And (LE.le s (Union.union s v)) (LT.lt (Union.union s v) t)
    -/
    exact ⟨le_sup_left, sup_lt_of_lt_sdiff_left hv h⟩
    /-
      🎉 no goals
    -/


/-- Cardinality of a non-empty `Icc` of finsets. -/
theorem card_Icc_finset (h : s ⊆ t) : (Icc s t).card = 2 ^ (t.card - s.card) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    ⊢ Eq (Finset.Icc s t).card (HPow.hPow 2 (HSub.hSub t.card s.card))
  -/
  rw [← card_sdiff h, ← card_powerset, Icc_eq_image_powerset h, Finset.card_image_iff]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    ⊢ Set.InjOn (fun x => Union.union s x) ↑(SDiff.sdiff t s).powerset
  -/
  rintro u hu v hv (huv : s ⊔ u = s ⊔ v)
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    u : Finset α
    hu : Membership.mem (↑(SDiff.sdiff t s).powerset) u
    v : Finset α
    hv : Membership.mem (↑(SDiff.sdiff t s).powerset) v
    huv : Eq (Max.max s u) (Max.max s v)
    ⊢ Eq u v
  -/
  rw [mem_coe, mem_powerset] at hu hv
  rw [← (disjoint_sdiff.mono_right hu : Disjoint s u).sup_sdiff_cancel_left, ←
    (disjoint_sdiff.mono_right hv : Disjoint s v).sup_sdiff_cancel_left, huv]


/-- Cardinality of an `Ico` of finsets. -/
theorem card_Ico_finset (h : s ⊆ t) : (Ico s t).card = 2 ^ (t.card - s.card) - 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    ⊢ Eq (Finset.Ico s t).card (HSub.hSub (HPow.hPow 2 (HSub.hSub t.card s.card)) 1)
  -/
  rw [card_Ico_eq_card_Icc_sub_one, card_Icc_finset h]
  /-
    🎉 no goals
  -/


/-- Cardinality of an `Ioc` of finsets. -/
theorem card_Ioc_finset (h : s ⊆ t) : (Ioc s t).card = 2 ^ (t.card - s.card) - 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    ⊢ Eq (Finset.Ioc s t).card (HSub.hSub (HPow.hPow 2 (HSub.hSub t.card s.card)) 1)
  -/
  rw [card_Ioc_eq_card_Icc_sub_one, card_Icc_finset h]
  /-
    🎉 no goals
  -/


/-- Cardinality of an `Ioo` of finsets. -/
theorem card_Ioo_finset (h : s ⊆ t) : (Ioo s t).card = 2 ^ (t.card - s.card) - 2 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    h : HasSubset.Subset s t
    ⊢ Eq (Finset.Ioo s t).card (HSub.hSub (HPow.hPow 2 (HSub.hSub t.card s.card)) 2)
  -/
  rw [card_Ioo_eq_card_Icc_sub_two, card_Icc_finset h]
  /-
    🎉 no goals
  -/


/-- Cardinality of an `Iic` of finsets. -/
                                                          /-
                                                            α : Type u_1
                                                            inst✝ : DecidableEq α
                                                            s : Finset α
                                                            ⊢ Eq (Finset.Iic s).card (HPow.hPow 2 s.card)
                                                          -/
theorem card_Iic_finset : (Iic s).card = 2 ^ s.card := by rw [Iic_eq_powerset, card_powerset]
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Cardinality of an `Iio` of finsets. -/
theorem card_Iio_finset : (Iio s).card = 2 ^ s.card - 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (Finset.Iio s).card (HSub.hSub (HPow.hPow 2 s.card) 1)
  -/
  rw [Iio_eq_ssubsets, ssubsets, card_erase_of_mem (mem_powerset_self _), card_powerset]
  /-
    🎉 no goals
  -/


/-- A function `f` from `Finset α` is monotone if and only if `f s ≤ f (cons a s ha)` for all `s`
and `a ∉ s`. -/
lemma monotone_iff_forall_le_cons : Monotone f ↔ ∀ s, ∀ ⦃a⦄ (ha), f s ≤ f (cons a s ha) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder β
    f : Finset α → β
    ⊢ Iff (Monotone f) (∀ (s : Finset α) ⦃a : α⦄ (ha : Not (Membership.mem s a)),  …
  -/
  classical simp [monotone_iff_forall_covBy, covBy_iff_exists_cons]
  /-
    🎉 no goals
  -/


/-- A function `f` from `Finset α` is antitone if and only if `f (cons a s ha) ≤ f s` for all
`s` and `a ∉ s`. -/
lemma antitone_iff_forall_cons_le : Antitone f ↔ ∀ s ⦃a⦄ ha, f (cons a s ha) ≤ f s :=
  monotone_iff_forall_le_cons (β := βᵒᵈ)


/-- A function `f` from `Finset α` is strictly monotone if and only if `f s < f (cons a s ha)` for
all `s` and `a ∉ s`. -/
lemma strictMono_iff_forall_lt_cons : StrictMono f ↔ ∀ s ⦃a⦄ ha, f s < f (cons a s ha) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder β
    f : Finset α → β
    ⊢ Iff (StrictMono f) (∀ (s : Finset α) ⦃a : α⦄ (ha : Not (Membership.mem s a)) …
  -/
  classical simp [strictMono_iff_forall_covBy, covBy_iff_exists_cons]
  /-
    🎉 no goals
  -/


/-- A function `f` from `Finset α` is strictly antitone if and only if `f (cons a s ha) < f s` for
all `s` and `a ∉ s`. -/
lemma strictAnti_iff_forall_cons_lt : StrictAnti f ↔ ∀ s ⦃a⦄ ha, f (cons a s ha) < f s :=
  strictMono_iff_forall_lt_cons (β := βᵒᵈ)


/-- A function `f` from `Finset α` is monotone if and only if `f s ≤ f (insert a s)` for all `s` and
`a ∉ s`. -/
lemma monotone_iff_forall_le_insert : Monotone f ↔ ∀ s ⦃a⦄, a ∉ s → f s ≤ f (insert a s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder β
    f : Finset α → β
    inst✝ : DecidableEq α
    ⊢ Iff (Monotone f) (∀ (s : Finset α) ⦃a : α⦄, Not (Membership.mem s a) → LE.le …
  -/
  simp [monotone_iff_forall_le_cons]
  /-
    🎉 no goals
  -/


/-- A function `f` from `Finset α` is antitone if and only if `f (insert a s) ≤ f s` for all
`s` and `a ∉ s`. -/
lemma antitone_iff_forall_insert_le : Antitone f ↔ ∀ s ⦃a⦄, a ∉ s → f (insert a s) ≤ f s :=
  monotone_iff_forall_le_insert (β := βᵒᵈ)


/-- A function `f` from `Finset α` is strictly monotone if and only if `f s < f (insert a s)` for
all `s` and `a ∉ s`. -/
lemma strictMono_iff_forall_lt_insert : StrictMono f ↔ ∀ s ⦃a⦄, a ∉ s → f s < f (insert a s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder β
    f : Finset α → β
    inst✝ : DecidableEq α
    ⊢ Iff (StrictMono f) (∀ (s : Finset α) ⦃a : α⦄, Not (Membership.mem s a) → LT. …
  -/
  simp [strictMono_iff_forall_lt_cons]
  /-
    🎉 no goals
  -/


/-- A function `f` from `Finset α` is strictly antitone if and only if `f (insert a s) < f s` for
all `s` and `a ∉ s`. -/
lemma strictAnti_iff_forall_lt_insert : StrictAnti f ↔ ∀ s ⦃a⦄, a ∉ s → f (insert a s) < f s :=
  strictMono_iff_forall_lt_insert (β := βᵒᵈ)


