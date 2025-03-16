@[simp] lemma covBy_cons (s : Multiset α) (a : α) : s ⋖ a ::ₘ s :=
  ⟨lt_cons_self _ _, fun t hst hts ↦ (covBy_succ _).2 (card_lt_card hst) <| by
    /-
      α : Type u_1
      s : Multiset α
      a : α
      t : Multiset α
      hst : LT.lt s t
      hts : LT.lt t (Multiset.cons a s)
      ⊢ LT.lt t.card (Order.succ s.card)
    -/
    simpa using card_lt_card hts⟩
    /-
      🎉 no goals
    -/


lemma _root_.CovBy.exists_multiset_cons (h : s ⋖ t) : ∃ a, a ::ₘ s = t :=
  (lt_iff_cons_le.1 h.lt).imp fun _a ha ↦ ha.eq_of_not_lt <| h.2 <| lt_cons_self _ _


lemma covBy_iff : s ⋖ t ↔ ∃ a, a ::ₘ s = t :=
                                  /-
                                    α : Type u_1
                                    s t : Multiset α
                                    ⊢ (Exists fun a => Eq (Multiset.cons a s) t) → CovBy s t
                                  -/
  ⟨CovBy.exists_multiset_cons, by rintro ⟨a, rfl⟩; exact covBy_cons _ _⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma _root_.CovBy.card_multiset (h : s ⋖ t) : card s ⋖ card t := by
  /-
    α : Type u_1
    s t : Multiset α
    h : CovBy s t
    ⊢ CovBy s.card t.card
  -/
  obtain ⟨a, rfl⟩ := h.exists_multiset_cons; rw [card_cons]; exact covBy_succ _
                                                             /-
                                                               🎉 no goals
                                                             -/


                                                 /-
                                                   α : Type u_1
                                                   s : Multiset α
                                                   ⊢ Iff (IsAtom s) (Exists fun a => Eq s (Singleton.singleton a))
                                                 -/
lemma isAtom_iff : IsAtom s ↔ ∃ a, s = {a} := by simp [← bot_covBy_iff, covBy_iff, eq_comm]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp] lemma isAtom_singleton (a : α) : IsAtom ({a} : Multiset α) := isAtom_iff.2 ⟨_, rfl⟩


instance instGradeMinOrder : GradeMinOrder ℕ (Multiset α) where
  grade := card
  grade_strictMono := card_strictMono
  covBy_grade _ _ := CovBy.card_multiset
                         /-
                           α : Type u_1
                           s✝ t : Multiset α
                           a : α
                           s : Multiset α
                           hs : IsMin s
                           ⊢ IsMin (GradeOrder.grade s)
                         -/
  isMin_grade s hs := by rw [isMin_iff_eq_bot.1 hs]; exact isMin_bot
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma grade_eq (m : Multiset α) : grade ℕ m = card m := rfl


/-- Finsets form an order-connected suborder of multisets. -/
lemma ordConnected_range_val : Set.OrdConnected (Set.range val : Set <| Multiset α) :=
      /-
        α : Type u_1
        ⊢ ∀ ⦃x : Multiset α⦄, Membership.mem (Set.range Finset.val) x → ∀ ⦃y : Multise …
      -/
  ⟨by rintro _ _ _ ⟨s, rfl⟩ t ht; exact ⟨⟨t, Multiset.nodup_of_le ht.2 s.2⟩, rfl⟩⟩
                                  /-
                                    🎉 no goals
                                  -/


/-- Finsets form an order-connected suborder of sets. -/
lemma ordConnected_range_coe : Set.OrdConnected (Set.range ((↑) : Finset α → Set α)) :=
      /-
        α : Type u_1
        ⊢ ∀ ⦃x : Set α⦄, Membership.mem (Set.range Finset.toSet) x → ∀ ⦃y : Set α⦄, Me …
      -/
  ⟨by rintro _ _ _ ⟨s, rfl⟩ t ht; exact ⟨_, (s.finite_toSet.subset ht.2).coe_toFinset⟩⟩
                                  /-
                                    🎉 no goals
                                  -/


@[simp] lemma val_wcovBy_val : s.1 ⩿ t.1 ↔ s ⩿ t :=
  ordConnected_range_val.apply_wcovBy_apply_iff ⟨⟨_, val_injective⟩, val_le_iff⟩


@[simp] lemma val_covBy_val : s.1 ⋖ t.1 ↔ s ⋖ t :=
  ordConnected_range_val.apply_covBy_apply_iff ⟨⟨_, val_injective⟩, val_le_iff⟩


@[simp] lemma coe_wcovBy_coe : (s : Set α) ⩿ t ↔ s ⩿ t :=
  ordConnected_range_coe.apply_wcovBy_apply_iff ⟨⟨_, coe_injective⟩, coe_subset⟩


@[simp] lemma coe_covBy_coe : (s : Set α) ⋖ t ↔ s ⋖ t :=
  ordConnected_range_coe.apply_covBy_apply_iff ⟨⟨_, coe_injective⟩, coe_subset⟩


alias ⟨_, _root_.WCovBy.finset_val⟩ := val_wcovBy_val

alias ⟨_, _root_.CovBy.finset_val⟩ := val_covBy_val

alias ⟨_, _root_.WCovBy.finset_coe⟩ := coe_wcovBy_coe

alias ⟨_, _root_.CovBy.finset_coe⟩ := coe_covBy_coe


                                                              /-
                                                                α : Type u_1
                                                                s : Finset α
                                                                a : α
                                                                ha : Not (Membership.mem s a)
                                                                ⊢ CovBy s (Finset.cons a s ha)
                                                              -/
@[simp] lemma covBy_cons (ha : a ∉ s) : s ⋖ s.cons a ha := by simp [← val_covBy_val]
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma _root_.CovBy.exists_finset_cons (h : s ⋖ t) : ∃ a, ∃ ha : a ∉ s, s.cons a ha = t :=
  let ⟨a, ha, hst⟩ := ssubset_iff_exists_cons_subset.1 h.lt
  ⟨a, ha, (hst.eq_of_not_ssuperset <| h.2 <| ssubset_cons _).symm⟩


lemma covBy_iff_exists_cons : s ⋖ t ↔ ∃ a, ∃ ha : a ∉ s, s.cons a ha = t :=
                                /-
                                  α : Type u_1
                                  s t : Finset α
                                  ⊢ (Exists fun a => Exists fun ha => Eq (Finset.cons a s ha) t) → CovBy s t
                                -/
  ⟨CovBy.exists_finset_cons, by rintro ⟨a, ha, rfl⟩; exact covBy_cons _⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma _root_.CovBy.card_finset (h : s ⋖ t) : s.card ⋖ t.card := (val_covBy_val.2 h).card_multiset


                                                                          /-
                                                                            α : Type u_1
                                                                            inst✝ : DecidableEq α
                                                                            s : Finset α
                                                                            a : α
                                                                            ⊢ WCovBy s (Insert.insert a s)
                                                                          -/
@[simp] lemma wcovBy_insert (s : Finset α) (a : α) : s ⩿ insert a s := by simp [← coe_wcovBy_coe]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

                                                                        /-
                                                                          α : Type u_1
                                                                          inst✝ : DecidableEq α
                                                                          s : Finset α
                                                                          a : α
                                                                          ⊢ WCovBy (s.erase a) s
                                                                        -/
@[simp] lemma erase_wcovBy (s : Finset α) (a : α) : s.erase a ⩿ s := by simp [← coe_wcovBy_coe]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


lemma covBy_insert (ha : a ∉ s) : s ⋖ insert a s :=
  (wcovBy_insert _ _).covBy_of_lt <| ssubset_insert ha


@[simp] lemma erase_covBy (ha : a ∈ s) : s.erase a ⋖ s := ⟨erase_ssubset ha, (erase_wcovBy _ _).2⟩


lemma _root_.CovBy.exists_finset_insert (h : s ⋖ t) : ∃ a ∉ s, insert a s = t := by
  /-
    α : Type u_1
    s t : Finset α
    inst✝ : DecidableEq α
    h : CovBy s t
    ⊢ Exists fun a => And (Not (Membership.mem s a)) (Eq (Insert.insert a s) t)
  -/
  simpa using h.exists_finset_cons
  /-
    🎉 no goals
  -/


lemma _root_.CovBy.exists_finset_erase (h : s ⋖ t) : ∃ a ∈ t, t.erase a = s := by
  /-
    α : Type u_1
    s t : Finset α
    inst✝ : DecidableEq α
    h : CovBy s t
    ⊢ Exists fun a => And (Membership.mem t a) (Eq (t.erase a) s)
  -/
  simpa only [← coe_inj, coe_erase] using h.finset_coe.exists_set_sdiff_singleton
  /-
    🎉 no goals
  -/


lemma covBy_iff_exists_insert : s ⋖ t ↔ ∃ a ∉ s, insert a s = t := by
  /-
    α : Type u_1
    s t : Finset α
    inst✝ : DecidableEq α
    ⊢ Iff (CovBy s t) (Exists fun a => And (Not (Membership.mem s a)) (Eq (Insert. …
  -/
  simp only [← coe_covBy_coe, Set.covBy_iff_exists_insert, ← coe_inj, coe_insert, mem_coe]
  /-
    🎉 no goals
  -/


lemma covBy_iff_card_sdiff_eq_one : t ⋖ s ↔ t ⊆ s ∧ (s \ t).card = 1 := by
  /-
    α : Type u_1
    s t : Finset α
    inst✝ : DecidableEq α
    ⊢ Iff (CovBy t s) (And (HasSubset.Subset t s) (Eq (SDiff.sdiff s t).card 1))
  -/
  rw [covBy_iff_exists_insert]
  /-
    α : Type u_1
    s t : Finset α
    inst✝ : DecidableEq α
    ⊢ Iff (Exists fun a => And (Not (Membership.mem t a)) (Eq (Insert.insert a t)  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      s t : Finset α
      inst✝ : DecidableEq α
      ⊢ (Exists fun a => And (Not (Membership.mem t a)) (Eq (Insert.insert a t) s))  …
    -/
  · rintro ⟨a, ha, rfl⟩
    /-
      case mp.intro.intro
      α : Type u_1
      t : Finset α
      inst✝ : DecidableEq α
      a : α
      ha : Not (Membership.mem t a)
      ⊢ And (HasSubset.Subset t (Insert.insert a t)) (Eq (SDiff.sdiff (Insert.insert …
    -/
    simp [*]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      s t : Finset α
      inst✝ : DecidableEq α
      ⊢ And (HasSubset.Subset t s) (Eq (SDiff.sdiff s t).card 1) → Exists fun a => A …
    -/
  · simp_rw [card_eq_one]
    /-
      case mpr
      α : Type u_1
      s t : Finset α
      inst✝ : DecidableEq α
      ⊢ And (HasSubset.Subset t s) (Exists fun a => Eq (SDiff.sdiff s t) (Singleton. …
    -/
    rintro ⟨hts, a, ha⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      s t : Finset α
      inst✝ : DecidableEq α
      hts : HasSubset.Subset t s
      a : α
      ha : Eq (SDiff.sdiff s t) (Singleton.singleton a)
      ⊢ Exists fun a => And (Not (Membership.mem t a)) (Eq (Insert.insert a t) s)
    -/
    refine ⟨a, (mem_sdiff.1 <| superset_of_eq ha <| mem_singleton_self _).2, ?_⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      s t : Finset α
      inst✝ : DecidableEq α
      hts : HasSubset.Subset t s
      a : α
      ha : Eq (SDiff.sdiff s t) (Singleton.singleton a)
      ⊢ Eq (Insert.insert a t) s
    -/
    rw [insert_eq, ← ha, sdiff_union_of_subset hts]
    /-
      🎉 no goals
    -/


lemma covBy_iff_exists_erase : s ⋖ t ↔ ∃ a ∈ t, t.erase a = s := by
  /-
    α : Type u_1
    s t : Finset α
    inst✝ : DecidableEq α
    ⊢ Iff (CovBy s t) (Exists fun a => And (Membership.mem t a) (Eq (t.erase a) s))
  -/
  simp only [← coe_covBy_coe, Set.covBy_iff_exists_sdiff_singleton, ← coe_inj, coe_erase, mem_coe]
  /-
    🎉 no goals
  -/


@[simp] lemma isAtom_singleton (a : α) : IsAtom ({a} : Finset α) :=
  ⟨singleton_ne_empty a, fun _ ↦ eq_empty_of_ssubset_singleton⟩


protected lemma isAtom_iff : IsAtom s ↔ ∃ a, s = {a} := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (IsAtom s) (Exists fun a => Eq s (Singleton.singleton a))
  -/
  simp [← bot_covBy_iff, covBy_iff_exists_cons, eq_comm]
  /-
    🎉 no goals
  -/


lemma isCoatom_compl_singleton (a : α) : IsCoatom ({a}ᶜ : Finset α) := (isAtom_singleton a).compl


protected lemma isCoatom_iff : IsCoatom s ↔ ∃ a, s = {a}ᶜ := by
  /-
    α : Type u_1
    s : Finset α
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    ⊢ Iff (IsCoatom s) (Exists fun a => Eq s (HasCompl.compl (Singleton.singleton  …
  -/
  simp_rw [← isAtom_compl, Finset.isAtom_iff, compl_eq_iff_isCompl, eq_compl_iff_isCompl]
  /-
    🎉 no goals
  -/


/-- Finsets are multiset-graded. This is not very meaningful mathematically but rather a handy way
to record that the inclusion `Finset α ↪ Multiset α` preserves the covering relation. -/
instance instGradeMinOrder_multiset : GradeMinOrder (Multiset α) (Finset α) where
  grade := val
  grade_strictMono := val_strictMono
  covBy_grade _ _ := CovBy.finset_val
                         /-
                           α : Type u_1
                           s✝ t : Finset α
                           a : α
                           s : Finset α
                           hs : IsMin s
                           ⊢ IsMin (GradeOrder.grade s)
                         -/
  isMin_grade s hs := by rw [isMin_iff_eq_bot.1 hs]; exact isMin_bot
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma grade_multiset_eq (s : Finset α) : grade (Multiset α) s = s.1 := rfl


instance instGradeMinOrder_nat : GradeMinOrder ℕ (Finset α) where
  grade := card
  grade_strictMono := card_strictMono
  covBy_grade _ _ := CovBy.card_finset
                         /-
                           α : Type u_1
                           s✝ t : Finset α
                           a : α
                           s : Finset α
                           hs : IsMin s
                           ⊢ IsMin (GradeOrder.grade s)
                         -/
  isMin_grade s hs := by rw [isMin_iff_eq_bot.1 hs]; exact isMin_bot
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma grade_eq (s : Finset α) : grade ℕ s = s.card := rfl


