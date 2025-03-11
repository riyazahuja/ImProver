instance instLocallyFiniteOrder : LocallyFiniteOrder (Multiset α) :=
  LocallyFiniteOrder.ofIcc (Multiset α)
    (fun s t => (Finset.Icc (toDFinsupp s) (toDFinsupp t)).map
      Multiset.equivDFinsupp.toEquiv.symm.toEmbedding)
                    /-
                      α : Type u_1
                      inst✝ : DecidableEq α
                      s✝ t✝ s t x : Multiset α
                      ⊢ Iff (Membership.mem ((fun s t => Finset.map Multiset.equivDFinsupp.symm.toEm …
                    -/
    fun s t x => by simp
                    /-
                      🎉 no goals
                    -/


theorem Icc_eq :
    Finset.Icc s t = (Finset.Icc (toDFinsupp s) (toDFinsupp t)).map
      Multiset.equivDFinsupp.toEquiv.symm.toEmbedding :=
  rfl


theorem uIcc_eq :
    uIcc s t =
      (uIcc (toDFinsupp s) (toDFinsupp t)).map Multiset.equivDFinsupp.toEquiv.symm.toEmbedding :=
                           /-
                             α : Type u_1
                             inst✝ : DecidableEq α
                             s t : Multiset α
                             ⊢ Eq (Finset.map Multiset.equivDFinsupp.symm.toEmbedding (Finset.Icc (Multiset …
                           -/
  (Icc_eq _ _).trans <| by simp [uIcc]
                           /-
                             🎉 no goals
                           -/


theorem card_Icc :
    #(Finset.Icc s t) = ∏ i ∈ s.toFinset ∪ t.toFinset, (t.count i + 1 - s.count i) := by
  simp_rw [Icc_eq, Finset.card_map, DFinsupp.card_Icc, Nat.card_Icc, Multiset.toDFinsupp_apply,
    toDFinsupp_support]


theorem card_Ico :
    #(Finset.Ico s t) = ∏ i ∈ s.toFinset ∪ t.toFinset, (t.count i + 1 - s.count i) - 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (Finset.Ico s t).card (HSub.hSub ((Union.union s.toFinset t.toFinset).pro …
  -/
  rw [Finset.card_Ico_eq_card_Icc_sub_one, card_Icc]
  /-
    🎉 no goals
  -/


theorem card_Ioc :
    #(Finset.Ioc s t) = ∏ i ∈ s.toFinset ∪ t.toFinset, (t.count i + 1 - s.count i) - 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (Finset.Ioc s t).card (HSub.hSub ((Union.union s.toFinset t.toFinset).pro …
  -/
  rw [Finset.card_Ioc_eq_card_Icc_sub_one, card_Icc]
  /-
    🎉 no goals
  -/


theorem card_Ioo :
    #(Finset.Ioo s t) = ∏ i ∈ s.toFinset ∪ t.toFinset, (t.count i + 1 - s.count i) - 2 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (Finset.Ioo s t).card (HSub.hSub ((Union.union s.toFinset t.toFinset).pro …
  -/
  rw [Finset.card_Ioo_eq_card_Icc_sub_two, card_Icc]
  /-
    🎉 no goals
  -/


theorem card_uIcc :
    (uIcc s t).card = ∏ i ∈ s.toFinset ∪ t.toFinset, ((t.count i - s.count i : ℤ).natAbs + 1) := by
  simp_rw [uIcc_eq, Finset.card_map, DFinsupp.card_uIcc, Nat.card_uIcc, Multiset.toDFinsupp_apply,
    toDFinsupp_support]


theorem card_Iic : (Finset.Iic s).card = ∏ i ∈ s.toFinset, (s.count i + 1) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Eq (Finset.Iic s).card (s.toFinset.prod fun i => HAdd.hAdd (Multiset.count i …
  -/
  simp_rw [Iic_eq_Icc, card_Icc, bot_eq_zero, toFinset_zero, empty_union, count_zero, tsub_zero]
  /-
    🎉 no goals
  -/


