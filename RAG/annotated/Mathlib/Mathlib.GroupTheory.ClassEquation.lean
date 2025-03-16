/-- Conjugacy classes form a partition of G, stated in terms of cardinality. -/
theorem sum_conjClasses_card_eq_card [Fintype <| ConjClasses G] [Fintype G]
    [∀ x : ConjClasses G, Fintype x.carrier] :
    ∑ x : ConjClasses G, x.carrier.toFinset.card = Fintype.card G := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Fintype (ConjClasses G)
    inst✝¹ : Fintype G
    inst✝ : (x : ConjClasses G) → Fintype ↑x.carrier
    ⊢ Eq (Finset.univ.sum fun x => x.carrier.toFinset.card) (Fintype.card G)
  -/
  suffices (Σ x : ConjClasses G, x.carrier) ≃ G by simpa using (Fintype.card_congr this)
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Fintype (ConjClasses G)
    inst✝¹ : Fintype G
    inst✝ : (x : ConjClasses G) → Fintype ↑x.carrier
    ⊢ Equiv (Sigma fun x => ↑x.carrier) G
  -/
  simpa [carrier_eq_preimage_mk] using Equiv.sigmaFiberEquiv ConjClasses.mk
  /-
    🎉 no goals
  -/


/-- Conjugacy classes form a partition of G, stated in terms of cardinality. -/
theorem Group.sum_card_conj_classes_eq_card [Finite G] :
    ∑ᶠ x : ConjClasses G, x.carrier.ncard = Nat.card G := by
  classical
  cases nonempty_fintype G
  rw [Nat.card_eq_fintype_card, ← sum_conjClasses_card_eq_card, finsum_eq_sum_of_fintype]
  simp [Set.ncard_eq_toFinset_card']


/-- The **class equation** for finite groups. The cardinality of a group is equal to the size
of its center plus the sum of the size of all its nontrivial conjugacy classes. -/
theorem Group.nat_card_center_add_sum_card_noncenter_eq_card [Finite G] :
    Nat.card (Subgroup.center G) + ∑ᶠ x ∈ noncenter G, Nat.card x.carrier = Nat.card G := by
  classical
  cases nonempty_fintype G
  rw [@Nat.card_eq_fintype_card G, ← sum_conjClasses_card_eq_card, ←
    Finset.sum_sdiff (ConjClasses.noncenter G).toFinset.subset_univ]
  simp only [Nat.card_eq_fintype_card, Set.toFinset_card]
  congr 1
  swap
  · convert finsum_cond_eq_sum_of_cond_iff _ _
    simp [Set.mem_toFinset]
  calc
    Fintype.card (Subgroup.center G) = Fintype.card ((noncenter G)ᶜ : Set _) :=
      Fintype.card_congr ((mk_bijOn G).equiv _)
    _ = Finset.card (Finset.univ \ (noncenter G).toFinset) := by
      rw [← Set.toFinset_card, Set.toFinset_compl, Finset.compl_eq_univ_sdiff]
    _ = _ := ?_
  rw [Finset.card_eq_sum_ones]
  refine Finset.sum_congr rfl ?_
  rintro ⟨g⟩ hg
  simp only [noncenter, Set.not_subsingleton_iff, Set.toFinset_setOf, Finset.mem_univ, true_and,
             forall_true_left, Finset.mem_sdiff, Finset.mem_filter, Set.not_nontrivial_iff] at hg
  rw [eq_comm, ← Set.toFinset_card, Finset.card_eq_one]
  exact ⟨g, Finset.coe_injective <| by simpa using hg.eq_singleton_of_mem mem_carrier_mk⟩


theorem Group.card_center_add_sum_card_noncenter_eq_card (G) [Group G]
    [∀ x : ConjClasses G, Fintype x.carrier] [Fintype G] [Fintype <| Subgroup.center G]
    [Fintype <| noncenter G] : Fintype.card (Subgroup.center G) +
  ∑ x ∈ (noncenter G).toFinset, x.carrier.toFinset.card = Fintype.card G := by
  /-
    G : Type u_2
    inst✝⁴ : Group G
    inst✝³ : (x : ConjClasses G) → Fintype ↑x.carrier
    inst✝² : Fintype G
    inst✝¹ : Fintype (Subtype fun x => Membership.mem (Subgroup.center G) x)
    inst✝ : Fintype ↑(ConjClasses.noncenter G)
    ⊢ Eq (HAdd.hAdd (Fintype.card (Subtype fun x => Membership.mem (Subgroup.cente …
  -/
  convert Group.nat_card_center_add_sum_card_noncenter_eq_card G using 2
    /-
      case h.e'_2.h.e'_5
      G : Type u_2
      inst✝⁴ : Group G
      inst✝³ : (x : ConjClasses G) → Fintype ↑x.carrier
      inst✝² : Fintype G
      inst✝¹ : Fintype (Subtype fun x => Membership.mem (Subgroup.center G) x)
      inst✝ : Fintype ↑(ConjClasses.noncenter G)
      ⊢ Eq (Fintype.card (Subtype fun x => Membership.mem (Subgroup.center G) x)) (N …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [← finsum_set_coe_eq_finsum_mem (noncenter G), finsum_eq_sum_of_fintype,
      ← Finset.sum_set_coe]
    /-
      case h.e'_2.h.e'_6
      G : Type u_2
      inst✝⁴ : Group G
      inst✝³ : (x : ConjClasses G) → Fintype ↑x.carrier
      inst✝² : Fintype G
      inst✝¹ : Fintype (Subtype fun x => Membership.mem (Subgroup.center G) x)
      inst✝ : Fintype ↑(ConjClasses.noncenter G)
      ⊢ Eq (Finset.univ.sum fun i => (↑i).carrier.toFinset.card) (Finset.univ.sum fu …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      G : Type u_2
      inst✝⁴ : Group G
      inst✝³ : (x : ConjClasses G) → Fintype ↑x.carrier
      inst✝² : Fintype G
      inst✝¹ : Fintype (Subtype fun x => Membership.mem (Subgroup.center G) x)
      inst✝ : Fintype ↑(ConjClasses.noncenter G)
      ⊢ Eq (Fintype.card G) (Nat.card G)
    -/
  · simp
    /-
      🎉 no goals
    -/

