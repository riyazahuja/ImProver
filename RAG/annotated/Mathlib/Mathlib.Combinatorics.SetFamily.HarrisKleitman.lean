theorem IsLowerSet.nonMemberSubfamily (h : IsLowerSet (𝒜 : Set (Finset α))) :
    IsLowerSet (𝒜.nonMemberSubfamily a : Set (Finset α)) := fun s t hts => by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h : IsLowerSet ↑𝒜
    s t : Finset α
    hts : LE.le t s
    ⊢ Membership.mem (↑(Finset.nonMemberSubfamily a 𝒜)) s → Membership.mem (↑(Fins …
  -/
  simp_rw [mem_coe, mem_nonMemberSubfamily]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h : IsLowerSet ↑𝒜
    s t : Finset α
    hts : LE.le t s
    ⊢ And (Membership.mem 𝒜 s) (Not (Membership.mem s a)) → And (Membership.mem 𝒜  …
  -/
  exact And.imp (h hts) (mt <| @hts _)
  /-
    🎉 no goals
  -/


theorem IsLowerSet.memberSubfamily (h : IsLowerSet (𝒜 : Set (Finset α))) :
    IsLowerSet (𝒜.memberSubfamily a : Set (Finset α)) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h : IsLowerSet ↑𝒜
    ⊢ IsLowerSet ↑(Finset.memberSubfamily a 𝒜)
  -/
  rintro s t hts
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h : IsLowerSet ↑𝒜
    s t : Finset α
    hts : LE.le t s
    ⊢ Membership.mem (↑(Finset.memberSubfamily a 𝒜)) s → Membership.mem (↑(Finset. …
  -/
  simp_rw [mem_coe, mem_memberSubfamily]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h : IsLowerSet ↑𝒜
    s t : Finset α
    hts : LE.le t s
    ⊢ And (Membership.mem 𝒜 (Insert.insert a s)) (Not (Membership.mem s a)) → And  …
  -/
  exact And.imp (h <| insert_subset_insert _ hts) (mt <| @hts _)
  /-
    🎉 no goals
  -/


theorem IsLowerSet.memberSubfamily_subset_nonMemberSubfamily (h : IsLowerSet (𝒜 : Set (Finset α))) :
    𝒜.memberSubfamily a ⊆ 𝒜.nonMemberSubfamily a := fun s => by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h : IsLowerSet ↑𝒜
    s : Finset α
    ⊢ Membership.mem (Finset.memberSubfamily a 𝒜) s → Membership.mem (Finset.nonMe …
  -/
  rw [mem_memberSubfamily, mem_nonMemberSubfamily]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    a : α
    h : IsLowerSet ↑𝒜
    s : Finset α
    ⊢ And (Membership.mem 𝒜 (Insert.insert a s)) (Not (Membership.mem s a)) → And  …
  -/
  exact And.imp_left (h <| subset_insert _ _)
  /-
    🎉 no goals
  -/


/-- **Harris-Kleitman inequality**: Any two lower sets of finsets correlate. -/
theorem IsLowerSet.le_card_inter_finset' (h𝒜 : IsLowerSet (𝒜 : Set (Finset α)))
    (hℬ : IsLowerSet (ℬ : Set (Finset α))) (h𝒜s : ∀ t ∈ 𝒜, t ⊆ s) (hℬs : ∀ t ∈ ℬ, t ⊆ s) :
    #𝒜 * #ℬ ≤ 2 ^ #s * #(𝒜 ∩ ℬ) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    h𝒜 : IsLowerSet ↑𝒜
    hℬ : IsLowerSet ↑ℬ
    h𝒜s : ∀ (t : Finset α), Membership.mem 𝒜 t → HasSubset.Subset t s
    hℬs : ∀ (t : Finset α), Membership.mem ℬ t → HasSubset.Subset t s
    ⊢ LE.le (HMul.hMul 𝒜.card ℬ.card) (HMul.hMul (HPow.hPow 2 s.card) (Inter.inter …
  -/
  induction' s using Finset.induction with a s hs ih generalizing 𝒜 ℬ
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 ℬ : Finset (Finset α)
      h𝒜 : IsLowerSet ↑𝒜
      hℬ : IsLowerSet ↑ℬ
      h𝒜s : ∀ (t : Finset α), Membership.mem 𝒜 t → HasSubset.Subset t EmptyCollectio …
      hℬs : ∀ (t : Finset α), Membership.mem ℬ t → HasSubset.Subset t EmptyCollectio …
      ⊢ LE.le (HMul.hMul 𝒜.card ℬ.card) (HMul.hMul (HPow.hPow 2 EmptyCollection.empt …
    -/
  · simp_rw [subset_empty, ← subset_singleton_iff', subset_singleton_iff] at h𝒜s hℬs
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 ℬ : Finset (Finset α)
      h𝒜 : IsLowerSet ↑𝒜
      hℬ : IsLowerSet ↑ℬ
      h𝒜s : Or (Eq 𝒜 EmptyCollection.emptyCollection) (Eq 𝒜 (Singleton.singleton Emp …
      hℬs : Or (Eq ℬ EmptyCollection.emptyCollection) (Eq ℬ (Singleton.singleton Emp …
      ⊢ LE.le (HMul.hMul 𝒜.card ℬ.card) (HMul.hMul (HPow.hPow 2 EmptyCollection.empt …
    -/
    obtain rfl | rfl := h𝒜s
      /-
        case empty.inl
        α : Type u_1
        inst✝ : DecidableEq α
        ℬ : Finset (Finset α)
        hℬ : IsLowerSet ↑ℬ
        hℬs : Or (Eq ℬ EmptyCollection.emptyCollection) (Eq ℬ (Singleton.singleton Emp …
        h𝒜 : IsLowerSet ↑EmptyCollection.emptyCollection
        ⊢ LE.le (HMul.hMul EmptyCollection.emptyCollection.card ℬ.card) (HMul.hMul (HP …
      -/
    · simp only [card_empty, zero_mul, empty_inter, mul_zero, le_refl]
      /-
        🎉 no goals
      -/
    /-
      case empty.inr
      α : Type u_1
      inst✝ : DecidableEq α
      ℬ : Finset (Finset α)
      hℬ : IsLowerSet ↑ℬ
      hℬs : Or (Eq ℬ EmptyCollection.emptyCollection) (Eq ℬ (Singleton.singleton Emp …
      h𝒜 : IsLowerSet ↑(Singleton.singleton EmptyCollection.emptyCollection)
      ⊢ LE.le (HMul.hMul (Singleton.singleton EmptyCollection.emptyCollection).card  …
    -/
    obtain rfl | rfl := hℬs
      /-
        case empty.inr.inl
        α : Type u_1
        inst✝ : DecidableEq α
        h𝒜 : IsLowerSet ↑(Singleton.singleton EmptyCollection.emptyCollection)
        hℬ : IsLowerSet ↑EmptyCollection.emptyCollection
        ⊢ LE.le (HMul.hMul (Singleton.singleton EmptyCollection.emptyCollection).card  …
      -/
    · simp only [card_empty, inter_empty, mul_zero, zero_mul, le_refl]
      /-
        🎉 no goals
      -/
    · simp only [card_empty, pow_zero, inter_singleton_of_mem, mem_singleton, card_singleton,
        le_refl]
  rw [card_insert_of_not_mem hs, ← card_memberSubfamily_add_card_nonMemberSubfamily a 𝒜, ←
    card_memberSubfamily_add_card_nonMemberSubfamily a ℬ, add_mul, mul_add, mul_add,
    add_comm (_ * _), add_add_add_comm]
  refine
    (add_le_add_right
          (mul_add_mul_le_mul_add_mul
              (card_le_card h𝒜.memberSubfamily_subset_nonMemberSubfamily) <|
            card_le_card hℬ.memberSubfamily_subset_nonMemberSubfamily)
          _).trans
      ?_
  /-
    case insert
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    hs : Not (Membership.mem s a)
    ih : ∀ {𝒜 ℬ : Finset (Finset α)}, IsLowerSet ↑𝒜 → IsLowerSet ↑ℬ → (∀ (t : Fins …
    𝒜 ℬ : Finset (Finset α)
    h𝒜 : IsLowerSet ↑𝒜
    hℬ : IsLowerSet ↑ℬ
    h𝒜s : ∀ (t : Finset α), Membership.mem 𝒜 t → HasSubset.Subset t (Insert.insert …
    hℬs : ∀ (t : Finset α), Membership.mem ℬ t → HasSubset.Subset t (Insert.insert …
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Finset.memberSubfamily a 𝒜).card (Fi …
  -/
  rw [← two_mul, pow_succ', mul_assoc]
  have h₀ : ∀ 𝒞 : Finset (Finset α), (∀ t ∈ 𝒞, t ⊆ insert a s) →
      ∀ t ∈ 𝒞.nonMemberSubfamily a, t ⊆ s := by
    rintro 𝒞 h𝒞 t ht
    rw [mem_nonMemberSubfamily] at ht
    exact (subset_insert_iff_of_not_mem ht.2).1 (h𝒞 _ ht.1)
  have h₁ : ∀ 𝒞 : Finset (Finset α), (∀ t ∈ 𝒞, t ⊆ insert a s) →
      ∀ t ∈ 𝒞.memberSubfamily a, t ⊆ s := by
    rintro 𝒞 h𝒞 t ht
    rw [mem_memberSubfamily] at ht
    exact (subset_insert_iff_of_not_mem ht.2).1 ((subset_insert _ _).trans <| h𝒞 _ ht.1)
  /-
    case insert
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    hs : Not (Membership.mem s a)
    ih : ∀ {𝒜 ℬ : Finset (Finset α)}, IsLowerSet ↑𝒜 → IsLowerSet ↑ℬ → (∀ (t : Fins …
    𝒜 ℬ : Finset (Finset α)
    h𝒜 : IsLowerSet ↑𝒜
    hℬ : IsLowerSet ↑ℬ
    h𝒜s : ∀ (t : Finset α), Membership.mem 𝒜 t → HasSubset.Subset t (Insert.insert …
    hℬs : ∀ (t : Finset α), Membership.mem ℬ t → HasSubset.Subset t (Insert.insert …
    h₀ : ∀ (𝒞 : Finset (Finset α)), (∀ (t : Finset α), Membership.mem 𝒞 t → HasSub …
    h₁ : ∀ (𝒞 : Finset (Finset α)), (∀ (t : Finset α), Membership.mem 𝒞 t → HasSub …
    ⊢ LE.le (HMul.hMul 2 (HAdd.hAdd (HMul.hMul (Finset.memberSubfamily a 𝒜).card ( …
  -/
  refine mul_le_mul_left' ?_ _
  refine (add_le_add (ih h𝒜.memberSubfamily hℬ.memberSubfamily (h₁ _ h𝒜s) <| h₁ _ hℬs) <|
    ih h𝒜.nonMemberSubfamily hℬ.nonMemberSubfamily (h₀ _ h𝒜s) <| h₀ _ hℬs).trans_eq ?_
  rw [← mul_add, ← memberSubfamily_inter, ← nonMemberSubfamily_inter,
    card_memberSubfamily_add_card_nonMemberSubfamily]


/-- **Harris-Kleitman inequality**: Any two lower sets of finsets correlate. -/
theorem IsLowerSet.le_card_inter_finset (h𝒜 : IsLowerSet (𝒜 : Set (Finset α)))
    (hℬ : IsLowerSet (ℬ : Set (Finset α))) : #𝒜 * #ℬ ≤ 2 ^ Fintype.card α * #(𝒜 ∩ ℬ) :=
h𝒜.le_card_inter_finset' hℬ (fun _ _ => subset_univ _) fun _ _ => subset_univ _


/-- **Harris-Kleitman inequality**: Upper sets and lower sets of finsets anticorrelate. -/
theorem IsUpperSet.card_inter_le_finset (h𝒜 : IsUpperSet (𝒜 : Set (Finset α)))
    (hℬ : IsLowerSet (ℬ : Set (Finset α))) :
    2 ^ Fintype.card α * #(𝒜 ∩ ℬ) ≤ #𝒜 * #ℬ := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    inst✝ : Fintype α
    h𝒜 : IsUpperSet ↑𝒜
    hℬ : IsLowerSet ↑ℬ
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (Inter.inter 𝒜 ℬ).card) (HMu …
  -/
  rw [← isLowerSet_compl, ← coe_compl] at h𝒜
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    inst✝ : Fintype α
    h𝒜 : IsLowerSet ↑(HasCompl.compl 𝒜)
    hℬ : IsLowerSet ↑ℬ
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (Inter.inter 𝒜 ℬ).card) (HMu …
  -/
  have := h𝒜.le_card_inter_finset hℬ
  rwa [card_compl, Fintype.card_finset, tsub_mul, tsub_le_iff_tsub_le, ← mul_tsub, ←
    card_sdiff inter_subset_right, sdiff_inter_self_right, sdiff_compl,
    _root_.inf_comm] at this


/-- **Harris-Kleitman inequality**: Lower sets and upper sets of finsets anticorrelate. -/
theorem IsLowerSet.card_inter_le_finset (h𝒜 : IsLowerSet (𝒜 : Set (Finset α)))
    (hℬ : IsUpperSet (ℬ : Set (Finset α))) :
    2 ^ Fintype.card α * #(𝒜 ∩ ℬ) ≤ #𝒜 * #ℬ := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    inst✝ : Fintype α
    h𝒜 : IsLowerSet ↑𝒜
    hℬ : IsUpperSet ↑ℬ
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (Inter.inter 𝒜 ℬ).card) (HMu …
  -/
  rw [inter_comm, mul_comm #𝒜]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    inst✝ : Fintype α
    h𝒜 : IsLowerSet ↑𝒜
    hℬ : IsUpperSet ↑ℬ
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (Inter.inter ℬ 𝒜).card) (HMu …
  -/
  exact hℬ.card_inter_le_finset h𝒜
  /-
    🎉 no goals
  -/


/-- **Harris-Kleitman inequality**: Any two upper sets of finsets correlate. -/
theorem IsUpperSet.le_card_inter_finset (h𝒜 : IsUpperSet (𝒜 : Set (Finset α)))
    (hℬ : IsUpperSet (ℬ : Set (Finset α))) :
    #𝒜 * #ℬ ≤ 2 ^ Fintype.card α * #(𝒜 ∩ ℬ) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    inst✝ : Fintype α
    h𝒜 : IsUpperSet ↑𝒜
    hℬ : IsUpperSet ↑ℬ
    ⊢ LE.le (HMul.hMul 𝒜.card ℬ.card) (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (I …
  -/
  rw [← isLowerSet_compl, ← coe_compl] at h𝒜
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    inst✝ : Fintype α
    h𝒜 : IsLowerSet ↑(HasCompl.compl 𝒜)
    hℬ : IsUpperSet ↑ℬ
    ⊢ LE.le (HMul.hMul 𝒜.card ℬ.card) (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (I …
  -/
  have := h𝒜.card_inter_le_finset hℬ
  rwa [card_compl, Fintype.card_finset, tsub_mul, le_tsub_iff_le_tsub, ← mul_tsub, ←
    card_sdiff inter_subset_right, sdiff_inter_self_right, sdiff_compl,
    _root_.inf_comm] at this
    /-
      case h₁
      α : Type u_1
      inst✝¹ : DecidableEq α
      𝒜 ℬ : Finset (Finset α)
      inst✝ : Fintype α
      h𝒜 : IsLowerSet ↑(HasCompl.compl 𝒜)
      hℬ : IsUpperSet ↑ℬ
      this : LE.le (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (Inter.inter (HasCompl. …
      ⊢ LE.le (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (Inter.inter (HasCompl.compl …
    -/
  · exact mul_le_mul_left' (card_le_card inter_subset_right) _
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      inst✝¹ : DecidableEq α
      𝒜 ℬ : Finset (Finset α)
      inst✝ : Fintype α
      h𝒜 : IsLowerSet ↑(HasCompl.compl 𝒜)
      hℬ : IsUpperSet ↑ℬ
      this : LE.le (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (Inter.inter (HasCompl. …
      ⊢ LE.le (HMul.hMul 𝒜.card ℬ.card) (HMul.hMul (HPow.hPow 2 (Fintype.card α)) ℬ. …
    -/
  · rw [← Fintype.card_finset]
    /-
      case h₂
      α : Type u_1
      inst✝¹ : DecidableEq α
      𝒜 ℬ : Finset (Finset α)
      inst✝ : Fintype α
      h𝒜 : IsLowerSet ↑(HasCompl.compl 𝒜)
      hℬ : IsUpperSet ↑ℬ
      this : LE.le (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (Inter.inter (HasCompl. …
      ⊢ LE.le (HMul.hMul 𝒜.card ℬ.card) (HMul.hMul (Fintype.card (Finset α)) ℬ.card)
    -/
    exact mul_le_mul_right' (card_le_univ _) _
    /-
      🎉 no goals
    -/

