/-- A set family `𝒜` shatters a set `s` if all subsets of `s` can be obtained as the intersection
of `s` and some element of the set family, and we denote this `𝒜.Shatters s`. We also say that `s`
is *traced* by `𝒜`. -/
def Shatters (𝒜 : Finset (Finset α)) (s : Finset α) : Prop := ∀ ⦃t⦄, t ⊆ s → ∃ u ∈ 𝒜, s ∩ u = t


instance : DecidablePred 𝒜.Shatters := fun _s ↦ decidableForallOfDecidableSubsets


lemma Shatters.exists_inter_eq_singleton (hs : Shatters 𝒜 s) (ha : a ∈ s) : ∃ t ∈ 𝒜, s ∩ t = {a} :=
  hs <| singleton_subset_iff.2 ha


lemma Shatters.mono_left (h : 𝒜 ⊆ ℬ) (h𝒜 : 𝒜.Shatters s) : ℬ.Shatters s :=
  fun _t ht ↦ let ⟨u, hu, hut⟩ := h𝒜 ht; ⟨u, h hu, hut⟩


lemma Shatters.mono_right (h : t ⊆ s) (hs : 𝒜.Shatters s) : 𝒜.Shatters t := fun u hu ↦ by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s t : Finset α
    h : HasSubset.Subset t s
    hs : 𝒜.Shatters s
    u : Finset α
    hu : HasSubset.Subset u t
    ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter t u_1) u)
  -/
  obtain ⟨v, hv, rfl⟩ := hs (hu.trans h); exact ⟨v, hv, inf_congr_right hu <| inf_le_of_left_le h⟩
                                          /-
                                            🎉 no goals
                                          -/


lemma Shatters.exists_superset (h : 𝒜.Shatters s) : ∃ t ∈ 𝒜, s ⊆ t :=
  let ⟨t, ht, hst⟩ := h Subset.rfl; ⟨t, ht, inter_eq_left.1 hst⟩


lemma shatters_of_forall_subset (h : ∀ t, t ⊆ s → t ∈ 𝒜) : 𝒜.Shatters s :=
  fun t ht ↦ ⟨t, h _ ht, inter_eq_right.2 ht⟩


protected lemma Shatters.nonempty (h : 𝒜.Shatters s) : 𝒜.Nonempty :=
  let ⟨t, ht, _⟩ := h Subset.rfl; ⟨t, ht⟩


@[simp] lemma shatters_empty : 𝒜.Shatters ∅ ↔ 𝒜.Nonempty :=
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : DecidableEq α
                                                      𝒜 : Finset (Finset α)
                                                      x✝ : 𝒜.Nonempty
                                                      t : Finset α
                                                      ht : HasSubset.Subset t EmptyCollection.emptyCollection
                                                      s : Finset α
                                                      hs : Membership.mem 𝒜 s
                                                      ⊢ Eq (Inter.inter EmptyCollection.emptyCollection s) t
                                                    -/
  ⟨Shatters.nonempty, fun ⟨s, hs⟩ t ht ↦ ⟨s, hs, by rwa [empty_inter, eq_comm, ← subset_empty]⟩⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


protected lemma Shatters.subset_iff (h : 𝒜.Shatters s) : t ⊆ s ↔ ∃ u ∈ 𝒜, s ∩ u = t :=
                     /-
                       α : Type u_1
                       inst✝ : DecidableEq α
                       𝒜 : Finset (Finset α)
                       s t : Finset α
                       h : 𝒜.Shatters s
                       ⊢ (Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter s u) t)) → HasSub …
                     -/
  ⟨fun ht ↦ h ht, by rintro ⟨u, _, rfl⟩; exact inter_subset_left⟩
                                         /-
                                           🎉 no goals
                                         -/


lemma shatters_iff : 𝒜.Shatters s ↔ 𝒜.image (fun t ↦ s ∩ t) = s.powerset :=
              /-
                α : Type u_1
                inst✝ : DecidableEq α
                𝒜 : Finset (Finset α)
                s : Finset α
                h : 𝒜.Shatters s
                ⊢ Eq (Finset.image (fun t => Inter.inter s t) 𝒜) s.powerset
              -/
  ⟨fun h ↦ by ext t; rw [mem_image, mem_powerset, h.subset_iff],
                     /-
                       🎉 no goals
                     -/
                    /-
                      α : Type u_1
                      inst✝ : DecidableEq α
                      𝒜 : Finset (Finset α)
                      s : Finset α
                      h : Eq (Finset.image (fun t => Inter.inter s t) 𝒜) s.powerset
                      t : Finset α
                      ht : HasSubset.Subset t s
                      ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter s u) t)
                    -/
    fun h t ht ↦ by rwa [← mem_powerset, ← h, mem_image] at ht⟩
                    /-
                      🎉 no goals
                    -/


lemma univ_shatters [Fintype α] : univ.Shatters s :=
  shatters_of_forall_subset fun _ _ ↦ mem_univ _


@[simp] lemma shatters_univ [Fintype α] : 𝒜.Shatters univ ↔ 𝒜 = univ := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    𝒜 : Finset (Finset α)
    inst✝ : Fintype α
    ⊢ Iff (𝒜.Shatters Finset.univ) (Eq 𝒜 Finset.univ)
  -/
  rw [shatters_iff, powerset_univ]; simp_rw [univ_inter, image_id']
                                    /-
                                      🎉 no goals
                                    -/


/-- The set family of sets that are shattered by `𝒜`. -/
def shatterer (𝒜 : Finset (Finset α)) : Finset (Finset α) :=
  {s ∈ 𝒜.biUnion powerset | 𝒜.Shatters s}


@[simp] lemma mem_shatterer : s ∈ 𝒜.shatterer ↔ 𝒜.Shatters s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Membership.mem 𝒜.shatterer s) (𝒜.Shatters s)
  -/
  refine mem_filter.trans <| and_iff_right_of_imp fun h ↦ ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    h : 𝒜.Shatters s
    ⊢ Membership.mem (𝒜.biUnion Finset.powerset) s
  -/
  simp_rw [mem_biUnion, mem_powerset]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    h : 𝒜.Shatters s
    ⊢ Exists fun a => And (Membership.mem 𝒜 a) (HasSubset.Subset s a)
  -/
  exact h.exists_superset
  /-
    🎉 no goals
  -/


@[gcongr] lemma shatterer_mono (h : 𝒜 ⊆ ℬ) : 𝒜.shatterer ⊆ ℬ.shatterer :=
             /-
               α : Type u_1
               inst✝ : DecidableEq α
               𝒜 ℬ : Finset (Finset α)
               h : HasSubset.Subset 𝒜 ℬ
               x✝ : Finset α
               ⊢ Membership.mem 𝒜.shatterer x✝ → Membership.mem ℬ.shatterer x✝
             -/
  fun _ ↦ by simpa using Shatters.mono_left h
             /-
               🎉 no goals
             -/


lemma subset_shatterer (h : IsLowerSet (𝒜 : Set (Finset α))) : 𝒜 ⊆ 𝒜.shatterer :=
  fun _s hs ↦ mem_shatterer.2 fun t ht ↦ ⟨t, h ht hs, inter_eq_right.2 ht⟩


@[simp] lemma isLowerSet_shatterer (𝒜 : Finset (Finset α)) :
                                                              /-
                                                                α : Type u_1
                                                                inst✝ : DecidableEq α
                                                                𝒜 : Finset (Finset α)
                                                                s t : Finset α
                                                                ⊢ LE.le t s → Membership.mem (↑𝒜.shatterer) s → Membership.mem (↑𝒜.shatterer) t
                                                              -/
    IsLowerSet (𝒜.shatterer : Set (Finset α)) := fun s t ↦ by simpa using Shatters.mono_right
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp] lemma shatterer_eq : 𝒜.shatterer = 𝒜 ↔ IsLowerSet (𝒜 : Set (Finset α)) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    ⊢ Iff (Eq 𝒜.shatterer 𝒜) (IsLowerSet ↑𝒜)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ Subset.antisymm (fun s hs ↦ ?_) <| subset_shatterer h⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      h : Eq 𝒜.shatterer 𝒜
      ⊢ IsLowerSet ↑𝒜
    -/
  · rw [← h]
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      h : Eq 𝒜.shatterer 𝒜
      ⊢ IsLowerSet ↑𝒜.shatterer
    -/
    exact isLowerSet_shatterer _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      h : IsLowerSet ↑𝒜
      s : Finset α
      hs : Membership.mem 𝒜.shatterer s
      ⊢ Membership.mem 𝒜 s
    -/
  · obtain ⟨t, ht, hst⟩ := (mem_shatterer.1 hs).exists_superset
    /-
      case refine_2.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      h : IsLowerSet ↑𝒜
      s : Finset α
      hs : Membership.mem 𝒜.shatterer s
      t : Finset α
      ht : Membership.mem 𝒜 t
      hst : HasSubset.Subset s t
      ⊢ Membership.mem 𝒜 s
    -/
    exact h hst ht
    /-
      🎉 no goals
    -/


                                                                         /-
                                                                           α : Type u_1
                                                                           inst✝ : DecidableEq α
                                                                           𝒜 : Finset (Finset α)
                                                                           ⊢ Eq 𝒜.shatterer.shatterer 𝒜.shatterer
                                                                         -/
@[simp] lemma shatterer_idem : 𝒜.shatterer.shatterer = 𝒜.shatterer := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp] lemma shatters_shatterer : 𝒜.shatterer.Shatters s ↔ 𝒜.Shatters s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (𝒜.shatterer.Shatters s) (𝒜.Shatters s)
  -/
  simp_rw [← mem_shatterer, shatterer_idem]
  /-
    🎉 no goals
  -/


protected alias ⟨_, Shatters.shatterer⟩ := shatters_shatterer


private lemma aux (h : ∀ t ∈ 𝒜, a ∉ t) (ht : 𝒜.Shatters t) : a ∉ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t : Finset α
    a : α
    h : ∀ (t : Finset α), Membership.mem 𝒜 t → Not (Membership.mem t a)
    ht : 𝒜.Shatters t
    ⊢ Not (Membership.mem t a)
  -/
  obtain ⟨u, hu, htu⟩ := ht.exists_superset; exact not_mem_mono htu <| h u hu
                                             /-
                                               🎉 no goals
                                             -/


/-- Pajor's variant of the **Sauer-Shelah lemma**. -/
lemma card_le_card_shatterer (𝒜 : Finset (Finset α)) : #𝒜 ≤ #𝒜.shatterer := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    ⊢ LE.le 𝒜.card 𝒜.shatterer.card
  -/
  refine memberFamily_induction_on 𝒜 ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      ⊢ LE.le EmptyCollection.emptyCollection.card EmptyCollection.emptyCollection.s …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      ⊢ LE.le (Singleton.singleton EmptyCollection.emptyCollection).card (Singleton. …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    ⊢ ∀ (a : α) ⦃𝒜 : Finset (Finset α)⦄, LE.le (Finset.nonMemberSubfamily a 𝒜).car …
  -/
  intros a 𝒜 ih₀ ih₁
  set ℬ : Finset (Finset α) :=
    ((memberSubfamily a 𝒜).shatterer ∩ (nonMemberSubfamily a 𝒜).shatterer).image (insert a)
  have hℬ : #ℬ = #((memberSubfamily a 𝒜).shatterer ∩ (nonMemberSubfamily a 𝒜).shatterer) := by
    refine card_image_of_injOn <| insert_erase_invOn.2.injOn.mono ?_
    simp only [coe_inter, Set.subset_def, Set.mem_inter_iff, mem_coe, Set.mem_setOf_eq, and_imp,
      mem_shatterer]
    exact fun s _ ↦ aux (fun t ht ↦ (mem_filter.1 ht).2)
  /-
    case refine_3
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜✝ : Finset (Finset α)
    a : α
    𝒜 : Finset (Finset α)
    ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
    ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
    ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
    hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
    ⊢ LE.le 𝒜.card 𝒜.shatterer.card
  -/
  rw [← card_memberSubfamily_add_card_nonMemberSubfamily a]
  /-
    case refine_3
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜✝ : Finset (Finset α)
    a : α
    𝒜 : Finset (Finset α)
    ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
    ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
    ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
    hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
    ⊢ LE.le (HAdd.hAdd (Finset.memberSubfamily a 𝒜).card (Finset.nonMemberSubfamil …
  -/
  refine (Nat.add_le_add ih₁ ih₀).trans ?_
  /-
    case refine_3
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜✝ : Finset (Finset α)
    a : α
    𝒜 : Finset (Finset α)
    ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
    ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
    ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
    hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
    ⊢ LE.le (HAdd.hAdd (Finset.memberSubfamily a 𝒜).shatterer.card (Finset.nonMemb …
  -/
  rw [← card_union_add_card_inter, ← hℬ, ← card_union_of_disjoint]
  /-
    case refine_3
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜✝ : Finset (Finset α)
    a : α
    𝒜 : Finset (Finset α)
    ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
    ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
    ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
    hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
    ⊢ LE.le (Union.union (Union.union (Finset.memberSubfamily a 𝒜).shatterer (Fins …
  -/
  swap
    /-
      case refine_3
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      ⊢ Disjoint (Union.union (Finset.memberSubfamily a 𝒜).shatterer (Finset.nonMemb …
    -/
  · simp only [ℬ, disjoint_left, mem_union, mem_shatterer, mem_image, not_exists, not_and]
    /-
      case refine_3
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      ⊢ ∀ ⦃a_1 : Finset α⦄, Or ((Finset.memberSubfamily a 𝒜).Shatters a_1) ((Finset. …
    -/
    rintro _ (hs | hs) s - rfl
      /-
        case refine_3.inl
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜✝ : Finset (Finset α)
        a : α
        𝒜 : Finset (Finset α)
        ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
        ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
        ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
        hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
        s : Finset α
        hs : (Finset.memberSubfamily a 𝒜).Shatters (Insert.insert a s)
        ⊢ False
      -/
    · exact aux (fun t ht ↦ (mem_memberSubfamily.1 ht).2) hs <| mem_insert_self _ _
      /-
        🎉 no goals
      -/
      /-
        case refine_3.inr
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜✝ : Finset (Finset α)
        a : α
        𝒜 : Finset (Finset α)
        ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
        ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
        ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
        hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
        s : Finset α
        hs : (Finset.nonMemberSubfamily a 𝒜).Shatters (Insert.insert a s)
        ⊢ False
      -/
    · exact aux (fun t ht ↦ (mem_nonMemberSubfamily.1 ht).2) hs <| mem_insert_self _ _
      /-
        🎉 no goals
      -/
  /-
    case refine_3
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜✝ : Finset (Finset α)
    a : α
    𝒜 : Finset (Finset α)
    ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
    ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
    ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
    hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
    ⊢ LE.le (Union.union (Union.union (Finset.memberSubfamily a 𝒜).shatterer (Fins …
  -/
  refine card_mono <| union_subset (union_subset ?_ <| shatterer_mono <| filter_subset _ _) ?_
    /-
      case refine_3.refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      ⊢ HasSubset.Subset (Finset.memberSubfamily a 𝒜).shatterer 𝒜.shatterer
    -/
  · simp only [subset_iff, mem_shatterer]
    /-
      case refine_3.refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      ⊢ ∀ ⦃x : Finset α⦄, (Finset.memberSubfamily a 𝒜).Shatters x → 𝒜.Shatters x
    -/
    rintro s hs t ht
    /-
      case refine_3.refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      s : Finset α
      hs : (Finset.memberSubfamily a 𝒜).Shatters s
      t : Finset α
      ht : HasSubset.Subset t s
      ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter s u) t)
    -/
    obtain ⟨u, hu, rfl⟩ := hs ht
    /-
      case refine_3.refine_1.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      s : Finset α
      hs : (Finset.memberSubfamily a 𝒜).Shatters s
      u : Finset α
      hu : Membership.mem (Finset.memberSubfamily a 𝒜) u
      ht : HasSubset.Subset (Inter.inter s u) s
      ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
    -/
    rw [mem_memberSubfamily] at hu
    /-
      case refine_3.refine_1.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      s : Finset α
      hs : (Finset.memberSubfamily a 𝒜).Shatters s
      u : Finset α
      hu : And (Membership.mem 𝒜 (Insert.insert a u)) (Not (Membership.mem u a))
      ht : HasSubset.Subset (Inter.inter s u) s
      ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
    -/
    refine ⟨insert a u, hu.1, inter_insert_of_not_mem fun ha ↦ ?_⟩
    /-
      case refine_3.refine_1.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      s : Finset α
      hs : (Finset.memberSubfamily a 𝒜).Shatters s
      u : Finset α
      hu : And (Membership.mem 𝒜 (Insert.insert a u)) (Not (Membership.mem u a))
      ht : HasSubset.Subset (Inter.inter s u) s
      ha : Membership.mem s a
      ⊢ False
    -/
    obtain ⟨v, hv, hsv⟩ := hs.exists_inter_eq_singleton ha
    /-
      case refine_3.refine_1.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      s : Finset α
      hs : (Finset.memberSubfamily a 𝒜).Shatters s
      u : Finset α
      hu : And (Membership.mem 𝒜 (Insert.insert a u)) (Not (Membership.mem u a))
      ht : HasSubset.Subset (Inter.inter s u) s
      ha : Membership.mem s a
      v : Finset α
      hv : Membership.mem (Finset.memberSubfamily a 𝒜) v
      hsv : Eq (Inter.inter s v) (Singleton.singleton a)
      ⊢ False
    -/
    rw [mem_memberSubfamily] at hv
    /-
      case refine_3.refine_1.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      s : Finset α
      hs : (Finset.memberSubfamily a 𝒜).Shatters s
      u : Finset α
      hu : And (Membership.mem 𝒜 (Insert.insert a u)) (Not (Membership.mem u a))
      ht : HasSubset.Subset (Inter.inter s u) s
      ha : Membership.mem s a
      v : Finset α
      hv : And (Membership.mem 𝒜 (Insert.insert a v)) (Not (Membership.mem v a))
      hsv : Eq (Inter.inter s v) (Singleton.singleton a)
      ⊢ False
    -/
    rw [← singleton_subset_iff (a := a), ← hsv] at hv
    /-
      case refine_3.refine_1.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      s : Finset α
      hs : (Finset.memberSubfamily a 𝒜).Shatters s
      u : Finset α
      hu : And (Membership.mem 𝒜 (Insert.insert a u)) (Not (Membership.mem u a))
      ht : HasSubset.Subset (Inter.inter s u) s
      ha : Membership.mem s a
      v : Finset α
      hv : And (Membership.mem 𝒜 (Insert.insert a v)) (Not (HasSubset.Subset (Inter. …
      hsv : Eq (Inter.inter s v) (Singleton.singleton a)
      ⊢ False
    -/
    exact hv.2 inter_subset_right
    /-
      🎉 no goals
    -/
    /-
      case refine_3.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      ⊢ HasSubset.Subset ℬ 𝒜.shatterer
    -/
  · refine forall_mem_image.2 fun s hs ↦ mem_shatterer.2 fun t ht ↦ ?_
    /-
      case refine_3.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      s : Finset α
      hs : Membership.mem (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finse …
      t : Finset α
      ht : HasSubset.Subset t (Insert.insert a s)
      ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter (Insert.insert a s …
    -/
    simp only [mem_inter, mem_shatterer] at hs
    /-
      case refine_3.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      s t : Finset α
      ht : HasSubset.Subset t (Insert.insert a s)
      hs : And ((Finset.memberSubfamily a 𝒜).Shatters s) ((Finset.nonMemberSubfamily …
      ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter (Insert.insert a s …
    -/
    rw [subset_insert_iff] at ht
    /-
      case refine_3.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜✝ : Finset (Finset α)
      a : α
      𝒜 : Finset (Finset α)
      ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
      ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
      ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
      hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
      s t : Finset α
      ht : HasSubset.Subset (t.erase a) s
      hs : And ((Finset.memberSubfamily a 𝒜).Shatters s) ((Finset.nonMemberSubfamily …
      ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter (Insert.insert a s …
    -/
    by_cases ha : a ∈ t
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜✝ : Finset (Finset α)
        a : α
        𝒜 : Finset (Finset α)
        ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
        ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
        ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
        hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
        s t : Finset α
        ht : HasSubset.Subset (t.erase a) s
        hs : And ((Finset.memberSubfamily a 𝒜).Shatters s) ((Finset.nonMemberSubfamily …
        ha : Membership.mem t a
        ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter (Insert.insert a s …
      -/
    · obtain ⟨u, hu, hsu⟩ := hs.1 ht
      /-
        case pos.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜✝ : Finset (Finset α)
        a : α
        𝒜 : Finset (Finset α)
        ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
        ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
        ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
        hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
        s t : Finset α
        ht : HasSubset.Subset (t.erase a) s
        hs : And ((Finset.memberSubfamily a 𝒜).Shatters s) ((Finset.nonMemberSubfamily …
        ha : Membership.mem t a
        u : Finset α
        hu : Membership.mem (Finset.memberSubfamily a 𝒜) u
        hsu : Eq (Inter.inter s u) (t.erase a)
        ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter (Insert.insert a s …
      -/
      rw [mem_memberSubfamily] at hu
      /-
        case pos.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜✝ : Finset (Finset α)
        a : α
        𝒜 : Finset (Finset α)
        ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
        ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
        ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
        hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
        s t : Finset α
        ht : HasSubset.Subset (t.erase a) s
        hs : And ((Finset.memberSubfamily a 𝒜).Shatters s) ((Finset.nonMemberSubfamily …
        ha : Membership.mem t a
        u : Finset α
        hu : And (Membership.mem 𝒜 (Insert.insert a u)) (Not (Membership.mem u a))
        hsu : Eq (Inter.inter s u) (t.erase a)
        ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter (Insert.insert a s …
      -/
      refine ⟨_, hu.1, ?_⟩
      /-
        case pos.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜✝ : Finset (Finset α)
        a : α
        𝒜 : Finset (Finset α)
        ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
        ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
        ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
        hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
        s t : Finset α
        ht : HasSubset.Subset (t.erase a) s
        hs : And ((Finset.memberSubfamily a 𝒜).Shatters s) ((Finset.nonMemberSubfamily …
        ha : Membership.mem t a
        u : Finset α
        hu : And (Membership.mem 𝒜 (Insert.insert a u)) (Not (Membership.mem u a))
        hsu : Eq (Inter.inter s u) (t.erase a)
        ⊢ Eq (Inter.inter (Insert.insert a s) (Insert.insert a u)) t
      -/
      rw [← insert_inter_distrib, hsu, insert_erase ha]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜✝ : Finset (Finset α)
        a : α
        𝒜 : Finset (Finset α)
        ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
        ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
        ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
        hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
        s t : Finset α
        ht : HasSubset.Subset (t.erase a) s
        hs : And ((Finset.memberSubfamily a 𝒜).Shatters s) ((Finset.nonMemberSubfamily …
        ha : Not (Membership.mem t a)
        ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter (Insert.insert a s …
      -/
    · obtain ⟨u, hu, hsu⟩ := hs.2 ht
      /-
        case neg.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜✝ : Finset (Finset α)
        a : α
        𝒜 : Finset (Finset α)
        ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
        ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
        ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
        hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
        s t : Finset α
        ht : HasSubset.Subset (t.erase a) s
        hs : And ((Finset.memberSubfamily a 𝒜).Shatters s) ((Finset.nonMemberSubfamily …
        ha : Not (Membership.mem t a)
        u : Finset α
        hu : Membership.mem (Finset.nonMemberSubfamily a 𝒜) u
        hsu : Eq (Inter.inter s u) (t.erase a)
        ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter (Insert.insert a s …
      -/
      rw [mem_nonMemberSubfamily] at hu
      /-
        case neg.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜✝ : Finset (Finset α)
        a : α
        𝒜 : Finset (Finset α)
        ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
        ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
        ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
        hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
        s t : Finset α
        ht : HasSubset.Subset (t.erase a) s
        hs : And ((Finset.memberSubfamily a 𝒜).Shatters s) ((Finset.nonMemberSubfamily …
        ha : Not (Membership.mem t a)
        u : Finset α
        hu : And (Membership.mem 𝒜 u) (Not (Membership.mem u a))
        hsu : Eq (Inter.inter s u) (t.erase a)
        ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter (Insert.insert a s …
      -/
      refine ⟨_, hu.1, ?_⟩
      /-
        case neg.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜✝ : Finset (Finset α)
        a : α
        𝒜 : Finset (Finset α)
        ih₀ : LE.le (Finset.nonMemberSubfamily a 𝒜).card (Finset.nonMemberSubfamily a  …
        ih₁ : LE.le (Finset.memberSubfamily a 𝒜).card (Finset.memberSubfamily a 𝒜).sha …
        ℬ : Finset (Finset α) := Finset.image (Insert.insert a) (Inter.inter (Finset.m …
        hℬ : Eq ℬ.card (Inter.inter (Finset.memberSubfamily a 𝒜).shatterer (Finset.non …
        s t : Finset α
        ht : HasSubset.Subset (t.erase a) s
        hs : And ((Finset.memberSubfamily a 𝒜).Shatters s) ((Finset.nonMemberSubfamily …
        ha : Not (Membership.mem t a)
        u : Finset α
        hu : And (Membership.mem 𝒜 u) (Not (Membership.mem u a))
        hsu : Eq (Inter.inter s u) (t.erase a)
        ⊢ Eq (Inter.inter (Insert.insert a s) u) t
      -/
      rwa [insert_inter_of_not_mem hu.2, hsu, erase_eq_self]
      /-
        🎉 no goals
      -/


lemma Shatters.of_compression (hs : (𝓓 a 𝒜).Shatters s) : 𝒜.Shatters s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    hs : (Down.compression a 𝒜).Shatters s
    ⊢ 𝒜.Shatters s
  -/
  intros t ht
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    hs : (Down.compression a 𝒜).Shatters s
    t : Finset α
    ht : HasSubset.Subset t s
    ⊢ Exists fun u => And (Membership.mem 𝒜 u) (Eq (Inter.inter s u) t)
  -/
  obtain ⟨u, hu, rfl⟩ := hs ht
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    hs : (Down.compression a 𝒜).Shatters s
    u : Finset α
    hu : Membership.mem (Down.compression a 𝒜) u
    ht : HasSubset.Subset (Inter.inter s u) s
    ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
  -/
  rw [Down.mem_compression] at hu
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    hs : (Down.compression a 𝒜).Shatters s
    u : Finset α
    hu : Or (And (Membership.mem 𝒜 u) (Membership.mem 𝒜 (u.erase a))) (And (Not (M …
    ht : HasSubset.Subset (Inter.inter s u) s
    ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
  -/
  obtain hu | hu := hu
    /-
      case intro.intro.inl
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      hs : (Down.compression a 𝒜).Shatters s
      u : Finset α
      ht : HasSubset.Subset (Inter.inter s u) s
      hu : And (Membership.mem 𝒜 u) (Membership.mem 𝒜 (u.erase a))
      ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
    -/
  · exact ⟨u, hu.1, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    s : Finset α
    a : α
    hs : (Down.compression a 𝒜).Shatters s
    u : Finset α
    ht : HasSubset.Subset (Inter.inter s u) s
    hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
    ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
  -/
  by_cases ha : a ∈ s
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      hs : (Down.compression a 𝒜).Shatters s
      u : Finset α
      ht : HasSubset.Subset (Inter.inter s u) s
      hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
      ha : Membership.mem s a
      ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
    -/
  · obtain ⟨v, hv, hsv⟩ := hs <| insert_subset ha ht
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      hs : (Down.compression a 𝒜).Shatters s
      u : Finset α
      ht : HasSubset.Subset (Inter.inter s u) s
      hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
      ha : Membership.mem s a
      v : Finset α
      hv : Membership.mem (Down.compression a 𝒜) v
      hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
      ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
    -/
    rw [Down.mem_compression] at hv
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      hs : (Down.compression a 𝒜).Shatters s
      u : Finset α
      ht : HasSubset.Subset (Inter.inter s u) s
      hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
      ha : Membership.mem s a
      v : Finset α
      hv : Or (And (Membership.mem 𝒜 v) (Membership.mem 𝒜 (v.erase a))) (And (Not (M …
      hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
      ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
    -/
    obtain hv | hv := hv
      /-
        case pos.intro.intro.inl
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜 : Finset (Finset α)
        s : Finset α
        a : α
        hs : (Down.compression a 𝒜).Shatters s
        u : Finset α
        ht : HasSubset.Subset (Inter.inter s u) s
        hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
        ha : Membership.mem s a
        v : Finset α
        hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
        hv : And (Membership.mem 𝒜 v) (Membership.mem 𝒜 (v.erase a))
        ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
      -/
    · refine ⟨erase v a, hv.2, ?_⟩
      /-
        case pos.intro.intro.inl
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜 : Finset (Finset α)
        s : Finset α
        a : α
        hs : (Down.compression a 𝒜).Shatters s
        u : Finset α
        ht : HasSubset.Subset (Inter.inter s u) s
        hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
        ha : Membership.mem s a
        v : Finset α
        hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
        hv : And (Membership.mem 𝒜 v) (Membership.mem 𝒜 (v.erase a))
        ⊢ Eq (Inter.inter s (v.erase a)) (Inter.inter s u)
      -/
      rw [inter_erase, hsv, erase_insert]
      /-
        case pos.intro.intro.inl
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜 : Finset (Finset α)
        s : Finset α
        a : α
        hs : (Down.compression a 𝒜).Shatters s
        u : Finset α
        ht : HasSubset.Subset (Inter.inter s u) s
        hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
        ha : Membership.mem s a
        v : Finset α
        hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
        hv : And (Membership.mem 𝒜 v) (Membership.mem 𝒜 (v.erase a))
        ⊢ Not (Membership.mem (Inter.inter s u) a)
      -/
      rintro ha
      /-
        case pos.intro.intro.inl
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜 : Finset (Finset α)
        s : Finset α
        a : α
        hs : (Down.compression a 𝒜).Shatters s
        u : Finset α
        ht : HasSubset.Subset (Inter.inter s u) s
        hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
        ha✝ : Membership.mem s a
        v : Finset α
        hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
        hv : And (Membership.mem 𝒜 v) (Membership.mem 𝒜 (v.erase a))
        ha : Membership.mem (Inter.inter s u) a
        ⊢ False
      -/
      rw [insert_eq_self.2 (mem_inter.1 ha).2] at hu
      /-
        case pos.intro.intro.inl
        α : Type u_1
        inst✝ : DecidableEq α
        𝒜 : Finset (Finset α)
        s : Finset α
        a : α
        hs : (Down.compression a 𝒜).Shatters s
        u : Finset α
        ht : HasSubset.Subset (Inter.inter s u) s
        hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 u)
        ha✝ : Membership.mem s a
        v : Finset α
        hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
        hv : And (Membership.mem 𝒜 v) (Membership.mem 𝒜 (v.erase a))
        ha : Membership.mem (Inter.inter s u) a
        ⊢ False
      -/
      exact hu.1 hu.2
      /-
        🎉 no goals
      -/
    /-
      case pos.intro.intro.inr
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      hs : (Down.compression a 𝒜).Shatters s
      u : Finset α
      ht : HasSubset.Subset (Inter.inter s u) s
      hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
      ha : Membership.mem s a
      v : Finset α
      hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
      hv : And (Not (Membership.mem 𝒜 v)) (Membership.mem 𝒜 (Insert.insert a v))
      ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
    -/
    rw [insert_eq_self.2 <| inter_subset_right (s₁ := s) ?_] at hv
    /-
      case pos.intro.intro.inr
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      hs : (Down.compression a 𝒜).Shatters s
      u : Finset α
      ht : HasSubset.Subset (Inter.inter s u) s
      hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
      ha : Membership.mem s a
      v : Finset α
      hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
      hv : And (Not (Membership.mem 𝒜 v)) (Membership.mem 𝒜 v)
      ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
    -/
    cases hv.1 hv.2
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      hs : (Down.compression a 𝒜).Shatters s
      u : Finset α
      ht : HasSubset.Subset (Inter.inter s u) s
      hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
      ha : Membership.mem s a
      v : Finset α
      hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
      hv : And (Not (Membership.mem 𝒜 v)) (Membership.mem 𝒜 (Insert.insert a v))
      ⊢ Membership.mem (Inter.inter s v) a
    -/
    rw [hsv]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      hs : (Down.compression a 𝒜).Shatters s
      u : Finset α
      ht : HasSubset.Subset (Inter.inter s u) s
      hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
      ha : Membership.mem s a
      v : Finset α
      hsv : Eq (Inter.inter s v) (Insert.insert a (Inter.inter s u))
      hv : And (Not (Membership.mem 𝒜 v)) (Membership.mem 𝒜 (Insert.insert a v))
      ⊢ Membership.mem (Insert.insert a (Inter.inter s u)) a
    -/
    exact mem_insert_self _ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      hs : (Down.compression a 𝒜).Shatters s
      u : Finset α
      ht : HasSubset.Subset (Inter.inter s u) s
      hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
      ha : Not (Membership.mem s a)
      ⊢ Exists fun u_1 => And (Membership.mem 𝒜 u_1) (Eq (Inter.inter s u_1) (Inter. …
    -/
  · refine ⟨insert a u, hu.2, ?_⟩
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      s : Finset α
      a : α
      hs : (Down.compression a 𝒜).Shatters s
      u : Finset α
      ht : HasSubset.Subset (Inter.inter s u) s
      hu : And (Not (Membership.mem 𝒜 u)) (Membership.mem 𝒜 (Insert.insert a u))
      ha : Not (Membership.mem s a)
      ⊢ Eq (Inter.inter s (Insert.insert a u)) (Inter.inter s u)
    -/
    rw [inter_insert_of_not_mem ha]
    /-
      🎉 no goals
    -/


lemma shatterer_compress_subset_shatterer (a : α) (𝒜 : Finset (Finset α)) :
    (𝓓 a 𝒜).shatterer ⊆ 𝒜.shatterer := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    𝒜 : Finset (Finset α)
    ⊢ HasSubset.Subset (Down.compression a 𝒜).shatterer 𝒜.shatterer
  -/
  simp only [subset_iff, mem_shatterer]; exact fun s hs ↦ hs.of_compression
                                         /-
                                           🎉 no goals
                                         -/


/-- The Vapnik-Chervonenkis dimension of a set family is the maximal size of a set it shatters. -/
def vcDim (𝒜 : Finset (Finset α)) : ℕ := 𝒜.shatterer.sup card


                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : DecidableEq α
                                                                     𝒜 ℬ : Finset (Finset α)
                                                                     h𝒜ℬ : HasSubset.Subset 𝒜 ℬ
                                                                     ⊢ LE.le 𝒜.vcDim ℬ.vcDim
                                                                   -/
@[gcongr] lemma vcDim_mono (h𝒜ℬ : 𝒜 ⊆ ℬ) : 𝒜.vcDim ≤ ℬ.vcDim := by unfold vcDim; gcongr
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


lemma Shatters.card_le_vcDim (hs : 𝒜.Shatters s) : #s ≤ 𝒜.vcDim := le_sup <| mem_shatterer.2 hs


/-- Down-compressing decreases the VC-dimension. -/
lemma vcDim_compress_le (a : α) (𝒜 : Finset (Finset α)) : (𝓓 a 𝒜).vcDim ≤ 𝒜.vcDim :=
  sup_mono <| shatterer_compress_subset_shatterer _ _


/-- The **Sauer-Shelah lemma**. -/
lemma card_shatterer_le_sum_vcDim [Fintype α] :
    #𝒜.shatterer ≤ ∑ k ∈ Iic 𝒜.vcDim, (Fintype.card α).choose k := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    𝒜 : Finset (Finset α)
    inst✝ : Fintype α
    ⊢ LE.le 𝒜.shatterer.card ((Finset.Iic 𝒜.vcDim).sum fun k => (Fintype.card α).c …
  -/
  simp_rw [← card_univ, ← card_powersetCard]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    𝒜 : Finset (Finset α)
    inst✝ : Fintype α
    ⊢ LE.le 𝒜.shatterer.card ((Finset.Iic 𝒜.vcDim).sum fun x => (Finset.powersetCa …
  -/
  refine (card_le_card fun s hs ↦ mem_biUnion.2 ⟨#s, ?_⟩).trans card_biUnion_le
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    𝒜 : Finset (Finset α)
    inst✝ : Fintype α
    s : Finset α
    hs : Membership.mem 𝒜.shatterer s
    ⊢ And (Membership.mem (Finset.Iic 𝒜.vcDim) s.card) (Membership.mem (Finset.pow …
  -/
  exact ⟨mem_Iic.2 (mem_shatterer.1 hs).card_le_vcDim, mem_powersetCard_univ.2 rfl⟩
  /-
    🎉 no goals
  -/


