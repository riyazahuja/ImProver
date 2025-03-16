/-- A topological space is **Alexandrov-discrete** or **finitely generated** if the intersection of
a family of open sets is open. -/
class AlexandrovDiscrete (α : Type*) [TopologicalSpace α] : Prop where
  /-- The intersection of a family of open sets is an open set. Use `isOpen_sInter` in the root
  namespace instead. -/
  protected isOpen_sInter : ∀ S : Set (Set α), (∀ s ∈ S, IsOpen s) → IsOpen (⋂₀ S)


instance DiscreteTopology.toAlexandrovDiscrete [DiscreteTopology α] : AlexandrovDiscrete α where
  isOpen_sInter _ _ := isOpen_discrete _


instance Finite.toAlexandrovDiscrete [Finite α] : AlexandrovDiscrete α where
  isOpen_sInter S := (toFinite S).isOpen_sInter


lemma isOpen_sInter : (∀ s ∈ S, IsOpen s) → IsOpen (⋂₀ S) := AlexandrovDiscrete.isOpen_sInter _


lemma isOpen_iInter (hf : ∀ i, IsOpen (f i)) : IsOpen (⋂ i, f i) :=
  isOpen_sInter <| forall_mem_range.2 hf


lemma isOpen_iInter₂ {f : ∀ i, κ i → Set α} (hf : ∀ i j, IsOpen (f i j)) :
    IsOpen (⋂ i, ⋂ j, f i j) :=
  isOpen_iInter fun _ ↦ isOpen_iInter <| hf _


lemma isClosed_sUnion (hS : ∀ s ∈ S, IsClosed s) : IsClosed (⋃₀ S) := by
  /-
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : AlexandrovDiscrete α
    S : Set (Set α)
    hS : ∀ (s : Set α), Membership.mem S s → IsClosed s
    ⊢ IsClosed S.sUnion
  -/
  simp only [← isOpen_compl_iff, compl_sUnion] at hS ⊢; exact isOpen_sInter <| forall_mem_image.2 hS
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma isClosed_iUnion (hf : ∀ i, IsClosed (f i)) : IsClosed (⋃ i, f i) :=
  isClosed_sUnion <| forall_mem_range.2 hf


lemma isClosed_iUnion₂ {f : ∀ i, κ i → Set α} (hf : ∀ i j, IsClosed (f i j)) :
    IsClosed (⋃ i, ⋃ j, f i j) :=
  isClosed_iUnion fun _ ↦ isClosed_iUnion <| hf _


lemma isClopen_sInter (hS : ∀ s ∈ S, IsClopen s) : IsClopen (⋂₀ S) :=
  ⟨isClosed_sInter fun s hs ↦ (hS s hs).1, isOpen_sInter fun s hs ↦ (hS s hs).2⟩


lemma isClopen_iInter (hf : ∀ i, IsClopen (f i)) : IsClopen (⋂ i, f i) :=
  ⟨isClosed_iInter fun i ↦ (hf i).1, isOpen_iInter fun i ↦ (hf i).2⟩


lemma isClopen_iInter₂ {f : ∀ i, κ i → Set α} (hf : ∀ i j, IsClopen (f i j)) :
    IsClopen (⋂ i, ⋂ j, f i j) :=
  isClopen_iInter fun _ ↦ isClopen_iInter <| hf _


lemma isClopen_sUnion (hS : ∀ s ∈ S, IsClopen s) : IsClopen (⋃₀ S) :=
  ⟨isClosed_sUnion fun s hs ↦ (hS s hs).1, isOpen_sUnion fun s hs ↦ (hS s hs).2⟩


lemma isClopen_iUnion (hf : ∀ i, IsClopen (f i)) : IsClopen (⋃ i, f i) :=
  ⟨isClosed_iUnion fun i ↦ (hf i).1, isOpen_iUnion fun i ↦ (hf i).2⟩


lemma isClopen_iUnion₂ {f : ∀ i, κ i → Set α} (hf : ∀ i j, IsClopen (f i j)) :
    IsClopen (⋃ i, ⋃ j, f i j) :=
  isClopen_iUnion fun _ ↦ isClopen_iUnion <| hf _


lemma interior_iInter (f : ι → Set α) : interior (⋂ i, f i) = ⋂ i, interior (f i) :=
  (interior_maximal (iInter_mono fun _ ↦ interior_subset) <| isOpen_iInter fun _ ↦
    isOpen_interior).antisymm' <| subset_iInter fun _ ↦ interior_mono <| iInter_subset _ _


lemma interior_sInter (S : Set (Set α)) : interior (⋂₀ S) = ⋂ s ∈ S, interior s := by
  /-
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : AlexandrovDiscrete α
    S : Set (Set α)
    ⊢ Eq (interior S.sInter) (Set.iInter fun s => Set.iInter fun h => interior s)
  -/
  simp_rw [sInter_eq_biInter, interior_iInter]
  /-
    🎉 no goals
  -/


lemma closure_iUnion (f : ι → Set α) : closure (⋃ i, f i) = ⋃ i, closure (f i) :=
  compl_injective <| by
    /-
      ι : Sort u_1
      α : Type u_3
      inst✝¹ : TopologicalSpace α
      inst✝ : AlexandrovDiscrete α
      f : ι → Set α
      ⊢ Eq (HasCompl.compl (closure (Set.iUnion fun i => f i))) (HasCompl.compl (Set …
    -/
    simpa only [← interior_compl, compl_iUnion] using interior_iInter fun i ↦ (f i)ᶜ
    /-
      🎉 no goals
    -/


lemma closure_sUnion (S : Set (Set α)) : closure (⋃₀ S) = ⋃ s ∈ S, closure s := by
  /-
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : AlexandrovDiscrete α
    S : Set (Set α)
    ⊢ Eq (closure S.sUnion) (Set.iUnion fun s => Set.iUnion fun h => closure s)
  -/
  simp_rw [sUnion_eq_biUnion, closure_iUnion]
  /-
    🎉 no goals
  -/


lemma Topology.IsInducing.alexandrovDiscrete [AlexandrovDiscrete α] {f : β → α} (h : IsInducing f) :
    AlexandrovDiscrete β where
  isOpen_sInter S hS := by
    /-
      α : Type u_3
      β : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : AlexandrovDiscrete α
      f : β → α
      h : Topology.IsInducing f
      S : Set (Set β)
      hS : ∀ (s : Set β), Membership.mem S s → IsOpen s
      ⊢ IsOpen S.sInter
    -/
    simp_rw [h.isOpen_iff] at hS ⊢
    /-
      α : Type u_3
      β : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : AlexandrovDiscrete α
      f : β → α
      h : Topology.IsInducing f
      S : Set (Set β)
      hS : ∀ (s : Set β), Membership.mem S s → Exists fun t => And (IsOpen t) (Eq (S …
      ⊢ Exists fun t => And (IsOpen t) (Eq (Set.preimage f t) S.sInter)
    -/
    choose U hU htU using hS
    /-
      α : Type u_3
      β : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : AlexandrovDiscrete α
      f : β → α
      h : Topology.IsInducing f
      S : Set (Set β)
      U : (s : Set β) → Membership.mem S s → Set α
      hU : ∀ (s : Set β) (a : Membership.mem S s), IsOpen (U s a)
      htU : ∀ (s : Set β) (a : Membership.mem S s), Eq (Set.preimage f (U s a)) s
      ⊢ Exists fun t => And (IsOpen t) (Eq (Set.preimage f t) S.sInter)
    -/
    refine ⟨_, isOpen_iInter₂ hU, ?_⟩
    /-
      α : Type u_3
      β : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : AlexandrovDiscrete α
      f : β → α
      h : Topology.IsInducing f
      S : Set (Set β)
      U : (s : Set β) → Membership.mem S s → Set α
      hU : ∀ (s : Set β) (a : Membership.mem S s), IsOpen (U s a)
      htU : ∀ (s : Set β) (a : Membership.mem S s), Eq (Set.preimage f (U s a)) s
      ⊢ Eq (Set.preimage f (Set.iInter fun i => Set.iInter fun j => U i j)) S.sInter
    -/
    simp_rw [preimage_iInter, htU, sInter_eq_biInter]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-28")]
alias Inducing.alexandrovDiscrete := IsInducing.alexandrovDiscrete


lemma AlexandrovDiscrete.sup {t₁ t₂ : TopologicalSpace α} (_ : @AlexandrovDiscrete α t₁)
    (_ : @AlexandrovDiscrete α t₂) :
    @AlexandrovDiscrete α (t₁ ⊔ t₂) :=
  @AlexandrovDiscrete.mk α (t₁ ⊔ t₂) fun _S hS ↦
    ⟨@isOpen_sInter _ t₁ _ _ fun _s hs ↦ (hS _ hs).1, isOpen_sInter fun _s hs ↦ (hS _ hs).2⟩


lemma alexandrovDiscrete_iSup {t : ι → TopologicalSpace α} (_ : ∀ i, @AlexandrovDiscrete α (t i)) :
    @AlexandrovDiscrete α (⨆ i, t i) :=
  @AlexandrovDiscrete.mk α (⨆ i, t i)
    fun _S hS ↦ isOpen_iSup_iff.2
      fun i ↦ @isOpen_sInter _ (t i) _ _
        fun _s hs ↦ isOpen_iSup_iff.1 (hS _ hs) _


@[simp] lemma isOpen_exterior : IsOpen (exterior s) := by
  /-
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : AlexandrovDiscrete α
    s : Set α
    ⊢ IsOpen (exterior s)
  -/
  rw [exterior_def]; exact isOpen_sInter fun _ ↦ And.left
                     /-
                       🎉 no goals
                     -/


lemma exterior_mem_nhdsSet : exterior s ∈ 𝓝ˢ s := isOpen_exterior.mem_nhdsSet.2 subset_exterior


@[simp] lemma exterior_eq_iff_isOpen : exterior s = s ↔ IsOpen s :=
  ⟨fun h ↦ h ▸ isOpen_exterior, IsOpen.exterior_eq⟩


@[simp] lemma exterior_subset_iff_isOpen : exterior s ⊆ s ↔ IsOpen s := by
  /-
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : AlexandrovDiscrete α
    s : Set α
    ⊢ Iff (HasSubset.Subset (exterior s) s) (IsOpen s)
  -/
  simp only [exterior_eq_iff_isOpen.symm, Subset.antisymm_iff, subset_exterior, and_true]
  /-
    🎉 no goals
  -/


lemma exterior_subset_iff : exterior s ⊆ t ↔ ∃ U, IsOpen U ∧ s ⊆ U ∧ U ⊆ t :=
  ⟨fun h ↦ ⟨exterior s, isOpen_exterior, subset_exterior, h⟩,
    fun ⟨_U, hU, hsU, hUt⟩ ↦ (exterior_minimal hsU hU).trans hUt⟩


lemma exterior_subset_iff_mem_nhdsSet : exterior s ⊆ t ↔ t ∈ 𝓝ˢ s :=
  exterior_subset_iff.trans mem_nhdsSet_iff_exists.symm


lemma exterior_singleton_subset_iff_mem_nhds : exterior {a} ⊆ t ↔ t ∈ 𝓝 a := by
  /-
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : AlexandrovDiscrete α
    t : Set α
    a : α
    ⊢ Iff (HasSubset.Subset (exterior (Singleton.singleton a)) t) (Membership.mem  …
  -/
  simp [exterior_subset_iff_mem_nhdsSet]
  /-
    🎉 no goals
  -/


lemma gc_exterior_interior : GaloisConnection (exterior : Set α → Set α) interior :=
               /-
                 α : Type u_3
                 inst✝¹ : TopologicalSpace α
                 inst✝ : AlexandrovDiscrete α
                 s t : Set α
                 ⊢ Iff (LE.le (exterior s) t) (LE.le s (interior t))
               -/
  fun s t ↦ by simp [exterior_subset_iff, subset_interior_iff]
               /-
                 🎉 no goals
               -/


@[simp] lemma principal_exterior (s : Set α) : 𝓟 (exterior s) = 𝓝ˢ s := by
  /-
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : AlexandrovDiscrete α
    s : Set α
    ⊢ Eq (Filter.principal (exterior s)) (nhdsSet s)
  -/
  rw [← nhdsSet_exterior, isOpen_exterior.nhdsSet_eq]
  /-
    🎉 no goals
  -/


lemma isOpen_iff_forall_specializes : IsOpen s ↔ ∀ x y, x ⤳ y → y ∈ s → x ∈ s := by
  simp only [← exterior_subset_iff_isOpen, Set.subset_def, mem_exterior_iff_specializes, exists_imp,
    and_imp, @forall_swap (_ ⤳ _)]


lemma alexandrovDiscrete_coinduced {β : Type*} {f : α → β} :
    @AlexandrovDiscrete β (coinduced f ‹_›) :=
  @AlexandrovDiscrete.mk β (coinduced f ‹_›) fun S hS ↦ by
    /-
      α : Type u_3
      inst✝¹ : TopologicalSpace α
      inst✝ : AlexandrovDiscrete α
      β : Type u_5
      f : α → β
      S : Set (Set β)
      hS : ∀ (s : Set β), Membership.mem S s → IsOpen s
      ⊢ IsOpen S.sInter
    -/
    rw [isOpen_coinduced, preimage_sInter]; exact isOpen_iInter₂ hS
                                            /-
                                              🎉 no goals
                                            -/


instance AlexandrovDiscrete.toFirstCountable : FirstCountableTopology α where
                                                                           /-
                                                                             ι : Sort u_1
                                                                             κ : ι → Sort u_2
                                                                             α : Type u_3
                                                                             β : Type u_4
                                                                             inst✝³ : TopologicalSpace α
                                                                             inst✝² : TopologicalSpace β
                                                                             inst✝¹ : AlexandrovDiscrete α
                                                                             inst✝ : AlexandrovDiscrete β
                                                                             s t : Set α
                                                                             a✝ a : α
                                                                             ⊢ Eq (nhds a) (Filter.generate (Singleton.singleton (exterior (Singleton.singl …
                                                                           -/
  nhds_generated_countable a := ⟨{exterior {a}}, countable_singleton _, by simp⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


instance AlexandrovDiscrete.toLocallyCompactSpace : LocallyCompactSpace α where
  local_compact_nhds a _U hU := ⟨exterior {a},
    isOpen_exterior.mem_nhds <| subset_exterior <| mem_singleton _,
      exterior_singleton_subset_iff_mem_nhds.2 hU, isCompact_singleton.exterior⟩


instance Subtype.instAlexandrovDiscrete {p : α → Prop} : AlexandrovDiscrete {a // p a} :=
  IsInducing.subtypeVal.alexandrovDiscrete


instance Quotient.instAlexandrovDiscrete {s : Setoid α} : AlexandrovDiscrete (Quotient s) :=
  alexandrovDiscrete_coinduced


instance Sum.instAlexandrovDiscrete : AlexandrovDiscrete (α ⊕ β) :=
  alexandrovDiscrete_coinduced.sup alexandrovDiscrete_coinduced


instance Sigma.instAlexandrovDiscrete {ι : Type*} {π : ι → Type*} [∀ i, TopologicalSpace (π i)]
    [∀ i, AlexandrovDiscrete (π i)] : AlexandrovDiscrete (Σ i, π i) :=
  alexandrovDiscrete_iSup fun _ ↦ alexandrovDiscrete_coinduced


