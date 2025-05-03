/-- The topological support of a function is the closure of its support, i.e. the closure of the
  set of all elements where the function is not equal to 1. -/
@[to_additive " The topological support of a function is the closure of its support. i.e. the
closure of the set of all elements where the function is nonzero. "]
def mulTSupport (f : X → α) : Set X := closure (mulSupport f)


@[to_additive]
theorem subset_mulTSupport (f : X → α) : mulSupport f ⊆ mulTSupport f :=
  subset_closure


@[to_additive]
theorem isClosed_mulTSupport (f : X → α) : IsClosed (mulTSupport f) :=
  isClosed_closure


@[to_additive]
theorem mulTSupport_eq_empty_iff {f : X → α} : mulTSupport f = ∅ ↔ f = 1 := by
  /-
    X : Type u_1
    α : Type u_2
    inst✝¹ : One α
    inst✝ : TopologicalSpace X
    f : X → α
    ⊢ Iff (Eq (mulTSupport f) EmptyCollection.emptyCollection) (Eq f 1)
  -/
  rw [mulTSupport, closure_empty_iff, mulSupport_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem image_eq_one_of_nmem_mulTSupport {f : X → α} {x : X} (hx : x ∉ mulTSupport f) : f x = 1 :=
  mulSupport_subset_iff'.mp (subset_mulTSupport f) x hx


@[to_additive]
theorem range_subset_insert_image_mulTSupport (f : X → α) :
    range f ⊆ insert 1 (f '' mulTSupport f) :=
  (range_subset_insert_image_mulSupport f).trans <|
    insert_subset_insert <| image_subset _ subset_closure


@[to_additive]
theorem range_eq_image_mulTSupport_or (f : X → α) :
    range f = f '' mulTSupport f ∨ range f = insert 1 (f '' mulTSupport f) :=
  (wcovBy_insert _ _).eq_or_eq (image_subset_range _ _) (range_subset_insert_image_mulTSupport f)


theorem tsupport_mul_subset_left {α : Type*} [MulZeroClass α] {f g : X → α} :
    (tsupport fun x => f x * g x) ⊆ tsupport f :=
  closure_mono (support_mul_subset_left _ _)


theorem tsupport_mul_subset_right {α : Type*} [MulZeroClass α] {f g : X → α} :
    (tsupport fun x => f x * g x) ⊆ tsupport g :=
  closure_mono (support_mul_subset_right _ _)


theorem tsupport_smul_subset_left {M α} [TopologicalSpace X] [Zero M] [Zero α] [SMulWithZero M α]
    (f : X → M) (g : X → α) : (tsupport fun x => f x • g x) ⊆ tsupport f :=
  closure_mono <| support_smul_subset_left f g


theorem tsupport_smul_subset_right {M α} [TopologicalSpace X] [Zero α] [SMulZeroClass M α]
    (f : X → M) (g : X → α) : (tsupport fun x => f x • g x) ⊆ tsupport g :=
  closure_mono <| support_smul_subset_right f g


@[to_additive]
theorem mulTSupport_mul [TopologicalSpace X] [Monoid α] {f g : X → α} :
    (mulTSupport fun x ↦ f x * g x) ⊆ mulTSupport f ∪ mulTSupport g :=
  closure_minimal
    ((mulSupport_mul f g).trans (union_subset_union (subset_mulTSupport _) (subset_mulTSupport _)))
    (isClosed_closure.union isClosed_closure)


@[to_additive]
theorem not_mem_mulTSupport_iff_eventuallyEq : x ∉ mulTSupport f ↔ f =ᶠ[𝓝 x] 1 := by
  simp_rw [mulTSupport, mem_closure_iff_nhds, not_forall, not_nonempty_iff_eq_empty, exists_prop,
    ← disjoint_iff_inter_eq_empty, disjoint_mulSupport_iff, eventuallyEq_iff_exists_mem]


@[to_additive]
theorem continuous_of_mulTSupport [TopologicalSpace β] {f : α → β}
    (hf : ∀ x ∈ mulTSupport f, ContinuousAt f x) : Continuous f :=
  continuous_iff_continuousAt.2 fun x => (em _).elim (hf x) fun hx =>
    (@continuousAt_const _ _ _ _ _ 1).congr (not_mem_mulTSupport_iff_eventuallyEq.mp hx).symm


/-- A function `f` *has compact multiplicative support* or is *compactly supported* if the closure
of the multiplicative support of `f` is compact. In a T₂ space this is equivalent to `f` being equal
to `1` outside a compact set. -/
@[to_additive " A function `f` *has compact support* or is *compactly supported* if the closure of
the support of `f` is compact. In a T₂ space this is equivalent to `f` being equal to `0` outside a
compact set. "]
def HasCompactMulSupport (f : α → β) : Prop :=
  IsCompact (mulTSupport f)


@[to_additive]
theorem hasCompactMulSupport_def : HasCompactMulSupport f ↔ IsCompact (closure (mulSupport f)) := by
  /-
    α : Type u_2
    β : Type u_4
    inst✝¹ : TopologicalSpace α
    inst✝ : One β
    f : α → β
    ⊢ Iff (HasCompactMulSupport f) (IsCompact (closure (Function.mulSupport f)))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem exists_compact_iff_hasCompactMulSupport [R1Space α] :
    (∃ K : Set α, IsCompact K ∧ ∀ x, x ∉ K → f x = 1) ↔ HasCompactMulSupport f := by
  simp_rw [← nmem_mulSupport, ← mem_compl_iff, ← subset_def, compl_subset_compl,
    hasCompactMulSupport_def, exists_isCompact_superset_iff]


@[to_additive]
theorem intro [R1Space α] {K : Set α} (hK : IsCompact K)
    (hfK : ∀ x, x ∉ K → f x = 1) : HasCompactMulSupport f :=
  exists_compact_iff_hasCompactMulSupport.mp ⟨K, hK, hfK⟩


@[to_additive]
theorem intro' {K : Set α} (hK : IsCompact K) (h'K : IsClosed K)
    (hfK : ∀ x, x ∉ K → f x = 1) : HasCompactMulSupport f := by
  have : mulTSupport f ⊆ K := by
    rw [← h'K.closure_eq]
    apply closure_mono (mulSupport_subset_iff'.2 hfK)
  /-
    α : Type u_2
    β : Type u_4
    inst✝¹ : TopologicalSpace α
    inst✝ : One β
    f : α → β
    K : Set α
    hK : IsCompact K
    h'K : IsClosed K
    hfK : ∀ (x : α), Not (Membership.mem K x) → Eq (f x) 1
    this : HasSubset.Subset (mulTSupport f) K
    ⊢ HasCompactMulSupport f
  -/
  exact IsCompact.of_isClosed_subset hK ( isClosed_mulTSupport f) this
  /-
    🎉 no goals
  -/


@[to_additive]
theorem of_mulSupport_subset_isCompact [R1Space α] {K : Set α}
    (hK : IsCompact K) (h : mulSupport f ⊆ K) : HasCompactMulSupport f :=
  hK.closure_of_subset h


@[to_additive]
theorem isCompact (hf : HasCompactMulSupport f) : IsCompact (mulTSupport f) :=
  hf


@[to_additive]
theorem _root_.hasCompactMulSupport_iff_eventuallyEq :
    HasCompactMulSupport f ↔ f =ᶠ[coclosedCompact α] 1 :=
  mem_coclosedCompact_iff.symm


@[to_additive]
theorem _root_.isCompact_range_of_mulSupport_subset_isCompact [TopologicalSpace β]
    (hf : Continuous f) {k : Set α} (hk : IsCompact k) (h'f : mulSupport f ⊆ k) :
    IsCompact (range f) := by
  /-
    α : Type u_2
    β : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : One β
    f : α → β
    inst✝ : TopologicalSpace β
    hf : Continuous f
    k : Set α
    hk : IsCompact k
    h'f : HasSubset.Subset (Function.mulSupport f) k
    ⊢ IsCompact (Set.range f)
  -/
  cases' range_eq_image_or_of_mulSupport_subset h'f with h2 h2 <;> rw [h2]
  /-
    case inl
    α : Type u_2
    β : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : One β
    f : α → β
    inst✝ : TopologicalSpace β
    hf : Continuous f
    k : Set α
    hk : IsCompact k
    h'f : HasSubset.Subset (Function.mulSupport f) k
    h2 : Eq (Set.range f) (Set.image f k)
    ⊢ IsCompact (Set.image f k)
  -/
  exacts [hk.image hf, (hk.image hf).insert 1]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem isCompact_range [TopologicalSpace β] (h : HasCompactMulSupport f)
    (hf : Continuous f) : IsCompact (range f) :=
  isCompact_range_of_mulSupport_subset_isCompact hf h (subset_mulTSupport f)


@[to_additive]
theorem mono' {f' : α → γ} (hf : HasCompactMulSupport f)
    (hff' : mulSupport f' ⊆ mulTSupport f) : HasCompactMulSupport f' :=
  IsCompact.of_isClosed_subset hf isClosed_closure <| closure_minimal hff' isClosed_closure


@[to_additive]
theorem mono {f' : α → γ} (hf : HasCompactMulSupport f)
    (hff' : mulSupport f' ⊆ mulSupport f) : HasCompactMulSupport f' :=
  hf.mono' <| hff'.trans subset_closure


@[to_additive]
theorem comp_left (hf : HasCompactMulSupport f) (hg : g 1 = 1) :
    HasCompactMulSupport (g ∘ f) :=
  hf.mono <| mulSupport_comp_subset hg f


@[to_additive]
theorem _root_.hasCompactMulSupport_comp_left (hg : ∀ {x}, g x = 1 ↔ x = 1) :
    HasCompactMulSupport (g ∘ f) ↔ HasCompactMulSupport f := by
  /-
    α : Type u_2
    β : Type u_4
    γ : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : One β
    inst✝ : One γ
    g : β → γ
    f : α → β
    hg : ∀ {x : β}, Iff (Eq (g x) 1) (Eq x 1)
    ⊢ Iff (HasCompactMulSupport (Function.comp g f)) (HasCompactMulSupport f)
  -/
  simp_rw [hasCompactMulSupport_def, mulSupport_comp_eq g (@hg) f]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem comp_isClosedEmbedding (hf : HasCompactMulSupport f) {g : α' → α}
    (hg : IsClosedEmbedding g) : HasCompactMulSupport (f ∘ g) := by
  /-
    α : Type u_2
    α' : Type u_3
    β : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace α'
    inst✝ : One β
    f : α → β
    hf : HasCompactMulSupport f
    g : α' → α
    hg : Topology.IsClosedEmbedding g
    ⊢ HasCompactMulSupport (Function.comp f g)
  -/
  rw [hasCompactMulSupport_def, Function.mulSupport_comp_eq_preimage]
  /-
    α : Type u_2
    α' : Type u_3
    β : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace α'
    inst✝ : One β
    f : α → β
    hf : HasCompactMulSupport f
    g : α' → α
    hg : Topology.IsClosedEmbedding g
    ⊢ IsCompact (closure (Set.preimage g (Function.mulSupport f)))
  -/
  refine IsCompact.of_isClosed_subset (hg.isCompact_preimage hf) isClosed_closure ?_
  /-
    α : Type u_2
    α' : Type u_3
    β : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace α'
    inst✝ : One β
    f : α → β
    hf : HasCompactMulSupport f
    g : α' → α
    hg : Topology.IsClosedEmbedding g
    ⊢ HasSubset.Subset (closure (Set.preimage g (Function.mulSupport f))) (Set.pre …
  -/
  rw [hg.isEmbedding.closure_eq_preimage_closure_image]
  /-
    α : Type u_2
    α' : Type u_3
    β : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace α'
    inst✝ : One β
    f : α → β
    hf : HasCompactMulSupport f
    g : α' → α
    hg : Topology.IsClosedEmbedding g
    ⊢ HasSubset.Subset (Set.preimage g (closure (Set.image g (Set.preimage g (Func …
  -/
  exact preimage_mono (closure_mono <| image_preimage_subset _ _)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")]
alias comp_closedEmbedding := comp_isClosedEmbedding


@[to_additive]
theorem comp₂_left (hf : HasCompactMulSupport f)
    (hf₂ : HasCompactMulSupport f₂) (hm : m 1 1 = 1) :
    HasCompactMulSupport fun x => m (f x) (f₂ x) := by
  /-
    α : Type u_2
    β : Type u_4
    γ : Type u_5
    δ : Type u_6
    inst✝³ : TopologicalSpace α
    inst✝² : One β
    inst✝¹ : One γ
    inst✝ : One δ
    f : α → β
    f₂ : α → γ
    m : β → γ → δ
    hf : HasCompactMulSupport f
    hf₂ : HasCompactMulSupport f₂
    hm : Eq (m 1 1) 1
    ⊢ HasCompactMulSupport fun x => m (f x) (f₂ x)
  -/
  rw [hasCompactMulSupport_iff_eventuallyEq] at hf hf₂ ⊢
  /-
    α : Type u_2
    β : Type u_4
    γ : Type u_5
    δ : Type u_6
    inst✝³ : TopologicalSpace α
    inst✝² : One β
    inst✝¹ : One γ
    inst✝ : One δ
    f : α → β
    f₂ : α → γ
    m : β → γ → δ
    hf : (Filter.coclosedCompact α).EventuallyEq f 1
    hf₂ : (Filter.coclosedCompact α).EventuallyEq f₂ 1
    hm : Eq (m 1 1) 1
    ⊢ (Filter.coclosedCompact α).EventuallyEq (fun x => m (f x) (f₂ x)) 1
  -/
  filter_upwards [hf, hf₂] with x hx hx₂
  /-
    case h
    α : Type u_2
    β : Type u_4
    γ : Type u_5
    δ : Type u_6
    inst✝³ : TopologicalSpace α
    inst✝² : One β
    inst✝¹ : One γ
    inst✝ : One δ
    f : α → β
    f₂ : α → γ
    m : β → γ → δ
    hf : (Filter.coclosedCompact α).EventuallyEq f 1
    hf₂ : (Filter.coclosedCompact α).EventuallyEq f₂ 1
    hm : Eq (m 1 1) 1
    x : α
    hx : Eq (f x) (1 x)
    hx₂ : Eq (f₂ x) (1 x)
    ⊢ Eq (m (f x) (f₂ x)) (1 x)
  -/
  simp_rw [hx, hx₂, Pi.one_apply, hm]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma isCompact_preimage [TopologicalSpace β]
    (h'f : HasCompactMulSupport f) (hf : Continuous f) {k : Set β} (hk : IsClosed k)
    (h'k : 1 ∉ k) : IsCompact (f ⁻¹' k) := by
  /-
    α : Type u_2
    β : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : One β
    f : α → β
    inst✝ : TopologicalSpace β
    h'f : HasCompactMulSupport f
    hf : Continuous f
    k : Set β
    hk : IsClosed k
    h'k : Not (Membership.mem k 1)
    ⊢ IsCompact (Set.preimage f k)
  -/
  apply IsCompact.of_isClosed_subset h'f (hk.preimage hf) (fun x hx ↦ ?_)
  /-
    α : Type u_2
    β : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : One β
    f : α → β
    inst✝ : TopologicalSpace β
    h'f : HasCompactMulSupport f
    hf : Continuous f
    k : Set β
    hk : IsClosed k
    h'k : Not (Membership.mem k 1)
    x : α
    hx : Membership.mem (Set.preimage f k) x
    ⊢ Membership.mem (mulTSupport f) x
  -/
  apply subset_mulTSupport
  /-
    case a
    α : Type u_2
    β : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : One β
    f : α → β
    inst✝ : TopologicalSpace β
    h'f : HasCompactMulSupport f
    hf : Continuous f
    k : Set β
    hk : IsClosed k
    h'k : Not (Membership.mem k 1)
    x : α
    hx : Membership.mem (Set.preimage f k) x
    ⊢ Membership.mem (Function.mulSupport f) x
  -/
  aesop
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulTSupport_extend_one_subset :
    mulTSupport (g.extend f 1) ⊆ g '' mulTSupport f :=
  (hf.image cont).isClosed.closure_subset_iff.mpr <|
    mulSupport_extend_one_subset.trans (image_subset g subset_closure)


@[to_additive]
theorem extend_one : HasCompactMulSupport (g.extend f 1) :=
  HasCompactMulSupport.of_mulSupport_subset_isCompact (hf.image cont)
    (subset_closure.trans <| hf.mulTSupport_extend_one_subset cont)


@[to_additive]
theorem mulTSupport_extend_one (inj : g.Injective) :
    mulTSupport (g.extend f 1) = g '' mulTSupport f :=
  (hf.mulTSupport_extend_one_subset cont).antisymm <|
    (image_closure_subset_closure_image cont).trans
      (closure_mono (mulSupport_extend_one inj).superset)


@[to_additive]
theorem continuous_extend_one [TopologicalSpace β] {U : Set α'} (hU : IsOpen U) {f : U → β}
    (cont : Continuous f) (supp : HasCompactMulSupport f) :
    Continuous (Subtype.val.extend f 1) :=
  continuous_of_mulTSupport fun x h ↦ by
    rw [show x = ↑(⟨x, Subtype.coe_image_subset _ _
      (supp.mulTSupport_extend_one_subset continuous_subtype_val h)⟩ : U) by rfl,
      ← (hU.isOpenEmbedding_subtypeVal).continuousAt_iff, extend_comp Subtype.val_injective]
    /-
      α' : Type u_3
      β : Type u_4
      inst✝³ : TopologicalSpace α'
      inst✝² : One β
      inst✝¹ : T2Space α'
      inst✝ : TopologicalSpace β
      U : Set α'
      hU : IsOpen U
      f : ↑U → β
      cont : Continuous f
      supp : HasCompactMulSupport f
      x : α'
      h : Membership.mem (mulTSupport (Function.extend Subtype.val f 1)) x
      ⊢ ContinuousAt f ⟨x, ⋯⟩
    -/
    exact cont.continuousAt
    /-
      🎉 no goals
    -/


/-- If `f` has compact multiplicative support, then `f` tends to 1 at infinity. -/
@[to_additive "If `f` has compact support, then `f` tends to zero at infinity."]
theorem is_one_at_infty {f : α → γ} [TopologicalSpace γ]
    (h : HasCompactMulSupport f) : Tendsto f (cocompact α) (𝓝 1) := by
  /-
    α : Type u_2
    γ : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : One γ
    f : α → γ
    inst✝ : TopologicalSpace γ
    h : HasCompactMulSupport f
    ⊢ Filter.Tendsto f (Filter.cocompact α) (nhds 1)
  -/
  intro N hN
  /-
    α : Type u_2
    γ : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : One γ
    f : α → γ
    inst✝ : TopologicalSpace γ
    h : HasCompactMulSupport f
    N : Set γ
    hN : Membership.mem (nhds 1) N
    ⊢ Membership.mem (Filter.map f (Filter.cocompact α)) N
  -/
  rw [mem_map, mem_cocompact']
  /-
    α : Type u_2
    γ : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : One γ
    f : α → γ
    inst✝ : TopologicalSpace γ
    h : HasCompactMulSupport f
    N : Set γ
    hN : Membership.mem (nhds 1) N
    ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset (HasCompl.compl (Set.pre …
  -/
  refine ⟨mulTSupport f, h.isCompact, ?_⟩
  /-
    α : Type u_2
    γ : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : One γ
    f : α → γ
    inst✝ : TopologicalSpace γ
    h : HasCompactMulSupport f
    N : Set γ
    hN : Membership.mem (nhds 1) N
    ⊢ HasSubset.Subset (HasCompl.compl (Set.preimage f N)) (mulTSupport f)
  -/
  rw [compl_subset_comm]
  /-
    α : Type u_2
    γ : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : One γ
    f : α → γ
    inst✝ : TopologicalSpace γ
    h : HasCompactMulSupport f
    N : Set γ
    hN : Membership.mem (nhds 1) N
    ⊢ HasSubset.Subset (HasCompl.compl (mulTSupport f)) (Set.preimage f N)
  -/
  intro v hv
  /-
    α : Type u_2
    γ : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : One γ
    f : α → γ
    inst✝ : TopologicalSpace γ
    h : HasCompactMulSupport f
    N : Set γ
    hN : Membership.mem (nhds 1) N
    v : α
    hv : Membership.mem (HasCompl.compl (mulTSupport f)) v
    ⊢ Membership.mem (Set.preimage f N) v
  -/
  rw [mem_preimage, image_eq_one_of_nmem_mulTSupport hv]
  /-
    α : Type u_2
    γ : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : One γ
    f : α → γ
    inst✝ : TopologicalSpace γ
    h : HasCompactMulSupport f
    N : Set γ
    hN : Membership.mem (nhds 1) N
    v : α
    hv : Membership.mem (HasCompl.compl (mulTSupport f)) v
    ⊢ Membership.mem N 1
  -/
  exact mem_of_mem_nhds hN
  /-
    🎉 no goals
  -/


/-- In a compact space `α`, any function has compact support. -/
@[to_additive]
theorem HasCompactMulSupport.of_compactSpace (f : α → γ) :
    HasCompactMulSupport f :=
  IsCompact.of_isClosed_subset isCompact_univ (isClosed_mulTSupport f)
    (Set.subset_univ (mulTSupport f))


@[to_additive]
theorem HasCompactMulSupport.mul (hf : HasCompactMulSupport f) (hf' : HasCompactMulSupport f') :
    HasCompactMulSupport (f * f') := hf.comp₂_left hf' (mul_one 1)


@[to_additive, simp]
protected lemma HasCompactMulSupport.one {α β : Type*} [TopologicalSpace α] [One β] :
    HasCompactMulSupport (1 : α → β) := by
  /-
    α : Type u_9
    β : Type u_10
    inst✝¹ : TopologicalSpace α
    inst✝ : One β
    ⊢ HasCompactMulSupport 1
  -/
  simp [HasCompactMulSupport, mulTSupport]
  /-
    🎉 no goals
  -/


@[to_additive]
protected lemma HasCompactMulSupport.inv' {α β : Type*} [TopologicalSpace α] [DivisionMonoid β]
    {f : α → β} (hf : HasCompactMulSupport f) :
    HasCompactMulSupport (f⁻¹) := by
  /-
    α : Type u_9
    β : Type u_10
    inst✝¹ : TopologicalSpace α
    inst✝ : DivisionMonoid β
    f : α → β
    hf : HasCompactMulSupport f
    ⊢ HasCompactMulSupport (Inv.inv f)
  -/
  simpa only [HasCompactMulSupport, mulTSupport, mulSupport_inv'] using hf
  /-
    🎉 no goals
  -/


theorem HasCompactSupport.smul_left (hf : HasCompactSupport f') : HasCompactSupport (f • f') := by
  /-
    α : Type u_2
    M : Type u_7
    R : Type u_8
    inst✝² : TopologicalSpace α
    inst✝¹ : Zero M
    inst✝ : SMulZeroClass R M
    f : α → R
    f' : α → M
    hf : HasCompactSupport f'
    ⊢ HasCompactSupport (HSMul.hSMul f f')
  -/
  rw [hasCompactSupport_iff_eventuallyEq] at hf ⊢
  /-
    α : Type u_2
    M : Type u_7
    R : Type u_8
    inst✝² : TopologicalSpace α
    inst✝¹ : Zero M
    inst✝ : SMulZeroClass R M
    f : α → R
    f' : α → M
    hf : (Filter.coclosedCompact α).EventuallyEq f' 0
    ⊢ (Filter.coclosedCompact α).EventuallyEq (HSMul.hSMul f f') 0
  -/
  exact hf.mono fun x hx => by simp_rw [Pi.smul_apply', hx, Pi.zero_apply, smul_zero]
  /-
    🎉 no goals
  -/


theorem HasCompactSupport.smul_right (hf : HasCompactSupport f) : HasCompactSupport (f • f') := by
  /-
    α : Type u_2
    M : Type u_7
    R : Type u_8
    inst✝³ : TopologicalSpace α
    inst✝² : Zero R
    inst✝¹ : Zero M
    inst✝ : SMulWithZero R M
    f : α → R
    f' : α → M
    hf : HasCompactSupport f
    ⊢ HasCompactSupport (HSMul.hSMul f f')
  -/
  rw [hasCompactSupport_iff_eventuallyEq] at hf ⊢
  /-
    α : Type u_2
    M : Type u_7
    R : Type u_8
    inst✝³ : TopologicalSpace α
    inst✝² : Zero R
    inst✝¹ : Zero M
    inst✝ : SMulWithZero R M
    f : α → R
    f' : α → M
    hf : (Filter.coclosedCompact α).EventuallyEq f 0
    ⊢ (Filter.coclosedCompact α).EventuallyEq (HSMul.hSMul f f') 0
  -/
  exact hf.mono fun x hx => by simp_rw [Pi.smul_apply', hx, Pi.zero_apply, zero_smul]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-05")]
alias HasCompactSupport.smul_left' := HasCompactSupport.smul_left


theorem HasCompactSupport.mul_right (hf : HasCompactSupport f) : HasCompactSupport (f * f') := by
  /-
    α : Type u_2
    β : Type u_4
    inst✝¹ : TopologicalSpace α
    inst✝ : MulZeroClass β
    f f' : α → β
    hf : HasCompactSupport f
    ⊢ HasCompactSupport (HMul.hMul f f')
  -/
  rw [hasCompactSupport_iff_eventuallyEq] at hf ⊢
  /-
    α : Type u_2
    β : Type u_4
    inst✝¹ : TopologicalSpace α
    inst✝ : MulZeroClass β
    f f' : α → β
    hf : (Filter.coclosedCompact α).EventuallyEq f 0
    ⊢ (Filter.coclosedCompact α).EventuallyEq (HMul.hMul f f') 0
  -/
  exact hf.mono fun x hx => by simp_rw [Pi.mul_apply, hx, Pi.zero_apply, zero_mul]
  /-
    🎉 no goals
  -/


theorem HasCompactSupport.mul_left (hf : HasCompactSupport f') : HasCompactSupport (f * f') := by
  /-
    α : Type u_2
    β : Type u_4
    inst✝¹ : TopologicalSpace α
    inst✝ : MulZeroClass β
    f f' : α → β
    hf : HasCompactSupport f'
    ⊢ HasCompactSupport (HMul.hMul f f')
  -/
  rw [hasCompactSupport_iff_eventuallyEq] at hf ⊢
  /-
    α : Type u_2
    β : Type u_4
    inst✝¹ : TopologicalSpace α
    inst✝ : MulZeroClass β
    f f' : α → β
    hf : (Filter.coclosedCompact α).EventuallyEq f' 0
    ⊢ (Filter.coclosedCompact α).EventuallyEq (HMul.hMul f f') 0
  -/
  exact hf.mono fun x hx => by simp_rw [Pi.mul_apply, hx, Pi.zero_apply, mul_zero]
  /-
    🎉 no goals
  -/


protected theorem HasCompactSupport.abs {f : α → β} (hf : HasCompactSupport f) :
    HasCompactSupport |f| :=
  hf.comp_left (g := abs) abs_zero


/-- If a family of functions `f` has locally-finite multiplicative support, subordinate to a family
of open sets, then for any point we can find a neighbourhood on which only finitely-many members of
`f` are not equal to 1. -/
@[to_additive " If a family of functions `f` has locally-finite support, subordinate to a family of
open sets, then for any point we can find a neighbourhood on which only finitely-many members of `f`
are non-zero. "]
theorem LocallyFinite.exists_finset_nhd_mulSupport_subset {U : ι → Set X} [One R] {f : ι → X → R}
    (hlf : LocallyFinite fun i => mulSupport (f i)) (hso : ∀ i, mulTSupport (f i) ⊆ U i)
    (ho : ∀ i, IsOpen (U i)) (x : X) :
    ∃ (is : Finset ι), ∃ n, n ∈ 𝓝 x ∧ (n ⊆ ⋂ i ∈ is, U i) ∧
      ∀ z ∈ n, (mulSupport fun i => f i z) ⊆ is := by
  /-
    X : Type u_1
    R : Type u_8
    ι : Type u_9
    inst✝¹ : TopologicalSpace X
    U : ι → Set X
    inst✝ : One R
    f : ι → X → R
    hlf : LocallyFinite fun i => Function.mulSupport (f i)
    hso : ∀ (i : ι), HasSubset.Subset (mulTSupport (f i)) (U i)
    ho : ∀ (i : ι), IsOpen (U i)
    x : X
    ⊢ Exists fun is => Exists fun n => And (Membership.mem (nhds x) n) (And (HasSu …
  -/
  obtain ⟨n, hn, hnf⟩ := hlf x
  classical
    let is := hnf.toFinset.filter fun i => x ∈ U i
    let js := hnf.toFinset.filter fun j => x ∉ U j
    refine
      ⟨is, (n ∩ ⋂ j ∈ js, (mulTSupport (f j))ᶜ) ∩ ⋂ i ∈ is, U i, inter_mem (inter_mem hn ?_) ?_,
        inter_subset_right, fun z hz => ?_⟩
    · exact (biInter_finset_mem js).mpr fun j hj => IsClosed.compl_mem_nhds (isClosed_mulTSupport _)
        (Set.not_mem_subset (hso j) (Finset.mem_filter.mp hj).2)
    · exact (biInter_finset_mem is).mpr fun i hi => (ho i).mem_nhds (Finset.mem_filter.mp hi).2
    · have hzn : z ∈ n := by
        rw [inter_assoc] at hz
        exact mem_of_mem_inter_left hz
      replace hz := mem_of_mem_inter_right (mem_of_mem_inter_left hz)
      simp only [js, Finset.mem_filter, Finite.mem_toFinset, mem_setOf_eq, mem_iInter,
        and_imp] at hz
      suffices (mulSupport fun i => f i z) ⊆ hnf.toFinset by
        refine hnf.toFinset.subset_coe_filter_of_subset_forall _ this fun i hi => ?_
        specialize hz i ⟨z, ⟨hi, hzn⟩⟩
        contrapose hz
        simp [hz, subset_mulTSupport (f i) hi]
      intro i hi
      simp only [Finite.coe_toFinset, mem_setOf_eq]
      exact ⟨z, ⟨hi, hzn⟩⟩


@[to_additive]
theorem locallyFinite_mulSupport_iff [CommMonoid M] {f : ι → X → M} :
    (LocallyFinite fun i ↦ mulSupport <| f i) ↔ LocallyFinite fun i ↦ mulTSupport <| f i :=
  ⟨LocallyFinite.closure, fun H ↦ H.subset fun _ ↦ subset_closure⟩


theorem LocallyFinite.smul_left [Zero R] [Zero M] [SMulWithZero R M]
    {s : ι → X → R} (h : LocallyFinite fun i ↦ support <| s i) (f : ι → X → M) :
    LocallyFinite fun i ↦ support <| s i • f i :=
                                      /-
                                        X : Type u_1
                                        M : Type u_7
                                        R : Type u_8
                                        ι : Type u_9
                                        inst✝³ : TopologicalSpace X
                                        inst✝² : Zero R
                                        inst✝¹ : Zero M
                                        inst✝ : SMulWithZero R M
                                        s : ι → X → R
                                        h✝ : LocallyFinite fun i => Function.support (s i)
                                        f : ι → X → M
                                        i : ι
                                        x : X
                                        h : Eq (s i x) 0
                                        ⊢ Eq (HSMul.hSMul (s i) (f i) x) 0
                                      -/
  h.subset fun i x ↦ mt <| fun h ↦ by rw [Pi.smul_apply', h, zero_smul]
                                      /-
                                        🎉 no goals
                                      -/


theorem LocallyFinite.smul_right [Zero M] [SMulZeroClass R M]
    {f : ι → X → M} (h : LocallyFinite fun i ↦ support <| f i) (s : ι → X → R) :
    LocallyFinite fun i ↦ support <| s i • f i :=
                                      /-
                                        X : Type u_1
                                        M : Type u_7
                                        R : Type u_8
                                        ι : Type u_9
                                        inst✝² : TopologicalSpace X
                                        inst✝¹ : Zero M
                                        inst✝ : SMulZeroClass R M
                                        f : ι → X → M
                                        h✝ : LocallyFinite fun i => Function.support (f i)
                                        s : ι → X → R
                                        i : ι
                                        x : X
                                        h : Eq (f i x) 0
                                        ⊢ Eq (HSMul.hSMul (s i) (f i) x) 0
                                      -/
  h.subset fun i x ↦ mt <| fun h ↦ by rw [Pi.smul_apply', h, smul_zero]
                                      /-
                                        🎉 no goals
                                      -/


