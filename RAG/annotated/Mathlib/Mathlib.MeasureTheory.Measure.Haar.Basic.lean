/-- The index or Haar covering number or ratio of `K` w.r.t. `V`, denoted `(K : V)`:
  it is the smallest number of (left) translates of `V` that is necessary to cover `K`.
  It is defined to be 0 if no finite number of translates cover `K`. -/
@[to_additive addIndex "additive version of `MeasureTheory.Measure.haar.index`"]
noncomputable def index (K V : Set G) : ℕ :=
  sInf <| Finset.card '' { t : Finset G | K ⊆ ⋃ g ∈ t, (fun h => g * h) ⁻¹' V }


@[to_additive addIndex_empty]
theorem index_empty {V : Set G} : index ∅ V = 0 := by
  /-
    G : Type u_1
    inst✝ : Group G
    V : Set G
    ⊢ Eq (MeasureTheory.Measure.haar.index EmptyCollection.emptyCollection V) 0
  -/
  simp only [index, Nat.sInf_eq_zero]; left; use ∅
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    V : Set G
    ⊢ And (Membership.mem (setOf fun t => HasSubset.Subset EmptyCollection.emptyCo …
  -/
  simp only [Finset.card_empty, empty_subset, mem_setOf_eq, eq_self_iff_true, and_self_iff]
  /-
    🎉 no goals
  -/


/-- `prehaar K₀ U K` is a weighted version of the index, defined as `(K : U)/(K₀ : U)`.
  In the applications `K₀` is compact with non-empty interior, `U` is open containing `1`,
  and `K` is any compact set.
  The argument `K` is a (bundled) compact set, so that we can consider `prehaar K₀ U` as an
  element of `haarProduct` (below). -/
@[to_additive "additive version of `MeasureTheory.Measure.haar.prehaar`"]
noncomputable def prehaar (K₀ U : Set G) (K : Compacts G) : ℝ :=
  (index (K : Set G) U : ℝ) / index K₀ U


@[to_additive]
theorem prehaar_empty (K₀ : PositiveCompacts G) {U : Set G} : prehaar (K₀ : Set G) U ⊥ = 0 := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    U : Set G
    ⊢ Eq (MeasureTheory.Measure.haar.prehaar (↑K₀) U Bot.bot) 0
  -/
  rw [prehaar, Compacts.coe_bot, index_empty, Nat.cast_zero, zero_div]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prehaar_nonneg (K₀ : PositiveCompacts G) {U : Set G} (K : Compacts G) :
                                       /-
                                         G : Type u_1
                                         inst✝¹ : Group G
                                         inst✝ : TopologicalSpace G
                                         K₀ : TopologicalSpace.PositiveCompacts G
                                         U : Set G
                                         K : TopologicalSpace.Compacts G
                                         ⊢ LE.le 0 (MeasureTheory.Measure.haar.prehaar (↑K₀) U K)
                                       -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    0 ≤ prehaar (K₀ : Set G) U K := by apply div_nonneg <;> norm_cast <;> apply zero_le
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- `haarProduct K₀` is the product of intervals `[0, (K : K₀)]`, for all compact sets `K`.
  For all `U`, we can show that `prehaar K₀ U ∈ haarProduct K₀`. -/
@[to_additive "additive version of `MeasureTheory.Measure.haar.haarProduct`"]
def haarProduct (K₀ : Set G) : Set (Compacts G → ℝ) :=
  pi univ fun K => Icc 0 <| index (K : Set G) K₀


@[to_additive (attr := simp)]
theorem mem_prehaar_empty {K₀ : Set G} {f : Compacts G → ℝ} :
    f ∈ haarProduct K₀ ↔ ∀ K : Compacts G, f K ∈ Icc (0 : ℝ) (index (K : Set G) K₀) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    K₀ : Set G
    f : TopologicalSpace.Compacts G → Real
    ⊢ Iff (Membership.mem (MeasureTheory.Measure.haar.haarProduct K₀) f) (∀ (K : T …
  -/
  simp only [haarProduct, Set.pi, forall_prop_of_true, mem_univ, mem_setOf_eq]
  /-
    🎉 no goals
  -/


/-- The closure of the collection of elements of the form `prehaar K₀ U`,
  for `U` open neighbourhoods of `1`, contained in `V`. The closure is taken in the space
  `compacts G → ℝ`, with the topology of pointwise convergence.
  We show that the intersection of all these sets is nonempty, and the Haar measure
  on compact sets is defined to be an element in the closure of this intersection. -/
@[to_additive "additive version of `MeasureTheory.Measure.haar.clPrehaar`"]
def clPrehaar (K₀ : Set G) (V : OpenNhdsOf (1 : G)) : Set (Compacts G → ℝ) :=
  closure <| prehaar K₀ '' { U : Set G | U ⊆ V.1 ∧ IsOpen U ∧ (1 : G) ∈ U }


/-- If `K` is compact and `V` has nonempty interior, then the index `(K : V)` is well-defined,
  there is a finite set `t` satisfying the desired properties. -/
@[to_additive addIndex_defined
"If `K` is compact and `V` has nonempty interior, then the index `(K : V)` is well-defined, there is
a finite set `t` satisfying the desired properties."]
theorem index_defined {K V : Set G} (hK : IsCompact K) (hV : (interior V).Nonempty) :
    ∃ n : ℕ, n ∈ Finset.card '' { t : Finset G | K ⊆ ⋃ g ∈ t, (fun h => g * h) ⁻¹' V } := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K V : Set G
    hK : IsCompact K
    hV : (interior V).Nonempty
    ⊢ Exists fun n => Membership.mem (Set.image Finset.card (setOf fun t => HasSub …
  -/
  rcases compact_covered_by_mul_left_translates hK hV with ⟨t, ht⟩; exact ⟨t.card, t, ht, rfl⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[to_additive addIndex_elim]
theorem index_elim {K V : Set G} (hK : IsCompact K) (hV : (interior V).Nonempty) :
    ∃ t : Finset G, (K ⊆ ⋃ g ∈ t, (fun h => g * h) ⁻¹' V) ∧ Finset.card t = index K V := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K V : Set G
    hK : IsCompact K
    hV : (interior V).Nonempty
    ⊢ Exists fun t => And (HasSubset.Subset K (Set.iUnion fun g => Set.iUnion fun  …
  -/
  have := Nat.sInf_mem (index_defined hK hV); rwa [mem_image] at this
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive le_addIndex_mul]
theorem le_index_mul (K₀ : PositiveCompacts G) (K : Compacts G) {V : Set G}
    (hV : (interior V).Nonempty) :
    index (K : Set G) V ≤ index (K : Set G) K₀ * index (K₀ : Set G) V := by
  classical
  obtain ⟨s, h1s, h2s⟩ := index_elim K.isCompact K₀.interior_nonempty
  obtain ⟨t, h1t, h2t⟩ := index_elim K₀.isCompact hV
  rw [← h2s, ← h2t, mul_comm]
  refine le_trans ?_ Finset.card_mul_le
  apply Nat.sInf_le; refine ⟨_, ?_, rfl⟩; rw [mem_setOf_eq]; refine Subset.trans h1s ?_
  apply iUnion₂_subset; intro g₁ hg₁; rw [preimage_subset_iff]; intro g₂ hg₂
  have := h1t hg₂
  rcases this with ⟨_, ⟨g₃, rfl⟩, A, ⟨hg₃, rfl⟩, h2V⟩; rw [mem_preimage, ← mul_assoc] at h2V
  exact mem_biUnion (Finset.mul_mem_mul hg₃ hg₁) h2V


@[to_additive addIndex_pos]
theorem index_pos (K : PositiveCompacts G) {V : Set G} (hV : (interior V).Nonempty) :
    0 < index (K : Set G) V := by
  classical
  rw [index, Nat.sInf_def, Nat.find_pos, mem_image]
  · rintro ⟨t, h1t, h2t⟩; rw [Finset.card_eq_zero] at h2t; subst h2t
    obtain ⟨g, hg⟩ := K.interior_nonempty
    show g ∈ (∅ : Set G)
    convert h1t (interior_subset hg); symm
    simp only [Finset.not_mem_empty, iUnion_of_empty, iUnion_empty]
  · exact index_defined K.isCompact hV


@[to_additive addIndex_mono]
theorem index_mono {K K' V : Set G} (hK' : IsCompact K') (h : K ⊆ K') (hV : (interior V).Nonempty) :
    index K V ≤ index K' V := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K K' V : Set G
    hK' : IsCompact K'
    h : HasSubset.Subset K K'
    hV : (interior V).Nonempty
    ⊢ LE.le (MeasureTheory.Measure.haar.index K V) (MeasureTheory.Measure.haar.ind …
  -/
  rcases index_elim hK' hV with ⟨s, h1s, h2s⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K K' V : Set G
    hK' : IsCompact K'
    h : HasSubset.Subset K K'
    hV : (interior V).Nonempty
    s : Finset G
    h1s : HasSubset.Subset K' (Set.iUnion fun g => Set.iUnion fun h => Set.preimag …
    h2s : Eq s.card (MeasureTheory.Measure.haar.index K' V)
    ⊢ LE.le (MeasureTheory.Measure.haar.index K V) (MeasureTheory.Measure.haar.ind …
  -/
  apply Nat.sInf_le; rw [mem_image]; exact ⟨s, Subset.trans h h1s, h2s⟩
                                     /-
                                       🎉 no goals
                                     -/


@[to_additive addIndex_union_le]
theorem index_union_le (K₁ K₂ : Compacts G) {V : Set G} (hV : (interior V).Nonempty) :
    index (K₁.1 ∪ K₂.1) V ≤ index K₁.1 V + index K₂.1 V := by
  classical
  rcases index_elim K₁.2 hV with ⟨s, h1s, h2s⟩
  rcases index_elim K₂.2 hV with ⟨t, h1t, h2t⟩
  rw [← h2s, ← h2t]
  refine le_trans ?_ (Finset.card_union_le _ _)
  apply Nat.sInf_le; refine ⟨_, ?_, rfl⟩; rw [mem_setOf_eq]
  apply union_subset <;> refine Subset.trans (by assumption) ?_ <;>
    apply biUnion_subset_biUnion_left <;> intro g hg <;> simp only [mem_def] at hg <;>
    simp only [mem_def, Multiset.mem_union, Finset.union_val, hg, or_true, true_or]


@[to_additive addIndex_union_eq]
theorem index_union_eq (K₁ K₂ : Compacts G) {V : Set G} (hV : (interior V).Nonempty)
    (h : Disjoint (K₁.1 * V⁻¹) (K₂.1 * V⁻¹)) :
    index (K₁.1 ∪ K₂.1) V = index K₁.1 V + index K₂.1 V := by
  classical
  apply le_antisymm (index_union_le K₁ K₂ hV)
  rcases index_elim (K₁.2.union K₂.2) hV with ⟨s, h1s, h2s⟩; rw [← h2s]
  have :
    ∀ K : Set G,
      (K ⊆ ⋃ g ∈ s, (fun h => g * h) ⁻¹' V) →
        index K V ≤ (s.filter fun g => ((fun h : G => g * h) ⁻¹' V ∩ K).Nonempty).card := by
    intro K hK; apply Nat.sInf_le; refine ⟨_, ?_, rfl⟩; rw [mem_setOf_eq]
    intro g hg; rcases hK hg with ⟨_, ⟨g₀, rfl⟩, _, ⟨h1g₀, rfl⟩, h2g₀⟩
    simp only [mem_preimage] at h2g₀
    simp only [mem_iUnion]; use g₀; constructor; swap
    · simp only [Finset.mem_filter, h1g₀, true_and]; use g
      simp only [hg, h2g₀, mem_inter_iff, mem_preimage, and_self_iff]
    exact h2g₀
  refine
    le_trans
      (add_le_add (this K₁.1 <| Subset.trans subset_union_left h1s)
        (this K₂.1 <| Subset.trans subset_union_right h1s)) ?_
  rw [← Finset.card_union_of_disjoint, Finset.filter_union_right]
  · exact s.card_filter_le _
  apply Finset.disjoint_filter.mpr
  rintro g₁ _ ⟨g₂, h1g₂, h2g₂⟩ ⟨g₃, h1g₃, h2g₃⟩
  simp only [mem_preimage] at h1g₃ h1g₂
  refine h.le_bot (?_ : g₁⁻¹ ∈ _)
  constructor <;> simp only [Set.mem_inv, Set.mem_mul, exists_exists_and_eq_and, exists_and_left]
  · refine ⟨_, h2g₂, (g₁ * g₂)⁻¹, ?_, ?_⟩
    · simp only [inv_inv, h1g₂]
    · simp only [mul_inv_rev, mul_inv_cancel_left]
  · refine ⟨_, h2g₃, (g₁ * g₃)⁻¹, ?_, ?_⟩
    · simp only [inv_inv, h1g₃]
    · simp only [mul_inv_rev, mul_inv_cancel_left]


@[to_additive add_left_addIndex_le]
theorem mul_left_index_le {K : Set G} (hK : IsCompact K) {V : Set G} (hV : (interior V).Nonempty)
    (g : G) : index ((fun h => g * h) '' K) V ≤ index K V := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    V : Set G
    hV : (interior V).Nonempty
    g : G
    ⊢ LE.le (MeasureTheory.Measure.haar.index (Set.image (fun h => HMul.hMul g h)  …
  -/
  rcases index_elim hK hV with ⟨s, h1s, h2s⟩; rw [← h2s]
  /-
    case intro.intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    V : Set G
    hV : (interior V).Nonempty
    g : G
    s : Finset G
    h1s : HasSubset.Subset K (Set.iUnion fun g => Set.iUnion fun h => Set.preimage …
    h2s : Eq s.card (MeasureTheory.Measure.haar.index K V)
    ⊢ LE.le (MeasureTheory.Measure.haar.index (Set.image (fun h => HMul.hMul g h)  …
  -/
  apply Nat.sInf_le; rw [mem_image]
  /-
    case intro.intro.hm
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    V : Set G
    hV : (interior V).Nonempty
    g : G
    s : Finset G
    h1s : HasSubset.Subset K (Set.iUnion fun g => Set.iUnion fun h => Set.preimage …
    h2s : Eq s.card (MeasureTheory.Measure.haar.index K V)
    ⊢ Exists fun x => And (Membership.mem (setOf fun t => HasSubset.Subset (Set.im …
  -/
  refine ⟨s.map (Equiv.mulRight g⁻¹).toEmbedding, ?_, Finset.card_map _⟩
  /-
    case intro.intro.hm
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    V : Set G
    hV : (interior V).Nonempty
    g : G
    s : Finset G
    h1s : HasSubset.Subset K (Set.iUnion fun g => Set.iUnion fun h => Set.preimage …
    h2s : Eq s.card (MeasureTheory.Measure.haar.index K V)
    ⊢ Membership.mem (setOf fun t => HasSubset.Subset (Set.image (fun h => HMul.hM …
  -/
  simp only [mem_setOf_eq]; refine Subset.trans (image_subset _ h1s) ?_
  /-
    case intro.intro.hm
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    V : Set G
    hV : (interior V).Nonempty
    g : G
    s : Finset G
    h1s : HasSubset.Subset K (Set.iUnion fun g => Set.iUnion fun h => Set.preimage …
    h2s : Eq s.card (MeasureTheory.Measure.haar.index K V)
    ⊢ HasSubset.Subset (Set.image (fun h => HMul.hMul g h) (Set.iUnion fun g => Se …
  -/
  rintro _ ⟨g₁, ⟨_, ⟨g₂, rfl⟩, ⟨_, ⟨hg₂, rfl⟩, hg₁⟩⟩, rfl⟩
  /-
    case intro.intro.hm.intro.intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    V : Set G
    hV : (interior V).Nonempty
    g : G
    s : Finset G
    h1s : HasSubset.Subset K (Set.iUnion fun g => Set.iUnion fun h => Set.preimage …
    h2s : Eq s.card (MeasureTheory.Measure.haar.index K V)
    g₁ g₂ : G
    hg₂ : Membership.mem s g₂
    hg₁ : Membership.mem ((fun h => Set.preimage (fun h => HMul.hMul g₂ h) V) hg₂) …
    ⊢ Membership.mem (Set.iUnion fun g_1 => Set.iUnion fun h => Set.preimage (fun  …
  -/
  simp only [mem_preimage] at hg₁
  simp only [exists_prop, mem_iUnion, Finset.mem_map, Equiv.coe_mulRight,
    exists_exists_and_eq_and, mem_preimage, Equiv.toEmbedding_apply]
  /-
    case intro.intro.hm.intro.intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    V : Set G
    hV : (interior V).Nonempty
    g : G
    s : Finset G
    h1s : HasSubset.Subset K (Set.iUnion fun g => Set.iUnion fun h => Set.preimage …
    h2s : Eq s.card (MeasureTheory.Measure.haar.index K V)
    g₁ g₂ : G
    hg₂ : Membership.mem s g₂
    hg₁ : Membership.mem V (HMul.hMul g₂ g₁)
    ⊢ Exists fun a => And (Membership.mem s a) (Membership.mem V (HMul.hMul (HMul. …
  -/
  refine ⟨_, hg₂, ?_⟩; simp only [mul_assoc, hg₁, inv_mul_cancel_left]
                       /-
                         🎉 no goals
                       -/


@[to_additive is_left_invariant_addIndex]
theorem is_left_invariant_index {K : Set G} (hK : IsCompact K) (g : G) {V : Set G}
    (hV : (interior V).Nonempty) : index ((fun h => g * h) '' K) V = index K V := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    g : G
    V : Set G
    hV : (interior V).Nonempty
    ⊢ Eq (MeasureTheory.Measure.haar.index (Set.image (fun h => HMul.hMul g h) K)  …
  -/
  refine le_antisymm (mul_left_index_le hK hV g) ?_
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    g : G
    V : Set G
    hV : (interior V).Nonempty
    ⊢ LE.le (MeasureTheory.Measure.haar.index K V) (MeasureTheory.Measure.haar.ind …
  -/
  convert mul_left_index_le (hK.image <| continuous_mul_left g) hV g⁻¹
  /-
    case h.e'_3.h.e'_3
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    g : G
    V : Set G
    hV : (interior V).Nonempty
    ⊢ Eq K (Set.image (fun h => HMul.hMul (Inv.inv g) h) (Set.image (fun b => HMul …
  -/
  rw [image_image]; symm; convert image_id' _ with h; apply inv_mul_cancel_left
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive add_prehaar_le_addIndex]
theorem prehaar_le_index (K₀ : PositiveCompacts G) {U : Set G} (K : Compacts G)
    (hU : (interior U).Nonempty) : prehaar (K₀ : Set G) U K ≤ index (K : Set G) K₀ := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    U : Set G
    K : TopologicalSpace.Compacts G
    hU : (interior U).Nonempty
    ⊢ LE.le (MeasureTheory.Measure.haar.prehaar (↑K₀) U K) ↑(MeasureTheory.Measure …
  -/
  unfold prehaar; rw [div_le_iff₀] <;> norm_cast
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      U : Set G
      K : TopologicalSpace.Compacts G
      hU : (interior U).Nonempty
      ⊢ LE.le (MeasureTheory.Measure.haar.index (↑K) U) (HMul.hMul (MeasureTheory.Me …
    -/
  · apply le_index_mul K₀ K hU
    /-
      🎉 no goals
    -/
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      U : Set G
      K : TopologicalSpace.Compacts G
      hU : (interior U).Nonempty
      ⊢ LT.lt 0 (MeasureTheory.Measure.haar.index (↑K₀) U)
    -/
  · exact index_pos K₀ hU
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prehaar_pos (K₀ : PositiveCompacts G) {U : Set G} (hU : (interior U).Nonempty) {K : Set G}
    (h1K : IsCompact K) (h2K : (interior K).Nonempty) : 0 < prehaar (K₀ : Set G) U ⟨K, h1K⟩ := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    U : Set G
    hU : (interior U).Nonempty
    K : Set G
    h1K : IsCompact K
    h2K : (interior K).Nonempty
    ⊢ LT.lt 0 (MeasureTheory.Measure.haar.prehaar (↑K₀) U { carrier := K, isCompac …
  -/
  apply div_pos <;> norm_cast
    /-
      case ha
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      U : Set G
      hU : (interior U).Nonempty
      K : Set G
      h1K : IsCompact K
      h2K : (interior K).Nonempty
      ⊢ LT.lt 0 (MeasureTheory.Measure.haar.index (↑{ carrier := K, isCompact' := h1 …
    -/
  · apply index_pos ⟨⟨K, h1K⟩, h2K⟩ hU
    /-
      🎉 no goals
    -/
    /-
      case hb
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      U : Set G
      hU : (interior U).Nonempty
      K : Set G
      h1K : IsCompact K
      h2K : (interior K).Nonempty
      ⊢ LT.lt 0 (MeasureTheory.Measure.haar.index (↑K₀) U)
    -/
  · exact index_pos K₀ hU
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prehaar_mono {K₀ : PositiveCompacts G} {U : Set G} (hU : (interior U).Nonempty)
    {K₁ K₂ : Compacts G} (h : (K₁ : Set G) ⊆ K₂.1) :
    prehaar (K₀ : Set G) U K₁ ≤ prehaar (K₀ : Set G) U K₂ := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    U : Set G
    hU : (interior U).Nonempty
    K₁ K₂ : TopologicalSpace.Compacts G
    h : HasSubset.Subset (↑K₁) K₂.carrier
    ⊢ LE.le (MeasureTheory.Measure.haar.prehaar (↑K₀) U K₁) (MeasureTheory.Measure …
  -/
  simp only [prehaar]; rw [div_le_div_iff_of_pos_right]
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      U : Set G
      hU : (interior U).Nonempty
      K₁ K₂ : TopologicalSpace.Compacts G
      h : HasSubset.Subset (↑K₁) K₂.carrier
      ⊢ LE.le ↑(MeasureTheory.Measure.haar.index (↑K₁) U) ↑(MeasureTheory.Measure.ha …
    -/
  · exact mod_cast index_mono K₂.2 h hU
    /-
      🎉 no goals
    -/
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      U : Set G
      hU : (interior U).Nonempty
      K₁ K₂ : TopologicalSpace.Compacts G
      h : HasSubset.Subset (↑K₁) K₂.carrier
      ⊢ LT.lt 0 ↑(MeasureTheory.Measure.haar.index (↑K₀) U)
    -/
  · exact mod_cast index_pos K₀ hU
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prehaar_self {K₀ : PositiveCompacts G} {U : Set G} (hU : (interior U).Nonempty) :
    prehaar (K₀ : Set G) U K₀.toCompacts = 1 :=
  div_self <| ne_of_gt <| mod_cast index_pos K₀ hU


@[to_additive]
theorem prehaar_sup_le {K₀ : PositiveCompacts G} {U : Set G} (K₁ K₂ : Compacts G)
    (hU : (interior U).Nonempty) :
    prehaar (K₀ : Set G) U (K₁ ⊔ K₂) ≤ prehaar (K₀ : Set G) U K₁ + prehaar (K₀ : Set G) U K₂ := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    U : Set G
    K₁ K₂ : TopologicalSpace.Compacts G
    hU : (interior U).Nonempty
    ⊢ LE.le (MeasureTheory.Measure.haar.prehaar (↑K₀) U (Max.max K₁ K₂)) (HAdd.hAd …
  -/
  simp only [prehaar]; rw [div_add_div_same, div_le_div_iff_of_pos_right]
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      U : Set G
      K₁ K₂ : TopologicalSpace.Compacts G
      hU : (interior U).Nonempty
      ⊢ LE.le (↑(MeasureTheory.Measure.haar.index (↑(Max.max K₁ K₂)) U)) (HAdd.hAdd  …
    -/
  · exact mod_cast index_union_le K₁ K₂ hU
    /-
      🎉 no goals
    -/
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      U : Set G
      K₁ K₂ : TopologicalSpace.Compacts G
      hU : (interior U).Nonempty
      ⊢ LT.lt 0 ↑(MeasureTheory.Measure.haar.index (↑K₀) U)
    -/
  · exact mod_cast index_pos K₀ hU
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prehaar_sup_eq {K₀ : PositiveCompacts G} {U : Set G} {K₁ K₂ : Compacts G}
    (hU : (interior U).Nonempty) (h : Disjoint (K₁.1 * U⁻¹) (K₂.1 * U⁻¹)) :
    prehaar (K₀ : Set G) U (K₁ ⊔ K₂) = prehaar (K₀ : Set G) U K₁ + prehaar (K₀ : Set G) U K₂ := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    U : Set G
    K₁ K₂ : TopologicalSpace.Compacts G
    hU : (interior U).Nonempty
    h : Disjoint (HMul.hMul K₁.carrier (Inv.inv U)) (HMul.hMul K₂.carrier (Inv.inv …
    ⊢ Eq (MeasureTheory.Measure.haar.prehaar (↑K₀) U (Max.max K₁ K₂)) (HAdd.hAdd ( …
  -/
  simp only [prehaar]; rw [div_add_div_same]
  -- Porting note: Here was `congr`, but `to_additive` failed to generate a theorem.
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    U : Set G
    K₁ K₂ : TopologicalSpace.Compacts G
    hU : (interior U).Nonempty
    h : Disjoint (HMul.hMul K₁.carrier (Inv.inv U)) (HMul.hMul K₂.carrier (Inv.inv …
    ⊢ Eq (HDiv.hDiv ↑(MeasureTheory.Measure.haar.index (↑(Max.max K₁ K₂)) U) ↑(Mea …
  -/
  refine congr_arg (fun x : ℝ => x / index K₀ U) ?_
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    U : Set G
    K₁ K₂ : TopologicalSpace.Compacts G
    hU : (interior U).Nonempty
    h : Disjoint (HMul.hMul K₁.carrier (Inv.inv U)) (HMul.hMul K₂.carrier (Inv.inv …
    ⊢ Eq (↑(MeasureTheory.Measure.haar.index (↑(Max.max K₁ K₂)) U)) (HAdd.hAdd ↑(M …
  -/
  exact mod_cast index_union_eq K₁ K₂ hU h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem is_left_invariant_prehaar {K₀ : PositiveCompacts G} {U : Set G} (hU : (interior U).Nonempty)
    (g : G) (K : Compacts G) :
    prehaar (K₀ : Set G) U (K.map _ <| continuous_mul_left g) = prehaar (K₀ : Set G) U K := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    U : Set G
    hU : (interior U).Nonempty
    g : G
    K : TopologicalSpace.Compacts G
    ⊢ Eq (MeasureTheory.Measure.haar.prehaar (↑K₀) U (TopologicalSpace.Compacts.ma …
  -/
  simp only [prehaar, Compacts.coe_map, is_left_invariant_index K.isCompact _ hU]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prehaar_mem_haarProduct (K₀ : PositiveCompacts G) {U : Set G} (hU : (interior U).Nonempty) :
    prehaar (K₀ : Set G) U ∈ haarProduct (K₀ : Set G) := by
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      U : Set G
      hU : (interior U).Nonempty
      ⊢ Membership.mem (MeasureTheory.Measure.haar.haarProduct ↑K₀) (MeasureTheory.M …
    -/
    rintro ⟨K, hK⟩ _; rw [mem_Icc]; exact ⟨prehaar_nonneg K₀ _, prehaar_le_index K₀ _ hU⟩
                                    /-
                                      🎉 no goals
                                    -/


@[to_additive]
theorem nonempty_iInter_clPrehaar (K₀ : PositiveCompacts G) :
    (haarProduct (K₀ : Set G) ∩ ⋂ V : OpenNhdsOf (1 : G), clPrehaar K₀ V).Nonempty := by
  have : IsCompact (haarProduct (K₀ : Set G)) := by
    apply isCompact_univ_pi; intro K; apply isCompact_Icc
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    this : IsCompact (MeasureTheory.Measure.haar.haarProduct ↑K₀)
    ⊢ (Inter.inter (MeasureTheory.Measure.haar.haarProduct ↑K₀) (Set.iInter fun V  …
  -/
  refine this.inter_iInter_nonempty (clPrehaar K₀) (fun s => isClosed_closure) fun t => ?_
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    this : IsCompact (MeasureTheory.Measure.haar.haarProduct ↑K₀)
    t : Finset (TopologicalSpace.OpenNhdsOf 1)
    ⊢ (Inter.inter (MeasureTheory.Measure.haar.haarProduct ↑K₀) (Set.iInter fun i  …
  -/
  let V₀ := ⋂ V ∈ t, (V : OpenNhdsOf (1 : G)).carrier
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    this : IsCompact (MeasureTheory.Measure.haar.haarProduct ↑K₀)
    t : Finset (TopologicalSpace.OpenNhdsOf 1)
    V₀ : Set G := Set.iInter fun V => Set.iInter fun h => V.carrier
    ⊢ (Inter.inter (MeasureTheory.Measure.haar.haarProduct ↑K₀) (Set.iInter fun i  …
  -/
  have h1V₀ : IsOpen V₀ := isOpen_biInter_finset <| by rintro ⟨⟨V, hV₁⟩, hV₂⟩ _; exact hV₁
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    this : IsCompact (MeasureTheory.Measure.haar.haarProduct ↑K₀)
    t : Finset (TopologicalSpace.OpenNhdsOf 1)
    V₀ : Set G := Set.iInter fun V => Set.iInter fun h => V.carrier
    h1V₀ : IsOpen V₀
    ⊢ (Inter.inter (MeasureTheory.Measure.haar.haarProduct ↑K₀) (Set.iInter fun i  …
  -/
  have h2V₀ : (1 : G) ∈ V₀ := by simp only [V₀, mem_iInter]; rintro ⟨⟨V, hV₁⟩, hV₂⟩ _; exact hV₂
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    this : IsCompact (MeasureTheory.Measure.haar.haarProduct ↑K₀)
    t : Finset (TopologicalSpace.OpenNhdsOf 1)
    V₀ : Set G := Set.iInter fun V => Set.iInter fun h => V.carrier
    h1V₀ : IsOpen V₀
    h2V₀ : Membership.mem V₀ 1
    ⊢ (Inter.inter (MeasureTheory.Measure.haar.haarProduct ↑K₀) (Set.iInter fun i  …
  -/
  refine ⟨prehaar K₀ V₀, ?_⟩
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    this : IsCompact (MeasureTheory.Measure.haar.haarProduct ↑K₀)
    t : Finset (TopologicalSpace.OpenNhdsOf 1)
    V₀ : Set G := Set.iInter fun V => Set.iInter fun h => V.carrier
    h1V₀ : IsOpen V₀
    h2V₀ : Membership.mem V₀ 1
    ⊢ Membership.mem (Inter.inter (MeasureTheory.Measure.haar.haarProduct ↑K₀) (Se …
  -/
  constructor
    /-
      case left
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      this : IsCompact (MeasureTheory.Measure.haar.haarProduct ↑K₀)
      t : Finset (TopologicalSpace.OpenNhdsOf 1)
      V₀ : Set G := Set.iInter fun V => Set.iInter fun h => V.carrier
      h1V₀ : IsOpen V₀
      h2V₀ : Membership.mem V₀ 1
      ⊢ Membership.mem (MeasureTheory.Measure.haar.haarProduct ↑K₀) (MeasureTheory.M …
    -/
  · apply prehaar_mem_haarProduct K₀; use 1; rwa [h1V₀.interior_eq]
                                             /-
                                               🎉 no goals
                                             -/
    /-
      case right
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      this : IsCompact (MeasureTheory.Measure.haar.haarProduct ↑K₀)
      t : Finset (TopologicalSpace.OpenNhdsOf 1)
      V₀ : Set G := Set.iInter fun V => Set.iInter fun h => V.carrier
      h1V₀ : IsOpen V₀
      h2V₀ : Membership.mem V₀ 1
      ⊢ Membership.mem (Set.iInter fun i => Set.iInter fun h => MeasureTheory.Measur …
    -/
  · simp only [mem_iInter]; rintro ⟨V, hV⟩ h2V; apply subset_closure
    /-
      case right.mk.a
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      this : IsCompact (MeasureTheory.Measure.haar.haarProduct ↑K₀)
      t : Finset (TopologicalSpace.OpenNhdsOf 1)
      V₀ : Set G := Set.iInter fun V => Set.iInter fun h => V.carrier
      h1V₀ : IsOpen V₀
      h2V₀ : Membership.mem V₀ 1
      V : TopologicalSpace.Opens G
      hV : Membership.mem V.carrier 1
      h2V : Membership.mem t { toOpens := V, mem' := hV }
      ⊢ Membership.mem (Set.image (MeasureTheory.Measure.haar.prehaar ↑K₀) (setOf fu …
    -/
    apply mem_image_of_mem; rw [mem_setOf_eq]
    /-
      case right.mk.a.h
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      this : IsCompact (MeasureTheory.Measure.haar.haarProduct ↑K₀)
      t : Finset (TopologicalSpace.OpenNhdsOf 1)
      V₀ : Set G := Set.iInter fun V => Set.iInter fun h => V.carrier
      h1V₀ : IsOpen V₀
      h2V₀ : Membership.mem V₀ 1
      V : TopologicalSpace.Opens G
      hV : Membership.mem V.carrier 1
      h2V : Membership.mem t { toOpens := V, mem' := hV }
      ⊢ And (HasSubset.Subset V₀ ↑{ toOpens := V, mem' := hV }.toOpens) (And (IsOpen …
    -/
    exact ⟨Subset.trans (iInter_subset _ ⟨V, hV⟩) (iInter_subset _ h2V), h1V₀, h2V₀⟩
    /-
      🎉 no goals
    -/


/-- This is the "limit" of `prehaar K₀ U K` as `U` becomes a smaller and smaller open
  neighborhood of `(1 : G)`. More precisely, it is defined to be an arbitrary element
  in the intersection of all the sets `clPrehaar K₀ V` in `haarProduct K₀`.
  This is roughly equal to the Haar measure on compact sets,
  but it can differ slightly. We do know that
  `haarMeasure K₀ (interior K) ≤ chaar K₀ K ≤ haarMeasure K₀ K`. -/
@[to_additive addCHaar "additive version of `MeasureTheory.Measure.haar.chaar`"]
noncomputable def chaar (K₀ : PositiveCompacts G) (K : Compacts G) : ℝ :=
  Classical.choose (nonempty_iInter_clPrehaar K₀) K


@[to_additive addCHaar_mem_addHaarProduct]
theorem chaar_mem_haarProduct (K₀ : PositiveCompacts G) : chaar K₀ ∈ haarProduct (K₀ : Set G) :=
  (Classical.choose_spec (nonempty_iInter_clPrehaar K₀)).1


@[to_additive addCHaar_mem_clAddPrehaar]
theorem chaar_mem_clPrehaar (K₀ : PositiveCompacts G) (V : OpenNhdsOf (1 : G)) :
    chaar K₀ ∈ clPrehaar (K₀ : Set G) V := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    V : TopologicalSpace.OpenNhdsOf 1
    ⊢ Membership.mem (MeasureTheory.Measure.haar.clPrehaar (↑K₀) V) (MeasureTheory …
  -/
  have := (Classical.choose_spec (nonempty_iInter_clPrehaar K₀)).2; rw [mem_iInter] at this
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    V : TopologicalSpace.OpenNhdsOf 1
    this : ∀ (i : TopologicalSpace.OpenNhdsOf 1), Membership.mem (MeasureTheory.Me …
    ⊢ Membership.mem (MeasureTheory.Measure.haar.clPrehaar (↑K₀) V) (MeasureTheory …
  -/
  exact this V
  /-
    🎉 no goals
  -/


@[to_additive addCHaar_nonneg]
theorem chaar_nonneg (K₀ : PositiveCompacts G) (K : Compacts G) : 0 ≤ chaar K₀ K := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K : TopologicalSpace.Compacts G
    ⊢ LE.le 0 (MeasureTheory.Measure.haar.chaar K₀ K)
  -/
  have := chaar_mem_haarProduct K₀ K (mem_univ _); rw [mem_Icc] at this; exact this.1
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[to_additive addCHaar_empty]
theorem chaar_empty (K₀ : PositiveCompacts G) : chaar K₀ ⊥ = 0 := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ Bot.bot) 0
  -/
  let eval : (Compacts G → ℝ) → ℝ := fun f => f ⊥
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f Bot.bot
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ Bot.bot) 0
  -/
  have : Continuous eval := continuous_apply ⊥
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f Bot.bot
    this : Continuous eval
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ Bot.bot) 0
  -/
  show chaar K₀ ∈ eval ⁻¹' {(0 : ℝ)}
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f Bot.bot
    this : Continuous eval
    ⊢ Membership.mem (Set.preimage eval (Singleton.singleton 0)) (MeasureTheory.Me …
  -/
  apply mem_of_subset_of_mem _ (chaar_mem_clPrehaar K₀ ⊤)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f Bot.bot
    this : Continuous eval
    ⊢ HasSubset.Subset (MeasureTheory.Measure.haar.clPrehaar (↑K₀) Top.top) (Set.p …
  -/
  unfold clPrehaar; rw [IsClosed.closure_subset_iff]
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f Bot.bot
      this : Continuous eval
      ⊢ HasSubset.Subset (Set.image (MeasureTheory.Measure.haar.prehaar ↑K₀) (setOf  …
    -/
  · rintro _ ⟨U, _, rfl⟩; apply prehaar_empty
                          /-
                            🎉 no goals
                          -/
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f Bot.bot
      this : Continuous eval
      ⊢ IsClosed (Set.preimage eval (Singleton.singleton 0))
    -/
  · apply continuous_iff_isClosed.mp this; exact isClosed_singleton
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive addCHaar_self]
theorem chaar_self (K₀ : PositiveCompacts G) : chaar K₀ K₀.toCompacts = 1 := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ K₀.toCompacts) 1
  -/
  let eval : (Compacts G → ℝ) → ℝ := fun f => f K₀.toCompacts
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f K₀.toCompacts
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ K₀.toCompacts) 1
  -/
  have : Continuous eval := continuous_apply _
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f K₀.toCompacts
    this : Continuous eval
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ K₀.toCompacts) 1
  -/
  show chaar K₀ ∈ eval ⁻¹' {(1 : ℝ)}
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f K₀.toCompacts
    this : Continuous eval
    ⊢ Membership.mem (Set.preimage eval (Singleton.singleton 1)) (MeasureTheory.Me …
  -/
  apply mem_of_subset_of_mem _ (chaar_mem_clPrehaar K₀ ⊤)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f K₀.toCompacts
    this : Continuous eval
    ⊢ HasSubset.Subset (MeasureTheory.Measure.haar.clPrehaar (↑K₀) Top.top) (Set.p …
  -/
  unfold clPrehaar; rw [IsClosed.closure_subset_iff]
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f K₀.toCompacts
      this : Continuous eval
      ⊢ HasSubset.Subset (Set.image (MeasureTheory.Measure.haar.prehaar ↑K₀) (setOf  …
    -/
  · rintro _ ⟨U, ⟨_, h2U, h3U⟩, rfl⟩; apply prehaar_self
    /-
      case intro.intro.intro.intro.hU
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f K₀.toCompacts
      this : Continuous eval
      U : Set G
      left✝ : HasSubset.Subset U ↑Top.top.toOpens
      h2U : IsOpen U
      h3U : Membership.mem U 1
      ⊢ (interior U).Nonempty
    -/
    rw [h2U.interior_eq]; exact ⟨1, h3U⟩
                          /-
                            🎉 no goals
                          -/
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => f K₀.toCompacts
      this : Continuous eval
      ⊢ IsClosed (Set.preimage eval (Singleton.singleton 1))
    -/
  · apply continuous_iff_isClosed.mp this; exact isClosed_singleton
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive addCHaar_mono]
theorem chaar_mono {K₀ : PositiveCompacts G} {K₁ K₂ : Compacts G} (h : (K₁ : Set G) ⊆ K₂) :
    chaar K₀ K₁ ≤ chaar K₀ K₂ := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : HasSubset.Subset ↑K₁ ↑K₂
    ⊢ LE.le (MeasureTheory.Measure.haar.chaar K₀ K₁) (MeasureTheory.Measure.haar.c …
  -/
  let eval : (Compacts G → ℝ) → ℝ := fun f => f K₂ - f K₁
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : HasSubset.Subset ↑K₁ ↑K₂
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f K₂ …
    ⊢ LE.le (MeasureTheory.Measure.haar.chaar K₀ K₁) (MeasureTheory.Measure.haar.c …
  -/
  have : Continuous eval := (continuous_apply K₂).sub (continuous_apply K₁)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : HasSubset.Subset ↑K₁ ↑K₂
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f K₂ …
    this : Continuous eval
    ⊢ LE.le (MeasureTheory.Measure.haar.chaar K₀ K₁) (MeasureTheory.Measure.haar.c …
  -/
  rw [← sub_nonneg]; show chaar K₀ ∈ eval ⁻¹' Ici (0 : ℝ)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : HasSubset.Subset ↑K₁ ↑K₂
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f K₂ …
    this : Continuous eval
    ⊢ Membership.mem (Set.preimage eval (Set.Ici 0)) (MeasureTheory.Measure.haar.c …
  -/
  apply mem_of_subset_of_mem _ (chaar_mem_clPrehaar K₀ ⊤)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : HasSubset.Subset ↑K₁ ↑K₂
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f K₂ …
    this : Continuous eval
    ⊢ HasSubset.Subset (MeasureTheory.Measure.haar.clPrehaar (↑K₀) Top.top) (Set.p …
  -/
  unfold clPrehaar; rw [IsClosed.closure_subset_iff]
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      h : HasSubset.Subset ↑K₁ ↑K₂
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f K₂ …
      this : Continuous eval
      ⊢ HasSubset.Subset (Set.image (MeasureTheory.Measure.haar.prehaar ↑K₀) (setOf  …
    -/
  · rintro _ ⟨U, ⟨_, h2U, h3U⟩, rfl⟩; simp only [eval, mem_preimage, mem_Ici, sub_nonneg]
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      h : HasSubset.Subset ↑K₁ ↑K₂
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f K₂ …
      this : Continuous eval
      U : Set G
      left✝ : HasSubset.Subset U ↑Top.top.toOpens
      h2U : IsOpen U
      h3U : Membership.mem U 1
      ⊢ LE.le (MeasureTheory.Measure.haar.prehaar (↑K₀) U K₁) (MeasureTheory.Measure …
    -/
    apply prehaar_mono _ h; rw [h2U.interior_eq]; exact ⟨1, h3U⟩
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      h : HasSubset.Subset ↑K₁ ↑K₂
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f K₂ …
      this : Continuous eval
      ⊢ IsClosed (Set.preimage eval (Set.Ici 0))
    -/
  · apply continuous_iff_isClosed.mp this; exact isClosed_Ici
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive addCHaar_sup_le]
theorem chaar_sup_le {K₀ : PositiveCompacts G} (K₁ K₂ : Compacts G) :
    chaar K₀ (K₁ ⊔ K₂) ≤ chaar K₀ K₁ + chaar K₀ K₂ := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    ⊢ LE.le (MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂)) (HAdd.hAdd (Meas …
  -/
  let eval : (Compacts G → ℝ) → ℝ := fun f => f K₁ + f K₂ - f (K₁ ⊔ K₂)
  have : Continuous eval := by
    exact ((continuous_apply K₁).add (continuous_apply K₂)).sub (continuous_apply (K₁ ⊔ K₂))
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
    this : Continuous eval
    ⊢ LE.le (MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂)) (HAdd.hAdd (Meas …
  -/
  rw [← sub_nonneg]; show chaar K₀ ∈ eval ⁻¹' Ici (0 : ℝ)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
    this : Continuous eval
    ⊢ Membership.mem (Set.preimage eval (Set.Ici 0)) (MeasureTheory.Measure.haar.c …
  -/
  apply mem_of_subset_of_mem _ (chaar_mem_clPrehaar K₀ ⊤)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
    this : Continuous eval
    ⊢ HasSubset.Subset (MeasureTheory.Measure.haar.clPrehaar (↑K₀) Top.top) (Set.p …
  -/
  unfold clPrehaar; rw [IsClosed.closure_subset_iff]
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
      this : Continuous eval
      ⊢ HasSubset.Subset (Set.image (MeasureTheory.Measure.haar.prehaar ↑K₀) (setOf  …
    -/
  · rintro _ ⟨U, ⟨_, h2U, h3U⟩, rfl⟩; simp only [eval, mem_preimage, mem_Ici, sub_nonneg]
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
      this : Continuous eval
      U : Set G
      left✝ : HasSubset.Subset U ↑Top.top.toOpens
      h2U : IsOpen U
      h3U : Membership.mem U 1
      ⊢ LE.le (MeasureTheory.Measure.haar.prehaar (↑K₀) U (Max.max K₁ K₂)) (HAdd.hAd …
    -/
    apply prehaar_sup_le; rw [h2U.interior_eq]; exact ⟨1, h3U⟩
                                                /-
                                                  🎉 no goals
                                                -/
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
      this : Continuous eval
      ⊢ IsClosed (Set.preimage eval (Set.Ici 0))
    -/
  · apply continuous_iff_isClosed.mp this; exact isClosed_Ici
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive addCHaar_sup_eq]
theorem chaar_sup_eq {K₀ : PositiveCompacts G}
    {K₁ K₂ : Compacts G} (h : Disjoint K₁.1 K₂.1) (h₂ : IsClosed K₂.1) :
    chaar K₀ (K₁ ⊔ K₂) = chaar K₀ K₁ + chaar K₀ K₂ := by
  rcases SeparatedNhds.of_isCompact_isCompact_isClosed K₁.2 K₂.2 h₂ h
    with ⟨U₁, U₂, h1U₁, h1U₂, h2U₁, h2U₂, hU⟩
  /-
    case intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint K₁.carrier K₂.carrier
    h₂ : IsClosed K₂.carrier
    U₁ U₂ : Set G
    h1U₁ : IsOpen U₁
    h1U₂ : IsOpen U₂
    h2U₁ : HasSubset.Subset K₁.carrier U₁
    h2U₂ : HasSubset.Subset K₂.carrier U₂
    hU : Disjoint U₁ U₂
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂)) (HAdd.hAdd (Measure …
  -/
  rcases compact_open_separated_mul_right K₁.2 h1U₁ h2U₁ with ⟨L₁, h1L₁, h2L₁⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint K₁.carrier K₂.carrier
    h₂ : IsClosed K₂.carrier
    U₁ U₂ : Set G
    h1U₁ : IsOpen U₁
    h1U₂ : IsOpen U₂
    h2U₁ : HasSubset.Subset K₁.carrier U₁
    h2U₂ : HasSubset.Subset K₂.carrier U₂
    hU : Disjoint U₁ U₂
    L₁ : Set G
    h1L₁ : Membership.mem (nhds 1) L₁
    h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier L₁) U₁
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂)) (HAdd.hAdd (Measure …
  -/
  rcases mem_nhds_iff.mp h1L₁ with ⟨V₁, h1V₁, h2V₁, h3V₁⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint K₁.carrier K₂.carrier
    h₂ : IsClosed K₂.carrier
    U₁ U₂ : Set G
    h1U₁ : IsOpen U₁
    h1U₂ : IsOpen U₂
    h2U₁ : HasSubset.Subset K₁.carrier U₁
    h2U₂ : HasSubset.Subset K₂.carrier U₂
    hU : Disjoint U₁ U₂
    L₁ : Set G
    h1L₁ : Membership.mem (nhds 1) L₁
    h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier L₁) U₁
    V₁ : Set G
    h1V₁ : HasSubset.Subset V₁ L₁
    h2V₁ : IsOpen V₁
    h3V₁ : Membership.mem V₁ 1
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂)) (HAdd.hAdd (Measure …
  -/
  replace h2L₁ := Subset.trans (mul_subset_mul_left h1V₁) h2L₁
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint K₁.carrier K₂.carrier
    h₂ : IsClosed K₂.carrier
    U₁ U₂ : Set G
    h1U₁ : IsOpen U₁
    h1U₂ : IsOpen U₂
    h2U₁ : HasSubset.Subset K₁.carrier U₁
    h2U₂ : HasSubset.Subset K₂.carrier U₂
    hU : Disjoint U₁ U₂
    L₁ : Set G
    h1L₁ : Membership.mem (nhds 1) L₁
    V₁ : Set G
    h1V₁ : HasSubset.Subset V₁ L₁
    h2V₁ : IsOpen V₁
    h3V₁ : Membership.mem V₁ 1
    h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂)) (HAdd.hAdd (Measure …
  -/
  rcases compact_open_separated_mul_right K₂.2 h1U₂ h2U₂ with ⟨L₂, h1L₂, h2L₂⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint K₁.carrier K₂.carrier
    h₂ : IsClosed K₂.carrier
    U₁ U₂ : Set G
    h1U₁ : IsOpen U₁
    h1U₂ : IsOpen U₂
    h2U₁ : HasSubset.Subset K₁.carrier U₁
    h2U₂ : HasSubset.Subset K₂.carrier U₂
    hU : Disjoint U₁ U₂
    L₁ : Set G
    h1L₁ : Membership.mem (nhds 1) L₁
    V₁ : Set G
    h1V₁ : HasSubset.Subset V₁ L₁
    h2V₁ : IsOpen V₁
    h3V₁ : Membership.mem V₁ 1
    h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
    L₂ : Set G
    h1L₂ : Membership.mem (nhds 1) L₂
    h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier L₂) U₂
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂)) (HAdd.hAdd (Measure …
  -/
  rcases mem_nhds_iff.mp h1L₂ with ⟨V₂, h1V₂, h2V₂, h3V₂⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint K₁.carrier K₂.carrier
    h₂ : IsClosed K₂.carrier
    U₁ U₂ : Set G
    h1U₁ : IsOpen U₁
    h1U₂ : IsOpen U₂
    h2U₁ : HasSubset.Subset K₁.carrier U₁
    h2U₂ : HasSubset.Subset K₂.carrier U₂
    hU : Disjoint U₁ U₂
    L₁ : Set G
    h1L₁ : Membership.mem (nhds 1) L₁
    V₁ : Set G
    h1V₁ : HasSubset.Subset V₁ L₁
    h2V₁ : IsOpen V₁
    h3V₁ : Membership.mem V₁ 1
    h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
    L₂ : Set G
    h1L₂ : Membership.mem (nhds 1) L₂
    h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier L₂) U₂
    V₂ : Set G
    h1V₂ : HasSubset.Subset V₂ L₂
    h2V₂ : IsOpen V₂
    h3V₂ : Membership.mem V₂ 1
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂)) (HAdd.hAdd (Measure …
  -/
  replace h2L₂ := Subset.trans (mul_subset_mul_left h1V₂) h2L₂
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint K₁.carrier K₂.carrier
    h₂ : IsClosed K₂.carrier
    U₁ U₂ : Set G
    h1U₁ : IsOpen U₁
    h1U₂ : IsOpen U₂
    h2U₁ : HasSubset.Subset K₁.carrier U₁
    h2U₂ : HasSubset.Subset K₂.carrier U₂
    hU : Disjoint U₁ U₂
    L₁ : Set G
    h1L₁ : Membership.mem (nhds 1) L₁
    V₁ : Set G
    h1V₁ : HasSubset.Subset V₁ L₁
    h2V₁ : IsOpen V₁
    h3V₁ : Membership.mem V₁ 1
    h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
    L₂ : Set G
    h1L₂ : Membership.mem (nhds 1) L₂
    V₂ : Set G
    h1V₂ : HasSubset.Subset V₂ L₂
    h2V₂ : IsOpen V₂
    h3V₂ : Membership.mem V₂ 1
    h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂)) (HAdd.hAdd (Measure …
  -/
  let eval : (Compacts G → ℝ) → ℝ := fun f => f K₁ + f K₂ - f (K₁ ⊔ K₂)
  have : Continuous eval :=
    ((continuous_apply K₁).add (continuous_apply K₂)).sub (continuous_apply (K₁ ⊔ K₂))
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint K₁.carrier K₂.carrier
    h₂ : IsClosed K₂.carrier
    U₁ U₂ : Set G
    h1U₁ : IsOpen U₁
    h1U₂ : IsOpen U₂
    h2U₁ : HasSubset.Subset K₁.carrier U₁
    h2U₂ : HasSubset.Subset K₂.carrier U₂
    hU : Disjoint U₁ U₂
    L₁ : Set G
    h1L₁ : Membership.mem (nhds 1) L₁
    V₁ : Set G
    h1V₁ : HasSubset.Subset V₁ L₁
    h2V₁ : IsOpen V₁
    h3V₁ : Membership.mem V₁ 1
    h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
    L₂ : Set G
    h1L₂ : Membership.mem (nhds 1) L₂
    V₂ : Set G
    h1V₂ : HasSubset.Subset V₂ L₂
    h2V₂ : IsOpen V₂
    h3V₂ : Membership.mem V₂ 1
    h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
    this : Continuous eval
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂)) (HAdd.hAdd (Measure …
  -/
  rw [eq_comm, ← sub_eq_zero]; show chaar K₀ ∈ eval ⁻¹' {(0 : ℝ)}
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint K₁.carrier K₂.carrier
    h₂ : IsClosed K₂.carrier
    U₁ U₂ : Set G
    h1U₁ : IsOpen U₁
    h1U₂ : IsOpen U₂
    h2U₁ : HasSubset.Subset K₁.carrier U₁
    h2U₂ : HasSubset.Subset K₂.carrier U₂
    hU : Disjoint U₁ U₂
    L₁ : Set G
    h1L₁ : Membership.mem (nhds 1) L₁
    V₁ : Set G
    h1V₁ : HasSubset.Subset V₁ L₁
    h2V₁ : IsOpen V₁
    h3V₁ : Membership.mem V₁ 1
    h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
    L₂ : Set G
    h1L₂ : Membership.mem (nhds 1) L₂
    V₂ : Set G
    h1V₂ : HasSubset.Subset V₂ L₂
    h2V₂ : IsOpen V₂
    h3V₂ : Membership.mem V₂ 1
    h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
    this : Continuous eval
    ⊢ Membership.mem (Set.preimage eval (Singleton.singleton 0)) (MeasureTheory.Me …
  -/
  let V := V₁ ∩ V₂
  apply
    mem_of_subset_of_mem _
      (chaar_mem_clPrehaar K₀
        ⟨⟨V⁻¹, (h2V₁.inter h2V₂).preimage continuous_inv⟩, by
          simp only [V, mem_inv, inv_one, h3V₁, h3V₂, mem_inter_iff, true_and]⟩)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint K₁.carrier K₂.carrier
    h₂ : IsClosed K₂.carrier
    U₁ U₂ : Set G
    h1U₁ : IsOpen U₁
    h1U₂ : IsOpen U₂
    h2U₁ : HasSubset.Subset K₁.carrier U₁
    h2U₂ : HasSubset.Subset K₂.carrier U₂
    hU : Disjoint U₁ U₂
    L₁ : Set G
    h1L₁ : Membership.mem (nhds 1) L₁
    V₁ : Set G
    h1V₁ : HasSubset.Subset V₁ L₁
    h2V₁ : IsOpen V₁
    h3V₁ : Membership.mem V₁ 1
    h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
    L₂ : Set G
    h1L₂ : Membership.mem (nhds 1) L₂
    V₂ : Set G
    h1V₂ : HasSubset.Subset V₂ L₂
    h2V₂ : IsOpen V₂
    h3V₂ : Membership.mem V₂ 1
    h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
    this : Continuous eval
    V : Set G := Inter.inter V₁ V₂
    ⊢ HasSubset.Subset (MeasureTheory.Measure.haar.clPrehaar ↑K₀ { carrier := Inv. …
  -/
  unfold clPrehaar; rw [IsClosed.closure_subset_iff]
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      h : Disjoint K₁.carrier K₂.carrier
      h₂ : IsClosed K₂.carrier
      U₁ U₂ : Set G
      h1U₁ : IsOpen U₁
      h1U₂ : IsOpen U₂
      h2U₁ : HasSubset.Subset K₁.carrier U₁
      h2U₂ : HasSubset.Subset K₂.carrier U₂
      hU : Disjoint U₁ U₂
      L₁ : Set G
      h1L₁ : Membership.mem (nhds 1) L₁
      V₁ : Set G
      h1V₁ : HasSubset.Subset V₁ L₁
      h2V₁ : IsOpen V₁
      h3V₁ : Membership.mem V₁ 1
      h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
      L₂ : Set G
      h1L₂ : Membership.mem (nhds 1) L₂
      V₂ : Set G
      h1V₂ : HasSubset.Subset V₂ L₂
      h2V₂ : IsOpen V₂
      h3V₂ : Membership.mem V₂ 1
      h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
      this : Continuous eval
      V : Set G := Inter.inter V₁ V₂
      ⊢ HasSubset.Subset (Set.image (MeasureTheory.Measure.haar.prehaar ↑K₀) (setOf  …
    -/
  · rintro _ ⟨U, ⟨h1U, h2U, h3U⟩, rfl⟩
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      h : Disjoint K₁.carrier K₂.carrier
      h₂ : IsClosed K₂.carrier
      U₁ U₂ : Set G
      h1U₁ : IsOpen U₁
      h1U₂ : IsOpen U₂
      h2U₁ : HasSubset.Subset K₁.carrier U₁
      h2U₂ : HasSubset.Subset K₂.carrier U₂
      hU : Disjoint U₁ U₂
      L₁ : Set G
      h1L₁ : Membership.mem (nhds 1) L₁
      V₁ : Set G
      h1V₁ : HasSubset.Subset V₁ L₁
      h2V₁ : IsOpen V₁
      h3V₁ : Membership.mem V₁ 1
      h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
      L₂ : Set G
      h1L₂ : Membership.mem (nhds 1) L₂
      V₂ : Set G
      h1V₂ : HasSubset.Subset V₂ L₂
      h2V₂ : IsOpen V₂
      h3V₂ : Membership.mem V₂ 1
      h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
      this : Continuous eval
      V : Set G := Inter.inter V₁ V₂
      U : Set G
      h1U : HasSubset.Subset U ↑{ carrier := Inv.inv V, is_open' := ⋯, mem' := ⋯ }.t …
      h2U : IsOpen U
      h3U : Membership.mem U 1
      ⊢ Membership.mem (Set.preimage eval (Singleton.singleton 0)) (MeasureTheory.Me …
    -/
    simp only [eval, mem_preimage, sub_eq_zero, mem_singleton_iff]; rw [eq_comm]
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      h : Disjoint K₁.carrier K₂.carrier
      h₂ : IsClosed K₂.carrier
      U₁ U₂ : Set G
      h1U₁ : IsOpen U₁
      h1U₂ : IsOpen U₂
      h2U₁ : HasSubset.Subset K₁.carrier U₁
      h2U₂ : HasSubset.Subset K₂.carrier U₂
      hU : Disjoint U₁ U₂
      L₁ : Set G
      h1L₁ : Membership.mem (nhds 1) L₁
      V₁ : Set G
      h1V₁ : HasSubset.Subset V₁ L₁
      h2V₁ : IsOpen V₁
      h3V₁ : Membership.mem V₁ 1
      h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
      L₂ : Set G
      h1L₂ : Membership.mem (nhds 1) L₂
      V₂ : Set G
      h1V₂ : HasSubset.Subset V₂ L₂
      h2V₂ : IsOpen V₂
      h3V₂ : Membership.mem V₂ 1
      h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
      this : Continuous eval
      V : Set G := Inter.inter V₁ V₂
      U : Set G
      h1U : HasSubset.Subset U ↑{ carrier := Inv.inv V, is_open' := ⋯, mem' := ⋯ }.t …
      h2U : IsOpen U
      h3U : Membership.mem U 1
      ⊢ Eq (MeasureTheory.Measure.haar.prehaar (↑K₀) U (Max.max K₁ K₂)) (HAdd.hAdd ( …
    -/
    apply prehaar_sup_eq
      /-
        case intro.intro.intro.intro.hU
        G : Type u_1
        inst✝² : Group G
        inst✝¹ : TopologicalSpace G
        inst✝ : TopologicalGroup G
        K₀ : TopologicalSpace.PositiveCompacts G
        K₁ K₂ : TopologicalSpace.Compacts G
        h : Disjoint K₁.carrier K₂.carrier
        h₂ : IsClosed K₂.carrier
        U₁ U₂ : Set G
        h1U₁ : IsOpen U₁
        h1U₂ : IsOpen U₂
        h2U₁ : HasSubset.Subset K₁.carrier U₁
        h2U₂ : HasSubset.Subset K₂.carrier U₂
        hU : Disjoint U₁ U₂
        L₁ : Set G
        h1L₁ : Membership.mem (nhds 1) L₁
        V₁ : Set G
        h1V₁ : HasSubset.Subset V₁ L₁
        h2V₁ : IsOpen V₁
        h3V₁ : Membership.mem V₁ 1
        h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
        L₂ : Set G
        h1L₂ : Membership.mem (nhds 1) L₂
        V₂ : Set G
        h1V₂ : HasSubset.Subset V₂ L₂
        h2V₂ : IsOpen V₂
        h3V₂ : Membership.mem V₂ 1
        h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
        eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
        this : Continuous eval
        V : Set G := Inter.inter V₁ V₂
        U : Set G
        h1U : HasSubset.Subset U ↑{ carrier := Inv.inv V, is_open' := ⋯, mem' := ⋯ }.t …
        h2U : IsOpen U
        h3U : Membership.mem U 1
        ⊢ (interior U).Nonempty
      -/
    · rw [h2U.interior_eq]; exact ⟨1, h3U⟩
                            /-
                              🎉 no goals
                            -/
      /-
        case intro.intro.intro.intro.h
        G : Type u_1
        inst✝² : Group G
        inst✝¹ : TopologicalSpace G
        inst✝ : TopologicalGroup G
        K₀ : TopologicalSpace.PositiveCompacts G
        K₁ K₂ : TopologicalSpace.Compacts G
        h : Disjoint K₁.carrier K₂.carrier
        h₂ : IsClosed K₂.carrier
        U₁ U₂ : Set G
        h1U₁ : IsOpen U₁
        h1U₂ : IsOpen U₂
        h2U₁ : HasSubset.Subset K₁.carrier U₁
        h2U₂ : HasSubset.Subset K₂.carrier U₂
        hU : Disjoint U₁ U₂
        L₁ : Set G
        h1L₁ : Membership.mem (nhds 1) L₁
        V₁ : Set G
        h1V₁ : HasSubset.Subset V₁ L₁
        h2V₁ : IsOpen V₁
        h3V₁ : Membership.mem V₁ 1
        h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
        L₂ : Set G
        h1L₂ : Membership.mem (nhds 1) L₂
        V₂ : Set G
        h1V₂ : HasSubset.Subset V₂ L₂
        h2V₂ : IsOpen V₂
        h3V₂ : Membership.mem V₂ 1
        h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
        eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
        this : Continuous eval
        V : Set G := Inter.inter V₁ V₂
        U : Set G
        h1U : HasSubset.Subset U ↑{ carrier := Inv.inv V, is_open' := ⋯, mem' := ⋯ }.t …
        h2U : IsOpen U
        h3U : Membership.mem U 1
        ⊢ Disjoint (HMul.hMul K₁.carrier (Inv.inv U)) (HMul.hMul K₂.carrier (Inv.inv U))
      -/
    · refine disjoint_of_subset ?_ ?_ hU
        /-
          case intro.intro.intro.intro.h.refine_1
          G : Type u_1
          inst✝² : Group G
          inst✝¹ : TopologicalSpace G
          inst✝ : TopologicalGroup G
          K₀ : TopologicalSpace.PositiveCompacts G
          K₁ K₂ : TopologicalSpace.Compacts G
          h : Disjoint K₁.carrier K₂.carrier
          h₂ : IsClosed K₂.carrier
          U₁ U₂ : Set G
          h1U₁ : IsOpen U₁
          h1U₂ : IsOpen U₂
          h2U₁ : HasSubset.Subset K₁.carrier U₁
          h2U₂ : HasSubset.Subset K₂.carrier U₂
          hU : Disjoint U₁ U₂
          L₁ : Set G
          h1L₁ : Membership.mem (nhds 1) L₁
          V₁ : Set G
          h1V₁ : HasSubset.Subset V₁ L₁
          h2V₁ : IsOpen V₁
          h3V₁ : Membership.mem V₁ 1
          h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
          L₂ : Set G
          h1L₂ : Membership.mem (nhds 1) L₂
          V₂ : Set G
          h1V₂ : HasSubset.Subset V₂ L₂
          h2V₂ : IsOpen V₂
          h3V₂ : Membership.mem V₂ 1
          h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
          eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
          this : Continuous eval
          V : Set G := Inter.inter V₁ V₂
          U : Set G
          h1U : HasSubset.Subset U ↑{ carrier := Inv.inv V, is_open' := ⋯, mem' := ⋯ }.t …
          h2U : IsOpen U
          h3U : Membership.mem U 1
          ⊢ HasSubset.Subset (HMul.hMul K₁.carrier (Inv.inv U)) U₁
        -/
      · refine Subset.trans (mul_subset_mul Subset.rfl ?_) h2L₁
        /-
          case intro.intro.intro.intro.h.refine_1
          G : Type u_1
          inst✝² : Group G
          inst✝¹ : TopologicalSpace G
          inst✝ : TopologicalGroup G
          K₀ : TopologicalSpace.PositiveCompacts G
          K₁ K₂ : TopologicalSpace.Compacts G
          h : Disjoint K₁.carrier K₂.carrier
          h₂ : IsClosed K₂.carrier
          U₁ U₂ : Set G
          h1U₁ : IsOpen U₁
          h1U₂ : IsOpen U₂
          h2U₁ : HasSubset.Subset K₁.carrier U₁
          h2U₂ : HasSubset.Subset K₂.carrier U₂
          hU : Disjoint U₁ U₂
          L₁ : Set G
          h1L₁ : Membership.mem (nhds 1) L₁
          V₁ : Set G
          h1V₁ : HasSubset.Subset V₁ L₁
          h2V₁ : IsOpen V₁
          h3V₁ : Membership.mem V₁ 1
          h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
          L₂ : Set G
          h1L₂ : Membership.mem (nhds 1) L₂
          V₂ : Set G
          h1V₂ : HasSubset.Subset V₂ L₂
          h2V₂ : IsOpen V₂
          h3V₂ : Membership.mem V₂ 1
          h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
          eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
          this : Continuous eval
          V : Set G := Inter.inter V₁ V₂
          U : Set G
          h1U : HasSubset.Subset U ↑{ carrier := Inv.inv V, is_open' := ⋯, mem' := ⋯ }.t …
          h2U : IsOpen U
          h3U : Membership.mem U 1
          ⊢ HasSubset.Subset (Inv.inv U) V₁
        -/
        exact Subset.trans (inv_subset.mpr h1U) inter_subset_left
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.intro.h.refine_2
          G : Type u_1
          inst✝² : Group G
          inst✝¹ : TopologicalSpace G
          inst✝ : TopologicalGroup G
          K₀ : TopologicalSpace.PositiveCompacts G
          K₁ K₂ : TopologicalSpace.Compacts G
          h : Disjoint K₁.carrier K₂.carrier
          h₂ : IsClosed K₂.carrier
          U₁ U₂ : Set G
          h1U₁ : IsOpen U₁
          h1U₂ : IsOpen U₂
          h2U₁ : HasSubset.Subset K₁.carrier U₁
          h2U₂ : HasSubset.Subset K₂.carrier U₂
          hU : Disjoint U₁ U₂
          L₁ : Set G
          h1L₁ : Membership.mem (nhds 1) L₁
          V₁ : Set G
          h1V₁ : HasSubset.Subset V₁ L₁
          h2V₁ : IsOpen V₁
          h3V₁ : Membership.mem V₁ 1
          h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
          L₂ : Set G
          h1L₂ : Membership.mem (nhds 1) L₂
          V₂ : Set G
          h1V₂ : HasSubset.Subset V₂ L₂
          h2V₂ : IsOpen V₂
          h3V₂ : Membership.mem V₂ 1
          h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
          eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
          this : Continuous eval
          V : Set G := Inter.inter V₁ V₂
          U : Set G
          h1U : HasSubset.Subset U ↑{ carrier := Inv.inv V, is_open' := ⋯, mem' := ⋯ }.t …
          h2U : IsOpen U
          h3U : Membership.mem U 1
          ⊢ HasSubset.Subset (HMul.hMul K₂.carrier (Inv.inv U)) U₂
        -/
      · refine Subset.trans (mul_subset_mul Subset.rfl ?_) h2L₂
        /-
          case intro.intro.intro.intro.h.refine_2
          G : Type u_1
          inst✝² : Group G
          inst✝¹ : TopologicalSpace G
          inst✝ : TopologicalGroup G
          K₀ : TopologicalSpace.PositiveCompacts G
          K₁ K₂ : TopologicalSpace.Compacts G
          h : Disjoint K₁.carrier K₂.carrier
          h₂ : IsClosed K₂.carrier
          U₁ U₂ : Set G
          h1U₁ : IsOpen U₁
          h1U₂ : IsOpen U₂
          h2U₁ : HasSubset.Subset K₁.carrier U₁
          h2U₂ : HasSubset.Subset K₂.carrier U₂
          hU : Disjoint U₁ U₂
          L₁ : Set G
          h1L₁ : Membership.mem (nhds 1) L₁
          V₁ : Set G
          h1V₁ : HasSubset.Subset V₁ L₁
          h2V₁ : IsOpen V₁
          h3V₁ : Membership.mem V₁ 1
          h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
          L₂ : Set G
          h1L₂ : Membership.mem (nhds 1) L₂
          V₂ : Set G
          h1V₂ : HasSubset.Subset V₂ L₂
          h2V₂ : IsOpen V₂
          h3V₂ : Membership.mem V₂ 1
          h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
          eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
          this : Continuous eval
          V : Set G := Inter.inter V₁ V₂
          U : Set G
          h1U : HasSubset.Subset U ↑{ carrier := Inv.inv V, is_open' := ⋯, mem' := ⋯ }.t …
          h2U : IsOpen U
          h3U : Membership.mem U 1
          ⊢ HasSubset.Subset (Inv.inv U) V₂
        -/
        exact Subset.trans (inv_subset.mpr h1U) inter_subset_right
        /-
          🎉 no goals
        -/
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      h : Disjoint K₁.carrier K₂.carrier
      h₂ : IsClosed K₂.carrier
      U₁ U₂ : Set G
      h1U₁ : IsOpen U₁
      h1U₂ : IsOpen U₂
      h2U₁ : HasSubset.Subset K₁.carrier U₁
      h2U₂ : HasSubset.Subset K₂.carrier U₂
      hU : Disjoint U₁ U₂
      L₁ : Set G
      h1L₁ : Membership.mem (nhds 1) L₁
      V₁ : Set G
      h1V₁ : HasSubset.Subset V₁ L₁
      h2V₁ : IsOpen V₁
      h3V₁ : Membership.mem V₁ 1
      h2L₁ : HasSubset.Subset (HMul.hMul K₁.carrier V₁) U₁
      L₂ : Set G
      h1L₂ : Membership.mem (nhds 1) L₂
      V₂ : Set G
      h1V₂ : HasSubset.Subset V₂ L₂
      h2V₂ : IsOpen V₂
      h3V₂ : Membership.mem V₂ 1
      h2L₂ : HasSubset.Subset (HMul.hMul K₂.carrier V₂) U₂
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (HAdd …
      this : Continuous eval
      V : Set G := Inter.inter V₁ V₂
      ⊢ IsClosed (Set.preimage eval (Singleton.singleton 0))
    -/
  · apply continuous_iff_isClosed.mp this; exact isClosed_singleton
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive is_left_invariant_addCHaar]
theorem is_left_invariant_chaar {K₀ : PositiveCompacts G} (g : G) (K : Compacts G) :
    chaar K₀ (K.map _ <| continuous_mul_left g) = chaar K₀ K := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    g : G
    K : TopologicalSpace.Compacts G
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (TopologicalSpace.Compacts.map (fun  …
  -/
  let eval : (Compacts G → ℝ) → ℝ := fun f => f (K.map _ <| continuous_mul_left g) - f K
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    g : G
    K : TopologicalSpace.Compacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f (T …
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (TopologicalSpace.Compacts.map (fun  …
  -/
  have : Continuous eval := (continuous_apply (K.map _ _)).sub (continuous_apply K)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    g : G
    K : TopologicalSpace.Compacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f (T …
    this : Continuous eval
    ⊢ Eq (MeasureTheory.Measure.haar.chaar K₀ (TopologicalSpace.Compacts.map (fun  …
  -/
  rw [← sub_eq_zero]; show chaar K₀ ∈ eval ⁻¹' {(0 : ℝ)}
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    g : G
    K : TopologicalSpace.Compacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f (T …
    this : Continuous eval
    ⊢ Membership.mem (Set.preimage eval (Singleton.singleton 0)) (MeasureTheory.Me …
  -/
  apply mem_of_subset_of_mem _ (chaar_mem_clPrehaar K₀ ⊤)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    g : G
    K : TopologicalSpace.Compacts G
    eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f (T …
    this : Continuous eval
    ⊢ HasSubset.Subset (MeasureTheory.Measure.haar.clPrehaar (↑K₀) Top.top) (Set.p …
  -/
  unfold clPrehaar; rw [IsClosed.closure_subset_iff]
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      g : G
      K : TopologicalSpace.Compacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f (T …
      this : Continuous eval
      ⊢ HasSubset.Subset (Set.image (MeasureTheory.Measure.haar.prehaar ↑K₀) (setOf  …
    -/
  · rintro _ ⟨U, ⟨_, h2U, h3U⟩, rfl⟩
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      g : G
      K : TopologicalSpace.Compacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f (T …
      this : Continuous eval
      U : Set G
      left✝ : HasSubset.Subset U ↑Top.top.toOpens
      h2U : IsOpen U
      h3U : Membership.mem U 1
      ⊢ Membership.mem (Set.preimage eval (Singleton.singleton 0)) (MeasureTheory.Me …
    -/
    simp only [eval, mem_singleton_iff, mem_preimage, sub_eq_zero]
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      g : G
      K : TopologicalSpace.Compacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f (T …
      this : Continuous eval
      U : Set G
      left✝ : HasSubset.Subset U ↑Top.top.toOpens
      h2U : IsOpen U
      h3U : Membership.mem U 1
      ⊢ Eq (MeasureTheory.Measure.haar.prehaar (↑K₀) U (TopologicalSpace.Compacts.ma …
    -/
    apply is_left_invariant_prehaar; rw [h2U.interior_eq]; exact ⟨1, h3U⟩
                                                           /-
                                                             🎉 no goals
                                                           -/
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      g : G
      K : TopologicalSpace.Compacts G
      eval : (TopologicalSpace.Compacts G → Real) → Real := fun f => HSub.hSub (f (T …
      this : Continuous eval
      ⊢ IsClosed (Set.preimage eval (Singleton.singleton 0))
    -/
  · apply continuous_iff_isClosed.mp this; exact isClosed_singleton
                                           /-
                                             🎉 no goals
                                           -/


/-- The function `chaar` interpreted in `ℝ≥0`, as a content -/
@[to_additive "additive version of `MeasureTheory.Measure.haar.haarContent`"]
noncomputable def haarContent (K₀ : PositiveCompacts G) : Content G where
  toFun K := ⟨chaar K₀ K, chaar_nonneg _ _⟩
                      /-
                        G : Type u_1
                        inst✝² : Group G
                        inst✝¹ : TopologicalSpace G
                        inst✝ : TopologicalGroup G
                        K₀ : TopologicalSpace.PositiveCompacts G
                        K₁ K₂ : TopologicalSpace.Compacts G
                        h : HasSubset.Subset ↑K₁ ↑K₂
                        ⊢ LE.le ((fun K => ⟨MeasureTheory.Measure.haar.chaar K₀ K, ⋯⟩) K₁) ((fun K =>  …
                      -/
  mono' K₁ K₂ h := by simp only [← NNReal.coe_le_coe, NNReal.toReal, chaar_mono, h]
                      /-
                        🎉 no goals
                      -/
                                     /-
                                       G : Type u_1
                                       inst✝² : Group G
                                       inst✝¹ : TopologicalSpace G
                                       inst✝ : TopologicalGroup G
                                       K₀ : TopologicalSpace.PositiveCompacts G
                                       K₁ K₂ : TopologicalSpace.Compacts G
                                       h : Disjoint ↑K₁ ↑K₂
                                       _h₁ : IsClosed ↑K₁
                                       h₂ : IsClosed ↑K₂
                                       ⊢ Eq ((fun K => ⟨MeasureTheory.Measure.haar.chaar K₀ K, ⋯⟩) (Max.max K₁ K₂)) ( …
                                     -/
  sup_disjoint' K₁ K₂ h _h₁ h₂ := by simp only [chaar_sup_eq h]; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  sup_le' K₁ K₂ := by
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      ⊢ LE.le ((fun K => ⟨MeasureTheory.Measure.haar.chaar K₀ K, ⋯⟩) (Max.max K₁ K₂) …
    -/
    simp only [← NNReal.coe_le_coe, NNReal.coe_add]
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      K₀ : TopologicalSpace.PositiveCompacts G
      K₁ K₂ : TopologicalSpace.Compacts G
      ⊢ LE.le (↑⟨MeasureTheory.Measure.haar.chaar K₀ (Max.max K₁ K₂), ⋯⟩) (HAdd.hAdd …
    -/
    simp only [NNReal.toReal, chaar_sup_le]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem haarContent_apply (K₀ : PositiveCompacts G) (K : Compacts G) :
    haarContent K₀ K = show NNReal from ⟨chaar K₀ K, chaar_nonneg _ _⟩ :=
  rfl


/-- The variant of `chaar_self` for `haarContent` -/
@[to_additive "The variant of `addCHaar_self` for `addHaarContent`."]
theorem haarContent_self {K₀ : PositiveCompacts G} : haarContent K₀ K₀.toCompacts = 1 := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ Eq ((fun s => ↑((MeasureTheory.Measure.haar.haarContent K₀).toFun s)) K₀.toC …
  -/
  simp_rw [← ENNReal.coe_one, haarContent_apply, ENNReal.coe_inj, chaar_self]; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- The variant of `is_left_invariant_chaar` for `haarContent` -/
@[to_additive "The variant of `is_left_invariant_addCHaar` for `addHaarContent`"]
theorem is_left_invariant_haarContent {K₀ : PositiveCompacts G} (g : G) (K : Compacts G) :
    haarContent K₀ (K.map _ <| continuous_mul_left g) = haarContent K₀ K := by
  simpa only [ENNReal.coe_inj, ← NNReal.coe_inj, haarContent_apply] using
    is_left_invariant_chaar g K


@[to_additive]
theorem haarContent_outerMeasure_self_pos (K₀ : PositiveCompacts G) :
    0 < (haarContent K₀).outerMeasure K₀ := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ LT.lt 0 ((MeasureTheory.Measure.haar.haarContent K₀).outerMeasure ↑K₀)
  -/
  refine zero_lt_one.trans_le ?_
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ LE.le 1 ((MeasureTheory.Measure.haar.haarContent K₀).outerMeasure ↑K₀)
  -/
  rw [Content.outerMeasure_eq_iInf]
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ LE.le 1 (iInf fun U => iInf fun hU => iInf fun x => (MeasureTheory.Measure.h …
  -/
  refine le_iInf₂ fun U hU => le_iInf fun hK₀ => le_trans ?_ <| le_iSup₂ K₀.toCompacts hK₀
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    K₀ : TopologicalSpace.PositiveCompacts G
    U : Set G
    hU : IsOpen U
    hK₀ : HasSubset.Subset (↑K₀) U
    ⊢ LE.le 1 ((fun s => ↑((MeasureTheory.Measure.haar.haarContent K₀).toFun s)) K …
  -/
  exact haarContent_self.ge
  /-
    🎉 no goals
  -/


@[to_additive]
theorem haarContent_outerMeasure_closure_pos (K₀ : PositiveCompacts G) :
    0 < (haarContent K₀).outerMeasure (closure K₀) :=
  (haarContent_outerMeasure_self_pos K₀).trans_le (OuterMeasure.mono _ subset_closure)


/-- The Haar measure on the locally compact group `G`, scaled so that `haarMeasure K₀ K₀ = 1`. -/
@[to_additive
"The Haar measure on the locally compact additive group `G`, scaled so that
`addHaarMeasure K₀ K₀ = 1`."]
noncomputable def haarMeasure (K₀ : PositiveCompacts G) : Measure G :=
  ((haarContent K₀).measure K₀)⁻¹ • (haarContent K₀).measure


@[to_additive]
theorem haarMeasure_apply {K₀ : PositiveCompacts G} {s : Set G} (hs : MeasurableSet s) :
    haarMeasure K₀ s = (haarContent K₀).outerMeasure s / (haarContent K₀).measure K₀ := by
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    s : Set G
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.haarMeasure K₀) s) (HDiv.hDiv ((MeasureTheory.Mea …
  -/
  change ((haarContent K₀).measure K₀)⁻¹ * (haarContent K₀).measure s = _
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    s : Set G
    hs : MeasurableSet s
    ⊢ Eq (HMul.hMul (Inv.inv ((MeasureTheory.Measure.haar.haarContent K₀).measure  …
  -/
  simp only [hs, div_eq_mul_inv, mul_comm, Content.measure_apply]
  /-
    🎉 no goals
  -/


@[to_additive]
instance isMulLeftInvariant_haarMeasure (K₀ : PositiveCompacts G) :
    IsMulLeftInvariant (haarMeasure K₀) := by
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ (MeasureTheory.Measure.haarMeasure K₀).IsMulLeftInvariant
  -/
  rw [← forall_measure_preimage_mul_iff]
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ ∀ (g : G) (A : Set G), MeasurableSet A → Eq ((MeasureTheory.Measure.haarMeas …
  -/
  intro g A hA
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    g : G
    A : Set G
    hA : MeasurableSet A
    ⊢ Eq ((MeasureTheory.Measure.haarMeasure K₀) (Set.preimage (fun h => HMul.hMul …
  -/
  rw [haarMeasure_apply hA, haarMeasure_apply (measurable_const_mul g hA)]
  -- Porting note: Here was `congr 1`, but `to_additive` failed to generate a theorem.
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    g : G
    A : Set G
    hA : MeasurableSet A
    ⊢ Eq (HDiv.hDiv ((MeasureTheory.Measure.haar.haarContent K₀).outerMeasure (Set …
  -/
  refine congr_arg (fun x : ℝ≥0∞ => x / (haarContent K₀).measure K₀) ?_
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    g : G
    A : Set G
    hA : MeasurableSet A
    ⊢ Eq ((MeasureTheory.Measure.haar.haarContent K₀).outerMeasure (Set.preimage ( …
  -/
  apply Content.is_mul_left_invariant_outerMeasure
  /-
    case h
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    g : G
    A : Set G
    hA : MeasurableSet A
    ⊢ ∀ (g : G) {K : TopologicalSpace.Compacts G}, Eq ((fun s => ↑((MeasureTheory. …
  -/
  apply is_left_invariant_haarContent
  /-
    🎉 no goals
  -/


@[to_additive]
theorem haarMeasure_self {K₀ : PositiveCompacts G} : haarMeasure K₀ K₀ = 1 := by
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ Eq ((MeasureTheory.Measure.haarMeasure K₀) ↑K₀) 1
  -/
  haveI : LocallyCompactSpace G := K₀.locallyCompactSpace_of_group
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    this : LocallyCompactSpace G
    ⊢ Eq ((MeasureTheory.Measure.haarMeasure K₀) ↑K₀) 1
  -/
  simp only [haarMeasure, coe_smul, Pi.smul_apply, smul_eq_mul]
  rw [← K₀.isCompact.measure_closure,
    Content.measure_apply _ isClosed_closure.measurableSet, ENNReal.inv_mul_cancel]
    /-
      case h0
      G : Type u_1
      inst✝⁴ : Group G
      inst✝³ : TopologicalSpace G
      inst✝² : TopologicalGroup G
      inst✝¹ : MeasurableSpace G
      inst✝ : BorelSpace G
      K₀ : TopologicalSpace.PositiveCompacts G
      this : LocallyCompactSpace G
      ⊢ Ne ((MeasureTheory.Measure.haar.haarContent K₀).outerMeasure (closure ↑K₀)) 0
    -/
  · exact (haarContent_outerMeasure_closure_pos K₀).ne'
    /-
      🎉 no goals
    -/
    /-
      case ht
      G : Type u_1
      inst✝⁴ : Group G
      inst✝³ : TopologicalSpace G
      inst✝² : TopologicalGroup G
      inst✝¹ : MeasurableSpace G
      inst✝ : BorelSpace G
      K₀ : TopologicalSpace.PositiveCompacts G
      this : LocallyCompactSpace G
      ⊢ Ne ((MeasureTheory.Measure.haar.haarContent K₀).outerMeasure (closure ↑K₀))  …
    -/
  · exact (Content.outerMeasure_lt_top_of_isCompact _ K₀.isCompact.closure).ne
    /-
      🎉 no goals
    -/


/-- The Haar measure is regular. -/
@[to_additive "The additive Haar measure is regular."]
instance regular_haarMeasure {K₀ : PositiveCompacts G} : (haarMeasure K₀).Regular := by
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ (MeasureTheory.Measure.haarMeasure K₀).Regular
  -/
  haveI : LocallyCompactSpace G := K₀.locallyCompactSpace_of_group
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    this : LocallyCompactSpace G
    ⊢ (MeasureTheory.Measure.haarMeasure K₀).Regular
  -/
  apply Regular.smul
  rw [← K₀.isCompact.measure_closure,
    Content.measure_apply _ isClosed_closure.measurableSet, ENNReal.inv_ne_top]
  /-
    case hx
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    this : LocallyCompactSpace G
    ⊢ Ne ((MeasureTheory.Measure.haar.haarContent K₀).outerMeasure (closure ↑K₀)) 0
  -/
  exact (haarContent_outerMeasure_closure_pos K₀).ne'
  /-
    🎉 no goals
  -/


@[to_additive]
theorem haarMeasure_closure_self {K₀ : PositiveCompacts G} : haarMeasure K₀ (closure K₀) = 1 := by
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ Eq ((MeasureTheory.Measure.haarMeasure K₀) (closure ↑K₀)) 1
  -/
  rw [K₀.isCompact.measure_closure, haarMeasure_self]
  /-
    🎉 no goals
  -/


/-- The Haar measure is sigma-finite in a second countable group. -/
@[to_additive "The additive Haar measure is sigma-finite in a second countable group."]
instance sigmaFinite_haarMeasure [SecondCountableTopology G] {K₀ : PositiveCompacts G} :
    SigmaFinite (haarMeasure K₀) := by
  /-
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : MeasurableSpace G
    inst✝¹ : BorelSpace G
    inst✝ : SecondCountableTopology G
    K₀ : TopologicalSpace.PositiveCompacts G
    ⊢ MeasureTheory.SigmaFinite (MeasureTheory.Measure.haarMeasure K₀)
  -/
  haveI : LocallyCompactSpace G := K₀.locallyCompactSpace_of_group; infer_instance
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- The Haar measure is a Haar measure, i.e., it is invariant and gives finite mass to compact
sets and positive mass to nonempty open sets. -/
@[to_additive
"The additive Haar measure is an additive Haar measure, i.e., it is invariant and gives finite mass
to compact sets and positive mass to nonempty open sets."]
instance isHaarMeasure_haarMeasure (K₀ : PositiveCompacts G) : IsHaarMeasure (haarMeasure K₀) := by
  apply
    isHaarMeasure_of_isCompact_nonempty_interior (haarMeasure K₀) K₀ K₀.isCompact
      K₀.interior_nonempty
    /-
      case h
      G : Type u_1
      inst✝⁴ : Group G
      inst✝³ : TopologicalSpace G
      inst✝² : TopologicalGroup G
      inst✝¹ : MeasurableSpace G
      inst✝ : BorelSpace G
      K₀ : TopologicalSpace.PositiveCompacts G
      ⊢ Ne ((MeasureTheory.Measure.haarMeasure K₀) ↑K₀) 0
    -/
  · simp only [haarMeasure_self]; exact one_ne_zero
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case h'
      G : Type u_1
      inst✝⁴ : Group G
      inst✝³ : TopologicalSpace G
      inst✝² : TopologicalGroup G
      inst✝¹ : MeasurableSpace G
      inst✝ : BorelSpace G
      K₀ : TopologicalSpace.PositiveCompacts G
      ⊢ Ne ((MeasureTheory.Measure.haarMeasure K₀) ↑K₀) Top.top
    -/
  · simp only [haarMeasure_self, ne_eq, ENNReal.one_ne_top, not_false_eq_true]
    /-
      🎉 no goals
    -/


/-- `haar` is some choice of a Haar measure, on a locally compact group. -/
@[to_additive
"`addHaar` is some choice of a Haar measure, on a locally compact additive group."]
noncomputable abbrev haar [LocallyCompactSpace G] : Measure G :=
  haarMeasure <| Classical.arbitrary _


/-- **Steinhaus Theorem** In any locally compact group `G` with an inner regular Haar measure `μ`,
for any measurable set `E` of positive measure, the set `E / E` is a neighbourhood of `1`. -/
@[to_additive
"**Steinhaus Theorem** In any locally compact group `G` with an inner regular Haar measure `μ`,
for any measurable set `E` of positive measure, the set `E - E` is a neighbourhood of `0`."]
theorem div_mem_nhds_one_of_haar_pos (μ : Measure G) [IsHaarMeasure μ] [LocallyCompactSpace G]
    [InnerRegular μ] (E : Set G) (hE : MeasurableSet E) (hEpos : 0 < μ E) :
    E / E ∈ 𝓝 (1 : G) := by
  /- For any inner regular measure `μ` and set `E` of positive measure, we can find a compact
    set `K` of positive measure inside `E`. Further, there exists a neighborhood `V` of the
    identity such that `v • K \ K` has small measure for all `v ∈ V`, say `< μ K`.
    Then `v • K` and `K` can not be disjoint, as otherwise `μ (v • K \ K) = μ (v • K) = μ K`.
    This show that `K / K` contains the neighborhood `V` of `1`, and therefore that it is
    itself such a neighborhood. -/
  obtain ⟨K, hKE, hK, K_closed, hKpos⟩ :
      ∃ (K : Set G), K ⊆ E ∧ IsCompact K ∧ IsClosed K ∧ 0 < μ K := by
    rcases MeasurableSet.exists_lt_isCompact hE hEpos with ⟨K, KE, K_comp, K_meas⟩
    refine ⟨closure K, ?_, K_comp.closure, isClosed_closure, ?_⟩
    · exact K_comp.closure_subset_measurableSet hE KE
    · rwa [K_comp.measure_closure]
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.InnerRegular
    E : Set G
    hE : MeasurableSet E
    hEpos : LT.lt 0 (μ E)
    K : Set G
    hKE : HasSubset.Subset K E
    hK : IsCompact K
    K_closed : IsClosed K
    hKpos : LT.lt 0 (μ K)
    ⊢ Membership.mem (nhds 1) (HDiv.hDiv E E)
  -/
  filter_upwards [eventually_nhds_one_measure_smul_diff_lt hK K_closed hKpos.ne' (μ := μ)] with g hg
  have : ¬Disjoint (g • K) K := fun hd ↦ by
    rw [hd.symm.sdiff_eq_right, measure_smul] at hg
    exact hg.false
  /-
    case h
    G : Type u_1
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.InnerRegular
    E : Set G
    hE : MeasurableSet E
    hEpos : LT.lt 0 (μ E)
    K : Set G
    hKE : HasSubset.Subset K E
    hK : IsCompact K
    K_closed : IsClosed K
    hKpos : LT.lt 0 (μ K)
    g : G
    hg : LT.lt (μ (SDiff.sdiff (HSMul.hSMul g K) K)) (μ K)
    this : Not (Disjoint (HSMul.hSMul g K) K)
    ⊢ Membership.mem (HDiv.hDiv E E) g
  -/
  rcases Set.not_disjoint_iff.1 this with ⟨_, ⟨x, hxK, rfl⟩, hgxK⟩
  /-
    case h.intro.intro.intro.intro
    G : Type u_1
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsHaarMeasure
    inst✝¹ : LocallyCompactSpace G
    inst✝ : μ.InnerRegular
    E : Set G
    hE : MeasurableSet E
    hEpos : LT.lt 0 (μ E)
    K : Set G
    hKE : HasSubset.Subset K E
    hK : IsCompact K
    K_closed : IsClosed K
    hKpos : LT.lt 0 (μ K)
    g : G
    hg : LT.lt (μ (SDiff.sdiff (HSMul.hSMul g K) K)) (μ K)
    this : Not (Disjoint (HSMul.hSMul g K) K)
    x : G
    hxK : Membership.mem K x
    hgxK : Membership.mem K ((fun x => HSMul.hSMul g x) x)
    ⊢ Membership.mem (HDiv.hDiv E E) g
  -/
  simpa using div_mem_div (hKE hgxK) (hKE hxK)
  /-
    🎉 no goals
  -/



/-- **Uniqueness of left-invariant measures**: In a second-countable locally compact group, any
  σ-finite left-invariant measure is a scalar multiple of the Haar measure.
  This is slightly weaker than assuming that `μ` is a Haar measure (in particular we don't require
  `μ ≠ 0`).
  See also `isMulLeftInvariant_eq_smul_of_regular`
  for a statement not assuming second-countability. -/
@[to_additive
"**Uniqueness of left-invariant measures**: In a second-countable locally compact additive group,
  any σ-finite left-invariant measure is a scalar multiple of the additive Haar measure.
  This is slightly weaker than assuming that `μ` is a additive Haar measure (in particular we don't
  require `μ ≠ 0`).
  See also `isAddLeftInvariant_eq_smul_of_regular`
  for a statement not assuming second-countability."]
theorem haarMeasure_unique (μ : Measure G) [SigmaFinite μ] [IsMulLeftInvariant μ]
    (K₀ : PositiveCompacts G) : μ = μ K₀ • haarMeasure K₀ := by
  have A : Set.Nonempty (interior (closure (K₀ : Set G))) :=
    K₀.interior_nonempty.mono (interior_mono subset_closure)
  have := measure_eq_div_smul μ (haarMeasure K₀)
    (measure_pos_of_nonempty_interior _ A).ne' K₀.isCompact.closure.measure_ne_top
  /-
    G : Type u_1
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : SecondCountableTopology G
    μ : MeasureTheory.Measure G
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : μ.IsMulLeftInvariant
    K₀ : TopologicalSpace.PositiveCompacts G
    A : (interior (closure ↑K₀)).Nonempty
    this : Eq μ (HSMul.hSMul (HDiv.hDiv (μ (closure ↑K₀)) ((MeasureTheory.Measure. …
    ⊢ Eq μ (HSMul.hSMul (μ ↑K₀) (MeasureTheory.Measure.haarMeasure K₀))
  -/
  rwa [haarMeasure_closure_self, div_one, K₀.isCompact.measure_closure] at this
  /-
    🎉 no goals
  -/


/-- Let `μ` be a σ-finite left invariant measure on `G`. Then `μ` is equal to the Haar measure
defined by `K₀` iff `μ K₀ = 1`. -/
@[to_additive]
theorem haarMeasure_eq_iff (K₀ : PositiveCompacts G) (μ : Measure G) [SigmaFinite μ]
    [IsMulLeftInvariant μ] :
    haarMeasure K₀ = μ ↔ μ K₀ = 1 :=
                                                   /-
                                                     G : Type u_1
                                                     inst✝⁷ : Group G
                                                     inst✝⁶ : TopologicalSpace G
                                                     inst✝⁵ : TopologicalGroup G
                                                     inst✝⁴ : MeasurableSpace G
                                                     inst✝³ : BorelSpace G
                                                     inst✝² : SecondCountableTopology G
                                                     K₀ : TopologicalSpace.PositiveCompacts G
                                                     μ : MeasureTheory.Measure G
                                                     inst✝¹ : MeasureTheory.SigmaFinite μ
                                                     inst✝ : μ.IsMulLeftInvariant
                                                     h : Eq (μ ↑K₀) 1
                                                     ⊢ Eq (MeasureTheory.Measure.haarMeasure K₀) μ
                                                   -/
  ⟨fun h => h.symm ▸ haarMeasure_self, fun h => by rw [haarMeasure_unique μ K₀, h, one_smul]⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- To show that an invariant σ-finite measure is regular it is sufficient to show that it is finite
  on some compact set with non-empty interior. -/
@[to_additive
"To show that an invariant σ-finite measure is regular it is sufficient to show that it is finite on
some compact set with non-empty interior."]
theorem regular_of_isMulLeftInvariant {μ : Measure G} [SigmaFinite μ] [IsMulLeftInvariant μ]
    {K : Set G} (hK : IsCompact K) (h2K : (interior K).Nonempty) (hμK : μ K ≠ ∞) : Regular μ := by
  /-
    G : Type u_1
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    inst✝² : SecondCountableTopology G
    μ : MeasureTheory.Measure G
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : μ.IsMulLeftInvariant
    K : Set G
    hK : IsCompact K
    h2K : (interior K).Nonempty
    hμK : Ne (μ K) Top.top
    ⊢ μ.Regular
  -/
  rw [haarMeasure_unique μ ⟨⟨K, hK⟩, h2K⟩]; exact Regular.smul hμK
                                            /-
                                              🎉 no goals
                                            -/


