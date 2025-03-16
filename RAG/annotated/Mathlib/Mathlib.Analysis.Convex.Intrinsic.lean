/-- The intrinsic interior of a set is its interior considered as a set in its affine span. -/
def intrinsicInterior (s : Set P) : Set P :=
  (↑) '' interior ((↑) ⁻¹' s : Set <| affineSpan 𝕜 s)


/-- The intrinsic frontier of a set is its frontier considered as a set in its affine span. -/
def intrinsicFrontier (s : Set P) : Set P :=
  (↑) '' frontier ((↑) ⁻¹' s : Set <| affineSpan 𝕜 s)


/-- The intrinsic closure of a set is its closure considered as a set in its affine span. -/
def intrinsicClosure (s : Set P) : Set P :=
  (↑) '' closure ((↑) ⁻¹' s : Set <| affineSpan 𝕜 s)


@[simp]
theorem mem_intrinsicInterior :
    x ∈ intrinsicInterior 𝕜 s ↔ ∃ y, y ∈ interior ((↑) ⁻¹' s : Set <| affineSpan 𝕜 s) ∧ ↑y = x :=
  mem_image _ _ _


@[simp]
theorem mem_intrinsicFrontier :
    x ∈ intrinsicFrontier 𝕜 s ↔ ∃ y, y ∈ frontier ((↑) ⁻¹' s : Set <| affineSpan 𝕜 s) ∧ ↑y = x :=
  mem_image _ _ _


@[simp]
theorem mem_intrinsicClosure :
    x ∈ intrinsicClosure 𝕜 s ↔ ∃ y, y ∈ closure ((↑) ⁻¹' s : Set <| affineSpan 𝕜 s) ∧ ↑y = x :=
  mem_image _ _ _


theorem intrinsicInterior_subset : intrinsicInterior 𝕜 s ⊆ s :=
  image_subset_iff.2 interior_subset


theorem intrinsicFrontier_subset (hs : IsClosed s) : intrinsicFrontier 𝕜 s ⊆ s :=
  image_subset_iff.2 (hs.preimage continuous_induced_dom).frontier_subset


theorem intrinsicFrontier_subset_intrinsicClosure : intrinsicFrontier 𝕜 s ⊆ intrinsicClosure 𝕜 s :=
  image_subset _ frontier_subset_closure


theorem subset_intrinsicClosure : s ⊆ intrinsicClosure 𝕜 s :=
  fun x hx => ⟨⟨x, subset_affineSpan _ _ hx⟩, subset_closure hx, rfl⟩


@[simp]
                                                                            /-
                                                                              𝕜 : Type u_1
                                                                              V : Type u_2
                                                                              P : Type u_5
                                                                              inst✝⁴ : Ring 𝕜
                                                                              inst✝³ : AddCommGroup V
                                                                              inst✝² : Module 𝕜 V
                                                                              inst✝¹ : TopologicalSpace P
                                                                              inst✝ : AddTorsor V P
                                                                              ⊢ Eq (intrinsicInterior 𝕜 EmptyCollection.emptyCollection) EmptyCollection.emp …
                                                                            -/
theorem intrinsicInterior_empty : intrinsicInterior 𝕜 (∅ : Set P) = ∅ := by simp [intrinsicInterior]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
                                                                            /-
                                                                              𝕜 : Type u_1
                                                                              V : Type u_2
                                                                              P : Type u_5
                                                                              inst✝⁴ : Ring 𝕜
                                                                              inst✝³ : AddCommGroup V
                                                                              inst✝² : Module 𝕜 V
                                                                              inst✝¹ : TopologicalSpace P
                                                                              inst✝ : AddTorsor V P
                                                                              ⊢ Eq (intrinsicFrontier 𝕜 EmptyCollection.emptyCollection) EmptyCollection.emp …
                                                                            -/
theorem intrinsicFrontier_empty : intrinsicFrontier 𝕜 (∅ : Set P) = ∅ := by simp [intrinsicFrontier]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
                                                                          /-
                                                                            𝕜 : Type u_1
                                                                            V : Type u_2
                                                                            P : Type u_5
                                                                            inst✝⁴ : Ring 𝕜
                                                                            inst✝³ : AddCommGroup V
                                                                            inst✝² : Module 𝕜 V
                                                                            inst✝¹ : TopologicalSpace P
                                                                            inst✝ : AddTorsor V P
                                                                            ⊢ Eq (intrinsicClosure 𝕜 EmptyCollection.emptyCollection) EmptyCollection.empt …
                                                                          -/
theorem intrinsicClosure_empty : intrinsicClosure 𝕜 (∅ : Set P) = ∅ := by simp [intrinsicClosure]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem intrinsicClosure_nonempty : (intrinsicClosure 𝕜 s).Nonempty ↔ s.Nonempty :=
      /-
        𝕜 : Type u_1
        V : Type u_2
        P : Type u_5
        inst✝⁴ : Ring 𝕜
        inst✝³ : AddCommGroup V
        inst✝² : Module 𝕜 V
        inst✝¹ : TopologicalSpace P
        inst✝ : AddTorsor V P
        s : Set P
        ⊢ (intrinsicClosure 𝕜 s).Nonempty → s.Nonempty
      -/
  ⟨by simp_rw [nonempty_iff_ne_empty]; rintro h rfl; exact h intrinsicClosure_empty,
                                                     /-
                                                       🎉 no goals
                                                     -/
    Nonempty.mono subset_intrinsicClosure⟩


alias ⟨Set.Nonempty.ofIntrinsicClosure, Set.Nonempty.intrinsicClosure⟩ := intrinsicClosure_nonempty

--attribute [protected] Set.Nonempty.intrinsicClosure -- Porting note: removed


@[simp]
theorem intrinsicInterior_singleton (x : P) : intrinsicInterior 𝕜 ({x} : Set P) = {x} := by
  simp only [intrinsicInterior, preimage_coe_affineSpan_singleton, interior_univ, image_univ,
    Subtype.range_coe_subtype, mem_affineSpan_singleton, setOf_eq_eq_singleton]


@[simp]
theorem intrinsicFrontier_singleton (x : P) : intrinsicFrontier 𝕜 ({x} : Set P) = ∅ := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    x : P
    ⊢ Eq (intrinsicFrontier 𝕜 (Singleton.singleton x)) EmptyCollection.emptyCollec …
  -/
  rw [intrinsicFrontier, preimage_coe_affineSpan_singleton, frontier_univ, image_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem intrinsicClosure_singleton (x : P) : intrinsicClosure 𝕜 ({x} : Set P) = {x} := by
  simp only [intrinsicClosure, preimage_coe_affineSpan_singleton, closure_univ, image_univ,
    Subtype.range_coe_subtype, mem_affineSpan_singleton, setOf_eq_eq_singleton]


theorem intrinsicClosure_mono (h : s ⊆ t) : intrinsicClosure 𝕜 s ⊆ intrinsicClosure 𝕜 t := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s t : Set P
    h : HasSubset.Subset s t
    ⊢ HasSubset.Subset (intrinsicClosure 𝕜 s) (intrinsicClosure 𝕜 t)
  -/
  refine image_subset_iff.2 fun x hx => ?_
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s t : Set P
    h : HasSubset.Subset s t
    x : Subtype fun x => Membership.mem (affineSpan 𝕜 s) x
    hx : Membership.mem (closure (Set.preimage Subtype.val s)) x
    ⊢ Membership.mem (Set.preimage Subtype.val (intrinsicClosure 𝕜 t)) x
  -/
  refine ⟨Set.inclusion (affineSpan_mono _ h) x, ?_, rfl⟩
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s t : Set P
    h : HasSubset.Subset s t
    x : Subtype fun x => Membership.mem (affineSpan 𝕜 s) x
    hx : Membership.mem (closure (Set.preimage Subtype.val s)) x
    ⊢ Membership.mem (closure (Set.preimage Subtype.val t)) (Set.inclusion ⋯ x)
  -/
  refine (continuous_inclusion (affineSpan_mono _ h)).closure_preimage_subset _ (closure_mono ?_ hx)
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s t : Set P
    h : HasSubset.Subset s t
    x : Subtype fun x => Membership.mem (affineSpan 𝕜 s) x
    hx : Membership.mem (closure (Set.preimage Subtype.val s)) x
    ⊢ HasSubset.Subset (Set.preimage Subtype.val s) (Set.preimage (Set.inclusion ⋯ …
  -/
  exact fun y hy => h hy
  /-
    🎉 no goals
  -/


theorem interior_subset_intrinsicInterior : interior s ⊆ intrinsicInterior 𝕜 s :=
  fun x hx => ⟨⟨x, subset_affineSpan _ _ <| interior_subset hx⟩,
    preimage_interior_subset_interior_preimage continuous_subtype_val hx, rfl⟩


theorem intrinsicClosure_subset_closure : intrinsicClosure 𝕜 s ⊆ closure s :=
  image_subset_iff.2 <| continuous_subtype_val.closure_preimage_subset _


theorem intrinsicFrontier_subset_frontier : intrinsicFrontier 𝕜 s ⊆ frontier s :=
  image_subset_iff.2 <| continuous_subtype_val.frontier_preimage_subset _


theorem intrinsicClosure_subset_affineSpan : intrinsicClosure 𝕜 s ⊆ affineSpan 𝕜 s :=
  (image_subset_range _ _).trans Subtype.range_coe.subset


@[simp]
theorem intrinsicClosure_diff_intrinsicFrontier (s : Set P) :
    intrinsicClosure 𝕜 s \ intrinsicFrontier 𝕜 s = intrinsicInterior 𝕜 s :=
  (image_diff Subtype.coe_injective _ _).symm.trans <| by
    /-
      𝕜 : Type u_1
      V : Type u_2
      P : Type u_5
      inst✝⁴ : Ring 𝕜
      inst✝³ : AddCommGroup V
      inst✝² : Module 𝕜 V
      inst✝¹ : TopologicalSpace P
      inst✝ : AddTorsor V P
      s : Set P
      ⊢ Eq (Set.image (fun a => ↑a) (SDiff.sdiff (closure (Set.preimage Subtype.val  …
    -/
    rw [closure_diff_frontier, intrinsicInterior]
    /-
      🎉 no goals
    -/


@[simp]
theorem intrinsicClosure_diff_intrinsicInterior (s : Set P) :
    intrinsicClosure 𝕜 s \ intrinsicInterior 𝕜 s = intrinsicFrontier 𝕜 s :=
  (image_diff Subtype.coe_injective _ _).symm


@[simp]
theorem intrinsicInterior_union_intrinsicFrontier (s : Set P) :
    intrinsicInterior 𝕜 s ∪ intrinsicFrontier 𝕜 s = intrinsicClosure 𝕜 s := by
  simp [intrinsicClosure, intrinsicInterior, intrinsicFrontier, closure_eq_interior_union_frontier,
    image_union]


@[simp]
theorem intrinsicFrontier_union_intrinsicInterior (s : Set P) :
    intrinsicFrontier 𝕜 s ∪ intrinsicInterior 𝕜 s = intrinsicClosure 𝕜 s := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s : Set P
    ⊢ Eq (Union.union (intrinsicFrontier 𝕜 s) (intrinsicInterior 𝕜 s)) (intrinsicC …
  -/
  rw [union_comm, intrinsicInterior_union_intrinsicFrontier]
  /-
    🎉 no goals
  -/


theorem isClosed_intrinsicClosure (hs : IsClosed (affineSpan 𝕜 s : Set P)) :
    IsClosed (intrinsicClosure 𝕜 s) :=
  hs.isClosedEmbedding_subtypeVal.isClosedMap _ isClosed_closure


theorem isClosed_intrinsicFrontier (hs : IsClosed (affineSpan 𝕜 s : Set P)) :
    IsClosed (intrinsicFrontier 𝕜 s) :=
  hs.isClosedEmbedding_subtypeVal.isClosedMap _ isClosed_frontier


@[simp]
theorem affineSpan_intrinsicClosure (s : Set P) :
    affineSpan 𝕜 (intrinsicClosure 𝕜 s) = affineSpan 𝕜 s :=
  (affineSpan_le.2 intrinsicClosure_subset_affineSpan).antisymm <|
    affineSpan_mono _ subset_intrinsicClosure


protected theorem IsClosed.intrinsicClosure (hs : IsClosed ((↑) ⁻¹' s : Set <| affineSpan 𝕜 s)) :
    intrinsicClosure 𝕜 s = s := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s : Set P
    hs : IsClosed (Set.preimage Subtype.val s)
    ⊢ Eq (intrinsicClosure 𝕜 s) s
  -/
  rw [intrinsicClosure, hs.closure_eq, image_preimage_eq_of_subset]
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s : Set P
    hs : IsClosed (Set.preimage Subtype.val s)
    ⊢ HasSubset.Subset s (Set.range Subtype.val)
  -/
  exact (subset_affineSpan _ _).trans Subtype.range_coe.superset
  /-
    🎉 no goals
  -/


@[simp]
theorem intrinsicClosure_idem (s : Set P) :
    intrinsicClosure 𝕜 (intrinsicClosure 𝕜 s) = intrinsicClosure 𝕜 s := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s : Set P
    ⊢ Eq (intrinsicClosure 𝕜 (intrinsicClosure 𝕜 s)) (intrinsicClosure 𝕜 s)
  -/
  refine IsClosed.intrinsicClosure ?_
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s : Set P
    ⊢ IsClosed (Set.preimage Subtype.val (intrinsicClosure 𝕜 s))
  -/
  set t := affineSpan 𝕜 (intrinsicClosure 𝕜 s) with ht
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s : Set P
    t : AffineSubspace 𝕜 P := affineSpan 𝕜 (intrinsicClosure 𝕜 s)
    ht : Eq t (affineSpan 𝕜 (intrinsicClosure 𝕜 s))
    ⊢ IsClosed (Set.preimage Subtype.val (intrinsicClosure 𝕜 s))
  -/
  clear_value t
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s : Set P
    t : AffineSubspace 𝕜 P
    ht : Eq t (affineSpan 𝕜 (intrinsicClosure 𝕜 s))
    ⊢ IsClosed (Set.preimage Subtype.val (intrinsicClosure 𝕜 s))
  -/
  obtain rfl := ht.trans (affineSpan_intrinsicClosure _)
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s : Set P
    ht : Eq (affineSpan 𝕜 s) (affineSpan 𝕜 (intrinsicClosure 𝕜 s))
    ⊢ IsClosed (Set.preimage Subtype.val (intrinsicClosure 𝕜 s))
  -/
  rw [intrinsicClosure, preimage_image_eq _ Subtype.coe_injective]
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup V
    inst✝² : Module 𝕜 V
    inst✝¹ : TopologicalSpace P
    inst✝ : AddTorsor V P
    s : Set P
    ht : Eq (affineSpan 𝕜 s) (affineSpan 𝕜 (intrinsicClosure 𝕜 s))
    ⊢ IsClosed (closure (Set.preimage Subtype.val s))
  -/
  exact isClosed_closure
  /-
    🎉 no goals
  -/


@[simp]
theorem image_intrinsicInterior (φ : P →ᵃⁱ[𝕜] Q) (s : Set P) :
    intrinsicInterior 𝕜 (φ '' s) = φ '' intrinsicInterior 𝕜 s := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    ⊢ Eq (intrinsicInterior 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicInteri …
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      𝕜 : Type u_1
      V : Type u_2
      W : Type u_3
      Q : Type u_4
      P : Type u_5
      inst✝⁸ : NormedField 𝕜
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : NormedSpace 𝕜 V
      inst✝⁴ : NormedSpace 𝕜 W
      inst✝³ : MetricSpace P
      inst✝² : PseudoMetricSpace Q
      inst✝¹ : NormedAddTorsor V P
      inst✝ : NormedAddTorsor W Q
      φ : AffineIsometry 𝕜 P Q
      ⊢ Eq (intrinsicInterior 𝕜 (Set.image (⇑φ) EmptyCollection.emptyCollection)) (S …
    -/
  · simp only [intrinsicInterior_empty, image_empty]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    hs : s.Nonempty
    ⊢ Eq (intrinsicInterior 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicInteri …
  -/
  haveI : Nonempty s := hs.to_subtype
  /-
    case inr
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    hs : s.Nonempty
    this : Nonempty ↑s
    ⊢ Eq (intrinsicInterior 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicInteri …
  -/
  let f := ((affineSpan 𝕜 s).isometryEquivMap φ).toHomeomorph
  /-
    case inr
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    hs : s.Nonempty
    this : Nonempty ↑s
    f : Homeomorph (Subtype fun x => Membership.mem (affineSpan 𝕜 s) x) (Subtype f …
    ⊢ Eq (intrinsicInterior 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicInteri …
  -/
  have : φ.toAffineMap ∘ (↑) ∘ f.symm = (↑) := funext isometryEquivMap.apply_symm_apply
  rw [intrinsicInterior, intrinsicInterior, ← φ.coe_toAffineMap, ← map_span φ.toAffineMap s, ← this,
    ← Function.comp_assoc, image_comp, image_comp, f.symm.image_interior, f.image_symm,
    ← preimage_comp, Function.comp_assoc, f.symm_comp_self, AffineIsometry.coe_toAffineMap,
    Function.comp_id, preimage_comp, φ.injective.preimage_image]


@[simp]
theorem image_intrinsicFrontier (φ : P →ᵃⁱ[𝕜] Q) (s : Set P) :
    intrinsicFrontier 𝕜 (φ '' s) = φ '' intrinsicFrontier 𝕜 s := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    ⊢ Eq (intrinsicFrontier 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicFronti …
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      𝕜 : Type u_1
      V : Type u_2
      W : Type u_3
      Q : Type u_4
      P : Type u_5
      inst✝⁸ : NormedField 𝕜
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : NormedSpace 𝕜 V
      inst✝⁴ : NormedSpace 𝕜 W
      inst✝³ : MetricSpace P
      inst✝² : PseudoMetricSpace Q
      inst✝¹ : NormedAddTorsor V P
      inst✝ : NormedAddTorsor W Q
      φ : AffineIsometry 𝕜 P Q
      ⊢ Eq (intrinsicFrontier 𝕜 (Set.image (⇑φ) EmptyCollection.emptyCollection)) (S …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    hs : s.Nonempty
    ⊢ Eq (intrinsicFrontier 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicFronti …
  -/
  haveI : Nonempty s := hs.to_subtype
  /-
    case inr
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    hs : s.Nonempty
    this : Nonempty ↑s
    ⊢ Eq (intrinsicFrontier 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicFronti …
  -/
  let f := ((affineSpan 𝕜 s).isometryEquivMap φ).toHomeomorph
  /-
    case inr
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    hs : s.Nonempty
    this : Nonempty ↑s
    f : Homeomorph (Subtype fun x => Membership.mem (affineSpan 𝕜 s) x) (Subtype f …
    ⊢ Eq (intrinsicFrontier 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicFronti …
  -/
  have : φ.toAffineMap ∘ (↑) ∘ f.symm = (↑) := funext isometryEquivMap.apply_symm_apply
  rw [intrinsicFrontier, intrinsicFrontier, ← φ.coe_toAffineMap, ← map_span φ.toAffineMap s, ← this,
    ← Function.comp_assoc, image_comp, image_comp, f.symm.image_frontier, f.image_symm,
    ← preimage_comp, Function.comp_assoc, f.symm_comp_self, AffineIsometry.coe_toAffineMap,
    Function.comp_id, preimage_comp, φ.injective.preimage_image]


@[simp]
theorem image_intrinsicClosure (φ : P →ᵃⁱ[𝕜] Q) (s : Set P) :
    intrinsicClosure 𝕜 (φ '' s) = φ '' intrinsicClosure 𝕜 s := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    ⊢ Eq (intrinsicClosure 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicClosure …
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      𝕜 : Type u_1
      V : Type u_2
      W : Type u_3
      Q : Type u_4
      P : Type u_5
      inst✝⁸ : NormedField 𝕜
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : NormedSpace 𝕜 V
      inst✝⁴ : NormedSpace 𝕜 W
      inst✝³ : MetricSpace P
      inst✝² : PseudoMetricSpace Q
      inst✝¹ : NormedAddTorsor V P
      inst✝ : NormedAddTorsor W Q
      φ : AffineIsometry 𝕜 P Q
      ⊢ Eq (intrinsicClosure 𝕜 (Set.image (⇑φ) EmptyCollection.emptyCollection)) (Se …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    hs : s.Nonempty
    ⊢ Eq (intrinsicClosure 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicClosure …
  -/
  haveI : Nonempty s := hs.to_subtype
  /-
    case inr
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    hs : s.Nonempty
    this : Nonempty ↑s
    ⊢ Eq (intrinsicClosure 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicClosure …
  -/
  let f := ((affineSpan 𝕜 s).isometryEquivMap φ).toHomeomorph
  /-
    case inr
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    Q : Type u_4
    P : Type u_5
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : SeminormedAddCommGroup W
    inst✝⁵ : NormedSpace 𝕜 V
    inst✝⁴ : NormedSpace 𝕜 W
    inst✝³ : MetricSpace P
    inst✝² : PseudoMetricSpace Q
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedAddTorsor W Q
    φ : AffineIsometry 𝕜 P Q
    s : Set P
    hs : s.Nonempty
    this : Nonempty ↑s
    f : Homeomorph (Subtype fun x => Membership.mem (affineSpan 𝕜 s) x) (Subtype f …
    ⊢ Eq (intrinsicClosure 𝕜 (Set.image (⇑φ) s)) (Set.image (⇑φ) (intrinsicClosure …
  -/
  have : φ.toAffineMap ∘ (↑) ∘ f.symm = (↑) := funext isometryEquivMap.apply_symm_apply
  rw [intrinsicClosure, intrinsicClosure, ← φ.coe_toAffineMap, ← map_span φ.toAffineMap s, ← this,
    ← Function.comp_assoc, image_comp, image_comp, f.symm.image_closure, f.image_symm,
    ← preimage_comp, Function.comp_assoc, f.symm_comp_self, AffineIsometry.coe_toAffineMap,
    Function.comp_id, preimage_comp, φ.injective.preimage_image]


@[simp]
theorem intrinsicClosure_eq_closure : intrinsicClosure 𝕜 s = closure s := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : CompleteSpace 𝕜
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace 𝕜 V
    inst✝² : FiniteDimensional 𝕜 V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    ⊢ Eq (intrinsicClosure 𝕜 s) (closure s)
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : CompleteSpace 𝕜
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace 𝕜 V
    inst✝² : FiniteDimensional 𝕜 V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    x : P
    ⊢ Iff (Membership.mem (intrinsicClosure 𝕜 s) x) (Membership.mem (closure s) x)
  -/
  simp only [mem_closure_iff, mem_intrinsicClosure]
  /-
    case h
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : CompleteSpace 𝕜
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace 𝕜 V
    inst✝² : FiniteDimensional 𝕜 V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    x : P
    ⊢ Iff (Exists fun y => And (∀ (o : Set (Subtype fun x => Membership.mem (affin …
  -/
  refine ⟨?_, fun h => ⟨⟨x, _⟩, ?_, Subtype.coe_mk _ ?_⟩⟩
    /-
      case h.refine_1
      𝕜 : Type u_1
      V : Type u_2
      P : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : CompleteSpace 𝕜
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : FiniteDimensional 𝕜 V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      x : P
      ⊢ (Exists fun y => And (∀ (o : Set (Subtype fun x => Membership.mem (affineSpa …
    -/
  · rintro ⟨x, h, rfl⟩ t ht hx
    /-
      case h.refine_1.intro.intro
      𝕜 : Type u_1
      V : Type u_2
      P : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : CompleteSpace 𝕜
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : FiniteDimensional 𝕜 V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      x : Subtype fun x => Membership.mem (affineSpan 𝕜 s) x
      h : ∀ (o : Set (Subtype fun x => Membership.mem (affineSpan 𝕜 s) x)), IsOpen o …
      t : Set P
      ht : IsOpen t
      hx : Membership.mem t ↑x
      ⊢ (Inter.inter t s).Nonempty
    -/
    obtain ⟨z, hz₁, hz₂⟩ := h _ (continuous_induced_dom.isOpen_preimage t ht) hx
    /-
      case h.refine_1.intro.intro.intro.intro
      𝕜 : Type u_1
      V : Type u_2
      P : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : CompleteSpace 𝕜
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : FiniteDimensional 𝕜 V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      x : Subtype fun x => Membership.mem (affineSpan 𝕜 s) x
      h : ∀ (o : Set (Subtype fun x => Membership.mem (affineSpan 𝕜 s) x)), IsOpen o …
      t : Set P
      ht : IsOpen t
      hx : Membership.mem t ↑x
      z : Subtype fun x => Membership.mem (affineSpan 𝕜 s) x
      hz₁ : Membership.mem (Set.preimage Subtype.val t) z
      hz₂ : Membership.mem (Set.preimage Subtype.val s) z
      ⊢ (Inter.inter t s).Nonempty
    -/
    exact ⟨z, hz₁, hz₂⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      𝕜 : Type u_1
      V : Type u_2
      P : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : CompleteSpace 𝕜
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : FiniteDimensional 𝕜 V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      x : P
      h : ∀ (o : Set P), IsOpen o → Membership.mem o x → (Inter.inter o s).Nonempty
      ⊢ ∀ (o : Set (Subtype fun x => Membership.mem (affineSpan 𝕜 s) x)), IsOpen o → …
    -/
  · rintro _ ⟨t, ht, rfl⟩ hx
    /-
      case h.refine_2.intro.intro
      𝕜 : Type u_1
      V : Type u_2
      P : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : CompleteSpace 𝕜
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : FiniteDimensional 𝕜 V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      x : P
      h : ∀ (o : Set P), IsOpen o → Membership.mem o x → (Inter.inter o s).Nonempty
      t : Set P
      ht : IsOpen t
      hx : Membership.mem (Set.preimage Subtype.val t) ⟨x, ?h.refine_3⟩
      ⊢ (Inter.inter (Set.preimage Subtype.val t) (Set.preimage Subtype.val s)).None …
    -/
    obtain ⟨y, hyt, hys⟩ := h _ ht hx
    /-
      case h.refine_2.intro.intro.intro.intro
      𝕜 : Type u_1
      V : Type u_2
      P : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : CompleteSpace 𝕜
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : FiniteDimensional 𝕜 V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      x : P
      h : ∀ (o : Set P), IsOpen o → Membership.mem o x → (Inter.inter o s).Nonempty
      t : Set P
      ht : IsOpen t
      hx : Membership.mem (Set.preimage Subtype.val t) ⟨x, ?h.refine_3⟩
      y : P
      hyt : Membership.mem t y
      hys : Membership.mem s y
      ⊢ (Inter.inter (Set.preimage Subtype.val t) (Set.preimage Subtype.val s)).None …
    -/
    exact ⟨⟨_, subset_affineSpan 𝕜 s hys⟩, hyt, hys⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_3
      𝕜 : Type u_1
      V : Type u_2
      P : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : CompleteSpace 𝕜
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : FiniteDimensional 𝕜 V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      x : P
      h : ∀ (o : Set P), IsOpen o → Membership.mem o x → (Inter.inter o s).Nonempty
      ⊢ Membership.mem (affineSpan 𝕜 s) x
    -/
  · by_contra hc
    /-
      case h.refine_3
      𝕜 : Type u_1
      V : Type u_2
      P : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : CompleteSpace 𝕜
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : FiniteDimensional 𝕜 V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      x : P
      h : ∀ (o : Set P), IsOpen o → Membership.mem o x → (Inter.inter o s).Nonempty
      hc : Not (Membership.mem (affineSpan 𝕜 s) x)
      ⊢ False
    -/
    obtain ⟨z, hz₁, hz₂⟩ := h _ (affineSpan 𝕜 s).closed_of_finiteDimensional.isOpen_compl hc
    /-
      case h.refine_3.intro.intro
      𝕜 : Type u_1
      V : Type u_2
      P : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : CompleteSpace 𝕜
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : FiniteDimensional 𝕜 V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      x : P
      h : ∀ (o : Set P), IsOpen o → Membership.mem o x → (Inter.inter o s).Nonempty
      hc : Not (Membership.mem (affineSpan 𝕜 s) x)
      z : P
      hz₁ : Membership.mem (HasCompl.compl ↑(affineSpan 𝕜 s)) z
      hz₂ : Membership.mem s z
      ⊢ False
    -/
    exact hz₁ (subset_affineSpan 𝕜 s hz₂)
    /-
      🎉 no goals
    -/


@[simp]
theorem closure_diff_intrinsicInterior (s : Set P) :
    closure s \ intrinsicInterior 𝕜 s = intrinsicFrontier 𝕜 s :=
  intrinsicClosure_eq_closure 𝕜 s ▸ intrinsicClosure_diff_intrinsicInterior s


@[simp]
theorem closure_diff_intrinsicFrontier (s : Set P) :
    closure s \ intrinsicFrontier 𝕜 s = intrinsicInterior 𝕜 s :=
  intrinsicClosure_eq_closure 𝕜 s ▸ intrinsicClosure_diff_intrinsicFrontier s


private theorem aux {α β : Type*} [TopologicalSpace α] [TopologicalSpace β] (φ : α ≃ₜ β)
    (s : Set β) : (interior s).Nonempty ↔ (interior (φ ⁻¹' s)).Nonempty := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    φ : Homeomorph α β
    s : Set β
    ⊢ Iff (interior s).Nonempty (interior (Set.preimage (⇑φ) s)).Nonempty
  -/
  rw [← φ.image_symm, ← φ.symm.image_interior, image_nonempty]
  /-
    🎉 no goals
  -/


/-- The intrinsic interior of a nonempty convex set is nonempty. -/
protected theorem Set.Nonempty.intrinsicInterior (hscv : Convex ℝ s) (hsne : s.Nonempty) :
    (intrinsicInterior ℝ s).Nonempty := by
  /-
    V : Type u_2
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    s : Set V
    hscv : Convex Real s
    hsne : s.Nonempty
    ⊢ (intrinsicInterior Real s).Nonempty
  -/
  haveI := hsne.coe_sort
  /-
    V : Type u_2
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    s : Set V
    hscv : Convex Real s
    hsne : s.Nonempty
    this : Nonempty ↑s
    ⊢ (intrinsicInterior Real s).Nonempty
  -/
  obtain ⟨p, hp⟩ := hsne
  /-
    case intro
    V : Type u_2
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    s : Set V
    hscv : Convex Real s
    this : Nonempty ↑s
    p : V
    hp : Membership.mem s p
    ⊢ (intrinsicInterior Real s).Nonempty
  -/
  let p' : _root_.affineSpan ℝ s := ⟨p, subset_affineSpan _ _ hp⟩
  rw [intrinsicInterior, image_nonempty,
    aux (AffineIsometryEquiv.constVSub ℝ p').symm.toHomeomorph,
    Convex.interior_nonempty_iff_affineSpan_eq_top, AffineIsometryEquiv.coe_toHomeomorph, ←
    AffineIsometryEquiv.coe_toAffineEquiv, ← comap_span, affineSpan_coe_preimage_eq_top, comap_top]
  exact hscv.affine_preimage
    ((_root_.affineSpan ℝ s).subtype.comp
      (AffineIsometryEquiv.constVSub ℝ p').symm.toAffineEquiv.toAffineMap)


theorem intrinsicInterior_nonempty (hs : Convex ℝ s) :
    (intrinsicInterior ℝ s).Nonempty ↔ s.Nonempty :=
      /-
        V : Type u_2
        inst✝² : NormedAddCommGroup V
        inst✝¹ : NormedSpace Real V
        inst✝ : FiniteDimensional Real V
        s : Set V
        hs : Convex Real s
        ⊢ (intrinsicInterior Real s).Nonempty → s.Nonempty
      -/
  ⟨by simp_rw [nonempty_iff_ne_empty]; rintro h rfl; exact h intrinsicInterior_empty,
                                                     /-
                                                       🎉 no goals
                                                     -/
    Set.Nonempty.intrinsicInterior hs⟩

